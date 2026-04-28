package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"

	"github.com/tensorwire/gguf"
)

// cmdMerge: mongoose merge <base-model> <adapters> [output-dir]
//
// Merges LoRA adapters back into the base model weights.
// output = base_weight + A @ B (for each adapted layer)
//
// This produces a full-size model that can be:
//   - Quantized: mongoose quantize merged-model q8
//   - Served:    mongoose serve merged-model
//   - Inferred:  mongoose infer merged-model "prompt"
//
// Example:
//   mongoose merge qwen2.5-14b ./finetuned/adapters.safetensors ./merged-model
func cmdMerge() {
	if len(os.Args) < 4 {
		fmt.Fprintln(os.Stderr, "Usage: mongoose merge <base-model> <adapters> [output-dir]")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  Merges LoRA adapters into base model weights.")
		fmt.Fprintln(os.Stderr, "  Result is a full model ready for quantize/serve/infer.")
		os.Exit(1)
	}

	basePath := os.Args[2]
	adapterPath := os.Args[3]
	outputDir := ""
	if len(os.Args) >= 5 {
		outputDir = os.Args[4]
	}

	baseDir := resolveModel(basePath)
	if baseDir == "" {
		log.Fatalf("Base model not found: %s", basePath)
	}

	if outputDir == "" {
		outputDir = filepath.Base(baseDir) + "-merged"
	}

	// Load LoRA config
	loraConfigPath := filepath.Join(filepath.Dir(adapterPath), "lora_config.json")
	var loraRank int
	var targetModules []string
	if data, err := os.ReadFile(loraConfigPath); err == nil {
		var cfg map[string]interface{}
		json.Unmarshal(data, &cfg)
		if r, ok := cfg["lora_rank"].(float64); ok {
			loraRank = int(r)
		}
		if mods, ok := cfg["target_modules"].([]interface{}); ok {
			for _, m := range mods {
				targetModules = append(targetModules, m.(string))
			}
		}
	}
	if loraRank == 0 {
		loraRank = 16
		log.Printf("No lora_config.json found, assuming rank=%d", loraRank)
	}

	fmt.Println("mongoose merge")
	fmt.Printf("  base:     %s\n", baseDir)
	fmt.Printf("  adapters: %s\n", adapterPath)
	fmt.Printf("  rank:     %d\n", loraRank)
	fmt.Printf("  targets:  %v\n", targetModules)
	fmt.Printf("  output:   %s\n", outputDir)
	fmt.Println()

	// Open base model
	baseST, err := gguf.OpenSafeTensors(baseDir)
	if err != nil {
		log.Fatalf("Open base model: %v", err)
	}

	// Open adapters
	adapterST, err := gguf.OpenSafeTensors(adapterPath)
	if err != nil {
		log.Fatalf("Open adapters: %v", err)
	}

	// Load model config for layer count
	configData, _ := os.ReadFile(filepath.Join(baseDir, "config.json"))
	var mcfg map[string]interface{}
	json.Unmarshal(configData, &mcfg)
	nLayers := 0
	if v, ok := mcfg["num_hidden_layers"].(float64); ok {
		nLayers = int(v)
	}

	// Merge: for each adapted weight, compute W_merged = W_base + A @ B
	mergedTensors := make(map[string]gguf.SaveTensor)
	mergedCount := 0

	for _, name := range baseST.TensorNames {
		baseData, _, err := baseST.ReadTensorFloat32(name)
		if err != nil {
			log.Printf("WARN: skip %s: %v", name, err)
			continue
		}

		// Check if this tensor has LoRA adapters
		loraAName := name
		// Convert weight name to adapter name:
		// "model.layers.0.self_attn.q_proj.weight" → "model.layers.0.self_attn.q_proj.lora_A"
		if isAdaptedWeight(name, targetModules) {
			baseName := name[:len(name)-len(".weight")]
			aName := baseName + ".lora_A"
			bName := baseName + ".lora_B"

			if adapterST.HasTensor(aName) && adapterST.HasTensor(bName) {
				aData, aInfo, _ := adapterST.ReadTensorFloat32(aName)
				bData, bInfo, _ := adapterST.ReadTensorFloat32(bName)

				if aData != nil && bData != nil {
					// A is [rank, inDim], B is [outDim, rank]
					// LoRA output = B @ A (added to weight W[outDim, inDim])
					rank := aInfo.Shape[0]
					inDim := aInfo.Shape[1]
					outDim := bInfo.Shape[0]

					// W_merged = W_base + B @ A
					for i := 0; i < outDim; i++ {
						for j := 0; j < inDim; j++ {
							var sum float64
							for r := 0; r < rank; r++ {
								sum += float64(bData[i*rank+r]) * float64(aData[r*inDim+j])
							}
							baseData[i*inDim+j] += float32(sum)
						}
					}
					mergedCount++
					_ = loraAName
				}
			}
		}

		ti, _ := baseST.GetInfo(name)
		mergedTensors[name] = gguf.SaveTensor{
			Data:  baseData,
			Shape: ti.Shape,
		}
	}

	fmt.Printf("Merged %d/%d adapted layers\n", mergedCount, nLayers*len(targetModules))

	// Save merged model
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("mkdir: %v", err)
	}

	fmt.Print("Saving merged model... ")
	outputST := filepath.Join(outputDir, "model.safetensors")
	if err := gguf.SaveSafeTensors(outputST, mergedTensors); err != nil {
		log.Fatalf("save: %v", err)
	}

	// Copy config files
	for _, f := range []string{"config.json", "tokenizer.json", "tokenizer_config.json",
		"generation_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"} {
		src := filepath.Join(baseDir, f)
		if data, err := os.ReadFile(src); err == nil {
			os.WriteFile(filepath.Join(outputDir, f), data, 0644)
		}
	}

	info, _ := os.Stat(outputST)
	if info != nil {
		fmt.Printf("done. %.1f GB → %s\n", float64(info.Size())/(1024*1024*1024), outputDir)
	}
	_ = math.Sqrt // ensure math import used
}

// isAdaptedWeight checks if a tensor name corresponds to a LoRA-adapted weight.
func isAdaptedWeight(name string, targets []string) bool {
	if len(targets) == 0 {
		// Default targets if no config
		targets = []string{"q_proj", "k_proj", "v_proj", "o_proj",
			"gate_proj", "up_proj", "down_proj"}
	}
	for _, t := range targets {
		if contains(name, t+".weight") {
			return true
		}
	}
	return false
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && searchString(s, sub)
}

func searchString(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
