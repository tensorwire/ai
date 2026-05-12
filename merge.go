package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"

	"github.com/tensorwire/gguf"
)

// cmdMerge: ai merge <base-model> <adapters> [output-dir]
//
// Merges LoRA adapters back into the base model weights.
// Uses streaming writes — only one tensor in memory at a time.
func cmdMerge() {
	if len(os.Args) < 4 {
		fmt.Fprintln(os.Stderr, "Usage: ai merge <base-model> <adapters> [output-dir]")
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

	baseST, err := gguf.OpenSafeTensors(baseDir)
	if err != nil {
		log.Fatalf("Open base model: %v", err)
	}
	adapterST, err := gguf.OpenSafeTensors(adapterPath)
	if err != nil {
		log.Fatalf("Open adapters: %v", err)
	}

	configData, _ := os.ReadFile(filepath.Join(baseDir, "config.json"))
	var mcfg map[string]interface{}
	json.Unmarshal(configData, &mcfg)
	nLayers := 0
	if v, ok := mcfg["num_hidden_layers"].(float64); ok {
		nLayers = int(v)
	}

	fmt.Println("ai merge")
	fmt.Printf("  base:     %s\n", baseDir)
	fmt.Printf("  adapters: %s\n", adapterPath)
	fmt.Printf("  rank:     %d\n", loraRank)
	fmt.Printf("  targets:  %v\n", targetModules)
	fmt.Printf("  output:   %s\n", outputDir)
	fmt.Println()

	// Pass 1: collect tensor names and shapes for the streaming header
	names := make([]string, len(baseST.TensorNames))
	copy(names, baseST.TensorNames)
	sort.Strings(names)

	metas := make([]gguf.TensorMeta, 0, len(names))
	for _, name := range names {
		ti, err := baseST.GetInfo(name)
		if err != nil {
			continue
		}
		elems := 1
		for _, d := range ti.Shape {
			elems *= d
		}
		metas = append(metas, gguf.TensorMeta{Name: name, Shape: ti.Shape, Elems: elems})
	}

	if err := os.MkdirAll(outputDir, 0755); err != nil {
		log.Fatalf("mkdir: %v", err)
	}

	outputST := filepath.Join(outputDir, "model.safetensors")
	w, err := gguf.NewStreamingSafeTensorsWriter(outputST, metas)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}

	// Pass 2: read each tensor, merge if adapted, write immediately
	mergedCount := 0
	for i, m := range metas {
		baseData, _, err := baseST.ReadTensorFloat32(m.Name)
		if err != nil {
			log.Printf("WARN: skip %s: %v", m.Name, err)
			w.WriteTensor(make([]float32, m.Elems))
			continue
		}

		if isAdaptedWeight(m.Name, targetModules) {
			baseName := m.Name[:len(m.Name)-len(".weight")]
			aName := baseName + ".lora_A"
			bName := baseName + ".lora_B"

			if adapterST.HasTensor(aName) && adapterST.HasTensor(bName) {
				aData, aInfo, _ := adapterST.ReadTensorFloat32(aName)
				bData, bInfo, _ := adapterST.ReadTensorFloat32(bName)

				if aData != nil && bData != nil {
					rank := aInfo.Shape[0]
					inDim := aInfo.Shape[1]
					outDim := bInfo.Shape[0]

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
				}
			}
		}

		if err := w.WriteTensor(baseData); err != nil {
			log.Fatalf("write tensor %s: %v", m.Name, err)
		}

		if (i+1)%50 == 0 || i == len(metas)-1 {
			fmt.Printf("\r  %d/%d tensors written", i+1, len(metas))
		}
	}
	fmt.Println()
	w.Close()

	fmt.Printf("Merged %d/%d adapted layers\n", mergedCount, nLayers*len(targetModules))

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
}

func isAdaptedWeight(name string, targets []string) bool {
	if len(targets) == 0 {
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
