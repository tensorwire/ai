package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/tensorwire/gguf"
)

// cmdQuantize: ai quantize <model> [q8|q4|f16]
//
// Quantizes a safetensors model to GGUF format for deployment.
// Weight matrices get quantized, norms/biases/embeddings stay FP32.
//
// Example:
//   ai quantize qwen2.5-14b q8       → qwen2.5-14b-q8.gguf
//   ai quantize qwen2.5-14b q4       → qwen2.5-14b-q4.gguf
//   ai quantize ./my-model f16       → my-model-f16.gguf
func cmdQuantize() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "Usage: ai quantize <model> [q8|q4|sq4|f16|f32]")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  Converts HuggingFace safetensors → quantized model.")
		fmt.Fprintln(os.Stderr, "  Default: q8 (INT8, ~1 byte/param)")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Quantization types:")
		fmt.Fprintln(os.Stderr, "  q8    INT8 per-row absmax → GGUF (best quality, ~1x model size)")
		fmt.Fprintln(os.Stderr, "  q4    INT4 block quantization → GGUF (~0.5x model size)")
		fmt.Fprintln(os.Stderr, "  sq4   SQ4 synaptic quantization → directory (~0.5x, band calibrated)")
		fmt.Fprintln(os.Stderr, "  f16   FP16 → GGUF (lossless for BF16 models, ~1x)")
		fmt.Fprintln(os.Stderr, "  f32   FP32 → GGUF (no quantization, ~2x)")
		os.Exit(1)
	}

	modelPath := os.Args[2]
	quantType := "q8_0" // default
	if len(os.Args) >= 4 {
		switch strings.ToLower(os.Args[3]) {
		case "q8", "q8_0", "int8":
			quantType = "q8_0"
		case "q4", "q4_0", "int4":
			quantType = "q4_0"
		case "sq4":
			quantType = "sq4"
		case "f16", "fp16":
			quantType = "f16"
		case "f32", "fp32":
			quantType = "f32"
		default:
			log.Fatalf("Unknown quantization type: %s (use q8, q4, f16, f32)", os.Args[3])
		}
	}

	modelDir := resolveModel(modelPath)
	if modelDir == "" {
		log.Fatalf("Model not found: %s", modelPath)
	}

	baseName := filepath.Base(modelDir)
	outputFile := baseName + "-" + strings.Replace(quantType, "_", "", 1) + ".gguf"
	if quantType == "sq4" {
		outputFile = baseName + "-sq4" // directory, not file
	}
	if len(os.Args) >= 5 {
		outputFile = os.Args[4]
	}

	// Estimate source size
	ms, err := OpenModel(modelDir)
	if err != nil {
		log.Fatalf("open model: %v", err)
	}
	st := ms.ST()
	if st == nil {
		log.Fatalf("quantize currently requires SafeTensors format — convert with: ai convert safetensors %s", modelDir)
	}
	var totalParams int64
	var sourceBytes int64
	for _, name := range st.TensorNames {
		ti, _ := st.GetInfo(name)
		if ti != nil {
			totalParams += int64(ti.NumElements())
			sourceBytes += int64(ti.ByteSize())
		}
	}

	fmt.Printf("ai quantize\n")
	fmt.Printf("  model:  %s\n", modelDir)
	fmt.Printf("  params: %s\n", formatParams(int(totalParams)))
	fmt.Printf("  source: %.1f GB (%s)\n", float64(sourceBytes)/(1024*1024*1024), st.TensorNames[0])
	fmt.Printf("  quant:  %s\n", quantType)
	fmt.Printf("  output: %s\n", outputFile)
	fmt.Println()

	if quantType == "sq4" {
		cmdQuantizeSQ4(modelDir, st, totalParams, sourceBytes)
		return
	}

	fmt.Print("Quantizing... ")
	err = gguf.ConvertSafetensorsToGGUF(modelDir, outputFile, quantType)
	if err != nil {
		log.Fatalf("\nQuantize failed: %v", err)
	}

	info, _ := os.Stat(outputFile)
	if info != nil {
		outGB := float64(info.Size()) / (1024 * 1024 * 1024)
		srcGB := float64(sourceBytes) / (1024 * 1024 * 1024)
		ratio := float64(info.Size()) / float64(sourceBytes) * 100
		if outGB > 1 {
			fmt.Printf("done.\n\n  %.1f GB → %.1f GB (%.0f%%) → %s\n", srcGB, outGB, ratio, outputFile)
		} else {
			fmt.Printf("done.\n\n  %.1f GB → %.0f MB (%.0f%%) → %s\n", srcGB, float64(info.Size())/(1024*1024), ratio, outputFile)
		}
	}
}

func cmdQuantizeSQ4(modelDir string, st *gguf.SafeTensors, totalParams, sourceBytes int64) {
	outputDir := filepath.Base(modelDir) + "-sq4"
	if len(os.Args) >= 5 {
		outputDir = os.Args[4]
	}
	os.MkdirAll(outputDir, 0755)

	// Read config for layer count
	configData, _ := os.ReadFile(filepath.Join(modelDir, "config.json"))
	var cfg map[string]interface{}
	json.Unmarshal(configData, &cfg)
	nLayers := 0
	if v, ok := cfg["num_hidden_layers"].(float64); ok {
		nLayers = int(v)
	}
	dim := 0
	if v, ok := cfg["hidden_size"].(float64); ok {
		dim = int(v)
	}

	fmt.Printf("  output: %s\n\n", outputDir)

	// Weight tensors to quantize (per-layer projections)
	weightSuffixes := []string{
		"self_attn.q_proj.weight", "self_attn.k_proj.weight",
		"self_attn.v_proj.weight", "self_attn.o_proj.weight",
		"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
	}

	// Create output files
	packedF, _ := os.Create(filepath.Join(outputDir, "sq4_packed.bin"))
	bandsF, _ := os.Create(filepath.Join(outputDir, "sq4_bands.bin"))
	outlierIdxF, _ := os.Create(filepath.Join(outputDir, "sq4_outlier_idx.bin"))
	outlierValF, _ := os.Create(filepath.Join(outputDir, "sq4_outlier_val.bin"))
	defer packedF.Close()
	defer bandsF.Close()
	defer outlierIdxF.Close()
	defer outlierValF.Close()

	type tensorMeta struct {
		Name         string `json:"name"`
		Rows         int    `json:"rows"`
		Cols         int    `json:"cols"`
		PackedOffset int    `json:"packed_offset"`
		PackedBytes  int    `json:"packed_bytes"`
		BandsOffset  int    `json:"bands_offset"`
		BandsFloats  int    `json:"bands_floats"`
		OutlierStart int    `json:"outlier_start"`
		OutlierCount int    `json:"outlier_count"`
	}
	var metas []tensorMeta

	packedOff := 0
	bandsOff := 0
	outlierOff := 0
	var totalSQ4Bytes int64

	for l := 0; l < nLayers; l++ {
		prefix := fmt.Sprintf("model.layers.%d.", l)
		for _, suffix := range weightSuffixes {
			name := prefix + suffix
			data, info, err := st.ReadTensorFloat32(name)
			if err != nil || data == nil {
				continue
			}

			rows := info.Shape[0]
			cols := info.Shape[1]
			sq4 := gguf.QuantizeToSQ4(data, rows, cols)

			meta := tensorMeta{
				Name:         name,
				Rows:         rows,
				Cols:         cols,
				PackedOffset: packedOff,
				PackedBytes:  len(sq4.Packed),
				BandsOffset:  bandsOff,
				BandsFloats:  len(sq4.Bands),
				OutlierStart: outlierOff,
				OutlierCount: len(sq4.OutlierIdx),
			}
			metas = append(metas, meta)

			packedF.Write(sq4.Packed)
			packedOff += len(sq4.Packed)

			bandsBytes := make([]byte, len(sq4.Bands)*4)
			for i, v := range sq4.Bands {
				binary.LittleEndian.PutUint32(bandsBytes[i*4:], math.Float32bits(v))
			}
			bandsF.Write(bandsBytes)
			bandsOff += len(sq4.Bands)

			idxBytes := make([]byte, len(sq4.OutlierIdx)*4)
			for i, v := range sq4.OutlierIdx {
				binary.LittleEndian.PutUint32(idxBytes[i*4:], v)
			}
			outlierIdxF.Write(idxBytes)

			valBytes := make([]byte, len(sq4.OutlierVal)*4)
			for i, v := range sq4.OutlierVal {
				binary.LittleEndian.PutUint32(valBytes[i*4:], math.Float32bits(v))
			}
			outlierValF.Write(valBytes)
			outlierOff += len(sq4.OutlierIdx)

			totalSQ4Bytes += int64(len(sq4.Packed) + len(sq4.Bands)*4 + len(sq4.OutlierIdx)*8)
		}
		fmt.Printf("\r  Layer %d/%d quantized to SQ4", l+1, nLayers)
	}
	fmt.Println()

	// Save metadata
	metaJSON := struct {
		Format  string       `json:"format"`
		Model   string       `json:"model"`
		Dim     int          `json:"dim"`
		Layers  int          `json:"layers"`
		Tensors []tensorMeta `json:"tensors"`
	}{
		Format:  "sq4",
		Model:   filepath.Base(modelDir),
		Dim:     dim,
		Layers:  nLayers,
		Tensors: metas,
	}
	metaBytes, _ := json.MarshalIndent(metaJSON, "", "  ")
	os.WriteFile(filepath.Join(outputDir, "sq4_meta.json"), metaBytes, 0644)

	// Copy config + tokenizer
	for _, f := range []string{"config.json", "tokenizer.json", "tokenizer_config.json",
		"tokenizer.model", "special_tokens_map.json", "generation_config.json"} {
		src := filepath.Join(modelDir, f)
		if data, err := os.ReadFile(src); err == nil {
			os.WriteFile(filepath.Join(outputDir, f), data, 0644)
		}
	}

	// Save embeddings + norms as SafeTensors (these stay FP32)
	fpTensors := map[string]gguf.SaveTensor{}
	for _, name := range []string{"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"} {
		data, info, err := st.ReadTensorFloat32(name)
		if err == nil && data != nil {
			fpTensors[name] = gguf.SaveTensor{Data: data, Shape: info.Shape}
		}
	}
	for l := 0; l < nLayers; l++ {
		prefix := fmt.Sprintf("model.layers.%d.", l)
		for _, n := range []string{"input_layernorm.weight", "post_attention_layernorm.weight"} {
			data, info, err := st.ReadTensorFloat32(prefix + n)
			if err == nil && data != nil {
				fpTensors[prefix+n] = gguf.SaveTensor{Data: data, Shape: info.Shape}
			}
		}
	}
	gguf.SaveSafeTensors(filepath.Join(outputDir, "fp32.safetensors"), fpTensors)

	srcGB := float64(sourceBytes) / (1024 * 1024 * 1024)
	outGB := float64(totalSQ4Bytes) / (1024 * 1024 * 1024)
	fmt.Printf("\n  SQ4: %.1f GB → %.1f GB (%.1fx compression)\n", srcGB, outGB, srcGB/outGB)
	fmt.Printf("  %d tensors quantized, %d outliers total\n", len(metas), outlierOff)
	fmt.Printf("  Output: %s\n", outputDir)
}
