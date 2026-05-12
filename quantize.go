package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
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

// sq4EncodeTensor quantizes a single FP32 weight tensor to SQ4.
// Copied from chain's CompressWeightsSQ (protocol/weight_sq.go) with
// output adapted to the opensource file format (separate mag/sign streams).
//
// Logic is IDENTICAL to chain:
// - 8 equal-count percentile bands on sorted |weights|
// - boundaries[b] = max |weight| in band b
// - means[b] = mean |weight| in band b (reconstruction value)
// - Outliers: |weight| > p99.9 threshold (strict greater)
// - Band assignment: first b where |weight| <= boundaries[b]
func sq4EncodeTensor(data []float32, rows, cols int) (mag []byte, sign []byte, bands [8]float32, outlierIdx []uint32, outlierVal []float32) {
	n := rows * cols
	numBands := 8

	// Sort absolute values to find percentile boundaries
	abs := make([]float32, n)
	for i, v := range data {
		if v < 0 {
			abs[i] = -v
		} else {
			abs[i] = v
		}
	}
	sorted := make([]float32, n)
	copy(sorted, abs)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	// Compute band boundaries and means
	boundaries := make([]float32, numBands)
	for b := 0; b < numBands; b++ {
		lo := n * b / numBands
		hi := n * (b + 1) / numBands
		if hi > n {
			hi = n
		}
		boundaries[b] = sorted[hi-1]
		var sum float64
		for _, v := range sorted[lo:hi] {
			sum += float64(v)
		}
		bands[b] = float32(sum / float64(hi-lo))
	}

	// Outlier threshold: p99.9
	outlierPos := n * 999 / 1000
	if outlierPos >= n {
		outlierPos = n - 1
	}
	outlierThresh := sorted[outlierPos]

	// Encode weights — matching chain's CompressWeightsSQ exactly
	magBits := n * 3
	mag = make([]byte, (magBits+7)/8)
	sign = make([]byte, (n+7)/8)

	for i, v := range data {
		absV := abs[i]

		// Outlier check (strict greater, matching chain)
		if absV > outlierThresh {
			outlierIdx = append(outlierIdx, uint32(i))
			outlierVal = append(outlierVal, v)
		}

		// Find magnitude band
		band := numBands - 1
		for b := 0; b < numBands-1; b++ {
			if absV <= boundaries[b] {
				band = b
				break
			}
		}

		// 3-bit magnitude: pack into bitstream
		bitPos := i * 3
		byteIdx := bitPos / 8
		bitOff := uint(bitPos % 8)
		mag[byteIdx] |= byte(band&0x07) << bitOff
		if bitOff > 5 {
			mag[byteIdx+1] |= byte(band&0x07) >> (8 - bitOff)
		}

		// 1-bit sign: 0=positive, 1=negative
		if v < 0 {
			sign[i/8] |= 1 << uint(i%8)
		}
	}
	return
}

func cmdQuantizeSQ4(modelDir string, st *gguf.SafeTensors, totalParams, sourceBytes int64) {
	outputDir := filepath.Base(modelDir) + "-sq4"
	if len(os.Args) >= 5 {
		outputDir = os.Args[4]
	}
	os.MkdirAll(outputDir, 0755)

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

	magF, _ := os.Create(filepath.Join(outputDir, "sq4_magnitude.bin"))
	signF, _ := os.Create(filepath.Join(outputDir, "sq4_sign.bin"))
	bandsF, _ := os.Create(filepath.Join(outputDir, "sq4_bands.bin"))
	outlierIdxF, _ := os.Create(filepath.Join(outputDir, "sq4_outlier_idx.bin"))
	outlierValF, _ := os.Create(filepath.Join(outputDir, "sq4_outlier_val.bin"))
	defer magF.Close()
	defer signF.Close()
	defer bandsF.Close()
	defer outlierIdxF.Close()
	defer outlierValF.Close()

	type tensorMeta struct {
		Name         string `json:"name"`
		Rows         int    `json:"rows"`
		Cols         int    `json:"cols"`
		MagOffset    int    `json:"mag_offset"`
		MagBytes     int    `json:"mag_bytes"`
		SignOffset   int    `json:"sign_offset"`
		SignBytes    int    `json:"sign_bytes"`
		BandsOffset  int    `json:"bands_offset"`
		OutlierStart int    `json:"outlier_start"`
		OutlierCount int    `json:"outlier_count"`
	}
	var metas []tensorMeta

	magOff := 0
	signOff := 0
	bandsOff := 0
	outlierOff := 0
	var totalSQ4Bytes int64

	writeTensor := func(name string, data []float32, rows, cols int) {
		mag, sign, bands, oIdx, oVal := sq4EncodeTensor(data, rows, cols)

		meta := tensorMeta{
			Name:         name,
			Rows:         rows,
			Cols:         cols,
			MagOffset:    magOff,
			MagBytes:     len(mag),
			SignOffset:   signOff,
			SignBytes:    len(sign),
			BandsOffset:  bandsOff,
			OutlierStart: outlierOff,
			OutlierCount: len(oIdx),
		}
		metas = append(metas, meta)

		magF.Write(mag)
		magOff += len(mag)

		signF.Write(sign)
		signOff += len(sign)

		bandsBytes := make([]byte, 8*4)
		for i := 0; i < 8; i++ {
			binary.LittleEndian.PutUint32(bandsBytes[i*4:], math.Float32bits(bands[i]))
		}
		bandsF.Write(bandsBytes)
		bandsOff += 8

		idxBytes := make([]byte, len(oIdx)*4)
		for i, v := range oIdx {
			binary.LittleEndian.PutUint32(idxBytes[i*4:], v)
		}
		outlierIdxF.Write(idxBytes)

		valBytes := make([]byte, len(oVal)*4)
		for i, v := range oVal {
			binary.LittleEndian.PutUint32(valBytes[i*4:], math.Float32bits(v))
		}
		outlierValF.Write(valBytes)
		outlierOff += len(oIdx)

		totalSQ4Bytes += int64(len(mag) + len(sign) + 8*4 + len(oIdx)*8)
	}

	// SQ4-encode embed + lm_head
	for _, name := range []string{"model.embed_tokens.weight", "lm_head.weight"} {
		data, info, err := st.ReadTensorFloat32(name)
		if err != nil || data == nil {
			continue
		}
		writeTensor(name, data, info.Shape[0], info.Shape[1])
		fmt.Printf("  %s → SQ4 (%dx%d)\n", name, info.Shape[0], info.Shape[1])
	}

	// SQ4-encode all layer projections
	weightSuffixes := []string{
		"self_attn.q_proj.weight", "self_attn.k_proj.weight",
		"self_attn.v_proj.weight", "self_attn.o_proj.weight",
		"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
	}
	for l := 0; l < nLayers; l++ {
		prefix := fmt.Sprintf("model.layers.%d.", l)
		for _, suffix := range weightSuffixes {
			name := prefix + suffix
			data, info, err := st.ReadTensorFloat32(name)
			if err != nil || data == nil {
				continue
			}
			writeTensor(name, data, info.Shape[0], info.Shape[1])
		}
		fmt.Printf("\r  Layer %d/%d quantized to SQ4", l+1, nLayers)
	}
	fmt.Println()

	// Metadata
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

	// Norms + biases stay FP32
	fpTensors := map[string]gguf.SaveTensor{}
	for _, name := range []string{"model.norm.weight"} {
		data, info, err := st.ReadTensorFloat32(name)
		if err == nil && data != nil {
			fpTensors[name] = gguf.SaveTensor{Data: data, Shape: info.Shape}
		}
	}
	for l := 0; l < nLayers; l++ {
		prefix := fmt.Sprintf("model.layers.%d.", l)
		for _, n := range []string{
			"input_layernorm.weight", "post_attention_layernorm.weight",
			"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias",
			"self_attn.o_proj.bias",
			"mlp.gate_proj.bias", "mlp.up_proj.bias", "mlp.down_proj.bias",
		} {
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
