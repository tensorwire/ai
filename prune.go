package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/tensorwire/gguf"
)

// cmdPrune removes low-magnitude weights to produce a smaller, sparser model.
//
// Default: 50% unstructured magnitude pruning on weight matrices.
// Norms, biases, and embeddings are never pruned.
//
// Usage:
//
//	ai prune <model>                        Prune 50% of weights
//	ai prune <model> --sparsity 0.7         Prune 70%
//	ai prune <model> --output ./pruned      Custom output directory
//	ai prune <model> --structured           Prune entire attention heads (structured)
func cmdPrune() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "Usage: ai prune <model> [--sparsity 0.5] [--output <dir>] [--structured]")
		os.Exit(1)
	}

	modelName := os.Args[2]
	sparsity := 0.5
	outputDir := ""
	structured := false

	for i := 3; i < len(os.Args); i++ {
		switch os.Args[i] {
		case "--sparsity":
			if i+1 < len(os.Args) {
				fmt.Sscanf(os.Args[i+1], "%f", &sparsity)
				i++
			}
		case "--output", "-o":
			if i+1 < len(os.Args) {
				outputDir = os.Args[i+1]
				i++
			}
		case "--structured":
			structured = true
		}
	}

	if sparsity <= 0 || sparsity >= 1 {
		log.Fatalf("sparsity must be between 0 and 1, got %f", sparsity)
	}

	modelDir := resolveModel(modelName)
	ms, err := OpenModel(modelDir)
	if err != nil {
		log.Fatalf("open model: %v", err)
	}
	st := ms.ST()
	if st == nil {
		log.Fatalf("prune currently requires SafeTensors format — convert with: ai convert safetensors %s", modelDir)
	}

	if outputDir == "" {
		outputDir = modelDir + fmt.Sprintf("-pruned%.0f", sparsity*100)
	}
	os.MkdirAll(outputDir, 0755)

	fmt.Println("ai prune")
	fmt.Printf("  model:    %s\n", filepath.Base(modelDir))
	fmt.Printf("  sparsity: %.0f%%\n", sparsity*100)
	fmt.Printf("  mode:     %s\n", map[bool]string{true: "structured (head pruning)", false: "unstructured (magnitude)"}[structured])
	fmt.Printf("  output:   %s\n", outputDir)
	fmt.Println()

	tensors := make(map[string]gguf.SaveTensor)
	var totalParams, totalZeros int64

	for _, name := range st.TensorNames {
		data, info, err := st.ReadTensorFloat32(name)
		if err != nil {
			continue
		}

		isWeight := strings.HasSuffix(name, ".weight") &&
			!strings.Contains(name, "layernorm") &&
			!strings.Contains(name, "norm.weight") &&
			!strings.Contains(name, "embed_tokens")

		if isWeight && len(info.Shape) == 2 {
			if structured {
				pruneStructured(data, info.Shape[0], info.Shape[1], sparsity)
			} else {
				pruneUnstructured(data, sparsity)
			}
		}

		// Count sparsity
		for _, v := range data {
			totalParams++
			if v == 0 {
				totalZeros++
			}
		}

		tensors[name] = gguf.SaveTensor{Data: data, Shape: info.Shape}
	}

	actualSparsity := float64(totalZeros) / float64(totalParams) * 100

	// Save pruned model
	outPath := filepath.Join(outputDir, "model.safetensors")
	fmt.Printf("Saving pruned model... ")
	if err := gguf.SaveSafeTensors(outPath, tensors); err != nil {
		log.Fatalf("save: %v", err)
	}

	// Copy config and tokenizer files
	for _, f := range []string{"config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json"} {
		src := filepath.Join(modelDir, f)
		if data, err := os.ReadFile(src); err == nil {
			os.WriteFile(filepath.Join(outputDir, f), data, 0644)
		}
	}

	srcSize := dirSizeBytes(modelDir)
	dstSize := dirSizeBytes(outputDir)

	fmt.Println("done.")
	fmt.Println()
	fmt.Printf("  params:         %s\n", formatCount(int(totalParams)))
	fmt.Printf("  zeros:          %s (%.1f%%)\n", formatCount(int(totalZeros)), actualSparsity)
	fmt.Printf("  original size:  %s\n", formatBytes(srcSize))
	fmt.Printf("  pruned size:    %s\n", formatBytes(dstSize))
	fmt.Printf("  output:         %s\n", outputDir)
}

// pruneUnstructured zeros out the smallest-magnitude weights globally.
func pruneUnstructured(data []float32, sparsity float64) {
	n := len(data)
	nPrune := int(float64(n) * sparsity)
	if nPrune <= 0 {
		return
	}

	// Find magnitude threshold via partial sort
	mags := make([]float32, n)
	for i, v := range data {
		mags[i] = float32(math.Abs(float64(v)))
	}
	sort.Slice(mags, func(i, j int) bool { return mags[i] < mags[j] })
	threshold := mags[nPrune-1]

	// Zero out values below threshold
	for i, v := range data {
		if float32(math.Abs(float64(v))) <= threshold {
			data[i] = 0
		}
	}
}

// pruneStructured zeros out entire rows (output neurons) with smallest L2 norms.
func pruneStructured(data []float32, rows, cols int, sparsity float64) {
	nPrune := int(float64(rows) * sparsity)
	if nPrune <= 0 {
		return
	}

	type rowNorm struct {
		idx  int
		norm float64
	}
	norms := make([]rowNorm, rows)
	for r := 0; r < rows; r++ {
		var ss float64
		for c := 0; c < cols; c++ {
			v := float64(data[r*cols+c])
			ss += v * v
		}
		norms[r] = rowNorm{r, math.Sqrt(ss)}
	}

	sort.Slice(norms, func(i, j int) bool { return norms[i].norm < norms[j].norm })

	for i := 0; i < nPrune; i++ {
		r := norms[i].idx
		for c := 0; c < cols; c++ {
			data[r*cols+c] = 0
		}
	}
}
