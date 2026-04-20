package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/open-ai-org/gguf"
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
		fmt.Fprintln(os.Stderr, "Usage: ai quantize <model> [q8|q4|f16|f32]")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "  Converts HuggingFace safetensors → GGUF with quantization.")
		fmt.Fprintln(os.Stderr, "  Default: q8 (INT8, ~1 byte/param)")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Quantization types:")
		fmt.Fprintln(os.Stderr, "  q8    INT8 per-row absmax (best quality, ~1x model size)")
		fmt.Fprintln(os.Stderr, "  q4    INT4 block quantization (~0.5x model size)")
		fmt.Fprintln(os.Stderr, "  f16   FP16 (lossless for BF16 models, ~1x)")
		fmt.Fprintln(os.Stderr, "  f32   FP32 (no quantization, ~2x)")
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
	if len(os.Args) >= 5 {
		outputFile = os.Args[4]
	}

	// Estimate source size
	st, err := gguf.OpenSafeTensors(modelDir)
	if err != nil {
		log.Fatalf("Open model: %v", err)
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
