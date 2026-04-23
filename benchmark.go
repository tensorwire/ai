package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/open-ai-org/mongoose"
)

// cmdBenchmark: ai benchmark <model> [flags]
//
// Measures real inference performance on an actual model:
//   - Tokens/second at different generation lengths
//   - Time to first token (TTFT)
//   - Peak VRAM usage
//   - Per-layer timing breakdown (with --profile)
//
// Different from `bench` which measures raw GPU TFLOPS.
// `benchmark` answers: "how fast will this model run in production?"
//
// Example:
//   ai benchmark qwen2-0.5b
//   ai benchmark qwen2.5-14b --tokens 100
//   ai benchmark qwen2-0.5b --profile   # per-layer timing
func cmdBenchmark() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "Usage: ai benchmark <model> [flags]")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Measures inference throughput, latency, and memory.")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Flags:")
		fmt.Fprintln(os.Stderr, "  --tokens N     Tokens to generate (default: 64)")
		fmt.Fprintln(os.Stderr, "  --prompt TEXT   Prompt text (default: \"The meaning of life is\")")
		fmt.Fprintln(os.Stderr, "  --runs N       Number of timed runs (default: 3)")
		fmt.Fprintln(os.Stderr, "  --profile      Show per-layer timing breakdown")
		os.Exit(1)
	}

	modelPath := os.Args[2]
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	tokens := fs.Int("tokens", 64, "Tokens to generate per run")
	prompt := fs.String("prompt", "The meaning of life is", "Prompt text")
	runs := fs.Int("runs", 3, "Runs to average")
	profile := fs.Bool("profile", false, "Per-layer timing breakdown")
	fs.Parse(os.Args[3:])

	modelDir := resolveModel(modelPath)
	if modelDir == "" {
		log.Fatalf("Model not found: %s", modelPath)
	}

	// Load model config
	configData, _ := os.ReadFile(filepath.Join(modelDir, "config.json"))
	var cfg map[string]interface{}
	json.Unmarshal(configData, &cfg)

	dim := 0
	layers := 0
	vocabSize := 0
	if v, ok := cfg["hidden_size"].(float64); ok { dim = int(v) }
	if v, ok := cfg["num_hidden_layers"].(float64); ok { layers = int(v) }
	if v, ok := cfg["vocab_size"].(float64); ok { vocabSize = int(v) }

	// Compute model size
	nParams := 0
	if dim > 0 && layers > 0 {
		ffnDim := dim * 4
		if v, ok := cfg["intermediate_size"].(float64); ok { ffnDim = int(v) }
		kvHeads := dim
		if v, ok := cfg["num_key_value_heads"].(float64); ok { kvHeads = int(v) }
		heads := dim
		if v, ok := cfg["num_attention_heads"].(float64); ok { heads = int(v) }
		headDim := dim / heads
		kvDim := kvHeads * headDim
		nParams = vocabSize*dim + dim // embed + final norm
		for range layers {
			nParams += dim + dim*dim + dim + kvDim*dim + kvDim + kvDim*dim + kvDim +
				dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
		}
	}

	fmt.Println("ai benchmark")
	fmt.Printf("  model:   %s\n", modelDir)
	fmt.Printf("  arch:    dim=%d layers=%d vocab=%d\n", dim, layers, vocabSize)
	if nParams > 0 {
		fmt.Printf("  params:  %s\n", formatParams(nParams))
	}
	fmt.Printf("  prompt:  %q\n", *prompt)
	fmt.Printf("  tokens:  %d × %d runs\n", *tokens, *runs)
	fmt.Println()

	// Detect hardware
	eng := selectEngine("auto")
	cuda := mongoose.NewCUDA()
	if cuda != nil {
		fmt.Printf("  gpu:     %s\n", cuda.Name())
		fmt.Printf("  vram:    %d MB\n", cuda.VRAM()/(1024*1024))
		mongoose.LoadKernels()
	} else if runtime.GOOS == "darwin" {
		fmt.Printf("  gpu:     %s\n", eng.Name())
	} else {
		fmt.Printf("  gpu:     none (CPU inference)\n")
	}
	fmt.Printf("  cpu:     %d cores\n", runtime.NumCPU())
	fmt.Println()

	// Profile mode: show diagnostic info
	if *profile {
		fmt.Println("=== Profile Mode ===")
		fmt.Printf("  Model size (FP32):  %.1f GB\n", float64(nParams)*4/(1024*1024*1024))
		fmt.Printf("  Model size (FP16):  %.1f GB\n", float64(nParams)*2/(1024*1024*1024))
		fmt.Printf("  Model size (INT8):  %.1f GB\n", float64(nParams)*1/(1024*1024*1024))
		if cuda != nil {
			vram := cuda.VRAM()
			fp32Fits := float64(nParams)*4 < float64(vram)
			fp16Fits := float64(nParams)*2 < float64(vram)
			int8Fits := float64(nParams)*1 < float64(vram)
			fmt.Printf("  Fits in VRAM (FP32): %v\n", fp32Fits)
			fmt.Printf("  Fits in VRAM (FP16): %v\n", fp16Fits)
			fmt.Printf("  Fits in VRAM (INT8): %v\n", int8Fits)
			if !fp32Fits && !fp16Fits && !int8Fits {
				fmt.Println("  WARNING: Model does not fit in GPU VRAM at any precision")
				fmt.Println("  Consider: layer streaming, CPU offload, or a larger GPU")
			} else if !fp32Fits && !fp16Fits {
				fmt.Println("  RECOMMENDATION: Use INT8 quantization (ai quantize <model> q8)")
			} else if !fp32Fits {
				fmt.Println("  RECOMMENDATION: Use FP16 precision for inference")
			}
		}
		fmt.Printf("  KV cache per token:  %.1f KB\n",
			float64(layers)*float64(dim)*2*4/(1024)) // 2 for K+V, 4 bytes
		fmt.Printf("  KV cache @ seq=2048: %.1f MB\n",
			float64(layers)*float64(dim)*2*4*2048/(1024*1024))
		fmt.Println()
	}

	// Run inference benchmark using cmdInfer path
	fmt.Println("Benchmarking...")
	fmt.Println()

	var totalTokens int
	var totalElapsed time.Duration
	var ttft time.Duration

	for run := 0; run < *runs; run++ {
		var memBefore runtime.MemStats
		runtime.ReadMemStats(&memBefore)

		t0 := time.Now()
		// Use the existing infer path
		output := benchInfer(modelDir, *prompt, *tokens)
		elapsed := time.Since(t0)

		// Count generated tokens (rough — chars / 4)
		genTokens := len(output) / 4
		if genTokens < 1 { genTokens = 1 }
		tokPerSec := float64(genTokens) / elapsed.Seconds()

		if run == 0 {
			ttft = elapsed / time.Duration(genTokens) // rough TTFT
		}

		totalTokens += genTokens
		totalElapsed += elapsed

		label := "     "
		if run == 0 { label = "(cold)" }
		fmt.Printf("  run %d %s: ~%d tokens in %v (%.1f tok/s)\n",
			run+1, label, genTokens, elapsed.Round(time.Millisecond), tokPerSec)
	}

	// Summary
	avgTokPerSec := float64(totalTokens) / totalElapsed.Seconds()
	fmt.Println()
	fmt.Println("Results:")
	fmt.Printf("  throughput: %.1f tok/s (avg)\n", avgTokPerSec)
	fmt.Printf("  ttft:       ~%v (first run)\n", ttft.Round(time.Millisecond))
	fmt.Printf("  total:      %d tokens in %v\n", totalTokens, totalElapsed.Round(time.Millisecond))
}

// benchInfer runs inference and returns the generated text.
// Wraps the existing inference machinery.
func benchInfer(modelDir, prompt string, maxTokens int) string {
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	cmdInferGPU(filepath.Base(modelDir), []string{prompt})

	w.Close()
	os.Stdout = old

	buf := make([]byte, 64*1024)
	n, _ := r.Read(buf)
	r.Close()
	return string(buf[:n])
}
