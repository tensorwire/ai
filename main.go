// ai — GPU compute CLI. Train, infer, quantize, serve.
//
// Usage:
//   ai train data=file.txt                        Train from scratch
//   ai train model=TinyLlama data=file.txt        Fine-tune a pretrained model
//   ai infer model "prompt"                       Generate text
//   ai pull org/model                             Download from HuggingFace
//   ai quantize model [q8|q4]                     Quantize model weights
//   ai serve model                                OpenAI-compatible API server

package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	ParseGlobalFlags()

	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	cmd := os.Args[1]
	args := parseKV(os.Args[2:])

	switch cmd {
	case "train":
		cmdTrainUnified(args)
	case "eval":
		cmdEval(args)
	case "profile":
		cmdProfile(args)
	case "dataset":
		cmdDataset(args)
	case "checkpoint":
		cmdCheckpoint(args)
	case "pull":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: ai pull <org/model>")
			os.Exit(1)
		}
		cmdPull(os.Args[2])
	case "models":
		cmdModels()
	case "info":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: ai info <model>")
			os.Exit(1)
		}
		cmdInfo(os.Args[2])
	case "infer":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "Usage: ai infer <model> \"prompt\"")
			os.Exit(1)
		}
		cmdInferGPU(os.Args[2], os.Args[3:])
	case "quantize":
		cmdQuantize()
	case "serve":
		cmdServe()
	case "bench":
		cmdBench()
	case "gpus":
		cmdGPUs()
	case "merge":
		cmdMerge()
	case "convert":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "Usage: ai convert gguf <model-dir> [output.gguf]")
			os.Exit(1)
		}
		cmdConvert(os.Args[2], os.Args[3:])
	case "benchmark":
		cmdBenchmark()
	case "export":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: ai export qdrant <url> <collection> [output.jsonl]")
			os.Exit(1)
		}
		cmdExport(os.Args[2:])
	// Hidden aliases for backwards compat
	case "train-cuda":
		cmdTrainCUDA()
	case "train-any":
		cmdTrainAny()
	case "finetune":
		cmdFinetune()

	case "-h", "--help", "help":
		usage()
	case "-v", "--version", "version":
		fmt.Println("ai v0.1.0 — powered by mongoose")
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		usage()
		os.Exit(1)
	}
}

// parseKV extracts key=value pairs and positional args from command-line arguments.
func parseKV(raw []string) map[string]string {
	m := make(map[string]string)
	pos := 0
	for _, arg := range raw {
		if k, v, ok := strings.Cut(arg, "="); ok {
			m[k] = v
		} else {
			m[fmt.Sprintf("_%d", pos)] = arg
			pos++
		}
	}
	return m
}

func usage() {
	fmt.Println("ai — GPU-accelerated ML. Zero Python, one binary.")
	fmt.Println()
	fmt.Println("Training:")
	fmt.Println("  ai train data=<file>                       Train from scratch")
	fmt.Println("  ai train model=<name> data=<file>          Fine-tune pretrained model")
	fmt.Println("  ai eval model=<name> data=<file>           Validation pass (loss + perplexity)")
	fmt.Println()
	fmt.Println("Inference:")
	fmt.Println("  ai infer <model> \"prompt\"                  Generate text")
	fmt.Println("  ai serve <model>                           OpenAI-compatible API")
	fmt.Println()
	fmt.Println("Models:")
	fmt.Println("  ai pull <org/model>                        Download from HuggingFace")
	fmt.Println("  ai models                                  List downloaded models")
	fmt.Println("  ai info <model>                            Show model architecture")
	fmt.Println()
	fmt.Println("Optimization:")
	fmt.Println("  ai quantize <model> [q8|q4|f16]            Quantize model weights")
	fmt.Println("  ai convert gguf <model>                    Convert to GGUF (for Ollama)")
	fmt.Println("  ai merge <base> <adapters>                 Merge LoRA into base model")
	fmt.Println()
	fmt.Println("Analysis:")
	fmt.Println("  ai profile [dim=N]                         Per-op GPU timing breakdown")
	fmt.Println("  ai dataset inspect <file>                  Dataset statistics")
	fmt.Println("  ai checkpoint ls [dir]                     List training checkpoints")
	fmt.Println("  ai checkpoint diff <a> <b>                 Compare two checkpoints")
	fmt.Println("  ai bench                                   Raw GPU benchmark")
	fmt.Println("  ai gpus                                    Detect and calibrate GPUs")
	fmt.Println("  ai benchmark <model>                       Profile model inference")
	fmt.Println()
	fmt.Println("Train overrides:  steps=N  lr=1e-4  dim=512  layers=8  seq=128  log=100")
	fmt.Println("Global flags:     --device cuda  --verbose")
}
