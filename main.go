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
		cmdTrain()
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
	case "prune":
		cmdPrune()
	case "distill":
		cmdDistill(args)
	case "sweep":
		cmdSweep(args)
	case "explain":
		cmdExplain()
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
	case "chat":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: ai chat <model>")
			os.Exit(1)
		}
		cmdChat(os.Args[2])

	case "resume":
		cmdResume()
	case "finetune":
		cmdFinetune()

	// Hidden aliases for backwards compat
	case "train-cuda":
		cmdTrainCUDA()
	case "train-metal":
		cmdTrainMetal()
	case "train-any":
		cmdTrainAny()

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
	fmt.Println("usage: ai <command> [flags]")
	fmt.Println()
	fmt.Println("Training:")
	fmt.Println("  train                  Train a model from scratch")
	fmt.Println("  finetune               Fine-tune a pretrained model")
	fmt.Println("  resume                 Continue training from a checkpoint")
	fmt.Println()
	fmt.Println("Evaluation & Inference:")
	fmt.Println("  eval                   Validation pass (loss + perplexity)")
	fmt.Println("  infer <model> \"prompt\" Run inference on a model")
	fmt.Println("  chat <model>           Interactive chat")
	fmt.Println("  benchmark <model>      Measure inference throughput and latency")
	fmt.Println()
	fmt.Println("Optimization:")
	fmt.Println("  quantize <model>       Reduce precision (Q8, Q4, F16)")
	fmt.Println("  convert gguf <model>   Export to GGUF (for Ollama)")
	fmt.Println("  merge <base> <lora>    Merge LoRA adapters into base model")
	fmt.Println()
	fmt.Println("Data:")
	fmt.Println("  dataset inspect <file> Preview dataset statistics")
	fmt.Println()
	fmt.Println("Deployment:")
	fmt.Println("  serve <model>          OpenAI-compatible API server")
	fmt.Println()
	fmt.Println("Introspection:")
	fmt.Println("  profile                Per-op GPU timing breakdown")
	fmt.Println("  checkpoint ls [dir]    List saved checkpoints")
	fmt.Println("  checkpoint diff <a> <b>  Compare two checkpoints")
	fmt.Println("  bench                  Raw GPU compute benchmark")
	fmt.Println("  gpus                   Detect and calibrate hardware")
	fmt.Println()
	fmt.Println("Models:")
	fmt.Println("  pull <org/model>       Download from HuggingFace")
	fmt.Println("  models                 List downloaded models")
	fmt.Println("  info <model>           Show model architecture")
	fmt.Println()
	fmt.Println("Global flags:")
	fmt.Println("  --device <device>      Target: cpu, cuda, cuda:0, metal (default: auto)")
	fmt.Println("  --out <dir>            Output directory for checkpoints and exports")
	fmt.Println("  --dry-run              Validate config without executing")
	fmt.Println("  --verbose              Show detailed logs")
	fmt.Println()
	fmt.Println("Run ai <command> --help for per-command flags.")
}
