// ai — GPU compute CLI. Train, infer, quantize, serve.
//
// Commands use key=value syntax (--flags accepted for backwards compat):
//
//	ai train data=corpus.txt                     Train from scratch
//	ai train model=TinyLlama data=corpus.txt     Fine-tune a pretrained model
//	ai infer Qwen2.5-0.5B "prompt"              Generate text
//	ai pull Qwen/Qwen2.5-0.5B                   Download from HuggingFace
//	ai quantize Qwen2.5-0.5B q8                 Quantize model weights
//	ai serve Qwen2.5-0.5B                       OpenAI-compatible API server

package main

import (
	"fmt"
	"os"
	"path/filepath"
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
	case "prune":
		cmdPrune()
	case "distill":
		cmdDistill(args)
	case "sweep":
		cmdSweep(args)
	case "explain":
		cmdExplain()
	case "serve":
		cmdServe(args)
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
		cmdResumeKV(args)

	// Hidden aliases for backwards compat
	case "finetune":
		cmdTrainUnified(args)
	case "train-cuda":
		cmdTrainCUDA()
	case "train-metal":
		cmdTrainMetal()
	case "train-any":
		cmdTrainAny()

	case "-h", "--help", "help":
		usage()
	case "-v", "--version", "version":
		fmt.Println("ai v1.4.1 — powered by mongoose")
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		usage()
		os.Exit(1)
	}
}

// parseKV extracts key=value pairs, --flag value pairs, and positional args.
// Values are expanded: ~/... becomes $HOME/..., glob patterns are resolved.
func parseKV(raw []string) map[string]string {
	m := make(map[string]string)
	pos := 0
	for i := 0; i < len(raw); i++ {
		arg := raw[i]
		if k, v, ok := strings.Cut(arg, "="); ok {
			m[k] = expandPath(v)
		} else if strings.HasPrefix(arg, "--") && i+1 < len(raw) && !strings.HasPrefix(raw[i+1], "--") {
			key := strings.TrimPrefix(arg, "--")
			i++
			m[key] = expandPath(raw[i])
		} else if strings.HasPrefix(arg, "--") {
			key := strings.TrimPrefix(arg, "--")
			m[key] = "true"
		} else {
			m[fmt.Sprintf("_%d", pos)] = arg
			pos++
		}
	}
	return m
}

// expandPath resolves ~ and globs in a value string.
func expandPath(v string) string {
	if strings.HasPrefix(v, "~/") || v == "~" {
		if home, err := os.UserHomeDir(); err == nil {
			v = filepath.Join(home, v[1:])
		}
	}
	if strings.ContainsAny(v, "*?[") {
		matches, err := filepath.Glob(v)
		if err == nil && len(matches) == 1 {
			return matches[0]
		}
		if err == nil && len(matches) > 1 {
			fmt.Fprintf(os.Stderr, "warning: glob matched %d files, using first: %s\n", len(matches), matches[0])
			return matches[0]
		}
	}
	return v
}

func usage() {
	fmt.Println("ai — GPU-accelerated ML. Zero Python, one binary.")
	fmt.Println()
	fmt.Println("usage: ai <command> [key=value ...]")
	fmt.Println()
	fmt.Println("Training:")
	fmt.Println("  train data=<file>                    Train from scratch")
	fmt.Println("  train model=<name> data=<file>       Fine-tune a pretrained model")
	fmt.Println("  resume checkpoint=<dir> data=<file>  Continue from checkpoint")
	fmt.Println()
	fmt.Println("Evaluation & Inference:")
	fmt.Println("  eval model=<name> data=<file>        Validation pass (loss + perplexity)")
	fmt.Println("  infer <model> \"prompt\"               Generate text")
	fmt.Println("  chat <model>                         Interactive chat")
	fmt.Println("  benchmark <model>                    Measure inference throughput")
	fmt.Println()
	fmt.Println("Optimization:")
	fmt.Println("  quantize <model> [q8|q4|f16]         Reduce precision")
	fmt.Println("  convert gguf <model>                 Export to GGUF (for Ollama)")
	fmt.Println("  merge <base> <lora>                  Merge LoRA adapters")
	fmt.Println()
	fmt.Println("Data:")
	fmt.Println("  dataset inspect <file>               Preview dataset statistics")
	fmt.Println("  dataset split <file>                 Partition into train/val/test")
	fmt.Println()
	fmt.Println("Tuning:")
	fmt.Println("  sweep data=<file> lr=1e-4,3e-4       Hyperparameter search")
	fmt.Println("  distill teacher=<model> data=<file>  Knowledge distillation")
	fmt.Println()
	fmt.Println("Deployment:")
	fmt.Println("  serve <model>                        OpenAI-compatible API server")
	fmt.Println()
	fmt.Println("Introspection:")
	fmt.Println("  profile                              Per-op GPU timing breakdown")
	fmt.Println("  checkpoint ls [dir]                  List saved checkpoints")
	fmt.Println("  checkpoint diff <a> <b>              Compare two checkpoints")
	fmt.Println("  bench                                Raw GPU compute benchmark")
	fmt.Println("  gpus                                 Detect and calibrate hardware")
	fmt.Println()
	fmt.Println("Models:")
	fmt.Println("  pull <org/model>                     Download from HuggingFace")
	fmt.Println("  models                               List downloaded models")
	fmt.Println("  info <model>                         Show model architecture")
	fmt.Println()
	fmt.Println("Global flags (before command):")
	fmt.Println("  --device <device>      Target: cpu, cuda, cuda:0, metal, vulkan (default: auto)")
	fmt.Println("  --out <dir>            Output directory for checkpoints and exports")
	fmt.Println("  --dry-run              Validate config without executing")
	fmt.Println("  --verbose              Show detailed logs")
}
