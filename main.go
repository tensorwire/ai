// tesseract — GPU compute CLI. Train, infer, quantize, serve.
//
// Usage:
//   tesseract pull <model>              Download a model from HuggingFace
//   tesseract models                    List downloaded models
//   tesseract info <model>              Show model architecture
//   tesseract bench                     Benchmark your GPU
//   tesseract gpus                      Detect GPUs and calibrate
//   tesseract train                     Train a model
//   tesseract infer <model> "prompt"    Generate text
//   tesseract quantize <model> [q8|q4]  Quantize model weights
//   tesseract serve <model>             OpenAI-compatible API server

package main

import (
	"fmt"
	"os"
)

func main() {
	ParseGlobalFlags()

	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "pull":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: tesseract pull <model>")
			os.Exit(1)
		}
		cmdPull(os.Args[2])
	case "models":
		cmdModels()
	case "info":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: tesseract info <model>")
			os.Exit(1)
		}
		cmdInfo(os.Args[2])
	case "bench":
		cmdBench()
	case "gpus":
		cmdGPUs()
	case "train":
		cmdTrain()
	case "train-cuda":
		cmdTrainCUDA()
	case "train-any":
		cmdTrainAny()
	case "finetune":
		cmdFinetune()
	case "quantize":
		cmdQuantize()
	case "resume":
		cmdResume()
	case "merge":
		cmdMerge()
	case "serve":
		cmdServe()
	case "benchmark":
		cmdBenchmark()
	case "convert":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "Usage: tesseract convert gguf <model-dir> [output.gguf]")
			os.Exit(1)
		}
		cmdConvert(os.Args[2], os.Args[3:])
	case "infer":
		if len(os.Args) < 4 {
			fmt.Fprintln(os.Stderr, "Usage: tesseract infer <model> \"prompt text\"")
			os.Exit(1)
		}
		cmdInfer(os.Args[2], os.Args[3:])
	case "export":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: tesseract export qdrant <url> <collection> <output.jsonl>")
			os.Exit(1)
		}
		cmdExport(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	case "-v", "--version", "version":
		fmt.Println("tesseract v1.0.0 — powered by mongoose")
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Println("tesseract — GPU compute for Go. Zero Python.")
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  pull <model>            Download model from HuggingFace")
	fmt.Println("  models                  List downloaded models")
	fmt.Println("  info <model>            Show model architecture")
	fmt.Println("  bench                   Benchmark your GPU")
	fmt.Println("  gpus                    Detect all GPUs and calibrate")
	fmt.Println("  train                   Train a model")
	fmt.Println("  finetune <model> <data> Fine-tune a model")
	fmt.Println("  quantize <model> [q8|q4] Quantize model weights")
	fmt.Println("  resume <ckpt-dir> <data> Resume training from checkpoint")
	fmt.Println("  merge <model> <adapters> Merge LoRA adapters into base model")
	fmt.Println("  serve <model>           OpenAI-compatible API server")
	fmt.Println("  benchmark <model>       Profile inference speed + memory")
	fmt.Println("  convert gguf <model>    Convert safetensors → GGUF (for Ollama)")
	fmt.Println("  infer <model> \"prompt\"   Generate text")
	fmt.Println()
	fmt.Println("Global Flags (before command):")
	fmt.Println("  --device <dev>     Target: auto, cuda, metal, cpu (default: auto)")
	fmt.Println("  --precision <mode> Compute: auto, fp32, fp16, int8 (default: auto)")
	fmt.Println("  --out <dir>        Output directory for all artifacts")
	fmt.Println("  --verbose          Show detailed logs")
	fmt.Println("  --helix            Enable helix DNA optimizer")
	fmt.Println("  --needle           Enable needle INT8 kernels")
	fmt.Println()
	fmt.Println("Examples:")
	fmt.Println("  tesseract pull Qwen/Qwen2.5-14B")
	fmt.Println("  tesseract bench")
	fmt.Println("  tesseract infer qwen2-0.5b \"The meaning of life is\"")
	fmt.Println("  tesseract quantize qwen2.5-14b q8")
	fmt.Println("  tesseract serve qwen2.5-14b")
}
