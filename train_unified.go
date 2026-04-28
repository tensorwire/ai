package main

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/tensorwire/mongoose"
)

// cmdTrainUnified handles both from-scratch and fine-tuning via a single command:
//   ai train data=file.txt                    → from scratch
//   ai train model=TinyLlama data=file.txt    → finetune
//
// All settings have opinionated defaults. Override with key=value:
//   ai train data=file.txt steps=5000 lr=3e-4 dim=512 layers=8
func cmdTrainUnified(args map[string]string) {
	dataPath := args["data"]
	modelPath := args["model"]

	if dataPath == "" {
		// Check positional args
		if v, ok := args["_0"]; ok && modelPath == "" {
			dataPath = v
		} else if v, ok := args["_1"]; ok {
			dataPath = v
		}
	}

	if dataPath == "" {
		dataPath = findData()
	}
	if dataPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ai train data=<file>")
		fmt.Fprintln(os.Stderr, "       ai train model=<name> data=<file>")
		os.Exit(1)
	}

	if modelPath != "" {
		modelPath = resolveModelPath(modelPath)
		runFinetune(modelPath, dataPath, args)
	} else {
		runFromScratch(dataPath, args)
	}
}

// resolveModelPath resolves a model name to a directory path.
func resolveModelPath(name string) string {
	if info, err := os.Stat(name); err == nil && info.IsDir() {
		return name
	}
	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(home, ".ai", "models", name),
		filepath.Join(home, ".tesseract", "models", name),
	}
	for _, c := range candidates {
		if _, err := os.Stat(filepath.Join(c, "config.json")); err == nil {
			return c
		}
	}
	// Try case-insensitive match
	for _, base := range []string{
		filepath.Join(home, ".ai", "models"),
		filepath.Join(home, ".tesseract", "models"),
	} {
		entries, err := os.ReadDir(base)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if strings.EqualFold(e.Name(), name) {
				return filepath.Join(base, e.Name())
			}
		}
	}
	log.Fatalf("model not found: %s", name)
	return ""
}

// findData returns empty — data must be explicitly provided.
func findData() string {
	return ""
}

// runFromScratch picks the best available backend and trains.
func runFromScratch(dataPath string, args map[string]string) {
	// Probe available backends without full init — check what we can use.
	// The actual command will initialize its own engine.
	backend := detectBestBackend()

	switch backend {
	case "cuda-kernels":
		log.Println("[ai] using CUDA kernel path")
		injectArgs(args, dataPath)
		cmdTrainCUDA()
	case "metal":
		log.Println("[ai] using Metal kernel path")
		injectArgs(args, dataPath)
		cmdTrainMetal()
	case "webgpu":
		log.Println("[ai] using WebGPU/Vulkan path (CPU training with GPU matmul)")
		injectArgs(args, dataPath)
		cmdTrainAny()
	default:
		log.Println("[ai] using CPU path")
		injectArgs(args, dataPath)
		cmdTrainAny()
	}
}

// detectBestBackend probes available engines and returns the best training backend.
func detectBestBackend() string {
	if runtime.GOOS == "darwin" {
		if m := mongoose.NewMetal(); m != nil {
			return "metal"
		}
	}
	if c := mongoose.NewCUDA(); c != nil {
		return "cuda-kernels"
	}
	if w := mongoose.NewWebGPU(); w != nil {
		return "webgpu"
	}
	return "cpu"
}

// runFinetune fine-tunes a pretrained model.
// Routes to the best available backend — CUDA uses cmdFinetune (INT8 + Helix),
// Metal uses cmdTrainMetal with --resume pointing at the pretrained weights.
func runFinetune(modelPath, dataPath string, args map[string]string) {
	backend := detectBestBackend()

	switch backend {
	case "cuda-kernels":
		profile := AutoDetect(modelPath)
		cmdFinetuneCUDA(modelPath, dataPath, args,
			kvInt(args, "steps", profile.Steps),
			kvFloat(args, "lr", profile.LR),
			kvInt(args, "log", 50))

	case "metal":
		args["resume"] = modelPath
		injectArgs(args, dataPath)
		cmdTrainMetal()

	default:
		log.Printf("[ai] finetune: no optimized GPU path for %s — using CPU training with pretrained weights", backend)
		args["resume"] = modelPath
		injectArgs(args, dataPath)
		cmdTrainAny()
	}
}

// cmdResumeKV wraps resume with key=value arg support.
//
//	ai resume checkpoint=./checkpoints data=corpus.txt
//	ai resume data=corpus.txt steps=500
func cmdResumeKV(args map[string]string) {
	var resumeArgs []string
	resumeArgs = append(resumeArgs, os.Args[0], "resume")

	if v, ok := args["checkpoint"]; ok {
		resumeArgs = append(resumeArgs, "--checkpoint", v)
	} else if v, ok := args["_0"]; ok {
		resumeArgs = append(resumeArgs, "--checkpoint", v)
	}

	if v, ok := args["data"]; ok {
		resumeArgs = append(resumeArgs, "--data", v)
	}
	if v, ok := args["steps"]; ok {
		resumeArgs = append(resumeArgs, "--steps", v)
	}
	if v, ok := args["lr"]; ok {
		resumeArgs = append(resumeArgs, "--lr", v)
	}
	if v, ok := args["log"]; ok {
		resumeArgs = append(resumeArgs, "--log-every", v)
	}

	os.Args = resumeArgs
	cmdResume()
}

// injectArgs rewrites os.Args for the legacy flag-based commands.
func injectArgs(args map[string]string, dataPath string) {
	var newArgs []string
	newArgs = append(newArgs, os.Args[0], os.Args[1])
	newArgs = append(newArgs, "--data", dataPath)

	kvToFlag := map[string]string{
		"dim":    "--dim",
		"heads":  "--heads",
		"kv":     "--kv-heads",
		"layers": "--layers",
		"ffn":    "--ffn-dim",
		"seq":    "--seq-len",
		"steps":  "--steps",
		"lr":     "--lr",
		"log":    "--log-every",
		"resume": "--resume",
	}

	for k, flag := range kvToFlag {
		if v, ok := args[k]; ok {
			newArgs = append(newArgs, flag, v)
		}
	}

	os.Args = newArgs
}

// kvInt extracts an int from key=value args with a default.
func kvInt(args map[string]string, key string, def int) int {
	if v, ok := args[key]; ok {
		n, err := strconv.Atoi(v)
		if err != nil {
			log.Fatalf("invalid %s=%s: %v", key, v, err)
		}
		return n
	}
	return def
}

// kvFloat extracts a float64 from key=value args with a default.
func kvFloat(args map[string]string, key string, def float64) float64 {
	if v, ok := args[key]; ok {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			log.Fatalf("invalid %s=%s: %v", key, v, err)
		}
		return f
	}
	return def
}
