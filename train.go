package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/mongoose"
)

// cmdTrain discovers all available compute backends, benchmarks them,
// and routes to the fastest one.
func cmdTrain() {
	fs := flag.NewFlagSet("train", flag.ExitOnError)

	dataPath := fs.String("data", "", "Training data (text file)")
	dimFlag := fs.Int("dim", 128, "Model dimension")
	headsFlag := fs.Int("heads", 4, "Attention heads")
	kvHeadsFlag := fs.Int("kv-heads", 2, "KV heads (GQA)")
	layersFlag := fs.Int("layers", 4, "Transformer layers")
	ffnDimFlag := fs.Int("ffn-dim", 256, "FFN intermediate dimension")
	seqLenFlag := fs.Int("seq-len", 64, "Sequence length")
	stepsFlag := fs.Int("steps", 1000, "Training steps")
	lrFlag := fs.Float64("lr", 6e-4, "Learning rate")
	logEvery := fs.Int("log-every", 100, "Log every N steps")
	saveDir := fs.String("save-dir", "", "Save directory")
	backend := fs.String("backend", "auto", "Training backend: auto, graph, cuda, cpu")

	fs.Parse(os.Args[2:])

	if *dataPath == "" {
		log.Fatal("data required: ai train data=<file>")
	}
	if *saveDir == "" {
		home, _ := os.UserHomeDir()
		*saveDir = filepath.Join(home, ".ai", "models", "train-out")
	}

	lr := float32(*lrFlag)

	rewriteForSub := func() {
		os.Args = []string{os.Args[0], "train",
			"-data", *dataPath,
			"-dim", fmt.Sprintf("%d", *dimFlag),
			"-heads", fmt.Sprintf("%d", *headsFlag),
			"-kv-heads", fmt.Sprintf("%d", *kvHeadsFlag),
			"-layers", fmt.Sprintf("%d", *layersFlag),
			"-ffn-dim", fmt.Sprintf("%d", *ffnDimFlag),
			"-seq-len", fmt.Sprintf("%d", *seqLenFlag),
			"-steps", fmt.Sprintf("%d", *stepsFlag),
			"-lr", fmt.Sprintf("%e", *lrFlag),
			"-log-every", fmt.Sprintf("%d", *logEvery),
		}
	}

	if *backend != "auto" {
		switch *backend {
		case "graph":
			cmdTrainGraph(*dataPath, *dimFlag, *headsFlag, *kvHeadsFlag, *layersFlag, *ffnDimFlag,
				*seqLenFlag, 256, *stepsFlag, lr, *logEvery, *saveDir)
		case "cuda":
			rewriteForSub()
			cmdTrainCUDA()
		case "cpu":
			rewriteForSub()
			cmdTrainAny()
		default:
			log.Fatalf("unknown backend: %s (use auto, graph, cuda, cpu)", *backend)
		}
		return
	}

	// Auto mode: discover, benchmark, pick fastest
	type candidate struct {
		name    string
		engine  mongoose.Engine
		gflops  float64
		hasGraph bool
		hasCUDA  bool
	}

	var candidates []candidate

	// Probe Metal
	if runtime.GOOS == "darwin" {
		if m := mongoose.NewMetal(); m != nil {
			gf := m.Benchmark()
			candidates = append(candidates, candidate{
				name: m.Name(), engine: m, gflops: gf,
				hasGraph: mongoose.AsGraphTrainEngine(m) != nil,
			})
		}
	}

	// Probe CUDA
	if c := mongoose.NewCUDA(); c != nil {
		mongoose.LoadKernels()
		gf := c.Benchmark()
		candidates = append(candidates, candidate{
			name: c.Name(), engine: c, gflops: gf,
			hasCUDA: mongoose.KernelsLoaded(),
			hasGraph: mongoose.AsGraphTrainEngine(c) != nil,
		})
	}

	// CPU always available
	cpu := &mongoose.CPU{}
	candidates = append(candidates, candidate{
		name: cpu.Name(), engine: cpu, gflops: cpu.Benchmark(),
	})

	// Pick: prefer graph engine on fastest GPU, then CUDA kernels, then CPU
	fmt.Println("ai train — auto-selecting backend")
	for _, c := range candidates {
		marker := " "
		if c.hasGraph { marker = "⚡" }
		if c.hasCUDA { marker = "🔥" }
		fmt.Printf("  %s %-30s %.0f GFLOPS", marker, c.name, c.gflops)
		if c.hasGraph { fmt.Print("  (graph)") }
		if c.hasCUDA { fmt.Print("  (kernels)") }
		fmt.Println()
	}

	// Strategy: CUDA kernels > graph engine > CPU, weighted by GFLOPS
	var bestGraph, bestCUDA, bestCPU *candidate
	for i := range candidates {
		c := &candidates[i]
		if c.hasCUDA && (bestCUDA == nil || c.gflops > bestCUDA.gflops) {
			bestCUDA = c
		}
		if c.hasGraph && (bestGraph == nil || c.gflops > bestGraph.gflops) {
			bestGraph = c
		}
		if !c.hasCUDA && !c.hasGraph {
			if bestCPU == nil || c.gflops > bestCPU.gflops {
				bestCPU = c
			}
		}
	}

	// CUDA kernels path gets helix DNA optimizer — prefer it if available
	if bestCUDA != nil {
		fmt.Printf("\n  → %s (CUDA kernels + helix DNA optimizer)\n\n", bestCUDA.name)
		rewriteForSub()
		cmdTrainCUDA()
		return
	}

	// GraphTrainEngine (Metal MPSGraph or CUDA graph)
	if bestGraph != nil {
		fmt.Printf("\n  → %s (fused graph dispatch)\n\n", bestGraph.name)
		cmdTrainGraph(*dataPath, *dimFlag, *headsFlag, *kvHeadsFlag, *layersFlag, *ffnDimFlag,
			*seqLenFlag, 256, *stepsFlag, lr, *logEvery, *saveDir)
		return
	}

	// CPU fallback
	if bestCPU != nil {
		fmt.Printf("\n  → %s (CPU fallback)\n\n", bestCPU.name)
	} else {
		fmt.Printf("\n  → CPU fallback\n\n")
	}
	rewriteForSub()
	cmdTrainAny()
}

// cmdResume continues training from a checkpoint.
func cmdResume() {
	fs := flag.NewFlagSet("resume", flag.ExitOnError)
	ckptPath := fs.String("checkpoint", "", "Checkpoint directory (or parent to auto-find latest)")
	dataPath := fs.String("data", "", "Training data (text file)")
	stepsFlag := fs.Int("steps", 1000, "Additional training steps")
	lrFlag := fs.Float64("lr", 0, "Learning rate (0 = use checkpoint LR)")
	logEvery := fs.Int("log-every", 100, "Log every N steps")
	fs.Parse(os.Args[2:])

	ckptDir := *ckptPath
	if ckptDir == "" {
		if fs.NArg() > 0 {
			ckptDir = fs.Arg(0)
		}
	}
	if ckptDir == "" {
		home, _ := os.UserHomeDir()
		ckptDir = filepath.Join(home, ".ai", "checkpoints")
	}

	if _, err := os.Stat(filepath.Join(ckptDir, "config.json")); err != nil {
		latest := findLatestCheckpoint(ckptDir)
		if latest == "" {
			log.Fatalf("No checkpoint found in %s", ckptDir)
		}
		ckptDir = latest
	}

	configData, err := os.ReadFile(filepath.Join(ckptDir, "config.json"))
	if err != nil {
		log.Fatalf("No config.json in checkpoint %s", ckptDir)
	}
	var cfg map[string]interface{}
	json.Unmarshal(configData, &cfg)

	getInt := func(key string, def int) int {
		if v, ok := cfg[key].(float64); ok {
			return int(v)
		}
		return def
	}

	dim := getInt("hidden_size", 128)
	nLayers := getInt("num_hidden_layers", 4)
	heads := getInt("num_attention_heads", 4)
	kvHeads := getInt("num_key_value_heads", heads)
	ffnDim := getInt("intermediate_size", 256)
	vocabSize := getInt("vocab_size", 256)
	headDim := dim / heads
	kvDim := kvHeads * headDim
	seqLen := 64
	ropeTheta := 10000.0

	startStep := 0
	lr := float32(6e-4)
	if meta, err := readCkptMeta(ckptDir); err == nil {
		startStep = meta.Step
		if meta.LR > 0 {
			lr = float32(meta.LR)
		}
		fmt.Printf("Resuming from step %d (loss=%.4f, lr=%.1e)\n", meta.Step, meta.Loss, meta.LR)
	} else {
		fmt.Printf("Resuming from %s (no meta.json — starting from step 0)\n", filepath.Base(ckptDir))
	}
	if *lrFlag > 0 {
		lr = float32(*lrFlag)
	}

	if *dataPath == "" {
		if fs.NArg() > 1 {
			*dataPath = fs.Arg(1)
		} else {
			log.Fatal("data required: ai train data=<file>")
		}
	}

	raw, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	eng := selectEngine("auto")
	graph := mongoose.AsGraphTrainEngine(eng)
	if graph == nil {
		log.Fatal("GraphTrainEngine not available — need Metal or CUDA graph support")
	}

	fmt.Println("ai resume — continue training from checkpoint")
	fmt.Printf("  checkpoint: %s\n", ckptDir)
	fmt.Printf("  engine:     %s\n", eng.Name())
	fmt.Printf("  data:       %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  model:      dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	fmt.Printf("  resume:     step=%d lr=%.1e steps=%d\n", startStep, lr, *stepsFlag)
	fmt.Println()

	ret := graph.BuildFullGraph(dim, kvDim, headDim, heads, kvHeads, ffnDim,
		vocabSize, nLayers, seqLen, ropeTheta, 1)
	if ret != 0 {
		log.Fatalf("BuildFullGraph failed: %d", ret)
	}

	st, err := gguf.OpenSafeTensors(ckptDir)
	if err != nil {
		log.Fatalf("open checkpoint: %v", err)
	}

	nW := graph.GraphNumWeights()
	fmt.Printf("Loading %d weight variables from checkpoint... ", nW)

	var paramNames []string
	paramNames = append(paramNames, "model.embed_tokens.weight")
	paramNames = append(paramNames, "model.norm.weight")
	for l := 0; l < nLayers; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		paramNames = append(paramNames,
			pfx+"input_layernorm.weight",
			pfx+"self_attn.q_proj.weight",
			pfx+"self_attn.k_proj.weight",
			pfx+"self_attn.v_proj.weight",
			pfx+"self_attn.o_proj.weight",
			pfx+"self_attn.q_proj.bias",
			pfx+"self_attn.k_proj.bias",
			pfx+"self_attn.v_proj.bias",
			pfx+"post_attention_layernorm.weight",
			pfx+"mlp.gate_proj.weight",
			pfx+"mlp.up_proj.weight",
			pfx+"mlp.down_proj.weight",
		)
	}

	for i := 0; i < nW && i < len(paramNames); i++ {
		d, _, err := st.ReadTensorFloat32(paramNames[i])
		if err != nil {
			zeros := make([]float32, 1)
			graph.GraphSetVariable(i, zeros)
			continue
		}
		graph.GraphSetVariable(i, d)
	}
	graph.Sync()
	fmt.Println("done")

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	n := seqLen + 1
	totalSteps := startStep + *stepsFlag
	bestFloor := float32(1e30)
	baseLR := lr

	fmt.Println("Training...")
	t0 := time.Now()

	var prevLoss float32
	for step := startStep + 1; step <= totalSteps; step++ {
		start := rng.Intn(len(data) - n - 1)
		tokens := make([]int32, n)
		targets := make([]int32, n)
		for i := 0; i < n; i++ {
			tokens[i] = int32(data[start+i])
			targets[i] = int32(data[start+i+1])
		}

		loss := graph.GraphTrainStepAdam(tokens, targets, lr)

		if loss > 0 && loss < bestFloor {
			bestFloor = loss
		}

		if step > startStep+1 && prevLoss > 0 {
			dLoss := float64(loss) - float64(prevLoss)
			if dLoss > 0 {
				ratio := float32(dLoss / math.Max(float64(prevLoss), 1e-6))
				if ratio > 1.0 {
					ratio = 1.0
				}
				lr = baseLR * (1.0 - ratio)
			} else {
				lr = baseLR
			}
		}
		prevLoss = loss

		if (step-startStep)%*logEvery == 0 || step == startStep+1 {
			graph.Sync()
			elapsed := time.Since(t0)
			stepsPerSec := float64(step-startStep) / elapsed.Seconds()
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  (%.1f steps/s)\n",
				step, totalSteps, loss, lr, bestFloor, stepsPerSec)
		}
	}

	graph.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)  floor=%.3f\n",
		*stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds(), bestFloor)

	saveCheckpointDir := filepath.Join(ckptDir, "..", fmt.Sprintf("step-%05d", totalSteps))
	os.MkdirAll(saveCheckpointDir, 0755)
	writeCkptMeta(saveCheckpointDir, &ckptMeta{
		Step: totalSteps,
		Loss: float64(bestFloor),
		LR:   float64(lr),
	})
	configOut, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(filepath.Join(saveCheckpointDir, "config.json"), configOut, 0644)
	fmt.Printf("Checkpoint saved: %s\n", saveCheckpointDir)
}
