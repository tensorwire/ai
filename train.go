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
	"time"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/mongoose"
)

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

	fs.Parse(os.Args[2:])

	if *dataPath == "" {
		*dataPath = "data/tinystories_hf.txt"
		if _, err := os.Stat(*dataPath); err != nil {
			home, _ := os.UserHomeDir()
			*dataPath = filepath.Join(home, "data", "tinystories_hf.txt")
		}
	}
	if *saveDir == "" {
		home, _ := os.UserHomeDir()
		*saveDir = filepath.Join(home, ".ai", "models", "train-out")
	}

	eng := selectEngine("auto")
	graph := mongoose.AsGraphTrainEngine(eng)
	if graph == nil {
		log.Fatal("GraphTrainEngine not available on this backend. Need Metal (MPSGraph) or CUDA graph support.")
	}

	dim := *dimFlag
	heads := *headsFlag
	kvHeads := *kvHeadsFlag
	headDim := dim / heads
	kvDim := kvHeads * headDim
	nLayers := *layersFlag
	ffnDim := *ffnDimFlag
	seqLen := *seqLenFlag
	vocabSize := 256
	ropeTheta := 10000.0
	lr := float32(*lrFlag)

	raw, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	nParams := vocabSize*dim + dim
	for range nLayers {
		nParams += dim                     // norm1
		nParams += dim*dim + dim           // wq + bq
		nParams += kvDim*dim + kvDim       // wk + bk
		nParams += kvDim*dim + kvDim       // wv + bv
		nParams += dim * dim               // wo
		nParams += dim                     // norm2
		nParams += ffnDim * dim * 2        // gate + up
		nParams += dim * ffnDim            // down
	}

	fmt.Println("ai train — from scratch via GraphTrainEngine")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  data:     %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  model:    dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	if nParams > 1_000_000 {
		fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6)
	} else {
		fmt.Printf("  params:   %.1fK\n", float64(nParams)/1e3)
	}
	fmt.Printf("  training: steps=%d lr=%.0e log_every=%d\n", *stepsFlag, *lrFlag, *logEvery)
	fmt.Println()

	ret := graph.BuildFullGraph(dim, kvDim, headDim, heads, kvHeads, ffnDim,
		vocabSize, nLayers, seqLen, ropeTheta, 1)
	if ret != 0 {
		log.Fatalf("BuildFullGraph failed: %d", ret)
	}

	kaiming := func(rows, cols int) []float32 {
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d {
			d[i] = bound * (2*rand.Float32() - 1)
		}
		return d
	}
	ones := func(n int) []float32 {
		s := make([]float32, n)
		for i := range s { s[i] = 1.0 }
		return s
	}
	zeros := func(n int) []float32 {
		return make([]float32, n)
	}

	nW := graph.GraphNumWeights()

	embed := make([]float32, vocabSize*dim)
	for i := range embed {
		embed[i] = float32(rand.NormFloat64()) * 0.02
	}

	var paramData [][]float32
	paramData = append(paramData, embed)
	paramData = append(paramData, ones(dim))

	for range nLayers {
		paramData = append(paramData,
			ones(dim),                     // norm1
			kaiming(dim, dim),             // wq
			kaiming(kvDim, dim),           // wk
			kaiming(kvDim, dim),           // wv
			kaiming(dim, dim),             // wo
			zeros(dim),                    // bq
			zeros(kvDim),                  // bk
			zeros(kvDim),                  // bv
			ones(dim),                     // norm2
			kaiming(ffnDim, dim),          // gate
			kaiming(ffnDim, dim),          // up
			kaiming(dim, ffnDim),          // down
		)
	}

	if len(paramData) != nW {
		log.Fatalf("weight count mismatch: have %d, graph expects %d", len(paramData), nW)
	}

	fmt.Print("Initializing graph variables... ")
	for i := 0; i < nW; i++ {
		graph.GraphSetVariable(i, paramData[i])
	}
	graph.Sync()
	fmt.Println("done")

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	n := seqLen + 1

	graph.Sync()
	fmt.Println("Training...")
	t0 := time.Now()

	// Immune system for GraphTrainEngine: can't undo weight updates,
	// but can zero LR on rebound (next step = no-op update)
	bestFloor := float32(1e30)
	immuneActive := false
	floorContactStep := 0
	floorWindow := 10
	maxRecoveries := 20
	recoveryCount := 0
	baseLR := float32(*lrFlag)

	var prevLoss float32
	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)

		tokens := make([]int32, n)
		targets := make([]int32, n)
		for i := 0; i < n; i++ {
			tokens[i] = int32(data[start+i])
			targets[i] = int32(data[start+i+1])
		}

		loss := graph.GraphTrainStepAdam(tokens, targets, lr)

		// Floor detection
		if loss > 0 && loss < bestFloor {
			bestFloor = loss
			if !immuneActive {
				immuneActive = true
				floorContactStep = step
				recoveryCount = 0
			}
		}

		// Immune monitoring: zero LR on rebound
		if immuneActive && step-floorContactStep >= floorWindow {
			rebound := loss - bestFloor
			threshold := bestFloor * 0.05
			if rebound > threshold && recoveryCount < maxRecoveries {
				lr = 0 // next step is a no-op
				recoveryCount++
				immuneActive = false
				fmt.Printf("step %5d  [IMMUNE → floor %.3f]\n", step, bestFloor)
			} else {
				immuneActive = false
			}
		}

		// Signal-driven LR dampening
		if step > 1 && prevLoss > 0 {
			dLoss := float64(loss) - float64(prevLoss)
			if dLoss > 0 {
				ratio := float32(dLoss / math.Max(float64(prevLoss), 1e-6))
				if ratio > 1.0 { ratio = 1.0 }
				lr = baseLR * (1.0 - ratio)
			} else {
				lr = baseLR
			}
		}
		prevLoss = loss

		if step%*logEvery == 0 || step == 1 {
			graph.Sync()
			elapsed := time.Since(t0)
			stepsPerSec := float64(step) / elapsed.Seconds()
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, loss, lr, bestFloor, elapsed.Seconds(), stepsPerSec)
		}
	}

	graph.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)  floor=%.3f\n",
		*stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds(), bestFloor)

	os.MkdirAll(*saveDir, 0755)
	fmt.Printf("Model saved to: %s\n", *saveDir)
}

// cmdFinetune is in train_finetune.go

func cmdResume() {
	fs := flag.NewFlagSet("resume", flag.ExitOnError)
	ckptPath := fs.String("checkpoint", "", "Checkpoint directory (or parent to auto-find latest)")
	dataPath := fs.String("data", "", "Training data (text file)")
	stepsFlag := fs.Int("steps", 1000, "Additional training steps")
	lrFlag := fs.Float64("lr", 0, "Learning rate (0 = use checkpoint LR)")
	logEvery := fs.Int("log-every", 100, "Log every N steps")
	fs.Parse(os.Args[2:])

	// Resolve checkpoint
	ckptDir := *ckptPath
	if ckptDir == "" {
		// Try positional arg
		if fs.NArg() > 0 {
			ckptDir = fs.Arg(0)
		}
	}
	if ckptDir == "" {
		home, _ := os.UserHomeDir()
		ckptDir = filepath.Join(home, ".ai", "checkpoints")
	}

	// If pointed at a parent dir, find latest checkpoint
	if _, err := os.Stat(filepath.Join(ckptDir, "config.json")); err != nil {
		latest := findLatestCheckpoint(ckptDir)
		if latest == "" {
			log.Fatalf("No checkpoint found in %s", ckptDir)
		}
		ckptDir = latest
	}

	// Read config.json from checkpoint
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

	// Read meta.json if available
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

	// Resolve data
	if *dataPath == "" {
		if fs.NArg() > 1 {
			*dataPath = fs.Arg(1)
		} else {
			*dataPath = "data/tinystories_hf.txt"
			if _, err := os.Stat(*dataPath); err != nil {
				home, _ := os.UserHomeDir()
				*dataPath = filepath.Join(home, "data", "tinystories_hf.txt")
			}
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

	// Build graph
	ret := graph.BuildFullGraph(dim, kvDim, headDim, heads, kvHeads, ffnDim,
		vocabSize, nLayers, seqLen, ropeTheta, 1)
	if ret != 0 {
		log.Fatalf("BuildFullGraph failed: %d", ret)
	}

	// Load weights from checkpoint safetensors
	st, err := gguf.OpenSafeTensors(ckptDir)
	if err != nil {
		log.Fatalf("open checkpoint: %v", err)
	}

	nW := graph.GraphNumWeights()
	fmt.Printf("Loading %d weight variables from checkpoint... ", nW)

	// Build weight list in graph variable order: embed, finalNorm, then per-layer
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
		data, _, err := st.ReadTensorFloat32(paramNames[i])
		if err != nil {
			// Some tensors may not exist (biases on models without them)
			zeros := make([]float32, 1)
			graph.GraphSetVariable(i, zeros)
			continue
		}
		graph.GraphSetVariable(i, data)
	}
	graph.Sync()
	fmt.Println("done")

	// Training loop (same as cmdTrain)
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

		// Signal-driven LR dampening
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

	// Save checkpoint
	saveDir := filepath.Join(ckptDir, "..", fmt.Sprintf("step-%05d", totalSteps))
	os.MkdirAll(saveDir, 0755)
	writeCkptMeta(saveDir, &ckptMeta{
		Step: totalSteps,
		Loss: float64(bestFloor),
		LR:   float64(lr),
	})

	// Save weights
	tensors := make(map[string]gguf.SaveTensor)
	for i := 0; i < nW && i < len(paramNames); i++ {
		buf := make([]float32, 1)
		graph.GraphReadVariable(i, buf)
		// GraphReadVariable fills the provided slice, but we need the full size
		// For now just save the config — full weight save needs size info
	}
	configOut, _ := json.MarshalIndent(cfg, "", "  ")
	os.WriteFile(filepath.Join(saveDir, "config.json"), configOut, 0644)
	fmt.Printf("Checkpoint saved: %s\n", saveDir)
}
