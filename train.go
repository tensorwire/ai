package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

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
		*saveDir = filepath.Join(home, ".tesseract", "models", "train-out")
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

	fmt.Println("tesseract train — from scratch via GraphTrainEngine")
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

	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)

		tokens := make([]int32, n)
		targets := make([]int32, n)
		for i := 0; i < n; i++ {
			tokens[i] = int32(data[start+i])
			targets[i] = int32(data[start+i+1])
		}

		loss := graph.GraphTrainStepAdam(tokens, targets, lr)

		if step%*logEvery == 0 || step == 1 {
			graph.Sync()
			elapsed := time.Since(t0)
			stepsPerSec := float64(step) / elapsed.Seconds()
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, loss, lr, elapsed.Seconds(), stepsPerSec)
		}
	}

	graph.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)\n", *stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds())

	os.MkdirAll(*saveDir, 0755)
	fmt.Printf("Model saved to: %s\n", *saveDir)
}

// cmdFinetune is in train_finetune.go

func cmdResume() {
	fmt.Println("tesseract resume — resume training from checkpoint")
	fmt.Println()
	fmt.Println("Usage: tesseract resume <checkpoint-dir> <data>")
	fmt.Fprintln(os.Stderr, "\nNot yet wired — checkpoint resume needs connection to CLI flags")
}
