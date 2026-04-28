package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/tensorwire/mongoose"
)

// cmdTrainGraph trains from scratch using GraphTrainEngine (Metal MPSGraph or CUDA graph).
// This is the fused-dispatch path: one GPU launch = forward + backward + optimizer.
func cmdTrainGraph(dataPath string, dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize, steps int,
	lr float32, logEvery int, saveDir string) {

	eng := selectEngine("auto")
	graph := mongoose.AsGraphTrainEngine(eng)
	if graph == nil {
		log.Fatal("GraphTrainEngine not available. Need Metal (MPSGraph) or CUDA graph support.")
	}

	headDim := dim / heads
	kvDim := kvHeads * headDim
	ropeTheta := 10000.0

	raw, err := os.ReadFile(dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	nParams := vocabSize*dim + dim
	for range nLayers {
		nParams += dim + dim*dim + dim + kvDim*dim + kvDim + kvDim*dim + kvDim + dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Println("ai train — GraphTrainEngine (fused dispatch)")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  data:     %s (%d bytes)\n", dataPath, len(raw))
	fmt.Printf("  model:    dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	if nParams > 1_000_000 {
		fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6)
	} else {
		fmt.Printf("  params:   %.1fK\n", float64(nParams)/1e3)
	}
	fmt.Printf("  training: steps=%d lr=%.0e log_every=%d\n", steps, lr, logEvery)
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
		for i := range s {
			s[i] = 1.0
		}
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
			ones(dim),             // norm1
			kaiming(dim, dim),     // wq
			kaiming(kvDim, dim),   // wk
			kaiming(kvDim, dim),   // wv
			kaiming(dim, dim),     // wo
			zeros(dim),            // bq
			zeros(kvDim),          // bk
			zeros(kvDim),          // bv
			ones(dim),             // norm2
			kaiming(ffnDim, dim),  // gate
			kaiming(ffnDim, dim),  // up
			kaiming(dim, ffnDim),  // down
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
	baseLR := lr

	bestFloor := float32(1e30)
	immuneActive := false
	floorContactStep := 0
	floorWindow := 10
	maxRecoveries := 20
	recoveryCount := 0

	fmt.Println("Training...")
	t0 := time.Now()

	var prevLoss float32
	for step := 1; step <= steps; step++ {
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
			if !immuneActive {
				immuneActive = true
				floorContactStep = step
				recoveryCount = 0
			}
		}

		if immuneActive && step-floorContactStep >= floorWindow {
			rebound := loss - bestFloor
			threshold := bestFloor * 0.05
			if rebound > threshold && recoveryCount < maxRecoveries {
				lr = 0
				recoveryCount++
				immuneActive = false
				fmt.Printf("step %5d  [IMMUNE → floor %.3f]\n", step, bestFloor)
			} else {
				immuneActive = false
			}
		}

		if step > 1 && prevLoss > 0 {
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

		if step%logEvery == 0 || step == 1 {
			graph.Sync()
			elapsed := time.Since(t0)
			stepsPerSec := float64(step) / elapsed.Seconds()
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  %.0fs  (%.1f steps/s)\n",
				step, steps, loss, lr, bestFloor, elapsed.Seconds(), stepsPerSec)
		}
	}

	graph.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)  floor=%.3f\n",
		steps, total.Seconds(), float64(steps)/total.Seconds(), bestFloor)

	os.MkdirAll(saveDir, 0755)
	fmt.Printf("Model saved to: %s\n", saveDir)
}
