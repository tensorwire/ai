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
)

// cmdTrainAny trains from scratch on any Engine backend (CPU, WebGPU, Metal, CUDA).
// Uses Engine.MatMul for all matmuls. No TensorEngine, no GraphTrainEngine.
// Works everywhere but slower than platform-specific paths.
func cmdTrainAny() {
	fs := flag.NewFlagSet("train-any", flag.ExitOnError)

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

	fs.Parse(os.Args[2:])

	if *dataPath == "" {
		*dataPath = "data/tinystories_hf.txt"
		if _, err := os.Stat(*dataPath); err != nil {
			home, _ := os.UserHomeDir()
			*dataPath = filepath.Join(home, "data", "tinystories_hf.txt")
		}
	}

	eng := selectEngine("auto")

	dim := *dimFlag
	heads := *headsFlag
	kvHeads := *kvHeadsFlag
	headDim := dim / heads
	kvDim := kvHeads * headDim
	nLayers := *layersFlag
	ffnDim := *ffnDimFlag
	seqLen := *seqLenFlag
	vocabSize := 256
	lr := float32(*lrFlag)
	normEps := float32(1e-6)

	raw, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}
	n := seqLen + 1

	kaiming := func(rows, cols int) []float32 {
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d {
			d[i] = bound * (2*rand.Float32() - 1)
		}
		return d
	}

	type param struct{ D, G, M, V []float32 }
	newP := func(sz int) param {
		return param{make([]float32, sz), make([]float32, sz), make([]float32, sz), make([]float32, sz)}
	}

	embed := newP(vocabSize * dim)
	for i := range embed.D {
		embed.D[i] = float32(rand.NormFloat64()) * 0.02
	}
	finalNorm := newP(dim)
	for i := range finalNorm.D {
		finalNorm.D[i] = 1.0
	}

	type layer struct {
		norm1, wq, wk, wv, wo param
		norm2, gate, up, down param
	}
	layers := make([]layer, nLayers)
	for l := range layers {
		layers[l].norm1 = newP(dim)
		for i := range layers[l].norm1.D { layers[l].norm1.D[i] = 1.0 }
		layers[l].wq = newP(dim * dim)
		copy(layers[l].wq.D, kaiming(dim, dim))
		layers[l].wk = newP(kvDim * dim)
		copy(layers[l].wk.D, kaiming(kvDim, dim))
		layers[l].wv = newP(kvDim * dim)
		copy(layers[l].wv.D, kaiming(kvDim, dim))
		layers[l].wo = newP(dim * dim)
		copy(layers[l].wo.D, kaiming(dim, dim))
		layers[l].norm2 = newP(dim)
		for i := range layers[l].norm2.D { layers[l].norm2.D[i] = 1.0 }
		layers[l].gate = newP(ffnDim * dim)
		copy(layers[l].gate.D, kaiming(ffnDim, dim))
		layers[l].up = newP(ffnDim * dim)
		copy(layers[l].up.D, kaiming(ffnDim, dim))
		layers[l].down = newP(dim * ffnDim)
		copy(layers[l].down.D, kaiming(dim, ffnDim))
	}

	mv := func(out, W, x []float32, rows, cols int) {
		r := eng.MatMul(W, x, rows, cols, 1)
		copy(out, r)
	}

	adamStep := func(p *param, step int) {
		bc1 := float32(1.0 - math.Pow(0.9, float64(step)))
		bc2 := float32(1.0 - math.Pow(0.999, float64(step)))
		for i := range p.D {
			g := p.G[i]
			p.M[i] = 0.9*p.M[i] + 0.1*g
			p.V[i] = 0.999*p.V[i] + 0.001*g*g
			mh := p.M[i] / bc1
			vh := p.V[i] / bc2
			p.D[i] -= lr * (mh/(float32(math.Sqrt(float64(vh)))+1e-8) + 0.01*p.D[i])
			p.G[i] = 0
		}
	}

	nParams := len(embed.D) + len(finalNorm.D)
	for range layers {
		nParams += dim + dim*dim + kvDim*dim + kvDim*dim + dim*dim +
			dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Println("ai train-any — universal Engine path")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  data:     %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  model:    dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	if nParams > 1_000_000 {
		fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6)
	} else {
		fmt.Printf("  params:   %.1fK\n", float64(nParams)/1e3)
	}
	fmt.Printf("  training: steps=%d lr=%.0e\n", *stepsFlag, *lrFlag)
	fmt.Println()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	fmt.Println("Training...")
	t0 := time.Now()

	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)
		tokens := data[start : start+n]

		hidden := make([]float32, n*dim)
		for i, t := range tokens {
			copy(hidden[i*dim:(i+1)*dim], embed.D[t*dim:(t+1)*dim])
		}

		for li := range layers {
			l := &layers[li]
			xIn := make([]float32, n*dim)
			copy(xIn, hidden)

			for pos := 0; pos < n; pos++ {
				eng.RMSNorm(hidden[pos*dim:(pos+1)*dim], l.norm1.D, normEps)
			}

			q := make([]float32, n*dim)
			k := make([]float32, n*kvDim)
			v := make([]float32, n*kvDim)
			for pos := 0; pos < n; pos++ {
				mv(q[pos*dim:(pos+1)*dim], l.wq.D, hidden[pos*dim:(pos+1)*dim], dim, dim)
				mv(k[pos*kvDim:(pos+1)*kvDim], l.wk.D, hidden[pos*dim:(pos+1)*dim], kvDim, dim)
				mv(v[pos*kvDim:(pos+1)*kvDim], l.wv.D, hidden[pos*dim:(pos+1)*dim], kvDim, dim)
				applyRoPESingle(q[pos*dim:(pos+1)*dim], pos, headDim, heads, 10000.0)
				applyRoPESingle(k[pos*kvDim:(pos+1)*kvDim], pos, headDim, kvHeads, 10000.0)
			}

			attnOut := make([]float32, n*dim)
			kvMul := heads / kvHeads
			for pos := 0; pos < n; pos++ {
				for h := 0; h < heads; h++ {
					qOff := pos*dim + h*headDim
					kvH := h / kvMul
					scale := float32(1.0 / math.Sqrt(float64(headDim)))
					scores := make([]float32, pos+1)
					for t := 0; t <= pos; t++ {
						var dot float32
						for j := 0; j < headDim; j++ {
							dot += q[qOff+j] * k[t*kvDim+kvH*headDim+j]
						}
						scores[t] = dot * scale
					}
					softmax(scores, len(scores))
					for t := 0; t <= pos; t++ {
						for j := 0; j < headDim; j++ {
							attnOut[pos*dim+h*headDim+j] += scores[t] * v[t*kvDim+kvH*headDim+j]
						}
					}
				}
			}

			proj := make([]float32, n*dim)
			for pos := 0; pos < n; pos++ {
				mv(proj[pos*dim:(pos+1)*dim], l.wo.D, attnOut[pos*dim:(pos+1)*dim], dim, dim)
			}
			copy(hidden, xIn)
			for i := range hidden {
				hidden[i] += proj[i]
			}

			for pos := 0; pos < n; pos++ {
				buf := make([]float32, dim)
				copy(buf, hidden[pos*dim:(pos+1)*dim])
				eng.RMSNorm(buf, l.norm2.D, normEps)

				gateBuf := make([]float32, ffnDim)
				upBuf := make([]float32, ffnDim)
				mv(gateBuf, l.gate.D, buf, ffnDim, dim)
				mv(upBuf, l.up.D, buf, ffnDim, dim)
				for i := 0; i < ffnDim; i++ {
					gateBuf[i] = silu(gateBuf[i]) * upBuf[i]
				}
				downBuf := make([]float32, dim)
				mv(downBuf, l.down.D, gateBuf, dim, ffnDim)
				for i := 0; i < dim; i++ {
					hidden[pos*dim+i] += downBuf[i]
				}
			}
		}

		for pos := 0; pos < n; pos++ {
			eng.RMSNorm(hidden[pos*dim:(pos+1)*dim], finalNorm.D, normEps)
		}

		var loss float32
		for pos := 0; pos < n-1; pos++ {
			target := tokens[pos+1]
			logits := make([]float32, vocabSize)
			mv(logits, embed.D, hidden[pos*dim:(pos+1)*dim], vocabSize, dim)

			maxL := logits[0]
			for _, v := range logits[1:] {
				if v > maxL {
					maxL = v
				}
			}
			var sumExp float32
			for i := range logits {
				logits[i] = float32(math.Exp(float64(logits[i] - maxL)))
				sumExp += logits[i]
			}
			loss -= float32(math.Log(float64(logits[target]/sumExp) + 1e-10))

			scale := 1.0 / float32(n-1)
			for i := range logits {
				logits[i] /= sumExp
			}
			logits[target] -= 1.0
			for i := 0; i < vocabSize; i++ {
				grad := logits[i] * scale
				for j := 0; j < dim; j++ {
					embed.G[i*dim+j] += grad * hidden[pos*dim+j]
				}
			}
		}
		loss /= float32(n - 1)

		adamStep(&embed, step)

		if step%*logEvery == 0 || step == 1 {
			elapsed := time.Since(t0)
			stepsPerSec := float64(step) / elapsed.Seconds()
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, loss, lr, elapsed.Seconds(), stepsPerSec)
		}
	}

	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)\n", *stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds())
}
