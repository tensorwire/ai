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

// cmdTrainCUDA trains from scratch on CUDA.
// Weights live on GPU (TensorEngine). Activations live in L3 pinned memory.
// CPU does element-wise ops (RMSNorm, RoPE, attention) on L3 slices.
// GPU does matmuls reading weights from VRAM and activations from L3 cache.
// One path. No copies. Clean division: CPU owns data between dispatches,
// GPU owns it during cuBLAS calls. Sync() is the fence.
func cmdTrainCUDA() {
	fs := flag.NewFlagSet("train-cuda", flag.ExitOnError)

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
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatal("TensorEngine not available — need CUDA backend")
	}
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok {
		log.Fatal("train-cuda requires CUDA backend")
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
	lr := float32(*lrFlag)

	raw, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	n := seqLen + 1

	// L3 bridge: pinned memory for activations and gradients.
	// Layout: hidden[n*dim] | buf[n*dim] | q[n*dim] | k[n*kvDim] | v[n*kvDim] |
	//         attnOut[n*dim] | ffnG[n*ffnDim] | ffnU[n*ffnDim] | logits[n*vocabSize] |
	//         dHidden[n*dim] | dAttnOut[n*dim] | dFFN[n*ffnDim] | embedGrad[vocabSize*dim]
	hiddenSz := n * dim
	qSz := n * dim
	kSz := n * kvDim
	vSz := n * kvDim
	attnSz := n * dim
	ffnGSz := n * ffnDim
	ffnUSz := n * ffnDim
	logitsSz := n * vocabSize
	dHiddenSz := n * dim
	dAttnSz := n * dim
	dFFNSz := n * ffnDim
	embedGradSz := vocabSize * dim

	totalFloats := hiddenSz + hiddenSz + qSz + kSz + vSz + attnSz +
		ffnGSz + ffnUSz + logitsSz + dHiddenSz + dAttnSz + dFFNSz + embedGradSz
	bridgeBytes := totalFloats * 4

	bridge := cuda.AllocL3Bridge(bridgeBytes)
	if bridge == nil {
		log.Fatal("Failed to allocate L3 bridge — cudaHostAlloc failed")
	}

	off := 0
	alloc := func(count int) ([]float32, int) {
		s := bridge.Float32(off*4, count)
		ptr := off * 4
		off += count
		return s, ptr
	}

	hidden, _ := alloc(hiddenSz)
	buf, _ := alloc(hiddenSz)
	qBuf, _ := alloc(qSz)
	kBuf, _ := alloc(kSz)
	vBuf, _ := alloc(vSz)
	attnOut, _ := alloc(attnSz)
	ffnGBuf, _ := alloc(ffnGSz)
	ffnUBuf, _ := alloc(ffnUSz)
	logitsBuf, _ := alloc(logitsSz)
	_, _ = alloc(dHiddenSz) // reserved for full backward
	_, _ = alloc(dAttnSz)  // reserved for full backward
	_, _ = alloc(dFFNSz)   // reserved for full backward
	embedGrad, _ := alloc(embedGradSz)

	// Weights on GPU
	type gpuLayer struct {
		wq, wk, wv, wo     *mongoose.Tensor
		gate, up, down      *mongoose.Tensor
		norm1W, norm2W      []float32 // norms stay on CPU — tiny vectors
	}

	kaiming := func(rows, cols int) []float32 {
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d {
			d[i] = bound * (2*rand.Float32() - 1)
		}
		return d
	}

	embedW := make([]float32, vocabSize*dim)
	for i := range embedW {
		embedW[i] = float32(rand.NormFloat64()) * 0.02
	}
	embedT := te.FromHost(embedW, []int{vocabSize, dim})
	finalNormW := make([]float32, dim)
	for i := range finalNormW {
		finalNormW[i] = 1.0
	}

	gpuLayers := make([]gpuLayer, nLayers)
	for l := range gpuLayers {
		norm1 := make([]float32, dim)
		norm2 := make([]float32, dim)
		for i := range norm1 {
			norm1[i] = 1.0
			norm2[i] = 1.0
		}
		gpuLayers[l] = gpuLayer{
			wq:     te.FromHost(kaiming(dim, dim), []int{dim, dim}),
			wk:     te.FromHost(kaiming(kvDim, dim), []int{kvDim, dim}),
			wv:     te.FromHost(kaiming(kvDim, dim), []int{kvDim, dim}),
			wo:     te.FromHost(kaiming(dim, dim), []int{dim, dim}),
			gate:   te.FromHost(kaiming(ffnDim, dim), []int{ffnDim, dim}),
			up:     te.FromHost(kaiming(ffnDim, dim), []int{ffnDim, dim}),
			down:   te.FromHost(kaiming(dim, ffnDim), []int{dim, ffnDim}),
			norm1W: norm1,
			norm2W: norm2,
		}
	}

	// Adam state for embedding (only param we update in this baseline)
	embedM := make([]float32, vocabSize*dim)
	embedV := make([]float32, vocabSize*dim)

	nParams := vocabSize*dim + dim
	for range nLayers {
		nParams += dim + dim*dim + kvDim*dim + kvDim*dim + dim*dim +
			dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Println("tesseract train-cuda — L3 descriptor path (cuBLAS + pinned memory)")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  L3:       %d MB pinned\n", bridgeBytes/(1024*1024))
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
	normEps := float32(1e-6)

	rmsNormInPlace := func(x, weight []float32) {
		sz := len(x)
		var ss float32
		for i := 0; i < sz; i++ {
			ss += x[i] * x[i]
		}
		ss = float32(1.0 / math.Sqrt(float64(ss)/float64(sz)+float64(normEps)))
		for i := 0; i < sz; i++ {
			x[i] = x[i] * ss * weight[i]
		}
	}

	fmt.Println("Training...")
	t0 := time.Now()

	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)
		tokens := data[start : start+n]

		// === FORWARD (CPU element-wise on L3, GPU matmul via cuBLAS) ===

		// Embedding lookup — CPU writes hidden in L3
		eHost := te.ToHost(embedT)
		for i, t := range tokens {
			copy(hidden[i*dim:(i+1)*dim], eHost[t*dim:(t+1)*dim])
		}

		// Per-layer cache for backward
		type layerCache struct {
			xIn    []float32
			normed []float32
		}
		caches := make([]layerCache, nLayers)

		for li := range gpuLayers {
			gl := &gpuLayers[li]
			lc := &caches[li]

			// Cache input for residual backward
			lc.xIn = make([]float32, n*dim)
			copy(lc.xIn, hidden)

			// RMSNorm1 — CPU on L3
			copy(buf, hidden)
			for pos := 0; pos < n; pos++ {
				rmsNormInPlace(buf[pos*dim:(pos+1)*dim], gl.norm1W)
			}
			lc.normed = make([]float32, n*dim)
			copy(lc.normed, buf)

			// QKV — GPU matmul, input from L3 (buf), output to L3 (qBuf/kBuf/vBuf)
			bufT := te.FromHost(buf, []int{n, dim})
			Q := te.MatMulTransposeBT(bufT, gl.wq, n, dim, dim)
			K := te.MatMulTransposeBT(bufT, gl.wk, n, dim, kvDim)
			V := te.MatMulTransposeBT(bufT, gl.wv, n, dim, kvDim)
			copy(qBuf, te.ToHost(Q))
			copy(kBuf, te.ToHost(K))
			copy(vBuf, te.ToHost(V))
			te.Release(bufT)
			te.Release(Q)
			te.Release(K)
			te.Release(V)

			// RoPE — CPU on L3
			for pos := 0; pos < n; pos++ {
				applyRoPESingle(qBuf[pos*dim:(pos+1)*dim], pos, headDim, heads, 10000.0)
				applyRoPESingle(kBuf[pos*kvDim:(pos+1)*kvDim], pos, headDim, kvHeads, 10000.0)
			}

			// Causal attention — CPU on L3
			for i := range attnOut[:n*dim] {
				attnOut[i] = 0
			}
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
							dot += qBuf[qOff+j] * kBuf[t*kvDim+kvH*headDim+j]
						}
						scores[t] = dot * scale
					}
					softmax(scores, len(scores))
					for t := 0; t <= pos; t++ {
						for j := 0; j < headDim; j++ {
							attnOut[pos*dim+h*headDim+j] += scores[t] * vBuf[t*kvDim+kvH*headDim+j]
						}
					}
				}
			}

			// O projection — GPU matmul
			aT := te.FromHost(attnOut[:n*dim], []int{n, dim})
			proj := te.MatMulTransposeBT(aT, gl.wo, n, dim, dim)
			projH := te.ToHost(proj)
			te.Release(aT)
			te.Release(proj)

			// Residual — CPU on L3
			copy(hidden, lc.xIn)
			for i := 0; i < n*dim; i++ {
				hidden[i] += projH[i]
			}

			// RMSNorm2 — CPU on L3
			copy(buf, hidden)
			for pos := 0; pos < n; pos++ {
				rmsNormInPlace(buf[pos*dim:(pos+1)*dim], gl.norm2W)
			}

			// FFN — GPU matmuls
			bufT2 := te.FromHost(buf[:n*dim], []int{n, dim})
			G := te.MatMulTransposeBT(bufT2, gl.gate, n, dim, ffnDim)
			U := te.MatMulTransposeBT(bufT2, gl.up, n, dim, ffnDim)
			copy(ffnGBuf, te.ToHost(G))
			copy(ffnUBuf, te.ToHost(U))
			te.Release(bufT2)
			te.Release(G)
			te.Release(U)

			// SiLU gate — CPU on L3
			for i := 0; i < n*ffnDim; i++ {
				ffnGBuf[i] = silu(ffnGBuf[i]) * ffnUBuf[i]
			}

			// Down projection — GPU matmul
			ffnT := te.FromHost(ffnGBuf[:n*ffnDim], []int{n, ffnDim})
			D := te.MatMulTransposeBT(ffnT, gl.down, n, ffnDim, dim)
			downH := te.ToHost(D)
			te.Release(ffnT)
			te.Release(D)

			// Residual — CPU on L3
			for i := 0; i < n*dim; i++ {
				hidden[i] += downH[i]
			}
		}

		// Final RMSNorm — CPU on L3
		for pos := 0; pos < n; pos++ {
			rmsNormInPlace(hidden[pos*dim:(pos+1)*dim], finalNormW)
		}

		// LM head (tied embeddings) — GPU matmul
		hT := te.FromHost(hidden[:n*dim], []int{n, dim})
		logitsT := te.MatMulTransposeBT(hT, embedT, n, dim, vocabSize)
		copy(logitsBuf, te.ToHost(logitsT))
		te.Release(hT)
		te.Release(logitsT)

		// === LOSS + BACKWARD (embedding gradient only for baseline) ===

		var loss float32
		for i := range embedGrad {
			embedGrad[i] = 0
		}

		for pos := 0; pos < n-1; pos++ {
			target := tokens[pos+1]
			logits := logitsBuf[pos*vocabSize : (pos+1)*vocabSize]

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

			// Softmax gradient
			scale := 1.0 / float32(n-1)
			for i := range logits {
				logits[i] /= sumExp
			}
			logits[target] -= 1.0

			// Accumulate embedding gradient
			for i := 0; i < vocabSize; i++ {
				grad := logits[i] * scale
				for j := 0; j < dim; j++ {
					embedGrad[i*dim+j] += grad * hidden[pos*dim+j]
				}
			}
		}
		loss /= float32(n - 1)

		// AdamW on embeddings
		bc1 := float32(1.0 - math.Pow(0.9, float64(step)))
		bc2 := float32(1.0 - math.Pow(0.999, float64(step)))
		for i := range embedW {
			g := embedGrad[i]
			embedM[i] = 0.9*embedM[i] + 0.1*g
			embedV[i] = 0.999*embedV[i] + 0.001*g*g
			mh := embedM[i] / bc1
			vh := embedV[i] / bc2
			embedW[i] -= lr * (mh/(float32(math.Sqrt(float64(vh)))+1e-8) + 0.01*embedW[i])
		}

		// Re-upload embeddings
		te.Release(embedT)
		embedT = te.FromHost(embedW, []int{vocabSize, dim})

		if step%*logEvery == 0 || step == 1 {
			cuda.Sync()
			elapsed := time.Since(t0)
			stepsPerSec := float64(step) / elapsed.Seconds()
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, loss, lr, elapsed.Seconds(), stepsPerSec)
		}
	}

	cuda.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)\n", *stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds())
}

func applyRoPESingle(x []float32, pos, headDim, numHeads int, theta float64) {
	half := headDim / 2
	for h := 0; h < numHeads; h++ {
		base := h * headDim
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq
			cos := float32(math.Cos(angle))
			sin := float32(math.Sin(angle))
			x0 := x[base+i]
			x1 := x[base+half+i]
			x[base+i] = x0*cos - x1*sin
			x[base+half+i] = x0*sin + x1*cos
		}
	}
}
