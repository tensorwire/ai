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
	if te == nil { log.Fatal("TensorEngine not available") }
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok { log.Fatal("train-cuda requires CUDA") }

	kernelsAvail := mongoose.LoadKernels()
	dim := *dimFlag
	useGPUKernels := kernelsAvail && dim >= 512
	if kernelsAvail {
		if useGPUKernels {
			log.Printf("[tesseract] CUDA kernels loaded — GPU kernels ACTIVE (dim=%d >= 512)", dim)
		} else {
			log.Printf("[tesseract] CUDA kernels loaded — cuBLAS path (dim=%d < 512, faster)", dim)
		}
	} else {
		log.Println("[tesseract] CUDA kernels not found — cuBLAS only")
	}
	_ = useGPUKernels
	heads := *headsFlag
	kvHeads := *kvHeadsFlag
	headDim := dim / heads
	kvDim := kvHeads * headDim
	nLayers := *layersFlag
	ffnDim := *ffnDimFlag
	seqLen := *seqLenFlag
	vocabSize := 256
	lr := float32(*lrFlag)
	n := seqLen

	raw, err := os.ReadFile(*dataPath)
	if err != nil { log.Fatalf("read data: %v", err) }
	data := make([]int, len(raw))
	for i, b := range raw { data[i] = int(b) }

	conductor := mongoose.NewConductor(vocabSize, 100)

	// RoPE tables on GPU
	halfHead := headDim / 2
	cosTab := make([]float32, seqLen*halfHead)
	sinTab := make([]float32, seqLen*halfHead)
	for pos := 0; pos < seqLen; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(10000.0, float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			cosTab[pos*halfHead+j] = float32(math.Cos(angle))
			sinTab[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}
	ropeCos := te.FromHost(cosTab, []int{seqLen, halfHead})
	ropeSin := te.FromHost(sinTab, []int{seqLen, halfHead})

	// Weights on GPU — allocated once, never released
	kaiming := func(rows, cols int) *mongoose.Tensor {
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d { d[i] = bound * (2*rand.Float32() - 1) }
		return te.FromHost(d, []int{rows, cols})
	}
	ones := func(sz int) *mongoose.Tensor {
		d := make([]float32, sz)
		for i := range d { d[i] = 1.0 }
		return te.FromHost(d, []int{1, sz})
	}

	embedData := make([]float32, vocabSize*dim)
	for i := range embedData { embedData[i] = float32(rand.NormFloat64()) * 0.02 }
	embed := te.FromHost(embedData, []int{vocabSize, dim})
	finalNorm := ones(dim)

	type layer struct{ wq, wk, wv, wo, gate, up, down, norm1, norm2 *mongoose.Tensor }
	lays := make([]layer, nLayers)
	for l := range lays {
		lays[l] = layer{
			wq: kaiming(dim, dim), wk: kaiming(kvDim, dim), wv: kaiming(kvDim, dim),
			wo: kaiming(dim, dim), gate: kaiming(ffnDim, dim), up: kaiming(ffnDim, dim),
			down: kaiming(dim, ffnDim), norm1: ones(dim), norm2: ones(dim),
		}
	}

	// Adam state — allocated once
	type as struct{ m, v *mongoose.Tensor }
	newAS := func(sz int) as { return as{te.Zeros([]int{sz}), te.Zeros([]int{sz})} }
	embedAS := newAS(vocabSize * dim)
	type layerAS struct{ wq, wk, wv, wo, gate, up, down as }
	layAS := make([]layerAS, nLayers)
	for l := range layAS {
		layAS[l] = layerAS{
			wq: newAS(dim * dim), wk: newAS(kvDim * dim), wv: newAS(kvDim * dim),
			wo: newAS(dim * dim), gate: newAS(ffnDim * dim), up: newAS(ffnDim * dim),
			down: newAS(dim * ffnDim),
		}
	}

	// === PRE-ALLOCATE ALL WORKSPACE — never allocate in the loop ===
	hidden := te.Zeros([]int{n, dim})
	tokGPU := te.Zeros([]int{n})
	targetsGPU := te.Zeros([]int{n})
	logitsBuf := te.Zeros([]int{n, vocabSize})
	lossesGPU := te.Zeros([]int{n})
	gradGPU := te.Zeros([]int{n, vocabSize})
	normedFinal := te.Zeros([]int{n, dim})
	finalScales := te.Zeros([]int{n})
	dEmbed := te.Zeros([]int{vocabSize, dim})
	dHidden := te.Zeros([]int{n, dim})

	type fwdBuf struct {
		xIn, normed, Q, K, V, attnOut          *mongoose.Tensor
		xMid, normed2, gatePre, upOut, ffnMid  *mongoose.Tensor
		rmsScale1, rmsScale2                   *mongoose.Tensor
		// backward scratch
		dFfnMid, dGate, dUp, dN2, dx           *mongoose.Tensor
		dAttnOut, dQ, dK, dV, dN1              *mongoose.Tensor
		dWDown, dWGate, dWUp, dWO, dWQ, dWK, dWV *mongoose.Tensor
	}
	bufs := make([]fwdBuf, nLayers)
	for i := range bufs {
		bufs[i] = fwdBuf{
			xIn: te.Zeros([]int{n, dim}), normed: te.Zeros([]int{n, dim}),
			Q: te.Zeros([]int{n, dim}), K: te.Zeros([]int{n, kvDim}), V: te.Zeros([]int{n, kvDim}),
			attnOut: te.Zeros([]int{n, dim}),
			xMid: te.Zeros([]int{n, dim}), normed2: te.Zeros([]int{n, dim}),
			gatePre: te.Zeros([]int{n, ffnDim}), upOut: te.Zeros([]int{n, ffnDim}),
			ffnMid: te.Zeros([]int{n, ffnDim}),
			rmsScale1: te.Zeros([]int{n}), rmsScale2: te.Zeros([]int{n}),
			dFfnMid: te.Zeros([]int{n, ffnDim}), dGate: te.Zeros([]int{n, ffnDim}),
			dUp: te.Zeros([]int{n, ffnDim}), dN2: te.Zeros([]int{n, dim}),
			dx: te.Zeros([]int{n, dim}),
			dAttnOut: te.Zeros([]int{n, dim}), dQ: te.Zeros([]int{n, dim}),
			dK: te.Zeros([]int{n, kvDim}), dV: te.Zeros([]int{n, kvDim}),
			dN1: te.Zeros([]int{n, dim}),
			dWDown: te.Zeros([]int{ffnDim, dim}), dWGate: te.Zeros([]int{dim, ffnDim}),
			dWUp: te.Zeros([]int{dim, ffnDim}), dWO: te.Zeros([]int{dim, dim}),
			dWQ: te.Zeros([]int{dim, dim}), dWK: te.Zeros([]int{dim, kvDim}),
			dWV: te.Zeros([]int{dim, kvDim}),
		}
	}

	nParams := vocabSize*dim + dim
	for range lays {
		nParams += dim + dim*dim + kvDim*dim*2 + dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Println("tesseract train-cuda — zero-alloc GPU kernels + AdamW")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  data:     %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  model:    dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	if nParams > 1e6 { fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6) } else {
		fmt.Printf("  params:   %.1fK\n", float64(nParams)/1e3)
	}
	fmt.Printf("  training: steps=%d lr=%.0e\n", *stepsFlag, *lrFlag)
	fmt.Println()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	adamW := func(param, grad, mS, vS *mongoose.Tensor, step int) {
		mongoose.KAdamW(param.DevicePtr(), grad.DevicePtr(), mS.DevicePtr(), vS.DevicePtr(),
			lr, 0.01, step, param.Size)
	}

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	// L3 bridge for N+1 pipeline — CPU prepares next step while GPU runs current
	l3Size := n * 4 * 2 // tokens + targets, double-buffered
	l3 := cuda.AllocL3Bridge(l3Size)
	var nextTokL3, nextTargL3 []float32
	if l3 != nil {
		nextTokL3 = l3.Float32(0, n)
		nextTargL3 = l3.Float32(n*4, n)
	}

	// Pre-stage first batch
	prepBatch := func(start int) {
		if l3 == nil { return }
		for i := 0; i < n; i++ {
			nextTokL3[i] = math.Float32frombits(uint32(int32(data[start+i])))
			nextTargL3[i] = math.Float32frombits(uint32(int32(data[start+i+1])))
		}
	}

	// Prepare step 1 tokens
	start0 := rng.Intn(len(data) - n - 1)
	prepBatch(start0)

	fmt.Println("Training...")
	t0 := time.Now()

	for step := 1; step <= *stepsFlag; step++ {
		// Current step uses tokens already in L3 (or uploaded)
		if l3 != nil {
			// Upload from L3 pinned memory — GPU reads from cache
			tmpTok := te.FromHost(nextTokL3, []int{n})
			cuda.CopyInto(tokGPU, tmpTok)
			te.Release(tmpTok)
			tmpTarg := te.FromHost(nextTargL3, []int{n})
			cuda.CopyInto(targetsGPU, tmpTarg)
			te.Release(tmpTarg)
		} else {
			start := rng.Intn(len(data) - n - 1)
			tokF := make([]float32, n)
			for i := 0; i < n; i++ { tokF[i] = math.Float32frombits(uint32(int32(data[start+i]))) }
			tmpTok := te.FromHost(tokF, []int{n})
			cuda.CopyInto(tokGPU, tmpTok)
			te.Release(tmpTok)
			targF := make([]float32, n)
			for i := 0; i < n; i++ { targF[i] = math.Float32frombits(uint32(int32(data[start+i+1]))) }
			tmpTarg := te.FromHost(targF, []int{n})
			cuda.CopyInto(targetsGPU, tmpTarg)
			te.Release(tmpTarg)
		}

		// N+1: start preparing next batch in goroutine while GPU runs
		nextStart := rng.Intn(len(data) - n - 1)
		go prepBatch(nextStart)

		// Conductor tracks hot tokens from current step
		tokIDs := make([]int32, n)
		for i := 0; i < n; i++ { tokIDs[i] = int32(data[nextStart+i]) }
		conductor.Observe(tokIDs)

		// === FORWARD ===
		// Embedding gather
		if useGPUKernels {
			zero(hidden)
			mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)
		} else {
			eH := te.ToHost(embed)
			hH := make([]float32, n*dim)
			for i, t := range tokIDs { copy(hH[i*dim:(i+1)*dim], eH[int(t)*dim:int(t)*dim+dim]) }
			tmpH := te.FromHost(hH, []int{n, dim})
			cuda.CopyInto(hidden, tmpH); te.Release(tmpH)
		}

		for li := range lays {
			l := &lays[li]
			b := &bufs[li]

			cuda.CopyInto(b.xIn, hidden)

			// RMSNorm1
			if useGPUKernels {
				zero(b.normed); zero(b.rmsScale1)
				mongoose.KRMSNormOutSave(hidden.DevicePtr(), b.normed.DevicePtr(),
					l.norm1.DevicePtr(), b.rmsScale1.DevicePtr(), n, dim)
			} else {
				cuda.CopyInto(b.normed, hidden)
				nH := te.ToHost(b.normed); wH := te.ToHost(l.norm1)
				for pos := 0; pos < n; pos++ { eng.RMSNorm(nH[pos*dim:(pos+1)*dim], wH[:dim], 1e-6) }
				tmpN := te.FromHost(nH, []int{n, dim})
				cuda.CopyInto(b.normed, tmpN); te.Release(tmpN)
			}

			cuda.MatMulTransposeBTInto(b.Q, b.normed, l.wq, n, dim, dim)
			cuda.MatMulTransposeBTInto(b.K, b.normed, l.wk, n, dim, kvDim)
			cuda.MatMulTransposeBTInto(b.V, b.normed, l.wv, n, dim, kvDim)

			// RoPE
			if useGPUKernels {
				mongoose.KRoPE(b.Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
				mongoose.KRoPE(b.K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)
			} else {
				qH := te.ToHost(b.Q); kH := te.ToHost(b.K)
				for pos := 0; pos < n; pos++ {
					applyRoPESingle(qH[pos*dim:(pos+1)*dim], pos, headDim, heads, 10000.0)
					applyRoPESingle(kH[pos*kvDim:(pos+1)*kvDim], pos, headDim, kvHeads, 10000.0)
				}
				tmpQ := te.FromHost(qH, []int{n, dim}); cuda.CopyInto(b.Q, tmpQ); te.Release(tmpQ)
				tmpK := te.FromHost(kH, []int{n, kvDim}); cuda.CopyInto(b.K, tmpK); te.Release(tmpK)
			}

			// Attention
			if useGPUKernels {
				zero(b.attnOut)
				mongoose.KCausalAttentionGQA(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.attnOut.DevicePtr(),
					n, dim, kvDim, heads, kvHeads)
			} else {
				qH := te.ToHost(b.Q); kH := te.ToHost(b.K); vH := te.ToHost(b.V)
				aH := make([]float32, n*dim)
				kvMul := heads / kvHeads
				for pos := 0; pos < n; pos++ {
					for h := 0; h < heads; h++ {
						qOff := pos*dim + h*headDim; kvH := h / kvMul
						scale := float32(1.0 / math.Sqrt(float64(headDim)))
						scores := make([]float32, pos+1)
						for t := 0; t <= pos; t++ {
							var dot float32
							for j := 0; j < headDim; j++ { dot += qH[qOff+j] * kH[t*kvDim+kvH*headDim+j] }
							scores[t] = dot * scale
						}
						softmax(scores, len(scores))
						for t := 0; t <= pos; t++ {
							for j := 0; j < headDim; j++ { aH[pos*dim+h*headDim+j] += scores[t] * vH[t*kvDim+kvH*headDim+j] }
						}
					}
				}
				tmpA := te.FromHost(aH, []int{n, dim}); cuda.CopyInto(b.attnOut, tmpA); te.Release(tmpA)
			}

			cuda.MatMulTransposeBTInto(b.dx, b.attnOut, l.wo, n, dim, dim)
			te.AddInPlace(hidden, b.dx)

			cuda.CopyInto(b.xMid, hidden)

			// RMSNorm2
			if useGPUKernels {
				zero(b.normed2); zero(b.rmsScale2)
				mongoose.KRMSNormOutSave(hidden.DevicePtr(), b.normed2.DevicePtr(),
					l.norm2.DevicePtr(), b.rmsScale2.DevicePtr(), n, dim)
			} else {
				cuda.CopyInto(b.normed2, hidden)
				nH := te.ToHost(b.normed2); wH := te.ToHost(l.norm2)
				for pos := 0; pos < n; pos++ { eng.RMSNorm(nH[pos*dim:(pos+1)*dim], wH[:dim], 1e-6) }
				tmpN := te.FromHost(nH, []int{n, dim})
				cuda.CopyInto(b.normed2, tmpN); te.Release(tmpN)
			}

			cuda.MatMulTransposeBTInto(b.gatePre, b.normed2, l.gate, n, dim, ffnDim)
			cuda.MatMulTransposeBTInto(b.upOut, b.normed2, l.up, n, dim, ffnDim)

			// SiLU gate
			if useGPUKernels {
				zero(b.ffnMid)
				mongoose.KSiLUGateMul(b.gatePre.DevicePtr(), b.upOut.DevicePtr(), b.ffnMid.DevicePtr(), n*ffnDim)
			} else {
				gH := te.ToHost(b.gatePre); uH := te.ToHost(b.upOut)
				for i := range gH { gH[i] = silu(gH[i]) * uH[i] }
				tmpF := te.FromHost(gH, []int{n, ffnDim}); cuda.CopyInto(b.ffnMid, tmpF); te.Release(tmpF)
			}

			cuda.MatMulTransposeBTInto(b.dx, b.ffnMid, l.down, n, ffnDim, dim)
			te.AddInPlace(hidden, b.dx)
		}

		// Final RMSNorm
		if useGPUKernels {
			zero(normedFinal); zero(finalScales)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
				finalNorm.DevicePtr(), finalScales.DevicePtr(), n, dim)
		} else {
			cuda.CopyInto(normedFinal, hidden)
			nH := te.ToHost(normedFinal); wH := te.ToHost(finalNorm)
			for pos := 0; pos < n; pos++ { eng.RMSNorm(nH[pos*dim:(pos+1)*dim], wH[:dim], 1e-6) }
			tmpN := te.FromHost(nH, []int{n, dim})
			cuda.CopyInto(normedFinal, tmpN); te.Release(tmpN)
		}

		cuda.MatMulTransposeBTInto(logitsBuf, normedFinal, embed, n, dim, vocabSize)

		// Loss + gradient
		if useGPUKernels {
			zero(lossesGPU); zero(gradGPU)
			invN := float32(1.0) / float32(n)
			mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
				lossesGPU.DevicePtr(), gradGPU.DevicePtr(), n, vocabSize, invN)
		} else {
			cuda.Sync()
			logH := te.ToHost(logitsBuf); tH := te.ToHost(targetsGPU)
			lH := make([]float32, n); gH := make([]float32, n*vocabSize)
			invN := float32(1.0) / float32(n)
			for pos := 0; pos < n; pos++ {
				target := int(math.Float32bits(tH[pos]))
				off := pos * vocabSize
				maxL := logH[off]
				for i := 1; i < vocabSize; i++ { if logH[off+i] > maxL { maxL = logH[off+i] } }
				var sumExp float32
				for i := 0; i < vocabSize; i++ {
					logH[off+i] = float32(math.Exp(float64(logH[off+i] - maxL)))
					sumExp += logH[off+i]
				}
				lH[pos] = -float32(math.Log(float64(logH[off+target]/sumExp) + 1e-10))
				for i := 0; i < vocabSize; i++ {
					gH[off+i] = (logH[off+i] / sumExp) * invN
				}
				gH[off+target] -= invN
			}
			tmpL := te.FromHost(lH, []int{n}); cuda.CopyInto(lossesGPU, tmpL); te.Release(tmpL)
			tmpG := te.FromHost(gH, []int{n, vocabSize}); cuda.CopyInto(gradGPU, tmpG); te.Release(tmpG)
		}

		// === BACKWARD (loss read deferred to log steps) ===
		// dEmbed = gradGPU^T @ normedFinal → [vocabSize, dim]
		cuda.MatMulTransposeATInto(dEmbed, gradGPU, normedFinal, vocabSize, n, dim)

		// dHidden = gradGPU @ embed
		cuda.MatMulTInto(dHidden, gradGPU, embed, n, vocabSize, dim)

		// Final RMSNorm backward
		if useGPUKernels {
			dScratch := te.Zeros([]int{n, dim})
			mongoose.KRMSNormBackward(dHidden.DevicePtr(), hidden.DevicePtr(),
				finalNorm.DevicePtr(), finalScales.DevicePtr(), dScratch.DevicePtr(), n, dim)
			cuda.CopyInto(dHidden, dScratch)
			te.Release(dScratch)
		} else {
			// CPU: dHidden *= weight * rmsScale (simplified — skip norm backward for now, pass through)
			// The proper RMSNorm backward needs saved rmsScale which we don't have in CPU path
			// dHidden passes through unchanged — the matmul gradients still flow
		}

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]
			b := &bufs[li]

			// FFN backward: dFfnMid = dHidden @ down
			cuda.MatMulTInto(b.dFfnMid, dHidden, l.down, n, dim, ffnDim)

			cuda.MatMulTransposeATInto(b.dWDown, b.ffnMid, dHidden, ffnDim, n, dim)
			adamW(l.down, b.dWDown, layAS[li].down.m, layAS[li].down.v, step)

			// SiLU gate backward
			if useGPUKernels {
				zero(b.dGate); zero(b.dUp)
				mongoose.KSiLUGateBackward(b.dFfnMid.DevicePtr(), b.gatePre.DevicePtr(),
					b.upOut.DevicePtr(), b.dGate.DevicePtr(), b.dUp.DevicePtr(), n*ffnDim)
			} else {
				cuda.Sync()
				dfH := te.ToHost(b.dFfnMid); gH := te.ToHost(b.gatePre); uH := te.ToHost(b.upOut)
				dgH := make([]float32, n*ffnDim); duH := make([]float32, n*ffnDim)
				for i := range dfH {
					s := silu(gH[i])
					dgH[i] = dfH[i] * uH[i] * (s + gH[i]*s*(1-s))
					duH[i] = dfH[i] * s
				}
				tmpDG := te.FromHost(dgH, []int{n, ffnDim}); cuda.CopyInto(b.dGate, tmpDG); te.Release(tmpDG)
				tmpDU := te.FromHost(duH, []int{n, ffnDim}); cuda.CopyInto(b.dUp, tmpDU); te.Release(tmpDU)
			}

			cuda.MatMulTInto(b.dN2, b.dGate, l.gate, n, ffnDim, dim)
			cuda.MatMulTInto(b.dx, b.dUp, l.up, n, ffnDim, dim)
			te.AddInPlace(b.dN2, b.dx)

			cuda.MatMulTransposeATInto(b.dWGate, b.normed2, b.dGate, dim, n, ffnDim)
			cuda.MatMulTransposeATInto(b.dWUp, b.normed2, b.dUp, dim, n, ffnDim)
			adamW(l.gate, b.dWGate, layAS[li].gate.m, layAS[li].gate.v, step)
			adamW(l.up, b.dWUp, layAS[li].up.m, layAS[li].up.v, step)

			// RMSNorm2 backward
			if useGPUKernels {
				zero(b.dx)
				mongoose.KRMSNormBackward(b.dN2.DevicePtr(), b.xMid.DevicePtr(),
					l.norm2.DevicePtr(), b.rmsScale2.DevicePtr(), b.dx.DevicePtr(), n, dim)
				te.AddInPlace(dHidden, b.dx)
			} else {
				te.AddInPlace(dHidden, b.dN2)
			}

			// Attention backward
			cuda.MatMulTInto(b.dAttnOut, dHidden, l.wo, n, dim, dim)
			cuda.MatMulTransposeATInto(b.dWO, b.attnOut, dHidden, dim, n, dim)
			adamW(l.wo, b.dWO, layAS[li].wo.m, layAS[li].wo.v, step)

			if useGPUKernels {
				zero(b.dQ); zero(b.dK); zero(b.dV)
				mongoose.KCausalAttentionBackward(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.dAttnOut.DevicePtr(),
					b.dQ.DevicePtr(), b.dK.DevicePtr(), b.dV.DevicePtr(), n, dim, kvDim, heads, kvHeads)
			} else {
				// CPU attention backward
				cuda.Sync()
				qH := te.ToHost(b.Q); kH := te.ToHost(b.K); vH := te.ToHost(b.V)
				daoH := te.ToHost(b.dAttnOut)
				dqH := make([]float32, n*dim); dkH := make([]float32, n*kvDim); dvH := make([]float32, n*kvDim)
				kvMul := heads / kvHeads
				for pos := 0; pos < n; pos++ {
					for h := 0; h < heads; h++ {
						qOff := pos*dim + h*headDim; kvHH := h / kvMul
						scale := float32(1.0 / math.Sqrt(float64(headDim)))
						scores := make([]float32, pos+1)
						for t := 0; t <= pos; t++ {
							var dot float32
							for j := 0; j < headDim; j++ { dot += qH[qOff+j] * kH[t*kvDim+kvHH*headDim+j] }
							scores[t] = dot * scale
						}
						softmax(scores, len(scores))
						for t := 0; t <= pos; t++ {
							var dScore float32
							for j := 0; j < headDim; j++ {
								dScore += daoH[pos*dim+h*headDim+j] * vH[t*kvDim+kvHH*headDim+j]
								dvH[t*kvDim+kvHH*headDim+j] += scores[t] * daoH[pos*dim+h*headDim+j]
							}
							for t2 := 0; t2 <= pos; t2++ {
								ds := scores[t] * (float32(0) - scores[t2]) * scale
								if t == t2 { ds = scores[t] * (1 - scores[t]) * scale }
								for j := 0; j < headDim; j++ {
									dqH[qOff+j] += ds * dScore * kH[t2*kvDim+kvHH*headDim+j]
									dkH[t2*kvDim+kvHH*headDim+j] += ds * dScore * qH[qOff+j]
								}
							}
						}
					}
				}
				tmpDQ := te.FromHost(dqH, []int{n, dim}); cuda.CopyInto(b.dQ, tmpDQ); te.Release(tmpDQ)
				tmpDK := te.FromHost(dkH, []int{n, kvDim}); cuda.CopyInto(b.dK, tmpDK); te.Release(tmpDK)
				tmpDV := te.FromHost(dvH, []int{n, kvDim}); cuda.CopyInto(b.dV, tmpDV); te.Release(tmpDV)
			}

			// RoPE backward
			if useGPUKernels {
				mongoose.KRoPEBackward(b.dQ.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
				mongoose.KRoPEBackward(b.dK.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)
			} else {
				cuda.Sync()
				dqH := te.ToHost(b.dQ); dkH := te.ToHost(b.dK)
				for pos := 0; pos < n; pos++ {
					applyRoPESingle(dqH[pos*dim:(pos+1)*dim], pos, headDim, heads, 10000.0)
					applyRoPESingle(dkH[pos*kvDim:(pos+1)*kvDim], pos, headDim, kvHeads, 10000.0)
				}
				tmpDQ := te.FromHost(dqH, []int{n, dim}); cuda.CopyInto(b.dQ, tmpDQ); te.Release(tmpDQ)
				tmpDK := te.FromHost(dkH, []int{n, kvDim}); cuda.CopyInto(b.dK, tmpDK); te.Release(tmpDK)
			}

			cuda.MatMulTInto(b.dN1, b.dQ, l.wq, n, dim, dim)
			cuda.MatMulTInto(b.dx, b.dK, l.wk, n, kvDim, dim)
			cuda.MatMulTInto(b.dN2, b.dV, l.wv, n, kvDim, dim)
			te.AddInPlace(b.dN1, b.dx); te.AddInPlace(b.dN1, b.dN2)

			cuda.MatMulTransposeATInto(b.dWQ, b.normed, b.dQ, dim, n, dim)
			cuda.MatMulTransposeATInto(b.dWK, b.normed, b.dK, dim, n, kvDim)
			cuda.MatMulTransposeATInto(b.dWV, b.normed, b.dV, dim, n, kvDim)
			adamW(l.wq, b.dWQ, layAS[li].wq.m, layAS[li].wq.v, step)
			adamW(l.wk, b.dWK, layAS[li].wk.m, layAS[li].wk.v, step)
			adamW(l.wv, b.dWV, layAS[li].wv.m, layAS[li].wv.v, step)

			// RMSNorm1 backward
			if useGPUKernels {
				zero(b.dx)
				mongoose.KRMSNormBackward(b.dN1.DevicePtr(), b.xIn.DevicePtr(),
					l.norm1.DevicePtr(), b.rmsScale1.DevicePtr(), b.dx.DevicePtr(), n, dim)
				te.AddInPlace(dHidden, b.dx)
			} else {
				te.AddInPlace(dHidden, b.dN1)
			}
		}

		adamW(embed, dEmbed, embedAS.m, embedAS.v, step)
		if step <= 3 || step%*logEvery == 0 {
			cuda.Sync()
			lossH := te.ToHost(lossesGPU)
			var totalLoss float32
			for _, l := range lossH { totalLoss += l }
			totalLoss /= float32(n)
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, totalLoss, lr, elapsed.Seconds(), float64(step)/elapsed.Seconds())
		}
	}

	cuda.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)\n", *stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds())
}
