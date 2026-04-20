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

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/mongoose"
)

func cmdFinetune() {
	fs := flag.NewFlagSet("finetune", flag.ExitOnError)

	modelPath := fs.String("model", "", "Model directory (safetensors)")
	dataPath := fs.String("data", "", "Training data (text file)")
	stepsFlag := fs.Int("steps", 100, "Training steps")
	lrFlag := fs.Float64("lr", 1e-5, "Learning rate")
	logEvery := fs.Int("log-every", 10, "Log every N steps")

	fs.Parse(os.Args[2:])

	if *modelPath == "" {
		home, _ := os.UserHomeDir()
		*modelPath = filepath.Join(home, ".mongoose", "models", "TinyLlama-1.1B-Chat-v1.0")
	}
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
	if !ok { log.Fatal("finetune requires CUDA") }

	if !mongoose.LoadKernels() {
		log.Fatal("CUDA kernels required")
	}

	st, err := gguf.OpenSafeTensors(*modelPath)
	if err != nil { log.Fatalf("open model: %v", err) }

	dim := 2048
	heads := 32
	kvHeads := 4
	headDim := dim / heads
	kvDim := kvHeads * headDim
	nLayers := 22
	ffnDim := 5632
	seqLen := 64
	vocabSize := 32000
	lr := float32(*lrFlag)
	n := seqLen

	log.Printf("[finetune] loading %s", *modelPath)

	embedData, _, err := st.ReadTensorFloat32("model.embed_tokens.weight")
	if err != nil { log.Fatalf("embed: %v", err) }
	embed := te.FromHost(embedData, []int{vocabSize, dim})

	lmHeadData, _, err := st.ReadTensorFloat32("lm_head.weight")
	if err != nil {
		lmHeadData = embedData
	}
	lmHead := te.FromHost(lmHeadData, []int{vocabSize, dim})

	fnData, _, _ := st.ReadTensorFloat32("model.norm.weight")
	if fnData == nil { fnData = make([]float32, dim); for i := range fnData { fnData[i] = 1 } }
	finalNorm := te.FromHost(fnData, []int{1, dim})

	type layer struct{ wq, wk, wv, wo, gate, up, down, norm1, norm2 *mongoose.Tensor }
	lays := make([]layer, nLayers)
	for l := 0; l < nLayers; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		loadW := func(name string, rows, cols int) *mongoose.Tensor {
			d, _, err := st.ReadTensorFloat32(pfx + name)
			if err != nil { log.Fatalf("layer %d %s: %v", l, name, err) }
			return te.FromHost(d, []int{rows, cols})
		}
		loadNorm := func(name string) *mongoose.Tensor {
			d, _, _ := st.ReadTensorFloat32(pfx + name)
			if d == nil { d = make([]float32, dim); for i := range d { d[i] = 1 } }
			return te.FromHost(d, []int{1, dim})
		}
		lays[l] = layer{
			wq:    loadW("self_attn.q_proj.weight", dim, dim),
			wk:    loadW("self_attn.k_proj.weight", kvDim, dim),
			wv:    loadW("self_attn.v_proj.weight", kvDim, dim),
			wo:    loadW("self_attn.o_proj.weight", dim, dim),
			gate:  loadW("mlp.gate_proj.weight", ffnDim, dim),
			up:    loadW("mlp.up_proj.weight", ffnDim, dim),
			down:  loadW("mlp.down_proj.weight", dim, ffnDim),
			norm1: loadNorm("input_layernorm.weight"),
			norm2: loadNorm("post_attention_layernorm.weight"),
		}
		if (l+1)%10 == 0 || l == nLayers-1 {
			log.Printf("[finetune] loaded layer %d/%d", l+1, nLayers)
		}
	}

	type as struct{ m, v *mongoose.Tensor }
	newAS := func(sz int) as { return as{te.Zeros([]int{sz}), te.Zeros([]int{sz})} }
	embedAS := newAS(vocabSize * dim)
	lmHeadAS := newAS(vocabSize * dim)
	type layerAS struct{ wq, wk, wv, wo, gate, up, down as }
	layAS := make([]layerAS, nLayers)
	for l := range layAS {
		layAS[l] = layerAS{
			wq: newAS(dim * dim), wk: newAS(kvDim * dim), wv: newAS(kvDim * dim),
			wo: newAS(dim * dim), gate: newAS(ffnDim * dim), up: newAS(ffnDim * dim),
			down: newAS(dim * ffnDim),
		}
	}

	hidden := te.Zeros([]int{n, dim})
	tokGPU := te.Zeros([]int{n})
	targetsGPU := te.Zeros([]int{n})
	logitsBuf := te.Zeros([]int{n, vocabSize})
	lossesGPU := te.Zeros([]int{n})
	gradGPU := te.Zeros([]int{n, vocabSize})
	normedFinal := te.Zeros([]int{n, dim})
	finalScales := te.Zeros([]int{n})
	dLmHead := te.Zeros([]int{vocabSize, dim})
	dHidden := te.Zeros([]int{n, dim})

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

	type fwdBuf struct {
		xIn, normed, Q, K, V, attnOut          *mongoose.Tensor
		xMid, normed2, gatePre, upOut, ffnMid  *mongoose.Tensor
		rmsScale1, rmsScale2                   *mongoose.Tensor
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
			dWDown: te.Zeros([]int{dim, ffnDim}), dWGate: te.Zeros([]int{ffnDim, dim}),
			dWUp: te.Zeros([]int{ffnDim, dim}), dWO: te.Zeros([]int{dim, dim}),
			dWQ: te.Zeros([]int{dim, dim}), dWK: te.Zeros([]int{kvDim, dim}),
			dWV: te.Zeros([]int{kvDim, dim}),
		}
	}

	nParams := vocabSize*dim*2 + dim
	for range lays {
		nParams += dim + dim*dim + kvDim*dim*2 + dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
	}

	raw, err := os.ReadFile(*dataPath)
	if err != nil { log.Fatalf("read data: %v", err) }
	data := make([]int, len(raw))
	for i, b := range raw { data[i] = int(b) }

	totalSteps := *stepsFlag
	warmupSteps := totalSteps / 10
	if warmupSteps < 5 { warmupSteps = 5 }
	minLR := lr / 10.0

	getLR := func(step int) float32 {
		if step < warmupSteps {
			return lr * float32(step) / float32(warmupSteps)
		}
		progress := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
		cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
		return minLR + float32(cosine)*float32(lr-minLR)
	}

	adamW := func(param, grad, mS, vS *mongoose.Tensor, step int) {
		stepLR := getLR(step)
		mongoose.KAdamW(param.DevicePtr(), grad.DevicePtr(), mS.DevicePtr(), vS.DevicePtr(),
			stepLR, 0.1, step, param.Size)
	}

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	fmt.Println("tesseract finetune — FP32 backward + AdamW")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  model:    %s\n", *modelPath)
	fmt.Printf("  data:     %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  arch:     dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6)
	fmt.Printf("  training: steps=%d lr=%.0e\n", *stepsFlag, *lrFlag)
	fmt.Println()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	fmt.Println("Training...")
	t0 := time.Now()

	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)
		tokF := make([]float32, n)
		targF := make([]float32, n)
		for i := 0; i < n; i++ {
			tokF[i] = math.Float32frombits(uint32(int32(data[start+i])))
			targF[i] = math.Float32frombits(uint32(int32(data[start+i+1])))
		}
		tmpTok := te.FromHost(tokF, []int{n}); cuda.CopyInto(tokGPU, tmpTok); te.Release(tmpTok)
		tmpTarg := te.FromHost(targF, []int{n}); cuda.CopyInto(targetsGPU, tmpTarg); te.Release(tmpTarg)

		// === FORWARD ===
		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		for li := range lays {
			l := &lays[li]
			b := &bufs[li]

			cuda.CopyInto(b.xIn, hidden)
			zero(b.normed); zero(b.rmsScale1)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), b.normed.DevicePtr(),
				l.norm1.DevicePtr(), b.rmsScale1.DevicePtr(), n, dim)

			cuda.MatMulTransposeBTInto(b.Q, b.normed, l.wq, n, dim, dim)
			cuda.MatMulTransposeBTInto(b.K, b.normed, l.wk, n, dim, kvDim)
			cuda.MatMulTransposeBTInto(b.V, b.normed, l.wv, n, dim, kvDim)

			mongoose.KRoPE(b.Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(b.K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			zero(b.attnOut)
			mongoose.KCausalAttentionGQA(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			cuda.MatMulTransposeBTInto(b.dx, b.attnOut, l.wo, n, dim, dim)
			te.AddInPlace(hidden, b.dx)

			cuda.CopyInto(b.xMid, hidden)
			zero(b.normed2); zero(b.rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), b.normed2.DevicePtr(),
				l.norm2.DevicePtr(), b.rmsScale2.DevicePtr(), n, dim)

			cuda.MatMulTransposeBTInto(b.gatePre, b.normed2, l.gate, n, dim, ffnDim)
			cuda.MatMulTransposeBTInto(b.upOut, b.normed2, l.up, n, dim, ffnDim)

			zero(b.ffnMid)
			mongoose.KSiLUGateMul(b.gatePre.DevicePtr(), b.upOut.DevicePtr(), b.ffnMid.DevicePtr(), n*ffnDim)

			cuda.MatMulTransposeBTInto(b.dx, b.ffnMid, l.down, n, ffnDim, dim)
			te.AddInPlace(hidden, b.dx)
		}

		zero(normedFinal); zero(finalScales)
		mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), n, dim)

		// Untied lm_head
		cuda.MatMulTransposeBTInto(logitsBuf, normedFinal, lmHead, n, dim, vocabSize)

		zero(lossesGPU); zero(gradGPU)
		invN := float32(1.0) / float32(n)
		mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
			lossesGPU.DevicePtr(), gradGPU.DevicePtr(), n, vocabSize, invN)

		var stepLoss float32
		if step <= 3 || step%*logEvery == 0 {
			cuda.Sync()
			lossH := te.ToHost(lossesGPU)
			for _, l := range lossH { stepLoss += l }
			stepLoss /= float32(n)
		}

		// === BACKWARD ===
		// lm_head gradient (untied)
		cuda.MatMulTransposeATInto(dLmHead, gradGPU, normedFinal, n, vocabSize, dim)
		cuda.MatMulTInto(dHidden, gradGPU, lmHead, n, vocabSize, dim)

		// Final RMSNorm backward
		dScratch := te.Zeros([]int{n, dim})
		mongoose.KRMSNormBackward(dHidden.DevicePtr(), hidden.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), dScratch.DevicePtr(), n, dim)
		cuda.CopyInto(dHidden, dScratch)
		te.Release(dScratch)

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]
			b := &bufs[li]

			cuda.MatMulTInto(b.dFfnMid, dHidden, l.down, n, dim, ffnDim)
			cuda.MatMulTransposeATInto(b.dWDown, dHidden, b.ffnMid, n, dim, ffnDim)
			adamW(l.down, b.dWDown, layAS[li].down.m, layAS[li].down.v, step)

			zero(b.dGate); zero(b.dUp)
			mongoose.KSiLUGateBackward(b.dFfnMid.DevicePtr(), b.gatePre.DevicePtr(),
				b.upOut.DevicePtr(), b.dGate.DevicePtr(), b.dUp.DevicePtr(), n*ffnDim)

			cuda.MatMulTInto(b.dN2, b.dGate, l.gate, n, ffnDim, dim)
			cuda.MatMulTInto(b.dx, b.dUp, l.up, n, ffnDim, dim)
			te.AddInPlace(b.dN2, b.dx)

			cuda.MatMulTransposeATInto(b.dWGate, b.dGate, b.normed2, n, ffnDim, dim)
			cuda.MatMulTransposeATInto(b.dWUp, b.dUp, b.normed2, n, ffnDim, dim)
			adamW(l.gate, b.dWGate, layAS[li].gate.m, layAS[li].gate.v, step)
			adamW(l.up, b.dWUp, layAS[li].up.m, layAS[li].up.v, step)

			zero(b.dx)
			mongoose.KRMSNormBackward(b.dN2.DevicePtr(), b.xMid.DevicePtr(),
				l.norm2.DevicePtr(), b.rmsScale2.DevicePtr(), b.dx.DevicePtr(), n, dim)
			te.AddInPlace(dHidden, b.dx)

			cuda.MatMulTInto(b.dAttnOut, dHidden, l.wo, n, dim, dim)
			cuda.MatMulTransposeATInto(b.dWO, dHidden, b.attnOut, n, dim, dim)
			adamW(l.wo, b.dWO, layAS[li].wo.m, layAS[li].wo.v, step)

			zero(b.dQ); zero(b.dK); zero(b.dV)
			mongoose.KCausalAttentionBackward(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.dAttnOut.DevicePtr(),
				b.dQ.DevicePtr(), b.dK.DevicePtr(), b.dV.DevicePtr(), n, dim, kvDim, heads, kvHeads)

			mongoose.KRoPEBackward(b.dQ.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPEBackward(b.dK.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			cuda.MatMulTInto(b.dN1, b.dQ, l.wq, n, dim, dim)
			cuda.MatMulTInto(b.dx, b.dK, l.wk, n, kvDim, dim)
			cuda.MatMulTInto(b.dN2, b.dV, l.wv, n, kvDim, dim)
			te.AddInPlace(b.dN1, b.dx); te.AddInPlace(b.dN1, b.dN2)

			cuda.MatMulTransposeATInto(b.dWQ, b.dQ, b.normed, n, dim, dim)
			cuda.MatMulTransposeATInto(b.dWK, b.dK, b.normed, n, kvDim, dim)
			cuda.MatMulTransposeATInto(b.dWV, b.dV, b.normed, n, kvDim, dim)
			adamW(l.wq, b.dWQ, layAS[li].wq.m, layAS[li].wq.v, step)
			adamW(l.wk, b.dWK, layAS[li].wk.m, layAS[li].wk.v, step)
			adamW(l.wv, b.dWV, layAS[li].wv.m, layAS[li].wv.v, step)

			zero(b.dx)
			mongoose.KRMSNormBackward(b.dN1.DevicePtr(), b.xIn.DevicePtr(),
				l.norm1.DevicePtr(), b.rmsScale1.DevicePtr(), b.dx.DevicePtr(), n, dim)
			te.AddInPlace(dHidden, b.dx)
		}

		adamW(lmHead, dLmHead, lmHeadAS.m, lmHeadAS.v, step)
		adamW(embed, dLmHead, embedAS.m, embedAS.v, step)

		if step <= 3 || step%*logEvery == 0 {
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, stepLoss, getLR(step), elapsed.Seconds(), float64(step)/elapsed.Seconds())
		}
	}

	cuda.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)\n", *stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds())
}
