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

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/helix"
	"github.com/tensorwire/mongoose"

	"unsafe"
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
		*modelPath = filepath.Join(home, ".ai", "models", "TinyLlama-1.1B-Chat-v1.0")
	}
	if *dataPath == "" { log.Fatal("data required: ai finetune model=<name> data=<file>") }

	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	if te == nil { log.Fatal("TensorEngine not available") }
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok { log.Fatalf("finetune requires CUDA (detected: %s). On Metal, use: ai train model=<name> data=<file>", eng.Name()) }

	if !mongoose.LoadKernels() {
		log.Fatal("CUDA kernels required")
	}

	ms, err := OpenModel(*modelPath)
	if err != nil { log.Fatalf("open model: %v", err) }

	dim := 2048
	heads := 32
	kvHeads := 4
	nLayers := 22
	ffnDim := 5632
	seqLen := 64
	vocabSize := 32000

	cfgPath := filepath.Join(*modelPath, "config.json")
	if cfgData, err := os.ReadFile(cfgPath); err == nil {
		var cfg map[string]interface{}
		json.Unmarshal(cfgData, &cfg)
		if v, ok := cfg["hidden_size"].(float64); ok { dim = int(v) }
		if v, ok := cfg["num_attention_heads"].(float64); ok { heads = int(v) }
		if v, ok := cfg["num_key_value_heads"].(float64); ok { kvHeads = int(v) }
		if v, ok := cfg["num_hidden_layers"].(float64); ok { nLayers = int(v) }
		if v, ok := cfg["intermediate_size"].(float64); ok { ffnDim = int(v) }
		if v, ok := cfg["vocab_size"].(float64); ok { vocabSize = int(v) }
		if v, ok := cfg["max_position_embeddings"].(float64); ok && int(v) < 2048 { seqLen = int(v) }
		log.Printf("[finetune] config: dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d", dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	}

	headDim := dim / heads
	kvDim := kvHeads * headDim
	lr := float32(*lrFlag)
	n := seqLen

	log.Printf("[finetune] loading %s", *modelPath)

	embedData, err := ms.ReadTensorFloat32("model.embed_tokens.weight")
	if err != nil { log.Fatalf("embed: %v", err) }
	embed := te.FromHost(embedData, []int{vocabSize, dim})

	lmHeadData, err := ms.ReadTensorFloat32("lm_head.weight")
	if err != nil {
		lmHeadData = embedData
	}
	lmHead := te.FromHost(lmHeadData, []int{vocabSize, dim})

	fnData, _ := ms.ReadTensorFloat32("model.norm.weight")
	if fnData == nil { fnData = make([]float32, dim); for i := range fnData { fnData[i] = 1 } }
	finalNorm := te.FromHost(fnData, []int{1, dim})

	// INT8 weight + FP32 dequant cache + needle state
	type needleWeight struct {
		q8    *mongoose.Int8Tensor  // INT8 weights (updated by needle)
		cache *mongoose.Tensor      // FP32 dequant cache (read by matmuls)
		mom   unsafe.Pointer        // FP16 momentum
		vel   unsafe.Pointer        // FP16 delta residual
		mask  unsafe.Pointer        // row mask (all active for full finetune)
	}
	quantLoad := func(data []float32, rows, cols int) needleWeight {
		qt := gguf.QuantizeToInt8(data, rows, cols)
		q8 := cuda.FromHostInt8(&mongoose.QuantizedTensor{
			DataInt8: qt.DataInt8, Scales: qt.Scales, Shape: qt.Shape,
			Rows: qt.Rows, Cols: qt.Cols,
		})
		cache := te.Zeros([]int{rows, cols})
		cuda.DequantToFP32(q8, cache.DevicePtr())
		nEl := rows * cols
		momPtr := cuda.AllocGPU(nEl * 2)
		velPtr := cuda.AllocGPU(nEl * 2)
		maskData := make([]float32, rows)
		for r := range maskData { maskData[r] = float32(r + 1) }
		maskT := te.FromHost(maskData, []int{rows})
		return needleWeight{q8: q8, cache: cache, mom: momPtr, vel: velPtr, mask: maskT.DevicePtr()}
	}

	type layer struct {
		wq, wk, wv, wo, gate, up, down needleWeight
		norm1, norm2                    *mongoose.Tensor
	}
	lays := make([]layer, nLayers)
	for l := 0; l < nLayers; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		loadQ := func(name string, rows, cols int) needleWeight {
			d, err := ms.ReadTensorFloat32(pfx + name)
			if err != nil { log.Fatalf("layer %d %s: %v", l, name, err) }
			return quantLoad(d, rows, cols)
		}
		loadNorm := func(name string) *mongoose.Tensor {
			d, _ := ms.ReadTensorFloat32(pfx + name)
			if d == nil { d = make([]float32, dim); for i := range d { d[i] = 1 } }
			return te.FromHost(d, []int{1, dim})
		}
		lays[l] = layer{
			wq:    loadQ("self_attn.q_proj.weight", dim, dim),
			wk:    loadQ("self_attn.k_proj.weight", kvDim, dim),
			wv:    loadQ("self_attn.v_proj.weight", kvDim, dim),
			wo:    loadQ("self_attn.o_proj.weight", dim, dim),
			gate:  loadQ("mlp.gate_proj.weight", ffnDim, dim),
			up:    loadQ("mlp.up_proj.weight", ffnDim, dim),
			down:  loadQ("mlp.down_proj.weight", dim, ffnDim),
			norm1: loadNorm("input_layernorm.weight"),
			norm2: loadNorm("post_attention_layernorm.weight"),
		}
		if (l+1)%10 == 0 || l == nLayers-1 {
			log.Printf("[finetune] loaded layer %d/%d (INT8 + FP32 cache)", l+1, nLayers)
		}
	}

	// Helix — CPU computes rung, GPU kernel applies DNA-coupled update
	hlx := helix.NewHelixOptimizer(lr, 0.9, 0.95, 1e-8, 0.1)

	// Embed + lm_head stay FP32 (not INT8-quantized)
	type as struct{ m, v *mongoose.Tensor }
	newAS := func(sz int) as { return as{te.Zeros([]int{sz}), te.Zeros([]int{sz})} }
	embedAS := newAS(vocabSize * dim)
	lmHeadAS := newAS(vocabSize * dim)

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

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	fmt.Println("ai train — FP32 backward + Helix DNA optimizer")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  model:    %s\n", *modelPath)
	fmt.Printf("  data:     %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  arch:     dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6)
	fmt.Printf("  training: steps=%d lr=%.0e\n", *stepsFlag, *lrFlag)
	fmt.Println()

	// Pre-allocate hot-path buffers outside the training loop
	dScratch := te.Zeros([]int{n, dim})
	tokF := make([]float32, n)
	targF := make([]float32, n)

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	fmt.Println("Training...")
	t0 := time.Now()

	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)
		for i := 0; i < n; i++ {
			tokF[i] = math.Float32frombits(uint32(int32(data[start+i])))
			targF[i] = math.Float32frombits(uint32(int32(data[start+i+1])))
		}
		cuda.UploadInto(tokGPU, tokF)
		cuda.UploadInto(targetsGPU, targF)

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

			cuda.MatMulTransposeBTInto(b.Q, b.normed, l.wq.cache, n, dim, dim)
			cuda.MatMulTransposeBTInto(b.K, b.normed, l.wk.cache, n, dim, kvDim)
			cuda.MatMulTransposeBTInto(b.V, b.normed, l.wv.cache, n, dim, kvDim)

			mongoose.KRoPE(b.Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(b.K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			zero(b.attnOut)
			mongoose.KCausalAttentionGQA(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			cuda.MatMulTransposeBTInto(b.dx, b.attnOut, l.wo.cache, n, dim, dim)
			te.AddInPlace(hidden, b.dx)

			cuda.CopyInto(b.xMid, hidden)
			zero(b.normed2); zero(b.rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), b.normed2.DevicePtr(),
				l.norm2.DevicePtr(), b.rmsScale2.DevicePtr(), n, dim)

			cuda.MatMulTransposeBTInto(b.gatePre, b.normed2, l.gate.cache, n, dim, ffnDim)
			cuda.MatMulTransposeBTInto(b.upOut, b.normed2, l.up.cache, n, dim, ffnDim)

			zero(b.ffnMid)
			mongoose.KSiLUGateMul(b.gatePre.DevicePtr(), b.upOut.DevicePtr(), b.ffnMid.DevicePtr(), n*ffnDim)

			cuda.MatMulTransposeBTInto(b.dx, b.ffnMid, l.down.cache, n, ffnDim, dim)
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

		// Always read loss — helix immune system needs it every step
		cuda.Sync()
		var stepLoss float32
		lossH := te.ToHost(lossesGPU)
		for _, l := range lossH { stepLoss += l }
		stepLoss /= float32(n)

		// === BACKWARD (GPU) ===
		cuda.MatMulTransposeATInto(dLmHead, gradGPU, normedFinal, n, vocabSize, dim)
		cuda.MatMulTInto(dHidden, gradGPU, lmHead, n, vocabSize, dim)

		zero(dScratch)
		mongoose.KRMSNormBackward(dHidden.DevicePtr(), hidden.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), dScratch.DevicePtr(), n, dim)
		cuda.CopyInto(dHidden, dScratch)

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]
			b := &bufs[li]

			cuda.MatMulTInto(b.dFfnMid, dHidden, l.down.cache, n, dim, ffnDim)
			cuda.MatMulTransposeATInto(b.dWDown, dHidden, b.ffnMid, n, dim, ffnDim)

			zero(b.dGate); zero(b.dUp)
			mongoose.KSiLUGateBackward(b.dFfnMid.DevicePtr(), b.gatePre.DevicePtr(),
				b.upOut.DevicePtr(), b.dGate.DevicePtr(), b.dUp.DevicePtr(), n*ffnDim)

			cuda.MatMulTInto(b.dN2, b.dGate, l.gate.cache, n, ffnDim, dim)
			cuda.MatMulTInto(b.dx, b.dUp, l.up.cache, n, ffnDim, dim)
			te.AddInPlace(b.dN2, b.dx)

			cuda.MatMulTransposeATInto(b.dWGate, b.dGate, b.normed2, n, ffnDim, dim)
			cuda.MatMulTransposeATInto(b.dWUp, b.dUp, b.normed2, n, ffnDim, dim)

			zero(b.dx)
			mongoose.KRMSNormBackward(b.dN2.DevicePtr(), b.xMid.DevicePtr(),
				l.norm2.DevicePtr(), b.rmsScale2.DevicePtr(), b.dx.DevicePtr(), n, dim)
			te.AddInPlace(dHidden, b.dx)

			cuda.MatMulTInto(b.dAttnOut, dHidden, l.wo.cache, n, dim, dim)
			cuda.MatMulTransposeATInto(b.dWO, dHidden, b.attnOut, n, dim, dim)

			zero(b.dQ); zero(b.dK); zero(b.dV)
			mongoose.KCausalAttentionBackward(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.dAttnOut.DevicePtr(),
				b.dQ.DevicePtr(), b.dK.DevicePtr(), b.dV.DevicePtr(), n, dim, kvDim, heads, kvHeads)

			mongoose.KRoPEBackward(b.dQ.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPEBackward(b.dK.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			cuda.MatMulTInto(b.dN1, b.dQ, l.wq.cache, n, dim, dim)
			cuda.MatMulTInto(b.dx, b.dK, l.wk.cache, n, kvDim, dim)
			cuda.MatMulTInto(b.dN2, b.dV, l.wv.cache, n, kvDim, dim)
			te.AddInPlace(b.dN1, b.dx); te.AddInPlace(b.dN1, b.dN2)

			cuda.MatMulTransposeATInto(b.dWQ, b.dQ, b.normed, n, dim, dim)
			cuda.MatMulTransposeATInto(b.dWK, b.dK, b.normed, n, kvDim, dim)
			cuda.MatMulTransposeATInto(b.dWV, b.dV, b.normed, n, kvDim, dim)

			zero(b.dx)
			mongoose.KRMSNormBackward(b.dN1.DevicePtr(), b.xIn.DevicePtr(),
				l.norm1.DevicePtr(), b.rmsScale1.DevicePtr(), b.dx.DevicePtr(), n, dim)
			te.AddInPlace(dHidden, b.dx)
		}

		// === HELIX + NEEDLE — CPU rung, GPU INT8 update ===
		hlx.Step(step, stepLoss, lr)
		r := hlx.CurrentRung()
		beta1 := float32(0.9)
		beta2 := float32(0.95)

		needleUpdate := func(nw *needleWeight, gradT *mongoose.Tensor) {
			mongoose.KHelixNeedle(nw.q8.DataPtr, nw.q8.ScalePtr, gradT.DevicePtr(),
				nw.mom, nw.vel, nw.mask,
				lr, beta1, beta2, step, 1e-8, 0.1, nw.q8.Rows*nw.q8.Cols, nw.q8.Cols)
			cuda.DequantToFP32(nw.q8, nw.cache.DevicePtr())
		}
		needlePaired := func(nw1, nw2 *needleWeight, g1, g2 *mongoose.Tensor, bs float32) {
			mongoose.KHelixNeedlePaired(
				nw1.q8.DataPtr, nw2.q8.DataPtr, nw1.q8.ScalePtr, nw2.q8.ScalePtr,
				g1.DevicePtr(), g2.DevicePtr(),
				nw1.mom, nw2.mom, nw1.vel, nw2.vel, nw1.mask,
				lr, beta1, beta2, step, 1e-8, 0.1,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				bs, nw1.q8.Rows*nw1.q8.Cols, nw1.q8.Cols)
			cuda.DequantToFP32(nw1.q8, nw1.cache.DevicePtr())
			cuda.DequantToFP32(nw2.q8, nw2.cache.DevicePtr())
		}
		adamW := func(param, grad, mS, vS *mongoose.Tensor) {
			mongoose.KAdamW(param.DevicePtr(), grad.DevicePtr(), mS.DevicePtr(), vS.DevicePtr(),
				lr, 0.1, step, param.Size)
		}

		for li := range lays {
			b := &bufs[li]
			needlePaired(&lays[li].gate, &lays[li].up, b.dWGate, b.dWUp, 3.0/5.0)
			if lays[li].wq.q8.Rows == lays[li].wk.q8.Rows && lays[li].wq.q8.Cols == lays[li].wk.q8.Cols {
				needlePaired(&lays[li].wq, &lays[li].wk, b.dWQ, b.dWK, 2.0/5.0)
			} else {
				needleUpdate(&lays[li].wq, b.dWQ)
				needleUpdate(&lays[li].wk, b.dWK)
			}
			needleUpdate(&lays[li].wv, b.dWV)
			needleUpdate(&lays[li].wo, b.dWO)
			needleUpdate(&lays[li].down, b.dWDown)
		}
		adamW(lmHead, dLmHead, lmHeadAS.m, lmHeadAS.v)
		adamW(embed, dLmHead, embedAS.m, embedAS.v)

		if step <= 3 || step%*logEvery == 0 {
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, stepLoss, lr, elapsed.Seconds(), float64(step)/elapsed.Seconds())
		}
	}

	cuda.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)\n", *stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds())
}
