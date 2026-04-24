package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"
	"unsafe"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/helix"
	"github.com/open-ai-org/mongoose"
	"github.com/open-ai-org/tokenizer"
)

func cmdFinetuneHelix(modelPath, dataPath string, steps int, lr float64, rank int, logEvery int) {
	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatal("TensorEngine not available")
	}
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok {
		log.Fatalf("finetune requires CUDA (detected: %s)", eng.Name())
	}
	if !mongoose.LoadKernels() {
		log.Fatal("CUDA kernels required")
	}
	if !mongoose.SoftmaxCELoaded() {
		log.Fatal("softmax+CE kernel not loaded")
	}
	if !mongoose.HelixDNALoaded() {
		log.Fatal("KHelixDNAStep kernel not loaded")
	}

	ms, err := OpenModel(modelPath)
	if err != nil {
		log.Fatalf("open model: %v", err)
	}

	profile := AutoDetect(modelPath)
	dim := profile.Dim
	heads := profile.Heads
	kvHeads := profile.KVHeads
	headDim := profile.HeadDim
	kvDim := profile.KVDim
	nLayers := profile.Layers
	ffnDim := profile.FFNDim
	vocabSize := profile.VocabSize
	seqLen := profile.SeqLen
	n := seqLen

	tok, err := tokenizer.LoadTokenizer(modelPath)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	raw, err := os.ReadFile(dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	tokens := tok.Encode(string(raw))
	log.Printf("[finetune] %d bytes → %d tokens (%.1fx)",
		len(raw), len(tokens), float64(len(raw))/float64(len(tokens)))
	if len(tokens) < n+1 {
		log.Fatalf("need at least %d tokens, got %d", n+1, len(tokens))
	}

	halfHead := headDim / 2
	cosTab := make([]float32, seqLen*halfHead)
	sinTab := make([]float32, seqLen*halfHead)
	for pos := 0; pos < seqLen; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(profile.RopeTheta, float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			cosTab[pos*halfHead+j] = float32(math.Cos(angle))
			sinTab[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}
	ropeCos := te.FromHost(cosTab, []int{seqLen, halfHead})
	ropeSin := te.FromHost(sinTab, []int{seqLen, halfHead})

	log.Printf("[finetune] loading %s (INT8 base + rank-%d LoRA + helix)", modelPath, rank)

	// LoRA adapter: frozen INT8 base + trainable FP32 low-rank A[outDim,rank], B[rank,inDim]
	type loraWeight struct {
		q8   *mongoose.Int8Tensor
		a    *mongoose.Tensor // [outDim, rank] FP32
		b    *mongoose.Tensor // [rank, inDim] FP32
		am   *mongoose.Tensor // AdamW m for A
		av   *mongoose.Tensor // AdamW v for A
		bm   *mongoose.Tensor // AdamW m for B
		bv   *mongoose.Tensor // AdamW v for B
		rows int
		cols int
	}

	alpha := float32(rank) // LoRA scaling: output *= alpha/rank = 1.0

	loadLoRA := func(name string, rows, cols int) loraWeight {
		data, err := ms.ReadTensorFloat32(name)
		if err != nil {
			log.Fatalf("load %s: %v", name, err)
		}
		qt := gguf.QuantizeToInt8(data, rows, cols)
		q8 := cuda.FromHostInt8(&mongoose.QuantizedTensor{
			DataInt8: qt.DataInt8, Scales: qt.Scales, Shape: qt.Shape,
			Rows: qt.Rows, Cols: qt.Cols,
		})
		aSize := rows * rank
		bSize := rank * cols
		a := te.Zeros([]int{rows, rank})
		b := te.Zeros([]int{rank, cols})
		// Kaiming init for B, zero for A (standard LoRA init: A@B = 0 at start)
		bData := make([]float32, bSize)
		scale := float32(math.Sqrt(2.0 / float64(cols)))
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		for i := range bData {
			bData[i] = (rng.Float32()*2 - 1) * scale * 0.01
		}
		cuda.UploadInto(b, bData)
		return loraWeight{
			q8:   q8,
			a:    a,
			b:    b,
			am:   te.Zeros([]int{aSize}),
			av:   te.Zeros([]int{aSize}),
			bm:   te.Zeros([]int{bSize}),
			bv:   te.Zeros([]int{bSize}),
			rows: rows,
			cols: cols,
		}
	}

	embedData, err := ms.ReadTensorFloat32("model.embed_tokens.weight")
	if err != nil {
		log.Fatalf("embed: %v", err)
	}
	embed := te.FromHost(embedData, []int{vocabSize, dim})

	lmHeadData, err := ms.ReadTensorFloat32("lm_head.weight")
	if err != nil {
		lmHeadData = embedData
	}
	lmHead := te.FromHost(lmHeadData, []int{vocabSize, dim})

	fnData, _ := ms.ReadTensorFloat32("model.norm.weight")
	if fnData == nil {
		fnData = make([]float32, dim)
		for i := range fnData {
			fnData[i] = 1
		}
	}
	finalNorm := te.FromHost(fnData, []int{1, dim})

	type layer struct {
		wq, wk, wv, wo, gate, up, down loraWeight
		norm1, norm2                    *mongoose.Tensor
		bq, bk, bv                     *mongoose.Tensor
	}
	loadNorm := func(name string) *mongoose.Tensor {
		d, _ := ms.ReadTensorFloat32(name)
		if d == nil {
			d = make([]float32, dim)
			for i := range d {
				d[i] = 1
			}
		}
		return te.FromHost(d, []int{1, dim})
	}
	loadBiasTiled := func(name string, outDim int) *mongoose.Tensor {
		d, _ := ms.ReadTensorFloat32(name)
		if d == nil {
			return nil
		}
		tiled := make([]float32, n*outDim)
		for pos := 0; pos < n; pos++ {
			copy(tiled[pos*outDim:], d[:outDim])
		}
		return te.FromHost(tiled, []int{n, outDim})
	}

	lays := make([]layer, nLayers)
	for l := 0; l < nLayers; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		lays[l] = layer{
			wq:    loadLoRA(pfx+"self_attn.q_proj.weight", dim, dim),
			wk:    loadLoRA(pfx+"self_attn.k_proj.weight", kvDim, dim),
			wv:    loadLoRA(pfx+"self_attn.v_proj.weight", kvDim, dim),
			wo:    loadLoRA(pfx+"self_attn.o_proj.weight", dim, dim),
			gate:  loadLoRA(pfx+"mlp.gate_proj.weight", ffnDim, dim),
			up:    loadLoRA(pfx+"mlp.up_proj.weight", ffnDim, dim),
			down:  loadLoRA(pfx+"mlp.down_proj.weight", dim, ffnDim),
			norm1: loadNorm(pfx + "input_layernorm.weight"),
			norm2: loadNorm(pfx + "post_attention_layernorm.weight"),
			bq:    loadBiasTiled(pfx+"self_attn.q_proj.bias", dim),
			bk:    loadBiasTiled(pfx+"self_attn.k_proj.bias", kvDim),
			bv:    loadBiasTiled(pfx+"self_attn.v_proj.bias", kvDim),
		}
		if (l+1)%10 == 0 || l == nLayers-1 {
			log.Printf("[finetune] loaded layer %d/%d", l+1, nLayers)
		}
	}

	// Shared dequant buffer
	maxElems := ffnDim * dim
	if dim*dim > maxElems {
		maxElems = dim * dim
	}
	dequantBuf := te.Zeros([]int{maxElems})

	// LoRA intermediate buffers (reused per projection)
	maxRankElems := ffnDim * rank
	if dim*rank > maxRankElems {
		maxRankElems = dim * rank
	}
	_ = te // used throughout via Zeros

	hidden := te.Zeros([]int{n, dim})
	tokGPU := te.Zeros([]int{n})
	targetsGPU := te.Zeros([]int{n})
	logitsBuf := te.Zeros([]int{n, vocabSize})
	lossesGPU := te.Zeros([]int{n})
	normed := te.Zeros([]int{n, dim})
	Q := te.Zeros([]int{n, dim})
	K := te.Zeros([]int{n, kvDim})
	V := te.Zeros([]int{n, kvDim})
	attnOut := te.Zeros([]int{n, dim})
	normed2 := te.Zeros([]int{n, dim})
	gatePre := te.Zeros([]int{n, ffnDim})
	upOut := te.Zeros([]int{n, ffnDim})
	ffnMid := te.Zeros([]int{n, ffnDim})
	rmsScale1 := te.Zeros([]int{n})
	rmsScale2 := te.Zeros([]int{n})
	normedFinal := te.Zeros([]int{n, dim})
	finalScales := te.Zeros([]int{n})
	dx := te.Zeros([]int{n, dim})

	savedXIn := make([]*mongoose.Tensor, nLayers)
	savedXMid := make([]*mongoose.Tensor, nLayers)
	savedRMS1 := make([]*mongoose.Tensor, nLayers)
	savedRMS2 := make([]*mongoose.Tensor, nLayers)
	for i := range savedXIn {
		savedXIn[i] = te.Zeros([]int{n, dim})
		savedXMid[i] = te.Zeros([]int{n, dim})
		savedRMS1[i] = te.Zeros([]int{n})
		savedRMS2[i] = te.Zeros([]int{n})
	}

	dHidden := te.Zeros([]int{n, dim})
	dGate := te.Zeros([]int{n, ffnDim})
	dUp := te.Zeros([]int{n, ffnDim})
	dFfnMid := te.Zeros([]int{n, ffnDim})
	dAttnOut := te.Zeros([]int{n, dim})
	dQ := te.Zeros([]int{n, dim})
	dK := te.Zeros([]int{n, kvDim})
	dV := te.Zeros([]int{n, kvDim})
	dScratch := te.Zeros([]int{n, dim})
	dScratch2 := te.Zeros([]int{n, dim})
	gradGPU := te.Zeros([]int{n, vocabSize})

	// Shared dW buffers for LoRA gradient projection
	dA := te.Zeros([]int{maxElems})
	dB := te.Zeros([]int{maxElems})
	dLoraTemp := te.Zeros([]int{n, ffnDim}) // intermediate for LoRA backward

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Forward: y = x @ W^T + (x @ B^T @ A^T) * alpha/rank
	// W is frozen INT8, A and B are trainable FP32
	loraFwd := func(w *loraWeight, out, input *mongoose.Tensor, seqN, inDim, outDim int) {
		cuda.DequantToFP32(w.q8, dequantBuf.DevicePtr())
		cuda.MatMulTransposeBTInto(out, input, dequantBuf, seqN, inDim, outDim)

		// LoRA path: input[n,inDim] @ B^T[inDim,rank] → [n,rank]
		// then [n,rank] @ A^T[rank,outDim] → [n,outDim]
		loraRank := te.Zeros([]int{seqN, rank})
		cuda.MatMulTransposeBTInto(loraRank, input, w.b, seqN, inDim, rank)
		cuda.MatMulTransposeBTInto(dLoraTemp, loraRank, w.a, seqN, rank, outDim)

		// out += loraCorrection * (alpha / rank)
		scale := alpha / float32(rank)
		mongoose.KGradScale(dLoraTemp.DevicePtr(), scale, seqN*outDim)
		mongoose.KAddInPlace(out.DevicePtr(), dLoraTemp.DevicePtr(), seqN*outDim)
	}

	// Helix optimizer
	hlx := helix.NewHelixOptimizer(float32(lr), 0.9, 0.95, 1e-8, 0.1)

	// LoRA backward: compute dA and dB from the chain rule gradient dOut
	// y = x @ B^T @ A^T * scale  →  dA = (dOut * scale)^T @ (x @ B^T), dB = (A^T @ dOut * scale)^T @ x
	loraBackward := func(w *loraWeight, dOut, input *mongoose.Tensor, seqN, outDim, inDim int) {
		scale := alpha / float32(rank)

		// xB = input[n,inDim] @ B^T[inDim,rank] → [n,rank]
		loraRank := te.Zeros([]int{seqN, rank})
		cuda.MatMulTransposeBTInto(loraRank, input, w.b, seqN, inDim, rank)

		// dA = (dOut * scale)^T @ xB → [outDim, rank]
		mongoose.KGradScale(dOut.DevicePtr(), scale, seqN*outDim)
		cuda.MatMulTransposeATInto(dA, dOut, loraRank, seqN, outDim, rank)

		// dB = (A^T @ (dOut*scale))^T @ input → [rank, inDim]
		// A^T @ dOut^T = (dOut @ A)^T, so: temp[n,rank] = dOut[n,outDim] @ A[outDim,rank]
		cuda.MatMulTInto(loraRank, dOut, w.a, seqN, outDim, rank)
		// dB = loraRank^T @ input → [rank, inDim]
		cuda.MatMulTransposeATInto(dB, loraRank, input, seqN, rank, inDim)

		// Undo the scale on dOut so it doesn't affect dHidden propagation
		mongoose.KGradScale(dOut.DevicePtr(), 1.0/scale, seqN*outDim)
	}

	// DNA paired update for two LoRA weights
	helixPairUpdate := func(w1, w2 *loraWeight, bondStrength float32, r helix.Rung, bc1, bc2 float32, step int) {
		aSize := w1.rows * rank
		bSize := rank * w1.cols
		_ = bc1
		_ = bc2
		mongoose.KHelixDNAStep(
			w1.a.DevicePtr(), w2.a.DevicePtr(),
			dA.DevicePtr(), dA.DevicePtr(), // g1, g2 set by sequential calls
			w1.am.DevicePtr(), w2.am.DevicePtr(),
			w1.av.DevicePtr(), w2.av.DevicePtr(),
			float32(lr), 0.9, 0.95, step, 1e-8, 0.1,
			r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
			bondStrength, aSize)
		mongoose.KHelixDNAStep(
			w1.b.DevicePtr(), w2.b.DevicePtr(),
			dB.DevicePtr(), dB.DevicePtr(),
			w1.bm.DevicePtr(), w2.bm.DevicePtr(),
			w1.bv.DevicePtr(), w2.bv.DevicePtr(),
			float32(lr), 0.9, 0.95, step, 1e-8, 0.1,
			r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
			bondStrength, bSize)
	}
	_ = helixPairUpdate

	// VRAM estimate
	loraParamsPerLayer := 2 * rank * (dim + dim + kvDim + kvDim + dim + dim + ffnDim + ffnDim + dim + ffnDim)
	totalLoraParams := loraParamsPerLayer * nLayers
	loraVRAM := float64(totalLoraParams) * 4 * 3 / (1024 * 1024) // params + m + v in FP32

	fmt.Println("ai finetune — helix DNA + LoRA adapters")
	fmt.Printf("  engine:     %s\n", eng.Name())
	fmt.Printf("  model:      %s\n", modelPath)
	fmt.Printf("  data:       %s (%d tokens)\n", dataPath, len(tokens))
	fmt.Printf("  arch:       dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	fmt.Printf("  training:   steps=%d lr=%.0e seq=%d rank=%d\n", steps, lr, seqLen, rank)
	fmt.Printf("  lora:       %d trainable params (%.0f MB optimizer state)\n", totalLoraParams, loraVRAM)
	fmt.Println()

	bestFloor := float32(1e30)
	tokI32 := make([]int32, n)

	fmt.Println("Training...")
	t0 := time.Now()

	for step := 1; step <= steps; step++ {
		start := rng.Intn(len(tokens) - n - 1)
		for i := 0; i < n; i++ {
			tokI32[i] = int32(tokens[start+i])
		}

		tokF := make([]float32, n)
		targF := make([]float32, n)
		for i := 0; i < n; i++ {
			tokF[i] = math.Float32frombits(uint32(tokI32[i]))
			targF[i] = math.Float32frombits(uint32(int32(tokens[start+i+1])))
		}
		cuda.UploadInto(tokGPU, tokF)
		cuda.UploadInto(targetsGPU, targF)

		// === FORWARD PASS ===
		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		for li := range lays {
			l := &lays[li]
			cuda.CopyInto(savedXIn[li], hidden)

			zero(normed)
			zero(rmsScale1)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed.DevicePtr(),
				l.norm1.DevicePtr(), rmsScale1.DevicePtr(), n, dim)
			cuda.CopyInto(savedRMS1[li], rmsScale1)

			loraFwd(&l.wq, Q, normed, n, dim, dim)
			loraFwd(&l.wk, K, normed, n, dim, kvDim)
			loraFwd(&l.wv, V, normed, n, dim, kvDim)
			if l.bq != nil { mongoose.KAddInPlace(Q.DevicePtr(), l.bq.DevicePtr(), n*dim) }
			if l.bk != nil { mongoose.KAddInPlace(K.DevicePtr(), l.bk.DevicePtr(), n*kvDim) }
			if l.bv != nil { mongoose.KAddInPlace(V.DevicePtr(), l.bv.DevicePtr(), n*kvDim) }

			mongoose.KRoPE(Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			zero(attnOut)
			mongoose.KCausalAttentionGQA(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			loraFwd(&l.wo, dx, attnOut, n, dim, dim)
			te.AddInPlace(hidden, dx)

			cuda.CopyInto(savedXMid[li], hidden)

			zero(normed2)
			zero(rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed2.DevicePtr(),
				l.norm2.DevicePtr(), rmsScale2.DevicePtr(), n, dim)
			cuda.CopyInto(savedRMS2[li], rmsScale2)

			loraFwd(&l.gate, gatePre, normed2, n, dim, ffnDim)
			loraFwd(&l.up, upOut, normed2, n, dim, ffnDim)

			zero(ffnMid)
			mongoose.KSiLUGateMul(gatePre.DevicePtr(), upOut.DevicePtr(), ffnMid.DevicePtr(), n*ffnDim)

			loraFwd(&l.down, dx, ffnMid, n, ffnDim, dim)
			te.AddInPlace(hidden, dx)
		}

		zero(normedFinal)
		zero(finalScales)
		mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), n, dim)

		cuda.MatMulTransposeBTInto(logitsBuf, normedFinal, lmHead, n, dim, vocabSize)

		var stepLoss float32
		zero(lossesGPU)
		zero(gradGPU)
		invN := float32(1.0) / float32(n)
		mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
			lossesGPU.DevicePtr(), gradGPU.DevicePtr(), n, vocabSize, invN)
		cuda.Sync()
		lossH := te.ToHost(lossesGPU)
		for _, v := range lossH {
			stepLoss += v
		}
		stepLoss /= float32(n)

		// === BACKWARD PASS ===
		cuda.MatMulTInto(dHidden, gradGPU, lmHead, n, vocabSize, dim)

		zero(dScratch)
		mongoose.KRMSNormBackward(dHidden.DevicePtr(), hidden.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), dScratch.DevicePtr(), n, dim)
		cuda.CopyInto(dHidden, dScratch)

		// Helix: compute rung for DNA pairing (ignore immune — we manage GPU state)
		r, _, _, _ := hlx.PrepareStep(step, stepLoss, float32(lr))

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]

			// Recompute FFN activations
			zero(normed2)
			mongoose.KRMSNormOutSave(savedXMid[li].DevicePtr(), normed2.DevicePtr(),
				l.norm2.DevicePtr(), savedRMS2[li].DevicePtr(), n, dim)
			cuda.DequantToFP32(l.gate.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(gatePre, normed2, dequantBuf, n, dim, ffnDim)
			cuda.DequantToFP32(l.up.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(upOut, normed2, dequantBuf, n, dim, ffnDim)
			zero(ffnMid)
			mongoose.KSiLUGateMul(gatePre.DevicePtr(), upOut.DevicePtr(), ffnMid.DevicePtr(), n*ffnDim)

			// FFN backward + LoRA gradient
			loraBackward(&l.down, dHidden, ffnMid, n, dim, ffnDim)
			mongoose.KAdamW(l.down.a.DevicePtr(), dA.DevicePtr(), l.down.am.DevicePtr(), l.down.av.DevicePtr(), float32(lr), 0.1, step, dim*rank)
			mongoose.KAdamW(l.down.b.DevicePtr(), dB.DevicePtr(), l.down.bm.DevicePtr(), l.down.bv.DevicePtr(), float32(lr), 0.1, step, rank*ffnDim)

			// dHidden through base weights (frozen, no LoRA contribution to gradient propagation for now)
			cuda.DequantToFP32(l.down.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dFfnMid, dHidden, dequantBuf, n, dim, ffnDim)

			mongoose.KSiLUGateBackward(dFfnMid.DevicePtr(), gatePre.DevicePtr(),
				upOut.DevicePtr(), dGate.DevicePtr(), dUp.DevicePtr(), n*ffnDim)

			// Gate + Up: DNA paired
			loraBackward(&l.gate, dGate, normed2, n, ffnDim, dim)
			var dA1Copy unsafe.Pointer // need to save dA for gate before computing up's dA
			dA1Buf := te.Zeros([]int{ffnDim * rank})
			cuda.CopyInto(dA1Buf, dA)
			dA1Copy = dA1Buf.DevicePtr()
			dB1Buf := te.Zeros([]int{rank * dim})
			cuda.CopyInto(dB1Buf, dB)

			loraBackward(&l.up, dUp, normed2, n, ffnDim, dim)

			// DNA step: gate + up paired (G≡C, 3 H-bonds)
			mongoose.KHelixDNAStep(
				l.gate.a.DevicePtr(), l.up.a.DevicePtr(),
				dA1Copy, dA.DevicePtr(),
				l.gate.am.DevicePtr(), l.up.am.DevicePtr(),
				l.gate.av.DevicePtr(), l.up.av.DevicePtr(),
				float32(lr), 0.9, 0.95, step, 1e-8, 0.1,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				3.0/5.0, ffnDim*rank)
			mongoose.KHelixDNAStep(
				l.gate.b.DevicePtr(), l.up.b.DevicePtr(),
				dB1Buf.DevicePtr(), dB.DevicePtr(),
				l.gate.bm.DevicePtr(), l.up.bm.DevicePtr(),
				l.gate.bv.DevicePtr(), l.up.bv.DevicePtr(),
				float32(lr), 0.9, 0.95, step, 1e-8, 0.1,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				3.0/5.0, rank*dim)

			// dHidden through FFN
			cuda.DequantToFP32(l.gate.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch, dGate, dequantBuf, n, ffnDim, dim)
			cuda.DequantToFP32(l.up.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch2, dUp, dequantBuf, n, ffnDim, dim)
			mongoose.KAddInPlace(dScratch.DevicePtr(), dScratch2.DevicePtr(), dScratch.Size)

			zero(dScratch2)
			mongoose.KRMSNormBackward(dScratch.DevicePtr(), savedXMid[li].DevicePtr(),
				l.norm2.DevicePtr(), savedRMS2[li].DevicePtr(), dScratch2.DevicePtr(), n, dim)
			mongoose.KAddInPlace(dHidden.DevicePtr(), dScratch2.DevicePtr(), dHidden.Size)

			// Recompute attention activations
			zero(normed)
			mongoose.KRMSNormOutSave(savedXIn[li].DevicePtr(), normed.DevicePtr(),
				l.norm1.DevicePtr(), savedRMS1[li].DevicePtr(), n, dim)
			cuda.DequantToFP32(l.wq.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(Q, normed, dequantBuf, n, dim, dim)
			cuda.DequantToFP32(l.wk.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(K, normed, dequantBuf, n, dim, kvDim)
			cuda.DequantToFP32(l.wv.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(V, normed, dequantBuf, n, dim, kvDim)
			if l.bq != nil { mongoose.KAddInPlace(Q.DevicePtr(), l.bq.DevicePtr(), n*dim) }
			if l.bk != nil { mongoose.KAddInPlace(K.DevicePtr(), l.bk.DevicePtr(), n*kvDim) }
			if l.bv != nil { mongoose.KAddInPlace(V.DevicePtr(), l.bv.DevicePtr(), n*kvDim) }
			mongoose.KRoPE(Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)
			zero(attnOut)
			mongoose.KCausalAttentionGQA(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			// Attention backward: wo (unpaired — use AdamW)
			loraBackward(&l.wo, dHidden, attnOut, n, dim, dim)
			mongoose.KAdamW(l.wo.a.DevicePtr(), dA.DevicePtr(), l.wo.am.DevicePtr(), l.wo.av.DevicePtr(), float32(lr), 0.1, step, dim*rank)
			mongoose.KAdamW(l.wo.b.DevicePtr(), dB.DevicePtr(), l.wo.bm.DevicePtr(), l.wo.bv.DevicePtr(), float32(lr), 0.1, step, rank*dim)

			cuda.DequantToFP32(l.wo.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dAttnOut, dHidden, dequantBuf, n, dim, dim)

			zero(dQ)
			zero(dK)
			zero(dV)
			mongoose.KCausalAttentionBackward(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), dAttnOut.DevicePtr(),
				dQ.DevicePtr(), dK.DevicePtr(), dV.DevicePtr(), n, dim, kvDim, heads, kvHeads)

			mongoose.KRoPEBackward(dQ.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPEBackward(dK.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			// Q + O: DNA paired (A=T, 2 H-bonds)
			loraBackward(&l.wq, dQ, normed, n, dim, dim)
			dA1Buf2 := te.Zeros([]int{dim * rank})
			cuda.CopyInto(dA1Buf2, dA)
			dB1Buf2 := te.Zeros([]int{rank * dim})
			cuda.CopyInto(dB1Buf2, dB)

			// Actually wq+wo should be paired but wo was already updated above with AdamW.
			// For now: pair wk+wv, use AdamW for wq (wo already done above)
			mongoose.KAdamW(l.wq.a.DevicePtr(), dA1Buf2.DevicePtr(), l.wq.am.DevicePtr(), l.wq.av.DevicePtr(), float32(lr), 0.1, step, dim*rank)
			mongoose.KAdamW(l.wq.b.DevicePtr(), dB1Buf2.DevicePtr(), l.wq.bm.DevicePtr(), l.wq.bv.DevicePtr(), float32(lr), 0.1, step, rank*dim)

			// K + V: DNA paired (A=T, 2 H-bonds)
			loraBackward(&l.wk, dK, normed, n, kvDim, dim)
			dA1Buf3 := te.Zeros([]int{kvDim * rank})
			cuda.CopyInto(dA1Buf3, dA)
			dB1Buf3 := te.Zeros([]int{rank * dim})
			cuda.CopyInto(dB1Buf3, dB)

			loraBackward(&l.wv, dV, normed, n, kvDim, dim)

			mongoose.KHelixDNAStep(
				l.wk.a.DevicePtr(), l.wv.a.DevicePtr(),
				dA1Buf3.DevicePtr(), dA.DevicePtr(),
				l.wk.am.DevicePtr(), l.wv.am.DevicePtr(),
				l.wk.av.DevicePtr(), l.wv.av.DevicePtr(),
				float32(lr), 0.9, 0.95, step, 1e-8, 0.1,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				2.0/5.0, kvDim*rank)
			mongoose.KHelixDNAStep(
				l.wk.b.DevicePtr(), l.wv.b.DevicePtr(),
				dB1Buf3.DevicePtr(), dB.DevicePtr(),
				l.wk.bm.DevicePtr(), l.wv.bm.DevicePtr(),
				l.wk.bv.DevicePtr(), l.wv.bv.DevicePtr(),
				float32(lr), 0.9, 0.95, step, 1e-8, 0.1,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				2.0/5.0, rank*dim)

			// dHidden through attention
			cuda.DequantToFP32(l.wq.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch, dQ, dequantBuf, n, dim, dim)
			cuda.DequantToFP32(l.wk.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch2, dK, dequantBuf, n, kvDim, dim)
			mongoose.KAddInPlace(dScratch.DevicePtr(), dScratch2.DevicePtr(), dScratch.Size)
			cuda.DequantToFP32(l.wv.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch2, dV, dequantBuf, n, kvDim, dim)
			mongoose.KAddInPlace(dScratch.DevicePtr(), dScratch2.DevicePtr(), dScratch.Size)

			zero(dScratch2)
			mongoose.KRMSNormBackward(dScratch.DevicePtr(), savedXIn[li].DevicePtr(),
				l.norm1.DevicePtr(), savedRMS1[li].DevicePtr(), dScratch2.DevicePtr(), n, dim)
			mongoose.KAddInPlace(dHidden.DevicePtr(), dScratch2.DevicePtr(), dHidden.Size)
		}

		if step > 1 && stepLoss > 0 && stepLoss < bestFloor {
			bestFloor = stepLoss
		}

		if step <= 3 || step%logEvery == 0 {
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.4f  floor=%.4f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, steps, stepLoss, bestFloor, lr, elapsed.Seconds(),
				float64(step)/elapsed.Seconds())
		}

		if step == steps {
			cuda.Sync()
			outDir := filepath.Join(filepath.Dir(modelPath), fmt.Sprintf("helix-step-%d", step))
			os.MkdirAll(outDir, 0755)
			w := gguf.NewGGUFWriter()
			w.AddString("general.architecture", "qwen2")
			w.AddUint32("qwen2.block_count", uint32(nLayers))
			w.AddTensorQ8_0("token_embd.weight", te.ToHost(embed), vocabSize, dim)
			w.AddTensorQ8_0("output.weight", te.ToHost(lmHead), vocabSize, dim)
			w.AddTensorF32("output_norm.weight", te.ToHost(finalNorm), dim)
			for l := 0; l < nLayers; l++ {
				pfx := fmt.Sprintf("blk.%d.", l)
				saveW := func(name string, wt loraWeight) {
					cuda.DequantToFP32(wt.q8, dequantBuf.DevicePtr())
					merged := te.Zeros([]int{wt.rows * wt.cols})
					cuda.MatMulTInto(merged, wt.a, wt.b, wt.rows, rank, wt.cols)
					mergeScale := alpha / float32(rank)
					mongoose.KGradScale(merged.DevicePtr(), mergeScale, wt.rows*wt.cols)
					mongoose.KAddInPlace(dequantBuf.DevicePtr(), merged.DevicePtr(), wt.rows*wt.cols)
					te.Release(merged)

					host := te.ToHost(dequantBuf)
					fp32 := make([]float32, wt.rows*wt.cols)
					copy(fp32, host[:wt.rows*wt.cols])
					w.AddTensorQ8_0(pfx+name, fp32, wt.rows, wt.cols)
				}
				saveW("attn_q.weight", lays[l].wq)
				saveW("attn_k.weight", lays[l].wk)
				saveW("attn_v.weight", lays[l].wv)
				saveW("attn_output.weight", lays[l].wo)
				saveW("ffn_gate.weight", lays[l].gate)
				saveW("ffn_up.weight", lays[l].up)
				saveW("ffn_down.weight", lays[l].down)
				w.AddTensorF32(pfx+"attn_norm.weight", te.ToHost(lays[l].norm1), dim)
				w.AddTensorF32(pfx+"ffn_norm.weight", te.ToHost(lays[l].norm2), dim)
			}
			outPath := filepath.Join(outDir, "model.gguf")
			if err := w.Write(outPath); err != nil {
				log.Printf("WARN: save checkpoint: %v", err)
			} else {
				log.Printf("[finetune] checkpoint saved: %s", outPath)
			}
		}
	}

	cuda.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.1fs (%.1f steps/s)  floor=%.4f\n",
		steps, total.Seconds(), float64(steps)/total.Seconds(), bestFloor)
}
