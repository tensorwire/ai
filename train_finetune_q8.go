package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/helix"
	"github.com/open-ai-org/mongoose"
	"github.com/open-ai-org/tokenizer"
)

// cmdFinetuneQ8 fine-tunes a pretrained model on CUDA using Q8+LoRA.
//
// Memory model (matches Apr 15 successful 14B training):
//   - Frozen INT8 base weights, all layers resident (~12.3GB for 14B)
//   - Shared FP32 dequant buffer, one projection at a time (~2.97GB)
//   - FP32 LoRA adapters (rank-N per projection, ~262MB at rank-16)
//   - GPU arena for activations + gradients (~5GB)
//   - No optimizer state on base weights — they're frozen
//   - Helix DNA optimizer on LoRA params only
func cmdFinetuneQ8(modelPath, dataPath string, steps int, lr float64, logEvery int) {
	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatal("TensorEngine not available")
	}
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok {
		log.Fatalf("finetune-q8 requires CUDA (detected: %s)", eng.Name())
	}
	if !mongoose.LoadKernels() {
		log.Fatal("CUDA kernels required")
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
	rank := profile.LoRARank

	tok, err := tokenizer.LoadTokenizer(modelPath)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	raw, err := os.ReadFile(dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	tokens := tok.Encode(string(raw))
	log.Printf("[q8+lora] BPE tokenized %d bytes → %d tokens (%.1fx compression)",
		len(raw), len(tokens), float64(len(raw))/float64(len(tokens)))
	if len(tokens) < n+1 {
		log.Fatalf("training data too small: %d tokens, need at least %d", len(tokens), n+1)
	}

	// RoPE
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

	// VRAM guard: estimate total before allocating
	{
		layerParams := int64(dim*dim*2 + kvDim*dim*2 + dim*dim + ffnDim*dim*2 + dim*ffnDim)
		int8Bytes := layerParams * int64(nLayers)                                       // INT8 base (1 byte/param)
		loraParams := int64(nLayers) * 7 * 2 * int64(rank) * int64(dim) * 4             // LoRA A+B FP32
		loraAdam := loraParams * 2                                                       // Adam M+V for LoRA
		embedBytes := int64(vocabSize) * int64(dim) * 4 * 2                              // embed + lm_head FP32
		dequantBytes := int64(ffnDim) * int64(dim) * 4                                   // shared dequant buffer
		actBytes := int64(n) * int64(dim) * 4 * 20                                       // ~20 activation buffers
		gradBytes := int64(n) * int64(vocabSize) * 4 * 2                                 // logitsBuf + gradGPU
		totalEst := int8Bytes + loraParams + loraAdam + embedBytes + dequantBytes + actBytes + gradBytes
		vramBytes := eng.VRAM()
		headroom := int64(float64(vramBytes) * 0.90)
		if totalEst > headroom {
			log.Fatalf("[q8+lora] estimated %.1f GB but only %.1f GB VRAM (90%% of %.1f GB). Reduce rank or seq length.",
				float64(totalEst)/(1024*1024*1024), float64(headroom)/(1024*1024*1024), float64(vramBytes)/(1024*1024*1024))
		}
		log.Printf("[q8+lora] VRAM estimate: %.1f GB / %.1f GB available",
			float64(totalEst)/(1024*1024*1024), float64(vramBytes)/(1024*1024*1024))
	}

	log.Printf("[q8+lora] loading %s (Q8 frozen base + LoRA rank-%d)", modelPath, rank)

	// --- Frozen INT8 base weights (no optimizer state) ---
	type frozenQ8 struct {
		q8         *mongoose.Int8Tensor
		rows, cols int
	}
	loadFrozen := func(name string, rows, cols int) frozenQ8 {
		data, err := ms.ReadTensorFloat32(name)
		if err != nil {
			log.Fatalf("load %s: %v", name, err)
		}
		qt := gguf.QuantizeToInt8(data, rows, cols)
		q8 := cuda.FromHostInt8(&mongoose.QuantizedTensor{
			DataInt8: qt.DataInt8, Scales: qt.Scales, Shape: qt.Shape,
			Rows: qt.Rows, Cols: qt.Cols,
		})
		return frozenQ8{q8: q8, rows: rows, cols: cols}
	}

	// --- LoRA adapters: B[outDim, rank] @ A[rank, inDim] ---
	type lora struct {
		A, B   *mongoose.Tensor
		aM, aV *mongoose.Tensor // Adam state for A
		bM, bV *mongoose.Tensor // Adam state for B
		out, in int
	}
	initLora := func(outDim, inDim int) lora {
		// A: Kaiming init, B: zero (LoRA starts as identity)
		scale := float32(math.Sqrt(2.0 / float64(rank)))
		aData := make([]float32, rank*inDim)
		for i := range aData {
			aData[i] = scale * (2*rand.Float32() - 1)
		}
		bData := make([]float32, outDim*rank)
		return lora{
			A:  te.FromHost(aData, []int{rank, inDim}),
			B:  te.FromHost(bData, []int{outDim, rank}),
			aM: te.Zeros([]int{rank * inDim}), aV: te.Zeros([]int{rank * inDim}),
			bM: te.Zeros([]int{outDim * rank}), bV: te.Zeros([]int{outDim * rank}),
			out: outDim, in: inDim,
		}
	}

	// --- FP32 embed + lm_head (trainable) ---
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
		for i := range fnData { fnData[i] = 1 }
	}
	finalNorm := te.FromHost(fnData, []int{1, dim})

	// --- Load layers: frozen Q8 base + LoRA adapters ---
	type layer struct {
		wq, wk, wv, wo, gate, up, down frozenQ8
		norm1, norm2                    *mongoose.Tensor
		lq, lk, lv, lo, lgate, lup, ldown lora
	}
	loadNorm := func(name string) *mongoose.Tensor {
		d, _ := ms.ReadTensorFloat32(name)
		if d == nil {
			d = make([]float32, dim)
			for i := range d { d[i] = 1 }
		}
		return te.FromHost(d, []int{1, dim})
	}

	var intBytes, fpBytes, loraBytes int64
	lays := make([]layer, nLayers)
	for l := 0; l < nLayers; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		lays[l] = layer{
			wq:    loadFrozen(pfx+"self_attn.q_proj.weight", dim, dim),
			wk:    loadFrozen(pfx+"self_attn.k_proj.weight", kvDim, dim),
			wv:    loadFrozen(pfx+"self_attn.v_proj.weight", kvDim, dim),
			wo:    loadFrozen(pfx+"self_attn.o_proj.weight", dim, dim),
			gate:  loadFrozen(pfx+"mlp.gate_proj.weight", ffnDim, dim),
			up:    loadFrozen(pfx+"mlp.up_proj.weight", ffnDim, dim),
			down:  loadFrozen(pfx+"mlp.down_proj.weight", dim, ffnDim),
			norm1: loadNorm(pfx + "input_layernorm.weight"),
			norm2: loadNorm(pfx + "post_attention_layernorm.weight"),
			lq:    initLora(dim, dim),
			lk:    initLora(kvDim, dim),
			lv:    initLora(kvDim, dim),
			lo:    initLora(dim, dim),
			lgate: initLora(ffnDim, dim),
			lup:   initLora(ffnDim, dim),
			ldown: initLora(dim, ffnDim),
		}
		// Track memory
		layerINT8 := int64(dim*dim + kvDim*dim*2 + dim*dim + ffnDim*dim*2 + dim*ffnDim) // bytes
		layerLora := int64(7 * 2 * rank * dim * 4) // rough: 7 projections × (A+B) × rank × dim × 4
		intBytes += layerINT8
		loraBytes += layerLora

		if (l+1)%10 == 0 || l == nLayers-1 {
			log.Printf("[q8+lora] loaded layer %d/%d (INT8: %.1fGB, FP32: %.1fMB, LoRA: %.1fMB)",
				l+1, nLayers,
				float64(intBytes)/(1024*1024*1024),
				float64(fpBytes)/(1024*1024),
				float64(loraBytes)/(1024*1024))
		}
	}

	// --- Shared FP32 dequant buffer (one projection at a time) ---
	maxElems := ffnDim * dim
	if dim*dim > maxElems {
		maxElems = dim * dim
	}
	dequantBuf := te.Zeros([]int{maxElems})

	// --- LoRA intermediate buffers ---
	loraMid := te.Zeros([]int{n, rank})
	loraAdd := te.Zeros([]int{n, ffnDim}) // sized for largest output dim

	// --- Activation buffers (single set, reused across layers) ---
	hidden := te.Zeros([]int{n, dim})
	tokGPU := te.Zeros([]int{n})
	targetsGPU := te.Zeros([]int{n})
	logitsBuf := te.Zeros([]int{n, vocabSize})
	lossesGPU := te.Zeros([]int{n})
	gradGPU := te.Zeros([]int{n, vocabSize})
	normedFinal := te.Zeros([]int{n, dim})
	finalScales := te.Zeros([]int{n})
	// lm_head frozen — no gradient buffer needed (saves 3.1GB for 14B)
	dHidden := te.Zeros([]int{n, dim})
	dScratch := te.Zeros([]int{n, dim})

	xIn := te.Zeros([]int{n, dim})
	normed := te.Zeros([]int{n, dim})
	Q := te.Zeros([]int{n, dim})
	K := te.Zeros([]int{n, kvDim})
	V := te.Zeros([]int{n, kvDim})
	attnOut := te.Zeros([]int{n, dim})
	xMid := te.Zeros([]int{n, dim})
	normed2 := te.Zeros([]int{n, dim})
	gatePre := te.Zeros([]int{n, ffnDim})
	upOut := te.Zeros([]int{n, ffnDim})
	ffnMid := te.Zeros([]int{n, ffnDim})
	rmsScale1 := te.Zeros([]int{n})
	rmsScale2 := te.Zeros([]int{n})
	dx := te.Zeros([]int{n, dim})

	// Gradient buffers (reused)
	dQ := te.Zeros([]int{n, dim})
	dK := te.Zeros([]int{n, kvDim})
	dV := te.Zeros([]int{n, kvDim})
	dAttnOut := te.Zeros([]int{n, dim})
	dN1 := te.Zeros([]int{n, dim})
	dN2 := te.Zeros([]int{n, dim})
	dFfnMid := te.Zeros([]int{n, ffnDim})
	dGate := te.Zeros([]int{n, ffnDim})
	dUp := te.Zeros([]int{n, ffnDim})

	// LoRA gradient buffers — two sets for paired projections
	dLoraMid := te.Zeros([]int{n, rank})
	dA1 := te.Zeros([]int{rank * ffnDim}) // pair slot 1
	dB1 := te.Zeros([]int{ffnDim * rank})
	dA2 := te.Zeros([]int{rank * ffnDim}) // pair slot 2
	dB2 := te.Zeros([]int{ffnDim * rank})

	// lm_head + embed frozen — LoRA adapters handle all trainable params

	hlx := helix.NewHelixOptimizer(float32(lr), 0.9, 0.95, 1e-8, 0.1)

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	// loraForward: out = input @ dequant(W)^T + input @ A^T @ B^T
	loraForward := func(out, input *mongoose.Tensor, fq frozenQ8, la lora, seqN, inDim, outDim int) {
		cuda.DequantToFP32(fq.q8, dequantBuf.DevicePtr())
		cuda.MatMulTransposeBTInto(out, input, dequantBuf, seqN, inDim, outDim)
		cuda.MatMulTransposeBTInto(loraMid, input, la.A, seqN, inDim, rank)
		cuda.MatMulTransposeBTInto(loraAdd, loraMid, la.B, seqN, rank, outDim)
		te.AddInPlace(out, loraAdd)
	}

	// loraBackward: compute dA and dB from output gradient into specified buffers
	loraBackward := func(dOut, input *mongoose.Tensor, la lora, dABuf, dBBuf *mongoose.Tensor, seqN, inDim, outDim int) {
		cuda.MatMulTransposeBTInto(loraMid, input, la.A, seqN, inDim, rank)
		cuda.MatMulTransposeATInto(dBBuf, dOut, loraMid, seqN, outDim, rank)
		cuda.MatMulTInto(dLoraMid, dOut, la.B, seqN, outDim, rank)
		cuda.MatMulTransposeATInto(dABuf, dLoraMid, input, seqN, rank, inDim)
	}

	// loraStepPaired: Helix DNA paired update using two gradient slots
	loraStepPaired := func(la1, la2 *lora, gA1, gB1, gA2, gB2 *mongoose.Tensor, r helix.Rung, bs float32, step int) {
		mongoose.KHelixDNAStep(
			la1.A.DevicePtr(), la2.A.DevicePtr(), gA1.DevicePtr(), gA2.DevicePtr(),
			la1.aM.DevicePtr(), la2.aM.DevicePtr(),
			la1.aV.DevicePtr(), la2.aV.DevicePtr(),
			float32(lr), 0.9, 0.95, step, 1e-8, 0.1,
			r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
			bs, la1.A.Size)
		mongoose.KHelixDNAStep(
			la1.B.DevicePtr(), la2.B.DevicePtr(), gB1.DevicePtr(), gB2.DevicePtr(),
			la1.bM.DevicePtr(), la2.bM.DevicePtr(),
			la1.bV.DevicePtr(), la2.bV.DevicePtr(),
			float32(lr), 0.9, 0.95, step, 1e-8, 0.1,
			r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
			bs, la1.B.Size)
	}

	loraStepAdam := func(la *lora, gA, gB *mongoose.Tensor, step int) {
		mongoose.KAdamW(la.A.DevicePtr(), gA.DevicePtr(), la.aM.DevicePtr(), la.aV.DevicePtr(),
			float32(lr), 0.1, step, la.A.Size)
		mongoose.KAdamW(la.B.DevicePtr(), gB.DevicePtr(), la.bM.DevicePtr(), la.bV.DevicePtr(),
			float32(lr), 0.1, step, la.B.Size)
	}

	totalLoraParams := int64(nLayers) * 7 * 2 * int64(rank) * int64(dim)
	fmt.Println("ai finetune — Q8 frozen base + LoRA + Helix DNA optimizer")
	fmt.Printf("  engine:     %s\n", eng.Name())
	fmt.Printf("  model:      %s\n", modelPath)
	fmt.Printf("  data:       %s (%d tokens)\n", dataPath, len(tokens))
	fmt.Printf("  arch:       dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	fmt.Printf("  lora:       rank=%d (%.1fM trainable params)\n", rank, float64(totalLoraParams)/1e6)
	fmt.Printf("  training:   steps=%d lr=%.0e seq=%d\n", steps, lr, seqLen)
	fmt.Println()

	// Checkpoint save closure
	saveCheckpoint := func(step int) {
		cuda.Sync()
		outDir := filepath.Join(filepath.Dir(modelPath), fmt.Sprintf("lora-step-%d", step))
		os.MkdirAll(outDir, 0755)
		w := gguf.NewGGUFWriter()
		w.AddString("general.architecture", "lora")
		w.AddUint32("lora.rank", uint32(rank))
		w.AddUint32("lora.layers", uint32(nLayers))
		for l := 0; l < nLayers; l++ {
			pfx := fmt.Sprintf("blk.%d.", l)
			save := func(name string, la lora) {
				w.AddTensorF32(pfx+name+".lora_a", te.ToHost(la.A), rank, la.in)
				w.AddTensorF32(pfx+name+".lora_b", te.ToHost(la.B), la.out, rank)
			}
			save("attn_q", lays[l].lq)
			save("attn_k", lays[l].lk)
			save("attn_v", lays[l].lv)
			save("attn_output", lays[l].lo)
			save("ffn_gate", lays[l].lgate)
			save("ffn_up", lays[l].lup)
			save("ffn_down", lays[l].ldown)
		}
		outPath := filepath.Join(outDir, "adapters.gguf")
		if err := w.Write(outPath); err != nil {
			log.Printf("WARN: save checkpoint: %v", err)
			return
		}
		log.Printf("[q8+lora] checkpoint saved: %s", outPath)
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	tokF := make([]float32, n)
	targF := make([]float32, n)

	fmt.Println("Training...")
	t0 := time.Now()

	for step := 1; step <= steps; step++ {
		start := rng.Intn(len(tokens) - n - 1)
		for i := 0; i < n; i++ {
			tokF[i] = math.Float32frombits(uint32(int32(tokens[start+i])))
			targF[i] = math.Float32frombits(uint32(int32(tokens[start+i+1])))
		}
		cuda.UploadInto(tokGPU, tokF)
		cuda.UploadInto(targetsGPU, targF)

		// === FORWARD ===
		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		for li := range lays {
			l := &lays[li]

			cuda.CopyInto(xIn, hidden)
			zero(normed); zero(rmsScale1)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed.DevicePtr(),
				l.norm1.DevicePtr(), rmsScale1.DevicePtr(), n, dim)

			loraForward(Q, normed, l.wq, l.lq, n, dim, dim)
			loraForward(K, normed, l.wk, l.lk, n, dim, kvDim)
			loraForward(V, normed, l.wv, l.lv, n, dim, kvDim)

			mongoose.KRoPE(Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			zero(attnOut)
			mongoose.KCausalAttentionGQA(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			loraForward(dx, attnOut, l.wo, l.lo, n, dim, dim)
			te.AddInPlace(hidden, dx)

			cuda.CopyInto(xMid, hidden)
			zero(normed2); zero(rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed2.DevicePtr(),
				l.norm2.DevicePtr(), rmsScale2.DevicePtr(), n, dim)

			loraForward(gatePre, normed2, l.gate, l.lgate, n, dim, ffnDim)
			loraForward(upOut, normed2, l.up, l.lup, n, dim, ffnDim)

			zero(ffnMid)
			mongoose.KSiLUGateMul(gatePre.DevicePtr(), upOut.DevicePtr(), ffnMid.DevicePtr(), n*ffnDim)

			loraForward(dx, ffnMid, l.down, l.ldown, n, ffnDim, dim)
			te.AddInPlace(hidden, dx)
		}

		zero(normedFinal); zero(finalScales)
		mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), n, dim)

		cuda.MatMulTransposeBTInto(logitsBuf, normedFinal, lmHead, n, dim, vocabSize)

		zero(lossesGPU); zero(gradGPU)
		invN := float32(1.0) / float32(n)
		mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
			lossesGPU.DevicePtr(), gradGPU.DevicePtr(), n, vocabSize, invN)

		cuda.Sync()
		var stepLoss float32
		lossH := te.ToHost(lossesGPU)
		for _, v := range lossH {
			stepLoss += v
		}
		stepLoss /= float32(n)

		// === BACKWARD ===
		// lm_head is frozen — skip dLmHead, just backprop through it
		cuda.MatMulTInto(dHidden, gradGPU, lmHead, n, vocabSize, dim)

		zero(dScratch)
		mongoose.KRMSNormBackward(dHidden.DevicePtr(), hidden.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), dScratch.DevicePtr(), n, dim)
		cuda.CopyInto(dHidden, dScratch)

		// Helix rung for this step
		hlx.Step(step, stepLoss, float32(lr))
		r := hlx.CurrentRung()
		skipPaired := true // TODO: debug KHelixDNAStep call convention for LoRA paired projections

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]

			// FFN backward
			cuda.DequantToFP32(l.down.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dFfnMid, dHidden, dequantBuf, n, dim, ffnDim)
			loraBackward(dHidden, ffnMid, l.ldown, dA1, dB1, n, ffnDim, dim)
			loraStepAdam(&l.ldown, dA1, dB1, step)

			zero(dGate); zero(dUp)
			mongoose.KSiLUGateBackward(dFfnMid.DevicePtr(), gatePre.DevicePtr(),
				upOut.DevicePtr(), dGate.DevicePtr(), dUp.DevicePtr(), n*ffnDim)

			cuda.DequantToFP32(l.gate.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dN2, dGate, dequantBuf, n, ffnDim, dim)

			cuda.DequantToFP32(l.up.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dx, dUp, dequantBuf, n, ffnDim, dim)
			te.AddInPlace(dN2, dx)

			// gate↔up paired: compute both gradients, then paired step
			loraBackward(dGate, normed2, l.lgate, dA1, dB1, n, dim, ffnDim)
			loraBackward(dUp, normed2, l.lup, dA2, dB2, n, dim, ffnDim)
			if !skipPaired { loraStepPaired(&l.lgate, &l.lup, dA1, dB1, dA2, dB2, r, 3.0/5.0, step) } else { loraStepAdam(&l.lgate, dA1, dB1, step); loraStepAdam(&l.lup, dA2, dB2, step) }

			zero(dx)
			mongoose.KRMSNormBackward(dN2.DevicePtr(), xMid.DevicePtr(),
				l.norm2.DevicePtr(), rmsScale2.DevicePtr(), dx.DevicePtr(), n, dim)
			te.AddInPlace(dHidden, dx)

			// Attention backward
			cuda.DequantToFP32(l.wo.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dAttnOut, dHidden, dequantBuf, n, dim, dim)
			loraBackward(dHidden, attnOut, l.lo, dA1, dB1, n, dim, dim)
			loraStepAdam(&l.lo, dA1, dB1, step)

			zero(dQ); zero(dK); zero(dV)
			mongoose.KCausalAttentionBackward(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), dAttnOut.DevicePtr(),
				dQ.DevicePtr(), dK.DevicePtr(), dV.DevicePtr(), n, dim, kvDim, heads, kvHeads)

			mongoose.KRoPEBackward(dQ.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPEBackward(dK.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			cuda.DequantToFP32(l.wq.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dN1, dQ, dequantBuf, n, dim, dim)

			cuda.DequantToFP32(l.wk.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dx, dK, dequantBuf, n, kvDim, dim)
			te.AddInPlace(dN1, dx)

			cuda.DequantToFP32(l.wv.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dx, dV, dequantBuf, n, kvDim, dim)
			te.AddInPlace(dN1, dx)

			// q↔k paired: compute both gradients, then paired step
			loraBackward(dQ, normed, l.lq, dA1, dB1, n, dim, dim)
			loraBackward(dK, normed, l.lk, dA2, dB2, n, dim, kvDim)
			if !skipPaired { loraStepPaired(&l.lq, &l.lk, dA1, dB1, dA2, dB2, r, 2.0/5.0, step) } else { loraStepAdam(&l.lq, dA1, dB1, step); loraStepAdam(&l.lk, dA2, dB2, step) }

			loraBackward(dV, normed, l.lv, dA1, dB1, n, dim, kvDim)
			loraStepAdam(&l.lv, dA1, dB1, step)

			zero(dx)
			mongoose.KRMSNormBackward(dN1.DevicePtr(), xIn.DevicePtr(),
				l.norm1.DevicePtr(), rmsScale1.DevicePtr(), dx.DevicePtr(), n, dim)
			te.AddInPlace(dHidden, dx)
		}

		// lm_head frozen — no optimizer step

		if step <= 3 || step%logEvery == 0 {
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s, %.0fms/step)\n",
				step, steps, stepLoss, lr, elapsed.Seconds(),
				float64(step)/elapsed.Seconds(), elapsed.Seconds()/float64(step)*1000)
		}

		if step%1000 == 0 || step == steps {
			saveCheckpoint(step)
		}
	}

	cuda.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.1fs (%.1f steps/s)\n",
		steps, total.Seconds(), float64(steps)/total.Seconds())
}
