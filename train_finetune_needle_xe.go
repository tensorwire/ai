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
	"github.com/open-ai-org/mongoose"
	"github.com/open-ai-org/tokenizer"
)

func calcMaxHot(rows, cols int) int {
	n := int(math.Sqrt(float64(rows * cols)))
	if n < 49 {
		n = 49
	}
	return n
}

func cmdFinetuneCUDA(modelPath, dataPath string, steps int, lr float64, logEvery int) {
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
	if !mongoose.NeedleSparseLoaded() {
		log.Fatal("KNeedleSparse kernel not loaded")
	}
	if !mongoose.SoftmaxCELoaded() {
		log.Fatal("softmax+CE kernel not loaded")
	}

	// Xe daemon — optional coprocessor for CE
	xe := mongoose.NewXeDaemon()
	if xe != nil {
		defer xe.Close()
		if xe.HasArena() {
			log.Printf("[finetune] Xe: %s, arena: %d MB", xe.Name(), 256)
		}
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
	precision := "int8" // needle always operates on INT8

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

	log.Printf("[finetune] loading %s (%s precision)", modelPath, precision)

	type weight struct {
		q8       *mongoose.Int8Tensor
		mom      *mongoose.Tensor
		delta    *mongoose.Tensor
		tracker  *mongoose.ProjectionTracker
		rows     int
		cols     int
		nHot     int
	}

	loadWeight := func(name string, rows, cols int) weight {
		data, err := ms.ReadTensorFloat32(name)
		if err != nil {
			log.Fatalf("load %s: %v", name, err)
		}
		qt := gguf.QuantizeToInt8(data, rows, cols)
		q8 := cuda.FromHostInt8(&mongoose.QuantizedTensor{
			DataInt8: qt.DataInt8, Scales: qt.Scales, Shape: qt.Shape,
			Rows: qt.Rows, Cols: qt.Cols,
		})
		nh := calcMaxHot(rows, cols)
		mom := cuda.AllocFP16Tensor(nh, []int{nh})
		delta := cuda.AllocFP16Tensor(nh, []int{nh})
		mongoose.KZero(mom.DevicePtr(), nh*2)
		mongoose.KZero(delta.DevicePtr(), nh*2)
		return weight{
			q8: q8, mom: mom, delta: delta,
			tracker: mongoose.NewProjectionTracker(rows, cols, 10),
			rows: rows, cols: cols, nHot: nh,
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
		wq, wk, wv, wo, gate, up, down weight
		norm1, norm2                    *mongoose.Tensor
		bq, bk, bv                     *mongoose.Tensor // attention biases tiled to [n, outDim]
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
			wq:    loadWeight(pfx+"self_attn.q_proj.weight", dim, dim),
			wk:    loadWeight(pfx+"self_attn.k_proj.weight", kvDim, dim),
			wv:    loadWeight(pfx+"self_attn.v_proj.weight", kvDim, dim),
			wo:    loadWeight(pfx+"self_attn.o_proj.weight", dim, dim),
			gate:  loadWeight(pfx+"mlp.gate_proj.weight", ffnDim, dim),
			up:    loadWeight(pfx+"mlp.up_proj.weight", ffnDim, dim),
			down:  loadWeight(pfx+"mlp.down_proj.weight", dim, ffnDim),
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

	conductor := mongoose.NewConductor(vocabSize, 1)

	// Shared dequant buffer for INT8 path (reused per projection)
	maxElems := ffnDim * dim
	if dim*dim > maxElems {
		maxElems = dim * dim
	}
	var dequantBuf *mongoose.Tensor
	if precision == "int8" {
		dequantBuf = te.Zeros([]int{maxElems})
	}

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

	// Saved activations per layer (for backward recompute)
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

	// Backward gradient buffers (reused across layers)
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

	// Shared dW buffer (one projection at a time)
	dW := te.Zeros([]int{maxElems})

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	globalMaxHot := calcMaxHot(ffnDim, dim)
	hotIdxGPU := te.Zeros([]int{globalMaxHot})
	hotF := make([]float32, globalMaxHot)
	observeBuf := make([]float32, n*ffnDim)

	needleLR := float32(lr)

	needleFwd := func(w *weight, out, input *mongoose.Tensor, step int, seqN, inDim, outDim int) {
		// Observe input for column tracking
		if step >= 1 {
			inSize := seqN * inDim
			if len(observeBuf) < inSize {
				observeBuf = make([]float32, inSize)
			}
			cuda.DownloadSlice(input, 0, observeBuf[:inSize])
			w.tracker.ObserveInput(observeBuf[:inSize], seqN, inDim)
		}

		cuda.DequantToFP32(w.q8, dequantBuf.DevicePtr())

		if step > 1 {
			hotPos := w.tracker.HotPositions(w.nHot)
			nh := len(hotPos)
			if nh > 0 {
				for i := 0; i < nh; i++ {
					hotF[i] = math.Float32frombits(uint32(hotPos[i]))
				}
				cuda.UploadInto(hotIdxGPU, hotF[:nh])

				mongoose.KNeedleSparse(
					w.q8.DataPtr, w.q8.ScalePtr, dequantBuf.DevicePtr(),
					w.mom.DevicePtr(), w.delta.DevicePtr(), hotIdxGPU.DevicePtr(),
					nil, 1.0, needleLR, 0.9, 0.1,
					nh, w.cols)
			}
		}

		cuda.MatMulTransposeBTInto(out, input, dequantBuf, seqN, inDim, outDim)

		// Observe output for row tracking
		if step >= 1 {
			outSize := seqN * outDim
			if len(observeBuf) < outSize {
				observeBuf = make([]float32, outSize)
			}
			cuda.DownloadSlice(out, 0, observeBuf[:outSize])
			w.tracker.ObserveOutput(observeBuf[:outSize], seqN, outDim)
		}
	}

	xeName := "none"
	if xe != nil {
		xeName = xe.Name()
	}
	fmt.Println("ai finetune — needle + backward pass + sparse update")
	fmt.Printf("  engine:     %s (xe: %s)\n", eng.Name(), xeName)
	fmt.Printf("  model:      %s\n", modelPath)
	fmt.Printf("  data:       %s (%d tokens)\n", dataPath, len(tokens))
	fmt.Printf("  arch:       dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	fmt.Printf("  precision:  %s\n", precision)
	fmt.Printf("  training:   steps=%d lr=%.0e seq=%d\n", steps, lr, seqLen)
	fmt.Printf("  needle:     KNeedleSparse, %d max hot positions, compacted FP16 mom/delta\n", globalMaxHot)
	fmt.Println()

	type sparseCheckpoint struct {
		momData   map[string][]float32
		deltaData map[string][]float32
		loss      float32
		step      int
	}
	var ckpt *sparseCheckpoint
	bestFloor := float32(1e30)
	immuneActive := false
	floorContactStep := 0
	floorWindow := 10
	maxRecoveries := 20
	recoveryCount := 0

	fp32Scratch := te.Zeros([]int{globalMaxHot})

	allWeights := func(li int) []struct {
		name string
		w    *weight
	} {
		l := &lays[li]
		return []struct {
			name string
			w    *weight
		}{
			{"wq", &l.wq}, {"wk", &l.wk}, {"wv", &l.wv},
			{"wo", &l.wo}, {"gate", &l.gate}, {"up", &l.up}, {"down", &l.down},
		}
	}

	downloadFP16 := func(src *mongoose.Tensor, nElem int) []float32 {
		mongoose.KFP16ToFP32(src.DevicePtr(), fp32Scratch.DevicePtr(), nElem)
		cuda.Sync()
		host := te.ToHost(fp32Scratch)
		out := make([]float32, nElem)
		copy(out, host[:nElem])
		return out
	}

	uploadFP16 := func(dst *mongoose.Tensor, data []float32) {
		cuda.UploadInto(fp32Scratch, data)
		mongoose.KFP32ToFP16(fp32Scratch.DevicePtr(), dst.DevicePtr(), len(data))
	}

	saveCheckpoint := func(loss float32, step int) {
		ckpt = &sparseCheckpoint{
			momData:   make(map[string][]float32),
			deltaData: make(map[string][]float32),
			loss:      loss,
			step:      step,
		}
		cuda.Sync()
		for li := range lays {
			for _, pw := range allWeights(li) {
				key := fmt.Sprintf("%d.%s", li, pw.name)
				w := pw.w
				ckpt.momData[key] = downloadFP16(w.mom, w.nHot)
				ckpt.deltaData[key] = downloadFP16(w.delta, w.nHot)
			}
		}
	}

	restoreCheckpoint := func() {
		if ckpt == nil {
			return
		}
		for li := range lays {
			for _, pw := range allWeights(li) {
				key := fmt.Sprintf("%d.%s", li, pw.name)
				w := pw.w
				if momF, ok := ckpt.momData[key]; ok {
					uploadFP16(w.mom, momF)
				}
				if deltaF, ok := ckpt.deltaData[key]; ok {
					uploadFP16(w.delta, deltaF)
				}
			}
		}
	}

	gradUpdate := func(w *weight, dOut, activations *mongoose.Tensor, seqN, outDim, inDim int) {
		cuda.MatMulTransposeATInto(dW, dOut, activations, seqN, outDim, inDim)

		hotPos := w.tracker.HotPositions(w.nHot)
		nh := len(hotPos)
		if nh == 0 {
			return
		}
		for i := 0; i < nh; i++ {
			hotF[i] = math.Float32frombits(uint32(hotPos[i]))
		}
		cuda.UploadInto(hotIdxGPU, hotF[:nh])

		mongoose.KNeedleSparse(
			w.q8.DataPtr, w.q8.ScalePtr, nil,
			w.mom.DevicePtr(), w.delta.DevicePtr(), hotIdxGPU.DevicePtr(),
			dW.DevicePtr(), 0, needleLR, 0.9, 0.1,
			nh, w.cols)
	}

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

		conductor.Observe(tokI32)

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

			needleFwd(&l.wq, Q, normed, step, n, dim, dim)
			needleFwd(&l.wk, K, normed, step, n, dim, kvDim)
			needleFwd(&l.wv, V, normed, step, n, dim, kvDim)
			if l.bq != nil { mongoose.KAddInPlace(Q.DevicePtr(), l.bq.DevicePtr(), n*dim) }
			if l.bk != nil { mongoose.KAddInPlace(K.DevicePtr(), l.bk.DevicePtr(), n*kvDim) }
			if l.bv != nil { mongoose.KAddInPlace(V.DevicePtr(), l.bv.DevicePtr(), n*kvDim) }

			mongoose.KRoPE(Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			zero(attnOut)
			mongoose.KCausalAttentionGQA(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			needleFwd(&l.wo, dx, attnOut, step, n, dim, dim)
			te.AddInPlace(hidden, dx)

			cuda.CopyInto(savedXMid[li], hidden)

			zero(normed2)
			zero(rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed2.DevicePtr(),
				l.norm2.DevicePtr(), rmsScale2.DevicePtr(), n, dim)
			cuda.CopyInto(savedRMS2[li], rmsScale2)

			needleFwd(&l.gate, gatePre, normed2, step, n, dim, ffnDim)
			needleFwd(&l.up, upOut, normed2, step, n, dim, ffnDim)

			zero(ffnMid)
			mongoose.KSiLUGateMul(gatePre.DevicePtr(), upOut.DevicePtr(), ffnMid.DevicePtr(), n*ffnDim)

			needleFwd(&l.down, dx, ffnMid, step, n, ffnDim, dim)
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

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]

			// Recompute FFN activations from savedXMid
			zero(normed2)
			zero(rmsScale2)
			mongoose.KRMSNormOutSave(savedXMid[li].DevicePtr(), normed2.DevicePtr(),
				l.norm2.DevicePtr(), savedRMS2[li].DevicePtr(), n, dim)

			cuda.DequantToFP32(l.gate.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(gatePre, normed2, dequantBuf, n, dim, ffnDim)
			cuda.DequantToFP32(l.up.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(upOut, normed2, dequantBuf, n, dim, ffnDim)
			zero(ffnMid)
			mongoose.KSiLUGateMul(gatePre.DevicePtr(), upOut.DevicePtr(), ffnMid.DevicePtr(), n*ffnDim)

			// FFN backward: dW_down = dHidden^T @ ffnMid, needle accumulates
			gradUpdate(&l.down, dHidden, ffnMid, n, dim, ffnDim)

			// dFfnMid = dHidden @ down^T  (down is [dim, ffnDim])
			cuda.DequantToFP32(l.down.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dFfnMid, dHidden, dequantBuf, n, dim, ffnDim)

			// SiLU gate backward: dGate, dUp from dFfnMid
			mongoose.KSiLUGateBackward(dFfnMid.DevicePtr(), gatePre.DevicePtr(),
				upOut.DevicePtr(), dGate.DevicePtr(), dUp.DevicePtr(), n*ffnDim)

			// dW_gate, dW_up → needle accumulates
			gradUpdate(&l.gate, dGate, normed2, n, ffnDim, dim)
			gradUpdate(&l.up, dUp, normed2, n, ffnDim, dim)

			// dHidden through FFN: dNormed2 = dGate @ gate + dUp @ up
			cuda.DequantToFP32(l.gate.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch, dGate, dequantBuf, n, ffnDim, dim)
			cuda.DequantToFP32(l.up.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch2, dUp, dequantBuf, n, ffnDim, dim)
			mongoose.KAddInPlace(dScratch.DevicePtr(), dScratch2.DevicePtr(), dScratch.Size)

			// RMSNorm backward + residual add
			zero(dScratch2)
			mongoose.KRMSNormBackward(dScratch.DevicePtr(), savedXMid[li].DevicePtr(),
				l.norm2.DevicePtr(), savedRMS2[li].DevicePtr(), dScratch2.DevicePtr(), n, dim)
			mongoose.KAddInPlace(dHidden.DevicePtr(), dScratch2.DevicePtr(), dHidden.Size)

			// Recompute attention activations from savedXIn
			zero(normed)
			zero(rmsScale1)
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

			// Attention backward: dW_wo = dHidden^T @ attnOut, needle accumulates
			gradUpdate(&l.wo, dHidden, attnOut, n, dim, dim)

			// dAttnOut = dHidden @ wo^T
			cuda.DequantToFP32(l.wo.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dAttnOut, dHidden, dequantBuf, n, dim, dim)

			zero(dQ)
			zero(dK)
			zero(dV)
			mongoose.KCausalAttentionBackward(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), dAttnOut.DevicePtr(),
				dQ.DevicePtr(), dK.DevicePtr(), dV.DevicePtr(), n, dim, kvDim, heads, kvHeads)

			mongoose.KRoPEBackward(dQ.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPEBackward(dK.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			// dW for q/k/v → needle accumulates
			gradUpdate(&l.wq, dQ, normed, n, dim, dim)
			gradUpdate(&l.wk, dK, normed, n, kvDim, dim)
			gradUpdate(&l.wv, dV, normed, n, kvDim, dim)

			// dHidden through attention: dNormed1 = dQ @ wq + dK @ wk + dV @ wv
			cuda.DequantToFP32(l.wq.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch, dQ, dequantBuf, n, dim, dim)
			cuda.DequantToFP32(l.wk.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch2, dK, dequantBuf, n, kvDim, dim)
			mongoose.KAddInPlace(dScratch.DevicePtr(), dScratch2.DevicePtr(), dScratch.Size)
			cuda.DequantToFP32(l.wv.q8, dequantBuf.DevicePtr())
			cuda.MatMulTInto(dScratch2, dV, dequantBuf, n, kvDim, dim)
			mongoose.KAddInPlace(dScratch.DevicePtr(), dScratch2.DevicePtr(), dScratch.Size)

			// RMSNorm backward + residual add
			zero(dScratch2)
			mongoose.KRMSNormBackward(dScratch.DevicePtr(), savedXIn[li].DevicePtr(),
				l.norm1.DevicePtr(), savedRMS1[li].DevicePtr(), dScratch2.DevicePtr(), n, dim)
			mongoose.KAddInPlace(dHidden.DevicePtr(), dScratch2.DevicePtr(), dHidden.Size)
		}

		// === IMMUNE SYSTEM (helix pattern, momentum restore, floor-anchored checkpoint) ===

		// Rolling checkpoint: save when loss is near the best floor (anchored, won't drift)
		if !immuneActive && step > 1 && stepLoss > 0 && bestFloor < 1e20 {
			if stepLoss < bestFloor*1.5 {
				saveCheckpoint(stepLoss, step)
			}
		}

		// Floor detection
		if step > 1 && stepLoss > 0 && stepLoss < bestFloor {
			bestFloor = stepLoss
			saveCheckpoint(stepLoss, step)
			if !immuneActive {
				immuneActive = true
				floorContactStep = step
				recoveryCount = 0
			}
		}

		// Immune monitoring: window-based judgment after floor contact
		immuneSkip := false
		if immuneActive && step-floorContactStep >= floorWindow {
			rebound := stepLoss - bestFloor
			threshold := bestFloor * 0.15

			if rebound > threshold && recoveryCount < maxRecoveries && ckpt != nil {
				restoreCheckpoint()
				recoveryCount++
				immuneActive = false
				immuneSkip = true
				elapsed := time.Since(t0)
				fmt.Printf("step %5d/%d  loss=%.4f  lr=%.1e  %.0fs  (%.1f steps/s) [IMMUNE → floor %.4f rebound=%.3f]\n",
					step, steps, stepLoss, lr, elapsed.Seconds(),
					float64(step)/elapsed.Seconds(), bestFloor, rebound)
			} else {
				// Loss stayed near floor — accept current state
				immuneActive = false
				saveCheckpoint(stepLoss, step)
			}
		}

		if immuneSkip {
			continue
		}

		if step <= 3 || step%logEvery == 0 {
			hot, dead, _ := conductor.Stats()
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.4f  lr=%.1e  %.0fs  (%.1f steps/s)  vocab: %d hot, %d dead\n",
				step, steps, stepLoss, lr, elapsed.Seconds(),
				float64(step)/elapsed.Seconds(), hot, dead)
		}

		if step%1000 == 0 || step == steps {
			cuda.Sync()
			outDir := filepath.Join(filepath.Dir(modelPath), fmt.Sprintf("needle-step-%d", step))
			os.MkdirAll(outDir, 0755)
			w := gguf.NewGGUFWriter()
			w.AddString("general.architecture", "qwen2")
			w.AddUint32("qwen2.block_count", uint32(nLayers))
			w.AddTensorQ8_0("token_embd.weight", te.ToHost(embed), vocabSize, dim)
			w.AddTensorQ8_0("output.weight", te.ToHost(lmHead), vocabSize, dim)
			w.AddTensorF32("output_norm.weight", te.ToHost(finalNorm), dim)
			for l := 0; l < nLayers; l++ {
				pfx := fmt.Sprintf("blk.%d.", l)
				saveW := func(name string, wt weight) {
					cuda.DequantToFP32(wt.q8, dequantBuf.DevicePtr())
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
