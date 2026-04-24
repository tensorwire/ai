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

const maxHot = 49

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
	precision := profile.Precision

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

	// weight: holds either INT8 (q8 != nil) or FP32 (fp32 != nil).
	// Needle operates on whichever is present.
	// INT8: dequant → needle → matmul from shared buffer
	// FP32/FP16: needle modifies fp32 tensor directly, matmul from it
	type weight struct {
		q8    *mongoose.Int8Tensor // non-nil for INT8
		fp32  *mongoose.Tensor    // non-nil for FP32/FP16
		mom   *mongoose.Tensor
		delta *mongoose.Tensor
		rows  int
		cols  int
	}

	loadWeight := func(name string, rows, cols int) weight {
		data, err := ms.ReadTensorFloat32(name)
		if err != nil {
			log.Fatalf("load %s: %v", name, err)
		}
		nElem := maxHot * cols
		mom := cuda.AllocFP16Tensor(nElem, []int{maxHot, cols})
		delta := cuda.AllocFP16Tensor(nElem, []int{maxHot, cols})
		mongoose.KZero(mom.DevicePtr(), nElem*2)
		mongoose.KZero(delta.DevicePtr(), nElem*2)

		w := weight{mom: mom, delta: delta, rows: rows, cols: cols}
		if precision == "int8" {
			qt := gguf.QuantizeToInt8(data, rows, cols)
			w.q8 = cuda.FromHostInt8(&mongoose.QuantizedTensor{
				DataInt8: qt.DataInt8, Scales: qt.Scales, Shape: qt.Shape,
				Rows: qt.Rows, Cols: qt.Cols,
			})
		} else {
			w.fp32 = te.FromHost(data, []int{rows, cols})
		}
		return w
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

	l3 := cuda.AllocL3Bridge(6 * 4)
	var rungL3 []float32
	if l3 != nil {
		rungL3 = l3.Float32(0, 6)
	}

	hlx := helix.NewHelixOptimizer(float32(lr), 0.9, 0.95, 1e-8, 0.1)

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	type hotBuf struct {
		gpu *mongoose.Tensor
		n   int
	}
	hotByRows := make(map[int]*hotBuf)
	for _, rc := range []int{dim, kvDim, ffnDim} {
		hotByRows[rc] = &hotBuf{gpu: te.Zeros([]int{maxHot}), n: 0}
	}
	hotF := make([]float32, maxHot)

	uploadAllHotIndices := func() {
		hotRows := conductor.HotRows()
		for rc, hb := range hotByRows {
			seen := make(map[int32]bool)
			nh := 0
			for _, h := range hotRows {
				r := h % int32(rc)
				if !seen[r] && nh < maxHot {
					seen[r] = true
					hotF[nh] = math.Float32frombits(uint32(r))
					nh++
				}
			}
			hb.n = nh
			if nh > 0 {
				cuda.UploadInto(hb.gpu, hotF[:nh])
			}
		}
	}

	// needleFwd: dequant (if INT8) → KNeedleSparse on hot rows → matmul
	// For FP32/FP16 weights, needle operates on the weight tensor directly.
	needleFwd := func(w *weight, out, input *mongoose.Tensor, step, seqN, inDim, outDim int) {
		var matB *mongoose.Tensor
		if w.q8 != nil {
			cuda.DequantToFP32(w.q8, dequantBuf.DevicePtr())
			matB = dequantBuf
			if step > 1 {
				hb := hotByRows[w.rows]
				if hb != nil && hb.n > 0 {
					ss := hlx.SignalScale()
					mongoose.KNeedleSparse(
						w.q8.DataPtr, w.q8.ScalePtr, dequantBuf.DevicePtr(),
						w.mom.DevicePtr(), w.delta.DevicePtr(), hb.gpu.DevicePtr(),
						ss, float32(lr), 0.9, 0.1,
						hb.n, w.cols)
				}
			}
		} else {
			matB = w.fp32
			if step > 1 {
				hb := hotByRows[w.rows]
				if hb != nil && hb.n > 0 {
					ss := hlx.SignalScale()
					mongoose.KNeedleSparse(
						w.fp32.DevicePtr(), w.fp32.DevicePtr(), w.fp32.DevicePtr(),
						w.mom.DevicePtr(), w.delta.DevicePtr(), hb.gpu.DevicePtr(),
						ss, float32(lr), 0.9, 0.1,
						hb.n, w.cols)
				}
			}
		}
		cuda.MatMulTransposeBTInto(out, input, matB, seqN, inDim, outDim)
	}

	xeName := "none"
	if xe != nil {
		xeName = xe.Name()
	}
	fmt.Println("ai finetune — needle + helix forward-only")
	fmt.Printf("  engine:     %s (xe: %s)\n", eng.Name(), xeName)
	fmt.Printf("  model:      %s\n", modelPath)
	fmt.Printf("  data:       %s (%d tokens)\n", dataPath, len(tokens))
	fmt.Printf("  arch:       dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	fmt.Printf("  precision:  %s\n", precision)
	fmt.Printf("  training:   steps=%d lr=%.0e seq=%d\n", steps, lr, seqLen)
	fmt.Printf("  needle:     KNeedleSparse, %d hot rows, compacted FP16 mom/delta\n", maxHot)
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
	var prevLoss float32

	maxMomElems := maxHot * ffnDim
	if maxHot*dim > maxMomElems {
		maxMomElems = maxHot * dim
	}
	fp32Scratch := te.Zeros([]int{maxMomElems})

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

	saveHotRows := func(hotRows []int32, loss float32, step int) {
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
				nElem := maxHot * w.cols
				ckpt.momData[key] = downloadFP16(w.mom, nElem)
				ckpt.deltaData[key] = downloadFP16(w.delta, nElem)
			}
		}
	}

	restoreHotRows := func() {
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

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
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
		uploadAllHotIndices()

		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		for li := range lays {
			l := &lays[li]

			zero(normed)
			zero(rmsScale1)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed.DevicePtr(),
				l.norm1.DevicePtr(), rmsScale1.DevicePtr(), n, dim)

			needleFwd(&l.wq, Q, normed, step, n, dim, dim)
			needleFwd(&l.wk, K, normed, step, n, dim, kvDim)
			needleFwd(&l.wv, V, normed, step, n, dim, kvDim)

			mongoose.KRoPE(Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			zero(attnOut)
			mongoose.KCausalAttentionGQA(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			needleFwd(&l.wo, dx, attnOut, step, n, dim, dim)
			te.AddInPlace(hidden, dx)

			zero(normed2)
			zero(rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed2.DevicePtr(),
				l.norm2.DevicePtr(), rmsScale2.DevicePtr(), n, dim)

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
		invN := float32(1.0) / float32(n)
		mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
			lossesGPU.DevicePtr(), logitsBuf.DevicePtr(), n, vocabSize, invN)
		cuda.Sync()
		lossH := te.ToHost(lossesGPU)
		for _, v := range lossH {
			stepLoss += v
		}
		stepLoss /= float32(n)

		// === SPARSE IMMUNE SYSTEM ===
		hotRows := conductor.HotRows()

		if !immuneActive && step > 1 && stepLoss > 0 {
			if stepLoss < bestFloor*1.1 {
				saveHotRows(hotRows, stepLoss, step)
			}
		}

		if stepLoss > 0 && stepLoss < bestFloor {
			bestFloor = stepLoss
			if !immuneActive {
				immuneActive = true
				floorContactStep = step
				recoveryCount = 0
			}
		}

		immuneSkip := false
		if immuneActive && step-floorContactStep >= floorWindow {
			rebound := stepLoss - bestFloor
			threshold := bestFloor * 0.05
			if rebound > threshold && recoveryCount < maxRecoveries && ckpt != nil {
				restoreHotRows()
				recoveryCount++
				immuneActive = false
				immuneSkip = true
				elapsed := time.Since(t0)
				fmt.Printf("step %5d/%d  loss=%.4f  lr=%.1e  %.0fs  (%.1f steps/s) [IMMUNE → floor %.4f]\n",
					step, steps, stepLoss, lr, elapsed.Seconds(),
					float64(step)/elapsed.Seconds(), ckpt.loss)
			} else {
				immuneActive = false
			}
		}

		if immuneSkip {
			prevLoss = stepLoss
			continue
		}

		stepLR := float32(lr)
		if step > 1 && prevLoss > 0 {
			dLoss := float64(stepLoss) - float64(prevLoss)
			if dLoss > 0 {
				ratio := float32(dLoss / math.Max(float64(prevLoss), 1e-6))
				if ratio > 1.0 {
					ratio = 1.0
				}
				stepLR *= (1.0 - ratio)
			}
		}
		prevLoss = stepLoss

		hlx.ForwardOnlyStep(step, stepLoss, stepLR)

		if l3 != nil && step > 1 {
			r := hlx.CurrentRung()
			rungL3[0] = r.Backbone1
			rungL3[1] = r.Glyco1
			rungL3[2] = r.Hbond1
			rungL3[3] = r.Hbond2
			rungL3[4] = r.Glyco2
			rungL3[5] = r.Backbone2
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
					var fp32 []float32
					if wt.q8 != nil {
						cuda.DequantToFP32(wt.q8, dequantBuf.DevicePtr())
						host := te.ToHost(dequantBuf)
						fp32 = make([]float32, wt.rows*wt.cols)
						copy(fp32, host[:wt.rows*wt.cols])
					} else {
						fp32 = te.ToHost(wt.fp32)
					}
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
