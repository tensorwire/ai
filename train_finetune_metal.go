//go:build darwin && cgo

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"
	"unsafe"

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/helix"
	"github.com/tensorwire/mongoose"
	"github.com/tensorwire/tokenizer"
)

func cmdFinetuneMetalLoRA(modelPath, dataPath string, steps int, lr float64, rank int, logEvery int) {
	eng := selectEngine("auto")
	mtl, ok := eng.(*mongoose.Metal)
	if !ok {
		log.Fatalf("Metal finetune requires Metal GPU (detected: %s)", eng.Name())
	}
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatal("TensorEngine not available")
	}
	if !mtl.FusedTrainAvailable() {
		log.Fatal("fused_train.metallib required for Metal LoRA finetune")
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
	ropeTheta := float32(profile.RopeTheta)

	tok, err := tokenizer.LoadTokenizer(modelPath)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	raw, err := os.ReadFile(dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	tokens := tok.Encode(string(raw))
	log.Printf("[finetune-metal] %d bytes → %d tokens (%.1fx)",
		len(raw), len(tokens), float64(len(raw))/float64(len(tokens)))
	if len(tokens) < n+1 {
		log.Fatalf("need at least %d tokens, got %d", n+1, len(tokens))
	}

	log.Printf("[finetune-metal] loading %s (INT8 base + rank-%d LoRA + helix)", modelPath, rank)

	type metalLoRA struct {
		q8Data   *mongoose.Tensor // INT8 bytes via AllocRaw
		q8Scales *mongoose.Tensor // FP32 per-row scales
		a        *mongoose.Tensor // [outDim, rank] FP32
		b        *mongoose.Tensor // [rank, inDim] FP32
		am, av   *mongoose.Tensor // AdamW state for A
		bm, bv   *mongoose.Tensor // AdamW state for B
		rows, cols int
	}

	_ = float32(rank) // alpha/rank = 1.0 for standard LoRA (no scaling needed)

	quantizeToInt8 := func(fp32 []float32, rows, cols int) (int8Data []int8, scales []float32) {
		int8Data = make([]int8, rows*cols)
		scales = make([]float32, rows)
		for r := 0; r < rows; r++ {
			var absMax float32
			for c := 0; c < cols; c++ {
				v := fp32[r*cols+c]
				if v < 0 {
					v = -v
				}
				if v > absMax {
					absMax = v
				}
			}
			if absMax < 1e-10 {
				absMax = 1e-10
			}
			scales[r] = absMax
			invScale := float32(127.0) / absMax
			for c := 0; c < cols; c++ {
				qi := fp32[r*cols+c] * invScale
				if qi > 127 {
					qi = 127
				}
				if qi < -127 {
					qi = -127
				}
				int8Data[r*cols+c] = int8(qi)
			}
		}
		return
	}

	loadLoRA := func(name string, rows, cols int) metalLoRA {
		data, err := ms.ReadTensorFloat32(name)
		if err != nil {
			log.Fatalf("load %s: %v", name, err)
		}
		i8, sc := quantizeToInt8(data, rows, cols)
		nElems := rows * cols

		q8Data := mtl.AllocRaw(nElems, nElems, []int{rows, cols})
		mtl.UploadRaw(q8Data, unsafe.Pointer(&i8[0]), nElems)
		q8Scales := te.FromHost(sc, []int{rows})

		a := te.Zeros([]int{rows, rank})
		b := te.Zeros([]int{rank, cols})
		bData := make([]float32, rank*cols)
		scale := float32(math.Sqrt(2.0/float64(cols))) * 0.01
		rng := rand.New(rand.NewSource(time.Now().UnixNano()))
		for i := range bData {
			bData[i] = (rng.Float32()*2 - 1) * scale
		}
		mtl.UploadInto(b, bData)

		aSize := rows * rank
		bSize := rank * cols
		return metalLoRA{
			q8Data: q8Data, q8Scales: q8Scales,
			a: a, b: b,
			am: te.Zeros([]int{aSize}), av: te.Zeros([]int{aSize}),
			bm: te.Zeros([]int{bSize}), bv: te.Zeros([]int{bSize}),
			rows: rows, cols: cols,
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
		wq, wk, wv, wo, gate, up, down metalLoRA
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
			log.Printf("[finetune-metal] loaded layer %d/%d", l+1, nLayers)
		}
	}

	maxElems := ffnDim * dim
	if dim*dim > maxElems {
		maxElems = dim * dim
	}
	dequantBuf := te.Zeros([]int{maxElems})
	zeroDelta := te.Zeros([]int{maxElems})

	hidden := te.Zeros([]int{n, dim})
	scores := te.Zeros([]int{n * heads, n})
	normed := te.Zeros([]int{n, dim})
	Q := te.Zeros([]int{n, dim})
	K := te.Zeros([]int{n, kvDim})
	V := te.Zeros([]int{n, kvDim})
	attnOut := te.Zeros([]int{n, dim})
	normed2 := te.Zeros([]int{n, dim})
	gatePre := te.Zeros([]int{n, ffnDim})
	upOut := te.Zeros([]int{n, ffnDim})
	ffnMid := te.Zeros([]int{n, ffnDim})
	gateAct := te.Zeros([]int{n, ffnDim})
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
	dGatePre := te.Zeros([]int{n, ffnDim})
	dUpOut := te.Zeros([]int{n, ffnDim})

	dA := te.Zeros([]int{maxElems})
	dB := te.Zeros([]int{maxElems})
	dLoraTemp := te.Zeros([]int{n, ffnDim})
	loraRank := te.Zeros([]int{n, rank})

	targetsGPU := te.Zeros([]int{n})

	bar := func() { mtl.FusedBarrierBuffers() }

	dequant := func(w *metalLoRA) {
		mtl.FusedDequantDelta(w.q8Data, w.q8Scales, zeroDelta, dequantBuf, w.rows*w.cols, w.cols)
		bar()
	}

	loraFwd := func(w *metalLoRA, out, input *mongoose.Tensor, seqN, inDim, outDim int) {
		dequant(w)
		mtl.FusedGemmF32BT(input, dequantBuf, out, seqN, inDim, outDim)
		bar()
		mtl.FusedGemmF32BT(input, w.b, loraRank, seqN, inDim, rank)
		bar()
		mtl.FusedGemmF32BT(loraRank, w.a, dLoraTemp, seqN, rank, outDim)
		bar()
		mtl.FusedAddInPlace(out, dLoraTemp, seqN*outDim)
		bar()
	}

	loraBackward := func(w *metalLoRA, dOut, input *mongoose.Tensor, seqN, outDim, inDim int) {
		mtl.FusedGemmF32BT(input, w.b, loraRank, seqN, inDim, rank)
		bar()
		mtl.FusedGemmF32TN(dOut, loraRank, dA, seqN, outDim, rank)
		bar()
		mtl.FusedGemmF32NN(dOut, w.a, loraRank, seqN, outDim, rank)
		bar()
		mtl.FusedGemmF32TN(loraRank, input, dB, seqN, rank, inDim)
		bar()
	}

	hlx := helix.NewHelixOptimizer(float32(lr), 0.9, 0.95, 1e-8, 0.1)

	loraParamsPerLayer := 2 * rank * (dim + dim + kvDim + kvDim + dim + dim + ffnDim + ffnDim + dim + ffnDim)
	totalLoraParams := loraParamsPerLayer * nLayers
	loraVRAM := float64(totalLoraParams) * 4 * 3 / (1024 * 1024)

	arch := "llama"
	if cfgData, err := os.ReadFile(filepath.Join(modelPath, "config.json")); err == nil {
		var cfg map[string]interface{}
		if json.Unmarshal(cfgData, &cfg) == nil {
			if a, ok := cfg["architectures"].([]interface{}); ok && len(a) > 0 {
				s, _ := a[0].(string)
				switch {
				case contains(s, "Qwen"):
					arch = "qwen2"
				case contains(s, "Phi"):
					arch = "phi"
				case contains(s, "Gemma"):
					arch = "gemma"
				}
			}
		}
	}

	fmt.Println("ai finetune — Metal helix DNA + LoRA adapters")
	fmt.Printf("  engine:     Metal (%s)\n", eng.Name())
	fmt.Printf("  model:      %s\n", modelPath)
	fmt.Printf("  data:       %s (%d tokens)\n", dataPath, len(tokens))
	fmt.Printf("  arch:       dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	fmt.Printf("  training:   steps=%d lr=%.0e seq=%d rank=%d\n", steps, lr, seqLen, rank)
	fmt.Printf("  lora:       %d trainable params (%.0f MB optimizer state)\n", totalLoraParams, loraVRAM)
	fmt.Println()

	// Gradient clipping buffers
	gradSumSq := te.Zeros([]int{1})
	const gradMaxNorm = float32(1.0)

	// Per-layer saved gradients for two-phase backward: compute all grads, then clip, then optimize
	type savedGrads struct {
		dA, dB *mongoose.Tensor
	}
	layerGrads := make([]struct {
		down, gate, up, wo, wq, wk, wv savedGrads
		gateDA1, gateDB1               *mongoose.Tensor // saved for DNA pairing
		kvDA1, kvDB1                   *mongoose.Tensor
	}, nLayers)
	for li := range layerGrads {
		lg := &layerGrads[li]
		lg.down.dA = te.Zeros([]int{dim * rank})
		lg.down.dB = te.Zeros([]int{rank * ffnDim})
		lg.gate.dA = te.Zeros([]int{ffnDim * rank})
		lg.gate.dB = te.Zeros([]int{rank * dim})
		lg.up.dA = te.Zeros([]int{ffnDim * rank})
		lg.up.dB = te.Zeros([]int{rank * dim})
		lg.wo.dA = te.Zeros([]int{dim * rank})
		lg.wo.dB = te.Zeros([]int{rank * dim})
		lg.wq.dA = te.Zeros([]int{dim * rank})
		lg.wq.dB = te.Zeros([]int{rank * dim})
		lg.wk.dA = te.Zeros([]int{kvDim * rank})
		lg.wk.dB = te.Zeros([]int{rank * dim})
		lg.wv.dA = te.Zeros([]int{kvDim * rank})
		lg.wv.dB = te.Zeros([]int{rank * dim})
		lg.gateDA1 = te.Zeros([]int{ffnDim * rank})
		lg.gateDB1 = te.Zeros([]int{rank * dim})
		lg.kvDA1 = te.Zeros([]int{kvDim * rank})
		lg.kvDB1 = te.Zeros([]int{rank * dim})
	}

	// LR schedule: warmup + cosine decay
	warmupSteps := 1
	minLR := float32(lr) / 10.0
	getLR := func(step int) float32 {
		if step < warmupSteps {
			return float32(lr) * float32(step) / float32(warmupSteps)
		}
		progress := float64(step-warmupSteps) / float64(steps-warmupSteps)
		cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
		return minLR + float32(cosine)*float32(float32(lr)-minLR)
	}

	// Immune system state
	bestFloor := float32(1e30)
	immuneActive := false
	floorContactStep := 0
	floorWindow := 10
	maxRecoveries := 20
	recoveryCount := 0
	var prevLoss float32

	type loraCheckpoint struct {
		aData, bData [][]float32 // per-layer × 7 projections
		loss         float32
		step         int
	}
	var ckpt *loraCheckpoint

	saveLoRACheckpoint := func(loss float32, step int) {
		c := &loraCheckpoint{loss: loss, step: step}
		for _, l := range lays {
			for _, w := range []*metalLoRA{&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down} {
				aH := te.ToHost(w.a)
				bH := te.ToHost(w.b)
				aCopy := make([]float32, len(aH))
				bCopy := make([]float32, len(bH))
				copy(aCopy, aH)
				copy(bCopy, bH)
				c.aData = append(c.aData, aCopy)
				c.bData = append(c.bData, bCopy)
			}
		}
		ckpt = c
	}

	restoreLoRACheckpoint := func() {
		if ckpt == nil {
			return
		}
		idx := 0
		for _, l := range lays {
			for _, w := range []*metalLoRA{&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down} {
				mtl.UploadInto(w.a, ckpt.aData[idx])
				mtl.UploadInto(w.b, ckpt.bData[idx])
				idx++
			}
		}
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	embedShared := mtl.SharedSlice(embed)
	hiddenShared := mtl.SharedSlice(hidden)
	targetsShared := mtl.SharedSlice(targetsGPU)

	fmt.Println("Training...")
	t0 := time.Now()
	var stepLoss float32

	for step := 1; step <= steps; step++ {
		start := rng.Intn(len(tokens) - n - 1)

		for i := 0; i < n; i++ {
			tokID := tokens[start+i]
			copy(hiddenShared[i*dim:(i+1)*dim], embedShared[tokID*dim:(tokID+1)*dim])
			targetsShared[i] = math.Float32frombits(uint32(int32(tokens[start+i+1])))
		}

		// === IMMUNE SYSTEM ===
		if !immuneActive && step > 1 && stepLoss > 0 {
			if stepLoss < bestFloor*1.1 {
				saveLoRACheckpoint(stepLoss, step)
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
				restoreLoRACheckpoint()
				recoveryCount++
				immuneActive = false
				immuneSkip = true
				elapsed := time.Since(t0)
				fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s) [IMMUNE → floor %.3f]\n",
					step, steps, stepLoss, getLR(step), elapsed.Seconds(),
					float64(step)/elapsed.Seconds(), ckpt.loss)
			} else {
				immuneActive = false
			}
		}

		// Signal-scaled LR
		stepLR := getLR(step)
		if prevLoss > 0 {
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

		if immuneSkip {
			continue
		}

		// === FORWARD PASS ===
		// Ensure embedding gather is visible to GPU (upload from CPU to GPU tensor)
		mtl.UploadInto(hidden, hiddenShared[:n*dim])
		mtl.FusedBegin()

		for li := range lays {
			l := &lays[li]
			mtl.FusedCopy(savedXIn[li], hidden, n*dim); bar()
			mtl.FusedCopy(normed, hidden, n*dim); bar()
			mtl.FusedRMSNorm(normed, l.norm1, rmsScale1, n, dim); bar()
			mtl.FusedCopy(savedRMS1[li], rmsScale1, n); bar()

			loraFwd(&l.wq, Q, normed, n, dim, dim)
			loraFwd(&l.wk, K, normed, n, dim, kvDim)
			loraFwd(&l.wv, V, normed, n, dim, kvDim)
			if l.bq != nil { mtl.FusedAddInPlace(Q, l.bq, n*dim); bar() }
			if l.bk != nil { mtl.FusedAddInPlace(K, l.bk, n*kvDim); bar() }
			if l.bv != nil { mtl.FusedAddInPlace(V, l.bv, n*kvDim); bar() }

			mtl.FusedRoPE(Q, headDim, heads, ropeTheta, dim, n); bar()
			mtl.FusedRoPE(K, headDim, kvHeads, ropeTheta, kvDim, n); bar()
			mtl.FusedAttention(Q, K, V, attnOut, scores, dim, kvDim, headDim, heads, kvHeads, n); bar()

			loraFwd(&l.wo, dx, attnOut, n, dim, dim)
			mtl.FusedAddInPlace(hidden, dx, n*dim); bar()
			mtl.FusedCopy(savedXMid[li], hidden, n*dim); bar()

			mtl.FusedCopy(normed2, hidden, n*dim); bar()
			mtl.FusedRMSNorm(normed2, l.norm2, rmsScale2, n, dim); bar()
			mtl.FusedCopy(savedRMS2[li], rmsScale2, n); bar()

			loraFwd(&l.gate, gatePre, normed2, n, dim, ffnDim)
			loraFwd(&l.up, upOut, normed2, n, dim, ffnDim)
			mtl.FusedSiLUGateMul(gatePre, upOut, ffnMid, n*ffnDim); bar()

			loraFwd(&l.down, dx, ffnMid, n, ffnDim, dim)
			mtl.FusedAddInPlace(hidden, dx, n*dim); bar()
		}

		mtl.FusedCopy(normedFinal, hidden, n*dim); bar()
		mtl.FusedRMSNorm(normedFinal, finalNorm, finalScales, n, dim); bar()
		mtl.FusedEnd()

		// Compute logits: normedFinal @ lmHead^T → [n, vocabSize]
		logitsBuf := te.Zeros([]int{n, vocabSize})
		mtl.FusedBegin()
		mtl.FusedGemmF32BT(normedFinal, lmHead, logitsBuf, n, dim, vocabSize); bar()
		mtl.FusedEnd()

		// Softmax CE + gradient (standalone, CUDA-style: modifies logits, writes grad)
		lossesBuf := te.Zeros([]int{n})
		gradBuf := te.Zeros([]int{n, vocabSize})
		invN := float32(1.0) / float32(n)
		mtl.SoftmaxCEGrad(logitsBuf, targetsGPU, lossesBuf, gradBuf, n, vocabSize, invN)

		lossesH := te.ToHost(lossesBuf)
		stepLoss = 0
		for i := 0; i < n; i++ {
			stepLoss += lossesH[i]
		}
		stepLoss /= float32(n)

		// dHidden = gradBuf @ lmHead  [n,vocabSize] @ [vocabSize,dim] = [n,dim]
		mtl.FusedBegin()
		mtl.FusedGemmF32NN(gradBuf, lmHead, dHidden, n, vocabSize, dim); bar()
		mtl.FusedEnd()

		te.Release(logitsBuf)
		te.Release(lossesBuf)
		te.Release(gradBuf)

		// === BACKWARD PASS (compute all gradients, save per-layer) ===
		mtl.FusedBegin()
		mtl.FusedRMSNormBwd(dHidden, savedXMid[nLayers-1], finalNorm, finalScales, dScratch, n, dim); bar()
		mtl.FusedCopy(dHidden, dScratch, n*dim); bar()

		r, _, _, _ := hlx.PrepareStep(step, stepLoss, stepLR)

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]
			lg := &layerGrads[li]

			// Recompute FFN activations
			mtl.FusedCopy(normed2, savedXMid[li], n*dim); bar()
			mtl.FusedRMSNorm(normed2, l.norm2, savedRMS2[li], n, dim); bar()
			dequant(&l.gate)
			mtl.FusedGemmF32BT(normed2, dequantBuf, gatePre, n, dim, ffnDim); bar()
			dequant(&l.up)
			mtl.FusedGemmF32BT(normed2, dequantBuf, upOut, n, dim, ffnDim); bar()
			mtl.FusedSiLUGateMul(gatePre, upOut, ffnMid, n*ffnDim); bar()

			// Down backward — save grads
			loraBackward(&l.down, dHidden, ffnMid, n, dim, ffnDim)
			mtl.FusedCopy(lg.down.dA, dA, dim*rank); bar()
			mtl.FusedCopy(lg.down.dB, dB, rank*ffnDim); bar()

			// dHidden through frozen down
			dequant(&l.down)
			mtl.FusedGemmF32NN(dHidden, dequantBuf, dFfnMid, n, dim, ffnDim); bar()

			// SiLU backward
			mtl.SiLUGateBackward(dFfnMid, gatePre, upOut, gateAct, dGatePre, dUpOut); bar()
			mtl.FusedCopy(dGate, dGatePre, n*ffnDim); bar()
			mtl.FusedCopy(dUp, dUpOut, n*ffnDim); bar()

			// Gate backward — save grads
			loraBackward(&l.gate, dGate, normed2, n, ffnDim, dim)
			mtl.FusedCopy(lg.gateDA1, dA, ffnDim*rank); bar()
			mtl.FusedCopy(lg.gateDB1, dB, rank*dim); bar()

			// Up backward — save grads
			loraBackward(&l.up, dUp, normed2, n, ffnDim, dim)
			mtl.FusedCopy(lg.up.dA, dA, ffnDim*rank); bar()
			mtl.FusedCopy(lg.up.dB, dB, rank*dim); bar()

			// dHidden through FFN
			dequant(&l.gate)
			mtl.FusedGemmF32NN(dGate, dequantBuf, dScratch, n, ffnDim, dim); bar()
			dequant(&l.up)
			mtl.FusedGemmF32NN(dUp, dequantBuf, dScratch2, n, ffnDim, dim); bar()
			mtl.FusedAddInPlace(dScratch, dScratch2, n*dim); bar()
			mtl.FusedRMSNormBwd(dScratch, savedXMid[li], l.norm2, savedRMS2[li], dScratch2, n, dim); bar()
			mtl.FusedAddInPlace(dHidden, dScratch2, n*dim); bar()

			// Recompute attention
			mtl.FusedCopy(normed, savedXIn[li], n*dim); bar()
			mtl.FusedRMSNorm(normed, l.norm1, savedRMS1[li], n, dim); bar()
			dequant(&l.wq)
			mtl.FusedGemmF32BT(normed, dequantBuf, Q, n, dim, dim); bar()
			dequant(&l.wk)
			mtl.FusedGemmF32BT(normed, dequantBuf, K, n, dim, kvDim); bar()
			dequant(&l.wv)
			mtl.FusedGemmF32BT(normed, dequantBuf, V, n, dim, kvDim); bar()
			if l.bq != nil { mtl.FusedAddInPlace(Q, l.bq, n*dim); bar() }
			if l.bk != nil { mtl.FusedAddInPlace(K, l.bk, n*kvDim); bar() }
			if l.bv != nil { mtl.FusedAddInPlace(V, l.bv, n*kvDim); bar() }
			mtl.FusedRoPE(Q, headDim, heads, ropeTheta, dim, n); bar()
			mtl.FusedRoPE(K, headDim, kvHeads, ropeTheta, kvDim, n); bar()
			mtl.FusedAttention(Q, K, V, attnOut, scores, dim, kvDim, headDim, heads, kvHeads, n); bar()

			// WO backward — save grads
			loraBackward(&l.wo, dHidden, attnOut, n, dim, dim)
			mtl.FusedCopy(lg.wo.dA, dA, dim*rank); bar()
			mtl.FusedCopy(lg.wo.dB, dB, rank*dim); bar()

			dequant(&l.wo)
			mtl.FusedGemmF32NN(dHidden, dequantBuf, dAttnOut, n, dim, dim); bar()
			mtl.FusedAttentionBwdQ(dAttnOut, Q, K, V, scores, dQ, dK, dV,
				dim, kvDim, headDim, heads, kvHeads, n, n); bar()
mtl.FusedRoPE(dQ, headDim, heads, -ropeTheta, dim, n); bar()
			mtl.FusedRoPE(dK, headDim, kvHeads, -ropeTheta, kvDim, n); bar()

			// WQ backward — save grads
			loraBackward(&l.wq, dQ, normed, n, dim, dim)
			mtl.FusedCopy(lg.wq.dA, dA, dim*rank); bar()
			mtl.FusedCopy(lg.wq.dB, dB, rank*dim); bar()

			// WK backward — save grads
			loraBackward(&l.wk, dK, normed, n, kvDim, dim)
			mtl.FusedCopy(lg.kvDA1, dA, kvDim*rank); bar()
			mtl.FusedCopy(lg.kvDB1, dB, rank*dim); bar()

			// WV backward — save grads
			loraBackward(&l.wv, dV, normed, n, kvDim, dim)
			mtl.FusedCopy(lg.wv.dA, dA, kvDim*rank); bar()
			mtl.FusedCopy(lg.wv.dB, dB, rank*dim); bar()

			// dHidden through attention weights
			dequant(&l.wq)
			mtl.FusedGemmF32NN(dQ, dequantBuf, dScratch, n, dim, dim); bar()
			dequant(&l.wk)
			mtl.FusedGemmF32NN(dK, dequantBuf, dScratch2, n, kvDim, dim); bar()
			mtl.FusedAddInPlace(dScratch, dScratch2, n*dim); bar()
			dequant(&l.wv)
			mtl.FusedGemmF32NN(dV, dequantBuf, dScratch2, n, kvDim, dim); bar()
			mtl.FusedAddInPlace(dScratch, dScratch2, n*dim); bar()
			mtl.FusedRMSNormBwd(dScratch, savedXIn[li], l.norm1, savedRMS1[li], dScratch2, n, dim); bar()
			mtl.FusedAddInPlace(dHidden, dScratch2, n*dim); bar()
		}

		// === GRADIENT CLIPPING ===
		// Zero gradSumSq, accumulate norm across all saved LoRA gradients
		mtl.FusedZeroScalar(gradSumSq)
		for li := range lays {
			lg := &layerGrads[li]
			mtl.FusedGradNormSq(lg.down.dA, gradSumSq, dim*rank)
			mtl.FusedGradNormSq(lg.down.dB, gradSumSq, rank*ffnDim)
			mtl.FusedGradNormSq(lg.gateDA1, gradSumSq, ffnDim*rank)
			mtl.FusedGradNormSq(lg.gateDB1, gradSumSq, rank*dim)
			mtl.FusedGradNormSq(lg.up.dA, gradSumSq, ffnDim*rank)
			mtl.FusedGradNormSq(lg.up.dB, gradSumSq, rank*dim)
			mtl.FusedGradNormSq(lg.wo.dA, gradSumSq, dim*rank)
			mtl.FusedGradNormSq(lg.wo.dB, gradSumSq, rank*dim)
			mtl.FusedGradNormSq(lg.wq.dA, gradSumSq, dim*rank)
			mtl.FusedGradNormSq(lg.wq.dB, gradSumSq, rank*dim)
			mtl.FusedGradNormSq(lg.kvDA1, gradSumSq, kvDim*rank)
			mtl.FusedGradNormSq(lg.kvDB1, gradSumSq, rank*dim)
			mtl.FusedGradNormSq(lg.wv.dA, gradSumSq, kvDim*rank)
			mtl.FusedGradNormSq(lg.wv.dB, gradSumSq, rank*dim)
		}
		bar()
		bar()

		// Apply gradient clipping: each call reads gradSumSq[0], computes norm and clip scale internally
		for li := range lays {
			lg := &layerGrads[li]
			mtl.FusedGradClipScale(lg.down.dA, gradSumSq, gradMaxNorm, dim*rank)
			mtl.FusedGradClipScale(lg.down.dB, gradSumSq, gradMaxNorm, rank*ffnDim)
			mtl.FusedGradClipScale(lg.gateDA1, gradSumSq, gradMaxNorm, ffnDim*rank)
			mtl.FusedGradClipScale(lg.gateDB1, gradSumSq, gradMaxNorm, rank*dim)
			mtl.FusedGradClipScale(lg.up.dA, gradSumSq, gradMaxNorm, ffnDim*rank)
			mtl.FusedGradClipScale(lg.up.dB, gradSumSq, gradMaxNorm, rank*dim)
			mtl.FusedGradClipScale(lg.wo.dA, gradSumSq, gradMaxNorm, dim*rank)
			mtl.FusedGradClipScale(lg.wo.dB, gradSumSq, gradMaxNorm, rank*dim)
			mtl.FusedGradClipScale(lg.wq.dA, gradSumSq, gradMaxNorm, dim*rank)
			mtl.FusedGradClipScale(lg.wq.dB, gradSumSq, gradMaxNorm, rank*dim)
			mtl.FusedGradClipScale(lg.kvDA1, gradSumSq, gradMaxNorm, kvDim*rank)
			mtl.FusedGradClipScale(lg.kvDB1, gradSumSq, gradMaxNorm, rank*dim)
			mtl.FusedGradClipScale(lg.wv.dA, gradSumSq, gradMaxNorm, kvDim*rank)
			mtl.FusedGradClipScale(lg.wv.dB, gradSumSq, gradMaxNorm, rank*dim)
		}

		mtl.FusedEnd()

		// === OPTIMIZER (with clipped gradients) ===
		bc1 := float32(1.0 - math.Pow(0.9, float64(step)))
		bc2 := float32(1.0 - math.Pow(0.95, float64(step)))

		for li := range lays {
			l := &lays[li]
			lg := &layerGrads[li]

			// Down: unpaired AdamW
			mtl.AdamWT(l.down.a, lg.down.dA, l.down.am, l.down.av, stepLR, 0.1, step)
			mtl.AdamWT(l.down.b, lg.down.dB, l.down.bm, l.down.bv, stepLR, 0.1, step)

			// Gate + Up: DNA paired (G≡C, 3 H-bonds)
			mtl.DNARungGPU(l.gate.a, lg.gateDA1, l.gate.am, l.gate.av,
				l.up.a, lg.up.dA, l.up.am, l.up.av,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				3.0/5.0, stepLR, 0.9, 0.95, bc1, bc2, 1e-8, 0.1, ffnDim*rank)
			mtl.DNARungGPU(l.gate.b, lg.gateDB1, l.gate.bm, l.gate.bv,
				l.up.b, lg.up.dB, l.up.bm, l.up.bv,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				3.0/5.0, stepLR, 0.9, 0.95, bc1, bc2, 1e-8, 0.1, rank*dim)

			// WO: unpaired AdamW
			mtl.AdamWT(l.wo.a, lg.wo.dA, l.wo.am, l.wo.av, stepLR, 0.1, step)
			mtl.AdamWT(l.wo.b, lg.wo.dB, l.wo.bm, l.wo.bv, stepLR, 0.1, step)

			// WQ: unpaired AdamW
			mtl.AdamWT(l.wq.a, lg.wq.dA, l.wq.am, l.wq.av, stepLR, 0.1, step)
			mtl.AdamWT(l.wq.b, lg.wq.dB, l.wq.bm, l.wq.bv, stepLR, 0.1, step)

			// K + V: DNA paired (A=T, 2 H-bonds)
			mtl.DNARungGPU(l.wk.a, lg.kvDA1, l.wk.am, l.wk.av,
				l.wv.a, lg.wv.dA, l.wv.am, l.wv.av,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				2.0/5.0, stepLR, 0.9, 0.95, bc1, bc2, 1e-8, 0.1, kvDim*rank)
			mtl.DNARungGPU(l.wk.b, lg.kvDB1, l.wk.bm, l.wk.bv,
				l.wv.b, lg.wv.dB, l.wv.bm, l.wv.bv,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				2.0/5.0, stepLR, 0.9, 0.95, bc1, bc2, 1e-8, 0.1, rank*dim)
		}

		if step <= 3 || step%logEvery == 0 {
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.4f  floor=%.4f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, steps, stepLoss, bestFloor, stepLR, elapsed.Seconds(),
				float64(step)/elapsed.Seconds())
		}

		if step == steps {
			outDir := filepath.Join(filepath.Dir(modelPath), fmt.Sprintf("helix-step-%d", step))
			os.MkdirAll(outDir, 0755)
			w := gguf.NewGGUFWriter()
			w.AddString("general.architecture", arch)
			w.AddUint32(arch+".block_count", uint32(nLayers))
			w.AddTensorQ8_0("token_embd.weight", te.ToHost(embed), vocabSize, dim)
			w.AddTensorQ8_0("output.weight", te.ToHost(lmHead), vocabSize, dim)
			w.AddTensorF32("output_norm.weight", te.ToHost(finalNorm), dim)
			for l := 0; l < nLayers; l++ {
				pfx := fmt.Sprintf("blk.%d.", l)
				saveW := func(name string, wt metalLoRA) {
					dequant(&wt)
					merged := te.Zeros([]int{wt.rows * wt.cols})
					mtl.FusedBegin()
					mtl.FusedGemmF32NN(wt.a, wt.b, merged, wt.rows, rank, wt.cols); bar()
					mtl.FusedAddInPlace(dequantBuf, merged, wt.rows*wt.cols)
					mtl.FusedEnd()
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
				log.Printf("[finetune-metal] checkpoint saved: %s", outPath)
			}
		}
	}

	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.1fs (%.1f steps/s)  floor=%.4f\n",
		steps, total.Seconds(), float64(steps)/total.Seconds(), bestFloor)
}
