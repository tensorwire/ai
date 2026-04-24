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

// cmdFinetuneNeedleXe fine-tunes a pretrained model using the full
// needle + helix + Xe + conductor architecture. Zero extra memory.
//
// Memory model:
//   - VRAM: INT8 weights only (~14GB for 14B). Needle updates in-place.
//   - L3 bridge (1MB): rung, signal, hot rows, batch, descriptors.
//   - Xe arena (256MB memfd): softmax+CE dispatch via descriptors.
//   - No backward pass. No LoRA. No Adam. No gradient buffers.
//
// Training loop:
//   1. CUDA forward: dequant INT8 → matmul → logits in VRAM
//   2. Xe: softmax+CE from arena descriptors → loss scalar
//   3. Helix ForwardOnlyStep: loss → rung + signal
//   4. Needle: reads rung from L3, updates INT8 in VRAM (hot rows only)
func cmdFinetuneNeedleXe(modelPath, dataPath string, steps int, lr float64, logEvery int) {
	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatal("TensorEngine not available")
	}
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok {
		log.Fatalf("finetune-needle-xe requires CUDA (detected: %s)", eng.Name())
	}
	if !mongoose.LoadKernels() {
		log.Fatal("CUDA kernels required")
	}

	// Xe daemon — optional, falls back to CUDA softmax+CE
	xe := mongoose.NewXeDaemon()
	if xe != nil {
		defer xe.Close()
		if xe.HasArena() {
			log.Printf("[needle-xe] Xe: %s, arena: %d MB", xe.Name(), 256)
		} else {
			log.Printf("[needle-xe] Xe arena unavailable, using CUDA CE fallback")
			xe.Close()
			xe = nil
		}
	} else {
		log.Printf("[needle-xe] no Xe daemon, using CUDA CE fallback")
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
	log.Printf("[needle-xe] BPE tokenized %d bytes → %d tokens (%.1fx compression)",
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

	log.Printf("[needle-xe] loading %s (INT8 needle + Xe + helix forward-only)", modelPath)

	// --- INT8 weights (needle updates in-place, no optimizer state) ---
	type q8Weight struct {
		q8         *mongoose.Int8Tensor
		rows, cols int
	}
	loadWeight := func(name string, rows, cols int) q8Weight {
		data, err := ms.ReadTensorFloat32(name)
		if err != nil {
			log.Fatalf("load %s: %v", name, err)
		}
		qt := gguf.QuantizeToInt8(data, rows, cols)
		q8 := cuda.FromHostInt8(&mongoose.QuantizedTensor{
			DataInt8: qt.DataInt8, Scales: qt.Scales, Shape: qt.Shape,
			Rows: qt.Rows, Cols: qt.Cols,
		})
		return q8Weight{q8: q8, rows: rows, cols: cols}
	}

	// Embeddings (FP32 for gather precision)
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

	// --- Load layers ---
	type layer struct {
		wq, wk, wv, wo, gate, up, down q8Weight
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
			log.Printf("[needle-xe] loaded layer %d/%d", l+1, nLayers)
		}
	}

	// --- Conductor ---
	conductor := mongoose.NewConductor(vocabSize, 1)

	// --- Shared dequant buffer ---
	maxElems := ffnDim * dim
	if dim*dim > maxElems {
		maxElems = dim * dim
	}
	dequantBuf := te.Zeros([]int{maxElems})

	// --- Activation buffers (single set, reused) ---
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

	// --- Shared needle buffers (compacted, resized per conductor window) ---
	// Mom/delta are compacted to [nHot × cols] and reset each window.
	// signalScale carries the training direction; momentum is just smoothing.
	maxHotRows := 256 // conductor typically gives ~200 hot rows
	maxCols := ffnDim
	if dim > maxCols {
		maxCols = dim
	}
	needleMom := te.Zeros([]int{maxHotRows * maxCols})   // FP16 compacted
	needleDelta := te.Zeros([]int{maxHotRows * maxCols})  // FP16 compacted
	hotIdxGPU := te.Zeros([]int{maxHotRows})              // int32 hot row indices

	// --- L3 bridge: descriptors + rung + batch ---
	// Layout: [batch tokens n*4] [batch targets n*4] [rung 6*4] [signalScale 4]
	batchOff := 0
	targOff := n * 4
	rungOff := targOff + n*4
	signalOff := rungOff + 6*4
	l3Size := signalOff + 4
	l3 := cuda.AllocL3Bridge(l3Size)

	var batchTokL3, batchTargL3, rungL3 []float32
	if l3 != nil {
		batchTokL3 = l3.Float32(batchOff, n)
		batchTargL3 = l3.Float32(targOff, n)
		rungL3 = l3.Float32(rungOff, 6)
	}

	// Arena layout constants (from mongoose/xe_client.go, linux-only)
	const arenaXeStart = 128*1024*1024 + 4096 // ArenaHalf + ArenaGuard

	// --- Xe arena: load cross-entropy SPIR-V kernel ---
	// The Xe daemon needs a softmax+CE kernel loaded.
	// Look for it relative to the binary.
	xeKernelIdx := -1
	for _, path := range []string{
		"./xe-daemon/softmax_ce.spv",
		filepath.Join(filepath.Dir(os.Args[0]), "xe-daemon", "softmax_ce.spv"),
		"/usr/local/share/mongoose/softmax_ce.spv",
	} {
		if _, err := os.Stat(path); err == nil {
			resp := xe.LoadSPIRV(path, "softmax_ce")
			if resp >= 0 {
				xeKernelIdx = resp
				log.Printf("[needle-xe] loaded Xe CE kernel: %s (idx=%d)", path, xeKernelIdx)
				break
			}
		}
	}
	if xeKernelIdx < 0 {
		log.Printf("[needle-xe] WARN: no Xe CE kernel found, falling back to CUDA softmax+CE")
	}

	// --- Helix optimizer (forward-only, no param registration) ---
	hlx := helix.NewHelixOptimizer(float32(lr), 0.9, 0.95, 1e-8, 0.1)

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	fmt.Println("ai finetune — needle + Xe + helix forward-only (zero extra memory)")
	fmt.Printf("  engine:     %s + %s\n", eng.Name(), xe.Name())
	fmt.Printf("  model:      %s\n", modelPath)
	fmt.Printf("  data:       %s (%d tokens)\n", dataPath, len(tokens))
	fmt.Printf("  arch:       dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	fmt.Printf("  training:   steps=%d lr=%.0e seq=%d (forward-only, no backward)\n", steps, lr, seqLen)
	fmt.Println()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	tokI32 := make([]int32, n)

	fmt.Println("Training...")
	t0 := time.Now()

	for step := 1; step <= steps; step++ {
		start := rng.Intn(len(tokens) - n - 1)
		for i := 0; i < n; i++ {
			tokI32[i] = int32(tokens[start+i])
		}

		// Batch to L3 + GPU
		if l3 != nil {
			for i := 0; i < n; i++ {
				batchTokL3[i] = math.Float32frombits(uint32(tokI32[i]))
				batchTargL3[i] = math.Float32frombits(uint32(int32(tokens[start+i+1])))
			}
			cuda.UploadInto(tokGPU, batchTokL3)
			cuda.UploadInto(targetsGPU, batchTargL3)
		} else {
			tokF := make([]float32, n)
			targF := make([]float32, n)
			for i := 0; i < n; i++ {
				tokF[i] = math.Float32frombits(uint32(tokI32[i]))
				targF[i] = math.Float32frombits(uint32(int32(tokens[start+i+1])))
			}
			cuda.UploadInto(tokGPU, tokF)
			cuda.UploadInto(targetsGPU, targF)
		}

		conductor.Observe(tokI32)

		// === FORWARD ONLY ===
		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		for li := range lays {
			l := &lays[li]

			zero(normed); zero(rmsScale1)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed.DevicePtr(),
				l.norm1.DevicePtr(), rmsScale1.DevicePtr(), n, dim)

			cuda.DequantToFP32(l.wq.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(Q, normed, dequantBuf, n, dim, dim)

			cuda.DequantToFP32(l.wk.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(K, normed, dequantBuf, n, dim, kvDim)

			cuda.DequantToFP32(l.wv.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(V, normed, dequantBuf, n, dim, kvDim)

			mongoose.KRoPE(Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			zero(attnOut)
			mongoose.KCausalAttentionGQA(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			cuda.DequantToFP32(l.wo.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(dx, attnOut, dequantBuf, n, dim, dim)
			te.AddInPlace(hidden, dx)

			zero(normed2); zero(rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed2.DevicePtr(),
				l.norm2.DevicePtr(), rmsScale2.DevicePtr(), n, dim)

			cuda.DequantToFP32(l.gate.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(gatePre, normed2, dequantBuf, n, dim, ffnDim)

			cuda.DequantToFP32(l.up.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(upOut, normed2, dequantBuf, n, dim, ffnDim)

			zero(ffnMid)
			mongoose.KSiLUGateMul(gatePre.DevicePtr(), upOut.DevicePtr(), ffnMid.DevicePtr(), n*ffnDim)

			cuda.DequantToFP32(l.down.q8, dequantBuf.DevicePtr())
			cuda.MatMulTransposeBTInto(dx, ffnMid, dequantBuf, n, ffnDim, dim)
			te.AddInPlace(hidden, dx)
		}

		zero(normedFinal); zero(finalScales)
		mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), n, dim)

		cuda.MatMulTransposeBTInto(logitsBuf, normedFinal, lmHead, n, dim, vocabSize)

		// === LOSS: Xe dispatch or CUDA fallback ===
		var stepLoss float32
		if xe != nil && xeKernelIdx >= 0 {
			// Xe path: dispatch softmax+CE via arena descriptors
			cuda.Sync()

			// Write logits + targets to arena Go region
			logitsOff := uint32(0)
			targetsOff := uint32(n * vocabSize * 4)
			lossesOff := uint32(arenaXeStart)
			gradOff := uint32(arenaXeStart + n*4)

			// Copy logits from VRAM to arena Go region
			logitsHost := te.ToHost(logitsBuf)
			goLogits := xe.GoRegion(int(logitsOff), n*vocabSize)
			copy(goLogits, logitsHost)

			targetsHost := te.ToHost(targetsGPU)
			goTargets := xe.GoRegionInt32(int(targetsOff), n)
			for i := 0; i < n; i++ {
				goTargets[i] = int32(math.Float32bits(targetsHost[i]))
			}

			invN := float32(1.0) / float32(n)
			xe.DispatchCrossEntropy(xeKernelIdx, logitsOff, targetsOff, lossesOff, gradOff,
				uint32(n), uint32(vocabSize), invN)
			xe.Sync()

			// Read losses from Xe region
			xeLosses := xe.XeRegion(int(lossesOff), n)
			for _, v := range xeLosses {
				stepLoss += v
			}
			stepLoss /= float32(n)
		} else {
			// CUDA fallback
			zero(lossesGPU)
			invN := float32(1.0) / float32(n)
			mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
				lossesGPU.DevicePtr(), nil, n, vocabSize, invN)
			cuda.Sync()
			lossH := te.ToHost(lossesGPU)
			for _, v := range lossH {
				stepLoss += v
			}
			stepLoss /= float32(n)
		}

		// === HELIX FORWARD-ONLY STEP ===
		// No backward. Loss scalar is the only signal.
		// First step is no-op (warmup — no momentum history).
		hlx.ForwardOnlyStep(step, stepLoss, float32(lr))

		// === NEEDLE: sparse INT8 weight update on conductor hot rows ===
		if step > 1 {
			signalScale := hlx.SignalScale()

			// Write rung to L3
			if l3 != nil {
				r := hlx.CurrentRung()
				rungL3[0] = r.Backbone1
				rungL3[1] = r.Glyco1
				rungL3[2] = r.Hbond1
				rungL3[3] = r.Hbond2
				rungL3[4] = r.Glyco2
				rungL3[5] = r.Backbone2
			}

			// Get conductor hot rows for embedding
			hotRows := conductor.HotRows()
			nHot := len(hotRows)
			if nHot > maxHotRows {
				nHot = maxHotRows
			}

			// Upload hot indices to GPU
			hotI32 := make([]float32, nHot)
			for i := 0; i < nHot; i++ {
				hotI32[i] = math.Float32frombits(uint32(hotRows[i]))
			}
			cuda.UploadInto(hotIdxGPU, hotI32)

			// Zero compacted mom/delta for this step's hot rows
			mongoose.KZero(needleMom.DevicePtr(), nHot*maxCols*2)
			mongoose.KZero(needleDelta.DevicePtr(), nHot*maxCols*2)

			// Update each layer's weights via sparse needle
			for li := range lays {
				l := &lays[li]
				needleUpdate := func(wt q8Weight) {
					mongoose.KNeedleSparse(
						wt.q8.DataPtr, wt.q8.ScalePtr, dequantBuf.DevicePtr(),
						needleMom.DevicePtr(), needleDelta.DevicePtr(), hotIdxGPU.DevicePtr(),
						signalScale, float32(lr), 0.9, 0.1,
						nHot, wt.cols)
				}
				needleUpdate(l.wq)
				needleUpdate(l.wk)
				needleUpdate(l.wv)
				needleUpdate(l.wo)
				needleUpdate(l.gate)
				needleUpdate(l.up)
				needleUpdate(l.down)
			}
			cuda.Sync()
		}

		if step <= 3 || step%logEvery == 0 {
			hot, dead, _ := conductor.Stats()
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s)  vocab: %d hot, %d dead\n",
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
				saveQ8 := func(name string, wt q8Weight) {
					cuda.DequantToFP32(wt.q8, dequantBuf.DevicePtr())
					fp32 := make([]float32, wt.rows*wt.cols)
					copy(fp32, te.ToHost(dequantBuf)[:wt.rows*wt.cols])
					w.AddTensorQ8_0(pfx+name, fp32, wt.rows, wt.cols)
				}
				saveQ8("attn_q.weight", lays[l].wq)
				saveQ8("attn_k.weight", lays[l].wk)
				saveQ8("attn_v.weight", lays[l].wv)
				saveQ8("attn_output.weight", lays[l].wo)
				saveQ8("ffn_gate.weight", lays[l].gate)
				saveQ8("ffn_up.weight", lays[l].up)
				saveQ8("ffn_down.weight", lays[l].down)
				w.AddTensorF32(pfx+"attn_norm.weight", te.ToHost(lays[l].norm1), dim)
				w.AddTensorF32(pfx+"ffn_norm.weight", te.ToHost(lays[l].norm2), dim)
			}
			outPath := filepath.Join(outDir, "model.gguf")
			if err := w.Write(outPath); err != nil {
				log.Printf("WARN: save checkpoint: %v", err)
			} else {
				log.Printf("[needle-xe] checkpoint saved: %s", outPath)
			}
		}
	}

	cuda.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.1fs (%.1f steps/s)\n",
		steps, total.Seconds(), float64(steps)/total.Seconds())
}
