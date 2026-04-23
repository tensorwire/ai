//go:build darwin && cgo

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
	"unsafe"

	"encoding/json"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/helix"
	"github.com/open-ai-org/mongoose"
)

func cmdTrainMetal() {
	fs := flag.NewFlagSet("train-metal", flag.ExitOnError)

	dataPath := fs.String("data", "", "Training data (text file)")
	resumePath := fs.String("resume", "", "Resume from checkpoint directory")
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

	// Resume: load model shape from checkpoint config
	var resumeST *gguf.SafeTensors
	if *resumePath != "" {
		ckptDir := *resumePath
		if _, err := os.Stat(filepath.Join(ckptDir, "config.json")); err != nil {
			latest := findLatestCheckpoint(ckptDir)
			if latest == "" {
				log.Fatalf("No checkpoint found in %s", ckptDir)
			}
			ckptDir = latest
		}
		cfgData, err := os.ReadFile(filepath.Join(ckptDir, "config.json"))
		if err != nil {
			log.Fatalf("No config.json in %s", ckptDir)
		}
		var cfg map[string]interface{}
		json.Unmarshal(cfgData, &cfg)
		explicit := map[string]bool{}
		fs.Visit(func(f *flag.Flag) { explicit[f.Name] = true })

		cfgInt := func(flagName, cfgKey string, target *int) {
			if explicit[flagName] {
				return
			}
			if v, ok := cfg[cfgKey].(float64); ok {
				*target = int(v)
			}
		}
		cfgInt("dim", "hidden_size", dimFlag)
		cfgInt("layers", "num_hidden_layers", layersFlag)
		cfgInt("heads", "num_attention_heads", headsFlag)
		cfgInt("kv-heads", "num_key_value_heads", kvHeadsFlag)
		cfgInt("ffn-dim", "intermediate_size", ffnDimFlag)

		resumeST, err = gguf.OpenSafeTensors(ckptDir)
		if err != nil {
			log.Fatalf("Load checkpoint: %v", err)
		}
		fmt.Printf("Resuming from %s (dim=%d layers=%d)\n", ckptDir, *dimFlag, *layersFlag)
	}

	if *dataPath == "" { log.Fatal("data required: ai train data=<file>") }

	eng := selectEngine("metal")
	mtl, ok := eng.(*mongoose.Metal)
	if !ok {
		log.Fatal("train-metal requires Metal")
	}
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatal("TensorEngine not available")
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
	n := seqLen

	// OOM guard: estimate allocation before touching the GPU.
	// int8Param = 13 bytes/elem, grad = 4 bytes/elem, fwd bufs ≈ 4 bytes/elem
	{
		layerElems := dim*dim*2 + kvDim*dim*2 + dim*dim + ffnDim*dim*2 + dim*ffnDim
		estBytes := int64(layerElems) * 21 * int64(nLayers)
		estBytes += int64(vocabSize*dim) * 13
		estBytes += int64(n*dim) * 4 * 20 * int64(nLayers)
		estBytes += int64(n*heads*n) * 4
		estBytes += int64(n*vocabSize) * 4 * 3
		vram := eng.VRAM()
		headroom := uint64(float64(vram) * 0.75)
		if uint64(estBytes) > headroom {
			log.Fatalf("model needs ~%.0fMB but safe budget is %.0fMB (%.0fMB VRAM × 0.75) — reduce dim or layers",
				float64(estBytes)/(1024*1024), float64(headroom)/(1024*1024), float64(vram)/(1024*1024))
		}
	}

	raw, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	conductor := mongoose.NewConductor(vocabSize, 1)
	hlx := helix.NewHelixOptimizer(lr, 0.9, 0.95, 1e-8, 0.1)

	quantizeToInt8 := func(fp32 []float32, rows, cols int) (int8Data []int8, scales []float32, delta []float32) {
		int8Data = make([]int8, rows*cols)
		scales = make([]float32, rows)
		delta = make([]float32, rows*cols)
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
			scale := absMax / 127.0
			for c := 0; c < cols; c++ {
				idx := r*cols + c
				qi := fp32[idx] * invScale
				if qi > 127 {
					qi = 127
				}
				if qi < -127 {
					qi = -127
				}
				qr := float32(math.Round(float64(qi)))
				int8Data[idx] = int8(qr)
				delta[idx] = fp32[idx] - qr*scale
			}
		}
		return
	}

	loadOrInit := func(name string, rows, cols int) []float32 {
		if resumeST != nil && resumeST.HasTensor(name) {
			data, _, err := resumeST.ReadTensorFloat32(name)
			if err == nil && len(data) == rows*cols {
				return data
			}
			log.Printf("[resume] warning: %s load failed, using random init", name)
		}
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d {
			d[i] = bound * (2*rand.Float32() - 1)
		}
		return d
	}
	loadOrOnes := func(name string, sz int) *mongoose.Tensor {
		if resumeST != nil && resumeST.HasTensor(name) {
			data, _, err := resumeST.ReadTensorFloat32(name)
			if err == nil && len(data) == sz {
				return te.FromHost(data, []int{1, sz})
			}
		}
		d := make([]float32, sz)
		for i := range d {
			d[i] = 1.0
		}
		return te.FromHost(d, []int{1, sz})
	}

	type int8Param struct {
		data   *mongoose.Tensor
		scales *mongoose.Tensor
		delta  *mongoose.Tensor
		mom    *mongoose.Tensor
		vel    *mongoose.Tensor
		live   *mongoose.Tensor
		rows   int
		cols   int
	}

	makeInt8Param := func(name string, rows, cols int) int8Param {
		fp32 := loadOrInit(name, rows, cols)
		i8, sc, dl := quantizeToInt8(fp32, rows, cols)
		nElems := rows * cols

		dataT := mtl.AllocRaw(nElems, nElems, []int{rows, cols})
		mtl.UploadRaw(dataT, unsafe.Pointer(&i8[0]), nElems)

		scalesT := te.FromHost(sc, []int{rows})
		deltaT := te.FromHost(dl, []int{rows, cols})
		momT := mtl.AllocRaw(nElems*2, nElems, []int{rows, cols})
		velT := mtl.AllocRaw(nElems*2, nElems, []int{rows, cols})
		liveT := te.FromHost(fp32, []int{rows, cols})

		return int8Param{data: dataT, scales: scalesT, delta: deltaT, mom: momT, vel: velT, live: liveT, rows: rows, cols: cols}
	}

	embedFP32 := loadOrInit("model.embed_tokens.weight", vocabSize, dim)
	embedI8, embedSc, embedDl := quantizeToInt8(embedFP32, vocabSize, dim)

	embedData := mtl.AllocRaw(vocabSize*dim, vocabSize*dim, []int{vocabSize, dim})
	mtl.UploadRaw(embedData, unsafe.Pointer(&embedI8[0]), vocabSize*dim)
	embedScales := te.FromHost(embedSc, []int{vocabSize})
	embedDelta := te.FromHost(embedDl, []int{vocabSize, dim})
	embedMom := mtl.AllocRaw(vocabSize*dim*2, vocabSize*dim, []int{vocabSize, dim})
	embedVel := mtl.AllocRaw(vocabSize*dim*2, vocabSize*dim, []int{vocabSize, dim})
	embed := te.FromHost(embedFP32, []int{vocabSize, dim})

	finalNorm := loadOrOnes("model.norm.weight", dim)

	type layer struct {
		wq, wk, wv, wo, gate, up, down int8Param
		norm1, norm2                    *mongoose.Tensor
	}
	lays := make([]layer, nLayers)
	for l := range lays {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		lays[l] = layer{
			wq:    makeInt8Param(pfx+"self_attn.q_proj.weight", dim, dim),
			wk:    makeInt8Param(pfx+"self_attn.k_proj.weight", kvDim, dim),
			wv:    makeInt8Param(pfx+"self_attn.v_proj.weight", kvDim, dim),
			wo:    makeInt8Param(pfx+"self_attn.o_proj.weight", dim, dim),
			gate:  makeInt8Param(pfx+"mlp.gate_proj.weight", ffnDim, dim),
			up:    makeInt8Param(pfx+"mlp.up_proj.weight", ffnDim, dim),
			down:  makeInt8Param(pfx+"mlp.down_proj.weight", dim, ffnDim),
			norm1: loadOrOnes(pfx+"input_layernorm.weight", dim),
			norm2: loadOrOnes(pfx+"post_attention_layernorm.weight", dim),
		}
	}

	embedMask := mtl.NewHotRowMask(vocabSize)
	type layerMasks struct{ wq, wk, wv, wo, gate, up, down *mongoose.HotRowMask }
	layMasks := make([]layerMasks, nLayers)
	for li := range layMasks {
		layMasks[li] = layerMasks{
			wq: mtl.NewHotRowMask(dim), wk: mtl.NewHotRowMask(kvDim),
			wv: mtl.NewHotRowMask(kvDim), wo: mtl.NewHotRowMask(dim),
			gate: mtl.NewHotRowMask(ffnDim), up: mtl.NewHotRowMask(ffnDim),
			down: mtl.NewHotRowMask(dim),
		}
	}

	trackerWindow := 1
	type layerTrackers struct{ wq, wk, wv, wo, gate, up, down *mongoose.ProjectionTracker }
	layTrk := make([]layerTrackers, nLayers)
	for li := range layTrk {
		layTrk[li] = layerTrackers{
			wq: mongoose.NewProjectionTracker(dim, dim, trackerWindow),
			wk: mongoose.NewProjectionTracker(kvDim, dim, trackerWindow),
			wv: mongoose.NewProjectionTracker(kvDim, dim, trackerWindow),
			wo: mongoose.NewProjectionTracker(dim, dim, trackerWindow),
			gate: mongoose.NewProjectionTracker(ffnDim, dim, trackerWindow),
			up: mongoose.NewProjectionTracker(ffnDim, dim, trackerWindow),
			down: mongoose.NewProjectionTracker(dim, ffnDim, trackerWindow),
		}
	}

	hidden := te.Zeros([]int{n, dim})
	tokGPU := te.Zeros([]int{n})
	targetsGPU := te.Zeros([]int{n})
	normedFinal := te.Zeros([]int{n, dim})
	finalScales := te.Zeros([]int{n})
	lmMaxLogit := te.Zeros([]int{n})
	lmSumExp := te.Zeros([]int{n})
	lmLoss := te.Zeros([]int{1})
	dEmbed := te.Zeros([]int{vocabSize, dim})
	dHidden := te.Zeros([]int{n, dim})
	dScratch := te.Zeros([]int{n, dim})
	gradSumSq := te.Zeros([]int{1})
	clipScaleBuf := te.Zeros([]int{1})
	const gradMaxNorm = float32(1.0)

	// Forward activations — double-buffered for coiled pipeline.
	// Set 0 and set 1 alternate: forward writes to act[cur], backward reads act[prev].
	type actBuf struct {
		xIn, normed, Q, K, V, attnOut         *mongoose.Tensor
		xMid, normed2, gatePre, upOut, ffnMid *mongoose.Tensor
		rmsScale1, rmsScale2                  *mongoose.Tensor
		gateAct                               *mongoose.Tensor
	}
	acts := make([]actBuf, nLayers)
	for i := range acts {
		acts[i] = actBuf{
			xIn: te.Zeros([]int{n, dim}), normed: te.Zeros([]int{n, dim}),
			Q: te.Zeros([]int{n, dim}), K: te.Zeros([]int{n, kvDim}), V: te.Zeros([]int{n, kvDim}),
			attnOut: te.Zeros([]int{n, dim}),
			xMid: te.Zeros([]int{n, dim}), normed2: te.Zeros([]int{n, dim}),
			gatePre: te.Zeros([]int{n, ffnDim}), upOut: te.Zeros([]int{n, ffnDim}),
			ffnMid: te.Zeros([]int{n, ffnDim}),
			rmsScale1: te.Zeros([]int{n}), rmsScale2: te.Zeros([]int{n}),
			gateAct: te.Zeros([]int{n, ffnDim}),
		}
	}
	scores := te.Zeros([]int{n * heads, n})

	// Backward scratch — single set, only used by backward queue
	type bwdBuf struct {
		dFfnMid, dGate, dUp, dN2, dx         *mongoose.Tensor
		dAttnOut, dQ, dK, dV, dN1             *mongoose.Tensor
		dWDown, dWGate, dWUp, dWO, dWQ, dWK, dWV *mongoose.Tensor
	}
	bwds := make([]bwdBuf, nLayers)
	for i := range bwds {
		bwds[i] = bwdBuf{
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

	nParams := vocabSize*dim + dim
	for range lays {
		nParams += dim + dim*dim + kvDim*dim*2 + dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
	}

	// Mutable constant buffers — CPU writes per step, ICB reads via buffer binding.
	lrBuf := te.Zeros([]int{1})
	bc1Buf := te.Zeros([]int{1})
	bc2Buf := te.Zeros([]int{1})
	maxNormBuf := te.FromHost([]float32{gradMaxNorm}, []int{1})
	bb1Buf := te.Zeros([]int{1})
	gly1Buf := te.Zeros([]int{1})
	hb1Buf := te.Zeros([]int{1})
	hb2Buf := te.Zeros([]int{1})
	gly2Buf := te.Zeros([]int{1})
	bb2Buf := te.Zeros([]int{1})
	bondStrBuf := te.Zeros([]int{1})

	// Build ICB
	icbActs := make([]mongoose.ICBLayerActs, nLayers)
	icbWeights := make([]mongoose.ICBLayerWeights, nLayers)
	icbBwds := make([]mongoose.ICBLayerBwd, nLayers)
	for li := range lays {
		a := &acts[li]
		icbActs[li] = mongoose.ICBLayerActs{
			XIn: a.xIn, Normed: a.normed, Q: a.Q, K: a.K, V: a.V, AttnOut: a.attnOut,
			XMid: a.xMid, Normed2: a.normed2, GatePre: a.gatePre, UpOut: a.upOut, FfnMid: a.ffnMid,
			RmsScale1: a.rmsScale1, RmsScale2: a.rmsScale2, GateAct: a.gateAct,
		}
		l := &lays[li]
		lm := &layMasks[li]
		icbWeights[li] = mongoose.ICBLayerWeights{
			WQ:   mongoose.ICBLayerInt8{Data: l.wq.data, Scales: l.wq.scales, Delta: l.wq.delta, Mom: l.wq.mom, Vel: l.wq.vel, Live: l.wq.live, Mask: lm.wq},
			WK:   mongoose.ICBLayerInt8{Data: l.wk.data, Scales: l.wk.scales, Delta: l.wk.delta, Mom: l.wk.mom, Vel: l.wk.vel, Live: l.wk.live, Mask: lm.wk},
			WV:   mongoose.ICBLayerInt8{Data: l.wv.data, Scales: l.wv.scales, Delta: l.wv.delta, Mom: l.wv.mom, Vel: l.wv.vel, Live: l.wv.live, Mask: lm.wv},
			WO:   mongoose.ICBLayerInt8{Data: l.wo.data, Scales: l.wo.scales, Delta: l.wo.delta, Mom: l.wo.mom, Vel: l.wo.vel, Live: l.wo.live, Mask: lm.wo},
			Gate: mongoose.ICBLayerInt8{Data: l.gate.data, Scales: l.gate.scales, Delta: l.gate.delta, Mom: l.gate.mom, Vel: l.gate.vel, Live: l.gate.live, Mask: lm.gate},
			Up:   mongoose.ICBLayerInt8{Data: l.up.data, Scales: l.up.scales, Delta: l.up.delta, Mom: l.up.mom, Vel: l.up.vel, Live: l.up.live, Mask: lm.up},
			Down: mongoose.ICBLayerInt8{Data: l.down.data, Scales: l.down.scales, Delta: l.down.delta, Mom: l.down.mom, Vel: l.down.vel, Live: l.down.live, Mask: lm.down},
			Norm1: l.norm1, Norm2: l.norm2,
		}
		b := &bwds[li]
		icbBwds[li] = mongoose.ICBLayerBwd{
			DFfnMid: b.dFfnMid, DGate: b.dGate, DUp: b.dUp, DN2: b.dN2, Dx: b.dx,
			DAttnOut: b.dAttnOut, DQ: b.dQ, DK: b.dK, DV: b.dV, DN1: b.dN1,
			DWDown: b.dWDown, DWGate: b.dWGate, DWUp: b.dWUp, DWO: b.dWO, DWQ: b.dWQ, DWK: b.dWK, DWV: b.dWV,
		}
	}
	embedInt8 := mongoose.ICBLayerInt8{Data: embedData, Scales: embedScales, Delta: embedDelta, Mom: embedMom, Vel: embedVel, Live: embed, Mask: embedMask}

	mtl.ICBBuildTraining(
		dim, kvDim, headDim, heads, kvHeads, ffnDim, vocabSize, n, nLayers,
		hidden, normedFinal, finalNorm, finalScales,
		lmMaxLogit, lmSumExp, lmLoss, targetsGPU,
		dHidden, dScratch, dEmbed,
		gradSumSq, clipScaleBuf, scores,
		embed, embedInt8,
		icbActs, icbWeights, icbBwds,
		lrBuf, bc1Buf, bc2Buf, maxNormBuf,
		bb1Buf, gly1Buf, hb1Buf, hb2Buf, gly2Buf, bb2Buf, bondStrBuf,
	)

	// Shared slices for writing per-step constants
	lrShared := mtl.SharedSlice(lrBuf)
	bc1Shared := mtl.SharedSlice(bc1Buf)
	bc2Shared := mtl.SharedSlice(bc2Buf)
	bb1Shared := mtl.SharedSlice(bb1Buf)
	gly1Shared := mtl.SharedSlice(gly1Buf)
	hb1Shared := mtl.SharedSlice(hb1Buf)
	hb2Shared := mtl.SharedSlice(hb2Buf)
	gly2Shared := mtl.SharedSlice(gly2Buf)
	bb2Shared := mtl.SharedSlice(bb2Buf)
	bondStrShared := mtl.SharedSlice(bondStrBuf)

	fmt.Println("ai train-metal — Metal ICB + Needle INT8 optimizer")
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  data:     %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  model:    dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	if nParams > 1e6 {
		fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6)
	} else {
		fmt.Printf("  params:   %.1fK\n", float64(nParams)/1e3)
	}
	fmt.Printf("  training: steps=%d lr=%.0e\n", *stepsFlag, *lrFlag)
	fmt.Println()

	totalSteps := *stepsFlag
	warmupSteps := 1
	minLR := lr / 10.0
	getLR := func(step int) float32 {
		if step < warmupSteps {
			return lr * float32(step) / float32(warmupSteps)
		}
		progress := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
		cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
		return minLR + float32(cosine)*float32(lr-minLR)
	}

	var curLR float32
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	type sparseCheckpoint struct {
		rows    []int32
		weights map[string][]float32
		loss    float32
		step    int
	}
	var ckpt *sparseCheckpoint
	bestFloor := float32(1e30)
	immuneActive := false
	floorContactStep := 0
	floorWindow := 10
	maxRecoveries := 20
	recoveryCount := 0

	ckptDir := filepath.Join(os.TempDir(), "ai-train-out", "checkpoints")
	if GlobalOutDir != "" {
		ckptDir = filepath.Join(GlobalOutDir, "checkpoints")
	}
	os.MkdirAll(ckptDir, 0755)
	lastCkptStep := 0
	ckptBestLoss := float32(999.0)

	saveHotRows := func(hotRows []int32, loss float32, step int) {
		ckpt = &sparseCheckpoint{
			rows:    make([]int32, len(hotRows)),
			weights: make(map[string][]float32),
			loss:    loss,
			step:    step,
		}
		copy(ckpt.rows, hotRows)
		mtl.Sync()
		for li := range lays {
			for _, proj := range []struct {
				name string
				w    *mongoose.Tensor
			}{
				{"wq", lays[li].wq.live}, {"wk", lays[li].wk.live}, {"wv", lays[li].wv.live},
				{"wo", lays[li].wo.live}, {"gate", lays[li].gate.live}, {"up", lays[li].up.live},
				{"down", lays[li].down.live},
			} {
				wH := te.ToHost(proj.w)
				cols := proj.w.Size / proj.w.Shape[0]
				key := fmt.Sprintf("%d.%s", li, proj.name)
				saved := make([]float32, 0, len(hotRows)*cols)
				for _, r := range hotRows {
					row := int(r)
					if row >= 0 && row < proj.w.Shape[0] {
						saved = append(saved, wH[row*cols:(row+1)*cols]...)
					}
				}
				ckpt.weights[key] = saved
			}
		}
	}

	restoreHotRows := func() {
		if ckpt == nil {
			return
		}
		for li := range lays {
			for _, proj := range []struct {
				name string
				w    *mongoose.Tensor
			}{
				{"wq", lays[li].wq.live}, {"wk", lays[li].wk.live}, {"wv", lays[li].wv.live},
				{"wo", lays[li].wo.live}, {"gate", lays[li].gate.live}, {"up", lays[li].up.live},
				{"down", lays[li].down.live},
			} {
				key := fmt.Sprintf("%d.%s", li, proj.name)
				saved := ckpt.weights[key]
				if len(saved) == 0 {
					continue
				}
				wH := te.ToHost(proj.w)
				cols := proj.w.Size / proj.w.Shape[0]
				idx := 0
				for _, r := range ckpt.rows {
					row := int(r)
					if row >= 0 && row < proj.w.Shape[0] && idx+cols <= len(saved) {
						copy(wH[row*cols:(row+1)*cols], saved[idx:idx+cols])
						idx += cols
					}
				}
				mtl.UploadInto(proj.w, wH)
			}
		}
	}

	saveFullCheckpoint := func(step int, loss float32) {
		mtl.Sync()
		tensors := map[string]gguf.SaveTensor{
			"model.embed_tokens.weight": {Data: te.ToHost(embed), Shape: []int{vocabSize, dim}},
			"model.norm.weight":         {Data: te.ToHost(finalNorm), Shape: []int{dim}},
		}
		for li := range lays {
			pfx := fmt.Sprintf("model.layers.%d.", li)
			tensors[pfx+"self_attn.q_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wq.live), Shape: []int{dim, dim}}
			tensors[pfx+"self_attn.k_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wk.live), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.v_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wv.live), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.o_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wo.live), Shape: []int{dim, dim}}
			tensors[pfx+"mlp.gate_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].gate.live), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.up_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].up.live), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.down_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].down.live), Shape: []int{dim, ffnDim}}
			tensors[pfx+"input_layernorm.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].norm1), Shape: []int{dim}}
			tensors[pfx+"post_attention_layernorm.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].norm2), Shape: []int{dim}}
		}
		stepDir := filepath.Join(ckptDir, fmt.Sprintf("step-%05d", step))
		os.MkdirAll(stepDir, 0755)
		stPath := filepath.Join(stepDir, "model.safetensors")
		if err := gguf.SaveSafeTensors(stPath, tensors); err != nil {
			log.Printf("[checkpoint] save error: %v", err)
		} else {
			cfgJSON := fmt.Sprintf(`{"architectures":["LlamaForCausalLM"],"hidden_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"intermediate_size":%d,"vocab_size":%d,"max_position_embeddings":2048,"rope_theta":10000.0,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":true}`,
				dim, nLayers, heads, kvHeads, ffnDim, vocabSize)
			os.WriteFile(filepath.Join(stepDir, "config.json"), []byte(cfgJSON), 0644)
			log.Printf("[checkpoint] saved step %d loss=%.3f → %s", step, loss, stepDir)
		}
	}

	embedShared := mtl.SharedSlice(embed)
	hiddenShared := mtl.SharedSlice(hidden)
	lmLossShared := mtl.SharedSlice(lmLoss)

	fmt.Println("Training...")
	t0 := time.Now()

	tokIDs := make([]int32, n)
	var prevLoss float32

	// Phase timing accumulators
	var tBatch, tFwd, tLoss, tObs, tBwd, tClip, tNeedle, tDequant, tImmune time.Duration
	var stepStart time.Time

	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)
		tokF := make([]float32, n)
		targF := make([]float32, n)
		stepStart = time.Now()
		ph := time.Now()
		for i := 0; i < n; i++ {
			tokF[i] = math.Float32frombits(uint32(int32(data[start+i])))
			targF[i] = math.Float32frombits(uint32(int32(data[start+i+1])))
			tokIDs[i] = int32(data[start+i])
		}
		mtl.UploadInto(tokGPU, tokF)
		mtl.UploadInto(targetsGPU, targF)
		conductor.Observe(tokIDs)

		for i := 0; i < n; i++ {
			tokID := data[start+i]
			copy(hiddenShared[i*dim:(i+1)*dim], embedShared[tokID*dim:(tokID+1)*dim])
		}
		tBatch += time.Since(ph)

		// Read loss from PREVIOUS step (one step stale, GPU already done).
		// Step 1 has no previous loss; use log(vocabSize) as initial estimate.
		stepLoss := float32(math.Log(float64(vocabSize)))
		if step > 1 {
			stepLoss = lmLossShared[0]
		}

		// === Immune system (uses previous step's loss) ===
		ph = time.Now()
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
				fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s) [IMMUNE → floor %.3f]\n",
					step, *stepsFlag, stepLoss, getLR(step), elapsed.Seconds(),
					float64(step)/elapsed.Seconds(), ckpt.loss)
			} else {
				immuneActive = false
			}
		}
		if stepLoss < ckptBestLoss && step-lastCkptStep >= 88 && step > 1 {
			ckptBestLoss = stepLoss
			lastCkptStep = step
			go saveFullCheckpoint(step, stepLoss)
		}

		// Signal-scaled LR
		stepLR := getLR(step)
		if prevLoss > 0 {
			dLoss := float64(stepLoss) - float64(prevLoss)
			if dLoss > 0 {
				ratio := float32(dLoss / math.Max(float64(prevLoss), 1e-6))
				if ratio > 1.0 { ratio = 1.0 }
				stepLR *= (1.0 - ratio)
			}
		}
		prevLoss = stepLoss
		curLR = stepLR
		tImmune += time.Since(ph)

		if immuneSkip { continue }

		// === Projection trackers + masks (use previous step's activations) ===
		ph = time.Now()
		if step > 2 {
			for li := range lays {
				a := &acts[li]
				layTrk[li].wq.ObserveOutput(mtl.SharedSlice(a.Q), n, dim)
				layTrk[li].wk.ObserveOutput(mtl.SharedSlice(a.K), n, kvDim)
				layTrk[li].wv.ObserveOutput(mtl.SharedSlice(a.V), n, kvDim)
				layTrk[li].wo.ObserveOutput(mtl.SharedSlice(a.attnOut), n, dim)
				layTrk[li].gate.ObserveOutput(mtl.SharedSlice(a.gatePre), n, ffnDim)
				layTrk[li].up.ObserveOutput(mtl.SharedSlice(a.upOut), n, ffnDim)
				layTrk[li].down.ObserveOutput(mtl.SharedSlice(a.ffnMid), n, ffnDim)
			}
		}
		embedMask.Set(conductor.HotRows())
		for li := range lays {
			layMasks[li].wq.Set(layTrk[li].wq.HotRows())
			layMasks[li].wk.Set(layTrk[li].wk.HotRows())
			layMasks[li].wv.Set(layTrk[li].wv.HotRows())
			layMasks[li].wo.Set(layTrk[li].wo.HotRows())
			layMasks[li].gate.Set(layTrk[li].gate.HotRows())
			layMasks[li].up.Set(layTrk[li].up.HotRows())
			layMasks[li].down.Set(layTrk[li].down.HotRows())
		}
		tObs += time.Since(ph)

		// === Helix rung geometry ===
		r, bc1, bc2, rewound := hlx.PrepareStep(step, stepLoss, curLR)
		if rewound {
			if step <= 3 || step%*logEvery == 0 {
				elapsed := time.Since(t0)
				fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  %.0fs  (%.1f steps/s) [helix-rewound]\n",
					step, *stepsFlag, stepLoss, curLR, bestFloor, elapsed.Seconds(), float64(step)/elapsed.Seconds())
			}
			continue
		}

		// Write per-step constants to shared memory (ICB reads via buffer binding)
		lrShared[0] = curLR
		bc1Shared[0] = bc1
		bc2Shared[0] = bc2
		bb1Shared[0] = r.Backbone1
		gly1Shared[0] = r.Glyco1
		hb1Shared[0] = r.Hbond1
		hb2Shared[0] = r.Hbond2
		gly2Shared[0] = r.Glyco2
		bb2Shared[0] = r.Backbone2
		bondStrShared[0] = 3.0/5.0 // GC bond for paired, singles use clipScale directly

		// Zero loss scalar (pass2 uses atomic add)
		lmLossShared[0] = 0

		// === ICB EXECUTE ===
		ph = time.Now()
		if step == 1 {
			mtl.ICBExecuteFwd()
			tFwd += time.Since(ph)
			if step <= 3 || step%*logEvery == 0 {
				mtl.Sync()
				elapsed := time.Since(t0)
				fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  %.0fs  (%.1f steps/s) [noop]\n",
					step, *stepsFlag, lmLossShared[0], curLR, bestFloor, elapsed.Seconds(), float64(step)/elapsed.Seconds())
			}
			continue
		}

		mtl.ICBExecuteFull()
		tFwd += time.Since(ph)

		stepDur := time.Since(stepStart)
		if step <= 3 || step%*logEvery == 0 || step == *stepsFlag {
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  %.0fs  (%.1f avg, %.1f this)\n",
				step, *stepsFlag, stepLoss, curLR, bestFloor, elapsed.Seconds(),
				float64(step)/elapsed.Seconds(), 1.0/stepDur.Seconds())
		}
		_ = r
	}

	mtl.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)  floor=%.3f\n",
		*stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds(), bestFloor)
	fmt.Printf("\nphase breakdown (wall time, includes GPU wait):\n")
	all := tBatch + tFwd + tLoss + tImmune + tObs + tBwd + tClip + tNeedle + tDequant
	pct := func(d time.Duration) string { return fmt.Sprintf("%.1f%%", float64(d)/float64(all)*100) }
	fmt.Printf("  batch:   %6.1fms  %s\n", float64(tBatch.Microseconds())/1000, pct(tBatch))
	fmt.Printf("  forward: %6.1fms  %s\n", float64(tFwd.Microseconds())/1000, pct(tFwd))
	fmt.Printf("  loss:    %6.1fms  %s\n", float64(tLoss.Microseconds())/1000, pct(tLoss))
	fmt.Printf("  immune:  %6.1fms  %s\n", float64(tImmune.Microseconds())/1000, pct(tImmune))
	fmt.Printf("  observe: %6.1fms  %s\n", float64(tObs.Microseconds())/1000, pct(tObs))
	fmt.Printf("  bwd:     %6.1fms  %s\n", float64(tBwd.Microseconds())/1000, pct(tBwd))
	fmt.Printf("  clip:    %6.1fms  %s\n", float64(tClip.Microseconds())/1000, pct(tClip))
	fmt.Printf("  needle:  %6.1fms  %s\n", float64(tNeedle.Microseconds())/1000, pct(tNeedle))
	fmt.Printf("  dequant: %6.1fms  %s\n", float64(tDequant.Microseconds())/1000, pct(tDequant))
	fmt.Printf("  total:   %6.1fms\n", float64(all.Microseconds())/1000)

	saveFullCheckpoint(*stepsFlag, bestFloor)
	embedMask.Release()
	for li := range layMasks {
		layMasks[li].wq.Release()
		layMasks[li].wk.Release()
		layMasks[li].wv.Release()
		layMasks[li].wo.Release()
		layMasks[li].gate.Release()
		layMasks[li].up.Release()
		layMasks[li].down.Release()
	}
	_ = mtl
	_ = scores
	_ = hlx
}
