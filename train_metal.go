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

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/helix"
	"github.com/open-ai-org/mongoose"
)

func cmdTrainMetal() {
	fs := flag.NewFlagSet("train-metal", flag.ExitOnError)

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

	raw, err := os.ReadFile(*dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	conductor := mongoose.NewConductor(vocabSize, 100)
	hlx := helix.NewHelixOptimizer(lr, 0.9, 0.95, 1e-8, 0.1)

	// Weights on GPU (Metal shared memory)
	kaiming := func(rows, cols int) *mongoose.Tensor {
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d {
			d[i] = bound * (2*rand.Float32() - 1)
		}
		return te.FromHost(d, []int{rows, cols})
	}
	ones := func(sz int) *mongoose.Tensor {
		d := make([]float32, sz)
		for i := range d {
			d[i] = 1.0
		}
		return te.FromHost(d, []int{1, sz})
	}

	embedData := make([]float32, vocabSize*dim)
	for i := range embedData {
		embedData[i] = float32(rand.NormFloat64()) * 0.02
	}
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

	// AdamW state
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

	// Buffers
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
	dScratch := te.Zeros([]int{n, dim})
	scores := te.Zeros([]int{n * heads, n}) // attention scores buffer

	type fwdBuf struct {
		xIn, normed, Q, K, V, attnOut         *mongoose.Tensor
		xMid, normed2, gatePre, upOut, ffnMid *mongoose.Tensor
		rmsScale1, rmsScale2                  *mongoose.Tensor
		dFfnMid, dGate, dUp, dN2, dx         *mongoose.Tensor
		dAttnOut, dQ, dK, dV, dN1             *mongoose.Tensor
		dWDown, dWGate, dWUp, dWO, dWQ, dWK, dWV *mongoose.Tensor
		gateAct                               *mongoose.Tensor // SiLU(gate) for backward
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
			gateAct: te.Zeros([]int{n, ffnDim}),
		}
	}

	nParams := vocabSize*dim + dim
	for range lays {
		nParams += dim + dim*dim + kvDim*dim*2 + dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Println("ai train-metal — Metal fused compute kernels + Helix DNA optimizer")
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

	// LR schedule
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
	adamW := func(param, grad, mS, vS *mongoose.Tensor, step int) {
		mongoose.KAdamW(param.DevicePtr(), grad.DevicePtr(), mS.DevicePtr(), vS.DevicePtr(),
			curLR, 0.1, step, param.Size)
	}

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Sparse immune checkpoint
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

	// Disk checkpoints
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
				{"wq", lays[li].wq}, {"wk", lays[li].wk}, {"wv", lays[li].wv},
				{"wo", lays[li].wo}, {"gate", lays[li].gate}, {"up", lays[li].up},
				{"down", lays[li].down},
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
				{"wq", lays[li].wq}, {"wk", lays[li].wk}, {"wv", lays[li].wv},
				{"wo", lays[li].wo}, {"gate", lays[li].gate}, {"up", lays[li].up},
				{"down", lays[li].down},
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
				tmp := te.FromHost(wH, proj.w.Shape)
				mtl.FusedCopy(proj.w, tmp, proj.w.Size)
				te.Release(tmp)
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
			tensors[pfx+"self_attn.q_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wq), Shape: []int{dim, dim}}
			tensors[pfx+"self_attn.k_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wk), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.v_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wv), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.o_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wo), Shape: []int{dim, dim}}
			tensors[pfx+"mlp.gate_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].gate), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.up_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].up), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.down_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].down), Shape: []int{dim, ffnDim}}
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
			log.Printf("[checkpoint] immune: new best %.3f at step %d → %s", loss, step, stepDir)
		}
	}

	// CPU softmax CE gradient (small: n × vocabSize = 64×256 = 16K floats)
	cpuSoftmaxCEGrad := func() float32 {
		mtl.Sync()
		logitsH := te.ToHost(logitsBuf)
		targH := te.ToHost(targetsGPU)
		gradH := make([]float32, n*vocabSize)
		invN := float32(1.0) / float32(n)
		var totalLoss float32

		for pos := 0; pos < n; pos++ {
			off := pos * vocabSize
			target := int(math.Float32frombits(math.Float32bits(targH[pos])))

			// Max for numerical stability
			mx := logitsH[off]
			for v := 1; v < vocabSize; v++ {
				if logitsH[off+v] > mx {
					mx = logitsH[off+v]
				}
			}
			// Sum exp
			var se float32
			for v := 0; v < vocabSize; v++ {
				se += float32(math.Exp(float64(logitsH[off+v] - mx)))
			}
			// Loss + gradient
			prob := float32(math.Exp(float64(logitsH[off+target]-mx))) / se
			if prob < 1e-10 {
				prob = 1e-10
			}
			totalLoss += -float32(math.Log(float64(prob)))

			for v := 0; v < vocabSize; v++ {
				sv := float32(math.Exp(float64(logitsH[off+v]-mx))) / se * invN
				if v == target {
					sv -= invN
				}
				gradH[off+v] = sv
			}
		}

		// Upload gradient to GPU
		tmp := te.FromHost(gradH, []int{n, vocabSize})
		mtl.FusedCopy(gradGPU, tmp, n*vocabSize)
		te.Release(tmp)

		return totalLoss / float32(n)
	}

	fmt.Println("Training...")
	t0 := time.Now()

	var prevLoss float32
	for step := 1; step <= *stepsFlag; step++ {
		start := rng.Intn(len(data) - n - 1)
		tokF := make([]float32, n)
		targF := make([]float32, n)
		for i := 0; i < n; i++ {
			tokF[i] = math.Float32frombits(uint32(int32(data[start+i])))
			targF[i] = math.Float32frombits(uint32(int32(data[start+i+1])))
		}
		tmpTok := te.FromHost(tokF, []int{n})
		mtl.FusedCopy(tokGPU, tmpTok, n)
		te.Release(tmpTok)
		tmpTarg := te.FromHost(targF, []int{n})
		mtl.FusedCopy(targetsGPU, tmpTarg, n)
		te.Release(tmpTarg)

		tokIDs := make([]int32, n)
		for i := 0; i < n; i++ {
			tokIDs[i] = int32(data[start+i])
		}
		conductor.Observe(tokIDs)

		// === FORWARD ===
		mtl.FusedBegin()

		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		for li := range lays {
			l := &lays[li]
			b := &bufs[li]

			mtl.FusedCopy(b.xIn, hidden, n*dim)

			mtl.FusedRMSNorm(hidden, l.norm1, b.rmsScale1, n, dim)
			mtl.FusedCopy(b.normed, hidden, n*dim)
			// Restore hidden (RMSNorm was in-place on hidden, but we need pre-norm for residual)
			mtl.FusedCopy(hidden, b.xIn, n*dim)

			mtl.FusedGemmBT(b.normed, l.wq, b.Q, n, dim, dim)
			mtl.FusedGemmBT(b.normed, l.wk, b.K, n, dim, kvDim)
			mtl.FusedGemmBT(b.normed, l.wv, b.V, n, dim, kvDim)

			mtl.FusedRoPE(b.Q, headDim, heads, 10000.0, dim, n)
			mtl.FusedRoPE(b.K, headDim, kvHeads, 10000.0, kvDim, n)

			mtl.FusedAttention(b.Q, b.K, b.V, b.attnOut, scores, dim, kvDim, headDim, heads, kvHeads, n)

			mtl.FusedGemmBT(b.attnOut, l.wo, b.dx, n, dim, dim)
			mtl.FusedAddInPlace(hidden, b.dx, n*dim)

			mtl.FusedCopy(b.xMid, hidden, n*dim)

			mtl.FusedRMSNorm(hidden, l.norm2, b.rmsScale2, n, dim)
			mtl.FusedCopy(b.normed2, hidden, n*dim)
			mtl.FusedCopy(hidden, b.xMid, n*dim)

			mtl.FusedGemmBT(b.normed2, l.gate, b.gatePre, n, dim, ffnDim)
			mtl.FusedGemmBT(b.normed2, l.up, b.upOut, n, dim, ffnDim)

			mtl.FusedSiLUGateMul(b.gatePre, b.upOut, b.ffnMid, n*ffnDim)

			mtl.FusedGemmBT(b.ffnMid, l.down, b.dx, n, ffnDim, dim)
			mtl.FusedAddInPlace(hidden, b.dx, n*dim)
		}

		mtl.FusedRMSNorm(hidden, finalNorm, finalScales, n, dim)
		mtl.FusedCopy(normedFinal, hidden, n*dim)

		mtl.FusedGemmBT(normedFinal, embed, logitsBuf, n, dim, vocabSize)

		mtl.FusedEnd()

		// Loss + gradient on CPU (small: 64×256)
		stepLoss := cpuSoftmaxCEGrad()

		// === Sparse immune system ===
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

		// Disk checkpoint
		if stepLoss < ckptBestLoss && step-lastCkptStep >= 88 && step > 1 {
			ckptBestLoss = stepLoss
			lastCkptStep = step
			go saveFullCheckpoint(step, stepLoss)
		}

		// === HELIX DNA optimizer ===
		stepLR := getLR(step)
		r, _, _, _ := hlx.PrepareStep(step, stepLoss, stepLR)

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
		curLR = stepLR

		if immuneSkip {
			continue
		}

		// === BACKWARD ===
		mtl.FusedBegin()

		// dEmbed = gradGPU^T @ normedFinal
		mtl.FusedGemmTN(gradGPU, normedFinal, dEmbed, vocabSize, n, dim)
		// dHidden = gradGPU @ embed
		mtl.FusedGemmNN(gradGPU, embed, dHidden, n, vocabSize, dim)

		// Final RMSNorm backward
		mtl.FusedRMSNormBwd(dHidden, hidden, finalNorm, finalScales, dScratch, n, dim)
		mtl.FusedCopy(dHidden, dScratch, n*dim)

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]
			b := &bufs[li]

			// FFN backward
			mtl.FusedGemmNN(dHidden, l.down, b.dFfnMid, n, dim, ffnDim)
			mtl.FusedGemmTN(dHidden, b.ffnMid, b.dWDown, dim, n, ffnDim)

			mtl.SiLUGateBackward(b.dFfnMid, b.gatePre, b.upOut, b.gateAct, b.dGate, b.dUp)

			mtl.FusedGemmNN(b.dGate, l.gate, b.dN2, n, ffnDim, dim)
			mtl.FusedGemmNN(b.dUp, l.up, b.dx, n, ffnDim, dim)
			mtl.FusedAddInPlace(b.dN2, b.dx, n*dim)

			mtl.FusedGemmTN(b.dGate, b.normed2, b.dWGate, ffnDim, n, dim)
			mtl.FusedGemmTN(b.dUp, b.normed2, b.dWUp, ffnDim, n, dim)

			// RMSNorm2 backward
			mtl.FusedRMSNormBwd(b.dN2, b.xMid, l.norm2, b.rmsScale2, b.dx, n, dim)
			mtl.FusedAddInPlace(dHidden, b.dx, n*dim)

			// Attention backward
			mtl.FusedGemmNN(dHidden, l.wo, b.dAttnOut, n, dim, dim)
			mtl.FusedGemmTN(dHidden, b.attnOut, b.dWO, dim, n, dim)

			mtl.FusedAttentionBwdQ(b.dAttnOut, b.Q, b.K, b.V, scores,
				b.dQ, b.dK, b.dV, dim, kvDim, headDim, heads, kvHeads, n, n)

			// RoPE backward — same as forward with conjugate (negate sin)
			mtl.FusedRoPE(b.dQ, headDim, heads, -10000.0, dim, n)
			mtl.FusedRoPE(b.dK, headDim, kvHeads, -10000.0, kvDim, n)

			mtl.FusedGemmNN(b.dQ, l.wq, b.dN1, n, dim, dim)
			mtl.FusedGemmNN(b.dK, l.wk, b.dx, n, kvDim, dim)
			mtl.FusedGemmNN(b.dV, l.wv, b.dN2, n, kvDim, dim)
			mtl.FusedAddInPlace(b.dN1, b.dx, n*dim)
			mtl.FusedAddInPlace(b.dN1, b.dN2, n*dim)

			mtl.FusedGemmTN(b.dQ, b.normed, b.dWQ, dim, n, dim)
			mtl.FusedGemmTN(b.dK, b.normed, b.dWK, kvDim, n, dim)
			mtl.FusedGemmTN(b.dV, b.normed, b.dWV, kvDim, n, dim)

			// RMSNorm1 backward
			mtl.FusedRMSNormBwd(b.dN1, b.xIn, l.norm1, b.rmsScale1, b.dx, n, dim)
			mtl.FusedAddInPlace(dHidden, b.dx, n*dim)
		}

		mtl.FusedEnd()

		// === Optimizer ===
		for li := range lays {
			l := &lays[li]
			b := &bufs[li]
			la := &layAS[li]

			mongoose.KHelixDNAStep(
				l.gate.DevicePtr(), l.up.DevicePtr(),
				b.dWGate.DevicePtr(), b.dWUp.DevicePtr(),
				la.gate.m.DevicePtr(), la.up.m.DevicePtr(),
				la.gate.v.DevicePtr(), la.up.v.DevicePtr(),
				curLR, 0.9, 0.95, step, 1e-8, 0.1,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				3.0/5.0, l.gate.Size)

			if l.wq.Size == l.wk.Size {
				mongoose.KHelixDNAStep(
					l.wq.DevicePtr(), l.wk.DevicePtr(),
					b.dWQ.DevicePtr(), b.dWK.DevicePtr(),
					la.wq.m.DevicePtr(), la.wk.m.DevicePtr(),
					la.wq.v.DevicePtr(), la.wk.v.DevicePtr(),
					curLR, 0.9, 0.95, step, 1e-8, 0.1,
					r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
					2.0/5.0, l.wq.Size)
			} else {
				adamW(l.wq, b.dWQ, la.wq.m, la.wq.v, step)
				adamW(l.wk, b.dWK, la.wk.m, la.wk.v, step)
			}

			adamW(l.wv, b.dWV, la.wv.m, la.wv.v, step)
			adamW(l.wo, b.dWO, la.wo.m, la.wo.v, step)
			adamW(l.down, b.dWDown, la.down.m, la.down.v, step)
		}

		adamW(embed, dEmbed, embedAS.m, embedAS.v, step)

		if step <= 3 || step%*logEvery == 0 {
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  floor=%.3f  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, stepLoss, curLR, bestFloor, elapsed.Seconds(), float64(step)/elapsed.Seconds())
		}
	}

	mtl.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)  floor=%.3f\n",
		*stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds(), bestFloor)

	saveFullCheckpoint(*stepsFlag, bestFloor)
	_ = mtl
	_ = scores
}
