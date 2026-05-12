package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unsafe"

	"github.com/tensorwire/mongoose"
	"github.com/tensorwire/tokenizer"
)

// cmdInferGPU runs inference with three tiers:
//   1. GPU-resident (TensorEngine + CUDA kernels) — weights + KV cache in VRAM
//   2. GPU-accelerated (Engine.MatMul) — weights streamed, matmul on GPU (Metal/CUDA/WebGPU)
//   3. CPU fallback — pure Go
func cmdInferGPU(model string, promptParts []string) {
	ctxOverride := 0
	var filtered []string
	for i := 0; i < len(promptParts); i++ {
		if promptParts[i] == "--ctx" && i+1 < len(promptParts) {
			fmt.Sscanf(promptParts[i+1], "%d", &ctxOverride)
			i++
		} else {
			filtered = append(filtered, promptParts[i])
		}
	}
	promptParts = filtered
	prompt := strings.Join(promptParts, " ")
	path := resolveModel(model)

	configData, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		log.Fatalf("no config.json in %s", path)
	}
	var cfg map[string]interface{}
	json.Unmarshal(configData, &cfg)

	getInt := func(key string, def int) int {
		if v, ok := cfg[key].(float64); ok {
			return int(v)
		}
		return def
	}

	dim := getInt("hidden_size", 0)
	nLayers := getInt("num_hidden_layers", 0)
	heads := getInt("num_attention_heads", 0)
	kvHeads := getInt("num_key_value_heads", heads)
	ffnDim := getInt("intermediate_size", 0)
	vocabSize := getInt("vocab_size", 0)
	headDim := getInt("head_dim", dim/heads)
	kvDim := kvHeads * headDim
	attnDim := heads * headDim
	maxSeq := getInt("max_position_embeddings", 2048)
	if ctxOverride > 0 {
		maxSeq = ctxOverride
	} else if maxSeq > 8192 {
		maxSeq = 8192
	}
	finalLogitCap := float32(0)
	if v, ok := cfg["final_logit_softcapping"].(float64); ok {
		finalLogitCap = float32(v)
	}
	attnLogitCap := float32(0)
	if v, ok := cfg["attn_logit_softcapping"].(float64); ok {
		attnLogitCap = float32(v)
	}
	_ = attnLogitCap
	ropeTheta := float32(10000.0)
	if v, ok := cfg["rope_theta"].(float64); ok {
		ropeTheta = float32(v)
	}
	normEps := float32(1e-6)
	if v, ok := cfg["rms_norm_eps"].(float64); ok {
		normEps = float32(v)
	}
	act := "silu"
	if v, ok := cfg["hidden_act"].(string); ok {
		act = v
	}

	if v, ok := cfg["layer_norm_eps"].(float64); ok && normEps == 1e-6 {
		normEps = float32(v)
	}
	partialRotary := float32(1.0)
	if v, ok := cfg["partial_rotary_factor"].(float64); ok {
		partialRotary = float32(v)
	}

	family := ""
	if arch, ok := cfg["architectures"].([]interface{}); ok && len(arch) > 0 {
		family = detectFamily(arch[0].(string))
	}
	if mt, ok := cfg["model_type"].(string); ok && family == "" {
		family = mt
	}

	gatedMLP := act == "silu" || act == "swiglu"
	geluGated := false
	oProjName := "self_attn.o_proj.weight"
	var mlpWeightNames [3]string

	splitRows := func(data []float32, totalRows, cols, splitAt int) ([]float32, []float32) {
		a := make([]float32, splitAt*cols)
		b := make([]float32, (totalRows-splitAt)*cols)
		copy(a, data[:splitAt*cols])
		copy(b, data[splitAt*cols:])
		return a, b
	}
	splitRows3 := func(data []float32, totalRows, cols, n1, n2 int) ([]float32, []float32, []float32) {
		a := make([]float32, n1*cols)
		b := make([]float32, n2*cols)
		c := make([]float32, (totalRows-n1-n2)*cols)
		copy(a, data[:n1*cols])
		copy(b, data[n1*cols:(n1+n2)*cols])
		copy(c, data[(n1+n2)*cols:])
		return a, b, c
	}

	rotaryDim := headDim
	if partialRotary < 1.0 {
		rotaryDim = int(float32(headDim) * partialRotary)
	}

	if dim == 0 || nLayers == 0 {
		log.Fatal("could not read model dimensions from config.json")
	}

	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	mongoose.LoadKernels()

	// Detect quantization
	isAWQ := false
	awqGroupSize := 128
	if qcfg, ok := cfg["quantization_config"]; ok {
		if qm, ok := qcfg.(map[string]interface{}); ok {
			if method, _ := qm["quant_method"].(string); method == "awq" || method == "gptq" {
				isAWQ = true
				if gs, ok := qm["group_size"].(float64); ok {
					awqGroupSize = int(gs)
				}
				fmt.Printf("Quantization: %s (4-bit, group_size=%d)\n", method, awqGroupSize)
			}
		}
	}

	// SQ4 model detection — check for sq4_meta.json
	sq4MetaPath := filepath.Join(path, "sq4_meta.json")
	if _, err := os.Stat(sq4MetaPath); err == nil {
		metal, _ := eng.(*mongoose.Metal)
		if metal != nil {
			cmdInferSQ4Metal(path, prompt, metal, te, cfg, dim, nLayers, heads, kvHeads, ffnDim,
				vocabSize, headDim, kvDim, attnDim, maxSeq, ropeTheta, normEps)
			return
		}
		log.Println("[SQ4] SQ4 model detected but Metal not available — falling back to FP32")
	}

	ms, err := OpenModel(path)
	if err != nil {
		log.Fatalf("open model: %v", err)
	}
	st := ms.ST()
	if st == nil {
		log.Fatalf("model format not supported for inference — convert with: ai convert safetensors %s", path)
	}

	fusedQKV := st.HasTensor("model.layers.0.self_attn.qkv_proj.weight")
	fusedGateUp := st.HasTensor("model.layers.0.mlp.gate_up_proj.weight")

	if !gatedMLP && strings.Contains(act, "gelu") && st.HasTensor("model.layers.0.mlp.gate_proj.weight") {
		gatedMLP = true
		geluGated = true
	}
	if gatedMLP {
		mlpWeightNames = [3]string{"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}
	} else {
		mlpWeightNames = [3]string{"mlp.fc1.weight", "mlp.fc2.weight", ""}
		if st.HasTensor("model.layers.0.self_attn.dense.weight") {
			oProjName = "self_attn.dense.weight"
		}
	}

	readWeight := func(name string, expectedRows, expectedCols int) ([]float32, int, int, error) {
		if isAWQ {
			qname := strings.TrimSuffix(name, "weight") + "qweight"
			if st.HasTensor(qname) {
				prefix := strings.TrimSuffix(name, "weight")
				return st.DequantAWQ(prefix, awqGroupSize)
			}
		}
		data, _, err := st.ReadTensorFloat32(name)
		if err != nil {
			return nil, 0, 0, err
		}
		return data, expectedRows, expectedCols, nil
	}

	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	modelName := strings.ToLower(filepath.Base(path))
	instructSuffixes := []string{"-instruct", "-chat", "-it", ".instruct", ".chat"}
	isInstruct := false
	for _, suffix := range instructSuffixes {
		if strings.Contains(modelName, suffix) {
			isInstruct = true
			break
		}
	}
	rawMode := false
	if len(promptParts) > 0 && promptParts[0] == "--raw" {
		rawMode = true
		promptParts = promptParts[1:]
		prompt = strings.Join(promptParts, " ")
	}

	var tokens []int
	if isInstruct && !rawMode {
		messages := []chatMessage{{Role: "user", Content: prompt}}
		tokens = applyChatTemplate(tok, messages, cfg)
	} else {
		tokens = tok.Encode(prompt)
	}

	stopTokens := discoverStopTokens(tok, cfg, path)

	fmt.Printf("Model: %s (dim=%d layers=%d heads=%d kv=%d ffn=%d vocab=%d act=%s)\n",
		filepath.Base(path), dim, nLayers, heads, kvHeads, ffnDim, vocabSize, act)
	fmt.Printf("Engine: %s\n", eng.Name())
	if isInstruct && !rawMode {
		fmt.Printf("Prompt: %q → %d tokens (chat template applied)\n", prompt, len(tokens))
	} else {
		fmt.Printf("Prompt: %q → %d tokens\n", prompt, len(tokens))
	}

	// Load embeddings + lm_head
	fmt.Print("Loading embeddings... ")
	embedData, _, _ := st.ReadTensorFloat32("model.embed_tokens.weight")
	var lmHeadData []float32
	lmHeadData, _, err = st.ReadTensorFloat32("lm_head.weight")
	if err != nil {
		lmHeadData = embedData
	}
	fmt.Println("done")

	// Decide inference tier
	gpuResident := false
	useGPUKernels := false

	type gpuLayer struct {
		wq, wk, wv, wo        *mongoose.Tensor
		gate, up, down         *mongoose.Tensor
		bq, bk, bv, bo        *mongoose.Tensor
		bgate, bdown           *mongoose.Tensor
		norm1, norm2           []float32
		gnorm1, gnorm2         *mongoose.Tensor
		norm1bias, norm2bias   *mongoose.Tensor
	}
	var gpuLayers []gpuLayer
	var gpuEmbed, gpuLMHead, gpuFinalNormT *mongoose.Tensor
	var gpuFinalNorm []float32

	// Estimate total GPU memory for FP32-resident path (weights + KV cache + embed/lmhead + scratch)
	fp32WeightBytes := int64(vocabSize)*int64(dim)*4*2 + // embed + lmHead on GPU
		int64(nLayers)*(int64(attnDim)*int64(dim)*4 + int64(kvDim)*int64(dim)*4*2 + int64(dim)*int64(attnDim)*4 + // q,k,v,o
			int64(ffnDim)*int64(dim)*4*3 + // gate, up, down
			int64(dim)*4*4) + // norms (up to 4 per layer for gemma)
		int64(2)*int64(nLayers)*int64(maxSeq)*int64(kvDim)*4 + // KV cache
		int64(maxSeq)*int64(headDim/2)*4*2 // RoPE tables
	fp32NeedsGB := float64(fp32WeightBytes) / (1024 * 1024 * 1024)

	skipFP32Resident := false
	if te != nil {
		vram := eng.VRAM()
		vramGB := float64(vram) / (1024 * 1024 * 1024)
		safeGB := vramGB * 0.85
		if fp32NeedsGB > safeGB {
			fmt.Printf("  Model FP32 weights: %.1f GB, GPU VRAM: %.1f GB — skipping FP32 GPU-resident, using Q8/Q4\n", fp32NeedsGB, vramGB)
			skipFP32Resident = true
		}
	}

	if te != nil && !skipFP32Resident {
		gpuResident = true
		fmt.Print("Loading weights to GPU... ")
		gpuLayers = make([]gpuLayer, nLayers)
		for l := 0; l < nLayers; l++ {
			prefix := fmt.Sprintf("model.layers.%d.", l)
			loadGPU := func(name string, rows, cols int) *mongoose.Tensor {
				data, r, c, err := readWeight(prefix+name, rows, cols)
				if err != nil {
					return nil
				}
				return te.FromHost(data, []int{r, c})
			}
			loadCPU := func(name string) []float32 {
				data, _, _ := st.ReadTensorFloat32(prefix + name)
				return data
			}
			gl := gpuLayer{
				wo:    loadGPU(oProjName, dim, attnDim),
				norm1: loadCPU("input_layernorm.weight"),
				norm2: loadCPU("post_attention_layernorm.weight"),
			}
			if fusedQKV {
				qkvData, _, _, _ := readWeight(prefix+"self_attn.qkv_proj.weight", dim+kvDim+kvDim, dim)
				if qkvData != nil {
					qD, kD, vD := splitRows3(qkvData, dim+kvDim+kvDim, dim, dim, kvDim)
					gl.wq = te.FromHost(qD, []int{dim, dim})
					gl.wk = te.FromHost(kD, []int{kvDim, dim})
					gl.wv = te.FromHost(vD, []int{kvDim, dim})
				}
			} else {
				gl.wq = loadGPU("self_attn.q_proj.weight", attnDim, dim)
				gl.wk = loadGPU("self_attn.k_proj.weight", kvDim, dim)
				gl.wv = loadGPU("self_attn.v_proj.weight", kvDim, dim)
			}
			if fusedGateUp {
				guData, _, _, _ := readWeight(prefix+"mlp.gate_up_proj.weight", ffnDim*2, dim)
				if guData != nil {
					gD, uD := splitRows(guData, ffnDim*2, dim, ffnDim)
					gl.gate = te.FromHost(gD, []int{ffnDim, dim})
					gl.up = te.FromHost(uD, []int{ffnDim, dim})
				}
				gl.down = loadGPU("mlp.down_proj.weight", dim, ffnDim)
			} else if gatedMLP {
				gl.gate = loadGPU(mlpWeightNames[0], ffnDim, dim)
				gl.up = loadGPU(mlpWeightNames[1], ffnDim, dim)
				gl.down = loadGPU(mlpWeightNames[2], dim, ffnDim)
			} else {
				gl.gate = loadGPU(mlpWeightNames[0], ffnDim, dim)
				gl.down = loadGPU(mlpWeightNames[1], dim, ffnDim)
			}
			gpuLayers[l] = gl
			gpuLayers[l].gnorm1 = te.FromHost(gpuLayers[l].norm1, []int{1, dim})
			gpuLayers[l].gnorm2 = te.FromHost(gpuLayers[l].norm2, []int{1, dim})
			if bq, _, e := st.ReadTensorFloat32(prefix + "self_attn.q_proj.bias"); e == nil {
				gpuLayers[l].bq = te.FromHost(bq, []int{1, dim})
			}
			if bk, _, e := st.ReadTensorFloat32(prefix + "self_attn.k_proj.bias"); e == nil {
				gpuLayers[l].bk = te.FromHost(bk, []int{1, kvDim})
			}
			if bv, _, e := st.ReadTensorFloat32(prefix + "self_attn.v_proj.bias"); e == nil {
				gpuLayers[l].bv = te.FromHost(bv, []int{1, kvDim})
			}
			oBiasName := strings.TrimSuffix(oProjName, ".weight") + ".bias"
			if bo, _, e := st.ReadTensorFloat32(prefix + oBiasName); e == nil {
				gpuLayers[l].bo = te.FromHost(bo, []int{1, dim})
			}
			if bg, _, e := st.ReadTensorFloat32(prefix + strings.TrimSuffix(mlpWeightNames[0], ".weight") + ".bias"); e == nil {
				gpuLayers[l].bgate = te.FromHost(bg, []int{1, ffnDim})
			}
			fc2BiasName := mlpWeightNames[1]
			if !gatedMLP { fc2BiasName = mlpWeightNames[1] }
			if gatedMLP { fc2BiasName = mlpWeightNames[2] }
			if bd, _, e := st.ReadTensorFloat32(prefix + strings.TrimSuffix(fc2BiasName, ".weight") + ".bias"); e == nil {
				gpuLayers[l].bdown = te.FromHost(bd, []int{1, dim})
			}
			if nb, _, e := st.ReadTensorFloat32(prefix + "input_layernorm.bias"); e == nil {
				gpuLayers[l].norm1bias = te.FromHost(nb, []int{1, dim})
			}
			if nb, _, e := st.ReadTensorFloat32(prefix + "post_attention_layernorm.bias"); e == nil {
				gpuLayers[l].norm2bias = te.FromHost(nb, []int{1, dim})
			}
			fmt.Printf("\r  Layer %d/%d loaded", l+1, nLayers)
		}
		gpuEmbed = te.FromHost(embedData, []int{vocabSize, dim})
		gpuLMHead = te.FromHost(lmHeadData, []int{vocabSize, dim})
		gpuFinalNorm, _, _ = st.ReadTensorFloat32("model.norm.weight")
		if gpuFinalNorm == nil {
			gpuFinalNorm, _, _ = st.ReadTensorFloat32("model.final_layernorm.weight")
		}
		if gpuFinalNorm != nil {
			gpuFinalNormT = te.FromHost(gpuFinalNorm, []int{1, dim})
		}
		fmt.Println(" — all weights on GPU")

		useGPUKernels = mongoose.TrainKernelsLoaded() && mongoose.KernelsLoaded()
	}

	// RoPE tables
	halfHead := rotaryDim / 2
	cosTab := make([]float32, maxSeq*halfHead)
	sinTab := make([]float32, maxSeq*halfHead)
	for pos := 0; pos < maxSeq; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(float64(ropeTheta), float64(2*j)/float64(rotaryDim))
			angle := float64(pos) * freq
			cosTab[pos*halfHead+j] = float32(math.Cos(angle))
			sinTab[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}

	// GPU RoPE tables + KV cache (needed for GPU-resident and Q8/Q4 paths)
	var gpuRopeCos, gpuRopeSin *mongoose.Tensor
	type gpuKVCache struct {
		k, v *mongoose.Tensor
	}
	var gpuKV []gpuKVCache

	needsGPUKV := (useGPUKernels && gpuResident) || (te != nil && mongoose.HasQ8Matvec() && mongoose.KernelsLoaded())
	if needsGPUKV {
		gpuRopeCos = te.FromHost(cosTab, []int{maxSeq, halfHead})
		gpuRopeSin = te.FromHost(sinTab, []int{maxSeq, halfHead})
		gpuKV = make([]gpuKVCache, nLayers)
		for l := 0; l < nLayers; l++ {
			gpuKV[l] = gpuKVCache{
				k: te.Zeros([]int{maxSeq, kvDim}),
				v: te.Zeros([]int{maxSeq, kvDim}),
			}
		}
	}

	if useGPUKernels && gpuResident {
		fmt.Println("  infer: GPU-resident (KV cache + RoPE in VRAM)")
	} else if gpuResident {
		fmt.Println("  infer: GPU-accelerated (weights in VRAM, attention on CPU)")
	} else if !needsGPUKV {
		fmt.Println("  infer: CPU (streaming weights)")
	}

	// CPU state buffers (used by tier 2 and 3)
	softcapLogits := func(logits []float32) {
		if finalLogitCap > 0 {
			for i := range logits {
				logits[i] = finalLogitCap * float32(math.Tanh(float64(logits[i]/finalLogitCap)))
			}
		}
	}

	x := make([]float32, dim)
	buf := make([]float32, dim)
	q := make([]float32, dim)
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	att := make([]float32, heads*maxSeq)
	attnOut := make([]float32, dim)
	ffnBuf := make([]float32, ffnDim)
	ffnBuf2 := make([]float32, ffnDim)
	keyCache := make([][]float32, nLayers)
	valCache := make([][]float32, nLayers)
	for l := 0; l < nLayers; l++ {
		keyCache[l] = make([]float32, maxSeq*kvDim)
		valCache[l] = make([]float32, maxSeq*kvDim)
	}

	// Tier 2 helper: GPU-resident matmul (weights stay on GPU, x uploaded per call)
	rwe := mongoose.AsResidentWeightEngine(eng)
	gpuResidentMatVec := func(out []float32, tW *mongoose.Tensor, xIn []float32, rows, cols int) {
		if rwe != nil {
			r := rwe.MatVecResidentW(tW, xIn, rows, cols)
			copy(out, r)
			return
		}
		tX := te.FromHost(xIn, []int{1, cols})
		tY := te.MatMulT(tW, tX, rows, cols, 1)
		copy(out, te.ToHost(tY))
		te.Release(tX)
		te.Release(tY)
	}

	// Tier 3 helper: streaming matmul (upload weight + x each call)
	streamMatVec := func(out []float32, W, xIn []float32, rows, cols int) {
		r := eng.MatMul(W, xIn, rows, cols, 1)
		copy(out, r)
	}

	rmsNorm := func(data, weight []float32, eps float32) {
		n := len(data)
		var ss float32
		for i := 0; i < n; i++ {
			ss += data[i] * data[i]
		}
		ss = 1.0 / float32(math.Sqrt(float64(ss/float32(n)+eps)))
		for i := 0; i < n; i++ {
			data[i] = data[i] * ss * weight[i]
		}
	}

	// === Tier 1: Fully GPU-resident forward (CUDA kernels) ===
	gpuForward := func(tokenID, pos int) []float32 {
		tokOff := tokenID * dim
		if tokOff+dim > len(embedData) {
			return nil
		}
		xGPU := te.FromHost(embedData[tokOff:tokOff+dim], []int{1, dim})

		for l := 0; l < nLayers; l++ {
			gl := &gpuLayers[l]

			normed := te.Zeros([]int{1, dim})
			mongoose.KRMSNormOut(xGPU.DevicePtr(), normed.DevicePtr(), gl.gnorm1.DevicePtr(), 1, dim)
			if gl.norm1bias != nil { te.AddInPlace(normed, gl.norm1bias) }

			tQ := te.MatMulTransposeBT(normed, gl.wq, 1, dim, attnDim)
			tK := te.MatMulTransposeBT(normed, gl.wk, 1, dim, kvDim)
			tV := te.MatMulTransposeBT(normed, gl.wv, 1, dim, kvDim)
			if gl.gnorm2 != nil {
				te.Release(normed)
			}

			if gl.bq != nil {
				te.AddInPlace(tQ, gl.bq)
			}
			if gl.bk != nil {
				te.AddInPlace(tK, gl.bk)
			}
			if gl.bv != nil {
				te.AddInPlace(tV, gl.bv)
			}

			if rotaryDim == headDim {
				cosOff := unsafe.Add(gpuRopeCos.DevicePtr(), uintptr(pos*halfHead*4))
				sinOff := unsafe.Add(gpuRopeSin.DevicePtr(), uintptr(pos*halfHead*4))
				mongoose.KRoPE(tQ.DevicePtr(), cosOff, sinOff, 1, dim, headDim, heads)
				mongoose.KRoPE(tK.DevicePtr(), cosOff, sinOff, 1, kvDim, headDim, kvHeads)
			} else {
				qHost := te.ToHost(tQ)
				kHost := te.ToHost(tK)
				applyRoPEPartial(qHost, pos, headDim, rotaryDim, heads, ropeTheta)
				applyRoPEPartial(kHost, pos, headDim, rotaryDim, kvHeads, ropeTheta)
				te.Release(tQ)
				te.Release(tK)
				tQ = te.FromHost(qHost, []int{1, dim})
				tK = te.FromHost(kHost, []int{1, kvDim})
			}

			kv := &gpuKV[l]
			mongoose.KCopy(
				unsafe.Add(kv.k.DevicePtr(), uintptr(pos*kvDim*4)),
				tK.DevicePtr(), kvDim*4)
			mongoose.KCopy(
				unsafe.Add(kv.v.DevicePtr(), uintptr(pos*kvDim*4)),
				tV.DevicePtr(), kvDim*4)
			te.Release(tK)
			te.Release(tV)

			tAttnOut := te.Zeros([]int{1, attnDim})
			mongoose.KDecodeAttention(tQ.DevicePtr(), kv.k.DevicePtr(), kv.v.DevicePtr(),
				tAttnOut.DevicePtr(), pos+1, attnDim, kvDim, heads, kvHeads)
			te.Release(tQ)

			tProj := te.MatMulTransposeBT(tAttnOut, gl.wo, 1, attnDim, dim)
			te.Release(tAttnOut)
			if gl.bo != nil { te.AddInPlace(tProj, gl.bo) }
			te.AddInPlace(xGPU, tProj)
			te.Release(tProj)

			var normed2 *mongoose.Tensor
			if gl.gnorm2 != nil {
				normed2 = te.Zeros([]int{1, dim})
				mongoose.KRMSNormOut(xGPU.DevicePtr(), normed2.DevicePtr(), gl.gnorm2.DevicePtr(), 1, dim)
				if gl.norm2bias != nil { te.AddInPlace(normed2, gl.norm2bias) }
			} else {
				normed2 = normed
			}

			if gatedMLP {
				tGate := te.MatMulTransposeBT(normed2, gl.gate, 1, dim, ffnDim)
				tUp := te.MatMulTransposeBT(normed2, gl.up, 1, dim, ffnDim)
				if gl.gnorm2 != nil { te.Release(normed2) } else { te.Release(normed) }

				var ffnMid *mongoose.Tensor
				if geluGated {
					gH := te.ToHost(tGate)
					uH := te.ToHost(tUp)
					te.Release(tGate)
					te.Release(tUp)
					for i := range gH { gH[i] = geluNew(gH[i]) * uH[i] }
					ffnMid = te.FromHost(gH, []int{1, ffnDim})
				} else {
					ffnMid = te.Zeros([]int{1, ffnDim})
					mongoose.KSiLUGateMul(tGate.DevicePtr(), tUp.DevicePtr(), ffnMid.DevicePtr(), ffnDim)
					te.Release(tGate)
					te.Release(tUp)
				}

				tDown := te.MatMulTransposeBT(ffnMid, gl.down, 1, ffnDim, dim)
				te.Release(ffnMid)
				if gl.bdown != nil { te.AddInPlace(tDown, gl.bdown) }
				te.AddInPlace(xGPU, tDown)
				te.Release(tDown)
			} else {
				tFC1 := te.MatMulTransposeBT(normed2, gl.gate, 1, dim, ffnDim)
				if gl.gnorm2 != nil { te.Release(normed2) } else { te.Release(normed) }
				if gl.bgate != nil { te.AddInPlace(tFC1, gl.bgate) }
				fc1Host := te.ToHost(tFC1)
				for i := range fc1Host { fc1Host[i] = geluNew(fc1Host[i]) }
				te.Release(tFC1)
				tAct := te.FromHost(fc1Host, []int{1, ffnDim})
				tFC2 := te.MatMulTransposeBT(tAct, gl.down, 1, ffnDim, dim)
				te.Release(tAct)
				if gl.bdown != nil { te.AddInPlace(tFC2, gl.bdown) }
				te.AddInPlace(xGPU, tFC2)
				te.Release(tFC2)
			}
		}

		mongoose.KRMSNorm(xGPU.DevicePtr(), gpuFinalNormT.DevicePtr(), 1, dim)

		tLogits := te.MatMulTransposeBT(xGPU, gpuLMHead, 1, dim, vocabSize)
		te.Release(xGPU)

		logits := te.ToHost(tLogits)
		te.Release(tLogits)
		softcapLogits(logits)
		return logits
	}

	// === CUDA Q8/Q4 fused path ===
	gpuForwardQ8 := func(tokenID, pos int) []float32 { return nil }
	useCUDAQ8 := false
	cudaUseQ4 := false

	if cuda, ok := eng.(*mongoose.CUDA); ok && mongoose.HasQ8Matvec() && mongoose.KernelsLoaded() {
		_ = cuda

		nParams := int64(vocabSize)*int64(dim)*2 // embed + lmHead
		nRows := int64(vocabSize)*2
		for l := 0; l < nLayers; l++ {
			nParams += int64(dim)*int64(dim)*2 + int64(kvDim)*int64(dim)*2 + int64(ffnDim)*int64(dim)*3
			nRows += int64(dim)*2 + int64(kvDim)*2 + int64(ffnDim)*3
		}
		// Q8: 1 byte/param + 4 bytes/row (scale), Q4: 0.5 bytes/param + 4 bytes/row
		q8Bytes := nParams + nRows*4
		q4Bytes := nParams/2 + nRows*4
		kvBytes := int64(2) * int64(nLayers) * int64(maxSeq) * int64(kvDim) * 4
		scratchBytes := int64(dim+ffnDim) * 4 * 20 // activation buffers

		vram := int64(eng.VRAM())

		q8Total := q8Bytes + kvBytes + scratchBytes
		q4Total := q4Bytes + kvBytes + scratchBytes

		// Always prefer Q8 for quality. Only use Q4 if Q8 clearly won't fit.
		// Use a generous margin because actual allocations include padding and alignment.
		if mongoose.HasQ4Matvec() && q8Total > vram && q4Total <= vram {
			cudaUseQ4 = true
		} else if mongoose.HasQ4Matvec() && q8Total > vram && q4Total > vram {
			cudaUseQ4 = true
			fmt.Printf("  WARNING: model may not fit in VRAM even at Q4 (need %.1f GB, have %.1f GB)\n",
				float64(q4Total)/(1024*1024*1024), float64(vram)/(1024*1024*1024))
		}

		type q8Weight struct {
			data   unsafe.Pointer // int8 or packed uint8 on GPU
			scales unsafe.Pointer // float32 on GPU [rows floats]
			rows   int
			cols   int
			q4     bool
		}
		quantizeToGPU := func(fp32 []float32, rows, cols int) q8Weight {
			s := make([]float32, rows)
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
				s[r] = absMax
			}

			if cudaUseQ4 {
				packed := make([]byte, rows*cols/2)
				for r := 0; r < rows; r++ {
					invScale := float32(0)
					if s[r] > 0 {
						invScale = 7.0 / s[r]
					}
					for c := 0; c < cols; c += 2 {
						v0 := fp32[r*cols+c] * invScale
						v1 := float32(0)
						if c+1 < cols {
							v1 = fp32[r*cols+c+1] * invScale
						}
						q0 := int(v0+0.5) + 8
						q1 := int(v1+0.5) + 8
						if v0 < 0 {
							q0 = int(v0-0.5) + 8
						}
						if v1 < 0 {
							q1 = int(v1-0.5) + 8
						}
						if q0 < 0 { q0 = 0 }
						if q0 > 15 { q0 = 15 }
						if q1 < 0 { q1 = 0 }
						if q1 > 15 { q1 = 15 }
						packed[r*(cols/2)+c/2] = byte(q0 | (q1 << 4))
					}
				}
				nBytes := len(packed)
				nFloats := (nBytes + 3) / 4
				buf := make([]float32, nFloats)
				copy((*[1 << 30]byte)(unsafe.Pointer(&buf[0]))[:nBytes], packed)
				tData := te.FromHost(buf, []int{1, nFloats})
				tScales := te.FromHost(s, []int{1, rows})
				return q8Weight{data: tData.DevicePtr(), scales: tScales.DevicePtr(), rows: rows, cols: cols, q4: true}
			}

			q := make([]int8, rows*cols)
			for r := 0; r < rows; r++ {
				invScale := float32(0)
				if s[r] > 0 {
					invScale = 127.0 / s[r]
				}
				for c := 0; c < cols; c++ {
					v := fp32[r*cols+c] * invScale
					if v > 127 {
						v = 127
					}
					if v < -127 {
						v = -127
					}
					q[r*cols+c] = int8(v)
				}
			}
			nBytes := rows * cols
			nFloats := (nBytes + 3) / 4
			buf := make([]float32, nFloats)
			copy((*[1 << 30]byte)(unsafe.Pointer(&buf[0]))[:nBytes], (*[1 << 30]byte)(unsafe.Pointer(&q[0]))[:nBytes])
			tData := te.FromHost(buf, []int{1, nFloats})
			tScales := te.FromHost(s, []int{1, rows})
			return q8Weight{data: tData.DevicePtr(), scales: tScales.DevicePtr(), rows: rows, cols: cols, q4: false}
		}

		type q8Layer struct {
			wq, wk, wv, wo        q8Weight
			gate, up, down         q8Weight
			bq, bk, bv, bo        *mongoose.Tensor
			bgate, bdown           *mongoose.Tensor
			gnorm1, gnorm2         *mongoose.Tensor
			norm1bias, norm2bias   *mongoose.Tensor
		}

		qLabel := "Q8"
		if cudaUseQ4 {
			qLabel = "Q4"
		}
		fmt.Printf("Quantizing weights to %s (%.1fB params)... ", qLabel, float64(nParams)/1e9)
		q8Layers := make([]q8Layer, nLayers)
		for l := 0; l < nLayers; l++ {
			q8l := &q8Layers[l]
			prefix := fmt.Sprintf("model.layers.%d.", l)

			if l < len(gpuLayers) {
				gl := &gpuLayers[l]
				q8l.gnorm1 = gl.gnorm1
				q8l.gnorm2 = gl.gnorm2
				q8l.bq = gl.bq
				q8l.bk = gl.bk
				q8l.bv = gl.bv
				q8l.bo = gl.bo
				q8l.bgate = gl.bgate
				q8l.bdown = gl.bdown
				q8l.norm1bias = gl.norm1bias
				q8l.norm2bias = gl.norm2bias
			} else {
				n1, _, _ := st.ReadTensorFloat32(prefix + "input_layernorm.weight")
				n2, _, _ := st.ReadTensorFloat32(prefix + "post_attention_layernorm.weight")
				if n1 != nil { q8l.gnorm1 = te.FromHost(n1, []int{1, dim}) }
				if n2 != nil { q8l.gnorm2 = te.FromHost(n2, []int{1, dim}) }
				if bq, _, e := st.ReadTensorFloat32(prefix + "self_attn.q_proj.bias"); e == nil {
					q8l.bq = te.FromHost(bq, []int{1, dim})
				}
				if bk, _, e := st.ReadTensorFloat32(prefix + "self_attn.k_proj.bias"); e == nil {
					q8l.bk = te.FromHost(bk, []int{1, kvDim})
				}
				if bv, _, e := st.ReadTensorFloat32(prefix + "self_attn.v_proj.bias"); e == nil {
					q8l.bv = te.FromHost(bv, []int{1, kvDim})
				}
				oBiasName := strings.TrimSuffix(oProjName, ".weight") + ".bias"
				if bo, _, e := st.ReadTensorFloat32(prefix + oBiasName); e == nil {
					q8l.bo = te.FromHost(bo, []int{1, dim})
				}
				if bg, _, e := st.ReadTensorFloat32(prefix + strings.TrimSuffix(mlpWeightNames[0], ".weight") + ".bias"); e == nil {
					q8l.bgate = te.FromHost(bg, []int{1, ffnDim})
				}
				fc2BiasName := mlpWeightNames[1]
				if gatedMLP { fc2BiasName = mlpWeightNames[2] }
				if bd, _, e := st.ReadTensorFloat32(prefix + strings.TrimSuffix(fc2BiasName, ".weight") + ".bias"); e == nil {
					q8l.bdown = te.FromHost(bd, []int{1, dim})
				}
				if nb, _, e := st.ReadTensorFloat32(prefix + "input_layernorm.bias"); e == nil {
					q8l.norm1bias = te.FromHost(nb, []int{1, dim})
				}
				if nb, _, e := st.ReadTensorFloat32(prefix + "post_attention_layernorm.bias"); e == nil {
					q8l.norm2bias = te.FromHost(nb, []int{1, dim})
				}
			}

			loadQ8 := func(name string, rows, cols int) q8Weight {
				data, _, _, _ := readWeight(prefix+name, rows, cols)
				return quantizeToGPU(data, rows, cols)
			}
			if fusedQKV {
				qkvData, _, _, _ := readWeight(prefix+"self_attn.qkv_proj.weight", dim+kvDim+kvDim, dim)
				if qkvData != nil {
					qD, kD, vD := splitRows3(qkvData, dim+kvDim+kvDim, dim, dim, kvDim)
					q8l.wq = quantizeToGPU(qD, attnDim, dim)
					q8l.wk = quantizeToGPU(kD, kvDim, dim)
					q8l.wv = quantizeToGPU(vD, kvDim, dim)
				}
			} else {
				q8l.wq = loadQ8("self_attn.q_proj.weight", attnDim, dim)
				q8l.wk = loadQ8("self_attn.k_proj.weight", kvDim, dim)
				q8l.wv = loadQ8("self_attn.v_proj.weight", kvDim, dim)
			}
			q8l.wo = loadQ8(oProjName, dim, attnDim)
			if fusedGateUp {
				guData, _, _, _ := readWeight(prefix+"mlp.gate_up_proj.weight", ffnDim*2, dim)
				if guData != nil {
					gD, uD := splitRows(guData, ffnDim*2, dim, ffnDim)
					q8l.gate = quantizeToGPU(gD, ffnDim, dim)
					q8l.up = quantizeToGPU(uD, ffnDim, dim)
				}
				q8l.down = loadQ8("mlp.down_proj.weight", dim, ffnDim)
			} else if gatedMLP {
				q8l.gate = loadQ8(mlpWeightNames[0], ffnDim, dim)
				q8l.up = loadQ8(mlpWeightNames[1], ffnDim, dim)
				q8l.down = loadQ8(mlpWeightNames[2], dim, ffnDim)
			} else {
				q8l.gate = loadQ8(mlpWeightNames[0], ffnDim, dim)
				q8l.down = loadQ8(mlpWeightNames[1], dim, ffnDim)
			}
			fmt.Printf("\r  Layer %d/%d quantized to %s", l+1, nLayers, qLabel)
		}
		fmt.Println()
		q8LMHead := quantizeToGPU(lmHeadData, vocabSize, dim)
		if gpuFinalNormT == nil {
			fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
			if fnorm == nil {
				fnorm, _, _ = st.ReadTensorFloat32("model.final_layernorm.weight")
			}
			if fnorm != nil {
				gpuFinalNormT = te.FromHost(fnorm, []int{1, dim})
			}
		}
		fmt.Println("done")

		useCUDAQ8 = true

		q8MV := func(actPtr unsafe.Pointer, w q8Weight, outPtr unsafe.Pointer) {
			if w.q4 {
				mongoose.KQ4Matvec(actPtr, w.data, w.scales, outPtr, w.rows, w.cols)
			} else {
				mongoose.KQ8Matvec(actPtr, w.data, w.scales, outPtr, w.rows, w.cols)
			}
		}

		embedScale := float32(0)
		if family == "gemma" || family == "gemma2" {
			embedScale = float32(math.Sqrt(float64(dim)))
		}

		gpuForwardQ8 = func(tokenID, pos int) []float32 {
			tokOff := tokenID * dim
			if tokOff+dim > len(embedData) {
				return nil
			}
			emb := make([]float32, dim)
			copy(emb, embedData[tokOff:tokOff+dim])
			if embedScale > 0 {
				for i := range emb { emb[i] *= embedScale }
			}
			xGPU := te.FromHost(emb, []int{1, dim})

			for l := 0; l < nLayers; l++ {
				ql := &q8Layers[l]

				normed := te.Zeros([]int{1, dim})
				mongoose.KRMSNormOut(xGPU.DevicePtr(), normed.DevicePtr(), ql.gnorm1.DevicePtr(), 1, dim)
				if ql.norm1bias != nil { te.AddInPlace(normed, ql.norm1bias) }

				tQ := te.Zeros([]int{1, attnDim})
				tK := te.Zeros([]int{1, kvDim})
				tV := te.Zeros([]int{1, kvDim})
				q8MV(normed.DevicePtr(), ql.wq, tQ.DevicePtr())
				q8MV(normed.DevicePtr(), ql.wk, tK.DevicePtr())
				q8MV(normed.DevicePtr(), ql.wv, tV.DevicePtr())
				if ql.gnorm2 != nil {
					te.Release(normed)
				}

				if ql.bq != nil {
					te.AddInPlace(tQ, ql.bq)
				}
				if ql.bk != nil {
					te.AddInPlace(tK, ql.bk)
				}
				if ql.bv != nil {
					te.AddInPlace(tV, ql.bv)
				}

				if rotaryDim == headDim {
					cosOff := unsafe.Add(gpuRopeCos.DevicePtr(), uintptr(pos*halfHead*4))
					sinOff := unsafe.Add(gpuRopeSin.DevicePtr(), uintptr(pos*halfHead*4))
					mongoose.KRoPE(tQ.DevicePtr(), cosOff, sinOff, 1, dim, headDim, heads)
					mongoose.KRoPE(tK.DevicePtr(), cosOff, sinOff, 1, kvDim, headDim, kvHeads)
				} else {
					qHost := te.ToHost(tQ)
					kHost := te.ToHost(tK)
					applyRoPEPartial(qHost, pos, headDim, rotaryDim, heads, ropeTheta)
					applyRoPEPartial(kHost, pos, headDim, rotaryDim, kvHeads, ropeTheta)
					te.Release(tQ)
					te.Release(tK)
					tQ = te.FromHost(qHost, []int{1, dim})
					tK = te.FromHost(kHost, []int{1, kvDim})
				}

				kv := &gpuKV[l]
				mongoose.KCopy(
					unsafe.Add(kv.k.DevicePtr(), uintptr(pos*kvDim*4)),
					tK.DevicePtr(), kvDim*4)
				mongoose.KCopy(
					unsafe.Add(kv.v.DevicePtr(), uintptr(pos*kvDim*4)),
					tV.DevicePtr(), kvDim*4)
				te.Release(tK)
				te.Release(tV)

				tAttnOut := te.Zeros([]int{1, attnDim})
				mongoose.KDecodeAttention(tQ.DevicePtr(), kv.k.DevicePtr(), kv.v.DevicePtr(),
					tAttnOut.DevicePtr(), pos+1, attnDim, kvDim, heads, kvHeads)
				te.Release(tQ)

				tProj := te.Zeros([]int{1, dim})
				q8MV(tAttnOut.DevicePtr(), ql.wo, tProj.DevicePtr())
				te.Release(tAttnOut)
				if ql.bo != nil { te.AddInPlace(tProj, ql.bo) }
				te.AddInPlace(xGPU, tProj)
				te.Release(tProj)

				var normed2 *mongoose.Tensor
				if ql.gnorm2 != nil {
					normed2 = te.Zeros([]int{1, dim})
					mongoose.KRMSNormOut(xGPU.DevicePtr(), normed2.DevicePtr(), ql.gnorm2.DevicePtr(), 1, dim)
					if ql.norm2bias != nil { te.AddInPlace(normed2, ql.norm2bias) }
				} else {
					normed2 = normed
				}

				if gatedMLP {
					tGate := te.Zeros([]int{1, ffnDim})
					tUp := te.Zeros([]int{1, ffnDim})
					q8MV(normed2.DevicePtr(), ql.gate, tGate.DevicePtr())
					q8MV(normed2.DevicePtr(), ql.up, tUp.DevicePtr())
					if ql.gnorm2 != nil { te.Release(normed2) } else { te.Release(normed) }

					var ffnMid *mongoose.Tensor
					if geluGated {
						gH := te.ToHost(tGate)
						uH := te.ToHost(tUp)
						te.Release(tGate)
						te.Release(tUp)
						for i := range gH { gH[i] = geluNew(gH[i]) * uH[i] }
						ffnMid = te.FromHost(gH, []int{1, ffnDim})
					} else {
						ffnMid = te.Zeros([]int{1, ffnDim})
						mongoose.KSiLUGateMul(tGate.DevicePtr(), tUp.DevicePtr(), ffnMid.DevicePtr(), ffnDim)
						te.Release(tGate)
						te.Release(tUp)
					}

					tDown := te.Zeros([]int{1, dim})
					q8MV(ffnMid.DevicePtr(), ql.down, tDown.DevicePtr())
					te.Release(ffnMid)
					te.AddInPlace(xGPU, tDown)
					te.Release(tDown)
				} else {
					tFC1 := te.Zeros([]int{1, ffnDim})
					q8MV(normed2.DevicePtr(), ql.gate, tFC1.DevicePtr())
					if ql.gnorm2 != nil { te.Release(normed2) } else { te.Release(normed) }
					if ql.bgate != nil { te.AddInPlace(tFC1, ql.bgate) }
					fc1Host := te.ToHost(tFC1)
					for i := range fc1Host { fc1Host[i] = geluNew(fc1Host[i]) }
					te.Release(tFC1)
					tAct := te.FromHost(fc1Host, []int{1, ffnDim})
					tFC2 := te.Zeros([]int{1, dim})
					q8MV(tAct.DevicePtr(), ql.down, tFC2.DevicePtr())
					te.Release(tAct)
					if ql.bdown != nil { te.AddInPlace(tFC2, ql.bdown) }
					te.AddInPlace(xGPU, tFC2)
					te.Release(tFC2)
				}
			}

			mongoose.KRMSNorm(xGPU.DevicePtr(), gpuFinalNormT.DevicePtr(), 1, dim)

			tLogits := te.Zeros([]int{1, vocabSize})
			mongoose.KQ8Matvec(xGPU.DevicePtr(), q8LMHead.data, q8LMHead.scales, tLogits.DevicePtr(), vocabSize, dim)
			te.Release(xGPU)

			logits := te.ToHost(tLogits)
			te.Release(tLogits)
			return logits
		}
	}

	// === Tier 2 & 3: CPU attention, GPU or CPU matmul ===
	forward := func(tokenID, pos int) []float32 {
		tokOff := tokenID * dim
		if tokOff+dim > len(embedData) {
			return nil
		}
		copy(x, embedData[tokOff:tokOff+dim])
		for i := range buf {
			buf[i] = 0
		}
		for i := range q {
			q[i] = 0
		}
		for i := range k {
			k[i] = 0
		}
		for i := range v {
			v[i] = 0
		}
		for i := range att {
			att[i] = 0
		}
		for i := range attnOut {
			attnOut[i] = 0
		}
		for i := range ffnBuf {
			ffnBuf[i] = 0
		}
		for i := range ffnBuf2 {
			ffnBuf2[i] = 0
		}

		for l := 0; l < nLayers; l++ {
			if gpuResident {
				gl := &gpuLayers[l]

				copy(buf, x)
				rmsNorm(buf, gl.norm1, normEps)

				gpuResidentMatVec(q, gl.wq, buf, dim, dim)
				gpuResidentMatVec(k[:kvDim], gl.wk, buf, kvDim, dim)
				gpuResidentMatVec(v[:kvDim], gl.wv, buf, kvDim, dim)

				if gl.bq != nil {
					bq := te.ToHost(gl.bq)
					for i := range bq {
						q[i] += bq[i]
					}
				}
				if gl.bk != nil {
					bk := te.ToHost(gl.bk)
					for i := range bk {
						k[i] += bk[i]
					}
				}
				if gl.bv != nil {
					bv := te.ToHost(gl.bv)
					for i := range bv {
						v[i] += bv[i]
					}
				}

				if rotaryDim == headDim {
					applyRoPE(q, k[:kvDim], pos, headDim, ropeTheta, heads, kvHeads)
				} else {
					applyRoPEPartial(q, pos, headDim, rotaryDim, heads, ropeTheta)
					applyRoPEPartial(k[:kvDim], pos, headDim, rotaryDim, kvHeads, ropeTheta)
				}

				copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], k[:kvDim])
				copy(valCache[l][pos*kvDim:(pos+1)*kvDim], v[:kvDim])

				kvMul := heads / kvHeads
				for i := range attnOut[:attnDim] {
					attnOut[i] = 0
				}
				for h := 0; h < heads; h++ {
					qOff := h * headDim
					kvH := h / kvMul
					kvOff := kvH * headDim
					scale := float32(1.0 / math.Sqrt(float64(headDim)))
					for t := 0; t <= pos; t++ {
						var dot float64
						for j := 0; j < headDim; j++ {
							dot += float64(q[qOff+j]) * float64(keyCache[l][t*kvDim+kvOff+j])
						}
						att[h*(pos+1)+t] = float32(dot) * scale
					}
					softmax(att[h*(pos+1):h*(pos+1)+(pos+1)], pos+1)
					for t := 0; t <= pos; t++ {
						w := att[h*(pos+1)+t]
						for j := 0; j < headDim; j++ {
							attnOut[qOff+j] += w * valCache[l][t*kvDim+kvOff+j]
						}
					}
				}

				proj := make([]float32, dim)
				gpuResidentMatVec(proj, gl.wo, attnOut, dim, dim)
				for i := 0; i < dim; i++ {
					x[i] += proj[i]
				}

				copy(buf, x)
				rmsNorm(buf, gl.norm2, normEps)

				gpuResidentMatVec(ffnBuf, gl.gate, buf, ffnDim, dim)
				gpuResidentMatVec(ffnBuf2, gl.up, buf, ffnDim, dim)
				if act == "relu" {
					for i := 0; i < ffnDim; i++ {
						if ffnBuf[i] < 0 {
							ffnBuf[i] = 0
						}
						ffnBuf[i] *= ffnBuf2[i]
					}
				} else {
					for i := 0; i < ffnDim; i++ {
						ffnBuf[i] = silu(ffnBuf[i]) * ffnBuf2[i]
					}
				}
				downOut := make([]float32, dim)
				gpuResidentMatVec(downOut, gl.down, ffnBuf, dim, ffnDim)
				for i := 0; i < dim; i++ {
					x[i] += downOut[i]
				}
			} else {
				prefix := fmt.Sprintf("model.layers.%d.", l)

				normW, _, _ := st.ReadTensorFloat32(prefix + "input_layernorm.weight")
				copy(buf, x)
				rmsNorm(buf, normW, normEps)
				normedBuf := make([]float32, dim)
				copy(normedBuf, buf)

				var wq, wk, wv []float32
				var qRows, qCols, kRows, kCols, vRows, vCols int
				if fusedQKV {
					qkvData, _, _, _ := readWeight(prefix+"self_attn.qkv_proj.weight", attnDim+kvDim+kvDim, dim)
					if qkvData != nil {
						wq = qkvData[:attnDim*dim]
						wk = qkvData[attnDim*dim : (attnDim+kvDim)*dim]
						wv = qkvData[(attnDim+kvDim)*dim:]
						qRows, qCols = attnDim, dim
						kRows, kCols = kvDim, dim
						vRows, vCols = kvDim, dim
					}
				} else {
					wq, qRows, qCols, _ = readWeight(prefix+"self_attn.q_proj.weight", attnDim, dim)
					wk, kRows, kCols, _ = readWeight(prefix+"self_attn.k_proj.weight", kvDim, dim)
					wv, vRows, vCols, _ = readWeight(prefix+"self_attn.v_proj.weight", kvDim, dim)
				}

				streamMatVec(q[:attnDim], wq, buf, qRows, qCols)
				streamMatVec(k[:kvDim], wk, buf, kRows, kCols)
				streamMatVec(v[:kvDim], wv, buf, vRows, vCols)

				if bq, _, e := st.ReadTensorFloat32(prefix + "self_attn.q_proj.bias"); e == nil {
					for i := range bq {
						q[i] += bq[i]
					}
				}
				if bk, _, e := st.ReadTensorFloat32(prefix + "self_attn.k_proj.bias"); e == nil {
					for i := range bk {
						k[i] += bk[i]
					}
				}
				if bv, _, e := st.ReadTensorFloat32(prefix + "self_attn.v_proj.bias"); e == nil {
					for i := range bv {
						v[i] += bv[i]
					}
				}

				if rotaryDim == headDim {
					applyRoPE(q, k[:kvDim], pos, headDim, ropeTheta, heads, kvHeads)
				} else {
					applyRoPEPartial(q, pos, headDim, rotaryDim, heads, ropeTheta)
					applyRoPEPartial(k[:kvDim], pos, headDim, rotaryDim, kvHeads, ropeTheta)
				}

				copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], k[:kvDim])
				copy(valCache[l][pos*kvDim:(pos+1)*kvDim], v[:kvDim])

				kvMul := heads / kvHeads
				for i := range attnOut[:attnDim] {
					attnOut[i] = 0
				}
				for h := 0; h < heads; h++ {
					qOff := h * headDim
					kvH := h / kvMul
					kvOff := kvH * headDim
					scale := float32(1.0 / math.Sqrt(float64(headDim)))
					for t := 0; t <= pos; t++ {
						var dot float32
						for j := 0; j < headDim; j++ {
							dot += q[qOff+j] * keyCache[l][t*kvDim+kvOff+j]
						}
						att[h*(pos+1)+t] = dot * scale
					}
					softmax(att[h*(pos+1):h*(pos+1)+(pos+1)], pos+1)
					for t := 0; t <= pos; t++ {
						w := att[h*(pos+1)+t]
						for j := 0; j < headDim; j++ {
							attnOut[qOff+j] += w * valCache[l][t*kvDim+kvOff+j]
						}
					}
				}

				wo, woRows, woCols, _ := readWeight(prefix+oProjName, dim, attnDim)
				proj := make([]float32, dim)
				streamMatVec(proj, wo, attnOut[:attnDim], woRows, woCols)
				for i := 0; i < dim; i++ {
					x[i] += proj[i]
				}

				normW2, _, _ := st.ReadTensorFloat32(prefix + "post_attention_layernorm.weight")
				if normW2 != nil {
					copy(buf, x)
					rmsNorm(buf, normW2, normEps)
				} else {
					copy(buf, normedBuf)
				}

				if gatedMLP {
					gate, gRows, gCols, _ := readWeight(prefix+mlpWeightNames[0], ffnDim, dim)
					up, uRows, uCols, _ := readWeight(prefix+mlpWeightNames[1], ffnDim, dim)
					down, dRows, dCols, _ := readWeight(prefix+mlpWeightNames[2], dim, ffnDim)
					streamMatVec(ffnBuf, gate, buf, gRows, gCols)
					streamMatVec(ffnBuf2, up, buf, uRows, uCols)
					if geluGated {
						for i := 0; i < ffnDim; i++ {
							ffnBuf[i] = geluNew(ffnBuf[i]) * ffnBuf2[i]
						}
					} else {
						for i := 0; i < ffnDim; i++ {
							ffnBuf[i] = silu(ffnBuf[i]) * ffnBuf2[i]
						}
					}
					downOut := make([]float32, dim)
					streamMatVec(downOut, down, ffnBuf, dRows, dCols)
					for i := 0; i < dim; i++ {
						x[i] += downOut[i]
					}
				} else {
					fc1, f1Rows, f1Cols, _ := readWeight(prefix+mlpWeightNames[0], ffnDim, dim)
					fc2, f2Rows, f2Cols, _ := readWeight(prefix+mlpWeightNames[1], dim, ffnDim)
					streamMatVec(ffnBuf, fc1, buf, f1Rows, f1Cols)
					for i := 0; i < ffnDim; i++ {
						ffnBuf[i] = geluNew(ffnBuf[i])
					}
					downOut := make([]float32, dim)
					streamMatVec(downOut, fc2, ffnBuf, f2Rows, f2Cols)
					for i := 0; i < dim; i++ {
						x[i] += downOut[i]
					}
				}
			}
		}

		if gpuResident {
			rmsNorm(x, gpuFinalNorm, normEps)
			logits := make([]float32, vocabSize)
			gpuResidentMatVec(logits, gpuLMHead, x, vocabSize, dim)
			return logits
		}

		finalNorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
		if finalNorm == nil {
			finalNorm, _, _ = st.ReadTensorFloat32("model.final_layernorm.weight")
		}
		rmsNorm(x, finalNorm, normEps)
		logits := make([]float32, vocabSize)
		streamMatVec(logits, lmHeadData, x, vocabSize, dim)
		softcapLogits(logits)
		return logits
	}

	// === Metal fused compute-shader forward (one command buffer per token) ===
	fusedForward := func(tokenID, pos int) []float32 { return nil }
	useFused := false

	if metal, ok := eng.(*mongoose.Metal); ok && gatedMLP && !geluGated && !fusedQKV && attnDim == dim {
		ret := metal.BuildFused(dim, kvDim, headDim, heads, kvHeads, ffnDim, vocabSize, nLayers, maxSeq, float64(ropeTheta), 1e-6)
		if ret == 0 {
			nw := metal.FusedNumWeights()
			fmt.Printf("  Metal fused compute pipeline (%d weight slots)\n", nw)

			wi := 0
			for l := 0; l < nLayers; l++ {
				prefix := fmt.Sprintf("model.layers.%d.", l)
				loadW := func(name string, rows, cols int) {
					data, _, _, _ := readWeight(prefix+name, rows, cols)
					if data != nil {
						metal.FusedSetWeight(wi, data)
					}
					wi++
				}
				loadNorm := func(name string) {
					data, _, _ := st.ReadTensorFloat32(prefix + name)
					if data != nil {
						metal.FusedSetWeight(wi, data)
					}
					wi++
				}
				loadBias := func(name string, sz int) {
					data, _, _ := st.ReadTensorFloat32(prefix + name)
					if data == nil {
						data = make([]float32, sz)
					}
					metal.FusedSetWeight(wi, data)
					wi++
				}
				loadNorm("input_layernorm.weight")
				loadW("self_attn.q_proj.weight", dim, dim)
				loadW("self_attn.k_proj.weight", kvDim, dim)
				loadW("self_attn.v_proj.weight", kvDim, dim)
				loadBias("self_attn.q_proj.bias", dim)
				loadBias("self_attn.k_proj.bias", kvDim)
				loadBias("self_attn.v_proj.bias", kvDim)
				loadW("self_attn.o_proj.weight", dim, dim)
				loadNorm("post_attention_layernorm.weight")
				loadW("mlp.gate_proj.weight", ffnDim, dim)
				loadW("mlp.up_proj.weight", ffnDim, dim)
				loadW("mlp.down_proj.weight", dim, ffnDim)
			}
			fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
			metal.FusedSetWeight(wi, fnorm)
			wi++
			metal.FusedSetWeight(wi, lmHeadData)
			wi++

			fmt.Printf("  Loaded %d weights\n", wi)
			useFused = true

			fHidden := make([]float32, dim)
			fLogits := make([]float32, vocabSize)

			fusedForward = func(tokenID, pos int) []float32 {
				tokOff := tokenID * dim
				if tokOff+dim > len(embedData) {
					return nil
				}
				copy(fHidden, embedData[tokOff:tokOff+dim])
				cosSlice := cosTab[pos*halfHead : pos*halfHead+halfHead]
				sinSlice := sinTab[pos*halfHead : pos*halfHead+halfHead]
				metal.FusedStep(fHidden, cosSlice, sinSlice, pos, fLogits)
				return fLogits
			}
		}
	}

	// === Metal MPSGraph forward (fallback — 2 dispatches/layer) ===
	metalForward := func(tokenID, pos int) []float32 { return nil }
	useMetalGraph := false

	if !useFused {
		if metal, ok := eng.(*mongoose.Metal); ok && gatedMLP && !geluGated && !fusedQKV && attnDim == dim {
		ret := metal.BuildInferGraph(dim, kvDim, headDim, heads, kvHeads, ffnDim, vocabSize, nLayers, float64(ropeTheta))
		if ret == 0 {
			nw := metal.InferNumWeights()
			fmt.Printf("  Metal inference graph compiled (%d weight slots)\n", nw)

			wi := 0
			for l := 0; l < nLayers; l++ {
				prefix := fmt.Sprintf("model.layers.%d.", l)
				loadW := func(name string, rows, cols int) {
					data, _, _, _ := readWeight(prefix+name, rows, cols)
					if data != nil {
						metal.InferSetWeight(wi, data)
					}
					wi++
				}
				loadNorm := func(name string) {
					data, _, _ := st.ReadTensorFloat32(prefix + name)
					if data != nil {
						metal.InferSetWeight(wi, data)
					}
					wi++
				}
				loadBias := func(name string, sz int) {
					data, _, _ := st.ReadTensorFloat32(prefix + name)
					if data == nil {
						data = make([]float32, sz)
					}
					metal.InferSetWeight(wi, data)
					wi++
				}
				loadNorm("input_layernorm.weight")
				loadW("self_attn.q_proj.weight", dim, dim)
				loadW("self_attn.k_proj.weight", kvDim, dim)
				loadW("self_attn.v_proj.weight", kvDim, dim)
				loadBias("self_attn.q_proj.bias", dim)
				loadBias("self_attn.k_proj.bias", kvDim)
				loadBias("self_attn.v_proj.bias", kvDim)
				loadW("self_attn.o_proj.weight", dim, dim)
				loadNorm("post_attention_layernorm.weight")
				loadW("mlp.gate_proj.weight", ffnDim, dim)
				loadW("mlp.up_proj.weight", ffnDim, dim)
				loadW("mlp.down_proj.weight", dim, ffnDim)
			}
			fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
			metal.InferSetWeight(wi, fnorm)
			wi++
			metal.InferSetWeight(wi, lmHeadData)
			wi++

			fmt.Printf("  Loaded %d weights into Metal graph\n", wi)
			useMetalGraph = true

			mHidden := make([]float32, dim)
			mQBuf := make([]float32, dim)
			mKBuf := make([]float32, kvDim)
			mVBuf := make([]float32, kvDim)
			mAttnOut := make([]float32, dim)
			mLogits := make([]float32, vocabSize)
			invSqrtHeadDim := float32(1.0 / math.Sqrt(float64(headDim)))
			kvMulConst := heads / kvHeads

			metalForward = func(tokenID, pos int) []float32 {
				tokOff := tokenID * dim
				if tokOff+dim > len(embedData) {
					return nil
				}
				copy(mHidden, embedData[tokOff:tokOff+dim])

				cosSlice := cosTab[pos*halfHead : pos*halfHead+halfHead]
				sinSlice := sinTab[pos*halfHead : pos*halfHead+halfHead]

				for l := 0; l < nLayers; l++ {
					metal.InferForwardA(mHidden, cosSlice, sinSlice, mQBuf, mKBuf, mVBuf, l)

					copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], mKBuf)
					copy(valCache[l][pos*kvDim:(pos+1)*kvDim], mVBuf)

					for i := range mAttnOut {
						mAttnOut[i] = 0
					}
					for h := 0; h < heads; h++ {
						qOff := h * headDim
						kvOff := (h / kvMulConst) * headDim
						scores := att[h*(pos+1) : h*(pos+1)+(pos+1)]
						for t := 0; t <= pos; t++ {
							var dot float64
							for j := 0; j < headDim; j++ {
								dot += float64(mQBuf[qOff+j]) * float64(keyCache[l][t*kvDim+kvOff+j])
							}
							scores[t] = float32(dot) * invSqrtHeadDim
						}
						softmax(scores, pos+1)
						for t := 0; t <= pos; t++ {
							w := scores[t]
							for j := 0; j < headDim; j++ {
								mAttnOut[qOff+j] += w * valCache[l][t*kvDim+kvOff+j]
							}
						}
					}

					metal.InferForwardB(mHidden, mAttnOut, l)
				}

				metal.InferLogits(mHidden, mLogits)
				return mLogits
			}
		}
		}
	}

	// Select tier
	fwd := forward
	if useFused {
		fwd = fusedForward
	} else if useCUDAQ8 {
		fwd = gpuForwardQ8
		if cudaUseQ4 {
			fmt.Println("  infer: CUDA Q4 fused matvec")
		} else {
			fmt.Println("  infer: CUDA Q8 fused matvec")
		}
	} else if useMetalGraph {
		fwd = metalForward
	} else if useGPUKernels && gpuResident {
		fwd = gpuForward
	}

	_ = gpuEmbed

	temperature := float32(0.7)
	topK := 40

	fmt.Print("Prefilling... ")
	var logits []float32
	for i, tid := range tokens {
		logits = fwd(tid, i)
	}
	fmt.Println("done")

	nextToken := sampleTopK(logits, temperature, topK)
	maxTokens := 200

	fmt.Printf("\nGenerating (max %d tokens):\n", maxTokens)
	if !isInstruct || rawMode {
		fmt.Print(prompt)
	}

	allTokens := make([]int, len(tokens))
	copy(allTokens, tokens)
	t0 := time.Now()

	for step := 0; step < maxTokens; step++ {
		if stopTokens[nextToken] {
			break
		}

		allTokens = append(allTokens, nextToken)
		decoded := tok.Decode([]int{nextToken})
		if idx := findSpecialToken(decoded); idx >= 0 {
			decoded = decoded[:idx]
			if len(decoded) > 0 {
				fmt.Print(decoded)
			}
			break
		}
		fmt.Print(decoded)

		pos := len(allTokens) - 1
		if pos >= maxSeq-1 {
			break
		}

		logits = fwd(allTokens[pos], pos)
		if logits == nil {
			break
		}
		nextToken = sampleTopK(logits, temperature, topK)
	}

	elapsed := time.Since(t0)
	genTokens := len(allTokens) - len(tokens)
	fmt.Printf("\n\n--- %d tokens in %v (%.1f tok/s) ---\n", genTokens, elapsed,
		float64(genTokens)/elapsed.Seconds())
}

// buildGGUFNameMap creates a mapping from HuggingFace names to GGUF names.
func buildGGUFNameMap(nLayers int) map[string]string {
	m := map[string]string{
		"model.embed_tokens.weight": "token_embd.weight",
		"model.norm.weight":         "output_norm.weight",
		"lm_head.weight":            "output.weight",
	}
	for l := 0; l < nLayers; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		g := fmt.Sprintf("blk.%d.", l)
		m[p+"input_layernorm.weight"] = g + "attn_norm.weight"
		m[p+"self_attn.q_proj.weight"] = g + "attn_q.weight"
		m[p+"self_attn.k_proj.weight"] = g + "attn_k.weight"
		m[p+"self_attn.v_proj.weight"] = g + "attn_v.weight"
		m[p+"self_attn.o_proj.weight"] = g + "attn_output.weight"
		m[p+"post_attention_layernorm.weight"] = g + "ffn_norm.weight"
		m[p+"mlp.gate_proj.weight"] = g + "ffn_gate.weight"
		m[p+"mlp.up_proj.weight"] = g + "ffn_up.weight"
		m[p+"mlp.down_proj.weight"] = g + "ffn_down.weight"
	}
	return m
}
