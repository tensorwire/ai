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

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/mongoose"
	"github.com/tensorwire/tokenizer"
)

// cmdInferGPU runs inference with three tiers:
//   1. GPU-resident (TensorEngine + CUDA kernels) — weights + KV cache in VRAM
//   2. GPU-accelerated (Engine.MatMul) — weights streamed, matmul on GPU (Metal/CUDA/WebGPU)
//   3. CPU fallback — pure Go
func cmdInferGPU(model string, promptParts []string) {
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
	headDim := dim / heads
	kvDim := kvHeads * headDim
	maxSeq := getInt("max_position_embeddings", 2048)
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

	st, err := gguf.OpenSafeTensors(path)
	if err != nil {
		log.Fatalf("open model: %v", err)
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

	tokens := tok.Encode(prompt)
	fmt.Printf("Model: %s (dim=%d layers=%d heads=%d kv=%d ffn=%d vocab=%d act=%s)\n",
		filepath.Base(path), dim, nLayers, heads, kvHeads, ffnDim, vocabSize, act)
	fmt.Printf("Engine: %s\n", eng.Name())
	fmt.Printf("Prompt: %q → %d tokens\n", prompt, len(tokens))

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
		wq, wk, wv, wo    *mongoose.Tensor
		gate, up, down     *mongoose.Tensor
		bq, bk, bv         *mongoose.Tensor
		norm1, norm2       []float32
		gnorm1, gnorm2     *mongoose.Tensor
	}
	var gpuLayers []gpuLayer
	var gpuEmbed, gpuLMHead, gpuFinalNormT *mongoose.Tensor
	var gpuFinalNorm []float32

	if te != nil {
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
			gpuLayers[l] = gpuLayer{
				wq:    loadGPU("self_attn.q_proj.weight", dim, dim),
				wk:    loadGPU("self_attn.k_proj.weight", kvDim, dim),
				wv:    loadGPU("self_attn.v_proj.weight", kvDim, dim),
				wo:    loadGPU("self_attn.o_proj.weight", dim, dim),
				gate:  loadGPU("mlp.gate_proj.weight", ffnDim, dim),
				up:    loadGPU("mlp.up_proj.weight", ffnDim, dim),
				down:  loadGPU("mlp.down_proj.weight", dim, ffnDim),
				norm1: loadCPU("input_layernorm.weight"),
				norm2: loadCPU("post_attention_layernorm.weight"),
			}
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
			fmt.Printf("\r  Layer %d/%d loaded", l+1, nLayers)
		}
		gpuEmbed = te.FromHost(embedData, []int{vocabSize, dim})
		gpuLMHead = te.FromHost(lmHeadData, []int{vocabSize, dim})
		gpuFinalNorm, _, _ = st.ReadTensorFloat32("model.norm.weight")
		gpuFinalNormT = te.FromHost(gpuFinalNorm, []int{1, dim})
		fmt.Println(" — all weights on GPU")

		useGPUKernels = mongoose.TrainKernelsLoaded() && mongoose.KernelsLoaded()
	}

	// RoPE tables
	halfHead := headDim / 2
	cosTab := make([]float32, maxSeq*halfHead)
	sinTab := make([]float32, maxSeq*halfHead)
	for pos := 0; pos < maxSeq; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(float64(ropeTheta), float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			cosTab[pos*halfHead+j] = float32(math.Cos(angle))
			sinTab[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}

	// GPU RoPE tables + KV cache (only for fully GPU-resident path)
	var gpuRopeCos, gpuRopeSin *mongoose.Tensor
	type gpuKVCache struct {
		k, v *mongoose.Tensor
	}
	var gpuKV []gpuKVCache

	if useGPUKernels && gpuResident {
		gpuRopeCos = te.FromHost(cosTab, []int{maxSeq, halfHead})
		gpuRopeSin = te.FromHost(sinTab, []int{maxSeq, halfHead})
		gpuKV = make([]gpuKVCache, nLayers)
		for l := 0; l < nLayers; l++ {
			gpuKV[l] = gpuKVCache{
				k: te.Zeros([]int{maxSeq, kvDim}),
				v: te.Zeros([]int{maxSeq, kvDim}),
			}
		}
		fmt.Println("  infer: GPU-resident (KV cache + RoPE in VRAM)")
	} else if gpuResident {
		fmt.Println("  infer: GPU-accelerated (weights in VRAM, attention on CPU)")
	} else {
		fmt.Println("  infer: CPU (streaming weights)")
	}

	// CPU state buffers (used by tier 2 and 3)
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

			tQ := te.MatMulTransposeBT(normed, gl.wq, 1, dim, dim)
			tK := te.MatMulTransposeBT(normed, gl.wk, 1, dim, kvDim)
			tV := te.MatMulTransposeBT(normed, gl.wv, 1, dim, kvDim)
			te.Release(normed)

			if gl.bq != nil {
				te.AddInPlace(tQ, gl.bq)
			}
			if gl.bk != nil {
				te.AddInPlace(tK, gl.bk)
			}
			if gl.bv != nil {
				te.AddInPlace(tV, gl.bv)
			}

			cosOff := unsafe.Add(gpuRopeCos.DevicePtr(), uintptr(pos*halfHead*4))
			sinOff := unsafe.Add(gpuRopeSin.DevicePtr(), uintptr(pos*halfHead*4))
			mongoose.KRoPE(tQ.DevicePtr(), cosOff, sinOff, 1, dim, headDim, heads)
			mongoose.KRoPE(tK.DevicePtr(), cosOff, sinOff, 1, kvDim, headDim, kvHeads)

			kv := &gpuKV[l]
			mongoose.KCopy(
				unsafe.Add(kv.k.DevicePtr(), uintptr(pos*kvDim*4)),
				tK.DevicePtr(), kvDim*4)
			mongoose.KCopy(
				unsafe.Add(kv.v.DevicePtr(), uintptr(pos*kvDim*4)),
				tV.DevicePtr(), kvDim*4)
			te.Release(tK)
			te.Release(tV)

			tAttnOut := te.Zeros([]int{1, dim})
			mongoose.KDecodeAttention(tQ.DevicePtr(), kv.k.DevicePtr(), kv.v.DevicePtr(),
				tAttnOut.DevicePtr(), pos+1, dim, kvDim, heads, kvHeads)
			te.Release(tQ)

			tProj := te.MatMulTransposeBT(tAttnOut, gl.wo, 1, dim, dim)
			te.Release(tAttnOut)
			te.AddInPlace(xGPU, tProj)
			te.Release(tProj)

			normed2 := te.Zeros([]int{1, dim})
			mongoose.KRMSNormOut(xGPU.DevicePtr(), normed2.DevicePtr(), gl.gnorm2.DevicePtr(), 1, dim)

			tGate := te.MatMulTransposeBT(normed2, gl.gate, 1, dim, ffnDim)
			tUp := te.MatMulTransposeBT(normed2, gl.up, 1, dim, ffnDim)
			te.Release(normed2)

			ffnMid := te.Zeros([]int{1, ffnDim})
			mongoose.KSiLUGateMul(tGate.DevicePtr(), tUp.DevicePtr(), ffnMid.DevicePtr(), ffnDim)
			te.Release(tGate)
			te.Release(tUp)

			tDown := te.MatMulTransposeBT(ffnMid, gl.down, 1, ffnDim, dim)
			te.Release(ffnMid)
			te.AddInPlace(xGPU, tDown)
			te.Release(tDown)
		}

		mongoose.KRMSNorm(xGPU.DevicePtr(), gpuFinalNormT.DevicePtr(), 1, dim)

		tLogits := te.MatMulTransposeBT(xGPU, gpuLMHead, 1, dim, vocabSize)
		te.Release(xGPU)

		logits := te.ToHost(tLogits)
		te.Release(tLogits)
		return logits
	}

	// === CUDA Q8/Q4 fused path ===
	gpuForwardQ8 := func(tokenID, pos int) []float32 { return nil }
	useCUDAQ8 := false
	cudaUseQ4 := false

	if cuda, ok := eng.(*mongoose.CUDA); ok && mongoose.HasQ8Matvec() && mongoose.KernelsLoaded() {
		_ = cuda

		nParams := int64(vocabSize)*int64(dim)*2 // embed + lmHead
		for l := 0; l < nLayers; l++ {
			nParams += int64(dim)*int64(dim)*2 + int64(kvDim)*int64(dim)*2 + int64(ffnDim)*int64(dim)*3
		}
		cudaUseQ4 = mongoose.HasQ4Matvec() && nParams > 4_000_000_000

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
			wq, wk, wv, wo    q8Weight
			gate, up, down     q8Weight
			bq, bk, bv         *mongoose.Tensor
			gnorm1, gnorm2     *mongoose.Tensor
		}

		qLabel := "Q8"
		if cudaUseQ4 {
			qLabel = "Q4"
		}
		fmt.Printf("Quantizing weights to %s (%.1fB params)... ", qLabel, float64(nParams)/1e9)
		q8Layers := make([]q8Layer, nLayers)
		for l := 0; l < nLayers; l++ {
			gl := &gpuLayers[l]
			q8l := &q8Layers[l]
			q8l.gnorm1 = gl.gnorm1
			q8l.gnorm2 = gl.gnorm2
			q8l.bq = gl.bq
			q8l.bk = gl.bk
			q8l.bv = gl.bv

			prefix := fmt.Sprintf("model.layers.%d.", l)
			loadQ8 := func(name string, rows, cols int) q8Weight {
				data, _, _, _ := readWeight(prefix+name, rows, cols)
				return quantizeToGPU(data, rows, cols)
			}
			q8l.wq = loadQ8("self_attn.q_proj.weight", dim, dim)
			q8l.wk = loadQ8("self_attn.k_proj.weight", kvDim, dim)
			q8l.wv = loadQ8("self_attn.v_proj.weight", kvDim, dim)
			q8l.wo = loadQ8("self_attn.o_proj.weight", dim, dim)
			q8l.gate = loadQ8("mlp.gate_proj.weight", ffnDim, dim)
			q8l.up = loadQ8("mlp.up_proj.weight", ffnDim, dim)
			q8l.down = loadQ8("mlp.down_proj.weight", dim, ffnDim)
		}
		q8LMHead := quantizeToGPU(lmHeadData, vocabSize, dim)
		fmt.Println("done")

		useCUDAQ8 = true

		q8MV := func(actPtr unsafe.Pointer, w q8Weight, outPtr unsafe.Pointer) {
			if w.q4 {
				mongoose.KQ4Matvec(actPtr, w.data, w.scales, outPtr, w.rows, w.cols)
			} else {
				mongoose.KQ8Matvec(actPtr, w.data, w.scales, outPtr, w.rows, w.cols)
			}
		}

		gpuForwardQ8 = func(tokenID, pos int) []float32 {
			tokOff := tokenID * dim
			if tokOff+dim > len(embedData) {
				return nil
			}
			xGPU := te.FromHost(embedData[tokOff:tokOff+dim], []int{1, dim})

			for l := 0; l < nLayers; l++ {
				ql := &q8Layers[l]

				normed := te.Zeros([]int{1, dim})
				mongoose.KRMSNormOut(xGPU.DevicePtr(), normed.DevicePtr(), ql.gnorm1.DevicePtr(), 1, dim)

				tQ := te.Zeros([]int{1, dim})
				tK := te.Zeros([]int{1, kvDim})
				tV := te.Zeros([]int{1, kvDim})
				q8MV(normed.DevicePtr(), ql.wq, tQ.DevicePtr())
				q8MV(normed.DevicePtr(), ql.wk, tK.DevicePtr())
				q8MV(normed.DevicePtr(), ql.wv, tV.DevicePtr())
				te.Release(normed)

				if ql.bq != nil {
					te.AddInPlace(tQ, ql.bq)
				}
				if ql.bk != nil {
					te.AddInPlace(tK, ql.bk)
				}
				if ql.bv != nil {
					te.AddInPlace(tV, ql.bv)
				}

				cosOff := unsafe.Add(gpuRopeCos.DevicePtr(), uintptr(pos*halfHead*4))
				sinOff := unsafe.Add(gpuRopeSin.DevicePtr(), uintptr(pos*halfHead*4))
				mongoose.KRoPE(tQ.DevicePtr(), cosOff, sinOff, 1, dim, headDim, heads)
				mongoose.KRoPE(tK.DevicePtr(), cosOff, sinOff, 1, kvDim, headDim, kvHeads)

				kv := &gpuKV[l]
				mongoose.KCopy(
					unsafe.Add(kv.k.DevicePtr(), uintptr(pos*kvDim*4)),
					tK.DevicePtr(), kvDim*4)
				mongoose.KCopy(
					unsafe.Add(kv.v.DevicePtr(), uintptr(pos*kvDim*4)),
					tV.DevicePtr(), kvDim*4)
				te.Release(tK)
				te.Release(tV)

				tAttnOut := te.Zeros([]int{1, dim})
				mongoose.KDecodeAttention(tQ.DevicePtr(), kv.k.DevicePtr(), kv.v.DevicePtr(),
					tAttnOut.DevicePtr(), pos+1, dim, kvDim, heads, kvHeads)
				te.Release(tQ)

				tProj := te.Zeros([]int{1, dim})
				q8MV(tAttnOut.DevicePtr(), ql.wo, tProj.DevicePtr())
				te.Release(tAttnOut)
				te.AddInPlace(xGPU, tProj)
				te.Release(tProj)

				normed2 := te.Zeros([]int{1, dim})
				mongoose.KRMSNormOut(xGPU.DevicePtr(), normed2.DevicePtr(), ql.gnorm2.DevicePtr(), 1, dim)

				tGate := te.Zeros([]int{1, ffnDim})
				tUp := te.Zeros([]int{1, ffnDim})
				q8MV(normed2.DevicePtr(), ql.gate, tGate.DevicePtr())
				q8MV(normed2.DevicePtr(), ql.up, tUp.DevicePtr())
				te.Release(normed2)

				ffnMid := te.Zeros([]int{1, ffnDim})
				mongoose.KSiLUGateMul(tGate.DevicePtr(), tUp.DevicePtr(), ffnMid.DevicePtr(), ffnDim)
				te.Release(tGate)
				te.Release(tUp)

				tDown := te.Zeros([]int{1, dim})
				q8MV(ffnMid.DevicePtr(), ql.down, tDown.DevicePtr())
				te.Release(ffnMid)
				te.AddInPlace(xGPU, tDown)
				te.Release(tDown)
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

				applyRoPE(q, k[:kvDim], pos, headDim, ropeTheta, heads, kvHeads)

				copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], k[:kvDim])
				copy(valCache[l][pos*kvDim:(pos+1)*kvDim], v[:kvDim])

				kvMul := heads / kvHeads
				for i := range attnOut[:dim] {
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

				wq, qRows, qCols, _ := readWeight(prefix+"self_attn.q_proj.weight", dim, dim)
				wk, kRows, kCols, _ := readWeight(prefix+"self_attn.k_proj.weight", kvDim, dim)
				wv, vRows, vCols, _ := readWeight(prefix+"self_attn.v_proj.weight", kvDim, dim)

				streamMatVec(q, wq, buf, qRows, qCols)
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

				applyRoPE(q, k[:kvDim], pos, headDim, ropeTheta, heads, kvHeads)

				copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], k[:kvDim])
				copy(valCache[l][pos*kvDim:(pos+1)*kvDim], v[:kvDim])

				kvMul := heads / kvHeads
				for i := range attnOut[:dim] {
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

				wo, woRows, woCols, _ := readWeight(prefix+"self_attn.o_proj.weight", dim, dim)
				proj := make([]float32, dim)
				streamMatVec(proj, wo, attnOut, woRows, woCols)
				for i := 0; i < dim; i++ {
					x[i] += proj[i]
				}

				normW2, _, _ := st.ReadTensorFloat32(prefix + "post_attention_layernorm.weight")
				copy(buf, x)
				rmsNorm(buf, normW2, normEps)

				gate, gRows, gCols, _ := readWeight(prefix+"mlp.gate_proj.weight", ffnDim, dim)
				up, uRows, uCols, _ := readWeight(prefix+"mlp.up_proj.weight", ffnDim, dim)
				down, dRows, dCols, _ := readWeight(prefix+"mlp.down_proj.weight", dim, ffnDim)
				streamMatVec(ffnBuf, gate, buf, gRows, gCols)
				streamMatVec(ffnBuf2, up, buf, uRows, uCols)
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
				streamMatVec(downOut, down, ffnBuf, dRows, dCols)
				for i := 0; i < dim; i++ {
					x[i] += downOut[i]
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
		rmsNorm(x, finalNorm, normEps)
		logits := make([]float32, vocabSize)
		streamMatVec(logits, lmHeadData, x, vocabSize, dim)
		return logits
	}

	// === Metal fused compute-shader forward (one command buffer per token) ===
	fusedForward := func(tokenID, pos int) []float32 { return nil }
	useFused := false

	if metal, ok := eng.(*mongoose.Metal); ok {
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
		if metal, ok := eng.(*mongoose.Metal); ok {
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
	fmt.Print(prompt)

	allTokens := make([]int, len(tokens))
	copy(allTokens, tokens)
	allTokens = append(allTokens, nextToken)
	fmt.Print(tok.Decode([]int{nextToken}))
	t0 := time.Now()

	for step := 1; step < maxTokens; step++ {
		pos := len(allTokens) - 1
		if pos >= maxSeq-1 {
			break
		}

		logits = fwd(allTokens[pos], pos)
		if logits == nil {
			break
		}

		nextToken = sampleTopK(logits, temperature, topK)
		allTokens = append(allTokens, nextToken)

		fmt.Print(tok.Decode([]int{nextToken}))

		if nextToken == tok.EOS || nextToken == 0 {
			break
		}
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
