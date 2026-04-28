package main

import (
	"fmt"
	"log"

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/mongoose"
)

// cudaQ8Inference holds pre-allocated GPU state for zero-alloc CUDA Q8 inference.
// All scratch buffers allocated once at model load, reused every token.
type cudaQ8Inference struct {
	cuda *mongoose.CUDA
	te   mongoose.TensorEngine

	dim, kvDim, ffnDim, vocabSize int
	heads, kvHeads, headDim       int
	halfHead, maxSeq, nLayers     int

	// Per-layer quantized weights (persistent, read-only)
	layers  []cudaQ8Layer
	lmHead  mongoose.Q8Weight

	// Persistent GPU tensors (norms, biases, RoPE tables)
	gpuFinalNorm     *mongoose.Tensor
	gpuRopeCos       *mongoose.Tensor
	gpuRopeSin       *mongoose.Tensor

	// Per-layer KV caches
	kCache []*mongoose.Tensor
	vCache []*mongoose.Tensor

	// Pre-allocated scratch buffers — reused every token, zero allocs in hot path
	xGPU    *mongoose.Tensor // [1, dim]     — residual stream
	normed  *mongoose.Tensor // [1, dim]     — post-RMSNorm
	tQ      *mongoose.Tensor // [1, dim]     — Q projection
	tK      *mongoose.Tensor // [1, kvDim]   — K projection
	tV      *mongoose.Tensor // [1, kvDim]   — V projection
	attnOut *mongoose.Tensor // [1, dim]     — attention output
	proj    *mongoose.Tensor // [1, dim]     — Wo projection / down projection
	normed2 *mongoose.Tensor // [1, dim]     — post-attention RMSNorm
	gate    *mongoose.Tensor // [1, ffnDim]  — gate projection
	up      *mongoose.Tensor // [1, ffnDim]  — up projection
	ffnMid  *mongoose.Tensor // [1, ffnDim]  — SiLU(gate) * up
	logits  *mongoose.Tensor // [1, vocabSize] — output logits

	// Host-side logits buffer (reused)
	logitsBuf []float32
}

type cudaQ8Layer struct {
	wq, wk, wv, wo mongoose.Q8Weight
	gate, up, down mongoose.Q8Weight
	bq, bk, bv     *mongoose.Tensor
	gnorm1, gnorm2 *mongoose.Tensor
}

// buildCUDAQ8Inference sets up the fused CUDA Q8 inference engine.
// All GPU memory allocated here, nothing allocated per-token.
func buildCUDAQ8Inference(s *serveState, st *gguf.SafeTensors, lmHeadData []float32) *cudaQ8Inference {
	cuda, ok := s.eng.(*mongoose.CUDA)
	if !ok || !mongoose.HasQ8Matvec() || !mongoose.KernelsLoaded() {
		return nil
	}
	te := mongoose.AsTensorEngine(s.eng)
	if te == nil {
		return nil
	}

	headDim := s.dim / s.heads
	kvDim := s.kvHeads * headDim

	ci := &cudaQ8Inference{
		cuda:      cuda,
		te:        te,
		dim:       s.dim,
		kvDim:     kvDim,
		ffnDim:    s.ffnDim,
		vocabSize: s.vocabSize,
		heads:     s.heads,
		kvHeads:   s.kvHeads,
		headDim:   headDim,
		halfHead:  s.halfHead,
		maxSeq:    s.maxSeq,
		nLayers:   s.layers,
	}

	nParams := int64(s.vocabSize) * int64(s.dim) * 2
	for l := 0; l < s.layers; l++ {
		nParams += int64(s.dim)*int64(s.dim)*2 + int64(kvDim)*int64(s.dim)*2 + int64(s.ffnDim)*int64(s.dim)*3
	}
	useQ4 := mongoose.HasQ4Matvec() && nParams > 4_000_000_000

	qLabel := "Q8"
	if useQ4 {
		qLabel = "Q4"
	}
	log.Printf("[serve] Quantizing weights to %s (%.1fB params)...", qLabel, float64(nParams)/1e9)

	// Quantize and upload all layer weights
	ci.layers = make([]cudaQ8Layer, s.layers)
	for l := 0; l < s.layers; l++ {
		prefix := fmt.Sprintf("model.layers.%d.", l)
		loadQ8 := func(name string, rows, cols int) mongoose.Q8Weight {
			data, _, _ := st.ReadTensorFloat32(prefix + name)
			return quantizeWeight(te, data, rows, cols, useQ4)
		}
		ql := &ci.layers[l]
		ql.wq = loadQ8("self_attn.q_proj.weight", s.dim, s.dim)
		ql.wk = loadQ8("self_attn.k_proj.weight", kvDim, s.dim)
		ql.wv = loadQ8("self_attn.v_proj.weight", kvDim, s.dim)
		ql.wo = loadQ8("self_attn.o_proj.weight", s.dim, s.dim)
		ql.gate = loadQ8("mlp.gate_proj.weight", s.ffnDim, s.dim)
		ql.up = loadQ8("mlp.up_proj.weight", s.ffnDim, s.dim)
		ql.down = loadQ8("mlp.down_proj.weight", s.dim, s.ffnDim)

		norm1, _, _ := st.ReadTensorFloat32(prefix + "input_layernorm.weight")
		norm2, _, _ := st.ReadTensorFloat32(prefix + "post_attention_layernorm.weight")
		ql.gnorm1 = te.FromHost(norm1, []int{1, s.dim})
		ql.gnorm2 = te.FromHost(norm2, []int{1, s.dim})

		if bq, _, e := st.ReadTensorFloat32(prefix + "self_attn.q_proj.bias"); e == nil {
			ql.bq = te.FromHost(bq, []int{1, s.dim})
		}
		if bk, _, e := st.ReadTensorFloat32(prefix + "self_attn.k_proj.bias"); e == nil {
			ql.bk = te.FromHost(bk, []int{1, kvDim})
		}
		if bv, _, e := st.ReadTensorFloat32(prefix + "self_attn.v_proj.bias"); e == nil {
			ql.bv = te.FromHost(bv, []int{1, kvDim})
		}
	}

	ci.lmHead = quantizeWeight(te, lmHeadData, s.vocabSize, s.dim, useQ4)

	fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
	ci.gpuFinalNorm = te.FromHost(fnorm, []int{1, s.dim})
	ci.gpuRopeCos = te.FromHost(s.cosTab, []int{s.maxSeq, s.halfHead})
	ci.gpuRopeSin = te.FromHost(s.sinTab, []int{s.maxSeq, s.halfHead})

	// KV caches
	ci.kCache = make([]*mongoose.Tensor, s.layers)
	ci.vCache = make([]*mongoose.Tensor, s.layers)
	for l := 0; l < s.layers; l++ {
		ci.kCache[l] = te.Zeros([]int{s.maxSeq, kvDim})
		ci.vCache[l] = te.Zeros([]int{s.maxSeq, kvDim})
	}

	// Pre-allocate ALL scratch buffers — the hot path does zero allocs
	ci.xGPU = te.Zeros([]int{1, s.dim})
	ci.normed = te.Zeros([]int{1, s.dim})
	ci.tQ = te.Zeros([]int{1, s.dim})
	ci.tK = te.Zeros([]int{1, kvDim})
	ci.tV = te.Zeros([]int{1, kvDim})
	ci.attnOut = te.Zeros([]int{1, s.dim})
	ci.proj = te.Zeros([]int{1, s.dim})
	ci.normed2 = te.Zeros([]int{1, s.dim})
	ci.gate = te.Zeros([]int{1, s.ffnDim})
	ci.up = te.Zeros([]int{1, s.ffnDim})
	ci.ffnMid = te.Zeros([]int{1, s.ffnDim})
	ci.logits = te.Zeros([]int{1, s.vocabSize})
	ci.logitsBuf = make([]float32, s.vocabSize)

	log.Printf("[serve] CUDA %s fused inference ready (zero-alloc hot path)", qLabel)
	return ci
}

// forward runs one token through the model. Zero GPU allocations.
func (ci *cudaQ8Inference) forward(tokenID, pos int, embedData []float32) []float32 {
	tokOff := tokenID * ci.dim
	if tokOff+ci.dim > len(embedData) {
		return nil
	}

	// Upload embedding into pre-allocated xGPU — no alloc
	ci.cuda.UploadInto(ci.xGPU, embedData[tokOff:tokOff+ci.dim])

	for l := 0; l < ci.nLayers; l++ {
		ql := &ci.layers[l]

		// Attention block: RMSNorm → QKV matvec → bias → RoPE → KV cache → decode attention → Wo → residual
		mongoose.TRMSNormOut(ci.xGPU, ci.normed, ql.gnorm1, 1, ci.dim)

		mongoose.TQ8Matvec(ci.normed, ql.wq, ci.tQ)
		mongoose.TQ8Matvec(ci.normed, ql.wk, ci.tK)
		mongoose.TQ8Matvec(ci.normed, ql.wv, ci.tV)

		if ql.bq != nil {
			mongoose.TAddInPlace(ci.tQ, ql.bq, ci.dim)
		}
		if ql.bk != nil {
			mongoose.TAddInPlace(ci.tK, ql.bk, ci.kvDim)
		}
		if ql.bv != nil {
			mongoose.TAddInPlace(ci.tV, ql.bv, ci.kvDim)
		}

		ropeByteOff := pos * ci.halfHead * 4
		mongoose.TRoPEAt(ci.tQ, ci.gpuRopeCos, ci.gpuRopeSin, ropeByteOff, 1, ci.dim, ci.headDim, ci.heads)
		mongoose.TRoPEAt(ci.tK, ci.gpuRopeCos, ci.gpuRopeSin, ropeByteOff, 1, ci.kvDim, ci.headDim, ci.kvHeads)

		kvByteOff := pos * ci.kvDim * 4
		mongoose.TCopyAt(ci.kCache[l], kvByteOff, ci.tK, ci.kvDim*4)
		mongoose.TCopyAt(ci.vCache[l], kvByteOff, ci.tV, ci.kvDim*4)

		mongoose.TDecodeAttention(ci.tQ, ci.kCache[l], ci.vCache[l], ci.attnOut, pos+1, ci.dim, ci.kvDim, ci.heads, ci.kvHeads)

		mongoose.TQ8Matvec(ci.attnOut, ql.wo, ci.proj)
		mongoose.TAddInPlace(ci.xGPU, ci.proj, ci.dim)

		// FFN block: RMSNorm → gate/up matvec → SiLU gate mul → down matvec → residual
		mongoose.TRMSNormOut(ci.xGPU, ci.normed2, ql.gnorm2, 1, ci.dim)

		mongoose.TQ8Matvec(ci.normed2, ql.gate, ci.gate)
		mongoose.TQ8Matvec(ci.normed2, ql.up, ci.up)

		mongoose.TSiLUGateMul(ci.gate, ci.up, ci.ffnMid, ci.ffnDim)

		mongoose.TQ8Matvec(ci.ffnMid, ql.down, ci.proj)
		mongoose.TAddInPlace(ci.xGPU, ci.proj, ci.dim)
	}

	// Final norm → lm_head → logits
	mongoose.TRMSNorm(ci.xGPU, ci.gpuFinalNorm, 1, ci.dim)
	mongoose.TQ8Matvec(ci.xGPU, ci.lmHead, ci.logits)

	// Download logits — single host transfer per token
	copy(ci.logitsBuf, ci.te.ToHost(ci.logits))
	return ci.logitsBuf
}

// resetKV zeroes all KV caches between conversations.
func (ci *cudaQ8Inference) resetKV() {
	for l := 0; l < ci.nLayers; l++ {
		mongoose.TZero(ci.kCache[l], ci.maxSeq*ci.kvDim*4)
		mongoose.TZero(ci.vCache[l], ci.maxSeq*ci.kvDim*4)
	}
}

// quantizeWeight converts FP32 weights to Q8 or Q4 and uploads to GPU.
func quantizeWeight(te mongoose.TensorEngine, fp32 []float32, rows, cols int, useQ4 bool) mongoose.Q8Weight {
	scales := make([]float32, rows)
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
		scales[r] = absMax
	}

	if useQ4 {
		packed := make([]byte, rows*cols/2)
		for r := 0; r < rows; r++ {
			invScale := float32(0)
			if scales[r] > 0 {
				invScale = 7.0 / scales[r]
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
				if q0 < 0 {
					q0 = 0
				}
				if q0 > 15 {
					q0 = 15
				}
				if q1 < 0 {
					q1 = 0
				}
				if q1 > 15 {
					q1 = 15
				}
				packed[r*(cols/2)+c/2] = byte(q0 | (q1 << 4))
			}
		}
		buf := mongoose.PackBytesToFloat32(packed)
		tData := te.FromHost(buf, []int{1, len(buf)})
		tScales := te.FromHost(scales, []int{1, rows})
		return mongoose.Q8Weight{Data: tData, Scales: tScales, Rows: rows, Cols: cols, Q4: true}
	}

	q := make([]int8, rows*cols)
	for r := 0; r < rows; r++ {
		invScale := float32(0)
		if scales[r] > 0 {
			invScale = 127.0 / scales[r]
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
	buf := mongoose.PackInt8ToFloat32(q)
	tData := te.FromHost(buf, []int{1, len(buf)})
	tScales := te.FromHost(scales, []int{1, rows})
	return mongoose.Q8Weight{Data: tData, Scales: tScales, Rows: rows, Cols: cols, Q4: false}
}
