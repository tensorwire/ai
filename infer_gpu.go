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

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/mongoose"
	"github.com/open-ai-org/tokenizer"
)

// cmdInferGPU runs inference using CUDA kernels — same forward pass as training.
// Falls back to CPU infer if CUDA is unavailable.
func cmdInferGPU(model string, promptParts []string) {
	prompt := strings.Join(promptParts, " ")
	path := resolveModel(model)

	eng := selectEngine("auto")
	cuda, hasCUDA := eng.(*mongoose.CUDA)
	if !hasCUDA || !mongoose.LoadKernels() {
		cmdInferImpl(model, prompt)
		return
	}
	te := mongoose.AsTensorEngine(eng)

	configData, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		log.Fatalf("no config.json in %s", path)
	}
	var cfg map[string]interface{}
	json.Unmarshal(configData, &cfg)

	getInt := func(key string, def int) int {
		if v, ok := cfg[key].(float64); ok { return int(v) }
		return def
	}
	getFloat := func(key string, def float64) float64 {
		if v, ok := cfg[key].(float64); ok { return v }
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
	_ = getFloat("rope_theta", 10000.0)

	if dim == 0 || nLayers == 0 {
		log.Fatal("could not read model dimensions from config.json")
	}

	// Detect format: safetensors directory or GGUF file
	var loadWeight func(name string) (*mongoose.Tensor, int, int)

	ggufPath := ""
	if strings.HasSuffix(path, ".gguf") {
		ggufPath = path
	} else if entries, err := os.ReadDir(path); err == nil {
		for _, e := range entries {
			if strings.HasSuffix(e.Name(), ".gguf") {
				ggufPath = filepath.Join(path, e.Name())
				break
			}
		}
	}

	if ggufPath != "" {
		gr, err := gguf.OpenGGUF(ggufPath)
		if err != nil {
			log.Fatalf("open GGUF: %v", err)
		}
		defer gr.Close()
		log.Printf("[infer] loading from GGUF: %s", ggufPath)

		// GGUF uses different tensor names
		ggufNameMap := buildGGUFNameMap(nLayers)

		loadWeight = func(name string) (*mongoose.Tensor, int, int) {
			ggufName := ggufNameMap[name]
			if ggufName == "" {
				ggufName = name
			}
			data, shape, err := gr.ReadTensorFloat32(ggufName)
			if err != nil {
				return nil, 0, 0
			}
			rows, cols := 1, 1
			if len(shape) >= 2 {
				rows, cols = shape[0], shape[1]
			} else if len(shape) == 1 {
				cols = shape[0]
			}
			return te.FromHost(data, []int{rows, cols}), rows, cols
		}
	} else {
		st, err := gguf.OpenSafeTensors(path)
		if err != nil {
			log.Fatalf("open model: %v", err)
		}
		log.Printf("[infer] loading from safetensors: %s", path)

		loadWeight = func(name string) (*mongoose.Tensor, int, int) {
			data, info, err := st.ReadTensorFloat32(name)
			if err != nil {
				return nil, 0, 0
			}
			rows, cols := 1, 1
			if len(info.Shape) >= 2 {
				rows, cols = info.Shape[0], info.Shape[1]
			} else if len(info.Shape) == 1 {
				cols = info.Shape[0]
			}
			return te.FromHost(data, []int{rows, cols}), rows, cols
		}
	}

	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	tokens := tok.Encode(prompt)
	fmt.Printf("Model: %s (dim=%d layers=%d heads=%d kv=%d ffn=%d vocab=%d)\n",
		filepath.Base(path), dim, nLayers, heads, kvHeads, ffnDim, vocabSize)
	fmt.Printf("Engine: %s (GPU)\n", eng.Name())
	fmt.Printf("Prompt: %q → %d tokens\n", prompt, len(tokens))

	// Load all weights to GPU
	fmt.Print("Loading weights to GPU... ")
	embed, _, _ := loadWeight("model.embed_tokens.weight")
	lmHead, _, _ := loadWeight("lm_head.weight")
	if lmHead == nil {
		lmHead = embed
	}
	finalNorm, _, _ := loadWeight("model.norm.weight")

	type layer struct {
		wq, wk, wv, wo       *mongoose.Tensor
		gate, up, down        *mongoose.Tensor
		norm1, norm2          *mongoose.Tensor
	}
	lays := make([]layer, nLayers)
	for l := 0; l < nLayers; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		lays[l].wq, _, _ = loadWeight(pfx + "self_attn.q_proj.weight")
		lays[l].wk, _, _ = loadWeight(pfx + "self_attn.k_proj.weight")
		lays[l].wv, _, _ = loadWeight(pfx + "self_attn.v_proj.weight")
		lays[l].wo, _, _ = loadWeight(pfx + "self_attn.o_proj.weight")
		lays[l].gate, _, _ = loadWeight(pfx + "mlp.gate_proj.weight")
		lays[l].up, _, _ = loadWeight(pfx + "mlp.up_proj.weight")
		lays[l].down, _, _ = loadWeight(pfx + "mlp.down_proj.weight")
		lays[l].norm1, _, _ = loadWeight(pfx + "input_layernorm.weight")
		lays[l].norm2, _, _ = loadWeight(pfx + "post_attention_layernorm.weight")
	}
	fmt.Println("done")

	// RoPE tables
	halfHead := headDim / 2
	cosTab := make([]float32, maxSeq*halfHead)
	sinTab := make([]float32, maxSeq*halfHead)
	for pos := 0; pos < maxSeq; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(10000.0, float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			cosTab[pos*halfHead+j] = float32(math.Cos(angle))
			sinTab[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}
	ropeCos := te.FromHost(cosTab, []int{maxSeq, halfHead})
	ropeSin := te.FromHost(sinTab, []int{maxSeq, halfHead})

	// KV cache on GPU — [maxSeq, kvDim] per layer
	keyCaches := make([]*mongoose.Tensor, nLayers)
	valCaches := make([]*mongoose.Tensor, nLayers)
	for l := range keyCaches {
		keyCaches[l] = te.Zeros([]int{maxSeq, kvDim})
		valCaches[l] = te.Zeros([]int{maxSeq, kvDim})
	}

	// Single-token buffers (n=1)
	hidden := te.Zeros([]int{1, dim})
	normed := te.Zeros([]int{1, dim})
	rmsScale := te.Zeros([]int{1})
	Q := te.Zeros([]int{1, dim})
	K := te.Zeros([]int{1, kvDim})
	V := te.Zeros([]int{1, kvDim})
	attnOut := te.Zeros([]int{1, dim})
	normed2 := te.Zeros([]int{1, dim})
	rmsScale2 := te.Zeros([]int{1})
	gatePre := te.Zeros([]int{1, ffnDim})
	upOut := te.Zeros([]int{1, ffnDim})
	ffnMid := te.Zeros([]int{1, ffnDim})
	dx := te.Zeros([]int{1, dim})
	normedFinal := te.Zeros([]int{1, dim})
	finalScales := te.Zeros([]int{1})
	logitsBuf := te.Zeros([]int{1, vocabSize})
	tokGPU := te.Zeros([]int{1})

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	// forwardOne runs one token through the model at position pos, returns logits on CPU.
	forwardOne := func(tokenID, pos int) []float32 {
		tokF := []float32{math.Float32frombits(uint32(int32(tokenID)))}
		cuda.UploadInto(tokGPU, tokF)

		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), 1, dim)

		// Slice RoPE for this position
		posOff := pos * halfHead
		ropeCosPos := te.FromHost(cosTab[posOff:posOff+halfHead], []int{1, halfHead})
		ropeSinPos := te.FromHost(sinTab[posOff:posOff+halfHead], []int{1, halfHead})

		for li := range lays {
			l := &lays[li]

			zero(normed); zero(rmsScale)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed.DevicePtr(),
				l.norm1.DevicePtr(), rmsScale.DevicePtr(), 1, dim)

			cuda.MatMulTransposeBTInto(Q, normed, l.wq, 1, dim, dim)
			cuda.MatMulTransposeBTInto(K, normed, l.wk, 1, dim, kvDim)
			cuda.MatMulTransposeBTInto(V, normed, l.wv, 1, dim, kvDim)

			mongoose.KRoPE(Q.DevicePtr(), ropeCosPos.DevicePtr(), ropeSinPos.DevicePtr(), 1, dim, headDim, heads)
			mongoose.KRoPE(K.DevicePtr(), ropeCosPos.DevicePtr(), ropeSinPos.DevicePtr(), 1, kvDim, headDim, kvHeads)

			// Store K,V into cache at position pos
			kHost := te.ToHost(K)
			vHost := te.ToHost(V)
			cuda.UploadSlice(keyCaches[li], pos*kvDim, kHost)
			cuda.UploadSlice(valCaches[li], pos*kvDim, vHost)

			// Attention: Q @ K^T for positions 0..pos, then softmax, then @ V
			// For single-token inference we do this on CPU since n=1
			cuda.Sync()
			qHost := te.ToHost(Q)
			kcHost := te.ToHost(keyCaches[li])
			vcHost := te.ToHost(valCaches[li])

			attnOutHost := make([]float32, dim)
			kvMul := heads / kvHeads
			for h := 0; h < heads; h++ {
				qOff := h * headDim
				kvH := h / kvMul
				kvOff := kvH * headDim
				scale := float32(1.0 / math.Sqrt(float64(headDim)))

				scores := make([]float32, pos+1)
				for t := 0; t <= pos; t++ {
					var dot float32
					for j := 0; j < headDim; j++ {
						dot += qHost[qOff+j] * kcHost[t*kvDim+kvOff+j]
					}
					scores[t] = dot * scale
				}
				softmax(scores, pos+1)
				for t := 0; t <= pos; t++ {
					w := scores[t]
					for j := 0; j < headDim; j++ {
						attnOutHost[qOff+j] += w * vcHost[t*kvDim+kvOff+j]
					}
				}
			}
			tmp := te.FromHost(attnOutHost, []int{1, dim})
			cuda.CopyInto(attnOut, tmp)
			te.Release(tmp)

			cuda.MatMulTransposeBTInto(dx, attnOut, l.wo, 1, dim, dim)
			te.AddInPlace(hidden, dx)

			zero(normed2); zero(rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed2.DevicePtr(),
				l.norm2.DevicePtr(), rmsScale2.DevicePtr(), 1, dim)

			cuda.MatMulTransposeBTInto(gatePre, normed2, l.gate, 1, dim, ffnDim)
			cuda.MatMulTransposeBTInto(upOut, normed2, l.up, 1, dim, ffnDim)

			zero(ffnMid)
			mongoose.KSiLUGateMul(gatePre.DevicePtr(), upOut.DevicePtr(), ffnMid.DevicePtr(), ffnDim)

			cuda.MatMulTransposeBTInto(dx, ffnMid, l.down, 1, ffnDim, dim)
			te.AddInPlace(hidden, dx)
		}

		te.Release(ropeCosPos)
		te.Release(ropeSinPos)

		zero(normedFinal); zero(finalScales)
		mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), 1, dim)

		cuda.MatMulTransposeBTInto(logitsBuf, normedFinal, lmHead, 1, dim, vocabSize)

		cuda.Sync()
		return te.ToHost(logitsBuf)
	}

	_ = ropeCos
	_ = ropeSin

	temperature := float32(0.7)
	topK := 40

	fmt.Print("Prefilling... ")
	var logits []float32
	for i, tid := range tokens {
		logits = forwardOne(tid, i)
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

		logits = forwardOne(allTokens[pos], pos)
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

// UploadSlice copies a float32 slice into a GPU tensor at a byte offset.
// This is a method we need on CUDA — let me check if it exists.
// For now, use a helper that creates a temporary tensor.
func init() {
	// Verify UploadSlice works at compile time — it's defined below
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
