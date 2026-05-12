package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/tensorwire/mongoose"
	"github.com/tensorwire/tokenizer"
)

func suppressOutput() (restore func()) {
	origLog := log.Writer()
	origStderr := os.Stderr
	log.SetOutput(io.Discard)
	os.Stderr, _ = os.Open(os.DevNull)

	return func() {
		os.Stderr = origStderr
		log.SetOutput(origLog)
	}
}

type chatMessage struct {
	Role    string
	Content string
}

func applyChatTemplate(tok *tokenizer.Tokenizer, messages []chatMessage, cfg map[string]interface{}) []int {
	hasIMStart := tok.HasToken("<|im_start|>")
	hasBeginOfText := tok.HasToken("<|begin_of_text|>")
	hasStartOfTurn := tok.HasToken("<start_of_turn>")

	if hasStartOfTurn {
		var sb strings.Builder
		for _, m := range messages {
			sb.WriteString("<start_of_turn>")
			sb.WriteString(m.Role)
			sb.WriteString("\n")
			sb.WriteString(m.Content)
			sb.WriteString("<end_of_turn>\n")
		}
		sb.WriteString("<start_of_turn>model\n")
		return tok.Encode(sb.String())
	}

	if hasIMStart {
		var sb strings.Builder
		for _, m := range messages {
			sb.WriteString("<|im_start|>")
			sb.WriteString(m.Role)
			sb.WriteString("\n")
			sb.WriteString(m.Content)
			sb.WriteString("<|im_end|>\n")
		}
		sb.WriteString("<|im_start|>assistant\n")
		return tok.Encode(sb.String())
	}

	if hasBeginOfText {
		var sb strings.Builder
		sb.WriteString("<|begin_of_text|>")
		for _, m := range messages {
			sb.WriteString("<|start_header_id|>")
			sb.WriteString(m.Role)
			sb.WriteString("<|end_header_id|>\n\n")
			sb.WriteString(m.Content)
			sb.WriteString("<|eot_id|>")
		}
		sb.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
		return tok.Encode(sb.String())
	}

	var sb strings.Builder
	for _, m := range messages {
		sb.WriteString(fmt.Sprintf("### %s:\n%s\n\n", strings.Title(m.Role), m.Content))
	}
	sb.WriteString("### Assistant:\n")
	return tok.Encode(sb.String())
}

func resolveModelName(name string) string {
	cleanName := name
	if idx := strings.LastIndex(name, ":"); idx > 0 {
		cleanName = name[:idx]
	}

	if _, err := os.Stat(cleanName); err == nil {
		return cleanName
	}

	home, _ := os.UserHomeDir()
	dirs := []string{
		filepath.Join(home, ".ai", "models"),
		filepath.Join(home, ".mongoose", "models"),
		filepath.Join(home, ".tesseract", "models"),
	}
	for _, d := range dirs {
		for _, suffix := range []string{"", "-hf", "-chat"} {
			p := filepath.Join(d, cleanName+suffix)
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}

	lower := strings.ToLower(cleanName)
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if strings.ToLower(e.Name()) == lower {
				return filepath.Join(d, e.Name())
			}
		}
	}
	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if strings.Contains(strings.ToLower(e.Name()), lower) {
				return filepath.Join(d, e.Name())
			}
		}
	}

	return ""
}

// discoverStopTokens finds all stop/EOS token IDs from the model's config files
// and tokenizer vocabulary. Works for any model — no hardcoded token strings.
func discoverStopTokens(tok *tokenizer.Tokenizer, cfg map[string]interface{}, modelDir string) map[int]bool {
	stop := map[int]bool{0: true}

	// 1. eos_token_id from config.json — can be int or []int
	if v, ok := cfg["eos_token_id"]; ok {
		switch eos := v.(type) {
		case float64:
			stop[int(eos)] = true
		case []interface{}:
			for _, id := range eos {
				if f, ok := id.(float64); ok {
					stop[int(f)] = true
				}
			}
		}
	}

	// 2. generation_config.json — may have eos_token_id as list
	if genData, err := os.ReadFile(filepath.Join(modelDir, "generation_config.json")); err == nil {
		var gen map[string]interface{}
		if json.Unmarshal(genData, &gen) == nil {
			if v, ok := gen["eos_token_id"]; ok {
				switch eos := v.(type) {
				case float64:
					stop[int(eos)] = true
				case []interface{}:
					for _, id := range eos {
						if f, ok := id.(float64); ok {
							stop[int(f)] = true
						}
					}
				}
			}
		}
	}

	// 3. tokenizer_config.json — eos_token string → resolve to ID
	if tokCfgData, err := os.ReadFile(filepath.Join(modelDir, "tokenizer_config.json")); err == nil {
		var tokCfg map[string]interface{}
		if json.Unmarshal(tokCfgData, &tokCfg) == nil {
			if eosStr, ok := tokCfg["eos_token"].(string); ok && tok.HasToken(eosStr) {
				stop[tok.Vocab[eosStr]] = true
			}
		}
	}

	// 4. Tokenizer EOS field
	if tok.EOS > 0 {
		stop[tok.EOS] = true
	}

	// 5. Scan vocab for known end-of-turn patterns — covers any model
	endPatterns := []string{
		"<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>",
		"<|end|>", "</s>", "<|end_of_turn|>", "<|stop|>",
		"<|end_header_id|>", "<end_of_turn>",
	}
	for _, pat := range endPatterns {
		if tok.HasToken(pat) {
			stop[tok.Vocab[pat]] = true
		}
	}

	// 6. Chat template implies stop tokens — ChatML uses im_end, Llama3 uses eot_id
	//    These are already covered by the vocab scan above, but add im_start too
	//    so we never generate a new turn header
	if tok.HasToken("<|im_start|>") {
		stop[tok.Vocab["<|im_start|>"]] = true
	}

	return stop
}

// chatEngine holds the loaded model and inference function.
type chatEngine struct {
	fwd        func(tokenID, pos int) []float32
	resetKV    func()
	tok        *tokenizer.Tokenizer
	cfg        map[string]interface{}
	embedData  []float32
	stopTokens map[int]bool
	maxSeq     int
	modelName  string
	engineName string
}

func buildChatEngine(modelArg string) *chatEngine {
	path := resolveModelName(modelArg)
	if path == "" {
		fmt.Printf("Model '%s' not found locally. Try: ai pull <org/model>\n", modelArg)
		os.Exit(1)
	}

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

	if dim == 0 || nLayers == 0 {
		log.Fatal("could not read model dimensions from config.json")
	}

	var eng mongoose.Engine
	if !GlobalVerbose {
		restore := suppressOutput()
		eng = selectEngine("auto")
		mongoose.LoadKernels()
		restore()
	} else {
		eng = selectEngine("auto")
		mongoose.LoadKernels()
	}

	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	ms, err := OpenModel(path)
	if err != nil {
		log.Fatalf("open model: %v", err)
	}
	st := ms.ST()
	if st == nil {
		log.Fatalf("model format not supported for chat — convert with: ai convert safetensors %s", path)
	}

	embedData, _, _ := st.ReadTensorFloat32("model.embed_tokens.weight")
	var lmHeadData []float32
	lmHeadData, _, err = st.ReadTensorFloat32("lm_head.weight")
	if err != nil {
		lmHeadData = embedData
	}

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

	var fwd func(tokenID, pos int) []float32
	var resetKV func()

	// Metal fused compute shaders
	if metal, ok := eng.(*mongoose.Metal); ok {
		ret := metal.BuildFused(dim, kvDim, headDim, heads, kvHeads, ffnDim, vocabSize, nLayers, maxSeq, float64(ropeTheta), 1e-6)
		if ret == 0 {
			wi := 0
			for l := 0; l < nLayers; l++ {
				prefix := fmt.Sprintf("model.layers.%d.", l)
				loadW := func(n string) {
					d, _, _ := st.ReadTensorFloat32(prefix + n)
					if d != nil {
						metal.FusedSetWeight(wi, d)
					}
					wi++
				}
				loadB := func(n string, sz int) {
					d, _, _ := st.ReadTensorFloat32(prefix + n)
					if d == nil {
						d = make([]float32, sz)
					}
					metal.FusedSetWeight(wi, d)
					wi++
				}
				loadW("input_layernorm.weight")
				loadW("self_attn.q_proj.weight"); loadW("self_attn.k_proj.weight"); loadW("self_attn.v_proj.weight")
				loadB("self_attn.q_proj.bias", dim); loadB("self_attn.k_proj.bias", kvDim); loadB("self_attn.v_proj.bias", kvDim)
				loadW("self_attn.o_proj.weight"); loadW("post_attention_layernorm.weight")
				loadW("mlp.gate_proj.weight"); loadW("mlp.up_proj.weight"); loadW("mlp.down_proj.weight")
			}
			fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
			metal.FusedSetWeight(wi, fnorm); wi++
			metal.FusedSetWeight(wi, lmHeadData); wi++

			fHidden := make([]float32, dim)
			fLogits := make([]float32, vocabSize)
			fwd = func(tokenID, pos int) []float32 {
				tokOff := tokenID * dim
				if tokOff+dim > len(embedData) { return nil }
				copy(fHidden, embedData[tokOff:tokOff+dim])
				metal.FusedStep(fHidden, cosTab[pos*halfHead:pos*halfHead+halfHead],
					sinTab[pos*halfHead:pos*halfHead+halfHead], pos, fLogits)
				return fLogits
			}
			resetKV = func() { metal.FusedInferReset() }
		}
	}

	// Metal inference graph fallback for Metal 3 (2 dispatches/layer, CPU attention)
	if fwd == nil {
		if metal, ok := eng.(*mongoose.Metal); ok {
			ret := metal.BuildInferGraph(dim, kvDim, headDim, heads, kvHeads, ffnDim, vocabSize, nLayers, float64(ropeTheta))
			if ret == 0 {
				wi := 0
				for l := 0; l < nLayers; l++ {
					prefix := fmt.Sprintf("model.layers.%d.", l)
					loadW := func(n string) {
						d, _, _ := st.ReadTensorFloat32(prefix + n)
						if d != nil { metal.InferSetWeight(wi, d) }
						wi++
					}
					loadB := func(n string, sz int) {
						d, _, _ := st.ReadTensorFloat32(prefix + n)
						if d == nil { d = make([]float32, sz) }
						metal.InferSetWeight(wi, d)
						wi++
					}
					loadW("input_layernorm.weight")
					loadW("self_attn.q_proj.weight"); loadW("self_attn.k_proj.weight"); loadW("self_attn.v_proj.weight")
					loadB("self_attn.q_proj.bias", dim); loadB("self_attn.k_proj.bias", kvDim); loadB("self_attn.v_proj.bias", kvDim)
					loadW("self_attn.o_proj.weight"); loadW("post_attention_layernorm.weight")
					loadW("mlp.gate_proj.weight"); loadW("mlp.up_proj.weight"); loadW("mlp.down_proj.weight")
				}
				fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
				metal.InferSetWeight(wi, fnorm); wi++
				metal.InferSetWeight(wi, lmHeadData); wi++

				keyCache := make([][]float32, nLayers)
				valCache := make([][]float32, nLayers)
				for l := 0; l < nLayers; l++ {
					keyCache[l] = make([]float32, maxSeq*kvDim)
					valCache[l] = make([]float32, maxSeq*kvDim)
				}
				att := make([]float32, heads*maxSeq)
				mHidden := make([]float32, dim)
				mQBuf := make([]float32, dim)
				mKBuf := make([]float32, kvDim)
				mVBuf := make([]float32, kvDim)
				mAttnOut := make([]float32, dim)
				mLogits := make([]float32, vocabSize)
				invSqrtHeadDim := float32(1.0 / math.Sqrt(float64(headDim)))
				kvMulConst := heads / kvHeads

				resetKV = func() {
					for l := 0; l < nLayers; l++ {
						for i := range keyCache[l] { keyCache[l][i] = 0 }
						for i := range valCache[l] { valCache[l][i] = 0 }
					}
				}

				fwd = func(tokenID, pos int) []float32 {
					tokOff := tokenID * dim
					if tokOff+dim > len(embedData) { return nil }
					copy(mHidden, embedData[tokOff:tokOff+dim])
					cosSlice := cosTab[pos*halfHead : pos*halfHead+halfHead]
					sinSlice := sinTab[pos*halfHead : pos*halfHead+halfHead]

					for l := 0; l < nLayers; l++ {
						metal.InferForwardA(mHidden, cosSlice, sinSlice, mQBuf, mKBuf, mVBuf, l)
						copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], mKBuf)
						copy(valCache[l][pos*kvDim:(pos+1)*kvDim], mVBuf)

						for i := range mAttnOut { mAttnOut[i] = 0 }
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

	// Generic fallback: works on any engine (CUDA, WebGPU, CPU)
	// Weights in VRAM (if TensorEngine) or streamed from disk, CPU attention
	if fwd == nil {
		normEps := float32(1e-6)
		if v, ok := cfg["rms_norm_eps"].(float64); ok {
			normEps = float32(v)
		}

		te := mongoose.AsTensorEngine(eng)
		rwe := mongoose.AsResidentWeightEngine(eng)

		keyCache := make([][]float32, nLayers)
		valCache := make([][]float32, nLayers)
		for l := 0; l < nLayers; l++ {
			keyCache[l] = make([]float32, maxSeq*kvDim)
			valCache[l] = make([]float32, maxSeq*kvDim)
		}
		att := make([]float32, heads*maxSeq)

		type gpuLayer struct {
			wq, wk, wv, wo, gate, up, down *mongoose.Tensor
			norm1, norm2                    []float32
		}

		gpuResident := te != nil
		var gpuLayers []gpuLayer
		var gpuLMHead *mongoose.Tensor
		var gpuFinalNorm []float32

		if gpuResident {
			gpuLayers = make([]gpuLayer, nLayers)
			for l := 0; l < nLayers; l++ {
				prefix := fmt.Sprintf("model.layers.%d.", l)
				loadGPU := func(name string, rows, cols int) *mongoose.Tensor {
					d, _, err := st.ReadTensorFloat32(prefix + name)
					if err != nil { return nil }
					return te.FromHost(d, []int{rows, cols})
				}
				loadCPU := func(name string) []float32 {
					d, _, _ := st.ReadTensorFloat32(prefix + name)
					return d
				}
				gpuLayers[l] = gpuLayer{
					wq: loadGPU("self_attn.q_proj.weight", dim, dim),
					wk: loadGPU("self_attn.k_proj.weight", kvDim, dim),
					wv: loadGPU("self_attn.v_proj.weight", kvDim, dim),
					wo: loadGPU("self_attn.o_proj.weight", dim, dim),
					gate: loadGPU("mlp.gate_proj.weight", ffnDim, dim),
					up:   loadGPU("mlp.up_proj.weight", ffnDim, dim),
					down: loadGPU("mlp.down_proj.weight", dim, ffnDim),
					norm1: loadCPU("input_layernorm.weight"),
					norm2: loadCPU("post_attention_layernorm.weight"),
				}
			}
			gpuLMHead = te.FromHost(lmHeadData, []int{vocabSize, dim})
			gpuFinalNorm, _, _ = st.ReadTensorFloat32("model.norm.weight")
		}

		rmsNorm := func(data, weight []float32, eps float32) {
			n := len(data)
			var ss float32
			for i := 0; i < n; i++ { ss += data[i] * data[i] }
			ss = 1.0 / float32(math.Sqrt(float64(ss/float32(n)+eps)))
			for i := 0; i < n; i++ { data[i] = data[i] * ss * weight[i] }
		}

		gpuMatVec := func(out []float32, tW *mongoose.Tensor, xIn []float32, rows, cols int) {
			if rwe != nil {
				copy(out, rwe.MatVecResidentW(tW, xIn, rows, cols))
				return
			}
			tX := te.FromHost(xIn, []int{1, cols})
			tY := te.MatMulT(tW, tX, rows, cols, 1)
			copy(out, te.ToHost(tY))
			te.Release(tX)
			te.Release(tY)
		}

		streamMatVec := func(out []float32, W, xIn []float32, rows, cols int) {
			copy(out, eng.MatMul(W, xIn, rows, cols, 1))
		}

		x := make([]float32, dim)
		buf := make([]float32, dim)
		q := make([]float32, dim)
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
		attnOut := make([]float32, dim)
		ffnBuf := make([]float32, ffnDim)
		ffnBuf2 := make([]float32, ffnDim)

		resetKV = func() {
			for l := 0; l < nLayers; l++ {
				for i := range keyCache[l] { keyCache[l][i] = 0 }
				for i := range valCache[l] { valCache[l][i] = 0 }
			}
		}

		fwd = func(tokenID, pos int) []float32 {
			tokOff := tokenID * dim
			if tokOff+dim > len(embedData) { return nil }
			copy(x, embedData[tokOff:tokOff+dim])

			for l := 0; l < nLayers; l++ {
				copy(buf, x)
				if gpuResident {
					rmsNorm(buf, gpuLayers[l].norm1, normEps)
					gpuMatVec(q, gpuLayers[l].wq, buf, dim, dim)
					gpuMatVec(k[:kvDim], gpuLayers[l].wk, buf, kvDim, dim)
					gpuMatVec(v[:kvDim], gpuLayers[l].wv, buf, kvDim, dim)
				} else {
					prefix := fmt.Sprintf("model.layers.%d.", l)
					normW, _, _ := st.ReadTensorFloat32(prefix + "input_layernorm.weight")
					rmsNorm(buf, normW, normEps)
					wq, _, _ := st.ReadTensorFloat32(prefix + "self_attn.q_proj.weight")
					wk, _, _ := st.ReadTensorFloat32(prefix + "self_attn.k_proj.weight")
					wv, _, _ := st.ReadTensorFloat32(prefix + "self_attn.v_proj.weight")
					streamMatVec(q, wq, buf, dim, dim)
					streamMatVec(k[:kvDim], wk, buf, kvDim, dim)
					streamMatVec(v[:kvDim], wv, buf, kvDim, dim)
				}

				// Biases
				for _, pair := range []struct{ name string; dst []float32; sz int }{
					{"self_attn.q_proj.bias", q, dim},
					{"self_attn.k_proj.bias", k[:kvDim], kvDim},
					{"self_attn.v_proj.bias", v[:kvDim], kvDim},
				} {
					prefix := fmt.Sprintf("model.layers.%d.", l)
					if b, _, e := st.ReadTensorFloat32(prefix + pair.name); e == nil {
						for i := range b { pair.dst[i] += b[i] }
					}
				}

				applyRoPE(q, k[:kvDim], pos, headDim, ropeTheta, heads, kvHeads)

				copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], k[:kvDim])
				copy(valCache[l][pos*kvDim:(pos+1)*kvDim], v[:kvDim])

				kvMul := heads / kvHeads
				for i := range attnOut { attnOut[i] = 0 }
				for h := 0; h < heads; h++ {
					qOff := h * headDim
					kvOff := (h / kvMul) * headDim
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
				if gpuResident {
					gpuMatVec(proj, gpuLayers[l].wo, attnOut, dim, dim)
				} else {
					prefix := fmt.Sprintf("model.layers.%d.", l)
					wo, _, _ := st.ReadTensorFloat32(prefix + "self_attn.o_proj.weight")
					streamMatVec(proj, wo, attnOut, dim, dim)
				}
				for i := 0; i < dim; i++ { x[i] += proj[i] }

				copy(buf, x)
				if gpuResident {
					rmsNorm(buf, gpuLayers[l].norm2, normEps)
					gpuMatVec(ffnBuf, gpuLayers[l].gate, buf, ffnDim, dim)
					gpuMatVec(ffnBuf2, gpuLayers[l].up, buf, ffnDim, dim)
				} else {
					prefix := fmt.Sprintf("model.layers.%d.", l)
					normW2, _, _ := st.ReadTensorFloat32(prefix + "post_attention_layernorm.weight")
					rmsNorm(buf, normW2, normEps)
					gate, _, _ := st.ReadTensorFloat32(prefix + "mlp.gate_proj.weight")
					up, _, _ := st.ReadTensorFloat32(prefix + "mlp.up_proj.weight")
					streamMatVec(ffnBuf, gate, buf, ffnDim, dim)
					streamMatVec(ffnBuf2, up, buf, ffnDim, dim)
				}
				for i := 0; i < ffnDim; i++ {
					ffnBuf[i] = silu(ffnBuf[i]) * ffnBuf2[i]
				}
				downOut := make([]float32, dim)
				if gpuResident {
					gpuMatVec(downOut, gpuLayers[l].down, ffnBuf, dim, ffnDim)
				} else {
					prefix := fmt.Sprintf("model.layers.%d.", l)
					down, _, _ := st.ReadTensorFloat32(prefix + "mlp.down_proj.weight")
					streamMatVec(downOut, down, ffnBuf, dim, ffnDim)
				}
				for i := 0; i < dim; i++ { x[i] += downOut[i] }
			}

			if gpuResident {
				rmsNorm(x, gpuFinalNorm, normEps)
				logits := make([]float32, vocabSize)
				gpuMatVec(logits, gpuLMHead, x, vocabSize, dim)
				return logits
			}
			finalNorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
			rmsNorm(x, finalNorm, normEps)
			logits := make([]float32, vocabSize)
			streamMatVec(logits, lmHeadData, x, vocabSize, dim)
			return logits
		}
	}
	if resetKV == nil {
		resetKV = func() {}
	}

	stopTokens := discoverStopTokens(tok, cfg, path)

	return &chatEngine{
		fwd:        fwd,
		resetKV:    resetKV,
		tok:        tok,
		cfg:        cfg,
		embedData:  embedData,
		stopTokens: stopTokens,
		maxSeq:     maxSeq,
		modelName:  filepath.Base(path),
		engineName: eng.Name(),
	}
}

// generate runs inference and sends the result on doneCh.
func (ce *chatEngine) generate(history []chatMessage, doneCh chan<- chatGenDone) {
	ce.resetKV()
	tokens := applyChatTemplate(ce.tok, history, ce.cfg)
	if len(tokens) >= ce.maxSeq-10 {
		doneCh <- chatGenDone{response: "[context full]"}
		return
	}

	var logits []float32
	for i, tid := range tokens {
		logits = ce.fwd(tid, i)
	}
	if logits == nil {
		doneCh <- chatGenDone{response: "[inference error]"}
		return
	}

	temp := float32(0.7)
	topK := 40
	maxTokens := 512
	allTokens := make([]int, len(tokens))
	copy(allTokens, tokens)
	nGen := 0
	t0 := time.Now()

	var response strings.Builder

	for step := 0; step < maxTokens; step++ {
		nextToken := sampleTopK(logits, temp, topK)
		allTokens = append(allTokens, nextToken)
		nGen++

		if ce.stopTokens[nextToken] {
			break
		}

		text := ce.tok.Decode([]int{nextToken})
		response.WriteString(text)

		full := response.String()
		if idx := findSpecialToken(full); idx >= 0 {
			response.Reset()
			response.WriteString(full[:idx])
			break
		}

		pos := len(allTokens) - 1
		if pos >= ce.maxSeq-1 {
			break
		}
		logits = ce.fwd(allTokens[pos], pos)
		if logits == nil {
			break
		}
	}

	final := response.String()
	if idx := findSpecialToken(final); idx >= 0 {
		final = final[:idx]
	}

	elapsed := time.Since(t0)
	tokPerSec := float64(nGen) / elapsed.Seconds()
	doneCh <- chatGenDone{
		nTokens:   nGen,
		tokPerSec: tokPerSec,
		response:  strings.TrimSpace(final),
	}
}

// findSpecialToken returns the index of the first <|...|> pattern in s, or -1.
func findSpecialToken(s string) int {
	i := 0
	for i < len(s) {
		start := strings.Index(s[i:], "<|")
		if start < 0 {
			return -1
		}
		start += i
		end := strings.Index(s[start:], "|>")
		if end < 0 {
			return -1
		}
		return start
	}
	return -1
}

// --- Bubbletea TUI ---

type chatGenDone struct {
	nTokens   int
	tokPerSec float64
	response  string
}

type doneMsg chatGenDone

var (
	headerStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("12"))
	userStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("10"))
	assistStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("15"))
	dimStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	promptStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("14"))
	statusStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("11"))
)

type chatModel struct {
	engine     *chatEngine
	history    []chatMessage
	input      string
	generating bool
	stats      string
	messages   []renderedMsg
	doneCh     chan chatGenDone
	width      int
	height     int
	quitting   bool
}

type renderedMsg struct {
	role    string
	content string
}

func newChatModel(engine *chatEngine) chatModel {
	return chatModel{
		engine:  engine,
		history: []chatMessage{{Role: "system", Content: "You are a helpful assistant."}},
		doneCh:  make(chan chatGenDone, 1),
		width:   80,
		height:  24,
	}
}

func (m chatModel) Init() tea.Cmd {
	return tea.Batch(tea.WindowSize(), tea.SetWindowTitle("ai chat — "+m.engine.modelName))
}

func waitForDone(ch <-chan chatGenDone) tea.Cmd {
	return func() tea.Msg {
		d := <-ch
		return doneMsg(d)
	}
}

func (m chatModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		return m, nil

	case tea.KeyMsg:
		if m.generating {
			if msg.String() == "ctrl+c" {
				m.quitting = true
				return m, tea.Quit
			}
			return m, nil
		}

		switch msg.String() {
		case "ctrl+c", "ctrl+d":
			m.quitting = true
			return m, tea.Quit
		case "enter":
			input := strings.TrimSpace(m.input)
			if input == "" {
				return m, nil
			}
			if input == "/clear" {
				m.history = m.history[:1]
				m.messages = nil
				m.input = ""
				m.stats = ""
				return m, nil
			}

			m.messages = append(m.messages, renderedMsg{role: "user", content: input})
			m.history = append(m.history, chatMessage{Role: "user", Content: input})
			m.input = ""
			m.generating = true
			m.stats = ""

			m.doneCh = make(chan chatGenDone, 1)
			go m.engine.generate(m.history, m.doneCh)
			return m, waitForDone(m.doneCh)

		case "backspace":
			if len(m.input) > 0 {
				m.input = m.input[:len(m.input)-1]
			}
		default:
			if len(msg.String()) == 1 || msg.String() == " " {
				m.input += msg.String()
			}
		}
		return m, nil

	case doneMsg:
		if msg.response != "" {
			m.messages = append(m.messages, renderedMsg{role: "assistant", content: msg.response})
			m.history = append(m.history, chatMessage{Role: "assistant", Content: msg.response})
		}
		m.generating = false
		if msg.nTokens > 0 {
			m.stats = fmt.Sprintf("%d tokens  %.1f tok/s", msg.nTokens, msg.tokPerSec)
		}
		return m, nil
	}

	return m, nil
}

func (m chatModel) View() string {
	if m.quitting {
		return ""
	}

	var b strings.Builder

	header := headerStyle.Render(fmt.Sprintf(" ai chat  %s  %s ", m.engine.modelName, m.engine.engineName))
	b.WriteString(header)
	b.WriteString("\n\n")

	for _, msg := range m.messages {
		switch msg.role {
		case "user":
			b.WriteString(userStyle.Render("You: "))
			b.WriteString(msg.content)
			b.WriteString("\n\n")
		case "assistant":
			b.WriteString(assistStyle.Render(msg.content))
			b.WriteString("\n\n")
		}
	}

	if m.generating {
		b.WriteString(dimStyle.Render("thinking..."))
		b.WriteString("\n\n")
	}

	if m.stats != "" {
		b.WriteString(dimStyle.Render(m.stats))
		b.WriteString("\n\n")
	}

	if !m.generating {
		b.WriteString(promptStyle.Render("> "))
		b.WriteString(m.input)
		b.WriteString(dimStyle.Render("_"))
	}

	b.WriteString("\n")
	b.WriteString(dimStyle.Render("  ctrl+c quit  /clear reset"))

	return b.String()
}

func cmdChat(modelArg string) {
	debug := GlobalVerbose

	ce := buildChatEngine(modelArg)

	if debug {
		cmdChatDebug(ce)
		return
	}

	p := tea.NewProgram(newChatModel(ce), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		log.Fatal(err)
	}
}

// cmdChatDebug is the raw debug path — shows every token including specials.
func cmdChatDebug(ce *chatEngine) {
	fmt.Printf("ai chat [debug] — %s (%s)\n", ce.modelName, ce.engineName)
	fmt.Printf("Type your message. Press Ctrl+D to exit.\n\n")

	history := []chatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
	}

	scanner := make([]byte, 0, 4096)
	buf := make([]byte, 1)
	for {
		fmt.Print("> ")
		scanner = scanner[:0]
		for {
			n, err := os.Stdin.Read(buf)
			if err != nil || n == 0 {
				fmt.Println()
				return
			}
			if buf[0] == '\n' {
				break
			}
			scanner = append(scanner, buf[0])
		}
		input := strings.TrimSpace(string(scanner))
		if input == "" {
			continue
		}
		if input == "/exit" || input == "/quit" {
			break
		}

		history = append(history, chatMessage{Role: "user", Content: input})

		doneCh := make(chan chatGenDone, 1)
		go ce.generate(history, doneCh)

		d := <-doneCh
		fmt.Print(d.response)
		fmt.Printf("\n\n[%d tokens, %.1f tok/s]\n\n", d.nTokens, d.tokPerSec)
		history = append(history, chatMessage{Role: "assistant", Content: d.response})
	}
}
