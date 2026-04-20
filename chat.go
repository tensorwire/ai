package main

import (
	"bufio"
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

type chatMessage struct {
	Role    string
	Content string
}

func applyChatTemplate(tok *tokenizer.Tokenizer, messages []chatMessage, cfg map[string]interface{}) []int {
	// Try to detect the chat format from tokenizer_config.json chat_template
	// Support: ChatML (Qwen, Yi), Llama3, Mistral, generic fallback

	// Check for common special tokens
	hasIMStart := tok.HasToken("<|im_start|>")
	hasBeginOfText := tok.HasToken("<|begin_of_text|>")

	if hasIMStart {
		// ChatML format (Qwen, Yi, etc.)
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
		// Llama 3 format
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

	// Generic fallback: just concatenate with role markers
	var sb strings.Builder
	for _, m := range messages {
		sb.WriteString(fmt.Sprintf("### %s:\n%s\n\n", strings.Title(m.Role), m.Content))
	}
	sb.WriteString("### Assistant:\n")
	return tok.Encode(sb.String())
}

func resolveModelName(name string) string {
	// Support Ollama-style names: llama3:latest, qwen2.5:0.5b, org/model
	// Strip :tag if present
	cleanName := name
	if idx := strings.LastIndex(name, ":"); idx > 0 {
		cleanName = name[:idx]
	}

	// Try resolveModel first (handles existing paths)
	if p := resolveModel(cleanName); p != "" {
		return p
	}
	if p := resolveModel(name); p != "" {
		return p
	}

	// Try common HuggingFace naming patterns
	home, _ := os.UserHomeDir()
	modelsDir := filepath.Join(home, ".ai", "models")

	// llama3 -> TinyLlama-*, Llama-3-*
	// qwen2.5:0.5b -> Qwen2.5-0.5B
	entries, err := os.ReadDir(modelsDir)
	if err != nil {
		return ""
	}

	lower := strings.ToLower(cleanName)
	for _, e := range entries {
		if strings.ToLower(e.Name()) == lower {
			return filepath.Join(modelsDir, e.Name())
		}
		if strings.Contains(strings.ToLower(e.Name()), lower) {
			return filepath.Join(modelsDir, e.Name())
		}
	}

	return ""
}

func cmdChat(modelArg string) {
	path := resolveModelName(modelArg)
	if path == "" {
		// Try pulling it
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

	eng := selectEngine("auto")
	mongoose.LoadKernels()

	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	st, err := gguf.OpenSafeTensors(path)
	if err != nil {
		log.Fatalf("open model: %v", err)
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

	// Set up forward pass
	var fwd func(tokenID, pos int) []float32

	if metal, ok := eng.(*mongoose.Metal); ok {
		ret := metal.BuildFused(dim, kvDim, headDim, heads, kvHeads, ffnDim, vocabSize, nLayers, maxSeq)
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
				loadW("self_attn.q_proj.weight")
				loadW("self_attn.k_proj.weight")
				loadW("self_attn.v_proj.weight")
				loadB("self_attn.q_proj.bias", dim)
				loadB("self_attn.k_proj.bias", kvDim)
				loadB("self_attn.v_proj.bias", kvDim)
				loadW("self_attn.o_proj.weight")
				loadW("post_attention_layernorm.weight")
				loadW("mlp.gate_proj.weight")
				loadW("mlp.up_proj.weight")
				loadW("mlp.down_proj.weight")
			}
			fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
			metal.FusedSetWeight(wi, fnorm)
			wi++
			metal.FusedSetWeight(wi, lmHeadData)
			wi++

			fHidden := make([]float32, dim)
			fLogits := make([]float32, vocabSize)
			fwd = func(tokenID, pos int) []float32 {
				tokOff := tokenID * dim
				if tokOff+dim > len(embedData) {
					return nil
				}
				copy(fHidden, embedData[tokOff:tokOff+dim])
				metal.FusedStep(fHidden, cosTab[pos*halfHead:pos*halfHead+halfHead],
					sinTab[pos*halfHead:pos*halfHead+halfHead], pos, fLogits)
				return fLogits
			}
		}
	}

	// CUDA Q8 path
	if fwd == nil {
		if _, ok := eng.(*mongoose.CUDA); ok && mongoose.HasQ8Matvec() && mongoose.KernelsLoaded() {
			te := mongoose.AsTensorEngine(eng)
			if te != nil {
				// Reuse the same Q8 quantize-on-load logic from infer_gpu.go
				// For now, fall through to tier 2
			}
		}
	}

	// Tier 2: GPU-resident with TensorEngine
	if fwd == nil {
		te := mongoose.AsTensorEngine(eng)
		if te != nil {
			// Use cmdInferGPU's tier 2 path — too much code to duplicate.
			// Fall through to CPU for chat.
		}
	}

	// CPU fallback using infer.go's cmdInferImpl approach
	if fwd == nil {
		log.Printf("[chat] no GPU forward pass available — GPU inference required for chat")
		os.Exit(1)
	}

	// Chat REPL
	modelBase := filepath.Base(path)

	fmt.Printf("ai chat — %s (%s)\n", modelBase, eng.Name())
	fmt.Printf("Type your message. Press Ctrl+D to exit.\n\n")

	// Resolve stop token IDs once
	stopTokens := map[int]bool{0: true}
	if tok.EOS > 0 {
		stopTokens[tok.EOS] = true
	}
	for _, s := range []string{"<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>"} {
		if tok.HasToken(s) {
			stopTokens[tok.Vocab[s]] = true
		}
		ids := tok.Encode(s)
		if len(ids) == 1 {
			stopTokens[ids[0]] = true
		}
	}

	history := []chatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
	}

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			fmt.Println()
			break
		}
		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}
		if input == "/exit" || input == "/quit" {
			break
		}
		if input == "/clear" {
			history = history[:1] // keep system message
			fmt.Println("(conversation cleared)")
			// TODO: reset KV cache
			continue
		}

		history = append(history, chatMessage{Role: "user", Content: input})

		tokens := applyChatTemplate(tok, history, cfg)
		if len(tokens) >= maxSeq-10 {
			fmt.Println("(context full — clearing history)")
			history = history[:1]
			history = append(history, chatMessage{Role: "user", Content: input})
			tokens = applyChatTemplate(tok, history, cfg)
		}

		// Prefill
		var logits []float32
		for i, tid := range tokens {
			logits = fwd(tid, i)
		}
		if logits == nil {
			fmt.Println("[error: forward pass returned nil]")
			continue
		}

		// Generate
		temp := float32(0.7)
		topK := 40
		maxTokens := 512
		allTokens := make([]int, len(tokens))
		copy(allTokens, tokens)
		var genTokens []int
		var response strings.Builder

		lastPrint := 0
		t0 := time.Now()
		for step := 0; step < maxTokens; step++ {
			nextToken := sampleTopK(logits, temp, topK)
			allTokens = append(allTokens, nextToken)
			genTokens = append(genTokens, nextToken)

			if stopTokens[nextToken] {
				break
			}

			if nextToken >= 151643 {
				break
			}
			text := tok.Decode([]int{nextToken})
			response.WriteString(text)
			// Check accumulated response for spelled-out special tokens
			r := response.String()
			if idx := strings.Index(r, "<|im_end|>"); idx >= 0 {
				fmt.Print(r[lastPrint:idx])
				break
			}
			if idx := strings.Index(r, "<|endoftext|>"); idx >= 0 {
				fmt.Print(r[lastPrint:idx])
				break
			}
			if idx := strings.Index(r, "<|eot_id|>"); idx >= 0 {
				fmt.Print(r[lastPrint:idx])
				break
			}
			// Print up to a safe point (hold back last 15 chars in case a special token is forming)
			safeLen := len(r) - 15
			if safeLen > lastPrint {
				fmt.Print(r[lastPrint:safeLen])
				lastPrint = safeLen
			}
			response.WriteString(text)

			pos := len(allTokens) - 1
			if pos >= maxSeq-1 {
				break
			}
			logits = fwd(allTokens[pos], pos)
			if logits == nil {
				break
			}
		}

		// Flush any remaining buffered text
		r := response.String()
		if lastPrint < len(r) {
			// Don't print past any special token
			remaining := r[lastPrint:]
			if idx := strings.Index(remaining, "<|"); idx >= 0 {
				remaining = remaining[:idx]
			}
			fmt.Print(remaining)
		}

		elapsed := time.Since(t0)
		nGen := len(genTokens)
		fmt.Printf("\n\n[%d tokens, %.1f tok/s]\n\n", nGen, float64(nGen)/elapsed.Seconds())

		// Strip special tokens from history
		cleanResponse := r
		for _, sp := range []string{"<|im_end|>", "<|endoftext|>", "<|eot_id|>"} {
			if idx := strings.Index(cleanResponse, sp); idx >= 0 {
				cleanResponse = cleanResponse[:idx]
			}
		}
		history = append(history, chatMessage{Role: "assistant", Content: strings.TrimSpace(cleanResponse)})
	}
}
