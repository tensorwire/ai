// serve.go — OpenAI + Ollama compatible API server for mongoose.
//
// Endpoints (OpenAI):
//   POST /v1/chat/completions   — streaming (SSE) + non-streaming
//   POST /v1/completions        — legacy completions
//   POST /v1/embeddings         — text embeddings (stub)
//   GET  /v1/models             — list loaded models
//
// Endpoints (Ollama-native):
//   POST /api/chat              — streaming chat (NDJSON)
//   POST /api/generate          — streaming generate (NDJSON)
//   POST /api/show              — model info
//   GET  /api/tags              — list models
//
// Common:
//   GET  /health                — liveness probe
//
// Usage:
//   ai serve Qwen2.5-0.5B
//   ai serve Qwen2.5-0.5B port=8080
//   ai serve Qwen2.5-0.5B --daemon

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/mongoose"
	"github.com/open-ai-org/tokenizer"
)

// -----------------------------------------------------------------------
// OpenAI API types
// -----------------------------------------------------------------------

type ChatCompletionRequest struct {
	Model            string          `json:"model"`
	Messages         []ChatMessage   `json:"messages"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	N                int             `json:"n,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	Stop             interface{}     `json:"stop,omitempty"`
	MaxTokens        *int            `json:"max_tokens,omitempty"`
	PresencePenalty  float64         `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64         `json:"frequency_penalty,omitempty"`
	User             string          `json:"user,omitempty"`
	Seed             *int            `json:"seed,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ResponseFormat struct {
	Type string `json:"type"`
}

type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   UsageInfo    `json:"usage"`
}

type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type ChatCompletionChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []ChatChunkChoice `json:"choices"`
}

type ChatChunkChoice struct {
	Index        int     `json:"index"`
	Delta        ChatDelta `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

type ChatDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type CompletionRequest struct {
	Model       string      `json:"model"`
	Prompt      string      `json:"prompt"`
	MaxTokens   *int        `json:"max_tokens,omitempty"`
	Temperature *float64    `json:"temperature,omitempty"`
	TopP        *float64    `json:"top_p,omitempty"`
	N           int         `json:"n,omitempty"`
	Stream      bool        `json:"stream,omitempty"`
	Stop        interface{} `json:"stop,omitempty"`
	Suffix      string      `json:"suffix,omitempty"`
}

type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   UsageInfo          `json:"usage"`
}

type CompletionChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

type EmbeddingRequest struct {
	Model          string      `json:"model"`
	Input          interface{} `json:"input"`
	EncodingFormat string      `json:"encoding_format"`
}

type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  UsageInfo       `json:"usage"`
}

type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens"`
}

type ModelInfo struct {
	ID       string `json:"id"`
	Object   string `json:"object"`
	Created  int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

type ErrorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// -----------------------------------------------------------------------
// Ollama API types
// -----------------------------------------------------------------------

type OllamaChatRequest struct {
	Model    string             `json:"model"`
	Messages []OllamaChatMsg    `json:"messages"`
	Stream   *bool              `json:"stream,omitempty"`
	Options  *OllamaOptions     `json:"options,omitempty"`
}

type OllamaChatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type OllamaOptions struct {
	Temperature float64 `json:"temperature,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	TopP        float64 `json:"top_p,omitempty"`
	NumPredict  int     `json:"num_predict,omitempty"`
	Seed        int     `json:"seed,omitempty"`
}

type OllamaChatResponse struct {
	Model           string      `json:"model"`
	CreatedAt       string      `json:"created_at"`
	Message         OllamaChatMsg `json:"message"`
	Done            bool        `json:"done"`
	TotalDuration   int64       `json:"total_duration,omitempty"`
	LoadDuration    int64       `json:"load_duration,omitempty"`
	PromptEvalCount int         `json:"prompt_eval_count,omitempty"`
	PromptEvalDur   int64       `json:"prompt_eval_duration,omitempty"`
	EvalCount       int         `json:"eval_count,omitempty"`
	EvalDuration    int64       `json:"eval_duration,omitempty"`
}

type OllamaGenerateRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	Stream  *bool          `json:"stream,omitempty"`
	Options *OllamaOptions `json:"options,omitempty"`
}

type OllamaGenerateResponse struct {
	Model           string `json:"model"`
	CreatedAt       string `json:"created_at"`
	Response        string `json:"response"`
	Done            bool   `json:"done"`
	TotalDuration   int64  `json:"total_duration,omitempty"`
	LoadDuration    int64  `json:"load_duration,omitempty"`
	PromptEvalCount int    `json:"prompt_eval_count,omitempty"`
	PromptEvalDur   int64  `json:"prompt_eval_duration,omitempty"`
	EvalCount       int    `json:"eval_count,omitempty"`
	EvalDuration    int64  `json:"eval_duration,omitempty"`
}

type OllamaShowResponse struct {
	Modelfile  string            `json:"modelfile"`
	Parameters string            `json:"parameters"`
	Template   string            `json:"template"`
	Details    OllamaModelDetail `json:"details"`
}

type OllamaModelDetail struct {
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

// -----------------------------------------------------------------------
// Inference request — internal channel-based queue
// -----------------------------------------------------------------------

type inferRequest struct {
	tokens    []int
	maxTokens int
	temp      float32
	topK      int
	stopToks  map[int]bool
	resultCh  chan inferToken
}

type inferToken struct {
	tokenID int
	done    bool
	err     error
}

// -----------------------------------------------------------------------
// Server state
// -----------------------------------------------------------------------

type serveState struct {
	mu sync.RWMutex

	modelName string
	modelDir  string
	vocabSize int
	dim       int
	layers    int
	heads     int
	kvHeads   int
	ffnDim    int
	maxSeq    int

	tokenizer *tokenizer.Tokenizer
	safetens  *gguf.SafeTensors
	cfg       map[string]interface{}
	eng       mongoose.Engine

	fwd     func(tokenID, pos int) []float32
	resetKV func()

	embedData []float32
	cosTab    []float32
	sinTab    []float32
	halfHead  int

	stopTokens map[int]bool

	noStream bool // --no-stream: disable streaming weight load, wait for full model in VRAM

	// Channel-based inference queue: serialize GPU access without mutex contention
	inferQueue chan *inferRequest
}

// -----------------------------------------------------------------------
// Server entrypoint
// -----------------------------------------------------------------------

func cmdServe(args map[string]string) {
	host := "0.0.0.0"
	port := "11434"
	modelName := ""

	if v, ok := args["model"]; ok {
		modelName = v
	} else if v, ok := args["_0"]; ok && !strings.HasPrefix(v, "--") {
		modelName = v
	}
	if v, ok := args["host"]; ok {
		host = v
	}
	if v, ok := args["port"]; ok {
		port = v
	}

	daemon := false
	noStream := false

	for i := 2; i < len(os.Args); i++ {
		switch os.Args[i] {
		case "--host":
			if i+1 < len(os.Args) { host = os.Args[i+1]; i++ }
		case "--port":
			if i+1 < len(os.Args) { port = os.Args[i+1]; i++ }
		case "--model":
			if i+1 < len(os.Args) { modelName = os.Args[i+1]; i++ }
		case "--daemon", "-d":
			daemon = true
		case "--no-stream":
			noStream = true
		case "--stop":
			stopExistingDaemon()
			fmt.Println("ai serve stopped")
			return
		}
	}
	if v, ok := args["daemon"]; ok && v == "true" {
		daemon = true
	}
	if _, ok := args["stop"]; ok {
		stopExistingDaemon()
		fmt.Println("ai serve stopped")
		return
	}

	if daemon {
		daemonize(host, port, modelName)
		return
	}

	state := &serveState{noStream: noStream}

	if modelName != "" {
		if err := state.loadModel(modelName); err != nil {
			log.Fatalf("Failed to load model %q: %v", modelName, err)
		}
		log.Printf("[serve] model loaded: %s (dim=%d, layers=%d, heads=%d, vocab=%d)",
			state.modelName, state.dim, state.layers, state.heads, state.vocabSize)
	}

	// Start inference worker goroutine
	state.inferQueue = make(chan *inferRequest, 16)
	go state.inferWorker()

	mux := http.NewServeMux()

	// OpenAI endpoints
	mux.HandleFunc("/v1/chat/completions", state.handleChatCompletions)
	mux.HandleFunc("/v1/completions", state.handleCompletions)
	mux.HandleFunc("/v1/embeddings", state.handleEmbeddings)
	mux.HandleFunc("/v1/models", state.handleModels)

	// Ollama-native endpoints
	mux.HandleFunc("/api/chat", state.handleOllamaChat)
	mux.HandleFunc("/api/generate", state.handleOllamaGenerate)
	mux.HandleFunc("/api/show", state.handleOllamaShow)
	mux.HandleFunc("/api/tags", state.handleModels)

	mux.HandleFunc("/health", state.handleHealth)

	addr := host + ":" + port
	log.Printf("[serve] mongoose API server listening on %s", addr)
	if modelName == "" {
		log.Printf("[serve] no model pre-loaded. Use: ai serve <model>")
	}
	log.Printf("[serve] endpoints: /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/models, /health")

	if err := http.ListenAndServe(addr, mux); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}

// -----------------------------------------------------------------------
// Inference worker — single goroutine owns GPU, processes queued requests
// -----------------------------------------------------------------------

func (s *serveState) inferWorker() {
	for req := range s.inferQueue {
		s.processInferRequest(req)
	}
}

func (s *serveState) processInferRequest(req *inferRequest) {
	defer close(req.resultCh)

	s.mu.RLock()
	fwd := s.fwd
	resetKV := s.resetKV
	s.mu.RUnlock()

	if fwd == nil {
		req.resultCh <- inferToken{err: fmt.Errorf("no model loaded")}
		return
	}

	if resetKV != nil {
		resetKV()
	}

	// Prefill: run all prompt tokens through the model
	var logits []float32
	for i, tid := range req.tokens {
		logits = fwd(tid, i)
	}
	if logits == nil {
		req.resultCh <- inferToken{err: fmt.Errorf("forward pass failed")}
		return
	}

	// Decode: generate tokens one at a time, sending each to the channel
	allTokens := make([]int, len(req.tokens))
	copy(allTokens, req.tokens)

	for step := 0; step < req.maxTokens; step++ {
		nextToken := sampleTopK(logits, req.temp, req.topK)
		allTokens = append(allTokens, nextToken)

		if req.stopToks[nextToken] {
			req.resultCh <- inferToken{tokenID: nextToken, done: true}
			return
		}

		req.resultCh <- inferToken{tokenID: nextToken}

		pos := len(allTokens) - 1
		if pos >= s.maxSeq-1 {
			req.resultCh <- inferToken{done: true}
			return
		}
		logits = fwd(nextToken, pos)
		if logits == nil {
			req.resultCh <- inferToken{done: true}
			return
		}
	}

	req.resultCh <- inferToken{done: true}
}

// submitInfer sends an inference request to the worker and returns the result channel.
func (s *serveState) submitInfer(tokens []int, maxTokens int, temp float32, topK int) chan inferToken {
	s.mu.RLock()
	stop := s.stopTokens
	s.mu.RUnlock()

	req := &inferRequest{
		tokens:    tokens,
		maxTokens: maxTokens,
		temp:      temp,
		topK:      topK,
		stopToks:  stop,
		resultCh:  make(chan inferToken, 64),
	}
	s.inferQueue <- req
	return req.resultCh
}

// -----------------------------------------------------------------------
// Chat template
// -----------------------------------------------------------------------

func (s *serveState) applyTemplate(messages []ChatMessage) []int {
	s.mu.RLock()
	tok := s.tokenizer
	cfg := s.cfg
	s.mu.RUnlock()

	cms := make([]chatMessage, len(messages))
	for i, m := range messages {
		cms[i] = chatMessage{Role: m.Role, Content: m.Content}
	}
	return applyChatTemplate(tok, cms, cfg)
}

// -----------------------------------------------------------------------
// Model loading (unchanged logic, cleaner structure)
// -----------------------------------------------------------------------

func (s *serveState) loadModel(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	path := resolveModel(name)

	configData, err := os.ReadFile(filepath.Join(path, "config.json"))
	if err != nil {
		return fmt.Errorf("no config.json: %w", err)
	}
	var cfg map[string]interface{}
	if err := json.Unmarshal(configData, &cfg); err != nil {
		return fmt.Errorf("parse config: %w", err)
	}

	getInt := func(key string, fallback int) int {
		if v, ok := cfg[key]; ok {
			if f, ok := v.(float64); ok { return int(f) }
		}
		return fallback
	}

	s.modelName = filepath.Base(path)
	s.cfg = cfg
	s.modelDir = path
	s.dim = getInt("hidden_size", 0)
	s.layers = getInt("num_hidden_layers", 0)
	s.heads = getInt("num_attention_heads", 0)
	s.kvHeads = getInt("num_key_value_heads", s.heads)
	s.ffnDim = getInt("intermediate_size", 0)
	s.vocabSize = getInt("vocab_size", 0)
	s.maxSeq = getInt("max_position_embeddings", 2048)

	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	s.tokenizer = tok

	st, err := gguf.OpenSafeTensors(path)
	if err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	s.safetens = st

	s.eng = selectEngine("auto")
	mongoose.LoadKernels()
	log.Printf("[serve] engine: %s", s.eng.Name())

	embedData, _, _ := st.ReadTensorFloat32("model.embed_tokens.weight")
	var lmHeadData []float32
	lmHeadData, _, err = st.ReadTensorFloat32("lm_head.weight")
	if err != nil {
		lmHeadData = embedData
	}
	s.embedData = embedData

	headDim := s.dim / s.heads
	s.halfHead = headDim / 2
	ropeTheta := float32(10000.0)
	if v, ok := cfg["rope_theta"].(float64); ok {
		ropeTheta = float32(v)
	}

	s.cosTab = make([]float32, s.maxSeq*s.halfHead)
	s.sinTab = make([]float32, s.maxSeq*s.halfHead)
	for pos := 0; pos < s.maxSeq; pos++ {
		for j := 0; j < s.halfHead; j++ {
			freq := 1.0 / math.Pow(float64(ropeTheta), float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			s.cosTab[pos*s.halfHead+j] = float32(math.Cos(angle))
			s.sinTab[pos*s.halfHead+j] = float32(math.Sin(angle))
		}
	}

	s.stopTokens = discoverStopTokens(tok, cfg, path)

	// Metal streaming + multi-slot path (preferred, unless --no-stream)
	if !s.noStream {
	if mi := buildMetalStreamingInference(s, st, lmHeadData); mi != nil {
		s.fwd = func(tokenID, pos int) []float32 {
			slot := mi.acquireSlot()
			defer mi.releaseSlot(slot)
			return mi.forward(slot, tokenID, pos)
		}
		s.resetKV = func() {
			for i := 0; i < mi.nSlots; i++ {
				mi.resetKV(i)
			}
		}
	}
	}

	// Metal fused compute path (legacy fallback)
	if s.fwd == nil {
	if metal, ok := s.eng.(*mongoose.Metal); ok {
		ret := metal.BuildFused(s.dim, s.kvHeads*headDim, headDim, s.heads, s.kvHeads, s.ffnDim, s.vocabSize, s.layers, s.maxSeq, float64(ropeTheta), 1e-6)
		if ret == 0 {
			wi := 0
			kvDim := s.kvHeads * headDim
			for l := 0; l < s.layers; l++ {
				prefix := fmt.Sprintf("model.layers.%d.", l)
				loadW := func(n string) { d, _, _ := st.ReadTensorFloat32(prefix + n); if d != nil { metal.FusedSetWeight(wi, d) }; wi++ }
				loadB := func(n string, sz int) { d, _, _ := st.ReadTensorFloat32(prefix + n); if d == nil { d = make([]float32, sz) }; metal.FusedSetWeight(wi, d); wi++ }
				loadW("input_layernorm.weight")
				loadW("self_attn.q_proj.weight"); loadW("self_attn.k_proj.weight"); loadW("self_attn.v_proj.weight")
				loadB("self_attn.q_proj.bias", s.dim); loadB("self_attn.k_proj.bias", kvDim); loadB("self_attn.v_proj.bias", kvDim)
				loadW("self_attn.o_proj.weight"); loadW("post_attention_layernorm.weight")
				loadW("mlp.gate_proj.weight"); loadW("mlp.up_proj.weight"); loadW("mlp.down_proj.weight")
			}
			fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
			metal.FusedSetWeight(wi, fnorm); wi++
			metal.FusedSetWeight(wi, lmHeadData); wi++

			fHidden := make([]float32, s.dim)
			fLogits := make([]float32, s.vocabSize)
			s.fwd = func(tokenID, pos int) []float32 {
				tokOff := tokenID * s.dim
				if tokenID < 0 || tokOff+s.dim > len(s.embedData) { return nil }
				copy(fHidden, s.embedData[tokOff:tokOff+s.dim])
				metal.FusedStep(fHidden, s.cosTab[pos*s.halfHead:pos*s.halfHead+s.halfHead],
					s.sinTab[pos*s.halfHead:pos*s.halfHead+s.halfHead], pos, fLogits)
				return fLogits
			}
			s.resetKV = func() { metal.FusedResetKV() }
			log.Printf("[serve] Metal fused inference ready (%d weights)", wi)
		}
	}
	}

	// CUDA Q8/Q4 fused kernel path (zero-alloc hot path)
	if s.fwd == nil {
		if ci := buildCUDAQ8Inference(s, st, lmHeadData); ci != nil {
			s.fwd = func(tokenID, pos int) []float32 {
				return ci.forward(tokenID, pos, s.embedData)
			}
			s.resetKV = func() { ci.resetKV() }
		}
	}

	// Metal inference graph fallback
	if s.fwd == nil {
		kvDim := s.kvHeads * headDim
		if metal, ok := s.eng.(*mongoose.Metal); ok {
			ret := metal.BuildInferGraph(s.dim, kvDim, headDim, s.heads, s.kvHeads, s.ffnDim, s.vocabSize, s.layers, float64(ropeTheta))
			if ret == 0 {
				wi := 0
				for l := 0; l < s.layers; l++ {
					prefix := fmt.Sprintf("model.layers.%d.", l)
					loadW := func(n string) { d, _, _ := st.ReadTensorFloat32(prefix + n); if d != nil { metal.InferSetWeight(wi, d) }; wi++ }
					loadB := func(n string, sz int) { d, _, _ := st.ReadTensorFloat32(prefix + n); if d == nil { d = make([]float32, sz) }; metal.InferSetWeight(wi, d); wi++ }
					loadW("input_layernorm.weight")
					loadW("self_attn.q_proj.weight"); loadW("self_attn.k_proj.weight"); loadW("self_attn.v_proj.weight")
					loadB("self_attn.q_proj.bias", s.dim); loadB("self_attn.k_proj.bias", kvDim); loadB("self_attn.v_proj.bias", kvDim)
					loadW("self_attn.o_proj.weight"); loadW("post_attention_layernorm.weight")
					loadW("mlp.gate_proj.weight"); loadW("mlp.up_proj.weight"); loadW("mlp.down_proj.weight")
				}
				fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
				metal.InferSetWeight(wi, fnorm); wi++
				metal.InferSetWeight(wi, lmHeadData); wi++

				keyCache := make([][]float32, s.layers)
				valCache := make([][]float32, s.layers)
				for l := 0; l < s.layers; l++ {
					keyCache[l] = make([]float32, s.maxSeq*kvDim)
					valCache[l] = make([]float32, s.maxSeq*kvDim)
				}
				att := make([]float32, s.heads*s.maxSeq)
				mHidden := make([]float32, s.dim)
				mQBuf := make([]float32, s.dim)
				mKBuf := make([]float32, kvDim)
				mVBuf := make([]float32, kvDim)
				mAttnOut := make([]float32, s.dim)
				mLogits := make([]float32, s.vocabSize)
				invSqrtHeadDim := float32(1.0 / math.Sqrt(float64(headDim)))
				kvMulConst := s.heads / s.kvHeads

				s.resetKV = func() {
					for l := 0; l < s.layers; l++ {
						for i := range keyCache[l] { keyCache[l][i] = 0 }
						for i := range valCache[l] { valCache[l][i] = 0 }
					}
				}

				s.fwd = func(tokenID, pos int) []float32 {
					tokOff := tokenID * s.dim
					if tokOff+s.dim > len(s.embedData) { return nil }
					copy(mHidden, s.embedData[tokOff:tokOff+s.dim])
					cosSlice := s.cosTab[pos*s.halfHead : pos*s.halfHead+s.halfHead]
					sinSlice := s.sinTab[pos*s.halfHead : pos*s.halfHead+s.halfHead]

					for l := 0; l < s.layers; l++ {
						metal.InferForwardA(mHidden, cosSlice, sinSlice, mQBuf, mKBuf, mVBuf, l)
						copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], mKBuf)
						copy(valCache[l][pos*kvDim:(pos+1)*kvDim], mVBuf)

						for i := range mAttnOut { mAttnOut[i] = 0 }
						for h := 0; h < s.heads; h++ {
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
				log.Printf("[serve] Metal inference graph ready (%d weights)", wi)
			}
		}
	}

	// Generic CPU fallback
	if s.fwd == nil {
		log.Printf("[serve] using generic inference (weights streamed, CPU attention)")
		normEps := float32(1e-6)
		if v, ok := cfg["rms_norm_eps"].(float64); ok { normEps = float32(v) }
		dim := s.dim
		nLayers := s.layers
		heads := s.heads
		kvHeads := s.kvHeads
		headDim := dim / heads
		kvDim := kvHeads * headDim
		ffnDim := s.ffnDim
		vocabSize := s.vocabSize
		maxSeq := s.maxSeq

		keyCache := make([][]float32, nLayers)
		valCache := make([][]float32, nLayers)
		for l := 0; l < nLayers; l++ {
			keyCache[l] = make([]float32, maxSeq*kvDim)
			valCache[l] = make([]float32, maxSeq*kvDim)
		}
		att := make([]float32, heads*maxSeq)
		x := make([]float32, dim)
		buf := make([]float32, dim)
		q := make([]float32, dim)
		kk := make([]float32, kvDim)
		vv := make([]float32, kvDim)
		attnOut := make([]float32, dim)
		ffnBuf := make([]float32, ffnDim)
		ffnBuf2 := make([]float32, ffnDim)

		s.resetKV = func() {
			for l := 0; l < nLayers; l++ {
				for i := range keyCache[l] { keyCache[l][i] = 0 }
				for i := range valCache[l] { valCache[l][i] = 0 }
			}
		}

		rmsNorm := func(data, weight []float32, eps float32) {
			n := len(data)
			var ss float32
			for i := 0; i < n; i++ { ss += data[i] * data[i] }
			ss = 1.0 / float32(math.Sqrt(float64(ss/float32(n)+eps)))
			for i := 0; i < n; i++ { data[i] = data[i] * ss * weight[i] }
		}
		mv := func(out, W, xIn []float32, rows, cols int) {
			copy(out, s.eng.MatMul(W, xIn, rows, cols, 1))
		}

		s.fwd = func(tokenID, pos int) []float32 {
			tokOff := tokenID * dim
			if tokOff+dim > len(s.embedData) { return nil }
			copy(x, s.embedData[tokOff:tokOff+dim])

			for l := 0; l < nLayers; l++ {
				prefix := fmt.Sprintf("model.layers.%d.", l)
				normW, _, _ := st.ReadTensorFloat32(prefix + "input_layernorm.weight")
				copy(buf, x)
				rmsNorm(buf, normW, normEps)

				wq, _, _ := st.ReadTensorFloat32(prefix + "self_attn.q_proj.weight")
				wk, _, _ := st.ReadTensorFloat32(prefix + "self_attn.k_proj.weight")
				wv, _, _ := st.ReadTensorFloat32(prefix + "self_attn.v_proj.weight")
				mv(q, wq, buf, dim, dim)
				mv(kk[:kvDim], wk, buf, kvDim, dim)
				mv(vv[:kvDim], wv, buf, kvDim, dim)

				if bq, _, err := st.ReadTensorFloat32(prefix + "self_attn.q_proj.bias"); err == nil {
					for i := range bq { q[i] += bq[i] }
				}
				if bk, _, err := st.ReadTensorFloat32(prefix + "self_attn.k_proj.bias"); err == nil {
					for i := range bk { kk[i] += bk[i] }
				}
				if bv, _, err := st.ReadTensorFloat32(prefix + "self_attn.v_proj.bias"); err == nil {
					for i := range bv { vv[i] += bv[i] }
				}

				applyRoPE(q, kk[:kvDim], pos, headDim, float32(ropeTheta), heads, kvHeads)

				copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], kk[:kvDim])
				copy(valCache[l][pos*kvDim:(pos+1)*kvDim], vv[:kvDim])

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

				wo, _, _ := st.ReadTensorFloat32(prefix + "self_attn.o_proj.weight")
				proj := make([]float32, dim)
				mv(proj, wo, attnOut, dim, dim)
				for i := 0; i < dim; i++ { x[i] += proj[i] }

				normW2, _, _ := st.ReadTensorFloat32(prefix + "post_attention_layernorm.weight")
				copy(buf, x)
				rmsNorm(buf, normW2, normEps)

				gate, _, _ := st.ReadTensorFloat32(prefix + "mlp.gate_proj.weight")
				up, _, _ := st.ReadTensorFloat32(prefix + "mlp.up_proj.weight")
				down, _, _ := st.ReadTensorFloat32(prefix + "mlp.down_proj.weight")
				mv(ffnBuf, gate, buf, ffnDim, dim)
				mv(ffnBuf2, up, buf, ffnDim, dim)
				for i := 0; i < ffnDim; i++ {
					ffnBuf[i] = silu(ffnBuf[i]) * ffnBuf2[i]
				}
				downOut := make([]float32, dim)
				mv(downOut, down, ffnBuf, dim, ffnDim)
				for i := 0; i < dim; i++ { x[i] += downOut[i] }
			}

			finalNorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
			rmsNorm(x, finalNorm, normEps)
			logits := make([]float32, vocabSize)
			mv(logits, lmHeadData, x, vocabSize, dim)
			return logits
		}
	}

	return nil
}

// -----------------------------------------------------------------------
// OpenAI handlers
// -----------------------------------------------------------------------

func (s *serveState) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}

	if err := s.ensureModel(req.Model); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "model_not_found")
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages array is required", "invalid_request_error")
		return
	}

	maxTokens := 256
	if req.MaxTokens != nil { maxTokens = *req.MaxTokens }
	temp := float32(0.7)
	if req.Temperature != nil { temp = float32(*req.Temperature) }
	topK := 40

	promptTokens := s.applyTemplate(req.Messages)

	if req.Stream {
		s.streamChatCompletions(w, promptTokens, maxTokens, temp, topK)
		return
	}

	// Non-streaming: collect all tokens, return at once
	t0 := time.Now()
	resultCh := s.submitInfer(promptTokens, maxTokens, temp, topK)

	s.mu.RLock()
	tok := s.tokenizer
	s.mu.RUnlock()

	var genTokens []int
	for it := range resultCh {
		if it.err != nil {
			writeError(w, http.StatusInternalServerError, it.err.Error(), "server_error")
			return
		}
		if it.done { break }
		genTokens = append(genTokens, it.tokenID)
	}

	generated := tok.Decode(genTokens)
	if idx := findSpecialToken(generated); idx >= 0 {
		generated = generated[:idx]
	}
	generated = strings.TrimSpace(generated)
	elapsed := time.Since(t0)

	resp := ChatCompletionResponse{
		ID:      generateID("chatcmpl"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.modelName,
		Choices: []ChatChoice{{
			Index:        0,
			Message:      ChatMessage{Role: "assistant", Content: generated},
			FinishReason: "stop",
		}},
		Usage: UsageInfo{
			PromptTokens:     len(promptTokens),
			CompletionTokens: len(genTokens),
			TotalTokens:      len(promptTokens) + len(genTokens),
		},
	}

	log.Printf("[serve] chat: %d prompt + %d gen tokens in %v (%.1f tok/s)",
		len(promptTokens), len(genTokens), elapsed.Round(time.Millisecond),
		float64(len(genTokens))/elapsed.Seconds())

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *serveState) streamChatCompletions(w http.ResponseWriter, promptTokens []int, maxTokens int, temp float32, topK int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported", "server_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := generateID("chatcmpl")
	now := time.Now().Unix()

	// Role chunk
	writeSSEChunk(w, flusher, ChatCompletionChunk{
		ID: id, Object: "chat.completion.chunk", Created: now, Model: s.modelName,
		Choices: []ChatChunkChoice{{Index: 0, Delta: ChatDelta{Role: "assistant"}}},
	})

	s.mu.RLock()
	tok := s.tokenizer
	s.mu.RUnlock()

	t0 := time.Now()
	resultCh := s.submitInfer(promptTokens, maxTokens, temp, topK)

	genCount := 0
	for it := range resultCh {
		if it.err != nil { break }
		if it.done { break }
		genCount++
		text := tok.Decode([]int{it.tokenID})
		writeSSEChunk(w, flusher, ChatCompletionChunk{
			ID: id, Object: "chat.completion.chunk", Created: now, Model: s.modelName,
			Choices: []ChatChunkChoice{{Index: 0, Delta: ChatDelta{Content: text}}},
		})
	}

	elapsed := time.Since(t0)
	log.Printf("[serve] stream: %d prompt + %d gen tokens in %v (%.1f tok/s)",
		len(promptTokens), genCount, elapsed.Round(time.Millisecond),
		float64(genCount)/elapsed.Seconds())

	stop := "stop"
	writeSSEChunk(w, flusher, ChatCompletionChunk{
		ID: id, Object: "chat.completion.chunk", Created: now, Model: s.modelName,
		Choices: []ChatChunkChoice{{Index: 0, Delta: ChatDelta{}, FinishReason: &stop}},
	})

	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *serveState) handleCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var req CompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}

	if err := s.ensureModel(req.Model); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "model_not_found")
		return
	}

	maxTokens := 256
	if req.MaxTokens != nil { maxTokens = *req.MaxTokens }
	temp := float32(0.7)
	if req.Temperature != nil { temp = float32(*req.Temperature) }
	topK := 40

	s.mu.RLock()
	tok := s.tokenizer
	s.mu.RUnlock()
	promptTokens := tok.Encode(req.Prompt)

	if req.Stream {
		s.streamCompletions(w, req, promptTokens, maxTokens, temp, topK)
		return
	}

	t0 := time.Now()
	resultCh := s.submitInfer(promptTokens, maxTokens, temp, topK)

	var genTokens []int
	for it := range resultCh {
		if it.err != nil {
			writeError(w, http.StatusInternalServerError, it.err.Error(), "server_error")
			return
		}
		if it.done { break }
		genTokens = append(genTokens, it.tokenID)
	}

	generated := tok.Decode(genTokens)
	if idx := findSpecialToken(generated); idx >= 0 {
		generated = generated[:idx]
	}
	elapsed := time.Since(t0)

	log.Printf("[serve] completion: %d prompt + %d gen tokens in %v (%.1f tok/s)",
		len(promptTokens), len(genTokens), elapsed.Round(time.Millisecond),
		float64(len(genTokens))/elapsed.Seconds())

	resp := CompletionResponse{
		ID:      generateID("cmpl"),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   s.modelName,
		Choices: []CompletionChoice{{Text: generated, Index: 0, FinishReason: "stop"}},
		Usage: UsageInfo{
			PromptTokens:     len(promptTokens),
			CompletionTokens: len(genTokens),
			TotalTokens:      len(promptTokens) + len(genTokens),
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *serveState) streamCompletions(w http.ResponseWriter, req CompletionRequest, promptTokens []int, maxTokens int, temp float32, topK int) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming not supported", "server_error")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	id := generateID("cmpl")
	now := time.Now().Unix()

	s.mu.RLock()
	tok := s.tokenizer
	s.mu.RUnlock()

	resultCh := s.submitInfer(promptTokens, maxTokens, temp, topK)

	for it := range resultCh {
		if it.err != nil { break }
		if it.done { break }
		text := tok.Decode([]int{it.tokenID})
		chunk := CompletionResponse{
			ID: id, Object: "text_completion", Created: now, Model: s.modelName,
			Choices: []CompletionChoice{{Text: text, Index: 0}},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	// Final chunk
	final := CompletionResponse{
		ID: id, Object: "text_completion", Created: now, Model: s.modelName,
		Choices: []CompletionChoice{{Text: "", Index: 0, FinishReason: "stop"}},
	}
	data, _ := json.Marshal(final)
	fmt.Fprintf(w, "data: %s\n\n", data)
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func (s *serveState) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}

	if err := s.ensureModel(req.Model); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "model_not_found")
		return
	}

	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []interface{}:
		for _, item := range v {
			if str, ok := item.(string); ok { inputs = append(inputs, str) }
		}
	default:
		writeError(w, http.StatusBadRequest, "input must be string or array of strings", "invalid_request_error")
		return
	}

	s.mu.RLock()
	tok := s.tokenizer
	s.mu.RUnlock()

	totalTokens := 0
	var data []EmbeddingData
	for i, text := range inputs {
		tokens := tok.Encode(text)
		totalTokens += len(tokens)
		embedding := make([]float32, s.dim)
		data = append(data, EmbeddingData{Object: "embedding", Embedding: embedding, Index: i})
	}

	resp := EmbeddingResponse{
		Object: "list", Data: data, Model: s.modelName,
		Usage: UsageInfo{PromptTokens: totalTokens, TotalTokens: totalTokens},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// -----------------------------------------------------------------------
// Ollama-native handlers
// -----------------------------------------------------------------------

func (s *serveState) handleOllamaChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var req OllamaChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}

	if err := s.ensureModel(req.Model); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "model_not_found")
		return
	}

	stream := true
	if req.Stream != nil { stream = *req.Stream }

	maxTokens := 256
	temp := float32(0.7)
	topK := 40
	if req.Options != nil {
		if req.Options.NumPredict > 0 { maxTokens = req.Options.NumPredict }
		if req.Options.Temperature > 0 { temp = float32(req.Options.Temperature) }
		if req.Options.TopK > 0 { topK = req.Options.TopK }
	}

	msgs := make([]ChatMessage, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = ChatMessage{Role: m.Role, Content: m.Content}
	}
	promptTokens := s.applyTemplate(msgs)

	s.mu.RLock()
	tok := s.tokenizer
	s.mu.RUnlock()

	tTotal := time.Now()
	resultCh := s.submitInfer(promptTokens, maxTokens, temp, topK)

	tEval := time.Now()
	evalCount := 0

	if stream {
		flusher, ok := w.(http.Flusher)
		if !ok {
			writeError(w, http.StatusInternalServerError, "streaming not supported", "server_error")
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Cache-Control", "no-cache")

		for it := range resultCh {
			if it.err != nil { break }
			if it.done { break }
			evalCount++
			text := tok.Decode([]int{it.tokenID})
			chunk := OllamaChatResponse{
				Model:     s.modelName,
				CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
				Message:   OllamaChatMsg{Role: "assistant", Content: text},
				Done:      false,
			}
			data, _ := json.Marshal(chunk)
			w.Write(data)
			w.Write([]byte("\n"))
			flusher.Flush()
		}

		evalDur := time.Since(tEval)
		totalDur := time.Since(tTotal)
		final := OllamaChatResponse{
			Model:           s.modelName,
			CreatedAt:       time.Now().UTC().Format(time.RFC3339Nano),
			Message:         OllamaChatMsg{Role: "assistant"},
			Done:            true,
			TotalDuration:   totalDur.Nanoseconds(),
			PromptEvalCount: len(promptTokens),
			PromptEvalDur:   tEval.Sub(tTotal).Nanoseconds(),
			EvalCount:       evalCount,
			EvalDuration:    evalDur.Nanoseconds(),
		}
		data, _ := json.Marshal(final)
		w.Write(data)
		w.Write([]byte("\n"))
		flusher.Flush()

		log.Printf("[serve] ollama chat: %d prompt + %d gen in %v (%.1f tok/s)",
			len(promptTokens), evalCount, totalDur.Round(time.Millisecond),
			float64(evalCount)/evalDur.Seconds())
	} else {
		var genTokens []int
		for it := range resultCh {
			if it.err != nil {
				writeError(w, http.StatusInternalServerError, it.err.Error(), "server_error")
				return
			}
			if it.done { break }
			genTokens = append(genTokens, it.tokenID)
		}

		generated := tok.Decode(genTokens)
		if idx := findSpecialToken(generated); idx >= 0 { generated = generated[:idx] }
		generated = strings.TrimSpace(generated)
		evalDur := time.Since(tEval)
		totalDur := time.Since(tTotal)

		resp := OllamaChatResponse{
			Model:           s.modelName,
			CreatedAt:       time.Now().UTC().Format(time.RFC3339Nano),
			Message:         OllamaChatMsg{Role: "assistant", Content: generated},
			Done:            true,
			TotalDuration:   totalDur.Nanoseconds(),
			PromptEvalCount: len(promptTokens),
			PromptEvalDur:   tEval.Sub(tTotal).Nanoseconds(),
			EvalCount:       len(genTokens),
			EvalDuration:    evalDur.Nanoseconds(),
		}

		log.Printf("[serve] ollama chat: %d prompt + %d gen in %v (%.1f tok/s)",
			len(promptTokens), len(genTokens), totalDur.Round(time.Millisecond),
			float64(len(genTokens))/evalDur.Seconds())

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func (s *serveState) handleOllamaGenerate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var req OllamaGenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error(), "invalid_request_error")
		return
	}

	if err := s.ensureModel(req.Model); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "model_not_found")
		return
	}

	stream := true
	if req.Stream != nil { stream = *req.Stream }

	maxTokens := 256
	temp := float32(0.7)
	topK := 40
	if req.Options != nil {
		if req.Options.NumPredict > 0 { maxTokens = req.Options.NumPredict }
		if req.Options.Temperature > 0 { temp = float32(req.Options.Temperature) }
		if req.Options.TopK > 0 { topK = req.Options.TopK }
	}

	s.mu.RLock()
	tok := s.tokenizer
	s.mu.RUnlock()
	promptTokens := tok.Encode(req.Prompt)

	tTotal := time.Now()
	resultCh := s.submitInfer(promptTokens, maxTokens, temp, topK)

	tEval := time.Now()
	evalCount := 0

	if stream {
		flusher, ok := w.(http.Flusher)
		if !ok {
			writeError(w, http.StatusInternalServerError, "streaming not supported", "server_error")
			return
		}
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.Header().Set("Cache-Control", "no-cache")

		for it := range resultCh {
			if it.err != nil { break }
			if it.done { break }
			evalCount++
			text := tok.Decode([]int{it.tokenID})
			chunk := OllamaGenerateResponse{
				Model:     s.modelName,
				CreatedAt: time.Now().UTC().Format(time.RFC3339Nano),
				Response:  text,
				Done:      false,
			}
			data, _ := json.Marshal(chunk)
			w.Write(data)
			w.Write([]byte("\n"))
			flusher.Flush()
		}

		evalDur := time.Since(tEval)
		totalDur := time.Since(tTotal)
		final := OllamaGenerateResponse{
			Model:           s.modelName,
			CreatedAt:       time.Now().UTC().Format(time.RFC3339Nano),
			Response:        "",
			Done:            true,
			TotalDuration:   totalDur.Nanoseconds(),
			PromptEvalCount: len(promptTokens),
			PromptEvalDur:   tEval.Sub(tTotal).Nanoseconds(),
			EvalCount:       evalCount,
			EvalDuration:    evalDur.Nanoseconds(),
		}
		data, _ := json.Marshal(final)
		w.Write(data)
		w.Write([]byte("\n"))
		flusher.Flush()

		log.Printf("[serve] ollama generate: %d prompt + %d gen in %v (%.1f tok/s)",
			len(promptTokens), evalCount, totalDur.Round(time.Millisecond),
			float64(evalCount)/evalDur.Seconds())
	} else {
		var genTokens []int
		for it := range resultCh {
			if it.err != nil {
				writeError(w, http.StatusInternalServerError, it.err.Error(), "server_error")
				return
			}
			if it.done { break }
			genTokens = append(genTokens, it.tokenID)
		}

		generated := tok.Decode(genTokens)
		if idx := findSpecialToken(generated); idx >= 0 { generated = generated[:idx] }
		evalDur := time.Since(tEval)
		totalDur := time.Since(tTotal)

		resp := OllamaGenerateResponse{
			Model:           s.modelName,
			CreatedAt:       time.Now().UTC().Format(time.RFC3339Nano),
			Response:        generated,
			Done:            true,
			TotalDuration:   totalDur.Nanoseconds(),
			PromptEvalCount: len(promptTokens),
			PromptEvalDur:   tEval.Sub(tTotal).Nanoseconds(),
			EvalCount:       len(genTokens),
			EvalDuration:    evalDur.Nanoseconds(),
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}
}

func (s *serveState) handleOllamaShow(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	nParams := int64(s.vocabSize) * int64(s.dim)
	for l := 0; l < s.layers; l++ {
		nParams += int64(s.dim)*int64(s.dim)*2 + int64(s.kvHeads*(s.dim/s.heads))*int64(s.dim)*2 + int64(s.ffnDim)*int64(s.dim)*3
	}

	paramSize := fmt.Sprintf("%.1fB", float64(nParams)/1e9)
	quantLevel := "Q8_0"
	if nParams > 4000000000 { quantLevel = "Q4_0" }

	family := "unknown"
	if arch, ok := s.cfg["model_type"].(string); ok { family = arch }

	resp := OllamaShowResponse{
		Parameters: fmt.Sprintf("num_params %d", nParams),
		Details: OllamaModelDetail{
			Format:            "safetensors",
			Family:            family,
			Families:          []string{family},
			ParameterSize:     paramSize,
			QuantizationLevel: quantLevel,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// -----------------------------------------------------------------------
// Common handlers
// -----------------------------------------------------------------------

func (s *serveState) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var models []ModelInfo

	entries, _ := os.ReadDir(modelsDir)
	for _, e := range entries {
		if !e.IsDir() { continue }
		info, _ := e.Info()
		created := int64(0)
		if info != nil { created = info.ModTime().Unix() }
		models = append(models, ModelInfo{ID: e.Name(), Object: "model", Created: created, OwnedBy: "mongoose"})
	}

	s.mu.RLock()
	if s.modelName != "" {
		found := false
		for _, m := range models {
			if m.ID == s.modelName { found = true; break }
		}
		if !found {
			models = append(models, ModelInfo{ID: s.modelName, Object: "model", Created: time.Now().Unix(), OwnedBy: "mongoose"})
		}
	}
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(ModelsResponse{Object: "list", Data: models})
}

func (s *serveState) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	modelLoaded := s.modelName != ""
	modelName := s.modelName
	hasGPU := s.eng != nil
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":       "ok",
		"model_loaded": modelLoaded,
		"model":        modelName,
		"gpu":          hasGPU,
		"version":      "ai v1.2.1",
	})
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

func (s *serveState) ensureModel(requestedModel string) error {
	if requestedModel == "" {
		s.mu.RLock()
		loaded := s.modelName != ""
		s.mu.RUnlock()
		if !loaded { return fmt.Errorf("no model specified and none pre-loaded") }
		return nil
	}

	s.mu.RLock()
	current := s.modelName
	s.mu.RUnlock()

	if current != "" && (current == requestedModel || strings.Contains(strings.ToLower(current), strings.ToLower(requestedModel))) {
		return nil
	}

	if current == "" {
		return fmt.Errorf("model %q not loaded — start server with: ai serve %s", requestedModel, requestedModel)
	}
	return nil
}

func generateID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

func writeError(w http.ResponseWriter, status int, message, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(ErrorResponse{Error: ErrorDetail{Message: message, Type: errType}})
}

func writeSSEChunk(w http.ResponseWriter, flusher http.Flusher, chunk ChatCompletionChunk) {
	data, err := json.Marshal(chunk)
	if err != nil { return }
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

// Silence unused import warnings.
var _ = mongoose.Engine(nil)

// -----------------------------------------------------------------------
// Daemon mode
// -----------------------------------------------------------------------

func servePidPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".ai", "serve.pid")
}

func daemonize(host, port, modelName string) {
	exe, err := os.Executable()
	if err != nil { log.Fatalf("cannot find self: %v", err) }

	stopExistingDaemon()

	args := []string{"serve"}
	if modelName != "" { args = append(args, fmt.Sprintf("model=%s", modelName)) }
	args = append(args, fmt.Sprintf("host=%s", host), fmt.Sprintf("port=%s", port))

	cmd := exec.Command(exe, args...)
	setSysProcAttr(cmd)
	cmd.Stdout = nil
	cmd.Stderr = nil
	cmd.Stdin = nil

	if err := cmd.Start(); err != nil { log.Fatalf("failed to start daemon: %v", err) }

	pid := cmd.Process.Pid

	home, _ := os.UserHomeDir()
	os.MkdirAll(filepath.Join(home, ".ai"), 0755)
	os.WriteFile(servePidPath(), []byte(strconv.Itoa(pid)), 0644)

	fmt.Printf("ai serve daemon started (pid=%d) on %s:%s\n", pid, host, port)
	if modelName != "" { fmt.Printf("  model: %s\n", modelName) }
	fmt.Printf("  pid:   %s\n", servePidPath())
	fmt.Printf("  stop:  ai serve --stop\n")
}

func stopExistingDaemon() {
	data, err := os.ReadFile(servePidPath())
	if err != nil { return }
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil { os.Remove(servePidPath()); return }
	proc, err := os.FindProcess(pid)
	if err != nil { os.Remove(servePidPath()); return }
	if err := proc.Signal(os.Signal(os.Interrupt)); err != nil { os.Remove(servePidPath()); return }
	proc.Signal(os.Interrupt)
	time.Sleep(500 * time.Millisecond)
	proc.Kill()
	os.Remove(servePidPath())
}
