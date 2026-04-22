// serve.go — OpenAI-compatible API server for mongoose.
//
// Implements the OpenAI API surface that tools (LangChain, LiteLLM, Continue,
// Open WebUI, etc.) expect. Drop-in replacement for Ollama's /v1 endpoints.
//
// Endpoints:
//   POST /v1/chat/completions   — streaming (SSE) + non-streaming
//   POST /v1/completions        — legacy completions
//   POST /v1/embeddings         — text embeddings
//   GET  /v1/models             — list loaded models
//   GET  /health                — liveness probe
//
// Usage:
//   ai serve [--host 0.0.0.0] [--port 11434] [--model <name>]
//   ai serve Qwen2.5-0.5B --daemon     Run in background, write PID to ~/.ai/serve.pid

package main

import (
	"bufio"
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
// OpenAI API request/response types
// -----------------------------------------------------------------------

// ChatCompletionRequest matches the OpenAI chat completions request body.
type ChatCompletionRequest struct {
	Model            string            `json:"model"`
	Messages         []ChatMessage     `json:"messages"`
	Temperature      *float64          `json:"temperature,omitempty"`
	TopP             *float64          `json:"top_p,omitempty"`
	N                int               `json:"n,omitempty"`
	Stream           bool              `json:"stream,omitempty"`
	Stop             interface{}       `json:"stop,omitempty"` // string or []string
	MaxTokens        *int              `json:"max_tokens,omitempty"`
	PresencePenalty  float64           `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64           `json:"frequency_penalty,omitempty"`
	User             string            `json:"user,omitempty"`
	Seed             *int              `json:"seed,omitempty"`
	ResponseFormat   *ResponseFormat   `json:"response_format,omitempty"`
}

// ChatMessage is a single message in a chat conversation.
type ChatMessage struct {
	Role    string `json:"role"`    // "system", "user", "assistant"
	Content string `json:"content"`
}

// ResponseFormat constrains the output format.
type ResponseFormat struct {
	Type string `json:"type"` // "text" or "json_object"
}

// ChatCompletionResponse is the non-streaming response.
type ChatCompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []ChatChoice       `json:"choices"`
	Usage   UsageInfo          `json:"usage"`
}

// ChatChoice is one completion choice.
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"` // "stop", "length"
}

// ChatCompletionChunk is one SSE chunk for streaming responses.
type ChatCompletionChunk struct {
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Created int64               `json:"created"`
	Model   string              `json:"model"`
	Choices []ChatChunkChoice   `json:"choices"`
}

// ChatChunkChoice is a streaming delta.
type ChatChunkChoice struct {
	Index        int            `json:"index"`
	Delta        ChatDelta      `json:"delta"`
	FinishReason *string        `json:"finish_reason"` // null until final chunk
}

// ChatDelta is the incremental content in a streaming chunk.
type ChatDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// CompletionRequest matches the legacy OpenAI completions endpoint.
type CompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	N           int      `json:"n,omitempty"`
	Stream      bool     `json:"stream,omitempty"`
	Stop        interface{} `json:"stop,omitempty"`
	Suffix      string   `json:"suffix,omitempty"`
}

// CompletionResponse is the legacy completions response.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   UsageInfo          `json:"usage"`
}

// CompletionChoice is one legacy completion choice.
type CompletionChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

// EmbeddingRequest matches the OpenAI embeddings endpoint.
type EmbeddingRequest struct {
	Model          string      `json:"model"`
	Input          interface{} `json:"input"`          // string or []string
	EncodingFormat string      `json:"encoding_format"` // "float" or "base64"
}

// EmbeddingResponse is the embeddings response.
type EmbeddingResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  UsageInfo       `json:"usage"`
}

// EmbeddingData is one embedding vector.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// UsageInfo tracks token consumption.
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens"`
}

// ModelInfo describes one model for the /v1/models response.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse is the /v1/models list response.
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ErrorResponse matches the OpenAI error envelope.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail is the inner error object.
type ErrorDetail struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

// -----------------------------------------------------------------------
// Server state
// -----------------------------------------------------------------------

// serveState holds the loaded model and engine state for the running server.
type serveState struct {
	mu      sync.RWMutex
	inferMu sync.Mutex

	// Model metadata
	modelName string
	modelDir  string
	vocabSize int
	dim       int
	layers    int
	heads     int
	kvHeads   int
	ffnDim    int
	maxSeq    int

	// Loaded components (nil until model is loaded)
	tokenizer *tokenizer.Tokenizer
	safetens  *gguf.SafeTensors
	cfg       map[string]interface{}

	// GPU engine (nil if CPU-only)
	eng mongoose.Engine

	// Inference pipeline
	fwd       func(tokenID, pos int) []float32
	resetKV   func()
	embedData []float32
	cosTab    []float32
	sinTab    []float32
	halfHead  int
}

// -----------------------------------------------------------------------
// Server entrypoint
// -----------------------------------------------------------------------

// cmdServe starts the OpenAI-compatible API server.
func cmdServe(args map[string]string) {
	host := "0.0.0.0"
	port := "11434"
	modelName := ""

	// key=value args: ai serve model=X host=0.0.0.0 port=8080
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

	// Also support --flag syntax for backwards compat
	for i := 2; i < len(os.Args); i++ {
		switch os.Args[i] {
		case "--host":
			if i+1 < len(os.Args) {
				host = os.Args[i+1]
				i++
			}
		case "--port":
			if i+1 < len(os.Args) {
				port = os.Args[i+1]
				i++
			}
		case "--model":
			if i+1 < len(os.Args) {
				modelName = os.Args[i+1]
				i++
			}
		case "--daemon", "-d":
			daemon = true
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

	state := &serveState{}

	// Pre-load model if specified
	if modelName != "" {
		if err := state.loadModel(modelName); err != nil {
			log.Fatalf("Failed to load model %q: %v", modelName, err)
		}
		log.Printf("[serve] model loaded: %s (dim=%d, layers=%d, heads=%d, vocab=%d)",
			state.modelName, state.dim, state.layers, state.heads, state.vocabSize)
	}

	// Register routes
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", state.handleChatCompletions)
	mux.HandleFunc("/v1/completions", state.handleCompletions)
	mux.HandleFunc("/v1/embeddings", state.handleEmbeddings)
	mux.HandleFunc("/v1/models", state.handleModels)
	mux.HandleFunc("/health", state.handleHealth)

	// Also serve at root paths for Ollama-compatible clients
	mux.HandleFunc("/api/chat", state.handleChatCompletions)
	mux.HandleFunc("/api/generate", state.handleCompletions)
	mux.HandleFunc("/api/embeddings", state.handleEmbeddings)
	mux.HandleFunc("/api/tags", state.handleModels)

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
// Model loading
// -----------------------------------------------------------------------

// loadModel initializes the model, tokenizer, and GPU engine.
func (s *serveState) loadModel(name string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	path := resolveModel(name)

	// Read config.json
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
			if f, ok := v.(float64); ok {
				return int(f)
			}
		}
		return fallback
	}

	s.modelName = name
	s.cfg = cfg
	s.modelDir = path
	s.dim = getInt("hidden_size", 0)
	s.layers = getInt("num_hidden_layers", 0)
	s.heads = getInt("num_attention_heads", 0)
	s.kvHeads = getInt("num_key_value_heads", s.heads)
	s.ffnDim = getInt("intermediate_size", 0)
	s.vocabSize = getInt("vocab_size", 0)
	s.maxSeq = getInt("max_position_embeddings", 2048)

	// Load tokenizer
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	s.tokenizer = tok

	// Open safetensors (lazy — headers only, no weight data read yet)
	st, err := gguf.OpenSafeTensors(path)
	if err != nil {
		return fmt.Errorf("safetensors: %w", err)
	}
	s.safetens = st

	// Init GPU
	s.eng = selectEngine("auto")
	mongoose.LoadKernels()
	log.Printf("[serve] engine: %s", s.eng.Name())

	// Load embeddings
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

	// RoPE tables
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

	// Set up forward pass using cmdInferGPU's Metal fused path or CPU fallback
	if metal, ok := s.eng.(*mongoose.Metal); ok {
		ret := metal.BuildFused(s.dim, s.kvHeads*headDim, headDim, s.heads, s.kvHeads, s.ffnDim, s.vocabSize, s.layers, s.maxSeq)
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
				if tokOff+s.dim > len(s.embedData) { return nil }
				copy(fHidden, s.embedData[tokOff:tokOff+s.dim])
				metal.FusedStep(fHidden, s.cosTab[pos*s.halfHead:pos*s.halfHead+s.halfHead],
					s.sinTab[pos*s.halfHead:pos*s.halfHead+s.halfHead], pos, fLogits)
				return fLogits
			}
			log.Printf("[serve] Metal fused inference ready (%d weights)", wi)
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

	// Generic fallback: any engine with MatMul
	if s.fwd == nil {
		log.Printf("[serve] using generic inference (weights streamed, CPU attention)")
		normEps := float32(1e-6)
		if v, ok := cfg["rms_norm_eps"].(float64); ok {
			normEps = float32(v)
		}
		dim := s.dim
		nLayers := s.layers
		heads := s.heads
		kvHeads := s.kvHeads
		headDim := dim / heads
		kvDim := kvHeads * headDim
		ffnDim := s.ffnDim
		vocabSize := s.vocabSize
		maxSeq := s.maxSeq
		halfHead := s.halfHead

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
		k := make([]float32, kvDim)
		v := make([]float32, kvDim)
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

		_ = halfHead

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
				mv(k[:kvDim], wk, buf, kvDim, dim)
				mv(v[:kvDim], wv, buf, kvDim, dim)

				applyRoPE(q, k[:kvDim], pos, headDim, 10000.0, heads, kvHeads)

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
// Handlers
// -----------------------------------------------------------------------

// handleChatCompletions implements POST /v1/chat/completions.
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

	// Resolve model — use pre-loaded if matches, otherwise lazy-load
	if err := s.ensureModel(req.Model); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "model_not_found")
		return
	}

	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages array is required", "invalid_request_error")
		return
	}

	maxTokens := 256
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	s.mu.RLock()
	tok := s.tokenizer
	cfg := s.cfg
	s.mu.RUnlock()

	// Build prompt: use last user message as raw text for now
	// ChatML template produces garbage on small models via Metal graph path
	var prompt string
	for _, m := range req.Messages {
		if m.Role == "user" {
			prompt = m.Content
		}
	}
	promptTokens := tok.Encode(prompt)

	if req.Stream {
		s.handleChatStream(w, req, promptTokens, maxTokens)
		return
	}

	// Non-streaming response: serialize inference (shared KV cache)
	s.inferMu.Lock()
	defer s.inferMu.Unlock()

	s.mu.RLock()
	fwd := s.fwd
	resetKV := s.resetKV
	s.mu.RUnlock()

	temp := float32(0.7)
	if req.Temperature != nil {
		temp = float32(*req.Temperature)
	}
	topK := 40

	if resetKV != nil {
		resetKV()
	}

	// Prefill
	var logits []float32
	for i, tid := range promptTokens {
		logits = fwd(tid, i)
	}
	if logits == nil {
		writeError(w, http.StatusInternalServerError, "inference failed — no GPU forward pass available", "server_error")
		return
	}

	stopTokens := discoverStopTokens(tok, cfg, s.modelDir)

	// Generate
	allTokens := make([]int, len(promptTokens))
	copy(allTokens, promptTokens)
	var genTokens []int
	for step := 0; step < maxTokens; step++ {
		nextToken := sampleTopK(logits, temp, topK)
		allTokens = append(allTokens, nextToken)
		genTokens = append(genTokens, nextToken)
		if stopTokens[nextToken] {
			break
		}
		pos := len(allTokens) - 1
		if pos >= s.maxSeq-1 {
			break
		}
		logits = fwd(allTokens[pos], pos)
		if logits == nil {
			break
		}
	}

	generated := tok.Decode(genTokens)
	if idx := findSpecialToken(generated); idx >= 0 {
		generated = generated[:idx]
	}
	generated = strings.TrimSpace(generated)
	completionTokens := len(genTokens)

	resp := ChatCompletionResponse{
		ID:      generateID("chatcmpl"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.modelName,
		Choices: []ChatChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    "assistant",
					Content: generated,
				},
				FinishReason: "stop",
			},
		},
		Usage: UsageInfo{
			PromptTokens:     len(promptTokens),
			CompletionTokens: completionTokens,
			TotalTokens:      len(promptTokens) + completionTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleChatStream sends SSE chunks for streaming chat completions.
func (s *serveState) handleChatStream(w http.ResponseWriter, req ChatCompletionRequest, promptTokens []int, maxTokens int) {
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

	// First chunk: role
	writeSSEChunk(w, flusher, ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: now,
		Model:   s.modelName,
		Choices: []ChatChunkChoice{
			{
				Index: 0,
				Delta: ChatDelta{Role: "assistant"},
			},
		},
	})

	s.mu.RLock()
	fwd := s.fwd
	tok := s.tokenizer
	streamResetKV := s.resetKV
	s.mu.RUnlock()

	temp := float32(0.7)
	if req.Temperature != nil {
		temp = float32(*req.Temperature)
	}
	topK := 40

	if streamResetKV != nil {
		streamResetKV()
	}

	// Prefill
	var logits []float32
	for i, tid := range promptTokens {
		logits = fwd(tid, i)
	}
	if logits == nil {
		writeSSEChunk(w, flusher, ChatCompletionChunk{
			ID: id, Object: "chat.completion.chunk", Created: now, Model: s.modelName,
			Choices: []ChatChunkChoice{{Index: 0, Delta: ChatDelta{Content: "[error: no GPU forward pass]"}}},
		})
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	// Generate token by token, streaming each
	allTokens := make([]int, len(promptTokens))
	copy(allTokens, promptTokens)
	for step := 0; step < maxTokens; step++ {
		nextToken := sampleTopK(logits, temp, topK)
		allTokens = append(allTokens, nextToken)
		if nextToken == tok.EOS || nextToken == 0 {
			break
		}
		text := tok.Decode([]int{nextToken})
		writeSSEChunk(w, flusher, ChatCompletionChunk{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: now,
			Model:   s.modelName,
			Choices: []ChatChunkChoice{
				{Index: 0, Delta: ChatDelta{Content: text}},
			},
		})

		pos := len(allTokens) - 1
		if pos >= s.maxSeq-1 {
			break
		}
		logits = fwd(allTokens[pos], pos)
		if logits == nil {
			break
		}
	}

	// Final chunk: finish_reason
	stop := "stop"
	writeSSEChunk(w, flusher, ChatCompletionChunk{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: now,
		Model:   s.modelName,
		Choices: []ChatChunkChoice{
			{
				Index:        0,
				Delta:        ChatDelta{},
				FinishReason: &stop,
			},
		},
	})

	// Terminator
	fmt.Fprintf(w, "data: [DONE]\n\n")
	flusher.Flush()
}

// handleCompletions implements POST /v1/completions (legacy).
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
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}
	_ = maxTokens

	s.mu.RLock()
	tok := s.tokenizer
	s.mu.RUnlock()

	promptTokens := tok.Encode(req.Prompt)

	// TODO: Run text completion inference:
	//   1. Tokenize prompt
	//   2. Forward pass through transformer layers
	//   3. Autoregressive generation until maxTokens or stop sequence
	//   4. Decode tokens to text
	//
	// The forward pass is identical to chat completions — the difference is
	// that legacy completions don't apply a chat template to the prompt.

	generated := "[mongoose] completion inference stub"
	completionTokens := len(tok.Encode(generated))

	if req.Stream {
		// TODO: SSE streaming for legacy completions
		// Same pattern as chat streaming but with CompletionChunk format
		flusher, ok := w.(http.Flusher)
		if !ok {
			writeError(w, http.StatusInternalServerError, "streaming not supported", "server_error")
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		chunk := map[string]interface{}{
			"id":      generateID("cmpl"),
			"object":  "text_completion",
			"created": time.Now().Unix(),
			"model":   s.modelName,
			"choices": []map[string]interface{}{
				{"text": generated, "index": 0, "finish_reason": "stop"},
			},
		}
		data, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", data)
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}

	resp := CompletionResponse{
		ID:      generateID("cmpl"),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   s.modelName,
		Choices: []CompletionChoice{
			{
				Text:         generated,
				Index:        0,
				FinishReason: "stop",
			},
		},
		Usage: UsageInfo{
			PromptTokens:     len(promptTokens),
			CompletionTokens: completionTokens,
			TotalTokens:      len(promptTokens) + completionTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleEmbeddings implements POST /v1/embeddings.
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

	// Normalize input to []string
	var inputs []string
	switch v := req.Input.(type) {
	case string:
		inputs = []string{v}
	case []interface{}:
		for _, item := range v {
			if str, ok := item.(string); ok {
				inputs = append(inputs, str)
			}
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

		// TODO: Generate embeddings:
		//   1. Tokenize input text
		//   2. Forward pass through all transformer layers (no generation)
		//   3. Pool the hidden states: mean-pool over sequence positions,
		//      or use the last token's hidden state (model-dependent)
		//   4. Normalize the embedding vector (L2 norm)
		//
		// The forward pass is the same as inference but stops after the
		// final layer norm — no lm_head projection, no sampling.
		//
		// For dedicated embedding models (e.g., nomic-embed, bge, gte),
		// there's often a separate pooling head to load.

		// Stub: return zero vector of model dimension
		embedding := make([]float32, s.dim)

		data = append(data, EmbeddingData{
			Object:    "embedding",
			Embedding: embedding,
			Index:     i,
		})
	}

	resp := EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  s.modelName,
		Usage: UsageInfo{
			PromptTokens: totalTokens,
			TotalTokens:  totalTokens,
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleModels implements GET /v1/models.
func (s *serveState) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error")
		return
	}

	var models []ModelInfo

	// List all downloaded models from ~/.mongoose/models/
	entries, _ := os.ReadDir(modelsDir)
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		info, _ := e.Info()
		created := int64(0)
		if info != nil {
			created = info.ModTime().Unix()
		}
		models = append(models, ModelInfo{
			ID:      e.Name(),
			Object:  "model",
			Created: created,
			OwnedBy: "mongoose",
		})
	}

	// If a model is loaded but not in the models dir (e.g., direct path), include it
	s.mu.RLock()
	if s.modelName != "" {
		found := false
		for _, m := range models {
			if m.ID == s.modelName {
				found = true
				break
			}
		}
		if !found {
			models = append(models, ModelInfo{
				ID:      s.modelName,
				Object:  "model",
				Created: time.Now().Unix(),
				OwnedBy: "mongoose",
			})
		}
	}
	s.mu.RUnlock()

	resp := ModelsResponse{
		Object: "list",
		Data:   models,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// handleHealth implements GET /health.
func (s *serveState) handleHealth(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	modelLoaded := s.modelName != ""
	modelName := s.modelName
	hasGPU := s.eng != nil
	s.mu.RUnlock()

	status := map[string]interface{}{
		"status":       "ok",
		"model_loaded": modelLoaded,
		"model":        modelName,
		"gpu":          hasGPU,
		"version":      "ai v1.1.1",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// ensureModel checks that the requested model is loaded, loading it if needed.
func (s *serveState) ensureModel(requestedModel string) error {
	if requestedModel == "" {
		s.mu.RLock()
		loaded := s.modelName != ""
		s.mu.RUnlock()
		if !loaded {
			return fmt.Errorf("no model specified and none pre-loaded")
		}
		return nil
	}

	s.mu.RLock()
	current := s.modelName
	s.mu.RUnlock()

	// If already loaded, use it (treat any non-empty loaded model as match
	// when the request model is a substring — handles "llama" matching "ReluLLaMA-7B")
	if current != "" && (current == requestedModel || strings.Contains(strings.ToLower(current), strings.ToLower(requestedModel))) {
		return nil
	}

	// TODO: Implement lazy model loading on first request.
	// This would resolve the model name, load tokenizer + weights,
	// and swap out the current model. Need to handle concurrent requests
	// during model swap (queue them, don't drop them).

	if current == "" {
		return fmt.Errorf("model %q not loaded — start server with: ai serve %s", requestedModel, requestedModel)
	}

	// For now, use whatever model is loaded
	return nil
}

// formatChatPrompt concatenates chat messages into a single prompt string.
// TODO: Replace with proper chat template rendering from tokenizer_config.json.
// Different models use different templates:
//   - ChatML:    <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n
//   - Llama:     [INST] <<SYS>>...<</SYS>> ... [/INST]
//   - Mistral:   [INST] ... [/INST]
//   - Qwen:      <|im_start|>system\n...<|im_end|>\n...
func formatChatPrompt(messages []ChatMessage) string {
	var b strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			b.WriteString(msg.Content)
			b.WriteString("\n\n")
		case "user":
			b.WriteString("User: ")
			b.WriteString(msg.Content)
			b.WriteString("\n")
		case "assistant":
			b.WriteString("Assistant: ")
			b.WriteString(msg.Content)
			b.WriteString("\n")
		}
	}
	b.WriteString("Assistant: ")
	return b.String()
}

// generateID creates a unique ID with the given prefix.
func generateID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

// writeError sends an OpenAI-format error response.
func writeError(w http.ResponseWriter, status int, message, errType string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(ErrorResponse{
		Error: ErrorDetail{
			Message: message,
			Type:    errType,
		},
	})
}

// writeSSEChunk marshals a chunk and writes it as an SSE event.
func writeSSEChunk(w http.ResponseWriter, flusher http.Flusher, chunk ChatCompletionChunk) {
	data, err := json.Marshal(chunk)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

// Silence unused import warnings. These are used in the handler implementations.
var _ = bufio.NewScanner
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
	if err != nil {
		log.Fatalf("cannot find self: %v", err)
	}

	// Kill existing daemon if running
	stopExistingDaemon()

	args := []string{"serve"}
	if modelName != "" {
		args = append(args, fmt.Sprintf("model=%s", modelName))
	}
	args = append(args, fmt.Sprintf("host=%s", host), fmt.Sprintf("port=%s", port))

	cmd := exec.Command(exe, args...)
	setSysProcAttr(cmd)
	cmd.Stdout = nil
	cmd.Stderr = nil
	cmd.Stdin = nil

	if err := cmd.Start(); err != nil {
		log.Fatalf("failed to start daemon: %v", err)
	}

	pid := cmd.Process.Pid

	home, _ := os.UserHomeDir()
	os.MkdirAll(filepath.Join(home, ".ai"), 0755)
	os.WriteFile(servePidPath(), []byte(strconv.Itoa(pid)), 0644)

	fmt.Printf("ai serve daemon started (pid=%d) on %s:%s\n", pid, host, port)
	if modelName != "" {
		fmt.Printf("  model: %s\n", modelName)
	}
	fmt.Printf("  pid:   %s\n", servePidPath())
	fmt.Printf("  stop:  ai serve --stop\n")
}

func stopExistingDaemon() {
	data, err := os.ReadFile(servePidPath())
	if err != nil {
		return
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		os.Remove(servePidPath())
		return
	}
	proc, err := os.FindProcess(pid)
	if err != nil {
		os.Remove(servePidPath())
		return
	}
	if err := proc.Signal(os.Signal(os.Interrupt)); err != nil {
		os.Remove(servePidPath())
		return
	}
	proc.Signal(os.Interrupt)
	time.Sleep(500 * time.Millisecond)
	proc.Kill()
	os.Remove(servePidPath())
}
