package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/open-ai-org/tokenizer"
)

func TestHealthEndpoint(t *testing.T) {
	state := &serveState{}
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	state.handleHealth(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp["status"] != "ok" {
		t.Errorf("status = %v, want ok", resp["status"])
	}
	if resp["model_loaded"] != false {
		t.Errorf("model_loaded = %v, want false", resp["model_loaded"])
	}
}

func TestModelsEndpointEmpty(t *testing.T) {
	state := &serveState{}
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()

	state.handleModels(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var resp ModelsResponse
	json.Unmarshal(w.Body.Bytes(), &resp)

	if resp.Object != "list" {
		t.Errorf("object = %q, want list", resp.Object)
	}
}

func TestChatCompletionsNoModel(t *testing.T) {
	state := &serveState{}
	body := `{"model":"test","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	w := httptest.NewRecorder()

	state.handleChatCompletions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

func TestChatCompletionsMethodNotAllowed(t *testing.T) {
	state := &serveState{}
	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	state.handleChatCompletions(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", w.Code)
	}
}

func TestCompletionsNoModel(t *testing.T) {
	state := &serveState{}
	body := `{"model":"test","prompt":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/completions", bytes.NewBufferString(body))
	w := httptest.NewRecorder()

	state.handleCompletions(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

func TestEmbeddingsNoModel(t *testing.T) {
	state := &serveState{}
	body := `{"model":"test","input":"hello"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewBufferString(body))
	w := httptest.NewRecorder()

	state.handleEmbeddings(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", w.Code)
	}
}

func TestApplyTemplateViaServeState(t *testing.T) {
	modelDir := filepath.Join(os.Getenv("HOME"), ".mongoose", "models", "Qwen2.5-0.5B")
	if _, err := os.Stat(filepath.Join(modelDir, "tokenizer.json")); err != nil {
		t.Skip("Qwen2.5-0.5B not available for template test")
	}
	tok, err := tokenizer.LoadTokenizer(modelDir)
	if err != nil {
		t.Skipf("tokenizer load failed: %v", err)
	}
	state := &serveState{tokenizer: tok, cfg: map[string]interface{}{}}

	msgs := []ChatMessage{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "Hi"},
	}
	tokens := state.applyTemplate(msgs)
	if len(tokens) == 0 {
		t.Error("applyTemplate returned no tokens")
	}
}

func TestGenerateID(t *testing.T) {
	id1 := generateID("chatcmpl")
	if len(id1) < 10 {
		t.Errorf("generateID too short: %q", id1)
	}
	if !strings.HasPrefix(id1, "chatcmpl-") {
		t.Errorf("generateID missing prefix: %q", id1)
	}
}

func TestWriteError(t *testing.T) {
	w := httptest.NewRecorder()
	writeError(w, 422, "bad input", "invalid_request_error")

	if w.Code != 422 {
		t.Fatalf("status = %d, want 422", w.Code)
	}

	var resp ErrorResponse
	json.Unmarshal(w.Body.Bytes(), &resp)
	if resp.Error.Message != "bad input" {
		t.Errorf("error message = %q, want 'bad input'", resp.Error.Message)
	}
	if resp.Error.Type != "invalid_request_error" {
		t.Errorf("error type = %q, want 'invalid_request_error'", resp.Error.Type)
	}
}
