package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

func modelAvailable() bool {
	path := resolveModel("Qwen2.5-0.5B")
	return path != ""
}

func serveOn(t *testing.T, port int) func() {
	t.Helper()
	state := &serveState{}
	if err := state.loadModel("Qwen2.5-0.5B"); err != nil {
		t.Fatalf("loadModel: %v", err)
	}
	state.inferQueue = make(chan *inferRequest, 16)
	go state.inferWorker()

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/completions", state.handleCompletions)
	mux.HandleFunc("/v1/chat/completions", state.handleChatCompletions)
	mux.HandleFunc("/health", state.handleHealth)

	addr := fmt.Sprintf("127.0.0.1:%d", port)
	srv := &http.Server{Addr: addr, Handler: mux}
	go srv.ListenAndServe()

	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		resp, err := http.Get("http://" + addr + "/health")
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				return func() { srv.Close() }
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatal("server did not become ready in 30s")
	return nil
}

func postJSON(t *testing.T, url, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	return resp
}

func TestIntegrationCompletion(t *testing.T) {
	if !modelAvailable() {
		t.Skip("Qwen2.5-0.5B not downloaded")
	}
	cleanup := serveOn(t, 19200)
	defer cleanup()

	resp := postJSON(t, "http://127.0.0.1:19200/v1/completions",
		`{"model":"Qwen2.5-0.5B","prompt":"The capital of France is","max_tokens":10}`)
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Fatalf("status %d", resp.StatusCode)
	}

	var cr CompletionResponse
	json.NewDecoder(resp.Body).Decode(&cr)

	if len(cr.Choices) == 0 {
		t.Fatal("no choices returned")
	}
	if cr.Usage.CompletionTokens == 0 {
		t.Fatal("zero completion tokens")
	}
	if cr.Usage.TotalTokens != cr.Usage.PromptTokens+cr.Usage.CompletionTokens {
		t.Errorf("token accounting: prompt=%d + comp=%d != total=%d",
			cr.Usage.PromptTokens, cr.Usage.CompletionTokens, cr.Usage.TotalTokens)
	}
	t.Logf("OK: %d tokens: %q", cr.Usage.CompletionTokens, cr.Choices[0].Text)
}

func TestIntegrationChat(t *testing.T) {
	if !modelAvailable() {
		t.Skip("Qwen2.5-0.5B not downloaded")
	}
	cleanup := serveOn(t, 19201)
	defer cleanup()

	resp := postJSON(t, "http://127.0.0.1:19201/v1/chat/completions",
		`{"model":"Qwen2.5-0.5B","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":10}`)
	defer resp.Body.Close()

	var cr ChatCompletionResponse
	json.NewDecoder(resp.Body).Decode(&cr)

	if len(cr.Choices) == 0 {
		t.Fatal("no choices")
	}
	content := cr.Choices[0].Message.Content
	if len(strings.TrimSpace(content)) == 0 {
		t.Error("empty response")
	}
	if strings.Contains(content, "4") {
		t.Logf("OK (correct): %q", content)
	} else {
		t.Logf("OK (generated, not exact): %q", content)
	}
}

func TestIntegrationThroughput(t *testing.T) {
	if !modelAvailable() {
		t.Skip("Qwen2.5-0.5B not downloaded")
	}
	cleanup := serveOn(t, 19202)
	defer cleanup()

	// Warm up
	resp := postJSON(t, "http://127.0.0.1:19202/v1/completions",
		`{"model":"Qwen2.5-0.5B","prompt":"Hello","max_tokens":5}`)
	resp.Body.Close()

	// Benchmark: 100 tokens
	start := time.Now()
	resp = postJSON(t, "http://127.0.0.1:19202/v1/completions",
		`{"model":"Qwen2.5-0.5B","prompt":"Once upon a time there was","max_tokens":100}`)
	defer resp.Body.Close()
	elapsed := time.Since(start)

	var cr CompletionResponse
	json.NewDecoder(resp.Body).Decode(&cr)

	tokens := cr.Usage.CompletionTokens
	if tokens == 0 {
		t.Fatal("zero tokens generated")
	}
	tokPerSec := float64(tokens) / elapsed.Seconds()
	t.Logf("OK: %d tokens in %v (%.1f tok/s)", tokens, elapsed.Round(time.Millisecond), tokPerSec)

	if tokPerSec < 50 {
		t.Errorf("throughput %.1f tok/s below minimum 50 tok/s", tokPerSec)
	}
}

func TestIntegrationSequentialRequests(t *testing.T) {
	if !modelAvailable() {
		t.Skip("Qwen2.5-0.5B not downloaded")
	}
	cleanup := serveOn(t, 19203)
	defer cleanup()

	for i := 0; i < 5; i++ {
		body := fmt.Sprintf(`{"model":"Qwen2.5-0.5B","prompt":"Number %d is","max_tokens":10}`, i)
		resp := postJSON(t, "http://127.0.0.1:19203/v1/completions", body)

		var cr CompletionResponse
		json.NewDecoder(resp.Body).Decode(&cr)
		resp.Body.Close()

		if len(cr.Choices) == 0 || cr.Usage.CompletionTokens == 0 {
			t.Fatalf("request %d: empty response", i)
		}
	}
	t.Log("OK: 5 sequential requests completed")
}

func TestIntegrationNoRegression(t *testing.T) {
	if !modelAvailable() {
		t.Skip("Qwen2.5-0.5B not downloaded")
	}
	cleanup := serveOn(t, 19204)
	defer cleanup()

	// Warm up
	resp := postJSON(t, "http://127.0.0.1:19204/v1/completions",
		`{"model":"Qwen2.5-0.5B","prompt":"Hi","max_tokens":5}`)
	resp.Body.Close()

	// Run 3 trials, take the best
	var bestTokPerSec float64
	for trial := 0; trial < 3; trial++ {
		start := time.Now()
		resp = postJSON(t, "http://127.0.0.1:19204/v1/completions",
			`{"model":"Qwen2.5-0.5B","prompt":"The quick brown fox jumps over","max_tokens":200}`)
		elapsed := time.Since(start)

		var cr CompletionResponse
		json.NewDecoder(resp.Body).Decode(&cr)
		resp.Body.Close()

		tokens := cr.Usage.CompletionTokens
		if tokens == 0 {
			continue
		}
		tps := float64(tokens) / elapsed.Seconds()
		if tps > bestTokPerSec {
			bestTokPerSec = tps
		}
		t.Logf("trial %d: %d tokens in %v (%.1f tok/s)", trial, tokens, elapsed.Round(time.Millisecond), tps)
	}

	t.Logf("best: %.1f tok/s", bestTokPerSec)
	if bestTokPerSec < 100 {
		t.Errorf("regression: best %.1f tok/s below 100 tok/s minimum", bestTokPerSec)
	}
}
