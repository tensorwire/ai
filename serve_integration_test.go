package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

// modelAvailable must never call resolveModel: that helper ends in log.Fatalf,
// which kills the whole test binary rather than returning "". Every test after
// this one in the package then never runs — and go test reports the package as
// FAIL with no indication that most of it was skipped. resolveModelName is the
// non-fatal equivalent.
func modelAvailable() bool {
	return resolveModelName("Qwen2.5-0.5B") != ""
}

func serveOn(t *testing.T, port int) func() {
	t.Helper()
	state := &serveState{noStream: true}
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

func TestIntegrationCoherence(t *testing.T) {
	if !modelAvailable() {
		t.Skip("Qwen2.5-0.5B not downloaded")
	}
	cleanup := serveOn(t, 19205)
	defer cleanup()

	// Warm up — run two requests to ensure streaming finishes and resident path activates
	resp := postJSON(t, "http://127.0.0.1:19205/v1/completions",
		`{"model":"Qwen2.5-0.5B","prompt":"Hello","max_tokens":5}`)
	resp.Body.Close()
	resp = postJSON(t, "http://127.0.0.1:19205/v1/completions",
		`{"model":"Qwen2.5-0.5B","prompt":"Hello","max_tokens":5}`)
	resp.Body.Close()

	// Test 1: continuation produces real language, not garbage
	resp = postJSON(t, "http://127.0.0.1:19205/v1/completions",
		`{"model":"Qwen2.5-0.5B","prompt":"Once upon a time in a small village, there lived a","max_tokens":40}`)
	defer resp.Body.Close()
	var cr CompletionResponse
	json.NewDecoder(resp.Body).Decode(&cr)
	if len(cr.Choices) == 0 {
		t.Fatal("no choices")
	}
	text := cr.Choices[0].Text
	words := strings.Fields(text)
	if len(words) < 10 {
		t.Errorf("too few words for 40 tokens: %d words: %q", len(words), text)
	} else {
		t.Logf("continuation OK (%d words): %q", len(words), text)
	}

	// Test 2: no degenerate repetition (same word repeated 10+ times in a row)
	resp2 := postJSON(t, "http://127.0.0.1:19205/v1/completions",
		`{"model":"Qwen2.5-0.5B","prompt":"The history of computing began with","max_tokens":80}`)
	var cr2 CompletionResponse
	json.NewDecoder(resp2.Body).Decode(&cr2)
	resp2.Body.Close()
	if len(cr2.Choices) == 0 {
		t.Fatal("no choices")
	}
	text2 := cr2.Choices[0].Text
	words2 := strings.Fields(text2)
	if len(words2) < 10 {
		t.Errorf("too few words (%d): %q", len(words2), text2)
	}
	streak := 1
	for i := 1; i < len(words2); i++ {
		if words2[i] == words2[i-1] {
			streak++
			if streak >= 10 {
				t.Errorf("degenerate repetition: %q repeated %d times in: %q", words2[i], streak, text2)
				break
			}
		} else {
			streak = 1
		}
	}
	if streak < 10 {
		t.Logf("no-repeat OK (%d words): %q", len(words2), text2)
	}

	// Test 3: output contains real English words (not token soup / garbage unicode)
	common := []string{"the", "a", "is", "of", "and", "in", "to", "it", "that", "was", "for", "on", "with", "as", "at", "by", "an", "be", "or", "not", "from", "this", "are", "but", "has", "have", "had", "its", "which", "one", "all", "can", "if", "will", "no", "do", "so", "my", "he", "she", "we", "you", "they", "his", "her", "our"}
	commonSet := make(map[string]bool)
	for _, w := range common {
		commonSet[w] = true
	}
	realWords := 0
	for _, w := range words2 {
		if commonSet[strings.ToLower(strings.Trim(w, ".,!?;:'\"()-"))] {
			realWords++
		}
	}
	ratio := float64(realWords) / float64(len(words2))
	if ratio < 0.1 && len(words2) > 10 {
		t.Errorf("token soup: only %.0f%% common English words in %d words: %q", ratio*100, len(words2), text2)
	} else {
		t.Logf("vocabulary OK: %.0f%% common words in %d words", ratio*100, len(words2))
	}
}
