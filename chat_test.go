package main

import (
	"strings"
	"testing"

	"github.com/open-ai-org/tokenizer"
)

func TestApplyChatTemplate_ChatML(t *testing.T) {
	tok, err := tokenizer.LoadTokenizer("testdata/qwen")
	if tok == nil || err != nil {
		t.Skip("no qwen test tokenizer available")
	}

	if !tok.HasToken("<|im_start|>") {
		t.Skip("tokenizer doesn't have ChatML tokens")
	}

	messages := []chatMessage{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Hello"},
	}

	tokens := applyChatTemplate(tok, messages, nil)
	if len(tokens) == 0 {
		t.Fatal("empty token output")
	}

	decoded := tok.Decode(tokens)
	if !strings.Contains(decoded, "<|im_start|>system") {
		t.Errorf("missing system tag in: %s", decoded)
	}
	if !strings.Contains(decoded, "<|im_start|>user") {
		t.Errorf("missing user tag in: %s", decoded)
	}
	if !strings.Contains(decoded, "<|im_start|>assistant") {
		t.Errorf("missing assistant tag in: %s", decoded)
	}
}

func TestApplyChatTemplate_GenericFallback(t *testing.T) {
	// Use a tokenizer without ChatML tokens
	tok, err := tokenizer.LoadTokenizer("testdata/gpt2")
	if tok == nil || err != nil {
		t.Skip("no gpt2 test tokenizer available")
	}

	messages := []chatMessage{
		{Role: "system", Content: "Be helpful."},
		{Role: "user", Content: "Hi"},
	}

	tokens := applyChatTemplate(tok, messages, nil)
	if len(tokens) == 0 {
		t.Fatal("empty token output")
	}
}

func TestResolveModelName(t *testing.T) {
	// Test exact name
	path := resolveModelName("Qwen2.5-0.5B")
	if path == "" {
		t.Skip("Qwen2.5-0.5B not downloaded")
	}
	t.Logf("resolved: %s", path)

	// Test case insensitive
	path2 := resolveModelName("qwen2.5-0.5b")
	if path2 == "" {
		t.Error("case-insensitive resolution failed")
	}

	// Note: can't test nonexistent model — resolveModel calls log.Fatal
}

func TestChatMessageFormat(t *testing.T) {
	msg := chatMessage{Role: "user", Content: "hello"}
	if msg.Role != "user" {
		t.Errorf("got role %q, want user", msg.Role)
	}
	if msg.Content != "hello" {
		t.Errorf("got content %q, want hello", msg.Content)
	}
}
