package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestAutoDetectWithConfig(t *testing.T) {
	dir := t.TempDir()

	cfg := map[string]interface{}{
		"architectures":         []interface{}{"LlamaForCausalLM"},
		"hidden_size":           2048.0,
		"num_hidden_layers":     22.0,
		"num_attention_heads":   32.0,
		"num_key_value_heads":   4.0,
		"intermediate_size":     5632.0,
		"vocab_size":            32000.0,
		"rope_theta":            10000.0,
		"rms_norm_eps":          1e-5,
		"max_position_embeddings": 2048.0,
	}
	data, _ := json.Marshal(cfg)
	os.WriteFile(filepath.Join(dir, "config.json"), data, 0644)

	p := AutoDetect(dir)

	if p.Dim != 2048 {
		t.Errorf("Dim = %d, want 2048", p.Dim)
	}
	if p.Layers != 22 {
		t.Errorf("Layers = %d, want 22", p.Layers)
	}
	if p.Heads != 32 {
		t.Errorf("Heads = %d, want 32", p.Heads)
	}
	if p.KVHeads != 4 {
		t.Errorf("KVHeads = %d, want 4", p.KVHeads)
	}
	if p.FFNDim != 5632 {
		t.Errorf("FFNDim = %d, want 5632", p.FFNDim)
	}
	if p.VocabSize != 32000 {
		t.Errorf("VocabSize = %d, want 32000", p.VocabSize)
	}
	if p.Family != "llama" {
		t.Errorf("Family = %q, want llama", p.Family)
	}
	if p.HeadDim != 64 {
		t.Errorf("HeadDim = %d, want 64", p.HeadDim)
	}
	if p.KVDim != 256 {
		t.Errorf("KVDim = %d, want 256", p.KVDim)
	}
}

func TestAutoDetectNoConfig(t *testing.T) {
	dir := t.TempDir()
	p := AutoDetect(dir)
	if p.Dim != 0 {
		t.Errorf("no config: Dim = %d, want 0", p.Dim)
	}
}

func TestDetectFamily(t *testing.T) {
	tests := map[string]string{
		"Qwen2ForCausalLM":     "qwen2",
		"LlamaForCausalLM":     "llama",
		"MistralForCausalLM":   "mistral",
		"PhiForCausalLM":       "phi",
		"GemmaForCausalLM":     "gemma",
		"GPT2LMHeadModel":      "gpt2",
		"DeepSeekV2ForCausalLM": "deepseek",
		"UnknownModel":          "unknownmodel",
	}
	for arch, want := range tests {
		got := detectFamily(arch)
		if got != want {
			t.Errorf("detectFamily(%q) = %q, want %q", arch, got, want)
		}
	}
}

func TestCountParams(t *testing.T) {
	p := &ModelProfile{
		Dim: 128, Layers: 4, Heads: 4, KVHeads: 2,
		FFNDim: 256, VocabSize: 256, HeadDim: 32, KVDim: 64,
	}
	n := countParams(p)
	if n <= 0 {
		t.Errorf("countParams returned %d, want > 0", n)
	}
	// Rough check: 128-dim, 4-layer model should be ~500K-1M params
	if n < 100000 || n > 5000000 {
		t.Errorf("countParams = %d, expected ~500K for dim=128 layers=4", n)
	}
}

func TestToTrainCfg(t *testing.T) {
	p := &ModelProfile{
		Dim: 2048, Layers: 22, Heads: 32, KVHeads: 4,
		FFNDim: 5632, VocabSize: 32000, HeadDim: 64, KVDim: 256,
		RopeTheta: 10000.0, ParamCount: 1100000000,
	}
	p.computeRecommendations()
	cfg := p.ToTrainCfg()

	if cfg.dim != 2048 {
		t.Errorf("cfg.dim = %d, want 2048", cfg.dim)
	}
	if cfg.nLayers != 22 {
		t.Errorf("cfg.nLayers = %d, want 22", cfg.nLayers)
	}
	if cfg.beta1 != 0.9 {
		t.Errorf("cfg.beta1 = %f, want 0.9", cfg.beta1)
	}
}
