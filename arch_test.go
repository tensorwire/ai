package main

import "testing"

// Granite's four multipliers, verbatim from granite-4.1-3b/config.json.
func graniteConfig() map[string]interface{} {
	return map[string]interface{}{
		"embedding_multiplier": 12.0,
		"residual_multiplier":  0.22,
		"attention_multiplier": 0.015625,
		"logits_scaling":       10.0,
		"hidden_size":          2560.0,
		"num_attention_heads":  40.0,
	}
}

func TestGraniteConfigProducesNonIdentityScalars(t *testing.T) {
	a, err := archFromConfig(graniteConfig())
	if err != nil {
		t.Fatalf("archFromConfig: %v", err)
	}
	if a.IsZero() {
		t.Fatal("Granite config produced zero scalars; the model would emit token soup")
	}
	if a.EmbeddingMultiplier != 12.0 {
		t.Errorf("embedding = %v, want 12", a.EmbeddingMultiplier)
	}
	if a.ResidualMultiplier != 0.22 {
		t.Errorf("residual = %v, want 0.22", a.ResidualMultiplier)
	}
	if a.LogitsScaling != 10.0 {
		t.Errorf("logits = %v, want 10", a.LogitsScaling)
	}
}

// The single most dangerous value in this file. Granite's attention_multiplier
// REPLACES 1/sqrt(headDim) and equals 1/headDim. For headDim 64 that is
// 0.015625, not 0.125 — an 8x difference that produces fluent-looking garbage
// rather than an error.
func TestAttentionScaleIsOneOverHeadDimNotSqrt(t *testing.T) {
	a, err := archFromConfig(graniteConfig())
	if err != nil {
		t.Fatalf("archFromConfig: %v", err)
	}

	const headDim = 64
	wantInvHeadDim := float32(1.0 / headDim) // 0.015625
	wantInvSqrt := float32(0.125)            // 1/sqrt(64) — the WRONG value

	if a.AttentionScale != wantInvHeadDim {
		t.Errorf("attention scale = %v, want %v (1/headDim)", a.AttentionScale, wantInvHeadDim)
	}
	if a.AttentionScale == wantInvSqrt {
		t.Error("attention scale is 1/sqrt(headDim); Granite uses 1/headDim")
	}
}

// A Llama-family config has none of these, and must produce the zero value so
// every use site is skipped and the forward pass stays byte-identical.
func TestLlamaConfigProducesZeroScalars(t *testing.T) {
	a, err := archFromConfig(map[string]interface{}{
		"hidden_size":         4096.0,
		"num_attention_heads": 32.0,
		"rope_theta":          10000.0,
	})
	if err != nil {
		t.Fatalf("archFromConfig: %v", err)
	}
	if !a.IsZero() {
		t.Errorf("Llama config produced non-zero scalars: %+v", a)
	}
}

// A config with some but not all multipliers is one we do not understand.
// Guessing the rest yields a model that is subtly wrong rather than obviously
// broken, so it must be an error.
func TestPartialMultiplierSetIsAnError(t *testing.T) {
	cfg := graniteConfig()
	delete(cfg, "logits_scaling")

	if _, err := archFromConfig(cfg); err == nil {
		t.Error("a config with 3 of 4 multipliers was accepted")
	}
}

// The mongoose conversion must carry all four inference scalars through.
// AdamBeta2 is training-only and must stay zero here.
func TestToMongoosePreservesScalars(t *testing.T) {
	a, err := archFromConfig(graniteConfig())
	if err != nil {
		t.Fatalf("archFromConfig: %v", err)
	}
	m := a.toMongoose()

	if m.EmbeddingMultiplier != a.EmbeddingMultiplier ||
		m.ResidualMultiplier != a.ResidualMultiplier ||
		m.AttentionScale != a.AttentionScale ||
		m.LogitsScaling != a.LogitsScaling {
		t.Errorf("conversion lost a scalar: %+v -> %+v", a, m)
	}
	if m.AdamBeta2 != 0 {
		t.Errorf("AdamBeta2 = %v, want 0 (training-only)", m.AdamBeta2)
	}
	if m.IsZero() {
		t.Error("converted params report zero; the graph would skip every scalar")
	}
}
