package main

import (
	"fmt"

	"github.com/tensorwire/mongoose"
)

// Architecture scalar multipliers.
//
// Granite applies four scalars on top of an otherwise Llama-shaped forward
// pass. Its tensor layout is identical to Llama's, so a model run without them
// loads cleanly, runs at full speed, and emits fluent-looking token soup. There
// is no shape mismatch, no NaN, and no error to catch it — the only signal is
// that the output is wrong.
//
// Values come from config.json. Two of the four differ between model sizes, so
// they are read rather than hardcoded.

// archScalars mirrors mongoose.ArchParams but is built from a parsed
// config.json. Keeping the parsing here means mongoose stays a compute library
// with no opinion about HuggingFace config layout.
type archScalars struct {
	EmbeddingMultiplier float32
	ResidualMultiplier  float32
	AttentionScale      float32
	LogitsScaling       float32
}

// IsZero reports whether no scalars were found, i.e. a Llama-family model that
// needs none of this.
func (a archScalars) IsZero() bool {
	return a.EmbeddingMultiplier == 0 && a.ResidualMultiplier == 0 &&
		a.AttentionScale == 0 && a.LogitsScaling == 0
}

// archFromConfig extracts the scalars from a parsed config.json.
//
// Returns the zero value for architectures that do not use them, which is
// exactly what makes it safe to call unconditionally: mongoose treats a zero
// field as "Llama default" and skips the corresponding graph node, leaving the
// forward pass byte-identical.
//
// Granite requires ALL of them. A config carrying some but not all is a config
// we do not understand, and guessing the rest would produce a model that is
// subtly wrong rather than obviously broken — so that is an error.
func archFromConfig(cfg map[string]interface{}) (archScalars, error) {
	var a archScalars

	get := func(key string) (float32, bool) {
		v, ok := cfg[key].(float64)
		return float32(v), ok
	}

	em, hasEM := get("embedding_multiplier")
	rm, hasRM := get("residual_multiplier")
	am, hasAM := get("attention_multiplier")
	ls, hasLS := get("logits_scaling")

	n := 0
	for _, ok := range []bool{hasEM, hasRM, hasAM, hasLS} {
		if ok {
			n++
		}
	}
	if n == 0 {
		return a, nil // Llama-family: no scalars, nothing to apply.
	}
	if n != 4 {
		return a, fmt.Errorf("config has %d of 4 architecture multipliers "+
			"(embedding=%v residual=%v attention=%v logits=%v); a partial set "+
			"would silently train or serve a different function",
			n, hasEM, hasRM, hasAM, hasLS)
	}

	a.EmbeddingMultiplier = em
	a.ResidualMultiplier = rm
	// attention_multiplier REPLACES 1/sqrt(headDim); it does not scale it.
	// Granite-4.1-3b ships 0.015625, which is 1/64 = 1/headDim, NOT
	// 1/sqrt(64) = 0.125. The two are only equal at headDim 1, and using the
	// sqrt form here is the single easiest way to reintroduce token soup.
	a.AttentionScale = am
	// logits_scaling DIVIDES despite the name. Granite-4.1-3b ships 10.0, so
	// multiplying instead would be a 100x error on every logit.
	a.LogitsScaling = ls
	return a, nil
}

// toMongoose converts to the compute library's representation.
//
// AdamB2 is left zero: it is a training-time override and inference ignores it.
func (a archScalars) toMongoose() mongoose.ArchParams {
	return mongoose.ArchParams{
		EmbeddingMultiplier: a.EmbeddingMultiplier,
		ResidualMultiplier:  a.ResidualMultiplier,
		AttentionScale:      a.AttentionScale,
		LogitsScaling:       a.LogitsScaling,
	}
}
