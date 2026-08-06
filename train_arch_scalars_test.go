package main

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// The Metal LoRA trainer applied NONE of Granite's four architecture scalars
// while ai serve applied all four. It therefore trained a different function
// than the one being served — silently, with a plausible loss curve, making
// every downstream measurement (SFT gains, GRPO reward, eval numbers)
// uninterpretable.
//
// This is a source-level test rather than a numerical one because the failure
// mode is OMISSION: there is no wrong answer to assert against, only an absent
// multiplication. A numerical test would need a full training run and a
// reference implementation; this catches the regression in milliseconds.
func TestTrainerAppliesAllGraniteArchScalars(t *testing.T) {
	src, err := os.ReadFile("train_finetune_metal.go")
	if err != nil {
		t.Fatalf("read trainer: %v", err)
	}
	s := string(src)

	for _, scalar := range []struct{ field, why string }{
		{"EmbeddingMultiplier", "embeddings are scaled once on entry to the stack"},
		{"AttentionScale", "replaces 1/sqrt(headDim); Granite uses 1/headDim"},
		{"ResidualMultiplier", "scales each block's contribution into the residual"},
		{"LogitsScaling", "DIVIDES the logits before the loss"},
	} {
		if !strings.Contains(s, "archSc."+scalar.field) {
			t.Errorf("trainer never applies %s — %s.\n"+
				"Without it the trainer optimizes a different function than "+
				"ai serve runs, and every downstream measurement is void.",
				scalar.field, scalar.why)
		}
	}
}

// The residual multiplier scales each block's OUTPUT, and a Granite block has
// two: the attention projection and the FFN down-projection. Applying it once
// is a subtle half-fix that still trains the wrong function.
func TestResidualMultiplierAppliedToBothBlockOutputs(t *testing.T) {
	src, _ := os.ReadFile("train_finetune_metal.go")
	n := strings.Count(string(src), "archSc.ResidualMultiplier")
	// One reference in each of the two `if` guards, plus one use inside each.
	if n < 4 {
		t.Errorf("ResidualMultiplier referenced %d times, want >= 4 "+
			"(guard + use, for BOTH the attention and FFN residuals)", n)
	}
}

// logits_scaling DIVIDES despite its name. Granite-4.1 ships 10.0, so applying
// it as a multiply is a 100x error on every logit — which does not crash, it
// just trains against a differently-shaped distribution than the one sampled
// at inference.
func TestLogitsScalingDividesRatherThanMultiplies(t *testing.T) {
	src, _ := os.ReadFile("train_finetune_metal.go")
	s := string(src)

	re := regexp.MustCompile(`ScaleInPlace\([^,]+,\s*([^,]+),`)
	var sawDivide bool
	for _, m := range re.FindAllStringSubmatch(s, -1) {
		if strings.Contains(m[1], "LogitsScaling") {
			if strings.Contains(m[1], "1.0/") || strings.Contains(m[1], "1/") {
				sawDivide = true
			} else {
				t.Errorf("logits scaled by %q — must be the RECIPROCAL, "+
					"logits_scaling divides", m[1])
			}
		}
	}
	if !sawDivide {
		t.Error("no reciprocal application of LogitsScaling found")
	}
}
