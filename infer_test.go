package main

import (
	"math"
	"testing"
)

func TestSampleTopK_Greedy(t *testing.T) {
	logits := []float32{1.0, 5.0, 3.0, 2.0, 4.0}
	tok := sampleTopK(logits, 0.0, 40)
	if tok != 1 {
		t.Errorf("greedy sample: got %d, want 1 (max logit)", tok)
	}
}

func TestSampleTopK_WithTemp(t *testing.T) {
	logits := make([]float32, 100)
	logits[42] = 100.0 // dominant logit
	tok := sampleTopK(logits, 0.1, 40)
	if tok != 42 {
		t.Errorf("low temp sample with dominant logit: got %d, want 42", tok)
	}
}

func TestApplyRoPE(t *testing.T) {
	headDim := 4
	q := []float32{1, 0, 0, 0, 0, 1, 0, 0} // 2 heads, headDim=4
	k := []float32{1, 0, 0, 0}               // 1 kv head

	qCopy := make([]float32, len(q))
	copy(qCopy, q)

	applyRoPE(q, k, 0, headDim, 10000.0, 2, 1)

	// At pos=0, cos=1 sin=0, so values should be unchanged
	for i := range q {
		if math.Abs(float64(q[i]-qCopy[i])) > 1e-5 {
			t.Errorf("RoPE at pos=0 changed value: q[%d] = %f, want %f", i, q[i], qCopy[i])
		}
	}

	// At pos>0, values should change
	q2 := []float32{1, 0, 0, 0, 0, 1, 0, 0}
	k2 := []float32{1, 0, 0, 0}
	applyRoPE(q2, k2, 5, headDim, 10000.0, 2, 1)

	changed := false
	for i := range q2 {
		if math.Abs(float64(q2[i]-qCopy[i])) > 1e-5 {
			changed = true
		}
	}
	if !changed {
		t.Error("RoPE at pos=5 should change values")
	}
}

func TestSilu(t *testing.T) {
	// silu(0) = 0
	if v := silu(0); math.Abs(float64(v)) > 1e-6 {
		t.Errorf("silu(0) = %f, want 0", v)
	}
	// silu(x) > 0 for x > 0
	if v := silu(5.0); v <= 0 {
		t.Errorf("silu(5) = %f, want > 0", v)
	}
	// silu(x) < 0 for small negative x
	if v := silu(-1.0); v >= 0 {
		t.Errorf("silu(-1) = %f, want < 0", v)
	}
}
