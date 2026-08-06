package main

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

// The per-parameter cost of Helix optimizer state over the hot set has been
// stated wrongly three times — 8, then 10, then finally 12 B/param — each time
// by omitting a buffer that helix_needle actually writes. Every version was
// transcribed from prose rather than derived from the allocation sites, and
// every version was used to decide whether 30B Q8 fits on a 48 GiB machine.
//
// The consequence of the 10 B/param version: a Week-8 gate authorizing
// "Proceed" on a 2-5% measured hot ratio, whose upper half needs ~45 GiB before
// any working set. This test exists so the constant can never drift from the
// code again.
const helixHotBytesPerParam = 4 + 2 + 2 + 4 // grad FP32, mom FP16, vel FP16, delta FP32

func TestHelixPerParamCostMatchesAllocationSites(t *testing.T) {
	src, err := os.ReadFile("train_metal.go")
	if err != nil {
		t.Skipf("read trainer: %v", err)
	}
	s := string(src)

	// momentum and velocity are SEPARATE allocations, each nElems*2 (FP16).
	// Costing them jointly at 2 B/param is the error this test guards.
	for _, name := range []string{"momT", "velT"} {
		re := regexp.MustCompile(name + `\s*:?=\s*mtl\.AllocRaw\(nElems\*2\b`)
		if !re.MatchString(s) {
			t.Errorf("%s is not allocated as nElems*2 (FP16) — the 2 B/param "+
				"assumption for this buffer no longer holds; recompute "+
				"helixHotBytesPerParam from the allocation sites", name)
		}
	}
	if strings.Count(s, "mtl.AllocRaw(nElems*2") < 2 {
		t.Error("fewer than two FP16 nElems*2 allocations — momentum and " +
			"velocity must each have their own")
	}

	if helixHotBytesPerParam != 12 {
		t.Errorf("helixHotBytesPerParam = %d, want 12 "+
			"(grad 4 + mom 2 + vel 2 + delta 4)", helixHotBytesPerParam)
	}
}

// The load-bearing consequence: what hot ratio fits on the target machine.
// Stated as a test so the number cannot be quoted from a stale table.
func TestThirtyBHotRatioCeiling(t *testing.T) {
	const (
		giB          = 1024 * 1024 * 1024
		params       = 30e9
		weightsGiB   = 27.94 // Q8 30B resident
		practicalGiB = 44.0  // 48 GiB machine less OS, Metal driver, server
	)

	stateGiB := func(hot float64) float64 {
		return params * helixHotBytesPerParam * hot / giB
	}

	// 5% was the spec's Proceed threshold. It does not fit, before any working
	// set at all.
	if total := weightsGiB + stateGiB(0.05); total <= practicalGiB {
		t.Errorf("5%% hot totals %.2f GiB and appears to fit — the corrected "+
			"arithmetic says it must not", total)
	}
	// 2% must still fit, or the whole approach is dead rather than tight.
	if total := weightsGiB + stateGiB(0.02); total > practicalGiB {
		t.Errorf("2%% hot totals %.2f GiB and does not fit; 30B Q8 needs a "+
			"different plan entirely", total)
	}

	ceiling := (practicalGiB - weightsGiB) / (params * helixHotBytesPerParam / giB)
	t.Logf("ceiling with a ZERO working set: %.2f%% hot", 100*ceiling)
	if ceiling > 0.05 {
		t.Errorf("ceiling %.2f%% exceeds 5%%; the gate band would be live again", 100*ceiling)
	}
}
