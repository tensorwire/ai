package main

import (
	"sync"
	"testing"
)

// newPrefixCacheTestHarness builds just enough of metalStreamingInference to
// exercise slot and prefix-cache bookkeeping. The GPU path is not involved:
// these are the pure decisions about which tokens to re-prefill, which is
// exactly where a subtle bug would silently serve the wrong KV cache.
func newPrefixCacheTestHarness(nSlots int) *metalStreamingInference {
	return &metalStreamingInference{
		nSlots:     nSlots,
		slotMu:     make([]sync.Mutex, nSlots),
		seqSlot:    -1,
		slotTokens: make([][]int, nSlots),
	}
}

// The bug this replaced: acquireSlot() was called once per TOKEN, so
// consecutive tokens of one sequence landed in alternating KV caches. A
// sequence must observe one slot for its entire life.
func TestSlotIsStableAcrossASequence(t *testing.T) {
	mi := newPrefixCacheTestHarness(4)

	mi.beginSequence()
	first := mi.currentSlot()
	for i := 0; i < 50; i++ {
		if got := mi.currentSlot(); got != first {
			t.Fatalf("slot changed mid-sequence at token %d: %d -> %d", i, first, got)
		}
	}
	mi.endSequence()
}

// Distinct sequences should still spread across slots, or concurrency is lost.
func TestSequencesRotateAcrossSlots(t *testing.T) {
	mi := newPrefixCacheTestHarness(2)

	mi.beginSequence()
	a := mi.currentSlot()
	mi.endSequence()

	mi.beginSequence()
	b := mi.currentSlot()
	mi.endSequence()

	if a == b {
		t.Errorf("consecutive sequences reused slot %d; slots are not rotating", a)
	}
}

func TestCachedPrefixLenOnEmptyCache(t *testing.T) {
	mi := newPrefixCacheTestHarness(1)
	mi.beginSequence()
	defer mi.endSequence()

	if n := mi.cachedPrefixLen([]int{1, 2, 3}); n != 0 {
		t.Errorf("empty cache reported %d cached tokens", n)
	}
}

// The case that motivates the whole mechanism: turn 2 of a chat resends the
// system prompt and turn 1 verbatim, so nearly the entire prompt is cached.
func TestCachedPrefixLenReusesSharedPrefix(t *testing.T) {
	mi := newPrefixCacheTestHarness(1)
	mi.beginSequence()
	defer mi.endSequence()

	turn1 := []int{100, 101, 102, 103}
	mi.noteSequence(turn1)

	turn2 := []int{100, 101, 102, 103, 200, 201}
	got := mi.cachedPrefixLen(turn2)
	if got != 4 {
		t.Errorf("cached prefix = %d, want 4", got)
	}
}

// The last cached position must be re-run: stepping it is what produces the
// logits used to sample the next token. Returning len(tokens) would skip the
// forward pass entirely and leave logits nil.
func TestCachedPrefixLenLeavesAtLeastOneTokenToRun(t *testing.T) {
	mi := newPrefixCacheTestHarness(1)
	mi.beginSequence()
	defer mi.endSequence()

	seq := []int{7, 8, 9}
	mi.noteSequence(seq)

	if n := mi.cachedPrefixLen(seq); n != len(seq)-1 {
		t.Errorf("identical resend cached %d of %d tokens; must leave one to run",
			n, len(seq))
	}
}

// A divergence must truncate the reuse at the first differing token, not
// anywhere later — otherwise the model attends to a prefix that was never
// actually computed for this conversation.
func TestCachedPrefixLenStopsAtFirstDivergence(t *testing.T) {
	mi := newPrefixCacheTestHarness(1)
	mi.beginSequence()
	defer mi.endSequence()

	mi.noteSequence([]int{1, 2, 3, 4, 5})

	got := mi.cachedPrefixLen([]int{1, 2, 99, 4, 5})
	if got != 2 {
		t.Errorf("cached prefix = %d, want 2 (diverges at index 2)", got)
	}
}

// A completely different conversation must share nothing.
func TestCachedPrefixLenRejectsUnrelatedSequence(t *testing.T) {
	mi := newPrefixCacheTestHarness(1)
	mi.beginSequence()
	defer mi.endSequence()

	mi.noteSequence([]int{1, 2, 3})
	if n := mi.cachedPrefixLen([]int{9, 8, 7}); n != 0 {
		t.Errorf("unrelated sequence reported %d cached tokens", n)
	}
}

// A shorter follow-up must not read past the end of the new token slice.
func TestCachedPrefixLenHandlesShorterFollowUp(t *testing.T) {
	mi := newPrefixCacheTestHarness(1)
	mi.beginSequence()
	defer mi.endSequence()

	mi.noteSequence([]int{1, 2, 3, 4, 5, 6})
	if n := mi.cachedPrefixLen([]int{1, 2}); n != 1 {
		t.Errorf("cached prefix = %d, want 1", n)
	}
}

// Caches must not leak between slots: two conversations on different slots are
// independent, and reading the wrong one is a correctness and privacy failure.
func TestSlotCachesAreIndependent(t *testing.T) {
	mi := newPrefixCacheTestHarness(2)

	mi.beginSequence()
	slotA := mi.currentSlot()
	mi.noteSequence([]int{1, 2, 3, 4})
	mi.endSequence()

	mi.beginSequence()
	slotB := mi.currentSlot()
	if slotA == slotB {
		t.Skip("slots did not rotate; covered by TestSequencesRotateAcrossSlots")
	}
	// Slot B has never seen these tokens, even though slot A has.
	if n := mi.cachedPrefixLen([]int{1, 2, 3, 4, 5}); n != 0 {
		t.Errorf("slot %d reported %d cached tokens from slot %d's cache",
			slotB, n, slotA)
	}
	mi.endSequence()
}

// noteSequence must copy: the caller owns the slice and keeps appending
// generated tokens to it, which would otherwise mutate the cache record and
// make it describe a sequence the KV cache never held.
func TestNoteSequenceCopiesTokens(t *testing.T) {
	mi := newPrefixCacheTestHarness(1)
	mi.beginSequence()
	defer mi.endSequence()

	tokens := make([]int, 3, 8)
	copy(tokens, []int{1, 2, 3})
	mi.noteSequence(tokens)

	tokens[0] = 999 // caller mutates its own slice

	if n := mi.cachedPrefixLen([]int{1, 2, 3, 4}); n != 3 {
		t.Errorf("cached prefix = %d, want 3; noteSequence aliased the caller's slice", n)
	}
}

func TestInvalidateSlotDropsTheCache(t *testing.T) {
	mi := newPrefixCacheTestHarness(1)
	mi.beginSequence()
	defer mi.endSequence()

	mi.noteSequence([]int{1, 2, 3})
	mi.invalidateSlot()

	if n := mi.cachedPrefixLen([]int{1, 2, 3, 4}); n != 0 {
		t.Errorf("cache survived invalidation: %d tokens", n)
	}
}
