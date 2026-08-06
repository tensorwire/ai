package main

import "testing"

// A single inference worker made the server strictly serial: the KV slots
// existed but nothing ever drove two at once, so a GRPO group of 8 rollouts
// cost 8x wall-clock instead of 4x. Every throughput number measured through
// this server understated the hardware.
func TestInferWorkersTracksSlotCount(t *testing.T) {
	s := &serveState{inferSlots: 2}
	if got := s.inferWorkers(); got != 2 {
		t.Errorf("inferWorkers() = %d with 2 slots, want 2 — the server is "+
			"still serial and the slots are idle", got)
	}
}

// More workers than slots only queues on the slot mutex. Fewer wastes a slot.
func TestInferWorkersNeverExceedsSlots(t *testing.T) {
	for _, slots := range []int{2, 4, 8} {
		s := &serveState{inferSlots: slots}
		if got := s.inferWorkers(); got > slots {
			t.Errorf("%d slots -> %d workers; excess workers only queue", slots, got)
		}
	}
}

// The non-streaming paths have not been audited for concurrent use, so an
// unset or single slot count must stay serial rather than defaulting to
// parallel and corrupting a shared KV cache.
func TestInferWorkersDefaultsToSerialWhenSlotsUnknown(t *testing.T) {
	for _, slots := range []int{0, 1} {
		s := &serveState{inferSlots: slots}
		if got := s.inferWorkers(); got != 1 {
			t.Errorf("inferSlots=%d -> %d workers, want 1 (conservative)", slots, got)
		}
	}
}
