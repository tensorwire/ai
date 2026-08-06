package main

import (
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/mongoose"
)

// metalStreamingInference provides streaming weight load + multi-slot inference.
//
// Weight streaming: layers load in the background. Inference starts as soon as
// all layers are loaded — but BuildFused + buffer allocation happens immediately,
// so time-to-first-token is bounded by weight I/O, not GPU setup.
//
// Multi-slot: each inference request gets a slot (round-robin). Slots have
// independent KV caches and scratch buffers, running on separate Metal command
// queues. Up to FusedNumSlots() concurrent requests.
type metalStreamingInference struct {
	metal    *mongoose.Metal
	s        *serveState
	nSlots   int
	slotMu   []sync.Mutex
	nextSlot atomic.Uint32

	// A sequence holds one slot for its whole lifetime; seqMu serializes
	// requests so seqSlot unambiguously names the in-flight sequence's slot.
	seqMu   sync.Mutex
	seqSlot int

	// slotTokens[i] is the exact token sequence resident in slot i's KV cache,
	// used to compute a reusable prefix. Stored as the token IDs themselves,
	// never a digest — see cachedPrefixLen.
	slotTokens [][]int

	layersReady atomic.Int32 // how many layers have been loaded
	allReady    atomic.Bool  // true once finalNorm + lmHead are also loaded

	fHidden   [][]float32            // per-slot hidden buffer
	fLogits   [][]float32            // per-slot logits buffer
	streamFwd *metalStreamingForward // ping-pong weight streaming (used before resident weights are loaded)
}

func buildMetalStreamingInference(s *serveState, st *gguf.SafeTensors, lmHeadData []float32) *metalStreamingInference {
	metal, ok := s.eng.(*mongoose.Metal)
	if !ok {
		return nil
	}

	headDim := s.dim / s.heads
	ropeTheta := 10000.0
	if v, ok := s.cfg["rope_theta"].(float64); ok {
		ropeTheta = v
	}
	// Architecture scalars must be set BEFORE BuildFused. Without them Granite
	// loads, runs at full speed, and emits token soup — which is exactly what
	// `ai serve` did while `ai infer` was correct, because only the infer path
	// had been wired.
	if arch, err := archFromConfig(s.cfg); err != nil {
		log.Printf("[serve] arch scalars: %v", err)
		return nil
	} else if !arch.IsZero() {
		if rc := metal.FusedSetArch(arch.toMongoose()); rc != 0 {
			log.Printf("[serve] arch scalars: FusedSetArch failed (%d)", rc)
			return nil
		}
		log.Printf("[serve] arch scalars: embed=%.4g residual=%.4g attn=%.6g logits=%.4g",
			arch.EmbeddingMultiplier, arch.ResidualMultiplier,
			arch.AttentionScale, arch.LogitsScaling)
	}

	ret := metal.BuildFused(s.dim, s.kvHeads*headDim, headDim, s.heads, s.kvHeads, s.ffnDim, s.vocabSize, s.layers, s.maxSeq, ropeTheta, 1e-6)
	if ret != 0 {
		return nil
	}

	nSlots := metal.FusedNumSlots()
	if nSlots < 1 {
		nSlots = 1
	}

	mi := &metalStreamingInference{
		metal:  metal,
		s:      s,
		nSlots: nSlots,
		slotMu: make([]sync.Mutex, nSlots),
		// -1 means "no sequence in flight"; currentSlot falls back to 0 so a
		// stray call cannot index out of range.
		seqSlot:    -1,
		slotTokens: make([][]int, nSlots),
		fHidden:    make([][]float32, nSlots),
		fLogits:    make([][]float32, nSlots),
	}

	for i := 0; i < nSlots; i++ {
		mi.fHidden[i] = make([]float32, s.dim)
		mi.fLogits[i] = make([]float32, s.vocabSize)
	}

	// Try streaming forward (ping-pong weight buffers, 19x less VRAM)
	mi.streamFwd = buildMetalStreamingForward(s, st, lmHeadData)

	// Also load weights into resident buffers in background for fast path
	go mi.loadWeights(st, lmHeadData)

	s.inferSlots = nSlots
	log.Printf("[serve] Metal streaming inference — %d slots, loading weights in background", nSlots)
	return mi
}

func (mi *metalStreamingInference) loadWeights(st *gguf.SafeTensors, lmHeadData []float32) {
	s := mi.s
	headDim := s.dim / s.heads
	kvDim := s.kvHeads * headDim
	wi := 0

	for l := 0; l < s.layers; l++ {
		prefix := fmt.Sprintf("model.layers.%d.", l)
		loadW := func(n string) {
			d, _, _ := st.ReadTensorFloat32(prefix + n)
			if d != nil {
				mi.metal.FusedSetWeight(wi, d)
			}
			wi++
		}
		loadB := func(n string, sz int) {
			d, _, _ := st.ReadTensorFloat32(prefix + n)
			if d == nil {
				d = make([]float32, sz)
			}
			mi.metal.FusedSetWeight(wi, d)
			wi++
		}
		loadW("input_layernorm.weight")
		loadW("self_attn.q_proj.weight")
		loadW("self_attn.k_proj.weight")
		loadW("self_attn.v_proj.weight")
		loadB("self_attn.q_proj.bias", s.dim)
		loadB("self_attn.k_proj.bias", kvDim)
		loadB("self_attn.v_proj.bias", kvDim)
		loadW("self_attn.o_proj.weight")
		loadW("post_attention_layernorm.weight")
		loadW("mlp.gate_proj.weight")
		loadW("mlp.up_proj.weight")
		loadW("mlp.down_proj.weight")

		mi.layersReady.Add(1)
		if l%4 == 3 || l == s.layers-1 {
			log.Printf("[serve] loaded layer %d/%d", l+1, s.layers)
		}
	}

	fnorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
	mi.metal.FusedSetWeight(wi, fnorm)
	wi++
	mi.metal.FusedSetWeight(wi, lmHeadData)
	wi++

	mi.allReady.Store(true)
	log.Printf("[serve] all %d weights loaded — inference ready", wi)
}

// forward runs one token. Uses streaming (ping-pong) path while weights are
// loading, switches to resident (monolithic command buffer) once all weights
// are in VRAM.
func (mi *metalStreamingInference) forward(slot int, tokenID, pos int) []float32 {
	if !mi.allReady.Load() && mi.streamFwd != nil {
		return mi.streamFwd.forward(tokenID, pos)
	}

	for !mi.allReady.Load() {
		time.Sleep(time.Millisecond)
	}

	s := mi.s
	tokOff := tokenID * s.dim
	if tokenID < 0 || tokOff+s.dim > len(s.embedData) {
		return nil
	}

	hidden := mi.fHidden[slot]
	logits := mi.fLogits[slot]
	copy(hidden, s.embedData[tokOff:tokOff+s.dim])

	mi.metal.FusedPartialStepSlot(slot, hidden, pos, 0, s.layers, nil, logits)
	return logits
}

// acquireSlot returns a slot index and locks it. Caller must call releaseSlot.
//
// A slot owns a KV cache, so it must be held for a whole SEQUENCE, never per
// token. Calling this once per token round-robins consecutive tokens of one
// conversation across independent caches: token N lands in slot 0 and token N+1
// in slot 1, so each token attends to roughly half its history and never to its
// immediate predecessor.
//
// That produces fluent-looking nonsense rather than an error, in every model —
// Qwen2.5-0.5B emitted "The capital of France is the the sum of the product of
// the company" — and it looks exactly like a bad checkpoint or a broken
// template, which is where the debugging time goes.
//
// Callers: acquire once per request, then step every token on that slot.
func (mi *metalStreamingInference) acquireSlot() int {
	slot := int(mi.nextSlot.Add(1)-1) % mi.nSlots
	mi.slotMu[slot].Lock()
	return slot
}

func (mi *metalStreamingInference) releaseSlot(slot int) {
	mi.slotMu[slot].Unlock()
}

// beginSequence claims a slot for the calling request and holds it until
// endSequence. Every token of the sequence then runs on that one KV cache.
func (mi *metalStreamingInference) beginSequence() {
	mi.seqMu.Lock()
	mi.seqSlot = mi.acquireSlot()
}

// beginSequenceFor claims the slot whose KV cache best matches `tokens`,
// falling back to round-robin when none does.
//
// Round-robin alone defeats the prefix cache for the common case. With two
// slots, consecutive turns of ONE conversation alternate between them, so a
// turn only ever finds its own history every other time — measured as 6.11s,
// 5.94s, 0.07s across three identical requests, where turns 2 and 3 should both
// have been instant.
//
// Slots exist for CONCURRENT sequences; picking by cache affinity keeps that
// property (a second, different conversation still lands elsewhere) while
// letting a repeat turn return to the slot that already holds its prefix.
func (mi *metalStreamingInference) beginSequenceFor(tokens []int) {
	mi.seqMu.Lock()

	best, bestLen := -1, 0
	for i := 0; i < mi.nSlots; i++ {
		if n := commonPrefix(mi.slotTokens[i], tokens); n > bestLen {
			best, bestLen = i, n
		}
	}
	if best >= 0 {
		mi.slotMu[best].Lock()
		mi.seqSlot = best
		return
	}
	mi.seqSlot = mi.acquireSlot()
}

// commonPrefix counts leading tokens shared by a and b, stopping one short of
// len(b) so there is always a token left to run — see cachedPrefixLen.
func commonPrefix(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n >= len(b) {
		n = len(b) - 1
	}
	i := 0
	for i < n && a[i] == b[i] {
		i++
	}
	return i
}

func (mi *metalStreamingInference) endSequence() {
	mi.releaseSlot(mi.seqSlot)
	mi.seqSlot = -1
	mi.seqMu.Unlock()
}

// currentSlot is the slot held by the in-flight sequence.
func (mi *metalStreamingInference) currentSlot() int {
	if mi.seqSlot < 0 {
		return 0
	}
	return mi.seqSlot
}

// cachedPrefixLen returns how many leading tokens of `tokens` are already
// resident in the current slot's KV cache, and therefore need no prefill.
//
// The comparison is element-by-element against the exact token sequence the
// cache was built from. It is deliberately NOT a hash: a collision would serve
// one conversation's KV cache to another, which is a correctness and privacy
// failure that no amount of speed justifies.
//
// This is what makes multi-turn chat fast. Turn 2 of a conversation repeats the
// entire system prompt plus turn 1, so the shared prefix is nearly the whole
// request and only the new user message needs prefilling.
func (mi *metalStreamingInference) cachedPrefixLen(tokens []int) int {
	return commonPrefix(mi.slotTokens[mi.currentSlot()], tokens)
}

// noteSequence records the token sequence now resident in the current slot.
func (mi *metalStreamingInference) noteSequence(tokens []int) {
	slot := mi.currentSlot()
	buf := make([]int, len(tokens))
	copy(buf, tokens)
	mi.slotTokens[slot] = buf
}

// invalidateSlot drops the cached sequence for the current slot, forcing a full
// prefill on the next request.
func (mi *metalStreamingInference) invalidateSlot() {
	mi.slotTokens[mi.currentSlot()] = nil
}

func (mi *metalStreamingInference) resetKV(slot int) {
	mi.metal.FusedResetKVSlot(slot)
}

func (mi *metalStreamingInference) ready() bool {
	return mi.allReady.Load()
}
