package main

import (
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/mongoose"
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

	layersReady atomic.Int32 // how many layers have been loaded
	allReady    atomic.Bool  // true once finalNorm + lmHead are also loaded

	fHidden [][]float32 // per-slot hidden buffer
	fLogits [][]float32 // per-slot logits buffer
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
	ret := metal.BuildFused(s.dim, s.kvHeads*headDim, headDim, s.heads, s.kvHeads, s.ffnDim, s.vocabSize, s.layers, s.maxSeq, ropeTheta, 1e-6)
	if ret != 0 {
		return nil
	}

	nSlots := metal.FusedNumSlots()
	if nSlots < 1 {
		nSlots = 1
	}

	mi := &metalStreamingInference{
		metal:   metal,
		s:       s,
		nSlots:  nSlots,
		slotMu:  make([]sync.Mutex, nSlots),
		fHidden: make([][]float32, nSlots),
		fLogits: make([][]float32, nSlots),
	}

	for i := 0; i < nSlots; i++ {
		mi.fHidden[i] = make([]float32, s.dim)
		mi.fLogits[i] = make([]float32, s.vocabSize)
	}

	// Stream weights in background — inference can start once allReady is true
	go mi.loadWeights(st, lmHeadData)

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

// forward runs one token through all layers on the given slot.
// Blocks until all weights are loaded if called during streaming load.
func (mi *metalStreamingInference) forward(slot int, tokenID, pos int) []float32 {
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
func (mi *metalStreamingInference) acquireSlot() int {
	slot := int(mi.nextSlot.Add(1)-1) % mi.nSlots
	mi.slotMu[slot].Lock()
	return slot
}

func (mi *metalStreamingInference) releaseSlot(slot int) {
	mi.slotMu[slot].Unlock()
}

func (mi *metalStreamingInference) resetKV(slot int) {
	mi.metal.FusedResetKVSlot(slot)
}

func (mi *metalStreamingInference) ready() bool {
	return mi.allReady.Load()
}
