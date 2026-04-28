package main

import (
	"fmt"
	"log"
	"sync"

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/mongoose"
)

// metalStreamingForward provides ping-pong weight-buffered inference.
// Only 2 layers' worth of weights in VRAM at any time — 19x less than resident.
// Per-layer command buffers with overlapped weight upload.
type metalStreamingForward struct {
	metal *mongoose.Metal
	s     *serveState
	st    *gguf.SafeTensors

	// CPU-side weight cache — read from safetensors once, reuse every token
	layerWeights []streamLayerWeights
	finalNorm    []float32
	lmHeadData   []float32
}

type streamLayerWeights struct {
	norm1, wq, wk, wv       []float32
	bq, bk, bv              []float32
	wo, norm2, gate, up, down []float32
}

func buildMetalStreamingForward(s *serveState, st *gguf.SafeTensors, lmHeadData []float32) *metalStreamingForward {
	metal, ok := s.eng.(*mongoose.Metal)
	if !ok {
		return nil
	}

	if metal.StreamBuild() != 0 {
		return nil
	}

	headDim := s.dim / s.heads
	kvDim := s.kvHeads * headDim

	sf := &metalStreamingForward{
		metal:      metal,
		s:          s,
		st:         st,
		lmHeadData: lmHeadData,
	}

	log.Printf("[serve] Loading weights into CPU cache for streaming inference...")
	sf.layerWeights = make([]streamLayerWeights, s.layers)
	for l := 0; l < s.layers; l++ {
		prefix := fmt.Sprintf("model.layers.%d.", l)
		read := func(name string) []float32 {
			d, _, _ := st.ReadTensorFloat32(prefix + name)
			return d
		}
		lw := &sf.layerWeights[l]
		lw.norm1 = read("input_layernorm.weight")
		lw.wq = read("self_attn.q_proj.weight")
		lw.wk = read("self_attn.k_proj.weight")
		lw.wv = read("self_attn.v_proj.weight")
		lw.bq, _, _ = st.ReadTensorFloat32(prefix + "self_attn.q_proj.bias")
		lw.bk, _, _ = st.ReadTensorFloat32(prefix + "self_attn.k_proj.bias")
		lw.bv, _, _ = st.ReadTensorFloat32(prefix + "self_attn.v_proj.bias")
		lw.wo = read("self_attn.o_proj.weight")
		lw.norm2 = read("post_attention_layernorm.weight")
		lw.gate = read("mlp.gate_proj.weight")
		lw.up = read("mlp.up_proj.weight")
		lw.down = read("mlp.down_proj.weight")
	}
	sf.finalNorm, _, _ = st.ReadTensorFloat32("model.norm.weight")

	// Upload finalNorm + lmHead into the resident g_inf buffers
	// (they're small and shared across all tokens)
	wi := s.layers * 12
	metal.FusedSetWeight(wi, sf.finalNorm)
	metal.FusedSetWeight(wi+1, lmHeadData)

	log.Printf("[serve] Metal streaming inference ready — %d layers, ping-pong weight buffers (%d MB VRAM vs %d MB resident)",
		s.layers,
		streamVRAM(s.dim, kvDim, s.ffnDim, s.vocabSize, s.layers, s.maxSeq)/(1024*1024),
		residentVRAM(s.dim, kvDim, s.ffnDim, s.vocabSize, s.layers, s.maxSeq)/(1024*1024))

	return sf
}

// forward runs one token with ping-pong weight streaming.
// Upload layer N+1 weights into the off-set while GPU computes layer N.
func (sf *metalStreamingForward) forward(tokenID, pos int) []float32 {
	s := sf.s
	tokOff := tokenID * s.dim
	if tokenID < 0 || tokOff+s.dim > len(s.embedData) {
		return nil
	}

	hidden := make([]float32, s.dim)
	copy(hidden, s.embedData[tokOff:tokOff+s.dim])
	sf.metal.StreamSetHidden(hidden)

	// Pre-upload layer 0 into set A
	lw := &sf.layerWeights[0]
	sf.metal.StreamUploadLayer(0, 0,
		lw.norm1, lw.wq, lw.wk, lw.wv,
		lw.bq, lw.bk, lw.bv,
		lw.wo, lw.norm2, lw.gate, lw.up, lw.down)

	var wg sync.WaitGroup

	for l := 0; l < s.layers; l++ {
		set := l % 2

		// Start uploading next layer into the other set while GPU runs this layer
		if l+1 < s.layers {
			nextSet := (l + 1) % 2
			nextLW := &sf.layerWeights[l+1]
			wg.Add(1)
			go func(ns, nl int, nw *streamLayerWeights) {
				sf.metal.StreamUploadLayer(ns, nl,
					nw.norm1, nw.wq, nw.wk, nw.wv,
					nw.bq, nw.bk, nw.bv,
					nw.wo, nw.norm2, nw.gate, nw.up, nw.down)
				wg.Done()
			}(nextSet, l+1, nextLW)
		}

		// GPU: run this layer
		sf.metal.StreamStepLayer(set, l, pos)

		// Wait for next layer's upload to finish before we loop
		if l+1 < s.layers {
			wg.Wait()
		}
	}

	logits := make([]float32, s.vocabSize)
	sf.metal.StreamStepFinal(pos, logits)
	return logits
}

func streamVRAM(dim, kvDim, ffnDim, vocabSize, nLayers, maxSeq int) int {
	// 2 sets of weight buffers + KV caches + scratch + finalNorm + lmHead
	weightBytes := 2 * (dim*dim + kvDim*dim*2 + dim*dim + ffnDim*dim*3) // 2 layers' worth, Q8
	kvBytes := nLayers * 2 * maxSeq * kvDim * 4
	scratchBytes := (dim*6 + kvDim*2 + ffnDim*3 + vocabSize) * 4
	finalBytes := (dim + vocabSize*dim) // finalNorm + lmHead (Q8)
	return weightBytes + kvBytes + scratchBytes + finalBytes
}

func residentVRAM(dim, kvDim, ffnDim, vocabSize, nLayers, maxSeq int) int {
	weightBytes := nLayers * (dim*dim + kvDim*dim*2 + dim*dim + ffnDim*dim*3)
	kvBytes := nLayers * 2 * maxSeq * kvDim * 4
	scratchBytes := (dim*6 + kvDim*2 + ffnDim*3 + vocabSize) * 4
	finalBytes := (dim + vocabSize*dim)
	return weightBytes + kvBytes + scratchBytes + finalBytes
}
