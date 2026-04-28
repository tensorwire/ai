package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorwire/gguf"
)

func TestSaveAndLoadSafeTensors(t *testing.T) {
	dir := t.TempDir()

	// Create a minimal model's worth of tensors
	dim := 16
	ffnDim := 32
	vocabSize := 64
	kvDim := 8
	nLayers := 2

	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight": {Data: randf(vocabSize * dim), Shape: []int{vocabSize, dim}},
		"model.norm.weight":         {Data: onesf(dim), Shape: []int{dim}},
	}
	for l := 0; l < nLayers; l++ {
		pfx := "model.layers." + itoa(l) + "."
		tensors[pfx+"self_attn.q_proj.weight"] = gguf.SaveTensor{Data: randf(dim * dim), Shape: []int{dim, dim}}
		tensors[pfx+"self_attn.k_proj.weight"] = gguf.SaveTensor{Data: randf(kvDim * dim), Shape: []int{kvDim, dim}}
		tensors[pfx+"self_attn.v_proj.weight"] = gguf.SaveTensor{Data: randf(kvDim * dim), Shape: []int{kvDim, dim}}
		tensors[pfx+"self_attn.o_proj.weight"] = gguf.SaveTensor{Data: randf(dim * dim), Shape: []int{dim, dim}}
		tensors[pfx+"mlp.gate_proj.weight"] = gguf.SaveTensor{Data: randf(ffnDim * dim), Shape: []int{ffnDim, dim}}
		tensors[pfx+"mlp.up_proj.weight"] = gguf.SaveTensor{Data: randf(ffnDim * dim), Shape: []int{ffnDim, dim}}
		tensors[pfx+"mlp.down_proj.weight"] = gguf.SaveTensor{Data: randf(dim * ffnDim), Shape: []int{dim, ffnDim}}
		tensors[pfx+"input_layernorm.weight"] = gguf.SaveTensor{Data: onesf(dim), Shape: []int{dim}}
		tensors[pfx+"post_attention_layernorm.weight"] = gguf.SaveTensor{Data: onesf(dim), Shape: []int{dim}}
	}

	stPath := filepath.Join(dir, "model.safetensors")
	if err := gguf.SaveSafeTensors(stPath, tensors); err != nil {
		t.Fatalf("save: %v", err)
	}

	if _, err := os.Stat(stPath); err != nil {
		t.Fatalf("safetensors file not created: %v", err)
	}

	// Verify we can read it back
	st, err := gguf.OpenSafeTensors(dir)
	if err != nil {
		t.Fatalf("open: %v", err)
	}

	// Check embed
	embedData, info, err := st.ReadTensorFloat32("model.embed_tokens.weight")
	if err != nil {
		t.Fatalf("read embed: %v", err)
	}
	if len(embedData) != vocabSize*dim {
		t.Errorf("embed size = %d, want %d", len(embedData), vocabSize*dim)
	}
	if len(info.Shape) != 2 || info.Shape[0] != vocabSize || info.Shape[1] != dim {
		t.Errorf("embed shape = %v, want [%d %d]", info.Shape, vocabSize, dim)
	}

	// Check a layer weight
	wq, _, err := st.ReadTensorFloat32("model.layers.0.self_attn.q_proj.weight")
	if err != nil {
		t.Fatalf("read wq: %v", err)
	}
	if len(wq) != dim*dim {
		t.Errorf("wq size = %d, want %d", len(wq), dim*dim)
	}

	// Verify roundtrip: saved data matches loaded data
	origEmbed := tensors["model.embed_tokens.weight"].Data
	for i := range origEmbed {
		if embedData[i] != origEmbed[i] {
			t.Errorf("embed[%d] roundtrip mismatch: saved %f, loaded %f", i, origEmbed[i], embedData[i])
			break
		}
	}
}

func TestSaveConfigJSON(t *testing.T) {
	dir := t.TempDir()

	cfg := map[string]interface{}{
		"architectures":    []string{"LlamaForCausalLM"},
		"hidden_size":      128,
		"num_hidden_layers": 4,
		"num_attention_heads": 4,
		"num_key_value_heads": 2,
		"intermediate_size": 256,
		"vocab_size":       256,
	}
	data, _ := json.Marshal(cfg)
	path := filepath.Join(dir, "config.json")
	os.WriteFile(path, data, 0644)

	// Verify AutoDetect can read it
	p := AutoDetect(dir)
	if p.Dim != 128 {
		t.Errorf("Dim = %d, want 128", p.Dim)
	}
	if p.Layers != 4 {
		t.Errorf("Layers = %d, want 4", p.Layers)
	}
}

func TestSaveGGUFRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.gguf")

	// Write a small GGUF
	w := gguf.NewGGUFWriter()
	w.AddString("general.architecture", "llama")
	w.AddUint32("llama.embedding_length", 16)
	data := make([]float32, 64)
	for i := range data { data[i] = float32(i) * 0.1 }
	w.AddTensorF32("token_embd.weight", data, 4, 16)
	if err := w.Write(path); err != nil {
		t.Fatalf("write GGUF: %v", err)
	}

	// Read it back
	r, err := gguf.OpenGGUF(path)
	if err != nil {
		t.Fatalf("open GGUF: %v", err)
	}
	defer r.Close()

	if r.MetadataString("general.architecture") != "llama" {
		t.Errorf("architecture = %q, want llama", r.MetadataString("general.architecture"))
	}
	if r.MetadataUint32("llama.embedding_length") != 16 {
		t.Errorf("embedding_length = %d, want 16", r.MetadataUint32("llama.embedding_length"))
	}

	readData, shape, err := r.ReadTensorFloat32("token_embd.weight")
	if err != nil {
		t.Fatalf("read tensor: %v", err)
	}
	if len(readData) != 64 {
		t.Errorf("tensor size = %d, want 64", len(readData))
	}
	if len(shape) != 2 || shape[0] != 4 || shape[1] != 16 {
		t.Errorf("tensor shape = %v, want [4 16]", shape)
	}

	// Verify data matches
	for i := range data {
		if readData[i] != data[i] {
			t.Errorf("tensor[%d] = %f, want %f", i, readData[i], data[i])
			break
		}
	}
}

func randf(n int) []float32 {
	s := make([]float32, n)
	for i := range s { s[i] = float32(i%100) * 0.01 }
	return s
}

func onesf(n int) []float32 {
	s := make([]float32, n)
	for i := range s { s[i] = 1.0 }
	return s
}

func itoa(n int) string {
	return fmt.Sprintf("%d", n)
}
