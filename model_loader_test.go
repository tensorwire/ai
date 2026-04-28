package main

import (
	"archive/zip"
	"os"
	"path/filepath"
	"testing"

	"github.com/tensorwire/gguf"
)

func TestOpenModelSafeTensors(t *testing.T) {
	dir := t.TempDir()

	// Create minimal safetensors model
	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight": {Data: make([]float32, 64*16), Shape: []int{64, 16}},
	}
	gguf.SaveSafeTensors(filepath.Join(dir, "model.safetensors"), tensors)
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"hidden_size":16,"num_hidden_layers":1}`), 0644)

	m, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if m.Format() != "safetensors" {
		t.Errorf("format = %s, want safetensors", m.Format())
	}
	if m.ConfigInt("hidden_size", 0) != 16 {
		t.Errorf("hidden_size = %d, want 16", m.ConfigInt("hidden_size", 0))
	}
	if !m.HasTensor("model.embed_tokens.weight") {
		t.Error("missing embed_tokens tensor")
	}

	data, err := m.ReadTensorFloat32("model.embed_tokens.weight")
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 64*16 {
		t.Errorf("tensor size = %d, want %d", len(data), 64*16)
	}
}

func TestOpenModelGGUF(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")

	// Create minimal GGUF file
	w := gguf.NewGGUFWriter()
	w.AddString("general.architecture", "llama")
	w.AddUint32("llama.block_count", 1)
	w.AddUint32("llama.embedding_length", 16)
	w.AddTensorF32("token_embd.weight", make([]float32, 64*16), 64, 16)
	if err := w.Write(path); err != nil {
		t.Fatal(err)
	}

	m, err := OpenModel(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if m.Format() != "gguf" {
		t.Errorf("format = %s, want gguf", m.Format())
	}
}

func TestOpenModelGGUFInDir(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.gguf")

	w := gguf.NewGGUFWriter()
	w.AddString("general.architecture", "llama")
	w.AddTensorF32("token_embd.weight", make([]float32, 16), 1, 16)
	w.Write(path)

	m, err := OpenModel(dir) // pass directory, not file
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if m.Format() != "gguf" {
		t.Errorf("format = %s, want gguf", m.Format())
	}
}

func TestOpenModelNotFound(t *testing.T) {
	_, err := OpenModel("/tmp/nonexistent_model_path_xyz")
	if err == nil {
		t.Error("expected error for nonexistent path")
	}
}

func TestOpenModelEmptyDir(t *testing.T) {
	dir := t.TempDir()
	_, err := OpenModel(dir)
	if err == nil {
		t.Error("expected error for empty directory")
	}
}

func TestHasSafeTensors(t *testing.T) {
	dir := t.TempDir()
	if hasSafeTensors(dir) {
		t.Error("empty dir should not have safetensors")
	}

	os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte{}, 0644)
	if !hasSafeTensors(dir) {
		t.Error("should detect safetensors file")
	}
}

func TestConfigHelpers(t *testing.T) {
	cfg := map[string]interface{}{
		"hidden_size": float64(256),
		"rope_theta":  float64(10000.0),
		"hidden_act":  "silu",
	}

	if configInt(cfg, "hidden_size", 0) != 256 {
		t.Error("configInt failed")
	}
	if configInt(cfg, "missing", 42) != 42 {
		t.Error("configInt default failed")
	}
	if configFloat(cfg, "rope_theta", 0) != 10000.0 {
		t.Error("configFloat failed")
	}
	if configInt(nil, "anything", 99) != 99 {
		t.Error("nil config should return default")
	}
}

func TestGGUFNameMapRoundTrip(t *testing.T) {
	nameMap := buildGGUFNameMap(2)

	if nameMap["model.embed_tokens.weight"] != "token_embd.weight" {
		t.Errorf("embed mapping wrong: %s", nameMap["model.embed_tokens.weight"])
	}
	if nameMap["model.layers.0.self_attn.q_proj.weight"] != "blk.0.attn_q.weight" {
		t.Errorf("q_proj mapping wrong: %s", nameMap["model.layers.0.self_attn.q_proj.weight"])
	}
	if nameMap["model.layers.1.mlp.gate_proj.weight"] != "blk.1.ffn_gate.weight" {
		t.Errorf("gate mapping wrong: %s", nameMap["model.layers.1.mlp.gate_proj.weight"])
	}
}

func TestOpenByMagicGGUF(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.bin") // no .gguf extension

	w := gguf.NewGGUFWriter()
	w.AddString("general.architecture", "llama")
	w.AddTensorF32("token_embd.weight", make([]float32, 16), 1, 16)
	w.Write(path)

	m, err := OpenModel(path) // should detect by magic bytes
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Format() != "gguf" {
		t.Errorf("format = %s, want gguf", m.Format())
	}
}

func TestOpenZip(t *testing.T) {
	dir := t.TempDir()
	zipPath := filepath.Join(dir, "model.zip")

	// Create a zip containing a safetensors model
	modelDir := filepath.Join(dir, "inner")
	os.MkdirAll(modelDir, 0755)
	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight": {Data: make([]float32, 32), Shape: []int{4, 8}},
	}
	gguf.SaveSafeTensors(filepath.Join(modelDir, "model.safetensors"), tensors)
	os.WriteFile(filepath.Join(modelDir, "config.json"),
		[]byte(`{"hidden_size":8,"num_hidden_layers":1,"vocab_size":4}`), 0644)

	// Zip it
	zf, _ := os.Create(zipPath)
	zw := zip.NewWriter(zf)
	for _, name := range []string{"model.safetensors", "config.json"} {
		data, _ := os.ReadFile(filepath.Join(modelDir, name))
		fw, _ := zw.Create("mymodel/" + name)
		fw.Write(data)
	}
	zw.Close()
	zf.Close()

	m, err := OpenModel(zipPath)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	if m.Format() != "safetensors" {
		t.Errorf("format = %s, want safetensors", m.Format())
	}
	if m.ConfigInt("hidden_size", 0) != 8 {
		t.Errorf("hidden_size = %d, want 8", m.ConfigInt("hidden_size", 0))
	}
}

func TestInferConfigFromTensors(t *testing.T) {
	dir := t.TempDir()
	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight":                      {Data: make([]float32, 256*64), Shape: []int{256, 64}},
		"model.layers.0.self_attn.q_proj.weight":         {Data: make([]float32, 64*64), Shape: []int{64, 64}},
		"model.layers.0.self_attn.k_proj.weight":         {Data: make([]float32, 64*64), Shape: []int{64, 64}},
		"model.layers.0.mlp.gate_proj.weight":            {Data: make([]float32, 128*64), Shape: []int{128, 64}},
		"model.layers.0.input_layernorm.weight":          {Data: make([]float32, 64), Shape: []int{64}},
		"model.layers.0.post_attention_layernorm.weight":  {Data: make([]float32, 64), Shape: []int{64}},
	}
	gguf.SaveSafeTensors(filepath.Join(dir, "model.safetensors"), tensors)
	// Deliberately NO config.json

	m, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if m.ConfigInt("hidden_size", 0) != 64 {
		t.Errorf("inferred hidden_size = %d, want 64", m.ConfigInt("hidden_size", 0))
	}
	if m.ConfigInt("vocab_size", 0) != 256 {
		t.Errorf("inferred vocab_size = %d, want 256", m.ConfigInt("vocab_size", 0))
	}
	if m.ConfigInt("intermediate_size", 0) != 128 {
		t.Errorf("inferred intermediate_size = %d, want 128", m.ConfigInt("intermediate_size", 0))
	}
	if m.ConfigInt("num_hidden_layers", 0) != 1 {
		t.Errorf("inferred layers = %d, want 1", m.ConfigInt("num_hidden_layers", 0))
	}
}

func TestEstimateParamCount(t *testing.T) {
	dir := t.TempDir()
	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight": {Data: make([]float32, 64*16), Shape: []int{64, 16}},
	}
	gguf.SaveSafeTensors(filepath.Join(dir, "model.safetensors"), tensors)
	os.WriteFile(filepath.Join(dir, "config.json"),
		[]byte(`{"hidden_size":128,"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":256,"vocab_size":256}`), 0644)

	m, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	n := m.EstimateParamCount()
	if n <= 0 {
		t.Errorf("param count = %d, want > 0", n)
	}
	t.Logf("estimated params: %s (%d)", m.FormatParamCount(), n)

	cat := m.SizeCategory()
	if cat != "small" {
		t.Errorf("size category = %s, want small", cat)
	}
}

func TestModelSourceFormat(t *testing.T) {
	dir := t.TempDir()
	tensors := map[string]gguf.SaveTensor{
		"test": {Data: make([]float32, 4), Shape: []int{4}},
	}
	gguf.SaveSafeTensors(filepath.Join(dir, "model.safetensors"), tensors)

	m, err := OpenModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	if m.Dir() != dir {
		t.Errorf("Dir() = %s, want %s", m.Dir(), dir)
	}
	if m.Format() != "safetensors" {
		t.Errorf("Format() = %s, want safetensors", m.Format())
	}
}
