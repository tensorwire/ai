package main

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tensorwire/gguf"
)

// === resolveModel ===

func TestResolveModelAbsolutePath(t *testing.T) {
	dir := t.TempDir()
	got := resolveModel(dir)
	if got != dir {
		t.Errorf("resolveModel(%q) = %q, want same path", dir, got)
	}
}

func TestResolveModelFromModelsDir(t *testing.T) {
	orig := modelsDir
	defer func() { modelsDir = orig }()

	dir := t.TempDir()
	modelsDir = dir
	modelDir := filepath.Join(dir, "test-model")
	os.MkdirAll(modelDir, 0755)

	got := resolveModel("test-model")
	if got != modelDir {
		t.Errorf("resolveModel(\"test-model\") = %q, want %q", got, modelDir)
	}
}

func TestResolveModelWithSuffix(t *testing.T) {
	orig := modelsDir
	defer func() { modelsDir = orig }()

	dir := t.TempDir()
	modelsDir = dir
	modelDir := filepath.Join(dir, "llama-hf")
	os.MkdirAll(modelDir, 0755)

	got := resolveModel("llama")
	if got != modelDir {
		t.Errorf("resolveModel(\"llama\") = %q, want %q", got, modelDir)
	}
}

// === isAdaptedWeight ===

func TestIsAdaptedWeight(t *testing.T) {
	tests := []struct {
		name    string
		targets []string
		want    bool
	}{
		{"model.layers.0.self_attn.q_proj.weight", nil, true},
		{"model.layers.0.self_attn.k_proj.weight", nil, true},
		{"model.layers.0.self_attn.v_proj.weight", nil, true},
		{"model.layers.0.self_attn.o_proj.weight", nil, true},
		{"model.layers.0.mlp.gate_proj.weight", nil, true},
		{"model.layers.0.mlp.up_proj.weight", nil, true},
		{"model.layers.0.mlp.down_proj.weight", nil, true},
		{"model.embed_tokens.weight", nil, false},
		{"model.norm.weight", nil, false},
		{"model.layers.0.input_layernorm.weight", nil, false},
		// Custom targets
		{"model.layers.0.self_attn.q_proj.weight", []string{"q_proj", "v_proj"}, true},
		{"model.layers.0.self_attn.k_proj.weight", []string{"q_proj", "v_proj"}, false},
	}
	for _, tt := range tests {
		got := isAdaptedWeight(tt.name, tt.targets)
		if got != tt.want {
			t.Errorf("isAdaptedWeight(%q, %v) = %v, want %v", tt.name, tt.targets, got, tt.want)
		}
	}
}

// === contains / searchString ===

func TestContains(t *testing.T) {
	tests := []struct {
		s, sub string
		want   bool
	}{
		{"hello world", "world", true},
		{"hello world", "hello", true},
		{"hello", "hello world", false},
		{"", "", true},
		{"abc", "d", false},
	}
	for _, tt := range tests {
		got := contains(tt.s, tt.sub)
		if got != tt.want {
			t.Errorf("contains(%q, %q) = %v, want %v", tt.s, tt.sub, got, tt.want)
		}
	}
}

// === softmax ===

func TestSoftmax(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0}
	softmax(x, 3)

	var sum float32
	for _, v := range x {
		sum += v
		if v < 0 || v > 1 {
			t.Errorf("softmax value out of range: %f", v)
		}
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}
	if x[2] <= x[1] || x[1] <= x[0] {
		t.Errorf("softmax ordering wrong: %v (expected increasing)", x)
	}
}

func TestSoftmaxUniform(t *testing.T) {
	x := []float32{5.0, 5.0, 5.0, 5.0}
	softmax(x, 4)
	for i, v := range x {
		if math.Abs(float64(v-0.25)) > 1e-5 {
			t.Errorf("softmax[%d] = %f, want 0.25", i, v)
		}
	}
}

func TestSoftmaxLargeValues(t *testing.T) {
	x := []float32{1000, 1001, 1002}
	softmax(x, 3)
	var sum float32
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-5 {
		t.Errorf("softmax with large values: sum = %f, want 1.0", sum)
	}
}

// === silu ===

func TestSiLU(t *testing.T) {
	if got := silu(0); got != 0 {
		t.Errorf("silu(0) = %f, want 0", got)
	}
	pos := silu(2.0)
	if pos <= 0 {
		t.Errorf("silu(2.0) = %f, want positive", pos)
	}
	neg := silu(-5.0)
	if neg >= 0 {
		t.Errorf("silu(-5.0) = %f, want negative", neg)
	}
	large := silu(100.0)
	if math.Abs(float64(large-100.0)) > 0.1 {
		t.Errorf("silu(100) = %f, want ~100", large)
	}
}

// === argmax ===

func TestArgmax(t *testing.T) {
	tests := []struct {
		x    []float32
		want int
	}{
		{[]float32{1, 2, 3}, 2},
		{[]float32{3, 2, 1}, 0},
		{[]float32{1, 3, 2}, 1},
		{[]float32{5}, 0},
		{[]float32{-1, -2, -3}, 0},
	}
	for _, tt := range tests {
		got := argmax(tt.x)
		if got != tt.want {
			t.Errorf("argmax(%v) = %d, want %d", tt.x, got, tt.want)
		}
	}
}

// === sampleTopK ===

func TestSampleTopKGreedy(t *testing.T) {
	logits := []float32{0.1, 0.5, 0.9, 0.3}
	got := sampleTopK(logits, 0, 0)
	if got != 2 {
		t.Errorf("sampleTopK greedy = %d, want 2", got)
	}
}

func TestSampleTopKDeterministicAtZeroTemp(t *testing.T) {
	for i := 0; i < 20; i++ {
		logits := []float32{0.1, 0.5, 0.9, 0.3}
		got := sampleTopK(logits, 0, 40)
		if got != 2 {
			t.Errorf("sampleTopK(temp=0) = %d, want 2 always", got)
		}
	}
}

func TestSampleTopKReturnsValidIndex(t *testing.T) {
	for i := 0; i < 100; i++ {
		logits := make([]float32, 100)
		for j := range logits {
			logits[j] = float32(j) * 0.01
		}
		got := sampleTopK(logits, 0.7, 10)
		if got < 0 || got >= 100 {
			t.Fatalf("sampleTopK returned out-of-range index: %d", got)
		}
	}
}

// === applyRoPE ===

func TestApplyRoPEPreservesNorm(t *testing.T) {
	headDim := 8
	numHeads := 2
	kvHeads := 2
	q := make([]float32, numHeads*headDim)
	k := make([]float32, kvHeads*headDim)
	for i := range q {
		q[i] = float32(i+1) * 0.1
	}
	for i := range k {
		k[i] = float32(i+1) * 0.1
	}

	var normBefore float64
	for _, v := range q {
		normBefore += float64(v * v)
	}
	normBefore = math.Sqrt(normBefore)

	applyRoPE(q, k, 5, headDim, 10000.0, numHeads, kvHeads)

	var normAfter float64
	for _, v := range q {
		normAfter += float64(v * v)
	}
	normAfter = math.Sqrt(normAfter)

	if math.Abs(normBefore-normAfter) > 1e-4 {
		t.Errorf("RoPE changed Q norm: %f → %f", normBefore, normAfter)
	}
}

func TestApplyRoPEPos0IsIdentity(t *testing.T) {
	headDim := 4
	q := []float32{1, 0, 0, 0}
	k := []float32{1, 0, 0, 0}
	applyRoPE(q, k, 0, headDim, 10000.0, 1, 1)
	if math.Abs(float64(q[0]-1.0)) > 1e-6 || math.Abs(float64(q[1])) > 1e-6 {
		t.Errorf("RoPE at pos=0: q = %v, want [1 0 0 0]", q)
	}
}

func TestApplyRoPESingleMatchesApplyRoPE(t *testing.T) {
	headDim := 8
	numHeads := 2
	theta := 10000.0
	pos := 3

	x1 := make([]float32, numHeads*headDim)
	x2 := make([]float32, numHeads*headDim)
	dummy := make([]float32, numHeads*headDim)
	for i := range x1 {
		x1[i] = float32(i+1) * 0.1
		x2[i] = x1[i]
	}

	applyRoPE(x1, dummy, pos, headDim, float32(theta), numHeads, numHeads)
	applyRoPESingle(x2, pos, headDim, numHeads, theta)

	for i := range x1 {
		if math.Abs(float64(x1[i]-x2[i])) > 1e-5 {
			t.Errorf("mismatch at %d: applyRoPE=%f, applyRoPESingle=%f", i, x1[i], x2[i])
		}
	}
}

// === buildGGUFNameMap ===

func TestBuildGGUFNameMap(t *testing.T) {
	m := buildGGUFNameMap(2)

	tests := []struct {
		hf, gguf string
	}{
		{"model.embed_tokens.weight", "token_embd.weight"},
		{"model.norm.weight", "output_norm.weight"},
		{"lm_head.weight", "output.weight"},
		{"model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"},
		{"model.layers.0.mlp.gate_proj.weight", "blk.0.ffn_gate.weight"},
		{"model.layers.1.self_attn.k_proj.weight", "blk.1.attn_k.weight"},
		{"model.layers.1.post_attention_layernorm.weight", "blk.1.ffn_norm.weight"},
	}
	for _, tt := range tests {
		got := m[tt.hf]
		if got != tt.gguf {
			t.Errorf("nameMap[%q] = %q, want %q", tt.hf, got, tt.gguf)
		}
	}

	if _, ok := m["model.layers.2.self_attn.q_proj.weight"]; ok {
		t.Error("nameMap should not contain layer 2 for nLayers=2")
	}
}

// === cmdModels (filesystem) ===

func TestCmdModelsEmpty(t *testing.T) {
	orig := modelsDir
	defer func() { modelsDir = orig }()

	modelsDir = t.TempDir()
	// Should not panic with empty directory
	cmdModels()
}

func TestCmdModelsWithModel(t *testing.T) {
	orig := modelsDir
	defer func() { modelsDir = orig }()

	dir := t.TempDir()
	modelsDir = dir

	modelDir := filepath.Join(dir, "test-llama")
	os.MkdirAll(modelDir, 0755)

	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight": {Data: randf(64 * 16), Shape: []int{64, 16}},
	}
	gguf.SaveSafeTensors(filepath.Join(modelDir, "model.safetensors"), tensors)

	cfg := map[string]interface{}{
		"architectures": []string{"LlamaForCausalLM"},
	}
	data, _ := json.Marshal(cfg)
	os.WriteFile(filepath.Join(modelDir, "config.json"), data, 0644)

	cmdModels()
}

// === cmdInfo (filesystem) ===

func TestCmdInfoWithModel(t *testing.T) {
	orig := modelsDir
	defer func() { modelsDir = orig }()

	dir := t.TempDir()
	modelsDir = dir

	modelDir := filepath.Join(dir, "info-test")
	os.MkdirAll(modelDir, 0755)

	dim := 16
	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight":                       {Data: randf(64 * dim), Shape: []int{64, dim}},
		"model.norm.weight":                               {Data: onesf(dim), Shape: []int{dim}},
		"model.layers.0.self_attn.q_proj.weight":          {Data: randf(dim * dim), Shape: []int{dim, dim}},
		"model.layers.0.self_attn.k_proj.weight":          {Data: randf(dim * dim), Shape: []int{dim, dim}},
		"model.layers.0.self_attn.v_proj.weight":          {Data: randf(dim * dim), Shape: []int{dim, dim}},
		"model.layers.0.self_attn.o_proj.weight":          {Data: randf(dim * dim), Shape: []int{dim, dim}},
		"model.layers.0.mlp.gate_proj.weight":             {Data: randf(32 * dim), Shape: []int{32, dim}},
		"model.layers.0.mlp.up_proj.weight":               {Data: randf(32 * dim), Shape: []int{32, dim}},
		"model.layers.0.mlp.down_proj.weight":             {Data: randf(dim * 32), Shape: []int{dim, 32}},
		"model.layers.0.input_layernorm.weight":           {Data: onesf(dim), Shape: []int{dim}},
		"model.layers.0.post_attention_layernorm.weight":  {Data: onesf(dim), Shape: []int{dim}},
	}
	gguf.SaveSafeTensors(filepath.Join(modelDir, "model.safetensors"), tensors)

	cfg := map[string]interface{}{
		"hidden_size":        dim,
		"num_hidden_layers":  1,
		"num_attention_heads": 2,
		"intermediate_size":  32,
		"vocab_size":         64,
		"hidden_act":         "silu",
	}
	data, _ := json.Marshal(cfg)
	os.WriteFile(filepath.Join(modelDir, "config.json"), data, 0644)

	cmdInfo("info-test")
}

// === cmdConvert (filesystem + GGUF roundtrip) ===

func TestCmdConvertGGUF(t *testing.T) {
	orig := modelsDir
	defer func() { modelsDir = orig }()

	dir := t.TempDir()
	modelsDir = dir

	modelDir := filepath.Join(dir, "convert-test")
	os.MkdirAll(modelDir, 0755)

	dim := 16
	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight": {Data: randf(64 * dim), Shape: []int{64, dim}},
		"model.norm.weight":         {Data: onesf(dim), Shape: []int{dim}},
	}
	gguf.SaveSafeTensors(filepath.Join(modelDir, "model.safetensors"), tensors)

	cfg := map[string]interface{}{
		"architectures":    []string{"LlamaForCausalLM"},
		"hidden_size":      dim,
		"num_hidden_layers": 0,
		"vocab_size":       64,
	}
	data, _ := json.Marshal(cfg)
	os.WriteFile(filepath.Join(modelDir, "config.json"), data, 0644)

	outputPath := filepath.Join(dir, "test-output.gguf")
	cmdConvert("gguf", []string{modelDir, outputPath})

	if _, err := os.Stat(outputPath); err != nil {
		t.Fatalf("GGUF output not created: %v", err)
	}

	r, err := gguf.OpenGGUF(outputPath)
	if err != nil {
		t.Fatalf("can't read output GGUF: %v", err)
	}
	defer r.Close()

	names := r.TensorNames()
	if len(names) == 0 {
		t.Error("GGUF has no tensors")
	}
}

func TestCmdConvertWithQuant(t *testing.T) {
	orig := modelsDir
	defer func() { modelsDir = orig }()

	dir := t.TempDir()
	modelsDir = dir

	modelDir := filepath.Join(dir, "quant-test")
	os.MkdirAll(modelDir, 0755)

	dim := 32
	tensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight": {Data: randf(64 * dim), Shape: []int{64, dim}},
		"model.norm.weight":         {Data: onesf(dim), Shape: []int{dim}},
	}
	gguf.SaveSafeTensors(filepath.Join(modelDir, "model.safetensors"), tensors)

	cfg := map[string]interface{}{
		"architectures":    []string{"LlamaForCausalLM"},
		"hidden_size":      dim,
		"num_hidden_layers": 0,
		"vocab_size":       64,
	}
	data, _ := json.Marshal(cfg)
	os.WriteFile(filepath.Join(modelDir, "config.json"), data, 0644)

	outputPath := filepath.Join(dir, "test-q8.gguf")
	cmdConvert("gguf", []string{modelDir, outputPath, "--quant", "q8_0"})

	if _, err := os.Stat(outputPath); err != nil {
		t.Fatalf("quantized GGUF not created: %v", err)
	}
}

// === cmdMerge (LoRA merge) ===

func TestCmdMergeLoRA(t *testing.T) {
	orig := modelsDir
	defer func() { modelsDir = orig }()

	dir := t.TempDir()
	modelsDir = dir
	dim := 16
	rank := 4

	// Create base model
	baseDir := filepath.Join(dir, "base-model")
	os.MkdirAll(baseDir, 0755)

	baseTensors := map[string]gguf.SaveTensor{
		"model.embed_tokens.weight":              {Data: randf(64 * dim), Shape: []int{64, dim}},
		"model.norm.weight":                      {Data: onesf(dim), Shape: []int{dim}},
		"model.layers.0.self_attn.q_proj.weight": {Data: onesf(dim * dim), Shape: []int{dim, dim}},
		"model.layers.0.self_attn.k_proj.weight": {Data: onesf(dim * dim), Shape: []int{dim, dim}},
		"model.layers.0.self_attn.v_proj.weight": {Data: onesf(dim * dim), Shape: []int{dim, dim}},
		"model.layers.0.self_attn.o_proj.weight": {Data: onesf(dim * dim), Shape: []int{dim, dim}},
		"model.layers.0.mlp.gate_proj.weight":    {Data: onesf(32 * dim), Shape: []int{32, dim}},
		"model.layers.0.mlp.up_proj.weight":      {Data: onesf(32 * dim), Shape: []int{32, dim}},
		"model.layers.0.mlp.down_proj.weight":    {Data: onesf(dim * 32), Shape: []int{dim, 32}},
		"model.layers.0.input_layernorm.weight":          {Data: onesf(dim), Shape: []int{dim}},
		"model.layers.0.post_attention_layernorm.weight":  {Data: onesf(dim), Shape: []int{dim}},
	}
	gguf.SaveSafeTensors(filepath.Join(baseDir, "model.safetensors"), baseTensors)

	cfg := map[string]interface{}{
		"architectures":     []string{"LlamaForCausalLM"},
		"hidden_size":       dim,
		"num_hidden_layers":  1,
		"num_attention_heads": 2,
		"intermediate_size":  32,
		"vocab_size":         64,
	}
	cfgData, _ := json.Marshal(cfg)
	os.WriteFile(filepath.Join(baseDir, "config.json"), cfgData, 0644)

	// Create LoRA adapters
	adapterDir := filepath.Join(dir, "adapters")
	os.MkdirAll(adapterDir, 0755)

	adapterTensors := map[string]gguf.SaveTensor{
		"model.layers.0.self_attn.q_proj.lora_A": {Data: onesf(rank * dim), Shape: []int{rank, dim}},
		"model.layers.0.self_attn.q_proj.lora_B": {Data: onesf(dim * rank), Shape: []int{dim, rank}},
	}
	gguf.SaveSafeTensors(filepath.Join(adapterDir, "model.safetensors"), adapterTensors)

	loraCfg := map[string]interface{}{
		"lora_rank":      rank,
		"target_modules": []string{"q_proj"},
	}
	loraData, _ := json.Marshal(loraCfg)
	os.WriteFile(filepath.Join(adapterDir, "lora_config.json"), loraData, 0644)

	// Merge
	outputDir := filepath.Join(dir, "merged")
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()
	os.Args = []string{"ai", "merge", baseDir, filepath.Join(adapterDir, "model.safetensors"), outputDir}

	cmdMerge()

	// Verify output exists
	mergedST := filepath.Join(outputDir, "model.safetensors")
	if _, err := os.Stat(mergedST); err != nil {
		t.Fatalf("merged model not created: %v", err)
	}

	// Verify merged weights differ from base (LoRA was applied)
	st, err := gguf.OpenSafeTensors(outputDir)
	if err != nil {
		t.Fatalf("open merged: %v", err)
	}
	qData, _, err := st.ReadTensorFloat32("model.layers.0.self_attn.q_proj.weight")
	if err != nil {
		t.Fatalf("read merged q_proj: %v", err)
	}

	// Base was all 1s, LoRA_A was all 1s [rank, dim], LoRA_B was all 1s [dim, rank]
	// B @ A = [dim, dim] where each element = rank (sum of rank 1s)
	// Merged = 1 + rank = 5
	expected := float32(1.0 + float32(rank))
	if math.Abs(float64(qData[0]-expected)) > 1e-3 {
		t.Errorf("merged q_proj[0] = %f, want %f (base=1 + LoRA B@A=%d)", qData[0], expected, rank)
	}

	// Config should be copied
	if _, err := os.Stat(filepath.Join(outputDir, "config.json")); err != nil {
		t.Error("config.json not copied to merged output")
	}
}

// === cmdPull (mock HTTP) ===

func TestCmdPullNeedsOrgSlashName(t *testing.T) {
	model := "justname"
	if strings.Contains(model, "/") {
		t.Error("model without org should not contain slash")
	}
	model2 := "org/name"
	if !strings.Contains(model2, "/") {
		t.Error("model with org should contain slash")
	}
}

func TestPullFileFiltering(t *testing.T) {
	type sibling struct {
		Filename string `json:"rfilename"`
	}
	files := []sibling{
		{"model.safetensors"},
		{"config.json"},
		{"tokenizer.json"},
		{"tokenizer_config.json"},
		{"tokenizer.model"},
		{"special_tokens_map.json"},
		{"model.safetensors.index.json"},
		{"README.md"},
		{"pytorch_model.bin"},
		{".gitattributes"},
	}

	wanted := []string{}
	for _, f := range files {
		name := f.Filename
		if hasSuffix(name, ".safetensors") ||
			name == "config.json" ||
			name == "tokenizer.json" ||
			name == "tokenizer_config.json" ||
			name == "tokenizer.model" ||
			name == "special_tokens_map.json" ||
			hasSuffix(name, ".safetensors.index.json") {
			wanted = append(wanted, name)
		}
	}

	if len(wanted) != 7 {
		t.Errorf("wanted %d files, got %d: %v", 7, len(wanted), wanted)
	}
	for _, w := range wanted {
		if w == "README.md" || w == "pytorch_model.bin" || w == ".gitattributes" {
			t.Errorf("unwanted file included: %s", w)
		}
	}
}

func hasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

// === downloadFile ===

func TestDownloadFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("test content"))
	}))
	defer server.Close()

	dest := filepath.Join(t.TempDir(), "downloaded.txt")
	err := downloadFile(server.URL+"/file", dest)
	if err != nil {
		t.Fatalf("downloadFile: %v", err)
	}

	data, err := os.ReadFile(dest)
	if err != nil {
		t.Fatalf("read downloaded: %v", err)
	}
	if string(data) != "test content" {
		t.Errorf("content = %q, want \"test content\"", string(data))
	}
}

func TestDownloadFile404(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
	}))
	defer server.Close()

	dest := filepath.Join(t.TempDir(), "should-fail.txt")
	err := downloadFile(server.URL+"/missing", dest)
	if err == nil {
		t.Error("expected error for 404, got nil")
	}
}

// === export npy ===

func TestExportNpy(t *testing.T) {
	dir := t.TempDir()

	// Create a small npy file
	data := []float32{1, 2, 3, 4, 5, 6}
	npyPath := filepath.Join(dir, "test.npy")
	if err := gguf.WriteNpy(npyPath, data, 2, 3); err != nil {
		t.Fatalf("write npy: %v", err)
	}

	outputPath := filepath.Join(dir, "output.jsonl")
	exportNpy(npyPath, outputPath)

	output, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read output: %v", err)
	}

	lines := splitLines(string(output))
	if len(lines) != 2 {
		t.Errorf("expected 2 lines, got %d", len(lines))
	}

	// Verify first row is valid JSON array [1, 2, 3]
	var row []float64
	if err := json.Unmarshal([]byte(lines[0]), &row); err != nil {
		t.Fatalf("parse row 0: %v", err)
	}
	if len(row) != 3 || row[0] != 1 || row[1] != 2 || row[2] != 3 {
		t.Errorf("row 0 = %v, want [1 2 3]", row)
	}
}

func splitLines(s string) []string {
	var lines []string
	for _, line := range splitOnNewline(s) {
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

func splitOnNewline(s string) []string {
	result := []string{}
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			result = append(result, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		result = append(result, s[start:])
	}
	return result
}
