package main

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/tensorwire/gguf"
)

// ModelSource abstracts over SafeTensors and GGUF in any packaging.
// All tensor reads return float32 regardless of storage format.
type ModelSource struct {
	format  string // "safetensors" or "gguf"
	st      *gguf.SafeTensors
	gr      *gguf.GGUFReader
	nameMap map[string]string // HF name → GGUF name
	config  map[string]interface{}
	dir     string // directory containing model files (for tokenizer)
	nLayers int
	tmpDir  string // non-empty if we extracted a zip (cleanup on Close)
}

// OpenModel auto-detects format and opens a model from any source:
//   - SafeTensors directory (with or without config.json)
//   - Single .safetensors file
//   - Single .gguf file
//   - Directory containing .gguf
//   - .zip archive containing any of the above
//   - Raw file (detected by magic bytes)
func OpenModel(path string) (*ModelSource, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("not found: %s", path)
	}

	if info.IsDir() {
		return openDir(path)
	}
	return openFile(path)
}

func openFile(path string) (*ModelSource, error) {
	ext := strings.ToLower(filepath.Ext(path))

	switch ext {
	case ".gguf":
		return openGGUFFile(path)
	case ".safetensors":
		return openSingleSafeTensors(path)
	case ".zip":
		return openZip(path)
	default:
		return openByMagic(path)
	}
}

func openDir(path string) (*ModelSource, error) {
	// SafeTensors first (most common for HuggingFace models)
	if hasSafeTensors(path) {
		return openSafeTensorsDir(path)
	}
	// GGUF file in directory
	if p := findGGUFInDir(path); p != "" {
		return openGGUFFile(p)
	}
	// Zip file in directory
	if p := findFileByExt(path, ".zip"); p != "" {
		return openZip(p)
	}
	return nil, fmt.Errorf("no model files found in %s (need .safetensors, .gguf, or .zip)", path)
}

// openByMagic detects format from file header bytes.
func openByMagic(path string) (*ModelSource, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	var magic [4]byte
	f.Read(magic[:])
	f.Close()

	// GGUF magic: "GGUF" (0x46554747)
	if magic[0] == 'G' && magic[1] == 'G' && magic[2] == 'U' && magic[3] == 'F' {
		return openGGUFFile(path)
	}
	// Zip magic: "PK\x03\x04"
	if magic[0] == 'P' && magic[1] == 'K' && magic[2] == 0x03 && magic[3] == 0x04 {
		return openZip(path)
	}
	// SafeTensors starts with a JSON header length (little-endian uint64)
	// The first 8 bytes are a small number (header size), so bytes 4-7 are usually 0
	if magic[2] == 0 && magic[3] == 0 {
		return openSingleSafeTensors(path)
	}

	return nil, fmt.Errorf("unknown model format: %s (unrecognized magic bytes)", path)
}

// openZip extracts a zip to a temp dir and opens the model inside.
func openZip(path string) (*ModelSource, error) {
	tmpDir, err := os.MkdirTemp("", "ai-model-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}

	r, err := zip.OpenReader(path)
	if err != nil {
		os.RemoveAll(tmpDir)
		return nil, fmt.Errorf("open zip: %w", err)
	}
	defer r.Close()

	for _, f := range r.File {
		if f.FileInfo().IsDir() {
			continue
		}
		// Flatten: strip leading directory from zip entries
		name := filepath.Base(f.Name)
		// But keep subdirectory structure for sharded safetensors
		relPath := f.Name
		if idx := strings.Index(relPath, "/"); idx >= 0 {
			relPath = relPath[idx+1:]
		}
		if relPath == "" {
			relPath = name
		}

		destPath := filepath.Join(tmpDir, relPath)
		os.MkdirAll(filepath.Dir(destPath), 0755)

		rc, err := f.Open()
		if err != nil {
			continue
		}
		out, err := os.Create(destPath)
		if err != nil {
			rc.Close()
			continue
		}
		io.Copy(out, rc)
		out.Close()
		rc.Close()
	}

	m, err := openDir(tmpDir)
	if err != nil {
		os.RemoveAll(tmpDir)
		return nil, fmt.Errorf("open extracted model: %w", err)
	}
	m.tmpDir = tmpDir
	return m, nil
}

// openSingleSafeTensors opens a lone .safetensors file (no directory).
func openSingleSafeTensors(path string) (*ModelSource, error) {
	dir := filepath.Dir(path)
	st, err := gguf.OpenSafeTensors(dir)
	if err != nil {
		// Try opening as if the file IS the model directory
		// Create a temp dir with a symlink
		tmpDir, _ := os.MkdirTemp("", "ai-model-*")
		dst := filepath.Join(tmpDir, filepath.Base(path))
		// Copy file (symlink may not work cross-device)
		data, err2 := os.ReadFile(path)
		if err2 != nil {
			os.RemoveAll(tmpDir)
			return nil, fmt.Errorf("read safetensors: %w", err2)
		}
		os.WriteFile(dst, data, 0644)
		st, err = gguf.OpenSafeTensors(tmpDir)
		if err != nil {
			os.RemoveAll(tmpDir)
			return nil, fmt.Errorf("open safetensors: %w", err)
		}
		cfg := inferConfigFromTensors(st)
		nLayers := configInt(cfg, "num_hidden_layers", 0)
		return &ModelSource{
			format: "safetensors", st: st, config: cfg,
			dir: tmpDir, nLayers: nLayers, tmpDir: tmpDir,
		}, nil
	}

	cfg := loadConfig(dir)
	if cfg == nil {
		cfg = inferConfigFromTensors(st)
	}
	nLayers := configInt(cfg, "num_hidden_layers", 0)

	return &ModelSource{
		format: "safetensors", st: st, config: cfg,
		dir: dir, nLayers: nLayers,
	}, nil
}

func openSafeTensorsDir(dir string) (*ModelSource, error) {
	st, err := gguf.OpenSafeTensors(dir)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}

	cfg := loadConfig(dir)
	if cfg == nil {
		cfg = inferConfigFromTensors(st)
	}
	nLayers := configInt(cfg, "num_hidden_layers", 0)

	return &ModelSource{
		format: "safetensors", st: st, config: cfg,
		dir: dir, nLayers: nLayers,
	}, nil
}

func openGGUFFile(path string) (*ModelSource, error) {
	gr, err := gguf.OpenGGUF(path)
	if err != nil {
		return nil, fmt.Errorf("open gguf: %w", err)
	}

	cfg := inferConfigFromGGUF(gr)

	// Merge with config.json if present alongside
	dir := filepath.Dir(path)
	if fileCfg := loadConfig(dir); fileCfg != nil {
		for k, v := range fileCfg {
			if _, exists := cfg[k]; !exists {
				cfg[k] = v
			}
		}
	}

	nLayers := configInt(cfg, "num_hidden_layers", 0)
	nameMap := buildGGUFNameMap(nLayers)

	return &ModelSource{
		format: "gguf", gr: gr, nameMap: nameMap,
		config: cfg, dir: dir, nLayers: nLayers,
	}, nil
}

// inferConfigFromGGUF extracts model config from GGUF metadata.
func inferConfigFromGGUF(gr *gguf.GGUFReader) map[string]interface{} {
	cfg := make(map[string]interface{})
	meta := gr.Metadata()

	// Map GGUF metadata keys → HuggingFace config keys
	ggufToHF := map[string]string{
		"llama.embedding_length":                    "hidden_size",
		"llama.block_count":                         "num_hidden_layers",
		"llama.attention.head_count":                "num_attention_heads",
		"llama.attention.head_count_kv":             "num_key_value_heads",
		"llama.feed_forward_length":                 "intermediate_size",
		"llama.vocab_size":                          "vocab_size",
		"llama.context_length":                      "max_position_embeddings",
		"llama.rope.freq_base":                      "rope_theta",
		"llama.attention.layer_norm_rms_epsilon":    "rms_norm_eps",
		// Qwen/generic prefixes
		"qwen2.embedding_length":                    "hidden_size",
		"qwen2.block_count":                         "num_hidden_layers",
		"qwen2.attention.head_count":                "num_attention_heads",
		"qwen2.attention.head_count_kv":             "num_key_value_heads",
		"qwen2.feed_forward_length":                 "intermediate_size",
	}

	for ggufKey, hfKey := range ggufToHF {
		if v, ok := meta[ggufKey]; ok {
			switch val := v.(type) {
			case uint32:
				cfg[hfKey] = float64(val)
			case float32:
				cfg[hfKey] = float64(val)
			case uint64:
				cfg[hfKey] = float64(val)
			case int32:
				cfg[hfKey] = float64(val)
			}
		}
	}

	if _, ok := cfg["hidden_act"]; !ok {
		cfg["hidden_act"] = "silu"
	}

	return cfg
}

// inferConfigFromTensors guesses model config from tensor names and shapes.
// Used when no config.json is available.
func inferConfigFromTensors(st *gguf.SafeTensors) map[string]interface{} {
	cfg := make(map[string]interface{})

	// Count layers
	maxLayer := -1
	for _, name := range st.TensorNames {
		if strings.HasPrefix(name, "model.layers.") {
			parts := strings.Split(name, ".")
			if len(parts) >= 3 {
				var n int
				fmt.Sscanf(parts[2], "%d", &n)
				if n > maxLayer {
					maxLayer = n
				}
			}
		}
	}
	if maxLayer >= 0 {
		cfg["num_hidden_layers"] = float64(maxLayer + 1)
	}

	// Infer dim from embed_tokens or layer 0 q_proj
	if info, err := st.GetInfo("model.embed_tokens.weight"); err == nil && info != nil {
		if len(info.Shape) == 2 {
			cfg["vocab_size"] = float64(info.Shape[0])
			cfg["hidden_size"] = float64(info.Shape[1])
		}
	}
	if info, err := st.GetInfo("model.layers.0.self_attn.q_proj.weight"); err == nil && info != nil {
		if len(info.Shape) == 2 {
			dim := info.Shape[0]
			cfg["hidden_size"] = float64(dim)
		}
	}
	if info, err := st.GetInfo("model.layers.0.self_attn.k_proj.weight"); err == nil && info != nil {
		if len(info.Shape) == 2 {
			kvDim := info.Shape[0]
			dim := info.Shape[1]
			cfg["hidden_size"] = float64(dim)
			// Infer heads: assume headDim = 64 or 128
			for _, hd := range []int{128, 64, 32} {
				if dim%hd == 0 {
					cfg["num_attention_heads"] = float64(dim / hd)
					cfg["num_key_value_heads"] = float64(kvDim / hd)
					break
				}
			}
		}
	}
	if info, err := st.GetInfo("model.layers.0.mlp.gate_proj.weight"); err == nil && info != nil {
		if len(info.Shape) == 2 {
			cfg["intermediate_size"] = float64(info.Shape[0])
		}
	}

	dim := configInt(cfg, "hidden_size", 0)
	if dim > 0 {
		cfg["max_position_embeddings"] = float64(2048)
		cfg["rope_theta"] = float64(10000.0)
		cfg["rms_norm_eps"] = float64(1e-6)
		cfg["hidden_act"] = "silu"
	}

	// Also try GGUF-style tensor names
	if maxLayer < 0 {
		for _, name := range st.TensorNames {
			if strings.HasPrefix(name, "blk.") {
				parts := strings.Split(name, ".")
				if len(parts) >= 2 {
					var n int
					fmt.Sscanf(parts[1], "%d", &n)
					if n > maxLayer {
						maxLayer = n
					}
				}
			}
		}
		if maxLayer >= 0 {
			cfg["num_hidden_layers"] = float64(maxLayer + 1)
		}
	}

	return cfg
}

// ReadTensorFloat32 reads a tensor by HuggingFace name.
// Auto-translates names for GGUF. Dequantizes Q8/Q4/F16 to float32.
func (m *ModelSource) ReadTensorFloat32(name string) ([]float32, error) {
	switch m.format {
	case "safetensors":
		data, _, err := m.st.ReadTensorFloat32(name)
		return data, err
	case "gguf":
		ggufName := name
		if mapped, ok := m.nameMap[name]; ok {
			ggufName = mapped
		}
		data, _, err := m.gr.ReadTensorFloat32(ggufName)
		if err != nil {
			data, _, err = m.gr.ReadTensorFloat32(name)
		}
		return data, err
	default:
		return nil, fmt.Errorf("unsupported format: %s", m.format)
	}
}

// ReadTensorFloat32Full returns data, shape info, and error — matches SafeTensors signature.
func (m *ModelSource) ReadTensorFloat32Full(name string) ([]float32, *gguf.TensorInfo, error) {
	switch m.format {
	case "safetensors":
		return m.st.ReadTensorFloat32(name)
	case "gguf":
		ggufName := name
		if mapped, ok := m.nameMap[name]; ok {
			ggufName = mapped
		}
		data, shape, err := m.gr.ReadTensorFloat32(ggufName)
		if err != nil {
			data, shape, err = m.gr.ReadTensorFloat32(name)
		}
		if err != nil {
			return nil, nil, err
		}
		return data, &gguf.TensorInfo{Shape: shape}, nil
	default:
		return nil, nil, fmt.Errorf("unsupported format: %s", m.format)
	}
}

// DequantAWQ delegates to SafeTensors AWQ dequantization. Returns nil for GGUF.
func (m *ModelSource) DequantAWQ(prefix string, groupSize int) ([]float32, int, int, error) {
	if m.st != nil {
		return m.st.DequantAWQ(prefix, groupSize)
	}
	return nil, 0, 0, fmt.Errorf("AWQ dequant not supported for %s format", m.format)
}

// TensorNames returns all tensor names in the model.
func (m *ModelSource) TensorNames() []string {
	if m.st != nil {
		return m.st.TensorNames
	}
	if m.gr != nil {
		return m.gr.TensorNames()
	}
	return nil
}

// HasTensor checks if a tensor exists by HF name.
func (m *ModelSource) HasTensor(name string) bool {
	switch m.format {
	case "safetensors":
		return m.st.HasTensor(name)
	case "gguf":
		ggufName := name
		if mapped, ok := m.nameMap[name]; ok {
			ggufName = mapped
		}
		return m.gr.HasTensor(ggufName) || m.gr.HasTensor(name)
	default:
		return false
	}
}

func (m *ModelSource) ST() *gguf.SafeTensors              { return m.st }
func (m *ModelSource) Config() map[string]interface{} { return m.config }
func (m *ModelSource) Format() string                 { return m.format }
func (m *ModelSource) Dir() string                    { return m.dir }
func (m *ModelSource) NLayers() int                   { return m.nLayers }

func (m *ModelSource) ConfigInt(key string, def int) int {
	return configInt(m.config, key, def)
}
func (m *ModelSource) ConfigFloat(key string, def float64) float64 {
	return configFloat(m.config, key, def)
}
func (m *ModelSource) ConfigString(key string, def string) string {
	if v, ok := m.config[key].(string); ok {
		return v
	}
	return def
}

func (m *ModelSource) Close() {
	if m.gr != nil {
		m.gr.Close()
	}
	if m.tmpDir != "" {
		os.RemoveAll(m.tmpDir)
	}
}

// helpers

func hasSafeTensors(dir string) bool {
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".safetensors") || e.Name() == "model.safetensors.index.json" {
			return true
		}
	}
	return false
}

func findGGUFInDir(dir string) string {
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".gguf") {
			return filepath.Join(dir, e.Name())
		}
	}
	return ""
}

func findFileByExt(dir, ext string) string {
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasSuffix(strings.ToLower(e.Name()), ext) {
			return filepath.Join(dir, e.Name())
		}
	}
	return ""
}

func loadConfig(dir string) map[string]interface{} {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil
	}
	var cfg map[string]interface{}
	json.Unmarshal(data, &cfg)
	return cfg
}

func configInt(cfg map[string]interface{}, key string, def int) int {
	if cfg == nil {
		return def
	}
	if v, ok := cfg[key].(float64); ok {
		return int(v)
	}
	return def
}

func configFloat(cfg map[string]interface{}, key string, def float64) float64 {
	if cfg == nil {
		return def
	}
	if v, ok := cfg[key].(float64); ok {
		return v
	}
	return def
}

// ResolveAndOpen resolves a model name and opens it.
func ResolveAndOpen(name string) *ModelSource {
	// Direct file path (any format)
	if _, err := os.Stat(name); err == nil {
		m, err := OpenModel(name)
		if err != nil {
			log.Fatalf("open model: %v", err)
		}
		return m
	}

	// Try resolveModel (searches ~/.ai/models, etc.)
	path := resolveModel(name)
	m, err := OpenModel(path)
	if err != nil {
		log.Fatalf("open model %s: %v", path, err)
	}
	return m
}

// EstimateParamCount estimates the parameter count from config.
func (m *ModelSource) EstimateParamCount() int64 {
	dim := int64(m.ConfigInt("hidden_size", 0))
	layers := int64(m.ConfigInt("num_hidden_layers", 0))
	heads := int64(m.ConfigInt("num_attention_heads", 0))
	kvHeads := int64(m.ConfigInt("num_key_value_heads", int(heads)))
	ffnDim := int64(m.ConfigInt("intermediate_size", 0))
	vocabSize := int64(m.ConfigInt("vocab_size", 0))

	if dim == 0 || layers == 0 {
		return 0
	}

	headDim := dim / heads
	kvDim := kvHeads * headDim

	nParams := vocabSize * dim * 2 // embed + lm_head
	for l := int64(0); l < layers; l++ {
		nParams += dim*dim*2 + kvDim*dim*2 + ffnDim*dim*3 + dim*2
	}
	return nParams
}

// SizeCategory returns "small" (<1B), "medium" (1-4B), "large" (4-13B), "xl" (13B+).
func (m *ModelSource) SizeCategory() string {
	n := m.EstimateParamCount()
	switch {
	case n > 13_000_000_000:
		return "xl"
	case n > 4_000_000_000:
		return "large"
	case n > 1_000_000_000:
		return "medium"
	default:
		return "small"
	}
}

// FormatParamCount returns a human-readable parameter count.
func (m *ModelSource) FormatParamCount() string {
	n := float64(m.EstimateParamCount())
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1fB", n/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.1fM", n/1e6)
	case n >= 1e3:
		return fmt.Sprintf("%.1fK", n/1e3)
	default:
		return fmt.Sprintf("%.0f", math.Max(n, 0))
	}
}
