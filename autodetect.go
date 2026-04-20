package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/open-ai-org/mongoose"
)

// ModelProfile contains everything mongoose needs to run a model.
// Auto-detected from config.json + hardware probing. Zero user config.
type ModelProfile struct {
	// Model identity
	Name       string // human-readable name
	Family     string // qwen2, llama, mistral, phi, gemma, etc.
	ParamCount int64  // total parameters

	// Architecture (from config.json)
	Dim       int
	Layers    int
	Heads     int
	KVHeads   int
	FFNDim    int
	VocabSize int
	HeadDim   int
	KVDim     int
	RopeTheta float64
	NormEps   float32
	MaxSeqLen int

	// Hardware (auto-probed)
	GPUName     string
	VRAM        int64  // bytes
	CPUCores    int
	SystemRAM   int64  // bytes (0 if unknown)
	HasCUDA     bool
	HasMetal    bool
	HasXe       bool

	// Recommended settings (computed)
	Precision   string // "fp32", "fp16", "int8"
	FitsInVRAM  bool
	LoRAFit     bool   // Q8+LoRA fits in VRAM
	LoRARank    int
	SeqLen      int
	LR          float64
	Steps       int
	BatchSize   int    // future use
}

// AutoDetect probes a model directory and available hardware.
// Returns a complete ModelProfile with recommended settings.
// This is the "it just works" engine — the user provides a model path
// and mongoose figures out everything else.
func AutoDetect(modelDir string) *ModelProfile {
	p := &ModelProfile{
		Name:      filepath.Base(modelDir),
		RopeTheta: 10000.0,
		NormEps:   1e-6,
		MaxSeqLen: 2048,
	}

	// --- Model architecture ---
	configPath := filepath.Join(modelDir, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		log.Printf("[autodetect] no config.json in %s", modelDir)
		return p
	}
	var cfg map[string]interface{}
	json.Unmarshal(configData, &cfg)

	// Detect model family
	if arch, ok := cfg["architectures"].([]interface{}); ok && len(arch) > 0 {
		archStr := arch[0].(string)
		p.Family = detectFamily(archStr)
	}
	if mt, ok := cfg["model_type"].(string); ok && p.Family == "" {
		p.Family = mt
	}

	// Read dimensions
	if v, ok := cfg["hidden_size"].(float64); ok { p.Dim = int(v) }
	if v, ok := cfg["num_hidden_layers"].(float64); ok { p.Layers = int(v) }
	if v, ok := cfg["num_attention_heads"].(float64); ok { p.Heads = int(v) }
	if v, ok := cfg["num_key_value_heads"].(float64); ok {
		p.KVHeads = int(v)
	} else {
		p.KVHeads = p.Heads // MHA default
	}
	if v, ok := cfg["intermediate_size"].(float64); ok { p.FFNDim = int(v) }
	if v, ok := cfg["vocab_size"].(float64); ok { p.VocabSize = int(v) }
	if v, ok := cfg["rope_theta"].(float64); ok { p.RopeTheta = v }
	if v, ok := cfg["rms_norm_eps"].(float64); ok { p.NormEps = float32(v) }
	if v, ok := cfg["max_position_embeddings"].(float64); ok { p.MaxSeqLen = int(v) }

	if p.Heads > 0 {
		p.HeadDim = p.Dim / p.Heads
	}
	p.KVDim = p.KVHeads * p.HeadDim

	// Count parameters
	p.ParamCount = countParams(p)

	// --- Hardware probe ---
	p.CPUCores = runtime.NumCPU()

	cuda := mongoose.NewCUDA()
	if cuda != nil {
		p.HasCUDA = true
		p.GPUName = cuda.Name()
		p.VRAM = int64(cuda.VRAM())
	}
	// Metal detection
	if runtime.GOOS == "darwin" {
		p.HasMetal = true
		if p.GPUName == "" {
			p.GPUName = "Apple Silicon"
		}
	}

	// --- Compute recommendations ---
	p.computeRecommendations()

	return p
}

func (p *ModelProfile) computeRecommendations() {
	modelFP32 := p.ParamCount * 4
	modelFP16 := p.ParamCount * 2
	modelINT8 := p.ParamCount * 1

	// Precision: what fits in VRAM?
	if p.VRAM > 0 {
		vram := p.VRAM
		// Reserve 20% for activations/KV cache/overhead
		usable := int64(float64(vram) * 0.80)

		if modelFP32 < usable {
			p.Precision = "fp32"
			p.FitsInVRAM = true
		} else if modelFP16 < usable {
			p.Precision = "fp16"
			p.FitsInVRAM = true
		} else if modelINT8 < usable {
			p.Precision = "int8"
			p.FitsInVRAM = true
		} else {
			p.Precision = "int8"
			p.FitsInVRAM = false
		}

		// Q8+LoRA: INT8 base + FP32 embed + FP32 LoRA + activations
		embedBytes := int64(p.VocabSize) * int64(p.Dim) * 4
		loraBytes := int64(p.Layers) * 7 * int64(16) * int64(p.Dim) * 4 * 2 // 7 projections, rank 16, A+B
		activationBytes := int64(p.Layers) * int64(128) * int64(p.Dim) * 4 * 20 // ~20 cached tensors per layer
		q8Total := modelINT8 + embedBytes + loraBytes + activationBytes
		p.LoRAFit = q8Total < vram
	} else {
		p.Precision = "fp32"
		p.FitsInVRAM = false
	}

	// LoRA rank: scale with model size
	if p.ParamCount > 10e9 {
		p.LoRARank = 32
	} else if p.ParamCount > 3e9 {
		p.LoRARank = 16
	} else {
		p.LoRARank = 8
	}

	// Sequence length: scale with VRAM headroom
	if p.VRAM > 0 {
		if p.VRAM > 48*1024*1024*1024 {
			p.SeqLen = 512
		} else if p.VRAM > 24*1024*1024*1024 {
			p.SeqLen = 256
		} else if p.VRAM > 16*1024*1024*1024 {
			p.SeqLen = 128
		} else {
			p.SeqLen = 64
		}
	} else {
		p.SeqLen = 64
	}

	// Learning rate: lower for larger models
	if p.ParamCount > 10e9 {
		p.LR = 1e-4
	} else if p.ParamCount > 1e9 {
		p.LR = 2e-4
	} else {
		p.LR = 5e-4
	}

	// Steps: estimate based on typical LoRA convergence
	// ~500 steps for small models, ~5000 for 14B+
	if p.ParamCount > 10e9 {
		p.Steps = 5000
	} else if p.ParamCount > 3e9 {
		p.Steps = 2000
	} else if p.ParamCount > 1e9 {
		p.Steps = 1000
	} else {
		p.Steps = 500
	}
}

func (p *ModelProfile) Print() {
	fmt.Printf("Model: %s (%s)\n", p.Name, p.Family)
	fmt.Printf("  params:    %s\n", formatParams(int(p.ParamCount)))
	fmt.Printf("  arch:      dim=%d layers=%d heads=%d kv=%d ffn=%d vocab=%d\n",
		p.Dim, p.Layers, p.Heads, p.KVHeads, p.FFNDim, p.VocabSize)
	fmt.Printf("  size:      %.1f GB (FP32) / %.1f GB (FP16) / %.1f GB (INT8)\n",
		float64(p.ParamCount)*4/(1024*1024*1024),
		float64(p.ParamCount)*2/(1024*1024*1024),
		float64(p.ParamCount)*1/(1024*1024*1024))
	fmt.Println()

	fmt.Printf("Hardware: %s\n", p.GPUName)
	if p.VRAM > 0 {
		fmt.Printf("  vram:      %.0f GB\n", float64(p.VRAM)/(1024*1024*1024))
	}
	fmt.Printf("  cpu:       %d cores\n", p.CPUCores)
	fmt.Println()

	fmt.Printf("Recommended:\n")
	fmt.Printf("  precision: %s\n", p.Precision)
	fmt.Printf("  fits vram: %v\n", p.FitsInVRAM)
	fmt.Printf("  lora fit:  %v (Q8+LoRA rank-%d)\n", p.LoRAFit, p.LoRARank)
	fmt.Printf("  seq_len:   %d\n", p.SeqLen)
	fmt.Printf("  lr:        %.0e\n", p.LR)
	fmt.Printf("  steps:     %d\n", p.Steps)
}

// ToTrainCfg converts auto-detected settings into a trainCfg.
func (p *ModelProfile) ToTrainCfg() *trainCfg {
	return &trainCfg{
		dim:         p.Dim,
		nHeads:      p.Heads,
		nKVHeads:    p.KVHeads,
		nLayers:     p.Layers,
		ffnDim:      p.FFNDim,
		seqLen:      p.SeqLen,
		vocabSize:   p.VocabSize,
		ropeTheta:   p.RopeTheta,
		headDim:     p.HeadDim,
		kvDim:       p.KVDim,
		lr:          float32(p.LR),
		beta1:       0.9,
		beta2:       0.999,
		adamEps:     1e-8,
		weightDecay: 0.01,
		gradClip:    1.0,
		steps:       p.Steps,
		logEvery:    max(p.Steps/20, 1), // ~20 log lines per run
	}
}

func detectFamily(arch string) string {
	arch = strings.ToLower(arch)
	families := map[string]string{
		"qwen2":   "qwen2",
		"llama":   "llama",
		"mistral": "mistral",
		"phi":     "phi",
		"gemma":   "gemma",
		"gpt2":    "gpt2",
		"falcon":  "falcon",
		"mpt":     "mpt",
		"bloom":   "bloom",
		"opt":     "opt",
		"stablelm": "stablelm",
		"starcoder": "starcoder",
		"codellama": "llama",
		"deepseek":  "deepseek",
		"yi":        "yi",
		"internlm":  "internlm",
		"baichuan":  "baichuan",
		"chatglm":   "chatglm",
	}
	for key, family := range families {
		if strings.Contains(arch, key) {
			return family
		}
	}
	return arch
}

func countParams(p *ModelProfile) int64 {
	if p.Dim == 0 || p.Layers == 0 {
		return 0
	}
	var n int64
	// Embedding + final norm
	n += int64(p.VocabSize) * int64(p.Dim)
	n += int64(p.Dim)
	// Per layer
	for range p.Layers {
		n += int64(p.Dim) // norm1
		n += int64(p.Dim) * int64(p.Dim) // wq
		n += int64(p.Dim) // bq (if present)
		n += int64(p.KVDim) * int64(p.Dim) // wk
		n += int64(p.KVDim) // bk
		n += int64(p.KVDim) * int64(p.Dim) // wv
		n += int64(p.KVDim) // bv
		n += int64(p.Dim) * int64(p.Dim) // wo
		n += int64(p.Dim) // norm2
		n += int64(p.FFNDim) * int64(p.Dim) // gate
		n += int64(p.FFNDim) * int64(p.Dim) // up
		n += int64(p.Dim) * int64(p.FFNDim) // down
	}
	return n
}
