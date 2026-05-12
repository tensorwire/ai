//go:build darwin && cgo

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/mongoose"
	"github.com/tensorwire/tokenizer"
)

func cmdInferSQ4Metal(path, prompt string, metal *mongoose.Metal, te mongoose.TensorEngine,
	cfg map[string]interface{},
	dim, nLayers, heads, kvHeads, ffnDim, vocabSize, headDim, kvDim, attnDim, maxSeq int,
	ropeTheta, normEps float32) {

	if !mongoose.SQ4MetalInit() {
		log.Fatal("[SQ4] sq4_matvec.metallib not loaded")
	}

	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	// Load SQ4 metadata
	metaBytes, err := os.ReadFile(filepath.Join(path, "sq4_meta.json"))
	if err != nil {
		log.Fatalf("read sq4_meta.json: %v", err)
	}
	var meta struct {
		Tensors []struct {
			Name         string `json:"name"`
			Rows         int    `json:"rows"`
			Cols         int    `json:"cols"`
			PackedOffset int    `json:"packed_offset"`
			PackedBytes  int    `json:"packed_bytes"`
			BandsOffset  int    `json:"bands_offset"`
			OutlierStart int    `json:"outlier_start"`
			OutlierCount int    `json:"outlier_count"`
		} `json:"tensors"`
	}
	json.Unmarshal(metaBytes, &meta)

	packedAll, _ := os.ReadFile(filepath.Join(path, "sq4_packed.bin"))
	bandsAll, _ := os.ReadFile(filepath.Join(path, "sq4_bands.bin"))
	outlierIdxAll, _ := os.ReadFile(filepath.Join(path, "sq4_outlier_idx.bin"))
	outlierValAll, _ := os.ReadFile(filepath.Join(path, "sq4_outlier_val.bin"))

	fpST, err := gguf.OpenSafeTensors(filepath.Join(path, "fp32.safetensors"))
	if err != nil {
		log.Fatalf("open fp32.safetensors: %v", err)
	}

	tensorByName := map[string]int{}
	for i, t := range meta.Tensors {
		tensorByName[t.Name] = i
	}

	// Build standalone SQ4 fused inference pipeline
	sq4Infer := mongoose.NewSQ4InferMetal(metal)
	ret := sq4Infer.Build(dim, kvDim, headDim, heads, kvHeads, ffnDim, vocabSize, nLayers, maxSeq,
		ropeTheta, normEps)
	if ret != 0 {
		log.Fatal("[SQ4] SQ4InferBuild failed")
	}
	fmt.Printf("  engine:  Metal SQ4 fused (%s)\n", metal.Name())

	// Allocate slabs and bulk-upload all SQ4 data
	totalOutliers := 0
	totalBandsFloats := 0
	for _, t := range meta.Tensors {
		totalOutliers += t.OutlierCount
		totalBandsFloats += 8
	}
	sq4Infer.AllocSlabs(len(packedAll), totalBandsFloats, totalOutliers)

	// Bulk upload packed data (already contiguous in file)
	sq4Infer.UploadPacked(0, packedAll)

	// Upload bands (convert from raw bytes to float32)
	allBands := make([]float32, totalBandsFloats)
	for i := range allBands {
		allBands[i] = math.Float32frombits(binary.LittleEndian.Uint32(bandsAll[i*4:]))
	}
	sq4Infer.UploadBands(0, allBands)

	// Upload outliers (convert from raw bytes)
	if totalOutliers > 0 {
		allOIdx := make([]uint32, totalOutliers)
		allOVal := make([]float32, totalOutliers)
		for i := 0; i < totalOutliers; i++ {
			allOIdx[i] = binary.LittleEndian.Uint32(outlierIdxAll[i*4:])
			allOVal[i] = math.Float32frombits(binary.LittleEndian.Uint32(outlierValAll[i*4:]))
		}
		sq4Infer.UploadOutliers(0, allOIdx, allOVal)
	}

	// Set weight descriptors (offsets into slabs)
	setDesc := func(wi int, name string) {
		idx, ok := tensorByName[name]
		if !ok {
			return
		}
		t := meta.Tensors[idx]
		sq4Infer.SetSQ4Desc(wi, t.PackedOffset, t.PackedBytes, t.BandsOffset, t.OutlierStart, t.OutlierCount, t.Rows, t.Cols)
	}

	loadFP32 := func(name string) []float32 {
		data, _, err := fpST.ReadTensorFloat32(name)
		if err != nil {
			return nil
		}
		return data
	}

	// Upload weights in fused pipeline order: 12 slots per layer + finalNorm + lmHead
	wi := 0
	for l := 0; l < nLayers; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)

		// slot 0: norm1 (FP32)
		norm1 := loadFP32(pfx + "input_layernorm.weight")
		if norm1 != nil {
			sq4Infer.SetFP32(wi,norm1)
		}
		wi++

		// slots 1-3: Q/K/V weights (SQ4)
		setDesc(wi, pfx+"self_attn.q_proj.weight"); wi++
		setDesc(wi, pfx+"self_attn.k_proj.weight"); wi++
		setDesc(wi, pfx+"self_attn.v_proj.weight"); wi++

		// slots 4-6: Q/K/V biases (FP32)
		for _, bn := range []string{"self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias"} {
			bias := loadFP32(pfx + bn)
			if bias == nil {
				sz := dim
				if strings.Contains(bn, "k_proj") || strings.Contains(bn, "v_proj") {
					sz = kvDim
				}
				bias = make([]float32, sz)
			}
			sq4Infer.SetFP32(wi,bias)
			wi++
		}

		// slot 7: O weight (SQ4)
		setDesc(wi, pfx+"self_attn.o_proj.weight"); wi++

		// slot 8: norm2 (FP32)
		norm2 := loadFP32(pfx + "post_attention_layernorm.weight")
		if norm2 != nil {
			sq4Infer.SetFP32(wi,norm2)
		}
		wi++

		// slots 9-11: gate/up/down weights (SQ4)
		setDesc(wi, pfx+"mlp.gate_proj.weight"); wi++
		setDesc(wi, pfx+"mlp.up_proj.weight"); wi++
		setDesc(wi, pfx+"mlp.down_proj.weight"); wi++

		if (l+1)%10 == 0 || l == nLayers-1 {
			log.Printf("[SQ4] loaded layer %d/%d", l+1, nLayers)
		}
	}

	// finalNorm (FP32)
	fnorm := loadFP32("model.norm.weight")
	if fnorm != nil {
		sq4Infer.SetFP32(wi,fnorm)
	}
	wi++

	// lm_head — SQ4. For tied weights, reuse embed's SQ4 data.
	if _, ok := tensorByName["lm_head.weight"]; ok {
		setDesc(wi, "lm_head.weight")
	} else {
		setDesc(wi, "model.embed_tokens.weight")
	}
	wi++

	// Dequant embed to FP32 for token lookup (CPU-side)
	embedIdx := tensorByName["model.embed_tokens.weight"]
	embedT := meta.Tensors[embedIdx]
	embedBandsStart := embedT.BandsOffset * 4
	var embedBands [8]float32
	for b := 0; b < 8; b++ {
		embedBands[b] = math.Float32frombits(binary.LittleEndian.Uint32(bandsAll[embedBandsStart+b*4:]))
	}
	embedData := make([]float32, vocabSize*dim)
	halfCols := dim / 2
	for r := 0; r < vocabSize; r++ {
		for c := 0; c < dim; c++ {
			byteIdx := embedT.PackedOffset + r*halfCols + c/2
			shift := uint((c & 1) * 4)
			nibble := (packedAll[byteIdx] >> shift) & 0x0F
			band := nibble & 0x07
			val := embedBands[band]
			if nibble&0x08 != 0 {
				val = -val
			}
			embedData[r*dim+c] = val
		}
	}
	for i := 0; i < embedT.OutlierCount; i++ {
		off := (embedT.OutlierStart + i) * 4
		flat := binary.LittleEndian.Uint32(outlierIdxAll[off:])
		oVal := math.Float32frombits(binary.LittleEndian.Uint32(outlierValAll[off:]))
		if int(flat) < len(embedData) {
			embedData[flat] = oVal
		}
	}

	// RoPE tables
	halfHead := headDim / 2
	cosTab := make([]float32, maxSeq*halfHead)
	sinTab := make([]float32, maxSeq*halfHead)
	for pos := 0; pos < maxSeq; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(float64(ropeTheta), float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			cosTab[pos*halfHead+j] = float32(math.Cos(angle))
			sinTab[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}

	fmt.Printf("  model:   %s\n", path)
	fmt.Printf("  arch:    dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	fmt.Printf("  weights: %d slots loaded\n", wi)
	fmt.Println()

	tokens := tok.Encode(prompt)
	if len(tokens) == 0 {
		tokens = []int{1}
	}

	fHidden := make([]float32, dim)
	fLogits := make([]float32, vocabSize)

	forward := func(tokenID, pos int) []float32 {
		tokOff := tokenID * dim
		if tokOff+dim > len(embedData) {
			return nil
		}
		copy(fHidden, embedData[tokOff:tokOff+dim])
		cosSlice := cosTab[pos*halfHead : pos*halfHead+halfHead]
		sinSlice := sinTab[pos*halfHead : pos*halfHead+halfHead]
		sq4Infer.Step(fHidden, cosSlice, sinSlice, pos, fLogits)
		return fLogits
	}

	fmt.Print("Prefilling... ")
	t0 := time.Now()
	var logits []float32
	for i, tid := range tokens {
		logits = forward(tid, i)
	}
	prefillTime := time.Since(t0)
	fmt.Printf("done (%d tokens, %.1fs)\n", len(tokens), prefillTime.Seconds())

	nextToken := sampleTopK(logits, 0.7, 40)
	maxTokens := 200
	pos := len(tokens)

	fmt.Printf("\nGenerating (max %d tokens):\n", maxTokens)
	fmt.Print(prompt)

	genStart := time.Now()
	generated := 0
	for i := 0; i < maxTokens; i++ {
		text := tok.Decode([]int{nextToken})
		fmt.Print(text)

		if nextToken == 0 {
			break
		}
		if strings.HasSuffix(text, "<|endoftext|>") || strings.HasSuffix(text, "<|im_end|>") {
			break
		}

		logits = forward(nextToken, pos)
		pos++
		generated++
		nextToken = sampleTopK(logits, 0.7, 40)
	}

	genTime := time.Since(genStart)
	fmt.Printf("\n\n--- %d tokens in %.1fs (%.1f tok/s) ---\n",
		generated, genTime.Seconds(), float64(generated)/genTime.Seconds())
}
