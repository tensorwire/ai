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
	"unsafe"

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/mongoose"
	"github.com/tensorwire/tokenizer"
)

type sq4CUDAWeight struct {
	packed   *mongoose.Tensor // nibble-packed on GPU (swizzled if TF32 available)
	bands    *mongoose.Tensor // 8 float32 band means on GPU
	oIdx     *mongoose.Tensor // outlier flat indices
	oVal     *mongoose.Tensor // outlier float32 values
	oApprox  []float32        // host-side band approximations for outliers (for TF32 path)
	oCnt     int
	rows     int
	cols     int
	swizzled bool
}

func sq4CUDAMatvec(act unsafe.Pointer, w *sq4CUDAWeight, out unsafe.Pointer) {
	if w.oCnt > 0 && w.oIdx != nil {
		mongoose.KSQ4MatvecFused(w.packed.DevicePtr(), w.bands.DevicePtr(),
			act, out, w.rows, w.cols,
			w.oIdx.DevicePtr(), w.oVal.DevicePtr(), w.oCnt)
	} else {
		mongoose.KSQ4Matvec(w.packed.DevicePtr(), w.bands.DevicePtr(),
			act, out, w.rows, w.cols)
	}
}

func cmdInferSQ4CUDA(path, prompt string, cuda *mongoose.CUDA, te mongoose.TensorEngine,
	cfg map[string]interface{},
	dim, nLayers, heads, kvHeads, ffnDim, vocabSize, headDim, kvDim, attnDim, maxSeq int,
	ropeTheta, normEps float32) {

	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	metaBytes, _ := os.ReadFile(filepath.Join(path, "sq4_meta.json"))
	var meta struct {
		Tensors []struct {
			Name         string `json:"name"`
			Rows         int    `json:"rows"`
			Cols         int    `json:"cols"`
			MagOffset    int    `json:"mag_offset"`
			MagBytes     int    `json:"mag_bytes"`
			SignOffset   int    `json:"sign_offset"`
			SignBytes    int    `json:"sign_bytes"`
			BandsOffset  int    `json:"bands_offset"`
			OutlierStart int    `json:"outlier_start"`
			OutlierCount int    `json:"outlier_count"`
		} `json:"tensors"`
	}
	json.Unmarshal(metaBytes, &meta)

	magAll, _ := os.ReadFile(filepath.Join(path, "sq4_magnitude.bin"))
	signAll, _ := os.ReadFile(filepath.Join(path, "sq4_sign.bin"))
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

	useTF32 := mongoose.HasSQ4TF32() && os.Getenv("SQ4_NO_TF32") == ""
	if useTF32 {
		log.Println("[SQ4-CUDA] TF32 tensor core path available — tile-swizzling weights")
	} else if mongoose.HasSQ4TF32() {
		log.Println("[SQ4-CUDA] TF32 available but disabled via SQ4_NO_TF32")
	}

	// Pack mag+sign into nibbles, optionally tile-swizzle, and upload per tensor
	packAndUpload := func(name string) *sq4CUDAWeight {
		idx, ok := tensorByName[name]
		if !ok {
			return nil
		}
		t := meta.Tensors[idx]
		n := t.Rows * t.Cols
		packed := make([]byte, n/2)
		magSlice := magAll[t.MagOffset : t.MagOffset+t.MagBytes]
		signSlice := signAll[t.SignOffset : t.SignOffset+t.SignBytes]
		for i := 0; i < n; i++ {
			bitPos := i * 3
			byteIdx := bitPos / 8
			bitOff := uint(bitPos % 8)
			raw := magSlice[byteIdx] >> bitOff
			if bitOff > 5 && byteIdx+1 < len(magSlice) {
				raw |= magSlice[byteIdx+1] << (8 - bitOff)
			}
			band := raw & 0x07
			signBit := (signSlice[i/8] >> uint(i%8)) & 1
			nibble := (signBit << 3) | band
			shift := uint((i & 1) * 4)
			packed[i/2] |= nibble << shift
		}

		bandsStart := t.BandsOffset * 4
		var bands [8]float32
		for b := 0; b < 8; b++ {
			bands[b] = math.Float32frombits(binary.LittleEndian.Uint32(bandsAll[bandsStart+b*4:]))
		}

		// Pre-compute outlier band approximations from row-major data (before swizzle)
		var oApprox []float32
		var oIdx []uint32
		var oVal []float32
		if t.OutlierCount > 0 {
			oIdx = make([]uint32, t.OutlierCount)
			oVal = make([]float32, t.OutlierCount)
			oApprox = make([]float32, t.OutlierCount)
			var table16 [16]float32
			for b := 0; b < 8; b++ {
				table16[b] = bands[b]
				table16[b+8] = -bands[b]
			}
			for i := 0; i < t.OutlierCount; i++ {
				off := (t.OutlierStart + i) * 4
				oIdx[i] = binary.LittleEndian.Uint32(outlierIdxAll[off:])
				oVal[i] = math.Float32frombits(binary.LittleEndian.Uint32(outlierValAll[off:]))
				flat := oIdx[i]
				byteIdx := flat / 2
				shift := (flat & 1) * 4
				nibble := (packed[byteIdx] >> shift) & 0x0F
				oApprox[i] = table16[nibble]
			}
		}

		// Tile-swizzle if TF32 available and dimensions are tile-aligned
		uploadPacked := packed
		swizzled := false
		if useTF32 && t.Cols%8 == 0 && t.Rows%16 == 0 {
			swizzled = true
			swizzBuf := make([]byte, len(packed))
			mongoose.SQ4SwizzleForTiles(packed, swizzBuf, t.Rows, t.Cols)
			uploadPacked = swizzBuf
		}

		packedFloats := (len(uploadPacked) + 3) / 4
		packedBuf := te.Zeros([]int{packedFloats})
		cuda.UploadRawBytes(packedBuf, unsafe.Pointer(&uploadPacked[0]), len(uploadPacked))
		bandsBuf := te.FromHost(bands[:], []int{8})

		w := &sq4CUDAWeight{
			packed:   packedBuf,
			bands:    bandsBuf,
			oCnt:     t.OutlierCount,
			rows:     t.Rows,
			cols:     t.Cols,
			swizzled: swizzled,
			oApprox:  oApprox,
		}

		if t.OutlierCount > 0 {
			w.oIdx = te.FromHost(oVal, []int{t.OutlierCount})
			cuda.UploadRawBytes(w.oIdx, unsafe.Pointer(&oIdx[0]), t.OutlierCount*4)
			w.oVal = te.FromHost(oVal, []int{t.OutlierCount})
		}

		return w
	}

	loadFP32 := func(name string) *mongoose.Tensor {
		data, _, err := fpST.ReadTensorFloat32(name)
		if err != nil || data == nil {
			return nil
		}
		return te.FromHost(data, []int{len(data)})
	}

	// Load all weights
	type layer struct {
		norm1, norm2     *mongoose.Tensor
		bq, bk, bv      *mongoose.Tensor
		wq, wk, wv, wo  *sq4CUDAWeight
		gate, up, down   *sq4CUDAWeight
	}
	layers := make([]layer, nLayers)
	for l := 0; l < nLayers; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		layers[l].norm1 = loadFP32(pfx + "input_layernorm.weight")
		layers[l].wq = packAndUpload(pfx + "self_attn.q_proj.weight")
		layers[l].wk = packAndUpload(pfx + "self_attn.k_proj.weight")
		layers[l].wv = packAndUpload(pfx + "self_attn.v_proj.weight")
		layers[l].bq = loadFP32(pfx + "self_attn.q_proj.bias")
		layers[l].bk = loadFP32(pfx + "self_attn.k_proj.bias")
		layers[l].bv = loadFP32(pfx + "self_attn.v_proj.bias")
		layers[l].wo = packAndUpload(pfx + "self_attn.o_proj.weight")
		layers[l].norm2 = loadFP32(pfx + "post_attention_layernorm.weight")
		layers[l].gate = packAndUpload(pfx + "mlp.gate_proj.weight")
		layers[l].up = packAndUpload(pfx + "mlp.up_proj.weight")
		layers[l].down = packAndUpload(pfx + "mlp.down_proj.weight")
		if (l+1)%10 == 0 || l == nLayers-1 {
			log.Printf("[SQ4-CUDA] loaded layer %d/%d", l+1, nLayers)
		}
	}
	finalNorm := loadFP32("model.norm.weight")
	lmHead := packAndUpload("lm_head.weight")
	if lmHead == nil {
		lmHead = packAndUpload("model.embed_tokens.weight")
	}

	// Embed dequant → upload to GPU once
	embedIdx := tensorByName["model.embed_tokens.weight"]
	embedT := meta.Tensors[embedIdx]
	embedBandsStart := embedT.BandsOffset * 4
	var embedBands [8]float32
	for b := 0; b < 8; b++ {
		embedBands[b] = math.Float32frombits(binary.LittleEndian.Uint32(bandsAll[embedBandsStart+b*4:]))
	}
	embedData := make([]float32, vocabSize*dim)
	n := vocabSize * dim
	for i := 0; i < n; i++ {
		bitPos := (embedT.MagOffset*8) + i*3
		byteIdx := bitPos / 8
		bitOff := uint(bitPos % 8)
		raw := magAll[byteIdx] >> bitOff
		if bitOff > 5 { raw |= magAll[byteIdx+1] << (8 - bitOff) }
		band := raw & 0x07
		val := embedBands[band]
		signByteIdx := embedT.SignOffset + i/8
		if signAll[signByteIdx]&(1<<uint(i%8)) != 0 { val = -val }
		embedData[i] = val
	}
	for i := 0; i < embedT.OutlierCount; i++ {
		off := (embedT.OutlierStart + i) * 4
		flat := binary.LittleEndian.Uint32(outlierIdxAll[off:])
		oVal := math.Float32frombits(binary.LittleEndian.Uint32(outlierValAll[off:]))
		if int(flat) < len(embedData) { embedData[flat] = oVal }
	}
	gpuEmbed := te.FromHost(embedData, []int{vocabSize * dim})
	embedData = nil

	// RoPE tables on GPU
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
	gpuCos := te.FromHost(cosTab, []int{maxSeq * halfHead})
	gpuSin := te.FromHost(sinTab, []int{maxSeq * halfHead})

	// Cap KV cache to fit VRAM
	if maxSeq > 2048 { maxSeq = 2048 }

	// KV cache (C-side fused pipeline manages scratch, but KV cache is per-layer on Go side)
	type kvCache struct{ k, v *mongoose.Tensor }
	gpuKV := make([]kvCache, nLayers)
	for l := 0; l < nLayers; l++ {
		gpuKV[l].k = te.Zeros([]int{maxSeq * kvDim})
		gpuKV[l].v = te.Zeros([]int{maxSeq * kvDim})
	}

	fmt.Printf("  model:   %s\n", filepath.Base(path))
	fmt.Printf("  arch:    dim=%d heads=%d kv=%d layers=%d ffn=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, vocabSize)
	fmt.Printf("  engine:  CUDA SQ4 (%s)\n", cuda.Name())

	// Build fused pipeline
	mongoose.SQ4InferBuild(dim, kvDim, headDim, heads, kvHeads, ffnDim, vocabSize, nLayers, maxSeq)

	// Set per-layer weight pointers into fused pipeline
	wp := func(w *sq4CUDAWeight) *mongoose.SQ4WeightPtrs {
		p := &mongoose.SQ4WeightPtrs{
			Packed: w.packed.DevicePtr(),
			Bands:  w.bands.DevicePtr(),
			OC:     w.oCnt,
			Rows:   w.rows,
			Cols:   w.cols,
		}
		if w.oIdx != nil { p.OIdx = w.oIdx.DevicePtr() }
		if w.oVal != nil { p.OVal = w.oVal.DevicePtr() }
		return p
	}
	nilPtr := unsafe.Pointer(nil)
	for l := 0; l < nLayers; l++ {
		ll := &layers[l]
		bqP, bkP, bvP := nilPtr, nilPtr, nilPtr
		if ll.bq != nil { bqP = ll.bq.DevicePtr() }
		if ll.bk != nil { bkP = ll.bk.DevicePtr() }
		if ll.bv != nil { bvP = ll.bv.DevicePtr() }
		mongoose.SQ4InferSetLayer(l,
			wp(ll.wq), wp(ll.wk), wp(ll.wv), wp(ll.wo),
			wp(ll.gate), wp(ll.up), wp(ll.down),
			ll.norm1.DevicePtr(), ll.norm2.DevicePtr(),
			bqP, bkP, bvP,
			gpuKV[l].k.DevicePtr(), gpuKV[l].v.DevicePtr())
	}
	mongoose.SQ4InferSetFinal(
		finalNorm.DevicePtr(), wp(lmHead),
		gpuEmbed.DevicePtr(), gpuCos.DevicePtr(), gpuSin.DevicePtr())

	// Enable TF32 tensor core dispatch if weights were swizzled
	if useTF32 && lmHead != nil && lmHead.swizzled {
		mongoose.SQ4InferSetSwizzled(true)
		// Upload outlier band approximations for all weights
		slotNames := []string{"wq", "wk", "wv", "wo", "gate", "up", "down"}
		_ = slotNames
		for l := 0; l < nLayers; l++ {
			ll := &layers[l]
			wts := []*sq4CUDAWeight{ll.wq, ll.wk, ll.wv, ll.wo, ll.gate, ll.up, ll.down}
			for s, w := range wts {
				if w != nil && len(w.oApprox) > 0 {
					mongoose.SQ4InferSetOutlierApprox(l, s, w.oApprox)
				}
			}
		}
		if lmHead != nil && len(lmHead.oApprox) > 0 {
			mongoose.SQ4InferSetOutlierApprox(-1, 0, lmHead.oApprox)
		}
		log.Println("[SQ4-CUDA] TF32 tensor core mode enabled")
	}

	log.Println("[SQ4-CUDA] fused inference pipeline ready")

	// One CGo call per token
	fLogits := make([]float32, vocabSize)
	forward := func(tokenID, pos int) []float32 {
		mongoose.SQ4InferStepLogits(tokenID, pos, fLogits)
		return fLogits
	}
	forwardSample := func(tokenID, pos int) int {
		return mongoose.SQ4InferStep(tokenID, pos)
	}
	_ = forwardSample

	tokens := tok.Encode(prompt)
	if len(tokens) == 0 { tokens = []int{1} }

	isInstruct := strings.Contains(strings.ToLower(filepath.Base(path)), "instruct")
	if isInstruct {
		msgs := []chatMessage{{Role: "user", Content: prompt}}
		tokens = applyChatTemplate(tok, msgs, cfg)
	}

	fmt.Print("Prefilling... ")
	var logits []float32
	for i, tid := range tokens {
		logits = forward(tid, i)
	}
	fmt.Println("done")

	nextToken := sampleTopK(logits, 0.7, 40)
	maxTokens := 200
	pos := len(tokens)

	stopTokens := discoverStopTokens(tok, cfg, path)

	fmt.Printf("\nGenerating (max %d tokens):\n", maxTokens)
	if !isInstruct { fmt.Print(prompt) }

	genStart := time.Now()
	generated := 0
	for i := 0; i < maxTokens; i++ {
		if stopTokens[nextToken] { break }

		text := tok.Decode([]int{nextToken})
		if idx := findSpecialToken(text); idx >= 0 {
			text = text[:idx]
			if len(text) > 0 { fmt.Print(text) }
			break
		}
		fmt.Print(text)

		pos++
		if pos >= maxSeq-1 { break }
		nextToken = mongoose.SQ4InferStep(nextToken, pos)
		generated++
	}

	genTime := time.Since(genStart)
	fmt.Printf("\n\n--- %d tokens in %.1fs (%.1f tok/s) ---\n",
		generated, genTime.Seconds(), float64(generated)/genTime.Seconds())
}
