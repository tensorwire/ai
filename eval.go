package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/mongoose"
)

// cmdEval runs a validation pass over a dataset and reports loss + perplexity.
//
//	ai eval model=TinyLlama data=val.txt [seq=64]
func cmdEval(args map[string]string) {
	modelPath := args["model"]
	dataPath := args["data"]
	if modelPath == "" || dataPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ai eval model=<name> data=<file> [seq=N]")
		os.Exit(1)
	}
	modelPath = resolveModelPath(modelPath)
	seqLen := kvInt(args, "seq", 64)

	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatalf("eval requires a GPU (detected: %s). CUDA and Metal supported.", eng.Name())
	}
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok {
		log.Fatalf("eval currently requires CUDA (detected: %s). Metal and WebGPU support planned.", eng.Name())
	}
	mongoose.LoadKernels()

	st, err := gguf.OpenSafeTensors(modelPath)
	if err != nil {
		log.Fatalf("open model: %v", err)
	}

	dim := 2048
	heads := 32
	kvHeads := 4
	headDim := dim / heads
	kvDim := kvHeads * headDim
	nLayers := 22
	ffnDim := 5632
	vocabSize := 32000
	n := seqLen

	// Read config if available
	prof := AutoDetect(modelPath)
	if prof.Dim > 0 {
		dim = prof.Dim
		heads = prof.Heads
		kvHeads = prof.KVHeads
		headDim = prof.HeadDim
		kvDim = prof.KVDim
		nLayers = prof.Layers
		ffnDim = prof.FFNDim
		vocabSize = prof.VocabSize
	}

	log.Printf("[eval] loading %s", modelPath)

	// Load weights as FP32
	loadW := func(name string, rows, cols int) *mongoose.Tensor {
		d, _, err := st.ReadTensorFloat32(name)
		if err != nil {
			log.Fatalf("weight %s: %v", name, err)
		}
		return te.FromHost(d, []int{rows, cols})
	}
	loadNorm := func(name string) *mongoose.Tensor {
		d, _, _ := st.ReadTensorFloat32(name)
		if d == nil {
			d = make([]float32, dim)
			for i := range d {
				d[i] = 1
			}
		}
		return te.FromHost(d, []int{1, dim})
	}

	embedData, _, err := st.ReadTensorFloat32("model.embed_tokens.weight")
	if err != nil {
		log.Fatalf("embed: %v", err)
	}
	embed := te.FromHost(embedData, []int{vocabSize, dim})

	lmHeadData, _, err := st.ReadTensorFloat32("lm_head.weight")
	if err != nil {
		lmHeadData = embedData
	}
	lmHead := te.FromHost(lmHeadData, []int{vocabSize, dim})

	finalNorm := loadNorm("model.norm.weight")

	type layer struct {
		wq, wk, wv, wo, gate, up, down, norm1, norm2 *mongoose.Tensor
	}
	lays := make([]layer, nLayers)
	for l := 0; l < nLayers; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		lays[l] = layer{
			wq:    loadW(pfx+"self_attn.q_proj.weight", dim, dim),
			wk:    loadW(pfx+"self_attn.k_proj.weight", kvDim, dim),
			wv:    loadW(pfx+"self_attn.v_proj.weight", kvDim, dim),
			wo:    loadW(pfx+"self_attn.o_proj.weight", dim, dim),
			gate:  loadW(pfx+"mlp.gate_proj.weight", ffnDim, dim),
			up:    loadW(pfx+"mlp.up_proj.weight", ffnDim, dim),
			down:  loadW(pfx+"mlp.down_proj.weight", dim, ffnDim),
			norm1: loadNorm(pfx + "input_layernorm.weight"),
			norm2: loadNorm(pfx + "post_attention_layernorm.weight"),
		}
	}

	halfHead := headDim / 2
	cosTab := make([]float32, n*halfHead)
	sinTab := make([]float32, n*halfHead)
	for pos := 0; pos < n; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(10000.0, float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			cosTab[pos*halfHead+j] = float32(math.Cos(angle))
			sinTab[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}
	ropeCos := te.FromHost(cosTab, []int{n, halfHead})
	ropeSin := te.FromHost(sinTab, []int{n, halfHead})

	hidden := te.Zeros([]int{n, dim})
	tokGPU := te.Zeros([]int{n})
	logitsBuf := te.Zeros([]int{n, vocabSize})
	lossesGPU := te.Zeros([]int{n})
	targetsGPU := te.Zeros([]int{n})

	type fwdBuf struct {
		normed, Q, K, V, attnOut                   *mongoose.Tensor
		normed2, gatePre, upOut, ffnMid, dx        *mongoose.Tensor
		rmsScale1, rmsScale2                       *mongoose.Tensor
	}
	bufs := make([]fwdBuf, nLayers)
	for i := range bufs {
		bufs[i] = fwdBuf{
			normed: te.Zeros([]int{n, dim}),
			Q: te.Zeros([]int{n, dim}), K: te.Zeros([]int{n, kvDim}), V: te.Zeros([]int{n, kvDim}),
			attnOut:   te.Zeros([]int{n, dim}),
			normed2:   te.Zeros([]int{n, dim}),
			gatePre:   te.Zeros([]int{n, ffnDim}),
			upOut:     te.Zeros([]int{n, ffnDim}),
			ffnMid:    te.Zeros([]int{n, ffnDim}),
			dx:        te.Zeros([]int{n, dim}),
			rmsScale1: te.Zeros([]int{n}),
			rmsScale2: te.Zeros([]int{n}),
		}
	}

	normedFinal := te.Zeros([]int{n, dim})
	finalScales := te.Zeros([]int{n})

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	raw, err := os.ReadFile(dataPath)
	if err != nil {
		log.Fatalf("read data: %v", err)
	}
	data := make([]int, len(raw))
	for i, b := range raw {
		data[i] = int(b)
	}

	nBatches := (len(data) - 1) / n
	if nBatches > 1000 {
		nBatches = 1000
	}

	fmt.Println("ai eval — forward-only validation pass")
	fmt.Printf("  model:    %s\n", filepath.Base(modelPath))
	fmt.Printf("  data:     %s (%d bytes)\n", dataPath, len(raw))
	fmt.Printf("  batches:  %d (seq=%d)\n", nBatches, n)
	fmt.Println()

	tokF := make([]float32, n)
	targF := make([]float32, n)

	var totalLoss float64
	var totalTokens int
	t0 := time.Now()

	for batch := 0; batch < nBatches; batch++ {
		start := batch * n
		if start+n+1 > len(data) {
			break
		}
		for i := 0; i < n; i++ {
			tokF[i] = math.Float32frombits(uint32(int32(data[start+i])))
			targF[i] = math.Float32frombits(uint32(int32(data[start+i+1])))
		}
		cuda.UploadInto(tokGPU, tokF)
		cuda.UploadInto(targetsGPU, targF)

		// Forward only
		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		for li := range lays {
			l := &lays[li]
			b := &bufs[li]

			zero(b.normed)
			zero(b.rmsScale1)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), b.normed.DevicePtr(),
				l.norm1.DevicePtr(), b.rmsScale1.DevicePtr(), n, dim)

			cuda.MatMulTransposeBTInto(b.Q, b.normed, l.wq, n, dim, dim)
			cuda.MatMulTransposeBTInto(b.K, b.normed, l.wk, n, dim, kvDim)
			cuda.MatMulTransposeBTInto(b.V, b.normed, l.wv, n, dim, kvDim)

			mongoose.KRoPE(b.Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(b.K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			zero(b.attnOut)
			mongoose.KCausalAttentionGQA(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			cuda.MatMulTransposeBTInto(b.dx, b.attnOut, l.wo, n, dim, dim)
			te.AddInPlace(hidden, b.dx)

			zero(b.normed2)
			zero(b.rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), b.normed2.DevicePtr(),
				l.norm2.DevicePtr(), b.rmsScale2.DevicePtr(), n, dim)

			cuda.MatMulTransposeBTInto(b.gatePre, b.normed2, l.gate, n, dim, ffnDim)
			cuda.MatMulTransposeBTInto(b.upOut, b.normed2, l.up, n, dim, ffnDim)

			zero(b.ffnMid)
			mongoose.KSiLUGateMul(b.gatePre.DevicePtr(), b.upOut.DevicePtr(), b.ffnMid.DevicePtr(), n*ffnDim)

			cuda.MatMulTransposeBTInto(b.dx, b.ffnMid, l.down, n, ffnDim, dim)
			te.AddInPlace(hidden, b.dx)
		}

		zero(normedFinal)
		zero(finalScales)
		mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), n, dim)

		cuda.MatMulTransposeBTInto(logitsBuf, normedFinal, lmHead, n, dim, vocabSize)

		zero(lossesGPU)
		invN := float32(1.0) / float32(n)
		mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
			lossesGPU.DevicePtr(), lossesGPU.DevicePtr(), n, vocabSize, invN)

		cuda.Sync()
		lossH := te.ToHost(lossesGPU)
		var batchLoss float32
		for _, l := range lossH {
			batchLoss += l
		}
		batchLoss /= float32(n)
		totalLoss += float64(batchLoss)
		totalTokens += n

		if (batch+1)%100 == 0 || batch == nBatches-1 {
			avgLoss := totalLoss / float64(batch+1)
			ppl := math.Exp(avgLoss)
			elapsed := time.Since(t0)
			fmt.Printf("batch %d/%d  loss=%.3f  ppl=%.1f  (%.0f tok/s)\n",
				batch+1, nBatches, avgLoss, ppl, float64(totalTokens)/elapsed.Seconds())
		}
	}

	avgLoss := totalLoss / float64(nBatches)
	ppl := math.Exp(avgLoss)
	elapsed := time.Since(t0)

	fmt.Println()
	fmt.Printf("Validation complete.\n")
	fmt.Printf("  avg loss:    %.4f\n", avgLoss)
	fmt.Printf("  perplexity:  %.2f\n", ppl)
	fmt.Printf("  tokens:      %d\n", totalTokens)
	fmt.Printf("  throughput:  %.0f tok/s\n", float64(totalTokens)/elapsed.Seconds())
	fmt.Printf("  time:        %.1fs\n", elapsed.Seconds())
}
