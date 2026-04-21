package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"
	"unsafe"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/helix"
	"github.com/open-ai-org/mongoose"
)

func cmdTrainCUDA() {
	fs := flag.NewFlagSet("train-cuda", flag.ExitOnError)

	dataPath := fs.String("data", "", "Training data (text file)")
	resumePath := fs.String("resume", "", "Resume from checkpoint directory")
	dimFlag := fs.Int("dim", 128, "Model dimension")
	headsFlag := fs.Int("heads", 4, "Attention heads")
	kvHeadsFlag := fs.Int("kv-heads", 2, "KV heads (GQA)")
	layersFlag := fs.Int("layers", 4, "Transformer layers")
	ffnDimFlag := fs.Int("ffn-dim", 256, "FFN intermediate dimension")
	seqLenFlag := fs.Int("seq-len", 64, "Sequence length")
	stepsFlag := fs.Int("steps", 1000, "Training steps")
	lrFlag := fs.Float64("lr", 6e-4, "Learning rate")
	logEvery := fs.Int("log-every", 100, "Log every N steps")

	fs.Parse(os.Args[2:])

	var resumeST *gguf.SafeTensors
	if *resumePath != "" {
		ckptDir := *resumePath
		if _, err := os.Stat(filepath.Join(ckptDir, "config.json")); err != nil {
			latest := findLatestCheckpoint(ckptDir)
			if latest == "" { log.Fatalf("No checkpoint found in %s", ckptDir) }
			ckptDir = latest
		}
		cfgData, err := os.ReadFile(filepath.Join(ckptDir, "config.json"))
		if err != nil { log.Fatalf("No config.json in %s", ckptDir) }
		var cfg map[string]interface{}
		json.Unmarshal(cfgData, &cfg)
		getInt := func(key string, def int) int {
			if v, ok := cfg[key].(float64); ok { return int(v) }
			return def
		}
		*dimFlag = getInt("hidden_size", *dimFlag)
		*layersFlag = getInt("num_hidden_layers", *layersFlag)
		*headsFlag = getInt("num_attention_heads", *headsFlag)
		*kvHeadsFlag = getInt("num_key_value_heads", *kvHeadsFlag)
		*ffnDimFlag = getInt("intermediate_size", *ffnDimFlag)
		stPath := filepath.Join(ckptDir, "model.safetensors")
		resumeST, err = gguf.OpenSafeTensors(stPath)
		if err != nil { log.Fatalf("Load checkpoint: %v", err) }
		fmt.Printf("Resuming from %s (dim=%d layers=%d)\n", ckptDir, *dimFlag, *layersFlag)
	}

	if *dataPath == "" {
		*dataPath = "data/tinystories_hf.txt"
		if _, err := os.Stat(*dataPath); err != nil {
			home, _ := os.UserHomeDir()
			*dataPath = filepath.Join(home, "data", "tinystories_hf.txt")
		}
	}

	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	if te == nil { log.Fatal("TensorEngine not available") }
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok { log.Fatal("train-cuda requires CUDA") }

	if !mongoose.LoadKernels() {
		log.Fatal("CUDA kernels required — compile kernels/mongoose.cu")
	}
	log.Println("[ai] CUDA kernels loaded")

	// Multi-GPU: split layers across devices
	mc := mongoose.NewMultiCUDA()
	nGPUs := 0
	if mc != nil {
		nGPUs = mc.DeviceCount
		log.Printf("[ai] multi-GPU: %d devices available", nGPUs)
	}

	sched := mongoose.NewScheduler(eng)
	_ = sched

	dim := *dimFlag
	heads := *headsFlag
	kvHeads := *kvHeadsFlag
	headDim := dim / heads
	kvDim := kvHeads * headDim
	nLayers := *layersFlag
	ffnDim := *ffnDimFlag
	seqLen := *seqLenFlag
	vocabSize := 256
	lr := float32(*lrFlag)
	n := seqLen

	raw, err := os.ReadFile(*dataPath)
	if err != nil { log.Fatalf("read data: %v", err) }
	data := make([]int, len(raw))
	for i, b := range raw { data[i] = int(b) }

	// Estimate VRAM requirement and auto-select precision
	vramBytes := eng.VRAM()
	vramGB := float64(vramBytes) / (1024 * 1024 * 1024)

	layerParams := int64(dim*dim*2 + kvDim*dim*2 + ffnDim*dim*3 + dim*2)
	totalParams := int64(vocabSize*dim) + int64(dim) + int64(nLayers)*layerParams
	// FP32: weights + gradients + Adam M + Adam V = 4x params
	fp32Bytes := totalParams * 4 * 4
	// FP16 weights + FP32 grads + FP32 Adam M + FP32 Adam V = params*(2+4+4+4) = 14 bytes/param
	fp16Bytes := totalParams * 14
	// Activation buffers: hidden, normed, Q, K, V, attnOut, ffnMid, etc. per layer
	bufBytes := int64(n) * int64(dim) * 4 * 20 * int64(nLayers) // ~20 buffers per layer
	fp32Need := float64(fp32Bytes+bufBytes) * 1.1
	fp16Need := float64(fp16Bytes+bufBytes) * 1.1

	useFP16Training := false
	if fp32Need > float64(vramBytes) && vramBytes > 0 {
		if fp16Need <= float64(vramBytes) {
			useFP16Training = true
			log.Printf("[ai] FP32 needs %.1f GB but only %.1f GB VRAM — switching to mixed precision (FP16 weights + FP32 grads)",
				fp32Need/(1024*1024*1024), vramGB)
		} else {
			log.Fatalf("[ai] model needs %.1f GB (FP16) but only %.1f GB VRAM. Reduce dim or layers, or use multi-GPU.",
				fp16Need/(1024*1024*1024), vramGB)
		}
	}

	// FP16 GEMMs (cublasLt) give 2x tensor core throughput but add FP32→FP16
	// conversion overhead per GEMM. At dim>=512 the GEMM compute dominates
	// and FP16 wins. At dim<512 the conversion overhead dominates and FP32 wins.
	useFP16Training = dim >= 512
	_ = useFP16Training

	conductor := mongoose.NewConductor(vocabSize, 100)

	halfHead := headDim / 2
	cosTab := make([]float32, seqLen*halfHead)
	sinTab := make([]float32, seqLen*halfHead)
	for pos := 0; pos < seqLen; pos++ {
		for j := 0; j < halfHead; j++ {
			freq := 1.0 / math.Pow(10000.0, float64(2*j)/float64(headDim))
			angle := float64(pos) * freq
			cosTab[pos*halfHead+j] = float32(math.Cos(angle))
			sinTab[pos*halfHead+j] = float32(math.Sin(angle))
		}
	}
	ropeCos := te.FromHost(cosTab, []int{seqLen, halfHead})
	ropeSin := te.FromHost(sinTab, []int{seqLen, halfHead})

	loadOrInit := func(name string, rows, cols int) *mongoose.Tensor {
		if resumeST != nil && resumeST.HasTensor(name) {
			data, _, err := resumeST.ReadTensorFloat32(name)
			if err == nil && len(data) == rows*cols {
				return te.FromHost(data, []int{rows, cols})
			}
			log.Printf("[resume] warning: %s load failed, using random init", name)
		}
		bound := float32(math.Sqrt(2.0 / float64(cols)))
		d := make([]float32, rows*cols)
		for i := range d { d[i] = bound * (2*rand.Float32() - 1) }
		return te.FromHost(d, []int{rows, cols})
	}
	loadOrOnes := func(name string, sz int) *mongoose.Tensor {
		if resumeST != nil && resumeST.HasTensor(name) {
			data, _, err := resumeST.ReadTensorFloat32(name)
			if err == nil && len(data) == sz {
				return te.FromHost(data, []int{1, sz})
			}
		}
		d := make([]float32, sz)
		for i := range d { d[i] = 1.0 }
		return te.FromHost(d, []int{1, sz})
	}

	embed := loadOrInit("model.embed_tokens.weight", vocabSize, dim)
	finalNorm := loadOrOnes("model.norm.weight", dim)

	type layer struct{ wq, wk, wv, wo, gate, up, down, norm1, norm2 *mongoose.Tensor }
	lays := make([]layer, nLayers)
	for l := range lays {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		lays[l] = layer{
			wq: loadOrInit(pfx+"self_attn.q_proj.weight", dim, dim),
			wk: loadOrInit(pfx+"self_attn.k_proj.weight", kvDim, dim),
			wv: loadOrInit(pfx+"self_attn.v_proj.weight", kvDim, dim),
			wo: loadOrInit(pfx+"self_attn.o_proj.weight", dim, dim),
			gate: loadOrInit(pfx+"mlp.gate_proj.weight", ffnDim, dim),
			up: loadOrInit(pfx+"mlp.up_proj.weight", ffnDim, dim),
			down: loadOrInit(pfx+"mlp.down_proj.weight", dim, ffnDim),
			norm1: loadOrOnes(pfx+"input_layernorm.weight", dim),
			norm2: loadOrOnes(pfx+"post_attention_layernorm.weight", dim),
		}
	}

	// FP16 weight copies for tensor core GEMMs (GPU 0 — primary)
	type fp16W struct{ wq, wk, wv, wo, gate, up, down *mongoose.Tensor }
	fp16 := make([]fp16W, nLayers)
	for l := range fp16 {
		fp16[l] = fp16W{
			wq: cuda.FromHostFP16(te.ToHost(lays[l].wq), lays[l].wq.Shape),
			wk: cuda.FromHostFP16(te.ToHost(lays[l].wk), lays[l].wk.Shape),
			wv: cuda.FromHostFP16(te.ToHost(lays[l].wv), lays[l].wv.Shape),
			wo: cuda.FromHostFP16(te.ToHost(lays[l].wo), lays[l].wo.Shape),
			gate: cuda.FromHostFP16(te.ToHost(lays[l].gate), lays[l].gate.Shape),
			up: cuda.FromHostFP16(te.ToHost(lays[l].up), lays[l].up.Shape),
			down: cuda.FromHostFP16(te.ToHost(lays[l].down), lays[l].down.Shape),
		}
	}
	embedFP16 := cuda.FromHostFP16(te.ToHost(embed), embed.Shape)

	// FP16 scratch for activation conversion before GEMMs
	maxActSize := n * dim
	if n*ffnDim > maxActSize { maxActSize = n * ffnDim }
	if n*vocabSize > maxActSize { maxActSize = n * vocabSize }
	fp16Scratch := cuda.AllocFP16Tensor(maxActSize, []int{maxActSize})

	// Multi-GPU: replicate FP16 weights to GPU 1 for parallel dispatch.
	// Independent GEMM pairs (Q+K, gate+up) fire simultaneously across devices.
	// Activations P2P-copied to GPU 1 before paired GEMMs, results copied back.
	type gpu1Weights struct {
		wq, wk, wv, wo, gate, up, down unsafe.Pointer
		actBuf, outBuf                  unsafe.Pointer // FP16 scratch on GPU 1
	}
	var gpu1 []gpu1Weights
	if nGPUs >= 2 && useFP16Training {
		gpu1 = make([]gpu1Weights, nLayers)
		for l := 0; l < nLayers; l++ {
			wqH := te.ToHost(lays[l].wq); wkH := te.ToHost(lays[l].wk)
			wvH := te.ToHost(lays[l].wv); woH := te.ToHost(lays[l].wo)
			gH := te.ToHost(lays[l].gate); uH := te.ToHost(lays[l].up)
			dH := te.ToHost(lays[l].down)
			gpu1[l] = gpu1Weights{
				wq: mc.Upload(1, wqH).Ptr, wk: mc.Upload(1, wkH).Ptr,
				wv: mc.Upload(1, wvH).Ptr, wo: mc.Upload(1, woH).Ptr,
				gate: mc.Upload(1, gH).Ptr, up: mc.Upload(1, uH).Ptr,
				down: mc.Upload(1, dH).Ptr,
			}
		}
		gpu1[0].actBuf = mc.Alloc(1, (maxActSize*2+3)/4)
		gpu1[0].outBuf = mc.Alloc(1, (maxActSize*2+3)/4)
		log.Printf("[multi-gpu] FP16 weights replicated to GPU 1, parallel dispatch enabled")
	}

	// Forward GEMM: FP16 tensor cores at dim>=512, FP32 TF32 at dim<512
	gemmBT := func(out, act *mongoose.Tensor, wFP32 *mongoose.Tensor, wFP16 *mongoose.Tensor, m, k, nn int) {
		if useFP16Training && wFP16 != nil {
			mongoose.KFP32ToFP16(act.DevicePtr(), fp16Scratch.DevicePtr(), m*k)
			result := cuda.MatMulFP16TransposeBT(fp16Scratch, wFP16, m, k, nn)
			mongoose.KCopy(out.DevicePtr(), result.DevicePtr(), m*nn*4)
			te.Release(result)
		} else {
			cuda.MatMulTransposeBTInto(out, act, wFP32, m, k, nn)
		}
	}


	// Sync FP32 master weights → FP16 copies after optimizer step
	syncFP16Weights := func() {
		for l := range lays {
			mongoose.KFP32ToFP16(lays[l].wq.DevicePtr(), fp16[l].wq.DevicePtr(), lays[l].wq.Size)
			mongoose.KFP32ToFP16(lays[l].wk.DevicePtr(), fp16[l].wk.DevicePtr(), lays[l].wk.Size)
			mongoose.KFP32ToFP16(lays[l].wv.DevicePtr(), fp16[l].wv.DevicePtr(), lays[l].wv.Size)
			mongoose.KFP32ToFP16(lays[l].wo.DevicePtr(), fp16[l].wo.DevicePtr(), lays[l].wo.Size)
			mongoose.KFP32ToFP16(lays[l].gate.DevicePtr(), fp16[l].gate.DevicePtr(), lays[l].gate.Size)
			mongoose.KFP32ToFP16(lays[l].up.DevicePtr(), fp16[l].up.DevicePtr(), lays[l].up.Size)
			mongoose.KFP32ToFP16(lays[l].down.DevicePtr(), fp16[l].down.DevicePtr(), lays[l].down.Size)
		}
		mongoose.KFP32ToFP16(embed.DevicePtr(), embedFP16.DevicePtr(), embed.Size)
		// P2P sync GPU 0 FP16 → GPU 1 FP16
		if gpu1 != nil {
			cuda.Sync()
			for l := range lays {
				src := []unsafe.Pointer{fp16[l].wq.DevicePtr(), fp16[l].wk.DevicePtr(),
					fp16[l].wv.DevicePtr(), fp16[l].wo.DevicePtr(),
					fp16[l].gate.DevicePtr(), fp16[l].up.DevicePtr(), fp16[l].down.DevicePtr()}
				dst := []unsafe.Pointer{gpu1[l].wq, gpu1[l].wk, gpu1[l].wv, gpu1[l].wo,
					gpu1[l].gate, gpu1[l].up, gpu1[l].down}
				sizes := []int{lays[l].wq.Size, lays[l].wk.Size, lays[l].wv.Size, lays[l].wo.Size,
					lays[l].gate.Size, lays[l].up.Size, lays[l].down.Size}
				for i := range src {
					mc.PeerCopyInto(0, src[i], 1, dst[i], sizes[i]*2)
				}
			}
		}
	}

	// Helix optimizer — DNA-coupled gradient descent
	hlx := helix.NewHelixOptimizer(lr, 0.9, 0.95, 1e-8, 0.1)

	type as struct{ m, v *mongoose.Tensor }
	newAS := func(sz int) as { return as{te.Zeros([]int{sz}), te.Zeros([]int{sz})} }
	embedAS := newAS(vocabSize * dim)
	type layerAS struct{ wq, wk, wv, wo, gate, up, down as }
	layAS := make([]layerAS, nLayers)
	for l := range layAS {
		layAS[l] = layerAS{
			wq: newAS(dim * dim), wk: newAS(kvDim * dim), wv: newAS(kvDim * dim),
			wo: newAS(dim * dim), gate: newAS(ffnDim * dim), up: newAS(ffnDim * dim),
			down: newAS(dim * ffnDim),
		}
	}

	hidden := te.Zeros([]int{n, dim})
	tokGPU := te.Zeros([]int{n})
	targetsGPU := te.Zeros([]int{n})
	logitsBuf := te.Zeros([]int{n, vocabSize})
	lossesGPU := te.Zeros([]int{n})
	gradGPU := te.Zeros([]int{n, vocabSize})
	normedFinal := te.Zeros([]int{n, dim})
	finalScales := te.Zeros([]int{n})
	dEmbed := te.Zeros([]int{vocabSize, dim})
	dHidden := te.Zeros([]int{n, dim})
	dScratch := te.Zeros([]int{n, dim})

	type fwdBuf struct {
		xIn, normed, Q, K, V, attnOut          *mongoose.Tensor
		xMid, normed2, gatePre, upOut, ffnMid  *mongoose.Tensor
		rmsScale1, rmsScale2                   *mongoose.Tensor
		dFfnMid, dGate, dUp, dN2, dx           *mongoose.Tensor
		dAttnOut, dQ, dK, dV, dN1              *mongoose.Tensor
		dWDown, dWGate, dWUp, dWO, dWQ, dWK, dWV *mongoose.Tensor
	}
	bufs := make([]fwdBuf, nLayers)
	for i := range bufs {
		bufs[i] = fwdBuf{
			xIn: te.Zeros([]int{n, dim}), normed: te.Zeros([]int{n, dim}),
			Q: te.Zeros([]int{n, dim}), K: te.Zeros([]int{n, kvDim}), V: te.Zeros([]int{n, kvDim}),
			attnOut: te.Zeros([]int{n, dim}),
			xMid: te.Zeros([]int{n, dim}), normed2: te.Zeros([]int{n, dim}),
			gatePre: te.Zeros([]int{n, ffnDim}), upOut: te.Zeros([]int{n, ffnDim}),
			ffnMid: te.Zeros([]int{n, ffnDim}),
			rmsScale1: te.Zeros([]int{n}), rmsScale2: te.Zeros([]int{n}),
			dFfnMid: te.Zeros([]int{n, ffnDim}), dGate: te.Zeros([]int{n, ffnDim}),
			dUp: te.Zeros([]int{n, ffnDim}), dN2: te.Zeros([]int{n, dim}),
			dx: te.Zeros([]int{n, dim}),
			dAttnOut: te.Zeros([]int{n, dim}), dQ: te.Zeros([]int{n, dim}),
			dK: te.Zeros([]int{n, kvDim}), dV: te.Zeros([]int{n, kvDim}),
			dN1: te.Zeros([]int{n, dim}),
			dWDown: te.Zeros([]int{dim, ffnDim}), dWGate: te.Zeros([]int{ffnDim, dim}),
			dWUp: te.Zeros([]int{ffnDim, dim}), dWO: te.Zeros([]int{dim, dim}),
			dWQ: te.Zeros([]int{dim, dim}), dWK: te.Zeros([]int{kvDim, dim}),
			dWV: te.Zeros([]int{kvDim, dim}),
		}
	}

	nParams := vocabSize*dim + dim
	for range lays {
		nParams += dim + dim*dim + kvDim*dim*2 + dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Printf("ai train — GPU kernels + Helix DNA optimizer (FP16=%v)\n", useFP16Training)
	fmt.Printf("  engine:   %s\n", eng.Name())
	fmt.Printf("  data:     %s (%d bytes)\n", *dataPath, len(raw))
	fmt.Printf("  model:    dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d vocab=%d\n",
		dim, heads, kvHeads, nLayers, ffnDim, seqLen, vocabSize)
	if nParams > 1e6 { fmt.Printf("  params:   %.2fM\n", float64(nParams)/1e6) } else {
		fmt.Printf("  params:   %.1fK\n", float64(nParams)/1e3)
	}
	fmt.Printf("  training: steps=%d lr=%.0e\n", *stepsFlag, *lrFlag)
	fmt.Println()

	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	totalSteps := *stepsFlag
	warmupSteps := 1
	minLR := lr / 10.0

	getLR := func(step int) float32 {
		if step < warmupSteps {
			return lr * float32(step) / float32(warmupSteps)
		}
		progress := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
		cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
		return minLR + float32(cosine)*float32(lr-minLR)
	}

	var curLR float32 // set each step by signal-scaled getLR
	adamW := func(param, grad, mS, vS *mongoose.Tensor, step int) {
		mongoose.KAdamW(param.DevicePtr(), grad.DevicePtr(), mS.DevicePtr(), vS.DevicePtr(),
			curLR, 0.1, step, param.Size)
	}

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	// L3 bridge: pointers + rung + batch data
	// GPU-write: 49 hot row addresses (float32-encoded pointers)
	// CPU-write: 6 rung coefficients
	// Batch: tokens + targets
	const nSparse = 49
	gpuWriteOff := 0                         // 49 × 4 = 196 bytes
	cpuWriteOff := nSparse*4 + 4             // 6 × 4 = 24 bytes
	batchOff := cpuWriteOff + 6*4 + 4        // n×4×2 bytes
	l3Size := batchOff + n*4*2
	l3 := cuda.AllocL3Bridge(l3Size)

	var hotRowsL3 []float32  // GPU-write: 49 hot row indices
	var rungL3 []float32     // CPU-write: 6 rung coefficients
	var nextTokL3, nextTargL3 []float32
	if l3 != nil {
		hotRowsL3 = l3.Float32(gpuWriteOff, nSparse)
		rungL3 = l3.Float32(cpuWriteOff, 6)
		nextTokL3 = l3.Float32(batchOff, n)
		nextTargL3 = l3.Float32(batchOff+n*4, n)
		log.Printf("[L3] bridge: %d bytes (hotRows:%d rung:%d batch:%d)",
			l3Size, nSparse*4, 6*4, n*4*2)
	}

	prepBatch := func(start int) {
		if l3 == nil { return }
		for i := 0; i < n; i++ {
			nextTokL3[i] = math.Float32frombits(uint32(int32(data[start+i])))
			nextTargL3[i] = math.Float32frombits(uint32(int32(data[start+i+1])))
		}
	}

	// Calibrate scheduler on training op shapes
	sched.CalibrateMatMul(n, dim, dim)       // QKV projections
	sched.CalibrateMatMul(n, dim, kvDim)     // K/V projections
	sched.CalibrateMatMul(n, dim, ffnDim)    // FFN gate/up
	sched.CalibrateMatMul(n, ffnDim, dim)    // FFN down
	sched.CalibrateMatMul(n, dim, vocabSize) // logits
	sched.CalibrateAll(mongoose.NormKey(dim), func(e mongoose.Engine) {
		d := make([]float32, dim)
		w := make([]float32, dim)
		for i := range w { w[i] = 1 }
		e.RMSNorm(d, w, 1e-6)
	})
	log.Printf("[scheduler] calibrated %d GPUs, %d op shapes", sched.NumGPUs(), 6)

	start0 := rng.Intn(len(data) - n - 1)
	prepBatch(start0)

	fmt.Println("Training...")
	t0 := time.Now()

	var batchReady chan struct{}
	curStart := start0
	// Sparse immune checkpoint: save/restore 49 hot rows on loss floor/rebound
	type sparseCheckpoint struct {
		rows    []int32              // hot row indices at checkpoint time
		weights map[string][]float32 // "layer.proj" → saved row data
		loss    float32
		step    int
	}
	var ckpt *sparseCheckpoint
	bestFloor := float32(1e30)
	immuneActive := false
	floorContactStep := 0
	floorWindow := 10
	maxRecoveries := 20
	recoveryCount := 0

	saveHotRows := func(hotRows []int32, loss float32, step int) {
		ckpt = &sparseCheckpoint{
			rows:    make([]int32, len(hotRows)),
			weights: make(map[string][]float32),
			loss:    loss,
			step:    step,
		}
		copy(ckpt.rows, hotRows)
		cuda.Sync()
		for li := range lays {
			for _, proj := range []struct{ name string; w *mongoose.Tensor }{
				{"wq", lays[li].wq}, {"wk", lays[li].wk}, {"wv", lays[li].wv},
				{"wo", lays[li].wo}, {"gate", lays[li].gate}, {"up", lays[li].up},
				{"down", lays[li].down},
			} {
				wH := te.ToHost(proj.w)
				cols := proj.w.Size / (proj.w.Shape[0])
				key := fmt.Sprintf("%d.%s", li, proj.name)
				saved := make([]float32, 0, len(hotRows)*cols)
				for _, r := range hotRows {
					row := int(r)
					if row >= 0 && row < proj.w.Shape[0] {
						saved = append(saved, wH[row*cols:(row+1)*cols]...)
					}
				}
				ckpt.weights[key] = saved
			}
		}
	}

	restoreHotRows := func() {
		if ckpt == nil { return }
		for li := range lays {
			for _, proj := range []struct{ name string; w *mongoose.Tensor }{
				{"wq", lays[li].wq}, {"wk", lays[li].wk}, {"wv", lays[li].wv},
				{"wo", lays[li].wo}, {"gate", lays[li].gate}, {"up", lays[li].up},
				{"down", lays[li].down},
			} {
				key := fmt.Sprintf("%d.%s", li, proj.name)
				saved := ckpt.weights[key]
				if len(saved) == 0 { continue }
				wH := te.ToHost(proj.w)
				cols := proj.w.Size / (proj.w.Shape[0])
				idx := 0
				for _, r := range ckpt.rows {
					row := int(r)
					if row >= 0 && row < proj.w.Shape[0] && idx+cols <= len(saved) {
						copy(wH[row*cols:(row+1)*cols], saved[idx:idx+cols])
						idx += cols
					}
				}
				cuda.UploadInto(proj.w, wH)
			}
		}
	}

	// Helix-pattern disk checkpoints: save on new best loss, 88-step cooldown
	ckptDir := filepath.Join(os.TempDir(), "ai-train-out", "checkpoints")
	if GlobalOutDir != "" {
		ckptDir = filepath.Join(GlobalOutDir, "checkpoints")
	}
	os.MkdirAll(ckptDir, 0755)
	lastCkptStep := 0
	ckptBestLoss := float32(999.0)
	ckptCooldown := 88 // double helix

	saveFullCheckpoint := func(step int, loss float32) {
		cuda.Sync()
		tensors := map[string]gguf.SaveTensor{
			"model.embed_tokens.weight": {Data: te.ToHost(embed), Shape: []int{vocabSize, dim}},
			"model.norm.weight":         {Data: te.ToHost(finalNorm), Shape: []int{dim}},
		}
		for li := range lays {
			pfx := fmt.Sprintf("model.layers.%d.", li)
			tensors[pfx+"self_attn.q_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wq), Shape: []int{dim, dim}}
			tensors[pfx+"self_attn.k_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wk), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.v_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wv), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.o_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].wo), Shape: []int{dim, dim}}
			tensors[pfx+"mlp.gate_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].gate), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.up_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].up), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.down_proj.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].down), Shape: []int{dim, ffnDim}}
			tensors[pfx+"input_layernorm.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].norm1), Shape: []int{dim}}
			tensors[pfx+"post_attention_layernorm.weight"] = gguf.SaveTensor{Data: te.ToHost(lays[li].norm2), Shape: []int{dim}}
		}
		stepDir := filepath.Join(ckptDir, fmt.Sprintf("step-%05d", step))
		os.MkdirAll(stepDir, 0755)
		stPath := filepath.Join(stepDir, "model.safetensors")
		if err := gguf.SaveSafeTensors(stPath, tensors); err != nil {
			log.Printf("[checkpoint] save error: %v", err)
		} else {
			cfgJSON := fmt.Sprintf(`{"architectures":["LlamaForCausalLM"],"hidden_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"intermediate_size":%d,"vocab_size":%d,"max_position_embeddings":2048,"rope_theta":10000.0,"rms_norm_eps":1e-6,"hidden_act":"silu","tie_word_embeddings":true}`,
				dim, nLayers, heads, kvHeads, ffnDim, vocabSize)
			os.WriteFile(filepath.Join(stepDir, "config.json"), []byte(cfgJSON), 0644)
			log.Printf("[checkpoint] immune: new best %.3f at step %d → %s", loss, step, stepDir)
		}
	}

	var prevLoss float32
	for step := 1; step <= *stepsFlag; step++ {
		// Wait for L3 prep goroutine from previous step
		if batchReady != nil {
			<-batchReady
		}

		// Upload current batch
		if l3 != nil {
			cuda.UploadInto(tokGPU, nextTokL3)
			cuda.UploadInto(targetsGPU, nextTargL3)
		} else {
			curStart = rng.Intn(len(data) - n - 1)
			tokF := make([]float32, n)
			for i := 0; i < n; i++ { tokF[i] = math.Float32frombits(uint32(int32(data[curStart+i]))) }
			cuda.UploadInto(tokGPU, tokF)
			targF := make([]float32, n)
			for i := 0; i < n; i++ { targF[i] = math.Float32frombits(uint32(int32(data[curStart+i+1]))) }
			cuda.UploadInto(targetsGPU, targF)
		}

		// Prep NEXT batch in L3 (async, overlaps with GPU training)
		nextStart := rng.Intn(len(data) - n - 1)
		batchReady = make(chan struct{})
		go func(s int, ch chan struct{}) {
			prepBatch(s)
			close(ch)
		}(nextStart, batchReady)

		tokIDs := make([]int32, n)
		for i := 0; i < n; i++ { tokIDs[i] = int32(data[nextStart+i]) }
		conductor.Observe(tokIDs)

		// === FORWARD ===
		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		for li := range lays {
			l := &lays[li]
			b := &bufs[li]

			cuda.CopyInto(b.xIn, hidden)

			zero(b.normed); zero(b.rmsScale1)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), b.normed.DevicePtr(),
				l.norm1.DevicePtr(), b.rmsScale1.DevicePtr(), n, dim)

			gemmBT(b.Q, b.normed, l.wq, fp16[li].wq, n, dim, dim)
			gemmBT(b.K, b.normed, l.wk, fp16[li].wk, n, dim, kvDim)
			gemmBT(b.V, b.normed, l.wv, fp16[li].wv, n, dim, kvDim)

			mongoose.KRoPE(b.Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPE(b.K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			zero(b.attnOut)
			mongoose.KCausalAttentionGQA(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.attnOut.DevicePtr(),
				n, dim, kvDim, heads, kvHeads)

			gemmBT(b.dx, b.attnOut, l.wo, fp16[li].wo, n, dim, dim)
			te.AddInPlace(hidden, b.dx)

			cuda.CopyInto(b.xMid, hidden)

			zero(b.normed2); zero(b.rmsScale2)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), b.normed2.DevicePtr(),
				l.norm2.DevicePtr(), b.rmsScale2.DevicePtr(), n, dim)

			gemmBT(b.gatePre, b.normed2, l.gate, fp16[li].gate, n, dim, ffnDim)
			gemmBT(b.upOut, b.normed2, l.up, fp16[li].up, n, dim, ffnDim)

			zero(b.ffnMid)
			mongoose.KSiLUGateMul(b.gatePre.DevicePtr(), b.upOut.DevicePtr(), b.ffnMid.DevicePtr(), n*ffnDim)

			gemmBT(b.dx, b.ffnMid, l.down, fp16[li].down, n, ffnDim, dim)
			te.AddInPlace(hidden, b.dx)
		}

		zero(normedFinal); zero(finalScales)
		mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), n, dim)

		gemmBT(logitsBuf, normedFinal, embed, embedFP16, n, dim, vocabSize)

		zero(lossesGPU); zero(gradGPU)
		invN := float32(1.0) / float32(n)
		mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
			lossesGPU.DevicePtr(), gradGPU.DevicePtr(), n, vocabSize, invN)

		// Always read loss — helix immune system needs it every step
		cuda.Sync()
		var stepLoss float32
		lossH := te.ToHost(lossesGPU)
		for _, l := range lossH { stepLoss += l }
		stepLoss /= float32(n)

		// === BACKWARD ===
		// All backward GEMMs use FP32 — FP16 backward blocked on FP16 element-wise kernels.
		// The dHidden chain will go FP16 when RMSNorm/RoPE/attention kernels get __half variants.
		cuda.MatMulTransposeATInto(dEmbed, gradGPU, normedFinal, n, vocabSize, dim)
		cuda.MatMulTInto(dHidden, gradGPU, embed, n, vocabSize, dim)

		zero(dScratch)
		mongoose.KRMSNormBackward(dHidden.DevicePtr(), hidden.DevicePtr(),
			finalNorm.DevicePtr(), finalScales.DevicePtr(), dScratch.DevicePtr(), n, dim)
		cuda.CopyInto(dHidden, dScratch)

		for li := nLayers - 1; li >= 0; li-- {
			l := &lays[li]
			b := &bufs[li]

			cuda.MatMulTInto(b.dFfnMid, dHidden, l.down, n, dim, ffnDim)
			cuda.MatMulTransposeATInto(b.dWDown, dHidden, b.ffnMid, n, dim, ffnDim)

			zero(b.dGate); zero(b.dUp)
			mongoose.KSiLUGateBackward(b.dFfnMid.DevicePtr(), b.gatePre.DevicePtr(),
				b.upOut.DevicePtr(), b.dGate.DevicePtr(), b.dUp.DevicePtr(), n*ffnDim)

			cuda.MatMulTInto(b.dN2, b.dGate, l.gate, n, ffnDim, dim)
			cuda.MatMulTInto(b.dx, b.dUp, l.up, n, ffnDim, dim)
			te.AddInPlace(b.dN2, b.dx)

			cuda.MatMulTransposeATInto(b.dWGate, b.dGate, b.normed2, n, ffnDim, dim)
			cuda.MatMulTransposeATInto(b.dWUp, b.dUp, b.normed2, n, ffnDim, dim)

			zero(b.dx)
			mongoose.KRMSNormBackward(b.dN2.DevicePtr(), b.xMid.DevicePtr(),
				l.norm2.DevicePtr(), b.rmsScale2.DevicePtr(), b.dx.DevicePtr(), n, dim)
			te.AddInPlace(dHidden, b.dx)

			cuda.MatMulTInto(b.dAttnOut, dHidden, l.wo, n, dim, dim)
			cuda.MatMulTransposeATInto(b.dWO, dHidden, b.attnOut, n, dim, dim)

			zero(b.dQ); zero(b.dK); zero(b.dV)
			mongoose.KCausalAttentionBackward(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.dAttnOut.DevicePtr(),
				b.dQ.DevicePtr(), b.dK.DevicePtr(), b.dV.DevicePtr(), n, dim, kvDim, heads, kvHeads)

			mongoose.KRoPEBackward(b.dQ.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
			mongoose.KRoPEBackward(b.dK.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

			cuda.MatMulTInto(b.dN1, b.dQ, l.wq, n, dim, dim)
			cuda.MatMulTInto(b.dx, b.dK, l.wk, n, kvDim, dim)
			cuda.MatMulTInto(b.dN2, b.dV, l.wv, n, kvDim, dim)
			te.AddInPlace(b.dN1, b.dx); te.AddInPlace(b.dN1, b.dN2)

			cuda.MatMulTransposeATInto(b.dWQ, b.dQ, b.normed, n, dim, dim)
			cuda.MatMulTransposeATInto(b.dWK, b.dK, b.normed, n, kvDim, dim)
			cuda.MatMulTransposeATInto(b.dWV, b.dV, b.normed, n, kvDim, dim)

			zero(b.dx)
			mongoose.KRMSNormBackward(b.dN1.DevicePtr(), b.xIn.DevicePtr(),
				l.norm1.DevicePtr(), b.rmsScale1.DevicePtr(), b.dx.DevicePtr(), n, dim)
			te.AddInPlace(dHidden, b.dx)
		}

		// === L3 signal exchange ===
		// GPU-write: conductor hot row indices → L3
		// CPU: helix reads hot rows, computes rung, writes rung → L3
		// GPU: optimizer reads rung from L3
		if l3 != nil {
			hotRows := conductor.HotRows()
			nHot := len(hotRows)
			if nHot > nSparse { nHot = nSparse }
			for i := 0; i < nHot; i++ {
				hotRowsL3[i] = float32(hotRows[i])
			}
			for i := nHot; i < nSparse; i++ {
				hotRowsL3[i] = -1
			}
		}

		// === Sparse immune system ===
		hotRows := conductor.HotRows()

		// Checkpoint when loss improves near floor
		if !immuneActive && step > 1 && stepLoss > 0 {
			if stepLoss < bestFloor*1.1 {
				saveHotRows(hotRows, stepLoss, step)
			}
		}

		// Floor detection
		if stepLoss > 0 && stepLoss < bestFloor {
			bestFloor = stepLoss
			if !immuneActive {
				immuneActive = true
				floorContactStep = step
				recoveryCount = 0
			}
		}

		// Helix-pattern disk checkpoint: new best loss + 88-step cooldown
		if stepLoss < ckptBestLoss && step-lastCkptStep >= ckptCooldown && step > 1 {
			ckptBestLoss = stepLoss
			lastCkptStep = step
			go saveFullCheckpoint(step, stepLoss)
		}

		// Immune monitoring: rebound detection
		immuneSkip := false
		if immuneActive && step-floorContactStep >= floorWindow {
			rebound := stepLoss - bestFloor
			threshold := bestFloor * 0.05
			if rebound > threshold && recoveryCount < maxRecoveries && ckpt != nil {
				restoreHotRows()
				recoveryCount++
				immuneActive = false
				immuneSkip = true
				if step <= 3 || step%*logEvery == 0 || true {
					elapsed := time.Since(t0)
					fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s) [IMMUNE → floor %.3f]\n",
						step, *stepsFlag, stepLoss, getLR(step), elapsed.Seconds(),
						float64(step)/elapsed.Seconds(), ckpt.loss)
				}
			} else {
				immuneActive = false
			}
		}

		// === HELIX DNA optimizer ===
		stepLR := getLR(step)
		r, _, _, _ := hlx.PrepareStep(step, stepLoss, stepLR)

		// Signal-driven LR scaling: dampen when loss rebounds
		if step > 1 && prevLoss > 0 {
			dLoss := float64(stepLoss) - float64(prevLoss)
			if dLoss > 0 {
				ratio := float32(dLoss / math.Max(float64(prevLoss), 1e-6))
				if ratio > 1.0 { ratio = 1.0 }
				stepLR *= (1.0 - ratio)
			}
		}
		prevLoss = stepLoss
		curLR = stepLR

		if immuneSkip {
			continue
		}

		// CPU-write: rung coefficients → L3
		if l3 != nil {
			rungL3[0] = r.Backbone1
			rungL3[1] = r.Glyco1
			rungL3[2] = r.Hbond1
			rungL3[3] = r.Hbond2
			rungL3[4] = r.Glyco2
			rungL3[5] = r.Backbone2
		}

		for li := range lays {
			l := &lays[li]
			b := &bufs[li]
			la := &layAS[li]

			// gate↔up: GC pair (3 H-bonds)
			mongoose.KHelixDNAStep(
				l.gate.DevicePtr(), l.up.DevicePtr(),
				b.dWGate.DevicePtr(), b.dWUp.DevicePtr(),
				la.gate.m.DevicePtr(), la.up.m.DevicePtr(),
				la.gate.v.DevicePtr(), la.up.v.DevicePtr(),
				curLR, 0.9, 0.95, step, 1e-8, 0.1,
				r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
				3.0/5.0, l.gate.Size)

			// wq↔wk: AT pair (2 H-bonds) — only when same size
			if l.wq.Size == l.wk.Size {
				mongoose.KHelixDNAStep(
					l.wq.DevicePtr(), l.wk.DevicePtr(),
					b.dWQ.DevicePtr(), b.dWK.DevicePtr(),
					la.wq.m.DevicePtr(), la.wk.m.DevicePtr(),
					la.wq.v.DevicePtr(), la.wk.v.DevicePtr(),
					curLR, 0.9, 0.95, step, 1e-8, 0.1,
					r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
					2.0/5.0, l.wq.Size)
			} else {
				adamW(l.wq, b.dWQ, la.wq.m, la.wq.v, step)
				adamW(l.wk, b.dWK, la.wk.m, la.wk.v, step)
			}

			// Singles
			adamW(l.wv, b.dWV, la.wv.m, la.wv.v, step)
			adamW(l.wo, b.dWO, la.wo.m, la.wo.v, step)
			adamW(l.down, b.dWDown, la.down.m, la.down.v, step)
		}

		adamW(embed, dEmbed, embedAS.m, embedAS.v, step)

		// Sync FP32 master weights → FP16 for next forward
		syncFP16Weights()

		if step <= 3 || step%*logEvery == 0 {
			elapsed := time.Since(t0)
			fmt.Printf("step %5d/%d  loss=%.3f  lr=%.1e  %.0fs  (%.1f steps/s)\n",
				step, *stepsFlag, stepLoss, getLR(step), elapsed.Seconds(), float64(step)/elapsed.Seconds())
		}
	}

	cuda.Sync()
	total := time.Since(t0)
	fmt.Printf("\ndone. %d steps in %.3fs (%.1f steps/s)  floor=%.3f\n",
		*stepsFlag, total.Seconds(), float64(*stepsFlag)/total.Seconds(), bestFloor)

	// Final save
	saveFullCheckpoint(*stepsFlag, bestFloor)
	fmt.Printf("floor: %.3f at best\n", bestFloor)
}
