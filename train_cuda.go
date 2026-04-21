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
	// Three precision tiers:
	// dim < 512:  FP32 TF32 — dispatch overhead > GEMM savings
	// dim >= 512: FP16 GEMMs via cublasLt (gemmBT conversion path)
	// dim >= 512 + multi-GPU: native FP16 (P2P needs single precision)
	// Native FP16 element-wise kernels only engage with multi-GPU.
	// Single-GPU cublasLt conversion path is faster at all dims because
	// cublasLt's algorithm search beats cublasGemmEx for FP16.
	useFP16Training = dim >= 512
	nativeFP16 := useFP16Training && mongoose.FP16TrainKernelsLoaded() && nGPUs >= 2
	if nativeFP16 {
		log.Println("[ai] native FP16 training kernels available — zero conversion path")
	}

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

	// Layer l lives entirely on GPU (l % nGPUs).
	// FP32 master weights, FP16 shadows, Adam M/V, gradient buffers — all on the owning device.
	// Only dHidden crosses the wire at device boundaries.
	// Post-optimizer: sparse P2P of hot rows only (conductor mask).
	layerDev := make([]int, nLayers)
	multiGPU := nGPUs >= 2 && nativeFP16
	for l := 0; l < nLayers; l++ {
		if multiGPU { layerDev[l] = l % nGPUs } else { layerDev[l] = 0 }
	}
	if multiGPU {
		for d := 0; d < nGPUs; d++ {
			count := 0
			for _, dev := range layerDev { if dev == d { count++ } }
			log.Printf("[multi-gpu] GPU %d owns %d layers", d, count)
		}
	}

	// Allocate FP32 tensor on the device that owns this layer
	loadOrInitOnDev := func(dev int, name string, rows, cols int) *mongoose.Tensor {
		var data []float32
		if resumeST != nil && resumeST.HasTensor(name) {
			d, _, err := resumeST.ReadTensorFloat32(name)
			if err == nil && len(d) == rows*cols { data = d }
		}
		if data == nil {
			bound := float32(math.Sqrt(2.0 / float64(cols)))
			data = make([]float32, rows*cols)
			for i := range data { data[i] = bound * (2*rand.Float32() - 1) }
		}
		if dev == 0 || !multiGPU {
			return te.FromHost(data, []int{rows, cols})
		}
		return mc.FromHostFP32OnDevice(dev, data, []int{rows, cols})
	}
	loadOrOnesOnDev := func(dev int, name string, sz int) *mongoose.Tensor {
		var data []float32
		if resumeST != nil && resumeST.HasTensor(name) {
			d, _, err := resumeST.ReadTensorFloat32(name)
			if err == nil && len(d) == sz { data = d }
		}
		if data == nil {
			data = make([]float32, sz)
			for i := range data { data[i] = 1.0 }
		}
		if dev == 0 || !multiGPU {
			return te.FromHost(data, []int{1, sz})
		}
		return mc.FromHostFP32OnDevice(dev, data, []int{1, sz})
	}

	// Embed + final norm always on GPU 0 (loss computation lives there)
	embed := loadOrInitOnDev(0, "model.embed_tokens.weight", vocabSize, dim)
	finalNorm := loadOrOnesOnDev(0, "model.norm.weight", dim)

	type layer struct{ wq, wk, wv, wo, gate, up, down, norm1, norm2 *mongoose.Tensor }
	lays := make([]layer, nLayers)
	for l := range lays {
		dev := layerDev[l]
		pfx := fmt.Sprintf("model.layers.%d.", l)
		lays[l] = layer{
			wq: loadOrInitOnDev(dev, pfx+"self_attn.q_proj.weight", dim, dim),
			wk: loadOrInitOnDev(dev, pfx+"self_attn.k_proj.weight", kvDim, dim),
			wv: loadOrInitOnDev(dev, pfx+"self_attn.v_proj.weight", kvDim, dim),
			wo: loadOrInitOnDev(dev, pfx+"self_attn.o_proj.weight", dim, dim),
			gate: loadOrInitOnDev(dev, pfx+"mlp.gate_proj.weight", ffnDim, dim),
			up: loadOrInitOnDev(dev, pfx+"mlp.up_proj.weight", ffnDim, dim),
			down: loadOrInitOnDev(dev, pfx+"mlp.down_proj.weight", dim, ffnDim),
			norm1: loadOrOnesOnDev(dev, pfx+"input_layernorm.weight", dim),
			norm2: loadOrOnesOnDev(dev, pfx+"post_attention_layernorm.weight", dim),
		}
	}

	// FP16 shadows — each on the same device as the FP32 master
	type fp16W struct{ wq, wk, wv, wo, gate, up, down, norm1, norm2 *mongoose.Tensor }
	fp16 := make([]fp16W, nLayers)
	for l := range fp16 {
		dev := layerDev[l]
		toFP16 := func(t *mongoose.Tensor) *mongoose.Tensor {
			if dev == 0 || !multiGPU {
				return cuda.FromHostFP16(te.ToHost(t), t.Shape)
			}
			// Convert on GPU 0 then P2P — no host round-trip for the FP16 data
			host := te.ToHost(t)
			return mc.FromHostFP16OnDevice(dev, host, t.Shape, cuda.FromHostFP16)
		}
		fp16[l] = fp16W{
			wq: toFP16(lays[l].wq), wk: toFP16(lays[l].wk), wv: toFP16(lays[l].wv),
			wo: toFP16(lays[l].wo), gate: toFP16(lays[l].gate), up: toFP16(lays[l].up),
			down: toFP16(lays[l].down), norm1: toFP16(lays[l].norm1), norm2: toFP16(lays[l].norm2),
		}
	}
	embedFP16 := cuda.FromHostFP16(te.ToHost(embed), embed.Shape)
	finalNormFP16 := cuda.FromHostFP16(te.ToHost(finalNorm), finalNorm.Shape)

	// FP16 scratch for activation conversion before GEMMs (GPU 0 only, fallback path)
	maxActSize := n * dim
	if n*ffnDim > maxActSize { maxActSize = n * ffnDim }
	if n*vocabSize > maxActSize { maxActSize = n * vocabSize }
	fp16Scratch := cuda.AllocFP16Tensor(maxActSize, []int{maxActSize})

	// Per-device RoPE tables and FP32 scratch for multi-GPU.
	// Remote-device layers need these on their own GPU — can't cross-device write FP32 scales.
	type devResources struct {
		ropeCos, ropeSin unsafe.Pointer // FP32 RoPE tables on this device
		rmsScale1, rmsScale2 unsafe.Pointer // FP32 [seqLen] scratch for RMSNorm backward
		xIn32, xMid32 unsafe.Pointer // FP32 [n*dim] scratch for RMSNorm backward
		dx32 unsafe.Pointer // FP32 [n*dim] scratch for backward
	}
	var devRes []devResources
	if multiGPU {
		devRes = make([]devResources, nGPUs)
		ropeBytes := seqLen * halfHead * 4 // FP32
		scaleBytes := n * 4                // FP32 [seqLen]
		hidBytes := n * dim * 4            // FP32 [n, dim]
		for d := 0; d < nGPUs; d++ {
			if d == 0 {
				devRes[d] = devResources{
					ropeCos: ropeCos.DevicePtr(), ropeSin: ropeSin.DevicePtr(),
				}
				continue
			}
			dr := devResources{}
			dr.ropeCos = mc.AllocBytesOnDevice(d, ropeBytes)
			dr.ropeSin = mc.AllocBytesOnDevice(d, ropeBytes)
			mc.PeerCopyInto(0, ropeCos.DevicePtr(), d, dr.ropeCos, ropeBytes)
			mc.PeerCopyInto(0, ropeSin.DevicePtr(), d, dr.ropeSin, ropeBytes)
			dr.rmsScale1 = mc.AllocBytesOnDevice(d, scaleBytes)
			dr.rmsScale2 = mc.AllocBytesOnDevice(d, scaleBytes)
			dr.xIn32 = mc.AllocBytesOnDevice(d, hidBytes)
			dr.xMid32 = mc.AllocBytesOnDevice(d, hidBytes)
			dr.dx32 = mc.AllocBytesOnDevice(d, hidBytes)
			devRes[d] = dr
		}
		mc.SyncAll()
		mongoose.SetDevice(0)
		log.Printf("[multi-gpu] per-device RoPE tables + FP32 scratch allocated")
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


	// FP32→FP16 sync: each device converts its own master weights locally.
	// No P2P for weight sync — everything lives on the owning device.
	syncFP16Weights := func() {
		for l := range lays {
			dev := layerDev[l]
			if multiGPU { mongoose.SetDevice(dev) }
			mongoose.KFP32ToFP16(lays[l].wq.DevicePtr(), fp16[l].wq.DevicePtr(), lays[l].wq.Size)
			mongoose.KFP32ToFP16(lays[l].wk.DevicePtr(), fp16[l].wk.DevicePtr(), lays[l].wk.Size)
			mongoose.KFP32ToFP16(lays[l].wv.DevicePtr(), fp16[l].wv.DevicePtr(), lays[l].wv.Size)
			mongoose.KFP32ToFP16(lays[l].wo.DevicePtr(), fp16[l].wo.DevicePtr(), lays[l].wo.Size)
			mongoose.KFP32ToFP16(lays[l].gate.DevicePtr(), fp16[l].gate.DevicePtr(), lays[l].gate.Size)
			mongoose.KFP32ToFP16(lays[l].up.DevicePtr(), fp16[l].up.DevicePtr(), lays[l].up.Size)
			mongoose.KFP32ToFP16(lays[l].down.DevicePtr(), fp16[l].down.DevicePtr(), lays[l].down.Size)
			mongoose.KFP32ToFP16(lays[l].norm1.DevicePtr(), fp16[l].norm1.DevicePtr(), lays[l].norm1.Size)
			mongoose.KFP32ToFP16(lays[l].norm2.DevicePtr(), fp16[l].norm2.DevicePtr(), lays[l].norm2.Size)
		}
		if multiGPU { mongoose.SetDevice(0) }
		mongoose.KFP32ToFP16(embed.DevicePtr(), embedFP16.DevicePtr(), embed.Size)
		mongoose.KFP32ToFP16(finalNorm.DevicePtr(), finalNormFP16.DevicePtr(), finalNorm.Size)
	}

	// Helix optimizer — DNA-coupled gradient descent
	hlx := helix.NewHelixOptimizer(lr, 0.9, 0.95, 1e-8, 0.1)

	// Adam state: M + V per weight, on owning device
	type as struct{ m, v *mongoose.Tensor }
	newAS := func(dev, sz int) as {
		if dev == 0 || !multiGPU {
			return as{te.Zeros([]int{sz}), te.Zeros([]int{sz})}
		}
		return as{mc.ZerosFP32OnDevice(dev, sz), mc.ZerosFP32OnDevice(dev, sz)}
	}
	embedAS := newAS(0, vocabSize * dim)
	type layerAS struct{ wq, wk, wv, wo, gate, up, down as }
	layAS := make([]layerAS, nLayers)
	for l := range layAS {
		dev := layerDev[l]
		layAS[l] = layerAS{
			wq: newAS(dev, dim * dim), wk: newAS(dev, kvDim * dim), wv: newAS(dev, kvDim * dim),
			wo: newAS(dev, dim * dim), gate: newAS(dev, ffnDim * dim), up: newAS(dev, ffnDim * dim),
			down: newAS(dev, dim * ffnDim),
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
		dev := layerDev[i]
		z := func(sz int) *mongoose.Tensor {
			if dev == 0 || !multiGPU { return te.Zeros([]int{sz}) }
			return mc.ZerosFP32OnDevice(dev, sz)
		}
		bufs[i] = fwdBuf{
			xIn: z(n*dim), normed: z(n*dim),
			Q: z(n*dim), K: z(n*kvDim), V: z(n*kvDim),
			attnOut: z(n*dim),
			xMid: z(n*dim), normed2: z(n*dim),
			gatePre: z(n*ffnDim), upOut: z(n*ffnDim),
			ffnMid: z(n*ffnDim),
			rmsScale1: z(n), rmsScale2: z(n),
			dFfnMid: z(n*ffnDim), dGate: z(n*ffnDim),
			dUp: z(n*ffnDim), dN2: z(n*dim),
			dx: z(n*dim),
			dAttnOut: z(n*dim), dQ: z(n*dim),
			dK: z(n*kvDim), dV: z(n*kvDim),
			dN1: z(n*dim),
			dWDown: z(dim*ffnDim), dWGate: z(ffnDim*dim),
			dWUp: z(ffnDim*dim), dWO: z(dim*dim),
			dWQ: z(dim*dim), dWK: z(kvDim*dim),
			dWV: z(kvDim*dim),
		}
	}
	if multiGPU { mongoose.SetDevice(0) }

	// Native FP16 per-layer buffers (only allocated when FP16 kernels available)
	type fp16Buf struct {
		h16, normed16, Q16, K16, V16, attnOut16 *mongoose.Tensor
		xIn16, xMid16, normed216                 *mongoose.Tensor
		gatePre16, upOut16, ffnMid16, dx16       *mongoose.Tensor
		dH16                                     *mongoose.Tensor // FP16 copy of dHidden for backward GEMMs
		dFfn16, dGate16, dUp16, dN216, dx16b     *mongoose.Tensor
		dAttn16, dQ16, dK16, dV16, dN116         *mongoose.Tensor
	}
	var fp16Bufs []fp16Buf
	var normedFinal16, h16Scratch *mongoose.Tensor
	// Per-device FP16 hidden transfer buffers for multi-GPU P2P
	var devHidden16 []unsafe.Pointer // devHidden16[dev] = FP16 hidden on that device
	if nativeFP16 {
		alloc16 := func(nElem int) *mongoose.Tensor {
			return cuda.AllocFP16Tensor(nElem, []int{nElem})
		}
		alloc16OnDev := func(dev, nElem int) *mongoose.Tensor {
			if dev == 0 || !multiGPU {
				return cuda.AllocFP16Tensor(nElem, []int{nElem})
			}
			ptr := mc.AllocFP16OnDevice(dev, nElem)
			return mongoose.TensorFromDevicePtr(ptr, nElem)
		}
		fp16Bufs = make([]fp16Buf, nLayers)
		for i := range fp16Bufs {
			dev := 0
			if multiGPU { dev = layerDev[i] }
			a16 := func(nElem int) *mongoose.Tensor { return alloc16OnDev(dev, nElem) }
			fp16Bufs[i] = fp16Buf{
				h16: a16(n * dim), normed16: a16(n * dim),
				Q16: a16(n * dim), K16: a16(n * kvDim), V16: a16(n * kvDim),
				attnOut16: a16(n * dim),
				xIn16: a16(n * dim), xMid16: a16(n * dim),
				normed216: a16(n * dim),
				gatePre16: a16(n * ffnDim), upOut16: a16(n * ffnDim),
				ffnMid16: a16(n * ffnDim), dx16: a16(n * dim),
				dH16: a16(n * dim),
				dFfn16: a16(n * ffnDim), dGate16: a16(n * ffnDim),
				dUp16: a16(n * ffnDim), dN216: a16(n * dim),
				dx16b: a16(n * dim),
				dAttn16: a16(n * dim), dQ16: a16(n * dim),
				dK16: a16(n * kvDim), dV16: a16(n * kvDim),
				dN116: a16(n * dim),
			}
		}
		normedFinal16 = alloc16(n * dim)
		h16Scratch = alloc16(n * dim)

		if multiGPU {
			devHidden16 = make([]unsafe.Pointer, nGPUs)
			for d := 0; d < nGPUs; d++ {
				devHidden16[d] = mc.AllocFP16OnDevice(d, n*dim)
			}
			log.Printf("[multi-gpu] per-device FP16 activation buffers allocated")
		}
	}

	nParams := vocabSize*dim + dim
	for range lays {
		nParams += dim + dim*dim + kvDim*dim*2 + dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Printf("ai train — GPU kernels + Helix DNA optimizer (FP16=%v native=%v)\n", useFP16Training, nativeFP16)
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

	// Download tensor to host — handles both GPU 0 and remote devices
	toHost := func(t *mongoose.Tensor, dev int) []float32 {
		if dev == 0 || !multiGPU {
			return te.ToHost(t)
		}
		return mc.DownloadFP32FromDevice(dev, t.DevicePtr(), t.Size)
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
				wH := toHost(proj.w, layerDev[li])
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
				wH := toHost(proj.w, layerDev[li])
				cols := proj.w.Size / (proj.w.Shape[0])
				idx := 0
				for _, r := range ckpt.rows {
					row := int(r)
					if row >= 0 && row < proj.w.Shape[0] && idx+cols <= len(saved) {
						copy(wH[row*cols:(row+1)*cols], saved[idx:idx+cols])
						idx += cols
					}
				}
				dev := layerDev[li]
				if dev == 0 || !multiGPU {
					cuda.UploadInto(proj.w, wH)
				} else {
					mc.UploadFP32OnDevice(dev, proj.w.DevicePtr(), wH)
				}
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
		if multiGPU { mc.SyncAll() }
		tensors := map[string]gguf.SaveTensor{
			"model.embed_tokens.weight": {Data: te.ToHost(embed), Shape: []int{vocabSize, dim}},
			"model.norm.weight":         {Data: te.ToHost(finalNorm), Shape: []int{dim}},
		}
		for li := range lays {
			dev := layerDev[li]
			pfx := fmt.Sprintf("model.layers.%d.", li)
			tensors[pfx+"self_attn.q_proj.weight"] = gguf.SaveTensor{Data: toHost(lays[li].wq, dev), Shape: []int{dim, dim}}
			tensors[pfx+"self_attn.k_proj.weight"] = gguf.SaveTensor{Data: toHost(lays[li].wk, dev), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.v_proj.weight"] = gguf.SaveTensor{Data: toHost(lays[li].wv, dev), Shape: []int{kvDim, dim}}
			tensors[pfx+"self_attn.o_proj.weight"] = gguf.SaveTensor{Data: toHost(lays[li].wo, dev), Shape: []int{dim, dim}}
			tensors[pfx+"mlp.gate_proj.weight"] = gguf.SaveTensor{Data: toHost(lays[li].gate, dev), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.up_proj.weight"] = gguf.SaveTensor{Data: toHost(lays[li].up, dev), Shape: []int{ffnDim, dim}}
			tensors[pfx+"mlp.down_proj.weight"] = gguf.SaveTensor{Data: toHost(lays[li].down, dev), Shape: []int{dim, ffnDim}}
			tensors[pfx+"input_layernorm.weight"] = gguf.SaveTensor{Data: toHost(lays[li].norm1, dev), Shape: []int{dim}}
			tensors[pfx+"post_attention_layernorm.weight"] = gguf.SaveTensor{Data: toHost(lays[li].norm2, dev), Shape: []int{dim}}
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

		// === FORWARD + BACKWARD ===
		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)

		var stepLoss float32
		invN := float32(1.0) / float32(n)

		if nativeFP16 {
			// === NATIVE FP16 PATH ===
			// All intra-layer ops in FP16. hidden/dHidden stay FP32 on GPU 0 for embed/loss.
			// Multi-GPU: P2P hidden(FP16) to target device at layer boundaries.
			// 2 conversions per layer (hidden FP32↔FP16) vs 42 in the conversion fallback.

			for li := range lays {
				b := &bufs[li]
				f := &fp16Bufs[li]
				dev := 0
				if multiGPU { dev = layerDev[li] }

				if !multiGPU || dev == 0 {
					// === SINGLE-GPU / GPU-0 LAYER ===
					mongoose.KFP32ToFP16(hidden.DevicePtr(), f.xIn16.DevicePtr(), n*dim)
					cuda.FP16Copy(f.h16, f.xIn16, n*dim*2)

					mongoose.KRMSNormOutSaveFP16(f.h16.DevicePtr(), f.normed16.DevicePtr(),
						fp16[li].norm1.DevicePtr(), b.rmsScale1.DevicePtr(), n, dim)

					cuda.MatMulAllFP16TransposeBTInto(f.Q16, f.normed16, fp16[li].wq, n, dim, dim)
					cuda.MatMulAllFP16TransposeBTInto(f.K16, f.normed16, fp16[li].wk, n, dim, kvDim)
					cuda.MatMulAllFP16TransposeBTInto(f.V16, f.normed16, fp16[li].wv, n, dim, kvDim)

					mongoose.KRoPEFP16(f.Q16.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
					mongoose.KRoPEFP16(f.K16.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)

					mongoose.KCausalAttentionGQAFP16(f.Q16.DevicePtr(), f.K16.DevicePtr(), f.V16.DevicePtr(),
						f.attnOut16.DevicePtr(), n, dim, kvDim, heads, kvHeads)

					cuda.MatMulAllFP16TransposeBTInto(f.dx16, f.attnOut16, fp16[li].wo, n, dim, dim)
					mongoose.KFP32AddFP16(hidden.DevicePtr(), f.dx16.DevicePtr(), n*dim)

					mongoose.KFP32ToFP16(hidden.DevicePtr(), f.xMid16.DevicePtr(), n*dim)

					mongoose.KRMSNormOutSaveFP16(f.xMid16.DevicePtr(), f.normed216.DevicePtr(),
						fp16[li].norm2.DevicePtr(), b.rmsScale2.DevicePtr(), n, dim)

					cuda.MatMulAllFP16TransposeBTInto(f.gatePre16, f.normed216, fp16[li].gate, n, dim, ffnDim)
					cuda.MatMulAllFP16TransposeBTInto(f.upOut16, f.normed216, fp16[li].up, n, dim, ffnDim)

					mongoose.KSiLUGateMulFP16(f.gatePre16.DevicePtr(), f.upOut16.DevicePtr(), f.ffnMid16.DevicePtr(), n*ffnDim)

					cuda.MatMulAllFP16TransposeBTInto(f.dx16, f.ffnMid16, fp16[li].down, n, ffnDim, dim)
					mongoose.KFP32AddFP16(hidden.DevicePtr(), f.dx16.DevicePtr(), n*dim)
				} else {
					// === REMOTE GPU LAYER ===
					// All weights (FP16 + FP32), activations, gradients live on this device.
					// Only dHidden crosses the wire.
					dr := &devRes[dev]

					// Convert hidden(FP32)→FP16 on GPU 0, P2P to target device
					mongoose.KFP32ToFP16(hidden.DevicePtr(), devHidden16[0], n*dim)
					mc.PeerCopyInto(0, devHidden16[0], dev, devHidden16[dev], n*dim*2)
					mc.SyncDevice(0)
					mongoose.SetDevice(dev)

					// Save xIn16 for backward
					mongoose.KCopy(f.xIn16.DevicePtr(), devHidden16[dev], n*dim*2)
					mongoose.KCopy(f.h16.DevicePtr(), devHidden16[dev], n*dim*2)

					// RMSNorm — fp16[li] weights are on this device now
					mongoose.KRMSNormOutSaveFP16(f.h16.DevicePtr(), f.normed16.DevicePtr(),
						fp16[li].norm1.DevicePtr(), b.rmsScale1.DevicePtr(), n, dim)

					// QKV GEMMs — fp16[li] weights are device-local
					mc.MatMulFP16TransBOnDevice(dev, f.normed16.DevicePtr(), fp16[li].wq.DevicePtr(), f.Q16.DevicePtr(), n, dim, dim)
					mc.MatMulFP16TransBOnDevice(dev, f.normed16.DevicePtr(), fp16[li].wk.DevicePtr(), f.K16.DevicePtr(), n, dim, kvDim)
					mc.MatMulFP16TransBOnDevice(dev, f.normed16.DevicePtr(), fp16[li].wv.DevicePtr(), f.V16.DevicePtr(), n, dim, kvDim)

					mongoose.KRoPEFP16(f.Q16.DevicePtr(), dr.ropeCos, dr.ropeSin, n, dim, headDim, heads)
					mongoose.KRoPEFP16(f.K16.DevicePtr(), dr.ropeCos, dr.ropeSin, n, kvDim, headDim, kvHeads)

					mongoose.KCausalAttentionGQAFP16(f.Q16.DevicePtr(), f.K16.DevicePtr(), f.V16.DevicePtr(),
						f.attnOut16.DevicePtr(), n, dim, kvDim, heads, kvHeads)

					mc.MatMulFP16TransBOnDevice(dev, f.attnOut16.DevicePtr(), fp16[li].wo.DevicePtr(), f.dx16.DevicePtr(), n, dim, dim)

					// P2P dx back to GPU 0 for residual
					mc.PeerCopyInto(dev, f.dx16.DevicePtr(), 0, devHidden16[0], n*dim*2)
					mc.SyncDevice(dev)
					mongoose.SetDevice(0)
					mongoose.KFP32AddFP16(hidden.DevicePtr(), devHidden16[0], n*dim)

					// Save mid-hidden, P2P to target device
					mongoose.KFP32ToFP16(hidden.DevicePtr(), devHidden16[0], n*dim)
					mc.PeerCopyInto(0, devHidden16[0], dev, devHidden16[dev], n*dim*2)
					mc.SyncDevice(0)
					mongoose.SetDevice(dev)
					mongoose.KCopy(f.xMid16.DevicePtr(), devHidden16[dev], n*dim*2)

					// Post-attn RMSNorm
					mongoose.KRMSNormOutSaveFP16(f.xMid16.DevicePtr(), f.normed216.DevicePtr(),
						fp16[li].norm2.DevicePtr(), b.rmsScale2.DevicePtr(), n, dim)

					mc.MatMulFP16TransBOnDevice(dev, f.normed216.DevicePtr(), fp16[li].gate.DevicePtr(), f.gatePre16.DevicePtr(), n, dim, ffnDim)
					mc.MatMulFP16TransBOnDevice(dev, f.normed216.DevicePtr(), fp16[li].up.DevicePtr(), f.upOut16.DevicePtr(), n, dim, ffnDim)

					mongoose.KSiLUGateMulFP16(f.gatePre16.DevicePtr(), f.upOut16.DevicePtr(), f.ffnMid16.DevicePtr(), n*ffnDim)

					mc.MatMulFP16TransBOnDevice(dev, f.ffnMid16.DevicePtr(), fp16[li].down.DevicePtr(), f.dx16.DevicePtr(), n, ffnDim, dim)

					// P2P dx back to GPU 0 for residual
					mc.PeerCopyInto(dev, f.dx16.DevicePtr(), 0, devHidden16[0], n*dim*2)
					mc.SyncDevice(dev)
					mongoose.SetDevice(0)
					mongoose.KFP32AddFP16(hidden.DevicePtr(), devHidden16[0], n*dim)
				}
			}

			// Ensure we're on GPU 0 for loss computation
			if multiGPU { mongoose.SetDevice(0) }

			// Final RMSNorm: FP32 hidden → FP16 normedFinal → FP32 logits (loss needs FP32)
			mongoose.KFP32ToFP16(hidden.DevicePtr(), h16Scratch.DevicePtr(), n*dim)
			mongoose.KRMSNormOutSaveFP16(h16Scratch.DevicePtr(), normedFinal16.DevicePtr(),
				finalNormFP16.DevicePtr(), finalScales.DevicePtr(), n, dim)

			// Logits GEMM: FP16 normed × FP16 embed → FP32 logits (for softmax/loss)
			cuda.MatMulFP16TransposeBTInto(logitsBuf, normedFinal16, embedFP16, n, dim, vocabSize)

			zero(lossesGPU); zero(gradGPU)
			mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
				lossesGPU.DevicePtr(), gradGPU.DevicePtr(), n, vocabSize, invN)

			cuda.Sync()
			lossH := te.ToHost(lossesGPU)
			for _, l := range lossH { stepLoss += l }
			stepLoss /= float32(n)

			// === BACKWARD (native FP16) ===
			// dW GEMMs: FP16 activations × FP16 grads → FP32 gradients
			// dHidden chain: convert dHidden→FP16 once per block, FP16 GEMM → FP32 out

			// LM head backward: gradGPU(FP32) × normedFinal16(FP16) → dEmbed(FP32)
			// normedFinal is FP16 now — need FP16→FP32 for the dW GEMM
			mongoose.KFP16ToFP32(normedFinal16.DevicePtr(), normedFinal.DevicePtr(), n*dim)
			cuda.MatMulTransposeATInto(dEmbed, gradGPU, normedFinal, n, vocabSize, dim)
			cuda.MatMulTInto(dHidden, gradGPU, embed, n, vocabSize, dim)

			zero(dScratch)
			mongoose.KRMSNormBackward(dHidden.DevicePtr(), hidden.DevicePtr(),
				finalNorm.DevicePtr(), finalScales.DevicePtr(), dScratch.DevicePtr(), n, dim)
			cuda.CopyInto(dHidden, dScratch)

			for li := nLayers - 1; li >= 0; li-- {
				b := &bufs[li]
				f := &fp16Bufs[li]
				dev := layerDev[li]
				remote := multiGPU && dev != 0

				// dH: device-local FP32 dHidden accumulator
				// GPU 0 → dHidden directly. Remote → b.dx as local copy.
				var dH *mongoose.Tensor
				if remote {
					mongoose.KFP32ToFP16(dHidden.DevicePtr(), devHidden16[0], n*dim)
					mc.PeerCopyInto(0, devHidden16[0], dev, devHidden16[dev], n*dim*2)
					mc.SyncDevice(0)
					mongoose.SetDevice(dev)
					mongoose.KFP16ToFP32(devHidden16[dev], b.dx.DevicePtr(), n*dim)
					dH = b.dx
				} else {
					dH = dHidden
				}

				mongoose.KFP32ToFP16(dH.DevicePtr(), f.dH16.DevicePtr(), n*dim)

				gemmFP16 := func(out, a, b16 *mongoose.Tensor, m, k, nn int) {
					if !remote {
						cuda.MatMulFP16Into(out, a, b16, m, k, nn)
					} else {
						mc.MatMulFP16FP32OutOnDevice(dev, a.DevicePtr(), b16.DevicePtr(), out.DevicePtr(), m, k, nn)
					}
				}
				gemmFP16TransA := func(out, a, b16 *mongoose.Tensor, m, k, nn int) {
					if !remote {
						cuda.MatMulFP16TransposeATInto(out, a, b16, m, k, nn)
					} else {
						mc.MatMulFP16TransAFP32OutOnDevice(dev, a.DevicePtr(), b16.DevicePtr(), out.DevicePtr(), m, k, nn)
					}
				}
				cosP, sinP := ropeCos.DevicePtr(), ropeSin.DevicePtr()
				if remote { cosP, sinP = devRes[dev].ropeCos, devRes[dev].ropeSin }

				// FFN backward
				gemmFP16(b.dFfnMid, f.dH16, fp16[li].down, n, dim, ffnDim)
				gemmFP16TransA(b.dWDown, f.dH16, f.ffnMid16, n, dim, ffnDim)

				mongoose.KFP32ToFP16(b.dFfnMid.DevicePtr(), f.dFfn16.DevicePtr(), n*ffnDim)
				mongoose.KSiLUGateBackwardFP16(f.dFfn16.DevicePtr(), f.gatePre16.DevicePtr(),
					f.upOut16.DevicePtr(), f.dGate16.DevicePtr(), f.dUp16.DevicePtr(), n*ffnDim)

				gemmFP16(b.dN2, f.dGate16, fp16[li].gate, n, ffnDim, dim)
				gemmFP16(b.dN1, f.dUp16, fp16[li].up, n, ffnDim, dim)
				mongoose.KAddInPlace(b.dN2.DevicePtr(), b.dN1.DevicePtr(), b.dN2.Size)

				gemmFP16TransA(b.dWGate, f.dGate16, f.normed216, n, ffnDim, dim)
				gemmFP16TransA(b.dWUp, f.dUp16, f.normed216, n, ffnDim, dim)

				// RMSNorm backward (post-attn)
				mongoose.KFP16ToFP32(f.xMid16.DevicePtr(), b.xMid.DevicePtr(), n*dim)
				mongoose.KZero(b.dN1.DevicePtr(), b.dN1.Size*4)
				mongoose.KRMSNormBackward(b.dN2.DevicePtr(), b.xMid.DevicePtr(),
					lays[li].norm2.DevicePtr(), b.rmsScale2.DevicePtr(), b.dN1.DevicePtr(), n, dim)
				mongoose.KAddInPlace(dH.DevicePtr(), b.dN1.DevicePtr(), n*dim)

				// Updated dHidden → FP16
				mongoose.KFP32ToFP16(dH.DevicePtr(), f.dH16.DevicePtr(), n*dim)

				// Attention backward
				gemmFP16(b.dAttnOut, f.dH16, fp16[li].wo, n, dim, dim)
				gemmFP16TransA(b.dWO, f.dH16, f.attnOut16, n, dim, dim)

				mongoose.KFP16ToFP32(f.Q16.DevicePtr(), b.Q.DevicePtr(), n*dim)
				mongoose.KFP16ToFP32(f.K16.DevicePtr(), b.K.DevicePtr(), n*kvDim)
				mongoose.KFP16ToFP32(f.V16.DevicePtr(), b.V.DevicePtr(), n*kvDim)
				mongoose.KZero(b.dQ.DevicePtr(), b.dQ.Size*4)
				mongoose.KZero(b.dK.DevicePtr(), b.dK.Size*4)
				mongoose.KZero(b.dV.DevicePtr(), b.dV.Size*4)
				mongoose.KCausalAttentionBackward(b.Q.DevicePtr(), b.K.DevicePtr(), b.V.DevicePtr(), b.dAttnOut.DevicePtr(),
					b.dQ.DevicePtr(), b.dK.DevicePtr(), b.dV.DevicePtr(), n, dim, kvDim, heads, kvHeads)

				mongoose.KRoPEBackward(b.dQ.DevicePtr(), cosP, sinP, n, dim, headDim, heads)
				mongoose.KRoPEBackward(b.dK.DevicePtr(), cosP, sinP, n, kvDim, headDim, kvHeads)

				mongoose.KFP32ToFP16(b.dQ.DevicePtr(), f.dQ16.DevicePtr(), n*dim)
				mongoose.KFP32ToFP16(b.dK.DevicePtr(), f.dK16.DevicePtr(), n*kvDim)
				mongoose.KFP32ToFP16(b.dV.DevicePtr(), f.dV16.DevicePtr(), n*kvDim)
				gemmFP16(b.dN1, f.dQ16, fp16[li].wq, n, dim, dim)
				gemmFP16(b.dN2, f.dK16, fp16[li].wk, n, kvDim, dim)
				mongoose.KAddInPlace(b.dN1.DevicePtr(), b.dN2.DevicePtr(), b.dN1.Size)
				gemmFP16(b.dN2, f.dV16, fp16[li].wv, n, kvDim, dim)
				mongoose.KAddInPlace(b.dN1.DevicePtr(), b.dN2.DevicePtr(), b.dN1.Size)

				gemmFP16TransA(b.dWQ, f.dQ16, f.normed16, n, dim, dim)
				gemmFP16TransA(b.dWK, f.dK16, f.normed16, n, kvDim, dim)
				gemmFP16TransA(b.dWV, f.dV16, f.normed16, n, kvDim, dim)

				// RMSNorm backward (pre-attn)
				mongoose.KFP16ToFP32(f.xIn16.DevicePtr(), b.xIn.DevicePtr(), n*dim)
				mongoose.KZero(b.dN2.DevicePtr(), b.dN2.Size*4)
				mongoose.KRMSNormBackward(b.dN1.DevicePtr(), b.xIn.DevicePtr(),
					lays[li].norm1.DevicePtr(), b.rmsScale1.DevicePtr(), b.dN2.DevicePtr(), n, dim)
				mongoose.KAddInPlace(dH.DevicePtr(), b.dN2.DevicePtr(), n*dim)

				if remote {
					// P2P dH back to GPU 0
					mongoose.KFP32ToFP16(dH.DevicePtr(), devHidden16[dev], n*dim)
					mc.PeerCopyInto(dev, devHidden16[dev], 0, devHidden16[0], n*dim*2)
					mc.SyncDevice(dev)
					mongoose.SetDevice(0)
					mongoose.KFP16ToFP32(devHidden16[0], dHidden.DevicePtr(), n*dim)
				}
			}
			if multiGPU { mongoose.SetDevice(0) }
		} else {
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

			// Fallback loss + backward
			zero(normedFinal); zero(finalScales)
			mongoose.KRMSNormOutSave(hidden.DevicePtr(), normedFinal.DevicePtr(),
				finalNorm.DevicePtr(), finalScales.DevicePtr(), n, dim)

			gemmBT(logitsBuf, normedFinal, embed, embedFP16, n, dim, vocabSize)

			zero(lossesGPU); zero(gradGPU)
			mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
				lossesGPU.DevicePtr(), gradGPU.DevicePtr(), n, vocabSize, invN)

			cuda.Sync()
			lossH := te.ToHost(lossesGPU)
			for _, l := range lossH { stepLoss += l }
			stepLoss /= float32(n)

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
			dev := layerDev[li]

			// Optimizer runs on the device that owns this layer
			if multiGPU { mongoose.SetDevice(dev) }

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
		if multiGPU { mongoose.SetDevice(0) }

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
