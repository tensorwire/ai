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

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/helix"
	"github.com/tensorwire/mongoose"
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
		explicit := map[string]bool{}
		fs.Visit(func(f *flag.Flag) { explicit[f.Name] = true })

		cfgInt := func(flagName, cfgKey string, target *int) {
			if explicit[flagName] {
				return
			}
			if v, ok := cfg[cfgKey].(float64); ok {
				*target = int(v)
			}
		}
		cfgInt("dim", "hidden_size", dimFlag)
		cfgInt("layers", "num_hidden_layers", layersFlag)
		cfgInt("heads", "num_attention_heads", headsFlag)
		cfgInt("kv-heads", "num_key_value_heads", kvHeadsFlag)
		cfgInt("ffn-dim", "intermediate_size", ffnDimFlag)
		resumeST, err = gguf.OpenSafeTensors(ckptDir)
		if err != nil { log.Fatalf("Load checkpoint: %v", err) }
		fmt.Printf("Resuming from %s (dim=%d layers=%d)\n", ckptDir, *dimFlag, *layersFlag)
	}

	if *dataPath == "" { log.Fatal("data required: ai train data=<file>") }

	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	if te == nil { log.Fatal("TensorEngine not available") }
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok { log.Fatal("train-cuda requires CUDA") }

	if !mongoose.LoadKernels() {
		log.Fatal("CUDA kernels required — compile kernels/mongoose.cu")
	}
	log.Println("[ai] CUDA kernels loaded")

	// Multi-GPU: only init when >1 GPU available
	var mc *mongoose.MultiCUDA
	nGPUs := 0
	devCount := mongoose.CUDADeviceCount()
	if devCount >= 2 {
		mc = mongoose.NewMultiCUDA()
		if mc != nil {
			nGPUs = mc.DeviceCount
			log.Printf("[ai] multi-GPU: %d devices available", nGPUs)
		}
	}

	var sched *mongoose.Scheduler
	if nGPUs >= 2 {
		sched = mongoose.NewScheduler(eng)
	}
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
	helixDispatch := nativeFP16 && nGPUs >= 2
	if helixDispatch {
		nativeFP16 = false // helix dispatch replaces the old layer-split path
		log.Println("[ai] Helix Dispatch: interleaved position parallelism")
	} else if nativeFP16 {
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

	// m = positions per GPU for Helix Dispatch
	m := n
	if nGPUs >= 2 { m = n / nGPUs }

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

	// Download tensor to host — handles both GPU 0 and remote devices
	toHost := func(t *mongoose.Tensor, dev int) []float32 {
		if dev == 0 || !multiGPU {
			return te.ToHost(t)
		}
		return mc.DownloadFP32FromDevice(dev, t.DevicePtr(), t.Size)
	}

	// FP16 shadows — only when FP16 training is active
	type fp16W struct{ wq, wk, wv, wo, gate, up, down, norm1, norm2 *mongoose.Tensor }
	fp16 := make([]fp16W, nLayers)
	if useFP16Training {
		for l := range fp16 {
			dev := layerDev[l]
			toFP16 := func(t *mongoose.Tensor) *mongoose.Tensor {
				host := toHost(t, dev)
				if dev == 0 || !multiGPU {
					return cuda.FromHostFP16(host, t.Shape)
				}
				return mc.FromHostFP16OnDevice(dev, host, t.Shape, cuda.FromHostFP16)
			}
			fp16[l] = fp16W{
				wq: toFP16(lays[l].wq), wk: toFP16(lays[l].wk), wv: toFP16(lays[l].wv),
				wo: toFP16(lays[l].wo), gate: toFP16(lays[l].gate), up: toFP16(lays[l].up),
				down: toFP16(lays[l].down), norm1: toFP16(lays[l].norm1), norm2: toFP16(lays[l].norm2),
			}
		}
	}
	var embedFP16, finalNormFP16, fp16Scratch *mongoose.Tensor
	if useFP16Training {
		embedFP16 = cuda.FromHostFP16(te.ToHost(embed), embed.Shape)
		finalNormFP16 = cuda.FromHostFP16(te.ToHost(finalNorm), finalNorm.Shape)
		maxActSize := n * dim
		if n*ffnDim > maxActSize { maxActSize = n * ffnDim }
		if n*vocabSize > maxActSize { maxActSize = n * vocabSize }
		fp16Scratch = cuda.AllocFP16Tensor(maxActSize, []int{maxActSize})
	}

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
		if !useFP16Training { return }
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
	type layerAS struct{ wq, wk, wv, wo, gate, up as }
	layAS := make([]layerAS, nLayers)
	for l := range layAS {
		dev := layerDev[l]
		layAS[l] = layerAS{
			wq: newAS(dev, dim * dim), wk: newAS(dev, kvDim * dim), wv: newAS(dev, kvDim * dim),
			wo: newAS(dev, dim * dim), gate: newAS(dev, ffnDim * dim), up: newAS(dev, ffnDim * dim),
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

	// === HELIX DISPATCH: per-GPU buffers for interleaved position parallelism ===
	type hdBufs struct {
		h16, normed16, Q16, K16, V16, attnOut16 *mongoose.Tensor
		xIn16, xMid16, normed216                 *mongoose.Tensor
		gatePre16, upOut16, ffnMid16, dx16       *mongoose.Tensor
		// Full K,V for attention (all positions, after all-gather)
		Kfull, Vfull *mongoose.Tensor
		// Backward
		dH16, dFfn16, dGate16, dUp16 *mongoose.Tensor
		dN1, dN2, dx32               *mongoose.Tensor
		dAttnOut, dQ, dK, dV         *mongoose.Tensor
		dWQ, dWK, dWV, dWO           *mongoose.Tensor
		dWGate, dWUp, dWDown         *mongoose.Tensor
		rmsScale1, rmsScale2         *mongoose.Tensor
		// FP32 scratch for attention backward (needs FP32 Q,K,V)
		Q32, K32, V32 *mongoose.Tensor
	}
	var hdLayerBufs [][]hdBufs // hdLayerBufs[dev][layer]
	var hdHidden16 []unsafe.Pointer   // per-GPU FP16 hidden (m rows)
	var hdFP16 [][]fp16W              // hdFP16[dev][layer] = FP16 weight replicas
	var hdEmbedFP16 []*mongoose.Tensor
	var hdFinalNormFP16 []*mongoose.Tensor
	var hdRopeCos, hdRopeSin []unsafe.Pointer // per-GPU RoPE tables
	// Per-GPU FP32 master weights + Helix M/V for optimizer
	type hdOptState struct {
		lays []layer
		gate_m, gate_v, up_m, up_v     []*mongoose.Tensor
		wq_m, wq_v, wo_m, wo_v         []*mongoose.Tensor
		wk_m, wk_v, wv_m, wv_v         []*mongoose.Tensor
	}
	var hdOpt []hdOptState

	if helixDispatch {
		hdLayerBufs = make([][]hdBufs, nGPUs)
		hdHidden16 = make([]unsafe.Pointer, nGPUs)
		hdFP16 = make([][]fp16W, nGPUs)
		hdEmbedFP16 = make([]*mongoose.Tensor, nGPUs)
		hdFinalNormFP16 = make([]*mongoose.Tensor, nGPUs)
		hdRopeCos = make([]unsafe.Pointer, nGPUs)
		hdRopeSin = make([]unsafe.Pointer, nGPUs)
		hdOpt = make([]hdOptState, nGPUs)

		for d := 0; d < nGPUs; d++ {
			if d > 0 { mongoose.SetDevice(d) }

			// Allocate FP16 hidden buffer (m rows per GPU)
			hdHidden16[d] = mc.AllocFP16OnDevice(d, m*dim)

			// FP16 weight replicas
			hdFP16[d] = make([]fp16W, nLayers)
			for l := range fp16 {
				if d == 0 {
					hdFP16[d][l] = fp16[l]
				} else {
					p2p := func(src *mongoose.Tensor) *mongoose.Tensor {
						ptr := mc.AllocFP16OnDevice(d, src.Size)
						mc.PeerCopyInto(0, src.DevicePtr(), d, ptr, src.Size*2)
						return mongoose.TensorFromDevicePtr(ptr, src.Size)
					}
					hdFP16[d][l] = fp16W{
						wq: p2p(fp16[l].wq), wk: p2p(fp16[l].wk), wv: p2p(fp16[l].wv),
						wo: p2p(fp16[l].wo), gate: p2p(fp16[l].gate), up: p2p(fp16[l].up),
						down: p2p(fp16[l].down), norm1: p2p(fp16[l].norm1), norm2: p2p(fp16[l].norm2),
					}
				}
			}
			if d == 0 {
				hdEmbedFP16[d] = embedFP16
				hdFinalNormFP16[d] = finalNormFP16
			} else {
				p2pT := func(src *mongoose.Tensor) *mongoose.Tensor {
					ptr := mc.AllocFP16OnDevice(d, src.Size)
					mc.PeerCopyInto(0, src.DevicePtr(), d, ptr, src.Size*2)
					return mongoose.TensorFromDevicePtr(ptr, src.Size)
				}
				hdEmbedFP16[d] = p2pT(embedFP16)
				hdFinalNormFP16[d] = p2pT(finalNormFP16)
			}

			// RoPE tables (FP32, small)
			if d == 0 {
				hdRopeCos[d] = ropeCos.DevicePtr()
				hdRopeSin[d] = ropeSin.DevicePtr()
			} else {
				ropeBytes := seqLen * halfHead * 4
				hdRopeCos[d] = mc.AllocBytesOnDevice(d, ropeBytes)
				hdRopeSin[d] = mc.AllocBytesOnDevice(d, ropeBytes)
				mc.PeerCopyInto(0, ropeCos.DevicePtr(), d, hdRopeCos[d], ropeBytes)
				mc.PeerCopyInto(0, ropeSin.DevicePtr(), d, hdRopeSin[d], ropeBytes)
			}

			// Per-layer activation + gradient buffers (m rows, not n)
			hdLayerBufs[d] = make([]hdBufs, nLayers)
			for l := range hdLayerBufs[d] {
				a16 := func(nElem int) *mongoose.Tensor {
					if d == 0 { return cuda.AllocFP16Tensor(nElem, []int{nElem}) }
					ptr := mc.AllocFP16OnDevice(d, nElem)
					return mongoose.TensorFromDevicePtr(ptr, nElem)
				}
				a32 := func(nElem int) *mongoose.Tensor {
					if d == 0 { return te.Zeros([]int{nElem}) }
					return mc.ZerosFP32OnDevice(d, nElem)
				}
				hdLayerBufs[d][l] = hdBufs{
					h16: a16(m*dim), normed16: a16(m*dim),
					Q16: a16(m*dim), K16: a16(m*kvDim), V16: a16(m*kvDim),
					attnOut16: a16(m*dim),
					xIn16: a16(m*dim), xMid16: a16(m*dim),
					normed216: a16(m*dim),
					gatePre16: a16(m*ffnDim), upOut16: a16(m*ffnDim),
					ffnMid16: a16(m*ffnDim), dx16: a16(m*dim),
					Kfull: a16(n*kvDim), Vfull: a16(n*kvDim),
					dH16: a16(m*dim), dFfn16: a16(m*ffnDim),
					dGate16: a16(m*ffnDim), dUp16: a16(m*ffnDim),
					dN1: a32(m*dim), dN2: a32(m*dim), dx32: a32(m*dim),
					dAttnOut: a32(m*dim),
					dQ: a32(m*dim), dK: a32(m*kvDim), dV: a32(m*kvDim),
					Q32: a32(m*dim), K32: a32(m*kvDim), V32: a32(m*kvDim),
					dWQ: a32(dim*dim), dWK: a32(kvDim*dim), dWV: a32(kvDim*dim),
					dWO: a32(dim*dim),
					dWGate: a32(ffnDim*dim), dWUp: a32(ffnDim*dim), dWDown: a32(dim*ffnDim),
					rmsScale1: a32(m), rmsScale2: a32(m),
				}
			}

			// Per-GPU FP32 master weights + Helix M/V (initialized identically from GPU 0)
			opt := hdOptState{
				lays:   make([]layer, nLayers),
				gate_m: make([]*mongoose.Tensor, nLayers), gate_v: make([]*mongoose.Tensor, nLayers),
				up_m: make([]*mongoose.Tensor, nLayers), up_v: make([]*mongoose.Tensor, nLayers),
				wq_m: make([]*mongoose.Tensor, nLayers), wq_v: make([]*mongoose.Tensor, nLayers),
				wo_m: make([]*mongoose.Tensor, nLayers), wo_v: make([]*mongoose.Tensor, nLayers),
				wk_m: make([]*mongoose.Tensor, nLayers), wk_v: make([]*mongoose.Tensor, nLayers),
				wv_m: make([]*mongoose.Tensor, nLayers), wv_v: make([]*mongoose.Tensor, nLayers),
			}
			if d == 0 {
				opt.lays = lays
				for l := range lays {
					la := &layAS[l]
					opt.gate_m[l] = la.gate.m; opt.gate_v[l] = la.gate.v
					opt.up_m[l] = la.up.m; opt.up_v[l] = la.up.v
					opt.wq_m[l] = la.wq.m; opt.wq_v[l] = la.wq.v
					opt.wo_m[l] = la.wo.m; opt.wo_v[l] = la.wo.v
					opt.wk_m[l] = la.wk.m; opt.wk_v[l] = la.wk.v
					opt.wv_m[l] = la.wv.m; opt.wv_v[l] = la.wv.v
				}
			} else {
				for l := range lays {
					p2p32 := func(src *mongoose.Tensor) *mongoose.Tensor {
						t := mc.ZerosFP32OnDevice(d, src.Size)
						mc.PeerCopyInto(0, src.DevicePtr(), d, t.DevicePtr(), src.Size*4)
						return t
					}
					z32 := func(sz int) *mongoose.Tensor { return mc.ZerosFP32OnDevice(d, sz) }
					opt.lays[l] = layer{
						wq: p2p32(lays[l].wq), wk: p2p32(lays[l].wk), wv: p2p32(lays[l].wv),
						wo: p2p32(lays[l].wo), gate: p2p32(lays[l].gate), up: p2p32(lays[l].up),
						down: p2p32(lays[l].down), norm1: p2p32(lays[l].norm1), norm2: p2p32(lays[l].norm2),
					}
					sz := lays[l].gate.Size
					opt.gate_m[l] = z32(sz); opt.gate_v[l] = z32(sz)
					opt.up_m[l] = z32(sz); opt.up_v[l] = z32(sz)
					opt.wq_m[l] = z32(lays[l].wq.Size); opt.wq_v[l] = z32(lays[l].wq.Size)
					opt.wo_m[l] = z32(lays[l].wo.Size); opt.wo_v[l] = z32(lays[l].wo.Size)
					opt.wk_m[l] = z32(lays[l].wk.Size); opt.wk_v[l] = z32(lays[l].wk.Size)
					opt.wv_m[l] = z32(lays[l].wv.Size); opt.wv_v[l] = z32(lays[l].wv.Size)
				}
			}
			hdOpt[d] = opt
		}
		mc.SyncAll()
		mongoose.SetDevice(0)
		// Helix Dispatch needs h16Scratch and normedFinal16 on GPU 0 for loss computation
		h16Scratch = cuda.AllocFP16Tensor(n*dim, []int{n*dim})
		normedFinal16 = cuda.AllocFP16Tensor(n*dim, []int{n*dim})
		log.Printf("[helix-dispatch] %d GPUs, m=%d positions each, %d layers", nGPUs, m, nLayers)
	}

	nParams := vocabSize*dim + dim
	for range lays {
		nParams += dim + dim*dim + kvDim*dim*2 + dim*dim + dim + ffnDim*dim*2 + dim*ffnDim
	}

	fmt.Printf("ai train — GPU kernels + Helix DNA optimizer (FP16=%v native=%v helix=%v)\n", useFP16Training, nativeFP16, helixDispatch)
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

	if sched != nil {
		sched.CalibrateMatMul(n, dim, dim)
		sched.CalibrateMatMul(n, dim, kvDim)
		sched.CalibrateMatMul(n, dim, ffnDim)
		sched.CalibrateMatMul(n, ffnDim, dim)
		sched.CalibrateMatMul(n, dim, vocabSize)
		sched.CalibrateAll(mongoose.NormKey(dim), func(e mongoose.Engine) {
			d := make([]float32, dim)
			w := make([]float32, dim)
			for i := range w { w[i] = 1 }
			e.RMSNorm(d, w, 1e-6)
		})
		log.Printf("[scheduler] calibrated %d GPUs, %d op shapes", sched.NumGPUs(), 6)
	}

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

		if helixDispatch {
			// === HELIX DISPATCH: interleaved position parallelism ===
			// Both GPUs run ALL layers simultaneously on their share of positions.
			// Zero pipeline stalls. Only K,V exchange for attention per layer.

			// Convert hidden(FP32) → FP16 into a full-size temp buffer on GPU 0
			fullH16 := cuda.AllocFP16Tensor(n*dim, []int{n*dim})
			mongoose.KFP32ToFP16(hidden.DevicePtr(), fullH16.DevicePtr(), n*dim)

			// Scatter: extract interleaved rows to each GPU
			mongoose.InterleaveExtract(fullH16.DevicePtr(), hdHidden16[0], n, dim, nGPUs, 0, 2)
			for d := 1; d < nGPUs; d++ {
				scatterBuf := cuda.AllocFP16Tensor(m*dim, []int{m*dim})
				mongoose.InterleaveExtract(fullH16.DevicePtr(), scatterBuf.DevicePtr(), n, dim, nGPUs, d, 2)
				mc.PeerCopyInto(0, scatterBuf.DevicePtr(), d, hdHidden16[d], m*dim*2)
			}
			mc.SyncAll()

			// Each GPU runs all layers on its m rows
			for d := 0; d < nGPUs; d++ {
				mongoose.SetDevice(d)
				h16Ptr := hdHidden16[d]

				for li := range lays {
					hb := &hdLayerBufs[d][li]
					w := &hdFP16[d][li]
					cosP, sinP := hdRopeCos[d], hdRopeSin[d]

					// Save xIn16 for backward
					mongoose.KCopy(hb.xIn16.DevicePtr(), h16Ptr, m*dim*2)
					mongoose.KCopy(hb.h16.DevicePtr(), h16Ptr, m*dim*2)

					// RMSNorm
					mongoose.KRMSNormOutSaveFP16(hb.h16.DevicePtr(), hb.normed16.DevicePtr(),
						w.norm1.DevicePtr(), hb.rmsScale1.DevicePtr(), m, dim)

					// QKV GEMMs (m rows × full weight)
					mc.MatMulFP16TransBOnDevice(d, hb.normed16.DevicePtr(), w.wq.DevicePtr(), hb.Q16.DevicePtr(), m, dim, dim)
					mc.MatMulFP16TransBOnDevice(d, hb.normed16.DevicePtr(), w.wk.DevicePtr(), hb.K16.DevicePtr(), m, dim, kvDim)
					mc.MatMulFP16TransBOnDevice(d, hb.normed16.DevicePtr(), w.wv.DevicePtr(), hb.V16.DevicePtr(), m, dim, kvDim)

					// RoPE on local positions only
					// Need per-GPU RoPE: GPU d's positions are [d, d+nGPUs, d+2*nGPUs, ...]
					// The cos/sin tables are indexed by position. Each GPU needs its interleaved subset.
					// For now: use full tables, RoPE kernel handles m positions starting at pos=0
					// TODO: extract interleaved cos/sin or pass position offsets
					mongoose.KRoPEFP16(hb.Q16.DevicePtr(), cosP, sinP, m, dim, headDim, heads)
					mongoose.KRoPEFP16(hb.K16.DevicePtr(), cosP, sinP, m, kvDim, headDim, kvHeads)

					// All-gather K,V for attention — each GPU needs all positions
					// Each GPU has m rows of K,V. Gather into Kfull[n, kvDim], Vfull[n, kvDim].
					// GPU d's K goes to rows [d, d+nGPUs, ...] in Kfull.
					// For 2 GPUs: GPU 0 has even K, GPU 1 has odd K.
					// Insert own rows, then receive other GPU's rows via P2P.
					mongoose.InterleaveInsert(hb.K16.DevicePtr(), hb.Kfull.DevicePtr(), n, kvDim, nGPUs, d, 2)
					mongoose.InterleaveInsert(hb.V16.DevicePtr(), hb.Vfull.DevicePtr(), n, kvDim, nGPUs, d, 2)
				}
				// Sync before K,V exchange
				mc.SyncDevice(d)
			}

			// K,V all-gather: each GPU sends its K,V rows to all other GPUs
			for li := range lays {
				for d := 0; d < nGPUs; d++ {
					srcK := hdLayerBufs[d][li].K16.DevicePtr()
					srcV := hdLayerBufs[d][li].V16.DevicePtr()
					for d2 := 0; d2 < nGPUs; d2++ {
						if d2 == d { continue }
						dstKfull := hdLayerBufs[d2][li].Kfull.DevicePtr()
						dstVfull := hdLayerBufs[d2][li].Vfull.DevicePtr()
						// Insert GPU d's rows into GPU d2's Kfull at interleaved positions
						// P2P the packed m rows, then interleave-insert on the target
						// For simplicity: P2P packed K to target, then insert
						tmpBuf := mc.AllocFP16OnDevice(d2, m*kvDim)
						mc.PeerCopyInto(d, srcK, d2, tmpBuf, m*kvDim*2)
						mongoose.SetDevice(d2)
						mongoose.InterleaveInsert(tmpBuf, dstKfull, n, kvDim, nGPUs, d, 2)
						tmpBufV := mc.AllocFP16OnDevice(d2, m*kvDim)
						mc.PeerCopyInto(d, srcV, d2, tmpBufV, m*kvDim*2)
						mongoose.InterleaveInsert(tmpBufV, dstVfull, n, kvDim, nGPUs, d, 2)
					}
				}
			}
			mc.SyncAll()

			// Continue forward: attention + FFN on each GPU
			for d := 0; d < nGPUs; d++ {
				mongoose.SetDevice(d)

				for li := range lays {
					hb := &hdLayerBufs[d][li]
					w := &hdFP16[d][li]

					// Attention: Q is local (m rows), K,V are full (n rows)
					// Need custom attention call with different Q vs K,V sizes
					// Q[m, dim], Kfull[n, kvDim], Vfull[n, kvDim] → attnOut[m, dim]
					mongoose.KCausalAttentionGQAFP16(hb.Q16.DevicePtr(), hb.Kfull.DevicePtr(), hb.Vfull.DevicePtr(),
						hb.attnOut16.DevicePtr(), m, dim, kvDim, heads, kvHeads)

					// Wo projection + residual
					mc.MatMulFP16TransBOnDevice(d, hb.attnOut16.DevicePtr(), w.wo.DevicePtr(), hb.dx16.DevicePtr(), m, dim, dim)
					mongoose.KFP16AddInPlace(hb.h16.DevicePtr(), hb.dx16.DevicePtr(), m*dim)

					// Save xMid16 for backward
					mongoose.KCopy(hb.xMid16.DevicePtr(), hb.h16.DevicePtr(), m*dim*2)

					// FFN
					mongoose.KRMSNormOutSaveFP16(hb.h16.DevicePtr(), hb.normed216.DevicePtr(),
						w.norm2.DevicePtr(), hb.rmsScale2.DevicePtr(), m, dim)

					mc.MatMulFP16TransBOnDevice(d, hb.normed216.DevicePtr(), w.gate.DevicePtr(), hb.gatePre16.DevicePtr(), m, dim, ffnDim)
					mc.MatMulFP16TransBOnDevice(d, hb.normed216.DevicePtr(), w.up.DevicePtr(), hb.upOut16.DevicePtr(), m, dim, ffnDim)

					mongoose.KSiLUGateMulFP16(hb.gatePre16.DevicePtr(), hb.upOut16.DevicePtr(), hb.ffnMid16.DevicePtr(), m*ffnDim)

					mc.MatMulFP16TransBOnDevice(d, hb.ffnMid16.DevicePtr(), w.down.DevicePtr(), hb.dx16.DevicePtr(), m, ffnDim, dim)
					mongoose.KFP16AddInPlace(hb.h16.DevicePtr(), hb.dx16.DevicePtr(), m*dim)
				}
				// h16 = final hidden for this GPU's positions
				hdHidden16[d] = hdLayerBufs[d][nLayers-1].h16.DevicePtr()
			}
			mc.SyncAll()

			// Gather: interleave results back to GPU 0 for loss computation
			mongoose.SetDevice(0)
			gatherBuf := h16Scratch.DevicePtr()
			mongoose.InterleaveInsert(hdHidden16[0], gatherBuf, n, dim, nGPUs, 0, 2)
			for d := 1; d < nGPUs; d++ {
				scratch0 := cuda.AllocFP16Tensor(m*dim, []int{m*dim})
				mc.PeerCopyInto(d, hdHidden16[d], 0, scratch0.DevicePtr(), m*dim*2)
				mongoose.InterleaveInsert(scratch0.DevicePtr(), gatherBuf, n, dim, nGPUs, d, 2)
			}

			// FP16→FP32 for loss
			mongoose.KFP16ToFP32(gatherBuf, hidden.DevicePtr(), n*dim)

			// Final RMSNorm + logits + loss (on GPU 0, full seqLen)
			mongoose.KFP32ToFP16(hidden.DevicePtr(), h16Scratch.DevicePtr(), n*dim)
			mongoose.KRMSNormOutSaveFP16(h16Scratch.DevicePtr(), normedFinal16.DevicePtr(),
				finalNormFP16.DevicePtr(), finalScales.DevicePtr(), n, dim)
			cuda.MatMulFP16TransposeBTInto(logitsBuf, normedFinal16, embedFP16, n, dim, vocabSize)

			zero(lossesGPU); zero(gradGPU)
			mongoose.KSoftmaxCE(logitsBuf.DevicePtr(), targetsGPU.DevicePtr(),
				lossesGPU.DevicePtr(), gradGPU.DevicePtr(), n, vocabSize, invN)

			cuda.Sync()
			lossH := te.ToHost(lossesGPU)
			for _, l := range lossH { stepLoss += l }
			stepLoss /= float32(n)

			// === BACKWARD (Helix Dispatch) ===
			// LM head backward on GPU 0 (full seqLen)
			mongoose.KFP16ToFP32(normedFinal16.DevicePtr(), normedFinal.DevicePtr(), n*dim)
			cuda.MatMulTransposeATInto(dEmbed, gradGPU, normedFinal, n, vocabSize, dim)
			cuda.MatMulTInto(dHidden, gradGPU, embed, n, vocabSize, dim)

			zero(dScratch)
			mongoose.KRMSNormBackward(dHidden.DevicePtr(), hidden.DevicePtr(),
				finalNorm.DevicePtr(), finalScales.DevicePtr(), dScratch.DevicePtr(), n, dim)
			cuda.CopyInto(dHidden, dScratch)

			// Scatter dHidden to GPUs (same interleave as forward)
			mongoose.KFP32ToFP16(dHidden.DevicePtr(), h16Scratch.DevicePtr(), n*dim)
			mongoose.InterleaveExtract(h16Scratch.DevicePtr(), hdHidden16[0], n, dim, nGPUs, 0, 2)
			for d := 1; d < nGPUs; d++ {
				scratch0 := cuda.AllocFP16Tensor(m*dim, []int{m*dim})
				mongoose.InterleaveExtract(h16Scratch.DevicePtr(), scratch0.DevicePtr(), n, dim, nGPUs, d, 2)
				mc.PeerCopyInto(0, scratch0.DevicePtr(), d, hdHidden16[d], m*dim*2)
			}
			mc.SyncAll()

			// Each GPU runs backward on all layers for its positions
			for d := 0; d < nGPUs; d++ {
				mongoose.SetDevice(d)
				dH16Ptr := hdHidden16[d]

				for li := nLayers - 1; li >= 0; li-- {
					hb := &hdLayerBufs[d][li]
					w := &hdFP16[d][li]
					cosP, sinP := hdRopeCos[d], hdRopeSin[d]

					// dH16 for this layer
					mongoose.KCopy(hb.dH16.DevicePtr(), dH16Ptr, m*dim*2)

					// FFN backward
					mc.MatMulFP16FP32OutOnDevice(d, hb.dH16.DevicePtr(), w.down.DevicePtr(), hb.dN1.DevicePtr(), m, dim, ffnDim)
					mc.MatMulFP16TransAFP32OutOnDevice(d, hb.dH16.DevicePtr(), hb.ffnMid16.DevicePtr(), hb.dWDown.DevicePtr(), m, dim, ffnDim)

					mongoose.KFP32ToFP16(hb.dN1.DevicePtr(), hb.dFfn16.DevicePtr(), m*ffnDim)
					mongoose.KSiLUGateBackwardFP16(hb.dFfn16.DevicePtr(), hb.gatePre16.DevicePtr(),
						hb.upOut16.DevicePtr(), hb.dGate16.DevicePtr(), hb.dUp16.DevicePtr(), m*ffnDim)

					mc.MatMulFP16FP32OutOnDevice(d, hb.dGate16.DevicePtr(), w.gate.DevicePtr(), hb.dN2.DevicePtr(), m, ffnDim, dim)
					mc.MatMulFP16FP32OutOnDevice(d, hb.dUp16.DevicePtr(), w.up.DevicePtr(), hb.dN1.DevicePtr(), m, ffnDim, dim)
					mongoose.KAddInPlace(hb.dN2.DevicePtr(), hb.dN1.DevicePtr(), m*dim)

					mc.MatMulFP16TransAFP32OutOnDevice(d, hb.dGate16.DevicePtr(), hb.normed216.DevicePtr(), hb.dWGate.DevicePtr(), m, ffnDim, dim)
					mc.MatMulFP16TransAFP32OutOnDevice(d, hb.dUp16.DevicePtr(), hb.normed216.DevicePtr(), hb.dWUp.DevicePtr(), m, ffnDim, dim)

					// RMSNorm backward (post-attn)
					mongoose.KFP16ToFP32(hb.xMid16.DevicePtr(), hb.dx32.DevicePtr(), m*dim)
					mongoose.KZero(hb.dN1.DevicePtr(), m*dim*4)
					mongoose.KRMSNormBackward(hb.dN2.DevicePtr(), hb.dx32.DevicePtr(),
						hdOpt[d].lays[li].norm2.DevicePtr(), hb.rmsScale2.DevicePtr(), hb.dN1.DevicePtr(), m, dim)
					mongoose.KFP32ToFP16(hb.dN1.DevicePtr(), hb.dx16.DevicePtr(), m*dim)
					mongoose.KFP16AddInPlace(hb.dH16.DevicePtr(), hb.dx16.DevicePtr(), m*dim)

					// Attention backward
					mongoose.KFP32ToFP16(hb.dH16.DevicePtr(), hb.dx16.DevicePtr(), m*dim)
					// Wait — dH16 is already FP16. The attention backward GEMM needs FP16→FP32.
					mc.MatMulFP16FP32OutOnDevice(d, hb.dH16.DevicePtr(), w.wo.DevicePtr(), hb.dAttnOut.DevicePtr(), m, dim, dim)
					mc.MatMulFP16TransAFP32OutOnDevice(d, hb.dH16.DevicePtr(), hb.attnOut16.DevicePtr(), hb.dWO.DevicePtr(), m, dim, dim)

					// Attention backward kernel (FP32)
					mongoose.KFP16ToFP32(hb.Q16.DevicePtr(), hb.Q32.DevicePtr(), m*dim)
					mongoose.KFP16ToFP32(hb.K16.DevicePtr(), hb.K32.DevicePtr(), m*kvDim)
					mongoose.KFP16ToFP32(hb.V16.DevicePtr(), hb.V32.DevicePtr(), m*kvDim)
					mongoose.KZero(hb.dQ.DevicePtr(), m*dim*4)
					mongoose.KZero(hb.dK.DevicePtr(), m*kvDim*4)
					mongoose.KZero(hb.dV.DevicePtr(), m*kvDim*4)
					mongoose.KCausalAttentionBackward(hb.Q32.DevicePtr(), hb.K32.DevicePtr(), hb.V32.DevicePtr(), hb.dAttnOut.DevicePtr(),
						hb.dQ.DevicePtr(), hb.dK.DevicePtr(), hb.dV.DevicePtr(), m, dim, kvDim, heads, kvHeads)

					mongoose.KRoPEBackward(hb.dQ.DevicePtr(), cosP, sinP, m, dim, headDim, heads)
					mongoose.KRoPEBackward(hb.dK.DevicePtr(), cosP, sinP, m, kvDim, headDim, kvHeads)

					// dN1 = dQ @ wq + dK @ wk + dV @ wv
					mongoose.KFP32ToFP16(hb.dQ.DevicePtr(), hb.dx16.DevicePtr(), m*dim)
					mc.MatMulFP16FP32OutOnDevice(d, hb.dx16.DevicePtr(), w.wq.DevicePtr(), hb.dN1.DevicePtr(), m, dim, dim)

					mongoose.KFP32ToFP16(hb.dK.DevicePtr(), hb.dx16.DevicePtr(), m*kvDim)
					mc.MatMulFP16FP32OutOnDevice(d, hb.dx16.DevicePtr(), w.wk.DevicePtr(), hb.dN2.DevicePtr(), m, kvDim, dim)
					mongoose.KAddInPlace(hb.dN1.DevicePtr(), hb.dN2.DevicePtr(), m*dim)

					mongoose.KFP32ToFP16(hb.dV.DevicePtr(), hb.dx16.DevicePtr(), m*kvDim)
					mc.MatMulFP16FP32OutOnDevice(d, hb.dx16.DevicePtr(), w.wv.DevicePtr(), hb.dN2.DevicePtr(), m, kvDim, dim)
					mongoose.KAddInPlace(hb.dN1.DevicePtr(), hb.dN2.DevicePtr(), m*dim)

					// dW for Q/K/V
					mongoose.KFP32ToFP16(hb.dQ.DevicePtr(), hb.dx16.DevicePtr(), m*dim)
					mc.MatMulFP16TransAFP32OutOnDevice(d, hb.dx16.DevicePtr(), hb.normed16.DevicePtr(), hb.dWQ.DevicePtr(), m, dim, dim)
					mongoose.KFP32ToFP16(hb.dK.DevicePtr(), hb.dx16.DevicePtr(), m*kvDim)
					mc.MatMulFP16TransAFP32OutOnDevice(d, hb.dx16.DevicePtr(), hb.normed16.DevicePtr(), hb.dWK.DevicePtr(), m, kvDim, dim)
					mongoose.KFP32ToFP16(hb.dV.DevicePtr(), hb.dx16.DevicePtr(), m*kvDim)
					mc.MatMulFP16TransAFP32OutOnDevice(d, hb.dx16.DevicePtr(), hb.normed16.DevicePtr(), hb.dWV.DevicePtr(), m, kvDim, dim)

					// RMSNorm backward (pre-attn)
					mongoose.KFP16ToFP32(hb.xIn16.DevicePtr(), hb.dx32.DevicePtr(), m*dim)
					mongoose.KZero(hb.dN2.DevicePtr(), m*dim*4)
					mongoose.KRMSNormBackward(hb.dN1.DevicePtr(), hb.dx32.DevicePtr(),
						hdOpt[d].lays[li].norm1.DevicePtr(), hb.rmsScale1.DevicePtr(), hb.dN2.DevicePtr(), m, dim)
					mongoose.KFP32ToFP16(hb.dN2.DevicePtr(), hb.dx16.DevicePtr(), m*dim)
					mongoose.KFP16AddInPlace(hb.dH16.DevicePtr(), hb.dx16.DevicePtr(), m*dim)

					dH16Ptr = hb.dH16.DevicePtr()
				}
			}
			mc.SyncAll()

		} else if nativeFP16 {
			// === NATIVE FP16 PATH ===
			// Hidden flows forward as FP16. Stays on whatever device owns the current layer.
			// Residual adds are device-local via KFP16AddInPlace. P2P only at device boundaries.
			// One FP32→FP16 conversion at embed, one FP16→FP32 at loss. Zero mid-pipeline conversions.

			// h16 = FP16 hidden state, lives on curDev
			curDev := 0
			h16Ptr := h16Scratch.DevicePtr()
			mongoose.KFP32ToFP16(hidden.DevicePtr(), h16Ptr, n*dim)

			for li := range lays {
				b := &bufs[li]
				f := &fp16Bufs[li]
				dev := layerDev[li]

				// P2P at device boundary — the ONLY cross-device transfer per transition
				if multiGPU && dev != curDev {
					mc.PeerCopyInto(curDev, h16Ptr, dev, f.h16.DevicePtr(), n*dim*2)
					mc.SyncDevice(curDev)
					mongoose.SetDevice(dev)
					curDev = dev
					h16Ptr = f.h16.DevicePtr()
				}

				// Save xIn16 for backward, copy h16 into layer's working buffer
				mongoose.KCopy(f.xIn16.DevicePtr(), h16Ptr, n*dim*2)
				if h16Ptr != f.h16.DevicePtr() {
					mongoose.KCopy(f.h16.DevicePtr(), h16Ptr, n*dim*2)
				}

				// Device-local pointers
				cosP, sinP := ropeCos.DevicePtr(), ropeSin.DevicePtr()
				if multiGPU && dev != 0 {
					cosP, sinP = devRes[dev].ropeCos, devRes[dev].ropeSin
				}

				// GEMM dispatch — device-local cuBLAS handle
				gemmBT16 := func(out, a, w *mongoose.Tensor, m, k, nn int) {
					if !multiGPU || dev == 0 {
						cuda.MatMulAllFP16TransposeBTInto(out, a, w, m, k, nn)
					} else {
						mc.MatMulFP16TransBOnDevice(dev, a.DevicePtr(), w.DevicePtr(), out.DevicePtr(), m, k, nn)
					}
				}

				// === ATTENTION BLOCK ===
				mongoose.KRMSNormOutSaveFP16(f.h16.DevicePtr(), f.normed16.DevicePtr(),
					fp16[li].norm1.DevicePtr(), b.rmsScale1.DevicePtr(), n, dim)

				gemmBT16(f.Q16, f.normed16, fp16[li].wq, n, dim, dim)
				gemmBT16(f.K16, f.normed16, fp16[li].wk, n, dim, kvDim)
				gemmBT16(f.V16, f.normed16, fp16[li].wv, n, dim, kvDim)

				mongoose.KRoPEFP16(f.Q16.DevicePtr(), cosP, sinP, n, dim, headDim, heads)
				mongoose.KRoPEFP16(f.K16.DevicePtr(), cosP, sinP, n, kvDim, headDim, kvHeads)

				mongoose.KCausalAttentionGQAFP16(f.Q16.DevicePtr(), f.K16.DevicePtr(), f.V16.DevicePtr(),
					f.attnOut16.DevicePtr(), n, dim, kvDim, heads, kvHeads)

				gemmBT16(f.dx16, f.attnOut16, fp16[li].wo, n, dim, dim)
				mongoose.KFP16AddInPlace(f.h16.DevicePtr(), f.dx16.DevicePtr(), n*dim)

				// Save xMid16 for backward
				mongoose.KCopy(f.xMid16.DevicePtr(), f.h16.DevicePtr(), n*dim*2)

				// === FFN BLOCK ===
				mongoose.KRMSNormOutSaveFP16(f.h16.DevicePtr(), f.normed216.DevicePtr(),
					fp16[li].norm2.DevicePtr(), b.rmsScale2.DevicePtr(), n, dim)

				gemmBT16(f.gatePre16, f.normed216, fp16[li].gate, n, dim, ffnDim)
				gemmBT16(f.upOut16, f.normed216, fp16[li].up, n, dim, ffnDim)

				mongoose.KSiLUGateMulFP16(f.gatePre16.DevicePtr(), f.upOut16.DevicePtr(), f.ffnMid16.DevicePtr(), n*ffnDim)

				gemmBT16(f.dx16, f.ffnMid16, fp16[li].down, n, ffnDim, dim)
				mongoose.KFP16AddInPlace(f.h16.DevicePtr(), f.dx16.DevicePtr(), n*dim)

				// h16 now lives on this device, ready for next layer
				h16Ptr = f.h16.DevicePtr()
			}

			// h16 is on curDev after last layer. P2P back to GPU 0 if needed for loss.
			if multiGPU && curDev != 0 {
				mc.PeerCopyInto(curDev, h16Ptr, 0, h16Scratch.DevicePtr(), n*dim*2)
				mc.SyncDevice(curDev)
				mongoose.SetDevice(0)
				h16Ptr = h16Scratch.DevicePtr()
			}

			// FP16→FP32 for loss computation on GPU 0
			mongoose.KFP16ToFP32(h16Ptr, hidden.DevicePtr(), n*dim)

			// Final RMSNorm + logits
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

			// dH16 flows backward as FP16. P2P only at device boundaries.
			// dW gradient GEMMs output FP32 and stay on the owning device.
			curDev = 0
			dH16Ptr := h16Scratch.DevicePtr()
			mongoose.KFP32ToFP16(dHidden.DevicePtr(), dH16Ptr, n*dim)

			for li := nLayers - 1; li >= 0; li-- {
				b := &bufs[li]
				f := &fp16Bufs[li]
				dev := layerDev[li]

				// P2P at device boundary (reverse direction)
				if multiGPU && dev != curDev {
					mc.PeerCopyInto(curDev, dH16Ptr, dev, f.dH16.DevicePtr(), n*dim*2)
					mc.SyncDevice(curDev)
					mongoose.SetDevice(dev)
					curDev = dev
					dH16Ptr = f.dH16.DevicePtr()
				} else if dH16Ptr != f.dH16.DevicePtr() {
					mongoose.KCopy(f.dH16.DevicePtr(), dH16Ptr, n*dim*2)
				}

				cosP, sinP := ropeCos.DevicePtr(), ropeSin.DevicePtr()
				if multiGPU && dev != 0 {
					cosP, sinP = devRes[dev].ropeCos, devRes[dev].ropeSin
				}

				// Device-local GEMM dispatch
				gemmFP16 := func(out, a, b16 *mongoose.Tensor, m, k, nn int) {
					if !multiGPU || dev == 0 {
						cuda.MatMulFP16Into(out, a, b16, m, k, nn)
					} else {
						mc.MatMulFP16FP32OutOnDevice(dev, a.DevicePtr(), b16.DevicePtr(), out.DevicePtr(), m, k, nn)
					}
				}
				gemmFP16TransA := func(out, a, b16 *mongoose.Tensor, m, k, nn int) {
					if !multiGPU || dev == 0 {
						cuda.MatMulFP16TransposeATInto(out, a, b16, m, k, nn)
					} else {
						mc.MatMulFP16TransAFP32OutOnDevice(dev, a.DevicePtr(), b16.DevicePtr(), out.DevicePtr(), m, k, nn)
					}
				}

				// === FFN BACKWARD ===
				// dH16 is FP16. dW GEMMs need FP16 in → FP32 out. dAct GEMMs need FP16 in → FP32 out.
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

				// RMSNorm backward (post-attn): FP32 kernel, device-local
				mongoose.KFP16ToFP32(f.xMid16.DevicePtr(), b.xMid.DevicePtr(), n*dim)
				mongoose.KZero(b.dN1.DevicePtr(), b.dN1.Size*4)
				mongoose.KRMSNormBackward(b.dN2.DevicePtr(), b.xMid.DevicePtr(),
					lays[li].norm2.DevicePtr(), b.rmsScale2.DevicePtr(), b.dN1.DevicePtr(), n, dim)
				// dN1 is FP32 RMSNorm result → convert to FP16, add to dH16
				mongoose.KFP32ToFP16(b.dN1.DevicePtr(), f.dx16.DevicePtr(), n*dim)
				mongoose.KFP16AddInPlace(f.dH16.DevicePtr(), f.dx16.DevicePtr(), n*dim)

				// Updated dH16 for attention backward GEMMs (already in f.dH16)

				// === ATTENTION BACKWARD ===
				gemmFP16(b.dAttnOut, f.dH16, fp16[li].wo, n, dim, dim)
				gemmFP16TransA(b.dWO, f.dH16, f.attnOut16, n, dim, dim)

				// Attention backward kernel is FP32 — convert saved FP16 activations
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

				// dN1 = dQ @ wq + dK @ wk + dV @ wv (FP16 in → FP32 out)
				mongoose.KFP32ToFP16(b.dQ.DevicePtr(), f.dQ16.DevicePtr(), n*dim)
				mongoose.KFP32ToFP16(b.dK.DevicePtr(), f.dK16.DevicePtr(), n*kvDim)
				mongoose.KFP32ToFP16(b.dV.DevicePtr(), f.dV16.DevicePtr(), n*kvDim)
				gemmFP16(b.dN1, f.dQ16, fp16[li].wq, n, dim, dim)
				gemmFP16(b.dN2, f.dK16, fp16[li].wk, n, kvDim, dim)
				mongoose.KAddInPlace(b.dN1.DevicePtr(), b.dN2.DevicePtr(), b.dN1.Size)
				gemmFP16(b.dN2, f.dV16, fp16[li].wv, n, kvDim, dim)
				mongoose.KAddInPlace(b.dN1.DevicePtr(), b.dN2.DevicePtr(), b.dN1.Size)

				// dW for Q/K/V
				gemmFP16TransA(b.dWQ, f.dQ16, f.normed16, n, dim, dim)
				gemmFP16TransA(b.dWK, f.dK16, f.normed16, n, kvDim, dim)
				gemmFP16TransA(b.dWV, f.dV16, f.normed16, n, kvDim, dim)

				// RMSNorm backward (pre-attn): FP32, device-local
				mongoose.KFP16ToFP32(f.xIn16.DevicePtr(), b.xIn.DevicePtr(), n*dim)
				mongoose.KZero(b.dN2.DevicePtr(), b.dN2.Size*4)
				mongoose.KRMSNormBackward(b.dN1.DevicePtr(), b.xIn.DevicePtr(),
					lays[li].norm1.DevicePtr(), b.rmsScale1.DevicePtr(), b.dN2.DevicePtr(), n, dim)
				mongoose.KFP32ToFP16(b.dN2.DevicePtr(), f.dx16.DevicePtr(), n*dim)
				mongoose.KFP16AddInPlace(f.dH16.DevicePtr(), f.dx16.DevicePtr(), n*dim)

				// dH16 now has accumulated gradient for this layer, ready for next
				dH16Ptr = f.dH16.DevicePtr()
			}

			// dH16 is on curDev. Convert back to FP32 on GPU 0 for embed gradient.
			if multiGPU && curDev != 0 {
				mc.PeerCopyInto(curDev, dH16Ptr, 0, h16Scratch.DevicePtr(), n*dim*2)
				mc.SyncDevice(curDev)
				mongoose.SetDevice(0)
				dH16Ptr = h16Scratch.DevicePtr()
			}
			mongoose.KFP16ToFP32(dH16Ptr, dHidden.DevicePtr(), n*dim)
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

		if helixDispatch {
			// Per-GPU Helix optimizer — each GPU updates its own FP32 master weights
			for d := 0; d < nGPUs; d++ {
				mongoose.SetDevice(d)
				opt := &hdOpt[d]
				for li := range lays {
					hb := &hdLayerBufs[d][li]
					l := &opt.lays[li]

					mongoose.KHelixDNAStep(
						l.gate.DevicePtr(), l.up.DevicePtr(),
						hb.dWGate.DevicePtr(), hb.dWUp.DevicePtr(),
						opt.gate_m[li].DevicePtr(), opt.up_m[li].DevicePtr(),
						opt.gate_v[li].DevicePtr(), opt.up_v[li].DevicePtr(),
						curLR, 0.9, 0.95, step, 1e-8, 0.1,
						r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
						3.0/5.0, l.gate.Size)

					mongoose.KHelixDNAStep(
						l.wq.DevicePtr(), l.wo.DevicePtr(),
						hb.dWQ.DevicePtr(), hb.dWO.DevicePtr(),
						opt.wq_m[li].DevicePtr(), opt.wo_m[li].DevicePtr(),
						opt.wq_v[li].DevicePtr(), opt.wo_v[li].DevicePtr(),
						curLR, 0.9, 0.95, step, 1e-8, 0.1,
						r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
						2.0/5.0, l.wq.Size)

					mongoose.KHelixDNAStep(
						l.wk.DevicePtr(), l.wv.DevicePtr(),
						hb.dWK.DevicePtr(), hb.dWV.DevicePtr(),
						opt.wk_m[li].DevicePtr(), opt.wv_m[li].DevicePtr(),
						opt.wk_v[li].DevicePtr(), opt.wv_v[li].DevicePtr(),
						curLR, 0.9, 0.95, step, 1e-8, 0.1,
						r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
						2.0/5.0, l.wk.Size)

					mongoose.KGradScale(hb.dWDown.DevicePtr(), -curLR, l.down.Size)
					mongoose.KAddInPlace(l.down.DevicePtr(), hb.dWDown.DevicePtr(), l.down.Size)

					// Device-local FP32→FP16 sync
					mongoose.KFP32ToFP16(l.wq.DevicePtr(), hdFP16[d][li].wq.DevicePtr(), l.wq.Size)
					mongoose.KFP32ToFP16(l.wk.DevicePtr(), hdFP16[d][li].wk.DevicePtr(), l.wk.Size)
					mongoose.KFP32ToFP16(l.wv.DevicePtr(), hdFP16[d][li].wv.DevicePtr(), l.wv.Size)
					mongoose.KFP32ToFP16(l.wo.DevicePtr(), hdFP16[d][li].wo.DevicePtr(), l.wo.Size)
					mongoose.KFP32ToFP16(l.gate.DevicePtr(), hdFP16[d][li].gate.DevicePtr(), l.gate.Size)
					mongoose.KFP32ToFP16(l.up.DevicePtr(), hdFP16[d][li].up.DevicePtr(), l.up.Size)
					mongoose.KFP32ToFP16(l.down.DevicePtr(), hdFP16[d][li].down.DevicePtr(), l.down.Size)
				}
			}
			mc.SyncAll()
			mongoose.SetDevice(0)
			mongoose.KAdamW(embed.DevicePtr(), dEmbed.DevicePtr(), embedAS.m.DevicePtr(), embedAS.v.DevicePtr(), curLR, 0.1, step, embed.Size)
		} else {

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

			helixPair := func(w1, w2, g1, g2, m1, m2, v1, v2 *mongoose.Tensor, bond float32, sz int) {
				mongoose.KHelixDNAStep(
					w1.DevicePtr(), w2.DevicePtr(),
					g1.DevicePtr(), g2.DevicePtr(),
					m1.DevicePtr(), m2.DevicePtr(),
					v1.DevicePtr(), v2.DevicePtr(),
					curLR, 0.9, 0.95, step, 1e-8, 0.1,
					r.Backbone1, r.Glyco1, r.Hbond1, r.Hbond2, r.Glyco2, r.Backbone2,
					bond, sz)
			}

			// gate↔up: GC pair (3 H-bonds)
			helixPair(l.gate, l.up, b.dWGate, b.dWUp, la.gate.m, la.up.m, la.gate.v, la.up.v, 3.0/5.0, l.gate.Size)

			// wq↔wo: AT pair (2 H-bonds) — query↔output projection
			helixPair(l.wq, l.wo, b.dWQ, b.dWO, la.wq.m, la.wo.m, la.wq.v, la.wo.v, 2.0/5.0, l.wq.Size)

			// wk↔wv: AT pair (2 H-bonds) — key↔value projection
			helixPair(l.wk, l.wv, b.dWK, b.dWV, la.wk.m, la.wv.m, la.wk.v, la.wv.v, 2.0/5.0, l.wk.Size)

			// down: rung-modulated SGD — FFN gradient signal captured by gate↔up pair
			mongoose.KGradScale(b.dWDown.DevicePtr(), -curLR, l.down.Size)
			mongoose.KAddInPlace(l.down.DevicePtr(), b.dWDown.DevicePtr(), l.down.Size)
		}
		if multiGPU { mongoose.SetDevice(0) }

		// embed: rung-modulated AdamW
		mongoose.KAdamW(embed.DevicePtr(), dEmbed.DevicePtr(), embedAS.m.DevicePtr(), embedAS.v.DevicePtr(), curLR, 0.1, step, embed.Size)

		// Sync FP32 master weights → FP16 for next forward
		syncFP16Weights()
		} // end !helixDispatch

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
