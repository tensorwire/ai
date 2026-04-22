package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"github.com/open-ai-org/mongoose"
)

// cmdProfile runs a training step with per-op GPU timing.
//
//	ai profile [dim=128] [data=file.txt]
func cmdProfile(args map[string]string) {
	dim := kvInt(args, "dim", 128)
	heads := kvInt(args, "heads", 4)
	kvHeads := kvInt(args, "kv", 2)
	layers := kvInt(args, "layers", 4)
	ffnDim := kvInt(args, "ffn", dim*2)
	seqLen := kvInt(args, "seq", 64)
	vocabSize := 256
	n := seqLen
	headDim := dim / heads
	kvDim := kvHeads * headDim

	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		log.Fatalf("profile requires a GPU (detected: %s)", eng.Name())
	}
	cuda, ok := eng.(*mongoose.CUDA)
	if !ok {
		log.Fatalf("profile currently requires CUDA (detected: %s). Metal and WebGPU support planned.", eng.Name())
	}
	mongoose.LoadKernels()

	fmt.Printf("ai profile — per-op GPU timing\n")
	fmt.Printf("  engine:  %s\n", eng.Name())
	fmt.Printf("  model:   dim=%d heads=%d kv=%d layers=%d ffn=%d seq=%d\n",
		dim, heads, kvHeads, layers, ffnDim, seqLen)
	fmt.Println()

	// Allocate minimal buffers
	embed := te.FromHost(make([]float32, vocabSize*dim), []int{vocabSize, dim})
	hidden := te.Zeros([]int{n, dim})
	tokGPU := te.Zeros([]int{n})
	normed := te.Zeros([]int{n, dim})
	rmsScale := te.Zeros([]int{n})
	Q := te.Zeros([]int{n, dim})
	K := te.Zeros([]int{n, kvDim})
	V := te.Zeros([]int{n, kvDim})
	attnOut := te.Zeros([]int{n, dim})
	gatePre := te.Zeros([]int{n, ffnDim})
	upOut := te.Zeros([]int{n, ffnDim})
	ffnMid := te.Zeros([]int{n, ffnDim})
	dx := te.Zeros([]int{n, dim})
	norm1 := te.Zeros([]int{1, dim})
	wq := te.Zeros([]int{dim, dim})
	wk := te.Zeros([]int{kvDim, dim})
	wv := te.Zeros([]int{kvDim, dim})
	wo := te.Zeros([]int{dim, dim})
	gate := te.Zeros([]int{ffnDim, dim})
	up := te.Zeros([]int{ffnDim, dim})
	down := te.Zeros([]int{dim, ffnDim})

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

	tokF := make([]float32, n)
	for i := range tokF {
		tokF[i] = math.Float32frombits(uint32(int32(rand.Intn(vocabSize))))
	}
	cuda.UploadInto(tokGPU, tokF)

	zero := func(t *mongoose.Tensor) {
		mongoose.KZero(t.DevicePtr(), t.Size*4)
	}

	// Warmup
	cuda.Sync()

	type opTime struct {
		name string
		us   float64
	}
	var ops []opTime

	timeOp := func(name string, fn func()) {
		cuda.Sync()
		t := time.Now()
		fn()
		cuda.Sync()
		ops = append(ops, opTime{name, float64(time.Since(t).Microseconds())})
	}

	// Profile one forward pass
	timeOp("embed_gather", func() {
		zero(hidden)
		mongoose.KEmbedGather2(hidden.DevicePtr(), embed.DevicePtr(), tokGPU.DevicePtr(), n, dim)
	})

	timeOp("rmsnorm", func() {
		zero(normed)
		zero(rmsScale)
		mongoose.KRMSNormOutSave(hidden.DevicePtr(), normed.DevicePtr(),
			norm1.DevicePtr(), rmsScale.DevicePtr(), n, dim)
	})

	timeOp("matmul_Q [n×dim→n×dim]", func() {
		cuda.MatMulTransposeBTInto(Q, normed, wq, n, dim, dim)
	})
	timeOp("matmul_K [n×dim→n×kvDim]", func() {
		cuda.MatMulTransposeBTInto(K, normed, wk, n, dim, kvDim)
	})
	timeOp("matmul_V [n×dim→n×kvDim]", func() {
		cuda.MatMulTransposeBTInto(V, normed, wv, n, dim, kvDim)
	})

	timeOp("rope", func() {
		mongoose.KRoPE(Q.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, dim, headDim, heads)
		mongoose.KRoPE(K.DevicePtr(), ropeCos.DevicePtr(), ropeSin.DevicePtr(), n, kvDim, headDim, kvHeads)
	})

	timeOp("attention_gqa", func() {
		zero(attnOut)
		mongoose.KCausalAttentionGQA(Q.DevicePtr(), K.DevicePtr(), V.DevicePtr(), attnOut.DevicePtr(),
			n, dim, kvDim, heads, kvHeads)
	})

	timeOp("matmul_O [n×dim→n×dim]", func() {
		cuda.MatMulTransposeBTInto(dx, attnOut, wo, n, dim, dim)
	})

	timeOp("matmul_gate [n×dim→n×ffn]", func() {
		cuda.MatMulTransposeBTInto(gatePre, normed, gate, n, dim, ffnDim)
	})
	timeOp("matmul_up [n×dim→n×ffn]", func() {
		cuda.MatMulTransposeBTInto(upOut, normed, up, n, dim, ffnDim)
	})

	timeOp("silu_gate_mul", func() {
		zero(ffnMid)
		mongoose.KSiLUGateMul(gatePre.DevicePtr(), upOut.DevicePtr(), ffnMid.DevicePtr(), n*ffnDim)
	})

	timeOp("matmul_down [n×ffn→n×dim]", func() {
		cuda.MatMulTransposeBTInto(dx, ffnMid, down, n, ffnDim, dim)
	})

	// Print results
	var total float64
	for _, op := range ops {
		total += op.us
	}

	fmt.Printf("%-35s %10s %6s\n", "Operation", "Time (µs)", "   %")
	fmt.Printf("%-35s %10s %6s\n", "---", "---", "---")
	for _, op := range ops {
		pct := op.us / total * 100
		fmt.Printf("%-35s %10.0f %5.1f%%\n", op.name, op.us, pct)
	}
	fmt.Printf("%-35s %10s %6s\n", "---", "---", "---")
	fmt.Printf("%-35s %10.0f\n", "total (1 layer fwd)", total)
	fmt.Printf("%-35s %10.0f\n", fmt.Sprintf("estimated %d layers", layers), total*float64(layers))
	fmt.Println()

	// Breakdown by category
	var matmulTime, kernelTime float64
	for _, op := range ops {
		if len(op.name) > 6 && op.name[:6] == "matmul" {
			matmulTime += op.us
		} else {
			kernelTime += op.us
		}
	}
	fmt.Printf("cuBLAS matmuls:   %8.0f µs (%.0f%%)\n", matmulTime, matmulTime/total*100)
	fmt.Printf("custom kernels:   %8.0f µs (%.0f%%)\n", kernelTime, kernelTime/total*100)
}
