//go:build darwin && cgo

package main

import (
	"math"
	"testing"
	"unsafe"

	"github.com/tensorwire/mongoose"
)

func TestMetalLoRA_DequantGEMM(t *testing.T) {
	eng := selectEngine("auto")
	mtl, ok := eng.(*mongoose.Metal)
	if !ok {
		t.Skip("Metal not available")
	}
	te := mongoose.AsTensorEngine(eng)
	if te == nil {
		t.Fatal("TensorEngine not available")
	}
	if !mtl.FusedTrainAvailable() {
		t.Fatal("fused_train.metallib required")
	}

	// Known FP32 weight: W = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
	// Quantize to INT8: scale per row
	rows, cols := 2, 3
	fp32 := []float32{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

	// Quantize
	int8Data := make([]int8, rows*cols)
	scales := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var absMax float32
		for c := 0; c < cols; c++ {
			v := fp32[r*cols+c]
			if v < 0 {
				v = -v
			}
			if v > absMax {
				absMax = v
			}
		}
		if absMax < 1e-10 {
			absMax = 1e-10
		}
		scales[r] = absMax
		invScale := float32(127.0) / absMax
		for c := 0; c < cols; c++ {
			qi := fp32[r*cols+c] * invScale
			if qi > 127 {
				qi = 127
			}
			if qi < -127 {
				qi = -127
			}
			int8Data[r*cols+c] = int8(qi)
		}
	}

	t.Logf("INT8 data: %v", int8Data)
	t.Logf("Scales: %v", scales)

	// Upload to GPU
	nElems := rows * cols
	q8Buf := mtl.AllocRaw(nElems, nElems, []int{rows, cols})
	mtl.UploadRaw(q8Buf, unsafe.Pointer(&int8Data[0]), nElems)
	scalesBuf := te.FromHost(scales, []int{rows})
	zeroDelta := te.Zeros([]int{nElems})
	dstBuf := te.Zeros([]int{nElems})

	// Dequant via standalone (non-fused) path first
	mtl.FusedBegin()
	mtl.FusedDequantDelta(q8Buf, scalesBuf, zeroDelta, dstBuf, nElems, cols)
	mtl.FusedEnd()

	// Read back
	result := te.ToHost(dstBuf)
	t.Logf("Dequant result: %v", result[:nElems])
	t.Logf("Expected ~:     %v", fp32)

	for i := 0; i < nElems; i++ {
		diff := math.Abs(float64(result[i] - fp32[i]))
		if diff > 0.1 {
			t.Errorf("dequant[%d] = %.4f, want ~%.4f (diff=%.4f)", i, result[i], fp32[i], diff)
		}
	}

	// Test GEMM with realistic dimensions (>= tile size)
	// A[64,32] identity-like, B[32,32] known values
	gM, gK, gN := 64, 32, 32
	gA := make([]float32, gM*gK)
	gB := make([]float32, gN*gK)
	for i := 0; i < gM && i < gK; i++ {
		gA[i*gK+i] = 1.0 // identity-ish
	}
	for i := range gB {
		gB[i] = float32(i%10) * 0.1
	}
	gAT := te.FromHost(gA, []int{gM, gK})
	gBT := te.FromHost(gB, []int{gN, gK})
	gOut := te.Zeros([]int{gM * gN})

	mtl.FusedBegin()
	mtl.FusedGemmF32BT(gAT, gBT, gOut, gM, gK, gN)
	mtl.FusedEnd()

	gC := te.ToHost(gOut)
	// Row 0 of output should be B[0..31,0..31] row 0 (since A row 0 = [1,0,...,0])
	// = B[0, 0..K-1] dot product with... wait, BT means C = A @ B^T
	// Row 0 of C = A[0,:] @ B^T[:,0..N-1] = A[0,:] dot B[j,:] for each j
	// A[0,:] = [1,0,...0] so C[0,j] = B[j,0]
	t.Logf("GEMM F32 BT[0,0]=%.4f (expect B[0,0]=%.1f)", gC[0], gB[0])
	t.Logf("GEMM F32 BT[0,1]=%.4f (expect B[1,0]=%.1f)", gC[1], gB[1*gK])
	t.Logf("GEMM F32 BT[1,0]=%.4f (expect B[0,1]=%.1f)", gC[gN], gB[1])

	if math.Abs(float64(gC[0]-gB[0])) > 0.01 {
		t.Errorf("GEMM F32 BT[0,0] = %.4f, want %.4f", gC[0], gB[0])
	}
	if math.Abs(float64(gC[1]-gB[gK])) > 0.01 {
		t.Errorf("GEMM F32 BT[0,1] = %.4f, want %.4f", gC[1], gB[gK])
	}

	// Also test small dimensions with the non-Metal4 fallback (tiled kernel)
	// Use MLP kernel GemmBT which we already verified
	smallW := mongoose.MtlBufFromHost([]float32{1, 2, 3, 4, 5, 6})
	smallIn := mongoose.MtlBufFromHost([]float32{1, 0, 0, 0, 1, 0})
	smallOut := mongoose.MtlBufZeros(4)
	mongoose.MtlGemmBT(smallIn, smallW, smallOut, 2, 3, 2)
	smallC := mongoose.MtlBufSharedSlice(smallOut, 4)
	t.Logf("MLP GemmBT small: [%.2f, %.2f, %.2f, %.2f] (expect [1,4,2,5])", smallC[0], smallC[1], smallC[2], smallC[3])
	smallExpected := []float32{1, 4, 2, 5}
	for i, exp := range smallExpected {
		if math.Abs(float64(smallC[i]-exp)) > 0.01 {
			t.Errorf("smallGEMM[%d] = %.4f, want %.4f", i, smallC[i], exp)
		}
	}
}

func TestMetalLoRA_LoRAForward(t *testing.T) {
	eng := selectEngine("auto")
	mtl, ok := eng.(*mongoose.Metal)
	if !ok {
		t.Skip("Metal not available")
	}
	te := mongoose.AsTensorEngine(eng)
	if !mtl.FusedTrainAvailable() {
		t.Fatal("fused_train.metallib required")
	}

	// Base weight W[4,3] (as dequanted FP32)
	W := te.FromHost([]float32{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
		1, 1, 0,
	}, []int{4, 3})

	// LoRA: A[4,2] = 0, B[2,3] = known values
	// With A=0, LoRA correction = 0, so out should just be x @ W^T
	rank := 2
	A := te.Zeros([]int{4, rank})
	B := te.FromHost([]float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, []int{rank, 3})

	input := te.FromHost([]float32{1, 2, 3}, []int{1, 3})
	out := te.Zeros([]int{1, 4})
	loraRank := te.Zeros([]int{1, rank})
	dLoraTemp := te.Zeros([]int{1, 4})

	// Forward: out = input @ W^T + input @ B^T @ A^T
	mtl.FusedBegin()
	mtl.FusedGemmBT(input, W, out, 1, 3, 4)
	mtl.FusedGemmBT(input, B, loraRank, 1, 3, rank)
	mtl.FusedGemmBT(loraRank, A, dLoraTemp, 1, rank, 4)
	mtl.FusedAddInPlace(out, dLoraTemp, 4)
	mtl.FusedEnd()

	result := te.ToHost(out)
	t.Logf("LoRA forward (A=0): [%.4f, %.4f, %.4f, %.4f]", result[0], result[1], result[2], result[3])

	// Expected: [1,2,3] @ [[1,0,0,1],[0,1,0,1],[0,0,1,0]] = [1, 2, 3, 3]
	expected := []float32{1, 2, 3, 3}
	for i, exp := range expected {
		diff := math.Abs(float64(result[i] - exp))
		if diff > 0.01 {
			t.Errorf("out[%d] = %.4f, want %.4f", i, result[i], exp)
		}
	}

	// Now set A to non-zero and verify LoRA correction
	AData := []float32{1, 0, 0, 1, 0, 0, 0, 0}
	mtl.UploadInto(A, AData)

	mtl.FusedBegin()
	mtl.FusedGemmBT(input, W, out, 1, 3, 4)
	mtl.FusedGemmBT(input, B, loraRank, 1, 3, rank)
	mtl.FusedGemmBT(loraRank, A, dLoraTemp, 1, rank, 4)
	mtl.FusedAddInPlace(out, dLoraTemp, 4)
	mtl.FusedEnd()

	result2 := te.ToHost(out)
	t.Logf("LoRA forward (A!=0): [%.4f, %.4f, %.4f, %.4f]", result2[0], result2[1], result2[2], result2[3])

	// LoRA: input @ B^T = [1,2,3] @ [[0.1,0.4],[0.2,0.5],[0.3,0.6]] = [1*0.1+2*0.2+3*0.3, 1*0.4+2*0.5+3*0.6]
	//      = [1.4, 3.2]
	// then [1.4, 3.2] @ A^T = [1.4, 3.2] @ [[1,0,0,0],[0,1,0,0]] = [1.4, 3.2, 0, 0]
	// Total = base + LoRA = [1+1.4, 2+3.2, 3+0, 3+0] = [2.4, 5.2, 3.0, 3.0]
	expected2 := []float32{2.4, 5.2, 3.0, 3.0}
	for i, exp := range expected2 {
		diff := math.Abs(float64(result2[i] - exp))
		if diff > 0.01 {
			t.Errorf("out[%d] = %.4f, want %.4f", i, result2[i], exp)
		}
	}
}

func TestMetalLoRA_RMSNormForwardBackward(t *testing.T) {
	eng := selectEngine("auto")
	mtl, ok := eng.(*mongoose.Metal)
	if !ok {
		t.Skip("Metal not available")
	}
	te := mongoose.AsTensorEngine(eng)
	if !mtl.FusedTrainAvailable() {
		t.Fatal("fused_train.metallib required")
	}

	dim := 4
	n := 2
	x := te.FromHost([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{n, dim})
	w := te.FromHost([]float32{1, 1, 1, 1}, []int{1, dim})
	scale := te.Zeros([]int{n})

	// RMSNorm forward: x_out = x / rms(x) * w
	mtl.FusedBegin()
	mtl.FusedRMSNorm(x, w, scale, n, dim)
	mtl.FusedEnd()

	result := te.ToHost(x)
	t.Logf("RMSNorm result: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
		result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7])

	// RMS for [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) = 2.7386
	// normalized: [0.3651, 0.7303, 1.0954, 1.4606]
	rms1 := float32(math.Sqrt(float64(1+4+9+16) / 4.0))
	for i := 0; i < dim; i++ {
		exp := float32(i+1) / rms1
		diff := math.Abs(float64(result[i] - exp))
		if diff > 0.01 {
			t.Errorf("rmsnorm[%d] = %.4f, want %.4f", i, result[i], exp)
		}
	}

	// Verify it didn't produce NaN
	for i := 0; i < n*dim; i++ {
		if math.IsNaN(float64(result[i])) {
			t.Fatalf("NaN at position %d", i)
		}
	}
}
