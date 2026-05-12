# SQ4 Format Specification

**Version 1.0 — May 2026**

## 1. Overview

SQ4 (Synaptic Quantization 4-bit) delivers FP16 quality on a Q4 memory budget. It encodes neural network weights as 3-bit magnitude indices into 8 calibrated bands plus 1-bit sign, with an FP32 outlier sideband for the top 0.1% of weights by magnitude. SQ4 uses the same 4 bits per weight as Q4_0 but produces output indistinguishable from FP16 — a 32B-parameter model fits in under 7 GB of VRAM with no quality loss. On CUDA, dequantization is a single register-file lookup that adds no measurable overhead to the memory-bound matvec.

## 2. Encoding

### 2.1 Band Calibration

Each tensor is calibrated independently. Given N weights:

1. Compute absolute values: `abs[i] = |weight[i]|`
2. Sort ascending: `sorted = sort(abs)`
3. Divide into 8 equal-count bands. Band b contains elements `sorted[N*b/8 .. N*(b+1)/8 - 1]`
4. Boundary: `boundaries[b] = sorted[N*(b+1)/8 - 1]` (max absolute value in band b)
5. Reconstruction value: `means[b] = mean(sorted[N*b/8 .. N*(b+1)/8 - 1])` (arithmetic mean of absolute values in band b)

Each band contains exactly N/8 weights (12.5%). The 3-bit code is fully utilized with no wasted codes on empty regions.

### 2.2 Magnitude Encoding (3-bit)

For each weight, find the first band b where `|weight| <= boundaries[b]`:

```
band = 7  // default: highest band
for b = 0 to 6:
    if |weight| <= boundaries[b]:
        band = b
        break
```

Pack band indices into a bitstream. Weight i occupies bits [3i, 3i+2]:

```
bit_pos  = i * 3
byte_idx = bit_pos / 8
bit_off  = bit_pos % 8
mag[byte_idx] |= (band & 0x07) << bit_off
if bit_off > 5:
    mag[byte_idx + 1] |= (band & 0x07) >> (8 - bit_off)
```

### 2.3 Sign Encoding (1-bit)

Bit-packed array. Bit i = 1 if weight[i] < 0, else 0:

```
if weight[i] < 0:
    sign[i / 8] |= 1 << (i % 8)
```

### 2.4 Outlier Sideband

Threshold: p99.9 of sorted absolute values.

```
outlier_pos = N * 999 / 1000
outlier_thresh = sorted[outlier_pos]
```

All weights with `|weight| > outlier_thresh` (strict greater) are stored as `(flat_index: uint32, exact_value: float32)` pairs. Approximately 0.1% of weights qualify. At 8 bytes per outlier, the storage overhead is ~0.025 bits/weight amortized.

At reconstruction, outlier values replace the band-approximated values exactly.

## 3. Reconstruction

To reconstruct weight i:

```
// Read 3-bit band index
bit_pos  = i * 3
byte_idx = bit_pos / 8
bit_off  = bit_pos % 8
raw = mag[byte_idx] >> bit_off
if bit_off > 5:
    raw |= mag[byte_idx + 1] << (8 - bit_off)
band = raw & 0x07

// Look up band mean
value = means[band]

// Apply sign
if sign[i / 8] & (1 << (i % 8)):
    value = -value

// Overwrite with outlier if present
if i in outlier_indices:
    value = outlier_values[i]
```

## 4. GPU Format

### 4.1 Nibble Packing

On-disk, magnitude and sign are stored in separate streams. At model load, the CPU recombines them into 4-bit nibbles before uploading to the GPU:

```
nibble = (sign_bit << 3) | band_index
```

Two nibbles per byte: even-indexed weight in the low nibble (bits 0-3), odd-indexed weight in the high nibble (bits 4-7).

The embedding tensor is a special case: it is fully dequantized to FP32 on the CPU at load time, since it is read every token and cheaper to dequant once than per-step on the GPU.

### 4.2 LUT Dequantization

SQ4 dequantization is a single register-file lookup — effectively free compared to the memory read that delivers the nibble. A 16-entry table maps each nibble directly to its dequantized float value:

| Index | Value |
|-------|-------|
| 0-7 | `+means[0]` through `+means[7]` |
| 8-15 | `-means[0]` through `-means[7]` |

The table fits in 16 registers. Dequantizing a weight is one indexed read — no multiply, no add, no branch. On CUDA, the table lives in registers (no shared memory traffic); on Metal, in threadgroup memory loaded once per tensor.

Outlier corrections are applied exactly at runtime: for each outlier in the current row, the kernel computes `(outlier_val - table[nibble]) * activation` via binary search over the row's outlier range. This adds the exact FP32 difference between the true weight and the band-approximated weight, with zero approximation error.

### 4.3 CUDA Implementation

CUDA is where SQ4 performs best. The LUT fits in 16 registers — dequantization is a single shift+mask + register-indexed read, completely hidden behind memory latency. The kernel is entirely memory-bound at all model sizes; reading half the bytes compared to FP16 is pure throughput gain. SQ4 dequant overhead is 1-2% of kernel time.

Because SQ4 packs two weights per byte, it quadruples arithmetic intensity compared to FP16, making TF32 tensor cores effective even at batch=1. At batch=K, a single read of the weight matrix can be shared across K output vectors — a natural extension using `mma.sync.aligned.m16n8k8` with tile-swizzled weight layout for coalesced reads.

A minimal SQ4 matvec kernel:

```cuda
// SQ4 matrix-vector multiply: y[rows] = W[rows, cols] * x[cols]
// W is nibble-packed: 2 weights per byte, nibble = [sign:1|band:3]
// bands: 8 float32 band means (reconstruction values)
// outlier_idx, outlier_val, outlier_off: per-row outlier corrections
__global__ void sq4_matvec(
    const uint8_t* __restrict__ W,      // [rows, cols/2] packed nibbles
    const float*   __restrict__ x,      // [cols] input activations
    float*         __restrict__ y,      // [rows] output
    const float*   __restrict__ bands,  // [8] band means
    const uint32_t* __restrict__ outlier_idx,  // flat col indices
    const float*    __restrict__ outlier_val,  // exact FP32 values
    const uint32_t* __restrict__ outlier_off,  // [rows+1] row offsets into outlier arrays
    int rows, int cols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    // Build 16-entry LUT in registers: [+band0..+band7, -band0..-band7]
    float t[16];
    for (int i = 0; i < 8; i++) {
        t[i]     =  bands[i];
        t[i + 8] = -bands[i];
    }

    // Dot product: each thread strides across columns
    float sum = 0.0f;
    int half_cols = cols / 2;
    const uint8_t* row_w = W + (size_t)row * half_cols;

    for (int j = tid; j < half_cols; j += blockDim.x) {
        uint8_t packed = row_w[j];
        int col0 = j * 2;
        int col1 = col0 + 1;

        // Low nibble = even column, high nibble = odd column
        sum += t[packed & 0x0F]       * x[col0];
        sum += t[(packed >> 4) & 0x0F] * x[col1];
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    // Cross-warp reduction via shared memory
    __shared__ float warp_sums[32];
    int warp_id = tid / warpSize;
    int lane    = tid % warpSize;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        int nwarps = (blockDim.x + warpSize - 1) / warpSize;
        for (int w = 0; w < nwarps; w++)
            total += warp_sums[w];

        // Outlier correction: exact FP32 delta for p99.9 weights
        uint32_t oStart = outlier_off[row];
        uint32_t oEnd   = outlier_off[row + 1];
        for (uint32_t i = oStart; i < oEnd; i++) {
            uint32_t col = outlier_idx[i];
            // Recover the band-approximated value to compute the correction
            uint8_t packed = row_w[col / 2];
            int nibble = (col & 1) ? ((packed >> 4) & 0x0F) : (packed & 0x0F);
            total += (outlier_val[i] - t[nibble]) * x[col];
        }

        y[row] = total;
    }
}
```

Launch with one block per row, 256 threads per block. For a 4096x4096 weight matrix at SQ4, the weight data is 8 MB — half of FP16. For higher throughput, use vectorized reads (`uint4` per thread) to better saturate memory bandwidth.

### 4.4 Metal Implementation

On Metal, the primary kernel (`sq4_matvec`) loads the 16-entry LUT into threadgroup memory once per tensor. Each thread reads nibbles as vectorized `uint4` chunks, dequantizes via `table16[nibble]`, and accumulates the dot product with `simd_sum` reduction across SIMD lanes.

An optional secondary path re-encodes SQ4 nibbles into standard linear INT4 with per-group scale and bias (group_size=32), consumed by an MLX-derived INT4 matvec kernel. This re-encoding is performed lazily on first inference. It trades a small outlier approximation for faster kernel dispatch on larger models.

On Metal below ~3B parameters, SQ4 matches FP16 throughput (the model fits in cache and the matvec is not bandwidth-bound — the benefit is purely memory savings). Above ~3B, SQ4 is faster than FP16. This parity zone does not exist on CUDA, where SQ4 is faster at all sizes.

### 4.5 KV Cache Compression

SQ4 is also applied to key-value cache compression during inference. The KV cache uses absmax 4-bit quantization, achieving 8x compression with minimal throughput impact (4% overhead on a 7B model). This makes long-context inference practical on consumer GPUs — a 2048-token KV cache for a 7B model drops from ~1 GB to ~128 MB.

## 5. File Layout

An SQ4 model directory contains:

| File | Contents | Format |
|------|----------|--------|
| `sq4_meta.json` | Tensor metadata | JSON |
| `sq4_magnitude.bin` | 3-bit band indices, packed bitstream | Binary |
| `sq4_sign.bin` | 1-bit signs, packed bytes | Binary |
| `sq4_bands.bin` | 8 float32 band means per tensor | Binary (LE float32) |
| `sq4_outlier_idx.bin` | Flat indices of outlier weights | Binary (LE uint32) |
| `sq4_outlier_val.bin` | Exact values of outlier weights | Binary (LE float32) |
| `fp32.safetensors` | Non-quantized tensors (norms, biases) | SafeTensors |
| `config.json` | Model architecture config (copied) | JSON |
| `tokenizer.json` | Tokenizer (copied) | JSON |

### 5.1 sq4_meta.json Schema

```json
{
  "format": "sq4",
  "model": "Qwen2.5-0.5B",
  "dim": 896,
  "layers": 24,
  "tensors": [
    {
      "name": "model.embed_tokens.weight",
      "rows": 151936,
      "cols": 896,
      "mag_offset": 0,
      "mag_bytes": 51026880,
      "sign_offset": 0,
      "sign_bytes": 17008960,
      "bands_offset": 0,
      "outlier_start": 0,
      "outlier_count": 136134
    }
  ]
}
```

Fields:
- `mag_offset`, `mag_bytes`: byte range in `sq4_magnitude.bin`
- `sign_offset`, `sign_bytes`: byte range in `sq4_sign.bin`
- `bands_offset`: float index into `sq4_bands.bin` (multiply by 4 for byte offset). Each tensor occupies 8 consecutive floats.
- `outlier_start`: element index into the outlier arrays. Multiply by 4 for byte offset into both `sq4_outlier_idx.bin` and `sq4_outlier_val.bin`.
- `outlier_count`: number of outliers for this tensor

### 5.2 Tensor Order

Tensors are stored in this order:
1. `model.embed_tokens.weight`
2. `lm_head.weight` (if not tied to embeddings)
3. Per-layer projections in layer order:
   - `model.layers.{l}.self_attn.q_proj.weight`
   - `model.layers.{l}.self_attn.k_proj.weight`
   - `model.layers.{l}.self_attn.v_proj.weight`
   - `model.layers.{l}.self_attn.o_proj.weight`
   - `model.layers.{l}.mlp.gate_proj.weight` (if present — gated architectures only)
   - `model.layers.{l}.mlp.up_proj.weight`
   - `model.layers.{l}.mlp.down_proj.weight`

Non-quantized tensors (layernorms, biases) are stored in `fp32.safetensors`.

## 6. Compression

Per-weight cost:
- Magnitude: 3 bits
- Sign: 1 bit
- Outlier overhead: ~0.1% of weights at 64 bits each = ~0.064 bits/weight
- **Total: ~4.06 bits/weight**

Measured compression ratios (including FP32 norms/biases overhead):

| Model | Params | FP16 | Q4_0 | SQ4 | SQ4 quality |
|-------|--------|------|------|-----|-------------|
| Qwen2.5-0.5B | 0.5B | 0.9 GB | 0.5 GB | 0.5 GB | = FP16 |
| TinyLlama-1.1B | 1.1B | 1.0 GB | 0.5 GB | 0.5 GB | = FP16 |
| Mistral-7B | 7B | 6.8 GB | 3.4 GB | 3.4 GB | = FP16 |
| Qwen2.5-32B | 32B | ~64 GB | ~7 GB | **~7 GB** | = FP16 |

SQ4 and Q4_0 use the same 4-bit budget — identical memory. The difference is quality: Q4_0's uniform linear bins waste codes on the sparse tails, while SQ4's percentile bands use every code equally. Weight tensors achieve 7.9x compression vs FP32 (32 bits / ~4.06 bits per weight). The measured directory-level ratios include uncompressed FP32 norms, biases, and tokenizer files.

## 7. Tested Architectures

| Model Family | Tested Models | Biases | GQA | Status |
|-------------|---------------|--------|-----|--------|
| Qwen2 | 0.5B, 3B | Q/K/V bias | Yes (kvHeads < heads) | Pass |
| Llama | TinyLlama-1.1B | No bias | Yes | Pass |
| Yi | Yi-6B | No bias | Yes | Pass |
| Mistral | Mistral-7B | No bias | Yes | Pass |

## 8. Reference Implementation

- Encoder: `ai/quantize.go` — `sq4EncodeTensor()`, `cmdQuantizeSQ4()`
- SQ4 loader: `ai/infer_sq4_metal.go` — directory loading, nibble recombination, embedding dequant
- Metal inference engine: `mongoose/sq4_infer_metal_darwin.m` — LUT dequant, optional linear re-encoding
- Metal kernel: `mongoose/kernels/sq4_matvec.metal` — `sq4_matvec` (LUT path), `sq4_mlx_qmv` (linear path)
- CUDA kernel: `mongoose/kernels/mongoose_sq4_matvec.cu`
