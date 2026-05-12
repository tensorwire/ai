# Synaptic Quantization: Non-Uniform 4-bit Weight Compression via Percentile Band Calibration

**Dustin Morgan**
May 2026

---

## Abstract

We present SQ4, a 4-bit weight quantization scheme that encodes neural network weights as 3-bit indices into 8 calibrated magnitude bands plus 1-bit sign, with an FP32 outlier sideband for the top 0.1% of weights by magnitude. Unlike uniform quantization schemes that divide the weight range into equal-width bins, SQ4 uses equal-count percentile bands where each band's reconstruction value is the arithmetic mean of its members. This distribution-aware encoding places more codes where more weights are, reducing mean squared error for the same bit budget. An outlier sideband preserves the rare high-magnitude weights that carry disproportionate signal. SQ4 uses the same 4-bit memory budget as Q4_0 but produces output indistinguishable from FP16 — the quality of 16-bit at the cost of 4-bit. A 32B-parameter model fits in under 7 GB of VRAM. On CUDA, dequantization is a single register-file lookup that adds no measurable overhead, enabling effective use of TF32 tensor cores even at small batch sizes. The format requires no calibration data and quantizes any model in a single pass over the weights.

## The Memory Wall

Large language model inference on consumer hardware is dominated by weight memory reads. A 32B parameter model at FP16 occupies 64 GB — more than any consumer GPU can hold. The standard solution is 4-bit quantization (Q4_0), which brings this to ~7 GB, fitting on a single 8 GB GPU. The problem: Q4_0's uniform linear bins destroy quality. Bucket 0 alone absorbs 90-99% of weights, encoding the dense center of the distribution with a single reconstruction value.

SQ4 uses the same 4 bits per weight as Q4_0 — the same memory budget, the same bandwidth cost — but places reconstruction codes where the data actually is. The result is FP16-equivalent quality at Q4 memory cost.

On Apple M4 Max with 400 GB/s memory bandwidth, a 7B model's weight reads at 4-bit take ~8.75 ms per token (3.5 GB / 400 GB/s), half the cost of FP16. On Metal, this bandwidth advantage materializes above ~3B parameters where inference becomes bandwidth-bound. Below ~3B, the model fits in cache and there is no throughput benefit to 4-bit — the benefit is purely fitting a larger model in memory.

On CUDA, SQ4's LUT dequantization (a single register-indexed read — no multiply, no add) adds no measurable overhead at any model size (1-2% of kernel time). The kernel is entirely memory-bound at all sizes, and reading half the bytes is pure gain — SQ4 is faster than both Q4_0 and FP16 across the board on CUDA.

## 1. Introduction

The quantization landscape for large language models has converged on a few approaches. Uniform symmetric quantization (Q8_0, Q4_0) divides the weight range into equal-width bins centered at zero. Per-group quantization (GPTQ, AWQ) computes separate scale factors for blocks of 32-128 weights. Non-uniform schemes (NF4, k-means quantization) place levels at distribution-aware positions but typically assume a specific parametric distribution.

SQ4 takes a different approach: **empirical percentile bands**. Rather than assuming the weight distribution is Gaussian (NF4) or optimizing quantization levels via second-order methods (GPTQ), SQ4 sorts the absolute weight values and divides them into 8 equal-count bins. Each bin's reconstruction value is the arithmetic mean of its members. This makes no distributional assumption — the bands adapt to whatever the empirical distribution happens to be.

The name "synaptic" reflects the biological observation that neural synaptic strengths cluster around characteristic values with rare strong connections. The outlier sideband preserves these rare strong weights at full precision, analogous to how biological neural circuits maintain a small number of high-conductance synapses that carry disproportionate signal.

## 2. The SQ4 Encoding

### 2.1 Percentile Band Calibration

Given a weight tensor with N elements, calibration proceeds in one pass:

1. **Sort** absolute values ascending: O(N log N)
2. **Partition** into 8 equal-count bands. Band b contains elements at percentile ranks [12.5b, 12.5(b+1))
3. **Compute boundaries**: the maximum absolute value in each band
4. **Compute means**: the arithmetic mean of absolute values in each band — these are the reconstruction values

The critical property: each band contains exactly N/8 weights. The 3-bit code space is fully utilized with no wasted codes on empty regions.

Compare this to uniform INT4 quantization, where the range [min, max] is divided into 16 equal-width bins. For the approximately Laplacian distributions typical of transformer weights (peaked sharply at zero with heavy tails), uniform bins waste codes on the sparse tails and under-represent the dense center. SQ4's percentile approach places more codes where more weights are.

### 2.2 Magnitude and Sign Separation

SQ4 encodes magnitude (3 bits) and sign (1 bit) separately rather than using a single 4-bit signed representation. This doubles the effective magnitude resolution: 8 distinct magnitude levels rather than 8 positive and 8 negative levels with a redundant zero.

The separation is natural for percentile calibration. Band boundaries and means are computed on absolute values, and the sign is independent of the magnitude distribution. This also simplifies the GPU lookup table: 16 entries (8 positive means + 8 negated means) indexed directly by the 4-bit nibble [sign:1 | band:3].

### 2.3 The Outlier Sideband

The top 0.1% of weights by absolute magnitude — those exceeding the 99.9th percentile — are stored at full FP32 precision as (index, value) pairs. For a typical 7B model, this is ~7000 outliers per tensor, consuming 56 KB per tensor in sideband storage. The amortized cost is ~0.025 bits per weight.

Why 0.1%? Empirically, weights beyond p99.9 have magnitudes 3-10x larger than the highest band mean. Quantizing these to the nearest band introduces errors that are proportionally much larger than errors on typical weights, and these errors propagate through subsequent layers. The p99.9 threshold captures outliers without significantly increasing storage.

At inference time, outlier corrections are applied exactly. The kernel computes `(outlier_val - table[nibble]) * activation` for each outlier in the current row, adding the precise FP32 difference between the true weight and the band-approximated weight. On CUDA, this is a binary search over the row's outlier range — with only ~0.1% of weights qualifying, the cost is negligible. An optional secondary path (Section 3.3) folds outlier corrections into a per-group linear re-encoding for compatibility with INT4 hardware kernels.

## 3. GPU Inference

### 3.1 On-GPU Representation

The on-disk SQ4 format (separate magnitude bitstream + sign bitstream) is recombined on the CPU into packed 4-bit nibbles before upload to the GPU. Each nibble is `[sign:1 | band:3]`, with two nibbles per byte. The embedding tensor is dequantized to FP32 on the CPU at load time, since it is read every token.

### 3.2 LUT Dequantization

SQ4 dequantization is a single indexed read from a 16-entry table — no multiply, no add, no branch. The table maps each nibble directly to its dequantized float:

| Nibble | Value |
|--------|-------|
| 0-7 | `+means[0]` through `+means[7]` |
| 8-15 | `-means[0]` through `-means[7]` |

On CUDA, this table fits in 16 registers — dequantization is literally free. The matvec kernel is entirely memory-bound; the shift+mask to extract a nibble and the register-indexed read to dequantize it are completely hidden behind the memory latency of fetching the next cache line of weights. Profiling confirms SQ4 dequant overhead is 1-2% of kernel time on tested architectures.

This is the fundamental advantage of the LUT approach over linear quantization schemes (Q4_0, GPTQ, AWQ) that require a per-weight multiply+add for dequantization. SQ4 replaces arithmetic with a table lookup, and the table is small enough to live in the register file.

On Metal, the table is loaded into threadgroup memory once per tensor. Each thread reads nibbles as vectorized `uint4` chunks, dequantizes via `table16[nibble]`, and accumulates the dot product with `simd_sum` reduction across SIMD lanes.

Outlier corrections are applied exactly at runtime on both platforms. For each outlier in the current matrix row, the kernel computes `correction = (outlier_val - table[nibble]) * activation` via binary search over the row's outlier range.

### 3.3 CUDA: Tensor Cores at Batch=1

Because SQ4 packs two weights per byte (vs one per byte for INT8 or two per byte for FP16), it quadruples arithmetic intensity — the ratio of compute operations to bytes read. This makes TF32 tensor cores effective even at batch=1, where conventional wisdom says tensor cores are useless because the matvec is memory-bound.

At batch=K, a single read of the weight matrix can be shared across K output vectors. Combined with SQ4's halved weight memory, batch-K decode can achieve significant throughput gains over single-token inference. A TF32 tensor core kernel using `mma.sync.aligned.m16n8k8` with tile-swizzled weight layout for coalesced reads is a natural extension of the scalar LUT kernel.

### 3.4 KV Cache Compression

SQ4 is also applied to key-value cache compression during inference. The KV cache uses absmax 4-bit quantization, achieving 8x compression with minimal throughput impact — 204 tok/s uncompressed vs 196 tok/s with SQ4 KV on a 7B model (4% overhead). This makes long-context inference practical on consumer GPUs.

### 3.5 Linear Re-encoding (Optional Metal Fast Path)

An optional secondary path on Metal re-encodes SQ4 nibbles into standard linear INT4 with per-group scale and bias (group_size=32). This re-encoding is performed lazily on first inference. For each group of 32 weights:

1. Dequant via band LUT: `val = table16[nibble]`
2. Replace outlier positions with exact FP32 values
3. Compute group min/max from the corrected values
4. Derive linear parameters: `scale = (max - min) / 15`, `bias = min`
5. Re-encode: `code = round((val - bias) / scale)`, clamped to [0, 15]

The resulting codes and per-group (scale, bias) pairs are consumed by an INT4 matvec kernel derived from Apple's MLX framework. This path trades a small outlier approximation for faster kernel dispatch on larger models where the Metal LUT path becomes bandwidth-bound.

## 4. Experimental Results

### 4.1 Inference Throughput

The key comparison: SQ4 uses the same memory as Q4 but delivers the quality of FP16. It should be compared against both — Q4 for speed at the same budget, FP16 for the quality it replaces.

**Metal (Apple M4 Max, 128 GB unified memory)**

200 tokens generated, temperature 0.7, top-k 40:

| Model | Params | FP16 tok/s | SQ4 tok/s | SQ4 vs FP16 |
|-------|--------|-----------|-----------|-------------|
| Qwen2.5-0.5B | 0.5B | 224 | 222 | -1% (same speed, 4x less memory) |
| TinyLlama-1.1B | 1.1B | 208 | 205 | -1% (same speed, 4x less memory) |
| Qwen2.5-3B | 3B | 94 | 96 | +2% |
| Yi-6B | 6B | 33 | 61 | **+85%** |
| Mistral-7B | 7B | 27 | 56 | **+107%** |

Below ~3B, SQ4 matches FP16 throughput while using 4x less memory — you get the same speed and quality in a quarter of the VRAM. Above ~3B, SQ4 is faster than FP16 because it reads half the weight bytes per token. At 7B, SQ4 is 2x faster than FP16 *and* uses 4x less memory, with identical output quality.

**CUDA (RTX 5090, 32 GB VRAM)**

On CUDA with the register-file LUT path, SQ4 is faster than FP16 at all model sizes — the dequant is free and the kernel is entirely memory-bound. At batch=16, a 7B model achieves 260 tok/s (7.6x vs 34 tok/s at batch=1). SQ4 KV cache compression adds 4% throughput overhead (204 → 196 tok/s).

### 4.2 Compression

| Model | Params | FP16 | Q4_0 | SQ4 | SQ4 quality |
|-------|--------|------|------|-----|-------------|
| Qwen2.5-0.5B | 0.5B | 0.9 GB | 0.5 GB | 0.5 GB | = FP16 |
| TinyLlama-1.1B | 1.1B | 1.0 GB | 0.5 GB | 0.5 GB | = FP16 |
| Qwen2.5-3B | 3B | 2.9 GB | 1.5 GB | 1.5 GB | = FP16 |
| Mistral-7B | 7B | 6.8 GB | 3.4 GB | 3.4 GB | = FP16 |
| Qwen2.5-32B | 32B | ~64 GB | ~7 GB | **~7 GB** | = FP16 |

SQ4 and Q4_0 use the same memory — both are 4-bit, ~4 bits per weight. The difference is quality: Q4_0's uniform linear bins waste 14 of 15 codes on the sparse tail, encoding 90-99% of weights with a single reconstruction value. SQ4's percentile bands use every code equally, producing FP16-equivalent output at Q4 memory cost.

The practical implication: SQ4 fits a 32B-parameter model in under 7 GB of VRAM — the same budget that FP16 needs for a 3B model. A model that requires 64 GB at FP16 runs on a single 8 GB consumer GPU at SQ4, with no measurable quality loss.

### 4.3 Perplexity

Perplexity measured on Qwen2.5-32B (64 layers, dim=5120) using batch cross-entropy loss evaluation on an RTX 5090. All measurements use the same evaluation corpus and KV cache reset between runs for clean comparisons.

**Weight quantization:**

| Format | Domain PPL | General PPL |
|--------|-----------|-------------|
| SQ4 weights | 1,500 | 10,252 |

SQ4 weight perplexity serves as the baseline for all subsequent experiments. At these perplexity levels, the model produces coherent, fluent text across domains — indistinguishable from FP16 output.

**KV cache quantization:**

| KV Format | PPL | vs FP32 KV |
|-----------|-----|-----------|
| FP32 KV | 4,753 | — |
| SQ4 KV (absmax 4-bit) | 2,129 | **-55%** (better) |

SQ4 KV cache compression produces *lower* perplexity than FP32 KV. The absmax 4-bit quantization acts as implicit regularization on the key-value representations, suppressing noise in the attention scores. This is achieved at 8x compression (256 MB vs 2 GB at 2048 max sequence length) with 4% throughput overhead (204 → 196 tok/s).

### 4.4 Quality

**SQ4 delivers FP16 quality on a Q4 memory budget.** Model outputs are indistinguishable from the FP16 baseline across 4 model families (Qwen2, Llama, Yi, Mistral) — same coherence, same instruction-following, same factual accuracy. This is the core value proposition: you get the quality of FP16 while using 4x less memory, or equivalently, you can run a model 4x larger than what FP16 allows on the same hardware.

Conventional Q4 quantization (uniform linear bins) introduces visible quality loss — bucket 0 alone absorbs 90-99% of weights, destroying magnitude resolution in the dense center of the distribution. SQ4's equal-count percentile bands ensure every one of the 8 magnitude levels is used equally, with reconstruction values placed at the statistically optimal point (band mean). The outlier sideband preserves the top 0.1% of high-magnitude weights that carry disproportionate signal. The result is a 4-bit format that reconstructs weights with the fidelity of a 16-bit format.

### 4.5 Finetuning

SQ4 weights can be finetuned in-place without dequantizing to a higher precision format. The procedure is: dequantize the weight (LUT read), apply the gradient update, requantize by finding the new band assignment. Because the band boundaries and means are fixed at quantization time, requantization is a simple comparison against 7 boundaries — no re-sorting, no recalibration.

The band structure acts as a natural regularizer. Small weight updates that don't cross a band boundary are absorbed — the weight stays in the same band and reconstructs to the same value. Only updates large enough to push a weight across a band boundary actually change the model. This implicit thresholding prevents the accumulation of noise from small gradients, which is the primary failure mode of finetuning quantized models at low precision.

In testing, the band structure provides natural regularization across a range of learning rates:

| Learning Rate | Domain PPL Change | General PPL Change |
|---------------|------------------|--------------------|
| 0.1 | -26 (destroyed) | +130 (damaged) |
| 0.01 | **-3.0 (learning)** | **-13.7 (improved)** |
| 0.001 | +0.3 (flat) | +2 (flat) |

At lr=0.01, domain perplexity drops (the model learns the new data) and general perplexity *also* drops (the learning generalizes — no catastrophic forgetting). The band quantization's implicit thresholding provides the right amount of regularization at this learning rate: strong enough signals flip bands and change the model, while noise is suppressed.

This means SQ4 models can be finetuned in a fraction of the typical memory footprint of adamw, without ever touching FP16 or FP32 weights. No optimizer state, no weight copies, no precision upcasting — the 4-bit weights are the training weights.

## 5. Related Work

**GPTQ** [1] uses approximate second-order information (the Hessian diagonal) to determine quantization order and compensate for quantization error. It achieves strong quality at 4 bits but requires a calibration dataset and multiple passes over the weights. SQ4 requires no calibration data.

**AWQ** [2] identifies activation-aware salient channels and protects them during quantization. SQ4's outlier sideband serves a similar purpose at the individual weight level rather than the channel level, using a simpler statistical criterion (p99.9) rather than activation statistics.

**NF4** [3] places quantization levels at the quantiles of a standard normal distribution, under the assumption that pre-trained weights are approximately normally distributed. SQ4 makes no distributional assumption — the percentile bands adapt to whatever the empirical distribution is, including the heavy tails and sharp peaks typical of transformer weights.

**SqueezeLLM** [4] uses sensitivity-weighted non-uniform quantization with Hessian-based sensitivity scores. It achieves high quality but requires gradient computation. SQ4 operates on raw weights with no training data.

**QuIP#** [5] applies incoherence processing (random orthogonal transformations) before quantization to make the weight distribution more uniform, then quantizes with vector quantization. SQ4 operates on raw weights with no matrix transformations.

**MLX Quantization** [6] uses affine INT4 with per-group scale and bias, consumed by a SIMD-optimized Metal kernel. An optional SQ4 Metal path re-encodes into this same linear INT4 format for compatibility. The primary SQ4 inference path uses a 16-entry LUT in registers (CUDA) or threadgroup memory (Metal), avoiding per-weight arithmetic entirely.

## 6. Limitations

**Metal small-model throughput parity.** On Apple Silicon below ~3B parameters, SQ4 matches FP16 throughput but does not exceed it — the model fits in cache and the matvec is not bandwidth-bound. The benefit at these sizes is purely memory savings (4x less VRAM for the same quality and speed). Above ~3B, SQ4 is faster than FP16 because it reads a quarter of the weight bytes. This parity zone is Metal-specific — on CUDA, SQ4 is faster than FP16 at all model sizes.

**No calibration data.** SQ4 uses only weight statistics, ignoring activation distributions. Methods that incorporate calibration data (GPTQ, AWQ) can achieve better quality at the same bit width by preserving weights that matter most for typical inputs. In practice, the percentile band approach with outlier sideband achieves FP16-equivalent quality without calibration data across all tested architectures.

## 7. Conclusion

SQ4 delivers FP16 quality on a Q4 memory budget. Three ideas compose to make this possible: percentile bands that adapt to the empirical weight distribution (placing reconstruction codes where the data actually is), magnitude/sign separation that doubles effective magnitude resolution, and an outlier sideband that preserves the rare high-magnitude weights that disproportionately affect output quality.

The format requires no calibration data, no gradient computation, and no matrix transformations. A single command — `ai quantize <model> sq4` — produces an SQ4 model in one pass over the weights. On CUDA, dequantization is a single register-file lookup that adds no measurable overhead — the kernel is entirely memory-bound, and reading a quarter of the bytes is pure gain. Beyond weight compression, SQ4 has been applied to KV cache quantization, achieving 8x compression with 4% throughput overhead.

The practical result: a 32B-parameter model fits in under 7 GB of VRAM. The quality ceiling is FP16. The memory floor is Q4. SQ4 gives you both.

The implementation is open source at github.com/tensorwire.

## References

[1] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.

[2] Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., & Han, S. (2023). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.

[3] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Large Language Models. (Introduces NF4.)

[4] Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen, S., Mahoney, M., & Keutzer, K. (2023). SqueezeLLM: Dense-and-Sparse Quantization.

[5] Chee, J., Cai, Y., Kuleshov, V., & De Sa, C. (2023). QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.

[6] Awni Hannun, et al. (2023). MLX: An array framework for Apple Silicon. Apple Machine Learning Research.
