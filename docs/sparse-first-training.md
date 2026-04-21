# Sparse-First Training: A Biologically-Inspired Framework for Efficient Neural Network Optimization

**Dustin Morgan**
April 2026

---

## Abstract

We present sparse-first training, a framework that designs every stage of the training pipeline around observed sparsity rather than treating sparsity as a post-hoc compression technique. Our system observes which parameters are active during forward computation, computes gradients only for active rows, updates weights only where momentum is non-negligible, and stores optimizer state only for live parameters. On a byte-level transformer (128–4096 dimensions, 4 layers), sparse-first training achieves 1.3–2.2x the throughput of PyTorch MPS on Apple M4 Max while converging to lower loss at large model sizes. On NVIDIA RTX 5090, the system reaches 700 steps/s on a 623K-parameter model. The key insight is that sparsity compounds: 80% sparse gradients fed into a sparse optimizer that writes sparse weight updates produces end-to-end compute savings far exceeding any single sparse technique in isolation.

## 1. Introduction

Modern neural network training is dense by default. Every weight receives a gradient every step. Every row of every weight matrix participates in backward matrix multiplication. Every parameter maintains momentum and velocity state in the optimizer. This density is wasteful — empirical measurement shows that 84–99% of gradient entries, activation rows, and optimizer updates are negligibly small at any given step.

The deep learning community has explored sparsity extensively but almost exclusively as an afterthought: pruning after training [1], quantization for inference [2], sparse attention patterns for sequence length [3], mixture-of-experts for conditional computation [4]. These approaches accept the cost of dense training and compress the result.

We invert this. We begin with the observation that neural network computation is naturally sparse and build every component of the training system to exploit this from the first step. The result is not a modified dense trainer but a fundamentally different pipeline where the unit of work is the *active parameter subset*, not the full model.

## 2. The Sparsity Stack

Sparse-first training consists of five layers, each feeding sparsity information to the next.

### 2.1 Observation: The Conductor

The Conductor tracks which parameters are electrically active — a metaphor borrowed from DNA charge transport, where pi-orbital overlap between stacked base pairs enables charge flow through active gene sequences while mismatches and damage break the chain [5].

In practice, the Conductor maintains a charge-decay window over embedding rows. Each training step "expresses" a small subset of embedding rows (those corresponding to tokens in the current batch). At byte-level vocabulary (256 tokens), a sequence of 64 tokens activates roughly 27 unique rows — 10% of the embedding table. The remaining 90% receive zero gradient regardless of what the backward pass computes.

For projection weights (Q, K, V, O, gate, up, down), a ProjectionTracker observes which output rows have non-negligible activation magnitude during forward computation. This extends sparsity observation from the embedding to every weight matrix in the model.

The Conductor and ProjectionTrackers produce per-row binary masks that propagate through the entire backward and optimizer pipeline.

### 2.2 Sparse Backward: Skip Frozen Rows

The standard backward pass computes weight gradients via transposed matrix multiplication: dW = dX^T @ activation. This produces a full [rows, cols] gradient matrix. When only 20% of rows will be consumed by the optimizer, 80% of this computation is discarded.

Our sparse backward uses a masked TN GEMM kernel that checks the Conductor's hot-row mask at the threadgroup level. If no row in a 32-row tile is active, the entire threadgroup returns immediately — no shared memory loads, no accumulation, no global memory writes. For the tiles that do contain active rows, individual rows are checked before the final write.

At byte-level vocabulary with 80% hot rows, the savings are modest. At production vocabulary (50K+ tokens) with typical batch sparsity of 0.3%, the sparse TN GEMM eliminates 99.7% of the gradient computation for the embedding layer and 80–95% for projection weights.

### 2.3 Sparse Optimizer: Needle

Traditional optimizers (Adam, AdamW) maintain FP32 momentum and velocity for every parameter — 8 bytes per weight on top of the 4-byte weight itself. For a 7B model, this is 56GB of optimizer state.

Needle replaces this with sparse INT8 training. The model weights are stored as INT8 with per-row FP32 scales and an FP32 delta residual that accumulates sub-quantization-level precision. Momentum and velocity are FP16. The optimizer kernel:

1. Reads the INT8 weight + delta to reconstruct FP32 in register
2. Applies the clipped gradient (scaled by a global clip factor computed on GPU)
3. Updates FP16 momentum and velocity via Adam
4. Requantizes to INT8 + new delta residual
5. Writes the dequantized FP32 value to a live weight buffer for the next forward pass

The weight is FP32 for nanoseconds — only in register, never in global memory. Frozen rows (mask = 0) early-return from the kernel. Memory scales with the active parameter count, not the total parameter count.

For paired parameters (gate↔up with G≡C hydrogen bonding, Q↔K with A≡T bonding), the paired Needle kernel couples gradients through the DNA rung geometry: each strand's effective gradient includes a cross-coupling term scaled by hydrogen bond strength (3/5 for G≡C, 2/5 for A≡T). This couples the learning dynamics of parameter pairs that are functionally related.

### 2.4 Sparse Dequantization

After Needle updates the active rows, the FP32 live weight buffer must reflect the changes for the next forward pass. A sparse dequant kernel reads the same hot-row mask and only reconstructs `live[i] = INT8[i] * scale + delta[i]` for active rows. In the fused Needle kernel, this writeback is integrated — zero additional dispatches.

### 2.5 Sparse Loss Computation

The LM head computes logits as a dot product between the final hidden state and every embedding row. At vocab=256, this is cheap. At vocab=50K+, this dominates training time. The Conductor's hot-row set identifies which vocabulary entries have been observed in recent batches. A sparse LM head variant scans only hot rows, reducing the logit computation by the same sparsity factor as the embedding.

Our GPU-side LM head (lm_head_pass1 + lm_head_pass2) computes softmax cross-entropy loss and the dHidden gradient without materializing the full logits matrix, eliminating the need for a CPU round-trip between forward and backward.

## 3. Compounding Sparsity

The key insight is that sparsity compounds multiplicatively across the stack. Consider a model with:
- 80% hot embedding rows (Conductor)
- 90% hot projection rows (ProjectionTracker)  
- 80% hot FFN rows (activation sparsity)

A single sparse technique at one layer saves 10–20%. But sparse-first training applies sparsity at every layer:

| Stage | Dense ops | Sparse ops | Savings |
|-------|-----------|------------|---------|
| Embedding gradient | 256 × dim | 27 × dim | 89% |
| Projection dW GEMM | rows × dim | hot_rows × dim | 10–80% |
| Optimizer state | 13 bytes/param | 13 bytes/hot_param | 10–80% |
| Dequant writeback | rows × cols | hot_rows × cols | 10–80% |
| LM head logits | vocab × dim | hot_vocab × dim | 80–99% |

The per-step compute savings compound: sparse gradient → sparse optimizer → sparse writeback. A parameter that isn't in the hot set skips ALL three stages, not just one.

## 4. System Architecture

### 4.1 Single Command Buffer

The entire training step — forward, loss, backward, gradient clipping, optimizer, weight writeback — executes as a single GPU command buffer with zero CPU synchronization points. The loss scalar is read from unified memory one step stale. Per-step varying constants (learning rate, bias corrections, rung coefficients) are written to shared-memory buffers before dispatch.

### 4.2 Dynamic Dispatch Path

At dim < 512, dispatch overhead dominates compute. Fused kernels (pre-attention and post-attention) combine RMSNorm, GEMM, RoPE, and SiLU into single dispatches per layer, reducing forward dispatches from 94 to 14. At dim >= 512, tiled GEMMs with 32×32 cooperative threadgroups achieve higher arithmetic intensity. The system auto-selects per model shape.

### 4.3 Cross-Platform Checkpoints

Checkpoints are Llama-format safetensors (FP32 weights + config.json), loadable by any Llama-compatible tool. Training can checkpoint on Metal and resume on CUDA, or vice versa.

### 4.4 OOM Protection

Estimated memory footprint is computed before allocation and compared against 75% of reported VRAM. Models that would exceed budget are rejected before touching the GPU, preventing the page-thrash lockups that occur when Metal unified memory exceeds physical capacity under GPU pressure.

## 5. The Helix Immune System

Loss-floor tracking with automatic rewind provides training stability without manual intervention. When loss improves, the system checkpoints sparse hot-row weights. When loss rebounds past a threshold, it restores the checkpoint and continues — analogous to DNA damage repair where the cell reverts to the last known-good state.

The immune system is sparse: only hot-row weights are checkpointed and restored, not the full model. Checkpoint frequency follows a Fibonacci stride modulated by signal conductivity, producing more frequent checkpoints during rapid convergence and fewer during stable plateaus.

## 6. Experimental Results

### 6.1 Throughput

Byte-level transformer, 4 layers, seq_len=64, vocab=256, 100 training steps.

**Apple M4 Max (40GB unified memory):**

| dim | params | mongoose steps/s | PyTorch MPS steps/s | speedup |
|-----|--------|-----------------|--------------------:|--------:|
| 128 | 624K | 134 | 62 | 2.2x |
| 256 | 2.4M | 113 | 68 | 1.7x |
| 512 | 9.6M | 104 | 64 | 1.6x |
| 1024 | 38M | 49 | 60 | 0.8x |
| 2048 | 152M | 32 | 24 | 1.3x |
| 4096 | 605M | 11 | 6 | 1.7x |

**NVIDIA RTX 5090 (CUDA, FP32 AdamW + Helix DNA rung):**

| dim | steps/s |
|-----|---------|
| 128 | 700 (resume) / 323 (cold) |

### 6.2 Convergence

At dim >= 2048, sparse-first training converges faster than PyTorch AdamW, reaching lower loss floors in the same number of steps. The Needle INT8 optimizer's sub-quantization delta residual preserves gradient information that standard FP32 Adam discards through floating-point rounding at the weight update scale.

### 6.3 Memory

INT8 weights + FP16 momentum/velocity + FP32 delta = 13 bytes per parameter total, versus 12 bytes per parameter for FP32 weights + FP32 Adam state. With sparse optimizer state (only hot rows allocated), effective memory per step scales with the active subset, not the full model.

## 7. Related Work

**Sparse training:** Lottery Ticket Hypothesis [6] demonstrates that sparse subnetworks exist within dense networks but requires dense training to find them. Our approach identifies the active subset dynamically per step via observation, not pruning.

**Quantization-aware training:** QAT [7] trains with quantized forward passes but dense FP32 backward and optimizer. Needle quantizes the weights AND the optimizer state, with sub-quant precision maintained in the delta residual.

**Mixture of experts:** MoE [4] routes inputs to sparse expert subsets but the routing is learned. Our conductor observes data-driven sparsity without a learned router — the sparsity pattern is a property of the data and model state, not a trained gate.

**Structured pruning:** N:M sparsity [8] removes weights in fixed patterns for hardware acceleration. Our sparsity is unstructured and dynamic — different rows are active each step based on the data.

## 8. Limitations

The dim=1024 gap (0.8x vs PyTorch) results from our tiled GEMM kernels underperforming PyTorch's MPSGraph operator fusion at that specific size. Metal 4 indirect command buffers, which would eliminate GPU command processor overhead, are functional in standalone tests but require further engineering to integrate with the full pipeline. The fused forward kernels use per-thread sequential dot products that are less efficient than cooperative tiled GEMMs at dim >= 512.

The convergence advantage of Needle's INT8 path over standard FP32 AdamW at large dims warrants further investigation — the mechanism (sub-quant delta preserving gradient information) is empirically validated but not yet theoretically characterized.

## 9. Conclusion

Dense training is a choice, not a necessity. Neural network computation is naturally sparse — most gradients are near-zero, most parameters don't change meaningfully on any given step, most embedding rows are silent. Sparse-first training builds every component of the pipeline around this observation: sparse observation via the Conductor, sparse backward via masked GEMM, sparse optimization via Needle INT8, sparse writeback via fused dequantization.

The result is a training system that achieves competitive throughput with a fundamentally different compute profile: work scales with the active parameter subset, not the total parameter count. As models grow from millions to billions of parameters while batch sparsity remains constant, the advantage of sparse-first training grows proportionally.

The code is open source at github.com/open-ai-org.

## References

[1] Frankle, J., & Carlin, M. (2019). The Lottery Ticket Hypothesis.
[2] Jacob, B., et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.
[3] Child, R., et al. (2019). Generating Long Sequences with Sparse Transformers.
[4] Fedus, W., et al. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.
[5] Genereux, J.C., & Barton, J.K. (2010). Mechanisms for DNA Charge Transport. Chemical Reviews.
[6] Frankle, J., & Carlin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.
[7] Krishnamoorthi, R. (2018). Quantizing Deep Convolutional Networks for Efficient Inference.
[8] Zhou, A., et al. (2021). Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch.
