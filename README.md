# ai

GPU-accelerated ML CLI. Train, infer, quantize, and serve LLMs. Zero Python, one binary.

Powered by the [mongoose](https://github.com/tensorwire/mongoose) GPU compute engine.

This is a solo passion project and budgets are tight for renting production hardware and acquiring commodity hardware, if you want to help out my sponsor link is below. 

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-ea4aaa?logo=github-sponsors)](https://github.com/sponsors/swizzley)


## Install

macOS (Metal — Apple Silicon):

```bash
go install github.com/tensorwire/ai@latest
```

Linux (CUDA — NVIDIA GPUs):

```bash
CGO_CFLAGS="-I/usr/local/cuda/include" \
CGO_LDFLAGS="-L/usr/local/cuda/lib64" \
go install github.com/tensorwire/ai@latest
```

Any GPU via Vulkan (WebGPU — AMD, Intel, NVIDIA):

```bash
CGO_ENABLED=0 go install github.com/tensorwire/ai@latest
```

CPU only:

```bash
CGO_ENABLED=0 go install github.com/tensorwire/ai@latest
```

Or build from source:

```bash
git clone https://github.com/tensorwire/ai.git
cd ai
go build -o ai .
```

## Quick Start

```bash
# Train from scratch
ai train data=corpus.txt

# Fine-tune a pretrained model
ai train model=Qwen2.5-0.5B data=corpus.txt

# Download a model
ai pull Qwen/Qwen2.5-0.5B

# Chat
ai chat Qwen2.5-0.5B

# Generate text
ai infer Qwen2.5-0.5B "The meaning of life is"

# Start an OpenAI-compatible API server
ai serve Qwen2.5-0.5B
```

## Commands

Commands use `key=value` for required args. `--flags` are advanced overrides for users who know what they want — everything has safe defaults.

### Training

| Command | Description |
|---------|-------------|
| `ai train data=<file>` | Train from scratch |
| `ai train model=<name> data=<file>` | Fine-tune a pretrained model |
| `ai resume checkpoint=<dir> data=<file>` | Continue training from checkpoint |

Auto-detects GPU: Metal on macOS, CUDA on Linux, Vulkan everywhere else.

Advanced overrides (optional — defaults are opinionated):

```bash
ai train data="/data/*.txt" --dim 512 --layers 8 --steps 5000 --lr 3e-4
```

### Evaluation & Inference

| Command | Description |
|---------|-------------|
| `ai chat <model>` | Interactive chat (bubbletea TUI) |
| `ai infer <model> "prompt"` | Generate text |
| `ai eval model=<name> data=<file>` | Validation pass (loss + perplexity) |
| `ai benchmark <model>` | Inference throughput and latency |

### Optimization

| Command | Description |
|---------|-------------|
| `ai quantize <model> [q8\|q4\|f16]` | Reduce precision |
| `ai prune <model>` | Remove low-magnitude weights (50% default) |
| `ai prune <model> --sparsity 0.7 --structured` | Structured head pruning |
| `ai convert gguf <model>` | Export to GGUF (for Ollama) |
| `ai merge <base> <adapters>` | Merge LoRA into base model |

### Data

| Command | Description |
|---------|-------------|
| `ai dataset inspect <file>` | Dataset statistics and recommendations |
| `ai dataset split <file>` | Partition into train/val/test |
| `ai dataset augment <file>` | Dedup, lowercase, repeat, shuffle |

### Tuning & Search

| Command | Description |
|---------|-------------|
| `ai sweep data=<file> lr=1e-4,3e-4,6e-4` | Hyperparameter search |
| `ai distill teacher=<model> data=<file>` | Knowledge distillation |
| `ai profile` | Per-op GPU timing breakdown |

### Deployment

| Command | Description |
|---------|-------------|
| `ai serve <model>` | OpenAI-compatible API server |
| `ai serve <model> --no-stream` | Serve without streaming weight load (full VRAM, max tok/s from first token) |

### Introspection

| Command | Description |
|---------|-------------|
| `ai explain <model> "prompt"` | Token-level attribution |
| `ai checkpoint ls` | List training checkpoints |
| `ai checkpoint diff <a> <b>` | Compare checkpoints |
| `ai bench` | Raw GPU compute benchmark |
| `ai gpus` | Detect and calibrate hardware |

### Models

| Command | Description |
|---------|-------------|
| `ai pull <org/model>` | Download from HuggingFace |
| `ai models` | List downloaded models |
| `ai info <model>` | Show model architecture |

## Platform Support Matrix

Every command works on every backend. Optimized paths are used when available, with automatic fallback.

| Command | Metal | CUDA | WebGPU (Vulkan) | CPU |
|---------|-------|------|-----------------|-----|
| train | yes | yes | yes | yes |
| finetune | yes | yes | yes (CPU train) | yes (CPU train) |
| chat | yes (fused/graph/generic) | yes (generic) | yes (generic) | yes (generic) |
| infer | yes (fused/graph/tier2) | yes (Q8/tier2) | yes (tier2/3) | yes (tier3) |
| serve | yes (streaming/fused) | yes (Q8 fused) | yes (generic) | yes (generic) |
| eval | yes (planned) | yes | planned | planned |
| profile | planned | yes | planned | planned |
| explain | yes | yes | yes | yes |
| sweep | yes (graph) | yes (exec) | yes (exec) | yes (exec) |

## Performance

### Training convergence — dim=512, RTX 5090

```
step 1     loss 6.17
step 100   loss 2.59   floor 2.37
step 300   loss 2.05   floor 1.76
step 500   loss 1.95   floor 1.29   365 steps/s
```

### Inference — Qwen2.5-0.5B, Q8, `ai serve`

| Metric | M4 Max | M1 Pro |
|--------|--------|--------|
| `ai benchmark` (avg) | 221.7 tok/s | 97.2 tok/s |
| `ai serve` throughput | 239–241 tok/s | 133–145 tok/s |
| TTFT (streaming) | 4ms | 29ms |

### Streaming inference — `ai serve`

Weights stream into GPU memory in the background using ping-pong double buffers. The server accepts requests immediately — no cold start.

| Mode | VRAM | Tok/s | When |
|------|------|-------|------|
| Streaming (cold) | 2 layers (~19x less) | ~9 tok/s | First request, weights still loading |
| Resident (warm) | Full model | 125–135 tok/s | After background load finishes |

Auto-switches from streaming to resident once all weights are loaded. Use `--no-stream` to skip streaming and wait for full load before accepting requests.

```bash
ai serve Qwen2.5-0.5B              # streaming (default) — instant cold start
ai serve Qwen2.5-0.5B --no-stream  # wait for full model load, max speed from first token
```

### CUDA serve — RTX 5090

| Path | Tok/s |
|------|-------|
| Generic (pre-v1.3.1) | 1.9 |
| Q8 fused (v1.3.1+) | 189 |

Zero-alloc hot path: all scratch buffers pre-allocated at model load.

### Architecture

- Automatic quantization: Q8 for models <4B params, Q4 for 7B+
- Metal 4 `matmul2d` TensorOp on macOS 26+
- Fused dequant-matvec kernels — zero intermediate buffers
- Custom Metal/CUDA compute shaders for RMSNorm, RoPE, GQA attention, SiLU
- Multi-slot inference: independent KV caches per request on separate Metal command queues

## GPU Support

| Backend | Hardware | Install |
|---------|----------|---------|
| Metal | Apple Silicon (M1+) | `go install` on macOS |
| CUDA | NVIDIA GPUs | `CGO_CFLAGS/LDFLAGS` + `go install` on Linux |
| Vulkan | Any GPU (AMD, Intel, NVIDIA) | `CGO_ENABLED=0 go install` (WebGPU) |
| CPU | Any | `CGO_ENABLED=0 go install` |

### Compute Kernels

Custom fused kernels unlock the fastest paths for training and inference. Without them, `ai` falls back to generic GPU matmul which still works but is slower.

**Metal (macOS)** — pre-compiled `.metallib` files ship in the mongoose repo. `brew install` places them automatically. For `go install`, copy them next to the binary:

```bash
# Metal 4 GEMM (optimal path — requires macOS >= 26.3)
cp mongoose/kernels/gemm_metal4.metallib $(go env GOPATH)/bin/

# Fused training kernels (RMSNorm, RoPE, attention, SiLU, AdamW)
cp mongoose/kernels/fused_train.metallib $(go env GOPATH)/bin/
```

Or compile from source:

```bash
cd mongoose/kernels
xcrun metal -std=metal4.0 -O2 -c gemm_metal4.metal -o gemm_metal4.air
xcrun metallib -o gemm_metal4.metallib gemm_metal4.air
cp gemm_metal4.metallib $(go env GOPATH)/bin/
```

Metal 4 `matmul2d` TensorOp requires macOS 26.3+ and Apple Silicon. On older macOS, `ai` falls back to MPS tiled GEMM automatically.

**CUDA (Linux)** — compile from source (architecture-specific):

```bash
cd mongoose/kernels
nvcc -shared -o libmongoose_kernels.so mongoose.cu \
  -Xcompiler -fPIC -gencode arch=compute_90,code=compute_90
cp libmongoose_kernels.so $(go env GOPATH)/bin/
```

Replace `compute_90` with your GPU's compute capability (e.g., `compute_89` for RTX 4090, `compute_120` for RTX 5090).

| Kernel | What it fuses |
|--------|--------------|
| RMSNorm | in-place + out-of-place, forward + backward |
| RoPE | rotate_half, forward + backward |
| GQA Attention | grouped query attention decode |
| SiLU Gate Mul | SiLU activation * gate, forward + backward |
| Q8/Q4 Matvec | fused dequant + matrix-vector multiply |
| AdamW | fused optimizer step |
| Embedding | gather + scatter |
| Cross-Entropy | softmax + CE loss + gradient in one pass |

## Dependencies

- [mongoose](https://github.com/tensorwire/mongoose) — GPU compute engine
- [gguf](https://github.com/tensorwire/gguf) — GGUF + SafeTensors I/O
- [tokenizer](https://github.com/tensorwire/tokenizer) — BPE tokenizer

## License

MIT
