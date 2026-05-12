# CLAUDE.md — ai

## What This Is

Command-line interface for GPU-accelerated ML. Train, infer, quantize, serve. Powered by the mongoose engine ecosystem. Zero Python, one binary.

## Build

```bash
go build -o ai .
```

On Linux with CUDA:
```bash
CGO_CFLAGS="-I/usr/local/cuda/include" CGO_LDFLAGS="-L/usr/local/cuda/lib64" go build -o ai .
```

CUDA kernels (optional, enables fused Q8/Q4 inference):
```bash
cd ../mongoose/kernels
nvcc -shared -o libmongoose_kernels.so mongoose.cu -Xcompiler -fPIC -gencode arch=compute_90,code=compute_90
cp libmongoose_kernels.so ../../ai/
```

Metal 4 kernels (macOS, optional — enables Metal 4 matmul2d TensorOp):
```bash
cd ../mongoose/kernels
xcrun metal -std=metal4.0 -O2 -c gemm_metal4.metal -o gemm_metal4.air
xcrun metallib -o gemm_metal4.metallib gemm_metal4.air
cp gemm_metal4.metallib ../../ai/
```

## Usage

```bash
ai pull Qwen/Qwen2.5-0.5B          # download from HuggingFace
ai chat Qwen2.5-0.5B               # interactive chat
ai infer Qwen2.5-0.5B "Hello"      # generate text
ai serve Qwen2.5-0.5B              # OpenAI-compatible API
ai train data=corpus.txt            # train from scratch
ai train model=Qwen2.5-0.5B data=corpus.txt  # fine-tune
ai resume checkpoint=./checkpoints data=corpus.txt
ai quantize Qwen2.5-0.5B q8        # quantize to INT8 GGUF
ai prune Qwen2.5-0.5B              # remove 50% of weights
ai explain Qwen2.5-0.5B "prompt"   # token attribution
ai sweep data=corpus.txt            # hyperparameter search
ai distill teacher=7B data=corpus.txt  # knowledge distillation
ai dataset inspect corpus.txt       # dataset statistics
ai dataset split corpus.txt         # train/val/test split
```

## Architecture

- `main.go` — command dispatch, usage
- `infer_gpu.go` — GPU inference with tiered dispatch:
  - Metal fused compute (custom kernels + Metal 4 matmul2d): Q8 or Q4
  - CUDA fused Q8/Q4 matvec: custom kernels + cuBLAS
  - GPU-resident tier 2: cuBLAS/MPS matmul + CPU attention
  - CPU streaming tier 3: pure Go fallback
- `infer.go` — CPU-only text generation
- `chat.go` — interactive chat TUI (bubbletea) with ChatML/Llama3 template detection, --verbose for raw debug
- `serve.go` — OpenAI-compatible API server (chat, completions, embeddings)
- `train_unified.go` — unified training entry point (dispatches to backend)
- `train_cuda.go` — CUDA training with helix optimizer
- `train_finetune.go` — fine-tuning pretrained models
- `train.go` — GraphTrainEngine training + `resume` from checkpoint
- `commands.go` — pull, models, info, bench, gpus, convert, export
- `quantize.go` — quantize safetensors to GGUF (Q8_0, Q4_0, F16, F32)
- `prune.go` — magnitude pruning (unstructured + structured/head pruning)
- `explain.go` — token attribution via embedding perturbation
- `sweep.go` — hyperparameter search (grid/random over lr, dim, layers)
- `distill.go` — knowledge distillation (teacher→student KL divergence)
- `dataset.go` — dataset inspection, split (train/val/test), augmentation
- `globals.go` — shared config, engine selection (`selectEngine`)
- `autodetect.go` — hardware detection and model profiling
- `checkpoint.go` — checkpoint management (list, diff, meta.json)
- `merge.go` — merge LoRA adapters into base model
- `benchmark.go` — model inference profiling
- `eval.go` — validation (loss + perplexity)
- `profile.go` — per-op GPU timing breakdown

## Inference Tier Selection

The `cmdInferGPU` function selects the fastest available path:

1. **Metal fused compute** (macOS): Custom Metal compute shaders (RMSNorm, RoPE, GQA attention, SiLU) + Metal 4 `matmul2d` TensorOp or fused Q8/Q4 matvec. One command buffer per token. Weights quantized to Q8 (<4B params) or Q4 (>4B params) on load.
2. **CUDA fused Q8/Q4** (Linux): Custom CUDA kernels for RMSNorm, RoPE, decode attention, SiLU, fused Q8/Q4 dequant-matvec. Same >4B param gate for Q4.
3. **GPU-resident tier 2**: Weights in VRAM, matmuls via cuBLAS/MPS, attention on CPU.
4. **CPU streaming tier 3**: Weights loaded from disk per layer, pure Go.

## Dependencies

- `github.com/tensorwire/mongoose` — GPU compute engine
- `github.com/tensorwire/gguf` — model serialization (GGUF + SafeTensors)
- `github.com/tensorwire/tokenizer` — BPE tokenizer
- `github.com/tensorwire/helix` — DNA optimizer (optional, --helix flag)
- `github.com/tensorwire/needle` — INT8 kernels (optional, --needle flag)

## Test

```bash
go test -v ./...
```

## Open Bugs (2026-04-23)

### train_finetune.go (cmdFinetune — CUDA fine-tune path)

1. **Hardcoded TinyLlama dimensions.** `dim=2048, heads=32, kvHeads=4, nLayers=22, vocabSize=32000` — any model that isn't TinyLlama loads with wrong dimensions. Fix: read `config.json` from model directory (same pattern as `train_cuda.go` and `train_metal.go` resume paths).

2. **No GGUF support.** Uses `gguf.OpenSafeTensors()` directly. Should use `OpenModel()` from `model_loader.go` which handles SafeTensors, GGUF, sharded models, and zips.

3. **No checkpoint/adapter saving.** Training loop runs to completion and exits. All trained weights are lost. Needs to save adapters or full model at end + periodic checkpoints.

4. **Embed gradient uses wrong tensor.** Line 419: `adamW(embed, dLmHead, embedAS.m, embedAS.v)` passes `dLmHead` (the lm_head weight gradient) as the gradient for the embedding layer. Should be a scatter gradient from `dHidden` back through the embedding gather.

5. **No VRAM pre-check.** Allocates all layers' INT8 weights + FP32 caches + FP16 mom/vel simultaneously without checking if they fit. Segfaults on GPU MMU fault instead of failing gracefully. `train_cuda.go` and `train_metal.go` both have VRAM guards.

6. **No LoRA path.** Does full-weight INT8+needle training (~9 bytes/param for all layers resident). For 14B models on 32GB VRAM this OOMs at layer 11/48. The Apr 15 successful 14B training used Q8+LoRA (frozen INT8 base + rank-16 adapters = 15.5GB total). `autodetect.go` already computes `LoRAFit` and `LoRARank` but nothing wires to a LoRA training path.

### train_cuda.go (cmdTrainCUDA — from-scratch CUDA path)

7. **Sharded model loading broken on resume.** Line 65: `filepath.Join(ckptDir, "model.safetensors")` fails for models with multiple shards (e.g., Qwen2.5-14B has 8 shards). Fix: `gguf.OpenSafeTensors(ckptDir)` — the library handles directories with `model.safetensors.index.json`.

### train_metal.go (cmdTrainMetal — from-scratch Metal path)

8. **Same sharded model loading bug.** Line 74: same `filepath.Join(ckptDir, "model.safetensors")` hardcode. Same fix.

### train_unified.go (routing)

9. **CUDA fine-tune routes to broken cmdFinetune.** `runFinetune()` on CUDA calls `cmdFinetune()` which has bugs 1-6 above. For large models, should route to a Q8+LoRA path instead. Metal already routes via `--resume` to `cmdTrainMetal` which reads config.json correctly.

### merge.go (cmdMerge — LoRA adapter merge)

10. **OOM on large models (Qwen 2.5 14B on 64GB RAM).** `cmdMerge` reads every base model tensor via `ReadTensorFloat32` and stores all of them in the `mergedTensors` map simultaneously before writing anything. For 14B params that's ~56GB of float32 held in RAM at once, plus adapter tensors and Go runtime overhead. Fix: stream tensors — read one tensor, merge if adapted, write immediately, release before loading the next. Alternatively, process in shards (write one output shard at a time). Reproduced on Beast (64GB system RAM) with Qwen 2.5 14B.

### Training Quality (from-scratch, MPS/Metal path)

11. **68% vs 90% accuracy gap against PyTorch.** Training from scratch with `ai` + mongoose on Metal produces 68% accuracy; PyTorch on same data/architecture/hyperparams produces 90%. Investigation tracked in chain. Top suspects: (a) Helix optimizer uses beta2=0.95 vs PyTorch AdamW's 0.999, and weight_decay=0.1 vs 0.01 — 10× more aggressive regularization may cause underfitting. (b) Backward pass numerical precision — Metal MPS may accumulate gradients in FP16 instead of FP32, zeroing small gradients in early layers. (c) Warmup is only 1 step vs PyTorch's typical 100-2000 — early training instability. Quick diagnostic: replace Helix with standard AdamW using PyTorch-matched hyperparams (beta2=0.999, wd=0.01, warmup=100). If accuracy jumps to ~90%, it's hyperparameters not a kernel bug. Also test CUDA path — if CUDA also gets 68%, the bug is in shared Go code, not Metal-specific.

### Format support (all commands)

12. **GGUF models rejected everywhere.** serve.go, infer_gpu.go, chat.go, quantize.go, merge.go, prune.go, distill.go, eval.go, explain.go, benchmark.go all hardcode `gguf.OpenSafeTensors()` instead of using the unified `OpenModel()` from model_loader.go which already handles GGUF, SafeTensors, shards, and zips. Error messages are opaque ("safetensors: \<error\>") with no guidance to convert. Fix: replace each `gguf.OpenSafeTensors()` call with `OpenModel()`.

13. **`ai convert` is one-way (SafeTensors→GGUF only).** No GGUF→SafeTensors, no GGUF→GGUF requantization. Users with GGUF models can't use any command without a conversion path that doesn't exist.

14. **`ai pull` skips GGUF-only models.** Download filter only grabs `.safetensors` files. Models published as GGUF-only (TheBloke, etc.) fail with "No safetensors files found" and no suggestion.

15. **`ai infer` doesn't apply chat templates.** Sends raw text to instruction-tuned models that expect `<|im_start|>user\n...<|im_end|>`. Produces garbage for any instruct/chat model. `ai chat` and `ai serve` apply templates correctly.

16. **Chat template detection ignores `chat_template` field.** Only detects ChatML and Llama3 by probing tokenizer vocabulary. Mistral (`[INST]`), Gemma (`<start_of_turn>`), Phi, and any model with a custom Jinja2 `chat_template` in `tokenizer_config.json` falls back to broken generic markdown format.

17. **No architecture validation on model load.** Tensor names are hardcoded to Llama-family layout. Phi, Gemma, GPT-NeoX, MPT use different naming and fail silently — nil tensor data, crash later during inference.

### serve.go

18. **No VRAM pre-check.** Non-streaming mode loads entire model into GPU without checking if it fits. Crashes with no helpful message.

19. **No CPU fallback.** Unlike infer_gpu.go which has a CPU streaming tier, serve.go crashes if no GPU available.

20. **Hardcoded concurrency limits.** Inference queue capped at 16, no request timeout, no context cancellation on client disconnect. Hung inference blocks queue forever.

### mongoose Q4 kernel

21. **Q4 matvec produces garbage output.** Tested with Mistral 7B — Q8 works perfectly, Q4 output is control tokens at 1000+ tok/s. Code review: Go-side packing and CUDA-side unpacking match (low nibble = even col, high nibble = odd col, offset-8 symmetric, scale = absMax/7). The 1000+ tok/s suggests results are near-zero/NaN causing immediate stop tokens. Root cause likely: 4-bit symmetric with only 15 levels (-7..+7) per-row is too coarse for 7B models — quantization error compounds across 32 layers. Fix: implement block-quantized Q4 (per-32-element scales like GGUF Q4_0) instead of per-row. Q8 is the reliable path until then.
