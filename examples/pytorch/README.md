# FlashInfer PyTorch Examples

This directory contains reusable FlashInfer PyTorch building blocks and model
examples that use them.

## Shared Modules

`flashinfer_modules.py` contains model-independent components that can be reused
by different examples:

| Component | Purpose |
|-----------|---------|
| `FlashInferAttentionDispatcher` | Model-independent FlashInfer attention backend dispatch, workspace/cache management, optional skip-softmax sparse routing |
| `FlashInferLinear` | Linear layer wrapper for Torch, BF16, FP8, FP4, MXFP8, and related FlashInfer GEMM backends |
| `FlashInferRMSNorm` | RMSNorm using `flashinfer.rmsnorm` |
| `FlashInferFP32LayerNorm` | FP32 LayerNorm helper |
| `FlashInferFeedForward` | FFN helper using FlashInfer-capable linear layers |

## FlashInfer API and Backend Selection

Backend selection is exposed through model config fields, command-line options,
and environment variables. The environment variables are read when the model is
constructed, so set them before starting Python.

### Attention Backend Selection

`FlashInferAttentionDispatcher` chooses the attention path from the model config.
For environment-variable control, use `FLASHINFER_ATTENTION_BACKEND` to set both
the single-request and batched backend defaults.

Without environment overrides, WAN uses `single` for `batch_size == 1` and
`cudnn` for `batch_size > 1`.

The backend string chooses the concrete FlashInfer code path:

| Backend value | FlashInfer path | Layout used by this example | SM support | Notes |
|---------------|-----------------|-----------------------------|------------|-------|
| `single` | `single_prefill_with_kv_cache` | One request, Q/K/V in `NHD` layout after removing the batch dimension | SM80+ | Dense single-request path. Use only when `batch_size == 1`; this is the default single-request backend. |
| `cudnn` | `cudnn_batch_prefill_with_kv_cache` | Flattened Q/K/V plus per-sequence lengths and offsets (`batch_offsets_*` are length `batch_size + 1`) | SM80+ | Dense batched attention path. Default for `batch_size > 1`. |
| `trtllm` | `trtllm_batch_context_with_kv_cache` | Paged KV cache in `HND` layout (page size 64 in this example) | **SM100/SM103 only** (Blackwell) | Dense by default. Also the only path used for skip-softmax sparse attention. On non-Blackwell GPUs the dispatcher warns and falls back to `single`/`cudnn`. |
| `torch` | `torch.nn.functional.scaled_dot_product_attention` | `(B, H, S, D)` (transposed internally from the FlashInfer `(B, S, H, D)`) | any | Universal fallback used when no FlashInfer attention kernel fits the current GPU; also the most fusable target for `torch.compile`. |

The `single` path can also take a `-<kernel>` suffix that forwards to
the underlying `single_prefill_with_kv_cache` kernel's own `backend`
kwarg — it itself dispatches over FA2 / FA3 / cuDNN / cutlass /
trtllm-gen, etc. Examples: `single-fa3`, `single-fa2`, `single-cudnn`.
The suffix is silently ignored on any base other than `single`.

Control attention with environment variables:

```bash
# Set both single-request and batched attention backend defaults.
FLASHINFER_ATTENTION_BACKEND=cudnn python wan/transformer_wan_flashinfer.py
```

Sparse attention is a separate switch. When
`FLASHINFER_USE_SKIP_SOFTMAX_SPARSE=1`, `FlashInferAttentionDispatcher` checks
the current GPU. On supported GPUs it forces the TRT-LLM path and passes
`skip_softmax_threshold_scale_factor` to
`trtllm_batch_context_with_kv_cache`. On unsupported GPUs it warns and falls
back to the selected dense backend.

```bash
FLASHINFER_ATTENTION_BACKEND=trtllm \
FLASHINFER_USE_SKIP_SOFTMAX_SPARSE=1 \
FLASHINFER_SKIP_SOFTMAX_THRESHOLD=1.0 \
python wan/transformer_wan_flashinfer.py
```

### GEMM APIs

| Backend value | Main API | Kernel suffixes (`<base>-<kernel>`) | SM support |
|---------------|----------|------------------------------------|------------|
| `torch` | `torch.nn.Linear` / `torch.nn.functional.linear` | — | any |
| `bf16` | `flashinfer.gemm.mm_bf16` | `cudnn` (default), `cutlass`, `tgv`, `auto` | SM100+ (Blackwell) |
| `fp8` | `flashinfer.gemm.mm_fp8` | — (TRT-LLM low-latency) | SM89/SM100+ (use `fp8_sm90` on SM90) |
| `fp8_sm90` | `flashinfer.gemm.fp8_blockscale_gemm_sm90` (W8A8 — input also FP8) | — | SM90 (Hopper) only |
| `bmm_fp8` | `flashinfer.gemm.bmm_fp8` | `cublas` (default), `cudnn`, `cutlass`, `auto` | SM89+ |
| `fp8_groupwise` | `flashinfer.gemm.gemm_fp8_nt_groupwise` | `cutlass` (default), `trtllm` | SM100+ (Blackwell) |
| `fp8_blockscaled` | `flashinfer.gemm.gemm_fp8_nt_blockscaled` | — | SM100+ (Blackwell) |
| `batch_deepgemm_fp8` | `flashinfer.gemm.batch_deepgemm_fp8_nt_groupwise` | — | SM100/SM103 |
| `fp4` | `flashinfer.gemm.mm_fp4` (with `nvfp4_quantize`) | `auto` (default), `cudnn`, `trtllm`, `cutlass`, `cute-dsl` | SM100+ (Blackwell) |
| `bmm_bf16` | `flashinfer.gemm.bmm_bf16` | `cudnn` (default), `cutlass`, `auto` | SM100+ (Blackwell) |
| `mxfp8` | `flashinfer.gemm.mm_mxfp8` | `auto` (default), `cutlass`, `cute-dsl`, `trtllm` | SM100+ (Blackwell) |
| `bmm_mxfp8` | `flashinfer.gemm.bmm_mxfp8` | `auto` (default), `cudnn`, `cutlass` | SM100+ (Blackwell) |

**`<base>-<kernel>` suffix syntax.** Each FlashInfer GEMM API that
accepts a `backend` keyword can be steered from this example by
appending `-<kernel>` to the backend name. Examples:

```bash
FLASHINFER_GEMM_BACKEND=fp4-cutlass            # mm_fp4(..., backend="cutlass")
FLASHINFER_GEMM_BACKEND=fp4-cudnn              # mm_fp4(..., backend="cudnn")
FLASHINFER_GEMM_BACKEND=bf16-cutlass           # mm_bf16(..., backend="cutlass") — bypasses cudnn
FLASHINFER_GEMM_BACKEND=bmm_bf16-cutlass       # bmm_bf16(..., backend="cutlass")
FLASHINFER_GEMM_BACKEND=bmm_fp8-cublas         # bmm_fp8(..., backend="cublas")
FLASHINFER_GEMM_BACKEND=mxfp8-cute-dsl         # mm_mxfp8(..., backend="cute-dsl")
FLASHINFER_GEMM_BACKEND=fp8_groupwise-trtllm   # gemm_fp8_nt_groupwise(..., backend="trtllm")
```

The suffix is silently ignored on backends that don't carry a `backend`
kwarg (`torch`, `fp8`, `fp8_sm90`, `fp8_blockscaled`, `batch_deepgemm_fp8`).
Unsupported backends on the current device fall back to `torch` with a
warning that names the required SM range. The `mxfp8` / `bmm_mxfp8`
paths additionally require an importable `cutlass` Python module from
`nvidia-cutlass-dsl` — the 4.5.2 release of that package is broken and
excluded in `requirements.txt`.

> ⚠️ `bf16` and `bmm_bf16` default to `cudnn`, which currently raises
> `'torch.Stream' object has no attribute 'cuda_stream'` under
> `torch.compile`. If you need either backend under compile, use
> `bf16-cutlass` / `bmm_bf16-cutlass` to bypass the cudnn path.

Control GEMM and activation quantization with:

```bash
FLASHINFER_GEMM_BACKEND=bf16-cutlass python wan/transformer_wan_flashinfer.py

FLASHINFER_GEMM_BACKEND=fp8 \
FLASHINFER_ONLINE_ACT_QUANT=0 \
python wan/transformer_wan_flashinfer.py
```

### Environment Variables

| Variable | Values | Meaning |
|----------|--------|---------|
| `FLASHINFER_GEMM_BACKEND` | `<base>` or `<base>-<kernel>` — base ∈ {`torch`, `bf16`, `fp8`, `fp8_sm90`, `bmm_fp8`, `fp8_groupwise`, `fp8_blockscaled`, `batch_deepgemm_fp8`, `fp4`, `bmm_bf16`, `mxfp8`, `bmm_mxfp8`}; kernel suffix forwarded to the chosen API's `backend` kwarg (e.g. `fp4-cutlass`, `bf16-cutlass`, `mxfp8-cute-dsl`). | Selects the `FlashInferLinear` GEMM implementation. Unsupported backends fall back to `torch` with a warning. |
| `FLASHINFER_ATTENTION_BACKEND` | `<base>` or `<base>-<kernel>` — base ∈ {`auto`, `single`, `cudnn`, `trtllm`, `torch`}; `-<kernel>` suffix on `single` is forwarded to `single_prefill_with_kv_cache`'s `backend` kwarg (`single-fa3`, `single-fa2`, `single-cudnn`, …). | Selects the FlashInfer attention path. `auto` uses `single_prefill_with_kv_cache` for `batch_size == 1` and `cudnn_batch_prefill_with_kv_cache` for `batch_size > 1`. |
| `FLASHINFER_ONLINE_ACT_QUANT` | `1/0`, `true/false`, `yes/no`, `on/off` | Controls FP8/FP4-family activation scaling: online scale from the current tensor vs fixed default scale. Backends that ignore this flag: `torch`, `bf16`, `bmm_bf16`, `mxfp8`, `bmm_mxfp8`. |
| `FLASHINFER_USE_SKIP_SOFTMAX_SPARSE` | `1/0`, `true/false`, `yes/no`, `on/off` | Enables skip-softmax sparse attention when the GPU supports it; this forces the TRT-LLM attention path. |
| `FLASHINFER_SKIP_SOFTMAX_THRESHOLD` | Float, for example `1.0` | Threshold scale passed to the TRT-LLM sparse attention path. |
| `FLASHINFER_USE_VSA` | `1/0`, `true/false`, `yes/no`, `on/off` | Enables Video Sparse Attention for self-attention (WAN example only). |
| `FLASHINFER_VSA_SPARSITY` | Float in `[0, 1)`, for example `0.9` | Fraction of KV blocks VSA drops per query block. |

## WAN Example

The WAN example lives in `wan/` and provides:

| File | Description |
|------|-------------|
| `wan/transformer_wan_flashinfer.py` | FlashInfer implementation of `WanTransformer3DModel` |
| `wan/pipeline_wan_flashinfer.py` | Diffusers `WanPipeline` loader that swaps in FlashInfer transformer(s), plus the Ulysses multi-GPU launcher |
| `wan/vsa_attention.py` | Video Sparse Attention (VSA) on FlashInfer's `bsa_attn_blk64_fwd` |
| `wan/vsa_sanity.py` | Correctness checks for the VSA path |

### Load a Transformer Checkpoint

```bash
python wan/transformer_wan_flashinfer.py \
  --model-id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --subfolder transformer \
  --gemm-backend torch
```

Command-line flags override environment defaults:

```bash
FLASHINFER_GEMM_BACKEND=bf16 \
FLASHINFER_ATTENTION_BACKEND=cudnn \
python wan/transformer_wan_flashinfer.py
```

For end-to-end forward latency, the example supports two acceleration
modes (mutually exclusive):

- `--cuda-graph` — capture and replay the forward pass with a CUDA
  graph. Eliminates per-launch overhead; only helps when the GPU has
  idle time between kernels.
- `--torch-compile` — wrap each transformer block with `torch.compile`.
  Fuses the per-layer activation-quant / bias-add / dtype-cast
  prologue identified as the bottleneck in `wan/BENCHMARK.md`. On
  B300 + NVFP4 it's the fastest configuration measured at 720p × 5s.
  Optional sub-flags: `--torch-compile-mode {default,reduce-overhead,
  max-autotune,max-autotune-no-cudagraphs}`, `--torch-compile-fullgraph`,
  `--torch-compile-dynamic`.

### Benchmark Results

See [`wan/BENCHMARK.md`](wan/BENCHMARK.md) for an end-to-end forward-latency
comparison of every GEMM backend on H100 PCIe, B200, and B300 for both
Wan2.1-T2V-1.3B and Wan2.2-T2V-A14B, including the `--torch-compile`
section, CUDA-graph attempt, and online-vs-offline activation-quant
breakdown.

### End-to-End Pipeline Evaluation

Load a diffusers Wan2.2 T2V pipeline, replace `transformer` and `transformer_2`
with FlashInfer versions, generate frames, and save an mp4:

```bash
python wan/pipeline_wan_flashinfer.py \
  --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --num-inference-steps 3 \
  --output wan_flashinfer.mp4
```

### Video Sparse Attention (VSA)

`--use-vsa` swaps dense self-attention for the two-stage VSA of
[arXiv:2505.13389](https://arxiv.org/abs/2505.13389), running the fine stage on
FlashInfer's `bsa_attn_blk64_fwd` kernel. The token grid is permuted so each
`(4, 4, 4)` spatio-temporal cube becomes one 64-token block, a coarse stage
attends over one pooled token per cube, and the top
`ceil((1 - sparsity) * num_blocks)` blocks per query block go through the sparse
kernel. `--vsa-sparsity` (default `0.9`) sets the fraction of blocks dropped.

Requirements: SM100, bf16, `head_dim == 128`, and a **VSA-finetuned checkpoint** —
the gated combine reads a per-block `to_gate_compress` projection that only such
checkpoints carry. Running `--use-vsa` on a stock Wan checkpoint loads a randomly
initialized gate and degrades quality.

```bash
python wan/vsa_sanity.py   # layout, kernel-vs-reference, head independence

python wan/pipeline_wan_flashinfer.py \
  --model-id FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 5.0 --flow-shift 5.0 \
  --gemm-backend fp4 --use-vsa --vsa-sparsity 0.9 \
  --output wan_vsa.mp4
```

### Multi-GPU (Ulysses context parallelism)

Launching the same script under `torchrun` shards the visual token sequence
across ranks and uses `flashinfer.comm.UlyssesCommunicator` for the
pre/post-attention all-to-all. `--ulysses-backend` picks the transport:
`nvlink` is FlashInfer's NVLink-P2P kernel, `nccl` is
`dist.all_to_all_single`, `auto` (default) takes NVLink when the topology
supports it. The two are numerically identical.

The sequence length must divide by the world size, as must the head count.
`--check-rank-sync` asserts every rank holds identical latents at each denoising
step, which is what lets the ranks run the scheduler independently.

```bash
torchrun --nproc_per_node=8 wan/pipeline_wan_flashinfer.py \
  --model-id FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers \
  --height 720 --width 1280 --num-frames 81 \
  --num-inference-steps 50 --guidance-scale 5.0 --flow-shift 5.0 \
  --gemm-backend fp4 --use-vsa --ulysses-backend nvlink \
  --warmup-steps 2 --timing-json timing.json --output wan_vsa_8gpu.mp4
```

### WAN Config Fields

`WanTransformer3DConfig` exposes the same backend controls as constructor fields:

| Field | Default | Description |
|-------|---------|-------------|
| `gemm_backend` | `torch` | GEMM backend for `FlashInferLinear` |
| `online_act_quant` | `True` | Online vs fixed activation quantization scale |
| `attention_backend` | `auto` | Attention backend. `auto` uses `single` for `batch_size == 1` and `cudnn` for `batch_size > 1`; explicit values are `single`, `cudnn`, and `trtllm` |
| `use_skip_softmax_sparse` | `False` | Enables sparse attention through TRT-LLM when supported |
| `skip_softmax_threshold_scale_factor` | `1.0` | Sparse attention threshold scale |
| `use_vsa` | `False` | Replaces dense self-attention with Video Sparse Attention (SM100, bf16, `head_dim == 128`, VSA-finetuned checkpoint) |
| `vsa_sparsity` | `0.9` | Fraction of KV blocks VSA drops per query block |

## Notes

- Unsupported GEMM backends fall back to `torch` with a warning that lists the
  required SM range.
- Skip-softmax sparse attention requires Blackwell (SM100/SM103) and the
  TRT-LLM attention path.
- The `trtllm` attention backend is Blackwell-only. On other architectures the
  dispatcher warns and falls back to `single`/`cudnn` automatically.
- `from_pretrained` remaps diffusers' `ffn.net.0.proj.*` / `ffn.net.2.*` keys
  to `FlashInferFeedForward`'s `proj_up.*` / `proj_down.*`; without the remap
  the FFN weights would be silently dropped.
