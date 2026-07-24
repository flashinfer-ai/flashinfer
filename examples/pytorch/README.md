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

### Fused-MoE APIs

`FlashInferFusedMoE` (used by the HunyuanImage-3 example) selects the fused
Mixture-of-Experts kernel via `FLASHINFER_MOE_BACKEND` / `--moe-backend`,
mirroring the GEMM convention. Routing (softmax → top-k → renormalize) always
stays in the model's own gate; every backend consumes the same precomputed
`topk_ids` / `topk_weights`, so routing decisions are identical across
backends.

| Backend value | Main API | Quantization | SM support |
|---------------|----------|--------------|------------|
| `torch` | eager per-expert loop | none (BF16) | any |
| `cutlass` | `flashinfer.fused_moe.cutlass_fused_moe` | none (BF16) | SM89/SM90/SM100+ |
| `cutlass_fp8` | `cutlass_fused_moe` (`quant_scales`, 4-tensor layout) | per-tensor FP8 W8A8 | SM89/SM90/SM100+ |
| `cutlass_fp8_blockscale` | `cutlass_fused_moe` (`use_deepseek_fp8_block_scale=True`) | DeepSeek-style 128×128 block-scale FP8 W8A8; activation quantized inside the kernel | SM90 only (the only arch with this kernel; CUDA ≥ 12.8) |
| `cutlass_w4a16` | `cutlass_fused_moe` (`use_w4_group_scaling=True`, SM90 interleaved weights) | MXFP4 weight-only, BF16 activation | SM90 only |
| `cutlass_nvfp4` | `cutlass_fused_moe` (6-tensor `quant_scales`) | NVFP4 W4A4 (per-expert global scales + 16-elem E4M3 block scales; BF16 activation quantized inside the kernel) | SM100/SM103/SM110/SM120/SM121 |
| `trtllm` | `flashinfer.fused_moe.trtllm_bf16_routed_moe` (packed `(expert_id << 16) \| bf16(weight)` routing) | none (BF16, shuffled BlockMajorK weights) | SM100/SM103 |
| `trtllm_fp8_blockscale` | `flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe` (packed routing, plain MajorK weights) | DeepSeek-style 128×128 block-scale FP8 W8A8, per-token-group-128 activation scales | SM100/SM103 |
| `trtllm_fp4` | `flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe` (packed routing) | NVFP4 activations × NVFP4 weights | SM100/SM103 |

Unsupported backends on the current device fall back to `torch` with a
warning, same as the GEMM backends.

Per-architecture availability of the FlashInfer-accelerated backends
(everything else falls back to `torch`):

| Arch | Available MoE backends |
|------|------------------------|
| SM75/SM80/SM86 | `torch` only (no cutlass fused-MoE JIT module) |
| SM89 (Ada) | `cutlass`, `cutlass_fp8` |
| SM90 (Hopper) | `cutlass`, `cutlass_fp8`, `cutlass_fp8_blockscale`, `cutlass_w4a16` |
| SM100/SM103 (Blackwell) | `cutlass`, `cutlass_fp8`, `cutlass_nvfp4`, `trtllm`, `trtllm_fp8_blockscale`, `trtllm_fp4` |
| SM110 | `cutlass`, `cutlass_fp8`, `cutlass_nvfp4` |
| SM120/SM121 | `cutlass`, `cutlass_fp8`, `cutlass_nvfp4` |

**Coverage notes — FlashInfer MoE APIs intentionally *not* wrapped here:**

- `trtllm_fp8_per_tensor_scale_moe`: takes routing logits rather than
  precomputed top-k ids (no `*_routed_*` variant), so it cannot preserve
  this wrapper's bit-identical-routing contract; the per-tensor FP8 recipe
  is available on Blackwell through `cutlass_fp8` instead.
- `trtllm_mxint4_block_scale_moe` and the MxFP8 variant of
  `trtllm_fp8_block_scale_moe`: weight formats that assume an
  offline-quantized checkpoint (MXINT4 / MXFP8); see
  `tests/moe/test_trtllm_gen_fused_moe.py` for the canonical recipes.
- `cutlass_fused_moe` W4A8 / `wfp4afp8` ("Humming") modes: SM90 mixed-input
  paths that likewise assume offline-quantized weights with 8-tensor
  `quant_scales` layouts (`tests/moe/test_trtllm_cutlass_fused_moe.py`).
- `cute_dsl_fused_moe_nvfp4` / `CuteDslMoEWrapper` (SM100/SM103) and
  `b12x_fused_moe` / `B12xMoEWrapper` (SM120/SM121, CUDA 13): CuTe-DSL
  alternatives to the NVFP4 paths already wrapped above (`cutlass_nvfp4`,
  `trtllm_fp4`); the `B12x` wrapper additionally has no expert-parallelism
  support.

> ⚠️ The Blackwell-only backends (`cutlass_nvfp4`, `trtllm`,
> `trtllm_fp8_blockscale`, `trtllm_fp4`) mirror the canonical recipes in
> `tests/moe/test_trtllm_cutlass_fused_moe.py` and
> `tests/moe/test_trtllm_gen_routed_fused_moe.py`; their weight-prep,
> packing, and fallback paths are covered by the checks runnable on any
> GPU, but the kernel launches themselves have not been exercised on
> SM100+ hardware from this example yet.

### Environment Variables

| Variable | Values | Meaning |
|----------|--------|---------|
| `FLASHINFER_GEMM_BACKEND` | `<base>` or `<base>-<kernel>` — base ∈ {`torch`, `bf16`, `fp8`, `fp8_sm90`, `bmm_fp8`, `fp8_groupwise`, `fp8_blockscaled`, `batch_deepgemm_fp8`, `fp4`, `bmm_bf16`, `mxfp8`, `bmm_mxfp8`}; kernel suffix forwarded to the chosen API's `backend` kwarg (e.g. `fp4-cutlass`, `bf16-cutlass`, `mxfp8-cute-dsl`). | Selects the `FlashInferLinear` GEMM implementation. Unsupported backends fall back to `torch` with a warning. |
| `FLASHINFER_ATTENTION_BACKEND` | `<base>` or `<base>-<kernel>` — base ∈ {`auto`, `single`, `cudnn`, `trtllm`, `torch`}; `-<kernel>` suffix on `single` is forwarded to `single_prefill_with_kv_cache`'s `backend` kwarg (`single-fa3`, `single-fa2`, `single-cudnn`, …). | Selects the FlashInfer attention path. `auto` uses `single_prefill_with_kv_cache` for `batch_size == 1` and `cudnn_batch_prefill_with_kv_cache` for `batch_size > 1`. |
| `FLASHINFER_MOE_BACKEND` | `torch`, `cutlass`, `cutlass_fp8`, `cutlass_fp8_blockscale`, `cutlass_w4a16`, `cutlass_nvfp4`, `trtllm`, `trtllm_fp8_blockscale`, `trtllm_fp4` (HunyuanImage-3 also accepts `eager` = keep the upstream per-expert loop) | Selects the `FlashInferFusedMoE` kernel — see the Fused-MoE APIs table above. Unsupported backends fall back to `torch` with a warning. |
| `FLASHINFER_MOE_IMPL` | `flashinfer`, `eager` | Deprecated alias of `FLASHINFER_MOE_BACKEND` (`flashinfer` → `cutlass`). |
| `FLASHINFER_ONLINE_ACT_QUANT` | `1/0`, `true/false`, `yes/no`, `on/off` | Controls FP8/FP4-family activation scaling: online scale from the current tensor vs fixed default scale. GEMM backends that ignore this flag: `torch`, `bf16`, `bmm_bf16`, `mxfp8`, `bmm_mxfp8`. MoE backends that consult it: `cutlass_fp8` and `trtllm_fp4` (`cutlass_fp8_blockscale` / `cutlass_nvfp4` quantize activations inside the kernel, `trtllm_fp8_blockscale` always computes per-token-group scales online, `cutlass_w4a16` doesn't quantize activations). |
| `FLASHINFER_USE_SKIP_SOFTMAX_SPARSE` | `1/0`, `true/false`, `yes/no`, `on/off` | Enables skip-softmax sparse attention when the GPU supports it; this forces the TRT-LLM attention path. |
| `FLASHINFER_SKIP_SOFTMAX_THRESHOLD` | Float, for example `1.0` | Threshold scale passed to the TRT-LLM sparse attention path. |

## WAN Example

The WAN example lives in `wan/` and provides:

| File | Description |
|------|-------------|
| `wan/transformer_wan_flashinfer.py` | FlashInfer implementation of `WanTransformer3DModel` |
| `wan/pipeline_wan_flashinfer.py` | Diffusers `WanPipeline` loader that swaps in FlashInfer transformer(s) |

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
  Fuses the per-layer activation-quant / bias-add / dtype-cast prologue
  that dominates the wrapper overhead. On B300 + NVFP4 it's the fastest
  configuration measured at 720p × 5s.
  Optional sub-flags: `--torch-compile-mode {default,reduce-overhead,
  max-autotune,max-autotune-no-cudagraphs}`, `--torch-compile-fullgraph`,
  `--torch-compile-dynamic`.

### End-to-End Pipeline Evaluation

Load a diffusers Wan2.2 T2V pipeline, replace `transformer` and `transformer_2`
with FlashInfer versions, generate frames, and save an mp4:

```bash
python wan/pipeline_wan_flashinfer.py \
  --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --num-inference-steps 3 \
  --output wan_flashinfer.mp4
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

## HunyuanImage-3.0 Example

The `hunyuan_image3/` directory hosts an end-to-end driver for
[`tencent/HunyuanImage-3.0-Instruct`](https://huggingface.co/tencent/HunyuanImage-3.0-Instruct),
inspired by [vllm-omni's `end2end.py`](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/hunyuan_image3):

| File | Description |
|------|-------------|
| `hunyuan_image3/modeling_hunyuan_image3_flashinfer.py` | `replace_backbone_with_flashinfer(model, ...)` swaps RMSNorm, QK-norm, GQA attention, SwiGLU MLP, and MoE in every decoder layer of a HuggingFace-loaded model. |
| `hunyuan_image3/pipeline_hunyuan_image3_flashinfer.py` | CLI driver for `text2img`, `img2img`, `img2text`, and `text2text` modalities. |

Quick example:

```bash
python hunyuan_image3/pipeline_hunyuan_image3_flashinfer.py \
  --modality text2img \
  --prompts "A cute cat sitting on a windowsill" \
  --gemm-backend bf16 \
  --output ./out
```

See [`hunyuan_image3/README.md`](hunyuan_image3/README.md) for backend flags
and per-modality details.

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
