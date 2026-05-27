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

| Backend value | FlashInfer path | Layout used by this example | Notes |
|---------------|-----------------|-----------------------------|-------|
| `single` | `single_prefill_with_kv_cache` | One request, Q/K/V in `NHD` layout after removing the batch dimension | Dense single-request path. Use only when `batch_size == 1`; this is the default single-request backend. |
| `cudnn` | `cudnn_batch_prefill_with_kv_cache` | Flattened Q/K/V plus per-sequence lengths and offsets | Dense batched attention path. It is the default for `batch_size > 1`. |
| `trtllm` | `trtllm_batch_context_with_kv_cache` | Paged KV cache in `HND` layout | Dense by default. This is also the only path used for skip-softmax sparse attention. |

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

| Backend value | Main API |
|---------------|----------|
| `torch` | `torch.nn.Linear` / `torch.nn.functional.linear` |
| `bf16` | `flashinfer.gemm.mm_bf16` |
| `fp8` | `flashinfer.gemm.mm_fp8` |
| `fp8_sm90` | `flashinfer.gemm.fp8_blockscale_gemm_sm90` |
| `bmm_fp8` | `flashinfer.gemm.bmm_fp8` |
| `fp8_groupwise` | `flashinfer.gemm.gemm_fp8_nt_groupwise` |
| `fp8_blockscaled` | `flashinfer.gemm.gemm_fp8_nt_blockscaled` |
| `batch_deepgemm_fp8` | `flashinfer.gemm.batch_deepgemm_fp8_nt_groupwise` |
| `fp4` | `flashinfer.gemm.mm_fp4` |
| `bmm_bf16` | `flashinfer.gemm.bmm_bf16` |
| `mxfp8` | `flashinfer.gemm.mm_mxfp8` |
| `bmm_mxfp8` | `flashinfer.gemm.bmm_mxfp8` |

Control GEMM and activation quantization with:

```bash
FLASHINFER_GEMM_BACKEND=bf16 python wan/transformer_wan_flashinfer.py

FLASHINFER_GEMM_BACKEND=fp8 \
FLASHINFER_ONLINE_ACT_QUANT=0 \
python wan/transformer_wan_flashinfer.py
```

### Environment Variables

| Variable | Values | Meaning |
|----------|--------|---------|
| `FLASHINFER_GEMM_BACKEND` | `torch`, `bf16`, `fp8`, `fp8_sm90`, `bmm_fp8`, `fp8_groupwise`, `fp8_blockscaled`, `batch_deepgemm_fp8`, `fp4`, `bmm_bf16`, `mxfp8`, `bmm_mxfp8` | Selects the `FlashInferLinear` GEMM implementation. Unsupported backends fall back to `torch` with a warning. |
| `FLASHINFER_ATTENTION_BACKEND` | `auto`, `single`, `cudnn`, `trtllm` | Selects the FlashInfer attention path. `auto` uses `single_prefill_with_kv_cache` for `batch_size == 1` and `cudnn_batch_prefill_with_kv_cache` for `batch_size > 1`. |
| `FLASHINFER_ONLINE_ACT_QUANT` | `1/0`, `true/false`, `yes/no`, `on/off` | Controls FP8-family activation scaling: online scale from the current tensor vs fixed default scale. |
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

### End-to-End Pipeline Evaluation

Load a diffusers Wan2.2 T2V pipeline, replace `transformer` and `transformer_2`
with FlashInfer versions, generate frames, and save an mp4:

```bash
python wan/pipeline_wan_flashinfer.py \
  --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --num-inference-steps 3 \
  --output wan_flashinfer.mp4
```

For numeric end-to-end comparison, run latent output for original diffusers and
FlashInfer using the same seed:

```bash
python wan/pipeline_wan_flashinfer.py \
  --model-id Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --compare-original \
  --num-inference-steps 3 \
  --output-type latent \
  --output wan_flashinfer_compare.pt
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

## Notes

- Unsupported GEMM backends fall back to `torch` with a warning.
- Skip-softmax sparse attention requires supported hardware and the TRT-LLM
  attention path.
- `pipeline_wan_flashinfer.py --compare-original` compares latent tensors by
  `max_abs_error`, `mean_abs_error`, and cosine similarity.
