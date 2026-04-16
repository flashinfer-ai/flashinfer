# FlashInfer-Optimized Wan Transformer

This directory contains a FlashInfer-optimized implementation of the [Wan Video Transformer](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py) for video generation.

## Overview

The `FlashInferWanTransformer3DModel` replaces standard PyTorch operations with FlashInfer kernels for improved inference performance. This implementation supports multiple GPU architectures and automatically selects the best available backend.

## Integrated FlashInfer APIs

### 1. Attention APIs

| API | Description | Supported SM | Notes |
|-----|-------------|--------------|-------|
| `single_prefill_with_kv_cache` | Single-request attention | SM75+ | Always used for batch_size=1 |
| `cudnn_batch_prefill_with_kv_cache` | Batched attention via cuDNN | SM80+ | Default for batch_size>1 |
| `trtllm_batch_context_with_kv_cache` | Batched attention via TRT-LLM | **SM100, SM103 only** | Supports skip-softmax sparse attention |

**Attention Backend Selection:**
- **batch_size == 1** (`single_attention_backend`):
  - `"single"` (default): Uses `single_prefill_with_kv_cache`
  - `"cudnn"`: Uses `cudnn_batch_prefill_with_kv_cache`
  - `"trtllm"`: Uses `trtllm_batch_context_with_kv_cache`
- **batch_size > 1** (`batch_attention_backend`):
  - `"cudnn"` (default): Uses `cudnn_batch_prefill_with_kv_cache`
  - `"trtllm"`: Uses `trtllm_batch_context_with_kv_cache` (required for skip-softmax sparse)

**Skip-Softmax Sparse Attention:**
- Based on [arXiv:2512.12087](https://arxiv.org/abs/2512.12087)
- Skips softmax computation for low-attention-score blocks
- Higher threshold = more sparsity = faster but less accurate
- **Only supported on SM100 (B100) and SM103 (B200)**, NOT SM90/SM110/SM120
- Falls back to standard attention on unsupported architectures

### 2. Normalization APIs

| API | Description | Supported SM | Notes |
|-----|-------------|--------------|-------|
| `flashinfer.rmsnorm` | Fused RMSNorm | SM75+ | Replaces `torch.nn.RMSNorm` |

### 3. GEMM APIs (Linear Layers)

| Backend | API | Supported SM | Notes |
|---------|-----|--------------|-------|
| `torch` | `torch.nn.Linear` | All | Fallback, always available |
| `bf16` | `flashinfer.gemm.mm_bf16` | **SM100+** | BF16 matrix multiplication |
| `fp8` | `flashinfer.gemm.mm_fp8` | SM89+ (except SM90) | FP8 with TRT-LLM backend |
| `fp8_sm90` | `flashinfer.gemm.fp8_blockscale_gemm_sm90` | **SM90 only** | FP8 blockscale optimized for Hopper |
| `bmm_fp8` | `flashinfer.gemm.bmm_fp8` | SM89+ | Batched FP8 with cuBLAS |
| `fp8_groupwise` | `flashinfer.gemm.gemm_fp8_nt_groupwise` | **SM100+** | FP8 NT GEMM with groupwise (1,128,128) scaling |
| `fp8_blockscaled` | `flashinfer.gemm.gemm_fp8_nt_blockscaled` | **SM100+** | FP8 NT GEMM with (128,128,128) block scaling |
| `batch_deepgemm_fp8` | `flashinfer.gemm.batch_deepgemm_fp8_nt_groupwise` | **SM100, SM103** | DeepGEMM batched FP8 NT GEMM |
| `fp4` | `flashinfer.gemm.mm_fp4` | **SM100+** | FP4 matrix multiplication |
| `bmm_bf16` | `flashinfer.gemm.bmm_bf16` | **SM100+** | Batched BF16 |
| `mxfp8` | `flashinfer.gemm.mm_mxfp8` | **SM100+** | Microscaling FP8 |
| `bmm_mxfp8` | `flashinfer.gemm.bmm_mxfp8` | **SM100+** | Batched microscaling FP8 |

### 4. Activation APIs

| API | Description | Supported SM | Notes |
|-----|-------------|--------------|-------|
| `flashinfer.activation.gelu_and_mul` | Fused GELU with multiplication | SM75+ | Used in FFN layers |

## GPU Architecture Compatibility

| GPU Generation | SM Version | Recommended GEMM Backend | Skip-Softmax Sparse |
|----------------|------------|-------------------------|---------------------|
| Turing (RTX 20xx) | SM75 | `torch` | No |
| Ampere (A100, RTX 30xx) | SM80/86 | `torch` | No |
| Ada Lovelace (RTX 40xx, L40) | SM89 | `bmm_fp8` | No |
| Hopper (H100, H20) | SM90 | `fp8_sm90` | No |
| Blackwell (B100) | SM100 | `bf16` or `fp4` | **Yes** |
| Blackwell (B200) | SM103 | `bf16` or `fp4` | **Yes** |
| Blackwell (GB10) | SM120/121 | `bf16` or `fp4` | No |

## Usage

### Basic Usage

```python
from transformer_wan_flashinfer import WanTransformer3DConfig, FlashInferWanTransformer3DModel

config = WanTransformer3DConfig(
    num_attention_heads=40,
    attention_head_dim=128,
    num_layers=40,
    # FlashInfer options
    gemm_backend="torch",       # Explicitly choose: "torch", "bf16", "fp8", "fp4", etc.
    online_act_quant=True,      # True: compute scale from data; False: use default scale
    single_attention_backend="single", # For bs=1: "single", "cudnn", or "trtllm"
    batch_attention_backend="cudnn",   # For bs>1: "cudnn" or "trtllm"
    use_skip_softmax_sparse=False,
)

model = FlashInferWanTransformer3DModel(config).cuda().half()
```

### Offline Activation Quantization

For scenarios where activation scales are pre-calibrated or a fixed default is acceptable:

```python
config = WanTransformer3DConfig(
    # ... other config ...
    gemm_backend="fp8",         # FP8 backend
    online_act_quant=False,     # Use fixed default scale instead of computing from tensor
)
```

### Attention Backends

```python
# Use cuDNN for both single and batch (e.g. for benchmarking)
config = WanTransformer3DConfig(
    # ... other config ...
    single_attention_backend="cudnn",
    batch_attention_backend="cudnn",
)

# TRT-LLM with skip-softmax sparse (SM100/SM103 only)
config = WanTransformer3DConfig(
    # ... other config ...
    single_attention_backend="trtllm",
    batch_attention_backend="trtllm",
    use_skip_softmax_sparse=True,
    skip_softmax_threshold_scale_factor=1.0,
)
```

### Command Line

```bash
# Standard attention with torch GEMM backend (default)
python transformer_wan_flashinfer.py --gemm-backend torch

# Use offline activation quantization
python transformer_wan_flashinfer.py --gemm-backend fp8 --offline-act-quant

# Specify attention backends
python transformer_wan_flashinfer.py --single-attention-backend cudnn --batch-attention-backend cudnn

# Enable skip-softmax sparse attention (SM100+ required)
python transformer_wan_flashinfer.py --single-attention-backend trtllm --batch-attention-backend trtllm --skip-softmax-sparse --skip-softmax-threshold 1.0
```

### Test and Benchmark

```bash
# Quick sanity check
python test_and_benchmark.py quick

# Run all unit tests
python test_and_benchmark.py test

# Run specific tests
python test_and_benchmark.py test --basic --gemm --sparse --offline --attention

# Benchmark GEMM backends
python test_and_benchmark.py benchmark --gemm-backend torch bf16

# Benchmark with offline activation quantization
python test_and_benchmark.py benchmark --gemm-backend fp8 --offline-act-quant

# Benchmark with specific attention backends
python test_and_benchmark.py benchmark --single-attention-backend cudnn --batch-attention-backend cudnn --batch-size 2

# Benchmark GEMM backends + sparse attention (combined)
python test_and_benchmark.py benchmark --gemm-backend torch bf16 --sparse --threshold 1.0

# Full benchmark with larger model
python test_and_benchmark.py benchmark --gemm-backend bf16 --sparse --num-layers 4 --num-frames 32
```

## Configuration Options

### `WanTransformer3DConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gemm_backend` | str | `"torch"` | GEMM backend: `torch`, `bf16`, `fp8`, `fp8_sm90`, `bmm_fp8`, `fp8_groupwise`, `fp8_blockscaled`, `batch_deepgemm_fp8`, `fp4`, `bmm_bf16`, `mxfp8`, `bmm_mxfp8` |
| `online_act_quant` | bool | `True` | `True`: compute activation scale from data (online). `False`: use fixed default scale (offline) |
| `single_attention_backend` | str | `"single"` | Attention backend for batch_size == 1: `single`, `cudnn`, `trtllm` |
| `batch_attention_backend` | str | `"cudnn"` | Attention backend for batch_size > 1: `cudnn`, `trtllm` |
| `use_skip_softmax_sparse` | bool | `False` | Enable skip-softmax sparse attention (requires `trtllm` backend, SM100+ only) |
| `skip_softmax_threshold_scale_factor` | float | `1.0` | Threshold for skip-softmax (higher = more sparse, less accurate) |

## Performance Notes

1. **Attention**: Skip-softmax sparse attention can provide 1.3-2x speedup on SM100+ depending on threshold setting.

2. **GEMM**: FP8/FP4 backends provide significant speedups over FP16/BF16 on supported hardware.

3. **Memory**: Skip-softmax sparse attention uses paged KV cache internally, which may have different memory characteristics than standard attention.

## Accuracy Considerations

Skip-softmax sparse attention trades accuracy for performance:

| Threshold | Expected Speedup | Accuracy Impact |
|-----------|-----------------|-----------------|
| 0.5 | 1.1-1.3x | Minimal |
| 1.0 | 1.3-1.5x | Moderate |
| 2.0 | 1.5-2.0x | Significant |

For production use, benchmark with your specific workload to find the optimal threshold.

## Files

| File | Description |
|------|-------------|
| `transformer_wan_flashinfer.py` | Main model implementation |
| `test_and_benchmark.py` | Unified test and benchmark script |
| `README.md` | This documentation |

## References

- [FlashInfer Documentation](https://docs.flashinfer.ai/)
- [Skip-Softmax Sparse Attention Paper](https://arxiv.org/abs/2512.12087)
- [Original Wan Transformer (Diffusers)](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py)
