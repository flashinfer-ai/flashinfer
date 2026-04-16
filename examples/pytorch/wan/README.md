# FlashInfer-Optimized Wan Transformer

This directory contains a FlashInfer-optimized implementation of the [Wan Video Transformer](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_wan.py) for video generation.

## Overview

The `FlashInferWanTransformer3DModel` replaces standard PyTorch operations with FlashInfer kernels for improved inference performance. This implementation supports multiple GPU architectures and automatically selects the best available backend.

## Integrated FlashInfer APIs

### 1. Attention APIs

| API | Description | Supported SM | Notes |
|-----|-------------|--------------|-------|
| `single_prefill_with_kv_cache` | Standard scaled dot-product attention | SM75+ | Default attention implementation |
| `trtllm_batch_context_with_kv_cache` | Skip-softmax sparse attention | **SM100, SM103 only** | Enables sparse attention via `skip_softmax_threshold_scale_factor` |

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
    gemm_backend="auto",  # Auto-select best backend
    use_skip_softmax_sparse=False,
)

model = FlashInferWanTransformer3DModel(config).cuda().half()
```

### Enable Skip-Softmax Sparse Attention (SM100+ only)

```python
config = WanTransformer3DConfig(
    # ... other config ...
    use_skip_softmax_sparse=True,
    skip_softmax_threshold_scale_factor=1.0,  # Higher = more sparse
)
```

### Command Line

```bash
# Standard attention with auto GEMM backend
python transformer_wan_flashinfer.py --gemm-backend auto

# Enable skip-softmax sparse attention (SM100+ required)
python transformer_wan_flashinfer.py --skip-softmax-sparse --skip-softmax-threshold 1.0
```

### Test and Benchmark

```bash
# Quick sanity check
python test_and_benchmark.py quick

# Run all unit tests
python test_and_benchmark.py test

# Run specific tests
python test_and_benchmark.py test --basic --gemm --sparse

# Benchmark GEMM backends only
python test_and_benchmark.py benchmark --gemm-backend auto torch

# Benchmark GEMM backends + sparse attention (combined)
python test_and_benchmark.py benchmark --gemm-backend auto torch --sparse --threshold 1.0

# Full benchmark with larger model
python test_and_benchmark.py benchmark --gemm-backend auto --sparse --num-layers 4 --num-frames 32
```

## Configuration Options

### `WanTransformer3DConfig`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gemm_backend` | str | `"auto"` | GEMM backend: `auto`, `torch`, `bf16`, `fp8`, `fp8_sm90`, `bmm_fp8`, `fp4`, `bmm_bf16`, `mxfp8`, `bmm_mxfp8` |
| `use_skip_softmax_sparse` | bool | `False` | Enable skip-softmax sparse attention (SM100+ only) |
| `skip_softmax_threshold_scale_factor` | float | `1.0` | Threshold for skip-softmax (higher = more sparse, less accurate) |

### Auto Backend Selection

When `gemm_backend="auto"`, the best backend is selected based on GPU:

| SM Version | Selected Backend |
|------------|-----------------|
| SM100+ | `bf16` |
| SM90 | `fp8_sm90` |
| SM89 | `torch` |
| < SM89 | `torch` |

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
