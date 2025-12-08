"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer
from flashinfer.norm import rmsnorm_fp4quant, CUDNN_AVAILABLE


def get_cc():
    """Get CUDA compute capability."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def llama_rms_norm(x, w, eps=1e-6):
    """Reference RMSNorm implementation (LLaMA style)."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


# FP4 E2M1 lookup table for dequantization
E2M1_TO_FLOAT32 = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def cast_from_fp4(x):
    """
    Convert packed FP4 output to float.
    
    cuDNN FP4_E2M1 packing: byte = [v_odd | v_even]
    - Low nibble = even index element
    - High nibble = odd index element
    """
    v_low = x & 0xF
    v_high = (x >> 4) & 0xF
    c = torch.stack((v_low, v_high), dim=-1)
    new_shape = c.shape[:-2] + (c.shape[-2] * c.shape[-1],)
    lookup_table = torch.tensor(E2M1_TO_FLOAT32, device=c.device)
    out = lookup_table[c.to(torch.long)].reshape(new_shape).to(torch.float32)
    return out


def ref_block_scale_quantize(x, block_size=16):
    """
    Reference block scale quantization matching cuDNN behavior.
    
    For each block of block_size elements:
    1. Find max absolute value
    2. Compute scale = max_abs / 6.0 (FP4_E2M1_MAX)
    3. Convert scale to FP8_E4M3
    4. Quantize: value / scale, then round to nearest FP4 value
    """
    FLOAT4_E2M1_MAX = 6.0
    
    orig_shape = x.shape
    batch_size = x.shape[0]
    hidden_size = x.shape[-1]
    
    # Reshape to [..., num_blocks, block_size]
    num_blocks = hidden_size // block_size
    x_blocked = x.view(batch_size, num_blocks, block_size)
    
    # Compute per-block max
    max_abs = x_blocked.abs().max(dim=-1, keepdim=True)[0].float()
    
    # Compute scale (max_abs / 6.0), then round to FP8
    scale = max_abs / FLOAT4_E2M1_MAX
    scale = scale.to(torch.float8_e4m3fn).float()
    
    # Avoid division by zero
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    
    # Quantize
    x_scaled = x_blocked.float() / scale
    x_quantized = torch.clamp(x_scaled, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX)
    
    # Round to nearest FP4 value
    x_fp4 = cast_to_fp4_value(x_quantized)
    
    return x_fp4.view(orig_shape), scale.view(batch_size, num_blocks).to(torch.float8_e4m3fn)


def cast_to_fp4_value(x):
    """Cast float values to nearest FP4 E2M1 representable value."""
    sign = torch.sign(x)
    x_abs = torch.abs(x)
    
    result = torch.zeros_like(x_abs)
    result[(x_abs >= 0.0) & (x_abs <= 0.25)] = 0.0
    result[(x_abs > 0.25) & (x_abs < 0.75)] = 0.5
    result[(x_abs >= 0.75) & (x_abs <= 1.25)] = 1.0
    result[(x_abs > 1.25) & (x_abs < 1.75)] = 1.5
    result[(x_abs >= 1.75) & (x_abs <= 2.5)] = 2.0
    result[(x_abs > 2.5) & (x_abs < 3.5)] = 3.0
    result[(x_abs >= 3.5) & (x_abs <= 5.0)] = 4.0
    result[x_abs > 5.0] = 6.0
    
    return result * sign


def requires_cudnn_fp4():
    """Check if cuDNN FP4 is available."""
    if not CUDNN_AVAILABLE:
        return False
    try:
        import cudnn
        return cudnn.backend_version() >= 90700
    except Exception:
        return False


def requires_blackwell():
    """Check if running on Blackwell GPU."""
    return get_cc() >= 100


# Skip conditions
cudnn_fp4_available = pytest.mark.skipif(
    not requires_cudnn_fp4(),
    reason="cuDNN FP4 support requires cuDNN >= 9.7.0"
)

blackwell_required = pytest.mark.skipif(
    not requires_blackwell(),
    reason="FP4 block scale quantization requires Blackwell GPU (compute capability >= 100)"
)


@cudnn_fp4_available
@blackwell_required
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_rmsnorm_fp4quant_2d(batch_size, hidden_size, dtype, eps):
    """Test fused RMSNorm + FP4 quantization with 2D input."""
    torch.manual_seed(42)
    
    block_size = 16
    
    # Create input tensors
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    
    # Run fused kernel
    y_fp4, block_scale = rmsnorm_fp4quant(x, weight, eps=eps, block_size=block_size)
    
    # Verify output shapes
    assert y_fp4.shape == (batch_size, hidden_size // 2), \
        f"Expected FP4 shape {(batch_size, hidden_size // 2)}, got {y_fp4.shape}"
    assert block_scale.shape == (batch_size, hidden_size // block_size), \
        f"Expected scale shape {(batch_size, hidden_size // block_size)}, got {block_scale.shape}"
    
    # Verify output dtypes
    assert y_fp4.dtype == torch.uint8, f"Expected uint8 for FP4, got {y_fp4.dtype}"
    assert block_scale.dtype == torch.float8_e4m3fn, \
        f"Expected float8_e4m3fn for scale, got {block_scale.dtype}"
    
    # Reference computation
    ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)
    ref_fp4, ref_scale = ref_block_scale_quantize(ref_rmsnorm, block_size=block_size)
    
    # Unpack FP4 values for comparison
    y_fp4_float = cast_from_fp4(y_fp4)
    
    # Compare FP4 quantized values
    # Due to precision differences, we allow some tolerance
    fp4_match_ratio = (y_fp4_float == ref_fp4).float().mean().item()
    assert fp4_match_ratio > 0.90, \
        f"FP4 match ratio too low: {fp4_match_ratio * 100:.2f}%"
    
    # Compare scales (after accounting for possible FP8 rounding differences)
    scale_diff = (block_scale.float() - ref_scale.float()).abs()
    assert scale_diff.max().item() < 0.1, \
        f"Scale difference too large: max={scale_diff.max().item()}"


@cudnn_fp4_available
@blackwell_required
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [16, 64, 128])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_fp4quant_3d(batch_size, seq_len, hidden_size, dtype):
    """Test fused RMSNorm + FP4 quantization with 3D input."""
    torch.manual_seed(42)
    
    block_size = 16
    eps = 1e-5
    
    # Create input tensors
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    
    # Run fused kernel
    y_fp4, block_scale = rmsnorm_fp4quant(x, weight, eps=eps, block_size=block_size)
    
    # Verify output shapes
    assert y_fp4.shape == (batch_size, seq_len, hidden_size // 2), \
        f"Expected FP4 shape {(batch_size, seq_len, hidden_size // 2)}, got {y_fp4.shape}"
    assert block_scale.shape == (batch_size, seq_len, hidden_size // block_size), \
        f"Expected scale shape {(batch_size, seq_len, hidden_size // block_size)}, got {block_scale.shape}"
    
    # Verify output dtypes
    assert y_fp4.dtype == torch.uint8
    assert block_scale.dtype == torch.float8_e4m3fn


@cudnn_fp4_available
@blackwell_required
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_fp4quant_dequantize_accuracy(dtype):
    """Test that dequantized output is close to original RMSNorm output."""
    torch.manual_seed(42)
    
    batch_size = 8
    hidden_size = 256
    block_size = 16
    eps = 1e-5
    
    # Create input tensors
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    
    # Run fused kernel
    y_fp4, block_scale = rmsnorm_fp4quant(x, weight, eps=eps, block_size=block_size)
    
    # Reference RMSNorm output
    ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)
    
    # Dequantize
    y_fp4_float = cast_from_fp4(y_fp4)
    y_dequant = y_fp4_float.view(batch_size, -1, block_size) * block_scale.float().unsqueeze(-1)
    y_dequant = y_dequant.view(batch_size, hidden_size)
    
    # Compare with reference
    error = (y_dequant - ref_rmsnorm.float()).abs()
    relative_error = error / (ref_rmsnorm.float().abs() + 1e-6)
    
    # FP4 quantization has limited precision, so we use loose tolerance
    assert relative_error.mean().item() < 0.5, \
        f"Mean relative error too high: {relative_error.mean().item()}"


@cudnn_fp4_available
@blackwell_required
def test_rmsnorm_fp4quant_vs_sequential():
    """Compare fused kernel with sequential RMSNorm + fp4_quantize."""
    torch.manual_seed(42)
    
    batch_size = 4
    hidden_size = 128
    block_size = 16
    eps = 1e-5
    dtype = torch.float16
    
    # Create input tensors
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    
    # Run fused kernel
    y_fp4_fused, scale_fused = rmsnorm_fp4quant(x, weight, eps=eps, block_size=block_size)
    
    # Sequential: RMSNorm then reference FP4 quantization
    rmsnorm_out = flashinfer.norm.rmsnorm(x, weight, eps=eps)
    ref_fp4, ref_scale = ref_block_scale_quantize(rmsnorm_out, block_size=block_size)
    
    # Compare FP4 values
    y_fp4_float = cast_from_fp4(y_fp4_fused)
    fp4_match_ratio = (y_fp4_float == ref_fp4).float().mean().item()
    
    # Should have high match ratio
    assert fp4_match_ratio > 0.90, \
        f"FP4 match ratio between fused and sequential: {fp4_match_ratio * 100:.2f}%"
    
    # Compare scales
    scale_diff = (scale_fused.float() - ref_scale.float()).abs()
    assert scale_diff.max().item() < 0.1, \
        f"Scale difference between fused and sequential: max={scale_diff.max().item()}"


@cudnn_fp4_available
@blackwell_required
def test_rmsnorm_fp4quant_invalid_dtype():
    """Test that invalid dtype raises ValueError."""
    x = torch.randn(4, 64, device="cuda", dtype=torch.float32)
    weight = torch.randn(64, device="cuda", dtype=torch.float32)
    
    with pytest.raises(ValueError, match="Unsupported input dtype"):
        rmsnorm_fp4quant(x, weight)


@cudnn_fp4_available
@blackwell_required
def test_rmsnorm_fp4quant_dtype_mismatch():
    """Test that mismatched dtypes raise ValueError."""
    x = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    
    with pytest.raises(ValueError, match="must have the same dtype"):
        rmsnorm_fp4quant(x, weight)


@cudnn_fp4_available
@blackwell_required
def test_rmsnorm_fp4quant_invalid_hidden_size():
    """Test that invalid hidden_size (not divisible by block_size) raises ValueError."""
    x = torch.randn(4, 65, device="cuda", dtype=torch.float16)  # 65 not divisible by 16
    weight = torch.randn(65, device="cuda", dtype=torch.float16)
    
    with pytest.raises(ValueError, match="must be divisible by block_size"):
        rmsnorm_fp4quant(x, weight, block_size=16)


@cudnn_fp4_available
@blackwell_required
def test_rmsnorm_fp4quant_deterministic():
    """Test that the function is deterministic."""
    torch.manual_seed(42)
    
    x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
    weight = torch.randn(128, device="cuda", dtype=torch.float16)
    
    y_fp4_1, scale_1 = rmsnorm_fp4quant(x, weight)
    y_fp4_2, scale_2 = rmsnorm_fp4quant(x, weight)
    
    assert torch.equal(y_fp4_1, y_fp4_2), "FP4 output is not deterministic"
    assert torch.equal(scale_1, scale_2), "Scale output is not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

