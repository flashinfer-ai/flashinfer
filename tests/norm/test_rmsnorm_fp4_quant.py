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
from tests.test_helpers.utils_fp4 import cast_from_fp4


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


def dequantize_fp4_output(y_fp4: torch.Tensor, block_scale: torch.Tensor, block_size: int):
    """
    Dequantize packed FP4 tensor using the associated block scales.

    Handles both 2D inputs shaped [B, H/2] and 3D inputs shaped [B, S, H/2].
    """
    y_fp4_float = cast_from_fp4(y_fp4)
    if y_fp4_float.dim() == 2:
        b, hidden_size = y_fp4_float.shape
        assert hidden_size % block_size == 0
        y_fp4_float = y_fp4_float.view(b, hidden_size // block_size, block_size)
        scales = block_scale.float().unsqueeze(-1)
        return (y_fp4_float * scales).reshape(b, hidden_size)
    elif y_fp4_float.dim() == 3:
        b, s, hidden_size = y_fp4_float.shape
        assert hidden_size % block_size == 0
        y_fp4_float = y_fp4_float.view(b, s, hidden_size // block_size, block_size)
        scales = block_scale.float().unsqueeze(-1)
        return (y_fp4_float * scales).reshape(b, s, hidden_size)
    else:
        raise ValueError(f"Unsupported FP4 output rank: {y_fp4_float.dim()}")


def ref_block_scale_quantize(x, block_size=16):
    """
    Reference block scale quantization matching cuDNN behavior.
    
    For each block of block_size elements:
    1. Find max absolute value
    2. Compute scale = max_abs / 6.0 (FP4_E2M1_MAX)
    3. Convert scale to FP8_E4M3
    4. Quantize: value / scale, then round to nearest FP4 value
    
    Handles both 2D inputs shaped [B, H] and 3D inputs shaped [B, S, H].
    """
    FLOAT4_E2M1_MAX = 6.0
    
    orig_shape = x.shape
    hidden_size = x.shape[-1]
    num_blocks = hidden_size // block_size
    
    if x.dim() == 2:
        batch_size = x.shape[0]
        # Reshape to [batch_size, num_blocks, block_size]
        x_blocked = x.view(batch_size, num_blocks, block_size)
        scale_shape = (batch_size, num_blocks)
    elif x.dim() == 3:
        batch_size, seq_len = x.shape[0], x.shape[1]
        # Reshape to [batch_size, seq_len, num_blocks, block_size]
        x_blocked = x.view(batch_size, seq_len, num_blocks, block_size)
        scale_shape = (batch_size, seq_len, num_blocks)
    else:
        raise ValueError(f"Unsupported input rank: {x.dim()}")
    
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
    
    return x_fp4.view(orig_shape), scale.view(scale_shape).to(torch.float8_e4m3fn)


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
@pytest.mark.parametrize("batch_size", [4, 16, 32])
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
# @pytest.mark.parametrize("batch_size", [128])
# @pytest.mark.parametrize("hidden_size", [1536])
# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("eps", [1e-5])
def test_rmsnorm_fp4quant_2d(batch_size, hidden_size, dtype, eps):
    """Test fused RMSNorm + FP4 quantization with 2D input."""
    torch.manual_seed(42)
    
    block_size = 16
    
    # Create input tensors
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    
    # Allocate output tensors
    y_fp4 = torch.empty(batch_size, hidden_size // 2, device="cuda", dtype=torch.uint8)
    block_scale = torch.empty(batch_size, hidden_size // block_size, device="cuda", dtype=torch.float8_e4m3fn)
    
    # Run fused kernel
    rmsnorm_fp4quant(x, weight, y_fp4, block_scale, eps=eps, block_size=block_size)
    
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
    
    # Dequantize FP4 output for value-level comparison
    # FP4 E2M1 has limited precision with values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    # The gap between 4 and 6 is 2, so max quantization error is 1.0 in FP4 space
    # After scaling, this error can be significant for boundary values (e.g., 5.0)
    y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
    torch.testing.assert_close(
        y_dequant,
        ref_rmsnorm.float(),
        rtol=0.5,
        atol=1.0,
    )


@cudnn_fp4_available
@blackwell_required
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [16, 64, 128])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("batch_size", [])
# @pytest.mark.parametrize("seq_len", [])
# @pytest.mark.parametrize("hidden_size", [])
# @pytest.mark.parametrize("dtype", [])
def test_rmsnorm_fp4quant_3d(batch_size, seq_len, hidden_size, dtype):
    """Test fused RMSNorm + FP4 quantization with 3D input."""
    torch.manual_seed(42)
    
    block_size = 16
    eps = 1e-5
    
    # Create input tensors
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    
    # Allocate output tensors
    y_fp4 = torch.empty(batch_size, seq_len, hidden_size // 2, device="cuda", dtype=torch.uint8)
    block_scale = torch.empty(batch_size, seq_len, hidden_size // block_size, device="cuda", dtype=torch.float8_e4m3fn)
    
    # Run fused kernel
    rmsnorm_fp4quant(x, weight, y_fp4, block_scale, eps=eps, block_size=block_size)
    
    # Verify output shapes
    assert y_fp4.shape == (batch_size, seq_len, hidden_size // 2), \
        f"Expected FP4 shape {(batch_size, seq_len, hidden_size // 2)}, got {y_fp4.shape}"
    assert block_scale.shape == (batch_size, seq_len, hidden_size // block_size), \
        f"Expected scale shape {(batch_size, seq_len, hidden_size // block_size)}, got {block_scale.shape}"
    
    # Verify output dtypes
    assert y_fp4.dtype == torch.uint8
    assert block_scale.dtype == torch.float8_e4m3fn

    # Reference computation
    ref_rmsnorm = llama_rms_norm(x, weight, eps=eps)
    ref_fp4, ref_scale = ref_block_scale_quantize(ref_rmsnorm, block_size=block_size)

    # Dequantize FP4 output for value-level comparison
    # FP4 E2M1 has limited precision with values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
    # The gap between 4 and 6 is 2, so max quantization error is 1.0 in FP4 space
    y_dequant = dequantize_fp4_output(y_fp4, block_scale, block_size)
    torch.testing.assert_close(
        y_dequant,
        ref_rmsnorm.float(),
        rtol=0.5,
        atol=1.0,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
