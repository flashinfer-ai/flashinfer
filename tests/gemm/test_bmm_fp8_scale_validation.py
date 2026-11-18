"""Tests for bmm_fp8 scale tensor validation.

Issue: https://github.com/flashinfer-ai/flashinfer/issues/4052
"""

import pytest
import torch

from flashinfer.gemm import bmm_fp8


def test_bmm_fp8_valid_scalar_scale():
    """Test that valid scalar scales work correctly."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, m, k, n = 2, 4, 8, 16

    A = torch.randn(b, m, k, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    B = torch.randn(b, k, n, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    A_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    B_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    out = bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")
    assert out.shape == (b, m, n)
    assert out.dtype == torch.bfloat16


def test_bmm_fp8_per_row_scale_rejected():
    """Test that per-row scale (wrong shape) is rejected."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, m, k, n = 2, 4, 8, 16

    A = torch.randn(b, m, k, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    B = torch.randn(b, k, n, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    A_scale = torch.ones(b, m, 1, dtype=torch.float32, device=device)  # Wrong: per-row
    B_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="A_scale must be a scalar"):
        bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")


def test_bmm_fp8_wrong_dtype_rejected():
    """Test that non-float32 scale is rejected."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, m, k, n = 2, 4, 8, 16

    A = torch.randn(b, m, k, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    B = torch.randn(b, k, n, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    A_scale = torch.tensor(1.0, dtype=torch.float16, device=device)  # Wrong: float16
    B_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    with pytest.raises(ValueError, match="A_scale must be torch.float32"):
        bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")


def test_bmm_fp8_wrong_device_rejected():
    """Test that scale on wrong device is rejected."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for device validation test")

    b, m, k, n = 2, 4, 8, 16

    A = torch.randn(b, m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    B = torch.randn(b, k, n, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    A_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    B_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    # Test with scale on CPU while operands are on CUDA
    A_scale_cpu = torch.tensor(1.0, dtype=torch.float32, device="cpu")

    with pytest.raises(ValueError, match="A_scale must be on the same CUDA device"):
        bmm_fp8(A, B, A_scale_cpu, B_scale, torch.bfloat16, backend="cublas")


def test_bmm_fp8_zero_dim_scalar_rejected():
    """Test that 0-dim scalar is accepted (this is the valid case)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, m, k, n = 2, 4, 8, 16

    A = torch.randn(b, m, k, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    B = torch.randn(b, k, n, device=device, dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    # Valid: 0-dim scalar
    A_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    B_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    assert A_scale.dim() == 0
    assert A_scale.numel() == 1
    out = bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")
    assert out.shape == (b, m, n)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
