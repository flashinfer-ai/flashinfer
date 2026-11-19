"""Tests for bmm_fp8 scale tensor validation.

Issue: https://github.com/flashinfer-ai/flashinfer/issues/4052
"""

import pytest
import torch

from flashinfer.gemm import bmm_fp8


def test_bmm_fp8_valid_scalar_scale():
    """Test that valid scalar scales work correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for bmm_fp8 test")

    b, m, k, n = 2, 16, 16, 16

    A = torch.randn(b, m, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    # B must be column-major: create (b, n, k) then transpose to (b, k, n)
    B = torch.randn(b, n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    ).transpose(-2, -1)
    A_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    B_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    out = bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")
    assert out.shape == (b, m, n)
    assert out.dtype == torch.bfloat16

    # Verify correctness against torch.bmm
    A_bf16 = A.to(torch.bfloat16)
    B_bf16 = B.to(torch.bfloat16)
    expected = torch.bmm(A_bf16, B_bf16)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)


def test_bmm_fp8_per_row_scale_rejected():
    """Test that per-row scale (wrong shape) is rejected."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for bmm_fp8 test")

    b, m, k, n = 2, 16, 16, 16

    A = torch.randn(b, m, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    B = torch.randn(b, n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    ).transpose(-2, -1)
    A_scale = torch.ones(b, m, 1, dtype=torch.float32, device="cuda")  # Wrong: per-row
    B_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="A_scale must be a scalar"):
        bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")


def test_bmm_fp8_wrong_dtype_rejected():
    """Test that non-float32 scale is rejected."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for bmm_fp8 test")

    b, m, k, n = 2, 16, 16, 16

    A = torch.randn(b, m, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    B = torch.randn(b, n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    ).transpose(-2, -1)
    A_scale = torch.tensor(1.0, dtype=torch.float16, device="cuda")  # Wrong: float16
    B_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="A_scale must be torch.float32"):
        bmm_fp8(A, B, A_scale, B_scale, torch.bfloat16, backend="cublas")


def test_bmm_fp8_wrong_device_rejected():
    """Test that scale on wrong device is rejected."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for device validation test")

    b, m, k, n = 2, 16, 16, 16

    A = torch.randn(b, m, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    B = torch.randn(b, n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    ).transpose(-2, -1)

    # Test with scale on CPU while operands are on CUDA
    A_scale_cpu = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    B_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    with pytest.raises(ValueError, match="A_scale must be on the same CUDA device"):
        bmm_fp8(A, B, A_scale_cpu, B_scale, torch.bfloat16, backend="cublas")


def test_bmm_fp8_one_element_scalar_accepted():
    """Test that 1-element tensor with shape (1,) or (1,1,1) is accepted."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for bmm_fp8 test")

    b, m, k, n = 2, 16, 16, 16

    A = torch.randn(b, m, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    B = torch.randn(b, n, k, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    ).transpose(-2, -1)

    # Test shape (1,)
    A_scale_1 = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    assert A_scale_1.shape == (1,)
    assert A_scale_1.numel() == 1
    B_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    out = bmm_fp8(A, B, A_scale_1, B_scale, torch.bfloat16, backend="cublas")
    assert out.shape == (b, m, n)

    # Test shape (1,1,1)
    A_scale_3d = torch.ones(1, 1, 1, dtype=torch.float32, device="cuda")
    assert A_scale_3d.shape == (1, 1, 1)
    assert A_scale_3d.numel() == 1

    out = bmm_fp8(A, B, A_scale_3d, B_scale, torch.bfloat16, backend="cublas")
    assert out.shape == (b, m, n)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
