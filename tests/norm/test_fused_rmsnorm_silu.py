"""
Tests for flashinfer.fused_rmsnorm_silu() — fused RMSNorm + SiLU via scrub kernel.
"""

import math
import pytest
import torch
import torch.nn.functional as F

import flashinfer


def rmsnorm_silu_reference(x, weight, eps):
    """PyTorch reference: RMSNorm + SiLU.

    RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    SiLU: y * sigmoid(y)
    """
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    normed = (x.float() / rms) * weight.float()
    return F.silu(normed).to(x.dtype)


C_VALUES = [64, 128, 160, 256, 320, 512, 640, 1024]
TOKEN_VALUES = [1560, 6240, 24960, 99840, 399360]


@pytest.mark.parametrize("C", C_VALUES)
@pytest.mark.parametrize("num_tokens", TOKEN_VALUES)
def test_fused_rmsnorm_silu_correctness(C, num_tokens):
    """Test fused RMSNorm+SiLU against PyTorch reference."""
    torch.manual_seed(42)

    x = (torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0)
    w = (torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5)
    eps = 1e-6

    out = flashinfer.fused_rmsnorm_silu(x, w, eps)
    ref = rmsnorm_silu_reference(x, w, eps)

    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("C", [64, 256, 512, 1024])
def test_fused_rmsnorm_silu_preallocated_out(C):
    """Test with pre-allocated output tensor."""
    torch.manual_seed(42)
    num_tokens = 1560

    x = torch.randn(num_tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    w = torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5
    out = torch.empty_like(x)

    result = flashinfer.fused_rmsnorm_silu(x, w, 1e-6, out=out)
    ref = rmsnorm_silu_reference(x, w, 1e-6)

    assert result is out, "Should return the same tensor when out is provided"
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_fused_rmsnorm_silu_output_properties():
    """Test basic output properties: shape, dtype, no NaN/Inf."""
    x = torch.randn(1560, 512, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0
    w = torch.ones(512, dtype=torch.bfloat16, device="cuda")

    out = flashinfer.fused_rmsnorm_silu(x, w, 1e-6)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, skipping")
        exit(0)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    passed = 0
    failed = 0
    for C in C_VALUES:
        for tokens in TOKEN_VALUES:
            try:
                torch.manual_seed(42)
                x = (torch.randn(tokens, C, dtype=torch.bfloat16, device="cuda") * 5.0 + 5.0)
                w = (torch.rand(C, dtype=torch.bfloat16, device="cuda") * 1.5 + 0.5)
                out = flashinfer.fused_rmsnorm_silu(x, w, 1e-6)
                ref = rmsnorm_silu_reference(x, w, 1e-6)
                max_diff = (out.float() - ref.float()).abs().max().item()
                mismatches = (~torch.isclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)).sum().item()
                status = "PASS" if mismatches == 0 else "FAIL"
                if mismatches == 0:
                    passed += 1
                else:
                    failed += 1
                print(f"  C={C:>4}, tokens={tokens:>6}: {status}  "
                      f"max_diff={max_diff:.3e}  mismatches={mismatches}/{out.numel()}")
            except Exception as e:
                failed += 1
                print(f"  C={C:>4}, tokens={tokens:>6}: ERROR  {e}")

    print(f"\n--- Results: {passed} passed, {failed} failed ---")
