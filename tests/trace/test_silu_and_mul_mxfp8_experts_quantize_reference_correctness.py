"""Reference correctness test for the silu_and_mul_mxfp8_experts_quantize trace API.

The template reference uses a plain layout; the kernel output is mapped back
to it and both are compared bit for bit on the valid rows of each expert.
"""

import pytest
import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _skip_if_not_sm100,
    _unswizzle_mxfp8_scales_128x4,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(B=2, M=128, K_doubled=4096), dict(B=3, M=200, K_doubled=2 * 2080)],
)
@torch.inference_mode()
def test_silu_and_mul_mxfp8_experts_quantize_reference_correctness(shape_kwargs):
    _skip_if_not_sm100()
    from flashinfer.trace.templates.quantize import (
        silu_and_mul_mxfp8_experts_quantize_trace as trace,
    )

    inputs = trace.init(**shape_kwargs)
    a = inputs["a"]
    _assert_finite(a)
    b, m, _ = a.shape
    # Partial masks, so only the valid rows are compared.
    mask = torch.randint(1, m + 1, (b,), dtype=torch.int32, device=a.device)

    _assert_matches_reference(a, mask, trace)


def _assert_matches_reference(a, mask, trace):
    import flashinfer

    b, m, k_doubled = a.shape
    k = k_doubled // 2
    padded_k = (k + 127) // 128 * 128
    out, sf = flashinfer.silu_and_mul_mxfp8_experts_quantize(a, mask)
    q_ref, sf_ref = trace.reference(a, mask)

    # out: logical [M, padded_K, B] -> [B, M, K]
    # sf:  logical [32, 4, rm, 4, rk, B] -> per expert [rm, rk, 32, 4, 4]
    q = out.permute(2, 0, 1)[:, :, :k]
    sf_grouped = sf.permute(5, 2, 4, 0, 1, 3).contiguous()
    for i, valid in enumerate(mask.tolist()):
        scale_bytes = _unswizzle_mxfp8_scales_128x4(sf_grouped[i], m, padded_k)
        torch.testing.assert_close(
            q[i, :valid].contiguous().view(torch.uint8),
            q_ref[i, :valid].view(torch.uint8),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            scale_bytes[:valid, : k // 32], sf_ref[i, :valid], rtol=0, atol=0
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_silu_and_mul_mxfp8_experts_quantize_reference_non_finite(dtype):
    """The reference models NaN, infinities and flushed scales as the kernel does."""
    _skip_if_not_sm100()
    from flashinfer.trace.templates.quantize import (
        silu_and_mul_mxfp8_experts_quantize_trace as trace,
    )

    torch.manual_seed(0)
    b, m, k = 2, 128, 256
    gate = torch.randn((b, m, k), device="cuda") * 4
    up = torch.randn((b, m, k), device="cuda") * 4
    gate[0, 0, 3] = float("nan")
    up[0, 0, 40] = float("nan")
    up[0, 2, 64:96] = float("nan")  # an all-NaN block
    gate[0, 1, 5] = 10.0
    up[0, 1, 5] = float("inf")
    gate[1, 7, 200] = 10.0
    up[1, 7, 200] = float("-inf")
    if dtype == torch.bfloat16:
        # A block whose float32 scale is subnormal, which the kernel flushes to zero.
        gate[1, 3, :32] = 1.0
        up[1, 3, :32] = 4e-36
    else:
        # silu(300) * 300 overflows float16 to infinity.
        gate[1, 9, 17] = 300.0
        up[1, 9, 17] = 300.0
    a = torch.cat([gate, up], dim=-1).to(dtype)
    mask = torch.full((b,), m, dtype=torch.int32, device="cuda")

    _assert_matches_reference(a, mask, trace)
