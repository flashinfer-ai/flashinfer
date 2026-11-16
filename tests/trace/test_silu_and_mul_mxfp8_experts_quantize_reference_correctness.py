"""Reference correctness test for the silu_and_mul_mxfp8_experts_quantize trace API.

Valid rows are compared bit for bit with the template reference.
"""

import pytest
import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
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


@torch.inference_mode()
def test_silu_and_mul_mxfp8_experts_quantize_check():
    """The template check compares the valid rows and nothing past them."""
    _skip_if_not_sm100()
    import flashinfer
    from flashinfer.trace.templates.quantize import (
        silu_and_mul_mxfp8_experts_quantize_trace as trace,
    )

    a = trace.init(B=2, M=128, K_doubled=512)["a"]
    a[0, 0, 0] = float("inf")  # NaN in the reference, 448 in the kernel.
    mask = torch.tensor([128, 5], dtype=torch.int32, device=a.device)
    ref = trace.reference(a, mask)
    out, sf = flashinfer.silu_and_mul_mxfp8_experts_quantize(a, mask)
    assert trace.check(ref, (out, sf), mask=mask)

    out_bytes = out.view(torch.uint8)
    out_bytes[5, 0, 1] ^= 1  # Past the valid rows of expert 1.
    assert trace.check(ref, (out, sf), mask=mask)
    out_bytes[4, 0, 1] ^= 1
    assert not trace.check(ref, (out, sf), mask=mask)
    out_bytes[4, 0, 1] ^= 1
    sf[0, 0, 0, 0, 0, 0] ^= 1
    assert not trace.check(ref, (out, sf), mask=mask)


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
    _check(trace, (q_ref, sf_ref), (out, sf), mask=mask)
