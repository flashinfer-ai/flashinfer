"""Reference correctness test for the mxfp8_grouped_quantize trace API.

Compares payload bytes and logical block-scale bytes with the shared MXFP8
reference. Padding outside valid rows is unspecified by the API.
"""

import pytest
import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _skip_if_not_sm100,
    _unswizzle_mxfp8_scales_128x4,
)
from tests.utils_fp8 import mxfp8_quantize_reference


@pytest.mark.parametrize(
    "shape_kwargs", [dict(B=2, M=128, K=4096), dict(B=3, M=256, K=2048)]
)
@torch.inference_mode()
def test_mxfp8_grouped_quantize_reference_correctness(shape_kwargs):
    _skip_if_not_sm100()
    import flashinfer
    from flashinfer.cutile import is_cuda_tile_available
    from flashinfer.trace.templates.quantize import mxfp8_grouped_quantize_trace

    if not is_cuda_tile_available():
        pytest.skip("cuda.tile is not available")

    inputs = mxfp8_grouped_quantize_trace.init(**shape_kwargs)
    a = inputs["a"]
    _assert_finite(a)
    # The guards above gate unavailable environments, so a kernel exception
    # here is a real regression, not a reason to skip.
    out, sf = flashinfer.mxfp8_grouped_quantize(a, inputs["mask"])

    b, m, k = a.shape
    padded_k = (k + 127) // 128 * 128

    # Map the API layout back to plain per-group tensors:
    #   out: logical [M, padded_K, B] (fp8)        -> [B, M, padded_K]
    #   sf:  logical [32, 4, rm, 4, rk, B] (uint8) -> per group [rm, rk, 32, 4, 4]
    q = out.permute(2, 0, 1)
    sf_grouped = sf.permute(5, 2, 4, 0, 1, 3).contiguous()

    # init uses a full mask (mask == M), so every row is valid and comparable.
    for i in range(b):
        q_ref, sf_ref = mxfp8_quantize_reference(
            a[i], is_sf_swizzled_layout=False, alignment=128
        )
        scale_bytes = _unswizzle_mxfp8_scales_128x4(sf_grouped[i], m, padded_k)
        torch.testing.assert_close(
            q[i].contiguous().view(torch.uint8),
            q_ref.contiguous().view(torch.uint8),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            scale_bytes, sf_ref.reshape(m, padded_k // 32), atol=0, rtol=0
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
