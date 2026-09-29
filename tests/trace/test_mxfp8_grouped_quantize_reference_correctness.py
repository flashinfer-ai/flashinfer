"""Reference correctness test for the mxfp8_grouped_quantize trace API.

Verifies the kernel by a dequantized round-trip: unswizzle the emitted MXFP8
block scales, dequantize the FP8 values, and compare the result against the
original input within FP8 E4M3 tolerance.
"""

import pytest
import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _skip_if_not_sm100,
)


def _unswizzle_mxfp8_scales_128x4(sf: torch.Tensor, row: int, col: int) -> torch.Tensor:
    """Invert the 128x4 swizzle to a plain ``[row, col // 32]`` scale grid."""
    scale_vec_size = 32
    factor = scale_vec_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    sf_reshaped = sf.reshape(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzled = sf_reshaped.transpose(1, 3)
    sf_unswizzled = sf_unswizzled.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    return sf_unswizzled[:row, : (col // scale_vec_size)].contiguous()


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
    sf_vec_size = 32

    # Map the API layout back to plain per-group tensors:
    #   out: logical [M, padded_K, B] (fp8)        -> [B, M, K]
    #   sf:  logical [32, 4, rm, 4, rk, B] (uint8) -> per group [rm, rk, 32, 4, 4]
    q = out.permute(2, 0, 1)[:, :, :k].contiguous()
    sf_grouped = sf.permute(5, 2, 4, 0, 1, 3).contiguous()

    # init uses a full mask (mask == M), so every row is valid and comparable.
    for i in range(b):
        scale_bytes = _unswizzle_mxfp8_scales_128x4(sf_grouped[i], m, padded_k)
        scale_bytes = scale_bytes[:, : k // sf_vec_size]
        # E8M0 dequant multiplier 2^(byte - 127), broadcast across each 32-wide block.
        mult = torch.exp2(scale_bytes.to(torch.float32) - 127.0)
        deq = (
            q[i].float().reshape(m, k // sf_vec_size, sf_vec_size) * mult.unsqueeze(-1)
        ).reshape(m, k)
        # Dequantized round-trip reproduces the input within E4M3 tolerance.
        torch.testing.assert_close(deq, a[i].float(), atol=0.1, rtol=0.2)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
