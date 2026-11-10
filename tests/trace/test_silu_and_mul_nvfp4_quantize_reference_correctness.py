"""Reference correctness test for the silu_and_mul_nvfp4_quantize trace API."""

import pytest
import torch

from tests.trace.reference_utils import (
    _check,
    _skip_if_not_sm100,
)

_BLOCK_SIZE = 16


def _unswizzle_sf_128x4(
    sf: torch.Tensor, row: int, col: int, block_size: int = _BLOCK_SIZE
) -> torch.Tensor:
    """Convert 128x4-swizzled scales to a linear [row, col // block_size] tensor."""
    factor = block_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    sf_reshaped = sf.reshape(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzled = sf_reshaped.transpose(1, 3).reshape(
        num_m_tiles * 32 * 4, num_k_tiles * 4
    )
    return sf_unswizzled[:row, : col // block_size].contiguous()


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(device="cuda", M=64, K_doubled=256),
        dict(device="cuda", M=17, K_doubled=512),
    ],
)
def test_silu_and_mul_nvfp4_quantize_reference_correctness(shape_kwargs):
    """Compare the default swizzled API output with a dequantized linear reference."""
    import flashinfer
    from flashinfer.cute_dsl import is_cute_dsl_available

    # Skip only unsupported hardware or missing CuTe-DSL.
    _skip_if_not_sm100()
    if not is_cute_dsl_available():
        pytest.skip("CuTe-DSL backend not available")

    from flashinfer.trace.templates.quantize import silu_and_mul_nvfp4_quantize_trace
    from flashinfer.trace.templates.moe import _unpack_fp4_e2m1

    inputs = silu_and_mul_nvfp4_quantize_trace.init(**shape_kwargs)
    # Exercise the default 128x4 layout; runtime errors must fail the test.
    api_packed, api_sf = flashinfer.silu_and_mul_nvfp4_quantize(
        inputs["input"], inputs["global_scale"]
    )
    ref_packed, ref_sf = silu_and_mul_nvfp4_quantize_trace.reference(
        inputs["input"], inputs["global_scale"]
    )

    m = shape_kwargs["M"]
    k = shape_kwargs["K_doubled"] // 2
    gs = inputs["global_scale"].to(torch.float32).reshape(())

    def _dequant(packed: torch.Tensor, sf_linear: torch.Tensor) -> torch.Tensor:
        """Reconstruct the activations from packed FP4 + linear e4m3 uint8 scales."""
        vals = _unpack_fp4_e2m1(packed)  # [m, k] float32
        scale = (
            sf_linear.reshape(m, k // _BLOCK_SIZE)
            .view(torch.float8_e4m3fn)
            .to(torch.float32)
            .repeat_interleave(_BLOCK_SIZE, dim=-1)
        )
        return vals * scale / gs

    # Compare dequantized outputs after unswizzling the API scales.
    api_recon = _dequant(api_packed, _unswizzle_sf_128x4(api_sf, m, k))
    ref_recon = _dequant(ref_packed, ref_sf)
    _check(
        silu_and_mul_nvfp4_quantize_trace,
        ref_recon,
        api_recon,
        rtol=0.3,
        atol=0.1,
        max_mismatch_pct=10.0,
        min_cos_sim=0.99,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
