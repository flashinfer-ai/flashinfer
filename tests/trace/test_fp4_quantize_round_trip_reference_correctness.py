"""Reference correctness test for the fp4_quantize_round_trip trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _skip_if_not_sm100,
)


@pytest.mark.parametrize("shape_kwargs", [dict(M=64, K=256), dict(M=17, K=128)])
def test_fp4_quantize_round_trip_reference_correctness(shape_kwargs):
    _skip_if_not_sm100()
    from flashinfer.trace.templates.quantize import fp4_quantize_trace
    from flashinfer.trace.templates.moe import _unpack_fp4_e2m1

    inputs = fp4_quantize_trace.init(**shape_kwargs)
    # FP4 round-trip is tightest in fp32; init builds bf16 by default.
    inputs["input"] = inputs["input"].to(torch.float32)
    _assert_finite(inputs["input"])
    x = inputs["input"]
    # The round-trip dynamic range only behaves cleanly when ``global_scale``
    # is close to 1.0; the init's ``448*6/amax(x)`` form is correct for the
    # kernel pipeline but compresses values into a near-zero range here.
    global_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    packed, scales = fp4_quantize_trace.reference(
        x,
        global_scale=global_scale,
        sf_vec_size=inputs["sf_vec_size"],
        sf_use_ue8m0=False,
    )
    _assert_finite(packed.float(), scales.float())
    assert packed.dtype == torch.uint8
    assert packed.shape == (shape_kwargs["M"], shape_kwargs["K"] // 2)
    # Dequantize and compare: within per-block quantization error.
    unpacked = _unpack_fp4_e2m1(packed)  # [M, K]
    block_size = inputs["sf_vec_size"]
    scale_f = scales.to(torch.float32).repeat_interleave(block_size, dim=-1)
    recon = unpacked * scale_f
    # FP4 relative error is bounded by ~1/6 per block.
    rel_err = ((recon - x).abs() / (x.abs() + 1e-3)).mean().item()
    assert rel_err < 0.5, f"round-trip error too large: {rel_err:.3f}"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
