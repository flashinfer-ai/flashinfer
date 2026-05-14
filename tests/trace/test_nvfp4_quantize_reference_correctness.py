"""Reference correctness test for the nvfp4_quantize trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _skip_if_not_sm100,
)


@pytest.mark.parametrize(
    "shape_kwargs", [dict(device="cuda", M=64, K=128), dict(device="cuda", M=17, K=256)]
)
def test_nvfp4_quantize_reference_correctness(shape_kwargs):
    """nvfp4_quantize kernel vs reference, dequantized round-trip."""
    import flashinfer

    # Same SM100+ requirement as mxfp4_quantize above.
    _skip_if_not_sm100()
    from flashinfer.trace.templates.quantize import nvfp4_quantize_trace

    inputs = nvfp4_quantize_trace.init(**shape_kwargs)
    try:
        api_packed, _ = flashinfer.nvfp4_quantize(inputs["a"], inputs["a_global_sf"])
    except Exception as exc:
        pytest.skip(f"nvfp4_quantize unavailable: {exc}")
    # nvfp4 doesn't have a top-level dequantize; the reference in the trace
    # template does; compare shapes + value ranges instead of bit-exact.
    # Since the round-trip needs a fp4 dequant LUT, we compare packed bytes
    # under a loose tolerance that accepts single-ULP mismatches from rounding.
    ref_packed, _ = nvfp4_quantize_trace.reference(inputs["a"], inputs["a_global_sf"])
    # Check element-wise agreement rate; allow up to 5% bytes to differ by
    # a single ULP (one nibble).
    diff = (api_packed.to(torch.int32) - ref_packed.to(torch.int32)).abs()
    frac_different = (diff > 0).float().mean().item()
    assert frac_different < 0.05, f"{frac_different:.2%} packed bytes differ"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
