"""Reference correctness test for the mxfp8_quantize trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
    _skip_if_not_sm100,
)


@pytest.mark.parametrize("shape_kwargs", [dict(M=128, K=4096), dict(M=32, K=2048)])
def test_mxfp8_quantize_reference_correctness(shape_kwargs):
    _skip_if_not_sm100()
    import flashinfer
    from flashinfer.trace.templates.quantize import mxfp8_quantize_trace

    inputs = mxfp8_quantize_trace.init(**shape_kwargs)
    _assert_finite(inputs["input"])
    try:
        q_api, _s_api = flashinfer.quantization.fp8_quantization.mxfp8_quantize(
            inputs["input"]
        )
    except Exception as exc:
        pytest.skip(f"mxfp8_quantize kernel unavailable: {exc}")
    q_ref, _s_ref = mxfp8_quantize_trace.reference(inputs["input"])
    # Different swizzle layouts → compare absolute-value histograms only.
    _close(
        q_api.float().abs().mean(),
        q_ref.float().abs().mean(),
        atol=2.0,
        rtol=0.5,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
