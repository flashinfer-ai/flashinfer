"""Reference correctness test for the mxfp8_quantize trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _skip_if_not_sm100,
)


@pytest.mark.parametrize("shape_kwargs", [dict(M=128, K=4096), dict(M=32, K=2048)])
@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_mxfp8_quantize_reference_correctness(shape_kwargs, is_sf_swizzled_layout):
    _skip_if_not_sm100()
    import flashinfer
    from flashinfer.trace.templates.quantize import mxfp8_quantize_trace

    inputs = mxfp8_quantize_trace.init(**shape_kwargs)
    _assert_finite(inputs["input"])
    try:
        q_api, s_api = flashinfer.quantization.fp8_quantization.mxfp8_quantize(
            inputs["input"],
            is_sf_swizzled_layout=is_sf_swizzled_layout,
        )
    except Exception as exc:
        pytest.skip(f"mxfp8_quantize kernel unavailable: {exc}")
    q_ref, s_ref = mxfp8_quantize_trace.reference(
        inputs["input"],
        is_sf_swizzled_layout=is_sf_swizzled_layout,
    )
    assert q_api.shape == q_ref.shape
    assert s_api.shape == s_ref.shape
    torch.testing.assert_close(
        q_api.view(torch.uint8),
        q_ref.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(s_api.view(torch.uint8), s_ref, atol=0, rtol=0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
