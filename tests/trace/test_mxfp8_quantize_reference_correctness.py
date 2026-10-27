"""Reference correctness test for the mxfp8_quantize trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _skip_if_not_sm100,
)
from tests.utils_fp8 import assert_mxfp8_quantize_exact


@pytest.mark.parametrize("shape_kwargs", [dict(M=128, K=4096), dict(M=32, K=2048)])
def test_mxfp8_quantize_reference_correctness(shape_kwargs):
    _skip_if_not_sm100()
    import flashinfer
    from flashinfer.trace.templates.quantize import mxfp8_quantize_trace

    inputs = mxfp8_quantize_trace.init(**shape_kwargs)
    _assert_finite(inputs["input"])
    q_api, s_api = flashinfer.quantization.fp8_quantization.mxfp8_quantize(
        inputs["input"]
    )
    assert_mxfp8_quantize_exact(inputs["input"], q_api, s_api)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
