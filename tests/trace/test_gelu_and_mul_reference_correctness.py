"""Reference correctness test for the gelu_and_mul trace API."""

import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


def test_gelu_and_mul_reference_correctness():
    """flashinfer.gelu_and_mul kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.activation import gelu_and_mul_trace

    inputs = gelu_and_mul_trace.init(num_tokens=8, hidden_size=2 * 128)
    inputs["input"] = inputs["input"].to(torch.float16)
    _assert_finite(inputs["input"])
    api = flashinfer.gelu_and_mul(inputs["input"])
    ref = gelu_and_mul_trace.reference(inputs["input"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
