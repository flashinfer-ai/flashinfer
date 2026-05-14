"""Reference correctness test for the layernorm trace API."""

import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


def test_layernorm_reference_correctness():
    """flashinfer.layernorm kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.norm import layernorm_trace

    inputs = layernorm_trace.init(batch_size=8, hidden_size=256)
    _assert_finite(inputs["input"], inputs["gemma"], inputs["beta"])
    api = flashinfer.layernorm(
        inputs["input"], inputs["gemma"], inputs["beta"], eps=1e-6
    )
    ref = layernorm_trace.reference(inputs["input"], inputs["gemma"], inputs["beta"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
