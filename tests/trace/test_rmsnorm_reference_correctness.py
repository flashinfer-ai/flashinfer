"""Reference correctness test for the rmsnorm trace API."""

import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


def test_rmsnorm_reference_correctness():
    """flashinfer.rmsnorm kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.norm import rmsnorm_trace

    inputs = rmsnorm_trace.init(batch_size=8, hidden_size=256)
    _assert_finite(inputs["input"], inputs["weight"])
    api = flashinfer.rmsnorm(inputs["input"], inputs["weight"], eps=1e-6)
    ref = rmsnorm_trace.reference(inputs["input"], inputs["weight"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
