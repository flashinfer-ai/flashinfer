"""Reference correctness test for the silu_and_mul trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(num_tokens=8, hidden_size=2 * 128), dict(num_tokens=5, hidden_size=2 * 96)],
)
def test_silu_and_mul_reference_correctness(shape_kwargs):
    """flashinfer.silu_and_mul kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.activation import silu_and_mul_trace

    # tests/utils/test_activation.py uses fp16; bf16 ULP (3e-2) exceeds 1e-3.
    inputs = silu_and_mul_trace.init(**shape_kwargs)
    inputs["input"] = inputs["input"].to(torch.float16)
    _assert_finite(inputs["input"])
    api = flashinfer.silu_and_mul(inputs["input"])
    ref = silu_and_mul_trace.reference(inputs["input"])
    _assert_finite(api, ref)
    _close(api, ref, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
