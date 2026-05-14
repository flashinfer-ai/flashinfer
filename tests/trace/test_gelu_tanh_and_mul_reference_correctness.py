"""Reference correctness test for the gelu_tanh_and_mul trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(num_tokens=8, hidden_size=2 * 128), dict(num_tokens=5, hidden_size=2 * 96)],
)
def test_gelu_tanh_and_mul_reference_correctness(shape_kwargs):
    """flashinfer.gelu_tanh_and_mul kernel vs reference."""
    import flashinfer
    from flashinfer.trace.templates.activation import gelu_tanh_and_mul_trace

    # tests/utils/test_activation.py uses fp16; bf16 ULP (3e-2) exceeds 1e-3.
    inputs = gelu_tanh_and_mul_trace.init(**shape_kwargs)
    x = inputs["input"].to(torch.float16)
    api = flashinfer.gelu_tanh_and_mul(x)
    ref = gelu_tanh_and_mul_trace.reference(x)
    # Matches tests/utils/test_activation.py.
    _close(api, ref, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
