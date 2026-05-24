"""Reference correctness test for the softmax trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=8, vocab_size=128), dict(batch_size=3, vocab_size=257)],
)
def test_softmax_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.sampling import softmax_trace

    inputs = softmax_trace.init(**shape_kwargs)
    _assert_finite(inputs["logits"])
    api_out = flashinfer.softmax(inputs["logits"], temperature=inputs["temperature"])
    ref_out = softmax_trace.reference(
        inputs["logits"], temperature=inputs["temperature"]
    )
    _assert_finite(api_out, ref_out)
    _close(api_out, ref_out, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
