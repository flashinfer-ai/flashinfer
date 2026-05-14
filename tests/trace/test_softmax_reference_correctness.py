"""Reference correctness test for the softmax trace API."""

import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


def test_softmax_reference_correctness():
    import flashinfer
    from flashinfer.trace.templates.sampling import softmax_trace

    inputs = softmax_trace.init(batch_size=8, vocab_size=128)
    _assert_finite(inputs["logits"])
    api_out = flashinfer.softmax(inputs["logits"], temperature=inputs["temperature"])
    ref_out = softmax_trace.reference(
        inputs["logits"], temperature=inputs["temperature"]
    )
    _assert_finite(api_out, ref_out)
    _close(api_out, ref_out, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
