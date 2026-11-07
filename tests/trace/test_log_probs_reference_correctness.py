"""Reference correctness test for the log_probs trace API (one id per row and K ids per row)."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=8, vocab_size=128), dict(batch_size=3, vocab_size=257)],
)
def test_log_probs_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.sampling import log_probs_trace

    template = log_probs_trace.templates[0]
    inputs = template.init(**shape_kwargs)
    _assert_finite(inputs["logits"])
    assert log_probs_trace(**inputs) is template
    api_out = flashinfer.log_probs(
        inputs["logits"], inputs["token_ids"], temperature=inputs["temperature"]
    )
    ref_out = template.reference(
        inputs["logits"], inputs["token_ids"], temperature=inputs["temperature"]
    )
    _assert_finite(api_out, ref_out)
    _check(template, ref_out, api_out, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(batch_size=8, vocab_size=128, num_ids=4),
        dict(batch_size=3, vocab_size=257, num_ids=40),
    ],
)
def test_log_probs_k_ids_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.sampling import log_probs_trace

    template = log_probs_trace.templates[1]
    inputs = template.init(**shape_kwargs)
    _assert_finite(inputs["logits"])
    assert log_probs_trace(**inputs) is template
    api_out = flashinfer.log_probs(
        inputs["logits"], inputs["token_ids"], temperature=inputs["temperature"]
    )
    ref_out = template.reference(
        inputs["logits"], inputs["token_ids"], temperature=inputs["temperature"]
    )
    _assert_finite(api_out, ref_out)
    _check(template, ref_out, api_out, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
