"""Reference correctness test for the top_k_mask_logits trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _check,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=4, vocab_size=128), dict(batch_size=3, vocab_size=96)],
)
def test_top_k_mask_logits_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.sampling import top_k_mask_logits_trace

    inputs = top_k_mask_logits_trace.init(**shape_kwargs)
    _assert_finite(inputs["logits"])
    api_out = flashinfer.top_k_mask_logits(inputs["logits"], inputs["top_k"])
    ref_out = top_k_mask_logits_trace.reference(inputs["logits"], inputs["top_k"])
    _check(top_k_mask_logits_trace, ref_out, api_out, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
