"""Reference correctness test for the top_k_mask_logits trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
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
    # Both should produce identical mask patterns; -inf cells compare as nan.
    api_finite = torch.isfinite(api_out)
    ref_finite = torch.isfinite(ref_out)
    assert torch.equal(api_finite, ref_finite), "mask positions differ"
    _close(api_out[api_finite], ref_out[ref_finite], atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
