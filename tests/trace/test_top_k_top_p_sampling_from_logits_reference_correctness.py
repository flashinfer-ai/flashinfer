"""Reference correctness test for the top_k_top_p_sampling_from_logits trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=4, vocab_size=64), dict(batch_size=4, vocab_size=96)],
)
def test_top_k_top_p_sampling_from_logits_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.sampling import (
        top_k_top_p_sampling_from_logits_trace,
    )

    inputs = top_k_top_p_sampling_from_logits_trace.init(**shape_kwargs)
    logits = inputs["logits"]
    logits.fill_(-1e4)
    target = torch.tensor([2, 19, 50, 7], dtype=torch.long, device="cuda")
    logits[torch.arange(4), target] = 10.0
    inputs["top_k"] = 20
    api_out = flashinfer.top_k_top_p_sampling_from_logits(
        logits, inputs["top_k"], inputs["top_p"], deterministic=True
    )
    ref_out = top_k_top_p_sampling_from_logits_trace.reference(
        logits, inputs["top_k"], inputs["top_p"]
    )
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
