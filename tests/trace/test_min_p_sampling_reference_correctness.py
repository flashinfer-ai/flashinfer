"""Reference correctness test for the min_p_sampling trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=4, vocab_size=64), dict(batch_size=4, vocab_size=96)],
)
def test_min_p_sampling_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.sampling import min_p_sampling_trace

    inputs = min_p_sampling_trace.init(**shape_kwargs)
    # Peaked distributions — deterministic kernel and argmax reference agree.
    probs = inputs["probs"]
    probs.fill_(1e-6)
    target = torch.tensor([5, 21, 60, 11], dtype=torch.long, device="cuda")
    probs[torch.arange(4), target] = 0.99
    probs = probs / probs.sum(dim=-1, keepdim=True)
    inputs["min_p"] = 0.5
    api_out = flashinfer.min_p_sampling_from_probs(
        probs, inputs["min_p"], deterministic=True
    )
    ref_out = min_p_sampling_trace.reference(probs, inputs["min_p"])
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
