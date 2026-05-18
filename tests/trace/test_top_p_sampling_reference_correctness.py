"""Reference correctness test for the top_p_sampling trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=4, vocab_size=128), dict(batch_size=4, vocab_size=96)],
)
def test_top_p_sampling_reference_correctness(shape_kwargs):
    """top_p_sampling_from_probs kernel vs reference on fully-one-hot probs."""
    import flashinfer
    from flashinfer.trace.templates.sampling import top_p_sampling_trace

    inputs = top_p_sampling_trace.init(**shape_kwargs)
    probs = inputs["probs"]
    B, V = probs.shape
    target = torch.tensor([7, 21, 60, 3], dtype=torch.long, device="cuda")
    probs.zero_()
    probs[torch.arange(B), target] = 1.0
    top_p = inputs["top_p"]
    top_p.fill_(0.9)
    api = flashinfer.top_p_sampling_from_probs(probs, top_p, deterministic=True)
    ref = top_p_sampling_trace.reference(probs, top_p)
    _close(api.to(torch.int64), ref, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
