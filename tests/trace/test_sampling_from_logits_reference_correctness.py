"""Reference correctness test for the sampling_from_logits trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=4, vocab_size=64), dict(batch_size=4, vocab_size=96)],
)
def test_sampling_from_logits_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.sampling import sampling_from_logits_trace

    inputs = sampling_from_logits_trace.init(**shape_kwargs)
    # Near-one-hot logits so both deterministic kernel and argmax reference agree.
    logits = inputs["logits"]
    logits.fill_(-1e4)
    target = torch.tensor([3, 17, 42, 0], dtype=torch.long, device="cuda")
    logits[torch.arange(4), target] = 10.0
    api_out = flashinfer.sampling_from_logits(logits, deterministic=True)
    ref_out = sampling_from_logits_trace.reference(logits)
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
