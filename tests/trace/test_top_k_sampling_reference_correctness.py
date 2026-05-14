"""Reference correctness test for the top_k_sampling trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [dict(batch_size=4, vocab_size=128), dict(batch_size=4, vocab_size=96)],
)
def test_top_k_sampling_reference_correctness(shape_kwargs):
    """top_k_sampling_from_probs kernel vs reference on fully-one-hot probs.

    With a one-hot distribution both the kernel and multinomial reference
    deterministically emit the peak index, so the comparison is exact.
    """
    import flashinfer
    from flashinfer.trace.templates.sampling import top_k_sampling_trace

    inputs = top_k_sampling_trace.init(**shape_kwargs)
    probs = inputs["probs"]
    B, V = probs.shape
    target = torch.tensor([3, 17, 42, 0], dtype=torch.long, device="cuda")
    probs.zero_()
    probs[torch.arange(B), target] = 1.0
    top_k = inputs["top_k"]
    top_k.fill_(10)
    api = flashinfer.top_k_sampling_from_probs(probs, top_k, deterministic=True)
    ref = top_k_sampling_trace.reference(probs, top_k)
    _close(api.to(torch.int64), ref, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
