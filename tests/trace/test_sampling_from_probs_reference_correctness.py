"""Reference correctness test for the sampling_from_probs trace API."""

import torch

from tests.trace.reference_utils import (
    _close,
)


def test_sampling_from_probs_reference_correctness():
    import flashinfer
    from flashinfer.trace.templates.sampling import sampling_from_probs_trace

    inputs = sampling_from_probs_trace.init(batch_size=4, vocab_size=32)
    # One-hot-like probs — argmax is unambiguous across non-deterministic samplers.
    probs = inputs["probs"]
    probs.zero_()
    probs[torch.arange(4), torch.arange(4) * 7 % 32] = 1.0
    api_out = flashinfer.sampling_from_probs(probs, deterministic=True)
    ref_out = sampling_from_probs_trace.reference(probs)
    _close(api_out.to(torch.int32), ref_out, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
