"""Reference correctness test for the top_k_renorm_probs trace API."""

import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


def test_top_k_renorm_probs_reference_correctness():
    import flashinfer
    from flashinfer.trace.templates.sampling import top_k_renorm_probs_trace

    inputs = top_k_renorm_probs_trace.init(batch_size=4, vocab_size=128)
    _assert_finite(inputs["probs"])
    api_out = flashinfer.top_k_renorm_probs(inputs["probs"], inputs["top_k"])
    ref_out = top_k_renorm_probs_trace.reference(inputs["probs"], inputs["top_k"])
    _assert_finite(api_out, ref_out)
    _close(api_out, ref_out, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
