"""Reference correctness test for the top_p_renorm_probs trace API."""

import torch

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


def test_top_p_renorm_probs_reference_correctness():
    import flashinfer
    from flashinfer.trace.templates.sampling import top_p_renorm_probs_trace

    inputs = top_p_renorm_probs_trace.init(batch_size=4, vocab_size=128)
    _assert_finite(inputs["probs"])
    api_out = flashinfer.top_p_renorm_probs(inputs["probs"], inputs["top_p"])
    ref_out = top_p_renorm_probs_trace.reference(inputs["probs"], inputs["top_p"])
    _assert_finite(api_out, ref_out)
    # Kernel uses AIR top-p (approximate); allow some slack.
    _close(api_out, ref_out, atol=1e-2, rtol=5e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
