"""Reference correctness test for the apply_llama31_rope_pos_ids trace API."""

import torch

from tests.trace.reference_utils import (
    _ROPE_KWARGS,
    _ROPE_TOL,
    _assert_finite,
    _close,
    _init_filtered,
)


def test_apply_llama31_rope_pos_ids_reference_correctness():
    import flashinfer
    from flashinfer.trace.templates.rope import apply_llama31_rope_pos_ids_trace

    inputs = _init_filtered(apply_llama31_rope_pos_ids_trace, **_ROPE_KWARGS)
    _assert_finite(inputs["q"], inputs["k"])
    q_api, k_api = flashinfer.apply_llama31_rope_pos_ids(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    q_ref, k_ref = apply_llama31_rope_pos_ids_trace.reference(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
