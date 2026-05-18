"""Reference correctness test for the apply_rope_pos_ids trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _ROPE_KWARGS,
    _ROPE_TOL,
    _assert_finite,
    _close,
    _init_filtered,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        _ROPE_KWARGS,
        dict(nnz=12, batch_size=3, num_q_heads=6, num_k_heads=3, head_dim=64),
    ],
)
def test_apply_rope_pos_ids_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_pos_ids_trace

    inputs = _init_filtered(apply_rope_pos_ids_trace, **shape_kwargs)
    _assert_finite(inputs["q"], inputs["k"])
    q_api, k_api = flashinfer.apply_rope_pos_ids(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    q_ref, k_ref = apply_rope_pos_ids_trace.reference(
        inputs["q"], inputs["k"], inputs["pos_ids"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
