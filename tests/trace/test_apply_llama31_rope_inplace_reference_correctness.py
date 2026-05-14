"""Reference correctness test for the apply_llama31_rope_inplace trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _ROPE_KWARGS,
    _ROPE_TOL,
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        _ROPE_KWARGS,
        dict(nnz=12, batch_size=3, num_q_heads=6, num_k_heads=3, head_dim=64),
    ],
)
def test_apply_llama31_rope_inplace_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.rope import apply_llama31_rope_inplace_trace

    inputs = apply_llama31_rope_inplace_trace.init(**shape_kwargs)
    _assert_finite(inputs["q"], inputs["k"])
    q_api = inputs["q"].clone()
    k_api = inputs["k"].clone()
    flashinfer.apply_llama31_rope_inplace(
        q_api, k_api, inputs["indptr"], inputs["offsets"]
    )
    q_ref, k_ref = apply_llama31_rope_inplace_trace.reference(
        inputs["q"], inputs["k"], inputs["indptr"], inputs["offsets"]
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
