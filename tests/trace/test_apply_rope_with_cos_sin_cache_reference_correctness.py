"""Reference correctness test for the apply_rope_with_cos_sin_cache trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _ROPE_TOL,
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(
            nnz=16,
            num_q_heads_x_head_size=4 * 64,
            num_k_heads_x_head_size=2 * 64,
            head_size=64,
            max_seq_len=8192,
            rotary_dim=64,
        ),
        dict(
            nnz=12,
            num_q_heads_x_head_size=6 * 64,
            num_k_heads_x_head_size=3 * 64,
            head_size=64,
            max_seq_len=4096,
            rotary_dim=64,
        ),
    ],
)
def test_apply_rope_with_cos_sin_cache_reference_correctness(shape_kwargs):
    import flashinfer
    from flashinfer.trace.templates.rope import apply_rope_with_cos_sin_cache_trace

    inputs = apply_rope_with_cos_sin_cache_trace.init(**shape_kwargs)
    _assert_finite(inputs["query"], inputs["key"], inputs["cos_sin_cache"])
    q_api, k_api = flashinfer.apply_rope_with_cos_sin_cache(
        inputs["positions"],
        inputs["query"],
        inputs["key"],
        inputs["head_size"],
        inputs["cos_sin_cache"],
        is_neox=True,
    )
    q_ref, k_ref = apply_rope_with_cos_sin_cache_trace.reference(
        inputs["positions"],
        inputs["query"],
        inputs["key"],
        inputs["head_size"],
        inputs["cos_sin_cache"],
        is_neox=True,
    )
    _assert_finite(q_api, k_api, q_ref, k_ref)
    _close(q_api, q_ref, **_ROPE_TOL)
    _close(k_api, k_ref, **_ROPE_TOL)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
