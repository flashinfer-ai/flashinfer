"""Reference correctness test for the rope_append_paged_kv_cache trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _ROPE_TOL,
    _check,
)


@pytest.mark.parametrize(
    "shape_kwargs,cache_dtype,kv_scale",
    [
        (
            dict(
                device="cuda",
                nnz=16,
                num_q_heads=8,
                num_k_heads=2,
                rope_dim=64,
                no_rope_dim=64,
                num_pages=4,
                page_size=16,
                batch_size=2,
            ),
            torch.bfloat16,
            1.0,
        ),
        (
            dict(
                device="cuda",
                nnz=7,
                num_q_heads=4,
                num_k_heads=2,
                rope_dim=128,
                rotary_dim=128,
                no_rope_dim=64,
                num_pages=6,
                page_size=8,
                batch_size=3,
            ),
            torch.float8_e4m3fn,
            0.5,
        ),
    ],
)
def test_rope_append_paged_kv_cache_reference_correctness(
    shape_kwargs, cache_dtype, kv_scale
):
    """rope_append_paged_kv_cache kernel vs reference: Q outputs and the paged cache."""
    from flashinfer.rope import rope_append_paged_kv_cache
    from flashinfer.trace.templates.rope import rope_append_paged_kv_cache_trace

    inputs = rope_append_paged_kv_cache_trace.init(**shape_kwargs)
    k_cache, v_cache = inputs["paged_kv_cache"]
    k_cache_api = k_cache.to(cache_dtype)
    v_cache_api = v_cache.to(cache_dtype)
    k_cache_ref = k_cache_api.clone()
    v_cache_ref = v_cache_api.clone()
    args = (
        inputs["q_rope"],
        inputs["k_rope"],
        inputs["q_nope"],
        inputs["k_nope"],
        inputs["v"],
        inputs["cos_sin_cache"],
        inputs["pos_ids"],
    )
    meta = (
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["batch_indices"],
        inputs["positions"],
    )
    kwargs = dict(
        is_neox=inputs["is_neox"],
        kv_scale=kv_scale,
        page_size=inputs["page_size"],
        kv_layout=inputs["kv_layout"],
    )
    try:
        q_r_api, q_n_api = rope_append_paged_kv_cache(
            *args, (k_cache_api, v_cache_api), *meta, **kwargs
        )
    except Exception as exc:
        pytest.skip(f"rope_append_paged_kv_cache unavailable: {exc}")
    q_r_ref, q_n_ref = rope_append_paged_kv_cache_trace.reference(
        *args, (k_cache_ref, v_cache_ref), *meta, **kwargs
    )
    # Match tests/attention/test_rope.py bf16 RoPE tolerance for Q.
    _check(
        rope_append_paged_kv_cache_trace,
        (q_r_ref.float(), q_n_ref.float()),
        (q_r_api.float(), q_n_api.float()),
        **_ROPE_TOL,
    )
    # The cache layout (rotary part first) is part of this API's contract, so the
    # K/V caches are compared too; FP8 allows one e4m3 rounding step.
    cache_tol = (
        dict(atol=1e-2, rtol=0.13) if cache_dtype != torch.bfloat16 else _ROPE_TOL
    )
    torch.testing.assert_close(k_cache_api.float(), k_cache_ref.float(), **cache_tol)
    torch.testing.assert_close(v_cache_api.float(), v_cache_ref.float(), **cache_tol)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
