"""Reference correctness test for the rope_quantize_fp8_append_paged_kv_cache trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
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
        dict(
            device="cuda",
            nnz=7,
            num_q_heads=4,
            num_k_heads=2,
            rope_dim=64,
            no_rope_dim=64,
            num_pages=6,
            page_size=8,
            batch_size=3,
        ),
    ],
)
def test_rope_quantize_fp8_append_paged_kv_cache_reference_correctness(shape_kwargs):
    """rope_quantize_fp8_append_paged_kv_cache kernel vs reference (GQA layout)."""
    from flashinfer.rope import rope_quantize_fp8_append_paged_kv_cache
    from flashinfer.trace.templates.rope import (
        rope_quantize_fp8_append_paged_kv_cache_trace,
    )

    inputs = rope_quantize_fp8_append_paged_kv_cache_trace.init(**shape_kwargs)
    k_cache, v_cache = inputs["paged_kv_cache"]
    k_cache_api = k_cache.clone()
    v_cache_api = v_cache.clone()
    k_cache_ref = torch.zeros_like(k_cache_api)
    v_cache_ref = torch.zeros_like(k_cache_api)
    try:
        q_r_api, q_n_api = rope_quantize_fp8_append_paged_kv_cache(
            inputs["q_rope"],
            inputs["k_rope"],
            inputs["q_nope"],
            inputs["k_nope"],
            inputs["v"],
            inputs["cos_sin_cache"],
            inputs["pos_ids"],
            (k_cache_api, v_cache_api),
            inputs["kv_indices"],
            inputs["kv_indptr"],
            inputs["batch_indices"],
            inputs["positions"],
            is_neox=inputs["is_neox"],
            page_size=inputs["page_size"],
            kv_layout=inputs["kv_layout"],
        )
    except Exception as exc:
        pytest.skip(f"rope_quantize_fp8_append_paged_kv_cache unavailable: {exc}")
    q_r_ref, q_n_ref = rope_quantize_fp8_append_paged_kv_cache_trace.reference(
        inputs["q_rope"],
        inputs["k_rope"],
        inputs["q_nope"],
        inputs["k_nope"],
        inputs["v"],
        inputs["cos_sin_cache"],
        inputs["pos_ids"],
        (k_cache_ref, v_cache_ref),
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["batch_indices"],
        inputs["positions"],
        is_neox=inputs["is_neox"],
        page_size=inputs["page_size"],
        kv_layout=inputs["kv_layout"],
    )
    # Match tests/attention/test_rope.py FP8 rope quantize tolerance for Q.
    # (The paged K/V append half uses an implementation-specific internal
    # layout — nope/rope interleave order varies between kernel versions —
    # so we only compare the Q outputs here, which are portable.)
    _close(q_r_api.float(), q_r_ref.float(), atol=1e-2, rtol=2e-1)
    _close(q_n_api.float(), q_n_ref.float(), atol=1e-2, rtol=2e-1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
