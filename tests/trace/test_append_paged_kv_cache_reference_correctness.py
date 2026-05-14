"""Reference correctness test for the append_paged_kv_cache trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _assert_finite,
    _close,
)


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(
            nnz_kv=4,
            batch_size=2,
            num_kv_heads=8,
            head_dim=64,
            page_size=16,
            num_pages=4,
        ),
        dict(
            nnz_kv=7,
            batch_size=3,
            num_kv_heads=2,
            head_dim=128,
            page_size=8,
            num_pages=6,
        ),
    ],
)
def test_append_paged_kv_cache_reference_correctness(shape_kwargs):
    """append_paged_kv_cache kernel vs reference (full cache comparison)."""
    import flashinfer
    from flashinfer.trace.templates.page import append_paged_kv_cache_trace

    inputs = append_paged_kv_cache_trace.init(**shape_kwargs)
    _assert_finite(inputs["append_key"], inputs["append_value"])
    # Make a deep copy of the cache for the reference run so the API and
    # reference each get a clean zero-initialized buffer to mutate.
    k_cache_api, v_cache_api = inputs["paged_kv_cache"]
    k_cache_ref = torch.zeros_like(k_cache_api)
    v_cache_ref = torch.zeros_like(v_cache_api)
    flashinfer.append_paged_kv_cache(
        inputs["append_key"],
        inputs["append_value"],
        inputs["batch_indices"],
        inputs["positions"],
        (k_cache_api, v_cache_api),
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["kv_last_page_len"],
    )
    append_paged_kv_cache_trace.reference(
        inputs["append_key"],
        inputs["append_value"],
        inputs["batch_indices"],
        inputs["positions"],
        (k_cache_ref, v_cache_ref),
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["kv_last_page_len"],
    )
    _assert_finite(k_cache_api, v_cache_api, k_cache_ref, v_cache_ref)
    _close(k_cache_api, k_cache_ref, atol=0.0, rtol=0.0)
    _close(v_cache_api, v_cache_ref, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
