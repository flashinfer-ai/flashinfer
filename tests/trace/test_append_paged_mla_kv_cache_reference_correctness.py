"""Reference correctness test for the append_paged_mla_kv_cache trace API."""

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
            nnz_kv=4,
            batch_size=2,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=16,
            num_pages=4,
        ),
        dict(
            device="cuda",
            nnz_kv=7,
            batch_size=3,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=8,
            num_pages=6,
        ),
    ],
)
def test_append_paged_mla_kv_cache_reference_correctness(shape_kwargs):
    """append_paged_mla_kv_cache kernel vs reference (full cache comparison)."""
    import flashinfer
    from flashinfer.trace.templates.page import append_paged_mla_kv_cache_trace

    inputs = append_paged_mla_kv_cache_trace.init(**shape_kwargs)
    ckv_api = inputs["ckv_cache"]
    kpe_api = inputs["kpe_cache"]
    ckv_ref = torch.zeros_like(ckv_api)
    kpe_ref = torch.zeros_like(kpe_api)
    flashinfer.append_paged_mla_kv_cache(
        inputs["append_ckv"],
        inputs["append_kpe"],
        inputs["batch_indices"],
        inputs["positions"],
        ckv_api,
        kpe_api,
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["kv_last_page_len"],
    )
    append_paged_mla_kv_cache_trace.reference(
        inputs["append_ckv"],
        inputs["append_kpe"],
        inputs["batch_indices"],
        inputs["positions"],
        ckv_ref,
        kpe_ref,
        inputs["kv_indices"],
        inputs["kv_indptr"],
        inputs["kv_last_page_len"],
    )
    _close(ckv_api, ckv_ref, atol=0.0, rtol=0.0)
    _close(kpe_api, kpe_ref, atol=0.0, rtol=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
