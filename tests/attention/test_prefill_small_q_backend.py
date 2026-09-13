"""Backend selection for MTP-shaped prefill calls.

fa3 uses a fixed CTA_Q=128 tile, so when each request only has a few query
rows (speculative decoding / MTP verification) most of the tile is padding.
The wrapper must therefore avoid fa3 for small per-request query lengths and
keep it for full prefill shapes.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability

if get_compute_capability(torch.device("cuda"))[0] != 9:
    pytest.skip("fa3 exists only on SM90", allow_module_level=True)

PAGE_SIZE, HQ, HKV, DIM, KV_LEN = 16, 32, 8, 128, 8192


def _plan(q_len, batch):
    pages = KV_LEN // PAGE_SIZE
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.int8, device="cuda")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace, "NHD", backend="auto"
    )
    wrapper.plan(
        qo_indptr=torch.arange(
            0, (batch + 1) * q_len, q_len, dtype=torch.int32, device="cuda"
        ),
        paged_kv_indptr=torch.arange(
            0, (batch + 1) * pages, pages, dtype=torch.int32, device="cuda"
        ),
        paged_kv_indices=torch.arange(pages, dtype=torch.int32, device="cuda").repeat(
            batch
        ),
        paged_kv_last_page_len=torch.full(
            (batch,), PAGE_SIZE, dtype=torch.int32, device="cuda"
        ),
        num_qo_heads=HQ,
        num_kv_heads=HKV,
        head_dim_qk=DIM,
        page_size=PAGE_SIZE,
        pos_encoding_mode="NONE",
        causal=True,
        logits_soft_cap=0.0,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    return wrapper


def test_mtp_shape_avoids_fa3():
    """A few query rows per request must not select the fixed CTA_Q=128 tile."""
    assert _plan(q_len=2, batch=2)._backend == "fa2"


def test_mtp_threshold_boundary():
    """The documented threshold is inclusive."""
    assert _plan(q_len=64, batch=2)._backend == "fa2"


def test_full_prefill_shape_keeps_fa3():
    """Large per-request query lengths keep the fa3 backend."""
    assert _plan(q_len=256, batch=2)._backend == "fa3"
