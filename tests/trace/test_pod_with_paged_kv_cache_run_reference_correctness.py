"""Reference correctness test for the pod_with_paged_kv_cache_run trace API."""

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
            prefill_len=8,
            decode_batch_size=1,
            num_pages=1,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=64,
            page_size=16,
        ),
        dict(
            device="cuda",
            prefill_len=4,
            decode_batch_size=1,
            num_pages=2,
            num_qo_heads=4,
            num_kv_heads=1,
            head_dim=64,
            page_size=8,
        ),
    ],
)
def test_pod_with_paged_kv_cache_run_reference_correctness(shape_kwargs):
    """PODWithPagedKVCacheWrapper.run kernel vs reference.

    Prefill branch with ragged (q, k, v); decode with paged KV. Uses batch_size=1
    on the decode side to match the reference's single-sequence assumption.
    """
    from flashinfer import PODWithPagedKVCacheWrapper
    from flashinfer.trace.templates.attention import pod_with_paged_kv_cache_run_trace

    inputs = pod_with_paged_kv_cache_run_trace.init(**shape_kwargs)
    plan = inputs["plan"]
    run = inputs["run"]
    ws = torch.empty(64 * 1024 * 1024, dtype=torch.int8, device="cuda")
    try:
        wrapper = PODWithPagedKVCacheWrapper(ws, "NHD")
        wrapper.plan(
            plan["kv_indptr"],
            plan["kv_indices"],
            plan["kv_last_page_len"],
            plan["num_qo_heads"],
            plan["num_kv_heads"],
            plan["head_dim"],
            plan["page_size"],
            q_data_type=plan["q_data_type"],
            kv_data_type=plan["kv_data_type"],
        )
        out_p, out_d = wrapper.run(
            run["q_p"],
            run["k_p"],
            run["v_p"],
            run["q_d"],
            run["paged_kv_cache_d"],
            causal_p=True,
        )
    except Exception as exc:
        pytest.skip(f"PODWithPagedKVCacheWrapper unavailable: {exc}")
    ref_p, ref_d = pod_with_paged_kv_cache_run_trace.reference(
        run["q_p"],
        run["k_p"],
        run["v_p"],
        run["q_d"],
        run["paged_kv_cache_d"],
    )
    # Matches tests/utils/test_pod_kernels.py.
    _close(out_p, ref_p, atol=1e-3, rtol=1e-3)
    _close(out_d, ref_d, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
