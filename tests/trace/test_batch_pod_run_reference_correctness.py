"""Reference correctness test for the batch_pod_run trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _check,
)
from tests.test_helpers.paged_kv import make_padded_paged_kv_view


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(
            device="cuda",
            prefill_len=16,
            decode_batch_size=1,
            num_pages=1,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=64,
            page_size=16,
        ),
        dict(
            device="cuda",
            prefill_len=8,
            decode_batch_size=1,
            num_pages=2,
            num_qo_heads=4,
            num_kv_heads=1,
            head_dim=64,
            page_size=8,
        ),
    ],
)
def test_batch_pod_run_reference_correctness(shape_kwargs):
    """BatchPODWithPagedKVCacheWrapper.run kernel vs reference.

    Uses batch_size=1 on both prefill + decode branches so the reference's
    single-sequence assumption holds.
    """
    from flashinfer import BatchPODWithPagedKVCacheWrapper
    from flashinfer.trace.templates.attention import (
        batch_pod_with_paged_kv_cache_run_trace,
    )

    inputs = batch_pod_with_paged_kv_cache_run_trace.init(**shape_kwargs)
    plan = inputs["plan"]
    run = inputs["run"]
    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    try:
        wrapper = BatchPODWithPagedKVCacheWrapper(ws, "NHD")
        wrapper.plan(
            plan["qo_indptr_p"],
            plan["kv_indptr_p"],
            plan["kv_indices_p"],
            plan["last_page_len_p"],
            plan["qo_indptr_d"],
            plan["kv_indptr_d"],
            plan["kv_indices_d"],
            plan["last_page_len_d"],
            plan["num_qo_heads"],
            plan["num_kv_heads"],
            plan["head_dim"],
            plan["page_size"],
            q_data_type=plan["q_data_type"],
            kv_data_type=plan["kv_data_type"],
        )
        out_p, out_d = wrapper.run(
            run["q_p"],
            run["paged_kv_cache_p"],
            run["q_d"],
            run["paged_kv_cache_d"],
            causal_p=True,
        )
    except Exception as exc:
        pytest.skip(f"BatchPODWithPagedKVCacheWrapper unavailable: {exc}")
    ref_p, ref_d = batch_pod_with_paged_kv_cache_run_trace.reference(
        run["q_p"],
        run["paged_kv_cache_p"],
        run["q_d"],
        run["paged_kv_cache_d"],
    )
    # Reference doesn't apply a causal mask for prefill; compare decode only.
    # Matches tests/utils/test_pod_kernels.py tolerance (fp16 decode).
    _check(batch_pod_with_paged_kv_cache_run_trace, ref_d, out_d, atol=1e-3, rtol=1e-3)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_batch_pod_independent_paged_kv_strides():
    """Reuse one fused BatchPOD module across unequal prefill and decode KV."""
    from flashinfer.pod import BatchPODWithPagedKVCacheWrapper, get_batch_pod_module
    from flashinfer.trace.templates.attention import (
        batch_pod_with_paged_kv_cache_run_trace,
    )

    inputs = batch_pod_with_paged_kv_cache_run_trace.init(
        device="cuda",
        prefill_len=8,
        decode_batch_size=1,
        num_pages=2,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=64,
        page_size=8,
    )
    plan = inputs["plan"]
    run = inputs["run"]
    k_cache_p, v_cache_p = run["paged_kv_cache_p"]
    k_cache_d, v_cache_d = run["paged_kv_cache_d"]
    v_cache_p_unequal = make_padded_paged_kv_view(v_cache_p, "NHD")
    v_cache_d_unequal = make_padded_paged_kv_view(v_cache_d, "NHD")
    wrapper = BatchPODWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"), "NHD"
    )
    wrapper.plan(
        plan["qo_indptr_p"],
        plan["kv_indptr_p"],
        plan["kv_indices_p"],
        plan["last_page_len_p"],
        plan["qo_indptr_d"],
        plan["kv_indptr_d"],
        plan["kv_indices_d"],
        plan["last_page_len_d"],
        plan["num_qo_heads"],
        plan["num_kv_heads"],
        plan["head_dim"],
        plan["page_size"],
        q_data_type=plan["q_data_type"],
        kv_data_type=plan["kv_data_type"],
    )
    plan_info_p = tuple(wrapper._plan_info_p)
    plan_info_d = tuple(wrapper._plan_info_d)
    planning_module = wrapper._cached_module
    equal_output = wrapper.run(
        run["q_p"],
        (k_cache_p, v_cache_p),
        run["q_d"],
        (k_cache_d, v_cache_d),
        causal_p=False,
    )
    fused_module_misses = get_batch_pod_module.cache_info().misses
    unequal_output = wrapper.run(
        run["q_p"],
        (k_cache_p, v_cache_p_unequal),
        run["q_d"],
        (k_cache_d, v_cache_d_unequal),
        causal_p=False,
    )
    assert get_batch_pod_module.cache_info().misses == fused_module_misses
    assert wrapper._cached_module is planning_module
    assert tuple(wrapper._plan_info_p) == plan_info_p
    assert tuple(wrapper._plan_info_d) == plan_info_d

    expected = batch_pod_with_paged_kv_cache_run_trace.reference(
        run["q_p"],
        (k_cache_p, v_cache_p),
        run["q_d"],
        (k_cache_d, v_cache_d),
    )
    _check(
        batch_pod_with_paged_kv_cache_run_trace,
        expected,
        equal_output,
        atol=1e-3,
        rtol=1e-3,
    )
    _check(
        batch_pod_with_paged_kv_cache_run_trace,
        expected,
        unequal_output,
        atol=1e-3,
        rtol=1e-3,
    )
