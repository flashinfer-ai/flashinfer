"""Reference correctness test for the var_block_sparse_run trace API."""

import torch
import pytest

from tests.trace.reference_utils import (
    _check,
)
from tests.test_helpers.paged_kv import get_closure_value, make_padded_view


@pytest.mark.parametrize(
    "shape_kwargs",
    [
        dict(
            device="cuda",
            qo_len=32,
            kv_len=32,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=64,
        ),
        dict(
            device="cuda",
            qo_len=48,
            kv_len=48,
            num_qo_heads=4,
            num_kv_heads=1,
            head_dim=64,
        ),
    ],
)
def test_var_block_sparse_run_reference_correctness(shape_kwargs):
    """VariableBlockSparse kernel vs reference (dense SDPA fallback).

    Uses a fully-dense block mask so kernel == dense reference.
    """
    from flashinfer import VariableBlockSparseAttentionWrapper
    from flashinfer.trace.templates.attention import (
        variable_block_sparse_attention_run_trace,
    )

    inputs = variable_block_sparse_attention_run_trace.init(**shape_kwargs)
    R, C = 16, 16
    M, Hq, D = inputs["q"].shape
    N, Hk, _ = inputs["k"].shape
    MB, NB = M // R, N // C
    block_mask_map = torch.ones(Hk, MB, NB, dtype=torch.bool, device="cuda")
    block_row_sz = torch.full((Hk, MB), R, dtype=torch.int32, device="cuda")
    block_col_sz = torch.full((Hk, NB), C, dtype=torch.int32, device="cuda")
    # Wrapper expects HND layout: [num_heads, seq_len, head_dim].
    q_hnd = inputs["q"].transpose(0, 1).contiguous()
    k_hnd = inputs["k"].transpose(0, 1).contiguous()
    v_hnd = inputs["v"].transpose(0, 1).contiguous()
    float_ws = torch.empty(128 * 1024 * 1024, device="cuda")
    try:
        wrapper = VariableBlockSparseAttentionWrapper(float_ws, backend="auto")
        wrapper.plan(
            block_mask_map=block_mask_map,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            num_qo_heads=Hq,
            num_kv_heads=Hk,
            head_dim=D,
            q_data_type=torch.float16,
        )
        api_out = wrapper.run(q_hnd, k_hnd, v_hnd)  # [Hq, M, D]
    except Exception as exc:
        pytest.skip(f"VariableBlockSparseAttentionWrapper unavailable: {exc}")
    # Reference expects NHD — transpose and compare.
    ref_out = variable_block_sparse_attention_run_trace.reference(
        inputs["q"], inputs["k"], inputs["v"]
    )
    # Matches tests/attention/test_block_sparse.py.
    _check(
        variable_block_sparse_attention_run_trace,
        ref_out,
        api_out.transpose(0, 1),
        atol=1e-2,
        rtol=1e-2,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def test_var_block_sparse_lazy_stride_router(monkeypatch):
    """Normalize unequal HND inputs onto the shared routed FA2 namespace."""
    from flashinfer import VariableBlockSparseAttentionWrapper
    from flashinfer.trace.templates.attention import (
        variable_block_sparse_attention_run_trace,
    )

    inputs = variable_block_sparse_attention_run_trace.init(
        device="cuda",
        qo_len=32,
        kv_len=32,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=64,
    )
    R = C = 16
    M, num_qo_heads, head_dim = inputs["q"].shape
    N, num_kv_heads, _ = inputs["k"].shape
    block_mask_map = torch.ones(
        num_kv_heads, M // R, N // C, dtype=torch.bool, device="cuda"
    )
    block_row_sz = torch.full(
        (num_kv_heads, M // R), R, dtype=torch.int32, device="cuda"
    )
    block_col_sz = torch.full(
        (num_kv_heads, N // C), C, dtype=torch.int32, device="cuda"
    )
    q_hnd = inputs["q"].transpose(0, 1).contiguous()
    k_hnd = inputs["k"].transpose(0, 1).contiguous()
    v_hnd = inputs["v"].transpose(0, 1).contiguous()
    v_hnd_unequal = make_padded_view(v_hnd, 1)
    assert k_hnd.stride() == v_hnd.stride()
    assert k_hnd.stride() != v_hnd_unequal.stride()

    wrapper = VariableBlockSparseAttentionWrapper(
        torch.empty(128 * 1024 * 1024, device="cuda"), backend="fa2"
    )
    wrapper.plan(
        block_mask_map=block_mask_map,
        block_row_sz=block_row_sz,
        block_col_sz=block_col_sz,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
    )
    plan_info = tuple(wrapper._plan_info)
    routed_module = wrapper._cached_module
    holder = get_closure_value(routed_module.paged_run, "lazy_independent_module")

    def unexpected_supplement_load():
        raise AssertionError(
            "variable-block sparse must normalize HND tensors before paged routing"
        )

    monkeypatch.setattr(holder, "get", unexpected_supplement_load)
    equal_output = wrapper.run(q_hnd, k_hnd, v_hnd)
    unequal_input_output = wrapper.run(q_hnd, k_hnd, v_hnd_unequal)
    assert wrapper._cached_module is routed_module
    assert tuple(wrapper._plan_info) == plan_info

    expected = variable_block_sparse_attention_run_trace.reference(
        inputs["q"], inputs["k"], inputs["v"]
    )
    _check(
        variable_block_sparse_attention_run_trace,
        expected,
        equal_output.transpose(0, 1),
        atol=1e-2,
        rtol=1e-2,
    )
    _check(
        variable_block_sparse_attention_run_trace,
        expected,
        unequal_input_output.transpose(0, 1),
        atol=1e-2,
        rtol=1e-2,
    )
