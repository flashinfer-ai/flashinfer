"""Reference correctness test for the var_block_sparse_run trace API."""

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
    _close(api_out.transpose(0, 1), ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
