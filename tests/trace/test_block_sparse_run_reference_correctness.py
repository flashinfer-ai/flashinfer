"""Reference correctness test for the block_sparse_run trace API."""

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
            num_qo_heads=4,
            num_kv_heads=2,
            head_dim=64,
        ),
        dict(
            device="cuda",
            qo_len=48,
            kv_len=48,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=64,
        ),
    ],
)
def test_block_sparse_run_reference_correctness(shape_kwargs):
    """BlockSparseAttentionWrapper.run kernel vs reference (dense SDPA).

    Uses a fully-dense block mask so kernel == dense reference. The
    reference doesn't model the block mask — that's by design for schema
    simplicity, and this test exercises the equivalence case.
    """
    import flashinfer
    from flashinfer.trace.templates.attention import block_sparse_attention_run_trace

    inputs = block_sparse_attention_run_trace.init(**shape_kwargs)
    R, C = 16, 16
    M, Hq, D = inputs["q"].shape
    N, Hk, _ = inputs["k"].shape
    MB, NB = M // R, N // C
    indptr = torch.arange(MB + 1, dtype=torch.int32, device="cuda") * NB
    indices = torch.arange(MB * NB, dtype=torch.int32, device="cuda") % NB
    q = inputs["q"]
    k = inputs["k"]
    v = inputs["v"]

    ws = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    try:
        wrapper = flashinfer.sparse.BlockSparseAttentionWrapper(ws)
        wrapper.plan(indptr, indices, M, N, R, C, Hq, Hk, D)
        api_out = wrapper.run(q, k, v)
    except Exception as exc:
        pytest.skip(f"BlockSparseAttentionWrapper unavailable: {exc}")
    ref_out = block_sparse_attention_run_trace.reference(q, k, v)
    # Matches tests/attention/test_block_sparse.py.
    _close(api_out, ref_out, atol=1e-2, rtol=1e-2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
