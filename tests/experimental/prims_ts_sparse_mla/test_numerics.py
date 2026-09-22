# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
pytest.importorskip("triton")
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.experimental.prims_ts_sparse_mla.policy import _SparseMlaTuning
from flashinfer.testing.sparse_mla import sparse_mla_reference
from flashinfer.testing.sparse_mla_metadata import prepare_sparse_mla_metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


# Force the two softmax implementations to test the mathematical boundary;
# these are not assertions about automatic dispatch or a tuning sweep.
@pytest.mark.parametrize("family,heads", [("keep", 64), ("2cta", 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_softmax_score_jumps(family, heads, dtype):
    # The first two BK128 tiles initialize both softmax streams at zero.
    # The following two tiles straddle the six-log2 update boundary. This
    # checks both retained-anchor and rescaling paths.
    q = torch.zeros((1, 1, heads, 512), device="cuda", dtype=dtype)
    q[..., 1] = 1
    swa = torch.zeros((1, 256, 512), device="cuda", dtype=q.dtype)
    kv = torch.zeros((384, 1, 512), device="cuda", dtype=q.dtype)
    kv[128:, :, 0] = 1
    si = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, -1)
    ci = torch.arange(384, device="cuda", dtype=torch.int32).view(1, 1, -1)
    sinks = torch.zeros(heads, device="cuda")
    sinks[0], sinks[1] = -torch.inf, torch.inf
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device="cuda")
    w = BatchSparseMLADecodePagedTSWrapper()
    w._impl._tuning = _SparseMlaTuning(
        family=family,
        tile_size_q=heads,
        split_kv=1,
        fuse_epilogue=True,
        direct_inputs=True,
        gather_issue_warps=4 if family == "keep" else 1,
        offset_cache="coalesced" if family == "keep" else "strided",
        reuse_kv=family == "keep",
        reuse_kv_stages=(10 if dtype == torch.float8_e4m3fn else 5)
        if family == "keep"
        else 0,
        defer_max_update=dtype == torch.bfloat16,
    )
    w.plan(
        q.device,
        1,
        heads,
        q_data_type=q.dtype,
        max_topk=128,
        max_extra_topk=384,
        has_sinks=True,
        return_lse=True,
    )
    softmax_scale = 0.6931471805599453
    meta = prepare_sparse_mla_metadata(
        w,
        q,
        swa,
        si,
        extra_kv_cache=kv,
        extra_indices=ci,
        sinks=sinks,
        softmax_scale=softmax_scale,
    )
    w.run(
        q,
        swa,
        meta,
        kv,
        sinks=sinks,
        softmax_scale=softmax_scale,
        out=out,
        lse=lse,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(
            q,
            swa,
            meta,
            kv,
            sinks=sinks,
            softmax_scale=softmax_scale,
            out=out,
            lse=lse,
            validate=False,
        )
    for delta in (5.5, 6.0, 6.5, 12.0):
        kv[128:, :, 1] = delta
        graph.replay()
        ref, ref_lse, bound = sparse_mla_reference(
            q,
            swa,
            kv,
            si,
            ci,
            sinks=sinks,
            softmax_scale=softmax_scale,
            return_fp8_error_bound=True,
        )
        assert torch.isfinite(out).all()
        if dtype == torch.float8_e4m3fn:
            assert ((out.double() - ref).abs() <= bound).all()
        else:
            torch.testing.assert_close(out.double(), ref, atol=8e-4, rtol=0.01)
        torch.testing.assert_close(lse.double(), ref_lse, atol=2e-4, rtol=1e-4)
