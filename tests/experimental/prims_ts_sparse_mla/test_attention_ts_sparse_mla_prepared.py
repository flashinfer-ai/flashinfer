# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
pytest.importorskip("triton")
from flashinfer.attention.prims_ts import BatchSparseMLADecodePagedTSWrapper
from flashinfer.experimental.prims_ts_sparse_mla.policy import _SparseMlaTuning
from flashinfer.testing.sparse_mla import sparse_mla_reference
from flashinfer.testing.sparse_mla_metadata import (
    map_sparse_indices,
    prepare_causal_indices,
    prepare_sparse_mla_metadata,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


def test_vllm_mapping_compacts_holes_and_refreshes_tail():
    device = "cuda"
    table = torch.tensor([[2, 0], [1, 3]], device=device, dtype=torch.int32)
    req = torch.tensor([0, 1, 0], device=device, dtype=torch.int32)
    idx = torch.tensor(
        [[0, -1, 5, 7], [6, 3, -1, 0], [0, 1, 2, 3]], device=device, dtype=torch.int32
    )
    lens = torch.tensor([4, 4, 0], device=device, dtype=torch.int32)
    out, counts = map_sparse_indices(
        idx,
        lens,
        page_size=4,
        page_stride_rows=6,
        block_table=table,
        token_to_request=req,
    )
    torch.testing.assert_close(
        out,
        torch.tensor(
            [[12, 1, 3, -1], [20, 9, 6, -1], [-1, -1, -1, -1]],
            device=device,
            dtype=torch.int32,
        ),
    )
    torch.testing.assert_close(
        counts, torch.tensor([3, 3, 0], device=device, dtype=torch.int32)
    )
    lens.zero_()
    map_sparse_indices(
        idx,
        lens,
        page_size=4,
        page_stride_rows=6,
        block_table=table,
        token_to_request=req,
        out=out,
        counts=counts,
    )
    assert (out == -1).all() and (counts == 0).all()


def test_prepared_unfused_and_empty_packed_queries():
    q = torch.zeros(3, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.ones(2, 1, 512, device=q.device, dtype=q.dtype)
    kv[1] *= 2
    idx = torch.tensor([[0, 1, -1]] * 3, device=q.device, dtype=torch.int32)
    offsets = torch.tensor([0, 1, 3], device=q.device, dtype=torch.int32)
    w = BatchSparseMLADecodePagedTSWrapper()
    w._impl._tuning = _SparseMlaTuning(family="swap", tile_size_q=16)
    w.plan(
        q.device, 2, 16, max_topk=3, max_seq_len_q=2, packed_query=True, return_lse=True
    )
    meta = prepare_sparse_mla_metadata(w, q, kv, idx)
    out, lse = w.run(q, kv, meta, qo_indptr=offsets)
    torch.testing.assert_close(out, torch.full_like(out, 1.5))
    torch.testing.assert_close(
        lse, torch.full_like(lse, 2.0).log(), atol=2e-4, rtol=1e-4
    )
    empty = q[:0]
    empty_meta = prepare_sparse_mla_metadata(w, empty, kv, idx[:0])
    result, result_lse = w.run(
        empty, kv, empty_meta, qo_indptr=torch.zeros_like(offsets)
    )
    assert result.shape == (0, 16, 512) and result_lse.shape == (0, 16)


@pytest.mark.parametrize("swa,ratio", [(True, 1), (False, 2), (False, 128)])
def test_vllm_causal_metadata(swa, ratio):
    pos = torch.tensor([0, 7, 15, -1], device="cuda", dtype=torch.int32)
    req = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    table = torch.tensor([[2, 0, 5, 1], [1, 3, 6, 4]], device="cuda", dtype=torch.int32)
    width = 4 if swa else 8
    out, lengths = prepare_causal_indices(
        pos,
        req,
        table,
        max_topk=width,
        page_size=4,
        page_stride_rows=6,
        swa=swa,
        compression_ratio=ratio,
    )
    reference = torch.full_like(out, -1)
    counts = torch.zeros_like(lengths)
    for row, position in enumerate(pos.cpu().tolist()):
        if position < 0:
            continue
        tokens = (
            list(range(max(0, position + 1 - width), position + 1))
            if swa
            else list(range(min((position + 1) // ratio, width)))
        )
        counts[row] = len(tokens)
        for col, token in enumerate(tokens):
            reference[row, col] = table[req[row], token // 4] * 6 + token % 4
    torch.testing.assert_close(out, reference)
    torch.testing.assert_close(lengths, counts)


@pytest.mark.parametrize(
    "dtype,family,heads,direct,splits,extra,independent,kv_tile",
    [
        (torch.bfloat16, "swap", 16, True, 3, True, False, 128),
        (torch.float8_e4m3fn, "keep", 64, True, 1, True, False, 128),
        (torch.bfloat16, "2cta", 128, False, 1, True, False, 128),
        (torch.float8_e4m3fn, "2cta", 128, False, 2, True, True, 128),
        (torch.bfloat16, "keep", 128, True, 1, False, False, 128),
        (torch.float8_e4m3fn, "keep", 128, True, 1, False, False, 128),
        (torch.bfloat16, "keep", 128, True, 3, True, False, 64),
        (torch.float8_e4m3fn, "keep", 64, False, 1, True, True, 128),
    ],
)
def test_prepared_native_graph(
    dtype, family, heads, direct, splits, extra, independent, kv_tile
):
    torch.manual_seed(131)
    q = (torch.randn(2, 3, heads, 512, device="cuda") * 0.15).to(dtype)
    a_storage = (torch.randn(4, 33, 512, device="cuda") * 0.15).to(dtype)
    b_storage = (torch.randn(4, 17, 512, device="cuda") * 0.15).to(dtype)
    a_storage[:, 32] = torch.nan
    b_storage[:, 16] = torch.nan
    a, b = a_storage[:, :32], b_storage[:, :16]
    ka, kb = 129, 65 if extra else 0
    ai = (torch.arange(ka, device="cuda", dtype=torch.int32) % 128).repeat(2, 3, 1)
    bi = (
        (torch.arange(kb, device="cuda", dtype=torch.int32) % 64).repeat(2, 3, 1)
        if extra
        else None
    )
    al = torch.full((2, 3), ka, device="cuda", dtype=torch.int32)
    bl = torch.full((2, 3), kb, device="cuda", dtype=torch.int32) if extra else None
    ai[..., 3::13] = -1
    if extra:
        bi[..., 2::11] = -1
    al[0, 1] = 0
    if extra:
        bl[0, 1] = 0
    sinks = torch.randn(heads, device="cuda")
    sinks[0], sinks[1] = torch.inf, -torch.inf
    kv_scale = torch.tensor(
        0.75 if dtype == torch.float8_e4m3fn else 1.0, device="cuda"
    )
    extra_scale = torch.tensor(1.5, device="cuda") if independent else kv_scale
    q_scale = torch.tensor(1.25 if dtype == torch.float8_e4m3fn else 1.0, device="cuda")
    w = BatchSparseMLADecodePagedTSWrapper()
    w._impl._tuning = _SparseMlaTuning(
        family=family,
        tile_size_q=64 if family == "keep" else heads,
        split_kv=splits,
        fuse_epilogue=True,
        direct_inputs=direct,
        gather_issue_warps=8
        if kv_tile == 64 or (family == "keep" and independent)
        else 4
        if family != "2cta"
        else 1,
        offset_cache="coalesced" if family != "2cta" else "strided",
        scheduler="clc"
        if (family == "2cta" and dtype == torch.bfloat16)
        or independent
        and family == "keep"
        else "nonpersistent",
        reuse_kv=family == "keep",
        reuse_kv_stages=(10 if dtype == torch.float8_e4m3fn or kv_tile == 64 else 5)
        if family == "keep"
        else 0,
        kv_tile_size=kv_tile,
        page_pipeline_stages=2 if kv_tile == 64 else 0,
        uniform_offset_cache=family == "keep" and independent,
        paired_correction=family == "keep" and independent,
    )
    w.plan(
        q.device,
        2,
        heads,
        max_topk=ka,
        max_extra_topk=kb,
        max_seq_len_q=3,
        q_data_type=dtype,
        has_sinks=True,
        return_lse=True,
        assume_valid_prefix=True,
    )
    kwargs = dict(
        q_scale=q_scale, kv_scale=kv_scale, extra_kv_scale=extra_scale, sinks=sinks
    )
    meta = prepare_sparse_mla_metadata(
        w,
        q,
        a,
        ai,
        al,
        extra_kv_cache=b if extra else None,
        extra_indices=bi,
        extra_lengths=bl,
        **kwargs,
    )
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device="cuda")

    def run():
        return w.run(
            q, a, meta, b if extra else None, out=out, lse=lse, validate=False, **kwargs
        )

    def check():
        ref, rlse, bound = sparse_mla_reference(
            q,
            a,
            b if extra else None,
            ai,
            bi,
            swa_topk_lens=al,
            compressed_topk_lens=bl,
            q_scale=q_scale.item(),
            swa_kv_scale=kv_scale.item(),
            compressed_kv_scale=extra_scale.item(),
            sinks=sinks,
            return_fp8_error_bound=True,
        )
        if dtype == torch.float8_e4m3fn:
            assert ((out.double() - ref).abs() <= bound).all()
        else:
            torch.testing.assert_close(out.double(), ref, atol=8e-4, rtol=0.01)
        torch.testing.assert_close(lse.double(), rlse, atol=2e-4, rtol=1e-4)

    run()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare_sparse_mla_metadata(
            w,
            q,
            a,
            ai,
            al,
            extra_kv_cache=b if extra else None,
            extra_indices=bi,
            extra_lengths=bl,
            out=meta,
            **kwargs,
        )
        run()
    ai.copy_(ai.flip(-1))
    al[1, 2] = 0
    if extra:
        bi.copy_(bi.flip(-1))
        bl[1, 2] = 0
    if dtype == torch.float8_e4m3fn:
        kv_scale.fill_(1.5)
        q_scale.fill_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    check()
    assert torch.count_nonzero(out[1, 2]) == 0
    assert torch.isneginf(lse[1, 2]).all()
