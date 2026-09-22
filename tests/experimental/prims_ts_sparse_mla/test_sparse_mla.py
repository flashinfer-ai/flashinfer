# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Public plan/run contracts and representative automatic kernel paths."""

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
pytest.importorskip("triton")
from flashinfer.attention.prims_ts import (
    BatchSparseMLADecodePagedTSWrapper,
    SparseMLAPreparedMetadata,
    batch_sparse_mla_decode_with_paged_kv_cache,
    get_prims_ts_sparse_mla_decode_workspace_size,
)
from flashinfer.experimental.prims_ts_sparse_mla.policy import _SparseMlaTuning
from flashinfer.testing.sparse_mla import sparse_mla_reference
from flashinfer.testing.sparse_mla_metadata import prepare_sparse_mla_metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "batch,queries,heads,topk,extra,packed,independent,prefix",
    [
        pytest.param(1, 3, 8, 129, True, False, False, False, id="small-head-tail"),
        pytest.param(
            2, 2, 24, 513, True, True, False, False, id="packed-partial-heads"
        ),
        pytest.param(
            2, 2, 6, 65, True, True, True, False, id="odd-head-independent-scales"
        ),
        pytest.param(2, 3, 64, 129, False, False, False, False, id="single-source"),
        pytest.param(4, 1, 64, 513, True, False, False, True, id="wide-query"),
        pytest.param(64, 1, 128, 513, True, False, False, True, id="two-cta"),
        pytest.param(1, 0, 64, 513, True, False, False, True, id="multiwave-prefill"),
        pytest.param(0, 4, 128, 2049, True, False, False, True, id="multiwave-decode"),
        pytest.param(
            1, 3, 8, 8193, True, False, False, False, id="long-split-reduction"
        ),
    ],
)
def test_native_graph(
    dtype, batch, queries, heads, topk, extra, packed, independent, prefix
):
    # Scale the two multiwave cases with hardware, without asserting exact
    # tiles, split counts, register budgets or private policy identities.
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    batch = batch or sms
    queries = queries or (sms + 1)
    rows = batch * queries - int(packed)
    shape = (rows, heads, 512) if packed else (batch, queries, heads, 512)
    torch.manual_seed(131)
    q = (torch.randn(shape, device="cuda") * 0.15).to(dtype)
    a_storage = (torch.randn(4, 33, 512, device="cuda") * 0.15).to(dtype)
    b_storage = (torch.randn(7, 17, 512, device="cuda") * 0.15).to(dtype)
    a_storage[:, 32] = torch.nan
    b_storage[:, 16] = torch.nan
    a, b = a_storage[:, :32], b_storage[:, :16]
    ka, kb = (128, topk) if extra else (topk, 0)
    ai = (torch.arange(ka, device="cuda", dtype=torch.int32) % 128).repeat(rows, 1)
    bi = (
        (torch.arange(kb, device="cuda", dtype=torch.int32) % 112).repeat(rows, 1)
        if extra
        else None
    )
    al = torch.full((rows,), ka, device="cuda", dtype=torch.int32)
    bl = torch.full((rows,), kb, device="cuda", dtype=torch.int32) if extra else None
    al[0] = 0
    if extra:
        bl[0] = 0
    sinks = torch.randn(heads, device="cuda")
    sinks[0], sinks[1] = torch.inf, -torch.inf
    kv_scale = torch.tensor(
        0.75 if dtype == torch.float8_e4m3fn else 1.0, device="cuda"
    )
    extra_scale = (
        torch.tensor(1.5 if dtype == torch.float8_e4m3fn else 1.0, device="cuda")
        if independent
        else kv_scale
    )
    q_scale = torch.tensor(1.25 if dtype == torch.float8_e4m3fn else 1.0, device="cuda")
    offsets = (
        torch.tensor([0, 1, 3], device="cuda", dtype=torch.int32) if packed else None
    )
    # Packed cases also cover the rank-four HND pool interface; the oracle
    # receives the same underlying padded storage as rank-three views.
    primary, secondary = (a.unsqueeze(1), b.unsqueeze(1)) if packed else (a, b)
    if not extra:
        primary = a.unsqueeze(2)  # NHD, singleton KV head
    w = BatchSparseMLADecodePagedTSWrapper()
    w.plan(
        q.device,
        batch,
        heads,
        max_topk=ka,
        max_extra_topk=kb,
        max_seq_len_q=queries,
        packed_query=packed,
        q_data_type=dtype,
        kv_layout="HND" if packed else "NHD",
        has_sinks=True,
        return_lse=True,
        assume_valid_prefix=prefix,
    )
    scales = dict(
        q_scale=q_scale, kv_scale=kv_scale, extra_kv_scale=extra_scale, sinks=sinks
    )

    def prepare(out=None):
        return prepare_sparse_mla_metadata(
            w,
            q,
            primary,
            ai,
            al,
            extra_kv_cache=secondary if extra else None,
            extra_indices=bi,
            extra_lengths=bl,
            out=out,
            **scales,
        )

    meta = prepare()
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device=q.device)

    def run(checked=False, target_lse=lse):
        return w.run(
            q,
            primary,
            meta,
            secondary if extra else None,
            qo_indptr=offsets,
            out=out,
            lse=target_lse,
            validate=checked,
            **scales,
        )

    def check(target_lse=lse):
        prefix_shape = q.shape[:-2]
        ref, ref_lse, bound = sparse_mla_reference(
            q,
            a,
            b if extra else None,
            ai.reshape(*prefix_shape, ka),
            bi.reshape(*prefix_shape, kb) if extra else None,
            swa_topk_lens=al.reshape(prefix_shape),
            compressed_topk_lens=bl.reshape(prefix_shape) if extra else None,
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
        torch.testing.assert_close(target_lse.double(), ref_lse, atol=2e-4, rtol=1e-4)

    run(checked=True)
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare(out=meta)
        run()
    # Transition from contiguous bulk loads to irregular gathers, activate an
    # empty row, and empty the last row. Preparation refreshes stable buffers.
    ai.copy_(ai.flip(-1))
    ai[:, 5::17] = -1
    al[0], al[-1] = 1, 0
    if extra:
        bi.copy_(bi.flip(-1))
        bi[:, 3::11] = -1
        bl[0], bl[-1] = 1, 0
    if dtype == torch.float8_e4m3fn:
        kv_scale.fill_(1.25)
        q_scale.fill_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    check()
    assert torch.count_nonzero(out.reshape(rows, heads, 512)[-1]) == 0
    assert torch.isneginf(lse.reshape(rows, heads)[-1]).all()
    if packed:
        unaligned = torch.empty(lse.numel() + 1, device=q.device)[1:].view_as(lse)
        run(target_lse=unaligned)
        check(unaligned)
    if packed and independent and dtype == torch.bfloat16:
        empty_meta = prepare_sparse_mla_metadata(
            w,
            q[:0],
            primary,
            ai[:0],
            al[:0],
            extra_kv_cache=secondary,
            extra_indices=bi[:0],
            extra_lengths=bl[:0],
            **scales,
        )
        empty, empty_lse = w.run(
            q[:0],
            primary,
            empty_meta,
            secondary,
            qo_indptr=torch.zeros_like(offsets),
            **scales,
        )
        assert empty.numel() == empty_lse.numel() == 0


def test_caller_workspace_single_source_and_eager_api():
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.ones(2, 1, 512, device="cuda", dtype=torch.bfloat16)
    kv[1] *= 2
    indices = torch.tensor([[[0, 1, -1]]], device="cuda", dtype=torch.int32)
    args = dict(
        device=q.device, batch_size=1, num_heads=16, max_topk=3, return_lse=True
    )
    size = get_prims_ts_sparse_mla_decode_workspace_size(**args)
    workspace = torch.empty(size, device=q.device, dtype=torch.uint8)
    wrapper = BatchSparseMLADecodePagedTSWrapper(workspace)
    wrapper.plan(**args)
    metadata = prepare_sparse_mla_metadata(wrapper, q, kv, indices)
    result, lse = wrapper.run(q, kv, metadata)
    torch.testing.assert_close(result, torch.full_like(q, 1.5))
    eager, eager_lse = batch_sparse_mla_decode_with_paged_kv_cache(
        q, kv, metadata, return_lse=True
    )
    torch.testing.assert_close(result, eager, atol=0, rtol=0)
    torch.testing.assert_close(lse, eager_lse, atol=0, rtol=0)
    first_lse = lse.clone()
    shorter = prepare_sparse_mla_metadata(
        wrapper, q, kv, indices, torch.ones((1, 1), device=q.device, dtype=torch.int32)
    )
    _, next_lse = wrapper.run(q, kv, shorter)
    torch.testing.assert_close(next_lse, torch.zeros_like(next_lse), atol=0, rtol=0)
    torch.testing.assert_close(lse, first_lse, atol=0, rtol=0)
    short = BatchSparseMLADecodePagedTSWrapper(workspace[:-1])
    with pytest.raises(ValueError, match="workspace"):
        short.plan(**args)
    bad = indices.clone()
    bad[..., 0] = 2
    with pytest.raises(ValueError, match="outside pool"):
        wrapper.run(q, kv, metadata._replace(indices=bad.view(1, 3)))
    with pytest.raises(ValueError, match="positive"):
        wrapper.run(q, kv, metadata, softmax_scale=1e-100)


def test_valid_prefix_eager_contract():
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.ones(2, 1, 512, device=q.device, dtype=q.dtype)
    swa[1] *= 2
    compressed = swa * 10
    si = torch.tensor([[[0, 1, -1]]], device=q.device, dtype=torch.int32)
    ci = torch.tensor([[[1, -1, 0]]], device=q.device, dtype=torch.int32)
    lengths = torch.tensor([[2]], device=q.device, dtype=torch.int32)
    ci[..., 1] = 0
    wrapper = BatchSparseMLADecodePagedTSWrapper()
    wrapper.plan(
        q.device, 1, 16, max_topk=3, max_extra_topk=3, assume_valid_prefix=True
    )
    metadata = prepare_sparse_mla_metadata(
        wrapper,
        q,
        swa,
        si,
        lengths,
        extra_kv_cache=compressed,
        extra_indices=ci,
        extra_lengths=lengths,
    )
    metadata.extra_indices[0, 1] = -1
    kwargs = dict(assume_valid_prefix=True, return_lse=True)
    with pytest.raises(ValueError, match="extra active prefix contains a hole"):
        batch_sparse_mla_decode_with_paged_kv_cache(
            q, swa, metadata, compressed, **kwargs
        )
    metadata.extra_indices[0, 1] = 0
    result, lse = batch_sparse_mla_decode_with_paged_kv_cache(
        q, swa, metadata, compressed, **kwargs
    )
    # Q=0 gives uniform attention over values 1, 2, 20, 10.
    torch.testing.assert_close(result, torch.full_like(result, 8.25))
    torch.testing.assert_close(lse, torch.full_like(lse, 4.0).log())


def test_direct_metadata_without_packed_fields_graph():
    """Direct kernels accept live source lists without preparation scratch."""
    q = torch.zeros(1, 1, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.ones(2, 1, 512, device=q.device, dtype=q.dtype)
    kv[1] *= 3
    meta = SparseMLAPreparedMetadata(
        torch.tensor([[0, -1, 1]], device=q.device, dtype=torch.int32),
        torch.tensor([3], device=q.device, dtype=torch.int32),
    )
    w = BatchSparseMLADecodePagedTSWrapper()
    w._impl._tuning = _SparseMlaTuning(
        family="swap",
        tile_size_q=16,
        split_kv=1,
        head_dim_ctas=1,
        fuse_epilogue=True,
        direct_inputs=True,
    )
    w.plan(q.device, 1, 16, max_topk=3, return_lse=True)
    out, lse = w.run(q, kv, meta)
    torch.testing.assert_close(out, torch.full_like(out, 2.0))
    torch.testing.assert_close(lse, torch.full_like(lse, 2.0).log())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w.run(q, kv, meta, out=out, lse=lse, validate=False)
    meta.indices[0, 2] = -1
    graph.replay()
    torch.testing.assert_close(out, torch.ones_like(out))
    torch.testing.assert_close(lse, torch.zeros_like(lse))
    meta.lengths.zero_()
    graph.replay()
    assert torch.count_nonzero(out) == 0 and torch.isneginf(lse).all()
