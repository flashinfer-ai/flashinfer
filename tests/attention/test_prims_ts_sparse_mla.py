# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 by FlashInfer team.
"""Accuracy and public-contract checks for native Prims-TS sparse MLA.

The test-only preparer adapts vLLM's page-stride mapping and stable compaction
(0aee727ff6131a1b647941186ceef8f4777bdac2); native Q/KV are never repacked.
"""

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")
triton = pytest.importorskip("triton")
import triton.language as tl
from flashinfer.attention.prims_ts import (
    BatchSparseMLADecodePagedTSWrapper,
    SparseMLAPreparedMetadata,
    batch_sparse_mla_decode_with_paged_kv_cache,
    get_prims_ts_sparse_mla_decode_workspace_size,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)


# Reference: FP64 math, explicit masks, and an unchanged FP8 rounding bound.
def _reference(
    query,
    kv_cache,
    extra_kv_cache,
    indices,
    extra_indices=None,
    *,
    lengths=None,
    extra_lengths=None,
    q_scale=1.0,
    kv_scale=1.0,
    extra_kv_scale=1.0,
    output_scale=1.0,
    sinks=None,
    softmax_scale=512**-0.5,
):
    rows, heads = query.numel() // (query.shape[-2] * 512), query.shape[-2]
    q = query.reshape(rows, heads, 512)
    output = torch.empty((rows, heads, 512), device=query.device, dtype=torch.float64)
    lse = torch.empty((rows, heads), device=query.device, dtype=torch.float64)
    bound = torch.empty_like(output)
    for start in range(0, rows, 32):
        end = min(start + 32, rows)
        values, masks = [], []
        for cache, ids, lens, scale in (
            (kv_cache, indices, lengths, kv_scale),
            (extra_kv_cache, extra_indices, extra_lengths, extra_kv_scale),
        ):
            if cache is None:
                continue
            ids = ids.reshape(rows, -1)[start:end]
            valid = ids >= 0
            if lens is not None:
                valid &= (
                    torch.arange(ids.shape[-1], device=query.device)[None, :]
                    < lens.reshape(-1)[start:end, None]
                )
            safe = ids.masked_fill(~valid, 0).long()
            value = (
                cache[safe // cache.shape[1], safe % cache.shape[1]].double() * scale
            )
            values.append(value.masked_fill(~valid[..., None], 0))
            masks.append(valid)
        kv, valid = torch.cat(values, dim=1), torch.cat(masks, dim=1)
        empty = ~valid.any(-1)
        logits = (
            torch.bmm(q[start:end].double() * q_scale, kv.transpose(1, 2))
            * softmax_scale
        )
        logits.masked_fill_(~valid[:, None, :], -torch.inf)
        row_lse = logits.logsumexp(-1)
        normalizer = (
            row_lse
            if sinks is None
            else torch.logaddexp(row_lse, sinks.double()[None, :])
        )
        normalizer = normalizer.masked_fill(empty[:, None], 0)
        weights = (logits - normalizer[..., None]).exp()
        output[start:end] = torch.bmm(weights, kv) * output_scale
        lse[start:end] = row_lse
        absolute = kv.abs()
        sensitivity = torch.bmm(weights, absolute)
        underflow = (logits.amax(-1) - normalizer).exp()[..., None] * absolute.sum(1)[
            :, None, :
        ]
        bound[start:end] = (
            output_scale
            * ((2**-4 + 3 * 2**-8 + 1e-5) * sensitivity + (2**-10 / 448) * underflow)
            + 1e-6
        ).masked_fill(empty[:, None, None], 0)
    return (
        output.reshape(query.shape),
        lse.reshape(query.shape[:-1]),
        bound.reshape(query.shape),
    )


# Preparation: convert physical page slots into compact storage-row lists.
@triton.jit
def _map_indices(
    indices,
    lengths,
    out,
    counts,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    HAS_LENGTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(lengths + row) if HAS_LENGTH else K
    token = tl.load(indices + row * STRIDE + col, col < K, -1)
    valid = (col < K) & (col < length) & (token >= 0)
    destination = tl.cumsum(valid.to(tl.int32)) - valid.to(tl.int32)
    storage_row = token // PAGE * PAGE_ROWS + token % PAGE
    tl.store(out + row * K + destination, storage_row, valid)
    count = tl.sum(valid.to(tl.int32))
    tl.store(counts + row, count)
    tl.store(out + row * K + col, -1, (col >= count) & (col < K))


def map_sparse_indices(
    indices, lengths, *, page_size, page_stride_rows, out=None, counts=None
):
    rows, width = indices.shape
    if out is None:
        out = torch.empty_like(indices, memory_format=torch.contiguous_format)
    if counts is None:
        counts = torch.empty(rows, device=indices.device, dtype=torch.int32)
    if rows:
        _map_indices[(rows,)](
            indices,
            lengths,
            out,
            counts,
            indices.stride(0),
            width,
            page_size,
            page_stride_rows,
            lengths is not None,
            triton.next_power_of_2(width),
            num_warps=4,
        )
    return out, counts


@triton.jit
def _pack_ts_metadata(
    si,
    ci,
    sl,
    cl,
    routes,
    lengths,
    counts,
    scales,
    sinks,
    sm,
    qs,
    ss,
    cs,
    os,
    KS: tl.constexpr,
    KC: tl.constexpr,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    CAP: tl.constexpr,
    INDEPENDENT: tl.constexpr,
    BLOCK: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.program_id(1)
    ls = tl.load(sl + row)
    lc = tl.load(cl + row) if KC else 0
    if INDEPENDENT:
        ls = tl.where(slot == 0, ls, 0)
        lc = tl.where(slot == 1, lc, 0)
    span_s = tl.cdiv(ls, 128) * 128
    span_c = tl.cdiv(lc, 128) * 128
    col = tl.arange(0, BLOCK)
    for start in range(0, CAP, BLOCK):
        p = start + col
        a = tl.load(si + row * KS + p, (p < ls) & (p < CAP), 0x7FFFFFFF)
        local = p - span_s
        if KC:
            b = tl.load(
                ci + row * KC + local,
                (local >= 0) & (local < lc) & (p < CAP),
                0x7FFFFFFF,
            )
        else:
            b = tl.full((BLOCK,), 0x7FFFFFFF, tl.int32)
        b = b | -2147483648
        word = tl.where(p < span_s, a, b)
        tl.store(routes + (slot * ROWS + row) * CAP + p, word, p < CAP)
    tl.store(lengths + slot * ROWS + row, tl.maximum(span_s + span_c, 1))
    tl.store(counts + slot * ROWS + row, ls + lc)
    scale_base = scales + slot * (2 + HEADS + ROWS)
    tl.store(scale_base + 2 + HEADS + row, (ls + lc).to(tl.float32))
    if row == 0:
        kv_scale = tl.load(ss)
        if INDEPENDENT:
            kv_scale = tl.where(slot == 1, tl.load(cs), kv_scale)
        tl.store(scale_base, tl.load(sm) * tl.load(qs) * kv_scale)
        tl.store(scale_base + 1, tl.load(os) * kv_scale)
        h = tl.arange(0, HEAD_BLOCK)
        sink = tl.load(sinks + h, h < HEADS, 0)
        tl.store(scale_base + 2 + h, sink, h < HEADS)


def prepare_sparse_mla_metadata(
    wrapper,
    query,
    kv_cache,
    indices,
    lengths=None,
    *,
    extra_kv_cache=None,
    extra_indices=None,
    extra_lengths=None,
    softmax_scale=512**-0.5,
    q_scale=1.0,
    kv_scale=1.0,
    extra_kv_scale=None,
    output_scale=1.0,
    sinks=None,
    out=None,
):
    """Test adapter producing every representation required by a prepared plan.

    Inputs are physical page-slot indices, as in the original model fixtures.
    All allocations/preparation belong outside measured attention. Supplying
    out reuses addresses and supports preparation in graph-lifetime tests.
    Scalar/sink arguments must match the subsequent run invocation.
    """
    state = wrapper._state
    if state is None:
        raise ValueError("a prepared attention plan is required")
    rows = query.numel() // (state["heads"] * 512)
    ks, kc, cap = state["ks"], state["kc"], state["capacity"]
    extra_kv_scale = kv_scale if extra_kv_scale is None else extra_kv_scale
    shared = extra_kv_scale is kv_scale or (
        not isinstance(kv_scale, torch.Tensor)
        and not isinstance(extra_kv_scale, torch.Tensor)
        and kv_scale == extra_kv_scale
    )
    independent = kc > 0 and not shared
    passes = 2 if independent else 1
    device = query.device

    def empty(shape, dtype=torch.int32):
        return torch.empty(shape, device=device, dtype=dtype)

    if out is None:
        out = SparseMLAPreparedMetadata(
            indices=empty((rows, ks)),
            lengths=empty((rows,)),
            extra_indices=empty((rows, kc)) if kc else None,
            extra_lengths=empty((rows,)) if kc else None,
            routes=empty((passes, rows, cap)),
            execution_lengths=empty((passes, rows)),
            valid_counts=empty((passes, rows)),
            scale_params=empty((passes, 2 + state["heads"] + rows), torch.float32),
        )

    def map_pool(cache, idx, lens, width, dst, count):
        axis = 2 if cache.ndim == 4 and state["kv_layout"] == "HND" else 1
        page = cache.shape[axis]
        map_sparse_indices(
            idx.view(rows, width),
            None if lens is None else lens.view(rows),
            page_size=page,
            page_stride_rows=cache.stride(0) // 512,
            out=dst,
            counts=count,
        )

    map_pool(kv_cache, indices, lengths, ks, out.indices, out.lengths)
    if kc:
        if extra_kv_cache is None or extra_indices is None:
            raise ValueError("the extra pool and indices are required")
        map_pool(
            extra_kv_cache,
            extra_indices,
            extra_lengths,
            kc,
            out.extra_indices,
            out.extra_lengths,
        )
    scalars = [
        wrapper._scalar(v, name, False)
        for v, name in (
            (softmax_scale, "softmax_scale"),
            (q_scale, "q_scale"),
            (kv_scale, "kv_scale"),
            (extra_kv_scale, "extra_kv_scale"),
            (output_scale, "output_scale"),
        )
    ]
    if rows:
        _pack_ts_metadata[(rows, passes)](
            out.indices,
            out.extra_indices,
            out.lengths,
            out.extra_lengths,
            out.routes,
            out.execution_lengths,
            out.valid_counts,
            out.scale_params,
            state["default_sinks"] if sinks is None else sinks,
            *scalars,
            ks,
            kc,
            rows,
            state["heads"],
            cap,
            independent,
            256,
            triton.next_power_of_2(state["heads"]),
            num_warps=4,
        )
    return out


# Accuracy and important public validation contracts.


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
        pytest.param(0, 1, 96, 513, True, False, False, True, id="partial-wide-heads"),
        pytest.param(
            0, 1, 8, 513, True, False, False, True, id="single-stream-short-kv"
        ),
        pytest.param(
            0, 2, 8, 2049, True, False, False, True, id="small-head-long-kv-reuse"
        ),
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
    # Scale throughput cases with hardware, without asserting exact
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
        ref, ref_lse, bound = _reference(
            q,
            a,
            b if extra else None,
            ai.reshape(*prefix_shape, ka),
            bi.reshape(*prefix_shape, kb) if extra else None,
            lengths=al.reshape(prefix_shape),
            extra_lengths=bl.reshape(prefix_shape) if extra else None,
            q_scale=q_scale.item(),
            kv_scale=kv_scale.item(),
            extra_kv_scale=extra_scale.item(),
            sinks=sinks,
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


@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_softmax_score_jumps(heads, dtype):
    batch = torch.cuda.get_device_properties(0).multi_processor_count + 1
    # The first two BK128 tiles initialize both softmax streams at zero.
    # The following two tiles straddle the six-log2 update boundary. This
    # checks both retained-anchor and rescaling paths.
    q = torch.zeros((batch, 1, heads, 512), device="cuda", dtype=dtype)
    q[..., 1] = 1
    swa = torch.zeros((1, 256, 512), device="cuda", dtype=q.dtype)
    kv = torch.zeros((384, 1, 512), device="cuda", dtype=q.dtype)
    kv[128:, :, 0] = 1
    si = (
        torch.arange(128, device="cuda", dtype=torch.int32)
        .view(1, 1, -1)
        .repeat(batch, 1, 1)
    )
    ci = (
        torch.arange(384, device="cuda", dtype=torch.int32)
        .view(1, 1, -1)
        .repeat(batch, 1, 1)
    )
    sinks = torch.zeros(heads, device="cuda")
    sinks[0], sinks[1] = -torch.inf, torch.inf
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(q.shape[:-1], device="cuda")
    w = BatchSparseMLADecodePagedTSWrapper()
    w.plan(
        q.device,
        batch,
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
        ref, ref_lse, bound = _reference(
            q,
            swa,
            kv,
            si,
            ci,
            sinks=sinks,
            softmax_scale=softmax_scale,
        )
        assert torch.isfinite(out).all()
        if dtype == torch.float8_e4m3fn:
            assert ((out.double() - ref).abs() <= bound).all()
        else:
            torch.testing.assert_close(out.double(), ref, atol=8e-4, rtol=0.01)
        torch.testing.assert_close(lse.double(), ref_lse, atol=2e-4, rtol=1e-4)
