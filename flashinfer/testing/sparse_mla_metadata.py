# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 by FlashInfer team.
"""External sparse MLA preparation for tests and prepared-input benchmarks.

Index mapping/compaction is adapted from vLLM sparse_utils.py; SWA and HCA
generation follow sparse_swa.py and deepseek_v4/sparse_mla.py at commit
0aee727ff6131a1b647941186ceef8f4777bdac2. This standalone subset supports
single-device causal attention without vLLM's engine/warmup dependencies.
It transforms integer metadata only, leaving native Q/KV storage unchanged.
"""

import torch
import triton
import triton.language as tl

from flashinfer.attention.prims_ts.sparse_mla_decode import SparseMLAPreparedMetadata


@triton.jit
def _map_indices(
    indices,
    lengths,
    requests,
    table,
    out,
    counts,
    input_stride: tl.constexpr,
    output_stride: tl.constexpr,
    K: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    HAS_TABLE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    length = tl.load(lengths + row)
    token = tl.load(indices + row * input_stride + col, col < K, -1)
    valid = (col < K) & (col < length) & (token >= 0)
    page = token // PAGE
    offset = token % PAGE
    if HAS_TABLE:
        request = tl.load(requests + row)
        valid &= (page >= 0) & (page < TABLE_WIDTH)
        page = tl.load(table + request * TABLE_STRIDE + page, valid, 0)
        valid &= page >= 0
    storage_row = page * PAGE_ROWS + offset
    # vLLM's single-program compaction preserves order and avoids atomics.
    flag = valid.to(tl.int32)
    destination = tl.cumsum(flag) - flag
    tl.store(out + row * output_stride + destination, storage_row, valid)
    count = tl.sum(flag)
    tl.store(counts + row, count)
    # Initialize the whole tail in the same program; no stale graph indices.
    tl.store(out + row * output_stride + col, -1, (col >= count) & (col < K))


def map_sparse_indices(
    indices,
    lengths,
    *,
    page_size,
    page_stride_rows=None,
    token_to_request=None,
    block_table=None,
    out=None,
    counts=None,
):
    """Map page-slot indices to storage rows and compact each valid prefix.

    With block_table, indices are request-local logical tokens. Without it,
    they already encode physical page*page_size+offset. Return lengths count
    valid entries, including duplicate selections; -1 holes are removed.
    """
    rows, width = indices.shape
    if indices.dtype != torch.int32 or lengths.dtype != torch.int32:
        raise ValueError("indices and lengths must be int32")
    if (
        indices.stride(-1) != 1
        or lengths.shape != (rows,)
        or not lengths.is_contiguous()
        or lengths.device != indices.device
    ):
        raise ValueError("expected affine index rows and one length per row")
    if page_size < 1 or (page_stride_rows or page_size) < page_size:
        raise ValueError("invalid page size/stride")
    if block_table is not None and (
        token_to_request is None or token_to_request.shape != (rows,)
    ):
        raise ValueError("block-table mapping needs one request id per token")
    if out is None:
        out = torch.empty((rows, width), device=indices.device, dtype=torch.int32)
    if counts is None:
        counts = torch.empty(rows, device=indices.device, dtype=torch.int32)
    for value, shape in ((out, (rows, width)), (counts, (rows,))):
        if (
            value.shape != shape
            or value.dtype != torch.int32
            or value.device != indices.device
            or not value.is_contiguous()
        ):
            raise ValueError(
                "mapping outputs must be contiguous int32 with matching extents"
            )
    if block_table is not None and (
        block_table.dtype != torch.int32
        or token_to_request.dtype != torch.int32
        or block_table.device != indices.device
        or token_to_request.device != indices.device
        or block_table.stride(-1) != 1
        or not token_to_request.is_contiguous()
    ):
        raise ValueError("invalid block table or token-to-request mapping")
    if rows:
        _map_indices[(rows,)](
            indices,
            lengths,
            token_to_request,
            block_table,
            out,
            counts,
            indices.stride(0),
            out.stride(0),
            width,
            page_size,
            page_stride_rows or page_size,
            0 if block_table is None else block_table.stride(0),
            0 if block_table is None else block_table.shape[1],
            block_table is not None,
            triton.next_power_of_2(max(width, 1)),
            num_warps=4 if width <= 1024 else 8,
        )
    return out, counts


@triton.jit
def _causal_indices(
    positions,
    requests,
    table,
    out,
    lengths,
    K: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    SWA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pos = tl.load(positions + row)
    req = tl.load(requests + row)
    col = tl.arange(0, BLOCK)
    if SWA:
        start = tl.maximum(pos - K + 1, 0)
        count = tl.minimum(pos + 1, K)
        token = start + col
    else:
        count = tl.minimum((pos + 1) // RATIO, K)
        token = col
    valid = (pos >= 0) & (col < count) & (col < K)
    page = tl.load(table + req * TABLE_STRIDE + token // PAGE, valid, 0)
    value = page * PAGE_ROWS + token % PAGE
    tl.store(out + row * K + col, tl.where(valid, value, -1), col < K)
    tl.store(lengths + row, tl.where(pos >= 0, count, 0))


def prepare_causal_indices(
    positions,
    token_to_request,
    block_table,
    *,
    max_topk,
    page_size,
    page_stride_rows=None,
    swa=False,
    compression_ratio=1,
    out=None,
    lengths=None,
):
    """Generate a causal SWA window or HCA compressed prefix in storage rows.

    Positions are zero-based absolute query positions; -1 pads a graph row.
    For SWA, max_topk is its window. HCA uses floor((position+1)/ratio).
    An application's indexer-selected top-k uses map_sparse_indices instead.
    """
    rows = positions.numel()
    if min(max_topk, page_size, compression_ratio) < 1:
        raise ValueError("capacities, page size and compression ratio must be positive")
    if out is None:
        out = torch.empty((rows, max_topk), device=positions.device, dtype=torch.int32)
    if lengths is None:
        lengths = torch.empty(rows, device=positions.device, dtype=torch.int32)
    if (
        positions.ndim != 1
        or not positions.is_contiguous()
        or token_to_request.shape != positions.shape
        or not token_to_request.is_contiguous()
        or token_to_request.dtype != torch.int32
        or block_table.dtype != torch.int32
        or block_table.stride(-1) != 1
        or min(page_stride_rows or page_size, page_size) != page_size
    ):
        raise ValueError("invalid causal metadata inputs")
    for value, shape in ((out, (rows, max_topk)), (lengths, (rows,))):
        if (
            value.shape != shape
            or value.dtype != torch.int32
            or value.device != positions.device
            or not value.is_contiguous()
        ):
            raise ValueError("invalid preallocated causal metadata outputs")
    if rows:
        _causal_indices[(rows,)](
            positions,
            token_to_request,
            block_table,
            out,
            lengths,
            max_topk,
            page_size,
            page_stride_rows or page_size,
            block_table.stride(0),
            compression_ratio,
            swa,
            triton.next_power_of_2(max_topk),
            num_warps=4,
        )
    return out, lengths


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
    state = wrapper._impl._state
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
    for value, shape, dtype in (
        (out.indices, (rows, ks), torch.int32),
        (out.lengths, (rows,), torch.int32),
        (out.routes, (passes, rows, cap), torch.int32),
        (out.execution_lengths, (passes, rows), torch.int32),
        (out.valid_counts, (passes, rows), torch.int32),
        (out.scale_params, (passes, 2 + state["heads"] + rows), torch.float32),
    ):
        if (
            value is None
            or value.shape != shape
            or value.dtype != dtype
            or value.device != device
            or not value.is_contiguous()
        ):
            raise ValueError("prepared output buffers do not match this invocation")

    def map_pool(cache, idx, lens, width, dst, count, default):
        axis = 2 if cache.ndim == 4 and state["kv_layout"] == "HND" else 1
        page = cache.shape[axis]
        map_sparse_indices(
            idx.view(rows, width),
            default[:rows] if lens is None else lens.view(rows),
            page_size=page,
            page_stride_rows=cache.stride(0) // 512,
            out=dst,
            counts=count,
        )

    map_pool(
        kv_cache, indices, lengths, ks, out.indices, out.lengths, state["default_sl"]
    )
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
            state["default_cl"],
        )
    scalars = [
        wrapper._impl._scalar(v, name, False)
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
