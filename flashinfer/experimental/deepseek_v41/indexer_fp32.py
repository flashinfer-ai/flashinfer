# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Paged MXFP4 tensor-core index scores with ordered FP32 head reductions."""

import torch
import triton as tr
import triton.language as tl

from ._formats import _fp4, _scale, _fp4_bits
from ._utils import _overlaps


@tr.jit
def _ordered_head_sum(weighted, BLOCK_N: tl.constexpr):
    # Preserve the previous BF16-MMA scorer's head reduction tree. The
    # tensor-core layout change otherwise changes BF16 boundary decisions.
    # Pair head-index bits in order 3, 4, 2, 1, 0.
    values = tl.trans(weighted).reshape(BLOCK_N, 2, 2, 2, 2, 2)
    values = values.permute(0, 5, 4, 3, 1, 2).reshape(BLOCK_N, 16, 2)
    left, right = tl.split(values)
    values = (left + right).reshape(BLOCK_N, 8, 2)
    left, right = tl.split(values)
    values = (left + right).reshape(BLOCK_N, 4, 2)
    left, right = tl.split(values)
    values = (left + right).reshape(BLOCK_N, 2, 2)
    left, right = tl.split(values)
    left, right = tl.split(left + right)
    return left + right


@tr.jit
def _scores(
    QD,
    QS,
    CACHE,
    W,
    VISIBLE,
    TABLE,
    CANDIDATES,
    OUT,
    PAGE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    PHYSICAL_PAGES: tl.constexpr,
    PAGES: tl.constexpr,
    CONTEXT: tl.constexpr,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    SPARSE: tl.constexpr,
    BLOCK_N: tl.constexpr = 64,
    FP4_BITS: tl.constexpr = False,
    NATIVE_MXFP4: tl.constexpr = False,
    WEIGHTED=None,
):
    batch = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    token = column
    if SPARSE:
        block = tl.load(
            CANDIDATES + batch.to(tl.int64) * (WIDTH // 8) + column // 8,
            column < WIDTH,
            -1,
        )
        token = block * 8 + column % 8
    visible = tl.load(VISIBLE + batch)
    valid = (column < WIDTH) & (token >= 0) & (token < visible) & (token < CONTEXT)
    page = tl.load(TABLE + batch.to(tl.int64) * PAGES + token // PAGE, valid, -1)
    valid = valid & (page >= 0) & (page < PHYSICAL_PAGES)
    if NATIVE_MXFP4:
        # Put token rows on the MMA M axis: the scaled TCGen5 lowering
        # requires M >= 128, whereas the logical query has only 32 heads.
        head = tl.arange(0, 32)
        packed_channel = tl.arange(0, 64)
        scale_channel = tl.arange(0, 4)
        qd = tl.load(
            QD
            + (batch.to(tl.int64) * 32 + head[:, None]) * 64
            + packed_channel[None, :],
            head[:, None] < 32,
            0,
        )
        qs = tl.load(
            QS + (batch.to(tl.int64) * 32 + head[:, None]) * 4 + scale_channel[None, :],
            head[:, None] < 32,
            127,
        )
        base = page.to(tl.int64) * CACHE_STRIDE
        local = token % PAGE
        kd = tl.load(
            CACHE + base[:, None] + local[:, None] * 64 + packed_channel[None, :],
            valid[:, None],
            0,
        )
        ks = tl.load(
            CACHE
            + base[:, None]
            + PAGE * 64
            + local[:, None] * 4
            + scale_channel[None, :],
            valid[:, None],
            127,
        )
        dots = tl.trans(tl.dot_scaled(kd, ks, "e2m1", tl.trans(qd), qs, "e2m1"))
    else:
        head = tl.arange(0, 32)
        dots = _dequant_dot(
            QD,
            QS,
            CACHE,
            batch,
            head,
            page,
            token,
            valid,
            PAGE,
            CACHE_STRIDE,
            BLOCK_N,
            FP4_BITS,
        )
    weights = tl.load(W + batch * 32 + head, head < 32, 0).to(tl.float32)
    weighted = tl.maximum(dots, 0.0) * weights[:, None]
    if NATIVE_MXFP4:
        scores = _ordered_head_sum(weighted, BLOCK_N)
    else:
        scores = tl.sum(weighted, 0)
    if WEIGHTED is not None:
        debug_base = (batch.to(tl.int64) * WIDTH + column) * 33
        tl.store(
            WEIGHTED + debug_base[None, :] + head[:, None],
            weighted,
            column[None, :] < WIDTH,
        )
        tl.store(WEIGHTED + debug_base + 32, scores, column < WIDTH)
    tl.store(
        OUT + batch.to(tl.int64) * STRIDE + column,
        tl.where(valid, scores, -float("inf")),
        column < WIDTH,
    )


@tr.jit
def _dequant_dot(
    QD,
    QS,
    CACHE,
    batch,
    head,
    page,
    token,
    valid,
    PAGE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP4_BITS: tl.constexpr,
):
    channel = tl.arange(0, 128)
    qd = tl.load(
        QD + (batch.to(tl.int64) * 32 + head[:, None]) * 64 + channel[None, :] // 2
    )
    qs = tl.load(
        QS + (batch.to(tl.int64) * 32 + head[:, None]) * 4 + channel[None, :] // 32
    )
    base = page.to(tl.int64) * CACHE_STRIDE
    local = token % PAGE
    kd = tl.load(
        CACHE + base[:, None] + local[:, None] * 64 + channel[None, :] // 2,
        valid[:, None],
        0,
    )
    ks = tl.load(
        CACHE + base[:, None] + PAGE * 64 + local[:, None] * 4 + channel[None, :] // 32,
        valid[:, None],
        127,
    )
    if FP4_BITS:
        query = (_fp4_bits(qd, channel[None, :]) * _scale(qs)).to(tl.bfloat16)
        key = (_fp4_bits(kd, channel[None, :]) * _scale(ks)).to(tl.bfloat16)
    else:
        query = (_fp4(qd, channel[None, :]) * _scale(qs)).to(tl.bfloat16)
        key = (_fp4(kd, channel[None, :]) * _scale(ks)).to(tl.bfloat16)
    return tl.dot(query, tl.trans(key))


def index_scores_fp32(
    q_data,
    q_scales,
    kv_cache,
    weights,
    visible,
    block_table,
    *,
    max_context_len,
    candidates=None,
    out=None,
):
    if q_data.device.type != "cuda" or torch.cuda.get_device_capability(
        q_data.device
    ) not in ((10, 0), (10, 3)):
        raise ValueError("FP32 indexer currently requires SM100/SM103")
    if q_data.ndim != 3 or q_data.shape[1:] != (32, 64) or q_data.shape[0] < 1:
        raise ValueError("Q data requires positive [B,32,64] shape")
    batch = q_data.shape[0]

    def check(value, shape, dtype, name):
        if (
            value.shape != shape
            or value.dtype != dtype
            or value.device != q_data.device
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"{name} shape/dtype/device/contiguous declaration mismatch"
            )

    check(q_data, (batch, 32, 64), torch.uint8, "Q data")
    check(q_scales, (batch, 32, 4), torch.uint8, "Q scales")
    check(weights, (batch, 32), torch.bfloat16, "weights")
    check(visible, (batch,), torch.int32, "visible")
    if block_table.ndim != 2 or block_table.shape[1] < 1:
        raise ValueError("positive [B,pages] block table required")
    check(block_table, (batch, block_table.shape[1]), torch.int32, "block table")
    if (
        kv_cache.ndim != 4
        or kv_cache.shape[0] < 1
        or kv_cache.shape[2:] != (1, 68)
        or kv_cache.shape[1] not in (32, 64, 128)
    ):
        raise ValueError("D128 MXFP4 cache pages required")
    page = kv_cache.shape[1]
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.device != q_data.device
        or kv_cache.stride()[1:] != (68, 68, 1)
        or kv_cache.stride(0) < page * 68
        or kv_cache.stride(0) % 512
    ):
        raise ValueError("index cache requires padded 512-byte-aligned page strides")
    if not isinstance(max_context_len, int) or not 1 <= max_context_len <= min(
        2**31 - 1, block_table.shape[1] * page
    ):
        raise ValueError("positive int32 logical context must fit the block table")
    width = max_context_len
    if candidates is not None:
        if candidates.ndim != 2 or not 1 <= candidates.shape[1] <= 2048:
            raise ValueError("candidate block8 IDs require [B,1..2048]")
        check(candidates, (batch, candidates.shape[1]), torch.int32, "candidates")
        width = candidates.shape[1] * 8
    if out is None:
        out = torch.empty_strided(
            (batch, width),
            (tr.cdiv(width, 512) * 512, 1),
            device=q_data.device,
            dtype=torch.bfloat16,
        )
    if (
        out.shape != (batch, width)
        or out.dtype != torch.bfloat16
        or out.device != q_data.device
        or out.stride(1) != 1
        or out.stride(0) < width
        or out.stride(0) % 512
        or out.data_ptr() % 16
    ):
        raise ValueError(
            "scores need contiguous BF16 columns and 1024-byte aligned rows"
        )
    inputs = (q_data, q_scales, kv_cache, weights, visible, block_table) + (
        () if candidates is None else (candidates,)
    )
    if torch.is_grad_enabled() and any(value.requires_grad for value in (*inputs, out)):
        raise ValueError("FP32 index scores are inference-only")
    if any(_overlaps(out, value) for value in inputs):
        raise ValueError("score output must not overlap inputs")
    block_n = 128 if width <= 2048 else 256
    with torch.cuda.device(q_data.device):
        _scores[(batch, tr.cdiv(width, block_n))](
            q_data,
            q_scales,
            kv_cache,
            weights,
            visible,
            block_table,
            candidates,
            out,
            page,
            kv_cache.stride(0),
            kv_cache.shape[0],
            block_table.shape[1],
            max_context_len,
            width,
            out.stride(0),
            candidates is not None,
            BLOCK_N=block_n,
            NATIVE_MXFP4=True,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return out
