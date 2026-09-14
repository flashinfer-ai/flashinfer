# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""MXFP4 paged index scores with FP32 head reduction for short contexts."""

import torch
import triton as tr
import triton.language as tl
from ._utils import _overlaps
from ._formats import _fp4, _scale


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
    PAGES: tl.constexpr,
    WIDTH: tl.constexpr,
    STRIDE: tl.constexpr,
    SPARSE: tl.constexpr,
):
    batch, column = tl.program_id(0), tl.program_id(1)
    token = column
    if SPARSE:
        block = tl.load(CANDIDATES + batch * (WIDTH // 8) + column // 8)
        token = block * 8 + column % 8
    visible = tl.load(VISIBLE + batch)
    if token >= 0 and token < visible and token < PAGES * PAGE:
        page = tl.load(TABLE + batch * PAGES + token // PAGE)
        if page >= 0:
            head = tl.arange(0, 32)
            channel = tl.arange(0, 128)
            qd = tl.load(QD + (batch * 32 + head[:, None]) * 64 + channel[None, :] // 2)
            qs = tl.load(QS + (batch * 32 + head[:, None]) * 4 + channel[None, :] // 32)
            base = page.to(tl.int64) * CACHE_STRIDE
            local = token % PAGE
            kd = tl.load(CACHE + base + local * 64 + channel // 2)
            ks = tl.load(CACHE + base + PAGE * 64 + local * 4 + channel // 32)
            query = _fp4(qd, channel[None, :]) * _scale(qs)
            key = _fp4(kd, channel) * _scale(ks)
            dots = tl.sum(query * key[None, :], 1)
            weights = tl.load(W + batch * 32 + head).to(tl.float32)
            score = tl.sum(tl.maximum(dots, 0.0) * weights, 0)
            tl.store(OUT + batch * STRIDE + column, score)
        else:
            tl.store(OUT + batch * STRIDE + column, -float("inf"))
    else:
        tl.store(OUT + batch * STRIDE + column, -float("inf"))


def small_index_scores(
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
        raise ValueError("V4.1 short indexer requires SM100/SM103")
    if q_data.ndim != 3 or q_data.shape[1:] != (32, 64) or q_data.shape[0] <= 0:
        raise ValueError("Q data must have positive shape [B,32,64]")
    batch = q_data.shape[0]

    def check(t, shape, dtype, name):
        if (
            t.shape != shape
            or t.dtype != dtype
            or t.device != q_data.device
            or not t.is_contiguous()
        ):
            raise ValueError(
                f"{name} requires shape {shape}, dtype {dtype}, contiguous layout and Q device"
            )
        if t.requires_grad and torch.is_grad_enabled():
            raise ValueError("indexer inference scores have no attached backward")

    check(q_data, (batch, 32, 64), torch.uint8, "Q data")
    check(q_scales, (batch, 32, 4), torch.uint8, "Q scales")
    check(weights, (batch, 32), torch.bfloat16, "weights")
    check(visible, (batch,), torch.int32, "visible")
    if block_table.ndim != 2 or block_table.shape[1] <= 0:
        raise ValueError("positive [B,pages] block table required")
    check(block_table, (batch, block_table.shape[1]), torch.int32, "block table")
    if (
        kv_cache.ndim != 4
        or kv_cache.shape[2:] != (1, 68)
        or kv_cache.shape[1] not in (32, 64, 128)
    ):
        raise ValueError("D128 MXFP4 paged cache required")
    page = kv_cache.shape[1]
    if (
        kv_cache.dtype != torch.uint8
        or kv_cache.device != q_data.device
        or kv_cache.shape[0] <= 0
        or kv_cache.stride()[1:] != (68, 68, 1)
        or kv_cache.stride(0) < page * 68
        or kv_cache.stride(0) % 512
    ):
        raise ValueError("index cache requires a 512-byte-aligned physical page stride")
    if (
        not isinstance(max_context_len, int)
        or not 1 <= max_context_len <= 128
        or max_context_len > block_table.shape[1] * page
    ):
        raise ValueError(
            "short indexer logical context must be 1..128 and fit block table"
        )
    width = max_context_len
    if candidates is not None:
        if candidates.ndim != 2 or not 1 <= candidates.shape[1] <= 16:
            raise ValueError("candidates must be [B,1..16] block8 IDs")
        check(candidates, (batch, candidates.shape[1]), torch.int32, "candidates")
        width = candidates.shape[1] * 8
    if out is None:
        out = torch.empty_strided(
            (batch, width), (512, 1), device=q_data.device, dtype=torch.bfloat16
        )
    if (
        out.shape != (batch, width)
        or out.dtype != torch.bfloat16
        or out.device != q_data.device
        or out.stride(1) != 1
        or out.stride(0) < width
        or out.stride(0) * 2 % 1024
        or out.data_ptr() % 16
    ):
        raise ValueError(
            "scores require [B,width] BF16, contiguous columns and 1024-byte row alignment"
        )
    inputs = (q_data, q_scales, kv_cache, weights, visible, block_table) + (
        () if candidates is None else (candidates,)
    )
    if any(_overlaps(out, t) for t in inputs):
        raise ValueError("score output must not overlap inputs")
    with torch.cuda.device(q_data.device):
        _scores[(batch, width)](
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
            block_table.shape[1],
            width,
            out.stride(0),
            candidates is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
