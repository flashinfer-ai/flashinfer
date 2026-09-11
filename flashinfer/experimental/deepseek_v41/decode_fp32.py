# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Mixed paged-cache decode with FP32 probabilities and TF32x3 PV products."""

import torch
import triton as tr
import triton.language as tl

from ._utils import _overlaps
from ._formats import _fp4, _scale, _fp4_bits


@tr.jit
def _partials(
    Q,
    SWA,
    GLOBAL,
    SWA_IDS,
    GLOBAL_IDS,
    PART,
    MAX,
    SUM,
    NQ: tl.constexpr,
    SWA_K: tl.constexpr,
    GLOBAL_K: tl.constexpr,
    SWA_PAGE: tl.constexpr,
    GLOBAL_PAGE: tl.constexpr,
    SWA_ROWS: tl.constexpr,
    GLOBAL_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TCGEN: tl.constexpr = False,
    HEAD_TILE: tl.constexpr = 16,
):
    query, head_group, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head = head_group * HEAD_TILE + tl.arange(0, HEAD_TILE)
    column = tl.arange(0, 512)
    local = tl.arange(0, BLOCK_K)
    if split * BLOCK_K < SWA_K:
        slot = tl.load(SWA_IDS + query.to(tl.int64) * SWA_K + split * BLOCK_K + local)
        valid = (slot >= 0) & (slot < SWA_ROWS)
        page = (slot // SWA_PAGE).to(tl.int64) * SWA_PAGE * 528
        row = slot % SWA_PAGE
        data = tl.load(
            SWA + page[:, None] + row[:, None] * 512 + column[None, :],
            valid[:, None],
            0,
        )
        sf = tl.load(
            SWA
            + page[:, None]
            + SWA_PAGE * 512
            + row[:, None] * 16
            + column[None, :] // 32,
            valid[:, None],
            127,
        )
        kv = data.to(tl.float8e4nv, bitcast=True).to(tl.float32) * _scale(sf)
    else:
        slot = tl.load(
            GLOBAL_IDS + query.to(tl.int64) * GLOBAL_K + split * BLOCK_K - SWA_K + local
        )
        valid = (slot >= 0) & (slot < GLOBAL_ROWS)
        page = (slot // GLOBAL_PAGE).to(tl.int64) * GLOBAL_PAGE * 288
        row = slot % GLOBAL_PAGE
        data = tl.load(
            GLOBAL + page[:, None] + row[:, None] * 256 + column[None, :] // 2,
            valid[:, None],
            0,
        )
        sf = tl.load(
            GLOBAL
            + page[:, None]
            + GLOBAL_PAGE * 256
            + row[:, None] * 32
            + column[None, :] // 16,
            valid[:, None],
            0,
        )
        if TCGEN:
            value = _fp4_bits(data, column[None, :])
        else:
            value = _fp4(data, column[None, :])
        kv = value * sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    kv = kv.to(tl.bfloat16)
    q = tl.load(Q + (query.to(tl.int64) * 64 + head[:, None]) * 512 + column[None, :])
    if TCGEN:
        scores = tl.trans(tl.dot(kv, tl.trans(q))) * (512.0**-0.5)
    else:
        scores = tl.dot(q, tl.trans(kv)) * (512.0**-0.5)
    scores = tl.where(valid[None, :], scores, -float("inf"))
    maximum = tl.max(scores, 1)
    safe_maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    probability = tl.exp(scores - safe_maximum[:, None])
    denominator = tl.sum(probability, 1)
    if TCGEN:
        # Keep FP32 softmax; represent P by three BF16 residual terms.
        # Transposed products put D512 on M, enabling TCGen5 instead of
        # the small-head legacy MMA. Accumulate small terms first.
        high = probability.to(tl.bfloat16)
        residual = probability - high.to(tl.float32)
        middle = residual.to(tl.bfloat16)
        low = (residual - middle.to(tl.float32)).to(tl.bfloat16)
        numerator_t = tl.dot(tl.trans(kv), tl.trans(low))
        numerator_t = tl.dot(tl.trans(kv), tl.trans(middle), numerator_t)
        numerator_t = tl.dot(tl.trans(kv), tl.trans(high), numerator_t)
        numerator = tl.trans(numerator_t)
    else:
        # Preserve FP32 P here; the fast FlashMLA route uses its own MMA recipe.
        numerator = tl.dot(probability, kv.to(tl.float32), input_precision="tf32x3")
    index = (split.to(tl.int64) * NQ + query) * 64 + head
    tl.store(PART + index[:, None] * 512 + column[None, :], numerator)
    tl.store(MAX + index, maximum)
    tl.store(SUM + index, denominator)


@tr.jit
def _merge(
    PART,
    MAX,
    SUM,
    SINK,
    OUT,
    LSE,
    NQ: tl.constexpr,
    SQ: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    query, head = tl.program_id(0), tl.program_id(1)
    split = tl.arange(0, BLOCK_SPLITS)
    column = tl.arange(0, 512)
    index = (split.to(tl.int64) * NQ + query) * 64 + head
    maxima = tl.load(MAX + index, split < SPLITS, -float("inf"))
    denominators = tl.load(SUM + index, split < SPLITS, 0.0)
    partial = tl.load(
        PART + index[:, None] * 512 + column[None, :], split[:, None] < SPLITS, 0.0
    )
    sink = tl.load(SINK + head)
    attention_maximum = tl.max(maxima, 0)
    has_attention = attention_maximum != -float("inf")
    safe_attention_maximum = tl.where(has_attention, attention_maximum, 0.0)
    attention_sum = tl.sum(denominators * tl.exp(maxima - safe_attention_maximum), 0)
    lse = tl.where(
        has_attention, attention_maximum + tl.log(attention_sum), -float("inf")
    )
    # Sink affects the denominator only. Separate normalization keeps LSE
    # correct even when a large sink would underflow all attention weights.
    maximum = tl.maximum(attention_maximum, sink)
    weight = tl.exp(maxima - maximum)
    denominator = tl.sum(denominators * weight, 0) + tl.exp(sink - maximum)
    result = tl.sum(partial * weight[:, None], 0) / denominator
    tl.store(OUT + (query.to(tl.int64) * 64 + head) * 512 + column, result)
    tl.store(LSE + (query // SQ).to(tl.int64) * 64 * SQ + head * SQ + query % SQ, lse)


def _bf16x3_head_tile(nq, global_k):
    # Measured B200 choices for 512 selected global keys. Short selections
    # and SWA-only verification retain the smaller tile.
    if global_k == 512 and nq >= 8:
        return 64
    if global_k == 512 and nq >= 4:
        return 32
    return 16


def decode_fp32(
    q,
    swa_cache,
    global_cache,
    swa_indices,
    global_indices,
    sink,
    *,
    workspace=None,
    out=None,
    lse=None,
    recipe="tf32x3",
    head_tile=16,
):
    if recipe not in ("tf32x3", "bf16x3_tcgen"):
        raise ValueError("unknown decode arithmetic recipe")
    if head_tile not in (None, 16, 32, 64) or (recipe == "tf32x3" and head_tile != 16):
        raise ValueError("invalid head tile for decode recipe")
    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise ValueError("FP32-probability decode currently requires SM100/SM103")
    if (
        q.ndim != 4
        or q.shape[-2:] != (64, 512)
        or min(q.shape[:2]) < 1
        or q.dtype != torch.bfloat16
    ):
        raise ValueError("Q must be BF16 [B,Sq,64,512] with positive B/Sq")
    if (global_cache is None) != (global_indices is None):
        raise ValueError("global cache and indices must both be present or absent")
    inputs = tuple(
        x
        for x in (q, swa_cache, global_cache, swa_indices, global_indices, sink)
        if x is not None
    )
    for tensor in inputs:
        if tensor.device != q.device or not tensor.is_contiguous():
            raise ValueError("all decode inputs must be contiguous on Q's device")
    for cache, width in ((swa_cache, 528), (global_cache, 288)):
        if cache is not None and (
            cache.ndim != 4
            or cache.shape[2:] != (1, width)
            or cache.shape[1] not in (32, 64, 128)
            or cache.dtype != torch.uint8
        ):
            raise ValueError("invalid paged cache declaration")
    for indices, bound in ((swa_indices, 192), (global_indices, 512)):
        if indices is not None and (
            indices.ndim != 3
            or indices.shape[:2] != q.shape[:2]
            or indices.dtype != torch.int32
            or not 64 <= indices.shape[2] <= bound
            or indices.shape[2] % 64
        ):
            raise ValueError(
                "indices require int32 [B,Sq,K], K a padded multiple of64 within SWA192/global512"
            )
    if sink.shape != (64,) or sink.dtype != torch.float32:
        raise ValueError("sink must be FP32 [64]")
    batch, sq = q.shape[:2]
    nq = batch * sq
    swa_k = swa_indices.shape[2]
    global_k = 0 if global_indices is None else global_indices.shape[2]
    if head_tile is None:
        head_tile = _bf16x3_head_tile(nq, global_k)
    block_k = 64 if recipe == "bf16x3_tcgen" else 32
    splits = (swa_k + global_k) // block_k
    declarations = {
        "partial": (splits, nq, 64, 512),
        "max": (splits, nq, 64),
        "sum": (splits, nq, 64),
    }
    if workspace is None:
        workspace = {
            name: torch.empty(shape, device=q.device, dtype=torch.float32)
            for name, shape in declarations.items()
        }
    if not isinstance(workspace, dict) or set(workspace) != set(declarations):
        raise ValueError("workspace must contain partial, max and sum buffers")
    for name, shape in declarations.items():
        value = workspace[name]
        if (
            value.shape != shape
            or value.dtype != torch.float32
            or value.device != q.device
            or not value.is_contiguous()
        ):
            raise ValueError("workspace declaration mismatch")
    if out is None:
        out = torch.empty_like(q)
    if lse is None:
        lse = torch.empty(batch, 64, sq, device=q.device, dtype=torch.float32)
    for tensor, shape, dtype in (
        (out, q.shape, q.dtype),
        (lse, (batch, 64, sq), torch.float32),
    ):
        if (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != q.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("output/LSE declaration mismatch")
    outputs = (out, lse, *workspace.values())
    if torch.is_grad_enabled() and any(x.requires_grad for x in (*inputs, *outputs)):
        raise ValueError("FP32-probability decode is inference-only")
    if any(
        _overlaps(value, source)
        for i, value in enumerate(outputs)
        for source in (*inputs, *outputs[:i])
    ):
        raise ValueError(
            "decode output/workspace may not overlap inputs or one another"
        )
    with torch.cuda.device(q.device):
        _partials[(nq, 64 // head_tile, splits)](
            q,
            swa_cache,
            swa_cache if global_cache is None else global_cache,
            swa_indices,
            swa_indices if global_indices is None else global_indices,
            workspace["partial"],
            workspace["max"],
            workspace["sum"],
            nq,
            swa_k,
            global_k,
            swa_cache.shape[1],
            64 if global_cache is None else global_cache.shape[1],
            swa_cache.shape[0] * swa_cache.shape[1],
            0
            if global_cache is None
            else global_cache.shape[0] * global_cache.shape[1],
            block_k,
            TCGEN=recipe == "bf16x3_tcgen",
            HEAD_TILE=head_tile,
            num_warps=8,
            num_stages=2,
            enable_fp_fusion=False,
        )
        _merge[(nq, 64)](
            workspace["partial"],
            workspace["max"],
            workspace["sum"],
            sink,
            out,
            lse,
            nq,
            sq,
            splits,
            tr.next_power_of_2(splits),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, lse, workspace
