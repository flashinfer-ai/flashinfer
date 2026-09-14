# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Incremental CSA2 state update, FP32 pair pooling and BF16 RMS normalization."""

import torch
import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

from ._utils import _overlaps


@tr.jit
def _update(
    current_kv,
    current_score,
    KV_STATE,
    SCORE_STATE,
    WEIGHT,
    STARTS,
    OUT,
    POSITIONS,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(STARTS + row)
    slot = start % 2
    col = tl.arange(0, 512)
    tl.store(KV_STATE + row * 1024 + slot * 512 + col, current_kv)
    tl.store(SCORE_STATE + row * 1024 + slot * 512 + col, current_score)
    tl.store(POSITIONS + row, tl.where(slot == 1, start // 2, -1))
    if slot == 1:
        previous_kv = tl.load(KV_STATE + row * 1024 + col)
        previous_score = tl.load(SCORE_STATE + row * 1024 + col)
        maximum = tl.maximum(previous_score, current_score)
        a, b = (
            libdevice.exp(previous_score - maximum),
            libdevice.exp(current_score - maximum),
        )
        total = a + b
        probability_a, probability_b = tl.div_rn(a, total), tl.div_rn(b, total)
        pooled = (
            (previous_kv * probability_a + current_kv * probability_b)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        reciprocal = tl.rsqrt(tl.sum(pooled * pooled, 0) / 512 + EPS)
        weight = tl.load(WEIGHT + col).to(tl.float32)
        output = (pooled * reciprocal * weight).to(tl.bfloat16)
        tl.store(OUT + row * 512 + col, output)


@tr.jit
def _decode(
    KV, SCORE, KV_STATE, SCORE_STATE, WEIGHT, STARTS, OUT, POSITIONS, EPS: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    current_kv = tl.load(KV + row * 512 + col)
    current_score = tl.load(SCORE + row * 512 + col)
    _update(
        current_kv,
        current_score,
        KV_STATE,
        SCORE_STATE,
        WEIGHT,
        STARTS,
        OUT,
        POSITIONS,
        EPS,
    )


@tr.jit
def _striped_partial_sum(PARTIALS, address, STRIDE: tl.constexpr):
    # The component projection's 8x256 reduction distributes even/odd
    # split-K rows across two warps. Each warp accumulates four rows in
    # ascending order, then their sums are added. Spell out that schedule:
    # a generic 8x512 tl.sum lowers to a different FP32 addition order.
    even = tl.load(PARTIALS + address) + tl.load(PARTIALS + address + 2 * STRIDE)
    even = even + tl.load(PARTIALS + address + 4 * STRIDE)
    even = even + tl.load(PARTIALS + address + 6 * STRIDE)
    odd = tl.load(PARTIALS + address + STRIDE) + tl.load(
        PARTIALS + address + 3 * STRIDE
    )
    odd = odd + tl.load(PARTIALS + address + 5 * STRIDE)
    odd = odd + tl.load(PARTIALS + address + 7 * STRIDE)
    return even + odd


@tr.jit
def _decode_partials(
    PARTIALS,
    KV_STATE,
    SCORE_STATE,
    WEIGHT,
    STARTS,
    OUT,
    POSITIONS,
    B: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, 512)
    address = row * 512 + col
    current_kv = _striped_partial_sum(PARTIALS, address, 2 * B * 512)
    current_score = _striped_partial_sum(PARTIALS, address + B * 512, 2 * B * 512)
    _update(
        current_kv,
        current_score,
        KV_STATE,
        SCORE_STATE,
        WEIGHT,
        STARTS,
        OUT,
        POSITIONS,
        EPS,
    )


def compressor_decode(
    kv,
    score,
    kv_state,
    score_state,
    norm_weight,
    starts,
    *,
    eps=1e-20,
    out=None,
    positions=None,
):
    import math

    if kv.device.type != "cuda" or torch.cuda.get_device_capability(kv.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise ValueError("V4.1 CSA2 decode currently requires SM100/SM103")
    if kv.ndim != 2 or kv.shape[0] < 1 or kv.shape[1] != 512:
        raise ValueError("KV projections must be positive [B,512]")
    batch = kv.shape[0]
    declarations = (
        (kv, (batch, 512), (torch.float32,)),
        (score, (batch, 512), (torch.float32,)),
        (kv_state, (batch, 2, 512), (torch.float32,)),
        (score_state, (batch, 2, 512), (torch.float32,)),
        (norm_weight, (512,), (torch.float32, torch.bfloat16)),
        (starts, (batch,), (torch.int32,)),
    )
    for tensor, shape, dtypes in declarations:
        if (
            tensor.shape != shape
            or tensor.dtype not in dtypes
            or tensor.device != kv.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("CSA2 input/state declaration mismatch")
        if tensor.requires_grad and torch.is_grad_enabled():
            raise ValueError("CSA2 stateful decode is inference-only")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive finite")
    outputs = []
    for tensor, shape, dtype in (
        (out, (batch, 512), torch.bfloat16),
        (positions, (batch,), torch.int32),
    ):
        if tensor is None:
            tensor = torch.empty(shape, device=kv.device, dtype=dtype)
        elif (
            tensor.shape != shape
            or tensor.dtype != dtype
            or tensor.device != kv.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("CSA2 output declaration mismatch")
        outputs.append(tensor)
    destinations = (kv_state, score_state, *outputs)
    for index, destination in enumerate(destinations):
        if any(
            _overlaps(destination, source)
            for source in (kv, score, norm_weight, starts, *destinations[:index])
        ):
            raise ValueError("CSA2 state/output buffers may not overlap other buffers")
    with torch.cuda.device(kv.device):
        _decode[(batch,)](
            kv,
            score,
            kv_state,
            score_state,
            norm_weight,
            starts,
            *outputs,
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return tuple(outputs)
