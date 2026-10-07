# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Minimal quantize-and-write helper for the two SM100 decode cache formats."""

import torch
import triton as tr
import triton.language as tl

from ._utils import _overlaps


@tr.jit
def _power2(amax):
    value = amax * (1.0 / 448.0)
    bits = value.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return (exponent << 23).to(tl.float32, bitcast=True), exponent.to(tl.uint8)


@tr.jit
def _quantize_cache(
    X, OUT, SLOTS, N, CAPACITY, FP4: tl.constexpr, HAS_SLOTS: tl.constexpr
):
    rows = tl.program_id(0) * 4 + tl.arange(0, 4)
    valid = rows < N
    slot = tl.load(SLOTS + rows, valid, -1) if HAS_SLOTS else rows
    # A physical slot can fit int32 while its byte address exceeds 2 GiB.
    slot = slot.to(tl.int64)
    if HAS_SLOTS:
        valid = valid & (slot >= 0) & (slot < CAPACITY)
    rows = rows.to(tl.int64)
    WIDTH: tl.constexpr = 256 if FP4 else 512
    GROUP: tl.constexpr = 16 if FP4 else 32
    SF: tl.constexpr = 512 // GROUP
    base = (slot // 64) * 64 * (WIDTH + SF)
    data_base = base + (slot % 64) * WIDTH
    scale_base = base + 64 * WIDTH + (slot % 64) * SF
    if FP4:
        col = tl.arange(0, 256)
        a = tl.load(
            X + rows[:, None] * 512 + col[None, :] * 2, rows[:, None] < N, 0
        ).to(tl.float32)
        b = tl.load(
            X + rows[:, None] * 512 + col[None, :] * 2 + 1, rows[:, None] < N, 0
        ).to(tl.float32)
        ga, gb = a.reshape(4, 32, 8), b.reshape(4, 32, 8)
        amax = tl.max(tl.maximum(tl.abs(ga), tl.abs(gb)), 2)
        scale8 = (tl.maximum(amax, 6 * 2.0**-9) / 6).to(tl.float8e4nv)
        scale = scale8.to(tl.float32)
        encoded = scale8.to(tl.uint8, bitcast=True)
        # Reciprocal multiplication perturbs E2M1 halfway values with E4M3
        # scales. Preserve round-to-nearest division before native conversion.
        a = tl.div_rn(ga, scale[:, :, None]).reshape(4, 256)
        b = tl.div_rn(gb, scale[:, :, None]).reshape(4, 256)
        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
            constraints="=r,f,f",
            args=[a, b],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        tl.store(
            OUT + data_base[:, None] + col[None, :], packed.to(tl.uint8), valid[:, None]
        )
    else:
        col = tl.arange(0, 512)
        x = tl.load(X + rows[:, None] * 512 + col[None, :], rows[:, None] < N, 0).to(
            tl.float32
        )
        grouped = x.reshape(4, 16, 32)
        scale, encoded = _power2(tl.maximum(tl.max(tl.abs(grouped), 2), 1e-4))
        normalized = (grouped / scale[:, :, None]).reshape(4, 512)
        fp8 = tl.minimum(tl.maximum(normalized, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(
            OUT + data_base[:, None] + col[None, :],
            fp8.to(tl.uint8, bitcast=True),
            valid[:, None],
        )
    groups = tl.arange(0, SF)
    tl.store(OUT + scale_base[:, None] + groups[None, :], encoded, valid[:, None])


def quantize_cache(x, *, format, page_size=64, out=None, slots=None):
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) != (10, 0):
        raise ValueError("DS4.1 decode cache preparation requires SM100")
    if (
        x.ndim != 2
        or x.shape[1] != 512
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("cache quantization requires contiguous BF16/FP32 [N,512]")
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError("cache quantization has no implicit QAT derivative")
    if format not in ("main_kv_fp4", "swa_mxfp8"):
        raise ValueError("cache format must be main_kv_fp4 or swa_mxfp8")
    if page_size != 64:
        raise ValueError("DS4.1 decode cache requires page_size=64")
    n = x.shape[0]
    width = 288 if format == "main_kv_fp4" else 528
    if out is None:
        if slots is not None:
            raise ValueError("scatter updates require caller-owned cache capacity")
        out = torch.empty(
            (tr.cdiv(n, 64), 64, 1, width), dtype=torch.uint8, device=x.device
        )
    elif (
        out.ndim != 4
        or out.shape[1:] != (64, 1, width)
        or out.dtype != torch.uint8
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "output must be contiguous uint8 decode cache on the input device"
        )
    if slots is None and out.shape[0] * 64 < n:
        raise ValueError("cache capacity is smaller than input")
    if slots is not None and (
        slots.dtype != torch.int32
        or slots.shape != (n,)
        or slots.device != x.device
        or not slots.is_contiguous()
    ):
        raise ValueError(
            "slots must be contiguous CUDA int32 with one entry per input row"
        )
    if n:
        if _overlaps(out, x) or (slots is not None and _overlaps(out, slots)):
            raise ValueError("cache output may not overlap input rows or slots")
        with torch.cuda.device(x.device):
            _quantize_cache[(tr.cdiv(n, 4),)](
                x,
                out,
                slots,
                n,
                out.shape[0] * 64,
                format == "main_kv_fp4",
                slots is not None,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out
