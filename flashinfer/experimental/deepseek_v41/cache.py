# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Quantize-and-write DS4.1 cache helpers for SM100, SM103 and SM107."""

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


@tr.jit
def _quantize_swa_fp8(X, OUT, SLOTS, N, CAPACITY, HAS_SLOTS: tl.constexpr):
    # Widen before every row/slot multiplication: a valid physical row may be
    # well below INT32_MAX while its input or output byte address exceeds 2 GiB.
    rows = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4).to(tl.int64)
    valid = rows < N
    slot = tl.load(SLOTS + rows, valid, -1) if HAS_SLOTS else rows
    slot = slot.to(tl.int64)
    if HAS_SLOTS:
        valid = valid & (slot >= 0) & (slot < CAPACITY)
    col = tl.arange(0, 512).to(tl.int64)
    x = tl.load(X + rows[:, None] * 512 + col[None, :], rows[:, None] < N, 0).to(
        tl.float32
    )
    # E4M3 stores common cache units without a scale plane. Explicit RTNE after the
    # finite-range clamp preserves signed zero and representable subnormals.
    fp8 = tl.minimum(tl.maximum(x, -448.0), 448.0).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )
    tl.store(OUT + slot[:, None] * 512 + col[None, :], fp8, valid[:, None])


def quantize_cache(x, *, format, page_size=64, out=None, slots=None, kv_layout="NHD"):
    """Write finite contiguous BF16/FP32 ``[N,512]`` rows into a decode pool.

    ``swa_fp8`` clamps to ``[-448,448]`` in cache units and rounds to nearest
    even, preserving signed zero and subnormals, with no scale plane. For
    shared nonunit descales, normalize both writers' inputs upstream.
    ``main_kv_fp4`` and ``swa_mxfp8`` retain their packed payload/scale layouts.
    Default NHD returns ``[pages,64,1,width]``; explicit HND returns
    ``[pages,1,64,width]`` with identical storage bytes.

    ``out`` owns the capacity for scatter updates. Live ``slots`` must be
    unique; negative or out-of-capacity slots are skipped. Unwritten rows,
    including page padding, are untouched (and uninitialized in a new pool).
    Finiteness and slot uniqueness are caller preconditions, not synchronized
    CPU checks. Reusing ``out`` and updating input/slot values in place is
    CUDA Graph compatible.
    """
    if kv_layout not in ("HND", "NHD"):
        raise ValueError("kv_layout must be HND or NHD")
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        raise ValueError(
            "DS4.1 decode cache preparation requires SM100, SM103 or SM107"
        )
    if (
        x.ndim != 2
        or x.shape[1] != 512
        or not x.is_contiguous()
        or x.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("cache quantization requires contiguous BF16/FP32 [N,512]")
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError("cache quantization has no implicit QAT derivative")
    if format not in ("main_kv_fp4", "swa_mxfp8", "swa_fp8"):
        raise ValueError("cache format must be main_kv_fp4, swa_mxfp8 or swa_fp8")
    if page_size != 64:
        raise ValueError("DS4.1 decode cache requires page_size=64")
    n = x.shape[0]
    fp8_swa = format == "swa_fp8"
    width = 512 if fp8_swa else 288 if format == "main_kv_fp4" else 528
    dtype = torch.float8_e4m3fn if fp8_swa else torch.uint8
    tail = (1, 64, width) if kv_layout == "HND" else (64, 1, width)
    if out is None:
        if slots is not None:
            raise ValueError("scatter updates require caller-owned cache capacity")
        out = torch.empty((tr.cdiv(n, 64), *tail), dtype=dtype, device=x.device)
    elif (
        out.ndim != 4
        or out.shape[1:] != tail
        or out.dtype != dtype
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            f"{format} output must be a contiguous {dtype} {kv_layout} decode cache "
            "on the input device"
        )
    if fp8_swa and out.requires_grad and torch.is_grad_enabled():
        raise ValueError("cache quantization has no implicit QAT derivative")
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
            if fp8_swa:
                _quantize_swa_fp8[(tr.cdiv(n, 4),)](
                    x,
                    out,
                    slots,
                    n,
                    out.shape[0] * 64,
                    slots is not None,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
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
