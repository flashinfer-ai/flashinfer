# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Blackwell JIT quantization using native E2M1 conversion, no global scale."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _power2(amax, inverse_bound: tl.constexpr):
    value = amax * inverse_bound
    bits = value.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    return scale, exponent.to(tl.uint8)


@tr.jit
def _rotate_pair(
    a, b, FREQS, rope_positions, valid, D: tl.constexpr, INVERSE: tl.constexpr = False
):
    pair = tl.arange(0, D // 2)
    rotary_pair = pair - (D - 64) // 2
    address = rope_positions[:, None].to(tl.int64) * 64 + rotary_pair[None, :] * 2
    mask = valid[:, None] & (rotary_pair[None, :] >= 0)
    cosine = tl.load(FREQS + address, mask, 1.0)
    sine = tl.load(FREQS + address + 1, mask, 0.0)
    if INVERSE:
        sine = -sine
    # Match the published adjacent-pair complex rotation and BF16 boundary
    # before either amax or low-precision conversion.
    # Torch's CUDA complex multiplication contracts a*cos - b*sin and
    # b*cos + a*sin in this order. Explicit FMAs preserve BF16 halfway
    # behavior without enabling contraction in the subsequent quantizer.
    real = tl.fma(a, cosine, -(b * sine)).to(tl.bfloat16).to(tl.float32)
    imaginary = tl.fma(b, cosine, a * sine).to(tl.bfloat16).to(tl.float32)
    return tl.where(rotary_pair[None, :] >= 0, real, a), tl.where(
        rotary_pair[None, :] >= 0, imaginary, b
    )


@tr.jit
def _quantize(
    X,
    DATA,
    SCALES,
    N,
    D: tl.constexpr,
    FORMAT: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    PAGED: tl.constexpr = False,
    PAGE: tl.constexpr = 64,
    SLOTS=None,
    HAS_SLOTS: tl.constexpr = False,
    PAGE_STRIDE: tl.constexpr = 0,
    ROPE: tl.constexpr = False,
    FREQS=None,
    ROT_POS=None,
    FREQ_ROWS: tl.constexpr = 0,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    WIDTH: tl.constexpr = D if FORMAT == 2 else D // 2
    SF: tl.constexpr = D // GROUP
    valid = rows < N
    if ROPE:
        rope_positions = tl.load(ROT_POS + rows, valid, -1)
        valid = valid & (rope_positions >= 0) & (rope_positions < FREQ_ROWS)
    if PAGED:
        slot = tl.load(SLOTS + rows, valid, -1) if HAS_SLOTS else rows
        valid = valid & (slot >= 0)
        # Packed byte offsets can exceed int32 even with int32 token IDs.
        slot = slot.to(tl.int64)
        stride: tl.constexpr = PAGE_STRIDE if PAGE_STRIDE else PAGE * (WIDTH + SF)
        base = (slot // PAGE) * stride
        data_base = base + (slot % PAGE) * WIDTH
        scale_base = base + PAGE * WIDTH + (slot % PAGE) * SF
    else:
        data_base, scale_base = rows * WIDTH, rows * SF
    if FORMAT != 2:
        col = tl.arange(0, D // 2)
        a = tl.load(X + rows[:, None] * D + col[None, :] * 2, rows[:, None] < N, 0).to(
            tl.float32
        )
        b = tl.load(
            X + rows[:, None] * D + col[None, :] * 2 + 1, rows[:, None] < N, 0
        ).to(tl.float32)
        if ROPE:
            a, b = _rotate_pair(a, b, FREQS, rope_positions, valid, D)
        ga, gb = (
            a.reshape(ROWS, D // GROUP, GROUP // 2),
            b.reshape(ROWS, D // GROUP, GROUP // 2),
        )
        amax = tl.max(tl.maximum(tl.abs(ga), tl.abs(gb)), 2)
        if FORMAT == 1:
            scale8 = (tl.maximum(amax, 6 * 2.0**-9) / 6).to(tl.float8e4nv)
            scale = scale8.to(tl.float32)
            encoded = scale8.to(tl.uint8, bitcast=True)
        else:
            scale, encoded = _power2(tl.maximum(amax, 6 * 2.0**-126), 1.0 / 6.0)
        # Approximate reciprocal multiplication perturbs exact E2M1 halfway
        # values with E4M3 scales. Preserve round-to-nearest division first.
        a = tl.div_rn(ga, scale[:, :, None]).reshape(ROWS, D // 2)
        b = tl.div_rn(gb, scale[:, :, None]).reshape(ROWS, D // 2)
        packed = tl.inline_asm_elementwise(
            "{ .reg .b8 v; cvt.rn.satfinite.e2m1x2.f32 v, $2, $1; cvt.u32.u8 $0, v; }",
            constraints="=r,f,f",
            args=[a, b],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        tl.store(
            DATA + data_base[:, None] + col[None, :],
            packed.to(tl.uint8),
            valid[:, None],
        )
    else:
        col = tl.arange(0, D)
        if ROPE:
            pair = tl.arange(0, D // 2)
            a = tl.load(
                X + rows[:, None] * D + pair[None, :] * 2, rows[:, None] < N, 0
            ).to(tl.float32)
            b = tl.load(
                X + rows[:, None] * D + pair[None, :] * 2 + 1, rows[:, None] < N, 0
            ).to(tl.float32)
            a, b = _rotate_pair(a, b, FREQS, rope_positions, valid, D)
            x = tl.interleave(a, b)
        else:
            x = tl.load(X + rows[:, None] * D + col[None, :], rows[:, None] < N, 0).to(
                tl.float32
            )
        grouped = x.reshape(ROWS, D // GROUP, GROUP)
        amax = tl.max(tl.abs(grouped), 2)
        scale, encoded = _power2(tl.maximum(amax, 1e-4), 1.0 / 448.0)
        normalized = (grouped / scale[:, :, None]).reshape(ROWS, D)
        fp8 = tl.minimum(tl.maximum(normalized, -448.0), 448.0).to(tl.float8e4nv)
        tl.store(
            DATA + data_base[:, None] + col[None, :],
            fp8.to(tl.uint8, bitcast=True),
            valid[:, None],
        )
    groups = tl.arange(0, D // GROUP)
    tl.store(SCALES + scale_base[:, None] + groups[None, :], encoded, valid[:, None])


@tr.jit
def _pack(
    DATA,
    SCALES,
    OUT,
    SLOTS,
    N,
    PAGE: tl.constexpr,
    WIDTH: tl.constexpr,
    SF: tl.constexpr,
    HAS_SLOTS: tl.constexpr,
):
    row = tl.program_id(0)
    slot = (tl.load(SLOTS + row) if HAS_SLOTS else row).to(tl.int64)
    columns = tl.arange(0, tr.next_power_of_2(WIDTH))
    sf_columns = tl.arange(0, tr.next_power_of_2(SF))
    if slot >= 0:
        page, local = slot // PAGE, slot % PAGE
        base = page * PAGE * (WIDTH + SF)
        value = tl.load(DATA + row * WIDTH + columns, columns < WIDTH, 0)
        scale = tl.load(SCALES + row * SF + sf_columns, sf_columns < SF, 0)
        tl.store(OUT + base + local * WIDTH + columns, value, columns < WIDTH)
        tl.store(
            OUT + base + PAGE * WIDTH + local * SF + sf_columns, scale, sf_columns < SF
        )


def _check(x):
    if x.device.type != "cuda" or torch.cuda.get_device_capability(x.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise ValueError("V4.1 quantization currently requires SM100/SM103")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("contiguous two-dimensional rows required")


def _output(value, shape, device):
    if value is None:
        return torch.empty(shape, dtype=torch.uint8, device=device)
    if (
        tuple(value.shape) != tuple(shape)
        or value.dtype != torch.uint8
        or value.device != device
        or not value.is_contiguous()
    ):
        raise ValueError(
            "output must be contiguous uint8 with matching shape and device"
        )
    return value


def quantize(x, *, format, data=None, scales=None):
    _check(x)
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("BF16 or FP32 inputs required")
    formats = {"mxfp4": (0, 32, 2), "main_kv_fp4": (1, 16, 2), "swa_mxfp8": (2, 32, 1)}
    if format not in formats:
        raise ValueError(f"unsupported V4.1 format: {format}")
    code, group, divisor = formats[format]
    n, d = x.shape
    if d not in (128, 512):
        raise ValueError("initial implementation supports D128 and D512")
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError("quantization has no implicit QAT derivative")
    data = _output(data, (n, d // divisor), x.device)
    scales = _output(scales, (n, d // group), x.device)
    if n:
        with torch.cuda.device(x.device):
            _quantize[(tr.cdiv(n, 4),)](
                x,
                data,
                scales,
                n,
                d,
                code,
                group,
                4,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return data, scales


def pack_cache(data, scales, *, page_size=64, out=None, slots=None):
    _check(data)
    n, width = data.shape
    if data.dtype != torch.uint8 or (width, scales.shape[-1]) not in (
        (512, 16),
        (256, 32),
    ):
        raise ValueError("D512 SWA FP8 or main-KV FP4 byte rows required")
    sf = scales.shape[-1]
    _output(scales, (n, sf), data.device)
    if page_size not in (32, 64, 128):
        raise ValueError("page_size must be 32, 64 or 128")
    if out is None:
        if slots is not None:
            raise ValueError("scatter updates require caller-owned cache capacity")
        out = _output(
            None, (tr.cdiv(n, page_size), page_size, 1, width + sf), data.device
        )
    else:
        if out.ndim != 4 or out.shape[1:] != (page_size, 1, width + sf):
            raise ValueError("invalid paged cache shape")
        _output(out, out.shape, data.device)
        if slots is None and out.shape[0] * page_size < n:
            raise ValueError("cache capacity is smaller than input")
    if slots is not None and (
        slots.dtype != torch.int32
        or slots.shape != (n,)
        or slots.device != data.device
        or not slots.is_contiguous()
    ):
        raise ValueError(
            "slots must be contiguous CUDA int32 with one entry per input row"
        )
    if n:
        with torch.cuda.device(data.device):
            _pack[(n,)](
                data,
                scales,
                out,
                slots,
                n,
                page_size,
                width,
                sf,
                slots is not None,
                num_warps=4,
            )
    return out


def quantize_cache(x, *, format, page_size=64, out=None, slots=None):
    _check(x)
    if x.dtype not in (torch.bfloat16, torch.float32) or x.shape[1] != 512:
        raise ValueError("cache quantization requires BF16/FP32 D512 rows")
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError("cache quantization has no implicit QAT derivative")
    if format not in ("main_kv_fp4", "swa_mxfp8"):
        raise ValueError("cache format must be main_kv_fp4 or swa_mxfp8")
    if page_size not in (32, 64, 128):
        raise ValueError("page_size must be 32, 64 or 128")
    code, group, width = (1, 16, 288) if format == "main_kv_fp4" else (2, 32, 528)
    n = x.shape[0]
    if out is None:
        if slots is not None:
            raise ValueError("scatter updates require caller-owned cache capacity")
        out = _output(None, (tr.cdiv(n, page_size), page_size, 1, width), x.device)
    else:
        if out.ndim != 4 or out.shape[1:] != (page_size, 1, width):
            raise ValueError("invalid paged cache shape")
        _output(out, out.shape, x.device)
        if slots is None and out.shape[0] * page_size < n:
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
        with torch.cuda.device(x.device):
            _quantize[(tr.cdiv(n, 4),)](
                x,
                out,
                out,
                n,
                512,
                code,
                group,
                4,
                True,
                page_size,
                slots,
                slots is not None,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out


def quantize_index_cache(x, *, page_size=64, out=None, slots=None):
    """DeepGEMM MXFP4 D128 pages, with data region then scale region."""
    _check(x)
    if x.dtype not in (torch.bfloat16, torch.float32) or x.shape[1] != 128:
        raise ValueError("index cache requires BF16/FP32 D128 rows")
    if x.requires_grad and torch.is_grad_enabled():
        raise ValueError("index cache quantization has no implicit QAT derivative")
    if page_size not in (32, 64, 128):
        raise ValueError("page_size must be 32, 64 or 128")
    n = x.shape[0]
    stride = tr.cdiv(page_size * 68, 512) * 512
    if out is None:
        if slots is not None:
            raise ValueError("scatter updates require caller-owned cache capacity")
        out = torch.empty_strided(
            (tr.cdiv(n, page_size), page_size, 1, 68),
            (stride, 68, 68, 1),
            dtype=torch.uint8,
            device=x.device,
        )
    else:
        if (
            out.ndim != 4
            or out.shape[1:] != (page_size, 1, 68)
            or out.dtype != torch.uint8
            or out.device != x.device
            or out.stride(0) < page_size * 68
            or out.stride(0) % 512
            or out.stride()[1:] != (68, 68, 1)
        ):
            raise ValueError(
                "index cache requires uint8 page layout with a 512-byte-aligned page stride"
            )
        stride = out.stride(0)
        if slots is None and out.shape[0] * page_size < n:
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
        with torch.cuda.device(x.device):
            _quantize[(tr.cdiv(n, 4),)](
                x,
                out,
                out,
                n,
                128,
                0,
                32,
                4,
                True,
                page_size,
                slots,
                slots is not None,
                stride,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return out
