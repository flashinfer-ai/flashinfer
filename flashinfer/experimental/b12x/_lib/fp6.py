"""
MX-FP6 (W6A6) utilities for SM120/SM121 CuTe DSL kernels.

Provides FP6 conversion intrinsics, 4-in-3-byte memory packing, inline-PTX
``mxf8f6f4`` MMA helpers, block quantizers (32-element UE8M0 blocks), and
pure-Torch reference paths for parity testing.
"""

from __future__ import annotations

import functools
import math
from typing import Literal, Tuple

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from b12x._lib.intrinsics import (
    align_up,
    cvt_f32_to_ue8m0,
    fabs_f32,
    fmax_f32,
    pack_f32x2_to_bfloat2,
    rcp_approx_ftz,
    swizzle_block_scale,
    ue8m0_to_output_scale,
)

# =============================================================================
# Constants
# =============================================================================

FLOAT6_E3M2_MAX = 28.0
FLOAT6_E2M3_MAX = 7.5
# FP8 E4M3 activations (W6A8): same mxf8f6f4 MMA kind, UE8M0 per-32 scales and
# byte containers as the FP6 paths; only the operand format / fmt_max differ.
FLOAT8_E4M3_MAX = 448.0
SF_VEC_SIZE_FP6 = 32
COPY_BITS = 128

# Max representable magnitude per operand sub-format, used by the global-scale
# recipe below (e4m3 covers the W6A8 activation path).
_GS_FMT_MAX = {
    "e3m2": FLOAT6_E3M2_MAX,
    "e2m3": FLOAT6_E2M3_MAX,
    "e4m3": FLOAT8_E4M3_MAX,
}


def mx_gs_numerator(fmt: str) -> float:
    """Numerator of the amax->global-scale recipe: ``gs = numerator / amax``.

    Fmt-aware ``f8e4m3_max * fmt_max`` (e3m2: 12544.0, e2m3: 3360.0, e4m3:
    200704.0 — all exactly representable in f32). Single source of truth for
    the host paths (``fp6_dense_weights``) and the fused small-M kernel
    (``bf16_to_fp6_small_m``), which must stay bit-identical. Under the ceil
    UE8M0 containment rule the gs largely cancels; the numerator only nudges
    tie-breaking at power-of-two boundaries, so this is a foot-gun removal,
    not an accuracy lever.
    """
    f8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    return f8_max * _GS_FMT_MAX[fmt]

Fp6Format = Literal["e3m2", "e2m3"]

# =============================================================================
# Torch FP6 decode / encode (LUT, OCP MX FP6 bias=1)
# =============================================================================


def _decode_fp6_e3m2(bits: int) -> float:
    # OCP MX FP6 E3M2: 3 exponent bits, bias 3, no inf/nan (exp=7 is a normal).
    bits &= 0x3F
    sign = -1.0 if (bits >> 5) & 1 else 1.0
    exp = (bits >> 2) & 0x7
    mant = bits & 0x3
    if exp == 0:
        if mant == 0:
            return 0.0 if sign > 0 else -0.0
        return sign * (2.0 ** (1 - 3)) * (mant / 4.0)
    return sign * (2.0 ** (exp - 3)) * (1.0 + mant / 4.0)


def _decode_fp6_e2m3(bits: int) -> float:
    # OCP MX FP6 E2M3: 2 exponent bits, bias 1, no inf/nan (exp=3 is a normal).
    bits &= 0x3F
    sign = -1.0 if (bits >> 5) & 1 else 1.0
    exp = (bits >> 3) & 0x3
    mant = bits & 0x7
    if exp == 0:
        if mant == 0:
            return 0.0 if sign > 0 else -0.0
        return sign * (2.0 ** (1 - 1)) * (mant / 8.0)
    return sign * (2.0 ** (exp - 1)) * (1.0 + mant / 8.0)


@functools.lru_cache(maxsize=1)
def _fp6_e3m2_lut() -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    values = tuple(_decode_fp6_e3m2(i) for i in range(64))
    codes = tuple(range(64))
    return values, codes


@functools.lru_cache(maxsize=1)
def _fp6_e2m3_lut() -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    values = tuple(_decode_fp6_e2m3(i) for i in range(64))
    codes = tuple(range(64))
    return values, codes


def build_fp6_decode_lut(
    fmt: Fp6Format,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the 64-entry FP6 code->value decode table as a 1-D tensor.

    The returned tensor is indexed directly by the 6-bit FP6 code (0..63), so a
    decode-focused micro kernel can resolve ``code -> value`` with a single shared-
    memory load instead of bit-twiddling the OCP minifloat fields per element.
    """
    decode = _decode_fp6_e3m2 if fmt == "e3m2" else _decode_fp6_e2m3
    values = [decode(i) for i in range(64)]
    return torch.tensor(values, dtype=dtype, device=device)


@cute.jit
def fp6_decode_code(lut: cute.Tensor, code: Uint32) -> Float32:
    """Decode one 6-bit FP6 code to f32 via a 64-entry decode LUT (``build_fp6_decode_lut``)."""
    return Float32(lut[Int32(code & Uint32(0x3F))])


@cute.jit
def fp6_decode_bytes4(
    lut: cute.Tensor, word: Uint32
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Decode four FP6 byte-containers packed in a uint32 (one code per byte, bits[5:0])."""
    c0 = word & Uint32(0x3F)
    c1 = (word >> Uint32(8)) & Uint32(0x3F)
    c2 = (word >> Uint32(16)) & Uint32(0x3F)
    c3 = (word >> Uint32(24)) & Uint32(0x3F)
    return (
        Float32(lut[Int32(c0)]),
        Float32(lut[Int32(c1)]),
        Float32(lut[Int32(c2)]),
        Float32(lut[Int32(c3)]),
    )


@cute.jit
def fp6_unpack4_codes(
    b0: Uint32, b1: Uint32, b2: Uint32
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Unpack three 3:4-packed FP6 bytes into four 6-bit codes (little-endian 24-bit lane).

    Inverse of ``pack_4_fp6_codes`` / device-side reader for the on-wire packed
    weight storage ``(..., 3*K//4)``: the four codes of element group ``g`` live in
    bytes ``[3g, 3g+1, 3g+2]``.
    """
    bits = (
        (b0 & Uint32(0xFF))
        | ((b1 & Uint32(0xFF)) << Uint32(8))
        | ((b2 & Uint32(0xFF)) << Uint32(16))
    )
    return (
        bits & Uint32(0x3F),
        (bits >> Uint32(6)) & Uint32(0x3F),
        (bits >> Uint32(12)) & Uint32(0x3F),
        (bits >> Uint32(18)) & Uint32(0x3F),
    )


@cute.jit
def mxfp6_swizzled_scale_offset(
    row: Int32, block: Int32, cols_padded_div4: Int32
) -> Int64:
    """Flat byte offset (within one expert) of the UE8M0 scale for logical ``(row, block)``.

    Inverts the cutlass block-scale swizzle produced by ``swizzle_block_scale`` for
    MX-FP6 storage shaped ``(pad128(rows), pad4(num_blocks))``. ``cols_padded_div4``
    is ``align_up(num_blocks, 4) // 4``. Matches ``unswizzle_mxfp6_scales``:

        flat = ((((row//128)*cpd4 + block//4)*32 + row%32)*4 + (row%128)//32)*4 + block%4

    Row-group * stride products are widened to Int64 before multiplication so
    the flat offset cannot wrap for large scale arrays (coding guideline: ID *
    stride arithmetic is Int64).
    """
    r0 = Int64(row >> Int32(7))  # row // 128
    r1 = Int64((row >> Int32(5)) & Int32(3))  # (row % 128) // 32
    r2 = Int64(row & Int32(31))  # row % 32
    c0 = Int64(block >> Int32(2))  # block // 4
    c1 = Int64(block & Int32(3))  # block % 4
    return (
        (
            (((r0 * Int64(cols_padded_div4) + c0) * Int64(32) + r2) * Int64(4) + r1)
            * Int64(4)
        )
        + c1
    )


def _encode_fp6_nearest(value: float, fmt: Fp6Format) -> int:
    if fmt == "e3m2":
        lut_vals, lut_codes = _fp6_e3m2_lut()
        max_val = FLOAT6_E3M2_MAX
    else:
        lut_vals, lut_codes = _fp6_e2m3_lut()
        max_val = FLOAT6_E2M3_MAX
    if value == 0.0:
        return 0
    value = float(max(-max_val, min(max_val, value)))
    best_code = 0
    best_dist = float("inf")
    for code, lut_v in zip(lut_codes, lut_vals):
        if math.isnan(lut_v):
            continue
        dist = abs(lut_v - value)
        if dist < best_dist:
            best_dist = dist
            best_code = code
        elif dist == best_dist and (code & 1) == 0 and (best_code & 1) == 1:
            # Exact midpoint: round half to even (pick the even mantissa LSB),
            # matching the hardware ``cvt.rn`` the GPU quantizer uses. Strict-``<``
            # alone keeps the first/lower code, which disagrees with round-to-even
            # whenever the *higher* representable value has the even mantissa bit.
            best_code = code
    return best_code


def fp6_quantize_values_torch(x: torch.Tensor, fmt: Fp6Format = "e3m2") -> torch.Tensor:
    """Quantize float32 values to the nearest FP6 representable values (Torch LUT)."""
    decode = _decode_fp6_e3m2 if fmt == "e3m2" else _decode_fp6_e2m3
    flat = x.detach().float().reshape(-1)
    out = torch.empty_like(flat)
    for i in range(flat.numel()):
        code = _encode_fp6_nearest(float(flat[i].item()), fmt)
        out[i] = decode(code)
    return out.reshape(x.shape)


def pack_4_fp6_codes(c0: int, c1: int, c2: int, c3: int) -> Tuple[int, int, int]:
    """Pack four 6-bit FP6 codes into three bytes (little-endian 24-bit lane)."""
    bits = (c0 & 0x3F) | ((c1 & 0x3F) << 6) | ((c2 & 0x3F) << 12) | ((c3 & 0x3F) << 18)
    return bits & 0xFF, (bits >> 8) & 0xFF, (bits >> 16) & 0xFF


def unpack_4_fp6_codes(b0: int, b1: int, b2: int) -> Tuple[int, int, int, int]:
    bits = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16)
    return bits & 0x3F, (bits >> 6) & 0x3F, (bits >> 12) & 0x3F, (bits >> 18) & 0x3F


def pack_fp6_codes_tensor(codes: torch.Tensor) -> torch.Tensor:
    """Pack 6-bit FP6 codes shaped ``[..., N]`` (N divisible by 4) into ``[..., 3*N//4]`` uint8.

    Vectorized (no Python per-group loop); bit-identical to the historical
    ``pack_4_fp6_codes`` little-endian 24-bit lane layout.
    """
    if codes.shape[-1] % 4 != 0:
        raise ValueError(f"last dim must be divisible by 4, got {codes.shape[-1]}")
    *lead, n = codes.shape
    groups = n // 4
    c = codes.to(torch.int32).reshape(*lead, groups, 4)
    bits = (
        (c[..., 0] & 0x3F)
        | ((c[..., 1] & 0x3F) << 6)
        | ((c[..., 2] & 0x3F) << 12)
        | ((c[..., 3] & 0x3F) << 18)
    )
    out = torch.empty((*lead, groups * 3), dtype=torch.uint8, device=codes.device)
    packed = out.reshape(*lead, groups, 3)
    packed[..., 0] = (bits & 0xFF).to(torch.uint8)
    packed[..., 1] = ((bits >> 8) & 0xFF).to(torch.uint8)
    packed[..., 2] = ((bits >> 16) & 0xFF).to(torch.uint8)
    return out


def unpack_fp6_packed_tensor(packed: torch.Tensor, num_fp6: int) -> torch.Tensor:
    """Unpack ``[..., 3*num_fp6//4]`` uint8 packed FP6 into 6-bit integer codes ``[..., num_fp6]``."""
    if num_fp6 % 4 != 0:
        raise ValueError(f"num_fp6 must be divisible by 4, got {num_fp6}")
    groups = num_fp6 // 4
    *lead, packed_cols = packed.shape
    if packed_cols != groups * 3:
        raise ValueError(f"packed last dim {packed_cols} != 3*num_fp6/4 ({groups * 3})")
    out = torch.empty((*lead, num_fp6), dtype=torch.uint8, device=packed.device)
    flat_packed = packed.reshape(-1, 3)
    # Each 3-byte group expands to 4 codes; mirror the packing layout where
    # ``pack_fp6_codes_tensor`` reshaped codes to ``(-1, 4)``. Indexing
    # ``flat_out`` per-group requires a 4-wide view, not a ``num_fp6``-wide one.
    flat_out = out.reshape(-1, 4)
    for g in range(flat_packed.shape[0]):
        b0, b1, b2 = (int(flat_packed[g, i].item()) for i in range(3))
        c0, c1, c2, c3 = unpack_4_fp6_codes(b0, b1, b2)
        flat_out[g, 0] = c0
        flat_out[g, 1] = c1
        flat_out[g, 2] = c2
        flat_out[g, 3] = c3
    return out


def expand_mxfp6_packed_to_bytes(packed: torch.Tensor, num_fp6: int) -> torch.Tensor:
    """Vectorized expansion of 3:4-packed FP6 into one byte per code.

    Input ``packed`` is ``[..., 3*num_fp6//4]`` uint8 (the on-disk / wire format).
    Output is ``[..., num_fp6]`` uint8 where each element holds a single 6-bit FP6
    code in bits ``[5:0]`` (bits ``[7:6]`` are zero). This is the byte-container
    layout the ``mxf8f6f4`` warp MMA consumes (8 bits per FP6 value), letting the
    dense/MoE GEMM reuse cutlass's native 8-bit smem/TMA/ldmatrix path while the
    MMA instruction reinterprets each byte as e3m2/e2m3.

    Unlike :func:`unpack_fp6_packed_tensor` this is a pure-tensor (vectorized)
    transform suitable for the per-launch load boundary.
    """
    if num_fp6 % 4 != 0:
        raise ValueError(f"num_fp6 must be divisible by 4, got {num_fp6}")
    groups = num_fp6 // 4
    *lead, packed_cols = packed.shape
    if packed_cols != groups * 3:
        raise ValueError(
            f"packed last dim {packed_cols} != 3*num_fp6/4 ({groups * 3})"
        )
    p = packed.reshape(*lead, groups, 3).to(torch.int32)
    bits = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
    c0 = bits & 0x3F
    c1 = (bits >> 6) & 0x3F
    c2 = (bits >> 12) & 0x3F
    c3 = (bits >> 18) & 0x3F
    out = torch.stack((c0, c1, c2, c3), dim=-1).reshape(*lead, num_fp6)
    return out.to(torch.uint8)


def unswizzle_mxfp6_scales(
    scales_ue8m0: torch.Tensor,
    rows: int,
    num_blocks: int,
) -> torch.Tensor:
    """Invert the MX-FP6 block-scale swizzle to row-major ``(-1, num_blocks)`` bytes.

    ``quantize_grouped_mxfp6_torch`` (and the kernel's SF TMA path) store UE8M0
    scales in the cutlass block-scale swizzle, in one of two equivalent shapes:

    * the 6-D grouped view from :func:`as_grouped_mxfp6_scale_view`
      ``(32, 4, rows_padded//128, 4, cols_padded//4, batch)``; or
    * a flat / 2-D ``(rows_padded, cols_padded)`` buffer (e.g. a kernel TMA dump).

    Both are inverted here back to row-major ``[row, block]`` bytes so the reference
    dequantizer applies the correct per-block scale (the swizzle is invisible under
    uniform scales, which is why it went unnoticed). The forward permutes used by
    :func:`swizzle_block_scale` / :func:`as_grouped_mxfp6_scale_view` are
    involutions on the swapped axes, so the inverse reuses the same index pattern.
    """
    rows_padded = align_up(rows, 128)
    cols_padded = align_up(num_blocks, 4)
    if scales_ue8m0.ndim == 6:
        batch = scales_ue8m0.shape[5]
        orig = (
            scales_ue8m0.permute(5, 2, 1, 0, 4, 3)
            .contiguous()
            .reshape(batch, rows_padded, cols_padded)
        )
        return orig[:, :rows, :num_blocks].reshape(-1, num_blocks)
    sw = scales_ue8m0.reshape(rows_padded // 128, cols_padded // 4, 32, 4, 4)
    orig = sw.permute(0, 3, 2, 1, 4).contiguous().reshape(rows_padded, cols_padded)
    return orig[:rows, :num_blocks].reshape(-1, num_blocks)


def dequant_mxfp6_torch(
    packed: torch.Tensor,
    scales_ue8m0: torch.Tensor,
    *,
    num_fp6: int,
    fmt: Fp6Format = "e3m2",
    global_scale: torch.Tensor | None = None,
    scales_swizzled: bool = True,
) -> torch.Tensor:
    """Dequantize MX-FP6 packed uint8 tensor to float32.

    ``packed`` is ``[..., 3*num_fp6//4]`` uint8. ``scales_ue8m0`` carries one UE8M0
    byte per 32-element block. By default (``scales_swizzled=True``) the scales are
    assumed to be in the cutlass block-scale swizzle produced by
    ``quantize_grouped_mxfp6_torch`` and are inverted to row-major before use; pass
    ``scales_swizzled=False`` when the scales are already plain row-major
    ``[row, block]`` bytes.
    """
    decode = _decode_fp6_e3m2 if fmt == "e3m2" else _decode_fp6_e2m3
    codes = unpack_fp6_packed_tensor(packed, num_fp6)
    flat_codes = codes.reshape(-1, num_fp6)
    flat_out = torch.empty(flat_codes.shape[0], num_fp6, dtype=torch.float32, device=packed.device)
    num_blocks = num_fp6 // SF_VEC_SIZE_FP6
    if scales_swizzled:
        scale_flat = unswizzle_mxfp6_scales(scales_ue8m0, flat_codes.shape[0], num_blocks)
    else:
        scale_flat = scales_ue8m0.reshape(-1, num_blocks)
    if scale_flat.shape[0] != flat_out.shape[0]:
        raise ValueError("scale row count does not match packed row count")
    gs = 1.0
    if global_scale is not None:
        gs = float(global_scale.reshape(-1)[0].item()) if global_scale.numel() > 1 else float(global_scale.item())
    # ``quantize`` stores ``code ~= x * gs / block_scale`` (scaled *up* by the global
    # scale), so reconstructing the original value divides it back out. Using ``* gs``
    # returns ``x * gs**2``; that cancels under the scale-invariant cosine checks but
    # blows the absolute-tolerance round-trip test by ~gs**2.
    inv_gs = 1.0 / gs if gs != 0.0 else 0.0
    for row in range(flat_out.shape[0]):
        for blk in range(num_fp6 // SF_VEC_SIZE_FP6):
            ue = int(scale_flat[row, blk].view(torch.uint8).item())
            block_scale = 0.0 if ue == 0 else float(torch.pow(torch.tensor(2.0), float(ue - 127)))
            for j in range(SF_VEC_SIZE_FP6):
                idx = blk * SF_VEC_SIZE_FP6 + j
                flat_out[row, idx] = decode(int(flat_codes[row, idx].item())) * block_scale * inv_gs
    return flat_out.reshape(*codes.shape[:-1], num_fp6)


def _ue8m0_scale_from_block_max(block_max: torch.Tensor, fmt_max: float) -> torch.Tensor:
    """Compute per-block UE8M0 scale bytes from block amax (Torch reference)."""
    block_max = block_max.float().clamp(min=0.0)
    ratio = block_max / fmt_max
    ratio = torch.where(ratio <= 0, torch.zeros_like(ratio), ratio)
    log2_val = torch.log2(ratio.clamp(min=1e-38))
    exp_int = torch.ceil(log2_val).to(torch.int32)
    ue = (exp_int + 127).clamp(0, 255).to(torch.uint8)
    return torch.where(block_max <= 0, torch.zeros_like(ue), ue)


def pack_grouped_fp6_values(
    values: torch.Tensor,
    fmt: Fp6Format = "e3m2",
) -> torch.Tensor:
    """Pack grouped FP6 float values ``[G, R, C]`` into uint8 ``[R, 3*C//4, G]``."""
    num_groups, rows, cols = values.shape
    if cols % 4 != 0:
        raise ValueError(f"cols must be divisible by 4 for FP6 packing, got {cols}")
    codes = torch.empty((num_groups, rows, cols), dtype=torch.uint8, device=values.device)
    for g in range(num_groups):
        for r in range(rows):
            for c in range(cols):
                codes[g, r, c] = _encode_fp6_nearest(float(values[g, r, c].item()), fmt)
    packed = torch.empty((rows, cols * 3 // 4, num_groups), dtype=torch.uint8, device=values.device)
    for g in range(num_groups):
        packed[..., g] = pack_fp6_codes_tensor(codes[g])
    return packed


def as_grouped_mxfp6_scale_view(
    scale_storage: torch.Tensor,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """View swizzled UE8M0 scales for MX-FP6 (``sf_vec_size=32``)."""
    batch = scale_storage.shape[0]
    rows_padded = align_up(rows, 128)
    cols_padded = align_up(cols // SF_VEC_SIZE_FP6, 4)
    sf = scale_storage.view(torch.float8_e8m0fnu)
    sf = sf.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    return sf.permute(3, 4, 1, 5, 2, 0)


def quantize_grouped_mxfp6_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    fmt: Fp6Format = "e3m2",
    activation_fmt: Fp6Format | None = None,
    bf16_round: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-Torch grouped MX-FP6 quantization with UE8M0 block scales.

    ``bf16_round`` rounds each scaled value through bfloat16 before FP6 encoding,
    faithfully modelling the GPU quantize kernel: the direct ``f32->e3m2`` cvt
    faults on sm_120, so the kernel runs ``f32 -> bf16 -> e3m2``. Without this the
    two paths disagree by 1 ULP on the handful of values sitting exactly on an FP6
    rounding boundary that bf16's coarser mantissa nudges across.
    """
    if activation_fmt is not None:
        fmt = activation_fmt
    fmt_max = FLOAT6_E3M2_MAX if fmt == "e3m2" else FLOAT6_E2M3_MAX
    num_groups, rows, cols = input_tensor.shape
    if cols % SF_VEC_SIZE_FP6 != 0:
        raise ValueError(f"cols must be divisible by {SF_VEC_SIZE_FP6}, got {cols}")
    if cols % 4 != 0:
        raise ValueError(f"cols must be divisible by 4 for FP6 packing, got {cols}")
    if global_scale.numel() == 1:
        global_scale = global_scale.expand(num_groups).contiguous()

    quantized = torch.zeros((num_groups, rows, cols), dtype=torch.float32, device=input_tensor.device)
    scales = torch.zeros(
        (num_groups, rows, cols // SF_VEC_SIZE_FP6),
        dtype=torch.uint8,
        device=input_tensor.device,
    )
    for group_idx in range(num_groups):
        valid_rows = int(row_counts[group_idx].item())
        if valid_rows == 0:
            continue
        x = input_tensor[group_idx, :valid_rows].float()
        gs = float(global_scale[group_idx].item())
        sliced = x.view(valid_rows, cols // SF_VEC_SIZE_FP6, SF_VEC_SIZE_FP6)
        block_max = sliced.abs().amax(dim=-1)
        ue = _ue8m0_scale_from_block_max(block_max * gs, fmt_max)
        block_scale = torch.where(
            ue == 0,
            torch.zeros_like(block_max),
            torch.pow(torch.tensor(2.0, device=x.device), ue.float() - 127.0),
        )
        output_scale = torch.where(
            block_scale == 0,
            torch.zeros_like(block_scale),
            gs / block_scale,
        )
        scaled = sliced * output_scale.unsqueeze(-1)
        if bf16_round:
            scaled = scaled.to(torch.bfloat16).to(torch.float32)
        clipped = scaled.clamp(-fmt_max, fmt_max)
        q = fp6_quantize_values_torch(clipped.view(valid_rows, cols), fmt=fmt)
        quantized[group_idx, :valid_rows] = q
        scales[group_idx, :valid_rows] = ue

    # Pack to [R, 3*C//4, G]
    packed_groups = []
    for group_idx in range(num_groups):
        valid_rows = int(row_counts[group_idx].item())
        if valid_rows == 0:
            packed_groups.append(
                torch.zeros((rows, cols * 3 // 4), dtype=torch.uint8, device=input_tensor.device)
            )
            continue
        codes = torch.zeros((valid_rows, cols), dtype=torch.uint8, device=input_tensor.device)
        for r in range(valid_rows):
            for c in range(cols):
                codes[r, c] = _encode_fp6_nearest(float(quantized[group_idx, r, c].item()), fmt)
        packed_groups.append(pack_fp6_codes_tensor(codes))
    packed = torch.stack(
        [p if p.shape[0] == rows else torch.cat([p, torch.zeros(rows - p.shape[0], cols * 3 // 4, dtype=torch.uint8, device=p.device)], dim=0) for p in packed_groups],
        dim=-1,
    )
    # ``scales`` already holds raw UE8M0 exponent bytes; reinterpret the bits as
    # e8m0 (``.view``) rather than numerically converting (``.to`` would map byte
    # 127/129 to the nearest power-of-2 and collapse distinct scales).
    swizzled = swizzle_block_scale(scales.view(torch.float8_e8m0fnu))
    scale_view = as_grouped_mxfp6_scale_view(swizzled.view(torch.uint8), rows, cols)
    return packed, scale_view


def silu_mul_quantize_grouped_mxfp6_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    fmt: Fp6Format = "e3m2",
) -> tuple[torch.Tensor, torch.Tensor]:
    cols = input_tensor.shape[-1] // 2
    left = input_tensor[..., :cols].float()
    right = input_tensor[..., cols:].float()
    activated = (F.silu(left) * right).to(input_tensor.dtype).to(torch.float32)
    return quantize_grouped_mxfp6_torch(activated, row_counts, global_scale, fmt=fmt)


def relu2_quantize_grouped_mxfp6_torch(
    input_tensor: torch.Tensor,
    row_counts: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    fmt: Fp6Format = "e3m2",
) -> tuple[torch.Tensor, torch.Tensor]:
    activated = torch.square(F.relu(input_tensor.float()))
    activated = activated.to(input_tensor.dtype).to(torch.float32)
    return quantize_grouped_mxfp6_torch(activated, row_counts, global_scale, fmt=fmt)


# =============================================================================
# PTX FP6 conversion intrinsics
# =============================================================================


@dsl_user_op
def cvt_bf16x2_to_e3m2x2(src: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert packed bf16x2 to packed e3m2x2 in the low 16 bits of a u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(src).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 out, zero;
                cvt.rn.satfinite.e3m2x2.bf16x2 out, $1;
                mov.u16 zero, 0;
                mov.b32 $0, {out, zero};
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_bf16x2x2_to_e3m2x4(lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert two bf16x2 values and pack the e3m2x2 results into one u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 out_lo, out_hi;
                cvt.rn.satfinite.e3m2x2.bf16x2 out_lo, $1;
                cvt.rn.satfinite.e3m2x2.bf16x2 out_hi, $2;
                mov.b32 $0, {out_lo, out_hi};
            }
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_bf16x2_to_e2m3x2(src: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert packed bf16x2 to packed e2m3x2 in the low 16 bits of a u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(src).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 out, zero;
                cvt.rn.satfinite.e2m3x2.bf16x2 out, $1;
                mov.u16 zero, 0;
                mov.b32 $0, {out, zero};
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_bf16x2x2_to_e2m3x4(lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert two bf16x2 values and pack the e2m3x2 results into one u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 out_lo, out_hi;
                cvt.rn.satfinite.e2m3x2.bf16x2 out_lo, $1;
                cvt.rn.satfinite.e2m3x2.bf16x2 out_hi, $2;
                mov.b32 $0, {out_lo, out_hi};
            }
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_f32_to_e3m2x2(
    hi: Float32,
    lo: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert two float32 values to two E3M2 byte containers (16-bit pair)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 out, zero;
                cvt.rn.satfinite.e3m2x2.f32 out, $2, $1;
                mov.u16 zero, 0;
                mov.b32 $0, {out, zero};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_f32_to_e4m3x2(
    hi: Float32,
    lo: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert two float32 values to two FP8 E4M3 bytes (16-bit pair).

    ``cvt.rn.satfinite.e4m3x2.f32`` is PTX ISA 7.8 (sm_89), so unlike the
    bf16x2-source FP6 forms it assembles under vLLM's negotiated PTX version.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 out, zero;
                cvt.rn.satfinite.e4m3x2.f32 out, $2, $1;
                mov.u16 zero, 0;
                mov.b32 $0, {out, zero};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_f32_to_e2m3x2(
    hi: Float32,
    lo: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert two float32 values to two E2M3 byte containers (16-bit pair)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 out, zero;
                cvt.rn.satfinite.e2m3x2.f32 out, $2, $1;
                mov.u16 zero, 0;
                mov.b32 $0, {out, zero};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _make_i16_const(value: int, *, loc=None, ip=None):
    i16_ty = cutlass._mlir.ir.IntegerType.get_signless(16)
    return cutlass._mlir.ir.Operation.create(
        "llvm.mlir.constant",
        results=[i16_ty],
        attributes={"value": cutlass._mlir.ir.IntegerAttr.get(i16_ty, int(value))},
    ).result


def _mxfp6_mma_inline(
    ptx_types: str,
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int,
    tid_a: int,
    bid_b: int,
    tid_b: int,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    bid_a_i16 = _make_i16_const(bid_a, loc=loc, ip=ip)
    tid_a_i16 = _make_i16_const(tid_a, loc=loc, ip=ip)
    bid_b_i16 = _make_i16_const(bid_b, loc=loc, ip=ip)
    tid_b_i16 = _make_i16_const(tid_b, loc=loc, ip=ip)
    ptx = (
        f"mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
        f"m16n8k32.row.col.f32.{ptx_types}.f32.ue8m0"
    )
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Uint32(sfa).ir_value(loc=loc, ip=ip),
            bid_a_i16,
            tid_a_i16,
            Uint32(sfb).ir_value(loc=loc, ip=ip),
            bid_b_i16,
            tid_b_i16,
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        f"""
        {ptx}
        {{$0, $1, $2, $3}},
        {{$4, $5, $6, $7}},
        {{$8, $9}},
        {{$0, $1, $2, $3}},
        {{$10}},
        {{$11, $12}},
        {{$13}},
        {{$14, $15}};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,h,h,r,h,h,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def mxfp6_mma_m16n8k32_f32_e3m2_e3m2(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA for SM120 MX-FP6 ``m16n8k32`` with E3M2 x E3M2 operands."""
    return _mxfp6_mma_inline(
        "e3m2.e3m2", d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        bid_a, tid_a, bid_b, tid_b, loc=loc, ip=ip,
    )


@dsl_user_op
def mxfp6_mma_m16n8k32_f32_e2m3_e2m3(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA for SM120 MX-FP6 ``m16n8k32`` with E2M3 x E2M3 operands."""
    return _mxfp6_mma_inline(
        "e2m3.e2m3", d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        bid_a, tid_a, bid_b, tid_b, loc=loc, ip=ip,
    )


@dsl_user_op
def mxfp6_mma_m16n8k32_f32_e2m3_e3m2(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA for SM120 MX-FP6 ``m16n8k32`` with E2M3 x E3M2 operands (default weight/act)."""
    return _mxfp6_mma_inline(
        "e2m3.e3m2", d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        bid_a, tid_a, bid_b, tid_b, loc=loc, ip=ip,
    )


@dsl_user_op
def mxfp6_mma_m16n8k32_f32_e3m2_e2m3(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA for SM120 MX-FP6 ``m16n8k32`` with E3M2 x E2M3 operands."""
    return _mxfp6_mma_inline(
        "e3m2.e2m3", d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        bid_a, tid_a, bid_b, tid_b, loc=loc, ip=ip,
    )


# W6A8 variants: FP8 E4M3 activations against FP6 weights. Same mxf8f6f4 MMA
# kind / scales / containers; only the activation operand's type qualifier
# changes. The activation side is operand A in the MoE kernel and operand B in
# the dense kernel, so both orientations exist for each weight format.


@dsl_user_op
def mxfp6_mma_m16n8k32_f32_e4m3_e2m3(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA ``m16n8k32`` with E4M3 (act) x E2M3 (weight) operands."""
    return _mxfp6_mma_inline(
        "e4m3.e2m3", d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        bid_a, tid_a, bid_b, tid_b, loc=loc, ip=ip,
    )


@dsl_user_op
def mxfp6_mma_m16n8k32_f32_e4m3_e3m2(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA ``m16n8k32`` with E4M3 (act) x E3M2 (weight) operands."""
    return _mxfp6_mma_inline(
        "e4m3.e3m2", d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        bid_a, tid_a, bid_b, tid_b, loc=loc, ip=ip,
    )


@dsl_user_op
def mxfp6_mma_m16n8k32_f32_e2m3_e4m3(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA ``m16n8k32`` with E2M3 (weight) x E4M3 (act) operands."""
    return _mxfp6_mma_inline(
        "e2m3.e4m3", d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        bid_a, tid_a, bid_b, tid_b, loc=loc, ip=ip,
    )


@dsl_user_op
def mxfp6_mma_m16n8k32_f32_e3m2_e4m3(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA ``m16n8k32`` with E3M2 (weight) x E4M3 (act) operands."""
    return _mxfp6_mma_inline(
        "e3m2.e4m3", d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, sfa, sfb,
        bid_a, tid_a, bid_b, tid_b, loc=loc, ip=ip,
    )


# =============================================================================
# Memory pack / unpack (4 FP6 codes -> 3 bytes; byte containers for MMA)
# =============================================================================


@cute.jit
def pack_4_fp6_codes_to_3bytes(
    c0: Uint32,
    c1: Uint32,
    c2: Uint32,
    c3: Uint32,
) -> Tuple[Uint8, Uint8, Uint8]:
    """Pack four 6-bit FP6 codes into three bytes."""
    mask = Uint32(0x3F)
    bits = (c0 & mask) | ((c1 & mask) << 6) | ((c2 & mask) << 12) | ((c3 & mask) << 18)
    return (
        Uint8(bits & Uint32(0xFF)),
        Uint8((bits >> 8) & Uint32(0xFF)),
        Uint8((bits >> 16) & Uint32(0xFF)),
    )


@cute.jit
def unpack_3bytes_to_4_fp6_codes(
    b0: Uint8,
    b1: Uint8,
    b2: Uint8,
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Unpack three bytes into four 6-bit FP6 codes."""
    bits = Uint32(b0) | (Uint32(b1) << 8) | (Uint32(b2) << 16)
    mask = Uint32(0x3F)
    return (
        bits & mask,
        (bits >> 6) & mask,
        (bits >> 12) & mask,
        (bits >> 18) & mask,
    )


@cute.jit
def unpack_3bytes_to_4_byte_containers(
    b0: Uint8,
    b1: Uint8,
    b2: Uint8,
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Unpack three packed bytes into four FP6 byte-container registers (low 6 bits set)."""
    c0, c1, c2, c3 = unpack_3bytes_to_4_fp6_codes(b0, b1, b2)
    return c0, c1, c2, c3


@cute.jit
def pack_4_byte_containers_to_3bytes(
    c0: Uint32,
    c1: Uint32,
    c2: Uint32,
    c3: Uint32,
) -> Tuple[Uint8, Uint8, Uint8]:
    """Pack four FP6 byte-container registers (6-bit payloads) into three bytes."""
    return pack_4_fp6_codes_to_3bytes(c0 & Uint32(0x3F), c1 & Uint32(0x3F), c2 & Uint32(0x3F), c3 & Uint32(0x3F))


@cute.jit
def cvt_e3m2x8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
) -> Uint64:
    """Convert eight float32 values to eight E3M2 byte containers in a uint64."""
    # Use the standard f32-source cvt (``cvt.rn.satfinite.e3m2x2.f32``). The
    # bf16x2-source form is a PTX ISA 9.1-only feature, and vLLM's in-process
    # compiler negotiates a lower PTX version (nvJitLink), so the bf16x2 cvt fails
    # to assemble inside a vLLM worker. ``cvt_f32_to_e3m2x2(hi, lo)`` reproduces the
    # byte order the old ``pack_f32x2_to_bfloat2(lo, hi)`` bf16x2 path produced.
    p0 = cvt_f32_to_e3m2x2(v1, v0)
    p1 = cvt_f32_to_e3m2x2(v3, v2)
    p2 = cvt_f32_to_e3m2x2(v5, v4)
    p3 = cvt_f32_to_e3m2x2(v7, v6)
    return (
        Uint64(p3) << Uint64(48)
        | Uint64(p2) << Uint64(32)
        | Uint64(p1) << Uint64(16)
        | Uint64(p0)
    )


@cute.jit
def cvt_e2m3x8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
) -> Uint64:
    """Convert eight float32 values to eight E2M3 byte containers in a uint64."""
    # See cvt_e3m2x8_f32: use the standard f32-source cvt so the kernel assembles
    # under vLLM's lower negotiated PTX version (the bf16x2 form needs PTX ISA 9.1).
    p0 = cvt_f32_to_e2m3x2(v1, v0)
    p1 = cvt_f32_to_e2m3x2(v3, v2)
    p2 = cvt_f32_to_e2m3x2(v5, v4)
    p3 = cvt_f32_to_e2m3x2(v7, v6)
    return (
        Uint64(p3) << Uint64(48)
        | Uint64(p2) << Uint64(32)
        | Uint64(p1) << Uint64(16)
        | Uint64(p0)
    )


@cute.jit
def cvt_e4m3x8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
) -> Uint64:
    """Convert eight float32 values to eight FP8 E4M3 bytes in a uint64."""
    p0 = cvt_f32_to_e4m3x2(v1, v0)
    p1 = cvt_f32_to_e4m3x2(v3, v2)
    p2 = cvt_f32_to_e4m3x2(v5, v4)
    p3 = cvt_f32_to_e4m3x2(v7, v6)
    return (
        Uint64(p3) << Uint64(48)
        | Uint64(p2) << Uint64(32)
        | Uint64(p1) << Uint64(16)
        | Uint64(p0)
    )


@cute.jit
def pack_8_byte_containers_to_6bytes(
    b0: Uint32,
    b1: Uint32,
    b2: Uint32,
    b3: Uint32,
    b4: Uint32,
    b5: Uint32,
    b6: Uint32,
    b7: Uint32,
) -> Tuple[Uint8, Uint8, Uint8, Uint8, Uint8, Uint8]:
    """Pack eight 6-bit FP6 byte-container registers into six bytes."""
    m = Uint32(0x3F)
    bits = (
        (b0 & m)
        | ((b1 & m) << 6)
        | ((b2 & m) << 12)
        | ((b3 & m) << 18)
        | ((b4 & m) << 24)
        | ((b5 & m) << 30)
        | ((b6 & m) << 36)
        | ((b7 & m) << 42)
    )
    return (
        Uint8(bits & Uint32(0xFF)),
        Uint8((bits >> 8) & Uint32(0xFF)),
        Uint8((bits >> 16) & Uint32(0xFF)),
        Uint8((bits >> 24) & Uint32(0xFF)),
        Uint8((bits >> 32) & Uint32(0xFF)),
        Uint8((bits >> 40) & Uint32(0xFF)),
    )


# =============================================================================
# Block quantizers (32-element blocks, UE8M0 scales)
# =============================================================================


@cute.jit
def max_abs_32(values: cute.Tensor) -> Float32:
    """Maximum absolute value across 32 float32 elements."""
    result = fabs_f32(values[0])
    for i in cutlass.range_constexpr(1, 32):
        result = fmax_f32(result, fabs_f32(values[i]))
    return result


@cute.jit
def quantize_and_pack_8_e3m2(y_f32: cute.Tensor, inv_scale: Float32) -> Uint64:
    """Quantize eight float32 values to E3M2 byte containers."""
    q = cute.make_rmem_tensor((8,), Float32)
    for i in cutlass.range_constexpr(8):
        q[i] = y_f32[i] * inv_scale
    return cvt_e3m2x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])


@cute.jit
def quantize_and_pack_8_e2m3(y_f32: cute.Tensor, inv_scale: Float32) -> Uint64:
    """Quantize eight float32 values to E2M3 byte containers."""
    q = cute.make_rmem_tensor((8,), Float32)
    for i in cutlass.range_constexpr(8):
        q[i] = y_f32[i] * inv_scale
    return cvt_e2m3x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])


@cute.jit
def pack_32_byte_containers_to_24bytes(
    containers: cute.Tensor,
) -> Tuple[Uint64, Uint64, Uint64]:
    """Pack 32 FP6 byte-container codes (6-bit) from a length-32 tensor into three uint64."""
    packed_bytes = cute.make_rmem_tensor((24,), Uint8)
    for gi in cutlass.range_constexpr(8):
        base = gi * 4
        b0, b1, b2 = pack_4_byte_containers_to_3bytes(
            containers[base],
            containers[base + 1],
            containers[base + 2],
            containers[base + 3],
        )
        off = gi * 3
        packed_bytes[off] = b0
        packed_bytes[off + 1] = b1
        packed_bytes[off + 2] = b2
    lo = Uint64(0)
    mid = Uint64(0)
    hi = Uint64(0)
    for i in cutlass.range_constexpr(8):
        lo = lo | (Uint64(packed_bytes[i]) << Uint64(i * 8))
        mid = mid | (Uint64(packed_bytes[8 + i]) << Uint64(i * 8))
        hi = hi | (Uint64(packed_bytes[16 + i]) << Uint64(i * 8))
    return lo, mid, hi


@cute.jit
def pack_32_byte_containers_to_4xu64(
    containers: cute.Tensor,
) -> Tuple[Uint64, Uint64, Uint64, Uint64]:
    """32 FP6 byte-containers -> four u64 lanes, one code per 8-bit byte lane.

    Unpacked analogue of :func:`pack_32_byte_containers_to_24bytes`: no 3:4
    compression, each byte holds one 6-bit code in bits[5:0]. Lets the quantizer
    emit the byte-container layout the ``mxf8f6f4`` GEMM consumes directly with
    four vectorized u64 stores per 32-element block.
    """
    q0 = Uint64(0)
    q1 = Uint64(0)
    q2 = Uint64(0)
    q3 = Uint64(0)
    for i in cutlass.range_constexpr(8):
        sh = Uint64(i * 8)
        q0 = q0 | (Uint64(containers[i]) << sh)
        q1 = q1 | (Uint64(containers[8 + i]) << sh)
        q2 = q2 | (Uint64(containers[16 + i]) << sh)
        q3 = q3 | (Uint64(containers[24 + i]) << sh)
    return q0, q1, q2, q3


@cute.jit
def _quantize_to_containers_e3m2(
    values: cute.Tensor,
    inv_scale: Float32,
) -> cute.Tensor:
    containers = cute.make_rmem_tensor((32,), Uint32)
    for chunk in cutlass.range_constexpr(4):
        q = cute.make_rmem_tensor((8,), Float32)
        for j in cutlass.range_constexpr(8):
            q[j] = values[chunk * 8 + j] * inv_scale
        packed64 = cvt_e3m2x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
        for j in cutlass.range_constexpr(8):
            containers[chunk * 8 + j] = Uint32(
                (packed64 >> Uint64(j * 8)) & Uint64(0xFF)
            )
    return containers


@cute.jit
def _quantize_to_containers_e2m3(
    values: cute.Tensor,
    inv_scale: Float32,
) -> cute.Tensor:
    containers = cute.make_rmem_tensor((32,), Uint32)
    for chunk in cutlass.range_constexpr(4):
        q = cute.make_rmem_tensor((8,), Float32)
        for j in cutlass.range_constexpr(8):
            q[j] = values[chunk * 8 + j] * inv_scale
        packed64 = cvt_e2m3x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
        for j in cutlass.range_constexpr(8):
            containers[chunk * 8 + j] = Uint32(
                (packed64 >> Uint64(j * 8)) & Uint64(0xFF)
            )
    return containers


@cute.jit
def _quantize_to_containers_e4m3(
    values: cute.Tensor,
    inv_scale: Float32,
) -> cute.Tensor:
    containers = cute.make_rmem_tensor((32,), Uint32)
    for chunk in cutlass.range_constexpr(4):
        q = cute.make_rmem_tensor((8,), Float32)
        for j in cutlass.range_constexpr(8):
            q[j] = values[chunk * 8 + j] * inv_scale
        packed64 = cvt_e4m3x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
        for j in cutlass.range_constexpr(8):
            containers[chunk * 8 + j] = Uint32(
                (packed64 >> Uint64(j * 8)) & Uint64(0xFF)
            )
    return containers


@dsl_user_op
def fp6_block_ue8m0_exact(
    max_abs: Float32,
    global_scale_val: Float32,
    fmt_max: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Exact UE8M0 block-scale byte matching ``_ue8m0_scale_from_block_max``.

    Computes ``ceil(log2((max_abs * gs) / fmt_max)) + 127`` with a true ``div.rn``
    (same operand order as the Torch reference) and IEEE-754 bit extraction
    (``exponent_field + (mantissa != 0)``) instead of ``lg2.approx``/``rcp.approx``.
    The approximate path rounds the ceil the wrong way near powers of two, flipping
    the exponent by one; that desync breaks power-of-two scale/code compensation for
    small block elements (they underflow to zero in only one path).
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(max_abs).ir_value(loc=loc, ip=ip),
                Float32(global_scale_val).ir_value(loc=loc, ip=ip),
                Float32(fmt_max).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_zero, p_neg, p_ovf, p_mant;
                .reg .f32 t, ratio;
                .reg .b32 bits, expf, mant, result;

                mul.f32 t, $1, $2;
                div.rn.f32 ratio, t, $3;

                setp.le.f32 p_zero, ratio, 0f00000000;
                mov.b32 bits, ratio;
                and.b32 expf, bits, 0x7F800000;
                shr.u32 expf, expf, 23;
                and.b32 mant, bits, 0x007FFFFF;
                setp.ne.u32 p_mant, mant, 0;
                selp.s32 result, 1, 0, p_mant;
                add.s32 result, result, expf;

                setp.lt.s32 p_neg, result, 0;
                setp.gt.s32 p_ovf, result, 255;
                selp.s32 result, 0, result, p_neg;
                selp.s32 result, 255, result, p_ovf;
                selp.s32 $0, 0, result, p_zero;
            }
            """,
            "=r,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ue8m0_output_scale_exact(
    ue8m0_val: Uint32,
    global_scale_val: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """Exact ``gs * 2^(127-ue)`` (== reference ``gs / 2^(ue-127)``).

    ``ue8m0_to_output_scale`` forms ``2^(127-ue)`` with ``ex2.approx``, which is
    not bit-exact for integer exponents (~1e-7 relative error). That perturbation
    flips the bf16/FP6 rounding for the few values sitting on a quantization
    boundary. Here the power of two is built directly from IEEE-754 bits
    (exponent field ``254-ue``, zero mantissa), so multiplying ``gs`` by it is
    exact and matches the Torch reference's scaled value bit-for-bit.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(ue8m0_val).ir_value(loc=loc, ip=ip),
                Float32(global_scale_val).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_zero;
                .reg .b32 ef, pw;
                .reg .f32 powf, result;

                setp.eq.u32 p_zero, $1, 0;
                sub.s32 ef, 254, $1;
                shl.b32 pw, ef, 23;
                mov.b32 powf, pw;
                mul.f32 result, powf, $2;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def quantize_block_fp6_e3m2_fast(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint8]:
    """Fast MX-FP6 E3M2 block quantize (32 elements) with UE8M0 scale byte."""
    scale_byte = Uint8(0)
    lo = Uint64(0)
    mid = Uint64(0)
    hi = Uint64(0)
    if global_scale_val != Float32(0.0):
        scale_u32 = fp6_block_ue8m0_exact(
            max_abs, global_scale_val, Float32(FLOAT6_E3M2_MAX)
        )
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_scale = ue8m0_output_scale_exact(scale_u32, global_scale_val)
        if inv_scale != Float32(0.0):
            containers = _quantize_to_containers_e3m2(values, inv_scale)
            lo, mid, hi = pack_32_byte_containers_to_24bytes(containers)
    return lo, mid, hi, scale_byte


@cute.jit
def quantize_block_fp6_e2m3_fast(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint8]:
    """Fast MX-FP6 E2M3 block quantize (32 elements) with UE8M0 scale byte."""
    scale_byte = Uint8(0)
    lo = Uint64(0)
    mid = Uint64(0)
    hi = Uint64(0)
    if global_scale_val != Float32(0.0):
        scale_u32 = fp6_block_ue8m0_exact(
            max_abs, global_scale_val, Float32(FLOAT6_E2M3_MAX)
        )
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_scale = ue8m0_output_scale_exact(scale_u32, global_scale_val)
        if inv_scale != Float32(0.0):
            containers = _quantize_to_containers_e2m3(values, inv_scale)
            lo, mid, hi = pack_32_byte_containers_to_24bytes(containers)
    return lo, mid, hi, scale_byte


@cute.jit
def quantize_block_fp6_e3m2_bytes(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint64, Uint8]:
    """E3M2 block quantize (32 elems) -> four u64 byte-container lanes + scale.

    Identical quantization math to :func:`quantize_block_fp6_e3m2_fast`; only
    the output packing differs (1 byte per code instead of 3:4), so the GEMM's
    per-call activation expansion is unnecessary."""
    scale_byte = Uint8(0)
    q0 = Uint64(0)
    q1 = Uint64(0)
    q2 = Uint64(0)
    q3 = Uint64(0)
    if global_scale_val != Float32(0.0):
        scale_u32 = fp6_block_ue8m0_exact(
            max_abs, global_scale_val, Float32(FLOAT6_E3M2_MAX)
        )
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_scale = ue8m0_output_scale_exact(scale_u32, global_scale_val)
        if inv_scale != Float32(0.0):
            containers = _quantize_to_containers_e3m2(values, inv_scale)
            q0, q1, q2, q3 = pack_32_byte_containers_to_4xu64(containers)
    return q0, q1, q2, q3, scale_byte


@cute.jit
def quantize_block_fp6_e2m3_bytes(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint64, Uint8]:
    """E2M3 block quantize (32 elems) -> four u64 byte-container lanes + scale.

    See :func:`quantize_block_fp6_e3m2_bytes`."""
    scale_byte = Uint8(0)
    q0 = Uint64(0)
    q1 = Uint64(0)
    q2 = Uint64(0)
    q3 = Uint64(0)
    if global_scale_val != Float32(0.0):
        scale_u32 = fp6_block_ue8m0_exact(
            max_abs, global_scale_val, Float32(FLOAT6_E2M3_MAX)
        )
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_scale = ue8m0_output_scale_exact(scale_u32, global_scale_val)
        if inv_scale != Float32(0.0):
            containers = _quantize_to_containers_e2m3(values, inv_scale)
            q0, q1, q2, q3 = pack_32_byte_containers_to_4xu64(containers)
    return q0, q1, q2, q3, scale_byte


@cute.jit
def quantize_block_fp8_e4m3_bytes(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint64, Uint8]:
    """E4M3 block quantize (32 elems) -> four u64 byte-container lanes + scale.

    W6A8 activation path: same UE8M0 amax-ceil scale and byte-container layout
    as :func:`quantize_block_fp6_e2m3_bytes`, but the bytes hold real E4M3 codes
    (``FLOAT8_E4M3_MAX``) for the ``e4m3.e2m3`` dense/MoE MMA."""
    scale_byte = Uint8(0)
    q0 = Uint64(0)
    q1 = Uint64(0)
    q2 = Uint64(0)
    q3 = Uint64(0)
    if global_scale_val != Float32(0.0):
        scale_u32 = fp6_block_ue8m0_exact(
            max_abs, global_scale_val, Float32(FLOAT8_E4M3_MAX)
        )
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_scale = ue8m0_output_scale_exact(scale_u32, global_scale_val)
        if inv_scale != Float32(0.0):
            containers = _quantize_to_containers_e4m3(values, inv_scale)
            q0, q1, q2, q3 = pack_32_byte_containers_to_4xu64(containers)
    return q0, q1, q2, q3, scale_byte


@cute.jit
def quantize_block_fp6_e3m2_containers(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[cute.Tensor, Uint8]:
    """E3M2 block quantize (32 elems) -> 32 byte-containers + UE8M0 scale byte.

    Like :func:`quantize_block_fp6_e3m2_fast` but skips the 3:4 pack and returns
    the 32 expanded byte-containers (one FP6 code in bits[5:0] of each ``Uint8``).
    Feeds the byte-container smem/``mxf8f6f4`` MMA path (the FP6 analogue of the
    dense load-boundary expansion, but produced in-kernel by the route-pack)."""
    out = cute.make_rmem_tensor((32,), Uint8)
    for i in cutlass.range_constexpr(32):
        out[i] = Uint8(0)
    scale_byte = Uint8(0)
    if global_scale_val != Float32(0.0):
        scale_u32 = fp6_block_ue8m0_exact(
            max_abs, global_scale_val, Float32(FLOAT6_E3M2_MAX)
        )
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_scale = ue8m0_output_scale_exact(scale_u32, global_scale_val)
        if inv_scale != Float32(0.0):
            containers = _quantize_to_containers_e3m2(values, inv_scale)
            for i in cutlass.range_constexpr(32):
                out[i] = Uint8(containers[i] & Uint32(0xFF))
    return out, scale_byte


@cute.jit
def quantize_block_fp6_e2m3_containers(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[cute.Tensor, Uint8]:
    """E2M3 block quantize (32 elems) -> 32 byte-containers + UE8M0 scale byte.

    See :func:`quantize_block_fp6_e3m2_containers`."""
    out = cute.make_rmem_tensor((32,), Uint8)
    for i in cutlass.range_constexpr(32):
        out[i] = Uint8(0)
    scale_byte = Uint8(0)
    if global_scale_val != Float32(0.0):
        scale_u32 = fp6_block_ue8m0_exact(
            max_abs, global_scale_val, Float32(FLOAT6_E2M3_MAX)
        )
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_scale = ue8m0_output_scale_exact(scale_u32, global_scale_val)
        if inv_scale != Float32(0.0):
            containers = _quantize_to_containers_e2m3(values, inv_scale)
            for i in cutlass.range_constexpr(32):
                out[i] = Uint8(containers[i] & Uint32(0xFF))
    return out, scale_byte


@cute.jit
def quantize_block_fp8_e4m3_containers(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[cute.Tensor, Uint8]:
    """FP8 E4M3 block quantize (32 elems) -> 32 bytes + UE8M0 scale byte.

    The W6A8 activation hop: identical structure to
    :func:`quantize_block_fp6_e2m3_containers` (same UE8M0 amax-ceil scale,
    same byte-container output consumed by the ``mxf8f6f4`` MMA), but the
    bytes hold real E4M3 codes and the scale uses ``FLOAT8_E4M3_MAX``."""
    out = cute.make_rmem_tensor((32,), Uint8)
    for i in cutlass.range_constexpr(32):
        out[i] = Uint8(0)
    scale_byte = Uint8(0)
    if global_scale_val != Float32(0.0):
        scale_u32 = fp6_block_ue8m0_exact(
            max_abs, global_scale_val, Float32(FLOAT8_E4M3_MAX)
        )
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_scale = ue8m0_output_scale_exact(scale_u32, global_scale_val)
        if inv_scale != Float32(0.0):
            containers = _quantize_to_containers_e4m3(values, inv_scale)
            for i in cutlass.range_constexpr(32):
                out[i] = Uint8(containers[i] & Uint32(0xFF))
    return out, scale_byte


@cute.jit
def quantize_block_fp6_e3m2(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint8]:
    """MX-FP6 E3M2 block quantize (32 elements) with UE8M0 scale byte."""
    return quantize_block_fp6_e3m2_fast(values, max_abs, global_scale_val)


@cute.jit
def quantize_block_fp6_e2m3(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint8]:
    """MX-FP6 E2M3 block quantize (32 elements) with UE8M0 scale byte."""
    return quantize_block_fp6_e2m3_fast(values, max_abs, global_scale_val)


@cute.jit
def silu_mul_32(
    gate: cute.Tensor,
    up: cute.Tensor,
) -> cute.Tensor:
    """Fused SiLU(gate) * up for 32 float32 element pairs."""
    out = cute.make_rmem_tensor((32,), Float32)
    for i in cutlass.range_constexpr(32):
        g = gate[i]
        sigmoid_g = cute.arch.rcp_approx(
            Float32(1.0) + cute.math.exp(-g, fastmath=False)
        )
        out[i] = g * sigmoid_g * up[i]
    return out


@cute.jit
def relu2_32(x: cute.Tensor) -> cute.Tensor:
    """Compute ReLU^2 for 32 float32 values."""
    out = cute.make_rmem_tensor((32,), Float32)
    for i in cutlass.range_constexpr(32):
        v = fmax_f32(x[i], Float32(0.0))
        out[i] = v * v
    return out


@cute.jit
def silu_mul_quantize_block_fp6_e3m2(
    gate: cute.Tensor,
    up: cute.Tensor,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint8]:
    """Fused SiLU(gate)*up + MX-FP6 E3M2 quantize for 32 element pairs."""
    activated = silu_mul_32(gate, up)
    block_max = max_abs_32(activated)
    return quantize_block_fp6_e3m2_fast(activated, block_max, global_scale_val)


@cute.jit
def silu_mul_quantize_block_fp6_e2m3(
    gate: cute.Tensor,
    up: cute.Tensor,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint8]:
    """Fused SiLU(gate)*up + MX-FP6 E2M3 quantize for 32 element pairs."""
    activated = silu_mul_32(gate, up)
    block_max = max_abs_32(activated)
    return quantize_block_fp6_e2m3_fast(activated, block_max, global_scale_val)


@cute.jit
def relu2_quantize_block_fp6_e3m2(
    x: cute.Tensor,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint8]:
    """Fuse ReLU^2 and MX-FP6 E3M2 quantization for 32 float32 values."""
    activated = relu2_32(x)
    block_max = max_abs_32(activated)
    return quantize_block_fp6_e3m2_fast(activated, block_max, global_scale_val)


@cute.jit
def relu2_quantize_block_fp6_e2m3(
    x: cute.Tensor,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint64, Uint64, Uint8]:
    """Fuse ReLU^2 and MX-FP6 E2M3 quantization for 32 float32 values."""
    activated = relu2_32(x)
    block_max = max_abs_32(activated)
    return quantize_block_fp6_e2m3_fast(activated, block_max, global_scale_val)