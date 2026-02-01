"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Common utilities for quantization kernels using CuTe-DSL.

This module contains shared PTX intrinsics and helper functions for MXFP8
quantization kernels.
"""

import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from ..cute_dsl.fp4_common import habs2, hmax2, bfloat2_habs2, bfloat2_hmax2


# =============================================================================
# MXFP8 Constants
# =============================================================================

# Scale factor vector size: each scale factor covers 32 elements
SF_VEC_SIZE = 32

# Inverse of max representable value in FP8 E4M3 format (1/448)
INV_FLOAT8_E4M3_MAX = 1.0 / 448.0

# Thread organization constants
WARP_SIZE = 32
ELTS_PER_THREAD = 8  # Each thread handles 8 FP16 elements (128 bits)
THREADS_PER_SF = SF_VEC_SIZE // ELTS_PER_THREAD  # 32 / 8 = 4 threads per SF block
SF_BLOCKS_PER_WARP = WARP_SIZE // THREADS_PER_SF  # 32 / 4 = 8 SF blocks per warp

# Row tiling for swizzled layout (128x4 pattern)
ROW_TILE_SIZE = 128


# =============================================================================
# Half2 SIMD Intrinsics for Max Reduction
# =============================================================================


@dsl_user_op
def hmax_reduce_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 FP16 values in a Half2 as Float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h0, h1;
                .reg .f32 f0, f1;
                mov.b32 {h0, h1}, $1;
                cvt.f32.f16 f0, h0;
                cvt.f32.f16 f1, h1;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_hmax_reduce_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 BF16 values as Float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# =============================================================================
# Fast UE8M0 Conversion
# =============================================================================


@dsl_user_op
def float_to_ue8m0_fast(value: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Convert float to UE8M0 format using fast log2 approximation.

    UE8M0 = ceil(log2(value)) + 127, clamped to [0, 255]
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_neg, p_ovf;
                .reg .f32 log2_val;
                .reg .s32 exp_int, result;

                setp.le.f32 p_zero, $1, 0f00000000;
                lg2.approx.f32 log2_val, $1;
                cvt.rpi.s32.f32 exp_int, log2_val;
                add.s32 result, exp_int, 127;
                setp.lt.s32 p_neg, result, 0;
                setp.gt.s32 p_ovf, result, 255;
                selp.s32 result, 0, result, p_neg;
                selp.s32 result, 255, result, p_ovf;
                selp.s32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ue8m0_to_inv_scale_fast(ue8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    """
    Convert UE8M0 to inverse scale using fast ex2.approx.

    Inverse scale = 2^(127 - ue8m0)
    Returns 0 for ue8m0 == 0.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(ue8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .s32 neg_exp;
                .reg .f32 neg_exp_f, result;

                setp.eq.u32 p_zero, $1, 0;
                sub.s32 neg_exp, 127, $1;
                cvt.rn.f32.s32 neg_exp_f, neg_exp;
                ex2.approx.f32 result, neg_exp_f;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# =============================================================================
# FP8 Conversion with Scaling
# =============================================================================


@dsl_user_op
def half2_to_fp8x2_scaled(
    h2: Uint32, inv_scale: Float32, *, loc=None, ip=None
) -> Uint32:
    """Convert Half2 to 2 FP8 E4M3 values with scaling."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(h2).ir_value(loc=loc, ip=ip),
                Float32(inv_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 h0, h1;
                .reg .f32 f0, f1;
                .reg .b16 fp8_pair;

                mov.b32 {h0, h1}, $1;
                cvt.f32.f16 f0, h0;
                cvt.f32.f16 f1, h1;
                mul.f32 f0, f0, $2;
                mul.f32 f1, f1, $2;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, f1, f0;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bfloat2_to_fp8x2_scaled(
    bf2: Uint32, inv_scale: Float32, *, loc=None, ip=None
) -> Uint32:
    """Convert BFloat16x2 to 2 FP8 E4M3 values with scaling."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(bf2).ir_value(loc=loc, ip=ip),
                Float32(inv_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                .reg .b16 fp8_pair;

                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                mul.f32 f0, f0, $2;
                mul.f32 f1, f1, $2;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, f1, f0;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_fp8x8_to_u64(
    fp8_01: Uint32, fp8_23: Uint32, fp8_45: Uint32, fp8_67: Uint32, *, loc=None, ip=None
) -> Uint64:
    """Pack 8 FP8 values into a 64-bit value for vectorized store."""
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [
                Uint32(fp8_01).ir_value(loc=loc, ip=ip),
                Uint32(fp8_23).ir_value(loc=loc, ip=ip),
                Uint32(fp8_45).ir_value(loc=loc, ip=ip),
                Uint32(fp8_67).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                .reg .b32 t01, t23, t45, t67;

                and.b32 t01, $1, 0xFFFF;
                and.b32 t23, $2, 0xFFFF;
                and.b32 t45, $3, 0xFFFF;
                and.b32 t67, $4, 0xFFFF;

                shl.b32 t23, t23, 16;
                or.b32 lo, t01, t23;

                shl.b32 t67, t67, 16;
                or.b32 hi, t45, t67;

                mov.b64 $0, {lo, hi};
            }
            """,
            "=l,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# =============================================================================
# Warp Shuffle for 4-Thread Reduction
# =============================================================================


@cute.jit
def shuffle_xor_f32(val: Float32, offset: int) -> Float32:
    """XOR shuffle for float32 values."""
    return cute.arch.shuffle_sync_bfly(val, offset=offset)


@cute.jit
def reduce_max_4threads(val: Float32) -> Float32:
    """Reduce max across 4 consecutive threads using 2 XOR shuffles."""
    from ..cute_dsl.fp4_common import fmax_f32

    other = shuffle_xor_f32(val, 1)
    val = fmax_f32(val, other)
    other = shuffle_xor_f32(val, 2)
    val = fmax_f32(val, other)
    return val


# =============================================================================
# Swizzled Index Computation (GPU-side)
# =============================================================================


@cute.jit
def compute_sf_index_swizzled_128x4_gpu(
    row_idx: Int32,
    col_idx: Int32,
    padded_cols: Int32,
) -> Int32:
    """Compute swizzled 128x4 scale factor index on GPU."""
    kColumnGroup0Size = Int32(4)
    kRowGroup0Size = Int32(32)
    kRowGroup1Size = Int32(128)

    columnIdxInGroup0 = col_idx % kColumnGroup0Size
    columnGroupIdx = col_idx // kColumnGroup0Size
    columnGroupStride = Int32(512)

    rowIdxInGroup0 = row_idx % kRowGroup0Size
    rowIdxInGroup1 = (row_idx % kRowGroup1Size) // kRowGroup0Size
    rowGroupIdx = row_idx // kRowGroup1Size

    rowGroup1Stride = Int32(4)
    rowGroup0Stride = Int32(16)
    rowGroupStride = kRowGroup1Size * padded_cols

    offset = (
        columnIdxInGroup0
        + columnGroupIdx * columnGroupStride
        + rowIdxInGroup0 * rowGroup0Stride
        + rowIdxInGroup1 * rowGroup1Stride
        + rowGroupIdx * rowGroupStride
    )

    return offset


# =============================================================================
# High-Level Helper Functions for MXFP8 Quantization
# =============================================================================


@cute.jit
def half2_max_abs_4(v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32) -> Uint32:
    """
    Compute max absolute value across 4 half2 values (8 FP16 elements).

    Uses tree reduction: 4 -> 2 -> 1 half2 values.
    Returns a half2 containing the max absolute value in both lanes.
    """
    abs0 = habs2(v0)
    abs1 = habs2(v1)
    abs2 = habs2(v2)
    abs3 = habs2(v3)
    max01 = hmax2(abs0, abs1)
    max23 = hmax2(abs2, abs3)
    return hmax2(max01, max23)


@cute.jit
def bfloat2_max_abs_4(v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32) -> Uint32:
    """
    Compute max absolute value across 4 bfloat2 values (8 BF16 elements).

    Uses tree reduction: 4 -> 2 -> 1 bfloat2 values.
    Returns a bfloat2 containing the max absolute value in both lanes.
    """
    abs0 = bfloat2_habs2(v0)
    abs1 = bfloat2_habs2(v1)
    abs2 = bfloat2_habs2(v2)
    abs3 = bfloat2_habs2(v3)
    max01 = bfloat2_hmax2(abs0, abs1)
    max23 = bfloat2_hmax2(abs2, abs3)
    return bfloat2_hmax2(max01, max23)


@cute.jit
def half2x4_to_fp8x8_packed(
    v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, inv_scale: Float32
) -> Uint64:
    """
    Convert 4 half2 values (8 FP16) to 8 FP8 E4M3 and pack into u64.

    Each half2 is converted to 2 FP8 values using the inverse scale,
    then all 8 FP8 values are packed into a single 64-bit value for
    efficient vectorized store.
    """
    fp8_01 = half2_to_fp8x2_scaled(v0, inv_scale)
    fp8_23 = half2_to_fp8x2_scaled(v1, inv_scale)
    fp8_45 = half2_to_fp8x2_scaled(v2, inv_scale)
    fp8_67 = half2_to_fp8x2_scaled(v3, inv_scale)
    return pack_fp8x8_to_u64(fp8_01, fp8_23, fp8_45, fp8_67)


@cute.jit
def bfloat2x4_to_fp8x8_packed(
    v0: Uint32, v1: Uint32, v2: Uint32, v3: Uint32, inv_scale: Float32
) -> Uint64:
    """
    Convert 4 bfloat2 values (8 BF16) to 8 FP8 E4M3 and pack into u64.

    Each bfloat2 is converted to 2 FP8 values using the inverse scale,
    then all 8 FP8 values are packed into a single 64-bit value for
    efficient vectorized store.
    """
    fp8_01 = bfloat2_to_fp8x2_scaled(v0, inv_scale)
    fp8_23 = bfloat2_to_fp8x2_scaled(v1, inv_scale)
    fp8_45 = bfloat2_to_fp8x2_scaled(v2, inv_scale)
    fp8_67 = bfloat2_to_fp8x2_scaled(v3, inv_scale)
    return pack_fp8x8_to_u64(fp8_01, fp8_23, fp8_45, fp8_67)


__all__ = [
    # Constants
    "SF_VEC_SIZE",
    "INV_FLOAT8_E4M3_MAX",
    "WARP_SIZE",
    "ELTS_PER_THREAD",
    "THREADS_PER_SF",
    "SF_BLOCKS_PER_WARP",
    "ROW_TILE_SIZE",
    # Low-level intrinsics (used directly by kernels)
    "hmax_reduce_to_f32",
    "bfloat2_hmax_reduce_to_f32",
    "float_to_ue8m0_fast",
    "ue8m0_to_inv_scale_fast",
    "reduce_max_4threads",
    "compute_sf_index_swizzled_128x4_gpu",
    # High-level helper functions
    "half2_max_abs_4",
    "bfloat2_max_abs_4",
    "half2x4_to_fp8x8_packed",
    "bfloat2x4_to_fp8x8_packed",
]
