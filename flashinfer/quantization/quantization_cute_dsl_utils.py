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
and MXFP4 quantization kernels.
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
# MXFP4 Constants
# =============================================================================

# Scale factor vector size for MXFP4: each scale factor covers 32 elements
MXFP4_SF_VEC_SIZE = 32

# Elements per thread for MXFP4: each thread handles 32 elements (one full SF block)
MXFP4_ELTS_PER_THREAD = 32

# Inverse of max representable value in FP4 E2M1 format (1/6)
INV_FLOAT4_E2M1_MAX = 1.0 / 6.0

# Global scale factor for MXFP4: 448 * 6 = 2688
MXFP4_GLOBAL_SCALE_FACTOR = 448.0 * 6.0


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
# E2M1 (FP4) Conversion for MXFP4
# =============================================================================


@dsl_user_op
def half2_to_float2_scaled(
    h2: Uint32, scale: Float32, *, loc=None, ip=None
) -> tuple[Float32, Float32]:
    """Convert Half2 to Float2 AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(h2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 h0, h1;
            .reg .f32 f0, f1;
            mov.b32 {h0, h1}, $2;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


@dsl_user_op
def bfloat2_to_float2_scaled(
    bf2: Uint32, scale: Float32, *, loc=None, ip=None
) -> tuple[Float32, Float32]:
    """Convert BFloat16x2 to Float2 AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


@dsl_user_op
def cvt_e2m1x8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """
    Convert eight float32 values to eight E2M1 (4-bit) values packed into uint32.

    Uses cvt.rn.satfinite.e2m1x2.f32 PTX instruction to convert pairs of f32
    to pairs of 4-bit E2M1 values, then packs all 8 values into a single u32.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
                Float32(v4).ir_value(loc=loc, ip=ip),
                Float32(v5).ir_value(loc=loc, ip=ip),
                Float32(v6).ir_value(loc=loc, ip=ip),
                Float32(v7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
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


# =============================================================================
# MXFP4 High-Level Helper Functions
# =============================================================================


@cute.jit
def half2_max_abs_8(
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    v4: Uint32,
    v5: Uint32,
    v6: Uint32,
    v7: Uint32,
) -> Uint32:
    """
    Compute max absolute value across 8 half2 values (16 FP16 elements).

    Uses tree reduction: 8 -> 4 -> 2 -> 1 half2 values.
    Returns a half2 containing the max absolute value in both lanes.
    """
    abs0 = habs2(v0)
    abs1 = habs2(v1)
    abs2 = habs2(v2)
    abs3 = habs2(v3)
    abs4 = habs2(v4)
    abs5 = habs2(v5)
    abs6 = habs2(v6)
    abs7 = habs2(v7)

    max01 = hmax2(abs0, abs1)
    max23 = hmax2(abs2, abs3)
    max45 = hmax2(abs4, abs5)
    max67 = hmax2(abs6, abs7)

    max0123 = hmax2(max01, max23)
    max4567 = hmax2(max45, max67)

    return hmax2(max0123, max4567)


@cute.jit
def bfloat2_max_abs_8(
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    v4: Uint32,
    v5: Uint32,
    v6: Uint32,
    v7: Uint32,
) -> Uint32:
    """
    Compute max absolute value across 8 bfloat2 values (16 BF16 elements).

    Uses tree reduction: 8 -> 4 -> 2 -> 1 bfloat2 values.
    Returns a bfloat2 containing the max absolute value in both lanes.
    """
    abs0 = bfloat2_habs2(v0)
    abs1 = bfloat2_habs2(v1)
    abs2 = bfloat2_habs2(v2)
    abs3 = bfloat2_habs2(v3)
    abs4 = bfloat2_habs2(v4)
    abs5 = bfloat2_habs2(v5)
    abs6 = bfloat2_habs2(v6)
    abs7 = bfloat2_habs2(v7)

    max01 = bfloat2_hmax2(abs0, abs1)
    max23 = bfloat2_hmax2(abs2, abs3)
    max45 = bfloat2_hmax2(abs4, abs5)
    max67 = bfloat2_hmax2(abs6, abs7)

    max0123 = bfloat2_hmax2(max01, max23)
    max4567 = bfloat2_hmax2(max45, max67)

    return bfloat2_hmax2(max0123, max4567)


@cute.jit
def process_mxfp4_block_half(row_tensor, elem_base: Int32) -> tuple:
    """
    Process a 32-element MXFP4 block for half precision input.

    Loads 32 FP16 elements, computes the UE8M0 scale factor, converts to E2M1,
    and packs the result into two u64 values.

    Args:
        row_tensor: Row tensor slice (mInput[row_idx, None])
        elem_base: Starting element index

    Returns:
        (scale_ue8m0_u32, scale_ue8m0_u8, packed64_0, packed64_1):
        - scale_ue8m0_u32: Scale factor as Uint32 (for inv_scale computation)
        - scale_ue8m0_u8: Scale factor as Uint8 (for storage)
        - packed64_0, packed64_1: Two Uint64 containing 16 E2M1 values each
    """
    from cutlass import Uint8

    from ..cute_dsl.fp4_common import get_ptr_as_int64, hmax2, ld_global_v4_u32

    # Load 32 elements (4 x 128-bit = 16 half2 values)
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
    ptr2 = get_ptr_as_int64(row_tensor, elem_base + Int32(16))
    ptr3 = get_ptr_as_int64(row_tensor, elem_base + Int32(24))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
    h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
    h12, h13, h14, h15 = ld_global_v4_u32(ptr3)

    # Compute max absolute value across 32 elements
    max_first = half2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    max_second = half2_max_abs_8(h8, h9, h10, h11, h12, h13, h14, h15)
    block_max_h2 = hmax2(max_first, max_second)
    block_max = hmax_reduce_to_f32(block_max_h2)

    # Compute UE8M0 scale factor
    inv_e2m1_max = Float32(INV_FLOAT4_E2M1_MAX)
    normalized_max = block_max * inv_e2m1_max
    scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
    scale_ue8m0_u8 = scale_ue8m0_u32.to(Uint8)

    # Compute inverse scale and convert to E2M1 packed format
    inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)
    packed64_0, packed64_1 = half2x16_to_e2m1x32_packed(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        h8,
        h9,
        h10,
        h11,
        h12,
        h13,
        h14,
        h15,
        inv_scale,
    )

    return scale_ue8m0_u32, scale_ue8m0_u8, packed64_0, packed64_1


@cute.jit
def process_mxfp4_block_bfloat(row_tensor, elem_base: Int32) -> tuple:
    """
    Process a 32-element MXFP4 block for bfloat16 precision input.

    Loads 32 BF16 elements, computes the UE8M0 scale factor, converts to E2M1,
    and packs the result into two u64 values.

    Args:
        row_tensor: Row tensor slice (mInput[row_idx, None])
        elem_base: Starting element index

    Returns:
        (scale_ue8m0_u32, scale_ue8m0_u8, packed64_0, packed64_1):
        - scale_ue8m0_u32: Scale factor as Uint32 (for inv_scale computation)
        - scale_ue8m0_u8: Scale factor as Uint8 (for storage)
        - packed64_0, packed64_1: Two Uint64 containing 16 E2M1 values each
    """
    from cutlass import Uint8

    from ..cute_dsl.fp4_common import bfloat2_hmax2, get_ptr_as_int64, ld_global_v4_u32

    # Load 32 elements (4 x 128-bit = 16 bfloat2 values)
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
    ptr2 = get_ptr_as_int64(row_tensor, elem_base + Int32(16))
    ptr3 = get_ptr_as_int64(row_tensor, elem_base + Int32(24))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)
    h8, h9, h10, h11 = ld_global_v4_u32(ptr2)
    h12, h13, h14, h15 = ld_global_v4_u32(ptr3)

    # Compute max absolute value across 32 elements
    max_first = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    max_second = bfloat2_max_abs_8(h8, h9, h10, h11, h12, h13, h14, h15)
    block_max_h2 = bfloat2_hmax2(max_first, max_second)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    # Compute UE8M0 scale factor
    inv_e2m1_max = Float32(INV_FLOAT4_E2M1_MAX)
    normalized_max = block_max * inv_e2m1_max
    scale_ue8m0_u32 = float_to_ue8m0_fast(normalized_max)
    scale_ue8m0_u8 = scale_ue8m0_u32.to(Uint8)

    # Compute inverse scale and convert to E2M1 packed format
    inv_scale = ue8m0_to_inv_scale_fast(scale_ue8m0_u32)
    packed64_0, packed64_1 = bfloat2x16_to_e2m1x32_packed(
        h0,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        h7,
        h8,
        h9,
        h10,
        h11,
        h12,
        h13,
        h14,
        h15,
        inv_scale,
    )

    return scale_ue8m0_u32, scale_ue8m0_u8, packed64_0, packed64_1


@cute.jit
def ld_32_elements(row_tensor, elem_base: Int32) -> tuple:
    """
    Load 32 elements (16 half2/bfloat2 values) from a row tensor.

    This loads 4 x 128-bit vectors (4 x v4_u32) starting at elem_base.

    Args:
        row_tensor: Row tensor slice (mInput[row_idx, None])
        elem_base: Starting element index

    Returns:
        Tuple of 16 Uint32 values (h0-h15), each containing 2 fp16/bf16 elements
    """
    from ..cute_dsl.fp4_common import get_ptr_as_int64, ld_global_v4_u32

    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))
    ptr2 = get_ptr_as_int64(row_tensor, elem_base + Int32(16))
    ptr3 = get_ptr_as_int64(row_tensor, elem_base + Int32(24))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)  # Elements 0-7
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)  # Elements 8-15
    h8, h9, h10, h11 = ld_global_v4_u32(ptr2)  # Elements 16-23
    h12, h13, h14, h15 = ld_global_v4_u32(ptr3)  # Elements 24-31

    return h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15


@cute.jit
def half2x16_to_e2m1x32_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    h8: Uint32,
    h9: Uint32,
    h10: Uint32,
    h11: Uint32,
    h12: Uint32,
    h13: Uint32,
    h14: Uint32,
    h15: Uint32,
    inv_scale: Float32,
) -> tuple:
    """
    Convert 16 half2 values (32 FP16) to 32 E2M1 and pack into two u64.

    Each half2 is converted to 2 float32 values using inv_scale,
    then groups of 8 floats are converted to 8 E2M1 values packed into u32,
    and finally combined into two u64 values for vectorized store.

    Returns:
        (packed64_0, packed64_1): Two Uint64 containing 16 E2M1 values each
    """
    # Scale and convert each half2 to 2 float32
    s0, s1 = half2_to_float2_scaled(h0, inv_scale)
    s2, s3 = half2_to_float2_scaled(h1, inv_scale)
    s4, s5 = half2_to_float2_scaled(h2, inv_scale)
    s6, s7 = half2_to_float2_scaled(h3, inv_scale)
    s8, s9 = half2_to_float2_scaled(h4, inv_scale)
    s10, s11 = half2_to_float2_scaled(h5, inv_scale)
    s12, s13 = half2_to_float2_scaled(h6, inv_scale)
    s14, s15 = half2_to_float2_scaled(h7, inv_scale)
    s16, s17 = half2_to_float2_scaled(h8, inv_scale)
    s18, s19 = half2_to_float2_scaled(h9, inv_scale)
    s20, s21 = half2_to_float2_scaled(h10, inv_scale)
    s22, s23 = half2_to_float2_scaled(h11, inv_scale)
    s24, s25 = half2_to_float2_scaled(h12, inv_scale)
    s26, s27 = half2_to_float2_scaled(h13, inv_scale)
    s28, s29 = half2_to_float2_scaled(h14, inv_scale)
    s30, s31 = half2_to_float2_scaled(h15, inv_scale)

    # Convert to E2M1 (4 x 8 floats -> 4 x uint32)
    packed0 = cvt_e2m1x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed1 = cvt_e2m1x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)
    packed2 = cvt_e2m1x8_f32(s16, s17, s18, s19, s20, s21, s22, s23)
    packed3 = cvt_e2m1x8_f32(s24, s25, s26, s27, s28, s29, s30, s31)

    # Pack into 2 x 64-bit values
    packed64_0 = (Uint64(packed1) << Uint64(32)) | Uint64(packed0)
    packed64_1 = (Uint64(packed3) << Uint64(32)) | Uint64(packed2)

    return packed64_0, packed64_1


@cute.jit
def bfloat2x16_to_e2m1x32_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    h8: Uint32,
    h9: Uint32,
    h10: Uint32,
    h11: Uint32,
    h12: Uint32,
    h13: Uint32,
    h14: Uint32,
    h15: Uint32,
    inv_scale: Float32,
) -> tuple:
    """
    Convert 16 bfloat2 values (32 BF16) to 32 E2M1 and pack into two u64.

    Each bfloat2 is converted to 2 float32 values using inv_scale,
    then groups of 8 floats are converted to 8 E2M1 values packed into u32,
    and finally combined into two u64 values for vectorized store.

    Returns:
        (packed64_0, packed64_1): Two Uint64 containing 16 E2M1 values each
    """
    # Scale and convert each bfloat2 to 2 float32
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)
    s16, s17 = bfloat2_to_float2_scaled(h8, inv_scale)
    s18, s19 = bfloat2_to_float2_scaled(h9, inv_scale)
    s20, s21 = bfloat2_to_float2_scaled(h10, inv_scale)
    s22, s23 = bfloat2_to_float2_scaled(h11, inv_scale)
    s24, s25 = bfloat2_to_float2_scaled(h12, inv_scale)
    s26, s27 = bfloat2_to_float2_scaled(h13, inv_scale)
    s28, s29 = bfloat2_to_float2_scaled(h14, inv_scale)
    s30, s31 = bfloat2_to_float2_scaled(h15, inv_scale)

    # Convert to E2M1 (4 x 8 floats -> 4 x uint32)
    packed0 = cvt_e2m1x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed1 = cvt_e2m1x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)
    packed2 = cvt_e2m1x8_f32(s16, s17, s18, s19, s20, s21, s22, s23)
    packed3 = cvt_e2m1x8_f32(s24, s25, s26, s27, s28, s29, s30, s31)

    # Pack into 2 x 64-bit values
    packed64_0 = (Uint64(packed1) << Uint64(32)) | Uint64(packed0)
    packed64_1 = (Uint64(packed3) << Uint64(32)) | Uint64(packed2)

    return packed64_0, packed64_1


__all__ = [
    # MXFP8 Constants
    "SF_VEC_SIZE",
    "INV_FLOAT8_E4M3_MAX",
    "WARP_SIZE",
    "ELTS_PER_THREAD",
    "THREADS_PER_SF",
    "SF_BLOCKS_PER_WARP",
    "ROW_TILE_SIZE",
    # MXFP4 Constants
    "MXFP4_SF_VEC_SIZE",
    "MXFP4_ELTS_PER_THREAD",
    "INV_FLOAT4_E2M1_MAX",
    "MXFP4_GLOBAL_SCALE_FACTOR",
    # Low-level intrinsics (MXFP8)
    "hmax_reduce_to_f32",
    "bfloat2_hmax_reduce_to_f32",
    "float_to_ue8m0_fast",
    "ue8m0_to_inv_scale_fast",
    "reduce_max_4threads",
    "compute_sf_index_swizzled_128x4_gpu",
    # Low-level intrinsics (MXFP4 - E2M1 conversion)
    "half2_to_float2_scaled",
    "bfloat2_to_float2_scaled",
    "cvt_e2m1x8_f32",
    # High-level helper functions (MXFP8)
    "half2_max_abs_4",
    "bfloat2_max_abs_4",
    "half2x4_to_fp8x8_packed",
    "bfloat2x4_to_fp8x8_packed",
    # High-level helper functions (MXFP4)
    "half2_max_abs_8",
    "bfloat2_max_abs_8",
    "process_mxfp4_block_half",
    "process_mxfp4_block_bfloat",
    "ld_32_elements",
    "half2x16_to_e2m1x32_packed",
    "bfloat2x16_to_e2m1x32_packed",
]
