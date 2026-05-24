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

# Default: optimized 2-thread-per-SF configuration for large problems
ELTS_PER_THREAD = 16  # Each thread handles 16 FP16 elements (2 × 128-bit loads)
THREADS_PER_SF = SF_VEC_SIZE // ELTS_PER_THREAD  # 32 / 16 = 2 threads per SF block
SF_BLOCKS_PER_WARP = WARP_SIZE // THREADS_PER_SF  # 32 / 2 = 16 SF blocks per warp

# Legacy: 4-thread-per-SF configuration for small problems (better grid occupancy)
ELTS_PER_THREAD_SMALL = 8
THREADS_PER_SF_SMALL = SF_VEC_SIZE // ELTS_PER_THREAD_SMALL  # 32 / 8 = 4
SF_BLOCKS_PER_WARP_SMALL = WARP_SIZE // THREADS_PER_SF_SMALL  # 32 / 4 = 8

# Threshold: use 2T/SF when total_sf_blocks >= this value (M*K >= 2M elements)
MXFP8_2T_SF_THRESHOLD = 65536

# Row tiling for swizzled layout (128x4 pattern)
ROW_TILE_SIZE = 128


# =============================================================================
# NVFP4 Constants
# =============================================================================

# Scale factor vector size for NVFP4: each scale factor covers 16 elements
NVFP4_SF_VEC_SIZE = 16

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
    Convert float to UE8M0 format using exact IEEE 754 bit manipulation.

    Matches the hardware __nv_cvt_float_to_e8m0(value, __NV_SATFINITE, cudaRoundPosInf):
    - Extract biased exponent from IEEE 754 float
    - If mantissa is nonzero, add 1 (round towards +inf / ceil behavior)
    - Clamp to [0, 254] (255 = NaN in E8M0)
    - Return 0 for zero/negative input
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_has_mant, p_ovf;
                .reg .u32 bits, exp_biased, mantissa, bump, result;

                setp.le.f32 p_zero, $1, 0f00000000;

                mov.b32 bits, $1;
                shr.b32 exp_biased, bits, 23;
                and.b32 exp_biased, exp_biased, 255;
                and.b32 mantissa, bits, 0x7FFFFF;

                setp.ne.u32 p_has_mant, mantissa, 0;
                selp.u32 bump, 1, 0, p_has_mant;
                add.u32 result, exp_biased, bump;

                setp.gt.u32 p_ovf, result, 254;
                selp.u32 result, 254, result, p_ovf;
                selp.u32 $0, 0, result, p_zero;
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
    Convert UE8M0 to inverse scale using integer bit construction.

    Constructs a float32 with exponent = (254 - ue8m0) and zero mantissa,
    which is exactly 2^(127 - ue8m0). No SFU dependency.
    Returns 0 for ue8m0 == 0.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(ue8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .s32 new_exp;
                .reg .b32 float_bits;
                .reg .pred p_zero;

                setp.eq.u32 p_zero, $1, 0;
                sub.s32 new_exp, 254, $1;
                max.s32 new_exp, new_exp, 0;
                shl.b32 float_bits, new_exp, 23;
                mov.b32 $0, float_bits;
                @p_zero mov.b32 $0, 0;
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
def reduce_max_2threads(val: Float32) -> Float32:
    """Reduce max across 2 consecutive threads using 1 XOR shuffle."""
    from ..cute_dsl.fp4_common import fmax_f32

    other = shuffle_xor_f32(val, 1)
    val = fmax_f32(val, other)
    return val


@cute.jit
def reduce_max_4threads(val: Float32) -> Float32:
    """Reduce max across 4 consecutive threads using 2 XOR shuffles.

    Kept for backward compatibility with MXFP4 kernels.
    """
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


@cute.jit
def compute_sf_index_swizzled_8x4_gpu(
    row_idx: Int32,
    col_idx: Int32,
    padded_cols: Int32,
) -> Int32:
    """Compute swizzled 8x4 scale factor index on GPU.

    Layout: [numMTiles, numKTiles, 8 (mTile), 4 (kTile)]
    Tile size: 32 elements (8 rows x 4 cols).
    """
    kMTileSize = Int32(8)
    kKTileSize = Int32(4)
    kTileElements = Int32(32)

    innerKIdx = col_idx % kKTileSize
    innerMIdx = row_idx % kMTileSize
    kTileIdx = col_idx // kKTileSize
    mTileIdx = row_idx // kMTileSize

    numKTiles = (padded_cols + kKTileSize - Int32(1)) // kKTileSize

    offset = (
        mTileIdx * (numKTiles * kTileElements)
        + kTileIdx * kTileElements
        + innerMIdx * kKTileSize
        + innerKIdx
    )

    return offset


@cute.jit
def compute_sf_index_linear_gpu(
    row_idx: Int32,
    col_idx: Int32,
    num_cols: Int32,
) -> Int32:
    """Compute linear (row-major) scale factor index on GPU."""
    return row_idx * num_cols + col_idx


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

    from ..cute_dsl.fp4_common import (
        get_ptr_as_int64,
        hmax2,
        ld_global_v4_u32,
        rcp_approx_ftz,
    )

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

    # Compute UE8M0 scale factor (rcp_approx matches CUDA's rcp.approx.ftz(6.0f))
    normalized_max = block_max * rcp_approx_ftz(Float32(6.0))
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

    from ..cute_dsl.fp4_common import (
        bfloat2_hmax2,
        get_ptr_as_int64,
        ld_global_v4_u32,
        rcp_approx_ftz,
    )

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

    # Compute UE8M0 scale factor (rcp_approx matches CUDA's rcp.approx.ftz(6.0f))
    normalized_max = block_max * rcp_approx_ftz(Float32(6.0))
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


# =============================================================================
# NVFP4 High-Level Helper Functions (sf_vec_size=16, E4M3 scale factors)
# =============================================================================


@cute.jit
def half2x8_to_e2m1x16_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
) -> Uint64:
    """
    Convert 8 half2 values (16 FP16) to 16 E2M1 and pack into u64.

    Returns:
        Uint64 containing 16 E2M1 values (8 bytes)
    """
    s0, s1 = half2_to_float2_scaled(h0, inv_scale)
    s2, s3 = half2_to_float2_scaled(h1, inv_scale)
    s4, s5 = half2_to_float2_scaled(h2, inv_scale)
    s6, s7 = half2_to_float2_scaled(h3, inv_scale)
    s8, s9 = half2_to_float2_scaled(h4, inv_scale)
    s10, s11 = half2_to_float2_scaled(h5, inv_scale)
    s12, s13 = half2_to_float2_scaled(h6, inv_scale)
    s14, s15 = half2_to_float2_scaled(h7, inv_scale)

    packed_lo = cvt_e2m1x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed_hi = cvt_e2m1x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)

    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def bfloat2x8_to_e2m1x16_packed(
    h0: Uint32,
    h1: Uint32,
    h2: Uint32,
    h3: Uint32,
    h4: Uint32,
    h5: Uint32,
    h6: Uint32,
    h7: Uint32,
    inv_scale: Float32,
) -> Uint64:
    """
    Convert 8 bfloat2 values (16 BF16) to 16 E2M1 and pack into u64.

    Returns:
        Uint64 containing 16 E2M1 values (8 bytes)
    """
    s0, s1 = bfloat2_to_float2_scaled(h0, inv_scale)
    s2, s3 = bfloat2_to_float2_scaled(h1, inv_scale)
    s4, s5 = bfloat2_to_float2_scaled(h2, inv_scale)
    s6, s7 = bfloat2_to_float2_scaled(h3, inv_scale)
    s8, s9 = bfloat2_to_float2_scaled(h4, inv_scale)
    s10, s11 = bfloat2_to_float2_scaled(h5, inv_scale)
    s12, s13 = bfloat2_to_float2_scaled(h6, inv_scale)
    s14, s15 = bfloat2_to_float2_scaled(h7, inv_scale)

    packed_lo = cvt_e2m1x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed_hi = cvt_e2m1x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)

    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def process_nvfp4_block_half(
    row_tensor, elem_base: Int32, global_scale: Float32
) -> tuple:
    """
    Process a 16-element NVFP4 block for half precision input.

    Loads 16 FP16 elements, computes the E4M3 scale factor using global_scale,
    converts to E2M1, and packs the result into a u64 value.

    Args:
        row_tensor: Row tensor slice (mInput[row_idx, None])
        elem_base: Starting element index
        global_scale: User-provided global scale factor

    Returns:
        (scale_e4m3_u8, packed64):
        - scale_e4m3_u8: E4M3 scale factor as Uint8
        - packed64: Uint64 containing 16 E2M1 values
    """
    from cutlass import Uint8

    from ..cute_dsl.fp4_common import (
        cvt_f32_to_e4m3,
        get_ptr_as_int64,
        ld_global_v4_u32,
        nvfp4_compute_output_scale,
        rcp_approx_ftz,
    )

    # Load 16 elements (2 x 128-bit = 8 half2 values)
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    # Compute max absolute value across 16 elements
    block_max_h2 = half2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = hmax_reduce_to_f32(block_max_h2)

    # E4M3 scale factor computation
    fp4_max_rcp = rcp_approx_ftz(Float32(6.0))
    scale_float = global_scale * (block_max * fp4_max_rcp)
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))

    # output_scale = rcp(float(E4M3(scale)) * rcp(global_scale)), matching CUDA
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

    # Convert to E2M1 and pack
    packed64 = half2x8_to_e2m1x16_packed(h0, h1, h2, h3, h4, h5, h6, h7, output_scale)

    return scale_fp8, packed64


@cute.jit
def process_nvfp4_block_bfloat(
    row_tensor, elem_base: Int32, global_scale: Float32
) -> tuple:
    """
    Process a 16-element NVFP4 block for bfloat16 precision input.

    Loads 16 BF16 elements, computes the E4M3 scale factor using global_scale,
    converts to E2M1, and packs the result into a u64 value.

    Args:
        row_tensor: Row tensor slice (mInput[row_idx, None])
        elem_base: Starting element index
        global_scale: User-provided global scale factor

    Returns:
        (scale_e4m3_u8, packed64):
        - scale_e4m3_u8: E4M3 scale factor as Uint8
        - packed64: Uint64 containing 16 E2M1 values
    """
    from cutlass import Uint8

    from ..cute_dsl.fp4_common import (
        cvt_f32_to_e4m3,
        get_ptr_as_int64,
        ld_global_v4_u32,
        nvfp4_compute_output_scale,
        rcp_approx_ftz,
    )

    # Load 16 elements (2 x 128-bit = 8 bfloat2 values)
    ptr0 = get_ptr_as_int64(row_tensor, elem_base)
    ptr1 = get_ptr_as_int64(row_tensor, elem_base + Int32(8))

    h0, h1, h2, h3 = ld_global_v4_u32(ptr0)
    h4, h5, h6, h7 = ld_global_v4_u32(ptr1)

    # Compute max absolute value across 16 elements
    block_max_h2 = bfloat2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = bfloat2_hmax_reduce_to_f32(block_max_h2)

    # E4M3 scale factor computation
    fp4_max_rcp = rcp_approx_ftz(Float32(6.0))
    scale_float = global_scale * (block_max * fp4_max_rcp)
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))

    # output_scale = rcp(float(E4M3(scale)) * rcp(global_scale)), matching CUDA
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

    # Convert to E2M1 and pack
    packed64 = bfloat2x8_to_e2m1x16_packed(h0, h1, h2, h3, h4, h5, h6, h7, output_scale)

    return scale_fp8, packed64


@cute.jit
def fp8x16_to_e2m1x16_packed(
    w0: Uint32,
    w1: Uint32,
    w2: Uint32,
    w3: Uint32,
    output_scale: Float32,
) -> Uint64:
    """Convert 16 packed FP8 E4M3 values (4 x uint32) to 16 E2M1 values packed as Uint64.

    Each uint32 contains 4 E4M3 bytes. Output is 16 E2M1 nibbles packed into 8 bytes.
    """
    from ..cute_dsl.fp4_common import cvt_e4m3x4_to_f32x4

    f0, f1, f2, f3 = cvt_e4m3x4_to_f32x4(w0)
    f4, f5, f6, f7 = cvt_e4m3x4_to_f32x4(w1)
    f8, f9, f10, f11 = cvt_e4m3x4_to_f32x4(w2)
    f12, f13, f14, f15 = cvt_e4m3x4_to_f32x4(w3)

    s0 = f0 * output_scale
    s1 = f1 * output_scale
    s2 = f2 * output_scale
    s3 = f3 * output_scale
    s4 = f4 * output_scale
    s5 = f5 * output_scale
    s6 = f6 * output_scale
    s7 = f7 * output_scale
    s8 = f8 * output_scale
    s9 = f9 * output_scale
    s10 = f10 * output_scale
    s11 = f11 * output_scale
    s12 = f12 * output_scale
    s13 = f13 * output_scale
    s14 = f14 * output_scale
    s15 = f15 * output_scale

    packed_lo = cvt_e2m1x8_f32(s0, s1, s2, s3, s4, s5, s6, s7)
    packed_hi = cvt_e2m1x8_f32(s8, s9, s10, s11, s12, s13, s14, s15)

    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def fp8_max_abs_16(w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32) -> Float32:
    """Compute max absolute value across 16 FP8 E4M3 values (4 x uint32).

    Converts all 16 values to float32, takes abs, and reduces to a single max.
    """
    from ..cute_dsl.fp4_common import cvt_e4m3x4_to_f32x4

    f0, f1, f2, f3 = cvt_e4m3x4_to_f32x4(w0)
    f4, f5, f6, f7 = cvt_e4m3x4_to_f32x4(w1)
    f8, f9, f10, f11 = cvt_e4m3x4_to_f32x4(w2)
    f12, f13, f14, f15 = cvt_e4m3x4_to_f32x4(w3)

    from ..cute_dsl.fp4_common import fabs_f32, fmax_f32

    a0 = fabs_f32(f0)
    a1 = fabs_f32(f1)
    a2 = fabs_f32(f2)
    a3 = fabs_f32(f3)
    a4 = fabs_f32(f4)
    a5 = fabs_f32(f5)
    a6 = fabs_f32(f6)
    a7 = fabs_f32(f7)
    a8 = fabs_f32(f8)
    a9 = fabs_f32(f9)
    a10 = fabs_f32(f10)
    a11 = fabs_f32(f11)
    a12 = fabs_f32(f12)
    a13 = fabs_f32(f13)
    a14 = fabs_f32(f14)
    a15 = fabs_f32(f15)

    m01 = fmax_f32(a0, a1)
    m23 = fmax_f32(a2, a3)
    m45 = fmax_f32(a4, a5)
    m67 = fmax_f32(a6, a7)
    m89 = fmax_f32(a8, a9)
    m1011 = fmax_f32(a10, a11)
    m1213 = fmax_f32(a12, a13)
    m1415 = fmax_f32(a14, a15)

    m0123 = fmax_f32(m01, m23)
    m4567 = fmax_f32(m45, m67)
    m891011 = fmax_f32(m89, m1011)
    m12131415 = fmax_f32(m1213, m1415)

    m_lo = fmax_f32(m0123, m4567)
    m_hi = fmax_f32(m891011, m12131415)

    return fmax_f32(m_lo, m_hi)


@cute.jit
def process_nvfp4_block_fp8(
    row_tensor, elem_base: Int32, global_scale: Float32
) -> tuple:
    """
    Process a 16-element NVFP4 block for FP8 E4M3 input.

    Matches the CUDA cvt_warp_fp8_to_fp4 behavior: FP8 values are first converted
    to float32, pre-scaled by 6/global_scale, and converted to half2. From there,
    the standard half2 pipeline is used for max-abs reduction, scale factor
    computation, and E2M1 conversion.

    Args:
        row_tensor: Row tensor slice (mInput[row_idx, None])
        elem_base: Starting element index
        global_scale: User-provided global scale factor

    Returns:
        (scale_e4m3_u8, packed64):
        - scale_e4m3_u8: E4M3 scale factor as Uint8
        - packed64: Uint64 containing 16 E2M1 values
    """
    from cutlass import Uint8

    from ..cute_dsl.fp4_common import (
        cvt_e4m3x4_to_f32x4,
        cvt_f32_to_e4m3,
        cvt_f32x2_to_half2,
        get_ptr_as_int64,
        ld_global_v4_u32,
        nvfp4_compute_output_scale,
        rcp_approx_ftz,
    )

    # Load 16 FP8 elements (1 x 128-bit = 4 x uint32 = 16 bytes)
    ptr = get_ptr_as_int64(row_tensor, elem_base)
    w0, w1, w2, w3 = ld_global_v4_u32(ptr)

    # Convert FP8 to float32 and pre-scale by 6/global_scale (matching CUDA)
    prescale = Float32(6.0) * rcp_approx_ftz(global_scale)

    f0, f1, f2, f3 = cvt_e4m3x4_to_f32x4(w0)
    f4, f5, f6, f7 = cvt_e4m3x4_to_f32x4(w1)
    f8, f9, f10, f11 = cvt_e4m3x4_to_f32x4(w2)
    f12, f13, f14, f15 = cvt_e4m3x4_to_f32x4(w3)

    # Pack pre-scaled float pairs into half2 (matching __float22half2_rn in CUDA)
    h0 = cvt_f32x2_to_half2(f0 * prescale, f1 * prescale)
    h1 = cvt_f32x2_to_half2(f2 * prescale, f3 * prescale)
    h2 = cvt_f32x2_to_half2(f4 * prescale, f5 * prescale)
    h3 = cvt_f32x2_to_half2(f6 * prescale, f7 * prescale)
    h4 = cvt_f32x2_to_half2(f8 * prescale, f9 * prescale)
    h5 = cvt_f32x2_to_half2(f10 * prescale, f11 * prescale)
    h6 = cvt_f32x2_to_half2(f12 * prescale, f13 * prescale)
    h7 = cvt_f32x2_to_half2(f14 * prescale, f15 * prescale)

    # From here, use the same half2 pipeline as process_nvfp4_block_half
    block_max_h2 = half2_max_abs_8(h0, h1, h2, h3, h4, h5, h6, h7)
    block_max = hmax_reduce_to_f32(block_max_h2)

    # E4M3 scale factor computation
    fp4_max_rcp = rcp_approx_ftz(Float32(6.0))
    scale_float = global_scale * (block_max * fp4_max_rcp)
    scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
    scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))

    # output_scale = rcp(float(E4M3(scale)) * rcp(global_scale)), matching CUDA
    output_scale = nvfp4_compute_output_scale(scale_fp8_u32, global_scale)

    # Convert pre-scaled half2 values to E2M1 and pack
    packed64 = half2x8_to_e2m1x16_packed(h0, h1, h2, h3, h4, h5, h6, h7, output_scale)

    return scale_fp8, packed64


__all__ = [
    # NVFP4 Constants
    "NVFP4_SF_VEC_SIZE",
    # MXFP8 Constants
    "SF_VEC_SIZE",
    "INV_FLOAT8_E4M3_MAX",
    "WARP_SIZE",
    "ELTS_PER_THREAD",
    "THREADS_PER_SF",
    "SF_BLOCKS_PER_WARP",
    "ELTS_PER_THREAD_SMALL",
    "THREADS_PER_SF_SMALL",
    "SF_BLOCKS_PER_WARP_SMALL",
    "MXFP8_2T_SF_THRESHOLD",
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
    "reduce_max_2threads",
    "reduce_max_4threads",
    "compute_sf_index_swizzled_128x4_gpu",
    "compute_sf_index_swizzled_8x4_gpu",
    "compute_sf_index_linear_gpu",
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
    # High-level helper functions (NVFP4)
    "half2x8_to_e2m1x16_packed",
    "bfloat2x8_to_e2m1x16_packed",
    "process_nvfp4_block_half",
    "process_nvfp4_block_bfloat",
    # High-level helper functions (NVFP4 - FP8 input)
    "fp8x16_to_e2m1x16_packed",
    "fp8_max_abs_16",
    "process_nvfp4_block_fp8",
]
