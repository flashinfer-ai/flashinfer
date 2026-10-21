# Copyright (c) 2025 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Packed numeric arithmetic and fragment layouts."""

from cutlass import Float32
from cutlass import Int32
from cutlass.cutlass_dsl import T
from typing import Tuple
from cutlass import Uint32
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


# =============================================================================
# Half2 SIMD Intrinsics
# =============================================================================


@dsl_user_op
def half2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two Half2 values element-wise: (a.x*b.x, a.y*b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "mul.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def broadcast_f32_to_half2(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack one float32 value into both lanes of an f16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $1, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hadd2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two Half2 values element-wise: (a.x+b.x, a.y+b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """Half2 absolute value - clears sign bits of both fp16 values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Half2 max - element-wise max of 2 fp16 pairs."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 fp16 values in half2 as float32."""
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
        )
    )


@dsl_user_op
def half2_to_float2_scaled(
    h2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert half2 to float2 AND multiply by scale."""
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


# =============================================================================
# BFloat2 SIMD Intrinsics
# =============================================================================


@dsl_user_op
def bfloat2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two BFloat2 values element-wise: (a.x*b.x, a.y*b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "mul.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_broadcast_lane(x: Uint32, lane: Int32, *, loc=None, ip=None) -> Uint32:
    """Duplicate one BF16 lane from a packed bf16x2 register into both lanes."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(x).ir_value(loc=loc, ip=ip),
                Int32(lane).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p;
                .reg .b32 lo, hi, val, shifted;
                and.b32 lo, $1, 0x0000ffff;
                shr.u32 hi, $1, 16;
                setp.eq.s32 p, $2, 0;
                @p  mov.b32 val, lo;
                @!p mov.b32 val, hi;
                shl.b32 shifted, val, 16;
                or.b32 $0, val, shifted;
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
def broadcast_f32_to_bfloat2(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack one float32 value into both lanes of a bf16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.bf16x2.f32 $0, $1, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def pack_f32x2_to_bfloat2(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack 2 float32 values into one bf16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.satfinite.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def pack_f32x2_to_half2(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack two float32 values into one f16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.f16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_bf16x2_to_f16x2_via_f32(packed: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert packed bf16x2 to packed f16x2 through exact FP32 values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 b_lo, b_hi;
                .reg .f32 f_lo, f_hi;
                mov.b32 {b_lo, b_hi}, $1;
                cvt.f32.bf16 f_lo, b_lo;
                cvt.f32.bf16 f_hi, b_hi;
                cvt.rn.f16x2.f32 $0, f_hi, f_lo;
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
def cvt_bf16x2_to_e4m3x2(src: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert packed bf16x2 to packed e4m3x2 in the low 16 bits of a u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(src).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 out, zero;
                cvt.rn.satfinite.e4m3x2.bf16x2 out, $1;
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
def cvt_bf16x2x2_to_e4m3x4(lo: Uint32, hi: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert two bf16x2 values and pack the e4m3x2 results into one u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(lo).ir_value(loc=loc, ip=ip), Uint32(hi).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 out_lo, out_hi;
                cvt.rn.satfinite.e4m3x2.bf16x2 out_lo, $1;
                cvt.rn.satfinite.e4m3x2.bf16x2 out_hi, $2;
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
def bfloat2_add(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two BFloat2 values element-wise: (a.x+b.x, a.y+b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "add.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fp8x4_e4m3_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Widen 4 packed E4M3 bytes into 2 packed bf16x2 registers exactly."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 low;
            .reg .b32 q, out0, out1, sign0, sign1, mant0, mant1, tmp, bias, bias_f32;
            prmt.b32 q, $2, 0, 0x1302;

            and.b32 sign0, q, 0x80008000;
            shr.u32 tmp, q, 4;
            and.b32 mant0, tmp, 0x07F007F0;
            or.b32 out0, mant0, sign0;

            shl.b32 tmp, q, 8;
            and.b32 sign1, tmp, 0x80008000;
            shl.b32 tmp, q, 4;
            and.b32 mant1, tmp, 0x07F007F0;
            or.b32 out1, mant1, sign1;

            mov.b32 bias_f32, 0x7B800000;
            cvt.rn.bf16.f32 low, bias_f32;
            mov.b32 bias, {low, low};

            mul.bf16x2 $0, out0, bias;
            mul.bf16x2 $1, out1, bias;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def fp8x4_e4m3_to_bfloat2x2_via_f16(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Widen four packed E4M3 values exactly through native FP16 conversion.

    Every E4M3 value is exactly representable in FP16, and its four-bit
    significand is exactly representable in BF16.  The FP32 intermediates are
    therefore lossless; they only bridge the packed FP16 and BF16 PTX
    conversion forms.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e01, e23, h0, h1, h2, h3;
            .reg .b32 p01, p23;
            .reg .f32 f0, f1, f2, f3;
            mov.b32 {e01, e23}, $2;
            cvt.rn.f16x2.e4m3x2 p01, e01;
            cvt.rn.f16x2.e4m3x2 p23, e23;
            mov.b32 {h0, h1}, p01;
            mov.b32 {h2, h3}, p23;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            cvt.f32.f16 f2, h2;
            cvt.f32.f16 f3, h3;
            cvt.rn.bf16x2.f32 $0, f1, f0;
            cvt.rn.bf16x2.f32 $1, f3, f2;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def nvfp4_pair_to_bf16x2_sm120(
    packed: Uint32, scale: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Decode raw NVFP4 and E4M3 scale bytes using PTX 9.2 on SM120a.

    Every finite E2M1 * E4M3 product is exactly representable in BF16.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(packed).ir_value(loc=loc, ip=ip),
                Uint32(scale).ir_value(loc=loc, ip=ip),
            ],
            """
        {
            .reg .b8 q;
            .reg .b16 sf;
            .reg .b32 values, factors, sf_pair;
            cvt.u8.u32 q, $1;
            shl.b32 sf_pair, $2, 8;
            or.b32 sf_pair, $2, sf_pair;
            cvt.u16.u32 sf, sf_pair;
            cvt.rn.bf16x2.e2m1x2 values, q;
            cvt.rn.bf16x2.e4m3x2 factors, sf;
            mul.bf16x2 $0, values, factors;
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
def mxfp8_pair_to_bf16x2_sm120(
    packed: Uint32, scale: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Decode E4M3 pairs and an unmodified UE8M0 byte on SM120a."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(packed).ir_value(loc=loc, ip=ip),
                Uint32(scale).ir_value(loc=loc, ip=ip),
            ],
            """
        {
            .reg .b16 q, sf;
            .reg .b32 values, factors, sf_pair;
            cvt.u16.u32 q, $1;
            cvt.rn.bf16x2.e4m3x2 values, q;
            shl.b32 sf_pair, $2, 8;
            or.b32 sf_pair, $2, sf_pair;
            cvt.u16.u32 sf, sf_pair;
            cvt.rn.bf16x2.ue8m0x2 factors, sf;
            mul.bf16x2 $0, values, factors;
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
def fp8x4_e4m3_to_bfloat2x2_native_sm120(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Widen four packed E4M3 values with SM120a's direct BF16 conversion.

    ``cvt.rn.bf16x2.e4m3x2`` requires PTX ISA 9.2 and an architecture-specific
    SM120 target. Callers must therefore keep this helper behind an SM120-only
    compiled entry instead of using it in a portable kernel specialization.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e01, e23;
            mov.b32 {e01, e23}, $2;
            cvt.rn.bf16x2.e4m3x2 $0, e01;
            cvt.rn.bf16x2.e4m3x2 $1, e23;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def fp8x4_e4m3_to_half2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Widen four packed E4M3 values exactly into two f16x2 registers."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 e01, e23;
            mov.b32 {e01, e23}, $2;
            cvt.rn.f16x2.e4m3x2 $0, e01;
            cvt.rn.f16x2.e4m3x2 $1, e23;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e2m1x4_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE2M1 -> BF16 register dequant for one packed 4-value fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2, tmp;

            and.b32 out1, $2, 0x80008000;
            and.b32 tmp, $2, 0x70007000;
            shr.u32 tmp, tmp, 6;
            or.b32 out1, out1, tmp;

            shl.b32 q, $2, 4;
            and.b32 out2, q, 0x80008000;
            and.b32 tmp, q, 0x70007000;
            shr.u32 tmp, tmp, 6;
            or.b32 out2, out2, tmp;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e2m1x4_to_half2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE2M1 -> FP16 register dequant for one packed 4-value fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2, tmp;

            and.b32 out1, $2, 0x80008000;
            and.b32 tmp, $2, 0x70007000;
            shr.u32 tmp, tmp, 3;
            or.b32 out1, out1, tmp;

            shl.b32 q, $2, 4;
            and.b32 out2, q, 0x80008000;
            and.b32 tmp, q, 0x70007000;
            shr.u32 tmp, tmp, 3;
            or.b32 out2, out2, tmp;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e4m3x4_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE4M3 scale dequant for one packed 4-value BF16 fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2, tmp;

            and.b32 tmp, $2, 0x80008000;
            shr.u32 out1, tmp, 1;
            and.b32 tmp, $2, 0x7F007F00;
            shr.u32 tmp, tmp, 4;
            or.b32 out1, out1, tmp;

            shl.b32 q, $2, 8;
            and.b32 tmp, q, 0x80008000;
            shr.u32 out2, tmp, 1;
            and.b32 tmp, q, 0x7F007F00;
            shr.u32 tmp, tmp, 4;
            or.b32 out2, out2, tmp;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e4m3x4_to_half2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE4M3 scale dequant for one packed 4-value FP16 fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2;

            and.b32 out1, $2, 0xFF00FF00;
            shr.u32 out1, out1, 1;

            shl.b32 q, $2, 8;
            and.b32 out2, q, 0xFF00FF00;
            shr.u32 out2, out2, 1;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e8m0x4_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """E8M0 compute-scale dequant for one packed 4-value BF16 fragment.

    The W4A16 FP4 unpack path represents E2M1 values scaled by 2^-126.  Match
    the existing NVFP4 split by materializing E8M0 scales multiplied by 2^7 in
    the MMA input and applying the remaining compensation in the kernel
    epilogue.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .u32 b0, b1, b2, b3;
            .reg .u32 h0, h1, h2, h3;
            .reg .u32 t0, t1;

            and.b32 b0, $2, 0x000000ff;
            shr.u32 b1, $2, 8;
            and.b32 b1, b1, 0x000000ff;
            shr.u32 b2, $2, 16;
            and.b32 b2, b2, 0x000000ff;
            shr.u32 b3, $2, 24;

            add.u32 h0, b0, 7;
            add.u32 h1, b1, 7;
            add.u32 h2, b2, 7;
            add.u32 h3, b3, 7;
            shl.b32 h0, h0, 7;
            shl.b32 h1, h1, 7;
            shl.b32 h2, h2, 7;
            shl.b32 h3, h3, 7;

            shl.b32 t0, h2, 16;
            or.b32 $0, h0, t0;
            shl.b32 t1, h3, 16;
            or.b32 $1, h1, t1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e8m0x4_to_half2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """E8M0 compute-scale dequant for one packed 4-value FP16 fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p0, p1, p2, p3;
            .reg .u32 b0, b1, b2, b3;
            .reg .s32 e0, e1, e2, e3;
            .reg .f32 ef0, ef1, ef2, ef3;
            .reg .f32 f0, f1, f2, f3;
            .reg .b16 h0, h1, h2, h3;

            and.b32 b0, $2, 0x000000ff;
            shr.u32 b1, $2, 8;
            and.b32 b1, b1, 0x000000ff;
            shr.u32 b2, $2, 16;
            and.b32 b2, b2, 0x000000ff;
            shr.u32 b3, $2, 24;

            setp.eq.u32 p0, b0, 0;
            setp.eq.u32 p1, b1, 0;
            setp.eq.u32 p2, b2, 0;
            setp.eq.u32 p3, b3, 0;

            cvt.s32.u32 e0, b0;
            cvt.s32.u32 e1, b1;
            cvt.s32.u32 e2, b2;
            cvt.s32.u32 e3, b3;
            sub.s32 e0, e0, 120;
            sub.s32 e1, e1, 120;
            sub.s32 e2, e2, 120;
            sub.s32 e3, e3, 120;

            cvt.rn.f32.s32 ef0, e0;
            cvt.rn.f32.s32 ef1, e1;
            cvt.rn.f32.s32 ef2, e2;
            cvt.rn.f32.s32 ef3, e3;
            ex2.approx.f32 f0, ef0;
            ex2.approx.f32 f1, ef1;
            ex2.approx.f32 f2, ef2;
            ex2.approx.f32 f3, ef3;
            selp.f32 f0, 0f00000000, f0, p0;
            selp.f32 f1, 0f00000000, f1, p1;
            selp.f32 f2, 0f00000000, f2, p2;
            selp.f32 f3, 0f00000000, f3, p3;

            cvt.rn.f16.f32 h0, f0;
            cvt.rn.f16.f32 h1, f1;
            cvt.rn.f16.f32 h2, f2;
            cvt.rn.f16.f32 h3, f3;

            setp.eq.u32 p0, b0, 255;
            setp.eq.u32 p1, b1, 255;
            setp.eq.u32 p2, b2, 255;
            setp.eq.u32 p3, b3, 255;
            selp.b16 h0, 0x7e00, h0, p0;
            selp.b16 h1, 0x7e00, h1, p1;
            selp.b16 h2, 0x7e00, h2, p2;
            selp.b16 h3, 0x7e00, h3, p3;

            mov.b32 $0, {h0, h2};
            mov.b32 $1, {h1, h3};
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def f32_to_raw_bits(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Return the raw IEEE-754 bits of one float32 value."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_to_tf32_bits(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Round one float32 value to TF32 format and return its 32-bit operand bits."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 tmp;
                cvt.rna.tf32.f32 tmp, $1;
                mov.b32 $0, tmp;
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
def bf16_rowsum_m16k16_f32(
    d0: Float32,
    d1: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """Row-sum helper matching FlashInfer's m16k16 BF16 rowsum MMA."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(1065369472).ir_value(loc=loc, ip=ip),
            Uint32(1065369472).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
            {$0, _, $1, _},
            {$2, $3, $4, $5},
            {$6, $7},
            {$0, 0., $1, 0.};
        }
        """,
        "=f,=f,r,r,r,r,r,r,0,1",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(r0), Float32(r1)


@dsl_user_op
def f16_rowsum_m16k16_f32(
    d0: Float32,
    d1: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """Row-sum helper for an FP16-packed m16k16 probability fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(1006648320).ir_value(loc=loc, ip=ip),
            Uint32(1006648320).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
            {$0, _, $1, _},
            {$2, $3, $4, $5},
            {$6, $7},
            {$0, 0., $1, 0.};
        }
        """,
        "=f,=f,r,r,r,r,r,r,0,1",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(r0), Float32(r1)


@dsl_user_op
def byte_perm(a: Uint32, b: Uint32, selector: Int32, *, loc=None, ip=None) -> Uint32:
    """PTX byte permutation helper."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
                Int32(selector).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def frag_layout_swizzle_16b_to_8b(x: Uint32) -> Uint32:
    tmp = cute.arch.shuffle_sync_bfly(x, offset=1)
    x = byte_perm(x, tmp, Int32(0x5410 if cute.arch.lane_idx() & 0x1 == 0 else 0x3276))
    tmp = cute.arch.shuffle_sync_bfly(x, offset=2)
    x = byte_perm(x, tmp, Int32(0x5410 if cute.arch.lane_idx() & 0x2 == 0 else 0x3276))
    return x


@cute.jit
def frag_layout_swizzle_16b_to_8b_trans(x: Uint32) -> Uint32:
    tmp = cute.arch.shuffle_sync_bfly(x, offset=4)
    x = byte_perm(x, tmp, Int32(0x6420 if cute.arch.lane_idx() & 0x4 == 0 else 0x3175))
    tmp = cute.arch.shuffle_sync_bfly(x, offset=8)
    x = byte_perm(x, tmp, Int32(0x5410 if cute.arch.lane_idx() & 0x8 == 0 else 0x3276))
    tmp = cute.arch.shuffle_sync_bfly(x, offset=16)
    x = byte_perm(x, tmp, Int32(0x5410 if cute.arch.lane_idx() & 0x10 == 0 else 0x3276))
    return x


@dsl_user_op
def bfloat2_habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """BFloat16x2 absolute value - clears sign bits of both bf16 values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """BFloat16x2 max - element-wise max of 2 bf16 pairs."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 bf16 values in bfloat2 as float32."""
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
        )
    )


@dsl_user_op
def bfloat2_to_float2_scaled(
    bf2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert bfloat16x2 to float2 AND multiply by scale."""
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
