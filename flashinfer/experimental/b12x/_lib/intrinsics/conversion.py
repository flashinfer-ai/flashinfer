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

"""Low-precision numeric conversion intrinsics."""

from cutlass import Float32
from cutlass.cutlass_dsl import T
from typing import Tuple
from cutlass import Uint16
from cutlass import Uint32
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


# =============================================================================
# FP8 E4M3 Intrinsics
# =============================================================================


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float32 to E4M3 using native cvt.rn.satfinite.e4m3x2.f32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cvt_f32x4_to_e4m3x4(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert 4 float32 values to 4 packed E4M3 bytes (v0 in the low byte)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
                cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def nvfp4_scale_from_amax(
    block_amax: Float32, global_scale: Float32, *, loc=None, ip=None
) -> Float32:
    """Compute block_amax * reciprocal_global_scale / 6 with CUDA tensor-scalar semantics."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(block_amax).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f64 amax_d, gs_d, q_d, six_d;
                cvt.f64.f32 amax_d, $1;
                cvt.f64.f32 gs_d, $2;
                mov.f64 six_d, 0d4018000000000000;
                mul.rn.f64 q_d, amax_d, gs_d;
                div.rn.f64 q_d, q_d, six_d;
                cvt.rn.f32.f64 $0, q_d;
            }
            """,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp8_e4m3_to_f32(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Decode the low E4M3 byte, including subnormals and signed zero."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 source, low, high;
                .reg .b32 pair;
                cvt.u16.u32 source, $1;
                cvt.rn.f16x2.e4m3x2 pair, source;
                mov.b32 {low, high}, pair;
                cvt.f32.f16 $0, low;
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
def fp8_e4m3_to_f32_and_rcp(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Return the decoded E4M3 reciprocal, or zero for either signed zero."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred zero;
                .reg .b16 source, low, high;
                .reg .b32 pair;
                .reg .f32 value, reciprocal;
                cvt.u16.u32 source, $1;
                cvt.rn.f16x2.e4m3x2 pair, source;
                mov.b32 {low, high}, pair;
                cvt.f32.f16 value, low;
                setp.eq.f32 zero, value, 0f00000000;
                rcp.approx.ftz.f32 reciprocal, value;
                selp.f32 $0, 0f00000000, reciprocal, zero;
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
# E4M3 -> F16/F32 Native Conversion Intrinsics
# =============================================================================


@dsl_user_op
def cvt_e4m3x2_to_f16x2_pair(
    packed_u32: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Decode 4 e4m3 bytes (packed u32) into 2 x f16x2 pairs."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 b16_01, b16_23;
            .reg .b32 hi32;
            cvt.u16.u32 b16_01, $2;
            shr.b32 hi32, $2, 16;
            cvt.u16.u32 b16_23, hi32;
            cvt.rn.f16x2.e4m3x2 $0, b16_01;
            cvt.rn.f16x2.e4m3x2 $1, b16_23;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def f16x2_to_f32x2(packed_h2: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Unpack f16x2 (u32) to two f32 values."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [Uint32(packed_h2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@cute.jit
def cvt_e4m3x4_to_f32x4(
    packed_u32: Uint32,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Decode 4 e4m3 bytes (packed u32) to 4 x f32 via hw-native e4m3->f16->f32."""
    h01, h23 = cvt_e4m3x2_to_f16x2_pair(packed_u32)
    f0, f1 = f16x2_to_f32x2(h01)
    f2, f3 = f16x2_to_f32x2(h23)
    return f0, f1, f2, f3


@dsl_user_op
def cvt_e4m3_to_f32_via_f16(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert single E4M3 byte to f32 via hw-native cvt.rn.f16x2.e4m3x2."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .b32 h2;
                .reg .b16 lo, hi;
                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2, fp8_pair;
                mov.b32 {lo, hi}, h2;
                cvt.f32.f16 $0, lo;
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
def cvt_w4a16_packed_e4m3_scale_to_f32(
    packed_byte: Uint32, *, loc=None, ip=None
) -> Float32:
    """Scalar mirror of packed_dequant_e4m3x4_to_bfloat2x2 for W4A16 scales."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(packed_byte).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 bits, tmp;
                .reg .b16 bf;
                and.b32 bits, $1, 0x80;
                shl.b32 bits, bits, 7;
                and.b32 tmp, $1, 0x7f;
                shl.b32 tmp, tmp, 4;
                or.b32 bits, bits, tmp;
                cvt.u16.u32 bf, bits;
                cvt.f32.bf16 $0, bf;
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
def dequant_kv_e4m3_pair_to_bf16x2(
    p0: Uint16, p1: Uint16, scale_f: Float32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FP8 -> BF16 K dequant for the BF16-QK m16n8k16 B operand.

    Byte-for-byte mirror of FlashInfer's sparse_mla prefill BF16-QK inline
    sequence (prefill_kernel.cuh:240-251): two ``cvt.rn.f16x2.e4m3x2`` decode
    the two e4m3 byte-pairs ``p0``/``p1`` (each a u16 holding two consecutive
    e4m3 K-dims), multiply the four resulting f16 lanes by the per-(token,blk)
    UE8M0 ``scale_f`` in f32, then two ``cvt.rn.bf16x2.f32`` pack them back to
    the bf16x2 ``b0``/``b1`` MMA operands. NON-saturating cvt (matches the
    reference; the K magnitudes are bounded by the e4m3 range so satfinite is a
    no-op, and the ref emits plain ``cvt.rn.bf16x2.f32``)."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [
            Uint16(p0).ir_value(loc=loc, ip=ip),
            Uint16(p1).ir_value(loc=loc, ip=ip),
            Float32(scale_f).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b32 h2_0, h2_1;
            .reg .b16 l0, h0, l1, h1;
            .reg .f32 fk0, fk1, fk2, fk3;
            cvt.rn.f16x2.e4m3x2 h2_0, $2;
            cvt.rn.f16x2.e4m3x2 h2_1, $3;
            mov.b32 {l0, h0}, h2_0;
            mov.b32 {l1, h1}, h2_1;
            cvt.f32.f16 fk0, l0;
            cvt.f32.f16 fk1, h0;
            cvt.f32.f16 fk2, l1;
            cvt.f32.f16 fk3, h1;
            mul.f32 fk0, fk0, $4;
            mul.f32 fk1, fk1, $4;
            mul.f32 fk2, fk2, $4;
            mul.f32 fk3, fk3, $4;
            cvt.rn.bf16x2.f32 $0, fk1, fk0;
            cvt.rn.bf16x2.f32 $1, fk3, fk2;
        }
        """,
        "=r,=r,h,h,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_e8m0_to_f32(e8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert a single E8M0 scale byte to its true f32 value 2**(byte-127).

    Byte 0 maps to 0.0. Unlike packed_dequant_e8m0x4_* (which fold a 2**7 bias
    into the MMA input), this returns the unbiased scale so callers can apply it
    directly in an f32 accumulator.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(e8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p0;
                .reg .u32 b0;
                .reg .s32 e0;
                .reg .f32 ef0;
                and.b32 b0, $1, 0x000000ff;
                setp.eq.u32 p0, b0, 0;
                cvt.s32.u32 e0, b0;
                sub.s32 e0, e0, 127;
                cvt.rn.f32.s32 ef0, e0;
                ex2.approx.f32 $0, ef0;
                selp.f32 $0, 0f00000000, $0, p0;
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
def cvt_e8m0x4_to_f32x4(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Decode 4 E8M0 scale bytes (packed u32) to 4 x true f32 (2**(byte-127))."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p0, p1, p2, p3;
            .reg .u32 b0, b1, b2, b3;
            .reg .s32 e0, e1, e2, e3;
            .reg .f32 ef0, ef1, ef2, ef3;
            and.b32 b0, $4, 0x000000ff;
            shr.u32 b1, $4, 8;
            and.b32 b1, b1, 0x000000ff;
            shr.u32 b2, $4, 16;
            and.b32 b2, b2, 0x000000ff;
            shr.u32 b3, $4, 24;
            setp.eq.u32 p0, b0, 0;
            setp.eq.u32 p1, b1, 0;
            setp.eq.u32 p2, b2, 0;
            setp.eq.u32 p3, b3, 0;
            cvt.s32.u32 e0, b0;
            cvt.s32.u32 e1, b1;
            cvt.s32.u32 e2, b2;
            cvt.s32.u32 e3, b3;
            sub.s32 e0, e0, 127;
            sub.s32 e1, e1, 127;
            sub.s32 e2, e2, 127;
            sub.s32 e3, e3, 127;
            cvt.rn.f32.s32 ef0, e0;
            cvt.rn.f32.s32 ef1, e1;
            cvt.rn.f32.s32 ef2, e2;
            cvt.rn.f32.s32 ef3, e3;
            ex2.approx.f32 $0, ef0;
            ex2.approx.f32 $1, ef1;
            ex2.approx.f32 $2, ef2;
            ex2.approx.f32 $3, ef3;
            selp.f32 $0, 0f00000000, $0, p0;
            selp.f32 $1, 0f00000000, $1, p1;
            selp.f32 $2, 0f00000000, $2, p2;
            selp.f32 $3, 0f00000000, $3, p3;
        }
        """,
        "=f,=f,=f,=f,r",
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


# =============================================================================
# FP4 (E2M1) Decode Intrinsics
# =============================================================================


@dsl_user_op
def fp4_decode_4bytes(
    packed_u32: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Decode 4 FP4 bytes (packed u32) into 4 x f16x2 pairs."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            mov.b32 {byte0, byte1, byte2, byte3}, $4;
            cvt.rn.f16x2.e2m1x2 $0, byte0;
            cvt.rn.f16x2.e2m1x2 $1, byte1;
            cvt.rn.f16x2.e2m1x2 $2, byte2;
            cvt.rn.f16x2.e2m1x2 $3, byte3;
        }
        """,
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def fp4_decode_2(byte_val: Uint32, *, loc=None, ip=None) -> Uint32:
    """Decode 1 FP4 byte into 1 f16x2 pair."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(byte_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b8 b0;
                cvt.u8.u32 b0, $1;
                cvt.rn.f16x2.e2m1x2 $0, b0;
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
def e2m1x8_to_e4m3x8(packed_u32: Uint32, *, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """Expand 8 packed E2M1 nibbles into 8 E4M3 bytes, losslessly.

    Every E2M1 value is exactly representable in E4M3, so this is a pure
    relabeling done with two prmt magnitude-LUT lookups plus a sign pass.
    Nibble i of the input becomes byte i of the (lo, hi) output pair.
    """
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 clo, chi, cslo, cshi, mlo, mhi, slo, shi;
            .reg .b32 tbl_a, tbl_b, tbl_s, zero;
            // E4M3 encodings of {0, .5, 1, 1.5, 2, 3, 4, 6}.
            mov.b32 tbl_a, 0x3C383000;
            mov.b32 tbl_b, 0x4C484440;
            mov.b32 tbl_s, 0x00008000;
            mov.b32 zero, 0;
            // Magnitude codes (low 3 bits of each nibble) as prmt selectors.
            and.b32 clo, $2, 0x00007777;
            shr.u32 chi, $2, 16;
            and.b32 chi, chi, 0x00007777;
            prmt.b32 mlo, tbl_a, tbl_b, clo;
            prmt.b32 mhi, tbl_a, tbl_b, chi;
            // Sign bit (nibble bit 3) -> byte bit 7 via a 2-entry LUT.
            shr.u32 cslo, $2, 3;
            and.b32 cslo, cslo, 0x00001111;
            shr.u32 cshi, $2, 19;
            and.b32 cshi, cshi, 0x00001111;
            prmt.b32 slo, tbl_s, zero, cslo;
            prmt.b32 shi, tbl_s, zero, cshi;
            or.b32 $0, mlo, slo;
            or.b32 $1, mhi, shi;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def e2m1x8_mul_residual_to_e4m3x8(
    packed_u32: Uint32, residual_h2: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Expand 8 E2M1 nibbles to E4M3 bytes with a residual multiplier.

    Decodes to f16 pairs, multiplies by ``residual_h2`` (the same residual
    broadcast to both f16 lanes), and rounds to E4M3.  Used by the w4a8
    NVFP4 path where the per-K/16 e4m3 scale is decomposed into a shared
    UE8M0 hardware exponent and this in-register residual.
    """
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(packed_u32).ir_value(loc=loc, ip=ip),
            Uint32(residual_h2).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b8 b0, b1, b2, b3;
            .reg .b32 h0, h1, h2, h3;
            .reg .b16 e0, e1, e2, e3;
            mov.b32 {b0, b1, b2, b3}, $2;
            cvt.rn.f16x2.e2m1x2 h0, b0;
            cvt.rn.f16x2.e2m1x2 h1, b1;
            cvt.rn.f16x2.e2m1x2 h2, b2;
            cvt.rn.f16x2.e2m1x2 h3, b3;
            mul.f16x2 h0, h0, $3;
            mul.f16x2 h1, h1, $3;
            mul.f16x2 h2, h2, $3;
            mul.f16x2 h3, h3, $3;
            cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;
            cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;
            cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;
            cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;
            mov.b32 $0, {e0, e1};
            mov.b32 $1, {e2, e3};
        }
        """,
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def cvt_fp32x2_to_e2m1x2(v0: Float32, v1: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert 2 x f32 to 1 x FP4 (E2M1) byte."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 b;
                cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;
                cvt.u32.u8 $0, b;
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


@cute.jit
def quant_dequant_2(
    v0: Float32, v1: Float32, sf_f32: Float32, eff_scale: Float32
) -> Tuple[Float32, Float32]:
    """Quantize-dequantize pair roundtrip through FP4."""
    inv_scale = Float32(1.0) / eff_scale
    fp4_byte = cvt_fp32x2_to_e2m1x2(v0 * inv_scale, v1 * inv_scale)
    h2 = fp4_decode_2(fp4_byte)
    f0, f1 = f16x2_to_f32x2(h2)
    return f0 * sf_f32, f1 * sf_f32


# =============================================================================
# Pack Intrinsics
# =============================================================================


@dsl_user_op
def pack_f32x2_to_f16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack two f32 values into one f16x2 u32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.f16.f32 lo, $1;
                cvt.rn.f16.f32 hi, $2;
                mov.b32 $0, {lo, hi};
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


# =============================================================================
# UE8M0 Intrinsics (for MXFP4)
# =============================================================================


@dsl_user_op
def pow2_ceil_ue8m0(
    scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Uint32]:
    """Round a positive FP32 ``scale`` UP to a power of two, bit-exactly.

    Returns ``(rounded_fp32, ue8m0_byte)`` where:
      * ``rounded_fp32`` == ``__uint_as_float(power_of_2_round(__float_as_uint(scale)))``
      * ``ue8m0_byte``   == ``(__float_as_uint(rounded_fp32) >> 23) & 0xFF``

    Bit-exact replica of FlashInfer's fp8_quant.cuh rounding +
    scale_convert.cuh fp32_to_ue8m0. Integer ops only (no lg2.approx), so unlike
    ``cvt_f32_to_ue8m0`` it matches the FlashInfer reference bit-for-bit.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        [Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred  p_mant;
            .reg .b32   bits, mant;

            // bits = __float_as_uint(scale)
            mov.b32 bits, $2;
            // mant = bits & 0x007FFFFF  (mantissa field)
            and.b32 mant, bits, 8388607;
            setp.ne.u32 p_mant, mant, 0;
            // if (mant) bits = (bits + 0x00800000) & 0x7F800000
            @p_mant add.u32 bits, bits, 8388608;
            @p_mant and.b32 bits, bits, 2139095040;
            // rounded fp32 scale = __uint_as_float(bits)
            mov.b32 $0, bits;
            // ue8m0 = (bits >> 23) & 0xFF
            bfe.u32 $1, bits, 23, 8;
        }
        """,
        "=f,=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    rounded = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    ue8m0 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Float32(rounded), Uint32(ue8m0)


@dsl_user_op
def cvt_f32_to_ue8m0(max_val: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Convert float32 max value to UE8M0 scale factor.

    UE8M0 is unsigned 8-bit exponent-only format:
    - value = 2^(ue8m0 - 127)
    - ue8m0 = ceil(log2(max_val)) + 127

    Uses lg2.approx.f32 for fast log2 approximation.
    Uses cvt.rpi (round towards positive infinity, i.e., ceiling).
    Returns value clamped to [0, 255].
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(max_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_neg, p_ovf;
                .reg .f32 log2_val;
                .reg .s32 exp_int, result;

                // Check for zero/negative
                setp.le.f32 p_zero, $1, 0f00000000;

                // Compute ceil(log2(max_val)) using cvt.rpi (round towards +inf)
                lg2.approx.f32 log2_val, $1;
                cvt.rpi.s32.f32 exp_int, log2_val;

                // Add bias and clamp to [0, 255]
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
        )
    )


@dsl_user_op
def ue8m0_to_output_scale(ue8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    """
    Convert UE8M0 to output_scale for MXFP4 quantization.

    UE8M0 value = 2^(ue8m0 - 127)
    Returns 1 / 2^(ue8m0 - 127) = 2^(127 - ue8m0)
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

                // Check for zero
                setp.eq.u32 p_zero, $1, 0;

                // Compute 2^(127 - ue8m0) = 1 / 2^(ue8m0 - 127)
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
        )
    )


# =============================================================================
# E2M1 Conversion
# =============================================================================


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
    """Convert eight float32 values to eight E2M1 (4-bit) values packed into uint32."""
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
