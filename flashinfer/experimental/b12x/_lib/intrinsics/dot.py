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

"""Packed dot-product intrinsics."""

from cutlass import Float32
from cutlass.cutlass_dsl import T
from typing import Tuple
from cutlass import Uint32
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from b12x._lib.intrinsics.conversion import f16x2_to_f32x2, fp4_decode_4bytes


# =============================================================================
# Half2 FMA Dot Product Intrinsics
# =============================================================================


@dsl_user_op
def hfma2_4(
    w0: Uint32,
    x0: Uint32,
    w1: Uint32,
    x1: Uint32,
    w2: Uint32,
    x2: Uint32,
    w3: Uint32,
    x3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """4-element half2 FMA dot product returning (lo, hi) f32."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(x0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(x1).ir_value(loc=loc, ip=ip),
            Uint32(w2).ir_value(loc=loc, ip=ip),
            Uint32(x2).ir_value(loc=loc, ip=ip),
            Uint32(w3).ir_value(loc=loc, ip=ip),
            Uint32(x3).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .f16x2 acc;
            .reg .b16 lo, hi;
            mov.b32 acc, 0;
            fma.rn.f16x2 acc, $2, $3, acc;
            fma.rn.f16x2 acc, $4, $5, acc;
            fma.rn.f16x2 acc, $6, $7, acc;
            fma.rn.f16x2 acc, $8, $9, acc;
            mov.b32 {lo, hi}, acc;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r,r,r,r,r,r,r,r",
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


@dsl_user_op
def hfma2_4_sum(
    w0: Uint32,
    x0: Uint32,
    w1: Uint32,
    x1: Uint32,
    w2: Uint32,
    x2: Uint32,
    w3: Uint32,
    x3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """4-element half2 FMA dot product returning lane sum."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(w0).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(w1).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(w2).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(w3).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, $1, $2, acc;
                fma.rn.f16x2 acc, $3, $4, acc;
                fma.rn.f16x2 acc, $5, $6, acc;
                fma.rn.f16x2 acc, $7, $8, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def hfma2_8(
    w0: Uint32,
    x0: Uint32,
    w1: Uint32,
    x1: Uint32,
    w2: Uint32,
    x2: Uint32,
    w3: Uint32,
    x3: Uint32,
    w4: Uint32,
    x4: Uint32,
    w5: Uint32,
    x5: Uint32,
    w6: Uint32,
    x6: Uint32,
    w7: Uint32,
    x7: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """8-element half2 FMA dot product returning (lo, hi) f32."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [
            Uint32(w0).ir_value(loc=loc, ip=ip),
            Uint32(x0).ir_value(loc=loc, ip=ip),
            Uint32(w1).ir_value(loc=loc, ip=ip),
            Uint32(x1).ir_value(loc=loc, ip=ip),
            Uint32(w2).ir_value(loc=loc, ip=ip),
            Uint32(x2).ir_value(loc=loc, ip=ip),
            Uint32(w3).ir_value(loc=loc, ip=ip),
            Uint32(x3).ir_value(loc=loc, ip=ip),
            Uint32(w4).ir_value(loc=loc, ip=ip),
            Uint32(x4).ir_value(loc=loc, ip=ip),
            Uint32(w5).ir_value(loc=loc, ip=ip),
            Uint32(x5).ir_value(loc=loc, ip=ip),
            Uint32(w6).ir_value(loc=loc, ip=ip),
            Uint32(x6).ir_value(loc=loc, ip=ip),
            Uint32(w7).ir_value(loc=loc, ip=ip),
            Uint32(x7).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .f16x2 acc;
            .reg .b16 lo, hi;
            mov.b32 acc, 0;
            fma.rn.f16x2 acc, $2, $3, acc;
            fma.rn.f16x2 acc, $4, $5, acc;
            fma.rn.f16x2 acc, $6, $7, acc;
            fma.rn.f16x2 acc, $8, $9, acc;
            fma.rn.f16x2 acc, $10, $11, acc;
            fma.rn.f16x2 acc, $12, $13, acc;
            fma.rn.f16x2 acc, $14, $15, acc;
            fma.rn.f16x2 acc, $16, $17, acc;
            mov.b32 {lo, hi}, acc;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
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


@dsl_user_op
def hfma2_8_sum(
    w0: Uint32,
    x0: Uint32,
    w1: Uint32,
    x1: Uint32,
    w2: Uint32,
    x2: Uint32,
    w3: Uint32,
    x3: Uint32,
    w4: Uint32,
    x4: Uint32,
    w5: Uint32,
    x5: Uint32,
    w6: Uint32,
    x6: Uint32,
    w7: Uint32,
    x7: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """8-element half2 FMA dot product returning lane sum."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(w0).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(w1).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(w2).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(w3).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
                Uint32(w4).ir_value(loc=loc, ip=ip),
                Uint32(x4).ir_value(loc=loc, ip=ip),
                Uint32(w5).ir_value(loc=loc, ip=ip),
                Uint32(x5).ir_value(loc=loc, ip=ip),
                Uint32(w6).ir_value(loc=loc, ip=ip),
                Uint32(x6).ir_value(loc=loc, ip=ip),
                Uint32(w7).ir_value(loc=loc, ip=ip),
                Uint32(x7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, $1, $2, acc;
                fma.rn.f16x2 acc, $3, $4, acc;
                fma.rn.f16x2 acc, $5, $6, acc;
                fma.rn.f16x2 acc, $7, $8, acc;
                fma.rn.f16x2 acc, $9, $10, acc;
                fma.rn.f16x2 acc, $11, $12, acc;
                fma.rn.f16x2 acc, $13, $14, acc;
                fma.rn.f16x2 acc, $15, $16, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# =============================================================================
# FP4 Dot Product Intrinsics
# =============================================================================


@dsl_user_op
def fp4_dot4_sum(
    u_packed: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """FP4 dot product: decode 4 FP4 bytes and dot with 4 x f16x2 inputs."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(u_packed).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 b0, b1, b2, b3;
                .reg .b32 h0, h1, h2, h3;
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 {b0, b1, b2, b3}, $1;
                cvt.rn.f16x2.e2m1x2 h0, b0;
                cvt.rn.f16x2.e2m1x2 h1, b1;
                cvt.rn.f16x2.e2m1x2 h2, b2;
                cvt.rn.f16x2.e2m1x2 h3, b3;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, h0, $2, acc;
                fma.rn.f16x2 acc, h1, $3, acc;
                fma.rn.f16x2 acc, h2, $4, acc;
                fma.rn.f16x2 acc, h3, $5, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp4_dot8_dual_sum(
    up_a: Uint32,
    up_b: Uint32,
    gate_a: Uint32,
    gate_b: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    x4: Uint32,
    x5: Uint32,
    x6: Uint32,
    x7: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32]:
    """Fused dual FP4 dot product (up + gate) over the SAME 8 shared f16x2
    activation words. Decodes both weight rows and runs two INDEPENDENT
    f16x2 accumulator chains in a single inline-asm block, then reduces each
    to f32. Bit-identical to running fp4_dot8_sum twice (same per-chain fma
    order, same lo+hi f32 reduction) but emits one asm block so ptxas can
    interleave the two independent fma chains for 2-way ILP and halves the
    per-pair epilogue (mov/cvt/add) op count on the issue-bound decode loop."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Uint32(up_a).ir_value(loc=loc, ip=ip),
            Uint32(up_b).ir_value(loc=loc, ip=ip),
            Uint32(gate_a).ir_value(loc=loc, ip=ip),
            Uint32(gate_b).ir_value(loc=loc, ip=ip),
            Uint32(x0).ir_value(loc=loc, ip=ip),
            Uint32(x1).ir_value(loc=loc, ip=ip),
            Uint32(x2).ir_value(loc=loc, ip=ip),
            Uint32(x3).ir_value(loc=loc, ip=ip),
            Uint32(x4).ir_value(loc=loc, ip=ip),
            Uint32(x5).ir_value(loc=loc, ip=ip),
            Uint32(x6).ir_value(loc=loc, ip=ip),
            Uint32(x7).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b8 ua0, ua1, ua2, ua3, ub0, ub1, ub2, ub3;
            .reg .b8 ga0, ga1, ga2, ga3, gb0, gb1, gb2, gb3;
            .reg .b32 uh0, uh1, uh2, uh3, uh4, uh5, uh6, uh7;
            .reg .b32 gh0, gh1, gh2, gh3, gh4, gh5, gh6, gh7;
            .reg .f16x2 uacc, gacc;
            .reg .b16 ulo, uhi, glo, ghi;
            .reg .f32 uflo, ufhi, gflo, gfhi;
            mov.b32 {ua0, ua1, ua2, ua3}, $2;
            mov.b32 {ub0, ub1, ub2, ub3}, $3;
            mov.b32 {ga0, ga1, ga2, ga3}, $4;
            mov.b32 {gb0, gb1, gb2, gb3}, $5;
            cvt.rn.f16x2.e2m1x2 uh0, ua0;
            cvt.rn.f16x2.e2m1x2 uh1, ua1;
            cvt.rn.f16x2.e2m1x2 uh2, ua2;
            cvt.rn.f16x2.e2m1x2 uh3, ua3;
            cvt.rn.f16x2.e2m1x2 uh4, ub0;
            cvt.rn.f16x2.e2m1x2 uh5, ub1;
            cvt.rn.f16x2.e2m1x2 uh6, ub2;
            cvt.rn.f16x2.e2m1x2 uh7, ub3;
            cvt.rn.f16x2.e2m1x2 gh0, ga0;
            cvt.rn.f16x2.e2m1x2 gh1, ga1;
            cvt.rn.f16x2.e2m1x2 gh2, ga2;
            cvt.rn.f16x2.e2m1x2 gh3, ga3;
            cvt.rn.f16x2.e2m1x2 gh4, gb0;
            cvt.rn.f16x2.e2m1x2 gh5, gb1;
            cvt.rn.f16x2.e2m1x2 gh6, gb2;
            cvt.rn.f16x2.e2m1x2 gh7, gb3;
            mov.b32 uacc, 0;
            mov.b32 gacc, 0;
            fma.rn.f16x2 uacc, uh0, $6, uacc;
            fma.rn.f16x2 gacc, gh0, $6, gacc;
            fma.rn.f16x2 uacc, uh1, $7, uacc;
            fma.rn.f16x2 gacc, gh1, $7, gacc;
            fma.rn.f16x2 uacc, uh2, $8, uacc;
            fma.rn.f16x2 gacc, gh2, $8, gacc;
            fma.rn.f16x2 uacc, uh3, $9, uacc;
            fma.rn.f16x2 gacc, gh3, $9, gacc;
            fma.rn.f16x2 uacc, uh4, $10, uacc;
            fma.rn.f16x2 gacc, gh4, $10, gacc;
            fma.rn.f16x2 uacc, uh5, $11, uacc;
            fma.rn.f16x2 gacc, gh5, $11, gacc;
            fma.rn.f16x2 uacc, uh6, $12, uacc;
            fma.rn.f16x2 gacc, gh6, $12, gacc;
            fma.rn.f16x2 uacc, uh7, $13, uacc;
            fma.rn.f16x2 gacc, gh7, $13, gacc;
            mov.b32 {ulo, uhi}, uacc;
            mov.b32 {glo, ghi}, gacc;
            cvt.f32.f16 uflo, ulo;
            cvt.f32.f16 ufhi, uhi;
            cvt.f32.f16 gflo, glo;
            cvt.f32.f16 gfhi, ghi;
            add.f32 $0, uflo, ufhi;
            add.f32 $1, gflo, gfhi;
        }
        """,
        "=f,=f,r,r,r,r,r,r,r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    up = Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip))
    gate = Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip))
    return up, gate


@dsl_user_op
def fp4_dot8_sum(
    u_a: Uint32,
    u_b: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    x4: Uint32,
    x5: Uint32,
    x6: Uint32,
    x7: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """FP4 dot product: decode 8 FP4 bytes (2 x u32) and dot with 8 x f16x2 inputs."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(u_a).ir_value(loc=loc, ip=ip),
                Uint32(u_b).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
                Uint32(x4).ir_value(loc=loc, ip=ip),
                Uint32(x5).ir_value(loc=loc, ip=ip),
                Uint32(x6).ir_value(loc=loc, ip=ip),
                Uint32(x7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 a0, a1, a2, a3;
                .reg .b8 b0, b1, b2, b3;
                .reg .b32 h0, h1, h2, h3, h4, h5, h6, h7;
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 {a0, a1, a2, a3}, $1;
                mov.b32 {b0, b1, b2, b3}, $2;
                cvt.rn.f16x2.e2m1x2 h0, a0;
                cvt.rn.f16x2.e2m1x2 h1, a1;
                cvt.rn.f16x2.e2m1x2 h2, a2;
                cvt.rn.f16x2.e2m1x2 h3, a3;
                cvt.rn.f16x2.e2m1x2 h4, b0;
                cvt.rn.f16x2.e2m1x2 h5, b1;
                cvt.rn.f16x2.e2m1x2 h6, b2;
                cvt.rn.f16x2.e2m1x2 h7, b3;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, h0, $3, acc;
                fma.rn.f16x2 acc, h1, $4, acc;
                fma.rn.f16x2 acc, h2, $5, acc;
                fma.rn.f16x2 acc, h3, $6, acc;
                fma.rn.f16x2 acc, h4, $7, acc;
                fma.rn.f16x2 acc, h5, $8, acc;
                fma.rn.f16x2 acc, h6, $9, acc;
                fma.rn.f16x2 acc, h7, $10, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _f16x2_dot_sum_f32acc(a_h2: Uint32, b_h2: Uint32) -> Float32:
    a0, a1 = f16x2_to_f32x2(a_h2)
    b0, b1 = f16x2_to_f32x2(b_h2)
    return a0 * b0 + a1 * b1


@cute.jit
def fp4_dot4_sum_f32acc(
    u_packed: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
) -> Float32:
    """FP4 dot product using f32 accumulation after FP4 decode."""
    h0, h1, h2, h3 = fp4_decode_4bytes(u_packed)
    return (
        _f16x2_dot_sum_f32acc(h0, x0)
        + _f16x2_dot_sum_f32acc(h1, x1)
        + _f16x2_dot_sum_f32acc(h2, x2)
        + _f16x2_dot_sum_f32acc(h3, x3)
    )


@cute.jit
def fp4_dot8_sum_f32acc(
    u_a: Uint32,
    u_b: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    x4: Uint32,
    x5: Uint32,
    x6: Uint32,
    x7: Uint32,
) -> Float32:
    """FP4 dot product using f32 accumulation after FP4 decode."""
    h0, h1, h2, h3 = fp4_decode_4bytes(u_a)
    h4, h5, h6, h7 = fp4_decode_4bytes(u_b)
    return (
        _f16x2_dot_sum_f32acc(h0, x0)
        + _f16x2_dot_sum_f32acc(h1, x1)
        + _f16x2_dot_sum_f32acc(h2, x2)
        + _f16x2_dot_sum_f32acc(h3, x3)
        + _f16x2_dot_sum_f32acc(h4, x4)
        + _f16x2_dot_sum_f32acc(h5, x5)
        + _f16x2_dot_sum_f32acc(h6, x6)
        + _f16x2_dot_sum_f32acc(h7, x7)
    )
