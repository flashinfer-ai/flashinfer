# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Rubin MLA decode FP8 example-local helpers.

Inline-ASM register-level packers used by the optional ``--use_fp16_softmax``
fast path in ``mla_decode_fp8_MTP.py`` / ``mla_decode_fp8.py`` /
``mla_decode_fp16.py``. Kept here rather than in ``cutlass.cute.arch`` so the
kernel example can ship these PTX wrappers without depending on a CUTLASS DSL
wheel release that exposes them as public ``cute.arch`` APIs.
"""

from typing import Optional, Tuple, Union

import cutlass
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.typing import Float16, Float32, Uint32, Int16


@dsl_user_op
def pack_f16x2(
    a: Float16,
    b: Float16,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Uint32:
    """Pack two Float16 values into one f16x2 Uint32 register."""
    vec_f16x2_type = ir.VectorType.get([2], Float16.mlir_type, loc=loc)
    a_val = Float16(a).ir_value(loc=loc, ip=ip)
    b_val = Float16(b).ir_value(loc=loc, ip=ip)
    vec = vector.from_elements(vec_f16x2_type, (a_val, b_val), loc=loc, ip=ip)
    return Uint32(llvm.bitcast(T.i32(), vec, loc=loc, ip=ip))


def _as_u32_value(
    packed: Uint32 | ir.Value,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    return Uint32(packed).ir_value(loc=loc, ip=ip)


def unpack_f16x2(packed, *, loc=None, ip=None):
    vec_type = ir.VectorType.get([2], Int16.mlir_type, loc=loc)
    vec = llvm.bitcast(vec_type, _as_u32_value(packed, loc=loc, ip=ip), loc=loc, ip=ip)
    i0 = vector.extract(vec, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    i1 = vector.extract(vec, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    return Int16(i0).bitcast(Float16, loc=loc, ip=ip), Int16(i1).bitcast(
        Float16, loc=loc, ip=ip
    )


@dsl_user_op
def add_packed_f16x2_u32(
    a: Uint32,
    b: Uint32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Uint32:
    """Add two packed f16x2 values stored in Uint32 registers."""
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


# {$nv-internal-release-sm109 begin}
@dsl_user_op
def exp2_packed_f16x2(
    src: Union[Tuple[Float16, Float16], Uint32],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Float16, Float16]:
    """Packed FP16x2 base-2 exponential: exp2(src) -> FP16x2.

    PTX: ``ex2.approx.f16x2 d, src;``
    The operand may be either an FP16x2 tuple or a pre-packed Uint32.
    """
    src_packed = (
        pack_f16x2(src[0], src[1], loc=loc, ip=ip) if isinstance(src, tuple) else src
    )

    r = llvm.inline_asm(
        T.i32(),
        [_as_u32_value(src_packed, loc=loc, ip=ip)],
        "ex2.approx.f16x2 $0, $1;",
        "=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return unpack_f16x2(r, loc=loc, ip=ip)


@dsl_user_op
def fma_f32x2_f16x2_f32x2_f32x2(
    src_a: Tuple[Float16, Float16],
    src_b: Tuple[Float32, Float32],
    src_c: Tuple[Float32, Float32],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Float32, Float32]:
    """Mixed-precision FMA (upconvert): FP16x2 * FP32x2 + FP32x2 → FP32x2.
    PTX: ``fma.rn.f32x2.f16x2.f32x2.f32x2 d, a, b, c;``
    Semantics: d[lo] = convert(a[lo]) * b[lo] + c[lo]; d[hi] = convert(a[hi]) * b[hi] + c[hi];
    """

    a0 = (
        Float16(src_a[0])
        .bitcast(cutlass.Int16, loc=loc, ip=ip)
        .ir_value(loc=loc, ip=ip)
    )
    a1 = (
        Float16(src_a[1])
        .bitcast(cutlass.Int16, loc=loc, ip=ip)
        .ir_value(loc=loc, ip=ip)
    )
    st = llvm.StructType.get_literal([T.f32(), T.f32()])
    r = llvm.inline_asm(
        st,
        [
            a0,
            a1,
            Float32(src_b[0]).ir_value(loc=loc, ip=ip),
            Float32(src_b[1]).ir_value(loc=loc, ip=ip),
            Float32(src_c[0]).ir_value(loc=loc, ip=ip),
            Float32(src_c[1]).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 a_packed;\n\t.reg .b64 b_packed, c_packed, d_packed;\n\t"
        "mov.b32 a_packed, {$2, $3};\n\tmov.b64 b_packed, {$4, $5};\n\tmov.b64 c_packed, {$6, $7};\n\t"
        "fma.rn.f32x2.f16x2.f32x2.f32x2 d_packed, a_packed, b_packed, c_packed;\n\t"
        "mov.b64 {$0, $1}, d_packed;\n\t}\n",
        "=f,=f,h,h,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), r, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), r, [1], loc=loc, ip=ip)),
    )


# {$nv-internal-release-sm109 begin}
@dsl_user_op
def fma_f16x2_f32x2_f32x2_f32x2(
    src_a: Tuple[Float32, Float32],
    src_b: Tuple[Float32, Float32],
    src_c: Tuple[Float32, Float32],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Float16, Float16]:
    """Mixed-precision FMA (downconvert): FP32x2 * FP32x2 + FP32x2 → FP16x2.

    PTX: ``fma.rz.f16x2.f32x2.f32x2.f32x2 d, a, b, c;`` (FHFMA2; SM109+)
    Fuses an FP32 FMA with a pack-and-convert to FP16x2, saving a separate
    CVT step relative to scalar FMA + CVT.RN.F16.F32 + pack.

    Semantics: d[lo] = cvt_f16(a[lo] * b[lo] + c[lo]);
               d[hi] = cvt_f16(a[hi] * b[hi] + c[hi]);
    Rounding is round-to-zero with ftz to match `mul_f16x2_f32x2_f32x2` and
    `add_f16x2_f32x2_f32x2` in this module.
    """
    r = llvm.inline_asm(
        T.i32(),
        [
            Float32(src_a[0]).ir_value(loc=loc, ip=ip),
            Float32(src_a[1]).ir_value(loc=loc, ip=ip),
            Float32(src_b[0]).ir_value(loc=loc, ip=ip),
            Float32(src_b[1]).ir_value(loc=loc, ip=ip),
            Float32(src_c[0]).ir_value(loc=loc, ip=ip),
            Float32(src_c[1]).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b64 a_packed, b_packed, c_packed;\n\t"
        "mov.b64 a_packed, {$1, $2};\n\tmov.b64 b_packed, {$3, $4};\n\tmov.b64 c_packed, {$5, $6};\n\t"
        "fma.rz.f16x2.f32x2.f32x2.f32x2 $0, a_packed, b_packed, c_packed;\n\t}\n",
        "=r,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return unpack_f16x2(r, loc=loc, ip=ip)


# {$nv-internal-release-sm109 end}


@dsl_user_op
def fma_f16x2_f16x2_f16x2_f16x2(
    src_a: Union[Tuple[Float16, Float16], Uint32],
    src_b: Union[Tuple[Float16, Float16], Uint32],
    src_c: Union[Tuple[Float16, Float16], Uint32],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Float16, Float16]:
    """Packed FP16 FMA: FP16x2 * FP16x2 + FP16x2 -> FP16x2.

    PTX: ``fma.rn.f16x2 d, a, b, c;``
    All three operands may be either FP16x2 tuples or pre-packed Uint32.
    Pack is done via LLVM ``vector.from_elements`` + ``llvm.bitcast`` (not
    inline-asm ``mov.b32``) so the optimizer can elide pack/unpack
    round-trips when the source values originate from a packed register.
    """

    a_packed = (
        pack_f16x2(src_a[0], src_a[1], loc=loc, ip=ip)
        if isinstance(src_a, tuple)
        else src_a
    )
    b_packed = (
        pack_f16x2(src_b[0], src_b[1], loc=loc, ip=ip)
        if isinstance(src_b, tuple)
        else src_b
    )
    c_packed = (
        pack_f16x2(src_c[0], src_c[1], loc=loc, ip=ip)
        if isinstance(src_c, tuple)
        else src_c
    )
    r = llvm.inline_asm(
        T.i32(),
        [
            _as_u32_value(a_packed, loc=loc, ip=ip),
            _as_u32_value(b_packed, loc=loc, ip=ip),
            _as_u32_value(c_packed, loc=loc, ip=ip),
        ],
        "fma.rn.f16x2 $0, $1, $2, $3;",
        "=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return unpack_f16x2(r, loc=loc, ip=ip)


@dsl_user_op
def cvt_f16x4_f8x4(
    src_a: Tuple[Float16, Float16],
    src_b: Tuple[Float16, Float16],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Uint32:
    """Convert two FP16x2 pairs to packed E4M3x4 (Uint32).

    PTX: two ``cvt.rn.satfinite.e4m3x2.f16x2`` + ``mov.b32 {e0, e1}``
    Compiles to chained F2FP.E4M3.MERGE_C, avoiding PRMT.
    """
    a0 = Float16(src_a[0]).bitcast(Int16, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    a1 = Float16(src_a[1]).bitcast(Int16, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    b0 = Float16(src_b[0]).bitcast(Int16, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    b1 = Float16(src_b[1]).bitcast(Int16, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    r = llvm.inline_asm(
        T.i32(),
        [a0, a1, b0, b1],
        "{\n\t"
        ".reg .b32 a_pk, b_pk;\n\t"
        ".reg .b16 e0, e1;\n\t"
        "mov.b32 a_pk, {$1, $2};\n\t"
        "mov.b32 b_pk, {$3, $4};\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e0, a_pk;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e1, b_pk;\n\t"
        "mov.b32 $0, {e0, e1};\n\t"
        "}\n",
        "=r,h,h,h,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return Uint32(r)


@dsl_user_op
def add_f32x2_f16x2_f32x2(
    src_a: Tuple[Float16, Float16],
    src_c: Tuple[Float32, Float32],
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Float32, Float32]:
    """Mixed-precision ADD: FP16x2 + FP32x2 → FP32x2.
    PTX: ``add.rn.f32x2.f16x2.f32x2 d, a, c;``
    """

    a0 = Float16(src_a[0]).bitcast(Int16, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    a1 = Float16(src_a[1]).bitcast(Int16, loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    st = llvm.StructType.get_literal([T.f32(), T.f32()])
    r = llvm.inline_asm(
        st,
        [
            a0,
            a1,
            Float32(src_c[0]).ir_value(loc=loc, ip=ip),
            Float32(src_c[1]).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .b32 a_packed;\n\t.reg .b64 c_packed, d_packed;\n\t"
        "mov.b32 a_packed, {$2, $3};\n\tmov.b64 c_packed, {$4, $5};\n\t"
        "add.rn.f32x2.f16x2.f32x2 d_packed, a_packed, c_packed;\n\t"
        "mov.b64 {$0, $1}, d_packed;\n\t}\n",
        "=f,=f,h,h,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), r, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), r, [1], loc=loc, ip=ip)),
    )


# {$nv-internal-release-sm109 end}


@dsl_user_op
def reduce_sum_packed_f16x2_to_f32(
    packed: Uint32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Float32:
    """Unpack a f16x2 Uint32 register and return the f32 sum of both lanes."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(packed).ir_value(loc=loc, ip=ip)],
            "{\n\t"
            ".reg .f16 lo, hi;\n\t"
            ".reg .f32 lo_f32, hi_f32;\n\t"
            "mov.b32 {lo, hi}, $1;\n\t"
            "cvt.f32.f16 lo_f32, lo;\n\t"
            "cvt.f32.f16 hi_f32, hi;\n\t"
            "add.f32 $0, lo_f32, hi_f32;\n\t"
            "}\n",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def softmax_f32x4_to_f16x2x2_and_e4m3x4(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    packed_b: Uint32,
    packed_c: Uint32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Uint32, Uint32, Uint32]:
    """Fused f32x4 softmax step returning two f16x2 sums and packed e4m3x4.

    Intended for the non-correction softmax stage where the accumulated error
    from FP16 mantissa rounding (via ``ex2.approx.f16x2``) does not compound.
    """
    a0_val = Float32(a0).ir_value(loc=loc, ip=ip)
    a1_val = Float32(a1).ir_value(loc=loc, ip=ip)
    a2_val = Float32(a2).ir_value(loc=loc, ip=ip)
    a3_val = Float32(a3).ir_value(loc=loc, ip=ip)
    b_val = Uint32(packed_b).ir_value(loc=loc, ip=ip)
    c_val = Uint32(packed_c).ir_value(loc=loc, ip=ip)

    f16x2_0 = llvm.inline_asm(
        T.i32(),
        [a0_val, a1_val, b_val, c_val],
        "{\n\t"
        ".reg .b32 packed_a, fma_result;\n\t"
        "cvt.rn.f16x2.f32 packed_a, $2, $1;\n\t"
        "fma.rn.f16x2 fma_result, packed_a, $3, $4;\n\t"
        "ex2.approx.f16x2 $0, fma_result;\n\t"
        "}\n",
        "=r,f,f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    f16x2_1 = llvm.inline_asm(
        T.i32(),
        [a2_val, a3_val, b_val, c_val],
        "{\n\t"
        ".reg .b32 packed_a, fma_result;\n\t"
        "cvt.rn.f16x2.f32 packed_a, $2, $1;\n\t"
        "fma.rn.f16x2 fma_result, packed_a, $3, $4;\n\t"
        "ex2.approx.f16x2 $0, fma_result;\n\t"
        "}\n",
        "=r,f,f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    fp8x4 = llvm.inline_asm(
        T.i32(),
        [f16x2_0, f16x2_1],
        "{\n\t"
        ".reg .b16 e0, e1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e0, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f16x2 e1, $2;\n\t"
        "mov.b32 $0, {e0, e1};\n\t"
        "}\n",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    return Uint32(f16x2_0), Uint32(f16x2_1), Uint32(fp8x4)


@dsl_user_op
def softmax_f32x4_to_f16x2x2(
    a0: Float32,
    a1: Float32,
    a2: Float32,
    a3: Float32,
    packed_b: Uint32,
    packed_c: Uint32,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> Tuple[Uint32, Uint32]:
    """Fused f32x4 softmax step returning two f16x2 (no e4m3 quantize).

    Same fma+exp2 fast path as ``softmax_f32x4_to_f16x2x2_and_e4m3x4`` but
    without the final ``cvt.rn.satfinite.e4m3x2.f16x2`` step — for kernels
    whose P operand is FP16 (e.g. ``mla_decode_fp16.py``).
    """
    a0_val = Float32(a0).ir_value(loc=loc, ip=ip)
    a1_val = Float32(a1).ir_value(loc=loc, ip=ip)
    a2_val = Float32(a2).ir_value(loc=loc, ip=ip)
    a3_val = Float32(a3).ir_value(loc=loc, ip=ip)
    b_val = Uint32(packed_b).ir_value(loc=loc, ip=ip)
    c_val = Uint32(packed_c).ir_value(loc=loc, ip=ip)

    f16x2_0 = llvm.inline_asm(
        T.i32(),
        [a0_val, a1_val, b_val, c_val],
        "{\n\t"
        ".reg .b32 packed_a, fma_result;\n\t"
        "cvt.rn.f16x2.f32 packed_a, $2, $1;\n\t"
        "fma.rn.f16x2 fma_result, packed_a, $3, $4;\n\t"
        "ex2.approx.f16x2 $0, fma_result;\n\t"
        "}\n",
        "=r,f,f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    f16x2_1 = llvm.inline_asm(
        T.i32(),
        [a2_val, a3_val, b_val, c_val],
        "{\n\t"
        ".reg .b32 packed_a, fma_result;\n\t"
        "cvt.rn.f16x2.f32 packed_a, $2, $1;\n\t"
        "fma.rn.f16x2 fma_result, packed_a, $3, $4;\n\t"
        "ex2.approx.f16x2 $0, fma_result;\n\t"
        "}\n",
        "=r,f,f,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    return Uint32(f16x2_0), Uint32(f16x2_1)
