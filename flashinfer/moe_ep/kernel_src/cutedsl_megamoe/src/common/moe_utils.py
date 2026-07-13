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

"""Shared MoE scheduler utilities and online TMA descriptor helpers."""

from abc import ABC, abstractmethod  # noqa: F401
from typing import Any, Callable, Literal, Optional, Tuple, Type, Union  # noqa: F401

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace, Numeric, Pointer  # noqa: F401
from cutlass.cute.nvgpu import cpasync  # noqa: F401
from cutlass.cute.arch import nvvm_wrappers  # noqa: F401
from cutlass.cutlass_dsl import dsl_user_op, Boolean, Int32, Float32, T  # noqa: F401
from cutlass._mlir import ir
from common.megamoe_constants import Log2E, Fp32Max, Fp8E4M3RcpLimit, Fp8E5M2RcpLimit
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import cute as _cute_ir  # noqa: F401
from cutlass._mlir.dialects import vector, arith
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir  # noqa: F401
from dataclasses import dataclass  # noqa: F401


# -----------------------------------------------------------------
# MoE Helper functions
# -----------------------------------------------------------------


@cute.jit
def swiglu_act(
    t_swiglu: cute.Tensor,
    t_up: cute.Tensor,
    t_gate: cute.Tensor,
    prob: Float32,
) -> None:
    """
    SwiGLU activation with prob function.
    """
    for i in cutlass.range_constexpr(0, cute.size(t_swiglu), 2):
        t_swiglu_log2e = cute.arch.mul_packed_f32x2(
            (t_gate[i], t_gate[i + 1]),
            (-Log2E, -Log2E),
            rnd="rn",
            ftz=False,
        )
        (
            t_swiglu[i],
            t_swiglu[i + 1],
        ) = cute.arch.add_packed_f32x2(
            (
                cute.math.exp2(t_swiglu_log2e[0], fastmath=True),
                cute.math.exp2(t_swiglu_log2e[1], fastmath=True),
            ),
            (1.0, 1.0),
        )
        t_swiglu[i] = cute.arch.rcp_approx(t_swiglu[i])
        t_swiglu[i + 1] = cute.arch.rcp_approx(t_swiglu[i + 1])
        (
            t_swiglu[i],
            t_swiglu[i + 1],
        ) = cute.arch.mul_packed_f32x2(
            (t_swiglu[i], t_swiglu[i + 1]),
            (t_gate[i + 0], t_gate[i + 1]),
            rnd="rn",
            ftz=False,
        )
        (
            t_swiglu[i],
            t_swiglu[i + 1],
        ) = cute.arch.mul_packed_f32x2(
            (t_swiglu[i], t_swiglu[i + 1]),
            (t_up[i], t_up[i + 1]),
            rnd="rn",
            ftz=False,
        )
        (
            t_swiglu[i],
            t_swiglu[i + 1],
        ) = cute.arch.mul_packed_f32x2(
            (t_swiglu[i], t_swiglu[i + 1]),
            (prob, prob),
            rnd="rn",
            ftz=False,
        )


def fmin(
    a: Union[float, Float32],
    b: Union[float, Float32],
    *,
    nan: bool = True,
    loc=None,
    ip=None,
) -> Float32:
    if nan:
        ptx_instr = f"min.NaN.f32 $0, $1, $2;"  # noqa: F541
    else:
        ptx_instr = f"min.f32 $0, $1, $2;"  # noqa: F541
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"{ptx_instr}",
            f"=f,f,f",  # noqa: F541
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def fmax(
    a: Union[float, Float32],
    b: Union[float, Float32],
    *,
    nan: bool = True,
    loc=None,
    ip=None,
) -> Float32:
    if nan:
        ptx_instr = f"max.NaN.f32 $0, $1, $2;"  # noqa: F541
    else:
        ptx_instr = f"max.f32 $0, $1, $2;"  # noqa: F541
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"{ptx_instr}",
            f"=f,f,f",  # noqa: F541
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def cvt_f32_to_f8_to_f32(fp32x1, fp8_type, loc=None, ip=None):
    src_fp32 = Float32(fp32x1).ir_value(loc=loc, ip=ip)

    cvt_instruction_downcast = ""
    cvt_instruction_upcast = ""
    if cutlass.const_expr(fp8_type == cutlass.Float8E8M0FNU):
        cvt_instruction_downcast = "cvt.rp.satfinite.ue8m0x2.f32"
        cvt_instruction_upcast = "cvt.rn.bf16x2.ue8m0x2"
    elif cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
        cvt_instruction_downcast = "cvt.rn.satfinite.e4m3x2.f32"
        cvt_instruction_upcast = "cvt.rn.bf16x2.e4m3x2"
    elif cutlass.const_expr(fp8_type == cutlass.Float8E5M2):
        cvt_instruction_downcast = "cvt.rn.satfinite.e5m2x2.f32"
        cvt_instruction_upcast = "cvt.rn.bf16x2.e5m2x2"
    else:
        with cute.arch.elect_one():
            cute.printf("error: unsupported fp8 element type")
        return

    asm_tmpl = (
        "{\n"
        "  .reg .b16 bf_lo;\n"
        f"  {cvt_instruction_downcast} bf_lo, 0f00000000, $1;\n"
        f"  {cvt_instruction_upcast}  $0, bf_lo;\n"
        "}"
    )
    packed_i32 = llvm.inline_asm(
        T.i32(),
        [src_fp32],
        asm_tmpl,
        "=r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    vec_bf16_ty = ir.Type.parse("vector<2xbf16>")
    bf2_lo = llvm.bitcast(vec_bf16_ty, packed_i32, loc=loc, ip=ip)
    h0 = vector.extract(bf2_lo, [], [0], loc=loc, ip=ip)
    dst_f32 = arith.extf(Float32.mlir_type, h0, loc=loc, ip=ip)

    return dst_f32


@cute.jit
def cvt_f32x4_to_f8x4_pack_i32(fp32x4, fp8_type, loc=None, ip=None):
    fp32x4 = fp32x4.load()
    src_vec4 = (
        fp32x4.ir_value(loc=loc, ip=ip) if hasattr(fp32x4, "ir_value") else fp32x4
    )

    src0 = Float32(vector.extract(src_vec4, [], [0])).ir_value(loc=loc, ip=ip)
    src1 = Float32(vector.extract(src_vec4, [], [1])).ir_value(loc=loc, ip=ip)
    src2 = Float32(vector.extract(src_vec4, [], [2])).ir_value(loc=loc, ip=ip)
    src3 = Float32(vector.extract(src_vec4, [], [3])).ir_value(loc=loc, ip=ip)

    cvt_instruction = ""
    if cutlass.const_expr(fp8_type == cutlass.Float8E8M0FNU):
        cvt_instruction = "cvt.rp.satfinite.ue8m0x2.f32"
    elif cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
        cvt_instruction = "cvt.rn.satfinite.e4m3x2.f32"
    elif cutlass.const_expr(fp8_type == cutlass.Float8E5M2):
        cvt_instruction = "cvt.rn.satfinite.e5m2x2.f32"
    else:
        with cute.arch.elect_one():
            cute.printf("error: unsupported fp8 element type")
        return

    asm_tmpl = (
        "{\n"
        "  .reg .b16 lo;\n"
        "  .reg .b16 hi;\n"
        f"  {cvt_instruction} lo, $2, $1;\n"
        f"  {cvt_instruction} hi, $4, $3;\n"
        "  mov.b32 $0, {lo, hi};\n"
        "}"
    )
    packed_i32 = llvm.inline_asm(
        T.i32(),
        [src0, src1, src2, src3],
        asm_tmpl,
        "=r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    return packed_i32


def cvt_f32x8_to_f4x8_pack_i32(fp32x8, loc=None, ip=None):
    """8 f32 -> 8 NVFP4 (e2m1) packed into one i32 (4 bytes), via 4x
    ``cvt.rn.satfinite.e2m1x2.f32`` (Blackwell hardware cvt).  The generic
    ``.to(Float4E2M1FN)`` lowers to a slow per-element software path that
    dominates the issue-bound fc2 STG epilogue; this single-instruction packed
    cvt is the fp4 analogue of ``cvt_f32x4_to_f8x4_pack_i32``.

    Byte order is sequential: i32 = [{e0,e1},{e2,e3},{e4,e5},{e6,e7}] with e0 in
    the lowest nibble (matching ``cvt d,a,b`` -> d = {cvt(a) high, cvt(b) low}).
    """
    fp32x8 = fp32x8.load()
    src_vec8 = (
        fp32x8.ir_value(loc=loc, ip=ip) if hasattr(fp32x8, "ir_value") else fp32x8
    )
    s = [
        Float32(vector.extract(src_vec8, [], [i])).ir_value(loc=loc, ip=ip)
        for i in range(8)
    ]
    asm_tmpl = (
        "{\n"
        "  .reg .b8 b0, b1, b2, b3;\n"
        "  .reg .b16 h0, h1;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n"
        "  cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\n"
        "  mov.b16 h0, {b0, b1};\n"
        "  mov.b16 h1, {b2, b3};\n"
        "  mov.b32 $0, {h0, h1};\n"
        "}"
    )
    packed_i32 = llvm.inline_asm(
        T.i32(),
        [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]],
        asm_tmpl,
        "=r,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return packed_i32


@cute.jit
def quant_sfd_row(
    src,
    dst,
    norm_const,
    sf_vec_size,
    sf_dtype,
    d_dtype,
) -> None:
    rcp_limit = Fp8E4M3RcpLimit if d_dtype == cutlass.Float8E4M3FN else Fp8E5M2RcpLimit
    acc_frg = src.load()
    abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
    abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
    avg_fp32 = (
        abs_acc_frg.reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
        * rcp_limit
        * norm_const
    )
    qpvscale_up = cvt_f32_to_f8_to_f32(avg_fp32, sf_dtype)
    acc_scale = norm_const * cute.arch.rcp_approx(qpvscale_up)
    acc_scale = fmin(acc_scale, Fp32Max, nan=True)
    for ei in cutlass.range_constexpr(0, sf_vec_size, 2):
        src[ei], src[ei + 1] = cute.arch.mul_packed_f32x2(
            (src[ei], src[ei + 1]),
            (acc_scale, acc_scale),
            rnd="rn",
            ftz=False,
        )
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    for ei in cutlass.range_constexpr(0, sf_vec_size, 4):
        fp32x4 = cute.make_rmem_tensor(4, Float32)
        fp32x4[0] = src[ei + 0]
        fp32x4[1] = src[ei + 1]
        fp32x4[2] = src[ei + 2]
        fp32x4[3] = src[ei + 3]
        fp8x4_i32 = cvt_f32x4_to_f8x4_pack_i32(fp32x4, d_dtype)
        dst_i32[ei // 4] = cutlass.Int32(fp8x4_i32)
    return qpvscale_up
