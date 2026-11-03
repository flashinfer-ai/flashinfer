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

from abc import ABC, abstractmethod
from typing import Any, Callable, Literal, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace, Numeric, Pointer
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.arch import nvvm_wrappers
from cutlass.cutlass_dsl import dsl_user_op, Boolean, Int32, Float32, T
from cutlass._mlir import ir
from common.megamoe_constants import Log2E, Fp32Max, Fp8E4M3RcpLimit, Fp8E5M2RcpLimit
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import cute as _cute_ir
from cutlass._mlir.dialects import vector, arith
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from dataclasses import dataclass


# -----------------------------------------------------------------
# MoE Helper functions
# -----------------------------------------------------------------


@cute.jit
def swiglu_act(
    t_swiglu: cute.Tensor,
    t_up: cute.Tensor,
    t_gate: cute.Tensor,
    prob: Optional[Float32] = None,
) -> None:
    """
    SwiGLU activation with prob function.
    """
    for i in cutlass.range_constexpr(0, cute.size(t_swiglu), 2):
        t_swiglu_log2e = cute.arch.mul_packed_f32x2(
            (t_gate[i], t_gate[i + 1]),
            (-Log2E, -Log2E),
            rnd='rn',
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
            rnd='rn',
            ftz=False,
        )
        (
            t_swiglu[i],
            t_swiglu[i + 1],
        ) = cute.arch.mul_packed_f32x2(
            (t_swiglu[i], t_swiglu[i + 1]),
            (t_up[i], t_up[i + 1]),
            rnd='rn',
            ftz=False,
        )
        if cutlass.const_expr(prob is not None):
            (
                t_swiglu[i],
                t_swiglu[i + 1],
            ) = cute.arch.mul_packed_f32x2(
                (t_swiglu[i], t_swiglu[i + 1]),
                (prob, prob),
                rnd='rn',
                ftz=False,
            )


@cute.jit
def dswiglu_act(
    t_dgate: cute.Tensor,
    t_dup: cute.Tensor,
    t_acc: cute.Tensor,
    t_gate: cute.Tensor,
    t_up: cute.Tensor,
    beta_val: cutlass.Float32,
    prob: cutlass.Float32,
) -> cutlass.Float32:
    """SwiGLU backward with beta scaling, prob weighting, and dprob accumulation.

    Given upstream gradient ``acc``, per-expert scalar ``beta_val``, per-token
    routing probability ``prob``, and forward pre-activations ``gate``/``up``:

        gate_b = gate * beta_val
        up_b   = up   * beta_val
        sig    = sigmoid(gate_b)
        swish  = gate_b * sig

        dprob += acc * up_b * swish   (returned to the caller)

        d_up   = acc * prob * swish
        d_gate = acc * prob * up_b * sig * (1 + gate_b * (1 - sig))
    """
    dprob_acc = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, cute.size(t_acc), 2):
        # gate_b = gate * beta, up_b = up * beta
        gate_b = cute.arch.mul_packed_f32x2(
            (t_gate[i], t_gate[i + 1]),
            (beta_val, beta_val),
            rnd='rn',
            ftz=False,
        )
        up_b = cute.arch.mul_packed_f32x2(
            (t_up[i], t_up[i + 1]),
            (beta_val, beta_val),
            rnd='rn',
            ftz=False,
        )

        # sig = 1 / (1 + exp(-gate_b)); exp(-x) = exp2(-Log2E * x)
        sig_rcp = cute.arch.mul_packed_f32x2(
            gate_b,
            (-Log2E, -Log2E),
            rnd='rn',
            ftz=False,
        )
        (sig0, sig1) = cute.arch.add_packed_f32x2(
            (
                cute.math.exp2(sig_rcp[0], fastmath=True),
                cute.math.exp2(sig_rcp[1], fastmath=True),
            ),
            (1.0, 1.0),
        )
        sig0 = cute.arch.rcp_approx(sig0)
        sig1 = cute.arch.rcp_approx(sig1)

        # swish = gate_b * sig
        swish = cute.arch.mul_packed_f32x2(
            gate_b,
            (sig0, sig1),
            rnd='rn',
            ftz=False,
        )

        # dprob contribution: acc * up_b * swish (accumulate both lanes into
        # the single running dprob_acc scalar).
        dp = cute.arch.mul_packed_f32x2(
            (t_acc[i], t_acc[i + 1]),
            (up_b[0], up_b[1]),
            rnd='rn',
            ftz=False,
        )
        dp = cute.arch.mul_packed_f32x2(
            dp,
            swish,
            rnd='rn',
            ftz=False,
        )
        dprob_acc = dprob_acc + dp[0] + dp[1]

        # acc * prob (shared factor for d_up and d_gate)
        acc_prob = cute.arch.mul_packed_f32x2(
            (t_acc[i], t_acc[i + 1]),
            (prob, prob),
            rnd='rn',
            ftz=False,
        )

        # d_up = acc * prob * swish
        (
            t_dup[i],
            t_dup[i + 1],
        ) = cute.arch.mul_packed_f32x2(
            acc_prob,
            swish,
            rnd='rn',
            ftz=False,
        )

        # d_gate = acc * prob * up_b * sig * (1 + gate_b * (1 - sig))
        one_minus_sig = cute.arch.add_packed_f32x2(
            (1.0, 1.0),
            (-sig0, -sig1),
            rnd='rn',
            ftz=False,
        )
        dsig = cute.arch.mul_packed_f32x2(
            gate_b,
            one_minus_sig,
            rnd='rn',
            ftz=False,
        )
        term = cute.arch.add_packed_f32x2(
            (dsig[0], dsig[1]),
            (1.0, 1.0),
            rnd='rn',
            ftz=False,
        )
        # d_gate = acc_prob * up_b * sig * term
        dgate = cute.arch.mul_packed_f32x2(
            acc_prob,
            (up_b[0], up_b[1]),
            rnd='rn',
            ftz=False,
        )
        dgate = cute.arch.mul_packed_f32x2(
            dgate,
            (sig0, sig1),
            rnd='rn',
            ftz=False,
        )
        (
            t_dgate[i],
            t_dgate[i + 1],
        ) = cute.arch.mul_packed_f32x2(
            dgate,
            term,
            rnd='rn',
            ftz=False,
        )
    return dprob_acc


def fmin(
    a: Union[float, Float32],
    b: Union[float, Float32],
    *,
    nan: bool = True,
    loc=None,
    ip=None,
) -> Float32:
    if nan:
        ptx_instr = f"min.NaN.f32 $0, $1, $2;"
    else:
        ptx_instr = f"min.f32 $0, $1, $2;"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"{ptx_instr}",
            f"=f,f,f",
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
        ptx_instr = f"max.NaN.f32 $0, $1, $2;"
    else:
        ptx_instr = f"max.f32 $0, $1, $2;"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"{ptx_instr}",
            f"=f,f,f",
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


@cute.jit
def quant_sfd_row(
    src, dst, norm_const,
    sf_vec_size, sf_dtype, d_dtype,
) -> None:
    rcp_limit = Fp8E4M3RcpLimit if d_dtype == cutlass.Float8E4M3FN else Fp8E5M2RcpLimit
    acc_frg = src.load()
    abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
    abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
    avg_fp32 = (
        abs_acc_frg.reduce(cute.ReductionOp.MAX, Float32(0.0), 0)
        * rcp_limit * norm_const
    )
    qpvscale_up = cvt_f32_to_f8_to_f32(avg_fp32, sf_dtype)
    acc_scale = norm_const * cute.arch.rcp_approx(qpvscale_up)
    acc_scale = fmin(acc_scale, Fp32Max, nan=True)
    for ei in cutlass.range_constexpr(0, sf_vec_size, 2):
        src[ei], src[ei + 1] = cute.arch.mul_packed_f32x2(
            (src[ei], src[ei + 1]), (acc_scale, acc_scale), rnd="rn", ftz=False,
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


@cute.jit
def quant_sfd_col(
    src, dst, norm_const,
    sf_vec_size, sf_dtype, d_dtype,
) -> None:
    rcp_limit = Fp8E4M3RcpLimit if d_dtype == cutlass.Float8E4M3FN else Fp8E5M2RcpLimit
    acc_frg = src.load()
    abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
    acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)

    qpvscale_up = Float32(0.0)
    tidx, _, _ = cute.arch.thread_idx()
    scale = rcp_limit * norm_const

    for vi in cutlass.range_constexpr(0, sf_vec_size, 4):
        # Warp-wide MAX across the 32 rows for each of the 4 lanes.
        max_value0 = Float32(
            cute.arch.warp_redux_sync(acc_frg[vi], "fmax", nan=True)
        )
        max_value1 = Float32(
            cute.arch.warp_redux_sync(acc_frg[vi + 1], "fmax", nan=True)
        )
        max_value2 = Float32(
            cute.arch.warp_redux_sync(acc_frg[vi + 2], "fmax", nan=True)
        )
        max_value3 = Float32(
            cute.arch.warp_redux_sync(acc_frg[vi + 3], "fmax", nan=True)
        )

        # Normalize: max * rcp_limit * norm_const, packed 2-at-a-time.
        (max_value0, max_value1) = cute.arch.mul_packed_f32x2(
            (max_value0, max_value1), (scale, scale), rnd='rn', ftz=False,
        )
        (max_value2, max_value3) = cute.arch.mul_packed_f32x2(
            (max_value2, max_value3), (scale, scale), rnd='rn', ftz=False,
        )

        # F8 round-trip: quantizes the scale to the SF's representable bin
        # so ``acc_scale`` matches what the dequant path will apply.
        max_value0 = cvt_f32_to_f8_to_f32(max_value0, sf_dtype)
        max_value1 = cvt_f32_to_f8_to_f32(max_value1, sf_dtype)
        max_value2 = cvt_f32_to_f8_to_f32(max_value2, sf_dtype)
        max_value3 = cvt_f32_to_f8_to_f32(max_value3, sf_dtype)

        # Each thread keeps its assigned column's pre-round-trip scale.
        if tidx % 32 == vi:
            qpvscale_up = max_value0
        if tidx % 32 == vi + 1:
            qpvscale_up = max_value1
        if tidx % 32 == vi + 2:
            qpvscale_up = max_value2
        if tidx % 32 == vi + 3:
            qpvscale_up = max_value3

        max_value_rcp0 = cute.arch.rcp_approx(max_value0)
        max_value_rcp1 = cute.arch.rcp_approx(max_value1)
        max_value_rcp2 = cute.arch.rcp_approx(max_value2)
        max_value_rcp3 = cute.arch.rcp_approx(max_value3)

        max_value_rcp0 = fmin(max_value_rcp0, Fp32Max, nan=True)
        max_value_rcp1 = fmin(max_value_rcp1, Fp32Max, nan=True)
        max_value_rcp2 = fmin(max_value_rcp2, Fp32Max, nan=True)
        max_value_rcp3 = fmin(max_value_rcp3, Fp32Max, nan=True)

        (acc_scale_col0, acc_scale_col1) = cute.arch.mul_packed_f32x2(
            (norm_const, norm_const), (max_value_rcp0, max_value_rcp1),
            rnd='rn', ftz=False,
        )
        (acc_scale_col2, acc_scale_col3) = cute.arch.mul_packed_f32x2(
            (norm_const, norm_const), (max_value_rcp2, max_value_rcp3),
            rnd='rn', ftz=False,
        )

        # Apply per-column scale in place; each thread's src[vi..vi+3] is its
        # row's contribution to columns vi..vi+3, so the warp-uniform
        # column scale is the correct multiplier here.
        (src[vi], src[vi + 1]) = cute.arch.mul_packed_f32x2(
            (src[vi], src[vi + 1]),
            (acc_scale_col0, acc_scale_col1), rnd='rn', ftz=False,
        )
        (src[vi + 2], src[vi + 3]) = cute.arch.mul_packed_f32x2(
            (src[vi + 2], src[vi + 3]),
            (acc_scale_col2, acc_scale_col3), rnd='rn', ftz=False,
        )

    # Convert scaled fp32 to fp8 and store into dst (same path as row).
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
