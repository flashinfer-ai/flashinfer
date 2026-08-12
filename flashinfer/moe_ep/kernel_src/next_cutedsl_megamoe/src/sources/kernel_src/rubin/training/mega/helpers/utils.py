# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import math as _math
from cutlass.cutlass_dsl import Float32

from ......helpers.constants import Fp32Max, Fp8E4M3RcpLimit, Fp8E5M2RcpLimit, Log2E
from ......helpers.ptx_helpers import cvt_f32_to_fp8_to_f32, cvt_f32x4_to_f8x4_pack_i32


@cute.jit
def swiglu_act(
    t_swiglu: cute.Tensor,
    t_up: cute.Tensor,
    t_gate: cute.Tensor,
    prob: Optional[Float32] = None,
) -> None:
    """SwiGLU: ``out = (up * gate) * sigmoid(gate)`` """
    for i in cutlass.range_constexpr(0, cute.size(t_swiglu), 2):
        up_gate = cute.arch.mul_packed_f32x2(
            (t_up[i], t_up[i + 1]),
            (t_gate[i], t_gate[i + 1]),
            rnd="rn",
            ftz=False,
        )
        gate_log2e = cute.arch.mul_packed_f32x2(
            (t_gate[i], t_gate[i + 1]), (-Log2E, -Log2E), rnd="rn", ftz=False,
        )
        one_plus_exp = cute.arch.add_packed_f32x2(
            (
                cute.math.exp2(gate_log2e[0], fastmath=True),
                cute.math.exp2(gate_log2e[1], fastmath=True),
            ),
            (1.0, 1.0),
        )
        sigmoid = (cute.arch.rcp_approx(one_plus_exp[0]), cute.arch.rcp_approx(one_plus_exp[1]))
        (t_swiglu[i], t_swiglu[i + 1]) = cute.arch.mul_packed_f32x2(
            up_gate, sigmoid, rnd="rn", ftz=False,
        )
        if cutlass.const_expr(prob is not None):
            (t_swiglu[i], t_swiglu[i + 1]) = cute.arch.mul_packed_f32x2(
                (t_swiglu[i], t_swiglu[i + 1]),
                (prob, prob),
                rnd="rn",
                ftz=False,
            )


@cute.jit
def quant_sfd_row(
    src: cute.Tensor,
    dst: cute.Tensor,
    norm_const,
    sf_vec_size,
    sf_dtype,
    d_dtype,
):
    """Quantize the ``sf_vec_size`` values in ``src`` to ``d_dtype`` with one block scale."""
    rcp_limit = Fp8E4M3RcpLimit if d_dtype == cutlass.Float8E4M3FN else Fp8E5M2RcpLimit
    acc_frg = src.load()
    abs_acc_frg_ir = _math.absf(acc_frg.ir_value())
    abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
    # Fuse the two loop-invariant constants into one multiply
    rcp_limit_norm = rcp_limit * norm_const
    avg_fp32 = abs_acc_frg.reduce(cute.ReductionOp.MAX, Float32(0.0), 0) * rcp_limit_norm
    qpvscale_up = cvt_f32_to_fp8_to_f32(avg_fp32, sf_dtype)
    acc_scale = norm_const * cute.arch.rcp_approx(qpvscale_up)
    acc_scale = cute.arch.fmin(acc_scale, Fp32Max, nan=True)
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
def dswiglu_act(
    t_dgate: cute.Tensor,
    t_dup: cute.Tensor,
    t_acc: cute.Tensor,
    t_gate: cute.Tensor,
    t_up: cute.Tensor,
    beta_val: Float32,
    prob: Float32,
) -> Float32:
    """SwiGLU backward with beta scaling, prob weighting, and dprob accumulation.

    Given upstream gradient ``acc``, per-expert scalar ``beta_val``, per-token routing
    probability ``prob``, and forward pre-activations ``gate``/``up``::

        gate_b = gate * beta_val
        up_b   = up   * beta_val
        sig    = sigmoid(gate_b)
        swish  = gate_b * sig

        dprob += acc * up_b * swish            (returned to the caller)
        d_up   = acc * prob * swish
        d_gate = acc * prob * up_b * sig * (1 + gate_b * (1 - sig))
    """
    dprob_acc = Float32(0.0)
    for i in cutlass.range_constexpr(0, cute.size(t_acc), 2):
        gate_b = cute.arch.mul_packed_f32x2(
            (t_gate[i], t_gate[i + 1]), (beta_val, beta_val), rnd="rn", ftz=False,
        )
        up_b = cute.arch.mul_packed_f32x2(
            (t_up[i], t_up[i + 1]), (beta_val, beta_val), rnd="rn", ftz=False,
        )

        # sig = 1 / (1 + exp(-gate_b)); exp(-x) = exp2(-Log2E * x)
        sig_rcp = cute.arch.mul_packed_f32x2(
            gate_b, (-Log2E, -Log2E), rnd="rn", ftz=False,
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
        swish = cute.arch.mul_packed_f32x2(gate_b, (sig0, sig1), rnd="rn", ftz=False)

        # dprob += acc * up_b * swish  (both lanes into the running scalar)
        dp = cute.arch.mul_packed_f32x2(
            (t_acc[i], t_acc[i + 1]), (up_b[0], up_b[1]), rnd="rn", ftz=False,
        )
        dp = cute.arch.mul_packed_f32x2(dp, swish, rnd="rn", ftz=False)
        dprob_acc = dprob_acc + dp[0] + dp[1]

        # acc * prob (shared factor for d_up and d_gate)
        acc_prob = cute.arch.mul_packed_f32x2(
            (t_acc[i], t_acc[i + 1]), (prob, prob), rnd="rn", ftz=False,
        )

        # d_up = acc * prob * swish
        (t_dup[i], t_dup[i + 1]) = cute.arch.mul_packed_f32x2(
            acc_prob, swish, rnd="rn", ftz=False,
        )

        # d_gate = acc * prob * up_b * sig * (1 + gate_b * (1 - sig))
        one_minus_sig = cute.arch.add_packed_f32x2(
            (1.0, 1.0), (-sig0, -sig1), rnd="rn", ftz=False,
        )
        dsig = cute.arch.mul_packed_f32x2(gate_b, one_minus_sig, rnd="rn", ftz=False)
        term = cute.arch.add_packed_f32x2(
            (dsig[0], dsig[1]), (1.0, 1.0), rnd="rn", ftz=False,
        )
        dgate = cute.arch.mul_packed_f32x2(
            acc_prob, (up_b[0], up_b[1]), rnd="rn", ftz=False,
        )
        dgate = cute.arch.mul_packed_f32x2(dgate, (sig0, sig1), rnd="rn", ftz=False)
        (t_dgate[i], t_dgate[i + 1]) = cute.arch.mul_packed_f32x2(
            dgate, term, rnd="rn", ftz=False,
        )
    return dprob_acc


@cute.jit
def quant_sfd_col(
    src: cute.Tensor,
    dst: cute.Tensor,
    norm_const,
    sf_vec_size,
    sf_dtype,
    d_dtype,
):
    """Column (cross-thread) block-scale quantize: the amax is a warp reduction."""
    rcp_limit = Fp8E4M3RcpLimit if d_dtype == cutlass.Float8E4M3FN else Fp8E5M2RcpLimit
    acc_frg = src.load()
    abs_acc_frg_ir = _math.absf(acc_frg.ir_value())
    acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)

    qpvscale_up = Float32(0.0)
    tidx, _, _ = cute.arch.thread_idx()
    scale = rcp_limit * norm_const

    for vi in cutlass.range_constexpr(0, sf_vec_size, 4):
        # Warp-wide MAX across the 32 rows for each of the 4 lanes.
        max_value0 = Float32(cute.arch.warp_redux_sync(acc_frg[vi], "fmax", nan=True))
        max_value1 = Float32(cute.arch.warp_redux_sync(acc_frg[vi + 1], "fmax", nan=True))
        max_value2 = Float32(cute.arch.warp_redux_sync(acc_frg[vi + 2], "fmax", nan=True))
        max_value3 = Float32(cute.arch.warp_redux_sync(acc_frg[vi + 3], "fmax", nan=True))

        (max_value0, max_value1) = cute.arch.mul_packed_f32x2(
            (max_value0, max_value1), (scale, scale), rnd="rn", ftz=False,
        )
        (max_value2, max_value3) = cute.arch.mul_packed_f32x2(
            (max_value2, max_value3), (scale, scale), rnd="rn", ftz=False,
        )

        max_value0 = cvt_f32_to_fp8_to_f32(max_value0, sf_dtype)
        max_value1 = cvt_f32_to_fp8_to_f32(max_value1, sf_dtype)
        max_value2 = cvt_f32_to_fp8_to_f32(max_value2, sf_dtype)
        max_value3 = cvt_f32_to_fp8_to_f32(max_value3, sf_dtype)

        # Each thread keeps its assigned column's pre-round-trip scale.
        if tidx % 32 == vi:
            qpvscale_up = max_value0
        if tidx % 32 == vi + 1:
            qpvscale_up = max_value1
        if tidx % 32 == vi + 2:
            qpvscale_up = max_value2
        if tidx % 32 == vi + 3:
            qpvscale_up = max_value3

        max_value_rcp0 = cute.arch.fmin(cute.arch.rcp_approx(max_value0), Fp32Max, nan=True)
        max_value_rcp1 = cute.arch.fmin(cute.arch.rcp_approx(max_value1), Fp32Max, nan=True)
        max_value_rcp2 = cute.arch.fmin(cute.arch.rcp_approx(max_value2), Fp32Max, nan=True)
        max_value_rcp3 = cute.arch.fmin(cute.arch.rcp_approx(max_value3), Fp32Max, nan=True)

        (acc_scale_col0, acc_scale_col1) = cute.arch.mul_packed_f32x2(
            (norm_const, norm_const), (max_value_rcp0, max_value_rcp1), rnd="rn", ftz=False,
        )
        (acc_scale_col2, acc_scale_col3) = cute.arch.mul_packed_f32x2(
            (norm_const, norm_const), (max_value_rcp2, max_value_rcp3), rnd="rn", ftz=False,
        )

        (src[vi], src[vi + 1]) = cute.arch.mul_packed_f32x2(
            (src[vi], src[vi + 1]), (acc_scale_col0, acc_scale_col1), rnd="rn", ftz=False,
        )
        (src[vi + 2], src[vi + 3]) = cute.arch.mul_packed_f32x2(
            (src[vi + 2], src[vi + 3]), (acc_scale_col2, acc_scale_col3), rnd="rn", ftz=False,
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


__all__ = ["dswiglu_act", "quant_sfd_col", "quant_sfd_row", "swiglu_act"]
