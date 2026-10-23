# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FC1 epilogue for the SM100 MXFP8 x MXFP4 MegaMoE kernel.

The mainloop uses swap-AB orientation and leaves FP32 gate/up accumulators in
TMEM.  This component provides FP32 arithmetic and quantization for the persistent
epilogue. The native path distributes a K32 block across lanes and paired
warps, using these numerical primitives:

1. optionally clamp ``gate`` above and ``up`` on both sides in FP32;
2. evaluate ``up * (gate * sigmoid(gate))`` in FP32;
3. form ``scale = ceil_pow2(amax / 448)`` as one raw E8M0 byte; and
4. multiply by the exact reciprocal power of two and convert to E4M3 with
   saturating round-to-nearest-even.

There is deliberately no BF16 conversion and no routing-weight multiplication
at this boundary.  This is the deterministic handoff specified by
``README.md``; FC2 and the output reducer own routing weights.

The persistent epilogue owns TMEM traversal, cross-lane amax reduction,
stores, and readiness publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Type

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, Uint32, dsl_user_op

from flashinfer.quantization.quantization_cute_dsl_utils import (
    float_to_ue8m0_fast,
)


SF_VEC_SIZE = 32
E4M3_MAX = 448.0
E4M3_RCP_MAX = 1.0 / E4M3_MAX


@dataclass(frozen=True)
class Fc1EpilogueConfig:
    """Compile-time FC1 epilogue policy.

    ``gate_up_clamp`` follows the MegaMoE public API: the gate is upper-clamped
    while the up projection is clamped to the symmetric interval.  ``None``
    disables clamping.  ``fast_math`` selects the exponential implementation;
    it does not alter the E8M0 or E4M3 rounding modes.
    """

    gate_up_clamp: Optional[float] = None
    fast_math: bool = True

    def __post_init__(self) -> None:
        if self.gate_up_clamp is not None and self.gate_up_clamp < 0.0:
            raise ValueError(
                f"gate_up_clamp must be None or non-negative, got {self.gate_up_clamp}"
            )


def validate_fc1_epilogue_contract(
    *,
    acc_dtype: Type[cutlass.Numeric],
    output_dtype: Type[cutlass.Numeric],
    scale_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
) -> None:
    """Validate the compile-time types expected by the fused FC1/FC2 kernel."""

    if acc_dtype is not cutlass.Float32:
        raise ValueError("FC1 accumulators must be Float32")
    if output_dtype is not cutlass.Float8E4M3FN:
        raise ValueError("FC1 handoff data must be MXFP8 Float8E4M3FN")
    if scale_dtype is not cutlass.Float8E8M0FNU:
        raise ValueError("FC1 handoff scales must be raw Float8E8M0FNU")
    if sf_vec_size != SF_VEC_SIZE:
        raise ValueError(f"FC1 MXFP8 handoff requires sf_vec_size={SF_VEC_SIZE}")


def handoff_storage_bytes(
    *, token_count: int, intermediate_size: int
) -> tuple[int, int]:
    """Return data and scale-plane byte counts for host workspace planning."""

    if token_count < 0:
        raise ValueError("token_count must be non-negative")
    if intermediate_size <= 0 or intermediate_size % SF_VEC_SIZE:
        raise ValueError("intermediate_size must be a positive multiple of 32")
    return (
        token_count * intermediate_size,
        token_count * (intermediate_size // SF_VEC_SIZE),
    )


@dsl_user_op
def _ue8m0_to_inv_scale_contract(raw_scale, *, loc=None, ip=None):
    """Decode the reciprocal of every finite E8M0 code, including code zero.

    The generic quantization helper treats raw zero as a zero-scale sentinel.
    ``reference.py`` instead defines it as the finite scale ``2**-127``.  The
    difference only affects extremely small nonzero blocks, but preserving it
    here makes the kernel recipe complete rather than relying on underflow.
    """

    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(raw_scale).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .u32 exponent, bits;
                sub.u32 exponent, 254, $1;
                shl.b32 bits, exponent, 23;
                mov.b32 $0, bits;
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
def _cvt_f32x2_to_e4m3x2_rn_satfinite(x0, x1, *, loc=None, ip=None):
    """Pack two FP32 values as E4M3 using explicit RTNE saturation."""

    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(x0).ir_value(loc=loc, ip=ip),
                cutlass.Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 packed, zero;
                cvt.rn.satfinite.e4m3x2.f32 packed, $2, $1;
                mov.b16 zero, 0;
                mov.b32 $0, {packed, zero};
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
def _clamp_and_swiglu(
    output: cute.Tensor,
    gate: cute.Tensor,
    up: cute.Tensor,
    gate_up_clamp: Optional[cutlass.Float32],
    fast_math: bool,
    count: cutlass.Constexpr[int] = SF_VEC_SIZE,
) -> None:
    """Evaluate the contract's FP32 SwiGLU association for ``count`` values."""

    for idx in cutlass.range_constexpr(0, count, 2):
        g0 = cutlass.Float32(gate[idx])
        g1 = cutlass.Float32(gate[idx + 1])
        u0 = cutlass.Float32(up[idx])
        u1 = cutlass.Float32(up[idx + 1])

        if cutlass.const_expr(gate_up_clamp is not None):
            limit = gate_up_clamp
            g0 = cute.arch.fmin(g0, limit)
            g1 = cute.arch.fmin(g1, limit)
            u0 = cute.arch.fmax(cute.arch.fmin(u0, limit), -limit)
            u1 = cute.arch.fmax(cute.arch.fmin(u1, limit), -limit)

        exp0 = cute.math.exp(-g0, fastmath=fast_math)
        exp1 = cute.math.exp(-g1, fastmath=fast_math)
        sigmoid0 = cute.math.rcp(
            cutlass.Float32(1.0) + exp0,
            approx=fast_math,
            ftz=fast_math,
        )
        sigmoid1 = cute.math.rcp(
            cutlass.Float32(1.0) + exp1,
            approx=fast_math,
            ftz=fast_math,
        )

        # Keep this association synchronized with README.md and reference.py.
        silu0, silu1 = cute.arch.mul_packed_f32x2(
            (g0, g1), (sigmoid0, sigmoid1), rnd="rn", ftz=False
        )
        output[idx], output[idx + 1] = cute.arch.mul_packed_f32x2(
            (u0, u1), (silu0, silu1), rnd="rn", ftz=False
        )


@cute.jit
def _quantize_with_amax(
    values: cute.Tensor,
    output: cute.Tensor,
    amax,
    count: cutlass.Constexpr[int] = SF_VEC_SIZE,
):
    """Quantize ``count`` values against a K32 block's amax; return E8M0."""

    # float_to_ue8m0_fast is the exact round-toward-+inf exponent encoding:
    # ceil_pow2(amax / 448), saturated below the reserved 0xff E8M0 code.
    raw_scale = float_to_ue8m0_fast(amax * cutlass.Float32(E4M3_RCP_MAX))
    inv_scale = _ue8m0_to_inv_scale_contract(raw_scale)
    packed_output = cute.recast_tensor(output, cutlass.Uint16)

    for idx in cutlass.range_constexpr(0, count, 2):
        scaled0, scaled1 = cute.arch.mul_packed_f32x2(
            (cutlass.Float32(values[idx]), cutlass.Float32(values[idx + 1])),
            (inv_scale, inv_scale),
            rnd="rn",
            ftz=False,
        )
        packed = _cvt_f32x2_to_e4m3x2_rn_satfinite(scaled0, scaled1)
        packed_output[idx // 2] = packed.to(cutlass.Uint16)

    return raw_scale


class Fc1Epilogue:
    """Validated FC1 handoff types and compile-time clamp configuration."""

    acc_dtype = cutlass.Float32
    output_dtype = cutlass.Float8E4M3FN
    scale_dtype = cutlass.Float8E8M0FNU
    sf_vec_size = SF_VEC_SIZE

    def __init__(self, config: Optional[Fc1EpilogueConfig] = None) -> None:
        if config is None:
            config = Fc1EpilogueConfig()
        validate_fc1_epilogue_contract(
            acc_dtype=self.acc_dtype,
            output_dtype=self.output_dtype,
            scale_dtype=self.scale_dtype,
            sf_vec_size=self.sf_vec_size,
        )
        self.config = config
        self._clamp = (
            cutlass.Float32(config.gate_up_clamp)
            if config.gate_up_clamp is not None
            else None
        )


__all__ = [
    "E4M3_MAX",
    "Fc1Epilogue",
    "Fc1EpilogueConfig",
    "SF_VEC_SIZE",
    "handoff_storage_bytes",
    "validate_fc1_epilogue_contract",
]
