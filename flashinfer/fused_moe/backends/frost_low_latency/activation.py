# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP32 activation contract shared by low-latency FMA and TC tactics."""

import cutlass
from cutlass import cute

from .quantization import _mul
from .fma.activations import apply_activation


@cute.jit
def swiglu(
    up,
    gate,
    limit: cutlass.Constexpr,
    alpha: cutlass.Constexpr,
    beta: cutlass.Constexpr,
):
    if cutlass.const_expr(limit is not None):
        gate = cutlass.min(gate, cutlass.Float32(limit))
        up = cutlass.max(
            cutlass.min(up, cutlass.Float32(limit)), cutlass.Float32(-limit)
        )
    gate_exp = gate
    if cutlass.const_expr(alpha != 1.0):
        gate_exp = _mul(gate, cutlass.Float32(alpha))
    if cutlass.const_expr(beta != 0.0):
        up = up + cutlass.Float32(beta)
    return _mul(gate / (1.0 + cute.math.exp(-gate_exp, fastmath=False)), up)


@cute.jit
def activate(
    up,
    gate,
    name: cutlass.Constexpr,
    limit: cutlass.Constexpr,
    alpha: cutlass.Constexpr,
    beta: cutlass.Constexpr,
    parameters: cutlass.Constexpr = None,
):
    # Preserve the published MX SwiGLU arithmetic. Other activations reuse the
    # shared FP32 bodies, without the legacy adapter's BF16 intermediate round.
    if cutlass.const_expr(name == "swiglu"):
        return swiglu(up, gate, limit, alpha, beta)
    else:
        return apply_activation(up, gate, name, False, parameters)
