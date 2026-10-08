# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""FP32 activation epilogues shared by the routed FMA kernels.

Keep DSL imports here, separate from the lightweight host activation contracts.
The caller applies GEMM scales first and rounds to BF16 after this epilogue.
"""

import cutlass
from cutlass import cute


@cute.jit
def _sigmoid(x):
    return cute.math.rcp(
        1.0 + cute.math.exp2(-x * 1.4426950408889634, fastmath=True),
        approx=True,
        ftz=True,
    )


@cute.jit
def _tanh(x):
    # Match the selected Frost epilogues, including their operation order.
    return 1.0 - 2.0 * cute.math.rcp(
        cute.math.exp2(x * 2.8853900817779268, fastmath=True) + 1.0,
        approx=True,
        ftz=True,
    )


@cute.jit
def apply_activation(
    up,
    gate,
    activation: cutlass.Constexpr,
    approximate_swiglu: cutlass.Constexpr = False,
):
    if cutlass.const_expr(activation == "swiglu"):
        # Preserve the existing SwiGLU arithmetic in each dtype family.
        if cutlass.const_expr(approximate_swiglu):
            sigmoid = cute.rcp(
                1.0 + cute.exp(-gate, fastmath=True), approx=True, ftz=True
            )
            value = (gate * sigmoid) * up
        else:
            value = up * (gate / (1.0 + cute.math.exp(-gate, fastmath=True)))
    elif cutlass.const_expr(activation == "swiglu_step"):
        value = cutlass.min(gate * _sigmoid(gate), 7.0) * cutlass.max(
            cutlass.min(up, 7.0), -7.0
        )
    elif cutlass.const_expr(activation == "situ"):
        value = ((4.0 * _tanh(gate / 4.0)) * _sigmoid(gate)) * (25.0 * _tanh(up / 25.0))
    elif cutlass.const_expr(activation == "geglu_tanh"):
        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
        value = (0.5 * gate * (1.0 + _tanh(inner))) * up
    elif cutlass.const_expr(activation == "geglu" or activation == "gelu"):
        x = gate if cutlass.const_expr(activation == "geglu") else up
        # Probe erf only for these activations; the general capability scanner
        # checks attribute expressions even in pruned constexpr branches.
        erf = getattr(cute.math, "erf")  # noqa: B009
        value = 0.5 * x * (1.0 + erf(x * 0.7071067811865475))
        if cutlass.const_expr(activation == "geglu"):
            value = value * up
    elif cutlass.const_expr(activation == "relu" or activation == "relu2"):
        value = cutlass.max(up, 0.0)
        if cutlass.const_expr(activation == "relu2"):
            value = value * value
    elif cutlass.const_expr(activation == "silu"):
        value = up * _sigmoid(up)
    else:
        value = up  # Identity; the host validates the activation name.
    return value
