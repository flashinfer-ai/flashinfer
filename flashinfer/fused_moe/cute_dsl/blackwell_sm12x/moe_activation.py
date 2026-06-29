"""Activation helper functions for the b12x fused-MoE kernels."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from flashinfer.cute_dsl.fp4_common import fmax_f32, fmin_f32


def is_gated_activation(activation: str) -> bool:
    return activation in ("silu", "gelu_tanh", "swigluoai_uninterleave")


def gated_activation_f32(
    g,
    u,
    *,
    activation: str,
    limit: float | None,
    alpha: float,
    beta: float,
    fast_math: bool,
):
    """Return ``act(gate) * up_term`` (Float32) for a gated MoE activation.

    - silu: ``g*sigmoid(g)*u``
    - gelu_tanh: ``g*sigmoid(2z)*u`` (tanh-approx GELU, sigmoid(2z) == 0.5*(1+tanh(z)),
      z = 0.7978845608*(g + 0.044715*g^3))
    - swigluoai: ``g*sigmoid(alpha*g)*(u+beta)``, clamped when ``limit`` is set
    """
    if cutlass.const_expr(activation == "swigluoai_uninterleave"):
        if cutlass.const_expr(limit is not None):
            lim = Float32(limit)
            g = fmin_f32(g, lim)
            u = fmax_f32(fmin_f32(u, lim), Float32(-limit))
        sig_arg = Float32(alpha) * g
        up_term = u + Float32(beta)
    elif cutlass.const_expr(activation == "gelu_tanh"):
        # Standard tanh-approximation GELU.
        sig_arg = Float32(2.0 * 0.7978845608028654) * (
            g + Float32(0.044715) * g * g * g
        )
        up_term = u
    else:
        sig_arg = g
        up_term = u
    sigmoid_g = cute.arch.rcp_approx(
        Float32(1.0) + cute.math.exp(-sig_arg, fastmath=fast_math)
    )
    return g * sigmoid_g * up_term
