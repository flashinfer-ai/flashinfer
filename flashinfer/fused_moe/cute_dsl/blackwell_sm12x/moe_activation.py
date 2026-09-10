"""Activation helper functions and metadata for the b12x fused-MoE kernels."""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32

from flashinfer.cute_dsl.fp4_common import fmax_f32, fmin_f32


SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
SWIGLUOAI_DEFAULT_LIMIT = 7.0
SWIGLUOAI_DEFAULT_ALPHA = 1.702
SWIGLUOAI_DEFAULT_BETA = 1.0

SUPPORTED_MOE_ACTIVATIONS = frozenset(
    {"silu", "gelu_tanh", "relu2", SWIGLUOAI_UNINTERLEAVE}
)
GATED_MOE_ACTIVATIONS = frozenset({"silu", "gelu_tanh", SWIGLUOAI_UNINTERLEAVE})


def normalize_moe_activation(activation: str) -> str:
    activation = str(activation)
    if activation not in SUPPORTED_MOE_ACTIVATIONS:
        raise ValueError(f"unsupported activation {activation!r}")
    return activation


def is_gated_moe_activation(activation: str) -> bool:
    return normalize_moe_activation(activation) in GATED_MOE_ACTIVATIONS


def normalize_swiglu_limit_for_activation(
    activation: str,
    swiglu_limit: float | None,
) -> float | None:
    activation = normalize_moe_activation(activation)
    if swiglu_limit is None:
        return SWIGLUOAI_DEFAULT_LIMIT if activation == SWIGLUOAI_UNINTERLEAVE else None
    if activation not in GATED_MOE_ACTIVATIONS:
        raise ValueError("swiglu_limit requires a gated MoE activation")
    limit = float(swiglu_limit)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(f"swiglu_limit must be positive and finite, got {limit}")
    return limit


def normalize_swiglu_alpha_for_activation(
    activation: str,
    swiglu_alpha: float | None,
) -> float:
    activation = normalize_moe_activation(activation)
    if activation != SWIGLUOAI_UNINTERLEAVE:
        # Accept the module default so callers can pass it unconditionally.
        if swiglu_alpha is not None and float(swiglu_alpha) not in (
            1.0,
            SWIGLUOAI_DEFAULT_ALPHA,
        ):
            raise ValueError(
                "swiglu_alpha is only configurable for swigluoai_uninterleave"
            )
        return 1.0
    alpha = SWIGLUOAI_DEFAULT_ALPHA if swiglu_alpha is None else float(swiglu_alpha)
    if not math.isfinite(alpha):
        raise ValueError(f"swiglu_alpha must be finite, got {alpha}")
    return alpha


def normalize_swiglu_beta_for_activation(
    activation: str,
    swiglu_beta: float | None,
) -> float:
    activation = normalize_moe_activation(activation)
    if activation != SWIGLUOAI_UNINTERLEAVE:
        # Accept the module default so callers can pass it unconditionally.
        if swiglu_beta is not None and float(swiglu_beta) not in (
            0.0,
            SWIGLUOAI_DEFAULT_BETA,
        ):
            raise ValueError(
                "swiglu_beta is only configurable for swigluoai_uninterleave"
            )
        return 0.0
    beta = SWIGLUOAI_DEFAULT_BETA if swiglu_beta is None else float(swiglu_beta)
    if not math.isfinite(beta):
        raise ValueError(f"swiglu_beta must be finite, got {beta}")
    return beta


def is_gated_activation(activation: str) -> bool:
    return activation in GATED_MOE_ACTIVATIONS


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
