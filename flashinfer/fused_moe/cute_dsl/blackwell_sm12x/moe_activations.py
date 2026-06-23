"""Shared activation metadata for fused MoE paths."""

from __future__ import annotations

import math


SWIGLUOAI_UNINTERLEAVE = "swigluoai_uninterleave"
SWIGLUOAI_DEFAULT_LIMIT = 7.0
SWIGLUOAI_DEFAULT_ALPHA = 1.702
SWIGLUOAI_DEFAULT_BETA = 1.0

SUPPORTED_MOE_ACTIVATIONS = frozenset({"silu", "relu2", SWIGLUOAI_UNINTERLEAVE})
GATED_MOE_ACTIVATIONS = frozenset({"silu", SWIGLUOAI_UNINTERLEAVE})


def normalize_moe_activation(activation: str) -> str:
    activation = str(activation)
    if activation not in SUPPORTED_MOE_ACTIVATIONS:
        raise ValueError(f"unsupported activation {activation!r}")
    return activation


def is_gated_moe_activation(activation: str) -> bool:
    return normalize_moe_activation(activation) in GATED_MOE_ACTIVATIONS


def moe_activation_w1_rows(activation: str, n: int) -> int:
    return (2 if is_gated_moe_activation(activation) else 1) * int(n)


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
        if swiglu_alpha is not None and float(swiglu_alpha) != 1.0:
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
        if swiglu_beta is not None and float(swiglu_beta) != 0.0:
            raise ValueError(
                "swiglu_beta is only configurable for swigluoai_uninterleave"
            )
        return 0.0
    beta = SWIGLUOAI_DEFAULT_BETA if swiglu_beta is None else float(swiglu_beta)
    if not math.isfinite(beta):
        raise ValueError(f"swiglu_beta must be finite, got {beta}")
    return beta
