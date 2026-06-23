"""Shared activation metadata for fused MoE paths.

Only ``silu`` (gated SwiGLU) and ``relu2`` (non-gated, squared ReLU) are
supported; any other activation is rejected by ``normalize_moe_activation``.
"""

from __future__ import annotations


SUPPORTED_MOE_ACTIVATIONS = frozenset({"silu", "relu2"})
GATED_MOE_ACTIVATIONS = frozenset({"silu"})


def normalize_moe_activation(activation: str) -> str:
    activation = str(activation)
    if activation not in SUPPORTED_MOE_ACTIVATIONS:
        raise ValueError(f"unsupported activation {activation!r}")
    return activation


def is_gated_moe_activation(activation: str) -> bool:
    return normalize_moe_activation(activation) in GATED_MOE_ACTIVATIONS
