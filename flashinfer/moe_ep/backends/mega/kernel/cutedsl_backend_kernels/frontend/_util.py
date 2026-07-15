# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared helpers for the thin MegaMoE frontend APIs."""

from __future__ import annotations

import warnings
from typing import Optional


def resolve_gate_up_clamp(
    *,
    gate_up_clamp: Optional[float],
    activation_clamp: Optional[float],
) -> Optional[float]:
    """Return the effective gate-up clamp, rejecting conflicting alias args."""
    if gate_up_clamp is not None and activation_clamp is not None:
        if gate_up_clamp != activation_clamp:
            raise ValueError(
                "gate_up_clamp and activation_clamp disagree "
                f"({gate_up_clamp} vs {activation_clamp}); pass only one."
            )
        warnings.warn(
            "activation_clamp is deprecated; use gate_up_clamp.",
            DeprecationWarning,
            stacklevel=3,
        )
    if gate_up_clamp is not None:
        return gate_up_clamp
    if activation_clamp is not None:
        warnings.warn(
            "activation_clamp is deprecated; use gate_up_clamp.",
            DeprecationWarning,
            stacklevel=3,
        )
        return activation_clamp
    return None
