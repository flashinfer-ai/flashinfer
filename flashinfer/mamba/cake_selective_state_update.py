"""Cake backend entry point for selective state update."""

from __future__ import annotations

from typing import Any

import torch

from .selective_state_update import selective_state_update


def cake_selective_state_update(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Run selective state update through the Cake backend when supported."""
    return selective_state_update(*args, backend="cake", **kwargs)


__all__ = ["cake_selective_state_update"]
