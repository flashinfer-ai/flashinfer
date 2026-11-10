"""Checkpoint transport capabilities without importing PyTorch eagerly."""

from __future__ import annotations

import operator

from ._native import load


def capabilities(device: int = 0) -> dict[str, int]:
    """Query CUDA capabilities used to select the checkpoint transport."""
    return load().capabilities(operator.index(device))
