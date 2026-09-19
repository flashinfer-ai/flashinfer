"""Small compiler-aware helpers shared by MUSA Mamba providers."""

from contextlib import nullcontext
import math

import torch


DT_MAX = 3.4028234663852886e38


def device_context(device_index: int):
    """Select a device eagerly without inserting a graph-time context op."""
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if callable(is_compiling) and is_compiling():
        # The compiled graph is launched on tensors already placed on the
        # current MUSA device. A device context is Python-only and is not
        # representable in an Inductor graph.
        return nullcontext()
    return torch.accelerator.device_index(device_index)


def finite_dt_limit(dt_limit):
    """Make the MUSA Triton constexpr bound code-generator friendly."""
    if math.isinf(dt_limit[1]):
        return (dt_limit[0], DT_MAX)
    return dt_limit
