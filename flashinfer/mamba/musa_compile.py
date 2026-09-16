"""Small compiler-aware helpers shared by MUSA Mamba providers."""

from contextlib import nullcontext

import torch


def device_context(device_index: int):
    """Select a device eagerly without inserting a graph-time context op."""
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if callable(is_compiling) and is_compiling():
        # The compiled graph is launched on tensors already placed on the
        # current MUSA device. A device context is Python-only and is not
        # representable in an Inductor graph.
        return nullcontext()
    return torch.accelerator.device_index(device_index)
