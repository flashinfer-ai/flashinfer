"""Shared abstractions for moe_ep backends."""

from .comm.fleet import Fleet, create_fleet
from .comm.handle import Handle
from .kernel.base import MegaKernelBackend, SplitKernelBackend, SplitKernelContext
from .kernel.registry import (
    create_mega_kernel,
    create_split_kernel,
    kernel_requires_weights,
    run_split_kernel,
)

__all__ = [
    "Fleet",
    "Handle",
    "MegaKernelBackend",
    "SplitKernelBackend",
    "SplitKernelContext",
    "create_fleet",
    "create_mega_kernel",
    "create_split_kernel",
    "kernel_requires_weights",
    "run_split_kernel",
]
