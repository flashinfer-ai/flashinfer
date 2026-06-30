"""Blackwell SM12x (SM120/SM121) MoE kernels for CuTe DSL (ported from b12x)."""

from .moe_dynamic_kernel import (
    MoEDynamicKernelBackend,
    MoEDynamicKernelRelu2,
    MoEDynamicKernelSilu,
)
from .moe_micro_kernel import (
    MoEMicroKernelBackend,
    MoEMicroKernelRelu2,
    MoEMicroKernelSilu,
)

# Default (gated SiLU) aliases.
MoEDynamicKernel = MoEDynamicKernelSilu
MoEMicroKernel = MoEMicroKernelSilu

__all__ = [
    "MoEDynamicKernelBackend",
    "MoEDynamicKernel",
    "MoEDynamicKernelRelu2",
    "MoEDynamicKernelSilu",
    "MoEMicroKernelBackend",
    "MoEMicroKernel",
    "MoEMicroKernelRelu2",
    "MoEMicroKernelSilu",
]
