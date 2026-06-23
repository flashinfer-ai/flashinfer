"""Blackwell SM12x (SM120/SM121) MoE kernels for CuTe DSL (ported from b12x)."""

from .moe_dynamic_kernel import MoEDynamicKernelBackend
from .moe_micro_kernel import MoEMicroKernelBackend
from .moe_relu2 import MoEDynamicKernelRelu2, MoEMicroKernelRelu2
from .moe_silu import MoEDynamicKernelSilu, MoEMicroKernelSilu
from .moe_reference import (
    MoERouteTrace,
    OracleMetrics,
    compare_to_reference,
    moe_reference_f32,
    moe_reference_nvfp4,
    trace_moe_reference_nvfp4_route,
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
    "MoERouteTrace",
    "OracleMetrics",
    "compare_to_reference",
    "moe_reference_f32",
    "moe_reference_nvfp4",
    "trace_moe_reference_nvfp4_route",
]
