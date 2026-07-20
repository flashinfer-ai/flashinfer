# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/__init__.py @ 377083ec (2026-06-03) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from .dynamic import MoEDynamicKernelBackend
from .micro import MoEMicroKernelBackend
from .relu2 import MoEDynamicKernelRelu2, MoEMicroKernelRelu2
from .silu import MoEDynamicKernelSilu, MoEMicroKernelSilu
from .reference import (
    MoERouteTrace,
    OracleMetrics,
    compare_to_reference,
    moe_reference_f32,
    moe_reference_nvfp4,
    trace_moe_reference_nvfp4_route,
)

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
