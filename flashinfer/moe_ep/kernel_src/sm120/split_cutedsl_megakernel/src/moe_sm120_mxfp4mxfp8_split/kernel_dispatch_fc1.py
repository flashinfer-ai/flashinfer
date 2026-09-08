# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""K1: dispatch + FC1 + SwiGLU + MXFP8 quant for SM120 Split-MegaMoE."""

from .megamoe_kernel import Sm120MegaMoEMxfp8SwapABKernel


def build_sm120_dispatch_fc1_kernel(*args, **kwargs):
    """Build the validated MegaMoE body with an FC1-only scheduler.

    This intentionally remains a factory rather than a Python subclass:
    CuTe DSL decorates ``__call__`` and a second inheritance layer changes
    the resolution of the base kernel's ``super().__call__``.
    """
    kwargs["split_role"] = "k1"
    kwargs["producer_sm_count"] = None
    return Sm120MegaMoEMxfp8SwapABKernel(*args, **kwargs)


# Kept for source compatibility with existing benchmark scripts. New callers
# should use the lower-case factory name so the symbol is not mistaken for a
# CuTe kernel class.
Sm120DispatchFc1Kernel = build_sm120_dispatch_fc1_kernel


__all__ = ["build_sm120_dispatch_fc1_kernel", "Sm120DispatchFc1Kernel"]
