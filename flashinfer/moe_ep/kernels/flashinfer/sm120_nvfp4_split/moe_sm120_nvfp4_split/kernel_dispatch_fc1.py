# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""K1: dispatch + FC1 + SwiGLU + NVFP4 quant for SM120 Split-MegaMoE."""

from .megamoe_kernel import Sm120MegaMoENvfp4SwapABKernel


def build_sm120_dispatch_fc1_kernel(*args, **kwargs):
    """Build the validated MegaMoE body with an FC1-only scheduler.

    This intentionally remains a factory rather than a Python subclass:
    CuTe DSL decorates ``__call__`` and a second inheritance layer changes
    the resolution of the base kernel's ``super().__call__``.
    """
    kwargs["split_role"] = "k1"
    kwargs["producer_sm_count"] = None
    return Sm120MegaMoENvfp4SwapABKernel(*args, **kwargs)


__all__ = ["build_sm120_dispatch_fc1_kernel"]
