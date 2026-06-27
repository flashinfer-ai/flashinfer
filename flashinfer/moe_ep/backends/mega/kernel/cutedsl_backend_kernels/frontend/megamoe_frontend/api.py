# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Backward-compatible re-exports for NVFP4 MegaMoE frontend types.

Prefer importing from :mod:`megamoe_frontend.common` (dist / sym heap) and
:mod:`megamoe_frontend.api_nvfp4` (NVFP4 kernel wrapper) directly.
"""

from __future__ import annotations

from .api_nvfp4 import (
    MegaMoENvfp4Config,
    MegaMoENvfp4Frontend,
    MegaMoENvfp4Inputs,
)
from .common import bootstrap_dist, free_sym_tensor, sym_zeros

# Deprecated aliases kept for callers not yet updated to dtype-specific names.
MegaMoEConfig = MegaMoENvfp4Config
MegaMoEFrontend = MegaMoENvfp4Frontend
MegaMoEInputs = MegaMoENvfp4Inputs

__all__ = [
    "MegaMoENvfp4Config",
    "MegaMoENvfp4Frontend",
    "MegaMoENvfp4Inputs",
    "MegaMoEConfig",
    "MegaMoEFrontend",
    "MegaMoEInputs",
    "bootstrap_dist",
    "free_sym_tensor",
    "sym_zeros",
]
