# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Installable MegaMoE frontend for the CuTeDSL NVFP4 / MXFP8 kernels."""

from __future__ import annotations

from .api_mxfp8 import (
    MegaMoEMxfp8Config,
    MegaMoEMxfp8Frontend,
    MegaMoEMxfp8Inputs,
)
from .api_nvfp4 import (
    MegaMoENvfp4Config,
    MegaMoENvfp4Frontend,
    MegaMoENvfp4Inputs,
)
from .common import bootstrap_dist, free_sym_tensor, sym_zeros

# Backward-compatible aliases (NVFP4-only names).
MegaMoEConfig = MegaMoENvfp4Config
MegaMoEFrontend = MegaMoENvfp4Frontend
MegaMoEInputs = MegaMoENvfp4Inputs

__all__ = [
    "MegaMoENvfp4Config",
    "MegaMoENvfp4Frontend",
    "MegaMoENvfp4Inputs",
    "MegaMoEMxfp8Config",
    "MegaMoEMxfp8Frontend",
    "MegaMoEMxfp8Inputs",
    "MegaMoEConfig",
    "MegaMoEFrontend",
    "MegaMoEInputs",
    "bootstrap_dist",
    "free_sym_tensor",
    "sym_zeros",
]
