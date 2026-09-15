# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""NVFP4 W4A16 MegaMoE frontend and autotuning."""

from .frontend import (
    MegaMoEW4A16Config,
    MegaMoEW4A16Frontend,
    MegaMoEW4A16Inputs,
    MegaMoEW4A16SymmBuffer,
    TransformedWeights,
    get_symm_buffer_for_w4a16_mega_moe,
    init_dist,
    w4a16_mega_launch_thunk,
    w4a16_mega_moe,
)
from .autotune import autotune_w4a16_mega_moe, w4a16_candidates

__all__ = [
    "MegaMoEW4A16Config",
    "MegaMoEW4A16Frontend",
    "MegaMoEW4A16Inputs",
    "MegaMoEW4A16SymmBuffer",
    "TransformedWeights",
    "get_symm_buffer_for_w4a16_mega_moe",
    "init_dist",
    "w4a16_mega_launch_thunk",
    "w4a16_mega_moe",
    "autotune_w4a16_mega_moe",
    "w4a16_candidates",
]
