# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""NVFP4 W4A16 MegaMoE frontend and autotuning."""

from .frontend import (
    MegaMoEBf16Nvfp4Config,
    MegaMoEBf16Nvfp4Frontend,
    MegaMoEBf16Nvfp4Inputs,
    MegaMoEBf16Nvfp4SymmBuffer,
    TransformedWeights,
    get_symm_buffer_for_bf16_nvfp4_mega_moe,
    init_dist,
    bf16_nvfp4_mega_launch_thunk,
    bf16_nvfp4_mega_moe,
)
from .autotune import autotune_bf16_nvfp4_mega_moe, bf16_nvfp4_candidates

__all__ = [
    "MegaMoEBf16Nvfp4Config",
    "MegaMoEBf16Nvfp4Frontend",
    "MegaMoEBf16Nvfp4Inputs",
    "MegaMoEBf16Nvfp4SymmBuffer",
    "TransformedWeights",
    "get_symm_buffer_for_bf16_nvfp4_mega_moe",
    "init_dist",
    "bf16_nvfp4_mega_launch_thunk",
    "bf16_nvfp4_mega_moe",
    "autotune_bf16_nvfp4_mega_moe",
    "bf16_nvfp4_candidates",
]
