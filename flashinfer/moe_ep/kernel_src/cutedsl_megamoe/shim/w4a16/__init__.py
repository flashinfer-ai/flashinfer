# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0
"""W4A16 adaptation of the CuTe DSL BF16 MegaMoE pipeline."""

from .frontend import (
    MegaMoEW4A16Config,
    MegaMoEW4A16Frontend,
    MegaMoEW4A16Inputs,
    MegaMoEW4A16SymmBuffer,
    get_symm_buffer_for_w4a16_mega_moe,
    w4a16_mega_launch_thunk,
    w4a16_mega_moe,
)

__all__ = [
    "MegaMoEW4A16Config",
    "MegaMoEW4A16Frontend",
    "MegaMoEW4A16Inputs",
    "MegaMoEW4A16SymmBuffer",
    "get_symm_buffer_for_w4a16_mega_moe",
    "w4a16_mega_launch_thunk",
    "w4a16_mega_moe",
]
