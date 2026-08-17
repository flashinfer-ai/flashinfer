# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shim layer for the ``next_cutedsl_megamoe`` drop (SM107 inference).

``bootstrap_paths()`` must run before anything imports the vendored
``sources`` package.  The re-exported modules keep their cutlass / cuda / drop
imports lazy, so importing this package stays CPU-safe.
"""

from ._paths import bootstrap_paths

bootstrap_paths()

from .block_scaled import (
    Sm107BlockScaledMoeConfig,
    Sm107BlockScaledSymmBuffer,
    Sm107QuantKind,
    TransformedBlockScaledWeights,
    get_symm_buffer_for_sm107_block_scaled_mega_moe,
    sm107_block_scaled_mega_launch_thunk,
    sm107_block_scaled_mega_moe,
)
from .comm import ensure_not_capturing, free_sym_tensor, sym_zeros

__all__ = [
    "Sm107BlockScaledMoeConfig",
    "Sm107BlockScaledSymmBuffer",
    "Sm107QuantKind",
    "TransformedBlockScaledWeights",
    "bootstrap_paths",
    "ensure_not_capturing",
    "free_sym_tensor",
    "get_symm_buffer_for_sm107_block_scaled_mega_moe",
    "sm107_block_scaled_mega_launch_thunk",
    "sm107_block_scaled_mega_moe",
    "sym_zeros",
]
