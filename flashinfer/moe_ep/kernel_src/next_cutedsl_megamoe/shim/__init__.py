# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shim layer for the ``next_cutedsl_megamoe`` drop (SM107 inference).

``bootstrap_paths()`` must run before anything imports the vendored
``sources`` package.  The re-exported modules keep their cutlass / cuda / drop
imports lazy, so importing this package stays CPU-safe.
"""

from ._paths import bootstrap_paths

bootstrap_paths()

from .autotune import (
    KNOB_KEYS,
    autotune_sm107_block_scaled_mega_moe,
    is_valid_sm107,
    sm107_candidates,
    sm107_schedule_candidates,
)
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
from .knob_cache import (
    default_knobs,
    knob_cache_path,
    lookup_knobs,
    record_knobs,
    resolve_knobs,
)

__all__ = [
    "KNOB_KEYS",
    "Sm107BlockScaledMoeConfig",
    "Sm107BlockScaledSymmBuffer",
    "Sm107QuantKind",
    "TransformedBlockScaledWeights",
    "autotune_sm107_block_scaled_mega_moe",
    "bootstrap_paths",
    "default_knobs",
    "ensure_not_capturing",
    "free_sym_tensor",
    "get_symm_buffer_for_sm107_block_scaled_mega_moe",
    "is_valid_sm107",
    "knob_cache_path",
    "lookup_knobs",
    "record_knobs",
    "resolve_knobs",
    "sm107_block_scaled_mega_launch_thunk",
    "sm107_block_scaled_mega_moe",
    "sm107_candidates",
    "sm107_schedule_candidates",
    "sym_zeros",
]
