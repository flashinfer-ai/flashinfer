# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Shim layer for the ``next_cutedsl_megamoe`` drop (SM107 GLU fprop).

``bootstrap_paths()`` must run before anything imports the vendored
``sources`` package.  The re-exported modules keep their cutlass / cuda / drop
imports lazy, so importing this package stays CPU-safe.
"""

from ._paths import bootstrap_paths

bootstrap_paths()

from .comm import ensure_not_capturing, free_sym_tensor, sym_zeros
from .mxfp8_glu import (
    Sm107MegaMoEMxfp8GluConfig,
    Sm107Mxfp8GluKind,
    Sm107Mxfp8GluSymmBuffer,
    TransformedMxfp8GluWeights,
    get_symm_buffer_for_sm107_mxfp8_glu_mega_moe,
    sm107_mxfp8_glu_mega_launch_thunk,
    sm107_mxfp8_glu_mega_moe,
)

__all__ = [
    "Sm107MegaMoEMxfp8GluConfig",
    "Sm107Mxfp8GluKind",
    "Sm107Mxfp8GluSymmBuffer",
    "TransformedMxfp8GluWeights",
    "bootstrap_paths",
    "ensure_not_capturing",
    "free_sym_tensor",
    "get_symm_buffer_for_sm107_mxfp8_glu_mega_moe",
    "sm107_mxfp8_glu_mega_launch_thunk",
    "sm107_mxfp8_glu_mega_moe",
    "sym_zeros",
]
