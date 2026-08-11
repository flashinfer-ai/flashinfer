# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Thin adapters over the SM90 (Hopper) ``src/`` kernel drop.

Mirrors the SM100 package's shim layer
(``kernel_src/sm100/cutedsl_megamoe/shim``): all adaptation over the verbatim
``src/`` drop lives here, and the package ``__init__`` re-exports only from
this layer.  ``comm`` holds dist / symmetric-heap / compile helpers;
``hopper_fp8`` holds the SM90 FP8 lazy-compile frontend plus the
symmetric-buffer + fused-launch wrappers.
"""

from __future__ import annotations

# hopper_fp8 imports the raw kernel packages (moe_hopper_fp8, common, ...)
# lazily, but src/ must be on sys.path before any of them resolve.  The
# bootstrap lives here in shim/ (not in src/, which is a verbatim kernel drop)
# and also guards against the sibling SM100 tree owning this process.
from ._paths import bootstrap_paths

bootstrap_paths()

from .comm import (
    bootstrap_dist,
    finalize_dist,
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)
from .hopper_fp8 import (
    MegaMoEHopperFp8Config,
    MegaMoEHopperFp8Frontend,
    MegaMoEHopperFp8Inputs,
    MegaMoEHopperFp8SymmBuffer,
    TransformedFp8Weights,
    create_dummy_inputs as create_dummy_hopper_fp8_inputs,
    get_symm_buffer_for_hopper_fp8_mega_moe,
    hopper_fp8_mega_launch_thunk,
    hopper_fp8_mega_moe,
    init_dist,
)

__all__ = [
    # paths
    "bootstrap_paths",
    # comm
    "bootstrap_dist",
    "finalize_dist",
    "free_sym_tensor",
    "reset_compiled_mega_workspaces",
    "resolve_gate_up_clamp",
    "sym_zeros",
    # hopper_fp8
    "MegaMoEHopperFp8Config",
    "MegaMoEHopperFp8Frontend",
    "MegaMoEHopperFp8Inputs",
    "MegaMoEHopperFp8SymmBuffer",
    "TransformedFp8Weights",
    "create_dummy_hopper_fp8_inputs",
    "get_symm_buffer_for_hopper_fp8_mega_moe",
    "hopper_fp8_mega_launch_thunk",
    "hopper_fp8_mega_moe",
    "init_dist",
]
