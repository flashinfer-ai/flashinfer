# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Thin adapters over the vendored SM120 swap-AB kernel ``src/`` (our code).

All adaptation of the verbatim kernel drop lives here; the package
``__init__`` re-exports this module's surface.  Path bootstrap MUST run before
any sibling shim module imports a kernel package.

Importing this package stays CPU-safe: the frontend module defers every
``cutlass`` / ``cuda`` / kernel-package import to call time, and all
``kernel_helpers`` re-exports are lazy (this drop's constants module pulls
cutlass at load, like the SM90 tree's).
"""

from ._paths import bootstrap_paths

bootstrap_paths()

from .comm import (  # noqa: E402
    bootstrap_dist,
    ensure_not_capturing,
    finalize_dist,
    free_sym_tensor,
    resolve_gate_up_clamp,
    sym_zeros,
)
from .mxfp8 import (  # noqa: E402
    MegaMoESm120Mxfp8Config,
    MegaMoESm120Mxfp8Frontend,
    MegaMoESm120Mxfp8Inputs,
    MegaMoESm120Mxfp8SymmBuffer,
    TransformedWeights,
    create_dummy_inputs,
    get_symm_buffer_for_sm120_mxfp8_mega_moe,
    init_dist,
    sm120_mxfp8_mega_launch_thunk,
    sm120_mxfp8_mega_moe,
)

__all__ = [
    # paths
    "bootstrap_paths",
    # comm
    "bootstrap_dist",
    "ensure_not_capturing",
    "finalize_dist",
    "free_sym_tensor",
    "resolve_gate_up_clamp",
    "sym_zeros",
    # mxfp8 frontend
    "MegaMoESm120Mxfp8Config",
    "MegaMoESm120Mxfp8Frontend",
    "MegaMoESm120Mxfp8Inputs",
    "MegaMoESm120Mxfp8SymmBuffer",
    "TransformedWeights",
    "create_dummy_inputs",
    "get_symm_buffer_for_sm120_mxfp8_mega_moe",
    "init_dist",
    "sm120_mxfp8_mega_launch_thunk",
    "sm120_mxfp8_mega_moe",
]
