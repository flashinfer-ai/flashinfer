# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Thin adapters over the raw CuTeDSL kernel sources under ``../src``.

``comm`` holds dist / symmetric-heap / compile helpers; ``nvfp4`` and ``mxfp8``
each hold their dtype's lazy-compile frontend plus the symmetric-buffer +
fused-launch wrappers.  The parent :mod:`..api` re-exports the curated subset
that FlashInfer ``moe_ep`` consumes.
"""

from __future__ import annotations

# The per-dtype modules import the raw kernel packages (moe_nvfp4_swapab,
# common, ...) at module load, so ``src/`` must be on sys.path first.
from ..src._bootstrap_paths import bootstrap_paths

bootstrap_paths()

from .comm import (
    bootstrap_dist,
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)
from .nvfp4 import (
    MegaMoENvfp4Config,
    MegaMoENvfp4Frontend,
    MegaMoENvfp4Inputs,
    MegaMoESymmBuffer,
    TransformedWeights,
    create_dummy_inputs as create_dummy_nvfp4_inputs,
    get_symm_buffer_for_mega_moe,
    init_dist,
    make_dummy_epilogue_params,
    nvfp4_mega_moe,
)
from .mxfp8 import (
    MegaMoEMxfp8Config,
    MegaMoEMxfp8Frontend,
    MegaMoEMxfp8Inputs,
    MegaMoEMxfp8SymmBuffer,
    create_dummy_inputs as create_dummy_mxfp8_inputs,
    get_symm_buffer_for_mxfp8_mega_moe,
    mxfp8_mega_moe,
)

__all__ = [
    # comm
    "bootstrap_dist",
    "free_sym_tensor",
    "reset_compiled_mega_workspaces",
    "resolve_gate_up_clamp",
    "sym_zeros",
    # nvfp4
    "MegaMoENvfp4Config",
    "MegaMoENvfp4Frontend",
    "MegaMoENvfp4Inputs",
    "MegaMoESymmBuffer",
    "TransformedWeights",
    "create_dummy_nvfp4_inputs",
    "get_symm_buffer_for_mega_moe",
    "init_dist",
    "make_dummy_epilogue_params",
    "nvfp4_mega_moe",
    # mxfp8
    "MegaMoEMxfp8Config",
    "MegaMoEMxfp8Frontend",
    "MegaMoEMxfp8Inputs",
    "MegaMoEMxfp8SymmBuffer",
    "create_dummy_mxfp8_inputs",
    "get_symm_buffer_for_mxfp8_mega_moe",
    "mxfp8_mega_moe",
]
