# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CuTeDSL MegaMoE kernel drop (NVFP4 + MXFP8).

This package is the single public boundary FlashInfer ``moe_ep`` imports from.
It exposes the symmetric-buffer allocators and fused-launch entry points and
talks only to the ``shim`` package (which adapts the raw kernel sources under
``src/``).

Layout::

    __init__.py  public API for moe_ep (this file); talks only to shim/
    shim/        thin adapters over the raw kernel sources (comm, nvfp4, mxfp8)
    src/         vendored kernel sources from the kernel team

Usage::

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        get_symm_buffer_for_mega_moe,
        init_dist,
        nvfp4_mega_moe,
    )
"""

from __future__ import annotations

# Put ``kernel_src/cutedsl_megamoe/src`` on sys.path before importing the shim,
# whose modules resolve the raw kernel packages (moe_nvfp4_swapab, common, ...)
# at import time.  Idempotent; the shim also bootstraps defensively.
from .src._bootstrap_paths import bootstrap_paths

bootstrap_paths()

from .shim import (
    MegaMoEMxfp8SymmBuffer,
    MegaMoESymmBuffer,
    TransformedWeights,
    create_dummy_mxfp8_inputs,
    create_dummy_nvfp4_inputs,
    get_symm_buffer_for_mega_moe,
    get_symm_buffer_for_mxfp8_mega_moe,
    init_dist,
    make_dummy_epilogue_params,
    mxfp8_mega_moe,
    nvfp4_mega_moe,
)

# Backward-compatible name: ``create_dummy_inputs`` historically meant NVFP4.
create_dummy_inputs = create_dummy_nvfp4_inputs

__all__ = [
    "MegaMoEMxfp8SymmBuffer",
    "MegaMoESymmBuffer",
    "TransformedWeights",
    "bootstrap_paths",
    "create_dummy_inputs",
    "create_dummy_mxfp8_inputs",
    "create_dummy_nvfp4_inputs",
    "get_symm_buffer_for_mega_moe",
    "get_symm_buffer_for_mxfp8_mega_moe",
    "init_dist",
    "make_dummy_epilogue_params",
    "mxfp8_mega_moe",
    "nvfp4_mega_moe",
]
