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

# Importing the shim puts ``kernel_src/cutedsl_megamoe/src`` on sys.path (via
# shim/_paths.bootstrap_paths) before its modules resolve the raw kernel
# packages (moe_nvfp4_swapab, common, ...).  ``bootstrap_paths`` is re-exported
# here so callers (e.g. core runtime) reach it through this public boundary.
from .shim import (
    CORRECTNESS_KNOBS,
    MegaMoEMxfp8SymmBuffer,
    MegaMoESymmBuffer,
    autotune_knobs,
    autotune_mxfp8_mega_moe,
    autotune_nvfp4_mega_moe,
    Mxfp8BlockSize,
    Mxfp8ScaleDtype,
    Nvfp4BlockSize,
    PERF_KNOBS,
    TransformedWeights,
    _stack_byte_reinterpretable_tensors,
    bootstrap_paths,
    ceil_div,
    create_dummy_mxfp8_inputs,
    create_dummy_nvfp4_inputs,
    get_symm_buffer_for_mega_moe,
    get_symm_buffer_for_mxfp8_mega_moe,
    init_dist,
    iter_candidates,
    kind_data_dtype,
    make_dummy_epilogue_params,
    mxfp8_mega_launch_thunk,
    mxfp8_mega_moe,
    mxfp8_quantize_per_block_32,
    nvfp4_mega_launch_thunk,
    nvfp4_mega_moe,
    nvfp4_quantize_per_block_16,
    round_up,
    to_blocked,
    tuner,
    with_knobs,
)

# Heavy kernel helpers (``mega_runner`` byte-stacking, ``mega_runner`` fp8/E8M0
# tensor makers, and the MXFP8 torch reference) pull ``cutlass`` transitively.
# Expose them lazily so ``import ...cutedsl_megamoe`` stays CPU-safe; the FI
# backend + verification tests still reach them only through this boundary (the
# access happens inside their functions).  See ``shim/kernel_helpers.py``.
_LAZY_HELPERS = (
    "_make_e8m0_scale_tensor",
    "_make_fp8_tensor",
    "compute_megamoe_reference_mxfp8",
)


def __getattr__(name):  # PEP 562
    if name in _LAZY_HELPERS:
        from .shim import kernel_helpers

        return getattr(kernel_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Backward-compatible name: ``create_dummy_inputs`` historically meant NVFP4.
create_dummy_inputs = create_dummy_nvfp4_inputs

__all__ = [
    "MegaMoEMxfp8SymmBuffer",
    "MegaMoESymmBuffer",
    "Mxfp8BlockSize",
    "Mxfp8ScaleDtype",
    "Nvfp4BlockSize",
    "TransformedWeights",
    "autotune_knobs",
    "autotune_mxfp8_mega_moe",
    "autotune_nvfp4_mega_moe",
    "bootstrap_paths",
    "ceil_div",
    "compute_megamoe_reference_mxfp8",
    "create_dummy_inputs",
    "create_dummy_mxfp8_inputs",
    "create_dummy_nvfp4_inputs",
    "get_symm_buffer_for_mega_moe",
    "get_symm_buffer_for_mxfp8_mega_moe",
    "init_dist",
    "kind_data_dtype",
    "make_dummy_epilogue_params",
    "mxfp8_mega_launch_thunk",
    "mxfp8_mega_moe",
    "mxfp8_quantize_per_block_32",
    "nvfp4_mega_launch_thunk",
    "nvfp4_mega_moe",
    "nvfp4_quantize_per_block_16",
    "round_up",
    "to_blocked",
    "tuner",
    "with_knobs",
    "iter_candidates",
    "CORRECTNESS_KNOBS",
    "PERF_KNOBS",
    "_make_e8m0_scale_tensor",
    "_make_fp8_tensor",
    "_stack_byte_reinterpretable_tensors",
]
