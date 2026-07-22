# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Curated re-exports of raw-kernel helpers/constants/reference from ``../src``.

SM90 counterpart of ``kernel_src/sm100/cutedsl_megamoe/shim/kernel_helpers.py``:
the SINGLE shim file that surfaces kernel-team utilities which are *not* part
of the frontend / fused-launch surface (FP8 quant helpers, block constants,
byte-reinterpret stacking, and the FP8 torch reference).  The FI backend glue
(``backends/mega/kernel/sm90_pull_fp8/{backend,staging,weights}.py``) and the
verification tests consume these through the package ``__init__`` so that
**nothing outside ``shim/`` imports ``src/`` packages directly**.  Keeping them
all here means a new ``src/`` drop has ONE file to re-audit for renamed helpers
instead of a dozen scattered call sites.

Import-safety: unlike the SM100 tree, EVERYTHING here is lazy (PEP 562 module
``__getattr__``): this drop's ``common.megamoe_constants`` imports
``cutlass.cutlass_dsl`` at module load, so even the "light" constants would
pull cutlass into the import-time path.  First attribute access happens inside
backend/test call sites, never at package load.
"""

from __future__ import annotations

import importlib

# name -> (module, attribute).  All of these pull cutlass transitively via
# common.megamoe_constants; resolved on first access only.
_LAZY = {
    # constants (common/megamoe_constants.py)
    "Fp8BlockScaleK": ("common.megamoe_constants", "Fp8BlockScaleK"),
    "Fp8E8M0SfVecSize": ("common.megamoe_constants", "Fp8E8M0SfVecSize"),
    "Fp8Fc2ActivationScaleK": ("common.megamoe_constants", "Fp8Fc2ActivationScaleK"),
    "Fp8GateUpInterleave": ("common.megamoe_constants", "Fp8GateUpInterleave"),
    "Fp8WeightScaleBlockK": ("common.megamoe_constants", "Fp8WeightScaleBlockK"),
    "Fp8WeightScaleBlockN": ("common.megamoe_constants", "Fp8WeightScaleBlockN"),
    # generic host helpers (common/host_utils.py, moe_nvfp4_swapab/runner_common.py)
    "kind_data_dtype": ("common.host_utils", "kind_data_dtype"),
    "ceil_div": ("moe_nvfp4_swapab.runner_common", "ceil_div"),
    "round_up": ("moe_nvfp4_swapab.runner_common", "round_up"),
    "to_blocked": ("moe_nvfp4_swapab.runner_common", "to_blocked"),
    "_stack_byte_reinterpretable_tensors": (
        "moe_nvfp4_swapab.runner_common",
        "_stack_byte_reinterpretable_tensors",
    ),
    # FP8 quant helpers (moe_hopper_fp8/hopper_moe_utils.py)
    "fp8_dtype_max": ("moe_hopper_fp8.hopper_moe_utils", "fp8_dtype_max"),
    "create_fp8_tensor": ("moe_hopper_fp8.hopper_moe_utils", "create_fp8_tensor"),
    "make_constant_block_scale": (
        "moe_hopper_fp8.hopper_moe_utils",
        "make_constant_block_scale",
    ),
    "make_fp8_per_tensor_dequant_scale": (
        "moe_hopper_fp8.hopper_moe_utils",
        "make_fp8_per_tensor_dequant_scale",
    ),
    "quantize_fp8_per_token_block": (
        "moe_hopper_fp8.hopper_moe_utils",
        "quantize_fp8_per_token_block",
    ),
    "quantize_fp8_weight_block_nk": (
        "moe_hopper_fp8.hopper_moe_utils",
        "quantize_fp8_weight_block_nk",
    ),
    # ground-truth torch reference (moe_hopper_fp8/mega_reference_fp8.py)
    "compute_megamoe_reference_fp8": (
        "moe_hopper_fp8.mega_reference_fp8",
        "compute_megamoe_reference_fp8",
    ),
}


def __getattr__(name):  # PEP 562: keep cutlass out of the import-time path
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module), attr)


__all__ = list(_LAZY)  # all lazy (resolved via PEP 562 __getattr__ above)
