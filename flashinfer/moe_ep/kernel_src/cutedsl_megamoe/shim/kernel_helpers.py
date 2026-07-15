# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Curated re-exports of raw-kernel helpers/constants/reference from ``../src``.

This is the SINGLE shim file that surfaces kernel-team utilities which are *not*
part of the frontend / fused-launch surface (block-size constants, quant/layout
helpers, byte-reinterpret stacking, and the MXFP8 torch reference).  The FI
backend glue (``backends/mega/kernel/{nvfp4,mxfp8}_cutedsl/{backend,staging,
weights}.py``) and the verification tests consume these through the package
``__init__`` so that **nothing outside ``shim/`` imports ``src/`` packages
directly**.  Keeping them all here means a new ``src/`` drop has ONE file to
re-audit for renamed helpers instead of a dozen scattered call sites.

Import-safety: the light constants/helpers are re-exported eagerly (they mirror
what the dtype shims already import at load).  The ``mega_runner`` /
``mega_reference`` helpers pull ``cutlass`` transitively (see
``moe_nvfp4_swapab/mega_runner.py`` ``import cutlass``), so they are exposed
lazily via module ``__getattr__`` to keep ``import
flashinfer.moe_ep.kernel_src.cutedsl_megamoe`` usable on CPU-only hosts.
"""

from __future__ import annotations

import importlib

# --- eager: light, import-safe (no cutlass/nvshmem pulled at module load) ---
from common.host_utils import kind_data_dtype, mxfp8_quantize_per_block_32
from common.megamoe_constants import Mxfp8BlockSize, Nvfp4BlockSize
from moe_nvfp4_swapab.runner_common import (
    Mxfp8ScaleDtype,
    _stack_byte_reinterpretable_tensors,
    ceil_div,
    nvfp4_quantize_per_block_16,
    round_up,
    to_blocked,
)

# --- lazy: pull cutlass transitively; imported only on first attribute access
#     (which happens inside the backend/test call sites, never at package load).
_LAZY = {
    "_make_fp8_tensor": ("moe_mxfp8_glu.mega_runner", "_make_fp8_tensor"),
    "_make_e8m0_scale_tensor": ("moe_mxfp8_glu.mega_runner", "_make_e8m0_scale_tensor"),
    "compute_megamoe_reference_mxfp8": (
        "moe_mxfp8_glu.mega_reference_mxfp8",
        "compute_megamoe_reference_mxfp8",
    ),
}


def __getattr__(name):  # PEP 562: keep cutlass out of the import-time path
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module), attr)


__all__ = [
    # eager
    "Mxfp8BlockSize",
    "Mxfp8ScaleDtype",
    "Nvfp4BlockSize",
    "ceil_div",
    "kind_data_dtype",
    "mxfp8_quantize_per_block_32",
    "nvfp4_quantize_per_block_16",
    "round_up",
    "to_blocked",
    "_stack_byte_reinterpretable_tensors",
    # lazy (resolved via PEP 562 __getattr__ above)
    "_make_e8m0_scale_tensor",  # noqa: F822
    "_make_fp8_tensor",  # noqa: F822
    "compute_megamoe_reference_mxfp8",  # noqa: F822
]
