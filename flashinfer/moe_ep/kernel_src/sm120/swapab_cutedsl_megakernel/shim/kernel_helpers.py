# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Single re-export point for raw-kernel helpers the FI backend + tests need.

Everything the FlashInfer backend or the tests consume from the raw kernel
packages (``common``, ``moe_sm120_mxfp8_swapab``, ``moe_mxfp8_glu``) is
funnelled through here, so a kernel drop that renames a helper breaks in ONE
file (the drop-audit point — see SKILL.md).

Every re-export is lazy (PEP 562): this drop's ``common.megamoe_constants``
imports ``cutlass.cutlass_dsl`` at module load (same as the SM90 tree), so
even the "light" constants would pull cutlass into the import-time path and
break the CPU-safe package import contract.
"""

from __future__ import annotations

import importlib

# name -> (module, attr).  Modules resolve through the sys.path bootstrap in
# _paths (this tree's src/), performed by shim/__init__ before first access.
_LAZY = {
    # constants (common/megamoe_constants.py)
    "Mxfp8BlockSize": ("common.megamoe_constants", "Mxfp8BlockSize"),
    "SfPaddingBlock": ("common.megamoe_constants", "SfPaddingBlock"),
    # SM120 swap-AB constants (moe_sm120_mxfp8_swapab/sm120_mma.py)
    "CTA_TOKEN_TILE": ("moe_sm120_mxfp8_swapab.sm120_mma", "CTA_TOKEN_TILE"),
    "SWAP_AB_INTERLEAVE": ("moe_sm120_mxfp8_swapab.sm120_mma", "SWAP_AB_INTERLEAVE"),
    # generic host helpers (moe_sm120_mxfp8_swapab/runner_common.py)
    "Mxfp8ScaleDtype": ("moe_sm120_mxfp8_swapab.runner_common", "Mxfp8ScaleDtype"),
    "ceil_div": ("moe_sm120_mxfp8_swapab.runner_common", "ceil_div"),
    "round_up": ("moe_sm120_mxfp8_swapab.runner_common", "round_up"),
    "to_blocked": ("moe_sm120_mxfp8_swapab.runner_common", "to_blocked"),
    "_stack_byte_reinterpretable_tensors": (
        "moe_sm120_mxfp8_swapab.runner_common",
        "_stack_byte_reinterpretable_tensors",
    ),
    # quantize helpers (common/host_utils.py; this drop split the old
    # mxfp8_quantize_per_block_32 into _row/_col variants)
    "kind_data_dtype": ("common.host_utils", "kind_data_dtype"),
    "mxfp8_quantize_per_block_32_row": (
        "common.host_utils",
        "mxfp8_quantize_per_block_32_row",
    ),
    "mxfp8_quantize_per_block_32_col": (
        "common.host_utils",
        "mxfp8_quantize_per_block_32_col",
    ),
    # random-tensor makers (this drop carries its own copies in the SM120
    # runner; the moe_mxfp8_glu twins would work but pull a wider closure)
    "_make_fp8_tensor": ("moe_sm120_mxfp8_swapab.mega_runner", "_make_fp8_tensor"),
    "_make_e8m0_scale_tensor": (
        "moe_sm120_mxfp8_swapab.mega_runner",
        "_make_e8m0_scale_tensor",
    ),
    # torch reference: the SM120 wrapper pins gate_up_interleave=8 (swap-AB
    # register interleave) and apply_topk_in_fc1=True over the generic
    # moe_mxfp8_glu reference
    "compute_megamoe_reference_mxfp8": (
        "moe_sm120_mxfp8_swapab.mega_reference",
        "compute_megamoe_reference_mxfp8",
    ),
}


def __getattr__(name):  # PEP 562: keep cutlass out of the import-time path
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module), attr)


__all__ = list(_LAZY)
