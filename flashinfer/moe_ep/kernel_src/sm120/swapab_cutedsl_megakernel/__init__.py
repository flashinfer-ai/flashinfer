# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM120 (Blackwell-consumer) swap-AB CuTeDSL MegaMoE kernel tree.

Public boundary for ``moe_ep``: FlashInfer backends import THIS module only.
Layering (see SKILL.md): ``__init__`` re-exports from ``shim/`` (our
adapters), ``shim/`` imports the raw kernel packages from ``src/`` (a verbatim
kernel-team drop; see VENDOR.md for provenance) via sys.path bootstrap.

Importing this package is CPU-safe; helpers that pull ``cutlass`` are exposed
lazily via PEP 562 (this drop's ``common.megamoe_constants`` imports cutlass
at module load, so ALL kernel-package helpers are lazy — SM90-tree style).
"""

from .shim import (
    MegaMoESm120Mxfp8Config,
    MegaMoESm120Mxfp8Frontend,
    MegaMoESm120Mxfp8Inputs,
    MegaMoESm120Mxfp8SymmBuffer,
    TransformedWeights,
    bootstrap_paths,
    create_dummy_inputs,
    finalize_dist,
    get_symm_buffer_for_sm120_mxfp8_mega_moe,
    init_dist,
    sm120_mxfp8_mega_launch_thunk,
    sm120_mxfp8_mega_moe,
)

# Naming parity with the other trees' dummy-input helpers.
create_dummy_sm120_mxfp8_inputs = create_dummy_inputs

_LAZY_HELPERS = (
    "CTA_TOKEN_TILE",
    "Mxfp8BlockSize",
    "Mxfp8ScaleDtype",
    "SWAP_AB_INTERLEAVE",
    "SfPaddingBlock",
    "_make_e8m0_scale_tensor",
    "_make_fp8_tensor",
    "_stack_byte_reinterpretable_tensors",
    "ceil_div",
    "compute_megamoe_reference_mxfp8",
    "kind_data_dtype",
    "mxfp8_quantize_per_block_32_col",
    "mxfp8_quantize_per_block_32_row",
    "round_up",
    "to_blocked",
)


def __getattr__(name):  # PEP 562
    if name in _LAZY_HELPERS:
        from .shim import kernel_helpers

        return getattr(kernel_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_LAZY_HELPERS,
    "MegaMoESm120Mxfp8Config",
    "MegaMoESm120Mxfp8Frontend",
    "MegaMoESm120Mxfp8Inputs",
    "MegaMoESm120Mxfp8SymmBuffer",
    "TransformedWeights",
    "bootstrap_paths",
    "create_dummy_inputs",
    "create_dummy_sm120_mxfp8_inputs",
    "finalize_dist",
    "get_symm_buffer_for_sm120_mxfp8_mega_moe",
    "init_dist",
    "sm120_mxfp8_mega_launch_thunk",
    "sm120_mxfp8_mega_moe",
]
