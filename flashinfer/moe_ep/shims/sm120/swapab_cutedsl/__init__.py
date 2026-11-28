# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM120 swap-AB MegaMoE adapters. Compiler and device imports stay lazy."""

from flashinfer.moe_ep.shims.sm120.swapab_cutedsl._paths import bootstrap_paths

bootstrap_paths()

from flashinfer.moe_ep.shims.sm120.swapab_cutedsl.comm import (
    bootstrap_dist,
    ensure_not_capturing,
    finalize_dist,
    free_sym_tensor,
    resolve_gate_up_clamp,
    sym_zeros,
)
from flashinfer.moe_ep.shims.sm120.swapab_cutedsl.mxfp8 import (
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
        from flashinfer.moe_ep.shims.sm120.swapab_cutedsl import kernel_helpers

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
