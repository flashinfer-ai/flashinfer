# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rubin standard and GenPhase MegaMoE adapters. Compiler and device imports stay lazy."""

from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe._paths import bootstrap_paths

bootstrap_paths()

from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.autotune import (
    KNOB_KEYS,
    autotune_sm107_block_scaled_mega_moe,
    is_valid_sm107,
    sm107_candidates,
    sm107_schedule_candidates,
)
from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.block_scaled import (
    Sm107BlockScaledMoeConfig,
    Sm107BlockScaledSymmBuffer,
    Sm107QuantKind,
    TransformedBlockScaledWeights,
    get_symm_buffer_for_sm107_block_scaled_mega_moe,
    sm107_block_scaled_mega_launch_thunk,
    sm107_block_scaled_mega_moe,
)
from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.comm import (
    ensure_not_capturing,
    free_sym_tensor,
    sym_zeros,
)
from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.dependencies import require_sm107_dsl
from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.correctness import (
    output_error,
    sampled_reference,
)
from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.knob_cache import (
    default_knobs,
    knob_cache_path,
    lookup_knobs,
    record_knobs,
    resolve_knobs,
)


from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe.shared_expert import (
    Sm107SharedExpertConfig,
    validate_feature_config,
)


# Torch-side kernel helpers (quant / SF swizzle / oracle reference) resolve
# lazily so this package import never pulls anything CUDA-adjacent.
_LAZY_HELPERS = (
    "GateUpInterleave",
    "Mxfp8BlockSize",
    "Nvfp4BlockSize",
    "ceil_div",
    "compute_megamoe_reference_sm107_block_scaled",
    "concatenate_block_scaled_weights",
    "e8m0_to_f32",
    "from_blocked",
    "interleave_gate_up_16",
    "pack_f32_to_fp4",
    "preprocess_block_scaled_weights",
    "preprocess_prequantized_block_scaled_weights",
    "quantize_mxfp4_block32",
    "quantize_mxfp8_block32",
    "quantize_nvfp4_block16",
    "round_up",
    "round_trip_combine",
    "scale_to_f32",
    "swizzled_flat_sf_size",
    "to_blocked",
    "unpack_fp4_to_f32",
)


def __getattr__(name):  # PEP 562
    if name in _LAZY_HELPERS:
        from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import kernel_helpers

        return getattr(kernel_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_LAZY_HELPERS,
    "KNOB_KEYS",
    "Sm107BlockScaledMoeConfig",
    "Sm107SharedExpertConfig",
    "Sm107BlockScaledSymmBuffer",
    "Sm107QuantKind",
    "TransformedBlockScaledWeights",
    "autotune_sm107_block_scaled_mega_moe",
    "bootstrap_paths",
    "default_knobs",
    "ensure_not_capturing",
    "free_sym_tensor",
    "get_symm_buffer_for_sm107_block_scaled_mega_moe",
    "is_valid_sm107",
    "knob_cache_path",
    "lookup_knobs",
    "output_error",
    "record_knobs",
    "require_sm107_dsl",
    "resolve_knobs",
    "sampled_reference",
    "sm107_block_scaled_mega_launch_thunk",
    "sm107_block_scaled_mega_moe",
    "sm107_candidates",
    "sm107_schedule_candidates",
    "sym_zeros",
]
