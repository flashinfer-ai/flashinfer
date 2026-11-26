# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Blackwell standard and GenPhase MegaMoE adapters. Compiler and device imports stay lazy."""

from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources._paths import (
    bootstrap_paths,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.autotune import (
    autotune_sm100_genphase_mega_moe,
    autotune_sm100_block_scaled_mega_moe,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.block_scaled import (
    _max_active_clusters as max_active_clusters,
    block_scaled_mega_launch_thunk as sm100_block_scaled_mega_launch_thunk,
    block_scaled_mega_moe as sm100_block_scaled_mega_moe,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.correctness import (
    output_error,
    sampled_reference,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.genphase import (
    Sm100GenphaseMoeConfig,
    Sm100GenphaseSymmBuffer,
    concatenate_genphase_weights,
    preprocess_genphase_prequantized_weights,
    preprocess_genphase_weights,
    quantize_genphase_mxfp8,
    sm100_genphase_mega_launch_thunk,
    sm100_genphase_mega_moe,
    validate_genphase_weights,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.kernel_helpers import (
    preprocess_block_scaled_weights,
    preprocess_prequantized_block_scaled_weights,
    quantize_mxfp4_block32,
    quantize_mxfp8_block32,
    quantize_nvfp4_block16,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.knob_cache import (
    knob_cache_path,
    live_token_buckets as genphase_live_token_buckets,
    resolve_knobs as resolve_genphase_knobs,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.standard import (
    Sm100BlockScaledMoeConfig,
    Sm100BlockScaledSymmBuffer,
    validate_standard_weights,
)
from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources.tuning import (
    default_sm100_genphase_knobs,
    sm100_genphase_candidates,
)

bootstrap_paths()


__all__ = [
    "Sm100BlockScaledMoeConfig",
    "Sm100BlockScaledSymmBuffer",
    "Sm100GenphaseMoeConfig",
    "Sm100GenphaseSymmBuffer",
    "autotune_sm100_genphase_mega_moe",
    "autotune_sm100_block_scaled_mega_moe",
    "bootstrap_paths",
    "concatenate_genphase_weights",
    "default_sm100_genphase_knobs",
    "genphase_live_token_buckets",
    "knob_cache_path",
    "max_active_clusters",
    "output_error",
    "preprocess_block_scaled_weights",
    "preprocess_genphase_prequantized_weights",
    "preprocess_genphase_weights",
    "preprocess_prequantized_block_scaled_weights",
    "quantize_genphase_mxfp8",
    "quantize_mxfp4_block32",
    "quantize_mxfp8_block32",
    "quantize_nvfp4_block16",
    "resolve_genphase_knobs",
    "sampled_reference",
    "sm100_block_scaled_mega_launch_thunk",
    "sm100_block_scaled_mega_moe",
    "sm100_genphase_candidates",
    "sm100_genphase_mega_launch_thunk",
    "sm100_genphase_mega_moe",
    "validate_genphase_weights",
    "validate_standard_weights",
]
