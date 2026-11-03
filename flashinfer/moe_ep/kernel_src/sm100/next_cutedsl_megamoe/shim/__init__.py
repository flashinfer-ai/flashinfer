# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Blackwell adapters; raw source imports stay inside the shim."""

from ._paths import bootstrap_paths
from .autotune import (
    autotune_sm100_genphase_mega_moe,
    autotune_sm100_block_scaled_mega_moe,
)
from .block_scaled import (
    block_scaled_mega_launch_thunk as sm100_block_scaled_mega_launch_thunk,
    block_scaled_mega_moe as sm100_block_scaled_mega_moe,
)
from .correctness import (
    output_error,
    sampled_reference,
)
from .genphase import (
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
from .kernel_helpers import (
    preprocess_block_scaled_weights,
    preprocess_prequantized_block_scaled_weights,
    quantize_mxfp4_block32,
    quantize_mxfp8_block32,
    quantize_nvfp4_block16,
)
from .knob_cache import (
    knob_cache_path,
    live_token_buckets as genphase_live_token_buckets,
    resolve_knobs as resolve_genphase_knobs,
)
from .standard import (
    Sm100BlockScaledMoeConfig,
    Sm100BlockScaledSymmBuffer,
    validate_standard_weights,
)
from .tuning import (
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
