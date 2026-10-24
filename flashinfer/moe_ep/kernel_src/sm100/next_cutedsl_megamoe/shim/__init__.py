# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Blackwell GenPhase host adapters; all raw imports stay in this package."""

from ._paths import bootstrap_paths

bootstrap_paths()

from .genphase import (
    concatenate_genphase_weights,
    Sm100GenphaseMoeConfig,
    Sm100GenphaseSymmBuffer,
    preprocess_genphase_weights,
    preprocess_genphase_prequantized_weights,
    validate_genphase_weights,
    quantize_genphase_mxfp8,
    sm100_genphase_mega_launch_thunk,
    sm100_genphase_mega_moe,
)
from .correctness import output_error, sampled_reference
from .autotune import autotune_sm100_genphase_mega_moe
from .tuning import sm100_genphase_candidates, default_sm100_genphase_knobs
from .knob_cache import (
    resolve_knobs as resolve_genphase_knobs,
    live_token_buckets as genphase_live_token_buckets,
    knob_cache_path,
)

__all__ = [
    "concatenate_genphase_weights",
    "Sm100GenphaseMoeConfig",
    "Sm100GenphaseSymmBuffer",
    "preprocess_genphase_weights",
    "preprocess_genphase_prequantized_weights",
    "validate_genphase_weights",
    "quantize_genphase_mxfp8",
    "sm100_genphase_mega_launch_thunk",
    "sm100_genphase_mega_moe",
    "output_error",
    "sampled_reference",
    "bootstrap_paths",
    "autotune_sm100_genphase_mega_moe",
    "sm100_genphase_candidates",
    "default_sm100_genphase_knobs",
    "resolve_genphase_knobs",
    "genphase_live_token_buckets",
    "knob_cache_path",
]
