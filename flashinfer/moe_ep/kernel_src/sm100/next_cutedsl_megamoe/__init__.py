# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent SM100/103 GenPhase package; native checks and DSL imports are lazy."""

from .shim import (
    Sm100GenphaseMoeConfig,
    Sm100GenphaseSymmBuffer,
    preprocess_genphase_weights,
    preprocess_genphase_prequantized_weights,
    validate_genphase_weights,
    quantize_genphase_mxfp8,
    sm100_genphase_mega_launch_thunk,
    sm100_genphase_mega_moe,
)
from .shim import bootstrap_paths, output_error, sampled_reference

__all__ = [
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
]
