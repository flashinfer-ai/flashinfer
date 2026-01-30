"""
FlashInfer Quantization Module
==============================

This module provides quantization functions for various formats:
- FP4 (NVFP4, MXFP4)
- FP8 (MXFP8)
- Packbits utilities

Copyright (c) 2025 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
"""

# Re-export packbits functions
from .packbits import packbits, segment_packbits

# Re-export FP8 quantization
from .fp8_quantization import mxfp8_quantize, mxfp8_dequantize_host

# Re-export FP4 quantization (all public symbols)
from .fp4_quantization import (
    SfLayout,
    block_scale_interleave,
    nvfp4_block_scale_interleave,
    e2m1_and_ufp8sf_scale_to_float,
    fp4_quantize,
    mxfp4_dequantize_host,
    mxfp4_dequantize,
    mxfp4_quantize,
    nvfp4_quantize,
    nvfp4_batched_quantize,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
    scaled_fp4_grouped_quantize,
    get_fp4_quantization_module,  # Used by activation.py
)

# CuTe-DSL kernels (conditionally exported)
# Note: is_cute_dsl_available is used internally but not re-exported;
# users should import from flashinfer.cute_dsl
from ..cute_dsl import is_cute_dsl_available

if is_cute_dsl_available():
    from .mxfp8_quantize_cute_dsl import mxfp8_quantize_cute_dsl

__all__ = [
    # Packbits
    "packbits",
    "segment_packbits",
    # FP8
    "mxfp8_quantize",
    "mxfp8_dequantize_host",
    # FP4
    "SfLayout",
    "block_scale_interleave",
    "nvfp4_block_scale_interleave",
    "e2m1_and_ufp8sf_scale_to_float",
    "fp4_quantize",
    "mxfp4_dequantize_host",
    "mxfp4_dequantize",
    "mxfp4_quantize",
    "nvfp4_quantize",
    "nvfp4_batched_quantize",
    "shuffle_matrix_a",
    "shuffle_matrix_sf_a",
    "scaled_fp4_grouped_quantize",
    "get_fp4_quantization_module",
]

if is_cute_dsl_available():
    __all__ += [
        "mxfp8_quantize_cute_dsl",
    ]
