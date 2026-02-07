"""
Backwards compatibility stub for flashinfer.fp4_quantization.

This module re-exports all symbols from flashinfer.quantization.fp4_quantization
to maintain backwards compatibility with existing code that imports from
flashinfer.fp4_quantization.

New code should import from flashinfer.quantization.fp4_quantization directly.
"""

# Re-export everything from the new location
from .quantization.fp4_quantization import (
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
    get_fp4_quantization_module,
    gen_fp4_quantization_module,
    gen_fp4_quantization_sm90_module,
    gen_fp4_quantization_sm100_module,
    gen_fp4_quantization_sm103_module,
    gen_fp4_quantization_sm110_module,
    gen_fp4_quantization_sm120_module,
    gen_fp4_quantization_sm121_module,
    # Private functions needed by some tests
    _pad_scale_factors,
    _compute_swizzled_layout_sf_size,
)

__all__ = [
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
    "gen_fp4_quantization_module",
    "gen_fp4_quantization_sm90_module",
    "gen_fp4_quantization_sm100_module",
    "gen_fp4_quantization_sm103_module",
    "gen_fp4_quantization_sm110_module",
    "gen_fp4_quantization_sm120_module",
    "gen_fp4_quantization_sm121_module",
    "_pad_scale_factors",
    "_compute_swizzled_layout_sf_size",
]
