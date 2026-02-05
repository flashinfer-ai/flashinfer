"""
Backwards compatibility stub for flashinfer.fp8_quantization.

This module re-exports all symbols from flashinfer.quantization.fp8_quantization
to maintain backwards compatibility with existing code that imports from
flashinfer.fp8_quantization.

New code should import from flashinfer.quantization.fp8_quantization directly.
"""

# Re-export everything from the new location
from .quantization.fp8_quantization import (
    mxfp8_quantize,
    mxfp8_dequantize_host,
    get_mxfp8_quantization_sm100_module,
    # Private functions for backwards compatibility
    _compute_swizzled_layout_sf_size,
)

__all__ = [
    "mxfp8_quantize",
    "mxfp8_dequantize_host",
    "get_mxfp8_quantization_sm100_module",
    "_compute_swizzled_layout_sf_size",
]
