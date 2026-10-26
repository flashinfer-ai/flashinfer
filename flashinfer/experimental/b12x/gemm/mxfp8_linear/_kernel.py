"""Compatibility aliases for :mod:`b12x.gemm.blockscaled`.

New code should use ``b12x.gemm.blockscaled`` directly.
"""

from ..blockscaled._linear import (
    MXFP8LinearWeight,
    is_mxfp8_linear_supported,
    mxfp8_linear,
    pack_mxfp8_linear_weight,
)

__all__ = [
    "MXFP8LinearWeight",
    "is_mxfp8_linear_supported",
    "mxfp8_linear",
    "pack_mxfp8_linear_weight",
]
