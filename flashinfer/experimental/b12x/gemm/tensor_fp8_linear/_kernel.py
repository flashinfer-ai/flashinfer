"""Compatibility aliases for :mod:`b12x.gemm.blockscaled`.

New code should use ``b12x.gemm.blockscaled`` directly.
"""

from ..blockscaled._linear import (
    TensorFP8LinearWeight,
    _use_block_fp8_recipe,
    is_tensor_fp8_linear_supported,
    pack_tensor_fp8_linear_weight,
    tensor_fp8_linear,
)

__all__ = [
    "TensorFP8LinearWeight",
    "_use_block_fp8_recipe",
    "is_tensor_fp8_linear_supported",
    "pack_tensor_fp8_linear_weight",
    "tensor_fp8_linear",
]
