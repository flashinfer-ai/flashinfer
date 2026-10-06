"""Public surface for gemm.mxfp8_linear (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from ..blockscaled._linear import (
    MXFP8LinearWeight as Weight,
)
from ..blockscaled.api import BlockscaledQuery, FixedBlockscaledQuery, mm, pack_weight, plan, query_from_call
from ._kernel import (
    is_mxfp8_linear_supported as _kernel_is_supported,
)
from . import META


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0, triton, and
    the kernel's own capability checks."""
    kernel_supported, _ = _kernel_is_supported()
    return default_is_supported(device, requires=META.requires) and kernel_supported


__all__ = ["Weight", "BlockscaledQuery", "FixedBlockscaledQuery", "plan", "query_from_call", "mm", "pack_weight", "is_supported"]
