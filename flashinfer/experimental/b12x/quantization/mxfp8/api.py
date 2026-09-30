"""Public surface for quantization.mxfp8 (docs in the op ``__init__``)."""

from __future__ import annotations

from ..._lib.gating import default_is_supported
from . import META
from ._preparation import Mxfp8Config, Mxfp8Query, plan, query_from_call, quantize_rows


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with nvidia-cutlass-dsl >= 4.6.0."""
    return default_is_supported(device, requires=META.requires)



__all__ = [
    "Mxfp8Config",
    "Mxfp8Query",
    "plan",
    "query_from_call",
    "quantize_rows",
    "is_supported",
]
