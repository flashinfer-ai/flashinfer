"""Grouped Matrix Multiplication (MoE Grouped GEMM).

Provides grouped variants of the dense ``mm_*`` GEMM APIs, where each
expert in a Mixture-of-Experts layer has its own weight matrix and tokens
are routed to experts via ``m_indptr``.

The public API is backend-agnostic and lives in :mod:`.core`.  Each
backend has its own subpackage (e.g. :mod:`.cudnn`); future backends
should follow the same convention.  The experimental Cake entry points
(``grouped_det_gemm_fwd``, ``grouped_det_gemm_dgrad``, ``grouped_det_gemm_wgrad``,
``cake_grouped_det_mm``) live in :mod:`.cake` and hand off to
:mod:`flashinfer.experimental.cake_moe_grouped_det_gemm`.
"""

from typing import Any as _Any

from .cake import cake_grouped_det_mm as cake_grouped_det_mm
from .cake import grouped_det_gemm_dgrad as grouped_det_gemm_dgrad
from .cake import grouped_det_gemm_fwd as grouped_det_gemm_fwd
from .cake import grouped_det_gemm_wgrad as grouped_det_gemm_wgrad
from .core import grouped_mm_bf16 as grouped_mm_bf16
from .core import grouped_mm_fp4 as grouped_mm_fp4
from .core import grouped_mm_fp8 as grouped_mm_fp8
from .core import grouped_mm_mxfp8 as grouped_mm_mxfp8
from .cute_sm120_fp8_groupwise import (
    moe_gemm_fp8_nt_groupwise as moe_gemm_fp8_nt_groupwise,
)
from .cute_sm120_mxfp8_groupwise import (
    moe_gemm_mxfp8_nt_groupwise as moe_gemm_mxfp8_nt_groupwise,
)

__all__ = [
    "cake_grouped_det_mm",
    "grouped_det_gemm_dgrad",
    "grouped_det_gemm_fwd",
    "grouped_det_gemm_wgrad",
    "grouped_mm_bf16",
    "grouped_mm_fp4",
    "grouped_mm_fp8",
    "grouped_mm_mxfp8",
    "moe_gemm_fp8_nt_groupwise",
    "moe_gemm_mxfp8_nt_groupwise",
]


def __getattr__(name: str) -> _Any:
    # The Cake entry-point names before the deterministic-family rename
    # (``grouped_gemm_fwd``, ``grouped_gemm_dgrad``, ``grouped_gemm_wgrad``,
    # ``cake_grouped_mm``): deprecated compatibility aliases of the ``*_det_*``
    # functions, resolved by :mod:`.cake`.
    from . import cake

    if name in cake._RENAMED:
        return getattr(cake, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
