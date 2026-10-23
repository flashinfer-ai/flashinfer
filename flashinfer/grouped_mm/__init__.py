"""Grouped Matrix Multiplication (MoE Grouped GEMM).

Provides grouped variants of the dense ``mm_*`` GEMM APIs, where each
expert in a Mixture-of-Experts layer has its own weight matrix and tokens
are routed to experts via ``m_indptr``.

The public API is backend-agnostic and lives in :mod:`.core`.  Each
backend has its own subpackage (e.g. :mod:`.cudnn`); future backends
should follow the same convention.  The experimental Cake entry points
(``grouped_gemm_fwd``, ``grouped_gemm_dgrad``, ``grouped_gemm_wgrad``,
``cake_grouped_mm``) live in :mod:`.cake` and hand off to
:mod:`flashinfer.experimental.cake_moe_grouped_gemm`.
"""

from .cake import cake_grouped_mm as cake_grouped_mm
from .cake import grouped_gemm_dgrad as grouped_gemm_dgrad
from .cake import grouped_gemm_fwd as grouped_gemm_fwd
from .cake import grouped_gemm_wgrad as grouped_gemm_wgrad
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
    "cake_grouped_mm",
    "grouped_gemm_dgrad",
    "grouped_gemm_fwd",
    "grouped_gemm_wgrad",
    "grouped_mm_bf16",
    "grouped_mm_fp4",
    "grouped_mm_fp8",
    "grouped_mm_mxfp8",
    "moe_gemm_fp8_nt_groupwise",
    "moe_gemm_mxfp8_nt_groupwise",
]
