"""cute SM120 MXFP8 groupwise backend for ``flashinfer.grouped_mm``.

The parent :mod:`flashinfer.grouped_mm` package exposes a backend-agnostic
public API; backend-specific implementations live in sibling subpackages
(this one is for the cute SM120 MXFP8 groupwise kernel). Public entries are
re-exported below; the dispatcher in :mod:`..core` imports the
``_check_*`` helpers and the ``get_gemm_sm120_module_cute_mxfp8`` accessor.
"""

from .core import (
    _check_m_indptr,
    _check_scale_granularity_mnk,
    _check_scale_major_mode_mxfp8,
    get_gemm_sm120_module_cute_mxfp8,
    moe_gemm_mxfp8_nt_groupwise,
)

__all__ = [
    "_check_m_indptr",
    "_check_scale_granularity_mnk",
    "_check_scale_major_mode_mxfp8",
    "get_gemm_sm120_module_cute_mxfp8",
    "moe_gemm_mxfp8_nt_groupwise",
]
