"""cuDNN backend for ``flashinfer.grouped_mm``.

The parent :mod:`flashinfer.grouped_mm` package exposes a backend-agnostic
public API; backend-specific implementations live in sibling subpackages
(this one is for cuDNN).  The dispatcher in :mod:`..core` imports the
``_run_*`` runners and the ``_check_cudnn_version`` / version-constant
helpers below.
"""

from .core import (
    _CUDNN_MOE_BLOCK_SCALE_MIN_VERSION,
    _CUDNN_MOE_MIN_VERSION,
    _check_cudnn_version,
    _run_cudnn_moe_block_scale_grouped_gemm_fp4,
    _run_cudnn_moe_block_scale_grouped_gemm_mxfp8,
    _run_cudnn_moe_grouped_gemm,
)

__all__ = [
    "_CUDNN_MOE_BLOCK_SCALE_MIN_VERSION",
    "_CUDNN_MOE_MIN_VERSION",
    "_check_cudnn_version",
    "_run_cudnn_moe_block_scale_grouped_gemm_fp4",
    "_run_cudnn_moe_block_scale_grouped_gemm_mxfp8",
    "_run_cudnn_moe_grouped_gemm",
]
