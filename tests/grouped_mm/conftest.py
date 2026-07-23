"""Shared fixtures and helpers for grouped_mm tests."""

import pytest
import torch

from flashinfer.grouped_mm.core import (
    _check_grouped_mm_bf16,
    _check_grouped_mm_fp4,
    _check_grouped_mm_fp8,
    _check_grouped_mm_mxfp8,
)
from flashinfer.grouped_mm.cudnn import _CUDNN_MOE_MIN_VERSION
from flashinfer.utils import get_compute_capability

try:
    import cudnn

    CUDNN_AVAILABLE = True
    CUDNN_BACKEND_VERSION = cudnn.backend_version()
    # `moe_grouped_matmul_mode` lives on the cuDNN Python frontend package, whose
    # version may lag behind the backend lib. Probing the attribute directly is
    # the only reliable signal that this binding actually exposes the MOE API.
    CUDNN_HAS_MOE_API = hasattr(cudnn, "moe_grouped_matmul_mode")
except (ImportError, OSError):
    CUDNN_AVAILABLE = False
    CUDNN_BACKEND_VERSION = 0
    CUDNN_HAS_MOE_API = False

# Both marks gate on the same runtime constant the cuDNN backend itself enforces
# (flashinfer.grouped_mm.cudnn._CUDNN_MOE_MIN_VERSION) instead of a hardcoded
# literal, so a future bump of that constant can't silently desync the test skip
# from the runtime check again (#4064: a prior hardcoded 91800 here diverged from
# the constant after #3797 raised it to 92100, so grouped_mm_bf16/fp8 tests ran
# instead of skipping on cuDNN 9.19-9.20 and hit the runtime RuntimeError).
requires_cudnn_moe = pytest.mark.skipif(
    not CUDNN_AVAILABLE
    or CUDNN_BACKEND_VERSION < _CUDNN_MOE_MIN_VERSION
    or not CUDNN_HAS_MOE_API,
    reason=f"cuDNN MOE requires backend >= {_CUDNN_MOE_MIN_VERSION} and a frontend exposing moe_grouped_matmul_mode",
)

requires_cudnn_moe_block_scale = pytest.mark.skipif(
    not CUDNN_AVAILABLE
    or CUDNN_BACKEND_VERSION < _CUDNN_MOE_MIN_VERSION
    or not CUDNN_HAS_MOE_API,
    reason=f"cuDNN MOE block-scale requires backend >= {_CUDNN_MOE_MIN_VERSION} and a frontend exposing moe_grouped_matmul_mode",
)


def _requires_supported_cc(check_fn):
    # Skip mark whose allowlist is the `_supported_ccs` set attached to
    # `check_fn` by `@supported_compute_capability`. Keeping the test
    # allowlist and the function allowlist in lockstep guarantees they
    # never drift apart.
    supported = sorted(check_fn._supported_ccs)
    if not torch.cuda.is_available():
        return pytest.mark.skipif(True, reason="CUDA is required")
    major, minor = get_compute_capability(torch.device("cuda"))
    current_cc = major * 10 + minor
    return pytest.mark.skipif(
        current_cc not in check_fn._supported_ccs,
        reason=(f"{check_fn.__name__} supports sm{supported}, got sm{current_cc}"),
    )


requires_grouped_mm_bf16_cc = _requires_supported_cc(_check_grouped_mm_bf16)
requires_grouped_mm_fp8_cc = _requires_supported_cc(_check_grouped_mm_fp8)
requires_grouped_mm_mxfp8_cc = _requires_supported_cc(_check_grouped_mm_mxfp8)
requires_grouped_mm_fp4_cc = _requires_supported_cc(_check_grouped_mm_fp4)


def ref_grouped_mm(a, b, m_indptr, out_dtype, alpha=None):
    """Loop over experts, matmul with transposed weight (NT layout)."""
    num_experts = b.shape[0]
    n = b.shape[1]
    cum_m = a.shape[0]
    out = torch.zeros(cum_m, n, dtype=torch.float32, device=a.device)
    for e in range(num_experts):
        start = m_indptr[e].item()
        end = m_indptr[e + 1].item()
        if start < end:
            out[start:end] = a[start:end].float() @ b[e].float().T
    if alpha is not None:
        out = out * alpha.float()
    return out.to(out_dtype)
