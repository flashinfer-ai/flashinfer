"""Shared fixtures and helpers for grouped_mm tests."""

import pytest
import torch

from flashinfer.grouped_mm.core import (
    _check_grouped_mm_bf16,
    _check_grouped_mm_fp4,
    _check_grouped_mm_fp8,
    _check_grouped_mm_mxfp8,
)
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

requires_cudnn_moe = pytest.mark.skipif(
    not CUDNN_AVAILABLE or CUDNN_BACKEND_VERSION < 91800 or not CUDNN_HAS_MOE_API,
    reason="cuDNN MOE requires backend >= 9.18.0 and a frontend exposing moe_grouped_matmul_mode",
)

requires_cudnn_moe_block_scale = pytest.mark.skipif(
    not CUDNN_AVAILABLE or CUDNN_BACKEND_VERSION < 92100 or not CUDNN_HAS_MOE_API,
    reason="cuDNN MOE block-scale requires backend >= 9.21.0 and a frontend exposing moe_grouped_matmul_mode",
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
