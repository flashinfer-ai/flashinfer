"""Shared fixtures and helpers for grouped_mm tests."""

import pytest
import torch

from flashinfer.utils import get_compute_capability

try:
    import cudnn

    CUDNN_AVAILABLE = True
    CUDNN_BACKEND_VERSION = cudnn.backend_version()
except (ImportError, OSError):
    CUDNN_AVAILABLE = False
    CUDNN_BACKEND_VERSION = 0

requires_cudnn_moe = pytest.mark.skipif(
    not CUDNN_AVAILABLE or CUDNN_BACKEND_VERSION < 91800,
    reason="cuDNN MOE requires backend >= 9.18.0",
)

requires_cudnn_moe_block_scale = pytest.mark.skipif(
    not CUDNN_AVAILABLE or CUDNN_BACKEND_VERSION < 92100,
    reason="cuDNN MOE block-scale requires backend >= 9.21.0",
)

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda"))[0] < 10,
    reason="Block-scale grouped GEMM requires SM100+",
)


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
