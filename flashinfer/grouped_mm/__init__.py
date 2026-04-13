"""cuDNN-backed Grouped Matrix Multiplication (MoE Grouped GEMM).

Provides grouped variants of the dense ``mm_*`` GEMM APIs, where each
expert in a Mixture-of-Experts layer has its own weight matrix and tokens
are routed to experts via ``m_indptr``.

All implementations use cuDNN's ``moe_grouped_matmul`` as the backend.
"""

from .core import grouped_mm_bf16 as grouped_mm_bf16

__all__ = [
    "grouped_mm_bf16",
]
