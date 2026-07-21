"""
KDA Blackwell (SM100) Kernels
=============================

CuTe-DSL chunked prefill kernel for Kimi Delta Attention on Blackwell (SM100)
GPUs.
"""

try:
    from .kda_prefill import chunk_kda_sm100
except (ImportError, RuntimeError):
    chunk_kda_sm100 = None  # type: ignore

__all__ = [
    "chunk_kda_sm100",
]
