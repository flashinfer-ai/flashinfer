"""
GDN Blackwell (SM100) Kernels
=============================

CuTe-DSL chunked prefill kernel for Gated Delta Net on Blackwell (SM100) GPUs.
"""

try:
    from .gdn_prefill import chunk_gated_delta_rule_sm100
except (ImportError, RuntimeError):
    chunk_gated_delta_rule_sm100 = None  # type: ignore

__all__ = [
    "chunk_gated_delta_rule_sm100",
]
