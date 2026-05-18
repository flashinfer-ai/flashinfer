"""
GDN Blackwell (SM100) Kernels
=============================

CuTe-DSL chunked prefill kernel for Gated Delta Net on Blackwell GPUs.
"""

try:
    from .gdn_prefill import chunk_gated_delta_rule_sm100

    _has_blackwell_prefill = True
except (ImportError, RuntimeError):
    _has_blackwell_prefill = False
    chunk_gated_delta_rule_sm100 = None  # type: ignore

__all__ = [
    "chunk_gated_delta_rule_sm100",
    "_has_blackwell_prefill",
]
