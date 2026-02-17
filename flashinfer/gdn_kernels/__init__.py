"""
GDN (Gated Delta Rule) Kernels - CuTe DSL Implementations
=========================================================

This module provides CuTe-DSL implementations of GDN kernels.

The main gdn_decode.py and gdn_prefill.py files at the top level contain reference
implementations and JIT-compiled kernels. This submodule provides high-performance
CuTe DSL variants optimized for specific use cases.

Exported Kernels:
- gated_delta_rule: BF16 hidden state decode kernel (T=1,2,3,4)
- GatedDeltaRuleKernel: Kernel class for advanced usage
"""

from typing import Optional, Type

try:
    from .gdn_decode_bf16_state import (
        gated_delta_rule,
        GatedDeltaRuleKernel,
    )

    _has_cute_dsl = True
except ImportError:
    _has_cute_dsl = False
    gated_delta_rule = None  # type: ignore
    GatedDeltaRuleKernel: Optional[Type] = None  # type: ignore

__all__ = [
    "gated_delta_rule",
    "GatedDeltaRuleKernel",
]
