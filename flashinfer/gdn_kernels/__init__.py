"""
GDN (Gated Delta Rule) Kernels - CuTe DSL Implementations
=========================================================

This module provides CuTe-DSL implementations of GDN kernels.

The main gdn_decode.py and gdn_prefill.py files at the top level contain reference
implementations and JIT-compiled kernels. This submodule provides high-performance
CuTe DSL variants optimized for specific use cases.

Exported Kernels:
- gated_delta_rule: BF16 hidden state decode kernel (T=1)
- gated_delta_rule_mtp: BF16 hidden state MTP kernel (T>=1)
"""

try:
    from .gdn_decode_bf16_state import (
        gated_delta_rule,
        gated_delta_rule_mtp,
        gated_delta_rule_bf16state_cooprow,  # backward compat alias
        gated_delta_rule_bf16state_cooprow_mtp,  # backward compat alias
    )

    _has_cute_dsl = True
except ImportError:
    _has_cute_dsl = False
    gated_delta_rule = None  # type: ignore
    gated_delta_rule_mtp = None  # type: ignore
    gated_delta_rule_bf16state_cooprow = None  # type: ignore
    gated_delta_rule_bf16state_cooprow_mtp = None  # type: ignore

__all__ = [
    "gated_delta_rule",
    "gated_delta_rule_mtp",
    "gated_delta_rule_bf16state_cooprow",
    "gated_delta_rule_bf16state_cooprow_mtp",
]
