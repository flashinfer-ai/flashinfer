"""
GDN (Gated Delta Rule) Kernels - CuTe DSL Implementations
=========================================================

This module provides CuTe-DSL implementations of GDN kernels.

The main gdn_decode.py file at the top level contains the API layer. This submodule
provides high-performance CuTe DSL kernel implementations for specific use cases.

Exported Kernels:
- gated_delta_rule: BF16 hidden state decode kernel (T=1,2,3,4)
- GatedDeltaRuleKernel: Kernel class for advanced usage
- run_pretranspose_decode: Pretranspose (V-major) decode kernel
- run_nontranspose_decode: Nontranspose (K-major) decode kernel
- run_mtp_decode: Multi-token processing decode kernel
- get_tile_v_mtp, get_vec_size_mtp: MTP hyperparameter helpers
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

try:
    from .gdn_decode_pretranspose import run_pretranspose_decode
except ImportError:
    run_pretranspose_decode = None  # type: ignore

try:
    from .gdn_decode_nontranspose import run_nontranspose_decode
except ImportError:
    run_nontranspose_decode = None  # type: ignore

try:
    from .gdn_decode_mtp import run_mtp_decode, get_tile_v_mtp, get_vec_size_mtp
except ImportError:
    run_mtp_decode = None  # type: ignore
    get_tile_v_mtp = None  # type: ignore
    get_vec_size_mtp = None  # type: ignore

try:
    from .cutile_gdn_prefill import chunk_gated_delta_rule_cutile
except ImportError:
    chunk_gated_delta_rule_cutile = None  # type: ignore

__all__ = [
    "gated_delta_rule",
    "GatedDeltaRuleKernel",
    "run_pretranspose_decode",
    "run_nontranspose_decode",
    "run_mtp_decode",
    "get_tile_v_mtp",
    "get_vec_size_mtp",
    "chunk_gated_delta_rule_cutile",
]
