"""
GDN (Gated Delta Rule) Kernels - CuTe DSL Implementations
=========================================================

This module provides CuTe-DSL implementations of GDN kernels.

The main gdn_decode.py file at the top level contains the API layer. This submodule
provides high-performance CuTe DSL kernel implementations for specific use cases.

Exported Kernels:
- gated_delta_rule: BF16 hidden state decode kernel (T=1)
- gated_delta_rule_mtp: BF16 hidden state MTP kernel (T>=1)
- gated_delta_rule_bf16state_cooprow: backward compat alias for gated_delta_rule
- gated_delta_rule_bf16state_cooprow_mtp: backward compat alias for gated_delta_rule_mtp
- run_pretranspose_decode: Pretranspose (V-major) decode kernel
- run_nontranspose_decode: Nontranspose (K-major) decode kernel
- run_mtp_decode: Multi-token processing decode kernel
- get_mtp_config, get_tile_v_mtp, get_vec_size_mtp: MTP hyperparameter helpers
- gated_delta_rule_mtp_ucache: u/d-cache spec-decode verify kernel (ring append,
  read-only state; legacy 16-deep flat ring)
- gated_delta_rule_mtp_ucache_flush: fused verify+flush ucache kernel (32-slot
  rotating ring; folds the window into the checkpoint state)
"""

try:
    from .gdn_decode_bf16_state import (
        gated_delta_rule,
        gated_delta_rule_mtp,
        gated_delta_rule_bf16state_cooprow,  # backward compat alias
        gated_delta_rule_bf16state_cooprow_mtp,  # backward compat alias
    )

    _has_cute_dsl = True
except (ImportError, RuntimeError):
    _has_cute_dsl = False
    gated_delta_rule = None  # type: ignore
    gated_delta_rule_mtp = None  # type: ignore
    gated_delta_rule_bf16state_cooprow = None  # type: ignore
    gated_delta_rule_bf16state_cooprow_mtp = None  # type: ignore

# Bank-conflict-eliminated, no-prepack, OUTPUT-ONLY BF16-state MTP decode (v18).
# Alternative implementation of the bf16_state MTP kernel for the output-only
# (frozen-state) case; consumes H0 in its natural (pool, HV, V, K) bf16 layout.
try:
    from .gdn_decode_bf16_wy_output_only import (
        gated_delta_rule_mtp as gated_delta_rule_mtp_wy_output_only,
    )

    _GDN_DECODE_BF16_WY_OUTPUT_ONLY_AVAILABLE = True
except (ImportError, RuntimeError):
    _GDN_DECODE_BF16_WY_OUTPUT_ONLY_AVAILABLE = False
    gated_delta_rule_mtp_wy_output_only = None  # type: ignore

# ReplaySSM u/d-cache spec-decode kernels. NOTE: the two ring formats are NOT
# compatible — verify-only uses the legacy 16-deep flat ring (no cache_base),
# the flush kernel a 32-slot rotating ring. Storage dtypes for the flush kernel
# are chosen at import time via GDN_UCACHE_{IO,STATE,RING}_DTYPE.
try:
    from .gdn_decode_bf16_wy_ucache import gated_delta_rule_mtp_ucache
except (ImportError, RuntimeError):
    gated_delta_rule_mtp_ucache = None  # type: ignore

try:
    from .gdn_decode_bf16_wy_ucache_flush import gated_delta_rule_mtp_ucache_flush
except (ImportError, RuntimeError):
    gated_delta_rule_mtp_ucache_flush = None  # type: ignore

try:
    from .gdn_decode_pretranspose import run_pretranspose_decode
except (ImportError, RuntimeError):
    run_pretranspose_decode = None  # type: ignore

try:
    from .gdn_decode_nontranspose import run_nontranspose_decode
except (ImportError, RuntimeError):
    run_nontranspose_decode = None  # type: ignore

try:
    from .gdn_decode_mtp import (
        run_mtp_decode,
        get_tile_v_mtp,
        get_vec_size_mtp,
        get_mtp_config,
    )
except (ImportError, RuntimeError):
    run_mtp_decode = None  # type: ignore
    get_tile_v_mtp = None  # type: ignore
    get_vec_size_mtp = None  # type: ignore
    get_mtp_config = None  # type: ignore

try:
    from .blackwell import (
        _gdn_cp_nvcc_version,
        _chunk_gated_delta_rule_source_sm100,
        chunk_gated_delta_rule_sm100,
        cp_delta_rule_dsl_sm100,
    )
except (ImportError, RuntimeError):
    _gdn_cp_nvcc_version = None  # type: ignore
    _chunk_gated_delta_rule_source_sm100 = None  # type: ignore
    chunk_gated_delta_rule_sm100 = None  # type: ignore
    cp_delta_rule_dsl_sm100 = None  # type: ignore

try:
    from .delta_rule_dsl import (
        chunk_gated_delta_rule_sm90,
        chunk_gated_delta_rule_sm120,
        cp_delta_rule_dsl_sm90,
        cp_delta_rule_dsl_sm120,
    )
except (ImportError, RuntimeError):
    chunk_gated_delta_rule_sm90 = None  # type: ignore
    chunk_gated_delta_rule_sm120 = None  # type: ignore
    cp_delta_rule_dsl_sm90 = None  # type: ignore
    cp_delta_rule_dsl_sm120 = None  # type: ignore

__all__ = [
    "gated_delta_rule",
    "gated_delta_rule_mtp",
    "gated_delta_rule_mtp_ucache",
    "gated_delta_rule_mtp_ucache_flush",
    "gated_delta_rule_mtp_wy_output_only",
    "gated_delta_rule_bf16state_cooprow",
    "gated_delta_rule_bf16state_cooprow_mtp",
    "run_pretranspose_decode",
    "run_nontranspose_decode",
    "run_mtp_decode",
    "get_tile_v_mtp",
    "get_vec_size_mtp",
    "get_mtp_config",
    "chunk_gated_delta_rule_sm90",
    "chunk_gated_delta_rule_sm100",
    "chunk_gated_delta_rule_sm120",
    "cp_delta_rule_dsl_sm90",
    "cp_delta_rule_dsl_sm100",
    "cp_delta_rule_dsl_sm120",
]
