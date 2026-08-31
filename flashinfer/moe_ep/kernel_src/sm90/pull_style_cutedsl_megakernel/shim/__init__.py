# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Thin adapters over the SM90 (Hopper) ``src/`` kernel drop.

Mirrors the SM100 package's shim layer
(``kernel_src/cutedsl_megamoe/shim``): all adaptation over the verbatim
``src/`` drop lives here, and the package ``__init__`` re-exports only from
this layer.  ``comm`` holds dist / symmetric-heap / compile helpers;
``hopper_fp8`` and ``hopper_mxfp4`` hold format-specific lazy-compile
frontends plus their symmetric-buffer + fused-launch wrappers.
"""

from __future__ import annotations

# hopper_fp8 imports the raw kernel packages (moe_hopper_fp8, common, ...)
# lazily, but src/ must be on sys.path before any of them resolve.  The
# bootstrap lives here in shim/ (not in src/, which is a verbatim kernel drop)
# and also guards against the sibling SM100 tree owning this process.
from ._paths import bootstrap_paths

bootstrap_paths()

from .comm import (
    bootstrap_dist,
    finalize_dist,
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)
from .autotune import (
    autotune_hopper_fp8_mega_moe,
    autotune_hopper_mxfp4_mega_moe,
    autotune_knobs,
    hopper_fp8_candidates,
)
from .mxfp4_split_autotune import autotune_hopper_mxfp4_split_mega_moe
from .knob_cache import (
    knob_cache_path,
    lookup_knobs,
    record_knobs,
    resolve_knobs,
)
from .tuner import (
    default_knobs,
    is_valid,
    iter_candidates,
    with_knobs,
)
from .mxfp4_tuner import (
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    MXFP4_TUNING_PROVENANCE,
    MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE,
    MXFP4_TUNING_ROUTING_PROFILES,
    MXFP4_TUNING_TOKEN_BUCKETS,
    Mxfp4ExecutionMode,
    Mxfp4RoutingProfile,
    hopper_mxfp4_candidate_records,
    hopper_mxfp4_candidates,
    hopper_mxfp4_candidates_for_shape,
    hopper_mxfp4_default_tactic,
    hopper_mxfp4_ordered_candidates,
    hopper_mxfp4_tuning_manifest,
    hopper_mxfp4_tuning_provenance,
    is_hopper_mxfp4_tactic_shape_compatible,
    is_valid_hopper_mxfp4_tactic,
    normalize_hopper_mxfp4_routing_profile,
    validate_hopper_mxfp4_tactic,
)
from .hopper_fp8 import (
    MegaMoEHopperFp8Config,
    MegaMoEHopperFp8Frontend,
    MegaMoEHopperFp8Inputs,
    MegaMoEHopperFp8SymmBuffer,
    TransformedFp8Weights,
    create_dummy_inputs as create_dummy_hopper_fp8_inputs,
    get_symm_buffer_for_hopper_fp8_mega_moe,
    hopper_fp8_mega_launch_thunk,
    hopper_fp8_mega_moe,
    init_dist,
)
from .hopper_mxfp4 import (
    MegaMoEHopperMxfp4Config,
    MegaMoEHopperMxfp4Frontend,
    MegaMoEHopperMxfp4Inputs,
    MegaMoEHopperMxfp4SymmBuffer,
    TransformedMxfp4Weights,
    get_symm_buffer_for_hopper_mxfp4_mega_moe,
    hopper_mxfp4_mega_launch_thunk,
    hopper_mxfp4_mega_moe,
)
from .hopper_mxfp4_split import (
    MegaMoEHopperMxfp4SplitConfig,
    MegaMoEHopperMxfp4SplitSession,
    MegaMoEHopperMxfp4SplitSymmBuffer,
    Mxfp4SplitError,
    Mxfp4SplitLifecycleError,
    Mxfp4SplitSessionPoisonedError,
    Mxfp4SplitUnavailableError,
    get_symm_buffer_for_hopper_mxfp4_split_mega_moe,
    hopper_mxfp4_split_mega_launch_thunk,
    hopper_mxfp4_split_mega_moe,
)

# These are clean-vendor protocol types. Import them only on attribute access:
# green_context imports the CUDA driver binding and split_mega_runner imports
# the CuTeDSL kernel classes transitively.
_LAZY_SPLIT_VENDOR_EXPORTS = {
    "GreenContextCleanupError": (
        "moe_hopper_fp8.green_context",
        "GreenContextCleanupError",
    ),
    "GreenContextConfigurationError": (
        "moe_hopper_fp8.green_context",
        "GreenContextConfigurationError",
    ),
    "GreenContextError": ("moe_hopper_fp8.green_context", "GreenContextError"),
    "GreenContextPartition": (
        "moe_hopper_fp8.green_context",
        "GreenContextPartition",
    ),
    "GreenContextSplit": ("moe_hopper_fp8.green_context", "GreenContextSplit"),
    "GreenContextSupport": (
        "moe_hopper_fp8.green_context",
        "GreenContextSupport",
    ),
    "GreenContextUnavailableError": (
        "moe_hopper_fp8.green_context",
        "GreenContextUnavailableError",
    ),
    "check_green_context_support": (
        "moe_hopper_fp8.green_context",
        "check_green_context_support",
    ),
    "GreenGraph": ("moe_hopper_fp8.green_graph", "GreenGraph"),
    "GreenGraphCaptureError": (
        "moe_hopper_fp8.green_graph",
        "GreenGraphCaptureError",
    ),
    "GreenGraphCleanupError": (
        "moe_hopper_fp8.green_graph",
        "GreenGraphCleanupError",
    ),
    "GreenGraphError": ("moe_hopper_fp8.green_graph", "GreenGraphError"),
    "GreenGraphTopology": ("moe_hopper_fp8.green_graph", "GreenGraphTopology"),
    "GreenGraphUnavailableError": (
        "moe_hopper_fp8.green_graph",
        "GreenGraphUnavailableError",
    ),
    "SplitEpochResetBarrier": (
        "moe_hopper_fp8.split_epoch_reset",
        "SplitEpochResetBarrier",
    ),
    "SplitK2GlobalJoin": ("moe_hopper_fp8.split_k3_join", "SplitK2GlobalJoin"),
    "SplitMegaCompileRequest": (
        "moe_hopper_fp8.split_mega_runner",
        "SplitMegaCompileRequest",
    ),
    "SplitMegaConfigurationError": (
        "moe_hopper_fp8.split_mega_runner",
        "SplitMegaConfigurationError",
    ),
    "SplitMegaExecutorRequired": (
        "moe_hopper_fp8.split_mega_runner",
        "SplitMegaExecutorRequired",
    ),
    "SplitMegaMxfp4KernelPair": (
        "moe_hopper_fp8.split_mega_runner",
        "SplitMegaMxfp4KernelPair",
    ),
    "SplitMegaPlan": ("moe_hopper_fp8.split_mega_runner", "SplitMegaPlan"),
    "SplitMegaWorkspaceContract": (
        "moe_hopper_fp8.split_mega_runner",
        "SplitMegaWorkspaceContract",
    ),
    "SplitMegaWorkspaceMismatch": (
        "moe_hopper_fp8.split_mega_runner",
        "SplitMegaWorkspaceMismatch",
    ),
    "build_mxfp4_split_kernel_pair": (
        "moe_hopper_fp8.split_mega_runner",
        "build_mxfp4_split_kernel_pair",
    ),
}


def __getattr__(name):  # PEP 562
    target = _LAZY_SPLIT_VENDOR_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = [
    *_LAZY_SPLIT_VENDOR_EXPORTS,
    # paths
    "bootstrap_paths",
    # comm
    "bootstrap_dist",
    "finalize_dist",
    "free_sym_tensor",
    "reset_compiled_mega_workspaces",
    "resolve_gate_up_clamp",
    "sym_zeros",
    # tuner / knob cache / autotune
    "autotune_hopper_fp8_mega_moe",
    "autotune_hopper_mxfp4_mega_moe",
    "autotune_hopper_mxfp4_split_mega_moe",
    "autotune_knobs",
    "default_knobs",
    "hopper_fp8_candidates",
    "is_valid",
    "iter_candidates",
    "knob_cache_path",
    "lookup_knobs",
    "record_knobs",
    "resolve_knobs",
    "with_knobs",
    # MXFP4 offline winners / bounded online candidates
    "MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE",
    "MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE",
    "MXFP4_TUNING_PROVENANCE",
    "MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE",
    "MXFP4_TUNING_ROUTING_PROFILES",
    "MXFP4_TUNING_TOKEN_BUCKETS",
    "Mxfp4ExecutionMode",
    "Mxfp4RoutingProfile",
    "hopper_mxfp4_candidate_records",
    "hopper_mxfp4_candidates",
    "hopper_mxfp4_candidates_for_shape",
    "hopper_mxfp4_default_tactic",
    "hopper_mxfp4_ordered_candidates",
    "hopper_mxfp4_tuning_manifest",
    "hopper_mxfp4_tuning_provenance",
    "is_hopper_mxfp4_tactic_shape_compatible",
    "is_valid_hopper_mxfp4_tactic",
    "normalize_hopper_mxfp4_routing_profile",
    "validate_hopper_mxfp4_tactic",
    # hopper_fp8
    "MegaMoEHopperFp8Config",
    "MegaMoEHopperFp8Frontend",
    "MegaMoEHopperFp8Inputs",
    "MegaMoEHopperFp8SymmBuffer",
    "TransformedFp8Weights",
    "create_dummy_hopper_fp8_inputs",
    "get_symm_buffer_for_hopper_fp8_mega_moe",
    "hopper_fp8_mega_launch_thunk",
    "hopper_fp8_mega_moe",
    "init_dist",
    # hopper_mxfp4
    "MegaMoEHopperMxfp4Config",
    "MegaMoEHopperMxfp4Frontend",
    "MegaMoEHopperMxfp4Inputs",
    "MegaMoEHopperMxfp4SymmBuffer",
    "TransformedMxfp4Weights",
    "get_symm_buffer_for_hopper_mxfp4_mega_moe",
    "hopper_mxfp4_mega_launch_thunk",
    "hopper_mxfp4_mega_moe",
    # hopper_mxfp4 split
    "MegaMoEHopperMxfp4SplitConfig",
    "MegaMoEHopperMxfp4SplitSession",
    "MegaMoEHopperMxfp4SplitSymmBuffer",
    "Mxfp4SplitError",
    "Mxfp4SplitLifecycleError",
    "Mxfp4SplitSessionPoisonedError",
    "Mxfp4SplitUnavailableError",
    "get_symm_buffer_for_hopper_mxfp4_split_mega_moe",
    "hopper_mxfp4_split_mega_launch_thunk",
    "hopper_mxfp4_split_mega_moe",
]
