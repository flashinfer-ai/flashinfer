# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM90 (Hopper) pull-style CuTeDSL mega-kernel package for moe_ep.

Public API for the ``moe_ep`` backends over the SM90 FP8 and Humming MXFP4
mega kernels (``Sm90MegaMoEFp8Kernel``, ``Sm90MegaMoESwapABFp8Kernel``, and
``Sm90MegaMoESwapABMxfp4Fp8Kernel``).  This tree is a
FORK of the SM100 package (``kernel_src/cutedsl_megamoe``): the kernel
team's SM90 work branched from the same repo, so ``src/`` duplicates the
shared runtime (``common``, ``src``, ``moe_nvfp4_swapab``) at the SM90 drop's
revision.  The two trees are separate backends and are mutually exclusive per
process (see ``shim/_paths.py``).

Layering (same rules as the SM100 package — see SKILL.md):
- ``src/`` is a verbatim kernel-team drop; never edit it.
- ``shim/`` is the only layer that imports the raw ``src/`` packages.
- moe_ep backends import from this ``__init__`` only.

Usage::

    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        get_symm_buffer_for_hopper_fp8_mega_moe,
        hopper_fp8_mega_moe,
        init_dist,
    )

The shim modules keep their cutlass / cuda imports lazy, so importing this
package stays CPU-safe (no GPU, no compiled kernels).
"""

from __future__ import annotations

# Importing the shim puts this tree's ``src/`` on sys.path (via
# shim/_paths.bootstrap_paths) before its modules resolve the raw kernel
# packages (moe_hopper_fp8, common, ...).  ``bootstrap_paths`` is re-exported
# here so callers (e.g. core runtime) reach it through this public boundary.
from .shim import (
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    MXFP4_TUNING_PROVENANCE,
    MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE,
    MXFP4_TUNING_ROUTING_PROFILES,
    MXFP4_TUNING_TOKEN_BUCKETS,
    MegaMoEHopperFp8Config,
    Mxfp4ExecutionMode,
    Mxfp4RoutingProfile,
    autotune_hopper_fp8_mega_moe,
    autotune_hopper_mxfp4_mega_moe,
    autotune_hopper_mxfp4_split_mega_moe,
    autotune_knobs,
    default_knobs,
    hopper_fp8_candidates,
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
    is_valid,
    iter_candidates,
    knob_cache_path,
    lookup_knobs,
    record_knobs,
    resolve_knobs,
    with_knobs,
    MegaMoEHopperFp8Frontend,
    MegaMoEHopperFp8Inputs,
    MegaMoEHopperFp8SymmBuffer,
    MegaMoEHopperMxfp4Config,
    MegaMoEHopperMxfp4Frontend,
    MegaMoEHopperMxfp4Inputs,
    MegaMoEHopperMxfp4SymmBuffer,
    MegaMoEHopperMxfp4SplitConfig,
    MegaMoEHopperMxfp4SplitSession,
    MegaMoEHopperMxfp4SplitSymmBuffer,
    Mxfp4SplitError,
    Mxfp4SplitLifecycleError,
    Mxfp4SplitSessionPoisonedError,
    Mxfp4SplitUnavailableError,
    TransformedFp8Weights,
    TransformedMxfp4Weights,
    bootstrap_paths,
    create_dummy_hopper_fp8_inputs,
    finalize_dist,
    get_symm_buffer_for_hopper_fp8_mega_moe,
    get_symm_buffer_for_hopper_mxfp4_mega_moe,
    get_symm_buffer_for_hopper_mxfp4_split_mega_moe,
    hopper_fp8_mega_launch_thunk,
    hopper_fp8_mega_moe,
    hopper_mxfp4_mega_launch_thunk,
    hopper_mxfp4_mega_moe,
    hopper_mxfp4_split_mega_launch_thunk,
    hopper_mxfp4_split_mega_moe,
    init_dist,
)

# Backward-compatible name with the SM100 package's convention:
# ``create_dummy_inputs`` retains the pre-MXFP4 convention and means Hopper
# ordinary FP8; MXFP4 callers use the explicitly named fused API.
create_dummy_inputs = create_dummy_hopper_fp8_inputs

# Raw-kernel helpers/constants/reference (FP8 quant helpers, block constants,
# the FP8 torch reference, ...) pull ``cutlass`` transitively.  Expose them
# lazily so ``import ...pull_style_cutedsl_megakernel`` stays CPU-safe; the FI
# backend + verification tests still reach them only through this boundary
# (the access happens inside their functions).  See ``shim/kernel_helpers.py``.
_LAZY_HELPERS = (
    "Fp8BlockScaleK",
    "Fp8E8M0SfVecSize",
    "Fp8Fc2ActivationScaleK",
    "Fp8GateUpInterleave",
    "Fp8WeightScaleBlockK",
    "Fp8WeightScaleBlockN",
    "_stack_byte_reinterpretable_tensors",
    "ceil_div",
    "compute_megamoe_reference_fp8",
    "convert_mxfp4_pair_preprocessed_signs",
    "convert_packed_a_kblock",
    "convert_packed_a_kblock_from_offset",
    "create_fp8_tensor",
    "fp8_dtype_max",
    "kind_data_dtype",
    "make_constant_block_scale",
    "make_expanded_offset_view",
    "make_fp8_per_tensor_dequant_scale",
    "make_offset_smem_layout",
    "make_packed_a_ldsm_views",
    "quantize_fp8_per_token_block",
    "quantize_fp8_weight_block_nk",
    "round_up",
    "to_blocked",
)


_LAZY_SPLIT_VENDOR = (
    "GreenContextCleanupError",
    "GreenContextConfigurationError",
    "GreenContextError",
    "GreenContextPartition",
    "GreenContextSplit",
    "GreenContextSupport",
    "GreenContextUnavailableError",
    "check_green_context_support",
    "GreenGraph",
    "GreenGraphCaptureError",
    "GreenGraphCleanupError",
    "GreenGraphError",
    "GreenGraphTopology",
    "GreenGraphUnavailableError",
    "SplitEpochResetBarrier",
    "SplitK2GlobalJoin",
    "SplitMegaCompileRequest",
    "SplitMegaConfigurationError",
    "SplitMegaExecutorRequired",
    "SplitMegaMxfp4KernelPair",
    "SplitMegaPlan",
    "SplitMegaWorkspaceContract",
    "SplitMegaWorkspaceMismatch",
    "build_mxfp4_split_kernel_pair",
)


def __getattr__(name):  # PEP 562
    if name in _LAZY_HELPERS:
        from .shim import kernel_helpers

        return getattr(kernel_helpers, name)
    if name in _LAZY_SPLIT_VENDOR:
        from . import shim

        value = getattr(shim, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    *_LAZY_HELPERS,
    *_LAZY_SPLIT_VENDOR,
    "MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE",
    "MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE",
    "MXFP4_TUNING_PROVENANCE",
    "MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE",
    "MXFP4_TUNING_ROUTING_PROFILES",
    "MXFP4_TUNING_TOKEN_BUCKETS",
    "MegaMoEHopperFp8Config",
    "Mxfp4ExecutionMode",
    "Mxfp4RoutingProfile",
    "autotune_hopper_fp8_mega_moe",
    "autotune_hopper_mxfp4_mega_moe",
    "autotune_hopper_mxfp4_split_mega_moe",
    "autotune_knobs",
    "default_knobs",
    "hopper_fp8_candidates",
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
    "is_valid",
    "iter_candidates",
    "knob_cache_path",
    "lookup_knobs",
    "record_knobs",
    "resolve_knobs",
    "with_knobs",
    "MegaMoEHopperFp8Frontend",
    "MegaMoEHopperFp8Inputs",
    "MegaMoEHopperFp8SymmBuffer",
    "MegaMoEHopperMxfp4Config",
    "MegaMoEHopperMxfp4Frontend",
    "MegaMoEHopperMxfp4Inputs",
    "MegaMoEHopperMxfp4SymmBuffer",
    "MegaMoEHopperMxfp4SplitConfig",
    "MegaMoEHopperMxfp4SplitSession",
    "MegaMoEHopperMxfp4SplitSymmBuffer",
    "Mxfp4SplitError",
    "Mxfp4SplitLifecycleError",
    "Mxfp4SplitSessionPoisonedError",
    "Mxfp4SplitUnavailableError",
    "TransformedFp8Weights",
    "TransformedMxfp4Weights",
    "bootstrap_paths",
    "create_dummy_hopper_fp8_inputs",
    "create_dummy_inputs",
    "finalize_dist",
    "get_symm_buffer_for_hopper_fp8_mega_moe",
    "get_symm_buffer_for_hopper_mxfp4_mega_moe",
    "get_symm_buffer_for_hopper_mxfp4_split_mega_moe",
    "hopper_fp8_mega_launch_thunk",
    "hopper_fp8_mega_moe",
    "hopper_mxfp4_mega_launch_thunk",
    "hopper_mxfp4_mega_moe",
    "hopper_mxfp4_split_mega_launch_thunk",
    "hopper_mxfp4_split_mega_moe",
    "init_dist",
]
