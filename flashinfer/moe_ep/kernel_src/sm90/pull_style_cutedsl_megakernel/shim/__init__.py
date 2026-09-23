# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Thin adapters over the SM90 (Hopper) ``src/`` kernel drop.

Mirrors the SM100 package's shim layer
(``kernel_src/sm100/cutedsl_megamoe/shim``): all adaptation over the verbatim
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
from .mxfp4_optimization import hopper_mxfp4_candidates
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
    MXFP4_TUNING_ROUTING_PROFILES,
    MXFP4_TUNING_TOKEN_BUCKETS,
    Mxfp4RoutingProfile,
    hopper_mxfp4_default_tactic,
    is_hopper_mxfp4_tactic_shape_compatible,
    is_valid_hopper_mxfp4_tactic,
    normalize_hopper_mxfp4_routing_profile,
    require_hopper_mxfp4_fused_tuning_device,
    validate_hopper_mxfp4_tactic,
)
from .hopper_fp8 import (
    MegaMoEHopperFp8Config,
    MegaMoEHopperFp8Frontend,
    MegaMoEHopperFp8Inputs,
    MegaMoEHopperFp8SymmBuffer,
    TransformedFp8Weights,
    _get_symm_buffer_for_hopper_fp8_mega_moe_from_resolved_config,
    create_dummy_inputs as create_dummy_hopper_fp8_inputs,
    get_symm_buffer_for_hopper_fp8_mega_moe,
    hopper_fp8_mega_launch_thunk,
    hopper_fp8_mega_moe,
    init_dist,
    resolve_hopper_fp8_mega_moe_config,
)
from .hopper_mxfp4 import (
    MegaMoEHopperMxfp4Config,
    MegaMoEHopperMxfp4Frontend,
    MegaMoEHopperMxfp4Inputs,
    MegaMoEHopperMxfp4SymmBuffer,
    TransformedMxfp4Weights,
    _get_symm_buffer_for_hopper_mxfp4_mega_moe_from_resolved_config,
    _resolve_hopper_mxfp4_mega_moe_config,
    get_symm_buffer_for_hopper_mxfp4_mega_moe,
    hopper_mxfp4_mega_launch_thunk,
    hopper_mxfp4_mega_moe,
    resolve_hopper_mxfp4_knobs,
)

__all__ = [
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
    "MXFP4_TUNING_ROUTING_PROFILES",
    "MXFP4_TUNING_TOKEN_BUCKETS",
    "Mxfp4RoutingProfile",
    "hopper_mxfp4_candidates",
    "hopper_mxfp4_default_tactic",
    "is_hopper_mxfp4_tactic_shape_compatible",
    "is_valid_hopper_mxfp4_tactic",
    "normalize_hopper_mxfp4_routing_profile",
    "require_hopper_mxfp4_fused_tuning_device",
    "validate_hopper_mxfp4_tactic",
    # hopper_fp8
    "MegaMoEHopperFp8Config",
    "MegaMoEHopperFp8Frontend",
    "MegaMoEHopperFp8Inputs",
    "MegaMoEHopperFp8SymmBuffer",
    "TransformedFp8Weights",
    "_get_symm_buffer_for_hopper_fp8_mega_moe_from_resolved_config",
    "create_dummy_hopper_fp8_inputs",
    "get_symm_buffer_for_hopper_fp8_mega_moe",
    "hopper_fp8_mega_launch_thunk",
    "hopper_fp8_mega_moe",
    "init_dist",
    "resolve_hopper_fp8_mega_moe_config",
    # hopper_mxfp4
    "MegaMoEHopperMxfp4Config",
    "MegaMoEHopperMxfp4Frontend",
    "MegaMoEHopperMxfp4Inputs",
    "MegaMoEHopperMxfp4SymmBuffer",
    "TransformedMxfp4Weights",
    "_get_symm_buffer_for_hopper_mxfp4_mega_moe_from_resolved_config",
    "_resolve_hopper_mxfp4_mega_moe_config",
    "get_symm_buffer_for_hopper_mxfp4_mega_moe",
    "hopper_mxfp4_mega_launch_thunk",
    "hopper_mxfp4_mega_moe",
    "resolve_hopper_mxfp4_knobs",
]
