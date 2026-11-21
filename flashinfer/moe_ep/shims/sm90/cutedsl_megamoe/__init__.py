# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Hopper pull-style MegaMoE adapters. Compiler and device imports stay lazy."""

from __future__ import annotations

from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe._paths import bootstrap_paths

bootstrap_paths()

from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.comm import (
    bootstrap_dist,
    finalize_dist,
    free_sym_tensor,
    reset_compiled_mega_workspaces,
    resolve_gate_up_clamp,
    sym_zeros,
)
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.autotune import (
    autotune_hopper_fp8_mega_moe,
    autotune_hopper_mxfp4_mega_moe,
    autotune_knobs,
    hopper_fp8_candidates,
)
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.mxfp4_optimization import (
    hopper_mxfp4_candidates,
)
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.nvfp4_autotune import (
    autotune_hopper_nvfp4_mega_moe,
)
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.nvfp4_tuner import (
    hopper_nvfp4_candidates,
)
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.knob_cache import (
    knob_cache_path,
    lookup_knobs,
    record_knobs,
    resolve_knobs,
)
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.tuner import (
    default_knobs,
    is_valid,
    iter_candidates,
    with_knobs,
)
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.mxfp4_tuner import (
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
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.hopper_fp8 import (
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
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.hopper_mxfp4 import (
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
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.autotune_bf16 import (
    autotune_hopper_bf16_mega_moe,
    hopper_bf16_candidates,
)
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.tuner_bf16 import (
    default_knobs as bf16_default_knobs,
    is_valid as bf16_is_valid,
    iter_candidates as bf16_iter_candidates,
    lookup_knobs as bf16_lookup_knobs,
    record_knobs as bf16_record_knobs,
    resolve_knobs as bf16_resolve_knobs,
)
from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.hopper_bf16 import (
    MegaMoEHopperBf16Config,
    MegaMoEHopperBf16Frontend,
    MegaMoEHopperBf16Inputs,
    MegaMoEHopperBf16SymmBuffer,
    TransformedBf16Weights,
    create_dummy_inputs as create_dummy_hopper_bf16_inputs,
    get_symm_buffer_for_hopper_bf16_mega_moe,
    hopper_bf16_mega_launch_thunk,
    hopper_bf16_mega_moe,
)

from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe.hopper_nvfp4 import (
    MegaMoEHopperNvfp4Config,
    MegaMoEHopperNvfp4Frontend,
    MegaMoEHopperNvfp4Inputs,
    MegaMoEHopperNvfp4SymmBuffer,
    TransformedNvfp4Weights,
    _get_symm_buffer_for_hopper_nvfp4_mega_moe_from_resolved_config,
    _resolve_hopper_nvfp4_mega_moe_config,
    get_symm_buffer_for_hopper_nvfp4_mega_moe,
    hopper_nvfp4_mega_launch_thunk,
    hopper_nvfp4_mega_moe,
    resolve_hopper_nvfp4_knobs,
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
    "Bf16NonzeroValue",
    "bf16_reference_mm",
    "_swiglu_pair_hw_match_cuda",
    "ceil_div",
    "compute_megamoe_reference_bf16",
    "compute_megamoe_reference_fp8",
    "create_bf16_tensor",
    "create_fp8_tensor",
    "fp8_dtype_max",
    "kind_data_dtype",
    "make_constant_block_scale",
    "make_fp8_per_tensor_dequant_scale",
    "quantize_fp8_per_token_block",
    "quantize_fp8_weight_block_nk",
    "round_up",
    "to_blocked",
)


def __getattr__(name):  # PEP 562
    if name in _LAZY_HELPERS:
        from flashinfer.moe_ep.shims.sm90.cutedsl_megamoe import kernel_helpers

        return getattr(kernel_helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MegaMoEHopperNvfp4Config",
    "MegaMoEHopperNvfp4Frontend",
    "MegaMoEHopperNvfp4Inputs",
    "MegaMoEHopperNvfp4SymmBuffer",
    "TransformedNvfp4Weights",
    "_get_symm_buffer_for_hopper_nvfp4_mega_moe_from_resolved_config",
    "_resolve_hopper_nvfp4_mega_moe_config",
    "get_symm_buffer_for_hopper_nvfp4_mega_moe",
    "hopper_nvfp4_mega_launch_thunk",
    "hopper_nvfp4_mega_moe",
    "resolve_hopper_nvfp4_knobs",
    *_LAZY_HELPERS,
    "MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE",
    "MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE",
    "MXFP4_TUNING_ROUTING_PROFILES",
    "MXFP4_TUNING_TOKEN_BUCKETS",
    "MegaMoEHopperFp8Config",
    "Mxfp4RoutingProfile",
    "autotune_hopper_fp8_mega_moe",
    "autotune_hopper_mxfp4_mega_moe",
    "autotune_hopper_nvfp4_mega_moe",
    "hopper_nvfp4_candidates",
    "autotune_knobs",
    "default_knobs",
    "hopper_fp8_candidates",
    "hopper_mxfp4_default_tactic",
    "hopper_mxfp4_candidates",
    "is_hopper_mxfp4_tactic_shape_compatible",
    "is_valid_hopper_mxfp4_tactic",
    "normalize_hopper_mxfp4_routing_profile",
    "require_hopper_mxfp4_fused_tuning_device",
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
    "TransformedFp8Weights",
    "TransformedMxfp4Weights",
    "_get_symm_buffer_for_hopper_fp8_mega_moe_from_resolved_config",
    "_get_symm_buffer_for_hopper_mxfp4_mega_moe_from_resolved_config",
    "_resolve_hopper_mxfp4_mega_moe_config",
    "bootstrap_paths",
    "create_dummy_hopper_fp8_inputs",
    "create_dummy_inputs",
    "finalize_dist",
    "get_symm_buffer_for_hopper_fp8_mega_moe",
    "get_symm_buffer_for_hopper_mxfp4_mega_moe",
    "hopper_fp8_mega_launch_thunk",
    "hopper_fp8_mega_moe",
    "hopper_mxfp4_mega_launch_thunk",
    "hopper_mxfp4_mega_moe",
    "resolve_hopper_mxfp4_knobs",
    "init_dist",
    # bf16
    "MegaMoEHopperBf16Config",
    "MegaMoEHopperBf16Frontend",
    "MegaMoEHopperBf16Inputs",
    "MegaMoEHopperBf16SymmBuffer",
    "TransformedBf16Weights",
    "autotune_hopper_bf16_mega_moe",
    "bf16_default_knobs",
    "bf16_is_valid",
    "bf16_iter_candidates",
    "bf16_lookup_knobs",
    "bf16_record_knobs",
    "bf16_resolve_knobs",
    "create_dummy_hopper_bf16_inputs",
    "get_symm_buffer_for_hopper_bf16_mega_moe",
    "hopper_bf16_candidates",
    "hopper_bf16_mega_launch_thunk",
    "hopper_bf16_mega_moe",
    "resolve_hopper_fp8_mega_moe_config",
]
