# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Expose provider helpers through the Hopper adapter.

Resolve constants, quantization helpers and references on first access because
their provider modules import the CuTe compiler.
"""

from __future__ import annotations

import importlib

# name -> (module, attribute).  All of these pull cutlass transitively via
# common.megamoe_constants; resolved on first access only.
_LAZY = {
    # constants (common/megamoe_constants.py)
    "Fp8BlockScaleK": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.megamoe_constants",
        "Fp8BlockScaleK",
    ),
    "Fp8E8M0SfVecSize": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.megamoe_constants",
        "Fp8E8M0SfVecSize",
    ),
    "Fp8Fc2ActivationScaleK": (
        __package__ + ".hopper_fp8",
        "_FP8_FC2_ACTIVATION_SCALE_K",
    ),
    "Fp8GateUpInterleave": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.megamoe_constants",
        "Fp8GateUpInterleave",
    ),
    "Fp8WeightScaleBlockK": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.megamoe_constants",
        "Fp8WeightScaleBlockK",
    ),
    "Fp8WeightScaleBlockN": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.megamoe_constants",
        "Fp8WeightScaleBlockN",
    ),
    # generic host helpers (common/host_utils.py, moe_nvfp4_swapab/runner_common.py)
    "kind_data_dtype": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.host_utils",
        "kind_data_dtype",
    ),
    "ceil_div": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.runner_common",
        "ceil_div",
    ),
    "round_up": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.runner_common",
        "round_up",
    ),
    "to_blocked": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.runner_common",
        "to_blocked",
    ),
    "_stack_byte_reinterpretable_tensors": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.runner_common",
        "_stack_byte_reinterpretable_tensors",
    ),
    # FP8 quant helpers (moe_hopper_fp8/hopper_moe_utils.py)
    "fp8_dtype_max": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_fp8.hopper_moe_utils",
        "fp8_dtype_max",
    ),
    "create_fp8_tensor": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_fp8.hopper_moe_utils",
        "create_fp8_tensor",
    ),
    "make_constant_block_scale": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_fp8.hopper_moe_utils",
        "make_constant_block_scale",
    ),
    "make_fp8_per_tensor_dequant_scale": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_fp8.hopper_moe_utils",
        "make_fp8_per_tensor_dequant_scale",
    ),
    "quantize_fp8_per_token_block": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_fp8.hopper_moe_utils",
        "quantize_fp8_per_token_block",
    ),
    "quantize_fp8_weight_block_nk": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_fp8.hopper_moe_utils",
        "quantize_fp8_weight_block_nk",
    ),
    "_swiglu_pair_hw_match_cuda": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.runner_common",
        "_swiglu_pair_hw_match_cuda",
    ),
    # ground-truth torch reference (moe_hopper_fp8/mega_reference_fp8.py)
    "compute_megamoe_reference_fp8": (
        __name__,
        "_compute_megamoe_reference_fp8",
    ),
    # BF16 helpers (moe_hopper_bf16/hopper_moe_utils.py) + torch reference
    # (moe_hopper_bf16/mega_reference_bf16.py)
    "Bf16NonzeroValue": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_bf16.hopper_moe_utils",
        "Bf16NonzeroValue",
    ),
    "create_bf16_tensor": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_bf16.hopper_moe_utils",
        "create_bf16_tensor",
    ),
    "bf16_reference_mm": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_bf16.hopper_moe_utils",
        "bf16_reference_mm",
    ),
    "compute_megamoe_reference_bf16": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_bf16.mega_reference_bf16",
        "compute_megamoe_reference_bf16",
    ),
}


def _compute_megamoe_reference_fp8(*args, **kwargs):
    """Use the intermediate scale block selected by the FI Hopper kernels."""
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_hopper_fp8.mega_reference_fp8 import (
        compute_megamoe_reference_fp8,
    )

    kwargs.setdefault("fc2_activation_scale_k", 64)
    return compute_megamoe_reference_fp8(*args, **kwargs)


def __getattr__(name):  # PEP 562: keep cutlass out of the import-time path
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module), attr)


__all__ = list(_LAZY)  # all lazy (resolved via PEP 562 __getattr__ above)
