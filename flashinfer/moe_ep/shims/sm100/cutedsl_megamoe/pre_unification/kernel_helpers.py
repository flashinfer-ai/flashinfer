# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Provider helpers used by the NVFP4 and BF16 adapters and their references."""

from __future__ import annotations

import importlib
import inspect

# --- eager: light, import-safe (no cutlass/nvshmem pulled at module load) ---
from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.host_utils import (
    kind_data_dtype,
    mxfp8_quantize_per_block_32_row,
)
from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.megamoe_constants import (
    Mxfp8BlockSize,
    Nvfp4BlockSize,
)
from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.runner_common import (
    Mxfp8ScaleDtype,
    _stack_byte_reinterpretable_tensors,
    ceil_div,
    nvfp4_quantize_per_block_16,
    round_up,
    to_blocked,
)

from flashinfer.moe_ep._compat import deprecated_aliases

mxfp8_quantize_per_block_32 = mxfp8_quantize_per_block_32_row

# --- lazy: pull cutlass transitively; imported only on first attribute access
#     (which happens inside the backend/test call sites, never at package load).
_LAZY = {
    "Contract": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.contract",
        "Contract",
    ),
    "FunctionMapping": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.contract",
        "FunctionMapping",
    ),
    "GpuReleaseFlagBatchTracker": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.comm.flag_batch",
        "GpuReleaseFlagBatchTracker",
    ),
    "MoESchedConsumer": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_persistent_scheduler",
        "MoESchedConsumer",
    ),
    "MoESchedExtension": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_persistent_scheduler",
        "MoESchedExtension",
    ),
    "MoESchedulerBase": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_persistent_scheduler",
        "MoESchedulerBase",
    ),
    "MoESchedulerParamsBase": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_persistent_scheduler",
        "MoESchedulerParamsBase",
    ),
    "MoEWorkTileInfo": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_persistent_scheduler",
        "MoEWorkTileInfo",
    ),
    "Space": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.contract",
        "Space",
    ),
    "SymBufferDeviceBase": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.comm.sym_buffer",
        "SymBufferDeviceBase",
    ),
    "SymBufferHost": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.comm.sym_buffer",
        "SymBufferHost",
    ),
    "TokenCommArgs": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.comm.token_comm",
        "TokenCommArgs",
    ),
    "TokenInPullTokenBackPush": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.comm.token_comm",
        "TokenInPullTokenBackPush",
    ),
    "TokenSrcMetadata": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.comm.token_comm",
        "TokenSrcMetadata",
    ),
    "TopkReduce": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.topk_reduce",
        "TopkReduce",
    ),
    "WorkTileState": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_persistent_scheduler",
        "WorkTileState",
    ),
    "_DEFAULT_SCHED_EXT": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_persistent_scheduler",
        "_DEFAULT_SCHED_EXT",
    ),
    "compute_expert_token_count_from_sizes": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_utils",
        "compute_expert_token_count_from_sizes",
    ),
    "eval_function_mapping": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.contract",
        "eval_function_mapping",
    ),
    "fmax": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.moe_utils",
        "fmax",
    ),
    "fmin": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.moe_utils",
        "fmin",
    ),
    "get_cutedsl_target_arch": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.common.host_utils",
        "get_cutedsl_target_arch",
    ),
    "iket": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.comm.iket_compat",
        "iket",
    ),
    "mbarrier_arrive_expect_tx_on_peer": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_utils",
        "mbarrier_arrive_expect_tx_on_peer",
    ),
    "rewrite_tensor_shape": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_utils",
        "rewrite_tensor_shape",
    ),
    "spin_wait": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_utils",
        "spin_wait",
    ),
    "store_i32_to_peer_cluster_smem_async": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.moe_utils",
        "store_i32_to_peer_cluster_smem_async",
    ),
    "CombineFormat": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.comm.token_comm",
        "CombineFormat",
    ),
    "combine_roundtrip_to_fp32": (
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.mega_reference",
        "combine_roundtrip_to_fp32",
    ),
}


def _reference_with_routing_weight_name(
    module_name,
    function_name,
    args,
    kwargs,
    apply_routing_weights_before_fc2,
):
    function = getattr(importlib.import_module(module_name), function_name)
    bound = inspect.signature(function).bind_partial(*args, **kwargs)
    if (
        apply_routing_weights_before_fc2 is not None
        or "apply_topk_in_fc1" not in bound.arguments
    ):
        bound.arguments["apply_topk_in_fc1"] = (
            False
            if apply_routing_weights_before_fc2 is None
            else apply_routing_weights_before_fc2
        )
    return function(*bound.args, **bound.kwargs)


@deprecated_aliases(apply_topk_in_fc1="apply_routing_weights_before_fc2")
def compute_megamoe_reference_bf16_mxfp8(
    *args, apply_routing_weights_before_fc2=None, **kwargs
):
    return _reference_with_routing_weight_name(
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_mxfp8_bf16_glu.mega_reference_mxfp8_bf16",
        "compute_megamoe_reference_mxfp8_bf16",
        args,
        kwargs,
        apply_routing_weights_before_fc2,
    )


@deprecated_aliases(apply_topk_in_fc1="apply_routing_weights_before_fc2")
def compute_megamoe_reference_bf16(
    *args, apply_routing_weights_before_fc2=None, **kwargs
):
    return _reference_with_routing_weight_name(
        "flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_bf16_glu.mega_reference_bf16",
        "compute_megamoe_reference",
        args,
        kwargs,
        apply_routing_weights_before_fc2,
    )


def __getattr__(name):  # PEP 562: keep cutlass out of the import-time path
    try:
        module, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(module), attr)


__all__ = [
    # eager
    "Mxfp8BlockSize",
    "Mxfp8ScaleDtype",
    "Nvfp4BlockSize",
    "ceil_div",
    "kind_data_dtype",
    "mxfp8_quantize_per_block_32",
    "nvfp4_quantize_per_block_16",
    "round_up",
    "to_blocked",
    "_stack_byte_reinterpretable_tensors",
    # lazy (resolved via PEP 562 __getattr__ above)
    "CombineFormat",  # noqa: F822
    "combine_roundtrip_to_fp32",  # noqa: F822
    "compute_megamoe_reference_bf16",  # noqa: F822
    "compute_megamoe_reference_bf16_mxfp8",  # noqa: F822
]
__all__ += [name for name in _LAZY if name not in __all__]
