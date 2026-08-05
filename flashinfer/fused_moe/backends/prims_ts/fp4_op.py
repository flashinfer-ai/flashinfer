# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prims-TS NVFP4xNVFP4 MoE public helper."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.shared.inputs import (
    MoeRunnerInputs,
    RoutingInputMode,
    alloc_trtllm_moe_output,
    unpack_trtllm_moe_output,
)
from flashinfer.prims_ts.moe.runner import (
    PrimsTsMxfp4Bf16MoERunner,
    PrimsTsMxfp4Mxfp8MoERunner,
    PrimsTsNvfp4MoERunner,
)
from flashinfer.prims_ts.moe.support import (
    is_prims_ts_mxfp4_bf16_supported,
    is_prims_ts_mxfp4_mxfp8_supported,
    is_prims_ts_nvfp4_supported,
)
from flashinfer.tllm_enums import ActivationType
from flashinfer.trace.templates.moe import (
    trtllm_fp4_block_scale_moe_trace_dispatch,
    trtllm_fp4_block_scale_routed_moe_trace,
)
from flashinfer.utils import (
    check_shape_dtype_device,
    device_support_pdl,
    register_custom_op,
    register_fake_op,
)


def _get_moe_op():
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module

    return get_trtllm_moe_sm100_module().moe_op


def _resolve_routing_inputs(
    *,
    routing_input_mode: int,
    routing_logits: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    mode = RoutingInputMode(routing_input_mode)
    if mode == RoutingInputMode.FromLogits:
        if routing_logits is None:
            raise ValueError("routing_logits is required for FromLogits routing")
        return (
            routing_logits,
            torch.empty(0, dtype=torch.int32, device=hidden_states.device),
            torch.empty(0, dtype=routing_logits.dtype, device=hidden_states.device),
        )

    if topk_ids is None:
        raise ValueError("topk_ids is required for precomputed routing")
    if routing_logits is not None:
        raise ValueError("routing_logits must be None for precomputed routing")
    if topk_ids.dtype != torch.int32:
        raise ValueError("topk_ids must be int32 for precomputed routing")
    if topk_ids.device != hidden_states.device:
        raise ValueError("topk_ids must be on the same device as hidden_states")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")

    if mode == RoutingInputMode.PackedPrecomputed:
        return (
            None,
            topk_ids,
            torch.empty(0, dtype=torch.bfloat16, device=hidden_states.device),
        )

    if topk_weights is None:
        raise ValueError("topk_weights is required for UnpackedPrecomputed routing")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights must have the same shape as topk_ids")
    if topk_weights.device != hidden_states.device:
        raise ValueError("topk_weights must be on the same device as hidden_states")
    if not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous")
    return None, topk_ids, topk_weights


@register_custom_op(
    "flashinfer::prims_ts_fp4_block_scale_moe",
    mutates_args=("routing_replay_out",),
)
@flashinfer_api(trace=trtllm_fp4_block_scale_moe_trace_dispatch)
def prims_ts_fp4_block_scale_moe(
    routing_logits: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = ActivationType.Swiglu.value,
    per_token_scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    routing_input_mode: int = RoutingInputMode.FromLogits,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    if hidden_states.dtype == torch.uint8:
        runner_cls = PrimsTsNvfp4MoERunner
        support_fn = is_prims_ts_nvfp4_supported
        mode_name = "NVFP4xNVFP4"
        hidden_size = hidden_states.shape[-1] * 2
        if hidden_states_scale is None:
            raise ValueError("hidden_states_scale is required for NVFP4 activations")
        if hidden_states_scale.dtype != torch.float8_e4m3fn:
            raise ValueError("hidden_states_scale must be float8_e4m3fn for NVFP4")
    elif hidden_states.dtype == torch.float8_e4m3fn:
        runner_cls = PrimsTsMxfp4Mxfp8MoERunner
        support_fn = is_prims_ts_mxfp4_mxfp8_supported
        mode_name = "MXFP4xMXFP8"
        hidden_size = hidden_states.shape[-1]
        if hidden_states_scale is None:
            raise ValueError("hidden_states_scale is required for MXFP8 activations")
    elif hidden_states.dtype == torch.bfloat16:
        runner_cls = PrimsTsMxfp4Bf16MoERunner
        support_fn = is_prims_ts_mxfp4_bf16_supported
        mode_name = "MXFP4xBF16"
        hidden_size = hidden_states.shape[-1]
        if hidden_states_scale is not None:
            raise ValueError("hidden_states_scale must be None for BF16 activations")
    else:
        raise ValueError(
            "Prims-TS FP4 path supports packed NVFP4 uint8, MXFP8 float8_e4m3fn, "
            "or BF16 activations"
        )

    if enable_pdl is None:
        enable_pdl = device_support_pdl(hidden_states.device)

    num_tokens = hidden_states.shape[0]
    if output is None:
        output = alloc_trtllm_moe_output(
            num_tokens, hidden_size, do_finalize, hidden_states.device
        )
    elif do_finalize:
        check_shape_dtype_device(
            output,
            (num_tokens, hidden_size),
            torch.bfloat16,
            hidden_states.device,
            "output",
        )

    routing_logits, topk_ids, expert_weights = _resolve_routing_inputs(
        routing_input_mode=routing_input_mode,
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
    )

    moe_op = _get_moe_op()
    moe_runner = runner_cls(
        moe_op,
        top_k=top_k,
        num_local_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_type=activation_type,
        use_shuffled_weight=True,
        weight_layout=weight_layout,
        use_per_token_scaling=per_token_scale is not None,
        num_experts=num_experts,
    )
    moe_inputs = MoeRunnerInputs(
        output=output,
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        expert_weights=expert_weights,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_lora_delta=None,
        per_token_scale=per_token_scale,
    )
    tuning_config = moe_runner._make_tuning_config(
        moe_inputs,
        tune_max_num_tokens=tune_max_num_tokens,
        use_cuda_graph=True,
        use_cold_l2_cache=True,
    )

    common_kwargs = dict(
        routing_bias=routing_bias,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm2_bias=gemm2_bias,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        per_token_scale=per_token_scale,
        num_experts=num_experts,
        n_group=n_group,
        topk_group=topk_group,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
        use_shuffled_weight=True,
        weight_layout=weight_layout,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
        routing_input_mode=routing_input_mode,
    )
    moe_runner.set_cache_key_static_extras(**common_kwargs)
    ok, reason = support_fn(moe_runner, moe_inputs, [-1, -1], **common_kwargs)
    if not ok:
        raise RuntimeError(f"Config not supported by Prims-TS {mode_name} kernel ({reason})")

    _, tactic = AutoTuner.get().choose_one(
        "flashinfer::prims_ts_fp4_block_scale_moe",
        [moe_runner],
        tuning_config,
        moe_inputs.to_list(),
        **common_kwargs,
    )

    resolved_tactic = [-1, -1] if tactic == -1 else tactic
    ok, reason = support_fn(
        moe_runner,
        moe_inputs,
        resolved_tactic,
        **common_kwargs,
    )
    if not ok:
        raise RuntimeError(f"Config not supported by Prims-TS {mode_name} kernel ({reason})")

    intermediate_output = moe_runner.forward(
        moe_inputs.to_list(),
        tactic=tactic,
        **common_kwargs,
    )
    return unpack_trtllm_moe_output(intermediate_output, output, do_finalize, None)


@register_fake_op("flashinfer::prims_ts_fp4_block_scale_moe")
def _fake_prims_ts_fp4_block_scale_moe(
    routing_logits: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = ActivationType.Swiglu.value,
    per_token_scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    routing_input_mode: int = RoutingInputMode.FromLogits,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    del (
        routing_logits,
        routing_bias,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_bias,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights,
        gemm2_weights_scale,
        gemm2_bias,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        weight_layout,
        enable_pdl,
        activation_type,
        per_token_scale,
        tune_max_num_tokens,
        norm_topk_prob,
        routing_replay_out,
        routing_input_mode,
        topk_ids,
        topk_weights,
    )
    hidden_size = (
        hidden_states.shape[-1] * 2
        if hidden_states.dtype == torch.uint8
        else hidden_states.shape[-1]
    )
    out = output if output is not None else hidden_states.new_empty(
        hidden_states.shape[0], hidden_size, dtype=torch.bfloat16
    )
    return [out] if do_finalize else [out, hidden_states.new_empty(0)]


@flashinfer_api(trace=trtllm_fp4_block_scale_routed_moe_trace)
def prims_ts_fp4_block_scale_routed_moe(
    topk_ids: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: Optional[torch.Tensor],
    output1_scale_gate_scalar: Optional[torch.Tensor],
    output2_scale_scalar: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    activation_type: int = ActivationType.Swiglu.value,
    per_token_scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
) -> List[torch.Tensor]:
    if isinstance(topk_ids, tuple):
        topk_ids_tensor, topk_weights = topk_ids
        routing_mode = RoutingInputMode.UnpackedPrecomputed
    else:
        topk_ids_tensor = topk_ids
        topk_weights = None
        routing_mode = RoutingInputMode.PackedPrecomputed

    return prims_ts_fp4_block_scale_moe(
        routing_logits=None,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm2_bias=gemm2_bias,
        output1_scale_scalar=output1_scale_scalar,
        output1_scale_gate_scalar=output1_scale_gate_scalar,
        output2_scale_scalar=output2_scale_scalar,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
        weight_layout=weight_layout,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        per_token_scale=per_token_scale,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        norm_topk_prob=True,
        routing_replay_out=None,
        routing_input_mode=routing_mode,
        topk_ids=topk_ids_tensor,
        topk_weights=topk_weights,
    )
