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

"""Prims-TS BF16 MoE public API and custom-op registration."""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.backends.trtllm.validation import (
    validate_bf16_gemm1_activation_params as _validate_bf16_gemm1_activation_params,
)
from flashinfer.fused_moe.shared.inputs import (
    MoeRunnerInputs,
    alloc_trtllm_moe_output,
    unpack_trtllm_moe_output,
)
from flashinfer.prims_ts.moe.runner import PrimsTsBf16MoERunner
from flashinfer.prims_ts.moe.support import is_prims_ts_bf16_supported
from flashinfer.tllm_enums import ActivationType, WeightLayout
from flashinfer.trace.templates.moe import (
    trtllm_bf16_moe_trace,
    trtllm_bf16_routed_moe_trace,
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


@register_custom_op(
    "flashinfer::prims_ts_bf16_moe",
    mutates_args=("routing_replay_out",),
)
def prims_ts_bf16_moe_op(
    routing_logits: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    expert_weights: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm1_lora_delta: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    use_shuffled_weight: bool,
    weight_layout: int,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    assert routing_logits is not None or topk_ids is not None, (
        "either routing_logits or topk_ids must be provided"
    )
    if gemm1_lora_delta is not None:
        raise NotImplementedError("Prims-TS BF16 MoE does not support gemm1_lora_delta")

    _validate_bf16_gemm1_activation_params(
        activation_type,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        local_num_experts,
        hidden_states.device,
    )
    if enable_pdl is None:
        enable_pdl = device_support_pdl(hidden_states.device)

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[-1]
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

    if routing_logits is not None:
        topk_ids = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
        expert_weights = torch.empty(
            0, dtype=routing_logits.dtype, device=hidden_states.device
        )
    else:
        assert topk_ids is not None
        if topk_ids.dtype != torch.int32:
            raise ValueError("topk_ids must be int32 for Prims-TS BF16 precomputed routing")
        expert_weights = (
            expert_weights
            if expert_weights is not None
            else torch.empty(0, dtype=torch.bfloat16, device=hidden_states.device)
        )

    moe_op = _get_moe_op()
    moe_runner = PrimsTsBf16MoERunner(
        moe_op,
        top_k=top_k,
        num_local_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_type=activation_type,
        use_shuffled_weight=use_shuffled_weight,
        weight_layout=weight_layout,
        num_experts=num_experts,
    )
    moe_inputs = MoeRunnerInputs(
        output=output,
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        expert_weights=expert_weights,
        hidden_states=hidden_states,
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
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
        gemm2_weights=gemm2_weights,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        num_experts=num_experts,
        n_group=n_group,
        topk_group=topk_group,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
        use_shuffled_weight=use_shuffled_weight,
        weight_layout=weight_layout,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
    )
    moe_runner.set_cache_key_static_extras(**common_kwargs)
    ok, reason = is_prims_ts_bf16_supported(
        moe_runner,
        moe_inputs,
        [-1, -1],
        **common_kwargs,
    )
    if not ok:
        raise RuntimeError(
            f"Config not supported by Prims-TS BF16 kernel ({reason})"
        )

    _, tactic = AutoTuner.get().choose_one(
        "flashinfer::prims_ts_bf16_moe",
        [moe_runner],
        tuning_config,
        moe_inputs.to_list(),
        **common_kwargs,
    )

    ok, reason = is_prims_ts_bf16_supported(
        moe_runner,
        moe_inputs,
        [-1, -1] if tactic == -1 else tactic,
        **common_kwargs,
    )
    if not ok:
        raise RuntimeError(
            f"Config not supported by Prims-TS BF16 kernel ({reason})"
        )

    intermediate_output = moe_runner.forward(
        moe_inputs.to_list(),
        tactic=tactic,
        **common_kwargs,
    )
    return unpack_trtllm_moe_output(intermediate_output, output, do_finalize, None)


@register_fake_op("flashinfer::prims_ts_bf16_moe")
def _fake_prims_ts_bf16_moe(
    routing_logits: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    expert_weights: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm1_lora_delta: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    use_shuffled_weight: bool,
    weight_layout: int,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    del (
        routing_bias,
        topk_ids,
        expert_weights,
        gemm1_weights,
        gemm2_weights,
        gemm1_lora_delta,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        routing_method_type,
        use_shuffled_weight,
        weight_layout,
        enable_pdl,
        tune_max_num_tokens,
        activation_type,
        norm_topk_prob,
        routing_replay_out,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
    )
    seq_len = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    out = output if output is not None else hidden_states.new_empty(
        seq_len, hidden_size, dtype=torch.bfloat16
    )
    return [out] if do_finalize else [out, hidden_states.new_empty(0)]


@flashinfer_api(trace=trtllm_bf16_moe_trace)
def prims_ts_bf16_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float] = None,
    routing_method_type: int = 0,
    use_shuffled_weight: bool = True,
    weight_layout: int = WeightLayout.MajorK,
    do_finalize: bool = True,
    enable_pdl: bool = True,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """BF16 MoE using the Prims-TS batched GEMM backend on SM100."""

    result = prims_ts_bf16_moe_op(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        topk_ids=None,
        expert_weights=None,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        gemm1_lora_delta=None,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
        use_shuffled_weight=use_shuffled_weight,
        weight_layout=weight_layout,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        tune_max_num_tokens=tune_max_num_tokens,
        activation_type=activation_type,
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        output=output,
    )
    return result[0] if do_finalize and len(result) == 1 else result


@flashinfer_api(trace=trtllm_bf16_routed_moe_trace)
def prims_ts_bf16_routed_moe(
    topk_ids: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm2_weights: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float] = None,
    routing_method_type: int = 0,
    use_shuffled_weight: bool = True,
    weight_layout: int = WeightLayout.MajorK,
    do_finalize: bool = True,
    enable_pdl: bool = True,
    gemm1_lora_delta: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    routing_replay_out: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """BF16 Prims-TS MoE with precomputed packed or unpacked routing."""

    if isinstance(topk_ids, tuple):
        topk_ids_tensor, expert_weights = topk_ids
    else:
        topk_ids_tensor = topk_ids
        expert_weights = None

    result = prims_ts_bf16_moe_op(
        routing_logits=None,
        routing_bias=None,
        topk_ids=topk_ids_tensor,
        expert_weights=expert_weights,
        hidden_states=hidden_states,
        gemm1_weights=gemm1_weights,
        gemm2_weights=gemm2_weights,
        gemm1_lora_delta=gemm1_lora_delta,
        num_experts=num_experts,
        top_k=top_k,
        n_group=n_group,
        topk_group=topk_group,
        intermediate_size=intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=routing_method_type,
        use_shuffled_weight=use_shuffled_weight,
        weight_layout=weight_layout,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        tune_max_num_tokens=tune_max_num_tokens,
        activation_type=activation_type,
        norm_topk_prob=True,
        routing_replay_out=routing_replay_out,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        output=output,
    )
    return result[0] if do_finalize and gemm1_lora_delta is None else result
