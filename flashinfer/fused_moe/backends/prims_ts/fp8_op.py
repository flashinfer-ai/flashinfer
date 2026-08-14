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

"""Prims-TS FP8 per-tensor MoE public helper."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.shared.inputs import (
    MoeRunnerInputs,
    RoutingInputMode,
    alloc_trtllm_moe_output,
)
from flashinfer.prims_ts.moe.runner import (
    PrimsTsFp8BlockScaleMoERunner,
    PrimsTsFp8PerTensorMoERunner,
)
from flashinfer.prims_ts.moe.support import (
    is_prims_ts_fp8_block_scale_supported,
    is_prims_ts_fp8_per_tensor_supported,
)
from flashinfer.tllm_enums import ActivationType, Fp8QuantizationType
from flashinfer.trace.templates.moe import (
    trtllm_fp8_block_scale_moe_trace_dispatch,
    trtllm_fp8_block_scale_routed_moe_trace,
    trtllm_fp8_per_tensor_scale_moe_trace,
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


def _validate_per_channel_weight_scale(
    tensor: Optional[torch.Tensor],
    *,
    name: str,
    hidden_states: torch.Tensor,
) -> None:
    if tensor is None:
        return
    if tensor.device != hidden_states.device:
        raise ValueError(f"{name} must be on the same device as hidden_states")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError(f"{name} must be float32, float16, or bfloat16")


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

    if mode != RoutingInputMode.UnpackedPrecomputed:
        raise NotImplementedError(
            "Prims-TS FP8 block-scale currently supports FromLogits, "
            "PackedPrecomputed, and UnpackedPrecomputed routing"
        )
    if topk_weights is None:
        raise ValueError("topk_weights is required for UnpackedPrecomputed routing")
    if topk_weights.shape != topk_ids.shape:
        raise ValueError("topk_weights must have the same shape as topk_ids")
    if topk_weights.dtype != torch.bfloat16:
        raise ValueError(
            "topk_weights must be bfloat16 for UnpackedPrecomputed routing"
        )
    if topk_weights.device != hidden_states.device:
        raise ValueError("topk_weights must be on the same device as hidden_states")
    if not topk_weights.is_contiguous():
        raise ValueError("topk_weights must be contiguous")
    return None, topk_ids, topk_weights


@register_custom_op(
    "flashinfer::prims_ts_fp8_per_tensor_scale_moe",
    mutates_args=("routing_replay_out",),
)
@flashinfer_api(trace=trtllm_fp8_per_tensor_scale_moe_trace)
def prims_ts_fp8_per_tensor_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    fc1_per_channel_weight_scale: Optional[torch.Tensor] = None,
    fc2_per_channel_weight_scale: Optional[torch.Tensor] = None,
    routing_replay_out: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    r"""FP8 per-tensor-scaled MoE using the Prims-TS backend on SM100.

    Same arguments and return value as
    :func:`~flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe`.

    Parameters
    ----------
    routing_logits : torch.Tensor
        ``[seq_len, num_experts]`` routing logits.
    routing_bias : Optional[torch.Tensor]
        Optional ``[num_experts]`` routing bias.
    hidden_states : torch.Tensor
        ``float8_e4m3fn`` activations.
    gemm1_weights : torch.Tensor
        ``float8_e4m3fn`` FC1 weights.
    output1_scales_scalar : torch.Tensor
        Per-expert FC1 output scales.
    output1_scales_gate_scalar : torch.Tensor
        Per-expert FC1 gate scales.
    gemm2_weights : torch.Tensor
        ``float8_e4m3fn`` FC2 weights.
    output2_scales_scalar : torch.Tensor
        Per-expert FC2 output scales.
    num_experts : int
        Total number of experts.
    top_k : int
        Experts selected per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Groups considered for top-k routing.
    intermediate_size : int
        Intermediate (FFN) width.
    local_expert_offset : int
        Global offset of the first local expert.
    local_num_experts : int
        Number of experts resident on this device.
    routed_scaling_factor : Optional[float]
        Optional routing scale.
    use_routing_scales_on_input : bool
        Apply routing scales on the input path when ``True``.
    routing_method_type : int
        Routing method selector (default ``0``).
    weight_layout : int
        Weight layout enum value (default ``MajorK``).
    do_finalize : bool
        If ``True``, return the finalized MoE output.
    enable_pdl : Optional[bool]
        Enable Programmatic Dependent Launch when supported.
    tune_max_num_tokens : int
        Autotune token-bucket upper bound (default ``8192``).
    activation_type : int
        Activation enum value (default Swiglu).
    norm_topk_prob : bool
        Normalize top-k routing probabilities.
    fc1_per_channel_weight_scale : Optional[torch.Tensor]
        Optional per-channel FC1 weight scales.
    fc2_per_channel_weight_scale : Optional[torch.Tensor]
        Optional per-channel FC2 weight scales.
    routing_replay_out : Optional[torch.Tensor]
        Optional buffer that captures selected expert IDs.
    output : Optional[torch.Tensor]
        Optional in-place output tensor.

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Same return contract as
        :func:`~flashinfer.fused_moe.trtllm_fp8_per_tensor_scale_moe`.
    """
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise ValueError("Prims-TS FP8 per-tensor path requires float8_e4m3fn input")
    if gemm1_weights.dtype != torch.float8_e4m3fn:
        raise ValueError("gemm1_weights must be float8_e4m3fn")
    if gemm2_weights.dtype != torch.float8_e4m3fn:
        raise ValueError("gemm2_weights must be float8_e4m3fn")
    _validate_per_channel_weight_scale(
        fc1_per_channel_weight_scale,
        name="fc1_per_channel_weight_scale",
        hidden_states=hidden_states,
    )
    _validate_per_channel_weight_scale(
        fc2_per_channel_weight_scale,
        name="fc2_per_channel_weight_scale",
        hidden_states=hidden_states,
    )
    scale_dtype = None
    for _, tensor in (
        ("fc1_per_channel_weight_scale", fc1_per_channel_weight_scale),
        ("fc2_per_channel_weight_scale", fc2_per_channel_weight_scale),
    ):
        if tensor is None:
            continue
        if scale_dtype is None:
            scale_dtype = tensor.dtype
        elif tensor.dtype != scale_dtype:
            raise ValueError(
                "fc1_per_channel_weight_scale and fc2_per_channel_weight_scale "
                "must use the same dtype"
            )
    if use_routing_scales_on_input:
        if routing_logits is None:
            raise ValueError(
                "routing_logits is required when use_routing_scales_on_input=True"
            )
        for name, tensor in (
            ("fc1_per_channel_weight_scale", fc1_per_channel_weight_scale),
            ("fc2_per_channel_weight_scale", fc2_per_channel_weight_scale),
        ):
            if tensor is not None and tensor.dtype != routing_logits.dtype:
                raise ValueError(f"{name} and routing_logits must use the same dtype")

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

    topk_ids = torch.empty(0, dtype=torch.int32, device=hidden_states.device)
    expert_weights = torch.empty(
        0, dtype=routing_logits.dtype, device=hidden_states.device
    )

    moe_op = _get_moe_op()
    moe_runner = PrimsTsFp8PerTensorMoERunner(
        moe_op,
        top_k=top_k,
        num_local_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_type=activation_type,
        use_shuffled_weight=True,
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
        output1_scale_scalar=output1_scales_scalar,
        output1_scale_gate_scalar=output1_scales_gate_scalar,
        gemm2_weights=gemm2_weights,
        output2_scale_scalar=output2_scales_scalar,
        num_experts=num_experts,
        n_group=n_group,
        topk_group=topk_group,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        routed_scaling_factor=routed_scaling_factor,
        use_routing_scales_on_input=use_routing_scales_on_input,
        routing_method_type=routing_method_type,
        use_shuffled_weight=True,
        weight_layout=weight_layout,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        norm_topk_prob=norm_topk_prob,
        fc1_per_channel_weight_scale=fc1_per_channel_weight_scale,
        fc2_per_channel_weight_scale=fc2_per_channel_weight_scale,
        routing_replay_out=routing_replay_out,
    )
    moe_runner.set_cache_key_static_extras(**common_kwargs)
    ok, reason = is_prims_ts_fp8_per_tensor_supported(
        moe_runner,
        moe_inputs,
        [-1, -1],
        **common_kwargs,
    )
    if not ok:
        raise RuntimeError(
            f"Config not supported by Prims-TS FP8 per-tensor kernel ({reason})"
        )

    _, tactic = AutoTuner.get().choose_one(
        "flashinfer::prims_ts_fp8_per_tensor_scale_moe",
        [moe_runner],
        tuning_config,
        moe_inputs.to_list(),
        **common_kwargs,
    )

    resolved_tactic = [-1, -1] if tactic == -1 else tactic
    ok, reason = is_prims_ts_fp8_per_tensor_supported(
        moe_runner,
        moe_inputs,
        resolved_tactic,
        **common_kwargs,
    )
    if not ok:
        raise RuntimeError(
            f"Config not supported by Prims-TS FP8 per-tensor kernel ({reason})"
        )

    intermediate_output = moe_runner.forward(
        moe_inputs.to_list(),
        tactic=tactic,
        **common_kwargs,
    )
    if do_finalize:
        return output
    return intermediate_output


@register_fake_op("flashinfer::prims_ts_fp8_per_tensor_scale_moe")
def _fake_prims_ts_fp8_per_tensor_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    output1_scales_scalar: torch.Tensor,
    output1_scales_gate_scalar: torch.Tensor,
    gemm2_weights: torch.Tensor,
    output2_scales_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int = 0,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    fc1_per_channel_weight_scale: Optional[torch.Tensor] = None,
    fc2_per_channel_weight_scale: Optional[torch.Tensor] = None,
    routing_replay_out: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    del (
        routing_logits,
        routing_bias,
        gemm1_weights,
        output1_scales_scalar,
        output1_scales_gate_scalar,
        gemm2_weights,
        output2_scales_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        use_routing_scales_on_input,
        routing_method_type,
        weight_layout,
        enable_pdl,
        tune_max_num_tokens,
        activation_type,
        norm_topk_prob,
        fc1_per_channel_weight_scale,
        fc2_per_channel_weight_scale,
        routing_replay_out,
    )
    out = (
        output
        if output is not None
        else hidden_states.new_empty(
            hidden_states.shape[0], hidden_states.shape[1], dtype=torch.bfloat16
        )
    )
    return out if do_finalize else [out, hidden_states.new_empty(0)]


@register_custom_op(
    "flashinfer::prims_ts_fp8_block_scale_moe",
    mutates_args=("routing_replay_out",),
)
@flashinfer_api(trace=trtllm_fp8_block_scale_moe_trace_dispatch)
def prims_ts_fp8_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Fp8QuantizationType = Fp8QuantizationType.DeepSeekFp8,
    num_fused_shared_experts: Optional[int] = None,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    gemm1_bias: Optional[torch.Tensor] = None,
    gemm2_bias: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    r"""FP8 block-scaled MoE using the Prims-TS backend on SM100.

    Same arguments and return value as
    :func:`~flashinfer.fused_moe.trtllm_fp8_block_scale_moe`.

    Parameters
    ----------
    routing_logits : torch.Tensor
        ``[seq_len, num_experts]`` routing logits.
    routing_bias : Optional[torch.Tensor]
        Optional ``[num_experts]`` routing bias.
    hidden_states : torch.Tensor
        Activations (BF16/FP16 or ``float8_e4m3fn`` depending on mode).
    hidden_states_scale : torch.Tensor
        Block scales for ``hidden_states``.
    gemm1_weights : torch.Tensor
        FC1 expert weights.
    gemm1_weights_scale : torch.Tensor
        FC1 block scales.
    gemm2_weights : torch.Tensor
        FC2 expert weights.
    gemm2_weights_scale : torch.Tensor
        FC2 block scales.
    num_experts : int
        Total number of experts.
    top_k : int
        Experts selected per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Groups considered for top-k routing.
    intermediate_size : int
        Intermediate (FFN) width.
    local_expert_offset : int
        Global offset of the first local expert.
    local_num_experts : int
        Number of experts resident on this device.
    routed_scaling_factor : Optional[float]
        Optional routing scale.
    routing_method_type : int
        Routing method selector (default ``0``).
    use_shuffled_weight : bool
        Whether weights use the shuffled layout (default ``False``).
    weight_layout : int
        Weight layout enum value (default ``MajorK``).
    do_finalize : bool
        If ``True``, return the finalized MoE output.
    enable_pdl : Optional[bool]
        Enable Programmatic Dependent Launch when supported.
    tune_max_num_tokens : int
        Autotune token-bucket upper bound (default ``8192``).
    fp8_quantization_type : Fp8QuantizationType
        Block-scale recipe (DeepSeek FP8 or MXFP8).
    num_fused_shared_experts : Optional[int]
        Number of fused shared experts (default ``None`` / ``0``).
    activation_type : int
        Activation enum value (default Swiglu).
    norm_topk_prob : bool
        Normalize top-k routing probabilities.
    routing_replay_out : Optional[torch.Tensor]
        Optional buffer that captures selected expert IDs.
    gemm1_alpha : Optional[torch.Tensor]
        Optional per-expert SwiGLU alpha.
    gemm1_beta : Optional[torch.Tensor]
        Optional per-expert SwiGLU beta.
    gemm1_clamp_limit : Optional[torch.Tensor]
        Optional per-expert clamp limit.
    output : Optional[torch.Tensor]
        Optional in-place output tensor.
    gemm1_bias : Optional[torch.Tensor]
        Optional FC1 bias (keyword-only).
    gemm2_bias : Optional[torch.Tensor]
        Optional FC2 bias (keyword-only).

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Same return contract as
        :func:`~flashinfer.fused_moe.trtllm_fp8_block_scale_moe`.
    """
    return _prims_ts_fp8_block_scale_moe_impl(
        routing_logits=routing_logits,
        topk_ids=None,
        expert_weights=None,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        output=output,
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
        fp8_quantization_type=fp8_quantization_type,
        num_fused_shared_experts=num_fused_shared_experts or 0,
        activation_type=activation_type,
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        routing_input_mode=RoutingInputMode.FromLogits,
    )


@flashinfer_api(trace=trtllm_fp8_block_scale_routed_moe_trace)
def prims_ts_fp8_block_scale_routed_moe(
    topk_ids: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    gemm1_lora_delta: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Fp8QuantizationType = Fp8QuantizationType.DeepSeekFp8,
    activation_type: int = ActivationType.Swiglu.value,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    *,
    gemm1_bias: Optional[torch.Tensor] = None,
    gemm2_bias: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    r"""Pre-routed FP8 block-scaled MoE using the Prims-TS backend on SM100.

    Same arguments and return value as
    :func:`~flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe`, plus
    optional keyword-only FC1/FC2 bias tensors.

    Parameters
    ----------
    topk_ids : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Packed ``(expert_id, weight)`` tensor or unpacked
        ``(topk_ids, topk_weights)`` pair.
    routing_bias : Optional[torch.Tensor]
        Optional ``[num_experts]`` routing bias.
    hidden_states : torch.Tensor
        Activations (BF16/FP16 or ``float8_e4m3fn`` depending on mode).
    hidden_states_scale : torch.Tensor
        Block scales for ``hidden_states``.
    gemm1_weights : torch.Tensor
        FC1 expert weights.
    gemm1_weights_scale : torch.Tensor
        FC1 block scales.
    gemm2_weights : torch.Tensor
        FC2 expert weights.
    gemm2_weights_scale : torch.Tensor
        FC2 block scales.
    num_experts : int
        Total number of experts.
    top_k : int
        Experts selected per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Groups considered for top-k routing.
    intermediate_size : int
        Intermediate (FFN) width.
    local_expert_offset : int
        Global offset of the first local expert.
    local_num_experts : int
        Number of experts resident on this device.
    routed_scaling_factor : Optional[float]
        Optional routing scale.
    routing_method_type : int
        Routing method selector (default ``0``).
    use_shuffled_weight : bool
        Whether weights use the shuffled layout (default ``False``).
    weight_layout : int
        Weight layout enum value (default ``MajorK``).
    do_finalize : bool
        If ``True``, return the finalized MoE output.
    enable_pdl : Optional[bool]
        Enable Programmatic Dependent Launch when supported.
    gemm1_lora_delta : Optional[torch.Tensor]
        Optional MoE LoRA delta applied before the gated activation.
    output : Optional[torch.Tensor]
        Optional in-place output tensor.
    tune_max_num_tokens : int
        Autotune token-bucket upper bound (default ``8192``).
    fp8_quantization_type : Fp8QuantizationType
        Block-scale recipe (DeepSeek FP8 or MXFP8).
    activation_type : int
        Activation enum value (default Swiglu).
    gemm1_alpha : Optional[torch.Tensor]
        Optional per-expert SwiGLU alpha.
    gemm1_beta : Optional[torch.Tensor]
        Optional per-expert SwiGLU beta.
    gemm1_clamp_limit : Optional[torch.Tensor]
        Optional per-expert clamp limit.
    gemm1_bias : Optional[torch.Tensor]
        Optional FC1 bias (keyword-only).
    gemm2_bias : Optional[torch.Tensor]
        Optional FC2 bias (keyword-only).

    Returns
    -------
    torch.Tensor or List[torch.Tensor]
        Same return contract as
        :func:`~flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe`.
    """
    if isinstance(topk_ids, tuple):
        topk_ids_tensor, expert_weights = topk_ids
        routing_mode = RoutingInputMode.UnpackedPrecomputed
    else:
        topk_ids_tensor = topk_ids
        expert_weights = None
        routing_mode = RoutingInputMode.PackedPrecomputed

    return _prims_ts_fp8_block_scale_moe_impl(
        routing_logits=None,
        topk_ids=topk_ids_tensor,
        expert_weights=expert_weights,
        routing_bias=routing_bias,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        output=output,
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
        fp8_quantization_type=fp8_quantization_type,
        num_fused_shared_experts=0,
        activation_type=activation_type,
        norm_topk_prob=True,
        routing_replay_out=None,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        gemm1_lora_delta=gemm1_lora_delta,
        routing_input_mode=routing_mode,
    )


def _prims_ts_fp8_block_scale_moe_impl(
    *,
    routing_logits: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    expert_weights: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm2_bias: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
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
    do_finalize: bool,
    enable_pdl: Optional[bool],
    tune_max_num_tokens: int,
    fp8_quantization_type: Fp8QuantizationType,
    num_fused_shared_experts: int,
    activation_type: int,
    norm_topk_prob: bool,
    routing_replay_out: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    routing_input_mode: int,
    gemm1_lora_delta: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    if hidden_states.dtype != torch.float8_e4m3fn:
        raise ValueError("Prims-TS FP8 block-scale path requires float8_e4m3fn input")
    if gemm1_weights.dtype != torch.float8_e4m3fn:
        raise ValueError("gemm1_weights must be float8_e4m3fn")
    if gemm2_weights.dtype != torch.float8_e4m3fn:
        raise ValueError("gemm2_weights must be float8_e4m3fn")
    if gemm1_lora_delta is not None:
        raise NotImplementedError("Prims-TS FP8 block-scale path does not support LoRA")
    if num_fused_shared_experts:
        raise NotImplementedError(
            "Prims-TS FP8 block-scale path does not support fused shared experts"
        )

    if enable_pdl is None:
        enable_pdl = device_support_pdl(hidden_states.device)

    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[-1]
    if (
        fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8
        and hidden_states_scale is not None
        and not hidden_states_scale.is_contiguous()
    ):
        # DeepSeek TS kernels consume raw FP32 scales as [K-blocks, tokens].
        # Normalize strided logical views such as scales.t() before passing
        # their data pointer to the kernel.
        hidden_states_scale = hidden_states_scale.contiguous()
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
        topk_weights=expert_weights,
        hidden_states=hidden_states,
    )

    moe_op = _get_moe_op()
    moe_runner = PrimsTsFp8BlockScaleMoERunner(
        moe_op,
        top_k=top_k,
        num_local_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        fp8_quantization_type=fp8_quantization_type,
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
        hidden_states_scale=hidden_states_scale,
        gemm1_lora_delta=gemm1_lora_delta,
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
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
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
        fp8_quantization_type=fp8_quantization_type,
        num_fused_shared_experts=num_fused_shared_experts,
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        routing_input_mode=routing_input_mode,
    )
    moe_runner.set_cache_key_static_extras(**common_kwargs)
    ok, reason = is_prims_ts_fp8_block_scale_supported(
        moe_runner,
        moe_inputs,
        [-1, -1],
        **common_kwargs,
    )
    if not ok:
        raise RuntimeError(
            f"Config not supported by Prims-TS FP8 block-scale kernel ({reason})"
        )

    _, tactic = AutoTuner.get().choose_one(
        "flashinfer::prims_ts_fp8_block_scale_moe",
        [moe_runner],
        tuning_config,
        moe_inputs.to_list(),
        **common_kwargs,
    )

    resolved_tactic = [-1, -1] if tactic == -1 else tactic
    ok, reason = is_prims_ts_fp8_block_scale_supported(
        moe_runner,
        moe_inputs,
        resolved_tactic,
        **common_kwargs,
    )
    if not ok:
        raise RuntimeError(
            f"Config not supported by Prims-TS FP8 block-scale kernel ({reason})"
        )

    intermediate_output = moe_runner.forward(
        moe_inputs.to_list(),
        tactic=tactic,
        **common_kwargs,
    )
    if do_finalize:
        return output
    return intermediate_output


@register_fake_op("flashinfer::prims_ts_fp8_block_scale_moe")
def _fake_prims_ts_fp8_block_scale_moe(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int = 0,
    use_shuffled_weight: bool = False,
    weight_layout: int = 0,
    do_finalize: bool = True,
    enable_pdl: Optional[bool] = None,
    tune_max_num_tokens: int = 8192,
    fp8_quantization_type: Fp8QuantizationType = Fp8QuantizationType.DeepSeekFp8,
    num_fused_shared_experts: Optional[int] = None,
    activation_type: int = ActivationType.Swiglu.value,
    norm_topk_prob: bool = True,
    routing_replay_out: Optional[torch.Tensor] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    *,
    gemm1_bias: Optional[torch.Tensor] = None,
    gemm2_bias: Optional[torch.Tensor] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    del (
        routing_logits,
        routing_bias,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
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
        fp8_quantization_type,
        num_fused_shared_experts,
        activation_type,
        norm_topk_prob,
        routing_replay_out,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm1_bias,
        gemm2_bias,
    )
    out = (
        output
        if output is not None
        else hidden_states.new_empty(
            hidden_states.shape[0], hidden_states.shape[1], dtype=torch.bfloat16
        )
    )
    return out if do_finalize else [out, hidden_states.new_empty(0)]
