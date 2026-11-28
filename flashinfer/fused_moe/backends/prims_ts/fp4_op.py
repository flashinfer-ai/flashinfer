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

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.shared.inputs import (
    MoeRunnerInputs,
    RoutingInputMode,
    alloc_trtllm_moe_output,
    unpack_trtllm_moe_output,
)
from flashinfer.prims_ts.moe.partition import (
    MoePartitionResources,
    PartitionStrategy,
    prepared_block_major_k_bytes,
    prepared_weight_layouts,
    require_partition_pair,
    validate_partition_arch,
)
from flashinfer.prims_ts.moe.runner import (
    PrimsTsMxfp4Bf16MoERunner,
    PrimsTsMxfp4Mxfp8MoERunner,
    PrimsTsNvfp4MoERunner,
    PrimsTsPartitionedMxfp4Bf16MoERunner,
    PrimsTsPartitionedMxfp4Mxfp8MoERunner,
    PrimsTsPartitionedNvfp4MoERunner,
)
from flashinfer.prims_ts.moe.support import (
    is_prims_ts_mxfp4_bf16_supported,
    is_prims_ts_mxfp4_mxfp8_supported,
    is_prims_ts_nvfp4_supported,
)
from flashinfer.tllm_enums import ActivationType, is_gated_activation
from flashinfer.trace.templates.moe import (
    prims_ts_fp4_block_scale_moe_partitioned_trace_dispatch,
    trtllm_fp4_block_scale_moe_trace_dispatch,
    trtllm_fp4_block_scale_routed_moe_trace,
)
from flashinfer.utils import (
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    register_custom_op,
    register_fake_op,
)

if TYPE_CHECKING:
    from flashinfer.fused_moe.core import (
        TrtllmMoERoutingMetadata,
        TrtllmMoERoutingMetadataSlot,
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


def _check_routing_metadata_args(
    *,
    routing_metadata,
    hidden_states: torch.Tensor,
    routing_input_mode: int,
    routing_replay_out: Optional[torch.Tensor],
    num_fused_shared_experts: Optional[int],
    num_experts: int,
    top_k: int,
    local_expert_offset: int,
    local_num_experts: int,
) -> None:
    # A collection records the expert interval it was populated for; a bare slot does not.
    if not hasattr(routing_metadata, "tile_n"):
        populated = (
            routing_metadata.num_experts,
            routing_metadata.top_k,
            routing_metadata.local_expert_offset,
            routing_metadata.num_local_experts,
        )
        call = (num_experts, top_k, local_expert_offset, local_num_experts)
        if populated != call:
            raise ValueError(
                "routing_metadata was populated for (num_experts, top_k, "
                f"local_expert_offset, num_local_experts)={populated}, but the call "
                f"uses {call}"
            )
    if hidden_states.dtype != torch.uint8:
        raise ValueError("routing_metadata supports NVFP4 activations only")
    if RoutingInputMode(routing_input_mode) == RoutingInputMode.FromLogits:
        raise ValueError("routing_metadata requires precomputed topk_ids")
    if routing_replay_out is not None or num_fused_shared_experts:
        raise ValueError(
            "routing_metadata supports neither routing_replay_out nor fused shared experts"
        )


def _tactic_tile_n(runner, moe_inputs: MoeRunnerInputs, tactic, kwargs: dict) -> int:
    """The tile forward() runs for this tactic, default included."""
    if tactic == -1 and runner._is_partitioned:
        tactic = runner.get_valid_tactics(moe_inputs.to_list(), None)[0]
    return int(
        runner._map_tactic(
            [-1, -1] if tactic == -1 else tactic, moe_inputs, kwargs
        ).tile_n
    )


def _routing_metadata_tactic(
    runner,
    moe_inputs: MoeRunnerInputs,
    tactic,
    routing_metadata: Union["TrtllmMoERoutingMetadataSlot", "TrtllmMoERoutingMetadata"],
    kwargs: dict,
):
    """Pick the tactic and the metadata slot it runs with."""
    inputs = moe_inputs.to_list()
    tile_n = _tactic_tile_n(runner, moe_inputs, tactic, kwargs)
    slots = getattr(routing_metadata, "slots", None) or (routing_metadata,)
    slot_by_tile = {int(slot.tile_n): slot for slot in slots}
    if tile_n in slot_by_tile:
        return tactic, slot_by_tile[tile_n]
    # No slot for this tile: run the first slot's tile with its first valid tactic.
    tile_n = int(slots[0].tile_n)
    for candidate in runner.get_valid_tactics(inputs, None):
        if candidate != -1 and int(candidate[0]) == tile_n:
            return list(candidate), slots[0]
    raise ValueError(f"No Prims-TS tactic runs routing_metadata tile_n={tile_n}")


def _fp4_moe_setup(
    partitioned: bool,
    *,
    routing_logits: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: Union[torch.Tensor, Sequence[torch.Tensor]],
    gemm1_weights_scale: Union[torch.Tensor, Sequence[torch.Tensor]],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: Union[torch.Tensor, Sequence[torch.Tensor]],
    gemm2_weights_scale: Union[torch.Tensor, Sequence[torch.Tensor]],
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
    routing_method_type: int,
    weight_layout: int,
    do_finalize: bool,
    enable_pdl: Optional[bool],
    activation_type: int,
    per_token_scale: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    tune_max_num_tokens: int,
    norm_topk_prob: bool,
    routing_replay_out: Optional[torch.Tensor],
    routing_metadata=None,
    gemm1_bias: Optional[torch.Tensor] = None,
    gemm2_bias: Optional[torch.Tensor] = None,
    routing_input_mode: int = RoutingInputMode.FromLogits,
    num_fused_shared_experts: Optional[int] = None,
    partition_resources: Optional[MoePartitionResources] = None,
    partition_strategy: Optional[PartitionStrategy] = None,
):
    """The runner, inputs, tuning config and kwargs of the FP4 call (``partitioned``: the
    partitioned call), with the inputs the key of its tactic, and ``(ok, reason)`` of its
    support check: ``reason`` is the call's error message when not ok."""
    if enable_pdl is None:
        enable_pdl = device_support_pdl(hidden_states.device)
    if partitioned and per_token_scale is not None:
        raise NotImplementedError(
            "Partitioned FP4 MoE does not support per-token FC1-output requantization"
        )
    if hidden_states.dtype == torch.uint8:
        runners = (PrimsTsNvfp4MoERunner, PrimsTsPartitionedNvfp4MoERunner)
        support_fn = is_prims_ts_nvfp4_supported
        mode_name = "NVFP4xNVFP4"
        hidden_size = hidden_states.shape[-1] * 2
        scale_k_group_size = 16
        if hidden_states_scale is None:
            raise ValueError("hidden_states_scale is required for NVFP4 activations")
        if hidden_states_scale.dtype != torch.float8_e4m3fn:
            raise ValueError("hidden_states_scale must be float8_e4m3fn for NVFP4")
    elif hidden_states.dtype == torch.float8_e4m3fn:
        runners = (PrimsTsMxfp4Mxfp8MoERunner, PrimsTsPartitionedMxfp4Mxfp8MoERunner)
        support_fn = is_prims_ts_mxfp4_mxfp8_supported
        mode_name = "MXFP4xMXFP8"
        hidden_size = hidden_states.shape[-1]
        scale_k_group_size = 32
        if hidden_states_scale is None:
            raise ValueError("hidden_states_scale is required for MXFP8 activations")
    elif hidden_states.dtype == torch.bfloat16:
        runners = (PrimsTsMxfp4Bf16MoERunner, PrimsTsPartitionedMxfp4Bf16MoERunner)
        support_fn = is_prims_ts_mxfp4_bf16_supported
        mode_name = "MXFP4xBF16"
        hidden_size = hidden_states.shape[-1]
        scale_k_group_size = 32
        if hidden_states_scale is not None:
            raise ValueError("hidden_states_scale must be None for BF16 activations")
    else:
        raise ValueError(
            "Partitioned FP4 MoE supports NVFP4, MXFP8, or BF16 activations"
            if partitioned
            else "Prims-TS FP4 path supports packed NVFP4 uint8, MXFP8 float8_e4m3fn, "
            "or BF16 activations"
        )
    runner_extras: dict = {}
    if partitioned:
        if partition_resources is None or partition_strategy is None:
            raise ValueError(
                "Partitioned FP4 MoE requires partition_resources and partition_strategy"
            )
        require_partition_pair("gemm1_weights", gemm1_weights)
        require_partition_pair("gemm1_weights_scale", gemm1_weights_scale)
        require_partition_pair("gemm2_weights", gemm2_weights)
        require_partition_pair("gemm2_weights_scale", gemm2_weights_scale)
        validate_partition_arch(gemm1_weights[0])
        routing_input_mode = (
            RoutingInputMode.FromLogits
            if routing_logits is not None
            else (
                RoutingInputMode.UnpackedPrecomputed
                if topk_weights is not None
                else RoutingInputMode.PackedPrecomputed
            )
        )
        if routing_metadata is not None:
            _check_routing_metadata_args(
                routing_metadata=routing_metadata,
                hidden_states=hidden_states,
                routing_input_mode=routing_input_mode,
                routing_replay_out=routing_replay_out,
                num_fused_shared_experts=None,
                num_experts=num_experts,
                top_k=top_k,
                local_expert_offset=local_expert_offset,
                local_num_experts=local_num_experts,
            )
        fc1_layout, fc2_layout = prepared_weight_layouts(
            weight_dtype=torch.uint8,
            weight_layout=weight_layout,
            use_shuffled_weight=True,
            scale_k_group_size=scale_k_group_size,
            scale_dtype=torch.float8_e4m3fn,
            fc1_block_major_k_bytes=prepared_block_major_k_bytes(
                gemm1_weights[0], weight_layout
            ),
            fc2_block_major_k_bytes=prepared_block_major_k_bytes(
                gemm2_weights[0], weight_layout
            ),
        )
        runner_extras = dict(
            partition_strategy=partition_strategy,
            partition_fc1_layout=fc1_layout,
            partition_fc2_layout=fc2_layout,
            partition_resources=partition_resources,
        )
    routing_logits, topk_ids, expert_weights = _resolve_routing_inputs(
        routing_input_mode=routing_input_mode,
        routing_logits=routing_logits,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
    )
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
    runner = runners[partitioned](
        _get_moe_op(),
        top_k=top_k,
        num_local_experts=local_num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation_type=activation_type,
        use_shuffled_weight=True,
        weight_layout=weight_layout,
        use_per_token_scaling=per_token_scale is not None,
        num_experts=num_experts,
        **runner_extras,
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
    common_kwargs = dict(
        routing_bias=routing_bias,
        gemm1_weights=gemm1_weights[0] if partitioned else gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale[0]
        if partitioned
        else gemm1_weights_scale,
        gemm1_bias=gemm1_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        gemm2_weights=gemm2_weights[0] if partitioned else gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale[0]
        if partitioned
        else gemm2_weights_scale,
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
        num_fused_shared_experts=num_fused_shared_experts,
    )
    if partitioned:
        common_kwargs.update(
            partition_resources=partition_resources,
            partition_fc1_weights=gemm1_weights,
            partition_fc2_weights=gemm2_weights,
            partition_fc1_scales=gemm1_weights_scale,
            partition_fc2_scales=gemm2_weights_scale,
        )
    runner.set_cache_key_static_extras(**common_kwargs)
    tuning_config = runner._make_tuning_config(
        moe_inputs,
        tune_max_num_tokens=tune_max_num_tokens,
        routing_input_mode=RoutingInputMode(routing_input_mode),
        use_cuda_graph=True,
        use_cold_l2_cache=not partitioned,
    )
    ok, reason = support_fn(runner, moe_inputs, [-1, -1], **common_kwargs)
    kernel = f"{'partitioned ' if partitioned else ''}Prims-TS {mode_name} kernel"
    reason = f"Config not supported by {kernel} ({reason})"
    return runner, moe_inputs, tuning_config, common_kwargs, ok, reason


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
    num_fused_shared_experts: Optional[int] = None,
    routing_metadata: Optional[
        Union["TrtllmMoERoutingMetadataSlot", "TrtllmMoERoutingMetadata"]
    ] = None,
) -> List[torch.Tensor]:
    r"""FP4 block-scaled MoE using the Prims-TS backend on SM100/SM103/SM107.

    Same arguments and return value as
    :func:`~flashinfer.fused_moe.trtllm_fp4_block_scale_moe`.

    Parameters
    ----------
    routing_logits : Optional[torch.Tensor]
        ``[seq_len, num_experts]`` routing logits, or ``None`` when using
        precomputed routing.
    routing_bias : Optional[torch.Tensor]
        Optional ``[num_experts]`` routing bias.
    hidden_states : torch.Tensor
        Activations (NVFP4 packed ``uint8``, MXFP8, or BF16 depending on mode).
    hidden_states_scale : torch.Tensor
        Block scales for ``hidden_states`` when required by the quant mode.
    gemm1_weights : torch.Tensor
        Packed FP4 FC1 weights.
    gemm1_weights_scale : torch.Tensor
        FC1 block scales.
    gemm1_bias : Optional[torch.Tensor]
        Optional FC1 bias.
    gemm1_alpha : Optional[torch.Tensor]
        Optional per-expert SwiGLU / SiTU alpha.
    gemm1_beta : Optional[torch.Tensor]
        Optional per-expert SwiGLU / SiTU beta.
    gemm1_clamp_limit : Optional[torch.Tensor]
        Optional per-expert clamp limit.
    gemm2_weights : torch.Tensor
        Packed FP4 FC2 weights.
    gemm2_weights_scale : torch.Tensor
        FC2 block scales.
    gemm2_bias : Optional[torch.Tensor]
        Optional FC2 bias.
    output1_scale_scalar : torch.Tensor
        Per-expert FC1 output scale.
    output1_scale_gate_scalar : torch.Tensor
        Per-expert FC1 gate scale.
    output2_scale_scalar : torch.Tensor
        Per-expert FC2 output scale.
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
    weight_layout : int
        Weight layout enum value (default ``MajorK``).
    do_finalize : bool
        If ``True``, return the finalized MoE output.
    enable_pdl : Optional[bool]
        Enable Programmatic Dependent Launch when supported.
    activation_type : int
        Activation enum value (default Swiglu).
    per_token_scale : Optional[torch.Tensor]
        Optional per-token scales.
    output : Optional[torch.Tensor]
        Optional in-place output tensor.
    tune_max_num_tokens : int
        Autotune token-bucket upper bound (default ``8192``).
    norm_topk_prob : bool
        Normalize top-k routing probabilities.
    routing_replay_out : Optional[torch.Tensor]
        Optional buffer that captures selected expert IDs.
    routing_input_mode : int
        Internal routing input mode (logits vs precomputed).
    topk_ids : Optional[torch.Tensor]
        Precomputed expert indices when not routing from logits.
    topk_weights : Optional[torch.Tensor]
        Precomputed routing weights when not routing from logits.
    num_fused_shared_experts : Optional[int]
        Number of fused shared experts (default ``None`` / ``0``).
    routing_metadata : Optional[Union[TrtllmMoERoutingMetadataSlot, TrtllmMoERoutingMetadata]]
        Routing metadata already built from ``topk_ids`` by
        :func:`~flashinfer.fused_moe.populate_trtllm_moe_routing_metadata_`. When given,
        the call skips the routing kernel and launches only the expert GEMMs, so the
        routing can run earlier, for example on another stream. A slot runs its tile.
        A multi-tile collection runs the slot of the tuned tile, or its first slot's tile
        when the tuned tile has none. The tuned config is used only when its tile has a
        slot, otherwise that tile's first valid config runs. Autotuning profiles the call
        with routing included. NVFP4 activations and precomputed routing only; not
        supported with ``routing_replay_out`` or fused shared experts (``ValueError``).
        With ``do_finalize=False`` the returned routing outputs share the slot's storage:
        consume them before the slot is repopulated, or copy them to keep them.
        Default ``None`` (route inside the call).

    Returns
    -------
    List[torch.Tensor]
        Same return contract as
        :func:`~flashinfer.fused_moe.trtllm_fp4_block_scale_moe`.
    """
    if num_fused_shared_experts:
        from flashinfer.fused_moe.core import _validate_fused_shared_experts

        _validate_fused_shared_experts(
            num_fused_shared_experts,
            local_expert_offset,
            local_num_experts,
            num_experts,
            routing_method_type,
        )

    if routing_metadata is not None:
        _check_routing_metadata_args(
            routing_metadata=routing_metadata,
            hidden_states=hidden_states,
            routing_input_mode=routing_input_mode,
            routing_replay_out=routing_replay_out,
            num_fused_shared_experts=num_fused_shared_experts,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=local_expert_offset,
            local_num_experts=local_num_experts,
        )
    runner, moe_inputs, tuning_config, common_kwargs, ok, reason = _fp4_moe_setup(
        False,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
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
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
        gemm1_bias=gemm1_bias,
        gemm2_bias=gemm2_bias,
        routing_input_mode=routing_input_mode,
        num_fused_shared_experts=num_fused_shared_experts,
    )
    if not ok:
        raise RuntimeError(reason)

    _, tactic = AutoTuner.get().choose_one(
        "flashinfer::prims_ts_fp4_block_scale_moe",
        [runner],
        tuning_config,
        moe_inputs.to_list(),
        **common_kwargs,
    )
    if routing_metadata is not None:
        tactic, routing_metadata = _routing_metadata_tactic(
            runner, moe_inputs, tactic, routing_metadata, common_kwargs
        )

    intermediate_output = runner.forward(
        moe_inputs.to_list(),
        tactic=tactic,
        routing_metadata=routing_metadata,
        **common_kwargs,
    )
    return unpack_trtllm_moe_output(
        intermediate_output, moe_inputs.output, do_finalize, None
    )


@flashinfer_api(trace=prims_ts_fp4_block_scale_moe_partitioned_trace_dispatch)
def prims_ts_fp4_block_scale_moe_partitioned(
    routing_logits: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    topk_ids: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: Sequence[torch.Tensor],
    gemm1_weights_scale: Sequence[torch.Tensor],
    gemm2_weights: Sequence[torch.Tensor],
    gemm2_weights_scale: Sequence[torch.Tensor],
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
    partition_resources: MoePartitionResources,
    partition_strategy: PartitionStrategy,
    routed_scaling_factor: Optional[float] = None,
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
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    routing_metadata: Optional[
        Union["TrtllmMoERoutingMetadataSlot", "TrtllmMoERoutingMetadata"]
    ] = None,
) -> List[torch.Tensor]:
    r"""Run two-way FP4 Prims-TS MoE with caller-owned shards and streams.

    Same computation as
    :func:`~flashinfer.fused_moe.prims_ts_fp4_block_scale_moe`, split across
    two green-context partitions on SM107 (Rubin) only.

    Parameters
    ----------
    routing_logits : Optional[torch.Tensor]
        ``[seq_len, num_experts]`` routing logits, or ``None`` when using
        precomputed routing.
    routing_bias : Optional[torch.Tensor]
        Optional ``[num_experts]`` routing bias.
    topk_ids : Optional[torch.Tensor]
        ``int32`` precomputed routing when ``routing_logits`` is ``None``:
        packed ``(expert_id, weight)`` values when ``topk_weights`` is
        ``None``, otherwise unpacked expert indices.
    topk_weights : Optional[torch.Tensor]
        Precomputed routing weights for unpacked ``topk_ids``.
    hidden_states : torch.Tensor
        Activations (NVFP4 packed ``uint8``, MXFP8, or BF16 depending on mode).
    hidden_states_scale : Optional[torch.Tensor]
        Block scales for ``hidden_states``; ``None`` for BF16 activations.
    gemm1_weights : Sequence[torch.Tensor]
        Two packed FP4 FC1 shards, one per partition, sharded with
        ``flashinfer.prims_ts.moe.shard_prepared_moe_weights``.
    gemm1_weights_scale : Sequence[torch.Tensor]
        Two FC1 block-scale shards.
    gemm2_weights : Sequence[torch.Tensor]
        Two packed FP4 FC2 shards.
    gemm2_weights_scale : Sequence[torch.Tensor]
        Two FC2 block-scale shards.
    output1_scale_scalar : torch.Tensor
        Per-expert FC1 output scale.
    output1_scale_gate_scalar : torch.Tensor
        Per-expert FC1 gate scale.
    output2_scale_scalar : torch.Tensor
        Per-expert FC2 output scale.
    num_experts : int
        Total number of experts.
    top_k : int
        Experts selected per token.
    n_group : Optional[int]
        Number of expert groups.
    topk_group : Optional[int]
        Groups considered for top-k routing.
    intermediate_size : int
        Unsharded intermediate (FFN) width.
    local_expert_offset : int
        Global offset of the first local expert.
    local_num_experts : int
        Number of experts resident on this device.
    partition_resources : MoePartitionResources
        Caller-owned green-context streams and reusable fork/join events.
        Construct before CUDA graph capture and keep alive for every launch
        or replay that uses it.
    partition_strategy : PartitionStrategy
        ``"tp_n_tp_n"`` splits FC2 output rows; ``"tp_n_tp_k"`` splits FC2
        K. FC1 rows are split in both. Must match the weight sharding.
    routed_scaling_factor : Optional[float]
        Optional routing scale.
    routing_method_type : int
        Routing method selector (default ``0``).
    weight_layout : int
        Weight layout enum value (default ``MajorK``).
    do_finalize : bool
        If ``True``, return the finalized MoE output.
    enable_pdl : Optional[bool]
        Enable Programmatic Dependent Launch when supported.
    activation_type : int
        Activation enum value (default Swiglu).
    per_token_scale : Optional[torch.Tensor]
        Must be ``None``; per-token FC1-output requantization is not
        supported by partitioned execution.
    output : Optional[torch.Tensor]
        Optional in-place output tensor.
    tune_max_num_tokens : int
        Autotune token-bucket upper bound (default ``8192``).
    norm_topk_prob : bool
        Normalize top-k routing probabilities.
    routing_replay_out : Optional[torch.Tensor]
        Optional buffer that captures selected expert IDs.
    gemm1_alpha : Optional[torch.Tensor]
        Optional per-expert SwiGLU / SiTU alpha.
    gemm1_beta : Optional[torch.Tensor]
        Optional per-expert SwiGLU / SiTU beta.
    gemm1_clamp_limit : Optional[torch.Tensor]
        Optional per-expert clamp limit.
    routing_metadata : Optional[Union[TrtllmMoERoutingMetadataSlot, TrtllmMoERoutingMetadata]]
        Prebuilt routing metadata; see
        :func:`~flashinfer.fused_moe.prims_ts_fp4_block_scale_moe`. Both partitions
        read the same metadata. Default ``None`` (route inside the call).

    Returns
    -------
    List[torch.Tensor]
        Same return contract as
        :func:`~flashinfer.fused_moe.prims_ts_fp4_block_scale_moe`.
    """

    runner, moe_inputs, tuning_config, common_kwargs, ok, reason = _fp4_moe_setup(
        True,
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
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
        norm_topk_prob=norm_topk_prob,
        routing_replay_out=routing_replay_out,
        routing_metadata=routing_metadata,
        partition_resources=partition_resources,
        partition_strategy=partition_strategy,
    )
    if not ok:
        raise RuntimeError(reason)
    _, tactic = AutoTuner.get().choose_one(
        "flashinfer::prims_ts_fp4_block_scale_moe_partitioned",
        [runner],
        tuning_config,
        moe_inputs.to_list(),
        **common_kwargs,
    )
    if routing_metadata is not None:
        tactic, routing_metadata = _routing_metadata_tactic(
            runner, moe_inputs, tactic, routing_metadata, common_kwargs
        )
    intermediate_output = runner.forward(
        moe_inputs.to_list(),
        tactic=tactic,
        routing_metadata=routing_metadata,
        **common_kwargs,
    )
    return unpack_trtllm_moe_output(
        intermediate_output, moe_inputs.output, do_finalize, None
    )


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
    out = (
        output
        if output is not None
        else hidden_states.new_empty(
            hidden_states.shape[0], hidden_size, dtype=torch.bfloat16
        )
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
    routing_metadata: Optional[
        Union["TrtllmMoERoutingMetadataSlot", "TrtllmMoERoutingMetadata"]
    ] = None,
) -> List[torch.Tensor]:
    r"""FP4 block-scaled MoE with precomputed routing (Prims-TS / SM100).

    Same arguments and return value as
    :func:`~flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`.

    Parameters
    ----------
    topk_ids : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        Packed ``(expert_id, weight)`` tensor or unpacked
        ``(topk_ids, topk_weights)`` pair.
    routing_bias : Optional[torch.Tensor]
        Optional ``[num_experts]`` routing bias.
    hidden_states : torch.Tensor
        Activations (NVFP4 packed ``uint8``, MXFP8, or BF16 depending on mode).
    hidden_states_scale : Optional[torch.Tensor]
        Block scales for ``hidden_states`` when required by the quant mode.
    gemm1_weights : torch.Tensor
        Packed FP4 FC1 weights.
    gemm1_weights_scale : torch.Tensor
        FC1 block scales.
    gemm1_bias : Optional[torch.Tensor]
        Optional FC1 bias.
    gemm1_alpha : Optional[torch.Tensor]
        Optional per-expert SwiGLU / SiTU alpha.
    gemm1_beta : Optional[torch.Tensor]
        Optional per-expert SwiGLU / SiTU beta.
    gemm1_clamp_limit : Optional[torch.Tensor]
        Optional per-expert clamp limit.
    gemm2_weights : torch.Tensor
        Packed FP4 FC2 weights.
    gemm2_weights_scale : torch.Tensor
        FC2 block scales.
    gemm2_bias : Optional[torch.Tensor]
        Optional FC2 bias.
    output1_scale_scalar : Optional[torch.Tensor]
        Per-expert FC1 output scale.
    output1_scale_gate_scalar : Optional[torch.Tensor]
        Per-expert FC1 gate scale.
    output2_scale_scalar : Optional[torch.Tensor]
        Per-expert FC2 output scale.
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
    weight_layout : int
        Weight layout enum value (default ``MajorK``).
    do_finalize : bool
        If ``True``, return the finalized MoE output.
    enable_pdl : Optional[bool]
        Enable Programmatic Dependent Launch when supported.
    activation_type : int
        Activation enum value (default Swiglu).
    per_token_scale : Optional[torch.Tensor]
        Optional per-token scales.
    output : Optional[torch.Tensor]
        Optional in-place output tensor.
    tune_max_num_tokens : int
        Autotune token-bucket upper bound (default ``8192``).
    routing_metadata : Optional[Union[TrtllmMoERoutingMetadataSlot, TrtllmMoERoutingMetadata]]
        Prebuilt routing metadata; see
        :func:`~flashinfer.fused_moe.prims_ts_fp4_block_scale_moe`.
        Default ``None`` (route inside the call).

    Returns
    -------
    List[torch.Tensor]
        Same return contract as
        :func:`~flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe`.
    """
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
        routing_metadata=routing_metadata,
    )


@flashinfer_api
def prims_ts_fp4_block_scale_moe_routing_tile_n(
    topk_ids: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    gemm1_weights: Union[torch.Tensor, Sequence[torch.Tensor]],
    gemm1_weights_scale: Union[torch.Tensor, Sequence[torch.Tensor]],
    num_experts: int,
    top_k: int,
    intermediate_size: int,
    local_num_experts: int,
    *,
    do_finalize: bool = True,
    activation_type: int = ActivationType.Swiglu.value,
    per_token_scale: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    output: Optional[torch.Tensor] = None,
    weight_layout: int = 0,
    enable_pdl: Optional[bool] = None,
    gemm1_alpha: Optional[torch.Tensor] = None,
    gemm1_beta: Optional[torch.Tensor] = None,
    gemm1_clamp_limit: Optional[torch.Tensor] = None,
    gemm2_weights: Optional[Sequence[torch.Tensor]] = None,
    partition_resources: Optional[MoePartitionResources] = None,
    partition_strategy: Optional[PartitionStrategy] = None,
) -> Optional[int]:
    """The tile_N :func:`prims_ts_fp4_block_scale_routed_moe` (no biases) runs with the same
    arguments, or with ``partition_strategy`` :func:`prims_ts_fp4_block_scale_moe_partitioned`
    (shards in ``gemm1_weights``, ``gemm1_weights_scale`` and ``gemm2_weights``): the
    autotuned tile if the tactic cache has one, else the default tile. ``None`` when the
    call does not support the config, so route inside the call instead.

    Same contract as :func:`~flashinfer.fused_moe.trtllm_fp4_block_scale_moe_routing_tile_n`:
    it only looks the cache up (inside a tuning context it returns the default tile), and the
    activation tensors and ``output`` may be ``meta`` tensors.
    """
    topk_ids, topk_weights = (
        topk_ids if isinstance(topk_ids, tuple) else (topk_ids, None)
    )
    shards = partition_strategy is not None
    if output is None:
        packed = 2 if hidden_states.dtype == torch.uint8 else 1
        hidden_size = hidden_states.shape[-1] * packed
        output = alloc_trtllm_moe_output(
            hidden_states.shape[0], hidden_size, do_finalize, torch.device("meta")
        )
    # Shape-only stand-ins on the call's device, which keys the tactics.
    hidden_states, hidden_states_scale, per_token_scale, output = (
        t
        if t is None or not t.is_meta
        else torch.empty((), dtype=t.dtype, device=topk_ids.device).expand(t.shape)
        for t in (hidden_states, hidden_states_scale, per_token_scale, output)
    )
    # Checked only for presence before the tactic is chosen; nothing launches.
    present = torch.empty(0, device=topk_ids.device)
    # The partitioned call raises NotImplementedError for these configs.
    if shards and (
        per_token_scale is not None
        or not is_gated_activation(activation_type)
        or get_compute_capability(topk_ids.device) != (10, 7)
    ):
        return None
    runner, moe_inputs, tuning_config, kwargs, ok, _ = _fp4_moe_setup(
        shards,
        routing_logits=None,
        routing_bias=None,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=(present, present) if shards else present,
        output1_scale_scalar=present,
        output1_scale_gate_scalar=present,
        output2_scale_scalar=present,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=local_num_experts,
        routed_scaling_factor=None,
        routing_method_type=0,
        weight_layout=weight_layout,
        do_finalize=do_finalize,
        enable_pdl=enable_pdl,
        activation_type=activation_type,
        per_token_scale=per_token_scale,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        norm_topk_prob=True,
        routing_replay_out=None,
        routing_input_mode=RoutingInputMode.PackedPrecomputed
        if topk_weights is None
        else RoutingInputMode.UnpackedPrecomputed,
        partition_resources=partition_resources,
        partition_strategy=partition_strategy,
    )
    if not ok:
        return None
    tactic = -1
    tuner = AutoTuner.get()
    # Outside tuning mode choose_one is the call's own cache lookup; it never profiles.
    if not tuner.is_tuning_mode:
        _, tactic = tuner.choose_one(
            "flashinfer::prims_ts_fp4_block_scale_moe_partitioned"
            if shards
            else "flashinfer::prims_ts_fp4_block_scale_moe",
            [runner],
            tuning_config,
            moe_inputs.to_list(),
            **kwargs,
        )
    if tactic == -1 and shards:
        try:  # forward() runs the first tactic the partition geometry allows.
            tactic = runner.get_valid_tactics(moe_inputs.to_list(), None)[0]
        except RuntimeError:  # No tactic fits this geometry: the call raises.
            return None
    return _tactic_tile_n(runner, moe_inputs, tactic, kwargs)
