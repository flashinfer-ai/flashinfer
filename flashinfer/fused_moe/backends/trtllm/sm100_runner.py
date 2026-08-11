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

"""TRT-LLM cubin MoE TunableRunner for SM100."""

from __future__ import annotations

from typing import Callable, List, Optional

import torch

from flashinfer.autotuner import OptimizationProfile, TunableRunner, TuningConfig
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs, RoutingInputMode
from flashinfer.fused_moe.shared.tuning import make_moe_tuning_config
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    WeightLayout,
    trtllm_gen_dtype_has_scale,
)


def create_trtllm_moe_runner_class(
    moe_op,
    *,
    logger,
    topk_ids_initializer_factory: Callable,
):
    """Build the cubin MoERunner class bound to a loaded JIT moe_op module."""

    class MoERunner(TunableRunner):
        # Cache valid tactics to reduce the overhead of re-querying the kernel.
        valid_tactics_dict = dict()

        def __init__(
            self,
            top_k: int,
            num_local_experts: int,
            dtype_act: DtypeTrtllmGen,
            dtype_weights: DtypeTrtllmGen,
            fp8_quantization_type: Fp8QuantizationType,
            hidden_size: int,
            intermediate_size: int,
            activation_type: int = ActivationType.Swiglu.value,
            use_shuffled_weight: bool = False,
            weight_layout: int = WeightLayout.MajorK,
            use_packed_weights: bool = False,
            use_per_token_scaling: bool = False,
            num_experts: Optional[int] = None,
            num_fused_shared_experts: int = 0,
        ):
            self.num_local_experts = num_local_experts
            self.top_k = top_k
            self.num_fused_shared_experts = num_fused_shared_experts or 0
            self.dtype_act = dtype_act
            self.dtype_weights = dtype_weights
            self.fp8_quantization_type = fp8_quantization_type
            self.hidden_size = hidden_size
            self.intermediate_size = intermediate_size
            self.activation_type = ActivationType(activation_type)
            self.use_shuffled_weight = use_shuffled_weight
            self.weight_layout = WeightLayout(weight_layout)
            self.use_packed_weights = use_packed_weights
            self.use_per_token_scaling = use_per_token_scaling
            self.num_experts = (
                num_experts if num_experts is not None else num_local_experts
            )

        def _make_tuning_config(
            self,
            moe_inputs: MoeRunnerInputs,
            tune_max_num_tokens: int = 8192,
            routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed,
            **kwargs,
        ) -> TuningConfig:
            return make_moe_tuning_config(
                moe_inputs,
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                fp8_quantization_type=self.fp8_quantization_type,
                init_packed_topk_ids=topk_ids_initializer_factory(
                    self.num_experts,
                    packed=(
                        routing_input_mode != RoutingInputMode.UnpackedPrecomputed
                    ),
                ),
                tune_max_num_tokens=tune_max_num_tokens,
                **kwargs,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            moe_inputs = MoeRunnerInputs.from_list(inputs)
            num_tokens = moe_inputs.hidden_states.shape[0]
            has_gemm1_lora_delta = moe_inputs.gemm1_lora_delta is not None
            nfse = self.num_fused_shared_experts
            instance_key = (
                self.dtype_act,
                self.dtype_weights,
                self.fp8_quantization_type,
                self.top_k + nfse,
                self.hidden_size,
                self.intermediate_size,
                self.num_local_experts + nfse,
                self.activation_type,
                self.use_shuffled_weight,
                self.weight_layout,
                self.use_per_token_scaling,
                num_tokens,
                has_gemm1_lora_delta,
            )
            if instance_key not in MoERunner.valid_tactics_dict:
                try:
                    valid_tactics = moe_op.trtllm_get_valid_moe_configs(*instance_key)
                except Exception as e:
                    logger.debug(
                        f"[Autotuner]: Failed to get valid tactics for {instance_key}. "
                        f"Error occurred: {e}"
                    )
                    return []
                MoERunner.valid_tactics_dict[instance_key] = valid_tactics
            return MoERunner.valid_tactics_dict[instance_key]

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ):
            moe_inputs = MoeRunnerInputs.from_list(inputs)
            output = moe_inputs.output
            routing_logits = moe_inputs.routing_logits
            topk_ids = moe_inputs.topk_ids
            expert_weights = moe_inputs.expert_weights
            topk_weights = expert_weights
            hidden_states = moe_inputs.hidden_states
            hidden_states_scale = (
                moe_inputs.hidden_states_scale
                if trtllm_gen_dtype_has_scale(self.dtype_act)
                else None
            )

            num_tokens = hidden_states.shape[0]
            assert output.shape[0] == num_tokens, (
                "output's first dimension must be batch size."
            )
            if routing_logits is not None:
                assert routing_logits.shape[0] == num_tokens, (
                    "routing_logits's first dimension must be batch size."
                )
            if topk_ids is not None and topk_ids.numel() > 0:
                assert topk_ids.shape[0] == num_tokens, (
                    "topk_ids's first dimension must be batch size."
                )
            if expert_weights is not None and expert_weights.numel() > 0:
                assert expert_weights.shape[0] == num_tokens, (
                    "expert_weights's first dimension must be batch size."
                )
            assert hidden_states.shape[0] == num_tokens, (
                "hidden_states's first dimension must be batch size."
            )
            if hidden_states_scale is not None:
                assert hidden_states_scale.dim() == 2, (
                    "hidden_states_scale must be a 2D tensor"
                )
                if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                    assert hidden_states_scale.shape[1] == num_tokens, (
                        f"DeepSeekFp8 hidden_states_scale shape "
                        f"{tuple(hidden_states_scale.shape)} expects num_tokens={num_tokens} "
                        f"at dim 1"
                    )
                else:
                    assert hidden_states_scale.shape[0] == num_tokens, (
                        f"hidden_states_scale shape {tuple(hidden_states_scale.shape)} "
                        f"expects num_tokens={num_tokens} at dim 0"
                    )

            if self.dtype_weights == DtypeTrtllmGen.Bfloat16:
                moe_op.trtllm_bf16_moe(
                    kwargs.get("routing_input_mode", RoutingInputMode.FromLogits),
                    routing_logits,
                    kwargs["routing_bias"],
                    topk_ids,
                    expert_weights,
                    hidden_states,
                    kwargs["gemm1_weights"],
                    kwargs["gemm2_weights"],
                    moe_inputs.gemm1_lora_delta,
                    kwargs.get("gemm1_alpha"),
                    kwargs.get("gemm1_beta"),
                    kwargs.get("gemm1_clamp_limit"),
                    output,
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["use_shuffled_weight"],
                    kwargs["weight_layout"],
                    kwargs["do_finalize"],
                    kwargs["enable_pdl"],
                    [-1, -1] if tactic == -1 else tactic,
                    self.activation_type,
                    kwargs.get("norm_topk_prob", True),
                    kwargs.get("routing_replay_out"),
                )
            elif (
                self.dtype_act == DtypeTrtllmGen.E4m3
                and self.dtype_weights == DtypeTrtllmGen.E4m3
            ) or (
                self.dtype_act == DtypeTrtllmGen.MxE4m3
                and self.dtype_weights == DtypeTrtllmGen.MxE4m3
            ):
                if (
                    self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8
                    or self.fp8_quantization_type == Fp8QuantizationType.MxFp8
                ):
                    current_num_tokens = hidden_states.shape[0]
                    current_hidden_size = hidden_states.shape[1]
                    if self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8:
                        current_hidden_states_scale = torch.full(
                            (current_hidden_size // 128, current_num_tokens),
                            2.0,
                            dtype=torch.float,
                            device=hidden_states.device,
                        )
                    elif self.fp8_quantization_type == Fp8QuantizationType.MxFp8:
                        current_hidden_states_scale = hidden_states_scale
                    else:
                        raise ValueError(
                            f"Unsupported FP8 quantization type: "
                            f"{self.fp8_quantization_type}"
                        )

                    moe_op.trtllm_fp8_block_scale_moe(
                        kwargs.get("routing_input_mode", RoutingInputMode.FromLogits),
                        routing_logits,
                        topk_ids,
                        topk_weights,
                        kwargs["routing_bias"],
                        hidden_states,
                        current_hidden_states_scale,
                        kwargs["gemm1_weights"],
                        kwargs["gemm1_weights_scale"],
                        moe_inputs.gemm1_lora_delta,
                        kwargs.get("gemm1_bias"),
                        kwargs.get("gemm1_alpha"),
                        kwargs.get("gemm1_beta"),
                        kwargs.get("gemm1_clamp_limit"),
                        kwargs["gemm2_weights"],
                        kwargs["gemm2_weights_scale"],
                        kwargs.get("gemm2_bias"),
                        output,
                        kwargs["num_experts"],
                        self.top_k,
                        kwargs.get("num_fused_shared_experts", 0),
                        kwargs["n_group"],
                        kwargs["topk_group"],
                        self.intermediate_size,
                        kwargs["local_expert_offset"],
                        self.num_local_experts,
                        kwargs["routed_scaling_factor"],
                        kwargs["routing_method_type"],
                        kwargs["use_shuffled_weight"],
                        kwargs["weight_layout"],
                        kwargs["do_finalize"],
                        kwargs["enable_pdl"],
                        [-1, -1] if tactic == -1 else tactic,
                        self.fp8_quantization_type,
                        self.activation_type,
                        kwargs.get("norm_topk_prob", True),
                        kwargs.get("routing_replay_out"),
                    )
                else:
                    moe_op.trtllm_fp8_per_tensor_scale_moe(
                        routing_logits,
                        kwargs["routing_bias"],
                        hidden_states,
                        kwargs["gemm1_weights"],
                        kwargs["output1_scales_scalar"],
                        kwargs["output1_scales_gate_scalar"],
                        kwargs["gemm2_weights"],
                        kwargs["output2_scales_scalar"],
                        output,
                        kwargs["num_experts"],
                        self.top_k,
                        kwargs["n_group"],
                        kwargs["topk_group"],
                        self.intermediate_size,
                        kwargs["local_expert_offset"],
                        self.num_local_experts,
                        kwargs["routed_scaling_factor"],
                        kwargs["use_routing_scales_on_input"],
                        kwargs["routing_method_type"],
                        kwargs["do_finalize"],
                        kwargs["enable_pdl"],
                        [-1, -1] if tactic == -1 else tactic,
                        self.activation_type,
                        kwargs.get("norm_topk_prob", True),
                        kwargs.get("routing_replay_out"),
                    )
            elif (
                self.dtype_act == DtypeTrtllmGen.Bfloat16
                and self.dtype_weights == DtypeTrtllmGen.MxInt4
            ):
                moe_op.trtllm_mxint4_block_scale_moe(
                    routing_logits,
                    kwargs["routing_bias"],
                    topk_ids,
                    expert_weights,
                    hidden_states,
                    kwargs["gemm1_weights"],
                    kwargs["gemm1_weights_scale"],
                    kwargs["gemm1_alpha"],
                    kwargs["gemm1_beta"],
                    kwargs["gemm1_clamp_limit"],
                    moe_inputs.gemm1_lora_delta,
                    kwargs["gemm2_weights"],
                    kwargs["gemm2_weights_scale"],
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["do_finalize"],
                    kwargs["enable_pdl"],
                    output,
                    [-1, -1] if tactic == -1 else tactic,
                    kwargs.get("norm_topk_prob", True),
                    kwargs.get("routing_replay_out"),
                )
            else:
                moe_op.trtllm_fp4_block_scale_moe(
                    kwargs.get("routing_input_mode", RoutingInputMode.FromLogits),
                    routing_logits,
                    topk_ids,
                    topk_weights,
                    kwargs["routing_bias"],
                    hidden_states,
                    hidden_states_scale,
                    kwargs["gemm1_weights"],
                    kwargs["gemm1_weights_scale"],
                    kwargs["gemm1_bias"],
                    moe_inputs.gemm1_lora_delta,
                    kwargs["gemm1_alpha"],
                    kwargs["gemm1_beta"],
                    kwargs["gemm1_clamp_limit"],
                    kwargs["gemm2_weights"],
                    kwargs["gemm2_weights_scale"],
                    kwargs["gemm2_bias"],
                    kwargs["output1_scale_scalar"],
                    kwargs["output1_scale_gate_scalar"],
                    kwargs["output2_scale_scalar"],
                    kwargs["per_token_scale"],
                    kwargs["num_experts"],
                    self.top_k,
                    kwargs.get(
                        "num_fused_shared_experts", self.num_fused_shared_experts
                    ),
                    kwargs["n_group"],
                    kwargs["topk_group"],
                    self.intermediate_size,
                    kwargs["local_expert_offset"],
                    self.num_local_experts,
                    kwargs["routed_scaling_factor"],
                    kwargs["routing_method_type"],
                    kwargs["do_finalize"],
                    kwargs["enable_pdl"],
                    self.activation_type,
                    output,
                    [-1, -1] if tactic == -1 else tactic,
                    kwargs.get("norm_topk_prob", True),
                    kwargs.get("routing_replay_out"),
                )

    return MoERunner
