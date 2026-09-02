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

from typing import Any, List, Optional

import torch

from flashinfer.autotuner import OptimizationProfile, TunableRunner, TuningConfig
from flashinfer.fused_moe.shared.inputs import (
    MoeRunnerInputs,
    RoutingInputMode,
    unpack_trtllm_moe_output,
)
from flashinfer.fused_moe.shared.tuning import (
    make_moe_tuning_config,
    make_repeating_tensor_initializer,
    moe_topk_ids_init,
)
from flashinfer.jit.core import logger
from flashinfer.tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    WeightLayout,
    trtllm_gen_dtype_has_scale,
)


class MoERunner(TunableRunner):
    """Tactic-aware runner for the TRT-LLM cubin MoE kernels.

    ``moe_op`` is a loaded JIT module. The Rubin and non-Rubin cubins are
    separate modules that can report different tactics for the same problem
    shape, so the module identity is part of the tactic cache key.
    """

    # Cache valid tactics to reduce the overhead of re-querying the kernel.
    valid_tactics_dict = dict[Any, Any]()

    def __init__(
        self,
        moe_op,
        *,
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
        self.moe_op = moe_op
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
        self.num_experts = num_experts if num_experts is not None else num_local_experts
        # Runtime routing tensors and their initializer closures carry object
        # identity, so keep them in an attribute excluded by
        # TunableRunner.__hash__.  Otherwise logically identical runners built
        # by successive API calls cannot reuse an in-memory tuned tactic.
        self._topk_initializer_cache = None

    def _make_tuning_config(
        self,
        moe_inputs: MoeRunnerInputs,
        tune_max_num_tokens: int = 8192,
        routing_input_mode: RoutingInputMode = RoutingInputMode.PackedPrecomputed,
        **kwargs,
    ) -> TuningConfig:
        if moe_inputs.topk_ids is not None and moe_inputs.topk_ids.numel() > 0:
            if (
                self._topk_initializer_cache is None
                or self._topk_initializer_cache[0] is not moe_inputs.topk_ids
            ):
                self._topk_initializer_cache = (
                    moe_inputs.topk_ids,
                    make_repeating_tensor_initializer(
                        moe_inputs.topk_ids,
                        num_experts=self.num_experts,
                        packed=(
                            routing_input_mode
                            != RoutingInputMode.UnpackedPrecomputed
                        ),
                    ),
                )
            init_packed_topk_ids = self._topk_initializer_cache[1]
        else:
            init_packed_topk_ids = moe_topk_ids_init(
                self.num_experts,
                packed=(routing_input_mode != RoutingInputMode.UnpackedPrecomputed),
            )
        return make_moe_tuning_config(
            moe_inputs,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            fp8_quantization_type=self.fp8_quantization_type,
            init_packed_topk_ids=init_packed_topk_ids,
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
        query_key = (
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
        instance_key = (id(self.moe_op), query_key)
        if instance_key not in MoERunner.valid_tactics_dict:
            try:
                valid_tactics = self.moe_op.trtllm_get_valid_moe_configs(*query_key)
            except Exception as e:
                logger.debug(
                    f"[Autotuner]: Failed to get valid tactics for {query_key}. "
                    f"Error occurred: {e}"
                )
                return []
            MoERunner.valid_tactics_dict[instance_key] = valid_tactics
        return MoERunner.valid_tactics_dict[instance_key]

    def get_factorized_tactic_space(
        self,
        inputs: List[torch.Tensor],
    ):
        """Return C++-declared legal FC1/FC2 factors and tile-local anchors."""
        from flashinfer.fused_moe.da_tuner import (
            FactorizedTactic,
            FactorizedTacticSpace,
        )

        moe_inputs = MoeRunnerInputs.from_list(inputs)
        rows = self.moe_op.trtllm_get_valid_moe_factorizations(
            self.dtype_act,
            self.dtype_weights,
            self.fp8_quantization_type,
            self.top_k + self.num_fused_shared_experts,
            self.hidden_size,
            self.intermediate_size,
            self.num_local_experts + self.num_fused_shared_experts,
            self.activation_type,
            self.use_shuffled_weight,
            self.weight_layout,
            self.use_per_token_scaling,
            moe_inputs.hidden_states.shape[0],
            moe_inputs.gemm1_lora_delta is not None,
        )
        tactics = []
        anchors = {}
        for tile_n, config, fc1, fc2, is_anchor in rows:
            identity = (int(tile_n), int(config))
            tactics.append(
                FactorizedTactic(
                    tactic=identity,
                    tile_n=int(tile_n),
                    fc1=int(fc1),
                    fc2=int(fc2),
                )
            )
            if is_anchor:
                anchors[int(tile_n)] = identity
        return FactorizedTacticSpace(tactics, anchors)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
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
        # Plain E4m3 returns false from trtllm_gen_dtype_has_scale, but DeepSeek
        # block-FP8 still requires the real per-1x128-block activation scales.
        hidden_states_scale = (
            moe_inputs.hidden_states_scale
            if (
                trtllm_gen_dtype_has_scale(self.dtype_act)
                or self.fp8_quantization_type == Fp8QuantizationType.DeepSeekFp8
            )
            else None
        )
        da_routing_metadata = kwargs.get("da_routing_metadata", ())
        da_body_workspace = kwargs.get("da_body_workspace", ())
        prepare_da_body = do_preparation and bool(da_routing_metadata)

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
            result = self.moe_op.trtllm_bf16_moe(
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
                list(da_routing_metadata),
                list(da_body_workspace),
                prepare_da_body,
            )
            if prepare_da_body or da_routing_metadata:
                return list(result)
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
                result = self.moe_op.trtllm_fp8_block_scale_moe(
                    kwargs.get("routing_input_mode", RoutingInputMode.FromLogits),
                    routing_logits,
                    topk_ids,
                    topk_weights,
                    kwargs["routing_bias"],
                    hidden_states,
                    hidden_states_scale,
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
                    list(da_routing_metadata),
                    list(da_body_workspace),
                    prepare_da_body,
                )
            else:
                common_args = (
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
                    list(da_routing_metadata),
                    list(da_body_workspace),
                    prepare_da_body,
                )
                if routing_logits is None:
                    # FP8 per tensor scale, pre-computed routing.
                    result = self.moe_op.trtllm_fp8_per_tensor_scale_routed_moe(
                        kwargs.get(
                            "routing_input_mode", RoutingInputMode.PackedPrecomputed
                        ),
                        topk_ids,
                        topk_weights,
                        *common_args,
                    )
                else:
                    result = self.moe_op.trtllm_fp8_per_tensor_scale_moe(
                        routing_logits,
                        *common_args,
                    )
                    # The FromLogits ABI allocates and returns expert weights
                    # internally instead of filling the caller's buffer.
                    expert_weights = None
            if prepare_da_body or da_routing_metadata:
                return list(result)
        elif (
            self.dtype_act == DtypeTrtllmGen.Bfloat16
            and self.dtype_weights == DtypeTrtllmGen.MxInt4
        ):
            result = self.moe_op.trtllm_mxint4_block_scale_moe(
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
                list(da_routing_metadata),
                list(da_body_workspace),
                prepare_da_body,
            )
            if prepare_da_body or da_routing_metadata:
                return list(result)

        else:
            result = self.moe_op.trtllm_fp4_block_scale_moe(
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
                kwargs.get("num_fused_shared_experts", self.num_fused_shared_experts),
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
                list(da_routing_metadata),
                list(da_body_workspace),
                prepare_da_body,
            )
            if prepare_da_body or da_routing_metadata:
                return list(result)

        return unpack_trtllm_moe_output(
            result,
            output,
            kwargs["do_finalize"],
            moe_inputs.gemm1_lora_delta,
            expert_weights,
        )
