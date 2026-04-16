"""Unified MoE runner adapters for the autotuned pre-routed NVFP4 path.

Each runner wraps one backend and translates (MoEActivationPack, MoEWeightPack)
into the backend's native calling convention.

MVP scope: NVFP4 only, two backends (CuteDSL, TRTLLM routed).
"""

from __future__ import annotations

from typing import Any, List

import torch

from ..autotuner import DynamicTensorSpec, TunableRunner, TuningConfig
from .api import MoEActivationPack, MoEConfig, MoEWeightPack
from .utils import get_last_power_of_2_num_tokens_buckets, last_positive_power_of_2


# ---------------------------------------------------------------------------
# CuteDSL NVFP4 runner — delegates to the existing CuteDslFusedMoENvfp4Runner
# ---------------------------------------------------------------------------


class CuteDslNvfp4Runner(TunableRunner):
    """Wraps CuteDslFusedMoENvfp4Runner, translating Pack inputs into its
    List[Tensor] convention."""

    backend_key = "cute_dsl_nvfp4"

    def __init__(self, config: MoEConfig, device: torch.device):
        from .cute_dsl.fused_moe import _cute_dsl_fused_moe_nvfp4_impl
        from .cute_dsl.tuner import CuteDslFusedMoENvfp4Runner

        experts = config.experts
        routing = config.routing
        num_local_experts = experts.local_num_experts or routing.num_experts

        self._inner = CuteDslFusedMoENvfp4Runner(
            forward_impl=_cute_dsl_fused_moe_nvfp4_impl,
            num_experts=routing.num_experts,
            top_k=routing.top_k,
            num_local_experts=num_local_experts,
            local_expert_offset=experts.local_expert_offset,
        )
        self.tuning_config = CuteDslFusedMoENvfp4Runner.tuning_config

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: Any) -> List[Any]:
        return self._inner.get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._inner.forward(inputs, tactic=tactic, **kwargs)

    @staticmethod
    def pack_inputs(
        act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        """Translate Packs → List[Tensor] expected by CuteDslFusedMoENvfp4Runner.

        Input order: x, x_sf, token_selected_experts, token_final_scales,
                     w1_weight, w1_weight_sf, w1_alpha, fc2_input_scale,
                     w2_weight, w2_weight_sf, w2_alpha
        """
        return [
            act.hidden_states_q,
            act.hidden_states_scale.unsqueeze(-1),  # CuteDSL expects [M, H//16, 1]
            act.selected_experts,
            act.final_scales,
            weights.w1_q,
            weights.w1_scale,
            weights.w1_alpha,
            weights.fc2_input_scale,
            weights.w2_q,
            weights.w2_scale,
            weights.w2_alpha,
        ]

    def __hash__(self):
        return hash(("cute_dsl_nvfp4", hash(self._inner)))


# ---------------------------------------------------------------------------
# TRTLLM FP4 routed runner — wraps the flat function
# ---------------------------------------------------------------------------


class TrtllmFp4RoutedRunner(TunableRunner):
    """Wraps moe_op.trtllm_fp4_block_scale_moe (routed path, routing_logits=None).

    Tactics are [gemm1_config, gemm2_config] int pairs enumerated by
    moe_op.trtllm_get_valid_moe_configs.  Mirrors existing MoERunner
    (core.py:818) but for the pre-routed Pack-based calling convention.
    """

    backend_key = "trtllm_fp4_routed"

    # Dynamic tensors: topk_ids (idx 0), hidden_states (1), hidden_states_scale (2)
    # all vary in dim 0 (num_tokens).  Weight tensors (idx 3..7) are fixed.
    _dynamic_tensor_initializers = [
        # 0: topk_ids [M, top_k] int32
        lambda shapes, dtype, device: torch.zeros(
            shapes, dtype=torch.int32, device=device
        ),
        # 1: hidden_states [M, H//2] uint8 (packed NVFP4)
        lambda shapes, dtype, device: torch.randint(
            0, 256, shapes, dtype=torch.uint8, device=device
        ),
        # 2: hidden_states_scale [M, H//16] float8_e4m3fn
        lambda shapes, dtype, device: torch.randint(
            1, 128, shapes, dtype=torch.uint8, device=device
        ).view(torch.float8_e4m3fn),
    ]
    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0, 1, 2),
                dim_idx=(0, 0, 0),
                gen_tuning_buckets=get_last_power_of_2_num_tokens_buckets(8192),
                map_to_tuning_buckets=lambda x: min(last_positive_power_of_2(x), 8192),
                tensor_initializers=_dynamic_tensor_initializers,
            ),
        ),
    )

    def __init__(self, config: MoEConfig, device: torch.device):
        from ..tllm_enums import (
            ActivationType,
            DtypeTrtllmGen,
            Fp8QuantizationType,
            WeightLayout,
        )
        from .core import get_trtllm_moe_sm100_module

        self.config = config
        self.device = device
        self._moe_op = get_trtllm_moe_sm100_module()

        # NVFP4 instance key for trtllm_get_valid_moe_configs
        routing = config.routing
        experts = config.experts
        num_local_experts = experts.local_num_experts or routing.num_experts
        # hidden_size is filled from tensor shape at first get_valid_tactics call
        self._num_local_experts = num_local_experts
        self._intermediate_size = experts.intermediate_size
        self._dtype_act = DtypeTrtllmGen.E2m1
        self._dtype_weights = DtypeTrtllmGen.E2m1
        self._quantization_type = Fp8QuantizationType.NoneFp8
        self._activation_type = ActivationType.Swiglu
        self._use_shuffled_weight = True
        self._weight_layout = int(WeightLayout.MajorK)
        self._tactics_cache: dict = {}

    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: Any
    ) -> List[List[int]]:
        # inputs[1] = hidden_states [num_tokens, hidden_size // 2] (FP4 packed)
        hidden_states = inputs[1]
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1] * 2  # FP4 packed

        key = (hidden_size, num_tokens)
        if key in self._tactics_cache:
            return self._tactics_cache[key]

        tactics = self._moe_op.trtllm_get_valid_moe_configs(
            self._dtype_act,
            self._dtype_weights,
            self._quantization_type,
            self.config.routing.top_k,
            hidden_size,
            self._intermediate_size,
            self._num_local_experts,
            int(self._activation_type),
            self._use_shuffled_weight,
            self._weight_layout,
            num_tokens,
        )
        self._tactics_cache[key] = tactics
        return tactics

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        (
            topk_ids,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm1_alpha,
            gemm2_weights,
            gemm2_weights_scale,
        ) = inputs

        cfg = self.config
        routing = cfg.routing
        num_local_experts = self._num_local_experts
        tactic_pair = [-1, -1] if tactic == -1 else list(tactic)

        # Direct call to the C++ op (bypasses the Python wrapper because
        # that wrapper doesn't accept a tactic argument).  routing_logits=None
        # tells the kernel to use the pre-packed topk_ids path.
        results = self._moe_op.trtllm_fp4_block_scale_moe(
            None,  # routing_logits
            topk_ids,
            None,  # expert_weights (fused into topk_ids)
            None,  # routing_bias
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            None,  # gemm1_bias
            gemm1_alpha,
            None,  # gemm1_beta
            None,  # gemm1_clamp_limit
            gemm2_weights,
            gemm2_weights_scale,
            None,  # gemm2_bias
            None,  # output1_scale_scalar
            None,  # output1_scale_gate_scalar
            None,  # output2_scale_scalar
            routing.num_experts,
            routing.top_k,
            routing.n_group,
            routing.topk_group,
            self._intermediate_size,
            cfg.experts.local_expert_offset,
            num_local_experts,
            routing.routed_scaling_factor,
            int(routing.method.value),
            cfg.execution.do_finalize,
            cfg.execution.enable_pdl,
            int(self._activation_type),
            None,  # output — let kernel allocate
            tactic_pair,
        )
        return results[0]

    @staticmethod
    def pack_inputs(
        act: MoEActivationPack,
        weights: MoEWeightPack,
        local_expert_offset: int = 0,
    ) -> List[torch.Tensor]:
        """Translate Packs → List[Tensor] for TRTLLM routed forward.

        Packs expert ids + routing weights into the int32 format TRTLLM expects:
        ((expert_id - offset) << 16) | bf16_bits_of_weight
        """
        ids = act.selected_experts - local_expert_offset
        weight_bf16_bits = (
            act.final_scales.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        )
        topk_ids = (ids << 16) | (weight_bf16_bits & 0xFFFF)

        return [
            topk_ids,
            act.hidden_states_q,
            act.hidden_states_scale,
            weights.w1_q,
            weights.w1_scale,
            weights.w1_alpha,
            weights.w2_q,
            weights.w2_scale,
        ]

    def __hash__(self):
        return hash(("trtllm_fp4_routed",))
