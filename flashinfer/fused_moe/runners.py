"""Unified MoE runner adapters for the autotuned pre-routed NVFP4 path.

Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Each runner wraps one backend and translates (MoEActivationPack, MoEWeightPack)
into the backend's native calling convention.  Both MVP runners are thin
adapters over an existing, canonical inner runner (CuteDSL's
``CuteDslFusedMoENvfp4Runner`` and trtllm-gen's ``core.MoERunner``) so the
fragile backend-specific kernel-launch code lives in exactly one place.

MVP scope: NVFP4 only, two backends (CuteDSL, TRTLLM routed).
"""

from __future__ import annotations

from typing import Any, List

import torch

from ..autotuner import TunableRunner
from .api import MoEActivationPack, MoEConfig, MoEWeightPack


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
        # tuning_config is an instance attribute on the inner runner (its
        # dummy expert-id span depends on num_experts/offset), so read it from
        # the instance we just built, not off the class.
        self.tuning_config = self._inner.tuning_config

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: Any) -> List[Any]:
        return self._inner.get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self._inner.forward(
            inputs, tactic=tactic, do_preparation=do_preparation, **kwargs
        )

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        """Translate Packs → List[Tensor] expected by CuteDslFusedMoENvfp4Runner.

        Expected weight view keys: w1_weight, w1_weight_sf, w1_alpha,
        fc2_input_scale, w2_weight, w2_weight_sf, w2_alpha.
        Input order: x, x_sf, token_selected_experts, token_final_scales,
                     w1_weight, w1_weight_sf, w1_alpha, fc2_input_scale,
                     w2_weight, w2_weight_sf, w2_alpha, moe_output.

        The trailing ``moe_output`` buffer (index 11) is optional for a direct
        ``forward`` (the inner runner allocates it), but the inner runner's
        tuning_config declares index 11 as a dynamic tensor, so it must be
        present for the autotuner profiling path to assign it a per-bucket
        initializer.
        """
        v = weights.get_view(self.backend_key)
        num_tokens = act.hidden_states_q.shape[0]
        hidden_size = act.hidden_states_q.shape[1] * 2  # FP4 packed
        moe_output = act.hidden_states_q.new_empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16
        )
        return [
            act.hidden_states_q,
            act.hidden_states_scale.unsqueeze(-1),  # CuteDSL expects [M, H//16, 1]
            act.selected_experts,
            act.final_scales,
            v["w1_weight"],
            v["w1_weight_sf"],
            v["w1_alpha"],
            v["fc2_input_scale"],
            v["w2_weight"],
            v["w2_weight_sf"],
            v["w2_alpha"],
            moe_output,
        ]

    def __hash__(self):
        return hash(("cute_dsl_nvfp4", hash(self._inner)))


# ---------------------------------------------------------------------------
# TRTLLM FP4 routed runner — delegates to the canonical trtllm-gen MoERunner
# ---------------------------------------------------------------------------


class TrtllmFp4RoutedRunner(TunableRunner):
    """Pre-routed NVFP4 adapter over the canonical trtllm-gen ``MoERunner``.

    Translates (MoEActivationPack, MoEWeightPack) into the ``MoeRunnerInputs`` list
    plus the static weight/config kwargs that ``core.MoERunner.forward``
    consumes, then delegates tactic enumeration, tuning-config construction, and
    the tactic'd forward to that inner runner.  This mirrors
    ``CuteDslNvfp4Runner`` (which wraps ``CuteDslFusedMoENvfp4Runner``) and keeps
    the fragile raw-op positional launch in exactly one place —
    ``core.MoERunner.forward``.

    Routing is pre-computed (``RoutingInputMode.PackedPrecomputed``): the packed
    int32 top-k ids carry ``((expert_id - local_offset) << 16) | bf16(weight)``.
    The inner ``MoERunner`` needs the hidden size for its tactic keys and tuning
    buckets, so it is built lazily on the first ``pack_inputs`` call.
    """

    backend_key = "trtllm_fp4_routed"

    def __init__(self, config: MoEConfig, device: torch.device):
        from ..tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
        from ..utils import device_support_pdl
        from .core import get_trtllm_moe_sm100_module

        self.config = config
        self.device = device
        self._module = get_trtllm_moe_sm100_module()

        routing = config.routing
        experts = config.experts
        execution = config.execution
        self._num_local_experts = experts.local_num_experts or routing.num_experts
        self._local_expert_offset = experts.local_expert_offset
        self._intermediate_size = experts.intermediate_size
        self._activation_type = int(config.activation.type)
        self._tune_max_num_tokens = execution.tune_max_num_tokens

        # NVFP4: E2m1 activations + weights, no fp8 sub-quant.
        self._dtype_act = DtypeTrtllmGen.E2m1
        self._dtype_weights = DtypeTrtllmGen.E2m1
        self._fp8_quantization_type = Fp8QuantizationType.NoneFp8

        # enable_pdl=None means "auto" — resolve once here exactly like the
        # high-level wrapper does before building its MoERunner, because the raw
        # op (reached via MoERunner.forward) expects a concrete bool.  Resolving
        # once also keeps the value stable across CUDA-graph capture/replay.
        enable_pdl = execution.enable_pdl
        if enable_pdl is None:
            enable_pdl = device_support_pdl(device)
        self._enable_pdl = enable_pdl

        # Built lazily on first pack_inputs once hidden_size is known.
        self._inner: Any = None
        self._static_kwargs: dict = {}
        self.tuning_config: Any = None

    def _ensure_inner(self, hidden_size: int) -> None:
        if self._inner is not None:
            return
        from ..tllm_enums import WeightLayout

        self._inner = self._module.MoERunner(
            top_k=self.config.routing.top_k,
            num_local_experts=self._num_local_experts,
            dtype_act=self._dtype_act,
            dtype_weights=self._dtype_weights,
            fp8_quantization_type=self._fp8_quantization_type,
            hidden_size=hidden_size,
            intermediate_size=self._intermediate_size,
            activation_type=self._activation_type,
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.MajorK),
            use_per_token_scaling=False,
            num_experts=self.config.routing.num_experts,
        )

    def get_valid_tactics(  # type: ignore[override]
        self, inputs: List[torch.Tensor], profile: Any
    ) -> List[Any]:
        # The inner runner reads num_tokens from inputs + its own instance key;
        # no static kwargs are needed for tactic enumeration.
        return self._inner.get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        # MoELayer's autotuner call passes no kwargs, so the static weight/config
        # kwargs are injected here.  The inner runner writes the result in-place
        # into inputs[0] (the output buffer of the MoeRunnerInputs list).
        self._inner.forward(
            inputs,
            tactic=tactic,
            do_preparation=do_preparation,
            **self._static_kwargs,
        )
        return inputs[0]

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        """Translate Packs → the ``MoeRunnerInputs`` list ``core.MoERunner`` expects.

        Expected weight view keys: gemm1_weights, gemm1_weights_scale,
        gemm1_alpha, gemm2_weights, gemm2_weights_scale, and optionally
        output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar.

        The local-shard offset comes from ``ExpertConfig.local_expert_offset``
        on the config this runner was built with.  For expert-parallel
        pre-routed inputs the kernel indexes local experts as
        ``[0, local_num_experts)``, so global expert ids are shifted down by the
        local offset before packing.
        """
        from .core import MoeRunnerInputs, RoutingInputMode

        v = weights.get_view(self.backend_key)
        routing = self.config.routing

        num_tokens = act.hidden_states_q.shape[0]
        hidden_size = act.hidden_states_q.shape[1] * 2  # FP4 packed

        # trtllm-gen requires the nvfp4 activation scale as float8_e4m3fn; the
        # canonical Pack may carry it as raw uint8 bytes.
        hidden_states_scale = act.hidden_states_scale
        if hidden_states_scale.dtype == torch.uint8:
            hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn)

        # Packed pre-routed top-k ids: (GLOBAL expert_id << 16) | bf16(weight).
        # The kernel expects GLOBAL ids and filters/maps them via the separately
        # passed ``local_expert_offset`` (mirrors trtllm_bf16_routed_moe in
        # tests/moe/test_trtllm_gen_routed_fused_moe.py). Do NOT pre-subtract the
        # offset: on ranks with local_expert_offset>0 that yields a local id below
        # the offset, which the kernel treats as non-local and skips → zero output.
        ids = act.selected_experts
        weight_bf16_bits = (
            act.final_scales.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        )
        topk_ids = (ids << 16) | (weight_bf16_bits & 0xFFFF)

        output = act.hidden_states_q.new_empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16
        )
        # PackedPrecomputed still requires a (kernel-side) topk_weights buffer:
        # the raw op declares it non-Optional.  The high-level wrapper allocates
        # an empty bf16 placeholder here; we mirror that since we bypass it.
        expert_weights = act.final_scales.new_empty(
            (num_tokens, routing.top_k), dtype=torch.bfloat16
        )
        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=None,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=act.hidden_states_q,
            hidden_states_scale=hidden_states_scale,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )

        # Static (num_tokens-invariant) launch arguments for the fp4 branch of
        # MoERunner.forward.  None-valued entries are the optional gemm bias /
        # swiglu beta-clamp / per-token-scale paths not used by the MVP.
        self._static_kwargs = dict(
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
            routing_bias=None,
            gemm1_weights=v["gemm1_weights"],
            gemm1_weights_scale=v["gemm1_weights_scale"],
            gemm1_bias=None,
            gemm1_alpha=v.get("gemm1_alpha"),
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=v["gemm2_weights"],
            gemm2_weights_scale=v["gemm2_weights_scale"],
            gemm2_bias=None,
            output1_scale_scalar=v.get("output1_scale_scalar"),
            output1_scale_gate_scalar=v.get("output1_scale_gate_scalar"),
            output2_scale_scalar=v.get("output2_scale_scalar"),
            per_token_scale=None,
            num_experts=routing.num_experts,
            n_group=routing.n_group,
            topk_group=routing.topk_group,
            local_expert_offset=self._local_expert_offset,
            routed_scaling_factor=routing.routed_scaling_factor,
            routing_method_type=int(routing.method),
            do_finalize=self.config.execution.do_finalize,
            enable_pdl=self._enable_pdl,
        )

        self._ensure_inner(hidden_size)
        # Reuse the inner runner's tuning-config builder so the num_tokens
        # buckets honor ExecutionConfig.tune_max_num_tokens (CR5).
        self.tuning_config = self._inner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=self._tune_max_num_tokens,
            # Match the canonical trtllm-gen wrappers' profiling regime so
            # choose_one() tunes under the same conditions as deployment
            # (otherwise it can cache a tactic picked under a different regime).
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )
        return moe_inputs.to_list()

    def __hash__(self):
        return hash(("trtllm_fp4_routed",))


# ---------------------------------------------------------------------------
# TRTLLM BF16 routed runner — canonical trtllm-gen MoERunner, bf16 dtypes
# ---------------------------------------------------------------------------


class TrtllmBf16RoutedRunner(TunableRunner):
    """Pre-routed BF16 adapter over the canonical trtllm-gen ``MoERunner``.

    Mirrors :class:`TrtllmFp4RoutedRunner` but with ``Bfloat16`` activation +
    weight dtypes and no scale-factor tensors, wrapping the same inner
    ``MoERunner`` (whose ``forward`` dispatches to ``moe_op.trtllm_bf16_moe`` when
    ``dtype_weights == Bfloat16``).  Used for the EP grouped-GEMM bf16 path: the
    packed pre-routed ids carry ``(GLOBAL expert_id << 16) | bf16(weight)`` (with
    ``local_expert_offset`` passed separately);
    with the EP bridge's synthesized ``top_k=1`` + ``weight=1`` and
    ``do_finalize=True``, the output comes back in input row order.

    The bf16 MoE entry point requires the ``BlockMajorK`` weight layout.
    """

    backend_key = "trtllm_bf16_routed"

    def __init__(self, config: MoEConfig, device: torch.device):
        from ..tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
        from ..utils import device_support_pdl
        from .core import get_trtllm_moe_sm100_module

        self.config = config
        self.device = device
        self._module = get_trtllm_moe_sm100_module()

        routing = config.routing
        experts = config.experts
        execution = config.execution
        self._num_local_experts = experts.local_num_experts or routing.num_experts
        self._local_expert_offset = experts.local_expert_offset
        self._intermediate_size = experts.intermediate_size
        self._activation_type = int(config.activation.type)
        self._tune_max_num_tokens = execution.tune_max_num_tokens

        self._dtype_act = DtypeTrtllmGen.Bfloat16
        self._dtype_weights = DtypeTrtllmGen.Bfloat16
        self._fp8_quantization_type = Fp8QuantizationType.NoneFp8

        enable_pdl = execution.enable_pdl
        if enable_pdl is None:
            enable_pdl = device_support_pdl(device)
        self._enable_pdl = enable_pdl

        self._inner: Any = None
        self._static_kwargs: dict = {}
        self.tuning_config: Any = None

    def _ensure_inner(self, hidden_size: int) -> None:
        if self._inner is not None:
            return
        from ..tllm_enums import WeightLayout

        self._inner = self._module.MoERunner(
            top_k=self.config.routing.top_k,
            num_local_experts=self._num_local_experts,
            dtype_act=self._dtype_act,
            dtype_weights=self._dtype_weights,
            fp8_quantization_type=self._fp8_quantization_type,
            hidden_size=hidden_size,
            intermediate_size=self._intermediate_size,
            activation_type=self._activation_type,
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.BlockMajorK),
            use_per_token_scaling=False,
            num_experts=self.config.routing.num_experts,
        )

    def get_valid_tactics(  # type: ignore[override]
        self, inputs: List[torch.Tensor], profile: Any
    ) -> List[Any]:
        return self._inner.get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._inner.forward(
            inputs,
            tactic=tactic,
            do_preparation=do_preparation,
            **self._static_kwargs,
        )
        return inputs[0]

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        """Translate Packs → the ``MoeRunnerInputs`` list for the bf16 path.

        Expected weight view keys: gemm1_weights, gemm2_weights (BlockMajorK,
        shuffled).  ``act.hidden_states_q`` carries the raw bf16 activations
        (the EP bridge does not quantize on the bf16 path);
        ``act.hidden_states_scale`` is unused.
        """
        from .core import MoeRunnerInputs, RoutingInputMode

        v = weights.get_view(self.backend_key)
        routing = self.config.routing

        hidden_states = act.hidden_states_q  # raw bf16 on this path
        num_tokens, hidden_size = hidden_states.shape

        # Packed pre-routed top-k ids: (GLOBAL expert_id << 16) | bf16(weight).
        # The kernel expects GLOBAL ids and filters/maps them via the separately
        # passed ``local_expert_offset`` (mirrors trtllm_bf16_routed_moe in
        # tests/moe/test_trtllm_gen_routed_fused_moe.py). Do NOT pre-subtract the
        # offset: on ranks with local_expert_offset>0 that yields a local id below
        # the offset, which the kernel treats as non-local and skips → zero output.
        ids = act.selected_experts
        weight_bf16_bits = (
            act.final_scales.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        )
        topk_ids = (ids << 16) | (weight_bf16_bits & 0xFFFF)

        output = hidden_states.new_empty((num_tokens, hidden_size))
        expert_weights = act.final_scales.new_empty(
            (num_tokens, routing.top_k), dtype=torch.bfloat16
        )
        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=None,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=hidden_states,
            hidden_states_scale=None,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )

        from ..tllm_enums import WeightLayout

        self._static_kwargs = dict(
            routing_input_mode=RoutingInputMode.PackedPrecomputed,
            routing_bias=None,
            gemm1_weights=v["gemm1_weights"],
            gemm2_weights=v["gemm2_weights"],
            gemm1_alpha=v.get("gemm1_alpha"),
            gemm1_beta=v.get("gemm1_beta"),
            gemm1_clamp_limit=v.get("gemm1_clamp_limit"),
            num_experts=routing.num_experts,
            n_group=routing.n_group,
            topk_group=routing.topk_group,
            local_expert_offset=self._local_expert_offset,
            routed_scaling_factor=routing.routed_scaling_factor,
            routing_method_type=int(routing.method),
            use_shuffled_weight=True,
            weight_layout=int(WeightLayout.BlockMajorK),
            do_finalize=self.config.execution.do_finalize,
            enable_pdl=self._enable_pdl,
            norm_topk_prob=False,
        )

        self._ensure_inner(hidden_size)
        self.tuning_config = self._inner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=self._tune_max_num_tokens,
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )
        return moe_inputs.to_list()

    def __hash__(self):
        return hash(("trtllm_bf16_routed",))
