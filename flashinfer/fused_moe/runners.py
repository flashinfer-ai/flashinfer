"""Unified MoE runner adapters for autotuned pre-routed and FromLogits paths.

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
into the backend's native calling convention. The adapters reuse existing
canonical inner runners (CuteDSL's
``CuteDslFusedMoENvfp4Runner`` and trtllm-gen's ``core.MoERunner``) so the
fragile backend-specific kernel-launch code lives in exactly one place.
"""

from __future__ import annotations

from typing import Any, ClassVar, List

import torch

from ..autotuner import TunableRunner, TuningConfig
from .api import (
    ActivationType,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantVariant,
    RoutingInputMode,
)


def _validate_pack_devices(act: MoEActivationPack, runner: str) -> None:
    """Recheck pack tensor placement at the mutable runner boundary."""
    expected = act.hidden_states_q.device
    for name in (
        "hidden_states_scale",
        "topk_ids",
        "topk_weights",
        "routing_logits",
        "routing_bias",
    ):
        tensor = getattr(act, name)
        if tensor is not None and tensor.device != expected:
            raise ValueError(
                f"{runner}: {name} is on {tensor.device}, expected {expected} "
                "(hidden_states_q device)."
            )


def _validate_prerouted_inputs(
    act: MoEActivationPack, num_tokens: int, top_k: int, runner: str
) -> None:
    """Runner-boundary validation for pre-routed packs.

    Raises (never asserts — these must survive ``python -O``): a shape or
    presence mismatch here silently mis-packs against the kernel's
    ``top_k``-sized buffers or reads out of bounds in C++.  Duplicates the
    construction-time ``MoEActivationPack.__post_init__`` checks on purpose —
    the pack is mutable, so the launch boundary is the airtight layer.
    """
    if act.topk_ids is None or act.topk_weights is None:
        raise ValueError(
            f"{runner}: routing_input_mode=PackedPrecomputed requires "
            "topk_ids + topk_weights."
        )
    if act.routing_logits is not None or act.routing_bias is not None:
        raise ValueError(
            f"{runner}: routing_logits/routing_bias are only consumed by "
            "in-kernel (FromLogits) routing."
        )
    _validate_pack_devices(act, runner)
    expected = (num_tokens, top_k)
    for name in ("topk_ids", "topk_weights"):
        shape = tuple(getattr(act, name).shape)
        if shape != expected:
            raise ValueError(
                f"{runner}: {name} shape {shape} != {expected} "
                "(num_tokens, RoutingConfig.top_k) — a column mismatch "
                "mis-packs against the kernel's top_k-sized buffers."
            )
    if act.topk_ids.dtype != torch.int32:
        # The launcher casts data_ptr without a dtype ICHECK, so an int64
        # tensor here is read as int32 bytes — silent garbage routing.
        raise TypeError(
            f"{runner}: topk_ids must be torch.int32, got {act.topk_ids.dtype} "
            "(torch.topk returns int64 — cast before packing)."
        )


def _validate_logits_inputs(
    act: MoEActivationPack, num_tokens: int, num_experts: int, runner: str
) -> None:
    """Runner-boundary validation for FromLogits packs (raises, see above).

    The dtype checks guard against SILENT corruption: the launcher maps
    bf16 -> Bfloat16 and anything else -> Fp32 with no dtype ICHECK, so an
    fp16 bias/logits tensor would be reinterpreted as fp32 bits.  Bias dtype
    is independent of logits dtype (mixed fp32 logits + bf16 bias is the
    standard DeepSeek-V3 shape — see test_routing_dtype_flexibility).
    """
    if act.routing_logits is None:
        raise ValueError(
            f"{runner}: routing_input_mode=FromLogits requires routing_logits."
        )
    if act.topk_ids is not None or act.topk_weights is not None:
        raise ValueError(
            f"{runner}: FromLogits computes topk_ids/topk_weights in-kernel; "
            "leave them None."
        )
    _validate_pack_devices(act, runner)
    logits = act.routing_logits
    if logits.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"{runner}: routing_logits must be float32 or bfloat16, got {logits.dtype}."
        )
    if tuple(logits.shape) != (num_tokens, num_experts):
        raise ValueError(
            f"{runner}: routing_logits shape {tuple(logits.shape)} != "
            f"({num_tokens}, {num_experts}) (num_tokens, num_experts) — "
            "routing scores are over the GLOBAL expert set."
        )
    if act.routing_bias is not None:
        if act.routing_bias.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(
                f"{runner}: routing_bias must be bfloat16 or float32, "
                f"got {act.routing_bias.dtype}."
            )
        if tuple(act.routing_bias.shape) != (num_experts,):
            raise ValueError(
                f"{runner}: routing_bias shape {tuple(act.routing_bias.shape)} "
                f"!= ({num_experts},) (num_experts,)."
            )


class MoERunner(TunableRunner):
    """Base class for unified MoE backend runners."""

    backend_key: ClassVar[str] = ""
    supported_routing_modes: tuple[RoutingInputMode, ...] = ()
    supported_quant_variants: ClassVar[tuple[QuantVariant, ...]] = ()

    config: MoEConfig

    def check_support(self) -> None:
        """Raise if the initialized runner cannot execute its configuration."""
        variant = self.config.quant.variant
        if variant not in self.supported_quant_variants:
            raise NotImplementedError(
                f"{type(self).__name__} does not support QuantVariant.{variant.name}."
            )


# ---------------------------------------------------------------------------
# CuteDSL NVFP4 runner — delegates to the existing CuteDslFusedMoENvfp4Runner
# ---------------------------------------------------------------------------


class CuteDslNvfp4Runner(MoERunner):
    """Wraps CuteDslFusedMoENvfp4Runner, translating Pack inputs into its
    List[Tensor] convention."""

    backend_key = "cute_dsl_nvfp4"
    # CuteDSL has no in-kernel router; it only consumes pre-routed packs.
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = (QuantVariant.NVFP4,)

    def check_support(self) -> None:
        super().check_support()
        if self.config.activation.type is not ActivationType.Swiglu:
            raise NotImplementedError(
                f"{type(self).__name__} supports only the Swiglu activation."
            )

    def __init__(self, config: MoEConfig, device: torch.device):
        from .cute_dsl.fused_moe import _cute_dsl_fused_moe_nvfp4_impl
        from .cute_dsl.tuner import CuteDslFusedMoENvfp4Runner

        self.config = config
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
        # MoELayer already filters by supported_routing_modes; this guards the
        # direct-runner path (tests/benchmarks) against silently forwarding a
        # logits pack's None topk tensors into the kernel launch.
        if act.routing_input_mode not in self.supported_routing_modes:
            raise NotImplementedError(
                f"CuteDslNvfp4Runner does not support "
                f"routing_input_mode={act.routing_input_mode!r} "
                "(only PackedPrecomputed is wired; CuteDSL has no in-kernel router)."
            )
        v = weights.get_view(self.backend_key)
        num_tokens = act.hidden_states_q.shape[0]
        _validate_prerouted_inputs(
            act, num_tokens, self._inner.top_k, "CuteDslNvfp4Runner"
        )
        hidden_size = act.hidden_states_q.shape[1] * 2  # FP4 packed
        moe_output = act.hidden_states_q.new_empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16
        )
        return [
            act.hidden_states_q,
            act.hidden_states_scale.unsqueeze(-1),  # CuteDSL expects [M, H//16, 1]
            act.topk_ids,
            act.topk_weights,
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


class TrtllmFp4RoutedRunner(MoERunner):
    """NVFP4 adapter over the canonical trtllm-gen ``MoERunner``.

    Translates (MoEActivationPack, MoEWeightPack) into the ``MoeRunnerInputs`` list
    plus the static weight/config kwargs that ``core.MoERunner.forward``
    consumes, then delegates tactic enumeration, tuning-config construction, and
    the tactic'd forward to that inner runner.  This mirrors
    ``CuteDslNvfp4Runner`` (which wraps ``CuteDslFusedMoENvfp4Runner``) and keeps
    the fragile raw-op positional launch in exactly one place —
    ``core.MoERunner.forward``.

    Routing mode is chosen per-call from ``act.routing_input_mode``:

    * **pre-routed** (``RoutingInputMode.PackedPrecomputed``): the pack carries
      ``topk_ids`` / ``topk_weights`` and the runner packs them into int32 top-k ids
      ``(GLOBAL expert_id << 16) | bf16(weight)`` (the kernel maps global ids to the
      local shard via the separately passed ``local_expert_offset``).
    * **in-kernel** (``RoutingInputMode.FromLogits``): the pack carries
      ``routing_logits`` (+ optional ``routing_bias``); the kernel computes the top-k
      selection per ``RoutingConfig.method`` and writes ``topk_ids`` / ``topk_weights``
      into the OUTPUT buffers we allocate.

    The inner ``MoERunner`` needs the hidden size for its tactic keys and tuning
    buckets, so it is built lazily on the first ``pack_inputs`` call.
    """

    backend_key = "trtllm_fp4_routed"
    supported_routing_modes = (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.FromLogits,
    )
    supported_quant_variants = (QuantVariant.NVFP4,)

    def check_support(self) -> None:
        super().check_support()
        if self.config.activation.type is not ActivationType.Swiglu:
            raise NotImplementedError(
                f"{type(self).__name__} supports only the Swiglu activation."
            )

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

        Routing mode is read from ``act.routing_input_mode``: ``FromLogits`` drives
        in-kernel routing from ``act.routing_logits``; ``PackedPrecomputed`` packs the
        pre-routed ``act.topk_ids`` / ``act.topk_weights``.

        The local-shard offset comes from ``ExpertConfig.local_expert_offset``
        on the config this runner was built with.  ``topk_ids`` carries
        GLOBAL expert ids and is packed as-is; the kernel performs the
        global→local mapping itself by subtracting ``local_expert_offset``
        (passed via the static kwargs) and dropping ids outside
        ``[offset, offset + local_num_experts)``.
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

        output = act.hidden_states_q.new_empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16
        )

        routing_input_mode = act.routing_input_mode
        if routing_input_mode == RoutingInputMode.FromLogits:
            # In-kernel routing: topk_ids/expert_weights are OUTPUT buffers the kernel fills.
            # Unlike the FP8 launcher, FP4 receives routing_input_mode explicitly;
            # non-empty output buffers therefore do not select precomputed routing.
            # We allocate them here (mirroring trtllm_fp4_block_scale_moe_op, core.py ~2268)
            # because MoERunner.forward calls the raw op directly, bypassing the buffer-allocating
            # wrapper. Weight dtype mirrors logits dtype (core.py:2253).
            _validate_logits_inputs(
                act, num_tokens, routing.num_experts, "TrtllmFp4RoutedRunner"
            )
            routing_logits = act.routing_logits
            routing_bias = act.routing_bias
            topk_ids = act.hidden_states_q.new_empty(
                (num_tokens, routing.top_k), dtype=torch.int32
            )
            # MUST be bf16 regardless of logits dtype: the fp4 routing kernel
            # writes bf16 expert weights, so inheriting fp32 from the logits
            # mislabels the returned buffer (gh #3595 — the canonical wrapper
            # in core.py hardcodes bf16 for the same reason).
            expert_weights = routing_logits.new_empty(
                (num_tokens, routing.top_k), dtype=torch.bfloat16
            )
        elif routing_input_mode == RoutingInputMode.PackedPrecomputed:
            # Pre-routed: pack the host selection into (GLOBAL expert_id << 16) | bf16(weight).
            # The kernel expects GLOBAL ids and filters/maps them via the separately
            # passed ``local_expert_offset`` (mirrors trtllm_bf16_routed_moe in
            # tests/moe/test_trtllm_gen_routed_fused_moe.py). Do NOT pre-subtract the
            # offset: on ranks with local_expert_offset>0 that yields a local id below
            # the offset, which the kernel treats as non-local and skips → zero output.
            _validate_prerouted_inputs(
                act, num_tokens, routing.top_k, "TrtllmFp4RoutedRunner"
            )
            routing_logits = None
            routing_bias = None
            ids = act.topk_ids
            weight_bf16_bits = (
                act.topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            )
            topk_ids = (ids << 16) | (weight_bf16_bits & 0xFFFF)
            # PackedPrecomputed still requires a (kernel-side) topk_weights buffer:
            # the raw op declares it non-Optional.  The high-level wrapper allocates
            # an empty bf16 placeholder here; we mirror that since we bypass it.
            expert_weights = act.topk_weights.new_empty(
                (num_tokens, routing.top_k), dtype=torch.bfloat16
            )
        else:
            raise NotImplementedError(
                f"TrtllmFp4RoutedRunner does not support "
                f"routing_input_mode={routing_input_mode!r} "
                "(only FromLogits and PackedPrecomputed are wired)."
            )

        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
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
            routing_input_mode=routing_input_mode,
            routing_bias=routing_bias,
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
# TRTLLM block-FP8 runner — DeepSeek FP8 and MXFP8
# ---------------------------------------------------------------------------


class TrtllmFp8BlockRunner(MoERunner):
    """Block-FP8 adapter over the canonical trtllm-gen ``MoERunner``.

    DeepSeek FP8 and MXFP8 share the kernel family but not scale contracts:
    DeepSeek uses FP32 128-element/128x128 block scales, while MXFP8 uses
    linear UE8M0 scales over 32-element K blocks.
    """

    backend_key = "trtllm_fp8_block"
    supported_routing_modes = (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.FromLogits,
    )
    supported_quant_variants = (
        QuantVariant.DeepSeekFp8,
        QuantVariant.MxFp8,
    )

    def check_support(self) -> None:
        super().check_support()
        if self.config.activation.type is not ActivationType.Swiglu:
            raise NotImplementedError(
                f"{type(self).__name__} supports only the Swiglu activation."
            )
        if not self.config.execution.do_finalize:
            raise NotImplementedError(
                f"{type(self).__name__} supports only do_finalize=True."
            )

    def __init__(self, config: MoEConfig, device: torch.device):
        from ..tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
        from ..utils import device_support_pdl
        from .api import QuantVariant
        from .core import get_trtllm_moe_sm100_module

        if config.quant.variant is QuantVariant.MxFp8:
            dtype = DtypeTrtllmGen.MxE4m3
            fp8_type = Fp8QuantizationType.MxFp8
        else:
            # Use a harmless default while construction precedes check_support().
            # Unsupported variants are rejected before the runner is registered.
            dtype = DtypeTrtllmGen.E4m3
            fp8_type = Fp8QuantizationType.DeepSeekFp8

        self.config = config
        self.device = device
        self._module = get_trtllm_moe_sm100_module()
        self._variant = config.quant.variant
        self._dtype_act = dtype
        self._dtype_weights = dtype
        self._fp8_quantization_type = fp8_type
        self._use_shuffled_weight = config.quant.variant is QuantVariant.MxFp8

        routing = config.routing
        experts = config.experts
        execution = config.execution
        self._num_local_experts = experts.local_num_experts or routing.num_experts
        self._local_expert_offset = experts.local_expert_offset
        self._intermediate_size = experts.intermediate_size
        self._activation_type = int(config.activation.type)
        self._tune_max_num_tokens = execution.tune_max_num_tokens

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
            use_shuffled_weight=self._use_shuffled_weight,
            weight_layout=int(WeightLayout.MajorK),
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

    def _validate_fp8_tensors(
        self,
        act: MoEActivationPack,
        view: dict,
        hidden_size: int,
    ) -> torch.Tensor:
        from .api import QuantVariant

        if act.hidden_states_q.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "TrtllmFp8BlockRunner requires float8_e4m3fn hidden_states_q, "
                f"got {act.hidden_states_q.dtype}."
            )
        scale = act.hidden_states_scale
        if scale is None:
            raise ValueError("TrtllmFp8BlockRunner requires hidden_states_scale.")
        num_tokens = act.hidden_states_q.shape[0]
        if self._variant is QuantVariant.DeepSeekFp8:
            expected_scale = (hidden_size // 128, num_tokens)
            if scale.dtype != torch.float32 or tuple(scale.shape) != expected_scale:
                raise ValueError(
                    "DeepSeekFp8 hidden_states_scale must be float32 with shape "
                    f"{expected_scale}, got {scale.dtype} {tuple(scale.shape)}."
                )
            expected_w1_scale = (
                self._num_local_experts,
                2 * self._intermediate_size // 128,
                hidden_size // 128,
            )
            expected_w2_scale = (
                self._num_local_experts,
                hidden_size // 128,
                self._intermediate_size // 128,
            )
            scale_dtype = torch.float32
        else:
            expected_scale = (num_tokens, hidden_size // 32)
            if scale.dtype != torch.uint8 or tuple(scale.shape) != expected_scale:
                raise ValueError(
                    "MxFp8 hidden_states_scale must be uint8 UE8M0 with shape "
                    f"{expected_scale}, got {scale.dtype} {tuple(scale.shape)}."
                )
            expected_w1_scale = (
                self._num_local_experts,
                2 * self._intermediate_size,
                hidden_size // 32,
            )
            expected_w2_scale = (
                self._num_local_experts,
                hidden_size,
                self._intermediate_size // 32,
            )
            scale_dtype = torch.uint8

        expected_weights = {
            "gemm1_weights": (
                self._num_local_experts,
                2 * self._intermediate_size,
                hidden_size,
            ),
            "gemm2_weights": (
                self._num_local_experts,
                hidden_size,
                self._intermediate_size,
            ),
        }
        for name, expected in expected_weights.items():
            tensor = view[name]
            if tensor.dtype != torch.float8_e4m3fn or tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must be float8_e4m3fn with shape {expected}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}."
                )
        for name, expected in (
            ("gemm1_weights_scale", expected_w1_scale),
            ("gemm2_weights_scale", expected_w2_scale),
        ):
            tensor = view[name]
            if tensor.dtype != scale_dtype or tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must be {scale_dtype} with shape {expected}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}."
                )
        return scale

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        from ..tllm_enums import WeightLayout
        from .core import MoeRunnerInputs, RoutingInputMode

        view = weights.get_view(self.backend_key)
        routing = self.config.routing
        num_tokens, hidden_size = act.hidden_states_q.shape
        hidden_states_scale = self._validate_fp8_tensors(act, view, hidden_size)

        output = act.hidden_states_q.new_empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16
        )
        routing_input_mode = act.routing_input_mode
        if routing_input_mode == RoutingInputMode.FromLogits:
            _validate_logits_inputs(
                act, num_tokens, routing.num_experts, "TrtllmFp8BlockRunner"
            )
            routing_logits = act.routing_logits
            routing_bias = act.routing_bias
            # FP8 infers the routing mode from the expert-index tensor: a
            # non-empty 2D tensor means PackedPrecomputed and suppresses
            # routing_logits. Empty placeholders select FromLogits, matching
            # the canonical trtllm_fp8_block_scale_moe wrapper.
            topk_ids = act.hidden_states_q.new_empty((0,), dtype=torch.int32)
            expert_weights = act.hidden_states_q.new_empty((0,), dtype=torch.bfloat16)
        elif routing_input_mode == RoutingInputMode.PackedPrecomputed:
            _validate_prerouted_inputs(
                act, num_tokens, routing.top_k, "TrtllmFp8BlockRunner"
            )
            routing_logits = None
            routing_bias = None
            weight_bits = (
                act.topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32)
            )
            topk_ids = (act.topk_ids << 16) | (weight_bits & 0xFFFF)
            expert_weights = act.topk_weights.new_empty(
                (num_tokens, routing.top_k), dtype=torch.bfloat16
            )
        else:
            raise NotImplementedError(
                "TrtllmFp8BlockRunner supports only FromLogits and "
                "PackedPrecomputed routing."
            )

        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=act.hidden_states_q,
            hidden_states_scale=hidden_states_scale,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )
        self._static_kwargs = dict(
            routing_input_mode=routing_input_mode,
            routing_bias=routing_bias,
            gemm1_weights=view["gemm1_weights"],
            gemm1_weights_scale=view["gemm1_weights_scale"],
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=view["gemm2_weights"],
            gemm2_weights_scale=view["gemm2_weights_scale"],
            num_experts=routing.num_experts,
            num_fused_shared_experts=0,
            n_group=routing.n_group,
            topk_group=routing.topk_group,
            local_expert_offset=self._local_expert_offset,
            routed_scaling_factor=routing.routed_scaling_factor,
            routing_method_type=int(routing.method),
            use_shuffled_weight=self._use_shuffled_weight,
            weight_layout=int(WeightLayout.MajorK),
            do_finalize=True,
            enable_pdl=self._enable_pdl,
            # Matches the legacy block-FP8 FromLogits wrapper. Pre-routed
            # execution ignores this flag because weights are already final.
            norm_topk_prob=True,
            routing_replay_out=None,
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
        return hash(("trtllm_fp8_block", self._variant))


# ---------------------------------------------------------------------------
# TRTLLM per-tensor FP8 runner — E4M3 activations/weights, FromLogits only
# ---------------------------------------------------------------------------


class TrtllmFp8PerTensorRunner(MoERunner):
    """Per-tensor-FP8 adapter over the canonical trtllm-gen ``MoERunner``.

    The kernel consumes prequantized E4M3 activations and weights. Its calibrated
    activation/weight multipliers are folded into three per-expert FP32 epilogue
    scale vectors, so ``MoEActivationPack.hidden_states_scale`` remains ``None``.
    """

    backend_key = "trtllm_fp8_per_tensor"
    supported_routing_modes = (RoutingInputMode.FromLogits,)
    supported_quant_variants = (QuantVariant.FP8PerTensor,)

    def check_support(self) -> None:
        super().check_support()
        from ..tllm_enums import RoutingMethodType

        if self.config.activation.type is not ActivationType.Swiglu:
            raise NotImplementedError(
                f"{type(self).__name__} supports only the Swiglu activation."
            )
        if not self.config.execution.do_finalize:
            raise NotImplementedError(
                f"{type(self).__name__} supports only do_finalize=True."
            )
        if (
            self.config.routing.method is RoutingMethodType.Llama4
            and self.config.routing.top_k != 1
        ):
            raise ValueError(
                f"{type(self).__name__} requires top_k=1 for Llama4 routing."
            )

    def __init__(self, config: MoEConfig, device: torch.device):
        from ..tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
        from ..utils import device_support_pdl
        from .core import get_trtllm_moe_sm100_module

        self.config = config
        self.device = device
        self._module = get_trtllm_moe_sm100_module()
        self._dtype_act = DtypeTrtllmGen.E4m3
        self._dtype_weights = DtypeTrtllmGen.E4m3
        self._fp8_quantization_type = Fp8QuantizationType.NoneFp8

        routing = config.routing
        experts = config.experts
        execution = config.execution
        self._num_local_experts = experts.local_num_experts or routing.num_experts
        self._local_expert_offset = experts.local_expert_offset
        self._intermediate_size = experts.intermediate_size
        self._activation_type = int(config.activation.type)
        self._tune_max_num_tokens = execution.tune_max_num_tokens

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
            weight_layout=int(WeightLayout.MajorK),
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

    def _validate_tensors(
        self, act: MoEActivationPack, view: dict, hidden_size: int
    ) -> None:
        if act.hidden_states_q.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"{type(self).__name__} requires float8_e4m3fn hidden_states_q, "
                f"got {act.hidden_states_q.dtype}."
            )
        if act.hidden_states_scale is not None:
            raise ValueError(
                f"{type(self).__name__} requires hidden_states_scale=None; "
                "the calibrated input scale is folded into epilogue scales."
            )

        for name, expected in (
            (
                "gemm1_weights",
                (
                    self._num_local_experts,
                    2 * self._intermediate_size,
                    hidden_size,
                ),
            ),
            (
                "gemm2_weights",
                (
                    self._num_local_experts,
                    hidden_size,
                    self._intermediate_size,
                ),
            ),
        ):
            tensor = view[name]
            if tensor.dtype != torch.float8_e4m3fn or tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must be float8_e4m3fn with shape {expected}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}."
                )
        for name in (
            "output1_scales_scalar",
            "output1_scales_gate_scalar",
            "output2_scales_scalar",
        ):
            tensor = view[name]
            expected_scale_shape = (self._num_local_experts,)
            if (
                tensor.dtype != torch.float32
                or tuple(tensor.shape) != expected_scale_shape
            ):
                raise ValueError(
                    f"{name} must be float32 with shape {expected_scale_shape}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}."
                )

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        from ..tllm_enums import RoutingMethodType
        from .core import MoeRunnerInputs

        view = weights.get_view(self.backend_key)
        routing = self.config.routing
        num_tokens, hidden_size = act.hidden_states_q.shape
        _validate_logits_inputs(
            act, num_tokens, routing.num_experts, type(self).__name__
        )
        self._validate_tensors(act, view, hidden_size)

        routing_logits = act.routing_logits
        output = act.hidden_states_q.new_empty(
            (num_tokens, hidden_size), dtype=torch.bfloat16
        )
        # These buffers are autotuner inputs and routing-kernel outputs. The
        # per-tensor FFI itself selects FromLogits from its non-optional logits.
        topk_ids = act.hidden_states_q.new_empty(
            (num_tokens, routing.top_k), dtype=torch.int32
        )
        expert_weights = routing_logits.new_empty((num_tokens, routing.top_k))
        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=act.hidden_states_q,
            hidden_states_scale=None,
            gemm1_lora_delta=None,
            per_token_scale=None,
        )
        self._static_kwargs = dict(
            routing_bias=act.routing_bias,
            gemm1_weights=view["gemm1_weights"],
            output1_scales_scalar=view["output1_scales_scalar"],
            output1_scales_gate_scalar=view["output1_scales_gate_scalar"],
            gemm2_weights=view["gemm2_weights"],
            output2_scales_scalar=view["output2_scales_scalar"],
            num_experts=routing.num_experts,
            n_group=routing.n_group,
            topk_group=routing.topk_group,
            local_expert_offset=self._local_expert_offset,
            routed_scaling_factor=routing.routed_scaling_factor,
            use_routing_scales_on_input=(routing.method is RoutingMethodType.Llama4),
            routing_method_type=int(routing.method),
            do_finalize=True,
            enable_pdl=self._enable_pdl,
            norm_topk_prob=True,
            routing_replay_out=None,
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
        return hash(("trtllm_fp8_per_tensor",))


# ---------------------------------------------------------------------------
# TRTLLM BF16 routed runner — canonical trtllm-gen MoERunner, bf16 dtypes
# ---------------------------------------------------------------------------


class TrtllmBf16RoutedRunner(MoERunner):
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
    # The bf16 kernel supports FromLogits too; wiring it here is a follow-up.
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = (QuantVariant.BF16,)

    def check_support(self) -> None:
        super().check_support()
        if self.config.activation.type is not ActivationType.Swiglu:
            raise NotImplementedError(
                f"{type(self).__name__} supports only the Swiglu activation."
            )

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

        if act.routing_input_mode != RoutingInputMode.PackedPrecomputed:
            raise NotImplementedError(
                f"TrtllmBf16RoutedRunner does not support "
                f"routing_input_mode={act.routing_input_mode!r} "
                "(only PackedPrecomputed is wired)."
            )
        # Packed pre-routed top-k ids: (GLOBAL expert_id << 16) | bf16(weight).
        # The kernel expects GLOBAL ids and filters/maps them via the separately
        # passed ``local_expert_offset`` (mirrors trtllm_bf16_routed_moe in
        # tests/moe/test_trtllm_gen_routed_fused_moe.py). Do NOT pre-subtract the
        # offset: on ranks with local_expert_offset>0 that yields a local id below
        # the offset, which the kernel treats as non-local and skips → zero output.
        _validate_prerouted_inputs(
            act, num_tokens, routing.top_k, "TrtllmBf16RoutedRunner"
        )
        ids = act.topk_ids
        weight_bf16_bits = (
            act.topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        )
        topk_ids = (ids << 16) | (weight_bf16_bits & 0xFFFF)

        output = hidden_states.new_empty((num_tokens, hidden_size))
        expert_weights = act.topk_weights.new_empty(
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


# ---------------------------------------------------------------------------
# SM12x b12x runners — fixed tactic, existing wrapper delegation
# ---------------------------------------------------------------------------


class _B12xRunner(MoERunner):
    """Shared unified adapter over ``B12xMoEWrapper``."""

    backend_key: ClassVar[str] = ""
    required_weight_keys: ClassVar[tuple[str, ...]] = ()

    def check_support(self) -> None:
        super().check_support()

        from ..cute_dsl import is_cute_dsl_available
        from ..jit.cpp_ext import get_cuda_version
        from ..utils import get_compute_capability

        if get_cuda_version().major < 13:
            raise ValueError("b12x unified MoE requires CUDA 13 or later.")
        if not is_cute_dsl_available():
            raise RuntimeError("b12x unified MoE requires the CuTe DSL package.")
        major, minor = get_compute_capability(self.device)
        if (major, minor) not in ((12, 0), (12, 1)):
            raise RuntimeError(
                f"b12x unified MoE requires SM120 or SM121, got SM{major}{minor}."
            )

        experts = self.config.experts
        local_num_experts = experts.local_num_experts or self.config.routing.num_experts
        if experts.local_expert_offset != 0 or (
            local_num_experts != self.config.routing.num_experts
        ):
            raise NotImplementedError(
                "b12x unified MoE does not support expert parallelism."
            )
        if not self.config.execution.do_finalize:
            raise NotImplementedError("b12x unified MoE requires do_finalize=True.")

    def __init__(self, config: MoEConfig, device: torch.device):
        from .utils import get_b12x_activation_name

        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.activation = get_b12x_activation_name(config.activation.type)
        self.tuning_config = TuningConfig()
        self._prepared_weights: dict[str, torch.Tensor] | None = None
        self._inner: Any = None

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: Any) -> List[Any]:
        return [-1]

    def _get_quant_mode_name(self) -> str:
        if len(self.supported_quant_variants) != 1:
            raise ValueError(
                f"{type(self).__name__} must support exactly one quant variant."
            )
        quant_variant = self.supported_quant_variants[0]
        if quant_variant is QuantVariant.NVFP4:
            return "nvfp4"
        if quant_variant is QuantVariant.W4A16:
            return "w4a16"
        raise ValueError(f"Unsupported b12x quant variant: {quant_variant!r}.")

    def _validate_prepared_weights(
        self, prepared_weights: dict[str, torch.Tensor]
    ) -> None:
        missing = [
            key for key in self.required_weight_keys if key not in prepared_weights
        ]
        if missing:
            raise KeyError(
                f"{self.backend_key} prepared weights are missing {missing}."
            )
        if any(
            not isinstance(prepared_weights[key], torch.Tensor)
            for key in self.required_weight_keys
        ):
            raise TypeError(f"{self.backend_key} prepared weights must be tensors.")

    def _ensure_inner(self, hidden_size: int, num_tokens: int) -> None:
        if (
            self._inner is not None
            and hidden_size == self._inner.hidden_size
            and num_tokens <= self._inner.max_num_tokens
        ):
            return
        from .cute_dsl import B12xMoEWrapper

        self._inner = B12xMoEWrapper(
            num_experts=self.config.routing.num_experts,
            top_k=self.config.routing.top_k,
            hidden_size=hidden_size,
            intermediate_size=self.config.experts.intermediate_size,
            use_cuda_graph=True,
            max_num_tokens=max(1, num_tokens),
            device=self.device,
            activation=self.activation,
            quant_mode=self._get_quant_mode_name(),
            source_format="modelopt",
        )

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        v = weights.get_view(self.backend_key)
        self._validate_prepared_weights(v)
        first_weight = v[self.required_weight_keys[0]]
        if first_weight.shape[0] != self.config.routing.num_experts:
            raise ValueError(
                f"{self.backend_key} prepared {first_weight.shape[0]} "
                f"experts, expected {self.config.routing.num_experts}."
            )

        hidden_states = act.hidden_states_q
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(
                f"{self.backend_key} requires BF16 hidden_states, "
                f"got {hidden_states.dtype}."
            )
        if hidden_states.device != self.device:
            raise ValueError(
                f"hidden_states is on {hidden_states.device}, expected {self.device}."
            )
        if hidden_states.ndim != 2:
            raise ValueError(
                "b12x hidden_states must have shape [num_tokens, hidden_size]."
            )
        prepared_hidden_size = int(v["w1_weight"].shape[2]) * 2
        prepared_intermediate_size = int(v["w2_weight"].shape[2]) * 2
        expected_w1_rows = prepared_intermediate_size * (
            2 if self.config.activation.is_gated else 1
        )
        if v["w1_weight"].shape[1] != expected_w1_rows:
            raise ValueError(
                f"{self.backend_key} prepared weights are incompatible with "
                f"activation {self.config.activation.type!r}."
            )
        if prepared_intermediate_size != self.config.experts.intermediate_size:
            raise ValueError(
                f"{self.backend_key} prepared intermediate size "
                f"{prepared_intermediate_size} does not match config "
                f"{self.config.experts.intermediate_size}."
            )
        if hidden_states.shape[1] != prepared_hidden_size:
            raise ValueError(
                f"hidden size {hidden_states.shape[1]} does not match prepared "
                f"weights ({prepared_hidden_size})."
            )
        _validate_prerouted_inputs(
            act,
            hidden_states.shape[0],
            self.config.routing.top_k,
            type(self).__name__,
        )
        if act.topk_weights.dtype != torch.float32:
            raise TypeError("b12x topk_weights must use torch.float32.")
        self._prepared_weights = v
        self._ensure_inner(hidden_states.shape[1], hidden_states.shape[0])
        return [hidden_states, act.topk_ids, act.topk_weights]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        if tactic != -1:
            raise ValueError(f"{self.backend_key} supports only tactic -1.")
        if self._prepared_weights is None:
            raise RuntimeError("pack_inputs must be called before b12x forward.")
        if self._inner is None:
            raise RuntimeError("pack_inputs must initialize the b12x wrapper.")
        if len(inputs) != 3:
            raise ValueError("b12x runner expects [hidden, expert_ids, weights].")
        return self._inner.run(
            x=inputs[0],
            w1_weight=self._prepared_weights["w1_weight"],
            w1_weight_sf=self._prepared_weights["w1_weight_sf"],
            w1_alpha=self._prepared_weights["w1_alpha"],
            fc2_input_scale=self._prepared_weights.get("fc2_input_scale"),
            w2_weight=self._prepared_weights["w2_weight"],
            w2_weight_sf=self._prepared_weights["w2_weight_sf"],
            w2_alpha=self._prepared_weights["w2_alpha"],
            token_selected_experts=inputs[1],
            token_final_scales=inputs[2],
        )

    def __hash__(self):
        return hash((self.backend_key, self.config))


class B12xNvfp4Runner(_B12xRunner):
    """Unified SM120/SM121 adapter for b12x NVFP4/W4A4 MoE."""

    backend_key = "b12x_nvfp4"
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = (QuantVariant.NVFP4,)
    required_weight_keys = (
        "w1_weight",
        "w1_weight_sf",
        "w1_alpha",
        "w2_weight",
        "w2_weight_sf",
        "w2_alpha",
        "fc2_input_scale",
    )


class B12xW4A16Runner(_B12xRunner):
    """Unified SM120/SM121 adapter for b12x W4A16 MoE."""

    backend_key = "b12x_w4a16"
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = (QuantVariant.W4A16,)
    required_weight_keys = (
        "w1_weight",
        "w1_weight_sf",
        "w1_alpha",
        "w2_weight",
        "w2_weight_sf",
        "w2_alpha",
    )

    def check_support(self) -> None:
        super().check_support()
        if self.config.activation.type not in (
            ActivationType.Swiglu,
            ActivationType.Relu2,
        ):
            raise NotImplementedError(
                f"{type(self).__name__} supports only the Swiglu or Relu2 activation."
            )
