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
``CuteDslFusedMoERunner`` and trtllm-gen's ``core.MoERunner``) so the
fragile backend-specific kernel-launch code lives in exactly one place.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional

import torch

from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
)
from ..utils import next_positive_power_of_2, round_up
from .api import (
    _CUTLASS_BF16_ARCHS,
    _CUTLASS_FP8_ARCHS,
    _CUTLASS_FP8_BLOCK_ARCHS,
    _CUTLASS_HUMMING_ARCHS,
    _CUTLASS_MXFP8_ARCHS,
    _CUTLASS_MXFP8_MXFP4_ARCHS,
    _CUTLASS_NVFP4_ARCHS,
    _CUTLASS_W4A16_ARCHS,
    _CUTLASS_W4A8_ARCHS,
    _CUTILE_BF16_ARCHS,
    _CUTILE_NVFP4_ARCHS,
    # Typed activation values
    ActivationConfig,
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    ReLU,
    ReLU2,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
    # Unified config and pack types
    ActivationType,
    MoEActivationPack,
    MoEConfig,
    MoEWeightPack,
    QuantVariant,
    RoutingInputMode,
)
from .utils import (
    make_hybrid_bucket_mapper,
    map_to_hybrid_bucket,
)


_CUTLASS_SEMANTIC_ACTIVATIONS: tuple[type[ActivationConfig], ...] = (
    SwiGLU,
    SwiGLUStep,
    GeGLU,
    GeGLUTanh,
    ReLU2,
    SiTU,
    Identity,
    GELU,
    ReLU,
    SiLU,
)

_CUTILE_INT32_INDEX_LIMIT = 1 << 31


def _validate_cutile_int32_routing(
    runner: str, num_assignments: int, max_padded_assignments: int
) -> None:
    """Keep routing IDs and positions within their intentional int32 storage."""
    if max(num_assignments, max_padded_assignments) >= _CUTILE_INT32_INDEX_LIMIT:
        raise NotImplementedError(
            f"{runner} requires fewer than 2^31 routed and padded assignments; "
            f"got assignments={num_assignments}, padded={max_padded_assignments}."
        )


def _validate_pack_devices(act: MoEActivationPack, runner: str) -> None:
    """Recheck pack tensor placement at the mutable runner boundary."""
    expected = act.hidden_states_q.device
    for name in (
        "hidden_states_scale",
        "topk_ids",
        "topk_weights",
        "per_token_scale",
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
    act: MoEActivationPack,
    num_tokens: int,
    top_k: int,
    runner: str,
    *,
    allowed_weights_dtypes: tuple[torch.dtype, ...] | None = None,
    require_contiguous: bool = False,
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
            f"{runner}: precomputed routing requires topk_ids + topk_weights."
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
            "(torch.topk returns int64 — cast before constructing the pack)."
        )
    if (
        allowed_weights_dtypes is not None
        and act.topk_weights.dtype not in allowed_weights_dtypes
    ):
        allowed = " or ".join(str(dtype) for dtype in allowed_weights_dtypes)
        raise TypeError(
            f"{runner}: topk_weights must be {allowed}, got {act.topk_weights.dtype}."
        )
    if require_contiguous and (
        not act.topk_ids.is_contiguous() or not act.topk_weights.is_contiguous()
    ):
        raise ValueError(
            f"{runner}: unpacked topk_ids/topk_weights must be contiguous."
        )


def _pack_prerouted_topk_ids(act: MoEActivationPack) -> torch.Tensor:
    """Pack global expert IDs and BF16 weight bits for TRTLLM routed APIs."""
    if act.topk_ids is None or act.topk_weights is None:
        raise ValueError("Packed routing requires topk_ids + topk_weights.")
    weight_bits = act.topk_weights.to(torch.bfloat16).view(torch.int16).to(torch.int32)
    return ((act.topk_ids << 16) | (weight_bits & 0xFFFF)).contiguous()


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
    if not logits.is_contiguous():
        raise ValueError(f"{runner}: routing_logits must be contiguous.")
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
        if not act.routing_bias.is_contiguous():
            raise ValueError(f"{runner}: routing_bias must be contiguous.")


def _validate_optional_gemm1_activation_params(
    view: dict,
    num_expert_rows: int,
    device: torch.device,
    runner: str,
) -> None:
    """Runner-boundary validation for the optional SwiGLU OA weight-side params.

    ``gemm1_alpha`` / ``gemm1_beta`` / ``gemm1_clamp_limit`` are independent: any
    subset may be absent, and absent means the neutral value (alpha=1, beta=0, no
    clamp).  The launcher re-checks all of this, but failing here names the runner
    and the offending key instead of surfacing a bare TVM-FFI ICHECK.
    """
    for key in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
        tensor = view.get(key)
        if tensor is None:
            continue
        if tensor.device != device:
            raise ValueError(
                f"{runner}: {key} is on {tensor.device}, expected {device}."
            )
        if tensor.dtype != torch.float32:
            raise TypeError(f"{runner}: {key} must be float32, got {tensor.dtype}.")
        if tuple(tensor.shape) != (num_expert_rows,):
            raise ValueError(
                f"{runner}: {key} shape {tuple(tensor.shape)} "
                f"!= expected ({num_expert_rows},)."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{runner}: {key} must be contiguous.")


def _cute_dsl_activation_kwargs(activation: ActivationConfig) -> dict[str, Any]:
    """Translate typed activation values to the existing CuTe-DSL scalar ABI."""
    if isinstance(activation, SwiGLU):
        return {
            "activation_type": int(activation.type),
            "swiglu_alpha": activation.alpha,
            "swiglu_beta": activation.beta,
            "swiglu_limit": activation.limit,
        }
    if isinstance(activation, SiTU):
        return {
            # CuTeDSL encodes SiTU as SwiGLU with a non-null situ_beta.
            "activation_type": int(ActivationType.Swiglu),
            "situ_beta": activation.gate_scale,
            # None means an unclamped linear branch.
            "situ_linear_beta": activation.linear_scale,
        }
    return {"activation_type": int(activation.type)}


def _validate_prepared_activation_params(
    view: dict[str, Optional[torch.Tensor]],
    activation: ActivationConfig,
    runner: str,
) -> None:
    """Validate activation scalars in a backend-prepared weight view."""
    required: tuple[str, ...] = ()
    if isinstance(activation, SwiGLU):
        # Omitted fields use neutral launcher defaults, so require only values
        # that differ from the typed default.
        default = SwiGLU()
        if activation.alpha != default.alpha:
            required += ("gemm1_alpha",)
        if activation.beta != default.beta:
            required += ("gemm1_beta",)
        if activation.limit != default.limit:
            required += ("gemm1_clamp_limit",)
    elif isinstance(activation, SiTU):
        # TRTLLM reuses gemm1_alpha/beta, whose null SiTU defaults are 1/1
        # rather than the typed 4/25. Require explicit tensors even at default.
        # linear_scale=None is rejected earlier because this ABI cannot encode it.
        required = ("gemm1_alpha",)
        if activation.linear_scale is not None:
            required += ("gemm1_beta",)
        if activation.clamp_limit is not None:
            required += ("gemm1_clamp_limit",)
    elif not activation.is_gated:
        # Non-gated kernels ignore these tensors; reject them rather than
        # silently accepting ineffective overrides. Whether a *gated*
        # activation accepts them is per-backend (FP4 GeGLU does, FP8 block and
        # BF16 restrict to Swiglu), so that stays a runner-level decision.
        supplied = [
            name
            for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")
            if view.get(name) is not None
        ]
        if supplied:
            raise ValueError(
                f"{runner}: {type(activation).__name__} does not consume "
                f"{supplied}; these per-expert overrides apply to gated "
                "activations only and would be ignored."
            )
    # None is equivalent to an absent launcher tensor.
    missing = [name for name in required if view.get(name) is None]
    if missing:
        raise ValueError(
            f"{runner}: prepared weights are missing activation parameters {missing}; "
            "prepare the backend view with the same typed activation or provide "
            "explicit per-expert overrides."
        )


@functools.lru_cache(maxsize=None)
def _cutlass_activation_required_keys(activation: ActivationConfig) -> frozenset[str]:
    """Return non-null CUTLASS scalar keys required by ``activation``."""
    # A zero-expert probe keeps this in sync with the materializer without
    # depending on expert count or device.
    probe = _cutlass_activation_params(activation, 0, torch.device("cpu"))
    return frozenset(name for name, value in probe.items() if value is not None)


def _cutlass_activation_params(
    activation: ActivationConfig,
    num_experts: int,
    device: torch.device,
) -> dict[str, torch.Tensor | None]:
    """Materialize CUTLASS's optional per-expert activation tensors."""
    params: dict[str, torch.Tensor | None] = {
        "swiglu_alpha": None,
        "swiglu_beta": None,
        "swiglu_limit": None,
        "situ_beta": None,
        "situ_linear_beta": None,
    }
    if isinstance(activation, SiTU):
        if activation.linear_scale is None:
            # Normal runner lifecycle rejects this in check_support; keep
            # direct helper calls explicit as well.
            raise NotImplementedError(
                "CUTLASS cannot express SiTU(linear_scale=None); its ABI "
                "has no unclamped linear-branch encoding."
            )
        # CUTLASS native defaults match SiTU(), so only custom values need tensors.
        if activation != SiTU():
            params["situ_beta"] = torch.full(
                (num_experts,),
                activation.gate_scale,
                dtype=torch.float32,
                device=device,
            )
            params["situ_linear_beta"] = torch.full(
                (num_experts,),
                activation.linear_scale,
                dtype=torch.float32,
                device=device,
            )
        return params
    if isinstance(activation, SwiGLU) and activation != SwiGLU():
        params = {
            "swiglu_alpha": torch.full(
                (num_experts,), activation.alpha, dtype=torch.float32, device=device
            ),
            "swiglu_beta": torch.full(
                (num_experts,), activation.beta, dtype=torch.float32, device=device
            ),
            "swiglu_limit": torch.full(
                (num_experts,), activation.limit, dtype=torch.float32, device=device
            ),
        }
    elif isinstance(activation, SwiGLUStep):
        params["swiglu_limit"] = torch.full(
            (num_experts,), activation.limit, dtype=torch.float32, device=device
        )
    return params


class MoERunner(TunableRunner):
    """Unified MoE runner lifecycle: validate, build once, then execute.

    Concrete runners implement ``_check_support()`` and ``_build()``. Keeping
    the public methods here ensures a failed support check cannot authorize a
    build and execution cannot silently initialize backend resources.
    """

    backend_key: ClassVar[str] = ""
    supported_routing_modes: tuple[RoutingInputMode, ...] = ()
    supported_quant_variants: ClassVar[tuple[QuantVariant, ...]] = ()
    # Default to no activations; each concrete runner must declare its support.
    supported_activation_classes: ClassVar[tuple[type[ActivationConfig], ...]] = ()
    supported_activation_classes_by_quant: ClassVar[
        dict[QuantVariant, tuple[type[ActivationConfig], ...]]
    ] = {}
    # Set to True only after S is wired through validation and launch.
    supports_fused_shared_experts: ClassVar[bool] = False
    # Cleared by backends whose kernels cannot map global expert ids onto a
    # local shard (local_expert_offset / local_num_experts != num_experts).
    supports_expert_parallelism: ClassVar[bool] = True

    config: MoEConfig

    def __init__(self) -> None:
        self._support_checked = False
        self._built = False

    def check_support(self) -> None:
        self._support_checked = False
        self._check_support()
        self._support_checked = True

    def _check_support(self) -> None:
        """Raise if the initialized runner cannot execute its configuration."""
        variant = self.config.quant.variant
        if variant not in self.supported_quant_variants:
            raise NotImplementedError(
                f"{type(self).__name__} does not support QuantVariant.{variant.name}."
            )
        if self.supported_activation_classes_by_quant:
            # Strict lookup: a runner that declares per-quant capabilities must
            # declare them for every variant it accepts. Falling back to the
            # permissive class default would silently admit every activation on
            # a newly added variant.
            try:
                supported_activations = self.supported_activation_classes_by_quant[
                    variant
                ]
            except KeyError:
                raise NotImplementedError(
                    f"{type(self).__name__} declares per-quantization activation "
                    f"support but has no entry for QuantVariant.{variant.name}; "
                    "add one to supported_activation_classes_by_quant."
                ) from None
        else:
            supported_activations = self.supported_activation_classes
        if not supported_activations:
            raise NotImplementedError(
                f"{type(self).__name__} declares no supported activation classes."
            )
        if not isinstance(self.config.activation, supported_activations):
            names = ", ".join(cls.__name__ for cls in supported_activations)
            raise NotImplementedError(
                f"{type(self).__name__} does not support "
                f"{type(self.config.activation).__name__} for QuantVariant."
                f"{variant.name}; supported activations are {names}."
            )
        self._assert_shared_experts_supported()
        self._assert_expert_parallelism_supported()
        self._check_activation_parameters()

    def _check_activation_parameters(self) -> None:
        """Reject typed scalar values a backend would otherwise silently drop."""

    def _assert_shared_experts_supported(self) -> None:
        """Reject S > 0 for backends that have not opted in."""
        s = self.config.experts.num_fused_shared_experts
        if s > 0 and not self.supports_fused_shared_experts:
            raise NotImplementedError(
                f"{type(self).__name__} does not support fused shared experts "
                f"(num_fused_shared_experts={s})."
            )

    def _assert_expert_parallelism_supported(self) -> None:
        """Reject EP shards for backends that cannot compute a local expert subset."""
        if self.supports_expert_parallelism:
            return
        experts = self.config.experts
        local_num_experts = experts.local_num_experts or self.config.routing.num_experts
        if experts.local_expert_offset != 0 or (
            local_num_experts != self.config.routing.num_experts
        ):
            raise NotImplementedError(
                f"{type(self).__name__} does not support expert parallelism "
                f"(local_expert_offset={experts.local_expert_offset}, "
                f"local_num_experts={local_num_experts} of "
                f"{self.config.routing.num_experts})."
            )

    def build(self) -> None:
        if getattr(self, "_built", False):
            return
        if not getattr(self, "_support_checked", False):
            raise RuntimeError(
                f"{type(self).__name__}.check_support() must succeed before build()."
            )
        self._build()
        self._built = True

    def _build(self) -> None:
        """Prepare shape-independent resources for a supported runner."""

    def _require_built(self) -> None:
        if not getattr(self, "_built", False):
            raise RuntimeError(
                f"{type(self).__name__}.build() must be called before execution."
            )

    # Anything the profiled tensor shapes cannot reveal has to be listed here.
    # One stable tuple feeds both __hash__ (in-memory) and the persisted key,
    # which excludes runner_hash. Shapes already in the profile are omitted.

    def _cache_key_extras(self) -> tuple:
        """Return stable tactic inputs absent from profiled tensor shapes."""
        routing = self.config.routing
        experts = self.config.experts
        local_num_experts = (
            experts.local_num_experts
            if experts.local_num_experts is not None
            else routing.num_experts
        )
        return (
            self.backend_key,
            # The persistent key uses str(), so prefer stable scalar values.
            self.config.quant.variant.name,
            int(routing.top_k),
            int(routing.num_experts),
            int(local_num_experts),
            int(experts.local_expert_offset),
            int(experts.intermediate_size),
            # S changes tactic enumeration but not profiled tensor shapes.
            int(experts.num_fused_shared_experts),
            int(self.config.activation.type),
            repr(self.config.activation),
            bool(self.config.finalize.do_finalize),
            # Routing shape affects expert-token distribution and tactic ranking.
            int(routing.method),
            routing.n_group,
            routing.topk_group,
            # Declared quant flags are keyed before runners begin consuming them.
            self.config.quant.per_token_scale,
            self.config.quant.swizzled_scale_factors,
        )

    def __hash__(self) -> int:
        return hash(self._cache_key_extras())

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        # Configuration-only, so synthesized profiling inputs use the same key.
        return self._cache_key_extras()


def _mxfp8_swizzled_act_sf_numel(num_tokens: int, hidden_size: int) -> int:
    """Byte count of a 128x4-swizzled MXFP8 activation scale buffer."""
    return round_up(num_tokens, 128) * round_up(hidden_size // 32, 4)


def _infer_mxfp8_swizzled_act_sf_numel(shapes: list) -> int:
    """ConstraintSpec callback: ``input_sf`` numel from hidden_states ``[M, H]``."""
    return _mxfp8_swizzled_act_sf_numel(shapes[1][0], shapes[1][1])


def _require_cutlass_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> None:
    if tensor.dtype is not dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}.")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape {tuple(tensor.shape)} != expected {shape}.")


# ---------------------------------------------------------------------------
# CUTLASS runners — dense BF16 and mixed-input W4A16
# ---------------------------------------------------------------------------


class _CutlassRunnerBase(MoERunner):
    """Shared launch, tuning, and workspace mechanics for CUTLASS MoE.

    Direct callers must invoke ``check_support()`` followed by ``build()``
    before ``pack_inputs()``, ``get_valid_tactics()``, or ``forward()``.
    ``MoELayer`` enforces this lifecycle when constructing its runners.

    A runner is a single-stream resource. Concurrent use or CUDA-graph replay
    on multiple streams requires one runner (or ``MoELayer``) per stream.
    """

    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    # Fail closed like MoERunner: every concrete CUTLASS runner declares its own
    # activations after proving the matching preparation geometry and numerical
    # coverage. A SwiGLU default would let a runner added later inherit support
    # it never validated instead of failing at check_support().
    supported_activation_classes: ClassVar[tuple[type[ActivationConfig], ...]] = ()
    supports_expert_parallelism = False
    _supported_archs: ClassVar[tuple[int, ...]]
    _x_dtype: ClassVar[torch.dtype] = torch.bfloat16
    _weight_dtype: ClassVar[torch.dtype]
    _use_w4_group_scaling: ClassVar[bool] = False
    _use_deepseek_fp8_block_scale: ClassVar[bool] = False
    _use_mxfp8_act_scaling: ClassVar[bool] = False
    _use_packed_weights: ClassVar[bool] = False
    _use_wfp4afp8_humming: ClassVar[bool] = False
    _required_weight_keys: ClassVar[tuple[str, ...]]
    _expected_num_inputs: ClassVar[int]
    # Keep the best N tactics per GEMM stage, then return their Cartesian
    # product as compound candidates for the outer end-to-end autotuner. N=1
    # preserves the legacy independent-winner behavior.
    _num_top_tactics_per_stage: ClassVar[int] = 2

    def _check_support(self) -> None:
        super()._check_support()
        if not self.config.finalize.do_finalize:
            raise NotImplementedError(
                f"{type(self).__name__} requires do_finalize=True."
            )
        activation = self.config.activation
        if isinstance(activation, SiTU):
            # The CUTLASS SiTU ABI carries situ_beta and situ_linear_beta only:
            # there is no unclamped-linear encoding and no clamp channel.
            if activation.linear_scale is None:
                raise NotImplementedError(
                    f"{type(self).__name__} cannot express "
                    "SiTU(linear_scale=None); the CUTLASS ABI has no unclamped "
                    "linear-branch encoding."
                )
            if activation.clamp_limit is not None:
                raise NotImplementedError(
                    f"{type(self).__name__} cannot express a SiTU clamp_limit; "
                    "the CUTLASS ABI exposes no clamp channel."
                )
        if self._device_arch not in self._supported_archs:
            raise RuntimeError(
                f"{type(self).__name__} does not support "
                f"SM{self._device_arch}; supported architectures are "
                f"{self._supported_archs}."
            )
        if self._use_mxfp8_act_scaling and (
            self.config.quant.swizzled_scale_factors is False
        ):
            raise NotImplementedError(
                f"{type(self).__name__} requires swizzled MXFP8 input_sf; "
                "linear scales (swizzled_scale_factors=False) are not supported."
            )
        if self._use_deepseek_fp8_block_scale:
            from ..jit.cpp_ext import is_cuda_version_at_least

            if not is_cuda_version_at_least("12.8"):
                raise NotImplementedError(
                    "FP8 block scaling requires CUDA 12.8 or newer."
                )

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        from ..utils import device_support_pdl, get_compute_capability

        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"{type(self).__name__} requires CUDA, got {device}.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        major, minor = get_compute_capability(self.device)
        self._device_arch = major * 10 + minor

        enable_pdl = config.execution.enable_pdl
        if enable_pdl is None:
            enable_pdl = device_support_pdl(self.device)
        self._enable_pdl = enable_pdl
        self._use_fused_finalize = True
        self._inner: Any = None

        # pack_inputs replaces this with the current MoELayer token bucket.
        self.tuning_config = TuningConfig()
        # Retain geometrically sized buffers because captured CUDA graphs keep
        # their raw pointers. Deterministic capacities make a request shape
        # select the same pointer regardless of allocation history.
        self._workspace_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._workspace: torch.Tensor | None = None
        self._workspace_num_tokens = 0
        self._workspace_hidden_size: int | None = None

    def _build(self) -> None:
        """Load the CUTLASS module and create the inner runner."""
        from .core import get_cutlass_fused_moe_module

        with torch.cuda.device(self.device):
            module = get_cutlass_fused_moe_module(str(self._device_arch))
            self._inner = module.MoERunner(
                x_dtype=self._x_dtype,
                weight_dtype=self._weight_dtype,
                output_dtype=torch.bfloat16,
                top_k=self.config.routing.top_k,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                cluster_size=1,
                cluster_rank=0,
                enable_alltoall=False,
                use_deepseek_fp8_block_scale=self._use_deepseek_fp8_block_scale,
                use_w4_group_scaling=self._use_w4_group_scaling,
                use_mxfp8_act_scaling=self._use_mxfp8_act_scaling,
                min_latency_mode=False,
                enable_pdl=self._enable_pdl,
                activation_type=self.config.activation.type,
                use_packed_weights=self._use_packed_weights,
                use_fused_finalize=self._use_fused_finalize,
                use_wfp4afp8_humming=self._use_wfp4afp8_humming,
            )
            activation = self.config.activation
            num_experts = self.config.routing.num_experts
            self._activation_params = _cutlass_activation_params(
                activation, num_experts, self.device
            )
            self._config_activation_params = dict(self._activation_params)

    def _resolve_activation_params(
        self, view: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | None]:
        """Apply optional per-expert weight-view overrides to typed scalars."""
        # Rebuild an incomplete cache, then retain its tensors so CUDA graphs
        # keep stable pointers across repeated packing.
        config_params = getattr(self, "_config_activation_params", None)
        required = _cutlass_activation_required_keys(self.config.activation)
        if config_params is None or any(
            config_params.get(name) is None for name in required
        ):
            config_params = _cutlass_activation_params(
                self.config.activation,
                self.config.routing.num_experts,
                self.device,
            )
            self._config_activation_params = config_params
        params = dict(config_params)
        activation = self.config.activation
        # SiTU carries its own native CUTLASS keys rather than the gemm1_*
        # spelling, so the accepted override names depend on the activation.
        # Reading only the gemm1_* set would drop a supplied situ_beta on the
        # floor while rejecting the gemm1_* names the caller does not have.
        if isinstance(activation, SiTU):
            aliases = {
                "situ_beta": "situ_beta",
                "situ_linear_beta": "situ_linear_beta",
            }
        else:
            aliases = {
                "gemm1_alpha": "swiglu_alpha",
                "gemm1_beta": "swiglu_beta",
                "gemm1_clamp_limit": "swiglu_limit",
            }
        present = [name for name in aliases if name in view]
        if isinstance(activation, SwiGLUStep):
            ignored = [name for name in ("gemm1_alpha", "gemm1_beta") if name in view]
            if ignored:
                raise ValueError(
                    f"{type(self).__name__}: SwiGLUStep does not consume "
                    f"per-expert overrides {ignored}; only gemm1_clamp_limit is valid."
                )
        if present and not isinstance(activation, (SwiGLU, SwiGLUStep, SiTU)):
            raise ValueError(
                f"{type(self).__name__}: per-expert activation overrides {present} "
                f"are invalid for {type(activation).__name__}."
            )
        # An override spelled for a different activation is a mistake worth
        # naming, not something to ignore -- in either direction.
        foreign: tuple[str, ...]
        correct: str
        if isinstance(activation, SiTU):
            foreign, correct = (
                ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"),
                "situ_beta / situ_linear_beta",
            )
        else:
            foreign, correct = (
                ("situ_beta", "situ_linear_beta"),
                "gemm1_alpha / gemm1_beta / gemm1_clamp_limit",
            )
        supplied = [name for name in foreign if name in view]
        if supplied:
            raise ValueError(
                f"{type(self).__name__}: {type(activation).__name__} takes "
                f"per-expert overrides as {correct}, not {supplied}."
            )
        expected_shape = (self.config.routing.num_experts,)
        for source, destination in aliases.items():
            if source not in view:
                continue
            tensor = view[source]
            if tensor.dtype is not torch.float32:
                raise TypeError(f"{source} must use torch.float32, got {tensor.dtype}.")
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"{source} must have shape {expected_shape}, got {tuple(tensor.shape)}."
                )
            if tensor.device != self.device:
                raise ValueError(
                    f"{source} is on {tensor.device}, expected {self.device}."
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{source} must be contiguous.")
            params[destination] = tensor
        return params

    def _prepare_tuning_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Populate synthesized routing inputs with a valid balanced pattern."""
        num_tokens = inputs[1].shape[0]
        self._ensure_workspace(num_tokens, inputs[1].shape[1])
        top_k = self.config.routing.top_k
        num_experts = self.config.routing.num_experts
        token_offsets = torch.arange(
            num_tokens, dtype=torch.int32, device=inputs[2].device
        ).unsqueeze(1)
        slots = torch.arange(
            top_k, dtype=torch.int32, device=inputs[2].device
        ).unsqueeze(0)
        inputs[2].copy_((token_offsets * top_k + slots) % num_experts)
        inputs[3].fill_(1.0 / top_k)
        if self._use_mxfp8_act_scaling:
            hidden_states = inputs[1]
            inputs[-1] = torch.full(
                (
                    _mxfp8_swizzled_act_sf_numel(
                        hidden_states.shape[0], hidden_states.shape[1]
                    ),
                ),
                127,
                dtype=torch.uint8,
                device=hidden_states.device,
            )
        return inputs

    def get_valid_tactics(self, inputs: List[torch.Tensor], _profile: Any) -> List[Any]:
        self._require_built()
        self._validate_input_count(inputs)
        # The two GEMMs have independent tactic spaces. Preserve the legacy
        # O(n1+n2) stage search, keep the best N tactics per stage, then let the
        # outer unified tuner profile only the N² compound pairs end-to-end.
        # FIXME: Prefer a first-class factorized/multi-stage autotuner API so
        # runners do not need to nest stage ranking inside get_valid_tactics().
        tuner = AutoTuner.get()
        profile_inputs = [inputs[1], inputs[4], None, inputs[5], None]
        stage_tuning_config = TuningConfig()
        num_top_tactics = self._num_top_tactics_per_stage
        try:
            self._inner.gemm_idx_for_tuning = 1
            gemm1_tactics = tuner.rank_tactics(
                f"moe_{self.backend_key}_sm{self._device_arch}_gemm1",
                [self._inner],
                stage_tuning_config,
                profile_inputs,
                k=num_top_tactics,
                gemm_idx=1,
            )
            self._inner.gemm_idx_for_tuning = 2
            gemm2_tactics = tuner.rank_tactics(
                f"moe_{self.backend_key}_sm{self._device_arch}_gemm2",
                [self._inner],
                stage_tuning_config,
                profile_inputs,
                k=num_top_tactics,
                gemm_idx=2,
            )
        finally:
            self._inner.gemm_idx_for_tuning = None
        pairs: List[Any] = [
            (int(gemm1), int(gemm2))
            for gemm1 in gemm1_tactics
            for gemm2 in gemm2_tactics
        ]
        return pairs if pairs else [-1]

    def _ensure_workspace(self, num_tokens: int, hidden_size: int) -> None:
        max_num_tokens = self.config.execution.tune_max_num_tokens
        if num_tokens > max_num_tokens:
            raise ValueError(
                f"workspace num_tokens={num_tokens} exceeds tune_max_num_tokens="
                f"{max_num_tokens}."
            )
        if (
            self._workspace_hidden_size is not None
            and hidden_size != self._workspace_hidden_size
        ):
            raise ValueError(
                "CUTLASS runner hidden_size changed after workspace allocation: "
                f"{self._workspace_hidden_size} -> {hidden_size}."
            )

        capacity = min(next_positive_power_of_2(num_tokens), max_num_tokens)
        key = (capacity, hidden_size)
        workspace = self._workspace_cache.get(key)
        if workspace is not None:
            self._workspace = workspace
            self._workspace_num_tokens = capacity
            return
        from .core import cutlass_fused_moe_workspace_size

        size = cutlass_fused_moe_workspace_size(
            capacity,
            hidden_size,
            self.config.experts.intermediate_size,
            self.config.routing.num_experts,
            self.config.routing.top_k,
            x_dtype=self._x_dtype,
            weight_dtype=self._weight_dtype,
            output_dtype=torch.bfloat16,
            activation_type=self.config.activation.type,
            use_deepseek_fp8_block_scale=self._use_deepseek_fp8_block_scale,
            use_w4_group_scaling=self._use_w4_group_scaling,
            use_mxfp8_act_scaling=self._use_mxfp8_act_scaling,
            use_fused_finalize=self._use_fused_finalize,
            use_packed_weights=self._use_packed_weights,
            use_wfp4afp8_humming=self._use_wfp4afp8_humming,
            device=self.device,
        )
        workspace = torch.empty(size, dtype=torch.uint8, device=self.device)
        self._workspace_cache[key] = workspace
        self._workspace = workspace
        self._workspace_num_tokens = capacity
        self._workspace_hidden_size = hidden_size

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        if act.routing_input_mode is not RoutingInputMode.PackedPrecomputed:
            raise NotImplementedError(
                f"{type(self).__name__} supports only PackedPrecomputed routing."
            )
        hidden_states = act.hidden_states_q
        if hidden_states.ndim != 2 or hidden_states.dtype is not self._x_dtype:
            raise TypeError(
                f"{type(self).__name__} requires 2D {self._x_dtype} hidden_states_q, "
                f"got shape={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}."
            )
        if hidden_states.device != self.device:
            raise ValueError(
                f"hidden_states_q is on {hidden_states.device}, expected {self.device}."
            )
        self._validate_activation_scale(act)

        num_tokens, hidden_size = hidden_states.shape
        ceiling = self.config.execution.tune_max_num_tokens
        if num_tokens > ceiling:
            raise ValueError(
                f"num_tokens={num_tokens} exceeds tune_max_num_tokens={ceiling}. "
                "Reconstruct the runner with a larger ceiling."
            )
        _validate_prerouted_inputs(
            act,
            num_tokens,
            self.config.routing.top_k,
            type(self).__name__,
            allowed_weights_dtypes=(torch.float32,),
            require_contiguous=True,
        )

        view = weights.get_view(self.backend_key)
        missing = [key for key in self._required_weight_keys if key not in view]
        if missing:
            raise KeyError(
                f"{self.backend_key} prepared weights are missing {missing}."
            )
        weight_inputs = self._pack_weight_inputs(view, hidden_size)
        scale_inputs = self._pack_activation_scale_inputs(act)
        self._activation_params = self._resolve_activation_params(view)

        # Token-dynamic dims are only the packed prerouted buffers (output,
        # hidden, topk_ids, topk_weights). Per-tensor FP8 dequant is 0-dim;
        # MXFP8 input_sf is a swizzled 1-D buffer resized by ConstraintSpec.
        # Sniffing shape[0] == num_tokens treated a (1,) scale at M=1 as
        # token-dynamic and let autotune replace it with a bucket-sized tensor.
        input_idxs: tuple[int, ...] = (0, 1, 2, 3)
        dim_idxs: tuple[int, ...] = (0, 0, 0, 0)

        bucket = map_to_hybrid_bucket(
            num_tokens, self.config.execution.tune_max_num_tokens
        )
        constraint_specs: tuple[ConstraintSpec, ...] = ()
        if self._use_mxfp8_act_scaling:
            constraint_specs = (
                ConstraintSpec(
                    4 + len(weight_inputs),
                    0,
                    _infer_mxfp8_swizzled_act_sf_numel,
                ),
            )
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=input_idxs,
                    dim_idx=dim_idxs,
                    gen_tuning_buckets=(bucket,),
                    map_to_tuning_buckets=make_hybrid_bucket_mapper(
                        self.config.execution.tune_max_num_tokens
                    ),
                ),
            ),
            constraint_specs=constraint_specs,
            use_cuda_graph=True,
            inputs_pre_hook=self._prepare_tuning_inputs,
        )
        self._ensure_workspace(bucket, hidden_size)
        output = torch.empty(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        return [
            output,
            hidden_states,
            act.topk_ids,
            act.topk_weights,
            *weight_inputs,
            *scale_inputs,
        ]

    def _validate_activation_scale(self, act: MoEActivationPack) -> None:
        if self._use_mxfp8_act_scaling:
            scale = act.hidden_states_scale
            num_tokens, hidden_size = act.hidden_states_q.shape
            expected = _mxfp8_swizzled_act_sf_numel(num_tokens, hidden_size)
            if (
                scale is None
                or scale.dtype is not torch.uint8
                or scale.numel() != expected
                or not scale.is_contiguous()
            ):
                got = (
                    None
                    if scale is None
                    else (scale.dtype, tuple(scale.shape), scale.is_contiguous())
                )
                raise ValueError(
                    f"{type(self).__name__} requires a contiguous uint8 swizzled "
                    f"input_sf with {expected} elements for M={num_tokens}, "
                    f"H={hidden_size}; got {got}."
                )
            return
        if self._x_dtype is torch.float8_e4m3fn:
            scale = act.hidden_states_scale
            if scale is None or scale.dim() != 0 or scale.dtype is not torch.float32:
                raise ValueError(
                    f"{type(self).__name__} requires a 0-dim float32 "
                    "hidden_states_scale dequant factor."
                )
            return
        if act.hidden_states_scale is not None:
            raise ValueError(
                f"{type(self).__name__} activations do not use hidden_states_scale."
            )

    def _pack_activation_scale_inputs(
        self, act: MoEActivationPack
    ) -> List[torch.Tensor]:
        if self._use_mxfp8_act_scaling or self._x_dtype is torch.float8_e4m3fn:
            assert act.hidden_states_scale is not None
            scale = act.hidden_states_scale
            if self._use_mxfp8_act_scaling:
                scale = scale.reshape(-1)
            return [scale]
        return []

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        raise NotImplementedError

    def _quant_scales(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return []

    def _input_sf(self, inputs: List[torch.Tensor]) -> torch.Tensor | None:
        if self._use_mxfp8_act_scaling:
            return inputs[-1]
        return None

    def _validate_weight_storage(self, tensors: tuple[torch.Tensor, ...]) -> None:
        if any(t.device != self.device for t in tensors):
            raise ValueError("CUTLASS prepared weights must match the runner device.")
        if any(not t.is_contiguous() for t in tensors):
            raise ValueError("CUTLASS prepared weights must be contiguous.")

    def _validate_input_count(self, inputs: List[torch.Tensor]) -> None:
        if len(inputs) != self._expected_num_inputs:
            raise ValueError(
                f"{type(self).__name__} expects {self._expected_num_inputs} inputs, "
                f"got {len(inputs)}."
            )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_built()
        self._validate_input_count(inputs)
        if self._workspace is None:
            raise RuntimeError("pack_inputs must allocate the CUTLASS workspace first.")
        if tactic == -1:
            profile_ids = [-1, -1]
        elif isinstance(tactic, (tuple, list)) and len(tactic) == 2:
            profile_ids = [int(tactic[0]), int(tactic[1])]
        else:
            raise ValueError(
                f"{type(self).__name__} tactic must be -1 or a (gemm1, gemm2) pair."
            )

        # CUTLASS tactics require a stage-specific preparation launch before
        # they can be selected by run_moe.  The legacy flat API gets this from
        # its two internal autotuner passes; the unified compound tactic must
        # preserve the same contract when the outer autotuner requests setup.
        if do_preparation:
            profile_inputs = [inputs[1], inputs[4], None, inputs[5], None]
            self._inner.forward(
                profile_inputs,
                tactic=profile_ids[0],
                do_preparation=True,
                gemm_idx=1,
            )
            self._inner.forward(
                profile_inputs,
                tactic=profile_ids[1],
                do_preparation=True,
                gemm_idx=2,
            )
            return inputs[0]

        # Select the deterministic geometric-capacity workspace for this
        # launch; older cached buffers remain alive for captured graphs.
        num_tokens, hidden_size = inputs[1].shape
        bucket = map_to_hybrid_bucket(
            num_tokens, self.config.execution.tune_max_num_tokens
        )
        self._ensure_workspace(bucket, hidden_size)

        from .core import cutlass_fused_moe

        cutlass_fused_moe(
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            output_dtype=torch.bfloat16,
            quant_scales=self._quant_scales(inputs),
            input_sf=self._input_sf(inputs),
            output=inputs[0],
            tune_max_num_tokens=self.config.execution.tune_max_num_tokens,
            enable_pdl=self._enable_pdl,
            activation_type=self.config.activation.type,
            use_deepseek_fp8_block_scale=self._use_deepseek_fp8_block_scale,
            use_w4_group_scaling=self._use_w4_group_scaling,
            use_mxfp8_act_scaling=self._use_mxfp8_act_scaling,
            use_packed_weights=self._use_packed_weights,
            use_wfp4afp8_humming=self._use_wfp4afp8_humming,
            use_fused_finalize=self._use_fused_finalize,
            swizzled_input_sf=True,
            profile_ids=profile_ids,
            workspace_buffer=self._workspace,
            **self._activation_params,
        )
        return inputs[0]

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + (
            self._device_arch,
            self._enable_pdl,
        )


class CutlassBf16Runner(_CutlassRunnerBase):
    """Unified adapter for dense BF16 CUTLASS fused MoE."""

    backend_key = "cutlass_bf16"
    supported_quant_variants = (QuantVariant.BF16,)
    supported_activation_classes = _CUTLASS_SEMANTIC_ACTIVATIONS
    _supported_archs = _CUTLASS_BF16_ARCHS
    _weight_dtype = torch.bfloat16
    _use_w4_group_scaling = False
    _required_weight_keys = ("fc1_expert_weights", "fc2_expert_weights")
    _expected_num_inputs = 6

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        w1, w2 = (view[key] for key in self._required_weight_keys)
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        gemm1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, gemm1_rows, hidden_size)
        expected_w2 = (num_experts, hidden_size, intermediate_size)
        if w1.dtype is not torch.bfloat16 or w2.dtype is not torch.bfloat16:
            raise TypeError("Cutlass BF16 prepared weights must use torch.bfloat16.")
        if tuple(w1.shape) != expected_w1 or tuple(w2.shape) != expected_w2:
            raise ValueError(
                f"Cutlass BF16 weight shapes {tuple(w1.shape)}/{tuple(w2.shape)} "
                f"!= expected {expected_w1}/{expected_w2}."
            )
        self._validate_weight_storage((w1, w2))
        return [w1, w2]


class CutlassW4A16Runner(_CutlassRunnerBase):
    """Unified adapter for MXFP4-weight x BF16-activation fused MoE."""

    backend_key = "cutlass_w4a16"
    supported_quant_variants = (QuantVariant.W4A16,)
    supported_activation_classes = _CUTLASS_SEMANTIC_ACTIVATIONS
    _supported_archs = _CUTLASS_W4A16_ARCHS
    _weight_dtype = torch.uint8
    _use_w4_group_scaling = True
    _required_weight_keys = (
        "fc1_expert_weights",
        "fc2_expert_weights",
        "fc1_expert_scales",
        "fc2_expert_scales",
    )
    _expected_num_inputs = 8

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        w1, w2, w1_scale, w2_scale = (view[key] for key in self._required_weight_keys)
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        gemm1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, gemm1_rows, hidden_size // 2)
        expected_w2 = (num_experts, hidden_size, intermediate_size // 2)
        expected_s1 = (
            num_experts,
            gemm1_rows // 64,
            hidden_size // 128,
            16,
            16,
        )
        expected_s2 = (
            num_experts,
            hidden_size // 64,
            intermediate_size // 128,
            16,
            16,
        )
        if any(t.dtype is not torch.uint8 for t in (w1, w2, w1_scale, w2_scale)):
            raise TypeError("Cutlass W4A16 prepared weights and scales must be uint8.")
        if (tuple(w1.shape), tuple(w2.shape)) != (expected_w1, expected_w2):
            raise ValueError(
                f"Cutlass W4A16 weight shapes {tuple(w1.shape)}/{tuple(w2.shape)} "
                f"!= expected {expected_w1}/{expected_w2}."
            )
        if (tuple(w1_scale.shape), tuple(w2_scale.shape)) != (
            expected_s1,
            expected_s2,
        ):
            raise ValueError(
                "Cutlass W4A16 scale shapes "
                f"{tuple(w1_scale.shape)}/{tuple(w2_scale.shape)} != expected "
                f"{expected_s1}/{expected_s2}."
            )
        self._validate_weight_storage((w1, w2, w1_scale, w2_scale))
        return [w1, w2, w1_scale, w2_scale]

    def _quant_scales(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [inputs[6].view(torch.int32), inputs[7].view(torch.int32)]


class CutlassNvfp4Runner(_CutlassRunnerBase):
    """Unified adapter for CUTLASS NVFP4 fused MoE.

    Weights stay packed uint8 in the ``MoEWeightPack`` view. At launch they are
    viewed as ``int64``, matching the flat ``cutlass_fused_moe`` NVFP4 ABI; the
    inner ``MoERunner`` selects the NVFP4 kernel from that dtype. Activations
    remain BF16 and are quantized inside the kernel with unit global scale.
    """

    backend_key = "cutlass_nvfp4"
    supported_quant_variants = (QuantVariant.NVFP4,)
    supported_activation_classes = _CUTLASS_SEMANTIC_ACTIVATIONS
    _supported_archs = _CUTLASS_NVFP4_ARCHS
    _weight_dtype = torch.int64
    _use_w4_group_scaling = False
    _required_weight_keys = (
        "fc1_expert_weights",
        "fc2_expert_weights",
        "fc1_act_global_scale",
        "fc1_weight_block_scale",
        "fc1_dequant_scale",
        "fc2_act_global_scale",
        "fc2_weight_block_scale",
        "fc2_dequant_scale",
    )
    _expected_num_inputs = 12

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        (
            w1,
            w2,
            a1_gs,
            w1_scale,
            fc1_dequant,
            a2_gs,
            w2_scale,
            fc2_dequant,
        ) = (view[key] for key in self._required_weight_keys)
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        if hidden_size % 16 != 0 or intermediate_size % 16 != 0:
            raise ValueError(
                "Cutlass NVFP4 requires hidden_size and intermediate_size "
                f"divisible by 16, got H={hidden_size}, I={intermediate_size}."
            )
        gemm1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, gemm1_rows, hidden_size // 2)
        expected_w2 = (num_experts, hidden_size, intermediate_size // 2)
        expected_s1 = (
            num_experts,
            round_up(gemm1_rows, 128),
            round_up(hidden_size // 16, 4),
        )
        expected_s2 = (
            num_experts,
            round_up(hidden_size, 128),
            round_up(intermediate_size // 16, 4),
        )
        if w1.dtype is not torch.uint8 or w2.dtype is not torch.uint8:
            raise TypeError("Cutlass NVFP4 packed weights must be uint8.")
        if w1_scale.dtype is not torch.uint8 or w2_scale.dtype is not torch.uint8:
            raise TypeError("Cutlass NVFP4 block scales must be uint8.")
        if any(
            t.dtype is not torch.float32
            for t in (a1_gs, a2_gs, fc1_dequant, fc2_dequant)
        ):
            raise TypeError("Cutlass NVFP4 global and dequant scales must be float32.")
        if (tuple(w1.shape), tuple(w2.shape)) != (expected_w1, expected_w2):
            raise ValueError(
                f"Cutlass NVFP4 weight shapes {tuple(w1.shape)}/{tuple(w2.shape)} "
                f"!= expected {expected_w1}/{expected_w2}."
            )
        if (tuple(w1_scale.shape), tuple(w2_scale.shape)) != (
            expected_s1,
            expected_s2,
        ):
            raise ValueError(
                "Cutlass NVFP4 scale shapes "
                f"{tuple(w1_scale.shape)}/{tuple(w2_scale.shape)} != expected "
                f"{expected_s1}/{expected_s2}."
            )
        expected_dequant = (num_experts,)
        if (
            tuple(fc1_dequant.shape) != expected_dequant
            or tuple(fc2_dequant.shape) != expected_dequant
        ):
            raise ValueError(
                "Cutlass NVFP4 dequant scale shapes "
                f"{tuple(fc1_dequant.shape)}/{tuple(fc2_dequant.shape)} != "
                f"expected {expected_dequant}."
            )
        self._validate_weight_storage(
            (w1, w2, a1_gs, w1_scale, fc1_dequant, a2_gs, w2_scale, fc2_dequant)
        )
        # Flat NVFP4 CUTLASS selects the kernel from weight dtype int64.
        return [
            w1.view(torch.int64),
            w2.view(torch.int64),
            a1_gs,
            w1_scale,
            fc1_dequant,
            a2_gs,
            w2_scale,
            fc2_dequant,
        ]

    def _quant_scales(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            inputs[6],
            inputs[7].view(torch.int32),
            inputs[8],
            inputs[9],
            inputs[10].view(torch.int32),
            inputs[11],
        ]


class CutlassFp8PerTensorRunner(_CutlassRunnerBase):
    """Unified adapter for CUTLASS per-tensor FP8 fused MoE."""

    backend_key = "cutlass_fp8_per_tensor"
    supported_quant_variants = (QuantVariant.FP8PerTensor,)
    supported_activation_classes = _CUTLASS_SEMANTIC_ACTIVATIONS
    _supported_archs = _CUTLASS_FP8_ARCHS
    _x_dtype = torch.float8_e4m3fn
    _weight_dtype = torch.float8_e4m3fn
    _required_weight_keys = (
        "fc1_expert_weights",
        "fc2_expert_weights",
        "fc1_dequant",
        "fc2_dequant",
    )
    _expected_num_inputs = 9

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        w1, w2, w1_dequant, w2_dequant = (
            view[key] for key in self._required_weight_keys
        )
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        gemm1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, gemm1_rows, hidden_size)
        expected_w2 = (num_experts, hidden_size, intermediate_size)
        if w1.dtype is not torch.float8_e4m3fn or w2.dtype is not torch.float8_e4m3fn:
            raise TypeError("Cutlass FP8 prepared weights must be float8_e4m3fn.")
        if tuple(w1.shape) != expected_w1 or tuple(w2.shape) != expected_w2:
            raise ValueError(
                f"Cutlass FP8 weight shapes {tuple(w1.shape)}/{tuple(w2.shape)} "
                f"!= expected {expected_w1}/{expected_w2}."
            )
        if (
            w1_dequant.dtype is not torch.float32
            or w2_dequant.dtype is not torch.float32
        ):
            raise TypeError("Cutlass FP8 dequant scales must be float32.")
        expected_scale = (num_experts,)
        if (
            tuple(w1_dequant.shape) != expected_scale
            or tuple(w2_dequant.shape) != expected_scale
        ):
            raise ValueError(
                "Cutlass FP8 dequant scale shapes "
                f"{tuple(w1_dequant.shape)}/{tuple(w2_dequant.shape)} != "
                f"expected {expected_scale}."
            )
        self._validate_weight_storage((w1, w2, w1_dequant, w2_dequant))
        return [w1, w2, w1_dequant, w2_dequant]

    def _quant_scales(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        act_scale = inputs[8]
        gemm2_act_quant = torch.ones((), device=act_scale.device, dtype=torch.float32)
        return [
            (inputs[6] * act_scale).float(),
            gemm2_act_quant,
            inputs[7].float(),
            act_scale,
        ]


class CutlassFp8BlockRunner(_CutlassRunnerBase):
    """Unified adapter for CUTLASS DeepSeek 128x128 FP8 block-scale MoE."""

    backend_key = "cutlass_fp8_block"
    supported_quant_variants = (QuantVariant.DeepSeekFp8,)
    supported_activation_classes = _CUTLASS_SEMANTIC_ACTIVATIONS
    _supported_archs = _CUTLASS_FP8_BLOCK_ARCHS
    _weight_dtype = torch.float8_e4m3fn
    _use_deepseek_fp8_block_scale = True
    _required_weight_keys = (
        "fc1_expert_weights",
        "fc2_expert_weights",
        "fc1_block_scale",
        "fc2_block_scale",
    )
    _expected_num_inputs = 8

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        from math import ceil

        w1, w2, w1_scale, w2_scale = (view[key] for key in self._required_weight_keys)
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        gemm1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, gemm1_rows, hidden_size)
        expected_w2 = (num_experts, hidden_size, intermediate_size)
        expected_s1 = (
            num_experts,
            ceil(gemm1_rows / 128),
            ceil(hidden_size / 128),
        )
        expected_s2 = (
            num_experts,
            ceil(hidden_size / 128),
            ceil(intermediate_size / 128),
        )
        if w1.dtype is not torch.float8_e4m3fn or w2.dtype is not torch.float8_e4m3fn:
            raise TypeError("Cutlass FP8-block prepared weights must be float8_e4m3fn.")
        if w1_scale.dtype is not torch.float32 or w2_scale.dtype is not torch.float32:
            raise TypeError("Cutlass FP8-block scales must be float32.")
        if (tuple(w1.shape), tuple(w2.shape)) != (expected_w1, expected_w2):
            raise ValueError(
                f"Cutlass FP8-block weight shapes {tuple(w1.shape)}/{tuple(w2.shape)} "
                f"!= expected {expected_w1}/{expected_w2}."
            )
        if (tuple(w1_scale.shape), tuple(w2_scale.shape)) != (expected_s1, expected_s2):
            raise ValueError(
                "Cutlass FP8-block scale shapes "
                f"{tuple(w1_scale.shape)}/{tuple(w2_scale.shape)} != expected "
                f"{expected_s1}/{expected_s2}."
            )
        self._validate_weight_storage((w1, w2, w1_scale, w2_scale))
        return [w1, w2, w1_scale, w2_scale]

    def _quant_scales(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [inputs[6], inputs[7]]


class CutlassMxfp8Mxfp4Runner(_CutlassRunnerBase):
    """Unified adapter for CUTLASS MXFP8 x MXFP4 fused MoE."""

    backend_key = "cutlass_mxfp8_mxfp4"
    supported_quant_variants = (QuantVariant.MXFP4,)
    supported_activation_classes = _CUTLASS_SEMANTIC_ACTIVATIONS
    _supported_archs = _CUTLASS_MXFP8_MXFP4_ARCHS
    _x_dtype = torch.float8_e4m3fn
    _weight_dtype = torch.int64
    _use_mxfp8_act_scaling = True
    _required_weight_keys = (
        "fc1_expert_weights",
        "fc2_expert_weights",
        "fc1_expert_scales",
        "fc2_expert_scales",
        "fc1_input_scale",
        "fc2_input_scale",
    )
    _expected_num_inputs = 11

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        w1, w2, w1_scale, w2_scale, a1_scale, a2_scale = (
            view[key] for key in self._required_weight_keys
        )
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
            raise ValueError(
                "Cutlass MXFP8xMXFP4 requires hidden_size and intermediate_size "
                f"divisible by 128, got H={hidden_size}, I={intermediate_size}."
            )
        gemm1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, gemm1_rows, hidden_size // 2)
        expected_w2 = (num_experts, hidden_size, intermediate_size // 2)
        if w1.dtype is not torch.uint8 or w2.dtype is not torch.uint8:
            raise TypeError("Cutlass MXFP8xMXFP4 packed weights must be uint8.")
        if (tuple(w1.shape), tuple(w2.shape)) != (expected_w1, expected_w2):
            raise ValueError(
                "Cutlass MXFP8xMXFP4 weight shapes "
                f"{tuple(w1.shape)}/{tuple(w2.shape)} != expected "
                f"{expected_w1}/{expected_w2}."
            )
        self._validate_weight_storage((w1, w2, w1_scale, w2_scale, a1_scale, a2_scale))
        expected_s1 = num_experts * _mxfp8_swizzled_act_sf_numel(
            gemm1_rows, hidden_size
        )
        expected_s2 = num_experts * _mxfp8_swizzled_act_sf_numel(
            hidden_size, intermediate_size
        )
        if w1_scale.dtype is not torch.uint8 or w2_scale.dtype is not torch.uint8:
            raise TypeError("Cutlass MXFP8xMXFP4 weight scales must be uint8.")
        if w1_scale.numel() != expected_s1 or w2_scale.numel() != expected_s2:
            raise ValueError(
                "Cutlass MXFP8xMXFP4 weight scale sizes "
                f"{w1_scale.numel()}/{w2_scale.numel()} != expected "
                f"{expected_s1}/{expected_s2}."
            )
        _require_cutlass_tensor(
            a1_scale,
            name="fc1_input_scale",
            dtype=torch.float32,
            shape=(num_experts,),
        )
        _require_cutlass_tensor(
            a2_scale,
            name="fc2_input_scale",
            dtype=torch.float32,
            shape=(num_experts,),
        )
        return [
            w1.view(torch.int64),
            w2.view(torch.int64),
            w1_scale,
            w2_scale,
            a1_scale,
            a2_scale,
        ]

    def _quant_scales(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            inputs[6].view(torch.int32),
            inputs[8],
            inputs[7].view(torch.int32),
            inputs[9],
        ]


class CutlassMxfp8Runner(_CutlassRunnerBase):
    """Unified adapter for CUTLASS MXFP8 x MXFP8 fused MoE."""

    backend_key = "cutlass_mxfp8"
    supported_quant_variants = (QuantVariant.MxFp8,)
    supported_activation_classes = _CUTLASS_SEMANTIC_ACTIVATIONS
    _supported_archs = _CUTLASS_MXFP8_ARCHS
    _x_dtype = torch.float8_e4m3fn
    _weight_dtype = torch.float8_e4m3fn
    _use_mxfp8_act_scaling = True
    _required_weight_keys = (
        "fc1_expert_weights",
        "fc2_expert_weights",
        "fc1_expert_scales",
        "fc2_expert_scales",
        "fc1_input_scale",
        "fc2_input_scale",
    )
    _expected_num_inputs = 11

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        w1, w2, w1_scale, w2_scale, a1_scale, a2_scale = (
            view[key] for key in self._required_weight_keys
        )
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        if hidden_size % 128 != 0 or intermediate_size % 128 != 0:
            raise ValueError(
                "Cutlass MXFP8 requires hidden_size and intermediate_size "
                f"divisible by 128, got H={hidden_size}, I={intermediate_size}."
            )
        gemm1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, gemm1_rows, hidden_size)
        expected_w2 = (num_experts, hidden_size, intermediate_size)
        if w1.dtype is not torch.float8_e4m3fn or w2.dtype is not torch.float8_e4m3fn:
            raise TypeError("Cutlass MXFP8 prepared weights must be float8_e4m3fn.")
        if (tuple(w1.shape), tuple(w2.shape)) != (expected_w1, expected_w2):
            raise ValueError(
                f"Cutlass MXFP8 weight shapes {tuple(w1.shape)}/{tuple(w2.shape)} "
                f"!= expected {expected_w1}/{expected_w2}."
            )
        # Binding aligns the semantic GEMM1 row count to the scale-factor tile.
        expected_s1 = (
            num_experts,
            round_up(gemm1_rows, 128),
            round_up(hidden_size // 32, 4) // 4,
        )
        expected_s2 = (
            num_experts,
            round_up(hidden_size, 128),
            round_up(intermediate_size // 32, 4) // 4,
        )
        _require_cutlass_tensor(
            w1_scale, name="fc1_expert_scales", dtype=torch.int32, shape=expected_s1
        )
        _require_cutlass_tensor(
            w2_scale, name="fc2_expert_scales", dtype=torch.int32, shape=expected_s2
        )
        _require_cutlass_tensor(
            a1_scale,
            name="fc1_input_scale",
            dtype=torch.float32,
            shape=(num_experts,),
        )
        _require_cutlass_tensor(
            a2_scale,
            name="fc2_input_scale",
            dtype=torch.float32,
            shape=(num_experts,),
        )
        self._validate_weight_storage((w1, w2, w1_scale, w2_scale, a1_scale, a2_scale))
        return [w1, w2, w1_scale, w2_scale, a1_scale, a2_scale]

    def _quant_scales(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [inputs[6], inputs[8], inputs[7], inputs[9]]


class CutlassW4A8Runner(_CutlassRunnerBase):
    """Unified adapter for CUTLASS INT4-weight x FP8-activation fused MoE."""

    backend_key = "cutlass_w4a8"
    supported_quant_variants = (QuantVariant.W4A8,)
    supported_activation_classes = _CUTLASS_SEMANTIC_ACTIVATIONS
    _supported_archs = _CUTLASS_W4A8_ARCHS
    _weight_dtype = torch.uint8
    _use_w4_group_scaling = True
    _use_packed_weights = True
    _required_weight_keys = (
        "fc1_expert_weights",
        "fc2_expert_weights",
        "fc1_expert_scales",
        "fc2_expert_scales",
        "fc1_act_scale",
        "fc2_act_scale",
        "fc1_zero",
        "fc2_zero",
        "fc1_alpha",
        "fc2_alpha",
    )
    _expected_num_inputs = 14

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        tensors = tuple(view[key] for key in self._required_weight_keys)
        w1, w2 = tensors[0], tensors[1]
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        gemm1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, gemm1_rows, hidden_size // 2)
        expected_w2 = (num_experts, hidden_size, intermediate_size // 2)
        if w1.dtype is not torch.uint8 or w2.dtype is not torch.uint8:
            raise TypeError("Cutlass W4A8 packed weights must be uint8.")
        if (tuple(w1.shape), tuple(w2.shape)) != (expected_w1, expected_w2):
            raise ValueError(
                f"Cutlass W4A8 weight shapes {tuple(w1.shape)}/{tuple(w2.shape)} "
                f"!= expected {expected_w1}/{expected_w2}."
            )
        expected_s1 = (
            num_experts,
            gemm1_rows // 64,
            hidden_size // 128,
            8,
            8,
        )
        expected_s2 = (
            num_experts,
            hidden_size // 64,
            intermediate_size // 128,
            8,
            8,
        )
        w1_scale, w2_scale = tensors[2], tensors[3]
        act1, act2, zero1, zero2, alpha1, alpha2 = tensors[4:]
        _require_cutlass_tensor(
            w1_scale,
            name="fc1_expert_scales",
            dtype=torch.bfloat16,
            shape=expected_s1,
        )
        _require_cutlass_tensor(
            w2_scale,
            name="fc2_expert_scales",
            dtype=torch.bfloat16,
            shape=expected_s2,
        )
        _require_cutlass_tensor(
            act1, name="fc1_act_scale", dtype=torch.bfloat16, shape=(hidden_size,)
        )
        _require_cutlass_tensor(
            act2,
            name="fc2_act_scale",
            dtype=torch.bfloat16,
            shape=(intermediate_size,),
        )
        _require_cutlass_tensor(
            zero1, name="fc1_zero", dtype=torch.bfloat16, shape=(0,)
        )
        _require_cutlass_tensor(
            zero2, name="fc2_zero", dtype=torch.bfloat16, shape=(0,)
        )
        _require_cutlass_tensor(
            alpha1, name="fc1_alpha", dtype=torch.float32, shape=(num_experts,)
        )
        _require_cutlass_tensor(
            alpha2, name="fc2_alpha", dtype=torch.float32, shape=(num_experts,)
        )
        self._validate_weight_storage(tensors)
        return list(tensors)

    def _quant_scales(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return list(inputs[6:14])


class CutlassHummingRunner(_CutlassRunnerBase):
    """Unified adapter for CUTLASS Humming MXFP4 x FP8 fused MoE."""

    backend_key = "cutlass_humming"
    supported_quant_variants = (QuantVariant.Humming,)
    supported_activation_classes = _CUTLASS_SEMANTIC_ACTIVATIONS
    _supported_archs = _CUTLASS_HUMMING_ARCHS
    _weight_dtype = torch.uint8
    _use_w4_group_scaling = True
    _use_wfp4afp8_humming = True
    _required_weight_keys = (
        "fc1_expert_weights",
        "fc2_expert_weights",
        "fc1_expert_scales",
        "fc2_expert_scales",
        "fc1_residual_scale",
        "fc2_residual_scale",
        "fc2_act_global",
    )
    _expected_num_inputs = 11

    def _pack_weight_inputs(
        self, view: dict[str, torch.Tensor], hidden_size: int
    ) -> List[torch.Tensor]:
        w1, w2, w1_scale, w2_scale, r1, r2, a2 = (
            view[key] for key in self._required_weight_keys
        )
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        gemm1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, gemm1_rows, hidden_size // 2)
        expected_w2 = (num_experts, hidden_size, intermediate_size // 2)
        if w1.dtype is not torch.uint8 or w2.dtype is not torch.uint8:
            raise TypeError("Cutlass Humming packed weights must be uint8.")
        if (tuple(w1.shape), tuple(w2.shape)) != (expected_w1, expected_w2):
            raise ValueError(
                "Cutlass Humming weight shapes "
                f"{tuple(w1.shape)}/{tuple(w2.shape)} != expected "
                f"{expected_w1}/{expected_w2}."
            )
        expected_s1 = (
            num_experts,
            gemm1_rows // 64,
            hidden_size // 128,
            16,
            16,
        )
        expected_s2 = (
            num_experts,
            hidden_size // 64,
            intermediate_size // 128,
            16,
            16,
        )
        _require_cutlass_tensor(
            w1_scale, name="fc1_expert_scales", dtype=torch.uint8, shape=expected_s1
        )
        _require_cutlass_tensor(
            w2_scale, name="fc2_expert_scales", dtype=torch.uint8, shape=expected_s2
        )
        _require_cutlass_tensor(
            r1, name="fc1_residual_scale", dtype=torch.float32, shape=(num_experts,)
        )
        _require_cutlass_tensor(
            r2, name="fc2_residual_scale", dtype=torch.float32, shape=(num_experts,)
        )
        _require_cutlass_tensor(
            a2, name="fc2_act_global", dtype=torch.float32, shape=()
        )
        self._validate_weight_storage((w1, w2, w1_scale, w2_scale, r1, r2, a2))
        return [w1, w2, w1_scale, w2_scale, r1, r2, a2]

    def _quant_scales(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [
            inputs[6].view(torch.int32),
            inputs[8],
            inputs[10],
            inputs[7].view(torch.int32),
            inputs[9],
        ]


# ---------------------------------------------------------------------------
# cuTile BF16 runner
# ---------------------------------------------------------------------------


_CUTILE_BF16_DEFAULT_GEMM_CONFIGS = {
    89: (128, 32, 4),
    90: (128, 64, 1),
    120: (128, 32, 4),
    121: (128, 32, 4),
}
_CUTILE_BF16_GEMM_CONFIGS = (
    (128, 32, 4),
    (128, 64, 1),
    (128, 128, 1),
    (128, 64, 2),
    (64, 128, 2),
    (64, 32, 4),
    (64, 64, 4),
    (128, 128, 2),
    (256, 32, 2),
    (256, 32, 1),
    (256, 64, 2),
    (256, 64, 1),
    (256, 128, 1),
)


@dataclass(frozen=True)
class _CuTileGemmProblem:
    stage: int
    arch: int
    block_size: int
    n: int
    k: int
    fused_epilogue: bool = False
    is_gated: bool = False
    input_sorted: bool = False


def _cutile_bf16_config_rejection_reason(
    problem: _CuTileGemmProblem, config: tuple[int, int, int]
) -> str | None:
    if config not in _CUTILE_BF16_GEMM_CONFIGS:
        return f"config={config} is outside the supported cuTile BF16 tile set"
    tile_n, tile_k, _ = config
    if tile_n > 2 * problem.n:
        return f"tile_n={tile_n} exceeds 2*N={2 * problem.n}"
    if tile_k > 2 * problem.k:
        return f"tile_k={tile_k} exceeds 2*K={2 * problem.k}"
    return None


def _valid_cutile_gemm_configs(
    problem: _CuTileGemmProblem,
    configs: tuple[tuple[int, int, int], ...],
    rejection_reason: Any,
    runner_name: str,
) -> list[tuple[int, int, int]]:
    valid = [config for config in configs if rejection_reason(problem, config) is None]
    if not valid:
        raise NotImplementedError(
            f"{runner_name}: no supported grouped GEMM configuration for "
            f"stage={problem.stage}, SM{problem.arch}, block_size={problem.block_size}, "
            f"N={problem.n}, K={problem.k}."
        )
    return valid


def _cutile_bf16_gemm_configs(
    problem: _CuTileGemmProblem,
) -> list[tuple[int, int, int]]:
    default = _CUTILE_BF16_DEFAULT_GEMM_CONFIGS[problem.arch]
    ordered = (default,) + tuple(
        config for config in _CUTILE_BF16_GEMM_CONFIGS if config != default
    )
    return _valid_cutile_gemm_configs(
        problem,
        ordered,
        _cutile_bf16_config_rejection_reason,
        "CuTileBf16Runner",
    )


def _resolve_cutile_stage_tactics(
    tactics: list[Any],
    fallback: tuple[int, ...],
    *,
    arity: int,
    name: str,
) -> list[tuple[int, ...]]:
    resolved: list[tuple[int, ...]] = []
    for tactic in tactics:
        values = fallback if tactic == -1 else tuple(map(int, tactic))
        if len(values) != arity:
            raise ValueError(f"{name} tactics must contain {arity} integers.")
        resolved.append(values)
    return list(dict.fromkeys(resolved))


def _factorized_cutile_tactics(
    parent: Any,
    inputs: list[torch.Tensor],
    block_size: int,
    gemm1_runner: TunableRunner,
    gemm2_runner: TunableRunner,
    gemm1_fallback: tuple[int, ...],
    gemm2_fallback: tuple[int, ...],
    *,
    name_suffix: str = "",
) -> list[tuple[int, ...]]:
    tuner = AutoTuner.get()
    tuning_config = TuningConfig(
        use_cuda_graph=True,
        inputs_pre_hook=parent._prepare_tuning_inputs,
    )
    prefix = f"moe_{parent.backend_key}_sm{parent._device_arch}_b{block_size}"
    top_k = parent._num_top_tactics_per_stage
    gemm1_tactics = tuner.rank_tactics(
        f"{prefix}_gemm1{name_suffix}",
        [gemm1_runner],
        tuning_config,
        inputs,
        k=top_k,
    )
    gemm1 = _resolve_cutile_stage_tactics(
        gemm1_tactics,
        gemm1_fallback,
        arity=len(gemm1_fallback),
        name=f"{parent.backend_key} GEMM1",
    )
    if isinstance(gemm2_runner, _CuTileStageRunner):
        gemm2_runner.fallback = (block_size, *gemm1[0], *gemm2_fallback)
    gemm2_tactics = tuner.rank_tactics(
        f"{prefix}_gemm2{name_suffix}",
        [gemm2_runner],
        tuning_config,
        inputs,
        k=top_k,
    )
    gemm2 = _resolve_cutile_stage_tactics(
        gemm2_tactics,
        gemm2_fallback,
        arity=len(gemm2_fallback),
        name=f"{parent.backend_key} GEMM2",
    )
    pairs = list(
        dict.fromkeys(
            (block_size, *gemm1_config, *gemm2_config)
            for gemm1_config in gemm1
            for gemm2_config in gemm2
        )
    )
    fallback = (block_size, *gemm1_fallback, *gemm2_fallback)
    if fallback not in pairs:
        pairs[-1] = fallback
    return pairs


class _CuTileStageRunner(TunableRunner):
    def __init__(
        self,
        parent: Any,
        stage: int,
        block_size: int,
        fallback: tuple[int, ...],
    ):
        self.parent = parent
        self.stage = stage
        self.block_size = block_size
        self.fallback = fallback

    def get_valid_tactics(self, inputs: list[torch.Tensor], _profile: Any) -> list[Any]:
        return self.parent._stage_tactics(
            inputs,
            stage=self.stage,
            block_size=self.block_size,
        )

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        stage_slice = self.parent._stage_tactic_slice(self.stage)
        values = self.fallback[stage_slice] if tactic == -1 else tuple(map(int, tactic))
        if len(values) != stage_slice.stop - stage_slice.start:
            raise ValueError(
                f"{self.parent.backend_key} GEMM{self.stage} tactic has "
                f"{len(values)} values."
            )
        compound = list(self.fallback)
        compound[stage_slice] = values
        return self.parent.forward(
            inputs,
            tactic=tuple(compound),
            do_preparation=do_preparation,
            **kwargs,
        )

    def get_cache_key_extras(self, inputs: list[torch.Tensor]) -> tuple[Any, ...]:
        return self.parent._stage_cache_key(
            inputs,
            stage=self.stage,
            block_size=self.block_size,
        )


class CuTileBf16Runner(MoERunner):
    """Unified adapter for the cuTile BF16 MoE pipeline."""

    backend_key = "cutile_bf16"
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = (QuantVariant.BF16,)
    supported_activation_classes = (SwiGLU, ReLU2)
    supports_expert_parallelism = False
    _block_sizes: ClassVar[tuple[int, ...]] = (32, 64, 128)
    _num_top_tactics_per_stage: ClassVar[int] = 2
    _supported_archs: ClassVar[tuple[int, ...]] = _CUTILE_BF16_ARCHS
    _precision_name: ClassVar[str] = "BF16"

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        from ..utils import get_compute_capability

        self.config = config
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"{type(self).__name__} requires CUDA, got {device}.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        major, minor = get_compute_capability(self.device)
        self._device_arch = major * 10 + minor
        self._num_sms = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self._kernel_module: Any = None
        self._workspace_cache: dict[tuple[int, int], Any] = {}
        self._workspace: Any = None
        self.tuning_config = TuningConfig()

    def _check_support(self) -> None:
        super()._check_support()
        if not self.config.finalize.do_finalize:
            raise NotImplementedError(
                f"{type(self).__name__} requires do_finalize=True."
            )
        if self.config.execution.enable_pdl is True:
            raise NotImplementedError(
                f"{type(self).__name__} does not support PDL launches."
            )
        if self._device_arch not in self._supported_archs:
            raise RuntimeError(
                f"{type(self).__name__} does not support SM{self._device_arch}; "
                f"supported architectures are {self._supported_archs}."
            )
        from ..cutile import is_cuda_tile_available

        with torch.cuda.device(self.device):
            available = is_cuda_tile_available()
        if not available:
            raise RuntimeError(
                f"cuTile {self._precision_name} requires cuda-tile and a "
                "tileiras/NVRTC toolchain "
                f"that supports SM{self._device_arch}."
            )

    def _check_activation_parameters(self) -> None:
        if (
            isinstance(self.config.activation, SwiGLU)
            and self.config.activation != SwiGLU()
        ):
            raise NotImplementedError(
                f"{type(self).__name__} cannot represent non-default SwiGLU scalars."
            )

    def _build(self) -> None:
        from .cutile import moe

        self._kernel_module = moe

    def _prepare_tuning_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Give synthesized profiles valid, deterministic balanced routing."""
        num_tokens, hidden_size = inputs[1].shape
        self._ensure_workspace(num_tokens, hidden_size)
        top_k = self.config.routing.top_k
        num_experts = self.config.routing.num_experts
        assignments = torch.arange(
            num_tokens * top_k, dtype=torch.int32, device=inputs[2].device
        ).reshape(num_tokens, top_k)
        inputs[2].copy_(assignments % num_experts)
        inputs[3].fill_(1.0 / top_k)
        return inputs

    def _ensure_workspace(self, num_tokens: int, hidden_size: int) -> None:
        self._require_built()
        ceiling = self.config.execution.tune_max_num_tokens
        if num_tokens > ceiling:
            raise ValueError(
                f"num_tokens={num_tokens} exceeds tune_max_num_tokens={ceiling}."
            )
        capacity = map_to_hybrid_bucket(num_tokens, ceiling)
        key = (capacity, hidden_size)
        workspace = self._workspace_cache.get(key)
        if workspace is None:
            workspace = self._kernel_module.allocate_workspace(
                num_tokens=capacity,
                hidden_size=hidden_size,
                intermediate_size=self.config.experts.intermediate_size,
                num_experts=self.config.routing.num_experts,
                top_k=self.config.routing.top_k,
                is_gated=self.config.activation.is_gated,
                block_sizes=self._block_sizes,
                device=self.device,
            )
            self._workspace_cache[key] = workspace
        self._workspace = workspace

    def _validate_inputs(self, act: MoEActivationPack) -> tuple[torch.Tensor, int, int]:
        self._require_built()
        if act.routing_input_mode is not RoutingInputMode.PackedPrecomputed:
            raise NotImplementedError(
                f"{type(self).__name__} supports only PackedPrecomputed routing."
            )
        hidden_states = act.hidden_states_q
        if hidden_states.ndim != 2 or hidden_states.dtype is not torch.bfloat16:
            raise TypeError(
                f"{type(self).__name__} requires 2D BF16 hidden_states_q, got "
                f"shape={tuple(hidden_states.shape)}, dtype={hidden_states.dtype}."
            )
        if hidden_states.device != self.device or not hidden_states.is_contiguous():
            raise ValueError(
                f"{type(self).__name__} requires contiguous hidden states on "
                f"{self.device}."
            )
        if act.hidden_states_scale is not None:
            raise ValueError(
                f"{type(self).__name__} requires hidden_states_scale=None."
            )
        num_tokens, hidden_size = hidden_states.shape
        if num_tokens == 0:
            raise NotImplementedError(
                f"{type(self).__name__} does not support zero tokens."
            )
        _validate_prerouted_inputs(
            act,
            num_tokens,
            self.config.routing.top_k,
            type(self).__name__,
            allowed_weights_dtypes=(torch.float32,),
            require_contiguous=True,
        )
        return hidden_states, num_tokens, hidden_size

    def _configure_tuning(self, num_tokens: int, hidden_size: int) -> None:
        bucket = map_to_hybrid_bucket(
            num_tokens, self.config.execution.tune_max_num_tokens
        )
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=(0, 1, 2, 3),
                    dim_idx=(0, 0, 0, 0),
                    gen_tuning_buckets=(bucket,),
                    map_to_tuning_buckets=make_hybrid_bucket_mapper(
                        self.config.execution.tune_max_num_tokens
                    ),
                ),
            ),
            use_cuda_graph=True,
            inputs_pre_hook=self._prepare_tuning_inputs,
        )
        self._ensure_workspace(num_tokens, hidden_size)

    def _gemm_problem(
        self,
        inputs: List[torch.Tensor],
        *,
        stage: int,
        block_size: int,
    ) -> _CuTileGemmProblem:
        hidden_size = inputs[1].shape[1]
        intermediate_size = self.config.experts.intermediate_size
        if stage == 1:
            n = intermediate_size * (2 if self.config.activation.is_gated else 1)
            k = hidden_size
        elif stage == 2:
            n, k = hidden_size, intermediate_size
        else:
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        return _CuTileGemmProblem(
            stage=stage,
            arch=self._device_arch,
            block_size=block_size,
            n=n,
            k=k,
        )

    def _stage_tactics(
        self,
        inputs: List[torch.Tensor],
        *,
        stage: int,
        block_size: int,
    ) -> List[Any]:
        return _cutile_bf16_gemm_configs(
            self._gemm_problem(inputs, stage=stage, block_size=block_size)
        )

    def _stage_tactic_slice(self, stage: int) -> slice:
        return slice(1, 4) if stage == 1 else slice(4, 7)

    def _stage_cache_key(
        self,
        inputs: List[torch.Tensor],
        *,
        stage: int,
        block_size: int,
    ) -> tuple[Any, ...]:
        del inputs
        return (
            "cutile_bf16_stage",
            self._device_arch,
            self._num_sms,
            stage,
            block_size,
            int(self.config.activation.type),
            self.config.routing.num_experts,
            self.config.routing.top_k,
            self.config.experts.intermediate_size,
        )

    def _fallback_tactic(
        self, inputs: List[torch.Tensor]
    ) -> tuple[int, int, int, int, int, int, int]:
        block_size = self._factorized_block_sizes(inputs[2].numel())[0]
        gemm1 = _cutile_bf16_gemm_configs(
            self._gemm_problem(inputs, stage=1, block_size=block_size)
        )[0]
        gemm2 = _cutile_bf16_gemm_configs(
            self._gemm_problem(inputs, stage=2, block_size=block_size)
        )[0]
        return (block_size, *gemm1, *gemm2)

    def _factorized_tactics(self, inputs: List[torch.Tensor]) -> list[tuple[int, ...]]:
        fallback = self._fallback_tactic(inputs)
        block_size = fallback[0]
        gemm1_runner = _CuTileStageRunner(self, 1, block_size, fallback)
        gemm2_runner = _CuTileStageRunner(self, 2, block_size, fallback)
        return _factorized_cutile_tactics(
            self,
            inputs,
            block_size,
            gemm1_runner,
            gemm2_runner,
            fallback[1:4],
            fallback[4:7],
        )

    def _candidate_non_gated_block_sizes(self, num_assignments: int) -> tuple[int, ...]:
        num_experts = self.config.routing.num_experts
        rows_per_expert = (num_assignments + num_experts - 1) // num_experts
        prefill_threshold = {89: 256, 90: 32, 120: 128, 121: 128}[self._device_arch]
        return (32,) if rows_per_expert < prefill_threshold else (64,)

    def _gated_block_size(self, num_assignments: int) -> int:
        num_experts = self.config.routing.num_experts
        rows_per_expert = (num_assignments + num_experts - 1) // num_experts
        if rows_per_expert < 64:
            return 32
        return 64 if rows_per_expert < 512 else 128

    def _factorized_block_sizes(self, num_assignments: int) -> tuple[int, ...]:
        if self.config.activation.is_gated:
            return (self._gated_block_size(num_assignments),)
        return self._candidate_non_gated_block_sizes(num_assignments)

    def get_valid_tactics(self, inputs: List[torch.Tensor], _profile: Any) -> List[Any]:
        self._require_built()
        if len(inputs) != 6:
            raise ValueError(
                f"{type(self).__name__} expects 6 inputs, got {len(inputs)}."
            )
        return self._factorized_tactics(inputs)

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        hidden_states, num_tokens, hidden_size = self._validate_inputs(act)
        view = weights.get_view(self.backend_key)
        missing = [key for key in ("w1", "w2") if key not in view]
        if missing:
            raise KeyError(
                f"{self.backend_key} prepared weights are missing {missing}."
            )
        w1, w2 = view["w1"], view["w2"]
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        expected_w1 = (
            num_experts,
            hidden_size,
            intermediate_size * (2 if self.config.activation.is_gated else 1),
        )
        expected_w2 = (num_experts, intermediate_size, hidden_size)
        if w1.dtype is not torch.bfloat16 or w2.dtype is not torch.bfloat16:
            raise TypeError("cuTile BF16 prepared weights must use torch.bfloat16.")
        if tuple(w1.shape) != expected_w1 or tuple(w2.shape) != expected_w2:
            raise ValueError(
                f"cuTile BF16 weight shapes {tuple(w1.shape)}/{tuple(w2.shape)} "
                f"!= expected {expected_w1}/{expected_w2}."
            )
        if any(t.device != self.device for t in (w1, w2)):
            raise ValueError(
                "cuTile BF16 prepared weights must match the runner device."
            )
        if not w1.is_contiguous() or not w2.is_contiguous():
            raise ValueError("cuTile BF16 prepared weights must be contiguous.")

        capacity = map_to_hybrid_bucket(
            num_tokens, self.config.execution.tune_max_num_tokens
        )
        num_assignments = capacity * self.config.routing.top_k
        max_padded_assignments = num_assignments + num_experts * (
            max(self._block_sizes) - 1
        )
        _validate_cutile_int32_routing(
            type(self).__name__, num_assignments, max_padded_assignments
        )
        self._configure_tuning(num_tokens, hidden_size)
        return [
            hidden_states.new_empty((num_tokens, hidden_size)),
            hidden_states,
            act.topk_ids,
            act.topk_weights,
            w1,
            w2,
        ]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_built()
        if len(inputs) != 6:
            raise ValueError(
                f"{type(self).__name__} expects 6 inputs, got {len(inputs)}."
            )
        hidden_size = inputs[1].shape[1]
        if tactic == -1:
            tactic = self._fallback_tactic(inputs)
        if not isinstance(tactic, (tuple, list)) or len(tactic) != 7:
            raise ValueError(
                f"{type(self).__name__} tactic must be -1 or "
                "(block, g1_tile_n, g1_tile_k, g1_occ, "
                "g2_tile_n, g2_tile_k, g2_occ)."
            )
        block_size, g1_n, g1_k, g1_occ, g2_n, g2_k, g2_occ = map(int, tactic)
        if block_size not in self._block_sizes:
            raise ValueError(
                f"cuTile BF16 block size must be one of {self._block_sizes}, "
                f"got {block_size}."
            )
        for stage, config in (
            (1, (g1_n, g1_k, g1_occ)),
            (2, (g2_n, g2_k, g2_occ)),
        ):
            problem = self._gemm_problem(inputs, stage=stage, block_size=block_size)
            reason = _cutile_bf16_config_rejection_reason(problem, config)
            if reason is not None:
                raise NotImplementedError(
                    f"CuTileBf16Runner GEMM{stage} tactic is unsupported: {reason}."
                )
        self._ensure_workspace(inputs[1].shape[0], hidden_size)
        gemm1_config = self._kernel_module.GemmConfig(g1_n, g1_k, g1_occ)
        gemm2_config = self._kernel_module.GemmConfig(g2_n, g2_k, g2_occ)
        return self._kernel_module.run_moe(
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[0],
            self._workspace,
            activation_type=self.config.activation.type,
            block_size=block_size,
            gemm1_config=gemm1_config,
            gemm2_config=gemm2_config,
        )

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + (
            self._device_arch,
            self._num_sms,
            self._block_sizes,
        )


# ---------------------------------------------------------------------------
# cuTile NVFP4 runners
# ---------------------------------------------------------------------------


_CUTILE_W4A4_GATED_FUSION_MIN_ASSIGNMENTS = 64
_CUTILE_W4A4_GEMM_CONFIGS = (
    (128, 128, 2),
    (256, 64, 2),
    (256, 128, 1),
    (128, 64, 2),
    (128, 64, 4),
    (128, 256, 2),
    (256, 64, 1),
    (256, 128, 2),
    (256, 256, 1),
)


def _cutile_w4a4_config_rejection_reason(
    problem: _CuTileGemmProblem, config: tuple[int, int, int]
) -> str | None:
    if config not in _CUTILE_W4A4_GEMM_CONFIGS:
        return f"config={config} is outside the supported cuTile W4A4 tile set"
    tile_n, tile_k, _ = config
    if tile_n % 128 != 0 or tile_k % 64 != 0:
        return "W4A4 requires tile_n divisible by 128 and tile_k by 64"
    effective_tile_n = (
        tile_n // 2
        if problem.stage == 1 and problem.fused_epilogue and problem.is_gated
        else tile_n
    )
    if effective_tile_n > 2 * problem.n:
        return f"effective tile_n={effective_tile_n} exceeds 2*N={2 * problem.n}"
    if tile_k > 2 * problem.k:
        return f"tile_k={tile_k} exceeds 2*K={2 * problem.k}"
    if problem.fused_epilogue:
        if problem.stage != 1:
            return "only GEMM1 supports a fused activation epilogue"
        if effective_tile_n < 128:
            return (
                "fused GEMM1 requires an effective tile_n of at least 128, "
                f"got {effective_tile_n}"
            )
        if problem.is_gated and problem.n % effective_tile_n != 0:
            return (
                f"gated fused GEMM1 requires N={problem.n} to be divisible by "
                f"effective tile_n={effective_tile_n}"
            )
    return None


def _cutile_w4a4_gemm_configs(
    problem: _CuTileGemmProblem,
) -> list[tuple[int, int, int]]:
    return _valid_cutile_gemm_configs(
        problem,
        _CUTILE_W4A4_GEMM_CONFIGS,
        _cutile_w4a4_config_rejection_reason,
        "CuTileNvfp4Runner",
    )


class CuTileNvfp4Runner(CuTileBf16Runner):
    """cuTile NVFP4 weights and inputs for both grouped GEMMs."""

    backend_key = "cutile_nvfp4"
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = (QuantVariant.NVFP4,)
    _block_sizes = (16, 32, 64, 128)
    _supported_archs = _CUTILE_NVFP4_ARCHS
    _precision_name = "NVFP4"

    def _build(self) -> None:
        from .cutile import fp4

        self._kernel_module = fp4

    def _candidate_block_sizes(self, num_assignments: int) -> tuple[int, ...]:
        num_experts = self.config.routing.num_experts
        rows_per_expert = (num_assignments + num_experts - 1) // num_experts
        if rows_per_expert <= 8:
            return (16, 32)
        intermediate_size = self.config.experts.intermediate_size
        if num_assignments >= 4096 and (
            intermediate_size >= 1024 or num_assignments >= 65536
        ):
            return (32, 64)
        return (32, 64, 128)

    def _w4a4_gemm_problem(
        self,
        inputs: List[torch.Tensor],
        *,
        stage: int,
        block_size: int,
        fuse_gemm1: bool | None = None,
    ) -> _CuTileGemmProblem:
        hidden_size = inputs[1].shape[1]
        intermediate_size = self.config.experts.intermediate_size
        if stage == 1:
            if fuse_gemm1 is None:
                fuse_gemm1 = True
            n = intermediate_size * (
                2 if self.config.activation.is_gated and not fuse_gemm1 else 1
            )
            k = hidden_size
        elif stage == 2:
            fuse_gemm1 = False
            n, k = hidden_size, intermediate_size
        else:
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        num_assignments = inputs[2].numel()
        from .cutile.fp4 import _use_row_major_scale_layout

        input_sorted = _use_row_major_scale_layout(num_assignments, intermediate_size)
        return _CuTileGemmProblem(
            stage=stage,
            arch=self._device_arch,
            block_size=block_size,
            n=n,
            k=k,
            fused_epilogue=bool(fuse_gemm1),
            is_gated=self.config.activation.is_gated,
            input_sorted=input_sorted,
        )

    def _w4a4_fused_gemm1_supported(
        self, inputs: List[torch.Tensor], block_size: int
    ) -> bool:
        problem = self._w4a4_gemm_problem(
            inputs, stage=1, block_size=block_size, fuse_gemm1=True
        )
        return any(
            _cutile_w4a4_config_rejection_reason(problem, config) is None
            for config in _CUTILE_W4A4_GEMM_CONFIGS
        )

    def _w4a4_fusion_modes(
        self, inputs: List[torch.Tensor], block_size: int
    ) -> tuple[bool, ...]:
        if not self.config.activation.is_gated:
            return (True,)
        fallback = self._w4a4_fallback_fuse_gemm1(inputs, block_size)
        if (
            inputs[2].numel() < _CUTILE_W4A4_GATED_FUSION_MIN_ASSIGNMENTS
            or not self._w4a4_fused_gemm1_supported(inputs, block_size)
        ):
            return (False,)
        return (fallback, not fallback)

    def _stage_tactics(
        self,
        inputs: List[torch.Tensor],
        *,
        stage: int,
        block_size: int,
    ) -> List[Any]:
        if stage == 2:
            return _cutile_w4a4_gemm_configs(
                self._w4a4_gemm_problem(inputs, stage=2, block_size=block_size)
            )
        return list(
            dict.fromkeys(
                (int(fuse_gemm1), *config)
                for fuse_gemm1 in self._w4a4_fusion_modes(inputs, block_size)
                for config in _cutile_w4a4_gemm_configs(
                    self._w4a4_gemm_problem(
                        inputs,
                        stage=1,
                        block_size=block_size,
                        fuse_gemm1=fuse_gemm1,
                    )
                )
            )
        )

    def _stage_tactic_slice(self, stage: int) -> slice:
        return slice(1, 5) if stage == 1 else slice(5, 8)

    def _stage_cache_key(
        self,
        inputs: List[torch.Tensor],
        *,
        stage: int,
        block_size: int,
    ) -> tuple[Any, ...]:
        problem = self._w4a4_gemm_problem(
            inputs,
            stage=stage,
            block_size=block_size,
            fuse_gemm1=False,
        )
        return (
            "cutile_w4a4_stage",
            self._device_arch,
            self._num_sms,
            stage,
            block_size,
            self._w4a4_fusion_modes(inputs, block_size) if stage == 1 else (False,),
            problem.input_sorted,
            int(self.config.activation.type),
            self.config.routing.num_experts,
            self.config.routing.top_k,
            self.config.experts.intermediate_size,
        )

    def _w4a4_fallback_fuse_gemm1(
        self, inputs: List[torch.Tensor], block_size: int
    ) -> bool:
        if not self.config.activation.is_gated:
            return True
        num_assignments = inputs[2].numel()
        rows_per_expert = (
            num_assignments + self.config.routing.num_experts - 1
        ) // self.config.routing.num_experts
        return (
            num_assignments >= _CUTILE_W4A4_GATED_FUSION_MIN_ASSIGNMENTS
            and block_size < 128
            and not (rows_per_expert >= 16 and rows_per_expert < block_size)
            and self._w4a4_fused_gemm1_supported(inputs, block_size)
        )

    def _w4a4_gated_block_size(self, num_assignments: int) -> int:
        rows_per_expert = (
            num_assignments + self.config.routing.num_experts - 1
        ) // self.config.routing.num_experts
        if rows_per_expert <= 8:
            return 16
        if rows_per_expert <= 32:
            return 32
        if rows_per_expert <= 64:
            return 64
        return 128 if rows_per_expert <= 128 else 64

    def _w4a4_block_size(self, num_assignments: int) -> int:
        if self.config.activation.is_gated:
            return self._w4a4_gated_block_size(num_assignments)
        candidates = self._candidate_block_sizes(num_assignments)
        rows_per_expert = (
            num_assignments + self.config.routing.num_experts - 1
        ) // self.config.routing.num_experts
        if rows_per_expert <= 32:
            return candidates[0]
        if rows_per_expert <= 64 and 64 in candidates:
            return 64
        if rows_per_expert <= 128 and 32 in candidates:
            return 32
        return 64 if 64 in candidates else candidates[0]

    def _w4a4_fallback_tactic(
        self, inputs: List[torch.Tensor]
    ) -> tuple[int, int, int, int, int, int, int, int]:
        num_assignments = inputs[2].numel()
        block_size = self._w4a4_block_size(num_assignments)
        if self.config.activation.is_gated:
            small_defaults = (
                (8, (0, 128, 256, 2, 256, 128, 2)),
                (16, (0, 256, 256, 1, 256, 256, 1)),
                (32, (0, 256, 128, 2, 128, 128, 2)),
                (64, (1, 256, 128, 2, 128, 128, 2)),
            )
            for limit, stage_tactics in small_defaults:
                if num_assignments > limit:
                    continue
                fuse_gemm1 = bool(stage_tactics[0])
                gemm1_problem = self._w4a4_gemm_problem(
                    inputs,
                    stage=1,
                    block_size=block_size,
                    fuse_gemm1=fuse_gemm1,
                )
                gemm2_problem = self._w4a4_gemm_problem(
                    inputs, stage=2, block_size=block_size
                )
                if (
                    _cutile_w4a4_config_rejection_reason(
                        gemm1_problem, stage_tactics[1:4]
                    )
                    is None
                    and _cutile_w4a4_config_rejection_reason(
                        gemm2_problem, stage_tactics[4:7]
                    )
                    is None
                ):
                    return (block_size, *stage_tactics)
                break
        fuse_gemm1 = self._w4a4_fallback_fuse_gemm1(inputs, block_size)
        gemm1 = _cutile_w4a4_gemm_configs(
            self._w4a4_gemm_problem(
                inputs,
                stage=1,
                block_size=block_size,
                fuse_gemm1=fuse_gemm1,
            )
        )[0]
        gemm2 = _cutile_w4a4_gemm_configs(
            self._w4a4_gemm_problem(inputs, stage=2, block_size=block_size)
        )[0]
        return (block_size, int(fuse_gemm1), *gemm1, *gemm2)

    def _factorized_w4a4_tactics(
        self, inputs: List[torch.Tensor]
    ) -> list[tuple[int, ...]]:
        fallback = self._w4a4_fallback_tactic(inputs)
        block_size = fallback[0]
        gemm1_runner = _CuTileStageRunner(self, 1, block_size, fallback)
        gemm2_runner = _CuTileStageRunner(self, 2, block_size, fallback)
        activation_name = self.config.activation.type.name.lower()
        return _factorized_cutile_tactics(
            self,
            inputs,
            block_size,
            gemm1_runner,
            gemm2_runner,
            fallback[1:5],
            fallback[5:8],
            name_suffix=f"_{activation_name}",
        )

    def get_valid_tactics(self, inputs: List[torch.Tensor], _profile: Any) -> List[Any]:
        self._require_built()
        if len(inputs) != 10:
            raise ValueError(
                f"{type(self).__name__} expects 10 inputs, got {len(inputs)}."
            )
        return self._factorized_w4a4_tactics(inputs)

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        hidden_states, num_tokens, hidden_size = self._validate_inputs(act)

        view = weights.get_view(self.backend_key)
        required = (
            "w1",
            "w1_scale",
            "w1_global_scale",
            "w2",
            "w2_scale",
            "w2_global_scale",
        )
        missing = [key for key in required if key not in view]
        if missing:
            raise KeyError(
                f"{self.backend_key} prepared weights are missing {missing}."
            )
        w1, w1_scale, w1_global, w2, w2_scale, w2_global = (
            view[key] for key in required
        )
        num_experts = self.config.routing.num_experts
        intermediate_size = self.config.experts.intermediate_size
        w1_rows = intermediate_size * (2 if self.config.activation.is_gated else 1)
        expected_w1 = (num_experts, w1_rows, hidden_size // 2)
        expected_w2 = (num_experts, hidden_size, intermediate_size // 2)
        if tuple(w1.shape) != expected_w1 or tuple(w2.shape) != expected_w2:
            raise ValueError(
                f"cuTile NVFP4 weight shapes {tuple(w1.shape)}/{tuple(w2.shape)} "
                f"!= expected {expected_w1}/{expected_w2}."
            )
        expected_s1: tuple[int, ...] = (
            num_experts,
            (w1_rows + 127) // 128,
            hidden_size // 64,
            32,
            16,
        )
        expected_s2: tuple[int, ...] = (
            num_experts,
            (hidden_size + 127) // 128,
            intermediate_size // 64,
            32,
            16,
        )
        if tuple(w1_scale.shape) != expected_s1 or tuple(w2_scale.shape) != expected_s2:
            raise ValueError(
                "cuTile NVFP4 block-scale shapes "
                f"{tuple(w1_scale.shape)}/{tuple(w2_scale.shape)} != "
                f"expected {expected_s1}/{expected_s2}."
            )
        if w1.dtype is not torch.uint8 or w2.dtype is not torch.uint8:
            raise TypeError("cuTile NVFP4 prepared weights must use torch.uint8.")
        if (
            w1_scale.dtype is not torch.float8_e4m3fn
            or w2_scale.dtype is not torch.float8_e4m3fn
        ):
            raise TypeError("cuTile NVFP4 block scales must use float8_e4m3fn.")
        if w1_global.dtype is not torch.float32 or w2_global.dtype is not torch.float32:
            raise TypeError("cuTile NVFP4 global scales must use torch.float32.")
        if any(
            t.device != self.device
            for t in (w1, w1_scale, w1_global, w2, w2_scale, w2_global)
        ):
            raise ValueError(
                "cuTile NVFP4 prepared weights must match the runner device."
            )
        if any(
            not t.is_contiguous()
            for t in (w1, w1_scale, w1_global, w2, w2_scale, w2_global)
        ):
            raise ValueError("cuTile NVFP4 prepared weights must be contiguous.")

        capacity = map_to_hybrid_bucket(
            num_tokens, self.config.execution.tune_max_num_tokens
        )
        num_assignments = capacity * self.config.routing.top_k
        max_padded_assignments = num_assignments + num_experts * (
            max(self._block_sizes) - 1
        )
        _validate_cutile_int32_routing(
            type(self).__name__, num_assignments, max_padded_assignments
        )
        self._configure_tuning(num_tokens, hidden_size)
        return [
            hidden_states.new_empty((num_tokens, hidden_size)),
            hidden_states,
            act.topk_ids,
            act.topk_weights,
            w1,
            w1_scale,
            w1_global,
            w2,
            w2_scale,
            w2_global,
        ]

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_built()
        if len(inputs) != 10:
            raise ValueError(
                f"{type(self).__name__} expects 10 inputs, got {len(inputs)}."
            )
        if tactic == -1:
            tactic = self._w4a4_fallback_tactic(inputs)
        if not isinstance(tactic, (tuple, list)) or len(tactic) != 8:
            raise ValueError(
                f"{type(self).__name__} tactic must be -1 or an eight-integer tuple."
            )
        (
            block_size,
            fuse_gemm1,
            g1_n,
            g1_k,
            g1_occ,
            g2_n,
            g2_k,
            g2_occ,
        ) = map(int, tactic)
        if fuse_gemm1 not in (0, 1):
            raise ValueError("cuTile NVFP4 GEMM1 fusion flag must be 0 or 1.")
        if block_size not in self._block_sizes:
            raise ValueError(
                f"cuTile NVFP4 block size must be one of {self._block_sizes}, "
                f"got {block_size}."
            )
        for stage, config in (
            (1, (g1_n, g1_k, g1_occ)),
            (2, (g2_n, g2_k, g2_occ)),
        ):
            problem = self._w4a4_gemm_problem(
                inputs,
                stage=stage,
                block_size=block_size,
                fuse_gemm1=bool(fuse_gemm1) if stage == 1 else False,
            )
            reason = _cutile_w4a4_config_rejection_reason(problem, config)
            if reason is not None:
                raise NotImplementedError(
                    f"CuTileNvfp4Runner GEMM{stage} tactic is unsupported: {reason}."
                )
        hidden_size = inputs[1].shape[1]
        self._ensure_workspace(inputs[1].shape[0], hidden_size)
        return self._kernel_module.run_moe(
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[8],
            inputs[9],
            inputs[0],
            self._workspace,
            activation_type=self.config.activation.type,
            fuse_gemm1=bool(fuse_gemm1),
            num_sms=self._num_sms,
            block_size=block_size,
            gemm1_config=self._kernel_module.GemmConfig(g1_n, g1_k, g1_occ),
            gemm2_config=self._kernel_module.GemmConfig(g2_n, g2_k, g2_occ),
        )


# ---------------------------------------------------------------------------
# CuteDSL runner — delegates to the matching W4A4, W4A8, or W4A16 runner
# ---------------------------------------------------------------------------


class CuteDslRunner(MoERunner):
    """Translate activation and weight packs into a CuTe DSL runner input list."""

    backend_key = "cute_dsl"
    # CuteDSL has no in-kernel router; it only consumes pre-routed packs.
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = (
        QuantVariant.NVFP4,
        QuantVariant.MXFP4,
        QuantVariant.W4A16,
    )
    supported_activation_classes = (SwiGLU, GeGLUTanh, ReLU2, SiTU)

    def _check_support(self) -> None:
        super()._check_support()
        if isinstance(self.config.activation, SiTU) and (
            self.config.activation.clamp_limit is not None
        ):
            raise NotImplementedError(
                f"{type(self).__name__} SiTU does not expose a separate clamp_limit."
            )
        if not self.config.finalize.do_finalize:
            raise NotImplementedError(
                f"{type(self).__name__} requires do_finalize=True."
            )
        if (
            self.config.quant.variant is not QuantVariant.NVFP4
            and self.config.quant.per_token_scale
        ):
            raise NotImplementedError(
                f"{type(self).__name__} does not support per-token "
                f"{self.config.quant.variant.name} activation scales."
            )
        if self.config.quant.variant is QuantVariant.MXFP4 and isinstance(
            self.config.activation, SiTU
        ):
            raise NotImplementedError("CuTe-DSL W4A8 does not support SiTU.")
        if (
            self.config.quant.variant is QuantVariant.MXFP4
            and not self.config.finalize.use_fused_finalize
        ):
            raise NotImplementedError("CuTe-DSL W4A8 requires fused finalize.")
        if self.config.quant.variant is QuantVariant.MXFP4 and hasattr(self, "device"):
            from ..utils import get_compute_capability

            if get_compute_capability(self.device) == (10, 7):
                raise NotImplementedError("CuTe-DSL W4A8 does not support SM107.")
        self._assert_rubin_cute_dsl_available()

    def _assert_rubin_cute_dsl_available(self) -> None:
        """Reject SM107 when the installed CuTe DSL cannot provide its kernels.

        The SM107 gather/activation-fusion and finalize-fusion kernels are built
        on ``cutlass.utils.rubin_helpers``, which only exists from CuTe DSL 4.8.
        Without it the kernel factories raise ``NotImplementedError`` when they
        are first called -- that is, in the middle of ``forward()``, long after
        this backend has been accepted as a candidate.

        Declining here instead lets ``MoELayer`` drop the backend at build time,
        so ``auto`` routes elsewhere and callers that enumerate backends see it
        absent rather than failing mid-call.

        The probe is arch-conditional on purpose: only the SM107 kernels need
        ``rubin_helpers``, so an older DSL is perfectly usable on SM100/SM103.
        """
        from ..utils import get_compute_capability

        # #4787 put this on CuteDslNvfp4Runner; #4793 unified that into
        # CuteDslRunner. Keep the original blast radius -- only the NVFP4 path
        # reaches the SM107 rubin kernels. MXFP4/W4A8 is already declined on
        # SM107 above, and W4A16 gates itself via require_cute_dsl_arch().
        if self.config.quant.variant is not QuantVariant.NVFP4:
            return

        # check_support() is also exercised on runners built with __new__ and only
        # a config attached (see TestMoERunnerSupport), so there may be no bound
        # device. Nothing arch-specific can be decided in that case; the real
        # dispatch path always sets device in __init__ before check_support().
        device = getattr(self, "device", None)
        if device is None:
            return
        if get_compute_capability(device) != (10, 7):
            return

        # ``cute_dsl.utils`` imports cutlass at module scope, so on a stack with no
        # CuTe DSL installed the probe cannot be reached at all. Failing to import
        # it is itself proof the SM107 kernels are unavailable, so decline rather
        # than propagating an ImportError out of a support check.
        #
        # #4753 adds a cutlass-free ``cute_dsl.availability`` module and reroutes
        # the package off ``utils``; once that lands this collapses to a plain
        # ``from ..cute_dsl.availability import is_rubin_cute_dsl_available``,
        # matching what release-v0.6.18 already does. The try/except is kept so
        # this commit is correct whichever of the two merges first.
        try:
            from ..cute_dsl.utils import is_rubin_cute_dsl_available

            rubin_dsl_available = is_rubin_cute_dsl_available()
        except ImportError:
            rubin_dsl_available = False

        if not rubin_dsl_available:
            raise NotImplementedError(
                f"{type(self).__name__} requires CuTe DSL >= 4.8 on SM107 "
                "(Rubin), which provides cutlass.utils.rubin_helpers; the "
                "installed CuTe DSL does not have it."
            )

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self._inner: Any = None
        self.tuning_config = TuningConfig()

    def _build(self) -> None:
        """Create the shape-independent CuTe DSL tuning runner."""
        from .cute_dsl.fused_moe import _cute_dsl_fused_moe_impl
        from .cute_dsl.tuner import (
            CuteDslFusedMoERunner,
            CuteDslFusedMoEW4A16Runner,
        )

        experts = self.config.experts
        routing = self.config.routing
        num_local_experts = experts.local_num_experts or routing.num_experts
        enable_pdl = (
            True
            if self.config.execution.enable_pdl is None
            else self.config.execution.enable_pdl
        )
        if self.config.quant.variant in (QuantVariant.NVFP4, QuantVariant.MXFP4):
            self._inner = CuteDslFusedMoERunner(
                forward_impl=_cute_dsl_fused_moe_impl,
                num_experts=routing.num_experts,
                top_k=routing.top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=experts.local_expert_offset,
                use_fused_finalize=self.config.finalize.use_fused_finalize,
                enable_pdl=enable_pdl,
                use_per_token_activation=bool(self.config.quant.per_token_scale),
                quant_mode=(
                    "w4a8"
                    if self.config.quant.variant is QuantVariant.MXFP4
                    else "w4a4"
                ),
                **_cute_dsl_activation_kwargs(self.config.activation),
            )
        elif self.config.quant.variant is QuantVariant.W4A16:
            self._inner = CuteDslFusedMoEW4A16Runner(
                num_experts=routing.num_experts,
                top_k=routing.top_k,
                num_local_experts=num_local_experts,
                local_expert_offset=experts.local_expert_offset,
                use_fused_finalize=self.config.finalize.use_fused_finalize,
                enable_pdl=enable_pdl,
                **_cute_dsl_activation_kwargs(self.config.activation),
            )
        else:
            raise NotImplementedError(
                f"CuteDslRunner does not support {self.config.quant.variant}."
            )
        # tuning_config is an instance attribute on the inner runner (its
        # dummy expert-id span depends on num_experts/offset), so read it from
        # the instance we just built, not off the class.
        self.tuning_config = self._inner.tuning_config

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: Any) -> List[Any]:
        self._require_built()
        return self._inner.get_valid_tactics(inputs, profile)

    def _cache_key_extras(self) -> tuple:
        return super()._cache_key_extras() + (
            bool(self._inner.use_fused_finalize),
            bool(self._inner.enable_pdl),
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._require_built()
        return self._inner.forward(
            inputs, tactic=tactic, do_preparation=do_preparation, **kwargs
        )

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        """Translate packs into the selected CuTe DSL runner's input list.

        Expected weight view keys: w1_weight, w1_weight_sf, w1_alpha,
        fc2_input_scale, w2_weight, w2_weight_sf, w2_alpha.
        The W4A4 per-token path inserts ``per_token_scale`` before the trailing
        ``moe_output`` buffer. W4A16 uses its own compact input layout. Both
        tuning configurations include the output buffer so profiling can replace
        it for each token bucket.
        """
        self._require_built()
        # MoELayer already filters by supported_routing_modes; this guards the
        # direct-runner path (tests/benchmarks) against silently forwarding a
        # logits pack's None topk tensors into the kernel launch.
        if act.routing_input_mode not in self.supported_routing_modes:
            raise NotImplementedError(
                f"CuteDslRunner does not support "
                f"routing_input_mode={act.routing_input_mode!r} "
                "(only PackedPrecomputed is wired; CuteDSL has no in-kernel router)."
            )
        v = weights.get_view(self.backend_key)
        num_tokens = act.hidden_states_q.shape[0]
        _validate_prerouted_inputs(act, num_tokens, self._inner.top_k, "CuteDslRunner")
        # prepare_weights defaults to SwiGLU, so a non-gated config paired with a
        # default-prepared view yields 2I rows. The tuner infers intermediate_size
        # from this tensor, so the mismatch would surface as a shape error deep in
        # the kernel rather than here.
        expected_rows = self.config.experts.intermediate_size * (
            2 if self.config.activation.is_gated else 1
        )
        actual_rows = v["w1_weight"].shape[1]
        if actual_rows != expected_rows:
            raise ValueError(
                f"CuteDslRunner: w1_weight has {actual_rows} GEMM1 rows, "
                f"expected {expected_rows} for "
                f"{type(self.config.activation).__name__}; prepare the view with "
                "the same typed activation."
            )

        quant_variant = self.config.quant.variant
        use_per_token_activation = bool(self.config.quant.per_token_scale)
        if (
            quant_variant in (QuantVariant.NVFP4, QuantVariant.MXFP4)
            and not use_per_token_activation
            and act.hidden_states_scale is not None
            and act.per_token_scale is None
        ):
            is_mxfp4 = quant_variant is QuantVariant.MXFP4
            hidden_size = act.hidden_states_q.shape[1] * (1 if is_mxfp4 else 2)
            moe_output = act.hidden_states_q.new_empty(
                (num_tokens, hidden_size), dtype=torch.bfloat16
            )
            return [
                act.hidden_states_q,
                (
                    act.hidden_states_scale
                    if is_mxfp4
                    else act.hidden_states_scale.unsqueeze(-1)
                ).view(torch.uint8 if is_mxfp4 else act.hidden_states_scale.dtype),
                act.topk_ids,
                act.topk_weights,
                v["w1_weight"],
                v["w1_weight_sf"],
                v["w1_alpha"],
                None if is_mxfp4 else v["fc2_input_scale"],
                v["w2_weight"],
                v["w2_weight_sf"],
                v["w2_alpha"],
                moe_output,
            ]
        elif (
            quant_variant is QuantVariant.NVFP4
            and use_per_token_activation
            and act.hidden_states_scale is not None
            and act.per_token_scale is not None
        ):
            hidden_size = act.hidden_states_q.shape[1] * 2  # FP4 packed
            moe_output = act.hidden_states_q.new_empty(
                (num_tokens, hidden_size), dtype=torch.bfloat16
            )
            return [
                act.hidden_states_q,
                act.hidden_states_scale.unsqueeze(-1),
                act.topk_ids,
                act.topk_weights,
                v["w1_weight"],
                v["w1_weight_sf"],
                v["w1_alpha"],
                v["fc2_input_scale"],
                v["w2_weight"],
                v["w2_weight_sf"],
                v["w2_alpha"],
                act.per_token_scale,
                moe_output,
            ]
        elif (
            quant_variant is QuantVariant.W4A16
            and act.hidden_states_scale is None
            and act.per_token_scale is None
        ):
            hidden_size = act.hidden_states_q.shape[1]
            moe_output = act.hidden_states_q.new_empty(
                (num_tokens, hidden_size), dtype=torch.bfloat16
            )
            return [
                act.hidden_states_q,
                act.topk_ids,
                act.topk_weights,
                v["w1_weight"],
                v["w1_weight_sf"],
                v["w1_alpha"],
                v["w2_weight"],
                v["w2_weight_sf"],
                v["w2_alpha"],
                moe_output,
            ]
        else:
            raise ValueError(
                "CuteDslRunner activation inputs must match W4A4, W4A4 "
                "per-token, W4A8, or W4A16"
            )


# ---------------------------------------------------------------------------
# TRTLLM runners — shared module lifecycle, shape-specific inner runners
# ---------------------------------------------------------------------------


class _TrtllmRunnerBase(MoERunner):
    """Load the shared TRTLLM-gen module after support validation."""

    _module: Any
    _inner: Any
    _static_kwargs: dict[str, Any]

    def _build(self) -> None:
        from .core import get_trtllm_moe_sm100_module

        self._module = get_trtllm_moe_sm100_module()

    def _forward_inner(
        self,
        inputs: List[torch.Tensor],
        tactic: Any,
        do_preparation: bool,
    ) -> torch.Tensor | List[torch.Tensor]:
        result = self._inner.forward(
            inputs,
            tactic=tactic,
            do_preparation=do_preparation,
            **self._static_kwargs,
        )
        if self.config.finalize.do_finalize:
            return inputs[0]
        return result


class TrtllmFp4RoutedRunner(_TrtllmRunnerBase):
    """FP4 adapter over the canonical trtllm-gen ``MoERunner``.

    Translates (MoEActivationPack, MoEWeightPack) into the ``MoeRunnerInputs`` list
    plus the static weight/config kwargs that ``core.MoERunner.forward``
    consumes, then delegates tactic enumeration, tuning-config construction, and
    the tactic'd forward to that inner runner.  This mirrors
    ``CuteDslRunner`` (which wraps ``CuteDslFusedMoERunner``) and keeps
    the fragile raw-op positional launch in exactly one place —
    ``core.MoERunner.forward``.

    Routing mode is chosen per-call from ``act.routing_input_mode``:

    * **pre-routed** (``RoutingInputMode.PackedPrecomputed``): the pack carries
      ``topk_ids`` / ``topk_weights`` and the runner packs them into int32 top-k ids
      ``(GLOBAL expert_id << 16) | bf16(weight)`` (the kernel maps global ids to the
      local shard via the separately passed ``local_expert_offset``).
    * **unpacked pre-routed** (``RoutingInputMode.UnpackedPrecomputed``): the
      pack carries plain int32 ids plus BF16 or FP32 weights, which are
      forwarded as separate kernel inputs without packed-id construction.
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
        RoutingInputMode.UnpackedPrecomputed,
        RoutingInputMode.FromLogits,
    )
    supported_quant_variants = (
        QuantVariant.NVFP4,
        QuantVariant.MXFP4,
        QuantVariant.W4A16,
    )
    supports_fused_shared_experts = True
    supported_activation_classes_by_quant: ClassVar[
        dict[QuantVariant, tuple[type[ActivationConfig], ...]]
    ] = {
        QuantVariant.NVFP4: (SwiGLU, GeGLU, SiTU, ReLU2),
        QuantVariant.MXFP4: (SwiGLU, GeGLU, SiTU, ReLU2),
        QuantVariant.W4A16: (SwiGLU,),
    }

    def _check_support(self) -> None:
        super()._check_support()
        activation = self.config.activation
        if isinstance(activation, SiTU) and activation.linear_scale is None:
            # gemm1_beta is a per-expert float tensor with no encoding for
            # "unclamped"; only the CuTe-DSL scalar ABI can express it.
            raise NotImplementedError(
                f"{type(self).__name__} cannot express SiTU(linear_scale=None); "
                "the TRT-LLM ABI has no unclamped linear-branch encoding."
            )
        variant = self.config.quant.variant
        if variant in self.supported_quant_variants:
            if self.config.quant.per_token_scale and variant is not QuantVariant.NVFP4:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support per-token scale for {variant.name}."
                )

            from ..utils import get_compute_capability

            # Direct-runner guard: #4280 relanded the SM107 cubins removed by
            # #4171. NVFP4/MXFP4 support SM100/SM103/SM107; W4A16 supports
            # SM100/SM107 and retains the upstream SM103 xfail in #1754.
            compute_capability = get_compute_capability(self.device)
            if compute_capability == (10, 7) and isinstance(activation, SiTU):
                # The pinned Rubin BMM lacks SiTuGlu; its enum value collides
                # with None and would silently disable the activation.
                # TODO: Update TRTLLM_GEN_BMM_RUBIN to an artifact with
                # SiTuGlu, then remove this guard and add SM107 parity coverage.
                raise NotImplementedError(
                    f"{type(self).__name__} does not support SiTU on SM107 with "
                    "the currently pinned Rubin BMM artifact."
                )
            if variant in (QuantVariant.NVFP4, QuantVariant.MXFP4):
                supported = compute_capability in ((10, 0), (10, 3), (10, 7))
            else:
                supported = compute_capability in ((10, 0), (10, 7))
            if not supported:
                raise NotImplementedError(
                    f"TRTLLM {variant.name} is unsupported on "
                    f"SM{compute_capability[0]}{compute_capability[1]}."
                )

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        from ..tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
        from ..utils import device_support_pdl

        self.config = config
        self.device = device
        self._module: Any = None

        routing = config.routing
        experts = config.experts
        execution = config.execution
        self._num_local_experts = experts.local_num_experts or routing.num_experts
        self._num_fused_shared_experts = experts.num_fused_shared_experts
        self._num_weight_rows = self._num_local_experts + self._num_fused_shared_experts
        self._local_expert_offset = experts.local_expert_offset
        self._intermediate_size = experts.intermediate_size
        self._activation_type = int(config.activation.type)
        self._tune_max_num_tokens = execution.tune_max_num_tokens

        variant = config.quant.variant
        if variant is QuantVariant.MXFP4:
            dtype_act = DtypeTrtllmGen.MxE4m3
            dtype_weights = DtypeTrtllmGen.MxE2m1
        elif variant is QuantVariant.W4A16:
            dtype_act = DtypeTrtllmGen.Bfloat16
            dtype_weights = DtypeTrtllmGen.MxE2m1
        else:
            # Harmless construction default; check_support rejects unknown variants.
            dtype_act = DtypeTrtllmGen.E2m1
            dtype_weights = DtypeTrtllmGen.E2m1
        self._variant = variant
        self._dtype_act = dtype_act
        self._dtype_weights = dtype_weights
        self._fp8_quantization_type = Fp8QuantizationType.NoneFp8
        self._per_token = bool(self.config.quant.per_token_scale)

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
        self._require_built()
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
            use_per_token_scaling=self._per_token,
            num_experts=self.config.routing.num_experts,
            num_fused_shared_experts=self._num_fused_shared_experts,
        )

    def get_valid_tactics(  # type: ignore[override]
        self, inputs: List[torch.Tensor], profile: Any
    ) -> List[Any]:
        self._require_built()
        # The inner runner reads num_tokens from inputs + its own instance key;
        # no static kwargs are needed for tactic enumeration.
        return self._inner.get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | List[torch.Tensor]:
        self._require_built()
        # MoELayer's autotuner call passes no kwargs, so the static weight/config
        # kwargs are injected here. Finalized calls write into inputs[0];
        # unfinalized calls return the flat API's three-tensor result.
        return self._forward_inner(inputs, tactic, do_preparation)

    def _validate_fp4_tensors(
        self,
        act: MoEActivationPack,
        view: dict,
        hidden_size: int,
    ) -> torch.Tensor | None:
        num_tokens = act.hidden_states_q.shape[0]
        if self._variant is QuantVariant.NVFP4:
            if act.hidden_states_q.dtype != torch.uint8:
                raise TypeError(
                    "NVFP4 hidden_states_q must be packed uint8, got "
                    f"{act.hidden_states_q.dtype}."
                )
            scale = act.hidden_states_scale
            expected_scale = (num_tokens, hidden_size // 16)
            if (
                scale is None
                or scale.dtype not in (torch.uint8, torch.float8_e4m3fn)
                or tuple(scale.shape) != expected_scale
            ):
                got = None if scale is None else (scale.dtype, tuple(scale.shape))
                raise ValueError(
                    "NVFP4 hidden_states_scale must have shape "
                    f"{expected_scale} and uint8/float8_e4m3fn storage, got {got}."
                )
            if scale.dtype == torch.uint8:
                scale = scale.view(torch.float8_e4m3fn)
            scale_dtype = torch.float8_e4m3fn
            sf_vec_size = 16
        elif self._variant is QuantVariant.MXFP4:
            if act.hidden_states_q.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    "MXFP4×MXFP8 hidden_states_q must be float8_e4m3fn, got "
                    f"{act.hidden_states_q.dtype}."
                )
            scale = act.hidden_states_scale
            expected_scale = (num_tokens, hidden_size // 32)
            if (
                scale is None
                or scale.dtype != torch.float8_e4m3fn
                or tuple(scale.shape) != expected_scale
            ):
                got = None if scale is None else (scale.dtype, tuple(scale.shape))
                raise ValueError(
                    "MXFP4×MXFP8 hidden_states_scale must carry UE8M0 bytes in "
                    f"a float8_e4m3fn tensor with shape {expected_scale}, got {got}."
                )
            scale_dtype = torch.float8_e4m3fn
            sf_vec_size = 32
        else:
            if act.hidden_states_q.dtype != torch.bfloat16:
                raise TypeError(
                    "TRTLLM W4A16 hidden_states_q must be bfloat16, got "
                    f"{act.hidden_states_q.dtype}."
                )
            if act.hidden_states_scale is not None:
                raise ValueError("TRTLLM W4A16 does not consume hidden_states_scale.")
            scale = None
            scale_dtype = torch.float8_e4m3fn
            sf_vec_size = 32

        expected_weights = {
            "gemm1_weights": (
                self._num_weight_rows,
                self._intermediate_size * (2 if self.config.activation.is_gated else 1),
                hidden_size // 2,
            ),
            "gemm2_weights": (
                self._num_weight_rows,
                hidden_size,
                self._intermediate_size // 2,
            ),
        }
        for name, expected in expected_weights.items():
            tensor = view[name]
            if tensor.dtype != torch.uint8 or tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must be packed uint8 with shape {expected}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}."
                )

        for name, expected in (
            (
                "gemm1_weights_scale",
                (
                    self._num_weight_rows,
                    self._intermediate_size
                    * (2 if self.config.activation.is_gated else 1),
                    hidden_size // sf_vec_size,
                ),
            ),
            (
                "gemm2_weights_scale",
                (
                    self._num_weight_rows,
                    hidden_size,
                    self._intermediate_size // sf_vec_size,
                ),
            ),
        ):
            tensor = view[name]
            if tensor.dtype != scale_dtype or tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must be {scale_dtype} with shape {expected}, got "
                    f"{tensor.dtype} {tuple(tensor.shape)}."
                )

        for name in (
            "gemm1_alpha",
            "gemm1_beta",
            "gemm1_clamp_limit",
            "output1_scale_scalar",
            "output1_scale_gate_scalar",
            "output2_scale_scalar",
        ):
            tensor = view.get(name)
            if tensor is None:
                continue
            if tensor.device != act.hidden_states_q.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected "
                    f"{act.hidden_states_q.device}."
                )
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} must be float32, got {tensor.dtype}.")
            if tuple(tensor.shape) != (self._num_weight_rows,):
                raise ValueError(
                    f"{name} must have shape ({self._num_weight_rows},), got "
                    f"{tuple(tensor.shape)}."
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous.")
        return scale

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        """Translate Packs → the ``MoeRunnerInputs`` list ``core.MoERunner`` expects.

        Expected weight view keys: gemm1_weights, gemm1_weights_scale,
        gemm1_alpha, gemm2_weights, gemm2_weights_scale, and optionally
        output1_scale_scalar, output1_scale_gate_scalar, output2_scale_scalar.

        Routing mode is read from ``act.routing_input_mode``: ``FromLogits`` drives
        in-kernel routing from ``act.routing_logits``; ``PackedPrecomputed`` packs the
        pre-routed ``act.topk_ids`` / ``act.topk_weights``; and
        ``UnpackedPrecomputed`` forwards int32 ids plus BF16 or FP32 weights directly.

        The local-shard offset comes from ``ExpertConfig.local_expert_offset``
        on the config this runner was built with.  ``topk_ids`` carries
        GLOBAL expert ids and is packed as-is; the kernel performs the
        global→local mapping itself by subtracting ``local_expert_offset``
        (passed via the static kwargs) and dropping ids outside
        ``[offset, offset + local_num_experts)``.
        """
        self._require_built()
        from .core import MoeRunnerInputs, RoutingInputMode

        v = weights.get_view(self.backend_key)
        _validate_prepared_activation_params(
            v, self.config.activation, type(self).__name__
        )
        routing = self.config.routing

        num_tokens = act.hidden_states_q.shape[0]
        hidden_size = (
            act.hidden_states_q.shape[1] * 2
            if self._variant is QuantVariant.NVFP4
            else act.hidden_states_q.shape[1]
        )
        hidden_states_scale = self._validate_fp4_tensors(act, v, hidden_size)

        output = act.hidden_states_q.new_empty(
            (
                num_tokens,
                hidden_size if self.config.finalize.do_finalize else 0,
            ),
            dtype=torch.bfloat16,
        )

        routing_input_mode = act.routing_input_mode
        if (
            self._num_fused_shared_experts > 0
            and routing_input_mode is not RoutingInputMode.FromLogits
        ):
            raise NotImplementedError(
                "Dedicated fused shared experts require FromLogits routing; "
                "pre-routed callers must append shared ids and weights themselves."
            )

        if self._per_token and act.per_token_scale is None:
            raise RuntimeError(
                "Per-token NVFP4 scale is configured but no activation scale is given."
            )

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
            if (
                self._variant in (QuantVariant.MXFP4, QuantVariant.W4A16)
                and routing_logits.dtype != torch.bfloat16
            ):
                raise TypeError(
                    f"{self._variant.name} FromLogits requires bfloat16 "
                    f"routing_logits, got {routing_logits.dtype}."
                )
            routing_bias = act.routing_bias
            topk_ids = act.hidden_states_q.new_empty(
                (
                    num_tokens,
                    routing.top_k + self._num_fused_shared_experts,
                ),
                dtype=torch.int32,
            )
            # MUST be bf16 regardless of logits dtype: the fp4 routing kernel
            # writes bf16 expert weights, so inheriting fp32 from the logits
            # mislabels the returned buffer (gh #3595 — the canonical wrapper
            # in core.py hardcodes bf16 for the same reason).
            expert_weights = routing_logits.new_empty(
                (
                    num_tokens,
                    routing.top_k + self._num_fused_shared_experts,
                ),
                dtype=torch.bfloat16,
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
            topk_ids = _pack_prerouted_topk_ids(act)
            # FP4 borrows this buffer but leaves its returned FFI slot undefined.
            # Supply the packed weights so _unpack_trtllm_moe_output() can return
            # them directly for do_finalize=False. Use BF16 to match the weights
            # encoded in topk_ids.
            expert_weights = act.topk_weights.to(torch.bfloat16).contiguous()
        elif routing_input_mode == RoutingInputMode.UnpackedPrecomputed:
            # UnpackedPrecomputed: both routing tensors are caller-owned kernel
            # inputs. Keep global ids intact; the launcher applies
            # local_expert_offset.
            _validate_prerouted_inputs(
                act,
                num_tokens,
                routing.top_k,
                "TrtllmFp4RoutedRunner",
                allowed_weights_dtypes=(torch.bfloat16, torch.float32),
                require_contiguous=True,
            )
            routing_logits = None
            routing_bias = None
            topk_ids = act.topk_ids
            expert_weights = act.topk_weights
        else:
            raise NotImplementedError(
                f"TrtllmFp4RoutedRunner does not support "
                f"routing_input_mode={routing_input_mode!r} "
                "(only FromLogits, PackedPrecomputed, and "
                "UnpackedPrecomputed are wired)."
            )

        moe_inputs = MoeRunnerInputs(
            output=output,
            routing_logits=routing_logits,
            topk_ids=topk_ids,
            expert_weights=expert_weights,
            hidden_states=act.hidden_states_q,
            hidden_states_scale=hidden_states_scale,
            gemm1_lora_delta=None,
            per_token_scale=act.per_token_scale,
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
            gemm1_beta=v.get("gemm1_beta"),
            gemm1_clamp_limit=v.get("gemm1_clamp_limit"),
            gemm2_weights=v["gemm2_weights"],
            gemm2_weights_scale=v["gemm2_weights_scale"],
            gemm2_bias=None,
            output1_scale_scalar=v.get("output1_scale_scalar"),
            output1_scale_gate_scalar=v.get("output1_scale_gate_scalar"),
            output2_scale_scalar=v.get("output2_scale_scalar"),
            per_token_scale=act.per_token_scale,
            num_experts=routing.num_experts,
            num_fused_shared_experts=self._num_fused_shared_experts,
            n_group=routing.n_group,
            topk_group=routing.topk_group,
            local_expert_offset=self._local_expert_offset,
            routed_scaling_factor=routing.routed_scaling_factor,
            routing_method_type=int(routing.method),
            do_finalize=self.config.finalize.do_finalize,
            enable_pdl=self._enable_pdl,
        )

        self._ensure_inner(hidden_size)
        # Reuse the inner runner's tuning-config builder so the num_tokens
        # buckets honor ExecutionConfig.tune_max_num_tokens (CR5).
        self.tuning_config = self._inner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=self._tune_max_num_tokens,
            routing_input_mode=routing_input_mode,
            # Match the canonical trtllm-gen wrappers' profiling regime so
            # choose_one() tunes under the same conditions as deployment
            # (otherwise it can cache a tactic picked under a different regime).
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )
        return moe_inputs.to_list()


# ---------------------------------------------------------------------------
# TRTLLM block-FP8 runner — DeepSeek FP8 and MXFP8
# ---------------------------------------------------------------------------


class TrtllmFp8BlockRunner(_TrtllmRunnerBase):
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
    supports_fused_shared_experts = True
    supported_activation_classes_by_quant: ClassVar[
        dict[QuantVariant, tuple[type[ActivationConfig], ...]]
    ] = {
        QuantVariant.DeepSeekFp8: (SwiGLU,),
        QuantVariant.MxFp8: (SwiGLU, GeGLU, ReLU2),
    }

    def _check_support(self) -> None:
        super()._check_support()
        from ..utils import get_compute_capability
        from .api import TrtllmFp8BlockConfig

        major, minor = get_compute_capability(self.device)
        arch = major * 10 + minor
        if not TrtllmFp8BlockConfig.supported(arch):
            raise NotImplementedError(
                f"{type(self).__name__} is enabled only on validated "
                f"SM100/SM103 targets, got sm{arch}."
            )

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        from ..tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
        from ..utils import device_support_pdl
        from .api import QuantVariant

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
        self._module: Any = None
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
        # Weights have local + S rows; the kernel still takes the routed count.
        self._num_fused_shared_experts = experts.num_fused_shared_experts
        self._num_weight_rows = self._num_local_experts + self._num_fused_shared_experts

        enable_pdl = execution.enable_pdl
        if enable_pdl is None:
            enable_pdl = device_support_pdl(device)
        self._enable_pdl = enable_pdl

        self._inner: Any = None
        self._static_kwargs: dict = {}
        self.tuning_config: Any = None

    def _ensure_inner(self, hidden_size: int) -> None:
        self._require_built()
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
            num_fused_shared_experts=self._num_fused_shared_experts,
        )

    def get_valid_tactics(  # type: ignore[override]
        self, inputs: List[torch.Tensor], profile: Any
    ) -> List[Any]:
        self._require_built()
        return self._inner.get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | List[torch.Tensor]:
        self._require_built()
        return self._forward_inner(inputs, tactic, do_preparation)

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
                self._num_weight_rows,
                self._intermediate_size
                * (2 if self.config.activation.is_gated else 1)
                // 128,
                hidden_size // 128,
            )
            expected_w2_scale = (
                self._num_weight_rows,
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
                self._num_weight_rows,
                self._intermediate_size * (2 if self.config.activation.is_gated else 1),
                hidden_size // 32,
            )
            expected_w2_scale = (
                self._num_weight_rows,
                hidden_size,
                self._intermediate_size // 32,
            )
            scale_dtype = torch.uint8

        expected_weights = {
            "gemm1_weights": (
                self._num_weight_rows,
                self._intermediate_size * (2 if self.config.activation.is_gated else 1),
                hidden_size,
            ),
            "gemm2_weights": (
                self._num_weight_rows,
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
        _validate_optional_gemm1_activation_params(
            view,
            self._num_weight_rows,
            act.hidden_states_q.device,
            "TrtllmFp8BlockRunner",
        )
        return scale

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        from ..tllm_enums import WeightLayout
        from .core import MoeRunnerInputs, RoutingInputMode

        view = weights.get_view(self.backend_key)
        _validate_prepared_activation_params(
            view, self.config.activation, type(self).__name__
        )
        routing = self.config.routing
        num_tokens, hidden_size = act.hidden_states_q.shape
        hidden_states_scale = self._validate_fp8_tensors(act, view, hidden_size)

        output = act.hidden_states_q.new_empty(
            (
                num_tokens,
                hidden_size if self.config.finalize.do_finalize else 0,
            ),
            dtype=torch.bfloat16,
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
            # The flat pre-routed API has no shared-expert argument. Callers can
            # append shared slots themselves and declare the fused totals.
            if self._num_fused_shared_experts > 0:
                raise NotImplementedError(
                    "TrtllmFp8BlockRunner requires FromLogits routing when "
                    "num_fused_shared_experts > 0. A pre-routed caller can fuse "
                    "the shared experts itself by appending the slots to "
                    "topk_ids/topk_weights and declaring num_experts + "
                    f"{self._num_fused_shared_experts} experts with top_k + "
                    f"{self._num_fused_shared_experts}."
                )
            _validate_prerouted_inputs(
                act, num_tokens, routing.top_k, "TrtllmFp8BlockRunner"
            )
            routing_logits = None
            routing_bias = None
            topk_ids = _pack_prerouted_topk_ids(act)
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
            # Optional SwiGLU OA controls; absent keys mean alpha=1 / beta=0 / no clamp.
            # Both block-scale variants consume them: MxFp8 in the fused FC1 epilogue,
            # DeepSeekFp8 in its separate activation kernel.
            gemm1_alpha=view.get("gemm1_alpha"),
            gemm1_beta=view.get("gemm1_beta"),
            gemm1_clamp_limit=view.get("gemm1_clamp_limit"),
            gemm2_weights=view["gemm2_weights"],
            gemm2_weights_scale=view["gemm2_weights_scale"],
            num_experts=routing.num_experts,
            num_fused_shared_experts=self._num_fused_shared_experts,
            n_group=routing.n_group,
            topk_group=routing.topk_group,
            local_expert_offset=self._local_expert_offset,
            routed_scaling_factor=routing.routed_scaling_factor,
            routing_method_type=int(routing.method),
            use_shuffled_weight=self._use_shuffled_weight,
            weight_layout=int(WeightLayout.MajorK),
            do_finalize=self.config.finalize.do_finalize,
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


# ---------------------------------------------------------------------------
# TRTLLM per-tensor FP8 runner — E4M3 activations/weights
# ---------------------------------------------------------------------------


class TrtllmFp8PerTensorRunner(_TrtllmRunnerBase):
    """Per-tensor-FP8 adapter over the canonical trtllm-gen ``MoERunner``.

    The kernel consumes prequantized E4M3 activations and weights. Its calibrated
    activation/weight multipliers are folded into three per-expert FP32 epilogue
    scale vectors, so ``MoEActivationPack.hidden_states_scale`` remains ``None``.
    Routing can be computed from logits or supplied as packed or unpacked
    precomputed expert IDs and weights.
    """

    backend_key = "trtllm_fp8_per_tensor"
    supported_routing_modes = (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
        RoutingInputMode.FromLogits,
    )
    supported_quant_variants = (QuantVariant.FP8PerTensor,)
    # The per-tensor cubin manifest has SwiGLU and ReLU2 epilogues. GeGLU is
    # representable by the enum but has no matching generated kernel.
    supported_activation_classes = (SwiGLU, ReLU2)

    def _check_activation_parameters(self) -> None:
        if (
            isinstance(self.config.activation, SwiGLU)
            and self.config.activation != SwiGLU()
        ):
            raise NotImplementedError(
                f"{type(self).__name__} cannot represent non-default SwiGLU scalars."
            )

    def _check_support(self) -> None:
        super()._check_support()
        from ..tllm_enums import RoutingMethodType
        from ..utils import get_compute_capability
        from .api import TrtllmFp8PerTensorConfig

        if (
            self.config.routing.method is RoutingMethodType.Llama4
            and self.config.routing.top_k != 1
        ):
            raise ValueError(
                f"{type(self).__name__} requires top_k=1 for Llama4 routing."
            )
        major, minor = get_compute_capability(self.device)
        arch = major * 10 + minor
        if not TrtllmFp8PerTensorConfig.supported(arch):
            raise NotImplementedError(
                f"{type(self).__name__} is enabled only on validated "
                f"SM100/SM103 targets, got sm{arch}."
            )

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        from ..tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
        from ..utils import device_support_pdl

        self.config = config
        self.device = device
        self._module: Any = None
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
        self._require_built()
        if self._inner is not None:
            return
        from ..tllm_enums import RoutingMethodType, WeightLayout

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
            use_per_token_scaling=(
                self.config.routing.method is RoutingMethodType.Llama4
            ),
            num_experts=self.config.routing.num_experts,
        )

    def get_valid_tactics(  # type: ignore[override]
        self, inputs: List[torch.Tensor], profile: Any
    ) -> List[Any]:
        self._require_built()
        return self._inner.get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | List[torch.Tensor]:
        self._require_built()
        return self._forward_inner(inputs, tactic, do_preparation)

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
                    self._intermediate_size
                    * (2 if self.config.activation.is_gated else 1),
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
        self._require_built()
        from ..tllm_enums import RoutingMethodType
        from .core import MoeRunnerInputs

        view = weights.get_view(self.backend_key)
        routing = self.config.routing
        num_tokens, hidden_size = act.hidden_states_q.shape
        self._validate_tensors(act, view, hidden_size)

        output = act.hidden_states_q.new_empty(
            (
                num_tokens,
                hidden_size if self.config.finalize.do_finalize else 0,
            ),
            dtype=torch.bfloat16,
        )
        routing_input_mode = act.routing_input_mode
        if routing_input_mode == RoutingInputMode.FromLogits:
            _validate_logits_inputs(
                act, num_tokens, routing.num_experts, type(self).__name__
            )
            routing_logits = act.routing_logits
            routing_bias = act.routing_bias
            # The routing kernel writes BF16 expert weights regardless of
            # whether routing_logits is BF16 or FP32.
            topk_ids = act.hidden_states_q.new_empty(
                (num_tokens, routing.top_k), dtype=torch.int32
            )
            expert_weights = act.hidden_states_q.new_empty(
                (num_tokens, routing.top_k), dtype=torch.bfloat16
            )
        elif routing_input_mode == RoutingInputMode.PackedPrecomputed:
            _validate_prerouted_inputs(
                act, num_tokens, routing.top_k, type(self).__name__
            )
            routing_logits = None
            routing_bias = None
            topk_ids = _pack_prerouted_topk_ids(act)
            expert_weights = act.topk_weights.new_empty(
                0, dtype=torch.bfloat16, device=act.topk_weights.device
            )
        elif routing_input_mode == RoutingInputMode.UnpackedPrecomputed:
            _validate_prerouted_inputs(
                act,
                num_tokens,
                routing.top_k,
                type(self).__name__,
                allowed_weights_dtypes=(torch.bfloat16, torch.float32),
                require_contiguous=True,
            )
            routing_logits = None
            routing_bias = None
            topk_ids = act.topk_ids
            expert_weights = act.topk_weights
        else:
            raise NotImplementedError(
                f"{type(self).__name__} supports only FromLogits, "
                "PackedPrecomputed, and UnpackedPrecomputed routing."
            )

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
            routing_input_mode=routing_input_mode,
            routing_bias=routing_bias,
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
            do_finalize=self.config.finalize.do_finalize,
            enable_pdl=self._enable_pdl,
            norm_topk_prob=True,
            routing_replay_out=None,
        )

        self._ensure_inner(hidden_size)
        self.tuning_config = self._inner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=self._tune_max_num_tokens,
            routing_input_mode=routing_input_mode,
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )
        return moe_inputs.to_list()


# ---------------------------------------------------------------------------
# TRTLLM BF16 runner — canonical trtllm-gen MoERunner, bf16 dtypes
# ---------------------------------------------------------------------------


class TrtllmBf16RoutedRunner(_TrtllmRunnerBase):
    """BF16 adapter over the canonical trtllm-gen ``MoERunner``.

    Mirrors :class:`TrtllmFp4RoutedRunner` but with ``Bfloat16`` activation +
    weight dtypes and no scale-factor tensors, wrapping the same inner
    ``MoERunner`` (whose ``forward`` dispatches to ``moe_op.trtllm_bf16_moe`` when
    ``dtype_weights == Bfloat16``).  Used for the EP grouped-GEMM bf16 path: the
    packed pre-routed ids carry ``(GLOBAL expert_id << 16) | bf16(weight)`` (with
    ``local_expert_offset`` passed separately), while ``FromLogits`` lets the
    kernel compute routing from BF16 or FP32 logits;
    with the EP bridge's synthesized ``top_k=1`` + ``weight=1`` and
    ``do_finalize=True``, the output comes back in input row order.

    The bf16 MoE entry point requires the ``BlockMajorK`` weight layout.
    """

    backend_key = "trtllm_bf16_routed"
    supported_routing_modes: tuple[RoutingInputMode, ...] = (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.FromLogits,
    )
    supported_quant_variants = (QuantVariant.BF16,)
    # The BF16 cubin manifest currently contains SwiGLU and ReLU2. GeGLU and
    # SiTU are represented by the launcher enum but have no matching kernels.
    supported_activation_classes = (SwiGLU, ReLU2)

    def _check_support(self) -> None:
        super()._check_support()
        from ..utils import get_compute_capability

        major, minor = get_compute_capability(self.device)
        arch = major * 10 + minor
        if arch not in (100, 103, 107):
            raise NotImplementedError(
                f"{type(self).__name__} is enabled only on routed-MoE cubin "
                f"targets SM100/SM103/SM107, got sm{arch}."
            )

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        from ..tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
        from ..utils import device_support_pdl

        self.config = config
        self.device = device
        self._module: Any = None

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
        self._require_built()
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
        self._require_built()
        return self._inner.get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | List[torch.Tensor]:
        self._require_built()
        return self._forward_inner(inputs, tactic, do_preparation)

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        """Translate Packs → the ``MoeRunnerInputs`` list for the bf16 path.

        Expected weight view keys: gemm1_weights, gemm2_weights (BlockMajorK,
        shuffled).  ``act.hidden_states_q`` carries the raw bf16 activations
        (the EP bridge does not quantize on the bf16 path);
        ``act.hidden_states_scale`` is unused.
        """
        self._require_built()
        from .core import MoeRunnerInputs, RoutingInputMode

        v = weights.get_view(self.backend_key)
        _validate_prepared_activation_params(
            v, self.config.activation, type(self).__name__
        )
        _validate_optional_gemm1_activation_params(
            v,
            self._num_local_experts,
            act.hidden_states_q.device,
            type(self).__name__,
        )
        routing = self.config.routing

        hidden_states = act.hidden_states_q  # raw bf16 on this path
        num_tokens, hidden_size = hidden_states.shape

        routing_input_mode = act.routing_input_mode
        if routing_input_mode == RoutingInputMode.FromLogits:
            _validate_logits_inputs(
                act, num_tokens, routing.num_experts, type(self).__name__
            )
            routing_logits = act.routing_logits
            routing_bias = act.routing_bias
            topk_ids = hidden_states.new_empty(
                (num_tokens, routing.top_k), dtype=torch.int32
            )
            # The BF16 routing kernel writes BF16 expert weights even for FP32
            # routing logits.
            expert_weights = hidden_states.new_empty(
                (num_tokens, routing.top_k), dtype=torch.bfloat16
            )
        elif routing_input_mode == RoutingInputMode.PackedPrecomputed:
            _validate_prerouted_inputs(
                act, num_tokens, routing.top_k, type(self).__name__
            )
            routing_logits = None
            routing_bias = None
            # Keep global IDs intact. The kernel applies local_expert_offset;
            # pre-subtracting it here would apply the offset twice and silently
            # drop experts on nonzero-offset ranks.
            topk_ids = _pack_prerouted_topk_ids(act)
            expert_weights = act.topk_weights.new_empty(
                (num_tokens, routing.top_k), dtype=torch.bfloat16
            )
        else:
            raise NotImplementedError(
                f"{type(self).__name__} supports only FromLogits and "
                "PackedPrecomputed routing."
            )

        output = hidden_states.new_empty(
            (
                num_tokens,
                hidden_size if self.config.finalize.do_finalize else 0,
            )
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

        from ..tllm_enums import WeightLayout

        self._static_kwargs = dict(
            routing_input_mode=routing_input_mode,
            routing_bias=routing_bias,
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
            do_finalize=self.config.finalize.do_finalize,
            enable_pdl=self._enable_pdl,
            # Matches the canonical BF16 FromLogits wrapper. Precomputed
            # routing ignores this flag because weights are already final.
            norm_topk_prob=True,
        )

        self._ensure_inner(hidden_size)
        self.tuning_config = self._inner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=self._tune_max_num_tokens,
            routing_input_mode=routing_input_mode,
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )
        return moe_inputs.to_list()


# ---------------------------------------------------------------------------
# TRTLLM MxInt4 runner — BF16 activations, packed INT4 BlockMajorK weights
# ---------------------------------------------------------------------------


class TrtllmMxInt4RoutedRunner(_TrtllmRunnerBase):
    """MxInt4 adapter over the canonical TRTLLM MoE runner."""

    backend_key = "trtllm_mxint4_routed"
    supported_routing_modes = (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.FromLogits,
    )
    supported_quant_variants = (QuantVariant.MxInt4,)
    supported_activation_classes = (SwiGLU,)

    def _check_support(self) -> None:
        super()._check_support()
        from ..utils import get_compute_capability
        from .api import TrtllmMxInt4Config

        major, minor = get_compute_capability(self.device)
        arch = major * 10 + minor
        if not TrtllmMxInt4Config.supported(arch):
            raise NotImplementedError(
                f"{type(self).__name__} is enabled only on supported "
                f"SM100/SM103/SM107 targets, got sm{arch}."
            )

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        from ..tllm_enums import DtypeTrtllmGen, Fp8QuantizationType
        from ..utils import device_support_pdl

        self.config = config
        self.device = device
        self._module: Any = None

        routing = config.routing
        experts = config.experts
        execution = config.execution
        self._num_local_experts = experts.local_num_experts or routing.num_experts
        self._local_expert_offset = experts.local_expert_offset
        self._intermediate_size = experts.intermediate_size
        self._activation_type = int(config.activation.type)
        self._tune_max_num_tokens = execution.tune_max_num_tokens

        self._dtype_act = DtypeTrtllmGen.Bfloat16
        self._dtype_weights = DtypeTrtllmGen.MxInt4
        self._fp8_quantization_type = Fp8QuantizationType.NoneFp8

        enable_pdl = execution.enable_pdl
        if enable_pdl is None:
            enable_pdl = device_support_pdl(device)
        self._enable_pdl = enable_pdl

        self._inner: Any = None
        self._static_kwargs: dict = {}
        self.tuning_config: Any = None

    def _ensure_inner(self, hidden_size: int) -> None:
        self._require_built()
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
        self._require_built()
        return self._inner.get_valid_tactics(inputs, profile)

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | List[torch.Tensor]:
        self._require_built()
        return self._forward_inner(inputs, tactic, do_preparation)

    def pack_inputs(
        self, act: MoEActivationPack, weights: MoEWeightPack
    ) -> List[torch.Tensor]:
        self._require_built()
        from .core import MoeRunnerInputs

        view = weights.get_view(self.backend_key)
        _validate_prepared_activation_params(
            view, self.config.activation, type(self).__name__
        )
        routing = self.config.routing
        hidden_states = act.hidden_states_q
        if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
            raise ValueError(f"{type(self).__name__}: hidden_states_q must be 2D BF16.")
        if not hidden_states.is_contiguous():
            raise ValueError(
                f"{type(self).__name__}: hidden_states_q must be contiguous."
            )
        if act.hidden_states_scale is not None:
            raise ValueError(
                f"{type(self).__name__}: hidden_states_scale must be None."
            )
        num_tokens, hidden_size = hidden_states.shape
        if hidden_size % 256 != 0 or self._intermediate_size % 256 != 0:
            raise ValueError(
                f"{type(self).__name__}: hidden_size and intermediate_size "
                "must be divisible by 256."
            )
        routing_input_mode = act.routing_input_mode
        if routing_input_mode == RoutingInputMode.FromLogits:
            _validate_logits_inputs(
                act, num_tokens, routing.num_experts, type(self).__name__
            )
            routing_logits = act.routing_logits
            routing_bias = act.routing_bias
            # MxInt4 infers routing mode from these placeholders rather than
            # receiving RoutingInputMode explicitly. Non-empty tensors select
            # precomputed routing and would suppress routing_logits.
            topk_ids = hidden_states.new_empty((0,), dtype=torch.int32)
            expert_weights = hidden_states.new_empty((0,), dtype=torch.bfloat16)
        elif routing_input_mode == RoutingInputMode.PackedPrecomputed:
            _validate_prerouted_inputs(
                act, num_tokens, routing.top_k, type(self).__name__
            )
            routing_logits = None
            routing_bias = None
            topk_ids = _pack_prerouted_topk_ids(act)
            expert_weights = act.topk_weights.new_empty(
                (num_tokens, routing.top_k), dtype=torch.bfloat16
            )
        else:
            raise NotImplementedError(
                f"{type(self).__name__} supports only FromLogits and "
                "PackedPrecomputed routing."
            )

        required = (
            "gemm1_weights",
            "gemm1_weights_scale",
            "gemm2_weights",
            "gemm2_weights_scale",
        )
        missing = [key for key in required if key not in view]
        if missing:
            raise KeyError(f"{self.backend_key} weight view is missing {missing}.")
        for key in required:
            tensor = view[key]
            if tensor.device != hidden_states.device:
                raise ValueError(
                    f"{type(self).__name__}: {key} is on {tensor.device}, "
                    f"expected {hidden_states.device}."
                )
            if not tensor.is_contiguous():
                raise ValueError(f"{type(self).__name__}: {key} must be contiguous.")
        if (
            view["gemm1_weights"].dtype != torch.uint8
            or view["gemm2_weights"].dtype != torch.uint8
        ):
            raise TypeError("MxInt4 packed weights must be uint8.")
        if (
            view["gemm1_weights_scale"].dtype != torch.bfloat16
            or view["gemm2_weights_scale"].dtype != torch.bfloat16
        ):
            raise TypeError("MxInt4 weight scales must be bfloat16.")
        expected_shapes = {
            "gemm1_weights": (
                self._num_local_experts,
                hidden_size // 256,
                2 * self._intermediate_size,
                128,
            ),
            "gemm1_weights_scale": (
                self._num_local_experts,
                2 * self._intermediate_size * hidden_size // 32,
            ),
            "gemm2_weights": (
                self._num_local_experts,
                self._intermediate_size // 256,
                hidden_size,
                128,
            ),
            "gemm2_weights_scale": (
                self._num_local_experts,
                hidden_size * self._intermediate_size // 32,
            ),
        }
        for key, expected in expected_shapes.items():
            if tuple(view[key].shape) != expected:
                raise ValueError(
                    f"{type(self).__name__}: {key} shape "
                    f"{tuple(view[key].shape)} != expected {expected}."
                )
        _validate_optional_gemm1_activation_params(
            view,
            self._num_local_experts,
            hidden_states.device,
            type(self).__name__,
        )

        output = hidden_states.new_empty(
            (
                num_tokens,
                hidden_size if self.config.finalize.do_finalize else 0,
            )
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

        self._static_kwargs = dict(
            routing_bias=routing_bias,
            gemm1_weights=view["gemm1_weights"],
            gemm1_weights_scale=view["gemm1_weights_scale"],
            gemm1_alpha=view.get("gemm1_alpha"),
            gemm1_beta=view.get("gemm1_beta"),
            gemm1_clamp_limit=view.get("gemm1_clamp_limit"),
            gemm2_weights=view["gemm2_weights"],
            gemm2_weights_scale=view["gemm2_weights_scale"],
            num_experts=routing.num_experts,
            n_group=routing.n_group,
            topk_group=routing.topk_group,
            local_expert_offset=self._local_expert_offset,
            routed_scaling_factor=routing.routed_scaling_factor,
            routing_method_type=int(routing.method),
            do_finalize=self.config.finalize.do_finalize,
            enable_pdl=self._enable_pdl,
            norm_topk_prob=True,
        )

        self._ensure_inner(hidden_size)
        self.tuning_config = self._inner._make_tuning_config(
            moe_inputs,
            tune_max_num_tokens=self._tune_max_num_tokens,
            routing_input_mode=routing_input_mode,
            use_cuda_graph=True,
            use_cold_l2_cache=True,
        )
        return moe_inputs.to_list()


# ---------------------------------------------------------------------------
# SM12x b12x runners — fixed tactic, existing wrapper delegation
# ---------------------------------------------------------------------------


class _B12xRunner(MoERunner):
    """Shared unified adapter over ``B12xMoEWrapper``."""

    backend_key: ClassVar[str] = ""
    supports_expert_parallelism = False
    required_weight_keys: ClassVar[tuple[str, ...]] = ()

    # Kernel activation name, resolved in _check_support() rather than
    # __init__() so an unsupported activation is a filterable
    # NotImplementedError instead of a constructor failure.
    activation: str | None

    def _check_activation_parameters(self) -> None:
        if (
            isinstance(self.config.activation, SwiGLU)
            and self.config.activation != SwiGLU()
        ):
            raise NotImplementedError(
                f"{type(self).__name__} cannot represent non-default SwiGLU scalars."
            )

    def _check_support(self) -> None:
        super()._check_support()

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

        if not self.config.finalize.do_finalize:
            raise NotImplementedError("b12x unified MoE requires do_finalize=True.")

        # super()._check_support() already rejected activations outside the
        # declared capability set; translate any residual gap in the name table
        # into the NotImplementedError that backend selection filters on.
        from .utils import get_b12x_activation_name

        try:
            self.activation = get_b12x_activation_name(self.config.activation.type)
        except ValueError as exc:
            raise NotImplementedError(str(exc)) from exc

    def __init__(self, config: MoEConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.activation = None
        self.tuning_config = TuningConfig()
        self._prepared_weights: dict[str, torch.Tensor] | None = None
        self._inner: Any = None
        self._wrapper_cls: Any = None

    def _build(self) -> None:
        """Load the b12x wrapper factory; shapes remain per-call."""
        from .cute_dsl import B12xMoEWrapper

        self._wrapper_cls = B12xMoEWrapper

    def get_valid_tactics(self, inputs: List[torch.Tensor], profile: Any) -> List[Any]:
        self._require_built()
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
        self._require_built()
        if (
            self._inner is not None
            and hidden_size == self._inner.hidden_size
            and num_tokens <= self._inner.max_num_tokens
        ):
            return
        self._inner = self._wrapper_cls(
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
        self._require_built()
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
        self._require_built()
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


class B12xNvfp4Runner(_B12xRunner):
    """Unified SM120/SM121 adapter for b12x NVFP4/W4A4 MoE."""

    backend_key = "b12x_nvfp4"
    supported_routing_modes = (RoutingInputMode.PackedPrecomputed,)
    supported_quant_variants = (QuantVariant.NVFP4,)
    supported_activation_classes = (SwiGLU, GeGLUTanh, ReLU2)
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
    supported_activation_classes = (SwiGLU, ReLU2)
    required_weight_keys = (
        "w1_weight",
        "w1_weight_sf",
        "w1_alpha",
        "w2_weight",
        "w2_weight_sf",
        "w2_alpha",
    )


def __getattr__(name: str):
    if name == "CuteDslNvfp4Runner":
        warnings.warn(
            "CuteDslNvfp4Runner is deprecated; use CuteDslRunner instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return CuteDslRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
