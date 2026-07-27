"""Unified MoE API — configuration dataclasses and tensor groupings.

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

Config objects are frozen (immutable). Use ``dataclasses.replace`` to derive
variants.  ``eval(repr(cfg))`` round-trips for every config type, enabling
repro-log serialization.

Tensor groupings are mutable containers — they hold runtime data, not
configuration.  They group related tensors for ergonomics (no more counting
20+ positional arguments).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from ..tllm_enums import ActivationType, RoutingInputMode, RoutingMethodType

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
# Routing and activation reuse the shared kernel-level enums directly
# (``RoutingMethodType`` / ``ActivationType`` from ``tllm_enums``): the API
# speaks the kernels' vocabulary rather than mirroring it, so there is a single
# source of truth (PR #3093 review G1).  Both are ``IntEnum`` — the value *is*
# the kernel ABI int — and carry an eval-safe ``__repr__`` (defined in
# ``tllm_enums``) plus ``ActivationType.is_gated`` for the repro round-trip and
# config helpers.
#
# ``QuantVariant`` below is the one genuinely API-level enum: it has no single
# kernel counterpart (the quant path is selected by dtype/scale wiring in the
# runners, not one enum), so it is defined here as a plain ``Enum``.


class QuantVariant(Enum):
    """Quantization variant — single knob for dtype + granularity + scale convention."""

    BF16 = 0
    FP8PerTensor = 1
    DeepSeekFp8 = 2
    MxFp8 = 3
    NVFP4 = 4  # day-1 MVP target
    MXFP4 = 5
    MxInt4 = 6
    W4A16 = 7

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"


# ---------------------------------------------------------------------------
# Component configs — each owns one concern
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingConfig:
    """Expert routing parameters.

    Parameters
    ----------
    num_experts : int
        Total number of experts (global, before EP sharding).
    top_k : int
        Number of experts selected per token.
    method : RoutingMethodType
        Routing strategy.
    n_group : int or None
        Expert group count for DeepSeekV3 routing.
    topk_group : int or None
        Number of groups selected in DeepSeekV3.
    routed_scaling_factor : float or None
        Fixed routing weight scaling (DeepSeekV3).
    """

    num_experts: int
    top_k: int
    method: RoutingMethodType = RoutingMethodType.Default
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    routed_scaling_factor: Optional[float] = None

    def __repr__(self) -> str:
        parts = [f"num_experts={self.num_experts!r}", f"top_k={self.top_k!r}"]
        if self.method != RoutingMethodType.Default:
            parts.append(f"method={self.method!r}")
        if self.n_group is not None:
            parts.append(f"n_group={self.n_group!r}")
        if self.topk_group is not None:
            parts.append(f"topk_group={self.topk_group!r}")
        if self.routed_scaling_factor is not None:
            parts.append(f"routed_scaling_factor={self.routed_scaling_factor!r}")
        return f"RoutingConfig({', '.join(parts)})"


@dataclass(frozen=True)
class QuantConfig:
    """Quantization scheme.

    Parameters
    ----------
    variant : QuantVariant
        Single knob for dtype + granularity + scale convention.
    swizzled_scale_factors : bool or None
        Whether block scale factors use the swizzled (vs linear) layout.
        ``None`` → backend default.  Mirrors core's ``swizzled_input_sf``.  Finer
        ``SfLayout`` (128x4 / 8x4 / linear) selection is deferred (design doc
        C42): unlike ``RoutingMethodType`` / ``ActivationType``, ``SfLayout`` has
        no eval-safe ``__repr__``, so exposing it here would break the
        ``eval(repr(cfg))`` round-trip — a bool keeps this config serializable.
    per_token_scale : bool or None
        Whether activations carry a per-token scale (vs per-tensor / block).
        ``None`` → backend default.
    """

    variant: QuantVariant = QuantVariant.BF16
    swizzled_scale_factors: Optional[bool] = None
    per_token_scale: Optional[bool] = None


@dataclass(frozen=True)
class ActivationConfig:
    """Fused activation between GEMM1 and GEMM2."""

    # Convenience singletons — populated after class definition
    swiglu: ClassVar[ActivationConfig]
    geglu: ClassVar[ActivationConfig]
    relu2: ClassVar[ActivationConfig]
    identity: ClassVar[ActivationConfig]

    type: ActivationType = ActivationType.Swiglu

    def __repr__(self) -> str:
        return f"ActivationConfig(type={self.type!r})"

    @property
    def is_gated(self) -> bool:
        return self.type.is_gated


ActivationConfig.swiglu = ActivationConfig(ActivationType.Swiglu)
ActivationConfig.geglu = ActivationConfig(ActivationType.Geglu)
ActivationConfig.relu2 = ActivationConfig(ActivationType.Relu2)
ActivationConfig.identity = ActivationConfig(ActivationType.Identity)


@dataclass(frozen=True)
class ExpertConfig:
    """Expert geometry.

    Parameters
    ----------
    intermediate_size : int
        Hidden dimension of the expert FFN (the N in gemm1's MxK → MxN).
    local_expert_offset : int
        Start index for expert-parallel sharding.
    local_num_experts : int or None
        Number of experts on this rank.  ``None`` → ``num_experts`` at runtime.
    """

    intermediate_size: int
    local_expert_offset: int = 0
    local_num_experts: Optional[int] = None

    def __repr__(self) -> str:
        parts = [f"intermediate_size={self.intermediate_size!r}"]
        if self.local_expert_offset != 0:
            parts.append(f"local_expert_offset={self.local_expert_offset!r}")
        if self.local_num_experts is not None:
            parts.append(f"local_num_experts={self.local_num_experts!r}")
        return f"ExpertConfig({', '.join(parts)})"


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime execution parameters.

    Parameters
    ----------
    do_finalize : bool
        Whether to apply routing-weight scaling and accumulate into output.
    enable_pdl : bool or None
        Persistent device launch.  ``None`` → auto (True for sm90+).
    tune_max_num_tokens : int
        Token budget hint for autotuner / CUDA graph capture.
    """

    do_finalize: bool = True
    enable_pdl: Optional[bool] = None
    tune_max_num_tokens: int = 8192

    def __repr__(self) -> str:
        parts = []
        if not self.do_finalize:
            parts.append(f"do_finalize={self.do_finalize!r}")
        if self.enable_pdl is not None:
            parts.append(f"enable_pdl={self.enable_pdl!r}")
        if self.tune_max_num_tokens != 8192:
            parts.append(f"tune_max_num_tokens={self.tune_max_num_tokens!r}")
        return f"ExecutionConfig({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Backend configs — each declares hardware preconditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrtllmFp4Config:
    """TensorRT-LLM FP4 block-scale backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        # SM100+ only: the routed runner delegates to the trtllm-gen sm100
        # module, which core.is_trtllm_moe_supported() gates on major >= 10.
        # Returning True on SM90 would mark the backend available on H100 and
        # then fail at dispatch.
        return arch >= 100

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device=None,
        permute_cache=None,
    ):
        """Build the ``trtllm_fp4_routed`` weight view from canonical bf16 weights.

        Register the result with ``MoEWeightPack.prepare_for("trtllm_fp4_routed", ...)``.
        See :func:`flashinfer.fused_moe.prepare.prepare_trtllm_fp4_weights`.
        """
        from .prepare import prepare_trtllm_fp4_weights

        return prepare_trtllm_fp4_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
            permute_cache=permute_cache,
        )

    def __repr__(self) -> str:
        return "TrtllmFp4Config()"


@dataclass(frozen=True)
class TrtllmFp8BlockConfig:
    """TensorRT-LLM FP8 block-scale backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        # The available TRTLLM block-FP8 BMM cubins are validated only on the
        # SM100 family. The outer JIT can compile for major 12, but its FP8
        # kernels currently fail at runtime on SM120/121.
        return arch in (100, 103)

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        variant: QuantVariant,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device=None,
    ):
        """Build the ``trtllm_fp8_block`` weight view from canonical BF16.

        ``variant`` must be :attr:`QuantVariant.DeepSeekFp8` or
        :attr:`QuantVariant.MxFp8`; their scale formats are intentionally
        prepared by separate paths. The shuffled MXFP8 view requires both
        ``hidden_size`` and ``intermediate_size`` to be divisible by 128 so its
        scale tensors fit TRTLLM's unpadded 128x4 physical layout.
        """
        from .prepare import prepare_trtllm_fp8_block_weights

        return prepare_trtllm_fp8_block_weights(
            w1_bf16,
            w2_bf16,
            variant=variant,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        )

    @staticmethod
    def prepare_activations(hidden_states_bf16, *, variant: QuantVariant):
        """Quantize BF16 activations for the selected block-FP8 convention."""
        from .prepare import prepare_trtllm_fp8_block_activations

        return prepare_trtllm_fp8_block_activations(hidden_states_bf16, variant=variant)

    def __repr__(self) -> str:
        return "TrtllmFp8BlockConfig()"


@dataclass(frozen=True)
class TrtllmFp8PerTensorConfig:
    """TensorRT-LLM FP8 per-tensor-scale backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in (100, 103)

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        hidden_states_scale_global,
        intermediate_scale_global,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device=None,
    ):
        """Build the ``trtllm_fp8_per_tensor`` MajorK weight view."""
        from .prepare import prepare_trtllm_fp8_per_tensor_weights

        return prepare_trtllm_fp8_per_tensor_weights(
            w1_bf16,
            w2_bf16,
            hidden_states_scale_global=hidden_states_scale_global,
            intermediate_scale_global=intermediate_scale_global,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        )

    @staticmethod
    def prepare_activations(hidden_states_bf16, *, hidden_states_scale_global):
        """Quantize BF16 activations with one calibrated E4M3 multiplier."""
        from .prepare import prepare_trtllm_fp8_per_tensor_activations

        return prepare_trtllm_fp8_per_tensor_activations(
            hidden_states_bf16,
            hidden_states_scale_global=hidden_states_scale_global,
        )

    def __repr__(self) -> str:
        return "TrtllmFp8PerTensorConfig()"


@dataclass(frozen=True)
class TrtllmBf16Config:
    """TensorRT-LLM BF16 (unquantized) backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch >= 100

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device=None,
        permute_cache=None,
    ):
        """Build the ``trtllm_bf16_routed`` weight view from canonical bf16 weights.

        Register the result with ``MoEWeightPack.prepare_for("trtllm_bf16_routed", ...)``.
        See :func:`flashinfer.fused_moe.prepare.prepare_trtllm_bf16_weights`.
        """
        from .prepare import prepare_trtllm_bf16_weights

        return prepare_trtllm_bf16_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
            permute_cache=permute_cache,
        )

    def __repr__(self) -> str:
        return "TrtllmBf16Config()"


@dataclass(frozen=True)
class TrtllmMxInt4Config:
    """TensorRT-LLM MxInt4 backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch >= 100

    def __repr__(self) -> str:
        return "TrtllmMxInt4Config()"


@dataclass(frozen=True)
class CutlassConfig:
    """CUTLASS backend — broadest architecture support."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return True  # universal fallback

    def __repr__(self) -> str:
        return "CutlassConfig()"


@dataclass(frozen=True)
class CuteDslConfig:
    """CuteDSL NVFP4 backend — SM100 family only (Blackwell SM100, SM103).

    The underlying CuteDSL kernel throws at launch on SM120/SM121/SM130.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        # SM100, SM103 — tighten when CuteDSL adds more targets
        return arch in (100, 103)

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        device=None,
    ):
        """Build the ``cute_dsl_nvfp4`` weight view from canonical bf16 weights.

        Register the result with ``MoEWeightPack.prepare_for("cute_dsl_nvfp4", ...)``.
        See :func:`flashinfer.fused_moe.prepare.prepare_cute_dsl_nvfp4_weights`.
        """
        from .prepare import prepare_cute_dsl_nvfp4_weights

        return prepare_cute_dsl_nvfp4_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device=device,
        )

    def __repr__(self) -> str:
        return "CuteDslConfig()"


@dataclass(frozen=True)
class B12xNvfp4Config:
    """SM120/SM121 CuTe-DSL b12x NVFP4/W4A4 backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in (120, 121)

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: ActivationConfig = ActivationConfig.swiglu,
        device=None,
    ):
        """Build the ``b12x_nvfp4`` weight view from canonical bf16 weights.

        Register the result with ``MoEWeightPack.prepare_for("b12x_nvfp4", ...)``.
        See :func:`flashinfer.fused_moe.prepare.prepare_b12x_nvfp4_weights`.
        """
        from .prepare import prepare_b12x_nvfp4_weights
        from .utils import get_b12x_activation_name

        return prepare_b12x_nvfp4_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=get_b12x_activation_name(activation.type),
            device=device,
        )

    def __repr__(self) -> str:
        return "B12xNvfp4Config()"


@dataclass(frozen=True)
class B12xW4A16Config:
    """SM120/SM121 CuTe-DSL b12x W4A16 backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in (120, 121)

    @staticmethod
    def prepare_weights(
        w1_fp4,
        w1_blockscale,
        w1_global_scale,
        w2_fp4,
        w2_blockscale,
        w2_global_scale,
        *,
        activation: ActivationConfig = ActivationConfig.swiglu,
        source_format: str = "modelopt",
    ):
        """Build the ``b12x_w4a16`` weight view from checkpoint fp4 weights.

        Register the result with ``MoEWeightPack.prepare_for("b12x_w4a16", ...)``.
        See :func:`flashinfer.fused_moe.prepare.prepare_b12x_w4a16_weights`.
        """
        from .prepare import prepare_b12x_w4a16_weights
        from .utils import get_b12x_activation_name

        return prepare_b12x_w4a16_weights(
            w1_fp4,
            w1_blockscale,
            w1_global_scale,
            w2_fp4,
            w2_blockscale,
            w2_global_scale,
            activation=get_b12x_activation_name(activation.type),
            source_format=source_format,
        )

    def __repr__(self) -> str:
        return "B12xW4A16Config()"


# Union type for backend config
BackendConfigType = Union[
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmBf16Config,
    TrtllmMxInt4Config,
    CutlassConfig,
    CuteDslConfig,
    B12xNvfp4Config,
    B12xW4A16Config,
]

ALL_BACKEND_CONFIGS = (
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmBf16Config,
    TrtllmMxInt4Config,
    CutlassConfig,
    CuteDslConfig,
    B12xNvfp4Config,
    B12xW4A16Config,
)


# ---------------------------------------------------------------------------
# BackendOptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendOptions:
    """Ordered list of backend candidates for dispatch / autotuning."""

    candidates: Tuple[BackendConfigType, ...] = ()  # type: ignore[type-arg]

    def valid_for(self, arch: int) -> list:
        """Return candidates whose hardware preconditions are met."""
        return [c for c in self.candidates if c.__class__.supported(arch)]

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)


# ---------------------------------------------------------------------------
# MoEConfig — top-level container
# ---------------------------------------------------------------------------

# Default backend search order
_DEFAULT_BACKEND = BackendOptions(
    candidates=(
        TrtllmFp4Config(),
        TrtllmFp8BlockConfig(),
        TrtllmFp8PerTensorConfig(),
        TrtllmBf16Config(),
        TrtllmMxInt4Config(),
        CutlassConfig(),
        CuteDslConfig(),
    )
)


@dataclass(frozen=True)
class MoEConfig:
    """Top-level MoE configuration.

    Combines all sub-configs into a single hashable, serializable object.
    Supports ``**config`` unpacking via the dict protocol.

    Example
    -------
    >>> config = MoEConfig(
    ...     routing=RoutingConfig(num_experts=64, top_k=8,
    ...                           method=RoutingMethodType.DeepSeekV3),
    ...     quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
    ...     experts=ExpertConfig(intermediate_size=2048),
    ... )
    >>> output = fused_moe(tensors, **config)
    """

    routing: RoutingConfig
    quant: QuantConfig
    experts: ExpertConfig
    activation: ActivationConfig = field(
        default_factory=lambda: ActivationConfig(ActivationType.Swiglu)
    )
    backend: BackendOptions = field(default_factory=lambda: _DEFAULT_BACKEND)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # --- Dict-unpacking protocol: enables ``**config`` at call sites ---

    def keys(self):
        return (f.name for f in dataclasses.fields(self))

    def __getitem__(self, key: str):
        return getattr(self, key)

    # --- Serialization ---
    #
    # ``repr(config)`` already round-trips to valid constructor syntax (frozen
    # dataclasses + qualified enum repr), which is all the MVP needs for logging.
    # A deserializer (``from_repr``/``from_dict``) is intentionally *not* shipped
    # here: eval-based parsing is a security smell, and the repro/serialization
    # design (versioned schema vs. same-version-only) is a documented post-MVP
    # item — see docs/design_docs/flashinfer_moe_api.md (C4-C5/C39, Post-MVP
    # Carryover). It will land with the repro tooling, not before.


# ---------------------------------------------------------------------------
# Activation / weight packs for autotuned pre-routed and FromLogits paths
# ---------------------------------------------------------------------------
# These are the runner-level inputs used by MoELayer (plan §1).
#
# Why two packs instead of one tensor bundle (PR #3093 review G5): the grouping
# axis is *lifetime/role*, which is invariant across backends —
#   * MoEActivationPack: per-call transient activation/routing data,
#     rebuilt every forward;
#   * MoEWeightPack:     long-lived weights, materialized once at load and read
#     every call, holding one native view *per backend* (the price of
#     cross-backend autotune) keyed by backend_key.
# A single per-call bundle cannot model a load-time, multi-backend weight cache
# without conflating the two lifetimes.  We deliberately do *not* group tensors
# by compute-graph stage (e.g. gemm1/gemm2): that mirrors the unfused two-GEMM
# implementation and would overfit it — a fused/megakernel backend has no such
# boundary, so a graph-shaped public API would leak one backend's internals.
# Each pack presents itself to a backend via prepare_for / get_view, keeping
# backend-specific layout logic out of the dispatch hot-path.


@dataclass
class MoEActivationPack:
    """Per-call backend-native activations plus routing inputs.

    Activation encoding depends on ``QuantConfig.variant``:

    * NVFP4: packed ``uint8 [M, H/2]`` values with
      ``float8_e4m3fn [M, H/16]`` block scales.
    * BF16: raw ``bfloat16 [M, H]`` values with no scale tensor.
    * DeepSeek FP8: ``float8_e4m3fn [M, H]`` values with transposed
      ``float32 [H/128, M]`` block scales.
    * MXFP8: ``float8_e4m3fn [M, H]`` values with token-major
      ``uint8 [M, H/32]`` UE8M0 scales.
    * FP8 per-tensor: ``float8_e4m3fn [M, H]`` values with no scale tensor;
      the calibrated scalar is folded into the backend's epilogue scales.

    ``routing_input_mode`` selects how routing reaches the kernel (the runner reads it directly):

    * ``PackedPrecomputed`` (default) — **pre-routed**: the caller computes expert
      selection on the host and passes ``topk_ids`` + ``topk_weights``.
      (``UnpackedPrecomputed`` exists at the kernel enum level but is not currently
      supported by the unified runners.)
    * ``FromLogits`` — **in-kernel**: the caller passes raw ``routing_logits`` (and, for bias-aware
      methods like DeepSeekV3/MiniMax2, ``routing_bias``); the kernel computes the top-k selection
      itself per ``RoutingConfig.method``.  ``topk_ids`` / ``topk_weights`` stay ``None`` — the
      runner allocates internal kernel-filled buffers, and the routing result is not surfaced
      back through the pack (routing replay is a separate, future capability). TRTLLM FP4,
      block-FP8, and per-tensor-FP8 runners support this mode; ``MoELayer`` dispatches a logits
      pack only to capable backends (see each runner's ``supported_routing_modes``).

    ``topk_ids`` / ``topk_weights`` follow the routed-MoE naming convention (gh #2425); they
    keep the field positions of the former ``selected_experts`` / ``final_scales``, so
    positional construction of pre-routed packs is unchanged.  The in-kernel routing fields
    are keyword-only.
    """

    # Backend-native activation payload; layouts documented above.
    hidden_states_q: Tensor
    # Variant-specific scales documented above; None for BF16/per-tensor FP8.
    hidden_states_scale: Optional[Tensor]
    # Pre-routed top-k selection (Packed/Unpacked modes); None under FromLogits.
    topk_ids: Optional[Tensor] = None  # [M, top_k] int32 (expert indices)
    topk_weights: Optional[Tensor] = None  # [M, top_k] float32 (routing weights)
    # In-kernel routing inputs (FromLogits) — keyword-only so a stale positional
    # call site fails loudly instead of silently binding a tensor to the mode.
    routing_input_mode: RoutingInputMode = field(
        default=RoutingInputMode.PackedPrecomputed, kw_only=True
    )
    routing_logits: Optional[Tensor] = field(
        default=None, kw_only=True
    )  # [M, num_experts] float32 or bfloat16
    routing_bias: Optional[Tensor] = field(
        default=None, kw_only=True
    )  # [num_experts] bfloat16 or float32 (independent of logits dtype)

    def __post_init__(self) -> None:
        """Fail fast on mode/field mismatches at construction time.

        Raises (not asserts) so the checks survive ``python -O``; catching the
        mismatch here names the offending field instead of a later failure deep
        in ``pack_inputs`` or a C++ ICHECK.
        """
        mode = self.routing_input_mode
        if mode == RoutingInputMode.FromLogits:
            if self.routing_logits is None:
                raise ValueError(
                    "routing_input_mode=FromLogits requires routing_logits."
                )
            if self.topk_ids is not None or self.topk_weights is not None:
                raise ValueError(
                    "FromLogits computes topk_ids/topk_weights in-kernel; "
                    "leave them None."
                )
        elif mode == RoutingInputMode.PackedPrecomputed:
            if self.topk_ids is None or self.topk_weights is None:
                raise ValueError(
                    "routing_input_mode=PackedPrecomputed requires "
                    "topk_ids + topk_weights."
                )
            if self.routing_logits is not None or self.routing_bias is not None:
                raise ValueError(
                    "routing_logits/routing_bias are only consumed by "
                    "in-kernel (FromLogits) routing."
                )
            if self.topk_ids.dtype != torch.int32:
                raise TypeError(
                    f"topk_ids must be torch.int32 (got {self.topk_ids.dtype}); "
                    "torch.topk returns int64 — cast before constructing the pack."
                )
        # UnpackedPrecomputed: no unified runner supports it; runners raise
        # NotImplementedError, so no field contract is enforced here.

        # All routing tensors must live with the activations; a stray CPU
        # tensor otherwise surfaces as a cryptic launch/ICHECK failure.
        dev = self.hidden_states_q.device
        for name in (
            "hidden_states_scale",
            "topk_ids",
            "topk_weights",
            "routing_logits",
            "routing_bias",
        ):
            t = getattr(self, name)
            if t is not None and t.device != dev:
                raise ValueError(
                    f"{name} is on {t.device} but hidden_states_q is on {dev}; "
                    "all pack tensors must be on the same device."
                )

    @property
    def num_tokens(self) -> int:
        return self.hidden_states_q.shape[0]


@dataclass
class MoEWeightPack:
    """Long-lived weight container with per-backend native materializations.

    Each backend's native weight layout (quantized, swizzled, MMA-ordered, etc.)
    is stored under its ``backend_key``.  Populated once at model-load /
    layer-init via ``prepare_for(key, view)``; read on every call via
    ``get_view(key)``.

    Holding multiple materializations is intentional — that's the memory cost
    the user pays for cross-backend autotune.  Each view is the exact kwargs
    dict that runner's ``forward`` expects for weight-side arguments.
    """

    native_views: Dict[str, Dict[str, Tensor]] = field(default_factory=dict)

    def prepare_for(self, backend_key: str, view: Dict[str, Tensor]) -> None:
        """Register a backend-native weight view.  Caller owns the quantization
        / swizzle / layout conversion — this method just stores the result."""
        self.native_views[backend_key] = view

    def get_view(self, backend_key: str) -> Dict[str, Tensor]:
        if backend_key not in self.native_views:
            raise KeyError(
                f"Weights not prepared for backend {backend_key!r}. "
                f"Available: {list(self.native_views)}"
            )
        return self.native_views[backend_key]
