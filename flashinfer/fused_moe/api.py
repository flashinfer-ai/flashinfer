"""Unified MoE API — configuration dataclasses and tensor groupings.

Copyright (c) 2025 by FlashInfer team.

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

from torch import Tensor

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
# These enums define the *unified API* vocabulary.  They intentionally mirror
# the existing ``tllm_enums`` values so that adapters can cast directly.


class RoutingMethod(Enum):
    """Routing strategy applied to expert logits."""

    Default = 0  # Softmax → TopK
    Renormalize = 1  # TopK → Softmax
    DeepSeekV3 = 2  # Sigmoid → Bias → TopK-group → TopK-expert
    Llama4 = 3  # Top1 → Sigmoid
    RenormalizeNaive = 4  # Softmax → TopK → Renormalize (Qwen3)
    TopK = 5  # TopK only (no softmax)

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"


class Activation(Enum):
    """Fused activation between GEMM1 and GEMM2."""

    Gelu = 0
    Relu = 1
    Silu = 2
    Swiglu = 3
    Geglu = 4
    SwigluBias = 5
    Relu2 = 6
    Identity = 7

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

    @property
    def is_gated(self) -> bool:
        return self in (Activation.Swiglu, Activation.Geglu, Activation.SwigluBias)


class QuantVariant(Enum):
    """Quantization variant — single knob for dtype + granularity + scale convention."""

    BF16 = 0
    FP8PerTensor = 1
    DeepSeekFp8 = 2
    MxFp8 = 3
    NVFP4 = 4  # day-1 MVP target
    MXFP4 = 5
    MxInt4 = 6

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
    method : RoutingMethod
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
    method: RoutingMethod = RoutingMethod.Default
    n_group: Optional[int] = None
    topk_group: Optional[int] = None
    routed_scaling_factor: Optional[float] = None

    def __repr__(self) -> str:
        parts = [f"num_experts={self.num_experts!r}", f"top_k={self.top_k!r}"]
        if self.method != RoutingMethod.Default:
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
    """Quantization scheme."""

    variant: QuantVariant = QuantVariant.BF16


@dataclass(frozen=True)
class ActivationConfig:
    """Fused activation between GEMM1 and GEMM2."""

    # Convenience singletons — populated after class definition
    swiglu: ClassVar[ActivationConfig]
    geglu: ClassVar[ActivationConfig]
    relu2: ClassVar[ActivationConfig]
    identity: ClassVar[ActivationConfig]

    type: Activation = Activation.Swiglu

    def __repr__(self) -> str:
        return f"ActivationConfig(type={self.type!r})"

    @property
    def is_gated(self) -> bool:
        return self.type.is_gated


ActivationConfig.swiglu = ActivationConfig(Activation.Swiglu)
ActivationConfig.geglu = ActivationConfig(Activation.Geglu)
ActivationConfig.relu2 = ActivationConfig(Activation.Relu2)
ActivationConfig.identity = ActivationConfig(Activation.Identity)


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
        return arch >= 90

    def __repr__(self) -> str:
        return "TrtllmFp4Config()"


@dataclass(frozen=True)
class TrtllmFp8BlockConfig:
    """TensorRT-LLM FP8 block-scale backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch >= 80

    def __repr__(self) -> str:
        return "TrtllmFp8BlockConfig()"


@dataclass(frozen=True)
class TrtllmFp8PerTensorConfig:
    """TensorRT-LLM FP8 per-tensor-scale backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch >= 80

    def __repr__(self) -> str:
        return "TrtllmFp8PerTensorConfig()"


@dataclass(frozen=True)
class TrtllmBf16Config:
    """TensorRT-LLM BF16 (unquantized) backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch >= 100

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

    def __repr__(self) -> str:
        return "CuteDslConfig()"


# Union type for backend config
BackendConfigType = Union[
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmBf16Config,
    TrtllmMxInt4Config,
    CutlassConfig,
    CuteDslConfig,
]

ALL_BACKEND_CONFIGS = (
    TrtllmFp4Config,
    TrtllmFp8BlockConfig,
    TrtllmFp8PerTensorConfig,
    TrtllmBf16Config,
    TrtllmMxInt4Config,
    CutlassConfig,
    CuteDslConfig,
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
    ...                           method=RoutingMethod.DeepSeekV3),
    ...     quant=QuantConfig(variant=QuantVariant.DeepSeekFp8),
    ...     experts=ExpertConfig(intermediate_size=2048),
    ... )
    >>> output = fused_moe(tensors, **config)
    """

    routing: RoutingConfig
    quant: QuantConfig
    experts: ExpertConfig
    activation: ActivationConfig = field(
        default_factory=lambda: ActivationConfig(Activation.Swiglu)
    )
    backend: BackendOptions = field(default_factory=lambda: _DEFAULT_BACKEND)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    # --- Dict-unpacking protocol: enables ``**config`` at call sites ---

    def keys(self):
        return (f.name for f in dataclasses.fields(self))

    def __getitem__(self, key: str):
        return getattr(self, key)

    # --- Serialization ---

    @classmethod
    def from_repr(cls, s: str) -> MoEConfig:
        """Reconstruct from ``repr()`` output (safe eval)."""
        ns = {
            "MoEConfig": MoEConfig,
            "RoutingConfig": RoutingConfig,
            "QuantConfig": QuantConfig,
            "ActivationConfig": ActivationConfig,
            "ExpertConfig": ExpertConfig,
            "ExecutionConfig": ExecutionConfig,
            "BackendOptions": BackendOptions,
            "TrtllmFp4Config": TrtllmFp4Config,
            "TrtllmFp8BlockConfig": TrtllmFp8BlockConfig,
            "TrtllmFp8PerTensorConfig": TrtllmFp8PerTensorConfig,
            "TrtllmBf16Config": TrtllmBf16Config,
            "TrtllmMxInt4Config": TrtllmMxInt4Config,
            "CutlassConfig": CutlassConfig,
            "CuteDslConfig": CuteDslConfig,
            "RoutingMethod": RoutingMethod,
            "Activation": Activation,
            "QuantVariant": QuantVariant,
            "__builtins__": {},
        }
        return eval(s, ns)


# ---------------------------------------------------------------------------
# Tensor groupings
# ---------------------------------------------------------------------------


@dataclass
class Gemm1Tensors:
    """Tensors for the first expert GEMM (FC1 / gate-up projection).

    Parameters
    ----------
    weights : Tensor
        Expert weights ``[E, N, K]`` or backend-specific layout.
    weights_scale : Tensor or None
        Scale factors for quantized weights.
    bias : Tensor or None
        Per-expert bias.
    alpha : Tensor or None
        Swiglu alpha scaling factor.
    beta : Tensor or None
        Swiglu beta scaling factor.
    clamp_limit : Tensor or None
        Swiglu clamp limit.
    output_scale : Tensor or None
        FP8 per-tensor output scale (gemm1 output).
    output_scale_gate : Tensor or None
        FP8 per-tensor output scale for gated path.
    """

    weights: Tensor = None
    weights_scale: Optional[Tensor] = None
    bias: Optional[Tensor] = None
    alpha: Optional[Tensor] = None
    beta: Optional[Tensor] = None
    clamp_limit: Optional[Tensor] = None
    output_scale: Optional[Tensor] = None
    output_scale_gate: Optional[Tensor] = None


@dataclass
class Gemm2Tensors:
    """Tensors for the second expert GEMM (FC2 / down projection).

    Parameters
    ----------
    weights : Tensor
        Expert weights ``[E, K, N]`` or backend-specific layout.
    weights_scale : Tensor or None
        Scale factors for quantized weights.
    bias : Tensor or None
        Per-expert bias.
    output_scale : Tensor or None
        FP8 per-tensor output scale.
    """

    weights: Tensor = None
    weights_scale: Optional[Tensor] = None
    bias: Optional[Tensor] = None
    output_scale: Optional[Tensor] = None


@dataclass
class MoETensors:
    """All tensors for a single MoE forward pass.

    Supports two invocation modes:

    **Monolithic** (routing fused into kernel): populate ``routing_logits``
    and optionally ``routing_bias``.  The backend handles routing internally.

    **Modular** (pre-routed): populate ``token_selected_experts`` and
    ``token_final_scales``.  Routing was done externally (e.g. by the
    serving framework).

    Parameters
    ----------
    hidden_states : Tensor
        Input activations ``[M, K]``.
    gemm1 : Gemm1Tensors
        First GEMM tensors.
    gemm2 : Gemm2Tensors
        Second GEMM tensors.
    routing_logits : Tensor or None
        Raw router logits ``[M, E]``.  Monolithic path.
    routing_bias : Tensor or None
        Router bias ``[E]``.  DeepSeekV3.
    hidden_states_scale : Tensor or None
        Per-token or block scale for quantized hidden states.
    token_selected_experts : Tensor or None
        Pre-routed expert indices ``[M, top_k]``.  Modular path.
    token_final_scales : Tensor or None
        Pre-computed routing weights ``[M, top_k]``.  Modular path.
    output : Tensor or None
        Pre-allocated output buffer ``[M, K]``.  If None, allocated by kernel.
    """

    hidden_states: Tensor = None
    gemm1: Gemm1Tensors = field(default_factory=Gemm1Tensors)
    gemm2: Gemm2Tensors = field(default_factory=Gemm2Tensors)
    # Monolithic path
    routing_logits: Optional[Tensor] = None
    routing_bias: Optional[Tensor] = None
    hidden_states_scale: Optional[Tensor] = None
    # Modular path
    token_selected_experts: Optional[Tensor] = None
    token_final_scales: Optional[Tensor] = None
    # Output
    output: Optional[Tensor] = None

    @property
    def is_monolithic(self) -> bool:
        """True if routing logits are provided (monolithic dispatch)."""
        return self.routing_logits is not None

    @property
    def is_modular(self) -> bool:
        """True if pre-routed expert assignments are provided."""
        return self.token_selected_experts is not None

    def validate(self) -> None:
        """Raise ValueError if the tensor configuration is inconsistent."""
        if self.hidden_states is None:
            raise ValueError("hidden_states is required")
        if self.is_monolithic and self.is_modular:
            raise ValueError(
                "Cannot provide both routing_logits (monolithic) and "
                "token_selected_experts (modular).  Choose one dispatch mode."
            )
        if not self.is_monolithic and not self.is_modular:
            raise ValueError(
                "Must provide either routing_logits (monolithic) or "
                "token_selected_experts (modular)."
            )
        if self.is_modular and self.token_final_scales is None:
            raise ValueError(
                "token_final_scales is required when using modular dispatch."
            )


# ---------------------------------------------------------------------------
# Activation / weight packs for the autotuned pre-routed path
# ---------------------------------------------------------------------------
# These are the runner-level inputs used by MoELayer (plan §1).
# Packs separate *per-call transient data* (activations) from *long-lived
# model state* (weights).  Each pack knows how to present itself to a
# specific backend via view_as / prepare_for — keeping backend-specific
# layout logic out of the dispatch hot-path.


@dataclass
class MoEActivationPack:
    """Per-call transient data — pre-quantized NVFP4 activations + pre-routed indices."""

    hidden_states_q: Tensor  # [M, H//2] uint8 (packed NVFP4)
    hidden_states_scale: Tensor  # [M, H//16] float8_e4m3fn
    selected_experts: Tensor  # [M, top_k] int32
    final_scales: Tensor  # [M, top_k] float32

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
