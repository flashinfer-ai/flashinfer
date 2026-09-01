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
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from ..tllm_enums import (
    DEFAULT_SITU_BETA,
    DEFAULT_SITU_LINEAR_BETA,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
    ActivationType,
    RoutingInputMode,
    RoutingMethodType,
)

# ---------------------------------------------------------------------------
# Kernel ceilings
# ---------------------------------------------------------------------------
# Mirrored from MaxSupportedTopExperts and NumNemotronExperts in the
# trtllm-gen DeepSeek router. These limits apply after adding shared experts.
MAX_SUPPORTED_TOP_EXPERTS = 32
MAX_SUPPORTED_TOTAL_EXPERTS = 512

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
# Routing reuses the shared kernel-level ``RoutingMethodType`` enum directly.
# Typed ActivationConfig values retain an accessible shared ``ActivationType``
# for the kernel ABI while also carrying activation-specific scalar semantics.
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
    MXFP4 = 5  # MXFP4 weights x MXFP8 activations (TRTLLM W4A8)
    MxInt4 = 6
    W4A16 = 7  # backend-specific 4-bit weights x BF16 activations
    W4A8 = 8  # INT4 weights x FP8 activations (CUTLASS SM90 packed mixed-input)
    Humming = 9  # MXFP4 weights x FP8 activations with Humming pre-MMA fusion

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
    """Typed fused activation value accepted by :class:`MoEConfig`.

    Concrete subclasses combine the activation identity with every scalar that
    changes its semantics. Per-expert tensor overrides remain properties of a
    backend-native weight view.
    """

    type: ClassVar[ActivationType]

    def __post_init__(self) -> None:
        if type(self) is ActivationConfig:
            raise TypeError(
                "ActivationConfig is a common base; construct a typed activation "
                "such as SwiGLU(), GeGLU(), or ReLU2()."
            )

    @property
    def is_gated(self) -> bool:
        return self.type.is_gated


def _validate_finite(name: str, value: float, *, positive: bool = False) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar, got {value!r}.")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}.")


@dataclass(frozen=True)
class SwiGLU(ActivationConfig):
    """SwiGLU/OA activation.

    The linear branch is clamped to ``[-limit, limit]`` and the gate branch to
    ``(-inf, limit]`` before evaluating
    ``gate * sigmoid(alpha * gate) * (linear + beta)``.
    """

    type: ClassVar[ActivationType] = ActivationType.Swiglu
    alpha: float = DEFAULT_SWIGLU_ALPHA
    beta: float = DEFAULT_SWIGLU_BETA
    limit: float = DEFAULT_SWIGLU_LIMIT

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_finite("alpha", self.alpha)
        _validate_finite("beta", self.beta)
        _validate_finite("limit", self.limit, positive=True)


@dataclass(frozen=True)
class SiTU(ActivationConfig):
    """SiTU v2 with canonical gate, linear, and optional clamp scales.

    ``linear_scale`` applies
    ``linear_scale * tanh(linear / linear_scale)``; ``None`` selects an
    unclamped linear branch, currently expressible only by CuTe-DSL. The
    defaults are the canonical SiTU (Kimi-K3) scales. Backend adapters must
    lower these exact values or reject unsupported semantics rather than rely
    on different native defaults.
    """

    type: ClassVar[ActivationType] = ActivationType.Situ
    gate_scale: float = DEFAULT_SITU_BETA
    linear_scale: Optional[float] = DEFAULT_SITU_LINEAR_BETA
    clamp_limit: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_finite("gate_scale", self.gate_scale, positive=True)
        if self.linear_scale is not None:
            _validate_finite("linear_scale", self.linear_scale, positive=True)
        if self.clamp_limit is not None:
            _validate_finite("clamp_limit", self.clamp_limit, positive=True)


@dataclass(frozen=True)
class GeGLU(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.Geglu


@dataclass(frozen=True)
class ReLU2(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.Relu2


@dataclass(frozen=True)
class GeGLUTanh(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.GegluTanh


@dataclass(frozen=True)
class SwiGLUStep(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.SwigluStep
    limit: float = 7.0

    def __post_init__(self) -> None:
        super().__post_init__()
        _validate_finite("limit", self.limit, positive=True)


@dataclass(frozen=True)
class Identity(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.Identity


@dataclass(frozen=True)
class GELU(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.Gelu


@dataclass(frozen=True)
class ReLU(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.Relu


@dataclass(frozen=True)
class SiLU(ActivationConfig):
    type: ClassVar[ActivationType] = ActivationType.Silu


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
    num_fused_shared_experts : int
        Number of shared experts run for every token. Their rows follow the
        routed experts, so weights have ``E + S`` rows while routing fields
        remain routed-only. Cross-field constraints are checked by
        :class:`MoEConfig`.
    """

    intermediate_size: int
    local_expert_offset: int = 0
    local_num_experts: Optional[int] = None
    num_fused_shared_experts: int = 0

    def __post_init__(self) -> None:
        if self.num_fused_shared_experts < 0:
            raise ValueError(
                "num_fused_shared_experts must be >= 0, got "
                f"{self.num_fused_shared_experts}."
            )

    def __repr__(self) -> str:
        parts = [f"intermediate_size={self.intermediate_size!r}"]
        if self.local_expert_offset != 0:
            parts.append(f"local_expert_offset={self.local_expert_offset!r}")
        if self.local_num_experts is not None:
            parts.append(f"local_num_experts={self.local_num_experts!r}")
        if self.num_fused_shared_experts != 0:
            parts.append(f"num_fused_shared_experts={self.num_fused_shared_experts!r}")
        return f"ExpertConfig({', '.join(parts)})"


@dataclass(frozen=True)
class MoEFinalizeConfig:
    """How the finalize (combine) step behaves.

    Split out of ``ExecutionConfig`` for the same reason ``RoutingConfig`` is
    its own config: finalize is a distinct architectural concern (how the
    per-expert partials are reduced back into one row per token), not a
    runtime knob like PDL or the autotuner token budget.

    Parameters
    ----------
    do_finalize : bool
        Whether to apply routing-weight scaling and accumulate the per-expert
        partial results into the output.  ``False`` returns the unreduced
        TRTLLM intermediates as ``[gemm2_output, expert_weights,
        expanded_idx_to_permuted_idx]``, leaving the combine to the caller.
        For FromLogits routing, the routing kernel emits ``expert_weights`` in
        bfloat16 regardless of the routing-logits dtype. ``PackedPrecomputed``
        routing also yields bfloat16 weights: the caller's values are narrowed
        to bfloat16 when packed into the top-k ids. Only
        ``UnpackedPrecomputed`` routing preserves the caller-provided weights
        dtype, since it forwards ``topk_weights`` to the kernel unchanged.
        Only backends that advertise unfinalized output support this mode.
    use_fused_finalize : bool
        Whether supported backends reduce routed outputs in the GEMM2 epilogue
        (atomic accumulation) instead of running a separate reduction kernel.
        Backends that do not support it ignore the flag.
    """

    do_finalize: bool = True
    use_fused_finalize: bool = True

    def __repr__(self) -> str:
        parts = []
        if not self.do_finalize:
            parts.append(f"do_finalize={self.do_finalize!r}")
        if not self.use_fused_finalize:
            parts.append(f"use_fused_finalize={self.use_fused_finalize!r}")
        return f"MoEFinalizeConfig({', '.join(parts)})"


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime execution parameters.

    Parameters
    ----------
    enable_pdl : bool or None
        Persistent device launch.  ``None`` → auto (True for sm90+).
    tune_max_num_tokens : int
        Token budget hint for autotuner / CUDA graph capture.
    """

    enable_pdl: Optional[bool] = None
    tune_max_num_tokens: int = 8192

    def __repr__(self) -> str:
        parts = []
        if self.enable_pdl is not None:
            parts.append(f"enable_pdl={self.enable_pdl!r}")
        if self.tune_max_num_tokens != 8192:
            parts.append(f"tune_max_num_tokens={self.tune_max_num_tokens!r}")
        return f"ExecutionConfig({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Backend configs — each declares hardware preconditions
# ---------------------------------------------------------------------------

# Architectures the TRT-LLM routed-MoE cubin manifest ships kernels for.  The
# manifest is a downloaded artifact, so this cannot be derived at import time and
# has to be kept in sync with the cubin-arch compatibility rules in
# csrc/trtllm_batched_gemm_runner.cu.  SM110/120/121 need upstream cubins first;
# claiming them here makes the batched-GEMM runner abort at dispatch (#4107).
_TRTLLM_ROUTED_ARCHS = (100, 103, 107)

# The FP8 kernels are validated on the SM100 family only — the outer JIT module
# compiles for major 12 as well, but those cubins fail at runtime on SM120/121.
_TRTLLM_ROUTED_FP8_ARCHS = (100, 103)

# Dense BF16 follows the architecture dispatch already exposed by the flat
# CUTLASS API.
_CUTLASS_BF16_ARCHS = (89, 90, 100, 103, 107, 110, 120, 121)

# W4A16 uses Hopper-specific mixed-input weight and scale layouts.
_CUTLASS_W4A16_ARCHS = (90,)

_CUTILE_BF16_ARCHS = (89, 90, 120, 121)
_CUTILE_NVFP4_ARCHS = (120, 121)
_CUTILE_SUPPORTED_ACTIVATIONS = (ActivationType.Swiglu, ActivationType.Relu2)

# NVFP4 CUTLASS fused MoE matches the flat-API skip: SM100/SM110/SM12x.
# Major 10/11/12 covers SM100/103/107, SM110, and SM120/121.
_CUTLASS_NVFP4_ARCHS = (100, 103, 107, 110, 120, 121)

# Per-tensor FP8 follows the dense CUTLASS architecture list.
_CUTLASS_FP8_ARCHS = _CUTLASS_BF16_ARCHS

# DeepSeek-style 128x128 FP8 block scaling is Hopper-only in the flat API.
_CUTLASS_FP8_BLOCK_ARCHS = (90,)

# MXFP8 activations x MXFP4 weights matches a *wider* flat-API skip than
# MXFP8 x MXFP8: ``capability[0] not in [10, 11, 12]`` (SM100/103/107, SM110,
# SM120/121). Do not collapse this to ``_CUTLASS_MXFP8_ARCHS``.
_CUTLASS_MXFP8_MXFP4_ARCHS = (100, 103, 107, 110, 120, 121)

# MXFP8 x MXFP8 follows the narrower flat skip: ``capability[0] not in [10]``
# (SM10x only, including B300 SM103). Not claimed on SM11x/SM12x.
_CUTLASS_MXFP8_ARCHS = (100, 103, 107)

# INT4 W4A8 and Humming MXFP4 x FP8 are Hopper mixed-input paths.
_CUTLASS_W4A8_ARCHS = (90,)
_CUTLASS_HUMMING_ARCHS = (90,)


@dataclass(frozen=True)
class TrtllmFp4Config:
    """TensorRT-LLM FP4 backend for NVFP4 and MXFP4 mixed-precision modes.

    ``supported(arch)`` reflects the routed-MoE cubin manifest. Variant-specific
    restrictions are applied by ``TrtllmFp4RoutedRunner.check_support()``:
    NVFP4/MXFP4 support SM100/SM103/SM107, while W4A16 supports SM100/SM107
    and remains disabled on SM103.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _TRTLLM_ROUTED_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        variant: QuantVariant = QuantVariant.NVFP4,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
        permute_cache=None,
    ):
        """Build a ``trtllm_fp4_routed`` weight view from canonical BF16 weights.

        Register the result with ``MoEWeightPack.prepare_for("trtllm_fp4_routed", ...)``.
        ``variant`` selects NVFP4, MXFP4xMXFP8, or ``QuantVariant.W4A16``
        (MXFP4 weights x BF16 activations).
        See :func:`flashinfer.fused_moe.prepare.prepare_trtllm_fp4_weights`.

        .. warning::
           ``num_local_experts`` is the physical row count: ``E_local + S``
           when fused shared experts are present. :class:`ExpertConfig` keeps
           ``local_num_experts`` as the routed-only count ``E_local``.
        """
        from .prepare import prepare_trtllm_fp4_weights

        return prepare_trtllm_fp4_weights(
            w1_bf16,
            w2_bf16,
            variant=variant,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
            permute_cache=permute_cache,
        )

    @staticmethod
    def prepare_activations(
        hidden_states_bf16,
        *,
        variant: QuantVariant = QuantVariant.NVFP4,
    ):
        """Prepare activations for NVFP4, MXFP4xMXFP8, or ``QuantVariant.W4A16``.

        W4A16 returns raw BF16 activations without an activation scale.
        """
        from .prepare import prepare_trtllm_fp4_activations

        return prepare_trtllm_fp4_activations(
            hidden_states_bf16,
            variant=variant,
        )

    def __repr__(self) -> str:
        return "TrtllmFp4Config()"


@dataclass(frozen=True)
class TrtllmFp8BlockConfig:
    """TensorRT-LLM FP8 block-scale backend."""

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _TRTLLM_ROUTED_FP8_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        variant: QuantVariant,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Build the ``trtllm_fp8_block`` weight view from canonical BF16.

        ``variant`` must be :attr:`QuantVariant.DeepSeekFp8` or
        :attr:`QuantVariant.MxFp8`; their scale formats are intentionally
        prepared by separate paths. The shuffled MXFP8 view requires both
        ``hidden_size`` and ``intermediate_size`` to be divisible by 128 so its
        scale tensors fit TRTLLM's unpadded 128x4 physical layout.

        .. warning::
           ``num_local_experts`` here is the **physical row count** of
           ``w1_bf16`` / ``w2_bf16``: ``E_local + S`` with shared experts.
           :attr:`ExpertConfig.local_num_experts` remains routed-only
           (``E_local``).
        """
        from .prepare import prepare_trtllm_fp8_block_weights

        return prepare_trtllm_fp8_block_weights(
            w1_bf16,
            w2_bf16,
            variant=variant,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
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
        return arch in _TRTLLM_ROUTED_FP8_ARCHS

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
        activation: Optional[ActivationConfig] = None,
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
            activation=activation,
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
        return arch in _TRTLLM_ROUTED_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
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
            activation=activation,
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
        # SM100/SM103 use the forward-compatible sm100f cubins; SM107 selects
        # the dedicated Rubin artifact.
        return arch in _TRTLLM_ROUTED_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
        permute_cache=None,
    ):
        """Build a ``trtllm_mxint4_routed`` view from canonical BF16 weights."""
        from .prepare import prepare_trtllm_mxint4_weights

        return prepare_trtllm_mxint4_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
            permute_cache=permute_cache,
        )

    def __repr__(self) -> str:
        return "TrtllmMxInt4Config()"


@dataclass(frozen=True)
class CutlassBf16Config:
    """CUTLASS BF16 backend for the unified MoE API.

    Architecture coverage follows the dense-BF16 legacy flat API. The unified
    GPU tests currently exercise SM90.

    This backend supports packed precomputed routing with all flat CUTLASS
    activation semantics and requires ``do_finalize=True``.
    Expert parallelism and shared experts are not supported.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTLASS_BF16_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Build the ``cutlass_bf16`` canonical BF16 weight view.

        GEMM1 uses the public ``[up, gate]`` row convention.  Unlike TRTLLM's
        BlockMajorK path, CUTLASS BF16 kernels consume these weights directly
        and need no physical reordering.
        """
        from .prepare import prepare_cutlass_bf16_weights

        return prepare_cutlass_bf16_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )

    def __repr__(self) -> str:
        return "CutlassBf16Config()"


@dataclass(frozen=True)
class CuTileBf16Config:
    """cuTile BF16 backend.

    Expert parallelism and fused shared experts are not supported.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTILE_BF16_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Build the ``cutile_bf16`` weight view from canonical BF16 weights."""
        from .prepare import prepare_cutile_bf16_weights

        return prepare_cutile_bf16_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_type=(activation or SwiGLU()).type,
            device=device,
        )

    def __repr__(self) -> str:
        return "CuTileBf16Config()"


@dataclass(frozen=True)
class CuTileNvfp4Config:
    """cuTile NVFP4-weight x NVFP4-activation backend.

    Expert parallelism and fused shared experts are not supported.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTILE_NVFP4_ARCHS

    @staticmethod
    def prepare_weights(
        w1_fp4,
        w1_block_scale,
        w1_global_scale,
        w2_fp4,
        w2_block_scale,
        w2_global_scale,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        source_format: str = "modelopt",
        device=None,
    ):
        """Build the ``cutile_nvfp4`` view from checkpoint NVFP4 weights."""
        from .prepare import prepare_cutile_nvfp4_weights

        return prepare_cutile_nvfp4_weights(
            w1_fp4,
            w1_block_scale,
            w1_global_scale,
            w2_fp4,
            w2_block_scale,
            w2_global_scale,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation_type=(activation or SwiGLU()).type,
            source_format=source_format,
            device=device,
        )

    def __repr__(self) -> str:
        return "CuTileNvfp4Config()"


@dataclass(frozen=True)
class CutlassW4A16Config:
    """CUTLASS MXFP4-weight x BF16-activation backend for SM90.

    This backend supports packed precomputed routing with all flat CUTLASS
    activation semantics and requires ``do_finalize=True``.
    Expert parallelism and shared experts are not supported. Both
    ``hidden_size`` and ``intermediate_size`` must be divisible by 128.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTLASS_W4A16_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Quantize and interleave canonical BF16 weights for SM90 W4A16."""
        from .prepare import prepare_cutlass_w4a16_weights

        return prepare_cutlass_w4a16_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )

    def __repr__(self) -> str:
        return "CutlassW4A16Config()"


@dataclass(frozen=True)
class CutlassNvfp4Config:
    """CUTLASS NVFP4 backend for SM100 / SM110 / SM12x.

    Packed precomputed routing with all flat CUTLASS activation semantics and
    ``do_finalize=True``. Expert
    parallelism and shared experts are not supported. Both ``hidden_size``
    and ``intermediate_size`` must be divisible by 16 (the NVFP4 scale-vector
    size). Activations stay BF16; the kernel quantizes them internally.

    This config is not in the default backend search list. TRTLLM NVFP4 uses
    a quantized activation pack, so the two contracts cannot share one
    ``MoEActivationPack``. Select it explicitly with
    ``BackendOptions((CutlassNvfp4Config(),))``.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTLASS_NVFP4_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Quantize canonical BF16 weights into CUTLASS-swizzled NVFP4.

        Uses the flat CUTLASS scale layout (``fp4_quantize`` with swizzled
        scales), not the TRTLLM shuffle / BlockMajorK path.
        """
        from .prepare import prepare_cutlass_nvfp4_weights

        return prepare_cutlass_nvfp4_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )

    def __repr__(self) -> str:
        return "CutlassNvfp4Config()"


@dataclass(frozen=True)
class CutlassFp8PerTensorConfig:
    """CUTLASS per-tensor FP8 backend.

    Activations are prequantized E4M3 with a scalar dequant scale on
    ``MoEActivationPack.hidden_states_scale``. Weights stay unshuffled; this is
    not the TRTLLM MajorK view. Packed precomputed routing with all flat
    CUTLASS activation semantics and
    ``do_finalize=True``. Not in the default backend search list: TRTLLM
    per-tensor FP8 folds the activation scale into the weight view, so the two
    contracts cannot share one pack without a conversion.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTLASS_FP8_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Quantize canonical BF16 weights into unshuffled per-tensor FP8."""
        from .prepare import prepare_cutlass_fp8_per_tensor_weights

        return prepare_cutlass_fp8_per_tensor_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )

    @staticmethod
    def prepare_activations(hidden_states_bf16):
        """Quantize BF16 activations to E4M3 plus a scalar dequant scale."""
        from .prepare import prepare_cutlass_fp8_per_tensor_activations

        return prepare_cutlass_fp8_per_tensor_activations(hidden_states_bf16)

    def __repr__(self) -> str:
        return "CutlassFp8PerTensorConfig()"


@dataclass(frozen=True)
class CutlassFp8BlockConfig:
    """CUTLASS DeepSeek-style 128x128 FP8 block-scale backend (SM90).

    Activations stay BF16; the kernel quantizes them internally. This is not
    the TRTLLM block-FP8 activation pack. Packed precomputed routing with
    all flat CUTLASS activation semantics and ``do_finalize=True``.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTLASS_FP8_BLOCK_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Quantize canonical BF16 weights into 128x128 FP8 block scales."""
        from .prepare import prepare_cutlass_fp8_block_weights

        return prepare_cutlass_fp8_block_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )

    def __repr__(self) -> str:
        return "CutlassFp8BlockConfig()"


@dataclass(frozen=True)
class CutlassMxfp8Mxfp4Config:
    """CUTLASS MXFP8-activation x MXFP4-weight backend for SM100 / SM110 / SM12x.

    Activations are MXFP8 with a swizzled ``input_sf``. Weights are packed
    MXFP4 viewed as int64 at launch. Packed precomputed routing with all flat
    CUTLASS activation semantics and ``do_finalize=True``. Both ``hidden_size`` and ``intermediate_size``
    must be divisible by 128.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTLASS_MXFP8_MXFP4_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Quantize canonical BF16 weights into CUTLASS MXFP4."""
        from .prepare import prepare_cutlass_mxfp8_mxfp4_weights

        return prepare_cutlass_mxfp8_mxfp4_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )

    @staticmethod
    def prepare_activations(hidden_states_bf16):
        """Quantize BF16 activations to MXFP8 with a swizzled scale buffer."""
        from .prepare import prepare_cutlass_mxfp8_activations

        return prepare_cutlass_mxfp8_activations(hidden_states_bf16)

    def __repr__(self) -> str:
        return "CutlassMxfp8Mxfp4Config()"


@dataclass(frozen=True)
class CutlassMxfp8Config:
    """CUTLASS MXFP8-activation x MXFP8-weight backend (SM100 / SM103 / SM107).

    Activations are MXFP8 with a swizzled ``input_sf``. Weights stay E4M3 with
    packed int32 scale tiles. ``hidden_size`` and ``intermediate_size`` must be
    divisible by 128 so the gated fc1 scale layout matches the binding.
    Packed precomputed routing with all flat CUTLASS activation semantics and
    ``do_finalize=True``.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTLASS_MXFP8_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Quantize canonical BF16 weights into CUTLASS MXFP8."""
        from .prepare import prepare_cutlass_mxfp8_weights

        return prepare_cutlass_mxfp8_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )

    @staticmethod
    def prepare_activations(hidden_states_bf16):
        """Quantize BF16 activations to MXFP8 with a swizzled scale buffer."""
        from .prepare import prepare_cutlass_mxfp8_activations

        return prepare_cutlass_mxfp8_activations(hidden_states_bf16)

    def __repr__(self) -> str:
        return "CutlassMxfp8Config()"


@dataclass(frozen=True)
class CutlassW4A8Config:
    """CUTLASS INT4-weight x FP8-activation backend for SM90.

    Activations stay BF16; the kernel quantizes them internally with the packed
    mixed-input INT4 layout. Packed precomputed routing with all flat CUTLASS
    activation semantics and
    ``do_finalize=True``.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTLASS_W4A8_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Quantize canonical BF16 weights into SM90 interleaved INT4."""
        from .prepare import prepare_cutlass_w4a8_weights

        return prepare_cutlass_w4a8_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )

    def __repr__(self) -> str:
        return "CutlassW4A8Config()"


@dataclass(frozen=True)
class CutlassHummingConfig:
    """CUTLASS Humming MXFP4-weight x FP8-activation backend for SM90.

    Activations stay BF16; weights use Humming pre-MMA E8M0 fusion plus the
    SM90 mixed-input interleave. Packed precomputed routing with all flat
    CUTLASS activation semantics and
    ``do_finalize=True``.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        return arch in _CUTLASS_HUMMING_ARCHS

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Quantize canonical BF16 weights into the Humming mixed-input layout."""
        from .prepare import prepare_cutlass_humming_weights

        return prepare_cutlass_humming_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
            device=device,
        )

    def __repr__(self) -> str:
        return "CutlassHummingConfig()"


@dataclass(frozen=True)
class CuteDslConfig:
    """CuteDSL FP4 backend for W4A4, W4A8, and W4A16.

    The underlying CuteDSL kernel throws at launch on SM120/SM121/SM130.
    """

    @classmethod
    def supported(cls, arch: int) -> bool:
        # SM100, SM103 (Blackwell) + SM107 (Rubin) — tighten when CuteDSL adds more targets
        return arch in (100, 103, 107)

    @staticmethod
    def prepare_weights(
        w1_bf16,
        w2_bf16,
        *,
        variant: QuantVariant = QuantVariant.NVFP4,
        num_local_experts: int,
        hidden_size: int,
        intermediate_size: int,
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Build the ``cute_dsl`` weight view from canonical BF16 weights.

        Register the result with ``MoEWeightPack.prepare_for("cute_dsl", ...)``.
        """
        from .prepare import prepare_cute_dsl_weights

        return prepare_cute_dsl_weights(
            w1_bf16,
            w2_bf16,
            variant=variant,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=activation,
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
        activation: Optional[ActivationConfig] = None,
        device=None,
    ):
        """Build the ``b12x_nvfp4`` weight view from canonical bf16 weights.

        Register the result with ``MoEWeightPack.prepare_for("b12x_nvfp4", ...)``.
        See :func:`flashinfer.fused_moe.prepare.prepare_b12x_nvfp4_weights`.
        """
        from .prepare import prepare_b12x_nvfp4_weights
        from .utils import resolve_b12x_activation_name

        return prepare_b12x_nvfp4_weights(
            w1_bf16,
            w2_bf16,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            activation=resolve_b12x_activation_name(activation),
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
        activation: Optional[ActivationConfig] = None,
        source_format: str = "modelopt",
    ):
        """Build the ``b12x_w4a16`` weight view from checkpoint fp4 weights.

        Register the result with ``MoEWeightPack.prepare_for("b12x_w4a16", ...)``.
        See :func:`flashinfer.fused_moe.prepare.prepare_b12x_w4a16_weights`.
        """
        from .prepare import prepare_b12x_w4a16_weights
        from .utils import resolve_b12x_activation_name

        return prepare_b12x_w4a16_weights(
            w1_fp4,
            w1_blockscale,
            w1_global_scale,
            w2_fp4,
            w2_blockscale,
            w2_global_scale,
            activation=resolve_b12x_activation_name(activation),
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
    CutlassBf16Config,
    CuTileBf16Config,
    CuTileNvfp4Config,
    CutlassW4A16Config,
    CutlassNvfp4Config,
    CutlassFp8PerTensorConfig,
    CutlassFp8BlockConfig,
    CutlassMxfp8Mxfp4Config,
    CutlassMxfp8Config,
    CutlassW4A8Config,
    CutlassHummingConfig,
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
    CutlassBf16Config,
    CuTileBf16Config,
    CuTileNvfp4Config,
    CutlassW4A16Config,
    CutlassNvfp4Config,
    CutlassFp8PerTensorConfig,
    CutlassFp8BlockConfig,
    CutlassMxfp8Mxfp4Config,
    CutlassMxfp8Config,
    CutlassW4A8Config,
    CutlassHummingConfig,
    CuteDslConfig,
    B12xNvfp4Config,
    B12xW4A16Config,
)


# ---------------------------------------------------------------------------
# BackendOptions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendOptions:
    """Ordered list of backend candidates for dispatch and autotuning.

    Each backend config implements ``supported(arch)``, where ``arch`` uses the
    CUDA compute-capability encoding documented by :meth:`valid_for`.
    """

    candidates: Tuple[BackendConfigType, ...] = ()  # type: ignore[type-arg]

    def valid_for(self, arch: int) -> list:
        """Return candidates whose hardware preconditions are met.

        Parameters
        ----------
        arch : int
            CUDA compute capability encoded as ``major * 10 + minor``. For
            example, SM90 is ``90``, SM100 is ``100``, and SM103 is ``103``.
            :class:`MoELayer` derives it from its selected CUDA device via
            ``get_compute_capability(self.device)``. This is not a CUDA device
            ordinal or CUDA toolkit version.
        """
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
        CutlassBf16Config(),
        CutlassW4A16Config(),
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
    activation: ActivationConfig = field(default_factory=SwiGLU)
    backend: BackendOptions = field(default_factory=lambda: _DEFAULT_BACKEND)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    # Appended last so existing positional construction keeps working.
    finalize: MoEFinalizeConfig = field(default_factory=MoEFinalizeConfig)

    def __post_init__(self) -> None:
        # Not in check_support(): MoELayer swallows its exceptions to filter
        # backends, so errors raised there surface as "no backend available".
        self._validate_fused_shared_experts()

    def _validate_fused_shared_experts(self) -> None:
        s = self.experts.num_fused_shared_experts
        if s == 0:
            return

        # Only DeepSeekV3 routing emits shared-expert slots.
        if self.routing.method is not RoutingMethodType.DeepSeekV3:
            raise ValueError(
                "num_fused_shared_experts > 0 requires DeepSeekV3 routing, got "
                f"method={self.routing.method!r}."
            )

        # Kernel limits apply to the fused totals.
        total_top_k = self.routing.top_k + s
        if total_top_k > MAX_SUPPORTED_TOP_EXPERTS:
            raise ValueError(
                f"top_k + num_fused_shared_experts must be <= "
                f"{MAX_SUPPORTED_TOP_EXPERTS}, got {self.routing.top_k} + {s} = "
                f"{total_top_k}."
            )
        total_experts = self.routing.num_experts + s
        if total_experts > MAX_SUPPORTED_TOTAL_EXPERTS:
            raise ValueError(
                f"num_experts + num_fused_shared_experts must be <= "
                f"{MAX_SUPPORTED_TOTAL_EXPERTS}, got {self.routing.num_experts} "
                f"+ {s} = {total_experts}."
            )

        # The kernel maps a shared id to a weight row as
        # (global_id - local_expert_offset), so all routed experts must be local.
        local_num_experts = (
            self.experts.local_num_experts
            if self.experts.local_num_experts is not None
            else self.routing.num_experts
        )
        if self.experts.local_expert_offset != 0 or (
            local_num_experts != self.routing.num_experts
        ):
            raise ValueError(
                "num_fused_shared_experts > 0 does not support expert "
                "parallelism: require local_expert_offset == 0 and "
                "local_num_experts == num_experts. Got "
                f"num_fused_shared_experts={s}, "
                f"local_expert_offset={self.experts.local_expert_offset}, "
                f"local_num_experts={local_num_experts}, "
                f"num_experts={self.routing.num_experts}."
            )

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
    * MXFP4 (W4A8): ``float8_e4m3fn [M, H]`` MXFP8 values with token-major
      ``float8_e4m3fn [M, H/32]`` tensors carrying UE8M0 scale bytes, matching
      the TRTLLM FP4 launcher ABI.
    * W4A16 with ``TrtllmFp4Config``: raw ``bfloat16 [M, H]`` values with no
      activation scale; weights use the MXFP4 preparation contract.
    * BF16: raw ``bfloat16 [M, H]`` values with no scale tensor.
    * MxInt4: raw ``bfloat16 [M, H]`` values with no scale tensor; weights are
      packed signed INT4 with BF16 block scales.
    * DeepSeek FP8: ``float8_e4m3fn [M, H]`` values with transposed
      ``float32 [H/128, M]`` block scales.
    * MXFP8: ``float8_e4m3fn [M, H]`` values with token-major
      ``uint8 [M, H/32]`` UE8M0 scales.
    * FP8 per-tensor: ``float8_e4m3fn [M, H]`` values with no scale tensor;
      the calibrated scalar is folded into the backend's epilogue scales.

    ``routing_input_mode`` selects how routing reaches the kernel (the runner reads it directly):

    * ``PackedPrecomputed`` (default) — **pre-routed**: the caller computes expert
      selection on the host and passes ``topk_ids`` + ``topk_weights``.
      The TRTLLM runners normally combine both fields into one packed ``int32``
      tensor before launch.
    * ``UnpackedPrecomputed`` — **pre-routed, separate kernel inputs**: supported
      by the TRTLLM runners. The caller supplies ``int32`` ids and BF16 or FP32
      weights directly, avoiding packed-id construction. The launcher consumes
      the weights in their native dtype.
    * ``FromLogits`` — **in-kernel**: the caller passes raw ``routing_logits`` (and, for bias-aware
      methods like DeepSeekV3/MiniMax2, ``routing_bias``); the kernel computes the top-k selection
      itself per ``RoutingConfig.method``.  ``topk_ids`` / ``topk_weights`` stay ``None`` — the
      runner allocates internal kernel-filled buffers, and the routing result is not surfaced
      back through the pack (routing replay is a separate, future capability). TRTLLM FP4,
      BF16, block-FP8, per-tensor-FP8, and MxInt4 runners support this mode;
      ``MoELayer`` dispatches a logits pack only to capable backends (see each runner's
      ``supported_routing_modes``).

    ``topk_ids`` / ``topk_weights`` follow the routed-MoE naming convention (gh #2425); they
    keep the field positions of the former ``selected_experts`` / ``final_scales``, so
    positional construction of pre-routed packs is unchanged. Additional activation metadata
    and the in-kernel routing fields are keyword-only.
    """

    # Backend-native activation payload; layouts documented above.
    hidden_states_q: Tensor
    # Variant-specific scales documented above; None for BF16/per-tensor FP8.
    hidden_states_scale: Optional[Tensor]
    # Pre-routed top-k selection (Packed/Unpacked modes); None under FromLogits.
    topk_ids: Optional[Tensor] = None  # [M, top_k] int32 (expert indices)
    # [M, top_k] routing weights: float32 for PackedPrecomputed; bfloat16 or
    # float32 for TRTLLM UnpackedPrecomputed.
    topk_weights: Optional[Tensor] = None
    # Per-token NVFP4 row scale, shape [M].
    per_token_scale: Optional[Tensor] = field(default=None, kw_only=True)
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
        elif mode == RoutingInputMode.UnpackedPrecomputed:
            if self.topk_ids is None or self.topk_weights is None:
                raise ValueError(
                    "routing_input_mode=UnpackedPrecomputed requires "
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
            if self.topk_weights.dtype not in (torch.bfloat16, torch.float32):
                raise TypeError(
                    "UnpackedPrecomputed topk_weights must be torch.bfloat16 "
                    "or torch.float32 "
                    f"(got {self.topk_weights.dtype})."
                )
            expected = (self.hidden_states_q.shape[0],)
            if self.topk_ids.ndim != 2 or self.topk_weights.ndim != 2:
                raise ValueError(
                    "UnpackedPrecomputed topk_ids/topk_weights must both be 2-D "
                    f"[num_tokens, top_k], got {tuple(self.topk_ids.shape)} and "
                    f"{tuple(self.topk_weights.shape)}."
                )
            if (
                self.topk_ids.shape != self.topk_weights.shape
                or self.topk_ids.shape[:1] != expected
            ):
                raise ValueError(
                    "UnpackedPrecomputed topk_ids/topk_weights must have matching "
                    f"[num_tokens, top_k] shapes, got {tuple(self.topk_ids.shape)} "
                    f"and {tuple(self.topk_weights.shape)} for "
                    f"num_tokens={expected[0]}."
                )
            if (
                not self.topk_ids.is_contiguous()
                or not self.topk_weights.is_contiguous()
            ):
                raise ValueError(
                    "UnpackedPrecomputed topk_ids/topk_weights must be contiguous."
                )
        else:
            raise ValueError(f"Unsupported routing_input_mode={mode!r}.")

        # All routing tensors must live with the activations; a stray CPU
        # tensor otherwise surfaces as a cryptic launch/ICHECK failure.
        dev = self.hidden_states_q.device
        for name in (
            "hidden_states_scale",
            "topk_ids",
            "topk_weights",
            "per_token_scale",
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
