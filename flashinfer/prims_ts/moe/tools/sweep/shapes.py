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

"""Host-side shape modeling for Prims-TS MoE sweeps."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Literal, Mapping

from ...config_mapper import (
    SUPPORTED_BF16_TILE_N,
    SUPPORTED_DSFP8_TILE_N,
    SUPPORTED_FP8_TILE_N,
    SUPPORTED_MXFP4_BF16_TILE_N,
    SUPPORTED_MXFP4_MXFP8_TILE_N,
    SUPPORTED_MXFP8_MXFP8_TILE_N,
    SUPPORTED_NVFP4_TILE_N,
    _selected_tile_ns,
)
from .defaults import ROUTING_SEED


FcName = Literal["fc1", "fc2"]

BALANCED_ROUTING = "balanced_round_robin"


@dataclass(frozen=True)
class ModelPreset:
    """Model-level inputs shared by all TP, EP, and token-count shapes."""

    quant_mode: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    prims_ts_activation: str
    trtllm_activation: str
    has_bias: bool = False
    gemm1_alpha: float | None = None
    gemm1_beta: float | None = None
    gemm1_clamp_limit: float | None = None
    note: str = ""

    @property
    def dtype(self) -> str:
        """Sweep dtype spelling used by the runner's quant-mode argument."""

        return self.quant_mode

    @property
    def activation(self) -> str:
        """Activation used by the Prims-TS runner."""

        return self.prims_ts_activation


MODEL_PRESETS: Mapping[str, ModelPreset] = {
    "deepseek-v3-nvfp4": ModelPreset(
        quant_mode="NvFP4xNvFP4",
        hidden_size=7168,
        intermediate_size=2048,
        num_experts=256,
        top_k=8,
        prims_ts_activation="Swiglu",
        trtllm_activation="Swiglu",
        note=("NvFP4 DeepSeek-V3/R1 routed experts; the shared expert is excluded."),
    ),
    "deepseek-v4-pro-nvfp4": ModelPreset(
        quant_mode="NvFP4xNvFP4",
        hidden_size=7168,
        intermediate_size=3072,
        num_experts=384,
        top_k=6,
        prims_ts_activation="Swiglu",
        trtllm_activation="Swiglu",
        gemm1_clamp_limit=10.0,
    ),
    "deepseek-v4.1-flash": ModelPreset(
        quant_mode="MxFP4xMxFP8",
        hidden_size=5120,
        intermediate_size=2304,
        num_experts=384,
        top_k=6,
        prims_ts_activation="Swiglu",
        trtllm_activation="Swiglu",
        gemm1_clamp_limit=10.0,
    ),
    "deepseek-v4.1-flash-mxfp8": ModelPreset(
        quant_mode="MxFP8xMxFP8",
        hidden_size=5120,
        intermediate_size=2304,
        num_experts=384,
        top_k=6,
        prims_ts_activation="Swiglu",
        trtllm_activation="Swiglu",
        gemm1_clamp_limit=10.0,
        note="MXFP8 weight variant of DeepSeek-V4.1-Flash.",
    ),
    "kimi-k3": ModelPreset(
        quant_mode="MxFP4xMxFP8",
        hidden_size=3584,
        intermediate_size=3072,
        num_experts=896,
        top_k=16,
        prims_ts_activation="Situ",
        trtllm_activation="Situ",
        gemm1_alpha=4.0,
        gemm1_beta=25.0,
        note=(
            "Prims-TS profiles Kimi K3's routed experts with SiTU; the two "
            "shared experts are excluded."
        ),
    ),
    "kimi-k3-nvfp4": ModelPreset(
        quant_mode="NvFP4xNvFP4",
        hidden_size=3584,
        intermediate_size=3072,
        num_experts=896,
        top_k=16,
        prims_ts_activation="Situ",
        trtllm_activation="Situ",
        gemm1_alpha=4.0,
        gemm1_beta=25.0,
        note=(
            "NvFP4 variant of Kimi K3's SiTU routed experts; the two shared "
            "experts are excluded."
        ),
    ),
    "qwen3-30b-a3b": ModelPreset(
        quant_mode="NvFP4xNvFP4",
        hidden_size=2048,
        intermediate_size=768,
        num_experts=128,
        top_k=8,
        prims_ts_activation="Swiglu",
        trtllm_activation="Swiglu",
    ),
    "gpt-oss-120b": ModelPreset(
        quant_mode="MxFP4xMxFP8",
        # The real hidden and intermediate sizes are both 2880. Prims-TS
        # MXFP4xMXFP8 requires both to be multiples of 128, and one hidden
        # size here serves as both FC1 K and FC2 N, so both are padded to
        # 3072, which stays 128-aligned per rank for TP in {1, 2, 4, 8}.
        hidden_size=3072,
        intermediate_size=3072,
        num_experts=128,
        top_k=4,
        prims_ts_activation="Swiglu",
        trtllm_activation="Swiglu",
        has_bias=True,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
        note=(
            "Prims-TS profiles GPT-OSS-120B as a padded shape proxy: sizes are "
            "padded from 2880 to 3072."
        ),
    ),
    "gpt-oss-120b-mxfp8": ModelPreset(
        quant_mode="MxFP8xMxFP8",
        # Keep the same aligned proxy geometry as the MXFP4xMXFP8 preset.
        hidden_size=3072,
        intermediate_size=3072,
        num_experts=128,
        top_k=4,
        prims_ts_activation="Swiglu",
        trtllm_activation="Swiglu",
        has_bias=True,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
        note=(
            "MXFP8 weight variant of the padded GPT-OSS-120B proxy; sizes are "
            "padded from 2880 to 3072."
        ),
    ),
    "minimax-m3-mxfp8": ModelPreset(
        quant_mode="MxFP8xMxFP8",
        hidden_size=6144,
        intermediate_size=3072,
        num_experts=128,
        top_k=4,
        # MiniMax-M3's "swigluoai" is GPT-OSS's clamped SwiGLU without
        # expert biases.
        prims_ts_activation="Swiglu",
        trtllm_activation="Swiglu",
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
        note=("MiniMax-M3-MXFP8 routed experts; the shared expert is excluded."),
    ),
    "nemotron-3.5-lightning-nvfp4": ModelPreset(
        quant_mode="NvFP4xNvFP4",
        hidden_size=2688,
        # The real intermediate size is 1856. ReLU2 FC1 is not gated, so its
        # output width equals the per-rank intermediate size, and swap-AB
        # requires a multiple of tile_m=128. Pad to 1920 for TP=1; TP>1
        # needs a wider pad because 1920 / TP is not 128-aligned.
        intermediate_size=1920,
        num_experts=128,
        top_k=6,
        prims_ts_activation="Relu2",
        trtllm_activation="Relu2",
        note=(
            "Nemotron-3.5-Lightning routed ReLU2 experts as a padded shape "
            "proxy: the intermediate size is padded from 1856 to 1920; the "
            "shared expert is excluded."
        ),
    ),
    "nemotron-3-ultra-nvfp4": ModelPreset(
        quant_mode="NvFP4xNvFP4",
        # Latent MoE: routed experts run in the 2048-wide latent space, not
        # the 8192-wide model hidden size.
        hidden_size=2048,
        intermediate_size=5120,
        num_experts=512,
        top_k=22,
        prims_ts_activation="Relu2",
        trtllm_activation="Relu2",
        note=(
            "Nemotron-3-Ultra routed ReLU2 experts in the latent space; the "
            "latent projections and the shared expert are excluded."
        ),
    ),
}


@dataclass(frozen=True)
class ActivationParams:
    """Optional runtime tensors that specialize a model's activation path."""

    has_bias: bool = False
    gemm1_alpha: float | None = None
    gemm1_beta: float | None = None
    gemm1_clamp_limit: float | None = None


def resolve_activation_params(
    model: str | None,
    *,
    has_bias: bool | None = None,
    gemm1_alpha: float | str | None = None,
    gemm1_beta: float | str | None = None,
    gemm1_clamp_limit: float | str | None = None,
) -> ActivationParams:
    """Resolve preset defaults and explicit CLI overrides.

    ``None`` means that an override was not supplied. The string ``"none"``
    explicitly disables a preset's floating-point parameter.
    """

    preset = MODEL_PRESETS.get(model) if model is not None else None

    def resolve_float(
        override: float | str | None, preset_value: float | None
    ) -> float | None:
        if override is None:
            return preset_value
        if isinstance(override, str) and override.lower() == "none":
            return None
        return float(override)

    return ActivationParams(
        has_bias=(
            bool(preset.has_bias)
            if has_bias is None and preset is not None
            else bool(has_bias)
        ),
        gemm1_alpha=resolve_float(
            gemm1_alpha, None if preset is None else preset.gemm1_alpha
        ),
        gemm1_beta=resolve_float(
            gemm1_beta, None if preset is None else preset.gemm1_beta
        ),
        gemm1_clamp_limit=resolve_float(
            gemm1_clamp_limit,
            None if preset is None else preset.gemm1_clamp_limit,
        ),
    )


@dataclass(frozen=True)
class ShapeOrigin:
    """One user-requested point represented by a modeled launch shape."""

    model: str | None
    tp: int
    ep: int
    global_num_tokens: int
    weight: float


@dataclass(frozen=True)
class MoeSweepShape:
    """One synthetic rank-local timing shape and its runtime selection context.

    ``global_num_tokens`` is deliberately distinct from
    ``timing_num_tokens``. The runtime's tactic enumeration sees the global
    hidden-state token count, while isolated EP timing uses
    ``ceil(global_num_tokens / ep)`` rank-local tokens.
    """

    model: str | None
    dtype: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    activation: str
    tp: int
    ep: int
    global_num_tokens: int
    timing_num_tokens: int
    local_num_experts: int
    local_intermediate_size: int
    routing: str
    routing_seed: int
    fc1_n: int
    fc1_k: int
    fc2_n: int
    fc2_k: int
    selected_tile_ns: tuple[int, ...]
    weight: float = 1.0
    origins: tuple[ShapeOrigin, ...] = ()

    def launch_key(self, fc: FcName | None = None) -> "MoeLaunchKey":
        """Return the exact modeled launch-equivalence key."""

        return launch_equivalence_key(self, fc=fc)


@dataclass(frozen=True)
class MoeLaunchKey:
    """Fields that can alter an isolated launch or its sweep applicability."""

    fc: FcName | None
    dtype: str
    timing_num_tokens: int
    local_num_experts: int
    top_k: int
    activation: str
    routing: str
    routing_seed: int
    selected_tile_ns: tuple[int, ...]
    fc1_n: int | None
    fc1_k: int | None
    fc2_n: int | None
    fc2_k: int | None


_DTYPE_ALIASES = {
    "bf16": "BF16",
    "nvfp4": "NvFP4xNvFP4",
    "nvfp4xnvfp4": "NvFP4xNvFP4",
    "mxfp4mxfp8": "MxFP4xMxFP8",
    "mxfp4xmxfp8": "MxFP4xMxFP8",
    "mxfp4bf16": "MxFP4xBf16",
    "mxfp4xbf16": "MxFP4xBf16",
    "fp8pertensor": "Fp8-Per-Tensor",
    "fp8block": "Fp8-Block",
    "mxfp8mxfp8": "MxFP8xMxFP8",
    "mxfp8xmxfp8": "MxFP8xMxFP8",
}

_SUPPORTED_TILES_BY_DTYPE = {
    "BF16": SUPPORTED_BF16_TILE_N,
    "NvFP4xNvFP4": SUPPORTED_NVFP4_TILE_N,
    "MxFP4xMxFP8": SUPPORTED_MXFP4_MXFP8_TILE_N,
    "MxFP4xBf16": SUPPORTED_MXFP4_BF16_TILE_N,
    "Fp8-Per-Tensor": SUPPORTED_FP8_TILE_N,
    "Fp8-Block": SUPPORTED_DSFP8_TILE_N,
    "MxFP8xMxFP8": SUPPORTED_MXFP8_MXFP8_TILE_N,
}

_K_ALIGNMENT_BY_DTYPE = {
    "NvFP4xNvFP4": 64,
    "MxFP4xMxFP8": 128,
    "MxFP4xBf16": 128,
    "MxFP8xMxFP8": 128,
}

_ACTIVATION_NAMES = {
    "gelu": "Gelu",
    "relu": "Relu",
    "silu": "Silu",
    "swiglu": "Swiglu",
    "geglu": "Geglu",
    "swiglubias": "SwigluBias",
    "relu2": "Relu2",
    "swiglustep": "SwigluStep",
    "geglutanh": "GegluTanh",
    "identity": "Identity",
    "situ": "Situ",
}
_GATED_ACTIVATIONS = {
    "Swiglu",
    "Geglu",
    "SwigluBias",
    "SwigluStep",
    "GegluTanh",
    "Situ",
}


def _normalized_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def normalize_dtype(dtype: str) -> str:
    """Normalize a CLI dtype/quant-mode spelling to the runner spelling."""

    try:
        return _DTYPE_ALIASES[_normalized_key(dtype)]
    except KeyError as exc:
        choices = ", ".join(sorted(_SUPPORTED_TILES_BY_DTYPE))
        raise ValueError(
            f"Unsupported dtype {dtype!r}; expected one of: {choices}"
        ) from exc


def normalize_activation(activation: str) -> str:
    """Normalize an activation spelling without importing torch-backed enums."""

    try:
        return _ACTIVATION_NAMES[_normalized_key(activation)]
    except KeyError as exc:
        choices = ", ".join(sorted(_ACTIVATION_NAMES.values()))
        raise ValueError(
            f"Unsupported activation {activation!r}; expected one of: {choices}"
        ) from exc


def _resolve_model_inputs(
    *,
    model: str | None,
    dtype: str | None,
    hidden_size: int | None,
    intermediate_size: int | None,
    num_experts: int | None,
    top_k: int | None,
    activation: str | None,
) -> tuple[str, int, int, int, int, str]:
    explicit = (
        dtype,
        hidden_size,
        intermediate_size,
        num_experts,
        top_k,
        activation,
    )
    if model is not None:
        if model not in MODEL_PRESETS:
            raise ValueError(
                f"Unknown model preset {model!r}; expected one of: "
                + ", ".join(sorted(MODEL_PRESETS))
            )
        if any(value is not None for value in explicit):
            raise ValueError(
                "Model presets cannot be combined with explicit model fields"
            )
        preset = MODEL_PRESETS[model]
        return (
            preset.dtype,
            preset.hidden_size,
            preset.intermediate_size,
            preset.num_experts,
            preset.top_k,
            preset.activation,
        )
    if any(value is None for value in explicit):
        raise ValueError(
            "Without --model, dtype, hidden_size, intermediate_size, "
            "num_experts, top_k, and activation are all required"
        )
    return (
        str(dtype),
        int(hidden_size),
        int(intermediate_size),
        int(num_experts),
        int(top_k),
        str(activation),
    )


def model_moe_shape(
    *,
    global_num_tokens: int,
    tp: int,
    ep: int,
    model: str | None = None,
    dtype: str | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    num_experts: int | None = None,
    top_k: int | None = None,
    activation: str | None = None,
    routing: str = BALANCED_ROUTING,
    routing_seed: int = ROUTING_SEED,
    weight: float = 1.0,
) -> MoeSweepShape:
    """Model one runtime selection point and one rank-local timing point."""

    (
        resolved_dtype,
        resolved_hidden,
        resolved_intermediate,
        resolved_experts,
        resolved_top_k,
        resolved_activation,
    ) = _resolve_model_inputs(
        model=model,
        dtype=dtype,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        activation=activation,
    )
    resolved_dtype = normalize_dtype(resolved_dtype)
    resolved_activation = normalize_activation(resolved_activation)
    _validate_inputs(
        hidden_size=resolved_hidden,
        intermediate_size=resolved_intermediate,
        num_experts=resolved_experts,
        top_k=resolved_top_k,
        global_num_tokens=global_num_tokens,
        tp=tp,
        ep=ep,
        weight=weight,
    )

    local_intermediate = resolved_intermediate // tp
    local_experts = resolved_experts // ep
    timing_tokens = (global_num_tokens + ep - 1) // ep
    gated_factor = 2 if resolved_activation in _GATED_ACTIVATIONS else 1
    fc1_n = gated_factor * local_intermediate
    fc2_n = resolved_hidden
    fc2_k = local_intermediate

    selected_tile_ns = _selected_tile_ns(
        # The real runner routes the global token batch over global experts,
        # while the timed launch uses the corresponding rank-local tensors.
        num_tokens=global_num_tokens,
        top_k=resolved_top_k,
        num_local_experts=local_experts,
        num_experts=resolved_experts,
        supported_tiles=_SUPPORTED_TILES_BY_DTYPE[resolved_dtype],
    )
    origin = ShapeOrigin(
        model=model,
        tp=tp,
        ep=ep,
        global_num_tokens=global_num_tokens,
        weight=float(weight),
    )
    return MoeSweepShape(
        model=model,
        dtype=resolved_dtype,
        hidden_size=resolved_hidden,
        intermediate_size=resolved_intermediate,
        num_experts=resolved_experts,
        top_k=resolved_top_k,
        activation=resolved_activation,
        tp=tp,
        ep=ep,
        global_num_tokens=global_num_tokens,
        timing_num_tokens=timing_tokens,
        local_num_experts=local_experts,
        local_intermediate_size=local_intermediate,
        routing=str(routing),
        routing_seed=int(routing_seed),
        fc1_n=fc1_n,
        fc1_k=resolved_hidden,
        fc2_n=fc2_n,
        fc2_k=fc2_k,
        selected_tile_ns=selected_tile_ns,
        weight=float(weight),
        origins=(origin,),
    )


def _validate_inputs(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    global_num_tokens: int,
    tp: int,
    ep: int,
    weight: float,
) -> None:
    for name, value in (
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
        ("num_experts", num_experts),
        ("global_num_tokens", global_num_tokens),
        ("tp", tp),
        ("ep", ep),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if top_k <= 0 or top_k > num_experts:
        raise ValueError(f"top_k must be in [1, num_experts], got {top_k}")
    if intermediate_size % tp:
        raise ValueError(
            f"intermediate_size={intermediate_size} must be divisible by TP={tp}"
        )
    if num_experts % ep:
        raise ValueError(f"num_experts={num_experts} must be divisible by EP={ep}")
    local_num_experts = num_experts // ep
    if top_k > local_num_experts:
        raise ValueError(
            f"top_k={top_k} cannot exceed rank-local experts={local_num_experts}"
        )
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"weight must be finite and positive, got {weight}")


def build_sweep_shapes(
    *,
    global_num_tokens: Iterable[int],
    tp: Iterable[int] = (),
    ep: Iterable[int] = (),
    tp_ep: Iterable[tuple[int, int]] | None = None,
    model: str | None = None,
    dtype: str | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    num_experts: int | None = None,
    top_k: int | None = None,
    activation: str | None = None,
    routing: str = BALANCED_ROUTING,
    routing_seed: int = ROUTING_SEED,
    weights: Mapping[int, float] | None = None,
    merge: bool = True,
    fc: FcName | None = None,
) -> tuple[MoeSweepShape, ...]:
    """Build requested parallelism points x tokens, optionally merging duplicates."""

    token_values = tuple(global_num_tokens)
    if tp_ep is None:
        parallelism = tuple(
            (int(tp_size), int(ep_size)) for tp_size in tp for ep_size in ep
        )
    else:
        parallelism = tuple((int(tp_size), int(ep_size)) for tp_size, ep_size in tp_ep)
    if not parallelism:
        raise ValueError("at least one TP/EP configuration is required")
    shapes = tuple(
        model_moe_shape(
            global_num_tokens=tokens,
            tp=tp_size,
            ep=ep_size,
            model=model,
            dtype=dtype,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            activation=activation,
            routing=routing,
            routing_seed=routing_seed,
            weight=1.0 if weights is None else weights.get(tokens, 1.0),
        )
        for tp_size, ep_size in parallelism
        for tokens in token_values
    )
    for shape in shapes:
        alignment = _K_ALIGNMENT_BY_DTYPE.get(shape.dtype)
        if alignment is None:
            continue
        fcs = (fc,) if fc is not None else ("fc1", "fc2")
        for shape_fc in fcs:
            problem_k = int(getattr(shape, f"{shape_fc}_k"))
            if problem_k % alignment:
                raise ValueError(
                    f"{shape.model or 'custom model'} TP={shape.tp}/EP={shape.ep} "
                    f"{shape_fc} K={problem_k} for dtype={shape.dtype} must be "
                    f"a multiple of {alignment}; use an explicitly padded model "
                    "preset"
                )
    return merge_duplicate_shapes(shapes, fc=fc) if merge else shapes


def default_global_num_tokens(max_num_tokens: int = 8192) -> tuple[int, ...]:
    """Return the standard hybrid MoE token buckets."""

    from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

    return get_hybrid_num_tokens_buckets(max_num_tokens)


def launch_equivalence_key(
    shape: MoeSweepShape, *, fc: FcName | None = None
) -> MoeLaunchKey:
    """Return a key containing every field that affects the modeled launch.

    Raw model, TP, EP, and global-token labels are intentionally absent. Their
    launch effects are represented by the derived fields. The selected tile-N
    set remains in the key because it controls which sweep groups time a shape.
    """

    if fc not in (None, "fc1", "fc2"):
        raise ValueError(f"fc must be 'fc1', 'fc2', or None, got {fc!r}")
    return MoeLaunchKey(
        fc=fc,
        dtype=shape.dtype,
        timing_num_tokens=shape.timing_num_tokens,
        local_num_experts=shape.local_num_experts,
        top_k=shape.top_k,
        activation=shape.activation,
        routing=shape.routing,
        routing_seed=shape.routing_seed,
        selected_tile_ns=shape.selected_tile_ns,
        fc1_n=shape.fc1_n if fc in (None, "fc1") else None,
        fc1_k=shape.fc1_k if fc in (None, "fc1") else None,
        fc2_n=shape.fc2_n if fc in (None, "fc2") else None,
        fc2_k=shape.fc2_k if fc in (None, "fc2") else None,
    )


def merge_duplicate_shapes(
    shapes: Iterable[MoeSweepShape], *, fc: FcName | None = None
) -> tuple[MoeSweepShape, ...]:
    """Merge launch-equivalent shapes, adding weights without rounding."""

    grouped: dict[MoeLaunchKey, list[MoeSweepShape]] = {}
    for shape in shapes:
        assert_modeled_shape_consistent(shape)
        grouped.setdefault(launch_equivalence_key(shape, fc=fc), []).append(shape)

    merged = []
    for duplicates in grouped.values():
        origins = tuple(
            origin
            for shape in duplicates
            for origin in (
                shape.origins
                or (
                    ShapeOrigin(
                        model=shape.model,
                        tp=shape.tp,
                        ep=shape.ep,
                        global_num_tokens=shape.global_num_tokens,
                        weight=shape.weight,
                    ),
                )
            )
        )
        merged.append(
            replace(
                duplicates[0],
                weight=math.fsum(shape.weight for shape in duplicates),
                origins=origins,
            )
        )
    return tuple(merged)


def modeled_shape_consistency_errors(shape: MoeSweepShape) -> tuple[str, ...]:
    """Return host-safe consistency errors for a modeled shape."""

    errors: list[str] = []
    try:
        expected = model_moe_shape(
            model=shape.model,
            dtype=None if shape.model is not None else shape.dtype,
            hidden_size=None if shape.model is not None else shape.hidden_size,
            intermediate_size=(
                None if shape.model is not None else shape.intermediate_size
            ),
            num_experts=None if shape.model is not None else shape.num_experts,
            top_k=None if shape.model is not None else shape.top_k,
            activation=None if shape.model is not None else shape.activation,
            global_num_tokens=shape.global_num_tokens,
            tp=shape.tp,
            ep=shape.ep,
            routing=shape.routing,
            routing_seed=shape.routing_seed,
            weight=shape.weight,
        )
    except (TypeError, ValueError) as exc:
        return (str(exc),)

    derived_fields = (
        "dtype",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "activation",
        "timing_num_tokens",
        "local_num_experts",
        "local_intermediate_size",
        "routing",
        "routing_seed",
        "fc1_n",
        "fc1_k",
        "fc2_n",
        "fc2_k",
        "selected_tile_ns",
    )
    for field in derived_fields:
        actual_value = getattr(shape, field)
        expected_value = getattr(expected, field)
        if actual_value != expected_value:
            errors.append(f"{field}={actual_value!r}, expected {expected_value!r}")
    if not math.isfinite(shape.weight) or shape.weight <= 0:
        errors.append(f"weight must be finite and positive, got {shape.weight}")
    if shape.origins:
        origin_weight = math.fsum(origin.weight for origin in shape.origins)
        if not math.isclose(origin_weight, shape.weight, rel_tol=1e-12, abs_tol=0.0):
            errors.append(
                f"origin weights sum to {origin_weight}, expected {shape.weight}"
            )
    return tuple(errors)


def assert_modeled_shape_consistent(shape: MoeSweepShape) -> None:
    """Raise if a shape disagrees with the pure host-side model."""

    errors = modeled_shape_consistency_errors(shape)
    if errors:
        raise ValueError("Inconsistent modeled MoE shape: " + "; ".join(errors))


__all__ = [
    "ActivationParams",
    "BALANCED_ROUTING",
    "MODEL_PRESETS",
    "FcName",
    "ModelPreset",
    "MoeLaunchKey",
    "MoeSweepShape",
    "ShapeOrigin",
    "assert_modeled_shape_consistent",
    "build_sweep_shapes",
    "default_global_num_tokens",
    "launch_equivalence_key",
    "merge_duplicate_shapes",
    "model_moe_shape",
    "modeled_shape_consistency_errors",
    "normalize_activation",
    "normalize_dtype",
    "resolve_activation_params",
]
