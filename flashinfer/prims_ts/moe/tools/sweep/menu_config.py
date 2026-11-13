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

"""Host-only configuration and shape construction for MoE menu re-timing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .shapes import (
    MODEL_PRESETS,
    MoeSweepShape,
    build_sweep_shapes,
    merge_duplicate_shapes,
    normalize_activation,
    normalize_dtype,
)

SCHEMA_VERSION = 1


def activation_class(activation: str) -> str:
    """Return the menu class shared by activations with the same geometry."""

    normalized = normalize_activation(activation)
    if normalized in ("Swiglu", "Geglu", "Situ"):
        return "gated"
    if normalized in ("Relu2", "Silu"):
        return "elementwise"
    raise ValueError(f"unsupported activation for menu selection: {activation!r}")


def runtime_twin(
    options: Mapping[str, object],
) -> dict[str, object] | None:
    """Return the scheduler-policy variant added by the runtime, if any."""

    if str(options.get("tile_scheduler", "static")).lower() != "persistent":
        return None
    if bool(options.get("use_clc_fast_drain", False)) and bool(
        options.get("use_work_throttle", False)
    ):
        return None
    return {
        **options,
        "use_clc_fast_drain": True,
        "use_work_throttle": True,
    }


@dataclass(frozen=True)
class AutotuneCost:
    seconds_per_pair: float = 0.0
    seconds_per_compile: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("seconds_per_pair", self.seconds_per_pair),
            ("seconds_per_compile", self.seconds_per_compile),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")


@dataclass(frozen=True)
class ParallelismTarget:
    tp: int
    ep: int
    tokens: tuple[int, ...]
    weight: float = 1.0
    token_weights: tuple[float, ...] | None = None
    tune_max_num_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.tp <= 0 or self.ep <= 0:
            raise ValueError("TP and EP must be positive")
        if not self.tokens or any(token <= 0 for token in self.tokens):
            raise ValueError("tokens must be a non-empty list of positive integers")
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("tokens must not contain duplicates")
        if self.tune_max_num_tokens is not None and self.tune_max_num_tokens < max(
            self.tokens
        ):
            raise ValueError(
                "tune_max_num_tokens must be at least the largest target token count"
            )
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("TP:EP weight must be positive and finite")
        if self.token_weights is not None:
            if len(self.token_weights) != len(self.tokens):
                raise ValueError("token_weights must have the same length as tokens")
            if any(not math.isfinite(v) or v <= 0 for v in self.token_weights):
                raise ValueError("token_weights must be positive and finite")


@dataclass(frozen=True)
class DeploymentTarget:
    name: str
    model: str | None
    dtype: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    activation: str
    parallelism: tuple[ParallelismTarget, ...]
    weight: float = 1.0
    has_bias: bool = False
    gemm1_alpha: float | None = None
    gemm1_beta: float | None = None
    gemm1_clamp_limit: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("deployment name must not be empty")
        if (
            min(
                self.hidden_size,
                self.intermediate_size,
                self.num_experts,
                self.top_k,
            )
            <= 0
        ):
            raise ValueError("model dimensions, experts, and top_k must be positive")
        if not self.parallelism:
            raise ValueError("deployment must contain at least one TP:EP target")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("deployment weight must be positive and finite")
        for name, value in (
            ("gemm1_alpha", self.gemm1_alpha),
            ("gemm1_beta", self.gemm1_beta),
            ("gemm1_clamp_limit", self.gemm1_clamp_limit),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class MenuTarget:
    sm_arch: int
    dtype: str
    routings: tuple[str, ...]
    deployments: tuple[DeploymentTarget, ...]
    autotune_cost: AutotuneCost = AutotuneCost()
    routing_seed: int = 42

    def __post_init__(self) -> None:
        if self.sm_arch <= 0:
            raise ValueError("sm_arch must be positive")
        if not self.routings or any(not routing for routing in self.routings):
            raise ValueError("routings must be non-empty strings")
        if len(set(self.routings)) != len(self.routings):
            raise ValueError("routings must not contain duplicates")
        if not self.deployments:
            raise ValueError("deployments must not be empty")
        mismatches = [
            deployment.name
            for deployment in self.deployments
            if deployment.dtype != self.dtype
        ]
        if mismatches:
            raise ValueError(
                f"all deployments must use dtype={self.dtype}: {mismatches}"
            )


def _positive_float(value: object, *, field: str, default: float = 1.0) -> float:
    result = default if value is None else float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be positive and finite")
    return result


def _parse_tp_ep(value: object) -> tuple[int, int]:
    if isinstance(value, str):
        pieces = value.split(":")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        pieces = list(value)
    else:
        raise ValueError(f"tp_ep must be 'TP:EP' or [TP, EP], got {value!r}")
    if len(pieces) != 2:
        raise ValueError(f"tp_ep must contain exactly two values, got {value!r}")
    tp, ep = (int(piece) for piece in pieces)
    if tp <= 0 or ep <= 0:
        raise ValueError(f"tp_ep values must be positive, got {value!r}")
    return tp, ep


def _parse_deployment(value: Mapping[str, Any], target_dtype: str) -> DeploymentTarget:
    model_value = value.get("model")
    model = None if model_value is None else str(model_value)
    if model is not None:
        try:
            preset = MODEL_PRESETS[model]
        except KeyError as exc:
            raise ValueError(f"unknown model preset {model!r}") from exc
        dtype = normalize_dtype(preset.dtype)
        hidden_size = preset.hidden_size
        intermediate_size = preset.intermediate_size
        num_experts = preset.num_experts
        top_k = preset.top_k
        activation = normalize_activation(preset.activation)
        has_bias = preset.has_bias
        gemm1_alpha = preset.gemm1_alpha
        gemm1_beta = preset.gemm1_beta
        gemm1_clamp_limit = preset.gemm1_clamp_limit
        name = str(value.get("name", model))
    else:
        required = (
            "name",
            "hidden",
            "intermediate",
            "num_experts",
            "top_k",
            "activation",
        )
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(f"custom deployment is missing fields: {missing}")
        name = str(value["name"])
        dtype = normalize_dtype(str(value.get("dtype", target_dtype)))
        hidden_size = int(value["hidden"])
        intermediate_size = int(value["intermediate"])
        num_experts = int(value["num_experts"])
        top_k = int(value["top_k"])
        activation = normalize_activation(str(value["activation"]))
        has_bias = bool(value.get("bias", False))
        gemm1_alpha = (
            None if value.get("gemm1_alpha") is None else float(value["gemm1_alpha"])
        )
        gemm1_beta = (
            None if value.get("gemm1_beta") is None else float(value["gemm1_beta"])
        )
        gemm1_clamp_limit = (
            None
            if value.get("gemm1_clamp_limit") is None
            else float(value["gemm1_clamp_limit"])
        )

    parallelism = []
    for index, point_value in enumerate(value.get("tp_ep", ())):
        if not isinstance(point_value, Mapping):
            raise ValueError(f"tp_ep[{index}] must be an object")
        tp, ep = _parse_tp_ep(point_value.get("tp_ep"))
        tokens = tuple(int(token) for token in point_value.get("tokens", ()))
        raw_token_weights = point_value.get("token_weights")
        token_weights = (
            None
            if raw_token_weights is None
            else tuple(float(weight) for weight in raw_token_weights)
        )
        parallelism.append(
            ParallelismTarget(
                tp=tp,
                ep=ep,
                tokens=tokens,
                weight=_positive_float(
                    point_value.get("weight"), field=f"{name}.tp_ep[{index}].weight"
                ),
                token_weights=token_weights,
                tune_max_num_tokens=(
                    None
                    if point_value.get("tune_max_num_tokens") is None
                    else int(point_value["tune_max_num_tokens"])
                ),
            )
        )
    return DeploymentTarget(
        name=name,
        model=model,
        dtype=dtype,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        activation=activation,
        parallelism=tuple(parallelism),
        weight=_positive_float(value.get("weight"), field=f"{name}.weight"),
        has_bias=has_bias,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def load_menu_target(path: str | Path) -> MenuTarget:
    """Load and validate a checked-in menu target JSON file."""

    source = Path(path)
    with source.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, Mapping):
        raise ValueError("menu target must be a JSON object")
    version = int(raw.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported menu target schema_version={version}; "
            f"expected {SCHEMA_VERSION}"
        )
    dtype = normalize_dtype(str(raw["dtype"]))
    cost_value = raw.get("autotune_cost", {})
    if not isinstance(cost_value, Mapping):
        raise ValueError("autotune_cost must be an object")
    deployments_value = raw.get("deployments", ())
    if not isinstance(deployments_value, Sequence) or isinstance(
        deployments_value, (str, bytes)
    ):
        raise ValueError("deployments must be a list")
    deployments = tuple(
        _parse_deployment(value, dtype)
        for value in deployments_value
        if isinstance(value, Mapping)
    )
    if len(deployments) != len(deployments_value):
        raise ValueError("every deployment must be an object")
    return MenuTarget(
        sm_arch=int(raw["sm_arch"]),
        dtype=dtype,
        routings=tuple(str(value) for value in raw.get("routings", ())),
        deployments=deployments,
        autotune_cost=AutotuneCost(
            seconds_per_pair=float(cost_value.get("seconds_per_pair", 0.0)),
            seconds_per_compile=float(cost_value.get("seconds_per_compile", 0.0)),
        ),
        routing_seed=int(raw.get("routing_seed", 42)),
    )


def build_deployment_shapes(
    target: MenuTarget,
    deployment: DeploymentTarget,
    *,
    fc: str | None = None,
) -> tuple[MoeSweepShape, ...]:
    """Build weighted shapes for one deployment."""

    shapes: list[MoeSweepShape] = []
    routing_share = 1.0 / len(target.routings)
    point_weight_sum = math.fsum(point.weight for point in deployment.parallelism)
    for point in deployment.parallelism:
        raw_token_weights = point.token_weights or (1.0,) * len(point.tokens)
        token_weight_sum = math.fsum(raw_token_weights)
        weights = {
            token: deployment.weight
            * point.weight
            / point_weight_sum
            * token_weight
            / token_weight_sum
            * routing_share
            for token, token_weight in zip(point.tokens, raw_token_weights, strict=True)
        }
        for routing in target.routings:
            kwargs: dict[str, object] = {
                "global_num_tokens": point.tokens,
                "tp_ep": ((point.tp, point.ep),),
                "routing": routing,
                "routing_seed": target.routing_seed,
                "weights": weights,
                "merge": False,
                "fc": fc,
            }
            if deployment.model is not None:
                kwargs["model"] = deployment.model
            else:
                kwargs.update(
                    dtype=deployment.dtype,
                    hidden_size=deployment.hidden_size,
                    intermediate_size=deployment.intermediate_size,
                    num_experts=deployment.num_experts,
                    top_k=deployment.top_k,
                    activation=deployment.activation,
                )
            shapes.extend(build_sweep_shapes(**kwargs))
    return merge_duplicate_shapes(shapes, fc=fc)


def build_menu_shapes(
    target: MenuTarget, *, fc: str | None = None
) -> tuple[MoeSweepShape, ...]:
    """Build all target shapes, merging launch-equivalent entries."""

    return merge_duplicate_shapes(
        (
            shape
            for deployment in target.deployments
            for shape in build_deployment_shapes(target, deployment, fc=fc)
        ),
        fc=fc,
    )
