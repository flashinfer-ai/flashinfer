# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph-safe lazy frontend boundary for MXFP8 x MXFP4 MegaMoE.

Integration compiles a :class:`KernelLaunchSpec` and binds a zero-argument
launcher. Compilation and binding caches preserve workspace ownership and
allow stream rebinding without rebuilding tensor adapters during capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Mapping, Protocol

import torch

from .workspace import MXFP8_SF_VEC_SIZE


@dataclass(frozen=True)
class FrontendConfig:
    hidden: int
    intermediate: int
    topk: int
    num_experts: int
    max_tokens: int
    tactic: Hashable

    def __post_init__(self) -> None:
        for name in ("hidden", "intermediate"):
            value = getattr(self, name)
            if value <= 0 or value % MXFP8_SF_VEC_SIZE:
                raise ValueError(f"{name} must be a positive multiple of 32")
        for name in ("topk", "num_experts", "max_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class KernelLaunchSpec:
    """All runtime state needed to bind one persistent-kernel launch."""

    activation: torch.Tensor
    activation_scales: torch.Tensor
    fc1_weight: torch.Tensor
    fc1_weight_scales: torch.Tensor
    fc2_weight: torch.Tensor
    fc2_weight_scales: torch.Tensor
    route_ids: torch.Tensor
    route_scores: torch.Tensor
    output: torch.Tensor
    workspace_regions: Mapping[str, torch.Tensor]
    num_tokens: int
    stream: int
    static_tactic: Hashable


class KernelAdapter(Protocol):
    """Compile and bind the persistent kernel for a launch specification."""

    def compile(self, config: FrontendConfig, spec: KernelLaunchSpec) -> Any: ...

    def bind(
        self, compiled: Any, config: FrontendConfig, spec: KernelLaunchSpec
    ) -> Callable[[], None]: ...


class StreamRebindingKernelAdapter(KernelAdapter, Protocol):
    """Optional adapter capability for graph-safe stream-only rebinding.

    This must reuse all eagerly prepared tensor views and kernel arguments.
    It may only change host-side stream handles: no device allocation,
    compilation, DLPack conversion, or synchronization is allowed.
    """

    def rebind_stream(
        self, launch: Callable[[], None], stream: int
    ) -> Callable[[], None]: ...


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def _static_tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def launch_cache_key(spec: KernelLaunchSpec) -> tuple:
    """Pointer/shape/stride/stream identity for a bound graph-safe launch."""

    operands = (
        spec.activation,
        spec.activation_scales,
        spec.fc1_weight,
        spec.fc1_weight_scales,
        spec.fc2_weight,
        spec.fc2_weight_scales,
        spec.route_ids,
        spec.route_scores,
        spec.output,
    )
    regions = tuple(
        (name, _tensor_signature(tensor))
        for name, tensor in sorted(spec.workspace_regions.items())
    )
    return (
        tuple(_tensor_signature(tensor) for tensor in operands),
        regions,
        spec.num_tokens,
        spec.stream,
        spec.static_tactic,
    )


def compile_cache_key(config: FrontendConfig, spec: KernelLaunchSpec) -> tuple:
    """Static signature controlling CuTeDSL compilation."""

    return (
        config,
        tuple(
            _static_tensor_signature(tensor)
            for tensor in (
                spec.activation,
                spec.activation_scales,
                spec.fc1_weight,
                spec.fc1_weight_scales,
                spec.fc2_weight,
                spec.fc2_weight_scales,
                spec.route_ids,
                spec.route_scores,
                spec.output,
            )
        ),
        tuple(
            (name, _static_tensor_signature(tensor))
            for name, tensor in sorted(spec.workspace_regions.items())
        ),
        spec.static_tactic,
    )


def validate_launch_spec(config: FrontendConfig, spec: KernelLaunchSpec) -> None:
    """Validate the stable frontend contract before compile or launch bind."""

    if spec.static_tactic != config.tactic:
        raise ValueError("launch static_tactic does not match frontend config")
    if not 0 <= spec.num_tokens <= config.max_tokens:
        raise ValueError(
            f"num_tokens must be in [0, {config.max_tokens}], got {spec.num_tokens}"
        )
    tensors = {
        "activation": spec.activation,
        "activation_scales": spec.activation_scales,
        "fc1_weight": spec.fc1_weight,
        "fc1_weight_scales": spec.fc1_weight_scales,
        "fc2_weight": spec.fc2_weight,
        "fc2_weight_scales": spec.fc2_weight_scales,
        "route_ids": spec.route_ids,
        "route_scores": spec.route_scores,
        "output": spec.output,
        **{
            f"workspace[{name!r}]": value
            for name, value in spec.workspace_regions.items()
        },
    }
    if not spec.workspace_regions:
        raise ValueError("workspace_regions must not be empty")
    device = spec.activation.device
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")

    packed_fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    if spec.fc1_weight.dtype not in (packed_fp4_dtype, torch.uint8):
        raise ValueError("fc1_weight must contain packed MXFP4 E2M1 values")
    if spec.fc2_weight.dtype not in (packed_fp4_dtype, torch.uint8):
        raise ValueError("fc2_weight must contain packed MXFP4 E2M1 values")
    for name, tensor in (
        ("fc1_weight_scales", spec.fc1_weight_scales),
        ("fc2_weight_scales", spec.fc2_weight_scales),
    ):
        if tensor.dtype not in (torch.float8_e8m0fnu, torch.uint8):
            raise ValueError(f"{name} must contain raw E8M0 values")
    for name, tensor in (
        ("activation", spec.activation),
        ("activation_scales", spec.activation_scales),
        ("fc1_weight_scales", spec.fc1_weight_scales),
        ("fc2_weight_scales", spec.fc2_weight_scales),
        ("route_ids", spec.route_ids),
        ("route_scores", spec.route_scores),
        ("output", spec.output),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    sf_cols = config.hidden // MXFP8_SF_VEC_SIZE
    if spec.activation.dtype != torch.float8_e4m3fn or spec.activation.shape != (
        config.max_tokens,
        config.hidden,
    ):
        raise ValueError("activation must be E4M3 [max_tokens, hidden]")
    if spec.activation_scales.ndim != 2:
        raise ValueError("activation_scales must be 2D")
    if spec.activation_scales.dtype != torch.float8_e8m0fnu:
        raise ValueError("activation_scales must contain raw E8M0 values")
    if (
        spec.activation_scales.shape[0] != config.max_tokens
        or spec.activation_scales.shape[1] < sf_cols
    ):
        raise ValueError("activation_scales must be [max_tokens, >= hidden/32]")
    if spec.route_ids.dtype != torch.int64 or spec.route_ids.shape != (
        config.max_tokens,
        config.topk,
    ):
        raise ValueError("route_ids must be int64 [max_tokens, topk]")
    if (
        spec.route_scores.dtype != torch.float32
        or spec.route_scores.shape != spec.route_ids.shape
    ):
        raise ValueError("route_scores must be float32 [max_tokens, topk]")
    if (
        spec.output.ndim != 2
        or spec.output.dtype != torch.bfloat16
        or not spec.num_tokens <= spec.output.shape[0] <= config.max_tokens
        or spec.output.shape[1] != config.hidden
    ):
        raise ValueError(
            "output must be BF16 [rows, hidden] with num_tokens <= rows <= max_tokens"
        )


def _ensure_not_capturing(what: str) -> None:
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{what} cannot run during CUDA graph capture; call warmup() "
            "with these exact buffers and stream before capture"
        )


@dataclass
class Mxfp8Mxfp4Frontend:
    """One-compiled-kernel frontend with graph-safe bound-launch caching."""

    config: FrontendConfig
    adapter: KernelAdapter
    _compiled: Any = field(default=None, init=False, repr=False)
    _compile_key: tuple | None = field(default=None, init=False, repr=False)
    _launch_key: tuple | None = field(default=None, init=False, repr=False)
    _launch: Callable[[], None] | None = field(default=None, init=False, repr=False)

    def warmup(self, spec: KernelLaunchSpec) -> None:
        """Compile and bind eagerly. Required before CUDA graph capture."""

        self._prepare(spec)

    def run(self, spec: KernelLaunchSpec) -> torch.Tensor:
        """Launch the kernel and return the live output view."""

        self._prepare(spec)
        assert self._launch is not None
        self._launch()
        return spec.output[: spec.num_tokens]

    def invalidate(self) -> None:
        """Drop host cache entries. Does not free adapter-owned resources."""

        _ensure_not_capturing("frontend cache invalidation")
        self._compiled = None
        self._compile_key = None
        self._launch_key = None
        self._launch = None

    def _prepare(self, spec: KernelLaunchSpec) -> None:
        compile_key = compile_cache_key(self.config, spec)
        launch_key = launch_cache_key(spec)
        if (
            self._compiled is not None
            and self._compile_key == compile_key
            and self._launch_key == launch_key
        ):
            return
        # Benchmark helpers may warm up on one side stream and capture on
        # another. Only an explicitly capture-safe adapter may reuse that
        # binding. Ordinary bind() can construct DLPack views or allocate.
        rebind_stream = getattr(self.adapter, "rebind_stream", None)
        if (
            self._compiled is not None
            and self._compile_key == compile_key
            and self._launch_key is not None
            and self._launch is not None
            and self._launch_key[:3] == launch_key[:3]
            and self._launch_key[4:] == launch_key[4:]
            and callable(rebind_stream)
        ):
            self._launch = rebind_stream(self._launch, spec.stream)
            self._launch_key = launch_key
            return
        _ensure_not_capturing("MXFP8 x MXFP4 frontend compile/bind cache miss")
        validate_launch_spec(self.config, spec)
        if self._compiled is None or self._compile_key != compile_key:
            self._compiled = self.adapter.compile(self.config, spec)
            self._compile_key = compile_key
        self._launch = self.adapter.bind(self._compiled, self.config, spec)
        self._launch_key = launch_key


__all__ = [
    "FrontendConfig",
    "KernelAdapter",
    "KernelLaunchSpec",
    "Mxfp8Mxfp4Frontend",
    "StreamRebindingKernelAdapter",
    "compile_cache_key",
    "launch_cache_key",
    "validate_launch_spec",
]
