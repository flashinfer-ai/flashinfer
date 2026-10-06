"""Planning, binding, and validation for HyperConnection primitives."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from b12x._lib.scratch import ScratchBufferSpec
from b12x.preparation import Plan
from ._tuning import HyperConnectionConfig, HyperConnectionQuery

_MAX_TRITON_REDUCTION_WIDTH = 65_536


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


@dataclass(frozen=True, kw_only=True)
class HyperConnectionCaps:
    """Token capacity and static multi-stream HyperConnection geometry."""

    device: torch.device | str
    max_tokens: int
    hidden_size: int
    streams: int = 4
    lowrank: int = 320
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = _canonical_device(self.device)
        max_tokens = int(self.max_tokens)
        hidden_size = int(self.hidden_size)
        streams = int(self.streams)
        lowrank = int(self.lowrank)
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if hidden_size <= 0 or hidden_size > _MAX_TRITON_REDUCTION_WIDTH:
            raise ValueError(f"hidden_size must be in [1, 65536], got {hidden_size}")
        if streams <= 0:
            raise ValueError(f"streams must be positive, got {streams}")
        if lowrank <= 0:
            raise ValueError(f"lowrank must be positive, got {lowrank}")
        if self.dtype != torch.bfloat16:
            raise TypeError(
                f"HyperConnection kernels require torch.bfloat16, got {self.dtype}"
            )
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "streams", streams)
        object.__setattr__(self, "lowrank", lowrank)


@dataclass(frozen=True)
class HyperConnectionBinding:
    """Caller-owned output capacity with token-limited accessors.

    Mutating custom ops target the capacity tensors directly. The accessors
    expose only the live token prefix to downstream operators.
    """

    state: "_HyperConnectionState"
    tokens: int
    normalized_capacity: torch.Tensor
    bottleneck_capacity: torch.Tensor
    block_input_capacity: torch.Tensor
    plan: Plan | None = None

    @property
    def normalized(self) -> torch.Tensor:
        return self.normalized_capacity[: self.tokens]

    @property
    def bottleneck(self) -> torch.Tensor:
        return self.bottleneck_capacity[: self.tokens]

    @property
    def block_input(self) -> torch.Tensor:
        return self.block_input_capacity[: self.tokens]


@dataclass(frozen=True)
class _HyperConnectionState:
    """One immutable, already-lowered native operation and its launchers."""

    caps: HyperConnectionCaps
    query: HyperConnectionQuery
    config: HyperConnectionConfig
    launch: Callable

    def require_operation(self, operation, *, eps=None):
        if self.query.operation != operation:
            raise ValueError(f"execution prepares {self.query.operation}, not {operation}")
        if eps is not None and float(eps) != self.query.eps:
            raise ValueError("normalization epsilon differs from prepared invocation")

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        """HyperConnection primitives use no anonymous scratch allocation."""
        return ()

    def output_shapes(self, tokens: int | None = None) -> dict[str, tuple[int, ...]]:
        live_tokens = self._live_tokens(tokens)
        caps = self.caps
        width = caps.streams * caps.hidden_size
        return {
            "normalized": (live_tokens, width),
            "bottleneck": (live_tokens, caps.lowrank),
            "block_input": (live_tokens, caps.hidden_size),
        }

    def _live_tokens(self, tokens: int | None) -> int:
        live_tokens = self.caps.max_tokens if tokens is None else int(tokens)
        if live_tokens < 0 or live_tokens > self.caps.max_tokens:
            raise ValueError(
                f"tokens={live_tokens} exceeds capacity {self.caps.max_tokens}"
            )
        return live_tokens

    def bind(
        self,
        *,
        normalized: torch.Tensor,
        bottleneck: torch.Tensor,
        block_input: torch.Tensor,
        tokens: int | None = None,
        plan: Plan | None = None,
    ) -> HyperConnectionBinding:
        live_tokens = self._live_tokens(tokens)
        caps = self.caps
        width = caps.streams * caps.hidden_size
        capacity_outputs = {
            "normalized": normalized,
            "bottleneck": bottleneck,
            "block_input": block_input,
        }
        tails = {
            "normalized": (width,),
            "bottleneck": (caps.lowrank,),
            "block_input": (caps.hidden_size,),
        }
        for name, tensor in capacity_outputs.items():
            _validate_capacity(
                tensor,
                tokens=live_tokens,
                tail=tails[name],
                dtype=caps.dtype,
                device=caps.device,
                name=name,
            )
        # Dynamo cannot trace storage/data-pointer inspection. Callers that bind
        # fixed workspaces inside a compiled forward must validate the same
        # buffers once before compilation.
        if not torch.compiler.is_compiling():
            for other_name in ("bottleneck", "block_input"):
                if _overlaps(normalized, capacity_outputs[other_name]):
                    raise ValueError(
                        f"normalized and {other_name} outputs must not overlap"
                    )
        return HyperConnectionBinding(
            state=self,
            plan=plan,
            tokens=live_tokens,
            normalized_capacity=normalized,
            bottleneck_capacity=bottleneck,
            block_input_capacity=block_input,
        )




def _validate_capacity(
    tensor: torch.Tensor,
    *,
    tokens: int,
    tail: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> None:
    if tensor.ndim != len(tail) + 1 or tuple(tensor.shape[1:]) != tail:
        raise ValueError(
            f"{name} must have tail shape {tail}, got {tuple(tensor.shape)}"
        )
    if int(tensor.shape[0]) < tokens:
        raise ValueError(
            f"{name} capacity {int(tensor.shape[0])} is smaller than tokens={tokens}"
        )
    if tensor.dtype != dtype or tensor.device != device:
        raise ValueError(
            f"{name} must use dtype={dtype} and device={device}; "
            f"got dtype={tensor.dtype}, device={tensor.device}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    span = (
        0
        if tensor.numel() == 0
        else 1
        + sum(
            (int(size) - 1) * int(stride)
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
    )
    return start, start + span * int(tensor.element_size())


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


def _validate_input(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    caps: HyperConnectionCaps,
    name: str,
    row_strided: bool = False,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != caps.dtype or tensor.device != caps.device:
        raise ValueError(
            f"{name} must use dtype={caps.dtype} and device={caps.device}; "
            f"got dtype={tensor.dtype}, device={tensor.device}"
        )
    if row_strided:
        if tensor.stride(1) != 1 or tensor.stride(0) < tensor.shape[1]:
            raise ValueError(f"{name} requires unit column stride and disjoint rows")
    elif not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_output_disjoint(
    output_name: str,
    output: torch.Tensor,
    inputs: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    # Dynamo cannot trace storage/data-pointer inspection. The launch remains
    # opaque through the mutating custom op; eager warmup performs this alias
    # check before serving compilation.
    if torch.compiler.is_compiling():
        return
    for input_name, tensor in inputs:
        if _overlaps(output, tensor):
            raise ValueError(f"{output_name} must not overlap {input_name}")


def _require_cuda(plan: _HyperConnectionState) -> None:
    if plan.caps.device.type != "cuda":
        raise ValueError(
            "HyperConnection GPU entry points require CUDA; use the "
            "explicit reference module for an oracle"
        )


def _require_eps(eps: float) -> float:
    value = float(eps)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {value}")
    return value


def run_grouped_rmsnorm_impl(state, weight, *, eps, plan: _HyperConnectionState, out, zero_centered=True):
    _require_cuda(plan)
    plan.require_operation("grouped_rmsnorm", eps=eps)
    if bool(zero_centered) != plan.query.zero_centered:
        raise ValueError("normalization recipe differs from preparation")
    caps = plan.caps
    tokens = plan._live_tokens(state.shape[0])
    width = caps.streams * caps.hidden_size
    _validate_input(state, shape=(tokens, width), caps=caps, name="state")
    if weight.dtype != getattr(torch, plan.query.weight_dtype):
        raise ValueError("affine weight dtype differs from preparation")
    if weight.shape != (width,) or weight.device != caps.device or not weight.is_contiguous():
        raise ValueError("affine weight must retain the declared contiguous geometry and device")
    _validate_capacity(out, tokens=tokens, tail=(width,), dtype=caps.dtype, device=caps.device, name="out")
    _validate_output_disjoint("normalized", out, (("state", state), ("weight", weight)))
    if tokens:
        plan.launch(state, weight, out, eps=eps)
    return out[:tokens]


def run_scaled_silu_impl(projected_down, *, plan: _HyperConnectionState, out):
    _require_cuda(plan)
    plan.require_operation("scaled_silu")
    caps = plan.caps
    tokens = plan._live_tokens(projected_down.shape[0])
    _validate_input(projected_down, shape=(tokens, caps.lowrank), caps=caps,
                    name="projected_down", row_strided=True)
    _validate_capacity(out, tokens=tokens, tail=(caps.lowrank,), dtype=caps.dtype,
                       device=caps.device, name="out")
    _validate_output_disjoint("bottleneck", out, (("projected_down", projected_down),))
    if tokens:
        plan.launch(projected_down, out)
    return out[:tokens]


def run_gate_mean_impl(normalized, gate_logits, *, plan: _HyperConnectionState, out):
    _require_cuda(plan)
    plan.require_operation("gate_mean")
    caps = plan.caps
    tokens = plan._live_tokens(normalized.shape[0])
    shape = (tokens, caps.streams * caps.hidden_size)
    _validate_input(normalized, shape=shape, caps=caps, name="normalized")
    _validate_input(gate_logits, shape=shape, caps=caps, name="gate_logits")
    _validate_capacity(out, tokens=tokens, tail=(caps.hidden_size,), dtype=caps.dtype,
                       device=caps.device, name="out")
    _validate_output_disjoint("block_input", out, (("normalized", normalized), ("gate_logits", gate_logits)))
    if tokens:
        plan.launch(normalized, gate_logits, out)
    return out[:tokens]


def _validate_combine_inputs(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    *,
    plan: _HyperConnectionState,
    tokens: int,
) -> None:
    caps = plan.caps
    _validate_input(
        state,
        shape=(tokens, caps.streams * caps.hidden_size),
        caps=caps,
        name="state",
    )
    _validate_input(
        block_output,
        shape=(tokens, caps.hidden_size),
        caps=caps,
        name="block_output",
    )
    _validate_input(
        injection_logits,
        shape=(tokens, caps.streams),
        caps=caps,
        name="injection_logits",
        row_strided=True,
    )


def run_combine_impl(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    *,
    plan: _HyperConnectionState,
) -> torch.Tensor:
    _require_cuda(plan)
    plan.require_operation("combine")
    tokens = plan._live_tokens(state.shape[0])
    _validate_combine_inputs(
        state,
        block_output,
        injection_logits,
        plan=plan,
        tokens=tokens,
    )
    combined = torch.empty_like(state)
    if tokens:
        plan.launch(state, block_output, injection_logits, combined)
    return combined


def run_combine_norm_impl(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    *,
    eps: float,
    plan: _HyperConnectionState,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_cuda(plan)
    plan.require_operation("combine_norm", eps=eps)
    caps = plan.caps
    tokens = plan._live_tokens(state.shape[0])
    _validate_combine_inputs(
        state,
        block_output,
        injection_logits,
        plan=plan,
        tokens=tokens,
    )
    _validate_input(
        next_norm_weight,
        shape=(caps.streams * caps.hidden_size,),
        caps=caps,
        name="next_norm_weight",
    )
    eps = _require_eps(eps)
    from ._cute_config import require_cute_combine_norm
    combined, normalized = torch.empty_like(state), torch.empty_like(state)
    require_cute_combine_norm(
        state=state, block_output=block_output, injection_logits=injection_logits,
        next_norm_weight=next_norm_weight, combined=combined, normalized=normalized,
        streams=caps.streams, hidden_size=caps.hidden_size,
    )
    if tokens:
        plan.launch(state, block_output, injection_logits, next_norm_weight, combined, normalized, eps=eps)
    return combined, normalized


def run_engram_mix_impl(
    state: torch.Tensor,
    projected_kv: torch.Tensor,
    norm_weights: torch.Tensor,
    *,
    eps: float,
    plan: _HyperConnectionState,
    out: torch.Tensor,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply Engram's signed-sqrt gate to caller-owned multi-stream state.

    ``projected_kv`` concatenates the S keys and one shared value. The FP32
    ``norm_weights`` is the offline product of the query/key affine weights;
    unlike Gemma normalization, these weights are not zero-centered.
    """
    _require_cuda(plan)
    plan.require_operation("engram_mix", eps=eps)
    if (token_mask is not None) != plan.query.token_mask:
        raise ValueError("mask presence differs from prepared Engram invocation")
    caps = plan.caps
    tokens = plan._live_tokens(state.shape[0])
    width = caps.streams * caps.hidden_size
    _validate_input(state, shape=(tokens, width), caps=caps, name="state")
    _validate_input(
        projected_kv,
        shape=(tokens, width + caps.hidden_size),
        caps=caps,
        name="projected_kv",
    )
    _validate_input(out, shape=(tokens, width), caps=caps, name="out")
    if (
        norm_weights.shape != (width,)
        or norm_weights.dtype != torch.float32
        or norm_weights.device != caps.device
        or not norm_weights.is_contiguous()
    ):
        raise ValueError(
            f"norm_weights must be contiguous FP32 [{width}] on {caps.device}"
        )
    if token_mask is not None and (
        token_mask.shape != (tokens,)
        or token_mask.dtype != torch.bool
        or token_mask.device != caps.device
        or not token_mask.is_contiguous()
    ):
        raise ValueError(
            f"token_mask must be contiguous bool [{tokens}] on {caps.device}"
        )
    _validate_output_disjoint(
        "out",
        out,
        (("state", state), ("projected_kv", projected_kv), ("norm_weights", norm_weights)),
    )
    eps = _require_eps(eps)
    if tokens:
        plan.launch(state, projected_kv, norm_weights,
                    state if token_mask is None else token_mask, out, eps=eps)
    return out


def _validate_pointwise_tensor(
    tensor: torch.Tensor, *, name: str, shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor.device.type != "cuda" or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")
    if tensor.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be BF16 or FP32")
    if tuple(tensor.shape) != shape or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous with shape {shape}")


def run_swiglu_impl(
    gate_up: torch.Tensor, *, limit: float, out: torch.Tensor,
    round_silu: bool = False,
    plan: _HyperConnectionState,
) -> torch.Tensor:
    """Clamp gate above only and up symmetrically, then compute FP32 SiLU*up.

    Inputs are BF16 FC1 outputs, with gate followed by up along the last axis.
    ``round_silu=True`` rounds SiLU to BF16 before multiplying for vision.
    ``limit=inf`` disables clamping. No activation scaling is applied.
    Warm once before capture; ``out`` is disjoint caller-owned BF16 storage.
    """
    _require_cuda(plan)
    plan.require_operation("swiglu")
    plan._live_tokens(gate_up.shape[0])
    if limit != plan.query.limit or bool(round_silu) != plan.query.round_silu:
        raise ValueError("activation recipe differs from prepared invocation")
    if out.shape[-1] != plan.query.hidden_size:
        raise ValueError("activation width differs from prepared invocation")
    if gate_up.ndim != 2 or gate_up.shape[1] <= 0 or gate_up.shape[1] % 2:
        raise ValueError("gate_up must have shape [T, 2I] with I positive")
    if gate_up.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise TypeError("gate_up and out must be BF16")
    _validate_pointwise_tensor(
        gate_up, name="gate_up", shape=tuple(gate_up.shape), device=gate_up.device,
    )
    _validate_pointwise_tensor(
        out, name="out", shape=(gate_up.shape[0], gate_up.shape[1] // 2),
        device=gate_up.device,
    )
    limit = float(limit)
    if math.isnan(limit) or limit <= 0:
        raise ValueError("limit must be positive (or infinity to disable clamping)")
    _validate_output_disjoint("out", out, (("gate_up", gate_up),))
    if out.numel():
        plan.launch(gate_up, gate_up, out)
    return out


def run_add_impl(
    left: torch.Tensor, right: torch.Tensor, *, out: torch.Tensor, plan: _HyperConnectionState,
) -> torch.Tensor:
    """Add BF16/FP32 operands in FP32, casting once to disjoint BF16/FP32 out."""
    _require_cuda(plan)
    plan.require_operation("add")
    if (str(left.dtype).removeprefix("torch.") != plan.query.left_dtype
            or str(right.dtype).removeprefix("torch.") != plan.query.right_dtype
            or str(out.dtype).removeprefix("torch.") != plan.query.output_dtype):
        raise ValueError("operand dtype differs from prepared add invocation")
    for name, tensor in (("left", left), ("right", right), ("out", out)):
        _validate_pointwise_tensor(
            tensor, name=name, shape=tuple(left.shape), device=left.device,
        )
    _validate_output_disjoint("out", out, (("left", left), ("right", right)))
    if out.numel():
        plan.launch(left, right, out)
    return out



def run_sigmoid_impl(source, *, out, plan: _HyperConnectionState):
    _require_cuda(plan)
    plan.require_operation("sigmoid")
    if (str(source.dtype).removeprefix("torch.") != plan.query.left_dtype
            or str(out.dtype).removeprefix("torch.") != plan.query.output_dtype):
        raise ValueError("operand dtype differs from prepared sigmoid invocation")
    for name, tensor in (("source", source), ("out", out)):
        _validate_pointwise_tensor(
            tensor, name=name, shape=tuple(source.shape), device=source.device,
        )
    _validate_output_disjoint("out", out, (("source", source),))
    if out.numel():
        plan.launch(source, source, out)
    return out


__all__ = [
    "HyperConnectionCaps",
    "HyperConnectionBinding",
    "run_grouped_rmsnorm_impl",
    "run_scaled_silu_impl",
    "run_gate_mean_impl",
    "run_combine_impl",
    "run_combine_norm_impl",
    "run_engram_mix_impl",
    "run_swiglu_impl",
    "run_add_impl",
    "run_sigmoid_impl",
]
