"""Capacity planning, binding, and validation for MTP feedback fusion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from b12x._lib.scratch import (
    ScratchBufferSpec,
    scratch_buffer_spec,
    scratch_tensor,
)
from b12x._lib.scratch_layout import (
    SCRATCH_ALIGN_BYTES,
    align_up,
    dtype_nbytes,
    materialize_scratch_view,
)
from b12x.preparation import FrozenMapping, Plan
from b12x.preparation.types import require_prepared

from ._cute_prefill_config import (
    QWEN_HIDDEN_SIZE,
    QWEN_STREAMS,
    projection_capacity_rows,
)
from ._tuning import MtpFeedbackConfig, MtpFeedbackQuery


def _canonical_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    return result


def _positive(name: str, value: int) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


@dataclass(frozen=True, kw_only=True)
class Caps:
    """Token capacity and static token/multi-stream feedback geometry."""

    device: torch.device | str
    max_tokens: int
    hidden_size: int = 2560
    streams: int = 4
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        device = _canonical_device(self.device)
        if device.type != "cuda":
            raise ValueError(f"MTP feedback requires a CUDA device, got {device}")
        max_tokens = _positive("max_tokens", self.max_tokens)
        hidden_size = _positive("hidden_size", self.hidden_size)
        streams = _positive("streams", self.streams)
        if hidden_size != QWEN_HIDDEN_SIZE or streams != QWEN_STREAMS:
            raise ValueError(
                "MTP feedback only implements the Qwen3.8 CuTe contract "
                f"S={QWEN_STREAMS},H={QWEN_HIDDEN_SIZE}; got "
                f"S={streams},H={hidden_size}"
            )
        if self.dtype != torch.bfloat16:
            raise TypeError(f"MTP feedback requires torch.bfloat16, got {self.dtype}")
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "streams", streams)


@dataclass(frozen=True)
class _Layout:
    """Fixed MTP feedback launch geometry and scratch-buffer contract."""

    caps: Caps
    token_normalized_offset_bytes: int
    state_partial_sums_offset_bytes: int
    state_normalized_offset_bytes: int
    token_path_offset_bytes: int
    _scratch_specs: tuple[ScratchBufferSpec, ...]
    token_projection_rows: int
    state_projection_rows: int
    norm_block_h: int
    norm_block_s: int
    norm_num_warps: int

    def scratch_specs(self) -> tuple[ScratchBufferSpec, ...]:
        return self._scratch_specs

    def shapes_and_dtypes(self) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        return tuple((spec.shape, spec.dtype) for spec in self._scratch_specs)

    def output_shape(self, tokens: int | None = None) -> tuple[int, int, int]:
        live_tokens = self._live_tokens(tokens)
        return (live_tokens, self.caps.streams, self.caps.hidden_size)

    def output_storage_shape(self) -> tuple[int, int, int]:
        """Return the fixed caller-owned logical-capacity output shape."""
        return self.output_shape()

    def _live_tokens(self, tokens: int | None) -> int:
        live_tokens = self.caps.max_tokens if tokens is None else int(tokens)
        if live_tokens < 0 or live_tokens > self.caps.max_tokens:
            raise ValueError(
                f"tokens={live_tokens} exceeds capacity {self.caps.max_tokens}"
            )
        return live_tokens



@dataclass(frozen=True)
class Binding:
    """Caller-owned MTP feedback inputs, output, and scratch views.

    Model inputs and weights are read-only; ``output`` is the mutable result
    buffer and all intermediate tensors are views of ``scratch``.
    """

    _state: _Layout
    tokens: int
    scratch: torch.Tensor
    token_normalized: torch.Tensor
    state_partial_sums: torch.Tensor
    state_normalized: torch.Tensor
    token_path: torch.Tensor
    token_embedding: torch.Tensor
    multi_state: torch.Tensor
    token_norm_weight: torch.Tensor
    state_norm_weight: torch.Tensor
    embedding_fc_weight: torch.Tensor
    hidden_fc_weight: torch.Tensor
    output: torch.Tensor
    plan: Plan | None = None


def _materialize_layout(caps: Caps, config: MtpFeedbackConfig) -> _Layout:
    token_projection_rows, state_projection_rows = projection_capacity_rows(
        max_tokens=caps.max_tokens,
        streams=caps.streams,
        hidden_size=caps.hidden_size,
    )
    h = caps.hidden_size
    s = caps.streams
    token_normalized_offset_bytes = align_up(0, SCRATCH_ALIGN_BYTES)
    cursor = token_normalized_offset_bytes + (
        token_projection_rows * h * dtype_nbytes(caps.dtype)
    )
    state_partial_sums_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = state_partial_sums_offset_bytes + (
        caps.max_tokens * s * dtype_nbytes(torch.float32)
    )
    state_normalized_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = state_normalized_offset_bytes + (
        state_projection_rows * h * dtype_nbytes(caps.dtype)
    )
    token_path_offset_bytes = align_up(cursor, SCRATCH_ALIGN_BYTES)
    cursor = token_path_offset_bytes + (
        token_projection_rows * h * dtype_nbytes(caps.dtype)
    )
    spec = scratch_buffer_spec("mtp_feedback", nbytes=cursor, device=caps.device)
    result = _Layout(
        caps=caps,
        token_normalized_offset_bytes=token_normalized_offset_bytes,
        state_partial_sums_offset_bytes=state_partial_sums_offset_bytes,
        state_normalized_offset_bytes=state_normalized_offset_bytes,
        token_path_offset_bytes=token_path_offset_bytes,
        _scratch_specs=(spec,),
        token_projection_rows=token_projection_rows,
        state_projection_rows=state_projection_rows,
        norm_block_h=config.norm_block_h,
        norm_block_s=config.norm_block_s,
        norm_num_warps=config.norm_num_warps,
    )
    return result


def plan(caps: Caps, *, invocation: FrozenMapping = FrozenMapping(), override: MtpFeedbackConfig | None = None) -> Plan:
    """Declare the exact MTP capacity without compiling projections."""
    from ._preparation import make_plan
    if not isinstance(caps, Caps):
        raise TypeError("plan requires MTP Caps")
    return make_plan(caps, invocation=invocation, override=override)


def bind(plan: Plan, **kwargs) -> Binding:
    state = require_prepared(plan, "sequence.mtp_feedback")
    return state.bind(_plan=plan, **kwargs)


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    caps: Caps,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != caps.dtype:
        raise TypeError(f"{name} must have dtype {caps.dtype}, got {tensor.dtype}")
    if tensor.device != caps.device:
        raise ValueError(f"{name} must be on {caps.device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.untyped_storage().data_ptr()) + int(
        tensor.storage_offset()
    ) * int(tensor.element_size())
    return start, start + int(tensor.numel()) * int(tensor.element_size())


def _overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_start, left_end = _byte_interval(left)
    right_start, right_end = _byte_interval(right)
    return left_start < right_end and right_start < left_end


def _bind(
    plan: _Layout,
    *,
    _plan: Plan | None = None,
    scratch: torch.Tensor | Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    token_embedding: torch.Tensor,
    multi_state: torch.Tensor,
    token_norm_weight: torch.Tensor,
    state_norm_weight: torch.Tensor,
    embedding_fc_weight: torch.Tensor,
    hidden_fc_weight: torch.Tensor,
    output: torch.Tensor,
    tokens: int | None = None,
) -> Binding:
    """Bind fixed-capacity tensors without allocating runtime storage."""
    if not isinstance(plan, _Layout):
        raise TypeError("MTP binding requires a materialized layout")
    caps = plan.caps
    live_tokens = plan._live_tokens(tokens)
    scratch_storage = scratch_tensor(
        scratch,
        plan.scratch_specs(),
        owner="MTP feedback",
    )
    token_normalized, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.token_normalized_offset_bytes,
        shape=(plan.token_projection_rows, caps.hidden_size),
        dtype=caps.dtype,
    )
    state_partial_sums, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.state_partial_sums_offset_bytes,
        shape=(caps.max_tokens, caps.streams),
        dtype=torch.float32,
    )
    state_normalized, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.state_normalized_offset_bytes,
        shape=(
            plan.state_projection_rows // caps.streams,
            caps.streams,
            caps.hidden_size,
        ),
        dtype=caps.dtype,
    )
    token_path, _ = materialize_scratch_view(
        scratch_storage,
        offset_bytes=plan.token_path_offset_bytes,
        shape=(plan.token_projection_rows, caps.hidden_size),
        dtype=caps.dtype,
    )
    capacity_shapes = {
        "token_embedding": (caps.max_tokens, caps.hidden_size),
        "multi_state": (caps.max_tokens, caps.streams, caps.hidden_size),
        "token_norm_weight": (caps.hidden_size,),
        "state_norm_weight": (caps.streams * caps.hidden_size,),
        "embedding_fc_weight": (caps.hidden_size, caps.hidden_size),
        "hidden_fc_weight": (caps.hidden_size, caps.hidden_size),
        "output": plan.output_storage_shape(),
    }
    tensors = {
        "token_embedding": token_embedding,
        "multi_state": multi_state,
        "token_norm_weight": token_norm_weight,
        "state_norm_weight": state_norm_weight,
        "embedding_fc_weight": embedding_fc_weight,
        "hidden_fc_weight": hidden_fc_weight,
        "output": output,
    }
    for name, shape in capacity_shapes.items():
        _require_tensor(name, tensors[name], shape=shape, caps=caps)

    mutable = (("scratch", scratch_storage), ("output", output))
    read_only = tuple(
        (name, tensor) for name, tensor in tensors.items() if name != "output"
    )
    for index, (left_name, left) in enumerate(mutable):
        for right_name, right in mutable[index + 1 :]:
            if _overlaps(left, right):
                raise ValueError(
                    f"mutable buffers {left_name} and {right_name} must not overlap"
                )
        for right_name, right in read_only:
            if _overlaps(left, right):
                raise ValueError(
                    f"mutable buffer {left_name} must not overlap read-only "
                    f"tensor {right_name}"
                )

    return Binding(
        _state=plan,
        plan=_plan,
        tokens=live_tokens,
        scratch=scratch_storage,
        token_normalized=token_normalized[:live_tokens],
        state_partial_sums=state_partial_sums[:live_tokens],
        state_normalized=state_normalized[:live_tokens],
        token_path=token_path[:live_tokens],
        token_embedding=token_embedding[:live_tokens],
        multi_state=multi_state[:live_tokens],
        token_norm_weight=token_norm_weight,
        state_norm_weight=state_norm_weight,
        embedding_fc_weight=embedding_fc_weight,
        hidden_fc_weight=hidden_fc_weight,
        output=output[:live_tokens],
    )


def run(binding: Binding, *, eps: float = 1e-6) -> torch.Tensor:
    """Write and return the caller-owned BF16 ``[T,S,H]`` draft input."""
    if not isinstance(binding, Binding):
        raise TypeError(f"binding must be Binding, got {type(binding)!r}")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps_value}")
    if binding.plan is None:
        raise TypeError("MTP run requires a session-prepared binding")
    from ._kernels import run_mtp_feedback

    run_mtp_feedback(
        binding.token_embedding,
        binding.multi_state,
        binding.token_norm_weight,
        binding.state_norm_weight,
        binding.embedding_fc_weight,
        binding.hidden_fc_weight,
        binding.scratch,
        binding.output,
        eps=eps_value,
        plan=binding.plan,
    )
    return binding.output


__all__ = ["Caps", "Plan", "Binding", "plan", "bind", "run"]
