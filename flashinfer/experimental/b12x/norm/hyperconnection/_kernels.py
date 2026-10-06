"""Triton kernels for learned low-rank HyperConnection primitives."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from b12x.preparation.types import plan_from_handle, require_prepared

from ._cute_config import require_cute_combine_norm


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


def _require_disjoint(
    output_name: str,
    output: torch.Tensor,
    inputs: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    output_start, output_end = _byte_interval(output)
    for input_name, tensor in inputs:
        input_start, input_end = _byte_interval(tensor)
        if output_start < input_end and input_start < output_end:
            raise ValueError(f"{output_name} must not overlap {input_name}")


@triton.jit
def _grouped_rmsnorm_kernel(
    state_ptr,
    weight_ptr,
    out_ptr,
    eps,
    HIDDEN_SIZE: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H).to(tl.int64)
    mask = cols < HIDDEN_SIZE
    offsets = row * HIDDEN_SIZE + cols
    values = tl.load(state_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / HIDDEN_SIZE
    inv_rms = tl.rsqrt(variance + eps)
    weight_offsets = (row % STREAMS) * HIDDEN_SIZE + cols
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, values * inv_rms * (1.0 + weight), mask=mask)


@triton.jit
def _scaled_silu_kernel(
    projected_ptr,
    out_ptr,
    elements,
    STREAMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < elements
    values = tl.load(projected_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scaled = (values / STREAMS).to(tl.bfloat16)
    activated = scaled.to(tl.float32) * tl.sigmoid(scaled.to(tl.float32))
    tl.store(out_ptr + offsets, activated, mask=mask)


@triton.jit
def _gate_mean_kernel(
    normalized_ptr,
    gate_logits_ptr,
    out_ptr,
    HIDDEN_SIZE: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1).to(tl.int64)
    cols = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    mask = cols < HIDDEN_SIZE
    token_base = token * STREAMS * HIDDEN_SIZE
    total = tl.zeros((BLOCK_H,), tl.float32)
    for stream in tl.static_range(0, STREAMS):
        offsets = token_base + stream * HIDDEN_SIZE + cols
        values = tl.load(normalized_ptr + offsets, mask=mask, other=0.0).to(tl.bfloat16)
        logits = tl.load(gate_logits_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.sigmoid(logits).to(tl.bfloat16)
        product = (gate * values).to(tl.bfloat16)
        total += product.to(tl.float32)
    out_offsets = token * HIDDEN_SIZE + cols
    tl.store(out_ptr + out_offsets, total / STREAMS, mask=mask)


@triton.jit
def _combine_kernel(
    state_ptr,
    block_output_ptr,
    injection_logits_ptr,
    combined_ptr,
    HIDDEN_SIZE: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    stream = tl.program_id(1).to(tl.int64)
    cols = tl.arange(0, BLOCK_H).to(tl.int64)
    mask = cols < HIDDEN_SIZE
    state_offsets = (token * STREAMS + stream) * HIDDEN_SIZE + cols
    output_offsets = token * HIDDEN_SIZE + cols
    state = tl.load(state_ptr + state_offsets, mask=mask, other=0.0).to(tl.float32)
    block_output = tl.load(block_output_ptr + output_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    logit = tl.load(injection_logits_ptr + token * STREAMS + stream).to(tl.float32)
    scale = 2.0 * tl.sigmoid(logit / STREAMS)
    tl.store(
        combined_ptr + state_offsets,
        state + scale * block_output,
        mask=mask,
    )


def _norm_launch(
    state: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> None:
    _grouped_rmsnorm_kernel[(int(state.shape[0]) * streams,)](
        state,
        weight,
        out,
        float(eps),
        HIDDEN_SIZE=hidden_size,
        STREAMS=streams,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )


def _scaled_silu_launch(
    projected_down: torch.Tensor,
    out: torch.Tensor,
    streams: int,
    block: int,
) -> None:
    elements = int(projected_down.numel())
    _scaled_silu_kernel[(triton.cdiv(elements, block),)](
        projected_down,
        out,
        elements,
        STREAMS=streams,
        BLOCK=block,
        num_warps=4,
    )


def _gate_mean_launch(
    normalized: torch.Tensor,
    gate_logits: torch.Tensor,
    out: torch.Tensor,
    streams: int,
    hidden_size: int,
    block_h: int,
) -> None:
    _gate_mean_kernel[(int(normalized.shape[0]), triton.cdiv(hidden_size, block_h))](
        normalized,
        gate_logits,
        out,
        HIDDEN_SIZE=hidden_size,
        STREAMS=streams,
        BLOCK_H=block_h,
        num_warps=4,
    )


def _combine_launch(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    combined: torch.Tensor,
    streams: int,
    hidden_size: int,
    block_h: int,
    num_warps: int,
) -> None:
    _combine_kernel[(int(state.shape[0]), streams)](
        state,
        block_output,
        injection_logits,
        combined,
        HIDDEN_SIZE=hidden_size,
        STREAMS=streams,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )


@torch.library.custom_op("b12x::hyperconnection_grouped_rmsnorm", mutates_args=("out",))
def _grouped_rmsnorm_op(
    state: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
    eps: float, plan_handle: int, zero_centered: bool = True,
) -> None:
    from ._impl import run_grouped_rmsnorm_impl
    prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", state.device)
    run_grouped_rmsnorm_impl(state, weight, eps=eps, plan=prepared, out=out, zero_centered=zero_centered)


@_grouped_rmsnorm_op.register_fake
def _grouped_rmsnorm_fake(
    state: torch.Tensor, weight: torch.Tensor, out: torch.Tensor,
    eps: float, plan_handle: int, zero_centered: bool = True,
) -> None:
    del state, weight, out, eps, plan_handle


@torch.library.custom_op("b12x::hyperconnection_scaled_silu", mutates_args=("out",))
def _scaled_silu_op(
    projected_down: torch.Tensor, out: torch.Tensor, plan_handle: int,
) -> None:
    from ._impl import run_scaled_silu_impl
    prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", projected_down.device)
    run_scaled_silu_impl(projected_down, plan=prepared, out=out)


@_scaled_silu_op.register_fake
def _scaled_silu_fake(
    projected_down: torch.Tensor, out: torch.Tensor, plan_handle: int,
) -> None:
    del projected_down, out, plan_handle


@torch.library.custom_op("b12x::hyperconnection_gate_mean", mutates_args=("out",))
def _gate_mean_op(
    normalized: torch.Tensor, gate_logits: torch.Tensor, out: torch.Tensor,
    plan_handle: int,
) -> None:
    from ._impl import run_gate_mean_impl
    prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", normalized.device)
    run_gate_mean_impl(normalized, gate_logits, plan=prepared, out=out)


@_gate_mean_op.register_fake
def _gate_mean_fake(
    normalized: torch.Tensor, gate_logits: torch.Tensor, out: torch.Tensor,
    plan_handle: int,
) -> None:
    del normalized, gate_logits, out, plan_handle


@torch.library.custom_op("b12x::hyperconnection_combine", mutates_args=())
def _combine_op(
    state: torch.Tensor, block_output: torch.Tensor, injection_logits: torch.Tensor,
    plan_handle: int,
) -> torch.Tensor:
    from ._impl import run_combine_impl
    prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", state.device)
    return run_combine_impl(state, block_output, injection_logits, plan=prepared)


@_combine_op.register_fake
def _combine_fake(
    state: torch.Tensor, block_output: torch.Tensor, injection_logits: torch.Tensor,
    plan_handle: int,
) -> torch.Tensor:
    del block_output, injection_logits, plan_handle
    return torch.empty_like(state)


@torch.library.custom_op("b12x::hyperconnection_combine_norm", mutates_args=())
def _combine_norm_op(
    state: torch.Tensor, block_output: torch.Tensor, injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor, eps: float, plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ._impl import run_combine_norm_impl
    prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", state.device)
    return run_combine_norm_impl(
        state, block_output, injection_logits, next_norm_weight, eps=eps, plan=prepared,
    )


@_combine_norm_op.register_fake
def _combine_norm_fake(
    state: torch.Tensor, block_output: torch.Tensor, injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor, eps: float, plan_handle: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, next_norm_weight, eps, plan_handle
    return torch.empty_like(state), torch.empty_like(state)


@torch.library.custom_op("b12x::hyperconnection_engram_mix", mutates_args=("out",))
def _engram_mix_op(
    state: torch.Tensor, projected_kv: torch.Tensor, norm_weights: torch.Tensor,
    token_mask: torch.Tensor | None, out: torch.Tensor, eps: float,
    plan_handle: int,
) -> None:
    from ._impl import run_engram_mix_impl
    prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", state.device)
    run_engram_mix_impl(state, projected_kv, norm_weights, eps=eps, plan=prepared,
                        out=out, token_mask=token_mask)


@_engram_mix_op.register_fake
def _engram_mix_fake(
    state: torch.Tensor, projected_kv: torch.Tensor, norm_weights: torch.Tensor,
    token_mask: torch.Tensor | None, out: torch.Tensor, eps: float,
    plan_handle: int,
) -> None:
    del state, projected_kv, norm_weights, token_mask, out, eps, plan_handle


@torch.library.custom_op("b12x::hyperconnection_swiglu", mutates_args=("out",))
def _swiglu_op(
    gate_up: torch.Tensor, out: torch.Tensor, limit: float, round_silu: bool,
    plan_handle: int,
) -> None:
    from ._impl import run_swiglu_impl
    prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", gate_up.device)
    run_swiglu_impl(gate_up, out=out, limit=limit, round_silu=round_silu, plan=prepared)


@_swiglu_op.register_fake
def _swiglu_fake(
    gate_up: torch.Tensor, out: torch.Tensor, limit: float, round_silu: bool,
    plan_handle: int,
) -> None:
    del gate_up, out, limit, round_silu, plan_handle


@torch.library.custom_op("b12x::hyperconnection_add", mutates_args=("out",))
def _add_op(
    left: torch.Tensor, right: torch.Tensor, out: torch.Tensor,
    plan_handle: int,
) -> None:
    from ._impl import run_add_impl
    prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", left.device)
    run_add_impl(left, right, out=out, plan=prepared)


@_add_op.register_fake
def _add_fake(
    left: torch.Tensor, right: torch.Tensor, out: torch.Tensor,
    plan_handle: int,
) -> None:
    del left, right, out, plan_handle


@torch.library.custom_op("b12x::hyperconnection_sigmoid", mutates_args=("out",))
def _sigmoid_op(source: torch.Tensor, out: torch.Tensor, plan_handle: int) -> None:
    from ._impl import run_sigmoid_impl
    prepared = require_prepared(plan_from_handle(plan_handle), "norm.hyperconnection", source.device)
    run_sigmoid_impl(source, out=out, plan=prepared)


@_sigmoid_op.register_fake
def _sigmoid_fake(source: torch.Tensor, out: torch.Tensor, plan_handle: int) -> None:
    return None
