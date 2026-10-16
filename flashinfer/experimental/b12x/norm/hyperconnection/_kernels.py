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


@triton.jit(do_not_specialize=["state_stride", "residual_stride"])
def _grouped_rmsnorm_kernel(
    state_ptr,
    weight_ptr,
    out_ptr,
    residual_ptr,
    eps,
    state_stride,
    residual_stride,
    HIDDEN_SIZE: tl.constexpr,
    STREAMS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ZERO_CENTERED: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    PAIRED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H).to(tl.int64)
    mask = cols < HIDDEN_SIZE
    offsets = row * HIDDEN_SIZE + cols
    source_offsets = row // STREAMS * state_stride + row % STREAMS * HIDDEN_SIZE + cols
    if PAIRED:
        if row % STREAMS == 0:
            values = tl.load(
                state_ptr + row // STREAMS * state_stride + cols, mask, other=0
            ).to(tl.float32)
        else:
            values = tl.load(
                residual_ptr + row // STREAMS * residual_stride + cols, mask, other=0
            ).to(tl.float32)
    else:
        values = tl.load(state_ptr + source_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
    if ADD_RESIDUAL:
        residual_offsets = (
            row // STREAMS * residual_stride + row % STREAMS * HIDDEN_SIZE + cols
        )
        values += tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        values = values.to(state_ptr.dtype.element_ty).to(tl.float32)
        tl.store(state_ptr + source_offsets, values, mask=mask)
    variance = tl.sum(values * values, axis=0) / HIDDEN_SIZE
    inv_rms = tl.rsqrt(variance + eps)
    weight_offsets = (row % STREAMS) * HIDDEN_SIZE + cols
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=0.0).to(tl.float32)
    if ZERO_CENTERED:
        weight += 1.0
    tl.store(out_ptr + offsets, values * inv_rms * weight, mask=mask)


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
    zero_centered=True,
    residual=None,
    paired=None,
) -> None:
    right = paired if paired is not None else residual
    _grouped_rmsnorm_kernel[(int(state.shape[0]) * streams,)](
        state,
        weight,
        out,
        state if right is None else right,
        float(eps),
        state.stride(0),
        state.stride(0) if right is None else right.stride(0),
        HIDDEN_SIZE=hidden_size,
        STREAMS=streams,
        BLOCK_H=block_h,
        ZERO_CENTERED=zero_centered,
        ADD_RESIDUAL=residual is not None,
        PAIRED=paired is not None,
        num_warps=num_warps,
        enable_fp_fusion=False,
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
    state: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    plan_handle: int,
    zero_centered: bool = True,
) -> None:
    from ._impl import run_grouped_rmsnorm_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", state.device
    )
    run_grouped_rmsnorm_impl(
        state, weight, eps=eps, plan=prepared, out=out, zero_centered=zero_centered
    )


@_grouped_rmsnorm_op.register_fake
def _grouped_rmsnorm_fake(
    state: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    plan_handle: int,
    zero_centered: bool = True,
) -> None:
    del state, weight, out, eps, plan_handle


@torch.library.custom_op(
    "b12x::hyperconnection_residual_rmsnorm", mutates_args=("state", "out")
)
def _residual_rmsnorm_op(
    state: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    plan_handle: int,
    zero_centered: bool = True,
) -> None:
    from ._impl import run_grouped_rmsnorm_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", state.device
    )
    run_grouped_rmsnorm_impl(
        state,
        weight,
        eps=eps,
        plan=prepared,
        out=out,
        zero_centered=zero_centered,
        residual=residual,
    )


@_residual_rmsnorm_op.register_fake
def _residual_rmsnorm_fake(
    state, residual, weight, out, eps, plan_handle, zero_centered=True
):
    pass


@torch.library.custom_op("b12x::hyperconnection_scaled_silu", mutates_args=("out",))
def _scaled_silu_op(
    projected_down: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
) -> None:
    from ._impl import run_scaled_silu_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", projected_down.device
    )
    run_scaled_silu_impl(projected_down, plan=prepared, out=out)


@_scaled_silu_op.register_fake
def _scaled_silu_fake(
    projected_down: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
) -> None:
    del projected_down, out, plan_handle


@torch.library.custom_op("b12x::hyperconnection_gate_mean", mutates_args=("out",))
def _gate_mean_op(
    normalized: torch.Tensor,
    gate_logits: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
) -> None:
    from ._impl import run_gate_mean_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", normalized.device
    )
    run_gate_mean_impl(normalized, gate_logits, plan=prepared, out=out)


@_gate_mean_op.register_fake
def _gate_mean_fake(
    normalized: torch.Tensor,
    gate_logits: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
) -> None:
    del normalized, gate_logits, out, plan_handle


@torch.library.custom_op("b12x::hyperconnection_combine", mutates_args=())
def _combine_op(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    plan_handle: int,
) -> torch.Tensor:
    from ._impl import run_combine_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", state.device
    )
    return run_combine_impl(state, block_output, injection_logits, plan=prepared)


@_combine_op.register_fake
def _combine_fake(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    plan_handle: int,
) -> torch.Tensor:
    del block_output, injection_logits, plan_handle
    return torch.empty_like(state)


@torch.library.custom_op("b12x::hyperconnection_combine_norm", mutates_args=())
def _combine_norm_op(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    eps: float,
    plan_handle: int,
    zero_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    from ._impl import run_combine_norm_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", state.device
    )
    return run_combine_norm_impl(
        state,
        block_output,
        injection_logits,
        next_norm_weight,
        eps=eps,
        plan=prepared,
        zero_centered=zero_centered,
    )


@_combine_norm_op.register_fake
def _combine_norm_fake(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    eps: float,
    plan_handle: int,
    zero_centered: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, next_norm_weight, eps, plan_handle
    return torch.empty_like(state), torch.empty_like(state)


@torch.library.custom_op(
    "b12x::hyperconnection_combine_norm_out", mutates_args=("combined", "normalized")
)
def _combine_norm_out_op(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    combined: torch.Tensor,
    normalized: torch.Tensor,
    eps: float,
    plan_handle: int,
    zero_centered: bool = True,
) -> None:
    from ._impl import run_combine_norm_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", state.device
    )
    run_combine_norm_impl(
        state,
        block_output,
        injection_logits,
        next_norm_weight,
        eps=eps,
        plan=prepared,
        zero_centered=zero_centered,
        out=(combined, normalized),
    )


@_combine_norm_out_op.register_fake
def _combine_norm_out_fake(
    state: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    next_norm_weight: torch.Tensor,
    combined: torch.Tensor,
    normalized: torch.Tensor,
    eps: float,
    plan_handle: int,
    zero_centered: bool = True,
) -> None:
    return None


@torch.library.custom_op("b12x::hyperconnection_engram_mix", mutates_args=("out",))
def _engram_mix_op(
    state: torch.Tensor,
    projected_kv: torch.Tensor,
    norm_weights: torch.Tensor,
    token_mask: torch.Tensor | None,
    out: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    from ._impl import run_engram_mix_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", state.device
    )
    run_engram_mix_impl(
        state,
        projected_kv,
        norm_weights,
        eps=eps,
        plan=prepared,
        out=out,
        token_mask=token_mask,
    )


@_engram_mix_op.register_fake
def _engram_mix_fake(
    state: torch.Tensor,
    projected_kv: torch.Tensor,
    norm_weights: torch.Tensor,
    token_mask: torch.Tensor | None,
    out: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    del state, projected_kv, norm_weights, token_mask, out, eps, plan_handle


@torch.library.custom_op("b12x::hyperconnection_swiglu", mutates_args=("out",))
def _swiglu_op(
    gate_up: torch.Tensor,
    out: torch.Tensor,
    limit: float,
    round_silu: bool,
    plan_handle: int,
) -> None:
    from ._impl import run_swiglu_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", gate_up.device
    )
    run_swiglu_impl(gate_up, out=out, limit=limit, round_silu=round_silu, plan=prepared)


@_swiglu_op.register_fake
def _swiglu_fake(
    gate_up: torch.Tensor,
    out: torch.Tensor,
    limit: float,
    round_silu: bool,
    plan_handle: int,
) -> None:
    del gate_up, out, limit, round_silu, plan_handle


@torch.library.custom_op("b12x::hyperconnection_add", mutates_args=("out",))
def _add_op(
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
) -> None:
    from ._impl import run_add_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", left.device
    )
    run_add_impl(left, right, out=out, plan=prepared)


@_add_op.register_fake
def _add_fake(
    left: torch.Tensor,
    right: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
) -> None:
    del left, right, out, plan_handle


@torch.library.custom_op("b12x::hyperconnection_sigmoid", mutates_args=("out",))
def _sigmoid_op(source: torch.Tensor, out: torch.Tensor, plan_handle: int) -> None:
    from ._impl import run_sigmoid_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", source.device
    )
    run_sigmoid_impl(source, out=out, plan=prepared)


@_sigmoid_op.register_fake
def _sigmoid_fake(source: torch.Tensor, out: torch.Tensor, plan_handle: int) -> None:
    return None


@torch.library.custom_op("b12x::hyperconnection_head_gate", mutates_args=("values",))
def _head_gate_op(values: torch.Tensor, gates: torch.Tensor, plan_handle: int) -> None:
    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", values.device
    )
    prepared.require_operation("head_gate")
    q = prepared.query
    if (
        values.ndim != 3
        or values.shape[1:] != (q.streams, q.hidden_size)
        or values.shape[0] > q.max_tokens
        or values.dtype != torch.bfloat16
        or not values.is_contiguous()
        or gates.shape != values.shape[:2]
        or gates.stride() != (q.gate_stride, 1)
        or gates.dtype != values.dtype
        or gates.device != values.device
    ):
        raise ValueError(
            "head gates must match the prepared BF16 head geometry and row stride"
        )
    prepared.launch(values, gates, values)


@_head_gate_op.register_fake
def _head_gate_fake(values, gates, plan_handle):
    pass


@torch.library.custom_op("b12x::hyperconnection_paired_rmsnorm", mutates_args=("out",))
def _paired_rmsnorm_op(
    left: torch.Tensor,
    right: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    from ._impl import run_grouped_rmsnorm_impl

    prepared = require_prepared(
        plan_from_handle(plan_handle), "norm.hyperconnection", left.device
    )
    run_grouped_rmsnorm_impl(
        left,
        weight,
        eps=eps,
        plan=prepared,
        out=out,
        zero_centered=prepared.query.zero_centered,
        paired=right,
    )


@_paired_rmsnorm_op.register_fake
def _paired_rmsnorm_fake(left, right, weight, out, eps, plan_handle):
    pass
