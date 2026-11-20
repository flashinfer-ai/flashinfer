"""Graph-safe sidecar collection for MoE quantization calibration.

The fused kernels keep route-indexed activation scratch alive until return.
These small Triton sidecars consume that scratch immediately, accumulating
gate-square-weighted moments and copying a deterministic subsample into
caller-owned ring buffers.  All allocations and persistence remain the
integration caller's responsibility; the functions below only launch kernels
against stable tensors and are therefore safe to capture and replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class RouteInputBuffers:
    enabled: torch.Tensor
    epoch_counter: torch.Tensor
    epoch: torch.Tensor
    tokens_routed: torch.Tensor
    gate_sum: torch.Tensor
    gate_sq_sum: torch.Tensor
    input_sq_sum: torch.Tensor
    input_weight_sum: torch.Tensor
    input_count: torch.Tensor
    sample_cursor: torch.Tensor
    sample_dropped: torch.Tensor
    sample_slots: torch.Tensor
    sample_values: torch.Tensor
    sample_weight: torch.Tensor
    sample_observation: torch.Tensor
    sample_experts: torch.Tensor
    sample_gates: torch.Tensor
    sample_split: torch.Tensor


@dataclass(frozen=True)
class MidBuffers:
    enabled: torch.Tensor
    epoch: torch.Tensor
    mid_sq_sum: torch.Tensor
    mid_weight_sum: torch.Tensor
    mid_count: torch.Tensor
    sample_cursor: torch.Tensor
    sample_dropped: torch.Tensor
    sample_slots: torch.Tensor
    sample_values: torch.Tensor
    sample_weight: torch.Tensor
    sample_observation: torch.Tensor
    sample_expert: torch.Tensor
    sample_split: torch.Tensor


@triton.jit
def _mix64(value):
    value ^= value >> 30
    value *= 0xBF58476D1CE4E5B9
    value ^= value >> 27
    value *= 0x94D049BB133111EB
    return value ^ (value >> 31)


@triton.jit
def _advance_epoch_kernel(enabled, epoch_counter, epoch):
    active = tl.load(enabled).to(tl.int32)
    current = tl.load(epoch_counter).to(tl.int64)
    tl.store(epoch, current)
    tl.store(epoch_counter, current + active)


@triton.jit
def _route_stats_kernel(
    topk_ids,
    topk_weights,
    is_padding,
    enabled,
    tokens_routed,
    gate_sum,
    gate_sq_sum,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    collect_routing: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    token = tl.program_id(0)
    if token >= num_tokens:
        return
    slots = tl.arange(0, BLOCK_TOPK)
    routes = token * topk + slots
    live_token = tl.load(enabled).to(tl.int1) & ~tl.load(is_padding + token)
    expert = tl.load(topk_ids + routes, mask=slots < topk, other=-1).to(tl.int32)
    live = live_token & (slots < topk) & (expert >= 0) & (expert < num_experts)
    if collect_routing:
        weight = tl.load(topk_weights + routes, mask=live, other=0.0).to(tl.float64)
        tl.atomic_add(tokens_routed + expert, 1, mask=live)
        tl.atomic_add(gate_sum + expert, weight, mask=live)
        tl.atomic_add(gate_sq_sum + expert, weight * weight, mask=live)


@triton.jit
def _input_moment_kernel(
    x,
    topk_ids,
    topk_weights,
    is_padding,
    enabled,
    epoch,
    square_sum,
    weight_sum,
    count,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    width: tl.constexpr,
    expert_begin: tl.constexpr,
    expert_end: tl.constexpr,
    sample_rate: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    if token >= num_tokens:
        return
    live_token = tl.load(enabled).to(tl.int1) & ~tl.load(is_padding + token)
    current_epoch = tl.load(epoch).to(tl.uint64)
    selected = (_mix64(current_epoch ^ token.to(tl.uint64)) % sample_rate) == 0
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    feature_mask = live_token & selected & (offsets < width)
    values = tl.load(x + token * width + offsets, mask=feature_mask, other=0.0).to(
        tl.float32
    )
    values_sq = values * values
    for route_slot in tl.static_range(0, topk):
        route = token * topk + route_slot
        expert = tl.load(topk_ids + route).to(tl.int32)
        live = live_token & selected & (expert >= expert_begin) & (expert < expert_end)
        local_expert = expert - expert_begin
        weight = tl.load(topk_weights + route, mask=live, other=0.0).to(tl.float32)
        out = local_expert * width + offsets
        tl.atomic_add(
            square_sum + out,
            values_sq * weight * weight,
            mask=live & (offsets < width),
        )
        if block == 0 and live:
            weight64 = weight.to(tl.float64)
            tl.atomic_add(weight_sum + local_expert, weight64 * weight64)
            tl.atomic_add(count + local_expert, 1)


@triton.jit
def _input_sample_meta_kernel(
    topk_ids,
    topk_weights,
    is_padding,
    enabled,
    epoch,
    cursor,
    dropped,
    slots,
    sample_weight,
    sample_observation,
    sample_experts,
    sample_gates,
    sample_split,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    capacity: tl.constexpr,
    sample_rate: tl.constexpr,
    validation_modulus: tl.constexpr,
):
    token = tl.program_id(0)
    if token >= num_tokens:
        return
    live = tl.load(enabled).to(tl.int1) & ~tl.load(is_padding + token)
    current_epoch = tl.load(epoch).to(tl.uint64)
    selected = (_mix64(current_epoch ^ token.to(tl.uint64)) % sample_rate) == 0
    live &= selected
    slot = tl.atomic_add(cursor, 1, mask=live)
    accepted = live & (slot < capacity)
    tl.store(slots + token, tl.where(accepted, slot, -1))
    if live & ~accepted:
        tl.atomic_add(dropped, 1)
    if accepted:
        route_offsets = token * topk + tl.arange(0, 32)
        mask = tl.arange(0, 32) < topk
        experts = tl.load(topk_ids + route_offsets, mask=mask, other=-1).to(tl.int32)
        weights = tl.load(topk_weights + route_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        total_weight = tl.sum(weights * weights, axis=0)
        observation = (current_epoch << 32) | token.to(tl.uint64)
        tl.store(sample_weight + slot, total_weight)
        tl.store(sample_observation + slot, observation.to(tl.int64))
        tl.store(sample_experts + slot * topk + tl.arange(0, 32), experts, mask=mask)
        tl.store(sample_gates + slot * topk + tl.arange(0, 32), weights, mask=mask)
        split_key = observation ^ 0x6A09E667F3BCC909
        split = (_mix64(split_key) % validation_modulus) == 0
        tl.store(sample_split + slot, split.to(tl.int8))


@triton.jit
def _sample_values_kernel(
    source,
    slots,
    output,
    rows: tl.constexpr,
    width: tl.constexpr,
    source_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    if row >= rows:
        return
    slot = tl.load(slots + row).to(tl.int32)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = (slot >= 0) & (offsets < width)
    values = tl.load(source + row * source_stride + offsets, mask=mask, other=0.0)
    tl.store(output + slot * width + offsets, values, mask=mask)


@triton.jit
def _paired_sample_values_kernel(
    source,
    slots,
    output,
    ready,
    rows: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    if row >= rows:
        return
    slot = tl.load(slots + row).to(tl.int32)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = (slot >= 0) & (offsets < width)
    values = tl.load(source + row * width + offsets, mask=mask, other=0.0)
    tl.store(output + slot * width + offsets, values, mask=mask)
    tl.store(ready + slot + offsets, 1, mask=(slot >= 0) & (offsets == 0))


@triton.jit
def _unscale_route_rows_kernel(
    values,
    topk_ids,
    expert_map,
    scales,
    num_routes: tl.constexpr,
    num_experts: tl.constexpr,
    width: tl.constexpr,
    scale_stride: tl.constexpr,
    scale_offset: tl.constexpr,
    BLOCK: tl.constexpr,
):
    route = tl.program_id(0)
    block = tl.program_id(1)
    if route >= num_routes:
        return
    expert = tl.load(topk_ids + route).to(tl.int32)
    live = (expert >= 0) & (expert < num_experts)
    local_expert = tl.load(expert_map + expert, mask=live, other=-1).to(tl.int32)
    live &= local_expert >= 0
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = live & (offsets < width)
    value = tl.load(values + route * width + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    scale = tl.load(
        scales + local_expert * scale_stride + scale_offset + offsets,
        mask=mask,
        other=1.0,
    ).to(tl.float32)
    tl.store(values + route * width + offsets, value / scale, mask=mask)


@triton.jit
def _mid_moment_kernel(
    source,
    topk_ids,
    topk_weights,
    expert_map,
    is_padding,
    enabled,
    epoch,
    square_sum,
    weight_sum,
    count,
    num_tokens: tl.constexpr,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    width: tl.constexpr,
    source_stride: tl.constexpr,
    sample_rate: tl.constexpr,
    has_expert_map: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    if token >= num_tokens:
        return
    live_token = tl.load(enabled).to(tl.int1) & ~tl.load(is_padding + token)
    current_epoch = tl.load(epoch).to(tl.uint64)
    selected = (_mix64(current_epoch ^ token.to(tl.uint64)) % sample_rate) == 0
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    for route_slot in tl.static_range(0, topk):
        route = token * topk + route_slot
        expert = tl.load(topk_ids + route).to(tl.int32)
        live = live_token & selected & (expert >= 0) & (expert < num_experts)
        if has_expert_map:
            local_expert = tl.load(expert_map + expert, mask=live, other=-1).to(
                tl.int32
            )
            live &= local_expert >= 0
        mask = live & (offsets < width)
        values = tl.load(
            source + route * source_stride + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        weight = tl.load(topk_weights + route, mask=live, other=0.0).to(tl.float32)
        out = expert * width + offsets
        tl.atomic_add(square_sum + out, values * values * weight * weight, mask=mask)
        if block == 0 and live:
            weight64 = weight.to(tl.float64)
            tl.atomic_add(weight_sum + expert, weight64 * weight64)
            tl.atomic_add(count + expert, 1)


@triton.jit
def _mid_sample_meta_kernel(
    topk_ids,
    topk_weights,
    expert_map,
    is_padding,
    enabled,
    epoch,
    cursor,
    dropped,
    slots,
    sample_weight,
    sample_observation,
    sample_expert,
    sample_split,
    num_routes: tl.constexpr,
    topk: tl.constexpr,
    num_experts: tl.constexpr,
    capacity: tl.constexpr,
    sample_rate: tl.constexpr,
    validation_modulus: tl.constexpr,
    has_expert_map: tl.constexpr,
):
    route = tl.program_id(0)
    if route >= num_routes:
        return
    token = route // topk
    route_slot = route - token * topk
    live = tl.load(enabled).to(tl.int1) & ~tl.load(is_padding + token)
    expert = tl.load(topk_ids + route).to(tl.int32)
    live &= (expert >= 0) & (expert < num_experts)
    if has_expert_map:
        local_expert = tl.load(expert_map + expert, mask=live, other=-1).to(tl.int32)
        live &= local_expert >= 0
    current_epoch = tl.load(epoch).to(tl.uint64)
    key = current_epoch ^ route.to(tl.uint64) ^ (expert.to(tl.uint64) << 20)
    selected = (_mix64(key) % sample_rate) == 0
    live &= selected
    slot = tl.atomic_add(cursor, 1, mask=live)
    accepted = live & (slot < capacity)
    tl.store(slots + route, tl.where(accepted, slot, -1))
    if live & ~accepted:
        tl.atomic_add(dropped, 1)
    if accepted:
        weight = tl.load(topk_weights + route).to(tl.float32)
        observation = (
            (current_epoch << 32)
            | (token.to(tl.uint64) << 8)
            | route_slot.to(tl.uint64)
        )
        tl.store(sample_weight + slot, weight * weight)
        tl.store(sample_observation + slot, observation.to(tl.int64))
        tl.store(sample_expert + slot, expert)
        token_observation = (current_epoch << 32) | token.to(tl.uint64)
        split_key = token_observation ^ 0x6A09E667F3BCC909
        split = (_mix64(split_key) % validation_modulus) == 0
        tl.store(sample_split + slot, split.to(tl.int8))


def collect_route_input(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    is_padding: torch.Tensor,
    buffers: RouteInputBuffers,
    *,
    num_experts: int,
    expert_begin: int,
    expert_end: int,
    moment_sample_rate: int,
    hessian_sample_rate: int,
    validation_modulus: int,
    collect_routing: bool,
) -> None:
    """Collect exact routing, input diagonal moments, and raw H13 samples."""

    if x.ndim != 2 or topk_ids.shape != topk_weights.shape:
        raise ValueError("invalid route/input calibration shapes")
    if topk_ids.shape[0] != x.shape[0] or is_padding.numel() < x.shape[0]:
        raise ValueError("route/input calibration token dimensions differ")
    m, width = map(int, x.shape)
    topk = int(topk_ids.shape[1])
    if topk > 32:
        raise ValueError("raw route sampling supports at most 32 experts per token")
    _advance_epoch_kernel[(1,)](buffers.enabled, buffers.epoch_counter, buffers.epoch)
    _route_stats_kernel[(m,)](
        topk_ids,
        topk_weights,
        is_padding,
        buffers.enabled,
        buffers.tokens_routed,
        buffers.gate_sum,
        buffers.gate_sq_sum,
        num_tokens=m,
        topk=topk,
        num_experts=int(num_experts),
        collect_routing=bool(collect_routing),
        BLOCK_TOPK=triton.next_power_of_2(topk),
    )
    block = 128
    _input_moment_kernel[(m, triton.cdiv(width, block))](
        x,
        topk_ids,
        topk_weights,
        is_padding,
        buffers.enabled,
        buffers.epoch,
        buffers.input_sq_sum,
        buffers.input_weight_sum,
        buffers.input_count,
        num_tokens=m,
        topk=topk,
        width=width,
        expert_begin=int(expert_begin),
        expert_end=int(expert_end),
        sample_rate=max(int(moment_sample_rate), 1),
        BLOCK=block,
    )
    if buffers.sample_values.numel():
        capacity = int(buffers.sample_values.shape[0])
        if tuple(buffers.sample_experts.shape) != (capacity, topk):
            raise ValueError("sample_experts must have shape [capacity, topk]")
        if tuple(buffers.sample_gates.shape) != (capacity, topk):
            raise ValueError("sample_gates must have shape [capacity, topk]")
        if buffers.sample_split.numel() != capacity:
            raise ValueError("sample_split must have one entry per sample slot")
        buffers.sample_slots[:m].fill_(-1)
        _input_sample_meta_kernel[(m,)](
            topk_ids,
            topk_weights,
            is_padding,
            buffers.enabled,
            buffers.epoch,
            buffers.sample_cursor,
            buffers.sample_dropped,
            buffers.sample_slots,
            buffers.sample_weight,
            buffers.sample_observation,
            buffers.sample_experts,
            buffers.sample_gates,
            buffers.sample_split,
            num_tokens=m,
            topk=topk,
            capacity=capacity,
            sample_rate=max(int(hessian_sample_rate), 1),
            validation_modulus=max(int(validation_modulus), 1),
        )
        _sample_values_kernel[(m, triton.cdiv(width, block))](
            x,
            buffers.sample_slots,
            buffers.sample_values,
            rows=m,
            width=width,
            source_stride=width,
            BLOCK=block,
        )


def collect_paired_token_rows(
    source: torch.Tensor,
    sample_slots: torch.Tensor,
    sample_values: torch.Tensor,
    sample_ready: torch.Tensor,
) -> None:
    """Copy rows selected by a preceding ``collect_route_input`` call."""

    if source.ndim != 2 or sample_values.ndim != 2:
        raise ValueError("paired token samples require rank-2 tensors")
    rows, width = map(int, source.shape)
    if int(sample_slots.numel()) < rows:
        raise ValueError("sample_slots is smaller than the paired source")
    if int(sample_values.shape[1]) != width:
        raise ValueError("paired sample width differs from its source")
    if int(sample_ready.numel()) != int(sample_values.shape[0]):
        raise ValueError("sample_ready must have one entry per sample slot")
    if not sample_values.numel():
        return
    block = 128
    _paired_sample_values_kernel[(rows, triton.cdiv(width, block))](
        source,
        sample_slots,
        sample_values,
        sample_ready,
        rows=rows,
        width=width,
        BLOCK=block,
    )


def unscale_route_rows_(
    values: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor,
    scales: torch.Tensor,
    *,
    num_experts: int,
    scale_stride: int,
    scale_offset: int = 0,
) -> None:
    """Undo an expert-local diagonal scale on route-major rows in place."""

    if values.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("route unscale requires rank-2 values and top-k IDs")
    routes = int(topk_ids.numel())
    width = int(values.shape[1])
    if int(values.shape[0]) != routes:
        raise ValueError("route unscale values must contain one row per top-k route")
    if int(expert_map.numel()) != int(num_experts):
        raise ValueError("expert_map length must equal num_experts")
    if int(scale_stride) < width or int(scale_offset) < 0:
        raise ValueError("invalid route unscale stride/offset")
    if scales.numel() < int(scale_offset) + width:
        raise ValueError("route unscale scale table is too small")
    block = 128
    _unscale_route_rows_kernel[(routes, triton.cdiv(width, block))](
        values,
        topk_ids,
        expert_map,
        scales,
        num_routes=routes,
        num_experts=int(num_experts),
        width=width,
        scale_stride=int(scale_stride),
        scale_offset=int(scale_offset),
        BLOCK=block,
    )


def collect_mid(
    source: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    is_padding: torch.Tensor,
    buffers: MidBuffers,
    *,
    num_experts: int,
    width: int,
    source_stride: int,
    moment_sample_rate: int,
    hessian_sample_rate: int,
    validation_modulus: int,
) -> None:
    """Collect canonical w2-input moments/samples from route-indexed scratch."""

    m, topk = map(int, topk_ids.shape)
    routes = m * topk
    if topk_weights.shape != topk_ids.shape or is_padding.numel() < m:
        raise ValueError("invalid mid calibration routing shapes")
    if source.numel() < routes * int(source_stride):
        raise ValueError("canonical activation source is smaller than route geometry")
    has_map = expert_map is not None
    if expert_map is None:
        expert_map = buffers.sample_expert  # valid non-null placeholder
    elif int(expert_map.numel()) != int(num_experts):
        raise ValueError("expert_map length must equal num_experts")
    block = 128
    _mid_moment_kernel[(m, triton.cdiv(int(width), block))](
        source,
        topk_ids,
        topk_weights,
        expert_map,
        is_padding,
        buffers.enabled,
        buffers.epoch,
        buffers.mid_sq_sum,
        buffers.mid_weight_sum,
        buffers.mid_count,
        num_tokens=m,
        topk=topk,
        num_experts=int(num_experts),
        width=int(width),
        source_stride=int(source_stride),
        sample_rate=max(int(moment_sample_rate), 1),
        has_expert_map=has_map,
        BLOCK=block,
    )
    if buffers.sample_values.numel():
        capacity = int(buffers.sample_values.shape[0])
        if buffers.sample_split.numel() != capacity:
            raise ValueError("sample_split must have one entry per sample slot")
        buffers.sample_slots[:routes].fill_(-1)
        _mid_sample_meta_kernel[(routes,)](
            topk_ids,
            topk_weights,
            expert_map,
            is_padding,
            buffers.enabled,
            buffers.epoch,
            buffers.sample_cursor,
            buffers.sample_dropped,
            buffers.sample_slots,
            buffers.sample_weight,
            buffers.sample_observation,
            buffers.sample_expert,
            buffers.sample_split,
            num_routes=routes,
            topk=topk,
            num_experts=int(num_experts),
            capacity=capacity,
            sample_rate=max(int(hessian_sample_rate), 1),
            validation_modulus=max(int(validation_modulus), 1),
            has_expert_map=has_map,
        )
        _sample_values_kernel[(routes, triton.cdiv(int(width), block))](
            source,
            buffers.sample_slots,
            buffers.sample_values,
            rows=routes,
            width=int(width),
            source_stride=int(source_stride),
            BLOCK=block,
        )


__all__ = [
    "MidBuffers",
    "RouteInputBuffers",
    "collect_mid",
    "collect_paired_token_rows",
    "collect_route_input",
    "unscale_route_rows_",
]
