"""Triton kernels for projected PLE math and short-convolution state."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from b12x.preparation.types import plan_from_handle, require_prepared

if TYPE_CHECKING:
    from ._contracts import LayerBinding

_CHANNEL_BLOCK = 128


@triton.jit
def _request_ids_kernel(
    query_start_loc_ptr,
    num_seqs_ptr,
    num_tokens_ptr,
    request_ids_ptr,
    MAX_TOKENS: tl.constexpr,
):
    token = tl.program_id(0)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
    live = token < num_tokens

    low = tl.zeros((), tl.int32)
    high = tl.maximum(num_seqs, 1)
    while low + 1 < high:
        middle = (low + high) // 2
        start = tl.load(query_start_loc_ptr + middle, mask=live, other=0).to(tl.int32)
        low = tl.where(start <= token, middle, low)
        high = tl.where(start <= token, high, middle)
    request = tl.where(live & (num_seqs > 0), low, -1)
    tl.store(request_ids_ptr + token, request)


@triton.jit
def _prepare_history_kernel(
    conv_state_ptr,
    state_slot_ids_ptr,
    state_is_fresh_ptr,
    num_accepted_tokens_ptr,
    request_is_prefill_ptr,
    num_seqs_ptr,
    gathered_state_ptr,
    CHANNELS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_channel: tl.constexpr,
    stride_state_position: tl.constexpr,
    MAX_SPECULATIVE: tl.constexpr,
    DECODE: tl.constexpr,
    MIXED: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    request = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < CHANNELS
    num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
    request_live = request < num_seqs
    slot = tl.load(state_slot_ids_ptr + request, mask=request_live, other=-1).to(
        tl.int64
    )
    fresh = tl.load(state_is_fresh_ptr + request, mask=request_live, other=True).to(
        tl.int1
    )
    live = request_live & (slot >= 0)
    if MIXED:
        request_decode = request_live & ~tl.load(
            request_is_prefill_ptr + request,
            mask=request_live,
            other=True,
        ).to(tl.int1)
    else:
        request_decode = request_live & DECODE
    if DECODE or MIXED:
        accepted = tl.load(
            num_accepted_tokens_ptr + request, mask=request_live, other=0
        ).to(tl.int32)
        rollback = tl.minimum(accepted - 1, MAX_SPECULATIVE)
        rollback = tl.where(request_decode, rollback, 0)
    else:
        rollback = tl.zeros((), tl.int32)

    for position in tl.static_range(0, STATE_LENGTH):
        source_position = position + rollback
        source_offsets = (
            slot.to(tl.int64) * stride_state_slot
            + channels.to(tl.int64) * stride_state_channel
            + source_position.to(tl.int64) * stride_state_position
        )
        value = tl.load(
            conv_state_ptr + source_offsets,
            mask=live & channel_mask & ~fresh & (source_position < STATE_CAPACITY),
            other=0.0,
        )
        gathered_offsets = (
            request.to(tl.int64) * CHANNELS + channels.to(tl.int64)
        ) * STATE_LENGTH + position
        tl.store(
            gathered_state_ptr + gathered_offsets,
            value,
            mask=channel_mask,
        )


@triton.jit
def _gated_u_norm_kernel(
    residual_ptr,
    key_ptr,
    value_ptr,
    k_norm_weight_ptr,
    q_norm_weight_ptr,
    u_norm_weight_ptr,
    request_ids_ptr,
    state_slot_ids_ptr,
    num_tokens_ptr,
    out_ptr,
    normalized_u_ptr,
    eps,
    STREAMS: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    CHANNELS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0)
    stream = tl.program_id(1)
    columns = tl.arange(0, BLOCK_H)
    column_mask = columns < HIDDEN_SIZE
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    token_live = token < num_tokens
    request = tl.load(request_ids_ptr + token).to(tl.int32)
    slot = tl.load(
        state_slot_ids_ptr + request,
        mask=token_live & (request >= 0),
        other=-1,
    ).to(tl.int64)
    live = token_live & (request >= 0) & (slot >= 0)
    offsets = (
        token.to(tl.int64) * CHANNELS + stream * HIDDEN_SIZE + columns.to(tl.int64)
    )
    weight_offsets = stream * HIDDEN_SIZE + columns

    key_value = tl.load(key_ptr + offsets, mask=column_mask & live, other=0.0).to(
        tl.float32
    )
    query_value = tl.load(
        residual_ptr + offsets, mask=column_mask & live, other=0.0
    ).to(tl.float32)
    k_weight = tl.load(
        k_norm_weight_ptr + weight_offsets,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)
    q_weight = tl.load(
        q_norm_weight_ptr + weight_offsets,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)
    key_inv_rms = tl.rsqrt(tl.sum(key_value * key_value, axis=0) / HIDDEN_SIZE + eps)
    query_inv_rms = tl.rsqrt(
        tl.sum(query_value * query_value, axis=0) / HIDDEN_SIZE + eps
    )
    key_normalized = (key_value * key_inv_rms * (1.0 + k_weight)).to(tl.bfloat16)
    query_normalized = (query_value * query_inv_rms * (1.0 + q_weight)).to(tl.bfloat16)
    similarity = tl.sum(
        key_normalized.to(tl.float32) * query_normalized.to(tl.float32), axis=0
    ) / tl.sqrt(float(HIDDEN_SIZE))
    sign = tl.where(similarity > 0.0, 1.0, tl.where(similarity < 0.0, -1.0, 0.0))
    warped = sign * tl.sqrt(tl.maximum(tl.abs(similarity), 1e-6))
    gate = tl.sigmoid(warped).to(tl.bfloat16)

    value_offsets = token.to(tl.int64) * HIDDEN_SIZE + columns.to(tl.int64)
    projected_value = tl.load(
        value_ptr + value_offsets,
        mask=column_mask & live,
        other=0.0,
    ).to(tl.bfloat16)
    u = (gate * projected_value).to(tl.bfloat16)
    tl.store(out_ptr + offsets, tl.where(live, u, 0.0), mask=column_mask)

    u_weight = tl.load(
        u_norm_weight_ptr + weight_offsets,
        mask=column_mask,
        other=0.0,
    ).to(tl.float32)
    u_float = u.to(tl.float32)
    u_inv_rms = tl.rsqrt(tl.sum(u_float * u_float, axis=0) / HIDDEN_SIZE + eps)
    u_normalized = (u_float * u_inv_rms * (1.0 + u_weight)).to(tl.bfloat16)
    tl.store(
        normalized_u_ptr + offsets,
        tl.where(live, u_normalized, 0.0),
        mask=column_mask,
    )


@triton.jit
def _dilated_conv_kernel(
    normalized_u_ptr,
    gathered_state_ptr,
    conv_weight_ptr,
    query_start_loc_ptr,
    request_ids_ptr,
    state_slot_ids_ptr,
    num_tokens_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < CHANNELS
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    token_live = token < num_tokens
    request = tl.load(request_ids_ptr + token).to(tl.int32)
    slot = tl.load(
        state_slot_ids_ptr + request,
        mask=token_live & (request >= 0),
        other=-1,
    ).to(tl.int64)
    live = token_live & (request >= 0) & (slot >= 0)
    query_start = tl.load(
        query_start_loc_ptr + request,
        mask=live,
        other=0,
    ).to(tl.int32)
    query_relative = token - query_start

    accumulator = tl.zeros((BLOCK_C,), tl.float32)
    for tap in tl.static_range(0, KERNEL_SIZE):
        history_distance = (KERNEL_SIZE - 1 - tap) * DILATION
        source_relative = query_relative - history_distance
        from_query = source_relative >= 0
        query_token = query_start + source_relative
        query_offsets = query_token.to(tl.int64) * CHANNELS + channels.to(tl.int64)
        query_value = tl.load(
            normalized_u_ptr + query_offsets,
            mask=live & channel_mask & from_query,
            other=0.0,
        ).to(tl.float32)
        history_position = STATE_LENGTH + source_relative
        history_offsets = (
            request.to(tl.int64) * CHANNELS + channels.to(tl.int64)
        ) * STATE_LENGTH + history_position
        history_value = tl.load(
            gathered_state_ptr + history_offsets,
            mask=(
                live
                & channel_mask
                & ~from_query
                & (history_position >= 0)
                & (history_position < STATE_LENGTH)
            ),
            other=0.0,
        ).to(tl.float32)
        source = tl.where(from_query, query_value, history_value)
        weight = tl.load(
            conv_weight_ptr + channels.to(tl.int64) * KERNEL_SIZE + tap,
            mask=channel_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += source * weight

    conv_bf16 = accumulator.to(tl.bfloat16)
    conv_float = conv_bf16.to(tl.float32)
    activated = (conv_float * tl.sigmoid(conv_float)).to(tl.bfloat16)
    output_offsets = token.to(tl.int64) * CHANNELS + channels.to(tl.int64)
    u = tl.load(out_ptr + output_offsets, mask=channel_mask, other=0.0).to(tl.bfloat16)
    contribution = (u + activated).to(tl.bfloat16)
    tl.store(
        out_ptr + output_offsets,
        tl.where(live, contribution, 0.0),
        mask=channel_mask,
    )


@triton.jit
def _update_state_kernel(
    normalized_u_ptr,
    gathered_state_ptr,
    query_start_loc_ptr,
    state_slot_ids_ptr,
    request_is_prefill_ptr,
    num_seqs_ptr,
    conv_state_ptr,
    CHANNELS: tl.constexpr,
    STATE_LENGTH: tl.constexpr,
    STATE_CAPACITY: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_channel: tl.constexpr,
    stride_state_position: tl.constexpr,
    MAX_SPECULATIVE: tl.constexpr,
    DECODE: tl.constexpr,
    MIXED: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    request = tl.program_id(0)
    channel_block = tl.program_id(1)
    channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
    channel_mask = channels < CHANNELS
    num_seqs = tl.load(num_seqs_ptr).to(tl.int32)
    request_live = request < num_seqs
    slot = tl.load(state_slot_ids_ptr + request, mask=request_live, other=-1).to(
        tl.int64
    )
    live = request_live & (slot >= 0)
    query_start = tl.load(query_start_loc_ptr + request, mask=request_live, other=0).to(
        tl.int32
    )
    query_end = tl.load(
        query_start_loc_ptr + request + 1, mask=request_live, other=0
    ).to(tl.int32)
    query_length = query_end - query_start
    if MIXED:
        request_decode = request_live & ~tl.load(
            request_is_prefill_ptr + request,
            mask=request_live,
            other=True,
        ).to(tl.int1)
    else:
        request_decode = request_live & DECODE

    for position in tl.static_range(0, STATE_LENGTH):
        if DECODE or MIXED:
            commit_current = tl.where(query_length > 0, 1, 0)
            decode_source_position = position + commit_current
            prefill_source_position = position + query_length
            source_position = tl.where(
                request_decode,
                decode_source_position,
                prefill_source_position,
            )
        else:
            source_position = position + query_length
        from_query = source_position >= STATE_LENGTH
        query_relative = source_position - STATE_LENGTH
        query_token = query_start + query_relative
        query_offsets = query_token.to(tl.int64) * CHANNELS + channels.to(tl.int64)
        query_value = tl.load(
            normalized_u_ptr + query_offsets,
            mask=(
                live
                & channel_mask
                & from_query
                & (query_relative >= 0)
                & (query_relative < query_length)
            ),
            other=0.0,
        )
        history_offsets = (
            request.to(tl.int64) * CHANNELS + channels.to(tl.int64)
        ) * STATE_LENGTH + source_position
        history_value = tl.load(
            gathered_state_ptr + history_offsets,
            mask=live & channel_mask & ~from_query,
            other=0.0,
        )
        value = tl.where(from_query, query_value, history_value)
        destination_offsets = (
            slot.to(tl.int64) * stride_state_slot
            + channels.to(tl.int64) * stride_state_channel
            + position * stride_state_position
        )
        tl.store(
            conv_state_ptr + destination_offsets,
            value,
            mask=live & channel_mask & (query_length > 0),
        )

    for candidate in tl.static_range(0, MAX_SPECULATIVE):
        if DECODE or MIXED:
            query_relative = candidate + 1
            query_token = query_start + query_relative
            query_offsets = query_token.to(tl.int64) * CHANNELS + channels.to(tl.int64)
            candidate_value = tl.load(
                normalized_u_ptr + query_offsets,
                mask=(
                    live
                    & request_decode
                    & channel_mask
                    & (query_relative < query_length)
                ),
                other=0.0,
            )
        else:
            candidate_value = tl.zeros((BLOCK_C,), tl.float32)
        destination_offsets = (
            slot.to(tl.int64) * stride_state_slot
            + channels.to(tl.int64) * stride_state_channel
            + (STATE_LENGTH + candidate) * stride_state_position
        )
        tl.store(
            conv_state_ptr + destination_offsets,
            candidate_value,
            mask=live & channel_mask & (query_length > 0),
        )




@torch.library.custom_op(
    "b12x::ple_layer_pipeline",
    mutates_args=(
        "conv_state",
        "out",
        "normalized_u",
        "gathered_state",
        "request_ids",
    ),
)
def _layer_pipeline_op(
    residual: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    k_norm_weight: torch.Tensor, q_norm_weight: torch.Tensor, u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor, query_start_loc: torch.Tensor, state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor, num_accepted_tokens: torch.Tensor,
    num_seqs: torch.Tensor, num_tokens: torch.Tensor, conv_state: torch.Tensor,
    out: torch.Tensor, normalized_u: torch.Tensor, gathered_state: torch.Tensor,
    request_ids: torch.Tensor, eps: float, plan_handle: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "sequence.ple", residual.device)
    if state.mixed:
        raise ValueError("mixed PLE plan requires the mixed operator")
    state.run_tensors(
        residual, key, value, k_norm_weight, q_norm_weight, u_norm_weight,
        conv_weight, query_start_loc, state_slot_ids, state_is_fresh, num_accepted_tokens,
        state_is_fresh, num_seqs, num_tokens, conv_state, out, normalized_u, gathered_state,
        request_ids, eps=eps,
    )


@_layer_pipeline_op.register_fake
def _layer_pipeline_fake(
    residual: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    k_norm_weight: torch.Tensor, q_norm_weight: torch.Tensor, u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor, query_start_loc: torch.Tensor, state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor, num_accepted_tokens: torch.Tensor,
    num_seqs: torch.Tensor, num_tokens: torch.Tensor, conv_state: torch.Tensor,
    out: torch.Tensor, normalized_u: torch.Tensor, gathered_state: torch.Tensor,
    request_ids: torch.Tensor, eps: float, plan_handle: int,
) -> None:
    del residual, key, value, k_norm_weight, q_norm_weight, u_norm_weight, conv_weight
    del query_start_loc, state_slot_ids, state_is_fresh, num_accepted_tokens, num_seqs, num_tokens
    del conv_state, out, normalized_u, gathered_state, request_ids, eps, plan_handle


@torch.library.custom_op(
    "b12x::ple_layer_mixed_pipeline",
    mutates_args=(
        "conv_state",
        "out",
        "normalized_u",
        "gathered_state",
        "request_ids",
    ),
)
def _layer_mixed_pipeline_op(
    residual: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    k_norm_weight: torch.Tensor, q_norm_weight: torch.Tensor, u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor, query_start_loc: torch.Tensor, state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor, num_accepted_tokens: torch.Tensor, request_is_prefill: torch.Tensor,
    num_seqs: torch.Tensor, num_tokens: torch.Tensor, conv_state: torch.Tensor,
    out: torch.Tensor, normalized_u: torch.Tensor, gathered_state: torch.Tensor,
    request_ids: torch.Tensor, eps: float, plan_handle: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "sequence.ple", residual.device)
    if not state.mixed:
        raise ValueError("mixed PLE operator requires a mixed plan")
    state.run_tensors(
        residual, key, value, k_norm_weight, q_norm_weight, u_norm_weight,
        conv_weight, query_start_loc, state_slot_ids, state_is_fresh, num_accepted_tokens,
        request_is_prefill, num_seqs, num_tokens, conv_state, out, normalized_u, gathered_state,
        request_ids, eps=eps,
    )


@_layer_mixed_pipeline_op.register_fake
def _layer_mixed_pipeline_fake(
    residual: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    k_norm_weight: torch.Tensor, q_norm_weight: torch.Tensor, u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor, query_start_loc: torch.Tensor, state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor, num_accepted_tokens: torch.Tensor, request_is_prefill: torch.Tensor,
    num_seqs: torch.Tensor, num_tokens: torch.Tensor, conv_state: torch.Tensor,
    out: torch.Tensor, normalized_u: torch.Tensor, gathered_state: torch.Tensor,
    request_ids: torch.Tensor, eps: float, plan_handle: int,
) -> None:
    del residual, key, value, k_norm_weight, q_norm_weight, u_norm_weight, conv_weight
    del query_start_loc, state_slot_ids, state_is_fresh, num_accepted_tokens, request_is_prefill
    del num_seqs, num_tokens, conv_state, out, normalized_u, gathered_state, request_ids, eps, plan_handle


def run_layer_kernels(binding: LayerBinding, *, eps: float) -> None:
    """Dispatch the opaque, mutation-declared PLE pipeline."""
    if binding.plan is None:
        raise TypeError("PLE plan requires a prepared binding")
    torch.ops.b12x.ple_layer_pipeline(
        binding.residual,
        binding.key,
        binding.value,
        binding.k_norm_weight,
        binding.q_norm_weight,
        binding.u_norm_weight,
        binding.conv_weight,
        binding.query_start_loc,
        binding.state_slot_ids,
        binding.state_is_fresh,
        binding.num_accepted_tokens,
        binding.num_seqs,
        binding.num_tokens,
        binding.conv_state,
        binding.out,
        binding.normalized_u,
        binding.gathered_state,
        binding.request_ids,
        eps,
        binding.plan.handle,
    )


def run_layer_mixed_kernels(
    binding: LayerBinding, *, eps: float, token_count: int
) -> None:
    """Dispatch the opaque mixed packed PLE pipeline."""
    if binding.plan is None:
        raise TypeError("PLE plan requires a prepared binding")
    request_is_prefill = binding.request_is_prefill
    assert request_is_prefill is not None
    torch.ops.b12x.ple_layer_mixed_pipeline(
        binding.residual[:token_count],
        binding.key,
        binding.value,
        binding.k_norm_weight,
        binding.q_norm_weight,
        binding.u_norm_weight,
        binding.conv_weight,
        binding.query_start_loc,
        binding.state_slot_ids,
        binding.state_is_fresh,
        binding.num_accepted_tokens,
        request_is_prefill,
        binding.num_seqs,
        binding.num_tokens,
        binding.conv_state,
        binding.out,
        binding.normalized_u,
        binding.gathered_state,
        binding.request_ids,
        eps,
        binding.plan.handle,
    )


__all__ = ["run_layer_kernels", "run_layer_mixed_kernels"]
