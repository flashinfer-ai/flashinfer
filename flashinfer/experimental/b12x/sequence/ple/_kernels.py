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


from .._short_conv import (
    _prepare_history_kernel,
    _request_ids_kernel,
    _short_conv_kernel,
    _update_state_kernel,
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
    _short_conv_kernel(
        normalized_u_ptr,
        gathered_state_ptr,
        conv_weight_ptr,
        query_start_loc_ptr,
        request_ids_ptr,
        state_slot_ids_ptr,
        num_tokens_ptr,
        out_ptr,
        CHANNELS,
        STATE_LENGTH,
        KERNEL_SIZE,
        DILATION,
        BLOCK_C,
        ADD_RESIDUAL=True,
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
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    conv_state: torch.Tensor,
    out: torch.Tensor,
    normalized_u: torch.Tensor,
    gathered_state: torch.Tensor,
    request_ids: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    state = require_prepared(
        plan_from_handle(plan_handle), "sequence.ple", residual.device
    )
    if state.mixed:
        raise ValueError("mixed PLE plan requires the mixed operator")
    state.run_tensors(
        residual,
        key,
        value,
        k_norm_weight,
        q_norm_weight,
        u_norm_weight,
        conv_weight,
        query_start_loc,
        state_slot_ids,
        state_is_fresh,
        num_accepted_tokens,
        state_is_fresh,
        num_seqs,
        num_tokens,
        conv_state,
        out,
        normalized_u,
        gathered_state,
        request_ids,
        eps=eps,
    )


@_layer_pipeline_op.register_fake
def _layer_pipeline_fake(
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    conv_state: torch.Tensor,
    out: torch.Tensor,
    normalized_u: torch.Tensor,
    gathered_state: torch.Tensor,
    request_ids: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    del residual, key, value, k_norm_weight, q_norm_weight, u_norm_weight, conv_weight
    del (
        query_start_loc,
        state_slot_ids,
        state_is_fresh,
        num_accepted_tokens,
        num_seqs,
        num_tokens,
    )
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
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    request_is_prefill: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    conv_state: torch.Tensor,
    out: torch.Tensor,
    normalized_u: torch.Tensor,
    gathered_state: torch.Tensor,
    request_ids: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    state = require_prepared(
        plan_from_handle(plan_handle), "sequence.ple", residual.device
    )
    if not state.mixed:
        raise ValueError("mixed PLE operator requires a mixed plan")
    state.run_tensors(
        residual,
        key,
        value,
        k_norm_weight,
        q_norm_weight,
        u_norm_weight,
        conv_weight,
        query_start_loc,
        state_slot_ids,
        state_is_fresh,
        num_accepted_tokens,
        request_is_prefill,
        num_seqs,
        num_tokens,
        conv_state,
        out,
        normalized_u,
        gathered_state,
        request_ids,
        eps=eps,
    )


@_layer_mixed_pipeline_op.register_fake
def _layer_mixed_pipeline_fake(
    residual: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    u_norm_weight: torch.Tensor,
    conv_weight: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_slot_ids: torch.Tensor,
    state_is_fresh: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    request_is_prefill: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    conv_state: torch.Tensor,
    out: torch.Tensor,
    normalized_u: torch.Tensor,
    gathered_state: torch.Tensor,
    request_ids: torch.Tensor,
    eps: float,
    plan_handle: int,
) -> None:
    del residual, key, value, k_norm_weight, q_norm_weight, u_norm_weight, conv_weight
    del (
        query_start_loc,
        state_slot_ids,
        state_is_fresh,
        num_accepted_tokens,
        request_is_prefill,
    )
    del (
        num_seqs,
        num_tokens,
        conv_state,
        out,
        normalized_u,
        gathered_state,
        request_ids,
        eps,
        plan_handle,
    )


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
