"""Opaque launches for packed sequential GDN decode and output gating."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from b12x.preparation.types import plan_from_handle, require_prepared


@triton.jit(
    do_not_specialize=[
        "sequence_capacity",
        "state_index_columns",
        "stride_indices_request",
        "stride_indices_column",
    ],
    do_not_specialize_on_alignment=["state_indices"],
)
def _packed_sequential_kda_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    recurrent_state,
    query_start_loc,
    num_accepted_tokens,
    state_indices,
    num_seqs,
    output,
    scale,
    lower_bound,
    sequence_capacity,
    state_index_columns,
    stride_mixed_token: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_a_head: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_b_head: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_state_head: tl.constexpr,
    stride_state_v: tl.constexpr,
    stride_indices_request,
    stride_indices_column,
    stride_output_token: tl.constexpr,
    stride_output_head: tl.constexpr,
    MAX_SEQS: tl.constexpr,
    KEY_HEADS: tl.constexpr,
    VALUE_HEADS: tl.constexpr,
    KEY_HEAD_DIM: tl.constexpr,
    VALUE_HEAD_DIM: tl.constexpr,
    STATE_INDEX_COLUMNS: tl.constexpr,
    BLOCK_V: tl.constexpr,
    QK_L2NORM: tl.constexpr,
    HAS_NULL_STATE_INDEX: tl.constexpr,
    NULL_STATE_INDEX: tl.constexpr,
):
    value_tile = tl.program_id(0)
    request_value_head = tl.program_id(1)
    request = request_value_head // VALUE_HEADS
    value_head = request_value_head % VALUE_HEADS
    key_head = value_head

    live_seqs = tl.load(num_seqs).to(tl.int32)
    if request >= tl.maximum(0, tl.minimum(live_seqs, sequence_capacity)):
        return

    start = tl.load(query_start_loc + request).to(tl.int32)
    end = tl.load(query_start_loc + request + 1).to(tl.int32)
    if end <= start:
        return
    accepted_column = tl.load(num_accepted_tokens + request).to(tl.int32) - 1
    source_idx = tl.load(
        state_indices
        + request.to(tl.int64) * stride_indices_request
        + accepted_column.to(tl.int64) * stride_indices_column
    ).to(tl.int64)

    key_cols = tl.arange(0, KEY_HEAD_DIM)
    value_rows = value_tile * BLOCK_V + tl.arange(0, BLOCK_V)
    key_mask = key_cols < KEY_HEAD_DIM
    value_mask = value_rows < VALUE_HEAD_DIM
    state_mask = value_mask[:, None] & key_mask[None, :]

    if HAS_NULL_STATE_INDEX:
        if source_idx == NULL_STATE_INDEX:
            for relative_token in range(STATE_INDEX_COLUMNS):
                if (relative_token < state_index_columns) & (
                    relative_token < (end - start)
                ):
                    token = start + relative_token
                    output_offsets = (
                        token.to(tl.int64) * stride_output_token
                        + value_head.to(tl.int64) * stride_output_head
                        + value_rows.to(tl.int64)
                    )
                    tl.store(output + output_offsets, 0.0, mask=value_mask)
            return

    # Pool-scaled arithmetic is explicitly Int64. Valid serving pools can place
    # recycled state slots beyond the signed-32-bit element-offset boundary.
    source_offsets = (
        source_idx * stride_state_slot
        + value_head.to(tl.int64) * stride_state_head
        + value_rows[:, None].to(tl.int64) * stride_state_v
        + key_cols[None, :].to(tl.int64)
    )
    state = tl.load(recurrent_state + source_offsets, mask=state_mask, other=0.0).to(
        tl.float32
    )

    for relative_token in range(STATE_INDEX_COLUMNS):
        if (relative_token < state_index_columns) & (relative_token < (end - start)):
            token = start + relative_token
            token_i64 = token.to(tl.int64)
            mixed_base = token_i64 * stride_mixed_token
            q_offsets = key_head * KEY_HEAD_DIM + key_cols
            k_offsets = KEY_HEADS * KEY_HEAD_DIM + q_offsets
            v_offsets = (
                2 * KEY_HEADS * KEY_HEAD_DIM + value_head * VALUE_HEAD_DIM + value_rows
            )
            q = tl.load(
                mixed_qkv + mixed_base + q_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            k = tl.load(
                mixed_qkv + mixed_base + k_offsets, mask=key_mask, other=0.0
            ).to(tl.float32)
            value = tl.load(
                mixed_qkv + mixed_base + v_offsets, mask=value_mask, other=0.0
            ).to(tl.float32)
            if QK_L2NORM:
                q = q * tl.rsqrt(tl.sum(q * q, axis=0) + 1.0e-6)
                k = k * tl.rsqrt(tl.sum(k * k, axis=0) + 1.0e-6)
            q *= scale

            b_value = tl.load(
                b + token_i64 * stride_b_token + value_head.to(tl.int64) * stride_b_head
            ).to(tl.float32)
            A_log_value = tl.load(A_log + value_head).to(tl.float32)
            raw_gate = tl.load(
                a
                + token_i64 * stride_a_token
                + value_head.to(tl.int64) * stride_a_head
                + key_cols.to(tl.int64),
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            gate_bias = tl.load(
                dt_bias
                + value_head.to(tl.int64) * stride_dt_bias_head
                + key_cols.to(tl.int64),
                mask=key_mask,
                other=0.0,
            ).to(tl.float32)
            log_decay = lower_bound * tl.sigmoid(
                tl.exp(A_log_value) * (raw_gate + gate_bias)
            )
            state *= tl.exp(log_decay)[None, :]
            beta = tl.sigmoid(b_value)

            value -= tl.sum(state * k[None, :], axis=1)
            value *= beta
            state += value[:, None] * k[None, :]
            decoded = tl.sum(state * q[None, :], axis=1)
            output_offsets = (
                token_i64 * stride_output_token
                + value_head.to(tl.int64) * stride_output_head
                + value_rows.to(tl.int64)
            )
            tl.store(output + output_offsets, decoded, mask=value_mask)

            destination_idx = tl.load(
                state_indices
                + request.to(tl.int64) * stride_indices_request
                + relative_token * stride_indices_column
            ).to(tl.int64)
            destination_offsets = (
                destination_idx * stride_state_slot
                + value_head.to(tl.int64) * stride_state_head
                + value_rows[:, None].to(tl.int64) * stride_state_v
                + key_cols[None, :].to(tl.int64)
            )
            destination_mask = state_mask
            if HAS_NULL_STATE_INDEX:
                destination_mask &= destination_idx != NULL_STATE_INDEX
            tl.store(
                recurrent_state + destination_offsets,
                state,
                mask=destination_mask,
            )


@triton.jit(
    do_not_specialize=[
        "token_capacity",
        "stride_output_token",
        "stride_output_head",
        "stride_z_token",
        "stride_z_head",
    ],
    do_not_specialize_on_alignment=["output", "z"],
)
def _gated_rmsnorm_kernel(
    output,
    z,
    norm_weight,
    num_tokens,
    eps,
    token_capacity,
    stride_output_token: tl.int64,
    stride_output_head: tl.int64,
    stride_z_token: tl.int64,
    stride_z_head: tl.int64,
    VALUE_HEADS: tl.constexpr,
    VALUE_HEAD_DIM: tl.constexpr,
    SIGMOID_GATE: tl.constexpr,
    NORM_WEIGHT_FP32: tl.constexpr,
    KDA_NORM_FP32: tl.constexpr,
):
    token_value_head = tl.program_id(0)
    token = token_value_head // VALUE_HEADS
    value_head = token_value_head % VALUE_HEADS
    cols = tl.arange(0, VALUE_HEAD_DIM)
    mask = cols < VALUE_HEAD_DIM
    token_i64 = token.to(tl.int64)
    output_offsets = (
        token_i64 * stride_output_token
        + value_head.to(tl.int64) * stride_output_head
        + cols.to(tl.int64)
    )
    live_tokens = tl.load(num_tokens).to(tl.int32)
    if token >= tl.maximum(0, tl.minimum(live_tokens, token_capacity)):
        tl.store(output + output_offsets, 0.0, mask=mask)
        return

    z_offsets = (
        token_i64 * stride_z_token
        + value_head.to(tl.int64) * stride_z_head
        + cols.to(tl.int64)
    )
    values = tl.load(output + output_offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / VALUE_HEAD_DIM
    normalized = values * tl.rsqrt(variance + eps)
    if KDA_NORM_FP32:
        weight = tl.load(norm_weight + cols, mask=mask, other=0.0).to(tl.float32)
        weighted = normalized * weight
    elif NORM_WEIGHT_FP32:
        normalized = normalized.to(tl.bfloat16)
        weight = tl.load(norm_weight + cols, mask=mask, other=0.0).to(tl.float32)
        weighted = normalized.to(tl.float32) * weight
    else:
        normalized = normalized.to(tl.bfloat16)
        weight = tl.load(norm_weight + cols, mask=mask, other=0.0).to(tl.bfloat16)
        weighted = (normalized * weight).to(tl.bfloat16).to(tl.float32)
    gate_input = tl.load(z + z_offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.sigmoid(gate_input)
    if not SIGMOID_GATE:
        gate *= gate_input
    tl.store(output + output_offsets, weighted * gate, mask=mask)


@torch.library.custom_op(
    "b12x::gdn_decode",
    mutates_args=("recurrent_state", "output"),
)
def _gdn_decode_op(
    mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor,
    A_log: torch.Tensor, dt_bias: torch.Tensor, norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor, query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor, state_indices: torch.Tensor,
    num_seqs: torch.Tensor, num_tokens: torch.Tensor, output: torch.Tensor,
    eps: float, scale: float, lower_bound: float, plan_handle: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "attention.gdn", mixed_qkv.device)
    state.run_tensors(
        mixed_qkv, a, b, z, A_log, dt_bias, norm_weight, recurrent_state,
        query_start_loc, num_accepted_tokens, state_indices, num_seqs, num_tokens,
        output, eps=eps, scale=scale, lower_bound=lower_bound,
    )


@_gdn_decode_op.register_fake
def _gdn_decode_fake(
    mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor, z: torch.Tensor,
    A_log: torch.Tensor, dt_bias: torch.Tensor, norm_weight: torch.Tensor,
    recurrent_state: torch.Tensor, query_start_loc: torch.Tensor,
    num_accepted_tokens: torch.Tensor, state_indices: torch.Tensor,
    num_seqs: torch.Tensor, num_tokens: torch.Tensor, output: torch.Tensor,
    eps: float, scale: float, lower_bound: float, plan_handle: int,
) -> None:
    del mixed_qkv, a, b, z, A_log, dt_bias, norm_weight, recurrent_state
    del query_start_loc, num_accepted_tokens, state_indices, num_seqs, num_tokens
    del output, eps, scale, lower_bound, plan_handle


def run_gdn_decode(*tensors, eps, scale, lower_bound, plan):
    torch.ops.b12x.gdn_decode(*tensors, float(eps), float(scale), float(lower_bound), plan.handle)


__all__ = ["run_gdn_decode"]
