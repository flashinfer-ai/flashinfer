"""Triton kernels for packed EOS-bounded PLE hashing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from b12x.preparation.types import plan_from_handle, require_prepared

if TYPE_CHECKING:
    from ._contracts import Binding


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
def _source_token(
    token_ids_ptr,
    committed_history_ptr,
    query_start,
    request,
    query_relative,
    relative_position,
    eos_token_id,
    live,
    MAX_ORDER: tl.constexpr,
):
    source_relative = query_relative + relative_position
    from_query = source_relative >= 0
    query_index = query_start + source_relative
    history_index = (MAX_ORDER - 1) + source_relative
    query_value = tl.load(
        token_ids_ptr + query_index.to(tl.int64),
        mask=live & from_query,
        other=eos_token_id,
    ).to(tl.int64)
    history_offset = request.to(tl.int64) * (MAX_ORDER - 1) + history_index.to(tl.int64)
    history_value = tl.load(
        committed_history_ptr + history_offset,
        mask=live & ~from_query & (history_index >= 0),
        other=eos_token_id,
    ).to(tl.int64)
    return tl.where(from_query, query_value, history_value)


@triton.jit
def _hash_ids_kernel(
    token_ids_ptr,
    query_start_loc_ptr,
    committed_history_ptr,
    num_tokens_ptr,
    request_ids_ptr,
    multipliers_ptr,
    prime_sizes_ptr,
    table_offsets_ptr,
    out_ptr,
    eos_token_id,
    MAX_TOKENS: tl.constexpr,
    MAX_ORDER: tl.constexpr,
    HEADS_PER_ORDER: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    request = tl.load(request_ids_ptr + token).to(tl.int32)
    live = (token < num_tokens) & (request >= 0)
    query_start = tl.load(query_start_loc_ptr + request, mask=live, other=0).to(
        tl.int32
    )
    query_relative = token - query_start
    order = head // HEADS_PER_ORDER + 2
    mixed = tl.zeros((), tl.int64)

    for position in tl.static_range(0, MAX_ORDER):
        in_order = position < order
        # Checkpoint multiplier indices are token lags: index zero belongs
        # to the current token for every n-gram order.
        distance_from_current = position
        relative_position = -distance_from_current
        value = _source_token(
            token_ids_ptr,
            committed_history_ptr,
            query_start,
            request,
            query_relative,
            relative_position,
            eos_token_id,
            live & in_order,
            MAX_ORDER=MAX_ORDER,
        )
        bounded = tl.full((), True, tl.int1)
        for back in tl.static_range(1, MAX_ORDER):
            boundary_value = _source_token(
                token_ids_ptr,
                committed_history_ptr,
                query_start,
                request,
                query_relative,
                -back,
                eos_token_id,
                live & in_order & (back < distance_from_current),
                MAX_ORDER=MAX_ORDER,
            )
            is_boundary = (back < distance_from_current) & (
                boundary_value == eos_token_id
            )
            bounded &= ~is_boundary
        effective = tl.where(bounded, value, eos_token_id).to(tl.int64)
        multiplier = tl.load(
            multipliers_ptr + position,
            mask=in_order,
            other=1,
        ).to(tl.int64)
        product = effective * multiplier
        mixed ^= tl.where(in_order, product, 0)

    prime = tl.load(prime_sizes_ptr + head).to(tl.int64)
    table_offset = tl.load(table_offsets_ptr + head).to(tl.int64)
    remainder = mixed % prime
    remainder = tl.where(remainder < 0, remainder + prime, remainder)
    embedding_id = table_offset + remainder
    output_offset = token.to(tl.int64) * HEAD_COUNT + head
    tl.store(out_ptr + output_offset, tl.where(live, embedding_id, -1))


@torch.library.custom_op(
    "b12x::ple_hash_pipeline",
    mutates_args=("out", "request_ids"),
)
def _hash_pipeline_op(
    token_ids: torch.Tensor, query_start_loc: torch.Tensor, committed_history: torch.Tensor,
    num_seqs: torch.Tensor, num_tokens: torch.Tensor, multipliers: torch.Tensor,
    prime_sizes: torch.Tensor, table_offsets: torch.Tensor, out: torch.Tensor,
    request_ids: torch.Tensor, plan_handle: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "sequence.ple_hash", token_ids.device)
    state.run_tensors(
        token_ids, query_start_loc, committed_history, num_seqs, num_tokens,
        multipliers, prime_sizes, table_offsets, out, request_ids,
    )


@_hash_pipeline_op.register_fake
def _hash_pipeline_fake(
    token_ids: torch.Tensor, query_start_loc: torch.Tensor, committed_history: torch.Tensor,
    num_seqs: torch.Tensor, num_tokens: torch.Tensor, multipliers: torch.Tensor,
    prime_sizes: torch.Tensor, table_offsets: torch.Tensor, out: torch.Tensor,
    request_ids: torch.Tensor, plan_handle: int,
) -> None:
    del token_ids, query_start_loc, committed_history, num_seqs, num_tokens
    del multipliers, prime_sizes, table_offsets, out, request_ids, plan_handle


def run_hash_kernel(binding: Binding) -> None:
    """Dispatch the opaque, mutation-declared hash pipeline."""
    if binding.plan is None:
        raise TypeError("PLE hashing requires a prepared binding")
    torch.ops.b12x.ple_hash_pipeline(
        binding.token_ids,
        binding.query_start_loc,
        binding.committed_history,
        binding.num_seqs,
        binding.num_tokens,
        binding.geometry.multipliers,
        binding.geometry.prime_sizes,
        binding.geometry.table_offsets,
        binding.out,
        binding.request_ids,
        binding.plan.handle,
    )


__all__ = ["run_hash_kernel"]
