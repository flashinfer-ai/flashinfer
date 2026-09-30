"""Triton local-shard gather and dequantization for PLE embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from b12x.preparation.types import plan_from_handle, require_prepared

if TYPE_CHECKING:
    from ._contracts import Binding


_BLOCK_D = 128


def _scratch_view(
    scratch: torch.Tensor,
    *,
    offset_bytes: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    nbytes = numel * int(dtype.itemsize)
    return scratch.narrow(0, int(offset_bytes), nbytes).view(dtype).view(shape)


def _pipeline_scratch_views(
    scratch: torch.Tensor,
    *,
    max_tokens: int,
    head_count: int,
    ids_offset_bytes: int,
    request_ids_offset_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = _scratch_view(
        scratch,
        offset_bytes=ids_offset_bytes,
        shape=(max_tokens, head_count),
        dtype=torch.int64,
    )
    request_ids = _scratch_view(
        scratch,
        offset_bytes=request_ids_offset_bytes,
        shape=(max_tokens,),
        dtype=torch.int32,
    )
    return ids, request_ids


@triton.jit
def _bf16_lookup_kernel(
    weight_ptr,
    ids_ptr,
    num_tokens_ptr,
    out_ptr,
    MAX_TOKENS: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    TABLE_VOCAB_SIZE: tl.constexpr,
    SHARD_START: tl.constexpr,
    SHARD_END: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPACT_ROWS: tl.constexpr = False,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    columns = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    column_mask = columns < HEAD_DIM
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    token_live = (token < num_tokens) & (num_tokens >= 0) & (num_tokens <= MAX_TOKENS)
    id_offset = token.to(tl.int64) * HEAD_COUNT + head.to(tl.int64)
    embedding_id = tl.load(ids_ptr + id_offset, mask=token_live, other=-1).to(tl.int64)
    local = (
        token_live
        & (embedding_id >= SHARD_START)
        & (embedding_id < SHARD_END)
        & (embedding_id < TABLE_VOCAB_SIZE)
    )
    local_row = tl.where(
        local,
        embedding_id - tl.full((), SHARD_START, tl.int64),
        0,
    ).to(tl.int64)
    if COMPACT_ROWS:
        local_row = id_offset
    row_base = local_row * tl.full((), HEAD_DIM, tl.int64)
    value = tl.load(
        weight_ptr + row_base + columns.to(tl.int64),
        mask=local & column_mask,
        other=0.0,
    ).to(tl.bfloat16)
    out_offset = (
        token.to(tl.int64) * EMBEDDING_DIM
        + head.to(tl.int64) * HEAD_DIM
        + columns.to(tl.int64)
    )
    tl.store(out_ptr + out_offset, tl.where(local, value, 0.0), mask=column_mask)


@triton.jit
def _fp8_lookup_kernel(
    weight_ptr,
    weight_scale_ptr,
    ids_ptr,
    num_tokens_ptr,
    out_ptr,
    MAX_TOKENS: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    TABLE_VOCAB_SIZE: tl.constexpr,
    SHARD_START: tl.constexpr,
    SHARD_END: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPACT_ROWS: tl.constexpr = False,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    column_block = tl.program_id(2)
    columns = column_block * BLOCK_D + tl.arange(0, BLOCK_D)
    column_mask = columns < HEAD_DIM
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    valid_count = (num_tokens >= 0) & (num_tokens <= MAX_TOKENS)
    token_live = (token < num_tokens) & valid_count
    id_offset = token.to(tl.int64) * HEAD_COUNT + head.to(tl.int64)
    embedding_id = tl.load(ids_ptr + id_offset, mask=token_live, other=-1).to(tl.int64)
    local = (
        token_live
        & (embedding_id >= SHARD_START)
        & (embedding_id < SHARD_END)
        & (embedding_id < TABLE_VOCAB_SIZE)
    )
    local_row = tl.where(
        local,
        embedding_id - tl.full((), SHARD_START, tl.int64),
        0,
    ).to(tl.int64)
    if COMPACT_ROWS:
        local_row = id_offset
    row_base = local_row * tl.full((), HEAD_DIM, tl.int64)
    quantized = tl.load(
        weight_ptr + row_base + columns.to(tl.int64),
        mask=local & column_mask,
        other=0.0,
    ).to(tl.float32)
    scale = tl.load(weight_scale_ptr).to(tl.float32)
    dequantized = (quantized * scale).to(tl.bfloat16)
    out_offset = (
        token.to(tl.int64) * EMBEDDING_DIM
        + head.to(tl.int64) * HEAD_DIM
        + columns.to(tl.int64)
    )
    tl.store(
        out_ptr + out_offset,
        tl.where(local, dequantized, 0.0),
        mask=column_mask,
    )


@triton.jit
def _nvfp4_lookup_kernel(
    weight_ptr,
    weight_scale_ptr,
    weight_scale_2_ptr,
    ids_ptr,
    num_tokens_ptr,
    out_ptr,
    MAX_TOKENS: tl.constexpr,
    HEAD_COUNT: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EMBEDDING_DIM: tl.constexpr,
    TABLE_VOCAB_SIZE: tl.constexpr,
    SHARD_START: tl.constexpr,
    SHARD_END: tl.constexpr,
    BLOCK_D: tl.constexpr,
    COMPACT_ROWS: tl.constexpr = False,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    columns = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    column_mask = columns < HEAD_DIM
    num_tokens = tl.load(num_tokens_ptr).to(tl.int32)
    token_live = (token < num_tokens) & (num_tokens >= 0) & (num_tokens <= MAX_TOKENS)
    id_offset = token.to(tl.int64) * HEAD_COUNT + head.to(tl.int64)
    embedding_id = tl.load(ids_ptr + id_offset, mask=token_live, other=-1).to(tl.int64)
    local = (
        token_live
        & (embedding_id >= SHARD_START)
        & (embedding_id < SHARD_END)
        & (embedding_id < TABLE_VOCAB_SIZE)
    )
    local_row = tl.where(
        local,
        embedding_id - tl.full((), SHARD_START, tl.int64),
        0,
    ).to(tl.int64)
    if COMPACT_ROWS:
        local_row = id_offset

    packed_row_base = local_row * tl.full((), HEAD_DIM // 2, tl.int64)
    packed = tl.load(
        weight_ptr + packed_row_base + (columns // 2).to(tl.int64),
        mask=local & column_mask,
        other=0,
    ).to(tl.uint8)
    nibble = tl.where(
        (columns & 1) == 0,
        packed & 0x0F,
        (packed >> 4) & 0x0F,
    ).to(tl.int32)
    magnitude_code = nibble & 0x07
    magnitude = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(
                            magnitude_code == 5,
                            3.0,
                            tl.where(magnitude_code == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    ).to(tl.float32)
    quantized = tl.where((nibble & 0x08) != 0, -magnitude, magnitude)

    scale_cols = HEAD_DIM // 16
    scale_row_base = local_row * tl.full((), scale_cols, tl.int64)
    block_scale = tl.load(
        weight_scale_ptr + scale_row_base + (columns // 16).to(tl.int64),
        mask=local & column_mask,
        other=0.0,
    ).to(tl.float32)
    global_scale = tl.load(weight_scale_2_ptr).to(tl.float32)
    dequantized = (quantized * block_scale * global_scale).to(tl.bfloat16)
    out_offset = (
        token.to(tl.int64) * EMBEDDING_DIM
        + head.to(tl.int64) * HEAD_DIM
        + columns.to(tl.int64)
    )
    tl.store(
        out_ptr + out_offset,
        tl.where(local, dequantized, 0.0),
        mask=column_mask,
    )


@torch.library.custom_op(
    "b12x::ple_embedding_pipeline",
    mutates_args=("scratch", "out"),
)
def _pipeline_op(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    weight_scale_2: torch.Tensor | None,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    scratch: torch.Tensor,
    out: torch.Tensor,
    token_count: int,
    plan_handle: int,
) -> None:
    state = require_prepared(plan_from_handle(plan_handle), "sequence.ple_embedding", out.device)
    state.run_tensors(
        weight, weight_scale, weight_scale_2, token_ids, query_start_loc,
        committed_history, num_seqs, num_tokens, multipliers, prime_sizes,
        table_offsets, scratch, out, token_count=token_count,
    )


@_pipeline_op.register_fake
def _pipeline_fake(
    weight: torch.Tensor,
    weight_scale: torch.Tensor | None,
    weight_scale_2: torch.Tensor | None,
    token_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    committed_history: torch.Tensor,
    num_seqs: torch.Tensor,
    num_tokens: torch.Tensor,
    multipliers: torch.Tensor,
    prime_sizes: torch.Tensor,
    table_offsets: torch.Tensor,
    scratch: torch.Tensor,
    out: torch.Tensor,
    token_count: int,
    plan_handle: int,
) -> None:
    del weight, weight_scale, weight_scale_2, token_ids, query_start_loc
    del committed_history, num_seqs, num_tokens, multipliers, prime_sizes
    del table_offsets, scratch, out, token_count, plan_handle


def run_pipeline(binding: Binding, *, token_count: int) -> None:
    geometry = binding._hash_binding.geometry
    torch.ops.b12x.ple_embedding_pipeline(
        binding.weight, binding.weight_scale, binding.weight_scale_2,
        binding.token_ids, binding.query_start_loc, binding.committed_history,
        binding.num_seqs, binding.num_tokens, geometry.multipliers,
        geometry.prime_sizes, geometry.table_offsets, binding.scratch,
        binding.out, token_count, binding.plan.handle,
    )


__all__ = ["run_pipeline"]
