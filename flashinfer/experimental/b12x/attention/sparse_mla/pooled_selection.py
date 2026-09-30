"""Pooled-index selection transforms for paged sparse MLA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import FrozenMapping, MemoryRequirements, Plan
from b12x.preparation.types import require_prepared

from ._tuning import SparseMlaConfig, SparseMlaQuery, TUNING


@triton.jit(do_not_specialize=(
    "pool_stride", "block_table_stride", "output_stride",
    "max_num_blocks", "num_cache_blocks",
))
def _expand_pooled_topk_to_physical_slots_kernel(
    pool_indices,
    last_token_positions,
    request_ids,
    block_table,
    output,
    active_counts,
    pool_stride,
    block_table_stride,
    output_stride,
    max_num_blocks,
    num_cache_blocks,
    HISTORY_TOKENS: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_STRIDE_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    column = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = column < OUTPUT_WIDTH

    sequence_length = tl.load(last_token_positions + row).to(tl.int64) + 1
    complete_pools = sequence_length // POOL_SIZE
    selected_pools = tl.minimum(complete_pools, HISTORY_TOKENS // POOL_SIZE)
    selected_history_tokens = selected_pools * POOL_SIZE
    tail_start = complete_pools * POOL_SIZE
    history = column < selected_history_tokens
    pool_column = column // POOL_SIZE
    pool_offset = column % POOL_SIZE
    pool_id = tl.load(
        pool_indices + row * pool_stride + pool_column,
        mask=mask & history,
        other=-1,
    ).to(tl.int64)
    history_value = tl.where(pool_id >= 0, pool_id * POOL_SIZE + pool_offset, -1)
    tail_offset = column - selected_history_tokens
    tail_count = sequence_length - tail_start
    in_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    logical_token = tl.where(
        history,
        history_value,
        tl.where(in_tail, tail_start + tail_offset, -1),
    )

    request = tl.load(request_ids + row).to(tl.int64)
    block_id = logical_token // BLOCK_SIZE
    in_block = logical_token - block_id * BLOCK_SIZE
    valid = (logical_token >= 0) & (block_id < max_num_blocks)
    page = tl.load(
        block_table + request * block_table_stride + block_id,
        mask=mask & valid,
        other=-1,
    ).to(tl.int64)
    valid &= (page >= 0) & (page < num_cache_blocks)
    physical_token = tl.where(
        valid,
        page * BLOCK_STRIDE_ROWS + in_block,
        -1,
    ).to(tl.int32)
    tl.store(output + row * output_stride + column, physical_token, mask=mask)

    active_count = selected_pools * POOL_SIZE + tail_count
    tl.store(active_counts + row, active_count, mask=tile == 0)


def expand_pooled_topk_to_physical_slots(
    pool_indices: torch.Tensor,
    last_token_positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    output: torch.Tensor,
    active_counts: torch.Tensor,
    *,
    pool_size: int,
    block_size: int,
    block_stride_rows: int,
    num_cache_blocks: int,
    plan: Plan | None = None,
) -> None:
    """Expand selected pools and the live tail into physical cache slots.

    The caller supplies request-relative pool IDs, scheduler metadata, and
    caller-owned outputs. Selected pools and the live tail occupy one contiguous
    prefix of the output. The output width must cover every selected pool plus
    at most ``pool_size - 1`` unpooled tail tokens. Physical-slot arithmetic is
    performed in 64 bits and the result remains an int32 sparse-MLA index.
    """
    tensors = (
        pool_indices,
        request_ids,
        block_table,
        output,
        active_counts,
    )
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError("pooled sparse-selection metadata must use int32 tensors")
    if last_token_positions.dtype != torch.int64:
        raise TypeError("last_token_positions must use int64")
    if pool_indices.ndim != 2 or int(pool_indices.shape[1]) < 1:
        raise ValueError("pool_indices must be a non-empty rank-two tensor")
    rows, pool_topk = map(int, pool_indices.shape)
    if last_token_positions.shape != (rows,) or request_ids.shape != (rows,):
        raise ValueError("pooled selection metadata must have one entry per row")
    if block_table.ndim != 2 or min(map(int, block_table.shape)) < 1:
        raise ValueError("block_table must be a non-empty rank-two tensor")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    output_width = pool_topk * pool_size + pool_size - 1
    if output.shape != (rows, output_width) or active_counts.shape != (rows,):
        raise ValueError(
            "pooled physical-selection outputs must have shapes "
            f"({rows}, {output_width}) and ({rows},)"
        )
    if block_size < 1 or block_stride_rows < block_size:
        raise ValueError("physical cache block geometry is invalid")
    max_physical_slot = (
        (int(num_cache_blocks) - 1) * int(block_stride_rows) + int(block_size) - 1
    )
    if num_cache_blocks < 1 or max_physical_slot > torch.iinfo(torch.int32).max:
        raise ValueError("physical cache slots exceed the int32 index range")
    all_tensors = (*tensors, last_token_positions)
    if any(tensor.device != pool_indices.device for tensor in all_tensors):
        raise ValueError("pooled sparse-selection tensors must share one device")
    if not pool_indices.is_contiguous() or not output.is_contiguous():
        raise ValueError("pool_indices and output must be contiguous")

    if plan is not None:
        state = require_prepared(plan, TUNING.component_id, pool_indices.device)
        if not isinstance(state, _PooledSelectionState):
            raise ValueError("pooled selection requires a pooled-selection plan")
        state.run(
            pool_indices, last_token_positions, request_ids, block_table,
            output, active_counts, pool_size=pool_size, block_size=block_size,
            block_stride_rows=block_stride_rows, num_cache_blocks=num_cache_blocks,
        )
    elif rows:
        block_cols = 128
        _expand_pooled_topk_to_physical_slots_kernel[
            (rows, triton.cdiv(output_width, block_cols))
        ](
            pool_indices,
            last_token_positions,
            request_ids,
            block_table,
            output,
            active_counts,
            int(pool_indices.stride(0)),
            int(block_table.stride(0)),
            int(output.stride(0)),
            int(block_table.shape[1]),
            int(num_cache_blocks),
            HISTORY_TOKENS=pool_topk * pool_size,
            OUTPUT_WIDTH=output_width,
            POOL_SIZE=pool_size,
            BLOCK_SIZE=block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            BLOCK_COLS=block_cols,
            num_warps=4,
        )


@dataclass(frozen=True)
class _CompilePointer:
    dtype: torch.dtype

    def data_ptr(self):
        return 8 if self.dtype == torch.int64 else 4


@program_cache(scope="preparation")
def compile_pooled_selection(query_payload, ordinal):
    """Compile pooled selection from planned metadata, without cache contents."""
    query = SparseMlaQuery(**dict(query_payload))
    i32, i64 = _CompilePointer(torch.int32), _CompilePointer(torch.int64)
    with torch.cuda.device(ordinal):
        return _expand_pooled_topk_to_physical_slots_kernel.warmup(
            i32, i64, i32, i32, i32, i32,
            query.pool_topk, query.max_page_table_width, query.max_width,
            query.max_page_table_width, query.num_cache_blocks,
            HISTORY_TOKENS=query.pool_topk * query.pool_size,
            OUTPUT_WIDTH=query.max_width, POOL_SIZE=query.pool_size,
            BLOCK_SIZE=query.page_size,
            BLOCK_STRIDE_ROWS=query.physical_block_size,
            BLOCK_COLS=128, num_warps=4, grid=(1, 1, 1),
        )


@dataclass(frozen=True)
class _PooledSelectionState:
    query: SparseMlaQuery
    program: object

    def run(self, pool_indices, positions, request_ids, block_table, output,
            active_counts, *, pool_size, block_size, block_stride_rows,
            num_cache_blocks):
        q = self.query
        rows = int(pool_indices.shape[0])
        if (pool_size, int(pool_indices.shape[1]), block_size, block_stride_rows) != (
            q.pool_size, q.pool_topk, q.page_size, q.physical_block_size
        ):
            raise ValueError("pooled selection geometry differs from preparation")
        if (rows > q.max_q_rows or block_table.shape[1] > q.max_page_table_width
                or not 0 < num_cache_blocks <= q.num_cache_blocks):
            raise ValueError("pooled selection exceeds prepared capacity")
        if rows:
            self.program[(rows, triton.cdiv(q.max_width, 128), 1)](
                pool_indices, positions, request_ids, block_table, output,
                active_counts, int(pool_indices.stride(0)),
                int(block_table.stride(0)), int(output.stride(0)),
                int(block_table.shape[1]), int(num_cache_blocks),
                q.pool_topk * q.pool_size, q.max_width, q.pool_size,
                q.page_size, q.physical_block_size, 128,
            )


def plan_pooled_selection(
    *, device, max_rows: int, page_size: int, max_page_table_width: int,
    num_cache_blocks: int, pool_size: int = 4, pool_topk: int = 512,
    block_stride_rows: int | None = None,
    override: SparseMlaConfig | None = None,
) -> Plan:
    """Declare the pool-to-physical-slot transform before graph capture.

    The plan owns the compiled program but no scratch or cache storage. Prepare
    its request, then pass the plan to ``expand_pooled_topk_to_physical_slots``.
    Live row counts do not specialize the compiled program.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("pooled selection requires a CUDA device")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if any(
        type(value) is not int or value <= 0
        for value in (
            max_rows, page_size, max_page_table_width, num_cache_blocks,
            pool_size, pool_topk,
        )
    ):
        raise ValueError("pooled-selection capacities must be positive integers")
    stride = page_size if block_stride_rows is None else block_stride_rows
    if stride < page_size:
        raise ValueError("physical cache block geometry is invalid")
    max_slot = (num_cache_blocks - 1) * stride + page_size - 1
    if max_slot > torch.iinfo(torch.int32).max:
        raise ValueError("physical cache slots exceed the int32 index range")
    query = SparseMlaQuery(
        mode="selection", dtype="int32", kv_dtype="int32",
        num_q_heads=0, qk_head_dim=0, v_head_dim=0,
        max_q_rows=max_rows, max_width=pool_topk * pool_size + pool_size - 1,
        page_size=page_size, model_type=2, head_major_output=False,
        scale_format=0, cache_record_bytes=0, fp8_rope=False,
        latent_scale_per_token=False, has_attention_sink=False,
        cache_layout="paged", operation="pooled_selection", slot_dtype="int32",
        prefill_mg_enabled=False, max_page_table_width=max_page_table_width,
        physical_block_size=stride, num_cache_blocks=num_cache_blocks,
        pool_size=pool_size, pool_topk=pool_topk,
    )
    payload = TUNING.encode_query(query)
    return Plan(
        contract=TUNING, query=query, invocation=FrozenMapping(), override=override,
        _compile_jobs=lambda config, detected: (CompileJob.create(
            "b12x.attention.sparse_mla.pooled_selection:compile_pooled_selection",
            payload, detected.ordinal,
        ),),
        _memory_requirements=lambda config, detected: MemoryRequirements(),
        _materialize=lambda selection, detected: _PooledSelectionState(
            query, compile_pooled_selection(payload, detected.ordinal)
        ),
        _device=device,
    )


__all__ = ["expand_pooled_topk_to_physical_slots", "plan_pooled_selection"]
