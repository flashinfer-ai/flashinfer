# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""Inputs of token-sparse attention, where each query token attends to its own
list of keys (``trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K)``)."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch

from .api_logging import flashinfer_api
from .cute_dsl.launch import launch
from .trace.templates.token_sparse import token_sparse_kv_indices_trace


@cute.kernel
def kv_indices_kernel(
    positions: cute.Tensor,
    requests: cute.Tensor,
    block_table: cute.Tensor,
    block_tables: cute.Tensor,
    seq_lens: cute.Tensor,
    requests_per_token: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
):
    """Map token t's key positions to cache slots, and write them as the K and V
    lists of each of its requests_per_token attention requests, with the number
    of valid keys as their length. Key k of request requests[t] is in slot
    block_table[request, k // 64] * 64 + k % 64. Negative positions map to -1,
    and come last. Thread t holds positions 4t to 4t + 3 of every 512."""
    thread, token = cute.arch.thread_idx()[0], cute.arch.block_idx()[0]
    warp_counts = utils.SmemAllocator().allocate_tensor(
        cutlass.Int32, cute.make_layout(4), 4
    )
    request = requests[token]
    keys = cute.make_rmem_tensor((4, top_k // 512), cutlass.Int32)
    slots = cute.make_rmem_tensor((4, top_k // 512), cutlass.Int32)
    for i in cutlass.range_constexpr(top_k // 512):
        chunk = cute.local_tile(positions[token, None], (4,), (thread + 128 * i,))
        keys[None, i].store(chunk.load())
    for j in cutlass.range(top_k // 128, unroll_full=True):
        slots[j] = block_table[request, cutlass.max(keys[j], 0) // 64]
    count = cutlass.Int32(0)
    for j in cutlass.range(top_k // 128, unroll_full=True):
        slots[j] = cutlass.select_(keys[j] < 0, -1, slots[j] * 64 + keys[j] % 64)
        count += cutlass.Int32(keys[j] >= 0)
    warp_counts[thread // 32] = cute.arch.warp_reduction_sum(count)
    cute.arch.sync_threads()
    count = sum(warp_counts[w] for w in range(4))
    for core in cutlass.range_constexpr(requests_per_token):
        row = token * requests_per_token + core
        for i in cutlass.range_constexpr(top_k // 512):
            for slot_list in cutlass.range_constexpr(2):  # K slots, then V slots
                chunk = block_tables[row, slot_list, None]
                chunk = cute.local_tile(chunk, (4,), (thread + 128 * i,))
                chunk.store(slots[None, i].load())
        if thread == 0:
            seq_lens[row] = count


@flashinfer_api(trace=token_sparse_kv_indices_trace)
def token_sparse_kv_indices(
    positions: torch.Tensor,
    requests: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    requests_per_token: int,
) -> None:
    """Build the K/V slot lists of token-sparse attention from key positions.

    Token ``t`` attends to the keys at ``positions[t]`` (int32 ``[tokens, K]``,
    ``K`` a multiple of 512), valid positions first, then -1. Its keys belong to
    request ``requests[t]``, whose cache ``block_table`` (int32
    ``[requests, blocks]``) pages 64 keys to a page, so key ``k`` is in slot
    ``block_table[request, k // 64] * 64 + k % 64``. Each token becomes
    ``requests_per_token`` consecutive attention requests, for example one per
    group of its query heads. Request ``r`` gets the slots in both rows of
    ``block_tables[r]`` (int32 ``[rows, 2, K]``, K then V slots, -1 past the
    count) and the count in ``seq_lens[r]`` (int32 ``[rows]``). Later rows are
    left as they are. These are the ``block_tables`` and ``seq_lens`` that
    ``trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K,
    uses_shared_paged_kv_idx=False)`` takes over a cache of one-token pages. It
    runs as one launch without host synchronization, so it can be captured in a
    CUDA graph (run it once before capture to compile it)."""
    num_tokens, top_k = positions.shape
    if top_k % 512:
        raise ValueError(f"The key lists must be a multiple of 512 long, got {top_k}")
    rows = num_tokens * requests_per_token  # attention requests
    if block_tables.shape[0] < rows or block_tables.shape[1:] != (2, top_k):
        raise ValueError("block_tables must be [rows or more, 2, K]")
    if seq_lens.shape[0] < rows:
        raise ValueError("seq_lens must hold rows or more lengths")
    args = (positions, requests, block_table, block_tables, seq_lens)
    if not all(t.is_contiguous() for t in args):
        raise ValueError("Positions, requests, tables and lengths must be contiguous")
    if num_tokens:
        constants = (requests_per_token, top_k)
        launch(
            kv_indices_kernel,
            (num_tokens, 1),
            128,
            args,
            constants,
            module="token_sparse",
        )
