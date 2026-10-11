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

# A thread loads LOAD key positions at a time (16 bytes), and THREADS threads read
# a token's list in passes of PASS positions, so lists are whole passes long.
LOAD, THREADS = 4, 128
PASS = LOAD * THREADS
PAGE = 64  # keys per block-table page


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
    block_table[request, k // PAGE] * PAGE + k % PAGE. Negative positions map to
    -1, and come last. In each pass, thread t holds positions LOAD * t to
    LOAD * t + LOAD - 1, so it holds top_k // THREADS keys in all."""
    thread, token = cute.arch.thread_idx()[0], cute.arch.block_idx()[0]
    warp_counts = utils.SmemAllocator().allocate_tensor(
        cutlass.Int32, cute.make_layout(THREADS // 32), 4
    )
    request = requests[token]
    keys = cute.make_rmem_tensor((LOAD, top_k // PASS), cutlass.Int32)
    slots = cute.make_rmem_tensor((LOAD, top_k // PASS), cutlass.Int32)
    for i in cutlass.range_constexpr(top_k // PASS):
        chunk = cute.local_tile(
            positions[token, None], (LOAD,), (thread + THREADS * i,)
        )
        keys[None, i].store(chunk.load())
    for j in cutlass.range(top_k // THREADS, unroll_full=True):
        slots[j] = block_table[request, cutlass.max(keys[j], 0) // PAGE]
    count = cutlass.Int32(0)
    for j in cutlass.range(top_k // THREADS, unroll_full=True):
        slots[j] = cutlass.select_(keys[j] < 0, -1, slots[j] * PAGE + keys[j] % PAGE)
        count += cutlass.Int32(keys[j] >= 0)
    warp_counts[thread // 32] = cute.arch.warp_reduction_sum(count)
    cute.arch.sync_threads()
    count = sum(warp_counts[w] for w in range(THREADS // 32))
    for core in cutlass.range_constexpr(requests_per_token):
        row = token * requests_per_token + core
        for i in cutlass.range_constexpr(top_k // PASS):
            for slot_list in cutlass.range_constexpr(2):  # K slots, then V slots
                chunk = block_tables[row, slot_list, None]
                chunk = cute.local_tile(chunk, (LOAD,), (thread + THREADS * i,))
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

    Each token becomes ``requests_per_token`` consecutive attention requests, for
    example one per group of its query heads, each with the token's K and V slot
    lists and count. These are the ``block_tables`` and ``seq_lens`` that
    ``trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K,
    uses_shared_paged_kv_idx=False)`` takes over a cache of one-token pages.
    It runs as one launch without host synchronization, so it can be captured in a
    CUDA graph (run it once before capture to compile it).

    Parameters
    ----------
    positions : torch.Tensor
        int32 ``[tokens, K]``, ``K`` a multiple of 512: each token's key
        positions, valid ones first, then -1.
    requests : torch.Tensor
        int32 ``[tokens]``: each token's request, a row of ``block_table``.
    block_table : torch.Tensor
        int32 ``[requests, blocks]``: each request's pages of 64 keys. Key ``k``
        is in slot ``block_table[request, k // 64] * 64 + k % 64``.
    block_tables : torch.Tensor
        int32 ``[rows, 2, K]``, output: request ``r`` gets the slots in both rows
        (K, then V), -1 past its count. Rows past ``tokens * requests_per_token``
        are left as they are.
    seq_lens : torch.Tensor
        int32 ``[rows]``, output: each request's count of valid keys.
    requests_per_token : int
        Attention requests per token.
    """
    num_tokens, top_k = positions.shape
    if top_k % PASS:
        raise ValueError(
            f"The key lists must be a multiple of {PASS} long, got {top_k}"
        )
    rows = num_tokens * requests_per_token  # attention requests
    if block_tables.shape[0] < rows or block_tables.shape[1:] != (2, top_k):
        raise ValueError("block_tables must be [rows or more, 2, K]")
    if seq_lens.shape[0] < rows or requests.shape[0] < num_tokens:
        raise ValueError("seq_lens must hold a length per row, requests one per token")
    if block_table.shape[1] == 0:
        raise ValueError("The block table must hold at least one page per request")
    args = (positions, requests, block_table, block_tables, seq_lens)
    if not all(
        t.is_cuda and t.is_contiguous() and t.dtype == torch.int32 for t in args
    ):
        raise ValueError(
            "Positions, tables and lengths must be contiguous int32 on CUDA"
        )
    if num_tokens:
        constants = (requests_per_token, top_k)
        launch(
            kv_indices_kernel,
            (num_tokens, 1),
            THREADS,
            args,
            constants,
            module="token_sparse",
            key_files=(__file__,),
        )
