# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""Inputs for token-sparse attention, where each query token attends to its own
list of keys. trtllm-gen's token-sparse kernels
(``trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K)``) read a K and a V
slot list per token. Dense attention kernels that take a mask can instead attend
over a group of tokens' union of keys, with a bitmask per token."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch

from .api_logging import flashinfer_api
from .cute_dsl.launch import launch
from .topk_varlen.kernels import block_scan
from .trace.templates.token_sparse import (
    group_union_masks_trace,
    token_sparse_kv_indices_trace,
)


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
    if top_k % 512:
        raise ValueError(f"The key lists must be a multiple of 512 long, got {top_k}")
    rows = num_tokens * requests_per_token  # attention requests
    if block_tables.shape[0] < rows or block_tables.shape[1:] != (2, top_k):
        raise ValueError("block_tables must be [rows or more, 2, K]")
    if seq_lens.shape[0] < rows or requests.shape[0] < num_tokens:
        raise ValueError("seq_lens must hold a length per row, requests one per token")
    if block_table.shape[1] == 0:
        raise ValueError("The block table must hold at least one page per request")
    args = (positions, requests, block_table, block_tables, seq_lens)
    if not all(t.is_contiguous() and t.dtype == torch.int32 for t in args):
        raise ValueError("Positions, tables and lengths must be contiguous int32")
    if num_tokens:
        constants = (requests_per_token, top_k)
        launch(
            kv_indices_kernel,
            (num_tokens, 1),
            128,
            args,
            constants,
            module="token_sparse",
            key_files=(__file__,),
        )


@cute.kernel
def union_masks_kernel(
    positions: cute.Tensor,
    query_rows: cute.Tensor,
    group_unions: cute.Tensor,
    requests: cute.Tensor,
    block_table: cute.Tensor,
    union: cute.Tensor,
    union_sizes: cute.Tensor,
    mask: cute.Tensor,
    key_tiles: cutlass.Int32,
    group: cutlass.Constexpr[int],
    words: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
):
    """CTA g builds union b = group_unions[g] from rows g * group to
    g * group + group - 1 of request requests[g]. A shared bitmap collects their
    keys, and a key's union column is the number of union keys below it.
    union_sizes[b] gets the union's size U, union[b, :U] its cache slots by
    ascending key, and mask row query_rows[r] gets bit j of word w when row r
    selected column 32w + j (w < 4 ceil(U / 128)); -1 rows are not written.
    Threads 128i to 128i + 127 take rows i, i + 4, ..., and thread t holds
    positions 4t to 4t + 3 of every 512."""
    thread, g = cute.arch.thread_idx()[0], cute.arch.block_idx()[0]
    lane, quarter = thread % 128, thread // 128
    stage_words = min(group * top_k // 32, words)
    shared = utils.SmemAllocator()
    bitmap = shared.allocate_tensor(cutlass.Int32, cute.make_layout(words), 16)
    # Union keys below each 128-key block, and a staging row of mask bits per quarter.
    below = shared.allocate_tensor(cutlass.Int32, cute.make_layout(words // 4), 16)
    stage = shared.allocate_tensor(
        cutlass.Int32, cute.make_layout((4, stage_words), stride=(stage_words, 1)), 16
    )
    sums = shared.allocate_tensor(cutlass.Int32, cute.make_layout(16), 16)
    first, row_stage = g * group, stage[quarter, None]
    request, batch = requests[g], group_unions[g]
    for base in cutlass.range(0, 4 * key_tiles, 512):  # whole rounds of the scan
        bitmap[base + thread] = 0
    for i in cutlass.range(lane, stage_words, 128):
        row_stage[i] = 0
    cute.arch.sync_threads()
    keys = cute.make_rmem_tensor((4, top_k // 512), cutlass.Int32)
    for r in cutlass.range(first + quarter, first + group, 4):
        for i in cutlass.range_constexpr(top_k // 512):
            chunk = cute.local_tile(positions[r, None], (4,), (lane + 128 * i,))
            keys[None, i].store(chunk.load())
        for j in cutlass.range(top_k // 128, unroll_full=True):
            if keys[j] >= 0:
                word = bitmap.iterator + (keys[j] >> 5)
                bit = cutlass.Int32(1) << (keys[j] & 31)
                cute.arch.atomic_or(word.llvm_ptr, bit, scope="cta")
    cute.arch.sync_threads()

    # Assign columns 512 words at a time. The number of union keys below a word is
    # a block prefix sum of the words' popcounts.
    size = cutlass.Int32(0)
    for base in cutlass.range(0, 4 * key_tiles, 512):
        word = base + thread
        bits = cutlass.Uint32(bitmap[word])
        count = cutlass.Int32(cute.arch.popc(bits))
        column, total = block_scan.block_prefix_sum_kernel(
            count, sums, thread, 512, 16, need_total_sum=True
        )
        column += size
        if word % 4 == 0:
            below[word // 4] = column - count
        if bits != 0 and batch >= 0:
            page = block_table[request, word // 2]
            for _ in cutlass.range(count):
                bit = cutlass.Int32(cute.arch.bfind(bits))
                bits ^= cutlass.Uint32(1) << bit
                column -= 1
                union[batch, column] = page * 64 + word % 2 * 32 + bit
        size += total
        cute.arch.sync_threads()  # `sums` is reused
    if thread == 0 and batch >= 0:
        union_sizes[batch] = size

    # Each quarter ORs a row's bits into its staging row, then stores the row.
    for r in cutlass.range(first + quarter, first + group, 4):
        for i in cutlass.range_constexpr(top_k // 512):
            chunk = cute.local_tile(positions[r, None], (4,), (lane + 128 * i,))
            keys[None, i].store(chunk.load())
        for j in cutlass.range(top_k // 128, unroll_full=True):
            key = keys[j]
            if key >= 0:
                word = key >> 5
                lower = (cutlass.Uint32(1) << cutlass.Uint32(key & 31)) - 1
                column = below[key >> 7] + cutlass.Int32(
                    cute.arch.popc(cutlass.Uint32(bitmap[word]) & lower)
                )
                for k in cutlass.range_constexpr(3):  # earlier words of its block
                    if k < word % 4:
                        column += cutlass.Int32(
                            cute.arch.popc(bitmap[word - word % 4 + k])
                        )
                bit = cutlass.Int32(1) << (column & 31)
                stage_word = row_stage.iterator + (column >> 5)
                cute.arch.atomic_or(stage_word.llvm_ptr, bit, scope="cta")
        query = query_rows[r]
        cute.arch.barrier(barrier_id=1 + quarter, number_of_threads=128)
        for i in cutlass.range(lane, (size + 127) // 128, 128):
            part = cute.local_tile(row_stage, (4,), (i,))
            if query >= 0:  # a row without a mask row still clears its stage
                cute.local_tile(mask[query, None], (4,), (i,)).store(part.load())
            part.fill(0)
        cute.arch.barrier(barrier_id=1 + quarter, number_of_threads=128)


@flashinfer_api(trace=group_union_masks_trace)
def group_union_masks(
    positions: torch.Tensor,
    query_rows: torch.Tensor,
    group_unions: torch.Tensor,
    requests: torch.Tensor,
    block_table: torch.Tensor,
    *,
    union: torch.Tensor,
    union_sizes: torch.Tensor,
    mask: torch.Tensor,
    num_keys: int,
) -> None:
    """Build the inputs of masked dense attention from key positions: each group
    of consecutive tokens' union of keys, and each token's bitmask over it.

    Neighboring tokens, such as a request's prefill tokens, select many of the
    same keys, so a group of them can attend to their union as one varlen batch
    entry, each token masked to its own keys. The tokens form ``len(requests)``
    equal groups. It runs as one launch without host synchronization, so it can be captured in a
    CUDA graph (run it once before capture to compile it).

    Parameters
    ----------
    positions : torch.Tensor
        int32 ``[tokens, K]``, as in :func:`token_sparse_kv_indices`.
    query_rows : torch.Tensor
        int32 ``[tokens]``: each token's row of ``mask``, -1 for none.
    group_unions : torch.Tensor
        int32 ``[groups]``: each group's row of ``union``, -1 for none.
    requests : torch.Tensor
        int32 ``[groups]``: each group's request, a row of ``block_table``.
    block_table : torch.Tensor
        int32 ``[requests, blocks]``: each request's pages of 64 keys.
    union : torch.Tensor
        int32, output: a row gets its group's union of ``U`` keys, as cache slots
        in ascending key order.
    union_sizes : torch.Tensor
        int32, output: ``U`` for each union row.
    mask : torch.Tensor
        int32, output: bit ``j`` of word ``w`` of a token's row is set when the
        token selected union column ``32w + j``. Only words
        ``w < 4 ceil(U / 128)`` are written.
    num_keys : int
        A bound on the key positions. The block table must cover it.
    """
    rows, top_k = positions.shape
    groups = len(requests)
    group = rows // max(groups, 1)
    key_tiles = (num_keys + 127) // 128
    columns = min(group * top_k, 128 * key_tiles)  # the most keys a union holds
    # Shared memory: the key bitmap (two words per 64-key page, in whole rounds of
    # 512 threads), its counts per 128 keys, and a mask staging row per 128 threads.
    words = 512 * -(-block_table.shape[1] // 256)
    shared = 4 * (words + words // 4 + 4 * min(group * top_k // 32, words) + 16)
    device = torch.cuda.get_device_properties(positions.device)
    if top_k % 512:
        raise ValueError(f"The key lists must be a multiple of 512 long, got {top_k}")
    if groups * group != rows or len(group_unions) != groups or len(query_rows) < rows:
        raise ValueError("Groups must be equal and cover the tokens, a mask row each")
    if len(union_sizes) < union.shape[0]:
        raise ValueError("union_sizes must hold a size per union row")
    if num_keys > 64 * block_table.shape[1]:
        raise ValueError("The block table must cover num_keys")
    if min(union.shape[1], 32 * mask.shape[1]) < columns:
        raise ValueError("union and mask rows must hold every group's union")
    if shared > device.shared_memory_per_block_optin:
        raise ValueError("A group's key bitmap and mask rows exceed shared memory")
    args = (
        positions,
        query_rows,
        group_unions,
        requests,
        block_table,
        union,
        union_sizes,
        mask,
        key_tiles,
    )
    if not all(t.is_contiguous() and t.dtype == torch.int32 for t in args[:-1]):
        raise ValueError("Positions, tables, unions and masks must be contiguous int32")
    if rows:
        launch(
            union_masks_kernel,
            (groups, 1),
            512,
            args,
            (group, words, top_k),
            module="token_sparse_union",
            key_files=(__file__, block_scan.__file__),
        )
