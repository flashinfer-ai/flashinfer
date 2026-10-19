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

# A thread loads LOAD key positions at a time (16 bytes), and THREADS threads read
# a token's list in passes of PASS positions, so lists are whole passes long.
LOAD, THREADS = 4, 128
PASS = LOAD * THREADS
PAGE = 64  # keys per block-table page
# group_union_masks: a CTA of ROWS teams of THREADS threads reads ROWS rows at once.
# Its bitmaps and masks hold 32 keys or columns per int32 word, and masks are
# written in whole attention key tiles of TILE columns.
ROWS, TILE = 4, 128
CTA, TILE_WORDS, PAGE_WORDS = ROWS * THREADS, TILE // 32, PAGE // 32
ALIGN = 16  # bytes, of its shared tensors: mask rows are copied 16 bytes at a time


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
    device = positions.device
    if device.type != "cuda" or not all(
        t.device == device and t.is_contiguous() and t.dtype == torch.int32
        for t in args
    ):
        raise ValueError("Positions, tables and lengths: contiguous int32, one GPU")
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
    num_keys: cutlass.Int32,
    group: cutlass.Constexpr[int],
    words: cutlass.Constexpr[int],
    top_k: cutlass.Constexpr[int],
):
    """Build one group's union of keys and its tokens' masks. CTA g takes rows
    g * group to g * group + group - 1 of request requests[g], in three steps:

    1. Mark: each row sets one bit per selected key in a shared bitmap of the
       request's keys (bit k % 32 of int32 word k // 32).
    2. Number: a prefix sum over the bitmap gives each union key its column, the
       number of union keys below it. union[b, :U] gets the cache slots by
       ascending key and union_sizes[b] the size U, for b = group_unions[g].
    3. Mask: row r sets bit j of word w of mask row query_rows[r] when it selected
       column 32w + j. Masks are written in whole tiles of TILE columns.

    Positions outside [0, num_keys) are skipped, and union and mask rows of -1 are
    not written. The CTA reads ROWS rows at a time, THREADS threads each, as in
    kv_indices_kernel."""
    thread, g = cute.arch.thread_idx()[0], cute.arch.block_idx()[0]
    # Team t (THREADS threads) takes rows t, t + ROWS, ... of the group.
    lane, team = thread % THREADS, thread // THREADS
    stage_words = min(group * top_k // 32, words)
    shared = utils.SmemAllocator()
    bitmap = shared.allocate_tensor(cutlass.Int32, cute.make_layout(words), ALIGN)
    # Union keys below each key tile, and a staging row of mask bits per team.
    below = shared.allocate_tensor(
        cutlass.Int32, cute.make_layout(words // TILE_WORDS), ALIGN
    )
    stage = shared.allocate_tensor(
        cutlass.Int32,
        cute.make_layout((ROWS, stage_words), stride=(stage_words, 1)),
        ALIGN,
    )
    sums = shared.allocate_tensor(cutlass.Int32, cute.make_layout(CTA // 32), ALIGN)
    first, row_stage = g * group, stage[team, None]
    request, batch = requests[g], group_unions[g]
    key_tiles = (num_keys + TILE - 1) // TILE
    keys_end = cutlass.Uint32(num_keys)  # unsigned: negative keys are past it
    for base in cutlass.range(0, TILE_WORDS * key_tiles, CTA):  # whole scan rounds
        bitmap[base + thread] = 0
    for i in cutlass.range(lane, stage_words, THREADS):
        row_stage[i] = 0
    cute.arch.sync_threads()

    # 1. Mark each row's keys in the bitmap.
    keys = cute.make_rmem_tensor((LOAD, top_k // PASS), cutlass.Int32)
    for r in cutlass.range(first + team, first + group, ROWS):
        for i in cutlass.range_constexpr(top_k // PASS):
            chunk = cute.local_tile(positions[r, None], (LOAD,), (lane + THREADS * i,))
            keys[None, i].store(chunk.load())
        for j in cutlass.range(top_k // THREADS, unroll_full=True):
            if cutlass.Uint32(keys[j]) < keys_end:
                word = bitmap.iterator + (keys[j] >> 5)
                bit = cutlass.Int32(1) << (keys[j] & 31)
                cute.arch.atomic_or(word.llvm_ptr, bit, scope="cta")
    cute.arch.sync_threads()

    # 2. Number the union keys, one word per thread in each round. The union keys
    # below a word are a block prefix sum of the words' popcounts.
    size = cutlass.Int32(0)
    for base in cutlass.range(0, TILE_WORDS * key_tiles, CTA):
        word = base + thread
        bits = cutlass.Uint32(bitmap[word])
        count = cutlass.Int32(cute.arch.popc(bits))
        column, total = block_scan.block_prefix_sum_kernel(
            count, sums, thread, CTA, CTA // 32, need_total_sum=True
        )
        column += size
        if word % TILE_WORDS == 0:
            below[word // TILE_WORDS] = column - count
        if bits != 0 and batch >= 0:  # write the word's keys as cache slots
            page = block_table[request, word // PAGE_WORDS]
            for _ in cutlass.range(count):
                bit = cutlass.Int32(cute.arch.bfind(bits))
                bits ^= cutlass.Uint32(1) << bit
                column -= 1
                union[batch, column] = page * PAGE + word % PAGE_WORDS * 32 + bit
        size += total
        cute.arch.sync_threads()  # `sums` is reused
    if thread == 0 and batch >= 0:
        union_sizes[batch] = size

    # 3. Each team ORs a row's columns into its staging row, then stores the row.
    for r in cutlass.range(first + team, first + group, ROWS):
        for i in cutlass.range_constexpr(top_k // PASS):
            chunk = cute.local_tile(positions[r, None], (LOAD,), (lane + THREADS * i,))
            keys[None, i].store(chunk.load())
        for j in cutlass.range(top_k // THREADS, unroll_full=True):
            key = keys[j]
            if cutlass.Uint32(key) < keys_end:
                # Its column: union keys below its tile, then below it in the tile.
                word = key >> 5
                lower = (cutlass.Uint32(1) << cutlass.Uint32(key & 31)) - 1
                column = below[word // TILE_WORDS] + cutlass.Int32(
                    cute.arch.popc(cutlass.Uint32(bitmap[word]) & lower)
                )
                for k in cutlass.range_constexpr(TILE_WORDS - 1):  # earlier words
                    if k < word % TILE_WORDS:
                        column += cutlass.Int32(
                            cute.arch.popc(bitmap[word - word % TILE_WORDS + k])
                        )
                bit = cutlass.Int32(1) << (column & 31)
                stage_word = row_stage.iterator + (column >> 5)
                cute.arch.atomic_or(stage_word.llvm_ptr, bit, scope="cta")
        query = query_rows[r]
        cute.arch.barrier(barrier_id=1 + team, number_of_threads=THREADS)
        for i in cutlass.range(lane, (size + TILE - 1) // TILE, THREADS):
            part = cute.local_tile(row_stage, (TILE_WORDS,), (i,))
            if query >= 0:  # a row without a mask row still clears its stage
                tile = cute.local_tile(mask[query, None], (TILE_WORDS,), (i,))
                tile.store(part.load())
            part.fill(0)
        cute.arch.barrier(barrier_id=1 + team, number_of_threads=THREADS)


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
        The number of keys: positions at or past it count as -1. The block table
        must cover it, and shared memory grows with it.
    """
    rows, top_k = positions.shape
    groups = len(requests)
    group = rows // max(groups, 1)
    key_tiles = (num_keys + TILE - 1) // TILE
    columns = min(group * top_k, TILE * key_tiles)  # the most keys a union holds
    # Shared memory, in int32 words: the key bitmap (a bit per key below num_keys,
    # in a power of two of whole scan rounds of CTA words, so few sizes compile),
    # its counts per key tile, a mask staging row per team, and the scan's sums.
    rounds = -(-TILE_WORDS * key_tiles // CTA)
    words = CTA << max(rounds - 1, 0).bit_length()
    stage = ROWS * min(group * top_k // 32, words)
    shared = 4 * (words + words // TILE_WORDS + stage + CTA // 32)
    device = torch.cuda.get_device_properties(positions.device)
    if top_k % PASS:
        raise ValueError(
            f"The key lists must be a multiple of {PASS} long, got {top_k}"
        )
    if groups * group != rows or len(group_unions) != groups or len(query_rows) < rows:
        raise ValueError("Groups must be equal and cover the tokens, a mask row each")
    if len(union_sizes) < union.shape[0]:
        raise ValueError("union_sizes must hold a size per union row")
    if num_keys > PAGE * block_table.shape[1]:
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
        num_keys,
    )
    if not all(
        t.is_cuda and t.is_contiguous() and t.dtype == torch.int32 for t in args[:-1]
    ):
        raise ValueError("Positions, tables, unions and masks must be contiguous int32")
    if rows:
        launch(
            union_masks_kernel,
            (groups, 1),
            CTA,
            args,
            (group, words, top_k),
            module="token_sparse_union",
            key_files=(__file__, block_scan.__file__),
        )
