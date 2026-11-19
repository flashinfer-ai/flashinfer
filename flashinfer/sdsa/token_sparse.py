# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""SDSA (Simplified DSA) inputs for token-sparse attention, where each query
token attends to its own list of keys. trtllm-gen's token-sparse kernels
(``trtllm_batch_decode_with_kv_cache(..., sparse_top_k=K)``) read a K and a V
slot list per token. Dense attention kernels that take a mask can instead attend
over a group of tokens' union of keys, with a bitmask per token."""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

import torch

from ..api_logging import flashinfer_api
from ..cute_dsl.launch import launch
from ..topk_varlen.kernels import block_scan
from ..trace.templates.sdsa import (
    compact_nvfp4_kv_cache_trace,
    gather_nvfp4_kv_cache_by_position_trace,
    group_union_masks_trace,
    token_sparse_kv_indices_trace,
)
from ..utils import get_compute_capability

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


def union_key_limit(group: int, top_k: int, device=None) -> int:
    """The most keys :func:`group_union_masks` takes for groups of ``group`` rows and
    ``top_k`` keys each: its key bitmap and mask rows must fit in shared memory.

    >>> union_key_limit(4, 2048)  # 227 KiB of shared memory, as on SM100
    1441792
    >>> union_key_limit(8, 2048)
    1310720
    >>> union_key_limit(32, 2048)
    1179648
    >>> union_key_limit(32, 512)
    1310720
    """
    optin = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    rounds = 1  # the key bitmap's sizes, as in group_union_masks
    while True:
        bigger = rounds * 2 if rounds < 64 else rounds + 8
        words = CTA * bigger
        stage = ROWS * min(group * top_k // 32, words)
        if 4 * (words + words // TILE_WORDS + stage + CTA // 32) > optin:
            return rounds * CTA * 32
        rounds = bigger


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
    # The key bitmap: a bit per key below num_keys, in whole scan rounds of CTA words.
    # A power of two of rounds, so few sizes compile. Past 64 rounds (2**20 keys), a
    # multiple of 8: a power of two would not fit in shared memory.
    rounds = -(-TILE_WORDS * key_tiles // CTA)
    rounds = (
        1 << max(rounds - 1, 0).bit_length() if rounds <= 64 else -(-rounds // 8) * 8
    )
    words = CTA * rounds
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
    if num_keys > union_key_limit(group, top_k, positions.device):
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
    device = positions.device
    if device.type != "cuda" or not all(
        t.device == device and t.is_contiguous() and t.dtype == torch.int32
        for t in args[:-1]
    ):
        raise ValueError(
            "Positions, tables, unions and masks: contiguous int32, one GPU"
        )
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


# ---------------------------------------------------------------------------
# NVFP4 KV cache -> FP8 attention workspace, from key positions
# ---------------------------------------------------------------------------


def _nvfp4_cache_dim(paged_kv_cache, kv_cache_sf, v_sf_swizzled: bool) -> int:
    """Check K/V NVFP4 data and scale pages, and return the head dim."""
    k = paged_kv_cache[0]
    if any(t.device != k.device for t in (*paged_kv_cache, *kv_cache_sf)):
        raise ValueError("K/V pages and scales must be on one device")
    from ._token_sparse_nvfp4 import (
        FP4_VALUES_PER_BYTE,
        HEAD_DIM_ALIGNMENT,
        NVFP4_SCALE_BLOCK_SIZE,
        V_SCALE_SWIZZLE_TOKENS,
    )

    k, v = paged_kv_cache
    ks, vs = kv_cache_sf
    if any(t.ndim != 4 for t in (k, v, ks, vs)):
        raise ValueError("Expected rank-four HND K/V data and scale pages")
    dim = k.shape[-1] * FP4_VALUES_PER_BYTE.value
    if k.shape[2] <= 0 or dim <= 0:
        raise ValueError("Expected positive page_size and head_dim")
    if k.dtype != torch.uint8 or v.dtype != torch.uint8:
        raise ValueError("NVFP4 data must be uint8")
    if ks.dtype != torch.float8_e4m3fn or vs.dtype != ks.dtype:
        raise ValueError("NVFP4 block scales must be FP8 E4M3")
    if k.shape != v.shape or k.stride() != v.stride() or k.shape[1] != 1:
        raise ValueError("Expected matching HND MQA K/V pages")
    if (
        ks.shape != vs.shape
        or ks.stride() != vs.stride()
        or ks.shape != (*k.shape[:-1], dim // NVFP4_SCALE_BLOCK_SIZE.value)
    ):
        raise ValueError("Expected matching K/V scale pages")
    if dim < HEAD_DIM_ALIGNMENT.value or dim & (dim - 1):
        # The conversion kernels' lane extents (tl.arange) need a power of two.
        raise ValueError(
            f"head_dim must be a power of two, {HEAD_DIM_ALIGNMENT.value} or more"
        )
    if (
        (v_sf_swizzled and k.shape[2] % V_SCALE_SWIZZLE_TOKENS.value)
        or dim % HEAD_DIM_ALIGNMENT.value
        or k.stride(-2) != dim // FP4_VALUES_PER_BYTE.value
        or k.stride(-1) != 1
        or ks.stride(-2) != dim // NVFP4_SCALE_BLOCK_SIZE.value
        or ks.stride(-1) != 1
    ):
        raise ValueError(
            "Expected contiguous token data and HND four-token scale swizzle"
        )
    if get_compute_capability(k.device)[0] not in (10, 11, 12):
        raise NotImplementedError(
            "NVFP4 to FP8 conversion needs a Blackwell GPU (compute capability "
            "10.x, 11.x or 12.x)"
        )
    return dim


def _check_selection(
    positions, block_table, out, block_tables, seq_lens, rows, dim, device
):
    """Shared checks of key positions, block table and outputs (on ``device``,
    the cache's)."""
    if any(
        t.device != device
        for t in (positions, block_table, out, block_tables, seq_lens)
    ):
        raise ValueError(
            "Positions, block table and outputs must be on the cache's device"
        )
    top_k = positions.shape[1] if positions.ndim == 2 else 0
    if (
        positions.ndim != 2
        or positions.dtype != torch.int32
        or positions.stride(-1) != 1
        or top_k <= 0
        or top_k & (top_k - 1)
    ):
        raise ValueError(
            "Expected int32 key positions [rows, top_k], top_k a power of two, "
            "contiguous in the last dimension"
        )
    if (
        block_table.ndim != 2
        or block_table.dtype != torch.int32
        or block_table.stride(-1) != 1
        or block_table.shape[1] < 1
    ):
        raise ValueError(
            "Expected an int32 block table [requests, pages], at least one page"
        )
    if (
        out.ndim != 4
        or out.shape[1:] != (1, 1, 2 * dim)
        or not out.is_contiguous()
        or out.dtype != torch.float8_e4m3fn
    ):
        raise ValueError("Expected contiguous FP8 K/V output rows [N, 1, 1, 2 * D]")
    if (
        block_tables.ndim != 3
        or block_tables.shape[0] < rows
        or block_tables.shape[1:] != (2, top_k)
        or not block_tables.is_contiguous()
        or block_tables.dtype != torch.int32
        or seq_lens.ndim != 1
        or seq_lens.shape[0] < rows
        or seq_lens.dtype != torch.int32
        or not seq_lens.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous int32 block_tables [>= rows, 2, top_k] and "
            "seq_lens [>= rows]"
        )
    return top_k


# compact_nvfp4_kv_cache's looping conversion grid per SM: enough programs to
# keep a large region parallel, few enough that small groups' programs do not
# run idle over the largest region's tiles (GB300: flat from about 100 to 400).
_CONVERT_PROGRAMS_PER_SM = 256
# The full grid (groups x the largest region's tiles) is used up to this many
# times the looping grid: below it, idle programs cost less than the loop
# (GB300: decode unions of 128 requests x 3-6 rows, 49K-98K programs, are 2-6%
# faster on the full grid); above it, most of the grid would be idle (e.g. one
# long dense prefill group next to short union groups).
_LOOP_ABOVE_FACTOR = 4


@functools.cache
def _sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _convert_programs(device: torch.device) -> int:
    return _CONVERT_PROGRAMS_PER_SM * _sm_count(device)


def _check_lengths(lengths, rows, device) -> None:
    if lengths is not None:
        _check_vector("lengths", lengths, rows, torch.int32, device)


def _check_vector(name, t, n, dtype, device) -> None:
    """A contiguous 1-D ``dtype`` tensor on ``device`` with ``n`` elements or
    more."""
    if (
        t.ndim != 1
        or t.dtype != dtype
        or t.device != device
        or t.shape[0] < n
        or (t.shape[0] > 1 and t.stride(0) != 1)
    ):
        raise ValueError(
            f"Expected {name}: contiguous 1-D {dtype} on {device}, "
            f"at least {n} elements"
        )


def _check_launch(requests_per_row, tokens_per_program, num_warps) -> None:
    if requests_per_row < 1:
        raise ValueError("requests_per_row must be positive")
    if tokens_per_program < 1 or tokens_per_program & (tokens_per_program - 1):
        raise ValueError("tokens_per_program must be a power of two")
    if num_warps < 1 or num_warps & (num_warps - 1):
        raise ValueError("num_warps must be a power of two")


@flashinfer_api(trace=gather_nvfp4_kv_cache_by_position_trace)
def gather_nvfp4_kv_cache_by_position(
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    requests: torch.Tensor,
    block_table: torch.Tensor,
    *,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    lengths: torch.Tensor | None = None,
    requests_per_row: int = 1,
    v_sf_swizzled: bool = True,
    tokens_per_program: int = 32,
    num_warps: int = 4,
) -> None:
    """Convert the NVFP4 K/V tokens at the given key positions into FP8
    attention workspace and write their K/V index lists, in one launch.

    Row ``r``'s key ``j`` (cache slot ``block_table[requests[r], key // P] * P
    + key % P``, ``P`` the page size) becomes FP8 row ``r * top_k + j`` of
    ``out`` (see Notes); a negative key gives a zero row. Each of the row's ``requests_per_row``
    attention requests (rows ``r * requests_per_row + i`` of ``block_tables``
    and ``seq_lens``) gets index lists ``r * top_k + j`` and the row's count of
    nonnegative keys: the ``block_tables`` and ``seq_lens`` that
    ``trtllm_batch_decode_with_kv_cache(..., sparse_top_k=top_k,
    uses_shared_paged_kv_idx=False)`` takes over ``out`` as a cache of
    one-token pages.

    Parameters
    ----------
    paged_kv_cache : tuple[torch.Tensor, torch.Tensor]
        K/V uint8 HND pages, each shaped
        ``[pages, 1, page_size, head_dim // 2]`` (two E2M1 values per byte),
        with matching strides and contiguous token data. ``page_size`` is
        positive, and a multiple of four for swizzled V scales; ``head_dim`` is
        a power of two, 64 or more.
    kv_cache_sf : tuple[torch.Tensor, torch.Tensor]
        K/V FP8 E4M3 block-scale pages, each shaped
        ``[pages, 1, page_size, head_dim // 16]``, with matching strides.
        K scales are linear. V scales follow ``v_sf_swizzled``.
    v_sf_swizzled : bool
        Whether V scales use TRT-LLM's HND four-token swizzle, as in caches
        written for trtllm-gen attention; ``page_size`` must then be a
        multiple of four. Pass False for linear V scales, as written by
        :func:`flashinfer.page.nvfp4_quantize_append_paged_kv_cache`.
    positions : torch.Tensor
        int32 ``[rows, top_k]`` key positions, valid ones first, then -1;
        ``top_k`` a power of two (512, 2048), a multiple of
        ``tokens_per_program``.
    requests : torch.Tensor
        int32 ``[rows]``: each row's row of ``block_table``.
    block_table : torch.Tensor
        int32 ``[requests, pages]``.
    out : torch.Tensor
        FP8 E4M3 ``[rows * top_k or more, 1, 1, 2 * head_dim]``, K then V.
    block_tables, seq_lens : torch.Tensor
        int32 outputs ``[rows * requests_per_row, 2, top_k]`` and
        ``[rows * requests_per_row]`` (or more rows).
    lengths : torch.Tensor, optional
        int32 ``[rows]``: each row's count of valid keys, when entries past it
        may hold stale values; those then give zero rows. Entries within the
        count must be valid keys: a negative one gives a zero row that the count
        still includes, so attention weights a zero token. None: count the
        nonnegative keys.
    requests_per_row : int
        Attention requests per row: row ``r``'s lists and count go to rows
        ``r * requests_per_row`` to ``(r + 1) * requests_per_row - 1`` of
        ``block_tables`` and ``seq_lens``.
    tokens_per_program, num_warps : int
        Launch tuning (powers of two): the tokens each program converts, and
        its warps.

    Graph safe: one launch, no host synchronization.

    Notes
    -----
    All tensors must be on the same CUDA device. Outputs are written in place.
    An FP8 row holds the token's decoded NVFP4 values (each E2M1 value times its
    E4M3 block scale) divided by six, excluding the global scales: multiply
    BMM1's K scale and BMM2's V scale by six. This covers NVFP4's full
    normalized range without FP8 saturation. The result is rounded to E4M3:
    products of E2M1 values and E4M3 scales can need more mantissa bits, and
    small scales can become subnormal or zero. Requires a Blackwell GPU
    (compute capability 10.x, 11.x or 12.x); validated on SM103.
    """
    from ._token_sparse_nvfp4 import position_gather_kernel

    dim = _nvfp4_cache_dim(paged_kv_cache, kv_cache_sf, v_sf_swizzled)
    _check_launch(requests_per_row, tokens_per_program, num_warps)
    device = paged_kv_cache[0].device
    rows = positions.shape[0]
    top_k = _check_selection(
        positions,
        block_table,
        out,
        block_tables,
        seq_lens,
        rows * requests_per_row,
        dim,
        device,
    )
    if out.shape[0] < rows * top_k or top_k % tokens_per_program:
        raise ValueError("out needs rows * top_k rows; top_k whole programs")
    _check_vector("requests", requests, rows, torch.int32, device)
    _check_lengths(lengths, rows, device)
    if not rows:
        return
    k, v = paged_kv_cache
    ks, vs = kv_cache_sf
    with torch.cuda.device(k.device):
        position_gather_kernel[(rows, top_k // tokens_per_program)](
            k,
            v,
            ks,
            vs,
            k.stride(0),
            ks.stride(0),
            positions,
            positions.stride(0),
            positions if lengths is None else lengths,
            requests,
            block_table,
            block_table.stride(0),
            out,
            block_tables,
            seq_lens,
            TOP_K=top_k,
            PAGE=k.shape[2],
            DIM=dim,
            TOKENS=tokens_per_program,
            V_SF_SWIZZLED=v_sf_swizzled,
            REPEAT=requests_per_row,
            HAS_LENGTHS=lengths is not None,
            num_warps=num_warps,
        )


def compact_nvfp4_scratch_numel(groups: int, max_keys: int, out_rows: int) -> int:
    """int32 elements of :func:`compact_nvfp4_kv_cache`'s ``scratch``: a prefix
    row per group, the groups' sizes and key ends, and a key per output row.

    Parameters
    ----------
    groups : int
        Groups of the launch.
    max_keys : int
        The launch's ``max_keys``.
    out_rows : int
        Rows of the launch's ``out``.

    Returns
    -------
    int
        The workspace size in int32 elements.
    """
    return groups * (-(-max_keys // 32) + 2) + out_rows


def compact_nvfp4_bitmap_numel(groups: int, max_keys: int) -> int:
    """int32 elements of :func:`compact_nvfp4_kv_cache`'s ``bitmap``: a bit per
    key and a key end per group.

    Parameters
    ----------
    groups : int
        Groups of the launch.
    max_keys : int
        The launch's ``max_keys``.

    Returns
    -------
    int
        The workspace size in int32 elements.
    """
    return groups * (-(-max_keys // 32) + 1)


@flashinfer_api(trace=compact_nvfp4_kv_cache_trace)
def compact_nvfp4_kv_cache(
    paged_kv_cache: tuple[torch.Tensor, torch.Tensor],
    kv_cache_sf: tuple[torch.Tensor, torch.Tensor],
    positions: torch.Tensor,
    row_groups: torch.Tensor,
    group_requests: torch.Tensor,
    block_table: torch.Tensor,
    group_offsets: torch.Tensor,
    *,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scratch: torch.Tensor,
    bitmap: torch.Tensor,
    max_keys: int,
    max_capacity: int,
    zero_row: int,
    group_base: int = 0,
    group_dense: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    requests_per_row: int = 1,
    v_sf_swizzled: bool = True,
    tokens_per_program: int = 32,
    num_warps: int = 4,
) -> None:
    """Convert each selected NVFP4 token once per group of rows, and point the
    rows' K/V index lists at the converted tokens.

    Rows that share a request (its prefill tokens, or its 1 + k MTP verify
    tokens) select mostly the same keys. This converts each group's union of
    selected keys once,
    into the group's region of ``out`` (rows ``group_offsets[g]`` on), and
    writes each row's lists as indices into that region. A dense group
    (``group_dense[g] != 0``) instead converts every key below its last
    selected key, key ``p`` to region row ``p``: no bitmap, worth it when its
    rows select more keys in all than its context holds.

    Every FP8 row holds its cache token's converted values (see Notes), the same
    bits as :func:`gather_nvfp4_kv_cache_by_position` writes for that token,
    and entry ``j`` of a row's lists holds the same key as entry ``j`` of its
    positions, so token-sparse attention over either result computes on the
    same values in the same order. Entries that are not live (negative keys,
    entries past the row's count, keys at or past ``max_keys``) point at
    ``zero_row``, which is zeroed. Each row's count is its number of nonnegative keys.

    Parameters
    ----------
    paged_kv_cache : tuple[torch.Tensor, torch.Tensor]
        K/V uint8 HND pages, each shaped
        ``[pages, 1, page_size, head_dim // 2]`` (two E2M1 values per byte),
        with matching strides and contiguous token data. ``page_size`` is
        positive, and a multiple of four for swizzled V scales; ``head_dim`` is
        a power of two, 64 or more.
    kv_cache_sf : tuple[torch.Tensor, torch.Tensor]
        K/V FP8 E4M3 block-scale pages, each shaped
        ``[pages, 1, page_size, head_dim // 16]``, with matching strides.
        K scales are linear. V scales follow ``v_sf_swizzled``.
    v_sf_swizzled : bool
        Whether V scales use TRT-LLM's HND four-token swizzle, as in caches
        written for trtllm-gen attention; ``page_size`` must then be a
        multiple of four. Pass False for linear V scales, as written by
        :func:`flashinfer.page.nvfp4_quantize_append_paged_kv_cache`.
    positions : torch.Tensor
        int32 ``[rows, top_k]`` key positions, valid ones first, then -1;
        ``top_k`` a power of two.
    row_groups : torch.Tensor
        int32 ``[rows]``: each row's group plus ``group_base``. A negative value
        marks a padding row: its lists point at ``zero_row``. Rows of groups
        outside ``[0, groups)`` (other than padding) are skipped, their lists and
        counts left as they are, so launches over different group ranges
        (``group_base``) can share ``positions`` and the output tables.
    group_requests : torch.Tensor
        int32 ``[groups]``: each group's row of ``block_table``.
    block_table : torch.Tensor
        int32 ``[requests, pages]``.
    group_offsets : torch.Tensor
        int64 ``[groups]``: each group's first row of ``out``. A union group's
        region must hold its union (at most its rows times ``top_k``, at most
        its keys); a dense group's must hold its keys up to the last selected.
        Regions must not overlap each other or ``zero_row``.
    out : torch.Tensor
        FP8 E4M3 ``[N, 1, 1, 2 * head_dim]``, K then V.
    block_tables, seq_lens : torch.Tensor
        int32 outputs ``[rows * requests_per_row, 2, top_k]`` and
        ``[rows * requests_per_row]`` (or more rows); row ``r``'s lists and
        count go to its ``requests_per_row`` consecutive rows.
    scratch : torch.Tensor
        int32 workspace of :func:`compact_nvfp4_scratch_numel` ``(groups,
        max_keys, N)`` elements or more; its contents do not matter.
    bitmap : torch.Tensor
        int32 workspace of :func:`compact_nvfp4_bitmap_numel` ``(groups,
        max_keys)`` elements or more, all zero; every call leaves it all zero
        (zero it once, and do not share it with other uses).
    max_keys : int
        Keys at or past it are not live (they point at ``zero_row``, and the
        row's count still includes them); sizes the bitmap.
    max_capacity : int
        Sizes the conversion grid: at least every group's region size (a union
        group's union of live keys, a dense group's last live key plus one).
        Region rows at or past it are not converted.
    zero_row : int
        A row of ``out`` outside every region.
    group_dense : torch.Tensor, optional
        int32 ``[groups]``, nonzero for dense groups; None: all union groups.
    lengths : torch.Tensor, optional
        int32 ``[rows]``: each row's count of valid keys, when entries past it
        may hold stale values (they are then not live). Entries within the count
        must be valid keys: a negative one points at ``zero_row`` and is still
        counted, so attention weights a zero token. None: count the nonnegative
        keys.
    group_base : int
        The ``row_groups`` value of this launch's group 0 (see ``row_groups``).
    requests_per_row : int
        Attention requests per row (see ``block_tables``).
    tokens_per_program, num_warps : int
        Launch tuning (powers of two): the region rows each conversion program
        converts at a time, and its warps.

    Graph safe: four launches, no host synchronization.

    Notes
    -----
    All tensors must be on the same CUDA device. Outputs are written in place.
    An FP8 row holds the token's decoded NVFP4 values (each E2M1 value times its
    E4M3 block scale) divided by six, excluding the global scales: multiply
    BMM1's K scale and BMM2's V scale by six. This covers NVFP4's full
    normalized range without FP8 saturation. The result is rounded to E4M3:
    products of E2M1 values and E4M3 scales can need more mantissa bits, and
    small scales can become subnormal or zero. Requires a Blackwell GPU
    (compute capability 10.x, 11.x or 12.x); validated on SM103.
    """
    from ._token_sparse_nvfp4 import (
        compact_convert_kernel,
        compact_mark_kernel,
        compact_number_kernel,
        compact_remap_kernel,
    )

    dim = _nvfp4_cache_dim(paged_kv_cache, kv_cache_sf, v_sf_swizzled)
    _check_launch(requests_per_row, tokens_per_program, num_warps)
    device = paged_kv_cache[0].device
    if group_offsets.ndim != 1:
        raise ValueError("Expected int64 group_offsets [groups]")
    rows, groups = positions.shape[0], group_offsets.shape[0]
    top_k = _check_selection(
        positions,
        block_table,
        out,
        block_tables,
        seq_lens,
        rows * requests_per_row,
        dim,
        device,
    )
    _check_vector("row_groups", row_groups, rows, torch.int32, device)
    _check_vector("group_requests", group_requests, groups, torch.int32, device)
    _check_vector("group_offsets", group_offsets, groups, torch.int64, device)
    if group_dense is not None:
        _check_vector("group_dense", group_dense, groups, torch.int32, device)
    _check_lengths(lengths, rows, device)
    if not 0 <= zero_row < out.shape[0] or max_keys <= 0 or max_capacity < 0:
        raise ValueError(
            "zero_row must be a row of out; max_keys positive; max_capacity >= 0"
        )
    words = -(-max_keys // 32)
    numel = compact_nvfp4_scratch_numel(groups, max_keys, out.shape[0])
    if (
        scratch.dtype != torch.int32
        or scratch.numel() < numel
        or not scratch.is_contiguous()
        or scratch.device != device
    ):
        raise ValueError(f"scratch needs {numel} contiguous int32 elements")
    bitmap_numel = compact_nvfp4_bitmap_numel(groups, max_keys)
    if (
        bitmap.dtype != torch.int32
        or bitmap.numel() < bitmap_numel
        or not bitmap.is_contiguous()
        or bitmap.device != device
    ):
        raise ValueError(f"bitmap needs {bitmap_numel} contiguous int32 elements")
    if groups > 65535:
        raise ValueError("At most 65535 groups")
    max_tiles = max(1, -(-max_capacity // tokens_per_program))
    if not rows:
        return
    # A grid of every group times the largest region's tiles is mostly idle
    # programs when regions differ; above _LOOP_ABOVE_FACTOR times
    # _convert_programs(device), that many programs loop over their group's own
    # tiles instead. A launch without groups only remaps (padding rows point at
    # the zero row) and zeroes the zero row.
    cap = _convert_programs(device)
    loop = groups > 0 and max_tiles * groups > _LOOP_ABOVE_FACTOR * cap
    tiles = min(max_tiles, max(1, -(-cap // groups))) if loop else max_tiles
    if tiles >= 1 << 31:
        raise ValueError("At most 2**31 - 1 conversion tiles per group")
    scratch, bitmap = scratch.view(-1), bitmap.view(-1)[:bitmap_numel]
    prefix = scratch[: groups * words]
    sizes = scratch[groups * words : groups * (words + 1)]
    ends = scratch[groups * (words + 1) : groups * (words + 2)]
    union_keys = scratch[groups * (words + 2) : numel]
    has_dense, has_lengths = group_dense is not None, lengths is not None
    dense = group_dense if has_dense else sizes  # unread without dense groups
    row_lengths = lengths if has_lengths else positions  # unread without lengths
    k, v = paged_kv_cache
    ks, vs = kv_cache_sf
    with torch.cuda.device(k.device):
        if groups:
            compact_mark_kernel[(rows,)](
                positions,
                positions.stride(0),
                row_lengths,
                row_groups,
                group_base,
                groups,
                dense,
                bitmap,
                words,
                max_keys,
                TOP_K=top_k,
                HAS_DENSE=has_dense,
                HAS_LENGTHS=has_lengths,
                num_warps=4,
            )
            compact_number_kernel[(groups,)](
                bitmap,
                prefix,
                sizes,
                ends,
                dense,
                words,
                HAS_DENSE=has_dense,
                num_warps=4,
            )
        compact_remap_kernel[(rows,)](
            positions,
            positions.stride(0),
            row_lengths,
            row_groups,
            group_base,
            groups,
            # without groups, padding rows read none of these (non-empty stand-ins)
            dense if groups else positions,
            group_offsets if groups else positions,
            bitmap if groups else positions,
            prefix if groups else positions,
            words,
            max_keys,
            block_tables,
            seq_lens,
            union_keys,
            zero_row,
            TOP_K=top_k,
            REPEAT=requests_per_row,
            HAS_DENSE=has_dense,
            HAS_LENGTHS=has_lengths,
            num_warps=4,
        )
        if not groups:
            out[zero_row].zero_()
            return
        compact_convert_kernel[(tiles, groups)](
            k,
            v,
            ks,
            vs,
            k.stride(0),
            ks.stride(0),
            block_table,
            block_table.stride(0),
            group_requests,
            group_offsets,
            dense,
            sizes,
            union_keys,
            bitmap,
            words,
            ends,
            out,
            out[zero_row],
            PAGE=k.shape[2],
            DIM=dim,
            TOKENS=tokens_per_program,
            MAX_CAPACITY=max_capacity,
            V_SF_SWIZZLED=v_sf_swizzled,
            HAS_DENSE=has_dense,
            LOOP=loop,
            num_warps=num_warps,
        )
