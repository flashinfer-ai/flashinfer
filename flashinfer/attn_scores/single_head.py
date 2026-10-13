# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""Single-head paged MQA logits, for sparse attention whose indexer scores keys
with one query head and no head weights."""

import math
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..cute_dsl.availability import is_cute_dsl_experimental_available
from ..trace.templates.attn_scores import (
    quantize_write_index_k_nvfp4_trace,
    single_head_paged_mqa_logits_trace,
    single_head_paged_mqa_threshold_select_trace,
)
from ..utils import get_compute_capability, get_device_sm_count


@flashinfer_api(trace=single_head_paged_mqa_logits_trace)
def single_head_paged_mqa_logits(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    requests: torch.Tensor,
    scores: torch.Tensor,
    query_scales: Optional[torch.Tensor] = None,
    key_scales: Optional[torch.Tensor] = None,
) -> None:
    """Compute FP32 scores of query rows against their request's paged keys.

    This is :func:`fp8_paged_mqa_logits` for an indexer with one query head.
    With one head there is no head weight and no ReLU, so a score is the plain
    dot product of a query row and a key.

    Keys have 128 dimensions and are paged 64 to a page, in BF16 or NVFP4. BF16:
    ``key_cache`` is ``[pages, 64, 128]`` and ``query`` is ``[rows, 128]``, with
    no scales. NVFP4 (E2M1 values, two to a byte, with an E4M3 scale per 16):
    ``key_cache`` is uint8 ``[pages, 64, 64]`` with ``key_scales``
    ``[pages, 64, 8]``, and ``query`` is uint8 ``[rows, 64]`` with
    ``query_scales`` ``[rows, 8]``. Pages can be any multiple of 16 bytes apart,
    such as views of one cache that holds each page's keys and then their
    scales. ``block_table`` (int32 ``[requests, blocks]``) maps each request's
    64-key blocks to their pages, for keys and scales alike. Query rows come in
    groups of 1 to 8 consecutive rows (``rows // requests.shape[0]``), such as a
    decode step's token and its draft tokens. Group ``j`` reads row
    ``requests[j]`` of the block table, up to length ``context_lengths[j]``.

    ``scores[q, k]`` is the dot product of query row ``q`` with key ``k`` of its
    request, for ``k`` below its group's length. Nothing past the length is
    written or read, so the block table only needs valid pages below it, and a
    group of length 0 can hold any request. ``scores`` is caller-owned float32
    ``[rows, max_context]``, and the block table must cover ``max_context``,
    which sizes the grid. It runs as one launch without host synchronization,
    so it can be captured in a CUDA graph (run it once before capture to
    compile it). Queries, keys, pages and lengths can change between replays.
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("single_head_paged_mqa_logits needs CuTe DSL 4.7 or newer")
    from .kernels.single_head_scorer import TILES_PER_CTA

    max_context = scores.shape[1]
    batch, width = requests.shape[0], block_table.shape[1]
    if min(batch, max_context, key_cache.shape[0], width) <= 0:
        raise ValueError("queries, keys, scores and the block table must be nonempty")
    rows = scores.shape[0] // batch
    if not 1 <= rows <= 8 or rows * batch != scores.shape[0]:
        raise ValueError("each request must serve 1 to 8 query rows")
    if query.shape[0] != scores.shape[0] or context_lengths.shape[0] != batch:
        raise ValueError("Mismatched query rows, scores, lengths or requests")
    if max_context > 64 * width:
        raise ValueError("The block table must cover the context capacity")
    # CTAs per request group: two per SM in all, and enough that a CTA's run over
    # the whole capacity fits the TILES_PER_CTA tiles whose pages it stages (the
    # lengths stay on the device). CTAs past a group's length exit right away.
    key_tiles = (max_context + 127) // 128
    runs = -(-key_tiles // TILES_PER_CTA)
    workers = max(2 * get_device_sm_count(query.device) // batch, runs, 1)
    keys = (query, query_scales, key_cache, key_scales)
    tables = (context_lengths, requests, block_table)
    _score("decode", rows, keys, (*tables, None, None, scores, max_context, workers, 0))


def _score(mode: str, rows: int, keys: tuple, args: tuple) -> None:
    """Check the key format, then run the scorer in `mode` for `rows` query rows
    per job. `keys` is (query, query_scales, key_cache, key_scales), and `args`
    holds the kernel's other arguments."""
    from ..cute_dsl.launch import argument_kind, compiled
    from .kernels import single_head_scorer

    query, query_scales, key_cache, key_scales = keys
    nvfp4 = key_cache.dtype == torch.uint8
    if (
        key_cache.dtype not in (torch.uint8, torch.bfloat16)
        or query.dtype != key_cache.dtype
    ):
        raise ValueError("Queries and keys must be both BF16 or both NVFP4 (uint8)")
    if nvfp4 != (key_scales is not None) or nvfp4 != (query_scales is not None):
        raise ValueError("NVFP4 keys and queries need scales, BF16 ones none")
    dim = 64 if nvfp4 else 128  # elements of key_cache's dtype per key
    if query.shape[1] != dim or key_cache.shape[1:] != (64, dim):
        raise ValueError("Keys must be [pages, 64, 128] and queries [rows, 128]")
    if nvfp4 and (key_scales.shape[1:] != (64, 8) or query_scales.shape[1] != 8):
        raise ValueError("Scales must be [pages, 64, 8] and [rows, 8]")
    if key_cache.stride()[1:] != (dim, 1) or nvfp4 and key_scales.stride(1) != 8:
        raise ValueError("A page's keys and their scales must be contiguous rows")
    if get_compute_capability(query.device)[0] != 10:
        raise ValueError("Single-head paged MQA kernels require SM10x")
    tensors = (query, query_scales, *args)
    if not all(t.is_contiguous() for t in tensors if isinstance(t, torch.Tensor)):
        raise ValueError("Queries, lengths, tables and outputs must be contiguous")
    args = (*keys, *args)
    function = compiled(
        single_head_scorer.scorers(mode, rows, nvfp4),
        "single_head_scorer",
        f"{mode}_{rows}_{'nvfp4' if nvfp4 else 'bf16'}",
        (),
        query.device,
        tuple(map(argument_kind, args)),
        (single_head_scorer.__file__,),
    )
    function(*args)


@flashinfer_api(trace=single_head_paged_mqa_threshold_select_trace)
def single_head_paged_mqa_threshold_select(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    row_ends: torch.Tensor,
    tile_requests: torch.Tensor,
    block_table: torch.Tensor,
    *,
    thresholds: torch.Tensor,
    counters: torch.Tensor,
    candidates: torch.Tensor,
    max_context: int,
    top_k: int,
    query_scales: Optional[torch.Tensor] = None,
    key_scales: Optional[torch.Tensor] = None,
) -> None:
    """Select the keys that score above each query row's threshold, in one pass
    over the row's paged keys and without writing its scores.

    Keys, queries and pages are as in :func:`single_head_paged_mqa_logits`.
    Query tile ``t`` (rows ``128t`` to ``128t + 127``) scores the keys of request
    ``tile_requests[t]`` below ``max_context``, which sizes the grid. Row ``r``
    sees keys below ``row_ends[r]``.

    Row ``r``'s threshold is ``thresholds[r]`` (float32 ``[rows]``). A key that
    scores above it is a candidate, and one that scores neither above nor below
    it (equal, or NaN) is a tie. ``counters[r]`` (int32 ``[rows, 2]``, zeroed by
    the caller) counts the row's candidates and ties, and each is stored as a
    ``(key index, score bits)`` pair in ``candidates[r]`` (int32
    ``[rows, top_k + capacity, 2]``): the ``n``-th candidate at ``top_k + n`` if
    ``n < capacity``, the ``n``-th tie at ``top_k - 1 - n`` if ``n < top_k``.
    Slot order is unspecified. Scores are stored less the row's threshold where
    that is exact (scores at most twice it), which keeps their order.

    A row fits when its candidates fit ``capacity`` and, with at most ``top_k``
    ties, reach ``min(row_ends[r], top_k)``, as its counters tell. Its exact
    Top-``top_k`` keys are then among its written slots, and a threshold at or a
    little below its ``top_k``-th largest score makes it fit. It runs as one
    launch without host synchronization, so it can be captured in a CUDA graph
    (run it once before capture to compile it).
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("Threshold select needs CuTe DSL 4.7 or newer")
    from .kernels.single_head_scorer import SELECT_TILES_PER_CTA

    tiles, width = tile_requests.shape[0], block_table.shape[1]
    key_tiles = (max_context + 127) // 128
    if min(tiles, key_cache.shape[0], width, key_tiles) <= 0:
        raise ValueError("queries, keys, the block table and context must be nonempty")
    per_row = {t.shape[0] for t in (query, row_ends, thresholds, counters, candidates)}
    if per_row != {tiles * 128} or counters.shape[1:] + candidates.shape[2:] != (2, 2):
        raise ValueError("Mismatched query rows (128 per tile) and per-row outputs")
    if max_context > 64 * width:
        raise ValueError("The block table must cover the context capacity")
    if not 0 <= top_k < candidates.shape[1]:
        raise ValueError(
            "candidates must hold the top_k tie slots and at least one more"
        )
    # CTAs per query tile: the fewest that give every SM the same number of CTAs
    # and keep each CTA's run within SELECT_TILES_PER_CTA. tiles * workers is a
    # multiple of the SM count exactly when workers is a multiple of unit.
    sms = get_device_sm_count(query.device)
    unit = sms // math.gcd(tiles, sms)
    runs = -(-key_tiles // SELECT_TILES_PER_CTA)
    workers = unit * -(-runs // unit)
    keys = (query, query_scales, key_cache, key_scales)
    tables = (row_ends, tile_requests, block_table)
    outputs = (thresholds, counters, candidates)
    _score("select", 128, keys, (*tables, *outputs, key_tiles * 128, workers, top_k))


@flashinfer_api(trace=quantize_write_index_k_nvfp4_trace)
def quantize_write_index_k_nvfp4(
    key: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    key_scales: torch.Tensor,
) -> None:
    """Quantize indexer keys to NVFP4 and write them to a paged key cache.

    Key ``i`` (row ``i`` of ``key``, BF16 or FP16 ``[tokens, 128]``) gets one
    E4M3 scale per 16 values, their amax over 6 (with a unit global scale). It
    is written at cache slot ``s = slots[i]`` (int64), in the layout
    :func:`single_head_paged_mqa_logits` reads: its 64 packed E2M1 bytes go to
    ``key_cache[s // 64, s % 64]`` (uint8 ``[pages, 64, 64]``) and its 8 scales
    to ``key_scales[s // 64, s % 64]`` (float8_e4m3fn ``[pages, 64, 8]``).
    Negative slots are skipped. It runs as one launch without host
    synchronization, so it can be captured in a CUDA graph (run it once before
    capture to compile it)."""
    from ..cute_dsl.launch import launch
    from .kernels import single_head_scorer

    if key.shape[1] != 128 or key_cache.shape[1:] != (64, 64):
        raise ValueError("Keys must be [tokens, 128] and the cache [pages, 64, 64]")
    if key_scales.shape[1:] != (64, 8) or slots.shape[0] < key.shape[0]:
        raise ValueError("key_scales must be [pages, 64, 8], with a slot per key")
    if key.shape[0]:
        launch(
            single_head_scorer.index_k_write_kernel,
            ((key.shape[0] + 7) // 8, 1),
            128,
            (key, slots, key_cache, key_scales),
            (),
            module="single_head_scorer",
            key_files=(single_head_scorer.__file__,),
        )
