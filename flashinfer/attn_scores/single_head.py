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
    gather_key_sample_trace,
    quantize_write_index_k_nvfp4_trace,
    single_head_paged_mqa_logits_trace,
    single_head_paged_mqa_threshold_select_trace,
    thresholds_from_sample_trace,
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
    dot product of a query row and a key. Query rows come in groups of 1 to 8
    consecutive rows, such as a decode step's token and its draft tokens.
    It runs as one launch without host synchronization, so it can be captured in a
    CUDA graph (run it once before capture to compile it). Queries, keys, pages and lengths can change between replays.

    Parameters
    ----------
    query : torch.Tensor
        BF16 ``[rows, 128]``, or NVFP4 uint8 ``[rows, 64]`` (E2M1 values, two to
        a byte).
    key_cache : torch.Tensor
        BF16 ``[pages, 64, 128]``, or NVFP4 uint8 ``[pages, 64, 64]``. Pages can
        be any multiple of 16 bytes apart, such as views of one cache that holds
        each page's keys and then their scales.
    block_table : torch.Tensor
        int32 ``[requests, blocks]``: each request's pages of 64 keys. Only pages
        below a group's length are read.
    context_lengths : torch.Tensor
        int32 ``[groups]``: each group's number of keys.
    requests : torch.Tensor
        int32 ``[groups]``: each group's request, a row of ``block_table``. A
        group of length 0 reads nothing, so its entry can hold any value.
    scores : torch.Tensor
        float32 ``[rows, max_context]``, output: ``scores[q, k]`` is the dot
        product of query row ``q`` with key ``k``, for ``k`` below its group's
        length. Later entries are not written. ``max_context`` sizes the grid,
        and the block table must cover it.
    query_scales : torch.Tensor, optional
        E4M3 ``[rows, 8]`` for NVFP4 queries, a scale per 16 values. None for
        BF16.
    key_scales : torch.Tensor, optional
        E4M3 ``[pages, 64, 8]`` for NVFP4 keys, paged like the keys. None for
        BF16.
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("single_head_paged_mqa_logits needs CuTe DSL 4.7 or newer")
    from .kernels.single_head_scorer import (
        CTAS_PER_SM,
        MAX_ROWS,
        PAGE,
        TILE,
        TILES_PER_CTA,
    )

    max_context = scores.shape[1]
    batch, width = requests.shape[0], block_table.shape[1]
    if min(batch, max_context, key_cache.shape[0], width) <= 0:
        raise ValueError("queries, keys, scores and the block table must be nonempty")
    rows = scores.shape[0] // batch
    if not 1 <= rows <= MAX_ROWS or rows * batch != scores.shape[0]:
        raise ValueError(f"each request must serve 1 to {MAX_ROWS} query rows")
    if query.shape[0] != scores.shape[0] or context_lengths.shape[0] != batch:
        raise ValueError("Mismatched query rows, scores, lengths or requests")
    if max_context > PAGE * width:
        raise ValueError("The block table must cover the context capacity")
    # CTAs per request group: two per SM in all, and enough that a CTA's run over
    # the whole capacity fits the TILES_PER_CTA tiles whose pages it stages (the
    # lengths stay on the device). CTAs past a group's length exit right away.
    key_tiles = (max_context + TILE - 1) // TILE
    runs = -(-key_tiles // TILES_PER_CTA)
    workers = max(CTAS_PER_SM * get_device_sm_count(query.device) // batch, runs, 1)
    keys = (query, query_scales, key_cache, key_scales)
    tables = (context_lengths, requests, block_table)
    outputs = (None, None, None, None, scores)
    _score("decode", rows, keys, (*tables, *outputs, max_context, workers, 0))


def _score(mode: str, rows: int, keys: tuple, args: tuple) -> None:
    """Check the key format, then run the scorer in `mode` for `rows` query rows
    per job. `keys` is (query, query_scales, key_cache, key_scales), and `args`
    holds the kernel's other arguments."""
    from ..cute_dsl.launch import argument_kind, compiled
    from .kernels import single_head_scorer
    from .kernels.single_head_scorer import DIM, PACKED_DIM, PAGE, SCALES_PER_KEY

    query, query_scales, key_cache, key_scales = keys
    nvfp4 = key_cache.dtype == torch.uint8
    if (
        key_cache.dtype not in (torch.uint8, torch.bfloat16)
        or query.dtype != key_cache.dtype
    ):
        raise ValueError("Queries and keys must be both BF16 or both NVFP4 (uint8)")
    if nvfp4 != (key_scales is not None) or nvfp4 != (query_scales is not None):
        raise ValueError("NVFP4 keys and queries need scales, BF16 ones none")
    dim = PACKED_DIM if nvfp4 else DIM  # elements of key_cache's dtype per key
    if query.shape[1] != dim or key_cache.shape[1:] != (PAGE, dim):
        raise ValueError(
            f"Keys must be [pages, {PAGE}, {dim}] and queries [rows, {dim}]"
        )
    if nvfp4 and (
        key_scales.shape != (key_cache.shape[0], PAGE, SCALES_PER_KEY)
        or query_scales.shape != (query.shape[0], SCALES_PER_KEY)
    ):
        raise ValueError(
            f"Scales must be [pages, {PAGE}, {SCALES_PER_KEY}] and [rows, {SCALES_PER_KEY}]"
        )
    if (
        key_cache.stride()[1:] != (dim, 1)
        or nvfp4
        and key_scales.stride(1) != SCALES_PER_KEY
    ):
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
    arrivals: Optional[torch.Tensor] = None,
    ranges: Optional[torch.Tensor] = None,
) -> None:
    """Select the keys that score above each query row's threshold, in one pass
    over the row's paged keys and without writing its scores.

    A key that scores above its row's threshold is a candidate, and one that
    scores neither above nor below it (equal, or NaN) is a tie. Each is counted
    and stored as a ``(key index, score bits)`` pair in the row's candidate or
    tie slots, its score less the threshold where that is exact (scores at most
    twice it), which keeps their order. A row fits when its candidates fit
    ``capacity`` and, with at most ``top_k`` ties, reach
    ``min(row_ends[r], top_k)``. Its exact Top-``top_k`` keys are then among its
    slots, and a threshold at or a little below its ``top_k``-th largest score
    makes it fit.

    With ``arrivals`` and ``ranges``, and a ``capacity`` of at least
    ``top_k - 1``, every row fits. Any threshold then gives the same result, and
    only decides how often a repair runs: the last CTA of a query tile to finish
    sweeps the tile again for each row that does not fit, and selects it again
    at its radix-selected ``min(row_ends[r], top_k)``-th largest score. It runs as one launch without host synchronization, so it can be captured in a
    CUDA graph (run it once before capture to compile it).

    Parameters
    ----------
    query : torch.Tensor
        As in :func:`single_head_paged_mqa_logits`, in tiles of 128 rows.
    key_cache : torch.Tensor
        As in :func:`single_head_paged_mqa_logits`.
    row_ends : torch.Tensor
        int32 ``[rows]``: row ``r`` sees keys below ``row_ends[r]``.
    tile_requests : torch.Tensor
        int32 ``[tiles]``: each query tile's request, a row of ``block_table``.
    block_table : torch.Tensor
        int32 ``[requests, blocks]``: each request's pages of 64 keys.
    thresholds : torch.Tensor
        float32 ``[rows]``: each row's threshold. A repaired row gets its new
        threshold.
    counters : torch.Tensor
        int32 ``[rows, 2]``, zeroed by the caller: each row's candidate and tie
        counts.
    candidates : torch.Tensor
        int32 ``[rows, top_k + capacity, 2]``, output: the ``n``-th candidate at
        ``top_k + n`` if ``n < capacity``, the ``n``-th tie at
        ``top_k - 1 - n`` if ``n < top_k``. Slot order is unspecified.
    max_context : int
        The most keys a row sees. It sizes the grid, and the block table must
        cover it.
    top_k : int
        The number of tie slots.
    query_scales, key_scales : torch.Tensor, optional
        NVFP4 scales, as in :func:`single_head_paged_mqa_logits`.
    arrivals : torch.Tensor, optional
        int32 ``[tiles]``: zeroed once, and left zeroed by every call.
    ranges : torch.Tensor, optional
        int32 ``[2, rows]``, output: the range of slots holding each row's
        candidates and ties. Other slots are unspecified.
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("Threshold select needs CuTe DSL 4.7 or newer")
    from .kernels.single_head_scorer import COUNTERS, PAGE, SELECT_TILES_PER_CTA, TILE

    tiles, width = tile_requests.shape[0], block_table.shape[1]
    key_tiles = (max_context + TILE - 1) // TILE
    if min(tiles, key_cache.shape[0], width, key_tiles) <= 0:
        raise ValueError("queries, keys, the block table and context must be nonempty")
    per_row = {t.shape[0] for t in (query, row_ends, thresholds, counters, candidates)}
    if per_row != {tiles * TILE} or counters.shape[1:] + candidates.shape[2:] != (
        COUNTERS,
        2,
    ):
        raise ValueError(f"Mismatched query rows ({TILE} per tile) and per-row outputs")
    if max_context > PAGE * width:
        raise ValueError("The block table must cover the context capacity")
    if (arrivals is None) != (ranges is None):
        raise ValueError("The repair takes both arrivals and ranges")
    if ranges is not None and (
        arrivals.shape[0] < tiles or ranges.shape != (2, tiles * TILE)
    ):
        raise ValueError("arrivals must hold a count per tile and ranges [2, rows]")
    if not 0 <= top_k < candidates.shape[1]:
        raise ValueError(
            "candidates must hold the top_k tie slots and at least one more"
        )
    if ranges is not None and candidates.shape[1] - top_k < top_k - 1:
        raise ValueError("The repair needs candidate slots for top_k - 1 keys")
    # CTAs per query tile: the fewest that give every SM the same number of CTAs
    # and keep each CTA's run within SELECT_TILES_PER_CTA. tiles * workers is a
    # multiple of the SM count exactly when workers is a multiple of unit.
    sms = get_device_sm_count(query.device)
    unit = sms // math.gcd(tiles, sms)
    runs = -(-key_tiles // SELECT_TILES_PER_CTA)
    workers = unit * -(-runs // unit)
    keys = (query, query_scales, key_cache, key_scales)
    tables = (row_ends, tile_requests, block_table)
    outputs = (thresholds, counters, arrivals, ranges, candidates)
    _score("select", TILE, keys, (*tables, *outputs, max_context, workers, top_k))


@flashinfer_api(trace=quantize_write_index_k_nvfp4_trace)
def quantize_write_index_k_nvfp4(
    key: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    key_scales: torch.Tensor,
) -> None:
    """Quantize indexer keys to NVFP4 and write them to a paged key cache, in the
    layout :func:`single_head_paged_mqa_logits` reads.

    Each key gets one E4M3 scale per 16 values, their amax over 6 (with a unit
    global scale). It runs as one launch without host synchronization, so it can be captured in a
    CUDA graph (run it once before capture to compile it).

    Parameters
    ----------
    key : torch.Tensor
        BF16 or FP16 ``[tokens, 128]``.
    slots : torch.Tensor
        int64 ``[tokens]``: each key's cache slot ``s``. Negative slots are
        skipped.
    key_cache : torch.Tensor
        uint8 ``[pages, 64, 64]``, output: a key's 64 packed E2M1 bytes go to
        ``key_cache[s // 64, s % 64]``.
    key_scales : torch.Tensor
        float8_e4m3fn ``[pages, 64, 8]``, output: a key's 8 scales go to
        ``key_scales[s // 64, s % 64]``.
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("quantize_write_index_k_nvfp4 needs CuTe DSL 4.7 or newer")
    from ..cute_dsl.launch import launch
    from .kernels import single_head_scorer
    from .kernels.single_head_scorer import (
        DIM,
        PACKED_DIM,
        PAGE,
        SCALES_PER_KEY,
        TILE,
        WRITE_KEYS_PER_CTA,
    )

    if key.shape[1] != DIM or key_cache.shape[1:] != (PAGE, PACKED_DIM):
        raise ValueError(
            f"Keys must be [tokens, {DIM}] and the cache [pages, {PAGE}, {PACKED_DIM}]"
        )
    if key_scales.shape[1:] != (PAGE, SCALES_PER_KEY) or slots.shape[0] < key.shape[0]:
        raise ValueError(
            f"key_scales must be [pages, {PAGE}, {SCALES_PER_KEY}], with a slot per key"
        )
    if key.shape[0]:
        launch(
            single_head_scorer.index_k_write_kernel,
            (-(-key.shape[0] // WRITE_KEYS_PER_CTA), 1),
            TILE,
            (key, slots, key_cache, key_scales),
            (),
            module="single_head_scorer",
            key_files=(single_head_scorer.__file__,),
        )


@flashinfer_api(trace=gather_key_sample_trace)
def gather_key_sample(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    offsets: torch.Tensor,
    sample_ends: torch.Tensor,
    *,
    sample_keys: torch.Tensor,
    samples_per_run: int,
    pages_per_run: int,
    key_scales: Optional[torch.Tensor] = None,
    sample_scales: Optional[torch.Tensor] = None,
) -> None:
    """Copy a strided random sample of each request's paged keys into dense
    sample rows, whose scores give thresholds for
    :func:`single_head_paged_mqa_threshold_select`.

    Viewed as 64-key pages (``sample_keys.view(-1, 64, d)``), the sample can be
    scored by :func:`single_head_paged_mqa_logits`. It runs as one launch
    without host synchronization, so it can be captured in a CUDA graph.

    Parameters
    ----------
    key_cache : torch.Tensor
        As in :func:`single_head_paged_mqa_logits`.
    block_table : torch.Tensor
        int32 ``[requests, blocks]``: each request's pages of 64 keys.
    offsets : torch.Tensor
        uint8 ``[capacity]``: sample slot ``s`` takes the key at ``offsets[s]``
        (below ``64 * pages_per_run``) into run ``s // samples_per_run``.
    sample_ends : torch.Tensor
        int32 ``[requests]``: each request's number of samples. Later slots are
        left as they are.
    sample_keys : torch.Tensor
        BF16 ``[requests, capacity, 128]`` or NVFP4 uint8 ``[..., 64]``, output,
        ``capacity`` a multiple of 128.
    samples_per_run : int
        Samples per run of ``pages_per_run`` pages.
    pages_per_run : int
        Pages per run.
    key_scales : torch.Tensor, optional
        NVFP4 key scales, as in :func:`single_head_paged_mqa_logits`.
    sample_scales : torch.Tensor, optional
        ``[requests, capacity, 8]``, output: the sampled keys' NVFP4 scales.
    """
    from ..cute_dsl.launch import launch
    from .kernels import single_head_scorer
    from .kernels.single_head_scorer import PAGE, TILE

    requests, capacity, width = sample_keys.shape
    if capacity % TILE or (key_scales is None) != (sample_scales is None):
        raise ValueError(
            f"Sample rows hold whole {TILE}-key tiles, with scales if NVFP4"
        )
    if (
        width != key_cache.shape[2]
        or sample_ends.shape[0] != requests
        or offsets.shape[0] < capacity
        or block_table.shape[0] < requests
        or samples_per_run < 1
    ):
        raise ValueError("Mismatched sample rows, keys, sample ends, offsets or table")
    args = (key_cache, key_scales, block_table, offsets, sample_ends, sample_keys)
    if requests and capacity:
        launch(
            single_head_scorer.key_sample_kernel,
            (requests, capacity // TILE),
            TILE,
            (*args, sample_scales),
            (samples_per_run, PAGE * pages_per_run, width),
            module="single_head_scorer",
            key_files=(single_head_scorer.__file__,),
        )


@flashinfer_api(trace=thresholds_from_sample_trace)
def thresholds_from_sample(
    sample_scores: torch.Tensor,
    sample_top: torch.Tensor,
    *,
    thresholds: torch.Tensor,
    counters: torch.Tensor,
) -> None:
    """Set each query row's threshold for
    :func:`single_head_paged_mqa_threshold_select` from a sample of its scores,
    and zero its counters. It runs as one launch without host synchronization,
    so it can be captured in a CUDA graph.

    Parameters
    ----------
    sample_scores : torch.Tensor
        float32 ``[rows, samples]``: each row's sampled scores.
    sample_top : torch.Tensor
        int32 ``[rows, rank]``: each row's Top positions among its samples.
    thresholds : torch.Tensor
        float32 ``[rows]``, output: the least of the row's sampled Top scores. A
        -1 position or a NaN score counts as -inf, since dummy and capture passes
        can feed NaN from uninitialized memory.
    counters : torch.Tensor
        int32 ``[rows, 2]``, output: zeroed.
    """
    from ..cute_dsl.launch import launch
    from .kernels import single_head_scorer
    from .kernels.single_head_scorer import THRESHOLD_ROWS_PER_CTA

    rows, rank = sample_top.shape
    if {sample_scores.shape[0], thresholds.shape[0], counters.shape[0]} != {rows}:
        raise ValueError("Mismatched sample rows, thresholds or counters")
    if rows:
        launch(
            single_head_scorer.thresholds_kernel,
            (-(-rows // THRESHOLD_ROWS_PER_CTA), 1),
            32 * THRESHOLD_ROWS_PER_CTA,
            (sample_scores, sample_top, thresholds, counters),
            (rank,),
            module="single_head_scorer",
            key_files=(single_head_scorer.__file__,),
        )
