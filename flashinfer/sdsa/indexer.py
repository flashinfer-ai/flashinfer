# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""APIs of the SDSA (Simplified DSA) indexer, which scores keys with one query
head and no head weights. These APIs score its keys, select the keys that score
above a threshold, set the thresholds from a key sample, write the Top-K keys of
the selection, and quantize its keys and queries to NVFP4."""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..cute_dsl.availability import is_cute_dsl_experimental_available
from ..trace.templates.sdsa import (
    gather_key_sample_trace,
    indexer_scores_trace,
    indexer_threshold_select_trace,
    quantize_write_index_qk_nvfp4_trace,
    thresholds_from_sample_trace,
    top_k_from_candidates_trace,
)
from ..utils import get_compute_capability, get_device_sm_count


@flashinfer_api(trace=indexer_scores_trace)
def indexer_scores(
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
    On SM10x it runs a CuTe DSL kernel; on other GPUs (e.g. Hopper) it scores BF16
    keys with a Triton kernel, and NVFP4 keys are not supported.

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
        E4M3 ``[pages, 64, 8]`` for NVFP4 keys, paged like the keys. Pages can
        be any multiple of 8 bytes apart. None for BF16.
    """
    from .shapes import DIM, MAX_ROWS, PAGE

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
    if get_compute_capability(query.device)[0] != 10:
        # The CuTe scorer needs SM10x (tcgen05); elsewhere, a Triton kernel scores
        # BF16 keys: one program per request group and 64-key page.
        if key_scales is not None or {query.dtype, key_cache.dtype} != {torch.bfloat16}:
            raise ValueError(
                "Off SM10x, queries and keys must be BF16 (NVFP4 needs SM10x)"
            )
        # The kernel reads DIM-value keys PAGE to a page and contiguous rows.
        if query.shape[1] != DIM or key_cache.shape[1:] != (PAGE, DIM):
            raise ValueError(
                f"Keys must be [pages, {PAGE}, {DIM}] and queries [rows, {DIM}]"
            )
        strides = (
            query.stride(1),
            key_cache.stride(2),
            scores.stride(1),
            block_table.stride(1),
            context_lengths.stride(0),
            requests.stride(0),
        )
        if strides != (1,) * len(strides):
            raise ValueError(
                "Queries, keys, scores, lengths, requests and the block table "
                "must have contiguous rows"
            )
        from .indexer_triton import sdsa_scorer_kernel

        sdsa_scorer_kernel[(batch, -(-max_context // PAGE))](
            query,
            query.stride(0),
            key_cache,
            key_cache.stride(0),
            key_cache.stride(1),
            block_table,
            block_table.stride(0),
            context_lengths,
            requests,
            scores,
            scores.stride(0),
            ROWS=rows,
        )
        return
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("indexer_scores needs CuTe DSL 4.7 or newer")
    from .indexer_sm100 import CTAS_PER_SM, TILE, TILES_PER_CTA

    # CTAs per request group: two per SM in all, and enough that a CTA's run over
    # the whole capacity fits the TILES_PER_CTA tiles whose pages it stages (the
    # lengths stay on the device). CTAs past a group's length exit right away.
    key_tiles = (max_context + TILE - 1) // TILE
    runs = -(-key_tiles // TILES_PER_CTA)
    workers = max(CTAS_PER_SM * get_device_sm_count(query.device) // batch, runs, 1)
    keys = (query, query_scales, key_cache, key_scales)
    tables = (context_lengths, requests, block_table)
    outputs = (None, None, None, None, scores)
    _score("decode", rows, keys, (*tables, *outputs, max_context, workers, 0, 0))


def _score(mode: str, rows: int, keys: tuple, args: tuple) -> None:
    """Check the key format, then run the scorer in `mode` for `rows` query rows
    per job. `keys` is (query, query_scales, key_cache, key_scales), and `args`
    holds the kernel's other arguments."""
    from ..cute_dsl.launch import argument_kind, compiled
    from . import indexer_sm100
    from .indexer_sm100 import (
        DIM,
        PACKED_DIM,
        PAGE,
        SCALES_PER_KEY,
    )

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
    scale_dtypes = (torch.float8_e4m3fn, torch.uint8)
    if nvfp4 and (
        key_scales.dtype not in scale_dtypes or query_scales.dtype not in scale_dtypes
    ):
        raise ValueError("NVFP4 scales must be float8_e4m3fn, or their uint8 bytes")
    if (
        key_cache.stride()[1:] != (dim, 1)
        or nvfp4
        and key_scales.stride()[1:] != (SCALES_PER_KEY, 1)
    ):
        raise ValueError("A page's keys and their scales must be contiguous rows")
    # The kernel compiles for 16-byte aligned key pages (TMA) and 8-byte aligned
    # scale pages (cute_dsl.launch.argument_kind).
    if key_cache.stride(0) * key_cache.element_size() % 16 or (
        nvfp4 and key_scales.stride(0) % 8
    ):
        raise ValueError(
            "Key pages must be a multiple of 16 bytes apart, and scale pages of 8"
        )
    if get_compute_capability(query.device)[0] != 10:
        raise ValueError("Single-head paged MQA kernels require SM10x")
    tensors = (query, query_scales, *args)
    if not all(t.is_contiguous() for t in tensors if isinstance(t, torch.Tensor)):
        raise ValueError("Queries, lengths, tables and outputs must be contiguous")
    args = (*keys, *args)
    function = compiled(
        indexer_sm100.scorers(mode, rows, nvfp4),
        "sdsa_indexer",
        f"{mode}_{rows}_{'nvfp4' if nvfp4 else 'bf16'}",
        (),
        query.device,
        tuple(map(argument_kind, args)),
        (indexer_sm100.__file__,),
    )
    function(*args)


@flashinfer_api(trace=indexer_threshold_select_trace)
def indexer_threshold_select(
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
        As in :func:`indexer_scores`, in tiles of 128 rows.
    key_cache : torch.Tensor
        As in :func:`indexer_scores`.
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
        NVFP4 scales, as in :func:`indexer_scores`.
    arrivals : torch.Tensor, optional
        int32 ``[tiles]``: zeroed once, and left zeroed by every call.
    ranges : torch.Tensor, optional
        int32 ``[2, rows]``, output: the range of slots holding each row's
        candidates and ties. Other slots are unspecified.
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("Threshold select needs CuTe DSL 4.7 or newer")
    from .indexer_sm100 import (
        COUNTERS,
        PAGE,
        SELECT_CTAS_PER_TILE,
        SELECT_MIN_TILES,
        SELECT_TILES_PER_CTA,
        TILE,
    )

    tiles, width = tile_requests.shape[0], block_table.shape[1]
    key_tiles = (max_context + TILE - 1) // TILE
    if min(tiles, key_cache.shape[0], width, key_tiles) <= 0:
        raise ValueError("queries, keys, the block table and context must be nonempty")
    per_row = {t.shape[0] for t in (query, row_ends, thresholds, counters, candidates)}
    if per_row != {tiles * TILE}:
        raise ValueError(f"Mismatched query rows ({TILE} per tile) and per-row outputs")
    if counters.shape[1:] != (COUNTERS,) or candidates.shape[2:] != (2,):
        raise ValueError(
            f"counters must be [rows, {COUNTERS}] and candidates [rows, slots, 2]"
        )
    int32 = (row_ends, tile_requests, block_table, counters, candidates)
    if any(t.dtype != torch.int32 for t in int32) or thresholds.dtype != torch.float32:
        raise ValueError(
            "row_ends, tile_requests, block_table, counters and candidates must be "
            "int32, and thresholds float32"
        )
    if max_context > PAGE * width:
        raise ValueError("The block table must cover the context capacity")
    if (arrivals is None) != (ranges is None):
        raise ValueError("The repair takes both arrivals and ranges")
    if ranges is not None and (
        arrivals.shape[0] < tiles
        or ranges.shape != (2, tiles * TILE)
        or {arrivals.dtype, ranges.dtype} != {torch.int32}
    ):
        raise ValueError(
            "arrivals must hold a count per tile and ranges [2, rows], both int32"
        )
    if not 0 <= top_k < candidates.shape[1]:
        raise ValueError(
            "candidates must hold the top_k tie slots and at least one more"
        )
    if ranges is not None and candidates.shape[1] - top_k < top_k - 1:
        raise ValueError("The repair needs candidate slots for top_k - 1 keys")
    # CTAs per query tile: enough that each CTA's run stays within
    # SELECT_TILES_PER_CTA, and at least SELECT_CTAS_PER_TILE while each keeps
    # SELECT_MIN_TILES key tiles, since one CTA per SM leaves the SM idle on
    # short contexts. Rounded up to whole waves (every SM the same number of
    # CTAs) and shared out so the first `wide` query tiles get one CTA more.
    sms = get_device_sm_count(query.device)
    runs = max(
        -(-key_tiles // SELECT_TILES_PER_CTA),
        min(SELECT_CTAS_PER_TILE, key_tiles // SELECT_MIN_TILES),
    )
    workers, wide = divmod(sms * -(-tiles * runs // sms), tiles)
    keys = (query, query_scales, key_cache, key_scales)
    tables = (row_ends, tile_requests, block_table)
    outputs = (thresholds, counters, arrivals, ranges, candidates)
    _score("select", TILE, keys, (*tables, *outputs, max_context, workers, wide, top_k))


@flashinfer_api(trace=top_k_from_candidates_trace)
def top_k_from_candidates(
    candidates: torch.Tensor,
    ranges: torch.Tensor,
    *,
    top_keys: torch.Tensor,
) -> None:
    """Write each query row's exact Top-K keys from the candidates and ties that
    :func:`indexer_threshold_select` stored in its range of slots.

    The select stores scores that keep the order of the row's scores, so the
    keys of the row's ``top_k`` largest stored scores are the keys of its
    ``top_k`` largest scores. A CTA takes a row: it finds the ``top_k``-th
    largest score from counts of the score bits, then writes the keys above it
    and those equal to it. It runs as one launch without host synchronization,
    so it can be captured in a CUDA graph (run it once before capture to compile
    it).

    Parameters
    ----------
    candidates : torch.Tensor
        int32 ``[rows, slots, 2]``: the select's (key, score bits) pairs.
    ranges : torch.Tensor
        int32 ``[2, rows]``: the select's ranges, the first and past-last slot
        of each row's pairs.
    top_keys : torch.Tensor
        int32 ``[rows, top_k]``, output: the keys of the row's ``top_k``
        largest scores, in no set order, and any of those equal to the
        ``top_k``-th. A row of at most ``top_k`` pairs gets all its keys, then
        -1.
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("top_k_from_candidates needs CuTe DSL 4.7 or newer")
    from ..cute_dsl.launch import launch
    from ..topk_varlen.kernels import block_scan
    from . import indexer_sm100
    from .indexer_sm100 import TOP_K_THREADS

    rows = candidates.shape[0]
    if (
        candidates.shape[2:] != (2,)
        or ranges.shape != (2, rows)
        or top_keys.dim() != 2
        or top_keys.shape[0] != rows
    ):
        raise ValueError(
            "candidates must be [rows, slots, 2], ranges [2, rows], and "
            "top_keys [rows, top_k]"
        )
    args = (candidates, ranges, top_keys)
    if not all(t.dtype == torch.int32 and t.is_contiguous() for t in args):
        raise ValueError("candidates, ranges and top_keys must be contiguous int32")
    if top_keys.shape[1] >= 1 << 16:
        raise ValueError("top_k must be below 65536")
    if rows and top_keys.shape[1]:
        launch(
            indexer_sm100.candidate_top_k_kernel,
            (rows, 1),
            TOP_K_THREADS,
            args,
            (),
            module="sdsa_top_k",
            key_files=(indexer_sm100.__file__, block_scan.__file__),
        )


@flashinfer_api(trace=quantize_write_index_qk_nvfp4_trace)
def quantize_write_index_qk_nvfp4(
    key: torch.Tensor,
    slots: torch.Tensor,
    query: torch.Tensor,
    query_rows: torch.Tensor,
    *,
    key_cache: torch.Tensor,
    key_scales: torch.Tensor,
    packed_query: torch.Tensor,
    query_scales: torch.Tensor,
) -> None:
    """Quantize indexer keys and queries to NVFP4 in one launch: the keys into a
    paged key cache and the queries into rows, the layouts
    :func:`indexer_scores` reads.

    Each key and query gets one E4M3 scale per 16 values: their amax over 6.
    There is no per-tensor global scale, so queries and keys are quantized the
    same way. Write all keys of a cache at the same scale: do not scale them by
    a factor that changes between calls, or their scores stop being comparable.
    Values above 2688 in magnitude saturate.

    It runs as one launch without host synchronization, so it can be captured
    in a CUDA graph (run it once before capture to compile it).

    Parameters
    ----------
    key : torch.Tensor
        BF16 or FP16 ``[tokens, 128]``.
    slots : torch.Tensor
        int64 ``[tokens]``: each key's cache slot ``s``. Negative slots are
        skipped.
    query : torch.Tensor
        BF16 or FP16 ``[queries, 128]``.
    query_rows : torch.Tensor
        int64 ``[queries]``: each query's row ``r``. Negative rows are skipped.
    key_cache : torch.Tensor
        uint8 ``[pages, 64, 64]``, output: a key's 64 packed E2M1 bytes go to
        ``key_cache[s // 64, s % 64]``.
    key_scales : torch.Tensor
        float8_e4m3fn ``[pages, 64, 8]``, output: a key's 8 scales go to
        ``key_scales[s // 64, s % 64]``.
    packed_query : torch.Tensor
        uint8 ``[rows, 64]``, output: a query's 64 packed E2M1 bytes go to row
        ``r``.
    query_scales : torch.Tensor
        float8_e4m3fn ``[rows, 8]``, output: a query's 8 scales go to row ``r``.
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("quantize_write_index_qk_nvfp4 needs CuTe DSL 4.7 or newer")
    from ..cute_dsl.launch import launch
    from . import indexer_sm100
    from .indexer_sm100 import (
        DIM,
        PACKED_DIM,
        PAGE,
        SCALES_PER_KEY,
        TILE,
        WRITE_ROWS_PER_CTA,
    )

    if (
        key.shape[1:] != (DIM,)
        or query.shape[1:] != (DIM,)
        or slots.shape[0] < key.shape[0]
        or query_rows.shape[0] < query.shape[0]
    ):
        raise ValueError(
            f"Keys and queries must be [tokens, {DIM}], with a slot or row each"
        )
    if (
        key_cache.shape[1:] != (PAGE, PACKED_DIM)
        or key_scales.shape != (key_cache.shape[0], PAGE, SCALES_PER_KEY)
        or packed_query.shape[1:] != (PACKED_DIM,)
        or query_scales.shape != (packed_query.shape[0], SCALES_PER_KEY)
    ):
        raise ValueError(
            f"key_cache must be [pages, {PAGE}, {PACKED_DIM}] and key_scales "
            f"[pages, {PAGE}, {SCALES_PER_KEY}]; packed_query [rows, {PACKED_DIM}] "
            f"and query_scales [rows, {SCALES_PER_KEY}]"
        )
    scale_dtypes = (torch.float8_e4m3fn, torch.uint8)
    if (
        key.dtype not in (torch.bfloat16, torch.float16)
        or query.dtype not in (torch.bfloat16, torch.float16)
        or {slots.dtype, query_rows.dtype} != {torch.int64}
        or {key_cache.dtype, packed_query.dtype} != {torch.uint8}
        or key_scales.dtype not in scale_dtypes
        or query_scales.dtype not in scale_dtypes
    ):
        raise ValueError(
            "key and query must be BF16 or FP16, slots and query_rows int64, "
            "key_cache and packed_query uint8, and the scales float8_e4m3fn"
        )
    if get_compute_capability(key.device)[0] != 10:
        raise ValueError("quantize_write_index_qk_nvfp4 requires SM10x")
    rows = key.shape[0] + query.shape[0]
    if rows:
        args = (key, slots, query, query_rows, key_cache, key_scales)
        launch(
            indexer_sm100.index_qk_write_kernel,
            (-(-rows // WRITE_ROWS_PER_CTA), 1),
            TILE,
            (*args, packed_query, query_scales),
            (),
            module="sdsa_indexer",
            key_files=(indexer_sm100.__file__,),
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
    :func:`indexer_threshold_select`.

    Viewed as 64-key pages (``sample_keys.view(-1, 64, d)``), the sample can be
    scored by :func:`indexer_scores`. It runs as one launch
    without host synchronization, so it can be captured in a CUDA graph.

    Parameters
    ----------
    key_cache : torch.Tensor
        As in :func:`indexer_scores`.
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
        NVFP4 key scales, as in :func:`indexer_scores`.
    sample_scales : torch.Tensor, optional
        ``[requests, capacity, 8]``, output: the sampled keys' NVFP4 scales.
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("gather_key_sample needs CuTe DSL 4.7 or newer")
    from ..cute_dsl.launch import launch
    from . import indexer_sm100
    from .indexer_sm100 import PAGE, SCALES_PER_KEY, TILE

    requests, capacity, width = sample_keys.shape
    if capacity % TILE or (key_scales is None) != (sample_scales is None):
        raise ValueError(
            f"Sample rows hold whole {TILE}-key tiles, with scales if NVFP4"
        )
    # The kernel writes SCALES_PER_KEY scales to every slot below a sample end.
    if key_scales is not None and (
        key_scales.shape != (key_cache.shape[0], PAGE, SCALES_PER_KEY)
        or sample_scales.shape != (requests, capacity, SCALES_PER_KEY)
    ):
        raise ValueError(
            f"Scales must be [pages, {PAGE}, {SCALES_PER_KEY}] and "
            f"[requests, capacity, {SCALES_PER_KEY}]"
        )
    if sample_keys.dtype != key_cache.dtype or (
        key_scales is not None and sample_scales.dtype != key_scales.dtype
    ):
        raise ValueError("Samples must have the dtypes of the keys and their scales")
    if (
        key_cache.shape[1:] != (PAGE, width)
        or sample_ends.shape[0] != requests
        or offsets.shape[0] < capacity
        or block_table.shape[0] < requests
        or samples_per_run < 1
    ):
        raise ValueError("Mismatched sample rows, keys, sample ends, offsets or table")
    args = (key_cache, key_scales, block_table, offsets, sample_ends, sample_keys)
    if requests and capacity:
        launch(
            indexer_sm100.key_sample_kernel,
            (requests, capacity // TILE),
            TILE,
            (*args, sample_scales),
            (samples_per_run, PAGE * pages_per_run, width),
            module="sdsa_indexer",
            key_files=(indexer_sm100.__file__,),
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
    :func:`indexer_threshold_select` from a sample of its scores,
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
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("thresholds_from_sample needs CuTe DSL 4.7 or newer")
    from ..cute_dsl.launch import launch
    from . import indexer_sm100
    from .indexer_sm100 import COUNTERS, THRESHOLD_ROWS_PER_CTA

    rows, rank = sample_top.shape
    if {sample_scores.shape[0], thresholds.shape[0], counters.shape[0]} != {rows}:
        raise ValueError("Mismatched sample rows, thresholds or counters")
    if counters.shape[1:] != (COUNTERS,):
        raise ValueError(f"counters must be [rows, {COUNTERS}]")
    if rows:
        launch(
            indexer_sm100.thresholds_kernel,
            (-(-rows // THRESHOLD_ROWS_PER_CTA), 1),
            32 * THRESHOLD_ROWS_PER_CTA,
            (sample_scores, sample_top, thresholds, counters),
            (rank,),
            module="sdsa_indexer",
            key_files=(indexer_sm100.__file__,),
        )
