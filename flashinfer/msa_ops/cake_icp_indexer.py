"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MiniMax-M3 index-context-parallel (ICP) indexer kernels.

The index keys of every sequence are sharded by context across the 2 or 4
ranks of a tensor-parallel group, and every rank holds all index heads' FP8
queries.  These entry points are the per-rank kernels of the indexer: score
this rank's keys per logical 128-token block (B128), select each (token, head)
row's local Top16, and merge the candidates of all ranks on the rank that owns
the head.  The collective part (allocating and exchanging the receive storage,
the prefill all-to-all) stays with the caller; the kernels only read and write
caller-owned buffers, never allocate, and never synchronize with the host.

Supported layouts: TP2 / TP4 with P128 pages (token rows interleaved across
ranks) or P256 pages (contiguous 128-row halves for TP2, 64-row quarters for
TP4).  Ranking is always on global B128 block ids: scores descending (NaN
ranks as ``-inf``), forced begin/end blocks first, exact ties to the lower id.
Compute capability 10.0, 10.3 and 10.7 only.
"""

from typing import Any, Optional

import torch

from ..api_logging import flashinfer_experimental_api

_FEATURE = (
    "MiniMax-M3 ICP indexer kernels (sharded index scores, Top16 select/publish, merge)"
)


def _cake(backend: str):
    if backend != "cake":
        raise ValueError(
            f"MSA ICP indexer supports backend='cake' only, got {backend!r}"
        )
    from ..experimental.cake_msa_icp_indexer import cake_backend

    return cake_backend


@flashinfer_experimental_api(feature=_FEATURE)
def msa_icp_index_score_sharded_plan(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    num_qo_heads: int,
    *,
    qo_offset: torch.Tensor,
    device: Optional[torch.device] = None,
    backend: str = "cake",
):
    r"""Plan the sharded prefill scorer from host segment lengths (eager only).

    Parameters
    ----------
    qo_segment_lens, kv_segment_lens, qo_offset : torch.Tensor
        int32 CPU ``[B]``: query tokens, key length and query start position
        (causal prefix) of every segment.  The lengths are read on the host.
    num_qo_heads : int
        Index heads; must be 4.
    device : Optional[torch.device]
        Device of the plan storage (default: the current CUDA device).
    backend : str
        Only ``"cake"``.

    Returns
    -------
    MSAIcpScorePlan
        The plan; its storage is allocated here.  Use
        :func:`msa_icp_index_score_sharded_plan_device` for CUDA Graphs.
    """
    return _cake(backend).index_score_sharded_plan(
        qo_segment_lens,
        kv_segment_lens,
        num_qo_heads,
        qo_offset=qo_offset,
        device=device,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def msa_icp_index_score_sharded_plan_device(
    qo_segment_lens: torch.Tensor,
    kv_segment_lens: torch.Tensor,
    qo_segment_offsets: torch.Tensor,
    kv_segment_offsets: torch.Tensor,
    qo_offset: torch.Tensor,
    num_qo_heads: int,
    *,
    total_q: int,
    max_query_len: int,
    max_seq_len: int,
    packed_work_range: torch.Tensor,
    packed_work_info: torch.Tensor,
    workspace: torch.Tensor,
    backend: str = "cake",
):
    r"""Plan the sharded prefill scorer on the device into caller storage.

    Graph safe: no device value is read on the host.  The plan is reused by
    every layer of the step; keep the storage of each concurrently live plan
    distinct.

    Parameters
    ----------
    qo_segment_lens, kv_segment_lens, qo_offset : torch.Tensor
        int32 CUDA ``[B]``, contiguous.  Zero-length segments are allowed.
    qo_segment_offsets, kv_segment_offsets : torch.Tensor
        int32 CUDA ``[B + 1]`` prefix sums of the lengths, contiguous.
    num_qo_heads : int
        Index heads; must be 4.
    total_q, max_query_len, max_seq_len : int
        Host upper bounds of the query tokens, the longest query segment and
        the longest key length.
    packed_work_range : torch.Tensor
        int64 ``[num_SMs]``, contiguous.
    packed_work_info : torch.Tensor
        int64, at least ``((total_q + 127) // 128 + B - 1) * num_qo_heads``
        elements (:func:`msa_icp_index_score_plan_storage`).
    workspace : torch.Tensor
        uint8, at least 32 MiB (accepted for MSA compatibility; not written).
    backend : str
        Only ``"cake"``.

    Returns
    -------
    MSAIcpScorePlan
    """
    return _cake(backend).index_score_sharded_plan_device(
        qo_segment_lens,
        kv_segment_lens,
        qo_segment_offsets,
        kv_segment_offsets,
        qo_offset,
        num_qo_heads,
        total_q=total_q,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        packed_work_range=packed_work_range,
        packed_work_info=packed_work_info,
        workspace=workspace,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def msa_icp_index_score_sharded(
    q: torch.Tensor,
    k: torch.Tensor,
    block_table: torch.Tensor,
    plan,
    *,
    rank: int,
    max_score: torch.Tensor,
    sm_scale: float = 1.0,
    initialize: bool = True,
    tp: int = 2,
    compact: bool = False,
    enable_pdl: bool = False,
    backend: str = "cake",
) -> None:
    r"""Score a prefill query chunk against this rank's index-K shard.

    For every token ``t``, head ``h`` and score column ``c``, writes the raw
    FP32 maximum of ``q[t, h] . k_row`` over this rank's keys of the column's
    B128 block that the token may see (causal), ``-inf`` when it sees none.

    Parameters
    ----------
    q : torch.Tensor
        float8_e4m3fn ``[T, 4, 128]``, contiguous; ``T`` is the plan's ``total_q``.
    k : torch.Tensor
        float8_e4m3fn ``[pages, P / tp, 128]``: this rank's shard, dense rows,
        16-byte aligned, page pitch a multiple of 16 bytes.
    block_table : torch.Tensor
        int32 ``[B, >= pages per segment]``, unit last stride: physical page ids,
        the same on every rank.
    plan : MSAIcpScorePlan
        From :func:`msa_icp_index_score_sharded_plan_device`.
    rank, tp : int
        This rank and the group size (2 or 4).
    max_score : torch.Tensor
        float32 ``[T, 4, K]``, unit last stride (a strided view is allowed),
        ``K`` at least the plan's B128 columns (half of them, rounded up, when
        ``compact``).  Columns past the token's causal prefix are written only
        when ``initialize`` (filled with ``-inf`` first).
    sm_scale : float
        Accepted for MSA compatibility; must be finite; scores are unscaled.
    compact : bool
        P256 only: one column per local page (global id ``2 c + rank * 2 // tp``)
        instead of one per global B128 block.
    enable_pdl : bool
        Launch with programmatic dependent launch.
    backend : str
        Only ``"cake"``.
    """
    _cake(backend).index_score_sharded(
        q,
        k,
        block_table,
        plan,
        rank=rank,
        max_score=max_score,
        sm_scale=sm_scale,
        initialize=initialize,
        tp=tp,
        compact=compact,
        enable_pdl=enable_pdl,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def msa_icp_topk_select_prefill(
    scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    carrier: torch.Tensor,
    *,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
    index_scale: int = 1,
    index_offset: int = 0,
    backend: str = "cake",
) -> None:
    r"""Write each (head, token) row's local Top16 into a prefill carrier.

    Parameters
    ----------
    scores : torch.Tensor
        float32 ``[T, 4, K]``, ``0 < K < 12288``, unit last stride.
    num_valid_pages : torch.Tensor
        int32 ``[T]``: the token's global B128 block count (0 for padding rows).
        Local column ``c`` is global block ``c * index_scale + index_offset``;
        blocks at or past ``min(num_valid_pages, 8192)`` are not candidates.
    carrier : torch.Tensor
        int32 ``[4, T, 33]``, unit last stride (a token slice of a larger
        head-major buffer is allowed).  Row ``[h, t]`` receives 16 raw score
        words, 16 global ids ascending (``(-inf, -1)`` padding) and ``1``.
    force_begin_blocks, force_end_blocks : int
        Blocks ``< force_begin_blocks`` and ``>= num_valid_pages - force_end_blocks``
        are always selected; sum at most 16.
    index_scale, index_offset : int
        ``(1, 0)`` for P128, ``(2, rank * 2 // tp)`` for compact P256 scores.
    backend : str
        Only ``"cake"``.
    """
    _cake(backend).topk_select_prefill(
        scores,
        num_valid_pages,
        carrier,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        index_scale=index_scale,
        index_offset=index_offset,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def msa_icp_decode_index_score_sharded(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    score_out: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    page_size: int,
    decode_query_len: int,
    max_decode_query_len: int,
    enable_pdl: bool = False,
    backend: str = "cake",
) -> None:
    r"""Score decode queries against this rank's index-K shard.

    Writes ``score_out[t, h, c]`` for the local page columns ``c`` of the
    request's ``ceil(seq_len / page_size)`` pages rounded up to a multiple of
    four (P128: column = global B128 block; P256: column = local page, global
    block ``2 c + rank * 2 // world_size``); other columns are not written.

    Parameters
    ----------
    index_q : torch.Tensor
        float8_e4m3fn ``[B * decode_query_len, 4, 128]``, contiguous; token
        ``j`` of a request sits at position ``seq_len - decode_query_len + j``.
    index_k : torch.Tensor
        float8_e4m3fn ``[pages, page_size / world_size, 128]`` shard.
    block_table : torch.Tensor
        int32 ``[B, >= ceil(seq_len / page_size)]``, unit last stride.
    seq_lens : torch.Tensor
        int32 ``[B]``.
    score_out : torch.Tensor
        float32 ``[T, 4, cols]``, ``0 < cols <= 8192``, ``cols % 4 == 0``, unit
        last stride (:func:`msa_icp_decode_score_columns`).
    rank, world_size, page_size : int
        Layout: ``world_size`` 2 or 4, ``page_size`` 128 or 256.
    decode_query_len, max_decode_query_len : int
        ``1 <= decode_query_len <= max_decode_query_len``,
        ``4 * max_decode_query_len <= 32``.
    enable_pdl : bool
        Launch with programmatic dependent launch. On compute capability 10.0
        and 10.3 with 64 or 128 local K rows per page (P128 TP2, P256 TP2 and
        TP4) the scorer then reads ``seq_lens`` and ``block_table`` and starts
        loading the ``index_k`` pages that end at or before
        ``seq_len - decode_query_len`` before it waits on the preceding
        kernel. The kernel launched just before it must not write them, and
        must either be launched without programmatic dependent launch or wait
        on its own predecessor before it triggers its dependents, so that
        every earlier write is complete. ``enable_pdl=False`` removes the
        requirement.
    backend : str
        Only ``"cake"``.
    """
    _cake(backend).decode_index_score_sharded(
        index_q,
        index_k,
        block_table,
        seq_lens,
        score_out,
        rank=rank,
        world_size=world_size,
        page_size=page_size,
        decode_query_len=decode_query_len,
        max_decode_query_len=max_decode_query_len,
        enable_pdl=enable_pdl,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def msa_icp_topk_select_publish(
    scores: torch.Tensor,
    num_valid_pages: torch.Tensor,
    recv_bases: torch.Tensor,
    errors: torch.Tensor,
    *,
    slot: int,
    rank: int,
    world_size: int,
    heads_per_rank: int,
    capacity: int,
    force_begin_blocks: int,
    force_end_blocks: int,
    spin_limit: int = 1 << 26,
    enable_pdl: bool = False,
    generations: Optional[torch.Tensor] = None,
    index_scale: int = 1,
    index_offset: int = 0,
    backend: str = "cake",
) -> None:
    r"""Select each decode (token, head) row's local Top16 and store it into
    the receive row of the head's owner rank.

    Row ``(t, h)`` goes to peer ``h // heads_per_rank`` at row
    ``((slot * world_size + rank) * heads_per_rank + h % heads_per_rank) * capacity + t``
    of its legacy (int32, 33 words: 16 raw scores, 16 ascending global ids,
    ready flag) or tagged (int64, 32 words of ``generation << 32 | word``)
    receive rows.  Every row is written on every call, padding rows included.

    Parameters
    ----------
    scores : torch.Tensor
        float32 ``[T, world_size * heads_per_rank, K]``, ``0 < K < 12288``,
        unit last stride.
    num_valid_pages : torch.Tensor
        int32 ``[T]`` global B128 block counts.
    recv_bases : torch.Tensor
        int64 ``[world_size]`` device addresses of every rank's receive rows
        (the tagged rows' addresses when ``generations`` is given).
    errors : torch.Tensor
        int32 ``[>= 1]``; a legacy publish that waits ``spin_limit`` polls for
        a free row adds one and traps.
    slot : int
        Receive slot set (0, 1 or 2).
    capacity : int
        Receive rows per (slot, source, head); ``T <= capacity``.
    generations : Optional[torch.Tensor]
        int32 ``[3, world_size * heads_per_rank, capacity]`` publish counters:
        tagged mode.  ``None``: legacy mode.  Tagged mode needs at least two
        sparse layers between uses of one slot set; the receive storage and
        counters are zeroed once, never between steps.
    backend : str
        Only ``"cake"``.

    The remaining parameters are as in :func:`msa_icp_topk_select_prefill`.
    """
    _cake(backend).topk_select_publish(
        scores,
        num_valid_pages,
        recv_bases,
        errors,
        slot=slot,
        rank=rank,
        world_size=world_size,
        heads_per_rank=heads_per_rank,
        capacity=capacity,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        spin_limit=spin_limit,
        enable_pdl=enable_pdl,
        generations=generations,
        index_scale=index_scale,
        index_offset=index_offset,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def msa_icp_decode_score_publish(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_valid_pages: torch.Tensor,
    pool: torch.Tensor,
    recv_bases: torch.Tensor,
    errors: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    page_size: int,
    decode_query_len: int,
    slot: int,
    capacity: int,
    force_begin_blocks: int,
    force_end_blocks: int,
    spin_limit: int = 1 << 26,
    generations: Optional[torch.Tensor] = None,
    enable_pdl: bool = True,
    backend: str = "cake",
) -> None:
    r"""Fused decode scorer and publisher: the receive rows equal those of
    :func:`msa_icp_decode_index_score_sharded` followed by
    :func:`msa_icp_topk_select_publish`, without the score rows.

    ``pool`` is uint8 scratch of :func:`msa_icp_decode_pool_bytes` bytes.  The
    other parameters are as in those two functions.

    This path is an option; the default decode chain (and the facade's
    ``decode`` without ``pool``) is the score plane and the select publish.
    Each request is cut into ``min(clamp(num_sms * ctas_per_sm // batch, 1,
    128), cap)`` splits, ``cap`` the architecture's ``TOPK16_SPLIT_CAP`` (48),
    from host metadata only.
    """
    _cake(backend).decode_score_publish(
        index_q,
        index_k,
        block_table,
        seq_lens,
        num_valid_pages,
        pool,
        recv_bases,
        errors,
        rank=rank,
        world_size=world_size,
        page_size=page_size,
        decode_query_len=decode_query_len,
        slot=slot,
        capacity=capacity,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        spin_limit=spin_limit,
        generations=generations,
        enable_pdl=enable_pdl,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def msa_icp_merge_candidates(
    recv_slot: torch.Tensor,
    num_valid_pages: torch.Tensor,
    output: torch.Tensor,
    errors: torch.Tensor,
    *,
    init_blocks: int,
    local_blocks: int,
    spin_limit: int = 1 << 26,
    enable_pdl: bool = False,
    expected_generations: Optional[torch.Tensor] = None,
    prefill: bool = False,
    backend: str = "cake",
) -> None:
    r"""Merge every rank's Top16 candidates of the heads this rank owns.

    Per (token, owned head): ids outside ``[0, min(num_valid_pages, 8192))``
    are dropped, a duplicate id keeps its largest score (NaN counts as
    ``-inf``), ids ``< init_blocks`` or ``>= num_valid_pages - local_blocks``
    rank first, exact ties go to the lower id, and the Top16 ids are written
    ascending with a ``-1`` tail.  A prefill row with ``num_valid_pages == 0``
    is written ``[0, -1, ...]``; a decode one all ``-1``.

    Parameters
    ----------
    recv_slot : torch.Tensor
        One slot set of the receive rows: legacy int32 ``[W, hpr, capacity, 33]``
        (decode: waits for each ready flag, then clears it; prefill: the
        exchanged carrier, read without waiting) or tagged int64
        ``[W, hpr, capacity, 32]`` (with ``expected_generations``).
        Unit last stride.
    num_valid_pages : torch.Tensor
        int32 ``[T]``.
    output : torch.Tensor
        int32 ``[T, hpr, 16]``, unit last stride, ``T <= capacity``.
    errors : torch.Tensor
        int32 ``[>= 1]``; a wait of ``spin_limit`` polls adds one and traps.
    expected_generations : Optional[torch.Tensor]
        int32 ``[hpr, capacity]`` merge counters of the slot set (tagged mode).
    prefill : bool
        Prefill rows; incompatible with tagged mode and ``enable_pdl``.
    backend : str
        Only ``"cake"``.
    """
    _cake(backend).merge_candidates(
        recv_slot,
        num_valid_pages,
        output,
        errors,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        spin_limit=spin_limit,
        enable_pdl=enable_pdl,
        expected_generations=expected_generations,
        prefill=prefill,
    )


def msa_icp_index_score_plan_storage(
    *, num_qo_heads: int, total_q: int, batch: int, device: torch.device
) -> dict:
    r"""Sizes of a prefill plan's storage: ``work_range_numel`` (int64, one per
    SM), ``work_info_numel`` (int64) and ``workspace_bytes`` (uint8)."""
    return _cake("cake").index_score_plan_storage(
        num_qo_heads=num_qo_heads, total_q=total_q, batch=batch, device=device
    )


def msa_icp_index_score_columns(max_seq_len: int, page_size: int) -> int:
    r"""Prefill score columns: ``ceil(max_seq_len / 128)`` for P128, half of it
    (rounded up) for compact P256 scores."""
    return _cake("cake").index_score_columns(max_seq_len, page_size)


def msa_icp_decode_score_columns(max_model_len: int, page_size: int) -> int:
    r"""Decode score columns: ``min(8192, ceil(max_model_len / (4 * page_size)) * 4)``."""
    return _cake("cake").decode_score_columns(max_model_len, page_size)


def msa_icp_decode_pool_bytes(*, max_decode_tokens: int, num_heads: int = 4) -> int:
    r"""Bytes of the fused decode path's ``pool``."""
    return _cake("cake").decode_pool_bytes(
        max_decode_tokens=max_decode_tokens, num_heads=num_heads
    )


def msa_icp_comm_layout(world_size: int, heads_per_rank: int, capacity: int) -> dict:
    r"""Layout of the caller's decode receive storage.

    Returns a dict: ``storage_words`` (int32 words of one rank's symmetric
    allocation), ``legacy_offset_bytes`` (0) and ``legacy_shape``
    (int32 ``[3, W, hpr, capacity, 33]``), ``tagged_offset_bytes`` and
    ``tagged_shape`` (int64 ``[3, W, hpr, capacity, 32]``), and the int32
    counter shapes ``generations_shape`` ``[3, W * hpr, capacity]`` and
    ``expected_shape`` ``[3, hpr, capacity]``.

    The caller zeroes the storage and both counters once, on every rank, and
    never resets them.  Every publish writes all ``T x H`` rows.  Tagged rows
    need at least two sparse layers between uses of one slot set (including
    across steps); legacy rows are safe at any distance.
    """
    return _cake("cake").comm_layout(world_size, heads_per_rank, capacity)


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_msa_icp_indexer(
    *,
    world_size: int,
    rank: int,
    num_index_heads: int,
    page_size: int,
    init_blocks: int,
    local_blocks: int,
    max_decode_tokens: int,
    max_decode_query_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    max_model_len: int,
    num_sparse_layers: int,
    device: torch.device,
    backend: str = "cake",
) -> Any:
    r"""Build this rank's ICP indexer from the serving limits.

    The returned :class:`MSAIcpIndexerPlan` runs one layer's chains with
    caller-owned buffers:

    * ``decode(index_q, index_k, block_table, seq_lens, num_valid_pages,
      scores, output, *, comm, slot, tagged, decode_query_len,
      enable_pdl=True, pool=None)``:
      score plane -> Top16 select and publish -> owner merge (the default;
      its score plane follows the ``enable_pdl`` contract of
      :func:`msa_icp_decode_index_score_sharded`);
      with ``pool`` the fused scorer and publisher of
      :func:`msa_icp_decode_score_publish` replace the first two steps;
    * ``prefill_window_plan(window, ...)`` once per query window and step,
      ``prefill_local(query, index_k, block_table, window_plan, score_scratch,
      num_valid_pages, carrier_view)`` per query chunk, the caller's one
      exchange per layer, then ``prefill_merge(received, num_valid_pages,
      output)``.

    ``comm`` holds the receive storage views and counters laid out by
    :func:`msa_icp_comm_layout` (``recv``, ``tagged_recv``, ``recv_bases``,
    ``tagged_recv_bases``, ``publish_generations``, ``expected_generations``,
    ``errors``; check them once with ``check_comm``).  Call ``warmup()`` on
    every rank before the first decode or prefill exchange and before CUDA
    Graph capture: it loads every kernel into the device's context, so no
    first launch waits on module loading while a peer's merge spins.  The facade allocates only its prefill plan storage,
    here; no method allocates or synchronizes.

    Parameters
    ----------
    world_size, rank : int
        Tensor-parallel group size (2 or 4) and this rank.
    num_index_heads : int
        Index heads across the group; must be 4.
    page_size : int
        128 or 256.
    init_blocks, local_blocks : int
        Forced begin / end blocks (sum at most 16).
    max_decode_tokens, max_decode_query_len : int
        Receive capacity and the longest decode query (``4 * len <= 32``).
    max_num_batched_tokens, max_num_seqs, max_model_len : int
        Scheduler limits that size the prefill plan storage.
    num_sparse_layers : int
        Sparse indexer layers per step; tagged decode rows need at least two.
    device : torch.device
        The rank's CUDA device.
    backend : str
        Only ``"cake"``.
    """
    _cake(backend)
    from ..experimental.cake_msa_icp_indexer.cake_indexer import MSAIcpIndexerPlan

    return MSAIcpIndexerPlan(
        world_size=world_size,
        rank=rank,
        num_index_heads=num_index_heads,
        page_size=page_size,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        max_decode_tokens=max_decode_tokens,
        max_decode_query_len=max_decode_query_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        num_sparse_layers=num_sparse_layers,
        device=device,
    )
