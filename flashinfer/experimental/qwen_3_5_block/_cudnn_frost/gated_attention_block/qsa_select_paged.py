# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_select_paged`` -- block selection for a DECODE step over the PAGED compressed-key cache: the DSL-free host entry of the
paged decode scorer (``kernels/qsa_score_paged.py``) and the host-side facts of that kernel.

A decode step of a block-sparse (``QsaSpec``) layer scores each of its ``S_q`` query rows per sequence against the compressed
keys of the complete 4-token blocks the row can see, and the top-k of those scores is the row's block list.  The serving
stacks keep the compressed keys PAGED on the main block table (one compressed row per 4 tokens, ``page_size / 4`` rows per
page); the scorer kernel reads them there and the deterministic radix top-k follows -- two launches, no torch prologue::

    pos_t   = kv_lens[b] - S_q + s                       (row s of sequence b; t = b * S_q + s)
    n_vis_t = clamp(floor((pos_t + 1) / 4), 0, n_blocks_max)
    logits[t, j] = scale * sum_h relu(qi[t, h] . kbar[b, j])        for j < n_vis_t, kbar[b, j] = cache[table[b, j // R], j % R]
    counts[t]     = n_vis_t                                          (R = page_size / 4 compressed rows per page)
    block_lens[t] = min(top_k, n_vis_t)
    block_ids[t]  = the top_k of logits[t, :counts[t]], ASCENDING, -1 past block_lens[t]

**Placement (Rule 7).**  This module imports no CuTe DSL -- torch, the stream helpers and the version gates only -- so the
documented import path resolves on every installed DSL, the public 4.6.2 wheel (the floor vLLM and SGLang pin) included.
:func:`qsa_select_decode_paged` checks every operand's FORM first, then gates the DSL floor, the ``sm_107a`` target on cc 10.7
and the radix top-k's cc 9.0 -- each a typed ``NotImplementedError`` naming the version or the device -- and only THEN imports
the kernel module (whose import executes the DSL, like its neighbours) inside the function body.  The kernel's host-side
facts a declaration may need before any kernel is reached -- the MMA row map of the query rows, the compile-key validator, the
knob defaults, the byte model -- live here for the same reason, and the kernel module imports them from here (never the other way
round at module scope).  The kernel's SMEM table, barrier table, PDL sites and degenerate-input matrix are in the kernel
module's docstring; :func:`qsa_select_decode_paged` documents the selection's contract.

**Streams (Rule 5).**  The call resolves ONE torch stream from ``stream`` first (:func:`cudnn._torch_stream.launch_stream`: a
``torch.cuda.Stream``, a raw handle or a ``cuda.CUstream``; ``None`` = torch's current stream on the operands' device) and
runs every torch operation inside its context -- the scorer's output allocations, the scorer launch, the top-k wrapper's own
output and scratch allocations and its launch -- handing both kernels that stream's raw handle.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost._torch_stream import launch_stream, stream_context
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.api_base import TupleDict
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

__all__ = [
    "qsa_select_decode_paged",
    "INDEX_HEAD_DIM",
    "BLOCK_SIZE",
    "QSA_TOP_K",
    "TOP_K_KERNEL_MAX",
    "MAX_QUERY_ROWS",
    "UNITS_PER_WARP_CHOICES",
    "RING_DEPTH_CHOICES",
    "QueryRowMap",
    "padded_heads",
    "query_row_map",
    "keys_per_warp",
    "blocks_per_cta",
    "default_units_per_warp",
    "validate_params",
    "moved_bytes",
]

INDEX_HEAD_DIM = 128  # the head dim the indexer scorer packs (qi and kbar rows): the one geometry the paged scorer addresses
BLOCK_SIZE = 4  # tokens per compressed block (the compression ratio of the indexer)
QSA_TOP_K = 512  # blocks kept per row (2048 selected tokens / 4)
TOP_K_KERNEL_MAX = 2048  # the radix top-k kernel's bound on top_k
THREADS = 128  # 4 warps per CTA
WARPS = THREADS // 32
# The warp-level MMA the scorer runs: mma.sync.m16n8k16 (bf16 in, fp32 accumulate) -- M = 16 padded (s, h) query rows, N = 8 keys per
# n-tile, K = 16 of the 128 D per k-step.
MMA_M = 16
MMA_N = 8
MMA_K = 16
K_STEPS = INDEX_HEAD_DIM // MMA_K  # 8 k-steps per key row
KEYS_PER_NTILE = MMA_N  # 8 compressed rows per MMA n-tile
NTILES_PER_UNIT = 4  # n-tiles per warp unit = the register ring depth: every key load of a unit is in flight before its first MMA
NTILES_IN_FLIGHT = NTILES_PER_UNIT  # the default ring depth (n-tiles of keys in flight per warp); RING_DEPTH_CHOICES is the knob's domain
RING_DEPTH_CHOICES = (2, 4, 8)
ROWS_PER_UNIT = KEYS_PER_NTILE * NTILES_PER_UNIT  # 32 compressed rows per warp unit
UNITS_PER_WARP_CHOICES = (1, 2, 4, 8)  # 32 / 64 / 128 / 256 rows per warp = blocks_per_cta 128 / 256 / 512 / 1024 -- a performance knob, the same function
MAX_QUERY_ROWS = 64  # S_q x next_pow2(H_i) padded (s, h) MMA rows resident in registers per warp: 4 m-tiles of 16 (32 registers each)
MIN_PAGE_SIZE = 2 * BLOCK_SIZE  # at least two compressed rows per page (a power-of-two row count)
CHUNK_ELEMS = 8  # bf16 elements per 16-byte chunk
CHUNKS_PER_ROW = INDEX_HEAD_DIM // CHUNK_ELEMS  # 16
CHUNKS_PER_LANE = CHUNKS_PER_ROW // 4  # the 4 chunks `q + 4 j` lane (g, q) loads of a row: one 16-byte load each
# The radix top-k's input contract, by value (``deepseek_sparse_attention.indexer_top_k``: ``INPUT_ALIGN`` = 32 bytes, fp32 rows read in
# 256-bit vectors): spelled here because the preallocated-output checks run BEFORE the gate that makes that kernel module importable.
_SCORE_ROW_ALIGN = 8  # n_blocks_max must be a multiple of 8 fp32 (one 256-bit vector)
_TOPK_INPUT_ALIGN = 32  # bytes: the radix top-k's input alignment
_E4M3 = torch.float8_e4m3fn


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


class QueryRowMap(NamedTuple):
    """How the ``S_q x H_i`` (s, h) query rows sit in the MMA's M dimension (the kernel module's docstring, "the row map"): the
    heads of a query are padded to ``P = next_pow2(H_i)`` rows; ``P' = min(P, 16)`` of them share a 16-row m-tile with
    ``QPT = 16 / P'`` queries; a query wider than a tile spans ``TPQ = P / 16`` tiles.  Lane ``(g = lane / 4, q = lane % 4)`` of the
    warp holds rows ``g`` and ``g + 8`` of every tile: with ``P' >= 2`` those two rows are heads ``2 gi`` / ``2 gi + 1`` of ONE query
    (``gi = g % G``, ``G = P' / 2``) -- the head sum is one in-thread add plus ``stages = log2(G)`` butterfly shuffles over the lanes
    ``xor 4, 8, 16``; with ``P' = 1`` the two rows are two queries and nothing is summed."""

    h_i: int
    s_q: int
    padded: int  # P
    tile_rows: int  # P'
    queries_per_tile: int  # QPT
    tiles_per_query: int  # TPQ
    m_tiles: int
    group: int  # G = P' / 2 (0 at P' = 1): consecutive g values of one query inside a tile
    stages: int  # butterfly stages of the head sum (lane xor 4 << k)

    def row_of(self, t: int, r: int):
        """The (s, h) of padded row ``r`` of m-tile ``t`` (the lane's half = ``r // 8``), or ``None`` for a padding row."""
        g, half = r % 8, r // 8
        if self.tile_rows == 1:
            s, h = t * self.queries_per_tile + r, 0
        else:
            s = (t // self.tiles_per_query) * self.queries_per_tile + g // self.group
            h = 16 * (t % self.tiles_per_query) + 2 * (g % self.group) + half
        return (s, h) if s < self.s_q and h < self.h_i else None

    def tile_row(self, s: int, h: int):
        """The inverse: ``(m-tile, padded row)`` of query row ``s``, head ``h``."""
        if self.tile_rows == 1:
            return s // self.queries_per_tile, s % self.queries_per_tile
        t = s // self.queries_per_tile * self.tiles_per_query + h // 16
        g = (s % self.queries_per_tile) * self.group + (h % 16) // 2
        return t, g + 8 * ((h % 16) % 2)


def padded_heads(h_i: int) -> int:
    """``next_pow2(h_i)``: the MMA rows one query occupies."""
    p = 1
    while p < int(h_i):
        p *= 2
    return p


def query_row_map(s_q: int, h_i: int) -> QueryRowMap:
    p = padded_heads(h_i)
    tile_rows = min(p, MMA_M)
    qpt = MMA_M // tile_rows
    tpq = max(1, p // MMA_M)
    group = tile_rows // 2
    return QueryRowMap(int(h_i), int(s_q), p, tile_rows, qpt, tpq, _ceil_div(int(s_q) * p, MMA_M), group, max(0, group.bit_length() - 1))


def keys_per_warp(units_per_warp: int) -> int:
    """Compressed rows one warp streams: ``units_per_warp`` units of 32 (4 MMA n-tiles each)."""
    return ROWS_PER_UNIT * int(units_per_warp)


def blocks_per_cta(units_per_warp: int) -> int:
    """Compressed rows per CTA: 4 independent warps x ``keys_per_warp``."""
    return WARPS * keys_per_warp(units_per_warp)


# The knob's default (fit on the 2026-10-21 decode ranking, B 1..32 x 8K..128K x S_q 1 / 4): ONE unit per warp on every measured
# cell -- every key of the warp is in flight at once, the shortest per-warp chain, and the machine fills from 10 warps per SM up.  Two
# units were within +-7 % (better on one cell, B 4 x 128K x S_q 4, worse on B 32 x 32K x S_q 4) and four / eight never won a cell: the
# longer per-warp chains cost more than the per-warp table traffic they save.  The arguments stay (the step's key count, the device's
# SM count, the rows per sequence) for a refit per arch; re-measure before changing.
UNITS_DEFAULT = 1


def default_units_per_warp(n_blocks_max: int, batch: int = 1, num_sms: Optional[int] = None, s_q: int = 1) -> int:
    return UNITS_DEFAULT


def validate_params(*, h_i: int, s_q: int, page_size: int, units_per_warp: int, ring_depth: int = NTILES_IN_FLIGHT) -> None:
    """The compile-key facts, typed (never an assert): the head count and the rows per sequence (their PADDED product is the
    register envelope of the resident query fragments), the page size (a power of two of at least two compressed rows: the
    in-kernel page / row split is a shift), the units knob."""
    if int(h_i) < 1:
        raise ValueError(f"h_i must be >= 1, got {h_i}")
    if int(s_q) < 1:
        raise ValueError(f"s_q must be >= 1, got {s_q}")
    padded = int(s_q) * padded_heads(h_i)
    if padded > MAX_QUERY_ROWS:
        raise ValueError(
            f"s_q x h_i = {s_q} x {h_i} pads to {padded} (s, h) MMA rows (h_i padded to the next power of two, {padded_heads(h_i)}), past the "
            f"{MAX_QUERY_ROWS}-row register envelope ({MAX_QUERY_ROWS // MMA_M} m-tiles of {MMA_M}): score more rows per sequence in several calls"
        )
    p = int(page_size)
    if p < MIN_PAGE_SIZE or p % BLOCK_SIZE or (p & (p - 1)):
        raise ValueError(
            f"page_size must be a power of two >= {MIN_PAGE_SIZE} (a power-of-two count of {BLOCK_SIZE}-token compressed rows per page), got {page_size}"
        )
    if int(units_per_warp) not in UNITS_PER_WARP_CHOICES:
        raise ValueError(f"units_per_warp must be one of {UNITS_PER_WARP_CHOICES}, got {units_per_warp}")
    if int(ring_depth) not in RING_DEPTH_CHOICES:
        raise ValueError(f"ring_depth must be one of {RING_DEPTH_CHOICES}, got {ring_depth}")


def _check_i32_1d(t, name: str, n: int, device) -> None:
    if not isinstance(t, torch.Tensor) or t.dtype != torch.int32 or t.dim() != 1 or int(t.numel()) != int(n):
        raise ValueError(f"{name} must be an int32 [{n}] tensor, got {getattr(t, 'shape', type(t).__name__)} {getattr(t, 'dtype', '')}")
    if not t.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tuple(t.stride())}")
    if t.device != device:
        raise ValueError(f"{name} must live on {device}, got {t.device}")


def _kbar_rows_view(kbar: torch.Tensor, rows_per_page: int, device) -> torch.Tensor:
    """The compressed-key cache as the kernel's ``[num_pages, R, D]`` view (a view, never a copy): ``[num_pages, R, 1, D]``
    (the serving layout, one compressed row per 4 tokens) or ``[num_pages, R, D]``; bf16 (an e4m3 cache is refused by the
    entry, a mixed pair is a form error there); ``D`` contiguous, 16-byte-aligned page / row strides and base."""
    if not isinstance(kbar, torch.Tensor):
        raise ValueError(f"kbar must be a [num_pages, {rows_per_page}, 1, {INDEX_HEAD_DIM}] bf16 tensor, got {type(kbar).__name__}")
    if kbar.dim() == 4:
        if int(kbar.shape[2]) != 1:
            raise ValueError(
                f"kbar must be [num_pages, {rows_per_page}, 1, {INDEX_HEAD_DIM}] (one compressed row per {BLOCK_SIZE} tokens), got {tuple(kbar.shape)}"
            )
        view = kbar.select(2, 0)
    elif kbar.dim() == 3:
        view = kbar
    else:
        raise ValueError(
            f"kbar must be [num_pages, {rows_per_page}, 1, {INDEX_HEAD_DIM}] or [num_pages, {rows_per_page}, {INDEX_HEAD_DIM}], got {tuple(kbar.shape)}"
        )
    num_pages, rows, d = (int(x) for x in view.shape)
    if num_pages < 1 or rows != int(rows_per_page) or d != INDEX_HEAD_DIM:
        raise ValueError(
            f"kbar must hold >= 1 pages of {rows_per_page} compressed rows (page_size {rows_per_page * BLOCK_SIZE} / {BLOCK_SIZE}) x {INDEX_HEAD_DIM}, got {tuple(kbar.shape)}"
        )
    s0, s1, s2 = (int(x) for x in view.stride())
    if s2 != 1 or s1 != INDEX_HEAD_DIM or (s0 * 2) % 16 or kbar.data_ptr() % 16:
        raise ValueError(
            f"kbar must hold a page's compressed rows contiguously (row stride {INDEX_HEAD_DIM}, D contiguous) with a 16-byte-aligned page stride and "
            f"base, got strides {tuple(kbar.stride())}, data_ptr {kbar.data_ptr():#x}"
        )
    if s0 * 2 >= 2**31:
        raise ValueError(f"kbar's page stride must be below 2 GiB (the kernel's 32-bit page stride), got {s0 * 2} bytes")
    if kbar.device != device:
        raise ValueError(f"kbar must live on {device}, got {kbar.device}")
    return view


def qsa_select_decode_paged(
    qi: torch.Tensor,
    kbar: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    seq_q: int,
    top_k: int = QSA_TOP_K,
    n_blocks_max: Optional[int] = None,
    scale: Optional[float] = None,
    units_per_warp: Optional[int] = None,
    use_pdl: bool = False,
    logits_out: Optional[torch.Tensor] = None,
    counts_out: Optional[torch.Tensor] = None,
    block_lens_out: Optional[torch.Tensor] = None,
    stream=None,
) -> TupleDict:
    """Block selection for a DECODE step over the PAGED compressed-key cache: the paged scorer kernel, then the deterministic
    radix top-k with one visible count per row -- two launches, no torch prologue (the positions and the counts are derived on
    device).

    Contract:

    * ``qi``: ``[T, H_i, 128]`` bf16, ``T = B x seq_q`` (row ``t = b x seq_q + s`` is row ``s`` of sequence ``b``, at position
      ``kv_lens[b] - seq_q + s``); a unit last stride and 16-byte-aligned token / head strides (a column slice of a wider slab
      reads in place).  ``float8_e4m3fn`` (the fp8 indexer cache) is refused BY NAME: that read arm is a follow-up of the
      kernel.
    * ``kbar``: ``[num_pages, page_size / 4, 1, 128]`` bf16, the paged compressed-key cache on the MAIN block table: block
      ``j`` of sequence ``b`` is row ``j % (page_size / 4)`` of page ``block_table[b, j // (page_size / 4)]``; ``page_size`` a
      power of two >= 8 (16 / 32 / 64 in serving).  A ``-1`` (or out-of-range) page under a visible block reads as a zero key
      (logit 0); a page past the visible range is never consulted.
    * ``block_table``: ``[B, max_pages]`` int32 with a unit column stride; ``kv_lens``: ``[B]`` int32 CUDA, the logical KV
      length INCLUDING this step's ``seq_q`` rows (``0`` = a padding sequence: nothing is read or written for it but its
      counts, which read 0).
    * ``n_blocks_max``: the logits width = the cache capacity in blocks; ``None`` = ``max_pages x page_size / 4`` (the table's
      reach).  A multiple of 8 (the radix top-k reads fp32 rows in 256-bit vectors; allocate the table with an even number
      of pages at page 16, or pass the capacity).
    * outputs, every row ``t``: ``block_ids [T, top_k]`` int32 -- the selected LOCAL block ids in ASCENDING order, ``-1``
      past ``block_lens[t] = min(top_k, counts[t])``; ``scores [T, top_k]`` fp32 (``-inf`` on the padding); ``logits
      [T, n_blocks_max]`` fp32 with columns ``[0, counts[t])`` written (the rest untouched: never read by the top-k);
      ``counts [T]`` int32 = the row's visible complete-block count ``clamp(floor((pos + 1) / 4), 0, n_blocks_max)``;
      ``block_lens [T]`` int32.  A tie at the k-th boundary keeps the smaller block id and the order is canonical, so two
      calls on the same inputs are BITWISE.  ``logits_out`` / ``counts_out`` / ``block_lens_out`` are optional preallocated
      scorer outputs (``logits_out`` 32-byte aligned, contiguous); the top-k's outputs and its radix scratch are allocated here
      per call on the launch stream (the standalone form; the block's stage carves them from its workspace).
    * ``top_k`` in ``[1, 2048]``; ``scale`` defaults to ``1 / sqrt(128)`` and must be positive; ``units_per_warp`` (1 / 2 / 4 / 8
      units of 32 compressed rows per warp, ``None`` = :func:`default_units_per_warp` from the step's total key count, the
      device's SM count and ``seq_q``) and ``use_pdl`` are performance knobs: the same function.
    * ``stream``: the LAUNCH stream -- a ``torch.cuda.Stream`` (on ``qi``'s device), a raw CUDA stream handle (an ``int`` or a
      ``cuda.CUstream``), or ``None`` = torch's current stream on ``qi``'s device.  It is resolved to one torch stream before any
      torch operation of the call, and every one of them (the output allocations here, the top-k wrapper's allocations and
      launch) runs on it (Rule 5); the two kernels receive its raw handle.

    Raises:
        NotImplementedError: a CuTe DSL below the floor / without the target, a device below cc 9.0 (the radix top-k's bound;
            the scorer alone runs anywhere -- ``kernels.qsa_score_paged.run_qsa_score_paged``), ``use_pdl`` below cc 9.0, or
            the e4m3 pair -- the message names the version, the device or the arm; raised BEFORE the kernel module is imported.
        ValueError: a malformed input (shape, dtype, device, alignment, the table's reach, ``top_k``, ``scale``, a stream on
            another device).
    """
    if not isinstance(qi, torch.Tensor) or qi.dim() != 3:
        raise ValueError(f"qi must be a [T, H_i, {INDEX_HEAD_DIM}] tensor, got {getattr(qi, 'shape', type(qi).__name__)}")
    if not isinstance(kbar, torch.Tensor) or kbar.dim() not in (3, 4):
        raise ValueError(f"kbar must be a [num_pages, page_size / {BLOCK_SIZE}, 1, {INDEX_HEAD_DIM}] tensor, got {getattr(kbar, 'shape', type(kbar).__name__)}")
    if qi.dtype == _E4M3 and kbar.dtype == _E4M3:
        raise NotImplementedError(
            "qsa_select_decode_paged scores the bf16 indexer cache; the e4m3 pair (qi and kbar float8_e4m3fn, the fp8 indexer cache) is the "
            "paged scorer's e4m3 read arm, a follow-up -- pass the bf16 pair, or select over a dense e4m3 cache with the dense decode selector"
        )
    if qi.dtype != torch.bfloat16 or kbar.dtype != torch.bfloat16:
        raise ValueError(f"qi and kbar must both be bfloat16 (the e4m3 pair is a typed decline of its own); got {qi.dtype} and {kbar.dtype}")
    if not qi.is_cuda or kbar.device != qi.device:
        raise ValueError("qi and kbar must be CUDA tensors on one device")
    if int(seq_q) < 1:
        raise ValueError(f"seq_q must be >= 1, got {seq_q}")
    if not isinstance(block_table, torch.Tensor) or block_table.dim() != 2 or block_table.dtype != torch.int32:
        raise ValueError(
            f"block_table must be an int32 [B, max_pages] tensor, got {getattr(block_table, 'shape', type(block_table).__name__)} {getattr(block_table, 'dtype', '')}"
        )
    t, h_i, d = (int(x) for x in qi.shape)
    batch, max_pages = (int(x) for x in block_table.shape)
    if d != INDEX_HEAD_DIM or (kbar.dim() == 4 and int(kbar.shape[3]) != INDEX_HEAD_DIM) or (kbar.dim() == 3 and int(kbar.shape[2]) != INDEX_HEAD_DIM):
        raise ValueError(f"qsa_select_decode_paged scores head_dim {INDEX_HEAD_DIM} indexer heads, got qi {tuple(qi.shape)} and kbar {tuple(kbar.shape)}")
    rows_per_page = int(kbar.shape[1])
    page_size = rows_per_page * BLOCK_SIZE
    if t != batch * int(seq_q):
        raise ValueError(
            f"qi holds T = {t} rows but block_table holds B = {batch} sequences of seq_q = {seq_q} rows (T must be B x seq_q = {batch * int(seq_q)})"
        )
    if n_blocks_max is None:
        n_blocks_max = max_pages * rows_per_page
    n_blocks_max = int(n_blocks_max)
    if n_blocks_max < 1 or n_blocks_max % _SCORE_ROW_ALIGN:
        raise ValueError(
            f"n_blocks_max must be a positive multiple of {_SCORE_ROW_ALIGN} (the radix top-k reads fp32 rows in 256-bit vectors), got {n_blocks_max}: "
            f"allocate the table with a multiple-of-{_SCORE_ROW_ALIGN} compressed-row reach (an even page count at page 16) or pass the capacity"
        )
    if not 1 <= int(top_k) <= TOP_K_KERNEL_MAX:
        raise ValueError(f"top_k must be in [1, {TOP_K_KERNEL_MAX}] (the radix top-k kernel's bound), got {top_k}")
    if scale is None:
        scale = INDEX_HEAD_DIM**-0.5
    if not float(scale) > 0.0:
        raise ValueError(f"scale must be positive (the ordering is scale-free), got {scale}")
    if units_per_warp is None:
        units_per_warp = default_units_per_warp(n_blocks_max, batch, torch.cuda.get_device_properties(qi.device).multi_processor_count, int(seq_q))
    validate_params(h_i=h_i, s_q=int(seq_q), page_size=page_size, units_per_warp=int(units_per_warp))
    _kbar_rows_view(kbar, rows_per_page, qi.device)  # the cache's form (its singleton row axis, D contiguous, the alignment) ahead of the gates
    for name, out, shape in (("logits_out", logits_out, (t, n_blocks_max)), ("counts_out", counts_out, (t,)), ("block_lens_out", block_lens_out, (t,))):
        if out is None:
            continue
        want = torch.float32 if name == "logits_out" else torch.int32
        if (
            not isinstance(out, torch.Tensor)
            or out.dtype != want
            or tuple(int(x) for x in out.shape) != shape
            or not out.is_contiguous()
            or out.device != qi.device
        ):
            raise ValueError(
                f"{name} must be a contiguous {want} {shape} tensor on {qi.device}, got {getattr(out, 'shape', type(out).__name__)} {getattr(out, 'dtype', '')}"
            )
        if name == "logits_out" and out.data_ptr() % _TOPK_INPUT_ALIGN:
            raise ValueError(f"logits_out must be {_TOPK_INPUT_ALIGN}-byte aligned (the radix top-k's input), got data_ptr {out.data_ptr():#x}")

    # Rule 7 first: the DSL floor (device-free), then the target, then the device family of the radix top-k -- BEFORE the kernel
    # module (whose import executes the DSL) or any kernel module of the DSA package is imported, so a sub-floor DSL gets the typed
    # message and never a DSL-internal import error.
    why = cutedsl_requirement_error("qsa_select_decode_paged")
    if why is not None:
        raise NotImplementedError(why)
    device_cc = tuple(torch.cuda.get_device_capability(qi.device))
    why = cutedsl_arch_requirement_error(device_cc)
    if why is not None:
        raise NotImplementedError(why)
    if device_cc < (9, 0):
        raise NotImplementedError(
            f"qsa_select_decode_paged pairs the paged scorer with the radix top-k kernel, which requires cc 9.0 or later; found cc {device_cc[0]}.{device_cc[1]} "
            "(the scorer alone, kernels.qsa_score_paged.run_qsa_score_paged, runs on this device)"
        )

    # only now the two kernel modules: the paged scorer (this package) and the radix top-k (the DSA package)
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_top_k import indexer_top_k_wrapper

    from .kernels.qsa_score_paged import compile_qsa_score_paged, run_qsa_score_paged

    dev = qi.device.index if qi.device.index is not None else torch.cuda.current_device()
    recipe = compile_qsa_score_paged(
        h_i=h_i, s_q=int(seq_q), page_size=page_size, units_per_warp=int(units_per_warp), has_block_lens=True, use_pdl=bool(use_pdl), device=dev
    )

    # Rule 5: ONE torch stream resolved from `stream` FIRST (a torch.cuda.Stream / a raw handle / a cuda.CUstream; None = torch's current
    # stream on qi's device); every torch operation below -- the scorer's output allocations, the top-k wrapper's own allocations and
    # launch -- runs inside its context, and both kernels receive its raw handle.
    launch = launch_stream(stream, qi.device)
    raw_stream = int(launch.cuda_stream)
    with stream_context(launch, qi.device):
        logits = torch.empty((t, n_blocks_max), dtype=torch.float32, device=qi.device) if logits_out is None else logits_out
        counts = torch.empty((t,), dtype=torch.int32, device=qi.device) if counts_out is None else counts_out
        block_lens = torch.empty((t,), dtype=torch.int32, device=qi.device) if block_lens_out is None else block_lens_out
        run_qsa_score_paged(recipe, qi, kbar, block_table, kv_lens, logits, counts, block_lens, top_k=int(top_k), scale=float(scale), stream=raw_stream)
        selected = indexer_top_k_wrapper(logits, counts, int(top_k), next_n=1, return_val=True, stream=raw_stream, tie_break=0, deterministic=True)
    return TupleDict(block_ids=selected["indices"], block_lens=block_lens, scores=selected["values"], logits=logits, counts=counts)


def moved_bytes(n_vis_rows, *, s_q: int, h_i: int, n_blocks_max: int, page_size: int, units_per_warp: int = 1) -> int:
    """The HBM-side byte model of one scorer launch for a SOL number: ``n_vis_rows`` = every row's visible count (host arithmetic
    from the caller's own ``kv_lens``, ``[T]`` in row order, ``T = B x s_q``): the compressed rows a sequence streams once
    (its last row's count x 256 B), the fp32 logits written (every row's count x 4 B), the padded query tile every LIVE CTA stages
    (``16 x 256 B`` per m-tile, L2-resident in practice, counted) and the table words (one per page, read by the warp that owns it)."""
    rows = [int(v) for v in n_vis_rows]
    s_q = int(s_q)
    if s_q < 1 or len(rows) % s_q:
        raise ValueError(f"n_vis_rows holds {len(rows)} rows, not a multiple of s_q = {s_q}")
    rows_per_page = int(page_size) // BLOCK_SIZE
    per_cta = blocks_per_cta(units_per_warp)
    n_ctas = _ceil_div(int(n_blocks_max), per_cta)
    tile_bytes = query_row_map(s_q, h_i).m_tiles * MMA_M * INDEX_HEAD_DIM * 2
    total = 0
    for b in range(len(rows) // s_q):
        seq = rows[b * s_q : (b + 1) * s_q]
        n_max = max(seq)
        total += n_max * INDEX_HEAD_DIM * 2  # the compressed rows read once
        total += sum(seq) * 4  # the logits written
        total += min(n_ctas, _ceil_div(n_max, per_cta)) * tile_bytes  # the query tile per LIVE CTA (none for a padding sequence)
        total += _ceil_div(n_max, rows_per_page) * 4  # the table words, one per page
    total += 2 * len(rows) * 4  # counts + block_lens
    return total
