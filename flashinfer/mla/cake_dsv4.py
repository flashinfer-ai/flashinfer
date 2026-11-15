"""Source-level CAKE backend for DeepSeek V4 sparse MLA on SM100 and SM103, and
for the FP8 decode route on SM107.

Host contract (flashinfer#4671 hardening)
-----------------------------------------

* **Metadata ABI.** Every generated variant consumes its per-token metadata
  through the same eight kernel parameters (:data:`KERNEL_METADATA_PARAMS`):
  ``swa_indices``, ``compressed_indices``, ``sparse_topk_lens``,
  ``swa_index_stride``, ``compressed_index_stride``, ``sparse_topk_lens_offset``,
  ``sparse_topk`` and ``num_query_tokens``. Combined-space column ``c`` of row
  ``t`` lives at ``swa_indices[t * swa_index_stride + c]`` when ``c < 128`` and
  at ``compressed_indices[t * compressed_index_stride + (c - 128)]`` otherwise.
  The active length of a row is
  ``clamp(sparse_topk_lens[t] + sparse_topk_lens_offset, 0, sparse_topk)``.
  The host resolves the parameters from either the combined FlashInfer table
  ``sparse_indices [T, sparse_topk]`` (the compressed view is a column offset of
  the same storage) or from separate tables (``sparse_indices [T, 128]`` plus
  ``extra_sparse_indices [T, topk_c]``; ``extra_sparse_topk_lens`` counts
  compressed slots only and implies ``sparse_topk_lens_offset += 128``).
  Resolution never copies: a non-unit column stride or a non-int32 table is an
  error. The kernels load index rows 16 bytes (four int32) at a time, so every
  bound table must start at a 16-byte aligned address and, with more than one
  row, have a row stride that is a multiple of 4 elements. Other tables (for
  example ``parent[:, 1:]`` of a ``[T, W + 1]`` parent) are an error as well.
* **Padded rows.** ``num_query_tokens`` is the metadata row count ``T``.
  ``query`` and ``out`` may carry more rows; rows ``>= T`` are neither read nor
  written. Grids and workspace views derive from ``T``, never from query rows.
* **Argument binding.** Kernel arguments are bound *by name* through the
  registration ``arg_plan`` (:func:`_prepare_variant`), so regenerated bindings
  only need names from the host vocabulary (:func:`is_bindable_arg`). Every
  route launches its variant kernels directly; two-stage routes bind the
  producer and the reducer first and issue both through one FFI call
  (``run_sequence`` in ``cake_dsv4_launch_sequence.cc``), so no host work
  sits between the two kernels.
* **Query layout.** Every variant receives the five query-layout parameters
  (:data:`QUERY_LAYOUT_PARAMS`): ``seq_lens`` (int32, the cache length of
  every request), ``cum_seq_lens_q``, ``ragged_query``, ``max_q_len`` and
  ``batch_size``. The kernels derive the trtllm-gen SWA validity window from
  them: with ``b`` the request owning metadata row ``t``, ``q_len_b`` its
  query length and ``q_off`` the row's position in the request, combined
  column ``c < 128`` is attended only when
  ``c < clamp(seq_lens[b] - (q_len_b - 1 - q_off), 0, 128)`` (CAKE-957). A
  dense call (``cum_seq_lens_q=None``) binds ``ragged_query = 0``,
  ``max_q_len`` = the caller's per-request query length, ``batch_size =
  seq_lens.numel()`` and ``seq_lens`` in place of the never-read
  ``cum_seq_lens_q`` pointer; a ragged call binds the caller's offsets with
  ``ragged_query = 1``. The producers in :data:`_RAGGED_ONLY_ROUTES` read the
  request boundaries from ``cum_seq_lens_q`` unconditionally and receive the
  cached dense offsets instead. No host value is ever ``None``.
* **Preconditions on the metadata.** Every metadata row must keep at least
  one column inside the three validity predicates' domain: the active length
  ``sparse_topk_lens[t] + sparse_topk_lens_offset`` must be ``>= 1`` and the
  owning request must satisfy ``seq_lens[b] >= q_len_b`` (so the SWA window
  holds at least the row's own token). sglang satisfies both by construction
  (constant 128 on SWA-only layers, the window always contains the current
  token). A row whose columns are all ``-1`` or all beyond the window is a
  defined input and returns the all-invalid zero result. An active length of
  0 is outside the trtllm-gen contract (the stock FP8 kernels read past the
  active length or return NaN there); the Cake programs regenerated for
  CAKE-957 all return the zero result for it (tile-count guards in the FP8
  persistent, BF16 H64 guard/prefill and FP8 H64 source-exact bodies). The
  host does not synchronise to check these values.
* **Workspace.** One caller-owned ``workspace_buffer`` is carved
  deterministically (:func:`cake_dsv4_workspace_layout`)::

      [0,      1024)          reserved (formerly the TMA descriptor slab)
      [1024,   1024 + 256 KiB) split-merge counters, uint32[65536]
      [P,      P + O_bytes)    partial_O  bf16 [T * H * S * 512]  (P = 1024 + 256 KiB;
                               O_bytes = 0 when the route runs one split)
      [P + O_bytes, + L_bytes) partial_lse f32 [T * H * S]
      [..., + Q_bytes)         shifted request offsets int32 [B + 1]
                               (row-tiled launches after the first chunk only)

  with every region 128-byte aligned. The requirement is exact per route
  (:func:`cake_dsv4_workspace_requirement`, from the same
  :func:`_route_plan` table the dispatcher launches with): routes without
  partial buffers need no workspace bytes, one-split routes need the LSE
  region only, split routes the partial O and LSE regions. The one-split
  FP8/H128 uniform-gather route at the shapes of the SM107 split rules is
  the exception: its ``single_launch_bytes`` is the split launch's carve
  ``cake_dsv4_workspace_layout(T, H, s)`` ("Split-KV pieces" below), so
  size those calls with ``single_launch_bytes`` or with
  :func:`get_cake_dsv4_workspace_bytes` without ``num_splits``, not with the
  requirement's ``num_splits`` (1); a smaller workspace runs the route's
  own unsplit launch. When
  ``workspace_buffer`` cannot hold one launch over all ``T`` metadata rows,
  the host tiles the rows into consecutive launches that fit
  (:func:`_dispatch_row_tiles`): ``Q``, ``O`` and the metadata tables are
  sliced by row (views, no copies) and every chunk after the first receives
  its request boundaries as ``cum_seq_lens_q - first_row`` (``B + 1`` int32
  written into the workspace's offsets region by one ``torch.sub`` in stream
  order, capturable) with ``ragged_query = 1``, so the window resolves for
  any chunk start. The smallest accepted workspace is the one-row
  requirement; :func:`get_cake_dsv4_workspace_bytes` is the single-launch
  upper bound. sglang's fixed 128 MiB buffer therefore admits every row
  count on every route; the only route whose single launch can outgrow it is
  ``bf16_h32_topk128x_early_v47`` (``ceil(sparse_topk / 128)`` splits of
  ``32 * 1028`` bytes per row, no token bound), which tiles.
* **Split-KV pieces (SM107).** A split launch of the FP8/H128 uniform-gather
  route (:func:`_k64_split_plan`) runs its ``W`` whole rows first and each
  later row ``t`` as ``s`` pieces of its sparse tiles, then merges the
  pieces into ``O`` (one ``run_sequence`` call) with the merge their rule's
  ``merge_by_splits`` names for ``s`` and their rows
  (:func:`_k64_split_merge`): on SM107, for 128 heads, a merge built for
  ``s`` pieces alone, one CTA per row and head (nine and five pieces) or one
  warp per row and head and two heads per CTA (three pieces at 16-20 rows),
  else the merge of any ``s``, one CTA per row and head. It carves
  ``cake_dsv4_workspace_layout(R, H, s)`` for its ``R = T - W`` split rows
  and stores the pieces in token-major order: piece ``j`` of row ``t`` is
  ``p = (t - W) * s + j``, its BF16 partial O the rows ``[p, :, :]`` of
  ``partial_O [R * s, H, 512]`` and its base-2 LSE ``partial_lse[p * H +
  h]``. Piece 0 of a row whose tiles fit one piece holds ``+inf``: it wrote
  the final ``O`` row itself and the merge leaves the row alone. A piece
  without a valid column, or one that ran no tile, holds ``-inf`` and may
  leave its partial O unwritten: the merge never reads it, and writes 0 for
  a row whose every piece holds ``-inf``. Row-tiled launches never split.
* **Split-merge counters.** ``bf16_h32_topk128x_early_v47`` (the only route
  with an in-kernel last-arriver merge) needs its ``T * ceil(H / 8)``
  counters zero at launch and leaves them zero afterwards. Zeroing is part
  of the launch contract and never raises: the first eager launch through a
  workspace zeroes the whole counter region once and registers the
  workspace in :data:`_primed_workspaces` (keyed by device and base address,
  guarded by a weak reference to the tensor owning the storage, so views of
  one buffer share the state and a freed and reused address is re-primed);
  a launch under CUDA Graph capture through an unregistered workspace
  records a zero fill of exactly the counters it uses into the graph
  (replays stay self-contained, nothing is registered). The kernels do not
  depend on any other host state. :func:`cake_dsv4_workspace_reset` zeroes
  and registers explicitly.

NVFP4 cache route (``kv_cache_format="nvfp4"``)
---------------------------------------------------------

:func:`run_cake_dsv4_nvfp4` hosts the DeepSeek-V4 NVFP4 sparse-MLA decode
family (Cake ``flashinfer_blackwell_sparse_mla_dsv4_nvfp4_decode`` and
``..._decode_tile`` plus their shared split merge). The pools are the SM120
NVFP4 cache ABI: opaque ``uint8`` pages of ``page_size * 352`` data bytes
(224 B E2M1 NoPE + 128 B BF16 RoPE per token) followed by ``page_size * 32``
scale bytes, i.e. 384 bytes per token; indices are flat token coordinates
``page * page_size + slot``, ``-1`` masks a candidate, and the optional length
vectors clamp the active prefix of each table. The two tables are
independent (``sparse_indices`` over ``swa_kv_cache``, ``extra_sparse_indices``
over ``compressed_kv_cache``); there is no combined-table column offset.

The launch plan (:func:`_nvfp4_plan`) reproduces the Cake family ``plan()``:
128-wide candidate tiles; rows with ``H <= 32`` run the persistent SwapsAB
member (``pv``) when its wave-aware split rule gives more than one tile per
CTA and the one-tile SwapsAB member (``swap``) otherwise; rows with ``H >= 64``
pick the member and split count with the lowest modelled chain cost among the
one-tile ``tile`` member, the persistent member, the 2-CTA ``cluster`` member
and the tile64 member (``t64``, 64 heads per CTA); a one-tile row with
``H <= 64`` whose grid gives 2 or 4 CTAs per work item runs the H64 tile member
(``tile_h64``: 64-row Q boxes, same grid and output as ``tile``). The one-tile
members run ``o_chunks`` CTAs per work item; the split merge runs on
``merge_heads_per_cta`` heads per CTA. Each member is one registered variant
(``_nvfp4_variant_name``: ``nvfp4_decode_persistent``, ``nvfp4_decode_cluster``,
``nvfp4_decode_t64_n64_oc1``, ``nvfp4_decode_pv``, ``nvfp4_decode_swap``,
``nvfp4_decode_tile``, ``nvfp4_decode_tile_h64``) plus ``nvfp4_merge``; the
templated members take their retrace knobs (``tile_n``: heads on the MMA N side,
``o_chunks``: CTAs per work item) as launch arguments and their launcher switches
over the explicit instantiations of the kernel unit. All are bound through the same
registration machinery as the BF16/FP8 routes from the generated sources under
``csrc/cake_dsv4/{sm_100a,sm_103a,common}``; a plan that selects
an unexported variant raises ``NotImplementedError``. Splits write
``partial_O [T, H, S, 512]`` BF16 and ``partial_lse [T, H, S]`` FP32 into the
workspace; the final base-2 LSE lives in an extra ``lse`` region appended
after ``partial_lse`` (:func:`cake_dsv4_workspace_layout` ``with_lse=True``).

A query packed to the NVFP4 row format (``uint8 [rows, H, 384]``: per row the
352 bytes of E2M1 NoPE codes and BF16 RoPE dims, then its 28 E4M3 scale factors
and 4 pad bytes, the cache's token record; :func:`cake_dsv4_nvfp4_pack_query`
packs a BF16 query) runs the NVFP4 prefill bodies instead, at any row count
(Cake ``cake_dsv4_nvfp4_query_block_prefill`` and
``flashinfer_blackwell_trtllm_sparse_mla_dsv4_nvfp4_{h64,h128}_prefill``,
chosen by :func:`_nvfp4_prefill_route`): the query-block bodies
``nvfp4_qb_h{16,32}*`` (16 / 32 heads on window-only rows with more query
tokens than SMs: one item per block of ``64 / H`` consecutive tokens, whose
shifted windows are gathered and transformed once), the per-token builds
``nvfp4_h64_*`` (``H <= 64``, one item per query token, the heads duplicated on
the two 64-row halves of a block-scaled M128 QK) and ``nvfp4_h128_prefill``
(``H128``, one two-CTA cluster per token, 64 heads per CTA). As the FP8
route takes an E4M3 query, they take the query packed by the caller: the call
packs nothing. A BF16 query runs the decode family at any row count, as it
did before the prefill bodies. On the DSv4 prefill shapes (128 to 8192 rows)
the prefill bodies with a packed query run 1.04-6.5x faster than the decode
family on B200 and GB300; below 64 rows the decode family is often the
faster one. The decode family caps a row at 12 candidate tiles, so wider
tables (C128 layers past 1408 compressed entries) run only with a packed
query. The prefill bodies read the
pools through 32-byte-pitch data-row and scale-row tensor maps, and write the
final BF16 output and base-2 LSE directly (no split, no merge), so their
workspace carve has one split (no ``partial_O`` region); the LSE lands in the
same ``lse`` region as the decode family's.
"""

from __future__ import annotations

import ctypes
import functools
import os
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Union

import torch

from ..utils import get_compute_capability


_HEAD_DIM = 512
_TILE_KV = 128
_SWA_WIDTH = 128
_ALIGN = 128
# The BF16 / FP8 programs load index rows this many bytes (four int32) at a time.
_INDEX_ROW_ALIGN = 16

# Deterministic workspace layout (byte offsets inside workspace_buffer).
_DESCRIPTOR_SLAB_OFFSET = 0
_DESCRIPTOR_SLAB_BYTES = 1024
_COUNTER_OFFSET = _DESCRIPTOR_SLAB_OFFSET + _DESCRIPTOR_SLAB_BYTES
_COUNTER_REGION_BYTES = 256 * 1024
_MAX_MERGE_GROUPS = _COUNTER_REGION_BYTES // 4
_PARTIAL_OFFSET = _COUNTER_OFFSET + _COUNTER_REGION_BYTES
# Largest fixed split count used by any route (bf16_h128_topk4x_v52 / fp8_h128).
_MAX_FIXED_SPLITS = 5
# BF16/H128 SWA-only and topk4x rows with this many metadata tokens or more use
# the persistent KV-reuse prefill body (mirrors the Cake dispatcher's
# BF16_H128_PREFILL_MIN_TOKENS).
_BF16_H128_PREFILL_MIN_TOKENS = 64
# The two-stage split4 program (4 owners x 2 CTAs per token) runs one wave
# only up to this many query tokens; wider grids lose to trtllm-gen.
# Mirrors the Cake seed's BF16_TOPK128X_SPLIT_MAX_TOKENS.
_BF16_TOPK128X_SPLIT_MAX_TOKENS = 16
# Mirrors the Cake seed's BF16_ROW_FIRST_V_HALF_SPLIT_MAX_TOKENS: row-first rows
# with at most this many tokens run the V-half split program (two 2-CTA
# clusters per token = 4 CTAs per token, one wave on 148+ SMs).
_BF16_ROW_FIRST_V_HALF_SPLIT_MAX_TOKENS = 37
# Widths the BF16/H128 four-owner split and row-first producers cover (two or
# three live KV tiles); other widths below the prefill token bound have no
# exported kernel.
_BF16_TOPK128X_MIN_WIDTH = 256
_BF16_TOPK128X_MAX_WIDTH = 388
# One FP8 low-head producer partition owns up to three sparse tiles; mirrors
# the producer's FP8_ONE_PARTITION_MAX_TILES = 3.
_FP8_ONE_PARTITION_MAX_WIDTH = 384
# Every FP8 body takes items of at most 1152 sparse slots (SWA 128 + 1024):
# the persistent and M64 bindings check ``sparse_topk <= 1152``, and the
# low-head producers stage an item's sparse indices in a shared-memory buffer
# of 1152 slots (nine tiles of 128) without a check of their own.  A C128
# layer's item holds 128 + context / 128 slots, so it passes the limit past
# 128K context (2176 at 256K).
_FP8_MAX_WIDTH = 1152
_BF16_H64_COMPRESSED_PREFILL_TOKENS = 24
_BF16_H64_PREFILL_MAX_SPARSE_WIDTH = 640
# Workspaces whose split-merge counter region is known to be zero:
# (device index, address of the workspace view) -> weak reference to the
# tensor owning the storage (the view's ``_base`` or the tensor itself). A dead
# reference or another owner at the same address means the memory was freed
# and possibly reused: the workspace is primed again.
_primed_workspaces: dict[tuple[Optional[int], int], "weakref.ref[torch.Tensor]"] = {}
_primed_lock = threading.Lock()

# NVFP4 cache route. Mirrors the Cake kernel module constants
# (flashinfer_blackwell_sparse_mla_dsv4_nvfp4_decode: TOKEN_DATA_BYTES +
# TOKEN_SF_BYTES, TILE_Q, MAX_SPLITS, SUPPORTED_HEAD_COUNTS).
_NVFP4_TOKEN_BYTES = 384
_NVFP4_TILE_Q = 128
_NVFP4_MAX_SPLITS = 12
_NVFP4_HEAD_COUNTS = (8, 16, 32, 64, 128)
_NVFP4_HEAD_DIM = 512
_NVFP4_VARIANT_PERSISTENT = "nvfp4_decode_persistent"
_NVFP4_VARIANT_CLUSTER = "nvfp4_decode_cluster"
_NVFP4_VARIANT_MERGE = "nvfp4_merge"
_NVFP4_CLUSTER_CTAS = (
    2  # CTAs per cluster of the cluster member (each owns 256 of the 512 output dims)
)
# Family planner (Cake flashinfer_blackwell_sparse_mla_dsv4_nvfp4_decode.plan, round 43 / 45).  Multi-tile rows
# with H <= _NVFP4_PV_MAX_HEADS take the persistent SwapsAB member ("pv") when its wave-aware split rule gives more
# than one tile per CTA; H > _NVFP4_PV_MAX_HEADS rows pick (member, splits) with the lowest modelled chain cost among
# the one-tile "tile" member, the persistent member, the 2-CTA cluster member (H >= _NVFP4_CLUSTER_MIN_HEADS) and
# the tile64 member "t64" (_NVFP4_T64_HEAD_COUNTS); one-tile rows take the SwapsAB one-tile member "swap" (H <=
# _NVFP4_SWAP_MAX_HEADS) or the "tile" member, each with its own CTAs-per-work-item split (o_chunks); a "tile" row
# with H <= _NVFP4_TILE_H64_MAX_HEADS whose split gives 2 or 4 CTAs per work item runs the H64 tile member "tile_h64"
# (64-row Q boxes; same grid, bitwise-equal output), while its 1-CTA form stays on "tile".
_NVFP4_PV_MAX_HEADS = 32
_NVFP4_SWAP_MAX_HEADS = 32
_NVFP4_TILE_H64_MAX_HEADS = 64
_NVFP4_CLUSTER_MIN_HEADS = 64
_NVFP4_T64_HEAD_COUNTS = (64, 128)
_NVFP4_T64_TILE_Q = 64  # heads per CTA of the tile64 member (H128 runs two head tiles)
# Round-43 chain model (Cake PLAN_CHAIN_US / PLAN_MERGE_US / PLAN_MERGE_US_PER_MB): member -> (HEAD, TILE, LOAD) us;
# a row costs waves * (HEAD + (tiles_per_split - 1) * TILE + LOAD * min(grid, slots) / slots) plus, with more than
# one split, MERGE + MERGE_PER_MB * (T * H * (splits + 1) * 512 * 2 bytes / 1e6).
_NVFP4_PLAN_CHAIN_US: Mapping[str, tuple[float, float, float]] = {
    "persistent": (11.4, 4.5, 3.1),
    "cluster": (10.6, 3.9, 0.2),
    "t64": (8.2, 3.5, 3.4),
    "tile": (8.8, 0.0, 1.7),
}
_NVFP4_PLAN_MERGE_US = 3.9
_NVFP4_PLAN_MERGE_US_PER_MB = 0.20
# Persistent SwapsAB member split rule (Cake flashinfer_blackwell_sparse_mla_dsv4_nvfp4_decode_swap_pv.plan):
# waves * (HEAD + (tiles_per_split - 1) * TILE) + MERGE over every split count, first minimum wins.
_NVFP4_PV_PLAN_HEAD_US = 8.6
_NVFP4_PV_PLAN_TILE_US = 3.0
_NVFP4_PV_PLAN_MERGE_US = 1.5
# Split merge (Cake MERGE_HEADS_PER_CTA / merge_heads_per_cta): 16 warps per CTA, one (token, head) per warp;
# heads_per_cta is the smallest power of two whose grid T x ceil(H / hpc) stays within one CTA per SM (round 45).
_NVFP4_MERGE_HEADS_PER_CTA = 16
# NVFP4 prefill bodies: a caller-packed NVFP4 query runs them (see the module docstring);
# a BF16 query runs the decode family at any row count.
# The per-token body takes up to 64 heads, the two-CTA body 128. Its builds (``_nvfp4_prefill_route``):
# ``tmem_v`` keeps V^T in TMEM (at most 32 heads); ``early_gather`` gathers a CTA's first item before
# the index hand-off (one item per CTA: at most as many query tokens as SMs); ``o_boxes`` stores O
# through TMA boxes; ``q_prefetch`` is ``o_boxes`` with the next item's query loaded ahead (window-only
# rows).
_NVFP4_VARIANT_H64_TMEM_V_PREFILL = "nvfp4_h64_tmem_v_prefill"
_NVFP4_VARIANT_H64_TMEM_V_EARLY_GATHER_PREFILL = "nvfp4_h64_tmem_v_early_gather_prefill"
_NVFP4_VARIANT_H64_EARLY_GATHER_PREFILL = "nvfp4_h64_early_gather_prefill"
_NVFP4_VARIANT_H64_O_BOXES_PREFILL = "nvfp4_h64_o_boxes_prefill"
_NVFP4_VARIANT_H64_Q_PREFETCH_PREFILL = "nvfp4_h64_q_prefetch_prefill"
_NVFP4_VARIANT_H128_PREFILL = "nvfp4_h128_prefill"
# The two-CTA body's build for items of more than two 128-key tiles (sparse_topk above 256, e.g. C4 and
# C128 past 16K context): its loaders transform two NoPE units per trip from one shared scale word.
# It is built for sm_103a; sm_100a keeps the persistent build.
_NVFP4_VARIANT_H128_V_UNIT_PAIRS_PREFILL = "nvfp4_h128_v_unit_pairs_prefill"
_NVFP4_H128_V_UNIT_PAIRS_MIN_TOPK = 257
_NVFP4_H128_V_UNIT_PAIRS_ARCHES = ("sm_103a",)
# Query-block bodies (16 / 32 heads, window-only rows: no extra table, a main table of at most 128
# keys on a row stride of at least 128): one CTA item per block of 64 / H consecutive query tokens,
# whose shifted windows are gathered and transformed once (window tables oldest or newest key first,
# as SGLang builds them). ``breaks`` (16 heads) splits a block at a sequence boundary through extra
# group items, up to the token count below. ``late_claim`` (32 heads) holds back the next claim of a
# CTA whose first block cannot share; it pays only while most CTAs take a second block, so it serves
# the rows of at most the token count below whose blocks exceed 11 / 8 of the SMs (on GB300's 152
# SMs: 420 to 512 tokens; the default build is 1-3 % faster on either side).
_NVFP4_VARIANT_QB_H16_PREFILL = "nvfp4_qb_h16_prefill"
_NVFP4_VARIANT_QB_H16_BREAKS_PREFILL = "nvfp4_qb_h16_breaks_prefill"
_NVFP4_VARIANT_QB_H32_PREFILL = "nvfp4_qb_h32_prefill"
_NVFP4_VARIANT_QB_H32_LATE_CLAIM_PREFILL = "nvfp4_qb_h32_late_claim_prefill"
_NVFP4_QB_BREAKS_MAX_TOKENS = 1024
_NVFP4_QB_LATE_CLAIM_MAX_TOKENS = 512
_NVFP4_QB_LATE_CLAIM_MIN_BLOCKS_PER_SM = (11, 8)
_NVFP4_QB_WINDOW = 128
_NVFP4_PREFILL_VARIANTS = (
    _NVFP4_VARIANT_QB_H16_PREFILL,
    _NVFP4_VARIANT_QB_H16_BREAKS_PREFILL,
    _NVFP4_VARIANT_QB_H32_PREFILL,
    _NVFP4_VARIANT_QB_H32_LATE_CLAIM_PREFILL,
    _NVFP4_VARIANT_H64_TMEM_V_PREFILL,
    _NVFP4_VARIANT_H64_TMEM_V_EARLY_GATHER_PREFILL,
    _NVFP4_VARIANT_H64_EARLY_GATHER_PREFILL,
    _NVFP4_VARIANT_H64_O_BOXES_PREFILL,
    _NVFP4_VARIANT_H64_Q_PREFETCH_PREFILL,
    _NVFP4_VARIANT_H128_PREFILL,
    _NVFP4_VARIANT_H128_V_UNIT_PAIRS_PREFILL,
)
# Prefill bodies that load index rows 16 bytes (four int32) at a time with cp.async or vector loads: the
# two-CTA H128 builds and the query-block builds. Their tables must start 16-byte aligned and keep a row
# stride that is a multiple of 4 elements (:func:`_check_index_row_alignment`); the per-token H <= 64
# builds read one index at a time.
_NVFP4_PREFILL_VECTOR_INDEX_VARIANTS = frozenset(
    {
        _NVFP4_VARIANT_QB_H16_PREFILL,
        _NVFP4_VARIANT_QB_H16_BREAKS_PREFILL,
        _NVFP4_VARIANT_QB_H32_PREFILL,
        _NVFP4_VARIANT_QB_H32_LATE_CLAIM_PREFILL,
        _NVFP4_VARIANT_H128_PREFILL,
        _NVFP4_VARIANT_H128_V_UNIT_PAIRS_PREFILL,
    }
)
_NVFP4_H64_PREFILL_MAX_HEADS = 64
_NVFP4_TMEM_V_MAX_HEADS = 32
_NVFP4_DATA_BYTES = 352  # E2M1 NoPE codes and BF16 RoPE dims of one NVFP4 row
_NVFP4_SF_BYTES = 32  # the row's 28 E4M3 scales and 4 pad bytes
_NVFP4_ROW_PITCH = 32  # pitch of the pool row views the prefill tensor maps read


# Work feed of the BF16/H128 persistent prefill body (mirrors the Cake seed's
# bf16_h128_prefill_uses_snake_feed).  With C = min(T, SMs // 2)
# clusters the striped feed gives base = T // C strided PREFIX items to the
# C - T % C regular clusters and base + 1 contiguous SUFFIX items to the T % C
# tail clusters; when the tail clusters are the majority the few regular
# clusters own a short heavy prefix (hardening-000037: 23-30 -> 17 critical
# tiles), so the boustrophedon program is launched instead.  One-tile items
# (SWA-only rows) deal identically under both feeds and keep the striped
# program.
def _bf16_h128_prefill_uses_snake_feed(
    num_query_tokens: int, sparse_topk: int, num_clusters: int
) -> bool:
    tokens = int(num_query_tokens)
    clusters = min(tokens, int(num_clusters))
    if clusters <= 0 or tokens <= clusters:
        return False
    if (int(sparse_topk) + _TILE_KV - 1) // _TILE_KV < 2:
        return False
    return 2 * (tokens % clusters) > clusters


@functools.lru_cache(maxsize=None)
def _device_sm_count(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


@functools.lru_cache(maxsize=None)
def _bf16_h128_prefill_num_clusters(device: torch.device) -> int:
    props = torch.cuda.get_device_properties(device)
    return max(int(props.multi_processor_count) // 2, 1)


KERNEL_METADATA_PARAMS = (
    "swa_indices",
    "compressed_indices",
    "sparse_topk_lens",
    "swa_index_stride",
    "compressed_index_stride",
    "sparse_topk_lens_offset",
    "sparse_topk",
    "num_query_tokens",
)

# Query-layout parameters of every generated variant (the SWA validity window;
# see the module docstring). Mirrors the Cake seeds' SWA_WINDOW_PARAMS.
QUERY_LAYOUT_PARAMS = (
    "seq_lens",
    "cum_seq_lens_q",
    "ragged_query",
    "max_q_len",
    "batch_size",
)

_scale_cache: dict[tuple[str, Optional[int], float], torch.Tensor] = {}
_scale_cache_lock = threading.Lock()
_dense_offsets_cache: dict[tuple[str, Optional[int], int, int], torch.Tensor] = {}


def _target_arch(device: torch.device) -> str:
    if device.type != "cuda":
        raise ValueError(f"CAKE DSv4 requires CUDA tensors, got {device}")
    major, minor = get_compute_capability(device)
    return f"sm_{major}{minor}a"


def _variant_module(variant: str, *, arch: str):
    from ..jit.cake_dsv4 import get_cake_dsv4_module

    return get_cake_dsv4_module(variant, arch=arch)


def _is_capturing(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    with torch.cuda.device(device):
        return bool(torch.cuda.is_current_stream_capturing())


# --------------------------------------------------------------------------- #
# Sparse metadata resolution                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SparseMetadata:
    """Resolved kernel-side view of the DSv4 sparse metadata (no copies)."""

    swa_indices: torch.Tensor
    compressed_indices: torch.Tensor
    sparse_topk_lens: torch.Tensor
    swa_index_stride: int
    compressed_index_stride: int
    sparse_topk_lens_offset: int
    sparse_topk: int
    num_query_tokens: int
    separate_tables: bool

    def kernel_kwargs(self) -> dict[str, Any]:
        """Keyword arguments in the shared kernel parameter vocabulary.

        The two index tables are handed over as contiguous spans (see
        ``_flat_index_span``): the generated bindings check every pointer
        buffer for contiguity, and the column-sliced compressed view of a
        combined table is not contiguous even though the kernel only reads
        its base pointer plus the row stride.
        """
        values = {name: getattr(self, name) for name in KERNEL_METADATA_PARAMS}
        values["swa_indices"] = _flat_index_span(self.swa_indices)
        values["compressed_indices"] = _flat_index_span(self.compressed_indices)
        return values

    @property
    def compressed_width(self) -> int:
        return self.sparse_topk - _SWA_WIDTH


def _flat_index_span(table: torch.Tensor) -> torch.Tensor:
    """Expose a row-strided int32 table as one contiguous span without a copy.

    A contiguous table is returned as is. A column-sliced view (row stride
    larger than its width) becomes the 1-D span from its first to its last
    element, which keeps ``data_ptr`` and the row stride the kernel indexes
    with while satisfying the binding's contiguity check.
    """
    if table.is_contiguous() or table.ndim != 2:
        return table
    rows, cols = int(table.shape[0]), int(table.shape[1])
    if rows == 0 or cols == 0:
        return table.reshape(-1)
    return table.as_strided(((rows - 1) * int(table.stride(0)) + cols,), (1,))


def _int32_table(tensor: torch.Tensor, name: str, *, rows: Optional[int] = None):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} must be a [rows, columns] table, got shape {tuple(tensor.shape)}"
        )
    if tensor.ndim > 2:
        try:
            tensor = tensor.view(-1, tensor.shape[-1])
        except RuntimeError as exc:
            raise ValueError(
                f"{name} leading dimensions must be densely packed so they fold "
                "into rows without a copy"
            ) from exc
    if rows is not None and tensor.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got {tensor.shape[0]}")
    if tensor.shape[1] and tensor.stride(1) != 1:
        raise ValueError(
            f"{name} must have a unit column stride; pass a row-strided view "
            "instead of a copy"
        )
    return tensor


def _check_index_row_alignment(table: torch.Tensor, name: str) -> None:
    """Reject an index table the 16-byte index-row loads cannot read.

    The BF16 / FP8 programs load four int32 indices at a time from
    ``data_ptr + 4 * (t * row_stride + c)`` with ``c`` a multiple of 4, so the
    table must start 16-byte aligned and, past one row, have a row stride that
    is a multiple of 4 elements. Row-tiled launches slice the table by row,
    which keeps both properties. The NVFP4 decode family and the per-token
    H <= 64 prefill builds read their tables one index at a time; the NVFP4
    H128 and query-block prefill builds load 16-byte rows and call this
    (``_NVFP4_PREFILL_VECTOR_INDEX_VARIANTS``).
    """
    misaligned = table.data_ptr() % _INDEX_ROW_ALIGN
    if misaligned:
        raise ValueError(
            f"{name} must start at a {_INDEX_ROW_ALIGN}-byte aligned address, got "
            f"data_ptr() % {_INDEX_ROW_ALIGN} == {misaligned}: the kernels load "
            f"index rows {_INDEX_ROW_ALIGN} bytes at a time"
        )
    if table.shape[0] > 1 and table.stride(0) % 4:
        raise ValueError(
            f"{name} must have a row stride that is a multiple of 4 elements, got "
            f"{table.stride(0)}: the kernels load index rows {_INDEX_ROW_ALIGN} "
            "bytes at a time"
        )


def _int32_lens(tensor: torch.Tensor, name: str, *, rows: int) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.ndim != 1:
        try:
            tensor = tensor.view(-1)
        except RuntimeError as exc:
            raise ValueError(
                f"{name} must flatten to one row per token without a copy"
            ) from exc
    if tensor.numel() != rows:
        raise ValueError(f"{name} must have {rows} entries, got {tensor.numel()}")
    if tensor.numel() and tensor.stride(0) != 1:
        raise ValueError(f"{name} must have unit stride")
    return tensor


def resolve_cake_dsv4_sparse_metadata(
    sparse_indices: torch.Tensor,
    sparse_topk_lens: Optional[torch.Tensor] = None,
    *,
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
    sparse_topk_lens_offset: int = 0,
    query_rows: int,
) -> SparseMetadata:
    """Resolve the shared metadata ABI from combined or separate host tables.

    Combined form: ``sparse_indices [T, sparse_topk]`` whose first 128 columns
    are SWA slots, with ``sparse_topk_lens`` counting those 128 slots.

    Separate form: ``sparse_indices [T, 128]`` is the SWA table and
    ``extra_sparse_indices [T, topk_c]`` the compressed table. Lengths come from
    ``sparse_topk_lens`` (combined convention) or ``extra_sparse_topk_lens``
    (compressed slots only; the host adds 128 to ``sparse_topk_lens_offset``).

    ``query_rows`` is the number of rows the query tensor provides; the metadata
    may describe fewer tokens (padded batch) but never more.

    Every bound table must start at a 16-byte aligned address and, with more
    than one row, have a row stride that is a multiple of 4 elements (the
    BF16 / FP8 kernels load index rows 16 bytes at a time); other tables raise
    ``ValueError``.
    """
    if isinstance(sparse_topk_lens_offset, bool) or not isinstance(
        sparse_topk_lens_offset, int
    ):
        raise TypeError("sparse_topk_lens_offset must be an int")
    offset = int(sparse_topk_lens_offset)
    if extra_sparse_indices is not None:
        swa = _int32_table(sparse_indices, "sparse_indices")
        if swa.shape[1] != _SWA_WIDTH:
            raise ValueError(
                "with extra_sparse_indices, sparse_indices is the SWA table and "
                f"must have {_SWA_WIDTH} columns, got {swa.shape[1]}"
            )
        rows = int(swa.shape[0])
        compressed = _int32_table(
            extra_sparse_indices, "extra_sparse_indices", rows=rows
        )
        compressed_width = int(compressed.shape[1])
        if compressed_width == 0:
            compressed = swa
        if extra_sparse_topk_lens is not None:
            if sparse_topk_lens is not None:
                raise ValueError(
                    "pass either sparse_topk_lens (combined, counting the 128 SWA "
                    "slots) or extra_sparse_topk_lens (compressed slots only), not both"
                )
            lens, lens_name = extra_sparse_topk_lens, "extra_sparse_topk_lens"
            offset += _SWA_WIDTH
        else:
            if sparse_topk_lens is None:
                raise ValueError(
                    "sparse_topk_lens or extra_sparse_topk_lens is required"
                )
            lens, lens_name = sparse_topk_lens, "sparse_topk_lens"
        separate = True
    else:
        if extra_sparse_topk_lens is not None:
            raise ValueError("extra_sparse_topk_lens requires extra_sparse_indices")
        if sparse_topk_lens is None:
            raise ValueError(
                "sparse_topk_lens is required with a combined sparse_indices table"
            )
        table = _int32_table(sparse_indices, "sparse_indices")
        rows = int(table.shape[0])
        if table.shape[1] < _SWA_WIDTH:
            raise ValueError(
                f"sparse_indices must have at least {_SWA_WIDTH} columns, got {table.shape[1]}"
            )
        swa = table
        compressed_width = int(table.shape[1]) - _SWA_WIDTH
        compressed = table[:, _SWA_WIDTH:] if compressed_width else table
        lens, lens_name = sparse_topk_lens, "sparse_topk_lens"
        separate = False

    if rows < 1:
        raise ValueError("sparse metadata must describe at least one query token")
    if rows > query_rows:
        raise ValueError(
            f"metadata has {rows} rows but the query only has {query_rows}"
        )
    sparse_topk = _SWA_WIDTH + compressed_width
    if sparse_topk % 4:
        raise ValueError(
            f"sparse_topk (128 + compressed columns) must be a multiple of 4, got {sparse_topk}"
        )
    _check_index_row_alignment(swa, "sparse_indices")
    if compressed is not swa:
        _check_index_row_alignment(
            compressed, "extra_sparse_indices" if separate else "sparse_indices"
        )
    lens = _int32_lens(lens, lens_name, rows=rows)
    return SparseMetadata(
        swa_indices=swa,
        compressed_indices=compressed,
        sparse_topk_lens=lens,
        swa_index_stride=int(swa.stride(0)),
        compressed_index_stride=int(compressed.stride(0)),
        sparse_topk_lens_offset=offset,
        sparse_topk=sparse_topk,
        num_query_tokens=rows,
        separate_tables=separate,
    )


# --------------------------------------------------------------------------- #
# Workspace                                                                   #
# --------------------------------------------------------------------------- #


def _align_up(num_bytes: int) -> int:
    return -(-num_bytes // _ALIGN) * _ALIGN


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _upper_bound_num_splits(sparse_topk: int) -> int:
    return max(-(-sparse_topk // _TILE_KV), _MAX_FIXED_SPLITS)


@dataclass(frozen=True)
class WorkspaceLayout:
    """Byte ``(offset, size)`` of every region carved from ``workspace_buffer``."""

    descriptor_slab: tuple[int, int]
    counters: tuple[int, int]
    partial_o: tuple[int, int]
    partial_lse: tuple[int, int]
    # Shifted request offsets of a row-tiled chunk (size 0 for whole launches).
    query_offsets: tuple[int, int]
    total_bytes: int
    # Final base-2 LSE f32 [T * H]; only carved for the NVFP4 route
    # (``with_lse=True``), otherwise an empty region at the layout end.
    lse: tuple[int, int] = (0, 0)
    # NVFP4 prefill bodies (``with_prefill=True``): two int32 [T] length
    # vectors the host fills when the caller passes none; otherwise an empty
    # region at the layout end.
    prefill_lens: tuple[int, int] = (0, 0)


def cake_dsv4_workspace_layout(
    num_query_tokens: int,
    num_heads: int,
    num_splits: int,
    *,
    num_query_offsets: int = 0,
    with_lse: bool = False,
    with_prefill: bool = False,
) -> WorkspaceLayout:
    """Deterministic carve of ``workspace_buffer`` for one launch shape.

    One-split launches (``num_splits == 1``) write the final output directly
    and carry no ``partial_O`` region (size 0); the LSE region follows the
    counters immediately. ``num_query_offsets`` (``batch_size + 1`` for a
    row-tiled chunk after the first, else 0) sizes the int32 region after
    ``partial_lse`` that holds the chunk's shifted ``cum_seq_lens_q``.
    ``with_lse`` appends the NVFP4 route's final-LSE region after that, and
    ``with_prefill`` the NVFP4 prefill bodies' length regions after the LSE; the default layout is unchanged.
    """
    tokens = _positive_int(num_query_tokens, "num_query_tokens")
    heads = _positive_int(num_heads, "num_heads")
    splits = _positive_int(num_splits, "num_splits")
    if (
        isinstance(num_query_offsets, bool)
        or not isinstance(num_query_offsets, int)
        or num_query_offsets < 0
    ):
        raise ValueError(
            f"num_query_offsets must be a non-negative int, got {num_query_offsets!r}"
        )
    partial_elems = tokens * heads * splits
    o_bytes = (
        _align_up(partial_elems * _HEAD_DIM * torch.bfloat16.itemsize)
        if splits > 1
        else 0
    )
    lse_bytes = _align_up(partial_elems * torch.float32.itemsize)
    offsets_bytes = _align_up(num_query_offsets * torch.int32.itemsize)
    o_offset = _PARTIAL_OFFSET
    lse_offset = o_offset + o_bytes
    offsets_offset = lse_offset + lse_bytes
    final_lse_offset = offsets_offset + offsets_bytes
    final_lse_bytes = (
        _align_up(tokens * heads * torch.float32.itemsize) if with_lse else 0
    )
    lens_offset = final_lse_offset + final_lse_bytes
    lens_bytes = 2 * _align_up(tokens * torch.int32.itemsize) if with_prefill else 0
    return WorkspaceLayout(
        descriptor_slab=(_DESCRIPTOR_SLAB_OFFSET, _DESCRIPTOR_SLAB_BYTES),
        counters=(_COUNTER_OFFSET, _COUNTER_REGION_BYTES),
        partial_o=(o_offset, o_bytes),
        partial_lse=(lse_offset, lse_bytes),
        query_offsets=(offsets_offset, offsets_bytes),
        total_bytes=lens_offset + lens_bytes,
        lse=(final_lse_offset, final_lse_bytes),
        prefill_lens=(lens_offset, lens_bytes),
    )


def get_cake_dsv4_workspace_bytes(
    num_query_tokens: int,
    num_heads: int,
    sparse_topk: int,
    dtype: torch.dtype,
    *,
    num_splits: Optional[int] = None,
    kv_cache_format: Literal["fp8", "nvfp4"] = "fp8",
    extra_topk: int = 0,
) -> int:
    """Single-launch upper bound of ``workspace_buffer`` for ``backend="cake"``.

    ``num_query_tokens`` is the metadata row count (padded query rows do not
    count). The result bounds every route's one-launch carve::

        S      = num_splits if given else max(ceil(sparse_topk / 128), 5)
        bytes  = 1024                                   # reserved slab
               + 262144                                 # split-merge counters (uint32[65536])
               + align128(num_query_tokens * num_heads * S * 512 * 2)   # partial_O (BF16), S > 1 only
               + align128(num_query_tokens * num_heads * S * 4)         # partial_lse (FP32)

    Partial buffers are BF16/FP32 for both BF16 and FP8 inputs; ``dtype`` is
    validated only. Pass ``num_splits`` to size for a known route (routes use
    ``ceil(sparse_topk / 128)`` or a fixed 1..5 splits). The FP8/H128
    uniform-gather route runs one split, but at the shapes of the SM107 split
    rules its split-KV launch carves ``cake_dsv4_workspace_layout(T, H, s)``
    (module docstring, "Split-KV pieces"): size those calls without
    ``num_splits`` (``S >= s``: a split has at most one piece per tile) or with
    :func:`cake_dsv4_workspace_requirement`'s ``single_launch_bytes``, not
    with ``num_splits=1``, which runs the route's own launch. A smaller buffer is
    not an error: the host tiles the metadata rows into launches that fit,
    down to the one-row requirement; :func:`cake_dsv4_workspace_requirement`
    reports the exact per-route numbers for a call.

    ``kv_cache_format="nvfp4"`` sizes the NVFP4 cache route instead: ``S`` is
    ``num_splits`` if given, else ``min(ceil(sparse_topk / 128) +
    ceil(extra_topk / 128), 12)`` (the plan never exceeds the candidate tile
    count or the kernel's 12-split cap), and a final-LSE region
    ``align128(num_query_tokens * num_heads * 4)`` is appended, then the
    prefill bodies' two ``align128(num_query_tokens * 4)`` length regions (the
    bound covers the decode family and the prefill bodies alike). A call with a
    packed NVFP4 query runs the prefill bodies, which write the final output and
    LSE directly and need only the one-split carve
    ``cake_dsv4_workspace_layout(num_query_tokens, num_heads, 1, with_lse=True,
    with_prefill=True).total_bytes``.
    """
    if dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(f"unsupported CAKE DSv4 dtype: {dtype}")
    topk = _positive_int(sparse_topk, "sparse_topk")
    if kv_cache_format == "nvfp4":
        if dtype != torch.bfloat16:
            raise ValueError(
                "the CAKE DSv4 NVFP4 route sizes its workspace with dtype "
                "torch.bfloat16 (a BF16 or a packed NVFP4 query)"
            )
        if (
            isinstance(extra_topk, bool)
            or not isinstance(extra_topk, int)
            or extra_topk < 0
        ):
            raise ValueError(
                f"extra_topk must be a non-negative int, got {extra_topk!r}"
            )
        splits = (
            min(
                _ceil_div(topk, _TILE_KV) + _ceil_div(extra_topk, _TILE_KV),
                _NVFP4_MAX_SPLITS,
            )
            if num_splits is None
            else _positive_int(num_splits, "num_splits")
        )
        return cake_dsv4_workspace_layout(
            num_query_tokens, num_heads, splits, with_lse=True, with_prefill=True
        ).total_bytes
    if kv_cache_format != "fp8":
        raise ValueError(f"unsupported CAKE DSv4 kv_cache_format: {kv_cache_format!r}")
    if topk < _SWA_WIDTH or topk % 4:
        raise ValueError(
            f"sparse_topk must be a multiple of 4 and at least {_SWA_WIDTH}, got {topk}"
        )
    splits = (
        _upper_bound_num_splits(topk)
        if num_splits is None
        else _positive_int(num_splits, "num_splits")
    )
    return cake_dsv4_workspace_layout(num_query_tokens, num_heads, splits).total_bytes


def _workspace_bytes(workspace: torch.Tensor) -> torch.Tensor:
    if not isinstance(workspace, torch.Tensor):
        raise TypeError("workspace_buffer must be a torch.Tensor")
    if not workspace.is_contiguous():
        raise ValueError("workspace_buffer must be contiguous")
    raw = workspace.view(torch.uint8).reshape(-1)
    if raw.data_ptr() % _ALIGN:
        raise ValueError(f"workspace_buffer must be {_ALIGN}-byte aligned")
    return raw


def _require_workspace_bytes(raw: torch.Tensor, needed: int) -> None:
    if raw.numel() < needed:
        raise ValueError(
            f"workspace_buffer requires at least {needed} bytes for this CAKE DSv4 "
            f"launch, got {raw.numel()}; size it with get_cake_dsv4_workspace_bytes()"
        )


def _partial_views(
    raw: torch.Tensor,
    out_rows: torch.Tensor,
    num_query_tokens: int,
    num_heads: int,
    num_splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = cake_dsv4_workspace_layout(num_query_tokens, num_heads, num_splits)
    _require_workspace_bytes(raw, layout.total_bytes)
    elems = num_query_tokens * num_heads * num_splits
    lse_offset = layout.partial_lse[0]
    partial_lse = raw[lse_offset : lse_offset + elems * torch.float32.itemsize].view(
        torch.float32
    )
    if num_splits == 1:
        # Single-partition routes write the final output directly.
        return out_rows.reshape(-1), partial_lse
    o_offset = layout.partial_o[0]
    partial_o = raw[
        o_offset : o_offset + elems * _HEAD_DIM * torch.bfloat16.itemsize
    ].view(torch.bfloat16)
    return partial_o, partial_lse


def _counters(raw: torch.Tensor, merge_groups: int) -> torch.Tensor:
    if merge_groups > _MAX_MERGE_GROUPS:
        raise ValueError(
            f"CAKE DSv4 split-merge route needs {merge_groups} counters; the workspace "
            f"counter region holds {_MAX_MERGE_GROUPS}"
        )
    _require_workspace_bytes(raw, _PARTIAL_OFFSET)
    return raw[_COUNTER_OFFSET : _COUNTER_OFFSET + merge_groups * 4].view(torch.uint32)


def _workspace_owner(workspace: torch.Tensor) -> torch.Tensor:
    """The tensor whose lifetime owns ``workspace``'s storage (its ``_base`` or itself)."""
    base = workspace._base
    return workspace if base is None else base


def _primed_key(raw: torch.Tensor) -> tuple[Optional[int], int]:
    return (raw.device.index, raw.data_ptr())


def _counters_primed(workspace: torch.Tensor, raw: torch.Tensor) -> bool:
    """Whether the counter region behind ``raw`` is registered as zero."""
    with _primed_lock:
        ref = _primed_workspaces.get(_primed_key(raw))
    return ref is not None and ref() is _workspace_owner(workspace)


def _register_primed(workspace: torch.Tensor, raw: torch.Tensor) -> None:
    with _primed_lock:
        for key in [k for k, ref in _primed_workspaces.items() if ref() is None]:
            del _primed_workspaces[key]
        _primed_workspaces[_primed_key(raw)] = weakref.ref(_workspace_owner(workspace))


def cake_dsv4_workspace_reset(workspace_buffer: torch.Tensor) -> None:
    """Zero the split-merge counter region of ``workspace_buffer`` and register it.

    Optional: the launch contract zeroes the counters itself (first eager use
    zeroes and registers the workspace; a capture through an unregistered
    workspace records the zero fill into the graph). Calling this after
    allocating a workspace only moves that first fill out of the hot path.
    Zeroing is an in-place fill; nothing is allocated.
    """
    raw = _workspace_bytes(workspace_buffer)
    _require_workspace_bytes(raw, _PARTIAL_OFFSET)
    raw[_COUNTER_OFFSET:_PARTIAL_OFFSET].zero_()
    _register_primed(workspace_buffer, raw)


def _ensure_counters_zeroed(
    workspace: torch.Tensor, raw: torch.Tensor, merge_groups: int
) -> None:
    """Make the ``merge_groups`` counters of this launch zero; never raises for capture.

    Registered workspace: nothing (the kernels leave the counters zero).
    Unregistered, eager: zero the whole region once and register it.
    Unregistered, under CUDA Graph capture: record a zero fill of exactly the
    counters this launch uses into the graph; the registry is left alone
    because the fill has not executed and only replays carry it.
    """
    if _counters_primed(workspace, raw):
        return
    if _is_capturing(workspace.device):
        raw[_COUNTER_OFFSET : _COUNTER_OFFSET + merge_groups * 4].zero_()
        return
    raw[_COUNTER_OFFSET:_PARTIAL_OFFSET].zero_()
    _register_primed(workspace, raw)


# --------------------------------------------------------------------------- #
# Name-based argument binding                                                 #
# --------------------------------------------------------------------------- #

_TMA_SOURCE_ALIASES: Mapping[str, str] = {
    "tmap_q": "Q",
    # The NVFP4 prefill bodies read the packed query [tokens, heads, 384] through
    # a data map and a scale map over the same rows, and each pool through a
    # data-row map and a scale-row map (``_nvfp4_pool_row_views``).
    "tmap_q_sf": "Q",
    "tmap_swa_sf": "SWA_sf",
    "tmap_compressed_sf": "compressed_KV_sf",
    "tmap_swa_k": "SWA_cache",
    "tmap_swa_v": "SWA_cache",
    "tmap_swa_kv": "SWA_cache",
    "tmap_compressed_k": "compressed_KV_cache",
    "tmap_compressed_v": "compressed_KV_cache",
    "tmap_compressed_kv": "compressed_KV_cache",
    # The FP8 persistent bodies (round 5) store O through a TMA descriptor over
    # the same [tokens, heads, 512] rows the plain ``O`` pointer argument sees.
    "tmap_o": "O",
    # The bf16 H64 guard program binds its ``O`` parameter itself as a 3-D
    # tensor map (box 64 x 16 x 1 over [tokens, heads, 512]; head rows >=
    # num_heads are clipped by the map), so the registration carries
    # ("tma_buffer", "O"): the same output rows, encoded by the grid_constant
    # binding -- no descriptor workspace and no host copy.
    "O": "O",
    # NVFP4 decode members store their (partial) output through a TMA
    # descriptor over the [tokens, heads, splits, 512] view of partial_O.
    "tmap_out": "partial_O_tiles",
    # The SM107 split-KV producer stores its pieces' partial O through a TMA
    # descriptor over the [pieces, heads, 512] view of partial_O.
    "tmap_partial_o": "partial_O_pieces",
    # The gather4 members (pv / t64) read the pools through TMA gather descriptors over 32-byte-pitch int32 row
    # views of each flat pool (data rows and footer rows; ``_nvfp4_gather4_views``).
    "tmap_g4d": "main_cache_g4d",
    "tmap_g4f": "main_cache_g4f",
    "tmap_g4dx": "extra_cache_g4d",
    "tmap_g4fx": "extra_cache_g4f",
}
_SCALAR_ALIASES: Mapping[str, str] = {
    "num_q_heads": "num_heads",
    "num_split": "num_splits",
}
_TENSOR_VALUE_NAMES = frozenset(
    {
        "Q",
        "SWA_cache",
        "compressed_KV_cache",
        "O",
        "partial_O",
        "partial_O_pieces",
        "partial_lse",
        "partition_arrivals",
        "seq_lens",
        "cum_seq_lens_q",
        "sinks",
        "bmm1_scale",
        "bmm2_scale",
        "swa_indices",
        "compressed_indices",
        "sparse_topk_lens",
        # NVFP4 cache route (two independent tables over two uint8 pools).
        "q_rows",
        "main_cache",
        "extra_cache",
        "main_indices",
        "extra_indices",
        "main_lengths",
        "extra_lengths",
        "partial_O_tiles",
        "lse_out",
        "main_cache_g4d",
        "main_cache_g4f",
        "extra_cache_g4d",
        "extra_cache_g4f",
        # NVFP4 prefill bodies (pool scale-row views, the final LSE, the extra table's lengths).
        "SWA_sf",
        "compressed_KV_sf",
        "LSE",
        "extra_topk_lens",
    }
)
# Kernel parameters bound as Python floats (the NVFP4 members take the LSE
# scales by value); every other parameter is an int.
_FLOAT_SCALAR_VALUE_NAMES = frozenset({"lse_partial_scale", "lse_scale"})
_SCALAR_VALUE_NAMES = frozenset(
    {
        "swa_index_stride",
        "compressed_index_stride",
        "sparse_topk_lens_offset",
        "sparse_topk",
        "num_query_tokens",
        "num_heads",
        "num_head_tiles",
        "has_sinks",
        "num_splits",
        "total_work_items",
        # Whole rows before the pieces of a split-KV launch.
        "num_whole_items",
        "batch_size",
        "max_q_len",
        "ragged_query",
        # Rows of the flat compressed-KV pool: SM107's last-tile L2 prefetch
        # sibling of the four-stage M64 program bounds the pool rows it
        # prefetches by it (_kv4_l2pf_variant).
        "compressed_KV_rows",
        # NVFP4 cache route.
        "num_main_tiles",
        "tiles_per_split",
        "total_tiles",
        "main_width",
        "extra_width",
        "main_index_stride",
        "extra_index_stride",
        "has_main_lengths",
        "has_extra_lengths",
        "main_page_shift",
        "extra_page_shift",
        "main_page_stride",
        "extra_page_stride",
        "heads_per_cta",
        # Retrace knobs of the templated NVFP4 members (runtime instantiation switch).
        "tile_n",
        "o_chunks",
        # NVFP4 prefill bodies: pool geometry in 32-byte units and the two table widths.
        "swa_page_log2",
        "swa_pitch_units",
        "swa_footer_units",
        "compressed_page_log2",
        "compressed_pitch_units",
        "compressed_footer_units",
        "swa_width",
        "compressed_width",
        *_FLOAT_SCALAR_VALUE_NAMES,
    }
)
_GRID_NAMES = ("grid_x", "grid_y", "grid_z")
# The per-call launch argument of the programs that leave programmatic dependent
# launch to each call (``["launch", "enable_pdl"]`` in their plan, 0 or 1).
_LAUNCH_NAMES = ("enable_pdl",)
# Producers whose generated ABI reads request boundaries from cum_seq_lens_q
# only; run_cake_dsv4 synthesizes the dense offsets for them.
_RAGGED_ONLY_ROUTES = frozenset(
    {
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h64_prefill_source_persistent_m64_multi_tile",
    }
)
# Mirrors the Cake seed's LANE_GATHER_MIN_TOKENS. Below it the
# persistent FP8 body runs the program with elected-lane uniform K/V gathers;
# from 128 tokens on, the lane-issued gathers (several waves per cluster) win.
_FP8_PERSISTENT_LANE_GATHER_MIN_TOKENS = 128

# Per architecture, (fewest query rows, longest request, topk4x items only)
# entries: launches that reach an entry's row count and whose requests hold at
# most its query rows keep the uniform-gather route past the lane-gather floor.
# On SM107 that route runs its K=64 sibling, which led the lane-issued program
# on these short-query decode launches (VR200, one process, eight calls per
# CUDA graph).  It trailed it at almost every measured row count on topk4x and
# topk128x prefill, with 16 or more query rows per request on topk4x items (8
# at 160 rows), with 8 or more on topk128x items, and with 4 on topk128x items
# 1152 wide (2 at 160 rows); it was mixed with 4 query rows on narrower
# topk128x and on SWA-only items.
_FP8_PERSISTENT_SHORT_QUERY_UNIFORM = {
    "sm_107a": ((160, 1, False), (256, 2, False), (160, 4, True), (256, 8, True)),
}


# Architectures whose SWA-only decode launches keep the uniform-gather route at
# every row count, with the longest request (``max_q_len``) they apply to.
# GB300 (152 SMs), 128 heads, newest-first windows: q_len 1 at 128-4096 rows,
# lane-issued route (its PDL-free twin from 384 rows) 11.58-160.90 us ->
# uniform 11.17-160.38 us (1.002-1.044x); q_len 2 at 512-4096 rows 0.993-1.016x
# (noise); q_len 4 and 8 lose from 1024 rows (0.939-0.989x), so they keep the
# lane-issued route.
_FP8_PERSISTENT_SWA_DECODE_UNIFORM = {"sm_103a": 2}


def _fp8_persistent_program(
    num_query_tokens: int,
    *,
    max_q_len: Optional[int] = None,
    topk4x: bool = False,
    arch: Optional[str] = None,
    swa: bool = False,
) -> str:
    if num_query_tokens < _FP8_PERSISTENT_LANE_GATHER_MIN_TOKENS:
        return "fp8_h128_prefill_source_persistent_uniform"
    if (
        swa
        and arch in _FP8_PERSISTENT_SWA_DECODE_UNIFORM
        and max_q_len is not None
        and max_q_len <= _FP8_PERSISTENT_SWA_DECODE_UNIFORM[arch]
    ):
        return "fp8_h128_prefill_source_persistent_uniform"
    entries = _FP8_PERSISTENT_SHORT_QUERY_UNIFORM.get(arch, ()) if arch else ()
    if max_q_len is not None and any(
        num_query_tokens >= rows and max_q_len <= longest and (topk4x or not only4x)
        for rows, longest, only4x in entries
    ):
        return "fp8_h128_prefill_source_persistent_uniform"
    return "fp8_h128_prefill_source_persistent"


# SM107: the FP8/H64 rows of one complete sparse tile and at least half of a
# second (widths 192-252) also run the persistent body past 12 tokens, and from
# 48 keys of a second (widths 176-191) up to 16 tokens.  Per architecture:
# (minimum width, minimum width up to 16 tokens).
_FP8_H64_PERSISTENT_ONE_TILE_MIN_WIDTH = {"sm_107a": (192, 176)}

# SM107 launches whose multi-tile M64 route runs its four-stage sibling
# (:func:`_fp8_h64_kv4_edges`): per complete-tile count, the minimum width that
# keeps the persistent body at 13-16 tokens.  R200 (208 and 216 SMs, three
# boards), q_len 1 and 4, eight distinct-input calls per CUDA graph / single
# cold calls, persistent body -> four-stage M64 body: width 176 1.007-1.030x /
# 1.037-1.085x, widths 260-352 0.999-1.060x / 1.029-1.091x; width 380 trailed
# in the graph at 13 tokens (q_len 1) and 16 tokens (q_len 4), down to 0.990x,
# so widths 353-383 keep the persistent body.
_FP8_H64_PERSISTENT_SHORT_MIN_WIDTH_KV4 = {"sm_107a": {1: 177, 2: 353}}

# Architectures whose FP8/H64 rows of two complete sparse tiles and a partial
# third (widths 257-383) run the persistent body up to 12 tokens as well, in
# place of the H64 cluster body (:func:`_fp8_h64_uses_cluster_body`).
_FP8_H64_PERSISTENT_PARTIAL_THIRD_ARCHES = frozenset({"sm_103a"})


def _fp8_h64_uses_persistent_body(
    sparse_topk: int,
    num_query_tokens: int,
    arch: Optional[str] = None,
    kv4: bool = False,
) -> bool:
    """FP8/H64 rows that run the FP8/H128 persistent body while it fits one wave
    (Cake seed ``portfolio_v34.uses_persistent_body``).

    On SM107 the rows of one complete tile and at least half of a second join
    them past 12 tokens, and from 48 keys of a second up to 16 tokens
    (``_FP8_H64_PERSISTENT_ONE_TILE_MIN_WIDTH``).  VR200 (208 SMs, two
    boards), q_len 1 and 4: widths 192-252 at 13-31 tokens, M64 body
    5.93-6.19 -> persistent body 5.65-6.17 us (0.99-1.05x; 1.00-1.05x up to
    24 tokens); width 176 at 13-16 tokens 5.93-6.03 -> 5.78-5.90 us
    (1.02-1.03x), 0.97-1.01x past 16 tokens.  Narrower widths stay on the M64
    body (1.00-1.02x up to 16 tokens, 0.97-1.00x past them, one board).
    Launches whose multi-tile M64 route runs its four-stage sibling (``kv4``)
    take the minimum widths up to 16 tokens from
    ``_FP8_H64_PERSISTENT_SHORT_MIN_WIDTH_KV4`` where it has an entry.

    On SM103 the rows of two complete tiles and a partial third (widths
    257-383) run it up to 12 tokens too
    (``_FP8_H64_PERSISTENT_PARTIAL_THIRD_ARCHES``).  GB300 (152 SMs), q_len 1,
    1-12 tokens: widths 260-380 with every slot live, H64 cluster body
    9.38-10.19 -> persistent body 8.22-9.17 us
    (1.08-1.20x); width 320 with 256 live (a C128 table padded to the
    server's context) 8.90-9.31 -> 7.52-7.62 us (1.18-1.23x).
    """
    full_tiles = sparse_topk // 128
    if num_query_tokens <= 12:
        if arch in _FP8_H64_PERSISTENT_PARTIAL_THIRD_ARCHES and full_tiles == 2:
            return sparse_topk > 2 * _TILE_KV
        return full_tiles >= 3
    if kv4 and num_query_tokens <= 16:
        short_min_width = _FP8_H64_PERSISTENT_SHORT_MIN_WIDTH_KV4.get(arch, {}).get(
            full_tiles
        )
        if short_min_width is not None:
            return sparse_topk >= short_min_width
    if full_tiles == 1:
        min_width, short_min_width = _FP8_H64_PERSISTENT_ONE_TILE_MIN_WIDTH.get(
            arch, (2 * _TILE_KV, 2 * _TILE_KV)
        )
        if num_query_tokens <= 16:
            min_width = short_min_width
        return sparse_topk >= min_width
    return full_tiles >= 2


# SM107: the M64 body (K=64 QK) takes the FP8/H64 decode rows of two to four
# sparse tiles before the half wave.  Per ``(ceil(sparse_topk / 128),
# sparse_topk // 128)`` (all tiles, complete tiles), ``(num, den)``: the M64
# body from ``den * tokens > num * sm_count`` on.
_FP8_H64_M64_EARLY_WAVE = {
    "sm_107a": {
        (2, 1): (3, 20),
        (2, 2): (1, 8),
        (3, 2): (51, 200),
        (3, 3): (1, 4),
        (4, 3): (3, 10),
        (4, 4): (13, 50),
    },
}

# The same form, read in place of ``_FP8_H64_M64_EARLY_WAVE`` on the launches
# whose multi-tile M64 route runs its four-stage sibling
# (:func:`_fp8_h64_kv4_edges`); the edges scale with the SM count, measured at
# 208, 212 and 216 SMs (each entry at the SM counts named below; the comments
# give the token count each edge starts at on 208 / 212 / 216 SMs).  Eight
# distinct-input calls per CUDA graph / single cold calls, persistent body ->
# four-stage M64 body on the rows each entry moves.
# R200 (208 and 216 SMs, three boards): width 256 (13-27 tokens, q_len 1 and
# 4) 1.018-1.073x / 1.039-1.099x, width 384 (1-16 tokens, q_len 1, 2 and 4)
# 1.011-1.039x / 1.035-1.091x, widths 260-380 (52-55 tokens, q_len 1 and 4)
# 1.002-1.023x / 1.011-1.044x, width 512 (52-56 tokens, q_len 1 and 4)
# 1.023-1.050x / 1.037-1.069x, widths 896-1152 (80-104 tokens, q_len 4 only)
# 1.026-1.087x / 1.015-1.097x.  R200 (208 SMs): width 512 (50 tokens, q_len 1
# and 2) 1.021-1.024x / 1.049-1.059x, width 896 (64-77 tokens, q_len 1, 2 and
# 4) 1.031-1.054x / 1.002-1.035x, width 1024 (84-92 tokens, q_len 1, 2 and 4)
# 1.022-1.035x / 1.024-1.070x, width 1152 (77 tokens, q_len 1) 1.022x /
# 1.012x.  R200 (216 SMs) and VR200 (212 SMs), q_len 1, 2 and 4: widths
# 400-508 (61-64 tokens) 1.016-1.074x / 1.000-1.077x, width 512 (40-51 tokens)
# 1.014-1.090x / 1.003-1.123x, widths 516-636 (65-106 tokens) 1.032-1.116x /
# 1.017-1.139x, width 640 (57-106 tokens) 1.035-1.135x / 1.016-1.141x, widths
# 644-764 (60-106 tokens) 1.016-1.104x / 1.002-1.108x, width 768 (56-106
# tokens) 1.015-1.112x / 1.009-1.117x, widths 772-892 (78-106 tokens)
# 1.021-1.080x / 1.009-1.093x, width 896 (64-79 tokens) 1.024-1.070x /
# 1.000-1.066x, widths 900-1020 (79-106 tokens) 1.014-1.062x / 1.013-1.087x,
# width 1024 (83-94 tokens) 1.033-1.060x / 1.017-1.080x, widths 1028-1148
# (92-106 tokens) 1.004-1.054x / 1.022-1.082x, width 1152 (78-79 tokens,
# q_len 1 and 2) 1.025-1.034x / 1.027-1.051x.
_FP8_H64_M64_EARLY_WAVE_KV4 = {
    "sm_107a": {
        (2, 1): (3, 20),
        (2, 2): (0, 1),  # width 256: every token count
        (3, 2): (14, 57),  # widths 257-383: 52 / 53 / 54 tokens
        (3, 3): (0, 1),  # width 384: every token count
        (4, 3): (21, 73),  # widths 385-511: 60 / 61 / 63 tokens
        (4, 4): (3, 16),  # width 512: 40 / 40 / 41 tokens
        (5, 4): (11, 36),  # widths 513-639: 64 / 65 / 67 tokens
        (5, 5): (4, 15),  # width 640: 56 / 57 / 58 tokens, 32-47 below
        (6, 5): (13, 46),  # widths 641-767: 59 / 60 / 62 tokens
        (6, 6): (19, 72),  # width 768: 55 / 56 / 58 tokens
        (7, 6): (11, 30),  # widths 769-895: 77 / 78 / 80 tokens
        (7, 7): (19, 63),  # width 896: 63 / 64 / 66 tokens
        (8, 7): (10, 27),  # widths 897-1023: 78 / 79 / 81 tokens
        (8, 8): (12, 31),  # width 1024: 81 / 83 / 84 tokens
        (9, 8): (13, 30),  # widths 1025-1151: 91 / 92 / 94 tokens
        (9, 9): (19, 52),  # width 1152: 77 / 78 / 79 tokens
    },
}

# Token windows of the same launches below the uniform route's K=64 floor
# (``_K64_SIBLING_MIN_TOKENS``), where the persistent body still runs its K=32
# program: per ``(ceil(sparse_topk / 128), sparse_topk // 128)``, ``(minimum
# width, maximum width, first tokens)``, the M64 body from ``first tokens`` up
# to that floor.  R200 (216 SMs) and VR200 (212 SMs), q_len 1, 2 and 4, eight
# distinct-input calls per CUDA graph / single cold calls, persistent body ->
# four-stage M64 body: widths 260-344 at 24-47 tokens 1.007-1.130x /
# 1.024-1.149x, widths 368 and 380 at 44-47 tokens 1.056-1.110x /
# 1.047-1.133x, width 640 at 32-47 tokens 1.007-1.083x / 1.029-1.111x.  From
# 48 tokens, where the persistent body runs its K=64 sibling, the M64 body
# trailed it down to 0.979x (VR200 and R200); at width 640, 48-51 tokens,
# q_len 1 and 2, the distinct-input calls gave 0.993-1.000x (R200).
_FP8_H64_M64_BELOW_K64_KV4 = {
    "sm_107a": {
        (3, 2): ((257, 352, 24), (353, 383, 44)),
        (5, 5): ((640, 640, 32),),
    },
}


def _fp8_h64_uses_m64_body(
    num_query_tokens: int,
    sm_count: int,
    *,
    sparse_topk: Optional[int] = None,
    arch: Optional[str] = None,
    kv4: bool = False,
) -> bool:
    """FP8/H64 decode rows (``max_q_len < 257``) whose two-CTA-per-token bodies
    no longer fit one wave.

    The FP8/H128 persistent body, the SWA producer and the H64 cluster body
    launch two CTAs per token; once ``2 * tokens`` exceeds the SM count their
    second wave costs more than the single-CTA M64 body (one CTA per token)
    saves by running each token on one SM.  GB300 (152 SMs), q_len 4,
    DSv4-Flash widths: 84-120 tokens SWA-only 11.2-11.9 -> 7.0-7.5 us, topk4x
    640 14.2-15.9 -> 14.3-14.8 us; 128-512 tokens the M64 body led already.

    On SM107 the M64 body wins earlier on the rows of two to four sparse tiles
    (``_FP8_H64_M64_EARLY_WAVE``).  VR200 (208 and 216 SMs), q_len 1, 2 and 4,
    persistent body -> M64 body over the rows each tile count adds: two tiles
    5.79-7.79 -> 5.54-6.24 us, three tiles 7.33-8.76 -> 7.06-7.92 us, four
    tiles 8.76-9.54 -> 8.23-9.09 us (0.99-1.25x).  Five tiles and up stay on
    the persistent body below the half wave (M64 0.83-1.02x there).  The
    complete-tile count refines the edges (VR200, 208 SMs, two boards): two
    tiles with one complete take the M64 body from 32 tokens, where the two
    bodies meet (6.03-6.10 vs 6.05-6.10 us); three tiles with two complete
    (widths 260-380, measured at 260 and 320) keep the persistent body at 53
    tokens (M64 7.22-7.34 vs 7.08-7.20 us); and four complete tiles (width
    512) take the M64 body from 55 tokens (8.28-8.65 -> 8.11-8.25 us at 55-62
    tokens).  Launches whose multi-tile M64 route runs its four-stage sibling
    (``kv4``) take the edges of ``_FP8_H64_M64_EARLY_WAVE_KV4`` instead, and
    also the M64 body in the token windows of ``_FP8_H64_M64_BELOW_K64_KV4``
    below the uniform route's K=64 floor (``_K64_SIBLING_MIN_TOKENS``).
    """
    if 2 * num_query_tokens > sm_count:
        return True
    if sparse_topk is None:
        return False
    tiles = (_ceil_div(sparse_topk, _TILE_KV), sparse_topk // _TILE_KV)
    table = _FP8_H64_M64_EARLY_WAVE
    if kv4 and arch in _FP8_H64_M64_EARLY_WAVE_KV4:
        table = _FP8_H64_M64_EARLY_WAVE_KV4
    early = table.get(arch, {}).get(tiles)
    if early is not None and early[1] * num_query_tokens > early[0] * sm_count:
        return True
    if not kv4:
        return False
    # The uniform route's K=64 floor, read at call time (defined further down).
    k64_rows = _K64_SIBLING_MIN_TOKENS.get(arch)
    if k64_rows is None or num_query_tokens >= k64_rows:
        return False
    windows = _FP8_H64_M64_BELOW_K64_KV4.get(arch, {}).get(tiles, ())
    return any(
        min_width <= sparse_topk <= max_width and num_query_tokens >= first
        for min_width, max_width, first in windows
    )


# Mirrors the Cake M64 seed's box_k_gather_for_width (TILE_KV = 128).  The M64
# body is exported twice: single-tile items (the SWA tile is the whole item)
# run the program whose load warp gathers contiguous 16-key SWA chunks through
# one box TMA (-0.35..-0.42 us on the 128/256-token SWA rows); every wider item
# runs the program without that block, which measured at +0.03..+0.11 us of
# load-warp code layout on the multi-tile rows even when the in-kernel gate
# kept it off.  Same bits from both programs.  SM107 runs the multi-tile
# program's launches on its four-stage sibling where this process can launch
# it (:func:`_kv4_variant`), and routes the FP8/H64 decode rows by the
# four-stage edges there (:func:`_fp8_h64_kv4_edges`).
def _fp8_h64_m64_program(sparse_topk: int) -> str:
    if sparse_topk == _TILE_KV:
        return "fp8_h64_prefill_source_persistent_m64"
    return "fp8_h64_prefill_source_persistent_m64_multi_tile"


# Architectures that run the FP8/H64 source-exact row (``_route``: two ragged
# requests of up to 257 tokens over topk4x items 640 wide) on the multi-tile
# M64 body, as its dense twin: SM107 registers no ``fp8_h64_source_exact``
# program.  SM100, SM103 and ``arch=None`` keep the source-exact program.
_FP8_H64_SOURCE_EXACT_M64_ARCHES = frozenset({"sm_107a"})


# Architectures that build the H64 cluster body (``fp8_lowhead_h64``) and route
# rows to it; SM107 registers no program for it, and SM103 runs its rows on the
# persistent body (``_FP8_H64_PERSISTENT_PARTIAL_THIRD_ARCHES``).
_FP8_LOWHEAD_H64_ARCHES = frozenset({"sm_100a"})


def _fp8_h64_uses_cluster_body(
    sparse_topk: int, num_query_tokens: int, arch: Optional[str]
) -> bool:
    """FP8/H64 decode rows of at most 12 tokens with two complete sparse tiles
    and a partial third (widths 260-380), which the persistent body leaves (it
    needs three complete tiles at <= 12 tokens): the H64 cluster body (two CTAs
    per token, one wave) beats the M64 body there (1-12 tokens, widths
    260-380, both cache layouts: B200 9.5-10.3 us vs 10.0-11.1 us, GB300
    9.1-10.0 us vs 9.4-10.7 us).  Width 256 (no partial tile: B200 8.8-9.4 us
    vs 7.6-8.9 us, GB300 8.5-9.2 us vs 7.2-8.3 us) and rows with one complete
    tile keep the M64 body; so does ``arch=None`` (unknown device).  On SM103
    the persistent body leads both (:func:`_fp8_h64_uses_persistent_body`)."""
    return (
        arch in _FP8_LOWHEAD_H64_ARCHES
        and num_query_tokens <= 12
        and 2 * _TILE_KV < sparse_topk < 3 * _TILE_KV
    )


# Architectures where the 8/16/32-head SWA-only decode rows past one wave
# leave the low-head prefill producer for the single-CTA M64 body
# (``_fp8_lowhead_swa_uses_m64_body``); ``arch=None`` keeps the producer.
_FP8_LOWHEAD_M64_ARCHES = frozenset({"sm_100a", "sm_103a", "sm_107a"})

# Architectures where the 32-head decode rows wider than one producer
# partition (``_FP8_ONE_PARTITION_MAX_WIDTH``) run the FP8/H128 persistent body
# while it fits one wave (``_fp8_lowhead_wide_uses_persistent_body``); the
# others run the M64 body at every token count.
_FP8_LOWHEAD_PERSISTENT_ARCHES = frozenset({"sm_100a", "sm_103a"})


def _fp8_lowhead_wide_uses_persistent_body(
    num_heads: int, num_query_tokens: int, sm_count: int, arch: Optional[str]
) -> bool:
    """FP8 decode rows (``max_q_len < 257``) of 8, 16 or 32 heads wider than
    one producer partition that run the FP8/H128 persistent body; the rest run
    the M64 body (one CTA per token, heads past ``num_heads`` predicated).

    32 heads only: the persistent epilogue stores O in 32-row boxes from row
    ``token * num_heads``, so at 8 or 16 heads a box would cover the next
    tokens' rows.  The persistent body launches two CTAs per token and leads
    while they fit one wave; past it the M64 body runs the rows, as on the
    64-head rows (:func:`_fp8_h64_uses_m64_body`).  GB300 (152 SMs), q_len 1
    and 4, width 640 / 1152: 1-64 tokens 10.3-11.7 / 13.2-15.1 us vs M64
    13.3-14.4 / 19.7-21.2 us; 128-256 tokens M64 15.8-25.6 / 22.7-40.5 us vs
    16.8-26.6 / 22.7-38.8 us (M64 1.04-1.08x at 640, 0.96-1.00x at 1152).
    At 8 and 16 heads the M64 body runs within one wave at 0.97-1.01x of the
    low-head prefill producer at width 640 and 1.05-1.09x at 1152, and
    1.69-2.22x past it.
    """
    return (
        num_heads == 32
        and arch in _FP8_LOWHEAD_PERSISTENT_ARCHES
        and not _fp8_h64_uses_m64_body(num_query_tokens, sm_count)
    )


def _fp8_lowhead_narrow_uses_one_partition(
    num_query_tokens: int, sm_count: int, arch: Optional[str]
) -> bool:
    """FP8 decode rows (``max_q_len < 257``) of 8, 16 or 32 heads with a
    compressed cache and at most ``_FP8_ONE_PARTITION_MAX_WIDTH`` sparse slots
    that run the one-partition producer; the rest run the M64 body.

    The producer launches two CTAs per token and runs these rows while they
    fit one wave.  Past it the M64 body (one CTA per token) leads at every
    width and fill: GB300 (152 SMs), q_len 1, newest-first windows, widths
    192-384 with every slot live and width 320 with 256 live (a C128 table
    padded to the server's context), 128 and 256 tokens: 15.2-36.5 us ->
    9.2-18.3 us (1.49-2.42x) at 8, 16 and 32 heads.  Within the wave neither
    body leads everywhere (M64 0.81-1.33x of the producer, behind on the
    rows whose slots are all live).  SM107 builds no one-partition program
    and runs the M64 body at every token count.
    """
    return arch != "sm_107a" and 2 * num_query_tokens <= sm_count


def _fp8_lowhead_swa_uses_m64_body(
    num_query_tokens: int, sm_count: int, arch: Optional[str]
) -> bool:
    """FP8 SWA-only decode rows (``max_q_len < 257``) of 8, 16 or 32 heads
    whose low-head prefill producer no longer fits one wave.

    The producer launches two CTAs per token (one per work item); once
    ``2 * tokens`` exceeds the SM count its second wave costs more than the
    single-CTA M64 body (one CTA per token) saves.  R200 (sm_107a, 216 SMs),
    q_len 4, CUDA graph + PDL, per call: 112-256 tokens 6.71-10.45 us ->
    4.06-5.69 us (1.56-1.85x) at 8, 16 and 32 heads; 96 and 108 tokens (one
    wave) 0.98-1.01x, which keep the producer.  GB300 (152 SMs), q_len 1,
    newest-first windows: 77-256 tokens 10.7-21.5 us -> 7.3-10.9 us
    (1.43-1.99x) at 8, 16 and 32 heads; 64 and 76 tokens 0.92-0.97x, which
    keep the producer.  A rule of its own, kept apart from the 64-head one
    (``_fp8_h64_uses_m64_body``), and on the architectures of
    ``_FP8_LOWHEAD_M64_ARCHES`` only.
    """
    return arch in _FP8_LOWHEAD_M64_ARCHES and 2 * num_query_tokens > sm_count


def canonical_arg_name(kind: str, name: str) -> str:
    """Map a registration ``arg_plan`` entry onto the host value vocabulary."""
    if kind == "tma_buffer":
        return _TMA_SOURCE_ALIASES.get(name, name)
    if kind == "parameter":
        return _SCALAR_ALIASES.get(name, name)
    return name


def is_bindable_arg(kind: str, name: str) -> bool:
    """Whether the host can supply this ``arg_plan`` entry at all."""
    canonical = canonical_arg_name(kind, name)
    if kind in ("buffer", "tma_buffer"):
        return canonical in _TENSOR_VALUE_NAMES
    if kind == "parameter":
        return canonical in _SCALAR_VALUE_NAMES
    if kind == "grid":
        return name in _GRID_NAMES
    if kind == "launch":
        return name in _LAUNCH_NAMES
    return False


def _bind_argument(
    values: Mapping[str, Any],
    kind: str,
    name: str,
    *,
    variant: str,
    grid: Mapping[str, int],
) -> Any:
    if kind == "grid":
        if name not in grid:
            raise ValueError(
                f"CAKE DSv4 {variant} has an unknown grid argument: {name}"
            )
        return grid[name]
    if kind == "launch":
        if name not in _LAUNCH_NAMES:
            raise ValueError(
                f"CAKE DSv4 {variant} has an unknown launch argument: {name}"
            )
        return int(bool(values.get(name, False)))
    if kind not in ("buffer", "tma_buffer", "parameter"):
        raise ValueError(
            f"CAKE DSv4 {variant} has an unknown argument kind {kind!r} for {name}"
        )
    canonical = canonical_arg_name(kind, name)
    if canonical not in values:
        raise ValueError(
            f"CAKE DSv4 {variant} argument {name!r} ({kind}) has no host value; "
            f"known names: {sorted(values)}"
        )
    value = values[canonical]
    if value is None:
        raise ValueError(
            f"CAKE DSv4 {variant} argument {name!r}: it is not available for this call"
        )
    if kind == "parameter":
        if canonical in _FLOAT_SCALAR_VALUE_NAMES:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"CAKE DSv4 {variant} parameter {name!r} must be a float, got {type(value).__name__}"
                )
            return float(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"CAKE DSv4 {variant} parameter {name!r} must be an int, got {type(value).__name__}"
            )
        return int(value)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"CAKE DSv4 {variant} buffer {name!r} must be a tensor, got {type(value).__name__}"
        )
    return value


def _grid_values(grid: tuple[int, int, int]) -> dict[str, int]:
    if len(grid) != 3 or any(
        isinstance(g, bool) or not isinstance(g, int) or g < 1 for g in grid
    ):
        raise ValueError(f"launch grid must be three positive ints, got {grid!r}")
    return dict(zip(_GRID_NAMES, grid, strict=True))


class _PreparedLaunch:
    """A bound variant launch: the FFI call is the only work left to do."""

    __slots__ = ("module", "bound")

    def __init__(self, module, bound: list):
        self.module = module
        self.bound = bound

    def __call__(self):
        # Direct-source bindings use the target FFI current stream.
        return self.module.run(*self.bound)


def _sequence_module():
    from ..jit.cake_dsv4 import get_cake_dsv4_launch_sequence_module

    return get_cake_dsv4_launch_sequence_module()


def _prepare_variant(
    variant: str,
    *,
    arch: str,
    grid: tuple[int, int, int],
    values: Mapping[str, Any],
) -> _PreparedLaunch:
    """Bind the generated ABI by name through the registration ``arg_plan``.

    Two-stage routes prepare the producer and the reducer before launching
    either, so no Python-side binding sits between the two kernels.
    """
    from ..jit.cake_dsv4 import get_cake_dsv4_spec

    contract = get_cake_dsv4_spec(variant, arch=arch)
    grid_values = _grid_values(grid)
    bound = [
        _bind_argument(values, kind, name, variant=variant, grid=grid_values)
        for kind, name in contract["arg_plan"]
    ]
    return _PreparedLaunch(_variant_module(variant, arch=arch), bound)


def _launch_variant(
    variant: str,
    *,
    arch: str,
    grid: tuple[int, int, int],
    values: Mapping[str, Any],
):
    """Prepare and launch one variant (single-launch convenience)."""
    return _prepare_variant(variant, arch=arch, grid=grid, values=values)()


# --------------------------------------------------------------------------- #
# Routing                                                                     #
# --------------------------------------------------------------------------- #


def _route(
    *,
    dtype: torch.dtype,
    num_heads: int,
    max_q_len: int,
    ragged: bool,
    sparse_topk: int,
    batch_size: int,
    compressed_page_size: int,
    num_query_tokens: int,
    sm_count: int,
    arch: Optional[str] = None,
    kv4: bool = False,
) -> str:
    """The variant one call runs; ``sm_count`` is the device's SM count (the
    FP8/H64 decode rows, the FP8 32-head decode rows wider than 384 on SM100
    and SM103, and on SM107 the FP8 8/16/32-head SWA-only decode rows, pick
    their body by the waves of its grid) and ``arch`` its
    ``sm_XYa`` target (``None``: the architecture-independent choice, which
    may name a program SM107 does not build: it has no BF16 programs, no FP8
    source-exact program and no FP8 low-head cluster or one-partition
    program).  ``kv4``: the call's multi-tile M64 launches run the four-stage
    sibling (:func:`_fp8_h64_kv4_edges`); only the FP8/H64 decode rows read
    it."""
    if max_q_len <= 0:
        raise ValueError("max_q_len must be positive")
    is_swa = sparse_topk == 128
    is_topk4x = not is_swa and compressed_page_size == 64
    is_topk128x = not is_swa and compressed_page_size == 2
    if dtype == torch.float8_e4m3fn:
        if sparse_topk > _FP8_MAX_WIDTH:
            # The backend='cake' NVFP4 route takes up to _NVFP4_MAX_SPLITS
            # tiles of 128 slots per token; only rows within it are sent there,
            # and only on the architectures that register NVFP4 programs
            # (``arch=None``: the architecture-independent choice).
            from ..jit.cake_dsv4 import _NVFP4_IDENTITIES

            nvfp4_max_width = _NVFP4_MAX_SPLITS * _TILE_KV
            if sparse_topk <= nvfp4_max_width and (
                arch is None or arch in _NVFP4_IDENTITIES
            ):
                elsewhere = (
                    "Run these rows with backend='trtllm-gen' on this cache, or on "
                    "an NVFP4 KV cache (kv_cache_format='nvfp4'), whose "
                    f"backend='cake' route takes up to {nvfp4_max_width} slots per "
                    "token."
                )
            else:
                elsewhere = "Run these rows with backend='trtllm-gen'."
            raise ValueError(
                f"backend='cake' has no FP8 kernel for sparse_topk {sparse_topk}: "
                f"the FP8 DSv4 kernels take at most {_FP8_MAX_WIDTH} sparse slots "
                "per token (128 SWA + 1024 compressed). A C128 layer is wider past "
                "128K context (128 + context / 128 slots, 2176 at 256K). " + elsewhere
            )
        if num_heads == 128:
            # One producer for every FP8/H128 row: the persistent body (one
            # work feed over T x 2 CTAs, runtime q_lens) matched or beat the
            # former per-(token, split) cluster producer on the 12-token
            # decode rows and beat trtllm-gen 1.12-1.75x on the MTP rows.
            return _fp8_persistent_program(
                num_query_tokens,
                max_q_len=max_q_len,
                topk4x=is_topk4x,
                arch=arch,
                swa=is_swa,
            )
        if num_heads not in (8, 16, 32, 64):
            raise ValueError(f"unsupported CAKE FP8 DSv4 head count: {num_heads}")
        if (
            num_heads == 64
            and batch_size == 2
            and max_q_len == 257
            and ragged
            and is_topk4x
            and sparse_topk == 640
            and arch not in _FP8_H64_SOURCE_EXACT_M64_ARCHES
        ):
            # The source-exact body where it is built; SM107 runs the row on
            # the M64 body below (_FP8_H64_SOURCE_EXACT_M64_ARCHES).
            return "fp8_h64_source_exact"
        if max_q_len >= 257:
            # The single-CTA M64 persistent body (num_heads runtime, heads
            # past num_heads predicated) beats the low-head prefill producer
            # on every prefill row of the DSv4 contract.  GB300: 2.1-5.2x on
            # the 16/32/64-head rows of 512-8192 tokens (widths 128-1024,
            # single and dual caches), 2.1-4.7x on the 8-head rows of
            # 512-8192 tokens (widths 128, 640 and 1152; two requests at
            # 8192 tokens, width 1152: 2240 -> 909 us).  R200
            # (sm_107a, 216 SMs), CUDA graph + PDL, 8 heads at (tokens x
            # width) 257 x 128 to 2048 x 1152: 1.87-4.66x.  Same oracle.  Two
            # programs, selected by the item width.
            return _fp8_h64_m64_program(sparse_topk)
        if (
            num_heads in (8, 16)
            and batch_size == 3
            and max_q_len == 5
            and ragged
            and (
                (is_topk128x and sparse_topk == 260)
                or (is_topk4x and sparse_topk == (192 if num_heads == 8 else 256))
            )
        ):
            # FP8 port of the 1-CTA SwapsAb body trtllm-gen runs
            # on these rows (heads on the MMA N side, kind::f8f6f4, P e4m3
            # x448); same shape lock as the BF16 source-exact route.  Paired
            # vs trtllm-gen: rows 55/58/67/70 GB300 1.27-1.30x / B200
            # 1.24-1.25x, width-260 rows 56/59/68/71 GB300 1.38-1.49x /
            # B200 1.33-1.43x (one-partition producer 0.99-1.27x).
            return "fp8_h8_h16_source_exact"
        if num_heads == 64:
            if _fp8_h64_uses_m64_body(
                num_query_tokens, sm_count, sparse_topk=sparse_topk, arch=arch, kv4=kv4
            ):
                # H64-specific single-CTA M64 persistent body (one CTA per
                # token, unified 128-row KV stage, no V gathers).  Two exported
                # programs, selected by the item width.
                return _fp8_h64_m64_program(sparse_topk)
            if _fp8_h64_uses_persistent_body(
                sparse_topk, num_query_tokens, arch, kv4=kv4
            ):
                # Same persistent FP8 body as FP8/H128 (heads >= num_heads
                # predicated) for the multi-tile rows of one wave: GB300 10.5-12.0
                # us vs 13.4-14.1 us for the M64 body on the 1-72-token topk4x
                # 640 rows.
                return _fp8_persistent_program(num_query_tokens)
            if is_swa:
                return "fp8_lowhead_prefill"
            if _fp8_h64_uses_cluster_body(sparse_topk, num_query_tokens, arch):
                return "fp8_lowhead_h64"
            # Rows with one complete tile (and width 256): the M64 body beats
            # the H64 cluster body at every token count (GB300, topk128x widths
            # 204-252, 1-72 tokens: 7.9-8.5 us vs 8.8-10.4 us).  On SM107 the
            # rows of _FP8_H64_PERSISTENT_ONE_TILE_MIN_WIDTH took the persistent
            # body above; with kv4, width 176 and widths 257-352 at 13-16 tokens
            # land here (_FP8_H64_PERSISTENT_SHORT_MIN_WIDTH_KV4), while the
            # windows of _FP8_H64_M64_BELOW_K64_KV4 took the M64 body above.
            return _fp8_h64_m64_program(sparse_topk)
        if is_swa:
            if _fp8_lowhead_swa_uses_m64_body(num_query_tokens, sm_count, arch):
                # 8/16/32-head SWA-only rows past one wave of the low-head
                # prefill producer: the single-tile M64 program (width 128).
                return _fp8_h64_m64_program(sparse_topk)
            return "fp8_lowhead_prefill"
        # One producer partition owns up to three sparse tiles (widths up to
        # 384) and writes final O directly (one partition measured 1.18-1.26x
        # on the width-260 rows); past one wave the M64 body runs them.
        if sparse_topk <= _FP8_ONE_PARTITION_MAX_WIDTH:
            if _fp8_lowhead_narrow_uses_one_partition(num_query_tokens, sm_count, arch):
                return "fp8_lowhead_one_partition"
            return _fp8_h64_m64_program(sparse_topk)
        # Wider rows (C4 at 640 or 1152, C128 past 32K context) run the
        # persistent or the M64 body.
        if _fp8_lowhead_wide_uses_persistent_body(
            num_heads, num_query_tokens, sm_count, arch
        ):
            return _fp8_persistent_program(num_query_tokens)
        return _fp8_h64_m64_program(sparse_topk)
    if dtype != torch.bfloat16:
        raise ValueError(f"unsupported CAKE DSv4 dtype: {dtype}")
    if (
        num_heads in (8, 16)
        and batch_size == 3
        and max_q_len == 5
        and ragged
        and (
            (is_topk128x and sparse_topk == 260)
            or (is_topk4x and sparse_topk == (192 if num_heads == 8 else 256))
        )
    ):
        return "bf16_h8_h16_source_exact"
    if num_heads in (8, 16):
        if is_swa:
            return "bf16_h8_swa128_v43" if num_heads == 8 else "bf16_h16_h32_swa128_v44"
        raise ValueError(
            f"backend='cake' has no BF16 kernel for {num_heads} heads with a "
            "compressed cache outside the batch-3, max_q_len-5 ragged rows"
        )
    if num_heads == 32:
        if is_swa:
            return "bf16_h16_h32_swa128_v44"
        if is_topk4x or is_topk128x:
            # The retained-KV body with the last-arriver merge
            # beats the topk4x body on the H32 topk4x rows on both targets.
            return "bf16_h32_topk128x_early_v47"
        raise ValueError("BF16 H32 compressed cache requires page size 64 or 2")
    if num_heads == 64:
        # Dense Q with equal q_lens is the same row memory as ragged Q: the
        # not-ragged rows follow the ragged rules (the portfolio producer beats
        # the former fixed-Q guard program on the sparse decode rows, GB300
        # 1.03x -> 1.21x; the fixed-Q programs are no longer exported).
        if max_q_len >= 257:
            return "bf16_h64_prefill"
        if is_swa:
            return (
                "bf16_h64_guard_q_tma_batch_r25"
                if max_q_len > 5
                else "bf16_swa128_single_cta"
            )
        if (
            num_query_tokens >= _BF16_H64_COMPRESSED_PREFILL_TOKENS
            and sparse_topk <= _BF16_H64_PREFILL_MAX_SPARSE_WIDTH
        ):
            # The KV-reuse prefill body (one Q64 CTA per token, full V) beats
            # the one-tile-per-split portfolio producer from 24 tokens on.
            return "bf16_h64_prefill"
        return "bf16_h64_compressed_q8_v38"
    if num_heads == 128:
        if max_q_len >= 257:
            return "bf16_h128_prefill_v42"
        if (is_swa or is_topk4x) and num_query_tokens >= _BF16_H128_PREFILL_MIN_TOKENS:
            # Persistent KV-reuse prefill body for the many-token MTP rows
            # (Cake rule BF16_H128_PREFILL_MIN_TOKENS): the dedicated SWA
            # producer repeats QK per V chunk and the per-token owners lose
            # from 64 tokens on.
            return "bf16_h128_prefill_v42"
        if is_swa:
            return "bf16_h128_swa128"
        if is_topk4x and sparse_topk == 1152:
            return "bf16_h128_topk4x_v52"
        if _BF16_TOPK128X_MIN_WIDTH < sparse_topk <= _BF16_TOPK128X_MAX_WIDTH:
            return "bf16_h128_topk128x"
        raise ValueError(
            f"backend='cake' has no BF16 H128 kernel for sparse_topk {sparse_topk} "
            f"below {_BF16_H128_PREFILL_MIN_TOKENS} tokens (supported: 128, "
            f"{_BF16_TOPK128X_MIN_WIDTH + 4}-{_BF16_TOPK128X_MAX_WIDTH} and 1152 "
            "with page size 64)"
        )
    raise ValueError(f"unsupported CAKE BF16 DSv4 head count: {num_heads}")


# --------------------------------------------------------------------------- #
# Route workspace plan                                                        #
# --------------------------------------------------------------------------- #

# Routes whose producers write the final output directly and keep no LSE:
# they never touch the workspace.
_NO_WORKSPACE_ROUTES = frozenset(
    {
        "bf16_swa128_single_cta",
        "bf16_h128_swa128",
        "bf16_h8_swa128_v43",
        "bf16_h16_h32_swa128_v44",
        "bf16_h8_h16_source_exact",
        "fp8_h8_h16_source_exact",
        "bf16_h64_guard_q_tma_batch_r25",
        "bf16_h64_prefill",
        "fp8_h64_source_exact",
    }
)
# Routes that launch one producer with ``partials(1)``: final O written
# directly, the LSE region of the workspace only.
_ONE_SPLIT_ROUTES = frozenset(
    {
        "bf16_h128_prefill_v42",
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h64_prefill_source_persistent_m64_multi_tile",
        "fp8_lowhead_one_partition",
        "fp8_lowhead_h64",
        "fp8_lowhead_prefill",
    }
)


@dataclass(frozen=True)
class _RoutePlan:
    """Workspace use of one route at one shape: the single source for the
    dispatcher's ``partials`` / ``counters`` calls and for the sizing API."""

    # Partial buffers per (token, head); 1 = LSE region only, no partial_O.
    num_splits: int
    # False: the route binds no partial buffers and needs no workspace bytes.
    uses_partials: bool
    # Split-merge counters per metadata row (0: no in-kernel merge).
    merge_groups_per_row: int


def _route_plan(
    route: str, *, num_query_tokens: int, num_heads: int, sparse_topk: int
) -> _RoutePlan:
    if route == "bf16_h64_compressed_q8_v38":
        # One KV tile per split plus the H64 reducer (single split: direct O).
        return _RoutePlan(_ceil_div(sparse_topk, _TILE_KV), True, 0)
    if route == "bf16_h32_topk128x_early_v47":
        # One KV tile per split; the last-arriving split of each (token, head
        # tile) merges in-kernel through one counter per group.
        return _RoutePlan(
            _ceil_div(sparse_topk, _TILE_KV), True, _ceil_div(num_heads, 8)
        )
    if route == "bf16_h128_topk4x_v52":
        # Five fixed full-V KV owners per token plus the split-5 reducer.
        return _RoutePlan(5, True, 0)
    if route == "bf16_h128_topk128x":
        # Four disjoint full-V owners plus the LSE reducer up to the token
        # bound; one row-first owner per token above it.
        return _RoutePlan(
            4 if num_query_tokens <= _BF16_TOPK128X_SPLIT_MAX_TOKENS else 1, True, 0
        )
    if route in _ONE_SPLIT_ROUTES:
        return _RoutePlan(1, True, 0)
    if route in _NO_WORKSPACE_ROUTES:
        return _RoutePlan(1, False, 0)
    raise RuntimeError(f"unhandled CAKE DSv4 route: {route}")


def _launch_workspace_bytes(
    plan: _RoutePlan, rows: int, num_heads: int, num_query_offsets: int = 0
) -> int:
    """Workspace bytes one launch of ``plan`` over ``rows`` metadata rows carves."""
    if not plan.uses_partials:
        return 0
    return cake_dsv4_workspace_layout(
        rows, num_heads, plan.num_splits, num_query_offsets=num_query_offsets
    ).total_bytes


def _launch_fits(
    plan: _RoutePlan,
    rows: int,
    num_heads: int,
    num_query_offsets: int,
    workspace_bytes: int,
) -> bool:
    if (
        plan.merge_groups_per_row
        and rows * plan.merge_groups_per_row > _MAX_MERGE_GROUPS
    ):
        return False
    return (
        _launch_workspace_bytes(plan, rows, num_heads, num_query_offsets)
        <= workspace_bytes
    )


def _rows_per_launch(
    plan: _RoutePlan,
    *,
    num_query_tokens: int,
    num_heads: int,
    batch_size: int,
    workspace_bytes: int,
) -> int:
    """Metadata rows per launch for a workspace of ``workspace_bytes``.

    All rows when one launch fits (no row tiling, no offsets region);
    otherwise the largest chunk whose carve, including the shifted request
    offsets (``batch_size + 1`` int32) of the chunks after the first, fits;
    0 when not even one row fits.
    """
    total = num_query_tokens
    if _launch_fits(plan, total, num_heads, 0, workspace_bytes):
        return total
    offsets = batch_size + 1
    per_row = (
        num_heads
        * plan.num_splits
        * (
            (_HEAD_DIM * torch.bfloat16.itemsize if plan.num_splits > 1 else 0)
            + torch.float32.itemsize
        )
    )
    rows = (
        workspace_bytes
        - _PARTIAL_OFFSET
        - _align_up(offsets * torch.int32.itemsize)
        - 2 * _ALIGN
    ) // per_row
    if plan.merge_groups_per_row:
        rows = min(rows, _MAX_MERGE_GROUPS // plan.merge_groups_per_row)
    rows = max(0, min(rows, total - 1))
    while rows > 0 and not _launch_fits(
        plan, rows, num_heads, offsets, workspace_bytes
    ):
        rows -= 1
    while rows + 1 < total and _launch_fits(
        plan, rows + 1, num_heads, offsets, workspace_bytes
    ):
        rows += 1
    return rows


@dataclass(frozen=True)
class WorkspaceRequirement:
    """Exact workspace numbers of one ``backend="cake"`` call.

    ``single_launch_bytes`` is what one launch over all metadata rows carves
    (0 for routes without partial buffers; at the shapes of the SM107 split
    rules, the split launch's carve, while ``num_splits`` stays the route's
    own count, 1); ``minimum_bytes`` is the smallest
    workspace the call accepts (one row per launch, row tiling); any size in
    between works with :meth:`rows_per_launch` rows per launch.
    """

    route: str
    num_splits: int
    uses_workspace: bool
    single_launch_bytes: int
    minimum_bytes: int
    num_query_tokens: int
    num_heads: int
    batch_size: int
    plan: _RoutePlan = field(repr=False)

    def rows_per_launch(self, workspace_bytes: int) -> int:
        """Metadata rows per launch with ``workspace_bytes`` (0: too small)."""
        return _rows_per_launch(
            self.plan,
            num_query_tokens=self.num_query_tokens,
            num_heads=self.num_heads,
            batch_size=self.batch_size,
            workspace_bytes=int(workspace_bytes),
        )


def cake_dsv4_workspace_requirement(
    *,
    dtype: torch.dtype,
    num_heads: int,
    num_query_tokens: int,
    sparse_topk: int,
    compressed_page_size: int,
    max_q_len: int,
    batch_size: int,
    ragged: bool,
    sm_count: Optional[int] = None,
    arch: Optional[str] = None,
    kv4: Optional[bool] = None,
) -> WorkspaceRequirement:
    """Route and exact workspace requirement of one call.

    The arguments are the route inputs of :func:`run_cake_dsv4`:
    ``num_query_tokens`` = metadata rows, ``sparse_topk`` = combined table
    width, ``compressed_page_size`` = ``compressed_kv_cache.shape[-2]``
    (irrelevant for SWA-only calls), ``max_q_len`` = dense per-request query
    length or the ragged maximum, ``batch_size`` = ``seq_lens.numel()``,
    ``ragged`` = ``cum_seq_lens_q is not None``, ``sm_count`` = the SM count
    of the device the call runs on (``None``: the current CUDA device's; the
    workspace numbers do not depend on it, the route name may), ``arch`` = its
    ``sm_XYa`` target (``None``: the current CUDA device's whenever CUDA is
    available, else the architecture-independent route, which may name a
    program SM107 does not build; the workspace numbers do not depend on it
    either, except as noted below), ``kv4`` = whether the call's multi-tile
    M64 launches run the four-stage sibling (:func:`_fp8_h64_kv4_edges`;
    ``None``: the current CUDA device's answer when ``arch`` is ``None`` and
    CUDA is available, else ``False``; the workspace numbers do not depend on
    it either). Raises
    ``ValueError`` for a shape ``backend="cake"`` has no kernel for.
    ``single_launch_bytes`` covers the SM107 split-KV launch of the shape
    (:func:`_k64_split_plan`, with or without PDL) on every architecture.
    Two exceptions to the independence above.  Below 69 SMs some 64-head
    split shapes take an M64 route, which never splits, so their
    ``single_launch_bytes`` can depend on ``sm_count``, ``arch`` and ``kv4``.
    The FP8/H64 source-exact row (two ragged requests, ``max_q_len`` 257,
    topk4x items 640 wide) takes no workspace where its source-exact program
    runs (SM100, SM103 and the architecture-independent route) and the LSE
    region of one split on SM107, which runs it on the M64 route
    (``_FP8_H64_SOURCE_EXACT_M64_ARCHES``); size it with ``arch="sm_107a"``
    or :func:`get_cake_dsv4_workspace_bytes`.
    """
    device_arch = arch is None
    if sm_count is None:
        sm_count = _device_sm_count(torch.device("cuda", torch.cuda.current_device()))
    if arch is None and torch.cuda.is_available():
        arch = _target_arch(torch.device("cuda", torch.cuda.current_device()))
    tokens = _positive_int(num_query_tokens, "num_query_tokens")
    heads = _positive_int(num_heads, "num_heads")
    batch = _positive_int(batch_size, "batch_size")
    topk = _positive_int(sparse_topk, "sparse_topk")
    q_len = _positive_int(max_q_len, "max_q_len")
    if kv4 is None:
        kv4 = (
            device_arch
            and arch is not None
            and _fp8_h64_kv4_edges(
                dtype=dtype,
                num_heads=heads,
                max_q_len=q_len,
                sparse_topk=topk,
                arch=arch,
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        )
    route = _route(
        dtype=dtype,
        num_heads=heads,
        max_q_len=q_len,
        ragged=bool(ragged),
        sparse_topk=topk,
        batch_size=batch,
        compressed_page_size=int(compressed_page_size),
        num_query_tokens=tokens,
        sm_count=_positive_int(sm_count, "sm_count"),
        arch=arch,
        kv4=bool(kv4),
    )
    plan = _route_plan(
        route, num_query_tokens=tokens, num_heads=heads, sparse_topk=topk
    )
    single = _launch_workspace_bytes(plan, tokens, heads, 0)
    minimum = min(single, _launch_workspace_bytes(plan, 1, heads, batch + 1))
    # A split-KV launch of this shape carves more (any architecture, with or
    # without PDL, so the number stays independent of both); a smaller
    # workspace runs the route's own launch.
    single = max(
        single,
        _k64_split_workspace_bytes(
            route, num_heads=heads, num_query_tokens=tokens, sparse_topk=topk
        ),
    )
    return WorkspaceRequirement(
        route=route,
        num_splits=plan.num_splits,
        uses_workspace=plan.uses_partials,
        single_launch_bytes=single,
        minimum_bytes=minimum,
        num_query_tokens=tokens,
        num_heads=heads,
        batch_size=batch,
        plan=plan,
    )


# --------------------------------------------------------------------------- #
# Output placement                                                            #
# --------------------------------------------------------------------------- #
_OUT_PHASE_PERIOD = 4096

# Output placement: on the routes below the kernel time is a 4 KiB-periodic
# function of the output buffer's base address (bits 0..11 only; bits >= 12 do
# nothing).  ``base % 4096 == 0`` -- what a fresh >= 2 MiB caching-allocator
# block gives, i.e. the default ``torch.empty`` for these outputs -- is the SLOW
# phase; ``0x800`` is the fast one.  Pinned-phase paired measurements (cold-L2
# CUPTI, both arm orders) on the exported programs, us at phase 0 ->
# phase 0x800, B200 / GB300:
#   bf16_h128_prefill_v42 (every bf16 H128 persistent row, striped and snake):
#     hardening-000035 40.13 -> 36.96 / 35.74 -> 33.25, hardening-000023 17.98 ->
#     16.70 / 16.58 -> 15.42, hardening-000029 26.78 -> 24.77 / 24.16 -> 22.94,
#     prefill-style-000088 90.66 -> 88.10 / 74.88 -> 72.48, prefill-style-000092
#     94.56 -> 92.26 / 78.21 -> 76.10 (-2.3 .. -8 %).
#   fp8_lowhead_prefill, 64 heads, SWA-128 table (the single-CTA K1 body):
#     decode-000006 7.07 -> 6.72 / 6.75 -> 6.43, decode-000009 7.07 -> 6.72 /
#     6.66 -> 6.30, hardening-000020 7.84 -> 7.49 / 7.55 -> 7.20 (-4.5 .. -5.3 %);
#     32 heads move +-0.1 us with a process-dependent sign and 8 / 16 heads do
#     not move, so they keep the allocator default.
# The kernels are untouched: the same program writes the same bits to the same
# (token, head, column) positions; only the buffer's base address is chosen.
# Applies to ``out=None`` only -- a caller-provided ``out`` is used as is, and a
# 2 MiB-aligned caller buffer sits in the slow phase.
_OUT_PHASE_BY_ROUTE: dict[str, int] = {
    "bf16_h128_prefill_v42": 0x800,
    "fp8_lowhead_prefill": 0x800,
    # 134-row public-API pass, allocator default -> 0x800, both arm orders:
    #   bf16_h64_prefill (086/087/090/091) -0.70..-1.50 us GB300,
    #   fp8_h64_source_exact -1.09..-1.18 us GB300, fp8_lowhead_h64 (008/011) -0.34 us GB300.
    "bf16_h64_prefill": 0x800,
    "fp8_h64_source_exact": 0x800,
    "fp8_lowhead_h64": 0x800,
    #   bf16_h128_topk128x: hardening-000025 / -000031 -0.74 / -0.70 us GB300, the
    #   other rows of the route flat (|d| <= 0.03 us).
    "bf16_h128_topk128x": 0x800,
}


# Alignment of O that each route's program needs beyond the 2 bytes a
# contiguous bf16 view guarantees, the widest O store of its kernels over every
# architecture that runs the route: 32 = ``st.global.v8.b32`` at 16-element
# offsets, 16 = ``int4`` stores at 8-element offsets or an O tensor map (TMA
# store; its binding rejects an address off 16 bytes).  bf16_h128_topk4x_v52
# writes O from its split-5 reducer (``int4`` on sm_103a, ``int2`` on sm_100a);
# bf16_h128_topk128x from the split reducer (``int2``) or, above 16 tokens, the
# row-first owners (``st.global.v8.b32``); bf16_h64_compressed_q8_v38 from the
# H64 reducer (``int2``) or, with one split, the producer
# (``st.global.v8.b32``).  A caller ``out`` off that alignment (a contiguous
# view at +2 or +16 bytes) runs into an aligned staging buffer that is then
# copied into ``out``: the same program, the same bits.  An aligned ``out``
# (every allocation, every phase of ``_OUT_PHASE_BY_ROUTE``) takes no staging.
_OUT_ALIGN_BY_ROUTE: dict[str, int] = {
    "bf16_h128_prefill_v42": 32,
    "bf16_h128_swa128": 16,
    "bf16_h128_topk128x": 32,
    "bf16_h128_topk4x_v52": 16,
    "bf16_h16_h32_swa128_v44": 16,
    "bf16_h32_topk128x_early_v47": 16,
    "bf16_h64_compressed_q8_v38": 32,
    "bf16_h64_guard_q_tma_batch_r25": 16,
    "bf16_h64_prefill": 32,
    "bf16_h8_h16_source_exact": 16,
    "bf16_h8_swa128_v43": 16,
    "bf16_swa128_single_cta": 16,
    "fp8_h128_prefill_source_persistent": 16,
    "fp8_h128_prefill_source_persistent_uniform": 16,
    "fp8_h64_prefill_source_persistent_m64": 32,
    "fp8_h64_prefill_source_persistent_m64_multi_tile": 32,
    "fp8_h64_source_exact": 32,
    "fp8_h8_h16_source_exact": 16,
    "fp8_lowhead_h64": 32,
    "fp8_lowhead_one_partition": 32,
    "fp8_lowhead_prefill": 32,
}


def cake_dsv4_out_phase(
    route: str, *, num_heads: int, sparse_topk: int
) -> Optional[int]:
    """Preferred ``out.data_ptr() % 4096`` for ``route`` (None = allocator default)."""
    phase = _OUT_PHASE_BY_ROUTE.get(route)
    if phase is None:
        return None
    if route == "fp8_lowhead_prefill" and not (num_heads == 64 and sparse_topk == 128):
        return None
    return phase


def allocate_cake_dsv4_output(
    shape: tuple[int, ...], device: torch.device, *, phase: Optional[int]
) -> torch.Tensor:
    """bf16 output of ``shape``; with ``phase`` the view's ``data_ptr() % 4096 == phase``
    (one extra 4 KiB page is allocated), with None a plain ``torch.empty``."""
    if phase is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=device)
    if phase % 2 or not 0 <= phase < _OUT_PHASE_PERIOD:
        raise ValueError(
            f"output phase must be an even byte offset below {_OUT_PHASE_PERIOD}, got {phase}"
        )
    numel = 1
    for dim in shape:
        numel *= int(dim)
    raw = torch.empty(
        numel + _OUT_PHASE_PERIOD // 2, dtype=torch.bfloat16, device=device
    )
    start = ((phase - raw.data_ptr()) % _OUT_PHASE_PERIOD) // 2
    return raw[start : start + numel].view(shape)


# --------------------------------------------------------------------------- #
# Launch orchestration                                                        #
# --------------------------------------------------------------------------- #


def _device_scale(
    value: Union[float, torch.Tensor], *, device: torch.device, name: str
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.numel() != 1:
            raise ValueError(f"{name} must be a one-element FP32 tensor")
        if value.device != device:
            raise ValueError(f"{name} must be on {device}, got {value.device}")
        if not value.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        return value

    device_index = device.index
    if device_index is None and device.type == "cuda":
        device_index = torch.cuda.current_device()
    key = (device.type, device_index, float(value))
    with _scale_cache_lock:
        result = _scale_cache.get(key)
        if result is None:
            # Process-lifetime constant; not a per-call allocation.
            result = torch.tensor([key[2]], dtype=torch.float32, device=device)
            _scale_cache[key] = result
    return result


def _dense_query_offsets(
    batch_size: int, q_len: int, *, device: torch.device
) -> torch.Tensor:
    """``cum_seq_lens_q`` for a dense ``[batch, q_len]`` query (process-lifetime constant).

    Variants that only accept ragged queries read the request boundaries
    from ``cum_seq_lens_q``; a dense query is the ragged query with every
    request ``q_len`` long. The offsets are cached per (device, batch,
    q_len) so no call path allocates and CUDA-graph replay sees a stable
    pointer.
    """
    device_index = device.index
    if device_index is None and device.type == "cuda":
        device_index = torch.cuda.current_device()
    key = (device.type, device_index, int(batch_size), int(q_len))
    with _scale_cache_lock:
        result = _dense_offsets_cache.get(key)
        if result is None:
            result = torch.arange(
                0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=device
            )
            _dense_offsets_cache[key] = result
    return result


def _dense_rows(cache: torch.Tensor, name: str, dtype: torch.dtype) -> torch.Tensor:
    if cache.dtype != dtype:
        raise ValueError(
            f"{name} dtype must match the query dtype {dtype}, got {cache.dtype}"
        )
    if cache.shape[-1] != _HEAD_DIM:
        raise ValueError(
            f"{name} must have head dim {_HEAD_DIM}, got {cache.shape[-1]}"
        )
    flat = cache.reshape(-1, _HEAD_DIM)
    if flat.data_ptr() != cache.data_ptr() or not flat.is_contiguous():
        raise ValueError(
            f"{name} must be a densely packed [..., {_HEAD_DIM}] pool; backend='cake' "
            "makes no host copy, so strided page layouts are not supported"
        )
    return flat


def _int32_vector(
    tensor: torch.Tensor, name: str, device: torch.device
) -> torch.Tensor:
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    # A 1-D vector is bound as the caller's own object (reshape would wrap it
    # in a new view object even when nothing changes); higher-rank inputs
    # flatten without a copy or are rejected.
    flat = tensor if tensor.dim() == 1 else tensor.reshape(-1)
    if flat.data_ptr() != tensor.data_ptr() or not flat.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous; backend='cake' makes no host copy"
        )
    return flat


# The routes whose programs leave PDL to each call and may route a PDL-free
# twin ``<route>_no_pdl``: the same kernel without its two griddepcontrol
# instructions, which a call that does not ask for PDL launches instead
# (without the attribute the grid starts after its predecessor completes, so
# it reads what the PDL program reads after its wait).  SM100 and SM103 route
# the twin to programs of their own.
_NO_PDL_TWIN_ROUTES = frozenset({"fp8_h128_prefill_source_persistent"})

# The fewest query rows a launch needs to take the twin, per architecture.
# GB300 without PDL: the twin beat the PDL program by 64-192 ns from 386 rows
# on (2 x 257-token prefill, 64 x 8 MTP) but trailed it by up to 64 ns at 256
# rows (64 x 4 MTP).
_NO_PDL_TWIN_MIN_TOKENS = {"sm_103a": 384}

# The fewest query tokens the call's longest request needs for its launches to
# take the twin, per architecture.  B200 without PDL: the twin led the PDL
# program by 32-128 ns (median 64 ns) on 2 x 257-token prefill launches but
# tied or trailed it by up to 32 ns on 2- to 8-token MTP launches.
_NO_PDL_TWIN_MIN_QUERY_LEN = {"sm_100a": 257}


# The routes with a K=64 sibling ``<route>_k64`` on an architecture whose FP8
# MMA runs at K=64 only as a one-CTA M128 tile (SM107): a program whose every
# MMA is that tile, where the route's pair-M128 MMAs run at K=32, with the
# route's arguments, grid and launch.
_K64_SIBLING_ROUTES = frozenset({"fp8_h128_prefill_source_persistent_uniform"})

# The fewest query rows a launch needs to take the K=64 sibling, per
# architecture.  VR200, eight back-to-back calls per CUDA graph: the sibling
# led the K=32 program from 48 rows on (FP8/H128 and FP8/H64, widths 128-1152)
# and trailed it by up to 4 % below.
_K64_SIBLING_MIN_TOKENS = {"sm_107a": 48}

# The dynamic shared memory of the K=64 sibling programs whose bindings launch
# them in CUDA 13.4's oversized shared-memory mode (above the 232448 B standard
# opt-in ceiling of SM107), per architecture.  The binding sets that mode
# through the CUDA runtime its JIT module links (``libcudart`` by soname, so
# the copy already in the process, e.g. torch's runtime wheel); a runtime
# older than 13.4 rejects it ("cudaFuncSetAttribute failed: invalid argument")
# even on a driver that supports it, so those launches run the program they
# would run without the sibling instead (:func:`_k64_sibling_launchable`,
# :func:`_k64_variant`).
_K64_SIBLING_OVERSIZED_SMEM_BYTES = {"sm_107a": 272512}

# The routes with a four-stage sibling ``<route>_kv4`` on an architecture whose
# oversized shared-memory mode holds two more 64 KiB K/V stages (SM107): the
# multi-tile M64 program with a four-stage K/V ring in that mode, where the
# route's program has two stages, with the route's arguments, grid and launch
# and the same output bits.  R200 (208 SMs), eight back-to-back calls per CUDA
# graph and single cold calls: the four-stage program led the two-stage one by
# 0.4-28.8 % on every multi-tile row but one and tied or trailed it by up to
# 2.3 % on that one, 64-head 2 x 257-token prefill on topk128x items 256 wide.
# Where the sibling launches, SM107 routes the FP8/H64 decode rows by its
# edges (:func:`_fp8_h64_kv4_edges`, ``_FP8_H64_M64_EARLY_WAVE_KV4``,
# ``_FP8_H64_M64_BELOW_K64_KV4``).
_KV4_SIBLING_ROUTES = frozenset({"fp8_h64_prefill_source_persistent_m64_multi_tile"})

# The dynamic shared memory of the four-stage sibling programs, per
# architecture (the route's two-stage program takes 191616 B).  Their bindings
# launch them in the oversized mode, as the K=64 siblings' bindings do, so a
# process that cannot launch them runs the route's program instead
# (:func:`_kv4_sibling_launchable`, :func:`_kv4_variant`).
_KV4_SIBLING_OVERSIZED_SMEM_BYTES = {"sm_107a": 322688}

# The TMEM-batch sibling ``<route>_kv4_bcl`` of the four-stage sibling on
# SM107: the four-stage program whose correction warps rescale each output
# stage in two 64-column TMEM load / multiply / store passes in place of eight
# 16-column steps and whose softmax waits once for its two max-reduction loads,
# with the four-stage program's arguments, grid, launch and shared memory and
# the same output bits.  It takes the four-stage launches of one window and
# every other launch keeps the four-stage program (:func:`_kv4_bcl_variant`):
# 64-head MTP decode launches of calls whose longest request has 2-4 query
# tokens, of at least 108 rows and at most two rows per SM of the device (432
# at 216 SMs, 424 at 212, 416 at 208), over items of at most six tiles.  R200
# (216 SMs) and VR200 (212 SMs), eight calls per CUDA graph over distinct
# inputs behind an L2 flush / over one input, four-stage program -> TMEM-batch
# program: q_len 2-4, widths 384-768, 108 rows to two per SM 0.996-1.011x /
# 1.000-1.012x, no row slower beyond the spread of two runs of the same
# program.  The window leaves out the launches on which the TMEM-batch program
# trailed: one-token requests, 96-row launches over items 768 wide, items 1152
# wide and 257- to 1024-token prefill.  The decode windows
# (``_KV4_BCL_DECODE_WINDOWS``) take back one-token launches of one layer,
# width and row range where the TMEM-batch program measured faster.
_KV4_BCL_HEADS = 64  # query heads
_KV4_BCL_MIN_Q = 2  # query tokens of the call's longest request
_KV4_BCL_MAX_Q = 4
_KV4_BCL_MIN_TOKENS = 108  # query rows of the launch
_KV4_BCL_MAX_TOKENS_PER_SM = 2  # query rows of the launch per SM of the device
_KV4_BCL_MAX_TOPK = 768  # item width (sparse_topk), six tiles

# The TMEM-batch sibling's decode windows: 64-head launches of calls whose
# ``(max_q_len, sparse_topk, compressed_page_size)`` names a window, of its
# rows and at most one row per SM, on a device of an SM count in
# ``_KV4_BCL_DECODE_SM_COUNTS`` (:func:`_kv4_bcl_variant`).  One window: the
# one-token decode of the c4 layer (compressed pages of 64 rows) over items
# 1152 wide, 176-208 rows.  R200 (216 SMs) and VR200 (212 SMs), two
# processes each, eight calls per CUDA graph over distinct inputs behind an
# L2 flush, four-stage program -> TMEM-batch program at every row count from
# 176 to 208: 0.007-0.133 us faster per call in every run, 19 of the 33 row
# counts beyond the spread of two runs of the same program in all four
# runs, no row slower.  The window leaves out the c128 layer (compressed
# pages of 2 rows), 144- and 160-row and 256- to 416-row launches (slower
# there) and devices of other SM counts (not measured; on 208 SMs a 208-row
# launch fills the device).
_KV4_BCL_DECODE_WINDOWS: dict[tuple[int, int, int], tuple[int, int]] = {
    (1, 1152, 64): (176, 208),
}
_KV4_BCL_DECODE_SM_COUNTS = frozenset({212, 216})

# The last-tile L2 prefetch sibling ``<route>_kv4_l2pf`` of the four-stage
# sibling on SM107: the four-stage program whose index warp, in a launch of at
# most one row per SM, prefetches into L2 the compressed-KV pool rows of each
# item's last tile (items of at least four tiles) after the item's last index
# pass, with the four-stage program's grid, launch and shared memory, the
# compressed-KV pool and its row count as two more arguments, and the same
# output bits.  It takes the four-stage launches of 64-head calls over
# compressed pages of 64 rows (the c4 layer) whose ``(sparse_topk,
# max_q_len)`` names a row ceiling in ``_KV4_L2PF_MAX_TOKENS``, of at most
# that many rows and at most one row per SM, on a device of an SM count in
# ``_KV4_L2PF_SM_COUNTS``; a launch of 4-token requests also has to be
# uniform (``num_query_tokens == batch_size * max_q_len``).  Every other
# launch keeps the four-stage program (:func:`_kv4_l2pf_variant`).  R200
# (216 SMs) and VR200 (212 SMs), two processes each, eight calls per CUDA
# graph over distinct inputs behind an L2 flush, four-stage program -> L2
# prefetch program, per row and call: items 640 wide, one-token requests,
# 32-47 rows 0.160-0.330 us faster; four-token requests, 32-44 and 60-64
# rows 0.060-0.328 us; items 512 wide, one-token requests at 40-41 rows
# 0.090-0.176 us, four-token requests at 40-56 rows 0.028-0.250 us.  The
# ceilings stop below the first measured row count that was not faster.
# The rows of these widths and request lengths that the host does not send
# to the four-stage program (48-57 rows over items 640 wide, for example)
# never reach it.
_KV4_L2PF_HEADS = 64  # query heads
_KV4_L2PF_PAGE = 64  # compressed page rows (the c4 layer)
_KV4_L2PF_MAX_TOKENS: dict[tuple[int, int], int] = {
    # (sparse_topk, max_q_len): most query rows of the launch
    (640, 1): 47,
    (640, 4): 64,
    (512, 1): 41,
    (512, 4): 56,
}
_KV4_L2PF_SM_COUNTS = frozenset({212, 216})

# The routes with an early-PV sibling ``<route>_overlap`` on SM107: the route's
# program whose P x V MMAs start once the softmax has written the low half of
# P and whose epilogues drain the output of the first V block while the MMA of
# the second one runs, with the route's arguments, grid and launch and the
# same output bits.  It takes the launches of at most the row ceiling below;
# the larger launches keep the route's program.
_OVERLAP_SIBLING_ROUTES = frozenset({"fp8_lowhead_prefill"})

# The most query rows a launch may have to take the early-PV sibling, per
# architecture, and the SM count of the part that measured it.  A part with
# fewer SMs scales the ceiling down with its SM count, which keeps the
# measured share of busy SMs (two CTAs per row); a part with more SMs keeps
# the measured ceiling (:func:`_overlap_variant`).
_OVERLAP_SIBLING_MAX_TOKENS = {"sm_107a": 64}
_OVERLAP_SIBLING_MEASURED_SMS = {"sm_107a": 216}

# The routes with a V-block sibling ``<route>_vblock`` on SM107: the early-PV
# sibling's program run with four work items per row instead of two, one
# 128-column V block of the output each on its own CTA, with the route's
# arguments and the same output bits.  Each CTA drains one V block instead of
# two, and the launch takes four CTAs per row.  It takes the launches of at
# most the row ceiling below; the larger launches keep the early-PV sibling or
# the route's program (:func:`_vblock_variant`).
_VBLOCK_SIBLING_ROUTES = frozenset({"fp8_lowhead_prefill"})

# The most query rows a launch may have to take the V-block sibling, per
# architecture, and the SM count of the part the ceiling is set for: at most
# half of its SMs busy at four CTAs per row.  A part with fewer SMs scales the
# ceiling down with its SM count; a part with more SMs keeps the ceiling.
_VBLOCK_SIBLING_MAX_TOKENS = {"sm_107a": 27}
_VBLOCK_SIBLING_MEASURED_SMS = {"sm_107a": 216}

# The work items, and CTAs, per query row of a V-block sibling launch.
_VBLOCK_ITEMS_PER_TOKEN = 4

# The CUDA toolkit and runtime (``cudaRuntimeGetVersion``) version that added
# the oversized shared-memory mode, and the driver's device attribute for its
# ceiling (CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK).
_OVERSIZED_SMEM_CUDA_VERSION = "13.4"
_OVERSIZED_SMEM_CUDART_VERSION = 13040
_CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK = 150


def _shared_library(soname: str) -> Optional[ctypes.CDLL]:
    """The process's copy of shared library ``soname``: the one already mapped
    (the copy a JIT module linking ``soname`` binds to), else the one the
    dynamic linker finds (JIT modules loaded later bind to it), else ``None``."""
    for mode in (os.RTLD_NOLOAD | os.RTLD_NOW, ctypes.DEFAULT_MODE):
        try:
            return ctypes.CDLL(soname, mode=mode)
        except OSError:
            continue
    return None


def _cuda_runtime_version(major: int) -> Optional[int]:
    """``cudaRuntimeGetVersion`` of ``libcudart.so.<major>``, the runtime the
    JIT modules link (``-lcudart``, no rpath), or ``None`` when that library
    does not load or the call fails."""
    cudart = _shared_library(f"libcudart.so.{major}")
    if cudart is None:
        return None
    version = ctypes.c_int(0)
    if cudart.cudaRuntimeGetVersion(ctypes.pointer(version)) != 0:
        return None
    return version.value


def _device_oversized_smem_per_block(device: torch.device) -> Optional[int]:
    """The oversized shared-memory ceiling per block of ``device``, read from
    the driver (a runtime older than 13.4 does not know the attribute), or
    ``None`` when the driver does not report it (one older than 13.4 has no
    oversized mode)."""
    cuda = _shared_library("libcuda.so.1")
    if cuda is None:
        return None
    ordinal = torch.cuda.current_device() if device.index is None else device.index
    handle = ctypes.c_int(0)
    value = ctypes.c_int(0)
    if cuda.cuInit(0) != 0 or cuda.cuDeviceGet(ctypes.pointer(handle), ordinal) != 0:
        return None
    attribute = _CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK
    if cuda.cuDeviceGetAttribute(ctypes.pointer(value), attribute, handle) != 0:
        return None
    return value.value


@functools.lru_cache(maxsize=None)
def _oversized_smem_launchable(smem_bytes: int, device: torch.device) -> bool:
    """Whether this process can launch a program of ``smem_bytes`` dynamic
    shared memory in the oversized mode on ``device``: it needs a CUDA toolkit
    of at least 13.4 (the program's binding does not build below it), a CUDA
    runtime of at least 13.4 in the process (an older one rejects the mode) and
    a driver that reports an oversized ceiling holding ``smem_bytes`` (one that
    reports none has no mode)."""
    from ..jit.cpp_ext import get_cuda_version, is_cuda_version_at_least

    if not is_cuda_version_at_least(_OVERSIZED_SMEM_CUDA_VERSION):
        return False
    runtime = _cuda_runtime_version(get_cuda_version().major)
    if runtime is None or runtime < _OVERSIZED_SMEM_CUDART_VERSION:
        return False
    ceiling = _device_oversized_smem_per_block(device)
    return ceiling is not None and ceiling >= smem_bytes


def _k64_sibling_launchable(arch: str, device: torch.device) -> bool:
    """Whether this process can launch ``arch``'s K=64 sibling on ``device``.

    A sibling within the standard opt-in ceiling always can; one in the
    oversized mode can when :func:`_oversized_smem_launchable` passes for its
    shared memory."""
    smem_bytes = _K64_SIBLING_OVERSIZED_SMEM_BYTES.get(arch)
    return smem_bytes is None or _oversized_smem_launchable(smem_bytes, device)


def _kv4_sibling_launchable(arch: str, device: torch.device) -> bool:
    """Whether this process can launch ``arch``'s four-stage sibling on
    ``device``, as :func:`_k64_sibling_launchable` for the K=64 sibling."""
    smem_bytes = _KV4_SIBLING_OVERSIZED_SMEM_BYTES.get(arch)
    return smem_bytes is None or _oversized_smem_launchable(smem_bytes, device)


def _k64_variant(
    route: str, *, arch: str, num_query_tokens: int, device: torch.device
) -> str:
    """The variant a launch of ``route`` over ``num_query_tokens`` rows on
    ``device`` runs: its K=64 sibling when ``arch`` has a floor in
    ``_K64_SIBLING_MIN_TOKENS``, the launch meets it, ``arch`` routes the
    sibling to a program other than the route's and this process can launch
    that program on ``device`` (:func:`_k64_sibling_launchable`), else
    ``route``.  When only that launch check fails, launches of at least
    ``_FP8_PERSISTENT_LANE_GATHER_MIN_TOKENS`` rows (the short-query launches
    SM107 keeps on the uniform route only for the sibling) run the lane-issued
    route ``fp8_h128_prefill_source_persistent`` instead."""
    if route not in _K64_SIBLING_ROUTES:
        return route
    min_tokens = _K64_SIBLING_MIN_TOKENS.get(arch)
    if min_tokens is None or num_query_tokens < min_tokens:
        return route
    from ..jit.cake_dsv4 import VARIANTS

    programs = VARIANTS.get(arch, {})
    sibling = f"{route}_k64"
    if programs.get(sibling, programs.get(route)) == programs.get(route):
        return route
    if not _k64_sibling_launchable(arch, device):
        if num_query_tokens >= _FP8_PERSISTENT_LANE_GATHER_MIN_TOKENS:
            return "fp8_h128_prefill_source_persistent"
        return route
    return sibling


def _kv4_variant(route: str, *, arch: str, device: torch.device) -> str:
    """The variant a launch of ``route`` on ``device`` runs: its four-stage
    sibling when ``route`` is in ``_KV4_SIBLING_ROUTES``, ``arch`` routes the
    sibling to a program other than the route's and this process can launch
    that program on ``device`` (:func:`_kv4_sibling_launchable`), else
    ``route``."""
    if route not in _KV4_SIBLING_ROUTES:
        return route
    from ..jit.cake_dsv4 import VARIANTS

    programs = VARIANTS.get(arch, {})
    sibling = f"{route}_kv4"
    if programs.get(sibling, programs.get(route)) == programs.get(route):
        return route
    return sibling if _kv4_sibling_launchable(arch, device) else route


def _fp8_h64_kv4_edges(
    *,
    dtype: torch.dtype,
    num_heads: int,
    max_q_len: int,
    sparse_topk: int,
    arch: Optional[str],
    device: torch.device,
) -> bool:
    """Whether :func:`_route` takes the four-stage-sibling edges for this call:
    an FP8/H64 decode call wider than one tile whose multi-tile M64 launches
    run the sibling on ``device`` (the predicate :func:`_kv4_variant` applies
    at launch).  Other calls do not reach the launch check."""
    route = "fp8_h64_prefill_source_persistent_m64_multi_tile"
    return (
        dtype == torch.float8_e4m3fn
        and num_heads == 64
        and max_q_len < 257
        and sparse_topk > _TILE_KV
        and arch is not None
        and _kv4_variant(route, arch=arch, device=device) != route
    )


def _kv4_bcl_variant(
    variant: str,
    *,
    arch: str,
    num_heads: int,
    max_q_len: int,
    num_query_tokens: int,
    sparse_topk: int,
    device: torch.device,
    compressed_page_size: Optional[int] = None,
) -> str:
    """The variant a launch over ``num_query_tokens`` rows of a call whose
    longest request has ``max_q_len`` tokens runs once :func:`_kv4_variant`
    chose ``variant``: its TMEM-batch sibling ``<variant>_bcl`` when
    ``variant`` is the four-stage sibling of a route in
    ``_KV4_SIBLING_ROUTES``, the call has ``_KV4_BCL_HEADS`` heads, ``arch``
    routes the TMEM-batch sibling to a program other than the four-stage one
    and the launch is in one of the sibling's windows, else ``variant``.  The
    MTP window: ``max_q_len`` is from ``_KV4_BCL_MIN_Q`` to
    ``_KV4_BCL_MAX_Q``, the items are at most ``_KV4_BCL_MAX_TOPK`` wide and
    the launch has from ``_KV4_BCL_MIN_TOKENS`` rows to
    ``_KV4_BCL_MAX_TOKENS_PER_SM`` rows per SM of ``device``.  A decode
    window: ``_KV4_BCL_DECODE_WINDOWS`` names the call's ``(max_q_len,
    sparse_topk, compressed_page_size)``, the launch has the window's rows and
    at most one row per SM of ``device``, and ``device`` has an SM count in
    ``_KV4_BCL_DECODE_SM_COUNTS``; a launch of an unknown
    ``compressed_page_size`` (``None``) takes no decode window.  The
    four-stage launch check :func:`_kv4_variant` applied covers the
    TMEM-batch sibling, which takes the same shared memory."""
    route = variant.removesuffix("_kv4")
    if route == variant or route not in _KV4_SIBLING_ROUTES:
        return variant
    if num_heads != _KV4_BCL_HEADS:
        return variant
    decode: Optional[tuple[int, int]] = None
    if compressed_page_size is not None:
        decode = _KV4_BCL_DECODE_WINDOWS.get(
            (max_q_len, sparse_topk, compressed_page_size)
        )
    if decode is None:
        if not _KV4_BCL_MIN_Q <= max_q_len <= _KV4_BCL_MAX_Q:
            return variant
        if sparse_topk > _KV4_BCL_MAX_TOPK or num_query_tokens < _KV4_BCL_MIN_TOKENS:
            return variant
    elif not decode[0] <= num_query_tokens <= decode[1]:
        return variant
    from ..jit.cake_dsv4 import VARIANTS

    programs = VARIANTS.get(arch, {})
    sibling = f"{variant}_bcl"
    if programs.get(sibling, programs.get(variant)) == programs.get(variant):
        return variant
    sm_count = _device_sm_count(device)
    if decode is None:
        ceiling = _KV4_BCL_MAX_TOKENS_PER_SM * sm_count
    elif sm_count in _KV4_BCL_DECODE_SM_COUNTS:
        ceiling = sm_count
    else:
        return variant
    if num_query_tokens > ceiling:
        return variant
    return sibling


def _kv4_l2pf_variant(
    variant: str,
    *,
    arch: str,
    num_heads: int,
    max_q_len: int,
    num_query_tokens: int,
    sparse_topk: int,
    device: torch.device,
    batch_size: Optional[int] = None,
    compressed_page_size: Optional[int] = None,
) -> str:
    """The variant a launch over ``num_query_tokens`` rows of a call of
    ``batch_size`` requests whose longest has ``max_q_len`` tokens runs once
    :func:`_kv4_bcl_variant` chose ``variant``: its last-tile L2 prefetch
    sibling ``<variant>_l2pf`` when ``variant`` is the four-stage sibling of a
    route in ``_KV4_SIBLING_ROUTES`` (not its TMEM-batch sibling), the call
    has ``_KV4_L2PF_HEADS`` heads and compressed pages of ``_KV4_L2PF_PAGE``
    rows, ``_KV4_L2PF_MAX_TOKENS`` gives ``(sparse_topk, max_q_len)`` a row
    ceiling the launch does not pass, a launch of requests of more than one
    token is uniform (``num_query_tokens == batch_size * max_q_len``),
    ``arch`` routes the sibling to a program other than the four-stage one,
    and ``device`` has an SM count in ``_KV4_L2PF_SM_COUNTS`` and at least
    one SM per row, else ``variant``.  A launch of an unknown
    ``compressed_page_size`` or, past one-token requests, ``batch_size``
    (``None``) keeps ``variant``.  The four-stage launch check
    :func:`_kv4_variant` applied covers the sibling, which takes the same
    shared memory."""
    route = variant.removesuffix("_kv4")
    if route == variant or route not in _KV4_SIBLING_ROUTES:
        return variant
    if num_heads != _KV4_L2PF_HEADS or compressed_page_size != _KV4_L2PF_PAGE:
        return variant
    ceiling = _KV4_L2PF_MAX_TOKENS.get((sparse_topk, max_q_len))
    if ceiling is None or num_query_tokens > ceiling:
        return variant
    if max_q_len > 1 and (
        batch_size is None or num_query_tokens != batch_size * max_q_len
    ):
        return variant
    from ..jit.cake_dsv4 import VARIANTS

    programs = VARIANTS.get(arch, {})
    sibling = f"{variant}_l2pf"
    if programs.get(sibling, programs.get(variant)) == programs.get(variant):
        return variant
    sm_count = _device_sm_count(device)
    if sm_count not in _KV4_L2PF_SM_COUNTS or num_query_tokens > sm_count:
        return variant
    return sibling


def _overlap_variant(
    route: str, *, arch: str, num_query_tokens: int, device: torch.device
) -> str:
    """The variant a launch of ``route`` over ``num_query_tokens`` rows on
    ``device`` runs: its early-PV sibling when ``route`` is in
    ``_OVERLAP_SIBLING_ROUTES``, ``arch`` has a ceiling in
    ``_OVERLAP_SIBLING_MAX_TOKENS``, the launch has at most that many rows
    scaled by the device's SM count up to ``_OVERLAP_SIBLING_MEASURED_SMS``
    (64 rows at 216 SMs, 62 at 212, 61 at 208) and ``arch`` routes the sibling
    to a program other than the route's, else ``route``."""
    if route not in _OVERLAP_SIBLING_ROUTES:
        return route
    max_tokens = _OVERLAP_SIBLING_MAX_TOKENS.get(arch)
    if max_tokens is None:
        return route
    measured_sms = _OVERLAP_SIBLING_MEASURED_SMS[arch]
    sm_count = min(_device_sm_count(device), measured_sms)
    if num_query_tokens > max_tokens * sm_count // measured_sms:
        return route
    from ..jit.cake_dsv4 import VARIANTS

    programs = VARIANTS.get(arch, {})
    sibling = f"{route}_overlap"
    if programs.get(sibling, programs.get(route)) == programs.get(route):
        return route
    return sibling


def _vblock_variant(
    route: str, *, arch: str, num_query_tokens: int, device: torch.device
) -> str:
    """The variant a launch of ``route`` over ``num_query_tokens`` rows on
    ``device`` runs before the early-PV sibling is considered: its V-block
    sibling when ``route`` is in ``_VBLOCK_SIBLING_ROUTES``, ``arch`` has a
    ceiling in ``_VBLOCK_SIBLING_MAX_TOKENS``, the launch has at most that many
    rows scaled by the device's SM count up to ``_VBLOCK_SIBLING_MEASURED_SMS``
    (27 rows at 216 SMs, 26 at 212 and at 208) and ``arch`` routes the sibling
    to a program other than the route's and the early-PV sibling's, else
    ``route``.  The V-block sibling launches ``_VBLOCK_ITEMS_PER_TOKEN`` work
    items, and CTAs, per row."""
    if route not in _VBLOCK_SIBLING_ROUTES:
        return route
    max_tokens = _VBLOCK_SIBLING_MAX_TOKENS.get(arch)
    if max_tokens is None:
        return route
    measured_sms = _VBLOCK_SIBLING_MEASURED_SMS[arch]
    sm_count = min(_device_sm_count(device), measured_sms)
    if num_query_tokens > max_tokens * sm_count // measured_sms:
        return route
    from ..jit.cake_dsv4 import VARIANTS

    programs = VARIANTS.get(arch, {})
    program = programs.get(f"{route}_vblock")
    if program is None or program in (
        programs.get(route),
        programs.get(f"{route}_overlap"),
    ):
        return route
    return f"{route}_vblock"


def _pdl_variant(
    route: str,
    *,
    arch: str,
    enable_pdl: bool,
    num_query_tokens: int,
    max_q_len: int,
) -> str:
    """The variant a launch of ``route`` over ``num_query_tokens`` rows of a
    call whose longest request has ``max_q_len`` tokens runs: its PDL-free twin
    when the call does not ask for PDL, ``arch`` has a floor in
    ``_NO_PDL_TWIN_MIN_TOKENS`` or ``_NO_PDL_TWIN_MIN_QUERY_LEN``, the launch
    meets each floor ``arch`` has and ``arch`` routes the twin to a program
    other than the route's, else ``route``."""
    if enable_pdl or route not in _NO_PDL_TWIN_ROUTES:
        return route
    min_tokens = _NO_PDL_TWIN_MIN_TOKENS.get(arch)
    min_q_len = _NO_PDL_TWIN_MIN_QUERY_LEN.get(arch)
    if min_tokens is None and min_q_len is None:
        return route
    if num_query_tokens < (min_tokens or 0) or max_q_len < (min_q_len or 0):
        return route
    from ..jit.cake_dsv4 import VARIANTS

    programs = VARIANTS.get(arch, {})
    twin = f"{route}_no_pdl"
    if programs.get(twin, programs.get(route)) == programs.get(route):
        return route
    return twin


# SM107 split-KV launches of the FP8/H128 uniform-gather route.  A launch of
# few query rows over wide items leaves most SMs idle with one two-CTA cluster
# per row; the split form of the K=64 program (``_K64_SPLIT_VARIANT``) runs
# each row as ``s`` pieces of its sparse tiles, two CTAs per piece, and writes
# partial O and LSE (module docstring, "Split-KV pieces"); the merge program
# (``_K64_SPLIT_MERGE``, one CTA per row and head, any ``s``) writes O from
# them, or, for the 128-head launches whose rule's ``merge_by_splits`` names
# a merge built for their ``s`` pieces alone at their rows, that merge
# (``_K64_SPLIT_COUNT_MERGES``, :func:`_k64_split_merge`).  Both launch back
# to back in one FFI call with the call's PDL setting.
_K64_SPLIT_ROUTE = "fp8_h128_prefill_source_persistent_uniform"
_K64_SPLIT_VARIANT = "fp8_h128_prefill_source_persistent_uniform_k64_split"
# A launch splits only where the registry routes both keys to programs of
# their own (:func:`_k64_split`).
_K64_SPLIT_MERGE = "fp8_h128_prefill_source_persistent_uniform_k64_split_reduce"
_K64_SPLIT_HEADS = 128
# The merges built for one piece count s, each with the heads one of its CTAs
# serves: ``_c{s}`` is the copy of ``_K64_SPLIT_MERGE`` for s pieces alone
# (one CTA per row and head, the same per-element operations; the merge of
# any count holds the copies for 1-9 pieces and picks one at run time) and
# ``_s3`` the three-piece merge with one warp per row and head (two heads per
# CTA).  A 128-head launch of s pieces takes the one its rule's
# ``merge_by_splits`` names for s at its rows (:func:`_k64_split_merge`).
_K64_SPLIT_COUNT_MERGES: Mapping[str, int] = {
    f"{_K64_SPLIT_MERGE}_c9": 1,
    f"{_K64_SPLIT_MERGE}_c5": 1,
    f"{_K64_SPLIT_MERGE}_s3": 2,
}


@dataclass(frozen=True)
class _K64SplitRule:
    """The launches of ``_K64_SPLIT_ROUTE`` that split on one architecture."""

    # Item width in sparse tiles, ``max(ceil(sparse_topk / 128), 1)``.
    num_tiles: int
    # The most query rows a split launch has with and without PDL (0: none).
    max_tokens_pdl: int
    max_tokens_no_pdl: int
    # ``(s, most pieces)`` in order of preference: a launch takes the first
    # ``s`` pieces per row with ``s`` at most ``num_tiles`` and rows x ``s``
    # at most that count's ``most pieces``.
    splits: tuple[tuple[int, int], ...]
    # ``(s, fewest rows, merge)``: a launch of ``s`` pieces per row merges
    # with the ``merge`` (``_K64_SPLIT_MERGE`` or a key of
    # ``_K64_SPLIT_COUNT_MERGES``) of the entry for ``s`` with the most
    # ``fewest rows`` at most its rows, and with ``_K64_SPLIT_MERGE`` where no
    # entry is (128 heads only).
    merge_by_splits: tuple[tuple[int, int, str], ...] = ()


# Per architecture.  R200 (sm_107a, 208 SMs), eight back-to-back calls per
# CUDA graph with PDL, items 1152 wide (nine tiles) whose rows use all of
# them, split launch vs the K=32 program: 1.50-1.51x at 1-2 rows and 1.43x
# at 4 (nine pieces), 1.34x at 8 (five).  Nine pieces at 5 rows (45 pieces)
# measured 0.97x of five, so nine and five stay at 40 pieces.  Three pieces
# run every launch of 9-20 rows (at most 20 x 3 = 60 pieces, so no launch
# takes two).  With the merge of any count, three pieces measured 1.23x of
# the K=32 program at 12 rows and 1.11-1.12x at 15, but 0.91-0.92x of two
# pieces at 16 rows (48 pieces).  The launches of 16-20 rows merge with
# the three-piece merge (``merge_by_splits``: one warp per row and head, two
# heads per CTA, the per-element operations of the merge of any count): in a
# test harness on a 216-SM R200, three pieces with a merge of the three-piece
# merge's geometry ran a median 0.80-0.93 us per call faster than two pieces
# with the merge of any count at 16 and 17 rows, and at least 0.3 us faster
# in each of 8 rounds.  At 18-20 rows, three pieces with the three-piece
# merge ran 0.84-1.07 us per call faster than the K=32 program (1.09-1.11x)
# in a test harness on a 216-SM R200 (CUDA graph of 8 calls, PDL, rows that
# use all nine tiles), 1.08-1.13x on a 208-SM R200 and 1.05-1.11x on a
# 212-SM VR200, faster beyond the A/A difference in every round on each.
# The launches of 9-15 rows do not take the three-piece
# merge: in a test harness on a 212-SM VR200, a 15-row batch of mixed row
# lengths ran slower with it in every round.  The launches of nine and five
# pieces merge with the merge built for their count alone
# (``merge_by_splits``: ``_c9``, ``_c5``), the merge of any count's code for
# that count, so the output bits do not change: in a test harness on a
# 216-SM R200 (CUDA graph of 8 calls, PDL) the launches of 4, 7 and 8 rows
# ran 0.12-0.37 us per call faster with it (1.01-1.05x), faster beyond the
# A/A difference in each of 8 rounds, and the launches of 1-3, 5 and 6 rows
# differed by at most 0.06 us; on a 208-SM R200 and a 212-SM VR200 no launch
# of 1-8 rows ran slower beyond the A/A difference.  The three-piece launches
# of 9-15 rows keep the merge of any count: the merge built for three pieces
# alone ran slower than it on a 208-SM R200 and on a 212-SM VR200.  With the
# merge of any count, a launch whose every row fits in one piece (at most
# ceil(9 / s) tiles: short requests, piece 0 writes O itself) still runs the
# merge kernel and measured 0.81-0.86x (0.97x of two pieces at 15 rows of at
# most three tiles); the host does not read row lengths, so those launches
# split as well.  At 18-20 rows, with the three-piece merge, such a launch
# measured 0.78-0.81x (1.57-1.83 us per call slower) and one whose every row
# uses at most five tiles 0.86-0.89x (0.97-1.33 us slower) on those three
# boards.  Without PDL the 17-row launch measured 0.98x with two pieces:
# no launch without PDL splits.
_K64_SPLIT_RULES: Mapping[str, _K64SplitRule] = {
    "sm_107a": _K64SplitRule(
        num_tiles=9,
        max_tokens_pdl=20,
        max_tokens_no_pdl=0,
        splits=((9, 40), (5, 40), (3, 60)),
        merge_by_splits=(
            (9, 1, f"{_K64_SPLIT_MERGE}_c9"),
            (5, 5, f"{_K64_SPLIT_MERGE}_c5"),
            (3, 16, f"{_K64_SPLIT_MERGE}_s3"),
        ),
    ),
}

# 64-head launches of the same route, per architecture: one rule per item
# width in tiles.  VR200 (sm_107a, 212 SMs), eight back-to-back calls per
# CUDA graph with PDL over one input set and over eight, L2 flushed before
# each replay, q_len 1 and 4, items n x 128 wide whose rows use every tile,
# split launch vs the K=32 program, the lowest at each row count over four
# rounds, both input protocols and every measured shape, at 1-5, 8, 9,
# 12-14, 16-18, 20 and 24 rows (1, 4, 8 and 16 at four tiles): four tiles
# (width 512) 1.05-1.18x at 1 and 4 rows, five (640) 1.03-1.22x up to 9
# rows, six (768) 1.02-1.34x up to 18, seven (896) 1.02-1.41x up to 17,
# eight (1024) 1.04-1.51x up to 20 and nine (1152) 1.10-1.63x up to 20; at
# the next measured row count a split gains under 2 % or loses.  Three tiles
# have no rule: where the four-stage M64 body runs them, a split lost
# against it from 4 rows.  A launch whose every row fits in one piece
# measured 0.83-0.97x (1152 and 640 wide).  The rules follow the 128-head
# policy: those launches split as well, and no launch without PDL splits.
_K64_SPLIT_HEADS_H64 = 64
_K64_SPLIT_RULES_H64: Mapping[str, tuple[_K64SplitRule, ...]] = {
    "sm_107a": (
        # num_tiles, max_tokens_pdl, max_tokens_no_pdl, splits.  Rows 1-4 in
        # four pieces; 1-9 in five; 1-9 in six, 10-18 in three; 1-9 in seven,
        # 10-17 in four; 1-7 in eight, 8-15 in four, 16-20 in three; 1-6 in
        # nine, 7-12 in five, 13-20 in three.
        _K64SplitRule(4, 4, 0, ((4, 16),)),
        _K64SplitRule(5, 9, 0, ((5, 45),)),
        _K64SplitRule(6, 18, 0, ((6, 54), (3, 54))),
        _K64SplitRule(7, 17, 0, ((7, 63), (4, 68))),
        _K64SplitRule(8, 20, 0, ((8, 56), (4, 60), (3, 60))),
        _K64SplitRule(9, 20, 0, ((9, 54), (5, 60), (3, 60))),
    ),
}


def _k64_split_plan(
    route: str,
    *,
    arch: str,
    num_heads: int,
    num_query_tokens: int,
    sparse_topk: int,
    enable_pdl: bool,
) -> Optional[tuple[int, int, int]]:
    """The split of a launch of ``route`` over ``num_query_tokens`` rows by
    ``arch``'s rule in ``_K64_SPLIT_RULES`` (128 heads) or its rule for the
    item width in ``_K64_SPLIT_RULES_H64`` (64 heads): ``(W, R, s)`` with ``W = 0``
    whole rows, ``R = num_query_tokens`` split rows and ``s`` pieces per row,
    or ``None`` (the route's own launch).  A function of the launch shape
    alone; :func:`_k64_split` adds the registry, workspace and device checks.
    """
    rule = _K64_SPLIT_RULES.get(arch)
    if num_heads == _K64_SPLIT_HEADS_H64:
        tiles = max(_ceil_div(sparse_topk, _TILE_KV), 1)
        rules = _K64_SPLIT_RULES_H64.get(arch, ())
        rule = next((r for r in rules if r.num_tiles == tiles), None)
    elif num_heads != _K64_SPLIT_HEADS:
        rule = None
    if rule is None or route != _K64_SPLIT_ROUTE:
        return None
    if max(_ceil_div(sparse_topk, _TILE_KV), 1) != rule.num_tiles:
        return None
    max_tokens = rule.max_tokens_pdl if enable_pdl else rule.max_tokens_no_pdl
    if not 1 <= num_query_tokens <= max_tokens:
        return None
    for splits, max_pieces in rule.splits:
        if splits <= rule.num_tiles and num_query_tokens * splits <= max_pieces:
            return (0, num_query_tokens, splits)
    return None


def _k64_split_merge(
    *, arch: str, num_heads: int, rows: int, num_splits: int
) -> tuple[str, int]:
    """The merge variant of a split launch and its heads per CTA: the
    per-count merge the 128-head rule of ``arch`` names for ``num_splits``
    pieces and ``rows`` rows (``merge_by_splits``) where the registry routes
    it to a program of its own (not the route's, not ``_K64_SPLIT_MERGE``'s),
    else ``_K64_SPLIT_MERGE`` (one CTA per row and head).  A per-count merge
    runs its own count of pieces only, and the three-piece merge serves two
    heads per CTA with no head bound: each must launch with its count and
    over ``H`` divided by its heads per CTA, which a registry entry pointing
    its key at either of those programs (a retired key) never does."""
    key, fewest = _K64_SPLIT_MERGE, 0
    if num_heads == _K64_SPLIT_HEADS and arch in _K64_SPLIT_RULES:
        for s, at_least, merge in _K64_SPLIT_RULES[arch].merge_by_splits:
            if s == num_splits and fewest < at_least <= rows:
                key, fewest = merge, at_least
    heads_per_cta = _K64_SPLIT_COUNT_MERGES.get(key)
    if heads_per_cta is not None:
        from ..jit.cake_dsv4 import VARIANTS

        programs = VARIANTS.get(arch, {})
        own = programs.get(key)
        if own is not None and own not in (
            programs.get(_K64_SPLIT_ROUTE),
            programs.get(_K64_SPLIT_MERGE),
        ):
            return key, heads_per_cta
    return _K64_SPLIT_MERGE, 1


def _k64_split_workspace_bytes(
    route: str, *, num_heads: int, num_query_tokens: int, sparse_topk: int
) -> int:
    """The largest workspace carve of a split launch of this shape on any
    architecture, with or without PDL (0: such a launch never splits)."""
    carve = 0
    for arch in {*_K64_SPLIT_RULES, *_K64_SPLIT_RULES_H64}:
        for enable_pdl in (False, True):
            split = _k64_split_plan(
                route,
                arch=arch,
                num_heads=num_heads,
                num_query_tokens=num_query_tokens,
                sparse_topk=sparse_topk,
                enable_pdl=enable_pdl,
            )
            if split is not None:
                _, rows, splits = split
                layout = cake_dsv4_workspace_layout(rows, num_heads, splits)
                carve = max(carve, layout.total_bytes)
    return carve


def _k64_split(
    route: str,
    *,
    arch: str,
    num_heads: int,
    num_query_tokens: int,
    sparse_topk: int,
    enable_pdl: bool,
    workspace_bytes: int,
    device: torch.device,
) -> Optional[tuple[int, int, int]]:
    """The split a launch over all ``num_query_tokens`` rows runs: the plan of
    :func:`_k64_split_plan` when ``workspace_bytes`` hold its carve, ``arch``
    routes ``_K64_SPLIT_VARIANT`` and ``_K64_SPLIT_MERGE`` to programs other
    than the route's and this process can launch the K=64 programs on
    ``device`` (:func:`_k64_sibling_launchable`), else ``None`` (the route's
    own launch, unchanged)."""
    split = _k64_split_plan(
        route,
        arch=arch,
        num_heads=num_heads,
        num_query_tokens=num_query_tokens,
        sparse_topk=sparse_topk,
        enable_pdl=enable_pdl,
    )
    if split is None:
        return None
    _, rows, splits = split
    if (
        workspace_bytes
        < cake_dsv4_workspace_layout(rows, num_heads, splits).total_bytes
    ):
        return None
    from ..jit.cake_dsv4 import VARIANTS

    programs = VARIANTS.get(arch, {})
    program = programs.get(route)
    if any(
        programs.get(key, program) == program
        for key in (_K64_SPLIT_VARIANT, _K64_SPLIT_MERGE)
    ):
        return None
    if not _k64_sibling_launchable(arch, device):
        return None
    return split


class _Launcher:
    """Per-call launch context: shared values plus per-variant overrides.

    ``split`` is the ``(W, R, s)`` split-KV plan of a whole launch
    (:func:`_k64_split`); ``None`` for every other launch and for every
    row-tiled chunk."""

    def __init__(
        self,
        *,
        arch: str,
        workspace: torch.Tensor,
        raw: torch.Tensor,
        values: dict[str, Any],
        split: Optional[tuple[int, int, int]] = None,
    ):
        self.arch = arch
        self.workspace = workspace
        self.raw = raw
        self.values = values
        self.split = split

    def variant(
        self, name: str, *, grid: tuple[int, int, int], **overrides: Any
    ) -> _PreparedLaunch:
        return _prepare_variant(
            name, arch=self.arch, grid=grid, values={**self.values, **overrides}
        )

    @staticmethod
    def run(*launches: _PreparedLaunch):
        """Issue the launches of one route.

        A single launch is one FFI call. Several launches go through the
        ``run_sequence`` host helper in one FFI call, so the host work of the
        second launch (argument conversion, Python-to-C transition) does not
        sit between the two kernels.
        """
        if len(launches) == 1:
            return launches[0]()
        flat: list[Any] = []
        for launch in launches:
            flat.append(launch.module.run)
            flat.append(len(launch.bound))
            flat.extend(launch.bound)
        return _sequence_module().run_sequence(*flat)

    def partials(self, num_splits: int) -> dict[str, Any]:
        partial_o, partial_lse = _partial_views(
            self.raw,
            self.values["O"],
            self.values["num_query_tokens"],
            self.values["num_heads"],
            num_splits,
        )
        return {
            "partial_O": partial_o,
            "partial_lse": partial_lse,
            "num_splits": num_splits,
        }

    def pieces(self, rows: int, num_splits: int) -> dict[str, Any]:
        """The workspace views of a split launch's ``rows`` split rows of
        ``num_splits`` pieces each, in token-major piece order: ``partial_O``
        flat and as the ``[rows * num_splits, H, 512]`` piece rows."""
        heads = self.values["num_heads"]
        partial_o, partial_lse = _partial_views(
            self.raw, self.values["O"], rows, heads, num_splits
        )
        return {
            "partial_O": partial_o,
            "partial_O_pieces": partial_o.view(rows * num_splits, heads, _HEAD_DIM),
            "partial_lse": partial_lse,
            "num_splits": num_splits,
        }

    def counters(self, merge_groups: int) -> torch.Tensor:
        counters = _counters(self.raw, merge_groups)
        _ensure_counters_zeroed(self.workspace, self.raw, merge_groups)
        return counters

    def reduce(self, reducer: str, **overrides: Any) -> _PreparedLaunch:
        tokens = self.values["num_query_tokens"]
        heads = self.values["num_heads"]
        return self.variant(reducer, grid=(tokens, heads, 1), **overrides)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def run_cake_dsv4(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[torch.Tensor],
    max_q_len: int,
    cum_seq_lens_q: Optional[torch.Tensor],
    seq_lens: torch.Tensor,
    backend: Literal["cake"],
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
    sparse_topk_lens_offset: int = 0,
    out_shape: Optional[tuple[int, ...]] = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Launch the CAKE DSv4 route for flattened ``query [rows, num_heads, 512]``.

    ``enable_pdl`` launches the programs that support it (the FP8 persistent,
    M64, low-head prefill and H64 cluster bodies) as programmatic dependents of the previous kernel on the
    stream; they wait for it before their first global read.  The other
    programs launch without the attribute.  Without ``enable_pdl`` the FP8
    H128 persistent route launches its PDL-free twin on SM103 from 384 query
    rows per launch and on SM100 when the longest request has at least 257
    query tokens (``_pdl_variant``).  On SM107, PDL launches of few metadata
    rows run a split-KV producer and a merge when the workspace holds their
    carve: FP8/H128 over items 1025-1152 wide (``_K64_SPLIT_RULES``; their
    launches of nine and five pieces and of three pieces at 16-20 rows
    merge with a merge built for their piece count where the registry has
    it, :func:`_k64_split_merge`) and FP8/H64 over items of four to nine
    tiles (``_K64_SPLIT_RULES_H64``) (:func:`_k64_split`;
    :func:`cake_dsv4_workspace_requirement` includes it).

    On SM107 the FP8 low-head prefill route runs launches of at most 27 query
    rows on its V-block sibling (``_vblock_variant``), launches of at most 64
    query rows on its early-PV sibling (``_overlap_variant``) and the larger
    ones on its own program; a part with fewer than 216 SMs lowers both
    ceilings.  Each program builds and loads its module on its first launch:
    warm up launches on both sides of each ceiling eagerly before a capture.

    On SM107 the four-stage multi-tile M64 program runs the 64-head launches of
    108 query rows up to two per SM over items at most 768 wide, of calls whose
    longest request has 2-4 query tokens, on its TMEM-batch sibling, and so
    the 64-head launches of 176-208 query rows of one-token requests over
    items 1152 wide with compressed pages of 64 rows on a device of 212 or 216
    SMs (``_KV4_BCL_DECODE_WINDOWS``), and every other launch on its own
    program (``_kv4_bcl_variant``).  Its 64-head launches of few query rows
    (at most 41 to 64 by item width and request length,
    ``_KV4_L2PF_MAX_TOKENS``) over items 512 or 640 wide with compressed pages
    of 64 rows, of one-token requests or of uniform 4-token requests, on
    a device of 212 or 216 SMs run its L2 prefetch sibling
    (``_kv4_l2pf_variant``).  Warm up launches on both sides of those windows
    eagerly before a capture as well.

    ``out=None`` allocates the output here, after the route is known, so the
    routes in ``_OUT_PHASE_BY_ROUTE`` get their measured-fast base phase;
    ``out_shape`` (default ``[rows, num_heads, 512]``) is the shape to allocate.

    ``query`` / ``out`` may have more rows than the metadata; only the first
    ``num_query_tokens`` (metadata rows) are read and written. ``max_q_len`` is
    the dense per-request query length (``query.shape[1]`` of the caller's
    ``[B, Q, H, 512]``) or the ragged maximum; ``seq_lens`` has one entry per
    request and ``cum_seq_lens_q`` (ragged only) ``batch_size + 1`` offsets.
    Apart from the ``out=None`` output and the staging buffer of a caller
    ``out`` off its route's store alignment (``_OUT_ALIGN_BY_ROUTE``; an
    aligned ``out`` takes none) no device memory is allocated here
    (the dense request offsets of the ragged-only producers and of row-tiled
    dense launches are process-lifetime constants, created on first use:
    warm such shapes up eagerly before a capture); see the module docstring
    for the workspace, counter and row-tiling contract.
    """
    if backend != "cake":
        raise ValueError(f"expected backend='cake', got {backend!r}")
    if query.ndim != 3:
        raise ValueError(
            f"query must be [num_tokens, num_heads, {_HEAD_DIM}], got shape {tuple(query.shape)}"
        )
    query_capacity, num_heads, head_dim = query.shape
    if head_dim != _HEAD_DIM:
        raise ValueError(f"CAKE DSv4 requires head dim {_HEAD_DIM}, got {head_dim}")
    if query.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError(f"unsupported CAKE DSv4 dtype: {query.dtype}")
    if not query.is_contiguous():
        raise ValueError("query must be contiguous; backend='cake' makes no host copy")
    device = query.device
    arch = _target_arch(device)

    meta = resolve_cake_dsv4_sparse_metadata(
        sparse_indices,
        sparse_topk_lens,
        extra_sparse_indices=extra_sparse_indices,
        extra_sparse_topk_lens=extra_sparse_topk_lens,
        sparse_topk_lens_offset=sparse_topk_lens_offset,
        query_rows=query_capacity,
    )
    for tensor, name in (
        (meta.swa_indices, "sparse_indices"),
        (meta.compressed_indices, "extra_sparse_indices"),
        (meta.sparse_topk_lens, "sparse_topk_lens"),
    ):
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    num_query_tokens = meta.num_query_tokens
    sparse_topk = meta.sparse_topk

    if out is not None:
        if out.dtype != torch.bfloat16:
            raise ValueError(f"out must be bfloat16, got {out.dtype}")
        if out.device != device:
            raise ValueError(f"out must be on {device}, got {out.device}")
        if not out.is_contiguous():
            raise ValueError(
                "out must be contiguous; backend='cake' makes no host copy"
            )
        row_elems = num_heads * _HEAD_DIM
        if out.numel() % row_elems or out.numel() < num_query_tokens * row_elems:
            raise ValueError(
                f"out must hold at least {num_query_tokens} rows of [{num_heads}, {_HEAD_DIM}], "
                f"got {tuple(out.shape)}"
            )
    query_rows = query[:num_query_tokens]

    swa = _dense_rows(swa_kv_cache, "swa_kv_cache", query.dtype)
    compressed = _dense_rows(compressed_kv_cache, "compressed_kv_cache", query.dtype)
    seq_lens = _int32_vector(seq_lens, "seq_lens", device)
    batch_size = seq_lens.numel()
    ragged = cum_seq_lens_q is not None
    if ragged:
        cum_seq_lens_q = _int32_vector(cum_seq_lens_q, "cum_seq_lens_q", device)
    scale1 = _device_scale(bmm1_scale, device=device, name="bmm1_scale")
    scale2 = _device_scale(bmm2_scale, device=device, name="bmm2_scale")
    if sinks is not None:
        if (
            sinks.dtype != torch.float32
            or sinks.device != device
            or not sinks.is_contiguous()
        ):
            raise ValueError(f"sinks must be a contiguous FP32 tensor on {device}")
    has_sinks = int(sinks is not None)
    sink_tensor = sinks if sinks is not None else scale1

    if workspace_buffer.device != device:
        raise ValueError(
            f"workspace_buffer must be on {device}, got {workspace_buffer.device}"
        )
    raw = _workspace_bytes(workspace_buffer)
    route = _route(
        dtype=query.dtype,
        num_heads=num_heads,
        max_q_len=max_q_len,
        ragged=ragged,
        sparse_topk=sparse_topk,
        batch_size=batch_size,
        compressed_page_size=compressed_kv_cache.shape[-2],
        num_query_tokens=meta.num_query_tokens,
        sm_count=_device_sm_count(device),
        arch=arch,
        kv4=_fp8_h64_kv4_edges(
            dtype=query.dtype,
            num_heads=num_heads,
            max_q_len=max_q_len,
            sparse_topk=sparse_topk,
            arch=arch,
            device=device,
        ),
    )

    if out is None:
        out = allocate_cake_dsv4_output(
            tuple(out_shape)
            if out_shape is not None
            else (query_capacity, num_heads, _HEAD_DIM),
            device,
            phase=cake_dsv4_out_phase(
                route, num_heads=num_heads, sparse_topk=sparse_topk
            ),
        )
        if out.numel() < num_query_tokens * num_heads * _HEAD_DIM:
            raise ValueError(
                f"out_shape {tuple(out.shape)} holds fewer than {num_query_tokens} rows"
            )
    out_rows = out.view(-1, num_heads, _HEAD_DIM)[:num_query_tokens]
    staged_out: Optional[torch.Tensor] = None
    out_align = _OUT_ALIGN_BY_ROUTE.get(route)
    if out_align is not None and out_rows.data_ptr() % out_align:
        staged_out = out_rows
        out_rows = torch.empty_like(out_rows)

    if not ragged and batch_size * max_q_len < num_query_tokens:
        raise ValueError(
            f"dense query layout [batch_size={batch_size}, max_q_len={max_q_len}] "
            f"holds {batch_size * max_q_len} rows but the metadata describes "
            f"{num_query_tokens}"
        )
    if ragged:
        request_offsets: Optional[torch.Tensor] = cum_seq_lens_q
        cum_seq_lens_q_value = cum_seq_lens_q
    else:
        # Dense rows: the kernels map t -> (t // max_q_len, t % max_q_len) and
        # never read cum_seq_lens_q, which is bound to the seq_lens pointer.
        # The ragged-only producers read request boundaries unconditionally
        # and receive the cached dense offsets (every request max_q_len long).
        request_offsets = None
        cum_seq_lens_q_value = (
            _dense_query_offsets(batch_size, max_q_len, device=device)
            if route in _RAGGED_ONLY_ROUTES
            else seq_lens
        )

    is_fp8 = query.dtype == torch.float8_e4m3fn
    values: dict[str, Any] = {
        "Q": query_rows.view(torch.uint8) if is_fp8 else query_rows,
        "SWA_cache": swa.view(torch.uint8) if is_fp8 else swa,
        "compressed_KV_cache": compressed.view(torch.uint8) if is_fp8 else compressed,
        "compressed_KV_rows": int(compressed.shape[0]),
        "O": out_rows,
        "seq_lens": seq_lens,
        "cum_seq_lens_q": cum_seq_lens_q_value,
        "sinks": sink_tensor,
        "bmm1_scale": scale1,
        "bmm2_scale": scale2,
        **meta.kernel_kwargs(),
        "num_heads": num_heads,
        "num_head_tiles": _ceil_div(num_heads, 64),
        "has_sinks": has_sinks,
        "batch_size": batch_size,
        "max_q_len": max_q_len,
        "ragged_query": int(ragged),
        # Host-only: the TMEM-batch sibling's decode windows and the L2
        # prefetch sibling key on it (_kv4_bcl_variant, _kv4_l2pf_variant);
        # no argument plan names it.
        "compressed_page_size": int(compressed_kv_cache.shape[-2]),
        "num_splits": 1,
        "total_work_items": num_query_tokens,
        "enable_pdl": bool(enable_pdl),
    }
    plan = _route_plan(
        route,
        num_query_tokens=num_query_tokens,
        num_heads=num_heads,
        sparse_topk=sparse_topk,
    )
    rows_per_launch = _rows_per_launch(
        plan,
        num_query_tokens=num_query_tokens,
        num_heads=num_heads,
        batch_size=batch_size,
        workspace_bytes=raw.numel(),
    )
    if rows_per_launch < 1:
        minimum = _launch_workspace_bytes(plan, 1, num_heads, batch_size + 1)
        raise ValueError(
            f"workspace_buffer holds {raw.numel()} bytes; the CAKE DSv4 route "
            f"{route} needs at least {minimum} bytes for one metadata row "
            f"({num_heads} heads, {plan.num_splits} splits); size it with "
            "get_cake_dsv4_workspace_bytes() or cake_dsv4_workspace_requirement()"
        )
    if rows_per_launch >= num_query_tokens:
        split = _k64_split(
            route,
            arch=arch,
            num_heads=num_heads,
            num_query_tokens=num_query_tokens,
            sparse_topk=sparse_topk,
            enable_pdl=bool(enable_pdl),
            workspace_bytes=raw.numel(),
            device=device,
        )
        _dispatch_route(
            route,
            _Launcher(
                arch=arch,
                workspace=workspace_buffer,
                raw=raw,
                values=values,
                split=split,
            ),
        )
    else:
        if request_offsets is None:
            request_offsets = _dense_query_offsets(batch_size, max_q_len, device=device)
        _dispatch_row_tiles(
            route,
            arch=arch,
            workspace=workspace_buffer,
            raw=raw,
            values=values,
            meta=meta,
            plan=plan,
            rows_per_launch=rows_per_launch,
            request_offsets=request_offsets,
        )
    if staged_out is not None:
        staged_out.copy_(out_rows)
    return out


def _dispatch_row_tiles(
    route: str,
    *,
    arch: str,
    workspace: torch.Tensor,
    raw: torch.Tensor,
    values: Mapping[str, Any],
    meta: SparseMetadata,
    plan: _RoutePlan,
    rows_per_launch: int,
    request_offsets: torch.Tensor,
) -> None:
    """Launch ``route`` over consecutive metadata-row chunks that fit the workspace.

    Every chunk sees row-sliced views of ``Q``, ``O`` and the metadata tables
    (no copies; the index tables keep their row strides) and its own row
    count. Chunks after the first cannot use the caller's layout (the kernels
    index rows from 0), so they run as ragged rows whose request boundaries
    are ``request_offsets - first_row`` (all ``batch_size + 1`` entries; the
    kernels' scan tolerates negative entries of the requests before the
    chunk), written into the workspace's offsets region in stream order --
    one captured ``torch.sub`` per chunk, nothing allocated.
    """
    total = values["num_query_tokens"]
    num_heads = values["num_heads"]
    num_offsets = int(request_offsets.numel())
    for first in range(0, total, rows_per_launch):
        rows = min(rows_per_launch, total - first)
        stop = first + rows
        chunk: dict[str, Any] = dict(values)
        chunk.update(
            Q=values["Q"][first:stop],
            O=values["O"][first:stop],
            swa_indices=_flat_index_span(meta.swa_indices[first:stop]),
            compressed_indices=_flat_index_span(meta.compressed_indices[first:stop]),
            sparse_topk_lens=meta.sparse_topk_lens[first:stop],
            num_query_tokens=rows,
            total_work_items=rows,
        )
        if first:
            layout = cake_dsv4_workspace_layout(
                rows, num_heads, plan.num_splits, num_query_offsets=num_offsets
            )
            begin = layout.query_offsets[0]
            shifted = raw[begin : begin + num_offsets * torch.int32.itemsize].view(
                torch.int32
            )
            torch.sub(request_offsets, first, out=shifted)
            chunk.update(cum_seq_lens_q=shifted, ragged_query=1)
        _dispatch_route(
            route, _Launcher(arch=arch, workspace=workspace, raw=raw, values=chunk)
        )


def _dispatch_route(route: str, L: _Launcher) -> None:
    v = L.values
    T = v["num_query_tokens"]
    H = v["num_heads"]
    topk = v["sparse_topk"]
    plan = _route_plan(route, num_query_tokens=T, num_heads=H, sparse_topk=topk)

    if route in (
        "bf16_swa128_single_cta",
        "bf16_h128_swa128",
        "bf16_h8_swa128_v43",
        "bf16_h16_h32_swa128_v44",
    ):
        head_tiles = _ceil_div(H, 64) if route == "bf16_h128_swa128" else 1
        L.run(
            L.variant(route, grid=(T * head_tiles * 4, 1, 1), num_head_tiles=head_tiles)
        )
        return

    if route == "bf16_h8_h16_source_exact":
        L.run(L.variant(route, grid=(v["max_q_len"], (H // 8) * 4, v["batch_size"])))
        return

    if route == "fp8_h8_h16_source_exact":
        # Same launch shape as the BF16 source-exact body (one CTA per
        # (query-within-sequence, value quarter, batch)); FP8 Q/KV pools.
        L.run(L.variant(route, grid=(v["max_q_len"], (H // 8) * 4, v["batch_size"])))
        return

    if route == "bf16_h64_guard_q_tma_batch_r25":
        L.run(L.variant(route, grid=(T, 2, 1)))
        return

    if route == "bf16_h64_compressed_q8_v38":
        num_splits = plan.num_splits
        parts = L.partials(num_splits)
        compressed = L.variant(route, grid=(T * num_splits * 2, 1, 1), **parts)
        if num_splits > 1:
            L.run(compressed, L.reduce("bf16_h64_compressed_reduce", **parts))
        else:
            L.run(compressed)
        return

    if route == "bf16_h32_topk128x_early_v47":
        num_splits = plan.num_splits
        head_tiles = plan.merge_groups_per_row
        parts = L.partials(num_splits)
        arrivals = L.counters(T * head_tiles)
        L.run(
            L.variant(
                route,
                grid=(T * num_splits * head_tiles, 1, 1),
                partition_arrivals=arrivals,
                num_head_tiles=head_tiles,
                **parts,
            )
        )
        return

    if route == "bf16_h64_prefill":
        L.run(L.variant(route, grid=(T, 1, 1), total_work_items=T))
        return

    if route in ("bf16_h128_topk128x", "bf16_h128_topk4x_v52", "bf16_h128_prefill_v42"):
        # The BF16/H128 producers launch directly with the grids their former
        # single-route family libraries computed: two CTAs per work item.
        producer = route
        num_splits = plan.num_splits
        grid_x = 2 * T
        if route == "bf16_h128_prefill_v42":
            # Persistent KV-reuse body: one two-CTA cluster per item, at most
            # half the SMs in clusters; the boustrophedon feed of the same body
            # serves the tail-majority rows (see the predicate above:
            # hardening-000037 0.77-0.90x -> 0.98-1.01x, hardening-000027
            # +8-10 % vs the striped feed).
            clusters = _bf16_h128_prefill_num_clusters(v["Q"].device)
            if _bf16_h128_prefill_uses_snake_feed(T, topk, clusters):
                producer = "bf16_h128_prefill_v42_snake"
            grid_x = min(2 * T, 2 * clusters)
        elif route == "bf16_h128_topk4x_v52":
            # Five fixed full-V KV owners per token plus the split-5 reducer
            # (plan.num_splits == 5).
            pass
        elif T > _BF16_TOPK128X_SPLIT_MAX_TOKENS:
            # Above the token bound one full-V owner per token whose invalid
            # (-1) sparse rows gather the tile's first index
            # (hardening-000025/31 0.45-0.95x -> 1.13-1.65x).  While four
            # CTAs per token still fit one wave, each token runs two 2-CTA
            # clusters that gather the full K tiles and only their 256-column
            # V half (same MMA operands and order per element -> identical
            # bits; -1.1 us / -8 % on the 32-token row on both targets): grid
            # = 4 * tokens, total_work_items keeps meaning query tokens.
            # Mirrors the Cake seed rule bf16_topk128x_uses_row_first_vsplit.
            if T <= _BF16_ROW_FIRST_V_HALF_SPLIT_MAX_TOKENS:
                producer = "bf16_h128_topk128x_row_first_vsplit"
                grid_x = 4 * T
            else:
                producer = "bf16_h128_topk128x_row_first"
        else:
            # Two-stage split program on both Blackwell targets: four disjoint
            # full-V KV owners (a three-tile row runs with the fourth tile
            # masked: 12.3-13.0 us vs 14.8 us for a three-owner pair) plus one
            # LSE reducer; GB300 width 260/388 rows measured 1.18-1.26x vs
            # trtllm-gen against 0.83-1.05x for the single-owner kernel
            # (plan.num_splits == 4).
            producer = "bf16_h128_topk128x_split4_sm100"
        parts = L.partials(num_splits)
        if num_splits == 1:
            L.run(L.variant(producer, grid=(grid_x, 1, 1), total_work_items=T, **parts))
            return
        # The split producers write their per-owner outputs through ``O``.
        work_items = T * num_splits
        split = L.variant(
            producer,
            grid=(2 * work_items, 1, 1),
            total_work_items=work_items,
            **{**parts, "O": parts["partial_O"]},
        )
        if num_splits == 5:
            # The two split-5 reducers are different kernels with different
            # launch contracts (copied from their former family libraries):
            # the sm_103a body runs grid (tokens, heads / 4), the sm_100a
            # body grid (tokens, heads).
            reducer_heads = H // 4 if L.arch == "sm_103a" else H
            reduce = L.variant(
                "bf16_h128_split5_reduce", grid=(T, reducer_heads, 1), **parts
            )
        else:
            reduce = L.reduce("split_reduce", **parts)
        L.run(split, reduce)
        return

    if route == "fp8_h64_source_exact":
        head_tiles = _ceil_div(H, 64)
        L.run(
            L.variant(
                route,
                grid=(v["max_q_len"], head_tiles, v["batch_size"]),
                num_head_tiles=head_tiles,
                total_work_items=v["max_q_len"] * head_tiles * v["batch_size"],
            )
        )
        return

    if route in (
        "fp8_h128_prefill_source_persistent",
        "fp8_h128_prefill_source_persistent_uniform",
    ):
        if L.split is not None:
            # SM107 split-KV launch (_k64_split): W whole rows, then R rows of
            # s pieces each (two CTAs per item); the merge writes the split
            # rows' O, one CTA per (row, head) or, for the three-piece merge,
            # per (row, two heads) (_k64_split_merge).  One run_sequence call.
            num_whole, rows, num_splits = L.split
            items = num_whole + rows * num_splits
            pieces = L.pieces(rows, num_splits)
            piece_launch = L.variant(
                _K64_SPLIT_VARIANT,
                grid=(2 * items, 1, 1),
                total_work_items=items,
                num_whole_items=num_whole,
                **pieces,
            )
            merge, heads_per_cta = _k64_split_merge(
                arch=L.arch, num_heads=H, rows=rows, num_splits=num_splits
            )
            merge_launch = L.variant(
                merge,
                grid=(rows, H // heads_per_cta, 1),
                O=v["O"][num_whole:],
                **pieces,
            )
            L.run(piece_launch, merge_launch)
            return
        parts = L.partials(1)
        program = _pdl_variant(
            _k64_variant(route, arch=L.arch, num_query_tokens=T, device=v["Q"].device),
            arch=L.arch,
            enable_pdl=v["enable_pdl"],
            num_query_tokens=T,
            max_q_len=v["max_q_len"],
        )
        L.run(L.variant(program, grid=(T * 2, 1, 1), total_work_items=T, **parts))
        return

    if route in (
        "fp8_h64_prefill_source_persistent_m64",
        "fp8_h64_prefill_source_persistent_m64_multi_tile",
    ):
        # One CTA per token (cta_group::1, M = 64); same kernel kwargs as the
        # FP8/H128 persistent body (num_heads runtime, -1 masking, padded rows,
        # caller-owned workspace), grid = tokens.  Both M64 programs share the
        # kernel ABI; _fp8_h64_m64_program picks one by the item width.  SM107
        # runs the multi-tile program's launches on its four-stage sibling
        # where this process can launch it (_kv4_variant); run_cake_dsv4
        # routed the FP8/H64 decode rows by the same check (_fp8_h64_kv4_edges).
        # The four-stage launches of the TMEM-batch windows (64-head MTP decode
        # and the decode windows of _KV4_BCL_DECODE_WINDOWS) run its TMEM-batch
        # sibling (_kv4_bcl_variant); those of the L2 prefetch window (64-head
        # decode of few rows over items 512 or 640 wide, _KV4_L2PF_MAX_TOKENS)
        # run its L2 prefetch sibling (_kv4_l2pf_variant).  The windows do not
        # overlap: the TMEM-batch windows start at 108 rows.
        parts = L.partials(1)
        device = v["Q"].device
        page = v.get("compressed_page_size")
        program = _kv4_bcl_variant(
            _kv4_variant(route, arch=L.arch, device=device),
            arch=L.arch,
            num_heads=H,
            max_q_len=v["max_q_len"],
            num_query_tokens=T,
            sparse_topk=topk,
            device=device,
            compressed_page_size=page,
        )
        program = _kv4_l2pf_variant(
            program,
            arch=L.arch,
            num_heads=H,
            max_q_len=v["max_q_len"],
            num_query_tokens=T,
            sparse_topk=topk,
            device=device,
            batch_size=v.get("batch_size"),
            compressed_page_size=page,
        )
        program = _pdl_variant(
            program,
            arch=L.arch,
            enable_pdl=v["enable_pdl"],
            num_query_tokens=T,
            max_q_len=v["max_q_len"],
        )
        L.run(L.variant(program, grid=(T, 1, 1), total_work_items=T, **parts))
        return

    if route in (
        "fp8_lowhead_one_partition",
        "fp8_lowhead_h64",
        "fp8_lowhead_prefill",
    ):
        # One producer partition writes the final output directly.  The
        # prefill producer runs one CTA per work item over two items per
        # token; the decode producers run one two-CTA cluster per token.  SM107
        # runs the prefill producer's launches up to a row ceiling on its
        # early-PV sibling (_overlap_variant), and the smallest ones on its
        # V-block sibling over four items per token (_vblock_variant).
        work_items = 2 * T if route == "fp8_lowhead_prefill" else T
        ctas = 2 * T
        program = _vblock_variant(
            route, arch=L.arch, num_query_tokens=T, device=v["Q"].device
        )
        if program != route:
            ctas = work_items = _VBLOCK_ITEMS_PER_TOKEN * T
        else:
            program = _overlap_variant(
                route, arch=L.arch, num_query_tokens=T, device=v["Q"].device
            )
        L.run(
            L.variant(
                program,
                grid=(ctas, 1, 1),
                total_work_items=work_items,
                **L.partials(1),
            )
        )
        return

    raise RuntimeError(f"unhandled CAKE DSv4 route: {route}")


# --------------------------------------------------------------------------- #
# NVFP4 cache route                                                           #
# --------------------------------------------------------------------------- #


NVFP4Member = Literal["persistent", "cluster", "t64", "pv", "swap", "tile", "tile_h64"]


@dataclass(frozen=True)
class NVFP4Plan:
    """Launch plan of the NVFP4 decode family (Cake ``plan()`` output)."""

    num_query_tokens: int
    num_heads: int
    num_head_tiles: int
    main_width: int
    extra_width: int
    num_main_tiles: int
    num_extra_tiles: int
    total_tiles: int
    num_splits: int
    tiles_per_split: int
    member: NVFP4Member
    sm_count: int
    tile_n: Optional[int] = (
        None  # heads on the MMA N side of the SwapsAB / tile64 members (retrace knob)
    )
    o_chunks: int = 1  # CTAs per work item of the one-tile members (retrace knob)
    merge_heads_per_cta: int = _NVFP4_MERGE_HEADS_PER_CTA

    @property
    def cluster_ctas(self) -> int:
        return _NVFP4_CLUSTER_CTAS if self.member == "cluster" else 1

    @property
    def grid(self) -> int:
        return (
            self.num_query_tokens
            * self.num_splits
            * self.num_head_tiles
            * self.cluster_ctas
            * self.o_chunks
        )

    @property
    def merge_groups(self) -> int:
        return self.num_query_tokens * self.num_head_tiles

    @property
    def merge_grid(self) -> tuple[int, int, int]:
        return (
            self.num_query_tokens,
            _ceil_div(self.num_heads, self.merge_heads_per_cta),
            1,
        )

    @property
    def variant(self) -> str:
        return _nvfp4_variant_name(self.member)


# Registered program of each family member (Cake flashinfer_blackwell_sparse_mla_dsv4_nvfp4_program).  The
# templated members (pv, swap, tile, tile_h64) take their retrace knobs -- ``tile_n`` (heads on the MMA N side)
# and ``o_chunks`` (CTAs per work item) -- as launch arguments of one program each; its launcher switches over
# the explicit instantiations of the kernel unit (swap: n{16,32} x oc{1,2,4}, tile: oc{1,2,4}, tile_h64:
# oc{2,4}, pv: n{16,32}).  The standalone tile64 program keeps its exported knob suffix.
_NVFP4_MEMBER_VARIANTS: Mapping[str, str] = {
    "persistent": _NVFP4_VARIANT_PERSISTENT,
    "cluster": _NVFP4_VARIANT_CLUSTER,
    "t64": "nvfp4_decode_t64_n64_oc1",
    "pv": "nvfp4_decode_pv",
    "swap": "nvfp4_decode_swap",
    "tile": "nvfp4_decode_tile",
    "tile_h64": "nvfp4_decode_tile_h64",
}


def _nvfp4_variant_name(member: str) -> str:
    """Registered variant name of a member's program (the merge is ``nvfp4_merge``)."""
    if member not in _NVFP4_MEMBER_VARIANTS:
        raise ValueError(f"unknown NVFP4 family member {member!r}")
    return _NVFP4_MEMBER_VARIANTS[member]


# Every decode variant a plan can select (the host names them statically so the export can check that each
# generated program is bindable).
_NVFP4_DECODE_VARIANTS = tuple(_NVFP4_MEMBER_VARIANTS.values())
_NVFP4_VARIANTS = (
    *_NVFP4_DECODE_VARIANTS,
    _NVFP4_VARIANT_MERGE,
    *_NVFP4_PREFILL_VARIANTS,
)


def _nvfp4_split_shape(total_tiles: int, num_splits: int) -> tuple[int, int]:
    """Canonical (splits, tiles per split): contiguous tile ranges, the split count is the number of non-empty
    ranges (Cake ``_split_shape``)."""
    num_splits = max(1, min(int(num_splits), total_tiles))
    tiles_per_split = _ceil_div(total_tiles, num_splits)
    num_splits = _ceil_div(total_tiles, tiles_per_split)
    if num_splits > _NVFP4_MAX_SPLITS:
        raise ValueError(
            f"num_splits {num_splits} exceeds the NVFP4 route cap {_NVFP4_MAX_SPLITS}"
        )
    return num_splits, tiles_per_split


def _nvfp4_plan_cost_us(
    member: str,
    *,
    num_tokens: int,
    num_heads: int,
    head_tiles: int,
    total_tiles: int,
    num_splits: int,
    sm_count: int,
) -> float:
    """Round-43 chain model of ``member`` running the row with ``num_splits`` splits (Cake ``plan_cost_us``)."""
    head_us, tile_us, load_us = _NVFP4_PLAN_CHAIN_US[member]
    num_splits, tiles_per_split = _nvfp4_split_shape(total_tiles, num_splits)
    ctas_per_unit = _NVFP4_CLUSTER_CTAS if member == "cluster" else 1
    grid = num_tokens * head_tiles * num_splits * ctas_per_unit
    slots = sm_count - (sm_count % ctas_per_unit)
    waves = _ceil_div(grid, slots)
    chain = (
        head_us + (tiles_per_split - 1) * tile_us + load_us * min(grid, slots) / slots
    )
    cost = waves * chain
    if num_splits > 1:
        cost += _NVFP4_PLAN_MERGE_US + _NVFP4_PLAN_MERGE_US_PER_MB * (
            num_tokens * num_heads * (num_splits + 1) * _NVFP4_HEAD_DIM * 2 / 1e6
        )
    return cost


def _nvfp4_plan_large_heads(
    num_tokens: int, num_heads: int, total_tiles: int, sm_count: int
) -> tuple[NVFP4Member, int]:
    """(member, splits) with the lowest modelled cost for an H > _NVFP4_PV_MAX_HEADS row (Cake ``_plan_large_heads``):
    the tile member at one tile per CTA, then persistent / cluster / tile64 over every canonical split count that
    leaves >= 2 tiles per CTA; ties keep the earlier candidate."""
    common = dict(
        num_tokens=num_tokens,
        num_heads=num_heads,
        total_tiles=total_tiles,
        sm_count=sm_count,
    )
    head_tiles = _ceil_div(num_heads, _NVFP4_TILE_Q)
    cands: list[tuple[NVFP4Member, int, float]] = [
        (
            "tile",
            total_tiles,
            _nvfp4_plan_cost_us(
                "tile", head_tiles=head_tiles, num_splits=total_tiles, **common
            ),
        )
    ]
    for splits in range(1, min(total_tiles - 1, _NVFP4_MAX_SPLITS) + 1):
        if _ceil_div(total_tiles, _ceil_div(total_tiles, splits)) != splits:
            continue  # not a canonical split count
        cands.append(
            (
                "persistent",
                splits,
                _nvfp4_plan_cost_us(
                    "persistent", head_tiles=head_tiles, num_splits=splits, **common
                ),
            )
        )
        if num_heads >= _NVFP4_CLUSTER_MIN_HEADS:
            cands.append(
                (
                    "cluster",
                    splits,
                    _nvfp4_plan_cost_us(
                        "cluster", head_tiles=head_tiles, num_splits=splits, **common
                    ),
                )
            )
        if num_heads in _NVFP4_T64_HEAD_COUNTS:
            cands.append(
                (
                    "t64",
                    splits,
                    _nvfp4_plan_cost_us(
                        "t64",
                        head_tiles=_ceil_div(num_heads, _NVFP4_T64_TILE_Q),
                        num_splits=splits,
                        **common,
                    ),
                )
            )
    best = min(cands, key=lambda c: c[2])
    return best[0], best[1]


def _nvfp4_pv_splits(
    num_tokens: int, total_tiles: int, sm_count: int
) -> tuple[int, int]:
    """Split count of the persistent SwapsAB member for an H <= 32 row (Cake swap_pv ``plan``: one CTA per SM, the
    chain ``waves * (HEAD + (tiles - 1) * TILE) + MERGE`` minimised over the split count; first minimum wins)."""
    best = None
    for cand in range(1, min(total_tiles, _NVFP4_MAX_SPLITS) + 1):
        tiles_per_split = _ceil_div(total_tiles, cand)
        splits = _ceil_div(total_tiles, tiles_per_split)
        waves = _ceil_div(num_tokens * splits, sm_count)
        cost = waves * (
            _NVFP4_PV_PLAN_HEAD_US + (tiles_per_split - 1) * _NVFP4_PV_PLAN_TILE_US
        ) + (_NVFP4_PV_PLAN_MERGE_US if splits > 1 else 0.0)
        if best is None or cost < best[0] - 1e-9:
            best = (cost, splits, tiles_per_split)
    return best[1], best[2]


def _nvfp4_swap_o_chunks(work_items: int, sm_count: int) -> int:
    """CTAs per work item of the one-tile SwapsAB member: the largest of 4 / 2 / 1 whose grid fits one wave."""
    if 4 * work_items <= sm_count:
        return 4
    if 2 * work_items <= sm_count:
        return 2
    return 1


def _nvfp4_tile_o_chunks(work_items: int, sm_count: int) -> int:
    """CTAs per work item of the one-tile member (two CTAs per SM at its footprint: half the SM count)."""
    if 4 * work_items <= sm_count // 2:
        return 4
    if 2 * work_items <= sm_count // 2:
        return 2
    return 1


def _nvfp4_merge_heads_per_cta(num_tokens: int, num_heads: int, sm_count: int) -> int:
    """Heads (warps) per merge CTA: the smallest power of two <= 16 whose grid stays within one CTA per SM, else 16
    (Cake ``merge_heads_per_cta``, round 45)."""
    for hpc in (1, 2, 4, 8, 16):
        if num_tokens * _ceil_div(num_heads, hpc) <= sm_count:
            return hpc
    return _NVFP4_MERGE_HEADS_PER_CTA


def _nvfp4_plan(
    *,
    num_query_tokens: int,
    num_heads: int,
    sparse_topk: int,
    extra_topk: int,
    sm_count: int,
) -> NVFP4Plan:
    """Reproduce the Cake NVFP4 family ``plan()`` and member dispatch.

    Candidate tiles are 128 wide (``total_tiles`` over both tables).  Rows with
    ``H <= 32`` run the persistent SwapsAB member (``pv``) when its wave-aware
    split rule gives more than one tile per CTA, otherwise the one-tile SwapsAB
    member (``swap``, ``o_chunks`` CTAs per work item).  Rows with ``H >= 64``
    pick (member, splits) from the round-43 chain model over the one-tile
    ``tile`` member, the persistent member, the 2-CTA ``cluster`` member and the
    tile64 member ``t64`` (64 heads per CTA; ``H128`` runs two head tiles); a
    one-tile row with ``H <= 64`` whose split gives 2 or 4 CTAs per work item
    runs the H64 tile member ``tile_h64`` instead of ``tile`` (same grid).  The
    split merge runs behind any member with ``num_splits > 1`` on
    ``merge_heads_per_cta`` heads per CTA.
    """
    tokens = _positive_int(num_query_tokens, "num_query_tokens")
    heads = _positive_int(num_heads, "num_heads")
    if heads not in _NVFP4_HEAD_COUNTS:
        raise ValueError(f"num_heads must be one of {_NVFP4_HEAD_COUNTS}, got {heads}")
    sms = _positive_int(sm_count, "sm_count")
    main_width = _positive_int(sparse_topk, "sparse_topk")
    if (
        isinstance(extra_topk, bool)
        or not isinstance(extra_topk, int)
        or extra_topk < 0
    ):
        raise ValueError(f"extra_topk must be a non-negative int, got {extra_topk!r}")
    main_tiles = _ceil_div(main_width, _TILE_KV)
    extra_tiles = _ceil_div(extra_topk, _TILE_KV)
    total_tiles = main_tiles + extra_tiles
    if total_tiles < 1:
        raise ValueError("at least one candidate tile is required")
    if total_tiles > _NVFP4_MAX_SPLITS:
        raise ValueError(
            f"total candidate tiles {total_tiles} exceeds the NVFP4 route cap {_NVFP4_MAX_SPLITS}; "
            "a query packed with cake_dsv4_nvfp4_pack_query runs the prefill bodies, "
            "which take any table width"
        )
    num_head_tiles = _ceil_div(heads, _NVFP4_TILE_Q)
    common = dict(
        num_query_tokens=tokens,
        num_heads=heads,
        main_width=main_width,
        extra_width=extra_topk,
        num_main_tiles=main_tiles,
        num_extra_tiles=extra_tiles,
        total_tiles=total_tiles,
        sm_count=sms,
        merge_heads_per_cta=_nvfp4_merge_heads_per_cta(tokens, heads, sms),
    )
    # the persistent member's own split rule: enough CTAs to cover the SMs
    _, p_tiles = _nvfp4_split_shape(
        total_tiles, max(1, min(total_tiles, _ceil_div(sms, tokens * num_head_tiles)))
    )
    if heads > _NVFP4_PV_MAX_HEADS:
        member, splits = _nvfp4_plan_large_heads(tokens, heads, total_tiles, sms)
        if member == "t64":
            splits, tiles_per_split = _nvfp4_split_shape(total_tiles, splits)
            return NVFP4Plan(
                num_head_tiles=_ceil_div(heads, _NVFP4_T64_TILE_Q),
                num_splits=splits,
                tiles_per_split=tiles_per_split,
                member="t64",
                tile_n=_NVFP4_T64_TILE_Q,
                o_chunks=1,
                **common,
            )
        if member in ("cluster", "persistent"):
            splits, tiles_per_split = _nvfp4_split_shape(total_tiles, splits)
            return NVFP4Plan(
                num_head_tiles=num_head_tiles,
                num_splits=splits,
                tiles_per_split=tiles_per_split,
                member=member,
                **common,
            )
        # the one-tile member: one candidate tile per CTA, o_chunks CTAs per work item;
        # H <= _NVFP4_TILE_H64_MAX_HEADS rows at 2 or 4 CTAs per work item run the H64
        # tile member (the 1-CTA form keeps the tile member)
        o_chunks = _nvfp4_tile_o_chunks(tokens * total_tiles * num_head_tiles, sms)
        return NVFP4Plan(
            num_head_tiles=num_head_tiles,
            num_splits=total_tiles,
            tiles_per_split=1,
            member=(
                "tile_h64"
                if heads <= _NVFP4_TILE_H64_MAX_HEADS and o_chunks >= 2
                else "tile"
            ),
            o_chunks=o_chunks,
            **common,
        )
    pv_splits, pv_tiles = _nvfp4_pv_splits(tokens, total_tiles, sms)
    if pv_tiles > 1:
        return NVFP4Plan(
            num_head_tiles=num_head_tiles,
            num_splits=pv_splits,
            tiles_per_split=pv_tiles,
            member="pv",
            tile_n=16 if heads <= 16 else 32,
            o_chunks=1,
            **common,
        )
    if p_tiles == 1:
        # one-tile rows with H <= _NVFP4_SWAP_MAX_HEADS: the one-tile SwapsAB member
        return NVFP4Plan(
            num_head_tiles=num_head_tiles,
            num_splits=total_tiles,
            tiles_per_split=1,
            member="swap",
            tile_n=16 if heads <= 16 else 32,
            o_chunks=_nvfp4_swap_o_chunks(tokens * total_tiles * num_head_tiles, sms),
            **common,
        )
    # Unreachable: a one-tile pv split (pv_tiles == 1) implies p_tiles == 1
    # (sweep over sm_count 1..264 x total_tiles 1..12 x T 1..4096: 0 hits).
    raise RuntimeError(
        f"CAKE DSv4 NVFP4 plan has no member for H={heads}, T={tokens}, "
        f"{total_tiles} tiles on {sms} SMs (pv tiles {pv_tiles}, persistent "
        f"tiles {p_tiles})"
    )


def _nvfp4_sm_count(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _require_nvfp4_variant(variant: str, *, arch: str) -> None:
    """Fail early and clearly while the NVFP4 sources are not exported yet."""
    from ..jit.cake_dsv4 import get_cake_dsv4_spec

    try:
        get_cake_dsv4_spec(variant, arch=arch)
    except ValueError as exc:
        raise NotImplementedError(
            "cake_dsv4 NVFP4 generated sources not yet exported: variant "
            f"{variant!r} has no registered contract for {arch} "
            "(csrc/cake_dsv4/<arch>/ and flashinfer/jit/cake_dsv4.py)"
        ) from exc


def _log2_page_size(page_size: int, *, name: str) -> int:
    if page_size <= 0 or page_size & (page_size - 1):
        raise ValueError(f"{name}: page_size must be a power of two, got {page_size}")
    return page_size.bit_length() - 1


def _nvfp4_cache_geometry(
    cache: torch.Tensor,
    *,
    name: str,
    kv_layout: Literal["HND", "NHD"],
    page_size: Optional[int],
    device: torch.device,
) -> tuple[torch.Tensor, int, int]:
    """``(flat uint8 [pages, page_stride] view, page_size, page_stride_bytes)`` of an NVFP4 pool.

    Mirrors the Cake ``_cache_geometry``: accepts ``[P, bytes]`` (needs the
    logical ``page_size``), HND ``[P, 1, page_size, 384]``, NHD
    ``[P, page_size, 1, 384]`` and ``[P, page_size, 384]`` tensors. Tokens of a
    page are contiguous 384-byte records; the page stride may be padded. No
    copy is made.
    """
    if not isinstance(cache, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if cache.dtype != torch.uint8:
        raise TypeError(f"{name} must be uint8, got {cache.dtype}")
    if cache.device != device:
        raise ValueError(f"{name} must be on {device}, got {cache.device}")
    if cache.dim() == 2:
        if not cache.is_contiguous():
            raise ValueError(f"{name}: a 2-D packed pool must be contiguous")
        if page_size is None:
            raise ValueError(
                f"{name}: a 2-D [pages, bytes] pool needs the logical page_size"
            )
        logical = _positive_int(page_size, f"{name} page_size")
        page_stride = int(cache.shape[1])
        if page_stride < logical * _NVFP4_TOKEN_BYTES:
            raise ValueError(
                f"{name}: page byte count {page_stride} is smaller than "
                f"page_size {logical} x {_NVFP4_TOKEN_BYTES}"
            )
        return cache, logical, page_stride
    if cache.dim() == 4:
        if kv_layout == "HND":
            if cache.shape[1] != 1:
                raise ValueError(
                    f"{name}: expected HND [P, 1, page_size, {_NVFP4_TOKEN_BYTES}], "
                    f"got {tuple(cache.shape)}"
                )
            page_dim = 2
        elif kv_layout == "NHD":
            if cache.shape[2] != 1:
                raise ValueError(
                    f"{name}: expected NHD [P, page_size, 1, {_NVFP4_TOKEN_BYTES}], "
                    f"got {tuple(cache.shape)}"
                )
            page_dim = 1
        else:
            raise ValueError(
                f"kv_layout must be either 'HND' or 'NHD', got {kv_layout}"
            )
    elif cache.dim() == 3:
        page_dim = 1
    else:
        raise ValueError(f"{name}: unsupported rank {cache.dim()}")
    geometry_page_size = int(cache.shape[page_dim])
    if (
        int(cache.shape[-1]) != _NVFP4_TOKEN_BYTES
        or int(cache.stride(-1)) != 1
        or int(cache.stride(page_dim)) != _NVFP4_TOKEN_BYTES
    ):
        raise ValueError(
            f"{name}: tokens must be contiguous {_NVFP4_TOKEN_BYTES}-byte records "
            f"(shape {tuple(cache.shape)}, strides {cache.stride()})"
        )
    page_stride = int(cache.stride(0))
    if page_stride < geometry_page_size * _NVFP4_TOKEN_BYTES:
        raise ValueError(
            f"{name}: page stride {page_stride} B is smaller than page_size * {_NVFP4_TOKEN_BYTES}"
        )
    if page_size is not None and int(page_size) != geometry_page_size:
        raise ValueError(
            f"{name}: page_size {page_size} does not match the cache geometry ({geometry_page_size})"
        )
    flat = cache.as_strided((int(cache.shape[0]), page_stride), (page_stride, 1))
    return flat, geometry_page_size, page_stride


def _nvfp4_lengths(
    tensor: Optional[torch.Tensor], name: str, *, rows: int, device: torch.device
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.dtype != torch.int32:
        raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous() or tensor.numel() < rows:
        raise ValueError(
            f"{name} must be a contiguous int32 tensor with >= {rows} entries"
        )
    return tensor


def run_cake_dsv4_nvfp4(
    *,
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    compressed_kv_cache: Optional[torch.Tensor],
    workspace_buffer: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_topk_lens: Optional[torch.Tensor],
    out: torch.Tensor,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[torch.Tensor],
    max_q_len: Optional[int],
    cum_seq_lens_q: Optional[torch.Tensor],
    seq_lens: Optional[torch.Tensor],
    backend: Literal["cake"],
    extra_sparse_indices: Optional[torch.Tensor] = None,
    extra_sparse_topk_lens: Optional[torch.Tensor] = None,
    kv_layout: Literal["HND", "NHD"] = "HND",
    swa_page_size: Optional[int] = None,
    compressed_page_size: Optional[int] = None,
    lse_scale: float = 1.0,
) -> torch.Tensor:
    """Launch the CAKE DSv4 NVFP4 route for ``query``: BF16 ``[rows, num_heads, 512]``
    (the decode family) or packed NVFP4 ``uint8
    [rows, num_heads, 384]`` (the prefill bodies; see the module docstring and
    :func:`cake_dsv4_nvfp4_pack_query`).

    ``sparse_indices [T, topk]`` (int32, unit inner stride) indexes
    ``swa_kv_cache``; ``sparse_topk_lens`` (optional, int32 ``>= T``) clamps its
    active prefix. ``extra_sparse_indices [T, extra_topk]`` indexes
    ``compressed_kv_cache`` with ``extra_sparse_topk_lens``; without it the
    compressed pool is unused. Both pools are opaque ``uint8`` NVFP4 pages (see
    the module docstring); ``swa_page_size`` / ``compressed_page_size`` are
    required for 2-D ``[pages, bytes]`` pools and otherwise checked against the
    geometry. ``query`` / ``out`` may carry more rows than ``T``; rows ``>= T``
    are untouched. ``cum_seq_lens_q`` and ``seq_lens`` are accepted for surface
    parity with :func:`run_cake_dsv4` but unused, and ``max_q_len`` is only a
    work-mapping constant of the prefill bodies: the kernels consume flat token
    coordinates. A packed query runs the NVFP4 prefill bodies
    (:func:`_nvfp4_prefill`); a BF16 query runs the decode family. The base-2
    LSE is written to the
    workspace (not returned). No device memory is allocated.
    """
    if backend != "cake":
        raise ValueError(f"expected backend='cake', got {backend!r}")
    if query.ndim != 3:
        raise ValueError(
            f"query must be [num_tokens, num_heads, {_HEAD_DIM}], got shape {tuple(query.shape)}"
        )
    query_capacity, num_heads, head_dim = (int(d) for d in query.shape)
    packed_query = query.dtype == torch.uint8
    if packed_query:
        if head_dim != _NVFP4_TOKEN_BYTES:
            raise ValueError(
                f"a packed NVFP4 query has {_NVFP4_TOKEN_BYTES}-byte rows, got {head_dim}"
            )
    elif query.dtype != torch.bfloat16:
        raise ValueError(
            "the CAKE DSv4 NVFP4 route takes a BF16 query or a packed NVFP4 "
            f"query (uint8, cake_dsv4_nvfp4_pack_query), got {query.dtype}"
        )
    elif head_dim != _HEAD_DIM:
        raise ValueError(f"CAKE DSv4 requires head dim {_HEAD_DIM}, got {head_dim}")
    if not query.is_contiguous():
        raise ValueError("query must be contiguous; backend='cake' makes no host copy")
    if num_heads not in _NVFP4_HEAD_COUNTS:
        raise ValueError(
            f"num_heads must be one of {_NVFP4_HEAD_COUNTS}, got {num_heads}"
        )
    device = query.device
    arch = _target_arch(device)

    main_indices = _int32_table(sparse_indices, "sparse_indices")
    if main_indices.device != device:
        raise ValueError(
            f"sparse_indices must be on {device}, got {main_indices.device}"
        )
    num_query_tokens = int(main_indices.shape[0])
    if not 1 <= num_query_tokens <= query_capacity:
        raise ValueError(
            f"sparse_indices must describe 1..{query_capacity} query rows, got {num_query_tokens}"
        )
    main_width = int(main_indices.shape[1])
    if extra_sparse_topk_lens is not None and extra_sparse_indices is None:
        raise ValueError("extra_sparse_topk_lens requires extra_sparse_indices")
    if extra_sparse_indices is not None:
        extra_indices = _int32_table(
            extra_sparse_indices, "extra_sparse_indices", rows=num_query_tokens
        )
        if extra_indices.device != device:
            raise ValueError(
                f"extra_sparse_indices must be on {device}, got {extra_indices.device}"
            )
        extra_width = int(extra_indices.shape[1])
        if compressed_kv_cache is None:
            raise ValueError("extra_sparse_indices requires compressed_kv_cache")
    else:
        extra_indices = main_indices
        extra_width = 0
    main_lengths = _nvfp4_lengths(
        sparse_topk_lens, "sparse_topk_lens", rows=num_query_tokens, device=device
    )
    extra_lengths = _nvfp4_lengths(
        extra_sparse_topk_lens,
        "extra_sparse_topk_lens",
        rows=num_query_tokens,
        device=device,
    )
    if seq_lens is not None and seq_lens.dtype != torch.int32:
        raise ValueError(f"seq_lens must be int32, got {seq_lens.dtype}")
    if cum_seq_lens_q is not None and cum_seq_lens_q.dtype != torch.int32:
        raise ValueError(f"cum_seq_lens_q must be int32, got {cum_seq_lens_q.dtype}")

    main_flat, main_page_size, main_page_stride = _nvfp4_cache_geometry(
        swa_kv_cache,
        name="swa_kv_cache",
        kv_layout=kv_layout,
        page_size=swa_page_size,
        device=device,
    )
    if extra_width:
        extra_flat, extra_page_size, extra_page_stride = _nvfp4_cache_geometry(
            compressed_kv_cache,
            name="compressed_kv_cache",
            kv_layout=kv_layout,
            page_size=compressed_page_size,
            device=device,
        )
    else:
        extra_flat, extra_page_size, extra_page_stride = (
            main_flat,
            main_page_size,
            main_page_stride,
        )

    if out.dtype != torch.bfloat16:
        raise ValueError(f"out must be bfloat16, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous; backend='cake' makes no host copy")
    row_elems = num_heads * _HEAD_DIM
    if out.numel() % row_elems or out.numel() < num_query_tokens * row_elems:
        raise ValueError(
            f"out must hold at least {num_query_tokens} rows of [{num_heads}, {_HEAD_DIM}], "
            f"got {tuple(out.shape)}"
        )
    query_rows = query[:num_query_tokens]
    out_rows = out.view(-1, num_heads, _HEAD_DIM)[:num_query_tokens]

    scale1 = _device_scale(bmm1_scale, device=device, name="bmm1_scale")
    scale2 = _device_scale(bmm2_scale, device=device, name="bmm2_scale")
    if sinks is not None:
        if (
            sinks.dtype != torch.float32
            or sinks.device != device
            or not sinks.is_contiguous()
            or sinks.numel() < num_heads
        ):
            raise ValueError(
                f"sinks must be a contiguous FP32 tensor with >= {num_heads} entries on {device}"
            )
    sink_tensor = sinks if sinks is not None else scale1
    lse_scale = float(lse_scale)

    # The decode family's split plan (its candidate tiles are capped); the
    # prefill bodies take any table width and run one split.
    plan = (
        None
        if packed_query
        else _nvfp4_plan(
            num_query_tokens=num_query_tokens,
            num_heads=num_heads,
            sparse_topk=main_width,
            extra_topk=extra_width,
            sm_count=_nvfp4_sm_count(device),
        )
    )
    num_splits = 1 if plan is None else plan.num_splits

    if workspace_buffer.device != device:
        raise ValueError(
            f"workspace_buffer must be on {device}, got {workspace_buffer.device}"
        )
    raw = _workspace_bytes(workspace_buffer)
    prefill = packed_query
    # The prefill bodies write the final O and LSE directly, so their carve has one
    # split (no partial_O) whatever the decode plan would split into.
    layout = cake_dsv4_workspace_layout(
        num_query_tokens,
        num_heads,
        num_splits,
        with_lse=True,
        with_prefill=prefill,
    )
    _require_workspace_bytes(raw, layout.total_bytes)
    lse_offset, lse_bytes = layout.lse
    lse_out = raw[
        lse_offset : lse_offset + num_query_tokens * num_heads * torch.float32.itemsize
    ].view(torch.float32)
    if prefill:
        _nvfp4_prefill(
            arch=arch,
            workspace=workspace_buffer,
            raw=raw,
            layout=layout,
            query_rows=query_rows,
            out_rows=out_rows,
            lse_out=lse_out,
            main=(main_flat, main_page_size, main_page_stride),
            extra=(extra_flat, extra_page_size, extra_page_stride),
            main_indices=main_indices,
            extra_indices=extra_indices,
            main_lengths=main_lengths,
            extra_lengths=extra_lengths,
            main_width=main_width,
            extra_width=extra_width,
            sinks=sinks,
            sink_tensor=sink_tensor,
            scale1=scale1,
            scale2=scale2,
            max_q_len=max_q_len,
        )
        return out
    partial_o, partial_lse = _partial_views(
        raw, out_rows, num_query_tokens, num_heads, num_splits
    )
    if num_splits == 1:
        # The decode member writes the final output and base-2 LSE directly.
        partial_lse = lse_out
        lse_partial_scale = lse_scale
        partial_o_tiles = out_rows.view(num_query_tokens, num_heads, 1, _HEAD_DIM)
    else:
        lse_partial_scale = 1.0
        partial_o_tiles = partial_o.view(
            num_query_tokens, num_heads, num_splits, _HEAD_DIM
        )

    assert plan is not None  # a BF16 query runs the decode family
    _require_nvfp4_variant(plan.variant, arch=arch)
    if num_splits > 1:
        _require_nvfp4_variant(_NVFP4_VARIANT_MERGE, arch=arch)
    main_g4d, main_g4f = _nvfp4_gather4_views(main_flat)
    extra_g4d, extra_g4f = _nvfp4_gather4_views(extra_flat)

    values: dict[str, Any] = {
        "Q": query_rows,
        "q_rows": query_rows,
        "partial_O_tiles": partial_o_tiles,
        "main_cache": main_flat,
        "extra_cache": extra_flat,
        "main_cache_g4d": main_g4d,
        "main_cache_g4f": main_g4f,
        "extra_cache_g4d": extra_g4d,
        "extra_cache_g4f": extra_g4f,
        "main_indices": main_indices,
        "extra_indices": extra_indices,
        "main_lengths": main_lengths if main_lengths is not None else main_indices,
        "extra_lengths": extra_lengths if extra_lengths is not None else main_indices,
        "sinks": sink_tensor,
        "bmm1_scale": scale1,
        "bmm2_scale": scale2,
        "partial_O": partial_o,
        "partial_lse": partial_lse,
        "O": out_rows,
        "lse_out": lse_out,
        "num_query_tokens": num_query_tokens,
        "num_heads": num_heads,
        "num_head_tiles": plan.num_head_tiles,
        "num_splits": num_splits,
        "num_main_tiles": plan.num_main_tiles,
        "tiles_per_split": plan.tiles_per_split,
        "total_tiles": plan.total_tiles,
        "main_width": main_width,
        "extra_width": extra_width,
        "main_index_stride": int(main_indices.stride(0)),
        "extra_index_stride": int(extra_indices.stride(0)),
        "has_main_lengths": int(main_lengths is not None),
        "has_extra_lengths": int(extra_lengths is not None),
        "main_page_shift": _log2_page_size(main_page_size, name="swa_kv_cache"),
        "extra_page_shift": _log2_page_size(
            extra_page_size, name="compressed_kv_cache"
        ),
        "main_page_stride": int(main_page_stride),
        "extra_page_stride": int(extra_page_stride),
        "has_sinks": int(sinks is not None),
        "heads_per_cta": plan.merge_heads_per_cta,
        "tile_n": plan.tile_n,
        "o_chunks": plan.o_chunks,
        "lse_partial_scale": float(lse_partial_scale),
        "lse_scale": lse_scale,
    }
    launcher = _Launcher(
        arch=arch,
        workspace=workspace_buffer,
        raw=raw,
        values=values,
    )
    launches = [launcher.variant(plan.variant, grid=(plan.grid, 1, 1))]
    if num_splits > 1:
        # Split merge: one CTA per (token, heads_per_cta heads) over partial_O / partial_lse.
        launches.append(launcher.variant(_NVFP4_VARIANT_MERGE, grid=plan.merge_grid))
    launcher.run(*launches)
    return out


def _nvfp4_pool_row_views(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row views of a flat NVFP4 pool (``[pages, page_stride]`` uint8) for the prefill bodies' tensor maps: the
    352-byte data rows and the 32-byte scale rows, both on a 32-byte pitch (token ``t`` of page ``p`` is data row
    ``p * page_stride / 32 + 11 t`` and scale row ``p * page_stride / 32 + 11 * page_size + t``).  The data rows
    overlap and must stay inside the storage."""
    raw = flat.reshape(-1)
    n = raw.numel()
    data = raw.as_strided(
        ((n - _NVFP4_DATA_BYTES) // _NVFP4_ROW_PITCH + 1, _NVFP4_DATA_BYTES),
        (_NVFP4_ROW_PITCH, 1),
    )
    scales = raw.as_strided(
        (n // _NVFP4_ROW_PITCH, _NVFP4_SF_BYTES), (_NVFP4_ROW_PITCH, 1)
    )
    return data, scales


def _nvfp4_pool_geometry(
    page_size: int, page_stride: int, *, name: str
) -> tuple[int, int, int]:
    """``(page_log2, pitch_units, footer_units)`` of a pool for the prefill bodies: the page size as a shift, the
    page stride and the offset of the page's scale rows in 32-byte units."""
    if page_stride % _NVFP4_ROW_PITCH:
        raise ValueError(
            f"{name}: page stride {page_stride} B must be a multiple of {_NVFP4_ROW_PITCH}"
        )
    return (
        _log2_page_size(page_size, name=name),
        page_stride // _NVFP4_ROW_PITCH,
        (_NVFP4_DATA_BYTES // _NVFP4_ROW_PITCH) * page_size,
    )


def cake_dsv4_nvfp4_pack_query(
    query: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Pack a BF16 query ``[..., num_heads, 512]`` to the NVFP4 row format the
    NVFP4 prefill bodies take: ``uint8 [..., num_heads, 384]``, each (token,
    head) row the cache's token record (224 bytes of E2M1 NoPE codes, 128 bytes
    of BF16 RoPE dims, then the 28 E4M3 scale factors of the 16-dim NoPE groups,
    ``e4m3(amax / 6)``, and 4 pad bytes), the same quantizer as
    :func:`nvfp4_quantize_pack_sparse_mla_cache`. ``out`` (contiguous uint8 of
    that shape on the query's device) receives the rows; without it a tensor is
    allocated. One kernel launch, capturable in a CUDA graph with ``out``.
    """
    from ._sparse_mla_sm120._execution import (
        get_sparse_mla_dsv4_nvfp4_cache_ops_module,
    )

    if query.dtype != torch.bfloat16 or query.ndim < 2:
        raise ValueError(
            f"query must be BF16 [..., num_heads, {_HEAD_DIM}], got {query.dtype} "
            f"{tuple(query.shape)}"
        )
    if int(query.shape[-1]) != _HEAD_DIM or not query.is_contiguous():
        raise ValueError(
            f"query must be contiguous [..., num_heads, {_HEAD_DIM}], got shape "
            f"{tuple(query.shape)}"
        )
    shape = (*query.shape[:-1], _NVFP4_TOKEN_BYTES)
    if out is None:
        out = torch.empty(shape, dtype=torch.uint8, device=query.device)
    elif (
        out.dtype != torch.uint8
        or tuple(out.shape) != shape
        or not out.is_contiguous()
        or out.device != query.device
    ):
        raise ValueError(
            f"out must be a contiguous uint8 {shape} tensor on {query.device}, "
            f"got {out.dtype} {tuple(out.shape)} on {out.device}"
        )
    rows = query.numel() // _HEAD_DIM
    if rows:
        get_sparse_mla_dsv4_nvfp4_cache_ops_module().sparse_mla_sm120_nvfp4_quantize_pack(
            query.reshape(rows, 1, _HEAD_DIM), out.view(rows, 1, 1, _NVFP4_TOKEN_BYTES)
        )
    return out


def _nvfp4_prefill(
    *,
    arch: str,
    workspace: torch.Tensor,
    raw: torch.Tensor,
    layout: WorkspaceLayout,
    query_rows: torch.Tensor,
    out_rows: torch.Tensor,
    lse_out: torch.Tensor,
    main: tuple[torch.Tensor, int, int],
    extra: tuple[torch.Tensor, int, int],
    main_indices: torch.Tensor,
    extra_indices: torch.Tensor,
    main_lengths: Optional[torch.Tensor],
    extra_lengths: Optional[torch.Tensor],
    main_width: int,
    extra_width: int,
    sinks: Optional[torch.Tensor],
    sink_tensor: torch.Tensor,
    scale1: torch.Tensor,
    scale2: torch.Tensor,
    max_q_len: Optional[int],
) -> None:
    """Launch one of the NVFP4 prefill bodies on ``num_query_tokens`` rows.

    :func:`_nvfp4_prefill_route` picks the body and its build. Every body
    reads the caller's packed NVFP4 query ``query_rows`` (uint8 ``[T, H,
    384]``) through a data map and a scale-factor map, the pools through 32-byte-pitch data and scale
    row views, and writes the final BF16 output and base-2 LSE directly. A
    table without a caller length vector gets one filled in the workspace
    (every entry the table width).
    """
    num_query_tokens, num_heads = int(query_rows.shape[0]), int(query_rows.shape[1])
    variant, grid_x = _nvfp4_prefill_route(
        num_heads=num_heads,
        num_query_tokens=num_query_tokens,
        main_width=main_width,
        extra_width=extra_width,
        main_index_stride=int(main_indices.stride(0)),
        sm_count=_device_sm_count(query_rows.device),
        arch=arch,
    )
    _require_nvfp4_variant(variant, arch=arch)
    if variant in _NVFP4_PREFILL_VECTOR_INDEX_VARIANTS:
        # Refused before any launch (and before the length fill below), as the
        # BF16 / FP8 route refuses such tables in resolve_cake_dsv4_sparse_metadata.
        _check_index_row_alignment(main_indices, "sparse_indices")
        if extra_width:
            _check_index_row_alignment(extra_indices, "extra_sparse_indices")

    q_packed = query_rows

    lens_offset, lens_bytes = layout.prefill_lens
    half = lens_bytes // 2

    def lengths(given: Optional[torch.Tensor], width: int, slot: int) -> torch.Tensor:
        if given is not None:
            return given
        start = lens_offset + slot * half
        view = raw[start : start + num_query_tokens * torch.int32.itemsize].view(
            torch.int32
        )
        view.fill_(width)
        return view

    main_lens = lengths(main_lengths, main_width, 0)
    extra_lens = lengths(extra_lengths, extra_width, 1) if extra_width else main_lens

    main_flat, main_page_size, main_page_stride = main
    main_data, main_sf = _nvfp4_pool_row_views(main_flat)
    main_page_log2, main_pitch_units, main_footer_units = _nvfp4_pool_geometry(
        main_page_size, main_page_stride, name="swa_kv_cache"
    )
    if extra_width:
        extra_flat, extra_page_size, extra_page_stride = extra
        extra_data, extra_sf = _nvfp4_pool_row_views(extra_flat)
        extra_page_log2, extra_pitch_units, extra_footer_units = _nvfp4_pool_geometry(
            extra_page_size, extra_page_stride, name="compressed_kv_cache"
        )
    else:
        extra_data, extra_sf = main_data, main_sf
        extra_page_log2, extra_pitch_units, extra_footer_units = (
            main_page_log2,
            main_pitch_units,
            main_footer_units,
        )

    values: dict[str, Any] = {
        "Q": q_packed,
        "SWA_cache": main_data,
        "SWA_sf": main_sf,
        "compressed_KV_cache": extra_data,
        "compressed_KV_sf": extra_sf,
        "O": out_rows,
        "LSE": lse_out,
        "swa_indices": _flat_index_rows(main_indices),
        "compressed_indices": _flat_index_rows(extra_indices),
        "sparse_topk_lens": main_lens,
        "extra_topk_lens": extra_lens,
        "sinks": sink_tensor,
        "bmm1_scale": scale1,
        "bmm2_scale": scale2,
        "num_heads": num_heads,
        "swa_index_stride": int(main_indices.stride(0)),
        "compressed_index_stride": int(extra_indices.stride(0)),
        "sparse_topk_lens_offset": 0,
        "num_query_tokens": num_query_tokens,
        "sparse_topk": main_width + extra_width,
        "has_sinks": int(sinks is not None),
        # Work items are query tokens; the body maps item (x, z) to token z * max_q_len + x
        # and the launch keeps every item in x, so max_q_len only has to be positive.
        "total_work_items": num_query_tokens,
        "max_q_len": int(max_q_len) if max_q_len is not None else num_query_tokens,
        "swa_page_log2": main_page_log2,
        "swa_pitch_units": main_pitch_units,
        "swa_footer_units": main_footer_units,
        "compressed_page_log2": extra_page_log2,
        "compressed_pitch_units": extra_pitch_units,
        "compressed_footer_units": extra_footer_units,
        "swa_width": main_width,
        "compressed_width": extra_width,
    }
    launcher = _Launcher(arch=arch, workspace=workspace, raw=raw, values=values)
    launcher.run(launcher.variant(variant, grid=(grid_x, 1, 1)))


def _nvfp4_qb_group_ids(blocks: int) -> int:
    """The group items the ``breaks`` build launches ahead of its ``blocks`` block items: one per 8
    blocks, at most 16 (Cake ``group_ids``)."""
    return min(_ceil_div(blocks, 8), 16)


def _nvfp4_prefill_route(
    *,
    num_heads: int,
    num_query_tokens: int,
    main_width: int,
    extra_width: int,
    main_index_stride: int,
    sm_count: int,
    arch: str = "sm_103a",
) -> tuple[str, int]:
    """The NVFP4 prefill variant of one call and its grid (CTAs along x).

    * 128 heads: the two-CTA body, one cluster per query token; on sm_103a its
      ``v_unit_pairs`` build once the row selects more than 256 keys (items of
      more than two tiles: C4 and long-context C128 layers), the persistent
      build otherwise (SWA and C128 up to 16K context, within 2 % of each other
      there).
    * 16 / 32 heads on window-only rows (no extra table, a main table of at
      most 128 keys on a row stride of at least 128) with more query tokens
      than SMs: the query-block bodies, one item per block of 64 / H tokens.
      16 heads take the ``breaks`` build up to 1024 tokens (plus its group
      items), the default build above. 32 heads take the ``late_claim`` build
      up to 512 tokens once their blocks exceed 11 / 8 of the SMs, the default
      build otherwise. With at most as many tokens as SMs the blocks would leave
      most SMs idle, and the per-token body is faster.
    * Every other row of at most 32 heads: the per-token body's ``tmem_v``
      build, with ``early_gather`` at most as many tokens as SMs.
    * 33 to 64 heads: ``early_gather`` at most as many tokens as SMs, else
      ``q_prefetch`` on window-only rows and ``o_boxes`` on rows with a
      compressed selection or an extra table.
    """
    tokens = num_query_tokens
    if num_heads > _NVFP4_H64_PREFILL_MAX_HEADS:
        if (
            main_width + extra_width >= _NVFP4_H128_V_UNIT_PAIRS_MIN_TOPK
            and arch in _NVFP4_H128_V_UNIT_PAIRS_ARCHES
        ):
            return _NVFP4_VARIANT_H128_V_UNIT_PAIRS_PREFILL, 2 * tokens
        return _NVFP4_VARIANT_H128_PREFILL, 2 * tokens
    window_only = extra_width == 0 and main_width <= _NVFP4_QB_WINDOW
    one_item_per_cta = tokens <= sm_count
    if num_heads <= _NVFP4_TMEM_V_MAX_HEADS:
        if (
            num_heads in (16, 32)
            and window_only
            and main_index_stride >= _NVFP4_QB_WINDOW
            and not one_item_per_cta
        ):
            blocks = _ceil_div(tokens, 64 // num_heads)
            if num_heads == 16:
                if tokens <= _NVFP4_QB_BREAKS_MAX_TOKENS:
                    return (
                        _NVFP4_VARIANT_QB_H16_BREAKS_PREFILL,
                        blocks + _nvfp4_qb_group_ids(blocks),
                    )
                return _NVFP4_VARIANT_QB_H16_PREFILL, blocks
            num, den = _NVFP4_QB_LATE_CLAIM_MIN_BLOCKS_PER_SM
            if (
                den * blocks > num * sm_count
                and tokens <= _NVFP4_QB_LATE_CLAIM_MAX_TOKENS
            ):
                return _NVFP4_VARIANT_QB_H32_LATE_CLAIM_PREFILL, blocks
            return _NVFP4_VARIANT_QB_H32_PREFILL, blocks
        if one_item_per_cta:
            return _NVFP4_VARIANT_H64_TMEM_V_EARLY_GATHER_PREFILL, tokens
        return _NVFP4_VARIANT_H64_TMEM_V_PREFILL, tokens
    if one_item_per_cta:
        return _NVFP4_VARIANT_H64_EARLY_GATHER_PREFILL, tokens
    if window_only:
        return _NVFP4_VARIANT_H64_Q_PREFETCH_PREFILL, tokens
    return _NVFP4_VARIANT_H64_O_BOXES_PREFILL, tokens


def _flat_index_rows(table: torch.Tensor) -> torch.Tensor:
    """A ``[rows, width]`` int32 table (unit column stride, any row stride) as the 1-D run of storage its
    rows span: the prefill bodies read row ``t`` at ``t * index_stride`` themselves."""
    rows, width = int(table.shape[0]), int(table.shape[1])
    if table.is_contiguous():
        return table
    return table.as_strided(((rows - 1) * int(table.stride(0)) + width,), (1,))


def _nvfp4_gather4_views(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """TMA gather views of a flat NVFP4 pool (``[pages, page_stride]`` uint8) for the gather4 members: 32-byte-pitch
    int32 rows -- data rows (88 x int32, overlapping: token ``t`` of page ``p`` is row ``p * page_stride / 32 + 11 t``)
    and footer rows (8 x int32).  Mirrors the Cake members' ``_gather4_views``."""
    f32 = flat.view(torch.int32).reshape(-1)
    n_rows = f32.numel() // 8
    n_data_rows = (
        f32.numel() - 88
    ) // 8 + 1  # the overlapping 88-element rows must stay inside the storage
    return f32.as_strided((n_data_rows, 88), (8, 1)), f32.as_strided(
        (n_rows, 8), (8, 1)
    )


__all__ = [
    "KERNEL_METADATA_PARAMS",
    "NVFP4Plan",
    "QUERY_LAYOUT_PARAMS",
    "SparseMetadata",
    "WorkspaceLayout",
    "WorkspaceRequirement",
    "cake_dsv4_workspace_layout",
    "cake_dsv4_workspace_requirement",
    "cake_dsv4_workspace_reset",
    "canonical_arg_name",
    "get_cake_dsv4_workspace_bytes",
    "is_bindable_arg",
    "resolve_cake_dsv4_sparse_metadata",
    "run_cake_dsv4",
    "run_cake_dsv4_nvfp4",
]
