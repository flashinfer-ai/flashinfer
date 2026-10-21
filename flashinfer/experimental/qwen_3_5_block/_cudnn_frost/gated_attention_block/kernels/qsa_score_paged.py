# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``qsa_score_paged`` -- the decode indexer's scorer over the PAGED compressed-key cache; its standalone selector is the host entry
``gated_attention_block/qsa_select_paged.py``.

A decode step of a block-sparse (``QsaSpec``) layer scores each of its ``S_q`` query rows per sequence against the compressed
keys of the complete 4-token blocks the row can see, and the top-k of those scores is the row's block list.  The serving
stacks keep the compressed keys PAGED on the main block table (one compressed row per 4 tokens, ``page_size / 4`` rows per
page), so this kernel reads them there -- once, in page order, never through a dense per-sequence copy::

    pos_t   = kv_lens[b] - S_q + s                       (row s of sequence b; t = b * S_q + s)
    n_vis_t = clamp(floor((pos_t + 1) / 4), 0, n_blocks_max)
    logits[t, j] = scale * sum_h relu(qi[t, h] . kbar[b, j])        for j < n_vis_t, kbar[b, j] = cache[table[b, j // R], j % R]
    counts[t]     = n_vis_t                                          (R = page_size / 4 compressed rows per page)
    block_lens[t] = min(top_k, n_vis_t)                              (optional output)

The ``logits [T, n_blocks_max]`` fp32 + ``counts [T]`` int32 pair is the input contract of the decode top-k that follows
(the radix kernel today, a FROST top-k later): the top-k reads row ``t``'s columns ``[0, counts[t])`` -- nothing else of the
row is written here -- and pads its output with ``-1`` past ``min(top_k, counts[t])``.  The host entry
``qsa_select_paged.qsa_select_decode_paged`` (a DSL-free module, so its import path resolves on every installed DSL) pairs the
scorer with the deterministic radix top-k (the smaller block id wins a tie at the cutoff, the selected ids come back in ASCENDING
order, so two calls are bitwise) and no torch prologue: the positions and the counts are derived on device by this kernel.

**The form: a warp-level MMA over the (s, h) columns, the keys streamed by ``cp.async``.**  A decode step has 1-4 query rows per
sequence, 4 indexer heads each: the ``S_q x H_i`` (s, h) rows pad to the M = 16 of one ``mma.sync.m16n8k16`` (bf16 in, fp32
accumulate), so one MMA per 8 keys per 16 of the 128 D scores EVERY column at once -- the work is flat in ``S_q`` and the compute
term is negligible against the bytes (131 k MAC per 16 KB of keys).  The padded query tile is staged ONCE per CTA in shared memory
(``cp.async``, every thread a 16-B granule, zero-fill for the padding rows) behind the kernel's one CTA barrier and read into A
fragments; the keys stream through a per-warp ring of ``cp.async`` groups -- page part by page part, whole 128-B lines per
request, an L2 evict-first policy (the keys pass through L2 once; the logits' write-combining lines and the table stay), a
zero-filled copy for a dead page -- into B fragments read with ``ld.shared.v4``; relu, the head sum and the scale run on the fp32
accumulator fragment.  No TMA, no TMEM, no cluster: the first key byte is requested ONE memory round trip after the launch
(``kv_lens``, the table words and the query tile are all in flight together), nothing holds a register while it waits, and the
kernel's roofline term is the HBM bytes of the compressed rows it streams (plus the fp32 logits it writes).

Work decomposition (every constant below is a module constant or a compile key; nothing is inferred at launch):

* grid ``(ceil(n_blocks_max / blocks_per_cta), B)``, CTA = 4 warps = 128 threads; a warp owns ``keys_per_warp = 32 x
  units_per_warp`` consecutive compressed rows (``units_per_warp`` in {1, 2, 4, 8} is a performance knob: the same function;
  ``blocks_per_cta`` = 128 / 256 / 512 / 1024; the default 1 = the best point of every measured cell), streamed as n-tiles of 8
  keys, ``ring_depth`` (2 / 4 / 8, default 4: a performance knob) n-tiles in flight;
* **the K permutation** shared by A and B (so the product is unchanged): lane ``(g = lane / 4, q = lane % 4)`` reads the 16-byte
  granules ``c = q + 4 j`` (``j = 0..3``) of its row, and k-step ``kappa = 2 j + r`` takes granule ``q + 4 j``'s words ``2 r`` (logical k
  ``2q, 2q + 1``) and ``2 r + 1`` (logical k ``2q + 8, 2q + 9``): a bijection ``(kappa, k) <-> D``, so the fragment registers ARE the
  granules' words (no shuffle, no ``ldmatrix``).  A per m-tile: ``reg0 / reg1`` = rows ``g`` / ``g + 8`` at word ``2 r``, ``reg2 /
  reg3`` the same rows at word ``2 r + 1``; B per n-tile: key ``n = g``'s words ``2 r`` / ``2 r + 1``; C: lane ``(g, q)`` holds rows
  ``g`` / ``g + 8`` for keys ``2q, 2q + 1``;
* **the row map** (``qsa_select_paged.query_row_map``): a query's heads pad to ``P = next_pow2(H_i)`` rows; ``P' = min(P, 16)``
  of them share a 16-row m-tile with ``QPT = 16 / P'`` queries, a query wider than a tile spans ``TPQ = P / 16`` tiles
  (``m_tiles = ceil(S_q x P / 16) <= 4``, the ``MAX_QUERY_ROWS`` envelope); with ``P' >= 2`` the two rows a lane holds are heads
  ``2 gi`` / ``2 gi + 1`` of ONE query (``gi = g % G``, ``G = P' / 2``) and the remaining heads sit in the ``G`` consecutive ``g``
  values = lanes ``xor 4, 8, 16``: the head sum is one in-thread add plus ``log2(G)`` butterfly shuffles (H_i 4: ONE), plus the
  in-thread sums over a query's tiles at ``P > 16``; with ``P' = 1`` the two rows are two queries and nothing is summed; a padding
  row (``h >= H_i`` inside the power of two, ``s >= S_q``) is a ZERO row of the staged tile (its copy is zero-filled);
* the prologue, every thread: ``kv_lens[b]``; the table words of the first ``2 x ring_depth`` n-tiles (per lane the word of ITS key,
  ``table[b, min(k, n_blocks_max - 1) / R]``: 1-4 distinct words per instruction, one sector); the query tile's copies (2 per m-tile,
  one ``cp.async`` group); then -- live warps only (``k_warp0 < n_vis_max``, warp-uniform) -- the first ``ring_depth`` n-tiles of
  keys (one group each); ``cp.async.wait_group`` for the query tile's group; the CTA barrier; the A fragments (8 ``ld.shared.v4``
  per m-tile per lane);
* the mainloop, live warps, in batches of ``mma_batch`` n-tiles (4 at ``units_per_warp`` 1, 2 above: the HMMA chains of a batch
  interleave) -- per n-tile: the table word of n-tile ``i + 2 x ring_depth``; ``cp.async.wait_group`` for group ``i``; ``bar.warp.sync``
  (the other lanes' copies visible, the previous slot's reads ordered before its reuse); four ``ld.shared.v4`` (the B fragments);
  the copies of n-tile ``i + ring_depth`` into the slot read one step earlier (the ring has ``ring_slots = min(ring_depth + 1,
  n_tiles)`` slots: one spare only where a mainloop copy exists); then the
  batch's ``8 x m_tiles`` MMAs per n-tile, relu per (s, h) element (fp32), the head sum, the scale; the lane with ``gi == 0`` (every
  lane at ``P' = 1``) stores query ``s``'s two adjacent keys ``8 i + 2q, + 1`` -- one ``st.global.v2.f32`` when both are visible to
  that row, ``st.global.f32`` when only the first is, nothing otherwise;
* a key copy = per page part (``min(R, 8)`` rows of one page) ONE shuffle (the part's table word from the lane that read it -- a
  constant source lane), one validity check, one ``mul.wide`` page address, then ``rows / 2`` ``cp.async.cg`` of 16 B per lane at
  distinct address registers (instruction ``e`` copies rows ``2 e`` by lanes 0-15 and ``2 e + 1`` by lanes 16-31: eight consecutive
  lanes per 128-B half, whole lines per request); a part on a ``-1`` / out-of-range page is ZERO-FILLED (``src-size 0``: nothing of the
  cache is read; the address is page 0, a real row); a row at or past the visible count or the capacity reads a real row (its table
  word was clamped to the capacity) and is never stored: its products stay in its own accumulator column;
* the op order: the MMA's fp32 accumulation over the 128 D per (s, h), ``relu``, then ``((h0 + h1) + (h2 + h3))`` and the tile sums,
  then ``x scale`` -- fixed, so two launches are bitwise; the fp64 reference's budget is met by the fp32 accumulation.

**SMEM buffer table** (`frost-kernels.md` section 5; no descriptor reads either buffer -- the swizzle's one job is the banks).  ONE
dynamic allocation sized by the DSL from the two ``cutlass.Array`` declarations (no ``smem=`` at the launch): ``sK`` at offset 0, ``sA``
at ``4 x ring_slots x 2 KiB``; total ``4 x ring_slots x 2 KiB + m_tiles x 4 KiB`` with ``ring_slots = min(ring_depth + 1, n_tiles)`` =
**36 KiB at the defaults** (units 1 = 4 n-tiles, depth 4: 4 slots; one m-tile), 44 KiB from units 2 (5 slots), 36 / 68 / 76 KiB at
depth 8 for units 1 / 2 / 4 and 8 (4 / 8 / 9 slots); every size a multiple of 2 KiB, both buffers 128-B aligned:

=======  =====  ========================  =============================================  ==============================================  ==========================================================================
name     dtype  elems                     written by                                     read by                                         per-lane stride / swizzle
=======  =====  ========================  =============================================  ==============================================  ==========================================================================
``sK``   Int32  ``4 x ring_slots x 512``  this warp's ``cp.async.cg`` (16 B per lane:    this warp's ``ld.shared.v4``: lane ``(g, q)``   the 16-B granule ``c`` of row ``r`` sits at ``r x 256 + (c ^ (4 x (r & 1))) x 16``:
                (a per-warp ring of       lanes 0-15 row ``2 e``, 16-31 row ``2 e + 1``)  reads row ``g``'s granules ``q + 4 j``          a quarter-warp of a write (one row's eight granules) or of a read (rows ``2 m``,
                ``ring_slots`` slots of                                                                                                   ``2 m + 1``, four granules each) spans all 32 banks: ONE wavefront per quarter-
                8 rows x 256 B; 32 KiB                                                                                                    warp, the minimum for 128-bit accesses (an unswizzled layout puts the eight rows'
                at the defaults)                                                                                                          granule ``c`` on the same 16 banks: 2-way)
``sA``   Int32  ``m_tiles x 1024`` (the   every thread's ``cp.async.cg`` (2 per m-tile:  every warp's ``ld.shared.v4``: lane ``(g, q)``  the same ``c ^ (4 x (row & 1))`` layout and the same one-wavefront-per-quarter-warp
                padded query tile, 16      thread ``t`` granule ``t % 16`` of rows ``t /  reads rows ``g`` / ``g + 8``'s granules ``q +   argument (the copy's 16 consecutive lanes cover one row = two 128-B lines)
                rows x 256 B per m-tile)   16`` and ``8 + t / 16``)                       4 j``
=======  =====  ========================  =============================================  ==============================================  ==========================================================================

**Barrier table** (`frost-kernels.md` section 4) -- no mbarrier, no TMA, no tcgen05, no cluster:

====================  ======================================  ==================================  ===================  ===============================================================================
name                  kind                                    issuing lanes                       init / expected      where
====================  ======================================  ==================================  ===================  ===============================================================================
``bar.sync 0``        the CTA barrier (``cute.arch.barrier``)  all 128 threads (4 warps x 32),     128 == the CTA size  after every thread's ``cp.async.wait_group`` of the query tile's group (a live warp
                                                              once per CTA: SUM = 128                                  allows its key groups to stay in flight, a dead warp waits for 0) and BEFORE the
                                                                                                                       per-warp live branch: every thread of every CTA reaches it exactly once
``cp.async`` groups   per-thread completion (commit / wait)    the issuing thread only              -- (no count)        the query tile: one group per thread, waited before the CTA barrier; the keys: one
                                                                                                                       group per n-tile per thread, group ``i`` waited at step ``i`` (``min(n_tiles, depth
                                                                                                                       + i) - i - 1`` younger groups may stay pending)
``bar.warp.sync``     the warp barrier (``cute.arch.sync_warp``)  all 32 lanes, once per n-tile     32 == the warp       inside the warp-uniform live branch only: after the wait of group ``i``, before the
                      with memory ordering                                                                              fragment reads of slot ``i`` and the copies into the slot read at step ``i - 1``
``mma.sync.aligned``  warp-collective                          all 32 lanes, ``8 x m_tiles`` per   32                   inside the live branch only (``k_warp0 < n_vis_max``: both operands warp-uniform)
                                                              n-tile
``shfl.sync.bfly``    warp-collective (mask ``0xffffffff``)    all 32 lanes, ``2 x stages`` per     32                   the same branch; the per-lane store predicates come AFTER the shuffles
                                                              query group per n-tile
``shfl.sync.idx``     warp-collective (mask ``0xffffffff``)    all 32 lanes, one per page part      32                   the same branch (the copies' table word from a constant source lane)
====================  ======================================  ==================================  ===================  ===============================================================================

PDL sites (under the ``use_pdl`` compile key, the launch attribute set with it): ``griddepcontrol.wait`` is the kernel's first
instruction -- ahead of the ``kv_lens`` / table / ``qi`` reads and so of every cache read --, ``griddepcontrol.launch_dependents``
follows the prologue's copies, BEFORE the counts and the logits are written.  That early trigger puts an obligation on the dependent
grid: a kernel launched with the programmatic-stream-serialization attribute behind this one may start while this one still runs and
MUST issue its own ``griddepcontrol.wait`` before it touches anything this kernel reads or writes (``logits``, ``counts``,
``block_lens``, and the inputs it could overwrite); the wait returns only after this grid has completed and flushed.  The standalone
selector (``qsa_select_decode_paged``) launches the radix top-k WITHOUT the attribute, so the pair is inert there (the top-k starts
after the scorer completes, as any stream-ordered launch); it matters in the fused decode chain, whose nodes carry the attribute and
wait.  Without PDL (``use_pdl=False``, the standalone default) neither instruction is traced and the kernel runs on any CuTe DSL
device from sm_80 (``mma.sync.m16n8k16``, ``cp.async``).

Degenerate inputs, each with its handling: ``kv_lens[b] == 0`` (a padding sequence) -> every ``pos_t < 0``, ``n_vis = 0``: no
logits column written, ``counts = block_lens = 0``, no warp of the sequence copies a key (its ``qi`` rows and table words ARE read
-- valid memory, the round trip that keeps the critical path at two -- and its threads stage the query tile and reach the barrier);
``kv_lens[b] < S_q`` (a contract violation) -> the rows at a negative position get ``n_vis = 0`` likewise; ``n_vis`` inside a page
-> the rows past ``n_vis_max`` of a live n-tile read a real row and store nothing (their products stay in their own accumulator
column); a ``-1`` table entry (or one at or past ``num_pages``) for a VISIBLE block -> the part's copy is zero-filled: logit exactly
0, still written -- the sparse core's dead-page convention, never an out-of-bounds read; a dead page past ``n_vis_max`` may be
consulted (its word read, its copy zero-filled or a real page read) and is never stored; fewer than ``top_k`` visible blocks (the
identity region) -> ``block_lens = n_vis``, the paired top-k lists every visible id ascending then ``-1``; ``n_vis`` past
``n_blocks_max`` -> clamped to the capacity (the host checks the table's reach ``max_pages x R >= n_blocks_max``; a lane past the
capacity indexes the table at ``n_blocks_max - 1``).  Refused at the host before any launch: a table narrower than the capacity, a
page size that is not a power of two (at least 8), ``S_q x next_pow2(H_i)`` past the envelope (``MAX_QUERY_ROWS``), a head dim
other than 128, a cache whose rows are not contiguous inside a page or whose page stride is 2 GiB or more, misaligned or
non-unit-stride operands, a mixed dtype pair, and -- by name, a typed ``NotImplementedError`` -- the e4m3 pair (the fp8 indexer
cache's read arm is a follow-up of this kernel: the bf16 pair is served).

Bytes per launch (the SOL denominator): ``sum_b n_vis_max(b) x 256 B`` of compressed rows read once; ``sum_t n_vis_t x 4 B`` of
logits written; the query tile per live CTA (``m_tiles x 4 KiB``, L2-resident) and one table word per page --
``qsa_select_paged.moved_bytes``.

Rule 7: this module imports the DSL at import time like its neighbours, so the documented host entry lives in the DSL-free
front door ``gated_attention_block/qsa_select_paged.py`` -- ``qsa_select_decode_paged`` gates the DSL floor, the ``sm_107a``
target on cc 10.7 and the radix top-k's cc 9.0 BEFORE importing this module -- and the kernel's host-side facts (the row map,
the compile-key validator, the knob defaults, the byte model) are defined there and imported here, never the other way round
at module scope.  :func:`compile_qsa_score_paged`, the kernel-level entry a caller may reach directly, gates the floor and the
target again BEFORE any compile, so a too-old DSL reads as a version decline there too; the tests skip below the floor.
"""

import hashlib
from pathlib import Path
from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached, template_key
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import compute_capability, current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.mma import mma_m16n8k16_f32
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import opaque_f32_zero, opaque_i32
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global, st_global_v2
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

# the kernel's host-side facts live in the DSL-free front door (Rule 7: a declaration reads them without importing this module)
from ..qsa_select_paged import (
    BLOCK_SIZE,
    CHUNKS_PER_LANE,
    INDEX_HEAD_DIM,
    K_STEPS,
    KEYS_PER_NTILE,
    MMA_M,
    NTILES_IN_FLIGHT,
    NTILES_PER_UNIT,
    THREADS,
    TOP_K_KERNEL_MAX,
    _ceil_div,
    _check_i32_1d,
    _kbar_rows_view,
    blocks_per_cta,
    keys_per_warp,
    query_row_map,
    validate_params,
)

__all__ = [
    "QsaScorePagedRecipe",
    "dsl_decline_reason",
    "compile_qsa_score_paged",
    "run_qsa_score_paged",
]

_FAKE_STREAM = None
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins
# it with a compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers
# every other ``.py`` the kernel is traced from.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
_CHUNK_BYTES = 16  # one 16-byte granule of a row: 8 bf16 = 4 words = one cp.async / ld.global.v4 / ld.shared.v4
_WORDS_PER_CHUNK = 4
_ROW_BYTES = INDEX_HEAD_DIM * 2  # 256 B per compressed row
_SLOT_BYTES = KEYS_PER_NTILE * _ROW_BYTES  # one n-tile of 8 rows in the ring: 2 KiB
_SLOT_ELEMS = _SLOT_BYTES // 4
_A_TILE_BYTES = MMA_M * _ROW_BYTES  # one m-tile of the padded query tile: 16 rows x 256 B
_A_TILE_ELEMS = _A_TILE_BYTES // 4


# --------------------------------------------------------------------------------------------- the kernel


@cute.jit
def _page_word(tb_row: cutlass.Int64, k: cutlass.Int32, n_blocks_max: cutlass.Int32, rows_per_page: cutlass.Constexpr[int]) -> cutlass.Int32:
    """The table word of compressed row ``k`` (a row past the capacity indexes a real word: ``n_blocks_max - 1``)."""
    kc = k if k < n_blocks_max else n_blocks_max - cutlass.Int32(1)
    pg = kc >> cutlass.Int32(rows_per_page.bit_length() - 1)
    return ld_global(tb_row + pg.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Int32)


@cute.jit
def _mul_wide_s32(a: cutlass.Int32, b: cutlass.Int32) -> cutlass.Int64:
    """``mul.wide.s32``: one IMAD.WIDE for a 32 x 32 -> 64-bit product (a page index x a page stride in bytes)."""
    return inline_ptx("mul.wide.s32 $0, $1, $2;", write_only_types=[cutlass.Int64], read_only_args=[a, b])


@cute.jit
def _copy_ntile(
    dst_lane: cutlass.Int64,
    src_lane_off: cutlass.Int32,
    kb_base: cutlass.Int64,
    kb_page_bytes: cutlass.Int32,
    word: cutlass.Int32,
    num_pages: cutlass.Int32,
    k_tile0: cutlass.Int32,
    policy: cutlass.Int64,
    rows_per_page: cutlass.Constexpr[int],
):
    """The n-tile's 8 compressed rows into one ring slot, PAGE PART by page part (``min(R, 8)`` rows of one page): per part ONE
    shuffle (the part's table word from the lane that read it -- a constant source lane), one validity check, one ``mul.wide``
    page address; then ``rows / 2`` ``cp.async.cg`` of 16 B per lane at immediate offsets -- instruction ``e`` copies rows ``2 e``
    (lanes 0-15) and ``2 e + 1`` (lanes 16-31), each row's two 128-B halves by eight consecutive lanes (whole lines per request).
    A part on a ``-1`` / out-of-range page is ZERO-FILLED (``src-size 0``: nothing of the cache is read; the address is page 0, a
    real row, either way).  A row at or past the visible count or the capacity reads a real row (its table word was clamped to
    the capacity) and is never stored: its products stay in its own accumulator column."""
    rows_pp = min(rows_per_page, KEYS_PER_NTILE)
    zero_i = cutlass.Int32(0)
    for part in cutlass.range_constexpr(KEYS_PER_NTILE // rows_pp):
        w = cute.arch.shuffle_sync(word, 4 * part * rows_pp)  # the word of the part's first row sits in lanes 4 r .. 4 r + 3
        ok = (w >= zero_i) & (w < num_pages)
        page_c = w if ok else zero_i
        size = cutlass.Int32(_CHUNK_BYTES) if ok else zero_i
        row_off = ((k_tile0 + cutlass.Int32(part * rows_pp)) & cutlass.Int32(rows_per_page - 1)) * cutlass.Int32(
            _ROW_BYTES
        )  # the part's first row inside its page
        src = kb_base + _mul_wide_s32(page_c, kb_page_bytes) + (row_off + src_lane_off).to(cutlass.Int64)
        dst = dst_lane + cutlass.Int64(part * rows_pp * _ROW_BYTES)
        for e in cutlass.range_constexpr(rows_pp // 2):
            # every cp.async gets its OWN address registers (an opaque offset, never base + immediate): four back-to-back LDGSTS on one
            # base register serialised on the register scoreboard under HBM latency (2x at B 32 x 128K, measured 2026-10-21)
            off_e = opaque_i32(cutlass.Int32(e * 2 * _ROW_BYTES)).to(cutlass.Int64)
            inline_ptx(
                "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2, $3;", write_only_types=[], read_only_args=[dst + off_e, src + off_e, size, policy]
            )


@cute.jit
def _read_key_frags(sK, slot_elems: cutlass.Int32, g: cutlass.Int32, qd: cutlass.Int32):
    """Lane ``(g, q)``'s B fragments of the n-tile in the ring slot: row ``g``'s granules ``q + 4 j`` (one ``ld.shared.v4`` each,
    at the swizzled granule ``c ^ (4 x (g & 1))``) -- the 16 words the 8 k-steps consume."""
    swz = (g & cutlass.Int32(1)) * cutlass.Int32(4)
    row = slot_elems + g * cutlass.Int32(_ROW_BYTES // 4)
    out = []
    for j in cutlass.range_constexpr(CHUNKS_PER_LANE):
        c_swz = (qd + cutlass.Int32(4 * j)) ^ swz
        v = sK.load(row + c_swz * cutlass.Int32(_WORDS_PER_CHUNK), vector_size=4, alignment=16).to_elements()
        for x in cutlass.range_constexpr(_WORDS_PER_CHUNK):
            out.append(v[x])
    return out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]


@cute.jit
def _copy_query_tile(
    sA_tile: cutlass.Int64,
    q_base: cutlass.Int64,
    q_s0: cutlass.Int64,
    q_s1: cutlass.Int64,
    t0: cutlass.Int32,
    tidx: cutlass.Int32,
    t: cutlass.Constexpr[int],
    s_q: cutlass.Constexpr[int],
    h_i: cutlass.Constexpr[int],
):
    """m-tile ``t`` of the padded (s, h) query tile into the CTA's SMEM staging: 16 rows x 256 B by the 128 threads in two
    ``cp.async.cg`` instructions (thread ``tidx`` copies granule ``tidx % 16`` of rows ``tidx / 16`` and ``8 + tidx / 16``: sixteen
    consecutive lanes per row, whole lines); a padding row (``h >= H_i`` inside the power of two, ``s >= S_q``) is ZERO-FILLED
    (``src-size 0``; its address is a real row).  The destination granule is ``c ^ (4 x (row & 1))``, the fragment reads' swizzle."""
    rm = query_row_map(s_q, h_i)
    zero_i = cutlass.Int32(0)
    s_q_i = cutlass.Int32(s_q)
    h_i_i = cutlass.Int32(h_i)
    c = tidx & cutlass.Int32(15)
    g = tidx >> cutlass.Int32(4)  # the row within the half: 0..7
    for half in cutlass.range_constexpr(2):
        if cutlass.const_expr(rm.tile_rows == 1):
            s_row = cutlass.Int32(t * rm.queries_per_tile + 8 * half) + g
            h_row = zero_i
        else:
            s_row = cutlass.Int32((t // rm.tiles_per_query) * rm.queries_per_tile) + (g >> cutlass.Int32(rm.stages))
            h_row = cutlass.Int32(16 * (t % rm.tiles_per_query) + half) + (g & cutlass.Int32(rm.group - 1)) * cutlass.Int32(2)
        valid = (s_row < s_q_i) & (h_row < h_i_i)
        s_c = s_row if s_row < s_q_i else s_q_i - cutlass.Int32(1)
        h_c = h_row if h_row < h_i_i else h_i_i - cutlass.Int32(1)
        src = (
            q_base + ((t0 + s_c).to(cutlass.Int64) * q_s0 + h_c.to(cutlass.Int64) * q_s1) * cutlass.Int64(2) + c.to(cutlass.Int64) * cutlass.Int64(_CHUNK_BYTES)
        )
        dst = sA_tile + (
            cutlass.Int32(half * 8 * _ROW_BYTES)
            + g * cutlass.Int32(_ROW_BYTES)
            + (c ^ ((g & cutlass.Int32(1)) * cutlass.Int32(4))) * cutlass.Int32(_CHUNK_BYTES)
        ).to(cutlass.Int64)
        size = cutlass.Int32(_CHUNK_BYTES) if valid else zero_i
        inline_ptx("cp.async.cg.shared.global [$0], [$1], 16, $2;", write_only_types=[], read_only_args=[dst, src, size])


@cute.jit
def _read_query_frags(sA, tile_elems: cutlass.Int32, g: cutlass.Int32, qd: cutlass.Int32, half: cutlass.Constexpr[int]):
    """Lane ``(g, q)``'s A fragments of one half (rows ``g`` / ``g + 8``) of an m-tile: granules ``q + 4 j`` of the row (one
    ``ld.shared.v4`` each at the swizzled granule ``c ^ (4 x (row & 1))``) -- the 16 words the 8 k-steps consume."""
    row = cutlass.Int32(half * 8) + g
    swz = (row & cutlass.Int32(1)) * cutlass.Int32(4)
    base = tile_elems + row * cutlass.Int32(_ROW_BYTES // 4)
    out = []
    for j in cutlass.range_constexpr(CHUNKS_PER_LANE):
        c_swz = (qd + cutlass.Int32(4 * j)) ^ swz
        v = sA.load(base + c_swz * cutlass.Int32(_WORDS_PER_CHUNK), vector_size=4, alignment=16).to_elements()
        for x in cutlass.range_constexpr(_WORDS_PER_CHUNK):
            out.append(v[x])
    return out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9], out[10], out[11], out[12], out[13], out[14], out[15]


@cute.kernel
def frost_qsa_score_paged(
    mQ: cute.Tensor,  # [T, H_i, D] bf16; token / head strides symbolic (Int64), D contiguous
    mKbar: cute.Tensor,  # [num_pages, R, D] bf16; page / row strides symbolic (Int64), D contiguous
    mTable: cute.Tensor,  # [B, max_pages] int32; row stride symbolic (Int64), unit column stride
    mKvLens: cute.Tensor,  # [B] int32, contiguous
    mLogits: cute.Tensor,  # [T, n_blocks_max] fp32; row stride symbolic (Int64), unit column stride
    mCounts: cute.Tensor,  # [T] int32, contiguous
    mBlockLens: Optional[cute.Tensor],  # [T] int32, contiguous, or None (folded out)
    top_k: cutlass.Int32,
    scale: cutlass.Float32,
    h_i: cutlass.Constexpr[int],
    s_q: cutlass.Constexpr[int],
    rows_per_page: cutlass.Constexpr[int],
    units_per_warp: cutlass.Constexpr[int],
    ring_depth: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """One CTA = chunk ``blockIdx.x`` of ``blocks_per_cta`` compressed rows of sequence ``blockIdx.y``, four independent warps of
    ``keys_per_warp`` rows each (module docstring)."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()
    rm = query_row_map(s_q, h_i)  # the (s, h) -> MMA-row map (host facts, compile-time)
    n_tiles = NTILES_PER_UNIT * units_per_warp  # n-tiles of 8 keys per warp
    depth = ring_depth  # n-tiles of keys in flight (cp.async groups) ahead of the MMAs
    ring_slots = min(
        ring_depth + 1, n_tiles
    )  # + the slot being read (the one warp sync per n-tile orders the WAR on the slot reused next); no spare when no mainloop copy exists
    # the two SMEM buffers (declared ahead of every branch): each warp's ring of ring_slots n-tiles of 8 compressed rows, and the
    # CTA's padded (s, h) query tile (16 rows x 256 B per m-tile), both written by cp.async and read as fragments
    sK = cutlass.Array(cutlass.Int32, THREADS // 32 * ring_slots * _SLOT_ELEMS, alignment=128, space=cutlass.AddressSpace.smem)
    sA = cutlass.Array(cutlass.Int32, rm.m_tiles * _A_TILE_ELEMS, alignment=128, space=cutlass.AddressSpace.smem)
    n_groups = rm.m_tiles // rm.tiles_per_query  # query groups per lane at P' >= 2 (one head-sum each)
    has_lens = cutlass.const_expr(mBlockLens is not None)
    pair_rows = cutlass.const_expr(rm.tile_rows > 1)  # the lane's two rows are two heads of ONE query (else two queries)
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx & cutlass.Int32(31)
    warp = tidx >> cutlass.Int32(5)
    g = lane >> cutlass.Int32(2)  # the MMA group id: the A / C rows g and g + 8, the B key n = g
    qd = lane & cutlass.Int32(3)  # the thread in the quad: chunks q + 4 j, the C keys 2q and 2q + 1
    bidx, bidy, _ = cute.arch.block_idx()
    chunk_id = cutlass.Int32(bidx)
    b = cutlass.Int32(bidy)
    n_blocks_max = cutlass.Int32(mLogits.shape[1])
    num_pages = cutlass.Int32(mKbar.shape[0])
    four = cutlass.Int64(4)  # int32 / fp32 bytes
    b64 = b.to(cutlass.Int64)
    zero_i = cutlass.Int32(0)
    s_q_i = cutlass.Int32(s_q)
    h_i_i = cutlass.Int32(h_i)
    t0 = b * s_q_i
    k_warp0 = chunk_id * cutlass.Int32(blocks_per_cta(units_per_warp)) + warp * cutlass.Int32(keys_per_warp(units_per_warp))
    # ---- the prologue: every load independent, all issued before any wait (ONE round trip ahead of the first key byte)
    kv_len = ld_global(mKvLens.iterator.toint() + b64 * four, cutlass.Int32)
    tb_row = mTable.iterator.toint() + b64 * cutlass.Int64(mTable.stride[0]) * four
    words = []  # the table word of this lane's key, per n-tile (issued two units ahead of the key loads)
    for i in cutlass.range_constexpr(min(2 * depth, n_tiles)):
        words.append(_page_word(tb_row, k_warp0 + cutlass.Int32(i * KEYS_PER_NTILE) + g, n_blocks_max, rows_per_page))
    # the query tile: every thread copies its granule of two rows per m-tile (one cp.async group for the whole tile)
    q_base = mQ.iterator.toint()
    q_s0 = cutlass.Int64(mQ.stride[0])
    q_s1 = cutlass.Int64(mQ.stride[1])
    sA_bytes = sA.data_ptr().toint()
    for t in cutlass.range_constexpr(rm.m_tiles):
        _copy_query_tile(sA_bytes + cutlass.Int64(t * _A_TILE_BYTES), q_base, q_s0, q_s1, t0, tidx, t, s_q, h_i)
    cute.arch.cp_async_commit_group()
    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()
    # ---- the visible counts: n_vis_max (warp-uniform; row s_q - 1 sits at position kv_len - 1) and the counts write of chunk 0
    nv_max = kv_len >> cutlass.Int32(2)  # / BLOCK_SIZE (4); a negative length is clamped below either way
    nv_max = nv_max if nv_max > zero_i else zero_i
    n_vis_max = nv_max if nv_max < n_blocks_max else n_blocks_max
    if (chunk_id == zero_i) & (tidx < s_q_i):
        pos1_t = kv_len - s_q_i + tidx + cutlass.Int32(1)
        nv_t = pos1_t >> cutlass.Int32(2)
        nv_t = nv_t if nv_t > zero_i else zero_i
        nv_t = nv_t if nv_t < n_blocks_max else n_blocks_max
        row64 = (t0 + tidx).to(cutlass.Int64)
        st_global(mCounts.iterator.toint() + row64 * four, nv_t, cutlass.Int32)
        if cutlass.const_expr(has_lens):
            bl = nv_t if nv_t < top_k else top_k
            st_global(mBlockLens.iterator.toint() + row64 * four, bl, cutlass.Int32)
    live_warp = k_warp0 < n_vis_max  # WARP-UNIFORM (the warp's first key, the sequence's visible count): the mainloop's one branch
    # ---- the key stream's first `depth` n-tiles (live warps only), ONE round trip behind the queries' and the table words'
    kb_base = mKbar.iterator.toint()
    kb_s0 = cutlass.Int64(mKbar.stride[0])
    tw = [words[i] for i in range(len(words))]  # the table words (the prologue's, then the ones issued in the mainloop)
    ring_elems = warp * cutlass.Int32(ring_slots * _SLOT_ELEMS)  # this warp's ring, in Int32 elements
    kb_page_bytes = (kb_s0 * cutlass.Int64(2)).to(cutlass.Int32)  # the page stride in bytes (the host checks it fits)
    # this lane's copy position inside a 2-row pair: row parity = lane / 16, granule = lane % 16 (the odd rows' granules XOR 4 in SMEM)
    half = lane >> cutlass.Int32(4)
    c_lane = lane & cutlass.Int32(15)
    src_lane_off = half * cutlass.Int32(_ROW_BYTES) + c_lane * cutlass.Int32(_CHUNK_BYTES)
    dst_lane_off = half * cutlass.Int32(_ROW_BYTES) + (c_lane ^ (half * cutlass.Int32(4))) * cutlass.Int32(_CHUNK_BYTES)
    dst_lane = sK.data_ptr().toint() + (ring_elems * cutlass.Int32(4) + dst_lane_off).to(cutlass.Int64)  # the cp.async destination base of slot 0
    # the keys stream through L2 ONCE: an evict-first policy on their copies keeps the logits' write-combining lines and the table (vLLM's
    # scorer marks its key loads the same way)
    policy = inline_ptx("createpolicy.fractional.L2::evict_first.b64 $0, 1.0;", write_only_types=[cutlass.Int64], read_only_args=[])
    if live_warp:
        for i in cutlass.range_constexpr(min(depth, n_tiles)):
            _copy_ntile(
                dst_lane + cutlass.Int64((i % ring_slots) * _SLOT_BYTES),
                src_lane_off,
                kb_base,
                kb_page_bytes,
                tw[i],
                num_pages,
                k_warp0 + cutlass.Int32(i * KEYS_PER_NTILE),
                policy,
                rows_per_page,
            )
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(min(depth, n_tiles))  # this thread's query-tile copies complete (the key groups may still fly)
    else:
        cute.arch.cp_async_wait_group(0)  # a dead warp has only the query-tile group in flight
    cute.arch.barrier()  # the ONE CTA barrier: every thread's query-tile copies visible to every thread's fragment reads below
    # the A fragments from the staged tile: frag_a[t][half] = the 16 words of rows g / g + 8 of m-tile t
    frag_a = []
    for t in cutlass.range_constexpr(rm.m_tiles):
        frag_a.append([_read_query_frags(sA, cutlass.Int32(t * _A_TILE_ELEMS), g, qd, 0), _read_query_frags(sA, cutlass.Int32(t * _A_TILE_ELEMS), g, qd, 1)])
    if live_warp:
        # ---- the store entries of this lane: per query group the query row s, its visible count, its logits row, the storing lane
        lg_base = mLogits.iterator.toint()
        lg_s0 = cutlass.Int64(mLogits.stride[0])
        s_rows = []
        if cutlass.const_expr(rm.tile_rows == 1):
            for t in cutlass.range_constexpr(rm.m_tiles):
                for half in cutlass.range_constexpr(2):
                    s_rows.append(cutlass.Int32(t * rm.queries_per_tile + 8 * half) + g)
        else:
            for sg in cutlass.range_constexpr(n_groups):
                s_rows.append(cutlass.Int32(sg * rm.queries_per_tile) + (g >> cutlass.Int32(rm.stages)))
        store_lane = (g & cutlass.Int32(max(rm.group, 1) - 1)) == zero_i  # every lane at P' = 1 (group 0 / 1)
        ent_nv = []
        ent_addr = []
        for e in cutlass.range_constexpr(len(s_rows)):
            s_row = s_rows[e]
            in_step = s_row < s_q_i
            pos1 = kv_len - s_q_i + s_row + cutlass.Int32(1)
            nv = pos1 >> cutlass.Int32(2)
            nv = nv if nv > zero_i else zero_i
            nv = nv if nv < n_blocks_max else n_blocks_max
            nv = nv if in_step else zero_i  # a padding query row never stores
            s_c = s_row if in_step else zero_i
            ent_nv.append(nv)
            ent_addr.append(lg_base + (t0 + s_c).to(cutlass.Int64) * lg_s0 * four)
        # ---- the key stream: the rolling ring (`depth` n-tiles in flight, the table words two units ahead)
        twb = [tw[i] for i in range(len(tw))]  # the branch's own list of table words (appended below; the outer one stays as it is)
        zeros = []
        for _ in cutlass.range_constexpr(4):
            zeros.append(opaque_f32_zero())  # the accumulator seeds (a constant reaching the mma's inline ptx would ICE libNVVM)
        k_col0 = k_warp0 + qd * cutlass.Int32(2)
        mma_batch = 4 if units_per_warp == 1 else 2  # n-tiles whose HMMA chains interleave (register cost 16 per n-tile)
        for i0 in cutlass.range_constexpr(0, n_tiles, mma_batch):
            batch = list(range(i0, min(i0 + mma_batch, n_tiles)))
            kws = []
            for bi0 in cutlass.range_constexpr(len(batch)):
                i = batch[bi0]
                if cutlass.const_expr(i + 2 * depth < n_tiles):
                    twb.append(_page_word(tb_row, k_warp0 + cutlass.Int32((i + 2 * depth) * KEYS_PER_NTILE) + g, n_blocks_max, rows_per_page))
                # group i complete: min(n_tiles, depth + i) groups were committed before this point, the i + 1 oldest must be done
                cute.arch.cp_async_wait_group(min(n_tiles, depth + i) - i - 1)
                cute.arch.sync_warp()  # the copies of the OTHER lanes visible; last step's fragment reads ordered before the slot's reuse below
                kws.append(_read_key_frags(sK, ring_elems + cutlass.Int32((i % ring_slots) * _SLOT_ELEMS), g, qd))
                if cutlass.const_expr(i + depth < n_tiles):
                    _copy_ntile(
                        dst_lane + cutlass.Int64(((i + depth) % ring_slots) * _SLOT_BYTES),
                        src_lane_off,
                        kb_base,
                        kb_page_bytes,
                        twb[i + depth],
                        num_pages,
                        k_warp0 + cutlass.Int32((i + depth) * KEYS_PER_NTILE),
                        policy,
                        rows_per_page,
                    )
                    cute.arch.cp_async_commit_group()
            # the batch's MMAs: k-step outer, n-tile and m-tile inner -- `len(batch) x m_tiles` independent accumulator chains
            accs = []
            for bi in cutlass.range_constexpr(len(batch)):
                acc = []
                for t in cutlass.range_constexpr(rm.m_tiles):
                    acc.append([zeros[0], zeros[1], zeros[2], zeros[3]])
                accs.append(acc)
            for kappa in cutlass.range_constexpr(K_STEPS):
                j = kappa // 2
                r = kappa % 2
                for bi in cutlass.range_constexpr(len(batch)):
                    kw = kws[bi]
                    b0 = kw[4 * j + 2 * r]
                    b1 = kw[4 * j + 2 * r + 1]
                    for t in cutlass.range_constexpr(rm.m_tiles):
                        a_lo = frag_a[t][0]
                        a_hi = frag_a[t][1]
                        c0, c1, c2, c3 = mma_m16n8k16_f32(
                            a_lo[4 * j + 2 * r],
                            a_hi[4 * j + 2 * r],
                            a_lo[4 * j + 2 * r + 1],
                            a_hi[4 * j + 2 * r + 1],
                            b0,
                            b1,
                            accs[bi][t][0],
                            accs[bi][t][1],
                            accs[bi][t][2],
                            accs[bi][t][3],
                            ab_dtype=cutlass.BFloat16,
                        )
                        accs[bi][t] = [c0, c1, c2, c3]
            for bi in cutlass.range_constexpr(len(batch)):
                i = batch[bi]
                acc = accs[bi]
                # relu per (s, h) element: c0 (row g, key 2q), c1 (row g, key 2q + 1), c2 (row g + 8, key 2q), c3 (row g + 8, key 2q + 1)
                relu = []
                for t in cutlass.range_constexpr(rm.m_tiles):
                    rt = []
                    for x in cutlass.range_constexpr(4):
                        rt.append(cute.math.max(acc[t][x], 0.0, ftz=True))
                    relu.append(rt)
                k0 = k_col0 + cutlass.Int32(i * KEYS_PER_NTILE)
                k1 = k0 + cutlass.Int32(1)
                # the head sums per store entry
                sums0 = []
                sums1 = []
                if cutlass.const_expr(rm.tile_rows == 1):
                    for t in cutlass.range_constexpr(rm.m_tiles):
                        sums0.append(relu[t][0])
                        sums1.append(relu[t][1])
                        sums0.append(relu[t][2])
                        sums1.append(relu[t][3])
                else:
                    for sg in cutlass.range_constexpr(n_groups):
                        v0 = relu[sg * rm.tiles_per_query][0] + relu[sg * rm.tiles_per_query][2]  # heads 2 gi and 2 gi + 1 of the query, key 2q
                        v1 = relu[sg * rm.tiles_per_query][1] + relu[sg * rm.tiles_per_query][3]  # the same, key 2q + 1
                        for ht in cutlass.range_constexpr(1, rm.tiles_per_query):
                            t = sg * rm.tiles_per_query + ht
                            v0 = v0 + (relu[t][0] + relu[t][2])
                            v1 = v1 + (relu[t][1] + relu[t][3])
                        for st in cutlass.range_constexpr(rm.stages):
                            v0 = v0 + cute.arch.shuffle_sync_bfly(v0, 4 << st)
                            v1 = v1 + cute.arch.shuffle_sync_bfly(v1, 4 << st)
                        sums0.append(v0)
                        sums1.append(v1)
                for e in cutlass.range_constexpr(len(ent_nv)):
                    nv = ent_nv[e]
                    out0 = sums0[e] * scale
                    out1 = sums1[e] * scale
                    addr = ent_addr[e] + k0.to(cutlass.Int64) * four
                    if store_lane & (k1 < nv):
                        st_global_v2(addr, (out0, out1), cutlass.Float32)
                    elif store_lane & (k0 < nv):
                        st_global(addr, out0, cutlass.Float32)


frost_qsa_score_paged.set_name_prefix("cudnn", remove_cutlass_symbol=True)


@cute.jit
def qsa_score_paged_launch(
    mQ: cute.Tensor,
    mKbar: cute.Tensor,
    mTable: cute.Tensor,
    mKvLens: cute.Tensor,
    mLogits: cute.Tensor,
    mCounts: cute.Tensor,
    mBlockLens: Optional[cute.Tensor],
    top_k: cutlass.Int32,
    scale: cutlass.Float32,
    n_chunks: cutlass.Int32,
    batch: cutlass.Int32,
    h_i: cutlass.Constexpr[int],
    s_q: cutlass.Constexpr[int],
    rows_per_page: cutlass.Constexpr[int],
    units_per_warp: cutlass.Constexpr[int],
    ring_depth: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qsa_score_paged(
        mQ, mKbar, mTable, mKvLens, mLogits, mCounts, mBlockLens, top_k, scale, h_i, s_q, rows_per_page, units_per_warp, ring_depth, use_pdl
    ).launch(grid=(n_chunks, batch, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=use_pdl)


# --------------------------------------------------------------------------------------------- compile + run

compiled_cache = {}


class QsaScorePagedRecipe(NamedTuple):
    """Build-time facts of one artifact (Rule 4: one artifact per (H_i, S_q, page size, units knob, block_lens presence, PDL));
    the batch, the cache capacity, the table width, the page count and every stride ride in at runtime."""

    compiled: object
    h_i: int
    s_q: int
    page_size: int
    rows_per_page: int
    units_per_warp: int
    has_block_lens: bool
    use_pdl: bool
    device: int
    ring_depth: int = NTILES_IN_FLIGHT


def _fake_q(h_i: int):
    return cute.runtime.make_fake_tensor(
        dtype=cutlass.BFloat16, shape=(cute.sym_int(), int(h_i), INDEX_HEAD_DIM), stride=(cute.sym_int(width=64), cute.sym_int(width=64), 1), assumed_align=16
    )


def _fake_kbar(rows_per_page: int):
    return cute.runtime.make_fake_tensor(
        dtype=cutlass.BFloat16,
        shape=(cute.sym_int(), int(rows_per_page), INDEX_HEAD_DIM),
        stride=(cute.sym_int(width=64), cute.sym_int(width=64), 1),
        assumed_align=16,
    )


def _fake_i32_2d():
    return cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(width=64), 1), assumed_align=4)


def _fake_f32_2d():
    return cute.runtime.make_fake_tensor(dtype=cutlass.Float32, shape=(cute.sym_int(), cute.sym_int()), stride=(cute.sym_int(width=64), 1), assumed_align=16)


def _fake_i32_1d():
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=4)


def dsl_decline_reason(device: Optional[int] = None) -> Optional[str]:
    """Rule 7: None when this DSL can build the kernel for this device, else the typed reason -- the version floor first
    (device-free), then the ``sm_107a`` target on cc 10.7.  The kernel itself (``mma.sync.m16n8k16``, ``cp.async``, shuffles, one
    CTA barrier) runs on any CuTe DSL device from sm_80; PDL adds the cc 9.0 requirement (:func:`compile_qsa_score_paged`)."""
    too_old = cutedsl_requirement_error("qsa_score_paged")
    if too_old is not None:
        return too_old
    dev = current_device() if device is None else int(device)
    return cutedsl_arch_requirement_error(compute_capability(dev))


def compile_qsa_score_paged(
    *,
    h_i: int,
    s_q: int,
    page_size: int,
    units_per_warp: int = 1,
    has_block_lens: bool = True,
    use_pdl: bool = False,
    device: Optional[int] = None,
    ring_depth: int = NTILES_IN_FLIGHT,
) -> QsaScorePagedRecipe:
    """Build from SHAPES ALONE (no allocation, no launch).  The DSL / arch gate runs BEFORE the compile (Rule 7): a decline
    is a ``NotImplementedError`` naming the version or the target, never an error from inside the DSL.  ``ring_depth`` (the
    n-tiles of keys in flight per warp, ``RING_DEPTH_CHOICES``) is a performance knob: the same function."""
    global _FAKE_STREAM
    validate_params(h_i=h_i, s_q=s_q, page_size=page_size, units_per_warp=units_per_warp, ring_depth=ring_depth)
    dev = current_device() if device is None else int(device)
    why = dsl_decline_reason(dev)
    if why is not None:
        raise NotImplementedError(why)
    cc = compute_capability(dev)
    if use_pdl and cc < (9, 0):
        raise NotImplementedError(
            f"use_pdl=True traces the programmatic-dependent-launch pair (griddepcontrol), an instruction of cc 9.0 and later; this device is cc {cc[0]}.{cc[1]}"
        )
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    rows_per_page = int(page_size) // BLOCK_SIZE
    key = (int(h_i), int(s_q), int(page_size), int(units_per_warp), bool(has_block_lens), bool(use_pdl), dev) + (
        () if int(ring_depth) == NTILES_IN_FLIGHT else (int(ring_depth),)
    )
    if key not in compiled_cache:
        # The dict is the in-process memo; behind it sits the persistent compiled-plan cache, keyed by this file's digest
        # (``FROST_SOURCE_DIGEST`` above) + the key's plain values: a second process reloads the object instead of re-tracing.
        cache_key = template_key(globals(), {"key": key}, "compile_qsa_score_paged")
        compiled_cache[key] = compile_cached(
            qsa_score_paged_launch,
            _fake_q(h_i),
            _fake_kbar(rows_per_page),
            _fake_i32_2d(),  # the table [B, max_pages]
            _fake_i32_1d(),  # kv_lens
            _fake_f32_2d(),  # logits [T, n_blocks_max]
            _fake_i32_1d(),  # counts
            _fake_i32_1d() if has_block_lens else None,
            cutlass.Int32(0),  # top_k    ) runtime; the zeros pin the TYPE only
            cutlass.Float32(0.0),  # scale    )
            cutlass.Int32(0),  # n_chunks )
            cutlass.Int32(0),  # batch    )
            int(h_i),
            int(s_q),
            int(rows_per_page),
            int(units_per_warp),
            int(ring_depth),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
            cache_key=cache_key,
            symbol="cudnn_block_qsa_score_paged",
        )
    return QsaScorePagedRecipe(
        compiled=compiled_cache[key],
        h_i=int(h_i),
        s_q=int(s_q),
        page_size=int(page_size),
        rows_per_page=rows_per_page,
        units_per_warp=int(units_per_warp),
        has_block_lens=bool(has_block_lens),
        use_pdl=bool(use_pdl),
        device=dev,
        ring_depth=int(ring_depth),
    )


def run_qsa_score_paged(
    r: QsaScorePagedRecipe,
    qi: torch.Tensor,
    kbar: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    logits: torch.Tensor,
    counts: torch.Tensor,
    block_lens: Optional[torch.Tensor],
    *,
    top_k: int,
    scale: float,
    stream,
) -> None:
    """Validate the FORM of every operand (no device read, Rule 3) and launch (Rule 1: no conversion, no allocation).

    ``qi [T, H_i, D]`` bf16 with a unit last stride and 16-byte-aligned token / head strides (a column slice of a wider slab
    reads in place), ``T = B x s_q``; ``kbar`` the paged cache (:func:`_kbar_rows_view`); ``block_table [B, max_pages]`` int32
    with a unit column stride, ``max_pages x R >= n_blocks_max``; ``kv_lens [B]`` int32; ``logits [T, n_blocks_max]`` fp32 with a
    unit column stride and 16-byte-aligned rows; ``counts [T]`` int32; ``block_lens [T]`` int32 iff the artifact traces it;
    ``top_k`` in ``[1, 2048]``; ``scale > 0`` (the ordering is scale-free); ``stream`` a raw CUDA stream handle (an ``int`` or a
    ``cuda.CUstream``; the host entry resolves a ``torch.cuda.Stream`` to it)."""
    if not isinstance(qi, torch.Tensor) or qi.dim() != 3:
        raise ValueError(f"qi must be a [T, H_i, {INDEX_HEAD_DIM}] bf16 tensor, got {getattr(qi, 'shape', type(qi).__name__)}")
    if qi.dtype != torch.bfloat16:
        raise ValueError(f"qi must be bfloat16 for this artifact, got {qi.dtype}")
    if not qi.is_cuda:
        raise ValueError(f"qi must be a CUDA tensor (every operand is read through a device pointer), got device {qi.device}")
    device = qi.device
    t, h_i, d = (int(x) for x in qi.shape)
    if h_i != r.h_i or d != INDEX_HEAD_DIM:
        raise ValueError(f"qi must be [T, H_i={r.h_i}, {INDEX_HEAD_DIM}] (H_i is fixed per artifact), got {tuple(qi.shape)}")
    qs0, qs1, qs2 = (int(x) for x in qi.stride())
    if qs2 != 1 or (qs0 * 2) % 16 or (qs1 * 2) % 16 or qi.data_ptr() % 16:
        raise ValueError(
            f"qi needs a unit last stride and 16-byte-aligned token / head strides and base, got strides {tuple(qi.stride())}, data_ptr {qi.data_ptr():#x}"
        )
    if not isinstance(block_table, torch.Tensor) or block_table.dim() != 2 or block_table.dtype != torch.int32:
        raise ValueError(
            f"block_table must be an int32 [B, max_pages] tensor, got {getattr(block_table, 'shape', type(block_table).__name__)} {getattr(block_table, 'dtype', '')}"
        )
    batch, max_pages = (int(x) for x in block_table.shape)
    if batch < 1 or max_pages < 1:
        raise ValueError(f"block_table must hold at least one sequence and one page column, got {tuple(block_table.shape)}")
    if int(block_table.stride(1)) != 1 or block_table.device != device:
        raise ValueError(f"block_table needs a unit column stride on {device}, got strides {tuple(block_table.stride())} on {block_table.device}")
    if t != batch * r.s_q:
        raise ValueError(
            f"qi holds T = {t} rows but block_table / kv_lens hold B = {batch} sequences of s_q = {r.s_q} rows (T must be B x s_q = {batch * r.s_q})"
        )
    _check_i32_1d(kv_lens, "kv_lens", batch, device)
    kbar_view = _kbar_rows_view(kbar, r.rows_per_page, device)
    if not isinstance(logits, torch.Tensor) or logits.dim() != 2 or logits.dtype != torch.float32:
        raise ValueError(
            f"logits must be an fp32 [T, n_blocks_max] tensor, got {getattr(logits, 'shape', type(logits).__name__)} {getattr(logits, 'dtype', '')}"
        )
    lt, n_blocks_max = (int(x) for x in logits.shape)
    if lt != t or n_blocks_max < 1:
        raise ValueError(f"logits must be [T={t}, n_blocks_max >= 1], got {tuple(logits.shape)}")
    ls0, ls1 = (int(x) for x in logits.stride())
    if ls1 != 1 or (ls0 * 4) % 16 or logits.data_ptr() % 16 or logits.device != device:
        raise ValueError(
            f"logits needs a unit column stride, 16-byte-aligned rows and base, on {device}; got strides {tuple(logits.stride())}, data_ptr {logits.data_ptr():#x}, {logits.device}"
        )
    if max_pages * r.rows_per_page < n_blocks_max:
        raise ValueError(
            f"block_table maps max_pages x R = {max_pages} x {r.rows_per_page} = {max_pages * r.rows_per_page} compressed rows, fewer than the "
            f"n_blocks_max = {n_blocks_max} columns of logits: widen the table or shrink the logits"
        )
    _check_i32_1d(counts, "counts", t, device)
    if r.has_block_lens != (block_lens is not None):
        raise ValueError(f"this artifact was compiled {'with' if r.has_block_lens else 'without'} block_lens; the call must match (Rule 1: no silent fallback)")
    if block_lens is not None:
        _check_i32_1d(block_lens, "block_lens", t, device)
    if not 1 <= int(top_k) <= TOP_K_KERNEL_MAX:
        raise ValueError(f"top_k must be in [1, {TOP_K_KERNEL_MAX}], got {top_k}")
    if not float(scale) > 0.0:
        raise ValueError(f"scale must be positive (the ordering is scale-free), got {scale}")
    n_chunks = _ceil_div(n_blocks_max, blocks_per_cta(r.units_per_warp))
    r.compiled(
        qi,
        kbar_view,
        block_table,
        kv_lens,
        logits,
        counts,
        block_lens,
        cutlass.Int32(int(top_k)),
        cutlass.Float32(float(scale)),
        cutlass.Int32(n_chunks),
        cutlass.Int32(batch),
        cuda.CUstream(int(stream)),
    )
