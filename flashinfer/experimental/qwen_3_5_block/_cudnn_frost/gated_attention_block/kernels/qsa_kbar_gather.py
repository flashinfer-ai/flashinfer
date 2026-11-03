# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The dense ``kbar`` gather + the extend step's device metadata: the paged compressed-key cache, through the main block table,
into the contiguous ``[B, n_blocks_max, 128]`` operand of the compact-logits scorer -- ONE launch, no SMEM, no mbarrier.

A prefill chunk of ``T`` rows behind a cached prefix (the in-block indexer's EXTEND step) selects its blocks over the PREFIX's
compressed keys, which live in the serving stack's PAGED cache ``index_kv_cache [num_pages, P / POOL, 1, D]`` behind the MAIN
block table ``block_table [B, max_pages]`` -- while the scorer (``cudnn.QSA.selection.qsa_select_forward``) takes a CONTIGUOUS
``kbar [B, n_blocks_max, D]``.  This kernel is that gather, a 16-byte-granule BIT COPY (bf16 rows of 256 B, e4m3 rows of 128 B:
the row's dtype is a compile key, no unpack, no cast, no fp8 instruction -- one kernel body serves both), plus the three small
metadata vectors the step needs from ``kv_lens`` ON DEVICE (a serving stack's lengths are device data; the host reads nothing
back).  Per sequence ``b`` with ``L = kv_lens[b]`` (the KV length AFTER the chunk), ``T`` the chunk's rows, ``NB = n_blocks_max =
ix_kbar.shape[1]`` (= the table's reach in compressed rows, ``max_pages x P / POOL``) and ``POOL = 4``::

    n_keys        = clamp(floor(L / POOL), 0, NB)
    row j < n_keys                              : ix_kbar[b, j] = cache[table[b, j // rpp], j % rpp]   (a -1 / out-of-range page -> a ZERO row:
                                                                                                        the paged scorer's own convention)
    row n_keys <= j < min(NB, ceil(T / POOL))   : ix_kbar[b, j] = 0     (every row a chunk row can read past n_keys is a DEFINED zero -- the
                                                                        prefill form's memset rule: a dead / short sequence's rows, and a SERVED
                                                                        sequence's row n_keys when its last group is partial, L // POOL == T // POOL
                                                                        with T % POOL != 0, e.g. L = T = 37 -- never read there, n_vis <= n_keys)
    row j past both                             : untouched
    ix_counts[b T + i]  = clamp(floor((L - T + i + 1) / POOL), 0, NB)                  the TRUE position's visible count (0 on a negative one);
                                                                                        computed from max(L, 0) -- the same value for every L,
                                                                                        and no int32 wrap of L - T at L near INT_MIN
    ix_qpos[p]          = clamp(max(L - T, 0) + r0_p, 0, POOL NB + POOL - 1 - rows_p)  p a piece of b: the scorer's non-negative first position,
                                                                                        AND the memory guard below (from max(L, 0) - T likewise)
    ix_seqrep[b]        = n_keys  (served: T <= L <= POOL NB) | -1 (dead: L <= 0) | -2 (0 < L < T) | -3 (L > POOL NB)

**The clamp of ``ix_qpos`` is the memory guard of the compact-logits scratch.**  The scorer stores row ``i`` of a piece at
``cand_batch_offsets[p] + G(q_pos0_p + i) - G(q_pos0_p)`` with a per-row region of ``floor((q_pos0_p + i + 1) / POOL)`` floats that
is UNCLAMPED by the key count, so a per-piece slab sized ``rows_p x NB`` floats (the step's plan-constant, loose bases) holds every row
only while ``floor((q_pos0_p + rows_p) / POOL) <= NB`` -- i.e. ``q_pos0_p <= POOL NB + POOL - 1 - rows_p``.  The kernel clamps there, so
whatever ``kv_lens`` holds (a stack bug included) no scorer write leaves its slab: memory safety is never a device-value contract.
The clamp needs ``rows_p <= T <= POOL x NB``, which the HOST guarantees before any launch (``run_qsa_kbar_gather`` refuses a chunk
longer than the table's reach, naming both numbers).  A clamped sequence (``L > POOL NB``: ``kv_lens[b] > max_pages x P``, the block's
existing device contract) is position-misaligned in its selection and REPORTED ``-3``; a sequence with ``0 < L < T`` is the other
violation (``-2``: a chunk extends a sequence of at least its own length); ``ix_seqrep`` is a device REPORT the tests read and the
block never reads; its precedence is dead > short > over > served (``-1`` over ``-2`` over ``-3`` over ``n_keys``), and short / over are
DISJOINT under the host rule ``T <= POOL x NB`` (``0 < L < T`` and ``L > POOL NB`` cannot both hold), so the order between those two
never decides a report.  A dead sequence (``L <= 0``) is fully defined WHATEVER ITS MAGNITUDE: zero rows up to ``ceil(T / POOL)``, counts 0,
``q_pos0 = r0_p`` -- the positions are derived from ``max(L, 0)``, never from ``L - T`` (int32 arithmetic on ``L = INT_MIN`` would wrap
positive and hand a dead sequence live counts and a clamped ``q_pos0``).

The piece table ``piece_seq / piece_r0 / piece_rows [n_pieces]`` (int32, device; the step's plan constants, uploaded once) names the
select windows' pieces: piece ``p`` is rows ``[r0_p, r0_p + rows_p)`` of sequence ``piece_seq[p]``.  Its VALUES are device data used in
arithmetic and in an equality against ``b`` only -- never as an index -- so a malformed table cannot address memory.

Grid ``(ceil(NB / groups_per_cta), B)``, ``threads_per_cta`` threads = ``threads / lanes`` lane groups of ``lanes = row_bytes / 16``
lanes (16 at bf16, 8 at e4m3), ONE group per compressed row ``j``; CTA ``(b, x = 0)`` also writes the metadata: ``ix_counts`` by
thread ``i`` in a stride-``threads`` loop over ``T``, ``ix_qpos[p]`` by the thread that finds ``piece_seq[p] == b`` (a stride-``threads``
loop over the pieces), ``ix_seqrep[b]`` by thread 0.  Bytes: ``B x n_keys x row_bytes`` read + written (8 MiB per sequence at a
128K reach, bf16); the roofline term is HBM bytes -- no perf claim is made here until one is measured on an exclusive GPU.

**Barrier table -- EMPTY (zero rows).**  No mbarrier, no ``cp.async`` group, no CTA barrier, no TMA, no tcgen05, no cluster, and
NO warp shuffle either: every lane's work is independent, so ``SUM(issuing lanes) == init_count`` has no row to reconcile and the
``frost-barrier-inspect`` audit of this kernel is THIS statement (the ``cache_write.py`` precedent), not an audit skipped.
**SMEM table -- EMPTY (zero rows):** no ``cutlass.Array(space=smem)``, no ``SmemAllocator``, no ``smem=`` at the launch; every operand
moves register <-> global through ``ld.global.v4`` / ``st.global.v4`` (16 B per lane) or 4-B scalar ``ld.global`` / ``st.global``.
**Per-lane stride of every global access** (the SMEM table's column, applied to global so the coalescing is justified): the cache
rows -- lane ``l`` of a row group reads granule ``l`` at ``row + 16 l`` B, 16 (bf16) / 8 (e4m3) consecutive lanes per 256-B / 128-B row
= whole 128-B lines, consecutive rows on consecutive groups; ``ix_kbar[b, j]`` -- the same granule mapping into a contiguous
destination (row stride 256 / 128 B), the zero-fill the same addresses with zero words; ``block_table[b, j // rpp]`` -- one 4-B load
per group, the ``rpp`` groups of a page on ONE word (L1 broadcast); ``kv_lens[b]`` -- one 4-B load per group; ``ix_counts[b T + i]`` --
thread ``i`` of CTA ``(b, 0)`` in a stride-``threads`` loop, consecutive threads on consecutive 4-B words; ``ix_qpos[p]`` /
``ix_seqrep[b]`` -- single 4-B stores by one thread each.
**``fence_proxy`` -- NONE:** no async proxy exists in this kernel; its generic-proxy stores are consumed by LATER kernels in stream
order (the gathered ``ix_kbar`` by the scorer's TMA loads -- a kernel boundary is a full proxy fence), so ``frost-tile-dsl.md`` § 1 has
no site here: stated, not skipped.
**Rule 7 / the target:** like ``cache_write.py`` the kernel carries NO arch pin (plain LDG / STG; the e4m3 bit copy needs no fp8
instruction): any CuTe DSL device from sm_80; ``compile_qsa_kbar_gather`` gates the DSL floor first (``cutedsl_requirement_error``).
The Rubin pin of the EXTEND form is the block's (``sm_107a`` at declaration), not this kernel's.

Programmatic dependent launch (``use_pdl``): the ``griddepcontrol.wait`` precedes the first dependent read (``kv_lens``, the table,
the cache rows the chunk compress just wrote) and the ``launch_dependents`` release sits RIGHT AFTER it (the streaming-kernel
placement of ``cache_write.py``).
"""

import hashlib
from pathlib import Path
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import compile_cached, template_key
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import opaque_i32_zero
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, ld_global_v4, st_global, st_global_v4

from ..cache_write_contract import DEFAULT_POOL
from .qk_norm_rope import ACCESS_BYTES
from .qsa_compress import _check_int32_vector, _check_row_tensor, _fake_compact, _fake_i32_table, _fake_paged_state, check_block_table, paged_block_row

DEFAULT_THREADS_PER_CTA = 128
"""Four warps per CTA: 8 bf16 rows (16 e4m3 rows) per CTA.  A streaming bit copy; re-measure per arch before changing it."""

_CACHE_DTYPES = (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
_FAKE_STREAM = None
# Imported normally (not through the template loader), so the module names its own source identity: ``template_key`` joins it with a
# compile site's key to name the persistent object (``compiled_cache``); the cache's environment manifest covers every other ``.py``
# the kernel is traced from.
FROST_SOURCE_DIGEST = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

SEQREP_DEAD = -1
"""``ix_seqrep`` value: a dead sequence (``kv_lens <= 0``; a padding request): zero rows, counts 0, every list ``-1`` downstream."""
SEQREP_SHORT = -2
"""``ix_seqrep`` value: ``0 < kv_lens < T`` -- a chunk longer than the sequence it extends (a contract violation): ``q_pos0`` clamped at
``r0``, the live rows' selection position-misaligned; memory-safe."""
SEQREP_OVER = -3
"""``ix_seqrep`` value: ``kv_lens > POOL x n_blocks_max`` (= ``max_pages x P``) -- a sequence past the table's reach (a contract
violation): ``q_pos0`` clamped so no scorer write leaves its slab, the selection position-misaligned; memory-safe by construction."""


def validate_gather_shape(d: int, elem_bytes: int, pool: int, threads_per_cta: int) -> None:
    """Raise on any geometry this kernel cannot address: a row of ``d x elem_bytes`` bytes is a whole number of 16-B granules, a CTA
    holds whole lane groups, the pool is positive.  Never an ``assert``: user-facing geometry must survive ``python -O`` and name itself."""
    if elem_bytes not in (1, 2):
        raise ValueError(f"the compressed-key cache holds 16-bit or e4m3 rows (1 or 2 bytes per element), got {elem_bytes}")
    row_bytes = int(d) * int(elem_bytes)
    if d < 1 or row_bytes % ACCESS_BYTES:
        raise ValueError(f"d x element bytes = {row_bytes} must be a positive multiple of {ACCESS_BYTES} B (one 16-B granule per lane), got d={d}")
    lanes = row_bytes // ACCESS_BYTES
    if threads_per_cta < lanes or threads_per_cta % lanes:
        raise ValueError(f"threads_per_cta must be a positive multiple of the {lanes} lanes of one row, got {threads_per_cta}")
    if int(pool) < 1:
        raise ValueError(f"pool must be >= 1, got {pool}")


@cute.kernel
def frost_qsa_kbar_gather(
    mCache: cute.Tensor,  # [num_pages, rpp, 1, D] the paged compressed-key cache (cache dtype): strides symbolic, D contiguous
    mTable: cute.Tensor,  # [B, max_pages] int32: the MAIN block table (row stride symbolic, unit column stride)
    mKvLens: cute.Tensor,  # [B] int32: the KV length AFTER the chunk
    mPieceSeq: cute.Tensor,  # [n_pieces] int32: the sequence of every select piece
    mPieceR0: cute.Tensor,  # [n_pieces] int32: the piece's first chunk row
    mPieceRows: cute.Tensor,  # [n_pieces] int32: the piece's rows
    mKbar: cute.Tensor,  # [B, NB, D] out (cache dtype), contiguous
    mQpos: cute.Tensor,  # [n_pieces] int32 out: the clamped first position per piece
    mCounts: cute.Tensor,  # [B * T] int32 out: the visible count per chunk row
    mSeqRep: cute.Tensor,  # [B] int32 out: the per-sequence report
    n_pieces: cutlass.Int32,
    t_rows: cutlass.Int32,  # T, the chunk's rows per sequence
    d: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """One lane group (``row_bytes / 16`` lanes) per compressed row ``j`` of sequence ``b = block_idx.y``; CTA ``(b, 0)`` also writes
    the sequence's metadata (module docstring).  A group past ``NB`` loads nothing and stores nothing (no shuffle anywhere, so a tail
    group needs no clamped participation)."""
    if cutlass.const_expr(use_pdl):
        # the wait before the first dependent read (kv_lens, the table, the cache rows the chunk compress wrote); the release right after it
        wait_on_dependent_grids()
        launch_dependent_grids()

    row_bytes = cutlass.const_expr(d * elem_bytes)
    lanes = cutlass.const_expr(row_bytes // ACCESS_BYTES)
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    bpe = cutlass.Int64(elem_bytes)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    b = cutlass.Int32(cute.arch.block_idx()[1])
    x = cutlass.Int32(cute.arch.block_idx()[0])
    j = x * cutlass.Int32(groups_per_cta) + grp
    b64 = b.to(cutlass.Int64)

    n_blocks_max = cutlass.Int32(mKbar.shape[1])
    length = ld_global(mKvLens.iterator.toint() + b64 * cutlass.Int64(4), cutlass.Int32)
    length0 = length if length > cutlass.Int32(0) else cutlass.Int32(0)
    n_keys = length0 // cutlass.Int32(pool)  # a non-negative dividend: the DSL's integer division truncates toward zero
    n_keys = n_keys if n_keys < n_blocks_max else n_blocks_max
    fill = (t_rows + cutlass.Int32(pool - 1)) // cutlass.Int32(pool)  # ceil(T / POOL): the rows a chunk row of a dead / short sequence can read
    fill = fill if fill < n_blocks_max else n_blocks_max
    fill_end = n_keys if n_keys > fill else fill

    # --- the gather: row j of sequence b, a bit copy through the table; zero words for an unassigned page and for the defined-zero rows ---
    in_grid = j < n_blocks_max
    j_r = j if in_grid else n_blocks_max - cutlass.Int32(1)  # in-bounds (NB >= 1, host-checked); a tail group stores nothing
    src, fits = paged_block_row(mCache, mTable, b, j_r, bpe)
    keep = (j_r < n_keys) & fits
    store = in_grid & (j_r < fill_end)
    lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    if store:
        words = ld_global_v4(src + lane_off, cutlass.Int32)  # a clamped, in-bounds row even when its words are not kept
        zero = opaque_i32_zero()
        out = [w if keep else zero for w in words]
        dst = mKbar.iterator.toint() + ((b64 * n_blocks_max.to(cutlass.Int64) + j_r.to(cutlass.Int64)) * cutlass.Int64(d)) * bpe + lane_off
        st_global_v4(dst, out, cutlass.Int32)

    # --- the metadata, by CTA (b, 0) ---
    if x == cutlass.Int32(0):
        # the position of the chunk's row 0 from the NON-NEGATIVE length: L - T for every L >= 0 (negative on a short sequence), -T for every
        # dead one -- L - T itself wraps int32 at L near INT_MIN (counts [NB, NB, NB, 0] and a clamped q_pos0 for a DEAD sequence)
        pos_base = length0 - t_rows
        for i in cutlass.range(tidx, t_rows, threads_per_cta):
            pos1 = pos_base + i + cutlass.Int32(1)  # the TRUE position + 1
            cnt = (pos1 // cutlass.Int32(pool)) if pos1 > cutlass.Int32(0) else cutlass.Int32(0)  # never a negative dividend
            cnt = cnt if cnt < n_blocks_max else n_blocks_max
            st_global(mCounts.iterator.toint() + (b64 * t_rows.to(cutlass.Int64) + i.to(cutlass.Int64)) * cutlass.Int64(4), cnt, cutlass.Int32)
        q_base = pos_base if pos_base > cutlass.Int32(0) else cutlass.Int32(0)  # the scorer's non-negative first position
        for p in cutlass.range(tidx, n_pieces, threads_per_cta):
            p64 = p.to(cutlass.Int64) * cutlass.Int64(4)
            seq = ld_global(mPieceSeq.iterator.toint() + p64, cutlass.Int32)
            if seq == b:
                r0 = ld_global(mPieceR0.iterator.toint() + p64, cutlass.Int32)
                rows = ld_global(mPieceRows.iterator.toint() + p64, cutlass.Int32)
                qpos = q_base + r0
                # the memory guard: floor((qpos + rows) / POOL) <= NB  <=>  qpos <= POOL NB + POOL - 1 - rows
                bound = cutlass.Int32(pool) * n_blocks_max + cutlass.Int32(pool - 1) - rows
                qpos = qpos if qpos < bound else bound
                qpos = qpos if qpos > cutlass.Int32(0) else cutlass.Int32(0)
                st_global(mQpos.iterator.toint() + p64, qpos, cutlass.Int32)
        if tidx == cutlass.Int32(0):
            rep = n_keys
            rep = cutlass.Int32(SEQREP_OVER) if length > cutlass.Int32(pool) * n_blocks_max else rep
            rep = cutlass.Int32(SEQREP_SHORT) if (length > cutlass.Int32(0)) & (length < t_rows) else rep
            rep = cutlass.Int32(SEQREP_DEAD) if length <= cutlass.Int32(0) else rep
            st_global(mSeqRep.iterator.toint() + b64 * cutlass.Int64(4), rep, cutlass.Int32)


@cute.jit
def qsa_kbar_gather_launch(
    cache: cute.Tensor,
    table: cute.Tensor,
    kv_lens: cute.Tensor,
    piece_seq: cute.Tensor,
    piece_r0: cute.Tensor,
    piece_rows: cute.Tensor,
    kbar: cute.Tensor,
    qpos: cute.Tensor,
    counts: cute.Tensor,
    seqrep: cute.Tensor,
    n_pieces: cutlass.Int32,
    t_rows: cutlass.Int32,
    grid_x: cutlass.Int32,
    grid_y: cutlass.Int32,
    d: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qsa_kbar_gather(
        cache, table, kv_lens, piece_seq, piece_r0, piece_rows, kbar, qpos, counts, seqrep, n_pieces, t_rows, d, elem_bytes, pool, threads_per_cta, use_pdl
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}


class QsaKbarGatherRecipe(NamedTuple):
    """Build-time facts of one gather launch -- the cache dtype, the head dim, the pool, the CTA shape, PDL (Rule 4: one artifact per
    (dtype, d, pool, threads, use_pdl, device)); the batch, the reach, the page geometry, the chunk rows and the piece count ride in
    as runtime shapes / scalars."""

    compiled: object
    dtype: object  # the cache's torch dtype (bf16 / f16 / float8_e4m3fn): a compile key, the runner refuses any other
    d: int
    elem_bytes: int
    pool: int
    groups_per_cta: int
    threads_per_cta: int
    use_pdl: bool
    device: int = -1  # appended: the CUDA device the artifact was compiled for (part of its key); the launch selects it, operands elsewhere are refused


def _fake_i32_vec(n):
    """``[n]`` contiguous int32 with a symbolic (or given) length, 4-byte aligned (a slice of a wider vector binds)."""
    return cute.runtime.make_fake_compact_tensor(cutlass.Int32, (n,), stride_order=(0,), assumed_align=4)


def compile_qsa_kbar_gather(
    *, dtype, d: int = 128, pool: int = DEFAULT_POOL, threads_per_cta: int = DEFAULT_THREADS_PER_CTA, use_pdl: bool = False
) -> QsaKbarGatherRecipe:
    """Build from SHAPES ALONE -- no allocation, no launch.  ``dtype`` is the compressed-key cache's (bf16 / f16 / float8_e4m3fn: the
    bit copy serves every width; ``ix_kbar`` takes the same dtype); the DSL floor is gated first (Rule 7).  Every ``Constexpr`` is in
    the persistent key, nothing bound at execute is."""
    global _FAKE_STREAM
    too_old = cutedsl_requirement_error("qsa_kbar_gather")
    if too_old is not None:
        raise NotImplementedError(too_old)
    if dtype not in _CACHE_DTYPES:
        raise ValueError(f"the compressed-key cache holds bf16 / f16 or torch.float8_e4m3fn rows, got {dtype}")
    elem_bytes = 1 if dtype == torch.float8_e4m3fn else 2
    validate_gather_shape(d, elem_bytes, pool, threads_per_cta)
    if _FAKE_STREAM is None:
        from cutlass.cute.runtime import make_fake_stream

        _FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
    key = (str(dtype), int(d), int(pool), int(threads_per_cta), bool(use_pdl), current_device())
    if key not in compiled_cache:
        # The dict is the in-process memo; behind it sits the persistent compiled-plan cache (``cudnn.frost.compiled_cache``), keyed by
        # this file's digest + the dict key's plain values: a second process reloads the object instead of re-tracing.
        cache_key = template_key(globals(), {"key": key}, "compile_qsa_kbar_gather")
        batch = cute.sym_int()
        npc = cute.sym_int()
        with torch.cuda.device(key[-1]):  # the compile and the DSL's load of the launchable run ON the artifact's device (the recipe's), whichever
            compiled_cache[key] = compile_cached(  # device is live -- a build_device() scope names a device the live context need not be
                qsa_kbar_gather_launch,
                _fake_paged_state(dtype, d),
                _fake_i32_table((batch, cute.sym_int()), (cute.sym_int(), 1)),
                _fake_i32_vec(batch),
                _fake_i32_vec(npc),
                _fake_i32_vec(npc),
                _fake_i32_vec(npc),
                _fake_compact(dtype, (batch, cute.sym_int(), d), (2, 1, 0)),
                _fake_i32_vec(npc),
                _fake_i32_vec(cute.sym_int()),
                _fake_i32_vec(batch),
                cutlass.Int32(0),  # n_pieces ) runtime; the zeros pin the TYPE only
                cutlass.Int32(1),  # t_rows   )
                cutlass.Int32(1),  # grid_x   )
                cutlass.Int32(1),  # grid_y   )
                int(d),
                int(elem_bytes),
                int(pool),
                int(threads_per_cta),
                bool(use_pdl),
                _FAKE_STREAM,
                options="--enable-tvm-ffi",
                cache_key=cache_key,
                symbol="cudnn_block_qsa_kbar_gather",
            )
    return QsaKbarGatherRecipe(
        compiled=compiled_cache[key],
        dtype=dtype,
        d=int(d),
        elem_bytes=int(elem_bytes),
        pool=int(pool),
        groups_per_cta=threads_per_cta // ((d * elem_bytes) // ACCESS_BYTES),
        threads_per_cta=int(threads_per_cta),
        use_pdl=bool(use_pdl),
        device=key[-1],
    )


def _check_i32(name: str, x, n: int, device) -> None:
    _check_int32_vector(name, x, (n,))
    if x.device != device:
        raise ValueError(f"{name} must live on {device} with the cache, got {x.device}")


def run_qsa_kbar_gather(
    r: QsaKbarGatherRecipe,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    piece_seq: torch.Tensor,
    piece_r0: torch.Tensor,
    piece_rows: torch.Tensor,
    ix_kbar: torch.Tensor,
    ix_qpos: torch.Tensor,
    ix_counts: torch.Tensor,
    ix_seqrep: torch.Tensor,
    *,
    t_rows: int,
    stream,
) -> None:
    """Launch.  Form checks only (no device read, no allocation): ``index_kv_cache [num_pages >= 1, rpp >= 1, 1, D]`` in the recipe's
    dtype (an e4m3 cache under a 16-bit recipe and the converse are refused naming the recipe's), ``block_table [B, max_pages]`` int32
    with unit column stride, ``kv_lens [B]`` int32, the piece table (three ``[n_pieces >= 1]`` int32 vectors), ``ix_kbar [B, NB, D]``
    contiguous in the recipe's dtype with ``1 <= NB <= max_pages x rpp`` (the table's reach), ``ix_qpos [n_pieces]``, ``ix_counts [B x
    T]`` (or ``[B, T]`` contiguous), ``ix_seqrep [B]`` int32; ``1 <= T <= POOL x NB`` -- a chunk longer than the table's reach is refused
    naming the two numbers (the clamp's precondition, module docstring).  Every operand on the cache's device."""
    if not isinstance(index_kv_cache, torch.Tensor) or index_kv_cache.dim() != 4:
        raise ValueError(f"index_kv_cache must be a 4-D [num_pages, P / pool, 1, D={r.d}] paged compressed-key cache, got {type(index_kv_cache).__name__}")
    c = index_kv_cache
    if c.dtype != r.dtype:
        raise ValueError(
            f"index_kv_cache is {c.dtype} but this artifact gathers a {r.dtype} cache (the recipe's dtype); compile the matching recipe -- no silent cast"
        )
    if int(c.shape[0]) < 1 or int(c.shape[1]) < 1 or int(c.shape[2]) != 1 or int(c.shape[3]) != r.d:
        raise ValueError(f"index_kv_cache must be [num_pages >= 1, P / pool >= 1, 1, D={r.d}], got shape {tuple(c.shape)}")
    if not c.is_cuda:
        raise ValueError(f"index_kv_cache must be a CUDA tensor (the kernel reads every operand through a device pointer), got device {c.device}")
    _check_row_tensor("index_kv_cache", c, dtype=r.dtype, last_dim=r.d)
    dev = c.device
    if not isinstance(block_table, torch.Tensor) or block_table.device != dev:
        raise ValueError(f"block_table must be a [B, max_pages] int32 tensor on {dev}")
    batch = int(block_table.shape[0]) if block_table.dim() == 2 else 0
    check_block_table("block_table", block_table, batch=batch)
    if batch < 1:
        raise ValueError(f"block_table must hold at least one sequence, got shape {tuple(block_table.shape)}")
    _check_i32("kv_lens", kv_lens, batch, dev)
    if not isinstance(piece_seq, torch.Tensor) or piece_seq.dim() != 1 or int(piece_seq.numel()) < 1:
        raise ValueError(f"piece_seq must be a [n_pieces >= 1] int32 vector, got {None if not isinstance(piece_seq, torch.Tensor) else tuple(piece_seq.shape)}")
    n_pieces = int(piece_seq.numel())
    for name, v in (("piece_seq", piece_seq), ("piece_r0", piece_r0), ("piece_rows", piece_rows), ("ix_qpos", ix_qpos)):
        _check_i32(name, v, n_pieces, dev)
    if (
        not isinstance(ix_kbar, torch.Tensor)
        or ix_kbar.dim() != 3
        or int(ix_kbar.shape[0]) != batch
        or int(ix_kbar.shape[2]) != r.d
        or int(ix_kbar.shape[1]) < 1
    ):
        raise ValueError(
            f"ix_kbar must be [B={batch}, n_blocks_max >= 1, D={r.d}], got {None if not isinstance(ix_kbar, torch.Tensor) else tuple(ix_kbar.shape)}"
        )
    if ix_kbar.dtype != r.dtype:
        raise ValueError(f"ix_kbar is {ix_kbar.dtype} but this artifact gathers a {r.dtype} cache (the recipe's dtype); the gather is a bit copy -- no cast")
    if not ix_kbar.is_contiguous():
        raise ValueError(f"ix_kbar must be contiguous (the scorer's operand form), got strides {tuple(ix_kbar.stride())}")
    if ix_kbar.device != dev:
        raise ValueError(f"ix_kbar must live on {dev} with the cache, got {ix_kbar.device}")
    _check_row_tensor("ix_kbar", ix_kbar, dtype=r.dtype, last_dim=r.d)
    n_blocks_max = int(ix_kbar.shape[1])
    reach = int(block_table.shape[1]) * int(c.shape[1])
    if n_blocks_max > reach:
        raise ValueError(
            f"ix_kbar holds {n_blocks_max} rows per sequence but the table reaches {int(block_table.shape[1])} pages x {int(c.shape[1])} rows = {reach}: "
            "rows past the table's reach could only be zeros -- size n_blocks_max as max_pages x P / pool"
        )
    t = int(t_rows)
    if t < 1:
        raise ValueError(f"t_rows (the chunk's rows per sequence) must be >= 1, got {t}")
    if t > r.pool * n_blocks_max:
        raise ValueError(
            f"a chunk of T={t} rows exceeds the table's reach of {r.pool} x {n_blocks_max} = {r.pool * n_blocks_max} tokens (max_pages x P): "
            "the compact-logits guard (q_pos0 <= pool x n_blocks_max + pool - 1 - rows) needs T <= pool x n_blocks_max"
        )
    if (
        not isinstance(ix_counts, torch.Tensor)
        or ix_counts.dtype != torch.int32
        or not ix_counts.is_contiguous()
        or int(ix_counts.numel()) != batch * t
        or ix_counts.device != dev
    ):
        got = None if not isinstance(ix_counts, torch.Tensor) else f"{ix_counts.dtype} {tuple(ix_counts.shape)} contiguous={ix_counts.is_contiguous()}"
        raise ValueError(f"ix_counts must be a contiguous int32 tensor of B x T = {batch} x {t} entries on {dev}, got {got}")
    _check_i32("ix_seqrep", ix_seqrep, batch, dev)
    if dev.index is not None and dev.index != r.device:
        raise ValueError(f"this artifact was compiled for CUDA device {r.device}; the operands live on {dev} (compile one per device)")
    grid_x = (n_blocks_max + r.groups_per_cta - 1) // r.groups_per_cta
    with torch.cuda.device(r.device):  # the launch runs in the artifact's context whichever device is current at the call
        r.compiled(
            index_kv_cache,
            block_table,
            kv_lens,
            piece_seq,
            piece_r0,
            piece_rows,
            ix_kbar,
            ix_qpos,
            ix_counts,
            ix_seqrep,
            cutlass.Int32(n_pieces),
            cutlass.Int32(t),
            cutlass.Int32(grid_x),
            cutlass.Int32(batch),
            cuda.CUstream(int(stream)),
        )


def build_qsa_kbar_gather(
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    piece_seq: torch.Tensor,
    piece_r0: torch.Tensor,
    piece_rows: torch.Tensor,
    ix_kbar: torch.Tensor,
    ix_qpos: torch.Tensor,
    ix_counts: torch.Tensor,
    ix_seqrep: torch.Tensor,
    *,
    t_rows: int,
    pool: int = DEFAULT_POOL,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    use_pdl: bool = False,
    stream,
) -> QsaKbarGatherRecipe:
    """Compile (cached) and run once -- the convenience form for tests.  Production callers split it: :func:`compile_qsa_kbar_gather`
    at plan time, :func:`run_qsa_kbar_gather` per execute."""
    if not isinstance(index_kv_cache, torch.Tensor) or index_kv_cache.dim() != 4:
        raise ValueError("index_kv_cache must be a 4-D [num_pages, P / pool, 1, D] paged compressed-key cache")
    r = compile_qsa_kbar_gather(dtype=index_kv_cache.dtype, d=int(index_kv_cache.shape[3]), pool=pool, threads_per_cta=threads_per_cta, use_pdl=use_pdl)
    run_qsa_kbar_gather(
        r, index_kv_cache, block_table, kv_lens, piece_seq, piece_r0, piece_rows, ix_kbar, ix_qpos, ix_counts, ix_seqrep, t_rows=t_rows, stream=stream
    )
    return r


def moved_bytes(n_keys_total: int, d: int, *, elem_bytes: int = 2) -> int:
    """HBM traffic of one launch -- the denominator for an SOL number: every gathered row read once and written once (the zero-fill
    rows written once); the table words and the metadata are negligible and excluded."""
    return 2 * int(n_keys_total) * int(d) * int(elem_bytes)


frost_qsa_kbar_gather.set_name_prefix("cudnn", remove_cutlass_symbol=True)
