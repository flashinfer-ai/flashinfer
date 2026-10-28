# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fused PREPARE launch of a block-sparse attention step: everything between the projection GEMM and attention in ONE kernel.

A serving step of a block-sparse (``QsaSpec``) layer runs, between the fused ``Q | GATE | K | V | INDEX`` projection and the
attention core, a chain of small pointwise passes over the step's ``T`` tokens -- four pointwise launches in the block's
unfused stage list plus two more in the indexer's decode form::

    qk_norm_rope        Q, K: per-head RMSNorm over D, then partial RoPE on [0, rope_dim)         (stages (2)+(3), in place)
    cache_write (K, V)  the post-RoPE K and the V of every token into the paged pools at slot_mapping     (stage (4w))
    cache_write (raw)   the band's raw indexer key into ITS pool at the same slots                        (stage (4w), band)
    qsa_compress, pool 1   the band's indexer QUERIES: RMSNorm over D_i + partial RoPE at the token       (the scorer's q_i)
    qsa_compress_step   the raw key into the per-sequence RING, the block it completes POOLED,
                        RMSNormed and rotated at the block START into the compressed-key cache            (the indexer's keys)

This kernel is that chain as ONE launch, row for row and bit for bit: every output is produced by the same fp32 operations
in the same order as the kernel it replaces (``qk_norm_rope.py``'s ``x * rstd * w`` then ``rope_rotate_half`` for Q / K;
``qsa_compress.py``'s ``mean -> rstd -> * (w + w_offset) -> RoPE`` for the indexer rows; a bit copy for V, the raw key and
the ring), so the fused step is BITWISE the unfused one -- a PERFORMANCE knob of the block (``fuse_prepare``), never a
numerics choice.  The gate split costs nothing here or there: the GATE band is addressed in place by its slab strides.

Work decomposition -- a flat space of WARP UNITS, three arms, every arm warp-uniform::

    [ attention rows: T x H_q Q rows | T x H_kv K rows | T x H_kv V rows ]  ->  one lane group of lanes_per_row(D) lanes per row
    [ indexer query rows: T x H_i rows of D_i ]                              ->  one lane group of lanes_per_row(D_i) lanes per row
    [ raw-key rows: T rows of D_i ]                                          ->  one lane group of lanes_per_row(D_i) lanes per row

A warp owns ``32 // lanes_per_row`` consecutive rows of ONE arm (one 256-wide row, or two 128-wide rows), so the branch on
the arm is uniform across the warp and every ``shfl.sync`` inside an arm is reached by all 32 lanes; a lane group past
its arm's row count computes on clamped, in-bounds loads and stores nothing (the sibling kernels' tail rule).  Per row: the
loads first (one ``ld.global.v4`` per lane per 16 elements), the sum of squares reduced across the group by a butterfly shuffle, the
norm weight and the table rows consumed AFTER the reduction (``qk_norm_rope.py``'s measured register discipline), one
``st.global.v4`` per destination.  A K row stores the SAME packed words twice -- into the slab (in place) and into the pool
-- so a pool row IS the row attention consumes; a V row and a raw-key row are bit copies.

**No shared memory, no mbarrier, no TMA, no tcgen05.**  Plain vectorized LDG / STG plus warp shuffles, so the two
pre-development tables of a FROST kernel (barriers, SMEM buffers) are EMPTY by construction; the shuffles are the only
cross-lane dependency, inside aligned lane groups of one warp.  Within a launch no row is read after another row writes it:
Q / K rows are rewritten by their own lane group only (every lane holds its full row before the first store), the raw key
band is read by the indexer arms and never written, and the ring's writes and reads of one launch never alias (host rule
``RING >= S_q + pool - 1``, ``qsa_compress.min_ring_rows`` -- the paged ring's ``cap`` is that ``RING``: a completing row reads
ring positions ``pos0 - pool + 1 .. pos0 - 1``, the step writes ``pos0 .. pos0 + S_q - 1``, disjoint rows of the sequence's one
block whenever ``cap >= S_q + pool - 1``; the two tables are read, never written).  Runs on any CuTe DSL device.

The arms are TRACED BY PRESENCE (``None`` folds an arm out at trace time, exactly as the weight presence folds the RMSNorm
out of ``qk_norm_rope``):

* always: Q / K norm + RoPE (RoPE-only when both weights are ``None``), the K / V pool writes at ``slot_mapping``;
* ``index_k_raw_src`` given: the raw-key arm -- the raw key's pool when ``index_k_raw_pool`` is given, the ring + the
  compressed-key cache when ``ring`` is given (``pos0`` and the position-indexed tables ``cos_pos`` / ``sin_pos`` come with
  it; ``n_commit`` optional -- the decode contract of ``qsa_compress.py``'s step form, verbatim);
* ``index_q_src`` given: the indexer-query arm into ``index_q_out`` with ``w_iq_norm``;
* ``block_table`` given (``QsaPrepareIndexArms.paged``): the compress arm's PAGED form -- ``ring`` is the stack's ring
  ``[num_ring_blocks, cap, 1, D_i]`` behind ``index_state_block_table [B, 1]``, ``index_k_compressed`` its compressed-key cache
  ``[num_pages, P / pool, 1, D_i]`` behind the MAIN ``block_table [B, max_pages]``, and the positions derive in-kernel from
  ``kv_lens [B]`` (the KV length AFTER the step: ``pos0 = kv_lens - S_q``) in place of ``pos0``, which is refused; the row
  arithmetic is ``qsa_compress.py``'s (``step_position``, ``ring_row_address``, ``paged_block_row`` -- ONE helper per table,
  two call sites: the step form and this launch); ``index_state_slot_mapping [T]`` (``ring_slots``) names the ring WRITE
  rows instead of deriving them (the stack's own slots; bitwise the derived form);
* ``cos`` / ``sin`` None (``rope_table``): the RoPE-table arm -- the Q / K and the indexer-query rows read ``cos_pos`` /
  ``sin_pos`` at ``pos0 + j``, bitwise the per-token form, so a serving stack hands the model's position-indexed table once
  instead of gathering per-token rows every step.  The table is the CEILING on positions, checked IN-KERNEL on the LIVE
  (device-derived) position of every row, never on the host: a row at ``pos0 + j >= S_pos`` has no rotation, so its Q / K row, its
  K pool row and its indexer-query row are NOT written and the compress arm reports it ``-2`` (no ring row, no block either; the V
  row and the raw key are position-free bit copies and are stored as in the per-token form) -- never a silently clamped rotation.
  The host also refuses BY FORM a table shorter than the positions the compress arm's cache can address (paged: ``max_pages x P``
  rows, the block table's width times the page; dense: ``NB x pool``), so a conforming stack never meets the ceiling with a position
  its cache holds, and the in-kernel clamp serves only the dead rows of a padding sequence (whose rotation nothing consumes).

Table forms -- two, because two kinds of position are involved: ``cos`` / ``sin`` ``[T, rope_dim]`` are the attention's own
PER-TOKEN rows (the block's contract: row ``t`` is token ``t``'s position), read by the Q / K and the indexer-query rows;
``cos_pos`` / ``sin_pos`` ``[1 or B, S_pos, rope_dim]`` are POSITION-indexed (the step form's contract), read by the compress
at the block START ``4 * blk`` -- a position of an EARLIER step that the per-token rows do not hold.

Degenerate inputs, each with its handling: a negative slot (padding) writes no pool row and a slot past the pool's capacity
writes none either (``live``); ``n_commit[b] = 0`` (a finished sequence) writes no ring row and no block, every slot report
``-1``; ``pos0 < 0`` is clamped to 0; a row completing a block past the cache reports ``-2`` and writes no block; a committed row at a
position at or past the RoPE table reports ``-2`` and writes nothing at all (the table ceiling, in both forms of the compress arm and in
the step form); a ragged last warp computes on the last row and stores nothing; a token count below one lane group per arm still launches
(the host never launches an empty grid: ``T >= 1`` is checked).  The paged form adds (``qsa_compress.py``, "The PAGED decode
form"): a row at a negative position (``kv_lens[b] < S_q``; a padding sequence has ``kv_lens 0``) is dead -- no ring row, no
block, slot ``-1`` -- so the null ring block and page 0 are never written for it; a sequence whose state-table entry is ``<= 0``
or past the ring's blocks owns no ring: it writes no ring row and a block of its that needs ring rows (one starting below
``pos0``) reports ``-2`` -- the null block's rows are never pooled into a live sequence's block; a completing block past the
table's width, on a negative page or on a page past the cache reports ``-2``.

The e4m3 STORE arm (``compile_qsa_prepare(index_cache_dtype=torch.float8_e4m3fn)``; the e4m3 compressed-key cache at decode, bf16 io
only): the indexer-query arm's ``index_q_out`` and the compress arm's ``index_k_compressed`` (dense or paged) are e4m3 -- each row the
bf16 form's packed words rounded ONCE more by ``cvt.rn.satfinite.e4m3x2.f32`` (``qsa_compress._store_row_chunk_e4m3``: the e4m3 cast OF
THE bf16 ROW, one ``st.global.v2.b32`` of 8 B per lane), the SAME helper and convention as the step form and the pool-1 compress, so the
fused launch stays bitwise the unfused chain under the attribute too.  The ring stays the io dtype (a bit copy, never quantized -- an e4m3
ring is a form error naming that rule); the fp8 ``cvt`` needs sm_89+ (an older part is declined by name at compile time, Rule 7).

Bytes per launch (the roofline term is HBM bytes; no perf claim is made here until one is measured on an exclusive GPU):
Q and K read once and written once, V read once and written once (the pool), the raw key read up to three times (pool,
ring, its own block) and written to the pool and the ring, the indexer queries read once and written once, plus the slot
and the small tables -- :func:`moved_bytes`.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16, rmsnorm_rstd, rope_rotate_half
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, ld_global_v2, ld_global_v4, st_global, st_global_v4

from ..cache_write_contract import QsaPrepareIndexArms, check_pool, check_slot_mapping  # the arms record: re-exported for this kernel's callers
from .cache_write import _fake_pool, _fake_slot
from .qk_norm_rope import (
    ACCESS_BYTES,
    ELEMS_PER_ACCESS,
    _fake,
    check_norm_weights_match_recipe,
    fake_rowmajor_dynamic_token_stride,
    lanes_per_row,
    validate_shape,
    vec_chunks,
)
from .qsa_compress import (
    DEFAULT_POOL,
    SLOT_CAPACITY,
    SLOT_NO_BLOCK,
    _check_int32_vector,
    _check_row_tensor,
    _fake_compact,
    _fake_i32_table,
    _fake_paged_state,
    _fake_strided,
    _store_row_chunk_e4m3,
    check_e4m3_store_target,
    check_index_cache_dtype,
    check_paged_state,
    check_paged_step_operands,
    min_ring_rows,
    paged_block_row,
    ring_entry,
    ring_row_address,
    ring_slot_address,
    step_position,
    validate_compress_shape,
)

DEFAULT_THREADS_PER_CTA = 128
"""Four warps per CTA.  Every row is one lane group, one row per group (no rows-in-flight knob): the step is a few hundred
rows on a 200-SM part -- latency-bound -- so parallelism, not per-thread state, is what it wants; re-measure per arch
before changing it."""

_IO_DTYPES = (torch.bfloat16, torch.float16)
_SLOT_DTYPES = (torch.int32, torch.int64)
_FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
_WARP = 32


def validate_prepare_shape(d: int, rope_dim: int, page_size: int, threads_per_cta: int, index_head_dim: Optional[int] = None, pool: int = DEFAULT_POOL) -> None:
    """Raise on any geometry this kernel cannot address: the attention rows' geometry is ``qk_norm_rope``'s
    (``validate_shape``), the indexer rows' is ``qsa_compress``'s (``validate_compress_shape``, which also takes the pool),
    the page size has to be positive (its multiple-of-16 contract is the API's), and a CTA holds whole warps."""
    validate_shape(d, rope_dim, threads_per_cta)
    if index_head_dim is not None:
        validate_compress_shape(index_head_dim, rope_dim, pool, threads_per_cta)
    if int(page_size) < 1:
        raise ValueError(f"page_size must be >= 1, got {page_size}")
    if threads_per_cta % _WARP or threads_per_cta < _WARP:
        raise ValueError(f"threads_per_cta must be a positive multiple of {_WARP} (a warp owns whole rows of one arm), got {threads_per_cta}")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def prepare_warp_counts(
    t: int, *, h_q: int, h_kv: int, d: int, index_heads: int = 0, index_head_dim: int = 0, index_q: bool = False, index_raw: bool = False
) -> tuple:
    """The warp units of one launch, per arm: ``(attention warps, indexer-query warps, raw-key warps)`` for ``t`` tokens
    -- the SAME arithmetic the kernel repeats on device from its runtime token count (``_ceil_div(rows, 32 // lanes)``), so
    the host's grid and the kernel's arm boundaries cannot drift."""
    rpw_a = _WARP // lanes_per_row(d)
    attn = _ceil_div(t * (h_q + 2 * h_kv), rpw_a)
    rpw_i = _WARP // lanes_per_row(index_head_dim) if index_head_dim else 1
    qi = _ceil_div(t * index_heads, rpw_i) if index_q else 0
    raw = _ceil_div(t, rpw_i) if index_raw else 0
    return attn, qi, raw


@cute.jit
def _step_row_position(mPos0: Optional[cute.Tensor], mKvLens: Optional[cute.Tensor], token: cutlass.Int32, s_q: cutlass.Int32):
    """The step position of token ``token`` of a ``B x S_q`` step and its sequence, ``(pos0(b) + j, b)`` with ``b = token // S_q``,
    ``j = token % S_q`` and ``pos0`` from ``kv_lens`` or the ``pos0`` vector exactly as the compress arm derives it
    (:func:`qsa_compress.step_position`), the position clamped at 0 (the dead rows of a padding sequence).  Under the RoPE-table arm
    EVERY lane of a row computes it: the rotation reads the table at it, and the row's stores are gated on ``pos < S_pos`` (the table
    is the ceiling on positions -- a row at or past it stores nothing rotated and the compress arm reports it ``-2``)."""
    b_t = token // s_q
    j_t = token - b_t * s_q
    pos = step_position(mPos0, mKvLens, b_t, s_q) + j_t
    pos = pos if pos > cutlass.Int32(0) else cutlass.Int32(0)
    return pos, b_t


@cute.jit
def _rope_row_addrs(
    mCos: Optional[cute.Tensor],
    mSin: Optional[cute.Tensor],
    mCosPos: Optional[cute.Tensor],
    mSinPos: Optional[cute.Tensor],
    pos: cutlass.Int32,
    b_t: cutlass.Int32,
    token: cutlass.Int32,
    lane_off: cutlass.Int64,
    bpe: cutlass.Int64,
):
    """The cos / sin row addresses of one token's rotation: the per-token rows ``mCos[t]`` / ``mSin[t]`` (the block's contract),
    or -- the RoPE-table arm, ``mCos`` None -- the position-indexed tables at ``pos`` (the row's step position from
    :func:`_step_row_position`, ``b_t`` its sequence), clamped into ``[0, S_pos)`` -- memory safety only: a row at or past the table
    stores nothing rotated (the arms gate their stores on ``pos < S_pos``) and the compress arm reports it ``-2``, so the clamped
    row is never consumed; :func:`run_qsa_prepare` also refuses by form a table shorter than the positions the compress arm's cache
    can address (paged ``max_pages x P``, dense ``NB x pool``), so a conforming stack never meets the ceiling with a live row."""
    token64 = token.to(cutlass.Int64)
    if cutlass.const_expr(mCos is None):
        s_pos = cutlass.Int32(mCosPos.shape[1])
        pos_r = pos if pos < s_pos else s_pos - cutlass.Int32(1)
        bc = b_t if cutlass.Int32(mCosPos.shape[0]) > cutlass.Int32(1) else cutlass.Int32(0)
        bc64 = bc.to(cutlass.Int64)
        pos64 = pos_r.to(cutlass.Int64)
        c_addr = mCosPos.iterator.toint() + (bc64 * cutlass.Int64(mCosPos.stride[0]) + pos64 * cutlass.Int64(mCosPos.stride[1])) * bpe + lane_off
        s_addr = mSinPos.iterator.toint() + (bc64 * cutlass.Int64(mSinPos.stride[0]) + pos64 * cutlass.Int64(mSinPos.stride[1])) * bpe + lane_off
    else:
        c_addr = mCos.iterator.toint() + token64 * cutlass.Int64(mCos.stride[0]) * bpe + lane_off
        s_addr = mSin.iterator.toint() + token64 * cutlass.Int64(mSin.stride[0]) * bpe + lane_off
    return c_addr, s_addr


@cute.kernel
def frost_qsa_prepare(
    mQ: cute.Tensor,  # [T, H_q,  D] in: the slab's Q band (token stride symbolic)
    mK: cute.Tensor,  # [T, H_kv, D] in: the K band
    mV: cute.Tensor,  # [T, H_kv, D] in: the V band (never modified; copied into its pool)
    mQo: cute.Tensor,  # [T, H_q,  D] out (may alias mQ)
    mKo: cute.Tensor,  # [T, H_kv, D] out (may alias mK)
    mWq: Optional[cute.Tensor],  # [D], or None: RoPE-only Q / K (no RMSNorm, no weight loads)
    mWk: Optional[cute.Tensor],  # [D], or None (both or neither -- the host checks)
    mCos: Optional[cute.Tensor],  # [T, ROPE_DIM] per-token rows (the attention's tables), or None: the RoPE-table arm reads mCosPos / mSinPos at pos0 + j
    mSin: Optional[cute.Tensor],  # [T, ROPE_DIM], or None with mCos
    mKPool: cute.Tensor,  # [P, H_kv, page_size, D]: strides symbolic (HND compact or NHD), D contiguous
    mVPool: cute.Tensor,  # same form
    mSlot: cute.Tensor,  # [T] int32 or int64, contiguous
    mIKsrc: Optional[cute.Tensor],  # [T, 1, D_i] the band's raw indexer key (token stride symbolic), or None (no raw-key arm)
    mIKpool: Optional[cute.Tensor],  # [P, 1, page_size, D_i] the raw key's pool, or None
    mIQsrc: Optional[cute.Tensor],  # [T, H_i, D_i] the band's indexer query heads (token stride symbolic), or None (no query arm)
    mIQout: Optional[cute.Tensor],  # [T, H_i, D_i] compact out
    mWiq: Optional[cute.Tensor],  # [D_i] the indexer queries' norm weight
    mRing: Optional[cute.Tensor],  # [B, RING, D_i] compact: row p % RING holds position p's raw key, or None (no compress arm); paged: [blocks, cap, 1, D_i]
    mComp: Optional[cute.Tensor],  # [B, NB, D_i] compact: the compressed-key cache, row blk = block blk; paged: [num_pages, P / pool, 1, D_i]
    mWik: Optional[cute.Tensor],  # [D_i] the compressed keys' norm weight
    mCosPos: Optional[cute.Tensor],  # [Bc, S_pos, ROPE_DIM] position-indexed, Bc in {1, B}
    mSinPos: Optional[cute.Tensor],  # same shape
    mPos0: Optional[cute.Tensor],  # [B] int32: the position of row 0 of every sequence (= the KV length before the step); None on the paged form
    mNCommit: Optional[cute.Tensor],  # [B] int32: the accepted prefix of the step's rows, or None (= every row)
    mCSlot: Optional[cute.Tensor],  # [B, S_q] int32 out: the block id written per row, -1 (none) or -2 (capacity)
    mKvLens: Optional[cute.Tensor],  # [B] int32: the KV length AFTER the step (the paged form's position source), or None
    mBlockTable: Optional[cute.Tensor],  # [B, max_pages] int32: the MAIN block table (the paged form), or None
    mStateTable: Optional[cute.Tensor],  # [B, 1] int32: the ring's block table (the paged form), or None
    mRingSlots: Optional[cute.Tensor],  # [T] int32 / int64 (the slot dtype): the stack's own ring slots (paged, ring_slots), or None
    n_tok: cutlass.Int32,  # T
    s_q: cutlass.Int32,  # rows per sequence (T = B * S_q); only the compress arm reads it
    n_slots: cutlass.Int64,  # P * page_size of the SMALLER K / V pool: a slot outside [0, n_slots) writes nothing
    n_islots: cutlass.Int64,  # the raw key's pool capacity (0 when it is absent)
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    h_i: cutlass.Int32,
    eps: cutlass.Float32,  # the attention norms' epsilon
    eps_i: cutlass.Float32,  # the indexer norms' epsilon
    w_offset_i: cutlass.Float32,  # the indexer norms' weight offset (0.0 = the pre-folded (1 + w) form, 1.0 = the checkpoint's w)
    d: cutlass.Constexpr[int],
    d_i: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
    slot_i64: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    h_i_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """One warp = ``32 // lanes_per_row`` consecutive rows of one arm (module docstring); the arm boundaries are the warp
    counts :func:`prepare_warp_counts` derives, recomputed here from ``n_tok`` so the host's grid and the device's row map
    share one formula."""
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    apply_norm = cutlass.const_expr(mWq is not None)
    has_raw = cutlass.const_expr(mIKsrc is not None)
    has_raw_pool = cutlass.const_expr(mIKpool is not None)
    has_qi = cutlass.const_expr(mIQsrc is not None)
    has_comp = cutlass.const_expr(mRing is not None)
    has_n_commit = cutlass.const_expr(mNCommit is not None)
    paged = cutlass.const_expr(mBlockTable is not None)
    has_ring_slots = cutlass.const_expr(mRingSlots is not None)
    table_dtype = mCosPos.element_type if cutlass.const_expr(mCos is None) else mCos.element_type  # the RoPE-table arm reads the position-indexed tables
    warps_per_cta = cutlass.const_expr(threads_per_cta // _WARP)
    lanes_a = cutlass.const_expr(lanes_per_row(d))
    chunks_a = cutlass.const_expr(vec_chunks(d))
    rpw_a = cutlass.const_expr(_WARP // lanes_a)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)
    lanes_i = cutlass.const_expr(lanes_per_row(d_i))
    chunks_i = cutlass.const_expr(vec_chunks(d_i))
    rpw_i = cutlass.const_expr(_WARP // lanes_i)
    bpe = cutlass.Int64(2)  # bf16 / f16: the only element widths this kernel serves (the host refuses the rest)

    _hq = cutlass.Int32(h_q_ct) if cutlass.const_expr(const_head_counts) else h_q
    _hkv = cutlass.Int32(h_kv_ct) if cutlass.const_expr(const_head_counts) else h_kv
    _hi = cutlass.Int32(h_i_ct) if cutlass.const_expr(const_head_counts) else h_i

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane32 = tidx % cutlass.Int32(_WARP)
    wid = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(warps_per_cta) + tidx // cutlass.Int32(_WARP)

    # The arm boundaries, in warps -- prepare_warp_counts on device.
    n_q_rows = n_tok * _hq
    n_qk_rows = n_q_rows + n_tok * _hkv
    n_attn_rows = n_qk_rows + n_tok * _hkv
    n_attn_warps = (n_attn_rows + cutlass.Int32(rpw_a - 1)) // cutlass.Int32(rpw_a)
    n_qi_rows = (n_tok * _hi) if cutlass.const_expr(has_qi) else cutlass.Int32(0)
    n_qi_warps = ((n_qi_rows + cutlass.Int32(rpw_i - 1)) // cutlass.Int32(rpw_i)) if cutlass.const_expr(has_qi) else cutlass.Int32(0)

    if wid < n_attn_warps:
        # ---------------- the attention arm: Q / K norm + RoPE in place, K / V into the pools -----------------------------
        lane = lane32 % cutlass.Int32(lanes_a)
        row = wid * cutlass.Int32(rpw_a) + lane32 // cutlass.Int32(lanes_a)
        valid = row < n_attn_rows
        row_r = row if valid else n_attn_rows - cutlass.Int32(1)
        is_q = row_r < n_q_rows
        is_v = row_r >= n_qk_rows
        is_k = (~is_q) & (~is_v)
        k_row = row_r - n_q_rows
        v_row = row_r - n_qk_rows
        # token / head: the Q band's at h_q heads, the K / V bands' at h_kv (compile-time shifts under const_head_counts).
        tok_q = row_r // _hq
        tok_k = k_row // _hkv
        tok_v = v_row // _hkv
        token = tok_q if is_q else (tok_v if is_v else tok_k)
        head = (row_r - tok_q * _hq) if is_q else ((v_row - tok_v * _hkv) if is_v else (k_row - tok_k * _hkv))
        token64 = token.to(cutlass.Int64)
        head64 = head.to(cutlass.Int64)
        lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
        # Operand select per row: Int64 scalars, no divergent branch (the cache-write kernel's form).
        src_base = mQ.iterator.toint() if is_q else (mV.iterator.toint() if is_v else mK.iterator.toint())
        src_tok = cutlass.Int64(mQ.stride[0]) if is_q else (cutlass.Int64(mV.stride[0]) if is_v else cutlass.Int64(mK.stride[0]))
        src_head = cutlass.Int64(mQ.stride[1]) if is_q else (cutlass.Int64(mV.stride[1]) if is_v else cutlass.Int64(mK.stride[1]))
        dst_base = mQo.iterator.toint() if is_q else mKo.iterator.toint()  # a V row never stores here
        dst_tok = cutlass.Int64(mQo.stride[0]) if is_q else cutlass.Int64(mKo.stride[0])
        dst_head = cutlass.Int64(mQo.stride[1]) if is_q else cutlass.Int64(mKo.stride[1])
        src_addr = src_base + (token64 * src_tok + head64 * src_head) * bpe
        dst_addr = dst_base + (token64 * dst_tok + head64 * dst_head) * bpe
        # The slot of this token and the pool row it names (K rows take the K pool, V rows the V pool; a Q row's values are
        # computed and never used).  int64 slots: two 32-bit halves recombined so a slot >= 2^31 can never alias an in-range one.
        if cutlass.const_expr(slot_i64):
            lo, hi = ld_global_v2(mSlot.iterator.toint() + token64 * cutlass.Int64(8), cutlass.Int32)
            slot = (hi.to(cutlass.Int64) * cutlass.Int64(1 << 32)) + (lo.to(cutlass.Int64) & cutlass.Int64(0xFFFFFFFF))
        else:
            slot = ld_global(mSlot.iterator.toint() + token64 * cutlass.Int64(4), cutlass.Int32).to(cutlass.Int64)
        live = valid & (slot >= cutlass.Int64(0)) & (slot < n_slots)
        slot32 = slot.to(cutlass.Int32)
        page = slot32 // cutlass.Int32(page_size)
        off = slot32 - page * cutlass.Int32(page_size)
        pool_base = mVPool.iterator.toint() if is_v else mKPool.iterator.toint()
        p0 = cutlass.Int64(mVPool.stride[0]) if is_v else cutlass.Int64(mKPool.stride[0])
        p1 = cutlass.Int64(mVPool.stride[1]) if is_v else cutlass.Int64(mKPool.stride[1])
        p2 = cutlass.Int64(mVPool.stride[2]) if is_v else cutlass.Int64(mKPool.stride[2])
        pool_addr = pool_base + (page.to(cutlass.Int64) * p0 + head64 * p1 + off.to(cutlass.Int64) * p2) * bpe
        # The RoPE-table arm: the row's step position on EVERY lane (the rotation reads the table at it) and the store gates -- a row at
        # or past the table has no rotation (the clamp in _rope_row_addrs is memory safety only): its Q / K slab row and its K pool row
        # are NOT written, and the compress arm reports it -2; the V row is a position-free bit copy, stored as in the per-token form.
        store_slab = valid & (~is_v)
        store_kpool = live & is_k
        pos_t = cutlass.Int32(0)
        b_t = cutlass.Int32(0)
        if cutlass.const_expr(mCos is None):
            pos_t, b_t = _step_row_position(mPos0, mKvLens, token, s_q)
            in_table = pos_t < cutlass.Int32(mCosPos.shape[1])
            store_slab = store_slab & in_table
            store_kpool = store_kpool & in_table

        # PASS 1: the row, as packed words (the V copy) and as fp32 (the norm).
        raw_words = []
        xs = []
        for c in cutlass.range_constexpr(chunks_a):
            coff = cutlass.Int64((c * lanes_a) * ACCESS_BYTES) + lane_off
            words = ld_global_v4(src_addr + coff, cutlass.Int32)
            raw_words.append(words)
            pairs = [f16x2_to_f32(w, dtype=mQ.element_type) for w in words]
            xs.append([v for pair in pairs for v in pair])

        # PASS 2: qk_norm_rope.py's chain, op for op.  The weight and the table rows are loaded AFTER the reduction.
        rstd = cutlass.Float32(1.0)
        if cutlass.const_expr(apply_norm):
            acc = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(chunks_a):
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    acc = acc + xs[c][i] * xs[c][i]
            rstd = rmsnorm_rstd(acc, lanes_a, d, eps)
        ys = []
        if cutlass.const_expr(apply_norm):
            w_addr = mWq.iterator.toint() if is_q else mWk.iterator.toint()
            for c in cutlass.range_constexpr(chunks_a):
                coff = cutlass.Int64((c * lanes_a) * ACCESS_BYTES) + lane_off
                w_pairs = [f16x2_to_f32(w, dtype=mWq.element_type) for w in ld_global_v4(w_addr + coff, cutlass.Int32)]
                w_vals = [v for pair in w_pairs for v in pair]
                ys.append([xs[c][i] * rstd * w_vals[i] for i in range(ELEMS_PER_ACCESS)])
        else:
            for c in cutlass.range_constexpr(chunks_a):
                ys.append(list(xs[c]))
        if cutlass.const_expr(rope_dim > 0):
            in_rope = lane < cutlass.Int32(rope_lanes)
            cos_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
            sin_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
            if in_rope:
                c_addr, s_addr = _rope_row_addrs(mCos, mSin, mCosPos, mSinPos, pos_t, b_t, token, lane_off, bpe)
                c_pairs = [f16x2_to_f32(w, dtype=table_dtype) for w in ld_global_v4(c_addr, cutlass.Int32)]
                s_pairs = [f16x2_to_f32(w, dtype=table_dtype) for w in ld_global_v4(s_addr, cutlass.Int32)]
                cos_v = [v for pair in c_pairs for v in pair]
                sin_v = [v for pair in s_pairs for v in pair]
            # The shuffle inside rope_rotate_half is unconditional (every lane of the warp reaches it); only the rope lanes keep it.
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                rotated = rope_rotate_half(ys[0][i], cos_v[i], sin_v[i], lane, half_lanes)
                ys[0][i] = rotated if in_rope else ys[0][i]

        # PASS 3: the stores -- Q / K into the slab (one rounding), K / V into the pools (K: the SAME packed words).
        out_words = []
        for c in cutlass.range_constexpr(chunks_a):
            y = ys[c]
            out_words.append([fp32_to_fp16(y[i], y[i + 1], dtype=mQo.element_type) for i in range(0, ELEMS_PER_ACCESS, 2)])
        if store_slab:
            for c in cutlass.range_constexpr(chunks_a):
                coff = cutlass.Int64((c * lanes_a) * ACCESS_BYTES) + lane_off
                st_global_v4(dst_addr + coff, out_words[c], cutlass.Int32)
        if store_kpool:
            for c in cutlass.range_constexpr(chunks_a):
                coff = cutlass.Int64((c * lanes_a) * ACCESS_BYTES) + lane_off
                st_global_v4(pool_addr + coff, out_words[c], cutlass.Int32)
        if live & is_v:
            for c in cutlass.range_constexpr(chunks_a):
                coff = cutlass.Int64((c * lanes_a) * ACCESS_BYTES) + lane_off
                st_global_v4(pool_addr + coff, raw_words[c], cutlass.Int32)
    elif wid < n_attn_warps + n_qi_warps:
        # ---------------- the indexer-query arm: qsa_compress.py at pool = 1, batch = token, block = head -------------------
        if cutlass.const_expr(has_qi):
            lane = lane32 % cutlass.Int32(lanes_i)
            idx = (wid - n_attn_warps) * cutlass.Int32(rpw_i) + lane32 // cutlass.Int32(lanes_i)
            valid = idx < n_qi_rows
            idx_r = idx if valid else n_qi_rows - cutlass.Int32(1)
            token = idx_r // _hi
            head = idx_r - token * _hi
            token64 = token.to(cutlass.Int64)
            head64 = head.to(cutlass.Int64)
            lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
            src_addr = mIQsrc.iterator.toint() + (token64 * cutlass.Int64(mIQsrc.stride[0]) + head64 * cutlass.Int64(mIQsrc.stride[1])) * bpe
            # the RoPE-table arm: the row's step position on every lane and the store gate (a row at or past the table stores nothing)
            store_qi = valid
            pos_t = cutlass.Int32(0)
            b_t = cutlass.Int32(0)
            if cutlass.const_expr(mCos is None):
                pos_t, b_t = _step_row_position(mPos0, mKvLens, token, s_q)
                store_qi = store_qi & (pos_t < cutlass.Int32(mCosPos.shape[1]))
            # pool = 1: acc = 0 + x, pooled = acc * 1.0 -- the compress kernel's own arithmetic, kept so the bits agree.
            acc = []
            for _ in cutlass.range_constexpr(chunks_i):
                acc.append([cutlass.Float32(0.0)] * ELEMS_PER_ACCESS)
            for c in cutlass.range_constexpr(chunks_i):
                coff = cutlass.Int64((c * lanes_i) * ACCESS_BYTES) + lane_off
                pairs = [f16x2_to_f32(wd, dtype=mIQsrc.element_type) for wd in ld_global_v4(src_addr + coff, cutlass.Int32)]
                vals = [v for pair in pairs for v in pair]
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    acc[c][i] = acc[c][i] + vals[i]
            inv_pool = cutlass.Float32(1.0)
            pooled = []
            for c in cutlass.range_constexpr(chunks_i):
                pooled.append([acc[c][i] * inv_pool for i in range(ELEMS_PER_ACCESS)])
            ssq = cutlass.Float32(0.0)
            for c in cutlass.range_constexpr(chunks_i):
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    ssq = ssq + pooled[c][i] * pooled[c][i]
            rstd = rmsnorm_rstd(ssq, lanes_i, d_i, eps_i)
            w_base = mWiq.iterator.toint()
            ys = []
            for c in cutlass.range_constexpr(chunks_i):
                coff = cutlass.Int64((c * lanes_i) * ACCESS_BYTES) + lane_off
                w_pairs = [f16x2_to_f32(wd, dtype=mWiq.element_type) for wd in ld_global_v4(w_base + coff, cutlass.Int32)]
                w_vals = [v for pair in w_pairs for v in pair]
                ys.append([pooled[c][i] * rstd * (w_vals[i] + w_offset_i) for i in range(ELEMS_PER_ACCESS)])
            if cutlass.const_expr(rope_dim > 0):
                in_rope = lane < cutlass.Int32(rope_lanes)
                cos_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                sin_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                if in_rope:
                    c_addr, s_addr = _rope_row_addrs(mCos, mSin, mCosPos, mSinPos, pos_t, b_t, token, lane_off, bpe)
                    c_pairs = [f16x2_to_f32(wd, dtype=table_dtype) for wd in ld_global_v4(c_addr, cutlass.Int32)]
                    s_pairs = [f16x2_to_f32(wd, dtype=table_dtype) for wd in ld_global_v4(s_addr, cutlass.Int32)]
                    cos_v = [v for pair in c_pairs for v in pair]
                    sin_v = [v for pair in s_pairs for v in pair]
                for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                    rotated = rope_rotate_half(ys[0][i], cos_v[i], sin_v[i], lane, half_lanes)
                    ys[0][i] = rotated if in_rope else ys[0][i]
            if store_qi:
                iq_fp8 = cutlass.const_expr(mIQout.element_type == cutlass.Float8E4M3FN)  # the e4m3 store arm: 1 B per element
                iq_bpe = cutlass.Int64(1) if cutlass.const_expr(iq_fp8) else bpe
                out_base = mIQout.iterator.toint() + (token64 * cutlass.Int64(mIQout.stride[0]) + head64 * cutlass.Int64(mIQout.stride[1])) * iq_bpe
                for c in cutlass.range_constexpr(chunks_i):
                    y = ys[c]
                    if cutlass.const_expr(iq_fp8):
                        _store_row_chunk_e4m3(out_base, c, lanes_i, lane, y, mIQsrc.element_type)
                    else:
                        coff = cutlass.Int64((c * lanes_i) * ACCESS_BYTES) + lane_off
                        packed = [fp32_to_fp16(y[i], y[i + 1], dtype=mIQout.element_type) for i in range(0, ELEMS_PER_ACCESS, 2)]
                        st_global_v4(out_base + coff, packed, cutlass.Int32)
    else:
        # ---------------- the raw-key arm: the key's pool, the ring and the block it completes (qsa_compress_step) ----------
        if cutlass.const_expr(has_raw):
            lane = lane32 % cutlass.Int32(lanes_i)
            idx = (wid - n_attn_warps - n_qi_warps) * cutlass.Int32(rpw_i) + lane32 // cutlass.Int32(lanes_i)
            in_row = idx < n_tok
            t = idx if in_row else n_tok - cutlass.Int32(1)
            t64 = t.to(cutlass.Int64)
            lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
            raw_tok = cutlass.Int64(mIKsrc.stride[0]) * bpe
            raw_addr = mIKsrc.iterator.toint() + t64 * raw_tok
            # (a) the raw key into ITS pool at the token's slot: a bit copy (the cache-write kernel's raw-key launch).
            if cutlass.const_expr(has_raw_pool):
                if cutlass.const_expr(slot_i64):
                    lo, hi = ld_global_v2(mSlot.iterator.toint() + t64 * cutlass.Int64(8), cutlass.Int32)
                    slot = (hi.to(cutlass.Int64) * cutlass.Int64(1 << 32)) + (lo.to(cutlass.Int64) & cutlass.Int64(0xFFFFFFFF))
                else:
                    slot = ld_global(mSlot.iterator.toint() + t64 * cutlass.Int64(4), cutlass.Int32).to(cutlass.Int64)
                live = in_row & (slot >= cutlass.Int64(0)) & (slot < n_islots)
                slot32 = slot.to(cutlass.Int32)
                page = slot32 // cutlass.Int32(page_size)
                off = slot32 - page * cutlass.Int32(page_size)
                pool_addr = (
                    mIKpool.iterator.toint()
                    + (page.to(cutlass.Int64) * cutlass.Int64(mIKpool.stride[0]) + off.to(cutlass.Int64) * cutlass.Int64(mIKpool.stride[2])) * bpe
                )
                if live:
                    for c in cutlass.range_constexpr(chunks_i):
                        coff = cutlass.Int64((c * lanes_i) * ACCESS_BYTES) + lane_off
                        words = ld_global_v4(raw_addr + coff, cutlass.Int32)
                        st_global_v4(pool_addr + coff, words, cutlass.Int32)
            # (b) the step form of the compress: the ring write and the block this row completes -- qsa_compress_step's body over
            # the SAME helpers (dense: the sequence's ring / cache rows; paged: the rows the two block tables name).
            if cutlass.const_expr(has_comp):
                b = t // s_q
                jr = t - b * s_q
                b64 = b.to(cutlass.Int64)
                pos0 = step_position(mPos0, mKvLens, b, s_q)
                n_commit = s_q
                if cutlass.const_expr(has_n_commit):
                    nc = ld_global(mNCommit.iterator.toint() + b64 * cutlass.Int64(4), cutlass.Int32)
                    nc = nc if nc > cutlass.Int32(0) else cutlass.Int32(0)
                    n_commit = nc if nc < s_q else s_q
                p_raw = pos0 + jr
                pos_ok = p_raw >= cutlass.Int32(0)  # always true on the dense form (pos0 clamped); a dead row of the paged form otherwise
                p = p_raw if pos_ok else cutlass.Int32(0)
                committed = in_row & (jr < n_commit) & pos_ok
                completes = (p % cutlass.Int32(pool)) == cutlass.Int32(pool - 1)
                blk = p // cutlass.Int32(pool)
                tok0 = blk * cutlass.Int32(pool)
                # the ring entry (the sequence, or its state block) and the block's cache row -- the one place the two forms differ
                comp_fp8 = cutlass.const_expr(mComp.element_type == cutlass.Float8E4M3FN)  # the e4m3 store arm: 1 B per element
                comp_bpe = cutlass.Int64(1) if cutlass.const_expr(comp_fp8) else bpe
                if cutlass.const_expr(paged):
                    entry, owns_ring = ring_entry(mStateTable, b, mRing)
                    out_row, fits = paged_block_row(mComp, mBlockTable, b, blk, comp_bpe)
                    # a block that needs ring rows (it starts below pos0) from a sequence that owns no ring: -2, never the null block's rows
                    fits = fits & (owns_ring | (tok0 >= pos0))
                else:
                    entry = b
                    owns_ring = cutlass.Boolean(True)
                    n_cap = cutlass.Int32(mComp.shape[1])
                    fits = blk < n_cap
                    blk_r = blk if fits else n_cap - cutlass.Int32(1)
                    out_row = (
                        mComp.iterator.toint() + (b64 * cutlass.Int64(mComp.stride[0]) + blk_r.to(cutlass.Int64) * cutlass.Int64(mComp.stride[1])) * comp_bpe
                    )
                # the RoPE table is the CEILING on positions (qsa_compress_step's rule): a committed row at p >= S_pos writes nothing -- no ring
                # row, no block (and no Q / K / indexer-query row under the RoPE-table arm) -- and reports -2; a block's rows all lie at or
                # below its completing row's position, so no block ever pools a ring row at or past the table
                in_table = cutlass.Boolean(True)
                past_table = cutlass.Boolean(False)
                if cutlass.const_expr(rope_dim > 0):
                    s_pos = cutlass.Int32(mCosPos.shape[1])
                    in_table = p < s_pos
                    past_table = p >= s_pos
                write_blk = committed & completes & fits & in_table
                # the committed row's raw key enters the ring (a bit copy): the derived row, or the stack's own slot (never a row at or past the table)
                ring_write = committed & owns_ring & in_table
                if cutlass.const_expr(has_ring_slots):
                    dst, slot_live = ring_slot_address(mRing, mRingSlots, t, slot_i64, bpe)
                    ring_write = ring_write & slot_live
                else:
                    dst = ring_row_address(mRing, entry, p, bpe)
                if ring_write:
                    for c in cutlass.range_constexpr(chunks_i):
                        coff = cutlass.Int64((c * lanes_i) * ACCESS_BYTES) + lane_off
                        words = ld_global_v4(raw_addr + coff, cutlass.Int32)
                        st_global_v4(dst + coff, words, cutlass.Int32)
                # pool the block's rows in POSITION order: below pos0 from the ring, at or above it from this step's rows
                seq_base = mIKsrc.iterator.toint() + (b * s_q).to(cutlass.Int64) * raw_tok
                acc = []
                for _ in cutlass.range_constexpr(chunks_i):
                    acc.append([cutlass.Float32(0.0)] * ELEMS_PER_ACCESS)
                for r in cutlass.range_constexpr(pool):
                    q = tok0 + cutlass.Int32(r)
                    from_new = q >= pos0
                    row_new = q - pos0
                    row_new = row_new if row_new > cutlass.Int32(0) else cutlass.Int32(0)
                    row_new = row_new if row_new < s_q else s_q - cutlass.Int32(1)
                    addr_new = seq_base + row_new.to(cutlass.Int64) * raw_tok
                    addr_ring = ring_row_address(mRing, entry, q, bpe)
                    row_base = addr_new if from_new else addr_ring
                    for c in cutlass.range_constexpr(chunks_i):
                        coff = cutlass.Int64((c * lanes_i) * ACCESS_BYTES) + lane_off
                        pairs = [f16x2_to_f32(wd, dtype=mIKsrc.element_type) for wd in ld_global_v4(row_base + coff, cutlass.Int32)]
                        vals = [v for pair in pairs for v in pair]
                        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                            acc[c][i] = acc[c][i] + vals[i]
                inv_pool = cutlass.Float32(1.0 / pool)
                pooled = []
                for c in cutlass.range_constexpr(chunks_i):
                    pooled.append([acc[c][i] * inv_pool for i in range(ELEMS_PER_ACCESS)])
                ssq = cutlass.Float32(0.0)
                for c in cutlass.range_constexpr(chunks_i):
                    for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                        ssq = ssq + pooled[c][i] * pooled[c][i]
                rstd = rmsnorm_rstd(ssq, lanes_i, d_i, eps_i)
                w_base = mWik.iterator.toint()
                ys = []
                for c in cutlass.range_constexpr(chunks_i):
                    coff = cutlass.Int64((c * lanes_i) * ACCESS_BYTES) + lane_off
                    w_pairs = [f16x2_to_f32(wd, dtype=mWik.element_type) for wd in ld_global_v4(w_base + coff, cutlass.Int32)]
                    w_vals = [v for pair in w_pairs for v in pair]
                    ys.append([pooled[c][i] * rstd * (w_vals[i] + w_offset_i) for i in range(ELEMS_PER_ACCESS)])
                if cutlass.const_expr(rope_dim > 0):
                    in_rope = lane < cutlass.Int32(rope_lanes)
                    bc = b if cutlass.Int32(mCosPos.shape[0]) > cutlass.Int32(1) else cutlass.Int32(0)
                    bc64 = bc.to(cutlass.Int64)
                    tok0_r = tok0 if tok0 < s_pos else s_pos - cutlass.Int32(1)
                    pos64 = tok0_r.to(cutlass.Int64)
                    cos_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                    sin_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
                    if in_rope:
                        c_addr = (
                            mCosPos.iterator.toint() + (bc64 * cutlass.Int64(mCosPos.stride[0]) + pos64 * cutlass.Int64(mCosPos.stride[1])) * bpe + lane_off
                        )
                        s_addr = (
                            mSinPos.iterator.toint() + (bc64 * cutlass.Int64(mSinPos.stride[0]) + pos64 * cutlass.Int64(mSinPos.stride[1])) * bpe + lane_off
                        )
                        c_pairs = [f16x2_to_f32(wd, dtype=mCosPos.element_type) for wd in ld_global_v4(c_addr, cutlass.Int32)]
                        s_pairs = [f16x2_to_f32(wd, dtype=mSinPos.element_type) for wd in ld_global_v4(s_addr, cutlass.Int32)]
                        cos_v = [v for pair in c_pairs for v in pair]
                        sin_v = [v for pair in s_pairs for v in pair]
                    for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                        rotated = rope_rotate_half(ys[0][i], cos_v[i], sin_v[i], lane, half_lanes)
                        ys[0][i] = rotated if in_rope else ys[0][i]
                if write_blk:
                    for c in cutlass.range_constexpr(chunks_i):
                        y = ys[c]
                        if cutlass.const_expr(comp_fp8):
                            _store_row_chunk_e4m3(out_row, c, lanes_i, lane, y, mIKsrc.element_type)
                        else:
                            coff = cutlass.Int64((c * lanes_i) * ACCESS_BYTES) + lane_off
                            packed = [fp32_to_fp16(y[i], y[i + 1], dtype=mComp.element_type) for i in range(0, ELEMS_PER_ACCESS, 2)]
                            st_global_v4(out_row + coff, packed, cutlass.Int32)
                slot_val = cutlass.Int32(SLOT_CAPACITY) if (committed & (completes | past_table)) else cutlass.Int32(SLOT_NO_BLOCK)
                slot_val = blk if write_blk else slot_val
                if in_row & (lane == cutlass.Int32(0)):
                    slot_addr = mCSlot.iterator.toint() + (
                        b64 * cutlass.Int64(mCSlot.stride[0]) + jr.to(cutlass.Int64) * cutlass.Int64(mCSlot.stride[1])
                    ) * cutlass.Int64(4)
                    st_global(slot_addr, slot_val, cutlass.Int32)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def qsa_prepare_launch(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    q_out: cute.Tensor,
    k_out: cute.Tensor,
    w_q: Optional[cute.Tensor],
    w_k: Optional[cute.Tensor],
    cos: Optional[cute.Tensor],
    sin: Optional[cute.Tensor],
    k_pool: cute.Tensor,
    v_pool: cute.Tensor,
    slot: cute.Tensor,
    ik_src: Optional[cute.Tensor],
    ik_pool: Optional[cute.Tensor],
    iq_src: Optional[cute.Tensor],
    iq_out: Optional[cute.Tensor],
    w_iq: Optional[cute.Tensor],
    ring: Optional[cute.Tensor],
    comp: Optional[cute.Tensor],
    w_ik: Optional[cute.Tensor],
    cos_pos: Optional[cute.Tensor],
    sin_pos: Optional[cute.Tensor],
    pos0: Optional[cute.Tensor],
    n_commit: Optional[cute.Tensor],
    cslot: Optional[cute.Tensor],
    kv_lens: Optional[cute.Tensor],
    block_table: Optional[cute.Tensor],
    state_table: Optional[cute.Tensor],
    ring_slots: Optional[cute.Tensor],
    n_tok: cutlass.Int32,
    s_q: cutlass.Int32,
    n_slots: cutlass.Int64,
    n_islots: cutlass.Int64,
    h_q: cutlass.Int32,
    h_kv: cutlass.Int32,
    h_i: cutlass.Int32,
    eps: cutlass.Float32,
    eps_i: cutlass.Float32,
    w_offset_i: cutlass.Float32,
    n_blocks: cutlass.Int32,
    d: cutlass.Constexpr[int],
    d_i: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    page_size: cutlass.Constexpr[int],
    slot_i64: cutlass.Constexpr[bool],
    threads_per_cta: cutlass.Constexpr[int],
    h_q_ct: cutlass.Constexpr[int],
    h_kv_ct: cutlass.Constexpr[int],
    h_i_ct: cutlass.Constexpr[int],
    const_head_counts: cutlass.Constexpr[bool],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qsa_prepare(
        q,
        k,
        v,
        q_out,
        k_out,
        w_q,
        w_k,
        cos,
        sin,
        k_pool,
        v_pool,
        slot,
        ik_src,
        ik_pool,
        iq_src,
        iq_out,
        w_iq,
        ring,
        comp,
        w_ik,
        cos_pos,
        sin_pos,
        pos0,
        n_commit,
        cslot,
        kv_lens,
        block_table,
        state_table,
        ring_slots,
        n_tok,
        s_q,
        n_slots,
        n_islots,
        h_q,
        h_kv,
        h_i,
        eps,
        eps_i,
        w_offset_i,
        d,
        d_i,
        rope_dim,
        pool,
        page_size,
        slot_i64,
        threads_per_cta,
        h_q_ct,
        h_kv_ct,
        h_i_ct,
        const_head_counts,
        use_pdl,
    ).launch(grid=(n_blocks, 1, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


compiled_cache = {}


# ``QsaPrepareIndexArms`` -- which indexer arms an artifact traces (every one a compile key) -- lives in ``cache_write_contract.py``: the
# block builds the record at DECLARATION, which must not import this module (the DSL at module scope; Rule 7).  Re-exported above.


class QsaPrepareRecipe(NamedTuple):
    """Build-time facts of one prepare launch -- every field derivable from the DECLARATION (dtypes, head counts, head dims,
    rope dim, page size, slot dtype, the traced arms), so it is a legal compile key.  The token count, the rows per
    sequence, the ring depth, the cache capacity and the pool capacities enter as ``cute.sym_int`` / runtime scalars: one
    artifact serves every step shape of its declaration."""

    compiled: object
    h_q: int
    h_kv: int
    d: int
    rope_dim: int
    eps: float
    page_size: int
    apply_norm: bool
    dtype: object
    slot_dtype: object
    index: Optional[QsaPrepareIndexArms]
    threads_per_cta: int
    index_cache_dtype: object = None  # appended: the indexer queries' / compressed-key cache's dtype -- None = the io dtype, float8_e4m3fn = the e4m3 store arm


def compile_qsa_prepare(
    *,
    dtype,
    h_q: int,
    h_kv: int,
    d: int,
    rope_dim: int,
    eps: float,
    page_size: int,
    slot_dtype=torch.int32,
    apply_norm: bool = True,
    index: Optional[QsaPrepareIndexArms] = None,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    const_head_counts: bool = True,
    use_pdl: bool = False,
    index_cache_dtype=None,
) -> QsaPrepareRecipe:
    """Build the artifact from SHAPES ALONE -- no device allocation, no launch (the block calls it from ``compile()``).

    ``apply_norm=False`` traces the RoPE-only Q / K artifact (both weight slots ``None`` -- the ``qk_norm_rope`` presence
    switch); ``index`` names the indexer arms to trace (``None`` = no indexer band: the attention rows and the K / V pools
    only); ``index_cache_dtype`` (appended) names the indexer queries' and the compressed-key cache's dtype -- ``None`` = the io
    dtype, ``torch.float8_e4m3fn`` = the e4m3 store arm of the query and the compress arms (module docstring; bf16 io only; sm_89+ --
    an older part is a typed ``NotImplementedError`` naming the instruction).  Every flag is in the cache key.
    """
    too_old = cutedsl_requirement_error("qsa_prepare")
    if too_old is not None:
        raise NotImplementedError(too_old)
    ix = index if (index is not None and index.any) else None
    validate_prepare_shape(d, rope_dim, page_size, threads_per_cta, None if ix is None else ix.head_dim, DEFAULT_POOL if ix is None else ix.pool)
    if dtype not in _IO_DTYPES:
        raise ValueError(f"qsa_prepare serves bf16 / f16 io only, got {dtype}")
    if slot_dtype not in _SLOT_DTYPES:
        raise ValueError(f"slot_mapping must be int32 or int64, got {slot_dtype}")
    if h_q < 1 or h_kv < 1:
        raise ValueError(f"h_q and h_kv must be >= 1, got {h_q} / {h_kv}")
    if not apply_norm and rope_dim == 0:
        raise ValueError("apply_norm=False with rope_dim=0 is an identity copy of Q / K; the cache write alone serves that step")
    if ix is not None and ix.heads < 1:
        raise ValueError(f"index.heads must be >= 1, got {ix.heads}")
    if ix is not None and ix.n_commit and not ix.compress:
        raise ValueError("index.n_commit traces the accepted-prefix load of the compress arm; it needs index.compress=True")
    if ix is not None and ix.paged and not ix.compress:
        raise ValueError("index.paged addresses the ring and the compressed-key cache through their block tables; it needs index.compress=True")
    if ix is not None and ix.ring_slots and not ix.paged:
        raise ValueError("index.ring_slots takes the ring write rows from the stack's slot mapping of the PAGED ring; it needs index.paged=True")
    if ix is not None and ix.rope_table and not ix.compress:
        raise ValueError(
            "index.rope_table reads the per-token rotations from the compress arm's position-indexed cos_pos / sin_pos; it needs index.compress=True"
        )
    if ix is not None and ix.rope_table and rope_dim == 0:
        raise ValueError("index.rope_table with rope_dim=0: there is no rotation to read from a table")
    check_index_cache_dtype(index_cache_dtype, dtype)
    cache_dtype = dtype if index_cache_dtype is None else index_cache_dtype
    if ix is not None and (ix.query or ix.compress):
        check_e4m3_store_target("qsa_prepare", cache_dtype)
    key = (
        str(dtype),
        int(h_q),
        int(h_kv),
        int(d),
        int(rope_dim),
        int(page_size),
        str(slot_dtype),
        bool(apply_norm),
        None if ix is None else tuple(ix),
        int(threads_per_cta),
        bool(const_head_counts),
        bool(use_pdl),
        current_device(),
        str(cache_dtype) if cache_dtype != dtype else None,  # appended: the e4m3 store arm is its own artifact
    )
    if key not in compiled_cache:
        tok = cute.sym_int()
        table_dim = rope_dim if rope_dim else 1
        bands = [fake_rowmajor_dynamic_token_stride(dtype, tok, h, d) for h in (h_q, h_kv, h_kv, h_q, h_kv)]  # q, k, v, q_out, k_out
        weights = [_fake(dtype, (d,), (0,)) for _ in range(2)] if apply_norm else [None, None]
        tables = [None, None] if (ix is not None and ix.rope_table) else [_fake(dtype, (tok, table_dim), (1, 0)) for _ in range(2)]
        pools = [_fake_pool(dtype, h_kv, page_size, d) for _ in range(2)]
        slot = _fake_slot(slot_dtype)
        ik_src = ik_pool = iq_src = iq_out = w_iq = ring = comp = w_ik = cos_pos = sin_pos = pos0 = n_commit = cslot = None
        kv_lens = block_table = state_table = ring_slots = None
        h_i = 0
        d_i = 8  # a legal width for the folded-out arms' constexpr arithmetic; never addressed
        pool = DEFAULT_POOL
        if ix is not None:
            h_i, d_i, pool = int(ix.heads), int(ix.head_dim), int(ix.pool)
            if ix.raw:
                ik_src = fake_rowmajor_dynamic_token_stride(dtype, tok, 1, d_i)
            if ix.raw_pool:
                ik_pool = _fake_pool(dtype, 1, page_size, d_i)
            if ix.query:
                iq_src = fake_rowmajor_dynamic_token_stride(dtype, tok, h_i, d_i)
                iq_out = _fake(cache_dtype, (tok, h_i, d_i), (2, 1, 0))
                w_iq = _fake(dtype, (d_i,), (0,))
            if ix.compress:
                batch = cute.sym_int()
                if ix.paged:
                    ring = _fake_paged_state(dtype, d_i)
                    comp = _fake_paged_state(cache_dtype, d_i)
                    kv_lens = _fake_compact(torch.int32, (batch,), (0,))
                    block_table = _fake_i32_table((batch, cute.sym_int()), (cute.sym_int(), 1))
                    state_table = _fake_i32_table((batch, 1), (cute.sym_int(), 1))
                    ring_slots = _fake_slot(slot_dtype) if ix.ring_slots else None
                else:
                    ring = _fake_compact(dtype, (batch, cute.sym_int(), d_i), (2, 1, 0))
                    comp = _fake_compact(cache_dtype, (batch, cute.sym_int(), d_i), (2, 1, 0))
                    pos0 = _fake_compact(torch.int32, (batch,), (0,))
                w_ik = _fake(dtype, (d_i,), (0,))
                cos_pos, sin_pos = [_fake_strided(dtype, (cute.sym_int(), cute.sym_int(), table_dim), (cute.sym_int(), cute.sym_int(), 1)) for _ in range(2)]
                n_commit = _fake_compact(torch.int32, (batch,), (0,)) if ix.n_commit else None
                cslot = _fake_compact(torch.int32, (batch, cute.sym_int()), (1, 0))
        compiled_cache[key] = cute.compile(
            qsa_prepare_launch,
            *bands,
            *weights,
            *tables,
            *pools,
            slot,
            ik_src,
            ik_pool,
            iq_src,
            iq_out,
            w_iq,
            ring,
            comp,
            w_ik,
            cos_pos,
            sin_pos,
            pos0,
            n_commit,
            cslot,
            kv_lens,
            block_table,
            state_table,
            ring_slots,
            cutlass.Int32(0),  # n_tok      ) runtime values: the literals pin
            cutlass.Int32(1),  # s_q        ) their TYPE at trace time only
            cutlass.Int64(0),  # n_slots    )
            cutlass.Int64(0),  # n_islots   )
            cutlass.Int32(h_q),
            cutlass.Int32(h_kv),
            cutlass.Int32(h_i),
            cutlass.Float32(eps),
            cutlass.Float32(1e-6),  # eps_i
            cutlass.Float32(0.0),  # w_offset_i
            cutlass.Int32(0),  # n_blocks
            int(d),
            int(d_i),
            int(rope_dim),
            int(pool),
            int(page_size),
            slot_dtype == torch.int64,
            int(threads_per_cta),
            int(h_q),
            int(h_kv),
            int(h_i),
            bool(const_head_counts),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QsaPrepareRecipe(
        compiled=compiled_cache[key],
        h_q=int(h_q),
        h_kv=int(h_kv),
        d=int(d),
        rope_dim=int(rope_dim),
        eps=float(eps),
        page_size=int(page_size),
        apply_norm=bool(apply_norm),
        dtype=dtype,
        slot_dtype=slot_dtype,
        index=ix,
        threads_per_cta=int(threads_per_cta),
        index_cache_dtype=cache_dtype,
    )


def _check_band(name: str, x: torch.Tensor, *, t: int, h: int, d: int, dtype) -> None:
    """A ``[T, H, D]`` band view: heads contiguous within a token (strides ``(*, D, 1)``), 16-B aligned rows, the recipe's dtype."""
    if not isinstance(x, torch.Tensor) or x.dim() != 3 or tuple(int(s) for s in x.shape) != (t, h, d):
        raise ValueError(f"{name} must be a [T={t}, H={h}, D={d}] tensor, got {tuple(x.shape) if isinstance(x, torch.Tensor) else type(x).__name__}")
    if int(x.stride(2)) != 1 or int(x.stride(1)) != d:
        raise ValueError(f"{name} must have contiguous heads (strides (*, {d}, 1)), got {tuple(x.stride())}")
    _check_row_tensor(name, x, dtype=dtype, last_dim=d)


def _given(**kw) -> list:
    return [nm for nm, x in kw.items() if x is not None]


def run_qsa_prepare(
    r: QsaPrepareRecipe,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_out: Optional[torch.Tensor],
    k_out: Optional[torch.Tensor],
    w_q: Optional[torch.Tensor],
    w_k: Optional[torch.Tensor],
    cos: Optional[torch.Tensor],
    sin: Optional[torch.Tensor],
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    index_k_raw_src: Optional[torch.Tensor] = None,
    index_k_raw_pool: Optional[torch.Tensor] = None,
    index_q_src: Optional[torch.Tensor] = None,
    index_q_out: Optional[torch.Tensor] = None,
    w_iq_norm: Optional[torch.Tensor] = None,
    ring: Optional[torch.Tensor] = None,
    index_k_compressed: Optional[torch.Tensor] = None,
    w_ik_norm: Optional[torch.Tensor] = None,
    cos_pos: Optional[torch.Tensor] = None,
    sin_pos: Optional[torch.Tensor] = None,
    pos0: Optional[torch.Tensor] = None,
    n_commit: Optional[torch.Tensor] = None,
    compressed_slot: Optional[torch.Tensor] = None,
    seq_q: Optional[int] = None,
    index_eps: float = 1e-6,
    index_norm_weight_offset: float = 0.0,
    stream,
    kv_lens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    index_state_block_table: Optional[torch.Tensor] = None,
    index_state_slot_mapping: Optional[torch.Tensor] = None,
) -> None:
    """The lowered launch: contract checks and address arithmetic only -- no conversion, no allocation, no device read.

    ``q`` / ``k`` / ``v`` are ``[T, H, D]`` band views (any token stride); ``q_out`` / ``k_out`` default to IN PLACE;
    ``w_q`` / ``w_k`` both ``None`` iff the recipe is RoPE-only; ``cos`` / ``sin`` ``[T, rope_dim]`` per-token rows;
    ``k_cache`` / ``v_cache`` ``[P, H_kv, page_size, D]`` pools; ``slot_mapping`` ``[T]``.  The indexer arguments are
    REQUIRED iff the recipe traced their arm and REFUSED otherwise (both directions typed): ``index_k_raw_src`` ``[T, 1, D_i]``
    whenever a raw-key arm exists, ``index_k_raw_pool`` ``[P, page_size, D_i]`` with the raw-pool arm, ``index_q_src`` /
    ``index_q_out`` ``[T, H_i, D_i]`` + ``w_iq_norm`` with the query arm, ``ring`` ``[B, RING, D_i]`` / ``index_k_compressed``
    ``[B, NB, D_i]`` / ``w_ik_norm`` / ``cos_pos`` / ``sin_pos`` ``[1 or B, S_pos, rope_dim]`` / ``pos0`` ``[B]`` /
    ``compressed_slot`` ``[B, S_q]`` + ``seq_q`` (= S_q, so B = T / S_q) with the compress arm, ``n_commit`` ``[B]`` iff traced.
    ``index_q_out`` and ``index_k_compressed`` are in the recipe's ``index_cache_dtype`` (the io dtype, or e4m3 under the e4m3 store
    arm -- refused both ways); the ring is always the io dtype.

    The PAGED compress arm (``index.paged``) binds the stack's state in the SAME slots -- ``ring`` is ``[blocks >= 1, cap >= S_q + pool - 1,
    1, D_i]``, ``index_k_compressed`` ``[num_pages >= 1, P / pool, 1, D_i]`` -- plus the appended ``kv_lens`` ``[B]`` int32 (the KV length
    AFTER the step), ``block_table`` ``[B, max_pages]`` int32 and ``index_state_block_table`` ``[B, 1]`` int32, and ``index_state_slot_mapping``
    ``[T]`` of the artifact's slot dtype iff ``index.ring_slots``; ``pos0`` is REFUSED there (the positions derive from ``kv_lens``), and a
    dense recipe REFUSES every paged keyword.  The RoPE-table arm (``index.rope_table``) REFUSES ``cos`` / ``sin`` (pass ``None``): the
    per-token rotations read ``cos_pos`` / ``sin_pos`` at ``pos0 + j``, and the table must hold at least as many rows as the compress
    arm's cache can address (paged: ``block_table.shape[1] x P``; dense: ``NB x pool``) -- refused by form otherwise.  The LIVE
    positions (``pos0`` / ``kv_lens`` are device data: no host bound without a sync) are checked IN-KERNEL against the table: a row at
    ``pos0 + j >= S_pos`` is reported ``-2`` in ``compressed_slot`` and none of its rotated outputs (its Q / K rows, its K pool row, its
    indexer queries), its ring row or its block is written -- check ``compressed_slot`` for ``-2`` before consuming a step's outputs
    (the existing capacity report; no silently clamped rotation).
    """
    t = int(q.shape[0])
    if t < 1:
        raise ValueError("q must hold at least one token")
    _check_band("q", q, t=t, h=r.h_q, d=r.d, dtype=r.dtype)
    _check_band("k", k, t=t, h=r.h_kv, d=r.d, dtype=r.dtype)
    _check_band("v", v, t=t, h=r.h_kv, d=r.d, dtype=r.dtype)
    q_out = q if q_out is None else q_out
    k_out = k if k_out is None else k_out
    _check_band("q_out", q_out, t=t, h=r.h_q, d=r.d, dtype=r.dtype)
    _check_band("k_out", k_out, t=t, h=r.h_kv, d=r.d, dtype=r.dtype)
    check_norm_weights_match_recipe(r.apply_norm, w_q, w_k)
    if r.apply_norm:
        for nm, w in (("w_q", w_q), ("w_k", w_k)):
            if w.dim() != 1 or int(w.shape[0]) != r.d or not w.is_contiguous():
                raise ValueError(f"{nm} must be a contiguous [{r.d}] norm weight, got shape {tuple(w.shape)}")
            _check_row_tensor(nm, w, dtype=r.dtype, last_dim=r.d)
    ix = r.index
    table_arm = ix is not None and ix.rope_table
    if r.rope_dim:
        if table_arm:
            if cos is not None or sin is not None:
                raise ValueError(
                    "cos / sin are refused on a recipe that traced the RoPE-table arm (index.rope_table=True): the per-token rotations read cos_pos / sin_pos "
                    "at pos0 + j -- pass cos=None, sin=None"
                )
        else:
            if cos is None or sin is None:
                raise ValueError(f"cos / sin [T={t}, {r.rope_dim}] per-token tables are required (this recipe traced no RoPE-table arm)")
            for nm, tb in (("cos", cos), ("sin", sin)):
                if tb.dim() != 2 or int(tb.shape[0]) != t or not tb.is_contiguous():
                    raise ValueError(
                        f"{nm} must be a contiguous [T={t}, {r.rope_dim}] per-token table, got shape {tuple(tb.shape)} strides {tuple(tb.stride())}"
                    )
                _check_row_tensor(nm, tb, dtype=r.dtype, last_dim=r.rope_dim)
    check_pool(k_cache, "k_cache", h=r.h_kv, page_size=r.page_size, d=r.d, dtype=r.dtype)
    check_pool(v_cache, "v_cache", h=r.h_kv, page_size=r.page_size, d=r.d, dtype=r.dtype)
    check_slot_mapping(slot_mapping, "slot_mapping", t=t)
    if slot_mapping.dtype != r.slot_dtype:
        raise ValueError(f"slot_mapping is {slot_mapping.dtype} but this artifact was compiled for {r.slot_dtype}; the slot dtype is fixed per artifact")
    n_slots = min(int(k_cache.shape[0]), int(v_cache.shape[0])) * r.page_size

    paged_names = ("pos0", "kv_lens", "block_table", "index_state_block_table", "index_state_slot_mapping")
    given = _given(
        index_k_raw_src=index_k_raw_src,
        index_k_raw_pool=index_k_raw_pool,
        index_q_src=index_q_src,
        index_q_out=index_q_out,
        w_iq_norm=w_iq_norm,
        ring=ring,
        index_k_compressed=index_k_compressed,
        w_ik_norm=w_ik_norm,
        cos_pos=cos_pos,
        sin_pos=sin_pos,
        pos0=pos0,
        n_commit=n_commit,
        compressed_slot=compressed_slot,
        kv_lens=kv_lens,
        block_table=block_table,
        index_state_block_table=index_state_block_table,
        index_state_slot_mapping=index_state_slot_mapping,
    )
    want = []
    if ix is not None:
        if ix.raw:
            want.append("index_k_raw_src")
        if ix.raw_pool:
            want.append("index_k_raw_pool")
        if ix.query:
            want += ["index_q_src", "index_q_out", "w_iq_norm"]
        if ix.compress:
            want += ["ring", "index_k_compressed", "w_ik_norm", "compressed_slot"] + (["cos_pos", "sin_pos"] if r.rope_dim else [])
            if ix.paged:
                want += ["kv_lens", "block_table", "index_state_block_table"] + (["index_state_slot_mapping"] if ix.ring_slots else [])
            else:
                want.append("pos0")
            if ix.n_commit:
                want.append("n_commit")
    judged_by_shared_check = (
        paged_names if (ix is not None and ix.compress) else ()
    )  # the position source and the tables: check_paged_step_operands, both directions
    missing = [nm for nm in want if nm not in given and nm not in judged_by_shared_check]
    extra = [nm for nm in given if nm not in want and nm not in judged_by_shared_check]
    if missing:
        raise ValueError(f"this artifact traced the indexer arm(s) {tuple(ix)}: {', '.join(missing)} must be bound at execute (no silent fallback)")
    if extra:
        raise ValueError(
            f"{', '.join(extra)}: this artifact traced "
            + ("no indexer arm" if ix is None else f"the indexer arms {tuple(ix)}")
            + " and would silently ignore them -- compile the matching recipe, or pass None"
        )
    n_islots = 0
    s_q = 1
    if ix is not None:
        d_i = ix.head_dim
        cache_dtype = r.index_cache_dtype if r.index_cache_dtype is not None else r.dtype  # the recipe's: the io dtype or the e4m3 store arm, both ways
        if ix.raw:
            _check_band("index_k_raw_src", index_k_raw_src, t=t, h=1, d=d_i, dtype=r.dtype)
        if ix.raw_pool:
            if index_k_raw_pool.dim() != 3:
                raise ValueError(f"index_k_raw_pool must be a 3-D [num_pages, {r.page_size}, {d_i}] pool, got shape {tuple(index_k_raw_pool.shape)}")
            check_pool(index_k_raw_pool.unsqueeze(1), "index_k_raw_pool", h=1, page_size=r.page_size, d=d_i, dtype=r.dtype)
            n_islots = int(index_k_raw_pool.shape[0]) * r.page_size
        if ix.query:
            _check_band("index_q_src", index_q_src, t=t, h=ix.heads, d=d_i, dtype=r.dtype)
            _check_band("index_q_out", index_q_out, t=t, h=ix.heads, d=d_i, dtype=cache_dtype)
            if not index_q_out.is_contiguous():
                raise ValueError(f"index_q_out must be a compact [T, {ix.heads}, {d_i}] buffer, got strides {tuple(index_q_out.stride())}")
            if w_iq_norm.dim() != 1 or int(w_iq_norm.shape[0]) != d_i or not w_iq_norm.is_contiguous():
                raise ValueError(f"w_iq_norm must be a contiguous [{d_i}] norm weight, got shape {tuple(w_iq_norm.shape)}")
            _check_row_tensor("w_iq_norm", w_iq_norm, dtype=r.dtype, last_dim=d_i)
        if ix.compress:
            if seq_q is None or int(seq_q) < 1 or t % int(seq_q):
                raise ValueError(f"the compress arm needs seq_q (the rows per sequence, S_q >= 1, dividing T={t}); got {seq_q}")
            s_q = int(seq_q)
            batch = t // s_q
            need = min_ring_rows(s_q, ix.pool)
            if ix.paged:
                check_paged_state(
                    "ring",
                    ring,
                    dtype=r.dtype,
                    d=d_i,
                    min_rows=need,
                    what=f"raw-key ring: a step of {s_q} rows needs rows + pool - 1 = {s_q} + {ix.pool} - 1 so this launch's ring writes never alias its ring reads",
                )
                check_paged_state(
                    "index_k_compressed",
                    index_k_compressed,
                    dtype=cache_dtype,
                    d=d_i,
                    min_rows=1,
                    what="compressed-key cache, P / pool rows per attention page",
                    cache=True,
                )
            else:
                if ring.dim() != 3 or int(ring.shape[0]) != batch or int(ring.shape[2]) != d_i or not ring.is_contiguous():
                    raise ValueError(f"ring must be a compact [B={batch}, RING, D_i={d_i}] tensor, got shape {tuple(ring.shape)}")
                if int(ring.shape[1]) < need:
                    raise ValueError(
                        f"ring holds {int(ring.shape[1])} rows per sequence; a step of {s_q} rows needs at least {need} (rows + pool - 1 = {s_q} + {ix.pool} - 1) "
                        "so this launch's ring writes never alias its ring reads and no committed key a later step pools is overwritten by a draft"
                    )
                _check_row_tensor("ring", ring, dtype=r.dtype, last_dim=d_i)
                c = index_k_compressed
                if c.dim() != 3 or int(c.shape[0]) != batch or int(c.shape[2]) != d_i or int(c.shape[1]) < 1 or not c.is_contiguous():
                    raise ValueError(f"index_k_compressed must be a compact [B={batch}, NB >= 1, D_i={d_i}] cache, got shape {tuple(c.shape)}")
                _check_row_tensor("index_k_compressed", c, dtype=cache_dtype, last_dim=d_i)
            if w_ik_norm.dim() != 1 or int(w_ik_norm.shape[0]) != d_i or not w_ik_norm.is_contiguous():
                raise ValueError(f"w_ik_norm must be a contiguous [{d_i}] norm weight, got shape {tuple(w_ik_norm.shape)}")
            _check_row_tensor("w_ik_norm", w_ik_norm, dtype=r.dtype, last_dim=d_i)
            if r.rope_dim:
                for nm, tb in (("cos_pos", cos_pos), ("sin_pos", sin_pos)):
                    if tb.dim() != 3 or int(tb.shape[0]) not in (1, batch) or int(tb.shape[1]) < 1:
                        raise ValueError(f"{nm} must be [1 or B={batch}, S_pos >= 1, {r.rope_dim}] (position-indexed), got shape {tuple(tb.shape)}")
                    _check_row_tensor(nm, tb, dtype=r.dtype, last_dim=r.rope_dim)
                if tuple(cos_pos.shape) != tuple(sin_pos.shape):
                    raise ValueError(f"cos_pos and sin_pos must share a shape, got {tuple(cos_pos.shape)} and {tuple(sin_pos.shape)}")
            check_paged_step_operands(
                paged=ix.paged,
                ring_slots=ix.ring_slots,
                ring_slot_dtype=r.slot_dtype,
                batch=batch,
                rows=s_q,
                pos0=pos0,
                kv_lens=kv_lens,
                block_table=block_table,
                index_state_block_table=index_state_block_table,
                index_state_slot_mapping=index_state_slot_mapping,
                ring=ring,
            )
            if table_arm and r.rope_dim:
                # the RoPE-table arm rotates EVERY live row at its position: the table must cover every position the compress arm's
                # cache can address, or a live row past the table would be rotated with a clamped row, silently (by form: shapes only)
                if ix.paged:
                    n_pos = int(block_table.shape[1]) * int(index_k_compressed.shape[1]) * ix.pool
                    where = f"the main block table addresses {int(block_table.shape[1])} pages x {int(index_k_compressed.shape[1]) * ix.pool} tokens"
                else:
                    n_pos = int(index_k_compressed.shape[1]) * ix.pool
                    where = f"the compressed-key cache holds {int(index_k_compressed.shape[1])} blocks x {ix.pool} tokens"
                if int(cos_pos.shape[1]) < n_pos:
                    raise ValueError(
                        f"cos_pos / sin_pos hold {int(cos_pos.shape[1])} positions, but the RoPE-table arm (index.rope_table=True) rotates every row of "
                        f"the step at its position and {where}: hand a table of at least {n_pos} rows (the model's position table) so no live row is "
                        "rotated with a clamped table row"
                    )
            if not ix.paged:
                _check_int32_vector("pos0", pos0, (batch,))
            _check_int32_vector("compressed_slot", compressed_slot, (batch, s_q))
            if ix.n_commit:
                _check_int32_vector("n_commit", n_commit, (batch,))
    attn_w, qi_w, raw_w = prepare_warp_counts(
        t,
        h_q=r.h_q,
        h_kv=r.h_kv,
        d=r.d,
        index_heads=0 if ix is None else ix.heads,
        index_head_dim=0 if ix is None else ix.head_dim,
        index_q=ix is not None and ix.query,
        index_raw=ix is not None and ix.raw,
    )
    warps_per_cta = r.threads_per_cta // _WARP
    n_blocks = _ceil_div(attn_w + qi_w + raw_w, warps_per_cta)
    if r.rope_dim == 0:
        cos_pos = sin_pos = None
    r.compiled(
        q,
        k,
        v,
        q_out,
        k_out,
        w_q,
        w_k,
        cos,
        sin,
        k_cache,
        v_cache,
        slot_mapping,
        index_k_raw_src,
        None if index_k_raw_pool is None else index_k_raw_pool.unsqueeze(1),
        index_q_src,
        index_q_out,
        w_iq_norm,
        ring,
        index_k_compressed,
        w_ik_norm,
        cos_pos,
        sin_pos,
        pos0,
        n_commit,
        compressed_slot,
        kv_lens,
        block_table,
        index_state_block_table,
        index_state_slot_mapping,
        cutlass.Int32(t),
        cutlass.Int32(s_q),
        cutlass.Int64(n_slots),
        cutlass.Int64(n_islots),
        cutlass.Int32(r.h_q),
        cutlass.Int32(r.h_kv),
        cutlass.Int32(0 if ix is None else ix.heads),
        cutlass.Float32(r.eps),
        cutlass.Float32(index_eps),
        cutlass.Float32(index_norm_weight_offset),
        cutlass.Int32(n_blocks),
        cuda.CUstream(int(stream)),
    )


def moved_bytes(
    t: int,
    h_q: int,
    h_kv: int,
    d: int,
    *,
    elem_bytes: int = 2,
    slot_bytes: int = 4,
    index: Optional[QsaPrepareIndexArms] = None,
    pool: int = DEFAULT_POOL,
    index_cache_bytes: Optional[int] = None,
) -> int:
    """HBM traffic of one launch -- the denominator for an SOL number.  Q read + written; K read + written twice (the slab
    and the pool); V read + written (the pool); the slot per token; with the band: the raw key read once per job (pool,
    ring, its own block -- ``pool`` rows per completing row, counted as one row per token on average) and written to the
    pool and the ring, the indexer queries read + written.  The tables and the norm weights are L2-resident and excluded,
    as in the sibling kernels' counts.  ``index_cache_bytes`` (appended; ``None`` = ``elem_bytes``) is the element size of the
    indexer queries' output and the compressed-key cache (1 under the e4m3 store arm): the rows the launch WRITES there."""
    cache_bytes = elem_bytes if index_cache_bytes is None else int(index_cache_bytes)
    total = (2 * t * h_q + 3 * t * h_kv + 2 * t * h_kv) * d * elem_bytes + t * slot_bytes
    if index is not None and index.any:
        d_i = index.head_dim
        if index.raw_pool:
            total += 2 * t * d_i * elem_bytes
        if index.compress:
            total += 2 * t * d_i * elem_bytes + t * d_i * cache_bytes  # the ring write (read + write) and one pooled row per token on average
        if index.query:
            total += t * index.heads * d_i * (elem_bytes + cache_bytes)  # read in the io dtype, written in the cache dtype
    return total


frost_qsa_prepare.set_name_prefix("cudnn", remove_cutlass_symbol=True)
