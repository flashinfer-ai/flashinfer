# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block compression for Qwen Sparse Attention (QSA): the indexer's key side.

QSA scores every COMPLETE block of ``POOL`` (= 4) consecutive tokens with one
compressed key ``kbar_b``.  This kernel produces those keys for a batch of
sequences in one launch (the prefill form)::

    pooled  = mean_fp32(k_raw[4b .. 4b+3, :])                          # the raw [S, 128] indexer keys
    y       = pooled * rsqrt(mean(pooled^2) + eps) * (w + w_offset)     # RMSNorm over the 128 dims
    y[:64]  = y[:64] * cos[4b] + rotate_half(y[:64]) * sin[4b]          # partial RoPE at the block START
    kbar_b  = io_dtype(y)                                               # ONE rounding

for every ``b < n_blocks_s = floor(min(seq_len_s, S) / POOL)`` of every sequence
``s``.  The HF reference pools in fp32, casts the pooled key to the key dtype,
norms with the zero-centred weight ``(1 + w)`` and rotates at the group's first
position; this kernel keeps fp32 through the whole chain and rounds ONCE at the
store -- the single-rounding contract of the block's own norm+RoPE kernel
(``qk_norm_rope.py``), whose per-row body this is with a ``POOL``-row mean
prepended.  The norm weight comes in two forms:

* ``w_offset = 0.0`` -- the weight vector already holds ``(1 + w)`` (a loader
  that folds the ``1`` in at load time; the 16-bit rounding of that factor is
  then the loader's);
* ``w_offset = 1.0`` -- ``w`` as the checkpoint stores it, the ``1`` added in
  fp32 in-kernel (HF-faithful).

Lane mapping (``D = 128``, 16-bit io): a row is 256 B and every lane moves 16 B,
so ``LANES = D // 8 = 16`` lanes cover one block -- two blocks per warp, eight per
128-thread CTA.  Per lane: ``POOL`` x ``ld.global.v4`` (one per pooled row), the
fp32 sum, the sum of squares reduced over the 16 lanes by a butterfly
(``tile_dsl.pointwise.rmsnorm_rstd``), one 16-B weight load, ``cos`` / ``sin``
loads on the eight rope lanes, one ``shfl.bfly`` per element for the
rotate-half partner (``tile_dsl.pointwise.rope_rotate_half``), one
``st.global.v4``.  Wider ``D`` falls back to several 16-B chunks per lane exactly
as the norm kernel does (``vec_chunks``).

**No mbarriers, no shared memory, no TMA, no tcgen05** -- plain LDG/STG plus
warp shuffles, so the barrier table and the SMEM table of this kernel are EMPTY
by construction; the shuffles are the only cross-lane dependency, and every
lane of a warp reaches every shuffle (a group without a block to write computes
on clamped loads and skips the store).  It runs on any CuTe DSL device.

Degenerate inputs, each with its handling:

* a sequence with fewer than ``POOL`` valid tokens (``seq_len_s < POOL``,
  0 included) -> ``n_blocks_s = 0``, nothing is written for it
  (``valid = blk < n_blocks_s``);
* an incomplete trailing group (``seq_len_s % POOL != 0``) -> never read,
  never written;
* ``S < POOL`` (no complete block anywhere) -> the host launches nothing;
* a ragged last CTA -> its groups past ``floor(S / POOL)`` clamp their loads to
  the last complete block of the batch entry (always inside the allocation) and
  skip the store;
* ``seq_lens`` entries above ``S`` or below 0 are clamped on device;
* ``out`` rows at or beyond ``floor(S / POOL)`` (a cache allocated for a longer
  context) are never written.

Bytes per launch: ``B x floor(S / POOL) x POOL x D x 2`` in and
``B x floor(S / POOL) x D x 2`` out -- 10 MiB per layer at S = 32768.  The
roofline term of this kernel is HBM bytes; no perf claim is made here until
one is measured on an exclusive GPU.

The DECODE form -- ``frost_qsa_compress_step`` and the serving contract
-------------------------------------------------------------------------

A decode step appends ``S_q`` tokens per sequence (one, or one plus a few
speculative drafts).  A block completes when its 4th token arrives, so the
step must compress the groups its new tokens complete, from raw keys that
partly belong to EARLIER steps.  The compressor state is a per-sequence
RING of raw keys, ``raw_key_ring [B, RING, D]`` in the io dtype:

* the ring is POSITION-indexed: the raw key of token position ``p`` lives in
  row ``p % RING`` (a bit copy of the io-dtype row; the ring never holds a
  compressed or converted value);
* ``pos0 [B]`` int32 is the position of the step's first row (``= kv_len``
  BEFORE the step; row ``j`` sits at ``pos0 + j``); ``n_commit [B]`` int32
  (optional; ``None`` = every row) is the ACCEPTED PREFIX of the step's rows:
  rows ``j < n_commit[b]`` are committed, the others are drafts awaiting
  verification and leave no trace;
* for every committed row the kernel (1) writes its raw key into the ring at
  ``(pos0 + j) % RING`` and (2) if the row completes a block (``(pos0 + j) %
  POOL == POOL - 1``) pools the block's rows -- positions below ``pos0`` from
  the ring, positions at or above it from this step's ``k_raw`` -- through
  the SAME fp32 chain as the prefill kernel (same op order: the result is
  bitwise the prefill compress of the same tokens) and stores the compressed
  key at ``out[b, blk]``, ``blk = (pos0 + j) // POOL`` (the compressed-key
  cache is indexed by block id, so a recompute of a block lands on the same
  row);
* ``compressed_slot [B, S_q]`` int32 reports per row: the block id written,
  ``-1`` = this row completes no block (or is an uncommitted draft), ``-2`` =
  the row's position lies at or past the RoPE table (``pos0 + j >= cos.shape[1]``:
  the table is the CEILING on positions, checked on the live position of every
  committed row -- the row writes nothing, no ring row either; no block ever
  pools a ring row at or past the table, since a block's rows all lie at or
  below its completing row's position) or the row would complete a block the
  compressed-key cache cannot hold (``blk >= out.shape[1]``: the block is not
  written; the raw key still enters the ring) -- size both for the context;
* ``RING >= S_q + POOL - 1`` (``min_ring_rows``), validated on the host: a
  completing row of this step reads ring positions down to ``pos0 - POOL +
  1`` while the step's committed rows write positions ``pos0 .. pos0 + S_q -
  1``; with that many rows no write of this launch aliases a read of this
  launch, and no committed key a later step still needs is overwritten by a
  draft (the hazard a 4-row ring has).  ``RING = 8`` serves ``S_q <= 5``.

MTP rollback contract (the two-phase commit).  Draft tokens may be rejected,
and a compressed row or a ring row written for a rejected draft would
poison the sequence's state.  The kernel therefore commits ONLY the accepted
prefix, and a serving step calls it twice:

1. at the start of the step with ``n_commit = 1``: row 0 (the accepted token
   whose K is computed in this step) enters the ring and completes its block
   if it is a 4th token -- the block the step-0 selection must see;
2. after verification with ``n_commit = 1 + accepted``: the accepted drafts
   enter the ring and complete their blocks; the rejected ones never touch
   the ring or the cache.  Row 0 is re-committed by the second call -- a
   bit-identical rewrite of the same ring row and (if any) the same block
   row, so the two calls compose to one call with the final ``n_commit``.

Under this contract every ring row and every compressed row holds a value
computed from accepted tokens only, so no rollback is ever needed; a stack
that commits eagerly (all ``S_q`` rows at the step start) must instead
recompute the blocks touched by rejected rows at the next step (the same
kernel, since a block recompute lands on its own row) and keep the ring wide
enough that a rejected draft cannot overwrite a committed key the next step
pools -- ``RING >= S_q + POOL - 1`` covers that too.  ``norm_weight_offset``
is threaded identically to the prefill form (0.0 = the pre-folded ``(1 + w)``,
1.0 = the zero-centred checkpoint weight).

Degenerate rows of the decode form: ``n_commit = 0`` (a finished sequence in
the batch) -> no ring write, no block, every slot ``-1``; ``pos0 < 0`` is
clamped to 0; a ragged last CTA computes on clamped loads and stores nothing.

The PAGED decode form -- the serving stack's two side caches, taken as they are
--------------------------------------------------------------------------------

A serving stack keeps the compressor state PAGED, and the decode step must read and write it in place (a per-step gather of
a dense per-sequence copy would double the long-context HBM term).  The raw-key ring is ONE block of ``cap`` rows per
sequence inside ``index_state_cache [num_ring_blocks, cap, 1, D]`` behind ``index_state_block_table [B, 1]`` (vLLM's
``CircularBufferSpec``: ``cap`` 4 without speculative rows, 8 with 1-4; block 0 is the NULL block that padding requests sit
on); the compressed-key cache is ``index_kv_cache [num_pages, P / POOL, 1, D]`` on the MAIN block table
``block_table [B, max_pages]`` (vLLM's ``MLAAttentionSpec(tokens_per_state=POOL)``: ``P / POOL`` compressed rows per
attention page of ``P`` tokens).  ``frost_qsa_compress_step`` takes both AS THEY ARE (``paged=True``, a compile key) -- no
gather, no conversion -- through ONE shared addressing helper per table (:func:`ring_row_address`, :func:`paged_block_row`;
the fused prepare launch of ``qsa_prepare.py`` calls the same two) and derives every position in-kernel
(:func:`step_position`)::

    pos0[b]          = kv_lens[b] - S_q                             # kv_lens = the KV length AFTER the step
    pos              = pos0[b] + j                                  # row j of sequence b
    ring row of pos  = (index_state_block_table[b, 0], pos % cap)   # == the slot  block * cap + pos % cap
    kbar row of blk  = (block_table[b, blk // (P / POOL)], blk % (P / POOL))

The ring's two forms address the SAME row through one formula: the dense ``[B, RING, D]`` row ``(b, p % RING)`` and the paged
``[blocks, cap, 1, D]`` row ``(table[b, 0], p % cap)`` are both ``base + entry * stride[0] + (p % shape[1]) * stride[1]``;
:func:`ring_row_address` takes the entry (``b`` or the state block) and the position.  ``index_state_slot_mapping [T]``
(the stack's own ring slots, int32 or int64, ``-1`` = no write) is accepted as an OPTIONAL alternative source of the WRITE
row (``ring_slots=True``, a compile key; the read rows of a completing block always come from the table) -- the two forms
are pinned bitwise.

Rows and their handling in the paged form: a row at a NEGATIVE position (``kv_lens[b] < S_q``; a padding sequence has
``kv_lens 0`` and sits on the null block) is dead -- no ring row, no block, slot ``-1`` -- so the step derives "nothing to
commit" from ``kv_lens == 0`` itself and the null block and page 0 are never written for it; a sequence whose state-table
entry is ``<= 0`` or past the ring's blocks (``>= index_state_cache.shape[0]``) owns no ring (the stack's own mapping writes a
ring row only for ``physical_block > 0``; an entry past the ring reads as the null block, in-bounds): it writes no ring row,
and a block of its that needs ring rows -- a block starting below ``pos0`` -- reports ``-2`` and writes nothing, so the null
block's rows are never pooled into a live sequence's block (a block pooled from this step's rows alone is still written); a
completing row whose block lies past the table's width, on a negative (unassigned) page or on a page past the
cache reports ``-2`` and writes nothing (the capacity rule of the dense form); a committed row at or past the RoPE table reports ``-2``
and writes nothing, no ring row either (the table ceiling of the dense form); ``n_commit`` keeps its meaning.  The aliasing
argument of the dense form holds unchanged: a completing row of this step reads ring positions ``pos0 - POOL + 1 ..
pos0 - 1`` while the step writes positions ``pos0 .. pos0 + S_q - 1``, disjoint rows whenever ``cap >= S_q + POOL - 1``
(``min_ring_rows``; a cap of 4 serves ``S_q = 1``, 8 serves ``S_q <= 5``), so within a launch no row is read after another
row writes it.

The e4m3 STORE arm (the e4m3 compressed-key cache at decode, ``QsaSpec.index_cache_dtype=torch.float8_e4m3fn``; 2026-10-21): the
compressed-key output of EITHER form (the prefill kernel's ``out`` through ``compile_qsa_compress(out_dtype=)`` -- the unfused decode
step's pool-1 query path --, the step form's dense ``[B, NB, D]`` and paged ``[num_pages, P / POOL, 1, D]`` caches through
``compile_qsa_compress_step(index_cache_dtype=)``) may be ``float8_e4m3fn`` on bf16 io.  The chain is the bf16 form's TO THE LAST BIT
and then ONE more rounding::

    kbar_b = e4m3_rn_satfinite(f32(bf16_rn(y)))        # == bf16_row.to(torch.float8_e4m3fn): the e4m3 cast OF THE bf16 ROW

i.e. the e4m3 cache is exactly the e4m3 cast of the bf16 cache the bf16 form writes (``tile_dsl.pointwise.bf16_words_to_e4m3_words``:
the four packed bf16 words widened exactly and rounded by ``cvt.rn.satfinite.e4m3x2.f32`` -- round-to-nearest-even, saturating at
+-448, never a NaN; one ``st.global.v2.b32`` of 8 B per lane in place of the ``v4`` of 16).  No scale (plain e4m3 at unit scale: the
serving stacks' indexer-cache recipe), the same convention as the prefill form's per-tensor cast of its bf16 rows, so a paged cache
primed from the prefill form's rows and appended by the step form is ONE chain, bitwise.  The raw-key RING stays the io dtype (a bit copy
of the step's raw keys, never quantized -- an e4m3 ring is a form error naming that rule), so the bf16 value pooled for a block is bitwise
the bf16 form's before the cast.  f16 io has no e4m3 arm (the helper widens bf16 words; ``check_index_cache_dtype`` refuses the pair by
name).  The fp8 ``cvt`` is an sm_89+ instruction: an older part is declined BY NAME at compile time (Rule 7), never through ptxas.

Still no shared memory, no mbarrier, no TMA, no tcgen05 in either form: the barrier table and the SMEM table of this kernel
stay EMPTY by construction.
"""

from typing import NamedTuple, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes import _convert_to_cutlass_data_type
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_requirement_error
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.device import compute_capability, current_device
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.pointwise import bf16_words_to_e4m3_words, f16x2_to_f32, fp32_to_fp16, rmsnorm_rstd, rope_rotate_half
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.tma import ld_global, ld_global_v2, ld_global_v4, st_global, st_global_v2, st_global_v4

from ..cache_write_contract import DEFAULT_POOL  # tokens per compressed block (the QSA compress ratio): defined DSL-free, re-exported here
from .qk_norm_rope import ACCESS_BYTES, ELEMS_PER_ACCESS, lanes_per_row, validate_shape, vec_chunks

# ``DEFAULT_POOL`` -- tokens per compressed block, the QSA compress ratio (``indexer_compress_ratio``) -- is imported above from the DSL-free
# contract module (the block's declaration reads it through ``QsaPrepareIndexArms``; Rule 7) and keeps its name here for this kernel's callers.
DEFAULT_THREADS_PER_CTA = 128

_IO_DTYPES = (torch.bfloat16, torch.float16)
_FAKE_STREAM = make_fake_stream(use_tvm_ffi_env_stream=False)
_E4M3 = torch.float8_e4m3fn
_FP8_CVT_MIN_CC = (8, 9)  # cvt.rn.satfinite.e4m3x2.f32 (the e4m3 store arm): Ada / Hopper / Blackwell / Rubin
E4M3_LANE_BYTES = 8  # the e4m3 store arm's bytes per lane per chunk: the 8 elements of one 16-B bf16 access as e4m3, one st.global.v2.b32


def validate_compress_shape(d: int, rope_dim: int, pool: int, threads_per_cta: int) -> None:
    """Raise on any geometry this kernel cannot address.

    The row geometry (``d``, ``rope_dim``, the per-row lane count) is the norm
    kernel's and is checked by its ``validate_shape``; the pool size is this
    kernel's own.  Never an ``assert``: user-facing geometry must survive
    ``python -O`` and name itself.
    """
    validate_shape(d, rope_dim, threads_per_cta)
    if pool < 1 or (pool & (pool - 1)) != 0:
        raise ValueError(f"pool must be a power of two >= 1 so the fp32 mean is an exact reciprocal multiply, got {pool}")


@cute.jit
def _store_row_chunk_e4m3(out_base: cutlass.Int64, c: cutlass.Constexpr[int], lanes: cutlass.Constexpr[int], lane: cutlass.Int32, y, io_dtype):
    """The e4m3 store arm of one 16-B chunk of a compressed row: the io-dtype (bf16) rounding of the eight fp32 values -- the bf16 form's
    words, bit for bit -- then :func:`bf16_words_to_e4m3_words` (``cvt.rn.satfinite.e4m3x2.f32`` on the exactly widened bf16: the e4m3
    cast of the bf16 row) and ONE ``st.global.v2.b32`` of 8 B at ``out_base + (c x lanes + lane) x 8``: the e4m3 row is 128 B = 16 lanes x
    8 B.  Called by the prefill kernel, the step form and the fused prepare launch (one convention, four store sites)."""
    packed = [fp32_to_fp16(y[i], y[i + 1], dtype=io_dtype) for i in range(0, ELEMS_PER_ACCESS, 2)]
    u0, u1 = bf16_words_to_e4m3_words(packed[0], packed[1], packed[2], packed[3])
    off8 = cutlass.Int64((c * lanes) * E4M3_LANE_BYTES) + lane.to(cutlass.Int64) * cutlass.Int64(E4M3_LANE_BYTES)
    st_global_v2(out_base + off8, (u0, u1), cutlass.Int32)


@cute.kernel
def frost_qsa_compress(
    mK: cute.Tensor,  # [B, S, D] io dtype; batch / token strides symbolic (a column slice of the projection slab is fine)
    mOut: cute.Tensor,  # [B, NB, D] io dtype, compact; NB >= S // POOL
    mW: cute.Tensor,  # [D] io dtype
    mCos: cute.Tensor,  # [Bc, S_pos, ROPE_DIM] 16-bit, Bc in {1, B}: row POOL*b of entry b (entry 0 when Bc == 1)
    mSin: cute.Tensor,  # same shape and dtype as mCos
    mSeqLens: Optional[cute.Tensor],  # [B] int32 valid lengths, or None (= S for every batch entry)
    eps: cutlass.Float32,
    w_offset: cutlass.Float32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    """One lane group (``lanes_per_row(d)`` lanes) per compressed block.

    Grid ``(ceil(floor(S / POOL) / groups_per_cta), B)``: ``block_idx.y`` is the
    batch entry, ``block_idx.x * groups_per_cta + grp`` the block index within
    it.  A group whose block index is at or past the entry's valid block count
    runs the same arithmetic on CLAMPED loads (so every lane of the warp reaches
    the butterfly shuffles) and skips the store.
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    has_seq_lens = cutlass.const_expr(mSeqLens is not None)
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(vec_chunks(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)
    bpe = cutlass.Int64(2)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    b = cutlass.Int32(cute.arch.block_idx()[1])
    blk = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp

    # Block counts.  n_full = the complete blocks S holds (>= 1 whenever the host
    # launched); n_blocks_b = the entry's VALID blocks from its clamped length.
    s_tok = cutlass.Int32(mK.shape[1])
    n_full = s_tok // cutlass.Int32(pool)
    n_tok = s_tok
    if cutlass.const_expr(has_seq_lens):
        slen = ld_global(mSeqLens.iterator.toint() + b.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Int32)
        slen = slen if slen > cutlass.Int32(0) else cutlass.Int32(0)
        n_tok = slen if slen < s_tok else s_tok
    n_blocks_b = n_tok // cutlass.Int32(pool)
    valid = blk < n_blocks_b
    # Loads are clamped to the last complete block of THIS entry: in-bounds for
    # every group of a ragged last CTA, and the result of a clamped group is
    # never stored.
    blk_r = blk if blk < n_full else n_full - cutlass.Int32(1)
    tok0 = blk_r * cutlass.Int32(pool)

    b64 = b.to(cutlass.Int64)
    lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    k_row = cutlass.Int64(mK.stride[1]) * bpe
    k_base = mK.iterator.toint() + (b64 * cutlass.Int64(mK.stride[0]) + tok0.to(cutlass.Int64) * cutlass.Int64(mK.stride[1])) * bpe

    # --- pool: POOL rows -> fp32 sum -> exact power-of-two reciprocal ---------
    acc = []
    for c in cutlass.range_constexpr(chunks):
        acc.append([cutlass.Float32(0.0)] * ELEMS_PER_ACCESS)
    for r in cutlass.range_constexpr(pool):
        for c in cutlass.range_constexpr(chunks):
            off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
            pairs = [f16x2_to_f32(wd, dtype=mK.element_type) for wd in ld_global_v4(k_base + cutlass.Int64(r) * k_row + off, cutlass.Int32)]
            vals = [v for pair in pairs for v in pair]
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                acc[c][i] = acc[c][i] + vals[i]
    inv_pool = cutlass.Float32(1.0 / pool)
    pooled = []
    for c in cutlass.range_constexpr(chunks):
        pooled.append([acc[c][i] * inv_pool for i in range(ELEMS_PER_ACCESS)])

    # --- RMSNorm over D: lane partial -> 16-lane butterfly -> rsqrt -----------
    ssq = cutlass.Float32(0.0)
    for c in cutlass.range_constexpr(chunks):
        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
            ssq = ssq + pooled[c][i] * pooled[c][i]
    rstd = rmsnorm_rstd(ssq, lanes, d, eps)

    # The weight is consumed AFTER the reduction (the norm kernel's measured
    # register discipline); ``w + w_offset`` is the zero-centred form in fp32.
    w_base = mW.iterator.toint()
    ys = []
    for c in cutlass.range_constexpr(chunks):
        off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
        w_pairs = [f16x2_to_f32(wd, dtype=mW.element_type) for wd in ld_global_v4(w_base + off, cutlass.Int32)]
        w_vals = [v for pair in w_pairs for v in pair]
        ys.append([pooled[c][i] * rstd * (w_vals[i] + w_offset) for i in range(ELEMS_PER_ACCESS)])

    # --- partial RoPE on dims [0, rope_dim) at the block START position -------
    if cutlass.const_expr(rope_dim > 0):
        in_rope = lane < cutlass.Int32(rope_lanes)
        # A [1, S, R] table serves every batch entry; a [B, S, R] one is read per entry.
        bc = b if cutlass.Int32(mCos.shape[0]) > cutlass.Int32(1) else cutlass.Int32(0)
        bc64 = bc.to(cutlass.Int64)
        pos64 = tok0.to(cutlass.Int64)
        cos_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
        sin_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
        # Only the rope lanes load a table row: a clamped read on every lane
        # tripled the norm kernel's L1 request count for no data.
        if in_rope:
            c_addr = mCos.iterator.toint() + (bc64 * cutlass.Int64(mCos.stride[0]) + pos64 * cutlass.Int64(mCos.stride[1])) * bpe + lane_off
            s_addr = mSin.iterator.toint() + (bc64 * cutlass.Int64(mSin.stride[0]) + pos64 * cutlass.Int64(mSin.stride[1])) * bpe + lane_off
            c_pairs = [f16x2_to_f32(wd, dtype=mCos.element_type) for wd in ld_global_v4(c_addr, cutlass.Int32)]
            s_pairs = [f16x2_to_f32(wd, dtype=mSin.element_type) for wd in ld_global_v4(s_addr, cutlass.Int32)]
            cos_v = [v for pair in c_pairs for v in pair]
            sin_v = [v for pair in s_pairs for v in pair]
        # The shuffle inside rope_rotate_half is unconditional -- every lane of
        # the warp reaches it -- and only the rope lanes keep the result.
        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
            rotated = rope_rotate_half(ys[0][i], cos_v[i], sin_v[i], lane, half_lanes)
            ys[0][i] = rotated if in_rope else ys[0][i]

    # --- the ONE rounding, 256 B per block row (the e4m3 store arm: the bf16 words rounded once more, 128 B per row) ------
    out_fp8 = cutlass.const_expr(mOut.element_type == cutlass.Float8E4M3FN)
    out_bpe = cutlass.Int64(1) if cutlass.const_expr(out_fp8) else bpe
    if valid:
        out_base = mOut.iterator.toint() + (b64 * cutlass.Int64(mOut.stride[0]) + blk.to(cutlass.Int64) * cutlass.Int64(mOut.stride[1])) * out_bpe
        for c in cutlass.range_constexpr(chunks):
            y = ys[c]
            if cutlass.const_expr(out_fp8):
                _store_row_chunk_e4m3(out_base, c, lanes, lane, y, mK.element_type)
            else:
                off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                packed = [fp32_to_fp16(y[i], y[i + 1], dtype=mOut.element_type) for i in range(0, ELEMS_PER_ACCESS, 2)]
                st_global_v4(out_base + off, packed, cutlass.Int32)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def qsa_compress_launch(
    k: cute.Tensor,
    out: cute.Tensor,
    w: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    seq_lens: Optional[cute.Tensor],
    eps: cutlass.Float32,
    w_offset: cutlass.Float32,
    grid_x: cutlass.Int32,
    grid_y: cutlass.Int32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qsa_compress(k, out, w, cos, sin, seq_lens, eps, w_offset, d, rope_dim, pool, threads_per_cta, use_pdl).launch(
        grid=(grid_x, grid_y, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl
    )


compiled_cache = {}


class QsaCompressRecipe(NamedTuple):
    """Build-time facts of one compress launch -- everything derivable from the
    DECLARATION (dtypes, head dim, rope dim, pool, whether per-batch lengths
    are bound), so it is a legal compile key.  Batch, sequence length and the
    output's block capacity enter the artifact as ``cute.sym_int`` and ride in
    as runtime tensor shapes: one compile serves every shape.

    ``has_seq_lens`` records whether the artifact traced the per-batch length
    load; ``run_qsa_compress`` checks the bound ``seq_lens`` against it in BOTH
    directions (a None bound to a traced load dereferences null; a tensor bound
    to a dense artifact would be silently ignored).
    """

    compiled: object
    d: int
    rope_dim: int
    pool: int
    blocks_per_cta: int
    has_seq_lens: bool
    dtype: object
    table_dtype: object
    out_dtype: object = None  # appended: ``out``'s dtype -- the io dtype, or float8_e4m3fn (the e4m3 store arm)


def _fake_strided(dtype, shape, stride):
    return cute.runtime.make_fake_tensor(dtype=_convert_to_cutlass_data_type(dtype), shape=shape, stride=stride, assumed_align=16)


def _fake_compact(dtype, shape, stride_order):
    return cute.runtime.make_fake_compact_tensor(dtype=_convert_to_cutlass_data_type(dtype), shape=shape, stride_order=stride_order, assumed_align=16)


def compile_qsa_compress(
    *,
    dtype,
    d: int,
    rope_dim: int,
    pool: int = DEFAULT_POOL,
    has_seq_lens: bool = True,
    table_dtype=None,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    use_pdl: bool = False,
    out_dtype=None,
) -> QsaCompressRecipe:
    """Build the artifact from SHAPES ALONE -- no device allocation, no launch.

    ``dtype`` is the io dtype of ``k_raw`` / ``w`` (bf16 or f16) and of ``out`` unless ``out_dtype`` (appended) names the e4m3 store
    arm: ``None`` = the io dtype, ``torch.float8_e4m3fn`` = the e4m3 cast of the bf16 row (module docstring; bf16 io only; sm_89+ --
    an older part is a typed ``NotImplementedError`` naming the instruction); any other value a ``ValueError``.  ``table_dtype`` is
    that of ``cos`` / ``sin`` (defaults to ``dtype``).  The batch stride and the token stride of ``k_raw`` are SYMBOLIC, so a column
    slice of the fused projection slab (token stride = the slab width) and a compact ``[B, S, D]`` buffer share one artifact with no
    repack.  ``out_dtype`` is in the cache key (an e4m3 artifact is a different trace).
    """
    too_old = cutedsl_requirement_error("qsa_compress")
    if too_old is not None:
        raise NotImplementedError(too_old)
    validate_compress_shape(d, rope_dim, pool, threads_per_cta)
    if dtype not in _IO_DTYPES:
        raise ValueError(f"qsa_compress serves bf16/f16 io only, got dtype={dtype}")
    table_dtype = dtype if table_dtype is None else table_dtype
    if table_dtype not in _IO_DTYPES:
        raise ValueError(f"qsa_compress takes 16-bit cos / sin tables (bf16/f16), got table_dtype={table_dtype}")
    check_index_cache_dtype(out_dtype, dtype)
    out_dtype = dtype if out_dtype is None else out_dtype
    check_e4m3_store_target("qsa_compress", out_dtype)

    key = (str(dtype), str(table_dtype), int(d), int(rope_dim), int(pool), int(threads_per_cta), bool(has_seq_lens), bool(use_pdl), current_device()) + (
        () if out_dtype == dtype else (str(out_dtype),)
    )
    if key not in compiled_cache:
        batch = cute.sym_int()
        k = _fake_strided(dtype, (batch, cute.sym_int(), d), (cute.sym_int(), cute.sym_int(), 1))
        out = _fake_compact(out_dtype, (batch, cute.sym_int(), d), (2, 1, 0))
        w = _fake_compact(dtype, (d,), (0,))
        table_dim = rope_dim if rope_dim else 1
        tables = [_fake_strided(table_dtype, (cute.sym_int(), cute.sym_int(), table_dim), (cute.sym_int(), cute.sym_int(), 1)) for _ in range(2)]
        seq_lens = _fake_compact(torch.int32, (batch,), (0,)) if has_seq_lens else None
        compiled_cache[key] = cute.compile(
            qsa_compress_launch,
            k,
            out,
            w,
            *tables,
            seq_lens,
            cutlass.Float32(1e-6),  # eps       ) runtime values: the literals pin
            cutlass.Float32(0.0),  # w_offset   ) their TYPE at trace time only
            cutlass.Int32(1),  # grid_x
            cutlass.Int32(1),  # grid_y
            int(d),
            int(rope_dim),
            int(pool),
            int(threads_per_cta),
            bool(use_pdl),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QsaCompressRecipe(
        compiled=compiled_cache[key],
        d=int(d),
        rope_dim=int(rope_dim),
        pool=int(pool),
        blocks_per_cta=threads_per_cta // lanes_per_row(d),
        has_seq_lens=bool(has_seq_lens),
        dtype=dtype,
        table_dtype=table_dtype,
        out_dtype=out_dtype,
    )


def _check_row_tensor(name: str, x: torch.Tensor, *, dtype, last_dim: int) -> None:
    """A ``[..., last_dim]`` tensor of ``dtype`` whose rows the kernel moves in 16-B vectors: unit last stride, every other stride a
    multiple of one 16-B vector of elements (8 at 16 bits, 16 at e4m3), base 16-B aligned."""
    if x.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype} (the recipe's), got {x.dtype}")
    if x.shape[-1] != last_dim:
        raise ValueError(f"{name} must have {last_dim} elements per row, got shape {tuple(x.shape)}")
    if x.stride(-1) != 1:
        raise ValueError(f"{name} must be contiguous along its last dim, got strides {tuple(x.stride())}")
    per_access = ACCESS_BYTES // max(1, x.element_size())  # elements per 16-B vector (8 at 16 bits, 16 at e4m3)
    for dim in range(x.ndim - 1):
        if x.stride(dim) % per_access != 0:
            raise ValueError(
                f"{name} stride {x.stride(dim)} (dim {dim}) must be a multiple of {per_access} elements so every row starts 16-B aligned; got strides {tuple(x.stride())}"
            )
    if x.data_ptr() % ACCESS_BYTES != 0:
        raise ValueError(f"{name} must be {ACCESS_BYTES}-B aligned for 128-bit accesses")


def run_qsa_compress(
    r: QsaCompressRecipe,
    k_raw: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_lens: Optional[torch.Tensor] = None,
    *,
    eps: float,
    norm_weight_offset: float = 0.0,
    stream,
) -> None:
    """The lowered launch: validation and address arithmetic only -- no
    conversion, no allocation.

    ``k_raw`` ``[B, S, D]`` (any batch / token stride with 16-B aligned rows),
    ``out`` ``[B, NB, D]`` compact with ``NB >= floor(S / pool)`` (rows at or
    beyond ``floor(S / pool)`` are never written), ``w`` ``[D]``, ``cos`` /
    ``sin`` ``[1 or B, >= pool * (floor(S / pool) - 1) + 1, rope_dim]``,
    ``seq_lens`` ``[B]`` int32 iff the recipe traced it.  ``norm_weight_offset``
    is added to ``w`` in fp32 (0.0 for a pre-folded ``(1 + w)``, 1.0 for the
    zero-centred checkpoint weight).  With no complete block (``S < pool``)
    nothing is launched.
    """
    if k_raw.ndim != 3:
        raise ValueError(f"k_raw must be [B, S, D], got shape {tuple(k_raw.shape)}")
    batch, s_tok, d = (int(x) for x in k_raw.shape)
    if d != r.d:
        raise ValueError(f"k_raw head dim {d} does not match the recipe's d={r.d}")
    _check_row_tensor("k_raw", k_raw, dtype=r.dtype, last_dim=r.d)
    n_full = s_tok // r.pool
    if out.ndim != 3 or int(out.shape[0]) != batch or int(out.shape[2]) != r.d:
        raise ValueError(f"out must be [B={batch}, NB, D={r.d}], got shape {tuple(out.shape)}")
    if int(out.shape[1]) < n_full:
        raise ValueError(f"out holds {int(out.shape[1])} blocks per batch entry but S={s_tok} has {n_full} complete blocks of {r.pool}")
    if not out.is_contiguous():
        raise ValueError("out must be a compact [B, NB, D] tensor")
    _check_row_tensor("out", out, dtype=r.out_dtype if r.out_dtype is not None else r.dtype, last_dim=r.d)  # the recipe's cache dtype, both ways
    if w.ndim != 1 or int(w.shape[0]) != r.d or not w.is_contiguous():
        raise ValueError(f"w must be a contiguous [{r.d}] norm weight, got shape {tuple(w.shape)}")
    _check_row_tensor("w", w, dtype=r.dtype, last_dim=r.d)
    if r.rope_dim:
        for name, t in (("cos", cos), ("sin", sin)):
            if t.ndim != 3:
                raise ValueError(f"{name} must be [1 or B, S_pos, rope_dim], got shape {tuple(t.shape)}")
            if int(t.shape[0]) not in (1, batch):
                raise ValueError(f"{name} batch dim must be 1 (broadcast) or B={batch}, got {int(t.shape[0])}")
            if n_full > 0 and int(t.shape[1]) < r.pool * (n_full - 1) + 1:
                raise ValueError(f"{name} has {int(t.shape[1])} positions; the last complete block starts at position {r.pool * (n_full - 1)}")
            _check_row_tensor(name, t, dtype=r.table_dtype, last_dim=r.rope_dim)
        if tuple(cos.shape) != tuple(sin.shape):
            raise ValueError(f"cos and sin must share a shape, got {tuple(cos.shape)} and {tuple(sin.shape)}")
    if r.has_seq_lens:
        if seq_lens is None:
            raise ValueError("this artifact was compiled WITH per-batch seq_lens; bind a [B] int32 tensor (no silent dense fallback)")
        if seq_lens.dtype != torch.int32 or tuple(seq_lens.shape) != (batch,) or not seq_lens.is_contiguous():
            raise ValueError(f"seq_lens must be a contiguous int32 [B={batch}] tensor, got {seq_lens.dtype} {tuple(seq_lens.shape)}")
    elif seq_lens is not None:
        raise ValueError("this artifact was compiled WITHOUT seq_lens (every entry has S valid tokens); seq_lens would be silently ignored -- pass None")
    if n_full == 0:
        return  # S < pool: no complete block anywhere, nothing to launch
    grid_x = (n_full + r.blocks_per_cta - 1) // r.blocks_per_cta
    # The seq_lens slot stays in the ABI when it traced to None (the artifact
    # folded out the LOAD, not the parameter), so it is always passed.
    r.compiled(
        k_raw,
        out,
        w,
        cos,
        sin,
        seq_lens,
        cutlass.Float32(eps),
        cutlass.Float32(norm_weight_offset),
        cutlass.Int32(grid_x),
        cutlass.Int32(batch),
        cuda.CUstream(int(stream)),
    )


def build_qsa_compress(
    k_raw: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seq_lens: Optional[torch.Tensor] = None,
    *,
    rope_dim: int,
    eps: float,
    norm_weight_offset: float = 0.0,
    pool: int = DEFAULT_POOL,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    use_pdl: bool = False,
    stream,
) -> QsaCompressRecipe:
    """Compile (cached) and run once -- the convenience form for tests and
    benchmarks.  Production callers split it: :func:`compile_qsa_compress` at
    plan time, :func:`run_qsa_compress` per execute."""
    if k_raw.ndim != 3:
        raise ValueError(f"k_raw must be [B, S, D], got shape {tuple(k_raw.shape)}")
    r = compile_qsa_compress(
        dtype=k_raw.dtype,
        d=int(k_raw.shape[2]),
        rope_dim=rope_dim,
        pool=pool,
        has_seq_lens=seq_lens is not None,
        table_dtype=cos.dtype,
        threads_per_cta=threads_per_cta,
        use_pdl=use_pdl,
        out_dtype=out.dtype,
    )
    run_qsa_compress(r, k_raw, out, w, cos, sin, seq_lens, eps=eps, norm_weight_offset=norm_weight_offset, stream=stream)
    return r


def moved_bytes(batch: int, seq_len: int, d: int, *, pool: int = DEFAULT_POOL, elem_bytes: int = 2) -> int:
    """HBM traffic of one launch -- the denominator for an SOL number: the
    complete blocks' ``pool`` rows read once, one row per block written.  The
    weight and the two table rows per block are L2-resident and excluded."""
    n_full = seq_len // pool
    return batch * n_full * (pool + 1) * d * elem_bytes


# ---------------------------------------------------------------------------
# The decode form: raw-key ring + this step's rows -> the blocks they complete
# ---------------------------------------------------------------------------

SLOT_NO_BLOCK = -1
"""``compressed_slot`` value: this row completes no block, or is an uncommitted draft."""
SLOT_CAPACITY = -2
"""``compressed_slot`` value: the committed row's position lies at or past the RoPE table (``pos >= S_pos`` -- the table is the ceiling on
positions, checked in-kernel on the live position of EVERY committed row, completing or not: no ring row, no block, and under the fused
launch's RoPE-table arm no Q / K / indexer-query row is written for it), or the row completes a block the compressed-key cache cannot
hold, or (paged form) a block that needs ring rows the sequence does not own or a block on an unassigned page (the block is not written;
the committed row's raw key still enters the ring, since the position itself is served)."""

_RING_SLOT_DTYPES = (torch.int32, torch.int64)


def check_e4m3_store_target(who: str, cache_dtype) -> None:
    """The e4m3 store arm's instruction (``cvt.rn.satfinite.e4m3x2.f32``) needs sm_89+: an older part is declined BY NAME at compile time
    (Rule 7), never through the ptxas error the trace would die with.  A no-op for a 16-bit cache."""
    if cache_dtype != _E4M3:
        return
    cc = tuple(compute_capability(current_device()))
    if cc < _FP8_CVT_MIN_CC:
        raise NotImplementedError(
            f"{who}: the e4m3 compressed-key store arm (cvt.rn.satfinite.e4m3x2.f32) needs sm_{_FP8_CVT_MIN_CC[0]}{_FP8_CVT_MIN_CC[1]} or later "
            f"(Ada / Hopper / Blackwell / Rubin); the current device is sm_{cc[0]}{cc[1]} -- declare a bf16 compressed-key cache there"
        )


def min_ring_rows(rows_per_step: int, pool: int = DEFAULT_POOL) -> int:
    """The smallest raw-key ring that serves a step of ``rows_per_step`` tokens per sequence: a completing row pools
    ring positions down to ``pos0 - pool + 1`` while the step writes positions ``pos0 .. pos0 + rows - 1``; with
    ``rows + pool - 1`` rows no write of a launch aliases one of its reads, and no committed key a later step pools is
    overwritten by a draft."""
    return int(rows_per_step) + int(pool) - 1


# -- the shared paged-addressing helpers: ONE formula per table, called by the step kernel and by the fused prepare launch --


@cute.jit
def step_position(mPos0: Optional[cute.Tensor], mKvLens: Optional[cute.Tensor], b: cutlass.Int32, s_q: cutlass.Int32) -> cutlass.Int32:
    """The position of row 0 of sequence ``b``: ``kv_lens[b] - S_q`` from the KV length AFTER the step (the paged form -- a
    negative value marks the rows below position 0 dead), or the ``pos0`` vector clamped at 0 (the dense form -- a negative
    ``pos0`` is the first step of a sequence).  Exactly one of the two vectors is bound (the host checks)."""
    if cutlass.const_expr(mKvLens is not None):
        pos0 = ld_global(mKvLens.iterator.toint() + b.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Int32) - s_q
    else:
        p0 = ld_global(mPos0.iterator.toint() + b.to(cutlass.Int64) * cutlass.Int64(4), cutlass.Int32)
        pos0 = p0 if p0 > cutlass.Int32(0) else cutlass.Int32(0)
    return pos0


@cute.jit
def ring_entry(mStateTable: cute.Tensor, b: cutlass.Int32, mRing: cute.Tensor):
    """The sequence's ring block from ``index_state_block_table [B, 1]`` as ``(entry, owns_ring)``, range-checked against the ring
    ``mRing [blocks, cap, 1, D]`` on BOTH sides: block 0 is the null block (a live request's entry is ``> 0``), a negative entry is
    unassigned, and an entry ``>= blocks`` lies past the allocation (a ring of one block per request while the ids start at 1);
    every one of them reads as block 0 -- in-bounds, since the ring holds at least one block -- with ``owns_ring`` False, so the
    row writes no ring row and pools no ring row (the callers report ``-2`` for a block that would need one)."""
    blk = ld_global(mStateTable.iterator.toint() + b.to(cutlass.Int64) * cutlass.Int64(mStateTable.stride[0]) * cutlass.Int64(4), cutlass.Int32)
    owns = (blk > cutlass.Int32(0)) & (blk < cutlass.Int32(mRing.shape[0]))
    entry = blk if owns else cutlass.Int32(0)
    return entry, owns


@cute.jit
def ring_row_address(mRing: cute.Tensor, entry: cutlass.Int32, pos: cutlass.Int32, bpe: cutlass.Int64) -> cutlass.Int64:
    """Byte address of the ring row of position ``pos`` in ring ``entry``: ``base + entry * stride[0] + (pos % shape[1]) * stride[1]``
    -- the dense ring ``[B, RING, D]`` (``entry`` = the sequence) and the paged ``[blocks, cap, 1, D]`` (``entry`` = the
    sequence's state block; the slot ``block * cap + pos % cap`` in rows) through ONE formula.  ``pos`` is >= 0 (the callers
    clamp a dead row's position to 0 and store nothing for it)."""
    ring_n = cutlass.Int32(mRing.shape[1])
    return (
        mRing.iterator.toint()
        + (entry.to(cutlass.Int64) * cutlass.Int64(mRing.stride[0]) + (pos % ring_n).to(cutlass.Int64) * cutlass.Int64(mRing.stride[1])) * bpe
    )


@cute.jit
def ring_slot_address(mRing: cute.Tensor, mSlots: cute.Tensor, t: cutlass.Int32, slot_i64: cutlass.Constexpr[bool], bpe: cutlass.Int64):
    """The ring WRITE row named by the stack's own slot mapping: ``index_state_slot_mapping[t]`` is the flat row
    ``block * cap + pos % cap`` of a ring whose blocks are contiguous (``stride[0] == cap * stride[1]``, host-checked), ``-1`` =
    no write.  Returns ``(byte address, live)``, the address clamped to row 0 when the slot is out of range.  int64 slots are
    read as two 32-bit halves so a slot >= 2^31 can never alias an in-range one."""
    t64 = t.to(cutlass.Int64)
    if cutlass.const_expr(slot_i64):
        lo, hi = ld_global_v2(mSlots.iterator.toint() + t64 * cutlass.Int64(8), cutlass.Int32)
        slot = (hi.to(cutlass.Int64) * cutlass.Int64(1 << 32)) + (lo.to(cutlass.Int64) & cutlass.Int64(0xFFFFFFFF))
    else:
        slot = ld_global(mSlots.iterator.toint() + t64 * cutlass.Int64(4), cutlass.Int32).to(cutlass.Int64)
    n_rows = cutlass.Int64(mRing.shape[0]) * cutlass.Int64(mRing.shape[1])
    live = (slot >= cutlass.Int64(0)) & (slot < n_rows)
    slot_r = slot if live else cutlass.Int64(0)
    addr = mRing.iterator.toint() + slot_r * cutlass.Int64(mRing.stride[1]) * bpe
    return addr, live


@cute.jit
def paged_block_row(mCache: cute.Tensor, mTable: cute.Tensor, b: cutlass.Int32, blk: cutlass.Int32, bpe: cutlass.Int64):
    """The compressed-key cache row of block ``blk`` of sequence ``b`` through the MAIN block table -- page
    ``table[b, blk // rpp]``, row ``blk % rpp`` with ``rpp = cache.shape[1]`` (= P / POOL: the compressed rows per attention
    page) -- as ``(byte address, fits)``: ``fits`` is False past the table's width, on a negative (unassigned) page and on a
    page past the cache; the address is then clamped in-bounds (page 0, the block's row) so a lane group that stores nothing
    still loads inside the allocation.  ``blk`` is >= 0 (the callers clamp)."""
    rpp = cutlass.Int32(mCache.shape[1])
    n_pages = cutlass.Int32(mCache.shape[0])
    width = cutlass.Int32(mTable.shape[1])
    lblk = blk // rpp
    row = blk - lblk * rpp
    in_table = lblk < width
    lblk_r = lblk if in_table else width - cutlass.Int32(1)
    page = ld_global(
        mTable.iterator.toint()
        + (b.to(cutlass.Int64) * cutlass.Int64(mTable.stride[0]) + lblk_r.to(cutlass.Int64) * cutlass.Int64(mTable.stride[1])) * cutlass.Int64(4),
        cutlass.Int32,
    )
    fits = in_table & (page >= cutlass.Int32(0)) & (page < n_pages)
    page_r = page if fits else cutlass.Int32(0)
    addr = (
        mCache.iterator.toint() + (page_r.to(cutlass.Int64) * cutlass.Int64(mCache.stride[0]) + row.to(cutlass.Int64) * cutlass.Int64(mCache.stride[1])) * bpe
    )
    return addr, fits


@cute.kernel
def frost_qsa_compress_step(
    mKnew: cute.Tensor,  # [B, S_q, D] io dtype; batch / token strides symbolic (a column slice of the projection slab is fine)
    mRing: cute.Tensor,  # dense: [B, RING, D] compact, row p % RING = position p's raw key; paged: [blocks, cap, 1, D] behind mStateTable
    mOut: cute.Tensor,  # dense: [B, NB, D] compact, row blk = block blk; paged: [num_pages, P / POOL, 1, D] behind mBlockTable
    mW: cute.Tensor,  # [D] io dtype
    mCos: cute.Tensor,  # [Bc, S_pos, ROPE_DIM] 16-bit, Bc in {1, B}
    mSin: cute.Tensor,  # same shape and dtype as mCos
    mPos0: Optional[cute.Tensor],  # [B] int32: the position of row 0 (= the KV length before the step); None on the paged form
    mNCommit: Optional[cute.Tensor],  # [B] int32: the accepted prefix of the step's rows, or None (= every row)
    mSlot: cute.Tensor,  # [B, S_q] int32 out: the block id written by the row, -1 (none) or -2 (capacity)
    mKvLens: Optional[cute.Tensor],  # [B] int32: the KV length AFTER the step (the paged form's position source), or None
    mBlockTable: Optional[cute.Tensor],  # [B, max_pages] int32: the MAIN block table (the paged form), or None
    mStateTable: Optional[cute.Tensor],  # [B, 1] int32: the ring's block table (the paged form), or None
    mRingSlots: Optional[cute.Tensor],  # [B * S_q] int32 / int64: the stack's own ring slots (paged, ring_slots), or None
    eps: cutlass.Float32,
    w_offset: cutlass.Float32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    ring_slot_i64: cutlass.Constexpr[bool],
) -> None:
    """One lane group (``lanes_per_row(d)`` lanes) per NEW ROW of the step.

    Grid ``(ceil(S_q / groups_per_cta), B)``: ``block_idx.y`` is the sequence, ``block_idx.x * groups_per_cta +
    grp`` the row ``j`` within the step.  A committed row copies its raw key into the ring; a committed row that
    completes a block pools the block's ``pool`` rows (positions below ``pos0`` from the ring, the rest from this
    step's rows) and runs the prefill kernel's fp32 chain -- the same operations in the same order.  Every other
    group runs the arithmetic on clamped, in-bounds loads (so every lane of the warp reaches the butterfly
    shuffles) and stores nothing.  The ring writes and the ring reads of one launch never alias (host rule
    ``RING >= S_q + pool - 1``), so there is no intra-launch ordering requirement.  The dense and the paged form
    (``mBlockTable`` present) differ ONLY in where a position's ring row and a block's cache row live
    (:func:`ring_row_address`, :func:`paged_block_row`) and in the position source (:func:`step_position`).
    """
    if cutlass.const_expr(use_pdl):
        wait_on_dependent_grids()

    has_n_commit = cutlass.const_expr(mNCommit is not None)
    paged = cutlass.const_expr(mBlockTable is not None)
    has_ring_slots = cutlass.const_expr(mRingSlots is not None)
    lanes = cutlass.const_expr(lanes_per_row(d))
    chunks = cutlass.const_expr(vec_chunks(d))
    groups_per_cta = cutlass.const_expr(threads_per_cta // lanes)
    rope_lanes = cutlass.const_expr(rope_dim // ELEMS_PER_ACCESS)
    half_lanes = cutlass.const_expr(rope_lanes // 2)
    bpe = cutlass.Int64(2)

    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    lane = tidx % cutlass.Int32(lanes)
    grp = tidx // cutlass.Int32(lanes)
    b = cutlass.Int32(cute.arch.block_idx()[1])
    j = cutlass.Int32(cute.arch.block_idx()[0]) * cutlass.Int32(groups_per_cta) + grp

    s_q = cutlass.Int32(mKnew.shape[1])
    s_pos = cutlass.Int32(mCos.shape[1])
    b64 = b.to(cutlass.Int64)

    in_row = j < s_q
    jr = j if in_row else s_q - cutlass.Int32(1)  # the ragged last CTA computes on the last row and stores nothing
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
    out_fp8 = cutlass.const_expr(mOut.element_type == cutlass.Float8E4M3FN)  # the e4m3 store arm (module docstring): 1 B per element
    out_bpe = cutlass.Int64(1) if cutlass.const_expr(out_fp8) else bpe
    if cutlass.const_expr(paged):
        entry, owns_ring = ring_entry(mStateTable, b, mRing)
        out_row, fits = paged_block_row(mOut, mBlockTable, b, blk, out_bpe)
        # a block that needs ring rows (it starts below pos0) from a sequence that owns no ring: -2, never the null block's rows
        fits = fits & (owns_ring | (tok0 >= pos0))
    else:
        entry = b
        owns_ring = cutlass.Boolean(True)
        n_cap = cutlass.Int32(mOut.shape[1])
        fits = blk < n_cap
        blk_r = blk if fits else n_cap - cutlass.Int32(1)
        out_row = mOut.iterator.toint() + (b64 * cutlass.Int64(mOut.stride[0]) + blk_r.to(cutlass.Int64) * cutlass.Int64(mOut.stride[1])) * out_bpe
    # The RoPE table is the CEILING on positions: a committed row at p >= S_pos has no table row (the fused launch's RoPE-table arm
    # would rotate it with a clamped row), so it writes nothing -- no ring row, no block -- and reports -2.  A block's rows all lie at
    # or below its completing row's position, so no block ever pools a ring row at or past the table, and the block start's table row
    # exists whenever the completing row's does (tok0 <= p).
    in_table = cutlass.Boolean(True)
    past_table = cutlass.Boolean(False)
    if cutlass.const_expr(rope_dim > 0):
        in_table = p < s_pos
        past_table = p >= s_pos
    write_blk = committed & completes & fits & in_table

    lane_off = lane.to(cutlass.Int64) * cutlass.Int64(ACCESS_BYTES)
    knew_tok = cutlass.Int64(mKnew.stride[1]) * bpe
    knew_b = mKnew.iterator.toint() + b64 * cutlass.Int64(mKnew.stride[0]) * bpe

    # --- (1) the committed row's raw key enters the ring: a bit copy, 16 B per lane (never for a row at or past the table) ----
    ring_write = committed & owns_ring & in_table
    if cutlass.const_expr(has_ring_slots):
        dst, slot_live = ring_slot_address(mRing, mRingSlots, b * s_q + jr, ring_slot_i64, bpe)
        ring_write = ring_write & slot_live
    else:
        dst = ring_row_address(mRing, entry, p, bpe)
    if ring_write:
        src = knew_b + jr.to(cutlass.Int64) * knew_tok
        for c in cutlass.range_constexpr(chunks):
            off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
            words = ld_global_v4(src + off, cutlass.Int32)
            st_global_v4(dst + off, words, cutlass.Int32)

    # --- (2) pool: the block's rows in POSITION order, each from the ring (below pos0) or this step ----
    acc = []
    for c in cutlass.range_constexpr(chunks):
        acc.append([cutlass.Float32(0.0)] * ELEMS_PER_ACCESS)
    for r in cutlass.range_constexpr(pool):
        q = tok0 + cutlass.Int32(r)
        from_new = q >= pos0
        row_new = q - pos0
        row_new = row_new if row_new > cutlass.Int32(0) else cutlass.Int32(0)
        row_new = row_new if row_new < s_q else s_q - cutlass.Int32(1)
        addr_new = knew_b + row_new.to(cutlass.Int64) * knew_tok
        addr_ring = ring_row_address(mRing, entry, q, bpe)
        row_base = addr_new if from_new else addr_ring
        for c in cutlass.range_constexpr(chunks):
            off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
            pairs = [f16x2_to_f32(wd, dtype=mKnew.element_type) for wd in ld_global_v4(row_base + off, cutlass.Int32)]
            vals = [v for pair in pairs for v in pair]
            for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
                acc[c][i] = acc[c][i] + vals[i]
    inv_pool = cutlass.Float32(1.0 / pool)
    pooled = []
    for c in cutlass.range_constexpr(chunks):
        pooled.append([acc[c][i] * inv_pool for i in range(ELEMS_PER_ACCESS)])

    # --- RMSNorm over D (the prefill kernel's chain) ------------------------------
    ssq = cutlass.Float32(0.0)
    for c in cutlass.range_constexpr(chunks):
        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
            ssq = ssq + pooled[c][i] * pooled[c][i]
    rstd = rmsnorm_rstd(ssq, lanes, d, eps)

    w_base = mW.iterator.toint()
    ys = []
    for c in cutlass.range_constexpr(chunks):
        off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
        w_pairs = [f16x2_to_f32(wd, dtype=mW.element_type) for wd in ld_global_v4(w_base + off, cutlass.Int32)]
        w_vals = [v for pair in w_pairs for v in pair]
        ys.append([pooled[c][i] * rstd * (w_vals[i] + w_offset) for i in range(ELEMS_PER_ACCESS)])

    # --- partial RoPE at the block START position (table row clamped in-bounds) ------
    if cutlass.const_expr(rope_dim > 0):
        in_rope = lane < cutlass.Int32(rope_lanes)
        bc = b if cutlass.Int32(mCos.shape[0]) > cutlass.Int32(1) else cutlass.Int32(0)
        bc64 = bc.to(cutlass.Int64)
        tok0_r = tok0 if tok0 < s_pos else s_pos - cutlass.Int32(1)
        pos64 = tok0_r.to(cutlass.Int64)
        cos_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
        sin_v = [cutlass.Float32(0.0)] * ELEMS_PER_ACCESS
        if in_rope:
            c_addr = mCos.iterator.toint() + (bc64 * cutlass.Int64(mCos.stride[0]) + pos64 * cutlass.Int64(mCos.stride[1])) * bpe + lane_off
            s_addr = mSin.iterator.toint() + (bc64 * cutlass.Int64(mSin.stride[0]) + pos64 * cutlass.Int64(mSin.stride[1])) * bpe + lane_off
            c_pairs = [f16x2_to_f32(wd, dtype=mCos.element_type) for wd in ld_global_v4(c_addr, cutlass.Int32)]
            s_pairs = [f16x2_to_f32(wd, dtype=mSin.element_type) for wd in ld_global_v4(s_addr, cutlass.Int32)]
            cos_v = [v for pair in c_pairs for v in pair]
            sin_v = [v for pair in s_pairs for v in pair]
        for i in cutlass.range_constexpr(ELEMS_PER_ACCESS):
            rotated = rope_rotate_half(ys[0][i], cos_v[i], sin_v[i], lane, half_lanes)
            ys[0][i] = rotated if in_rope else ys[0][i]

    # --- the ONE rounding, into the block's own cache row (the e4m3 store arm: the bf16 words rounded once more) ----------
    if write_blk:
        for c in cutlass.range_constexpr(chunks):
            y = ys[c]
            if cutlass.const_expr(out_fp8):
                _store_row_chunk_e4m3(out_row, c, lanes, lane, y, mKnew.element_type)
            else:
                off = cutlass.Int64((c * lanes) * ACCESS_BYTES) + lane_off
                packed = [fp32_to_fp16(y[i], y[i + 1], dtype=mOut.element_type) for i in range(0, ELEMS_PER_ACCESS, 2)]
                st_global_v4(out_row + off, packed, cutlass.Int32)

    # --- the per-row report: the block id written, -1 (no block / draft), -2 (capacity: a block the cache cannot hold, or a row at or
    # past the table) ----
    slot_val = cutlass.Int32(SLOT_CAPACITY) if (committed & (completes | past_table)) else cutlass.Int32(SLOT_NO_BLOCK)
    slot_val = blk if write_blk else slot_val
    if in_row & (lane == cutlass.Int32(0)):
        slot_addr = mSlot.iterator.toint() + (b64 * cutlass.Int64(mSlot.stride[0]) + jr.to(cutlass.Int64) * cutlass.Int64(mSlot.stride[1])) * cutlass.Int64(4)
        st_global(slot_addr, slot_val, cutlass.Int32)

    if cutlass.const_expr(use_pdl):
        launch_dependent_grids()


@cute.jit
def qsa_compress_step_launch(
    k_new: cute.Tensor,
    ring: cute.Tensor,
    out: cute.Tensor,
    w: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    pos0: Optional[cute.Tensor],
    n_commit: Optional[cute.Tensor],
    slot: cute.Tensor,
    kv_lens: Optional[cute.Tensor],
    block_table: Optional[cute.Tensor],
    state_table: Optional[cute.Tensor],
    ring_slots: Optional[cute.Tensor],
    eps: cutlass.Float32,
    w_offset: cutlass.Float32,
    grid_x: cutlass.Int32,
    grid_y: cutlass.Int32,
    d: cutlass.Constexpr[int],
    rope_dim: cutlass.Constexpr[int],
    pool: cutlass.Constexpr[int],
    threads_per_cta: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
    ring_slot_i64: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_qsa_compress_step(
        k_new,
        ring,
        out,
        w,
        cos,
        sin,
        pos0,
        n_commit,
        slot,
        kv_lens,
        block_table,
        state_table,
        ring_slots,
        eps,
        w_offset,
        d,
        rope_dim,
        pool,
        threads_per_cta,
        use_pdl,
        ring_slot_i64,
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_cta, 1, 1), stream=stream, use_pdl=use_pdl)


step_compiled_cache = {}


class QsaCompressStepRecipe(NamedTuple):
    """Build-time facts of one decode-step compress launch (a legal compile key): dtypes, head dim, rope dim, pool,
    whether an accepted-prefix vector is bound, whether the state is PAGED and whether the ring's write rows come from the
    stack's own slot mapping (and its dtype).  Batch, rows per step, ring depth, the cache capacity and the page geometry
    are ``cute.sym_int`` and ride in as runtime shapes: one artifact serves every step shape, every ring and every page size."""

    compiled: object
    d: int
    rope_dim: int
    pool: int
    rows_per_cta: int
    has_n_commit: bool
    dtype: object
    table_dtype: object
    paged: bool = False
    ring_slots: bool = False
    ring_slot_dtype: object = torch.int64
    index_cache_dtype: object = None  # appended: the compressed-key cache's dtype -- None = the io dtype, float8_e4m3fn = the e4m3 store arm


def _fake_i32_table(shape, stride):
    """An int32 table ``[B, cols]`` with a symbolic (or fixed) column count and a SYMBOLIC row stride -- a row slice of a
    wider table binds too -- declared 4-byte aligned (scalar loads; a slice's base is a multiple of 4 only)."""
    return cute.runtime.make_fake_tensor(dtype=cutlass.Int32, shape=shape, stride=stride, assumed_align=4)


def _fake_ring_slots(slot_dtype):
    """``[T]`` contiguous int32 / int64 with a symbolic length; 4- / 8-byte aligned (the natural alignment of its element)."""
    i64 = slot_dtype == torch.int64
    return cute.runtime.make_fake_compact_tensor(cutlass.Int64 if i64 else cutlass.Int32, (cute.sym_int(),), stride_order=(0,), assumed_align=8 if i64 else 4)


def _fake_paged_state(dtype, d):
    """``[blocks, rows, 1, D]`` -- the paged ring or the paged compressed-key cache -- with D contiguous and every other
    stride symbolic (the stack's own strides bind, no repack)."""
    return _fake_strided(dtype, (cute.sym_int(), cute.sym_int(), 1, d), (cute.sym_int(), cute.sym_int(), cute.sym_int(), 1))


def check_index_cache_dtype(index_cache_dtype, dtype) -> None:
    """The compressed-key cache's dtype against the compress arms' contract: ``None`` or the io dtype is served; ``float8_e4m3fn`` is
    the e4m3 store arm (the e4m3 cast of the bf16 row -- bf16 io only: f16 io has no e4m3 arm); any other dtype is a ``ValueError``
    naming the three."""
    if index_cache_dtype is None or index_cache_dtype == dtype:
        return
    if index_cache_dtype == _E4M3:
        if dtype != torch.bfloat16:
            raise ValueError(
                f"the e4m3 compressed-key cache is the e4m3 cast of the bf16 row (the serving stacks' indexer-cache convention); it needs bf16 io, got {dtype}"
            )
        return
    raise ValueError(
        f"the compressed-key cache must hold the io dtype {dtype} or torch.float8_e4m3fn (the e4m3 store arm) on the compress arms, got {index_cache_dtype}"
    )


def compile_qsa_compress_step(
    *,
    dtype,
    d: int,
    rope_dim: int,
    pool: int = DEFAULT_POOL,
    has_n_commit: bool = True,
    table_dtype=None,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    use_pdl: bool = False,
    paged: bool = False,
    ring_slots: bool = False,
    ring_slot_dtype=torch.int64,
    index_cache_dtype=None,
) -> QsaCompressStepRecipe:
    """Build the decode-step artifact from SHAPES ALONE -- no device allocation, no launch (see
    :func:`compile_qsa_compress` for the io / table dtype rules, which are the same here).

    ``paged=True`` traces the PAGED form (module docstring): the ring ``[blocks, cap, 1, D]`` behind ``index_state_block_table``,
    the compressed-key cache ``[num_pages, P / pool, 1, D]`` behind the main ``block_table``, positions from ``kv_lens``;
    ``ring_slots=True`` (paged only) takes the ring WRITE rows from ``index_state_slot_mapping`` of ``ring_slot_dtype``
    instead of deriving them; ``index_cache_dtype`` names the cache's dtype -- ``None`` = the io dtype, ``torch.float8_e4m3fn`` =
    the e4m3 store arm (module docstring; the dense AND the paged form, bf16 io only, sm_89+ -- an older part is a typed
    ``NotImplementedError`` naming the instruction).  Every flag is in the cache key.
    """
    too_old = cutedsl_requirement_error("qsa_compress_step")
    if too_old is not None:
        raise NotImplementedError(too_old)
    validate_compress_shape(d, rope_dim, pool, threads_per_cta)
    if dtype not in _IO_DTYPES:
        raise ValueError(f"qsa_compress_step serves bf16/f16 io only, got dtype={dtype}")
    table_dtype = dtype if table_dtype is None else table_dtype
    if table_dtype not in _IO_DTYPES:
        raise ValueError(f"qsa_compress_step takes 16-bit cos / sin tables (bf16/f16), got table_dtype={table_dtype}")
    if ring_slots and not paged:
        raise ValueError("ring_slots=True takes the ring write rows from the stack's slot mapping of the PAGED ring; it needs paged=True")
    if ring_slot_dtype not in _RING_SLOT_DTYPES:
        raise ValueError(f"ring_slot_dtype must be int32 or int64, got {ring_slot_dtype}")
    check_index_cache_dtype(index_cache_dtype, dtype)
    cache_dtype = dtype if index_cache_dtype is None else index_cache_dtype
    check_e4m3_store_target("qsa_compress_step", cache_dtype)

    key = (
        str(dtype),
        str(table_dtype),
        int(d),
        int(rope_dim),
        int(pool),
        int(threads_per_cta),
        bool(has_n_commit),
        bool(use_pdl),
        current_device(),
        bool(paged),
        bool(ring_slots),
        str(ring_slot_dtype) if ring_slots else None,
        str(cache_dtype) if cache_dtype != dtype else None,  # appended: the e4m3 store arm is its own artifact
    )
    if key not in step_compiled_cache:
        batch = cute.sym_int()
        rows = cute.sym_int()
        k_new = _fake_strided(dtype, (batch, rows, d), (cute.sym_int(), cute.sym_int(), 1))
        w = _fake_compact(dtype, (d,), (0,))
        table_dim = rope_dim if rope_dim else 1
        tables = [_fake_strided(table_dtype, (cute.sym_int(), cute.sym_int(), table_dim), (cute.sym_int(), cute.sym_int(), 1)) for _ in range(2)]
        n_commit = _fake_compact(torch.int32, (batch,), (0,)) if has_n_commit else None
        slot = _fake_compact(torch.int32, (batch, rows), (1, 0))
        if paged:
            ring = _fake_paged_state(dtype, d)
            out = _fake_paged_state(cache_dtype, d)
            pos0 = None
            kv_lens = _fake_compact(torch.int32, (batch,), (0,))
            block_table = _fake_i32_table((batch, cute.sym_int()), (cute.sym_int(), 1))
            state_table = _fake_i32_table((batch, 1), (cute.sym_int(), 1))
            slots = _fake_ring_slots(ring_slot_dtype) if ring_slots else None
        else:
            ring = _fake_compact(dtype, (batch, cute.sym_int(), d), (2, 1, 0))
            out = _fake_compact(cache_dtype, (batch, cute.sym_int(), d), (2, 1, 0))
            pos0 = _fake_compact(torch.int32, (batch,), (0,))
            kv_lens = block_table = state_table = slots = None
        step_compiled_cache[key] = cute.compile(
            qsa_compress_step_launch,
            k_new,
            ring,
            out,
            w,
            *tables,
            pos0,
            n_commit,
            slot,
            kv_lens,
            block_table,
            state_table,
            slots,
            cutlass.Float32(1e-6),  # eps       ) runtime values: the literals pin
            cutlass.Float32(0.0),  # w_offset   ) their TYPE at trace time only
            cutlass.Int32(1),  # grid_x
            cutlass.Int32(1),  # grid_y
            int(d),
            int(rope_dim),
            int(pool),
            int(threads_per_cta),
            bool(use_pdl),
            bool(ring_slots and ring_slot_dtype == torch.int64),
            _FAKE_STREAM,
            options="--enable-tvm-ffi",
        )
    return QsaCompressStepRecipe(
        compiled=step_compiled_cache[key],
        d=int(d),
        rope_dim=int(rope_dim),
        pool=int(pool),
        rows_per_cta=threads_per_cta // lanes_per_row(d),
        has_n_commit=bool(has_n_commit),
        dtype=dtype,
        table_dtype=table_dtype,
        paged=bool(paged),
        ring_slots=bool(ring_slots),
        ring_slot_dtype=ring_slot_dtype,
        index_cache_dtype=cache_dtype,
    )


def _check_int32_vector(name: str, x: torch.Tensor, shape: tuple) -> None:
    if not isinstance(x, torch.Tensor) or x.dtype != torch.int32 or tuple(x.shape) != tuple(shape) or not x.is_contiguous() or not x.is_cuda:
        got = None if not isinstance(x, torch.Tensor) else f"{x.dtype} {tuple(x.shape)} contiguous={x.is_contiguous()} cuda={x.is_cuda}"
        raise ValueError(f"{name} must be a contiguous int32 CUDA tensor of shape {tuple(shape)}, got {got}")


def check_paged_state(name: str, x: torch.Tensor, *, dtype, d: int, min_rows: int, what: str, cache: bool = False) -> None:
    """A paged state tensor ``[blocks >= 1, rows >= min_rows, 1, D]`` of ``dtype`` whose rows the kernels move in 16-B vectors (the
    stack's ring ``[num_ring_blocks, cap, 1, D]`` in the io dtype, or the compressed-key cache ``[num_pages, P / pool, 1, D]`` in the
    RECIPE's cache dtype -- ``cache=True``: an e4m3 cache under a 16-bit recipe and a 16-bit cache under an e4m3 recipe are both
    refused, each naming the recipe's dtype).  An e4m3 ring is a ``ValueError`` naming the ring's rule (a bit copy of the step's raw
    keys, never quantized), as is every other dtype mismatch; typed and naming the operand."""
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{name} must be a 4-D [blocks, rows, 1, D={d}] {what}, got {type(x).__name__}")
    if x.dtype == _E4M3 and dtype != _E4M3:
        if cache:
            raise ValueError(
                f"{name} is float8_e4m3fn but this artifact stores the compressed-key cache in {dtype} (the recipe's index_cache_dtype): compile the "
                "e4m3 store arm (index_cache_dtype=torch.float8_e4m3fn) for an e4m3 cache -- no silent cast"
            )
        raise ValueError(
            f"{name} must hold the activation dtype {dtype}: the raw-key ring is a bit copy of this step's raw keys and is never quantized "
            f"(the e4m3 form of the paged arms is the compressed-key cache's: index_cache_dtype), got {x.dtype}"
        )
    if x.dim() != 4 or int(x.shape[2]) != 1 or int(x.shape[3]) != d or int(x.shape[0]) < 1:
        raise ValueError(f"{name} must be a 4-D [blocks >= 1, rows, 1, D={d}] {what}, got shape {tuple(x.shape)}")
    if int(x.shape[1]) < min_rows:
        raise ValueError(f"{name} holds {int(x.shape[1])} rows per block; this launch needs at least {min_rows} ({what})")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor, got device {x.device}")
    _check_row_tensor(name, x, dtype=dtype, last_dim=d)


def check_block_table(name: str, x: torch.Tensor, *, batch: int, cols: Optional[int] = None) -> None:
    """A block table ``[B, cols]`` (``cols`` fixed, or any ``>= 1`` when None) of int32 on the device, columns contiguous, rows at
    any stride (a row slice of a wider table binds); the VALUES are device data the kernels range-check."""
    if not isinstance(x, torch.Tensor) or x.dim() != 2 or x.dtype != torch.int32 or not x.is_cuda:
        got = None if not isinstance(x, torch.Tensor) else f"{x.dtype} {tuple(x.shape)} cuda={x.is_cuda}"
        raise ValueError(f"{name} must be a 2-D int32 CUDA block table [B={batch}, {'cols' if cols is None else cols}], got {got}")
    if int(x.shape[0]) != batch or (cols is None and int(x.shape[1]) < 1) or (cols is not None and int(x.shape[1]) != cols):
        raise ValueError(f"{name} must be [B={batch}, {'cols >= 1' if cols is None else cols}], got shape {tuple(x.shape)}")
    if int(x.stride(1)) != 1:
        raise ValueError(f"{name} must have contiguous columns (stride 1 on the last axis), got strides {tuple(x.stride())}")


def check_ring_slots(name: str, x: torch.Tensor, *, t: int, dtype) -> None:
    """The stack's own ring slot mapping: a contiguous 1-D ``[T]`` tensor of the recipe's ``ring_slot_dtype`` on the device."""
    if not isinstance(x, torch.Tensor) or x.dim() != 1 or int(x.numel()) != t or x.dtype != dtype or not x.is_contiguous() or not x.is_cuda:
        got = None if not isinstance(x, torch.Tensor) else f"{x.dtype} {tuple(x.shape)} contiguous={x.is_contiguous()} cuda={x.is_cuda}"
        raise ValueError(f"{name} must be a contiguous {dtype} CUDA tensor of shape ({t},) (one flat ring row per step row, -1 = no write), got {got}")


def check_paged_step_operands(
    *,
    paged: bool,
    ring_slots: bool,
    ring_slot_dtype,
    batch: int,
    rows: int,
    pos0,
    kv_lens,
    block_table,
    index_state_block_table,
    index_state_slot_mapping,
    ring: torch.Tensor,
) -> None:
    """The position source and the tables against the recipe's form, BOTH directions typed: the paged form takes ``kv_lens`` +
    the two block tables (+ the slot mapping iff traced) and REFUSES ``pos0``; the dense form takes ``pos0`` and REFUSES every
    paged operand (it would be silently ignored).  Shared by the step form and the fused prepare launch."""
    if paged:
        if pos0 is not None:
            raise ValueError("pos0 is refused on the paged form: positions derive in-kernel from kv_lens (the KV length after the step) -- pass pos0=None")
        if kv_lens is None or block_table is None or index_state_block_table is None:
            raise ValueError(
                "this artifact traced the PAGED compress arms: kv_lens [B] int32, block_table [B, max_pages] int32 and index_state_block_table [B, 1] int32 must be bound (no silent dense fallback)"
            )
        _check_int32_vector("kv_lens", kv_lens, (batch,))
        check_block_table("block_table", block_table, batch=batch)
        check_block_table("index_state_block_table", index_state_block_table, batch=batch, cols=1)
        if ring_slots:
            if index_state_slot_mapping is None:
                raise ValueError(
                    "this artifact traced the ring's slot-mapping arm (ring_slots=True); bind index_state_slot_mapping [T] (no silent derived fallback)"
                )
            check_ring_slots("index_state_slot_mapping", index_state_slot_mapping, t=batch * rows, dtype=ring_slot_dtype)
            if int(ring.stride(0)) != int(ring.shape[1]) * int(ring.stride(1)):
                raise ValueError(
                    f"index_state_slot_mapping names flat ring rows (block x cap + pos % cap), which needs the ring's blocks contiguous "
                    f"(stride[0] == cap x stride[1]); got shape {tuple(ring.shape)} strides {tuple(ring.stride())}"
                )
        elif index_state_slot_mapping is not None:
            raise ValueError(
                "this artifact derives the ring rows from index_state_block_table (ring_slots=False); index_state_slot_mapping would be silently ignored -- pass None"
            )
    else:
        extra = [
            nm
            for nm, x in (
                ("kv_lens", kv_lens),
                ("block_table", block_table),
                ("index_state_block_table", index_state_block_table),
                ("index_state_slot_mapping", index_state_slot_mapping),
            )
            if x is not None
        ]
        if extra:
            raise ValueError(
                f"{', '.join(extra)}: this artifact was compiled WITHOUT the paged compress arms (dense ring / cache, positions from pos0) and would silently ignore them -- compile paged=True, or pass None"
            )
        if pos0 is None:
            raise ValueError("the dense form needs pos0 [B] int32 (the position of row 0 of every sequence)")


def run_qsa_compress_step(
    r: QsaCompressStepRecipe,
    k_new: torch.Tensor,
    ring: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos0: Optional[torch.Tensor],
    slot_out: torch.Tensor,
    n_commit: Optional[torch.Tensor] = None,
    *,
    eps: float,
    norm_weight_offset: float = 0.0,
    stream,
    kv_lens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    index_state_block_table: Optional[torch.Tensor] = None,
    index_state_slot_mapping: Optional[torch.Tensor] = None,
) -> None:
    """The lowered decode-step launch: validation and address arithmetic only -- no conversion, no allocation, no device read.

    ``k_new`` ``[B, S_q, D]`` this step's raw keys (any batch / token stride with 16-B aligned rows), ``ring``
    ``[B, RING, D]`` compact with ``RING >= S_q + pool - 1``, ``out`` ``[B, NB, D]`` compact (the compressed-key
    cache in the recipe's ``index_cache_dtype`` -- the io dtype or e4m3, refused both ways; a block past ``NB`` is reported ``-2`` and
    not written), ``w`` ``[D]``, ``cos`` / ``sin`` ``[1 or B, S_pos,
    rope_dim]`` (the table is the ceiling on positions: a committed row at a position at or past ``S_pos`` is reported ``-2`` and
    writes nothing -- size the table for the context), ``pos0`` ``[B]`` int32 (the position of row 0), ``slot_out`` ``[B, S_q]``
    int32 (written for every row), ``n_commit`` ``[B]`` int32 iff the
    recipe traced it (the accepted prefix of the step's rows).  ``norm_weight_offset`` as in the prefill form.

    On a PAGED recipe the SAME positional slots carry the paged state: ``ring`` is the stack's ring ``[blocks >= 1, cap >= S_q + pool - 1,
    1, D]``, ``out`` its compressed-key cache ``[num_pages >= 1, P / pool, 1, D]``, ``pos0`` must be ``None`` (REFUSED: the positions
    derive from ``kv_lens``) and the appended keywords bind the rest: ``kv_lens`` ``[B]`` int32 (the KV length AFTER the step),
    ``block_table`` ``[B, max_pages]`` int32, ``index_state_block_table`` ``[B, 1]`` int32, ``index_state_slot_mapping`` ``[B * S_q]``
    iff the recipe traced ``ring_slots``.  A dense recipe REFUSES every paged keyword (it would be silently ignored).
    """
    if k_new.ndim != 3:
        raise ValueError(f"k_new must be [B, S_q, D], got shape {tuple(k_new.shape)}")
    batch, rows, d = (int(x) for x in k_new.shape)
    if d != r.d:
        raise ValueError(f"k_new head dim {d} does not match the recipe's d={r.d}")
    if rows < 1:
        raise ValueError("k_new must hold at least one row per sequence")
    _check_row_tensor("k_new", k_new, dtype=r.dtype, last_dim=r.d)
    need = min_ring_rows(rows, r.pool)
    if r.paged:
        check_paged_state(
            "ring",
            ring,
            dtype=r.dtype,
            d=r.d,
            min_rows=need,
            what=f"raw-key ring: a step of {rows} rows needs rows + pool - 1 = {rows} + {r.pool} - 1 so this launch's ring writes never alias its ring reads",
        )
        check_paged_state(
            "out",
            out,
            dtype=r.index_cache_dtype if r.index_cache_dtype is not None else r.dtype,
            d=r.d,
            min_rows=1,
            what="compressed-key cache, P / pool rows per attention page",
            cache=True,
        )
    else:
        if ring.ndim != 3 or int(ring.shape[0]) != batch or int(ring.shape[2]) != r.d or not ring.is_contiguous():
            raise ValueError(f"ring must be a compact [B={batch}, RING, D={r.d}] tensor, got shape {tuple(ring.shape)}")
        if int(ring.shape[1]) < need:
            raise ValueError(
                f"ring holds {int(ring.shape[1])} rows per sequence; a step of {rows} rows needs at least {need} (rows + pool - 1 = {rows} + {r.pool} - 1) "
                "so this launch's ring writes never alias its ring reads and no committed key a later step pools is overwritten by a draft"
            )
        _check_row_tensor("ring", ring, dtype=r.dtype, last_dim=r.d)
        if out.ndim != 3 or int(out.shape[0]) != batch or int(out.shape[2]) != r.d or int(out.shape[1]) < 1 or not out.is_contiguous():
            raise ValueError(f"out must be a compact [B={batch}, NB >= 1, D={r.d}] tensor, got shape {tuple(out.shape)}")
        _check_row_tensor("out", out, dtype=r.index_cache_dtype if r.index_cache_dtype is not None else r.dtype, last_dim=r.d)  # the recipe's cache dtype
    if w.ndim != 1 or int(w.shape[0]) != r.d or not w.is_contiguous():
        raise ValueError(f"w must be a contiguous [{r.d}] norm weight, got shape {tuple(w.shape)}")
    _check_row_tensor("w", w, dtype=r.dtype, last_dim=r.d)
    if r.rope_dim:
        for name, t in (("cos", cos), ("sin", sin)):
            if t.ndim != 3:
                raise ValueError(f"{name} must be [1 or B, S_pos, rope_dim], got shape {tuple(t.shape)}")
            if int(t.shape[0]) not in (1, batch):
                raise ValueError(f"{name} batch dim must be 1 (broadcast) or B={batch}, got {int(t.shape[0])}")
            if int(t.shape[1]) < 1:
                raise ValueError(f"{name} must hold at least one position row")
            _check_row_tensor(name, t, dtype=r.table_dtype, last_dim=r.rope_dim)
        if tuple(cos.shape) != tuple(sin.shape):
            raise ValueError(f"cos and sin must share a shape, got {tuple(cos.shape)} and {tuple(sin.shape)}")
    check_paged_step_operands(
        paged=r.paged,
        ring_slots=r.ring_slots,
        ring_slot_dtype=r.ring_slot_dtype,
        batch=batch,
        rows=rows,
        pos0=pos0,
        kv_lens=kv_lens,
        block_table=block_table,
        index_state_block_table=index_state_block_table,
        index_state_slot_mapping=index_state_slot_mapping,
        ring=ring,
    )
    if not r.paged:
        _check_int32_vector("pos0", pos0, (batch,))
    _check_int32_vector("slot_out", slot_out, (batch, rows))
    if r.has_n_commit:
        if n_commit is None:
            raise ValueError("this artifact was compiled WITH an accepted-prefix vector; bind n_commit [B] int32 (no silent commit-everything fallback)")
        _check_int32_vector("n_commit", n_commit, (batch,))
    elif n_commit is not None:
        raise ValueError("this artifact was compiled WITHOUT n_commit (every row of the step is committed); n_commit would be silently ignored -- pass None")
    grid_x = (rows + r.rows_per_cta - 1) // r.rows_per_cta
    r.compiled(
        k_new,
        ring,
        out,
        w,
        cos,
        sin,
        pos0,
        n_commit,
        slot_out,
        kv_lens,
        block_table,
        index_state_block_table,
        index_state_slot_mapping,
        cutlass.Float32(eps),
        cutlass.Float32(norm_weight_offset),
        cutlass.Int32(grid_x),
        cutlass.Int32(batch),
        cuda.CUstream(int(stream)),
    )


def build_qsa_compress_step(
    k_new: torch.Tensor,
    ring: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    pos0: Optional[torch.Tensor],
    slot_out: torch.Tensor,
    n_commit: Optional[torch.Tensor] = None,
    *,
    rope_dim: int,
    eps: float,
    norm_weight_offset: float = 0.0,
    pool: int = DEFAULT_POOL,
    threads_per_cta: int = DEFAULT_THREADS_PER_CTA,
    use_pdl: bool = False,
    stream,
    kv_lens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    index_state_block_table: Optional[torch.Tensor] = None,
    index_state_slot_mapping: Optional[torch.Tensor] = None,
) -> QsaCompressStepRecipe:
    """Compile (cached) and run one decode step -- the convenience form for tests.  Production callers split it:
    :func:`compile_qsa_compress_step` at plan time, :func:`run_qsa_compress_step` per step.  The paged form is traced
    when ``block_table`` is given (its slot-mapping arm when ``index_state_slot_mapping`` is)."""
    if k_new.ndim != 3:
        raise ValueError(f"k_new must be [B, S_q, D], got shape {tuple(k_new.shape)}")
    paged = block_table is not None
    r = compile_qsa_compress_step(
        dtype=k_new.dtype,
        d=int(k_new.shape[2]),
        rope_dim=rope_dim,
        pool=pool,
        has_n_commit=n_commit is not None,
        table_dtype=cos.dtype,
        threads_per_cta=threads_per_cta,
        use_pdl=use_pdl,
        paged=paged,
        ring_slots=paged and index_state_slot_mapping is not None,
        ring_slot_dtype=index_state_slot_mapping.dtype if (paged and index_state_slot_mapping is not None) else torch.int64,
        index_cache_dtype=out.dtype,  # both forms: the dense arm is the e4m3 paged arm's oracle twin
    )
    run_qsa_compress_step(
        r,
        k_new,
        ring,
        out,
        w,
        cos,
        sin,
        pos0,
        slot_out,
        n_commit,
        eps=eps,
        norm_weight_offset=norm_weight_offset,
        stream=stream,
        kv_lens=kv_lens,
        block_table=block_table,
        index_state_block_table=index_state_block_table,
        index_state_slot_mapping=index_state_slot_mapping,
    )
    return r


frost_qsa_compress.set_name_prefix("cudnn", remove_cutlass_symbol=True)
frost_qsa_compress_step.set_name_prefix("cudnn", remove_cutlass_symbol=True)
