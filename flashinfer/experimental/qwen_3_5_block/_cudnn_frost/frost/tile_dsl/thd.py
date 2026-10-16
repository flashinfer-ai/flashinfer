# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""THD / varlen (packed ``[T, H, D]`` + ``cu_seqlens``) device primitives.

Pass- and op-neutral: the metadata buffer, the batch ranking, the unit decode,
the persistent claim counter, and the per-sequence TMA-descriptor patchers.
Everything here is a ``@cute.jit`` macro that emits IR at the call site, so the
CALLER owns its elect / barrier discipline — each docstring names what it wants.

Promoted out of ``cudnn/sdpa/fwd/kernels/thd_sm100.py`` (issue #552's device-side
setup) when the backward needed the same pieces: a second copy of a metadata
LAYOUT is a silent-wrong-answer waiting to happen, since a reader offset and a
writer offset that disagree by one word decode garbage batches rather than
failing.  ``cudnn/linear_attention/frost/common/thd.py`` still carries an
Apache-2.0 sibling of :func:`emit_seq_descs`; collapsing it onto this one is a
cross-license move and therefore a maintainer's call, not a drive-by.

Metadata buffer, in int32 words (``THD_META_WORDS(B)`` of them)::

    [ seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1) | batch_remap(B) | live | ctr ]

``batch_remap`` is a permutation of ``[0, B)`` by DESCENDING Q length, so the
tile scheduler walks the longest sequences first (longest-processing-time): the
tail of a THD launch is then made of short sequences, which is what bounds the
ragged last wave.  ``live`` is the total unit count (device-computed — the host
cannot know ``SUM_b ceil(s_b/tile)*QH`` without a D2H) and ``ctr`` is the
persistent scheduler's claim counter.
"""

from cutlass.base_dsl.typing import Pointer
from cutlass.experimental import primitives as nvvm

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith

# int64 words per 128-byte TMA descriptor.
TENSOR_MAP_QWORDS = 128 // 8
# Byte alignment every tensor-map object must sit on: CUtensorMap is declared `alignas(128)`
TENSOR_MAP_ALIGN = 128
TENSOR_MAP_BIT21 = 1 << 21  # qword 1: encoder's "tensor >= 128 KiB" flag; tensormap.replace does not update it (issue #1013)

THD_META_WORDS = lambda b: 4 * b + 4  # noqa: E731
THD_CU_K_TOTAL_OFF = lambda b: 3 * b + 1  # noqa: E731   cu_k[B]: the live packed kv total (cu_q occupies [B, 2B], cu_k [2B+1, 3B+1])
THD_CU_Q_TOTAL_OFF = lambda b: 2 * b  # noqa: E731   cu_q[B]: the live packed q total (the last word of cu_q; a packed dQ fold's row limit)
THD_REMAP_OFF = lambda b: 3 * b + 2  # noqa: E731
THD_LIVE_OFF = lambda b: 4 * b + 2  # noqa: E731   live unit total (device-computed)
THD_CTR_OFF = lambda b: 4 * b + 3  # noqa: E731    persistent-scheduler claim counter

# --- optional BACKWARD extension -------------------------------------------
# The d512 backward's S/dS workspace is BLOCKED over packed Q tokens: sequence b
# owns ``ceil(s_q[b]/gran)*gran`` rows starting at ``row_off[b]``, blocks packed
# end to end (see write_thd_row_offsets).  Three readers -- stage 2's workspace
# store, stage 3's A operand, and the host's reservation -- must agree on where
# that array lives, so the offsets are defined HERE with the rest of the layout
# rather than in any one of them.  The extension only APPENDS: every forward
# offset above is unchanged, so one buffer serves both passes.
#
#   [ ...the forward layout... | row_off(B+1) ]
THD_ROWOFF_OFF = lambda b: 4 * b + 4  # noqa: E731
THD_BWD_META_WORDS = lambda b: 5 * b + 5  # noqa: E731

# --- optional TENSOR-MAP extension -----------------------------------------
# For a launch ABI with no descriptor operand: the ``B + 3`` tensor maps follow the
# forward layout on the next ``TENSOR_MAP_ALIGN`` boundary of a buffer based on one.
#
#   [ ...the forward layout... | pad | maps((B+3) * TENSOR_MAP_QWORDS int64) ]
THD_MAPS_OFF = lambda b: -(-THD_META_WORDS(b) * 4 // TENSOR_MAP_ALIGN) * TENSOR_MAP_ALIGN // 4  # noqa: E731   int32 words
THD_MAPS_META_WORDS = lambda b: THD_MAPS_OFF(b) + (b + 3) * TENSOR_MAP_QWORDS * 2  # noqa: E731
# The BACKWARD twin: ``n_maps`` tensor maps after the backward layout (``row_off`` included) on the
# next ``TENSOR_MAP_ALIGN`` boundary -- for a backward main kernel whose launch ABI has no descriptor
# operand either (the sm107 d256 backward: five packed-total-clamped input maps + one clipped dV map
# per sequence ride inside its metadata buffer).
#
#   [ ...the backward layout... | pad | maps(n_maps * TENSOR_MAP_QWORDS int64) ]
THD_BWD_MAPS_OFF = lambda b: -(-THD_BWD_META_WORDS(b) * 4 // TENSOR_MAP_ALIGN) * TENSOR_MAP_ALIGN // 4  # noqa: E731   int32 words
THD_BWD_MAPS_META_WORDS = lambda b, n_maps: THD_BWD_MAPS_OFF(b) + n_maps * TENSOR_MAP_QWORDS * 2  # noqa: E731

# Threads for a THD setup launch. Callers write metadata on an elected thread
# or cooperating warps; batch-remap ranking is parallel over batches. Larger
# batches loop over the same block (B > THD_SETUP_THREADS is supported).
THD_SETUP_THREADS = 256


@cute.jit
def exit_if_dead_thd_cluster(meta_t, n_batch: cutlass.Int32, cga_m: int) -> None:
    """All threads of a dead cluster exit before TMEM allocation or barriers.

    The setup launch publishes the live work count. Every CTA in a cluster
    has the same unit id, including role-split clusters, so this predicate is
    cluster-uniform. Call at kernel entry, before acquiring any shared resource.
    Empty initial units must not enter the producer/consumer barrier pipeline.
    """
    meta = cutlass.make_array_view(meta_t)
    uid = cute.arch.block_idx()[0] // cutlass.Int32(cga_m)
    if uid >= cutlass.Int32(meta[THD_LIVE_OFF(n_batch)]):
        nvvm.exit()


@cute.jit
def write_thd_meta(meta, ql, kl, lens_form: cutlass.Int32, n_batch: cutlass.Int32) -> None:
    """Single-thread body of the device-side THD metadata build (issue #552).

    Writes ``[seq_kv_lens(B) | cu_seqlens_q(B+1) | cu_seqlens_k(B+1)]`` from the
    caller's length tensors — ``(B,)`` per-batch lengths (serial cumsum; B is
    small) or the ``(B+1,)`` cu prefix-sum form, per side via ``lens_form``
    (bit 0: Q is cu, bit 1: KV is cu).  cu prefixes are NORMALIZED (element 0
    subtracted): the packed buffers are addressed from token 0, so a cu tensor
    sliced from a larger prefix means the same lengths — and the host can no
    longer validate ``cu[0] == 0`` (Rule 3), so an unnormalized base must not
    leak into the offsets the tiles and the dead-unit sentinel read.

    Callers run this under ``elect_sync``.
    """
    cuq0 = n_batch
    cuk0 = cutlass.Int32(2) * n_batch + cutlass.Int32(1)
    q_is_cu = (lens_form & cutlass.Int32(1)) != cutlass.Int32(0)
    kv_is_cu = (lens_form & cutlass.Int32(2)) != cutlass.Int32(0)
    if q_is_cu:
        base_q = cutlass.Int32(ql[0])
        for b in cutlass.range(0, n_batch + cutlass.Int32(1), 1, unroll=1):
            meta[cuq0 + b] = cutlass.Int32(ql[b]) - base_q
    else:
        acc = cutlass.Int32(0)
        meta[cuq0] = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            acc = acc + cutlass.Int32(ql[b])
            meta[cuq0 + b + cutlass.Int32(1)] = acc
    if kv_is_cu:
        base_k = cutlass.Int32(kl[0])
        meta[cuk0] = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            meta[cuk0 + b + cutlass.Int32(1)] = cutlass.Int32(kl[b + cutlass.Int32(1)]) - base_k
            meta[b] = cutlass.Int32(kl[b + cutlass.Int32(1)]) - cutlass.Int32(kl[b])
    else:
        acc_k = cutlass.Int32(0)
        meta[cuk0] = cutlass.Int32(0)
        for b in cutlass.range(0, n_batch, 1, unroll=1):
            lkv = cutlass.Int32(kl[b])
            meta[b] = lkv
            acc_k = acc_k + lkv
            meta[cuk0 + b + cutlass.Int32(1)] = acc_k


@cute.jit
def write_thd_prefix_warp(
    meta,
    lens,
    n_batch: cutlass.Int32,
    prefix_offset: cutlass.Int32,
    is_cu: cutlass.Boolean,
    lane: cutlass.Int32,
    *,
    store_lengths: cutlass.Constexpr[bool],
    round_to: cutlass.Int32 = 1,
):
    """One FULL warp writes a normalized prefix and optional adjacent lengths.

    All 32 lanes must participate, including the inactive tail of a batch.
    Per-batch lengths use a warp scan with a carry between 32-element chunks;
    cumulative inputs copy adjacent entries after subtracting their first one.
    ``round_to`` pads each length before scanning, for blocked workspace offsets;
    stored lengths remain unrounded. Rounded cumulative inputs scan differences.
    The caller publishes these disjoint writes with a block barrier before
    another warp reads them. Integer arithmetic matches :func:`write_thd_meta`.
    """
    if lane == cutlass.Int32(0):
        meta[prefix_offset] = cutlass.Int32(0)
    if is_cu and round_to == 1:
        base = cutlass.Int32(lens[0])
        for start in cutlass.range(0, n_batch, 32, unroll=1):
            b = start + lane
            if b < n_batch:
                next_value = cutlass.Int32(lens[b + cutlass.Int32(1)])
                meta[prefix_offset + b + cutlass.Int32(1)] = next_value - base
                if cutlass.const_expr(store_lengths):
                    meta[b] = next_value - cutlass.Int32(lens[b])
    else:
        carry = cutlass.Int32(0)
        for start in cutlass.range(0, n_batch, 32, unroll=1):
            b = start + lane
            value = cutlass.Int32(0)
            if b < n_batch:
                if is_cu:
                    value = cutlass.Int32(lens[b + 1]) - cutlass.Int32(lens[b])
                else:
                    value = cutlass.Int32(lens[b])
                if cutlass.const_expr(store_lengths):
                    meta[b] = value
            value = ((value + round_to - 1) // round_to) * round_to
            for shift in cutlass.range_constexpr(5):
                delta = 1 << shift
                prior = cute.arch.shuffle_sync_up(value, offset=delta, mask_and_clamp=0)
                if lane >= delta:
                    value = value + prior
            prefix = carry + value
            if b < n_batch:
                meta[prefix_offset + b + cutlass.Int32(1)] = prefix
            carry = cute.arch.shuffle_sync(prefix, 31)


@cute.jit
def write_thd_row_offsets(meta, n_batch: cutlass.Int32, gran: cutlass.Int32, cu0=None) -> None:
    """Fill ``row_off(B+1)``: where each sequence's block starts in a ragged,
    row-BLOCKED workspace, plus the total at ``row_off[B]``.

    ``row_off[b] = SUM_{i<b} ceil(s[i] / gran) * gran`` over the lengths of the
    ``(B+1,)`` prefix at ``cu0`` inside ``meta`` -- the Q prefix by default (the
    SM100 d512 backward blocks its S/dS workspace over packed Q tokens), the KV
    prefix (``cu0 = 2B+1``) for a consumer whose unit is a kv block (the sm107
    d256 backward's kv-major workspace).  ``row_off`` is over whichever axis
    the consumer blocks.  Rounding each block
    UP is what keeps a sequence's tail tile inside its own block: at an
    unrounded offset the tail would overlap the next sequence's first rows and
    quietly corrupt them.

    **Choosing ``gran`` -- it is bracketed from BOTH sides, and the padding is
    pure waste, so it wants to be the smallest legal value:**

    * **At least the CONSUMER's k-tile.**  The reader walks whole k-tiles, so
      the rows between ``s_q[b]`` and the k-tile boundary must be inside the
      block and must hold the zeros the producer wrote there.  Any smaller
      ``gran`` and the reader's last k-tile reaches into the NEXT sequence's
      live rows -- nonzero data, silently summed into the wrong gradient.
    * **At least the PRODUCER's per-CTA store box, if you want to stay
      descriptor-free.**  With ``gran`` == that box height, every box is wholly
      inside its block or wholly outside it, so an out-of-range box is a
      boolean SKIP at the store site.  Below it a box STRADDLES the boundary,
      which no predicate can express -- it needs a per-sequence descriptor with
      a clipped ``GLOBAL_DIM``, i.e. a GMEM descriptor read plus a
      GENERIC->TENSORMAP acquire fence on every store.  That is real hot-path
      cost to save (box - k_tile) rows per sequence.

    For the SM100 d512 backward that is ``gran = TILE_M = 128`` (k-tile 64,
    per-CTA box 128).  The cluster spans ``TILE_M * CTA_MMA = 256`` rows, but
    each CTA stores its own 128 -- padding to 256 would double the waste for
    nothing.  At B=64 short sequences that is 20 % of the workspace rather
    than 33 %.

    Must run AFTER :func:`write_thd_meta` (it reads the ``cu_seqlens_q`` it
    wrote); callers run it on the same elected thread, where program order
    suffices.
    """
    cuq0 = n_batch if cutlass.const_expr(cu0 is None) else cu0
    off0 = cutlass.Int32(4) * n_batch + cutlass.Int32(4)
    run = cutlass.Int32(0)
    for b in cutlass.range(0, n_batch, 1, unroll=1):
        meta[off0 + b] = run
        s_b = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + b])
        run = run + ((s_b + gran - cutlass.Int32(1)) // gran) * gran
    meta[off0 + n_batch] = run


@cute.jit
def write_thd_batch_remap(meta, n_batch: cutlass.Int32, tid: cutlass.Int32, nthreads: cutlass.Int32) -> None:
    """Fill ``batch_remap`` with ``[0, B)`` sorted by descending Q length.

    Rank-by-counting rather than a sort network: each thread owns a batch and
    counts how many sequences outrank it, which is O(B^2) comparisons but fully
    parallel, branch-free and trivially deterministic.  Ties break on the
    original index, so the permutation is stable and reproducible run to run.

    WHOLE-BLOCK (not elected).  Must be called AFTER :func:`write_thd_meta`
    (it reads the ``cu_seqlens_q`` it wrote) with a barrier in between.
    """
    cuq0 = n_batch
    remap0 = cutlass.Int32(3) * n_batch + cutlass.Int32(2)
    i = tid
    while i < n_batch:
        len_i = cutlass.Int32(meta[cuq0 + i + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + i])
        rank = cutlass.Int32(0)
        for j in cutlass.range(0, n_batch, 1, unroll=1):
            len_j = cutlass.Int32(meta[cuq0 + j + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + j])
            # Descending by length; ties resolved by the lower original index.
            outranks = (len_j > len_i) | ((len_j == len_i) & (j < i))
            rank = rank + cutlass.Int32(arith.select(outranks.ir_value(), cutlass.Int32(1).ir_value(), cutlass.Int32(0).ir_value()))
        meta[remap0 + rank] = i
        i = i + nthreads


@cute.jit
def write_thd_live_and_ctr(
    meta,
    n_batch: cutlass.Int32,
    n_qh: cutlass.Int32,
    unit_rows: cutlass.Int32,
    n_ctas: cutlass.Int32,
    tidx: cutlass.Int32,
    splits: cutlass.Constexpr[int] = 1,
    cu0=None,
) -> None:
    """Publish the live-unit total and seed the persistent claim counter.

    A unit is ``unit_rows`` rows of one head of one sequence along the prefix
    at ``cu0`` -- Q rows by default (``cu0 = B``, the forward and the SM100 d512
    backward), kv rows for a consumer whose unit is a kv block (``cu0 = 2B+1``,
    the sm107 d256 backward) -- so ``live = SUM_b ceil(s[b] / unit_rows) * n_qh``,
    which the host cannot know without a D2H (issue #552), hence the kernel
    reading its own bound from here.  The counter starts at ``n_ctas``: cluster
    ``c`` takes unit ``c`` from its blockIdx, then pulls from the counter.

    Leaving these two words unwritten hands out units off uninitialized
    workspace — an illegal-instruction fault, not a silent wrong answer.

    WHOLE-BLOCK (not elected): warp 0 reduces strided batches in parallel;
    B <= 1 keeps the single-thread path. The publisher is ``tidx == 0``,
    deliberately WITHOUT ``elect_sync``: its implementation-defined lane can
    disagree with ``tidx == 0`` and leave both words unwritten. The caller must
    have barriered after the prefix writes so ``cu_seqlens_q`` is visible.
    """
    cuq0 = n_batch if cutlass.const_expr(cu0 is None) else cu0
    if n_batch <= cutlass.Int32(1):
        if tidx == cutlass.Int32(0):
            live = cutlass.Int32(0)
            if n_batch == cutlass.Int32(1):
                s_b = cutlass.Int32(meta[cuq0 + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0])
                live = ((s_b + unit_rows - cutlass.Int32(1)) // unit_rows) * n_qh
            meta[cutlass.Int32(4) * n_batch + cutlass.Int32(2)] = live * cutlass.Int32(splits)
            meta[cutlass.Int32(4) * n_batch + cutlass.Int32(3)] = n_ctas
    elif tidx < cutlass.Int32(32):
        live = cutlass.Int32(0)
        for b in cutlass.range(tidx, n_batch, 32, unroll=1):
            s_b = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + b])
            live = live + ((s_b + unit_rows - cutlass.Int32(1)) // unit_rows) * n_qh
        for i in cutlass.range_constexpr(5):
            live = live + cute.arch.shuffle_sync_bfly(live, 1 << i)
        if tidx == cutlass.Int32(0):
            meta[cutlass.Int32(4) * n_batch + cutlass.Int32(2)] = live * cutlass.Int32(splits)
            meta[cutlass.Int32(4) * n_batch + cutlass.Int32(3)] = n_ctas


@cute.jit
def thd_decode_unit(
    meta,
    n_batch: cutlass.Int32,
    uid: cutlass.Int32,
    n_qh: cutlass.Int32,
    q_tile: cutlass.Int32,
    reverse_rows: bool,
    cu0=None,
) -> tuple:
    """Map a linear unit id to ``(q_tile_idx, batch, head)`` through ``batch_remap``.

    A unit is ``q_tile`` rows of one head of one sequence along the prefix at
    ``cu0`` -- the Q prefix by default; the KV prefix (``cu0 = 2B+1``) for a
    consumer whose unit is a kv block, where the returned tile index is the
    sequence-local kv block (the sm107 d256 backward).  The count per sequence
    must be the one :func:`write_thd_live_and_ctr` published from the SAME prefix.
    Sequences are walked LONGEST FIRST (the remap, by descending Q length: a kv
    block's cost is its sequence's q-tile count, so that is the LPT order for kv
    units too), and the head is the major axis within a sequence so consecutive
    units sweep the tiles of a single head — those share a K/V head, which is
    what keeps the claim order L2-friendly.  ``reverse_rows`` walks a sequence's
    tiles from the diagonal back, putting the causal-heavy tiles first.

    A uid past the live total keeps ``batch == n_batch``; the caller is expected
    to bound uid against the live count instead of relying on that sentinel.
    """
    cuq0 = n_batch if cutlass.const_expr(cu0 is None) else cu0
    remap0 = cutlass.Int32(3) * n_batch + cutlass.Int32(2)
    f_batch = n_batch
    f_head = cutlass.Int32(0)
    f_qt = cutlass.Int32(0)
    done = cutlass.Int32(0)
    acc = cutlass.Int32(0)
    for i in cutlass.range(0, n_batch, 1, unroll=1):
        b = cutlass.Int32(meta[remap0 + i])
        s_i = cutlass.Int32(meta[cuq0 + b + cutlass.Int32(1)]) - cutlass.Int32(meta[cuq0 + b])
        tb = (s_i + q_tile - cutlass.Int32(1)) // q_tile
        # A zero-length sequence contributes no unit; keep the divisor legal
        # anyway, since both quotients below are evaluated before the select.
        tb_nz = cute.math.max(tb, cutlass.Int32(1))
        units_b = tb * n_qh
        in_rng = (done == cutlass.Int32(0)) & (uid < acc + units_b)
        local = uid - acc
        qt = local % tb_nz
        if cutlass.const_expr(reverse_rows):
            qt = tb - cutlass.Int32(1) - qt
        f_batch = cutlass.Int32(arith.select(in_rng.ir_value(), b.ir_value(), f_batch.ir_value()))
        f_head = cutlass.Int32(arith.select(in_rng.ir_value(), (local // tb_nz).ir_value(), f_head.ir_value()))
        f_qt = cutlass.Int32(arith.select(in_rng.ir_value(), qt.ir_value(), f_qt.ir_value()))
        done = cutlass.Int32(arith.select(in_rng.ir_value(), cutlass.Int32(1).ir_value(), done.ir_value()))
        acc = acc + units_b
    return f_qt, f_batch, f_head


@cute.jit
def thd_claim_next(meta_t: cute.Tensor, ctr_off: cutlass.Int32, slot, tidx: cutlass.Int32) -> cutlass.Int32:
    """Take the next unit from the device-side claim counter.

    One atomic for the whole CTA, broadcast through a single SMEM word.  The
    leading barrier also separates the previous unit's use of the shared K/V
    staging from the next unit's, so the caller does not need its own.
    """
    ctr_ptr = Pointer(meta_t.iterator.raw_ptr(), dtype=cutlass.Int32) + ctr_off
    nvvm.barrier_cta_sync(0)
    if tidx == cutlass.Int32(0):
        slot[0] = cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, ctr_ptr, cutlass.Int32(1)))
    nvvm.barrier_cta_sync(0)
    return cutlass.Int32(slot[0])


# ---------------------------------------------------------------------------
# Per-sequence TMA descriptor arrays
#
# A packed output tensor cannot be reached by a batch COORDINATE (the row base
# cu[b] is irregular), and the last tile of a sequence overshoots into the NEXT
# sequence's rows with a live accumulator behind it.  Both are solved by giving
# each sequence its own descriptor: GLOBAL_ADDRESS at that sequence's first row,
# GLOBAL_DIM[seq] at its length, so the overshoot is TMA-clipped in hardware —
# dense packing with no predicated epilogue.
# ---------------------------------------------------------------------------


@cute.jit
def set_tensor_map_bit21(dptr, new_bytes: cutlass.Int64) -> None:
    """Recompute bit 21 from the patched extent (as cuTensorMapEncodeTiled would)."""
    w = (dptr + 1).load() & cutlass.Int64(~TENSOR_MAP_BIT21)
    (dptr + 1).store((w | cutlass.Int64(TENSOR_MAP_BIT21)) if new_bytes >= cutlass.Int64(128 << 10) else w)


@cute.jit
def emit_seq_descs(
    base_desc,
    desc_words,
    cu,
    cu0: cutlass.Int32,
    base_ptr,
    n_batch: cutlass.Int32,
    row_stride: cutlass.Int64,
    seq_ord: cutlass.Constexpr[int],
    slot_base=0,
    first_batch=0,
    batch_step=1,
) -> None:
    """Build a per-BATCH descriptor array over a packed ``[T, H, D]`` tensor.

    ``cu`` is an int32 array VIEW and ``cu0`` the offset of the relevant
    ``(B+1,)`` prefix inside it — so the same helper serves a standalone
    cu_seqlens tensor (``cu0 = 0``) and a prefix living inside the THD metadata
    buffer (``cu0 = THD cu_q / cu_k offset``).  ``row_stride`` is in ELEMENTS of
    the tensor's dtype (``.raw_ptr()`` is element-addressed). Keep it Int64
    through the caller and setup-kernel ABI: widening after an Int32 cast
    cannot recover a legal strided output's high bits.  ``seq_ord`` is
    innermost-first, so for ``[1, T, H, D]`` with D contiguous the sequence axis
    is **2**, not 1.  ``slot_base`` (a RUNTIME value -- the arrays are B slots long and B is not
    a compile-time constant) lets several arrays share one buffer.

    Each participating warp elects one writer for batches
    ``first_batch, first_batch + batch_step, ...``. These ranges must be
    disjoint; defaults preserve the single-writer contract. The caller must
    issue a ``GENERIC -> TENSORMAP`` release fence on every writer afterwards.
    """
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    base = base_ptr.iterator.raw_ptr()
    for b in cutlass.range(first_batch, n_batch, batch_step, unroll=1):
        cu_b = cutlass.Int32(cu[cu0 + b])
        s_b = cutlass.Int32(cu[cu0 + b + cutlass.Int32(1)]) - cu_b
        dptr = desc_base + (b + cutlass.Int32(slot_base)) * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        # Int64 fold: the product is in ELEMENTS, and a packed tensor can carry
        # more than 2^31 of them (128k tokens x 128 heads x 128 lanes already
        # does), so the Int32 form the inlined SDPA-forward version used was a
        # latent overflow. Matches the linear-attention sibling.
        row_base = base + cutlass.Int64(cu_b) * cutlass.Int64(row_stride)
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_ADDRESS,
            dptr,
            new_value=row_base.toint(cutlass.Int64),
        )
        # Clamped to >= 1: a tensor map with a ZERO extent is not merely empty,
        # it is INVALID, and any access through it raises
        # cudaErrorIllegalInstruction -- so a zero-length sequence cannot be
        # left to the hardware clip.  The extent-1 descriptor is structurally
        # valid and never dereferenced: a consumer that can reach an empty
        # sequence's tiles must skip the access itself (stage 3's epilogue
        # does; the forward's scheduler hands out no unit for one).
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_DIM,
            dptr,
            new_value=cute.math.max(s_b, cutlass.Int32(1)),
            ord=seq_ord,
        )
        set_tensor_map_bit21(
            dptr, cutlass.Int64(cute.math.max(s_b, cutlass.Int32(1))) * cutlass.Int64(row_stride) * cutlass.Int64(base_ptr.element_type.width // 8)
        )


@cute.jit
def emit_clamped_desc(
    base_desc,
    desc_words,
    slot: cutlass.Int32,
    extent: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
) -> None:
    """Copy a base descriptor into ``slot`` with its sequence extent clamped.

    Issue #624: a THD caller binds K/V (and, in the backward, Q/dO) at buffer
    CAPACITY, not at the packed total, so a tile-tail load steps into
    caller-owned bytes that may never have been written.  Masked score columns
    are NaN-safe (the mask is a select) but the MMA is not — ``0 * NaN == NaN``
    wipes every valid row of the tile.  Clamping GLOBAL_DIM[seq] to the packed
    total makes those rows TMA-OOB, so they land as EXACT ZEROS without touching
    memory: no fill kernel, and nothing written into the caller's buffer.

    Runs on ONE elected thread; the caller elects and fences.
    """
    dptr = desc_words.iterator.raw_ptr() + slot * cutlass.Int32(TENSOR_MAP_QWORDS)
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
        (dptr + i).store((src_words + i).load())
    nvvm.tensormap_replace(nvvm.TensormapField.GLOBAL_DIM, dptr, new_value=extent, ord=seq_ord)


# --- caller-buffer token origins (issue #737) -------------------------------
# A ragged port's sequence b starts at element ``ro[b] * M`` of its buffer (M
# the port's ragged_offset_multiplier).  FROST engines take whole-token offsets
# as a supported-input precondition, so that is token ``ro[b] * M // ts`` (ts
# the port's token stride in elements).  Origins are a separate Int64
# ``[rows, B]`` array, one row per port a path materializes; they never replace
# the compact cu_seqlens above, which keep addressing internal packed buffers.


@cute.jit
def write_thd_port_origins(
    org,
    ro,
    row: cutlass.Constexpr[int],
    mult: cutlass.Constexpr[int],
    ts: cutlass.Constexpr[int],
    n_batch: cutlass.Int32,
    tid: cutlass.Int32,
    nthreads: cutlass.Int32,
) -> None:
    """``org[row, b] = ro[b] * mult // ts`` for this thread's batches (strided by
    ``nthreads``).  ``org`` is the Int64 origins tensor; ``ro`` the bound offset
    tensor (Int32 or Int64).  Arithmetic is Int64 throughout."""
    org_p = cutlass.make_array_view(org).data_ptr() + cutlass.Int64(row) * cutlass.Int64(org.stride[0])
    ro_p = cutlass.make_array_view(ro).data_ptr()
    for b in cutlass.range(tid, n_batch, nthreads, unroll=1):
        v = cutlass.Int64(Pointer(ro_p + b, dtype=ro.element_type).load())
        Pointer(org_p + b, dtype=cutlass.Int64).store((v * cutlass.Int64(mult)) // cutlass.Int64(ts))


@cute.jit
def thd_port_origin(org, row: cutlass.Constexpr[int], b: cutlass.Int32, compact: cutlass.Int32) -> cutlass.Int64:
    """Token origin of sequence ``b`` in a port's buffer: the materialized origin
    when ``row >= 0``, else the compact prefix ``compact`` (a port bound without
    offsets, or an internal buffer; ``org`` is then never read)."""
    origin = cutlass.Int64(compact)
    if cutlass.const_expr(row >= 0):
        origin = Pointer(cutlass.make_array_view(org).data_ptr() + cutlass.Int64(row) * cutlass.Int64(org.stride[0]) + b, dtype=cutlass.Int64).load()
    return origin


__all__ = [
    "TENSOR_MAP_ALIGN",
    "TENSOR_MAP_QWORDS",
    "THD_BWD_MAPS_META_WORDS",
    "THD_BWD_MAPS_OFF",
    "THD_BWD_META_WORDS",
    "THD_CTR_OFF",
    "THD_LIVE_OFF",
    "THD_MAPS_META_WORDS",
    "THD_MAPS_OFF",
    "THD_META_WORDS",
    "THD_REMAP_OFF",
    "THD_ROWOFF_OFF",
    "THD_SETUP_THREADS",
    "emit_clamped_desc",
    "emit_seq_descs",
    "exit_if_dead_thd_cluster",
    "set_tensor_map_bit21",
    "thd_claim_next",
    "thd_decode_unit",
    "thd_port_origin",
    "write_thd_batch_remap",
    "write_thd_live_and_ctr",
    "write_thd_meta",
    "write_thd_prefix_warp",
    "write_thd_port_origins",
    "write_thd_row_offsets",
]
