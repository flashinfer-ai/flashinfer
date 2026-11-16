# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Permutation kernels (plumbing, no floating point): tokens per expert, the deterministic position of every
(token, slot) inside its expert's segment, and the row gather.

* ``histogram``: ``counts[e] += #{(t, r) : experts[t, r] = e}`` with integer atomics (exact and order-independent; the
  caller zeroes ``counts``);
* ``positions``: one CTA per expert walks the ``T k`` slots in ``(t, r)`` order and assigns ``dest[t k + r] = off_e +
  rank`` where ``rank`` is the number of earlier slots of the same expert (ballot prefix counts: deterministic);
  ``src[dest] = t k + r`` is the inverse map. ``off_e`` is the exclusive prefix of the counts rounded up to ``align``
  rows (``align = 1``: the bf16 class's dense packing; the block-scaled classes pad to their 128-row scale-factor
  atoms). Positions are a placement (the order inside a group never enters the arithmetic), but a deterministic one
  makes every buffer reproducible;
* ``plan``: ``histogram`` + ``positions`` in one launch -- every CTA rebuilds the histogram of the slots in shared memory
  (integer atomics: exact, order-independent), CTA 0 stores it as ``counts``, and each CTA assigns its expert's
  positions as ``positions`` does;
* ``plan_multi``: the same plan for large ``T k`` as two launches of ``ceil(T k / PLAN_BLOCK)`` CTAs -- ``PlanBlocksKernel``
  ranks every slot inside its block among the earlier slots of its expert (``match.any`` groups per 32 slots, per-warp
  running counts, a per-warp prefix inside the block: no atomics at all) and stores the block's histogram;
  ``PlanFinishKernel`` sums the block histograms (the counts, the group tables, the prefix of the earlier blocks) and
  turns the block-local ranks into the positions. The one-launch ``plan`` has every one of its ``E`` CTAs rebuild the
  whole histogram (``E x T k`` shared-memory atomics); the two-launch form reads every slot three times in all;
* ``gather``: ``x_perm[p, :] = x[src[p] // k, :]`` in 16-byte vectors (the bf16 class, dense packing).

An expert id outside ``[0, E)`` (a caller's dropped slot) is dropped: no plan counts it, its ``dest`` (and ``dest_pad``) is
``-1``, ``src`` (and ``pos_pad``) holds ``-1`` past the valid rows and the gather skips those rows.

Every kernel here launches with programmatic dependent launch (``use_pdl``): it waits for its predecessor's grid
(``griddepcontrol.wait``) before touching dependent data and lets its successor start its prologue once its own work
is done (``griddepcontrol.launch_dependents``) -- a scheduling placement, no bits.
"""

from __future__ import annotations

import functools
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Uint32

from ...batch_invariant import jit
from ...batch_invariant.fp_interface.cute import get_cutlass_dtype
from .blocks.scheduler import MAX_E, build_group_tables
from . import pdl

MODULE = "batch_invariant_moe"
THREADS = 1024
CHUNK = 4 * THREADS
WARPS = 4
LANES = 32


class HistogramKernel:
    @cute.jit
    def __call__(
        self, mExp: cute.Tensor, mCounts: cute.Tensor, n: Int32, E: Int32, stream
    ):
        self.kernel(mExp, mCounts, n, E).launch(
            grid=[cute.ceil_div(n, CHUNK), 1, 1],
            block=[THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, mExp: cute.Tensor, mCounts: cute.Tensor, n: Int32, E: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        smem = cutlass.utils.SmemAllocator()
        hist = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E,)), byte_alignment=16
        )
        if tidx < MAX_E:
            hist[tidx] = Int32(0)
        cute.arch.barrier()
        end = cutlass.min(n, (bidx + 1) * CHUNK)
        for i in cutlass.range(bidx * CHUNK + tidx, end, THREADS):
            e = mExp[i]
            if (e >= Int32(0)) & (
                e < E
            ):  # an id outside [0, E) is a dropped slot: not counted
                cute.arch.atomic_add(hist.iterator + e, Int32(1))
        cute.arch.barrier()
        if tidx < E:
            v = hist[tidx]
            if v > 0:
                cute.arch.atomic_add(mCounts.iterator + tidx, v)
        pdl.trigger_one_wave()


class PositionKernel:
    """``align``: the group offsets are the exclusive prefix of the counts rounded up to ``align`` rows."""

    def __init__(self, align: int = 1):
        assert align >= 1
        self.align = align

    @cute.jit
    def __call__(
        self,
        mExp: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        n: Int32,
        E: Int32,
        stream,
    ):
        self.kernel(mExp, mCounts, mDest, mSrc, n, E).launch(
            grid=[E, 1, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        mExp: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        n: Int32,
        E: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        e, _, _ = cute.arch.block_idx()
        warp = tidx // LANES
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        smem = cutlass.utils.SmemAllocator()
        s_off = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
        )
        s_pre = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
        )
        warp_tot = smem.allocate_tensor(
            Int32, cute.make_layout((THREADS // LANES,)), byte_alignment=16
        )
        if warp == 0:
            build_group_tables(mCounts, E, 1, s_off, s_pre, self.align)
        cute.arch.barrier()
        base = s_off[e]
        running = Int32(0)
        n_chunks = cute.ceil_div(n, THREADS)
        for c in cutlass.range(0, n_chunks, 1):
            i = c * THREADS + tidx
            ei = Int32(0)
            if i < n:
                ei = mExp[i]
            match = (i < n) & (ei == e)
            ballot = cute.arch.vote_ballot_sync(match)
            lane_prefix = cute.arch.popc(Uint32(ballot) & cute.arch.lanemask_lt())
            if lane == 0:
                warp_tot[warp] = cute.arch.popc(Uint32(ballot))
            cute.arch.barrier()
            prefix = running + Int32(lane_prefix)
            total = Int32(0)
            for w in cutlass.range_constexpr(THREADS // LANES):
                v = warp_tot[w]
                total = total + v
                if Int32(w) < warp:
                    prefix = prefix + v
            if match:
                mDest[i] = base + prefix
                mSrc[base + prefix] = i
            if (e == Int32(0)) & (i < n) & ((ei < Int32(0)) | (ei >= E)):
                mDest[i] = Int32(
                    -1
                )  # a dropped slot (id outside [0, E)) has no position; CTA 0 marks it
            running = running + total
            cute.arch.barrier()
        if e == Int32(0):
            # the rows past the total: the dropped slots' share of the buffer holds -1, which the gather skips
            for p in cutlass.range(s_off[MAX_E] + tidx, mSrc.shape[0], THREADS):
                mSrc[p] = Int32(-1)
        pdl.trigger_one_wave()


class PlanKernel:
    """``histogram`` + ``positions`` in one launch of ``E`` CTAs: every CTA counts the slots of all experts into a
    shared-memory histogram (integer atomics: exact in every order), CTA 0 stores it as ``counts`` (plain stores: the
    caller need not zero ``counts``), each CTA builds the group tables from its own histogram and assigns the positions
    of its expert ``e`` in slot order ``(t, r)`` -- the same ranks as ``PositionKernel``, in two barrier-free passes:
    warp ``w`` owns the contiguous slot range ``[w n / 32, (w + 1) n / 32)``, counts its expert's slots there by ballots
    (pass 1), the warps' totals are prefix-summed once, and the warp writes the positions of its range (pass 2); the
    ``(t, r)`` order is preserved because the ranges are ordered and each ballot prefix is. (``PositionKernel``'s
    chunk-synchronous scan costs one barrier and 32 shared loads per 1024 slots per CTA.) ``align``: the group offsets'
    alignment of ``dest`` / ``src`` (1 dense); ``dual``: ``dest_pad`` / ``pos_pad`` with the groups padded to
    ``pad_align`` rows on top of the dense ``dest`` / ``src`` (the block-scaled classes' second packing; the bf16 class
    passes ``dual=False`` and any two int32 tensors for the padded outputs)."""

    def __init__(self, align: int = 1, dual: bool = False, pad_align: int = 128):
        assert align >= 1 and pad_align > 1 and not (dual and align > 1)
        self.align, self.dual, self.pad_align = align, dual, pad_align

    @cute.jit
    def __call__(
        self,
        mExp: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        mDestPad: cute.Tensor,
        mPosPad: cute.Tensor,
        n: Int32,
        E: Int32,
        stream,
    ):
        self.kernel(mExp, mCounts, mDest, mSrc, mDestPad, mPosPad, n, E).launch(
            grid=[E, 1, 1], block=[THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        mExp: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        mDestPad: cute.Tensor,
        mPosPad: cute.Tensor,
        n: Int32,
        E: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        e, _, _ = cute.arch.block_idx()
        warp = tidx // LANES
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        smem = cutlass.utils.SmemAllocator()
        hist = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E,)), byte_alignment=16
        )
        s_off = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
        )
        s_pre = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
        )
        s_off_pad = None
        if cutlass.const_expr(self.dual):
            s_off_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
            s_pre_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
        warp_tot = smem.allocate_tensor(
            Int32, cute.make_layout((THREADS // LANES,)), byte_alignment=16
        )
        if tidx < MAX_E:
            hist[tidx] = Int32(0)
        cute.arch.barrier()
        for i in cutlass.range(tidx, n, THREADS):
            ei = mExp[i]
            if (ei >= Int32(0)) & (ei < E):
                cute.arch.atomic_add(hist.iterator + ei, Int32(1))
            if (e == Int32(0)) & ((ei < Int32(0)) | (ei >= E)):
                # a dropped slot (id outside [0, E)): not counted, no position; CTA 0 marks it
                mDest[i] = Int32(-1)
                if cutlass.const_expr(self.dual):
                    mDestPad[i] = Int32(-1)
        cute.arch.barrier()
        if (e == 0) & (tidx < E):
            mCounts[tidx] = hist[tidx]
        if warp == 0:
            build_group_tables(hist, E, 1, s_off, s_pre, self.align)
            if cutlass.const_expr(self.dual):
                build_group_tables(hist, E, 1, s_off_pad, s_pre_pad, self.pad_align)
        # pass 1: this warp's slot range, its count of expert e (ballots over 32 slots at a time)
        n_warps = Int32(THREADS // LANES)
        per_warp = (n + n_warps - Int32(1)) // n_warps
        lo = warp * per_warp
        hi = cutlass.min(lo + per_warp, n)
        local = Int32(0)
        for i0 in cutlass.range(lo, hi, LANES):
            i = i0 + lane
            match = Boolean(False)
            if i < hi:
                match = mExp[i] == e
            ballot = cute.arch.vote_ballot_sync(match)
            local = local + Int32(cute.arch.popc(Uint32(ballot)))
        if lane == 0:
            warp_tot[warp] = local
        cute.arch.barrier()  # the tables and the warp totals are complete
        if e == Int32(0):
            # the rows past the total: the dropped slots' share of the buffer holds -1, which the gather skips
            for p in cutlass.range(s_off[MAX_E] + tidx, mSrc.shape[0], THREADS):
                mSrc[p] = Int32(-1)
                if cutlass.const_expr(self.dual):
                    mPosPad[p] = Int32(-1)
        base = s_off[e]
        base_pad = Int32(0)
        if cutlass.const_expr(self.dual):
            base_pad = s_off_pad[e]
        for w in cutlass.range_constexpr(THREADS // LANES):
            if Int32(w) < warp:
                base = base + warp_tot[w]
                base_pad = base_pad + warp_tot[w]
        # pass 2: the positions of this warp's range in slot order
        running = Int32(0)
        for i0 in cutlass.range(lo, hi, LANES):
            i = i0 + lane
            match = Boolean(False)
            if i < hi:
                match = mExp[i] == e
            ballot = cute.arch.vote_ballot_sync(match)
            prefix = running + Int32(
                cute.arch.popc(Uint32(ballot) & cute.arch.lanemask_lt())
            )
            if match:
                mDest[i] = base + prefix
                mSrc[base + prefix] = i
                if cutlass.const_expr(self.dual):
                    mDestPad[i] = base_pad + prefix
                    mPosPad[base + prefix] = base_pad + prefix
            running = running + Int32(cute.arch.popc(Uint32(ballot)))
        pdl.trigger_one_wave()


PLAN_THREADS = 256  # plan_multi: threads per CTA (8 warps)
PLAN_WARPS = PLAN_THREADS // LANES
PLAN_IDS_PER_THREAD = 8
PLAN_BLOCK = PLAN_THREADS * PLAN_IDS_PER_THREAD  # slots per CTA


class PlanBlocksKernel:
    """``plan_multi``, launch 1 of 2 (``ceil(n / PLAN_BLOCK)`` CTAs): ``dest[i]`` = the rank of slot ``i`` among the earlier
    slots of its expert inside its block (slot order), ``hist_blk[b, e]`` = the block's histogram. Warp ``w`` owns the
    block's slots ``[w 256, (w + 1) 256)`` in 8 chunks of 32: ``match.any`` groups the lanes of one expert, the group's
    rank base is the warp's running count of that expert (shared memory, updated by the group's first lane), the lane's
    offset inside the group its ``popc`` of the lower matching lanes; then the per-warp counts are prefix-summed per
    expert and added. Integer, no atomics, deterministic by construction."""

    @cute.jit
    def __call__(
        self,
        mExp: cute.Tensor,
        mDest: cute.Tensor,
        mHistBlk: cute.Tensor,
        n: Int32,
        E: Int32,
        stream,
    ):
        self.kernel(mExp, mDest, mHistBlk, n, E).launch(
            grid=[cute.ceil_div(n, PLAN_BLOCK), 1, 1],
            block=[PLAN_THREADS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mExp: cute.Tensor,
        mDest: cute.Tensor,
        mHistBlk: cute.Tensor,
        n: Int32,
        E: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        warp = tidx // LANES
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        smem = cutlass.utils.SmemAllocator()
        wcnt = smem.allocate_tensor(
            Int32, cute.make_layout((PLAN_WARPS * MAX_E,)), byte_alignment=16
        )
        for j in cutlass.range(tidx, PLAN_WARPS * MAX_E, PLAN_THREADS):
            wcnt[j] = Int32(0)
        cute.arch.barrier()
        base_w = b * PLAN_BLOCK + warp * (PLAN_IDS_PER_THREAD * LANES)
        es = []
        rs = []
        oks = []
        for c in cutlass.range_constexpr(PLAN_IDS_PER_THREAD):
            i = base_w + c * LANES + lane
            valid = i < n
            e = Int32(-1)
            if valid:
                e = mExp[i]
            ok = (
                valid & (e >= Int32(0)) & (e < E)
            )  # a dropped slot (id outside [0, E)) is ranked by nothing
            m = cute.arch.match_sync(Uint32(0xFFFFFFFF), e)
            intra = Int32(cute.arch.popc(m & cute.arch.lanemask_lt()))
            r = Int32(0)
            if ok:
                r = wcnt[warp * MAX_E + e] + intra
            cute.arch.sync_warp()
            if ok & (intra == Int32(0)):
                wcnt[warp * MAX_E + e] = wcnt[warp * MAX_E + e] + Int32(
                    cute.arch.popc(m)
                )
            cute.arch.sync_warp()
            es.append(e)
            rs.append(r)
            oks.append(ok)
        cute.arch.barrier()
        # per expert: the exclusive prefix over the warps (in place) and the block total
        for e in cutlass.range(tidx, E, PLAN_THREADS):
            run = Int32(0)
            for w in cutlass.range_constexpr(PLAN_WARPS):
                v = wcnt[w * MAX_E + e]
                wcnt[w * MAX_E + e] = run
                run = run + v
            mHistBlk[b * E + e] = run
        cute.arch.barrier()
        for c in cutlass.range_constexpr(PLAN_IDS_PER_THREAD):
            i = base_w + c * LANES + lane
            if i < n:
                d = Int32(-1)  # a dropped slot: no rank, no position
                if oks[c]:
                    d = rs[c] + wcnt[warp * MAX_E + es[c]]
                mDest[i] = d
        pdl.trigger_one_wave()


class PlanFinishKernel:
    """``plan_multi``, launch 2 of 2 (the same CTAs): every CTA sums the block histograms (the expert totals -> the group
    tables; the earlier blocks' totals -> its prefix per expert), CTA 0 stores the totals as ``counts``, and each slot's
    block-local rank becomes its position: ``dest[i] = off_e + prefix_b[e] + rank``, ``src[dest[i]] = i`` (+ the padded
    pair when ``dual``). ``align`` / ``dual`` / ``pad_align`` as ``PlanKernel``."""

    def __init__(self, align: int = 1, dual: bool = False, pad_align: int = 128):
        assert align >= 1 and pad_align > 1 and not (dual and align > 1)
        self.align, self.dual, self.pad_align = align, dual, pad_align

    @cute.jit
    def __call__(
        self,
        mExp: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        mDestPad: cute.Tensor,
        mPosPad: cute.Tensor,
        mHistBlk: cute.Tensor,
        n: Int32,
        E: Int32,
        nb: Int32,
        stream,
    ):
        self.kernel(
            mExp, mCounts, mDest, mSrc, mDestPad, mPosPad, mHistBlk, n, E, nb
        ).launch(
            grid=[nb, 1, 1], block=[PLAN_THREADS, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        mExp: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        mDestPad: cute.Tensor,
        mPosPad: cute.Tensor,
        mHistBlk: cute.Tensor,
        n: Int32,
        E: Int32,
        nb: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        warp = tidx // LANES
        cute.arch.griddepcontrol_wait()
        smem = cutlass.utils.SmemAllocator()
        hist = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E,)), byte_alignment=16
        )
        pre_b = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E,)), byte_alignment=16
        )
        s_off = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
        )
        s_pre = smem.allocate_tensor(
            Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
        )
        s_off_pad = None
        if cutlass.const_expr(self.dual):
            s_off_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
            s_pre_pad = smem.allocate_tensor(
                Int32, cute.make_layout((MAX_E + 1,)), byte_alignment=16
            )
        for e in cutlass.range(tidx, E, PLAN_THREADS):
            tot = Int32(0)
            pre = Int32(0)
            for bb in cutlass.range(0, nb, 1):
                v = mHistBlk[bb * E + e]
                if bb < b:
                    pre = pre + v
                tot = tot + v
            hist[e] = tot
            pre_b[e] = pre
            if b == 0:
                mCounts[e] = tot
        cute.arch.barrier()
        if warp == 0:
            build_group_tables(hist, E, 1, s_off, s_pre, self.align)
            if cutlass.const_expr(self.dual):
                build_group_tables(hist, E, 1, s_off_pad, s_pre_pad, self.pad_align)
        cute.arch.barrier()
        if b == Int32(0):
            # the rows past the total: the dropped slots' share of the buffer holds -1, which the gather skips
            for p in cutlass.range(s_off[MAX_E] + tidx, mSrc.shape[0], PLAN_THREADS):
                mSrc[p] = Int32(-1)
                if cutlass.const_expr(self.dual):
                    mPosPad[p] = Int32(-1)
        for c in cutlass.range_constexpr(PLAN_IDS_PER_THREAD):
            i = b * PLAN_BLOCK + c * PLAN_THREADS + tidx
            if i < n:
                e = mExp[i]
                if (e >= Int32(0)) & (
                    e < E
                ):  # a dropped slot keeps the blocks kernel's -1
                    r = mDest[i]
                    p = s_off[e] + pre_b[e] + r
                    mDest[i] = p
                    mSrc[p] = i
                    if cutlass.const_expr(self.dual):
                        pp = s_off_pad[e] + pre_b[e] + r
                        mDestPad[i] = pp
                        mPosPad[p] = pp
                if cutlass.const_expr(self.dual):
                    if (e < Int32(0)) | (e >= E):
                        mDestPad[i] = Int32(-1)
        pdl.trigger_one_wave()


class PlanMultiKernel:
    """The two launches of ``plan_multi`` from one compiled host function (``hist_blk``: int32 scratch of ``nb x E``)."""

    def __init__(self, align: int = 1, dual: bool = False, pad_align: int = 128):
        self.blocks = PlanBlocksKernel()
        self.finish = PlanFinishKernel(align, dual, pad_align)

    @cute.jit
    def __call__(
        self,
        mExp: cute.Tensor,
        mCounts: cute.Tensor,
        mDest: cute.Tensor,
        mSrc: cute.Tensor,
        mDestPad: cute.Tensor,
        mPosPad: cute.Tensor,
        mHistBlk: cute.Tensor,
        n: Int32,
        E: Int32,
        nb: Int32,
        stream,
    ):
        self.blocks.kernel(mExp, mDest, mHistBlk, n, E).launch(
            grid=[nb, 1, 1], block=[PLAN_THREADS, 1, 1], stream=stream, use_pdl=True
        )
        self.finish.kernel(
            mExp, mCounts, mDest, mSrc, mDestPad, mPosPad, mHistBlk, n, E, nb
        ).launch(
            grid=[nb, 1, 1], block=[PLAN_THREADS, 1, 1], stream=stream, use_pdl=True
        )


# The row copies' placement rule: a warp keeps at least ROW_SLABS_PER_WARP 32-vector slabs of its row (the serial work that
# hides a row's index load and address arithmetic), unless the launch holds fewer than ROW_WARPS_PER_SM_SMALL warps per SM
# (half the SM's resident warps: such a launch is latency-bound, so the row splits down to one slab per warp).
ROW_SLABS_PER_WARP = 4
ROW_WARPS_PER_SM_SMALL = 32


def warps_per_row(n_rows: int, n_vec: int, sms: int) -> int:
    """The row copies' placement at (rows, 16-byte vectors per row): a power of two of warps per row, at most 8 and at most
    one 32-vector slab per warp, doubled while every warp keeps ``ROW_SLABS_PER_WARP`` slabs of the row, or while the launch
    holds fewer than ``ROW_WARPS_PER_SM_SMALL`` warps per SM. ``sms``: the device's SM count."""
    slabs = max(1, -(-n_vec // LANES))
    cap = 1
    while cap * 2 <= min(8, slabs):
        cap *= 2
    w = 1
    while w * 2 <= cap and (
        slabs // (w * 2) >= ROW_SLABS_PER_WARP
        or n_rows * w * 2 <= ROW_WARPS_PER_SM_SMALL * sms
    ):
        w *= 2
    return w


class GatherKernel:
    """``x_perm[p, :] = x[src[p] // k, :]`` in 16-byte vectors (a row is a whole number of them); ``wpr`` warps per row (a
    placement: warp ``w`` of a row owns the 32-vector slabs ``w, w + wpr, ...``)."""

    def __init__(self, k: int, wpr: int = 1):
        assert wpr >= 1 and wpr & (wpr - 1) == 0, wpr
        self.k, self.wpr = k, wpr

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mSrc: cute.Tensor,
        mXp: cute.Tensor,
        Tk: Int32,
        H: Int32,
        stream,
    ):
        self.kernel(mX, mSrc, mXp, Tk, H).launch(
            grid=[cute.ceil_div(Tk * Int32(self.wpr), WARPS), 1, 1],
            block=[WARPS * LANES, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self, mX: cute.Tensor, mSrc: cute.Tensor, mXp: cute.Tensor, Tk: Int32, H: Int32
    ):
        wpr = self.wpr
        vec = 128 // mX.element_type.width  # elements per 16-byte vector
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        cute.arch.griddepcontrol_wait()
        g = bidx * WARPS + tidx // LANES
        p_raw = g // Int32(wpr)
        part = g % Int32(wpr)
        valid = p_raw < Tk
        p = cutlass.min(p_raw, Tk - Int32(1))
        s = mSrc[p]
        valid = valid & (
            s >= Int32(0)
        )  # -1: a row past the valid rows (the dropped slots' share), nothing to gather
        t = cutlass.max(s, Int32(0)) // Int32(self.k)
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mX.element_type, num_bits_per_copy=128
        )
        n_vec = H // Int32(vec)
        frag = cute.make_rmem_tensor((vec,), mX.element_type)
        row_x = mX[
            (t, None)
        ]  # 1-D row slices: a (1, vec) 2-D tile mis-vectorises with a size-1 leading mode
        row_p = mXp[(p, None)]
        n_slabs = (n_vec + Int32(LANES - 1)) // Int32(LANES)
        n_mine = (n_slabs - part + Int32(wpr - 1)) // Int32(wpr)
        for jj in cutlass.range(0, n_mine, 1):
            j = (jj * Int32(wpr) + part) * LANES + lane
            jc = cutlass.min(j, n_vec - Int32(1))
            cute.copy(atom, cute.local_tile(row_x, (vec,), (jc,)), frag)
            if valid & (j < n_vec):
                cute.copy(atom, frag, cute.local_tile(row_p, (vec,), (jc,)))
        pdl.trigger_one_wave()


def plan_kernel_name(kind: str, align: int, dual: bool, pad_align: int) -> str:
    """``moe_plan_a<align>[_dual<pad_align>]`` / ``moe_plan_multi_...``."""
    return jit.sanitize_symbol_name(
        f"moe_{kind}_a{align}" + (f"_dual{pad_align}" if dual else "")
    )


@functools.cache
def get_compiled_histogram() -> Any:
    """``kernel(experts [T k] i32, counts [E] i32 (zeroed by the caller, accumulated), n, E)``."""

    def compile_fn():
        sym_n, sym_e = cute.sym_int(32), cute.sym_int(32)
        e_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            HistogramKernel(),
            e_fake,
            c_fake,
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(MODULE, "moe_histogram", compile_fn)


@functools.cache
def get_compiled_positions(align: int = 1) -> Any:
    """``kernel(experts [T k] i32, counts [E] i32, dest [T k] i32, src [rows] i32, n, E)``."""

    def compile_fn():
        sym_n, sym_e, sym_p = cute.sym_int(32), cute.sym_int(32), cute.sym_int(32)
        e_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        s_fake = cute.runtime.make_fake_tensor(Int32, (sym_p,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            PositionKernel(align),
            e_fake,
            c_fake,
            d_fake,
            s_fake,
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(
        MODULE, "moe_positions" + (f"_a{align}" if align > 1 else ""), compile_fn
    )


@functools.cache
def get_compiled_plan(align: int = 1, dual: bool = False, pad_align: int = 128) -> Any:
    """``kernel(experts [T k] i32, counts [E] i32 (written), dest [T k], src [Tk(_pad)], dest_pad [T k], pos_pad [T k], n, E)``;
    without ``dual`` pass any two int32 tensors for the last two."""

    def compile_fn():
        sym_n, sym_e, sym_p = cute.sym_int(32), cute.sym_int(32), cute.sym_int(32)
        e_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        s_fake = cute.runtime.make_fake_tensor(Int32, (sym_p,), (1,), assumed_align=4)
        dp_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        pp_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            PlanKernel(align, dual, pad_align),
            e_fake,
            c_fake,
            d_fake,
            s_fake,
            dp_fake,
            pp_fake,
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(
        MODULE, plan_kernel_name("plan", align, dual, pad_align), compile_fn
    )


@functools.cache
def get_compiled_plan_multi(
    align: int = 1, dual: bool = False, pad_align: int = 128
) -> Any:
    """``kernel(experts [T k] i32, counts [E] i32 (written), dest [T k], src [Tk(_pad)], dest_pad [T k], pos_pad [T k],
    hist_blk [nb E] i32 (scratch), n, E, nb)`` with ``nb = ceil(n / PLAN_BLOCK)``: the two-launch plan of large batches
    (same outputs as ``get_compiled_plan``)."""

    def compile_fn():
        sym_n, sym_e, sym_p, sym_h = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32),
        )
        e_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        c_fake = cute.runtime.make_fake_tensor(Int32, (sym_e,), (1,), assumed_align=4)
        d_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        s_fake = cute.runtime.make_fake_tensor(Int32, (sym_p,), (1,), assumed_align=4)
        dp_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        pp_fake = cute.runtime.make_fake_tensor(Int32, (sym_n,), (1,), assumed_align=4)
        h_fake = cute.runtime.make_fake_tensor(Int32, (sym_h,), (1,), assumed_align=4)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            PlanMultiKernel(align, dual, pad_align),
            e_fake,
            c_fake,
            d_fake,
            s_fake,
            dp_fake,
            pp_fake,
            h_fake,
            Int32(1),
            Int32(1),
            Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(
        MODULE, plan_kernel_name("plan_multi", align, dual, pad_align), compile_fn
    )


@functools.cache
def get_compiled_gather(k: int, in_fmt: str = "bf16", wpr: int = 1) -> Any:
    """``kernel(x [T, H], src [Tk] i32, x_perm [Tk, H], Tk, H)``, ``H`` a multiple of 16 bytes; ``wpr`` warps per row
    (``warps_per_row``)."""
    dt = get_cutlass_dtype(in_fmt)
    vec = 128 // dt.width

    def compile_fn():
        sym_t, sym_tk, sym_h = (
            cute.sym_int(32),
            cute.sym_int(32),
            cute.sym_int(32, divisibility=vec),
        )
        x_fake = cute.runtime.make_fake_tensor(
            dt, (sym_t, sym_h), (cute.sym_int64(divisibility=vec), 1), assumed_align=16
        )
        s_fake = cute.runtime.make_fake_tensor(Int32, (sym_tk,), (1,), assumed_align=4)
        xp_fake = cute.runtime.make_fake_tensor(
            dt, (sym_tk, sym_h), (cute.sym_int64(divisibility=vec), 1), assumed_align=16
        )
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            GatherKernel(k, wpr),
            x_fake,
            s_fake,
            xp_fake,
            Int32(1),
            Int32(vec),
            stream_fake,
            options="--enable-tvm-ffi",
        )

    return jit.build_and_load(
        MODULE,
        jit.sanitize_symbol_name(
            f"moe_gather_k{k}_{in_fmt}" + (f"_w{wpr}" if wpr > 1 else "")
        ),
        compile_fn,
    )
