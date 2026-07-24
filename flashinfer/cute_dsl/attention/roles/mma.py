# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MmaOps — QK/PV GEMM primitives and orchestration for attention kernels.

Reusable primitives (pipeline-unaware, for composing new kernel variants):
- gemm_qk(): single QK GEMM with kphase unrolling
- gemm_pv(): single PV GEMM with configurable accumulation
- alloc_tmem(): TMEM allocation with barrier sync
- dealloc_tmem(): TMEM deallocation after barrier wait

Orchestration (prefill-specific, uses raw CuTe ops for JIT compatibility):
- run(): double-buffered interleaved QK/PV with S0/S1 and O0/O1

Inner kphase loops in ``gemm_qk`` / ``gemm_pv`` use ``cutlass.range_constexpr``
(compile-time unrolled) for maximum tcgen05 MMA dispatch throughput.  Each
helper constructs its own local ``TiledMma`` via ``make_trivial_tiled_mma``
so the unrolled ``.set(ACCUMULATE, ...)`` mutations stay inside the helper's
frame and never leak SSA values across the persistent tile-scheduler
``while`` loop in ``run()``.  Same isolation pattern the MLA decode MMA
roles use for the same reason.
"""

from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Float32

from cutlass.pipeline import PipelineProducer, PipelineConsumer

from ..config import AttentionConfig
from ..fusion.mask import get_peel_sections, get_trip_count
from ..scheduler.persistent import (
    FmhaStaticTileScheduler,
    FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler,
)


class MmaRole:
    """MMA warp for attention kernels — computes QK and PV GEMMs.

    Created from AttentionConfig in the kernel's __init__.
    """

    def __init__(
        self,
        config: AttentionConfig,
        tmem_alloc_cols,
        tmem_alloc_sync_bar_id,
        threads_per_warp,
        has_logits_transform: bool = False,
        has_statistics_update: bool = False,
    ):
        self.cta_tiler = config.cta_tiler
        self.mask_spec = config.mask_spec
        self.tmem_alloc_cols = tmem_alloc_cols
        self.tmem_alloc_sync_bar_id = tmem_alloc_sync_bar_id
        self.threads_per_warp = threads_per_warp
        self.has_logits_transform = has_logits_transform
        self.has_statistics_update = has_statistics_update
        # Used to (re)construct local TiledMma instances inside gemm_qk /
        # gemm_pv so the helper's .set(ACCUMULATE, ...) mutations don't
        # leak SSA values into the caller's persistent loop.
        self.qk_acc_dtype = config.qk_acc_dtype
        self.pv_acc_dtype = config.pv_acc_dtype
        self.qk_mma_tiler = config.qk_mma_tiler
        self.pv_mma_tiler = config.pv_mma_tiler
        # self.q_dtype / self.v_dtype / self.q_major_mode / self.k_major_mode /
        # self.v_major_mode are populated by set_dtypes() — they're only known
        # at __call__ time on the kernel.

    def set_dtypes(
        self,
        q_dtype: Type[cutlass.Numeric],
        v_dtype: Type[cutlass.Numeric],
        q_major_mode: cute.nvgpu.OperandMajorMode,
        k_major_mode: cute.nvgpu.OperandMajorMode,
        v_major_mode: cute.nvgpu.OperandMajorMode,
    ) -> None:
        """Set tensor element types and operand major modes discovered at call time.

        Required so the GEMM helpers can reconstruct local TiledMma instances
        via ``make_trivial_tiled_mma``.
        """
        self.q_dtype: Type[cutlass.Numeric] = q_dtype
        self.v_dtype: Type[cutlass.Numeric] = v_dtype
        self.q_major_mode: cute.nvgpu.OperandMajorMode = q_major_mode
        self.k_major_mode: cute.nvgpu.OperandMajorMode = k_major_mode
        self.v_major_mode: cute.nvgpu.OperandMajorMode = v_major_mode

    @cute.jit
    def _make_local_qk_mma(self) -> cute.TiledMma:
        """Fresh QK TiledMma — mutations on this instance never escape the
        helper that constructs it, so the inner kphase loop can use
        ``range_constexpr`` without leaking SSA values into the enclosing
        persistent ``while`` loop in ``run()``."""
        return sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.qk_mma_tiler[:2],
        )

    @cute.jit
    def _make_local_pv_mma(self) -> cute.TiledMma:
        """Fresh PV TiledMma — same isolation rationale as ``_make_local_qk_mma``.

        P operand source is TMEM (P comes from the QK accumulator), matching
        ``build_fmha_launch_params``.
        """
        return sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            cute.nvgpu.OperandMajorMode.K,
            self.v_major_mode,
            self.pv_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.pv_mma_tiler[:2],
            tcgen05.OperandSource.TMEM,
        )

    # =========================================================================
    #  Reusable primitives — no pipeline awareness, for composing new kernels
    #
    #  All primitives below are SAFE to call from run() and other @cute.jit
    #  methods. They are void (no return values) and only use compile-time
    #  indexing (unrolled kphase loops), avoiding the CuTe DSL JIT
    #  limitations with runtime tensor views and return values.
    # =========================================================================

    @cute.jit
    def gemm_qk(
        self,
        tStS: cute.Tensor,
        tSrQ_slice: cute.Tensor,
        tSrK_slice: cute.Tensor,
    ):
        """Single QK GEMM: S += Q * K^T with kphase unrolling.

        Always starts a fresh accumulation (first kphase non-accumulate).
        Constructs and mutates a fresh local TiledMma so the unrolled
        ``.set(ACCUMULATE, ...)`` chain dies inside this helper's frame.
        """
        local_mma = self._make_local_qk_mma()
        num_kphases = cute.size(tSrQ_slice, mode=[2])
        for kphase_idx in cutlass.range_constexpr(num_kphases):
            coord = (None, None, kphase_idx)
            local_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(local_mma, tStS, tSrQ_slice[coord], tSrK_slice[coord], tStS)

    @cute.jit
    def gemm_pv(
        self,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        tOrV_slice: cute.Tensor,
        accumulate: bool,
    ):
        """Single PV GEMM: O += P * V with kphase unrolling.

        Args:
            accumulate: If False, first kphase starts fresh (non-accumulate),
                       rest accumulate. If True, all kphases accumulate.

        Constructs a fresh local TiledMma — see ``gemm_qk`` for rationale.
        """
        local_mma = self._make_local_pv_mma()
        num_kphases = cute.size(tOrP, mode=[2])
        for kphase_idx in cutlass.range_constexpr(num_kphases):
            coord = (None, None, kphase_idx)
            local_mma.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0 or accumulate)
            cute.gemm(local_mma, tOtO, tOrP[coord], tOrV_slice[coord], tOtO)

    @cute.jit
    def alloc_tmem(self, storage: cute.Tensor):
        """Allocate TMEM buffer and synchronize."""
        tmem_alloc_cols = Int32(self.tmem_alloc_cols)
        cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf.ptr)
        cute.arch.barrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            number_of_threads=self.threads_per_warp,
        )

    @cute.jit
    def dealloc_tmem(self, storage: cute.Tensor, tmem_dealloc_mbar_ptr: Int32):
        """Wait for all warps, then deallocate TMEM buffer."""
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
        tmem_alloc_cols = Int32(self.tmem_alloc_cols)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            Float32,
            alignment=16,
            ptr_to_buffer_holding_addr=storage.tmem_holding_buf.ptr,
        )
        cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

    # =========================================================================
    #  Prefill orchestration — uses primitives for GEMMs and TMEM lifecycle
    # =========================================================================

    @cute.jit
    def run(
        self,
        pv_tiled_mma: cute.TiledMma,
        tStS0: cute.Tensor,
        tStS1: cute.Tensor,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tOrP0: cute.Tensor,
        tOrP1: cute.Tensor,
        tOrV: cute.Tensor,
        seqlen_q_global: Int32,
        seqlen_k_global: Int32,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        window_left: Int32,
        window_right: Int32,
        load_q_consumer: PipelineConsumer,
        load_kv_consumer: PipelineConsumer,
        mma_s0_producer: PipelineProducer,
        mma_s1_producer: PipelineProducer,
        mma_corr_producer: PipelineProducer | None,
        tile_sched_params: FmhaStaticTileSchedulerParams,
        storage: cute.Tensor,
        tmem_dealloc_mbar_ptr: Int32,
    ):
        """MMA warp orchestration loop (prefill-specific).

        Double-buffered interleaved QK/PV GEMMs with pipeline synchronization.

        For has_logits_transform variants, mma_corr_producer is None. PV GEMM
        results piggyback on subsequent QK tcgen05.commit() calls to mma_s0/s1,
        which makes all prior TMEM writes visible to softmax warps.
        """
        # Alloc tmem buffer
        self.alloc_tmem(storage)
        # Tracks O0's "first PV0 overwrites, rest accumulate" state, mirroring
        # pv_tiled_mma's role for O1 (see the comment at the loop below).  A
        # separate instance because both flags are live at once.
        pv0_tiled_mma = self._make_local_pv_mma()
        tile_sched = create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            seqlen_q_ = seqlen_q_global
            continue_cond = False
            if cutlass.const_expr(cum_seqlen_q is not None):
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q_ = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q_,
                    )
                )

            if not continue_cond:
                seqlen_k = seqlen_k_global
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                # Head / main / tail split (get_peel_sections): banded masks
                # give the two 128-row halves KV bands offset by one block at
                # each bound, so the leading `head` blocks are stage-0-only
                # and the trailing `tail` blocks are stage-1-only.  main >= 1
                # always (the helper borrows a head block if the bands don't
                # overlap), so the main prologue/epilogue stay straight-line.
                # Variants with per-step side effects keep full union
                # lockstep (head = tail = 0).
                if cutlass.const_expr(
                    not self.has_logits_transform and not self.has_statistics_update
                ):
                    _, head_cnt, main_cnt, tail_cnt, _ = get_peel_sections(
                        self.mask_spec,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_k,
                        seqlen_q_,
                        window_left,
                        window_right,
                    )
                else:
                    head_cnt = Int32(0)
                    tail_cnt = Int32(0)
                    main_cnt = get_trip_count(
                        self.mask_spec,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_k,
                        seqlen_q_,
                        window_left,
                        window_right,
                    )

                q0_handle_consumer = load_q_consumer.wait_and_advance()
                tSrQ0 = tSrQ[None, None, None, q0_handle_consumer.index]
                # Track each O tile's "first PV overwrites, then accumulate"
                # state on a TiledMma ACCUMULATE field rather than a Python
                # bool.  Stored on the TiledMma type, the cute-DSL JIT
                # propagates the value so the `accumulate or kphase_idx > 0`
                # expression inside gemm_pv folds per issue site; a Python
                # bool would be demoted to a runtime register through the kv
                # loops.  pv0_tiled_mma tracks O0 (its first PV0 lands in the
                # head loop or the main prologue depending on runtime head
                # count); pv_tiled_mma tracks O1 as before.
                pv0_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                s0_handle_producer = mma_s0_producer.acquire_and_advance()

                # HEAD: stage-0-only trips.  Per trip: QK0 commits the
                # previously acquired S0 slot; the re-acquire doubles as the
                # P-ready wait for this trip's PV0 (same slot discipline as
                # the main loop).  K/V have no stage-1 consumer here.
                #
                # The empty o1 commit keeps the shared mma_corr ring strictly
                # o0/o1-alternating: with the ring's depth of 2, PV_X(t)'s
                # acquire then waits on correction's release of oX(t-1),
                # which correction performs at rescale(X, t) — the ordering
                # that keeps a PV from accumulating into an O tile before
                # that trip's rescale.  Consecutive same-stage commits would
                # break it.
                for _h in cutlass.range(0, head_cnt, 1, unroll=1):
                    k_handle_consumer = load_kv_consumer.wait_and_advance()
                    tSrKh = tSrK[None, None, None, k_handle_consumer.index]
                    self.gemm_qk(tStS0, tSrQ0, tSrKh)
                    s0_handle_producer.commit()
                    k_handle_consumer.release()
                    v_handle_consumer = load_kv_consumer.wait_and_advance()
                    tOrVh = tOrV[None, None, None, v_handle_consumer.index]
                    if cutlass.const_expr(not self.has_logits_transform):
                        o0_handle_producer = mma_corr_producer.acquire_and_advance()
                    s0_handle_producer = mma_s0_producer.acquire_and_advance()
                    pv0_acc = pv0_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                    self.gemm_pv(tOtO0, tOrP0, tOrVh, pv0_acc)
                    pv0_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    if cutlass.const_expr(not self.has_logits_transform):
                        o0_handle_producer.commit()
                        o1_dummy_producer = mma_corr_producer.acquire_and_advance()
                        o1_dummy_producer.commit()
                    v_handle_consumer.release()

                # MAIN prologue: first block visited by both halves.
                # GEMM_QK0 (Q0 * K -> S0)
                k_handle_consumer = load_kv_consumer.wait_and_advance()
                tSrK0 = tSrK[None, None, None, k_handle_consumer.index]
                self.gemm_qk(tStS0, tSrQ0, tSrK0)
                s0_handle_producer.commit()

                # GEMM_QK1 (Q1 * K -> S1)
                q1_handle_consumer = load_q_consumer.wait_and_advance()
                tSrQ1 = tSrQ[None, None, None, q1_handle_consumer.index]
                s1_handle_producer = mma_s1_producer.acquire_and_advance()
                self.gemm_qk(tStS1, tSrQ1, tSrK0)
                s1_handle_producer.commit()
                k_handle_consumer.release()

                # GEMM_PV0 (P0 * V -> O0)
                v_handle_consumer = load_kv_consumer.wait_and_advance()
                tOrVi = tOrV[None, None, None, v_handle_consumer.index]
                if cutlass.const_expr(not self.has_logits_transform):
                    o0_handle_producer = mma_corr_producer.acquire_and_advance()
                s0_handle_producer = mma_s0_producer.acquire_and_advance()
                pv0_acc = pv0_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                self.gemm_pv(tOtO0, tOrP0, tOrVi, pv0_acc)
                pv0_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                if cutlass.const_expr(not self.has_logits_transform):
                    o0_handle_producer.commit()

                for _i in cutlass.range(0, main_cnt - 1, 1, unroll=1):
                    # GEMM_QK0i
                    k_handle_consumer = load_kv_consumer.wait_and_advance()
                    tSrKi = tSrK[None, None, None, k_handle_consumer.index]
                    self.gemm_qk(tStS0, tSrQ0, tSrKi)
                    s0_handle_producer.commit()

                    # GEMM_PV1(i-1) — read the constant-folded ACCUMULATE
                    # bit just before the call so gemm_pv sees a JIT-time bool.
                    if cutlass.const_expr(not self.has_logits_transform):
                        o1_handle_producer = mma_corr_producer.acquire_and_advance()
                    s1_handle_producer = mma_s1_producer.acquire_and_advance()
                    pv_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                    self.gemm_pv(tOtO1, tOrP1, tOrVi, pv_acc)
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    if cutlass.const_expr(not self.has_logits_transform):
                        o1_handle_producer.commit()
                    v_handle_consumer.release()

                    # GEMM_QK1i
                    self.gemm_qk(tStS1, tSrQ1, tSrKi)
                    s1_handle_producer.commit()
                    k_handle_consumer.release()

                    # GEMM_PV0i
                    v_handle_consumer = load_kv_consumer.wait_and_advance()
                    tOrVi = tOrV[None, None, None, v_handle_consumer.index]
                    if cutlass.const_expr(not self.has_logits_transform):
                        o0_handle_producer = mma_corr_producer.acquire_and_advance()
                    s0_handle_producer = mma_s0_producer.acquire_and_advance()
                    self.gemm_pv(tOtO0, tOrP0, tOrVi, True)
                    if cutlass.const_expr(not self.has_logits_transform):
                        o0_handle_producer.commit()

                # MAIN epilogue: PV1 of the last shared block.
                if cutlass.const_expr(not self.has_logits_transform):
                    o1_handle = mma_corr_producer.acquire_and_advance()
                s1_handle_producer = mma_s1_producer.acquire_and_advance()
                pv_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                self.gemm_pv(tOtO1, tOrP1, tOrVi, pv_acc)
                pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                if cutlass.const_expr(not self.has_logits_transform):
                    o1_handle.commit()
                v_handle_consumer.release()

                # Stage 0's stream is complete: commit its tile-end S slot
                # NOW, before the tail.  Softmax 0's end-of-item wait pairs
                # with this commit, and its final vec publish gates the tail
                # tokens it produces for stage 1 — committing after the tail
                # would close a dependency cycle (tail PV1 -> softmax1 tail
                # step -> stage-0 tail token -> softmax0 final -> this
                # commit -> tail PV1) and deadlock.  Early commit also lets
                # the O0 drain overlap the tail trips.
                s0_handle_producer.commit()

                # TAIL: stage-1-only trips (same slot discipline as HEAD;
                # the empty o0 commit preserves o-ring alternation — see the
                # HEAD comment).
                for _t in cutlass.range(0, tail_cnt, 1, unroll=1):
                    k_handle_consumer = load_kv_consumer.wait_and_advance()
                    tSrKt = tSrK[None, None, None, k_handle_consumer.index]
                    self.gemm_qk(tStS1, tSrQ1, tSrKt)
                    s1_handle_producer.commit()
                    k_handle_consumer.release()
                    if cutlass.const_expr(not self.has_logits_transform):
                        o0_dummy_producer = mma_corr_producer.acquire_and_advance()
                        o0_dummy_producer.commit()
                    v_handle_consumer = load_kv_consumer.wait_and_advance()
                    tOrVt = tOrV[None, None, None, v_handle_consumer.index]
                    if cutlass.const_expr(not self.has_logits_transform):
                        o1_handle = mma_corr_producer.acquire_and_advance()
                    s1_handle_producer = mma_s1_producer.acquire_and_advance()
                    pv_acc = pv_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                    self.gemm_pv(tOtO1, tOrP1, tOrVt, pv_acc)
                    pv_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    if cutlass.const_expr(not self.has_logits_transform):
                        o1_handle.commit()
                    v_handle_consumer.release()

                # release Q0 & Q1 (Q1 is read by TAIL QK1 gemms)
                q0_handle_consumer.release()
                q1_handle_consumer.release()

                s1_handle_producer.commit()

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        # dealloc tmem buffer
        self.dealloc_tmem(storage, tmem_dealloc_mbar_ptr)
