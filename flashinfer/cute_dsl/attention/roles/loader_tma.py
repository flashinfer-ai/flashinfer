# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LoaderOps — TMA load primitives and orchestration for attention kernels.

Reusable primitives (pipeline-unaware, for composing new kernel variants):
- partition_q(): partition Q global tensor for TMA loads
- partition_k(): partition K global tensor for TMA loads
- partition_v(): partition V global tensor for TMA loads
- load_tile(): issue a single TMA load with barrier

Orchestration (prefill-specific):
- run(): Q0/Q1 double-buffered loads with KV streaming
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from cutlass.pipeline import PipelineProducer

from ..config import AttentionConfig
from ..fusion.mask import get_kv_block_range
from ..scheduler.persistent import (
    FmhaStaticTileScheduler,
    FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler,
)


class LoaderRole:
    """Loader warp for attention kernels — TMA loads Q, K, V into SMEM.

    Created from AttentionConfig in the kernel's __init__.
    """

    def __init__(self, config: AttentionConfig):
        self.cta_tiler = config.cta_tiler
        self.qk_mma_tiler = config.qk_mma_tiler
        self.pv_mma_tiler = config.pv_mma_tiler
        self.mask_spec = config.mask_spec
        # Paged KV: tokens per page (None = ragged) and the per-tile
        # TMA copy count, both compile-time.
        self.page_size = config.page_size
        self.pages_per_kv_tile = config.pages_per_kv_tile
        # Populated by set_v_tx_bytes() once dtypes are known at __call__
        # time.  None means K and V have the same width (plain acquires).
        self.v_tx_bytes = None

    def set_v_tx_bytes(self, v_tx_bytes) -> None:
        """Set V's TMA byte count for mixed K/V dtype builds.

        The shared K/V ring's barriers are initialized with K's byte
        count; when the widths differ, each V acquire re-arms its slot
        with V's byte count via ``acquire_and_advance(expected_tx=...)``
        (pass None for uniform builds: expected_tx=None keeps the
        barrier-init count).
        """
        self.v_tx_bytes = v_tx_bytes

    @cute.jit
    def _load_paged_tile(
        self,
        tma_atom,
        tGg,
        tGs,
        stage_index,
        barrier,
        kv_tile,
        table_base,
        page_idx_lb,
        num_pages_kv,
        kv_page_table,
        first_tile: cutlass.Constexpr,
        is_k: cutlass.Constexpr,
    ):
        """Issue one K or V tile's per-page TMA copies.

        ``unroll=1`` is required: left to heuristics, LLVM/ptxas
        fully unroll this constant-trip loop (static UTMALDGs 12 -> 68)
        and the enlarged loader text regresses ps16 to ~1.2x ragged via
        instruction-fetch stalls (four warp-group code regions share
        fetch).  K and V run separate loops and re-read the ids —
        fusing the loops or staging ids across them measured worse.

        A page id of -1 puts the TMA coordinate out of bounds: the copy
        writes zeros to smem and still credits the barrier with the full
        box bytes.  Two kinds of pages are mapped to -1: pages past the
        end of the sequence, and — on windowed kernels — pages below
        ``page_idx_lb``, whose table slots serving frameworks repoint at
        a shared scratch block that may hold NaN (trtllm-gen's pageIdxLb
        contract).  Loading zeros there, rather than relying on the
        softmax mask, is required for correctness: masked-out positions
        still pass through the PV MMA with P=0, and 0 * NaN = NaN.
        Only an item's first tile can contain the window lower bound,
        so only ``first_tile`` applies the window clamp.  The table
        index is min-clamped separately so the table read itself never
        goes past the item's slice; this relies on every item having at
        least one page (``num_pages_kv >= 1``, i.e. ``safe_page >= 0``),
        which plan() enforces by rejecting zero-length KV items.
        """
        logical0 = kv_tile * self.pages_per_kv_tile
        for p in cutlass.range(self.pages_per_kv_tile, unroll=1):
            logical_page = logical0 + p
            safe_page = cutlass.min(logical_page, num_pages_kv - 1)
            page_idx = kv_page_table[table_base + safe_page]
            if cutlass.const_expr(first_tile and self.mask_spec.has_window_left):
                in_range = (logical_page >= page_idx_lb) & (logical_page < num_pages_kv)
            else:
                in_range = logical_page < num_pages_kv
            page_idx = cutlass.select_(in_range, page_idx, Int32(-1))
            if cutlass.const_expr(is_k):
                cute.copy(
                    tma_atom,
                    tGg[None, page_idx],
                    tGs[None, p, 0, stage_index],
                    tma_bar_ptr=barrier,
                )
            else:
                cute.copy(
                    tma_atom,
                    tGg[None, page_idx],
                    tGs[None, 0, p, stage_index],
                    tma_bar_ptr=barrier,
                )

    # =========================================================================
    #  Reusable primitives — for composing new kernel variants
    #
    #  Hazard (nvidia-cutlass-dsl 4.6.0): the DSL picks a dynamic loop's
    #  carried values from the CALLER's own syntax (assignments and
    #  x.method() receivers), so pipeline/scheduler state advanced inside
    #  a helper is not yielded across while/for iterations and silently
    #  resets each iteration.  Keep acquire/advance calls in run()'s loop
    #  body and pass handle.index / handle.barrier into helpers (the
    #  _load_paged_tile pattern), or return the mutated object and rebind
    #  it at the call site (the softmax.py step() pattern).
    # =========================================================================

    @cute.jit
    def partition_q(
        self,
        qk_thr_mma: cute.ThrMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        sQ: cute.Tensor,
        block_coord: tuple,
    ):
        """Partition Q global tensor for TMA loads. Returns (tQsQ, tQgQ)."""
        gQ_qdl = cute.flat_divide(mQ_qdl, cute.select(self.qk_mma_tiler, mode=[0, 2]))
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )
        tQgQ = tQgQ_qdl[None, None, 0, block_coord[2]]
        return tQsQ, tQgQ

    @cute.jit
    def partition_k(
        self,
        qk_thr_mma: cute.ThrMma,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        sK: cute.Tensor,
        block_coord: tuple,
    ):
        """Partition K global tensor for TMA loads. Returns (tKsK, tKgK)."""
        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tSgK_kdl, 0, 3),
        )
        tKgK = tKgK_kdl[None, None, 0, block_coord[2]]
        return tKsK, tKgK

    @cute.jit
    def partition_v(
        self,
        pv_thr_mma: cute.ThrMma,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        sV: cute.Tensor,
        block_coord: tuple,
    ):
        """Partition V global tensor for TMA loads. Returns (tVsV, tVgV)."""
        gV_dkl = cute.flat_divide(mV_dkl, cute.select(self.pv_mma_tiler, mode=[1, 2]))
        tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tSgV_dkl, 0, 3),
        )
        tVgV = tVgV_dkl[None, 0, None, block_coord[2]]
        return tVsV, tVgV

    @cute.jit
    def load_tile(
        self,
        tma_atom: cute.CopyAtom,
        src_global: cute.Tensor,
        dst_smem: cute.Tensor,
        index: Int32,
        barrier,
    ):
        """Issue a single TMA load into a runtime-indexed SMEM stage slot.

        The pipeline acquire stays in the caller's loop body (see the
        helper-method rules above): do ``handle =
        producer.acquire_and_advance()`` inline and pass ``handle.index``
        / ``handle.barrier`` here.
        """
        cute.copy(
            tma_atom,
            src_global,
            dst_smem[None, index],
            tma_bar_ptr=barrier,
        )

    # =========================================================================
    #  Prefill orchestration — proven-correct inline implementation
    # =========================================================================

    @cute.jit
    def run(
        self,
        qk_thr_mma: cute.ThrMma,
        pv_thr_mma: cute.ThrMma,
        tma_atom_q: cute.CopyAtom,
        tma_atom_k: cute.CopyAtom,
        tma_atom_v: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        mK_kdl: cute.Tensor,
        mV_dkl: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        window_left: Int32,
        window_right: Int32,
        load_q_producer: PipelineProducer,
        load_kv_producer: PipelineProducer,
        tile_sched_params: FmhaStaticTileSchedulerParams,
        sK_tma: cute.Tensor | None,
        sV_tma: cute.Tensor | None,
        kv_page_table: cute.Tensor | None,
        kv_page_indptr: cute.Tensor | None,
    ):
        """Loader warp orchestration loop (prefill-specific).

        Q0/Q1 double-buffered loads with KV tile streaming.  With paged KV
        (``config.page_size`` set), K/V TMA boxes shrink to one page, each
        tile issues ``pages_per_kv_tile`` copies indexed by runtime page
        ids from ``kv_page_table``, and ``sK_tma``/``sV_tma`` are the
        per-page-divided views of the same K/V smem ring.
        """
        tile_sched = create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            continue_cond = False
            cuseqlen_q = Int32(0)
            seqlen_q = mQ_qdl.shape[0]
            if cutlass.const_expr(cum_seqlen_q is not None):
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q,
                    )
                )
            if not continue_cond:
                mQ_qdl_ = mQ_qdl
                mK_kdl_ = mK_kdl
                mV_dkl_ = mV_dkl
                seqlen_k = mK_kdl.shape[0]
                curr_block_coord_q = curr_block_coord
                curr_block_coord_kv = curr_block_coord

                if cutlass.const_expr(cum_seqlen_q is not None):
                    logical_offset_mQ = (cuseqlen_q, 0, (0, 0))
                    mQ_qdl_ = cute.domain_offset(logical_offset_mQ, mQ_qdl)
                    curr_block_coord_q = (
                        curr_block_coord[0],
                        curr_block_coord[1],
                        (curr_block_coord[2][0], Int32(0)),
                    )

                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    if cutlass.const_expr(self.page_size is None):
                        logical_offset_mK = (cuseqlen_k, 0, (0, 0))
                        logical_offset_mV = (0, cuseqlen_k, (0, 0))
                        mK_kdl_ = cute.domain_offset(logical_offset_mK, mK_kdl)
                        mV_dkl_ = cute.domain_offset(logical_offset_mV, mV_dkl)
                        curr_block_coord_kv = (
                            curr_block_coord[0],
                            curr_block_coord[1],
                            (curr_block_coord[2][0], Int32(0)),
                        )

                # Local tile partition global tensors
                tQsQ, tQgQ = self.partition_q(
                    qk_thr_mma, tma_atom_q, mQ_qdl_, sQ, curr_block_coord_q
                )

                table_base = Int32(0)
                num_pages_kv = Int32(0)
                page_idx_lb = Int32(0)
                if cutlass.const_expr(self.page_size is not None):
                    # Paged: per-page TMA boxes over the page pool.  No
                    # thr_mma partition_B — the box is one page, not the
                    # MMA tile — and no domain shift: the trailing page
                    # mode is indexed with a runtime page id per copy
                    # (the MLA decode loader pattern).
                    gK_kdl = cute.tiled_divide(
                        mK_kdl_, (self.page_size, self.qk_mma_tiler[2])
                    )
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        0,
                        cute.make_layout(1),
                        sK_tma,
                        gK_kdl[None, 0, 0, None],
                    )
                    tKgK = tKgK_kdl[None, (curr_block_coord[2][0], None)]

                    gV_dkl = cute.tiled_divide(
                        mV_dkl_, (self.pv_mma_tiler[1], self.page_size)
                    )
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        0,
                        cute.make_layout(1),
                        sV_tma,
                        gV_dkl[None, 0, 0, None],
                    )
                    tVgV = tVgV_dkl[None, (curr_block_coord[2][0], None)]

                    table_base = kv_page_indptr[batch_coord]
                    num_pages_kv = cute.ceil_div(seqlen_k, self.page_size)
                    # Window clamp lower bound: pages wholly below every Q
                    # row's band start are never attended, and their table
                    # slots may point at a reclaimed null block (see
                    # _paged_page_idx).  Row 0 of the Q tile has the
                    # smallest band start: lo = q0 + (seqlen_k - seqlen_q)
                    # - window_left.
                    if cutlass.const_expr(self.mask_spec.has_window_left):
                        q0 = curr_block_coord[0] * self.cta_tiler[0]
                        lo_min = q0 + (seqlen_k - seqlen_q) - window_left
                        page_idx_lb = cutlass.max(lo_min, Int32(0)) // self.page_size
                else:
                    tKsK, tKgK = self.partition_k(
                        qk_thr_mma, tma_atom_k, mK_kdl_, sK, curr_block_coord_kv
                    )
                    tVsV, tVgV = self.partition_v(
                        pv_thr_mma, tma_atom_v, mV_dkl_, sV, curr_block_coord_kv
                    )

                # Q0
                q0_coord = 2 * curr_block_coord_q[0]
                q0_handle_producer = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ[None, q0_coord],
                    tQsQ[None, q0_handle_producer.index],
                    tma_bar_ptr=q0_handle_producer.barrier,
                )
                # K0
                kv_block_start, kv_block_end = get_kv_block_range(
                    self.mask_spec,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_k,
                    seqlen_q,
                    window_left,
                    window_right,
                )
                kv_coord = kv_block_start
                k_handle_producer = load_kv_producer.acquire_and_advance()
                if cutlass.const_expr(self.page_size is not None):
                    self._load_paged_tile(
                        tma_atom_k,
                        tKgK,
                        tKsK,
                        k_handle_producer.index,
                        k_handle_producer.barrier,
                        kv_coord,
                        table_base,
                        page_idx_lb,
                        num_pages_kv,
                        kv_page_table,
                        True,
                        True,
                    )
                else:
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, kv_coord],
                        tKsK[None, k_handle_producer.index],
                        tma_bar_ptr=k_handle_producer.barrier,
                    )
                # Q1
                q1_coord = q0_coord + 1
                q1_handle_producer = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ[None, q1_coord],
                    tQsQ[None, q1_handle_producer.index],
                    tma_bar_ptr=q1_handle_producer.barrier,
                )
                # V0.  With mixed K/V widths, expected_tx re-arms the slot
                # with V's byte count (the ring's barriers are init-armed
                # with K's); expected_tx=None is the plain acquire.
                v_handle = load_kv_producer.acquire_and_advance(
                    expected_tx=self.v_tx_bytes
                )
                if cutlass.const_expr(self.page_size is not None):
                    self._load_paged_tile(
                        tma_atom_v,
                        tVgV,
                        tVsV,
                        v_handle.index,
                        v_handle.barrier,
                        kv_coord,
                        table_base,
                        page_idx_lb,
                        num_pages_kv,
                        kv_page_table,
                        True,
                        False,
                    )
                else:
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, kv_coord],
                        tVsV[None, v_handle.index],
                        tma_bar_ptr=v_handle.barrier,
                    )
                kv_coord += 1

                seqlen_kv_loop_steps = kv_block_end - kv_block_start - 1
                for _i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                    # Ki
                    k_handle_producer = load_kv_producer.acquire_and_advance()
                    if cutlass.const_expr(self.page_size is not None):
                        self._load_paged_tile(
                            tma_atom_k,
                            tKgK,
                            tKsK,
                            k_handle_producer.index,
                            k_handle_producer.barrier,
                            kv_coord,
                            table_base,
                            page_idx_lb,
                            num_pages_kv,
                            kv_page_table,
                            False,
                            True,
                        )
                    else:
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_coord],
                            tKsK[None, k_handle_producer.index],
                            tma_bar_ptr=k_handle_producer.barrier,
                        )
                    # Vi (see V0 for the expected_tx contract)
                    v_handle = load_kv_producer.acquire_and_advance(
                        expected_tx=self.v_tx_bytes
                    )
                    if cutlass.const_expr(self.page_size is not None):
                        self._load_paged_tile(
                            tma_atom_v,
                            tVgV,
                            tVsV,
                            v_handle.index,
                            v_handle.barrier,
                            kv_coord,
                            table_base,
                            page_idx_lb,
                            num_pages_kv,
                            kv_page_table,
                            False,
                            False,
                        )
                    else:
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, kv_coord],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=v_handle.barrier,
                        )
                    kv_coord += 1

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
