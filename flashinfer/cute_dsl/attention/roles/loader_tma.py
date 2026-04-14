# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""LoaderOps — TMA load primitives and orchestration for attention kernels.

Reusable primitives (pipeline-unaware, for composing new kernel variants):
- partition_q(): partition Q global tensor for TMA loads
- partition_k(): partition K global tensor for TMA loads
- partition_v(): partition V global tensor for TMA loads
- load_tile(): issue a single TMA load with barrier

Orchestration (prefill-specific, uses raw CuTe ops for JIT compatibility):
- run(): Q0/Q1 double-buffered loads with KV streaming
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from cutlass.pipeline import PipelineProducer

from ..config import AttentionConfig
from ..fusion.mask import get_trip_count, get_kv_start_block_idx
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
        self.mask_type = config.mask_type
        self.window_left = config.window_left

    # =========================================================================
    #  Reusable primitives — for composing new kernel variants
    #
    #  NOTE on CuTe DSL JIT limitations:
    #  - partition_q/k/v(): Return tensor tuples — CuTe DSL JIT does not
    #    reliably handle returning tensors from @cute.jit methods.
    #  - load_tile(): Uses runtime indexing (handle.index) to create tensor
    #    views internally — causes correctness issues in CuTe DSL JIT.
    #  These primitives document the intended decomposition but cannot be
    #  used inside run() until CuTe DSL JIT support improves. Use the
    #  inline patterns in run() as the working reference.
    # =========================================================================

    @cute.jit
    def partition_q(
        self,
        qk_thr_mma: cute.core.ThrMma,
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
        qk_thr_mma: cute.core.ThrMma,
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
        pv_thr_mma: cute.core.ThrMma,
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
        producer: PipelineProducer,
    ):
        """Issue a single TMA load into SMEM with pipeline barrier."""
        handle = producer.acquire_and_advance()
        cute.copy(
            tma_atom,
            src_global,
            dst_smem[None, handle.index],
            tma_bar_ptr=handle.barrier,
        )

    # =========================================================================
    #  Prefill orchestration — proven-correct inline implementation
    # =========================================================================

    @cute.jit
    def run(
        self,
        qk_thr_mma: cute.core.ThrMma,
        pv_thr_mma: cute.core.ThrMma,
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
        load_q_producer: PipelineProducer,
        load_kv_producer: PipelineProducer,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        """Loader warp orchestration loop (prefill-specific).

        Q0/Q1 double-buffered loads with KV tile streaming.
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
                gQ_qdl = cute.flat_divide(
                    mQ_qdl_, cute.select(self.qk_mma_tiler, mode=[0, 2])
                )
                tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(tSgQ_qdl, 0, 3),
                )
                tQgQ = tQgQ_qdl[None, None, 0, curr_block_coord_q[2]]

                gK_kdl = cute.flat_divide(
                    mK_kdl_, cute.select(self.qk_mma_tiler, mode=[1, 2])
                )
                tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_k,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(tSgK_kdl, 0, 3),
                )
                tKgK = tKgK_kdl[None, None, 0, curr_block_coord_kv[2]]

                gV_dkl = cute.flat_divide(
                    mV_dkl_, cute.select(self.pv_mma_tiler, mode=[1, 2])
                )
                tSgV_dkl = pv_thr_mma.partition_B(gV_dkl)
                tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_v,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sV, 0, 3),
                    cute.group_modes(tSgV_dkl, 0, 3),
                )
                tVgV = tVgV_dkl[None, 0, None, curr_block_coord_kv[2]]

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
                kv_coord = get_kv_start_block_idx(
                    self.mask_type,
                    self.window_left,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_k,
                    seqlen_q,
                )
                k_handle_producer = load_kv_producer.acquire_and_advance()
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
                # V0
                v_handle_producer = load_kv_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_v,
                    tVgV[None, kv_coord],
                    tVsV[None, v_handle_producer.index],
                    tma_bar_ptr=v_handle_producer.barrier,
                )
                kv_coord += 1

                seqlen_kv_loop_steps = (
                    get_trip_count(
                        self.mask_type,
                        self.window_left,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_k,
                        seqlen_q,
                    )
                    - 1
                )
                for _i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                    # Ki
                    k_handle_producer = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK[None, kv_coord],
                        tKsK[None, k_handle_producer.index],
                        tma_bar_ptr=k_handle_producer.barrier,
                    )
                    # Vi
                    v_handle_producer = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV[None, kv_coord],
                        tVsV[None, v_handle_producer.index],
                        tma_bar_ptr=v_handle_producer.barrier,
                    )
                    kv_coord += 1

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
