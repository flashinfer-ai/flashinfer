# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""EpilogueOps — TMA store primitives and orchestration for attention output.

Reusable primitives (pipeline-unaware, for composing new kernel variants):
- partition_output(): partition output global tensor for TMA stores
- store_tile(): issue a single TMA store + commit group

Orchestration (prefill-specific, uses raw CuTe ops for JIT compatibility):
- run(): O0/O1 double-buffered TMA stores with pipeline sync
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

from cutlass.pipeline import PipelineConsumer

from ..config import AttentionConfig
from ..scheduler.persistent import (
    FmhaStaticTileScheduler,
    FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler,
)


class EpilogueRole:
    """Epilogue warp for attention kernels — TMA stores output to global memory.

    Created from AttentionConfig in the kernel's __init__.
    """

    def __init__(self, config: AttentionConfig):
        self.pv_mma_tiler = config.pv_mma_tiler
        self.cta_tiler = config.cta_tiler

    # =========================================================================
    #  Reusable primitives — for composing new kernel variants
    #
    #  NOTE on CuTe DSL JIT limitations:
    #  - partition_output(): Returns tensor tuples — CuTe DSL JIT does not
    #    reliably handle returning tensors from @cute.jit methods.
    #  - store_tile(): SAFE — takes pre-sliced tensors as arguments, no
    #    runtime indexing or return values. Used in run() successfully.
    # =========================================================================

    @cute.jit
    def partition_output(
        self,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        sO: cute.Tensor,
        block_coord: tuple,
    ):
        """Partition output global tensor for TMA stores. Returns (tOsO, tOgO)."""
        gO_qdl = cute.flat_divide(mO_qdl, cute.select(self.pv_mma_tiler, mode=[0, 1]))
        gO = gO_qdl[None, None, None, 0, block_coord[2]]
        tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
            tma_atom_o,
            0,
            cute.make_layout(1),
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )
        return tOsO, tOgO

    @cute.jit
    def store_tile(
        self,
        tma_atom_o: cute.CopyAtom,
        tOsO_slice: cute.Tensor,
        tOgO_slice: cute.Tensor,
    ):
        """Issue a single TMA store from SMEM to GMEM + commit group."""
        cute.copy(tma_atom_o, tOsO_slice, tOgO_slice)
        cute.arch.cp_async_bulk_commit_group()

    # =========================================================================
    #  Prefill orchestration — proven-correct inline implementation
    # =========================================================================

    @cute.jit
    def run(
        self,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        sO: cute.Tensor,
        cum_seqlen_q: cute.Tensor | None,
        corr_epi_consumer: PipelineConsumer | None,
        s0_epi_consumer: PipelineConsumer | None,
        s1_epi_consumer: PipelineConsumer | None,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        """Epilogue warp orchestration loop (prefill-specific).

        O0/O1 double-buffered TMA stores with pipeline synchronization.

        Standard path: consumes from corr_epi (correction -> epilogue).
        Transform path: consumes from s0_epi/s1_epi (softmax -> epilogue).
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
            seqlen_q = mO_qdl.shape[0]

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
                curr_block_coord_o = curr_block_coord
                mO_qdl_ = mO_qdl
                if cutlass.const_expr(cum_seqlen_q is not None):
                    logical_offset_mO = (
                        mO_qdl_.shape[0] - seqlen_q,
                        0,
                        (0, cuseqlen_q + seqlen_q),
                    )
                    mO_qdl_ = cute.domain_offset(logical_offset_mO, mO_qdl_)
                    curr_block_coord_o = (
                        curr_block_coord[0],
                        curr_block_coord[1],
                        (curr_block_coord[2][0], 0),
                    )

                o0_coord = 2 * curr_block_coord_o[0]
                o1_coord = o0_coord + 1
                gO_qdl = cute.flat_divide(
                    mO_qdl_, cute.select(self.pv_mma_tiler, mode=[0, 1])
                )
                gO = gO_qdl[None, None, None, 0, curr_block_coord_o[2]]
                tOsO, tOgO = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_o,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sO, 0, 2),
                    cute.group_modes(gO, 0, 2),
                )

                if cutlass.const_expr(corr_epi_consumer is not None):
                    # Standard path: O0/O1 from correction warp
                    o0_handle_consumer = corr_epi_consumer.wait_and_advance()
                    self.store_tile(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])

                    o1_handle_consumer = corr_epi_consumer.wait_and_advance()
                    self.store_tile(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])

                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                    o0_handle_consumer.release()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    o1_handle_consumer.release()
                else:
                    # Transform path: O0 from softmax0, O1 from softmax1
                    o0_handle_consumer = s0_epi_consumer.wait_and_advance()
                    self.store_tile(tma_atom_o, tOsO[None, 0], tOgO[None, o0_coord])

                    o1_handle_consumer = s1_epi_consumer.wait_and_advance()
                    self.store_tile(tma_atom_o, tOsO[None, 1], tOgO[None, o1_coord])

                    cute.arch.cp_async_bulk_wait_group(1, read=True)
                    o0_handle_consumer.release()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    o1_handle_consumer.release()

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
