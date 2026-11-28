# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correction of the 2CTA MLA O accumulators."""

import cutlass.cute as cute
import cutlass
from cutlass import Float32, Int32
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field
from ...device.constants import TCGEN05_32B_REGS_PER_LOAD, TCGEN05_32B_SHAPE
from ..config import MlaDecodeConfig
from .common import HighThroughputMlaResource
from typing import Any, ClassVar


# =====================================================================
# TmemCorrResource — Correction factors via TMEM, Async pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemCorrResource(HighThroughputMlaResource):
    """Correction factors in TMEM.  Producer: SoftmaxTask.  Consumer: CorrectionTask.

    Pipeline: Async, 2 stages.
    Carries (row_sum, row_max, correction_factor, no_correction) per thread.
    """

    # Static configuration
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    correction_factor: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    epilogue_row_max: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    epilogue_row_sum: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    no_correction: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_sum: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("row_sum", Float32, Float32(0), "Final running row sum."),
        ("row_max", Float32, Float32(0), "Final running row max."),
        ("epilogue_row_sum", Float32, Float32(0), "Final exchanged row sum."),
        ("epilogue_row_max", Float32, Float32(0), "Final row max for LSE."),
        (
            "correction_factor",
            Float32,
            Float32(0),
            "Correction factor for the previous O tile.",
        ),
        ("no_correction", Int32, Int32(0), "Whether O correction may be skipped."),
    )

    # Per-task state
    cta_rank: Any = field(init=False, default=None)
    final_row_stats: Any = field(init=False, default=None)
    page_offsets: Any = None
    score_resource: Any = None
    smem_exchange: Any = None
    tmem_base_addr: Any = None

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(row_sum, row_max, correction_factor, no_correction),
    )
    @cute.jit
    def init_load_state(self, stage_info: StageInfo):
        """Create row-stat variables consumed by correction and epilogue code."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.final_row_stats = cutlass.Array(
            Float32,
            2,
            space=cutlass.AddressSpace.rmem,
        )
        return Float32(0), Float32(0), Float32(0), Int32(0)

    @consumer_work(returns=(row_sum, row_max, correction_factor, no_correction))
    @cute.jit
    def load_corr(self, stage_info: StageInfo):
        """Load correction factors from TMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        tidx = cute.arch.thread_idx()[0]
        # Use local tidx within 4-warp correction group (matching bare-metal)
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        warp_id = local_tidx >> 5

        col_offset = cfg.correction_factor_offset + stage_idx * 4
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        # Load from the same packed row/column address used by store_corr so the
        # correction consumer sees the row statistics for its current stage.
        tmem_raw_addr = (tmem_warp_row_id << 16) | col_offset
        tmem_ptr_arr = prims.make_tmem_ptr(tmem_raw_addr, Float32)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        loaded = prims.tcgen05_ld(
            TCGEN05_32B_SHAPE,
            tmem_ptr_arr,
            num=4,
        )

        self.final_row_stats[0] = loaded[0]
        self.final_row_stats[1] = loaded[1]
        return loaded[0], loaded[1], loaded[2], loaded[3].bitcast(Int32)

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY, returns=(epilogue_row_sum, epilogue_row_max)
    )
    @cute.jit
    def prepare_epilogue_slice_store(self, stage_info: StageInfo):
        """Exchange final row statistics once before per-slice O stores."""
        del stage_info
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        row_sum = self.final_row_stats[0]
        row_max = self.final_row_stats[1]

        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        smem_ex_ptr = cutlass.inttoptr(
            self.smem_exchange + local_tidx * 4,
            3,
            Float32,
        )
        smem_ex_ptr.store(row_sum)
        prims.barrier_cta_sync(
            cfg.epilogue_sync_bar_id, thread_count=cfg.epilogue_sync_threads
        )
        # The two CTAs in the cluster own complementary halves of the 2CTA row.
        # Exchanging row sums through SMEM gives both epilogue slices the same
        # denominator while preserving each CTA's local row max for LSE.
        peer_idx = (local_tidx + 64) % num_compute_threads
        peer_ptr = cutlass.inttoptr(
            self.smem_exchange + peer_idx * 4,
            3,
            Float32,
        )
        row_sum = row_sum + peer_ptr.load()
        prims.barrier_cta_sync(
            cfg.epilogue_sync_bar_id, thread_count=cfg.epilogue_sync_threads
        )
        return row_sum, row_max

    @producer_work
    @cute.jit
    def store_corr(
        self,
        stage_info: StageInfo,
        *,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ) -> None:
        """Store correction metadata for the even softmax group."""
        self.store_corr_impl(
            stage_info,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
            softmax_group_id=0,
            arrive_peer=True,
        )

    @cute.jit
    def store_corr_impl(
        self,
        stage_info: StageInfo,
        *,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
        softmax_group_id: cutlass.Constexpr[int] = 0,
        arrive_peer: cutlass.Constexpr[bool] = True,
    ) -> None:
        """Store correction factors [row_sum, row_max, correction, no_correction] to TMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        warp_id = local_tidx >> 5

        col_offset = cfg.correction_factor_offset + stage_idx * 4
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        # tcgen05 addresses pack the TMEM row into the high 16 bits.  Each warp
        # owns one correction row and four adjacent columns for sum/max/scale.
        tmem_raw_addr = (tmem_warp_row_id << 16) | col_offset
        tmem_ptr_arr = prims.make_tmem_ptr(tmem_raw_addr, Float32)

        correction_regs = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        correction_regs[0] = row_sum_out
        correction_regs[1] = row_max_new
        if cutlass.const_expr(cfg.has_fp8_primary):
            work = stage_info.work_tile
            current = (work.k_index_base + Int32(stage_info.loop_offset)) * Int32(
                cfg.mma_qk_tiler[1]
            )
            if cutlass.const_expr(self.page_offsets.is_bound):
                routes = self.page_offsets
            else:
                routes = self.page_offsets.bind_query_row(
                    work.tile_idx[2],
                    self.score_resource.request_lengths[0],
                    self.score_resource.request_lengths[1],
                )
            ratio = routes.tensor_scale_ratio(
                current, current - Int32(cfg.mma_qk_tiler[1])
            )
            if stage_info.loop_offset != Int32(0):
                correction_factor_out *= ratio
                no_correction_out = Int32(
                    no_correction_out != 0 and ratio == cutlass.Float32(1)
                )
        correction_regs[2] = correction_factor_out
        correction_regs[3] = prims.mov_b32(no_correction_out, target_type=Float32)

        prims.tcgen05_st(
            TCGEN05_32B_SHAPE,
            tmem_ptr_arr,
            correction_regs[0:4],
        )
        cute.arch.fence_view_async_tmem_store()

        if cutlass.const_expr(cfg.use_fp8_dual_softmax_schedule and arrive_peer):
            # Dual-softmax pipes are ordered one loop tile apart: after one pipe
            # publishes correction state, it releases the peer pipe for the next
            # loop tile if that peer still has work.
            has_peer_next = stage_info.loop_offset + Int32(1) < stage_info.loop_end
            if has_peer_next:
                prims.barrier_cta_arrive(
                    (
                        cfg.softmax_order_bar_0_id
                        if cutlass.const_expr(softmax_group_id == 1)
                        else cfg.softmax_order_bar_1_id
                    ),
                    2 * num_compute_threads,
                )

    @producer_work
    @cute.jit
    def store_corr_odd(
        self,
        stage_info: StageInfo,
        *,
        row_sum_out_odd,
        row_max_new_odd,
        correction_factor_out_odd,
        no_correction_out_odd,
    ) -> None:
        """Store correction metadata for the odd softmax group."""
        self.store_corr_impl(
            stage_info,
            row_sum_out=row_sum_out_odd,
            row_max_new=row_max_new_odd,
            correction_factor_out=correction_factor_out_odd,
            no_correction_out=no_correction_out_odd,
            softmax_group_id=1,
            arrive_peer=True,
        )
