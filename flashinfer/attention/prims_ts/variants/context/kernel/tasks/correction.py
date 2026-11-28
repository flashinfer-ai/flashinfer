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

"""Correction tasks of the context kernel: O rescale, normalization and partial output."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ..resources.gmem_o import GmemOResource
from ..resources.smem_o import SmemOResource
from ..resources.sync import TmemStatsDoneResource
from ..resources.tmem_o import TmemOResource
from ..resources.tmem_stats import TmemStatsResource
from .common import (
    captured_loop_bounds,
    packed_context_skip_predicate,
    schedule_with_work_queue,
    src_resources,
    work_tile_schedule_loop,
)
from typing import Any


def create_correction_task(
    tmem_vec0: TmemStatsResource,
    tmem_vec1: TmemStatsResource | None,
    tmem_o: TmemOResource,
    smem_o_0: SmemOResource,
    smem_o_1: SmemOResource | None,
    gmem_o_0: GmemOResource,
    gmem_o_1: GmemOResource | None,
    tmem_vec_done_0: TmemStatsDoneResource,
    tmem_vec_done_1: TmemStatsDoneResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the four-warp Correction task (warps 8-11)."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = packed_context_skip_predicate(work_queue)

    def create_single_instance_task() -> Task:
        fuse_epilogue = smem_o_0.cfg.fuse_epilogue_into_correction
        num_o_head_dim_stages = smem_o_0.cfg.num_o_head_dim_stages

        src = src_resources(
            tmem_vec0,
            tmem_o,
            *([] if tmem_vec0.cfg.stats_via_smem else [tmem_vec_done_0]),
            *([smem_o_0] if fuse_epilogue else []),
            work_queue=work_queue,
        )

        @schedule
        def correction_schedule(
            v0: TmemStatsResource,
            to: TmemOResource,
            so0: SmemOResource,
            go0: GmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            v0.init_read_state()
            scale_softmax_log2 = v0.load_scale_softmax_log2()
            output_scale = v0.load_output_scale()
            to.init_correction_state()
            so0.init_store_state()
            if fuse_epilogue:
                go0.init_store_state()
            with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                v0.init_read_work_tile_state()
                to.init_correction_work_tile_state()
                so0.init_store_work_tile_state()

                v0.wait()
                v0.release()
                if not tmem_vec0.cfg.stats_via_smem:
                    vd0.wait()
                    vd0.release()
                with domain_loop(loop_start, loop_end, loop_step):
                    v0.wait()
                    vec_old_max, vec_new_max, _, vec_scale = v0.read_vec(
                        scale_softmax_log2=scale_softmax_log2,
                    )
                    if not tmem_vec0.cfg.stats_via_smem:
                        vd0.wait()
                        vd0.release()
                    to.wait()
                    to.correct(
                        vec_old_max=vec_old_max,
                        vec_new_max=vec_new_max,
                        vec_scale=vec_scale,
                        inst_idx=0,
                    )
                    v0.release()
                    to.release()

                v0.wait()
                _, _, vec_row_sum, vec_scale = v0.read_vec(
                    scale_softmax_log2=scale_softmax_log2,
                    final_stats=True,
                )
                if not tmem_vec0.cfg.stats_via_smem:
                    vd0.wait()
                    vd0.release()
                v0.release()
                to.wait()
                for head_dim_stage_idx in range(num_o_head_dim_stages):
                    so0.acquire()
                    so0.store_o(
                        vec_row_sum=vec_row_sum,
                        vec_scale=vec_scale,
                        output_scale=output_scale,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    so0.commit()
                    if fuse_epilogue:
                        # The same four-warp group consumes the completed SMEM
                        # stage.  Only its first warp issues TMA; all four wait
                        # and release the pipeline stage together.
                        so0.wait()
                        head_coord, batch_coord, seq_coord_q = (
                            so0.compute_output_coords()
                        )
                        go0.tma_store(
                            head_coord=head_coord,
                            batch_coord=batch_coord,
                            seq_coord_q=seq_coord_q,
                            head_dim_stage_idx=head_dim_stage_idx,
                            correction_fused=True,
                        )
                        so0.release()
                to.release()
                if tmem_vec0.cfg.stats_via_smem:
                    # Consume the cursor-balancing record emitted by Softmax.
                    v0.wait()
                    v0.release()

        captured_schedule = schedule_with_work_queue(
            correction_schedule,
            tmem_vec0,
            tmem_o,
            smem_o_0,
            gmem_o_0,
            tmem_vec_done_0,
            work_queue=work_queue,
        )
        dst = [smem_o_0]
        if fuse_epilogue:
            dst.append(gmem_o_0)
        return task_class(
            src_resources=src,
            dst_resources=dst,
            warp_idx=smem_o_0.cfg.correction_warp_ids[0],
            num_warps=4,
            schedule=captured_schedule,
            num_registers=smem_o_0.cfg.num_regs_correction,
            name="CorrectionTask",
            **task_kwargs,
        )

    def create_paired_task() -> Task:
        if tmem_vec1 is None or smem_o_1 is None or tmem_vec_done_1 is None:
            raise ValueError("paired correction scheduling requires peer-1 resources")
        release_stats_after_read = tmem_vec0.cfg.p_in_smem
        if release_stats_after_read and not tmem_vec0.cfg.stats_via_smem:
            raise ValueError("early stats release requires SMEM-staged stats")

        src = src_resources(
            tmem_vec0,
            tmem_vec1,
            tmem_o,
            *(
                []
                if tmem_vec0.cfg.stats_via_smem
                else [tmem_vec_done_0, tmem_vec_done_1]
            ),
            work_queue=work_queue,
        )

        @schedule
        def correction_schedule(
            v0: TmemStatsResource,
            v1: TmemStatsResource,
            to: TmemOResource,
            so0: SmemOResource,
            so1: SmemOResource,
            vd0: TmemStatsDoneResource,
            vd1: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Captured schedule for O rescale and SMEM staging."""
            v0.init_read_state()
            v1.init_read_state()
            scale_softmax_log2_v0 = v0.load_scale_softmax_log2()
            scale_softmax_log2_v1 = v1.load_scale_softmax_log2()
            output_scale0 = v0.load_output_scale()
            output_scale1 = v1.load_output_scale()
            to.init_correction_state()
            so0.init_store_state()
            so1.init_store_state()
            with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                # Per-tile TMEM/SMEM cached addresses are computed here.
                v0.init_read_work_tile_state()
                v1.init_read_work_tile_state()
                to.init_correction_work_tile_state()
                so0.init_store_work_tile_state()
                so1.init_store_work_tile_state()
                # The empty stats pipeline needs no priming. Discard its first
                # slot and retain TmemStats1 for the first loop cross-release.
                v0.wait()
                v0.release()
                v1.wait()
                if release_stats_after_read:
                    v1.release()
                # The correction loop consumes vec/O pairs in alternating order so
                # each half can unblock the other half's next producer. With P in
                # SMEM each stats slot is released right after read_vec instead.
                with domain_loop(loop_start, loop_end, loop_step):
                    # Part 1: consume TmemStats0 + O0, release TmemStats1.
                    v0.wait()
                    vec_old_max, vec_new_max, _, vec_scale = v0.read_vec(
                        scale_softmax_log2=scale_softmax_log2_v0,
                    )
                    if release_stats_after_read:
                        v0.release()
                    to.wait()
                    to.correct(
                        vec_old_max=vec_old_max,
                        vec_new_max=vec_new_max,
                        vec_scale=vec_scale,
                        inst_idx=0,
                    )
                    if not release_stats_after_read:
                        v1.release()
                    to.release()
                    # Part 2: consume TmemStats1 + O1, release TmemStats0.
                    v1.wait()
                    vec_old_max, vec_new_max, _, vec_scale = v1.read_vec(
                        scale_softmax_log2=scale_softmax_log2_v1,
                    )
                    if release_stats_after_read:
                        v1.release()
                    to.wait()
                    to.correct(
                        vec_old_max=vec_old_max,
                        vec_new_max=vec_new_max,
                        vec_scale=vec_scale,
                        inst_idx=1,
                    )
                    if not release_stats_after_read:
                        v0.release()
                    to.release()
                # Tail: read the final stats, then write corrected O0/O1 to smem.
                if not release_stats_after_read:
                    v1.release()
                v0.wait()
                _, _, vec_row_sum, vec_scale = v0.read_vec(
                    scale_softmax_log2=scale_softmax_log2_v0,
                    final_stats=True,
                )
                if not tmem_vec0.cfg.stats_via_smem:
                    vd0.wait()
                    vd0.release()
                v0.release()
                to.wait()
                for head_dim_stage_idx in range(smem_o_0.cfg.num_o_head_dim_stages):
                    so0.acquire()
                    so0.store_o(
                        vec_row_sum=vec_row_sum,
                        vec_scale=vec_scale,
                        output_scale=output_scale0,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    so0.commit()
                to.release()
                v1.wait()
                _, _, vec_row_sum, vec_scale = v1.read_vec(
                    scale_softmax_log2=scale_softmax_log2_v1,
                    final_stats=True,
                )
                if not tmem_vec0.cfg.stats_via_smem:
                    vd1.wait()
                    vd1.release()
                v1.release()
                to.wait()
                for head_dim_stage_idx in range(smem_o_0.cfg.num_o_head_dim_stages):
                    so1.acquire()
                    so1.store_o(
                        vec_row_sum=vec_row_sum,
                        vec_scale=vec_scale,
                        output_scale=output_scale1,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    so1.commit()
                to.release()

        captured_schedule = schedule_with_work_queue(
            correction_schedule,
            tmem_vec0,
            tmem_vec1,
            tmem_o,
            smem_o_0,
            smem_o_1,
            tmem_vec_done_0,
            tmem_vec_done_1,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=src,
            dst_resources=[smem_o_0, smem_o_1],
            warp_idx=8,
            num_warps=4,
            schedule=captured_schedule,
            num_registers=smem_o_0.cfg.num_regs_correction,
            name="CorrectionTask",
            **task_kwargs,
        )

    if smem_o_0.cfg.single_qkv_instance:
        return create_single_instance_task()
    return create_paired_task()
