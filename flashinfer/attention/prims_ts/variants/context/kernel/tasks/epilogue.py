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

"""Epilogue tasks of the context kernel: output stores."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ..resources.gmem_o import GmemOResource
from ..resources.smem_o import SmemOResource
from .common import (
    captured_loop_bounds,
    packed_context_skip_predicate,
    schedule_with_work_queue,
    src_resources,
    work_tile_schedule_loop,
)
from typing import Any


def create_epilogue_task(
    smem_o_0: SmemOResource,
    smem_o_1: SmemOResource | None,
    gmem_o_0: GmemOResource,
    gmem_o_1: GmemOResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp Epilogue store task (warp 14)."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = packed_context_skip_predicate(work_queue)

    def create_single_instance_task() -> Task:
        src = src_resources(smem_o_0, work_queue=work_queue)
        num_o_head_dim_stages = smem_o_0.cfg.num_o_head_dim_stages

        @schedule
        def epilogue_schedule(
            so0: SmemOResource,
            go0: GmemOResource,
            wq: WorkQueue | None = None,
        ) -> None:
            so0.init_output_state()
            go0.init_store_state()
            with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                so0.init_output_work_tile_state()
                with domain_loop(loop_start, loop_end, loop_step):
                    pass
                for head_dim_stage_idx in range(num_o_head_dim_stages):
                    so0.wait()
                    head_coord, batch_coord, seq_coord_q = so0.compute_output_coords()
                    go0.tma_store(
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        seq_coord_q=seq_coord_q,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    so0.release()

        captured_schedule = schedule_with_work_queue(
            epilogue_schedule,
            smem_o_0,
            gmem_o_0,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=src,
            dst_resources=[gmem_o_0],
            warp_idx=gmem_o_0.cfg.epilogue_warp_id,
            num_warps=1,
            schedule=captured_schedule,
            num_registers=gmem_o_0.cfg.num_regs_other,
            name="EpilogueTask",
            **task_kwargs,
        )

    def create_paired_task() -> Task:
        if smem_o_1 is None or gmem_o_1 is None:
            raise ValueError("paired epilogue scheduling requires peer-1 resources")

        src = src_resources(smem_o_0, smem_o_1, work_queue=work_queue)

        @schedule
        def epilogue_schedule(
            so0: SmemOResource,
            so1: SmemOResource,
            go0: GmemOResource,
            go1: GmemOResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Captured schedule for GMEM O stores."""
            so0.init_output_state()
            so1.init_output_state()
            go0.init_store_state()
            go1.init_store_state()
            with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                # Per-tile SMEM O address base is computed in work vars.
                so0.init_output_work_tile_state()
                so1.init_output_work_tile_state()
                with domain_loop(loop_start, loop_end, loop_step):
                    pass
                # Store the first corrected O tile through gmem_o_0.
                for head_dim_stage_idx in range(smem_o_0.cfg.num_o_head_dim_stages):
                    so0.wait()
                    head_coord, batch_coord, seq_coord_q = so0.compute_output_coords()
                    go0.tma_store(
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        seq_coord_q=seq_coord_q,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    so0.release()
                # Store the second corrected O tile through gmem_o_1.
                for head_dim_stage_idx in range(smem_o_0.cfg.num_o_head_dim_stages):
                    so1.wait()
                    head_coord, batch_coord, seq_coord_q = so1.compute_output_coords()
                    go1.tma_store(
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        seq_coord_q=seq_coord_q,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    so1.release()

        captured_schedule = schedule_with_work_queue(
            epilogue_schedule,
            smem_o_0,
            smem_o_1,
            gmem_o_0,
            gmem_o_1,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=src,
            dst_resources=[gmem_o_0, gmem_o_1],
            warp_idx=14,
            num_warps=1,
            schedule=captured_schedule,
            num_registers=gmem_o_0.cfg.num_regs_other,
            name="EpilogueTask",
            **task_kwargs,
        )

    if smem_o_0.cfg.single_qkv_instance:
        return create_single_instance_task()
    return create_paired_task()
