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

"""Softmax tasks of the throughput 2CTA MLA decode kernel."""

from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ...device.schedule import captured_loop_bounds
from ..resources import SmemPResource, TmemCorrResource, TmemSResource
from .common import MlaTask, capture_clc_work_tile_body
from ..work_queue import MlaWorkQueue


def create_softmax_task(
    tmem_s: TmemSResource,
    tmem_corr: TmemCorrResource,
    smem_p: SmemPResource,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    page_offsets=None,
    warp_idx: int = 0,
    name: str = "SoftmaxTask",
    softmax_group_id: int = 0,
    **task_kwargs,
) -> Task:
    """Create the Softmax task (warps 0-3, 4 warps, 192 regs).

    domain_start=0: processes all k_tile_count S tiles.

    LOOP: consume S, compute softmax, produce correction factors + P.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    def softmax_prelude(tmem_s):
        """Create softmax register arrays before the dynamic skip guard."""
        init_softmax_state = (
            tmem_s.init_softmax_state_odd
            if softmax_group_id == 1
            else tmem_s.init_softmax_state
        )
        return init_softmax_state()

    def softmax_body(tmem_s, tmem_corr, smem_p, softmax_state, page_offsets=None):
        """Consume S tiles, materialize P, and publish correction factors."""
        load_s = (
            tmem_s.load_s_odd
            if softmax_group_id == 1
            else tmem_s.load_s_cached
            if page_offsets is not None
            else tmem_s.load_s
        )
        finish_softmax = (
            tmem_s.finish_softmax_odd
            if softmax_group_id == 1
            else tmem_s.finish_softmax
        )
        finish_row_sum = (
            tmem_s.finish_row_sum_odd
            if softmax_group_id == 1
            else tmem_s.finish_row_sum
        )
        store_corr = (
            tmem_corr.store_corr_odd if softmax_group_id == 1 else tmem_corr.store_corr
        )
        store_p = smem_p.store_p_odd if softmax_group_id == 1 else smem_p.store_p
        (
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ) = softmax_state

        tmem_s.bind_sparse_request()
        with domain_loop(loop_start, loop_end, loop_step):
            mask_kwargs = {}
            if page_offsets is not None:
                page_offsets.wait()
                mask_low, mask_high = page_offsets.read_sparse_mask()
                page_offsets.release()
                mask_kwargs = {
                    "cached_mask_low": mask_low,
                    "cached_mask_high": mask_high,
                }
            tmem_s.wait()
            if softmax_group_id == 1:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = load_s(
                    qk_acc_regs_odd=qk_acc_regs,
                    row_max_odd=row_max,
                    row_sum_odd=row_sum,
                    row_sum_out_odd=row_sum_out,
                    row_max_new_odd=row_max_new,
                    correction_factor_out_odd=correction_factor_out,
                    no_correction_out_odd=no_correction_out,
                )
            else:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = load_s(
                    qk_acc_regs=qk_acc_regs,
                    row_max=row_max,
                    row_sum=row_sum,
                    row_sum_out=row_sum_out,
                    row_max_new=row_max_new,
                    correction_factor_out=correction_factor_out,
                    no_correction_out=no_correction_out,
                    **mask_kwargs,
                )
            tmem_s.release()
            if softmax_group_id == 1:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = finish_softmax(
                    qk_acc_regs_odd=qk_acc_regs,
                    row_max_odd=row_max,
                    row_sum_odd=row_sum,
                    row_sum_out_odd=row_sum_out,
                    row_max_new_odd=row_max_new,
                    correction_factor_out_odd=correction_factor_out,
                    no_correction_out_odd=no_correction_out,
                )
            else:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = finish_softmax(
                    qk_acc_regs=qk_acc_regs,
                    row_max=row_max,
                    row_sum=row_sum,
                    row_sum_out=row_sum_out,
                    row_max_new=row_max_new,
                    correction_factor_out=correction_factor_out,
                    no_correction_out=no_correction_out,
                )
            smem_p.acquire()
            if softmax_group_id == 1:
                store_p(qk_acc_regs_odd=qk_acc_regs)
            else:
                store_p(qk_acc_regs=qk_acc_regs)
            smem_p.commit()
            if softmax_group_id == 1:
                row_sum, row_sum_out = finish_row_sum(
                    qk_acc_regs_odd=qk_acc_regs,
                    row_sum_odd=row_sum,
                    correction_factor_out_odd=correction_factor_out,
                )
            else:
                row_sum, row_sum_out = finish_row_sum(
                    qk_acc_regs=qk_acc_regs,
                    row_sum=row_sum,
                    correction_factor_out=correction_factor_out,
                )
            tmem_corr.acquire()
            if softmax_group_id == 1:
                store_corr(
                    row_sum_out_odd=row_sum_out,
                    row_max_new_odd=row_max_new,
                    correction_factor_out_odd=correction_factor_out,
                    no_correction_out_odd=no_correction_out,
                )
            else:
                store_corr(
                    row_sum_out=row_sum_out,
                    row_max_new=row_max_new,
                    correction_factor_out=correction_factor_out,
                    no_correction_out=no_correction_out,
                )
            tmem_corr.commit()

    def capture_softmax(tmem_s, tmem_corr, smem_p, work_queue, page_offsets):
        capture_clc_work_tile_body(
            work_queue,
            lambda softmax_state: softmax_body(
                tmem_s, tmem_corr, smem_p, softmax_state, page_offsets
            ),
            lambda: softmax_prelude(tmem_s),
            use_clc_dynamic=use_clc_dynamic,
        )

    @schedule
    def softmax_schedule(tmem_s, tmem_corr, smem_p, work_queue=None):
        capture_softmax(tmem_s, tmem_corr, smem_p, work_queue, None)

    @schedule
    def sparse_softmax_schedule(
        tmem_s, tmem_corr, smem_p, page_offsets, work_queue=None
    ):
        capture_softmax(tmem_s, tmem_corr, smem_p, work_queue, page_offsets)

    if page_offsets is None:
        schedule_result = (
            softmax_schedule(tmem_s, tmem_corr, smem_p)
            if work_queue is None
            else softmax_schedule(tmem_s, tmem_corr, smem_p, work_queue)
        )
    else:
        schedule_result = (
            sparse_softmax_schedule(tmem_s, tmem_corr, smem_p, page_offsets)
            if work_queue is None
            else sparse_softmax_schedule(
                tmem_s, tmem_corr, smem_p, page_offsets, work_queue
            )
        )

    src = [tmem_s]
    if page_offsets is not None:
        src.append(page_offsets)
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_corr, smem_p],
        warp_idx=warp_idx,
        num_warps=4,
        schedule=schedule_result,
        name=name,
        **task_kwargs,
    )
