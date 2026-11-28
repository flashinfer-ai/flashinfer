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

"""Correction tasks of the latency 1CTA MLA decode kernel: O rescale, normalization and partial output."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ...device.schedule import runtime_work_tile_skip_if, work_tile_schedule_loop
from .common import MlaDecodeTask


def create_keeps_mma_ab_correction_task(
    tmem_softmax_local,
    tmem_o,
    tmem_corr,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the single-pipe correction/store task for keeps-MMA-AB."""

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def correction_schedule(
        tmem_softmax_local,
        tmem_o,
        tmem_corr,
        work_queue=None,
    ):
        """Captured keeps-MMA-AB correction schedule for loop and tail."""

        def load_initial_local_stats(local_state):
            """Wait for the keeps-MMA-AB pipe and return its initial stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_initial_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def load_loop_local_stats(local_state):
            """Wait for the keeps-MMA-AB pipe and load loop stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_loop_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def load_tail_local_stats(local_state):
            """Wait for the keeps-MMA-AB pipe and load tail stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_tail_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def correct_loop_o(local_state, tail_o_stage_idx_0, tail_o_stage_idx_1):
            """Rescale one keeps-MMA-AB loop O stage."""
            local_state = load_loop_local_stats(local_state)
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                _inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                _inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=0,
            )
            tmem_corr.correct_loop_and_store(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
            )
            tmem_o.release()
            return local_state, tail_o_stage_idx_0, tail_o_stage_idx_1

        def correct_tail_o(local_state, tail_o_stage_idx_0, tail_o_stage_idx_1):
            """Normalize and store the final keeps-MMA-AB O stage."""
            local_state = load_tail_local_stats(local_state)
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                _inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                _inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=0,
                is_tail=True,
            )
            if cfg.ws_tail_staged_stripes:
                # Read O, release it to the next tile's PV, then finish the tail.
                tmem_corr.correct_tail_and_stage(
                    old_max_arr=old_max_arr,
                    new_max_arr=new_max_arr,
                    sum_arr=sum_arr,
                    inst0_new_max_arr=inst0_new_max_arr,
                    inst0_sum_arr=inst0_sum_arr,
                    inst1_new_max_arr=inst1_new_max_arr,
                    inst1_sum_arr=inst1_sum_arr,
                    o_stage_idx=o_stage_idx,
                    tail_o_stage_idx_0=tail_o_stage_idx_0,
                    tail_o_stage_idx_1=tail_o_stage_idx_1,
                )
                tmem_o.release()
                tmem_corr.store_staged_tail()
                return local_state, tail_o_stage_idx_0, tail_o_stage_idx_1
            tmem_corr.correct_tail_and_store(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
            )
            tmem_o.release()
            return local_state, tail_o_stage_idx_0, tail_o_stage_idx_1

        local_state = tmem_softmax_local.init_stats_state()
        _, tail_o_stage_idx_0, tail_o_stage_idx_1 = tmem_o.init_stage_state()
        tmem_corr.init_store_state()

        def init_work_tile_state():
            local_state = tmem_softmax_local.init_stats_work_tile_state()
            _, tail_stage_0, tail_stage_1 = tmem_o.init_stage_work_tile_state()
            return local_state, tail_stage_0, tail_stage_1

        with work_tile_schedule_loop(
            work_queue,
            skip_if=work_tile_skip_if,
            non_skippable_prelude=(
                init_work_tile_state if work_queue is not None else None
            ),
        ) as (_, work_tile_state):
            if work_queue is not None:
                (
                    local_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = work_tile_state
            # Tail-only global inputs load before the first stats wait.
            tmem_corr.prefetch_tail_inputs()
            local_state = load_initial_local_stats(local_state)
            with domain_loop(0, domain, 1):
                (
                    local_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = correct_loop_o(local_state, tail_o_stage_idx_0, tail_o_stage_idx_1)

            (
                local_state,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = correct_tail_o(local_state, tail_o_stage_idx_0, tail_o_stage_idx_1)
        if cfg.ws_tma_output:
            tmem_corr.drain_output_stores()

    schedule_result = (
        correction_schedule(tmem_softmax_local, tmem_o, tmem_corr)
        if work_queue is None
        else correction_schedule(
            tmem_softmax_local,
            tmem_o,
            tmem_corr,
            work_queue,
        )
    )
    src = [tmem_softmax_local, tmem_o]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_corr],
        cfg=cfg,
        warp_idx=cfg.correction_warp_idx,
        num_warps=cfg.correction_num_warps,
        schedule=schedule_result,
        num_registers=cfg.correction_regs,
        name="CorrectionTask",
        **kw,
    )


def create_throughput_latency_correction_task(
    tmem_softmax_local0,
    tmem_softmax_local1,
    tmem_o,
    tmem_corr0,
    tmem_corr1,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the correction task that normalizes and stores 1CTA output.

    The task consumes local softmax statistics and TMEM O stages from both
    interleaved MMA pipes, applies online-softmax correction, and stores O/LSE
    or split-KV partials.
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def correction_schedule(
        tmem_softmax_local0,
        tmem_softmax_local1,
        tmem_o,
        tmem_corr0,
        tmem_corr1,
        work_queue=None,
    ):
        """Captured correction schedule for loop and tail O-stage draining."""

        def load_initial_local_stats(tmem_softmax_local, local_state):
            """Wait for one softmax pipe and return its initial stat state."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_initial_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def load_loop_local_stats(tmem_softmax_local, local_state):
            """Wait for one softmax pipe and load loop old/new stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_loop_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def load_tail_local_stats(tmem_softmax_local, local_state):
            """Wait for one softmax pipe and load tail sum/new-max stats."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_softmax_local.wait()
            local_state = tmem_softmax_local.load_tail_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_old_max_arr=inst0_old_max_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_old_max_arr=inst1_old_max_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
            )
            tmem_softmax_local.release()
            return local_state

        def correct_loop_o(
            tmem_softmax_local,
            tmem_corr,
            local_state,
            tail_o_stage_idx_0,
            tail_o_stage_idx_1,
            o_stage_inst_idx,
        ):
            """Rescale one loop O accumulator using local softmax stats."""
            local_state = load_loop_local_stats(tmem_softmax_local, local_state)
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                _inst0_old_max_arr,
                inst0_new_max_arr,
                inst0_sum_arr,
                _inst1_old_max_arr,
                inst1_new_max_arr,
                inst1_sum_arr,
            ) = local_state
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=o_stage_inst_idx,
            )
            tmem_corr.correct_loop_and_store(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
            )
            tmem_o.release()
            return local_state, tail_o_stage_idx_0, tail_o_stage_idx_1

        local0_state = tmem_softmax_local0.init_stats_state()
        local1_state = tmem_softmax_local1.init_stats_state()
        _, tail_o_stage_idx_0, tail_o_stage_idx_1 = tmem_o.init_stage_state()
        tmem_corr0.init_store_state()
        tmem_corr1.init_store_state()

        def init_work_tile_state():
            local0_state = tmem_softmax_local0.init_stats_work_tile_state()
            local1_state = tmem_softmax_local1.init_stats_work_tile_state()
            _, tail_stage_0, tail_stage_1 = tmem_o.init_stage_work_tile_state()
            return local0_state, local1_state, tail_stage_0, tail_stage_1

        with work_tile_schedule_loop(
            work_queue,
            skip_if=work_tile_skip_if,
            non_skippable_prelude=(
                init_work_tile_state if work_queue is not None else None
            ),
        ) as (_, work_tile_state):
            if work_queue is not None:
                (
                    local0_state,
                    local1_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = work_tile_state
            # With only one tile per stream there is no correction loop;
            # consume only final statistics, published after both P tiles.
            if not cfg.short_merged_softmax:
                local0_state = load_initial_local_stats(
                    tmem_softmax_local0, local0_state
                )
                local1_state = load_initial_local_stats(
                    tmem_softmax_local1, local1_state
                )

            with domain_loop(0, domain, 1):
                (
                    local0_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = correct_loop_o(
                    tmem_softmax_local0,
                    tmem_corr0,
                    local0_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                    0,
                )
                (
                    local1_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                ) = correct_loop_o(
                    tmem_softmax_local1,
                    tmem_corr1,
                    local1_state,
                    tail_o_stage_idx_0,
                    tail_o_stage_idx_1,
                    1,
                )

            stats0 = load_tail_local_stats(tmem_softmax_local0, local0_state)
            (
                _old_max_arr0,
                _new_max_arr0,
                _sum_arr0,
                _inst0_old_max_arr0,
                inst0_new_max_arr0,
                inst0_sum_arr0,
                _inst1_old_max_arr0,
                _inst1_new_max_arr0,
                _inst1_sum_arr0,
            ) = stats0
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=0,
                is_tail=True,
            )
            # Loop-body correction already rescaled stream 0 to the final
            # stream-0 max before the tail PV0 accumulation. Keep the tail
            # stage index for final stream combination, but avoid a second
            # rescale of the same O0 data.
            del o_stage_idx
            stats1 = load_tail_local_stats(tmem_softmax_local1, local1_state)
            (
                old_max_arr1,
                new_max_arr1,
                sum_arr1,
                _inst0_old_max_arr1,
                _inst0_new_max_arr1,
                _inst0_sum_arr1,
                _inst1_old_max_arr1,
                inst1_new_max_arr1,
                inst1_sum_arr1,
            ) = stats1
            tmem_o.wait()
            (
                o_stage_idx,
                tail_o_stage_idx_0,
                tail_o_stage_idx_1,
            ) = tmem_o.o_stage(
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst_idx=1,
                is_tail=True,
            )
            tmem_corr1.correct_tail_and_store(
                old_max_arr=old_max_arr1,
                new_max_arr=new_max_arr1,
                sum_arr=sum_arr1,
                inst0_new_max_arr=inst0_new_max_arr0,
                inst0_sum_arr=inst0_sum_arr0,
                inst1_new_max_arr=inst1_new_max_arr1,
                inst1_sum_arr=inst1_sum_arr1,
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
            )
            tmem_o.release()
            tmem_o.release()

    captured_schedule = (
        correction_schedule(
            tmem_softmax_local0,
            tmem_softmax_local1,
            tmem_o,
            tmem_corr0,
            tmem_corr1,
        )
        if work_queue is None
        else correction_schedule(
            tmem_softmax_local0,
            tmem_softmax_local1,
            tmem_o,
            tmem_corr0,
            tmem_corr1,
            work_queue,
        )
    )
    src = [tmem_softmax_local0, tmem_softmax_local1, tmem_o]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_corr0, tmem_corr1],
        cfg=cfg,
        warp_idx=cfg.correction_warp_idx,
        num_warps=cfg.correction_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.correction_regs,
        name="CorrectionTask",
        **kw,
    )
