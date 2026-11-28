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

"""Softmax tasks of the latency 1CTA MLA decode kernel."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ......device.stage import WorkStage
from ...device.schedule import runtime_work_tile_skip_if, work_tile_schedule_loop
from .common import MlaDecodeTask


def create_keeps_mma_ab_softmax_task(
    tmem_s,
    tmem_softmax_local,
    tmem_p,
    tmem_softmax_global,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the single-pipe softmax/P task for the keeps-MMA-AB path.

    Loop correction rescales O from the old/new maxima only, which are final
    once update_softmax returns, so publish them before materializing P: the
    correction then overlaps P materialization instead of following it, and
    PV no longer waits for both in sequence. The running sum still updates
    after P, before the tail stats that normalize the output.

    Loop stats therefore carry the previous tile's running sum. That is safe
    only because the Keep-AB loop correction (``correct_loop_and_store``)
    reads the maxima alone and Keep-AB never takes cluster finishing, which
    configure_sparse_mla rejects; a loop correction that reads the sum needs
    ``store_stats_before_p=False``.
    """
    if cfg.use_cluster_reduction:
        raise ValueError("Keep-AB early loop stats require non-cluster finishing")
    return create_throughput_latency_softmax_task_impl(
        tmem_s,
        tmem_softmax_local,
        tmem_p,
        tmem_softmax_global,
        work_queue,
        cfg,
        inst_id=0,
        domain=domain,
        store_stats_before_p=True,
        task_name="SoftmaxTask",
        task_class=task_class,
        **kw,
    )


def create_short_merged_softmax_task(s0, s1, l0, l1, p0, p1, g0, g1, cfg, **kw):
    @schedule
    def merged(s0, s1, l0, l1, p0, p1, g0, g1):
        def process(s, local, p, global_stats):
            old, total, maximum, local_sum, scores = s.init_softmax_state()
            p.init_materialize_state()
            s.wait()
            old, total, maximum, local_sum, scores = s.update_softmax(
                old_max_arr=old,
                sum_arr=total,
                new_max_arr=maximum,
                local_sum_arr=local_sum,
                s_arr=scores,
                section=WorkStage.Loop,
            )
            s.release()
            p.acquire()
            p.materialize_p(new_max_arr=maximum, s_arr=scores, local_sum_arr=local_sum)
            p.commit()
            global_stats.track_global()
            old, total, maximum, local_sum, scores = s.update_softmax_sum(
                old_max_arr=old,
                sum_arr=total,
                new_max_arr=maximum,
                local_sum_arr=local_sum,
                s_arr=scores,
            )
            return old, maximum, total

        def publish_final(local, state):
            old, maximum, total = state
            local.acquire()
            local.store_loop_stats(
                old_max_arr=old, new_max_arr=maximum, sum_arr=total, inst_idx=1
            )
            local.commit()

        with domain_loop(0, 1, 1):
            state0 = process(s0, l0, p0, g0)
            state1 = process(s1, l1, p1, g1)
            publish_final(l0, state0)
            publish_final(l1, state1)

    return MlaDecodeTask(
        src_resources=[s0, s1],
        dst_resources=[l0, l1, p0, p1, g0, g1],
        cfg=cfg,
        warp_idx=0,
        num_warps=4,
        num_registers=cfg.softmax_regs,
        schedule=merged(s0, s1, l0, l1, p0, p1, g0, g1),
        name="ShortMergedSoftmaxTask",
        domain_bias=1,
        **kw,
    )


def create_throughput_latency_softmax0_task(
    tmem_s0,
    tmem_softmax_local0,
    smem_p0,
    tmem_softmax_global0,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the softmax/P producer task for score instance 0."""
    return create_throughput_latency_softmax_task_impl(
        tmem_s0,
        tmem_softmax_local0,
        smem_p0,
        tmem_softmax_global0,
        work_queue,
        cfg,
        inst_id=0,
        domain=domain,
        task_class=task_class,
        **kw,
    )


def create_throughput_latency_softmax1_task(
    tmem_s1,
    tmem_softmax_local1,
    smem_p1,
    tmem_softmax_global1,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the softmax/P producer task for score instance 1."""
    return create_throughput_latency_softmax_task_impl(
        tmem_s1,
        tmem_softmax_local1,
        smem_p1,
        tmem_softmax_global1,
        work_queue,
        cfg,
        inst_id=1,
        domain=domain,
        task_class=task_class,
        **kw,
    )


def create_throughput_latency_softmax_task_impl(
    tmem_s,
    tmem_softmax_local,
    p_resource,
    tmem_softmax_global,
    work_queue: WorkQueue | None,
    cfg,
    *,
    inst_id: int,
    domain,
    store_stats_before_p: bool = True,
    task_name: str | None = None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create one softmax task for a throughput-latency 1CTA score pipe.

    The task consumes TMEM scores, updates online-softmax state, materializes
    P for PV MMA, and stores per-iteration statistics for correction.
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def softmax_schedule(
        tmem_s,
        tmem_softmax_local,
        p_resource,
        tmem_softmax_global,
        work_queue=None,
    ):
        """Captured softmax schedule for one of the two interleaved instances."""
        (
            old_max_arr,
            sum_arr,
            new_max_arr,
            local_sum_arr,
            s_arr,
        ) = tmem_s.init_softmax_state()
        if store_stats_before_p and hasattr(p_resource, "init_materialize_state"):
            # SMEM P prepares its stage view here; TMEM P needs no setup.
            p_resource.init_materialize_state()

        def store_stats(inst_idx: int):
            """Store local online-softmax stats for correction."""
            tmem_softmax_local.acquire()
            tmem_softmax_local.store_loop_stats(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst_idx=inst_idx,
            )
            tmem_softmax_local.commit()

        def init_work_tile_state():
            return tmem_s.init_softmax_work_tile_state()

        with work_tile_schedule_loop(
            work_queue,
            skip_if=work_tile_skip_if,
            non_skippable_prelude=(
                init_work_tile_state if work_queue is not None else None
            ),
        ) as (_, work_tile_state):
            if work_queue is not None:
                (
                    old_max_arr,
                    sum_arr,
                    new_max_arr,
                    local_sum_arr,
                    s_arr,
                ) = work_tile_state
            with domain_loop(0, domain, 1) as d:
                tmem_s.wait()
                (
                    old_max_arr,
                    sum_arr,
                    new_max_arr,
                    local_sum_arr,
                    s_arr,
                ) = tmem_s.update_softmax(
                    old_max_arr=old_max_arr,
                    sum_arr=sum_arr,
                    new_max_arr=new_max_arr,
                    local_sum_arr=local_sum_arr,
                    s_arr=s_arr,
                    section=WorkStage.Loop,
                )
                tmem_s.release()
                if store_stats_before_p:
                    store_stats(0)
                # AsyncUmma protects P until PV MMA releases this stage.
                p_resource.acquire()
                p_resource.materialize_p(
                    new_max_arr=new_max_arr,
                    s_arr=s_arr,
                    local_sum_arr=local_sum_arr,
                )
                p_resource.commit()
                tmem_softmax_global.track_global()
                (
                    old_max_arr,
                    sum_arr,
                    new_max_arr,
                    local_sum_arr,
                    s_arr,
                ) = tmem_s.update_softmax_sum(
                    old_max_arr=old_max_arr,
                    sum_arr=sum_arr,
                    new_max_arr=new_max_arr,
                    local_sum_arr=local_sum_arr,
                    s_arr=s_arr,
                )
                if not store_stats_before_p:
                    store_stats(0)
                with d.last_iter():
                    store_stats(1)

    captured_schedule = (
        softmax_schedule(
            tmem_s,
            tmem_softmax_local,
            p_resource,
            tmem_softmax_global,
        )
        if work_queue is None
        else softmax_schedule(
            tmem_s,
            tmem_softmax_local,
            p_resource,
            tmem_softmax_global,
            work_queue,
        )
    )
    src = [tmem_s]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_softmax_local, p_resource, tmem_softmax_global],
        cfg=cfg,
        warp_idx=cfg.softmax0_warp_idx if inst_id == 0 else cfg.softmax1_warp_idx,
        num_warps=cfg.softmax0_num_warps if inst_id == 0 else cfg.softmax1_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.softmax_regs,
        name=task_name or f"Softmax{inst_id}Task",
        **kw,
    )
