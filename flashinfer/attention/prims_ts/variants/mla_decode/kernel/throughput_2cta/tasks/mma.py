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

"""MMA tasks of the throughput 2CTA MLA decode kernel: QK and PV UMMA issue."""

from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ...device.schedule import (
    captured_loop_bounds,
    staged_pv_mma,
    staged_pv_mma_v_tile,
    staged_pv_mma_v_tile_per_n,
    staged_qk_mma,
    staged_qk_mma_k_tile,
    work_queue_tail,
)
from ..resources import (
    SmemKResource,
    SmemKVResource,
    SmemPResource,
    SmemQResource,
    SmemVResource,
    TmemOResource,
    TmemSResource,
    WorkThrottleBarrierResource,
)
from .common import MlaTask, capture_clc_work_tile_body
from ..work_queue import MlaWorkQueue


def create_mma_pv_direct_task(
    smem_v: SmemVResource,
    smem_p: SmemPResource,
    tmem_o: TmemOResource,
    iterations_pv: int = 4,
    iterations_pv_k: int = 2,
    iterations_pv_n: int = 2,
    per_n_o_pipeline: bool = False,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 PV task with one V stage per domain-loop iteration."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def mma_pv_direct_schedule(smem_v, smem_p, tmem_o, work_queue=None):
        """Consume one P/V pair and publish one O stage per K tile."""
        with domain_loop(loop_start, loop_end, loop_step):
            if per_n_o_pipeline:
                staged_pv_mma_v_tile_per_n(
                    smem_v,
                    smem_p,
                    tmem_o,
                    iterations_pv_k=iterations_pv_k,
                    iterations_pv_n=iterations_pv_n,
                )
            else:
                staged_pv_mma_v_tile(smem_v, smem_p, tmem_o, iterations_pv)
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_pv_direct_schedule(smem_v, smem_p, tmem_o)
        if work_queue is None
        else mma_pv_direct_schedule(smem_v, smem_p, tmem_o, work_queue)
    )

    src = [smem_v, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_o],
        warp_idx=11,
        num_warps=1,
        schedule=schedule_result,
        name="MmaPvTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_pv_task(
    smem_kv: SmemKVResource,
    smem_p: SmemPResource,
    tmem_o: TmemOResource,
    iterations_pv: int = 8,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 PV-only MMA task (warp 11).

    The PV task consumes P[n-1]/V[n-1] while QK produces S[n], then handles the
    final P/V tile in TAIL. This matches the existing correction schedule.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)

    @schedule
    def mma_pv_schedule(smem_kv, smem_p, tmem_o, work_queue=None):
        """Consume delayed V and P stages and accumulate O."""
        with domain_loop(loop_start, loop_end, loop_step):
            staged_pv_mma(smem_kv, smem_p, tmem_o, iterations_pv)

        staged_pv_mma(
            smem_kv,
            smem_p,
            tmem_o,
            iterations_pv,
            is_tail=True,
        )
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_pv_schedule(smem_kv, smem_p, tmem_o)
        if work_queue is None
        else mma_pv_schedule(smem_kv, smem_p, tmem_o, work_queue)
    )

    src = [smem_kv, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_o],
        warp_idx=11,
        num_warps=1,
        schedule=schedule_result,
        name="MmaPvTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_qk_direct_task(
    smem_q: SmemQResource,
    smem_k: SmemKResource,
    tmem_s: TmemSResource,
    iterations_qk: int = 5,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 QK task with one K stage per domain-loop iteration."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def mma_qk_direct_schedule(smem_q, smem_k, tmem_s, work_queue=None):
        """Consume Q once and publish one S stage per K tile."""
        smem_q.wait()
        smem_q.q_desc()
        with domain_loop(loop_start, loop_end, loop_step):
            staged_qk_mma_k_tile(smem_k, tmem_s, iterations_qk)
        smem_q.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_qk_direct_schedule(smem_q, smem_k, tmem_s)
        if work_queue is None
        else mma_qk_direct_schedule(smem_q, smem_k, tmem_s, work_queue)
    )

    src = [smem_q, smem_k]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s],
        warp_idx=8,
        num_warps=1,
        schedule=schedule_result,
        name="MmaQkTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_qk_task(
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    tmem_s: TmemSResource,
    iterations_qk: int = 9,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 QK-only MMA task (warp 8).

    QK stays one k-tile ahead of PV. Keeping QK and PV on separate warps avoids
    serializing the two UMMA issue streams while preserving the existing K-before-V
    TMA cadence and one-softmax schedule.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)

    @schedule
    def mma_qk_schedule(smem_q, smem_kv, tmem_s, work_queue=None):
        """Consume Q/K stages and publish S for every k-tile."""
        smem_q.wait()
        smem_q.q_desc()
        staged_qk_mma(smem_kv, tmem_s, iterations_qk)

        with domain_loop(loop_start, loop_end, loop_step):
            staged_qk_mma(smem_kv, tmem_s, iterations_qk)

        smem_q.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_qk_schedule(smem_q, smem_kv, tmem_s)
        if work_queue is None
        else mma_qk_schedule(smem_q, smem_kv, tmem_s, work_queue)
    )

    src = [smem_q, smem_kv]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s],
        warp_idx=8,
        num_warps=1,
        schedule=schedule_result,
        name="MmaQkTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_task(
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    smem_p: SmemPResource,
    tmem_s: TmemSResource,
    tmem_o: TmemOResource,
    iterations_qk: int = 9,
    iterations_pv: int = 8,
    work_queue: MlaWorkQueue = None,
    work_throttle: WorkThrottleBarrierResource = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the MMA task (warp 8, 1 warp, 96 regs).

    domain_start=1: HEAD handles first QK MMA, LOOP handles PV+QK pairs.
    run_only_on_cta_id=0 keeps the schedule on the CTA-pair leader, matching
    the 2CTA UMMA contract.

    HEAD: consume Q, QK MMA for k-tile[0] -> S[0].
    LOOP: PV MMA for k-tile[n-1] -> O[n-1], then QK MMA for k-tile[n] -> S[n].
    TAIL: PV MMA for last k-tile -> O[last], release Q.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    def mma_body(
        smem_q,
        smem_kv,
        smem_p,
        tmem_s,
        tmem_o,
        work_throttle=None,
    ):
        """Run QK one tile ahead of PV and keep UMMA on the leader CTA."""
        # HEAD: wait Q once and compute S[0] from K[0].
        smem_q.wait()
        if work_throttle is not None:
            # The leader MMA reaching Q proves that this cluster has started
            # the current tile.  Permit the scheduler to prepare one more work
            # ID without relying on CTA-private resource-ownership fields.
            work_throttle.try_acquire()
            work_throttle.acquire()
            work_throttle.commit()
        smem_q.q_desc()
        staged_qk_mma(smem_kv, tmem_s, iterations_qk)

        with domain_loop(loop_start, loop_end, loop_step):
            # LOOP: compute S[n] before PV[n-1]. This gives softmax the QK
            # instruction window to produce P[n-1] before PV consumes it.
            staged_qk_mma(smem_kv, tmem_s, iterations_qk)
            staged_pv_mma(
                smem_kv,
                smem_p,
                tmem_o,
                iterations_pv,
            )

        # TAIL: finish PV[last], then release the Q descriptor stage.
        staged_pv_mma(
            smem_kv,
            smem_p,
            tmem_o,
            iterations_pv,
            is_tail=True,
        )
        smem_q.release()

    @schedule
    def mma_schedule(
        smem_q,
        smem_kv,
        smem_p,
        tmem_s,
        tmem_o,
        work_queue=None,
        work_throttle=None,
    ):
        """Capture one active MMA tile and unconditional queue progress."""

        capture_clc_work_tile_body(
            work_queue,
            lambda: mma_body(
                smem_q,
                smem_kv,
                smem_p,
                tmem_s,
                tmem_o,
                work_throttle,
            ),
            use_clc_dynamic=use_clc_dynamic,
        )

    if work_queue is None:
        captured_schedule = mma_schedule(smem_q, smem_kv, smem_p, tmem_s, tmem_o)
    elif work_throttle is None:
        captured_schedule = mma_schedule(
            smem_q,
            smem_kv,
            smem_p,
            tmem_s,
            tmem_o,
            work_queue,
        )
    else:
        captured_schedule = mma_schedule(
            smem_q,
            smem_kv,
            smem_p,
            tmem_s,
            tmem_o,
            work_queue,
            work_throttle,
        )

    src = [smem_q, smem_kv, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    dst = [tmem_s, tmem_o]
    if work_throttle is not None:
        dst.append(work_throttle)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=8,
        num_warps=1,
        schedule=captured_schedule,
        name="MmaTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )
