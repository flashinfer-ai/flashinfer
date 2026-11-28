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

"""MMA tasks of the latency 1CTA MLA decode kernel: QK and PV UMMA issue."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ...device.schedule import (
    runtime_work_tile_skip_if,
    staged_pv_mma,
    staged_pv_mma_tmem_p,
    staged_qk_mma,
    work_tile_schedule_loop,
)
from .common import MlaDecodeTask


def create_single_kv_pipe_mma_task(
    smem_q,
    smem_kv,
    tmem_s,
    p_resource,
    tmem_o,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Schedule one QK/PV stream with SMEM P (swap) or TMEM P (keep).

    HEAD computes the first QK; TAIL drains the last PV. Retaining one BF16
    BK128 tile requires PV-before-next-QK; two retained tiles permit overlap.
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)
    swap = cfg.one_insts_kv_swap
    pv_mma = staged_pv_mma if swap else staged_pv_mma_tmem_p
    pv_kwargs = {"whole_kv": cfg.whole_kv_tile}
    if not swap:
        pv_kwargs["ws_pair"] = cfg.ws_pv
    pv_loop = "pv_mma_loop_0" if swap else "pv_mma_loop_tmem_p"
    pv_tail = "pv_mma_tail_0" if swap else "pv_mma_tail_tmem_p"
    v_desc = "v_desc_reused_0" if swap else "v_desc_reused"
    if not cfg.sparse_reuse_kv:
        v_desc = "v_desc_0"
    pv_before_qk = (
        cfg.sparse_reuse_kv
        and not cfg.whole_kv_tile
        and cfg.kv_stages < 2 * cfg.qk_head_dim_stages
    )

    @schedule
    def mma_schedule(
        smem_q,
        smem_kv,
        tmem_s,
        p_resource,
        tmem_o,
        work_queue=None,
    ):
        """Capture identical pipeline ownership for the two P layouts."""
        smem_q.init_descriptor_state()
        smem_kv.init_descriptor_state()
        tmem_s.init_descriptor_state()
        tmem_o.init_mma_state()
        if swap:
            p_resource.init_descriptor_state()
        else:
            p_resource.init_stage_state()

        with work_tile_schedule_loop(work_queue, skip_if=work_tile_skip_if):
            if work_queue is not None:
                tmem_s.reset_softmax_work_tile_state()
                if not swap:
                    p_resource.init_stage_work_tile_state()
            smem_q.wait()
            if cfg.q_in_kv:
                # Both K/V ring slices of the tile's Q, then the copy to TMEM.
                smem_kv.wait()
                smem_kv.wait()
                q_desc, q_desc_rope = smem_kv.q_desc_from_kv()
            else:
                q_desc, q_desc_rope = smem_q.q_desc()
            tmem_s.set_q_desc(q_desc=q_desc, q_desc_rope=q_desc_rope)
            staged_qk_mma(
                smem_kv,
                tmem_s,
                head_dim_stages=cfg.qk_head_dim_stages,
                whole_kv=cfg.whole_kv_tile,
                consumer_label="k_desc_0",
                release_k=not cfg.sparse_reuse_kv,
            )
            if cfg.q_tmem_mma_copy:
                # QK reads Q from TMEM, so the commit-based release arrives after
                # the copy and the first QK; the loader prefetches the next Q
                # during the rest of this tile.
                smem_q.release()
                if cfg.q_in_kv:
                    # The oldest unreleased ring slices are Q's: release them
                    # before any of the K slices QK retains for PV.
                    smem_kv.release()
                    smem_kv.release()
            with domain_loop(0, domain, 1):
                if pv_before_qk:
                    pv_mma(
                        smem_kv,
                        p_resource,
                        tmem_o,
                        head_dim_stages=cfg.v_head_dim_stages,
                        consumer_label=v_desc,
                        reuse_k=True,
                        retained_k_stages=cfg.qk_head_dim_stages,
                        producer_label=pv_loop,
                        **pv_kwargs,
                    )
                staged_qk_mma(
                    smem_kv,
                    tmem_s,
                    head_dim_stages=cfg.qk_head_dim_stages,
                    whole_kv=cfg.whole_kv_tile,
                    consumer_label="k_desc_0",
                    release_k=not cfg.sparse_reuse_kv,
                )
                if not pv_before_qk:
                    pv_mma(
                        smem_kv,
                        p_resource,
                        tmem_o,
                        head_dim_stages=cfg.v_head_dim_stages,
                        consumer_label=v_desc,
                        reuse_k=cfg.sparse_reuse_kv,
                        retained_k_stages=cfg.qk_head_dim_stages,
                        producer_label=pv_loop,
                        **pv_kwargs,
                    )
            if cfg.q_tma_stage:
                # The loaders' TMA-staged Q lives in TMEM and only QK reads it: the
                # commit-based release arrives once the tile's last QK completes,
                # so the next tile's Q staging overlaps the tail PV.
                smem_q.release()
            pv_mma(
                smem_kv,
                p_resource,
                tmem_o,
                head_dim_stages=cfg.v_head_dim_stages,
                consumer_label=v_desc,
                reuse_k=cfg.sparse_reuse_kv,
                retained_k_stages=cfg.qk_head_dim_stages,
                producer_label=pv_tail,
                **pv_kwargs,
            )
            if not (cfg.q_tmem_mma_copy or cfg.q_tma_stage):
                smem_q.release()

    schedule_result = (
        mma_schedule(smem_q, smem_kv, tmem_s, p_resource, tmem_o)
        if work_queue is None
        else mma_schedule(smem_q, smem_kv, tmem_s, p_resource, tmem_o, work_queue)
    )
    src = [smem_q, smem_kv, p_resource]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s, tmem_o],
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx,
        num_warps=cfg.mma_num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_regs,
        name="MmaTask",
        **kw,
    )


def create_throughput_latency_mma_task(
    smem_q,
    smem_kv,
    tmem_s0,
    tmem_s1,
    smem_p0,
    smem_p1,
    tmem_o,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the MMA task for interleaved QK and PV stages.

    The task consumes Q/K/V SMEM descriptors, produces two TMEM score pipes,
    materializes PV output into TMEM O, and drains the final V tiles in tail.
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def mma_schedule(
        smem_q,
        smem_kv,
        tmem_s0,
        tmem_s1,
        smem_p0,
        smem_p1,
        tmem_o,
        work_queue=None,
    ):
        """Captured MMA schedule with K-prefetch and V-tail drain."""
        smem_q.init_descriptor_state()
        smem_kv.init_descriptor_state()
        smem_p0.init_descriptor_state()
        smem_p1.init_descriptor_state()
        tmem_s0.init_descriptor_state()
        tmem_s1.init_descriptor_state()
        tmem_o.init_mma_state()

        with work_tile_schedule_loop(work_queue, skip_if=work_tile_skip_if):
            if work_queue is not None:
                tmem_s0.reset_softmax_work_tile_state()
                tmem_s1.reset_softmax_work_tile_state()
            smem_q.wait()
            q_desc, q_desc_rope = smem_q.q_desc()
            tmem_s0.set_q_desc(q_desc=q_desc, q_desc_rope=q_desc_rope)
            tmem_s1.set_q_desc(q_desc=q_desc, q_desc_rope=q_desc_rope)
            if cfg.serial_swap_reuse:
                staged_qk_mma(
                    smem_kv,
                    tmem_s0,
                    head_dim_stages=cfg.qk_head_dim_stages,
                    consumer_label="k_desc_0",
                    release_k=False,
                )
                with domain_loop(0, domain, 1):
                    staged_pv_mma(
                        smem_kv,
                        smem_p0,
                        tmem_o,
                        head_dim_stages=cfg.v_head_dim_stages,
                        consumer_label="v_desc_reused_0",
                        reuse_k=True,
                        retained_k_stages=cfg.qk_head_dim_stages,
                        producer_label="pv_mma_loop_0",
                    )
                    staged_qk_mma(
                        smem_kv,
                        tmem_s1,
                        head_dim_stages=cfg.qk_head_dim_stages,
                        consumer_label="k_desc_1",
                        release_k=False,
                    )
                    staged_pv_mma(
                        smem_kv,
                        smem_p1,
                        tmem_o,
                        head_dim_stages=cfg.v_head_dim_stages,
                        consumer_label="v_desc_reused_1",
                        reuse_k=True,
                        retained_k_stages=cfg.qk_head_dim_stages,
                        producer_label="pv_mma_loop_1",
                    )
                    staged_qk_mma(
                        smem_kv,
                        tmem_s0,
                        head_dim_stages=cfg.qk_head_dim_stages,
                        consumer_label="k_desc_0",
                        release_k=False,
                    )
                staged_pv_mma(
                    smem_kv,
                    smem_p0,
                    tmem_o,
                    head_dim_stages=cfg.v_head_dim_stages,
                    consumer_label="v_desc_reused_0",
                    reuse_k=True,
                    retained_k_stages=cfg.qk_head_dim_stages,
                    producer_label="pv_mma_tail_0",
                )
                staged_qk_mma(
                    smem_kv,
                    tmem_s1,
                    head_dim_stages=cfg.qk_head_dim_stages,
                    consumer_label="k_desc_1",
                    release_k=False,
                )
                staged_pv_mma(
                    smem_kv,
                    smem_p1,
                    tmem_o,
                    head_dim_stages=cfg.v_head_dim_stages,
                    consumer_label="v_desc_reused_1",
                    reuse_k=True,
                    retained_k_stages=cfg.qk_head_dim_stages,
                    producer_label="pv_mma_tail_1",
                )
            else:
                staged_qk_mma(
                    smem_kv,
                    tmem_s0,
                    head_dim_stages=cfg.qk_head_dim_stages,
                    consumer_label="k_desc_0",
                    release_k=not cfg.sparse_reuse_kv,
                )
                staged_qk_mma(
                    smem_kv,
                    tmem_s1,
                    head_dim_stages=cfg.qk_head_dim_stages,
                    consumer_label="k_desc_1",
                    release_k=not cfg.sparse_reuse_kv,
                )
                with domain_loop(0, domain, 1):
                    tmem_s0.acquire()
                    staged_pv_mma(
                        smem_kv,
                        smem_p0,
                        tmem_o,
                        head_dim_stages=cfg.v_head_dim_stages,
                        consumer_label=(
                            "v_desc_reused" if cfg.sparse_reuse_kv else "v_desc_0"
                        ),
                        reuse_k=cfg.sparse_reuse_kv,
                        retained_k_stages=cfg.qk_head_dim_stages,
                        producer_label="pv_mma_loop_0",
                    )
                    staged_qk_mma(
                        smem_kv,
                        tmem_s0,
                        head_dim_stages=cfg.qk_head_dim_stages,
                        consumer_label="k_desc_0",
                        release_k=not cfg.sparse_reuse_kv,
                        include_acquire=False,
                    )
                    tmem_s1.acquire()
                    staged_pv_mma(
                        smem_kv,
                        smem_p1,
                        tmem_o,
                        head_dim_stages=cfg.v_head_dim_stages,
                        consumer_label=(
                            "v_desc_reused" if cfg.sparse_reuse_kv else "v_desc_1"
                        ),
                        reuse_k=cfg.sparse_reuse_kv,
                        retained_k_stages=cfg.qk_head_dim_stages,
                        producer_label="pv_mma_loop_1",
                    )
                    staged_qk_mma(
                        smem_kv,
                        tmem_s1,
                        head_dim_stages=cfg.qk_head_dim_stages,
                        consumer_label="k_desc_1",
                        release_k=not cfg.sparse_reuse_kv,
                        include_acquire=False,
                    )
                staged_pv_mma(
                    smem_kv,
                    smem_p0,
                    tmem_o,
                    head_dim_stages=cfg.v_head_dim_stages,
                    consumer_label=(
                        "v_desc_reused" if cfg.sparse_reuse_kv else "v_desc_0"
                    ),
                    reuse_k=cfg.sparse_reuse_kv,
                    retained_k_stages=cfg.qk_head_dim_stages,
                    producer_label="pv_mma_tail_0",
                )
                staged_pv_mma(
                    smem_kv,
                    smem_p1,
                    tmem_o,
                    head_dim_stages=cfg.v_head_dim_stages,
                    consumer_label=(
                        "v_desc_reused" if cfg.sparse_reuse_kv else "v_desc_1"
                    ),
                    reuse_k=cfg.sparse_reuse_kv,
                    retained_k_stages=cfg.qk_head_dim_stages,
                    producer_label="pv_mma_tail_1",
                )
            smem_q.release()

    captured_schedule = (
        mma_schedule(
            smem_q,
            smem_kv,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
        )
        if work_queue is None
        else mma_schedule(
            smem_q,
            smem_kv,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
            work_queue,
        )
    )
    src = [smem_q, smem_kv, smem_p0, smem_p1]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s0, tmem_s1, tmem_o],
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx,
        num_warps=cfg.mma_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_regs,
        name="MmaTask",
        **kw,
    )
