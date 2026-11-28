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

"""Load tasks of the latency 1CTA MLA decode kernel: Q/K/V TMA loads."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ......device.schedule import schedule_token_throttle_head
from ......device.stage import WorkStage
from ...device.schedule import (
    runtime_work_tile_skip_if,
    staged_kv_load,
    work_tile_schedule_loop,
)
from .common import MlaDecodeTask


def create_throughput_latency_load_task(
    smem_q,
    smem_kv,
    work_queue: WorkQueue | None,
    schedule_token_throttle,
    cfg,
    *,
    domain,
    smem_page_offsets=None,
    use_page_offsets=False,
    warp_idx=None,
    num_warps=None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the Q/K/V load task.

    The load warp stages Q once, then follows the selected K-before-V cadence.
    Swaps-MMA-AB uses two interleaved K/V streams, while keeps-MMA-AB uses a
    single K stream followed by the deferred V stream.
    """
    page_offsets = smem_page_offsets if use_page_offsets else None
    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    def load_schedule_body(
        smem_q,
        smem_kv,
        smem_page_offsets,
        work_queue,
        schedule_token_throttle,
    ):
        """Shared captured load schedule body for optional page offsets/WQ paths."""
        smem_q.init_load_state()
        smem_kv.init_load_state()
        if smem_page_offsets is not None:
            cached_page_ids = smem_page_offsets.init_read_state()
        else:
            cached_page_ids = None
        reuse_delayed_v_page_ids = (
            smem_page_offsets is not None and not cfg.single_kv_pipe
        )

        def load_kv(
            *,
            head_dim_stages,
            producer_label,
            section,
            page_id_slot=0,
            reuse_cached_page_ids=False,
        ):
            """Select the native paged swaps cache or the existing load path."""
            nonlocal cached_page_ids
            if reuse_delayed_v_page_ids:
                assert not cfg.early_raw_snapshot
                cached_page_ids = staged_kv_load_with_reused_page_ids(
                    smem_kv,
                    head_dim_stages=head_dim_stages,
                    producer_label=producer_label,
                    section=section,
                    smem_page_offsets=smem_page_offsets,
                    cached_page_ids=cached_page_ids,
                    page_id_slot=page_id_slot,
                    consume_offsets=not reuse_cached_page_ids,
                )
            else:
                cached_page_ids = staged_kv_load(
                    smem_kv,
                    whole_kv=cfg.whole_kv_tile,
                    pre_acquire_label=(
                        "snapshot_raw_record"
                        if cfg.early_raw_snapshot
                        and producer_label.startswith("load_k")
                        else None
                    ),
                    head_dim_stages=head_dim_stages,
                    producer_label=producer_label,
                    section=section,
                    smem_page_offsets=smem_page_offsets,
                    cached_page_ids=cached_page_ids,
                )

        with work_tile_schedule_loop(work_queue, skip_if=work_tile_skip_if):
            schedule_token_throttle_head(schedule_token_throttle)
            smem_q.acquire()
            if cfg.q_in_kv:
                # The Q pipeline hands the tile over without bytes; its Q lands
                # in the next two K/V ring slices, ahead of the tile's K.
                smem_q.commit()
                for half in range(2):
                    smem_kv.acquire()
                    smem_kv.load_q_half(half=half)
                    smem_kv.commit()
            else:
                smem_q.load_q()
                smem_q.commit()
            if cfg.serial_swap_reuse:
                load_kv(
                    head_dim_stages=cfg.qk_head_dim_stages,
                    producer_label="load_k0",
                    section=WorkStage.Head,
                    page_id_slot=0,
                )
                with domain_loop(0, domain, 1):
                    load_kv(
                        head_dim_stages=cfg.qk_head_dim_stages,
                        producer_label="load_k1",
                        section=WorkStage.Loop,
                        page_id_slot=1,
                    )
                    load_kv(
                        head_dim_stages=cfg.qk_head_dim_stages,
                        producer_label="load_k0",
                        section=WorkStage.Loop,
                        page_id_slot=0,
                    )
                load_kv(
                    head_dim_stages=cfg.qk_head_dim_stages,
                    producer_label="load_k1",
                    section=WorkStage.Tail,
                    page_id_slot=1,
                )
            else:
                load_kv(
                    head_dim_stages=cfg.qk_head_dim_stages,
                    producer_label="load_k0",
                    section=WorkStage.Head,
                    page_id_slot=0,
                )
                if not cfg.single_kv_pipe:
                    load_kv(
                        head_dim_stages=cfg.qk_head_dim_stages,
                        producer_label="load_k1",
                        section=WorkStage.Head,
                        page_id_slot=1,
                    )

                with domain_loop(0, domain, 1):
                    if cfg.single_kv_pipe:
                        load_kv(
                            head_dim_stages=cfg.qk_head_dim_stages,
                            producer_label="load_k0",
                            section=WorkStage.Loop,
                        )
                        if not cfg.sparse_reuse_kv:
                            load_kv(
                                head_dim_stages=cfg.v_head_dim_stages,
                                producer_label="load_v0",
                                section=WorkStage.Loop,
                            )
                    else:
                        if not cfg.sparse_reuse_kv:
                            load_kv(
                                head_dim_stages=cfg.v_head_dim_stages,
                                producer_label="load_v0",
                                section=WorkStage.Loop,
                                page_id_slot=0,
                                reuse_cached_page_ids=True,
                            )
                        load_kv(
                            head_dim_stages=cfg.qk_head_dim_stages,
                            producer_label="load_k0",
                            section=WorkStage.Loop,
                            page_id_slot=0,
                        )
                        if not cfg.sparse_reuse_kv:
                            load_kv(
                                head_dim_stages=cfg.v_head_dim_stages,
                                producer_label="load_v1",
                                section=WorkStage.Loop,
                                page_id_slot=1,
                                reuse_cached_page_ids=True,
                            )
                        load_kv(
                            head_dim_stages=cfg.qk_head_dim_stages,
                            producer_label="load_k1",
                            section=WorkStage.Loop,
                            page_id_slot=1,
                        )

                if not cfg.sparse_reuse_kv:
                    load_kv(
                        head_dim_stages=cfg.v_head_dim_stages,
                        producer_label="load_v0",
                        section=WorkStage.Tail,
                        page_id_slot=0,
                        reuse_cached_page_ids=True,
                    )
                if not cfg.single_kv_pipe and not cfg.sparse_reuse_kv:
                    load_kv(
                        head_dim_stages=cfg.v_head_dim_stages,
                        producer_label="load_v1",
                        section=WorkStage.Tail,
                        page_id_slot=1,
                        reuse_cached_page_ids=True,
                    )

    @schedule
    def load_schedule(smem_q, smem_kv):
        """Captured load schedule without persistent work queue."""
        load_schedule_body(smem_q, smem_kv, None, None, None)

    @schedule
    def load_wq_schedule(smem_q, smem_kv, work_queue):
        """Captured load schedule with a persistent work queue."""
        load_schedule_body(smem_q, smem_kv, None, work_queue, None)

    @schedule
    def load_wq_throttle_schedule(
        smem_q,
        smem_kv,
        work_queue,
        schedule_token_throttle,
    ):
        """Captured load schedule with persistent work queue throttling."""
        load_schedule_body(
            smem_q,
            smem_kv,
            None,
            work_queue,
            schedule_token_throttle,
        )

    @schedule
    def load_page_offsets_schedule(
        smem_q,
        smem_kv,
        smem_page_offsets,
    ):
        """Captured load schedule with precomputed page offsets."""
        load_schedule_body(
            smem_q,
            smem_kv,
            smem_page_offsets,
            None,
            None,
        )

    @schedule
    def load_page_offsets_wq_schedule(
        smem_q,
        smem_kv,
        smem_page_offsets,
        work_queue,
    ):
        """Captured load schedule with page offsets and persistent work queue."""
        load_schedule_body(
            smem_q,
            smem_kv,
            smem_page_offsets,
            work_queue,
            None,
        )

    @schedule
    def load_page_offsets_wq_throttle_schedule(
        smem_q,
        smem_kv,
        smem_page_offsets,
        work_queue,
        schedule_token_throttle,
    ):
        """Captured load schedule with page offsets, work queue, and throttling."""
        load_schedule_body(
            smem_q,
            smem_kv,
            smem_page_offsets,
            work_queue,
            schedule_token_throttle,
        )

    if page_offsets is None:
        if work_queue is None:
            captured_schedule = load_schedule(smem_q, smem_kv)
        elif schedule_token_throttle is None:
            captured_schedule = load_wq_schedule(smem_q, smem_kv, work_queue)
        else:
            captured_schedule = load_wq_throttle_schedule(
                smem_q,
                smem_kv,
                work_queue,
                schedule_token_throttle,
            )
    elif work_queue is None:
        captured_schedule = load_page_offsets_schedule(
            smem_q,
            smem_kv,
            page_offsets,
        )
    elif schedule_token_throttle is None:
        captured_schedule = load_page_offsets_wq_schedule(
            smem_q,
            smem_kv,
            page_offsets,
            work_queue,
        )
    else:
        captured_schedule = load_page_offsets_wq_throttle_schedule(
            smem_q,
            smem_kv,
            page_offsets,
            work_queue,
            schedule_token_throttle,
        )
    src = []
    if use_page_offsets:
        src.append(smem_page_offsets)
    if work_queue is not None:
        src.append(work_queue)
    dst = [smem_q, smem_kv]
    if schedule_token_throttle is not None:
        dst.append(schedule_token_throttle)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        cfg=cfg,
        warp_idx=cfg.load_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.load_num_warps if num_warps is None else num_warps,
        schedule=captured_schedule,
        num_registers=cfg.kv_load_regs,
        name="LoadTmaTask",
        **kw,
    )


def staged_kv_load_with_reused_page_ids(
    smem_kv,
    *,
    head_dim_stages,
    producer_label,
    section: WorkStage,
    smem_page_offsets,
    cached_page_ids,
    page_id_slot: int,
    consume_offsets: bool,
):
    """Stage one swaps-MMA K/V tile using its K-owned page-ID cache slot."""
    if consume_offsets:
        smem_page_offsets.wait()
        cached_page_ids = smem_page_offsets.read_offsets(
            cached_page_ids=cached_page_ids,
            cache_slot=page_id_slot,
        )
    for stage_idx in range(head_dim_stages):
        smem_kv.acquire()
        getattr(smem_kv, producer_label)(
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )
        smem_kv.commit()
    if consume_offsets:
        smem_page_offsets.release()
    return cached_page_ids
