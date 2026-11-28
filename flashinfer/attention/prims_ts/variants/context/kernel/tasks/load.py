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

"""Load tasks of the context kernel: Q/K/V TMA loads."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ..resources.gmem_qkv import GmemQKVResource
from ..resources.smem_kv import SmemKVResource, SmemPageOffsetsKvResource
from ..resources.smem_mu import SmemMuResource
from ..resources.smem_q import SmemQResource
from ..resources.smem_sage_k_scales import SmemSageKScalesResource
from .common import (
    captured_loop_bounds,
    packed_context_skip_predicate,
    schedule_with_work_queue,
    src_resources,
    work_tile_schedule_loop,
)
from typing import Any


def create_load_task(
    gmem_qkv: GmemQKVResource,
    smem_q: SmemQResource,
    smem_k_or_kv: SmemKVResource,
    smem_v: SmemKVResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    smem_page_offsets_kv: SmemPageOffsetsKvResource | None = None,
    smem_page_offsets_v: SmemPageOffsetsKvResource | None = None,
    smem_mu: SmemMuResource | None = None,
    smem_sage_k: SmemSageKScalesResource | None = None,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp TMA load task.

    ``smem_mu`` (VC-Attention) stages one bf16 tile-mean operand per K/V tile
    beside V.

    When ``smem_page_offsets_kv`` is provided, each K/V TMA load consumes page
    IDs prefetched by the auxiliary warp through the ordinary asynchronous
    page-offset pipeline. When ``smem_sage_k`` is provided, the warp also
    stages each K tile's Sage ``sfK`` words after the tile's K load.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = packed_context_skip_predicate(work_queue)
    src = src_resources(gmem_qkv, work_queue=work_queue)
    # One K/V resource when they share a buffer, two when their dtypes split it.
    split_kv = smem_k_or_kv.cfg.split_kv_pipelines
    if split_kv and smem_v is None:
        raise ValueError("split K/V staging requires a separate V buffer")
    kv_resources = (smem_k_or_kv, smem_v) if split_kv else (smem_k_or_kv,)
    dst = [smem_q, *kv_resources]
    if smem_mu is not None:
        dst.append(smem_mu)
    if smem_page_offsets_kv is not None:
        src.append(smem_page_offsets_kv)
    if smem_page_offsets_v is not None:
        src.append(smem_page_offsets_v)
    if smem_sage_k is not None:
        dst.append(smem_sage_k)
    if smem_q.cfg.single_qkv_instance and smem_q.cfg.has_tmem_p_pipeline:
        num_head_dim_stages_k = smem_k_or_kv.cfg.num_head_dim_stages_k
        num_head_dim_stages_v = smem_k_or_kv.cfg.num_head_dim_stages_v

        if smem_page_offsets_v is not None:
            if smem_page_offsets_kv is None:
                raise ValueError("a V page window requires a matching K page window")
            pages_per_tile = (
                smem_k_or_kv.cfg.kv_tile_n // smem_k_or_kv.cfg.num_tokens_per_page
            )
            page_window_period = (
                smem_k_or_kv.cfg.page_table_window_entries // pages_per_tile
            )
            if (
                not isinstance(loop_start, int)
                or not isinstance(loop_end, int)
                or not isinstance(loop_step, int)
                or loop_start != 0
                or loop_step != 1
                or loop_end < page_window_period
                or loop_end % page_window_period != 0
            ):
                raise ValueError(
                    "reused page windows require a compile-time K/V domain "
                    "divisible by the topology-derived page-window period"
                )

            def load_reused_page_windows_schedule_body(
                gqkv: GmemQKVResource,
                sq: SmemQResource,
                sk: SmemKVResource,
                sv: SmemKVResource,
                spok: SmemPageOffsetsKvResource,
                spov: SmemPageOffsetsKvResource,
                wq: WorkQueue | None,
            ) -> None:
                """Load staged K/V while retaining each page-ID window."""
                sq.init_load_state()
                sk.init_load_state()
                sv.init_load_state()
                spok.init_read_state()
                cached_v_page_ids = spov.init_cached_read_state()

                with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                    (
                        _seq_coord,
                        head_coord,
                        kv_head_coord,
                        _head_coord_kv,
                        batch_coord,
                        seq_coord_q,
                        cuseqlen_q,
                        cuseqlen_k,
                        seqlen_q,
                        seqlen_k,
                        kv_tile_start,
                        kv_request_begin,
                        kv_page_idx_lb,
                        kv_page_idx_ub,
                    ) = gqkv.compute_coords()
                    sq.acquire()
                    sq.tma_load(
                        seq_coord_q=seq_coord_q,
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_q=cuseqlen_q,
                        seqlen_q=seqlen_q,
                        inst_idx=0,
                    )
                    sq.commit()

                    def load_k_tile(*, tile_offset: int) -> None:
                        """Issue the K TMA loads for one tile across the K head-dim stages."""
                        for head_dim_stage_idx in range(num_head_dim_stages_k):
                            sk.try_acquire()
                            sk.acquire()
                            sk.k_load_stage(
                                stage_id=head_dim_stage_idx,
                                tile_offset=tile_offset,
                                kv_head_coord=kv_head_coord,
                                batch_coord=batch_coord,
                                cuseqlen_k=cuseqlen_k,
                                seqlen_k=seqlen_k,
                                kv_tile_start=kv_tile_start,
                                kv_request_begin=kv_request_begin,
                                kv_page_idx_lb=kv_page_idx_lb,
                                kv_page_idx_ub=kv_page_idx_ub,
                            )
                            sk.commit()

                    def cache_v_tile(*, tile_offset: int) -> None:
                        nonlocal cached_v_page_ids
                        cached_v_page_ids = spov.cache_tile_page_ids(
                            cached_page_ids=cached_v_page_ids,
                            kv_tile_start=kv_tile_start,
                            tile_offset=tile_offset,
                        )

                    def load_v_tile(
                        *, tile_offset: int, reuse_cached_page_ids: bool = False
                    ) -> None:
                        """Issue the V TMA loads for one tile across the V head-dim stages.

                        With ``reuse_cached_page_ids`` the loads reuse page IDs staged by
                        ``cache_v_tile`` instead of re-reading the page table.
                        """
                        for head_dim_stage_idx in range(num_head_dim_stages_v):
                            sv.try_acquire()
                            sv.acquire()
                            if reuse_cached_page_ids:
                                sv.v_load_stage_cached(
                                    cached_v_page_ids=cached_v_page_ids,
                                    stage_id=head_dim_stage_idx,
                                    tile_offset=tile_offset,
                                    kv_head_coord=kv_head_coord,
                                    batch_coord=batch_coord,
                                    cuseqlen_k=cuseqlen_k,
                                    seqlen_k=seqlen_k,
                                    kv_tile_start=kv_tile_start,
                                    kv_request_begin=kv_request_begin,
                                    kv_page_idx_lb=kv_page_idx_lb,
                                    kv_page_idx_ub=kv_page_idx_ub,
                                )
                            else:
                                sv.v_load_stage(
                                    stage_id=head_dim_stage_idx,
                                    tile_offset=tile_offset,
                                    kv_head_coord=kv_head_coord,
                                    batch_coord=batch_coord,
                                    cuseqlen_k=cuseqlen_k,
                                    seqlen_k=seqlen_k,
                                    kv_tile_start=kv_tile_start,
                                    kv_request_begin=kv_request_begin,
                                    kv_page_idx_lb=kv_page_idx_lb,
                                    kv_page_idx_ub=kv_page_idx_ub,
                                )
                            sv.commit()

                    # Window zero: K stays one tile ahead of V. Cache the final
                    # V IDs before releasing the window because its last V
                    # tile is delayed across the boundary.
                    spok.wait()
                    spok.read_offsets()
                    load_k_tile(tile_offset=0)
                    load_k_tile(tile_offset=1)
                    spov.wait()
                    load_v_tile(tile_offset=0)
                    for tile_delta in range(2, page_window_period - 1):
                        load_k_tile(tile_offset=tile_delta)
                        load_v_tile(tile_offset=tile_delta - 1)
                    load_k_tile(tile_offset=page_window_period - 1)
                    spok.release()
                    load_v_tile(tile_offset=page_window_period - 2)
                    cache_v_tile(tile_offset=page_window_period - 1)
                    spov.release()

                    # Each structural iteration consumes one complete K/V page
                    # window. Only register page IDs cross the loop boundary.
                    with domain_loop(
                        page_window_period,
                        loop_end,
                        page_window_period,
                    ):
                        spok.wait()
                        spok.read_offsets()
                        load_k_tile(tile_offset=0)
                        load_v_tile(tile_offset=-1, reuse_cached_page_ids=True)
                        load_k_tile(tile_offset=1)
                        spov.wait()
                        load_v_tile(tile_offset=0)
                        for tile_delta in range(2, page_window_period - 1):
                            load_k_tile(tile_offset=tile_delta)
                            load_v_tile(tile_offset=tile_delta - 1)
                        load_k_tile(tile_offset=page_window_period - 1)
                        spok.release()
                        load_v_tile(tile_offset=page_window_period - 2)
                        cache_v_tile(tile_offset=page_window_period - 1)
                        spov.release()

                    load_v_tile(
                        tile_offset=page_window_period - 1,
                        reuse_cached_page_ids=True,
                    )

            @schedule
            def load_reused_page_windows_schedule(
                gqkv: GmemQKVResource,
                sq: SmemQResource,
                skv: SmemKVResource,
                spok: SmemPageOffsetsKvResource,
                spov: SmemPageOffsetsKvResource,
                wq: WorkQueue | None = None,
            ) -> None:
                """Shared-buffer captured schedule."""
                load_reused_page_windows_schedule_body(
                    gqkv, sq, skv, skv, spok, spov, wq
                )

            @schedule
            def load_reused_page_windows_split_schedule(
                gqkv: GmemQKVResource,
                sq: SmemQResource,
                sk: SmemKVResource,
                sv: SmemKVResource,
                spok: SmemPageOffsetsKvResource,
                spov: SmemPageOffsetsKvResource,
                wq: WorkQueue | None = None,
            ) -> None:
                """Split K/V captured schedule."""
                load_reused_page_windows_schedule_body(gqkv, sq, sk, sv, spok, spov, wq)

            captured_schedule = schedule_with_work_queue(
                load_reused_page_windows_split_schedule
                if split_kv
                else load_reused_page_windows_schedule,
                gmem_qkv,
                smem_q,
                *kv_resources,
                smem_page_offsets_kv,
                smem_page_offsets_v,
                work_queue=work_queue,
            )
            return task_class(
                src_resources=src,
                dst_resources=dst,
                warp_idx=smem_k_or_kv.cfg.load_warp_id,
                num_warps=1,
                schedule=captured_schedule,
                num_registers=smem_k_or_kv.cfg.num_regs_other,
                name="LoadTask",
                **task_kwargs,
            )

        def load_schedule_body(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            spo: SmemPageOffsetsKvResource | None,
            wq: WorkQueue | None,
        ) -> None:
            """Load-warp schedule: stage Q, then stream K and V tiles through their rings.

            K runs one tile ahead of V so QK(i+1) can start while PV(i) waits for V.
            Page offsets are read once per tile when paged.
            """
            sq.init_load_state()
            sk.init_load_state()
            sv.init_load_state()
            if spo is not None:
                spo.init_read_state()
            with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                coords = gqkv.compute_coords()
                (
                    _seq_coord,
                    head_coord,
                    kv_head_coord,
                    _head_coord_kv,
                    batch_coord,
                    seq_coord_q,
                    cuseqlen_q,
                    cuseqlen_k,
                    seqlen_q,
                    seqlen_k,
                    kv_tile_start,
                    kv_request_begin,
                    kv_page_idx_lb,
                    kv_page_idx_ub,
                ) = coords
                sq.acquire()
                sq.tma_load(
                    seq_coord_q=seq_coord_q,
                    head_coord=head_coord,
                    batch_coord=batch_coord,
                    cuseqlen_q=cuseqlen_q,
                    seqlen_q=seqlen_q,
                    inst_idx=0,
                )
                sq.commit()

                for head_dim_stage_idx in range(num_head_dim_stages_k):
                    sk.try_acquire()
                    if spo is not None and head_dim_stage_idx == 0:
                        spo.wait()
                        spo.read_offsets()
                    sk.acquire()
                    sk.k_load_stage(
                        stage_id=head_dim_stage_idx,
                        kv_head_coord=kv_head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_k=cuseqlen_k,
                        seqlen_k=seqlen_k,
                        kv_tile_start=kv_tile_start,
                        kv_request_begin=kv_request_begin,
                        kv_page_idx_lb=kv_page_idx_lb,
                        kv_page_idx_ub=kv_page_idx_ub,
                    )
                    sk.commit()
                if spo is not None:
                    spo.release()

                with domain_loop(loop_start + 1, loop_end, loop_step):
                    for head_dim_stage_idx in range(num_head_dim_stages_k):
                        sk.try_acquire()
                        if spo is not None and head_dim_stage_idx == 0:
                            spo.wait()
                            spo.read_offsets()
                        sk.acquire()
                        sk.k_load_stage(
                            stage_id=head_dim_stage_idx,
                            kv_head_coord=kv_head_coord,
                            batch_coord=batch_coord,
                            cuseqlen_k=cuseqlen_k,
                            seqlen_k=seqlen_k,
                            kv_tile_start=kv_tile_start,
                            kv_request_begin=kv_request_begin,
                            kv_page_idx_lb=kv_page_idx_lb,
                            kv_page_idx_ub=kv_page_idx_ub,
                        )
                        sk.commit()
                    if spo is not None:
                        spo.release()

                    for head_dim_stage_idx in range(num_head_dim_stages_v):
                        sv.try_acquire()
                        if spo is not None and head_dim_stage_idx == 0:
                            spo.wait()
                            spo.read_offsets()
                        sv.acquire()
                        sv.v_load_stage(
                            stage_id=head_dim_stage_idx,
                            previous=True,
                            kv_head_coord=kv_head_coord,
                            batch_coord=batch_coord,
                            cuseqlen_k=cuseqlen_k,
                            seqlen_k=seqlen_k,
                            kv_tile_start=kv_tile_start,
                            kv_request_begin=kv_request_begin,
                            kv_page_idx_lb=kv_page_idx_lb,
                            kv_page_idx_ub=kv_page_idx_ub,
                        )
                        sv.commit()
                    if spo is not None:
                        spo.release()

                for head_dim_stage_idx in range(num_head_dim_stages_v):
                    sv.try_acquire()
                    if spo is not None and head_dim_stage_idx == 0:
                        spo.wait()
                        spo.read_offsets()
                    sv.acquire()
                    sv.v_load_stage(
                        stage_id=head_dim_stage_idx,
                        previous=False,
                        kv_head_coord=kv_head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_k=cuseqlen_k,
                        seqlen_k=seqlen_k,
                        kv_tile_start=kv_tile_start,
                        kv_request_begin=kv_request_begin,
                        kv_page_idx_lb=kv_page_idx_lb,
                        kv_page_idx_ub=kv_page_idx_ub,
                    )
                    sv.commit()
                if spo is not None:
                    spo.release()

        @schedule
        def load_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Shared-buffer captured schedule."""
            load_schedule_body(gqkv, sq, skv, skv, None, wq)

        @schedule
        def load_split_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Split K/V captured schedule."""
            load_schedule_body(gqkv, sq, sk, sv, None, wq)

        @schedule
        def load_page_offsets_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            spo: SmemPageOffsetsKvResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Shared-buffer captured schedule with a page-ID ring."""
            load_schedule_body(gqkv, sq, skv, skv, spo, wq)

        @schedule
        def load_page_offsets_split_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            spo: SmemPageOffsetsKvResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Split K/V captured schedule with a page-ID ring."""
            load_schedule_body(gqkv, sq, sk, sv, spo, wq)

        if smem_page_offsets_kv is None:
            captured_schedule = schedule_with_work_queue(
                load_split_schedule if split_kv else load_schedule,
                gmem_qkv,
                smem_q,
                *kv_resources,
                work_queue=work_queue,
            )
        else:
            captured_schedule = schedule_with_work_queue(
                load_page_offsets_split_schedule
                if split_kv
                else load_page_offsets_schedule,
                gmem_qkv,
                smem_q,
                *kv_resources,
                smem_page_offsets_kv,
                work_queue=work_queue,
            )
        return task_class(
            src_resources=src,
            dst_resources=dst,
            warp_idx=smem_k_or_kv.cfg.load_warp_id,
            num_warps=1,
            schedule=captured_schedule,
            num_registers=smem_k_or_kv.cfg.num_regs_other,
            name="LoadTask",
            **task_kwargs,
        )

    if smem_q.cfg.single_qkv_instance:
        raise ValueError("single-instance context requires the staged TMEM-P topology")
    if smem_page_offsets_kv is not None or smem_page_offsets_v is not None:
        raise ValueError("paired context resolves paged K/V IDs directly")

    def load_schedule_body(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        skr: SmemSageKScalesResource | None,
        wq: WorkQueue | None,
        smu: SmemMuResource | None = None,
    ) -> None:
        """Load paired Q instances and their directly addressed K/V tiles.

        With ``skr`` each K tile's Sage ``sfK`` words follow the tile's K
        load into the scale ring.
        """
        sq.init_load_state()
        sk.init_load_state()
        sv.init_load_state()
        if smu is not None:
            smu.init_load_state()

        def stage_sage_k_scales() -> None:
            if skr is None:
                return
            skr.acquire()
            skr.copy_tile(
                kv_head_coord=kv_head_coord,
                batch_coord=batch_coord,
                kv_tile_start=kv_tile_start,
            )
            skr.commit()

        with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):  # noqa: SIM117
            # The first K-loop iteration also loads Q0/Q1. Later iterations
            # only stream the next K/V tiles through their respective pipelines.
            with domain_loop(loop_start, loop_end, loop_step) as d:
                with d.first_iter():
                    (
                        _seq_coord,
                        head_coord,
                        kv_head_coord,
                        _head_coord_kv,
                        batch_coord,
                        seq_coord_q,
                        cuseqlen_q,
                        cuseqlen_k,
                        seqlen_q,
                        seqlen_k,
                        kv_tile_start,
                        kv_request_begin,
                        kv_page_idx_lb,
                        kv_page_idx_ub,
                    ) = gqkv.compute_coords()
                    # Load Q0 for the first Q tile in this work tile.
                    sq.acquire()
                    sq.tma_load(
                        seq_coord_q=seq_coord_q,
                        head_coord=head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_q=cuseqlen_q,
                        seqlen_q=seqlen_q,
                        inst_idx=0,
                    )
                    sq.commit()
                if smem_k_or_kv.cfg.stage_kv_by_head_dim:
                    with d.first_iter():
                        # Load Q1 for the second Q tile in this work tile.
                        sq.acquire()
                        sq.tma_load(
                            seq_coord_q=seq_coord_q,
                            head_coord=head_coord,
                            batch_coord=batch_coord,
                            cuseqlen_q=cuseqlen_q,
                            seqlen_q=seqlen_q,
                            inst_idx=1,
                        )
                        sq.commit()
                    for head_dim_stage_idx in range(
                        smem_k_or_kv.cfg.num_head_dim_stages_k
                    ):
                        sk.try_acquire()
                        sk.acquire()
                        sk.k_load_stage(
                            stage_id=head_dim_stage_idx,
                            kv_head_coord=kv_head_coord,
                            batch_coord=batch_coord,
                            cuseqlen_k=cuseqlen_k,
                            kv_tile_start=kv_tile_start,
                            seqlen_k=seqlen_k,
                            kv_request_begin=kv_request_begin,
                            kv_page_idx_lb=kv_page_idx_lb,
                            kv_page_idx_ub=kv_page_idx_ub,
                        )
                        sk.commit()
                    stage_sage_k_scales()
                else:
                    # Throttle TMA before reserving a KV stage.
                    sk.try_acquire()
                    # Load Ki, with K0 handled by the first iteration.
                    sk.acquire()
                    sk.k_load(
                        kv_head_coord=kv_head_coord,
                        batch_coord=batch_coord,
                        cuseqlen_k=cuseqlen_k,
                        kv_tile_start=kv_tile_start,
                        seqlen_k=seqlen_k,
                        kv_request_begin=kv_request_begin,
                        kv_page_idx_lb=kv_page_idx_lb,
                        kv_page_idx_ub=kv_page_idx_ub,
                    )
                    sk.commit()
                    stage_sage_k_scales()
                    with d.first_iter():
                        # Load Q1 for the second Q tile in this work tile.
                        sq.acquire()
                        sq.tma_load(
                            seq_coord_q=seq_coord_q,
                            head_coord=head_coord,
                            batch_coord=batch_coord,
                            cuseqlen_q=cuseqlen_q,
                            seqlen_q=seqlen_q,
                            inst_idx=1,
                        )
                        sq.commit()
                # Throttle TMA before reserving a KV stage.
                sv.try_acquire()
                # Load Vi, with V0 handled by the first iteration.
                sv.acquire()
                sv.v_load(
                    kv_head_coord=kv_head_coord,
                    batch_coord=batch_coord,
                    cuseqlen_k=cuseqlen_k,
                    seqlen_k=seqlen_k,
                    kv_tile_start=kv_tile_start,
                    kv_request_begin=kv_request_begin,
                    kv_page_idx_lb=kv_page_idx_lb,
                    kv_page_idx_ub=kv_page_idx_ub,
                )
                sv.commit()
                if smu is not None:
                    # VC-Attention: the mean operand of tile i-1 rides beside V_i.
                    smu.try_acquire()
                    smu.acquire()
                    smu.mu_load(
                        kv_head_coord=kv_head_coord,
                        batch_coord=batch_coord,
                        kv_tile_start=kv_tile_start,
                    )
                    smu.commit()
            if smu is not None:
                # The loop step at tile i adds the means of the group that ended
                # at tile i - 1, so the loop covers groups 0 .. N - 2 and the
                # group of the last tile never meets a following loop tile. Its
                # operand is fetched once more here for the tail mean step.
                (
                    _seq_coord,
                    _head_coord,
                    kv_head_coord,
                    _head_coord_kv,
                    batch_coord,
                    _seq_coord_q,
                    _cuseqlen_q,
                    _cuseqlen_k,
                    _seqlen_q,
                    _seqlen_k,
                    kv_tile_start,
                    _,
                    _kv_page_idx_lb,
                    _,
                ) = gqkv.compute_coords()
                smu.try_acquire()
                smu.acquire()
                smu.mu_load(
                    kv_head_coord=kv_head_coord,
                    batch_coord=batch_coord,
                    kv_tile_start=kv_tile_start,
                    is_tail=True,
                )
                smu.commit()

    @schedule
    def load_schedule_mu(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        smu: SmemMuResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Shared K/V buffer plus the VC-Attention tile-mean ring."""
        load_schedule_body(gqkv, sq, skv, skv, None, wq, smu)  # type: ignore[call-arg]

    @schedule
    def load_split_schedule_mu(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        smu: SmemMuResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K and V buffers plus the VC-Attention tile-mean ring."""
        load_schedule_body(gqkv, sq, sk, sv, None, wq, smu)  # type: ignore[call-arg]

    @schedule
    def load_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Contiguous-KV captured schedule over a shared K/V buffer."""
        # Mypy retains the earlier branch's five-argument closure signature.
        load_schedule_body(gqkv, sq, skv, skv, None, wq)  # type: ignore[call-arg]

    @schedule
    def load_sage_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        skr: SmemSageKScalesResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Contiguous-KV captured schedule that also stages Sage ``sfK`` tiles."""
        load_schedule_body(gqkv, sq, skv, skv, skr, wq)  # type: ignore[call-arg]

    @schedule
    def load_split_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Contiguous-KV captured schedule over split K and V buffers."""
        load_schedule_body(gqkv, sq, sk, sv, None, wq)  # type: ignore[call-arg]

    if smem_mu is not None:
        captured_schedule = schedule_with_work_queue(
            load_split_schedule_mu if split_kv else load_schedule_mu,
            gmem_qkv,
            smem_q,
            *kv_resources,
            smem_mu,
            work_queue=work_queue,
        )
    elif smem_sage_k is not None:
        if split_kv:
            raise ValueError("the Sage scale ring requires a shared K/V buffer")
        captured_schedule = schedule_with_work_queue(
            load_sage_schedule,
            gmem_qkv,
            smem_q,
            *kv_resources,
            smem_sage_k,
            work_queue=work_queue,
        )
    else:
        captured_schedule = schedule_with_work_queue(
            load_split_schedule if split_kv else load_schedule,
            gmem_qkv,
            smem_q,
            *kv_resources,
            work_queue=work_queue,
        )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=smem_k_or_kv.cfg.load_warp_id,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=smem_k_or_kv.cfg.num_regs_other,
        name="LoadTask",
        **task_kwargs,
    )
