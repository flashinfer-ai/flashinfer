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

"""Correction tasks: O rescaling, final normalization and the epilogue."""

import cutlass
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop
from cutlass.experimental.task_scheduling.task import Task
from ..config import FmhaDecodeConfig
from ..device.constants import KV_INST0, KV_INST1
from .common import (
    DecodeGenTask,
    LocalStatsState,
    TaskKwarg,
    decode_work_tile_schedule,
    schedule_with_optional_resources,
)


# ======================================================================
# CorrectionTask — warps 8-11, 4 warps, profile-selected register budget
# HEAD: drain initial softmax-local stats
# LOOP: correct O0 / O1 in-place before the next BMM2 accumulation
# TAIL: combine final O0/O1, normalize, store to GMEM
# ======================================================================
def create_correction_task(
    tmem_softmax_local0: MemoryResource,
    tmem_softmax_local1: MemoryResource,
    tmem_o: MemoryResource,
    tmem_corr0: MemoryResource,
    tmem_corr1: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    tmem_stats_done0: MemoryResource | None = None,
    tmem_stats_done1: MemoryResource | None = None,
    sage_v_scales: MemoryResource | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the two-instance correction and output task."""

    def correction_schedule_body(
        tmem_softmax_local0: MemoryResource,
        tmem_softmax_local1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr0: MemoryResource,
        tmem_corr1: MemoryResource,
        tmem_stats_done0: MemoryResource | None,
        tmem_stats_done1: MemoryResource | None,
        sage_v_scales: MemoryResource | None,
    ) -> None:
        """Schedule two-instance O correction and final output normalization."""

        def consume_local_with_load(
            tmem_softmax_local: MemoryResource,
            tmem_stats_done: MemoryResource | None,
            local_state: tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            load_stats_work,
        ) -> tuple[
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
        ]:
            """Wait, load, and release one phase-specific stats payload."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst_old_max_arr,
                inst_new_max_arr,
                inst_sum_arr,
            ) = local_state
            # ConsWait/ConsWork: load the phase-specific softmax stats payload
            # from TMEM-local storage.
            tmem_softmax_local.wait()
            local_state = load_stats_work(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst_old_max_arr=inst_old_max_arr,
                inst_new_max_arr=inst_new_max_arr,
                inst_sum_arr=inst_sum_arr,
            )
            if tmem_stats_done is not None:
                # Return the matching S-overwrite credit only after the stats
                # payload is resident in Correction registers.
                tmem_stats_done.wait()
                tmem_stats_done.release()
            # ConsRelease: the softmax-local payload can now be reused.
            tmem_softmax_local.release()
            return local_state

        def correct_o(
            tmem_softmax_local: MemoryResource,
            tmem_stats_done: MemoryResource | None,
            tmem_corr: MemoryResource,
            local_state: LocalStatsState,
            tail_0: cutlass.Int32,
            tail_1: cutlass.Int32,
        ) -> tuple[
            LocalStatsState,
            cutlass.Int32,
            cutlass.Int32,
        ]:
            """Consume loop stats and rescale the matching O stage."""
            # Load old/new max for the O stage that is about to be corrected.
            local_state = consume_local_with_load(
                tmem_softmax_local,
                tmem_stats_done,
                local_state,
                tmem_softmax_local.load_loop_stats,
            )
            old_max_arr = local_state[0]
            new_max_arr = local_state[1]
            # ConsWait/ConsWork: wait for the matching TMEM O stage and record
            # which O buffer is being consumed.
            tmem_o.wait()
            o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_loop(
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
            )
            # ProdWork: rescale O in place before the next BMM2 accumulation.
            tmem_corr.correction_loop_epilogue(
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
            )
            tmem_o.release()
            return local_state, tail_0, tail_1

        local0_state = tmem_softmax_local0.init_stats_state()
        local1_state = tmem_softmax_local1.init_stats_state()
        _, tail_0, tail_1 = tmem_o.init_stage_state()
        tmem_corr0.init_epilogue_state()
        tmem_corr1.init_epilogue_state()
        if sage_v_scales is not None:
            # ProdWork: stage the work tile's V channel scales while no
            # output is pending.
            sage_v_scales.acquire()
            sage_v_scales.stage_tile()
            sage_v_scales.commit()

        # HEAD: drain the first softmax-local handoff. No O is corrected here;
        # this aligns the stats pipeline before loop work starts.
        local0_state = consume_local_with_load(
            tmem_softmax_local0,
            tmem_stats_done0,
            local0_state,
            tmem_softmax_local0.load_head_stats,
        )
        local1_state = consume_local_with_load(
            tmem_softmax_local1,
            tmem_stats_done1,
            local1_state,
            tmem_softmax_local1.load_head_stats,
        )

        # LOOP: each iteration corrects O0 and O1 before later BMM2 waves
        # accumulate into the same TMEM columns.
        with domain_loop(0, domain, 1, unroll=1):
            local0_state, tail_0, tail_1 = correct_o(
                tmem_softmax_local0,
                tmem_stats_done0,
                tmem_corr0,
                local0_state,
                tail_0,
                tail_1,
            )
            local1_state, tail_0, tail_1 = correct_o(
                tmem_softmax_local1,
                tmem_stats_done1,
                tmem_corr1,
                local1_state,
                tail_0,
                tail_1,
            )

        # TAIL inst0: consume the final stats and mark the first tail O stage.
        local0_state = consume_local_with_load(
            tmem_softmax_local0,
            None,
            local0_state,
            tmem_softmax_local0.load_tail_stats,
        )
        old_max_arr = local0_state[0]
        new_max_arr = local0_state[1]
        inst0_new_max_arr = local0_state[4]
        inst0_sum_arr = local0_state[5]
        tmem_o.wait()
        o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_tail(
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            inst_idx=KV_INST0,
        )
        tmem_corr0.correction_tail_epilogue(
            o_stage_idx=o_stage_idx,
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            inst0_new_max_arr=inst0_new_max_arr,
            inst0_sum_arr=inst0_sum_arr,
            inst1_new_max_arr=inst0_new_max_arr,
            inst1_sum_arr=inst0_sum_arr,
        )
        # TAIL inst1: consume final stats for the second instance. This call
        # performs the final two-instance normalization and output store.
        local1_state = consume_local_with_load(
            tmem_softmax_local1,
            None,
            local1_state,
            tmem_softmax_local1.load_tail_stats,
        )
        old_max_arr = local1_state[0]
        new_max_arr = local1_state[1]
        inst1_new_max_arr = local1_state[4]
        inst1_sum_arr = local1_state[5]
        tmem_o.wait()
        o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_tail(
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            inst_idx=KV_INST1,
        )
        if sage_v_scales is not None:
            # ConsWait: every lane reads the scales the other lanes staged.
            sage_v_scales.wait()
        tmem_corr1.correction_tail_epilogue(
            o_stage_idx=o_stage_idx,
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            inst0_new_max_arr=inst0_new_max_arr,
            inst0_sum_arr=inst0_sum_arr,
            inst1_new_max_arr=inst1_new_max_arr,
            inst1_sum_arr=inst1_sum_arr,
        )
        if sage_v_scales is not None:
            # ConsRelease: the next work tile may restage the block.
            sage_v_scales.release()
        # Inst1 final reduction consumes both O0 and O1, so defer O0 release
        # until after inst1 has finished reading it.
        tmem_o.release()
        tmem_o.release()

    @schedule_with_optional_resources
    def correction_schedule(
        tmem_softmax_local0: MemoryResource,
        tmem_softmax_local1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr0: MemoryResource,
        tmem_corr1: MemoryResource,
        tmem_stats_done0: MemoryResource | None,
        tmem_stats_done1: MemoryResource | None,
        sage_v_scales: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Capture correction with its optional stats gates and scale staging."""
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: correction_schedule_body(
                tmem_softmax_local0,
                tmem_softmax_local1,
                tmem_o,
                tmem_corr0,
                tmem_corr1,
                tmem_stats_done0,
                tmem_stats_done1,
                sage_v_scales,
            ),
        )

    # The Keeps stats lifetime gates come as a pair; the Swaps path has none.
    if tmem_stats_done0 is None or tmem_stats_done1 is None:
        tmem_stats_done0 = None
        tmem_stats_done1 = None
    captured_schedule = correction_schedule(
        tmem_softmax_local0,
        tmem_softmax_local1,
        tmem_o,
        tmem_corr0,
        tmem_corr1,
        tmem_stats_done0,
        tmem_stats_done1,
        sage_v_scales,
        work_queue,
    )
    src = [tmem_softmax_local0, tmem_softmax_local1, tmem_o]
    if tmem_stats_done0 is not None:
        src.extend([tmem_stats_done0, tmem_stats_done1])
    dst = [tmem_corr0, tmem_corr1]
    if sage_v_scales is not None:
        # The scale block is produced and consumed by this task.
        src.append(sage_v_scales)
        dst.append(sage_v_scales)
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=((tmem_corr0, True), (tmem_corr1, True)),
        cfg=cfg,
        warp_idx=cfg.correction_warp_idx,
        num_warps=cfg.correction_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.correction_task_num_registers,
        name="CorrectionTask",
        **kw,
    )


def create_correction_task_one_inst_qkv(
    tmem_softmax_local: MemoryResource,
    tmem_o: MemoryResource,
    tmem_corr: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    tmem_stats_done: MemoryResource | None,
    domain: int | cutlass.Int32,
    sage_v_scales: MemoryResource | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the correction task for the one-inst QKV path."""

    def correction_schedule_body(
        tmem_softmax_local: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr: MemoryResource,
        tmem_stats_done: MemoryResource | None,
        sage_v_scales: MemoryResource | None,
    ) -> None:
        """Schedule one-inst O correction and final output normalization."""

        def consume_local_with_load(
            local_state: tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            load_stats_work,
            release_stats_done: bool,
        ) -> tuple[
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
            cutlass.Array,
        ]:
            """Wait, load, and release one one-inst stats payload."""
            (
                old_max_arr,
                new_max_arr,
                sum_arr,
                inst_old_max_arr,
                inst_new_max_arr,
                inst_sum_arr,
            ) = local_state
            # ConsWait/ConsWork: load the stage-specific softmax-local payload
            # produced by the one-inst softmax task.
            tmem_softmax_local.wait()
            local_state = load_stats_work(
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst_old_max_arr=inst_old_max_arr,
                inst_new_max_arr=inst_new_max_arr,
                inst_sum_arr=inst_sum_arr,
            )
            if release_stats_done and tmem_stats_done is not None:
                # Return one S-overwrite credit for this QK-derived payload.
                tmem_stats_done.wait()
                tmem_stats_done.release()
            # ConsRelease: the softmax-local payload can now be reused.
            tmem_softmax_local.release()
            return local_state

        def correct_o(
            local_state: tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            tail_0: cutlass.Int32,
            tail_1: cutlass.Int32,
        ) -> tuple[
            tuple[
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
                cutlass.Array,
            ],
            cutlass.Int32,
            cutlass.Int32,
        ]:
            """Consume loop stats and rescale the one-inst O stage."""
            local_state = consume_local_with_load(
                local_state,
                tmem_softmax_local.load_loop_stats,
                True,
            )
            old_max_arr = local_state[0]
            new_max_arr = local_state[1]
            # ConsWait/ConsWork: consume the O stage produced by BMM2. The
            # stage state tracks which TMEM O slot is live and which two slots
            # become the final tail pair.
            tmem_o.wait()
            o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_loop(
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
            )
            # ProdWork: loop phase rescales O in place before the next BMM2
            # accumulation.
            tmem_corr.correction_loop_epilogue(
                o_stage_idx=o_stage_idx,
                tail_o_stage_idx_0=tail_0,
                tail_o_stage_idx_1=tail_1,
                old_max_arr=old_max_arr,
                new_max_arr=new_max_arr,
            )
            # ConsRelease: the consumed O stage can be reused by later BMM2.
            tmem_o.release()
            return local_state, tail_0, tail_1

        local_state = tmem_softmax_local.init_stats_state()
        _, tail_0, tail_1 = tmem_o.init_stage_state()
        tmem_corr.init_epilogue_state()
        if sage_v_scales is not None:
            # ProdWork: stage the work tile's V channel scales while no
            # output is pending.
            sage_v_scales.acquire()
            sage_v_scales.stage_tile()
            sage_v_scales.commit()

        # HEAD: drain the first stats handoff. The first payload only seeds
        # the correction state; there is no prior O stage to rescale.
        local_state = consume_local_with_load(
            local_state,
            tmem_softmax_local.load_head_stats,
            True,
        )
        # LOOP: each payload/O pair corresponds to a completed BMM2 wave whose
        # accumulator must be max-corrected before more V work accumulates.
        with domain_loop(0, domain, 1, unroll=1):
            local_state, tail_0, tail_1 = correct_o(local_state, tail_0, tail_1)
        # TAIL: the final payload carries sum/new max and the final O stage is
        # normalized/stored instead of being staged for another accumulation.
        local_state = consume_local_with_load(
            local_state,
            tmem_softmax_local.load_tail_stats,
            False,
        )
        if cutlass.const_expr(
            cfg.use_persistent_scheduler and cfg.uses_staged_one_inst_tmem_p
        ):
            # Match the producer's empty tail stage before the persistent
            # work queue advances to the next tile.
            tmem_softmax_local.wait()
            tmem_softmax_local.release()
        old_max_arr = local_state[0]
        new_max_arr = local_state[1]
        inst_new_max_arr = local_state[4]
        inst_sum_arr = local_state[5]
        # ConsWait/ConsWork: consume the final O stage produced by BMM2.
        tmem_o.wait()
        o_stage_idx, tail_0, tail_1 = tmem_o.update_o_stage_tail(
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            inst_idx=KV_INST0,
        )
        if sage_v_scales is not None:
            # ConsWait: every lane reads the scales the other lanes staged.
            sage_v_scales.wait()
        # ProdWork: tail phase normalizes and stores final O.
        tmem_corr.correction_tail_epilogue(
            o_stage_idx=o_stage_idx,
            tail_o_stage_idx_0=tail_0,
            tail_o_stage_idx_1=tail_1,
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            inst0_new_max_arr=inst_new_max_arr,
            inst0_sum_arr=inst_sum_arr,
            inst1_new_max_arr=inst_new_max_arr,
            inst1_sum_arr=inst_sum_arr,
        )
        if sage_v_scales is not None:
            # ConsRelease: the next work tile may restage the block.
            sage_v_scales.release()
        # ConsRelease: the final O stage is no longer needed.
        tmem_o.release()

    @schedule_with_optional_resources
    def correction_schedule(
        tmem_softmax_local: MemoryResource,
        tmem_o: MemoryResource,
        tmem_corr: MemoryResource,
        tmem_stats_done: MemoryResource | None,
        sage_v_scales: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Wrap one-inst correction in packed persistent skip handling."""
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: correction_schedule_body(
                tmem_softmax_local,
                tmem_o,
                tmem_corr,
                tmem_stats_done,
                sage_v_scales,
            ),
        )

    schedule_result = correction_schedule(
        tmem_softmax_local,
        tmem_o,
        tmem_corr,
        tmem_stats_done,
        sage_v_scales,
        work_queue,
    )
    src = [tmem_softmax_local, tmem_o]
    if tmem_stats_done is not None:
        src.append(tmem_stats_done)
    dst = [tmem_corr]
    if sage_v_scales is not None:
        # The scale block is produced and consumed by this task.
        src.append(sage_v_scales)
        dst.append(sage_v_scales)
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=((tmem_corr, True),),
        cfg=cfg,
        warp_idx=cfg.correction_warp_idx,
        num_warps=cfg.correction_num_warps,
        schedule=schedule_result,
        num_registers=cfg.correction_task_num_registers,
        name="CorrectionTask",
        **kw,
    )
