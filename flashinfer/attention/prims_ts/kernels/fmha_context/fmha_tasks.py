# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Task definitions for the TS FMHA kernel.

Tasks own ordering, not data movement bodies. Each schedule below sequences
resource waits, acquires, work calls, commits, and releases for one warp role.
The resource methods contain the actual TMA, MMA, softmax, correction, and
epilogue work.

Schedule phase terms follow TS schedule-builder naming. HEAD is the one-time
schedule before the repeated K/V tile loop, LOOP is the repeated K/V tile body,
and TAIL is the one-time cleanup and drain after LOOP exits.
"""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from ..stage import FmhaStage
from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.task import Task

from .fmha_resources import (
    GmemOResource,
    GmemQKVResource,
    S0S1SequenceResource,
    SmemKVResource,
    SmemOResource,
    SmemQResource,
    TmemOResource,
    TmemSPResource,
    TmemStatsResource,
    TmemStatsDoneResource,
)


def _persistent_tail(work_queue: WorkQueue) -> None:
    """Advance and release the persistent work tile after one task body."""
    work_queue.wait()
    work_queue.get_and_advance_work_tile()
    work_queue.release()


def _src_resources(
    *resources: MemoryResource,
    work_queue: WorkQueue | None,
) -> list[MemoryResource]:
    """Build a task source-resource list, including WorkQueue when present."""
    src = list(resources)
    if work_queue is not None:
        src.append(work_queue)
    return src


def _schedule_with_work_queue(
    schedule: Callable[..., object],
    *resources: MemoryResource,
    work_queue: WorkQueue | None,
) -> object:
    """Invoke a captured schedule with the optional WorkQueue argument."""
    if work_queue is None:
        return schedule(*resources)
    return schedule(*resources, work_queue)


@contextmanager
def _work_tile_schedule_loop(
    work_queue: WorkQueue | None,
) -> Generator[object | None, None, None]:
    """Wrap a task body once per persistent work tile, or once for static schedules."""
    if work_queue is not None:
        with work_tile_loop(work_queue) as work_tile:
            yield work_tile
            _persistent_tail(work_queue)
    else:
        yield None


def _captured_loop_bounds(
    task_kwargs: dict[str, object],
) -> tuple[object, object, object]:
    """Infer loop bounds for captured schedules from task kwargs."""
    loop_start = task_kwargs.pop("domain_start", 0)
    loop_step = task_kwargs.pop("step", 1)
    loop_end = task_kwargs.pop("domain", None)
    if loop_end is None and "num_kv_tiles" in task_kwargs:
        loop_end = task_kwargs["num_kv_tiles"] - task_kwargs.get("offset", 0)
        if isinstance(loop_end, int):
            loop_end = max(loop_end, 0)
    if loop_end is None:
        # Structural fallback only; real runtime domain still comes from task kwargs.
        loop_end = 1
    return loop_start, loop_end, loop_step


def create_load_task(
    gmem_qkv: GmemQKVResource,
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp TMA load task."""
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_kwargs)
    src = _src_resources(gmem_qkv, work_queue=work_queue)
    dst = [smem_q, smem_kv]

    @schedule
    def load_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Captured schedule for Q/K/V TMA loads."""
        sq.init_load_state()
        skv.init_load_state()
        with _work_tile_schedule_loop(wq):  # noqa: SIM117
            # The first K-loop iteration also loads Q0/Q1. Later iterations
            # only stream the next K/V tiles through the SmemKV pipeline.
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
                        _seqlen_k,
                        kv_tile_start,
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
                # Throttle TMA before reserving a KV stage.
                skv.try_acquire()
                # Load Ki, with K0 handled by the first iteration.
                skv.acquire()
                skv.k_load(
                    kv_head_coord=kv_head_coord,
                    batch_coord=batch_coord,
                    cuseqlen_k=cuseqlen_k,
                    kv_tile_start=kv_tile_start,
                )
                skv.commit()
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
                skv.try_acquire()
                # Load Vi, with V0 handled by the first iteration.
                skv.acquire()
                skv.v_load(
                    kv_head_coord=kv_head_coord,
                    batch_coord=batch_coord,
                    cuseqlen_k=cuseqlen_k,
                    kv_tile_start=kv_tile_start,
                )
                skv.commit()

    captured_schedule = _schedule_with_work_queue(
        load_schedule, gmem_qkv, smem_q, smem_kv, work_queue=work_queue
    )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=13,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=gmem_qkv.cfg.num_regs_other,
        name="LoadTask",
        **task_kwargs,
    )


def create_mma_task(
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    tmem_sp0: TmemSPResource,
    tmem_sp1: TmemSPResource,
    tmem_o: TmemOResource,
    tmem_vec_done_0: TmemStatsDoneResource,
    tmem_vec_done_1: TmemStatsDoneResource,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp MMA compute task."""
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_kwargs)
    src = _src_resources(smem_q, smem_kv, work_queue=work_queue)

    @schedule
    def mma_schedule(
        sq: SmemQResource,
        skv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Captured schedule for interleaved QK and PV MMA work."""
        sq.init_descriptor_state()
        skv.init_descriptor_state()
        sp0.init_mma_state()
        sp1.init_mma_state()
        to.init_mma_state()
        with _work_tile_schedule_loop(wq):
            # HEAD: consume Q0, K0, Q1, and V0. TmemStatsDone starts empty, so
            # the first acquire succeeds without priming. On later work tiles,
            # correction has released the previous stats slot.
            #
            # Consume Q0, K0, then QK(Q0,K0)→S0.
            sq.wait()
            desc_q0_base = sq.q0_desc(inst_idx=0)
            skv.wait()
            desc_k_base = skv.k_desc()
            vd0.acquire()
            sp0.acquire()
            sp0.qk_mma(
                desc_q_base=desc_q0_base,
                desc_k_base=desc_k_base,
                section=FmhaStage.Head,
            )
            sp0.commit()
            vd0.commit()
            # Consume Q1, then QK(Q1,K0)→S1.
            sq.wait()
            desc_q1_base = sq.q1_desc(inst_idx=1)
            vd1.acquire()
            sp1.acquire()
            sp1.qk_mma(
                desc_q_base=desc_q1_base,
                desc_k_base=desc_k_base,
                section=FmhaStage.Head,
            )
            sp1.commit()
            vd1.commit()
            # Q0/Q1 stay live because UMMA reads Q throughout the K-loop.
            # Release K0 (done with QK→S0 and QK→S1), then consume V0.
            skv.release()
            skv.wait()
            desc_v_base = skv.v_desc()
            # Acquire O first (off critical path), then acquire SP0 and run PV→O0.
            to.acquire()
            sp0.acquire()
            sp0.p_read()
            to.pv_mma(desc_v_base=desc_v_base, section=FmhaStage.Head)
            to.commit()

            # LOOP: interleave QK and PV work while preserving the previous V
            # tile until its PV MMA has consumed it:
            #   QK0(deferred commit) -> PV1(V_prev, release V_prev) ->
            #   QK1(commit) -> release Ki+1 -> wait Vi+1 -> PV0(no commit)
            with domain_loop(loop_start, loop_end, loop_step):
                skv.wait()
                desc_k_base = skv.k_desc()
                # QK0: QK(Q0,Ki+1) → S0 (no acquire; handle held from PV0).
                sp0.qk_mma(
                    desc_q_base=desc_q0_base,
                    desc_k_base=desc_k_base,
                    section=FmhaStage.Loop,
                )
                sp0.commit()
                # PV1(V_prev): P1 * V_prev → O1.
                to.acquire()
                sp1.acquire()
                sp1.p_read()
                to.pv_mma(desc_v_base=desc_v_base, section=FmhaStage.Loop)
                to.commit()
                # Release V_prev after PV1 UMMA consumed SMEM data.
                skv.release()
                # QK1: QK(Q1,Ki+1) → S1 (no acquire; handle held from PV1).
                sp1.qk_mma(
                    desc_q_base=desc_q1_base,
                    desc_k_base=desc_k_base,
                    section=FmhaStage.Loop,
                )
                sp1.commit()
                # Release Ki+1, then wait Vi+1.
                skv.release()
                skv.wait()
                desc_v_base = skv.v_desc()
                # PV0: P0 * Vi+1 → O0.
                to.acquire()
                sp0.acquire()
                sp0.p_read()
                to.pv_mma(
                    desc_v_base=desc_v_base,
                    section=FmhaStage.Loop,
                    inst_idx=1,
                )
                to.commit()

            # TAIL: release Qs, close the deferred SP state, and run the final
            # PV→O1 MMA.
            sq.release()
            sq.release()
            sp0.commit()
            to.acquire()
            sp1.acquire()
            sp1.p_read()
            to.pv_mma(
                desc_v_base=desc_v_base,
                section=FmhaStage.Tail,
                is_tail=True,
            )
            to.commit()
            skv.release()
            sp1.commit()

    captured_schedule = _schedule_with_work_queue(
        mma_schedule,
        smem_q,
        smem_kv,
        tmem_sp0,
        tmem_sp1,
        tmem_o,
        tmem_vec_done_0,
        tmem_vec_done_1,
        work_queue=work_queue,
    )
    return task_class(
        src_resources=src,
        dst_resources=[tmem_sp0, tmem_sp1, tmem_o, tmem_vec_done_0, tmem_vec_done_1],
        warp_idx=12,
        num_warps=1,
        schedule=captured_schedule,
        name="MmaTask",
        num_registers=smem_q.cfg.num_regs_other,
        **task_kwargs,
    )


def create_softmax_task(
    index: int,
    tmem_sp: TmemSPResource,
    tmem_vec: TmemStatsResource,
    s0s1_seq: S0S1SequenceResource,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create a four-warp Softmax task.

    index=0: warps 0-3 (Softmax0Task) — S0-S1 producer (acquire/commit)
    index=1: warps 4-7 (Softmax1Task) — S0-S1 consumer (wait/release)

    Args:
        task_class: Task subclass used to instantiate the softmax schedule.
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_kwargs)
    if s0s1_seq is not None and index == 1:
        src = _src_resources(tmem_sp, s0s1_seq, work_queue=work_queue)
    else:
        src = _src_resources(tmem_sp, work_queue=work_queue)
    dst = [tmem_vec]
    if s0s1_seq is not None and index == 0:
        dst.append(s0s1_seq)

    @schedule
    def softmax_schedule(
        sp: TmemSPResource,
        vec: TmemStatsResource,
        seq: S0S1SequenceResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Captured schedule for one softmax warp group."""
        p_chunk = sp.init_softmax_state()
        scale_softmax_log2 = sp.load_scale_softmax_log2()
        vec.init_store_state()
        with _work_tile_schedule_loop(wq):
            # Recompute per-tile SP/Vec TMEM state.
            old_row_max, row_max, row_sum, q_offset = sp.init_softmax_work_tile_state()
            vec.init_store_work_tile_state()
            if tmem_sp.uses_varlen_q_offset_cache:
                q_offset = sp.cache_q_offset()
            if tmem_sp.uses_packed_dense_k_mask:
                seqlen_k = sp.cache_seqlen_k()
            # Reserve a stats slot before the first softmax result is published.
            vec.acquire()
            with domain_loop(loop_start, loop_end, loop_step):
                sp.wait()
                # Compute row max and publish vec.
                if tmem_sp.uses_left_window_loop_mask:
                    old_row_max, row_max = sp.left_masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                    )
                elif tmem_sp.uses_varlen_loop_right_mask:
                    old_row_max, row_max = sp.right_masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                        section=FmhaStage.Loop,
                    )
                elif tmem_sp.uses_query_paired_q_offset_loop_mask:
                    old_row_max, row_max = sp.loop_masked_row_max(
                        row_max=row_max,
                        q_offset=q_offset,
                    )
                elif tmem_sp.uses_fixed_dense_k_tail_mask:
                    old_row_max, row_max = sp.fixed_dense_k_tail_masked_row_max(
                        row_max=row_max,
                    )
                elif tmem_sp.uses_packed_dense_k_mask:
                    old_row_max, row_max = sp.packed_dense_k_masked_row_max(
                        row_max=row_max,
                        seqlen_k=seqlen_k,
                        section=FmhaStage.Loop,
                    )
                else:
                    old_row_max, row_max = sp.compute_row_max(row_max=row_max)
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if index == 0:
                    # Softmax0 is the S0-S1 producer: acquire/commit sequence.
                    seq.acquire()
                else:
                    # Softmax1 is the S0-S1 consumer: wait/release sequence.
                    seq.wait()
                # Apply softmax and write P.
                p_chunk = sp.exp2_p(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                )
                if index == 0:
                    seq.commit()
                else:
                    seq.release()
                sp.release()
                # Reduction.
                row_sum = sp.softmax_aux_reduce(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    p_chunk=p_chunk,
                    scale_softmax_log2=scale_softmax_log2,
                )
                # Acquire vec for next iter.
                vec.acquire()

            if tmem_sp.uses_head_paired_causal_tail_mask:
                # Head-paired maps Q0/Q1 to adjacent Hq slices at the same S
                # tile. Its tail mask uses right_masked_row_max(), which keeps
                # that mapping and applies both sliding-window bounds when
                # enabled.
                sp.wait()
                old_row_max, row_max = sp.right_masked_row_max(
                    row_max=row_max,
                    q_offset=q_offset,
                    section=FmhaStage.Tail,
                )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                if index == 0:
                    seq.acquire()
                else:
                    seq.wait()
                p_chunk = sp.exp2_p(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                )
                if index == 0:
                    seq.commit()
                else:
                    seq.release()
                sp.release()
                row_sum = sp.softmax_aux_reduce(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    p_chunk=p_chunk,
                    scale_softmax_log2=scale_softmax_log2,
                )
                vec.acquire()
                sp.wait()
                sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
            elif tmem_sp.uses_query_paired_causal_tail_mask:
                # Query-paired maps Q1 to the next S tile. Its generic causal
                # tail uses masked_row_max(), which includes q_half * q_tile_m
                # so each peer tile is masked at the right sequence boundary.
                sp.wait()
                old_row_max, row_max = sp.masked_row_max(
                    row_max=row_max,
                    q_offset=q_offset,
                )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                seq.acquire()
                p_chunk = sp.masked_exp2_p(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                )
                seq.commit()
                sp.release()
                row_sum = sp.softmax_aux_reduce(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    p_chunk=p_chunk,
                    scale_softmax_log2=scale_softmax_log2,
                )
                if tmem_sp.uses_query_paired_invalid_tail:
                    sp.wait()
                    old_row_max, row_max = sp.invalid_row_max(row_max=row_max)
                    vec.acquire()
                    vec.store_vec(
                        old_row_max=old_row_max,
                        row_max=row_max,
                        row_sum=row_sum,
                    )
                    vec.commit()
                    seq.acquire()
                    sp.invalid_exp2_p(row_max=row_max)
                    seq.commit()
                    sp.release()
                sp.wait()
                sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.acquire()
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
            elif tmem_sp.cfg.is_causal:
                # Causal softmax1 TAIL handles masked rows and cleanup.
                sp.wait()
                old_row_max, row_max = sp.masked_row_max(
                    row_max=row_max,
                    q_offset=q_offset,
                )
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
                seq.wait()
                p_chunk = sp.masked_exp2_p(
                    row_max=row_max,
                    scale_softmax_log2=scale_softmax_log2,
                )
                seq.release()
                sp.release()
                row_sum = sp.softmax_aux_reduce(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                    p_chunk=p_chunk,
                    scale_softmax_log2=scale_softmax_log2,
                )
                sp.wait()
                sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.acquire()
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()
            else:
                # Non-causal TAIL commits the reserved stats slot and lets MMA
                # complete its cleanup path.
                sp.wait()
                sp.release()
                old_row_max = sp.softmax_aux_identity(row_max=row_max)
                vec.store_vec(
                    old_row_max=old_row_max,
                    row_max=row_max,
                    row_sum=row_sum,
                )
                vec.commit()

    captured_schedule = _schedule_with_work_queue(
        softmax_schedule, tmem_sp, tmem_vec, s0s1_seq, work_queue=work_queue
    )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=index * 4,
        num_warps=4,
        schedule=captured_schedule,
        num_registers=tmem_sp.cfg.num_regs_softmax,
        name=f"Softmax{index}Task",
        **task_kwargs,
    )


def create_correction_task(
    tmem_vec0: TmemStatsResource,
    tmem_vec1: TmemStatsResource,
    tmem_o: TmemOResource,
    smem_o_0: SmemOResource,
    smem_o_1: SmemOResource,
    tmem_vec_done_0: TmemStatsDoneResource,
    tmem_vec_done_1: TmemStatsDoneResource,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the four-warp Correction task (warps 8-11)."""
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_kwargs)
    src = _src_resources(
        tmem_vec0,
        tmem_vec1,
        tmem_o,
        tmem_vec_done_0,
        tmem_vec_done_1,
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
        with _work_tile_schedule_loop(wq):
            # Per-tile TMEM/SMEM cached addresses are computed here.
            v0.init_read_work_tile_state()
            v1.init_read_work_tile_state()
            to.init_correction_work_tile_state()
            so0.init_store_work_tile_state()
            so1.init_store_work_tile_state()
            # No tmem-stats-done priming is needed because the pipeline starts empty.
            # Discard the first TmemStats0 slot and hold TmemStats1 for cross-release in the
            # first K-loop correction step.
            v0.wait()
            v0.release()
            v1.wait()
            # The correction loop consumes vec/O pairs in alternating order so
            # each half can unblock the other half's next producer.
            with domain_loop(loop_start, loop_end, loop_step):
                # Part 1: consume TmemStats0 + O0, release TmemStats1.
                v0.wait()
                vec_old_max, vec_new_max, _, vec_scale = v0.read_vec(
                    scale_softmax_log2=scale_softmax_log2_v0,
                )
                to.wait()
                to.correct(
                    vec_old_max=vec_old_max,
                    vec_new_max=vec_new_max,
                    vec_scale=vec_scale,
                    inst_idx=0,
                )
                v1.release()
                to.release()
                # Part 2: consume TmemStats1 + O1, release TmemStats0.
                v1.wait()
                vec_old_max, vec_new_max, _, vec_scale = v1.read_vec(
                    scale_softmax_log2=scale_softmax_log2_v1,
                )
                to.wait()
                to.correct(
                    vec_old_max=vec_old_max,
                    vec_new_max=vec_new_max,
                    vec_scale=vec_scale,
                    inst_idx=1,
                )
                v0.release()
                to.release()
            # TAIL: consume remaining stats, release tmem-stats-done gates, and
            # stage corrected O0/O1 into SMEM for the epilogue task.
            v1.release()
            v0.wait()
            _, _, vec_row_sum, vec_scale = v0.read_vec(
                scale_softmax_log2=scale_softmax_log2_v0,
            )
            vd0.wait()
            vd0.release()
            v0.release()
            to.wait()
            so0.acquire()
            so0.store_o(
                vec_row_sum=vec_row_sum,
                vec_scale=vec_scale,
                output_scale=output_scale0,
            )
            so0.commit()
            to.release()
            v1.wait()
            _, _, vec_row_sum, vec_scale = v1.read_vec(
                scale_softmax_log2=scale_softmax_log2_v1,
            )
            vd1.wait()
            vd1.release()
            v1.release()
            to.wait()
            so1.acquire()
            so1.store_o(
                vec_row_sum=vec_row_sum,
                vec_scale=vec_scale,
                output_scale=output_scale1,
            )
            so1.commit()
            to.release()

    captured_schedule = _schedule_with_work_queue(
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
        num_registers=tmem_vec0.cfg.num_regs_correction,
        name="CorrectionTask",
        **task_kwargs,
    )


def create_epilogue_task(
    smem_o_0: SmemOResource,
    smem_o_1: SmemOResource,
    gmem_o_0: GmemOResource,
    gmem_o_1: GmemOResource,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp Epilogue store task (warp 14)."""
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_kwargs)
    src = _src_resources(smem_o_0, smem_o_1, work_queue=work_queue)

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
        with _work_tile_schedule_loop(wq):
            # Per-tile SMEM O address base is computed in work vars.
            so0.init_output_work_tile_state()
            so1.init_output_work_tile_state()
            with domain_loop(loop_start, loop_end, loop_step):
                pass
            # Store the first corrected O tile through gmem_o_0.
            so0.wait()
            head_coord, batch_coord, seq_coord_q = so0.compute_output_coords()
            go0.tma_store(
                head_coord=head_coord,
                batch_coord=batch_coord,
                seq_coord_q=seq_coord_q,
            )
            so0.release()
            # Store the second corrected O tile through gmem_o_1.
            so1.wait()
            head_coord, batch_coord, seq_coord_q = so1.compute_output_coords()
            go1.tma_store(
                head_coord=head_coord,
                batch_coord=batch_coord,
                seq_coord_q=seq_coord_q,
            )
            so1.release()

    captured_schedule = _schedule_with_work_queue(
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
        num_registers=smem_o_0.cfg.num_regs_other,
        name="EpilogueTask",
        **task_kwargs,
    )


def create_padding_task(
    work_queue: WorkQueue | None,
    num_registers: int = 32,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp padding task (warp 15).

    Required in ALL modes (persistent and non-persistent) because
    ``setmaxnreg.sync`` requires every warp in the warp group to
    participate.  Warps 12-15 form warp group 3; without the padding
    task warp 15 never calls ``setmaxregister``, deadlocking the group.

    In persistent mode the task also consumes work_queue tiles so that
    warp 15 participates in the persistent outer loop.

    In CLC dynamic mode, warp 15 is replaced by a scheduler task
    (see ``create_scheduler_task``).
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_kwargs)
    src = _src_resources(work_queue=work_queue)

    @schedule
    def padding_schedule(wq: WorkQueue | None = None) -> None:
        """Captured schedule for warp-group register participation."""
        with _work_tile_schedule_loop(wq):  # noqa: SIM117
            with domain_loop(loop_start, loop_end, loop_step):
                pass

    captured_schedule = _schedule_with_work_queue(
        padding_schedule, work_queue=work_queue
    )
    return task_class(
        src_resources=src,
        dst_resources=[],
        warp_idx=15,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=num_registers,
        name="PaddingTask",
        **task_kwargs,
    )


def create_scheduler_task(
    work_queue: WorkQueue,
    num_registers: int = 32,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp CLC scheduler task (warp 15).

    Replaces the padding task in CLC dynamic persistent mode.
    Issues CLC tile-fetch queries (producer side) and participates in
    the persistent outer loop.  Still satisfies the ``setmaxnreg.sync``
    requirement for warp group 3 (warps 12-15).
    """
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_kwargs)

    @schedule
    def scheduler_schedule(wq: WorkQueue) -> None:
        """Captured schedule for CLC work-tile fetches."""
        with _work_tile_schedule_loop(wq):
            with domain_loop(loop_start, loop_end, loop_step):
                pass
            # Producer side: issue CLC tile-fetch query.
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()

    captured_schedule = scheduler_schedule(work_queue)
    return task_class(
        src_resources=[work_queue],
        dst_resources=[work_queue],
        warp_idx=15,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=num_registers,
        name="SchedulerTask",
        **task_kwargs,
    )
