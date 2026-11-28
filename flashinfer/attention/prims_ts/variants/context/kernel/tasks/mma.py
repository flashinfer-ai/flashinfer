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

"""MMA tasks of the context kernel: QK and PV UMMA issue."""

import cutlass
from cutlass import Int32
from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from .....device.stage import WorkStage
from ..resources.gmem_qkv import GmemQKVResource
from ..resources.smem_kv import SmemKVResource
from ..resources.smem_mu import SmemMuResource
from ..resources.smem_p import SmemPResource
from ..resources.smem_q import SmemQResource
from ..resources.sync import TmemPPrefixReadyResource, TmemStatsDoneResource
from ..resources.tmem_o import TmemOResource
from ..resources.tmem_p import TmemPResource
from ..resources.tmem_sp import TmemSPResource
from .common import (
    captured_loop_bounds,
    packed_context_skip_predicate,
    schedule_with_work_queue,
    src_resources,
    work_tile_schedule_loop,
)
from typing import Any


def create_mma_task(
    gmem_qkv: GmemQKVResource,
    smem_q: SmemQResource,
    smem_k_or_kv: SmemKVResource,
    smem_v: SmemKVResource | None,
    tmem_sp0: TmemSPResource,
    tmem_sp1: TmemSPResource | None,
    tmem_p0: TmemPResource | None,
    tmem_o: TmemOResource,
    tmem_vec_done_0: TmemStatsDoneResource,
    tmem_vec_done_1: TmemStatsDoneResource | None,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    tmem_p_prefix_ready_0: TmemPPrefixReadyResource | None = None,
    tmem_p_prefix_ready_1: TmemPPrefixReadyResource | None = None,
    smem_p0: SmemPResource | None = None,
    smem_p1: SmemPResource | None = None,
    smem_mu: SmemMuResource | None = None,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp MMA compute task.

    ``smem_mu`` (VC-Attention) supplies the bf16 tile-mean operand consumed by
    one extra ``kind::f16`` step after each tile's PV steps.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = packed_context_skip_predicate(work_queue)
    # Only paged MMA schedules read request coordinates from global metadata.
    qkv_resources = [gmem_qkv] if smem_q.cfg.use_paged_kv else []
    split_kv = smem_k_or_kv.cfg.split_kv_pipelines
    if split_kv and smem_v is None:
        raise ValueError("split K/V staging requires a separate V buffer")
    kv_resources = (smem_k_or_kv, smem_v) if split_kv else (smem_k_or_kv,)
    pv_half_overlap = smem_q.cfg.pv_half_overlap
    if pv_half_overlap and (
        tmem_p_prefix_ready_0 is None or tmem_p_prefix_ready_1 is None
    ):
        raise ValueError("pv_half_overlap requires both P-prefix barriers")
    p_prefix_resources = (
        (tmem_p_prefix_ready_0, tmem_p_prefix_ready_1) if pv_half_overlap else ()
    )
    p_in_smem = smem_q.cfg.p_in_smem
    if p_in_smem and (smem_p0 is None or smem_p1 is None):
        raise ValueError("p_in_smem requires both SMEM P resources")
    smem_p_resources = (smem_p0, smem_p1) if p_in_smem else ()
    mu_resources = (smem_mu,) if smem_mu is not None else ()
    src = src_resources(
        *qkv_resources,
        smem_q,
        *kv_resources,
        *p_prefix_resources,
        *smem_p_resources,
        *mu_resources,
        work_queue=work_queue,
    )
    num_head_dim_stages_k = smem_k_or_kv.cfg.num_head_dim_stages_k
    num_head_dim_stages_v = smem_k_or_kv.cfg.num_head_dim_stages_v

    if (
        smem_q.cfg.single_qkv_instance
        and smem_q.cfg.has_tmem_p_pipeline
        and tmem_p0 is not None
    ):
        split_src = src_resources(
            *qkv_resources, smem_q, *kv_resources, tmem_p0, work_queue=work_queue
        )
        num_head_dim_stages_k = smem_k_or_kv.cfg.num_head_dim_stages_k
        num_head_dim_stages_v = smem_k_or_kv.cfg.num_head_dim_stages_v
        if num_head_dim_stages_k not in (1, 2) or num_head_dim_stages_v not in (1, 2):
            raise ValueError(
                "loop-carried split S/P scheduling expects one or two K/V stages"
            )

        def mma_schedule_body(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            sp0: TmemSPResource,
            tp0: TmemPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """MMA-warp schedule with separate S and P TMEM pipelines.

            Prologue issues QK for the first tile; the steady-state loop then alternates
            QK(i+1) into S with PV(i) from P, consuming K and V from independent rings.
            """
            sq.init_descriptor_state()
            sk.init_descriptor_state()
            sv.init_descriptor_state()
            sp0.init_mma_state()
            to.init_mma_state()
            with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                sp0.init_mma_work_tile_state()
                to.init_mma_work_tile_state()
                v_seqlen_k = Int32(0)
                v_kv_tile_start = Int32(0)
                if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                    (
                        _seq_coord,
                        _head_coord,
                        _kv_head_coord,
                        _head_coord_kv,
                        _batch_coord,
                        _seq_coord_q,
                        _cuseqlen_q,
                        _cuseqlen_k,
                        _seqlen_q,
                        v_seqlen_k,
                        v_kv_tile_start,
                        kv_request_begin,
                        _kv_page_idx_lb,
                        kv_page_idx_ub,
                    ) = gqkv.compute_coords()

                sq.wait()
                desc_q0_base = sq.q0_desc(inst_idx=0)
                if not smem_q.cfg.stats_via_smem:
                    vd0.acquire()
                sp0.acquire()
                for head_dim_stage_idx in range(num_head_dim_stages_k):
                    sk.wait()
                    desc_k_base = sk.k_desc()
                    sp0.qk_mma(
                        desc_q_base=desc_q0_base,
                        desc_k_base=desc_k_base,
                        section=WorkStage.Head,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                    sk.release()
                sp0.commit()
                if not smem_q.cfg.stats_via_smem:
                    vd0.commit()
                sp0.acquire()

                # Loop offset i is local to the steady state. LoadTask has
                # already advanced K by one tile, so these waits consume K(i+1)
                # for QK and V(i) for PV.
                with domain_loop(loop_start, loop_end, loop_step):
                    if not smem_q.cfg.stats_via_smem:
                        vd0.acquire()
                    sk.wait()
                    desc_k_base = sk.k_desc()
                    sp0.qk_mma(
                        desc_q_base=desc_q0_base,
                        desc_k_base=desc_k_base,
                        section=WorkStage.Loop,
                        head_dim_stage_idx=0,
                    )
                    sk.release()

                    if num_head_dim_stages_k > 1:
                        sk.wait()
                        desc_k_base = sk.k_desc()
                        sp0.qk_mma(
                            desc_q_base=desc_q0_base,
                            desc_k_base=desc_k_base,
                            section=WorkStage.Loop,
                            head_dim_stage_idx=1,
                        )
                        sk.release()
                    sp0.commit()
                    if not smem_q.cfg.stats_via_smem:
                        vd0.commit()
                    sp0.acquire()

                    to.acquire()
                    tp0.wait()
                    tmem_p_base = tp0.p_base()
                    to.set_p_base(tmem_p_base=tmem_p_base)

                    # TODO: sv.wait() only depends on V's own TMA-load barrier,
                    # not on tp0 (softmax P). Now that K/V have independent
                    # pipelines, it could be issued earlier to overlap with the
                    # K/QK/softmax work above instead of waiting until here.
                    sv.wait()
                    if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                        desc_v_base = sv.v_desc_paged(
                            section=WorkStage.Loop,
                            seqlen_k=v_seqlen_k,
                            kv_tile_start=v_kv_tile_start,
                        )
                    else:
                        desc_v_base = sv.v_desc()
                    to.pv_mma(
                        desc_v_base=desc_v_base,
                        section=WorkStage.Loop,
                        head_dim_stage_idx=0,
                    )
                    sv.release()

                    if num_head_dim_stages_v > 1:
                        sv.wait()
                        if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                            desc_v_base = sv.v_desc_paged(
                                section=WorkStage.Loop,
                                seqlen_k=v_seqlen_k,
                                kv_tile_start=v_kv_tile_start,
                            )
                        else:
                            desc_v_base = sv.v_desc()
                        to.pv_mma(
                            desc_v_base=desc_v_base,
                            section=WorkStage.Loop,
                            head_dim_stage_idx=1,
                        )
                        sv.release()
                    to.commit()
                    tp0.release()

                sq.release()
                to.acquire()
                tp0.wait()
                tmem_p_base = tp0.p_base()
                to.set_p_base(tmem_p_base=tmem_p_base)
                for head_dim_stage_idx in range(num_head_dim_stages_v):
                    sv.wait()
                    if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                        desc_v_base = sv.v_desc_paged(
                            section=WorkStage.Tail,
                            seqlen_k=v_seqlen_k,
                            kv_tile_start=v_kv_tile_start,
                        )
                    else:
                        desc_v_base = sv.v_desc()
                    to.pv_mma(
                        desc_v_base=desc_v_base,
                        section=WorkStage.Tail,
                        head_dim_stage_idx=head_dim_stage_idx,
                        is_tail=True,
                    )
                    sv.release()
                to.commit()
                tp0.release()
                if not smem_q.cfg.stats_via_smem:
                    vd0.acquire()
                sp0.commit()
                if not smem_q.cfg.stats_via_smem:
                    vd0.commit()
                tp0.wait()
                tp0.release()

        @schedule
        def mma_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            sp0: TmemSPResource,
            tp0: TmemPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Shared-buffer captured schedule."""
            mma_schedule_body(gqkv, sq, skv, skv, sp0, tp0, to, vd0, wq)

        @schedule
        def mma_split_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            sp0: TmemSPResource,
            tp0: TmemPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Split K/V captured schedule."""
            mma_schedule_body(gqkv, sq, sk, sv, sp0, tp0, to, vd0, wq)

        captured_schedule = schedule_with_work_queue(
            mma_split_schedule if split_kv else mma_schedule,
            gmem_qkv,
            smem_q,
            *kv_resources,
            tmem_sp0,
            tmem_p0,
            tmem_o,
            tmem_vec_done_0,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=split_src,
            dst_resources=[tmem_sp0, tmem_o]
            + ([] if smem_q.cfg.stats_via_smem else [tmem_vec_done_0]),
            warp_idx=smem_q.cfg.mma_warp_id,
            num_warps=1,
            schedule=captured_schedule,
            name="MmaTask",
            num_registers=smem_q.cfg.num_regs_other,
            **task_kwargs,
        )

    if smem_q.cfg.single_qkv_instance:
        num_head_dim_stages_k = smem_k_or_kv.cfg.num_head_dim_stages_k
        num_head_dim_stages_v = smem_k_or_kv.cfg.num_head_dim_stages_v

        def mma_schedule_body(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            sp0: TmemSPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            desc_q0_base, _desc_q1_base = sq.create_function_variables()
            # Every instance declares both descriptor slots; a split buffer
            # only ever fills the half matching its role.
            desc_k_base, _desc_v_base = sk.create_function_variables()
            _desc_k_base, desc_v_base = sv.create_function_variables()
            sp0.create_function_variables()
            to.create_function_variables()
            vd0.create_function_variables()
            with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
                v_seqlen_k = Int32(0)
                v_kv_tile_start = Int32(0)
                if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                    (
                        _seq_coord,
                        _head_coord,
                        _kv_head_coord,
                        _head_coord_kv,
                        _batch_coord,
                        _seq_coord_q,
                        _cuseqlen_q,
                        _cuseqlen_k,
                        _seqlen_q,
                        v_seqlen_k,
                        v_kv_tile_start,
                        kv_request_begin,
                        _kv_page_idx_lb,
                        kv_page_idx_ub,
                    ) = gqkv.compute_coords()
                if wq is not None:
                    sp0.create_work_tile_variables()
                    to.create_work_tile_variables()

                with domain_loop(loop_start, loop_end, loop_step) as d:
                    with d.first_iter():
                        sq.wait()
                        desc_q0_base = sq.q0_desc(inst_idx=0)
                        vd0.acquire()
                        sp0.acquire()
                    for head_dim_stage_idx in range(num_head_dim_stages_k):
                        sk.wait()
                        desc_k_base = sk.k_desc()
                        sp0.qk_mma(
                            desc_q_base=desc_q0_base,
                            desc_k_base=desc_k_base,
                            section=WorkStage.Loop,
                            head_dim_stage_idx=head_dim_stage_idx,
                        )
                        sk.release()
                    sp0.commit()
                    with d.first_iter():
                        vd0.commit()
                    to.acquire()
                    sp0.acquire()
                    sp0.p_read()
                    for head_dim_stage_idx in range(num_head_dim_stages_v):
                        sv.wait()
                        if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                            desc_v_base = sv.v_desc_paged(
                                section=WorkStage.Loop,
                                seqlen_k=v_seqlen_k,
                                kv_tile_start=v_kv_tile_start,
                            )
                        else:
                            desc_v_base = sv.v_desc()
                        to.pv_mma(
                            desc_v_base=desc_v_base,
                            section=WorkStage.Loop,
                            head_dim_stage_idx=head_dim_stage_idx,
                        )
                        sv.release()
                    to.commit()

                sq.release()
                sp0.commit()

        @schedule
        def mma_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            skv: SmemKVResource,
            sp0: TmemSPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Shared-buffer captured schedule."""
            mma_schedule_body(gqkv, sq, skv, skv, sp0, to, vd0, wq)

        @schedule
        def mma_split_schedule(
            gqkv: GmemQKVResource,
            sq: SmemQResource,
            sk: SmemKVResource,
            sv: SmemKVResource,
            sp0: TmemSPResource,
            to: TmemOResource,
            vd0: TmemStatsDoneResource,
            wq: WorkQueue | None = None,
        ) -> None:
            """Split K/V captured schedule."""
            mma_schedule_body(gqkv, sq, sk, sv, sp0, to, vd0, wq)

        captured_schedule = schedule_with_work_queue(
            mma_split_schedule if split_kv else mma_schedule,
            gmem_qkv,
            smem_q,
            *kv_resources,
            tmem_sp0,
            tmem_o,
            tmem_vec_done_0,
            work_queue=work_queue,
        )
        return task_class(
            src_resources=src,
            dst_resources=[tmem_sp0, tmem_o, tmem_vec_done_0],
            warp_idx=smem_q.cfg.mma_warp_id,
            num_warps=1,
            schedule=captured_schedule,
            name="MmaTask",
            num_registers=smem_q.cfg.num_regs_other,
            **task_kwargs,
        )

    if tmem_sp1 is None or tmem_vec_done_1 is None:
        raise ValueError("paired MMA scheduling requires peer-1 resources")

    # Padded QK is at most 256 for the admitted paired geometries. Keep each
    # 128-wide slice in its own TS binding so later slices cannot replace it.
    # The stage loops below handle both the full and partial final slice.
    if num_head_dim_stages_k > 2:
        raise ValueError("paired QK staging supports at most two 128-wide slices")

    def qk_mma_slice(sp, desc_q_base, desc_k, section, head_dim_stage_idx):
        """Route a slice through its own descriptor binding to the shared QK body."""
        if head_dim_stage_idx == 0:
            sp.qk_mma(
                desc_q_base=desc_q_base,
                desc_k_base=desc_k,
                section=section,
                head_dim_stage_idx=head_dim_stage_idx,
            )
        else:
            sp.qk_mma_stage(
                desc_q_base=desc_q_base,
                desc_k_stage=desc_k,
                section=section,
                head_dim_stage_idx=head_dim_stage_idx,
            )

    def load_k_stage(sk, head_dim_stage_idx):
        """Retain the descriptor of each waited K slice for both query tiles."""
        sk.wait()
        if head_dim_stage_idx == 0:
            return sk.k_desc()
        return sk.k_stage_desc()

    def seed_scores(sp):
        """Seed an acquired S stage with the INT32 score bias before its K wait."""
        if smem_q.cfg.uses_int32_scores:
            sp.seed_scores()

    def qk_mma_stages(sk, sp, desc_q_base, section, *, descriptors=None):
        """Issue all K slices, loading descriptors once and reusing them for Q1."""
        stages = []
        for head_dim_stage_idx in range(num_head_dim_stages_k):
            desc_k = (
                load_k_stage(sk, head_dim_stage_idx)
                if descriptors is None
                else descriptors[head_dim_stage_idx]
            )
            qk_mma_slice(sp, desc_q_base, desc_k, section, head_dim_stage_idx)
            stages.append(desc_k)
        return stages

    def mma_schedule_body(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        wq: WorkQueue | None = None,
        pr0: TmemPPrefixReadyResource | None = None,
        pr1: TmemPPrefixReadyResource | None = None,
        pb0: SmemPResource | None = None,
        pb1: SmemPResource | None = None,
        smu: SmemMuResource | None = None,
    ) -> None:
        """Interleave paired QK/PV while retaining each K slice for both Qs.

        MXFP8 Q/K keeps each QK step's scales in the tail of the peer S tile.
        S1 is acquired before the first query tile's step and S0 again before
        the second's, which waits for softmax0's load of S0. The repeated
        acquire of a stage not yet committed passes at once.
        """
        # VC-Attention: the current tile's mean-operand descriptor, waited and
        # released in lockstep with V.
        mu_state: dict[str, Any] = {"desc": None}

        def pv(
            sp: TmemSPResource,
            pr: TmemPPrefixReadyResource | None,
            pb: SmemPResource | None,
            *,
            group: int,
            **kw: Any,
        ) -> None:
            """PV for query group ``group``. P-ready comes from the SMEM P tile with
            P in SMEM, from the P-prefix barrier with the half overlap, else from the SP acquire."""
            if p_in_smem:
                pb.wait()
                to.pv_mma(inst_idx=group, **kw)
                if smu is not None:
                    # VC-Attention: a group's row sums travel with the P of the
                    # tile after it, so its mean step follows that tile's PV steps.
                    to.vc_mean_mma(
                        inst_idx=group,
                        desc_mu_base=mu_state["desc"],
                        is_tail=kw.get("is_tail", False),
                    )
                pb.release()
                return
            if pv_half_overlap:
                pr.wait()
                to.pv_mma(k_half=0, **kw)
                pr.release()
            sp.acquire()
            sp.p_read()
            to.pv_mma(k_half=1 if pv_half_overlap else None, **kw)

        sq.init_descriptor_state()
        sk.init_descriptor_state()
        sv.init_descriptor_state()
        if smu is not None:
            smu.init_descriptor_state()
        sp0.init_mma_state()
        sp1.init_mma_state()
        to.init_mma_state()
        with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            v_seqlen_k = Int32(0)
            v_kv_tile_start = Int32(0)
            if cutlass.const_expr(smem_q.cfg.use_paged_kv):
                (
                    _seq_coord,
                    _head_coord,
                    _kv_head_coord,
                    _head_coord_kv,
                    _batch_coord,
                    _seq_coord_q,
                    _cuseqlen_q,
                    _cuseqlen_k,
                    _seqlen_q,
                    v_seqlen_k,
                    v_kv_tile_start,
                    kv_request_begin,
                    _kv_page_idx_lb,
                    kv_page_idx_ub,
                ) = gqkv.compute_coords()
            # HEAD: consume Q0, K0, Q1, and V0. TmemStatsDone starts empty, so
            # the first acquire succeeds without priming. On later work tiles,
            # correction has released the previous stats slot.
            #
            # Consume Q0, K0, then QK(Q0,K0)→S0.
            sq.wait()
            desc_q0_base = sq.q0_desc(inst_idx=0)
            for head_dim_stage_idx in range(num_head_dim_stages_k):
                desc_k = load_k_stage(sk, head_dim_stage_idx)
                if head_dim_stage_idx == 0:
                    if not smem_q.cfg.stats_via_smem:
                        vd0.acquire()
                    sp0.acquire()
                    seed_scores(sp0)
                    if smem_q.cfg.use_mxfp8_qk:
                        sp1.acquire()
                qk_mma_slice(
                    sp0, desc_q0_base, desc_k, WorkStage.Head, head_dim_stage_idx
                )
                if head_dim_stage_idx == num_head_dim_stages_k - 1:
                    sp0.commit()
                    if not smem_q.cfg.stats_via_smem:
                        vd0.commit()
                if head_dim_stage_idx == 0:
                    sq.wait()
                    desc_q1_base = sq.q1_desc(inst_idx=1)
                    if not smem_q.cfg.stats_via_smem:
                        vd1.acquire()
                    sp1.acquire()
                    if smem_q.cfg.use_mxfp8_qk:
                        sp0.acquire()
                    seed_scores(sp1)
                qk_mma_slice(
                    sp1, desc_q1_base, desc_k, WorkStage.Head, head_dim_stage_idx
                )
                if head_dim_stage_idx > 0:
                    sk.release()
            sp1.commit()
            if not smem_q.cfg.stats_via_smem:
                vd1.commit()
            # Q0/Q1 stay live because UMMA reads Q throughout the K-loop.
            # Release K0 (done with QK→S0 and QK→S1), then consume V0.
            sk.release()
            # TODO: sv.wait() (V0) only depends on V's own TMA-load barrier,
            # not on sk/sp0/sp1, so it could move earlier to overlap with the
            # QK0/QK1 work above.
            sv.wait()
            if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                desc_v_base = sv.v_desc_paged(
                    section=WorkStage.Head,
                    seqlen_k=v_seqlen_k,
                    kv_tile_start=v_kv_tile_start,
                )
            else:
                desc_v_base = sv.v_desc()
            # Acquire O first (off critical path), then acquire SP0 and run PV→O0.
            if not p_in_smem:
                to.acquire()
                pv(
                    sp0,
                    pr0,
                    pb0,
                    group=0,
                    desc_v_base=desc_v_base,
                    section=WorkStage.Head,
                )
                to.commit()

            def wait_next_v() -> Any:
                """Release Ki+1, then wait Vi+1 and build its descriptor."""
                for _ in range(num_head_dim_stages_k):
                    sk.release()
                # TODO: sv.wait() (Vi+1) could move earlier in this iteration
                # to overlap with QK0/PV1/QK1.
                sv.wait()
                if cutlass.const_expr(smem_q.cfg.needs_paged_v_tail_clear):
                    return sv.v_desc_paged(
                        section=WorkStage.Loop,
                        tile_offset=1,
                        seqlen_k=v_seqlen_k,
                        kv_tile_start=v_kv_tile_start,
                    )
                return sv.v_desc()

            def release_mu() -> None:
                """Release the mean operand of the tile whose PV steps were issued."""
                if smu is not None:
                    smu.release()

            def wait_next_mu() -> None:
                """Wait for the next tile's mean operand and cache its descriptor."""
                if smu is not None:
                    smu.wait()
                    mu_state["desc"] = smu.mu_desc()

            if p_in_smem:
                # With P in SMEM, softmax releases the S stage right after loading
                # S, so QK(i+1) overlaps exp2(i). The MMA issues QK0(i+1), PV0(i),
                # QK1(i+1), PV1(i). Each QK acquires its SP slot here and each PV
                # waits on its group's P tile instead. The tail drains no SP slot.
                with domain_loop(loop_start, loop_end, loop_step):
                    sp0.acquire()
                    seed_scores(sp0)
                    if smem_q.cfg.use_mxfp8_qk:
                        sp1.acquire()
                    k_descriptors = qk_mma_stages(sk, sp0, desc_q0_base, WorkStage.Loop)
                    sp0.commit()
                    # Wait for the mean operand here, off the QK issue path.
                    wait_next_mu()
                    to.acquire()
                    pv(
                        sp0,
                        pr0,
                        pb0,
                        group=0,
                        desc_v_base=desc_v_base,
                        section=WorkStage.Loop,
                    )
                    to.commit()
                    sp1.acquire()
                    if smem_q.cfg.use_mxfp8_qk:
                        sp0.acquire()
                    seed_scores(sp1)
                    qk_mma_stages(
                        sk, sp1, desc_q1_base, WorkStage.Loop, descriptors=k_descriptors
                    )
                    sp1.commit()
                    to.acquire()
                    pv(
                        sp1,
                        pr1,
                        pb1,
                        group=1,
                        desc_v_base=desc_v_base,
                        section=WorkStage.Loop,
                    )
                    to.commit()
                    sv.release()
                    release_mu()
                    desc_v_base = wait_next_v()
                sq.release()
                sq.release()
                if smu is not None:
                    # The loop issued the mean step of group g at the first tile
                    # after it, so with N groups it covered 0 .. N - 2. The tail
                    # O window of each query group adds group N - 1 from the
                    # operand the loader fetched after the loop. Both ring slots
                    # are held until then.
                    wait_next_mu()
                    smu.wait()
                    desc_mu_last = smu.mu_desc_last()

                def tail_mean(group: int, pb: Any) -> None:
                    """Issue the final mean step of ``group`` inside its O window."""
                    if smu is not None:
                        pb.wait()
                        to.vc_mean_mma_last(inst_idx=group, desc_mu_last=desc_mu_last)
                        pb.release()

                to.acquire()
                pv(
                    sp0,
                    pr0,
                    pb0,
                    group=0,
                    desc_v_base=desc_v_base,
                    section=WorkStage.Tail,
                    is_tail=True,
                )
                tail_mean(0, pb0)
                to.commit()
                to.acquire()
                pv(
                    sp1,
                    pr1,
                    pb1,
                    group=1,
                    desc_v_base=desc_v_base,
                    section=WorkStage.Tail,
                    is_tail=True,
                )
                tail_mean(1, pb1)
                to.commit()
                sv.release()
                release_mu()
                release_mu()
            else:
                # QK0 -> PV1 -> QK1 -> PV0: retain each K slice and the previous V
                # until both peer query tiles have consumed the corresponding data.
                with domain_loop(loop_start, loop_end, loop_step):
                    k_descriptors = qk_mma_stages(sk, sp0, desc_q0_base, WorkStage.Loop)
                    sp0.commit()
                    to.acquire()
                    pv(
                        sp1,
                        pr1,
                        pb1,
                        group=1,
                        desc_v_base=desc_v_base,
                        section=WorkStage.Loop,
                    )
                    to.commit()
                    # Release V_prev after PV1 UMMA consumed SMEM data.
                    sv.release()
                    qk_mma_stages(
                        sk, sp1, desc_q1_base, WorkStage.Loop, descriptors=k_descriptors
                    )
                    sp1.commit()
                    desc_v_base = wait_next_v()
                    # PV0: P0 * Vi+1 → O0.
                    to.acquire()
                    pv(
                        sp0,
                        pr0,
                        pb0,
                        group=0,
                        desc_v_base=desc_v_base,
                        section=WorkStage.Loop,
                        inst_idx=1,
                    )
                    to.commit()

                sq.release()
                sq.release()
                sp0.commit()
                to.acquire()
                pv(
                    sp1,
                    pr1,
                    pb1,
                    group=1,
                    desc_v_base=desc_v_base,
                    section=WorkStage.Tail,
                    is_tail=True,
                )
                to.commit()
                sv.release()
                sp1.commit()

    @schedule
    def mma_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Shared-buffer captured schedule."""
        mma_schedule_body(gqkv, sq, skv, skv, sp0, sp1, to, vd0, vd1, wq)

    @schedule
    def mma_split_schedule(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K/V captured schedule."""
        mma_schedule_body(gqkv, sq, sk, sv, sp0, sp1, to, vd0, vd1, wq)

    # Overlap variants carry the two P-prefix barriers.
    @schedule
    def mma_schedule_pr(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pr0: TmemPPrefixReadyResource,
        pr1: TmemPPrefixReadyResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Shared-buffer captured schedule with P-prefix overlap."""
        mma_schedule_body(gqkv, sq, skv, skv, sp0, sp1, to, vd0, vd1, wq, pr0, pr1)

    @schedule
    def mma_split_schedule_pr(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pr0: TmemPPrefixReadyResource,
        pr1: TmemPPrefixReadyResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K/V captured schedule with P-prefix overlap."""
        mma_schedule_body(gqkv, sq, sk, sv, sp0, sp1, to, vd0, vd1, wq, pr0, pr1)

    # P-in-SMEM variants carry the two SMEM P tiles.
    @schedule
    def mma_split_schedule_pb(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pb0: SmemPResource,
        pb1: SmemPResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K/V captured schedule with P staged in SMEM."""
        mma_schedule_body(
            gqkv, sq, sk, sv, sp0, sp1, to, vd0, vd1, wq, None, None, pb0, pb1
        )

    @schedule
    def mma_schedule_pb(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pb0: SmemPResource,
        pb1: SmemPResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Shared-buffer captured schedule with P staged in SMEM."""
        mma_schedule_body(
            gqkv, sq, skv, skv, sp0, sp1, to, vd0, vd1, wq, None, None, pb0, pb1
        )

    @schedule
    def mma_split_schedule_pb_mu(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        sk: SmemKVResource,
        sv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pb0: SmemPResource,
        pb1: SmemPResource,
        smu: SmemMuResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Split K/V captured schedule with P in SMEM and VC tile means."""
        mma_schedule_body(
            gqkv, sq, sk, sv, sp0, sp1, to, vd0, vd1, wq, None, None, pb0, pb1, smu
        )

    @schedule
    def mma_schedule_pb_mu(
        gqkv: GmemQKVResource,
        sq: SmemQResource,
        skv: SmemKVResource,
        sp0: TmemSPResource,
        sp1: TmemSPResource,
        to: TmemOResource,
        vd0: TmemStatsDoneResource,
        vd1: TmemStatsDoneResource,
        pb0: SmemPResource,
        pb1: SmemPResource,
        smu: SmemMuResource,
        wq: WorkQueue | None = None,
    ) -> None:
        """Shared-buffer captured schedule with P in SMEM and VC tile means."""
        mma_schedule_body(
            gqkv, sq, skv, skv, sp0, sp1, to, vd0, vd1, wq, None, None, pb0, pb1, smu
        )

    if smem_mu is not None:
        selected_mma_schedule = (
            mma_split_schedule_pb_mu if split_kv else mma_schedule_pb_mu
        )
    elif p_in_smem:
        selected_mma_schedule = mma_split_schedule_pb if split_kv else mma_schedule_pb
    elif pv_half_overlap:
        selected_mma_schedule = mma_split_schedule_pr if split_kv else mma_schedule_pr
    else:
        selected_mma_schedule = mma_split_schedule if split_kv else mma_schedule
    captured_schedule = schedule_with_work_queue(
        selected_mma_schedule,
        gmem_qkv,
        smem_q,
        *kv_resources,
        tmem_sp0,
        tmem_sp1,
        tmem_o,
        tmem_vec_done_0,
        tmem_vec_done_1,
        *p_prefix_resources,
        *smem_p_resources,
        *mu_resources,
        work_queue=work_queue,
    )
    return task_class(
        src_resources=src,
        dst_resources=[tmem_sp0, tmem_sp1, tmem_o]
        + ([] if smem_q.cfg.stats_via_smem else [tmem_vec_done_0, tmem_vec_done_1]),
        warp_idx=12,
        num_warps=1,
        schedule=captured_schedule,
        name="MmaTask",
        num_registers=smem_q.cfg.num_regs_other,
        **task_kwargs,
    )
