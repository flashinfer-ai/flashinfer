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

"""MMA tasks: the QK and PV tcgen05 MMAs."""

import cutlass
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.task import Task
from ..config import FmhaDecodeConfig
from ..device.constants import KV_INST0, KV_INST1
from .common import (
    DecodeGenTask,
    PackedDecodeWorkQueue,
    TaskKwarg,
    schedule_with_optional_resources,
    work_queue_tail,
)
from .....device.stage import WorkStage
from typing import Any, Callable


def consume_staged_pv_mma(
    smem_kv: MemoryResource,
    tmem_p: MemoryResource,
    tmem_o: MemoryResource,
    v_desc_label: str,
    vp_mma_label: str,
    p_desc_idx: int,
    section: WorkStage,
    cfg: FmhaDecodeConfig,
) -> None:
    """Consume all V head-dim stages for one PV MMA wave."""
    _ = section
    if cutlass.const_expr(cfg.streams_tmem_p_fragments):
        consume_streamed_pv_fragments(
            smem_kv, tmem_p, tmem_o, v_desc_label, vp_mma_label, cfg
        )
        return

    tmem_p.wait()
    p_desc_0, p_desc_1, p_tmem_addr_0, p_tmem_addr_1 = tmem_p.p_operands()
    p_desc = p_desc_0 if p_desc_idx == KV_INST0 else p_desc_1
    p_tmem_addr = p_tmem_addr_0 if p_desc_idx == KV_INST0 else p_tmem_addr_1
    tmem_o.acquire()
    slot_stages = transformed_kv_slot_stages(cfg)
    if cfg.uses_fused_slot_mma_issue:
        smem_kv.wait()
        v_desc = getattr(smem_kv, v_desc_label)()
        getattr(tmem_o, f"{vp_mma_label}_slot")(v_desc=v_desc, p_desc=p_desc)
        smem_kv.release()
        tmem_o.commit()
        tmem_p.release()
        return
    for slot_idx in range(cfg.num_head_dim_stages_kv // slot_stages):
        smem_kv.wait()
        v_desc = getattr(smem_kv, v_desc_label)()
        for sub_stage in range(slot_stages):
            getattr(tmem_o, vp_mma_label)(
                v_desc_0=v_desc,
                v_desc_1=v_desc,
                p_desc_0=p_desc,
                p_desc_1=p_desc,
                p_tmem_addr_0=p_tmem_addr,
                p_tmem_addr_1=p_tmem_addr,
                inst_idx=p_desc_idx,
                head_dim_stage_idx=slot_idx * slot_stages + sub_stage,
            )
        smem_kv.release()
    tmem_o.commit()
    tmem_p.release()


def consume_staged_qk_mma(
    smem_kv: MemoryResource,
    tmem_s: MemoryResource,
    q_desc: Any,
    k_desc_label: str,
    qk_mma_label: str,
    section: WorkStage,
    cfg: FmhaDecodeConfig,
) -> None:
    """Consume all K head-dim stages for one QK MMA wave.

    Streamed KV256 aliases P with the S columns this QK overwrites. The
    preceding same-instance PV reads P as its TMEM A operand from the same
    issuing thread, and the tensor core interlocks that read against a later
    MMA's accumulator write, so no completion wait is needed before QK.
    """
    tmem_s.acquire()
    issue_score_seed(tmem_s, section, cfg)
    # One transformed ring slot may hold several head-dim stages: wait and
    # release once per slot (see transformed_kv_hd_stages_per_slot).
    slot_stages = transformed_kv_slot_stages(cfg)
    if cfg.uses_fused_slot_mma_issue:
        smem_kv.wait()
        kv_desc = getattr(smem_kv, k_desc_label)()
        getattr(tmem_s, f"{qk_mma_label}_slot")(q_desc=q_desc, kv_desc=kv_desc)
        smem_kv.release()
        tmem_s.commit()
        return
    for slot_idx in range(cfg.num_head_dim_stages_kv // slot_stages):
        smem_kv.wait()
        kv_desc = getattr(smem_kv, k_desc_label)()
        for sub_stage in range(slot_stages):
            head_dim_stage_idx = slot_idx * slot_stages + sub_stage
            if cfg.uses_q_desc_ref:
                getattr(tmem_s, f"{qk_mma_label}_from_q_ref")(
                    kv_desc=kv_desc,
                    head_dim_stage_idx=head_dim_stage_idx,
                )
            else:
                getattr(tmem_s, qk_mma_label)(
                    q_desc=q_desc,
                    kv_desc=kv_desc,
                    head_dim_stage_idx=head_dim_stage_idx,
                )
        smem_kv.release()
    tmem_s.commit()


def consume_streamed_pv_fragments(
    smem_kv: MemoryResource,
    tmem_p: MemoryResource,
    tmem_o: MemoryResource,
    v_desc_label: str,
    vp_mma_label: str,
    cfg: FmhaDecodeConfig,
) -> None:
    """Issue one PV wave as its K32 P fragments become ready.

    P fragment 0 is the earliest dependency: wait for it and for the
    correction credit before holding the V stage. Later fragments may become
    ready while the previous PV fragment is already executing; every slot
    stays live through the complete async UMMA wave so the producer cannot
    overwrite an operand prematurely.
    """
    assert cfg.num_head_dim_stages_kv == 1
    fragment_label = f"{vp_mma_label}_fragment"
    p_tmem_addr = tmem_p.wait_p_fragment(fragment_idx=0)
    tmem_o.acquire()
    smem_kv.wait()
    v_desc = getattr(smem_kv, v_desc_label)()
    getattr(tmem_o, fragment_label)(
        v_desc=v_desc,
        p_tmem_addr=p_tmem_addr,
        fragment_idx=0,
    )
    for fragment_idx in range(1, cfg.num_softmax_score_fragments):
        p_tmem_addr = tmem_p.wait_p_fragment(fragment_idx=fragment_idx)
        getattr(tmem_o, fragment_label)(
            v_desc=v_desc,
            p_tmem_addr=p_tmem_addr,
            fragment_idx=fragment_idx,
        )
    smem_kv.release()
    tmem_o.commit()


def create_mma_task(
    smem_q: MemoryResource,
    smem_kv: MemoryResource,
    tmem_s0: MemoryResource,
    tmem_s1: MemoryResource | None,
    smem_p0: MemoryResource,
    smem_p1: MemoryResource | None,
    tmem_o: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    smem_kv1: MemoryResource | None = None,
    **kw: TaskKwarg,
) -> Task:
    """Create the MMA task for the shared K/V ring path.

    ``smem_kv1``, when given, supplies instance 1's K/V descriptors from its
    own ring (per-instance converters); otherwise both instances read
    ``smem_kv``.
    """

    def mma_schedule_body(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        smem_kv1: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource | None,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource | None,
        tmem_o: MemoryResource,
        q_desc: Any,
    ) -> None:
        """Schedule shared-KV QK and PV waves across HEAD/LOOP/TAIL."""
        if cfg.num_insts_kv != 1:
            assert smem_p1 is not None
            assert tmem_s1 is not None
        # HEAD: consume Q once and launch BMM1 for each active instance.
        consume_staged_qk_mma(
            smem_kv,
            tmem_s0,
            q_desc,
            "k_desc_0",
            "qk_mma_head",
            WorkStage.Head,
            cfg,
        )
        if cfg.num_insts_kv != 1:
            consume_staged_qk_mma(
                smem_kv1,
                tmem_s1,
                q_desc,
                "k_desc_1",
                "qk_mma_head",
                WorkStage.Head,
                cfg,
            )

        # LOOP: consume aliased TMEM P before the next same-instance QK
        # overwrites its S columns. Full-SMEM P remains score-dependent during
        # Softmax replay, but owns independent storage once the replay commits
        # and releases S; that completed handoff enables the QK-before-PV cadence.
        with domain_loop(0, domain, 1, unroll=1):
            if cfg.uses_two_inst_tmem_p:
                consume_staged_pv_mma(
                    smem_kv,
                    smem_p0,
                    tmem_o,
                    "v_desc_0",
                    "vp_mma_loop",
                    KV_INST0,
                    WorkStage.Loop,
                    cfg,
                )
            consume_staged_qk_mma(
                smem_kv,
                tmem_s0,
                q_desc,
                "k_desc_0",
                "qk_mma_loop",
                WorkStage.Loop,
                cfg,
            )
            if not cfg.uses_two_inst_tmem_p:
                consume_staged_pv_mma(
                    smem_kv,
                    smem_p0,
                    tmem_o,
                    "v_desc_0",
                    "vp_mma_loop",
                    KV_INST0,
                    WorkStage.Loop,
                    cfg,
                )
            if cfg.num_insts_kv != 1:
                if cfg.uses_two_inst_tmem_p:
                    consume_staged_pv_mma(
                        smem_kv1,
                        smem_p1,
                        tmem_o,
                        "v_desc_1",
                        "vp_mma_loop",
                        KV_INST1,
                        WorkStage.Loop,
                        cfg,
                    )
                consume_staged_qk_mma(
                    smem_kv1,
                    tmem_s1,
                    q_desc,
                    "k_desc_1",
                    "qk_mma_loop",
                    WorkStage.Loop,
                    cfg,
                )
                if not cfg.uses_two_inst_tmem_p:
                    consume_staged_pv_mma(
                        smem_kv1,
                        smem_p1,
                        tmem_o,
                        "v_desc_1",
                        "vp_mma_loop",
                        KV_INST1,
                        WorkStage.Loop,
                        cfg,
                    )

        # Q is live for every BMM1 call, and the last BMM1 has been issued once
        # the loop ends. Releasing here commits after those MMAs complete, so
        # the next tile's Q load overlaps the final softmax and BMM2 waves
        # instead of waiting for them.
        smem_q.release()

        # TAIL: no future K tiles remain, so only the final BMM2 waves run.
        consume_staged_pv_mma(
            smem_kv,
            smem_p0,
            tmem_o,
            "v_desc_0",
            "vp_mma_tail",
            KV_INST0,
            WorkStage.Tail,
            cfg,
        )
        if cfg.num_insts_kv != 1:
            consume_staged_pv_mma(
                smem_kv1,
                smem_p1,
                tmem_o,
                "v_desc_1",
                "vp_mma_tail",
                KV_INST1,
                WorkStage.Tail,
                cfg,
            )

    def mma_schedule_prelude(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        smem_kv1: MemoryResource | None,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource | None,
    ) -> None:
        """Initialize invariant shared-ring descriptor slots."""
        smem_q.init_descriptor_state()
        smem_kv.init_descriptor_state()
        if smem_kv1 is not None:
            smem_kv1.init_descriptor_state()
        smem_p0.init_descriptor_state()
        if cfg.num_insts_kv != 1:
            assert smem_p1 is not None
            smem_p1.init_descriptor_state()

    @schedule_with_optional_resources
    def mma_schedule(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        smem_kv1: MemoryResource | None,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource | None,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource | None,
        tmem_o: MemoryResource,
        work_queue: WorkQueue | None,
    ) -> None:
        """Wrap shared-ring MMA work in packed persistent skip handling."""
        decode_work_tile_schedule_with_invariant_bridge(
            cfg,
            work_queue,
            lambda: mma_schedule_prelude(
                smem_q,
                smem_kv,
                smem_kv1,
                smem_p0,
                smem_p1,
            ),
            lambda: smem_q.q_desc(),
            lambda: smem_q.wait(),
            lambda q_desc: mma_schedule_body(
                smem_q,
                smem_kv,
                smem_kv if smem_kv1 is None else smem_kv1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                q_desc,
            ),
        )

    captured_schedule = mma_schedule(
        smem_q,
        smem_kv,
        smem_kv1,
        tmem_s0,
        tmem_s1,
        smem_p0,
        smem_p1,
        tmem_o,
        work_queue,
    )
    src = [smem_q, smem_kv, smem_p0]
    if smem_kv1 is not None:
        src.append(smem_kv1)
    dst = [tmem_s0, tmem_o]
    if cfg.num_insts_kv != 1:
        src.append(smem_p1)
        dst.insert(1, tmem_s1)
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx,
        num_warps=cfg.mma_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_task_num_registers,
        name="MmaTask",
        **kw,
    )


def create_mma_task_dual_transform_kv(
    smem_q: MemoryResource,
    tmem_k: MemoryResource,
    tmem_v: MemoryResource,
    tmem_s0: MemoryResource,
    smem_p0: MemoryResource,
    tmem_o: MemoryResource,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    **kw: TaskKwarg,
) -> Task:
    """Create the MMA task that reads K and V from separate converted FIFOs.

    HEAD issues QK for the first tile. Each LOOP iteration issues QK for the
    next tile before PV for the current one, so QK can proceed while Softmax
    still owns the current P. TAIL issues the last PV after releasing Q.
    """
    assert cfg.uses_dual_kv_transform

    @schedule
    def mma_schedule(
        smem_q: MemoryResource,
        tmem_k: MemoryResource,
        tmem_v: MemoryResource,
        tmem_s0: MemoryResource,
        smem_p0: MemoryResource,
        tmem_o: MemoryResource,
    ):
        """Issue QK from the K FIFO and PV from the V FIFO for each work tile."""

        def prelude():
            """Initialize descriptor slots that stay valid across work tiles."""
            smem_q.init_descriptor_state()
            tmem_k.init_descriptor_state()
            tmem_v.init_descriptor_state()
            smem_p0.init_descriptor_state()

        def body(q_desc):
            """Issue the HEAD/LOOP/TAIL QK and PV waves for one work tile."""
            consume_staged_qk_mma(
                tmem_k, tmem_s0, q_desc, "k_desc_0", "qk_mma_head", WorkStage.Head, cfg
            )
            with domain_loop(0, domain, 1, unroll=1):
                consume_staged_qk_mma(
                    tmem_k,
                    tmem_s0,
                    q_desc,
                    "k_desc_0",
                    "qk_mma_loop",
                    WorkStage.Loop,
                    cfg,
                )
                consume_staged_pv_mma(
                    tmem_v,
                    smem_p0,
                    tmem_o,
                    "v_desc_0",
                    "vp_mma_loop",
                    0,
                    WorkStage.Loop,
                    cfg,
                )
            smem_q.release()
            consume_staged_pv_mma(
                tmem_v,
                smem_p0,
                tmem_o,
                "v_desc_0",
                "vp_mma_tail",
                0,
                WorkStage.Tail,
                cfg,
            )

        decode_work_tile_schedule_with_invariant_bridge(
            cfg, None, prelude, lambda: smem_q.q_desc(), lambda: smem_q.wait(), body
        )

    return DecodeGenTask(
        src_resources=[smem_q, tmem_k, tmem_v, smem_p0],
        dst_resources=[tmem_s0, tmem_o],
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx,
        num_warps=cfg.mma_num_warps,
        schedule=mma_schedule(smem_q, tmem_k, tmem_v, tmem_s0, smem_p0, tmem_o),
        num_registers=cfg.mma_load_task_num_registers,
        name="MmaTask",
        **kw,
    )


def create_mma_task_one_inst_qkv(
    smem_q: MemoryResource,
    smem_k: MemoryResource,
    smem_v: MemoryResource,
    tmem_s: MemoryResource,
    smem_p: MemoryResource,
    tmem_o: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    tmem_stats_done: MemoryResource | None,
    domain: int | cutlass.Int32,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the one-inst QKV MMA task."""

    def mma_schedule_body(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        tmem_s: MemoryResource,
        smem_p: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done: MemoryResource | None,
        q_desc: Any,
    ) -> None:
        """Schedule single-instance QK and PV waves across HEAD/LOOP/TAIL."""

        def qk_mma(q_desc, qk_mma_label: str, section: WorkStage) -> None:
            """Issue one scheduled single-instance QK wave."""
            if tmem_stats_done is not None:
                tmem_stats_done.acquire()
            tmem_s.acquire()
            issue_score_seed(tmem_s, section, cfg)
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                smem_k.wait()
                kv_desc = smem_k.kv_desc()
                if cfg.uses_q_desc_ref:
                    getattr(tmem_s, f"{qk_mma_label}_from_q_ref")(
                        kv_desc=kv_desc,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                else:
                    getattr(tmem_s, qk_mma_label)(
                        q_desc=q_desc,
                        kv_desc=kv_desc,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                smem_k.release()
            tmem_s.commit()
            if tmem_stats_done is not None:
                tmem_stats_done.commit()

        def pv_mma(vp_mma_label: str, section: WorkStage) -> None:
            """Issue one scheduled single-instance PV wave."""
            _ = section
            smem_p.wait()
            p_desc_0, p_desc_1, p_tmem_addr_0, p_tmem_addr_1 = smem_p.p_operands()
            tmem_o.acquire()
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                smem_v.wait()
                v_desc = smem_v.v_desc()
                getattr(tmem_o, vp_mma_label)(
                    v_desc_0=v_desc,
                    v_desc_1=v_desc,
                    p_desc_0=p_desc_0,
                    p_desc_1=p_desc_1,
                    p_tmem_addr_0=p_tmem_addr_0,
                    p_tmem_addr_1=p_tmem_addr_1,
                    inst_idx=KV_INST0,
                    head_dim_stage_idx=head_dim_stage_idx,
                )
                smem_v.release()
            tmem_o.commit()
            smem_p.release()

        qk_mma(q_desc, "qk_mma_head", WorkStage.Head)

        with domain_loop(0, domain, 1, unroll=1):
            qk_mma(q_desc, "qk_mma_loop", WorkStage.Loop)
            pv_mma("vp_mma_loop", WorkStage.Loop)

        # BMM2 never reads Q; release it before the final wave
        # (see create_mma_task_split_kv).
        smem_q.release()
        pv_mma("vp_mma_tail", WorkStage.Tail)

    def mma_schedule_prelude(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        smem_p: MemoryResource,
    ) -> None:
        """Initialize invariant one-inst descriptor slots."""
        smem_q.init_descriptor_state()
        smem_k.init_descriptor_state()
        smem_v.init_descriptor_state()
        smem_p.init_descriptor_state()

    @schedule_with_optional_resources
    def mma_schedule(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        tmem_s: MemoryResource,
        smem_p: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Wrap one-inst MMA work in packed persistent skip handling."""
        decode_work_tile_schedule_with_invariant_bridge(
            cfg,
            work_queue,
            lambda: mma_schedule_prelude(smem_q, smem_k, smem_v, smem_p),
            lambda: smem_q.q_desc(),
            lambda: smem_q.wait(),
            lambda q_desc: mma_schedule_body(
                smem_q,
                smem_k,
                smem_v,
                tmem_s,
                smem_p,
                tmem_o,
                tmem_stats_done,
                q_desc,
            ),
        )

    schedule_result = mma_schedule(
        smem_q, smem_k, smem_v, tmem_s, smem_p, tmem_o, tmem_stats_done, work_queue
    )
    src = [smem_q, smem_k, smem_v, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s, tmem_o]
        + ([tmem_stats_done] if tmem_stats_done is not None else []),
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.mma_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="MmaTask",
        **kw,
    )


def create_mma_task_split_kv(
    smem_q: MemoryResource,
    smem_k0: MemoryResource,
    smem_k1: MemoryResource,
    smem_v0: MemoryResource,
    smem_v1: MemoryResource,
    tmem_s0: MemoryResource,
    tmem_s1: MemoryResource,
    smem_p0: MemoryResource,
    smem_p1: MemoryResource,
    tmem_o: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    tmem_stats_done0: MemoryResource | None = None,
    tmem_stats_done1: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the MMA task for split K/V resources."""

    def mma_schedule_body(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done0: MemoryResource | None,
        tmem_stats_done1: MemoryResource | None,
        q_desc: Any,
    ) -> None:
        """Schedule split-resource QK and PV waves across HEAD/LOOP/TAIL."""

        def qk_mma(
            smem_kv: MemoryResource,
            tmem_s: MemoryResource,
            tmem_stats_done: MemoryResource | None,
            q_desc,
            qk_mma_label: str,
            section: WorkStage,
        ) -> None:
            """Issue one scheduled QK wave using the selected phase work."""
            if tmem_stats_done is not None:
                tmem_stats_done.acquire()
            tmem_s.acquire()
            issue_score_seed(tmem_s, section, cfg)
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                smem_kv.wait()
                kv_desc = smem_kv.kv_desc()
                if cfg.uses_q_desc_ref:
                    getattr(tmem_s, f"{qk_mma_label}_from_q_ref")(
                        kv_desc=kv_desc,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                else:
                    getattr(tmem_s, qk_mma_label)(
                        q_desc=q_desc,
                        kv_desc=kv_desc,
                        head_dim_stage_idx=head_dim_stage_idx,
                    )
                smem_kv.release()
            tmem_s.commit()
            if tmem_stats_done is not None:
                tmem_stats_done.commit()

        def pv_mma(
            smem_kv: MemoryResource,
            tmem_p: MemoryResource,
            vp_mma_label: str,
            inst_idx: int,
            section: WorkStage,
        ) -> None:
            """Issue one scheduled PV wave using the selected phase work."""
            consume_staged_pv_mma(
                smem_kv, tmem_p, tmem_o, "v_desc", vp_mma_label, inst_idx, section, cfg
            )

        qk_mma(
            smem_k0,
            tmem_s0,
            tmem_stats_done0,
            q_desc,
            "qk_mma_head",
            WorkStage.Head,
        )
        qk_mma(
            smem_k1,
            tmem_s1,
            tmem_stats_done1,
            q_desc,
            "qk_mma_head",
            WorkStage.Head,
        )

        # TMEM P aliases S, so an instance's next QK must follow the PV that
        # reads P. A Q128 tile with SMEM P leaves S once Softmax has the scores
        # in registers: the next QK goes first and overlaps the exponentials
        # of this tile (``stages_q128_p_in_smem``).
        qk_before_pv = cfg.stages_q128_p_in_smem
        with domain_loop(0, domain, 1, unroll=1):
            for inst_idx, smem_k, smem_v, tmem_s, tmem_stats_done, smem_p in (
                (KV_INST0, smem_k0, smem_v0, tmem_s0, tmem_stats_done0, smem_p0),
                (KV_INST1, smem_k1, smem_v1, tmem_s1, tmem_stats_done1, smem_p1),
            ):
                if not qk_before_pv:
                    pv_mma(smem_v, smem_p, "vp_mma_loop", inst_idx, WorkStage.Loop)
                qk_mma(
                    smem_k,
                    tmem_s,
                    tmem_stats_done,
                    q_desc,
                    "qk_mma_loop",
                    WorkStage.Loop,
                )
                if qk_before_pv:
                    pv_mma(smem_v, smem_p, "vp_mma_loop", inst_idx, WorkStage.Loop)

        # The last BMM1 has been issued once the loop ends, and BMM2 never reads
        # Q. Releasing it here lets a persistent CTA load the next work tile's
        # Q while the final softmax and BMM2 waves run, as the KV256 MMA does.
        smem_q.release()

        pv_mma(smem_v0, smem_p0, "vp_mma_tail", KV_INST0, WorkStage.Tail)
        pv_mma(smem_v1, smem_p1, "vp_mma_tail", KV_INST1, WorkStage.Tail)

    def mma_schedule_prelude(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
    ) -> None:
        """Initialize invariant split-resource descriptor slots."""
        smem_q.init_descriptor_state()
        for resource in (smem_k0, smem_k1, smem_v0, smem_v1):
            if resource is not None:
                resource.init_descriptor_state()
        smem_p0.init_descriptor_state()
        smem_p1.init_descriptor_state()

    def run_mma_schedule(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done0: MemoryResource | None,
        tmem_stats_done1: MemoryResource | None,
        work_queue: WorkQueue | None,
    ) -> None:
        """Wrap split-resource MMA work with optional stats lifetime gates."""
        decode_work_tile_schedule_with_invariant_bridge(
            cfg,
            work_queue,
            lambda: mma_schedule_prelude(
                smem_q, smem_k0, smem_k1, smem_v0, smem_v1, smem_p0, smem_p1
            ),
            lambda: smem_q.q_desc(),
            lambda: smem_q.wait(),
            lambda q_desc: mma_schedule_body(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                tmem_stats_done0,
                tmem_stats_done1,
                q_desc,
            ),
        )

    @schedule
    def mma_schedule(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Capture the Swaps split-resource MMA schedule."""
        run_mma_schedule(
            smem_q,
            smem_k0,
            smem_k1,
            smem_v0,
            smem_v1,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
            None,
            None,
            work_queue,
        )

    @schedule
    def mma_keeps_schedule(
        smem_q: MemoryResource,
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        tmem_s0: MemoryResource,
        tmem_s1: MemoryResource,
        smem_p0: MemoryResource,
        smem_p1: MemoryResource,
        tmem_o: MemoryResource,
        tmem_stats_done0: MemoryResource,
        tmem_stats_done1: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Capture Keeps MMA with explicit stats lifetime gates."""
        run_mma_schedule(
            smem_q,
            smem_k0,
            smem_k1,
            smem_v0,
            smem_v1,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
            tmem_stats_done0,
            tmem_stats_done1,
            work_queue,
        )

    if tmem_stats_done0 is None or tmem_stats_done1 is None:
        schedule_result = (
            mma_schedule(
                smem_q,
                smem_k0,
                smem_k0 if smem_k1 is None else smem_k1,
                smem_v0,
                smem_v0 if smem_v1 is None else smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
            )
            if work_queue is None
            else mma_schedule(
                smem_q,
                smem_k0,
                smem_k0 if smem_k1 is None else smem_k1,
                smem_v0,
                smem_v0 if smem_v1 is None else smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                work_queue,
            )
        )
        dst = [tmem_s0, tmem_s1, tmem_o]
    else:
        schedule_result = (
            mma_keeps_schedule(
                smem_q,
                smem_k0,
                smem_k0 if smem_k1 is None else smem_k1,
                smem_v0,
                smem_v0 if smem_v1 is None else smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                tmem_stats_done0,
                tmem_stats_done1,
            )
            if work_queue is None
            else mma_keeps_schedule(
                smem_q,
                smem_k0,
                smem_k0 if smem_k1 is None else smem_k1,
                smem_v0,
                smem_v0 if smem_v1 is None else smem_v1,
                tmem_s0,
                tmem_s1,
                smem_p0,
                smem_p1,
                tmem_o,
                tmem_stats_done0,
                tmem_stats_done1,
                work_queue,
            )
        )
        dst = [
            tmem_s0,
            tmem_s1,
            tmem_o,
            tmem_stats_done0,
            tmem_stats_done1,
        ]
    src = [
        resource
        for resource in (smem_q, smem_k0, smem_k1, smem_v0, smem_v1, smem_p0, smem_p1)
        if resource is not None
    ]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        cfg=cfg,
        warp_idx=cfg.mma_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.mma_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="MmaTask",
        **kw,
    )


def decode_work_tile_schedule_with_invariant_bridge(
    cfg: FmhaDecodeConfig,
    work_queue: WorkQueue | None,
    invariant_setup: Callable[[], None],
    bridge: Callable[[], Any],
    active_prelude: Callable[[], None],
    body: Callable[[Any], None],
) -> None:
    """Trace data work around an invariant value needed across schedule phases.

    The bridge is reserved for descriptor construction that reads only
    task-local descriptor metadata. It must not issue a memory operation,
    barrier, or pipeline-state transition.
    """
    if cfg.use_persistent_scheduler:
        assert work_queue is not None
        # Descriptor backing storage depends only on the task's SMEM
        # allocation, so initialize it once before the persistent loop.
        invariant_setup()
        with work_tile_loop(
            work_queue,
            skip_if=PackedDecodeWorkQueue.skip_work_tile_if,
        ) as work_tiles:
            with work_tiles.skippable():
                active_prelude()
                # Packed persistent QK reads Q's consumer-work stage directly,
                # so it needs no descriptor task local crossing this guard.
                if cfg.use_variable_seqlens_q:
                    body(None)
                else:
                    invariant = bridge()
                    body(invariant)
            work_queue_tail(work_queue)
        return

    # Fixed paths initialize descriptor state, wait for Q, construct the
    # descriptor, and then issue the remaining MMA work.
    invariant_setup()
    active_prelude()
    invariant = bridge()
    body(invariant)
    work_queue_tail(work_queue)


def issue_score_seed(
    tmem_s: MemoryResource, section: WorkStage, cfg: FmhaDecodeConfig
) -> None:
    """Seed the acquired S slot of an INT32-score QK wave before its K wait.

    The seed MMA reads only constant tiles, so its issue overlaps the K wait.
    """
    if not cfg.uses_int32_scores:
        return
    if section == WorkStage.Head:
        tmem_s.seed_scores_head()
    else:
        tmem_s.seed_scores_loop()


def transformed_kv_slot_stages(cfg: FmhaDecodeConfig) -> int:
    """Head-dim stages per transformed TMEM ring slot (1 for other K/V rings)."""
    if cfg.store_transformed_kv_in_tmem:
        return cfg.transformed_kv_hd_stages_per_slot
    return 1
