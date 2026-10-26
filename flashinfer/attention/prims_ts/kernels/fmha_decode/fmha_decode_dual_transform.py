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

"""Independent K/V conversion and MMA schedules for wide H512 decode."""

import cutlass
from cutlass.experimental.task_scheduling.resources import MemoryResource
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task

from ..stage import FmhaStage
from .fmha_decode_config import FmhaDecodeConfig
from .fmha_decode_tasks import (
    DecodeGenTask,
    TaskKwarg,
    _consume_staged_pv_mma,
    _consume_staged_qk_mma,
    _decode_work_tile_schedule,
    _decode_work_tile_schedule_with_invariant_bridge,
)


def create_single_kv_transform_task(
    smem_kv: MemoryResource,
    tmem_kv: MemoryResource,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    is_v: bool,
    **kw: TaskKwarg,
) -> Task:
    """Convert one operand stream without waiting for the other operand."""
    assert cfg.uses_parallel_kv_transform
    label = "transform_v0" if is_v else "transform_k0"

    @schedule
    def conversion_schedule(smem_kv: MemoryResource, tmem_kv: MemoryResource):
        smem_kv.init_descriptor_state()

        def body():
            tmem_kv.init_load_state()

            def convert_tile():
                for head_stage in range(cfg.num_head_dim_stages_kv):
                    tmem_kv.acquire()
                    smem_kv.wait()
                    getattr(tmem_kv, label)(head_dim_stage_idx=head_stage)
                    tmem_kv.commit()
                    smem_kv.release()

            if not is_v:
                convert_tile()
            with domain_loop(0, domain, 1, unroll=1):
                convert_tile()
            if is_v:
                convert_tile()

        _decode_work_tile_schedule(cfg, None, body)

    return DecodeGenTask(
        src_resources=[smem_kv],
        dst_resources=[tmem_kv],
        cfg=cfg,
        warp_idx=cfg.transform_v_warp_idx if is_v else cfg.transform_kv_warp_idx,
        num_warps=cfg.transform_v_num_warps if is_v else cfg.transform_kv_num_warps,
        schedule=conversion_schedule(smem_kv, tmem_kv),
        num_registers=cfg.transform_v_task_num_registers
        if is_v
        else cfg.transform_kv_task_num_registers,
        name="TransformVTask" if is_v else "TransformKTask",
        **kw,
    )


def create_mma_task_separate_transformed_kv(
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
    """Issue next QK before current PV, with independent converted operands."""
    assert cfg.uses_parallel_kv_transform

    @schedule
    def mma_schedule(
        smem_q: MemoryResource,
        tmem_k: MemoryResource,
        tmem_v: MemoryResource,
        tmem_s0: MemoryResource,
        smem_p0: MemoryResource,
        tmem_o: MemoryResource,
    ):
        def prelude():
            smem_q.init_descriptor_state()
            tmem_k.init_descriptor_state()
            tmem_v.init_descriptor_state()
            smem_p0.init_descriptor_state()

        def body(q_desc):
            _consume_staged_qk_mma(
                tmem_k, tmem_s0, q_desc, "k_desc_0", "qk_mma_head", FmhaStage.Head, cfg
            )
            with domain_loop(0, domain, 1, unroll=1):
                _consume_staged_qk_mma(
                    tmem_k,
                    tmem_s0,
                    q_desc,
                    "k_desc_0",
                    "qk_mma_loop",
                    FmhaStage.Loop,
                    cfg,
                )
                _consume_staged_pv_mma(
                    tmem_v,
                    smem_p0,
                    tmem_o,
                    "v_desc_0",
                    "vp_mma_loop",
                    0,
                    FmhaStage.Loop,
                    cfg,
                )
            smem_q.release()
            _consume_staged_pv_mma(
                tmem_v,
                smem_p0,
                tmem_o,
                "v_desc_0",
                "vp_mma_tail",
                0,
                FmhaStage.Tail,
                cfg,
            )

        _decode_work_tile_schedule_with_invariant_bridge(
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
