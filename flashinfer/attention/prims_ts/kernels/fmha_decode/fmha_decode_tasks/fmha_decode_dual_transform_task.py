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

"""Separate K and V conversion tasks and the MMA task that consumes them.

The shared ``TransformKvTask`` converts the K0, V0, K1, ... stream in order,
so a slow V conversion also delays the next K. With the dual transform, K and
V each own a raw SMEM ring, a converter warp group, and a TMEM FIFO:

    LoadTask --K--> smemK0 --TransformKTask--> tmemTransformedKv --QK--> S
    LoadTask --V--> smemV0 --TransformVTask--> tmemTransformedV  --PV--> O

Both converters walk every head-dimension stage of a KV tile, so the schedule
is independent of the head dimension. ``FmhaDecodeConfig.uses_dual_kv_transform``
selects it where conversion is the bottleneck.
"""

import cutlass
from cutlass.experimental.task_scheduling.resources import MemoryResource
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task

from ...stage import FmhaStage
from ..fmha_decode_config import FmhaDecodeConfig
from .fmha_decode_tasks import (
    DecodeGenTask,
    TaskKwarg,
    _consume_staged_pv_mma,
    _consume_staged_qk_mma,
    _decode_work_tile_schedule,
    _decode_work_tile_schedule_with_invariant_bridge,
)


def create_dual_transform_kv_task(
    smem_kv: MemoryResource,
    tmem_kv: MemoryResource,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    is_v: bool,
    **kw: TaskKwarg,
) -> Task:
    """Create the converter for one operand of the dual K/V transform.

    ``is_v`` selects the V converter (``transform_v_*`` warps, V source ring
    and FIFO); otherwise this is the K converter. With
    ``uses_dual_smem_transform`` (D256 Keeps), ``tmem_kv`` is instead an FP8
    SMEM ring for the Keeps MMA and V uses the ``transform_kv1_*`` warps. The K stream runs one tile
    ahead of V, matching the MMA task: HEAD converts the first K tile, each
    LOOP iteration converts the next K (or the current V), and TAIL converts
    the last V. Every tile converts all ``num_head_dim_stages_kv`` stages.
    """
    assert cfg.uses_dual_kv_transform or cfg.uses_dual_smem_transform
    label = "transform_v0" if is_v else "transform_k0"
    # The SMEM dual transform runs V in the transform_kv1 warp group.
    v_warp_idx = (
        cfg.transform_kv1_warp_idx
        if cfg.uses_dual_smem_transform
        else cfg.transform_v_warp_idx
    )
    v_num_warps = (
        cfg.transform_kv1_num_warps
        if cfg.uses_dual_smem_transform
        else cfg.transform_v_num_warps
    )
    v_num_registers = (
        cfg.transform_kv1_task_num_registers
        if cfg.uses_dual_smem_transform
        else cfg.transform_v_task_num_registers
    )

    # At D512 the four unrolled per-stage bodies of each converter crowd the
    # Q64 softmax out of the instruction cache. Run the steady state once per
    # raw stage instead and select the stage at run time.
    stages = cfg.num_head_dim_stages_kv
    flatten_loop = stages == 4

    @schedule
    def conversion_schedule(smem_kv: MemoryResource, tmem_kv: MemoryResource):
        """Convert one operand's raw ring into its TMEM FIFO for each work tile."""
        smem_kv.init_descriptor_state()

        def body():
            """Convert every tile of one work tile into the TMEM FIFO."""
            tmem_kv.init_load_state()

            def convert_stage(work):
                """Convert one raw stage into one TMEM FIFO slot."""
                tmem_kv.acquire()
                smem_kv.wait()
                work()
                tmem_kv.commit()
                smem_kv.release()

            def convert_tile():
                """Convert one KV tile, one head-dimension stage at a time."""
                for head_stage in range(stages):
                    convert_stage(
                        lambda head_stage=head_stage: getattr(tmem_kv, label)(
                            head_dim_stage_idx=head_stage
                        )
                    )

            if not is_v:
                convert_tile()
            if flatten_loop:
                with domain_loop(0, domain * stages, 1, unroll=1):
                    convert_stage(getattr(tmem_kv, f"{label}_loop_stage"))
            else:
                with domain_loop(0, domain, 1, unroll=1):
                    convert_tile()
            if is_v:
                convert_tile()

        _decode_work_tile_schedule(cfg, None, body)

    return DecodeGenTask(
        src_resources=[smem_kv],
        dst_resources=[tmem_kv],
        cfg=cfg,
        warp_idx=v_warp_idx if is_v else cfg.transform_kv_warp_idx,
        num_warps=v_num_warps if is_v else cfg.transform_kv_num_warps,
        schedule=conversion_schedule(smem_kv, tmem_kv),
        num_registers=v_num_registers if is_v else cfg.transform_kv_task_num_registers,
        name="TransformVTask" if is_v else "TransformKTask",
        domain_scale=stages if flatten_loop else 1,
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
    """Create the MMA task that reads K and V from separate TMEM FIFOs.

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
