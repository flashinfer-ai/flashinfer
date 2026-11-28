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

"""K/V transform tasks: FP8 and NVFP4 K/V conversion before the MMA."""

import cutlass
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ..config import FmhaDecodeConfig
from ..device.constants import KV_KIND_K, KV_KIND_V
from .common import DecodeGenTask, TaskKwarg, decode_work_tile_schedule
from typing import Callable


def create_kv_operand_transform_task(
    smem_kv: MemoryResource,
    converted_kv: MemoryResource,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    is_v: bool,
    **kw: TaskKwarg,
) -> Task:
    """Create the converter for one operand of the dual K/V transform.

    Each call creates one task. ``is_v`` selects its operand and warp group.
    The destination is an FP8 SMEM ring for D256 Keeps or a TMEM FIFO for
    the dual-transform Swaps profile. K runs one tile ahead of V, matching
    the MMA task: HEAD converts the first K tile, each
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
    # Q64 softmax out of the instruction cache. Register-prefetched D256 also
    # keeps one conversion body per operand. Select the head slice at run time.
    stages = cfg.num_head_dim_stages_kv
    early_prefetch = cfg.uses_prefetched_smem_nvfp4_transform
    flatten_loop = stages == 4 or early_prefetch

    @schedule
    def conversion_schedule(smem_kv: MemoryResource, converted_kv: MemoryResource):
        """Convert one operand's raw ring into its converted FIFO for each work tile."""
        smem_kv.init_descriptor_state()

        def body():
            """Convert every tile of one work tile into the converted FIFO."""
            converted_kv.init_load_state()

            def convert_stage(work):
                """Convert one raw stage into one converted FIFO slot."""
                converted_kv.acquire()
                smem_kv.wait()
                work()
                converted_kv.commit()
                smem_kv.release()

            def convert_prefetched_stage(read_row):
                """Release the raw slot before waiting for the FP8 destination."""
                smem_kv.wait()
                packet = read_row()
                smem_kv.release()
                converted_kv.acquire()
                converted_kv.transform_prefetched_row(packet=packet)
                converted_kv.commit()

            def convert_tile():
                """Convert one KV tile, one head-dimension stage at a time."""
                for head_stage in range(stages):
                    if early_prefetch:
                        convert_prefetched_stage(
                            lambda head_stage=head_stage: smem_kv.prefetch_nvfp4_row(
                                head_dim_stage_idx=head_stage
                            )
                        )
                    else:
                        convert_stage(
                            lambda head_stage=head_stage: getattr(converted_kv, label)(
                                head_dim_stage_idx=head_stage
                            )
                        )

            if not is_v:
                convert_tile()
            if flatten_loop:
                with domain_loop(0, domain * stages, 1, unroll=1):
                    if early_prefetch:
                        convert_prefetched_stage(smem_kv.prefetch_nvfp4_loop_row)
                    else:
                        convert_stage(getattr(converted_kv, f"{label}_loop_stage"))
            else:
                with domain_loop(0, domain, 1, unroll=1):
                    convert_tile()
            if is_v:
                convert_tile()

        decode_work_tile_schedule(cfg, None, body)

    return DecodeGenTask(
        src_resources=[smem_kv],
        dst_resources=[converted_kv],
        cfg=cfg,
        warp_idx=v_warp_idx if is_v else cfg.transform_kv_warp_idx,
        num_warps=v_num_warps if is_v else cfg.transform_kv_num_warps,
        schedule=conversion_schedule(smem_kv, converted_kv),
        num_registers=v_num_registers if is_v else cfg.transform_kv_task_num_registers,
        name="TransformVTask" if is_v else "TransformKTask",
        domain_scale=stages if flatten_loop else 1,
        **kw,
    )


def create_nvfp4_keeps_transform_task(
    smem_kv_raw: MemoryResource,
    smem_k0: MemoryResource,
    smem_v0: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Feed an FP8 or BF16 KeepsMmaAb pipeline from a raw NVFP4 ring.

    The shared K/V load task fills ``smem_kv_raw`` in the order K(head),
    then K(i+1), V(i) per loop iteration, then V(tail); this task converts
    each raw stage, in the same order, into the matching K or V ring: one
    D256 stage for FP8, or one stage per D128 head-dim slice for BF16.
    """
    assert cfg.uses_keeps_transformed_kv and cfg.num_insts_kv == 1

    def transform_body(
        smem_kv_raw: MemoryResource,
        smem_k0: MemoryResource,
        smem_v0: MemoryResource,
    ) -> None:
        """Convert one work tile's raw stages in the load task's K/V order."""
        smem_k0.init_load_state()
        smem_v0.init_load_state()

        def transform_one(dst: MemoryResource, kv_kind: int) -> None:
            """Convert the next raw ``kv_kind`` stage into ``dst``, then return it.

            ConsWait on the raw ring, ProdAcquire/Work/Commit on the K/V ring
            for each head-dim stage, and only then ConsRelease: the loader may
            overwrite the raw stage once its rows have been read (into
            registers, with register capture, before the first acquire).
            """
            smem_kv_raw.wait()
            if cfg.uses_nvfp4_raw_register_capture:
                # A single raw stage: read it into registers and return it
                # first, so the loader refills it during the conversion.
                smem_kv_raw.capture_tile(kv_kind=kv_kind)
                smem_kv_raw.release()
            for head_dim_stage_idx in range(cfg.num_head_dim_stages_kv):
                dst.acquire()
                dst.transform_from_raw(head_dim_stage_idx=head_dim_stage_idx)
                dst.commit()
            if not cfg.uses_nvfp4_raw_register_capture:
                smem_kv_raw.release()

        transform_one(smem_k0, KV_KIND_K)
        with domain_loop(0, domain, 1, unroll=1):
            transform_one(smem_k0, KV_KIND_K)
            transform_one(smem_v0, KV_KIND_V)
        transform_one(smem_v0, KV_KIND_V)

    @schedule
    def transform_schedule(
        smem_kv_raw: MemoryResource,
        smem_k0: MemoryResource,
        smem_v0: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Bind the raw ring's consumer state, then run each work tile."""
        smem_kv_raw.init_descriptor_state()
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: transform_body(smem_kv_raw, smem_k0, smem_v0),
        )

    captured_schedule = (
        transform_schedule(smem_kv_raw, smem_k0, smem_v0)
        if work_queue is None
        else transform_schedule(smem_kv_raw, smem_k0, smem_v0, work_queue)
    )
    src = [smem_kv_raw]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_k0, smem_v0],
        cfg=cfg,
        warp_idx=cfg.transform_kv_warp_idx,
        num_warps=cfg.transform_kv_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.transform_kv_task_num_registers,
        name="TransformKvTask",
        **kw,
    )


def create_transform_kv_task(
    smem_kv: MemoryResource | None,
    smem_k0: MemoryResource | None,
    smem_k1: MemoryResource | None,
    smem_v0: MemoryResource | None,
    smem_v1: MemoryResource | None,
    smem_transformed_kv: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    inst_id: int | None = None,
    **kw: TaskKwarg,
) -> Task:
    """Create the K/V transform task.

    ``inst_id`` restricts the task to one KV instance's stream, converted
    into that instance's own ring by the transform_kv (instance 0) or
    transform_kv1 (instance 1) warp group.

    Consume one shared ``smem_kv``, or the four split `smem_k0/k1/v0/v1`
    and publish the same ordered K0/V0/K1/V1 stream into ``smem_transformed_kv``.
    """
    head_labels = (
        ("transform_k0",) if cfg.num_insts_kv == 1 else ("transform_k0", "transform_k1")
    )
    loop_labels = (
        ("transform_k0", "transform_v0")
        if cfg.num_insts_kv == 1
        else ("transform_k0", "transform_v0", "transform_k1", "transform_v1")
    )
    tail_labels = (
        ("transform_v0",) if cfg.num_insts_kv == 1 else ("transform_v0", "transform_v1")
    )
    if inst_id is not None:
        # Same HEAD/LOOP/TAIL order as the MMA task's use of this instance.
        head_labels = (f"transform_k{inst_id}",)
        loop_labels = (f"transform_k{inst_id}", f"transform_v{inst_id}")
        tail_labels = (f"transform_v{inst_id}",)

    split_sources = (smem_k0, smem_k1, smem_v0, smem_v1)
    use_split_sources = smem_kv is None
    if use_split_sources and any(resource is None for resource in split_sources):
        raise ValueError(
            "create_transform_kv_task requires either smem_kv, or smem_k0, smem_k1, smem_v0, and smem_v1"
        )
    # One shared raw ring with several head-dimension stages can run its
    # steady state once per raw stage, letting the TMEM converter pick K or V
    # and the stage at run time. Unrolling every (operand, stage) pair puts
    # 2 * num_head_dim_stages_kv converter bodies in the hot loop (about 16 KB
    # of code at D512). With the larger D512 Q32 softmax and correction code
    # that overflows the instruction cache (GB300 B128/KV128K/sQ4: 4.39 ->
    # 3.79 ms); narrower tiles fit and keep the unrolled bodies (D512 Q8/Q16
    # and D256 Q8/Q16 were 5-12% slower flattened).
    loop_stages_per_step = 2 * cfg.num_head_dim_stages_kv
    flatten_loop = (
        cfg.store_transformed_kv_in_tmem
        and cfg.num_insts_kv == 1
        and inst_id is None
        and not use_split_sources
        and cfg.num_head_dim_stages_kv == 4
        and cfg.tile_size_q >= 32
    )

    def transform_kv_schedule_body(
        smem_transformed_kv: MemoryResource,
        source_for_label: Callable[[str], MemoryResource],
    ) -> None:
        """Transform the HEAD/LOOP/TAIL stream from the selected raw rings."""
        smem_transformed_kv.init_load_state()

        def transform_one(label: str) -> None:
            source = source_for_label(label)
            if cfg.num_raw_kv_head_dim_stages == 1 < cfg.num_head_dim_stages_kv:
                # One raw stage holds every head-dim slice: convert all of
                # them before returning it to the loader.
                source.wait()
                slot_stages = cfg.transformed_kv_hd_stages_per_slot
                for slot_idx in range(cfg.num_head_dim_stages_kv // slot_stages):
                    smem_transformed_kv.acquire()
                    for sub_stage in range(slot_stages):
                        getattr(smem_transformed_kv, label)(
                            head_dim_stage_idx=slot_idx * slot_stages + sub_stage
                        )
                    smem_transformed_kv.commit()
                source.release()
                return
            # One ring slot covers slot_stages head-dim stages: acquire and
            # commit once per slot; raw stages are still waited and released
            # one head-dim stage at a time.
            slot_stages = cfg.transformed_kv_hd_stages_per_slot
            for slot_idx in range(cfg.num_head_dim_stages_kv // slot_stages):
                smem_transformed_kv.acquire()
                for sub_stage in range(slot_stages):
                    source.wait()
                    getattr(smem_transformed_kv, label)(
                        head_dim_stage_idx=slot_idx * slot_stages + sub_stage
                    )
                    source.release()
                smem_transformed_kv.commit()

        for label in head_labels:
            transform_one(label)

        # Keep the transformed-K/V steady state as one compact runtime loop.
        # Static decode domains can be large, and the compiler default may
        # otherwise fully unroll this transform body once per K/V tile.
        if flatten_loop:
            source = source_for_label(loop_labels[0])
            with domain_loop(0, domain * loop_stages_per_step, 1, unroll=1):
                smem_transformed_kv.acquire()
                source.wait()
                smem_transformed_kv.transform_kv_loop_stage()
                smem_transformed_kv.commit()
                source.release()
        else:
            with domain_loop(0, domain, 1, unroll=1):
                for label in loop_labels:
                    transform_one(label)

        for label in tail_labels:
            transform_one(label)

    @schedule
    def transform_shared_kv_schedule(
        smem_kv: MemoryResource,
        smem_transformed_kv: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        smem_kv.init_descriptor_state()
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: transform_kv_schedule_body(smem_transformed_kv, lambda _: smem_kv),
        )

    @schedule
    def transform_split_kv_schedule(
        smem_k0: MemoryResource,
        smem_k1: MemoryResource,
        smem_v0: MemoryResource,
        smem_v1: MemoryResource,
        smem_transformed_kv: MemoryResource,
        work_queue: WorkQueue | None = None,
    ) -> None:
        smem_k0.init_descriptor_state()
        smem_k1.init_descriptor_state()
        smem_v0.init_descriptor_state()
        smem_v1.init_descriptor_state()

        def source_for_label(label: str) -> MemoryResource:
            if label == "transform_k0":
                return smem_k0
            if label == "transform_k1":
                return smem_k1
            if label == "transform_v0":
                return smem_v0
            return smem_v1

        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: transform_kv_schedule_body(smem_transformed_kv, source_for_label),
        )

    if use_split_sources:
        split_schedule_args = [*split_sources, smem_transformed_kv]
        if work_queue is not None:
            split_schedule_args.append(work_queue)
        captured_schedule = transform_split_kv_schedule(*split_schedule_args)
        src = list(split_sources)
    else:
        assert smem_kv is not None
        captured_schedule = (
            transform_shared_kv_schedule(smem_kv, smem_transformed_kv)
            if work_queue is None
            else transform_shared_kv_schedule(smem_kv, smem_transformed_kv, work_queue)
        )
        src = [smem_kv]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_transformed_kv],
        cfg=cfg,
        warp_idx=cfg.transform_kv1_warp_idx
        if inst_id == 1
        else cfg.transform_kv_warp_idx,
        num_warps=(
            cfg.transform_kv1_num_warps if inst_id == 1 else cfg.transform_kv_num_warps
        ),
        schedule=captured_schedule,
        num_registers=(
            cfg.transform_kv1_task_num_registers
            if inst_id == 1
            else cfg.transform_kv_task_num_registers
        ),
        name="TransformKv1Task" if inst_id == 1 else "TransformKvTask",
        domain_scale=loop_stages_per_step if flatten_loop else 1,
        **kw,
    )
