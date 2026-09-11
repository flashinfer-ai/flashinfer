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

"""Resource definitions for the throughput 2CTA MLA decode TS kernel.

Resource classes matching the throughput 2CTA pipeline structure:

GMEM/register resources (not pipelined)
----------------------------------------
- PageOffsetWindowResource: one warp-wide, 32-page-ID register window

SMEM resources (pipelined)
--------------------------
- SmemQResource     : TmaUmmaAsync(1 stage),   LoadTma -> Mma
- SmemKVResource    : TmaUmmaAsync(7 stages), LoadTma -> Mma

SMEM/TMEM resources (pipelined)
-------------------------------
- SmemPResource     : UmmaConsumerAsync(2 stages), SoftmaxTask -> MmaTask
- TmemSResource     : UmmaProducerAsync(2 stages), MmaTask -> SoftmaxTask
- TmemCorrResource  : Async(2 stages),              SoftmaxTask -> CorrectionTask
- TmemOResource     : UmmaProducerAsync(1 stage),   MmaTask -> CorrectionTask

GMEM (no pipeline)
------------------
- GmemOResource     : No pipeline, Correction -> GMEM
"""

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
from cutlass.experimental import primitives as prims
from cutlass import Boolean, Float32, Int16, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T as mlir_T
from cutlass.pipeline.helpers import _get_thread_arrive_count

from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    WorkQueue,
)
from cutlass.experimental.task_scheduling.resources import consumer_work, producer_work
from cutlass.experimental.task_scheduling.enums import PipelineType, WorkAttr
from flashinfer.cute_dsl.attention.dsa.gather4_utils import (
    gather4_tma_descriptor_address,
    issue_gather4_tma_2cta,
)
from ...mask import kv_tile_needs_right_mask
from ...tensor_map import transform_ragged_coords

from .config import (
    V_SMEM_K_BLOCK_TOKENS,
    V_TMA_LATENT_ELEMENTS,
    MlaDecodeConfig,
)
from ..helpers.constants import (
    BF16_OUTPUT_VECTOR_ELEMENTS,
    EPILOGUE_COLUMN_GROUP_SHIFT,
    EPILOGUE_ROW_MASK,
    EPILOGUE_THREAD_TILE_MASK,
    EPILOGUE_THREAD_TILE_THREADS,
    FP8_OUTPUT_VECTOR_ELEMENTS,
    PACKED_FP8_OUTPUT_REGS,
    TCGEN05_32B_REGS_PER_LOAD,
    TCGEN05_32B_SHAPE,
    WARP_LANE_SHIFT,
)
from ..helpers.tile_scheduler import (
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
    divmod_constexpr_power_of_two_or_fdd,
)
from ..helpers.math import (
    NEG_FLT_MAX,
    ceil_div,
    mma_k_step_for_qkv,
    mma_kind_for_qkv,
    add_packed_f32x2,
    fadd2,
    ffma2,
    fma_packed_f32x2,
    mul_packed_f32x2,
    output_dtype,
    p_desc_layout,
    p_desc_leading_byte_offset,
    p_desc_stride_byte_offset,
    qk_desc_layout,
    qk_desc_layout_for_head_dim,
    qk_desc_leading_byte_offset,
    qk_desc_leading_byte_offset_for_head_dim,
    qk_desc_stride_byte_offset,
    qk_desc_stride_byte_offset_for_head_dim,
    qkv_dtype,
    qkv_major_k_stride_bytes_for,
)
from ..helpers.ops import (
    add_ftz_f32,
    affine2_contractible_f32,
    convert_f32_vector_to_bf16_satfinite,
    dsv4_clc_response_predicated,
    dsv4_cp_async_cg_l2_128b,
    dsv4_code_fence,
    dsv4_reset_cold_block,
    dsv4_sched_res_busy_xu64,
    dsv4_set_cold_block,
    dsv4_warp_switch,
    fp8_log2_quant_scale,
    fp8_quant_scale_rcp,
    fma_ftz_f32,
    fabs_f32,
    fmax_f32,
    fnma_ftz_f32,
    mul_ftz_f32,
    pack_float4_to_fp8_e4m3,
    rcp_approx_ftz_f32,
    sub_ftz_f32,
)
from ..helpers.mask import MaskType, mask_visible_k_length
from ..helpers.query import (
    flat_query_row_state,
    query_batch_bounds,
    runtime_flat_query_tile_has_rows,
)
from .work_partition import (
    runtime_split_kv_cap,
    runtime_split_tile_range,
)


@cutlass.dsl_user_op
def max_xorsign_abs_f32(lhs, rhs, *, loc=None, ip=None):
    """Return the larger FP32 magnitude; the result sign is intentionally junk."""

    return Float32(
        llvm.inline_asm(
            mlir_T.f32(),
            [
                Float32(lhs).ir_value(loc=loc, ip=ip),
                Float32(rhs).ir_value(loc=loc, ip=ip),
            ],
            "max.xorsign.abs.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cutlass.dsl_user_op
def max3_abs_ftz_f32(first, second, third, *, loc=None, ip=None):
    """Return the largest magnitude of three FP32 inputs on SM100."""

    return Float32(
        llvm.inline_asm(
            mlir_T.f32(),
            [
                Float32(first).ir_value(loc=loc, ip=ip),
                Float32(second).ir_value(loc=loc, ip=ip),
                Float32(third).ir_value(loc=loc, ip=ip),
            ],
            "max.abs.ftz.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _install_task_local_specs(resource: object, specs: tuple[tuple, ...]) -> None:
    """Install TaskLocalVariable fields declared by resource classes."""
    for spec in specs:
        field_name, dtype, default, docs = spec[:4]
        runtime_slot_name = spec[4] if len(spec) > 4 else None
        object.__setattr__(
            resource,
            field_name,
            TaskLocalVariable(
                dtype=dtype,
                default=default,
                docs=docs,
                runtime_slot_name=runtime_slot_name,
            ),
        )


def _make_deferred_mbarrier_array(
    barrier_storage,
    num_stages: int,
    agent,
    tx_count: int = 0,
    name: str = "",
):
    """Build ``MbarrierArray`` state without emitting constructor init ops.

    CUTLASS exposes the same no-init construction pattern through
    ``MbarrierArray.recast_to_new_op_type``.  DSV4 uses it directly here so
    all active arrays can be initialized together after TaskManager has
    assigned their final unified-SMEM pointers.
    """

    sync = object.__new__(pipeline.MbarrierArray)
    sync.barrier_storage = barrier_storage
    sync.tx_count = tx_count
    sync.num_stages = num_stages
    sync.op_type, sync.cg = agent
    sync.arrive_count = _get_thread_arrive_count(sync.cg)
    sync.mbarrier_layout = pipeline.MbarrierLayout.V0
    sync.name = name
    sync.mbarrier_base = barrier_storage
    return sync


def _dsv4_cta_layout(pipeline_config):
    layout = pipeline_config.cta_layout_vmnk
    if layout is None:
        return cute.make_layout((1, 1, 1, 1))
    if not isinstance(layout, cute.Layout):
        return cute.make_layout(layout)
    return layout


def _create_dsv4_deferred_pipeline(resource, pipeline_config):
    """Mirror the active CUTLASS pipeline constructors but defer all init."""

    barrier_ptr = pipeline_config.barrier_ptr
    if barrier_ptr is None:
        raise ValueError("source mbarrier init requires unified barrier storage")
    stages = pipeline_config.num_stages
    full_ptr = barrier_ptr.align(min_align=8)
    empty_ptr = full_ptr + stages
    pipeline_type = pipeline_config.pipeline_type

    if pipeline_type == PipelineType.AsyncAsync:
        full = _make_deferred_mbarrier_array(
            full_ptr,
            stages,
            (pipeline_config.async_producer_op, pipeline_config.producer_group),
            name=f"{resource.name}.full",
        )
        empty = _make_deferred_mbarrier_array(
            empty_ptr,
            stages,
            (pipeline.PipelineOp.AsyncThread, pipeline_config.consumer_group),
            name=f"{resource.name}.empty",
        )
        return pipeline.PipelineAsync(full, empty, stages, None, None)

    if pipeline_type == PipelineType.TmaUmma:
        cta_layout = _dsv4_cta_layout(pipeline_config)
        full = _make_deferred_mbarrier_array(
            full_ptr,
            stages,
            (pipeline.PipelineOp.TmaLoad, pipeline_config.producer_group),
            pipeline_config.num_bytes,
            f"{resource.name}.full",
        )
        empty = _make_deferred_mbarrier_array(
            empty_ptr,
            stages,
            (pipeline.PipelineOp.TCGen05Mma, pipeline_config.consumer_group),
            name=f"{resource.name}.empty",
        )
        if cute.size(cta_layout) == 1:
            producer_mask = None
            is_leader_cta = True
        else:
            producer_mask = pipeline.PipelineTmaUmma._compute_mcast_arrival_mask(
                cta_layout, pipeline_config.mcast_mode_mn
            )
            is_leader_cta = pipeline.PipelineTmaUmma._compute_is_leader_cta(cta_layout)
        consumer_mask = producer_mask
        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cute.size(cta_layout, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )
        result = pipeline.PipelineTmaUmma(
            full,
            empty,
            stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
        )
        return resource._apply_task_warp_leader_to_tma_umma(result)

    if pipeline_type == PipelineType.UmmaAsync:
        cta_layout = _dsv4_cta_layout(pipeline_config)
        full = _make_deferred_mbarrier_array(
            full_ptr,
            stages,
            (pipeline.PipelineOp.TCGen05Mma, pipeline_config.producer_group),
            name=f"{resource.name}.full",
        )
        empty = _make_deferred_mbarrier_array(
            empty_ptr,
            stages,
            (pipeline.PipelineOp.AsyncThread, pipeline_config.consumer_group),
            name=f"{resource.name}.empty",
        )
        producer_mask = (
            None
            if cute.size(cta_layout) == 1
            else pipeline.PipelineUmmaAsync._compute_tmem_sync_mask(cta_layout)
        )
        consumer_mask = (
            None
            if cute.size(cta_layout, mode=[0]) == 1
            else pipeline.PipelineUmmaAsync._compute_peer_cta_rank()
        )
        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cute.size(cta_layout, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )
        return pipeline.PipelineUmmaAsync(
            full,
            empty,
            stages,
            producer_mask,
            consumer_mask,
            cta_group,
        )

    if pipeline_type == PipelineType.ClcFetchAsync:
        cta_layout = _dsv4_cta_layout(pipeline_config)
        full = _make_deferred_mbarrier_array(
            full_ptr,
            stages,
            (pipeline.PipelineOp.ClcLoad, pipeline_config.producer_group),
            pipeline_config.num_bytes,
            f"{resource.name}.full",
        )
        empty = _make_deferred_mbarrier_array(
            empty_ptr,
            stages,
            (pipeline.PipelineOp.AsyncThread, pipeline_config.consumer_group),
            name=f"{resource.name}.empty",
        )
        tidx, _, _ = cute.arch.thread_idx()
        producer_mask, is_signaling_thread = (
            pipeline.PipelineClcFetchAsync._init_full_barrier_arrive_signal(
                cta_layout, tidx
            )
        )
        return pipeline.PipelineClcFetchAsync(
            full,
            empty,
            stages,
            producer_mask,
            0,
            is_signaling_thread,
        )

    raise ValueError(
        f"DSV4 source mbarrier init does not support pipeline type {pipeline_type}"
    )


def _dsv4_arrival_runs(arrival_counts: tuple[int, ...]):
    """Compress a static per-slot count table for compact scalar selects."""

    runs = []
    start = 0
    current = arrival_counts[0]
    for index, count in enumerate(arrival_counts[1:], 1):
        if count != current:
            runs.append((start, index, current))
            start = index
            current = count
    runs.append((start, len(arrival_counts), current))
    return tuple(runs)


def _dsv4_select_arrival_count(slot, arrival_counts: tuple[int, ...]):
    result = Int32(arrival_counts[0])
    for start, end, count in _dsv4_arrival_runs(arrival_counts)[1:]:
        in_run = slot >= Int32(start)
        if end != len(arrival_counts):
            in_run = in_run & (slot < Int32(end))
        result = Int32(cutlass.select_(in_run, Int32(count), result))
    return result


def prepare_dsv4_source_mbarriers(task_manager):
    """Allocate and bind one contiguous barrier block before resource create.

    This throughput kernel supplies its payload arrays from the kernel
    prologue, so TaskManager intentionally has no unified ``SmemAllocator``.
    Explicitly binding only the barrier storage makes contiguity a real DSV4
    contract instead of relying on incidental static-allocation placement.
    """

    barrier_resources = tuple(
        resource
        for resource in task_manager.resources
        if resource.pipeline_config is not None and resource.pipeline_group is None
    )
    total_slots = sum(
        2 * resource.pipeline_config.num_stages for resource in barrier_resources
    )
    if total_slots <= 0 or total_slots > 64:
        raise ValueError(
            f"source mbarrier init requires 1..64 active slots, got {total_slots}"
        )
    barrier_storage = cutlass.Array(
        Int64,
        total_slots,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    base_ptr = cute.make_ptr(
        Int64,
        barrier_storage.data_ptr(),
        mem_space=cutlass.AddressSpace.smem,
    )
    offset = 0
    for resource in barrier_resources:
        pipeline_config = resource.pipeline_config
        resource.pipeline_config = replace(
            pipeline_config,
            barrier_ptr=base_ptr + offset,
        )
        offset += 2 * pipeline_config.num_stages
    task_manager._dsv4_source_barrier_resources = barrier_resources
    task_manager._dsv4_source_barrier_storage = barrier_storage
    return barrier_storage


@cute.jit
def _initialize_dsv4_source_mbarrier_block(
    base_ptr,
    arrival_counts: cutlass.Constexpr,
) -> None:
    """Emit the single staged W0 initialization region."""

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx & Int32(31)
        first_slots = min(len(arrival_counts), 32)
        if first_slots == 32:
            cute.arch.mbarrier_init(
                base_ptr + lane,
                _dsv4_select_arrival_count(lane, arrival_counts),
            )
        else:
            if lane < Int32(first_slots):
                cute.arch.mbarrier_init(
                    base_ptr + lane,
                    _dsv4_select_arrival_count(lane, arrival_counts),
                )
        if len(arrival_counts) > 32:
            second_slot = lane + Int32(32)
            if lane < Int32(len(arrival_counts) - 32):
                cute.arch.mbarrier_init(
                    base_ptr + second_slot,
                    _dsv4_select_arrival_count(second_slot, arrival_counts),
                )


def initialize_dsv4_source_mbarriers(task_manager) -> None:
    """Initialize the explicitly contiguous barrier block from W0 lanes."""

    barrier_resources = task_manager._dsv4_source_barrier_resources
    if not barrier_resources:
        return

    arrival_counts = []
    for resource in barrier_resources:
        pipeline_object = resource.pipeline
        stages = resource.pipeline_config.num_stages
        full_count = pipeline_object.sync_object_full.arrive_count
        empty_count = pipeline_object.sync_object_empty.arrive_count
        if not isinstance(full_count, int) or not isinstance(empty_count, int):
            raise ValueError("source mbarrier init requires static arrival counts")
        arrival_counts.extend([full_count] * stages)
        arrival_counts.extend([empty_count] * stages)
    arrival_counts = tuple(arrival_counts)
    if len(arrival_counts) > 64:
        raise ValueError("source mbarrier init supports at most 64 active slots")

    base_ptr = barrier_resources[0].pipeline_config.barrier_ptr.align(min_align=8)
    _initialize_dsv4_source_mbarrier_block(base_ptr, arrival_counts)


@dataclass(kw_only=True)
class HighThroughputMlaResource(MemoryResource):
    """Base class that binds captured-schedule task-local variables."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = ()

    def __post_init__(self) -> None:
        _install_task_local_specs(self, self._task_local_specs)

    def create_pipeline(self, pipeline_config):
        cfg = getattr(self, "cfg", None)
        if cfg is not None and cfg.dsv4_use_source_mbarrier_init:
            return _create_dsv4_deferred_pipeline(self, pipeline_config)
        return super().create_pipeline(pipeline_config)


@cute.jit
def routing_row_index(cfg, tile_idx, logical_seq_len_q, cu_seqlens_q=None):
    """Return the metadata row owned by one logical MLA work tile.

    Dense MLA metadata is per batch.  DSV4 CSA instead supplies one routing
    row per query token, flattened in ``(batch, query)`` order.
    """

    if cutlass.const_expr(cfg.is_dynamic_token_sparse):
        batch_idx = Int32(tile_idx[2])
        query_idx = Int32(tile_idx[1])
        if cutlass.const_expr(cu_seqlens_q is not None):
            query_start = Int32(cu_seqlens_q[batch_idx])
            query_len = Int32(cu_seqlens_q[batch_idx + Int32(1)]) - query_start
            # The grid is sized for the largest request.  Clamp inactive
            # padded tiles before indexing packed per-query metadata; the
            # work queue subsequently gives those tiles an empty K domain.
            safe_query_idx = cute.math.min(
                query_idx, cute.math.max(query_len - Int32(1), Int32(0))
            )
            return query_start + safe_query_idx
        return batch_idx * Int32(logical_seq_len_q) + query_idx
    return Int32(tile_idx[2])


# =====================================================================
# Dsv4PageOffsetRingResource -- W9 page-index cp.async ring
# =====================================================================


@dataclass(kw_only=True)
class Dsv4PageOffsetRingResource(HighThroughputMlaResource):
    """Stage DSV4 sparse token indices in the source kernel's W9 ring.

    One producer stage is ``int32[256]``: the first and second halves are
    the two adjacent K128 sparse-token tiles.  W9 copies four indices per
    lane into both halves using 16-byte ``cp.async`` operations.  The
    ``AsyncLoad`` pipeline configuration turns ``commit`` into a
    ``cp.async.mbarrier.arrive`` for every W9 lane, so a W12--W15 consumer
    can wait on data completion rather than merely instruction issue.
    """

    page_offsets: Any = None
    cu_seqlens_q: Any = None
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    smem_page_offsets: Any = None
    _smem_page_offsets: Any = field(init=False, default=None)
    page_offset_stage: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "page_offset_stage",
            Int32,
            Int32(0),
            "Physical W9 ring stage selected by the current consumer wait.",
        ),
    )

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Bind the kernel-prologue page-index SMEM allocation.

        This throughput kernel predates TaskManager's unified
        ``SmemAllocator``: Q/K/V/P are all static ``cutlass.Array`` objects
        allocated in the kernel prologue and passed to their resources.  The
        DSV4 ring must use that same allocation model; ``stage_info.context``
        deliberately has no unified ``smem_base`` in this kernel.
        """
        del stage_info
        assert self.smem_page_offsets is not None
        self._smem_page_offsets = self.smem_page_offsets

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize W9's view before issuing the page-index copies."""
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo) -> None:
        """Initialize W12--W15's view of the same page-index ring."""
        self._init_smem_state(stage_info)

    @producer_work
    @cute.jit
    def prefetch_pair(self, stage_info: StageInfo) -> None:
        """Copy two adjacent K128 index sets into one W9 pipeline stage.

        The index calculation is the source ``SmemPageOffsetsKv`` mapping:
        ``(k_tile_base + loop_offset) * 128 + lane * 4`` and the same
        address plus 128.  Source addresses are clamped to the last aligned
        four-index vector, preserving the generated kernel's tail behaviour.
        """
        cfg = self.cfg
        work_tile = stage_info.work_tile
        route_row = routing_row_index(
            cfg, work_tile.tile_idx, self.logical_seq_len_q, self.cu_seqlens_q
        )
        page_offsets_row = self.page_offsets[None, route_row]
        page_offsets_flat = cute.flat_divide(page_offsets_row, (1,))
        lane_idx = cute.arch.thread_idx()[0] & Int32(31)
        k_tile = work_tile.k_index_base + Int32(stage_info.loop_offset)
        pair_base = k_tile * Int32(128)
        # The producer is never entered for an empty domain.  Retain a
        # non-negative clamped vector address for the final odd K128 tile.
        last_vector_base = (
            cute.math.max(work_tile.k_len - Int32(1), Int32(0)) >> Int32(2)
        ) << Int32(2)
        src_idx_0 = cute.math.min(pair_base + lane_idx * Int32(4), last_vector_base)
        src_idx_1 = cute.math.min(
            pair_base + Int32(128) + lane_idx * Int32(4), last_vector_base
        )
        stage_base = Int32(stage_info.stage_idx) * Int32(
            cfg.dsv4_page_offsets_entries_per_stage
        )
        # ``cp.async`` takes a 32-bit shared address.  Do not pass the DSL's
        # generic pointer representation directly: its address-space cast is
        # not implicit on SM100 and produces a hardware exception at issue.
        dst_0 = cutlass.inttoptr(
            self._smem_page_offsets.data_ptr(stage_base + lane_idx * Int32(4)).toint(
                Int32
            ),
            3,
            Int32,
        )
        dst_1 = cutlass.inttoptr(
            self._smem_page_offsets.data_ptr(
                stage_base + Int32(128) + lane_idx * Int32(4)
            ).toint(Int32),
            3,
            Int32,
        )
        src_0 = page_offsets_flat[None, src_idx_0].iterator.llvm_ptr
        src_1 = page_offsets_flat[None, src_idx_1].iterator.llvm_ptr
        if cutlass.const_expr(cfg.dsv4_use_source_page_offsets_cache_policy):
            dsv4_cp_async_cg_l2_128b(dst_0, src_0)
            dsv4_cp_async_cg_l2_128b(dst_1, src_1)
        else:
            prims.cp_async_shared_global(dst_0, src_0, 16, "ca")
            prims.cp_async_shared_global(dst_1, src_1, 16, "ca")
        if cutlass.const_expr(cfg.dsv4_use_source_page_offsets_transcnt):
            # Match trtllm::dev::CutlassCpAsyncPipeline exactly: this
            # counted async arrival has a net-zero pending-count effect, and
            # TaskManager's following AsyncThread producer_commit supplies
            # the source's ordinary per-lane barrier arrival.
            assert self.pipeline is not None
            prims.cp_async_mbarrier_arrive(
                self.pipeline.sync_object_full.get_barrier(stage_info.stage_idx),
                noinc=False,
            )

    @consumer_work(returns=page_offset_stage)
    @cute.jit
    def read_k_stage(self, stage_info: StageInfo) -> Int32:
        """Publish the W9 ring stage consumed by the K Gather4 edge.

        K and V intentionally consume different, identically populated W9
        stages for each adjacent K128 pair.  Keep two labelled consumer
        operations so the source-pair task can recover and validate both
        dependency segments from the captured schedule; the returned physical
        stage still comes from the same consumer pipeline state.
        """
        del stage_info
        return Int32(self.state_src.consumer_work_stage)

    @consumer_work(returns=page_offset_stage)
    @cute.jit
    def read_v_stage(self, stage_info: StageInfo) -> Int32:
        """Publish the W9 ring stage consumed by the V Gather4 edge."""
        del stage_info
        return Int32(self.state_src.consumer_work_stage)

    @cute.jit
    def page_quad(self, offset: Int32) -> cutlass.Array:
        """Read four page indices from the currently waited ring stage."""
        stage_base = self.state_src.consumer_work_stage * Int32(
            self.cfg.dsv4_page_offsets_entries_per_stage
        )
        return self._smem_page_offsets.load(
            stage_base + Int32(offset), vector_size=4, alignment=16
        )


# =====================================================================
# WorkThrottleBarrierResource — Cluster-safe CLC scheduler pacing
# =====================================================================


@dataclass(kw_only=True)
class WorkThrottleBarrierResource(MemoryResource):
    """Pace CLC schedule token reuse against the leader CTA's active MMA task.

    The barrier has no payload.  Source advances this software-pipeline state
    only on CTA rank 0.  Some TS users (notably DSV4 W12--W15) execute the
    surrounding producer task on both CTAs and gate only the signal, so mark
    the state itself leader-owned as well.  Otherwise CTA rank 1 advances an
    uncommitted local cursor and deadlocks in ``producer_tail`` at kernel exit.
    """

    is_barrier: cutlass.Constexpr[bool] = True
    producer_state_owned_by_signaling_ctas_only: cutlass.Constexpr[bool] = True
    cfg: cutlass.Constexpr = None

    def create_pipeline(self, pipeline_config):
        if self.cfg is not None and self.cfg.dsv4_use_source_mbarrier_init:
            return _create_dsv4_deferred_pipeline(self, pipeline_config)
        return super().create_pipeline(pipeline_config)


# =====================================================================
# MlaWorkQueue — Persistent tile scheduler for MLA decode
# =====================================================================


class MlaTsWorkTileInfo:
    """MLA work tile with scalar fields for TS persistent loop carry.

    Besides the scheduler coordinate, cache the per-tile K-domain values that
    hot resource paths need. Each persistent warp loop computes the K domain
    once and passes the derived indices to the page-offset, TMA, MMA, and
    softmax bodies.
    """

    @cute.jit
    def __init__(
        self,
        tile_idx,
        is_valid,
        k_len=0,
        k_tile_count=0,
        k_index_base=0,
    ):
        """Initialize the staged tile coordinate and cached K-domain metadata."""
        cluster_idx, seq_q_idx, batch_idx, split_kv_idx = tile_idx
        self._cluster_idx = Int32(cluster_idx)
        self._seq_q_idx = Int32(seq_q_idx)
        self._batch_idx = Int32(batch_idx)
        self._split_kv_idx = Int32(split_kv_idx)
        self._is_valid = Boolean(is_valid)
        self._k_len = Int32(k_len)
        self._k_tile_count = Int32(k_tile_count)
        self._k_index_base = Int32(k_index_base)

    @property
    @cute.jit
    def tile_idx(self):
        """Return the scheduler tile coordinate tuple."""
        return (
            self._cluster_idx,
            self._seq_q_idx,
            self._batch_idx,
            self._split_kv_idx,
        )

    @property
    @cute.jit
    def is_valid_tile(self):
        """Return whether this tile participates in the current launch."""
        return self._is_valid

    @property
    @cute.jit
    def k_len(self):
        """Return the request's graph-live K length for this tile."""
        return self._k_len

    @property
    @cute.jit
    def k_tile_count(self):
        """Return the number of K tiles assigned to this work tile."""
        return self._k_tile_count

    @property
    @cute.jit
    def k_index_base(self):
        """Return the first K tile index owned by this work tile."""
        return self._k_index_base

    @cute.jit
    def update_from(self, other) -> None:
        """Replace this tile info with another tile info object."""
        cluster_idx, seq_q_idx, batch_idx, split_kv_idx = other.tile_idx
        self._cluster_idx = cluster_idx
        self._seq_q_idx = seq_q_idx
        self._batch_idx = batch_idx
        self._split_kv_idx = split_kv_idx
        self._is_valid = Boolean(other.is_valid_tile)
        self._k_len = Int32(other.k_len)
        self._k_tile_count = Int32(other.k_tile_count)
        self._k_index_base = Int32(other.k_index_base)

    def __extract_mlir_values__(self):
        """Extract scalar MLIR values for value-type lowering."""
        values = cutlass.extract_mlir_values(self._cluster_idx)
        values += cutlass.extract_mlir_values(self._seq_q_idx)
        values += cutlass.extract_mlir_values(self._batch_idx)
        values += cutlass.extract_mlir_values(self._split_kv_idx)
        values += cutlass.extract_mlir_values(self._is_valid)
        values += cutlass.extract_mlir_values(self._k_len)
        values += cutlass.extract_mlir_values(self._k_tile_count)
        values += cutlass.extract_mlir_values(self._k_index_base)
        return values

    def __new_from_mlir_values__(self, values):
        """Rebuild a tile info object from lowered scalar MLIR values."""
        return MlaTsWorkTileInfo(
            (
                cutlass.new_from_mlir_values(self._cluster_idx, [values[0]]),
                cutlass.new_from_mlir_values(self._seq_q_idx, [values[1]]),
                cutlass.new_from_mlir_values(self._batch_idx, [values[2]]),
                cutlass.new_from_mlir_values(self._split_kv_idx, [values[3]]),
            ),
            cutlass.new_from_mlir_values(self._is_valid, [values[4]]),
            cutlass.new_from_mlir_values(self._k_len, [values[5]]),
            cutlass.new_from_mlir_values(self._k_tile_count, [values[6]]),
            cutlass.new_from_mlir_values(self._k_index_base, [values[7]]),
        )


@dataclass(kw_only=True)
class MlaWorkQueue(WorkQueue):
    """WorkQueue that preserves MLA coordinates for static or CLC scheduling.

    Static scheduling uses ``MLAStaticTileScheduler`` directly.  BF16 CLC
    scheduling uses the public CUTLASS ``WorkQueue`` implementation, then maps
    its cluster response ``(cluster_rank, 0, linear_cluster)`` back to MLA's
    ``(cluster_rank, s_idx, b_idx, split_kv_idx)`` coordinate.  Both modes feed
    the same K-domain cache and task-local value type.

    ``tile_sched_params`` is the ``MLAStaticTileSchedulerParams`` object created
    by the kernel.
    """

    tile_sched_params: Any = None
    cache_seqs: Any = None  # raw per-request KV sequence lengths
    sparse_topk_lens: Any = None  # per-packed-Q sparse scan widths (DSV4)
    split_kv: Any = None  # maximum split slots in the launch/workspace
    block_split_kvs: Any = None  # optional per-batch split caps
    is_var_split_kv: cutlass.Constexpr[bool] = False
    cfg: Any = None  # MlaDecodeConfig for tile sizes
    static_split_kv: cutlass.Constexpr = None
    static_seq_len_k: cutlass.Constexpr = None
    cu_seqlens_q: Any = None
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    static_problem_shape_b: cutlass.Constexpr[int] = None
    static_problem_shape_s: cutlass.Constexpr[int] = None
    use_clc_dynamic: cutlass.Constexpr[bool] = False
    work_tile: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    skip_work_tile: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def __init__(
        self,
        tile_sched_params,
        cache_seqs=None,
        sparse_topk_lens=None,
        split_kv=None,
        block_split_kvs=None,
        is_var_split_kv=False,
        cfg=None,
        static_split_kv=None,
        static_seq_len_k=None,
        cu_seqlens_q=None,
        logical_num_heads_q=128,
        logical_seq_len_q=1,
        static_problem_shape_b=None,
        static_problem_shape_s=None,
        use_clc_dynamic=False,
        tile_scheduler_config=None,
        **kwargs,
    ):
        self.use_clc_dynamic = use_clc_dynamic
        if use_clc_dynamic:
            WorkQueue.__init__(
                self,
                tile_scheduler_config=tile_scheduler_config,
                **kwargs,
            )
        else:
            # The custom static scheduler does not use TileSchedulerConfig.
            MemoryResource.__init__(self, **kwargs)
            self.tile_scheduler_config = None
        self.tile_sched_params = tile_sched_params
        self.cache_seqs = cache_seqs
        self.sparse_topk_lens = sparse_topk_lens
        self.split_kv = split_kv
        self.block_split_kvs = block_split_kvs
        self.is_var_split_kv = is_var_split_kv
        self.cfg = cfg
        self.static_split_kv = static_split_kv
        self.static_seq_len_k = static_seq_len_k
        self.cu_seqlens_q = cu_seqlens_q
        self.logical_num_heads_q = logical_num_heads_q
        self.logical_seq_len_q = logical_seq_len_q
        self.static_problem_shape_b = static_problem_shape_b
        self.static_problem_shape_s = static_problem_shape_s
        self.work_tile = TaskLocalVariable(
            dtype=MlaTsWorkTileInfo,
            default=MlaTsWorkTileInfo(
                (Int32(0), Int32(0), Int32(0), Int32(0)),
                Boolean(False),
                Int32(0),
                Int32(0),
                Int32(0),
            ),
            docs="Current MLA persistent-scheduler work tile.",
        )
        self.skip_work_tile = TaskLocalVariable(
            dtype=Boolean,
            default=Boolean(False),
            docs="Whether the current work tile should skip skippable work.",
        )

    def create_tile_scheduler(self):
        if cutlass.const_expr(self.use_clc_dynamic):
            return WorkQueue.create_tile_scheduler(self)
        return create_mla_static_tile_scheduler(
            self.tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )

    def create_pipeline(self, pipeline_config):
        if self.cfg is not None and self.cfg.dsv4_use_source_mbarrier_init:
            return _create_dsv4_deferred_pipeline(self, pipeline_config)
        return super().create_pipeline(pipeline_config)

    def _create_placeholder_tile_scheduler(self):
        """Create a dead structural scheduler for the shared prologue.

        Real persistent scheduling is created per task in MlaTask.  Keeping
        this placeholder independent of grid_dim avoids shared-prologue nctaid
        values being CSE'd into every warp-dispatch branch.
        """
        blk = cute.arch.block_idx()
        dummy_params = MLAStaticTileSchedulerParams(
            self.tile_sched_params.is_persistent,
            blk[0],
            blk[0],
            self.tile_sched_params.cluster_shape_mnk,
            blk[0],
        )
        return MLAStaticTileScheduler(dummy_params, blk[0], blk, blk)

    def create(self) -> None:
        """Create pipeline (if any) and the scheduler used by all TS tasks."""
        if cutlass.const_expr(self.use_clc_dynamic):
            WorkQueue.create(self)
            return
        # Call MemoryResource.create() (not WorkQueue.create) to avoid
        # the tile_scheduler_config check in the parent.
        MemoryResource.create(self)
        self.tile_scheduler = self._create_placeholder_tile_scheduler()

    @cute.jit
    def initial_work_tile_info(self):
        work_tile = self.tile_scheduler.initial_work_tile_info()
        if cutlass.const_expr(self.use_clc_dynamic):
            return self._work_tile_from_clc_tile(work_tile)
        return self.wrap_work_tile(work_tile)

    @cute.jit
    def wrap_work_tile(self, work_tile):
        return MlaTsWorkTileInfo(work_tile.tile_idx, work_tile.is_valid_tile)

    @cute.jit
    def _work_tile_from_clc_tile(self, work_tile):
        """Map one cluster-wide CLC response to MLA's four coordinates."""
        query_cluster_idx, _, split_batch_idx = work_tile.tile_idx
        params = self.tile_sched_params
        cluster_width = Int32(params.cluster_shape_mnk[0])
        s_idx = query_cluster_idx // cluster_width
        cluster_idx = query_cluster_idx % cluster_width
        split_kv_idx, b_idx = divmod_constexpr_power_of_two_or_fdd(
            split_batch_idx,
            self.static_problem_shape_b,
            params.problem_shape_b_fdd,
        )
        return self._make_work_tile_info(
            (cluster_idx, s_idx, b_idx, split_kv_idx),
            work_tile.is_valid_tile,
        )

    @cute.jit
    def _dsv4_work_tile_from_clc_response(self, result_addr):
        """Decode a CLC response and read DSV4 metadata only when valid."""

        m_idx, n_idx, l_idx, valid_i32 = dsv4_clc_response_predicated(result_addr)
        if cutlass.const_expr(self.tile_scheduler.insert_fence):
            cute.arch.fence_proxy("async.shared", space="cta")
        is_valid = Boolean(valid_i32 == Int32(1))

        def make_valid_tile():
            m_raster, _, l_raster = self.tile_scheduler._swizzle_and_rasterize(
                m_idx, n_idx, l_idx
            )
            cta_m, _, _ = self.tile_scheduler.cta_id_in_cluster
            query_cluster_idx = m_raster + cta_m
            split_batch_idx = l_raster

            params = self.tile_sched_params
            cluster_width = Int32(params.cluster_shape_mnk[0])
            s_idx = query_cluster_idx // cluster_width
            cluster_idx = query_cluster_idx % cluster_width
            split_kv_idx, b_idx = divmod_constexpr_power_of_two_or_fdd(
                split_batch_idx,
                self.static_problem_shape_b,
                params.problem_shape_b_fdd,
            )
            tile = self._make_work_tile_info(
                (cluster_idx, s_idx, b_idx, split_kv_idx),
                Boolean(True),
            )
            return (
                tile.tile_idx[0],
                tile.tile_idx[1],
                tile.tile_idx[2],
                tile.tile_idx[3],
                tile.k_len,
                tile.k_tile_count,
                tile.k_index_base,
            )

        def make_terminal_tile():
            zero = Int32(0)
            return (zero, zero, zero, zero, zero, zero, zero)

        tile_values = cutlass.if_generate(
            is_valid,
            make_valid_tile,
            make_terminal_tile,
            return_types=[Int32] * 7,
        )
        return MlaTsWorkTileInfo(
            tuple(tile_values[:4]),
            is_valid,
            tile_values[4],
            tile_values[5],
            tile_values[6],
        )

    @cute.jit
    def _make_work_tile_info(
        self,
        tile_idx,
        is_valid,
    ):
        cfg = self.cfg
        params = self.tile_sched_params
        _, s_idx, b_idx, split_kv_idx = tile_idx
        safe_b_idx = cute.math.min(b_idx, params.problem_shape_b - Int32(1))
        if cutlass.const_expr(self.static_seq_len_k is not None):
            K = Int32(self.static_seq_len_k)
        else:
            metadata_row = routing_row_index(
                cfg,
                (Int32(0), s_idx, safe_b_idx, split_kv_idx),
                self.logical_seq_len_q,
                self.cu_seqlens_q,
            )
            K = Int32(
                self.sparse_topk_lens[metadata_row]
                if cutlass.const_expr(cfg.is_dynamic_token_sparse)
                else self.cache_seqs[metadata_row]
            )
        sequence_k_len = K

        # The launch uses the largest flat-Q tile count in the batch. Shorter
        # variable-Q requests can therefore receive a whole tile containing
        # no real query rows. Give that tile an empty K domain so every task
        # skips it before any Q/K/V or partial-output access.
        _, q_len = query_batch_bounds(
            self.cu_seqlens_q,
            safe_b_idx,
            self.logical_seq_len_q,
        )
        query_tile_has_rows = runtime_flat_query_tile_has_rows(
            s_idx,
            cfg.mma_qk_tiler[0],
            self.logical_num_heads_q,
            self.logical_seq_len_q,
            self.cu_seqlens_q,
            safe_b_idx,
        )
        if cutlass.const_expr(
            cfg.mask_type == MaskType.CAUSAL.value
            and self.logical_seq_len_q > 1
            and not cfg.is_dynamic_token_sparse
        ):
            # Split partitioning owns the mask-visible CTA domain. The final
            # physical row resolves to the tile's latest valid Q
            # token. Row-causal softmax narrows earlier rows without changing
            # producer split geometry.
            _, _, logical_q_idx, _, _ = flat_query_row_state(
                Int32(cfg.mma_qk_tiler[0] - 1),
                s_idx,
                cfg.mma_qk_tiler[0],
                self.logical_num_heads_q,
                self.logical_seq_len_q,
                self.cu_seqlens_q,
                safe_b_idx,
            )
            K = mask_visible_k_length(cfg.mask_type, K, logical_q_idx, q_len)
        K = K if query_tile_has_rows else Int32(0)
        k_tile_total = (K + Int32(cfg.mma_qk_tiler[1] - 1)) // Int32(
            cfg.mma_qk_tiler[1]
        )
        if cutlass.const_expr(self.static_split_kv == 1):
            # Single-split fixed profiles process the whole K domain in one CTA.
            # Avoid the runtime split-KV ceil-div/min/max sequence in every
            # persistent task branch.
            k_index_base = Int32(0)
            k_tile_count = k_tile_total
        else:
            # Grid/workspace geometry stays at ``split_kv`` while each batch's
            # optional cap and valid K select the configured-span nonempty prefix.
            if cutlass.const_expr(
                self.static_split_kv is not None and not self.is_var_split_kv
            ):
                split_kv_cap = Int32(self.static_split_kv)
            else:
                split_kv_cap = runtime_split_kv_cap(
                    self.split_kv,
                    self.is_var_split_kv,
                    self.block_split_kvs,
                    safe_b_idx,
                )
            k_index_base, k_tile_count = runtime_split_tile_range(
                k_tile_total,
                split_kv_cap,
                split_kv_idx,
            )
        return MlaTsWorkTileInfo(
            tile_idx,
            is_valid,
            sequence_k_len,
            k_tile_count,
            k_index_base,
        )

    @cute.jit
    def k_tile_count_for_tile(self, tile_idx):
        """Return the dynamic K-loop bound required by stock Task."""

        return self._make_work_tile_info(tile_idx, Boolean(True)).k_tile_count

    @cute.jit
    def skip_work_tile_if(self, work_tile):
        """Skip zero-K CLC tiles while retaining WorkQueue bookkeeping."""

        return work_tile.k_tile_count <= Int32(0)

    @cute.jit
    def _work_tile_from_linear_idx(self, current_work_linear_idx):
        params = self.tile_sched_params
        current_work_cluster_batch, cluster_idx = (
            current_work_linear_idx // params.cluster_shape_mnk[0],
            current_work_linear_idx % params.cluster_shape_mnk[0],
        )
        current_work_s_batch, s_idx = divmod_constexpr_power_of_two_or_fdd(
            current_work_cluster_batch,
            self.static_problem_shape_s,
            params.problem_shape_s_fdd,
        )
        current_work_b_batch, b_idx = divmod_constexpr_power_of_two_or_fdd(
            current_work_s_batch,
            self.static_problem_shape_b,
            params.problem_shape_b_fdd,
        )
        if cutlass.const_expr(self.static_split_kv == 1):
            split_kv_idx = Int32(0)
            num_blocks = (
                params.cluster_shape_mnk[0]
                * params.problem_shape_s
                * params.problem_shape_b
            )
        else:
            _, split_kv_idx = divmod(
                current_work_b_batch,
                params.split_kv_fdd,
            )
            num_blocks = (
                params.cluster_shape_mnk[0]
                * params.problem_shape_s
                * params.problem_shape_b
                * params.split_kv
            )
        return self._make_work_tile_info(
            (cluster_idx, s_idx, b_idx, split_kv_idx),
            current_work_linear_idx < num_blocks,
        )

    @cute.jit
    def _work_tile_from_block_idx(self, block_idx):
        params = self.tile_sched_params
        s_idx, b_idx = divmod_constexpr_power_of_two_or_fdd(
            block_idx[1],
            self.static_problem_shape_b,
            params.problem_shape_b_fdd,
        )
        return self._make_work_tile_info(
            (block_idx[0], s_idx, b_idx, block_idx[2]),
            Boolean(True),
        )

    @cute.jit
    def _linear_idx_from_tile(self, tile_idx):
        params = self.tile_sched_params
        cluster_idx, s_idx, b_idx, split_kv_idx = tile_idx
        return (
            ((split_kv_idx * params.problem_shape_b + b_idx) * params.problem_shape_s)
            + s_idx
        ) * params.cluster_shape_mnk[0] + cluster_idx

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=(work_tile, skip_work_tile))
    @cute.jit
    def init_work_tile(self, stage_info: StageInfo):
        """Seed captured schedules from the current custom MLA work tile."""
        return stage_info.work_tile, Boolean(False)

    @consumer_work(returns=work_tile)
    @cute.jit
    def advance_tile(self, stage_info: StageInfo):
        """Advance CLC through its response pipeline; static is task-owned."""
        if cutlass.const_expr(self.use_clc_dynamic):
            # DSV4 locally predicates CLC coordinate decode and metadata on a
            # successful cancellation. Dense MLA retains the public scheduler
            # path below. Both produce the same MLA task-local value type.
            assert self.tile_scheduler_config is not None
            assert self.tile_scheduler_config.response_ptr is not None
            assert self.tile_scheduler is not None
            assert self.pipeline_config is not None
            stage_response_ptr = self.tile_scheduler_config.response_ptr
            if cutlass.const_expr(self.pipeline_config.num_stages > 1):
                stage_response_ptr = stage_response_ptr + stage_info.stage_idx
            if cutlass.const_expr(self.cfg.is_dynamic_token_sparse):
                return self._dsv4_work_tile_from_clc_response(stage_response_ptr)
            work_tile = self.tile_scheduler.work_tile_info_from_clc_response(
                stage_response_ptr
            )
            return self._work_tile_from_clc_tile(work_tile)
        return stage_info.work_tile


# =====================================================================
# PageOffsetWindowResource — TMA-warp register page-table window
# =====================================================================


@dataclass(kw_only=True)
class PageOffsetWindowResource(HighThroughputMlaResource):
    """GMEM page table cached as one register per lane of the TMA warp."""

    page_offsets: Any = None  # GMEM page-offset tensor
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cached_k_pages: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_v_pages: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_next_v_pages: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_window_page: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "cached_k_pages",
            cutlass.Array,
            None,
            "Cached page ids for the current K tile.",
        ),
        (
            "cached_v_pages",
            cutlass.Array,
            None,
            "Cached page ids for the delayed V tile.",
        ),
        (
            "cached_next_v_pages",
            cutlass.Array,
            None,
            "Cached page ids for the next delayed V tile.",
        ),
        (
            "cached_window_page",
            Int32,
            Int32(0),
            "Per-lane page id retained across one 32-entry page-table window.",
        ),
    )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            cached_k_pages,
            cached_v_pages,
            cached_next_v_pages,
            cached_window_page,
        ),
    )
    @cute.jit
    def init_read_state(self, stage_info: StageInfo):
        """Create cached page-index arrays used by staged KV TMA loads."""
        del stage_info
        cfg = self.cfg
        return (
            cutlass.Array(
                Int32,
                cfg.pages_per_k_cta,
                space=cutlass.AddressSpace.rmem,
            ),
            cutlass.Array(
                Int32,
                cfg.pages_per_v_tile,
                space=cutlass.AddressSpace.rmem,
            ),
            cutlass.Array(
                Int32,
                cfg.pages_per_v_tile,
                space=cutlass.AddressSpace.rmem,
            ),
            Int32(0),
        )

    @consumer_work(
        returns=(
            cached_k_pages,
            cached_v_pages,
            cached_next_v_pages,
            cached_window_page,
        )
    )
    @cute.jit
    def read_page_offset_window(
        self,
        stage_info: StageInfo,
        *,
        cached_k_pages,
        cached_v_pages,
        cached_next_v_pages,
        cached_window_page,
        init_v_cache: cutlass.Constexpr[bool] = False,
    ):
        """Load and reuse one warp-wide 32-page-ID window.

        The TMA warp owns one page ID per lane. It refreshes the register
        window after ``32 / pages_per_k_tile`` logical K tiles, when the next
        group of 32 page-table entries is needed, then uses indexed warp
        shuffles to assemble the IDs for the current K/V tile. The refresh
        cadence is derived entirely from page and tile geometry.
        """
        cfg = self.cfg
        work_tile = stage_info.work_tile
        blk_coord = work_tile.tile_idx
        local_k_index = Int32(stage_info.loop_offset)
        global_k_index = work_tile.k_index_base + local_k_index
        pages_per_k_tile = cutlass.const_expr(cfg.pages_per_k_tile)
        page_window_tiles = cutlass.const_expr(32 // pages_per_k_tile)
        page_window_mask = cutlass.const_expr(page_window_tiles - 1)
        page_offsets_batch = self.page_offsets[None, blk_coord[2]]
        lane_idx = cute.arch.thread_idx()[0] & Int32(31)

        if (local_k_index & Int32(page_window_mask)) == Int32(0):
            logical_page_idx = global_k_index * Int32(pages_per_k_tile) + lane_idx
            bounded_page_idx = cute.math.min(
                logical_page_idx, Int32(page_offsets_batch.shape[0] - 1)
            )
            cached_window_page = Int32(page_offsets_batch[bounded_page_idx])

        page_lane_base = (local_k_index & Int32(page_window_mask)) * Int32(
            pages_per_k_tile
        )
        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cached_k = cached_k_pages
        cached_v = cached_v_pages
        cached_next_v = cached_next_v_pages

        if cutlass.const_expr(init_v_cache):
            for pk in cutlass.range_constexpr(cfg.pages_per_v_tile):
                cached_v[pk] = Int32(0)
        else:
            for pk in cutlass.range_constexpr(cfg.pages_per_v_tile):
                cached_v[pk] = cached_next_v[pk]

        for pk in cutlass.range_constexpr(cfg.pages_per_k_cta):
            source_lane = (
                page_lane_base + cta_v * Int32(cfg.pages_per_k_cta) + Int32(pk)
                if cfg.pages_per_k_tile > 1
                else page_lane_base
            )
            cached_k[pk] = Int32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_window_page,
                    offset=source_lane,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
            )

        for pk in cutlass.range_constexpr(cfg.pages_per_v_tile):
            source_lane = (
                page_lane_base
                if cfg.pages_per_v_tile == 1
                else page_lane_base + Int32(pk)
            )
            cached_next_v[pk] = Int32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_window_page,
                    offset=source_lane,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
            )
        return cached_k, cached_v, cached_next_v, cached_window_page

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(cached_k_pages, cached_v_pages, cached_next_v_pages),
    )
    @cute.jit
    def forward_page_ids(
        self,
        stage_info: StageInfo,
        *,
        cached_k_pages,
        cached_v_pages,
        cached_next_v_pages,
    ):
        """Forward the latest delayed-V page IDs after the domain loop."""
        del stage_info
        return cached_k_pages, cached_v_pages, cached_next_v_pages


# =====================================================================
# SmemQResource — Q SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemQResource(HighThroughputMlaResource):
    """SMEM Q buffer (latent + rope).  Producer: LoadTma (TMA).  Consumer: Mma.

    Pipeline: TmaUmmaAsync, 1 stage.
    Q is loaded once (LoopFirstIter) and released once (LoopLastIter).
    """

    smem_q_latent: Any = None  # SMEM Q latent array
    smem_q_rope: Any = None  # SMEM Q rope array
    tma_desc_q_latent: Any = None
    tma_desc_q_rope: Any = None
    cu_seqlens_q: Any = None
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize CTA-local Q-load state for the high-throughput path."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0

    @cute.jit
    def _query_tma_coords(
        self,
        dim_coord,
        local_flat_query_row,
        batch_idx,
        storage_flat_query_row,
        query_row_extent,
    ):
        """Return dense or ragged TMA coordinates for one Q dimension slice."""
        if cutlass.const_expr(self.cu_seqlens_q is not None):
            return transform_ragged_coords(
                (dim_coord, storage_flat_query_row),
                ragged_dim_idx=1,
                ragged_box_size=self.cfg.mma_qk_tiler[0] // self.cfg.num_mma_ctas,
                ragged_extent=query_row_extent,
            )
        return dim_coord, local_flat_query_row, batch_idx

    @producer_work
    @cute.jit
    def tma_load(self, stage_info: StageInfo) -> None:
        """TMA load Q latent + Q rope into SMEM (all sub-tiles in one commit)."""
        cfg = self.cfg
        work_tile = stage_info.work_tile
        blk_coord = work_tile.tile_idx

        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_m = cta_v * Int32(cfg.mma_qk_tiler[0] // cfg.num_mma_ctas)
        physical_tile_rows = Int32(cfg.mma_qk_tiler[0])
        flat_query_row = Int32(blk_coord[1]) * physical_tile_rows + coord_m
        batch_idx = Int32(blk_coord[2])
        storage_flat_query_row = flat_query_row
        query_row_extent = Int32(cfg.mma_qk_tiler[0] // cfg.num_mma_ctas)
        if cutlass.const_expr(self.cu_seqlens_q is not None):
            q_start, q_len = query_batch_bounds(
                self.cu_seqlens_q,
                batch_idx,
                self.logical_seq_len_q,
            )
            storage_flat_query_row = (
                q_start * Int32(self.logical_num_heads_q) + flat_query_row
            )
            query_row_extent = q_len * Int32(self.logical_num_heads_q) - flat_query_row
        mask_q = Int16(Int32(1) << cta_v)
        q_mbar_arr = cutlass.Array(stage_info.barrier.data_ptr(), dtype=Int64)

        # In the DSV4 split-load schedule this resource is called by the
        # W12--W15 Gather4 task.  The generated kernel permits every one of
        # those warps to enter the Q pipeline, but emits its TMA transactions
        # only from its first warp (W12).  Without this predicate each warp
        # independently elects a lane and sends the same TMA four times.
        issue_q_tma = prims.elect_sync()
        if cutlass.const_expr(cfg.is_dynamic_token_sparse):
            issue_q_tma = issue_q_tma & (
                cute.arch.make_warp_uniform(cute.arch.warp_idx())
                == Int32(cfg.load_v_warp_id)
            )
        if issue_q_tma:
            # Load Q latent sub-tiles
            q_latent_stage_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[0] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
            )
            for i in cutlass.range(cfg.iterations_qk_latent):
                coord_kl = cutlass.Int32(i * cfg.mma_qk_tiler_k)
                q_latent_coords = self._query_tma_coords(
                    coord_kl,
                    flat_query_row,
                    batch_idx,
                    storage_flat_query_row,
                    query_row_extent,
                )
                q_smem_arr = cutlass.Array(
                    self.smem_q_latent.data_ptr(i * q_latent_stage_elems),
                    dtype=qkv_dtype(cfg),
                )
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    q_smem_arr,
                    self.tma_desc_q_latent,
                    q_latent_coords,
                    q_mbar_arr,
                    [],
                    multicast_mask=mask_q,
                    group=prims.CTAGroup.CTA_2,
                )
            # Load Q rope sub-tiles
            for i in cutlass.range(cfg.iterations_qk_rope):
                coord_kr = cutlass.Int32(i * cfg.mma_qk_tiler_k)
                q_rope_coords = self._query_tma_coords(
                    coord_kr,
                    flat_query_row,
                    batch_idx,
                    storage_flat_query_row,
                    query_row_extent,
                )
                qr_smem_arr = cutlass.Array(
                    self.smem_q_rope.data_ptr(), dtype=qkv_dtype(cfg)
                )
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    qr_smem_arr,
                    self.tma_desc_q_rope,
                    q_rope_coords,
                    q_mbar_arr,
                    [],
                    multicast_mask=mask_q,
                    group=prims.CTAGroup.CTA_2,
                )

    @consumer_work
    @cute.jit
    def q_desc(self, stage_info: StageInfo) -> None:
        """Schedule marker after the Q SMEM stage is waited."""
        del stage_info


# =====================================================================
# SmemKVResource — K/V SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemKVResource(HighThroughputMlaResource):
    """SMEM K/V buffer.  Producer: LoadTma (TMA).  Consumer: Mma.

    Pipeline: TmaUmmaAsync, 7 stages.
    Per logical k-tile: K sub-tiles are loaded before delayed V sub-tiles.
    """

    smem_kv: Any = None
    tma_desc_c_latent: Any = None
    tma_desc_c_rope: Any = None
    tma_desc_c_transpose: Any = None
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)
    desc_k_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    desc_v_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_k_base", Int64, Int64(0), "SMEM descriptor for staged K."),
        ("desc_v_base", Int64, Int64(0), "SMEM descriptor for staged V."),
    )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize CTA-local KV-load descriptors and leader state."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0

    @producer_work
    @cute.jit
    def tma_load(
        self,
        stage_info: StageInfo,
        *,
        cached_k_pages,
        cached_v_pages,
        cached_next_v_pages,
        is_v: cutlass.Constexpr[bool],
        subtile_idx: cutlass.Constexpr[int],
        use_next_v_pages: cutlass.Constexpr[bool] = False,
    ) -> None:
        """TMA load one K or delayed-V sub-tile into the shared KV ring."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        kc_page_smem_elems = cfg.kc_page_tile_size * cfg.mma_qk_tiler_k
        kv_mbar_arr = cutlass.Array(stage_info.barrier.data_ptr(), dtype=Int64)
        kv_stage_stride_elems = cutlass.const_expr(cfg.smem_k_stage_elems)

        pages_per_k_cta = cfg.pages_per_k_cta

        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_n_k = (cta_v * Int32(cfg.mma_qk_tiler[1] // cfg.num_mma_ctas)) % Int32(
            cfg.page_size
        )
        mask_k = Int16(Int32(1) << cta_v)

        is_v_subtile = is_v
        k_call = subtile_idx

        if cutlass.const_expr(
            not is_v_subtile and k_call < cfg.iterations_qk_latent_stages
        ):
            cached_k = cached_k_pages
            k_subtile_smem_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
            )
            for stage_subtile_idx in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
                logical_k_call = k_call * cfg.kv_subtiles_per_stage + stage_subtile_idx
                coord_kcl = cutlass.Int32(logical_k_call * cfg.mma_qk_tiler_k)
                for pk in cutlass.range_constexpr(pages_per_k_cta):
                    if prims.elect_sync():
                        kcl_smem = cutlass.Array(
                            self.smem_kv.data_ptr(
                                stage_idx * kv_stage_stride_elems
                                + stage_subtile_idx * k_subtile_smem_elems
                                + pk * kc_page_smem_elems
                            ),
                            dtype=qkv_dtype(cfg),
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            kcl_smem,
                            self.tma_desc_c_latent,
                            (coord_kcl, coord_n_k, cutlass.Int32(cached_k[pk])),
                            kv_mbar_arr,
                            [],
                            multicast_mask=mask_k,
                            group=prims.CTAGroup.CTA_2,
                        )

        elif cutlass.const_expr(not is_v_subtile and k_call < cfg.iterations_qk_stages):
            rope_idx = k_call - cfg.iterations_qk_latent_stages
            cached_k = cached_k_pages

            coord_kcr = cutlass.Int32(rope_idx * cfg.mma_qk_rope_tiler[2])
            rope_page_smem_elems = cfg.kc_page_tile_size * cfg.mma_qk_rope_tiler[2]
            rope_tile_smem_elems = (
                cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_rope_tiler[2]
            )
            for pk in cutlass.range_constexpr(pages_per_k_cta):
                if prims.elect_sync():
                    kcr_smem = cutlass.Array(
                        self.smem_kv.data_ptr(
                            stage_idx * kv_stage_stride_elems
                            + pk * rope_page_smem_elems
                        ),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        kcr_smem,
                        self.tma_desc_c_rope,
                        (coord_kcr, coord_n_k, cutlass.Int32(cached_k[pk])),
                        kv_mbar_arr,
                        [],
                        multicast_mask=mask_k,
                        group=prims.CTAGroup.CTA_2,
                    )
                    if cutlass.const_expr(cfg.kv_subtiles_per_stage > 1):
                        # The BF16 stage contains two legal K64 transfers.
                        # Duplicate the final K64 RoPE slice into the unused
                        # half so its mbarrier transaction count matches the
                        # common K128 stage contract.
                        kcr_smem_dup = cutlass.Array(
                            self.smem_kv.data_ptr(
                                stage_idx * kv_stage_stride_elems
                                + rope_tile_smem_elems
                                + pk * rope_page_smem_elems
                            ),
                            dtype=qkv_dtype(cfg),
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            kcr_smem_dup,
                            self.tma_desc_c_rope,
                            (coord_kcr, coord_n_k, cutlass.Int32(cached_k[pk])),
                            kv_mbar_arr,
                            [],
                            multicast_mask=mask_k,
                            group=prims.CTAGroup.CTA_2,
                        )

        else:
            pv_n_per_cta = cutlass.const_expr(cfg.mma_pv_tiler[1] // cfg.num_mma_ctas)
            coord_n_v = cta_v * pv_n_per_cta
            mask_v = Int16(Int32(1) << cta_v)
            v_tma_copy_smem_elems = cfg.v_tma_token_count * V_TMA_LATENT_ELEMENTS
            v_subtile_smem_elems = cutlass.const_expr(
                cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
            )

            pages_per_v_subtile = cfg.pages_per_v_subtile
            cached_v = (
                cached_next_v_pages
                if cutlass.const_expr(use_next_v_pages)
                else cached_v_pages
            )

            # A physical V stage contains two adjacent K32 slices for one
            # D256 output panel. Across four stages this is the same
            # head-dimension-128/token-partition-2 decomposition used by the
            # 2CTA BF16 schedule: (D0,K0:64), (D0,K64:128), then D256.
            pv_j = subtile_idx // cfg.kv_subtiles_per_stage
            token_partition = subtile_idx % cfg.kv_subtiles_per_stage
            for stage_subtile_idx in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
                pv_i = token_partition * cfg.kv_subtiles_per_stage + stage_subtile_idx
                coord_k_v = cutlass.Int32((pv_i * cfg.mma_pv_tiler[2]) % cfg.page_size)
                coord_nj = coord_n_v + cutlass.Int32(pv_j * cfg.mma_pv_tiler[1])

                for pk in cutlass.range_constexpr(pages_per_v_subtile):
                    k_idx_i = cached_v[
                        pk + pv_i // cfg.v_subtiles_per_page * pages_per_v_subtile
                    ]
                    if prims.elect_sync():
                        # Keep both 64-wide V panels contiguous within each
                        # K32 slice; the next K32 slice follows the complete
                        # first slice in this physical stage.
                        stage_subtile_base = stage_subtile_idx * v_subtile_smem_elems
                        v_page_offset = pk * v_tma_copy_smem_elems
                        v_second_panel_offset = (
                            pages_per_v_subtile * v_tma_copy_smem_elems + v_page_offset
                        )
                        v_smem_0 = cutlass.Array(
                            self.smem_kv.data_ptr(
                                stage_idx * kv_stage_stride_elems
                                + stage_subtile_base
                                + v_page_offset
                            ),
                            dtype=qkv_dtype(cfg),
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            v_smem_0,
                            self.tma_desc_c_transpose,
                            (coord_nj, coord_k_v, k_idx_i),
                            kv_mbar_arr,
                            [],
                            multicast_mask=mask_v,
                            group=prims.CTAGroup.CTA_2,
                        )
                        v_smem_1 = cutlass.Array(
                            self.smem_kv.data_ptr(
                                stage_idx * kv_stage_stride_elems
                                + stage_subtile_base
                                + v_second_panel_offset
                            ),
                            dtype=qkv_dtype(cfg),
                        )
                        prims.cp_async_bulk_tensor_shared_cluster_global(
                            v_smem_1,
                            self.tma_desc_c_transpose,
                            (
                                coord_nj + V_TMA_LATENT_ELEMENTS,
                                coord_k_v,
                                k_idx_i,
                            ),
                            kv_mbar_arr,
                            [],
                            multicast_mask=mask_v,
                            group=prims.CTAGroup.CTA_2,
                        )

    @consumer_work(returns=desc_k_base)
    @cute.jit
    def k_desc(self, stage_info: StageInfo, *, k_subtile_idx: cutlass.Constexpr[int]):
        """Build the SMEM descriptor consumed by QK MMA."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        kc_copy_elems = cutlass.const_expr(cfg.smem_k_stage_elems)
        tile_rows = cutlass.const_expr(cfg.mma_qk_tiler[1] // cfg.num_mma_ctas)
        leading_byte_offset = cutlass.const_expr(qk_desc_leading_byte_offset(cfg))
        stride_byte_offset = cutlass.const_expr(qk_desc_stride_byte_offset(cfg))
        layout = cutlass.const_expr(qk_desc_layout(cfg))
        if cutlass.const_expr(
            cfg.is_fp8_qkv()
            and k_subtile_idx >= cfg.iterations_qk_latent
            and k_subtile_idx < cfg.iterations_qk
        ):
            leading_byte_offset = cutlass.const_expr(
                qk_desc_leading_byte_offset_for_head_dim(
                    cfg, tile_rows, cfg.mma_qk_rope_tiler[2]
                )
            )
            stride_byte_offset = cutlass.const_expr(
                qk_desc_stride_byte_offset_for_head_dim(cfg, cfg.mma_qk_rope_tiler[2])
            )
            layout = cutlass.const_expr(
                qk_desc_layout_for_head_dim(cfg, cfg.mma_qk_rope_tiler[2])
            )

        sk_ptr = self.smem_kv.data_ptr(stage_idx * kc_copy_elems)
        desc = Int64(
            prims.Tcgen05SmemDesc.build(
                start_address=sk_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc(self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]):
        """Build the SMEM descriptor consumed by PV MMA."""
        del v_subtile_idx
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        svc_copy_elems = cutlass.const_expr(cfg.smem_k_stage_elems)
        leading_byte_offset = cutlass.const_expr(4096)
        stride_byte_offset = cutlass.const_expr(1024)
        layout = cutlass.const_expr(2)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            leading_byte_offset = cutlass.const_expr(
                V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS * cfg.qkv_dtype_bytes
            )
            stride_byte_offset = cutlass.const_expr(
                qkv_major_k_stride_bytes_for(cfg, cfg.mma_pv_tiler[2])
            )
            layout = cutlass.const_expr(
                qk_desc_layout_for_head_dim(cfg, cfg.mma_pv_tiler[2])
            )

        svc_ptr = self.smem_kv.data_ptr(stage_idx * svc_copy_elems)
        desc = Int64(
            prims.Tcgen05SmemDesc.build(
                start_address=svc_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc


# =====================================================================
# SmemKResource — FP8 K SMEM buffer with one pipeline stage per K tile
# =====================================================================


@dataclass(kw_only=True)
class SmemKResource(HighThroughputMlaResource):
    """SMEM K buffer for the FP8 split-MMA path.

    Producer: LoadTma. Consumer: MmaQkTask. A single producer stage contains
    all latent K sub-tiles plus the RoPE K sub-tile for one logical K tile.
    """

    smem_k: Any = None
    page_offsets: Any = None
    tma_desc_c_latent: Any = None
    tma_desc_c_rope: Any = None
    sparse_tma_atom_swa: Any = None
    sparse_tma_tensor_swa: Any = None
    sparse_tma_atom_compressed: Any = None
    sparse_tma_tensor_compressed: Any = None
    raw_tma_descriptor_swa: Any = None
    raw_tma_descriptor_compressed: Any = None
    sparse_coord_tensor: Any = None
    sparse_smem_tensor: Any = None
    smem_page_offsets: Any = None
    cu_seqlens_q: Any = None
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    desc_k_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_k_base", Int64, Int64(0), "SMEM descriptor for staged K."),
    )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize per-warp state used by K TMA loads."""
        del stage_info

    @producer_work
    @cute.jit
    def tma_load_direct(self, stage_info: StageInfo) -> None:
        """TMA load one dense K tile using page offsets read from GMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        stage_base = stage_idx * cfg.smem_k_stage_elems
        work_tile = stage_info.work_tile
        blk_coord = work_tile.tile_idx
        k_index = work_tile.k_index_base + Int32(stage_info.loop_offset)

        page_row_idx = routing_row_index(
            cfg, blk_coord, self.logical_seq_len_q, self.cu_seqlens_q
        )
        page_offsets_batch = self.page_offsets[None, page_row_idx]
        kv_mbar_arr = cutlass.Array(stage_info.barrier.data_ptr(), dtype=Int64)
        pages_per_k_cta = cfg.pages_per_k_cta
        kc_page_smem_elems = cfg.kc_page_tile_size * cfg.mma_qk_tiler_k
        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        coord_n_k = (cta_v * Int32(cfg.mma_qk_tiler[1] // cfg.num_mma_ctas)) % Int32(
            cfg.page_size
        )
        mask_k = Int16(Int32(1) << cta_v)
        k_latent_subtile_elems = cutlass.const_expr(
            cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
        )
        # Use page 0 for masked fragments beyond a compact page table. A direct
        # bounds predicate avoids carrying runtime K-length division or a full
        # page-ID register array through this producer warp.

        for k_call in cutlass.range_constexpr(cfg.iterations_qk_latent):
            coord_kcl = cutlass.Int32(k_call * cfg.mma_qk_tiler_k)
            subtile_base = stage_base + k_call * k_latent_subtile_elems
            for pk in cutlass.range_constexpr(pages_per_k_cta):
                logical_page_idx = (
                    k_index
                    if cfg.pages_per_k_tile == 1
                    else (k_index * Int32(cfg.num_mma_ctas) + cta_v)
                    * Int32(pages_per_k_cta)
                    + Int32(pk)
                )
                page_idx = Int32(0)
                if cute.elem_less(logical_page_idx, page_offsets_batch.shape[0]):
                    page_idx = page_offsets_batch[logical_page_idx]
                if prims.elect_sync():
                    kcl_smem = cutlass.Array(
                        self.smem_k.data_ptr(subtile_base + pk * kc_page_smem_elems),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        kcl_smem,
                        self.tma_desc_c_latent,
                        (coord_kcl, coord_n_k, cutlass.Int32(page_idx)),
                        kv_mbar_arr,
                        [],
                        multicast_mask=mask_k,
                        group=prims.CTAGroup.CTA_2,
                    )

        rope_stage_base = stage_base + k_latent_subtile_elems * cfg.iterations_qk_latent
        rope_page_smem_elems = cfg.kc_page_tile_size * cfg.mma_qk_rope_tiler[2]
        k_rope_subtile_elems = cutlass.const_expr(
            cfg.mma_qk_rope_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_rope_tiler[2]
        )
        for rope_idx in cutlass.range_constexpr(cfg.iterations_qk_rope):
            coord_kcr = cutlass.Int32(rope_idx * cfg.mma_qk_rope_tiler[2])
            subtile_base = rope_stage_base + rope_idx * k_rope_subtile_elems
            for pk in cutlass.range_constexpr(pages_per_k_cta):
                logical_page_idx = (
                    k_index
                    if cfg.pages_per_k_tile == 1
                    else (k_index * Int32(cfg.num_mma_ctas) + cta_v)
                    * Int32(pages_per_k_cta)
                    + Int32(pk)
                )
                page_idx = Int32(0)
                if cute.elem_less(logical_page_idx, page_offsets_batch.shape[0]):
                    page_idx = page_offsets_batch[logical_page_idx]
                if prims.elect_sync():
                    kcr_smem = cutlass.Array(
                        self.smem_k.data_ptr(subtile_base + pk * rope_page_smem_elems),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        kcr_smem,
                        self.tma_desc_c_rope,
                        (coord_kcr, coord_n_k, cutlass.Int32(page_idx)),
                        kv_mbar_arr,
                        [],
                        multicast_mask=mask_k,
                        group=prims.CTAGroup.CTA_2,
                    )

    @producer_work
    @cute.jit
    def tma_load_from_page_ring(
        self,
        stage_info: StageInfo,
        *,
        page_offset_stage: Int32,
        k_tile_delta: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Issue the DSV4 Gather4 K tile selected by W9's SMEM ring."""
        self._tma_load_sparse_from_ring(
            stage_info,
            page_offset_stage=page_offset_stage,
            k_tile_delta=k_tile_delta,
        )

    @producer_work
    @cute.jit
    def tma_load_from_page_ring_pair(
        self,
        stage_info: StageInfo,
        *,
        page_offset_stage: Int32,
    ) -> None:
        """Issue K from the source pair-ring half selected by its K-tile parity.

        Unlike ``k_tile_delta``, selector-half selection must not change the
        logical K index or producer pipeline stage.  This is used only by the
        dedicated source-pair FSM, whose ``stage_info.loop_offset`` remains
        the full K-tile index while W9 holds one selector stage across even
        and odd tiles.
        """
        self._tma_load_sparse_from_ring(
            stage_info,
            page_offset_stage=page_offset_stage,
            page_offset_half=Int32(stage_info.loop_offset) & Int32(1),
        )

    @producer_work
    @cute.jit
    def tma_load_sparse(
        self,
        stage_info: StageInfo,
        *,
        k_tile_delta: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Issue the coordinate-tensor Gather4 K fallback for diagnosis."""
        self._tma_load_sparse(stage_info, k_tile_delta=k_tile_delta)

    @cute.jit
    def _issue_sparse_ring_bundle(
        self,
        *,
        raw_tma,
        stage_base,
        cta_rank,
        local_load_warp,
        completion_mbarrier,
        page_offset_stage,
        page_offset_half,
    ) -> None:
        """Issue one source-shaped K Gather4 bundle for a fixed tensor map.

        Keeping the descriptor fixed across this inlined bundle is important:
        ptxas can predicate each elected-lane TMA exactly as it does for the
        generated source.  A descriptor branch inside the elected region
        instead creates one divergent reconvergence region per transaction.
        """
        cfg = self.cfg
        for quad_idx in cutlass.range_constexpr(4):
            token_group = local_load_warp * Int32(4) + Int32(quad_idx * 16)
            page_quad = self.smem_page_offsets.load(
                page_offset_stage * Int32(cfg.dsv4_page_offsets_entries_per_stage)
                + page_offset_half * Int32(128)
                + cta_rank * Int32(64)
                + token_group,
                vector_size=4,
                alignment=16,
            )
            dst_token_base = stage_base + token_group * Int32(128)
            for head_dim_stage in cutlass.range_constexpr(4):
                dst_smem = cute.make_ptr(
                    qkv_dtype(cfg),
                    self.smem_k.data_ptr(dst_token_base + Int32(head_dim_stage * 8192)),
                    mem_space=cutlass.AddressSpace.smem,
                )
                if prims.elect_sync():
                    issue_gather4_tma_2cta(
                        raw_tma,
                        dst_smem,
                        completion_mbarrier,
                        Int32(head_dim_stage * 128),
                        (page_quad[0], page_quad[1], page_quad[2], page_quad[3]),
                    )

    @cute.jit
    def _tma_load_sparse_from_ring(
        self,
        stage_info: StageInfo,
        *,
        page_offset_stage: Int32,
        k_tile_delta: cutlass.Constexpr[int] = 0,
        page_offset_half: Int32 | None = None,
    ) -> None:
        """Issue the source W12--W15 K Gather4 work from the W9 SMEM ring.

        Every load warp owns four page quads and emits all four D128 slices.
        This is deliberately a raw Gather4 path: CuTe's high-level atom takes
        indices from a GMEM coordinate tensor, while DSV4's latency-hiding
        contract requires the four indices to come from W9's cp.async ring.
        """

        cfg = self.cfg
        work_tile = stage_info.work_tile
        k_index = (
            work_tile.k_index_base + Int32(stage_info.loop_offset) + Int32(k_tile_delta)
        )
        stage_base = Int32(stage_info.stage_idx) * Int32(cfg.smem_k_stage_elems)
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        local_load_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.load_v_warp_id
        )
        completion_mbarrier = cute.make_ptr(
            Int64,
            stage_info.barrier.data_ptr(),
            mem_space=cutlass.AddressSpace.smem,
            assumed_align=8,
        )
        raw_tma_swa = (
            self.raw_tma_descriptor_swa
            if self.raw_tma_descriptor_swa is not None
            else gather4_tma_descriptor_address(self.sparse_tma_atom_swa)
        )
        raw_tma_compressed = (
            self.raw_tma_descriptor_compressed
            if self.raw_tma_descriptor_compressed is not None
            else gather4_tma_descriptor_address(self.sparse_tma_atom_compressed)
        )
        # K's 64 sparse-token rows are split across the two CTA ranks.  The
        # destination mapping and four H128 slices are verbatim from
        # ``SmemKv.h`` in the generated TRT-LLM kernel.
        page_offset_half = (
            Int32(k_tile_delta)
            if cutlass.const_expr(page_offset_half is None)
            else page_offset_half
        )
        # Source places this tile0 branch outside the entire unrolled bundle.
        # Both paths are intentionally separate rather than merged through a
        # descriptor-pointer phi: the latter does not preserve tensor-map
        # pointer semantics in the current CuTe DSL lowering.
        if k_index == Int32(0):
            self._issue_sparse_ring_bundle(
                raw_tma=raw_tma_swa,
                stage_base=stage_base,
                cta_rank=cta_rank,
                local_load_warp=local_load_warp,
                completion_mbarrier=completion_mbarrier,
                page_offset_stage=page_offset_stage,
                page_offset_half=page_offset_half,
            )
        else:
            self._issue_sparse_ring_bundle(
                raw_tma=raw_tma_compressed,
                stage_base=stage_base,
                cta_rank=cta_rank,
                local_load_warp=local_load_warp,
                completion_mbarrier=completion_mbarrier,
                page_offset_stage=page_offset_stage,
                page_offset_half=page_offset_half,
            )

    @cute.jit
    def _tma_load_sparse(
        self,
        stage_info: StageInfo,
        *,
        k_tile_delta: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Gather one DSV4 CSA K tile through the two-CTA Gather4 path."""

        cfg = self.cfg
        work_tile = stage_info.work_tile
        k_index = (
            work_tile.k_index_base + Int32(stage_info.loop_offset) + Int32(k_tile_delta)
        )
        route_row = routing_row_index(
            cfg, work_tile.tile_idx, self.logical_seq_len_q, self.cu_seqlens_q
        )
        tiles_per_route = self.page_offsets.shape[0] // Int32(cfg.mma_qk_tiler[1])
        route_tile = route_row * tiles_per_route + k_index
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        gather_tile_idx = route_tile * Int32(cfg.num_mma_ctas) + cta_rank
        cta_tiler = (
            cfg.mma_qk_tiler[1] // cfg.num_mma_ctas,
            cfg.mma_qk_tiler[2],
        )

        g_swa = cute.zipped_divide(self.sparse_tma_tensor_swa, cta_tiler)
        g_compressed = cute.zipped_divide(self.sparse_tma_tensor_compressed, cta_tiler)
        g_indices = cute.zipped_divide(self.sparse_coord_tensor, cta_tiler)
        t_smem_swa, t_gmem_swa, t_indices_swa = cpasync.tma_partition(
            self.sparse_tma_atom_swa,
            0,
            cute.make_layout(1),
            self.sparse_smem_tensor,
            [g_swa, g_indices],
        )
        t_smem_compressed, t_gmem_compressed, t_indices_compressed = (
            cpasync.tma_partition(
                self.sparse_tma_atom_compressed,
                0,
                cute.make_layout(1),
                self.sparse_smem_tensor,
                [g_compressed, g_indices],
            )
        )
        tma_bar_ptr = cute.make_ptr(
            Int64,
            stage_info.barrier.data_ptr(),
            mem_space=cutlass.AddressSpace.smem,
            assumed_align=8,
        )
        # The source's W12--W15 LoadTask issues one independent K Gather4
        # descriptor per warp.  The four FP8 latent-D128 slices form the same
        # work partition here; one warp remains elected for each descriptor.
        local_load_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.load_v_warp_id
        )
        for k_iter in cutlass.range_constexpr(cfg.iterations_qk_latent):
            if local_load_warp == Int32(k_iter % cfg.load_v_num_warps):
                if k_index == Int32(0):
                    cute.copy(
                        self.sparse_tma_atom_swa,
                        [
                            t_gmem_swa[None, (gather_tile_idx, k_iter)],
                            t_indices_swa[None, (gather_tile_idx, k_iter)],
                        ],
                        t_smem_swa[None, 0, 0, (k_iter, stage_info.stage_idx)],
                        tma_bar_ptr=tma_bar_ptr,
                    )
                else:
                    cute.copy(
                        self.sparse_tma_atom_compressed,
                        [
                            t_gmem_compressed[None, (gather_tile_idx, k_iter)],
                            t_indices_compressed[None, (gather_tile_idx, k_iter)],
                        ],
                        t_smem_compressed[None, 0, 0, (k_iter, stage_info.stage_idx)],
                        tma_bar_ptr=tma_bar_ptr,
                    )

    @consumer_work(returns=desc_k_base)
    @cute.jit
    def k_desc(self, stage_info: StageInfo, *, k_subtile_idx: cutlass.Constexpr[int]):
        """Build the K SMEM descriptor for the current QK sub-MMA."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        k_latent_subtile_elems = cutlass.const_expr(
            cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
        )
        k_rope_subtile_elems = cutlass.const_expr(
            cfg.mma_qk_rope_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_rope_tiler[2]
        )
        subtile_offset = stage_idx * cfg.smem_k_stage_elems
        tile_rows = cutlass.const_expr(cfg.mma_qk_tiler[1] // cfg.num_mma_ctas)
        leading_byte_offset = cutlass.const_expr(qk_desc_leading_byte_offset(cfg))
        stride_byte_offset = cutlass.const_expr(qk_desc_stride_byte_offset(cfg))
        layout = cutlass.const_expr(qk_desc_layout(cfg))

        if cutlass.const_expr(k_subtile_idx < cfg.iterations_qk_latent):
            subtile_offset += k_subtile_idx * k_latent_subtile_elems
        else:
            rope_idx = k_subtile_idx - cfg.iterations_qk_latent
            subtile_offset += (
                k_latent_subtile_elems * cfg.iterations_qk_latent
                + rope_idx * k_rope_subtile_elems
            )
            leading_byte_offset = cutlass.const_expr(
                qk_desc_leading_byte_offset_for_head_dim(
                    cfg, tile_rows, cfg.mma_qk_rope_tiler[2]
                )
            )
            stride_byte_offset = cutlass.const_expr(
                qk_desc_stride_byte_offset_for_head_dim(cfg, cfg.mma_qk_rope_tiler[2])
            )
            layout = cutlass.const_expr(
                qk_desc_layout_for_head_dim(cfg, cfg.mma_qk_rope_tiler[2])
            )

        sk_ptr = self.smem_k.data_ptr(subtile_offset)
        desc = Int64(
            prims.Tcgen05SmemDesc.build(
                start_address=sk_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc


# =====================================================================
# SmemVResource — FP8 V SMEM buffer with one pipeline stage per V tile
# =====================================================================


@dataclass(kw_only=True)
class SmemVResource(HighThroughputMlaResource):
    """SMEM V buffer for the FP8 split-MMA path.

    Producer: LoadTma. Consumer: MmaPvTask. A single stage contains every V
    sub-tile needed by the PV MMA for one logical K tile.
    """

    smem_v: Any = None
    page_offsets: Any = None
    tma_desc_c_transpose: Any = None
    sparse_tma_atom_swa: Any = None
    sparse_tma_tensor_swa: Any = None
    sparse_tma_atom_compressed: Any = None
    sparse_tma_tensor_compressed: Any = None
    # Generated DSV4 uses the K tensor map for both raw K and raw V Gather4
    # instructions.  Keep V's own atom for the high-level debugging fallback.
    raw_tma_atom_swa: Any = None
    raw_tma_atom_compressed: Any = None
    raw_tma_descriptor_swa: Any = None
    raw_tma_descriptor_compressed: Any = None
    sparse_coord_tensor: Any = None
    sparse_smem_tensor: Any = None
    smem_page_offsets: Any = None
    cu_seqlens_q: Any = None
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    desc_v_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_v_base", Int64, Int64(0), "SMEM descriptor for staged V."),
    )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize per-warp state used by V TMA loads."""
        del stage_info

    @producer_work
    @cute.jit
    def tma_load_direct(self, stage_info: StageInfo) -> None:
        """TMA load one dense V tile using page offsets read from GMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        stage_base = stage_idx * cfg.smem_v_stage_elems
        work_tile = stage_info.work_tile
        blk_coord = work_tile.tile_idx
        k_index = work_tile.k_index_base + Int32(stage_info.loop_offset)
        page_row_idx = routing_row_index(
            cfg, blk_coord, self.logical_seq_len_q, self.cu_seqlens_q
        )
        page_offsets_batch = self.page_offsets[None, page_row_idx]

        v_mbar_arr = cutlass.Array(stage_info.barrier.data_ptr(), dtype=Int64)
        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        pv_n_per_cta = cutlass.const_expr(cfg.mma_pv_tiler[1] // cfg.num_mma_ctas)
        coord_n_v = cta_v * pv_n_per_cta
        mask_v = Int16(Int32(1) << cta_v)
        pages_per_v_tile = cfg.pages_per_v_tile
        v_smem_panel_elems = V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS
        v_smem_k_block_elems = cfg.num_mma_ctas * v_smem_panel_elems
        svc_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        # As for K, out-of-table fragments use page 0 and are masked by the
        # runtime sequence length. p128 still shares one page ID across CTAs.

        for v_call in cutlass.range_constexpr(
            cfg.iterations_pv_k * cfg.iterations_pv_n
        ):
            pv_i = v_call // cfg.iterations_pv_n
            pv_j = v_call % cfg.iterations_pv_n
            coord_nj = coord_n_v + cutlass.Int32(pv_j * cfg.mma_pv_tiler[1])
            subtile_base = stage_base + v_call * svc_copy_elems

            # Assemble the fixed K32 SMEM blocks consumed by tcgen05 from
            # page-bounded TMA copies.  Physical pages only select the GMEM
            # page/coordinate; they never change the SMEM descriptor layout.
            for copy_idx in cutlass.range_constexpr(cfg.v_tma_copies_per_subtile):
                local_token_offset = copy_idx * cfg.v_tma_token_count
                tile_token_offset = pv_i * cfg.mma_pv_tiler[2] + local_token_offset
                page_offset = tile_token_offset // cfg.page_size
                coord_k_v = Int32(tile_token_offset % cfg.page_size)
                logical_page_idx = (
                    k_index
                    if pages_per_v_tile == 1
                    else k_index * Int32(pages_per_v_tile) + Int32(page_offset)
                )
                page_idx = Int32(0)
                if cute.elem_less(logical_page_idx, page_offsets_batch.shape[0]):
                    page_idx = page_offsets_batch[logical_page_idx]
                if prims.elect_sync():
                    smem_k_block_idx = local_token_offset // V_SMEM_K_BLOCK_TOKENS
                    token_offset_in_k_block = local_token_offset % V_SMEM_K_BLOCK_TOKENS
                    v_copy_offset = (
                        smem_k_block_idx * v_smem_k_block_elems
                        + token_offset_in_k_block * V_TMA_LATENT_ELEMENTS
                    )
                    v_smem_0 = cutlass.Array(
                        self.smem_v.data_ptr(subtile_base + v_copy_offset),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        v_smem_0,
                        self.tma_desc_c_transpose,
                        (coord_nj, coord_k_v, page_idx),
                        v_mbar_arr,
                        [],
                        multicast_mask=mask_v,
                        group=prims.CTAGroup.CTA_2,
                    )
                    v_smem_1 = cutlass.Array(
                        self.smem_v.data_ptr(
                            subtile_base + v_copy_offset + v_smem_panel_elems
                        ),
                        dtype=qkv_dtype(cfg),
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        v_smem_1,
                        self.tma_desc_c_transpose,
                        (
                            coord_nj + V_TMA_LATENT_ELEMENTS,
                            coord_k_v,
                            page_idx,
                        ),
                        v_mbar_arr,
                        [],
                        multicast_mask=mask_v,
                        group=prims.CTAGroup.CTA_2,
                    )

    @producer_work
    @cute.jit
    def tma_load_from_page_ring(
        self,
        stage_info: StageInfo,
        *,
        page_offset_stage: Int32,
        k_tile_delta: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Issue the DSV4 Gather4 V tile selected by W9's SMEM ring."""
        self._tma_load_sparse_from_ring(
            stage_info,
            page_offset_stage=page_offset_stage,
            k_tile_delta=k_tile_delta,
        )

    @producer_work
    @cute.jit
    def tma_load_from_page_ring_pair(
        self,
        stage_info: StageInfo,
        *,
        page_offset_stage: Int32,
    ) -> None:
        """Issue V from the source pair-ring half selected by K-tile parity."""
        self._tma_load_sparse_from_ring(
            stage_info,
            page_offset_stage=page_offset_stage,
            page_offset_half=Int32(stage_info.loop_offset) & Int32(1),
        )

    @producer_work
    @cute.jit
    def tma_load_sparse(
        self,
        stage_info: StageInfo,
        *,
        k_tile_delta: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Issue the layout-transforming V Gather4 fallback for PV K64."""
        self._tma_load_sparse(stage_info, k_tile_delta=k_tile_delta)

    @cute.jit
    def _issue_sparse_ring_bundle(
        self,
        *,
        raw_tma,
        stage_base,
        cta_rank,
        local_load_warp,
        completion_mbarrier,
        page_offset_stage,
        page_offset_half,
    ) -> None:
        """Issue one source-shaped V Gather4 bundle for a fixed tensor map."""
        cfg = self.cfg
        for quad_idx in cutlass.range_constexpr(8):
            token_group = local_load_warp * Int32(4) + Int32(quad_idx * 16)
            page_quad = self.smem_page_offsets.load(
                page_offset_stage * Int32(cfg.dsv4_page_offsets_entries_per_stage)
                + page_offset_half * Int32(128)
                + token_group,
                vector_size=4,
                alignment=16,
            )
            dst_token_base = stage_base + token_group * Int32(128)
            for head_dim_stage in cutlass.range_constexpr(2):
                dst_smem = cute.make_ptr(
                    qkv_dtype(cfg),
                    self.smem_v.data_ptr(
                        dst_token_base + Int32(head_dim_stage * 16384)
                    ),
                    mem_space=cutlass.AddressSpace.smem,
                )
                if prims.elect_sync():
                    issue_gather4_tma_2cta(
                        raw_tma,
                        dst_smem,
                        completion_mbarrier,
                        cta_rank * Int32(128) + Int32(head_dim_stage * 256),
                        (page_quad[0], page_quad[1], page_quad[2], page_quad[3]),
                    )

    @cute.jit
    def _tma_load_sparse_from_ring(
        self,
        stage_info: StageInfo,
        *,
        page_offset_stage: Int32,
        k_tile_delta: cutlass.Constexpr[int] = 0,
        page_offset_half: Int32 | None = None,
    ) -> None:
        """Issue the source W12--W15 V Gather4 work from the W9 SMEM ring."""

        cfg = self.cfg
        work_tile = stage_info.work_tile
        k_index = (
            work_tile.k_index_base + Int32(stage_info.loop_offset) + Int32(k_tile_delta)
        )
        stage_base = Int32(stage_info.stage_idx) * Int32(cfg.smem_v_stage_elems)
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        local_load_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.load_v_warp_id
        )
        completion_mbarrier = cute.make_ptr(
            Int64,
            stage_info.barrier.data_ptr(),
            mem_space=cutlass.AddressSpace.smem,
            assumed_align=8,
        )
        raw_tma_swa = (
            self.raw_tma_descriptor_swa
            if self.raw_tma_descriptor_swa is not None
            else gather4_tma_descriptor_address(
                self.raw_tma_atom_swa
                if self.raw_tma_atom_swa is not None
                else self.sparse_tma_atom_swa
            )
        )
        raw_tma_compressed = (
            self.raw_tma_descriptor_compressed
            if self.raw_tma_descriptor_compressed is not None
            else gather4_tma_descriptor_address(
                self.raw_tma_atom_compressed
                if self.raw_tma_atom_compressed is not None
                else self.sparse_tma_atom_compressed
            )
        )
        # V is multicast across the CTA pair, so its source page quad has no
        # CTA-rank term. CTA rank instead selects interleaved V128 columns.
        # The generated kernel gives every W12--W15 warp eight interleaved
        # page quads and two H256 slices each: 4 * 8 * 2 = 64 Gather4
        # transactions, satisfying the source-sized 64 KiB completion
        # barrier. Four quads would cover just 64 KV rows and deadlock the
        # consumer wait with half of the expected transaction count missing.
        page_offset_half = (
            Int32(k_tile_delta)
            if cutlass.const_expr(page_offset_half is None)
            else page_offset_half
        )
        if k_index == Int32(0):
            self._issue_sparse_ring_bundle(
                raw_tma=raw_tma_swa,
                stage_base=stage_base,
                cta_rank=cta_rank,
                local_load_warp=local_load_warp,
                completion_mbarrier=completion_mbarrier,
                page_offset_stage=page_offset_stage,
                page_offset_half=page_offset_half,
            )
        else:
            self._issue_sparse_ring_bundle(
                raw_tma=raw_tma_compressed,
                stage_base=stage_base,
                cta_rank=cta_rank,
                local_load_warp=local_load_warp,
                completion_mbarrier=completion_mbarrier,
                page_offset_stage=page_offset_stage,
                page_offset_half=page_offset_half,
            )

    @cute.jit
    def _tma_load_sparse(
        self,
        stage_info: StageInfo,
        *,
        k_tile_delta: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Gather one DSV4 CSA V tile through the two-CTA Gather4 path."""

        cfg = self.cfg
        # Work methods are materialized while the task graph is compiled,
        # including methods that the selected schedule will not invoke.  The
        # legacy fallback partitions a K64 descriptor and is not even a
        # well-formed TMA partition for the source BMM2 K128 operand.  Keep
        # its body out of the K128 specialization; that specialization is
        # serviced exclusively by ``tma_load_from_page_ring`` above.
        if cutlass.const_expr(cfg.mma_pv_tiler[2] == 128):
            return
        work_tile = stage_info.work_tile
        k_index = (
            work_tile.k_index_base + Int32(stage_info.loop_offset) + Int32(k_tile_delta)
        )
        route_row = routing_row_index(
            cfg, work_tile.tile_idx, self.logical_seq_len_q, self.cu_seqlens_q
        )
        tiles_per_route = self.page_offsets.shape[0] // Int32(cfg.mma_qk_tiler[1])
        route_tile = route_row * tiles_per_route + k_index
        cta_tiler = (
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas,
            cfg.mma_pv_tiler[2],
        )
        g_swa = cute.zipped_divide(self.sparse_tma_tensor_swa, cta_tiler)
        g_compressed = cute.zipped_divide(self.sparse_tma_tensor_compressed, cta_tiler)
        g_indices = cute.zipped_divide(self.sparse_coord_tensor, cta_tiler)
        t_smem_swa, t_gmem_swa, t_indices_swa = cpasync.tma_partition(
            self.sparse_tma_atom_swa,
            0,
            cute.make_layout(1),
            self.sparse_smem_tensor,
            [g_swa, g_indices],
        )
        t_smem_compressed, t_gmem_compressed, t_indices_compressed = (
            cpasync.tma_partition(
                self.sparse_tma_atom_compressed,
                0,
                cute.make_layout(1),
                self.sparse_smem_tensor,
                [g_compressed, g_indices],
            )
        )
        tma_bar_ptr = cute.make_ptr(
            Int64,
            stage_info.barrier.data_ptr(),
            mem_space=cutlass.AddressSpace.smem,
            assumed_align=8,
        )
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        local_load_warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.load_v_warp_id
        )
        for v_iter in cutlass.range_constexpr(
            cfg.iterations_pv_k * cfg.iterations_pv_n
        ):
            if local_load_warp == Int32(v_iter % cfg.load_v_num_warps):
                pv_n = v_iter // cfg.iterations_pv_k
                pv_k = v_iter % cfg.iterations_pv_k
                gather_tile_idx = route_tile * Int32(cfg.iterations_pv_k) + Int32(pv_k)
                latent_panel_idx = pv_n * cfg.num_mma_ctas + cta_rank
                segment_idx = (
                    stage_info.stage_idx * (cfg.iterations_pv_k * cfg.iterations_pv_n)
                    + pv_k * cfg.iterations_pv_n
                    + pv_n
                )
                if k_index == Int32(0):
                    cute.copy(
                        self.sparse_tma_atom_swa,
                        [
                            t_gmem_swa[
                                None,
                                (latent_panel_idx, gather_tile_idx),
                            ],
                            t_indices_swa[
                                None,
                                (latent_panel_idx, gather_tile_idx),
                            ],
                        ],
                        t_smem_swa[None, 0, 0, segment_idx],
                        tma_bar_ptr=tma_bar_ptr,
                    )
                else:
                    cute.copy(
                        self.sparse_tma_atom_compressed,
                        [
                            t_gmem_compressed[
                                None,
                                (latent_panel_idx, gather_tile_idx),
                            ],
                            t_indices_compressed[
                                None,
                                (latent_panel_idx, gather_tile_idx),
                            ],
                        ],
                        t_smem_compressed[None, 0, 0, segment_idx],
                        tma_bar_ptr=tma_bar_ptr,
                    )

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc(self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]):
        """Build the V SMEM descriptor for the current PV sub-MMA."""
        cfg = self.cfg
        svc_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        subtile_offset = (
            stage_info.stage_idx * cfg.smem_v_stage_elems
            + v_subtile_idx * svc_copy_elems
        )
        if cutlass.const_expr(cfg.is_dynamic_token_sparse):
            # The generated DSV4 BMM2 descriptor is
            # ``leadingDimInBytes=0, strideInBytes=1024, swizzleMode=S128B``.
            # Raw Gather4 has already placed both H256 pieces at the S128B
            # locations, so adding a leading dimension makes tcgen05 consume
            # a different V tile.
            leading_byte_offset = cutlass.const_expr(0)
            stride_byte_offset = cutlass.const_expr(
                8 * (cfg.mma_pv_tiler[1] // cfg.num_mma_ctas) * cfg.qkv_dtype_bytes
            )
            layout = cutlass.const_expr(2)  # S128B
        else:
            leading_byte_offset = cutlass.const_expr(
                V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS * cfg.qkv_dtype_bytes
            )
            stride_byte_offset = cutlass.const_expr(
                qkv_major_k_stride_bytes_for(cfg, cfg.mma_pv_tiler[2])
            )
            layout = cutlass.const_expr(
                qk_desc_layout_for_head_dim(cfg, cfg.mma_pv_tiler[2])
            )
        svc_ptr = self.smem_v.data_ptr(subtile_offset)
        desc = Int64(
            prims.Tcgen05SmemDesc.build(
                start_address=svc_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc_n_major(
        self,
        stage_info: StageInfo,
        *,
        pv_n_idx: cutlass.Constexpr[int],
        pv_k_idx: cutlass.Constexpr[int],
    ):
        """Build the V descriptor when PV commits one output-N slice at a time."""
        cfg = self.cfg
        pv_j = cutlass.const_expr(pv_n_idx)
        pv_i = cutlass.const_expr(pv_k_idx)
        v_call = pv_i * cfg.iterations_pv_n + pv_j
        svc_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        subtile_offset = stage_info.stage_idx * cfg.smem_v_stage_elems + (
            v_call * svc_copy_elems
        )
        if cutlass.const_expr(cfg.is_dynamic_token_sparse):
            # Keep the N-major PV path byte-for-byte compatible with the
            # source BMM2 descriptor (leading=0, stride=1024, S128B).
            leading_byte_offset = cutlass.const_expr(0)
            stride_byte_offset = cutlass.const_expr(
                8 * (cfg.mma_pv_tiler[1] // cfg.num_mma_ctas) * cfg.qkv_dtype_bytes
            )
            layout = cutlass.const_expr(2)  # S128B
        else:
            leading_byte_offset = cutlass.const_expr(
                V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS * cfg.qkv_dtype_bytes
            )
            stride_byte_offset = cutlass.const_expr(
                qkv_major_k_stride_bytes_for(cfg, cfg.mma_pv_tiler[2])
            )
            layout = cutlass.const_expr(
                qk_desc_layout_for_head_dim(cfg, cfg.mma_pv_tiler[2])
            )
        svc_ptr = self.smem_v.data_ptr(subtile_offset)
        desc = Int64(
            prims.Tcgen05SmemDesc.build(
                start_address=svc_ptr.toint(Int32),
                leading_byte_offset=leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=layout,
            )
        )
        return desc


# =====================================================================
# TmemSResource — S scores in TMEM, UmmaProducerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemSResource(HighThroughputMlaResource):
    """TMEM S scores.  Producer: MmaTask (QK MMA).  Consumer: SoftmaxTask.

    Pipeline: UmmaProducerAsync, 2 stages.
    Producer call_idx 0..iterations_qk-1: issue QK MMA for each K sub-tile.
    Consumer: load S, apply mask, compute softmax, and stage P for PV MMA.
    """

    tmem_base_addr: Any = None  # TMEM base address (from alloc)
    smem_q_latent: Any = None  # SMEM Q pointers for descriptor building
    smem_q_rope: Any = None
    smem_p: Any = None  # SMEM P array
    smem_exchange: Any = None  # SMEM array for cross-warp max exchange
    softmax_scale_log2: Any = None  # softmax_scale * log2(e)
    adjusted_skip_corr_threshold: Any = None  # threshold / softmax_scale_log2
    log2_softmax_p_scale: Any = None  # log2(1.75) when skip correction is enabled
    cache_seqs: Any = None  # per-batch valid K length
    cu_seqlens_q: Any = None  # cumulative compact-Q offsets, or None for fixed Q
    split_kv: Any = None  # per-work-tile split count
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    tiled_mma_qk: Any = None
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)
    row_max_state: Any = field(init=False, default=None)
    row_sum_state: Any = field(init=False, default=None)
    qk_acc_regs: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_sum: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_sum_out: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max_new: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    correction_factor_out: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    no_correction_out: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    qk_acc_regs_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_sum_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_sum_out_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max_new_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    correction_factor_out_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    no_correction_out_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("qk_acc_regs", cutlass.Array, None, "Registers holding softmax P values."),
        ("row_max", Float32, Float32(-Float32.inf), "Running row maximum."),
        ("row_sum", Float32, Float32(0), "Running row sum."),
        ("row_sum_out", Float32, Float32(0), "Row sum published to correction."),
        ("row_max_new", Float32, Float32(0), "Updated row maximum."),
        (
            "correction_factor_out",
            Float32,
            Float32(0),
            "Correction factor for the previous O tile.",
        ),
        (
            "no_correction_out",
            Int32,
            Int32(0),
            "Whether O correction may be skipped.",
        ),
        (
            "qk_acc_regs_odd",
            cutlass.Array,
            None,
            "Odd-lane registers holding softmax P values.",
        ),
        ("row_max_odd", Float32, Float32(-Float32.inf), "Odd-lane row maximum."),
        ("row_sum_odd", Float32, Float32(0), "Odd-lane row sum."),
        (
            "row_sum_out_odd",
            Float32,
            Float32(0),
            "Odd-lane row sum published to correction.",
        ),
        ("row_max_new_odd", Float32, Float32(0), "Odd-lane updated row maximum."),
        (
            "correction_factor_out_odd",
            Float32,
            Float32(0),
            "Odd-lane correction factor for the previous O tile.",
        ),
        (
            "no_correction_out_odd",
            Int32,
            Int32(0),
            "Whether odd-lane O correction may be skipped.",
        ),
    )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ),
    )
    @cute.jit
    def init_softmax_state(self, stage_info: StageInfo):
        """Create softmax accumulator registers and row-stat state."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0
        initial_row_max = Float32(-Float32.inf)
        if cutlass.const_expr(self.cfg.dsv4_use_source_softmax_pipeline):
            initial_row_max = Float32(NEG_FLT_MAX)
        self.row_max_state = initial_row_max
        self.row_sum_state = Float32(0)
        return (
            cutlass.Array(
                Float32,
                64,
                space=cutlass.AddressSpace.rmem,
            ),
            initial_row_max,
            Float32(0),
            Float32(0),
            Float32(0),
            Float32(0),
            Int32(0),
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs_odd,
            row_max_odd,
            row_sum_odd,
            row_sum_out_odd,
            row_max_new_odd,
            correction_factor_out_odd,
            no_correction_out_odd,
        ),
    )
    @cute.jit
    def init_softmax_state_odd(self, stage_info: StageInfo):
        """Create odd-lane softmax accumulator registers and row-stat state."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0
        initial_row_max = Float32(-Float32.inf)
        if cutlass.const_expr(self.cfg.dsv4_use_source_softmax_pipeline):
            initial_row_max = Float32(NEG_FLT_MAX)
        self.row_max_state = initial_row_max
        self.row_sum_state = Float32(0)
        return (
            cutlass.Array(
                Float32,
                64,
                space=cutlass.AddressSpace.rmem,
            ),
            initial_row_max,
            Float32(0),
            Float32(0),
            Float32(0),
            Float32(0),
            Int32(0),
        )

    @producer_work
    @cute.jit
    def qk_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_k_base,
        k_subtile_idx: cutlass.Constexpr[int],
    ) -> None:
        """Issue one QK MMA sub-tile (K latent or K rope).
        Only leader CTA issues MMA (2CTA UMMA principle).

        Descriptor computation is hoisted OUTSIDE the leader-CTA gate so
        ptxas keeps values in uniform registers.
        Only the actual tcgen05_mma is gated by elect_sync + leader check.
        """
        cfg = self.cfg
        call_idx = k_subtile_idx

        # Hoist descriptor computation outside leader-CTA gate to preserve
        # uniform register allocation (avoids R2UR demote/promote).
        tmem_s_addr = self.tmem_base_addr + 64 * stage_info.stage_idx
        tmem_s_ptr = prims.make_tmem_ptr(tmem_s_addr, Float32)

        idesc_qk = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            n_dim=cfg.mma_qk_tiler[1],
            m_dim=cfg.mma_qk_tiler[0],
        )
        mma_kind = mma_kind_for_qkv(cfg)
        cta_group = prims.CTAGroup.CTA_2
        k_block_count = cutlass.const_expr(
            ceil_div(cfg.mma_qk_tiler_k, mma_k_step_for_qkv(cfg))
        )
        is_leader_cta = (
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
        )

        desc_k = desc_k_base

        if cutlass.const_expr(call_idx < cfg.iterations_qk_latent_stages):
            qc_copy_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[0] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
            )
            kc_copy_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[1] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
            )
            desc_k_delta = cutlass.const_expr(kc_copy_elems * cfg.qkv_dtype_bytes // 16)
            for stage_subtile_idx in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
                logical_call_idx = (
                    call_idx * cfg.kv_subtiles_per_stage + stage_subtile_idx
                )
                q_smem_ptr = self.smem_q_latent.data_ptr(
                    logical_call_idx * qc_copy_elems
                )
                desc_q = Int64(
                    prims.Tcgen05SmemDesc.build(
                        start_address=q_smem_ptr.toint(Int32),
                        leading_byte_offset=qk_desc_leading_byte_offset(cfg),
                        stride_byte_offset=qk_desc_stride_byte_offset(cfg),
                        layout=qk_desc_layout(cfg),
                    )
                )
                desc_k_subtile = desc_k + stage_subtile_idx * desc_k_delta
                if is_leader_cta:
                    for k_block in cutlass.range_constexpr(k_block_count):
                        scale_d = cutlass.const_expr(
                            logical_call_idx > 0 or k_block > 0
                        )
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                mma_kind,
                                cta_group,
                                tmem_s_ptr,
                                desc_q + k_block * 2,
                                desc_k_subtile + k_block * 2,
                                idesc_qk,
                                Boolean(scale_d),
                            )

        elif cutlass.const_expr(call_idx < cfg.iterations_qk_stages):
            # K rope: build Q rope descriptor unconditionally
            rope_idx = call_idx - cfg.iterations_qk_latent_stages
            qc_copy_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[0] // cfg.num_mma_ctas * cfg.mma_qk_rope_tiler[2]
            )
            q_rope_smem_ptr = self.smem_q_rope.data_ptr(rope_idx * qc_copy_elems)
            q_rope_rows = cutlass.const_expr(cfg.mma_qk_tiler[0] // cfg.num_mma_ctas)
            q_rope_dim = cutlass.const_expr(cfg.mma_qk_rope_tiler[2])
            desc_qr = Int64(
                prims.Tcgen05SmemDesc.build(
                    start_address=q_rope_smem_ptr.toint(Int32),
                    leading_byte_offset=qk_desc_leading_byte_offset_for_head_dim(
                        cfg, q_rope_rows, q_rope_dim
                    ),
                    stride_byte_offset=qk_desc_stride_byte_offset_for_head_dim(
                        cfg, q_rope_dim
                    ),
                    layout=qk_desc_layout_for_head_dim(cfg, q_rope_dim),
                )
            )
            if is_leader_cta:
                for k_block in cutlass.range_constexpr(
                    cfg.rope_dim // mma_k_step_for_qkv(cfg)
                ):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            mma_kind,
                            cta_group,
                            tmem_s_ptr,
                            desc_qr + k_block * 2,
                            desc_k + k_block * 2,
                            idesc_qk,
                            Boolean(True),
                        )

    @cute.jit
    def source_preacquire_pair(self) -> None:
        """Acquire S[0] and S[1] once before source W8's persistent loop.

        Source uses one syntactic state for this lookahead and for commits,
        but the state is observed at two different logical times.  Prim-TS
        represents that contract explicitly: ``producer_state`` is the
        acquire cursor and ``producer_commit_state`` is the lagging QK commit
        cursor.  Opening both empty stages here leaves acquire exactly two
        tokens ahead without making a cloned state escape captured control
        flow.
        """
        assert self.pipeline is not None
        assert self.pipeline_config is not None
        assert self.pipeline_config.advance_on_acquire
        for _ in cutlass.range_constexpr(2):
            token = self.pipeline.producer_try_acquire(self.producer_state)
            self.pipeline.producer_acquire(self.producer_state, token)
            self.producer_state.advance()

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def source_acquire_next_stage(self, stage_info: StageInfo) -> None:
        """Wait for ``producer_state + 1`` without moving the owned cursor.

        W8 uses this shifted tail acquire to observe the final softmax P store
        before PV consumes it.  It corresponds to generated
        ``makePipelineState(tmemS0ProdState, 1)``.
        """
        del stage_info
        assert self.pipeline is not None
        next_state = self.producer_state.clone()
        next_state.advance()
        next_token = self.pipeline.producer_try_acquire(next_state)
        self.pipeline.producer_acquire(next_state, next_token)

    @cute.jit
    def source_balance_after_persistent_loop(self) -> None:
        """Emit source W8's two final S commits after all persistent tiles."""
        assert self.pipeline is not None
        assert self.pipeline_config is not None
        assert self.pipeline_config.advance_on_acquire
        self.pipeline.producer_commit(self.producer_commit_state)
        self.producer_commit_state.advance()
        self.pipeline.producer_commit(self.producer_commit_state)
        self.producer_commit_state.advance()

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=(row_sum, row_sum_out))
    @cute.jit
    def finish_row_sum(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_sum,
        correction_factor_out,
    ):
        """Finish row-sum reduction after P is published to SMEM."""
        del stage_info
        cfg = self.cfg
        if cutlass.const_expr(
            cfg.dsv4_use_source_local_ptx_knobs and cfg.dsv4_use_source_direct_p
        ):
            # The generated task places this compiler fence and a second
            # async-shared proxy fence after S release and before the global
            # softmax sum.  TmemP's first view fence remains attached to the
            # packed P store itself.
            dsv4_code_fence()
            prims.fence_proxy(
                kind=prims.Proxy.ASYNC_SHARED,
                space=prims.SharedSpace.shared_cta,
            )
        if cutlass.const_expr(cfg.dsv4_use_source_softmax_pipeline):
            sum0 = (Float32(0), Float32(0))
            sum1 = (Float32(0), Float32(0))
            sum2 = (Float32(0), Float32(0))
            sum3 = (Float32(0), Float32(0))
            for i in cutlass.range_constexpr(0, 64, 8):
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                sum0 = fadd2(sum0, (qk_acc_regs[i], qk_acc_regs[i + 1]))
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                sum1 = fadd2(sum1, (qk_acc_regs[i + 2], qk_acc_regs[i + 3]))
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                sum2 = fadd2(sum2, (qk_acc_regs[i + 4], qk_acc_regs[i + 5]))
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                sum3 = fadd2(sum3, (qk_acc_regs[i + 6], qk_acc_regs[i + 7]))
            sum0 = fadd2(sum0, sum1)
            sum2 = fadd2(sum2, sum3)
            sum0 = fadd2(sum0, sum2)
            local_sum = add_ftz_f32(sum0[0], sum0[1])
            # The structural P path carries oldMax-newMax in this slot so
            # expScale remains after the local reduction, exactly as in
            # TmemSoftmaxGlobal rather than being hoisted ahead of MUFU P.
            correction_factor = Float32(1)
            if cutlass.const_expr(cfg.dsv4_enable_skip_correction):
                # Source computeExpScale leaves the identity value in place
                # when max freeze made oldMax == newMax.  Avoiding this MUFU
                # is part of skip correction's performance contract.
                if correction_factor_out != Float32(0):
                    correction_factor = cute.math.exp2(
                        mul_ftz_f32(correction_factor_out, self.softmax_scale_log2),
                        fastmath=True,
                    )
            else:
                correction_factor = cute.math.exp2(
                    mul_ftz_f32(correction_factor_out, self.softmax_scale_log2),
                    fastmath=True,
                )
            row_sum = fma_ftz_f32(correction_factor, row_sum, local_sum)
        else:
            row_sum = row_sum * correction_factor_out
            row_sum_vec = (Float32(0), Float32(0))
            for i in cutlass.range_constexpr(0, 64, 2):
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                row_sum_vec = add_packed_f32x2(
                    row_sum_vec,
                    (qk_acc_regs[i], qk_acc_regs[i + 1]),
                )
            row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum
        self.row_sum_state = row_sum
        return row_sum, row_sum

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY, returns=(row_sum_odd, row_sum_out_odd)
    )
    @cute.jit
    def finish_row_sum_odd(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs_odd,
        row_sum_odd,
        correction_factor_out_odd,
    ):
        """Finish odd-lane row-sum reduction after P is published to SMEM."""
        del stage_info
        cfg = self.cfg
        if cutlass.const_expr(
            cfg.dsv4_use_source_local_ptx_knobs and cfg.dsv4_use_source_direct_p
        ):
            dsv4_code_fence()
            prims.fence_proxy(
                kind=prims.Proxy.ASYNC_SHARED,
                space=prims.SharedSpace.shared_cta,
            )
        if cutlass.const_expr(cfg.dsv4_use_source_softmax_pipeline):
            sum0 = (Float32(0), Float32(0))
            sum1 = (Float32(0), Float32(0))
            sum2 = (Float32(0), Float32(0))
            sum3 = (Float32(0), Float32(0))
            for i in cutlass.range_constexpr(0, 64, 8):
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                sum0 = fadd2(sum0, (qk_acc_regs_odd[i], qk_acc_regs_odd[i + 1]))
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                sum1 = fadd2(sum1, (qk_acc_regs_odd[i + 2], qk_acc_regs_odd[i + 3]))
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                sum2 = fadd2(sum2, (qk_acc_regs_odd[i + 4], qk_acc_regs_odd[i + 5]))
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                sum3 = fadd2(sum3, (qk_acc_regs_odd[i + 6], qk_acc_regs_odd[i + 7]))
            sum0 = fadd2(sum0, sum1)
            sum2 = fadd2(sum2, sum3)
            sum0 = fadd2(sum0, sum2)
            local_sum = add_ftz_f32(sum0[0], sum0[1])
            correction_factor = Float32(1)
            if cutlass.const_expr(cfg.dsv4_enable_skip_correction):
                if correction_factor_out_odd != Float32(0):
                    correction_factor = cute.math.exp2(
                        mul_ftz_f32(
                            correction_factor_out_odd,
                            self.softmax_scale_log2,
                        ),
                        fastmath=True,
                    )
            else:
                correction_factor = cute.math.exp2(
                    mul_ftz_f32(correction_factor_out_odd, self.softmax_scale_log2),
                    fastmath=True,
                )
            row_sum = fma_ftz_f32(correction_factor, row_sum_odd, local_sum)
        else:
            row_sum = row_sum_odd * correction_factor_out_odd
            row_sum_vec = (Float32(0), Float32(0))
            for i in cutlass.range_constexpr(0, 64, 2):
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                row_sum_vec = add_packed_f32x2(
                    row_sum_vec,
                    (qk_acc_regs_odd[i], qk_acc_regs_odd[i + 1]),
                )
            row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum
        self.row_sum_state = row_sum
        return row_sum, row_sum

    @cute.jit
    def _load_s_impl(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ):
        """Load S from TMEM and compute the local row max."""
        del row_sum_out, row_max_new, correction_factor_out, no_correction_out
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        stage_idx = stage_info.stage_idx

        work_tile = stage_info.work_tile
        K = Int32(work_tile.k_len)
        k_index = work_tile.k_index_base + Int32(stage_info.loop_offset)
        tile_offset_k = k_index * Int32(cfg.mma_qk_tiler[1])
        needs_row_causal_mask = cutlass.const_expr(
            cfg.mask_type == MaskType.CAUSAL.value and self.logical_seq_len_q > 1
        )
        if cutlass.const_expr(needs_row_causal_mask):
            batch_idx = Int32(work_tile.tile_idx[2])
            _, logical_seq_len_q = query_batch_bounds(
                self.cu_seqlens_q,
                batch_idx,
                self.logical_seq_len_q,
            )
            first_flat_query_row = Int32(work_tile.tile_idx[1]) * Int32(
                cfg.mma_qk_tiler[0]
            )
            # A row r sees key k iff H * (k - K + SQ) <= r.  Apply the
            # equivalent boundary test to the first row in this physical tile
            # so non-power-of-two H never needs a quotient on the softmax path.
            first_mask_flat_row = Int32(self.logical_num_heads_q) * (
                tile_offset_k
                + Int32(cfg.mma_qk_tiler[1])
                - K
                + logical_seq_len_q
                - Int32(1)
            )
            group_needs_mask = first_flat_query_row < first_mask_flat_row
        else:
            group_needs_mask = kv_tile_needs_right_mask(
                tile_offset_k,
                Int32(cfg.mma_qk_tiler[1]),
                K,
            )
        # DSV4's first sparse tile owns the fixed SWA slot range.  It is
        # independently bounded by the causal raw timeline even if Lq is a
        # multiple of 128, so it always needs the right-mask path.
        if cutlass.const_expr(cfg.is_dynamic_token_sparse):
            group_needs_mask = group_needs_mask | (k_index == Int32(0))

        neg_inf = Float32(-Float32.inf)
        if cutlass.const_expr(cfg.dsv4_use_source_softmax_pipeline):
            neg_inf = Float32(NEG_FLT_MAX)
        row_max_tile = row_max
        warp_id = local_tidx >> 5
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        stage_offset = stage_idx * 64
        tmem_raw_addr = (tmem_warp_row_id << 16) | stage_offset
        t2r_shape = TCGEN05_32B_SHAPE
        for load_idx in cutlass.range_constexpr(2):
            curr_addr = tmem_raw_addr + load_idx * TCGEN05_32B_REGS_PER_LOAD
            tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
            loaded = prims.tcgen05_ld(
                t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
            )
            qk_acc_regs.store(loaded, load_idx * TCGEN05_32B_REGS_PER_LOAD)

        if group_needs_mask:
            if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                dsv4_set_cold_block()
            row_k_len = K
            if cutlass.const_expr(cfg.is_dynamic_token_sparse):
                if k_index == Int32(0):
                    batch_idx = Int32(work_tile.tile_idx[2])
                    _, logical_seq_len_q = query_batch_bounds(
                        self.cu_seqlens_q,
                        batch_idx,
                        self.logical_seq_len_q,
                    )
                    _, _, logical_q_idx, _, _ = flat_query_row_state(
                        Int32(self.logical_num_heads_q - 1),
                        work_tile.tile_idx[1],
                        cfg.mma_qk_tiler[0],
                        self.logical_num_heads_q,
                        self.logical_seq_len_q,
                        self.cu_seqlens_q,
                        batch_idx,
                    )
                    raw_visible = mask_visible_k_length(
                        MaskType.CAUSAL.value,
                        self.cache_seqs[batch_idx],
                        logical_q_idx,
                        logical_seq_len_q,
                    )
                    row_k_len = cute.math.min(
                        cute.math.min(raw_visible, Int32(cfg.sparse_swa_topk)), K
                    )
            elif cutlass.const_expr(needs_row_causal_mask):
                # Clamp padded physical tail rows to the request's last real
                # row. Their Q/output accesses remain independently
                # predicated, while this keeps the masking arithmetic safe.
                row_in_tile = self.cta_rank * Int32(
                    cfg.mma_qk_tiler[0] // cfg.num_mma_ctas
                ) + (local_tidx & Int32(EPILOGUE_ROW_MASK))
                flat_query_row = (
                    Int32(work_tile.tile_idx[1]) * Int32(cfg.mma_qk_tiler[0])
                    + row_in_tile
                )
                last_flat_query_row = cute.math.max(
                    logical_seq_len_q * Int32(self.logical_num_heads_q) - Int32(1),
                    Int32(0),
                )
                flat_query_row = cute.math.min(
                    flat_query_row,
                    last_flat_query_row,
                )
            tidx_col = (
                local_tidx >> EPILOGUE_COLUMN_GROUP_SHIFT
            ) << EPILOGUE_COLUMN_GROUP_SHIFT
            for i in cutlass.range_constexpr(64):
                token_idx = tile_offset_k + tidx_col + Int32(i)
                if cutlass.const_expr(cfg.is_dynamic_token_sparse):
                    token_is_visible = token_idx < row_k_len
                elif cutlass.const_expr(needs_row_causal_mask):
                    mask_flat_row = Int32(self.logical_num_heads_q) * (
                        token_idx - K + logical_seq_len_q
                    )
                    token_is_visible = flat_query_row >= mask_flat_row
                else:
                    token_is_visible = token_idx < K
                qk_acc_regs[i] = qk_acc_regs[i] if token_is_visible else neg_inf
            if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                dsv4_reset_cold_block()

        max0 = row_max_tile
        max1 = row_max_tile
        max2 = row_max_tile
        max3 = row_max_tile
        if cutlass.const_expr(cfg.dsv4_use_source_softmax_pipeline):
            # TmemS consumes sequential groups of four into independent ILP
            # chains.  The older TS path used four strided 16-element chains;
            # both compute the same max, but expose different dependencies to
            # ptxas and therefore need a structural A/B switch.
            for i in cutlass.range_constexpr(0, 64, 4):
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                max0 = fmax_f32(max0, qk_acc_regs[i])
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                max1 = fmax_f32(max1, qk_acc_regs[i + 1])
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                max2 = fmax_f32(max2, qk_acc_regs[i + 2])
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                max3 = fmax_f32(max3, qk_acc_regs[i + 3])
            if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                dsv4_warp_switch()
            max0 = fmax_f32(max0, max2)
            if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                dsv4_warp_switch()
            max1 = fmax_f32(max1, max3)
            if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                dsv4_warp_switch()
            row_max_tile = fmax_f32(max0, max1)
        else:
            for i in cutlass.range_constexpr(16):
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                max0 = fmax_f32(max0, qk_acc_regs[i])
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                max1 = fmax_f32(max1, qk_acc_regs[i + 16])
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                max2 = fmax_f32(max2, qk_acc_regs[i + 32])
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                max3 = fmax_f32(max3, qk_acc_regs[i + 48])
            if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                dsv4_warp_switch()
                max01 = fmax_f32(max0, max1)
                dsv4_warp_switch()
                max23 = fmax_f32(max2, max3)
                dsv4_warp_switch()
                row_max_tile = fmax_f32(max01, max23)
            else:
                row_max_tile = fmax_f32(fmax_f32(max0, max1), fmax_f32(max2, max3))
        cute.arch.fence_view_async_tmem_load()
        return (
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum,
            row_max_tile,
            Float32(1),
            Int32(1),
        )

    @consumer_work(
        returns=(
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ),
    )
    @cute.jit
    def load_s(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ):
        """Load S for the even softmax group."""
        return self._load_s_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs,
            row_max=row_max,
            row_sum=row_sum,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
        )

    @consumer_work(
        returns=(
            qk_acc_regs_odd,
            row_max_odd,
            row_sum_odd,
            row_sum_out_odd,
            row_max_new_odd,
            correction_factor_out_odd,
            no_correction_out_odd,
        ),
    )
    @cute.jit
    def load_s_odd(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs_odd,
        row_max_odd,
        row_sum_odd,
        row_sum_out_odd,
        row_max_new_odd,
        correction_factor_out_odd,
        no_correction_out_odd,
    ):
        """Load S for the odd softmax group."""
        return self._load_s_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs_odd,
            row_max=row_max_odd,
            row_sum=row_sum_odd,
            row_sum_out=row_sum_out_odd,
            row_max_new=row_max_new_odd,
            correction_factor_out=correction_factor_out_odd,
            no_correction_out=no_correction_out_odd,
        )

    @cute.jit
    def _finish_softmax_max_impl(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
        softmax_group_id: cutlass.Constexpr[int] = 0,
    ):
        """Finish row-max exchange without materializing P.

        Keeping this phase separate lets the DSV4 source schedule publish
        ``(old_max, new_max)`` to Correction before the comparatively long P
        exponentiation/store path.  The regular schedule immediately calls
        :meth:`_materialize_softmax_p_impl`, preserving its prior behavior.
        """
        del row_sum_out, correction_factor_out, no_correction_out
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        row_max_prev = row_max
        row_sum_prev = row_sum
        row_max_tile = row_max_new

        if cutlass.const_expr(cfg.dsv4_use_source_softmax_pair_reduction):
            # Match source reduceWarpGrp2x2 exactly.  The M128 cta_group_2
            # result is laid out as a 2x2 warp group, so W0 exchanges with W2
            # and W1 with W3.  Each S pipeline stage owns a disjoint 128-float
            # buffer.  Passing the barrier slot as a runtime Int32 is
            # intentional: barrier.cta.sync accepts a register barrier ID.
            stage_exchange_base = Int32(stage_info.stage_idx) * Int32(
                num_compute_threads
            )
            warp_idx_local = local_tidx >> 5
            row_idx = local_tidx % Int32(64)
            col_idx = local_tidx >> 6
            exchange_idx = stage_exchange_base + row_idx * Int32(2) + col_idx
            peer_idx = stage_exchange_base + row_idx * Int32(2) + (col_idx ^ Int32(1))
            self.smem_exchange[exchange_idx] = row_max_tile
            pair_barrier_id = Int32(cfg.softmax_sync_bar_id) + (
                warp_idx_local & Int32(1)
            )
            prims.barrier_cta_sync(
                pair_barrier_id,
                thread_count=cfg.softmax_sync_threads,
            )
            row_max_tile = fmax_f32(row_max_tile, self.smem_exchange[peer_idx])
        else:
            group_exchange_base = Int32(softmax_group_id * num_compute_threads)
            self.smem_exchange[group_exchange_base + local_tidx] = row_max_tile
            prims.barrier_cta_sync(
                cfg.softmax_sync_bar_id + softmax_group_id,
                thread_count=cfg.softmax_sync_threads,
            )
            peer_idx = (local_tidx + 64) % num_compute_threads
            row_max_tile = fmax_f32(
                row_max_tile, self.smem_exchange[group_exchange_base + peer_idx]
            )
            # The generic path uses one exchange buffer on consecutive KV
            # iterations, so keep every peer read ahead of the next write.
            prims.barrier_cta_sync(
                cfg.softmax_sync_bar_id + softmax_group_id,
                thread_count=cfg.softmax_sync_threads,
            )

        if cutlass.const_expr(cfg.use_fp8_dual_softmax_schedule):
            stage_idx = stage_info.stage_idx

            def load_peer_state():
                """Load the peer softmax group's correction state from TMEM."""

                peer_stage_idx = (stage_idx + Int32(1)) % Int32(cfg.p_cor_stage)
                corr_col_offset = cfg.correction_factor_offset + peer_stage_idx * 4
                peer_warp_id = local_tidx >> 5
                peer_tmem_row = (
                    self.tmem_base_addr + peer_warp_id * TCGEN05_32B_REGS_PER_LOAD
                )
                peer_tmem_addr = (peer_tmem_row << 16) | corr_col_offset
                peer_tmem_ptr = prims.make_tmem_ptr(peer_tmem_addr, Float32)
                peer_corr = prims.tcgen05_ld(TCGEN05_32B_SHAPE, peer_tmem_ptr, num=2)
                return peer_corr[1], peer_corr[0]

            if cutlass.const_expr(softmax_group_id == 1):
                if stage_info.loop_offset != Int32(0):
                    prims.barrier_cta_sync(
                        cfg.softmax_order_bar_1_id,
                        thread_count=2 * num_compute_threads,
                    )
                    cute.arch.fence_acq_rel_cta()
                    row_max_prev, row_sum_prev = load_peer_state()
            else:
                if stage_info.loop_offset != Int32(0):
                    prims.barrier_cta_sync(
                        cfg.softmax_order_bar_0_id,
                        thread_count=2 * num_compute_threads,
                    )
                    cute.arch.fence_acq_rel_cta()
                    row_max_prev, row_sum_prev = load_peer_state()

        row_max_new = fmax_f32(row_max_prev, row_max_tile)
        if cutlass.const_expr(cfg.dsv4_enable_skip_correction):
            # TRTLLM-gen pre-divides the user threshold by scaleSoftmaxLog2,
            # then freezes a candidate row max whose raw-score increase fits
            # inside that bound.  P can consequently grow by at most 2^T and
            # the previous O accumulation needs no rescale for this row.
            if self.adjusted_skip_corr_threshold > Float32(0):
                max_increase = sub_ftz_f32(row_max_new, row_max_prev)
                if max_increase <= self.adjusted_skip_corr_threshold:
                    row_max_new = row_max_prev
        return (
            qk_acc_regs,
            row_max_prev,
            row_sum_prev,
            row_sum_prev,
            row_max_new,
            Float32(0),
            Int32(0),
        )

    @cute.jit
    def _store_source_packed_p(self, stage_info: StageInfo, packed_p) -> None:
        """Store the 16 delayed E4M3x4 registers in source's P layout."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        stage_idx = Int32(stage_info.loop_offset) & Int32(1)
        stage_stride_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0]
            // cfg.num_mma_ctas
            * cfg.mma_pv_tiler[2]
            * cfg.iterations_pv_k
        )
        smem_p_base_bytes = (
            self.smem_p.data_ptr().toint(Int32)
            + stage_idx * stage_stride_elems * cfg.qkv_dtype_bytes
        )
        row = local_tidx % Int32(64)
        warp_col = local_tidx // Int32(64)
        xor_col = local_tidx % Int32(8)
        for store_i in cutlass.range_constexpr(4):
            smem_col = Int32(store_i) + warp_col * Int32(4)
            swizzled_col = smem_col ^ xor_col
            dst_addr = smem_p_base_bytes + row * Int32(128) + swizzled_col * Int32(16)
            smem_ptr = cutlass.inttoptr(dst_addr, 3, Int32)
            smem_ptr.store(packed_p.load(store_i * 4, 4), alignment=16)
        # This is TmemP's first fence_view_async_shared.  The generated task
        # has a second cfence + async-shared proxy fence only after S release,
        # immediately before the global row-sum reduction.
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _materialize_softmax_p_impl(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ):
        """Apply online correction and exponentiate the current P tile."""
        del row_sum_out, correction_factor_out, no_correction_out
        cfg = self.cfg
        if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
            dsv4_sched_res_busy_xu64()
        neg_inf = Float32(-Float32.inf)
        if cutlass.const_expr(cfg.dsv4_use_source_softmax_pipeline):
            neg_inf = Float32(NEG_FLT_MAX)
        row_max_prev = row_max
        row_sum_prev = row_sum
        row_has_values = row_max_new != neg_inf
        safe_row_max_prev = row_max_prev if row_has_values else Float32(0)
        safe_row_max_new = row_max_new if row_has_values else Float32(0)
        # Exact max equality makes the correction scale exactly one. Keep that
        # lane on the identity value and avoid issuing exp2 altogether.
        max_changed = safe_row_max_prev != safe_row_max_new
        max_diff = (
            sub_ftz_f32(safe_row_max_prev, safe_row_max_new)
            if cutlass.const_expr(cfg.dsv4_use_source_softmax_pipeline)
            else safe_row_max_prev - safe_row_max_new
        )
        correction_factor = Float32(1)
        if cutlass.const_expr(cfg.dsv4_use_source_softmax_pipeline):
            # TmemSoftmaxGlobal evaluates exp2 only after its four-way local
            # sum.  Carry max_diff through the existing task-local slot so
            # the structural diagnostic preserves that program order.
            correction_factor = max_diff
        elif max_changed:
            correction_factor = cute.math.exp2(
                max_diff * self.softmax_scale_log2,
                fastmath=True,
            )
        no_correction = Int32(not max_changed)

        fma_b = self.softmax_scale_log2
        fma_bias = Float32(0)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            fma_bias = (
                self.log2_softmax_p_scale
                if cutlass.const_expr(cfg.is_dynamic_token_sparse)
                else fp8_log2_quant_scale()
            )
        if cutlass.const_expr(cfg.dsv4_use_source_softmax_pipeline):
            # Source --use_fast_math contracts ``-max * scale + bias`` into
            # one FFMA.FTZ before the scalar-prefetched first P pair.
            fma_c = fnma_ftz_f32(safe_row_max_new, fma_b, fma_bias)
        else:
            fma_c = Float32(0) - safe_row_max_new * fma_b + fma_bias
        if cutlass.const_expr(cfg.dsv4_use_source_softmax_pipeline):
            packed_p = cutlass.Array(
                Int32,
                16,
                space=cutlass.AddressSpace.rmem,
            )
            # mNumPrefetchedFmas=4.  Source deliberately spells the first
            # pair as scalar FMA to avoid later register-move lowering.  The
            # generated C++ spells every subsequent pair as FMUL2+FADD2, but
            # source ptxas contracts those 31 pairs to FFMA2 in final SASS.
            qk_acc_regs[0] = fma_ftz_f32(fma_b, qk_acc_regs[0], fma_c)
            qk_acc_regs[1] = fma_ftz_f32(fma_b, qk_acc_regs[1], fma_c)
            if cutlass.const_expr(cfg.dsv4_use_source_contractible_ffma):
                fma_result = affine2_contractible_f32(
                    fma_b,
                    fma_b,
                    qk_acc_regs[2],
                    qk_acc_regs[3],
                    fma_c,
                    fma_c,
                )
            else:
                fma_result = ffma2(
                    (fma_b, fma_b),
                    (qk_acc_regs[2], qk_acc_regs[3]),
                    (fma_c, fma_c),
                )
            qk_acc_regs[2] = fma_result[0]
            qk_acc_regs[3] = fma_result[1]

            for i in cutlass.range_constexpr(0, 64, 2):
                # mNumDelayedCvtElts=12. Convert one E4M3x4 register before
                # every other even MUFU after the twelve-element latency
                # window; the final three conversions remain post-loop.
                if cutlass.const_expr(i >= 12 and i % 4 == 0):
                    cvt_offset = i - 12
                    packed_p[cvt_offset // 4] = pack_float4_to_fp8_e4m3(
                        qk_acc_regs[cvt_offset],
                        qk_acc_regs[cvt_offset + 1],
                        qk_acc_regs[cvt_offset + 2],
                        qk_acc_regs[cvt_offset + 3],
                    )
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_code_fence()
                qk_acc_regs[i] = cute.math.exp2(qk_acc_regs[i], fastmath=True)
                if cutlass.const_expr(i + 4 < 64):
                    fma_offset = i + 4
                    if cutlass.const_expr(cfg.dsv4_use_source_contractible_ffma):
                        fma_result = affine2_contractible_f32(
                            fma_b,
                            fma_b,
                            qk_acc_regs[fma_offset],
                            qk_acc_regs[fma_offset + 1],
                            fma_c,
                            fma_c,
                        )
                    else:
                        fma_result = ffma2(
                            (fma_b, fma_b),
                            (
                                qk_acc_regs[fma_offset],
                                qk_acc_regs[fma_offset + 1],
                            ),
                            (fma_c, fma_c),
                        )
                    qk_acc_regs[fma_offset] = fma_result[0]
                    qk_acc_regs[fma_offset + 1] = fma_result[1]
                qk_acc_regs[i + 1] = cute.math.exp2(qk_acc_regs[i + 1], fastmath=True)
            if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                dsv4_code_fence()
            for cvt_offset in cutlass.range_constexpr(52, 64, 4):
                packed_p[cvt_offset // 4] = pack_float4_to_fp8_e4m3(
                    qk_acc_regs[cvt_offset],
                    qk_acc_regs[cvt_offset + 1],
                    qk_acc_regs[cvt_offset + 2],
                    qk_acc_regs[cvt_offset + 3],
                )
            self._store_source_packed_p(stage_info, packed_p)
        else:
            for i in cutlass.range_constexpr(0, 64, 2):
                fma_result = fma_packed_f32x2(
                    (qk_acc_regs[i], qk_acc_regs[i + 1]),
                    (fma_b, fma_b),
                    (fma_c, fma_c),
                )
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    # Source places FenceCode before each even MUFU.  This
                    # pragma-only control leaves the old FMA/MUFU order intact.
                    dsv4_code_fence()
                qk_acc_regs[i] = cute.math.exp2(fma_result[0], fastmath=True)
                qk_acc_regs[i + 1] = cute.math.exp2(fma_result[1], fastmath=True)
            if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                dsv4_code_fence()

        self.row_max_state = row_max_new
        return (
            qk_acc_regs,
            row_max_new,
            row_sum_prev,
            row_sum_prev,
            row_max_new,
            correction_factor,
            no_correction,
        )

    @cute.jit
    def _finish_softmax_impl(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
        softmax_group_id: cutlass.Constexpr[int] = 0,
    ):
        """Finish row max and immediately materialize P (legacy schedule)."""
        (
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ) = self._finish_softmax_max_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs,
            row_max=row_max,
            row_sum=row_sum,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
            softmax_group_id=softmax_group_id,
        )
        return self._materialize_softmax_p_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs,
            row_max=row_max,
            row_sum=row_sum,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ),
    )
    @cute.jit
    def finish_softmax_max(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ):
        """Publishable row-max phase for source DSV4 early correction."""
        return self._finish_softmax_max_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs,
            row_max=row_max,
            row_sum=row_sum,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
            softmax_group_id=0,
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ),
    )
    @cute.jit
    def materialize_softmax_p(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ):
        """Exponentiate P after the source early-correction commit."""
        return self._materialize_softmax_p_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs,
            row_max=row_max,
            row_sum=row_sum,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ),
    )
    @cute.jit
    def finish_softmax(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs,
        row_max,
        row_sum,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ):
        """Finish softmax for the even group after S release."""
        return self._finish_softmax_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs,
            row_max=row_max,
            row_sum=row_sum,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
            softmax_group_id=0,
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            qk_acc_regs_odd,
            row_max_odd,
            row_sum_odd,
            row_sum_out_odd,
            row_max_new_odd,
            correction_factor_out_odd,
            no_correction_out_odd,
        ),
    )
    @cute.jit
    def finish_softmax_odd(
        self,
        stage_info: StageInfo,
        *,
        qk_acc_regs_odd,
        row_max_odd,
        row_sum_odd,
        row_sum_out_odd,
        row_max_new_odd,
        correction_factor_out_odd,
        no_correction_out_odd,
    ):
        """Finish softmax for the odd group after S release."""
        return self._finish_softmax_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs_odd,
            row_max=row_max_odd,
            row_sum=row_sum_odd,
            row_sum_out=row_sum_out_odd,
            row_max_new=row_max_new_odd,
            correction_factor_out=correction_factor_out_odd,
            no_correction_out=no_correction_out_odd,
            softmax_group_id=1,
        )


# =====================================================================
# SmemPResource — P in SMEM, UmmaConsumerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemPResource(HighThroughputMlaResource):
    """SMEM P buffer.  Producer: SoftmaxTask.  Consumer: MmaTask (PV MMA).

    Pipeline: UmmaConsumerAsync, 2 stages.
    """

    smem_p: Any = None
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    desc_p_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_p_base", Int64, Int64(0), "SMEM descriptor for staged P."),
    )

    @cute.jit
    def _store_p_impl(self, stage_idx, *, qk_acc_regs) -> None:
        """Store softmax P tile to SMEM for PV MMA.

        Converts P values to FP16 and stores to SMEM with correct swizzle layout.
        """
        cfg = self.cfg
        num_mma_ctas = cfg.num_mma_ctas

        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        lane_idx = local_tidx & 31
        warp_idx = local_tidx >> 5

        # Convert F32 -> QKV dtype into a vectorized local buffer.
        qkv_element_dtype = qkv_dtype(cfg)
        s_regs = cutlass.Array(qkv_element_dtype, 64, space=cutlass.AddressSpace.rmem)
        s_regs.store(qk_acc_regs.load(0, 64).to(qkv_element_dtype), 0)

        # Compute SMEM P stage offset
        sp_stage_stride_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0]
            // num_mma_ctas
            * cfg.mma_pv_tiler[2]
            * cfg.iterations_pv_k
        )
        smem_p_base_bytes = (
            self.smem_p.data_ptr().toint(cutlass.Int32)
            + stage_idx * sp_stage_stride_elems * cfg.qkv_dtype_bytes
        )

        if cutlass.const_expr(
            cfg.is_dynamic_token_sparse and cfg.mma_pv_tiler[2] == 128
        ):
            # TRTLLM-gen's ``storeWarpGrp2x2SmemP<1, 16>`` layout.  BMM2
            # owns one K128 P partition; each W0--W3 thread stores four
            # 16-B vectors into a 128-B S128B-swizzled row.
            row = local_tidx % Int32(64)
            warp_col = local_tidx // Int32(64)
            xor_col = local_tidx % Int32(8)
            for store_i in cutlass.range_constexpr(4):
                smem_col = Int32(store_i) + warp_col * Int32(4)
                swizzled_col = smem_col ^ xor_col
                dst_addr = (
                    smem_p_base_bytes + row * Int32(128) + swizzled_col * Int32(16)
                )
                vec = s_regs.load(store_i * 16, 16)
                smem_ptr = cutlass.inttoptr(dst_addr, 3, qkv_element_dtype)
                smem_ptr.store(vec, alignment=16)
        elif cutlass.const_expr(cfg.is_fp8_qkv()):
            # FP8 P layout:
            # S<2,4,3> o ((64,32),1,2,(2,2)):((64,1),0,32,(4096,8192)).
            # Each universal SMEM copy is 128 bits, i.e. 16 E4M3 elements.
            # Lane pairs write opposite halves of the 128B swizzled row.  The
            # signed strides below walk the four 16B blocks inside that row.
            m = ((lane_idx >> 1) & 3) * 128
            base = cutlass.Int32(0)
            base = base + (lane_idx & 1) * 64
            base = base + (lane_idx >> 3) * 512
            base = base + (warp_idx & 1) * 2048
            base = base + (warp_idx >> 1) * 4096
            swizzle_xor = m ^ ((m & 384) >> 3)
            off_base = base + swizzle_xor

            stride_a = cutlass.Int32(16)
            if (m & 128) != 0:
                stride_a = cutlass.Int32(-16)
            stride_b = cutlass.Int32(32)
            if (m & 256) != 0:
                stride_b = cutlass.Int32(-32)

            dst_blk_offs = (
                cutlass.Int32(0),
                stride_a,
                stride_b,
                stride_a + stride_b,
            )
            for blk in cutlass.range_constexpr(4):
                src_blk_base = blk * 16
                dst_blk_addr = smem_p_base_bytes + off_base + dst_blk_offs[blk]
                vec = s_regs.load(src_blk_base, 16)
                smem_ptr = cutlass.inttoptr(dst_blk_addr, 3, qkv_element_dtype)
                smem_ptr.store(vec, alignment=16)
        else:
            # BF16 P layout:
            # S<2,4,3> o ((64,16),1,2,(4,2)):((32,1),0,16,(2048,8192)).
            # Each universal SMEM copy is 128 bits, i.e. 8 BF16 elements.
            # The two K slices live 2048 elements apart in the staged P tile.
            # Per-block strides mirror the FP8 swizzle at half the byte width.
            m = ((lane_idx >> 1) & 3) * 64
            base = cutlass.Int32(0)
            base = base + (lane_idx & 1) * 32
            base = base + (lane_idx >> 3) * 256
            base = base + (warp_idx & 1) * 1024
            base = base + (warp_idx >> 1) * 4096
            swizzle_xor = m ^ ((m & 192) >> 3)
            off_base = base + swizzle_xor

            stride_a = cutlass.Int32(8)
            if (m & 64) != 0:
                stride_a = cutlass.Int32(-8)
            stride_b = cutlass.Int32(16)
            if (m & 128) != 0:
                stride_b = cutlass.Int32(-16)

            dst_blk_offs = (
                cutlass.Int32(0),
                stride_a,
                stride_b,
                stride_a + stride_b,
            )
            for k in cutlass.range_constexpr(2):
                k_base = off_base + k * 2048
                src_k_base = k * 32
                for blk in cutlass.range_constexpr(4):
                    src_blk_base = src_k_base + blk * 8
                    dst_blk_addr = (
                        smem_p_base_bytes
                        + (k_base + dst_blk_offs[blk]) * cfg.qkv_dtype_bytes
                    )
                    vec = s_regs.load(src_blk_base, 8)
                    smem_ptr = cutlass.inttoptr(dst_blk_addr, 3, qkv_element_dtype)
                    smem_ptr.store(vec, alignment=16)

        # Fence between SMEM store and MMA read
        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )

    @producer_work
    @cute.jit
    def store_p(self, stage_info: StageInfo, *, qk_acc_regs) -> None:
        """Store P from the even softmax group."""
        self._store_p_impl(stage_info.stage_idx, qk_acc_regs=qk_acc_regs)

    @producer_work
    @cute.jit
    def store_p_odd(self, stage_info: StageInfo, *, qk_acc_regs_odd) -> None:
        """Store P from the odd softmax group."""
        self._store_p_impl(stage_info.stage_idx, qk_acc_regs=qk_acc_regs_odd)

    @producer_work
    @cute.jit
    def store_p_source_direct(self, stage_info: StageInfo, *, qk_acc_regs) -> None:
        """Store source DSV4 P without a P mbarrier producer transition.

        The generated kernel uses the logical sparse K-tile parity as the
        two-buffer selector.  The preceding source-shaped S acquire supplies
        the P-ready happens-before; this method intentionally does not touch
        ``self.pipeline``.
        """
        self._store_p_impl(
            cutlass.Int32(stage_info.loop_offset) & cutlass.Int32(1),
            qk_acc_regs=qk_acc_regs,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def mark_p_source_pipelined_store(self, stage_info: StageInfo) -> None:
        """Represent TmemS's fused packed-P store in the resource DAG.

        The delayed-conversion schedule must keep ``regsP`` local to TmemS,
        so the physical store happens inside ``materialize_softmax_p``.  This
        zero-instruction work item retains the logical tmem_s -> smem_p edge
        without adding a P mbarrier or duplicating the store.
        """
        del stage_info

    @cute.jit
    def _p_desc_at_stage(self, stage_idx):
        """Build the PV descriptor for an explicit two-buffer P stage."""
        cfg = self.cfg
        sp_stage_stride_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0]
            // cfg.num_mma_ctas
            * cfg.mma_pv_tiler[2]
            * cfg.iterations_pv_k
        )
        sp_ptr = self.smem_p.data_ptr(stage_idx * sp_stage_stride_elems)
        return Int64(
            prims.Tcgen05SmemDesc.build(
                start_address=sp_ptr.toint(Int32),
                leading_byte_offset=p_desc_leading_byte_offset(cfg),
                stride_byte_offset=p_desc_stride_byte_offset(cfg),
                layout=p_desc_layout(cfg),
            )
        )

    @consumer_work(returns=desc_p_base)
    @cute.jit
    def p_desc(self, stage_info: StageInfo):
        """MMA warp builds P SMEM descriptor for PV MMA."""
        return self._p_desc_at_stage(stage_info.stage_idx)

    @consumer_work(returns=desc_p_base)
    @cute.jit
    def p_desc_source_direct_prior(self, stage_info: StageInfo):
        """Return P[(current K tile - 1) mod 2] for source W8 LOOP PV."""
        return self._p_desc_at_stage(
            (cutlass.Int32(stage_info.loop_offset) - cutlass.Int32(1))
            & cutlass.Int32(1)
        )

    @consumer_work(returns=desc_p_base)
    @cute.jit
    def p_desc_source_direct_tail(self, stage_info: StageInfo):
        """Return P[(final K tile) mod 2] for source W8 TAIL PV."""
        return self._p_desc_at_stage(
            cutlass.Int32(stage_info.loop_offset) & cutlass.Int32(1)
        )


# =====================================================================
# TmemCorrResource — Correction factors via TMEM, Async pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemCorrResource(HighThroughputMlaResource):
    """Correction factors in TMEM.  Producer: SoftmaxTask.  Consumer: CorrectionTask.

    Pipeline: Async, 2 stages.
    Carries (row_sum, row_max, correction_factor, no_correction) per thread.
    """

    tmem_base_addr: Any = None
    smem_exchange: Any = None
    softmax_scale_log2: Any = None
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    final_row_stats: Any = field(init=False, default=None)
    row_sum: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_max: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    epilogue_row_sum: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    epilogue_row_max: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    correction_factor: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    no_correction: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("row_sum", Float32, Float32(0), "Final running row sum."),
        ("row_max", Float32, Float32(0), "Final running row max."),
        ("epilogue_row_sum", Float32, Float32(0), "Final exchanged row sum."),
        ("epilogue_row_max", Float32, Float32(0), "Final row max for LSE."),
        (
            "correction_factor",
            Float32,
            Float32(0),
            "Correction factor for the previous O tile.",
        ),
        ("no_correction", Int32, Int32(0), "Whether O correction may be skipped."),
    )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(row_sum, row_max, correction_factor, no_correction),
    )
    @cute.jit
    def init_load_state(self, stage_info: StageInfo):
        """Create row-stat variables consumed by correction and epilogue code."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.final_row_stats = cutlass.Array(
            Float32,
            2,
            space=cutlass.AddressSpace.rmem,
        )
        return Float32(0), Float32(0), Float32(0), Int32(0)

    @cute.jit
    def _store_corr_impl(
        self,
        stage_info: StageInfo,
        *,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
        softmax_group_id: cutlass.Constexpr[int] = 0,
        arrive_peer: cutlass.Constexpr[bool] = True,
    ) -> None:
        """Store correction factors [row_sum, row_max, correction, no_correction] to TMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        warp_id = local_tidx >> 5

        col_offset = cfg.correction_factor_offset + stage_idx * 4
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        # tcgen05 addresses pack the TMEM row into the high 16 bits.  Each warp
        # owns one correction row and four adjacent columns for sum/max/scale.
        tmem_raw_addr = (tmem_warp_row_id << 16) | col_offset
        tmem_ptr_arr = prims.make_tmem_ptr(tmem_raw_addr, Float32)

        correction_regs = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        correction_regs[0] = row_sum_out
        correction_regs[1] = row_max_new
        correction_regs[2] = correction_factor_out
        correction_regs[3] = prims.mov_b32(no_correction_out, target_type=Float32)

        prims.tcgen05_st(
            TCGEN05_32B_SHAPE,
            tmem_ptr_arr,
            correction_regs[0:4],
        )
        cute.arch.fence_view_async_tmem_store()

        if cutlass.const_expr(cfg.use_fp8_dual_softmax_schedule and arrive_peer):
            # Dual-softmax pipes are ordered one loop tile apart: after one pipe
            # publishes correction state, it releases the peer pipe for the next
            # loop tile if that peer still has work.
            has_peer_next = stage_info.loop_offset + Int32(1) < stage_info.loop_end
            if has_peer_next:
                prims.barrier_cta_arrive(
                    (
                        cfg.softmax_order_bar_0_id
                        if cutlass.const_expr(softmax_group_id == 1)
                        else cfg.softmax_order_bar_1_id
                    ),
                    2 * num_compute_threads,
                )

    @producer_work
    @cute.jit
    def store_corr(
        self,
        stage_info: StageInfo,
        *,
        row_sum_out,
        row_max_new,
        correction_factor_out,
        no_correction_out,
    ) -> None:
        """Store correction metadata for the even softmax group."""
        self._store_corr_impl(
            stage_info,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
            softmax_group_id=0,
            arrive_peer=True,
        )

    @producer_work
    @cute.jit
    def store_corr_odd(
        self,
        stage_info: StageInfo,
        *,
        row_sum_out_odd,
        row_max_new_odd,
        correction_factor_out_odd,
        no_correction_out_odd,
    ) -> None:
        """Store correction metadata for the odd softmax group."""
        self._store_corr_impl(
            stage_info,
            row_sum_out=row_sum_out_odd,
            row_max_new=row_max_new_odd,
            correction_factor_out=correction_factor_out_odd,
            no_correction_out=no_correction_out_odd,
            softmax_group_id=1,
            arrive_peer=True,
        )

    @cute.jit
    def _store_source_stats_pair(
        self,
        stage_info: StageInfo,
        *,
        first,
        second,
    ) -> None:
        """Store one generated-DSV4 two-float stats payload in TMEM."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        warp_id = local_tidx >> 5
        # TmemSoftmaxLocal reserves one 32-column tcgen05 row per stage:
        # [128,160) and [160,192) in the generated DSV4 kernel.
        col_offset = cfg.correction_factor_offset + stage_info.stage_idx * 32
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        tmem_raw_addr = (tmem_warp_row_id << 16) | col_offset
        tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, Float32)
        stats = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
        stats[0] = first
        stats[1] = second
        prims.tcgen05_st(TCGEN05_32B_SHAPE, tmem_ptr, stats[0:2])
        cute.arch.fence_view_async_tmem_store()

    @producer_work
    @cute.jit
    def store_source_early(
        self,
        stage_info: StageInfo,
        *,
        row_max_old,
        row_max_new,
    ) -> None:
        """Publish ``(old_max, new_max)`` before source DSV4 creates P."""
        self._store_source_stats_pair(
            stage_info,
            first=row_max_old,
            second=row_max_new,
        )

    @producer_work
    @cute.jit
    def store_source_final(
        self,
        stage_info: StageInfo,
        *,
        row_sum,
        row_max,
    ) -> None:
        """Publish terminal ``(row_sum, row_max)`` after the final P tile."""
        self._store_source_stats_pair(
            stage_info,
            first=row_sum,
            second=row_max,
        )

    @consumer_work
    @cute.jit
    def consume_source_head(self, stage_info: StageInfo) -> None:
        """Advance over early[0], for which no previous O tile exists."""
        pass

    @consumer_work(returns=(row_sum, row_max, correction_factor, no_correction))
    @cute.jit
    def load_source_early(self, stage_info: StageInfo):
        """Load an early max pair and compute source's O rescale factor."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        warp_id = local_tidx >> 5
        col_offset = cfg.correction_factor_offset + stage_info.stage_idx * 32
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        tmem_raw_addr = (tmem_warp_row_id << 16) | col_offset
        tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, Float32)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        loaded = prims.tcgen05_ld(TCGEN05_32B_SHAPE, tmem_ptr, num=2)
        cute.arch.fence_view_async_tmem_load()

        max_changed = loaded[0] != loaded[1]
        max_diff = sub_ftz_f32(loaded[0], loaded[1]) if max_changed else Float32(0)
        correction_factor = Float32(1)
        no_correction = Int32(0)
        if cutlass.const_expr(cfg.dsv4_enable_skip_correction):
            no_correction = Int32(not max_changed)
            if max_changed:
                correction_factor = cute.math.exp2(
                    mul_ftz_f32(max_diff, self.softmax_scale_log2),
                    fastmath=True,
                )
        else:
            correction_factor = cute.math.exp2(
                mul_ftz_f32(max_diff, self.softmax_scale_log2),
                fastmath=True,
            )
        return Float32(0), loaded[1], correction_factor, no_correction

    @consumer_work(returns=(row_sum, row_max, correction_factor, no_correction))
    @cute.jit
    def load_source_final(self, stage_info: StageInfo):
        """Load the terminal sum/max pair used by the output epilogue."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        warp_id = local_tidx >> 5
        col_offset = cfg.correction_factor_offset + stage_info.stage_idx * 32
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        tmem_raw_addr = (tmem_warp_row_id << 16) | col_offset
        tmem_ptr = prims.make_tmem_ptr(tmem_raw_addr, Float32)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        loaded = prims.tcgen05_ld(TCGEN05_32B_SHAPE, tmem_ptr, num=2)
        cute.arch.fence_view_async_tmem_load()
        self.final_row_stats[0] = loaded[0]
        self.final_row_stats[1] = loaded[1]
        return loaded[0], loaded[1], Float32(1), Int32(1)

    @consumer_work(returns=(row_sum, row_max, correction_factor, no_correction))
    @cute.jit
    def load_corr(self, stage_info: StageInfo):
        """Load correction factors from TMEM."""
        cfg = self.cfg
        stage_idx = stage_info.stage_idx

        tidx = cute.arch.thread_idx()[0]
        # Use local tidx within 4-warp correction group (matching bare-metal)
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        warp_id = local_tidx >> 5

        col_offset = cfg.correction_factor_offset + stage_idx * 4
        tmem_warp_row_id = self.tmem_base_addr + warp_id * TCGEN05_32B_REGS_PER_LOAD
        # Load from the same packed row/column address used by store_corr so the
        # correction consumer sees the row statistics for its current stage.
        tmem_raw_addr = (tmem_warp_row_id << 16) | col_offset
        tmem_ptr_arr = prims.make_tmem_ptr(tmem_raw_addr, Float32)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        loaded = prims.tcgen05_ld(
            TCGEN05_32B_SHAPE,
            tmem_ptr_arr,
            num=4,
        )

        self.final_row_stats[0] = loaded[0]
        self.final_row_stats[1] = loaded[1]
        return loaded[0], loaded[1], loaded[2], loaded[3].bitcast(Int32)

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY, returns=(epilogue_row_sum, epilogue_row_max)
    )
    @cute.jit
    def prepare_epilogue_slice_store(self, stage_info: StageInfo):
        """Exchange final row statistics once before per-slice O stores."""
        del stage_info
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        row_sum = self.final_row_stats[0]
        row_max = self.final_row_stats[1]

        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        smem_ex_ptr = cutlass.inttoptr(
            self.smem_exchange + local_tidx * 4,
            3,
            Float32,
        )
        smem_ex_ptr.store(row_sum)
        prims.barrier_cta_sync(
            cfg.epilogue_sync_bar_id, thread_count=cfg.epilogue_sync_threads
        )
        # The two CTAs in the cluster own complementary halves of the 2CTA row.
        # Exchanging row sums through SMEM gives both epilogue slices the same
        # denominator while preserving each CTA's local row max for LSE.
        peer_idx = (local_tidx + 64) % num_compute_threads
        peer_ptr = cutlass.inttoptr(
            self.smem_exchange + peer_idx * 4,
            3,
            Float32,
        )
        row_sum = (
            add_ftz_f32(row_sum, peer_ptr.load())
            if cutlass.const_expr(cfg.is_dynamic_token_sparse)
            else row_sum + peer_ptr.load()
        )
        prims.barrier_cta_sync(
            cfg.epilogue_sync_bar_id, thread_count=cfg.epilogue_sync_threads
        )
        return row_sum, row_max


# =====================================================================
# TmemOResource — O accumulator in TMEM, UmmaProducerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemOResource(HighThroughputMlaResource):
    """TMEM O accumulator.  Producer: Mma (PV MMA).  Consumer: Correction.

    Pipeline: UmmaProducerAsync, 1 stage.
    """

    tmem_base_addr: Any = None
    tmem_corr_ref: Any = None  # Reference to TmemCorrResource for correction data
    smem_p: Any = None  # SMEM P for PV MMA
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)

    @producer_work
    @cute.jit
    def pv_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_p_base,
        desc_v_base,
        v_subtile_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Issue one PV MMA sub-tile.
        Only leader CTA issues MMA (2CTA UMMA principle).

        Descriptor computation is hoisted OUTSIDE the leader-CTA gate so
        ptxas keeps values in uniform registers.
        """
        cfg = self.cfg

        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            n_dim=cfg.mma_pv_tiler[1],
            m_dim=cfg.mma_pv_tiler[0],
            b_major=1,
        )
        mma_kind = mma_kind_for_qkv(cfg)
        cta_group = prims.CTAGroup.CTA_2
        is_leader_cta = (
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
        )

        sp_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        desc_p_delta = cutlass.const_expr(sp_copy_elems // 8)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            desc_p_delta = cutlass.const_expr(256)
        sv_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        desc_v_delta = cutlass.const_expr(sv_copy_elems * cfg.qkv_dtype_bytes // 16)

        pv_k_block_count = cutlass.const_expr(
            cfg.mma_pv_tiler[2] // mma_k_step_for_qkv(cfg)
        )

        # Match the producer's physical V-stage decomposition. Each stage
        # carries two adjacent K32 slices for one D256 output panel.
        pv_j = v_subtile_idx // cfg.kv_subtiles_per_stage
        token_partition = v_subtile_idx % cfg.kv_subtiles_per_stage
        tmem_o_base_addr = self.tmem_base_addr + cfg.tmem_o_offset
        tmem_o_addr = tmem_o_base_addr + pv_j * 128

        for stage_subtile_idx in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
            pv_i = token_partition * cfg.kv_subtiles_per_stage + stage_subtile_idx
            desc_p = desc_p_base + pv_i * desc_p_delta
            desc_v = desc_v_base + stage_subtile_idx * desc_v_delta

            # Clear O for each output-N panel on its first PV K slice, then
            # accumulate the remaining slices and subsequent sequence tiles.
            scale_d_pv = Boolean(True)
            if cutlass.const_expr(pv_i == 0):
                if cutlass.const_expr(is_tail):
                    scale_d_pv = Boolean(
                        stage_info.loop_end > Int32(stage_info.loop_start)
                    )
                else:
                    scale_d_pv = Boolean(
                        stage_info.loop_offset != Int32(stage_info.loop_start)
                    )
            if is_leader_cta:
                for k_block in cutlass.range_constexpr(pv_k_block_count):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            mma_kind,
                            cta_group,
                            prims.make_tmem_ptr(tmem_o_addr, Float32),
                            desc_p + k_block * 2,
                            desc_v + k_block * (256 if cfg.is_fp8_qkv() else 128),
                            idesc_pv,
                            Boolean(scale_d_pv),
                        )
                    scale_d_pv = Boolean(True)

    @producer_work
    @cute.jit
    def pv_mma_n_major(
        self,
        stage_info: StageInfo,
        *,
        desc_p_base,
        desc_v_base,
        pv_n_idx: cutlass.Constexpr[int],
        pv_k_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Issue PV MMA in output-N-major order for per-slice O tokens."""
        cfg = self.cfg

        pv_j = pv_n_idx
        pv_i = pv_k_idx
        tmem_o_base_addr = self.tmem_base_addr + cfg.tmem_o_offset
        tmem_o_addr = tmem_o_base_addr + pv_j * 128

        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            n_dim=cfg.mma_pv_tiler[1],
            m_dim=cfg.mma_pv_tiler[0],
            b_major=1,
        )
        mma_kind = mma_kind_for_qkv(cfg)
        cta_group = prims.CTAGroup.CTA_2
        is_leader_cta = (
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
        )

        sp_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        desc_p_delta = cutlass.const_expr(sp_copy_elems // 8)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            desc_p_delta = cutlass.const_expr(256)
        desc_p = desc_p_base + pv_i * desc_p_delta

        pv_k_block_count = cutlass.const_expr(
            cfg.mma_pv_tiler[2] // mma_k_step_for_qkv(cfg)
        )
        scale_d_pv = Boolean(True)
        if cutlass.const_expr(pv_i == 0):
            if cutlass.const_expr(is_tail):
                scale_d_pv = Boolean(stage_info.loop_end > Int32(stage_info.loop_start))
            else:
                scale_d_pv = Boolean(
                    stage_info.loop_offset != Int32(stage_info.loop_start)
                )

        if is_leader_cta:
            for k_block in cutlass.range_constexpr(pv_k_block_count):
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        mma_kind,
                        cta_group,
                        prims.make_tmem_ptr(tmem_o_addr, Float32),
                        desc_p + k_block * 2,
                        desc_v_base + k_block * (256 if cfg.is_fp8_qkv() else 128),
                        idesc_pv,
                        Boolean(scale_d_pv),
                    )
                scale_d_pv = Boolean(True)

    @consumer_work
    @cute.jit
    def rescale_o(
        self, stage_info: StageInfo, *, correction_factor, no_correction
    ) -> None:
        """Rescale O in-place in TMEM by correction_factor."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]

        # Use local tidx within 4-warp correction group (matching bare-metal)
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        tmem_warp_row_id = (
            self.tmem_base_addr
            + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | cfg.tmem_o_offset

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_ops = 4  # 4 loads x 32 = 128 elements per iter_n

        skip_correction = prims.vote_sync(
            cute.arch.FULL_MASK,
            no_correction == 1,
            prims.VoteSync.ALL,
        )

        if not skip_correction:
            for iter_n in cutlass.range_constexpr(cfg.iterations_pv_n):
                tmem_addr_offset = iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
                for idx in cutlass.range_constexpr(num_tmem_ops):
                    curr_addr = (
                        tmem_raw_addr
                        + tmem_addr_offset
                        + idx * TCGEN05_32B_REGS_PER_LOAD
                    )
                    tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                    chunk = prims.tcgen05_ld(
                        t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                    )
                    if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                        scaled = cutlass.Array(
                            Float32,
                            TCGEN05_32B_REGS_PER_LOAD,
                            space=cutlass.AddressSpace.rmem,
                        )
                        for pair in cutlass.range_constexpr(
                            0, TCGEN05_32B_REGS_PER_LOAD, 2
                        ):
                            dsv4_warp_switch()
                            scaled_pair = mul_packed_f32x2(
                                (chunk[pair], chunk[pair + 1]),
                                (correction_factor, correction_factor),
                            )
                            scaled[pair] = scaled_pair[0]
                            scaled[pair + 1] = scaled_pair[1]
                        scaled = scaled.load(0, TCGEN05_32B_REGS_PER_LOAD)
                    else:
                        scaled = chunk * cutlass.full_like(chunk, correction_factor)
                    prims.tcgen05_st(t2r_shape, tmem_ptr, scaled)

            if cutlass.const_expr(cfg.dsv4_fuses_inv_rope_fp8_quant):
                # The fence wrapper is itself one tcgen05.wait::st.  Match the
                # generated DSV4 correction path: complete issued stores once,
                # and do no TMEM completion work when correction was skipped.
                cute.arch.fence_view_async_tmem_store()

        if cutlass.const_expr(not cfg.dsv4_fuses_inv_rope_fp8_quant):
            # Preserve the existing completion protocol outside RopeQuant;
            # F-150 is intentionally limited to its source-equivalent path.
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            cute.arch.fence_view_async_tmem_store()

    @consumer_work
    @cute.jit
    def rescale_o_slice(
        self,
        stage_info: StageInfo,
        *,
        correction_factor,
        no_correction,
        iter_n: cutlass.Constexpr[int],
    ) -> None:
        """Rescale one PV output-N slice after its O token is ready."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        tmem_warp_row_id = (
            self.tmem_base_addr
            + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | (
            cfg.tmem_o_offset + iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
        )

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_ops = 4
        skip_correction = prims.vote_sync(
            cute.arch.FULL_MASK,
            no_correction == 1,
            prims.VoteSync.ALL,
        )
        if not skip_correction:
            for idx in cutlass.range_constexpr(num_tmem_ops):
                curr_addr = tmem_raw_addr + idx * TCGEN05_32B_REGS_PER_LOAD
                tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                chunk = prims.tcgen05_ld(
                    t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                )
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    scaled = cutlass.Array(
                        Float32,
                        TCGEN05_32B_REGS_PER_LOAD,
                        space=cutlass.AddressSpace.rmem,
                    )
                    for pair in cutlass.range_constexpr(
                        0, TCGEN05_32B_REGS_PER_LOAD, 2
                    ):
                        dsv4_warp_switch()
                        scaled_pair = mul_packed_f32x2(
                            (chunk[pair], chunk[pair + 1]),
                            (correction_factor, correction_factor),
                        )
                        scaled[pair] = scaled_pair[0]
                        scaled[pair + 1] = scaled_pair[1]
                    scaled = scaled.load(0, TCGEN05_32B_REGS_PER_LOAD)
                else:
                    scaled = chunk * cutlass.full_like(chunk, correction_factor)
                prims.tcgen05_st(t2r_shape, tmem_ptr, scaled)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        cute.arch.fence_view_async_tmem_store()


# =====================================================================
# GmemOResource — Output to GMEM (no pipeline)
# =====================================================================


@dataclass(kw_only=True)
class GmemOResource(HighThroughputMlaResource):
    """GMEM output.  Producer: Correction (epilogue store).  No pipeline."""

    output: Any = None
    dsv4_inv_rope_cos_sin_cache: Any = None
    dsv4_o_scale: Any = None
    cache_seqs: Any = None
    partial_output: Any = None
    lse: Any = None
    partial_lse: Any = None
    tmem_o_ref: Any = None  # Reference to TmemOResource for tmem_base_addr
    tmem_corr_ref: Any = None  # Reference to TmemCorrResource for correction data
    output_scale: Any = None
    softmax_scale_log2: Any = None
    softmax_p_scale_rcp: Any = None
    smem_exchange: Any = None  # SMEM for row_sum exchange (as Int32 base addr)
    split_kv: Any = None
    cu_seqlens_q: Any = None
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    logical_seq_len_q: cutlass.Constexpr[int] = 1
    stores_lse: cutlass.Constexpr[bool] = True
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)

    @cute.jit
    def _query_row_state(self, row_in_tile, query_tile_idx, batch_idx):
        """Map one physical flat-tile row to public storage."""
        return flat_query_row_state(
            row_in_tile,
            query_tile_idx,
            self.cfg.mma_qk_tiler[0],
            self.logical_num_heads_q,
            self.logical_seq_len_q,
            self.cu_seqlens_q,
            batch_idx,
        )

    @producer_work
    @cute.jit
    def epilogue_store(self, stage_info: StageInfo) -> None:
        """Full epilogue: load O from TMEM, normalize by row_sum, store to GMEM.

        Steps:
        1. Exchange row_sum across warp pairs via SMEM
        2. Load O from TMEM
        3. Normalize by output_scale / row_sum
        4. Store to GMEM
        5. Compute and store LSE
        """
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        # Use work_tile's blk_coord (updated by persistent loop) instead of
        # self.blk_coord (captured at construction time) so that each work
        # tile writes to the correct output location.
        blk_coord = stage_info.work_tile.tile_idx

        row_sum = self.tmem_corr_ref.final_row_stats[0]
        row_max = self.tmem_corr_ref.final_row_stats[1]

        # Exchange row_sum between warp pairs (0,2) and (1,3) via SMEM
        # Use local thread index within the 4-warp correction group,
        # matching bare-metal: tidx % (num_compute_warps * threads_per_warp)
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        smem_ex_ptr = cutlass.inttoptr(
            self.smem_exchange + local_tidx * 4,
            3,
            Float32,
        )
        smem_ex_ptr.store(row_sum)
        epilogue_barrier_id = Int32(cfg.epilogue_sync_bar_id)
        if cutlass.const_expr(cfg.dsv4_use_source_epilogue_pair_barrier):
            epilogue_barrier_id = epilogue_barrier_id + (
                (local_tidx >> WARP_LANE_SHIFT) & Int32(1)
            )
        prims.barrier_cta_sync(
            epilogue_barrier_id, thread_count=cfg.epilogue_sync_threads
        )
        peer_idx = (local_tidx + 64) % num_compute_threads
        peer_ptr = cutlass.inttoptr(
            self.smem_exchange + peer_idx * 4,
            3,
            Float32,
        )
        row_sum = (
            add_ftz_f32(row_sum, peer_ptr.load())
            if cutlass.const_expr(cfg.is_dynamic_token_sparse)
            else row_sum + peer_ptr.load()
        )
        if cutlass.const_expr(not cfg.dsv4_use_source_epilogue_pair_barrier):
            prims.barrier_cta_sync(
                cfg.epilogue_sync_bar_id, thread_count=cfg.epilogue_sync_threads
            )

        # TMEM address for O — use local tidx within 4-warp correction group
        tmem_base_addr = self.tmem_o_ref.tmem_base_addr
        tmem_warp_row_id = (
            tmem_base_addr + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | cfg.tmem_o_offset

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_loads = 4  # 4 x 32 = 128 elements per iter_n

        # Per-thread O indexing — use local tidx
        tile_h = cfg.mma_pv_tiler[0] // cfg.num_mma_ctas  # 64
        tile_d = cfg.mma_pv_tiler[1]  # 256
        tidx_g = local_tidx & EPILOGUE_THREAD_TILE_MASK
        g_i = tidx_g & EPILOGUE_ROW_MASK
        g_j = (tidx_g >> EPILOGUE_COLUMN_GROUP_SHIFT) * EPILOGUE_THREAD_TILE_THREADS

        # Public O/LSE remain in logical coordinates. Split-KV partials retain
        # physical flat-tile coordinates until final reduction.
        logical_num_heads_q = Int32(self.logical_num_heads_q)
        physical_tile_rows = Int32(cfg.mma_qk_tiler[0])
        D = cfg.latent_dim
        head_tile_idx = blk_coord[0]
        seq_q_idx = blk_coord[1]
        batch_idx = blk_coord[2]
        split_kv_idx = blk_coord[3]
        row_in_tile = head_tile_idx * tile_h + g_i
        (
            storage_flat_query_row,
            logical_head_idx,
            logical_q_idx,
            storage_q_idx,
            query_is_valid,
        ) = self._query_row_state(row_in_tile, seq_q_idx, batch_idx)

        # Fully masked split rows can occur when physical tail rows or earlier
        # causal query rows have no visible K values in this split. Store zero
        # O and -inf LSE so split-KV reduction gives those rows zero weight.
        row_has_values = row_sum > Float32(0)
        safe_row_sum = row_sum if row_has_values else Float32(1)
        row_sum_rcp = cute.math.rcp(safe_row_sum, approx=True)
        norm_scale = (
            mul_ftz_f32(self.output_scale, row_sum_rcp)
            if cutlass.const_expr(cfg.is_dynamic_token_sparse)
            else self.output_scale * row_sum_rcp
        )

        for iter_n in cutlass.range_constexpr(cfg.iterations_pv_n):
            # Load O from TMEM
            qk_acc_regs = cutlass.Array(Float32, 128, space=cutlass.AddressSpace.rmem)
            tmem_raw_addr_n = tmem_raw_addr + (
                iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
            )
            for load_idx in cutlass.range_constexpr(num_tmem_loads):
                curr_addr = tmem_raw_addr_n + load_idx * TCGEN05_32B_REGS_PER_LOAD
                tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                loaded = prims.tcgen05_ld(
                    t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                )
                qk_acc_regs.store(loaded, load_idx * TCGEN05_32B_REGS_PER_LOAD)

            # Normalize: O = O * output_scale / row_sum
            for i in cutlass.range_constexpr(0, 128, 2):
                if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                    dsv4_warp_switch()
                scaled = mul_packed_f32x2(
                    (qk_acc_regs[i], qk_acc_regs[i + 1]),
                    (norm_scale, norm_scale),
                )
                qk_acc_regs[i] = scaled[0]
                qk_acc_regs[i + 1] = scaled[1]

            # Store O to GMEM
            if row_in_tile < physical_tile_rows and query_is_valid:
                if cutlass.const_expr(cfg.dsv4_fuses_inv_rope_fp8_quant):
                    # Source physical output layout: [H/8, T, 8, D].
                    total_q = Int64(cute.size(self.output)) // Int64(
                        self.logical_num_heads_q * D
                    )
                    group_idx = Int64(logical_head_idx >> Int32(3))
                    head_idx_in_group = Int64(logical_head_idx & Int32(7))
                    head_dim_offset = Int32(iter_n * tile_d) + Int32(g_j)
                    quant_block_idx = Int64(head_dim_offset >> Int32(7))
                    # DSv4 inverse RoPE covers D[448:512], which is the upper
                    # half of the D384 quant block.  iter_n is compile-time;
                    # the runtime head-dim choice is warp-pair uniform.
                    if cutlass.const_expr(iter_n == 1):
                        _, query_len = query_batch_bounds(
                            self.cu_seqlens_q,
                            batch_idx,
                            self.logical_seq_len_q,
                        )
                        position = (
                            Int32(self.cache_seqs[batch_idx])
                            - Int32(query_len)
                            + Int32(logical_q_idx)
                        )
                        cos_sin_ptr = (
                            self.dsv4_inv_rope_cos_sin_cache.iterator.raw_ptr()
                            + Int64(position) * Int64(64)
                        )
                        for rope_pair_idx in cutlass.range_constexpr(32):
                            value_idx = 64 + rope_pair_idx * 2
                            first = qk_acc_regs[value_idx]
                            second = qk_acc_regs[value_idx + 1]
                            rotated_first = first
                            rotated_second = second
                            if head_dim_offset == Int32(384):
                                cos_value = (cos_sin_ptr + rope_pair_idx).load()
                                sin_value = (cos_sin_ptr + 32 + rope_pair_idx).load()
                                rotated_first = fma_ftz_f32(
                                    first,
                                    cos_value,
                                    mul_ftz_f32(second, sin_value),
                                )
                                rotated_second = fma_ftz_f32(
                                    second,
                                    cos_value,
                                    -mul_ftz_f32(first, sin_value),
                                )
                            qk_acc_regs[value_idx] = rotated_first
                            qk_acc_regs[value_idx + 1] = rotated_second
                    max0 = Float32(1.0e-12)
                    max1 = Float32(1.0e-12)
                    for reduce_idx in cutlass.range_constexpr(32):
                        value_idx = reduce_idx * 4
                        max0 = max3_abs_ftz_f32(
                            max0,
                            qk_acc_regs[value_idx],
                            qk_acc_regs[value_idx + 1],
                        )
                        max1 = max3_abs_ftz_f32(
                            max1,
                            qk_acc_regs[value_idx + 2],
                            qk_acc_regs[value_idx + 3],
                        )
                    amax = max3_abs_ftz_f32(
                        max0,
                        max1,
                        Float32(1.0e-12),
                    )
                    scale_buf_m = (total_q + Int64(3)) & Int64(-4)
                    scale_offset = (
                        group_idx * Int64(32) * scale_buf_m
                        + (head_idx_in_group * Int64(4) + quant_block_idx) * scale_buf_m
                        + Int64(storage_q_idx)
                    )
                    dequant_scale = amax * Float32(1.0 / 448.0)
                    inv_scale = rcp_approx_ftz_f32(amax) * Float32(448.0)
                    (self.dsv4_o_scale.iterator.raw_ptr() + scale_offset).store(
                        dequant_scale
                    )
                    fp8_output_offset = (
                        group_idx * total_q * Int64(8 * D)
                        + Int64(storage_q_idx) * Int64(8 * D)
                        + head_idx_in_group * Int64(D)
                        + Int64(head_dim_offset)
                    )
                    output_base = self.output.iterator.raw_ptr() + fp8_output_offset
                    for load_idx in cutlass.range_constexpr(num_tmem_loads):
                        offset = load_idx * TCGEN05_32B_REGS_PER_LOAD
                        vec_f32 = qk_acc_regs.load(offset, TCGEN05_32B_REGS_PER_LOAD)
                        packed_o = cutlass.Array(
                            Int32,
                            PACKED_FP8_OUTPUT_REGS * 2,
                            space=cutlass.AddressSpace.rmem,
                        )
                        for pack_idx in cutlass.range_constexpr(
                            PACKED_FP8_OUTPUT_REGS * 2
                        ):
                            pack_offset = pack_idx * 4
                            packed_o[pack_idx] = pack_float4_to_fp8_e4m3(
                                vec_f32[pack_offset] * inv_scale,
                                vec_f32[pack_offset + 1] * inv_scale,
                                vec_f32[pack_offset + 2] * inv_scale,
                                vec_f32[pack_offset + 3] * inv_scale,
                            )
                        raw_ptr = cutlass.inttoptr(
                            (output_base + load_idx * TCGEN05_32B_REGS_PER_LOAD).toint(
                                Int64
                            ),
                            mem_space=1,
                            dtype=Int32,
                        )
                        raw_ptr.store(
                            packed_o.load(0, PACKED_FP8_OUTPUT_REGS * 2),
                            alignment=32,
                        )
                elif cutlass.const_expr(False):
                    # TRTLLM-gen's DSv4 fused output is a physical grouped
                    # tensor, not logical [T,H,D]:
                    #   O_fp8 [H/8, T, 8, 512]
                    #   scale [H/8, 8*4, pad4(T)]
                    # Each correction thread owns exactly one contiguous D128
                    # quant block (g_j is 0 or 128; iter_n selects D256).
                    head_dim_offset = Int32(iter_n * tile_d) + Int32(g_j)
                    quant_block_idx = head_dim_offset >> Int32(7)

                    # Only the second D256 stage can overlap [448:512].  Keep
                    # iter_n compile-time so the first half has no RoPE code.
                    # In the second stage, the two warp pairs select D256 and
                    # D384 respectively.  Identity coefficients avoid a huge
                    # 128-register branch phi while retaining source math for
                    # the D384 block; this is revisited after the first
                    # correctness gate if its extra D256 FMAs are measurable.
                    # Diagnostic compile gate: first establish that the
                    # physical FP8 + per-block scale path is backend-safe;
                    # inverse RoPE is restored immediately after that gate.
                    if cutlass.const_expr(False):
                        _, query_len = query_batch_bounds(
                            self.cu_seqlens_q,
                            batch_idx,
                            self.logical_seq_len_q,
                        )
                        position = (
                            Int32(self.cache_seqs[batch_idx])
                            - Int32(query_len)
                            + Int32(logical_q_idx)
                        )
                        cos_sin_ptr = (
                            self.dsv4_inv_rope_cos_sin_cache.iterator.raw_ptr()
                            + Int64(position) * Int64(64)
                        )
                        for rope_pair_idx in cutlass.range_constexpr(32):
                            value_idx = 64 + rope_pair_idx * 2
                            first = qk_acc_regs[value_idx]
                            second = qk_acc_regs[value_idx + 1]
                            cos_value = Float32(1.0)
                            sin_value = Float32(0.0)
                            if head_dim_offset == Int32(384):
                                cos_value = (cos_sin_ptr + rope_pair_idx).load()
                                sin_value = (cos_sin_ptr + 32 + rope_pair_idx).load()
                            rotated_first = fma_ftz_f32(
                                first,
                                cos_value,
                                mul_ftz_f32(second, sin_value),
                            )
                            rotated_second = fnma_ftz_f32(
                                first,
                                sin_value,
                                mul_ftz_f32(second, cos_value),
                            )
                            qk_acc_regs[value_idx] = rotated_first
                            qk_acc_regs[value_idx + 1] = rotated_second

                    # Match reduceMaxAbs<128>'s four independent chains. Max
                    # is exact/associative for this finite output domain, so
                    # reducing after RoPE is numerically identical to source's
                    # split first-64/rotated-pair reduction.
                    if cutlass.const_expr(False):
                        max0 = Float32(1.0e-12)
                        max1 = Float32(1.0e-12)
                        max2 = Float32(1.0e-12)
                        max3 = Float32(1.0e-12)
                        for reduce_idx in cutlass.range_constexpr(32):
                            value_idx = reduce_idx * 4
                            max0 = fmax_f32(max0, fabs_f32(qk_acc_regs[value_idx]))
                            max1 = fmax_f32(max1, fabs_f32(qk_acc_regs[value_idx + 1]))
                            max2 = fmax_f32(max2, fabs_f32(qk_acc_regs[value_idx + 2]))
                            max3 = fmax_f32(max3, fabs_f32(qk_acc_regs[value_idx + 3]))
                        amax = fmax_f32(fmax_f32(max0, max1), fmax_f32(max2, max3))
                    else:
                        amax = Float32(448.0)

                    fp8_max = Float32(448.0)
                    dequant_scale = mul_ftz_f32(amax, Float32(1.0 / 448.0))
                    inv_scale = mul_ftz_f32(cute.math.rcp(amax, approx=True), fp8_max)
                    scale_buf_m = Int64(cute.size(self.dsv4_o_scale)) // Int64(16 * 32)
                    group_idx = Int64(logical_head_idx >> Int32(3))
                    head_idx_in_group = Int64(logical_head_idx & Int32(7))
                    scale_offset = (
                        group_idx * Int64(32) * scale_buf_m
                        + (head_idx_in_group * Int64(4) + Int64(quant_block_idx))
                        * scale_buf_m
                        + Int64(storage_q_idx)
                    )
                    (self.dsv4_o_scale.iterator.raw_ptr() + scale_offset).store(
                        dequant_scale
                    )

                    total_q = Int64(cute.size(self.output)) // Int64(
                        self.logical_num_heads_q * D
                    )
                    fp8_output_offset = (
                        group_idx * total_q * Int64(8 * D)
                        + Int64(storage_q_idx) * Int64(8 * D)
                        + head_idx_in_group * Int64(D)
                        + Int64(head_dim_offset)
                    )
                    output_base = self.output.iterator.raw_ptr() + fp8_output_offset
                    if cutlass.const_expr(False):
                        for load_idx in cutlass.range_constexpr(num_tmem_loads):
                            for j in cutlass.range_constexpr(2):
                                offset = (
                                    load_idx * TCGEN05_32B_REGS_PER_LOAD
                                    + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                )
                                packed_o = cutlass.Array(
                                    Int32,
                                    PACKED_FP8_OUTPUT_REGS,
                                    space=cutlass.AddressSpace.rmem,
                                )
                                for pack_idx in cutlass.range_constexpr(
                                    PACKED_FP8_OUTPUT_REGS
                                ):
                                    pack_offset = offset + pack_idx * 4
                                    packed_o[pack_idx] = pack_float4_to_fp8_e4m3(
                                        mul_ftz_f32(
                                            qk_acc_regs[pack_offset], inv_scale
                                        ),
                                        mul_ftz_f32(
                                            qk_acc_regs[pack_offset + 1], inv_scale
                                        ),
                                        mul_ftz_f32(
                                            qk_acc_regs[pack_offset + 2], inv_scale
                                        ),
                                        mul_ftz_f32(
                                            qk_acc_regs[pack_offset + 3], inv_scale
                                        ),
                                    )
                                raw_ptr = cutlass.inttoptr(
                                    (
                                        output_base
                                        + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                        + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                    ).toint(Int64),
                                    mem_space=1,
                                    dtype=Int32,
                                )
                                raw_ptr.store(
                                    packed_o.load(0, PACKED_FP8_OUTPUT_REGS),
                                    alignment=16,
                                )
                elif cutlass.const_expr(self.partial_output is not None):
                    # Split-KV partial O uses BF16 workspace storage.  LSE and
                    # the eventual cross-split accumulation remain FP32.
                    S_q = (
                        cutlass.Int32(self.partial_output.shape[3])
                        if self.partial_output is not None
                        else Int32(1)
                    )
                    o_base_ptr = (
                        self.partial_output.iterator.raw_ptr()
                        + Int64(row_in_tile) * Int64(self.split_kv) * Int64(D)
                        + Int64(split_kv_idx) * Int64(D)
                        + Int64(seq_q_idx)
                        * Int64(self.split_kv)
                        * Int64(physical_tile_rows)
                        * Int64(D)
                        + Int64(batch_idx)
                        * Int64(physical_tile_rows)
                        * Int64(self.split_kv)
                        * Int64(S_q)
                        * Int64(D)
                    )
                    output_base = o_base_ptr + iter_n * tile_d + g_j
                    for load_idx in cutlass.range_constexpr(num_tmem_loads):
                        for j in cutlass.range_constexpr(4):
                            offset = (
                                load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * BF16_OUTPUT_VECTOR_ELEMENTS
                            )
                            vec_f32 = qk_acc_regs.load(
                                offset, BF16_OUTPUT_VECTOR_ELEMENTS
                            )
                            vec_partial = convert_f32_vector_to_bf16_satfinite(
                                vec_f32, BF16_OUTPUT_VECTOR_ELEMENTS
                            )
                            (
                                output_base
                                + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * BF16_OUTPUT_VECTOR_ELEMENTS
                            ).nvvm_store_ext(
                                vec_partial,
                                evict="noallocate",
                            )
                else:
                    # 16-bit output (split_kv == 1, direct output)
                    if cutlass.const_expr(self.cu_seqlens_q is not None):
                        o_base_ptr = self.output.iterator.raw_ptr() + Int64(
                            storage_flat_query_row
                        ) * Int64(D)
                    else:
                        S_q = (
                            cutlass.Int32(self.output.shape[2])
                            if self.output is not None
                            else Int32(1)
                        )
                        o_base_ptr = (
                            self.output.iterator.raw_ptr()
                            + Int64(logical_head_idx) * Int64(D)
                            + Int64(logical_q_idx)
                            * Int64(logical_num_heads_q)
                            * Int64(D)
                            + Int64(batch_idx)
                            * Int64(logical_num_heads_q)
                            * Int64(D)
                            * Int64(S_q)
                        )
                    output_base = o_base_ptr + iter_n * tile_d + g_j
                    for load_idx in cutlass.range_constexpr(num_tmem_loads):
                        for j in cutlass.range_constexpr(2):
                            offset = (
                                load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * FP8_OUTPUT_VECTOR_ELEMENTS
                            )
                            vec_f32 = qk_acc_regs.load(
                                offset, FP8_OUTPUT_VECTOR_ELEMENTS
                            )
                            if cutlass.const_expr(cfg.use_fp8_output == 1):
                                packed_o = cutlass.Array(
                                    Int32,
                                    PACKED_FP8_OUTPUT_REGS,
                                    space=cutlass.AddressSpace.rmem,
                                )
                                for pack_idx in cutlass.range_constexpr(
                                    PACKED_FP8_OUTPUT_REGS
                                ):
                                    pack_offset = pack_idx * PACKED_FP8_OUTPUT_REGS
                                    packed_o[pack_idx] = pack_float4_to_fp8_e4m3(
                                        vec_f32[pack_offset],
                                        vec_f32[pack_offset + 1],
                                        vec_f32[pack_offset + 2],
                                        vec_f32[pack_offset + 3],
                                    )
                                raw_ptr = cutlass.inttoptr(
                                    (
                                        output_base
                                        + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                        + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                    ).toint(Int64),
                                    mem_space=1,
                                    dtype=Int32,
                                )
                                raw_ptr.store(
                                    packed_o.load(0, PACKED_FP8_OUTPUT_REGS),
                                    alignment=16,
                                )
                            else:
                                if cutlass.const_expr(
                                    output_dtype(self.cfg) is cutlass.BFloat16
                                ):
                                    vec_o = convert_f32_vector_to_bf16_satfinite(
                                        vec_f32, FP8_OUTPUT_VECTOR_ELEMENTS
                                    )
                                else:
                                    vec_o = vec_f32.to(output_dtype(self.cfg))
                                (
                                    output_base
                                    + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                    + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                ).nvvm_store_ext(
                                    vec_o,
                                    evict="noallocate",
                                )

        if cutlass.const_expr(self.stores_lse):
            # Compute and store LSE in the same row-sum domain used by P.
            lse_row_sum = row_sum
            if cutlass.const_expr(cfg.is_fp8_qkv()):
                lse_row_sum = lse_row_sum * (
                    self.softmax_p_scale_rcp
                    if cutlass.const_expr(cfg.is_dynamic_token_sparse)
                    else fp8_quant_scale_rcp()
                )
            lse = (
                cute.math.log2(lse_row_sum, fastmath=True)
                + self.softmax_scale_log2 * row_max
                if row_has_values
                else Float32(-Float32.inf)
            )

            # Use local_tidx (0..127 within correction warpgroup) for LSE
            # indexing, not global tidx (which is 128..255 for correction warps).
            lse_tidx = local_tidx
            if lse_tidx < tile_h:
                lse_row_in_tile = head_tile_idx * tile_h + lse_tidx
                (
                    storage_flat_lse_row,
                    logical_lse_head_idx,
                    logical_lse_q_idx,
                    _,
                    lse_query_is_valid,
                ) = self._query_row_state(lse_row_in_tile, seq_q_idx, batch_idx)
                if lse_row_in_tile < physical_tile_rows and lse_query_is_valid:
                    if cutlass.const_expr(self.partial_lse is not None):
                        S_q = (
                            cutlass.Int32(self.partial_lse.shape[2])
                            if self.partial_lse is not None
                            else Int32(1)
                        )
                        lse_base_ptr = (
                            self.partial_lse.iterator.raw_ptr()
                            + Int64(lse_row_in_tile) * Int64(self.split_kv)
                            + Int64(split_kv_idx)
                            + Int64(seq_q_idx)
                            * Int64(physical_tile_rows)
                            * Int64(self.split_kv)
                            + Int64(batch_idx)
                            * Int64(physical_tile_rows)
                            * Int64(self.split_kv)
                            * Int64(S_q)
                        )
                        lse_base_ptr.store(lse)
                    elif cutlass.const_expr(self.lse is not None):
                        if cutlass.const_expr(self.cu_seqlens_q is not None):
                            lse_base_ptr = (
                                self.lse.iterator.raw_ptr() + storage_flat_lse_row
                            )
                        else:
                            S_q = (
                                cutlass.Int32(self.lse.shape[1])
                                if self.lse is not None
                                else Int32(1)
                            )
                            lse_base_ptr = (
                                self.lse.iterator.raw_ptr()
                                + Int64(logical_lse_head_idx)
                                + Int64(logical_lse_q_idx) * Int64(logical_num_heads_q)
                                + Int64(batch_idx)
                                * Int64(logical_num_heads_q)
                                * Int64(S_q)
                            )
                        lse_base_ptr.store(lse)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()

    @producer_work
    @cute.jit
    def epilogue_store_slice(
        self,
        stage_info: StageInfo,
        *,
        row_sum,
        row_max,
        iter_n: cutlass.Constexpr[int],
    ) -> None:
        """Store one output-N slice after its O pipeline token is ready."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        blk_coord = stage_info.work_tile.tile_idx

        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads

        tmem_base_addr = self.tmem_o_ref.tmem_base_addr
        tmem_warp_row_id = (
            tmem_base_addr + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | (
            cfg.tmem_o_offset + iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
        )

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_loads = 4
        qk_acc_regs = cutlass.Array(Float32, 128, space=cutlass.AddressSpace.rmem)
        for load_idx in cutlass.range_constexpr(num_tmem_loads):
            curr_addr = tmem_raw_addr + load_idx * TCGEN05_32B_REGS_PER_LOAD
            tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
            loaded = prims.tcgen05_ld(
                t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
            )
            qk_acc_regs.store(loaded, load_idx * TCGEN05_32B_REGS_PER_LOAD)

        row_has_values = row_sum > Float32(0)
        safe_row_sum = row_sum if row_has_values else Float32(1)
        row_sum_rcp = cute.math.rcp(safe_row_sum, approx=True)
        norm_scale = (
            mul_ftz_f32(self.output_scale, row_sum_rcp)
            if cutlass.const_expr(cfg.is_dynamic_token_sparse)
            else self.output_scale * row_sum_rcp
        )
        for i in cutlass.range_constexpr(0, 128, 2):
            if cutlass.const_expr(cfg.dsv4_use_source_local_ptx_knobs):
                dsv4_warp_switch()
            scaled = mul_packed_f32x2(
                (qk_acc_regs[i], qk_acc_regs[i + 1]),
                (norm_scale, norm_scale),
            )
            qk_acc_regs[i] = scaled[0]
            qk_acc_regs[i + 1] = scaled[1]

        tile_h = cfg.mma_pv_tiler[0] // cfg.num_mma_ctas
        tile_d = cfg.mma_pv_tiler[1]
        tidx_g = local_tidx & EPILOGUE_THREAD_TILE_MASK
        g_i = tidx_g & EPILOGUE_ROW_MASK
        g_j = (tidx_g >> EPILOGUE_COLUMN_GROUP_SHIFT) * EPILOGUE_THREAD_TILE_THREADS

        logical_num_heads_q = Int32(self.logical_num_heads_q)
        physical_tile_rows = Int32(cfg.mma_qk_tiler[0])
        D = cfg.latent_dim
        head_tile_idx = blk_coord[0]
        seq_q_idx = blk_coord[1]
        batch_idx = blk_coord[2]
        split_kv_idx = blk_coord[3]
        row_in_tile = head_tile_idx * tile_h + g_i
        (
            storage_flat_query_row,
            logical_head_idx,
            logical_q_idx,
            _,
            query_is_valid,
        ) = self._query_row_state(row_in_tile, seq_q_idx, batch_idx)

        if row_in_tile < physical_tile_rows and query_is_valid:
            if cutlass.const_expr(self.partial_output is not None):
                S_q = (
                    cutlass.Int32(self.partial_output.shape[3])
                    if self.partial_output is not None
                    else Int32(1)
                )
                o_base_ptr = (
                    self.partial_output.iterator.raw_ptr()
                    + Int64(row_in_tile) * Int64(self.split_kv) * Int64(D)
                    + Int64(split_kv_idx) * Int64(D)
                    + Int64(seq_q_idx)
                    * Int64(self.split_kv)
                    * Int64(physical_tile_rows)
                    * Int64(D)
                    + Int64(batch_idx)
                    * Int64(physical_tile_rows)
                    * Int64(self.split_kv)
                    * Int64(S_q)
                    * Int64(D)
                )
                output_base = o_base_ptr + iter_n * tile_d + g_j
                for load_idx in cutlass.range_constexpr(num_tmem_loads):
                    for j in cutlass.range_constexpr(4):
                        offset = (
                            load_idx * TCGEN05_32B_REGS_PER_LOAD
                            + j * BF16_OUTPUT_VECTOR_ELEMENTS
                        )
                        vec_f32 = qk_acc_regs.load(offset, BF16_OUTPUT_VECTOR_ELEMENTS)
                        vec_partial = convert_f32_vector_to_bf16_satfinite(
                            vec_f32, BF16_OUTPUT_VECTOR_ELEMENTS
                        )
                        (
                            output_base
                            + load_idx * TCGEN05_32B_REGS_PER_LOAD
                            + j * BF16_OUTPUT_VECTOR_ELEMENTS
                        ).nvvm_store_ext(
                            vec_partial,
                            evict="noallocate",
                        )
            else:
                if cutlass.const_expr(self.cu_seqlens_q is not None):
                    o_base_ptr = self.output.iterator.raw_ptr() + Int64(
                        storage_flat_query_row
                    ) * Int64(D)
                else:
                    S_q = (
                        cutlass.Int32(self.output.shape[2])
                        if self.output is not None
                        else Int32(1)
                    )
                    o_base_ptr = (
                        self.output.iterator.raw_ptr()
                        + Int64(logical_head_idx) * Int64(D)
                        + Int64(logical_q_idx) * Int64(logical_num_heads_q) * Int64(D)
                        + Int64(batch_idx)
                        * Int64(logical_num_heads_q)
                        * Int64(D)
                        * Int64(S_q)
                    )
                output_base = o_base_ptr + iter_n * tile_d + g_j
                for load_idx in cutlass.range_constexpr(num_tmem_loads):
                    for j in cutlass.range_constexpr(2):
                        offset = (
                            load_idx * TCGEN05_32B_REGS_PER_LOAD
                            + j * FP8_OUTPUT_VECTOR_ELEMENTS
                        )
                        vec_f32 = qk_acc_regs.load(offset, FP8_OUTPUT_VECTOR_ELEMENTS)
                        if cutlass.const_expr(cfg.use_fp8_output == 1):
                            packed_o = cutlass.Array(
                                Int32,
                                PACKED_FP8_OUTPUT_REGS,
                                space=cutlass.AddressSpace.rmem,
                            )
                            for pack_idx in cutlass.range_constexpr(
                                PACKED_FP8_OUTPUT_REGS
                            ):
                                pack_offset = pack_idx * PACKED_FP8_OUTPUT_REGS
                                packed_o[pack_idx] = pack_float4_to_fp8_e4m3(
                                    vec_f32[pack_offset],
                                    vec_f32[pack_offset + 1],
                                    vec_f32[pack_offset + 2],
                                    vec_f32[pack_offset + 3],
                                )
                            raw_ptr = cutlass.inttoptr(
                                (
                                    output_base
                                    + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                    + j * FP8_OUTPUT_VECTOR_ELEMENTS
                                ).toint(Int64),
                                mem_space=1,
                                dtype=Int32,
                            )
                            raw_ptr.store(
                                packed_o.load(0, PACKED_FP8_OUTPUT_REGS),
                                alignment=16,
                            )
                        else:
                            if cutlass.const_expr(
                                output_dtype(self.cfg) is cutlass.BFloat16
                            ):
                                vec_o = convert_f32_vector_to_bf16_satfinite(
                                    vec_f32, FP8_OUTPUT_VECTOR_ELEMENTS
                                )
                            else:
                                vec_o = vec_f32.to(output_dtype(self.cfg))
                            (
                                output_base
                                + load_idx * TCGEN05_32B_REGS_PER_LOAD
                                + j * FP8_OUTPUT_VECTOR_ELEMENTS
                            ).nvvm_store_ext(
                                vec_o,
                                evict="noallocate",
                            )

        if cutlass.const_expr(self.stores_lse and iter_n == 0):
            lse_row_sum = row_sum
            if cutlass.const_expr(cfg.is_fp8_qkv()):
                lse_row_sum = lse_row_sum * (
                    self.softmax_p_scale_rcp
                    if cutlass.const_expr(cfg.is_dynamic_token_sparse)
                    else fp8_quant_scale_rcp()
                )
            lse = (
                cute.math.log2(lse_row_sum, fastmath=True)
                + self.softmax_scale_log2 * row_max
                if row_has_values
                else Float32(-Float32.inf)
            )
            lse_tidx = local_tidx
            if lse_tidx < tile_h:
                lse_row_in_tile = head_tile_idx * tile_h + lse_tidx
                (
                    storage_flat_lse_row,
                    logical_lse_head_idx,
                    logical_lse_q_idx,
                    _,
                    lse_query_is_valid,
                ) = self._query_row_state(lse_row_in_tile, seq_q_idx, batch_idx)
                if lse_row_in_tile < physical_tile_rows and lse_query_is_valid:
                    if cutlass.const_expr(self.partial_lse is not None):
                        S_q = (
                            cutlass.Int32(self.partial_lse.shape[2])
                            if self.partial_lse is not None
                            else Int32(1)
                        )
                        lse_base_ptr = (
                            self.partial_lse.iterator.raw_ptr()
                            + Int64(lse_row_in_tile) * Int64(self.split_kv)
                            + Int64(split_kv_idx)
                            + Int64(seq_q_idx)
                            * Int64(physical_tile_rows)
                            * Int64(self.split_kv)
                            + Int64(batch_idx)
                            * Int64(physical_tile_rows)
                            * Int64(self.split_kv)
                            * Int64(S_q)
                        )
                        lse_base_ptr.store(lse)
                    elif cutlass.const_expr(self.lse is not None):
                        if cutlass.const_expr(self.cu_seqlens_q is not None):
                            lse_base_ptr = (
                                self.lse.iterator.raw_ptr() + storage_flat_lse_row
                            )
                        else:
                            S_q = (
                                cutlass.Int32(self.lse.shape[1])
                                if self.lse is not None
                                else Int32(1)
                            )
                            lse_base_ptr = (
                                self.lse.iterator.raw_ptr()
                                + Int64(logical_lse_head_idx)
                                + Int64(logical_lse_q_idx) * Int64(logical_num_heads_q)
                                + Int64(batch_idx)
                                * Int64(logical_num_heads_q)
                                * Int64(S_q)
                            )
                        lse_base_ptr.store(lse)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()
