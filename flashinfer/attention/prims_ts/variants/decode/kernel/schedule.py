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

"""The decode task schedule: HEAD, LOOP and TAIL sections of every task.

``build_decode_gen_schedule`` orders the resource work of the load, MMA,
softmax, correction and scheduler tasks for one kernel configuration.
"""

import cutlass.cute as cute
import cutlass
import math
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float32, Int32, Int64
from cutlass.experimental.task_scheduling.enums import PipelineType, SignalingThreads
from cutlass.experimental.task_scheduling.memory import (
    SmemAllocation,
    SmemAllocator,
    TmemAllocator,
)
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    TileSchedulerConfig,
)
from cutlass.experimental.task_scheduling.task import Task
from ...block_sparse.prepared import BlockSparseRouteLayout
from ....common.hardware import is_sm107_compile_target
from .config import FmhaDecodeConfig
from .device.constants import (
    BYTES_PER_KIB,
    KV_KIND_K,
    KV_KIND_V,
    MAX_KV_STAGE_SMEM_KIB,
    SMEM_CAPACITY_KIB,
)
from .device.direct_sparse_metadata import DirectSparseMetadataView
from .resources import (
    SmemBlockSparseKvMetadataResource,
    SmemBlockSparseSoftmaxMetadataResource,
    SmemKvResource,
    SmemKvTileResource,
    SmemPResource,
    SmemPageOffsetsKvResource,
    SmemQResource,
    SmemTransformedKvResource,
    TmemCorrResource,
    TmemOResource,
    TmemSResource,
    TmemSoftmaxGlobalResource,
    TmemSoftmaxLocalResource,
    TmemSoftmaxOrderResource,
    TmemStatsDoneResource,
    TmemTransformedKvResource,
)
from .device.q_rows import UniformQueryOffsets
from .resources.sage_scales import SageKScalesResource, SageVScalesResource
from .resources.tmem_transformed_kv import (
    SmemKvFromNvfp4Resource,
    SmemNvfp4RawKvResource,
)
from .tasks.common import PackedDecodeWorkQueue, can_hold_native_page_window
from .resources.sync import (
    ScheduleTokenThrottleResource,
    SparseMembershipLifetimeResource,
)
from .tasks.correction import (
    create_correction_task,
    create_correction_task_one_inst_qkv,
)
from .tasks.load import (
    create_block_sparse_load_tasks_per_inst,
    create_load_task,
    create_load_task_one_inst_qkv,
    create_load_task_split_kv,
)
from .tasks.mma import (
    create_mma_task,
    create_mma_task_dual_transform_kv,
    create_mma_task_one_inst_qkv,
    create_mma_task_split_kv,
)
from .tasks.page_offsets import (
    create_page_offsets_task,
    create_page_offsets_task_one_inst_qkv,
    create_page_offsets_task_split_kv,
)
from .tasks.scheduler import create_padding_task, create_scheduler_task
from .tasks.softmax import create_softmax0_task, create_softmax1_task
from .tasks.transform_kv import (
    create_kv_operand_transform_task,
    create_nvfp4_keeps_transform_task,
    create_transform_kv_task,
)


PERSISTENT_SCHEDULE_TOKEN_STAGES = 2
# Double-buffer CLC responses independently of the schedule-token throttle.
# A single response slot triggers register spills in grouped sparse work loops.
WORK_QUEUE_STAGES = 2


def build_decode_gen_schedule(
    cfg: FmhaDecodeConfig,
    total_kv_tiles: int | Int32,
    scale_softmax_log2: Float32 | None = None,
    o_ptr: cute.Pointer | None = None,
    output_scale: Float32 | None = None,
    partial_o_ptr: cute.Pointer | None = None,
    partial_stats_ptr: cute.Pointer | None = None,
    split_kv_counter_ptr: cute.Pointer | None = None,
    attention_sinks_ptr: cute.Pointer | None = None,
    seqlens_kv: cute.Pointer | None = None,
    cu_seqlens_q: cute.Pointer | UniformQueryOffsets | None = None,
    max_seq_len_kv: int | Int32 = 0,
    corr_max_seq_len_kv: int | Int32 | None = None,
    num_heads_kv: Int32 | None = None,
    h_r: Int32 | None = None,
    max_q_len: int | Int32 = 1,
    tma_desc_q: cutlass.Pointer | None = None,
    tma_desc_k: cutlass.Pointer | None = None,
    tma_desc_v: cutlass.Pointer | None = None,
    tma_desc_k_sf: cutlass.Pointer | None = None,
    tma_desc_v_sf: cutlass.Pointer | None = None,
    tma_desc_k_atom: cutlass.Pointer | None = None,
    tma_desc_v_atom: cutlass.Pointer | None = None,
    tma_desc_k_summary: cutlass.Pointer | None = None,
    tma_desc_v_summary: cutlass.Pointer | None = None,
    tma_desc_k_summary_atom: cutlass.Pointer | None = None,
    tma_desc_v_summary_atom: cutlass.Pointer | None = None,
    page_idx_kv: cute.Pointer | None = None,
    page_table_stride: Int64 | None = None,
    page_table_capacity: Int32 | None = None,
    q_token_kv_block_sparse_page_memberships: cute.Pointer | None = None,
    q_token_kv_block_sparse_page_membership_stride: Int32 | None = None,
    h_k_idx: Int32 | None = None,
    b_idx: Int32 | None = None,
    q_group_idx: Int32 | None = None,
    q_token_offset: Int32 | None = None,
    seq_len_q: Int32 | None = None,
    active_splits_kv: Int32 | None = None,
    static_full_split_prefix: bool = False,
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams | None = None,
    use_variable_seqlens_kv: bool = False,
    use_native_paged_kv: bool = False,
    use_static_native_seqlens_kv: bool = False,
    sparse_row_route_offsets: cute.Pointer | None = None,
    sparse_row_route_counts: cute.Pointer | None = None,
    sparse_route_metadata: cute.Pointer | None = None,
    sparse_row_route_begin: Int32 | None = None,
    sparse_route_count: Int32 | None = None,
    sage_scales_ptr: cute.Pointer | None = None,
    lse_ptr: cute.Pointer | None = None,
    requested_splits=None,
) -> tuple[
    list[Task],
    dict[MemoryResource, list[MemoryResource]],
    dict[tuple[MemoryResource, MemoryResource], set[str]],
    SmemAllocator,
    TmemAllocator,
    list[MemoryResource],
]:
    """Build all resources, tasks, and dep graph.

    Parameters
    ----------
    cfg : FmhaDecodeConfig
        Kernel configuration.
    total_kv_tiles : int or Int32
        Total number of KV tiles (seqLenKv / tileSizeKv).
    tma_desc_q/k/v : TMA descriptor pointers (None for validation-only mode).
    h_k_idx, b_idx, q_group_idx : Grid coordinates (None for validation-only mode).
    corr_max_seq_len_kv : Bound passed to TmemCorrResource; defaults to
        ``max_seq_len_kv``. The GPU kernel uses the constexpr ``seq_len_kv``
        here (full sequence) while other resources use the static/varlen
        runtime value.

    Returns
    -------
    tuple
        (task_list, dep_graph, dma labels, smem_allocator, tmem_allocator,
        eager_init_resources)
    """
    if cfg.use_block_sparse and cfg.num_insts_kv != 2:
        raise ValueError("block-sparse attention requires num_insts_kv == 2")
    if cfg.use_keeps_mma_ab and cfg.num_insts_kv == 1 and not cfg.uses_tmem_p:
        raise ValueError(
            "one-instance KeepsMmaAb is enabled only for the staged headDim=256 "
            "profile with D128 bands (or D256 for sparse FP8) and o_stages=1"
        )
    if use_native_paged_kv and not cfg.use_paged_kv:
        raise ValueError("native paged-KV ABI requires cfg.use_paged_kv=True")
    if use_native_paged_kv and (
        page_idx_kv is None or page_table_capacity is None or page_table_stride is None
    ):
        raise ValueError(
            "native paged-KV ABI requires block tables, capacity, and row stride"
        )
    if cfg.uses_scattered_page_route and not use_native_paged_kv:
        raise ValueError("scattered page routes require the native paged-KV ABI")
    if (
        cfg.uses_q_token_kv_block_sparse_page_membership
        and tma_desc_q is not None
        and q_token_kv_block_sparse_page_memberships is None
    ):
        raise ValueError(
            "grouped QToken-KvBlock-Sparse-Attention kernel construction requires q_token_kv_block_sparse_page_memberships"
        )
    if cfg.use_paged_kv:
        cfg.validate_paged_kv_staging_config()
    if cfg.use_block_sparse:
        if tma_desc_q is not None:
            if sparse_row_route_offsets is None:
                raise ValueError(
                    "sparse_row_route_offsets is required for block-sparse kernel "
                    "construction"
                )
            if sparse_row_route_counts is None:
                raise ValueError(
                    "sparse_row_route_counts is required for block-sparse kernel "
                    "construction"
                )
            if sparse_route_metadata is None:
                raise ValueError(
                    "sparse_route_metadata is required for block-sparse kernel "
                    "construction"
                )
            segment_tensormaps = {
                "tma_desc_k_atom": tma_desc_k_atom,
                "tma_desc_v_atom": tma_desc_v_atom,
            }
            if cfg.use_block_sparse_proxy_routes:
                segment_tensormaps.update(
                    {
                        "tma_desc_k_summary": tma_desc_k_summary,
                        "tma_desc_v_summary": tma_desc_v_summary,
                        "tma_desc_k_summary_atom": tma_desc_k_summary_atom,
                        "tma_desc_v_summary_atom": tma_desc_v_summary_atom,
                    }
                )
            for name, descriptor in segment_tensormaps.items():
                if descriptor is None:
                    raise ValueError(
                        f"{name} is required for block-sparse kernel construction"
                    )
            if num_heads_kv is None:
                raise ValueError(
                    "num_heads_kv is required for block-sparse kernel construction"
                )
    if corr_max_seq_len_kv is None:
        corr_max_seq_len_kv = max_seq_len_kv
    if h_k_idx is None:
        h_k_idx = Int32(0)
    if b_idx is None:
        b_idx = Int32(0)
    if q_group_idx is None:
        q_group_idx = Int32(0)
    if q_token_offset is None:
        q_token_offset = Int32(0)
    if seq_len_q is None:
        seq_len_q = Int32(cfg.max_seq_len_q)

    WARP_SIZE = 32
    Agent = pipeline.Agent
    cta_layout = (1, 1, 1, 1)

    # ------------------------------------------------------------------
    # Cooperative groups
    # ------------------------------------------------------------------
    tma_producer = pipeline.CooperativeGroup(Agent.Thread)
    page_offsets_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.page_offsets_num_warps * WARP_SIZE
    )
    load_grp = pipeline.CooperativeGroup(Agent.Thread, cfg.load_num_warps * WARP_SIZE)
    if cfg.use_transform_kv:
        transform_kv_grp = pipeline.CooperativeGroup(
            Agent.Thread, cfg.transform_kv_num_warps * WARP_SIZE
        )
    umma_hw = pipeline.CooperativeGroup(Agent.Thread)
    # The staged one-instance S/P overlay uses this group for overwrite credit.
    mma_grp = pipeline.CooperativeGroup(Agent.Thread, cfg.mma_num_warps * WARP_SIZE)
    softmax0_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.softmax0_num_warps * WARP_SIZE
    )
    softmax1_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.softmax1_num_warps * WARP_SIZE
    )
    correction_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.correction_num_warps * WARP_SIZE
    )
    scheduler_grp = pipeline.CooperativeGroup(
        Agent.Thread, cfg.scheduler_num_warps * WARP_SIZE
    )
    tma_producer_signaling = (
        SignalingThreads.TaskWarpLeader
        if cfg.load_num_warps > 1
        else SignalingThreads.All
    )
    q_bytes_per_load_warp = (
        cfg.smem_q_tile_bytes // cfg.load_num_warps if cfg.load_num_warps > 1 else None
    )

    # ------------------------------------------------------------------
    # Pipeline configs
    # ------------------------------------------------------------------
    # Leave barrier_ptr unset so SmemAllocator packs every pipeline barrier
    # into the unified block. Separate barrier arrays create an alignment gap
    # before the 1024-byte-aligned data block and overflow near-capacity Q128.
    use_one_inst_kv = cfg.num_insts_kv == 1
    uses_page_offsets_task = cfg.uses_page_offsets_task
    use_one_inst_qkv = cfg.use_keeps_mma_ab and cfg.num_insts_kv == 1
    # K and V of different byte widths take the split-resource paths with one
    # K ring and one V ring, each shared by both K/V instances; equal widths
    # (including Int8 K with E4M3 V) share one ring.
    use_shared_inst_kv_rings = (
        cfg.k_dtype_bytes != cfg.v_dtype_bytes and cfg.tile_size_kv != 256
    ) or cfg.uses_dual_kv_transform
    one_inst_tmem_stages = 2 if use_one_inst_qkv else 1
    # Keep complete head-dimension rounds in each balanced K/V ring. FP8's
    # smaller stage tiles can prefetch farther without increasing the budget.
    one_inst_kv_stages = (
        max(
            cfg.num_head_dim_stages_kv,
            cfg.kv_stages
            // (2 * cfg.num_head_dim_stages_kv)
            * cfg.num_head_dim_stages_kv,
        )
        if use_one_inst_qkv
        else 1
    )
    use_distributed_split_kv_stages = not use_one_inst_qkv
    if cfg.tile_size_q == 128 and use_distributed_split_kv_stages:
        # Q128's four instruction-local K0/K1/V0/V1 rings need equal depth.
        # Round the inferred aggregate budget down to a complete balanced set;
        # on the FP8 decode profile this is 2/2/2/2, matching the roughly
        # 165-KiB staged footprint of the corresponding reference profile.
        balanced_total_stages = max(
            (cfg.kv_stages // (2 * cfg.num_insts_kv)) * cfg.num_insts_kv,
            cfg.num_insts_kv,
        )
        split_total_k_stages = balanced_total_stages
        split_total_v_stages = balanced_total_stages
    else:
        split_total_k_stages = (
            max(cfg.kv_stages // 2, cfg.num_insts_kv)
            if use_distributed_split_kv_stages
            else cfg.num_insts_kv
        )
        split_total_v_stages = (
            max(cfg.kv_stages - split_total_k_stages, cfg.num_insts_kv)
            if use_distributed_split_kv_stages
            else cfg.num_insts_kv
        )
    split_k0_stages = (
        one_inst_kv_stages
        if use_one_inst_qkv
        else max((split_total_k_stages + cfg.num_insts_kv - 1) // cfg.num_insts_kv, 1)
    )
    split_k1_stages = (
        1
        if use_one_inst_qkv
        else max((split_total_k_stages + cfg.num_insts_kv - 2) // cfg.num_insts_kv, 1)
    )
    split_v0_stages = (
        one_inst_kv_stages
        if use_one_inst_qkv
        else max((split_total_v_stages + cfg.num_insts_kv - 1) // cfg.num_insts_kv, 1)
    )
    split_v1_stages = (
        1
        if use_one_inst_qkv
        else max((split_total_v_stages + cfg.num_insts_kv - 2) // cfg.num_insts_kv, 1)
    )
    if use_shared_inst_kv_rings and not use_one_inst_qkv:
        # One K and one V stage per unit of depth.
        shared_inst_kv_stages = (
            max(cfg.kv_stages // 2, 1)
            if cfg.uses_dual_kv_transform
            else max(
                (MAX_KV_STAGE_SMEM_KIB * BYTES_PER_KIB)
                // (cfg.smem_k_tile_bytes + cfg.smem_v_tile_bytes),
                1,
            )
        )
        split_k0_stages = shared_inst_kv_stages
        split_v0_stages = shared_inst_kv_stages
    use_ordered_softmax_barrier = (
        not use_one_inst_kv and cfg.uses_ordered_softmax_barrier
    )
    # A two-inst Keeps profile can use the deeper shared K/V FIFO when stats
    # are standalone and P remains in SMEM.  Keep instruction-local FIFOs when
    # stats or TMEM-P alias S: their overwrite-credit cadence is tied to each
    # instruction. Dense Swaps uses the shared FIFO, including staged H256.
    # Sparse KV128 keeps instruction-local rings in either MMA orientation;
    # sparse KV256 reuses the shared data ring at its element-width-derived
    # depth while retaining instruction-local route metadata. The load warp
    # issues V(route R) before replacing that metadata with route R+1, so its
    # lifetime remains independent of the K/V data-ring depth.
    # With cfg.keeps_stats_via_smem the stats-alias justification no longer
    # applies, but the shared FIFO still causes a material Q128 regression, so
    # the instruction-local FIFO gate remains part of that kernel policy.
    use_per_inst_kv_resources = (
        use_one_inst_qkv
        or (cfg.use_block_sparse and cfg.tile_size_kv != 256)
        or use_shared_inst_kv_rings
        or (
            cfg.use_keeps_mma_ab
            and cfg.tile_size_kv != 256
            and (not cfg.keeps_separates_tmem_s_and_stats or cfg.uses_two_inst_tmem_p)
        )
    )
    # B8/B16 issue enough fine-grained TMA copies to benefit from reusing a
    # padding warp as a second issuer. The host policy applies one KV-side
    # crossover across all two-instance Swaps Q tiles.
    supports_per_inst_block_sparse_load_tasks = (
        cfg.use_block_sparse
        and cfg.use_parallel_sparse_kv_loads
        and not cfg.use_keeps_mma_ab
        and cfg.kv_block_size in (8, 16)
        and cfg.num_insts_kv == 2
    )
    # Independent K0/K1/V0/V1 data rings share identical native sparse locators.
    # A held route is published once and retained through the final V issue;
    # only schedules that replace locators per tile need separate K/V state.
    use_separate_kv_page_offset_resources = (
        uses_page_offsets_task
        and use_per_inst_kv_resources
        and not use_one_inst_qkv
        and not cfg.runtime_page_window
        and not (
            use_native_paged_kv
            and cfg.uses_q_token_kv_block_sparse_page_route
            and cfg.uses_held_encoded_locator_window
        )
    )
    # Paired resources publish independent K0/K1 and V0/V1 stages. Shared
    # split-KV retains the aligned 32-ID representation for its optional
    # native held-window path.
    stage_page_ids_per_tile = stages_page_ids_per_tile(
        use_separate_kv_page_offset_resources,
    )

    smem_q_cfg = PipelineConfig.create_tma_umma_pipeline_cfg(
        num_stages=cfg.q_stages,
        num_bytes=cfg.smem_q_tile_bytes,
        producer_group=tma_producer,
        consumer_group=umma_hw,
        cta_layout_vmnk=cta_layout,
        producer_signaling_threads=tma_producer_signaling,
        num_bytes_per_warp_per_cta=q_bytes_per_load_warp,
        advance_on_wait=True,
    )

    def make_raw_kv_cfg(
        num_stages: int, tile_bytes: int, is_v: bool = False
    ) -> PipelineConfig:
        """Create a per-operand raw pipeline with its actual TMA byte count."""
        num_bytes = tile_bytes + (
            cfg.sf_stage_transaction_bytes(is_v)
            if cfg.use_nvfp4_kv
            else cfg.smem_kv_sf_tile_transaction_bytes
        )
        bytes_per_load_warp = (
            num_bytes // cfg.load_num_warps if cfg.load_num_warps > 1 else None
        )
        if cfg.uses_decode_kv_converter:
            # NVFP4 payload and scale factors complete the same raw barrier.
            return PipelineConfig(
                num_stages=num_stages,
                num_bytes=num_bytes,
                producer_group=tma_producer,
                consumer_group=transform_kv_grp,
                pipeline_type=PipelineType.TmaAsync,
                cta_layout_vmnk=cta_layout,
                producer_signaling_threads=tma_producer_signaling,
                num_bytes_per_warp_per_cta=bytes_per_load_warp,
                advance_on_wait=True,
            )
        return PipelineConfig.create_tma_umma_pipeline_cfg(
            num_stages=num_stages,
            num_bytes=num_bytes,
            producer_group=tma_producer,
            consumer_group=umma_hw,
            cta_layout_vmnk=cta_layout,
            producer_signaling_threads=tma_producer_signaling,
            num_bytes_per_warp_per_cta=bytes_per_load_warp,
            advance_on_wait=True,
        )

    smem_kv_cfg = None
    if not use_per_inst_kv_resources:
        smem_kv_cfg = make_raw_kv_cfg(cfg.kv_stages, cfg.smem_kv_tile_bytes)
    smem_transformed_kv_cfg = None
    if cfg.uses_decode_kv_converter:
        smem_transformed_kv_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.transformed_kv_stages,
            producer_group=transform_kv_grp,
            consumer_group=umma_hw,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )
    smem_transformed_k_cfg = smem_transformed_v_cfg = smem_transformed_kv_cfg
    if (
        cfg.uses_dual_smem_transform
        and cfg.transformed_v_stages > 0
        and cfg.transformed_v_stages != cfg.transformed_kv_stages
    ):
        smem_transformed_v_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
            num_stages=cfg.transformed_v_stages,
            producer_group=transform_kv_grp,
            consumer_group=umma_hw,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )
    smem_k0_cfg = make_raw_kv_cfg(split_k0_stages, cfg.smem_k_tile_bytes)
    smem_k1_cfg = make_raw_kv_cfg(split_k1_stages, cfg.smem_k_tile_bytes)
    smem_v0_cfg = make_raw_kv_cfg(split_v0_stages, cfg.smem_v_tile_bytes, True)
    smem_v1_cfg = make_raw_kv_cfg(split_v1_stages, cfg.smem_v_tile_bytes, True)
    smem_kv_raw_cfg = None
    if cfg.uses_keeps_transformed_kv:
        # The loader fills a raw NVFP4 ring for TransformKvTask, which in turn
        # produces the FP8 or BF16 K and V rings MMA consumes. Separate scale
        # tensors complete the same raw barrier as the payload.
        assert use_one_inst_qkv and cfg.transformed_kv_stages > 0
        raw_tile_bytes = cfg.smem_kv_tile_bytes + cfg.smem_kv_sf_tile_transaction_bytes
        smem_kv_raw_cfg = PipelineConfig(
            num_stages=cfg.kv_stages,
            num_bytes=raw_tile_bytes,
            producer_group=tma_producer,
            consumer_group=transform_kv_grp,
            pipeline_type=PipelineType.TmaAsync,
            cta_layout_vmnk=cta_layout,
            producer_signaling_threads=tma_producer_signaling,
            num_bytes_per_warp_per_cta=(
                raw_tile_bytes // cfg.load_num_warps if cfg.load_num_warps > 1 else None
            ),
            advance_on_wait=True,
        )
        smem_k0_cfg, smem_v0_cfg = (
            PipelineConfig.create_async_umma_pipeline_cfg(
                num_stages=cfg.transformed_kv_stages,
                producer_group=transform_kv_grp,
                consumer_group=umma_hw,
                cta_layout_vmnk=cta_layout,
                advance_on_wait=True,
            )
            for _ in range(2)
        )

    def make_page_offsets_cfg(num_stages: int | None = None) -> PipelineConfig:
        """Create the async page-offsets pipeline for the selected stage count."""
        if num_stages is None:
            num_stages = cfg.page_offsets_stages
        return PipelineConfig(
            num_stages=num_stages,
            num_bytes=0,
            producer_group=page_offsets_grp,
            consumer_group=load_grp,
            pipeline_type=PipelineType.AsyncAsync,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )

    smem_page_offsets_cfg = None
    smem_page_offsets_v_cfg = None
    if uses_page_offsets_task:
        page_offsets_stages = (
            3 if use_separate_kv_page_offset_resources else cfg.page_offsets_stages
        )
        if use_native_paged_kv and cfg.uses_held_encoded_locator_window:
            # QToken-KvBlock-Sparse-Attention publishes its complete CTA-local locator span once per work
            # tile and holds it through the K/V tail. Only one stage is live
            # for direct, persistent, and split-KV routes alike.
            page_offsets_stages = 1
        smem_page_offsets_cfg = make_page_offsets_cfg(page_offsets_stages)
        if use_separate_kv_page_offset_resources:
            smem_page_offsets_v_cfg = make_page_offsets_cfg(page_offsets_stages)
    sparse_softmax_metadata0_cfg = None
    sparse_softmax_metadata1_cfg = None
    # Load publishes a block-sparse route, and a Sage plan's K scales into the
    # instance's ``sfK`` ring, right after the K tile. Softmax releases the
    # route when it starts that tile and the scales after the tile's last pass
    # that reads them. When the K/V ring frees a slot for an instance's next K
    # tile, the oldest buffered K tile's QK has run, after the previous tile's
    # P left the TMEM columns it aliases or, with SMEM P, after the previous
    # tile's max pass released S. A route or ``sfK`` stage is read no later
    # than that pass (an SMEM-P Sage row is dequantized there), so the
    # instance's Softmax has released every stage older than the K tiles the
    # ring buffers per instance; one stage more than those K tiles never
    # stalls the load warp on either pipeline.
    if use_per_inst_kv_resources:
        inst_k_tiles = max(split_k0_stages, split_k1_stages)
    else:
        # One ring cycles K and V tiles of every instance.
        inst_k_tiles = math.ceil(cfg.kv_stages / (2 * cfg.num_insts_kv))
    k_tile_stages = inst_k_tiles + 1
    if cfg.use_block_sparse:
        route_metadata_stages = k_tile_stages
        sparse_softmax_metadata0_cfg = PipelineConfig(
            num_stages=route_metadata_stages,
            num_bytes=0,
            producer_group=load_grp,
            consumer_group=softmax0_grp,
            pipeline_type=PipelineType.AsyncAsync,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )
        sparse_softmax_metadata1_cfg = PipelineConfig(
            num_stages=route_metadata_stages,
            num_bytes=0,
            producer_group=load_grp,
            consumer_group=softmax1_grp,
            pipeline_type=PipelineType.AsyncAsync,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )
    # tmem_s0/s1, smem_p0/p1, tmem_o, softmax_local cfgs go through direct
    # PipelineConfig() so advance_on_wait=True can be set (factories don't
    # expose it).
    tmem_s0_cfg = PipelineConfig(
        num_stages=one_inst_tmem_stages,
        num_bytes=0,
        producer_group=umma_hw,
        consumer_group=softmax0_grp,
        pipeline_type=PipelineType.UmmaAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    tmem_s1_cfg = PipelineConfig(
        num_stages=1,
        num_bytes=0,
        producer_group=umma_hw,
        consumer_group=softmax1_grp,
        pipeline_type=PipelineType.UmmaAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    smem_p0_cfg = None
    smem_p1_cfg = None
    if not cfg.streams_tmem_p_fragments:
        smem_p0_cfg = PipelineConfig(
            num_stages=one_inst_tmem_stages,
            num_bytes=0,
            producer_group=softmax0_grp,
            consumer_group=umma_hw,
            pipeline_type=PipelineType.AsyncUmma,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )
        smem_p1_cfg = PipelineConfig(
            num_stages=one_inst_tmem_stages,
            num_bytes=0,
            producer_group=softmax1_grp,
            consumer_group=umma_hw,
            pipeline_type=PipelineType.AsyncUmma,
            cta_layout_vmnk=cta_layout,
            advance_on_wait=True,
        )
    tmem_o_cfg = PipelineConfig(
        num_stages=cfg.o_stages,
        num_bytes=0,
        producer_group=umma_hw,
        consumer_group=correction_grp,
        pipeline_type=PipelineType.UmmaAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )
    softmax_local0_cfg = PipelineConfig(
        num_stages=one_inst_tmem_stages,
        num_bytes=0,
        producer_group=softmax0_grp,
        consumer_group=correction_grp,
        pipeline_type=PipelineType.AsyncAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )

    softmax_local1_cfg = PipelineConfig(
        num_stages=1,
        num_bytes=0,
        producer_group=softmax1_grp,
        consumer_group=correction_grp,
        pipeline_type=PipelineType.AsyncAsync,
        cta_layout_vmnk=cta_layout,
        advance_on_wait=True,
    )

    # The single MMA issuer orders PV before QK reuses the same S/P slot.
    # PTX's same-warp mma(A-read) -> mma(D-write) pipeline protects that alias.
    # Separate SMEM stats have their own softmax-local handoff, just as in
    # two-instance Keeps; correction need not additionally gate S reuse.
    stats_done0_cfg = None
    stats_done1_cfg = None
    resource_dependency_graph: dict[MemoryResource, list[MemoryResource]]
    if use_one_inst_qkv and (not cfg.keeps_stats_via_smem or cfg.mma_num_warps != 1):
        stats_done0_cfg = PipelineConfig.create_async_async_pipeline_cfg(
            num_stages=one_inst_tmem_stages,
            producer_group=mma_grp,
            consumer_group=correction_grp,
            cta_layout_vmnk=cta_layout,
        )

    # tmemSoftmaxGlobal, tmemCorr: no pipeline (pipeline_config=None)

    # ------------------------------------------------------------------
    # Create resources
    # ------------------------------------------------------------------
    work_queue = None
    schedule_token_throttle = None
    # CLC remains the single persistent policy for every supported topology.
    # The stock static WorkQueue advances and decodes coordinates separately
    # in every task, which regresses multi-wave decode workloads. CLC computes
    # each schedule token once on the scheduler warp and broadcasts it to the workers.
    use_clc_dynamic = cfg.use_persistent_scheduler
    per_inst_block_sparse_load_topology = None
    if supports_per_inst_block_sparse_load_tasks:
        # Dual issuers are a performance choice. If a future recipe has no
        # compatible idle-warp placement, the common one-warp load task remains
        # correct and consumes the same disjoint K/V metadata pipelines.
        per_inst_block_sparse_load_topology = (
            resolve_block_sparse_per_inst_load_topology(
                cfg,
                use_clc_dynamic=use_clc_dynamic,
            )
        )
    use_per_inst_block_sparse_load_tasks = (
        per_inst_block_sparse_load_topology is not None
    )
    if use_clc_dynamic:
        num_consumer_threads = cfg.threads_per_cta
        wq_pipeline_config = PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=WORK_QUEUE_STAGES,
            num_bytes=16,
            producer_group=pipeline.CooperativeGroup(Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                Agent.Thread, num_consumer_threads
            ),
            cta_layout_vmnk=cta_layout,
        )
        # The scheduler config needs the CLC response slots' address; it is
        # bound once the unified SMEM layout below is computed.
        work_queue_kwargs = {
            "tile_scheduler_config": None,
            "pipeline_config": wq_pipeline_config,
            "name": "work_queue",
        }
        work_queue = PackedDecodeWorkQueue(
            cfg=cfg,
            cu_seqlens_q=cu_seqlens_q,
            seqlens_kv=seqlens_kv,
            num_heads_kv=num_heads_kv,
            **work_queue_kwargs,
        )

        schedule_token_throttle = ScheduleTokenThrottleResource(
            pipeline_config=PipelineConfig.create_async_async_pipeline_cfg(
                num_stages=PERSISTENT_SCHEDULE_TOKEN_STAGES,
                producer_group=load_grp,
                consumer_group=scheduler_grp,
                cta_layout_vmnk=cta_layout,
            ),
            name="schedule_token_throttle",
        )
    # Static-length routes and runtime query-major planes stage memberships in
    # SMEM (a bounded prefix with a GMEM fallback for the latter); runtime
    # page-major rows are read from immutable GMEM on demand.
    cache_memberships_in_smem = cfg.fixed_seqlen or cfg.query_major_memberships
    membership_lifetime = None
    if (
        use_clc_dynamic
        and cfg.uses_q_token_kv_block_sparse_page_membership
        and cache_memberships_in_smem
    ):
        # Page IDs are consumed by Load, but membership bytes are consumed by
        # Softmax. Do not overwrite the next work item's membership row until
        # every score stream has finished reading the current one. Rows read
        # from GMEM need no lifetime, so the page-offset producer may start
        # the next work item early.
        membership_lifetime = SparseMembershipLifetimeResource(
            name="sparseMembershipLifetime",
            pipeline_config=PipelineConfig(
                num_stages=1,
                num_bytes=0,
                producer_group=page_offsets_grp,
                consumer_group=pipeline.CooperativeGroup(
                    Agent.Thread,
                    WARP_SIZE
                    * (
                        cfg.softmax0_num_warps
                        + (0 if use_one_inst_kv else cfg.softmax1_num_warps)
                    ),
                ),
                pipeline_type=PipelineType.AsyncAsync,
                cta_layout_vmnk=cta_layout,
                advance_on_wait=True,
            ),
        )
    smem_q = SmemQResource(
        pipeline_config=smem_q_cfg,
        cfg=cfg,
        tma_desc_q=tma_desc_q,
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        q_group_idx=q_group_idx,
        q_token_offset=q_token_offset,
        seq_len_q=seq_len_q,
        name="smemQ",
    )
    # Native dense page tables pair fixed-capacity rows with one canonical
    # sequence-length tensor, so native mode reuses the existing
    # variable-length domain, split, sliding-window, and masking paths.
    use_runtime_seqlens_kv = use_variable_seqlens_kv or (
        use_native_paged_kv and not use_static_native_seqlens_kv
    )
    kv_seqlens = seqlens_kv if use_runtime_seqlens_kv else None
    smem_page_offsets = None
    smem_page_offsets_v = None
    membership_plane_stride = None
    if cutlass.const_expr(cfg.page_membership_planes > 1):
        # Plane p of a G9-G32 route row starts p * (row words / planes) words
        # in. A widened per-head stride spans Hkv rows; planes stay inside one
        # head's row, so undo the widening once per CTA.
        assert not isinstance(page_idx_kv, DirectSparseMetadataView)
        membership_row_words = q_token_kv_block_sparse_page_membership_stride
        if cutlass.const_expr(cfg.widens_sparse_metadata_rows_per_head):
            membership_row_words = membership_row_words // num_heads_kv
        membership_plane_stride = membership_row_words // Int32(
            cfg.page_membership_planes
        )
    if uses_page_offsets_task:
        smem_page_offsets = SmemPageOffsetsKvResource(
            pipeline_config=smem_page_offsets_cfg,
            cfg=cfg,
            stage_page_ids_per_tile=stage_page_ids_per_tile,
            # Runtime query planes use a bounded prefix cache with a GMEM
            # fallback, so its allocation never specializes sequence lengths.
            cache_memberships_in_smem=cache_memberships_in_smem,
            page_idx_kv=page_idx_kv,
            page_table_stride=page_table_stride,
            num_heads_kv=num_heads_kv,
            q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
            q_token_kv_block_sparse_page_membership_stride=q_token_kv_block_sparse_page_membership_stride,
            q_token_kv_block_sparse_page_membership_plane_stride=membership_plane_stride,
            seqlens_kv=kv_seqlens,
            use_native_paged_kv=use_native_paged_kv,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            name=(
                "smemPageOffsetsKvK"
                if use_separate_kv_page_offset_resources
                else "smemPageOffsetsKv"
            ),
        )
        if use_separate_kv_page_offset_resources:
            smem_page_offsets_v = SmemPageOffsetsKvResource(
                pipeline_config=smem_page_offsets_v_cfg,
                cfg=cfg,
                stage_page_ids_per_tile=stage_page_ids_per_tile,
                # Softmax consumes only the K-side membership view.
                cache_memberships_in_smem=False,
                page_idx_kv=page_idx_kv,
                page_table_stride=page_table_stride,
                num_heads_kv=num_heads_kv,
                q_token_kv_block_sparse_page_memberships=q_token_kv_block_sparse_page_memberships,
                q_token_kv_block_sparse_page_membership_stride=q_token_kv_block_sparse_page_membership_stride,
                q_token_kv_block_sparse_page_membership_plane_stride=membership_plane_stride,
                seqlens_kv=kv_seqlens,
                use_native_paged_kv=use_native_paged_kv,
                max_seq_len_kv=max_seq_len_kv,
                h_k_idx=h_k_idx,
                b_idx=b_idx,
                q_group_idx=q_group_idx,
                seq_len_q=seq_len_q,
                name="smemPageOffsetsKvV",
            )
    sparse_kv_metadata0 = None
    sparse_kv_metadata1 = None
    sparse_softmax_metadata0 = None
    sparse_softmax_metadata1 = None
    if cfg.use_block_sparse:
        # This selects the prepared-record storage ABI. Causal consumers still
        # intersect these column-validity words with each Q row's causal mask.
        prepared_route_layout = BlockSparseRouteLayout.create(
            kv_route_size=cfg.tile_size_kv,
            kv_block_size=cfg.kv_block_size,
            has_token_bits=cfg.uses_prepared_score_keep_words,
            route_metadata_capacity=0,
            num_rows=1,
            page_size=cfg.num_tokens_per_page if cfg.use_paged_kv else None,
        )
        sparse_kv_metadata0 = SmemBlockSparseKvMetadataResource(
            pipeline_config=None,
            cfg=cfg,
            inst_id=0,
            route_metadata=sparse_route_metadata,
            route_layout=prepared_route_layout,
            tma_oob_origin=max_seq_len_kv,
            name="smemBlockSparseKvMetadata0",
        )
        sparse_kv_metadata1 = SmemBlockSparseKvMetadataResource(
            pipeline_config=None,
            cfg=cfg,
            inst_id=1,
            route_metadata=sparse_route_metadata,
            route_layout=prepared_route_layout,
            tma_oob_origin=max_seq_len_kv,
            name="smemBlockSparseKvMetadata1",
        )
        sparse_softmax_metadata0 = SmemBlockSparseSoftmaxMetadataResource(
            pipeline_config=sparse_softmax_metadata0_cfg,
            cfg=cfg,
            inst_id=0,
            route_metadata=sparse_route_metadata,
            route_layout=prepared_route_layout,
            name="smemBlockSparseSoftmaxMetadata0",
        )
        sparse_softmax_metadata1 = SmemBlockSparseSoftmaxMetadataResource(
            pipeline_config=sparse_softmax_metadata1_cfg,
            cfg=cfg,
            inst_id=1,
            route_metadata=sparse_route_metadata,
            route_layout=prepared_route_layout,
            name="smemBlockSparseSoftmaxMetadata1",
        )
    smem_kv = None
    smem_k0 = None
    smem_k1 = None
    smem_v0 = None
    smem_v1 = None
    smem_kv_raw = None
    if cfg.uses_keeps_transformed_kv:
        smem_kv_raw = SmemNvfp4RawKvResource(
            pipeline_config=smem_kv_raw_cfg,
            cfg=cfg,
            uses_register_capture=cfg.uses_nvfp4_raw_register_capture,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            tma_desc_k_sf=tma_desc_k_sf,
            tma_desc_v_sf=tma_desc_v_sf,
            page_offsets_kv=smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            name="smemKvRaw",
        )
        # The rings hold K/V in Q's compute dtype (see
        # FmhaDecodeConfig.keeps_transformed_kv_ring_cfg).
        nvfp4_ring_kwargs = dict(
            cfg=cfg.keeps_transformed_kv_ring_cfg,
            raw_src=smem_kv_raw,
            page_offsets_kv=smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            inst_id=0,
        )
        smem_k0 = SmemKvFromNvfp4Resource(
            pipeline_config=smem_k0_cfg,
            kv_kind=KV_KIND_K,
            name="smemK0",
            **nvfp4_ring_kwargs,
        )
        smem_v0 = SmemKvFromNvfp4Resource(
            pipeline_config=smem_v0_cfg,
            kv_kind=KV_KIND_V,
            name="smemV0",
            **nvfp4_ring_kwargs,
        )
    elif use_per_inst_kv_resources:
        smem_k0 = SmemKvTileResource(
            pipeline_config=smem_k0_cfg,
            cfg=cfg,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            tma_desc_k_sf=tma_desc_k_sf,
            tma_desc_v_sf=tma_desc_v_sf,
            tma_desc_k_atom=tma_desc_k_atom,
            tma_desc_v_atom=tma_desc_v_atom,
            tma_desc_k_summary=tma_desc_k_summary,
            tma_desc_v_summary=tma_desc_v_summary,
            tma_desc_k_summary_atom=tma_desc_k_summary_atom,
            tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            sparse_kv_metadata=sparse_kv_metadata0,
            page_offsets_kv=smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            inst_id=0,
            kv_kind=KV_KIND_K,
            name="smemK0",
        )
        if not use_shared_inst_kv_rings:
            smem_k1 = SmemKvTileResource(
                pipeline_config=smem_k1_cfg,
                cfg=cfg,
                tma_desc_k=tma_desc_k,
                tma_desc_v=tma_desc_v,
                tma_desc_k_sf=tma_desc_k_sf,
                tma_desc_v_sf=tma_desc_v_sf,
                tma_desc_k_atom=tma_desc_k_atom,
                tma_desc_v_atom=tma_desc_v_atom,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                sparse_kv_metadata=sparse_kv_metadata1,
                page_offsets_kv=smem_page_offsets,
                seqlens_kv=kv_seqlens,
                max_seq_len_kv=max_seq_len_kv,
                h_k_idx=h_k_idx,
                b_idx=b_idx,
                q_group_idx=q_group_idx,
                seq_len_q=seq_len_q,
                inst_id=1,
                kv_kind=KV_KIND_K,
                name="smemK1",
            )
        smem_v0 = SmemKvTileResource(
            pipeline_config=smem_v0_cfg,
            cfg=cfg,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            tma_desc_k_sf=tma_desc_k_sf,
            tma_desc_v_sf=tma_desc_v_sf,
            tma_desc_k_atom=tma_desc_k_atom,
            tma_desc_v_atom=tma_desc_v_atom,
            tma_desc_k_summary=tma_desc_k_summary,
            tma_desc_v_summary=tma_desc_v_summary,
            tma_desc_k_summary_atom=tma_desc_k_summary_atom,
            tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            sparse_kv_metadata=sparse_kv_metadata0,
            page_offsets_kv=smem_page_offsets_v or smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            inst_id=0,
            kv_kind=KV_KIND_V,
            name="smemV0",
        )
        if not use_shared_inst_kv_rings:
            smem_v1 = SmemKvTileResource(
                pipeline_config=smem_v1_cfg,
                cfg=cfg,
                tma_desc_k=tma_desc_k,
                tma_desc_v=tma_desc_v,
                tma_desc_k_sf=tma_desc_k_sf,
                tma_desc_v_sf=tma_desc_v_sf,
                tma_desc_k_atom=tma_desc_k_atom,
                tma_desc_v_atom=tma_desc_v_atom,
                tma_desc_k_summary=tma_desc_k_summary,
                tma_desc_v_summary=tma_desc_v_summary,
                tma_desc_k_summary_atom=tma_desc_k_summary_atom,
                tma_desc_v_summary_atom=tma_desc_v_summary_atom,
                sparse_kv_metadata=sparse_kv_metadata1,
                page_offsets_kv=smem_page_offsets_v or smem_page_offsets,
                seqlens_kv=kv_seqlens,
                max_seq_len_kv=max_seq_len_kv,
                h_k_idx=h_k_idx,
                b_idx=b_idx,
                q_group_idx=q_group_idx,
                seq_len_q=seq_len_q,
                inst_id=1,
                kv_kind=KV_KIND_V,
                name="smemV1",
            )
    else:
        smem_kv = SmemKvResource(
            pipeline_config=smem_kv_cfg,
            cfg=cfg,
            tma_desc_k=tma_desc_k,
            tma_desc_v=tma_desc_v,
            tma_desc_k_sf=tma_desc_k_sf,
            tma_desc_v_sf=tma_desc_v_sf,
            tma_desc_k_atom=tma_desc_k_atom,
            tma_desc_v_atom=tma_desc_v_atom,
            tma_desc_k_summary=tma_desc_k_summary,
            tma_desc_v_summary=tma_desc_v_summary,
            tma_desc_k_summary_atom=tma_desc_k_summary_atom,
            tma_desc_v_summary_atom=tma_desc_v_summary_atom,
            sparse_kv_metadata0=sparse_kv_metadata0,
            sparse_kv_metadata1=sparse_kv_metadata1,
            page_offsets_kv=smem_page_offsets,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            name="smemKv",
        )
    transformed_kv = None
    transformed_v = None
    transformed_kv1 = None
    mma_smem_kv = smem_kv
    if cfg.uses_decode_kv_converter:
        if cfg.store_transformed_kv_in_tmem:
            source_k = smem_k0 if cfg.uses_dual_kv_transform else smem_kv
            assert source_k is not None
            transformed_kv = TmemTransformedKvResource(
                pipeline_config=smem_transformed_kv_cfg,
                cfg=cfg,
                src_smem_kv=source_k,
                name="tmemTransformedKv",
            )
            if cfg.uses_dual_kv_transform:
                assert smem_v0 is not None
                transformed_v = TmemTransformedKvResource(
                    pipeline_config=smem_transformed_kv_cfg,
                    cfg=cfg,
                    src_smem_kv=smem_v0,
                    name="tmemTransformedV",
                )
        elif cfg.uses_dual_smem_transform:
            # One-instance Keeps: separate K and V rings replace the raw K and
            # V rings at the MMA.
            transformed_kv = SmemTransformedKvResource(
                pipeline_config=smem_transformed_k_cfg,
                cfg=cfg,
                src_smem_k0=smem_k0,
                name="smemTransformedK",
            )
            transformed_kv1 = SmemTransformedKvResource(
                pipeline_config=smem_transformed_v_cfg,
                cfg=cfg,
                src_smem_v0=smem_v0,
                name="smemTransformedV",
            )
        else:
            transformed_kv = SmemTransformedKvResource(
                pipeline_config=smem_transformed_kv_cfg,
                cfg=cfg,
                src_smem_kv=smem_kv,
                src_smem_k0=smem_k0,
                src_smem_k1=smem_k1,
                src_smem_v0=smem_v0,
                src_smem_v1=smem_v1,
                page_idx_kv=page_idx_kv,
                num_heads_kv=num_heads_kv,
                name="smemTransformedKv",
            )
            if cfg.uses_per_inst_smem_transform:
                # KV instance 1 converts into its own ring.
                transformed_kv1 = SmemTransformedKvResource(
                    pipeline_config=smem_transformed_kv_cfg,
                    cfg=cfg,
                    src_smem_kv=smem_kv,
                    src_smem_k0=smem_k0,
                    src_smem_k1=smem_k1,
                    src_smem_v0=smem_v0,
                    src_smem_v1=smem_v1,
                    page_idx_kv=page_idx_kv,
                    num_heads_kv=num_heads_kv,
                    name="smemTransformedKv1",
                )
        mma_smem_kv = transformed_kv

    # The correction task stages and reads the work tile's V scales itself.
    sage_v_scales = None
    if cfg.use_sage_attention:
        sage_v_scales = SageVScalesResource(
            pipeline_config=PipelineConfig(
                num_stages=1,
                num_bytes=0,
                producer_group=correction_grp,
                consumer_group=correction_grp,
                pipeline_type=PipelineType.AsyncAsync,
                cta_layout_vmnk=cta_layout,
                advance_on_wait=True,
            ),
            cfg=cfg,
            sage_scales_ptr=sage_scales_ptr,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            num_heads_kv=num_heads_kv,
            name="sageVScales",
        )
    # Each softmax instance reads its own ``sfK`` words from a ring that the
    # load warp fills after each K tile, through ``cp.async`` copies that the
    # ``AsyncLoad`` commit ties to the stage: the tile's atoms of the prepared
    # scales, a block-sparse route's or a dense tile's.
    sage_k_scales0 = None
    sage_k_scales1 = None
    if cfg.use_sage_attention:
        # Connect one K-scale SMEM ring from the load warp to its softmax instance.
        def sage_k_scales(inst_id: int, softmax_grp) -> SageKScalesResource:
            return SageKScalesResource(
                pipeline_config=PipelineConfig(
                    num_stages=k_tile_stages,
                    num_bytes=0,
                    producer_group=load_grp,
                    consumer_group=softmax_grp,
                    pipeline_type=PipelineType.AsyncAsync,
                    cta_layout_vmnk=cta_layout,
                    advance_on_wait=True,
                    async_producer_op=pipeline.PipelineOp.AsyncLoad,
                ),
                inst_id=inst_id,
                route_layout=prepared_route_layout if cfg.use_block_sparse else None,
                name=f"sageKScales{inst_id}",
                cfg=cfg,
                seqlens_kv=kv_seqlens,
                max_seq_len_kv=max_seq_len_kv,
                q_group_idx=q_group_idx,
                seq_len_q=seq_len_q,
                h_k_idx=h_k_idx,
                b_idx=b_idx,
                num_heads_kv=num_heads_kv,
                sage_scales_ptr=sage_scales_ptr,
            )

        sage_k_scales0 = sage_k_scales(0, softmax0_grp)
        sage_k_scales1 = sage_k_scales(1, softmax1_grp)

    tmem_s0 = TmemSResource(
        inst_id=0,
        pipeline_config=tmem_s0_cfg,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        seqlens_kv=kv_seqlens,
        max_seq_len_kv=max_seq_len_kv,
        h_r=h_r,
        q_group_idx=q_group_idx,
        seq_len_q=seq_len_q,
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        num_heads_kv=num_heads_kv,
        sync_barrier_id=0,
        name="tmemS0",
        sage_scales_ptr=sage_scales_ptr,
    )
    tmem_s1 = None
    if not use_one_inst_kv:
        tmem_s1 = TmemSResource(
            inst_id=1,
            pipeline_config=tmem_s1_cfg,
            cfg=cfg,
            scale_softmax_log2=scale_softmax_log2,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=max_seq_len_kv,
            h_r=h_r,
            q_group_idx=q_group_idx,
            seq_len_q=seq_len_q,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            num_heads_kv=num_heads_kv,
            sync_barrier_id=1,
            score_seed_owner=tmem_s0,
            name="tmemS1",
            sage_scales_ptr=sage_scales_ptr,
        )
    # Packed persistent QK derives the descriptor from Q's just-waited
    # consumer stage, avoiding a routed HEAD-to-LOOP descriptor value across
    # the guarded work-tile region. Fixed/static schedules keep their existing
    # explicit descriptor route.
    tmem_s0.q_ref = smem_q
    if tmem_s1 is not None:
        tmem_s1.q_ref = smem_q
    if cfg.uses_q_token_kv_block_sparse_page_membership:
        assert smem_page_offsets is not None
        tmem_s0.page_offsets_ref = smem_page_offsets
        if tmem_s1 is not None:
            tmem_s1.page_offsets_ref = smem_page_offsets

    smem_p0 = SmemPResource(
        inst_id=0,
        pipeline_config=smem_p0_cfg,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        use_variable_seqlens_kv=use_runtime_seqlens_kv,
        name="smemP0",
    )
    smem_p1 = None
    if not use_one_inst_kv:
        smem_p1 = SmemPResource(
            inst_id=1,
            pipeline_config=smem_p1_cfg,
            cfg=cfg,
            scale_softmax_log2=scale_softmax_log2,
            use_variable_seqlens_kv=use_runtime_seqlens_kv,
            name="smemP1",
        )

    tmem_o = TmemOResource(
        pipeline_config=tmem_o_cfg,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        name="tmemO",
    )

    tmem_softmax_local0 = TmemSoftmaxLocalResource(
        inst_id=0,
        pipeline_config=softmax_local0_cfg,
        cfg=cfg,
        name="tmemSoftmaxLocal0",
    )
    tmem_softmax_local1 = None
    if not use_one_inst_kv:
        tmem_softmax_local1 = TmemSoftmaxLocalResource(
            inst_id=1,
            pipeline_config=softmax_local1_cfg,
            cfg=cfg,
            name="tmemSoftmaxLocal1",
        )
    tmem_stats_done0 = (
        TmemStatsDoneResource(
            pipeline_config=stats_done0_cfg,
            name="tmemStatsDone0",
        )
        if stats_done0_cfg is not None
        else None
    )
    tmem_stats_done1 = (
        TmemStatsDoneResource(
            pipeline_config=stats_done1_cfg,
            name="tmemStatsDone1",
        )
        if stats_done1_cfg is not None
        else None
    )
    tmem_softmax_global0 = TmemSoftmaxGlobalResource(
        inst_id=0,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        sum_barrier_id=2,
        name="tmemSoftmaxGlobal0",
    )
    tmem_softmax_global1 = None
    if not use_one_inst_kv:
        tmem_softmax_global1 = TmemSoftmaxGlobalResource(
            inst_id=1,
            cfg=cfg,
            scale_softmax_log2=scale_softmax_log2,
            sum_barrier_id=3,
            name="tmemSoftmaxGlobal1",
        )
    tmem_softmax_order = (
        TmemSoftmaxOrderResource(cfg=cfg, name="tmemSoftmaxOrder")
        if use_ordered_softmax_barrier
        else None
    )
    smem_p0.tmem_s_ref = tmem_s0
    smem_p0.tmem_o_ref = tmem_o
    tmem_softmax_global0.p_ref = smem_p0
    tmem_softmax_global0.tmem_s_ref = tmem_s0
    if not use_one_inst_kv:
        smem_p1.tmem_s_ref = tmem_s1
        smem_p1.tmem_o_ref = tmem_o
        tmem_softmax_global1.p_ref = smem_p1
        tmem_softmax_global1.tmem_s_ref = tmem_s1

    # Compact runtime splits pack partials by the launch fanout.
    partial_split_stride = None
    if cfg.uses_compact_split_grid and requested_splits is not None:
        partial_split_stride = (
            requested_splits[0]
            if isinstance(requested_splits, tuple)
            else requested_splits
        )
    tmem_corr0 = TmemCorrResource(
        inst_id=0,
        cfg=cfg,
        scale_softmax_log2=scale_softmax_log2,
        output_scale=output_scale,
        o_ptr=o_ptr,
        partial_o_ptr=partial_o_ptr,
        partial_stats_ptr=partial_stats_ptr,
        split_kv_counter_ptr=split_kv_counter_ptr,
        attention_sinks_ptr=attention_sinks_ptr,
        lse_ptr=lse_ptr,
        seqlens_kv=kv_seqlens,
        max_seq_len_kv=corr_max_seq_len_kv,
        max_q_len=max_q_len,
        num_heads_kv=num_heads_kv,
        h_r=h_r,
        h_k_idx=h_k_idx,
        b_idx=b_idx,
        q_group_idx=q_group_idx,
        q_token_offset=q_token_offset,
        seq_len_q=seq_len_q,
        active_splits_kv=active_splits_kv,
        partial_split_stride=partial_split_stride,
        static_full_split_prefix=static_full_split_prefix,
        name="tmemCorr0",
        sage_v_scales_ref=sage_v_scales,
    )
    tmem_corr0.tmem_o_ref = tmem_o
    tmem_corr0.softmax_local0_ref = tmem_softmax_local0
    tmem_corr0.softmax_local1_ref = None if use_one_inst_kv else tmem_softmax_local1
    tmem_corr1 = None
    if not use_one_inst_kv:
        tmem_corr1 = TmemCorrResource(
            inst_id=1,
            cfg=cfg,
            scale_softmax_log2=scale_softmax_log2,
            output_scale=output_scale,
            o_ptr=o_ptr,
            partial_o_ptr=partial_o_ptr,
            partial_stats_ptr=partial_stats_ptr,
            split_kv_counter_ptr=split_kv_counter_ptr,
            attention_sinks_ptr=attention_sinks_ptr,
            lse_ptr=lse_ptr,
            seqlens_kv=kv_seqlens,
            max_seq_len_kv=corr_max_seq_len_kv,
            max_q_len=max_q_len,
            num_heads_kv=num_heads_kv,
            h_r=h_r,
            h_k_idx=h_k_idx,
            b_idx=b_idx,
            q_group_idx=q_group_idx,
            q_token_offset=q_token_offset,
            seq_len_q=seq_len_q,
            active_splits_kv=active_splits_kv,
            partial_split_stride=partial_split_stride,
            static_full_split_prefix=static_full_split_prefix,
            name="tmemCorr1",
            sage_v_scales_ref=sage_v_scales,
        )
        tmem_corr1.smem_p0_ref = smem_p0
        tmem_corr1.smem_p1_ref = smem_p1
        tmem_corr1.tmem_o_ref = tmem_o
        tmem_corr1.softmax_local0_ref = tmem_softmax_local0
        tmem_corr0.softmax_local1_ref = tmem_softmax_local1
        tmem_corr1.softmax_local1_ref = tmem_softmax_local1

    # ------------------------------------------------------------------
    # Finalize SMEM and membership mode before tracing tasks.
    # ------------------------------------------------------------------
    smem_resources = []
    if work_queue is not None:
        smem_resources.append(work_queue)
    if schedule_token_throttle is not None:
        smem_resources.append(schedule_token_throttle)
    if membership_lifetime is not None:
        smem_resources.append(membership_lifetime)
    smem_resources.append(smem_q)
    if smem_page_offsets is not None:
        smem_resources.append(smem_page_offsets)
    if smem_page_offsets_v is not None:
        smem_resources.append(smem_page_offsets_v)
    if sparse_kv_metadata0 is not None:
        smem_resources.append(sparse_kv_metadata0)
        smem_resources.append(sparse_kv_metadata1)
    if sparse_softmax_metadata0 is not None:
        smem_resources.append(sparse_softmax_metadata0)
        smem_resources.append(sparse_softmax_metadata1)
    if smem_kv_raw is not None:
        smem_resources.append(smem_kv_raw)
    if use_one_inst_qkv:
        smem_resources.append(smem_k0)
        smem_resources.append(smem_v0)
    elif use_per_inst_kv_resources:
        for resource in (smem_k0, smem_k1, smem_v0, smem_v1):
            if resource is not None:
                smem_resources.append(resource)
    else:
        smem_resources.append(smem_kv)
    if transformed_kv is not None:
        smem_resources.append(transformed_kv)
    if transformed_v is not None:
        smem_resources.append(transformed_v)
    if transformed_kv1 is not None:
        smem_resources.append(transformed_kv1)
    smem_resources.append(smem_p0)
    if not use_one_inst_kv:
        smem_resources.append(smem_p1)
    # An instance's ``sfK`` ring follows its S resource in the layout.
    smem_resources.append(tmem_s0)
    if sage_k_scales0 is not None:
        smem_resources.append(sage_k_scales0)
    if not use_one_inst_kv:
        smem_resources.append(tmem_s1)
        if sage_k_scales1 is not None:
            smem_resources.append(sage_k_scales1)
    smem_resources.append(tmem_o)
    smem_resources.append(tmem_softmax_local0)
    if not use_one_inst_kv:
        smem_resources.append(tmem_softmax_local1)
    smem_resources.append(tmem_softmax_global0)
    if not use_one_inst_kv:
        smem_resources.append(tmem_softmax_global1)
    smem_resources.append(tmem_corr0)
    if not use_one_inst_kv:
        smem_resources.append(tmem_corr1)
    if sage_v_scales is not None:
        smem_resources.append(sage_v_scales)

    # Lay out all resource storage, CLC responses and the shared TMEM pointer.
    def allocate_smem() -> tuple[SmemAllocator, SmemAllocation | None]:
        allocator = SmemAllocator()
        clc_response_alloc = None
        for resource in smem_resources:
            allocator.add_resource(resource)
            if resource is work_queue and tile_sched_params is not None:
                # A compile-time offset from the unified base spares every
                # role a separate pointer that ptxas spills to local memory.
                clc_response_alloc = allocator.add(
                    SmemAllocation(
                        "clc_response",
                        dtype=cutlass.Int128,
                        count=WORK_QUEUE_STAGES,
                        alignment=16,
                    )
                )
        allocator.add_tmem_ptr(
            SmemAllocation("fmha_tmem_ptr_i32", dtype=cutlass.Int32, alignment=4)
        )
        allocator.compute_layout()
        return allocator, clc_response_alloc

    def smem_bytes(allocator: SmemAllocator) -> int:
        # Include barriers and conservatively round data to tensor alignment.
        return (
            allocator.total_smem_bytes + cfg.stensor_align - 1
        ) // cfg.stensor_align * cfg.stensor_align + allocator.barrier_smem_bytes

    # The static profile checks bound only the Q and K/V pipelines; check the
    # complete layout against the compile target's SMEM capacity.
    # Native SM107 targets can use 327 KiB. A family
    # sm_100f compilation must retain the portable 227-KiB limit.
    target_smem_kib = 327 if is_sm107_compile_target() else SMEM_CAPACITY_KIB
    smem_capacity_bytes = target_smem_kib * BYTES_PER_KIB
    smem_allocator, clc_response_alloc = allocate_smem()
    if (
        cfg.uses_q_token_kv_block_sparse_page_membership
        and smem_bytes(smem_allocator) > smem_capacity_bytes
    ):
        # A full union can be much larger than a KV tile. Keep small routes
        # cached, but let Softmax read immutable packed GMEM words on demand
        # when the complete resource layout cannot hold the membership row.
        # This also accounts for the fixed K/V rings of one-inst D256.
        assert smem_page_offsets is not None
        smem_page_offsets.cache_memberships_in_smem = False
        smem_page_offsets.init_placeholder_state()
        if membership_lifetime is not None:
            # Nothing in SMEM needs protecting once memberships stay in GMEM.
            smem_resources.remove(membership_lifetime)
            membership_lifetime = None
        smem_allocator, clc_response_alloc = allocate_smem()
    launch_smem_bytes = smem_bytes(smem_allocator)
    if launch_smem_bytes > smem_capacity_bytes:
        raise ValueError(
            "decode resources exceed the SM shared-memory capacity: "
            f"q_stages={cfg.q_stages}, kv_stages={cfg.kv_stages} need "
            f"{launch_smem_bytes} bytes, capacity is {smem_capacity_bytes} bytes"
        )
    if clc_response_alloc is not None:
        # Bind the scheduler config before the task manager creates the queue.
        smem_allocator.allocate()
        work_queue.tile_scheduler_config = (
            TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                tile_scheduler_params=tile_sched_params,
                response_ptr=cute.make_ptr(
                    cutlass.Int128,
                    smem_allocator.get(clc_response_alloc).data_ptr(),
                    mem_space=cutlass.AddressSpace.smem,
                    assumed_align=16,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Domain computation
    # ------------------------------------------------------------------
    # HEAD handles the first 2 K tiles, LOOP advances the staggered
    # Qk/Pv steady-state, and TAIL drains the final V wave. For odd tile
    # counts, the final wave still requires one additional loop iteration so
    # that inst0 can process the last K/V tile.
    local_kv_tiles = compute_local_kv_tiles(cfg, total_kv_tiles)
    loop_domain = compute_decode_gen_loop_domain(local_kv_tiles, cfg.num_insts_kv)

    load_domain = loop_domain
    mma_domain = loop_domain
    softmax_domain = loop_domain + 1
    corr_domain = loop_domain

    # ------------------------------------------------------------------
    # Create tasks
    # ------------------------------------------------------------------
    task_runtime_kwargs = {
        "seqlens_kv": kv_seqlens,
        "max_seq_len_kv": max_seq_len_kv,
        "seq_len_q": seq_len_q,
        "sparse_row_route_offsets": sparse_row_route_offsets,
        "sparse_row_route_counts": sparse_row_route_counts,
        "sparse_row_route_begin": sparse_row_route_begin,
        "sparse_route_count": sparse_route_count,
        "num_heads_kv": num_heads_kv,
    }
    # A Sage plan's ``sfK`` rings follow the K tiles.
    sage_k_scale_rings = (
        {"sage_k_scales0": sage_k_scales0, "sage_k_scales1": sage_k_scales1}
        if cfg.use_sage_attention
        else {}
    )
    if cfg.uses_keeps_transformed_kv:
        load_tasks = (
            create_load_task(
                smem_q,
                smem_kv_raw,
                work_queue,
                schedule_token_throttle,
                cfg,
                domain=load_domain,
                domain_bias=0,
                warp_idx=cfg.clc_load_warp_idx if use_clc_dynamic else None,
                smem_page_offsets=smem_page_offsets,
                **task_runtime_kwargs,
            ),
        )
    elif use_one_inst_qkv:
        load_tasks = (
            create_load_task_one_inst_qkv(
                smem_q,
                smem_k0,
                smem_v0,
                work_queue,
                schedule_token_throttle,
                cfg,
                domain=load_domain,
                smem_page_offsets=smem_page_offsets,
                domain_bias=0,
                warp_idx=cfg.clc_load_warp_idx if use_clc_dynamic else None,
                **task_runtime_kwargs,
            ),
        )
    elif use_per_inst_kv_resources:
        if use_per_inst_block_sparse_load_tasks:
            assert per_inst_block_sparse_load_topology is not None
            load_warp_indices, _ = per_inst_block_sparse_load_topology
            load_tasks = create_block_sparse_load_tasks_per_inst(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                work_queue,
                schedule_token_throttle,
                cfg,
                domain=load_domain,
                sparse_kv_metadata0=sparse_kv_metadata0,
                sparse_kv_metadata1=sparse_kv_metadata1,
                sparse_softmax_metadata0=sparse_softmax_metadata0,
                sparse_softmax_metadata1=sparse_softmax_metadata1,
                domain_bias=0,
                warp_indices=load_warp_indices,
                **sage_k_scale_rings,
                **task_runtime_kwargs,
            )
        else:
            load_tasks = (
                create_load_task_split_kv(
                    smem_q,
                    smem_k0,
                    smem_k1,
                    smem_v0,
                    smem_v1,
                    work_queue,
                    schedule_token_throttle,
                    cfg,
                    domain=load_domain,
                    smem_page_offsets=smem_page_offsets,
                    smem_page_offsets_v=smem_page_offsets_v,
                    sparse_kv_metadata0=sparse_kv_metadata0,
                    sparse_kv_metadata1=sparse_kv_metadata1,
                    sparse_softmax_metadata0=sparse_softmax_metadata0,
                    sparse_softmax_metadata1=sparse_softmax_metadata1,
                    domain_bias=0,
                    warp_idx=cfg.clc_load_warp_idx if use_clc_dynamic else None,
                    **sage_k_scale_rings,
                    **task_runtime_kwargs,
                ),
            )
    else:
        load_tasks = (
            create_load_task(
                smem_q,
                smem_kv,
                work_queue,
                schedule_token_throttle,
                cfg,
                domain=load_domain,
                domain_bias=0,
                warp_idx=cfg.clc_load_warp_idx if use_clc_dynamic else None,
                smem_page_offsets=smem_page_offsets,
                sparse_kv_metadata0=sparse_kv_metadata0,
                sparse_kv_metadata1=sparse_kv_metadata1,
                sparse_softmax_metadata0=sparse_softmax_metadata0,
                sparse_softmax_metadata1=sparse_softmax_metadata1,
                **sage_k_scale_rings,
                **task_runtime_kwargs,
            ),
        )
    page_offsets_task = None
    if uses_page_offsets_task:
        page_offsets_warp_idx = cfg.page_offsets_warp_idx
        if use_separate_kv_page_offset_resources:
            page_offsets_task = create_page_offsets_task_split_kv(
                smem_page_offsets,
                smem_page_offsets_v,
                work_queue,
                cfg,
                domain=load_domain,
                domain_bias=0,
                warp_idx=page_offsets_warp_idx,
                num_warps=cfg.page_offsets_num_warps,
                membership_lifetime=membership_lifetime,
                block_table_capacity=page_table_capacity
                if use_native_paged_kv
                else None,
                **task_runtime_kwargs,
            )
        else:
            page_offsets_task_fn = (
                create_page_offsets_task_one_inst_qkv
                if use_one_inst_qkv and not cfg.uses_keeps_transformed_kv
                else create_page_offsets_task
            )
            page_offsets_task = page_offsets_task_fn(
                smem_page_offsets,
                work_queue,
                cfg,
                domain=load_domain,
                domain_bias=0,
                warp_idx=page_offsets_warp_idx,
                num_warps=cfg.page_offsets_num_warps,
                membership_lifetime=membership_lifetime,
                block_table_capacity=page_table_capacity
                if use_native_paged_kv
                else None,
                **task_runtime_kwargs,
            )
    if cfg.uses_dual_kv_transform:
        assert transformed_kv is not None and transformed_v is not None
        mma_task = create_mma_task_dual_transform_kv(
            smem_q,
            transformed_kv,
            transformed_v,
            tmem_s0,
            smem_p0,
            tmem_o,
            cfg,
            domain=mma_domain,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    elif use_one_inst_qkv:
        mma_task = create_mma_task_one_inst_qkv(
            smem_q,
            transformed_kv if cfg.uses_dual_smem_transform else smem_k0,
            transformed_kv1 if cfg.uses_dual_smem_transform else smem_v0,
            tmem_s0,
            smem_p0,
            tmem_o,
            work_queue,
            cfg,
            tmem_stats_done=tmem_stats_done0,
            domain=mma_domain,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    elif use_per_inst_kv_resources and not cfg.uses_decode_kv_converter:
        mma_task = create_mma_task_split_kv(
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
            work_queue,
            cfg,
            domain=mma_domain,
            tmem_stats_done0=tmem_stats_done0,
            tmem_stats_done1=tmem_stats_done1,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    else:
        mma_task = create_mma_task(
            smem_q,
            mma_smem_kv,
            tmem_s0,
            tmem_s1,
            smem_p0,
            smem_p1,
            tmem_o,
            work_queue,
            cfg,
            domain=mma_domain,
            domain_bias=0,
            smem_kv1=transformed_kv1,
            **task_runtime_kwargs,
        )
    transform_kv_task = None
    transform_v_task = None
    transform_kv1_task = None
    if cfg.uses_keeps_transformed_kv:
        transform_kv_task = create_nvfp4_keeps_transform_task(
            smem_kv_raw,
            smem_k0,
            smem_v0,
            work_queue,
            cfg,
            domain=load_domain,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    elif cfg.uses_dual_kv_transform:
        transform_kv_task = create_kv_operand_transform_task(
            smem_k0,
            transformed_kv,
            cfg,
            domain=load_domain,
            is_v=False,
            domain_bias=0,
            **task_runtime_kwargs,
        )
        transform_v_task = create_kv_operand_transform_task(
            smem_v0,
            transformed_v,
            cfg,
            domain=load_domain,
            is_v=True,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    elif cfg.uses_dual_smem_transform:
        transform_kv_task, transform_kv1_task = (
            create_kv_operand_transform_task(
                raw,
                ring,
                cfg,
                domain=load_domain,
                is_v=is_v,
                domain_bias=0,
                **task_runtime_kwargs,
            )
            for raw, ring, is_v in (
                (smem_k0, transformed_kv, False),
                (smem_v0, transformed_kv1, True),
            )
        )
    elif cfg.uses_per_inst_smem_transform:
        transform_kv_task, transform_kv1_task = (
            create_transform_kv_task(
                smem_kv,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                ring,
                work_queue,
                cfg,
                domain=load_domain,
                domain_bias=0,
                inst_id=inst_id,
                **task_runtime_kwargs,
            )
            for inst_id, ring in ((0, transformed_kv), (1, transformed_kv1))
        )
    elif cfg.use_transform_kv:
        transform_kv_task = create_transform_kv_task(
            smem_kv,
            smem_k0,
            smem_k1,
            smem_v0,
            smem_v1,
            transformed_kv,
            work_queue,
            cfg,
            domain=load_domain,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    softmax0_task = create_softmax0_task(
        tmem_s0,
        tmem_softmax_local0,
        smem_p0,
        tmem_softmax_global0,
        tmem_softmax_order,
        sparse_softmax_metadata0,
        work_queue,
        cfg,
        domain=softmax_domain,
        domain_bias=1,
        membership_lifetime=membership_lifetime,
        sage_k_scales=sage_k_scales0,
        **task_runtime_kwargs,
    )
    softmax1_task = None
    if not use_one_inst_kv:
        softmax1_task = create_softmax1_task(
            tmem_s1,
            tmem_softmax_local1,
            smem_p1,
            tmem_softmax_global1,
            tmem_softmax_order,
            sparse_softmax_metadata1,
            work_queue,
            cfg,
            domain=softmax_domain,
            domain_bias=1,
            membership_lifetime=membership_lifetime,
            sage_k_scales=sage_k_scales1,
            **task_runtime_kwargs,
        )
    if use_one_inst_kv:
        correction_task = create_correction_task_one_inst_qkv(
            tmem_softmax_local0,
            tmem_o,
            tmem_corr0,
            work_queue,
            cfg,
            tmem_stats_done=tmem_stats_done0,
            sage_v_scales=sage_v_scales,
            domain=corr_domain,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    else:
        correction_task = create_correction_task(
            tmem_softmax_local0,
            tmem_softmax_local1,
            tmem_o,
            tmem_corr0,
            tmem_corr1,
            work_queue,
            cfg,
            domain=corr_domain,
            tmem_stats_done0=tmem_stats_done0,
            tmem_stats_done1=tmem_stats_done1,
            sage_v_scales=sage_v_scales,
            domain_bias=0,
            **task_runtime_kwargs,
        )
    if use_per_inst_block_sparse_load_tasks:
        assert per_inst_block_sparse_load_topology is not None
        _, residual_padding = per_inst_block_sparse_load_topology
        padding_warp_ranges = (
            (residual_padding,) if residual_padding is not None else ()
        )
    else:
        padding_warp_ranges = (
            (cfg.wg0_padding_warp_idx, cfg.wg0_padding_num_warps),
            (cfg.wg1_padding_warp_idx, cfg.wg1_padding_num_warps),
            (cfg.wg2_padding_warp_idx, cfg.wg2_padding_num_warps),
            (cfg.wg3_padding_warp_idx, cfg.wg3_padding_num_warps),
            (cfg.wg4_padding_warp_idx, cfg.wg4_padding_num_warps),
            (cfg.wg5_padding_warp_idx, cfg.wg5_padding_num_warps),
        )
    padding_tasks = [
        create_padding_task(
            cfg,
            work_queue,
            warp_idx=warp_idx,
            num_warps=num_warps,
        )
        for warp_idx, num_warps in padding_warp_ranges
        if num_warps > 0
    ]
    scheduler_task = None
    if use_clc_dynamic:
        scheduler_task = create_scheduler_task(work_queue, schedule_token_throttle, cfg)

    task_list = []
    if page_offsets_task is not None:
        task_list.append(page_offsets_task)
    task_list.extend(load_tasks)
    if transform_kv_task is not None:
        task_list.append(transform_kv_task)
    if transform_v_task is not None:
        task_list.append(transform_v_task)
    if transform_kv1_task is not None:
        task_list.append(transform_kv1_task)
    if use_one_inst_qkv and not use_clc_dynamic:
        task_list.extend([correction_task, mma_task])
        task_list.append(softmax0_task)
    else:
        task_list.append(softmax0_task)
        if softmax1_task is not None:
            task_list.append(softmax1_task)
        task_list.extend([correction_task, mma_task])
    if scheduler_task is not None:
        task_list.append(scheduler_task)
    task_list.extend(padding_tasks)
    # ------------------------------------------------------------------
    # Resource dependency graph
    # ------------------------------------------------------------------
    smem_kv_deps = []
    if smem_page_offsets is not None:
        smem_kv_deps.append(smem_page_offsets)
    smem_k_deps = list(smem_kv_deps)
    smem_v_deps = (
        [smem_page_offsets_v] if smem_page_offsets_v is not None else list(smem_kv_deps)
    )
    if use_one_inst_qkv:
        mma_k = transformed_kv if cfg.uses_dual_smem_transform else smem_k0
        mma_v = transformed_kv1 if cfg.uses_dual_smem_transform else smem_v0
        resource_dependency_graph = {
            smem_q: [],
            smem_k0: [smem_kv_raw] if smem_kv_raw is not None else smem_kv_deps,
            smem_v0: [smem_kv_raw] if smem_kv_raw is not None else smem_kv_deps,
            **(
                {transformed_kv: [smem_k0], transformed_kv1: [smem_v0]}
                if cfg.uses_dual_smem_transform
                else {}
            ),
            tmem_s0: [mma_k, smem_q],
            smem_p0: [tmem_s0],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_global0: [tmem_s0],
            tmem_o: [smem_p0, mma_v],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
        }
    elif cfg.uses_dual_kv_transform:
        resource_dependency_graph = {
            smem_q: [],
            smem_k0: smem_k_deps,
            smem_v0: smem_v_deps,
            transformed_kv: [smem_k0],
            transformed_v: [smem_v0],
            tmem_s0: [transformed_kv, smem_q],
            smem_p0: [tmem_s0],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_global0: [tmem_s0],
            tmem_o: [smem_p0, transformed_v],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
        }
    elif cfg.uses_decode_kv_converter and use_per_inst_kv_resources:
        resource_dependency_graph = {
            smem_q: [],
            smem_k0: smem_k_deps,
            smem_k1: smem_k_deps,
            smem_v0: smem_v_deps,
            smem_v1: smem_v_deps,
            transformed_kv: [smem_k0, smem_v0]
            if transformed_kv1 is not None
            else [smem_k0, smem_k1, smem_v0, smem_v1],
            **(
                {transformed_kv1: [smem_k1, smem_v1]}
                if transformed_kv1 is not None
                else {}
            ),
            tmem_s0: [transformed_kv, smem_q],
            smem_p0: [tmem_s0],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_global0: [tmem_s0],
            tmem_o: [smem_p0, transformed_kv],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
        }
        if not use_one_inst_kv:
            resource_dependency_graph.update(
                {
                    tmem_s1: [
                        transformed_kv if transformed_kv1 is None else transformed_kv1,
                        smem_q,
                    ],
                    smem_p1: [tmem_s1],
                    tmem_softmax_local1: [tmem_s1],
                    tmem_softmax_global1: [tmem_s1],
                    tmem_o: [smem_p0, smem_p1, transformed_kv]
                    + ([transformed_kv1] if transformed_kv1 is not None else []),
                    tmem_corr1: [
                        tmem_softmax_local0,
                        tmem_softmax_local1,
                        tmem_o,
                    ],
                }
            )
    elif use_one_inst_kv:
        resource_dependency_graph = {
            smem_q: [],
            smem_kv: smem_kv_deps,
            tmem_s0: [mma_smem_kv, smem_q],
            smem_p0: [tmem_s0],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_global0: [tmem_s0],
            tmem_o: [smem_p0, mma_smem_kv],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
        }
    elif use_per_inst_kv_resources:
        resource_dependency_graph = {
            **(
                {
                    sparse_kv_metadata0: [],
                    sparse_kv_metadata1: [],
                }
                if sparse_kv_metadata0 is not None
                else {}
            ),
            **(
                {
                    sparse_softmax_metadata0: [sparse_kv_metadata0],
                    sparse_softmax_metadata1: [sparse_kv_metadata1],
                }
                if sparse_softmax_metadata0 is not None
                else {}
            ),
            smem_q: [],
            smem_k0: smem_k_deps
            + ([sparse_kv_metadata0] if sparse_kv_metadata0 is not None else []),
            smem_v0: smem_v_deps
            + ([sparse_kv_metadata0] if sparse_kv_metadata0 is not None else []),
            **(
                {
                    smem_k1: smem_k_deps
                    + (
                        [sparse_kv_metadata1] if sparse_kv_metadata1 is not None else []
                    ),
                    smem_v1: smem_v_deps
                    + (
                        [sparse_kv_metadata1] if sparse_kv_metadata1 is not None else []
                    ),
                }
                if smem_k1 is not None
                else {}
            ),
            tmem_s0: [smem_k0, smem_q],
            tmem_s1: [smem_k0 if smem_k1 is None else smem_k1, smem_q],
            smem_p0: [tmem_s0],
            smem_p1: [tmem_s1],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_local1: [tmem_s1],
            tmem_softmax_global0: [tmem_s0],
            tmem_softmax_global1: [tmem_s1],
            tmem_o: [smem_p0, smem_p1, smem_v0]
            if smem_v1 is None
            else [smem_p0, smem_p1, smem_v0, smem_v1],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
            tmem_corr1: [tmem_softmax_local0, tmem_softmax_local1, tmem_o],
        }
    else:
        resource_dependency_graph = {
            **(
                {
                    sparse_kv_metadata0: [],
                    sparse_kv_metadata1: [],
                    sparse_softmax_metadata0: [sparse_kv_metadata0],
                    sparse_softmax_metadata1: [sparse_kv_metadata1],
                }
                if sparse_kv_metadata0 is not None
                else {}
            ),
            smem_q: [],
            smem_kv: smem_kv_deps
            + (
                [sparse_kv_metadata0, sparse_kv_metadata1]
                if sparse_kv_metadata0 is not None
                else []
            ),
            tmem_s0: [mma_smem_kv, smem_q],
            tmem_s1: [mma_smem_kv, smem_q],
            smem_p0: [tmem_s0],
            smem_p1: [tmem_s1],
            tmem_softmax_local0: [tmem_s0],
            tmem_softmax_local1: [tmem_s1],
            tmem_softmax_global0: [tmem_s0],
            tmem_softmax_global1: [tmem_s1],
            tmem_o: [smem_p0, smem_p1, mma_smem_kv],
            tmem_corr0: [tmem_softmax_local0, tmem_o],
            tmem_corr1: [tmem_softmax_local0, tmem_softmax_local1, tmem_o],
        }
    if transformed_kv is not None and not use_per_inst_kv_resources:
        resource_dependency_graph[transformed_kv] = [smem_kv]
    if smem_kv_raw is not None:
        resource_dependency_graph[smem_kv_raw] = list(smem_kv_deps)
    if sparse_softmax_metadata0 is not None:
        resource_dependency_graph[smem_p0].append(sparse_softmax_metadata0)
        resource_dependency_graph[tmem_softmax_local0].append(sparse_softmax_metadata0)
        resource_dependency_graph[tmem_softmax_global0].append(sparse_softmax_metadata0)
        assert sparse_softmax_metadata1 is not None
        resource_dependency_graph[smem_p1].append(sparse_softmax_metadata1)
        resource_dependency_graph[tmem_softmax_local1].append(sparse_softmax_metadata1)
        resource_dependency_graph[tmem_softmax_global1].append(sparse_softmax_metadata1)
    if membership_lifetime is not None:
        resource_dependency_graph[membership_lifetime] = []
        for resource in (smem_p0, tmem_softmax_local0, tmem_softmax_global0):
            resource_dependency_graph[resource].append(membership_lifetime)
        if not use_one_inst_kv:
            for resource in (smem_p1, tmem_softmax_local1, tmem_softmax_global1):
                resource_dependency_graph[resource].append(membership_lifetime)
    if tmem_stats_done0 is not None:
        resource_dependency_graph[tmem_s0].append(tmem_stats_done0)
        resource_dependency_graph[tmem_stats_done0] = [tmem_softmax_local0]
        if tmem_stats_done1 is not None:
            assert tmem_s1 is not None
            assert tmem_softmax_local1 is not None
            resource_dependency_graph[tmem_s1].append(tmem_stats_done1)
            resource_dependency_graph[tmem_stats_done1] = [tmem_softmax_local1]
    # A softmax instance's outputs read its K scale words. The correction
    # epilogue reads the V channel scales.
    for scales, outputs in (
        (sage_k_scales0, (smem_p0, tmem_softmax_local0, tmem_softmax_global0)),
        (sage_k_scales1, (smem_p1, tmem_softmax_local1, tmem_softmax_global1)),
    ):
        if scales is None:
            continue
        resource_dependency_graph[scales] = []
        for resource in outputs:
            resource_dependency_graph[resource].append(scales)
    if sage_v_scales is not None:
        resource_dependency_graph[sage_v_scales] = []
        for resource in (tmem_corr0, tmem_corr1):
            resource_dependency_graph[resource].append(sage_v_scales)
    if cutlass.const_expr(use_ordered_softmax_barrier):
        resource_dependency_graph[tmem_softmax_order] = [tmem_s0]
        resource_dependency_graph[smem_p1] = [
            *resource_dependency_graph[smem_p1],
            tmem_softmax_order,
        ]
    if smem_page_offsets is not None:
        resource_dependency_graph[smem_page_offsets] = []
    if smem_page_offsets_v is not None:
        resource_dependency_graph[smem_page_offsets_v] = []
    if work_queue is not None:
        for deps in resource_dependency_graph.values():
            deps.append(work_queue)
        resource_dependency_graph[work_queue] = (
            [work_queue, schedule_token_throttle]
            if schedule_token_throttle is not None
            else ([work_queue] if use_clc_dynamic else [])
        )
    if schedule_token_throttle is not None:
        resource_dependency_graph[schedule_token_throttle] = [work_queue]
    dma_consumer_release_labels: dict[
        tuple[MemoryResource, MemoryResource], set[str]
    ] = {}
    if cfg.uses_dual_smem_transform:
        # One-instance Keeps reads K from one ring and V from the other.
        dma_consumer_release_labels[(transformed_kv, tmem_s0)] = {"kv_desc"}
        dma_consumer_release_labels[(transformed_kv1, tmem_o)] = {"v_desc"}
    elif transformed_kv1 is not None:
        # Each instance ring publishes its own K and V descriptors.
        dma_consumer_release_labels[(transformed_kv, tmem_s0)] = {"k_desc_0"}
        dma_consumer_release_labels[(transformed_kv, tmem_o)] = {"v_desc_0"}
        dma_consumer_release_labels[(transformed_kv1, tmem_s1)] = {"k_desc_1"}
        dma_consumer_release_labels[(transformed_kv1, tmem_o)] = {"v_desc_1"}
    elif transformed_kv is not None:
        dma_consumer_release_labels[(transformed_kv, tmem_s0)] = {"k_desc_0"}
        dma_consumer_release_labels[
            (transformed_kv if transformed_v is None else transformed_v, tmem_o)
        ] = {"v_desc_0"}
        if not use_one_inst_kv:
            dma_consumer_release_labels[(transformed_kv, tmem_s1)] = {"k_desc_1"}
            dma_consumer_release_labels[(transformed_kv, tmem_o)].add("v_desc_1")
    if smem_page_offsets is not None:
        if smem_kv_raw is not None:
            dma_consumer_release_labels[(smem_page_offsets, smem_kv_raw)] = (
                {"read_offsets"}
                if can_hold_native_page_window(cfg, smem_page_offsets)
                else {"read_offsets_k0", "read_offsets_v0"}
            )
        elif use_one_inst_qkv:
            if can_hold_native_page_window(cfg, smem_page_offsets):
                # One consumer stage remains live across the complete K/V
                # cadence, so both DMA edges share its final release label.
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {"read_offsets"},
                        (smem_page_offsets, smem_v0): {"read_offsets"},
                    }
                )
            else:
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {"read_offsets_k0"},
                        (smem_page_offsets, smem_v0): {"read_offsets_v0"},
                    }
                )
        elif use_per_inst_kv_resources:
            if (
                smem_page_offsets_v is None
                and smem_page_offsets.holds_encoded_locator_window
            ):
                # Independent K/V data FIFOs can still share a single
                # read-only locator window until the last V tile is issued.
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, resource): {"read_offsets"}
                        for resource in (smem_k0, smem_k1, smem_v0, smem_v1)
                    }
                )
            elif smem_k1 is None:
                # One ring per operand serves both instances, so the single key
                # carries the release labels of both.
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {
                            "read_offsets_k0",
                            "read_offsets_k1",
                        },
                        (smem_page_offsets_v, smem_v0): {
                            "read_offsets_v0",
                            "read_offsets_v1",
                        },
                    }
                )
            elif smem_page_offsets_v is not None:
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {"read_offsets_k0"},
                        (smem_page_offsets, smem_k1): {"read_offsets_k1"},
                        (smem_page_offsets_v, smem_v0): {"read_offsets_v0"},
                        (smem_page_offsets_v, smem_v1): {"read_offsets_v1"},
                    }
                )
            else:
                dma_consumer_release_labels.update(
                    {
                        (smem_page_offsets, smem_k0): {"read_offsets_k0"},
                        (smem_page_offsets, smem_k1): {"read_offsets_k1"},
                        (smem_page_offsets, smem_v0): {"read_offsets_v0"},
                        (smem_page_offsets, smem_v1): {"read_offsets_v1"},
                    }
                )
        else:
            hold_page_window = can_hold_native_page_window(cfg, smem_page_offsets)
            if cfg.num_head_dim_stages_kv > 1 and not cfg.uses_scattered_page_route:
                page_offset_labels = {"cache_page_ids"}
            elif hold_page_window:
                page_offset_labels = {"read_offsets"}
            elif use_one_inst_kv:
                page_offset_labels = {"read_offsets_k0", "read_offsets_v0"}
            else:
                page_offset_labels = {
                    "read_offsets_k0",
                    "read_offsets_k1",
                    "read_offsets_v0",
                    "read_offsets_v1",
                }
            dma_consumer_release_labels[(smem_page_offsets, smem_kv)] = (
                page_offset_labels
            )
    if smem_kv is not None and transformed_kv is None:
        dma_consumer_release_labels[(smem_kv, tmem_s0)] = {"k_desc_0"}
        dma_consumer_release_labels[(smem_kv, tmem_o)] = {"v_desc_0"}
        if not use_one_inst_kv:
            dma_consumer_release_labels[(smem_kv, tmem_s1)] = {"k_desc_1"}
            dma_consumer_release_labels[(smem_kv, tmem_o)].add("v_desc_1")

    # ------------------------------------------------------------------
    # TMEM allocator
    # ------------------------------------------------------------------
    tmem_allocator = TmemAllocator()
    if cfg.use_keeps_mma_ab:
        if use_one_inst_qkv:
            tmem_allocator.add_resource(tmem_s0)
        else:
            # Build the two instruction-local phases from the resources that
            # actually use TMEM.  Depending on the profile, P can overlay S
            # or live in SMEM, and stats can be standalone TMEM or an SMEM
            # handoff.  Empty resources must not become scheduler aliases.
            for tmem_s, p in (
                (tmem_s0, smem_p0),
                (tmem_s1, smem_p1),
            ):
                p_requirements = p.get_tmem_requirements()
                if p_requirements:
                    tmem_allocator.add_alias_group(
                        [tmem_s.get_tmem_requirements(), p_requirements]
                    )
                else:
                    tmem_allocator.add_resource(tmem_s)
            # Register standalone stats only after both S/P phases, preserving
            # the established allocation order. SMEM-backed stats are a no-op.
            tmem_allocator.add_resource(tmem_softmax_local0)
            tmem_allocator.add_resource(tmem_softmax_local1)
    else:
        tmem_allocator.add_resource(tmem_s0)
        tmem_allocator.add_resource(tmem_softmax_local0)
        if not use_one_inst_kv:
            tmem_allocator.add_resource(tmem_s1)
            tmem_allocator.add_resource(tmem_softmax_local1)
    tmem_allocator.add_resource(tmem_o)
    if cfg.store_transformed_kv_in_tmem:
        assert transformed_kv is not None
        tmem_allocator.add_resource(transformed_kv)
        if transformed_v is not None:
            tmem_allocator.add_resource(transformed_v)
    tmem_allocator.compute_layout()
    if cfg.use_keeps_mma_ab and not use_one_inst_qkv and not cfg.uses_tmem_p:
        # Two-inst Keeps with SMEM P (Q64/KV128 and 8-bit Q128/KV128) keeps the
        # standalone-first S/O layout after unused stats aliases are removed.
        s0_alloc = tmem_s0.get_tmem_requirements()[0]
        s1_alloc = tmem_s1.get_tmem_requirements()[0]
        o_alloc = tmem_o.get_tmem_requirements()[0]
        stats0_requirements = tmem_softmax_local0.get_tmem_requirements()
        stats1_requirements = tmem_softmax_local1.get_tmem_requirements()
        if stats0_requirements:
            assert len(stats0_requirements) == len(stats1_requirements) == 1
            stats0_requirements[0].offset = 0
            stats1_requirements[0].offset = cfg.tmem_stats_cols
            o_alloc.offset = 2 * cfg.tmem_stats_cols
        else:
            assert not stats1_requirements
            o_alloc.offset = 0
        s0_alloc.offset = o_alloc.offset + o_alloc.num_columns
        s1_alloc.offset = s0_alloc.offset + cfg.tmem_s_cols
        assert s1_alloc.offset + cfg.tmem_s_cols == cfg.tmem_total_cols
    elif cfg.uses_two_inst_tmem_p:
        # P reuses each independent S region after Softmax consumes QK. Stats
        # are either standalone TMEM or an SMEM handoff.
        s0_alloc = tmem_s0.get_tmem_requirements()[0]
        s1_alloc = tmem_s1.get_tmem_requirements()[0]
        p0_alloc = smem_p0.get_tmem_requirements()[0]
        p1_alloc = smem_p1.get_tmem_requirements()[0]
        o_alloc = tmem_o.get_tmem_requirements()[0]
        if cfg.streams_tmem_p_fragments:
            # Streamed profiles keep O in the low 256 columns and overlay
            # packed P on each S region from its first column. Softmax streams
            # K32 fragments in order, so every 16-column P store only
            # overwrites scores that have already been consumed. Starting P
            # after the nominal stats columns would instead clobber the next
            # unread S fragment; streamed profiles keep their softmax stats in
            # SMEM.
            assert cfg.keeps_stats_via_smem
            o_alloc.offset = 0
            s0_alloc.offset = 2 * cfg.tmem_o_stage_cols
            s1_alloc.offset = s0_alloc.offset + cfg.tmem_s_cols
            p0_alloc.offset = s0_alloc.offset
            p1_alloc.offset = s1_alloc.offset
        else:
            # Re-state the intended phase offsets after layout so every
            # resource observes the same S/P alias. Stats remain standalone
            # when the whole allocation fits; otherwise they use SMEM. Keep
            # the historical gap before P so this modeling cleanup does not
            # change runtime addresses.
            s0_alloc.offset = 0
            s1_alloc.offset = cfg.tmem_s_cols
            p0_alloc.offset = (
                0 if cfg.keeps_separates_tmem_s_and_stats else cfg.tmem_stats_cols
            )
            p1_alloc.offset = cfg.tmem_s_cols + (
                0 if cfg.keeps_separates_tmem_s_and_stats else cfg.tmem_stats_cols
            )
            if cfg.keeps_separates_tmem_s_and_stats:
                stats0_alloc = tmem_softmax_local0.get_tmem_requirements()[0]
                stats1_alloc = tmem_softmax_local1.get_tmem_requirements()[0]
                stats0_alloc.offset = 2 * cfg.tmem_s_cols
                stats1_alloc.offset = 2 * cfg.tmem_s_cols + cfg.tmem_stats_cols
                o_alloc.offset = 2 * (cfg.tmem_s_cols + cfg.tmem_stats_cols)
            else:
                o_alloc.offset = 2 * cfg.tmem_s_cols
        expected_p_cols = cfg.tmem_p_cols_per_inst
        assert p0_alloc.num_columns == p1_alloc.num_columns == expected_p_cols
        assert s0_alloc.offset <= p0_alloc.offset
        assert p0_alloc.offset + expected_p_cols <= s0_alloc.offset + cfg.tmem_s_cols
        assert s1_alloc.offset <= p1_alloc.offset
        assert p1_alloc.offset + expected_p_cols <= s1_alloc.offset + cfg.tmem_s_cols
        assert (
            max(
                s0_alloc.offset + cfg.tmem_s_cols,
                s1_alloc.offset + cfg.tmem_s_cols,
                o_alloc.offset + cfg.tmem_o_stage_cols * cfg.o_stages,
            )
            == cfg.tmem_total_cols
        )
    if use_one_inst_qkv:
        tmem_s_alloc = tmem_s0.get_tmem_requirements()[0]
        # One-inst Keeps also transports stats through SMEM, so there is no
        # TMEM stats allocation to alias with S.
        assert cfg.keeps_stats_via_smem
        assert cfg.tmem_stats_cols + cfg.tmem_p_cols <= cfg.tmem_s_cols
        smem_p0.get_tmem_requirements()[0].offset = (
            tmem_s_alloc.offset + cfg.tmem_stats_cols
        )

    eager_init_resources = [tmem_corr0] if use_one_inst_kv else [tmem_corr0, tmem_corr1]
    if cfg.streams_tmem_p_fragments:
        # Streamed TMEM P operands use one-way per-fragment ready barriers.
        # Initialize them beside correction's manually managed SMEM state.
        eager_init_resources.extend([smem_p0, smem_p1])
    if cfg.uses_int32_scores:
        # The score-seed operand tile is written once by its owning instance.
        eager_init_resources.append(tmem_s0)

    for task in task_list:
        task.requested_splits = requested_splits
    return (
        task_list,
        resource_dependency_graph,
        dma_consumer_release_labels,
        smem_allocator,
        tmem_allocator,
        eager_init_resources,
    )


def compute_decode_gen_loop_domain(total_kv_tiles: int, num_insts_kv: int) -> int:
    """Number of post-head steady-state iterations.

    HEAD consumes the first `num_insts_kv` K tiles. The remaining tiles are
    processed in staggered groups of `num_insts_kv`, so odd tail groups still
    need one final loop iteration, matching the schedule's pull-down behavior.
    """
    remaining_kv_tiles = max(total_kv_tiles - num_insts_kv, 0)
    return (remaining_kv_tiles + num_insts_kv - 1) // num_insts_kv


def compute_local_kv_tiles(cfg: FmhaDecodeConfig, total_kv_tiles: int) -> int:
    """KV tiles covered by each CtaKv in split-KV mode."""
    if not cfg.use_split_kv:
        return total_kv_tiles
    tiles_per_cta_group = cfg.splits_kv * cfg.num_insts_kv
    num_groups = (total_kv_tiles + tiles_per_cta_group - 1) // tiles_per_cta_group
    return max(
        cfg.num_insts_kv,
        num_groups * cfg.num_insts_kv,
    )


def resolve_block_sparse_per_inst_load_topology(
    cfg: FmhaDecodeConfig,
    *,
    use_clc_dynamic: bool,
) -> tuple[tuple[int, int], tuple[int, int] | None] | None:
    """Reuse idle WG3 padding warps for two independent sparse load streams.

    The return value contains the two load warp indices and an optional
    residual padding task ``(warp_idx, num_warps)``. ``None`` identifies a
    noncanonical override for which the caller keeps the common load task.
    """

    if cfg.load_num_warps != 1:
        return None
    padding_warps = tuple(
        range(
            cfg.wg3_padding_warp_idx,
            cfg.wg3_padding_warp_idx + cfg.wg3_padding_num_warps,
        )
    )
    if use_clc_dynamic:
        # Load1 consumes the only otherwise-idle warp in WG3.
        if cfg.scheduler_num_warps != 1 or len(padding_warps) != 1:
            return None
        role_warps = (
            cfg.mma_warp_idx,
            cfg.scheduler_warp_idx,
            cfg.clc_load_warp_idx,
            padding_warps[0],
        )
        if len(set(role_warps)) != len(role_warps):
            return None
        if len({warp_idx // 4 for warp_idx in role_warps}) != 1:
            return None
        return (cfg.clc_load_warp_idx, padding_warps[0]), None

    if len(padding_warps) != 2:
        return None
    role_warps = (cfg.mma_warp_idx, cfg.load_warp_idx, *padding_warps)
    if len(set(role_warps)) != len(role_warps):
        return None
    if len({warp_idx // 4 for warp_idx in role_warps}) != 1:
        return None
    return (cfg.load_warp_idx, padding_warps[0]), (padding_warps[1], 1)


def stages_page_ids_per_tile(uses_paired_page_offset_resources: bool) -> bool:
    """Return whether each published page-offset stage owns exactly one tile."""

    # Shared resources retain an aligned 32-ID window so all lanes issue one
    # coalesced page-table transaction. Paired K0/K1 and V0/V1 resources use
    # exact per-tile stages so a pair may safely cross a 32-ID boundary.
    return uses_paired_page_offset_resources
