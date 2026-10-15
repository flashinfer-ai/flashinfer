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

"""SMEM staging resources for Q, page offsets, and K/V tiles."""

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from cutlass.experimental import primitives as prims
from ....tensor_map import transform_ragged_coords

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import Boolean, dsl_user_op, if_generate
from cutlass.pipeline import PipelineAsync, PipelineState
from cutlass.experimental import primitives as cprims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...helpers.constants import (
    CP_ASYNC_CACHE_CA,
    PAGE_OFFSET_BYTES,
    WARP_LANES,
)

from ...helpers.layout import (
    decode_gen_task_cache,
    head_dim_cta_offset_v,
    q_stage_smem_element_offset,
    smem_array,
    tma_inner_dim_elems,
)
from ...helpers.math import (
    qkv_dtype,
    qkv_major_k_stride_bytes_for,
    qkv_smem_swizzle,
    qkv_smem_swizzle_for_head_dim,
)
from ...helpers.gather4 import issue_gather4_tma_cta
from ...helpers.ops import warp_ballot_sync
from ...helpers.nvfp4 import (
    fence_proxy_async_shared,
    mbarrier_try_wait_parity_u32,
    cp_async_commit_group,
    cp_async_wait_group,
    st_shared_u8,
    cp_async_8b_zfill,
    f32_to_f16x2,
    ld_shared_v2,
    ld_shared_v4,
    qmul4_hi,
    qmul4_lo,
    scaled_e4m3_pair_broadcast,
    st_shared_v4,
    warp_sync_all,
)
from ...helpers.ops import lane_idx_from_thread
from ...helpers.query import query_batch_bounds
from ...helpers.stage import MlaStage
from ...helpers.tile import (
    batch_idx_for_stage_cfg,
    cta_idx_head_dim_v_for_stage,
    cta_idx_kv_for_stage,
    cta_idx_q_for_stage,
    global_kv_tile_idx,
    head_idx_for_stage,
    local_kv_tile_idx,
    runtime_local_kv_tiles,
    runtime_seq_len_kv_from_task_cache,
    staged_kv_head_dim_call_idx,
)

from .common import (
    TCGEN05_BF16_K_BLOCK_WIDTH,
    TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES,
    TCGEN05_BF16_SWIZZLE_STRIDE_BYTES,
    MlaResource,
)

# Token-sparse per-key validity ring: ``SPARSE_MASK_SLOTS`` tiles x four
# 32-bit words. Word ``j`` bit ``l`` is token ``4 * l + j`` of the 128-token
# tile (lane ``l`` of the index-producer warp stages tokens ``4l .. 4l + 3``).
SPARSE_MASK_SLOTS = 16
SPARSE_MASK_WORDS = 4

# =====================================================================
# SmemQResource — Q SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemQResource(MlaResource):
    """Stage Q latent/RoPE tiles in SMEM and publish Q MMA descriptors."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "q_desc_var",
            Int64,
            Int64(0),
            "SMEM descriptor for the staged Q latent tile.",
            "q_desc",
        ),
        (
            "q_desc_rope_var",
            Int64,
            Int64(0),
            "SMEM descriptor for the staged Q rope tile.",
            "q_desc_rope",
        ),
    )
    tma_desc_q_latent: object = None
    tma_desc_q_rope: object = None
    head_idx: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    _smem_q: object = None
    _q_desc_base: object = None
    q_desc_var: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    q_desc_rope_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def get_smem_requirements(self):
        """Return the SMEM allocation used for staged Q tiles."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.qk_smem_tile_bytes * self.cfg.q_stages,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Create the Q SMEM view and base descriptor state."""
        context = stage_info.context
        elem_count = (
            self.cfg.qk_smem_tile_bytes * self.cfg.q_stages
        ) // self.cfg.qkv_dtype_bytes
        self._smem_q = smem_array(context, self._alloc, qkv_dtype(self.cfg), elem_count)
        if self._smem_q is not None:
            # The BF16 leading offset advances by at most one 64-wide K block.
            # FP8 uses a dtype-specific major-K stride and therefore rebuilds
            # both descriptor offsets below.
            q_leading_byte_offset = Int32(
                self.cfg.tile_size_q
                * min(self.cfg.head_dim_per_stage_kv, TCGEN05_BF16_K_BLOCK_WIDTH)
                * self.cfg.qkv_dtype_bytes
            )
            q_stride_byte_offset = Int32(TCGEN05_BF16_SWIZZLE_STRIDE_BYTES)
            if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                q_leading_byte_offset = Int32(
                    self.cfg.tile_size_q
                    * self.cfg.head_dim_per_stage_kv
                    * self.cfg.qkv_dtype_bytes
                )
                q_stride_byte_offset = Int32(
                    qkv_major_k_stride_bytes_for(
                        self.cfg, self.cfg.head_dim_per_stage_kv
                    )
                )
            self._q_desc_base = cprims.Tcgen05SmemDesc.build(
                self._smem_q,
                leading_byte_offset=q_leading_byte_offset,
                stride_byte_offset=q_stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize the Q SMEM view for producer-side TMA loads."""

        # Producer aux work runs before the load task starts issuing Q TMA
        # copies.  It creates the SMEM view used by producer_work(load_q).
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize the Q SMEM view for consumer-side MMA descriptors."""

        # Consumer aux work creates the same SMEM view in the MMA task so the
        # consumer can build descriptors after it waits on the Q stage.
        self._init_smem_state(stage_info)

    @cute.jit
    def _query_tma_coords(self, dim_offset, local_flat_query_row, batch_idx):
        """Return fixed or ragged TMA coordinates for one Q slice."""

        if cutlass.const_expr(self.cu_seqlens_q is None):
            return (dim_offset, local_flat_query_row, batch_idx)

        query_start, query_length = query_batch_bounds(
            self.cu_seqlens_q,
            batch_idx,
            self.cfg.logical_seq_len_q,
        )
        storage_flat_query_row = (
            query_start * Int32(self.cfg.logical_num_heads_q) + local_flat_query_row
        )
        ragged_extent = (
            query_length * Int32(self.cfg.logical_num_heads_q) - local_flat_query_row
        )
        return transform_ragged_coords(
            (dim_offset, storage_flat_query_row),
            ragged_dim_idx=1,
            ragged_box_size=self.cfg.tile_size_q,
            ragged_extent=ragged_extent,
        )

    @cute.jit
    def _load_q_stage(
        self,
        stage_info: StageInfo,
        stage_base,
        qk_stage_idx: int,
        dim_offset: Int32,
        tma_desc,
    ):
        """Issue one staged TMA load for a Q head-dimension slice."""
        if cutlass.const_expr(tma_desc is None):
            return
        active_width = self.cfg.qk_head_stage_width(qk_stage_idx)
        inner_width = min(tma_inner_dim_elems(self.cfg), active_width)
        head_idx = head_idx_for_stage(self.head_idx, self.cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, self.cfg, stage_info)
        # Ragged storage adds the cumulative batch row offset while preserving
        # the unclamped logical row so a fully padded CTA becomes an OOB
        # zero-fill instead of reloading the final real query row.
        local_flat_query_row = cta_idx_q * Int32(self.cfg.num_heads_q) + head_idx
        query_coords = self._query_tma_coords(
            dim_offset,
            local_flat_query_row,
            batch_idx,
        )
        if prims.elect_sync():
            smem_offset = Int32(q_stage_smem_element_offset(self.cfg, qk_stage_idx))
            prims.cp_async_bulk_tensor_shared_cta_global(
                stage_base.data_ptr(smem_offset),
                tma_desc,
                query_coords,
                stage_info.barrier,
            )
            if cutlass.const_expr(active_width > inner_width):
                second_query_coords = self._query_tma_coords(
                    dim_offset + Int32(inner_width),
                    local_flat_query_row,
                    batch_idx,
                )
                prims.cp_async_bulk_tensor_shared_cta_global(
                    stage_base.data_ptr(
                        smem_offset + Int32(inner_width * self.cfg.tile_size_q)
                    ),
                    tma_desc,
                    second_query_coords,
                    stage_info.barrier,
                )

    @producer_work
    @cute.jit
    def load_q(self, stage_info: StageInfo):
        """Load the grouped-head Q tile as 8 latent chunks plus 1 rope chunk."""
        # This is the actual Q payload producer: the load task fills the
        # acquired Q SMEM stage and the pipeline commit makes it visible to MMA.
        if cutlass.const_expr(
            self.tma_desc_q_latent is None or self.tma_desc_q_rope is None
        ):
            return
        cfg = self.cfg
        if cutlass.const_expr(cfg.load_num_warps > 1):
            # Multi-warp loader: only the task's first warp issues the Q TMAs.
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == Int32(cfg.load_warp_idx):
                self._load_q_all_stages(stage_info)
            return
        self._load_q_all_stages(stage_info)

    @cute.jit
    def _load_q_all_stages(self, stage_info: StageInfo):
        cfg = self.cfg
        stage_elems = cfg.q_smem_tile_elements
        stage_base = self._smem_q.subview(stage_info.stage_idx * Int32(stage_elems))
        latent_stages = cfg.latent_dim // cfg.head_dim_per_stage_kv
        for qk_stage_idx in cutlass.range_constexpr(latent_stages):
            dim_offset = Int32(qk_stage_idx * cfg.head_dim_per_stage_kv)
            self._load_q_stage(
                stage_info,
                stage_base,
                qk_stage_idx,
                dim_offset,
                self.tma_desc_q_latent,
            )
        self._load_q_stage(
            stage_info,
            stage_base,
            latent_stages,
            Int32(0),
            self.tma_desc_q_rope,
        )

    @consumer_work(returns=("q_desc", "q_desc_rope"))
    @cute.jit
    def q_desc(self, stage_info: StageInfo):
        """Publish the Q SMEM descriptor for staged QK MMA."""
        # The MMA task has waited for the Q stage.  It converts the live SMEM
        # stage into descriptors and returns them as task-local values for the
        # downstream TmemSResource producer_work(qk_mma).
        stage_base = self._smem_q.subview(
            stage_info.stage_idx * Int32(self.cfg.q_smem_tile_elements)
        )
        # Build the same descriptor as producer init, but relative to the live
        # pipeline stage that the MMA task has already waited on.
        q_leading_byte_offset = Int32(
            self.cfg.tile_size_q
            * min(self.cfg.head_dim_per_stage_kv, TCGEN05_BF16_K_BLOCK_WIDTH)
            * self.cfg.qkv_dtype_bytes
        )
        q_stride_byte_offset = Int32(TCGEN05_BF16_SWIZZLE_STRIDE_BYTES)
        if cutlass.const_expr(self.cfg.is_fp8_qkv()):
            q_leading_byte_offset = Int32(
                self.cfg.tile_size_q
                * self.cfg.head_dim_per_stage_kv
                * self.cfg.qkv_dtype_bytes
            )
            q_stride_byte_offset = Int32(
                qkv_major_k_stride_bytes_for(self.cfg, self.cfg.head_dim_per_stage_kv)
            )
        desc_q = cprims.Tcgen05SmemDesc.build(
            stage_base,
            leading_byte_offset=q_leading_byte_offset,
            stride_byte_offset=q_stride_byte_offset,
            layout=qkv_smem_swizzle(self.cfg),
        )
        desc_q_rope = desc_q
        if cutlass.const_expr(self.cfg.is_fp8_qkv() and self.cfg.rope_dim == 64):
            rope_stage_idx = self.cfg.latent_dim // self.cfg.head_dim_per_stage_kv
            rope_stage_offset = Int32(
                q_stage_smem_element_offset(self.cfg, rope_stage_idx)
            )
            desc_q_rope = cprims.Tcgen05SmemDesc.build(
                stage_base.subview(rope_stage_offset),
                leading_byte_offset=Int32(
                    self.cfg.tile_size_q * self.cfg.rope_dim * self.cfg.qkv_dtype_bytes
                ),
                stride_byte_offset=Int32(
                    qkv_major_k_stride_bytes_for(self.cfg, self.cfg.rope_dim)
                ),
                layout=qkv_smem_swizzle_for_head_dim(self.cfg, self.cfg.rope_dim),
            )
        return desc_q, desc_q_rope


# =====================================================================
# SmemPageOffsetsResource — Page-offset SMEM buffer with Async pipeline
# =====================================================================


@dataclass(frozen=True)
class _StructuredWaitPipelineAsync(PipelineAsync):
    """Page-offset pipeline with a public structured mbarrier retry loop."""

    @cute.jit
    def _retry_wait(
        self,
        sync_object: object,
        state: PipelineState,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        while not sync_object.try_wait(
            state.index,
            state.phase,
            loc=loc,
            ip=ip,
        ):
            pass

    @dsl_user_op
    def producer_acquire(
        self,
        state: PipelineState,
        try_acquire_token: Optional[Boolean] = None,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        if_generate(
            try_acquire_token is None or try_acquire_token == 0,
            lambda: self._retry_wait(self.sync_object_empty, state, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def consumer_wait(
        self,
        state: PipelineState,
        try_wait_token: Optional[Boolean] = None,
        *,
        loc: Any = None,
        ip: Any = None,
    ) -> None:
        if_generate(
            try_wait_token is None or try_wait_token == 0,
            lambda: self._retry_wait(self.sync_object_full, state, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )


@dataclass(kw_only=True)
class SmemPageOffsetsResource(MlaResource):
    """Prefetch page ids for the K/V tile currently owned by the load task."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "cached_page_ids",
            cutlass.Array,
            None,
            "Cached logical page ids for the staged K/V tile.",
        ),
    )
    page_offsets: object = None
    cache_seqs: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    cta_idx_kv: object = None
    _smem_page_offsets: object = None
    cached_page_ids: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def create_pipeline(self, pipeline_config):
        """Preserve stock barrier objects and specialize only their wait path."""
        base = super().create_pipeline(pipeline_config)
        if not isinstance(base, PipelineAsync):
            raise TypeError(
                "page-offset staging requires cutlass.pipeline.PipelineAsync, "
                f"got {type(base).__name__}"
            )
        return _StructuredWaitPipelineAsync(
            sync_object_full=base.sync_object_full,
            sync_object_empty=base.sync_object_empty,
            num_stages=base.num_stages,
            producer_mask=base.producer_mask,
            consumer_mask=base.consumer_mask,
        )

    @property
    def sparse_mask_word_base(self) -> int:
        """Int32 index of the token-sparse validity ring inside this allocation."""
        return self.cfg.page_offsets_stages * self.cfg.page_offsets_entries_per_stage

    @property
    def total_words(self) -> int:
        words = self.sparse_mask_word_base
        if self.cfg.is_dynamic_token_sparse:
            words += SPARSE_MASK_SLOTS * SPARSE_MASK_WORDS
        return words

    def get_smem_requirements(self):
        """Return the SMEM allocation for cached page offsets (+ sparse validity ring)."""
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.total_words * PAGE_OFFSET_BYTES,
                alignment=128,
            )
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Create the SMEM page-offset array view."""
        context = stage_info.context
        count = self.total_words
        self._smem_page_offsets = smem_array(
            context,
            self._alloc,
            Int32,
            count,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize page-offset SMEM before producer copies begin."""

        # Producer aux work creates the page-offset SMEM view for the page load
        # warp before it issues cp.async copies.
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=cached_page_ids)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo):
        """Create register caches for the active K/V page-ID streams."""

        # Consumer aux work allocates the register cache that read_offsets()
        # returns after the page-offset stage is ready.
        self._init_smem_state(stage_info)
        cache_slots = 2 if self.cfg.kernel_variant != "keeps_mma_ab" else 1
        cache_entries = max(1, cache_slots * self.cfg.pages_per_kv_tile)
        if self.cfg.is_dynamic_token_sparse:
            # The Gather4 loader reads its four row indices per lane straight
            # from the SMEM stage; a 128-entry register cache would not fit.
            cache_entries = 1
        return cutlass.Array(
            Int32,
            cache_entries,
            space=cutlass.AddressSpace.rmem,
        )

    @cute.jit
    def page_id(self, page_fragment_idx: int):
        """Return one cached logical page id from the consumer stage."""
        offset = self.consumer_work_stage * Int32(
            self.cfg.page_offsets_entries_per_stage
        ) + Int32(page_fragment_idx)
        return Int32(self._smem_page_offsets[offset])

    @cute.jit
    def token_row(self, entry_idx):
        """Return one staged token row index (token-sparse stages, dynamic entry)."""
        offset = self.consumer_work_stage * Int32(
            self.cfg.page_offsets_entries_per_stage
        ) + Int32(entry_idx)
        return Int32(self._smem_page_offsets[offset])

    @cute.jit
    def _producer_load_page_offsets(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        section: cutlass.Constexpr[MlaStage],
    ):
        """Load page ids for one K or V producer stage into SMEM."""
        if cutlass.const_expr(self.page_offsets is None):
            return
        if cutlass.const_expr(self.cfg.is_dynamic_token_sparse):
            self._producer_load_token_rows(stage_info, inst_id, is_v, section=section)
            return
        cfg = self.cfg
        lane_idx = lane_idx_from_thread(cute.arch.thread_idx()[0])
        pages_per_tile = Int32(cfg.pages_per_kv_tile)
        entries_per_lane = 1
        for entry_sub in cutlass.range_constexpr(entries_per_lane):
            entry_idx = lane_idx * Int32(entries_per_lane) + Int32(entry_sub)
            if entry_idx < pages_per_tile:
                batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
                cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
                cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
                seq_len_kv = runtime_seq_len_kv_from_task_cache(
                    cfg,
                    decode_gen_task_cache(stage_info),
                    cta_idx_q,
                    self.cu_seqlens_q,
                    batch_idx,
                )
                local_tile_idx = local_kv_tile_idx(
                    cfg, stage_info, inst_id, is_v, section=section
                )
                tile_idx = global_kv_tile_idx(
                    cfg, local_tile_idx, seq_len_kv, cta_idx_kv
                )
                last_valid_page = cute.math.max(
                    (seq_len_kv + Int32(cfg.num_tokens_per_page - 1))
                    // Int32(cfg.num_tokens_per_page)
                    - Int32(1),
                    Int32(0),
                )
                logical_page_idx = cute.math.min(
                    tile_idx * pages_per_tile + entry_idx,
                    last_valid_page,
                )
                smem_offset = (
                    stage_info.stage_idx * Int32(cfg.page_offsets_entries_per_stage)
                    + entry_idx
                )
                page_offsets_batch = self.page_offsets[None, batch_idx]
                page_offsets_flat = cute.flat_divide(page_offsets_batch, (1,))
                gmem_ptr = page_offsets_flat[None, logical_page_idx].iterator.llvm_ptr
                smem_ptr = cutlass.inttoptr(
                    self._smem_page_offsets.data_ptr(smem_offset).toint(cutlass.Int32),
                    3,
                    cutlass.Int32,
                )
                prims.cp_async_shared_global(
                    smem_ptr, gmem_ptr, PAGE_OFFSET_BYTES, CP_ASYNC_CACHE_CA
                )

    @cute.jit
    def _producer_load_token_rows(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        section: cutlass.Constexpr[MlaStage],
    ):
        """Token-sparse stage: lane ``l`` stages row indices of tokens ``4l .. 4l+3``.

        Four independent 4-byte ``cp.async`` copies per lane keep the producer
        fully asynchronous (a synchronous load per stage would serialize the
        global-memory latency into the K/V cadence).  Entries past the token's
        active width are clamped to its last entry; those keys are removed by
        the KV-length tail mask.  The per-key validity words are published by
        the Gather4 loader, which already holds the indices in registers.
        """
        cfg = self.cfg
        lane_idx = lane_idx_from_thread(cute.arch.thread_idx()[0])
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            cfg,
            decode_gen_task_cache(stage_info),
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        local_tile_idx = local_kv_tile_idx(
            cfg, stage_info, inst_id, is_v, section=section
        )
        tile_idx = global_kv_tile_idx(cfg, local_tile_idx, seq_len_kv, cta_idx_kv)
        last_valid = cute.math.max(seq_len_kv - Int32(1), Int32(0))
        stage_base = stage_info.stage_idx * Int32(cfg.page_offsets_entries_per_stage)
        tile_entry_base = tile_idx * Int32(cfg.tile_size_kv) + lane_idx * Int32(4)
        page_offsets_batch = self.page_offsets[None, batch_idx]
        page_offsets_flat = cute.flat_divide(page_offsets_batch, (1,))
        for entry_sub in cutlass.range_constexpr(4):
            logical_idx = cute.math.min(tile_entry_base + Int32(entry_sub), last_valid)
            smem_offset = stage_base + lane_idx * Int32(4) + Int32(entry_sub)
            gmem_ptr = page_offsets_flat[None, logical_idx].iterator.llvm_ptr
            smem_ptr = cutlass.inttoptr(
                self._smem_page_offsets.data_ptr(smem_offset).toint(cutlass.Int32),
                3,
                cutlass.Int32,
            )
            prims.cp_async_shared_global(
                smem_ptr, gmem_ptr, PAGE_OFFSET_BYTES, CP_ASYNC_CACHE_CA
            )

    @producer_work
    @cute.jit
    def load_k0(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[MlaStage],
    ):
        """Prefetch page offsets for K instance 0."""

        # Producer work for the page-offset stage that feeds K instance 0.
        self._producer_load_page_offsets(stage_info, 0, 0, section=section)

    @producer_work
    @cute.jit
    def load_k1(self, stage_info: StageInfo, *, section: cutlass.Constexpr[MlaStage]):
        """Prefetch page offsets for K instance 1."""

        # Producer work for the page-offset stage that feeds K instance 1.
        self._producer_load_page_offsets(stage_info, 1, 0, section=section)

    @producer_work
    @cute.jit
    def load_v0(self, stage_info: StageInfo, *, section: cutlass.Constexpr[MlaStage]):
        """Prefetch deferred page offsets for V instance 0."""

        # Producer work for the deferred page-offset stage that feeds V instance 0.
        self._producer_load_page_offsets(stage_info, 0, 1, section=section)

    @producer_work
    @cute.jit
    def load_v1(self, stage_info: StageInfo, *, section: cutlass.Constexpr[MlaStage]):
        """Prefetch deferred page offsets for V instance 1."""

        # Producer work for the deferred page-offset stage that feeds V instance 1.
        self._producer_load_page_offsets(stage_info, 1, 1, section=section)

    @consumer_work(returns=cached_page_ids)
    @cute.jit
    def read_offsets(
        self,
        stage_info: StageInfo,
        *,
        cached_page_ids,
        cache_slot: cutlass.Constexpr[int] = 0,
    ):
        """Cache staged page IDs in the selected K stream's register slot."""
        # The load task waits on the page-offset stage, snapshots the page ids
        # into registers, and keeps swaps K0/K1 live until their delayed V use.
        del stage_info
        if cutlass.const_expr(self.cfg.is_dynamic_token_sparse):
            return cached_page_ids
        cache_base = cache_slot * self.cfg.pages_per_kv_tile
        for page_frag in cutlass.range_constexpr(self.cfg.pages_per_kv_tile):
            cached_page_ids[cache_base + page_frag] = self.page_id(page_frag)
        return cached_page_ids


# =====================================================================
# SmemKvResource — K/V SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemKvResource(MlaResource):
    """Shared K/V staging resource for latent and rope tiles."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "kv_desc_var",
            Int64,
            Int64(0),
            "SMEM descriptor for staged K tiles.",
            "kv_desc",
        ),
        (
            "v_desc_0_var",
            Int64,
            Int64(0),
            "First V descriptor consumed by PV MMA.",
            "v_desc_0",
        ),
        (
            "v_desc_1_var",
            Int64,
            Int64(0),
            "Second V descriptor consumed by PV MMA.",
            "v_desc_1",
        ),
    )
    tma_desc_c_latent: object = None
    tma_desc_c_rope: object = None
    tma_desc_v: object = None
    c_rope_tensor: object = None
    # Token-sparse: GMEM tensor holding two encoded CUtensorMaps (128 B each)
    # over the flat ``[576 B, N]`` KV pool: [0] 128 B box / SWIZZLE_128B for the
    # latent D128 slices, [1] 64 B box / SWIZZLE_64B for the RoPE slice.
    raw_gather4_descs: object = None
    # Token-sparse NVFP4 loader-owned indices: flat ``[N * scale_row_bytes]``
    # uint8 scale pool staged by the loader lanes.
    kv_scale_pool: object = None
    # Token-sparse NVFP4 dequant producer inputs: the raw-KV ring resource and
    # the per-tensor dequantization factor folded into the block scales.
    raw_kv: object = None
    kv_dequant_scale: object = None
    page_offsets_kv: object = None
    page_offsets: object = None
    cache_seqs: object = None
    head_idx: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    cta_idx_kv: object = None
    cta_idx_head_dim_v: object = None
    _smem_kv: object = None
    _k_desc_base: object = None
    _v_desc_base: object = None
    kv_desc_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    v_desc_0_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    v_desc_1_var: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def get_smem_requirements(self):
        """Return the SMEM allocation used for staged K/V tiles."""
        num_stages = (
            self.pipeline_config.num_stages
            if self.pipeline_config is not None
            else self.cfg.kv_stages
        )
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.kv_smem_tile_bytes * num_stages,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        """Create K/V SMEM views and base descriptors."""
        context = stage_info.context
        num_stages = (
            self.pipeline_config.num_stages
            if self.pipeline_config is not None
            else self.cfg.kv_stages
        )
        elem_count = (
            self.cfg.kv_smem_tile_bytes * num_stages
        ) // self.cfg.qkv_dtype_bytes
        self._smem_kv = smem_array(
            context, self._alloc, qkv_dtype(self.cfg), elem_count
        )
        if self._smem_kv is not None:
            # FP8 descriptors use dtype-specific major-K strides rather than
            # the BF16 128B-row swizzle group.
            k_leading_byte_offset = Int32(TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES)
            v_leading_byte_offset = (
                Int32(0)
                if self.cfg.head_dim_per_stage_v == TCGEN05_BF16_K_BLOCK_WIDTH
                else Int32(TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES)
            )
            stride_byte_offset = Int32(TCGEN05_BF16_SWIZZLE_STRIDE_BYTES)
            if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                k_leading_byte_offset = Int32(self.cfg.kv_smem_tile_bytes)
                v_leading_byte_offset = Int32(0)
                stride_byte_offset = Int32(
                    qkv_major_k_stride_bytes_for(
                        self.cfg, self.cfg.head_dim_per_stage_kv
                    )
                )
            self._k_desc_base = cprims.Tcgen05SmemDesc.build(
                self._smem_kv,
                leading_byte_offset=k_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )
            self._v_desc_base = cprims.Tcgen05SmemDesc.build(
                self._smem_kv,
                leading_byte_offset=v_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize K/V SMEM state before producer TMA loads."""

        # Producer aux work initializes K/V SMEM views for the load task.
        self._init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize K/V SMEM state before descriptor consumption."""

        # Consumer aux work initializes K/V SMEM views for the MMA task.
        self._init_smem_state(stage_info)

    @cute.jit
    def _stage_base(self, stage_info: StageInfo):
        """Return the base of the current K/V SMEM pipeline stage."""

        stage_elems = self.cfg.kv_smem_stage_elements
        return self._smem_kv.subview(stage_info.stage_idx * Int32(stage_elems))

    @cute.jit
    def _producer_gather_token_rows(
        self,
        stage_info: StageInfo,
        stage_base,
        *,
        dim_offset,
        active_width: cutlass.Constexpr[int],
        is_rope_stage: cutlass.Constexpr[bool],
        uses_compact_fp8_rope_stage: cutlass.Constexpr[bool],
        publish_mask: cutlass.Constexpr[bool] = False,
        tile_idx=None,
    ):
        """Token-sparse K/V stage: 32 lanes x one Gather4 of four token rows each.

        Lane ``l`` owns tokens ``4l .. 4l+3`` of the 128-token tile.  It reads
        their physical row indices from the staged index ring and issues one
        Gather4 whose four ``active_width``-byte rows land at consecutive SMEM
        rows of the stage, so the TMA swizzle (SW128 latent / SW64 RoPE) is the
        one the UMMA descriptors expect.  Rows addressed by ``-1`` (or any
        out-of-range index) are zero-filled by the TMA unit and masked in
        softmax; they still count toward the stage's transaction bytes.
        """
        cfg = self.cfg
        lane_idx = lane_idx_from_thread(cute.arch.thread_idx()[0])
        ring = self.page_offsets_kv
        num_load_warps = cfg.load_num_warps
        warp_in_task = Int32(0)
        if cutlass.const_expr(num_load_warps > 1):
            warp_in_task = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
                cfg.load_warp_idx
            )
        descriptor = self.raw_gather4_descs.iterator
        if cutlass.const_expr(is_rope_stage):
            descriptor = descriptor + Int64(128)
        row_bytes = active_width
        if cutlass.const_expr(publish_mask):
            # First K slice of the tile: warp w publishes the validity word of
            # tokens 32w .. 32w+31 (bit = token & 31).  With one loader warp
            # all four words come from that warp.
            mask_base = Int32(ring.sparse_mask_word_base) + (
                Int32(tile_idx) % Int32(SPARSE_MASK_SLOTS)
            ) * Int32(SPARSE_MASK_WORDS)
            words_per_warp = max(1, SPARSE_MASK_WORDS // num_load_warps)
            for word_sub in cutlass.range_constexpr(words_per_warp):
                word_idx = warp_in_task * Int32(words_per_warp) + Int32(word_sub)
                # With more loader warps than mask words only the first
                # SPARSE_MASK_WORDS warps publish (one word each).
                if word_idx < Int32(SPARSE_MASK_WORDS):
                    row = ring.token_row(word_idx * Int32(32) + lane_idx)
                    valid = Int32(1) if row >= Int32(0) else Int32(0)
                    bits = warp_ballot_sync(valid)
                    if lane_idx == Int32(0):
                        ring._smem_page_offsets[mask_base + word_idx] = bits
        quads_per_warp = 32 // num_load_warps
        # Lane l < quads_per_warp issues quad (warp_in_task * quads_per_warp + l).
        quad = warp_in_task * Int32(quads_per_warp) + lane_idx
        q_base = quad * Int32(4)
        row0 = ring.token_row(q_base)
        row1 = ring.token_row(q_base + Int32(1))
        row2 = ring.token_row(q_base + Int32(2))
        row3 = ring.token_row(q_base + Int32(3))
        if lane_idx < Int32(quads_per_warp):
            issue_gather4_tma_cta(
                descriptor,
                stage_base.data_ptr(quad * Int32(4 * row_bytes)),
                stage_info.barrier,
                Int32(dim_offset),
                (row0, row1, row2, row3),
            )
        if cutlass.const_expr(uses_compact_fp8_rope_stage):
            if warp_in_task == Int32(0):
                if prims.elect_sync():
                    missing_stage_bytes = (
                        cfg.tile_size_kv
                        * (cfg.head_dim_per_stage_kv - active_width)
                        * cfg.qkv_dtype_bytes
                    )
                    prims.mbarrier_complete_tx(
                        stage_info.barrier, Int32(missing_stage_bytes)
                    )

    @producer_work
    @cute.jit
    def dequant_v_native(
        self,
        stage_info: StageInfo,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        inst: cutlass.Constexpr[int] = 0,
    ):
        """Dequantize V slice ``stage_idx`` of the K tile of this section.

        The tile follows the *K* cadence (Head: tile ``inst``, Loop n: tile
        ``n * num_insts_kv + num_insts_kv + inst``) because V is converted from
        the resident K latent slices right after the tile lands.
        """
        tile = local_kv_tile_idx(self.cfg, stage_info, inst, 0, section=section)
        self._dequant_stage(
            stage_info,
            0,
            1,
            stage_idx=stage_idx,
            section=section,
            native_tile_idx=tile,
        )

    @producer_work
    @cute.jit
    def dequant_v_native_regs(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[MlaStage],
        v3_raw,
        v3_scale,
        inst: cutlass.Constexpr[int] = 0,
    ):
        """Convert the last V slice from register-staged raw/scale words.

        The words were loaded by ``SmemRawKvResource.load_v_slice_regs`` before
        the raw stage was released, so this slice's wait for a free V-ring stage
        (PV of this tile) no longer holds the raw stage, and the loader can
        refill it one tile earlier.
        """
        tile = local_kv_tile_idx(self.cfg, stage_info, inst, 0, section=section)
        self._dequant_stage(
            stage_info,
            0,
            1,
            stage_idx=self.cfg.v_head_dim_stages - 1,
            section=section,
            native_tile_idx=tile,
            preloaded=(v3_raw, v3_scale),
        )

    @cute.jit
    def _dequant_stage(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        native_tile_idx,
        preload_only: cutlass.Constexpr[bool] = False,
        preloaded=None,
        raw_override=None,
    ):
        """Warp ``w`` converts tokens ``w * tpw + 8i + (l >> 2)``, raw chunk ``l & 3``.

        ``tpw = tile_size_kv / dequant_num_warps``.  V slices come from the K
        tile ring (one 40 KB stage per tile = [lat0 | lat1 | rope]); the dequant
        is not a registered consumer of that ring, so the stage is addressed by
        ``native_tile_idx``.

        Latent slices: 32 E2M1 per chunk x E4M3 block scale (x s_dequant) ->
        32 E4M3 written as two 16-byte chunks of the token's SW128 row.

        ``preload_only`` runs phase 1 only and returns the slice's raw and scale
        words as two register arrays; ``preloaded=(raw_arr, scale_arr)`` runs
        phase 2 from such arrays (V slice 3 is converted after the raw stage was
        released, see ``load_v_slice_regs``).
        """
        cfg = self.cfg
        qk_stage_idx = staged_kv_head_dim_call_idx(
            cfg, stage_info, inst_id, is_v, stage_idx=stage_idx, section=section
        )
        slot = native_tile_idx % Int32(cfg.nvfp4_scale_slots)
        raw = self.raw_kv if raw_override is None else raw_override
        raw_base = raw.smem_byte_address(
            (native_tile_idx % Int32(cfg.kv_raw_stages)) * Int32(cfg.kv_raw_stage_bytes)
            + Int32((qk_stage_idx // 2) * cfg.tile_size_kv * 128)
        )
        scales_base = raw.smem_byte_address(raw.scale_slot_byte_offset(slot))
        dst_base = self._stage_base(stage_info).data_ptr(Int32(0)).toint(cutlass.Int32)
        thread_idx = cute.arch.thread_idx()[0] - Int32(cfg.dequant_warp_idx * 32)
        warp_in_task = thread_idx >> Int32(5)
        lane_idx = thread_idx & Int32(31)
        chunk = lane_idx & Int32(3)
        factor = Int32(0)
        if cutlass.const_expr(not preload_only):
            factor = f32_to_f16x2(self.kv_dequant_scale)
        scale_byte_in_row = 8 * qk_stage_idx
        # prmt selector for scale bytes 2c and 2c+1 of this stage's 8-byte pair.
        pair_sel = (chunk * Int32(2)) | ((chunk * Int32(2) + Int32(1)) << Int32(4))
        tokens_per_warp = cfg.tile_size_kv // cfg.dequant_num_warps
        num_iters = tokens_per_warp // 8
        # Phase 1: issue every shared load of the slice up front.  The inline-asm
        # loads/stores carry memory side effects, so the compiler would otherwise
        # serialize load -> convert -> store per iteration and expose each LDS
        # latency (ncu: dequant warps ~10% issue, the rest scoreboard stalls).
        staged = []
        # Lane -> token within each 8-token group.  LDS.128 / STS.128 form one
        # wavefront per quarter-warp (8 lanes = 2 tokens x 4 chunks).  With the
        # SW128 swizzle (slot = chunk ^ (token & 7)) the two tokens of a quarter
        # must differ in bit 2 for the raw loads and in bit 0 for the E4M3 stores
        # to hit 8 distinct 16-byte slots, so quarter q covers tokens q and q ^ 5
        # (pairs (0,5) (1,4) (2,7) (3,6)); the naive (lane >> 2) mapping made
        # every dequant LDS/STS a 2-way conflict (ncu: 8 wavefronts, ideal 4).
        lane_token = (lane_idx >> Int32(3)) ^ (
            ((lane_idx >> Int32(2)) & Int32(1)) * Int32(5)
        )

        def _scale_words(token):
            return ld_shared_v2(
                scales_base
                + token * Int32(cfg.nvfp4_scale_row_bytes)
                + Int32(scale_byte_in_row)
            )

        if cutlass.const_expr(preloaded is not None):
            raw_arr, sc_arr = preloaded
            for it in cutlass.range_constexpr(num_iters):
                token = (
                    warp_in_task * Int32(tokens_per_warp) + Int32(it * 8) + lane_token
                )
                sc_words = (Int32(sc_arr[2 * it]), Int32(sc_arr[2 * it + 1]))
                staged.append(
                    (
                        token,
                        (
                            Int32(raw_arr[4 * it]),
                            Int32(raw_arr[4 * it + 1]),
                            Int32(raw_arr[4 * it + 2]),
                            Int32(raw_arr[4 * it + 3]),
                        ),
                        sc_words,
                    )
                )
        for it in cutlass.range_constexpr(0 if preloaded is not None else num_iters):
            token = warp_in_task * Int32(tokens_per_warp) + Int32(it * 8) + lane_token
            # 128-byte raw rows (SWIZZLE_128B) hold two slices; this slice
            # is half (qk_stage_idx & 1), chunk index XOR-swizzled by row.
            raw_addr = (
                raw_base
                + token * Int32(128)
                + (
                    ((Int32((qk_stage_idx & 1) * 4) + chunk) ^ (token & Int32(7)))
                    * Int32(16)
                )
            )
            raw_words = ld_shared_v4(raw_addr)
            scale_words = _scale_words(token)
            staged.append((token, raw_words, scale_words))
        if cutlass.const_expr(preload_only):
            raw_arr = cutlass.Array(
                Int32, 4 * num_iters, space=cutlass.AddressSpace.rmem
            )
            sc_arr = cutlass.Array(
                Int32, 2 * num_iters, space=cutlass.AddressSpace.rmem
            )
            for it in cutlass.range_constexpr(num_iters):
                _, (w0, w1, w2, w3), (s_lo, s_hi) = staged[it]
                raw_arr[4 * it] = w0
                raw_arr[4 * it + 1] = w1
                raw_arr[4 * it + 2] = w2
                raw_arr[4 * it + 3] = w3
                sc_arr[2 * it] = s_lo
                sc_arr[2 * it + 1] = s_hi
            return raw_arr, sc_arr
        # Phase 2: convert and store.
        for it in cutlass.range_constexpr(num_iters):
            token, (w0, w1, w2, w3), (s_lo, s_hi) = staged[it]
            sc0, sc1 = scaled_e4m3_pair_broadcast(s_lo, s_hi, pair_sel, factor)
            o0 = qmul4_lo(w0, sc0)
            o1 = qmul4_hi(w0, sc0)
            o2 = qmul4_lo(w1, sc0)
            o3 = qmul4_hi(w1, sc0)
            o4 = qmul4_lo(w2, sc1)
            o5 = qmul4_hi(w2, sc1)
            o6 = qmul4_lo(w3, sc1)
            o7 = qmul4_hi(w3, sc1)
            swz = token & Int32(7)
            row_addr = dst_base + token * Int32(128)
            c0 = ((chunk * Int32(2)) ^ swz) * Int32(16)
            c1 = ((chunk * Int32(2) + Int32(1)) ^ swz) * Int32(16)
            st_shared_v4(row_addr + c0, o0, o1, o2, o3)
            st_shared_v4(row_addr + c1, o4, o5, o6, o7)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def _producer_copy_rope_stage(
        self,
        stage_info: StageInfo,
        page_ids,
        stage_base,
    ):
        """Copy paged rope values into the staged K/V SMEM tile."""
        cfg = self.cfg
        lane_idx = lane_idx_from_thread(cute.arch.thread_idx()[0])
        total_elems = Int32(cfg.tile_size_kv * cfg.head_dim_per_stage_kv)
        for elem_idx in cutlass.range(
            lane_idx, total_elems, Int32(WARP_LANES), unroll=1
        ):
            token_idx = elem_idx // Int32(cfg.head_dim_per_stage_kv)
            dim_idx = elem_idx - token_idx * Int32(cfg.head_dim_per_stage_kv)
            src_dim_idx = dim_idx
            if dim_idx >= Int32(cfg.rope_dim):
                src_dim_idx = dim_idx - Int32(cfg.rope_dim)
            page_frag = token_idx // Int32(cfg.num_tokens_per_page)
            token_in_page = token_idx - page_frag * Int32(cfg.num_tokens_per_page)
            page_id = Int32(page_ids[page_frag])
            stage_base[elem_idx] = self.c_rope_tensor[
                token_in_page, src_dim_idx, page_id
            ]
        cute.arch.fence_view_async_shared()
        if prims.elect_sync():
            prims.mbarrier_complete_tx(
                stage_info.barrier, Int32(cfg.kv_smem_tile_bytes)
            )

    @cute.jit
    def _producer_load_kv_stage(
        self,
        stage_info: StageInfo,
        inst_id: int,
        is_v: int,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids=None,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load one staged K or V tile using the consumed page-id payload."""
        cfg = self.cfg
        qk_stage_idx = staged_kv_head_dim_call_idx(
            cfg,
            stage_info,
            inst_id,
            is_v,
            stage_idx=stage_idx,
            section=section,
        )
        if cutlass.const_expr(is_v):
            active_width = cfg.v_head_stage_width(qk_stage_idx)
            cta_idx_head_dim_v = cta_idx_head_dim_v_for_stage(
                self.cta_idx_head_dim_v, stage_info
            )
            dim_offset = head_dim_cta_offset_v(cfg, cta_idx_head_dim_v) + Int32(
                qk_stage_idx * cfg.head_dim_per_stage_v
            )
            tma_desc = self.tma_desc_v
        else:
            active_width = cfg.qk_head_stage_width(qk_stage_idx)
            if cutlass.const_expr(
                qk_stage_idx < cfg.latent_dim // cfg.head_dim_per_stage_kv
            ):
                dim_offset = Int32(qk_stage_idx * cfg.head_dim_per_stage_kv)
                tma_desc = self.tma_desc_c_latent
            else:
                dim_offset = Int32(0)
                tma_desc = self.tma_desc_c_rope

        if cutlass.const_expr(tma_desc is None):
            return
        local_tile_idx = local_kv_tile_idx(
            cfg, stage_info, inst_id, is_v, section=section
        )
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            cfg,
            decode_gen_task_cache(stage_info),
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        tile_idx = global_kv_tile_idx(cfg, local_tile_idx, seq_len_kv, cta_idx_kv)
        stage_base = self._stage_base(stage_info)
        inner_width = min(tma_inner_dim_elems(cfg), active_width)
        uses_compact_fp8_rope_stage = (
            (cfg.load_num_warps == 1 or cfg.is_dynamic_token_sparse)
            and cfg.is_fp8_qkv()
            and not is_v
            and cfg.rope_dim == 64
            and active_width == cfg.rope_dim
            and cfg.head_dim_per_stage_kv == 128
        )

        if cutlass.const_expr(cfg.is_dynamic_token_sparse):
            is_rope_stage = (not is_v) and (
                qk_stage_idx >= cfg.latent_dim // cfg.head_dim_per_stage_kv
            )
            self._producer_gather_token_rows(
                stage_info,
                stage_base,
                dim_offset=Int32(cfg.latent_dim) if is_rope_stage else dim_offset,
                active_width=active_width,
                is_rope_stage=is_rope_stage,
                uses_compact_fp8_rope_stage=uses_compact_fp8_rope_stage,
                publish_mask=(not is_v) and qk_stage_idx == 0,
                tile_idx=tile_idx,
            )
            return
        if cutlass.const_expr(cfg.use_paged_kv == 1):
            if cutlass.const_expr(
                self.page_offsets_kv is None and self.page_offsets is None
            ):
                return
            page_ids = None
            if cutlass.const_expr(self.page_offsets_kv is not None):
                page_ids = cached_page_ids
            pages_per_tile = cfg.pages_per_kv_tile
            first_page_elems = inner_width * cfg.num_tokens_per_page
            first_tile_elems = inner_width * cfg.tile_size_kv
            active_tile_elems = active_width * cfg.tile_size_kv
            last_valid_page = cute.math.max(
                (seq_len_kv + Int32(cfg.num_tokens_per_page - 1))
                // Int32(cfg.num_tokens_per_page)
                - Int32(1),
                Int32(0),
            )
            # Keep the small set of independent page TMA issues straight-line.
            for page_frag in cutlass.range(pages_per_tile, unroll_full=True):
                if cutlass.const_expr(self.page_offsets_kv is not None):
                    page_id = Int32(
                        page_ids[page_id_slot * cfg.pages_per_kv_tile + page_frag]
                    )
                else:
                    logical_page_idx = cute.math.min(
                        tile_idx * Int32(pages_per_tile) + Int32(page_frag),
                        last_valid_page,
                    )
                    page_id = Int32(self.page_offsets[logical_page_idx, batch_idx])
                page_base = Int32(page_frag * first_page_elems)
                smem_page_offset = page_base
                if prims.elect_sync():
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        stage_base.data_ptr(smem_page_offset),
                        tma_desc,
                        (dim_offset, Int32(0), page_id),
                        stage_info.barrier,
                    )
                if cutlass.const_expr(active_width > inner_width):
                    second_half_offset = Int32(first_tile_elems) + smem_page_offset
                    if prims.elect_sync():
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            stage_base.data_ptr(second_half_offset),
                            tma_desc,
                            (dim_offset + Int32(inner_width), Int32(0), page_id),
                            stage_info.barrier,
                        )
                if cutlass.const_expr(
                    active_width < cfg.head_dim_per_stage_kv
                    and not uses_compact_fp8_rope_stage
                ):
                    if prims.elect_sync():
                        # The shared K/V TMA pipeline expects one full QK-sized
                        # stage. Duplicate short K/V slices into the unused half
                        # so the expected transaction byte count is satisfied.
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            stage_base.data_ptr(
                                Int32(active_tile_elems) + smem_page_offset
                            ),
                            tma_desc,
                            (dim_offset, Int32(0), page_id),
                            stage_info.barrier,
                        )
            if cutlass.const_expr(uses_compact_fp8_rope_stage):
                if prims.elect_sync():
                    # The FP8 RoPE MMA consumes only the 64-wide s64b payload.
                    # Retire the intentionally unwritten half of the fixed
                    # 128-wide pipeline stage instead of rereading every page.
                    missing_stage_bytes = (
                        cfg.tile_size_kv
                        * (cfg.head_dim_per_stage_kv - active_width)
                        * cfg.qkv_dtype_bytes
                    )
                    prims.mbarrier_complete_tx(
                        stage_info.barrier, Int32(missing_stage_bytes)
                    )
        else:
            tile_offset = tile_idx * Int32(cfg.tile_size_kv)
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cta_global(
                    stage_base,
                    tma_desc,
                    (
                        dim_offset,
                        tile_offset,
                        head_idx_for_stage(self.head_idx, cfg, stage_info),
                        batch_idx,
                    ),
                    stage_info.barrier,
                )
                if cutlass.const_expr(active_width > inner_width):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        stage_base.data_ptr(Int32(inner_width * cfg.tile_size_kv)),
                        tma_desc,
                        (
                            dim_offset + Int32(inner_width),
                            tile_offset,
                            head_idx_for_stage(self.head_idx, cfg, stage_info),
                            batch_idx,
                        ),
                        stage_info.barrier,
                    )
                if cutlass.const_expr(active_width < cfg.head_dim_per_stage_kv):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        stage_base.data_ptr(Int32(active_width * cfg.tile_size_kv)),
                        tma_desc,
                        (
                            dim_offset,
                            tile_offset,
                            head_idx_for_stage(self.head_idx, cfg, stage_info),
                            batch_idx,
                        ),
                        stage_info.barrier,
                    )

    @producer_work
    @cute.jit
    def load_k0(
        self,
        stage_info: StageInfo,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids=None,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load K instance 0 into the acquired K/V SMEM stage."""

        # ProdWork: fill K instance 0 from the page IDs consumed by the load task.
        self._producer_load_kv_stage(
            stage_info,
            0,
            0,
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )

    @producer_work
    @cute.jit
    def load_k1(
        self,
        stage_info: StageInfo,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids=None,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load K instance 1 into the acquired K/V SMEM stage."""

        # ProdWork: fill K instance 1 from the page IDs consumed by the load task.
        self._producer_load_kv_stage(
            stage_info,
            1,
            0,
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )

    @producer_work
    @cute.jit
    def load_v0(
        self,
        stage_info: StageInfo,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids=None,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load deferred V instance 0 into the acquired K/V SMEM stage."""

        # ProdWork: fill V instance 0 from the page IDs consumed by the load task.
        self._producer_load_kv_stage(
            stage_info,
            0,
            1,
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )

    @producer_work
    @cute.jit
    def load_v1(
        self,
        stage_info: StageInfo,
        *,
        stage_idx: cutlass.Constexpr[int],
        section: cutlass.Constexpr[MlaStage],
        cached_page_ids=None,
        page_id_slot: cutlass.Constexpr[int] = 0,
    ):
        """Load deferred V instance 1 into the acquired K/V SMEM stage."""

        # ProdWork: fill V instance 1 from the page IDs consumed by the load task.
        self._producer_load_kv_stage(
            stage_info,
            1,
            1,
            stage_idx=stage_idx,
            section=section,
            cached_page_ids=cached_page_ids,
            page_id_slot=page_id_slot,
        )

    @cute.jit
    def _set_desc(
        self,
        stage_info: StageInfo,
        is_v: int,
        inst_id: int = 0,
        *,
        k_subtile_idx: cutlass.Constexpr[int] = 0,
    ):
        """Build a K or V SMEM descriptor for the current producer stage."""
        # Descriptor offsets mirror _init_smem_state, but are rebuilt against the
        # current pipeline stage because K and delayed-V can be consumed from
        # different stages.
        stride_byte_offset = Int32(TCGEN05_BF16_SWIZZLE_STRIDE_BYTES)
        if cutlass.const_expr(self.cfg.is_fp8_qkv()):
            stride_byte_offset = Int32(
                qkv_major_k_stride_bytes_for(self.cfg, self.cfg.head_dim_per_stage_kv)
            )
        if cutlass.const_expr(is_v):
            v_leading_byte_offset = (
                Int32(0)
                if self.cfg.head_dim_per_stage_v == TCGEN05_BF16_K_BLOCK_WIDTH
                else Int32(TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES)
            )
            if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                v_leading_byte_offset = Int32(0)
            desc = cprims.Tcgen05SmemDesc.build(
                self._stage_base(stage_info),
                leading_byte_offset=v_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )
            return desc
        else:
            k_leading_byte_offset = Int32(TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES)
            if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                k_leading_byte_offset = Int32(self.cfg.kv_smem_tile_bytes)
                qk_stage_idx = k_subtile_idx
                if cutlass.const_expr(
                    self.cfg.rope_dim == 64
                    and not self.cfg.is_dynamic_token_sparse_nvfp4
                    and qk_stage_idx
                    == self.cfg.latent_dim // self.cfg.head_dim_per_stage_kv
                ):
                    k_leading_byte_offset = Int32(
                        self.cfg.tile_size_kv
                        * self.cfg.rope_dim
                        * self.cfg.qkv_dtype_bytes
                    )
                    return cprims.Tcgen05SmemDesc.build(
                        self._stage_base(stage_info),
                        leading_byte_offset=k_leading_byte_offset,
                        stride_byte_offset=Int32(
                            qkv_major_k_stride_bytes_for(self.cfg, self.cfg.rope_dim)
                        ),
                        layout=qkv_smem_swizzle_for_head_dim(
                            self.cfg, self.cfg.rope_dim
                        ),
                    )
            desc = cprims.Tcgen05SmemDesc.build(
                self._stage_base(stage_info),
                leading_byte_offset=k_leading_byte_offset,
                stride_byte_offset=stride_byte_offset,
                layout=qkv_smem_swizzle(self.cfg),
            )
            return desc

    @consumer_work(returns=("kv_desc",))
    @cute.jit
    def k_desc_0(self, stage_info: StageInfo, *, k_subtile_idx: cutlass.Constexpr[int]):
        """Return the K descriptor for consumer instance 0."""
        # MMA has waited on the K stage; return a descriptor for qk_mma().
        return self._set_desc(stage_info, 0, 0, k_subtile_idx=k_subtile_idx)

    @consumer_work(returns=("kv_desc",))
    @cute.jit
    def k_desc_1(self, stage_info: StageInfo, *, k_subtile_idx: cutlass.Constexpr[int]):
        """Return the K descriptor for consumer instance 1."""
        # Same K SMEM payload, second interleaved QK instance.
        return self._set_desc(stage_info, 0, 1, k_subtile_idx=k_subtile_idx)

    @consumer_work(returns=("v_desc_0",))
    @cute.jit
    def v_desc_0(self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]):
        """Return the V descriptor for consumer instance 0."""
        # PV MMA consumes this descriptor after V instance 0 is ready.
        del v_subtile_idx
        return self._set_desc(stage_info, 1, 0)

    @consumer_work(returns=("v_desc_1",))
    @cute.jit
    def v_desc_1(self, stage_info: StageInfo, *, v_subtile_idx: cutlass.Constexpr[int]):
        """Return the V descriptor for consumer instance 1."""
        # PV MMA consumes this descriptor after V instance 1 is ready.
        del v_subtile_idx
        return self._set_desc(stage_info, 1, 1)


# =====================================================================
# SmemKResource — K-only SMEM view helper
# =====================================================================


@dataclass(kw_only=True)
class SmemKResource(MlaResource):
    """K/V SMEM staging buffer and descriptor producer for QK/PV MMA."""

    inst_id: cutlass.Constexpr[int] = 0
    is_v: cutlass.Constexpr[int] = 0


# =====================================================================
# SmemVResource — V-only SMEM view helper
# =====================================================================


@dataclass(kw_only=True)
class SmemVResource(MlaResource):
    """V-only view used by validation and resource graph naming."""

    inst_id: cutlass.Constexpr[int] = 0
    is_v: cutlass.Constexpr[int] = 1


class SmemRawKvResource(SmemKvResource):
    """Raw NVFP4 K ring: ``kv_raw_stages`` stages of one whole 128-token K tile
    ([lat0 16 KB | lat1 16 KB | rope 8 KB]) gathered by TMA, followed by
    ``nvfp4_scale_slots`` per-tile scale slots (token-major 40-byte rows filled
    by ``cp.async``), the loader-owned validity ring and the index scratch.

    Producer: the Gather4 loader warps (one stage per K tile).  Consumers: the
    MMA reads the K tile natively (TMA -> UMMA) after observing the full barrier
    directly (``observe_k_tile_full``); the dequant warps (the registered
    consumer) convert the resident latent slices into the E4M3 V ring.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        SmemKvResource._task_local_specs
        + (
            (
                "v3_raw_var",
                cutlass.Array,
                None,
                "Register-staged raw E2M1 words of the last V slice.",
                "v3_raw",
            ),
            (
                "v3_scale_var",
                cutlass.Array,
                None,
                "Register-staged block-scale words of the last V slice.",
                "v3_scale",
            ),
        )
    )

    # Loader-owned indices: after the scale slots the allocation holds the
    # per-tile validity ring (``SPARSE_MASK_SLOTS`` x 32 quad nibbles, one byte
    # per quad) and a lane-private index scratch (``load_num_warps`` x 32 lanes
    # x ``LDIDX_TILE_SLOTS`` tile slots x 4 rows; the rows are prefetched two
    # tiles ahead, so rows t .. t + 3 are live).
    LDIDX_MASK_SLOT_BYTES = 32
    LDIDX_TILE_SLOTS = 4

    # K tiles per work tile (dequant co-issue guard).
    num_local_tiles: cutlass.Constexpr[int] = 0

    @property
    def _ring_and_scale_bytes(self) -> int:
        cfg = self.cfg
        return (
            cfg.kv_raw_stages * cfg.kv_raw_stage_bytes
            + cfg.nvfp4_scale_slots * cfg.nvfp4_scale_slot_bytes
        )

    @property
    def _ldidx_mask_bytes(self) -> int:
        return SPARSE_MASK_SLOTS * self.LDIDX_MASK_SLOT_BYTES

    @property
    def _ldidx_quads_per_warp(self) -> int:
        n = self.cfg.load_num_warps
        return (32 + n - 1) // n

    @property
    def _ldidx_scratch_bytes(self) -> int:
        # Only the lanes that own a quad keep index slots (inactive lanes read
        # an aliased slot and discard the value).
        return (
            self.cfg.load_num_warps
            * self._ldidx_quads_per_warp
            * self.LDIDX_TILE_SLOTS
            * 16
        )

    @property
    def total_bytes(self) -> int:
        return (
            self._ring_and_scale_bytes
            + self._ldidx_mask_bytes
            + self._ldidx_scratch_bytes
        )

    @property
    def total_words(self) -> int:
        """Int32 count of the allocation (softmax maps the validity ring through it)."""
        return self.total_bytes // 4

    @property
    def sparse_mask_word_base(self) -> int:
        """Int32 index of the loader-owned validity ring (mirrors SmemPageOffsetsResource)."""
        return self._ring_and_scale_bytes // 4

    @property
    def _ldidx_scratch_word_base(self) -> int:
        return (self._ring_and_scale_bytes + self._ldidx_mask_bytes) // 4

    def get_smem_requirements(self):
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.total_bytes,
                alignment=1024,
            )
        return [self._alloc]

    @cute.jit
    def _init_smem_state(self, stage_info: StageInfo) -> None:
        context = stage_info.context
        self._smem_kv = smem_array(
            context, self._alloc, cutlass.Uint8, self.total_bytes
        )
        self._smem_words = smem_array(context, self._alloc, Int32, self.total_words)

    @cute.jit
    def _stage_base(self, stage_info: StageInfo):
        return self._smem_kv.subview(
            stage_info.stage_idx * Int32(self.cfg.kv_raw_stage_bytes)
        )

    @cute.jit
    def smem_byte_address(self, byte_offset) -> Int32:
        """Return the 32-bit shared address of ``byte_offset`` inside this allocation."""
        return self._smem_kv.data_ptr(Int32(byte_offset)).toint(cutlass.Int32)

    def scale_slot_byte_offset(self, slot) -> Int32:
        cfg = self.cfg
        return Int32(cfg.kv_raw_stages * cfg.kv_raw_stage_bytes) + Int32(slot) * Int32(
            cfg.nvfp4_scale_slot_bytes
        )

    @consumer_work(returns=("v3_raw", "v3_scale"))
    @cute.jit
    def load_v_slice_regs(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[MlaStage],
        inst: cutlass.Constexpr[int] = 0,
    ):
        """Load the last V slice's raw and scale words into registers.

        Issued after the first three V slices were converted and before the raw
        stage is released; ``SmemKvResource.dequant_v_native_regs`` converts
        them once PV of this tile frees a V-ring stage.
        """
        tile = local_kv_tile_idx(self.cfg, stage_info, inst, 0, section=section)
        return self._dequant_stage(
            stage_info,
            0,
            1,
            stage_idx=self.cfg.v_head_dim_stages - 1,
            section=section,
            native_tile_idx=tile,
            preload_only=True,
            raw_override=self,
        )

    @cute.jit
    def observe_k_tile_full(self, local_tile_idx) -> None:
        """Spin until the raw stage of ``local_tile_idx`` is full (non-consumer observer).

        The MMA warp is not a registered consumer of the raw ring: the dequant
        warps are, and they release a stage only after BMM1 of that tile has
        completed (``Bmm1DoneResource``), so the stage cannot be refilled before
        the MMA has seen it and the parity below is unambiguous.
        """
        num_stages = self.cfg.kv_raw_stages
        stage = local_tile_idx % Int32(num_stages)
        parity = (local_tile_idx // Int32(num_stages)) & Int32(1)
        bar = self.full_barrier_smem_address(stage)
        while mbarrier_try_wait_parity_u32(bar, parity) == Int32(0):
            pass

    # Loader-owned indices: producer works without the page-id token (the
    # schedule builder routes every non-constexpr work argument as a token).
    @producer_work
    @cute.jit
    def load_k_tile_owned(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[MlaStage],
        inst: cutlass.Constexpr[int] = 0,
    ):
        """Gather one whole K tile ([lat0 | lat1 | rope]) into one raw stage.

        With two instances per step the per-tile index / scale prefetch cadence
        is unchanged: only the very first tile of the work tile (Head, instance
        0) runs the full prefetch; every later tile (Head instance 1 included)
        runs the Loop path and is followed by ``prefetch_k_tile_ahead``.
        """
        cfg = self.cfg
        local_tile_idx = local_kv_tile_idx(cfg, stage_info, inst, 0, section=section)
        # Trace-time choice (conditional expression: the DSL frontend would try
        # to stage a Python enum reassigned inside an ``if``).
        prefetch_section = (
            MlaStage.Loop if (section == MlaStage.Head and inst != 0) else section
        )
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            cfg,
            decode_gen_task_cache(stage_info),
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        self._producer_gather_owned_tile(
            stage_info,
            self._stage_base(stage_info),
            local_tile_idx=local_tile_idx,
            seq_len_kv=seq_len_kv,
            batch_idx=batch_idx,
            section=prefetch_section,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prefetch_k_tile_ahead(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[MlaStage],
        inst: cutlass.Constexpr[int] = 0,
    ):
        """After a stage commit (every tile but the first): index rows of tile + 3
        and scale rows of tile + 2 (see ``_producer_gather_owned_tile``)."""
        cfg = self.cfg
        local_tile_idx = local_kv_tile_idx(cfg, stage_info, inst, 0, section=section)
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            cfg,
            decode_gen_task_cache(stage_info),
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        self._producer_gather_owned_tile(
            stage_info,
            None,
            local_tile_idx=local_tile_idx,
            seq_len_kv=seq_len_kv,
            batch_idx=batch_idx,
            section=section,
            mode="prefetch",
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def issue_gathers_ahead(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[MlaStage],
        inst: cutlass.Constexpr[int] = 0,
    ):
        """Dequant warps issue the lat1 and rope Gather4 of tile t + 2 right after
        releasing tile t's raw stage (the loader itself covers tiles 0 and 1).

        The stage is free (this task is its only consumer and BMM1(t) completed),
        the row indices of t + 2 are in the loader scratch (prefetched two steps
        earlier and complete before raw stage t was released), and the stage's
        full barrier is in the phase of t + 2 (t's phase completed before this
        step).  The bytes land on that barrier ahead of the loader's
        arrive.expect_tx, which the mbarrier tx-count allows; the loader still
        issues lat0, the mask, the proxy fence and the surplus completion.
        64 gathers over 8 warps: lane l < 8 of warp w issues quad w * 4 + (l & 3),
        lat1 for l < 4 and rope for 4 <= l < 8.
        """
        cfg = self.cfg
        tile = local_kv_tile_idx(cfg, stage_info, inst, 0, section=section)
        target = tile + Int32(2)
        # The loader never loads tiles past this work tile's runtime count
        # (ragged lengths): gathers for them would land in a stage nobody
        # waits for (and after the CTA exits), so guard on the runtime count.
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            cfg,
            decode_gen_task_cache(stage_info),
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        runtime_tiles = runtime_local_kv_tiles(cfg, seq_len_kv)
        words = self._smem_words
        thread_idx = cute.arch.thread_idx()[0] - Int32(cfg.dequant_warp_idx * 32)
        if (target < Int32(self.num_local_tiles)) & (target < runtime_tiles):
            stage = target % Int32(cfg.kv_raw_stages)
            stage_byte = self.smem_byte_address(stage * Int32(cfg.kv_raw_stage_bytes))
            bar_ptr = cutlass.inttoptr(
                self.full_barrier_smem_address(stage), 3, cutlass.Int64
            )
            warp_in_task = thread_idx >> Int32(5)
            lane_idx = thread_idx & Int32(31)
            # Spread the quads over all dequant warps (the Gather4 issue is
            # serialized per lane inside a warp, ~45 ns each): every warp
            # issues 4 quads x 2 boxes on lanes 0-7 (measured 360 ns per tile
            # with the first four warps carrying 8 issues each).
            qpw = max(1, (cfg.tile_size_kv // 4) // cfg.dequant_num_warps)
            quad = warp_in_task * Int32(qpw) + (lane_idx % Int32(qpw))
            kind = (lane_idx // Int32(qpw)) & Int32(1)
            active = (lane_idx < Int32(2 * qpw)) & (quad < Int32(cfg.tile_size_kv // 4))
            word_base = (
                Int32(self._ldidx_scratch_word_base)
                + quad * Int32(self.LDIDX_TILE_SLOTS * 4)
                + (target & Int32(self.LDIDX_TILE_SLOTS - 1)) * Int32(4)
            )
            row0 = Int32(words[word_base])
            row1 = Int32(words[word_base + Int32(1)])
            row2 = Int32(words[word_base + Int32(2)])
            row3 = Int32(words[word_base + Int32(3)])
            seg_bytes = cfg.tile_size_kv * 128
            desc128 = self.raw_gather4_descs.iterator + Int64(384)
            desc64 = self.raw_gather4_descs.iterator + Int64(256)
            if active and kind == Int32(0):
                issue_gather4_tma_cta(
                    desc128,
                    cutlass.inttoptr(
                        stage_byte + Int32(seg_bytes) + quad * Int32(4 * 128),
                        3,
                        cutlass.Int32,
                    ),
                    bar_ptr,
                    Int32(128),
                    (row0, row1, row2, row3),
                )
            if active and kind == Int32(1):
                issue_gather4_tma_cta(
                    desc64,
                    cutlass.inttoptr(
                        stage_byte + Int32(2 * seg_bytes) + quad * Int32(4 * 64),
                        3,
                        cutlass.Int32,
                    ),
                    bar_ptr,
                    Int32(cfg.latent_dim // 2),
                    (row0, row1, row2, row3),
                )

    @cute.jit
    def _producer_gather_owned_tile(
        self,
        stage_info: StageInfo,
        stage_base,
        *,
        local_tile_idx,
        seq_len_kv,
        batch_idx,
        section: cutlass.Constexpr[MlaStage],
        mode: cutlass.Constexpr[str] = "gather",
    ):
        """Loader-owned indices, one K tile per raw stage.

        Per tile: lane ``l`` of loader warp ``w`` owns quad ``w*qpw + l``; it
        prefetches the row indices of tile + 3 (slot ``tile & 3``), stages the
        scale rows of tile + 2, publishes its validity nibble and issues the
        latent-0 Gather4 (128 B).  The lat1 / rope boxes are issued here for
        tiles 0 and 1 only; from tile 2 on the dequant warps issue them
        (``issue_gathers_ahead``).
        """
        cfg = self.cfg
        num_load_warps = cfg.load_num_warps
        lane_idx = lane_idx_from_thread(cute.arch.thread_idx()[0])
        warp_in_task = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.load_warp_idx
        )
        num_quads = cfg.tile_size_kv // 4
        qpw = (num_quads + num_load_warps - 1) // num_load_warps
        quad = (
            warp_in_task * Int32(qpw)
            + lane_idx
            + (Int32(num_quads) if lane_idx >= Int32(qpw) else Int32(0))
        )
        lane_active = quad < Int32(num_quads)
        words = self._smem_words
        last_valid = cute.math.max(seq_len_kv - Int32(1), Int32(0))
        page_offsets_batch = self.page_offsets[None, batch_idx]
        page_offsets_flat = cute.flat_divide(page_offsets_batch, (1,))
        scratch_lane_base = Int32(self._ldidx_scratch_word_base) + (
            warp_in_task * Int32(qpw) + cute.math.min(lane_idx, Int32(qpw - 1))
        ) * Int32(self.LDIDX_TILE_SLOTS * 4)

        def slot_word(tile):
            return scratch_lane_base + (
                tile & Int32(self.LDIDX_TILE_SLOTS - 1)
            ) * Int32(4)

        def prefetch_rows(tile):
            base = tile * Int32(cfg.tile_size_kv) + quad * Int32(4)
            for j in cutlass.range_constexpr(4):
                logical_idx = cute.math.min(base + Int32(j), last_valid)
                gmem_ptr = page_offsets_flat[None, logical_idx].iterator.llvm_ptr
                smem_ptr = cutlass.inttoptr(
                    words.data_ptr(slot_word(tile) + Int32(j)).toint(cutlass.Int32),
                    3,
                    cutlass.Int32,
                )
                if lane_active:
                    prims.cp_async_shared_global(
                        smem_ptr, gmem_ptr, PAGE_OFFSET_BYTES, CP_ASYNC_CACHE_CA
                    )
            cp_async_commit_group()

        scale_row_bytes = cfg.nvfp4_scale_row_bytes

        def stage_scales(tile):
            # The 40-byte scale row of each of the lane's four tokens lands
            # token-major in the slot (40-byte pitch, five 8-byte copies: odd
            # rows are only 8-byte aligned in the pool).  The dequant reads V
            # scales as LDS.64 and the dequant warps write SFB with tcgen05.st,
            # so no block order is needed.  Invalid rows zero-fill.
            slot_base = self.scale_slot_byte_offset(tile % Int32(cfg.nvfp4_scale_slots))
            scale_pool_base = self.kv_scale_pool.iterator.toint()
            for j in cutlass.range_constexpr(4):
                row = Int32(words[slot_word(tile) + Int32(j)])
                token = quad * Int32(4) + Int32(j)
                dst = self.smem_byte_address(slot_base + token * Int32(scale_row_bytes))
                src = scale_pool_base + Int64(row) * Int64(scale_row_bytes)
                n8 = Int32(8) if row >= Int32(0) else Int32(0)
                if lane_active:
                    for k in cutlass.range_constexpr(scale_row_bytes // 8):
                        cp_async_8b_zfill(dst + Int32(8 * k), src + Int64(8 * k), n8)
            cp_async_commit_group()

        tile = local_tile_idx
        if cutlass.const_expr(mode == "prefetch"):
            # Tail of a Loop step, after the stage commit: the scale rows of
            # tile + 2 (slot of tile - 2, whose readers finished before this
            # step's stage was acquired), then the row indices of tile + 3.
            # Issued here so the ~44 scattered cp.async per lane do not sit
            # between the stage acquire and the Gather4 issue.  Both groups
            # issued at the previous tail (scales(tile + 1), rows(tile + 2)) are
            # a whole tile old; rows(tile + 2) must have landed before its scale
            # rows are staged.  Committing the scales before the rows makes the
            # scale group the older of the two groups in flight, so the
            # ``scales_landed`` publication at the next loop top completes it
            # with ``cp.async.wait_group 1`` and leaves only the row prefetch.
            cp_async_wait_group(0)
            stage_scales(tile + Int32(2))
            prefetch_rows(tile + Int32(3))
            return
        if cutlass.const_expr(section == MlaStage.Head):
            prefetch_rows(tile)
            prefetch_rows(tile + Int32(1))
            prefetch_rows(tile + Int32(2))
            prefetch_rows(tile + Int32(3))
            # Only rows(tile) must have landed to issue the first gather; the
            # scale rows are staged after it (below).
            cp_async_wait_group(3)
        else:
            # The ``scales_landed`` publication at the top of this loop step
            # already completed every group but the previous tail's row
            # prefetch (rows(tile + 2)); this step needs only rows(tile) and the
            # scale rows of tile, so let that newest group stay in flight
            # instead of eating its HBM latency here (wait_group 0 measured
            # +5 us at B=256).
            cp_async_wait_group(1)
        # The landed scale rows of this tile are handed to their readers through
        # the stage barrier once it flips: order the generic-proxy cp.async
        # writes ahead of the release below.
        fence_proxy_async_shared()
        row0 = Int32(words[slot_word(tile)])
        row1 = Int32(words[slot_word(tile) + Int32(1)])
        row2 = Int32(words[slot_word(tile) + Int32(2)])
        row3 = Int32(words[slot_word(tile) + Int32(3)])
        nibble = (
            (Int32(1) if row0 >= Int32(0) else Int32(0))
            | ((Int32(1) if row1 >= Int32(0) else Int32(0)) << Int32(1))
            | ((Int32(1) if row2 >= Int32(0) else Int32(0)) << Int32(2))
            | ((Int32(1) if row3 >= Int32(0) else Int32(0)) << Int32(3))
        )
        # Tokens at or beyond the KV length (the clamped duplicates of the last
        # valid index) are invalid too, so the softmax needs no per-score tail
        # test.
        tok0 = tile * Int32(cfg.tile_size_kv) + quad * Int32(4)
        nibble = nibble & (
            (Int32(1) if tok0 < seq_len_kv else Int32(0))
            | ((Int32(1) if tok0 + Int32(1) < seq_len_kv else Int32(0)) << Int32(1))
            | ((Int32(1) if tok0 + Int32(2) < seq_len_kv else Int32(0)) << Int32(2))
            | ((Int32(1) if tok0 + Int32(3) < seq_len_kv else Int32(0)) << Int32(3))
        )
        mask_byte = (
            Int32(self.sparse_mask_word_base * 4)
            + (tile % Int32(SPARSE_MASK_SLOTS)) * Int32(self.LDIDX_MASK_SLOT_BYTES)
            + quad
        )
        if lane_active:
            st_shared_u8(self.smem_byte_address(mask_byte), nibble)
        desc128 = self.raw_gather4_descs.iterator + Int64(384)
        desc64 = self.raw_gather4_descs.iterator + Int64(256)
        seg_bytes = cfg.tile_size_kv * 128
        if lane_active:
            issue_gather4_tma_cta(
                desc128,
                stage_base.data_ptr(quad * Int32(4 * 128)),
                stage_info.barrier,
                Int32(0),
                (row0, row1, row2, row3),
            )
            # The dequant warps issue lat1 / rope of tile t + 2 right after they
            # release tile t's stage (issue_gathers_ahead), so the loader covers
            # them only for tiles 0 and 1.
            if tile < Int32(2):
                issue_gather4_tma_cta(
                    desc128,
                    stage_base.data_ptr(Int32(seg_bytes) + quad * Int32(4 * 128)),
                    stage_info.barrier,
                    Int32(128),
                    (row0, row1, row2, row3),
                )
                issue_gather4_tma_cta(
                    desc64,
                    stage_base.data_ptr(Int32(2 * seg_bytes) + quad * Int32(4 * 64)),
                    stage_info.barrier,
                    Int32(cfg.latent_dim // 2),
                    (row0, row1, row2, row3),
                )
        declared = cfg.nvfp4_raw_stage_tx_bytes
        gathered = cfg.kv_raw_stage_bytes
        surplus = declared - gathered
        warp_sync_all()
        if lane_idx == Int32(0):
            share = Int32(surplus // num_load_warps) + (
                Int32(1) if warp_in_task < Int32(surplus % num_load_warps) else Int32(0)
            )
            prims.fence_acq_rel(prims.MemScope.CTA)
            prims.mbarrier_complete_tx(stage_info.barrier, share)
        if cutlass.const_expr(section == MlaStage.Head):
            # Gather mode (the prefetch mode returned above): with the first
            # gather in flight, complete rows(1..3) and stage the scale rows of
            # tiles 0 / 1 / 2 (the HEAD publications of sf_landed follow this
            # call and only fence).
            cp_async_wait_group(0)
            stage_scales(tile)
            stage_scales(tile + Int32(1))
            stage_scales(tile + Int32(2))
            cp_async_wait_group(0)
