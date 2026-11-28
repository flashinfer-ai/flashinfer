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

"""SMEM K/V ring, its paged-KV page offsets and their TMA loads."""

import cutlass.cute as cute
import cutlass
from cutlass import Int16, Int32
from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field, replace
from ..config import FmhaConfig
from ..device.windows import resolve_kv_tile_idx_context
from ..device.common import (
    MX_SCALE_TILE_BYTES,
    SmemDescOffsets,
    mx_scale_slot,
    qk_smem_desc_offsets,
    qk_smem_layout,
    smem_layout_for_inner_bytes,
)
from .....device.placeholders import placeholder_smem_array
from .....device.stage import WorkStage
from typing import Any, Optional


# ---------------------------------------------------------------------------
# SmemKVResource -- SMEM K/V tile buffer with TMA pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SmemKVResource(MemoryResource):
    """SMEM buffer for K and V tiles with a capacity-derived TmaUmma pipeline.

    K and V tiles alternate in the pipeline stages: K0, V0, K1, V1, ...
    Producer: LoadTask (TMA loads K/V tiles).
    Consumer: MmaTask (builds SMEM descriptors for QK and PV MMAs).

    ``role="kv"`` shares one buffer and pipeline, which requires equal K and V
    element widths. Mixed QK/PV dtypes instead build one ``role="k"`` and one
    ``role="v"`` instance, each sized from its own dtype.
    """

    # Static configuration
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    desc_k_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_k_stage_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_v_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    role: Constexpr[str] = field(init=False, default="kv")

    # Linked resources
    page_offsets_kv: Optional["SmemPageOffsetsKvResource"] = field(
        init=False, default=None
    )
    page_offsets_v: Optional["SmemPageOffsetsKvResource"] = field(
        init=False, default=None
    )

    # GMEM pointers and tensors
    block_tables: cute.Pointer | None = field(init=False, default=None)
    k_scales: cute.Tensor | None = field(init=False, default=None)
    tma_k_desc: cutlass.Pointer | None = field(init=False, default=None)
    tma_v_desc: cutlass.Pointer | None = field(init=False, default=None)

    # Allocations
    alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    # Per-task state
    sK_array: cutlass.Array = field(init=False, default=None)

    def __init__(
        self,
        tma_k_desc: cutlass.Pointer | None,
        tma_v_desc: cutlass.Pointer | None,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        k_scales: cute.Tensor | None = None,
        page_offsets_kv: Optional["SmemPageOffsetsKvResource"] = None,
        page_offsets_v: Optional["SmemPageOffsetsKvResource"] = None,
        block_tables: cute.Pointer | None = None,
        role: str = "kv",
        **kwargs: Any,
    ) -> None:
        """Bind K/V TMA descriptors and reserve this role's SMEM staging."""
        if role not in ("kv", "k", "v"):
            raise ValueError(f"unsupported SmemKVResource role: {role}")
        if role == "kv" and cfg.k_dtype.width != cfg.v_dtype.width:
            raise ValueError(
                "a shared K/V buffer requires K and V to have the same element "
                f"width; got {cfg.k_dtype.width} and {cfg.v_dtype.width}"
            )
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.role = role
        self.tma_k_desc = tma_k_desc
        self.tma_v_desc = tma_v_desc
        self.k_scales = k_scales
        self.page_offsets_kv = page_offsets_kv
        self.page_offsets_v = (
            page_offsets_v if page_offsets_v is not None else page_offsets_kv
        )
        self.block_tables = block_tables
        self.cfg = cfg
        # Every role stages pipeline_config.num_stages tiles; for "kv" that is
        # cfg.kv_stage, which is cfg.sK_shape[0].
        total_elements = pipeline_config.num_stages * cfg.sK_shape[1]
        size_bytes = total_elements * self.buffer_dtype(cfg, role).width // 8
        if cfg.use_mxfp8_qk and role == "k":
            size_bytes += pipeline_config.num_stages * MX_SCALE_TILE_BYTES
        self.alloc = SmemAllocation(
            f"smem_{role}", size_bytes, alignment=cfg.buffer_align_bytes
        )
        self.sK_array = placeholder_smem_array(self.buffer_dtype(cfg, role))
        self.desc_k_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the current K tile.",
        )
        self.desc_k_stage_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor for the later K slice retained by both Q tiles.",
        )
        self.desc_v_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the current V tile.",
        )

    @property
    def holds_k(self) -> bool:
        """Return whether this buffer stages K tiles."""
        return self.role in ("kv", "k")

    @property
    def holds_v(self) -> bool:
        """Return whether this buffer stages V tiles."""
        return self.role in ("kv", "v")

    @property
    def loop_offset_sensitive(self) -> bool:
        """Return true because K/V loads index the current loop tile."""
        # producer_work uses loop_offset to compute seq_coord_kv.
        return True

    @staticmethod
    def buffer_dtype(cfg: FmhaConfig, role: str) -> type:
        """Return the element type staged by a buffer with this role."""
        return cfg.v_dtype if role == "v" else cfg.k_dtype

    def build_v_descriptor(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Build the current stage's V descriptor after any required clear."""
        smem_stage_elements = self.cfg.tma_copy_kv_elements
        sK_curr = self.sK_array.subview(stage_info.stage_idx * smem_stage_elements)
        leading_byte_offset, stride_byte_offset = pv_smem_desc_offsets(self.cfg)
        return prims.Tcgen05SmemDesc.build(
            sK_curr,
            leading_byte_offset=leading_byte_offset,
            stride_byte_offset=stride_byte_offset,
            layout=pv_smem_layout(self.cfg),
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the SMEM allocation required for staged K/V tiles."""
        return [self.alloc]

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        self.init_smem_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self.init_smem_state(stage_info)

    @cute.jit
    def init_smem_state(self, stage_info: StageInfo) -> None:
        """Materialize K/V SMEM storage and descriptor dataflow slots."""
        smem_base = stage_info.context.smem_base
        total_elements = self.pipeline_config.num_stages * self.cfg.sK_shape[1]
        self.sK_array = cutlass.Array(
            smem_base.data_ptr() + self.alloc.offset,
            dtype=self.buffer_dtype(self.cfg, self.role),
            shape=(total_elements,),
            addrspace=3,
        )

    @consumer_work(returns=desc_k_base)
    @cute.jit
    def k_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Build K SMEM descriptor (K-major layout for QK MMA) -> desc_k_base."""
        smem_stage_elements = self.cfg.tma_copy_kv_elements
        sK_curr = self.sK_array.subview(stage_info.stage_idx * smem_stage_elements)
        leading_byte_offset, stride_byte_offset = qk_smem_desc_offsets(self.cfg)
        desc_k_base = prims.Tcgen05SmemDesc.build(
            sK_curr,
            leading_byte_offset=leading_byte_offset,
            stride_byte_offset=stride_byte_offset,
            layout=qk_smem_layout(self.cfg),
        )
        return desc_k_base

    @producer_work
    @cute.jit
    def k_load(
        self,
        stage_info: StageInfo,
        *,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        tile_offset: cutlass.Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_lb: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """TMA load K tile from GMEM to SMEM."""
        self.tma_load_impl(
            stage_info,
            self.tma_k_desc,
            tile_offset=tile_offset,
            head_dim_stage_idx=head_dim_stage_idx,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_lb=kv_page_idx_lb,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @producer_work
    @cute.jit
    def k_load_stage(
        self,
        stage_info: StageInfo,
        *,
        stage_id: Constexpr[int],
        tile_offset: Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_lb: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """TMA load one K head-dim stage for split D scheduling."""
        self.tma_load_impl(
            stage_info,
            self.tma_k_desc,
            False,
            tile_offset=tile_offset,
            head_dim_stage_idx=stage_id,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_lb=kv_page_idx_lb,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @consumer_work(returns=desc_k_stage_base)
    @cute.jit
    def k_stage_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Keep the later K descriptor live alongside the first K and V."""
        smem_stage_elements = self.cfg.tma_copy_kv_elements
        sK_curr = self.sK_array.subview(stage_info.stage_idx * smem_stage_elements)
        leading_byte_offset, stride_byte_offset = qk_smem_desc_offsets(self.cfg)
        return prims.Tcgen05SmemDesc.build(
            sK_curr,
            leading_byte_offset=leading_byte_offset,
            stride_byte_offset=stride_byte_offset,
            layout=qk_smem_layout(self.cfg),
        )

    @cute.jit
    def tma_load_impl(
        self,
        stage_info: StageInfo,
        tma_desc: cutlass.Pointer | None,
        is_v: cutlass.Constexpr[bool] = False,
        tile_offset: cutlass.Constexpr[int] = 0,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        cached_page_ids: cutlass.Array | None = None,
        *,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_lb: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """Issue TMA bulk-copy for one K or V tile."""
        seq_offset = (
            kv_tile_start + stage_info.loop_offset + tile_offset
        ) * self.cfg.kv_tile_n
        smem_stage_elements = self.cfg.tma_copy_kv_elements
        sK_curr = self.sK_array.subview(stage_info.stage_idx * smem_stage_elements)
        # V carries its own copy granularity because v_dtype may be narrower.
        # The two sets coincide when the widths match.
        if cutlass.const_expr(is_v):
            d_granu_inner = self.cfg.tma_copy_v_granu_inner
            d_iter_elems = self.cfg.tma_copy_v_granu_elems
            stage_iters = self.cfg.tma_copy_v_stage_iters
        else:
            d_granu_inner = self.cfg.tma_copy_kv_granu_inner
            d_iter_elems = self.cfg.tma_copy_kv_granu_elems
            stage_iters = self.cfg.tma_copy_kv_stage_iters

        if cutlass.const_expr(self.cfg.use_paged_kv):
            # Paged-KV path: read pre-staged page IDs and issue one TMA per
            # (page fragment, d fragment). The descriptor is shaped
            # (d_inner, num_tokens_per_page, h_kv, total_pages); coords are
            # (d_off, 0, kv_head_coord, page_id). SMEM layout per stage matches
            # the contiguous path: two d-halves (d_iter_elems each) with page
            # fragments concatenated along the seq axis inside each d-half.
            tile_idx = kv_tile_start + stage_info.loop_offset + tile_offset
            pages_per_tile = self.cfg.kv_tile_n // self.cfg.num_tokens_per_page
            page_d_elems = self.cfg.num_tokens_per_page * d_granu_inner
            if prims.elect_sync():
                # Only the elected TMA-issuing lane consumes page IDs. Loading
                # the vector outside this guard made every lane perform the
                # same SMEM read for every K and V tile.
                page_ids = cached_page_ids
                if cutlass.const_expr(page_ids is None):
                    page_offsets = (
                        self.page_offsets_v
                        if cutlass.const_expr(is_v)
                        else self.page_offsets_kv
                    )
                    if cutlass.const_expr(page_offsets is not None):
                        page_ids = page_offsets.page_ids(tile_idx)
                    else:
                        # The paired K/V schedule consumes K and V together.
                        # Reading its four contiguous page IDs directly avoids
                        # a producer warp spinning on an always-full auxiliary
                        # pipeline and leaves that warp available for CLC.
                        # K and V share the same fixed logical-to-physical page
                        # row. Clamp both to the pages covered by the request's
                        # runtime sequence length so padding IDs are untouched.
                        logical_page_idx = tile_idx * Int32(pages_per_tile)
                        page_ids = cutlass.Array(
                            Int32,
                            pages_per_tile,
                            space=cutlass.AddressSpace.rmem,
                        )
                        for frag in cutlass.range_constexpr(pages_per_tile):
                            clamped_page_idx = cute.math.min(
                                logical_page_idx + Int32(frag), kv_page_idx_ub
                            )
                            page_ids[frag] = Int32(
                                self.block_tables[kv_request_begin + clamped_page_idx]
                            )
                for frag in cutlass.range_constexpr(pages_per_tile):
                    page_id = Int32(page_ids[frag])
                    if cutlass.const_expr(self.cfg.kv_tile_start_window_size_left > 0):
                        # Old pages may be recycled while their table IDs persist.
                        # Zero-fill their K/V rather than multiplying masked NaNs.
                        if (
                            tile_idx * Int32(pages_per_tile) + Int32(frag)
                            < kv_page_idx_lb
                        ):
                            page_id = Int32(-1)
                    for i in cutlass.range_constexpr(stage_iters):
                        d_offset = Int32(
                            head_dim_stage_idx * self.cfg.head_dim_per_stage_kv
                            + i * d_granu_inner
                        )
                        smem_offset = Int32(i * d_iter_elems + frag * page_d_elems)
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            sK_curr.subview(smem_offset),
                            tma_desc,
                            (d_offset, Int32(0), kv_head_coord, page_id),
                            stage_info.barrier,
                        )
            return

        if prims.elect_sync():
            seq_coord_kv = cuseqlen_k + seq_offset
            d_base = Int32(0)
            if cutlass.const_expr(self.cfg.two_cta_umma):
                # This CTA stages its half of the K rows or V columns.
                cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                if cutlass.const_expr(is_v):
                    d_base = cta_rank * Int32(self.cfg.pv_n_per_cta)
                else:
                    seq_coord_kv = seq_coord_kv + cta_rank * Int32(
                        self.cfg.kv_tile_rows_per_cta
                    )
            for i in cutlass.range_constexpr(stage_iters):
                d_offset = (
                    head_dim_stage_idx * self.cfg.head_dim_per_stage_kv
                    + i * d_granu_inner
                )
                kv_coords = (
                    d_offset + d_base,
                    kv_head_coord,
                    seq_coord_kv,
                    batch_coord,
                )
                if cutlass.const_expr(self.cfg.has_varlen):
                    kv_coords = (d_offset + d_base, kv_head_coord, seq_coord_kv)
                if cutlass.const_expr(self.cfg.two_cta_umma):
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sK_curr.subview(i * d_iter_elems),
                        tma_desc,
                        kv_coords,
                        cutlass.Array(
                            stage_info.barrier.data_ptr(), dtype=cutlass.Int64
                        ),
                        [],
                        multicast_mask=Int16(Int32(1) << cta_rank),
                        group=prims.CTAGroup.CTA_2,
                    )
                else:
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sK_curr.subview(i * d_iter_elems),
                        tma_desc,
                        kv_coords,
                        stage_info.barrier,
                    )
            if cutlass.const_expr(self.cfg.use_mxfp8_qk and not is_v):
                # The K scale atom of this tile lands behind the K ring.
                prims.cp_async_bulk_shared_cluster_global(
                    mx_scale_slot(
                        stage_info.context.smem_base,
                        self.alloc.offset
                        + self.pipeline_config.num_stages * self.cfg.tma_copy_kv_bytes
                        + stage_info.stage_idx * MX_SCALE_TILE_BYTES,
                    ),
                    self.k_scales.iterator
                    + cute.crd2idx(
                        (
                            batch_coord,
                            kv_head_coord,
                            # The causal paired domain rounds the K tile count
                            # up to even; the padded iteration's rows are masked,
                            # so it reads the last atom instead of the next head's.
                            cute.math.min(
                                seq_offset // self.cfg.kv_tile_n,
                                Int32(self.k_scales.shape[2] - 1),
                            ),
                            0,
                        ),
                        self.k_scales.layout,
                    ),
                    stage_info.barrier,
                    MX_SCALE_TILE_BYTES,
                )

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc(self, stage_info: StageInfo) -> prims.Tcgen05SmemDesc:
        """Build a V SMEM descriptor that needs no paged-tail clear."""
        return self.build_v_descriptor(stage_info)

    @consumer_work(returns=desc_v_base)
    @cute.jit
    def v_desc_paged(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[WorkStage],
        tile_offset: cutlass.Constexpr[int] = 0,
        seqlen_k: Int32,
        kv_tile_start: Int32,
    ) -> prims.Tcgen05SmemDesc:
        """Clear invalid paged-V rows, then build its SMEM descriptor."""
        self.zero_paged_v_tail(
            stage_info,
            section=section,
            tile_offset=tile_offset,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
        )
        return self.build_v_descriptor(stage_info)

    @producer_work
    @cute.jit
    def v_load(
        self,
        stage_info: StageInfo,
        *,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        tile_offset: cutlass.Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_lb: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """TMA load V tile from GMEM to SMEM."""
        self.tma_load_impl(
            stage_info,
            self.tma_v_desc,
            True,
            tile_offset=tile_offset,
            head_dim_stage_idx=head_dim_stage_idx,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_lb=kv_page_idx_lb,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @producer_work
    @cute.jit
    def v_load_stage(
        self,
        stage_info: StageInfo,
        *,
        stage_id: Constexpr[int],
        previous: Constexpr[bool] = False,
        tile_offset: Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_lb: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """TMA load one current or previous V head-dim stage."""
        self.tma_load_impl(
            stage_info,
            self.tma_v_desc,
            True,
            tile_offset=tile_offset + (-1 if previous else 0),
            head_dim_stage_idx=stage_id,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_lb=kv_page_idx_lb,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @producer_work
    @cute.jit
    def v_load_stage_cached(
        self,
        stage_info: StageInfo,
        *,
        cached_v_page_ids: cutlass.Array,
        stage_id: Constexpr[int],
        tile_offset: Constexpr[int] = 0,
        kv_head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_k: Int32,
        seqlen_k: Int32,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_lb: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """Load one V head-dimension stage using register-cached page IDs."""
        self.tma_load_impl(
            stage_info,
            self.tma_v_desc,
            True,
            tile_offset=tile_offset,
            head_dim_stage_idx=stage_id,
            cached_page_ids=cached_v_page_ids,
            kv_head_coord=kv_head_coord,
            batch_coord=batch_coord,
            cuseqlen_k=cuseqlen_k,
            seqlen_k=seqlen_k,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_lb=kv_page_idx_lb,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @cute.jit
    def zero_paged_v_tail(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[WorkStage],
        tile_offset: cutlass.Constexpr[int],
        seqlen_k: Int32,
        kv_tile_start: Int32,
    ) -> None:
        """Overwrite request-invalid V rows after TMA completion."""
        if cutlass.const_expr(section == WorkStage.Head):
            domain_tile_idx = stage_info.loop_start
        elif cutlass.const_expr(section == WorkStage.Tail):
            domain_tile_idx = stage_info.loop_end
        else:
            domain_tile_idx = stage_info.loop_offset
        logical_v_tile_idx = kv_tile_start + domain_tile_idx + tile_offset
        valid_rows = cute.math.min(
            cute.math.max(
                seqlen_k - logical_v_tile_idx * Int32(self.cfg.kv_tile_n),
                Int32(0),
            ),
            Int32(self.cfg.kv_tile_n),
        )

        if valid_rows < Int32(self.cfg.kv_tile_n):
            # Each paged TMA transaction writes one swizzled
            # (D-fragment, page-token) box. Pages are concatenated within a D
            # iteration, and D iterations are concatenated within the stage.
            # Mirror that exact physical layout: a flat row-major clear would
            # target the wrong bytes under the s128b swizzle.
            d_granu_inner = self.cfg.tma_copy_v_granu_inner
            chunks_per_d_iter = d_granu_inner // 16
            chunks_per_v_row = self.cfg.tma_copy_v_stage_iters * chunks_per_d_iter
            page_d_elems = self.cfg.num_tokens_per_page * d_granu_inner
            d_iter_elems = self.cfg.tma_copy_v_granu_elems
            invalid_chunks = (Int32(self.cfg.kv_tile_n) - valid_rows) * Int32(
                chunks_per_v_row
            )
            zero_vec = cutlass.vector.full(
                [16], self.cfg.v_dtype(0.0), dtype=self.cfg.v_dtype
            )
            sV_curr = self.sK_array.subview(
                stage_info.stage_idx * self.cfg.tma_copy_kv_elements
            )
            lane_idx = cute.arch.lane_idx()
            for tail_chunk in cutlass.range(
                lane_idx,
                invalid_chunks,
                Int32(cute.arch.WARP_SIZE),
                unroll=1,
            ):
                invalid_row = tail_chunk // Int32(chunks_per_v_row)
                d_chunk = tail_chunk - invalid_row * Int32(chunks_per_v_row)
                d_iter = d_chunk // Int32(chunks_per_d_iter)
                d_chunk_in_iter = d_chunk - d_iter * Int32(chunks_per_d_iter)
                logical_row = valid_rows + invalid_row
                page_frag = logical_row // Int32(self.cfg.num_tokens_per_page)
                row_in_page = logical_row - page_frag * Int32(
                    self.cfg.num_tokens_per_page
                )
                smem_offset = (
                    d_iter * Int32(d_iter_elems)
                    + page_frag * Int32(page_d_elems)
                    + row_in_page * Int32(d_granu_inner)
                    + d_chunk_in_iter * Int32(16)
                )
                sV_curr.subview(smem_offset).data_ptr().store_swizzled(
                    zero_vec,
                    alignment=16,
                    swizzle=pv_smem_swizzle(self.cfg),
                )

            # v_desc is called only after this stage's skv.wait(), which makes
            # the TMA writes visible. Converge the one MMA warp after its
            # generic stores, then publish them to the async SMEM proxy before
            # tcgen05 consumes the descriptor.
            cute.arch.sync_warp()
            prims.fence_proxy(
                kind=prims.Proxy.ASYNC_SHARED,
                space=prims.SharedSpace.shared_cta,
            )


# ---------------------------------------------------------------------------
# SmemPageOffsetsKvResource -- paged-KV page-table cache in SMEM
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SmemPageOffsetsKvResource(MemoryResource):
    """Paged-KV logical-to-physical page IDs staged in SMEM (context kernel).

    The staged D256 path uses a dedicated warp to prefetch page-table
    entries for the next K/V tile so the TMA load warp can read SMEM-cached
    offsets. Each pipeline stage holds one topology-derived page-ID window
    from the request's fixed-table row; all 32 lanes co-load it. ``page_ids`` slices
    ``pages_per_tile`` entries for the current tile.

    Differences from decode:
    - Single ``load_k`` / ``load_v`` producer pair (context has no
      ``num_insts_kv > 1`` four-way split).
    - Consumer release labels bind to ``{"k_load", "v_load"}`` (matching
      ``SmemKVResource`` producer names).
    - Driven by the staged D256 ``cfg.empty_warp_id`` (warp 11).
      Paired D128 does not instantiate this resource.
    """

    # Static configuration
    cached_page_ids: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    page_table_is_v: Constexpr[bool] = field(init=False, default=False)

    # GMEM pointers and tensors
    block_tables: cute.Pointer | None = field(init=False, default=None)

    # Allocations
    alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    # Per-task state
    smem_page_offsets: cutlass.Array = field(init=False, default=None)

    def __init__(
        self,
        block_tables: cute.Pointer | None,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        page_table_is_v: bool = False,
        **kwargs: Any,
    ) -> None:
        # ``page_ids`` runs from the downstream K/V resource after this
        # resource's ConsumerWait. Preserve the waited stage so the nested
        # lookup reads the matching page-table data.
        pipeline_config = replace(pipeline_config, advance_on_wait=True)
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.block_tables = block_tables
        self.page_table_is_v = page_table_is_v
        num_stages = pipeline_config.num_stages
        total_entries = num_stages * cfg.page_table_window_entries
        self.alloc = SmemAllocation(
            "smem_page_offsets_v" if page_table_is_v else "smem_page_offsets_k",
            size_bytes=total_entries * 4,
            # Eight-ID vector loads also serve smaller pages in chunks.
            alignment=(32 if cfg.kv_tile_n // cfg.num_tokens_per_page >= 8 else 16),
        )
        self.smem_page_offsets = placeholder_smem_array(Int32, total_entries)
        self.cached_page_ids = TaskLocalVariable(
            dtype=cutlass.Array,
            default=None,
            docs="Page IDs retained while a delayed V tile crosses a page window.",
        )

    @consumer_work(returns=cached_page_ids)
    @cute.jit
    def cache_tile_page_ids(
        self,
        stage_info: StageInfo,
        *,
        cached_page_ids: cutlass.Array,
        kv_tile_start: Int32,
        tile_offset: cutlass.Constexpr[int] = 0,
    ) -> cutlass.Array:
        """Retain one tile's page IDs after its SMEM window is released."""
        tile_idx = resolve_kv_tile_idx_context(
            stage_info, kv_tile_start, tile_offset=tile_offset
        )
        page_ids = self.page_ids(tile_idx)
        pages_per_tile = self.cfg.kv_tile_n // self.cfg.num_tokens_per_page
        for page_frag in cutlass.range_constexpr(pages_per_tile):
            cached_page_ids[page_frag] = Int32(page_ids[page_frag])
        return cached_page_ids

    def dma_consumer_release_labels_for(
        self, downstream: MemoryResource
    ) -> set[str] | None:
        """Bind page-offset releases to the K/V TMA loads that consumed them."""
        if isinstance(downstream, SmemKVResource):
            labels: set[str] = set()
            binds_k = not (self.cfg.reuses_page_table_windows and self.page_table_is_v)
            binds_v = not (
                self.cfg.reuses_page_table_windows and not self.page_table_is_v
            )
            if downstream.holds_k and binds_k:
                labels |= {"k_load", "k_load_stage"}
            if downstream.holds_v and binds_v:
                labels |= {"v_load", "v_load_stage", "v_load_stage_cached"}
            return labels or None
        return None

    def get_smem_requirements(self) -> list[SmemAllocation]:
        return [self.alloc]

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=cached_page_ids)
    @cute.jit
    def init_cached_read_state(self, stage_info: StageInfo) -> cutlass.Array:
        """Initialize the SMEM view and register cache for one tile's page IDs."""
        self.init_smem_state(stage_info)
        return cutlass.Array(
            Int32,
            self.cfg.kv_tile_n // self.cfg.num_tokens_per_page,
            space=cutlass.AddressSpace.rmem,
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self.init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo) -> None:
        self.init_smem_state(stage_info)

    @cute.jit
    def init_smem_state(self, stage_info: StageInfo) -> None:
        smem_base = stage_info.context.smem_base
        num_stages = self.pipeline_config.num_stages
        total_entries = num_stages * self.cfg.page_table_window_entries
        self.smem_page_offsets = cutlass.Array(
            smem_base.data_ptr() + self.alloc.offset,
            dtype=cutlass.Int32,
            shape=(total_entries,),
            addrspace=3,
        )

    @producer_work
    @cute.jit
    def load_k(
        self,
        stage_info: StageInfo,
        *,
        tile_offset: cutlass.Constexpr[int] = 0,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """Prefetch K-side page IDs for the current K tile."""
        self.producer_load_page_offsets(
            stage_info,
            tile_offset=tile_offset,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @producer_work
    @cute.jit
    def load_v(
        self,
        stage_info: StageInfo,
        *,
        previous: cutlass.Constexpr[bool] = False,
        tile_offset: cutlass.Constexpr[int] = 0,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        """Prefetch V-side page IDs for the current V tile."""
        self.producer_load_page_offsets(
            stage_info,
            tile_offset=tile_offset + (-1 if previous else 0),
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )

    @cute.jit
    def page_ids(self, tile_idx: Int32) -> cutlass.Array:
        """Slice ``pages_per_tile`` entries from the cached page-ID stage.

        ``tile_idx`` is the runtime-resolved K/V tile index (same expression
        the page-offsets producer uses). The window-aligned base is implicit
        in the stage's contents; this LDS picks the per-tile entries.
        """
        cfg = self.cfg
        pages_per_tile = cfg.kv_tile_n // cfg.num_tokens_per_page
        window_entries = cfg.page_table_window_entries
        stage_idx = self.state_src.consumer_work_stage
        group_page_idx = (tile_idx * Int32(pages_per_tile)) & Int32(window_entries - 1)
        offset = stage_idx * Int32(window_entries) + group_page_idx
        if cutlass.const_expr(pages_per_tile <= 8):
            return self.smem_page_offsets.load(
                offset, vector_size=pages_per_tile, alignment=pages_per_tile * 4
            )
        tile_page_ids = cutlass.Array(
            Int32, pages_per_tile, space=cutlass.AddressSpace.rmem
        )
        for chunk in cutlass.range_constexpr(pages_per_tile // 8):
            values = self.smem_page_offsets.load(
                offset + Int32(chunk * 8), vector_size=8, alignment=32
            )
            for i in cutlass.range_constexpr(8):
                tile_page_ids[chunk * 8 + i] = Int32(values[i])
        return tile_page_ids

    # Stage the current K/V tile page IDs into its acquired SMEM ring slot.
    @cute.jit
    def producer_load_page_offsets(
        self,
        stage_info: StageInfo,
        tile_offset: cutlass.Constexpr[int] = 0,
        *,
        kv_tile_start: Int32,
        kv_request_begin: Int32,
        kv_page_idx_ub: Int32,
    ) -> None:
        cfg = self.cfg
        # Context's K/V tile index is kv_tile_start + loop_offset; for V the
        # producer runs one tile ahead in TAIL, so reuse the same expression
        # but consult kv_tile_start with the loop's stage_info.
        tile_idx = resolve_kv_tile_idx_context(
            stage_info, kv_tile_start, tile_offset=tile_offset
        )
        pages_per_tile = Int32(cfg.kv_tile_n // cfg.num_tokens_per_page)

        block_tables = self.block_tables
        smem_page_offsets = self.smem_page_offsets
        lane_idx = cute.arch.thread_idx()[0] & Int32(0x1F)
        # Lanes cooperatively fetch one topology-derived aligned window. A
        # split-D window gives each lane one scalar per D stage; consumers then
        # slice per-tile entries via ``page_ids``.
        window_entries = cfg.page_table_window_entries
        grouped_base_page_idx = (
            (tile_idx * pages_per_tile) // Int32(window_entries)
        ) * Int32(window_entries)
        grouped_smem_base = stage_info.stage_idx * Int32(window_entries)
        entries_per_lane = window_entries // cute.arch.WARP_SIZE
        for lane_group in cutlass.range_constexpr(entries_per_lane):
            lane_offset = lane_idx + Int32(lane_group * cute.arch.WARP_SIZE)
            grouped_logical_page_idx = cute.math.min(
                grouped_base_page_idx + lane_offset, kv_page_idx_ub
            )
            prims.cp_async_shared_global(
                smem_page_offsets.data_ptr() + grouped_smem_base + lane_offset,
                block_tables + kv_request_begin + grouped_logical_page_idx,
                4,
                "ca",
            )

    @consumer_work
    @cute.jit
    def read_offsets(self, stage_info: StageInfo) -> None:
        return


def pv_inner_dim_size_bytes(cfg: FmhaConfig) -> int:
    """Return the byte width of one P/V tile inner dimension."""
    return cfg.smem_v_head_dim // cfg.cta_group_size * cfg.v_dtype.width // 8


def pv_smem_desc_offsets(cfg: FmhaConfig) -> SmemDescOffsets:
    """Return V descriptor leading and stride byte offsets for PV MMA."""
    if cfg.two_cta_umma:
        # One fragment of pv_n_per_cta columns per CTA.
        return 0, cfg.tma_copy_v_granu_inner * cfg.v_dtype.width
    leading_byte_offset = 0
    if cfg.tma_copy_v_iters != 1:
        tma_copy_v_iters = (
            cfg.tma_copy_v_stage_iters
            if cfg.stage_kv_by_head_dim
            else cfg.tma_copy_v_iters
        )
        leading_byte_offset = cfg.tma_copy_v_bytes // tma_copy_v_iters
    stride_byte_offset = cfg.tma_copy_v_granu_inner * cfg.v_dtype.width
    return leading_byte_offset, stride_byte_offset


def pv_smem_layout(cfg: FmhaConfig) -> int:
    """Return the tcgen05 descriptor layout selector for V SMEM tiles."""
    return smem_layout_for_inner_bytes(pv_inner_dim_size_bytes(cfg))


def pv_smem_swizzle(cfg: FmhaConfig) -> cutlass.Swizzle:
    """Return the physical TMA swizzle used by V SMEM fragments."""
    inner_dim_size = pv_inner_dim_size_bytes(cfg)
    if inner_dim_size % 128 == 0:
        return cutlass.Swizzle(3, 4, 3)
    if inner_dim_size == 64:
        return cutlass.Swizzle(2, 4, 3)
    if inner_dim_size == 32:
        return cutlass.Swizzle(1, 4, 3)
    raise RuntimeError(f"Unsupported inner dimension size: {inner_dim_size}")
