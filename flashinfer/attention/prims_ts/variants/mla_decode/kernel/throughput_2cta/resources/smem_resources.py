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

"""SMEM Q, K/V and page-offset rings of the 2CTA MLA kernel."""

import cutlass.cute as cute
import cutlass
from cutlass import Int16, Int32, Int64
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field
from ......device.tensor_map import transform_ragged_coords
from ...device.constants import TMA_GATHER_ROWS, WARP_LANES
from ...device.gather import (
    gather4_cached,
    gather4_uniform_quad_slices,
    load_sparse_quad,
    load_sparse_rows,
)
from ...device.math import (
    qk_desc_layout,
    qk_desc_layout_for_head_dim,
    qk_desc_leading_byte_offset,
    qk_desc_leading_byte_offset_for_head_dim,
    qk_desc_stride_byte_offset,
    qk_desc_stride_byte_offset_for_head_dim,
    qkv_dtype,
    qkv_major_k_stride_bytes_for,
)
from ...device.nvfp4_kv import prefetch_nvfp4_kv_row
from ...device.query import copy_q_fp8_to_tmem, query_batch_bounds
from ...device.sparse_indices import (
    SparseKVIndices,
    SparseSourceTile,
    cache_sparse_source_lengths,
)
from ..config import MlaDecodeConfig, V_SMEM_K_BLOCK_TOKENS, V_TMA_LATENT_ELEMENTS
from ..nvfp4_kv_load import load_nvfp4_kv_stage, publish_nvfp4_v
from .common import HighThroughputMlaResource, with_cluster_acquire_waits
from typing import Any, ClassVar


# =====================================================================
# PageOffsetWindowResource — TMA-warp register page-table window
# =====================================================================


@dataclass(kw_only=True)
class PageOffsetWindowResource(HighThroughputMlaResource):
    """Page-coordinate staging for paged and sparse KV loads.

    FP8 sparse loaders read uniform quads from the prefetched shared ring.
    Other schedules retain page windows in registers.
    """

    # Static configuration
    cached_k_pages: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_mask_high: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_mask_low: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_next_v_pages: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_next_window_page: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_page_stage: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_v_pages: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cached_window_page: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("cached_page_stage", Int32, Int32(0), "Held shared-offset stage."),
        ("cached_mask_low", Int32, Int32(0), "Low half of the prefetched sparse mask."),
        (
            "cached_mask_high",
            Int32,
            Int32(0),
            "High half of the prefetched sparse mask.",
        ),
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
        (
            "cached_next_window_page",
            Int32,
            Int32(0),
            "This lane's page id in the next 32-entry window (dense FP8 "
            "loaders), loaded one window ahead so that a window boundary "
            "does not wait on the page table.",
        ),
    )

    # Per-task state
    extra_cache: Any = None
    page_offsets: Any = None  # GMEM page-offset tensor
    primary_cache: Any = None
    request_lengths: Any = field(init=False, default=None)
    smem_sparse_offsets: Any = None

    @cute.jit
    def advance_dense_page_window(
        self, stage_info: StageInfo, cached_window_page, cached_next_window_page
    ):
        """Advance the FP8 dense loader's prefetched page-ID window.

        A window is one page ID per lane (32 per warp). Each lane holds its ID
        of the current window and of the next. The next window's load is
        issued at the start of the current window (``32 / pages_per_k_tile`` K
        tiles before it is used); ptxas copies the loop-carried register at the
        next K tile, so the load has at least one K tile to land. Only the
        first window of a work tile is loaded synchronously, so no page-table
        read sits between a freed K/V slot and its TMA. Indices clamp to the
        last table column; fragments past the sequence are masked by the
        runtime length.
        """
        cfg = self.cfg
        pages_per_tile = cutlass.const_expr(cfg.pages_per_k_tile)
        window_tiles = cutlass.const_expr(WARP_LANES // pages_per_tile)
        work_tile = stage_info.work_tile
        local_k_index = Int32(stage_info.loop_offset)
        global_k_index = work_tile.k_index_base + local_k_index
        page_offsets_batch = self.page_offsets[None, work_tile.tile_idx[2]]
        last_page_col = Int32(page_offsets_batch.shape[0] - 1)
        lane_idx = cute.arch.thread_idx()[0] & Int32(31)
        window_pos = local_k_index & Int32(window_tiles - 1)
        if local_k_index == Int32(0):
            cached_window_page = Int32(
                page_offsets_batch[
                    cute.math.min(
                        global_k_index * Int32(pages_per_tile) + lane_idx,
                        last_page_col,
                    )
                ]
            )
        if window_pos == Int32(0):
            if local_k_index != Int32(0):
                cached_window_page = cached_next_window_page
            # Short split-KV work tiles fit in one window; skip a dead prefetch.
            if local_k_index + Int32(window_tiles) < Int32(work_tile.k_tile_count):
                cached_next_window_page = Int32(
                    page_offsets_batch[
                        cute.math.min(
                            (global_k_index + Int32(window_tiles))
                            * Int32(pages_per_tile)
                            + lane_idx,
                            last_page_col,
                        )
                    ]
                )
        return cached_window_page, cached_next_window_page, window_pos

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def bind_sparse_request(self, stage_info: StageInfo):
        if cutlass.const_expr(
            isinstance(self.page_offsets, SparseKVIndices)
            and not self.page_offsets.is_bound
        ):
            row = stage_info.work_tile.tile_idx[2]
            self.request_lengths[0] = Int32(self.page_offsets.primary_lengths[row])
            self.request_lengths[1] = Int32(self.page_offsets.extra_lengths[row])

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

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(cached_k_pages, cached_window_page, cached_next_window_page),
    )
    @cute.jit
    def init_dense_k_window(self, stage_info: StageInfo):
        """Create the FP8 dense K loader's page-ID window state."""
        del stage_info
        return (
            cutlass.Array(
                Int32, self.cfg.pages_per_k_cta, space=cutlass.AddressSpace.rmem
            ),
            Int32(0),
            Int32(0),
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(cached_v_pages, cached_window_page, cached_next_window_page),
    )
    @cute.jit
    def init_dense_v_window(self, stage_info: StageInfo):
        """Create the FP8 dense V loader's page-ID window state."""
        del stage_info
        return (
            cutlass.Array(
                Int32, self.cfg.pages_per_v_tile, space=cutlass.AddressSpace.rmem
            ),
            Int32(0),
            Int32(0),
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
        # Uniform caches repeat each warp's K/current-V coordinates in all
        # lanes, saving shuffles on later feature slices. Keep pending V in
        # compact lane fragments until needed to limit live registers.
        return (
            cutlass.Array(
                Int32,
                (
                    cfg.tokens_per_k_cta // cfg.load_num_warps
                    if cfg.cache_uniform_sparse_pages
                    else cfg.tokens_per_k_cta // WARP_LANES
                )
                if cfg.page_size == 1
                else cfg.pages_per_k_cta,
                space=cutlass.AddressSpace.rmem,
            ),
            cutlass.Array(
                Int32,
                (
                    cfg.tokens_per_v_tile // cfg.load_num_warps
                    if cfg.cache_uniform_sparse_pages
                    else cfg.tokens_per_v_tile // WARP_LANES
                )
                if cfg.page_size == 1
                else cfg.pages_per_v_tile,
                space=cutlass.AddressSpace.rmem,
            ),
            cutlass.Array(
                Int32,
                cfg.tokens_per_v_tile // WARP_LANES
                if cfg.page_size == 1
                else cfg.pages_per_v_tile,
                space=cutlass.AddressSpace.rmem,
            ),
            Int32(0),
        )

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_sparse_state(self, stage_info: StageInfo):
        del stage_info
        # Allocate outside skipped-work guards; refresh values for every tile.
        self.request_lengths = cutlass.Array(Int32, 2, space=cutlass.AddressSpace.rmem)

    @producer_work
    @cute.jit
    def load_sparse_offsets(self, stage_info: StageInfo):
        """Stage source-tagged physical row IDs for the next K/V tile.

        With assume_valid_prefix, entries before each live length contain no
        -1 holes, so each CTA can retain only its own 64 rows. Otherwise keep
        the full 128-row metadata tile for score masking. Valid-prefix does
        not imply contiguous physical rows, sorted indices, or no tail padding."""
        lane = cute.arch.thread_idx()[0] % 32
        k_index = stage_info.work_tile.k_index_base + Int32(stage_info.loop_offset)
        token = k_index * Int32(128) + lane * Int32(4)
        row = stage_info.work_tile.tile_idx[2]
        if cutlass.const_expr(
            isinstance(self.page_offsets, SparseKVIndices)
            and not self.page_offsets.is_bound
        ):
            routes = self.page_offsets.bind_query_row(
                row, self.request_lengths[0], self.request_lengths[1]
            )
        else:
            routes = self.page_offsets
        active = cutlass.Boolean(True)
        if cutlass.const_expr(
            self.cfg.has_nvfp4_kv
            and getattr(self.page_offsets, "assume_valid_prefix", False)
        ):
            active = lane // Int32(16) == cute.arch.make_warp_uniform(
                cute.arch.block_idx_in_cluster()
            )
        if active:
            quad = load_sparse_quad(routes, row, token)
            if cutlass.const_expr(self.cfg.has_nvfp4_kv):
                cta = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                if lane // Int32(16) == cta:
                    for j in cutlass.range_constexpr(4):
                        prefetch_nvfp4_kv_row(
                            self.primary_cache, self.extra_cache, quad[j], self.cfg
                        )

            self.smem_sparse_offsets.data_ptr(
                stage_info.stage_idx * 128 + lane * 4
            ).store(quad.load(0, 4), alignment=16)
            cute.arch.fence_view_async_shared()

    @consumer_work(
        returns=(cached_k_pages, cached_window_page, cached_next_window_page)
    )
    @cute.jit
    def read_dense_k_window(
        self,
        stage_info: StageInfo,
        *,
        cached_k_pages,
        cached_window_page,
        cached_next_window_page,
    ):
        """Return this CTA's K page IDs for the current tile from the window."""
        cfg = self.cfg
        cached_window_page, cached_next_window_page, window_pos = (
            self.advance_dense_page_window(
                stage_info, cached_window_page, cached_next_window_page
            )
        )
        cta_v = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        for pk in cutlass.range_constexpr(cfg.pages_per_k_cta):
            # Same logical page order as ``SmemKResource.tma_load_paged``.
            source_lane = window_pos * Int32(cfg.pages_per_k_tile) + (
                cta_v * Int32(cfg.pages_per_k_cta) + Int32(pk)
                if cfg.pages_per_k_tile > 1
                else Int32(0)
            )
            cached_k_pages[pk] = Int32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_window_page,
                    offset=source_lane,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
            )
        return cached_k_pages, cached_window_page, cached_next_window_page

    @consumer_work(
        returns=(cached_v_pages, cached_window_page, cached_next_window_page)
    )
    @cute.jit
    def read_dense_v_window(
        self,
        stage_info: StageInfo,
        *,
        cached_v_pages,
        cached_window_page,
        cached_next_window_page,
    ):
        """Return the V page IDs of the current tile from the window."""
        cfg = self.cfg
        cached_window_page, cached_next_window_page, window_pos = (
            self.advance_dense_page_window(
                stage_info, cached_window_page, cached_next_window_page
            )
        )
        for pk in cutlass.range_constexpr(cfg.pages_per_v_tile):
            cached_v_pages[pk] = Int32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=cached_window_page,
                    offset=window_pos * Int32(cfg.pages_per_v_tile) + Int32(pk),
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
            )
        return cached_v_pages, cached_window_page, cached_next_window_page

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
        if cutlass.const_expr(cfg.page_size == 1):
            cta = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            # K and pending V cover the same source-aligned BK128 tile.
            # Bind its source once, not again for every lane/fragment.
            routes = (
                self.page_offsets.select_source_tile(
                    global_k_index * cfg.tokens_per_k_tile, blk_coord[2]
                )
                if cutlass.const_expr(isinstance(self.page_offsets, SparseKVIndices))
                else self.page_offsets
            )
            if cutlass.const_expr(isinstance(routes, SparseSourceTile)):
                new_v = load_sparse_rows(
                    routes,
                    blk_coord[2],
                    global_k_index * cfg.tokens_per_v_tile,
                    fragments=cfg.tokens_per_v_tile // WARP_LANES,
                )
                # K is one CTA half of this freshly loaded pending-V window.
                # cached_v_pages instead belongs to the previous K tile.
                fragments_k = cfg.tokens_per_k_cta // WARP_LANES
                new_k = cutlass.Array(
                    Int32, fragments_k, space=cutlass.AddressSpace.rmem
                )
                for fragment in cutlass.range_constexpr(fragments_k):
                    new_k[fragment] = (
                        new_v[fragment]
                        if cta == Int32(0)
                        else new_v[fragment + fragments_k]
                    )
            else:
                new_k = load_sparse_rows(
                    routes,
                    blk_coord[2],
                    global_k_index * cfg.tokens_per_k_tile + cta * cfg.tokens_per_k_cta,
                    fragments=cfg.tokens_per_k_cta // WARP_LANES,
                )
                new_v = load_sparse_rows(
                    routes,
                    blk_coord[2],
                    global_k_index * cfg.tokens_per_v_tile,
                    fragments=cfg.tokens_per_v_tile // WARP_LANES,
                )
            # Expand only current K/V coordinates once per tile. "Uniform"
            # means identical across lanes, not consecutive physical KV rows.
            # Reuse across D slices trades register capacity for fewer shuffles.
            if cutlass.const_expr(cfg.cache_uniform_sparse_pages):
                warp = (
                    cute.arch.make_warp_uniform(cute.arch.warp_idx())
                    - cfg.load_tma_warp_id
                )
                fragment = (
                    new_k[0] if warp < Int32(cfg.load_num_warps // 2) else new_k[1]
                )
                for i in cutlass.range_constexpr(64 // cfg.load_num_warps):
                    cached_k_pages[i] = cute.arch.make_warp_uniform(
                        prims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=fragment,
                            offset=(warp % Int32(cfg.load_num_warps // 2))
                            * Int32(64 // cfg.load_num_warps)
                            + Int32(i),
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.IDX,
                        )
                    )
            else:
                for i in cutlass.range_constexpr(2):
                    cached_k_pages[i] = new_k[i]
            if cutlass.const_expr(cfg.cache_uniform_sparse_pages):
                # Expand only the delayed V tile; retain the next V tile in
                # lane fragments to keep the register footprint bounded.
                for i in cutlass.range_constexpr(128 // cfg.load_num_warps):
                    if cutlass.const_expr(init_v_cache):
                        cached_v_pages[i] = Int32(0x7FFFFFFF)
                    else:
                        cached_v_pages[i] = cute.arch.make_warp_uniform(
                            prims.shfl_sync(
                                thread_mask=0xFFFFFFFF,
                                val=cached_next_v_pages[
                                    i // (32 // cfg.load_num_warps)
                                ],
                                offset=warp * Int32(32 // cfg.load_num_warps)
                                + Int32(i % (32 // cfg.load_num_warps)),
                                mask_and_clamp=0x1F,
                                kind=prims.Shfl.IDX,
                            )
                        )
                for i in cutlass.range_constexpr(4):
                    cached_next_v_pages[i] = new_v[i]
            else:
                for i in cutlass.range_constexpr(4):
                    cached_v_pages[i] = (
                        Int32(0x7FFFFFFF) if init_v_cache else cached_next_v_pages[i]
                    )
                    cached_next_v_pages[i] = new_v[i]
            return (
                cached_k_pages,
                cached_v_pages,
                cached_next_v_pages,
                cached_window_page,
            )
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

    @consumer_work(returns=(cached_mask_low, cached_mask_high))
    @cute.jit
    def read_sparse_mask(self, stage_info: StageInfo):
        lane = cute.arch.thread_idx()[0] % 32
        half = (cute.arch.thread_idx()[0] % 128) // 64
        base = stage_info.stage_idx * 128 + half * 64
        tile = stage_info.work_tile.k_index_base + Int32(stage_info.loop_offset)
        token = tile * Int32(128) + half * Int32(64) + lane
        length = Int32(stage_info.work_tile.k_len)
        low = self.smem_sparse_offsets[base + lane]
        high = self.smem_sparse_offsets[base + lane + 32]
        low_bits = Int32(
            cute.arch.vote_ballot_sync(
                ((low & Int32(0x7FFFFFFF)) == Int32(0x7FFFFFFF)) | (token >= length)
            )
        )
        high_bits = Int32(
            cute.arch.vote_ballot_sync(
                ((high & Int32(0x7FFFFFFF)) == Int32(0x7FFFFFFF))
                | (token + Int32(32) >= length)
            )
        )
        return low_bits, high_bits

    @consumer_work(returns=cached_page_stage)
    @cute.jit
    def sparse_stage(self, stage_info: StageInfo):
        """Return the shared-ring slot protected by the current consumer token."""
        return stage_info.stage_idx


# =====================================================================
# SmemKResource — FP8 K SMEM buffer with one pipeline stage per K tile
# =====================================================================


@dataclass(kw_only=True)
class SmemKResource(HighThroughputMlaResource):
    """SMEM K buffer for the FP8 split-MMA path.

    Producer: LoadKTask. Consumer: MmaQkTask. A single producer stage contains
    all latent K sub-tiles and optional RoPE channels for one logical K tile.
    """

    # Static configuration
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    desc_k_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    paired_v: cutlass.Constexpr = None
    task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_k_base", Int64, Int64(0), "SMEM descriptor for staged K."),
    )

    # Runtime scalars
    logical_seq_len_q: Int32 = 1

    # Per-task state
    # Source metadata for NVFP4 gathers; dense loaders consume the page window.
    page_offsets: Any = None
    raw_mbar: Any = None
    request_lengths: Any = field(init=False, default=None)
    smem_k: Any = None
    smem_raw: Any = None
    smem_sparse_offsets: Any = None
    tma_desc_c_latent: Any = None
    tma_desc_c_rope: Any = None
    tma_desc_sparse_extra_rope: Any = None
    tma_desc_sparse_rope: Any = None

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def bind_sparse_request(self, stage_info: StageInfo):
        """Cache live source lengths for this query before its KV/softmax loop."""
        if cutlass.const_expr(self.cfg.has_nvfp4_kv):
            cache_sparse_source_lengths(
                self.page_offsets,
                stage_info.work_tile.tile_idx[2],
                self.request_lengths,
                include_primary_tiles=True,
            )

    def create_pipeline(self, pipeline_config):
        """Use cluster-scope consumer acquires for K published by both conversion CTAs."""
        base = super().create_pipeline(pipeline_config)
        if self.cfg.has_nvfp4_kv:
            return with_cluster_acquire_waits(base)
        return base

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize per-warp state used by K TMA loads."""
        del stage_info
        if cutlass.const_expr(self.cfg.has_nvfp4_kv):
            self.request_lengths = cutlass.Array(
                Int32,
                3,
                space=cutlass.AddressSpace.rmem,
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

    @producer_work
    @cute.jit
    def tma_load_paged(self, stage_info: StageInfo, *, cached_page_ids) -> None:
        """Load a paged K tile with this CTA's page IDs from the page window.

        ``PageOffsetWindowResource.read_dense_k_window`` supplies the IDs in
        the logical order ``k_index * pages_per_k_tile + cta * pages_per_k_cta
        + pk`` (or ``k_index`` when one page spans the tile).
        """
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        stage_base = stage_idx * cfg.smem_k_stage_elems
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
        # Fragments past the page table use the window's clamped last column;
        # the runtime sequence length masks them.

        for k_call in cutlass.range_constexpr(cfg.iterations_qk_latent):
            coord_kcl = cutlass.Int32(k_call * cfg.mma_qk_tiler_k)
            subtile_base = stage_base + k_call * k_latent_subtile_elems
            for pk in cutlass.range_constexpr(pages_per_k_cta):
                page_idx = Int32(cached_page_ids[pk])
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
                page_idx = Int32(cached_page_ids[pk])
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
    def tma_load_shared(self, stage_info: StageInfo, *, page_offset_stage):
        """Read one warp-uniform SMEM quad and reuse it across FP8 feature slices."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.has_nvfp4_kv):
            load_nvfp4_kv_stage(
                self,
                self.paired_v,
                stage_info,
                self.paired_v.current_slot[0],
                page_offset_stage,
            )
            return
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - cfg.load_k_warp_id
        base = (
            cute.arch.make_warp_uniform(stage_info.stage_idx) * cfg.smem_k_stage_elems
        )
        for owner in cutlass.range_constexpr(
            cfg.tokens_per_k_cta // TMA_GATHER_ROWS // cfg.load_num_warps
        ):
            # All lanes read the same quad: shared-memory broadcast replaces
            # four register shuffles. Retain the quad across feature slices.
            quad = warp + Int32(owner * cfg.load_num_warps)
            offset = (
                page_offset_stage * Int32(128)
                + quad * Int32(4)
                + cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                * Int32(64)
            )
            raw = self.smem_sparse_offsets.data_ptr(offset).load(count=4, alignment=16)
            gather4_uniform_quad_slices(
                self.smem_k.data_ptr(base + quad * 4 * cfg.mma_qk_tiler_k),
                self.tma_desc_c_latent,
                self.tma_desc_c_rope,
                Int32(0),
                raw,
                stage_info.barrier,
                num_slices=cfg.iterations_qk_latent,
                dst_stride=64 * cfg.mma_qk_tiler_k,
                col_stride=cfg.mma_qk_tiler_k,
            )
            if cutlass.const_expr(cfg.rope_dim):
                gather4_uniform_quad_slices(
                    self.smem_k.data_ptr(
                        base + 64 * cfg.latent_dim + quad * 4 * cfg.rope_dim
                    ),
                    self.tma_desc_sparse_rope,
                    self.tma_desc_sparse_extra_rope,
                    Int32(0),
                    raw,
                    stage_info.barrier,
                    num_slices=1,
                    dst_stride=64 * cfg.rope_dim,
                    col_stride=0,
                )


# =====================================================================
# SmemKVResource — K/V SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemKVResource(HighThroughputMlaResource):
    """SMEM K/V buffer.  Producer: LoadTma (TMA).  Consumer: Mma.

    Pipeline: TmaUmmaAsync, 7 stages.
    Per logical k-tile: K sub-tiles are loaded before delayed V sub-tiles.
    """

    # Static configuration
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    desc_k_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    desc_v_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_k_base", Int64, Int64(0), "SMEM descriptor for staged K."),
        ("desc_v_base", Int64, Int64(0), "SMEM descriptor for staged V."),
    )

    # Per-task state
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)
    smem_kv: Any = None
    tma_desc_c_latent: Any = None
    tma_desc_c_rope: Any = None
    tma_desc_c_transpose: Any = None
    tma_desc_extra_v: Any = None
    tma_desc_sparse_extra_rope: Any = None
    tma_desc_sparse_rope: Any = None

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize CTA-local KV-load descriptors and leader state."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0

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

        if cutlass.const_expr(cfg.page_size == 1):
            cta = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            base = stage_idx * cfg.smem_k_stage_elems
            warp = (
                cute.arch.make_warp_uniform(cute.arch.warp_idx()) - cfg.load_tma_warp_id
            )
            rope = not is_v and subtile_idx >= cfg.iterations_qk_latent_stages
            primary_k = self.tma_desc_sparse_rope if rope else self.tma_desc_c_latent
            extra_k = self.tma_desc_sparse_extra_rope if rope else self.tma_desc_c_rope
            # Each issuer owns disjoint four-row quads. K covers 64 rows per
            # CTA (16 quads); V copies 32-token subtiles (8 quads). BF16 V is
            # blocked as 32x64 panels: two 64-feature/128-byte slices cover
            # each CTA's 128 features within a 256-feature PV panel. Source
            # columns are elements; gather4 destination strides are bytes.
            if cutlass.const_expr(cfg.load_num_warps in (4, 8)):
                if cutlass.const_expr(not is_v):
                    if cutlass.const_expr(not cfg.cache_uniform_sparse_pages):
                        fragment = (
                            cached_k_pages[0]
                            if warp < Int32(cfg.load_num_warps // 2)
                            else cached_k_pages[1]
                        )
                    for owner in cutlass.range_constexpr(16 // cfg.load_num_warps):
                        raw = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
                        for j in cutlass.range_constexpr(4):
                            if cutlass.const_expr(cfg.cache_uniform_sparse_pages):
                                raw[j] = cached_k_pages[owner * 4 + j]
                            else:
                                lane_base = (
                                    warp % Int32(cfg.load_num_warps // 2)
                                ) * Int32(64 // cfg.load_num_warps) + Int32(owner * 4)
                                raw[j] = cute.arch.make_warp_uniform(
                                    prims.shfl_sync(
                                        thread_mask=0xFFFFFFFF,
                                        val=fragment,
                                        offset=lane_base + Int32(j),
                                        mask_and_clamp=0x1F,
                                        kind=prims.Shfl.IDX,
                                    )
                                )
                        group = warp * Int32(16 // cfg.load_num_warps) + Int32(owner)
                        gather4_uniform_quad_slices(
                            self.smem_kv.data_ptr(
                                base + group * 4 * cfg.mma_qk_tiler_k
                            ),
                            primary_k,
                            extra_k,
                            Int32(
                                0
                                if rope
                                else subtile_idx
                                * cfg.kv_subtiles_per_stage
                                * cfg.mma_qk_tiler_k
                            ),
                            raw,
                            stage_info.barrier,
                            num_slices=cfg.kv_subtiles_per_stage,
                            dst_stride=64 * cfg.mma_qk_tiler_k * cfg.qkv_dtype_bytes,
                            col_stride=0 if rope else cfg.mma_qk_tiler_k,
                        )
                else:
                    # PV consumes the delayed current-V window; the tail can
                    # instead request the compact pending-V coordinates.
                    cached = cached_next_v_pages if use_next_v_pages else cached_v_pages
                    pv_j = subtile_idx // cfg.kv_subtiles_per_stage
                    token_partition = subtile_idx % cfg.kv_subtiles_per_stage
                    for part in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
                        pv_i = token_partition * cfg.kv_subtiles_per_stage + part
                        for owner in cutlass.range_constexpr(8 // cfg.load_num_warps):
                            group = warp * Int32(8 // cfg.load_num_warps) + Int32(owner)
                            raw = cutlass.Array(
                                Int32, 4, space=cutlass.AddressSpace.rmem
                            )
                            for j in cutlass.range_constexpr(4):
                                if cutlass.const_expr(
                                    cfg.cache_uniform_sparse_pages
                                    and not use_next_v_pages
                                ):
                                    raw[j] = cached_v_pages[
                                        pv_i * (32 // cfg.load_num_warps)
                                        + owner * 4
                                        + j
                                    ]
                                else:
                                    raw[j] = cute.arch.make_warp_uniform(
                                        prims.shfl_sync(
                                            thread_mask=0xFFFFFFFF,
                                            val=cached[pv_i],
                                            offset=group * Int32(4) + Int32(j),
                                            mask_and_clamp=0x1F,
                                            kind=prims.Shfl.IDX,
                                        )
                                    )
                            gather4_uniform_quad_slices(
                                self.smem_kv.data_ptr(
                                    base + part * 32 * 128 + group * 4 * 64
                                ),
                                self.tma_desc_c_transpose,
                                self.tma_desc_extra_v,
                                cta * Int32(128) + Int32(pv_j * 256),
                                raw,
                                stage_info.barrier,
                                num_slices=2,
                                dst_stride=32 * 64 * cfg.qkv_dtype_bytes,
                                col_stride=64,
                            )
                return
            # Fewer issuers reconstruct quads from compact lane fragments.
            if cutlass.const_expr(not is_v):
                for part in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
                    column = Int32(
                        0
                        if rope
                        else (subtile_idx * cfg.kv_subtiles_per_stage + part)
                        * cfg.mma_qk_tiler_k
                    )
                    for local_group in cutlass.range(
                        16 // cfg.load_num_warps, unroll_full=True
                    ):
                        group = local_group
                        gather4_cached(
                            self.smem_kv.data_ptr(
                                base
                                + part * 64 * cfg.mma_qk_tiler_k
                                + group * 4 * cfg.mma_qk_tiler_k
                            ),
                            primary_k,
                            extra_k,
                            column,
                            cached_k_pages,
                            group * 4,
                            stage_info.barrier,
                            cta_group=2,
                        )
            else:
                cached = cached_next_v_pages if use_next_v_pages else cached_v_pages
                pv_j = subtile_idx // cfg.kv_subtiles_per_stage
                token_partition = subtile_idx % cfg.kv_subtiles_per_stage
                for part in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
                    pv_i = token_partition * cfg.kv_subtiles_per_stage + part
                    for panel in cutlass.range_constexpr(2):
                        column = cta * 128 + Int32(pv_j * 256 + panel * 64)
                        for local_group in cutlass.range(
                            8 // cfg.load_num_warps, unroll_full=True
                        ):
                            group = local_group
                            gather4_cached(
                                self.smem_kv.data_ptr(
                                    base
                                    + part * 32 * 128
                                    + panel * 32 * 64
                                    + group * 4 * 64
                                ),
                                self.tma_desc_c_transpose,
                                self.tma_desc_extra_v,
                                column,
                                cached,
                                Int32(pv_i * 32) + group * 4,
                                stage_info.barrier,
                                cta_group=2,
                            )
            return

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
# SmemQResource — Q SMEM buffer with TmaUmmaAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class SmemQResource(HighThroughputMlaResource):
    """SMEM Q buffer (latent + rope).  Producer: LoadTma (TMA).  Consumer: Mma.

    Pipeline: TmaUmmaAsync, 1 stage.
    Q is loaded once (LoopFirstIter) and released once (LoopLastIter).
    """

    # Static configuration
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    logical_num_heads_q: cutlass.Constexpr[int] = 128

    # Runtime scalars
    logical_seq_len_q: Int32 = 1

    # Per-task state
    cta_rank: Any = field(init=False, default=None)
    cu_seqlens_q: Any = None
    is_leader: Any = field(init=False, default=None)
    smem_q_latent: Any = None  # SMEM Q latent array
    smem_q_rope: Any = None  # SMEM Q rope array
    tma_desc_q_latent: Any = None
    tma_desc_q_rope: Any = None
    tmem_base_addr: Any = None

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize CTA-local Q-load state for the high-throughput path."""
        del stage_info
        self.cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        self.is_leader = self.cta_rank == 0

    @consumer_work
    @cute.jit
    def q_desc(self, stage_info: StageInfo) -> None:
        """Stage64 local Q rows into the replicated group-2 TMEM A layout."""
        del stage_info
        if cutlass.const_expr(self.cfg.has_nvfp4_kv):
            if prims.elect_sync():
                copy_q_fp8_to_tmem(
                    self.smem_q_latent.data_ptr(),
                    self.tmem_base_addr + Int32(384),
                    cta_group=2,
                )

    @cute.jit
    def query_tma_coords(
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
        """Let one warp issue Q while the full loader group signals readiness."""
        if cutlass.const_expr(self.cfg.load_num_warps > 1):
            if cute.arch.thread_idx()[0] // 32 == self.cfg.load_k_warp_id:
                self.tma_load_impl(stage_info)
        else:
            self.tma_load_impl(stage_info)

    @cute.jit
    def tma_load_impl(self, stage_info: StageInfo) -> None:
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

        if prims.elect_sync():
            # Load Q latent sub-tiles
            q_latent_stage_elems = cutlass.const_expr(
                cfg.mma_qk_tiler[0] // cfg.num_mma_ctas * cfg.mma_qk_tiler_k
            )
            for i in cutlass.range(cfg.iterations_qk_latent):
                coord_kl = cutlass.Int32(i * cfg.mma_qk_tiler_k)
                q_latent_coords = self.query_tma_coords(
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
                q_rope_coords = self.query_tma_coords(
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


# =====================================================================
# SmemVResource — FP8 V SMEM buffer with one pipeline stage per V tile
# =====================================================================


@dataclass(kw_only=True)
class SmemVResource(HighThroughputMlaResource):
    """SMEM V buffer for the FP8 split-MMA path.

    Producer: LoadVTask. Consumer: MmaPvTask. A single stage contains every V
    sub-tile needed by the PV MMA for one logical K tile.
    """

    # Static configuration
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    desc_v_base: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("desc_v_base", Int64, Int64(0), "SMEM descriptor for staged V."),
    )

    # Runtime scalars
    logical_seq_len_q: Int32 = 1

    # Per-task state
    current_slot: Any = None
    raw_mbar: Any = None
    smem_raw: Any = None
    smem_sparse_offsets: Any = None
    smem_v: Any = None
    tma_desc_c_transpose: Any = None
    tma_desc_extra_v: Any = None
    tma_desc_sparse_extra_rope: Any = None
    tma_desc_sparse_rope: Any = None

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def combined_stage(self, stage_info: StageInfo):
        """Expose the acquired V stage index and phase to the paired K/V producer."""
        self.current_slot[0] = stage_info.stage_idx
        self.current_slot[1] = self.producer_state.phase ^ Int32(1)

    def create_pipeline(self, pipeline_config):
        """Use cluster-scope consumer acquires for V assembled from local and peer data."""
        base = super().create_pipeline(pipeline_config)
        if self.cfg.has_nvfp4_kv:
            return with_cluster_acquire_waits(base)
        return base

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize per-warp state used by V TMA loads."""
        del stage_info
        if cutlass.const_expr(self.cfg.has_nvfp4_kv):
            self.current_slot = cutlass.Array(Int32, 2, space=cutlass.AddressSpace.rmem)

    @producer_work
    @cute.jit
    def tma_load_paged(self, stage_info: StageInfo, *, cached_page_ids) -> None:
        """Load a paged V tile with the tile's page IDs from the page window.

        ``PageOffsetWindowResource.read_dense_v_window`` supplies the IDs in
        the logical order ``k_index * pages_per_v_tile + page_offset``.
        """
        cfg = self.cfg
        stage_idx = stage_info.stage_idx
        stage_base = stage_idx * cfg.smem_v_stage_elems

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
        # As for K, out-of-table fragments use the clamped last column and are
        # masked by the runtime sequence length. p128 shares one page ID across
        # CTAs.

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
                page_idx = Int32(
                    cached_page_ids[page_offset if pages_per_v_tile > 1 else 0]
                )
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
    def tma_load_shared(self, stage_info: StageInfo, *, page_offset_stage):
        """Read one warp-uniform SMEM quad and reuse it across FP8 V panels."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.has_nvfp4_kv):
            # Match this warp's share of the cluster-wide V pipeline credit.
            publish_nvfp4_v(
                stage_info.barrier,
                cfg.tma_copy_v_tile_bytes // cfg.num_mma_ctas // cfg.load_num_warps,
            )
            return
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - cfg.load_v_warp_id
        cta = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        base = (
            cute.arch.make_warp_uniform(stage_info.stage_idx) * cfg.smem_v_stage_elems
        )
        for owner in cutlass.range_constexpr(
            cfg.tokens_per_v_tile // TMA_GATHER_ROWS // cfg.load_num_warps
        ):
            # All lanes read the same quad: shared-memory broadcast replaces
            # four register shuffles. Retain the quad across feature slices.
            quad = warp + Int32(owner * cfg.load_num_warps)
            offset = page_offset_stage * Int32(128) + quad * Int32(4)
            raw = self.smem_sparse_offsets.data_ptr(offset).load(count=4, alignment=16)
            call = (quad // Int32(16)) * cfg.iterations_pv_n
            gather4_uniform_quad_slices(
                self.smem_v.data_ptr(
                    base
                    + call * 128 * cfg.mma_pv_tiler[2]
                    + (quad % Int32(16)) * 4 * 128
                ),
                self.tma_desc_c_transpose,
                self.tma_desc_extra_v,
                cta * 128,
                raw,
                stage_info.barrier,
                num_slices=cfg.iterations_pv_n,
                dst_stride=128 * cfg.mma_pv_tiler[2],
                col_stride=256,
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
        leading_byte_offset = cutlass.const_expr(
            V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS * cfg.qkv_dtype_bytes
        )
        stride_byte_offset = cutlass.const_expr(
            qkv_major_k_stride_bytes_for(cfg, cfg.mma_pv_tiler[2])
        )
        layout = cutlass.const_expr(
            qk_desc_layout_for_head_dim(cfg, cfg.mma_pv_tiler[2])
        )
        if cutlass.const_expr(cfg.page_size == 1):
            leading_byte_offset = 0
            stride_byte_offset = 1024
            layout = 2
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
        leading_byte_offset = cutlass.const_expr(
            V_SMEM_K_BLOCK_TOKENS * V_TMA_LATENT_ELEMENTS * cfg.qkv_dtype_bytes
        )
        stride_byte_offset = cutlass.const_expr(
            qkv_major_k_stride_bytes_for(cfg, cfg.mma_pv_tiler[2])
        )
        layout = cutlass.const_expr(
            qk_desc_layout_for_head_dim(cfg, cfg.mma_pv_tiler[2])
        )
        if cutlass.const_expr(cfg.page_size == 1):
            leading_byte_offset = 0
            stride_byte_offset = 1024
            layout = 2
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
