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

"""Task base class, work queues and schedule helpers shared by the decode tasks."""

import cutlass.cute as cute
import cutlass
import functools
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.memory import ResourceContext
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    WorkQueue,
)
from cutlass.experimental.task_scheduling.schedule_builder import (
    Schedule,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.task import Task
from cutlass.utils.static_persistent_tile_scheduler import WorkTileInfo
from dataclasses import dataclass, field
from ..config import FmhaDecodeConfig
from ..device.direct_sparse_metadata import (
    DirectSparseMetadataView,
    HeadIndexedMetadataView,
)
from ..device.q_rows import UniformQueryOffsets, q_group_token_base, q_seq_bounds
from ..device.common import (
    assume_nonnegative_i32,
    warp_broadcast_i32,
    split_kv_grid_coord,
)
from ..device.kv_tile_idx import (
    runtime_configured_local_kv_tiles,
    runtime_last_valid_page_idx,
    runtime_total_kv_tiles,
    sliding_window_start_idx,
    cta_caches_visible_kv_interval,
    cta_visible_kv_interval,
)
from .....device.task import task_context_kwargs
from typing import Any, Callable


LocalStatsState = tuple[
    cutlass.Array,
    cutlass.Array,
    cutlass.Array,
    cutlass.Array,
    cutlass.Array,
    cutlass.Array,
]
QBoundBinding = tuple[MemoryResource, bool]
TaskKwarg = (
    int
    | bool
    | cutlass.Int32
    | cute.Pointer
    | FmhaDecodeConfig
    | tuple[QBoundBinding, ...]
    | None
)


class DecodeGenTask(Task):
    """Decode-gen task with task-cached values used by hot resource paths."""

    def __init__(self, **kwargs: TaskKwarg) -> None:
        """Capture decode-specific task config and initialize cache slots."""
        self.cfg = kwargs.pop("cfg", None)
        self.requested_splits = kwargs.pop("requested_splits", None)
        self.seqlens_kv = kwargs.pop("seqlens_kv", None)
        self.block_table_capacity = kwargs.pop("block_table_capacity", None)
        self.sparse_row_route_offsets = kwargs.pop("sparse_row_route_offsets", None)
        self.sparse_row_route_counts = kwargs.pop("sparse_row_route_counts", None)
        # Static tiles may pass the already loaded row header instead.
        self.sparse_row_route_begin = kwargs.pop("sparse_row_route_begin", None)
        self.sparse_route_count = kwargs.pop("sparse_route_count", None)
        self.num_heads_kv = kwargs.pop("num_heads_kv", None)
        self.max_seq_len_kv = kwargs.pop("max_seq_len_kv", cutlass.Int32(0))
        self.seq_len_q = kwargs.pop("seq_len_q", None)
        self.domain_bias = kwargs.pop("domain_bias", 0)
        # Iterations per loop step of the shared domain; the static schedule
        # bound must already count scaled iterations.
        self.domain_scale = kwargs.pop("domain_scale", 1)
        self.q_bound_resources = kwargs.pop("q_bound_resources", ())
        # Only the softmax tasks read the keys every row of a work tile sees.
        self.consumes_visible_kv_interval = kwargs.pop(
            "consumes_visible_kv_interval", False
        )
        super().__init__(**kwargs)
        self.tmem_base_offset = cutlass.Int32(0)
        self.warp_grp_thread_idx = cutlass.Int32(0)
        self.local_warp_idx = cutlass.Int32(0)
        self.lane_idx = cutlass.Int32(0)
        self.seq_len_kv = cutlass.Int32(0)
        self.kv_request_begin = cutlass.Int32(0)
        self.kv_page_idx_ub = cutlass.Int32(0)
        self.kv_raw_tile_base = cutlass.Int32(0)
        self.kv_valid_tile_end = cutlass.Int32(0)
        self.kv_window_start = cutlass.Int32(0)
        if self.caches_visible_kv_interval:
            # Appended to the task cache only for configs that cache it. An
            # empty interval masks every tile, which is always exact.
            self.kv_visible_begin = cutlass.Int32(0)
            self.kv_visible_end = cutlass.Int32(0)
        # Keep the DSL loop-carried task structure stable. Persistent loops
        # assign this liveness marker on every path, so it must exist before
        # the first dynamic ``while`` is lowered by the stock compiler.
        self.dummy = cutlass.Boolean(False)

    @property
    def caches_visible_kv_interval(self) -> bool:
        """Return whether this task computes and carries the visible K interval.

        Only softmax tasks consume it, and only configs whose bounds need a
        multi-row intersection cache it; every other task keeps the plain
        task-cache layout and get_domain work.
        """
        return (
            self.consumes_visible_kv_interval
            and self.cfg is not None
            and cta_caches_visible_kv_interval(self.cfg)
        )

    @cute.jit
    def _run_task_body_impl(
        self,
        work_tile: cute.Coord,
        skip_work_tile: Any = None,
        context: ResourceContext | None = None,
    ) -> None:
        """Run one ordinary task tile and synchronize attention-sink tails."""
        queue_work_tile = work_tile
        if cutlass.const_expr(self.cfg is not None and self.cfg.runtime_split_kv):
            work_tile = WorkTileInfo(
                split_kv_grid_coord(self.cfg, work_tile.tile_idx),
                work_tile.is_valid_tile,
            )
        Task._run_task_body_impl(
            self,
            work_tile,
            skip_work_tile,
            **task_context_kwargs(context),
        )
        if cutlass.const_expr(
            self.cfg is not None
            and self.cfg.use_persistent_scheduler
            and self.cfg.use_attention_sinks
        ):
            # Attention sinks extend correction's tail beyond the ordinary
            # task graph.  Keep all persistent tasks on the same logical tile
            # until that tail has drained.  KV256's shared-KV alias instead
            # uses a narrow Load/Correction credit in the captured schedule.
            # Tasks with ``skip_if`` reach this barrier only on tiles the work
            # queue keeps, so tasks without it test the same predicate.
            assert isinstance(self.work_queue, PackedDecodeWorkQueue)
            if not self.work_queue.skip_work_tile_if(queue_work_tile):
                prims.barrier_cta_sync(12, thread_count=16 * 32)

    @cute.jit
    def _run_task_body_persistent(
        self,
        context: ResourceContext | None = None,
    ) -> None:
        """Drain inactive tiles before each unconditional active body."""
        use_packed_early_stop = (
            self.cfg is not None
            and self.cfg.use_persistent_scheduler
            and self._has_skip_if
        )
        if cutlass.const_expr(not use_packed_early_stop):
            Task._run_task_body_persistent(self, **task_context_kwargs(context))
            return

        assert self.work_queue is not None
        work_tile = self.work_queue.initial_work_tile_info()
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)

        self._run_pre_work_loop_entries(work_tile, **task_context_kwargs(context))
        work_tile = self.work_queue._get_consumer_var_from_ts("work_tile")
        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource.pipeline_config is not None
                and resource.pipeline_config.advance_on_acquire
                and not self._is_fork_secondary(resource)
            ):
                self._thread_advance_on_acquire_state(resource)

        # Consume overlaunched Q groups before entering the active loop. The
        # inner loop executes only the non-skippable WorkQueue tail, so no TMA,
        # descriptor, pipeline, task data, or sink barrier is issued.
        while work_tile.is_valid_tile and self._should_skip_work_tile(work_tile):
            self.run_packed_skip_iteration(work_tile, context)
            work_tile = self.work_queue._get_consumer_var_from_ts("work_tile")
            self.dummy = cutlass.Boolean(True)

        while work_tile.is_valid_tile:
            if cutlass.const_expr(self.cfg.use_variable_seqlens_q):
                self.refresh_packed_q_bounds(work_tile)
            # The tile is known active here. Running the complete schedule
            # without a dynamic skip guard keeps HEAD-produced pipeline state
            # in scope for LOOP and TAIL.
            Task._run_task_body_impl(
                self, work_tile, None, **task_context_kwargs(context)
            )
            if cutlass.const_expr(self.cfg.use_attention_sinks):
                prims.barrier_cta_sync(12, thread_count=16 * 32)
            work_tile = self.work_queue._get_consumer_var_from_ts("work_tile")
            self.dummy = cutlass.Boolean(True)

            while work_tile.is_valid_tile and self._should_skip_work_tile(work_tile):
                self.run_packed_skip_iteration(work_tile, context)
                work_tile = self.work_queue._get_consumer_var_from_ts("work_tile")
                self.dummy = cutlass.Boolean(True)

        self._run_post_work_loop_entries(work_tile, **task_context_kwargs(context))
        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource.pipeline_config is not None
                and resource is not self.work_queue
                and not self._is_fork_secondary(resource)
            ):
                pipeline_config = resource.pipeline_config
                assert pipeline_config is not None
                if cutlass.const_expr(pipeline_config.advance_on_acquire):
                    self._thread_advance_on_acquire_state(resource)
                self._producer_tail(resource)
        if cutlass.const_expr(
            self.work_queue in self.dst_resources
            and self.work_queue.pipeline_config is not None
        ):
            self.work_queue.producer_tail()
        self.dummy = cutlass.Boolean(True)

    def cache_visible_kv_interval(
        self, tile_coord: cute.Coord, seq_len_kv: cutlass.Int32
    ) -> None:
        """Cache the keys every row of the work tile sees, for softmax.

        Softmax compares each K tile against this interval instead of
        rebuilding the CTA's causal/window/tail bounds per tile.
        """
        cfg = self.cfg
        if cfg.uses_compact_split_grid:
            q_group_idx = cutlass.Int32(cute.arch.block_idx()[2]) % cfg.num_q_ctas
        elif cfg.has_single_q_cta:
            q_group_idx = cutlass.Int32(0)
        elif cfg.use_split_kv:
            q_group_idx = cutlass.Int32(tile_coord[0]) // cutlass.Int32(cfg.splits_kv)
        else:
            q_group_idx = cutlass.Int32(tile_coord[0])
        seq_len_q = (
            cutlass.Int32(cfg.max_seq_len_q)
            if self.seq_len_q is None
            else cutlass.Int32(self.seq_len_q)
        )
        self.kv_visible_begin, self.kv_visible_end = cta_visible_kv_interval(
            cfg,
            seq_len_kv,
            seq_len_q,
            q_group_token_base(cfg, q_group_idx),
        )

    def get_domain(self, tile_coord: cute.Coord) -> cutlass.Int32 | int:
        """Return this task's loop domain for one static or persistent tile.

        ``domain_scale`` multiplies the shared MMA-loop domain for tasks that
        run several schedule iterations per loop step. A static fallback
        returns the schedule's own bound, which is already scaled.
        """
        domain = self.shared_loop_domain(tile_coord)
        if self.domain_scale == 1 or domain is self.domain:
            return domain
        return domain * cutlass.Int32(self.domain_scale)

    def init_variables(self, context: cute.Pointer | None = None) -> None:
        """Initialize per-task thread and TMEM cached values."""
        super().init_variables(context)
        # Cache thread identity inside the 4-warp task group for TMEM and SMEM
        # resource operations.
        tidx, _, _ = cute.arch.thread_idx()
        warp_grp_start = cutlass.Int32((self.warp_idx // 4) * 4 * 32)
        self.warp_grp_thread_idx = tidx - warp_grp_start
        self.local_warp_idx = self.warp_grp_thread_idx >> cutlass.Int32(5)
        self.lane_idx = self.warp_grp_thread_idx & cutlass.Int32(0x1F)

        if context is not None and context.tmem_ptr_i32 is not None:
            # tcgen05_alloc provides the CTA's TMEM base through shared
            # context. Broadcast it so all lanes issue TMEM operations from the
            # same base column.
            loaded = cutlass.Int32(context.tmem_ptr_i32.load())
            self.tmem_base_offset = warp_broadcast_i32(loaded, 0)

    @cute.jit
    def make_task_cache(self):
        """Return cached task-local values passed through StageInfo."""
        # The cache is threaded through StageInfo so resource methods can read
        # task-local constants without recomputing them or adding more
        # resource variables to every schedule edge.
        values = (
            self.tmem_base_offset,
            self.warp_grp_thread_idx,
            self.local_warp_idx,
            self.lane_idx,
            self.seq_len_kv,
            self.kv_request_begin,
            self.kv_page_idx_ub,
            self.kv_raw_tile_base,
            self.kv_valid_tile_end,
            self.kv_window_start,
        )

        if cutlass.const_expr(self.cfg.runtime_split_kv):
            count = (
                cutlass.Int32(self.cfg.splits_kv)
                if self.requested_splits is None
                else self.requested_splits
            )
            values = (*values, count)
        if cutlass.const_expr(self.caches_visible_kv_interval):
            # Slot task_cache_visible_kv_interval_index(cfg), then the
            # sliding window's begin.
            values = (*values, self.kv_visible_end)
            if cutlass.const_expr(self.cfg.use_sliding_window_causal):
                values = (*values, self.kv_visible_begin)
        return values

    @cute.jit
    def refresh_packed_q_bounds(self, work_tile: Any) -> None:
        """Attach one active CLC tile's packed Q bounds to its data resources."""
        assert self.cfg is not None
        assert isinstance(self.work_queue, PackedDecodeWorkQueue)
        _, _, b_idx = work_tile.tile_idx
        q_token_offset, seq_len_q = q_seq_bounds(
            self.cfg,
            self.work_queue.cu_seqlens_q,
            cutlass.Int32(b_idx),
        )
        if cutlass.const_expr(self.seq_len_q is not None):
            self.seq_len_q = seq_len_q
        for resource, updates_q_token_offset in self.q_bound_resources:
            if cutlass.const_expr(updates_q_token_offset):
                resource.q_token_offset = q_token_offset
            resource.seq_len_q = seq_len_q

    @cute.jit
    def run_packed_skip_iteration(
        self,
        work_tile: Any,
        context: ResourceContext | None = None,
    ) -> None:
        """Advance one inactive tile through WorkQueue bookkeeping only."""
        # Packed schedules place every data-path entry inside ``skippable()``;
        # only the WorkQueue wait/advance/release tail remains outside it. Use
        # a unit domain solely to populate that tail's StageInfo. In particular,
        # do not call get_domain(), which would read per-batch KV metadata for a
        # tile whose Q sequence is empty or overlaunched.
        bookkeeping_domain = cutlass.Int32(1)
        for is_skippable_head, head_entries in self._head_exec_groups:
            if cutlass.const_expr(not is_skippable_head):
                self._run_head_entry_group(
                    head_entries,
                    work_tile,
                    bookkeeping_domain,
                    **task_context_kwargs(context),
                )
        for is_skippable_tail, tail_entries in self._tail_exec_groups:
            if cutlass.const_expr(not is_skippable_tail):
                self._run_tail_entry_group(
                    tail_entries,
                    work_tile,
                    bookkeeping_domain,
                    **task_context_kwargs(context),
                )

    def shared_loop_domain(self, tile_coord: cute.Coord) -> cutlass.Int32 | int:
        """Return the MMA-loop domain plus bias, or the static schedule bound."""
        if self.cfg is None:
            return self.domain
        if cutlass.const_expr(not self.cfg.fixed_seqlen and self.seq_len_q is None):
            # Padding and scheduler tasks do not consume request dimensions.
            return self.domain

        # Sparse rows have a runtime-dependent number of prepared KV routes
        # even when sequence lengths are static. Load their compact header
        # before the fixed-dense early return below. Persistent workers pass
        # the logical WorkQueue tile here, so static and CLC schedules share
        # the same (q_group, head, batch) mapping.
        if self.cfg.use_block_sparse:
            # Validation-only TaskManagers intentionally omit prepared GMEM
            # pointers and retain their configured static graph domain.
            row_route_offsets = self.sparse_row_route_offsets
            row_route_counts = self.sparse_row_route_counts
            if row_route_offsets is None or row_route_counts is None:
                return self.domain
            if self.num_heads_kv is None:
                raise ValueError(
                    "num_heads_kv is required to resolve block-sparse rows"
                )

            q_group_idx = cutlass.Int32(tile_coord[0])
            h_idx = cutlass.Int32(tile_coord[1])
            b_idx = cutlass.Int32(tile_coord[2])
            if self.sparse_row_route_begin is not None:
                # The static kernel prologue already loaded this tile's header.
                row_route_begin = self.sparse_row_route_begin
                route_count = self.sparse_route_count
            else:
                row_address = prepared_sparse_row_address(
                    self.cfg, q_group_idx, h_idx, b_idx, self.num_heads_kv
                )
                row_route_begin, route_count = load_prepared_sparse_row_warp(
                    row_route_offsets,
                    row_route_counts,
                    row_address,
                    self.lane_idx,
                )

            # Sparse route-span accessors share two underlying cache words
            # with paged KV. Clear dense/paged-only coordinates on every
            # logical tile because persistent tasks reuse the same task object.
            if self.seqlens_kv is None:
                self.seq_len_kv = self.max_seq_len_kv
            else:
                self.seq_len_kv = cutlass.Int32(self.seqlens_kv[b_idx])
            self.kv_request_begin = row_route_begin
            self.kv_page_idx_ub = route_count
            self.kv_raw_tile_base = cutlass.Int32(0)
            self.kv_valid_tile_end = route_count
            self.kv_window_start = cutlass.Int32(0)
            if self.caches_visible_kv_interval:
                # Sparse routes carry their own masks; no tile skips them.
                self.kv_visible_begin = cutlass.Int32(0)
                self.kv_visible_end = cutlass.Int32(0)

            loop_domain = loop_domain_after_head(
                route_count,
                num_insts_kv=self.cfg.num_insts_kv,
            )
            return loop_domain + cutlass.Int32(self.domain_bias)

        # Resolve the sequence length for this work tile. Static-seqlen kernels
        # can use the configured max length; variable-seqlen kernels read the
        # batch-specific length from GMEM.
        b_idx = cutlass.Int32(tile_coord[2])
        if self.seqlens_kv is None:
            seq_len_kv = cutlass.Int32(self.max_seq_len_kv)
        else:
            metadata_row = cutlass.Int64(b_idx)
            if cutlass.const_expr(
                self.cfg.uses_q_token_kv_block_sparse_page_route
                and self.cfg.use_persistent_scheduler
                and not self.cfg.shares_sparse_pattern
                and not isinstance(self.seqlens_kv, DirectSparseMetadataView)
            ):
                metadata_row = metadata_row * cutlass.Int64(
                    self.num_heads_kv
                ) + cutlass.Int64(tile_coord[1])
            seq_len_kv = cutlass.Int32(self.seqlens_kv[metadata_row])
        self.seq_len_kv = seq_len_kv
        if self.caches_visible_kv_interval:
            self.cache_visible_kv_interval(tile_coord, seq_len_kv)
        if cutlass.const_expr(self.cfg.use_paged_kv):
            # Native paged KV uses a fixed-stride dense page table. The row's
            # live extent therefore comes from seq_lens; padded entries beyond
            # this bound must never be staged.
            self.kv_page_idx_ub = runtime_last_valid_page_idx(self.cfg, seq_len_kv)
            if cutlass.const_expr(self.block_table_capacity is not None):
                self.kv_page_idx_ub = cute.math.min(
                    self.kv_page_idx_ub,
                    cutlass.Int32(self.block_table_capacity) - cutlass.Int32(1),
                )
        if (
            self.seqlens_kv is None
            and not self.cfg.use_split_kv
            and self.cfg.fixed_seqlen
            and not self.cfg.uses_runtime_q_kv_union
        ):
            return self.domain
        tile_size_kv = cutlass.Int32(self.cfg.tile_size_kv)

        # Q-independent full-K nonsplit decode has no leading window skip.
        # Resolve its runtime loop span directly and avoid repeating the general
        # causal/window/split coordinate construction in every task warp.
        if cutlass.const_expr(
            not self.cfg.use_split_kv
            and self.cfg.fixed_seqlen
            and not self.cfg.uses_runtime_q_kv_union
            and not self.cfg.use_sliding_window_causal
        ):
            total_kv_tiles = (
                seq_len_kv + tile_size_kv - cutlass.Int32(1)
            ) // tile_size_kv
            self.kv_window_start = cutlass.Int32(0)
            self.kv_valid_tile_end = total_kv_tiles
            self.kv_raw_tile_base = cutlass.Int32(0)
            loop_domain = loop_domain_after_head(
                total_kv_tiles,
                num_insts_kv=self.cfg.num_insts_kv,
            )
            return loop_domain + cutlass.Int32(self.domain_bias)

        # Work-tile coordinates retain the configured capacity stride.
        q_group_cta_idx = cutlass.Int32(tile_coord[0])
        q_group_idx = q_group_cta_idx
        if self.cfg.uses_compact_split_grid:
            q_group_idx = cutlass.Int32(cute.arch.block_idx()[2]) % self.cfg.num_q_ctas
        elif self.cfg.use_split_kv:
            q_group_idx = q_group_cta_idx // cutlass.Int32(self.cfg.splits_kv)
        q_token_base = q_group_token_base(self.cfg, q_group_idx)
        seq_len_q = (
            cutlass.Int32(self.cfg.max_seq_len_q)
            if self.seq_len_q is None
            else cutlass.Int32(self.seq_len_q)
        )
        self.kv_window_start = sliding_window_start_idx(
            self.cfg,
            seq_len_kv,
            seq_len_q,
            q_token_base,
        )
        skipped_tiles = self.kv_window_start // tile_size_kv
        total_kv_tiles = runtime_total_kv_tiles(
            self.cfg,
            seq_len_kv,
            seq_len_q,
            q_token_base,
        )
        self.kv_valid_tile_end = skipped_tiles + total_kv_tiles

        # Split-KV groups CTAs by K/V split. The loop domain is rounded so
        # each CTA in the group executes the same number of instruction pairs.
        if self.cfg.use_split_kv:
            # The domain and tile consumers share the instruction-aligned
            # partition. Runtime launches carry its host-prepared reciprocal.
            total_kv_tiles = runtime_configured_local_kv_tiles(
                self.cfg,
                seq_len_kv,
                seq_len_q,
                q_token_base,
                requested_splits=self.requested_splits,
            )
            split_idx = cutlass.Int32(cute.arch.block_idx()[0])
            if not self.cfg.uses_compact_split_grid:
                split_idx = cutlass.Int32(tile_coord[0]) % cutlass.Int32(
                    self.cfg.splits_kv
                )
            self.kv_raw_tile_base = skipped_tiles + split_idx * total_kv_tiles
        else:
            self.kv_raw_tile_base = skipped_tiles
        loop_domain = loop_domain_after_head(
            total_kv_tiles,
            num_insts_kv=self.cfg.num_insts_kv,
        )
        # All tasks share the MMA-loop domain; tail-only tasks add a bias.
        return loop_domain + cutlass.Int32(self.domain_bias)


@dataclass(kw_only=True)
class PackedDecodeWorkQueue(WorkQueue):
    """CLC work queue that drops Q padding and inactive KV requests."""

    # Static configuration
    cfg: cutlass.Constexpr[FmhaDecodeConfig] = field(init=False, default=None)

    # Per-task state
    cu_seqlens_q: Any = field(init=False, default=None)
    num_heads_kv: Any = field(init=False, default=None)
    seqlens_kv: Any = field(init=False, default=None)

    def __init__(
        self,
        cfg: FmhaDecodeConfig,
        cu_seqlens_q: cute.Pointer | UniformQueryOffsets,
        seqlens_kv: cute.Pointer | None = None,
        num_heads_kv: cutlass.Int32 | None = None,
        **kwargs: Any,
    ) -> None:
        """Attach request metadata to the shared persistent work queue."""
        super().__init__(**kwargs)
        self.cfg = cfg
        self.cu_seqlens_q = cu_seqlens_q
        self.seqlens_kv = seqlens_kv
        self.num_heads_kv = num_heads_kv

    @cute.jit
    def _work_tile_info_from_clc_response(self, response_ptr: Any) -> Any:
        """Decode a CLC response, then apply the longest-first route order."""
        return self.route_ordered(
            WorkQueue._work_tile_info_from_clc_response(self, response_ptr)
        )

    @cute.jit
    def initial_work_tile_info(self) -> Any:
        """Return this CTA's first work tile in longest-first route order."""
        return self.route_ordered(self._make_initial_work_tile())

    @cute.jit
    def route_ordered(self, work_tile: Any) -> Any:
        """Map CLC route index i to ``route_order[i]`` (longest first).

        The metadata grid stores the order right after its per-route lengths;
        TaskManager reads it after the attention grid's dependency wait. A
        failed CLC cancel (drained queue) leaves the decoded coordinates
        undefined, so invalid tiles read entry 0 instead (no branch).
        """
        if cutlass.const_expr(self.cfg.uses_longest_first_route_order):
            q_group_idx, h_idx, route_idx = work_tile.tile_idx
            valid = cutlass.Int32(work_tile.is_valid_tile)
            routes = cutlass.Int32(cute.arch.grid_dim()[2])
            route_idx = cutlass.Int32(
                self.seqlens_kv[routes + cutlass.Int32(route_idx) * valid]
            )
            return type(work_tile)(
                (q_group_idx, h_idx, route_idx), work_tile.is_valid_tile
            )
        return work_tile

    @cute.jit
    def skip_work_tile_if(self, work_tile: Any) -> cutlass.Boolean:
        """Skip empty KV requests and Q groups outside their live sequence."""
        q_group_cta_idx, h_idx, b_idx = work_tile.tile_idx
        empty_kv = cutlass.Boolean(False)
        if cutlass.const_expr(self.seqlens_kv is not None):
            lengths = self.seqlens_kv
            if cutlass.const_expr(
                self.cfg.use_q_token_kv_block_sparse_route
                and not self.cfg.shares_sparse_pattern
            ):
                if cutlass.const_expr(isinstance(lengths, DirectSparseMetadataView)):
                    lengths = lengths.with_head(h_idx)
                else:
                    lengths = HeadIndexedMetadataView(lengths, self.num_heads_kv, h_idx)
            empty_kv = cutlass.Int32(lengths[b_idx]) <= cutlass.Int32(0)
        q_group_idx = cutlass.Int32(q_group_cta_idx)
        if cutlass.const_expr(self.cfg.uses_compact_split_grid):
            q_group_idx = cutlass.Int32(cute.arch.block_idx()[2]) % self.cfg.num_q_ctas
        elif cutlass.const_expr(self.cfg.use_split_kv):
            q_group_idx = q_group_idx // cutlass.Int32(self.cfg.splits_kv)
        _, seq_len_q = q_seq_bounds(
            self.cfg,
            self.cu_seqlens_q,
            cutlass.Int32(b_idx),
        )
        return empty_kv or q_group_token_base(self.cfg, q_group_idx) >= seq_len_q


def can_hold_native_page_window(
    cfg: FmhaDecodeConfig,
    smem_page_offsets: MemoryResource | None,
) -> bool:
    """Return whether one native page-ID stage covers every runtime KV range.

    A runtime launch selects the CTA-relative window only when its host KV
    bound and count guarantee the entire instruction-aligned span fits.
    Longer spans retain the ordinary staged pipeline.
    """
    if (
        smem_page_offsets is None
        or not smem_page_offsets.use_native_paged_kv
        or cfg.use_sliding_window_causal
        or (
            not cfg.fixed_seqlen
            and not (
                cfg.runtime_split_kv
                and cfg.runtime_page_window
                and cfg.runtime_page_window_pages > 32
            )
        )
    ):
        return False
    if cfg.runtime_split_kv:
        return cfg.runtime_page_window
    pages_per_tile = cfg.tile_size_kv // cfg.num_tokens_per_page
    if cfg.uses_scattered_page_route:
        # Each sparse route gets a page-offset stage sized for its complete
        # instruction-aligned local span. Holding that stage lets
        # K and V reuse the same dense-row locators, including -1 padding for
        # a short or odd tail. This applies to direct and persistent kernels as
        # well as split-KV: each work tile stages its own slice of that row.
        return smem_page_offsets.holds_encoded_locator_window
    if not cfg.use_split_kv:
        return False
    # Runtime ragged lengths can reduce the split-local span in
    # ``num_insts_kv`` increments and therefore change every split rank's
    # aligned starting page. Hold one stage only when every possible span
    # both fits and evenly partitions a 32-ID window; otherwise a shorter row
    # could straddle the next window even when the static maximum does not.
    for local_tiles in range(
        cfg.num_insts_kv,
        cfg.static_local_kv_tiles + 1,
        cfg.num_insts_kv,
    ):
        window_pages = local_tiles * pages_per_tile
        if window_pages <= 0 or window_pages > 32 or 32 % window_pages != 0:
            return False
    return True


def decode_work_tile_schedule(
    cfg: FmhaDecodeConfig,
    work_queue: WorkQueue | None,
    body: Callable[[], None],
    non_skippable_head: Callable[[], None] | None = None,
) -> None:
    """Trace data work separately from persistent queue bookkeeping."""
    if cfg.use_persistent_scheduler:
        assert work_queue is not None
        # Schedule tracing passes resource proxies, so use the static config.
        with work_tile_loop(
            work_queue,
            skip_if=PackedDecodeWorkQueue.skip_work_tile_if,
        ) as work_tiles:
            if non_skippable_head is not None:
                non_skippable_head()
            with work_tiles.skippable():
                body()
            # WorkQueue operations stay inside the persistent loop but outside
            # the skippable data region. DecodeGenTask therefore advances the
            # queue exactly once for both active and inactive packed tiles.
            work_queue_tail(work_queue)
        return

    # Non-persistent kernels execute one unguarded data region.
    body()
    work_queue_tail(work_queue)


@cute.jit
def load_prepared_sparse_row_warp(
    row_route_offsets: cute.Pointer,
    row_route_counts: cute.Pointer,
    row_address: cutlass.Int32,
    lane_idx: cutlass.Int32,
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Load one prepared row header once per warp and broadcast it.

    DecodeGenTask.get_domain is a regular Task override, so staged control flow
    lives in this JIT helper rather than in the Python method itself.
    """

    loaded_row_route_begin = cutlass.Int32(0)
    loaded_route_count = cutlass.Int32(0)
    if lane_idx == cutlass.Int32(0):
        loaded_row_route_begin = cutlass.Int32(row_route_offsets[row_address])
        loaded_route_count = cutlass.Int32(row_route_counts[row_address])
    row_route_begin = warp_broadcast_i32(loaded_row_route_begin, 0)
    route_count = assume_nonnegative_i32(warp_broadcast_i32(loaded_route_count, 0))
    return row_route_begin, route_count


def loop_domain_after_head(
    total_kv_tiles: cutlass.Int32,
    *,
    num_insts_kv: int,
) -> cutlass.Int32:
    """Return LOOP iterations after HEAD reserves one KV tile per instance.

    Dense tiles and block-sparse routes share this recurrence; only the tile
    count's source differs.
    """

    remaining = total_kv_tiles - cutlass.Int32(num_insts_kv)
    remaining = cute.math.max(remaining, cutlass.Int32(0))
    insts = cutlass.Int32(num_insts_kv)
    return (remaining + insts - cutlass.Int32(1)) // insts


@cute.jit
def prefetch_prepared_sparse_row(
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    row_route_offsets: cute.Pointer,
    row_route_counts: cute.Pointer,
    q_group_idx: cutlass.Int32,
    h_idx: cutlass.Int32,
    b_idx: cutlass.Int32,
    num_heads_kv: cutlass.Int32,
) -> tuple[cutlass.Int32, cutlass.Int32]:
    """Load one static tile's prepared row header before its tasks start.

    Every thread loads the same two words, so each warp issues one request
    and the global-memory latency overlaps the CTA prologue (TMEM allocation
    and barrier setup) instead of stalling every task at its first step.
    """

    row_address = prepared_sparse_row_address(
        cfg, q_group_idx, h_idx, b_idx, num_heads_kv
    )
    row_route_begin = cutlass.Int32(row_route_offsets[row_address])
    route_count = assume_nonnegative_i32(cutlass.Int32(row_route_counts[row_address]))
    return row_route_begin, route_count


@cute.jit
def prepared_sparse_row_address(
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    q_group_idx: cutlass.Int32,
    h_idx: cutlass.Int32,
    b_idx: cutlass.Int32,
    num_heads_kv: cutlass.Int32,
) -> cutlass.Int32:
    """Map a (q_group, head, batch) tile to its prepared row header index."""

    q_token_base = q_group_token_base(cfg, q_group_idx)
    q_block = q_token_base // cutlass.Int32(cfg.q_block_size)
    num_q_blocks = (cfg.max_seq_len_q + cfg.q_block_size - 1) // cfg.q_block_size
    if cutlass.const_expr(cfg.shares_sparse_pattern):
        return b_idx * cutlass.Int32(num_q_blocks) + q_block
    return (b_idx * num_heads_kv + h_idx) * cutlass.Int32(num_q_blocks) + q_block


def schedule_with_optional_resources(
    fn: Callable[..., None],
) -> Callable[..., Schedule]:
    """Capture one schedule while omitting absent resources from its graph.

    CUTLASS ``@schedule`` intentionally accepts only concrete resources.  Some
    FMHA task variants have orthogonal optional resources, such as block-sparse
    metadata and a persistent work queue.  Filter absent slots before tracing,
    then restore their named positions for the schedule body.  This adapter is
    entirely host-side: each captured schedule contains only present resources
    and the ``None`` branches disappear while tracing.
    """

    @functools.wraps(fn)
    def traced(*resource_slots: object) -> Schedule:
        present_slots = tuple(slot is not None for slot in resource_slots)
        resources = tuple(
            slot
            for slot, is_present in zip(resource_slots, present_slots, strict=True)
            if is_present
        )

        @functools.wraps(fn)
        def restore_slots(*resource_proxies: object) -> None:
            proxy_iter = iter(resource_proxies)
            restored_slots = tuple(
                next(proxy_iter) if is_present else None for is_present in present_slots
            )
            fn(*restored_slots)

        return schedule(restore_slots)(*resources)

    return traced


def work_queue_tail(work_queue: WorkQueue | None, work_tile=None):
    """Finish one persistent-scheduler tile after task-local work drains."""
    _ = work_tile
    if work_queue is None:
        return None
    # ConsWait: wait for the scheduler token for this task.
    work_queue.wait()
    # ConsWork: advance the task-local tile cursor.
    work_queue.get_and_advance_work_tile()
    # ConsRelease: allow the scheduler to hand out the next tile.
    work_queue.release()
    return None
