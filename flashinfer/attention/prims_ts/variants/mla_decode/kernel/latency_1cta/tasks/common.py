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

"""Task base classes, work queues and schedule helpers shared by the latency 1CTA MLA decode kernel tasks."""

import cutlass.cute as cute
import cutlass
from cutlass import Int32
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.task import Task
from ......device.task import task_context_kwargs
from ...device.constants import (
    WARPGROUP_WARPS,
    WARP_LANES,
    WARP_LANE_MASK,
    WARP_LANE_SHIFT,
)
from ...device.tile import (
    cta_rows_have_distinct_kv_prefix,
    cta_visible_kv_end,
    runtime_base_seq_len_kv,
    runtime_local_kv_tiles,
    runtime_seq_len_kv_from_task_cache,
    runtime_split_count,
)


class MlaDecodeTask(Task):
    """MLA decode task with cached thread state and Q-aware KV domain logic."""

    def __init__(self, **kwargs):
        """Initialize MLA-specific task parameters and cached thread indices."""
        self.cfg = kwargs.pop("cfg", None)
        self.runtime_split_kv = kwargs.pop("runtime_split_kv", None)
        self.seqlens_kv = kwargs.pop("seqlens_kv", None)
        self.cu_seqlens_q = kwargs.pop("cu_seqlens_q", None)
        self.cp_rank = kwargs.pop("cp_rank", 0)
        self.domain_bias = kwargs.pop("domain_bias", 0)
        super().__init__(**kwargs)
        self.tmem_base_offset = Int32(0)
        self.warp_grp_thread_idx = Int32(0)
        self.local_warp_idx = Int32(0)
        self.lane_idx = Int32(0)
        self.seq_len_kv = Int32(0)
        self.local_kv_tiles = Int32(1)
        if self.caches_visible_kv_end:
            # An empty prefix masks every tile, which is always exact.
            self.kv_visible_end = Int32(0)
        self.runtime_body_active = None

    @property
    def caches_visible_kv_end(self) -> bool:
        """Return whether this task caches the keys every row of a tile sees.

        Only tiles whose rows see different K prefixes need it.
        """
        return self.cfg is not None and cta_rows_have_distinct_kv_prefix(self.cfg)

    def get_domain(self, tile_coord):
        """Return the runtime loop domain for the current 1CTA work tile."""
        if self.cfg is None:
            return self.domain
        if cutlass.const_expr(self.seqlens_kv is None and self.cfg.seq_len_kv is None):
            # Padding and scheduler tasks have no K/V metadata or mainloop.
            return self.domain

        # Persistent 1CTA coordinates combine batch and head tile in z;
        # non-persistent grids keep z as batch. cache_seqs remains batch-owned.
        if isinstance(tile_coord[2], int):
            if self.cfg.use_persistent_scheduler == 1:
                batch_idx = tile_coord[2] // self.cfg.num_ctas_for_all_heads
            else:
                batch_idx = tile_coord[2]
        else:
            batch_idx = Int32(tile_coord[2])
            if self.cfg.use_persistent_scheduler == 1:
                batch_idx = batch_idx // Int32(self.cfg.num_ctas_for_all_heads)

        # Task-domain ownership must match K/V, softmax, and reduction: dense
        # uses the full runtime K length, while causal uses the largest
        # logical-Q-visible length in this physical flat-query tile. Using the
        # raw batch length for causal can move an all-masked tile into tail and
        # replace a real loop accumulator at a tile boundary.
        cta_idx_q = tile_coord[0]
        if cutlass.const_expr(
            self.cfg.fixed_seqlen
            and self.cfg.use_persistent_scheduler != 1
            and self.cfg.num_ctas_per_seq_q == 1
        ):
            cta_idx_q = Int32(0)
        elif cutlass.const_expr(self.cfg.use_persistent_scheduler != 1):
            grid_splits = Int32(self.cfg.num_ctas_per_seq_kv)
            if cutlass.const_expr(
                self.runtime_split_kv is not None
                and self.cfg.use_cluster_reduction != 1
            ):
                grid_splits = runtime_split_count(self.runtime_split_kv)
            # Heads are already flattened into Q; grid-X packs Q and KV split.
            cta_idx_q = tile_coord[0] // grid_splits
            if cutlass.const_expr(self.cfg.fixed_seqlen):
                cta_idx_q = cta_idx_q % Int32(self.cfg.num_ctas_per_seq_q)
        # Cache the raw batch length once per task/work tile. Resource work
        # methods derive their CTA-visible length from this task cache instead
        # of independently reloading cache_seqs throughout the hot loop.
        self.seq_len_kv = runtime_base_seq_len_kv(
            self.cfg,
            self.seqlens_kv,
            batch_idx,
        )
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            self.cfg,
            self.make_task_cache(),
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
            cp_rank=self.cp_rank,
        )
        # Softmax masks only K tiles reaching past the keys every row sees.
        if self.caches_visible_kv_end:
            self.kv_visible_end = cta_visible_kv_end(
                self.cfg,
                self.seqlens_kv,
                batch_idx,
                cta_idx_q,
                self.cu_seqlens_q,
                self.cp_rank,
            )
        if self.cfg.fixed_sparse_kv_tiles > 0:
            return Int32(self.domain_bias)
        total_kv_tiles = runtime_local_kv_tiles(
            self.cfg, seq_len_kv, requested_splits=self.runtime_split_kv
        )
        self.local_kv_tiles = total_kv_tiles
        remaining_kv_tiles = cute.math.max(
            total_kv_tiles - Int32(self.cfg.num_insts_kv), Int32(0)
        )
        num_insts_kv = Int32(self.cfg.num_insts_kv)
        loop_domain = (remaining_kv_tiles + num_insts_kv - Int32(1)) // num_insts_kv
        return loop_domain + Int32(self.domain_bias)

    # Deliberately not @cute.jit, like make_task_cache and get_domain: these
    # mutate Python-held task state, which a jitted method under FrontendNext
    # commits later than the jitted readers that follow. run_body above needs
    # @cute.jit for its runtime branch. test_deepseek_v41_defaults_cpu checks
    # the three stay undecorated.
    def init_variables(self, context=None):
        """Initialize per-task thread cache and TMEM base state."""
        super().init_variables(context)
        tidx, _, _ = cute.arch.thread_idx()
        warp_grp_start = Int32(
            (self.warp_idx // WARPGROUP_WARPS) * WARPGROUP_WARPS * WARP_LANES
        )
        self.warp_grp_thread_idx = tidx - warp_grp_start
        self.local_warp_idx = self.warp_grp_thread_idx >> Int32(WARP_LANE_SHIFT)
        self.lane_idx = self.warp_grp_thread_idx & Int32(WARP_LANE_MASK)
        if context is not None and context.tmem_ptr_i32 is not None:
            loaded = self.load_allocated_tmem_base(context.tmem_ptr_i32)
            self.tmem_base_offset = cute.arch.make_warp_uniform(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=loaded,
                    offset=0,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
            )

    @cute.jit
    def load_allocated_tmem_base(self, pointer):
        # TaskManager initializes every task before dispatch. Sparse loaders
        # may already be running while the MMA warp allocates TMEM, so only
        # the warps participating in the readiness barrier may read its result.
        if cutlass.const_expr(self.cfg is not None and self.cfg.sparse_direct):
            warp = cute.arch.warp_idx()
            base = Int32(0)
            if warp < Int32(
                self.cfg.correction_warp_idx + self.cfg.correction_num_warps
            ) or warp == Int32(self.cfg.mma_warp_idx):
                base = Int32(pointer.load(is_volatile=True))
            return base
        else:
            return Int32(pointer.load())

    def make_task_cache(self):
        """Snapshot the compact task state at the tracing call site."""
        # get_domain updates seq_len_kv in Python; under FrontendNext a jitted
        # getter can read the old staged slot before that mutation is committed.
        return (
            self.tmem_base_offset,
            self.warp_grp_thread_idx,
            self.local_warp_idx,
            self.lane_idx,
            self.seq_len_kv,
            self.local_kv_tiles,
            self.kv_visible_end if self.caches_visible_kv_end else Int32(0),
            Int32(0),
        )

    @cute.jit
    def run_body(self, context=None):
        """Guard data work only, after TaskManager initializes every task."""
        if cutlass.const_expr(self.runtime_body_active is not None):
            if self.runtime_body_active:
                super().run_body(**task_context_kwargs(context))
        else:
            super().run_body(**task_context_kwargs(context))
