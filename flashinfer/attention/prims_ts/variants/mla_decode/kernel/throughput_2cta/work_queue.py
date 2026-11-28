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

"""Persistent work queue and per-tile work info of the 2CTA MLA kernel."""

import cutlass.cute as cute
import cutlass
from cutlass import Boolean, Int32
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    WorkQueue,
    consumer_work,
)
from dataclasses import dataclass
from ..device.mask import MaskType, mask_visible_k_length, rows_have_distinct_kv_prefix
from ..device.query import (
    flat_query_row_state,
    query_batch_bounds,
    runtime_flat_query_tile_has_rows,
)
from ..device.tile_scheduler import (
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
    divmod_constexpr_power_of_two_or_fdd,
)
from .work_partition import runtime_split_kv_cap, runtime_split_tile_range
from typing import Any
from ..device.context_parallel import local_kv_length


class MlaTsWorkTileInfo:
    """MLA work tile with scalar fields for TS persistent loop carry.

    Besides the scheduler coordinate, cache the per-tile K-domain values that
    hot resource paths need. Each persistent warp loop computes the K domain
    once and passes the derived indices to the page-offset, TMA, MMA, and
    softmax bodies. ``k_visible_end`` is the rank-local key prefix every query
    row sees; it is carried only when rows see different prefixes
    (``rows_have_distinct_kv_prefix``) and is ``None`` otherwise.
    """

    @cute.jit
    def __init__(
        self,
        tile_idx,
        is_valid,
        k_len=0,
        k_tile_count=0,
        k_index_base=0,
        k_visible_end=None,
    ):
        """Initialize the staged tile coordinate and cached K-domain metadata."""
        cluster_idx, seq_q_idx, batch_idx, split_kv_idx = tile_idx
        self.cluster_idx = Int32(cluster_idx)
        self.seq_q_idx = Int32(seq_q_idx)
        self.batch_idx = Int32(batch_idx)
        self.split_kv_idx = Int32(split_kv_idx)
        self.is_valid = Boolean(is_valid)
        self.k_len = Int32(k_len)
        self.k_tile_count = Int32(k_tile_count)
        self.k_index_base = Int32(k_index_base)
        self.k_visible_end = None
        if cutlass.const_expr(k_visible_end is not None):
            self.k_visible_end = Int32(k_visible_end)

    def __extract_mlir_values__(self):
        """Extract scalar MLIR values for value-type lowering."""
        values = cutlass.extract_mlir_values(self.cluster_idx)
        values += cutlass.extract_mlir_values(self.seq_q_idx)
        values += cutlass.extract_mlir_values(self.batch_idx)
        values += cutlass.extract_mlir_values(self.split_kv_idx)
        values += cutlass.extract_mlir_values(self.is_valid)
        values += cutlass.extract_mlir_values(self.k_len)
        values += cutlass.extract_mlir_values(self.k_tile_count)
        values += cutlass.extract_mlir_values(self.k_index_base)
        if self.k_visible_end is not None:
            values += cutlass.extract_mlir_values(self.k_visible_end)
        return values

    def __new_from_mlir_values__(self, values):
        """Rebuild a tile info object from lowered scalar MLIR values."""
        return MlaTsWorkTileInfo(
            (
                cutlass.new_from_mlir_values(self.cluster_idx, [values[0]]),
                cutlass.new_from_mlir_values(self.seq_q_idx, [values[1]]),
                cutlass.new_from_mlir_values(self.batch_idx, [values[2]]),
                cutlass.new_from_mlir_values(self.split_kv_idx, [values[3]]),
            ),
            cutlass.new_from_mlir_values(self.is_valid, [values[4]]),
            cutlass.new_from_mlir_values(self.k_len, [values[5]]),
            cutlass.new_from_mlir_values(self.k_tile_count, [values[6]]),
            cutlass.new_from_mlir_values(self.k_index_base, [values[7]]),
            None
            if self.k_visible_end is None
            else cutlass.new_from_mlir_values(self.k_visible_end, [values[8]]),
        )

    @property
    @cute.jit
    def is_valid_tile(self):
        """Return whether this tile participates in the current launch."""
        return self.is_valid

    @property
    @cute.jit
    def tile_idx(self):
        """Return the scheduler tile coordinate tuple."""
        return (
            self.cluster_idx,
            self.seq_q_idx,
            self.batch_idx,
            self.split_kv_idx,
        )

    @cute.jit
    def update_from(self, other) -> None:
        """Replace this tile info with another tile info object."""
        cluster_idx, seq_q_idx, batch_idx, split_kv_idx = other.tile_idx
        self.cluster_idx = cluster_idx
        self.seq_q_idx = seq_q_idx
        self.batch_idx = batch_idx
        self.split_kv_idx = split_kv_idx
        self.is_valid = Boolean(other.is_valid_tile)
        self.k_len = Int32(other.k_len)
        self.k_tile_count = Int32(other.k_tile_count)
        self.k_index_base = Int32(other.k_index_base)
        if cutlass.const_expr(self.k_visible_end is not None):
            self.k_visible_end = Int32(other.k_visible_end)


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

    # Static configuration
    is_var_split_kv: cutlass.Constexpr[bool] = False
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    skip_work_tile: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    static_problem_shape_b: cutlass.Constexpr[int] = None
    static_problem_shape_s: cutlass.Constexpr[int] = None
    static_seq_len_k: cutlass.Constexpr = None
    static_split_kv: cutlass.Constexpr = None
    use_clc_dynamic: cutlass.Constexpr[bool] = False
    work_tile: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    # Runtime scalars
    logical_seq_len_q: Int32 = 1

    # Per-task state
    block_split_kvs: Any = None  # optional per-batch split caps
    cache_seqs: Any = None  # global lengths for k_tile_count
    cfg: Any = None  # MlaDecodeConfig for tile sizes
    cp_rank: Any = 0
    cu_seqlens_q: Any = None
    split_kv: Any = None  # maximum split slots in the launch/workspace
    tile_sched_params: Any = None

    def __init__(
        self,
        tile_sched_params,
        cache_seqs=None,
        cp_rank=0,
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
        self.cp_rank = cp_rank
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
                Int32(0) if self.caches_visible_k_end else None,
            ),
            docs="Current MLA persistent-scheduler work tile.",
        )
        self.skip_work_tile = TaskLocalVariable(
            dtype=Boolean,
            default=Boolean(False),
            docs="Whether the current work tile should skip skippable work.",
        )

    @property
    def caches_visible_k_end(self) -> bool:
        """Return whether work tiles carry the key prefix every row sees."""
        return self.cfg is not None and rows_have_distinct_kv_prefix(
            self.cfg.mask_type, self.cu_seqlens_q is not None, self.logical_seq_len_q
        )

    @consumer_work(returns=work_tile)
    @cute.jit
    def advance_tile(self, stage_info: StageInfo):
        """Advance CLC through its response pipeline; static is task-owned."""
        if cutlass.const_expr(self.use_clc_dynamic):
            # Decode through the public scheduler/config surfaces so the
            # task-local value keeps its MLA type and cached runtime K-domain
            # fields without depending on WorkQueue's private response helpers.
            assert self.tile_scheduler_config is not None
            assert self.tile_scheduler_config.response_ptr is not None
            assert self.tile_scheduler is not None
            assert self.pipeline_config is not None
            stage_response_ptr = self.tile_scheduler_config.response_ptr
            if cutlass.const_expr(self.pipeline_config.num_stages > 1):
                stage_response_ptr = stage_response_ptr + stage_info.stage_idx
            work_tile = self.tile_scheduler.work_tile_info_from_clc_response(
                stage_response_ptr
            )
            return self.work_tile_from_clc_tile(work_tile)
        return stage_info.work_tile

    def create(self) -> None:
        """Create pipeline (if any) and the scheduler used by all TS tasks."""
        if cutlass.const_expr(self.use_clc_dynamic):
            WorkQueue.create(self)
            return
        # Call MemoryResource.create() (not WorkQueue.create) to avoid
        # the tile_scheduler_config check in the parent.
        MemoryResource.create(self)
        self.tile_scheduler = self.create_placeholder_tile_scheduler()

    def create_placeholder_tile_scheduler(self):
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

    def create_tile_scheduler(self):
        if cutlass.const_expr(self.use_clc_dynamic):
            return WorkQueue.create_tile_scheduler(self)
        return create_mla_static_tile_scheduler(
            self.tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=(work_tile, skip_work_tile))
    @cute.jit
    def init_work_tile(self, stage_info: StageInfo):
        """Seed captured schedules from the current custom MLA work tile."""
        return stage_info.work_tile, Boolean(False)

    @cute.jit
    def initial_work_tile_info(self):
        work_tile = self.tile_scheduler.initial_work_tile_info()
        if cutlass.const_expr(self.use_clc_dynamic):
            return self.work_tile_from_clc_tile(work_tile)
        return self.wrap_work_tile(work_tile)

    @cute.jit
    def k_tile_count_for_tile(self, tile_idx):
        """Return the dynamic K-loop bound required by stock Task."""

        return self.make_work_tile_info(tile_idx, Boolean(True)).k_tile_count

    @cute.jit
    def linear_idx_from_tile(self, tile_idx):
        params = self.tile_sched_params
        cluster_idx, s_idx, b_idx, split_kv_idx = tile_idx
        return (
            ((split_kv_idx * params.problem_shape_b + b_idx) * params.problem_shape_s)
            + s_idx
        ) * params.cluster_shape_mnk[0] + cluster_idx

    # Bind a scheduler tile to its live causal KV range and split; inactive Q gets zero
    # tiles.
    @cute.jit
    def make_work_tile_info(
        self,
        tile_idx,
        is_valid,
    ):
        cfg = self.cfg
        params = self.tile_sched_params
        _, s_idx, b_idx, split_kv_idx = tile_idx
        safe_b_idx = cute.math.min(b_idx, params.problem_shape_b - Int32(1))
        if cutlass.const_expr(self.static_seq_len_k is not None and cfg.cp_world == 1):
            K = Int32(self.static_seq_len_k)
        else:
            K = Int32(self.cache_seqs[safe_b_idx])
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
        if cutlass.const_expr(cfg.mask_type == MaskType.CAUSAL.value):
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
        K = local_kv_length(cfg, K, self.cp_rank)
        K = K if query_tile_has_rows else Int32(0)
        k_tile_total = (K + Int32(cfg.mma_qk_tiler[1] - 1)) // Int32(
            cfg.mma_qk_tiler[1]
        )
        k_visible_end = None
        if cutlass.const_expr(self.caches_visible_k_end):
            k_visible_end = self.visible_k_end(s_idx, safe_b_idx, sequence_k_len, q_len)
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
                split_divisor=params.split_kv_fdd
                if self.static_split_kv is None and not self.is_var_split_kv
                else None,
            )
        return MlaTsWorkTileInfo(
            tile_idx,
            is_valid,
            sequence_k_len,
            k_tile_count,
            k_index_base,
            k_visible_end,
        )

    @cute.jit
    def skip_work_tile_if(self, work_tile):
        """Skip zero-K CLC tiles while retaining WorkQueue bookkeeping."""

        return work_tile.k_tile_count <= Int32(0)

    @cute.jit
    def visible_k_end(self, s_idx, b_idx, sequence_k_len, q_len):
        """Return the rank-local keys ``[0, end)`` every row of a work tile sees.

        Bottom-right causal prefixes, and their rank-local counts under context
        parallelism, are nondecreasing in the flat query row, so the tile's
        first row has the shortest prefix. Computed once per work tile; softmax
        compares each K tile against it.
        """
        cfg = self.cfg
        _, _, first_q_idx, _, _ = flat_query_row_state(
            Int32(0),
            s_idx,
            cfg.mma_qk_tiler[0],
            self.logical_num_heads_q,
            self.logical_seq_len_q,
            self.cu_seqlens_q,
            b_idx,
        )
        return local_kv_length(
            cfg,
            mask_visible_k_length(cfg.mask_type, sequence_k_len, first_q_idx, q_len),
            self.cp_rank,
        )

    @cute.jit
    def work_tile_from_block_idx(self, block_idx):
        params = self.tile_sched_params
        s_idx, b_idx = divmod_constexpr_power_of_two_or_fdd(
            block_idx[1],
            self.static_problem_shape_b,
            params.problem_shape_b_fdd,
        )
        return self.make_work_tile_info(
            (block_idx[0], s_idx, b_idx, block_idx[2]),
            Boolean(True),
        )

    @cute.jit
    def work_tile_from_clc_tile(self, work_tile):
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
        return self.make_work_tile_info(
            (cluster_idx, s_idx, b_idx, split_kv_idx),
            work_tile.is_valid_tile,
        )

    @cute.jit
    def work_tile_from_linear_idx(self, current_work_linear_idx):
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
        return self.make_work_tile_info(
            (cluster_idx, s_idx, b_idx, split_kv_idx),
            current_work_linear_idx < num_blocks,
        )

    @cute.jit
    def wrap_work_tile(self, work_tile):
        return MlaTsWorkTileInfo(
            work_tile.tile_idx,
            work_tile.is_valid_tile,
            k_visible_end=Int32(0) if self.caches_visible_k_end else None,
        )
