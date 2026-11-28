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

"""TMEM softmax statistics passed from softmax to correction."""

import cutlass.cute as cute
import cutlass
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation, TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field
from ..config import FmhaConfig
from ..device.common import SoftmaxScalar, TmemAddr, TmemPtr, resolve_work_tile_coords
from .....device.lse_merge import merge_log2_lse
from .....device.placeholders import placeholder_tmem_ptr
from typing import Any, Optional


# ---------------------------------------------------------------------------
# TmemStatsResource -- TMEM correction statistics with AsyncAsync pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemStatsResource(MemoryResource):
    """Correction statistics with an AsyncAsync pipeline.

    Producer: Softmax writes old_max/row_max/row_sum stats. Consumer:
    Correction reads them for O rescaling. The staged SMEM ring keeps the
    payload disjoint from S/P TMEM columns, including while QK and correction
    overlap in a non-persistent single-instance schedule.
    """

    # Static configuration
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    peer_idx: Constexpr[int] = field(init=False, default=0)
    tmem_addr_cached: TmemAddr | None = field(init=False, default=None)
    tmem_ptr_vec_cached: TmemPtr | None = field(init=False, default=None)
    # Precomputed per-warp TMEM vec address (once, before persistent loop).
    tmem_vec_addr_cached: TmemAddr | None = field(init=False, default=None)
    tmem_vec_offset: Constexpr[int] = field(init=False, default=None)
    vec_new_max: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vec_old_max: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vec_row_sum: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    vec_scale: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    # Runtime scalars
    query_seq_len: Int32 | int = field(init=False, default=0)
    scale_softmax_log2: Float32 | None = field(init=False, default=None)

    # GMEM pointers and tensors
    attention_sinks: cute.Tensor | None = field(init=False, default=None)
    cum_seqlen_q: cute.Tensor | None = field(init=False, default=None)
    lse_output: cute.Tensor | None = field(init=False, default=None)
    output_scale: Float32 | cute.Tensor | None = field(init=False, default=None)

    # Allocations
    alloc: Constexpr[Optional[TmemAllocation]] = field(init=False, default=None)
    smem_alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        tmem_vec_offset: int,
        scale_softmax_log2: Float32 | None = None,
        output_scale: Float32 | cute.Tensor | None = None,
        lse_output: cute.Tensor | None = None,
        attention_sinks: cute.Tensor | None = None,
        cum_seqlen_q: cute.Tensor | None = None,
        query_seq_len: Int32 | int = 0,
        peer_idx: int = 0,
        **kwargs: Any,
    ) -> None:
        """Bind the correction-stat TMEM vector offset and allocation."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.tmem_vec_offset = tmem_vec_offset
        self.scale_softmax_log2 = scale_softmax_log2
        self.output_scale = output_scale
        self.lse_output = lse_output
        self.attention_sinks = attention_sinks
        self.cum_seqlen_q = cum_seqlen_q
        self.query_seq_len = query_seq_len
        self.peer_idx = peer_idx
        self.alloc = TmemAllocation(f"tmem_vec_{tmem_vec_offset}", cfg.tmem_stats_cols)
        self.smem_alloc = None
        if cfg.stats_via_smem:
            stats_rows = len(cfg.softmax0_warp_ids) * cute.arch.WARP_SIZE
            # Each row publishes two packed FP32 values. Loop records hold
            # (old_max, new_max), while the final record reuses the pair for
            # (row_sum, new_max).
            stage_bytes = stats_rows * 2 * 4
            self.smem_alloc = SmemAllocation(
                f"smem_vec_{tmem_vec_offset}",
                pipeline_config.num_stages * stage_bytes,
                alignment=16,
            )
        self.tmem_addr_cached = Int32(0)
        self.tmem_vec_addr_cached = Int32(0)
        self.tmem_ptr_vec_cached = placeholder_tmem_ptr()
        self.vec_old_max = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="Previous row maximum read from TMEM stats.",
        )
        self.vec_new_max = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="Current row maximum read from TMEM stats.",
        )
        self.vec_row_sum = TaskLocalVariable(
            dtype=Float32,
            default=Float32(0.0),
            docs="Softmax denominator read from TMEM stats.",
        )
        self.vec_scale = TaskLocalVariable(
            dtype=Float32,
            default=Float32(1.0),
            docs="Correction scale derived from TMEM stats.",
        )
        self.scale_softmax_log2_value = TaskLocalVariable(
            dtype=Float32,
            # Placeholder before load_scale_softmax_log2 reads the runtime tensor.
            default=Float32(0.0),
            docs="Softmax scale cached from the runtime scale tensor.",
        )
        self.output_scale_value = TaskLocalVariable(
            dtype=Float32,
            # Placeholder before load_output_scale reads the runtime tensor.
            default=Float32(1.0),
            docs="Output scale cached from the runtime scale tensor.",
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the staged SMEM stats ring when TMEM aliasing is disabled."""
        if not self.cfg.stats_via_smem:
            return []
        return [self.smem_alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Return the TMEM allocation for one correction-stat vector."""
        if self.cfg.stats_via_smem:
            return []
        return [self.alloc]

    @cute.jit
    def init_function_state(self, stage_info: StageInfo) -> None:
        """Initialize vec address fields to establish DSL type before scf.while.

        Real values computed by per-work-tile auxiliary work after setmaxnreg.

        Emits the correction stat slots (vec_old_max, vec_new_max,
        vec_row_sum, vec_scale) consumed by TmemO / SmemO via consumer-
        to-consumer routing; producer-side old_row_max / row_max /
        row_sum slots are auto-mirrored from TmemSP by
        Task.init_variables.
        """
        self.tmem_vec_addr_cached = Int32(0)
        self.tmem_ptr_vec_cached = prims.make_tmem_ptr(Int32(0), cutlass.Int8)
        _ = stage_info

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_state(self, stage_info: StageInfo) -> None:
        self.init_function_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_read_work_tile_state(self, stage_info: StageInfo) -> None:
        self.init_work_tile_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        self.init_function_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_work_tile_state(self, stage_info: StageInfo) -> None:
        self.init_work_tile_state(stage_info)

    @cute.jit
    def init_work_tile_state(self, stage_info: StageInfo) -> None:
        """Compute per-warp TMEM vec address and pointer each tile.

        Deferred from function-scope auxiliary work so the arithmetic runs
        after setmaxnreg and does not spill across the register-budget boundary.
        """
        if cutlass.const_expr(self.cfg.stats_via_smem):
            return
        # Softmax producer and correction consumer both use 4 warps.
        num_warps = 4
        warp_id_in_wg = cute.arch.warp_idx() % num_warps
        tmem_raw_addr = self.tmem_addr_cached
        tmem_base_row = tmem_raw_addr >> 16
        tmem_base_col = tmem_raw_addr & Int32(0xFFFF)
        row_id = tmem_base_row + warp_id_in_wg * cute.arch.WARP_SIZE
        self.tmem_vec_addr_cached = (row_id << 16) | (
            tmem_base_col + self.tmem_vec_offset
        )
        self.tmem_ptr_vec_cached = prims.make_tmem_ptr(
            self.tmem_vec_addr_cached, cutlass.Int8
        )
        _ = stage_info

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns="output_scale_value")
    @cute.jit
    def load_output_scale(self, stage_info: StageInfo) -> Float32:
        """Load the runtime output scale once before the correction loop."""
        _ = stage_info
        if cutlass.const_expr(self.output_scale is None):
            # Identity fallback for validation-only resource construction.
            return Float32(1.0)
        if cutlass.const_expr(self.cfg.vc_attention):
            # VC-Attention: output_scale is the per-(batch, head, channel) V
            # residual scale table, applied per column in SmemOResource.store_o_impl.
            return Float32(1.0)
        return self.output_scale

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns="scale_softmax_log2_value")
    @cute.jit
    def load_scale_softmax_log2(self, stage_info: StageInfo) -> Float32:
        """Load the runtime softmax scale once before the correction loop."""
        _ = stage_info
        if cutlass.const_expr(self.cfg.vc_attention):
            # VC-Attention softmax publishes row maxima already in log2 units.
            return Float32(1.0)
        if cutlass.const_expr(self.scale_softmax_log2 is None):
            # Safe fallback for validation-only resource construction.
            return Float32(0.0)
        return self.scale_softmax_log2

    @consumer_work(returns=(vec_old_max, vec_new_max, vec_row_sum, vec_scale))
    @cute.jit
    def read_vec(
        self,
        stage_info: StageInfo,
        *,
        scale_softmax_log2: SoftmaxScalar,
        final_stats: cutlass.Constexpr[bool] = False,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar, SoftmaxScalar, SoftmaxScalar]:
        """Read correction stats using the cached runtime softmax scale."""
        return self.read_vec_impl(stage_info, scale_softmax_log2, final_stats)

    @cute.jit
    def read_vec_impl(
        self,
        stage_info: StageInfo,
        scale_softmax_log2: SoftmaxScalar,
        final_stats: cutlass.Constexpr[bool] = False,
    ) -> tuple[SoftmaxScalar, SoftmaxScalar, SoftmaxScalar, SoftmaxScalar]:
        """Read correction stats from TMEM and cache in consumer_vars.

        CRITICAL: This read must happen here (immediately after the
        pipeline wait) rather than being deferred to TmemO.correct,
        because the TMEM stats region (cols 128-131 for TmemStats1, cols 0-3
        for TmemStats0) overlaps with the S0/S1 score regions. After the MMA warp
        commits O and continues to QK→S, the UMMA write to S can
        overwrite the stats data.  Reading here, before the O pipeline
        wait, ensures the stats are captured before any S overwrite.

        TmemO.correct retrieves the cached values from this
        resource's consumer_vars via a direct reference.
        """
        tmem_shape_vec = "32x32b"
        # Vector layout: [old_max, new_max, row_sum, pad].
        tmem_x_vec = self.cfg.tmem_stats_cols

        if cutlass.const_expr(self.cfg.stats_via_smem):
            vec_rmem = self.stats_smem_ptr(stage_info).load(
                count=2,
                alignment=8,
            )
        else:
            tmem_ptr_vec = prims.make_tmem_ptr(
                self.tmem_vec_addr_cached + self.stage_col_offset(stage_info),
                self.cfg.qk_acc_dtype,
            )
            vec_rmem = cutlass.Array(self.cfg.qk_acc_dtype, tmem_x_vec)
            vec_rmem[0:tmem_x_vec] = prims.tcgen05_ld(
                tmem_shape_vec, tmem_ptr_vec, num=tmem_x_vec
            )
            cute.arch.fence_view_async_tmem_load()

        vec_old_max = vec_rmem[0]
        vec_new_max = vec_rmem[1]
        vec_row_sum = Float32(0.0)
        if cutlass.const_expr(not self.cfg.stats_via_smem):
            vec_row_sum = vec_rmem[2]
        scale = Float32(1.0)
        if cutlass.const_expr(not (self.cfg.stats_via_smem and final_stats)):
            scale_ = scale_softmax_log2 * (vec_old_max - vec_new_max)
            scale = cute.math.exp2(scale_, fastmath=True)
        else:
            vec_row_sum = vec_rmem[0]
            vec_old_max = vec_new_max
        if cutlass.const_expr(
            final_stats
            and (self.lse_output is not None or self.attention_sinks is not None)
        ):
            scale *= self.store_final_lse(
                stage_info, vec_new_max, vec_row_sum, scale_softmax_log2
            )
        return vec_old_max, vec_new_max, vec_row_sum, scale

    @cute.jit
    def stage_col_offset(self, stage_info: StageInfo) -> Int32 | int:
        """Return the TMEM column offset for stage-scoped stats."""
        stage_col_offset = Int32(0)
        if cutlass.const_expr(self.cfg.stage_scoped_tmem_stats):
            stage_col_offset = stage_info.stage_idx * self.cfg.qk_mma_tiler[1]
        return stage_col_offset

    @cute.jit
    def stats_smem_ptr(self, stage_info: StageInfo) -> cute.Pointer:
        """Return this warp-group thread's staged SMEM stats slot."""
        context = stage_info.context
        assert context is not None and context.smem_base is not None
        assert self.smem_alloc is not None
        stats_rows = len(self.cfg.softmax0_warp_ids) * cute.arch.WARP_SIZE
        stats_elems_per_row = 2
        stage_elems = stats_rows * stats_elems_per_row
        tidx, _, _ = cute.arch.thread_idx()
        row_idx = tidx % stats_rows
        base_ptr = context.smem_base.data_ptr() + self.smem_alloc.offset
        view = cutlass.Array(
            base_ptr,
            dtype=Float32,
            shape=(self.pipeline_config.num_stages * stage_elems,),
            addrspace=3,
        )
        elem_offset = stage_info.stage_idx * stage_elems + row_idx * stats_elems_per_row
        return view.subview(elem_offset).data_ptr()

    @cute.jit
    def store_final_lse(self, stage_info, row_max, row_sum, scale_softmax_log2):
        # Match GmemO's logical row mapping, including paired heads/queries and
        # cluster tiles. Correction has one thread for each physical Q row.
        seq_tile, head_tile, batch = resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        head = (
            head_tile * self.cfg.work_tile_q_heads
            + self.peer_idx * self.cfg.peer_q_head_stride
        )
        row = cute.arch.thread_idx()[0] % Int32(128)
        query = (
            seq_tile * self.cfg.work_tile_q_seq_tiles
            + self.peer_idx * self.cfg.peer_q_seq_tile_stride
        ) * self.cfg.q_tile_m + row
        heads = (
            Int32(self.lse_output.shape[-1])
            if self.lse_output is not None
            else Int32(self.attention_sinks.shape[0])
        )
        weight = Float32(1.0)
        start = Int32(0)
        length = Int32(self.query_seq_len)
        if cutlass.const_expr(self.cum_seqlen_q is not None):
            start = Int32(self.cum_seqlen_q[batch])
            length = Int32(self.cum_seqlen_q[batch + Int32(1)]) - start
        if query < length and head < heads:
            flat_query = batch * length + query
            if cutlass.const_expr(self.cum_seqlen_q is not None):
                flat_query = start + query
            value = Float32(-Float32.inf)
            if row_sum > Float32(0.0):
                value = (
                    scale_softmax_log2 * row_max
                    + cute.math.log2(row_sum, fastmath=True)
                    - Float32(self.cfg.pv_p_scale_log2)
                )
            if cutlass.const_expr(self.attention_sinks is not None):
                sink = Float32(self.attention_sinks[head]) * Float32(1.4426950408889634)
                value, weight, _ = merge_log2_lse(value, sink)
            if cutlass.const_expr(self.lse_output is not None):
                offset = cutlass.Int64(flat_query) * cutlass.Int64(
                    heads
                ) + cutlass.Int64(head)
                (self.lse_output.iterator + offset).store(value)
        return weight

    @producer_work
    @cute.jit
    def store_vec(
        self,
        stage_info: StageInfo,
        *,
        old_row_max: SoftmaxScalar,
        row_max: SoftmaxScalar,
        row_sum: SoftmaxScalar,
        final_stats: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Softmax: publish correction statistics for one row.

        The TMEM-backed topology writes four elements per row:
          [0] = old_row_max (previous iteration's max)
          [1] = new_row_max (current iteration's max)
          [2] = row_sum (accumulated softmax denominator)
          [3] = padding

        The compact SMEM-backed topology writes ``[old_max, new_max]`` during
        the loop. Its final publication repurposes slot 0 for ``row_sum``.

        The Correction warp reads these to compute the rescale factor:
          scale = exp2(scale_log2 * (old_max - new_max))
        and to forward row_sum to SmemO for the final normalization.
        """
        if cutlass.const_expr(self.cfg.stats_via_smem):
            stat0 = old_row_max
            if cutlass.const_expr(final_stats):
                stat0 = row_sum
            vec_data = cutlass.Vector.from_elements(
                (stat0, row_max),
                self.cfg.qk_acc_dtype,
            )
            self.stats_smem_ptr(stage_info).store(vec_data, alignment=8)
        else:
            vec_data = cutlass.Vector.from_elements(
                (old_row_max, row_max, row_sum, Float32(0.0)),
                self.cfg.qk_acc_dtype,
            )
            tmem_ptr_vec = prims.make_tmem_ptr(
                self.tmem_vec_addr_cached + self.stage_col_offset(stage_info),
                cutlass.Int8,
            )
            prims.tcgen05_st(
                "32x32b",
                tmem_ptr_vec,
                vec_data,
            )
            cute.arch.fence_view_async_tmem_store()
