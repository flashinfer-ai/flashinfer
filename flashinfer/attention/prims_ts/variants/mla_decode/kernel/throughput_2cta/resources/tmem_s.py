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

"""TMEM S scores and the online softmax of the 2CTA MLA kernel."""

import cutlass.cute as cute
import cutlass
from cutlass import Boolean, Float32, Int32, Int64
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field
from ......device.mask import kv_tile_needs_right_mask
from ...device.constants import (
    EPILOGUE_COLUMN_GROUP_SHIFT,
    EPILOGUE_ROW_MASK,
    TCGEN05_32B_REGS_PER_LOAD,
    TCGEN05_32B_SHAPE,
)
from ...device.gather import CachedSparseMask, cache_sparse_mask, invalid_sparse_token
from ...device.context_parallel import local_kv_length
from ...device.mask import MaskType, mask_visible_k_length, rows_have_distinct_kv_prefix
from ...device.math import (
    add_packed_f32x2,
    ceil_div,
    fma_packed_f32x2,
    mma_k_step_for_qkv,
    mma_kind_for_qkv,
    mul_packed_f32x2,
    qk_desc_layout,
    qk_desc_layout_for_head_dim,
    qk_desc_leading_byte_offset,
    qk_desc_leading_byte_offset_for_head_dim,
    qk_desc_stride_byte_offset,
    qk_desc_stride_byte_offset_for_head_dim,
    qkv_dtype,
)
from ...device.ops import fmax_f32, fp8_log2_p_scale
from ...device.query import query_batch_bounds, flat_query_row_state
from ...device.sparse_indices import SparseKVIndices, cache_sparse_source_lengths
from ..config import MlaDecodeConfig
from .common import HighThroughputMlaResource
from typing import Any, ClassVar


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

    # Static configuration
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)
    correction_factor_out: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    correction_factor_out_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    logical_num_heads_q: cutlass.Constexpr[int] = 128
    no_correction_out: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    no_correction_out_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    qk_acc_regs: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    qk_acc_regs_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_max_new: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max_new_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_max_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_sum: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    row_sum_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_sum_out: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    row_sum_out_odd: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    task_local_specs: ClassVar[tuple[tuple, ...]] = (
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

    # Runtime scalars
    logical_seq_len_q: Int32 = 1

    # Per-task state
    cache_seqs: Any = None  # per-batch valid K length
    cp_rank: Any = 0  # runtime context-parallel rank
    cta_rank: Any = field(init=False, default=None)
    cu_seqlens_q: Any = None  # cumulative compact-Q offsets, or None for fixed Q
    is_leader: Any = field(init=False, default=None)
    page_offsets: Any = None
    request_lengths: Any = field(init=False, default=None)
    row_max_state: Any = field(init=False, default=None)
    row_sum_state: Any = field(init=False, default=None)
    scale_softmax_log2: Any = None  # softmax_scale * log2(e)
    smem_exchange: Any = None  # SMEM array for cross-warp max exchange
    smem_p: Any = None  # SMEM P array
    smem_q_latent: Any = None  # SMEM Q pointers for descriptor building
    smem_q_rope: Any = None
    split_kv: Any = None  # per-work-tile split count
    tiled_mma_qk: Any = None
    tmem_base_addr: Any = None  # TMEM base address (from alloc)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def bind_sparse_request(self, stage_info: StageInfo):
        """Cache live source lengths for this query before its KV/softmax loop."""
        if cutlass.const_expr(isinstance(self.page_offsets, SparseKVIndices)):
            cache_sparse_source_lengths(
                self.page_offsets,
                stage_info.work_tile.tile_idx[2],
                self.request_lengths,
                include_primary_tiles=False,
            )

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
        row_sum = row_sum * correction_factor_out
        row_sum_vec = (Float32(0), Float32(0))
        for i in cutlass.range_constexpr(0, 64, 2):
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
        row_sum = row_sum_odd * correction_factor_out_odd
        row_sum_vec = (Float32(0), Float32(0))
        for i in cutlass.range_constexpr(0, 64, 2):
            row_sum_vec = add_packed_f32x2(
                row_sum_vec,
                (qk_acc_regs_odd[i], qk_acc_regs_odd[i + 1]),
            )
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum
        self.row_sum_state = row_sum
        return row_sum, row_sum

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
        return self.finish_softmax_impl(
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

    @cute.jit
    def finish_softmax_impl(
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
        """Finish row-max exchange, online correction, and P exponentiation."""
        del row_sum_out, correction_factor_out, no_correction_out
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        neg_inf = Float32(-Float32.inf)
        row_max_prev = row_max
        row_sum_prev = row_sum
        row_max_tile = row_max_new

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
        # The exchange buffer is reused on the next KV iteration.  Keep every
        # peer read ahead of any warp's next write to the same slot.
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
        if cutlass.const_expr(cfg.softmax_rescale_threshold_log2 > 0.0):
            # Freeze the running row max while this K tile raises it by at most
            # the threshold. P then grows by at most 2^T, and the unchanged max
            # keeps this row's O rescale on the existing exact-equality skip
            # path below. The first tile (row_max_prev == -inf) yields an
            # infinite or NaN increase and never freezes. Compare in log2 units with
            # the threshold as an immediate: the scale is already live here.
            max_increase_log2 = (row_max_new - row_max_prev) * self.scale_softmax_log2
            if max_increase_log2 <= Float32(cfg.softmax_rescale_threshold_log2):
                row_max_new = row_max_prev
        row_has_values = row_max_new != neg_inf
        safe_row_max_prev = row_max_prev if row_has_values else Float32(0)
        safe_row_max_new = row_max_new if row_has_values else Float32(0)
        # Exact max equality makes the correction scale exactly one. Keep that
        # lane on the identity value and avoid issuing exp2 altogether.
        max_changed = safe_row_max_prev != safe_row_max_new
        correction_factor = Float32(1)
        if max_changed:
            correction_factor = cute.math.exp2(
                (safe_row_max_prev - safe_row_max_new) * self.scale_softmax_log2,
                fastmath=True,
            )
        no_correction = Int32(not max_changed)

        fma_b = self.scale_softmax_log2
        fma_c = Float32(0) - safe_row_max_new * self.scale_softmax_log2
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            # Use the same selected FP8 probability scale as output and LSE:
            # 448 with exact maxima, 448 * 2**-T with a rescale threshold T.
            fma_c = fma_c + fp8_log2_p_scale(cfg)
        for i in cutlass.range_constexpr(0, 64, 2):
            fma_result = fma_packed_f32x2(
                (qk_acc_regs[i], qk_acc_regs[i + 1]),
                (fma_b, fma_b),
                (fma_c, fma_c),
            )
            qk_acc_regs[i] = cute.math.exp2(fma_result[0], fastmath=True)
            qk_acc_regs[i + 1] = cute.math.exp2(fma_result[1], fastmath=True)

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
        return self.finish_softmax_impl(
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
        self.row_max_state = Float32(-Float32.inf)
        self.row_sum_state = Float32(0)
        # Allocate outside skipped-work guards; refresh values for every tile.
        self.request_lengths = cutlass.Array(Int32, 2, space=cutlass.AddressSpace.rmem)
        return (
            cutlass.Array(
                Float32,
                64,
                space=cutlass.AddressSpace.rmem,
            ),
            Float32(-Float32.inf),
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
        self.row_max_state = Float32(-Float32.inf)
        self.row_sum_state = Float32(0)
        return (
            cutlass.Array(
                Float32,
                64,
                space=cutlass.AddressSpace.rmem,
            ),
            Float32(-Float32.inf),
            Float32(0),
            Float32(0),
            Float32(0),
            Float32(0),
            Int32(0),
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
        return self.load_s_impl(
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
    def load_s_cached(
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
        cached_mask_low,
        cached_mask_high,
    ):
        """Load S for the even softmax group."""
        return self.load_s_impl(
            stage_info,
            qk_acc_regs=qk_acc_regs,
            row_max=row_max,
            row_sum=row_sum,
            row_sum_out=row_sum_out,
            row_max_new=row_max_new,
            correction_factor_out=correction_factor_out,
            no_correction_out=no_correction_out,
            cached_mask_low=cached_mask_low,
            cached_mask_high=cached_mask_high,
        )

    @cute.jit
    def load_s_impl(
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
        cached_mask_low=None,
        cached_mask_high=None,
    ):
        """Load S from TMEM and compute the local row max."""
        del row_sum_out, row_max_new, correction_factor_out, no_correction_out
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        num_compute_threads = cfg.num_compute_warps * cfg.threads_per_warp
        local_tidx = tidx % num_compute_threads
        stage_idx = stage_info.stage_idx

        work_tile = stage_info.work_tile
        K = local_kv_length(cfg, Int32(work_tile.k_len), self.cp_rank)
        k_index = work_tile.k_index_base + Int32(stage_info.loop_offset)
        tile_offset_k = k_index * Int32(cfg.mma_qk_tiler[1])
        needs_row_causal_mask = cutlass.const_expr(
            cfg.cp_world == 1 and cfg.mask_type == MaskType.CAUSAL.value
        )
        needs_cp_causal_mask = cutlass.const_expr(
            cfg.cp_world > 1 and cfg.mask_type == MaskType.CAUSAL.value
        )
        # Mask only tiles reaching past the keys every row sees: the request's
        # K length, or the work tile's cached prefix when rows differ. The
        # per-row bounds load only in the masked branch below.
        visible_k_end = K
        if cutlass.const_expr(
            rows_have_distinct_kv_prefix(
                cfg.mask_type, self.cu_seqlens_q is not None, self.logical_seq_len_q
            )
        ):
            visible_k_end = Int32(work_tile.k_visible_end)
        group_needs_mask = kv_tile_needs_right_mask(
            tile_offset_k, Int32(cfg.mma_qk_tiler[1]), visible_k_end
        )

        neg_inf = Float32(-Float32.inf)
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

        if cutlass.const_expr(cfg.page_size == 1):
            group_needs_mask = True
        if cutlass.const_expr(
            isinstance(self.page_offsets, SparseKVIndices)
            and not self.page_offsets.is_bound
        ):
            mask_page_offsets = self.page_offsets.bind_query_row(
                work_tile.tile_idx[2], self.request_lengths[0], self.request_lengths[1]
            )
        else:
            mask_page_offsets = self.page_offsets
        if cutlass.const_expr(cfg.has_fp8_primary):
            scale = mask_page_offsets.tensor_scale(tile_offset_k)
            if scale != cutlass.Float32(1):
                for i in cutlass.range_constexpr(0, 64, 2):
                    pair = mul_packed_f32x2(
                        (qk_acc_regs[i], qk_acc_regs[i + 1]), (scale, scale)
                    )
                    qk_acc_regs[i], qk_acc_regs[i + 1] = pair[0], pair[1]

        if cutlass.const_expr(cached_mask_low is not None):
            mask_routes = CachedSparseMask((cached_mask_low, cached_mask_high), False)
            # The cached invalid bits already include the live K boundary.
            # Whole valid halves need no per-score masking instructions.
            if cutlass.const_expr(not needs_row_causal_mask):
                group_needs_mask = (cached_mask_low | cached_mask_high) != Int32(0)
        else:
            raw_mask_routes = (
                mask_page_offsets.select_source_tile(
                    tile_offset_k, work_tile.tile_idx[2], for_mask=True
                )
                if cutlass.const_expr(isinstance(mask_page_offsets, SparseKVIndices))
                else self.page_offsets
            )
            mask_routes = (
                cache_sparse_mask(
                    raw_mask_routes,
                    tile_offset_k,
                    K,
                    work_tile.tile_idx[2],
                    swaps=False,
                    column_half=local_tidx >> EPILOGUE_COLUMN_GROUP_SHIFT,
                )
                if cutlass.const_expr(
                    cfg.page_size == 1
                    and (
                        isinstance(self.logical_seq_len_q, int)
                        and self.logical_seq_len_q == 1
                    )
                )
                else raw_mask_routes
            )
            if cutlass.const_expr(
                cfg.page_size == 1
                and (
                    isinstance(self.logical_seq_len_q, int)
                    and self.logical_seq_len_q == 1
                )
                and not needs_row_causal_mask
            ):
                group_needs_mask = (mask_routes.low | mask_routes.high) != Int32(0)
        if group_needs_mask:
            row_kv_bound, logical_seq_len_q = self.masked_row_prefix_bounds(
                work_tile, local_tidx, needs_row_causal_mask, needs_cp_causal_mask
            )
            if cutlass.const_expr(needs_row_causal_mask):
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
                if cutlass.const_expr(needs_cp_causal_mask):
                    token_is_visible = token_idx < row_kv_bound
                elif cutlass.const_expr(needs_row_causal_mask):
                    mask_flat_row = Int32(self.logical_num_heads_q) * (
                        token_idx - K + logical_seq_len_q
                    )
                    token_is_visible = flat_query_row >= mask_flat_row
                elif cutlass.const_expr(
                    cached_mask_low is not None
                    or getattr(self.page_offsets, "assume_valid_prefix", False)
                ):
                    token_is_visible = not mask_routes.is_invalid(
                        token_idx, work_tile.tile_idx[2]
                    )
                else:
                    token_is_visible = not invalid_sparse_token(
                        token_idx,
                        K,
                        mask_routes,
                        work_tile.tile_idx[2],
                        sparse=cfg.page_size == 1,
                    )
                qk_acc_regs[i] = qk_acc_regs[i] if token_is_visible else neg_inf

        max0 = neg_inf
        max1 = neg_inf
        max2 = neg_inf
        max3 = neg_inf
        for i in cutlass.range_constexpr(16):
            max0 = fmax_f32(max0, qk_acc_regs[i])
            max1 = fmax_f32(max1, qk_acc_regs[i + 16])
            max2 = fmax_f32(max2, qk_acc_regs[i + 32])
            max3 = fmax_f32(max3, qk_acc_regs[i + 48])
        row_max_tile = fmax_f32(
            row_max_tile,
            fmax_f32(fmax_f32(max0, max1), fmax_f32(max2, max3)),
        )
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
        return self.load_s_impl(
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
    def masked_row_prefix_bounds(
        self, work_tile, local_tidx, needs_row_causal_mask, needs_cp_causal_mask
    ):
        """Load a masked tile's rank-local row bound and the request's Q length.

        Called only on masked tiles, so the metadata loads stay off the
        unmasked ones.
        """
        cfg = self.cfg
        row_kv_bound = Int32(0)
        logical_seq_len_q = Int32(0)
        if cutlass.const_expr(needs_row_causal_mask or needs_cp_causal_mask):
            batch_idx = Int32(work_tile.tile_idx[2])
            _, logical_seq_len_q = query_batch_bounds(
                self.cu_seqlens_q, batch_idx, self.logical_seq_len_q
            )
        if cutlass.const_expr(needs_cp_causal_mask):
            row_in_tile = self.cta_rank * Int32(
                cfg.mma_qk_tiler[0] // cfg.num_mma_ctas
            ) + (local_tidx & Int32(EPILOGUE_ROW_MASK))
            _, _, logical_q_idx, _, _ = flat_query_row_state(
                row_in_tile,
                Int32(work_tile.tile_idx[1]),
                cfg.mma_qk_tiler[0],
                self.logical_num_heads_q,
                self.logical_seq_len_q,
                self.cu_seqlens_q,
                batch_idx,
            )
            global_bound = mask_visible_k_length(
                cfg.mask_type, Int32(work_tile.k_len), logical_q_idx, logical_seq_len_q
            )
            row_kv_bound = local_kv_length(cfg, global_bound, self.cp_rank)
        return row_kv_bound, logical_seq_len_q

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
                            if cutlass.const_expr(cfg.has_nvfp4_kv):
                                # Q NoPE occupies TMEM columns 384:512:
                                # 512 E4M3 bytes / 4 bytes per column = 128.
                                # Each D128 slice advances 32 columns; each
                                # K32 MMA step advances another 8 columns.
                                prims.tcgen05_mma(
                                    mma_kind,
                                    cta_group,
                                    tmem_s_ptr,
                                    prims.make_tmem_ptr(
                                        self.tmem_base_addr
                                        + Int32(
                                            384 + logical_call_idx * 32 + k_block * 8
                                        ),
                                        qkv_dtype(cfg),
                                    ),
                                    desc_k_subtile + k_block * 2,
                                    idesc_qk,
                                    Boolean(scale_d),
                                )
                            else:
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
