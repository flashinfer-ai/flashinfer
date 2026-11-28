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

"""TMEM O accumulation and its rescaling by the correction warps."""

import cutlass.cute as cute
import cutlass
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
    StageInfo,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field
from ..config import FmhaConfig
from ..device.common import SOFTMAX_MASK_VALUE, SoftmaxScalar, TmemAddr, TmemPtr
from .smem_p import SmemPResource
from .tmem_stats import TmemStatsResource
from .smem_mu import VC_MEAN_TILE_LBO, VC_MEAN_TILE_SBO
from .....device.packing import freeze_smem_descriptor
from .....device.placeholders import placeholder_tmem_ptr
from .....device.stage import WorkStage
from ....vc_attention import VC_MEAN_GROUP_TILES, VC_MEAN_OPERANDS
from typing import Any, Optional


# ---------------------------------------------------------------------------
# TmemOResource -- TMEM O accumulation with UmmaAsync pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class TmemOResource(MemoryResource):
    """TMEM O accumulation with a topology-derived UmmaAsync pipeline.

    Producer: MMA writes P*V into one O accumulator per Q instance.
    Consumer: Correction rescales O in-place.

    Paired schedules use two stages so MMA can commit O0, work on O1,
    commit O1, then acquire O0. Single-instance schedules use one stage
    because they write one physical O accumulator.
    """

    # Static configuration
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    tmem_addr_cached: TmemAddr | None = field(init=False, default=None)
    tmem_o0_offset: Constexpr[int] = field(init=False, default=None)
    tmem_o1_offset: Constexpr[int] = field(init=False, default=None)
    # Precomputed per-warp TMEM O address base: (row_id << 16) | tmem_base_col.
    # consumer_work adds tmem_o_offset to get the final O0/O1 address.
    tmem_o_addr_base_cached: TmemAddr | None = field(init=False, default=None)
    # P-stage base supplied by TmemPResource for split S/P scheduling.
    tmem_p_base_cached: TmemAddr | None = field(init=False, default=None)
    # Precomputed TMEM raw pointer (inttoptr of tmem_addr_cached).
    tmem_ptr_raw_cached: TmemPtr | None = field(init=False, default=None)
    # SMEM descriptors of the two row-sum A operands of the mean step, one per
    # UMMA step of a tile group.
    vc_desc_l_cached: Any | None = field(init=False, default=None)
    # UMMA instruction descriptor of the bf16 K=16 mean step.
    vc_idesc_mu_cached: Any | None = field(init=False, default=None)
    # VC-Attention mean step state, set once per kernel in the prologue.
    # Whether this CTA issues the mean UMMA. Under two-CTA UMMA only the leader
    # CTA of the pair issues, so this is block_idx_in_cluster() == 0 made warp
    # uniform; otherwise it is True.
    vc_mean_issue_cached: Any | None = field(init=False, default=None)

    # Linked resources
    # p_in_smem: per-group SMEM P tiles read by the SS PV MMA.
    smem_p0_resource: Optional[SmemPResource] = field(init=False, default=None)
    smem_p1_resource: Optional[SmemPResource] = field(init=False, default=None)
    # References to TmemStats resources for reading cached correction stats.
    # consumer_work reads stats from these instead of from TMEM, because
    # the stats TMEM region overlaps with S0/S1 and can be overwritten by
    # MMA's QK→S before correction reads it.
    tmem_vec0_resource: TmemStatsResource | None = field(init=False, default=None)
    tmem_vec1_resource: TmemStatsResource | None = field(init=False, default=None)

    # Allocations
    alloc_o0: Constexpr[Optional[TmemAllocation]] = field(init=False, default=None)
    alloc_o1: Constexpr[Optional[TmemAllocation]] = field(init=False, default=None)

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        tmem_o0_offset: int,
        tmem_o1_offset: int,
        tmem_vec0_resource: TmemStatsResource | None = None,
        tmem_vec1_resource: TmemStatsResource | None = None,
        smem_p0_resource: Optional[SmemPResource] = None,
        smem_p1_resource: Optional[SmemPResource] = None,
        **kwargs: Any,
    ) -> None:
        """Bind O TMEM offsets, correction-stat resources, and SMEM P tiles."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.smem_p0_resource = smem_p0_resource
        self.smem_p1_resource = smem_p1_resource
        self.tmem_o0_offset = tmem_o0_offset
        self.tmem_o1_offset = tmem_o1_offset
        self.tmem_vec0_resource = tmem_vec0_resource
        self.tmem_vec1_resource = tmem_vec1_resource
        self.alloc_o0 = TmemAllocation("tmem_o0", cfg.epi_tile[1])
        self.alloc_o1 = TmemAllocation("tmem_o1", cfg.epi_tile[1])
        self.tmem_addr_cached = Int32(0)
        self.tmem_ptr_raw_cached = placeholder_tmem_ptr()
        self.tmem_o_addr_base_cached = Int32(0)

    @property
    def loop_offset_sensitive(self) -> bool:
        """Return true because PV accumulation scale depends on loop_offset."""
        # producer_work uses loop_offset to compute scale_d.
        return True

    @consumer_work
    @cute.jit
    def correct(
        self,
        stage_info: StageInfo,
        *,
        vec_old_max: SoftmaxScalar,
        vec_new_max: SoftmaxScalar,
        vec_scale: SoftmaxScalar,
        inst_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Correction using the scale cached by TmemStatsResource."""
        self.correct_impl(
            stage_info,
            vec_old_max=vec_old_max,
            vec_new_max=vec_new_max,
            scale_softmax_log2=Float32(0.0),
            vec_scale=vec_scale,
            use_cached_scale=True,
            inst_idx=inst_idx,
            is_tail=is_tail,
        )

    @cute.jit
    def correct_impl(
        self,
        stage_info: StageInfo,
        *,
        vec_old_max: SoftmaxScalar,
        vec_new_max: SoftmaxScalar,
        scale_softmax_log2: SoftmaxScalar,
        vec_scale: SoftmaxScalar,
        use_cached_scale: Constexpr[bool],
        inst_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Correction: read cached stats, compute scale, rescale O.

        Reads the correction stats [old_max, new_max, row_sum] from the
        TmemStatsResource's consumer_vars (cached by TmemStats.read_vec
        right after the pipeline wait).  This avoids a TMEM race: the stats
        region overlaps with S0/S1, and MMA's QK->S can overwrite it
        between O commit and the time Correction reads here.

        Uses inst_idx to select O0/O1 and forwards row_sum to
        SmemO.producer_vars for the tail epilog.

        With skip_correction enabled, rescale only when a row with prior
        visible keys changes its maximum. Empty histories and unchanged
        maxima skip the TMEM load+rescale+store.
        """
        # Select O offset from captured correction-call position.
        if cutlass.const_expr(self.cfg.single_qkv_instance or inst_idx == 0):
            tmem_o_offset = self.tmem_o0_offset
        else:
            tmem_o_offset = self.tmem_o1_offset

        # In causal mode with no Q right offset, skip the invalid O0 correction
        # in the last LOOP iteration. The task domain pads partial final CTAs so
        # this slot is always outside peer0's causal reach.
        skip_o0_invalid = False
        if cutlass.const_expr(
            self.cfg.skip_causal_invalid_peer0
            and not self.cfg.single_qkv_instance
            and inst_idx == 0
        ):
            # This is O0 correction; check if this is the last LOOP iteration.
            if not is_tail:
                skip_o0_invalid = stage_info.loop_offset == (stage_info.loop_end - 1)

        # Check if we should skip correction (when old_max == new_max)
        should_rescale = cutlass.Boolean(True)
        if cutlass.const_expr(self.cfg.enable_skip_correction):
            # A neutral old maximum means every prior P tile was zero.
            # The first visible tile establishes an anchor but has no old O
            # contribution to rescale, including a late first tile in one Q peer.
            needs_rescale = (vec_old_max != vec_new_max) & (
                vec_old_max != Float32(SOFTMAX_MASK_VALUE)
            )
            vote_ballot_cnt = cute.arch.vote_ballot_sync(needs_rescale)
            should_rescale = vote_ballot_cnt != Int32(0)

        scale = Float32(1.0)
        if should_rescale:
            if cutlass.const_expr(use_cached_scale):
                scale = vec_scale
            else:
                scale_ = scale_softmax_log2 * (vec_old_max - vec_new_max)
                scale = cute.math.exp2(scale_, fastmath=True)

        # PTX ISA 9.7.16.6.4.4: Non-pipelined instructions, different thread.
        # MMA (Thread 0) does tcgen05.mma → tcgen05.commit on O_full.
        # Correction (Thread 1) does mbarrier.try_wait on O_full → tcgen05.ld.
        # The fence orders the prior tcgen05.commit's completion with our tcgen05.ld.
        prims.tcgen05_fence("after")

        # Only rescale if old_max != new_max AND not in invalid O0 iteration
        if should_rescale and not skip_o0_invalid:
            # Load O, rescale, store back
            tmem_o_addr = self.tmem_o_addr_base_cached + tmem_o_offset

            tmem_shape = "32x32b"
            # Amortize TMEM handshakes with 32-value correction transfers.
            tmem_x = 32

            num_iters = self.cfg.pv_mma_tiler[1] // tmem_x
            for i in cutlass.range_constexpr(num_iters):
                tmem_tile_addr = tmem_o_addr + i * tmem_x
                tmem_ptr = cutlass.inttoptr(
                    tmem_tile_addr,
                    mem_space=6,
                    dtype=self.cfg.pv_acc_dtype,
                )

                # Load from TMEM as vector, scale, store back
                o_vec = prims.tcgen05_ld(tmem_shape, tmem_ptr, num=tmem_x)
                cute.arch.fence_view_async_tmem_load()
                scale_vec = cutlass.vector.full_like(o_vec, scale)
                o_scaled = o_vec * scale_vec
                prims.tcgen05_st(tmem_shape, tmem_ptr, o_scaled)

            cute.arch.fence_view_async_tmem_store()
        # else: skip TMEM rescale entirely when scale=1.0

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Return the TMEM allocations for the double-buffered O accumulators."""
        return (
            [self.alloc_o0]
            if self.cfg.single_qkv_instance
            else [self.alloc_o0, self.alloc_o1]
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_correction_state(self, stage_info: StageInfo) -> None:
        self.init_function_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_correction_work_tile_state(self, stage_info: StageInfo) -> None:
        self.init_work_tile_state(stage_info)

    @cute.jit
    def init_function_state(self, stage_info: StageInfo) -> None:
        """Precompute MMA-warp TMEM raw pointer (once, ungated).

        Per-warp O address base for correction warps is deferred to
        per-work-tile auxiliary work to avoid crossing the setmaxnreg boundary.
        tmem_o_addr_base_cached initialized to Int32(0) to establish DSL type.

        Pure consumer/producer of upstream emitters — emits no
        consumer vars itself.  Producer-side desc_v_base slot is
        auto-mirrored from SmemKV; consumer-side vec_old_max /
        vec_new_max slots are auto-mirrored from TmemStats via
        consumer-to-consumer routing in the captured schedule.
        """
        self.tmem_ptr_raw_cached = prims.make_tmem_ptr(
            self.tmem_addr_cached, cutlass.Int8
        )
        self.tmem_o_addr_base_cached = Int32(0)
        if cutlass.const_expr(self.cfg.vc_restores_means):
            issue = cutlass.Boolean(True)
            if cutlass.const_expr(self.cfg.two_cta_umma):
                issue = (
                    cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
                )
            self.vc_mean_issue_cached = issue
            self.vc_desc_l_cached = tuple(
                self.vc_rowsum_desc(stage_info, smem_p)
                for smem_p in (self.smem_p0_resource, self.smem_p1_resource)
            )
            self.vc_idesc_mu_cached = prims.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float32,
                a_dtype=cutlass.BFloat16,
                b_dtype=cutlass.BFloat16,
                n_dim=self.cfg.pv_mma_tiler[1],
                m_dim=self.cfg.pv_mma_tiler[0] * self.cfg.cta_group_size,
            )
        self.tmem_p_base_cached = Int32(0)
        _ = stage_info

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_state(self, stage_info: StageInfo) -> None:
        self.init_function_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_mma_work_tile_state(self, stage_info: StageInfo) -> None:
        self.init_work_tile_state(stage_info)

    @cute.jit
    def init_work_tile_state(self, stage_info: StageInfo) -> None:
        """Compute per-warp TMEM O address base each tile (after setmaxnreg)."""
        num_correction_warps = 4
        warp_id_in_wg = cute.arch.warp_idx() % num_correction_warps
        tmem_raw_addr = self.tmem_addr_cached
        tmem_base_row = tmem_raw_addr >> 16
        tmem_base_col = tmem_raw_addr & Int32(0xFFFF)
        row_id = tmem_base_row + warp_id_in_wg * cute.arch.WARP_SIZE
        self.tmem_o_addr_base_cached = (row_id << 16) | tmem_base_col
        _ = stage_info

    @producer_work
    @cute.jit
    def pv_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_v_base: prims.Tcgen05SmemDesc,
        section: cutlass.Constexpr[WorkStage],
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        inst_idx: cutlass.Constexpr[int] = 0,
        is_tail: cutlass.Constexpr[bool] = False,
        k_half: cutlass.Constexpr[int | None] = None,
    ) -> None:
        """PV MMA: P*V -> O (double-buffered O0/O1). See ``pv_mma_impl``."""
        self.pv_mma_impl(
            stage_info,
            desc_v_base=desc_v_base,
            section=section,
            head_dim_stage_idx=head_dim_stage_idx,
            inst_idx=inst_idx,
            is_tail=is_tail,
            k_half=k_half,
        )

    @cute.jit
    def pv_mma_impl(
        self,
        stage_info: StageInfo,
        *,
        desc_v_base: prims.Tcgen05SmemDesc,
        section: cutlass.Constexpr[WorkStage],
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        inst_idx: cutlass.Constexpr[int] = 0,
        is_tail: cutlass.Constexpr[bool] = False,
        k_half: cutlass.Constexpr[int | None] = None,
    ) -> None:
        """PV MMA: P*V -> O (double-buffered O0/O1).

        ``k_half`` issues only the first (0) or second (1) half of the K slices
        so the schedule can run PV on the leading half of P before the rest is
        stored. The second half accumulates onto the first.

        Uses captured schedule section and call index to select O0/O1 and
        scale_d statically.
        Reads P from TMEM (via tmem_p offset on the corresponding SP resource)
        and V descriptor from SmemV consumer vars.

        In causal mode with no Q right offset, skips P0*V→O0 MMA in the last
        LOOP iteration, since Softmax0's domain is N-2 but MMA's domain is N-1.
        The task domain pads partial final CTAs so this slot is always outside
        peer0's causal reach.
        """
        if cutlass.const_expr(self.cfg.p_in_smem):
            # PV0(i), PV1(i) every iteration and in the tail, so inst_idx is the group.
            writes_o0 = inst_idx == 0
            first_o0_write = False
            first_o1_write_maybe = False
        elif cutlass.const_expr(section == WorkStage.Head):
            writes_o0 = True
            first_o0_write = True
            first_o1_write_maybe = False
        elif cutlass.const_expr(section == WorkStage.Loop):
            writes_o0 = inst_idx == 1
            first_o0_write = False
            first_o1_write_maybe = inst_idx == 0
        else:
            writes_o0 = False
            first_o0_write = False
            first_o1_write_maybe = True

        # In causal mode, check if O0 MMA should skip peer 0's invalid last tile:
        # the last loop iteration, or the tail when P in SMEM moves PV0 there.
        skip_o0_invalid = False
        if cutlass.const_expr(self.cfg.skip_causal_invalid_peer0 and writes_o0):
            if cutlass.const_expr(self.cfg.p_in_smem):
                skip_o0_invalid = is_tail
            elif cutlass.const_expr(section == WorkStage.Loop):
                if not is_tail:
                    skip_o0_invalid = stage_info.loop_offset == (
                        stage_info.loop_end - 1
                    )

        if not skip_o0_invalid:
            tmem_ptr_raw = self.tmem_ptr_raw_cached

            if cutlass.const_expr(self.cfg.v_dtype.width == 8):
                mma_kind = prims.Tcgen05MMAKind.F8F6F4
                # E4M3 operands use the Float16 encoding handle.
                ab_format = cutlass.Float16
            else:
                mma_kind = prims.Tcgen05MMAKind.F16
                if cutlass.const_expr(self.cfg.v_dtype == cutlass.BFloat16):
                    ab_format = cutlass.BFloat16
                else:
                    ab_format = cutlass.Float16

            pv_fp8_k64 = self.cfg.v_dtype.width == 8 and self.cfg.umma_fp8_k64
            idesc_pv = prims.Tcgen05InstrDesc.build(
                c_dtype=cutlass.Float32,
                a_dtype=ab_format,
                b_dtype=ab_format,
                n_dim=(
                    min(
                        self.cfg.head_dim_per_stage_kv,
                        self.cfg.pv_mma_tiler[1]
                        - head_dim_stage_idx * self.cfg.head_dim_per_stage_kv,
                    )
                    if self.cfg.stage_kv_by_head_dim
                    else self.cfg.pv_mma_tiler[1]
                ),
                m_dim=self.cfg.pv_mma_tiler[0] * self.cfg.cta_group_size,
                # V is row-major / MN-major.
                b_major=1,
                k_dim=1 if pv_fp8_k64 else 0,
            )
            cta_group = (
                prims.CTAGroup.CTA_2
                if cutlass.const_expr(self.cfg.two_cta_umma)
                else prims.CTAGroup.CTA_1
            )
            issue_mma = cutlass.Boolean(True)
            if cutlass.const_expr(self.cfg.two_cta_umma):
                issue_mma = (
                    cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
                )

            pv_n_dim = self.cfg.pv_mma_tiler[1]
            num_head_dim_stages = 1
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                pv_n_dim = self.cfg.head_dim_per_stage_kv
                num_head_dim_stages = self.cfg.num_head_dim_stages_v
            head_dim_stage_start = 0
            num_head_dim_stages_to_issue = num_head_dim_stages
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                num_head_dim_stages_to_issue = 1
                head_dim_stage_start = head_dim_stage_idx

            k_dim_per_mma = 16
            if cutlass.const_expr(self.cfg.v_dtype.width != 16):
                k_dim_per_mma = 64 if pv_fp8_k64 else 32
            num_kphases_pv = self.cfg.pv_mma_tiler[2] // k_dim_per_mma
            inc_tmem_p = (
                k_dim_per_mma * self.cfg.v_dtype.width // self.cfg.qk_acc_dtype.width
            )
            tma_copy_iters_per_head_dim_stage = (
                self.cfg.tma_copy_v_iters // num_head_dim_stages
            )
            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                tma_copy_iters_per_head_dim_stage = self.cfg.tma_copy_v_stage_iters
            inc_bytes_v = (
                k_dim_per_mma
                * (pv_n_dim // tma_copy_iters_per_head_dim_stage)
                * self.cfg.v_dtype.width
                // 8
            )
            if cutlass.const_expr(self.cfg.two_cta_umma):
                # One V fragment of tma_copy_v_granu_inner columns per CTA.
                inc_bytes_v = (
                    k_dim_per_mma
                    * self.cfg.tma_copy_v_granu_inner
                    * self.cfg.v_dtype.width
                    // 8
                )
            v_chunk_bytes = self.cfg.tma_copy_v_bytes // self.cfg.tma_copy_v_stage_iters
            head_dim_stage_bytes_v = v_chunk_bytes * tma_copy_iters_per_head_dim_stage

            # Select O buffer and P offset at trace time (compile-time constant)
            if cutlass.const_expr(self.cfg.single_qkv_instance or writes_o0):
                tmem_ptr_o = tmem_ptr_raw.subview(self.tmem_o0_offset)
                tmem_p_base = self.cfg.tmem_p0_offset
            else:
                tmem_ptr_o = tmem_ptr_raw.subview(self.tmem_o1_offset)
                tmem_p_base = self.cfg.tmem_p1_offset
            if cutlass.const_expr(self.cfg.single_qkv_instance):
                tmem_p_base = self.tmem_p_base_cached

            # scale_d at trace time.
            # O0 (even counters): HEAD initializes O0, so all
            # subsequent O0 writes (counter 2, 4, ...) always accumulate.
            # O1 (odd counters): first written in LOOP. Dynamic check needed
            # because with loop peeling + causal domain=1, the peeled iteration
            # may be the first O1 write (loop_offset=0 → scale_d=False).
            # TAIL O1: always accumulates because LOOP or the peeled iteration
            # already wrote O1 before TAIL runs.
            if cutlass.const_expr(self.cfg.single_qkv_instance):
                if cutlass.const_expr(self.cfg.has_tmem_p_pipeline):
                    if cutlass.const_expr(section == WorkStage.Loop):
                        scale_d = stage_info.loop_offset > stage_info.loop_start
                    elif cutlass.const_expr(is_tail):
                        scale_d = stage_info.loop_end > stage_info.loop_start
                    else:
                        scale_d = False
                elif cutlass.const_expr(is_tail):
                    scale_d = stage_info.loop_end > 0
                elif cutlass.const_expr(section == WorkStage.Loop):
                    scale_d = stage_info.loop_offset > 0
                else:
                    scale_d = False
            elif cutlass.const_expr(self.cfg.p_in_smem):
                if cutlass.const_expr(is_tail):
                    scale_d = stage_info.loop_end > 0
                else:
                    scale_d = stage_info.loop_offset > 0
            elif cutlass.const_expr(first_o0_write):
                # Head O0 is the first O0 write.
                scale_d = False
            elif cutlass.const_expr(first_o1_write_maybe and section == WorkStage.Tail):
                # TAIL O1 accumulates if LOOP already wrote O1 (domain >= 1).
                # When domain=0, TAIL is the first O1 write.
                scale_d = stage_info.loop_end > 0
            elif cutlass.const_expr(first_o1_write_maybe):
                # Loop O1 initializes on the first iteration and accumulates later.
                scale_d = stage_info.loop_offset > 0
            else:
                # O0 after the head write always accumulates.
                scale_d = True
            # Prevent LLVM from rematerializing V descriptor inside
            # each elect_sync block (same pattern as QK MMA above).
            desc_v_base_ = freeze_smem_descriptor(desc_v_base)

            if cutlass.const_expr(self.cfg.stage_kv_by_head_dim):
                tmem_ptr_o_stage = tmem_ptr_o.subview(head_dim_stage_start * pv_n_dim)
                scale_d_stage = scale_d
                for k_idx in cutlass.range_constexpr(num_kphases_pv):
                    dp = tmem_ptr_raw.subview(tmem_p_base + k_idx * inc_tmem_p)
                    increment = (inc_bytes_v * k_idx) >> 4
                    dv = desc_v_base_ + increment
                    if issue_mma:
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                mma_kind,
                                cta_group,
                                tmem_ptr_o_stage,
                                dp,
                                dv,
                                idesc_pv,
                                scale_d_stage,
                            )
                    scale_d_stage = True
            else:
                if cutlass.const_expr(self.cfg.p_in_smem):
                    smem_p = (
                        self.smem_p0_resource if writes_o0 else self.smem_p1_resource
                    )
                    sP = cutlass.Array(
                        stage_info.context.smem_base.data_ptr() + smem_p.alloc.offset,
                        dtype=cutlass.Int8,
                        shape=(self.cfg.smem_p_bytes,),
                        addrspace=3,
                    )
                    p_lbo, p_sbo = smem_p.descriptor_offsets()
                    desc_p_base_ = freeze_smem_descriptor(
                        prims.Tcgen05SmemDesc.build(
                            sP,
                            leading_byte_offset=p_lbo,
                            stride_byte_offset=p_sbo,
                            layout=smem_p.descriptor_layout(),
                        )
                    )
                    inc_bytes_p = k_dim_per_mma * self.cfg.v_dtype.width // 8
                for head_dim_stage_idx in cutlass.range_constexpr(
                    num_head_dim_stages_to_issue
                ):
                    tmem_ptr_o_stage = tmem_ptr_o.subview(head_dim_stage_idx * pv_n_dim)
                    v_stage_increment = (
                        head_dim_stage_bytes_v * head_dim_stage_idx
                    ) >> 4
                    if cutlass.const_expr(k_half is None):
                        k_lo, k_hi = 0, num_kphases_pv
                    else:
                        k_lo = k_half * (num_kphases_pv // 2)
                        k_hi = k_lo + num_kphases_pv // 2
                    scale_d_stage = scale_d if k_lo == 0 else True
                    for k_idx in cutlass.range_constexpr(k_lo, k_hi):
                        if cutlass.const_expr(self.cfg.p_in_smem):
                            dp = desc_p_base_ + ((inc_bytes_p * k_idx) >> 4)
                        else:
                            dp = tmem_ptr_raw.subview(tmem_p_base + k_idx * inc_tmem_p)
                        increment = v_stage_increment + ((inc_bytes_v * k_idx) >> 4)
                        dv = desc_v_base_ + increment
                        if issue_mma:
                            if prims.elect_sync():
                                prims.tcgen05_mma(
                                    mma_kind,
                                    cta_group,
                                    tmem_ptr_o_stage,
                                    dp,
                                    dv,
                                    idesc_pv,
                                    scale_d_stage,
                                )
                        scale_d_stage = True

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def set_p_base(self, stage_info: StageInfo, *, tmem_p_base: Int32) -> None:
        """Cache the P TMEM column base selected by TmemPResource."""
        _ = stage_info
        self.tmem_p_base_cached = tmem_p_base

    def vc_mean_cta_group(self):
        """cta_group::2 under two-CTA UMMA (M=256 over both CTAs' row sums)."""
        if self.cfg.two_cta_umma:
            return prims.CTAGroup.CTA_2
        return prims.CTAGroup.CTA_1

    @producer_work
    @cute.jit
    def vc_mean_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_mu_base: prims.Tcgen05SmemDesc,
        inst_idx: cutlass.Constexpr[int] = 0,
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """VC-Attention: ``O += sum_i rowsum(P_i) * mean_i`` over the 16 tiles
        before this one for query group ``inst_idx``, two bf16 K=16 UMMA steps
        after that group's PV steps every 16th tile. At tile i this adds the
        group that ended at tile i - 1, so the loop never adds the last group.
        The tail call adds it with the operand loaded after the loop."""
        if cutlass.const_expr(is_tail):
            tile = Int32(stage_info.loop_end)
        else:
            tile = Int32(stage_info.loop_offset)
        group_done = ((tile & Int32(VC_MEAN_GROUP_TILES - 1)) == Int32(0)) & (
            tile > Int32(0)
        )
        self.vc_mean_mma_impl(
            inst_idx, desc_mu_base, self.vc_mean_issue_cached & group_done
        )

    @cute.jit
    def vc_mean_mma_impl(
        self,
        inst_idx: cutlass.Constexpr[int],
        desc_mu_base: prims.Tcgen05SmemDesc,
        issue_mma: Any,
    ) -> None:
        """Issue ``O += sum_i rowsum(P_i) * mean_i`` for query group ``inst_idx`` as
        ``VC_MEAN_OPERANDS`` bf16 K=16 UMMA steps back to back.

        A is the query group's row-sum operands behind its P tile, B the
        TMA-staged mean operands of the tile group.
        """
        if issue_mma:
            if prims.elect_sync():
                if cutlass.const_expr(inst_idx == 0):
                    tmem_ptr_o = self.tmem_ptr_raw_cached.subview(self.tmem_o0_offset)
                else:
                    tmem_ptr_o = self.tmem_ptr_raw_cached.subview(self.tmem_o1_offset)
                desc_l = self.vc_desc_l_cached[inst_idx]
                desc_mu = freeze_smem_descriptor(desc_mu_base)
                for operand in cutlass.range_constexpr(VC_MEAN_OPERANDS):
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        self.vc_mean_cta_group(),
                        tmem_ptr_o,
                        desc_l + ((operand * self.cfg.vc_rowsum_operand_bytes) >> 4),
                        desc_mu
                        + (
                            (
                                operand
                                * self.cfg.vc_mean_operand_bytes
                                // self.cfg.cta_group_size
                            )
                            >> 4
                        ),
                        self.vc_idesc_mu_cached,
                        True,
                    )

    @producer_work
    @cute.jit
    def vc_mean_mma_last(
        self,
        stage_info: StageInfo,
        *,
        desc_mu_last: prims.Tcgen05SmemDesc,
        inst_idx: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Tail variant of ``vc_mean_mma``. Adds the last group, the only one with
        no loop tile after it, from the operand the loader fetched after the loop."""
        self.vc_mean_mma_impl(inst_idx, desc_mu_last, self.vc_mean_issue_cached)

    @cute.jit
    def vc_rowsum_desc(
        self, stage_info: StageInfo, smem_p: SmemPResource
    ) -> prims.Tcgen05SmemDesc:
        """K-major descriptor of a query group's row-sum operands (A of the mean steps)."""
        sL = cutlass.Array(
            stage_info.context.smem_base.data_ptr()
            + smem_p.alloc.offset
            + smem_p.rowsum_tile_offset,
            dtype=cutlass.Int8,
            shape=(self.cfg.vc_rowsum_tile_bytes,),
            addrspace=3,
        )
        return freeze_smem_descriptor(
            prims.Tcgen05SmemDesc.build(
                sL,
                leading_byte_offset=VC_MEAN_TILE_LBO,
                stride_byte_offset=VC_MEAN_TILE_SBO,
                layout=prims.Tcgen05SmemSwizzle.NONE,
            )
        )
