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

"""TMEM O accumulation of the 2CTA MLA kernel."""

import cutlass.cute as cute
import cutlass
from cutlass import Boolean, Float32, Int32
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    consumer_work,
    producer_work,
)
from dataclasses import dataclass, field
from ...device.constants import (
    TCGEN05_32B_REGS_PER_LOAD,
    TCGEN05_32B_SHAPE,
    WARP_LANE_SHIFT,
)
from ...device.math import mma_k_step_for_qkv, mma_kind_for_qkv, qkv_dtype
from ..config import MlaDecodeConfig
from .common import HighThroughputMlaResource
from typing import Any


# =====================================================================
# TmemOResource — O accumulator in TMEM, UmmaProducerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemOResource(HighThroughputMlaResource):
    """TMEM O accumulator.  Producer: Mma (PV MMA).  Consumer: Correction.

    Pipeline: UmmaProducerAsync, 1 stage.
    """

    # Static configuration
    cfg: cutlass.Constexpr = field(default_factory=MlaDecodeConfig)

    # Linked resources
    tmem_corr_ref: Any = None  # Reference to TmemCorrResource for correction data

    # Per-task state
    cta_rank: Any = field(init=False, default=None)
    is_leader: Any = field(init=False, default=None)
    smem_p: Any = None  # SMEM P for PV MMA
    tmem_base_addr: Any = None

    @producer_work
    @cute.jit
    def pv_mma(
        self,
        stage_info: StageInfo,
        *,
        desc_p_base,
        desc_v_base,
        v_subtile_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Issue one PV MMA sub-tile.
        Only leader CTA issues MMA (2CTA UMMA principle).

        Descriptor computation is hoisted OUTSIDE the leader-CTA gate so
        ptxas keeps values in uniform registers.
        """
        cfg = self.cfg

        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            n_dim=cfg.mma_pv_tiler[1],
            m_dim=cfg.mma_pv_tiler[0],
            b_major=1,
        )
        mma_kind = mma_kind_for_qkv(cfg)
        cta_group = prims.CTAGroup.CTA_2
        is_leader_cta = (
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
        )

        sp_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        desc_p_delta = cutlass.const_expr(sp_copy_elems // 8)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            desc_p_delta = cutlass.const_expr(256)
        sv_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[1] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        desc_v_delta = cutlass.const_expr(sv_copy_elems * cfg.qkv_dtype_bytes // 16)

        pv_k_block_count = cutlass.const_expr(
            cfg.mma_pv_tiler[2] // mma_k_step_for_qkv(cfg)
        )

        # Match the producer's physical V-stage decomposition. Each stage
        # carries two adjacent K32 slices for one D256 output panel.
        pv_j = v_subtile_idx // cfg.kv_subtiles_per_stage
        token_partition = v_subtile_idx % cfg.kv_subtiles_per_stage
        tmem_o_base_addr = self.tmem_base_addr + cfg.tmem_o_offset
        tmem_o_addr = tmem_o_base_addr + pv_j * 128

        for stage_subtile_idx in cutlass.range_constexpr(cfg.kv_subtiles_per_stage):
            pv_i = token_partition * cfg.kv_subtiles_per_stage + stage_subtile_idx
            desc_p = desc_p_base + pv_i * desc_p_delta
            desc_v = desc_v_base + stage_subtile_idx * desc_v_delta

            # Clear O for each output-N panel on its first PV K slice, then
            # accumulate the remaining slices and subsequent sequence tiles.
            scale_d_pv = Boolean(True)
            if cutlass.const_expr(pv_i == 0):
                if cutlass.const_expr(is_tail):
                    scale_d_pv = Boolean(
                        stage_info.loop_end > Int32(stage_info.loop_start)
                    )
                else:
                    scale_d_pv = Boolean(
                        stage_info.loop_offset != Int32(stage_info.loop_start)
                    )
            if is_leader_cta:
                for k_block in cutlass.range_constexpr(pv_k_block_count):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            mma_kind,
                            cta_group,
                            prims.make_tmem_ptr(tmem_o_addr, Float32),
                            desc_p + k_block * 2,
                            desc_v + k_block * (256 if cfg.is_fp8_qkv() else 128),
                            idesc_pv,
                            Boolean(scale_d_pv),
                        )
                    scale_d_pv = Boolean(True)

    @producer_work
    @cute.jit
    def pv_mma_n_major(
        self,
        stage_info: StageInfo,
        *,
        desc_p_base,
        desc_v_base,
        pv_n_idx: cutlass.Constexpr[int],
        pv_k_idx: cutlass.Constexpr[int],
        is_tail: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Issue PV MMA in output-N-major order for per-slice O tokens."""
        cfg = self.cfg

        pv_j = pv_n_idx
        pv_i = pv_k_idx
        tmem_o_base_addr = self.tmem_base_addr + cfg.tmem_o_offset
        tmem_o_addr = tmem_o_base_addr + pv_j * 128

        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=qkv_dtype(cfg),
            b_dtype=qkv_dtype(cfg),
            n_dim=cfg.mma_pv_tiler[1],
            m_dim=cfg.mma_pv_tiler[0],
            b_major=1,
        )
        mma_kind = mma_kind_for_qkv(cfg)
        cta_group = prims.CTAGroup.CTA_2
        is_leader_cta = (
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster()) == 0
        )

        sp_copy_elems = cutlass.const_expr(
            cfg.mma_pv_tiler[0] // cfg.num_mma_ctas * cfg.mma_pv_tiler[2]
        )
        desc_p_delta = cutlass.const_expr(sp_copy_elems // 8)
        if cutlass.const_expr(cfg.is_fp8_qkv()):
            desc_p_delta = cutlass.const_expr(256)
        desc_p = desc_p_base + pv_i * desc_p_delta

        pv_k_block_count = cutlass.const_expr(
            cfg.mma_pv_tiler[2] // mma_k_step_for_qkv(cfg)
        )
        scale_d_pv = Boolean(True)
        if cutlass.const_expr(pv_i == 0):
            if cutlass.const_expr(is_tail):
                scale_d_pv = Boolean(stage_info.loop_end > Int32(stage_info.loop_start))
            else:
                scale_d_pv = Boolean(
                    stage_info.loop_offset != Int32(stage_info.loop_start)
                )

        if is_leader_cta:
            for k_block in cutlass.range_constexpr(pv_k_block_count):
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        mma_kind,
                        cta_group,
                        prims.make_tmem_ptr(tmem_o_addr, Float32),
                        desc_p + k_block * 2,
                        desc_v_base + k_block * (256 if cfg.is_fp8_qkv() else 128),
                        idesc_pv,
                        Boolean(scale_d_pv),
                    )
                scale_d_pv = Boolean(True)

    @consumer_work
    @cute.jit
    def rescale_o(
        self, stage_info: StageInfo, *, correction_factor, no_correction
    ) -> None:
        """Rescale O in-place in TMEM by correction_factor."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]

        # Use local tidx within 4-warp correction group (matching bare-metal)
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        tmem_warp_row_id = (
            self.tmem_base_addr
            + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | cfg.tmem_o_offset

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_ops = 4  # 4 loads x 32 = 128 elements per iter_n

        skip_correction = prims.vote_sync(
            cute.arch.FULL_MASK,
            no_correction == 1,
            prims.VoteSync.ALL,
        )

        if not skip_correction:
            for iter_n in cutlass.range_constexpr(cfg.iterations_pv_n):
                tmem_addr_offset = iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
                for idx in cutlass.range_constexpr(num_tmem_ops):
                    curr_addr = (
                        tmem_raw_addr
                        + tmem_addr_offset
                        + idx * TCGEN05_32B_REGS_PER_LOAD
                    )
                    tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                    chunk = prims.tcgen05_ld(
                        t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                    )
                    scaled = chunk * cutlass.full_like(chunk, correction_factor)
                    prims.tcgen05_st(t2r_shape, tmem_ptr, scaled)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        cute.arch.fence_view_async_tmem_store()

    @consumer_work
    @cute.jit
    def rescale_o_slice(
        self,
        stage_info: StageInfo,
        *,
        correction_factor,
        no_correction,
        iter_n: cutlass.Constexpr[int],
    ) -> None:
        """Rescale one PV output-N slice after its O token is ready."""
        cfg = self.cfg
        tidx = cute.arch.thread_idx()[0]
        local_tidx = tidx % (cfg.num_compute_warps * cfg.threads_per_warp)
        tmem_warp_row_id = (
            self.tmem_base_addr
            + (local_tidx >> WARP_LANE_SHIFT) * TCGEN05_32B_REGS_PER_LOAD
        )
        tmem_raw_addr = (tmem_warp_row_id << 16) | (
            cfg.tmem_o_offset + iter_n * (cfg.mma_pv_tiler[1] // cfg.warps_in_n)
        )

        t2r_shape = TCGEN05_32B_SHAPE
        num_tmem_ops = 4
        skip_correction = prims.vote_sync(
            cute.arch.FULL_MASK,
            no_correction == 1,
            prims.VoteSync.ALL,
        )
        if not skip_correction:
            for idx in cutlass.range_constexpr(num_tmem_ops):
                curr_addr = tmem_raw_addr + idx * TCGEN05_32B_REGS_PER_LOAD
                tmem_ptr = prims.make_tmem_ptr(curr_addr, Float32)
                chunk = prims.tcgen05_ld(
                    t2r_shape, tmem_ptr, num=TCGEN05_32B_REGS_PER_LOAD
                )
                scaled = chunk * cutlass.full_like(chunk, correction_factor)
                prims.tcgen05_st(t2r_shape, tmem_ptr, scaled)

        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        cute.arch.fence_view_async_tmem_store()
