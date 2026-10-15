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

"""TMEM score resources for QK MMA and online softmax."""

from dataclasses import dataclass
from typing import ClassVar, Optional

from cutlass.experimental import primitives as prims

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation, TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...helpers.constants import (
    OCTET_LANES,
    QUAD_LANE_MASK,
    QUAD_LANE_SHIFT,
    SCORE_ROWS_PER_Q_PAIR,
    SCORE_TOKENS_PER_QK_GROUP,
    SMEM_WORD_BYTES,
    TCGEN05_16X256B_SHAPE,
    TCGEN05_DESC_NEXT_K_BLOCK_UNITS,
    TCGEN05_DESC_WRAPPED_K_BLOCK_UNITS,
    WARP_LANES,
    WARPGROUP_THREADS,
    WARPGROUP_WARPS,
)


from ...helpers.layout import (
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    decode_gen_task_cache,
    num_q_repeats,
    num_s_regs_per_thread,
    num_softmax_scale_groups,
    q_p_desc_k_block_wrap_units,
    q_stage_smem_element_offset,
    smem_array,
    softmax_scratch_words,
)
from ...helpers.math import (
    ffma2,
    float_to_u32_for_atomic_max,
    init_softmax_scratch_u32,
    mma_k_step_for_qkv,
    mma_kind_for_qkv,
    neg_max_f32,
    qkv_dtype,
    smem_atomic_max_u32,
    u32_to_float_for_atomic_max,
)
from ...helpers.mask import MaskType
from ...helpers.ops import (
    float_to_u32_bits,
    freeze_smem_descriptor,
    softmax_sum_state_ptr,
    tcgen05_ld_16x32bx2_f32,
    tcgen05_second_panel_addr,
    u32_bits_to_float,
)
from ...helpers.stage import MlaStage
from ...helpers.tile import (
    local_kv_tile_idx,
    batch_idx_for_stage_cfg,
    cta_idx_kv_for_stage,
    cta_idx_q_for_stage,
    global_kv_tile_idx,
    runtime_seq_len_kv_for_query_row,
    runtime_seq_len_kv_from_task_cache,
    softmax_kv_tile_idx,
)

# Column-split max exchange: [parity 2][half 2][64 head rows] words after the
# softmax scratch.
XBAR_ROWS = 64
XBAR_WORDS = 2 * 2 * XBAR_ROWS
XBAR_BARRIER_ID = 2
from .common import (
    MlaResource,
)

# =====================================================================
# TmemSResource — S scores in TMEM, UmmaProducerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemSResource(MlaResource):
    """TMEM score resource plus task-local online-softmax state."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("old_max_arr", cutlass.Array, None, "Previous running softmax maxima."),
        ("sum_arr", cutlass.Array, None, "Running softmax denominators."),
        ("new_max_arr", cutlass.Array, None, "Current running softmax maxima."),
        ("local_sum_arr", cutlass.Array, None, "Local denominator contributions."),
        ("s_arr", cutlass.Array, None, "Loaded S scores for the current tile."),
        ("regs_p", cutlass.Array, None, "Packed P registers (compute_p_s)."),
    )
    inst_id: cutlass.Constexpr[int] = 0
    scale_softmax_log2: Float32 = None
    # threshold / scale_softmax_log2; zero disables the row-max freeze.
    adjusted_skip_corr_threshold: Float32 = None
    p_ref: Optional[MemoryResource] = None
    global_ref: Optional[MemoryResource] = None
    cache_seqs: object = None
    head_idx: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    cta_idx_kv: object = None
    sync_barrier_id: cutlass.Constexpr[int] = 0
    # Token-sparse: the page-offset resource whose allocation carries the
    # per-tile validity ring (see ``SPARSE_MASK_SLOTS`` in smem_resources).
    page_offsets_kv: object = None
    _sparse_mask_words: object = None
    q_desc_current: object = None
    q_desc_rope_current: object = None
    # Native FP4 QK: NVFP4 Q descriptors and the SFA / SFB TMEM resources.
    q_desc_fp4_lat0: object = None
    q_desc_fp4_lat1: object = None
    q_desc_fp4_rope: object = None
    qfp4_ref: Optional[MemoryResource] = None
    raw_ref: Optional[MemoryResource] = None
    new_max_state: object = None
    sum_state: object = None
    _scratch_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _softmax_scratch: object = None
    _p_local_sum_arr: object = None
    old_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    sum_arr: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    new_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    local_sum_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    s_arr: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    regs_p: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    @cute.jit
    def _reset_softmax_state(self):
        """Reset per-resource softmax max and sum state arrays."""
        num_scale_groups = num_softmax_scale_groups(self.cfg)
        self.new_max_state = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        self.sum_state = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        for idx in cutlass.range_constexpr(num_scale_groups):
            self.new_max_state[idx] = neg_max_f32()
            self.sum_state[idx] = Float32(0.0)

    def get_smem_requirements(self):
        """Return the softmax scratch SMEM allocation."""
        if self._scratch_alloc is None:
            # Native FP4 QK: the column-split max exchange words follow the scratch.
            xbar_bytes = (
                XBAR_WORDS * SMEM_WORD_BYTES
                if self.cfg.is_dynamic_token_sparse_nvfp4
                else 0
            )
            self._scratch_alloc = SmemAllocation(
                name=f"{self.name}_softmaxScratch",
                size_bytes=softmax_scratch_words(self.cfg) * SMEM_WORD_BYTES
                + xbar_bytes,
                alignment=16,
            )
        return [self._scratch_alloc]

    def _s_tmem_offset(self) -> int:
        return self._tmem_alloc.offset

    def get_tmem_requirements(self):
        """Return the TMEM allocation for score tiles."""
        if self._tmem_alloc is None:
            num_stages = (
                self.pipeline_config.num_stages
                if self.cfg.kernel_variant == "keeps_mma_ab"
                and self.pipeline_config is not None
                else 1
            )
            self._tmem_alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=self.cfg.tmem_s_cols * num_stages,
            )
        return [self._tmem_alloc]

    @cute.jit
    def _make_initial_softmax_vars(self):
        """Create fresh softmax state arrays for the current work tile."""
        self._reset_softmax_state()
        num_scale_groups = num_softmax_scale_groups(self.cfg)
        num_s_regs = num_s_regs_per_thread(self.cfg)
        self._p_local_sum_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        old_max_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        sum_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        new_max_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        local_sum_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        s_arr = cutlass.Array(Float32, num_s_regs, space=cutlass.AddressSpace.rmem)
        for idx in cutlass.range_constexpr(num_scale_groups):
            old_max_arr[idx] = neg_max_f32()
            sum_arr[idx] = Float32(0.0)
            new_max_arr[idx] = neg_max_f32()
            local_sum_arr[idx] = Float32(0.0)
            self._p_local_sum_arr[idx] = Float32(0.0)
        for idx in cutlass.range_constexpr(num_s_regs):
            s_arr[idx] = neg_max_f32()
        return old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr),
    )
    @cute.jit
    def init_softmax_state(self, stage_info: StageInfo):
        """Create softmax arrays and scratch state for the first work tile."""
        # Consumer aux work belongs to the softmax task.  It creates the local
        # arrays that will be threaded through update_softmax(),
        # materialize_p(), and update_softmax_sum().
        context = stage_info.context
        self._init_tmem_state(stage_info)
        self._softmax_scratch = smem_array(
            context,
            self._scratch_alloc,
            Uint32,
            softmax_scratch_words(self.cfg)
            + (XBAR_WORDS if self.cfg.is_dynamic_token_sparse_nvfp4 else 0),
        )
        self.q_desc_current = Int64(0)
        self.q_desc_rope_current = Int64(0)
        if cutlass.const_expr(self.cfg.is_dynamic_token_sparse):
            self._sparse_mask_words = smem_array(
                context,
                self.page_offsets_kv._alloc,
                Int32,
                self.page_offsets_kv.total_words,
            )
        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            thread_idx = cute.arch.thread_idx()[0]
            state_ptr = self._softmax_scratch.data_ptr(thread_idx)
            state_ptr.store(
                float_to_u32_bits(neg_max_f32()),
                is_volatile=True,
                alignment=4,
            )
            softmax_sum_state_ptr(state_ptr).store(
                float_to_u32_bits(Float32(0.0)),
                is_volatile=True,
                alignment=4,
            )
        return self._make_initial_softmax_vars()

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr),
    )
    @cute.jit
    def init_softmax_work_tile_state(self, stage_info: StageInfo):
        """Create fresh softmax arrays for each persistent work tile."""
        # Persistent schedules reuse the same task graph for multiple work
        # tiles, so the consumer-side softmax state must be reset per tile.
        del stage_info
        return self._make_initial_softmax_vars()

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def reset_softmax_work_tile_state(self, stage_info: StageInfo) -> None:
        """Reset per-resource softmax state for a persistent work tile."""
        # Producer aux work mirrors the consumer reset for schedules where this
        # resource later produces QK scores into TMEM.
        del stage_info
        self._make_initial_softmax_vars()

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def set_q_desc(self, stage_info: StageInfo, *, q_desc, q_desc_rope):
        """Cache the Q descriptor once while the Q SMEM stage is live."""
        # q_desc() returns descriptors from the SmemQ consumer side.  This
        # producer aux method stores them in the score resource so each QK MMA
        # call can reuse them without adding another schedule edge.
        del stage_info
        self.q_desc_current = q_desc
        self.q_desc_rope_current = q_desc_rope

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def set_q_desc_fp4(
        self, stage_info: StageInfo, *, q_desc_lat0, q_desc_lat1, q_desc_rope
    ):
        """Cache the three NVFP4 Q descriptors (native FP4 QK) for the work tile."""
        del stage_info
        self.q_desc_fp4_lat0 = q_desc_lat0
        self.q_desc_fp4_lat1 = q_desc_lat1
        self.q_desc_fp4_rope = q_desc_rope

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_native_qk_state(self, stage_info: StageInfo) -> None:
        """MMA-side views of the raw K ring and the SF TMEM blocks (native FP4 QK)."""
        self.raw_ref._init_smem_state(stage_info)
        self.qfp4_ref._init_tmem_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def stage_sfb(self, stage_info: StageInfo, *, section: cutlass.Constexpr[MlaStage]):
        """Native FP4 QK: wait for this K tile's raw stage to land (observer spin).

        The MMA is not a registered consumer of the raw K ring (the dequant
        warps are); the tile's token block scales (SFB) are written into TMEM
        by the dequant warps (``SfbReadyResource``), which the MMA waits on
        right after this.  Issued ahead of PV(n) so the wait overlaps the PV
        MMAs instead of sitting between PV(n) and BMM1(n + 1).
        """
        cfg = self.cfg
        tile = local_kv_tile_idx(cfg, stage_info, self.inst_id, 0, section=section)
        self.raw_ref.observe_k_tile_full(tile)

    @producer_work
    @cute.jit
    def qk_mma_fp4(
        self,
        stage_info: StageInfo,
        *,
        section: cutlass.Constexpr[MlaStage],
    ):
        """Native FP4 QK: ten ``kind::mxf4nvf4`` M128 N128 K64 MMAs for one K tile.

        Chunks 0-3 / 4-7 read the two SW128 latent segments, chunk 8 the rope
        main half and chunk 9 the rope residual half of the SW64 segment against
        the same Q rope (Q rope is stored twice).  Block scales come from TMEM,
        4 columns per chunk: SFA (Q, staged once per work tile by ``q_desc``)
        and SFB (tokens, written by the dequant warps with ``tcgen05.st`` and
        made visible through ``SfbReadyResource.sfb_visible``).
        """
        cfg = self.cfg
        task_cache = decode_gen_task_cache(stage_info)
        tmem_base = task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
        tmem_ptr = prims.make_tmem_ptr(
            tmem_base
            + Int32(self._tmem_alloc.offset)
            + stage_info.stage_idx * Int32(cfg.tmem_s_cols),
            Float32,
        )
        sfa_col = tmem_base + Int32(self.qfp4_ref._tmem_alloc.offset)
        sfb_col = sfa_col + Int32(cfg.tmem_sf_cols)
        idesc = prims.Tcgen05MxOmmaInstrDesc.build(
            a_dtype=cutlass.Float4E2M1FN,
            b_dtype=cutlass.Float4E2M1FN,
            scale_format=0,  # UE4M3 block scales
            a_major=0,
            b_major=0,
            n_dim=cfg.tile_size_kv,
            m_dim=128,
            sparsity_version=0,
        )
        q_descs = (
            freeze_smem_descriptor(self.q_desc_fp4_lat0),
            freeze_smem_descriptor(self.q_desc_fp4_lat1),
            freeze_smem_descriptor(self.q_desc_fp4_rope),
        )
        # The MMA is not a registered consumer of the raw K ring (the dequant
        # warps are; they release a stage only after this tile's BMM1 completed):
        # ``stage_sfb`` observed the tile's full barrier; address its stage directly.
        tile = local_kv_tile_idx(cfg, stage_info, self.inst_id, 0, section=section)
        raw = self.raw_ref
        seg_bytes = cfg.tile_size_kv * 128
        k_base = raw.smem_byte_address(
            (tile % Int32(cfg.kv_raw_stages)) * Int32(cfg.kv_raw_stage_bytes)
        )
        k_descs = (
            freeze_smem_descriptor(
                prims.Tcgen05SmemDesc.build(
                    k_base,
                    leading_byte_offset=seg_bytes,
                    stride_byte_offset=1024,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                )
            ),
            freeze_smem_descriptor(
                prims.Tcgen05SmemDesc.build(
                    k_base + Int32(seg_bytes),
                    leading_byte_offset=seg_bytes,
                    stride_byte_offset=1024,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                )
            ),
            freeze_smem_descriptor(
                prims.Tcgen05SmemDesc.build(
                    k_base + Int32(2 * seg_bytes),
                    leading_byte_offset=cfg.tile_size_kv * 64,
                    stride_byte_offset=512,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
                )
            ),
        )
        if prims.elect_sync():
            for c in cutlass.range_constexpr(10):
                seg = 0 if c < 4 else (1 if c < 8 else 2)
                blk = c - 4 * seg
                # +32 B per K64 block inside the swizzled row (16-byte units).
                a_desc = q_descs[seg] + Int64(2 * blk)
                b_desc = k_descs[seg] + Int64(2 * blk)
                prims.tcgen05_mma_block_scale(
                    prims.Tcgen05MMAKind.MXF4NVF4,
                    prims.CTAGroup.CTA_1,
                    tmem_ptr,
                    a_desc,
                    b_desc,
                    idesc,
                    Boolean(c != 0),
                    prims.make_tmem_ptr(sfa_col + Int32(4 * c), Int32),
                    prims.make_tmem_ptr(sfb_col + Int32(4 * c), Int32),
                    scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                )

    @producer_work
    @cute.jit
    def qk_mma(
        self,
        stage_info: StageInfo,
        *,
        kv_desc,
        k_subtile_idx: cutlass.Constexpr[int],
    ):
        """Issue the staged QK MMA for one 128-wide MLA head-dim slice."""
        # Producer work for TmemSResource: the MMA warp writes score fragments
        # into the acquired TMEM score stage.  The paired softmax task consumes
        # this TMEM stage through update_softmax().
        cfg = self.cfg
        qk_stage_idx = k_subtile_idx
        q_stage_offset_bytes = Int32(
            q_stage_smem_element_offset(cfg, qk_stage_idx) * cfg.qkv_dtype_bytes
        )
        q_desc = freeze_smem_descriptor(self.q_desc_current)
        if cutlass.const_expr(cfg.is_fp8_qkv() and cfg.rope_dim == 64):
            if cutlass.const_expr(
                qk_stage_idx == cfg.latent_dim // cfg.head_dim_per_stage_kv
            ):
                q_desc = freeze_smem_descriptor(self.q_desc_rope_current)
        kv_desc = freeze_smem_descriptor(kv_desc)
        if cutlass.const_expr(
            not cfg.is_fp8_qkv()
            or cfg.rope_dim != 64
            or qk_stage_idx != cfg.latent_dim // cfg.head_dim_per_stage_kv
        ):
            q_desc = q_desc + (q_stage_offset_bytes >> 4)

        task_cache = decode_gen_task_cache(stage_info)
        stage_col_offset = Int32(0)
        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            stage_col_offset = stage_info.stage_idx * Int32(cfg.tmem_s_cols)
        tmem_ptr = prims.make_tmem_ptr(
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._tmem_alloc.offset)
            + stage_col_offset,
            Float32,
        )
        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=qkv_dtype(cfg),
                b_dtype=qkv_dtype(cfg),
                n_dim=cfg.tile_size_kv,
                m_dim=cfg.tile_size_q,
            )
        else:
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=qkv_dtype(cfg),
                b_dtype=qkv_dtype(cfg),
                n_dim=cfg.tile_size_q,
                m_dim=cfg.tile_size_kv,
            )
        k_block_count = cfg.qk_head_stage_width(qk_stage_idx) // mma_k_step_for_qkv(cfg)
        if prims.elect_sync():
            scale_d = Boolean(qk_stage_idx != 0)
            for k_block in cutlass.range_constexpr(k_block_count):
                if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
                    prims.tcgen05_mma(
                        mma_kind_for_qkv(cfg),
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        q_desc,
                        kv_desc,
                        idesc,
                        scale_d,
                    )
                else:
                    prims.tcgen05_mma(
                        mma_kind_for_qkv(cfg),
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        kv_desc,
                        q_desc,
                        idesc,
                        scale_d,
                    )
                scale_d = Boolean(True)
                if cutlass.const_expr(k_block + 1 < k_block_count):
                    if cutlass.const_expr(k_block == 3):
                        kv_desc = kv_desc + Int32(TCGEN05_DESC_WRAPPED_K_BLOCK_UNITS)
                        q_desc = q_desc + Int32(q_p_desc_k_block_wrap_units(cfg))
                    else:
                        kv_desc = kv_desc + Int32(TCGEN05_DESC_NEXT_K_BLOCK_UNITS)
                        q_desc = q_desc + Int32(TCGEN05_DESC_NEXT_K_BLOCK_UNITS)

    @consumer_work(returns=(old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr))
    @cute.jit
    def update_softmax(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        sum_arr,
        new_max_arr,
        local_sum_arr,
        s_arr,
        section: cutlass.Constexpr[MlaStage],
    ):
        """Read S from TMEM and update the flat-query softmax state."""
        # Consumer work for TmemSResource: softmax waits for the QK MMA score
        # stage, loads S from TMEM, applies masks, and returns updated local
        # softmax state to the captured schedule.
        cfg = self.cfg
        task_cache = decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            warp_grp_thread_idx = cute.arch.thread_idx()[0]
        q_repeats = num_q_repeats(cfg)
        num_scale_groups = num_softmax_scale_groups(cfg)
        old_max_vals = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        sum_vals = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        new_max_vals = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        local_max_vals = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        s_vals = cutlass.Array(
            Float32, num_s_regs_per_thread(cfg), space=cutlass.AddressSpace.rmem
        )

        for idx in cutlass.range_constexpr(num_scale_groups):
            if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
                state_ptr = self._softmax_scratch.data_ptr(warp_grp_thread_idx)
                old_max_vals[idx] = u32_bits_to_float(
                    state_ptr.load(is_volatile=True, alignment=4)
                )
                sum_vals[idx] = u32_bits_to_float(
                    softmax_sum_state_ptr(state_ptr).load(is_volatile=True, alignment=4)
                )
            else:
                old_max_vals[idx] = new_max_arr[idx]
                sum_vals[idx] = sum_arr[idx]
            new_max_vals[idx] = old_max_vals[idx]
            local_max_vals[idx] = neg_max_f32()
        for idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
            s_vals[idx] = neg_max_f32()

        # Normalize both score layouts into one local S register array before
        # masking: keeps-MMA-AB reads a single TMEM panel, swaps-MMA-AB reads
        # two 16x256b panels.
        stage_col_offset = Int32(0)
        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            stage_col_offset = stage_info.stage_idx * Int32(cfg.tmem_s_cols)
        base_addr = (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._s_tmem_offset())
            + stage_col_offset
            # Native FP4 QK column split: this 128-thread half's 64 columns.
            + (
                (task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX] >> Int32(7))
                if cutlass.const_expr(cfg.is_dynamic_token_sparse_nvfp4)
                else Int32(0)
            )
            * Int32(cfg.tile_size_kv // 2)
        )
        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            loaded = tcgen05_ld_16x32bx2_f32(
                prims.make_tmem_ptr(base_addr, Float32),
                num=num_s_regs_per_thread(cfg),
                offset=Int32(num_s_regs_per_thread(cfg)),
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            for reg_idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
                s_vals[reg_idx] = loaded[reg_idx]
        else:
            loaded0 = prims.tcgen05_ld(
                TCGEN05_16X256B_SHAPE,
                prims.make_tmem_ptr(base_addr, Float32),
                num=q_repeats,
            )
            loaded1 = prims.tcgen05_ld(
                TCGEN05_16X256B_SHAPE,
                prims.make_tmem_ptr(tcgen05_second_panel_addr(base_addr), Float32),
                num=q_repeats,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            for repeat_idx in cutlass.range_constexpr(q_repeats):
                ld_base = repeat_idx * 4
                s_vals[ld_base + 0] = loaded0[ld_base + 0]
                s_vals[ld_base + 1] = loaded0[ld_base + 1]
                s_vals[ld_base + 2] = loaded0[ld_base + 2]
                s_vals[ld_base + 3] = loaded0[ld_base + 3]
                s_vals[q_repeats * 4 + ld_base + 0] = loaded1[ld_base + 0]
                s_vals[q_repeats * 4 + ld_base + 1] = loaded1[ld_base + 1]
                s_vals[q_repeats * 4 + ld_base + 2] = loaded1[ld_base + 2]
                s_vals[q_repeats * 4 + ld_base + 3] = loaded1[ld_base + 3]

        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            cfg,
            task_cache,
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        if cutlass.const_expr(cfg.is_dynamic_token_sparse_nvfp4):
            # Head / Loop sections on the K cadence (tile inst, then
            # n * num_insts_kv + num_insts_kv + inst).
            local_tile_idx = local_kv_tile_idx(
                cfg, stage_info, self.inst_id, 0, section=section
            )
        else:
            local_tile_idx = softmax_kv_tile_idx(cfg, stage_info, self.inst_id)
        tile_idx = global_kv_tile_idx(cfg, local_tile_idx, seq_len_kv, cta_idx_kv)
        tile_offset_k = tile_idx * Int32(cfg.tile_size_kv)
        next_tile_offset_k = tile_offset_k + Int32(cfg.tile_size_kv)
        should_apply_dense_mask = (
            (seq_len_kv % Int32(cfg.tile_size_kv)) != Int32(0)
        ) or (next_tile_offset_k > seq_len_kv)
        if cutlass.const_expr(cfg.is_dynamic_token_sparse):
            # Every tile may carry -1 (empty) slots; the validity ring is
            # consulted unconditionally.
            should_apply_dense_mask = cutlass.Boolean(True)
        needs_row_causal_mask = cutlass.const_expr(
            cfg.mask_type == MaskType.CAUSAL.value and cfg.logical_seq_len_q > 1
        )
        if cutlass.const_expr(needs_row_causal_mask):
            min_seq_len_kv = runtime_seq_len_kv_for_query_row(
                cfg,
                self.cache_seqs,
                batch_idx,
                cta_idx_q,
                Int32(0),
                self.cu_seqlens_q,
            )
            should_apply_dense_mask = should_apply_dense_mask or (
                next_tile_offset_k > min_seq_len_kv
            )
        if should_apply_dense_mask:
            # The CTA domain follows the latest logical query row in the flat
            # tile. Earlier rows can have a narrower bottom-right causal limit.
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
                local_col_base = (lane_idx >> Int32(4)) * Int32(
                    num_s_regs_per_thread(cfg)
                ) + (
                    (task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX] >> Int32(7))
                    if cutlass.const_expr(cfg.is_dynamic_token_sparse_nvfp4)
                    else Int32(0)
                ) * Int32(cfg.tile_size_kv // 2)
                local_row_idx = (warp_idx & Int32(3)) * Int32(16) + (
                    lane_idx & Int32(0xF)
                )
                row_seq_len_kv = seq_len_kv
                if cutlass.const_expr(needs_row_causal_mask):
                    row_seq_len_kv = runtime_seq_len_kv_for_query_row(
                        cfg,
                        self.cache_seqs,
                        batch_idx,
                        cta_idx_q,
                        local_row_idx,
                        self.cu_seqlens_q,
                    )
                if cutlass.const_expr(
                    not (
                        cfg.is_dynamic_token_sparse_nvfp4 and not needs_row_causal_mask
                    )
                ):
                    # NVFP4: the loader folds this test into the validity nibble.
                    for reg_idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
                        token_idx = tile_offset_k + local_col_base + Int32(reg_idx)
                        if token_idx >= row_seq_len_kv:
                            s_vals[reg_idx] = neg_max_f32()
                if cutlass.const_expr(cfg.is_dynamic_token_sparse):
                    from .smem_resources import SPARSE_MASK_SLOTS, SPARSE_MASK_WORDS

                    half_cols = cfg.tile_size_kv // 2
                    hi_half = local_col_base >= Int32(half_cols)
                    if cutlass.const_expr(cfg.is_dynamic_token_sparse_nvfp4):
                        # Loader-owned indices: one byte per quad (bit = token & 3),
                        # 32 bytes = 8 words per tile slot; this thread's row span
                        # is words (local_col_base >> 4) + {0 .. words - 1}.
                        words_per_thread = num_s_regs_per_thread(cfg) // 16
                        mask_base = (
                            Int32(self.page_offsets_kv.sparse_mask_word_base)
                            + (tile_idx % Int32(SPARSE_MASK_SLOTS)) * Int32(8)
                            + (local_col_base >> Int32(4))
                        )
                        mask_sel = []
                        for w in cutlass.range_constexpr(words_per_thread):
                            mask_sel.append(
                                Int32(self._sparse_mask_words[mask_base + Int32(w)])
                            )
                        # Every quad byte holds its 4 validity bits in the low
                        # nibble (bit = token & 3 of quad (token >> 2) & 3).
                        for reg_idx in cutlass.range_constexpr(
                            num_s_regs_per_thread(cfg)
                        ):
                            mask_word = mask_sel[reg_idx >> 4]
                            bit = (reg_idx & 3) | (((reg_idx >> 2) & 3) << 3)
                            if ((mask_word >> Int32(bit)) & Int32(1)) == Int32(0):
                                s_vals[reg_idx] = neg_max_f32()
                    else:
                        mask_base = Int32(
                            self.page_offsets_kv.sparse_mask_word_base
                        ) + (tile_idx % Int32(SPARSE_MASK_SLOTS)) * Int32(
                            SPARSE_MASK_WORDS
                        )
                        mask_w0 = Int32(self._sparse_mask_words[mask_base])
                        mask_w1 = Int32(self._sparse_mask_words[mask_base + Int32(1)])
                        mask_w2 = Int32(self._sparse_mask_words[mask_base + Int32(2)])
                        mask_w3 = Int32(self._sparse_mask_words[mask_base + Int32(3)])
                        # Word w covers tokens 32w .. 32w+31 (bit = token & 31).
                        # ``local_col_base`` is 0 or 64, so this thread's 64 tokens
                        # use words (local_col_base >> 5) + {0, 1}.
                        word_a = mask_w2 if hi_half else mask_w0
                        word_b = mask_w3 if hi_half else mask_w1
                        for reg_idx in cutlass.range_constexpr(
                            num_s_regs_per_thread(cfg)
                        ):
                            mask_word = word_b if reg_idx >= 32 else word_a
                            if ((mask_word >> Int32(reg_idx & 31)) & Int32(1)) == Int32(
                                0
                            ):
                                s_vals[reg_idx] = neg_max_f32()
            else:
                local_idx_k0 = warp_idx * Int32(WARP_LANES) + (
                    lane_idx >> Int32(QUAD_LANE_SHIFT)
                )
            if cutlass.const_expr(cfg.kernel_variant != "keeps_mma_ab"):
                # Score TMEM is read as two panels: each repeat contributes
                # the first two 8-token groups, then the second panel carries
                # the next two groups.
                local_q_pair = (lane_idx & Int32(QUAD_LANE_MASK)) * Int32(
                    SCORE_ROWS_PER_Q_PAIR
                )
                for repeat_idx in cutlass.range_constexpr(q_repeats):
                    row_in_tile_0 = (
                        Int32(repeat_idx * SCORE_TOKENS_PER_QK_GROUP) + local_q_pair
                    )
                    row_in_tile_1 = row_in_tile_0 + Int32(1)
                    seq_len_kv_0 = seq_len_kv
                    seq_len_kv_1 = seq_len_kv
                    if cutlass.const_expr(needs_row_causal_mask):
                        seq_len_kv_0 = runtime_seq_len_kv_for_query_row(
                            cfg,
                            self.cache_seqs,
                            batch_idx,
                            cta_idx_q,
                            row_in_tile_0,
                            self.cu_seqlens_q,
                        )
                        seq_len_kv_1 = runtime_seq_len_kv_for_query_row(
                            cfg,
                            self.cache_seqs,
                            batch_idx,
                            cta_idx_q,
                            row_in_tile_1,
                            self.cu_seqlens_q,
                        )
                    s_base = repeat_idx * 4
                    s_second_panel_base = q_repeats * 4 + s_base
                    token_idx = tile_offset_k + local_idx_k0
                    if token_idx >= seq_len_kv_0:
                        s_vals[s_base + 0] = neg_max_f32()
                    if token_idx >= seq_len_kv_1:
                        s_vals[s_base + 1] = neg_max_f32()
                    token_idx = (
                        tile_offset_k + local_idx_k0 + Int32(SCORE_TOKENS_PER_QK_GROUP)
                    )
                    if token_idx >= seq_len_kv_0:
                        s_vals[s_base + 2] = neg_max_f32()
                    if token_idx >= seq_len_kv_1:
                        s_vals[s_base + 3] = neg_max_f32()
                    token_idx = (
                        tile_offset_k
                        + local_idx_k0
                        + Int32(2 * SCORE_TOKENS_PER_QK_GROUP)
                    )
                    if token_idx >= seq_len_kv_0:
                        s_vals[s_second_panel_base + 0] = neg_max_f32()
                    if token_idx >= seq_len_kv_1:
                        s_vals[s_second_panel_base + 1] = neg_max_f32()
                    token_idx = (
                        tile_offset_k
                        + local_idx_k0
                        + Int32(3 * SCORE_TOKENS_PER_QK_GROUP)
                    )
                    if token_idx >= seq_len_kv_0:
                        s_vals[s_second_panel_base + 2] = neg_max_f32()
                    if token_idx >= seq_len_kv_1:
                        s_vals[s_second_panel_base + 3] = neg_max_f32()

        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            # Keeps-MMA-AB keeps the softmax state in scratch words indexed by
            # CTA thread, so only a warp-level max is needed here.
            # Column split (native FP4 QK): the tile's raw half-row max only;
            # the exchange below folds in the running max and applies the
            # threshold freeze.
            max_seed = old_max_vals[0]
            if cutlass.const_expr(cfg.is_dynamic_token_sparse_nvfp4):
                max_seed = neg_max_f32()
            local_max = max_seed
            for reg_idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
                local_max = cute.math.max(local_max, s_vals[reg_idx], ftz=True)
            local_max = cute.math.max(
                local_max,
                Float32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=local_max,
                        offset=16,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                ),
                ftz=True,
            )
            if cutlass.const_expr(not cfg.is_dynamic_token_sparse_nvfp4):
                local_max = self._freeze_row_max(old_max_vals[0], local_max)
            else:
                # Both 128-thread halves publish their half-row max for this
                # tile in a parity-double-buffered SMEM slot and meet at one
                # 256-thread named barrier; each then merges prev / own / peer
                # with the same threshold freeze.
                xb = Int32(softmax_scratch_words(cfg)) + (
                    local_tile_idx & Int32(1)
                ) * Int32(2 * XBAR_ROWS)
                # Task-local thread index (0..255): warp_grp_thread_idx is the
                # global thread index in the keeps-MMA-AB variant.
                xt = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
                row = ((xt & Int32(127)) >> Int32(5)) * Int32(16) + (xt & Int32(15))
                own_ptr = self._softmax_scratch.data_ptr(
                    xb + (xt >> Int32(7)) * Int32(XBAR_ROWS) + row
                )
                own_ptr.store(
                    float_to_u32_bits(local_max), is_volatile=True, alignment=4
                )
                prims.barrier_cta_sync(barrier_id=XBAR_BARRIER_ID, thread_count=256)
                peer_ptr = self._softmax_scratch.data_ptr(
                    xb + (Int32(1) - (xt >> Int32(7))) * Int32(XBAR_ROWS) + row
                )
                peer = u32_bits_to_float(peer_ptr.load(is_volatile=True, alignment=4))
                cand = cute.math.max(
                    cute.math.max(old_max_vals[0], local_max, ftz=True), peer, ftz=True
                )
                local_max = self._freeze_row_max(old_max_vals[0], cand)
            new_max_vals[0] = local_max
            self.new_max_state[0] = local_max
        else:
            # Swaps-MMA-AB reduces per-scale maxima first within each warp and
            # then across the four softmax columns through SMEM atomics.
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                s_base = Int32((scale_idx // 2) * 4 + (scale_idx & 1))
                s_stride = Int32(q_repeats * 4)
                local_max = cute.math.max(
                    cute.math.max(s_vals[s_base + 0], s_vals[s_base + 2], ftz=True),
                    cute.math.max(
                        s_vals[s_base + s_stride],
                        s_vals[s_base + s_stride + Int32(2)],
                        ftz=True,
                    ),
                    ftz=True,
                )
                local_max = cute.math.max(local_max, old_max_vals[scale_idx], ftz=True)
                local_max = cute.math.max(
                    local_max,
                    Float32(
                        prims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=local_max,
                            offset=16,
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.BFLY,
                        )
                    ),
                    ftz=True,
                )
                local_max = cute.math.max(
                    local_max,
                    Float32(
                        prims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=local_max,
                            offset=8,
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.BFLY,
                        )
                    ),
                    ftz=True,
                )
                local_max_vals[scale_idx] = local_max

            # This is state initialization, not schedule dispatch: the first
            # softmax tile in a work tile must clear the shared atomic scratch.
            # The sync covers one four-warp softmax group.
            if cutlass.const_expr(section == MlaStage.Head):
                init_softmax_scratch_u32(
                    self._softmax_scratch,
                    warp_grp_thread_idx,
                    WARPGROUP_WARPS * num_softmax_scale_groups(cfg),
                )
                prims.barrier_cta_sync(
                    barrier_id=self.sync_barrier_id, thread_count=WARPGROUP_THREADS
                )
            elif cutlass.const_expr(section == MlaStage.Loop):
                if stage_info.loop_offset == stage_info.loop_start:
                    init_softmax_scratch_u32(
                        self._softmax_scratch,
                        warp_grp_thread_idx,
                        WARPGROUP_WARPS * num_softmax_scale_groups(cfg),
                    )
                    prims.barrier_cta_sync(
                        barrier_id=self.sync_barrier_id,
                        thread_count=WARPGROUP_THREADS,
                    )

            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            col_group_idx = lane_idx & Int32(QUAD_LANE_MASK)
            if lane_idx < Int32(OCTET_LANES):
                atomic_reduce_base = col_group_idx * Int32(num_scale_groups)
                for scale_idx in cutlass.range_constexpr(num_scale_groups):
                    smem_atomic_max_u32(
                        self._softmax_scratch.data_ptr(
                            atomic_reduce_base + Int32(scale_idx)
                        ),
                        float_to_u32_for_atomic_max(local_max_vals[scale_idx]),
                    )
            prims.barrier_cta_sync(
                barrier_id=self.sync_barrier_id, thread_count=WARPGROUP_THREADS
            )

            reduced_max_ptr = self._softmax_scratch.data_ptr(
                col_group_idx * Int32(num_scale_groups)
            )
            reduced_max = reduced_max_ptr.load(
                count=num_scale_groups,
                alignment=16 if num_scale_groups == 4 else 8,
            )
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                new_max_vals[scale_idx] = self._freeze_row_max(
                    old_max_vals[scale_idx],
                    u32_to_float_for_atomic_max(reduced_max[scale_idx]),
                )
                self.new_max_state[scale_idx] = new_max_vals[scale_idx]
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            old_max_arr[scale_idx] = old_max_vals[scale_idx]
            sum_arr[scale_idx] = sum_vals[scale_idx]
            new_max_arr[scale_idx] = new_max_vals[scale_idx]
        for idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
            s_arr[idx] = s_vals[idx]
        return old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr

    @cute.jit
    def _freeze_row_max(self, old_max, new_max):
        """Apply threshold skip correction to one row-max update.

        When the raw-score increase of this K tile fits the adjusted threshold
        the previous row max is kept, so P grows by at most ``2**threshold``
        and the running O accumulation needs no rescale for this tile.  The
        first tile (``old_max == -inf``) always adopts the new max.
        """
        frozen = new_max
        if cutlass.const_expr(self.cfg.enable_skip_correction):
            if self.adjusted_skip_corr_threshold > Float32(0.0):
                if old_max != neg_max_f32():
                    if new_max - old_max <= self.adjusted_skip_corr_threshold:
                        frozen = old_max
        return frozen

    @consumer_work(returns=regs_p)
    @cute.jit
    def compute_p_s(self, stage_info: StageInfo, *, new_max_arr, s_arr, local_sum_arr):
        """Native FP4 QK: exp / pack this half's P into registers inside the S
        consumer section (right after the merged max is known), before the
        single P stage is acquired; the S stage is released afterwards.
        ``local_sum_arr`` is updated in place."""
        return self.p_ref._compute_p_impl(
            stage_info,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
            local_sum_arr=local_sum_arr,
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr),
    )
    @cute.jit
    def update_softmax_sum(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        sum_arr,
        new_max_arr,
        local_sum_arr,
        s_arr,
    ):
        """Apply the online-softmax sum correction after P is materialized."""
        # This consumer aux step runs after SmemP/TmemP has produced P and
        # published local_sum_arr.  It updates the running denominator for the
        # next score tile without consuming a new score payload.
        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            state_idx = cute.arch.thread_idx()[0]
            state_ptr = self._softmax_scratch.data_ptr(state_idx)
            old_max = old_max_arr[0]
            new_max = u32_bits_to_float(state_ptr.load(is_volatile=True, alignment=4))
            updated_sum = u32_bits_to_float(
                softmax_sum_state_ptr(state_ptr).load(is_volatile=True, alignment=4)
            )
            prims.barrier_cta_sync(
                barrier_id=self.sync_barrier_id,
                thread_count=WARPGROUP_THREADS,
            )
            self.new_max_state[0] = new_max
            self.sum_state[0] = updated_sum
            old_max_arr[0] = old_max
            sum_arr[0] = updated_sum
            new_max_arr[0] = new_max
            return old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr
        for scale_base in cutlass.range_constexpr(
            0, num_softmax_scale_groups(self.cfg), 2
        ):
            old_max_0 = old_max_arr[scale_base]
            old_max_1 = old_max_arr[scale_base + 1]
            new_max_0 = new_max_arr[scale_base]
            new_max_1 = new_max_arr[scale_base + 1]
            local_sum_0 = local_sum_arr[scale_base]
            local_sum_1 = local_sum_arr[scale_base + 1]
            if cutlass.const_expr(self.p_ref is not None):
                local_sum_0 = self._p_local_sum_arr[scale_base]
                local_sum_1 = self._p_local_sum_arr[scale_base + 1]
            sum_0 = sum_arr[scale_base]
            sum_1 = sum_arr[scale_base + 1]

            exp_scale0 = Float32(0.0)
            exp_scale1 = Float32(0.0)
            if (old_max_0 != neg_max_f32()) and (new_max_0 != neg_max_f32()):
                exp_scale0 = cute.math.exp2(
                    self.scale_softmax_log2 * (old_max_0 - new_max_0),
                    fastmath=True,
                )
            if (old_max_1 != neg_max_f32()) and (new_max_1 != neg_max_f32()):
                exp_scale1 = cute.math.exp2(
                    self.scale_softmax_log2 * (old_max_1 - new_max_1),
                    fastmath=True,
                )
            updated_sums = ffma2(
                (exp_scale0, exp_scale1),
                (sum_0, sum_1),
                (local_sum_0, local_sum_1),
            )
            self.sum_state[scale_base] = updated_sums[0]
            self.sum_state[scale_base + 1] = updated_sums[1]
            sum_arr[scale_base] = updated_sums[0]
            sum_arr[scale_base + 1] = updated_sums[1]
            old_max_arr[scale_base] = old_max_0
            old_max_arr[scale_base + 1] = old_max_1
            new_max_arr[scale_base] = new_max_0
            new_max_arr[scale_base + 1] = new_max_1
            local_sum_arr[scale_base] = local_sum_0
            local_sum_arr[scale_base + 1] = local_sum_1
        return old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr


# =====================================================================
# TmemSKeepsResource — Keeps-MMA-AB S scores in TMEM
# =====================================================================


@dataclass(kw_only=True)
class TmemSKeepsResource(TmemSResource):
    """Keeps-MMA-AB score resource with a single softmax scale group."""
