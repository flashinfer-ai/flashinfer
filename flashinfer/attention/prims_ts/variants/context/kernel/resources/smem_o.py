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

"""SMEM O staging between correction and the epilogue."""

import cutlass.cute as cute
import cutlass
from cutlass import Float32, Int32, Int64
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
from dataclasses import dataclass, field
from ..config import FmhaConfig
from ..device.common import SoftmaxScalar, TmemAddr, resolve_work_tile_coords
from .tmem_stats import TmemStatsResource
from ..device.sage_scales import load_sage_v_channels, sage_v_section_address
from .....device.placeholders import placeholder_smem_array
from ....sage.kernel.index import sage_sequence_words
from typing import Any, Optional


# ---------------------------------------------------------------------------
# SmemOResource -- SMEM O buffer with AsyncAsync pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SmemOResource(MemoryResource):
    """SMEM buffer for one O-subtile (1-stage AsyncAsync pipeline).

    Each instance (``smem_o_0``, ``smem_o_1``) owns a distinct smem
    region for its subtile, so the checker can verify that stage-0
    and stage-1 accesses never conflict.

    Producer: CorrectionTask (correction_epilog writes converted O to SMEM).
    Consumer: EpilogueTask (TMA stores O from SMEM to GMEM).
    """

    # Static configuration
    batch_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    head_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seq_coord_q: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    stage_idx: Constexpr[int] = field(init=False, default=0)
    tmem_addr_cached: TmemAddr | None = field(init=False, default=None)
    tmem_o_addr_base_cached: TmemAddr | None = field(init=False, default=None)

    # Linked resources
    # Reference to the TmemStats resource for this stage's correction stats.
    tmem_vec_resource: TmemStatsResource | None = field(init=False, default=None)

    # Runtime scalars
    num_kv_heads: int | Int32 = field(init=False, default=0)

    # GMEM pointers and tensors
    # Sage attention: the prepared scales and the KV head count that strides
    # their sequences; the epilogue reads the work tile's ``sfV`` / ``v_mean``.
    sage_scales: cute.Pointer | None = field(init=False, default=None)

    # Allocations
    alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    # Per-task state
    sO_array: cutlass.Array = field(init=False, default=None)

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        stage_idx: int = 0,
        tmem_vec_resource: TmemStatsResource | None = None,
        sage_scales: cute.Pointer | None = None,
        num_kv_heads: int | Int32 = 0,
        **kwargs: Any,
    ) -> None:
        """Bind one output subtile stage and reserve its SMEM staging buffer."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        self.stage_idx = stage_idx
        self.tmem_vec_resource = tmem_vec_resource
        self.sage_scales = sage_scales
        self.num_kv_heads = num_kv_heads
        stage_elements = cfg.sO_stage_elements
        size_bytes = stage_elements * cfg.o_dtype.width // 8
        self.alloc = SmemAllocation(
            f"smem_o_{stage_idx}", size_bytes, alignment=cfg.buffer_align_bytes
        )
        self.sO_array = placeholder_smem_array(cfg.o_dtype)
        self.tmem_addr_cached = Int32(0)
        self.tmem_o_addr_base_cached = Int32(0)
        self.head_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q/O head coordinate for the output subtile.",
        )
        self.batch_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Batch coordinate for the output subtile.",
        )
        self.seq_coord_q = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Output row coordinate for the output subtile.",
        )

    @consumer_work(returns=(head_coord, batch_coord, seq_coord_q))
    @cute.jit
    def compute_output_coords(
        self, stage_info: StageInfo
    ) -> tuple[Int32, Int32, Int32]:
        """Return output-tile coordinates for downstream GMEM TMA store."""
        seq_coord, head_coord, batch_coord = resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )
        seq_coord_q = seq_coord * self.cfg.q_tile_m * self.cfg.work_tile_q_seq_tiles
        return head_coord, batch_coord, seq_coord_q

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the SMEM allocation for this O staging subtile."""
        return [self.alloc]

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_output_state(self, stage_info: StageInfo) -> None:
        self.init_smem_state(stage_info)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_output_work_tile_state(self, stage_info: StageInfo) -> None:
        self.init_work_tile_state(stage_info)

    @cute.jit
    def init_smem_state(self, stage_info: StageInfo) -> None:
        """Derive sO_array from context (once, ungated).

        Per-warp TMEM O address base deferred to per-work-tile auxiliary work
        to avoid crossing the setmaxnreg boundary.
        tmem_o_addr_base_cached initialized to Int32(0) to establish DSL type.

        Emits per-tile output coordinates consumed downstream by
        GmemO via the EpilogueTask; producer-side vec_row_sum /
        vec_scale slots are auto-mirrored from TmemStats by
        Task.init_variables.
        """
        smem_base = stage_info.context.smem_base
        stage_elements = self.cfg.sO_stage_elements
        self.sO_array = cutlass.Array(
            smem_base.data_ptr() + self.alloc.offset,
            dtype=self.cfg.o_dtype,
            shape=(stage_elements,),
            addrspace=3,
        )
        self.tmem_o_addr_base_cached = Int32(0)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        self.init_smem_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_work_tile_state(self, stage_info: StageInfo) -> None:
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
    def store_o(
        self,
        stage_info: StageInfo,
        *,
        vec_row_sum: SoftmaxScalar,
        vec_scale: SoftmaxScalar,
        output_scale: SoftmaxScalar,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Store O using the output scale cached before the loop."""
        self.store_o_impl(
            stage_info,
            vec_row_sum=vec_row_sum,
            vec_scale=vec_scale,
            output_scale=output_scale,
            head_dim_stage_idx=head_dim_stage_idx,
        )

    @cute.jit
    def store_o_impl(
        self,
        stage_info: StageInfo,
        *,
        vec_row_sum: SoftmaxScalar,
        vec_scale: SoftmaxScalar,
        output_scale: SoftmaxScalar,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
    ) -> None:
        """Correct and normalize O in one TMEM pass, then stage it in SMEM.

        With Sage attention column ``c`` becomes ``o * scale * sfV[c] +
        output_scale * v_mean[c]``: the per-channel scale joins the row's
        normalization and the mean is added after it. The scales are read
        from the prepared buffer once per output slice, as warp broadcasts.
        """
        if cutlass.const_expr(self.stage_idx == 0):
            tmem_o_offset = self.cfg.tmem_o0_offset
        else:
            tmem_o_offset = self.cfg.tmem_o1_offset
        sO_base = self.sO_array

        # Read precomputed correction scale and row_sum from TmemStats.
        # vec_scale = exp2(scale_log2 * (old_max - new_max)) was computed
        # in TmemStats.consumer_work.
        correction_scale = vec_scale
        scale = output_scale * correction_scale / vec_row_sum
        if cutlass.const_expr(self.cfg.lse_mode != "none"):
            # Empty dense requests publish a neutral output alongside -inf LSE.
            if vec_row_sum == Float32(0.0):
                scale = Float32(0.0)

        num_correction_warps = 4
        tidx, _, _ = cute.arch.thread_idx()
        tid_in_wg = tidx % (cute.arch.WARP_SIZE * num_correction_warps)

        vc_scale_row = Int64(0)
        if cutlass.const_expr(self.cfg.vc_attention):
            # Start of this batch and head row in the per-channel V scale table.
            assert self.tmem_vec_resource is not None
            assert self.tmem_vec_resource.output_scale is not None
            assert self.cfg.work_tile_q_heads == 1
            vc_scale = self.tmem_vec_resource.output_scale
            _, head_coord_wt, batch_coord = resolve_work_tile_coords(
                self.cfg, stage_info.work_tile.tile_idx
            )
            vc_scale_row = vc_scale.iterator.toint() + Int64(
                cute.crd2idx(
                    (batch_coord, head_coord_wt * self.cfg.work_tile_q_heads, 0),
                    vc_scale.layout,
                )
            ) * Int64(4)

        o_head_dim = self.cfg.epi_tile[1]
        tma_copy_o_iters = self.cfg.tma_copy_o_iters
        if cutlass.const_expr(self.cfg.stage_o_by_head_dim):
            o_head_dim = self.cfg.head_dim_per_stage_o
            tma_copy_o_iters = self.cfg.tma_copy_o_stage_iters

        tmem_offset_o = (
            self.tmem_o_addr_base_cached
            + tmem_o_offset
            + head_dim_stage_idx * o_head_dim
        )

        tmem_shape = "32x32b"
        tmem_x = 16
        # Do not load unallocated TMEM columns in a partial final O band.
        valid_head_dim = min(
            o_head_dim, self.cfg.epi_tile[1] - head_dim_stage_idx * o_head_dim
        )
        num_iters = valid_head_dim // tmem_x

        smem_o_swizzle = make_smem_o_swizzle(self.cfg)

        d_block_size = o_head_dim // tma_copy_o_iters
        row_offset = tid_in_wg * d_block_size

        if cutlass.const_expr(self.cfg.use_sage_attention):
            # The sequence's first sfV word plus this stage's first channel;
            # the per-iteration channels are constant offsets from it.
            _, head_coord, batch_coord = resolve_work_tile_coords(
                self.cfg, stage_info.work_tile.tile_idx
            )
            sage_sequence = sage_sequence_words(
                self.cfg.sage_scale_layout,
                batch_idx=batch_coord,
                kv_head_idx=head_coord // Int32(self.cfg.h_r),
                num_kv_heads=self.num_kv_heads,
            )
            sage_v_address = sage_v_section_address(
                self.cfg.sage_scale_layout,
                self.sage_scales,
                sequence_words=sage_sequence,
            ) + Int64(head_dim_stage_idx * o_head_dim) * Int64(4)

        for i in cutlass.range_constexpr(num_iters):
            tmem_offset_tile = tmem_offset_o + i * tmem_x

            tmem_ptr = cutlass.inttoptr(
                tmem_offset_tile,
                mem_space=6,
                dtype=self.cfg.pv_acc_dtype,
            )

            o_rmem = prims.tcgen05_ld(tmem_shape, tmem_ptr, num=tmem_x)
            cute.arch.fence_view_async_tmem_load()

            scale_vec = cutlass.vector.full_like(o_rmem, scale)
            if cutlass.const_expr(self.cfg.use_sage_attention):
                v_scales, v_means = load_sage_v_channels(
                    self.cfg.sage_scale_layout,
                    sage_v_address,
                    offset=i * tmem_x,
                    count=tmem_x,
                )
                o_rmem = o_rmem * (scale_vec * v_scales)
                if cutlass.const_expr(v_means is not None):
                    o_rmem = o_rmem + v_means * cutlass.vector.full_like(
                        o_rmem, output_scale
                    )
            elif cutlass.const_expr(self.cfg.vc_attention):
                o_rmem = o_rmem * scale_vec
                # The tile means were stored divided by the per-channel V scale,
                # so the whole accumulator is rescaled by it here.
                col0 = head_dim_stage_idx * o_head_dim + i * tmem_x
                chan_scale = cutlass.inttoptr(
                    vc_scale_row + Int64(col0) * Int64(4), mem_space=1, dtype=Float32
                ).load(count=tmem_x, alignment=64)
                o_rmem = o_rmem * chan_scale
            else:
                o_rmem = o_rmem * scale_vec

            o_rmem_dtype = o_rmem.to(self.cfg.o_dtype)

            col_offset = (i * tmem_x) % d_block_size
            block_idx = (i * tmem_x) // d_block_size
            block_offset = block_idx * self.cfg.tma_copy_o_granu_elems
            smem_offset = block_offset + row_offset + col_offset
            smem_ptr = (sO_base.subview(smem_offset)).data_ptr()

            if cutlass.const_expr(self.cfg.o_dtype.width == 8):
                o_rmem_i8 = o_rmem_dtype.bitcast(cutlass.Int8)
                smem_ptr.store_swizzled(o_rmem_i8, alignment=64, swizzle=smem_o_swizzle)
            else:
                smem_ptr.store_swizzled(
                    o_rmem_dtype, alignment=64, swizzle=smem_o_swizzle
                )

        prims.fence_proxy(
            kind=prims.Proxy.ASYNC_SHARED,
            space=prims.SharedSpace.shared_cta,
        )


def make_smem_o_swizzle(cfg: FmhaConfig) -> cutlass.Swizzle:
    """Return the shared-memory swizzle used when staging O for TMA store."""
    inner_dim_size = o_inner_dim_size_bytes(cfg)
    if inner_dim_size % 128 == 0:
        return cutlass.Swizzle(3, 4, 3)
    if inner_dim_size == 64:
        return cutlass.Swizzle(2, 4, 3)
    if inner_dim_size == 32:
        return cutlass.Swizzle(1, 4, 3)
    raise RuntimeError(f"Unsupported inner dimension size: {inner_dim_size}")


def o_inner_dim_size_bytes(cfg: FmhaConfig) -> int:
    """Return the byte width of one O tile inner dimension."""
    return cfg.smem_o_head_dim * cfg.o_dtype.width // 8
