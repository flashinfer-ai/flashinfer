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

"""SMEM Q ring loaded by TMA and consumed by the QK MMA."""

import cutlass.cute as cute
import cutlass
from cutlass import Int16, Int32
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
from ..device.common import (
    MX_SCALE_TILE_BYTES,
    mx_scale_slot,
    qk_smem_desc_offsets,
    qk_smem_layout,
)
from .....device.placeholders import placeholder_smem_array
from .....device.tensor_map import transform_ragged_coords
from typing import Any, Optional


# ---------------------------------------------------------------------------
# SmemQResource -- SMEM Q tile buffer with TMA pipeline
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class SmemQResource(MemoryResource):
    """SMEM buffer for Q tiles with a topology-derived TmaUmma pipeline.

    Producer: LoadTask (TMA loads Q0 and Q1 in the first K-loop iteration).
    Consumer: MmaTask (builds SMEM descriptors, holds Q across K-loop).
    """

    # Static configuration
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    desc_q0_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    desc_q1_base: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    # GMEM pointers and tensors
    q_scales: cute.Tensor | None = field(init=False, default=None)
    tma_q_desc: cutlass.Pointer | None = field(init=False, default=None)

    # Allocations
    alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    # Per-task state
    sQ_array: cutlass.Array = field(init=False, default=None)

    def __init__(
        self,
        tma_q_desc: cutlass.Pointer | None,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        q_scales: cute.Tensor | None = None,
        **kwargs: Any,
    ) -> None:
        """Bind the Q TMA descriptor and reserve SMEM for staged Q tiles."""
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.tma_q_desc = tma_q_desc
        self.q_scales = q_scales
        self.cfg = cfg
        total_elements = cfg.sQ_shape[0] * cfg.sQ_shape[1]
        size_bytes = total_elements * cfg.q_dtype.width // 8
        if cfg.use_mxfp8_qk:
            size_bytes += cfg.sQ_shape[0] * MX_SCALE_TILE_BYTES
        self.alloc = SmemAllocation(
            "smem_q", size_bytes, alignment=cfg.buffer_align_bytes
        )
        self.sQ_array = placeholder_smem_array(cfg.q_dtype)
        self.desc_q0_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the first Q half.",
        )
        self.desc_q1_base = TaskLocalVariable(
            dtype=cutlass.Int64,
            default=cutlass.Int64(0),
            docs="SMEM descriptor base for the second Q half.",
        )

    def build_q_descriptor(self, inst_idx: int) -> prims.Tcgen05SmemDesc:
        """Build SMEM descriptor for the current Q tile.

        Uses inst_idx (not stage_idx) to compute the SMEM offset because
        Q is consumed twice in HEAD without an intervening ConsumerRelease,
        which would otherwise advance consumer_state.
        """
        sQ_curr = self.sQ_array.subview(inst_idx * self.cfg.tma_copy_q_elements)
        leading_byte_offset, stride_byte_offset = qk_smem_desc_offsets(self.cfg)
        return prims.Tcgen05SmemDesc.build(
            sQ_curr,
            leading_byte_offset=leading_byte_offset,
            stride_byte_offset=stride_byte_offset,
            layout=qk_smem_layout(self.cfg),
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the SMEM allocation required for staged Q tiles."""
        return [self.alloc]

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        self.init_smem_state(stage_info)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        self.init_smem_state(stage_info)

    @cute.jit
    def init_smem_state(self, stage_info: StageInfo) -> None:
        """Materialize the Q SMEM array and descriptor dataflow slots."""
        smem_base = stage_info.context.smem_base
        total_elements = self.cfg.sQ_shape[0] * self.cfg.sQ_shape[1]
        self.sQ_array = cutlass.Array(
            smem_base.data_ptr() + self.alloc.offset,
            dtype=self.cfg.q_dtype,
            shape=(total_elements,),
            addrspace=3,
        )

    @consumer_work(returns=desc_q0_base)
    @cute.jit
    def q0_desc(
        self, stage_info: StageInfo, *, inst_idx: cutlass.Constexpr[int]
    ) -> prims.Tcgen05SmemDesc:
        """Build Q0 SMEM descriptor -> desc_q0_base."""
        return self.build_q_descriptor(inst_idx)

    @consumer_work(returns=desc_q1_base)
    @cute.jit
    def q1_desc(
        self, stage_info: StageInfo, *, inst_idx: cutlass.Constexpr[int]
    ) -> prims.Tcgen05SmemDesc:
        """Build Q1 SMEM descriptor -> desc_q1_base."""
        return self.build_q_descriptor(inst_idx)

    @producer_work
    @cute.jit
    def tma_load(
        self,
        stage_info: StageInfo,
        *,
        seq_coord_q: Int32,
        head_coord: Int32,
        batch_coord: Int32,
        cuseqlen_q: Int32,
        seqlen_q: Int32,
        inst_idx: cutlass.Constexpr[int],
    ) -> None:
        """TMA load one Q tile (Q0 or Q1) from GMEM to SMEM.

        inst_idx 0 = Q0, inst_idx 1 = Q1.
        Uses seq_coord_q from producer variables (forwarded from GmemQKV).
        """
        q_head_coord = (
            head_coord * self.cfg.work_tile_q_heads
            + inst_idx * self.cfg.peer_q_head_stride
        )
        q_seq_offset = (
            seq_coord_q + inst_idx * self.cfg.peer_q_seq_tile_stride * self.cfg.q_tile_m
        )
        q_seq_extent = Int32(0)
        if cutlass.const_expr(self.cfg.has_varlen):
            q_seq_extent = cuseqlen_q + seqlen_q - q_seq_offset
        smem_stage_elements = self.cfg.tma_copy_q_elements
        d_granu_inner = self.cfg.tma_copy_q_granu_inner

        sQ_curr = self.sQ_array.subview(stage_info.stage_idx * smem_stage_elements)
        if prims.elect_sync():
            for i in cutlass.range_constexpr(self.cfg.tma_copy_qkv_iters):
                d_offset = i * d_granu_inner
                q_coords = (d_offset, q_head_coord, q_seq_offset, batch_coord)
                if cutlass.const_expr(self.cfg.has_varlen):
                    q_coords = (d_offset, q_head_coord, q_seq_offset)
                    q_coords = transform_ragged_coords(
                        q_coords,
                        ragged_dim_idx=2,
                        ragged_box_size=self.cfg.qk_mma_tiler[0],
                        ragged_extent=q_seq_extent,
                    )
                if cutlass.const_expr(self.cfg.two_cta_umma):
                    # Each CTA loads only its own 128 Q rows (multicast mask = own
                    # rank), but as a cta_group::2 copy, so the byte completion
                    # signals the mbarrier in the leader CTA's SMEM. The leader's
                    # MMA waits there for both CTAs' Q before the M=256 UMMA.
                    cta_rank = cute.arch.make_warp_uniform(
                        cute.arch.block_idx_in_cluster()
                    )
                    prims.cp_async_bulk_tensor_shared_cluster_global(
                        sQ_curr.subview(i * self.cfg.tma_copy_q_granu_elems),
                        self.tma_q_desc,
                        q_coords,
                        cutlass.Array(
                            stage_info.barrier.data_ptr(), dtype=cutlass.Int64
                        ),
                        [],
                        multicast_mask=Int16(Int32(1) << cta_rank),
                        group=prims.CTAGroup.CTA_2,
                    )
                else:
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        sQ_curr.subview(i * self.cfg.tma_copy_q_granu_elems),
                        self.tma_q_desc,
                        q_coords,
                        stage_info.barrier,
                    )
            if cutlass.const_expr(self.cfg.use_mxfp8_qk):
                # The Q scale atom of this tile lands behind the Q tiles.
                prims.cp_async_bulk_shared_cluster_global(
                    mx_scale_slot(
                        stage_info.context.smem_base,
                        self.alloc.offset
                        + self.cfg.sQ_shape[0] * self.cfg.tma_copy_q_bytes
                        + inst_idx * MX_SCALE_TILE_BYTES,
                    ),
                    self.q_scales.iterator
                    + cute.crd2idx(
                        (
                            batch_coord,
                            q_head_coord,
                            # The second Q tile of the last work tile can lie
                            # past the sequence; its rows are dropped, so it
                            # reads the last atom instead of the next head's.
                            cute.math.min(
                                q_seq_offset // self.cfg.q_tile_m,
                                Int32(self.q_scales.shape[2] - 1),
                            ),
                            0,
                        ),
                        self.q_scales.layout,
                    ),
                    stage_info.barrier,
                    MX_SCALE_TILE_BYTES,
                )
