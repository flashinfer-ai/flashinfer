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

"""GMEM O TensorMap for the epilogue stores."""

import cutlass.cute as cute
import cutlass
from cutlass import Int32
from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    producer_work,
)
from dataclasses import dataclass, field
from ..config import FmhaConfig
from .....device.placeholders import placeholder_smem_array
from .....device.tensor_map import transform_ragged_coords
from typing import Any, Optional


# ---------------------------------------------------------------------------
# GmemOResource -- global memory O output (no pipeline)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class GmemOResource(MemoryResource):
    """TMA store for one O-subtile to global memory.

    Each instance (``gmem_o_0``, ``gmem_o_1``) has its own smem
    staging region aliased with the matching ``SmemOResource``
    instance, keeping per-subtile accesses independently trackable.
    No pipeline — point-access only.

    Producer: EpilogueTask stores O tiles from SMEM to GMEM via TMA.
    """

    # Static configuration
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    stage_idx: Constexpr[int] = field(init=False, default=0)

    # GMEM pointers and tensors
    cum_seqlen_q: cute.Tensor | None = field(init=False, default=None)
    tma_o_desc: cutlass.Pointer | None = field(init=False, default=None)

    # Allocations
    alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    # Per-task state
    sO_array: cutlass.Array = field(init=False, default=None)

    def __init__(
        self,
        tma_o_desc: cutlass.Pointer | None,
        cum_seqlen_q: cute.Tensor | None,
        cfg: FmhaConfig,
        stage_idx: int = 0,
        **kwargs: Any,
    ) -> None:
        """Bind the O TMA descriptor and reserve store-side SMEM staging."""
        super().__init__(**kwargs)
        self.tma_o_desc = tma_o_desc
        self.cum_seqlen_q = cum_seqlen_q
        self.cfg = cfg
        self.stage_idx = stage_idx
        stage_elements = cfg.sO_stage_elements
        size_bytes = stage_elements * cfg.o_dtype.width // 8
        self.alloc = SmemAllocation(
            f"gmem_o_{stage_idx}_smem", size_bytes, alignment=cfg.buffer_align_bytes
        )
        self.sO_array = placeholder_smem_array(cfg.o_dtype)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return the SMEM allocation used as the O TMA store source."""
        return [self.alloc]

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        """Materialize the O store staging buffer; this resource emits no slots."""
        # Pure sink — producer-side head_coord / batch_coord / seq_coord_q
        # slots are auto-mirrored from upstream SmemO by Task.init_variables.
        smem_base = stage_info.context.smem_base
        stage_elements = self.cfg.sO_stage_elements
        self.sO_array = cutlass.Array(
            smem_base.data_ptr() + self.alloc.offset,
            dtype=self.cfg.o_dtype,
            shape=(stage_elements,),
            addrspace=3,
        )

    @producer_work
    @cute.jit
    def tma_store(
        self,
        stage_info: StageInfo,
        *,
        head_coord: Int32,
        batch_coord: Int32,
        seq_coord_q: Int32,
        head_dim_stage_idx: cutlass.Constexpr[int] = 0,
        correction_fused: cutlass.Constexpr[bool] = False,
    ) -> None:
        """TMA store O from SMEM to GMEM.

        Coordinates are produced by SmemOResource.consumer_work() and routed
        via schedule dataflow into this producer call.  In head-paired mode,
        the two output stages map to consecutive Q heads instead of consecutive
        Q sequence tiles.
        """
        head_coord = (
            head_coord * self.cfg.work_tile_q_heads
            + self.stage_idx * self.cfg.peer_q_head_stride
        )
        seq_offset_o = (
            seq_coord_q
            + self.stage_idx * self.cfg.peer_q_seq_tile_stride * self.cfg.q_tile_m
        )
        sO_base = self.sO_array
        should_store = True
        q_seq_extent = Int32(0)
        if cutlass.const_expr(self.cfg.has_varlen):
            if cutlass.const_expr(self.cfg.has_uniform_varlen):
                cuseqlen_q = batch_coord * Int32(self.cfg.uniform_seq_len_q)
                seq_end = cuseqlen_q + Int32(self.cfg.uniform_seq_len_q)
            else:
                cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord])
                seq_end = Int32(self.cum_seqlen_q[batch_coord + Int32(1)])
            seq_offset_o = cuseqlen_q + seq_offset_o
            q_seq_extent = seq_end - seq_offset_o
            should_store = seq_offset_o < seq_end

        tma_copy_o_iters = self.cfg.tma_copy_o_iters
        if cutlass.const_expr(self.cfg.stage_o_by_head_dim):
            tma_copy_o_iters = self.cfg.tma_copy_o_stage_iters

        is_store_warp = True
        if cutlass.const_expr(correction_fused):
            # elect_sync elects one lane *per warp*.  A correction-fused call
            # runs on four warps, so only the first correction warp may enter
            # the TMA issue/commit/wait body.
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            is_store_warp = warp_idx == self.cfg.correction_warp_ids[0]
        if is_store_warp:
            if should_store:
                if prims.elect_sync():
                    for i in cutlass.range_constexpr(tma_copy_o_iters):
                        d_offset = (
                            head_dim_stage_idx * self.cfg.head_dim_per_stage_o
                            + i * self.cfg.tma_copy_o_granu_inner
                        )
                        o_coords = (d_offset, head_coord, seq_offset_o, batch_coord)
                        if cutlass.const_expr(self.cfg.has_varlen):
                            o_coords = (d_offset, head_coord, seq_offset_o)
                            o_coords = transform_ragged_coords(
                                o_coords,
                                ragged_dim_idx=2,
                                ragged_box_size=self.cfg.epi_tile[0],
                                ragged_extent=q_seq_extent,
                            )
                        prims.cp_async_bulk_tensor_global_shared_cta(
                            self.tma_o_desc,
                            sO_base.subview(i * self.cfg.tma_copy_o_granu_elems),
                            o_coords,
                        )
            # should_store is CTA-uniform because it depends only on batch and
            # Q tile coordinates. Keep commit paired with an actual store.
            if should_store:
                prims.cp_async_bulk_commit_group()
                if cutlass.const_expr(self.cfg.gmem_o_store_wait_after_write):
                    prims.cp_async_bulk_wait_group(0, read=True)
