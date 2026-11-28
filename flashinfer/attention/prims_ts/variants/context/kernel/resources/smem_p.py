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

"""SMEM P staging for the PV MMA."""

from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PipelineConfig,
)
from dataclasses import dataclass, field
from ..config import FmhaConfig
from ..device.common import SmemDescOffsets, smem_layout_for_inner_bytes
from typing import Any, Optional


# ---------------------------------------------------------------------------
# SmemPResource -- fp8 P tile staged in SMEM per query group
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class SmemPResource(MemoryResource):
    """One K-major SW128 fp8 P tile per query group in SMEM: softmax produces,
    the PV UMMA consumes, so softmax can release the S stage early."""

    # Static configuration
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)

    # Allocations
    alloc: Constexpr[Optional[SmemAllocation]] = field(init=False, default=None)

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        cfg: FmhaConfig,
        group_idx: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(pipeline_config=pipeline_config, **kwargs)
        self.cfg = cfg
        # VC-Attention appends the group's bf16 [128 x 16] row-sum operands behind
        # the P tile when it restores the tile means, and VC-Attention-QK8 its
        # K-scale table behind those.
        extra_bytes = 0
        if cfg.vc_restores_means:
            extra_bytes += cfg.vc_rowsum_tile_bytes
        if cfg.vc_qk8:
            extra_bytes += cfg.vc_kscale_table_bytes
        self.alloc = SmemAllocation(
            f"smem_p{group_idx}",
            cfg.smem_p_bytes + extra_bytes,
            alignment=cfg.buffer_align_bytes,
        )

    @property
    def kscale_table_offset(self) -> int:
        """Byte offset of the VC-Attention-QK8 K-scale table inside this allocation,
        behind the row-sum operand when the tile means are restored."""
        rowsum = self.cfg.vc_rowsum_tile_bytes if self.cfg.vc_restores_means else 0
        return self.cfg.smem_p_bytes + rowsum

    @property
    def row_bytes(self) -> int:
        return self.cfg.qk_mma_tiler[1] * self.cfg.v_dtype.width // 8

    @property
    def rowsum_tile_offset(self) -> int:
        """Byte offset of the VC-Attention row-sum operands inside this allocation."""
        return self.cfg.smem_p_bytes

    def descriptor_layout(self) -> int:
        return smem_layout_for_inner_bytes(self.row_bytes)

    def descriptor_offsets(self) -> SmemDescOffsets:
        """LBO and SBO of the K-major swizzled tile: eight rows per swizzle atom."""
        leading_byte_offset = 16
        return leading_byte_offset, 8 * self.row_bytes

    def get_smem_requirements(self) -> list[SmemAllocation]:
        return [self.alloc]
