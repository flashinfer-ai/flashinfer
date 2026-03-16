# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""WarpSchedule — warp role assignment and register budgets.

Mirrors C++ CUTLASS's KernelSchedule concept (e.g. Sm100FmhaCtxKernelWarpspecializedSchedule).
Separates warp-to-role mapping and register allocation from the kernel and config,
making it swappable between FMHA, MLA, and future attention variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class WarpSchedule:
    """Defines warp role assignment and register budgets for attention kernels.

    Each field maps directly to C++ CUTLASS's KernelSchedule:
    - Warp ID ranges for each role
    - Register allocation per role (controls spill/occupancy tradeoff)
    - Barrier IDs for CTA sync and TMEM allocation
    """

    softmax0_warp_ids: Tuple[int, ...] = (0, 1, 2, 3)
    softmax1_warp_ids: Tuple[int, ...] = (4, 5, 6, 7)
    correction_warp_ids: Tuple[int, ...] = (8, 9, 10, 11)
    mma_warp_id: int = 12
    load_warp_id: int = 13
    epilogue_warp_id: int = 14
    empty_warp_id: int = 15

    num_regs_softmax: int = 192
    num_regs_correction: int = 96
    num_regs_other: int = 32
    num_regs_empty: int = 24

    threads_per_warp: int = 32
    cta_sync_bar_id: int = 0
    tmem_alloc_sync_bar_id: int = 1

    @property
    def all_warp_ids(self) -> Tuple[int, ...]:
        return (
            *self.softmax0_warp_ids,
            *self.softmax1_warp_ids,
            *self.correction_warp_ids,
            self.mma_warp_id,
            self.load_warp_id,
            self.epilogue_warp_id,
            self.empty_warp_id,
        )

    @property
    def num_warps(self) -> int:
        return len(self.all_warp_ids)

    @property
    def threads_per_cta(self) -> int:
        return self.threads_per_warp * self.num_warps

    @property
    def num_warps_per_warpgroup(self) -> int:
        return 4

    @property
    def softmax_warpgroup_count(self) -> int:
        total_softmax_warps = len(self.softmax0_warp_ids) + len(self.softmax1_warp_ids)
        return total_softmax_warps // self.num_warps_per_warpgroup

    @property
    def tmem_dealloc_arrive_count(self) -> int:
        """Number of threads that must arrive at the TMEM dealloc barrier."""
        return self.threads_per_warp * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
            )
        )


PREFILL_SCHEDULE = WarpSchedule()

MLA_SCHEDULE = WarpSchedule(
    num_regs_softmax=184,
    num_regs_other=48,
)
