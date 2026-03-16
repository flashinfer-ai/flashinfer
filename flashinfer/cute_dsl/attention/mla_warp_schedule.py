# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAWarpSchedule — warp role assignment for MLA decode kernels.

Separate concrete type from WarpSchedule (FMHA), following the C++ CUTLASS
pattern. MLA merges softmax+correction+epilogue into a single "Compute" role
and uses 6-8 warps instead of 16.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MLAWarpSchedule:
    """Warp role assignment and register budgets for MLA decode.

    Warp layout (non-cpasync, 6 warps):
        0-3: Compute (merged softmax + correction + epilogue)
        4:   MMA (Q*K^T, P*V)
        5:   TMA Load

    Warp layout (cpasync, 8 warps):
        0-3: Compute
        4:   MMA
        5-6: CP Async Load
        7:   Page Table Load
    """

    compute_warp_ids: Tuple[int, ...] = (0, 1, 2, 3)
    mma_warp_id: int = 4
    load_tma_warp_id: int = 5
    load_cp_async_warp_ids: Tuple[int, ...] = (5, 6)
    load_pt_warp_id: int = 7

    threads_per_warp: int = 32

    tmem_ptr_sync_bar_id: int = 1
    exchange_sync_bar_id: int = 2

    @property
    def num_compute_warps(self) -> int:
        return len(self.compute_warp_ids)

    @property
    def warps_in_n(self) -> int:
        return 2

    def threads_per_cta(self, is_cpasync: bool) -> int:
        if is_cpasync:
            num_warps = len(
                (
                    self.mma_warp_id,
                    *self.load_cp_async_warp_ids,
                    self.load_pt_warp_id,
                    *self.compute_warp_ids,
                )
            )
        else:
            num_warps = len(
                (self.mma_warp_id, self.load_tma_warp_id, *self.compute_warp_ids)
            )
        return self.threads_per_warp * num_warps

    @property
    def tmem_ptr_sync_num_threads(self) -> int:
        """MMA warp + all compute warps participate in tmem pointer sync."""
        return self.threads_per_warp + self.threads_per_warp * self.num_compute_warps

    @property
    def exchange_sync_num_threads(self) -> int:
        """All compute warps participate in row_sum exchange."""
        return self.threads_per_warp * self.num_compute_warps


MLA_DECODE_SCHEDULE = MLAWarpSchedule()
