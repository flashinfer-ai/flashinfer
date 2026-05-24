# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAWarpSchedule — warp role assignment and register budgets for MLA decode.

Separate concrete type from WarpSchedule (FMHA prefill). The MLA decode kernel
uses 12 warps with a fundamentally different role layout:
- 4 compute warps (softmax + exchange)
- 4 correction warps (rescale + epilogue)
- 1 MMA warp
- 1 TMA load warp
- 1 page table load warp
- 1 empty warp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cutlass.pipeline as pipeline


@dataclass(frozen=True)
class MLAWarpSchedule:
    """Warp role assignment and register budgets for MLA decode kernels."""

    compute_warp_ids: Tuple[int, ...] = (0, 1, 2, 3)
    correction_warp_ids: Tuple[int, ...] = (4, 5, 6, 7)
    mma_warp_id: int = 8
    load_tma_warp_id: int = 9
    load_pt_warp_id: int = 10
    empty_warp_ids: Tuple[int, ...] = (11,)

    softmax_reg_num: int = 192
    correction_reg_num: int = 208
    other_reg_num: int = 96

    threads_per_warp: int = 32

    # Named barrier IDs
    tmem_ptr_sync_bar_id: int = 1
    softmax_exchange_bar_id: int = 2
    epilogue_exchange_bar_id: int = 3

    @property
    def all_warp_ids(self) -> Tuple[int, ...]:
        return (
            *self.compute_warp_ids,
            *self.correction_warp_ids,
            self.mma_warp_id,
            self.load_tma_warp_id,
            self.load_pt_warp_id,
            *self.empty_warp_ids,
        )

    @property
    def num_warps(self) -> int:
        return len(self.all_warp_ids)

    @property
    def threads_per_cta(self) -> int:
        return self.threads_per_warp * self.num_warps

    @property
    def num_compute_warps(self) -> int:
        return len(self.compute_warp_ids)

    def make_named_barriers(self) -> Tuple[pipeline.NamedBarrier, ...]:
        """Create the named barriers used by the MLA decode kernel.

        Returns (tmem_ptr_sync_bar, softmax_exchange_sync_bar, epilogue_exchange_sync_bar).
        """
        n_compute = self.num_compute_warps
        tpw = self.threads_per_warp

        # MMA warp + compute warps + correction warps synchronize TMEM pointer
        tmem_ptr_sync = pipeline.NamedBarrier(
            barrier_id=self.tmem_ptr_sync_bar_id,
            num_threads=tpw + tpw * n_compute * 2,
        )
        # Compute warps exchange row-max during softmax
        softmax_exchange = pipeline.NamedBarrier(
            barrier_id=self.softmax_exchange_bar_id,
            num_threads=tpw * n_compute,
        )
        # Correction warps exchange row-sum during epilogue
        epilogue_exchange = pipeline.NamedBarrier(
            barrier_id=self.epilogue_exchange_bar_id,
            num_threads=tpw * n_compute,
        )
        return tmem_ptr_sync, softmax_exchange, epilogue_exchange


MLA_DECODE_SCHEDULE = MLAWarpSchedule()


@dataclass(frozen=True)
class MLAWarpScheduleFP8:
    """Warp role assignment and register budgets for FP8 MLA decode kernels.

    FP8 replaces the page-table loader warp with a second TMA loader warp
    (separate K and V loading), eliminating the load_pt pipeline entirely.
    """

    compute_warp_ids: Tuple[int, ...] = (0, 1, 2, 3)
    correction_warp_ids: Tuple[int, ...] = (4, 5, 6, 7)
    mma_warp_id: int = 8
    load_tma_k_warp_id: int = 9
    load_tma_v_warp_id: int = 10
    empty_warp_ids: Tuple[int, ...] = (11,)

    softmax_reg_num: int = 192
    correction_reg_num: int = 256
    other_reg_num: int = 48

    threads_per_warp: int = 32

    # Named barrier IDs (same as FP16)
    tmem_ptr_sync_bar_id: int = 1
    softmax_exchange_bar_id: int = 2
    epilogue_exchange_bar_id: int = 3

    @property
    def all_warp_ids(self) -> Tuple[int, ...]:
        return (
            *self.compute_warp_ids,
            *self.correction_warp_ids,
            self.mma_warp_id,
            self.load_tma_k_warp_id,
            self.load_tma_v_warp_id,
            *self.empty_warp_ids,
        )

    @property
    def num_warps(self) -> int:
        return len(self.all_warp_ids)

    @property
    def threads_per_cta(self) -> int:
        return self.threads_per_warp * self.num_warps

    @property
    def num_compute_warps(self) -> int:
        return len(self.compute_warp_ids)

    def make_named_barriers(self) -> Tuple[pipeline.NamedBarrier, ...]:
        """Create the named barriers used by the FP8 MLA decode kernel.

        Returns (tmem_ptr_sync_bar, softmax_exchange_sync_bar, epilogue_exchange_sync_bar).
        """
        n_compute = self.num_compute_warps
        tpw = self.threads_per_warp

        tmem_ptr_sync = pipeline.NamedBarrier(
            barrier_id=self.tmem_ptr_sync_bar_id,
            num_threads=tpw + tpw * n_compute * 2,
        )
        softmax_exchange = pipeline.NamedBarrier(
            barrier_id=self.softmax_exchange_bar_id,
            num_threads=tpw * n_compute,
        )
        epilogue_exchange = pipeline.NamedBarrier(
            barrier_id=self.epilogue_exchange_bar_id,
            num_threads=tpw * n_compute,
        )
        return tmem_ptr_sync, softmax_exchange, epilogue_exchange


MLA_DECODE_FP8_SCHEDULE = MLAWarpScheduleFP8()
