# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Cross-rank epoch barrier following split-MegaMoE graph reset work."""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64

from src.ptx_helpers import red_add_release_sys_s32_raw
from src.sym_buffer import SingleRankSymBufferDevice


class SplitEpochResetBarrier:
    """Publish completed K0 resets before either Green branch may start.

    The two signal slots and monotonically advancing phase counter are outside
    the reset prefixes. Four calls form a +1/+1/-1/-1 cycle, so the protocol
    is graph-replay safe without a host-side epoch reset.
    """

    _threads = 32

    def __init__(self, world_size: int, local_rank: int):
        if not isinstance(world_size, int) or not 1 <= world_size <= self._threads:
            raise ValueError(
                f"world_size must be in [1, {self._threads}], got {world_size}"
            )
        if not isinstance(local_rank, int) or not 0 <= local_rank < world_size:
            raise ValueError(
                f"local_rank must be in [0, {world_size}), got {local_rank}"
            )
        self.world_size = world_size
        self.local_rank = local_rank

    @cute.jit
    def __call__(
        self,
        barrier_signal: cute.Tensor,
        phase_counter: cute.Tensor,
        peer_rank_ptr_mapper_host: Any,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.world_size == 1):
            peer_rank_ptr_mapper = SingleRankSymBufferDevice()
        else:
            peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()
        self._barrier_kernel(
            barrier_signal, phase_counter, peer_rank_ptr_mapper
        ).launch(
            grid=[1, 1, 1],
            block=[self._threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def _barrier_kernel(
        self,
        barrier_signal: cute.Tensor,
        phase_counter: cute.Tensor,
        peer_rank_ptr_mapper: Any,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()

        # The preceding async memsets are stream-ordered before this kernel;
        # the system fence publishes their writes before any peer can leave K0.
        cute.arch.fence_acq_rel_sys()
        status = phase_counter[0] & Int32(3)
        signal_phase = status & Int32(1)
        signal_sign = status >> Int32(1)
        signal_delta = Int32(1)
        target = Int32(self.world_size)
        if signal_sign != Int32(0):
            signal_delta = Int32(-1)
            target = Int32(0)

        if lane < self.world_size:
            peer_signal_addr = peer_rank_ptr_mapper.map(
                barrier_signal.iterator.toint(),
                lane,
                Int64(signal_phase * Int32(4)),
            )
            red_add_release_sys_s32_raw(peer_signal_addr, signal_delta)
        cute.arch.sync_warp()

        if tidx == 0:
            cute.arch.atomic_add(
                phase_counter.iterator,
                Int32(1),
                sem="relaxed",
                scope="gpu",
            )
            local_signal = barrier_signal.iterator + signal_phase
            arrived = cute.arch.load(
                local_signal, Int32, sem="acquire", scope="sys"
            )
            while arrived != target:
                arrived = cute.arch.load(
                    local_signal, Int32, sem="acquire", scope="sys"
                )
            cute.arch.fence_acq_rel_sys()


__all__ = ["SplitEpochResetBarrier"]
