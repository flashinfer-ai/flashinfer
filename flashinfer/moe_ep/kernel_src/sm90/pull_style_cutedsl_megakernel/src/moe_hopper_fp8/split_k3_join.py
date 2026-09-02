# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Cross-rank K2 completion join for the split Hopper MegaMoE graph."""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64

from src.ptx_helpers import red_add_release_sys_s32_raw
from src.sym_buffer import SingleRankSymBufferDevice


class SplitK2GlobalJoin:
    """Publish local K2 completion and wait for every rank before K3.

    The parent CUDA graph orders this kernel after the local K2 child graph.
    Each lane then release-adds one arrival into the corresponding peer's
    symmetric counter.  Lane zero waits with system-scope acquire loads until
    all ranks have arrived, making every peer's K2 combine writes visible to K3.
    """

    _threads = 32

    def __init__(self, world_size: int, local_rank: int):
        if not isinstance(world_size, int) or not 1 <= world_size <= self._threads:
            raise ValueError(f"world_size must be in [1, {self._threads}], got {world_size}")
        if not isinstance(local_rank, int) or not 0 <= local_rank < world_size:
            raise ValueError(
                f"local_rank must be in [0, {world_size}), got {local_rank}"
            )
        self.world_size = world_size
        self.local_rank = local_rank

    @cute.jit
    def __call__(
        self,
        join_counter: cute.Tensor,
        peer_rank_ptr_mapper_host: Any,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self.world_size == 1):
            peer_rank_ptr_mapper = SingleRankSymBufferDevice()
        else:
            peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()
        self._join_kernel(join_counter, peer_rank_ptr_mapper).launch(
            grid=[1, 1, 1],
            block=[self._threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def _join_kernel(
        self,
        join_counter: cute.Tensor,
        peer_rank_ptr_mapper: Any,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()

        cute.arch.fence_acq_rel_sys()
        if lane < self.world_size:
            peer_counter_addr = peer_rank_ptr_mapper.map(
                join_counter.iterator.toint(), lane, Int64(0)
            )
            red_add_release_sys_s32_raw(peer_counter_addr, Int32(1))
        cute.arch.sync_warp()

        if tidx == 0:
            arrived = cute.arch.load(
                join_counter.iterator, Int32, sem="acquire", scope="sys"
            )
            while arrived < self.world_size:
                arrived = cute.arch.load(
                    join_counter.iterator, Int32, sem="acquire", scope="sys"
                )
            cute.arch.fence_acq_rel_sys()
