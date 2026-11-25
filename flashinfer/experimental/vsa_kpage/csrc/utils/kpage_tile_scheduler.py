# ruff: noqa: B905
# Preserve upstream CuTe DSL constructs during the KPage extraction.
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Tri Dao.
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modifications are licensed under Apache-2.0. Pre-existing code retains
# its MIT terms; see LICENSING.md and THIRD_PARTY_LICENSES.txt.

# Persistent tile scheduler for the KPage forward kernels.  Same as
# BlockSparsePersistentTileScheduler without split-KV, plus an optional work
# queue: when ``work_indices`` is given, the scheduler iterates over that
# compact list of flattened (batch, head, q_block) ids instead of the full
# grid, so Q blocks routed to the other kernel (or empty ones) cost nothing.

from typing import Optional, Tuple
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass._mlir.dialects import nvvm
from flashinfer.experimental.vsa_kpage._cutlass_compat import FastDivmodDivisor

from .tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
    WorkTileInfo,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.cute_dsl_utils import ParamsBase


class KPagePersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        total_blocks: Int32
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        has_work_queue: cutlass.Constexpr[bool] = False
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC

        @staticmethod
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            num_work: Optional[Int32] = None,
            loc=None,
            ip=None,
        ) -> "KPagePersistentTileScheduler.Params":
            total_blocks = (
                args.num_block * args.num_head * args.num_batch
                if num_work is None
                else num_work
            )
            return KPagePersistentTileScheduler.Params(
                FastDivmodDivisor(args.num_block),
                FastDivmodDivisor(args.num_head),
                total_blocks,
                args.num_block,
                args.num_head,
                args.num_batch,
                has_work_queue=num_work is not None,
                scheduling_mode=scheduling_mode,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        work_indices: Optional[cute.Tensor] = None,
        clc_scheduler=None,
        clc_pipeline=None,
        clc_consumer_state=None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._work_indices = work_indices
        self._clc_scheduler = clc_scheduler
        self._clc_pipeline = clc_pipeline
        self._clc_consumer_state = clc_consumer_state
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(
        args: TileSchedulerArguments,
        *,
        scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
        num_work: Optional[Int32] = None,
        loc=None,
        ip=None,
    ) -> Params:
        return KPagePersistentTileScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, num_work=num_work, loc=loc, ip=ip
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        clc_response_ptr=None,
        work_indices: Optional[cute.Tensor] = None,
        *,
        loc=None,
        ip=None,
    ) -> "KPagePersistentTileScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            from cutlass.utils import (
                ClcDynamicPersistentTileScheduler,
                ClcDynamicPersistentTileSchedulerParams,
            )

            problem_shape_ntile_mnl = (
                (params.total_blocks, Int32(1), Int32(1))
                if const_expr(params.has_work_queue)
                else (params.num_block, params.num_head, params.num_batch)
            )
            cutlass_params = ClcDynamicPersistentTileSchedulerParams(
                problem_shape_ntile_mnl=problem_shape_ntile_mnl,
                cluster_shape_mnk=(1, 1, 1),
            )
            block_idx = cute.arch.block_idx()
            grid_dim = cute.arch.grid_dim()
            clc_scheduler = ClcDynamicPersistentTileScheduler.create(
                cutlass_params,
                block_idx,
                grid_dim,
                clc_response_ptr,
            )
            return KPagePersistentTileScheduler(
                params, block_idx[0], work_indices, clc_scheduler, loc=loc, ip=ip
            )
        return KPagePersistentTileScheduler(
            params, cute.arch.block_idx()[0], work_indices, loc=loc, ip=ip
        )

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            if const_expr(params.has_work_queue):
                return (params.total_blocks, Int32(1), Int32(1))
            return (params.num_block, params.num_head, params.num_batch)
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        grid_x = cutlass.min(sm_count, params.total_blocks)
        return (grid_x, Int32(1), Int32(1))

    @cute.jit
    def _coords(self, flat_idx: Int32, is_valid) -> WorkTileInfo:
        hn_idx, block_idx = divmod(flat_idx, self.params.num_block_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        return WorkTileInfo(
            (Int32(block_idx), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    @cute.jit
    def _clc_work_to_coords(self, work) -> WorkTileInfo:
        if const_expr(self._work_indices is None):
            return WorkTileInfo(
                (
                    Int32(work.tile_idx[0]),
                    Int32(work.tile_idx[1]),
                    Int32(work.tile_idx[2]),
                    Int32(0),
                ),
                work.is_valid_tile,
            )
        return self._coords(
            self._work_indices[Int32(work.tile_idx[0])], work.is_valid_tile
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self._clc_scheduler.get_current_work()
            self._tile_idx = work.tile_idx[0]
            return self._clc_work_to_coords(work)
        tile_idx = self._tile_idx
        if const_expr(self._work_indices is not None):
            # consumer_advance calls this once past the last work item; clamp
            # the speculative queue load, is_valid still reports the overrun.
            tile_idx = self._work_indices[
                cutlass.min(tile_idx, self.params.total_blocks - 1)
            ]
        return self._coords(tile_idx, self._tile_idx < self.params.total_blocks)

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self._clc_scheduler.initial_work_tile_info()
            self._tile_idx = work.tile_idx[0]
            return self._clc_work_to_coords(work)
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None, mbarrier_addr=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            assert mbarrier_addr is not None
            self._clc_scheduler.advance_to_next_work(mbarrier_addr)
        else:
            assert mbarrier_addr is None
            self._tile_idx += cute.arch.grid_dim()[0]

    def consumer_advance(self, *, loc=None, ip=None):
        # All warps converge before the next work item is consumed. Every warp role arrives from its own loop,
        # so this must be a non-aligned barrier: `bar.sync.aligned` (cute.arch.sync_threads) requires one call site.
        nvvm.barrier_cta_sync(
            Int32(0).ir_value(),
            thread_count=Int32(cute.size(cute.arch.block_dim())).ir_value(),
            aligned=False,
        )
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self._clc_pipeline.consumer_wait(self._clc_consumer_state)
            work_tile = self.get_current_work()
            self._clc_pipeline.consumer_release(self._clc_consumer_state)
            self._clc_consumer_state.advance()
            return work_tile
        self.advance_to_next_work()
        return self.get_current_work()

    def set_clc_pipeline(self, clc_pipeline, clc_consumer_state):
        self._clc_pipeline = clc_pipeline
        self._clc_consumer_state = clc_consumer_state

    def _mlir_objs(self):
        objs = [self.params, self._tile_idx]
        if const_expr(self._work_indices is not None):
            objs.append(self._work_indices)
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._clc_scheduler, self._clc_pipeline, self._clc_consumer_state]
        return objs

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in self._mlir_objs():
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(self._mlir_objs(), self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        params, tile_idx = obj_list[:2]
        pos = 2
        work_indices = None
        if const_expr(self._work_indices is not None):
            work_indices = obj_list[pos]
            pos += 1
        clc = (
            tuple(obj_list[pos : pos + 3])
            if const_expr(self.params.scheduling_mode == SchedulingMode.CLC)
            else (None, None, None)
        )
        return KPagePersistentTileScheduler(
            params, tile_idx, work_indices, *clc, loc=self._loc
        )
