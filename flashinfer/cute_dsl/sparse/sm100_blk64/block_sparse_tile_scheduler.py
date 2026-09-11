# Copyright (c) 2025, Tri Dao.

from typing import Tuple, Protocol, runtime_checkable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor

from .tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
    WorkTileInfo,
)
from .cute_dsl_utils import ParamsBase


@runtime_checkable
class TileSchedulerProtocol(Protocol):
    """Protocol defining the interface all tile schedulers must implement."""

    def get_current_work(self) -> "WorkTileInfo": ...
    def initial_work_tile_info(self) -> "WorkTileInfo": ...
    def advance_to_next_work(self, *, mbarrier_addr=None) -> None: ...
    def prefetch_next_work(self) -> None: ...
    def consumer_advance(self) -> "WorkTileInfo": ...


class BlockSparsePersistentTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block_divmod: FastDivmodDivisor
        num_head_divmod: FastDivmodDivisor
        num_splits_divmod: FastDivmodDivisor
        total_base_blocks: Int32
        total_blocks: Int32
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        is_split_kv: cutlass.Constexpr[bool] = False
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC

        @staticmethod
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "BlockSparsePersistentTileScheduler.Params":
            total_base_blocks = args.num_block * args.num_head * args.num_batch
            total_blocks = (
                total_base_blocks
                if scheduling_mode == SchedulingMode.CLC
                else total_base_blocks * (args.num_splits if args.is_split_kv else 1)
            )
            return BlockSparsePersistentTileScheduler.Params(
                FastDivmodDivisor(args.num_block),
                FastDivmodDivisor(args.num_head),
                FastDivmodDivisor(args.num_splits),
                total_base_blocks,
                total_blocks,
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                is_split_kv=args.is_split_kv,
                scheduling_mode=scheduling_mode,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        clc_scheduler=None,
        clc_pipeline=None,
        clc_consumer_state=None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
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
        loc=None,
        ip=None,
    ) -> Params:
        return BlockSparsePersistentTileScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, clc_response_ptr=None, *, loc=None, ip=None
    ) -> "BlockSparsePersistentTileScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            from cutlass.utils import (
                ClcDynamicPersistentTileScheduler,
                ClcDynamicPersistentTileSchedulerParams,
            )

            problem_shape_ntile_mnl = (
                (
                    params.total_base_blocks,
                    params.num_splits,
                    Int32(1),
                )
                if const_expr(params.is_split_kv)
                else (
                    params.num_block,
                    params.num_head,
                    params.num_batch,
                )
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
            return BlockSparsePersistentTileScheduler(
                params, block_idx[0], clc_scheduler, loc=loc, ip=ip
            )
        # Static path
        return BlockSparsePersistentTileScheduler(
            params, cute.arch.block_idx()[0], loc=loc, ip=ip
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
            if const_expr(params.is_split_kv):
                return (
                    params.total_base_blocks,
                    params.num_splits,
                    Int32(1),
                )
            return (
                params.num_block,
                params.num_head,
                params.num_batch,
            )
        hardware_info = cutlass.utils.HardwareInfo()
        sm_count = hardware_info.get_device_multiprocessor_count()
        grid_x = cutlass.min(sm_count, params.total_blocks)
        return (grid_x, Int32(1), Int32(1))

    @cute.jit
    def _clc_work_to_coords(self, work) -> WorkTileInfo:
        """Convert a CLC response to block-sparse attention work coordinates."""
        if const_expr(self.params.is_split_kv):
            hn_idx, block_idx = divmod(
                Int32(work.tile_idx[0]), self.params.num_block_divmod
            )
            batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
            split_idx = Int32(work.tile_idx[1])
        else:
            block_idx = Int32(work.tile_idx[0])
            head_idx = Int32(work.tile_idx[1])
            batch_idx = Int32(work.tile_idx[2])
            split_idx = Int32(0)
        return WorkTileInfo(
            (
                Int32(block_idx),
                Int32(head_idx),
                Int32(batch_idx),
                Int32(split_idx),
            ),
            work.is_valid_tile,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            work = self._clc_scheduler.get_current_work()
            self._tile_idx = work.tile_idx[0]
            return self._clc_work_to_coords(work)
        if const_expr(self.params.is_split_kv):
            base_idx, split_idx = divmod(self._tile_idx, self.params.num_splits_divmod)
        else:
            base_idx = self._tile_idx
            split_idx = Int32(0)
        hn_idx, block_idx = divmod(base_idx, self.params.num_block_divmod)
        batch_idx, head_idx = divmod(hn_idx, self.params.num_head_divmod)
        is_valid = self._tile_idx < self.params.total_blocks
        return WorkTileInfo(
            (
                Int32(block_idx),
                Int32(head_idx),
                Int32(batch_idx),
                Int32(split_idx),
            ),
            is_valid,
        )

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
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            # All warps must converge before consuming the next CLC response.
            cute.arch.sync_threads()
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

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._clc_scheduler, self._clc_pipeline, self._clc_consumer_state]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self._clc_scheduler, self._clc_pipeline, self._clc_consumer_state]
        for obj, n_items in zip(objs, self._values_pos, strict=False):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return BlockSparsePersistentTileScheduler(*(tuple(obj_list)), loc=self._loc)
