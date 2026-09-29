# Copyright (c) 2025, Tri Dao, Siyu Wang, Shengbin Di, Yuxi Chi, Johnsonms, Linfeng Zheng, Haoyan Huang, Lanbo Li, Yun Zhong, Man Yuan, Minmin Sun, Yong Li, Wei Lin.

from enum import IntEnum, auto
from typing import Optional, Tuple
from dataclasses import dataclass

from typing_extensions import override

import cutlass
from cutlass.pipeline import PipelineClcFetchAsync, PipelineState
from cutlass._mlir import ir
import cutlass.cute as cute
from cutlass import Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.utils import (
    ClcDynamicPersistentTileScheduler,
    ClcDynamicPersistentTileSchedulerParams,
)

from .cute_dsl_utils import ParamsBase

from . import kernel_utils as utils


class SchedulingMode(IntEnum):
    NONE = auto()
    STATIC = auto()
    DYNAMIC = auto()
    CLC = auto()


@dataclass
class ClcState(ParamsBase):
    """Owns the runtime state shared by CLC-capable tile schedulers.

    `BlockSparseAttnForwardSm100Blk128` constructs this state because it owns the CLC
    response buffer, mbarrier storage, and launch geometry needed to initialize
    the hardware scheduler and async pipeline. Individual tile schedulers then
    consume this state and map the returned hardware work tiles into their own
    logical `WorkTileInfo` coordinates.

    To add CLC support to a scheduler:
    - implement `clc_problem_shape(params)` so the kernel can create the hardware scheduler
    - accept `clc: ClcState | None` in `create(...)` / `__init__`
    - map `clc.initial_work_tile_info()` and `clc.get_current_work()` into scheduler coordinates
    """

    _hw_scheduler: ClcDynamicPersistentTileScheduler
    _pipeline: PipelineClcFetchAsync
    _consumer_state: PipelineState
    _producer_state: PipelineState

    @staticmethod
    def create(
        *,
        hw_scheduler: ClcDynamicPersistentTileScheduler,
        pipeline: PipelineClcFetchAsync,
        consumer_state: PipelineState,
        producer_state: PipelineState,
    ) -> "ClcState":
        return ClcState(hw_scheduler, pipeline, consumer_state, producer_state)

    def initial_work_tile_info(self):
        return self._hw_scheduler.initial_work_tile_info()

    def get_current_work(self):
        return self._hw_scheduler.get_current_work()

    def prefetch_next_work(self, *, loc=None, ip=None):
        self._pipeline.producer_acquire(self._producer_state, loc=loc, ip=ip)
        mbarrier_addr = self._pipeline.producer_get_barrier(
            self._producer_state, loc=loc, ip=ip
        )
        self._hw_scheduler.advance_to_next_work(mbarrier_addr, loc=loc, ip=ip)
        self._producer_state.advance(loc=loc, ip=ip)

    def consumer_wait(self, *, loc=None, ip=None):
        self._pipeline.consumer_wait(self._consumer_state, loc=loc, ip=ip)

    def consumer_release(self, *, loc=None, ip=None):
        self._pipeline.consumer_release(self._consumer_state, loc=loc, ip=ip)
        self._consumer_state.advance(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        self._pipeline.producer_tail(self._producer_state, loc=loc, ip=ip)


class WorkTileInfo(cutlass.utils.WorkTileInfo):
    """Altered WorkTileInfo which includes four axes: (block, head, batch, split)"""

    @override
    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "WorkTileInfo":
        assert len(values) == 5
        new_tile_idx = cutlass.new_from_mlir_values(self._tile_idx, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(
            self._is_valid_tile, [values[-1]]
        )
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)


@dataclass
class TileSchedulerArguments(ParamsBase):
    num_block: Int32
    num_head: Int32
    num_batch: Int32
    num_splits: Int32
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    mCuSeqlensQ: Optional[cute.Tensor] = None
    mSeqUsedQ: Optional[cute.Tensor] = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        num_splits: Int32
        num_splits_divmod: FastDivmodDivisor
        is_split_kv: cutlass.Constexpr[bool] = False

        @staticmethod
        def create(
            args: TileSchedulerArguments, *, loc=None, ip=None
        ) -> "SingleTileScheduler.Params":
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.num_splits,
                FastDivmodDivisor(args.num_splits),
                args.is_split_kv,
            )

    def __init__(self, params: Params, blk_coord: cute.Coord, *, loc=None, ip=None):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
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
        assert scheduling_mode == SchedulingMode.STATIC, (
            f"SingleTileScheduler only supports STATIC, got {scheduling_mode!r}"
        )
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileScheduler":
        return SingleTileScheduler(params, cute.arch.block_idx(), loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        return (
            params.num_block,
            params.num_head * params.num_splits,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        block_idx, head_idx, batch_idx = self._blk_coord
        if const_expr(self.params.is_split_kv):
            head_idx, split_idx = divmod(head_idx, self.params.num_splits_divmod)
        else:
            split_idx = Int32(0)
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, split_idx),
            self._is_first_block,
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        pass

    def advance_to_next_work(self, *, loc=None, ip=None):
        self._is_first_block = False
        return self.get_current_work()

    def consumer_advance(self, *, loc=None, ip=None):
        return self.advance_to_next_work(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None):
        pass

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [self.params, self._blk_coord], self._values_pos, strict=True
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        num_splits: Int32
        max_kvblock_in_l2: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        mCuSeqlensQ: Optional[cute.Tensor] = None
        mSeqUsedQ: Optional[cute.Tensor] = None
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
        lpt: cutlass.Constexpr[bool] = False
        is_split_kv: cutlass.Constexpr[bool] = False
        head_swizzle: cutlass.Constexpr[bool] = False
        scheduling_mode: cutlass.Constexpr[SchedulingMode] = SchedulingMode.STATIC

        @staticmethod
        @cute.jit
        def create(
            args: TileSchedulerArguments,
            *,
            scheduling_mode: SchedulingMode = SchedulingMode.STATIC,
            loc=None,
            ip=None,
        ) -> "SingleTileVarlenScheduler.Params":
            assert scheduling_mode in (SchedulingMode.STATIC, SchedulingMode.CLC), (
                f"Only STATIC and CLC are supported, got {scheduling_mode!r}"
            )
            size_l2 = 50 * 1024 * 1024  # 50 MB for K & V
            # if backward, this is qdo block size
            kv_block_size = (
                (args.headdim + args.headdim_v)
                * args.element_size
                * args.tile_shape_mn[1]
            )
            # if backward, add dqaccum block size to calculate swizzle
            if args.head_swizzle:
                kv_block_size += args.headdim * 4 * args.tile_shape_mn[1]
            max_kvblock_in_l2 = size_l2 // kv_block_size
            assert args.mCuSeqlensQ is not None or args.mSeqUsedQ is not None, (
                "At least one of mCuSeqlensQ or mSeqUsedQ must be provided"
            )
            return SingleTileVarlenScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                num_splits=args.num_splits,
                max_kvblock_in_l2=max_kvblock_in_l2,
                tile_shape_mn=args.tile_shape_mn,
                mCuSeqlensQ=args.mCuSeqlensQ,
                mSeqUsedQ=args.mSeqUsedQ,
                qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
                lpt=args.lpt,
                is_split_kv=args.is_split_kv,
                head_swizzle=args.head_swizzle,
                scheduling_mode=scheduling_mode,
            )

    def __init__(
        self,
        params: Params,
        tile_idx: Int32,
        split_idx: Int32,
        clc: ClcState | None = None,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._tile_idx = tile_idx
        self._split_idx = split_idx
        self._is_first_block = True
        self.clc = clc
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
        return SingleTileVarlenScheduler.Params.create(
            args, scheduling_mode=scheduling_mode, loc=loc, ip=ip
        )

    @staticmethod
    @cute.jit
    def clc_problem_shape(params: Params):
        return ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl=SingleTileVarlenScheduler.get_grid_shape(params),
            cluster_shape_mnk=(1, 1, 1),
        )

    @staticmethod
    @cute.jit
    def create(
        params: Params, clc: ClcState | None = None, *, loc=None, ip=None
    ) -> "SingleTileVarlenScheduler":
        if const_expr(params.scheduling_mode == SchedulingMode.CLC):
            block_idx = cute.arch.block_idx()
            split_idx = Int32(0)
            if const_expr(params.is_split_kv):
                split_idx = block_idx[1]
            return SingleTileVarlenScheduler(
                params,
                block_idx[0],
                split_idx,
                clc,
                loc=loc,
                ip=ip,
            )
        tile_idx, split_idx, _ = cute.arch.block_idx()
        return SingleTileVarlenScheduler(params, tile_idx, split_idx, loc=loc, ip=ip)

    # called by host
    @staticmethod
    def get_grid_shape(
        params: Params,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Int32, Int32, Int32]:
        total_blocks_max = (
            params.total_q + params.num_batch * (params.tile_shape_mn[0] - 1)
        ) // params.tile_shape_mn[0]
        return (total_blocks_max * params.num_head, params.num_splits, Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        if cutlass.const_expr(params.mSeqUsedQ is not None):
            seqlen = Int32(0)
            if batch_idx < params.num_batch:
                seqlen = params.mSeqUsedQ[batch_idx]
        else:
            assert params.mCuSeqlensQ is not None
            cur_cu_seqlen = Int32(0)
            if batch_idx <= params.num_batch:
                cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
            next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
            seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
            seqlen *= params.qhead_per_kvhead_packgqa
        return (
            cute.ceil_div(seqlen, params.tile_shape_mn[0])
            if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def _varlen_coord_map(self) -> WorkTileInfo:
        """Map self._tile_idx to (block, head, batch) via warp-level prefix sums."""
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
        # Total number of blocks for the next 31 batches
        m_blocks_in_group = cute.arch.shuffle_sync(
            num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
        )
        # Same for all lanes
        group_end_tile = m_blocks_in_group * params.num_head
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = self._tile_idx
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= params.num_batch:
                batch_idx = Int32(params.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
                num_m_blocks_cumulative = utils.warp_prefix_sum(num_m_blocks, lane_idx)
                m_blocks_in_group = cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
                )
                group_end_tile += m_blocks_in_group * params.num_head
        is_valid = False
        if batch_idx >= params.num_batch:
            block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
        else:
            group_start_tile = group_end_tile - m_blocks_in_group * params.num_head
            # The next problem to process is the first one that does not have ending tile position
            # that is greater than or equal to tile index.
            batch_idx_in_group = cute.arch.popc(
                cute.arch.vote_ballot_sync(
                    group_start_tile + num_m_blocks_cumulative * params.num_head
                    <= next_tile_idx
                )
            )
            batch_idx += batch_idx_in_group
            num_m_blocks_prev_lane = (
                0
                if batch_idx_in_group == 0
                else cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, batch_idx_in_group - 1
                )
            )
            num_m_blocks = cute.arch.shuffle_sync(num_m_blocks, batch_idx_in_group)
            mh_block = (
                next_tile_idx
                - group_start_tile
                - num_m_blocks_prev_lane * params.num_head
            )
            if cutlass.const_expr(params.lpt or params.head_swizzle):
                # Apply longest-processing-time ordering while accounting for
                # sequence lengths that vary by batch.
                # TODO: is there any case where num_m_blocks is 0?
                # TODO: by right we should read the seqlen_kv but we're assuming seqlen_q == seqlen_k here
                num_n_blocks = (
                    num_m_blocks
                    * params.tile_shape_mn[0]
                    // params.qhead_per_kvhead_packgqa
                    // params.tile_shape_mn[1]
                )
                # Seems faster to have this be a power of 2
                nheads_in_l2 = (
                    16
                    if num_n_blocks * 16 <= params.max_kvblock_in_l2
                    else (
                        8
                        if num_n_blocks * 8 <= params.max_kvblock_in_l2
                        else (
                            4
                            if num_n_blocks * 4 <= params.max_kvblock_in_l2
                            else (
                                2 if num_n_blocks * 2 <= params.max_kvblock_in_l2 else 1
                            )
                        )
                    )
                )
                nheads_in_l2 = min(nheads_in_l2, params.num_head)
                mh_in_l2 = nheads_in_l2 * num_m_blocks
                section_idx = mh_block // mh_in_l2
                l2_mod = mh_block - section_idx * mh_in_l2
                # Deal with tail section
                nheads_in_this_section = (
                    nheads_in_l2
                    if nheads_in_l2 * (section_idx + 1) <= params.num_head
                    else params.num_head - section_idx * nheads_in_l2
                )
                block = l2_mod // nheads_in_this_section
                head_idx_residual = l2_mod - block * nheads_in_this_section
                head_idx = section_idx * nheads_in_l2 + head_idx_residual
                if cutlass.const_expr(params.lpt):
                    block = num_m_blocks - 1 - block
            else:
                head_idx = mh_block // num_m_blocks
                block = mh_block - head_idx * num_m_blocks
            is_valid = self._is_first_block and batch_idx < params.num_batch
        split_idx = self._split_idx if const_expr(params.is_split_kv) else Int32(0)
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), split_idx), is_valid
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self.clc.get_current_work()
            # Default to grid_dim (one past last valid flat index) so _varlen_coord_map
            # returns is_valid=False when CLC is exhausted. CLC tile_idx is garbage when
            # invalid, so we can't trust it. Local-then-assign avoids CuTe DSL structural
            # mismatch on self inside the runtime if.
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        return self._varlen_coord_map()

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            clc_work = self.clc.initial_work_tile_info()
            # See get_current_work for why grid_dim and local-then-assign.
            new_tile_idx = cute.arch.grid_dim()[0]
            new_split_idx = Int32(0)
            if clc_work.is_valid_tile:
                new_tile_idx = clc_work.tile_idx[0]
                if const_expr(self.params.is_split_kv):
                    new_split_idx = clc_work.tile_idx[1]
            self._tile_idx = new_tile_idx
            self._split_idx = new_split_idx
        return self._varlen_coord_map()

    def prefetch_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.prefetch_next_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.consumer_wait(loc=loc, ip=ip)
            work = self.get_current_work()
            self.clc.consumer_release(loc=loc, ip=ip)
            return work
        self._is_first_block = False
        return self.get_current_work()

    def producer_tail(self, *, loc=None, ip=None):
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            self.clc.producer_tail(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj in objs:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        objs = [self.params, self._tile_idx, self._split_idx]
        if const_expr(self.params.scheduling_mode == SchedulingMode.CLC):
            objs += [self.clc]
        for obj, n_items in zip(objs, self._values_pos, strict=True):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*obj_list, loc=self._loc)
