# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/contiguous/tile_scheduler.py @ 16aba799 (2026-05-23) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
from dataclasses import dataclass
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir import ir

from flashinfer.experimental.sm12x.attention._shared.cute import ops as cute_ops
from flashinfer.experimental.sm12x.attention._shared.contiguous.cute_dsl_utils import (
    ParamsBase,
)


class WorkTileInfo(cutlass.utils.WorkTileInfo):
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
    seqlen_k: Int32
    headdim: Int32
    headdim_v: Int32
    total_q: Int32
    tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
    cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)
    mCuSeqlensQ: cute.Tensor | None = None
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    element_size: cutlass.Constexpr[int] = 2
    is_persistent: cutlass.Constexpr[bool] = False
    lpt: cutlass.Constexpr[bool] = False
    head_swizzle: cutlass.Constexpr[bool] = False


class SingleTileScheduler:
    @dataclass
    class Params(ParamsBase):
        num_block: Int32
        num_head: Int32
        num_batch: Int32
        cluster_shape_mn: cutlass.Constexpr[Tuple[int, int]] = (1, 1)

        @staticmethod
        def create(args: TileSchedulerArguments, *, loc=None, ip=None):
            del loc, ip
            return SingleTileScheduler.Params(
                args.num_block,
                args.num_head,
                args.num_batch,
                args.cluster_shape_mn,
            )

    def __init__(
        self,
        params: "SingleTileScheduler.Params",
        blk_coord: cute.Coord,
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self._blk_coord = blk_coord
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None):
        return SingleTileScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: "SingleTileScheduler.Params", *, loc=None, ip=None):
        return SingleTileScheduler(params, cute.arch.block_idx(), loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params: "SingleTileScheduler.Params", *, loc=None, ip=None):
        del loc, ip
        assert params.cluster_shape_mn[1] == 1
        return (
            cute.round_up(params.num_block, params.cluster_shape_mn[0]),
            params.num_head,
            params.num_batch,
        )

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        del loc, ip
        block_idx, head_idx, batch_idx = self._blk_coord
        return WorkTileInfo(
            (block_idx, head_idx, batch_idx, Int32(0)), self._is_first_block
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        del loc, ip

    def advance_to_next_work(self, *, loc=None, ip=None):
        del loc, ip
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._blk_coord]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._blk_coord], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileScheduler(*(tuple(obj_list)), loc=self._loc)


class SingleTileVarlenScheduler:
    @dataclass
    class Params(ParamsBase):
        num_head: Int32
        num_batch: Int32
        total_q: Int32
        max_kvblock_in_l2: Int32
        tile_shape_mn: cutlass.Constexpr[Tuple[int, int]]
        mCuSeqlensQ: Optional[cute.Tensor] = None
        qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
        lpt: cutlass.Constexpr[bool] = False
        head_swizzle: cutlass.Constexpr[bool] = False
        cluster_shape_m: cutlass.Constexpr[int] = 1

        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArguments, *, loc=None, ip=None):
            del loc, ip
            size_l2 = 50 * 1024 * 1024
            max_kvblock_in_l2 = size_l2 // (
                (args.headdim + args.headdim_v)
                * args.element_size
                * args.tile_shape_mn[1]
            )
            assert args.mCuSeqlensQ is not None, "mCuSeqlensQ must be provided"
            assert args.cluster_shape_mn[1] == 1, (
                "Only cluster_shape_mn[1] == 1 is supported"
            )
            return SingleTileVarlenScheduler.Params(
                num_head=args.num_head,
                num_batch=args.num_batch,
                total_q=args.total_q,
                max_kvblock_in_l2=max_kvblock_in_l2,
                tile_shape_mn=args.tile_shape_mn,
                mCuSeqlensQ=args.mCuSeqlensQ,
                qhead_per_kvhead_packgqa=args.qhead_per_kvhead_packgqa,
                lpt=args.lpt,
                head_swizzle=args.head_swizzle,
                cluster_shape_m=args.cluster_shape_mn[0],
            )

    def __init__(self, params: Params, tile_idx: Int32, *, loc=None, ip=None):
        self.params = params
        self._tile_idx = tile_idx
        self._is_first_block = True
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: TileSchedulerArguments, *, loc=None, ip=None):
        return SingleTileVarlenScheduler.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    def create(params: Params, *, loc=None, ip=None):
        tile_idx, _, _ = cute.arch.block_idx()
        return SingleTileVarlenScheduler(params, tile_idx, loc=loc, ip=ip)

    @staticmethod
    def get_grid_shape(params: Params, *, loc=None, ip=None):
        del loc, ip
        total_blocks_max = (
            params.total_q
            + params.num_batch * (params.cluster_shape_m * params.tile_shape_mn[0] - 1)
        ) // params.tile_shape_mn[0]
        total_blocks_max = (
            total_blocks_max // params.cluster_shape_m * params.cluster_shape_m
        )
        return (total_blocks_max * params.num_head, Int32(1), Int32(1))

    @cute.jit
    def _get_num_m_blocks(self, lane: Int32, bidb_start: Int32) -> Int32:
        params = self.params
        batch_idx = lane + bidb_start
        assert params.mCuSeqlensQ is not None
        cur_cu_seqlen = Int32(0)
        if batch_idx <= params.num_batch:
            cur_cu_seqlen = params.mCuSeqlensQ[batch_idx]
        next_cu_seqlen = cute.arch.shuffle_sync_down(cur_cu_seqlen, offset=1)
        seqlen = next_cu_seqlen - cur_cu_seqlen
        if cutlass.const_expr(params.qhead_per_kvhead_packgqa > 1):
            seqlen *= params.qhead_per_kvhead_packgqa
        return (
            cute.ceil_div(
                cute.ceil_div(seqlen, params.tile_shape_mn[0]), params.cluster_shape_m
            )
            if batch_idx < params.num_batch and lane < cute.arch.WARP_SIZE - 1
            else Int32(0)
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        del loc, ip
        params = self.params
        lane_idx = cute.arch.lane_idx()
        num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=0)
        num_m_blocks_cumulative = cute_ops.warp_prefix_sum(num_m_blocks, lane_idx)
        m_blocks_in_group = cute.arch.shuffle_sync(
            num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
        )
        group_end_tile = m_blocks_in_group * params.num_head
        block, head_idx, batch_idx = Int32(0), Int32(0), Int32(0)
        next_tile_idx = self._tile_idx // params.cluster_shape_m
        while group_end_tile <= next_tile_idx:
            batch_idx += cute.arch.WARP_SIZE - 1
            if batch_idx >= params.num_batch:
                batch_idx = Int32(params.num_batch)
                group_end_tile = next_tile_idx + 1
            else:
                num_m_blocks = self._get_num_m_blocks(lane_idx, bidb_start=batch_idx)
                num_m_blocks_cumulative = cute_ops.warp_prefix_sum(
                    num_m_blocks, lane_idx
                )
                m_blocks_in_group = cute.arch.shuffle_sync(
                    num_m_blocks_cumulative, cute.arch.WARP_SIZE - 1
                )
                group_end_tile += m_blocks_in_group * params.num_head
        is_valid = False
        if batch_idx >= params.num_batch:
            block, head_idx, batch_idx = Int32(0), Int32(0), Int32(params.num_batch)
        else:
            group_start_tile = group_end_tile - m_blocks_in_group * params.num_head
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
                num_n_blocks = (
                    num_m_blocks
                    * params.tile_shape_mn[0]
                    // params.qhead_per_kvhead_packgqa
                    // params.tile_shape_mn[1]
                )
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
            if cutlass.const_expr(params.cluster_shape_m > 1):
                bidx_in_cluster = cute.arch.block_in_cluster_idx()
                block = block * params.cluster_shape_m + bidx_in_cluster[0]
        return WorkTileInfo(
            (Int32(block), Int32(head_idx), Int32(batch_idx), Int32(0)), is_valid
        )

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None):
        del loc, ip

    def advance_to_next_work(self, *, loc=None, ip=None):
        del loc, ip
        self._is_first_block = False

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [self.params, self._tile_idx]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip([self.params, self._tile_idx], self._values_pos):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return SingleTileVarlenScheduler(*(tuple(obj_list)), loc=self._loc)
