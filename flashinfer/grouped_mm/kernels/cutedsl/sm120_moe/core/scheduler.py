# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM120 persistent tile scheduler for the static-tiling family."""
import enum

import cutlass
from cutlass.cutlass_dsl import Int32, Boolean, extract_mlir_values, new_from_mlir_values, dsl_user_op
from cutlass._mlir import ir
import cutlass.cute as cute

from ._common import ceil_div


class GemmType(enum.IntEnum):
    NORMAL = 0
    BATCHED = 1
    MGROUPED_CONTIGUOUS = 2
    MGROUPED_MASKED = 3
    MGROUPED_CONTIGUOUS_WITH_PSUM_LAYOUT = 4
    MGROUPED_CONTIGUOUS_WITH_ZERO_PADDING = 5


kDefaultNumSMs = 188


def get_num_1d_blocks_per_group(bm, bn, num_sms=None):
    if num_sms is None:
        num_sms = kDefaultNumSMs
    usage8 = 8 * bm + ((num_sms + 7) // 8) * bn
    usage16 = 16 * bm + ((num_sms + 15) // 16) * bn
    return 8 if usage8 <= usage16 else 16


def swizzle_block(block_idx, num_n_blocks, num_m_blocks, k_1d, min_fn, max_fn):
    num_blocks_per_group = num_n_blocks * k_1d
    group_idx = block_idx // num_blocks_per_group
    first_block_idx = group_idx * k_1d
    in_group_idx = block_idx % num_blocks_per_group
    # max(1,...): guard divisor on the terminating index (result ignored when has_next false).
    num_blocks_in_group = max_fn(1, min_fn(k_1d, num_m_blocks - first_block_idx))
    m_block = first_block_idx + in_group_idx % num_blocks_in_group
    n_block = in_group_idx // num_blocks_in_group
    return m_block, n_block


def decode_block(block_idx, gemm_type, num_m_blocks, num_n_blocks, k_1d, min_fn, max_fn):
    if gemm_type == GemmType.BATCHED:
        blocks_per_batch = num_m_blocks * num_n_blocks
        batch = block_idx // blocks_per_batch
        m_block, n_block = swizzle_block(block_idx % blocks_per_batch, num_n_blocks, num_m_blocks, k_1d, min_fn, max_fn)
        return batch, m_block, n_block
    m_block, n_block = swizzle_block(block_idx, num_n_blocks, num_m_blocks, k_1d, min_fn, max_fn)
    return 0, m_block, n_block


# ------------------------------ StaticTileScheduler: Dense / Batched ------------------------------


class StaticTileScheduler:
    def __init__(self, gemm_type, bm, bn, grid_dim_x, num_groups, num_m_blocks, num_n_blocks,
                 total_blocks, block_idx_x, current_iter, current_group_idx):
        self.gemm_type = gemm_type
        self.bm = bm
        self.bn = bn
        self.grid_dim_x = grid_dim_x
        self.num_groups = num_groups
        self.k_1d = get_num_1d_blocks_per_group(bm, bn)
        self.num_m_blocks = num_m_blocks
        self.num_n_blocks = num_n_blocks
        self.total_blocks = total_blocks
        self.block_idx_x = block_idx_x
        self.current_iter = current_iter
        self.current_group_idx = current_group_idx

    @staticmethod
    @dsl_user_op
    def create(gemm_type, bm, bn, shape_m, shape_n, num_groups, grid_dim_x, block_idx_x, *, loc=None, ip=None):
        num_m_blocks = Int32(ceil_div(shape_m, bm))
        num_n_blocks = Int32(ceil_div(shape_n, bn))
        total_blocks = num_m_blocks * num_n_blocks
        if gemm_type == GemmType.BATCHED:
            total_blocks = total_blocks * num_groups
        return StaticTileScheduler(gemm_type, bm, bn, grid_dim_x, num_groups,
                                   num_m_blocks, num_n_blocks, total_blocks, Int32(block_idx_x), Int32(-1), Int32(0))

    def __extract_mlir_values__(self):
        values = extract_mlir_values(self.num_m_blocks)
        values.extend(extract_mlir_values(self.num_n_blocks))
        values.extend(extract_mlir_values(self.total_blocks))
        values.extend(extract_mlir_values(self.block_idx_x))
        values.extend(extract_mlir_values(self.current_iter))
        values.extend(extract_mlir_values(self.current_group_idx))
        return values

    def __new_from_mlir_values__(self, values):
        fields = [self.num_m_blocks, self.num_n_blocks, self.total_blocks,
                  self.block_idx_x, self.current_iter, self.current_group_idx]
        new_fields = []
        i = 0
        for f in fields:
            w = len(extract_mlir_values(f))
            new_fields.append(new_from_mlir_values(f, values[i:i + w]))
            i += w
        return StaticTileScheduler(self.gemm_type, self.bm, self.bn, self.grid_dim_x, self.num_groups, *new_fields)

    @dsl_user_op
    @cute.jit
    def get_next_block(self, *, loc=None, ip=None):
        self.current_iter = self.current_iter + Int32(1)
        next_block_idx = self.current_iter * self.grid_dim_x + self.block_idx_x
        batch, m, n = decode_block(next_block_idx, self.gemm_type, self.num_m_blocks, self.num_n_blocks,
                                   self.k_1d, cute.min, cute.max)
        self.current_group_idx = batch
        return next_block_idx < self.total_blocks, m, n

    @dsl_user_op
    @cute.jit
    def get_group_idx(self, *, loc=None, ip=None):
        return self.current_group_idx


# ------------------------------- MoeTileScheduler: MoE token-packed -------------------------------


MoeSchedStages = 2


class MoeWorkTile:
    def __init__(self, m_block, n_block, group, m_offset, m_boundary, valid):
        self.m_block, self.n_block, self.group = m_block, n_block, group
        self.m_offset, self.m_boundary, self.valid = m_offset, m_boundary, valid

    FIELDS = 6

    def store(self, work: cute.Tensor, sched_stage):
        for i, v in enumerate((self.m_block, self.n_block, self.group,
                               self.m_offset, self.m_boundary, self.valid)):
            work[sched_stage, i] = v

    @staticmethod
    def load(work: cute.Tensor, sched_stage) -> "MoeWorkTile":
        return MoeWorkTile(*(work[sched_stage, i] for i in range(MoeWorkTile.FIELDS)))

    def __extract_mlir_values__(self):
        vals = []
        for f in (self.m_block, self.n_block, self.group, self.m_offset, self.m_boundary, self.valid):
            vals.extend(extract_mlir_values(f))
        return vals

    def __new_from_mlir_values__(self, values):
        fields = [self.m_block, self.n_block, self.group, self.m_offset, self.m_boundary, self.valid]
        new_fields, i = [], 0
        for f in fields:
            w = len(extract_mlir_values(f))
            new_fields.append(new_from_mlir_values(f, values[i:i + w]))
            i += w
        return MoeWorkTile(*new_fields)


class MoeSchedProducer:
    def __init__(self, stages, sched_iter):
        self.stages = stages
        self.sched_iter = sched_iter

    @staticmethod
    def create(stages) -> "MoeSchedProducer":
        return MoeSchedProducer(stages, Int32(0))

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.sched_iter)

    def __new_from_mlir_values__(self, values):
        return MoeSchedProducer(self.stages, new_from_mlir_values(self.sched_iter, values))

    @dsl_user_op
    @cute.jit
    def publish(self, work: cute.Tensor, full, empty, tile: MoeWorkTile, *, loc=None, ip=None):
        sched_stage = self.sched_iter % self.stages
        cute.arch.mbarrier_wait(empty + sched_stage, ((self.sched_iter // self.stages) & 1) ^ 1)
        with cute.arch.elect_one():
            tile.store(work, sched_stage)
            cute.arch.mbarrier_arrive(full + sched_stage)
        self.sched_iter = self.sched_iter + Int32(1)

    @dsl_user_op
    @cute.jit
    def publish_sentinel(self, work: cute.Tensor, full, empty, *, loc=None, ip=None):
        sched_stage = self.sched_iter % self.stages
        cute.arch.mbarrier_wait(empty + sched_stage, ((self.sched_iter // self.stages) & 1) ^ 1)
        with cute.arch.elect_one():
            work[sched_stage, MoeWorkTile.FIELDS - 1] = Int32(0)
            cute.arch.mbarrier_arrive(full + sched_stage)


class MoeSchedConsumer:
    def __init__(self, stages, sched_iter):
        self.stages = stages
        self.sched_iter = sched_iter

    @staticmethod
    def create(stages) -> "MoeSchedConsumer":
        return MoeSchedConsumer(stages, Int32(0))

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.sched_iter)

    def __new_from_mlir_values__(self, values):
        return MoeSchedConsumer(self.stages, new_from_mlir_values(self.sched_iter, values))

    @dsl_user_op
    @cute.jit
    def get_next_tile(self, work: cute.Tensor, full, empty, *, loc=None, ip=None) -> MoeWorkTile:
        sched_stage = self.sched_iter % self.stages
        cute.arch.mbarrier_wait(full + sched_stage, (self.sched_iter // self.stages) & 1)
        tile = MoeWorkTile.load(work, sched_stage)
        cute.arch.sync_warp()  # release only after every lane owns its copy
        if tile.valid != Int32(0):  # the sentinel is never released
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(empty + sched_stage)
            self.sched_iter = self.sched_iter + Int32(1)
        return tile


class MoeTileScheduler:
    def __init__(self, bm, grid_dim_x, num_groups, num_n_blocks, block_idx_x, current_iter,
                 current_group_idx, prev_psum_m, current_psum_m, window_base, window_start_block, lane_off, lane_next):
        self.bm = bm
        self.grid_dim_x = grid_dim_x
        self.num_groups = num_groups
        self.num_n_blocks = num_n_blocks
        self.block_idx_x = block_idx_x
        self.current_iter = current_iter
        self.current_group_idx = current_group_idx
        self.prev_psum_m = prev_psum_m
        self.current_psum_m = current_psum_m
        self.window_base = window_base
        self.window_start_block = window_start_block
        self.lane_off = lane_off
        self.lane_next = lane_next

    @staticmethod
    @dsl_user_op
    def create(bm, num_groups, num_n_blocks, grid_dim_x, block_idx_x, offsets, *, loc=None, ip=None):
        ng = Int32(num_groups)
        lane = cute.arch.lane_idx()
        lane_off = offsets[cute.min(lane, ng)]
        lane_next = offsets[cute.min(lane + Int32(1), ng)]
        return MoeTileScheduler(bm, grid_dim_x, ng, Int32(num_n_blocks), Int32(block_idx_x), Int32(-1),
                                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), lane_off, lane_next)

    def __extract_mlir_values__(self):
        vals = []
        for f in (self.num_groups, self.num_n_blocks, self.block_idx_x, self.current_iter, self.current_group_idx,
                  self.prev_psum_m, self.current_psum_m, self.window_base, self.window_start_block,
                  self.lane_off, self.lane_next):
            vals.extend(extract_mlir_values(f))
        return vals

    def __new_from_mlir_values__(self, values):
        fields = [self.num_groups, self.num_n_blocks, self.block_idx_x, self.current_iter, self.current_group_idx,
                  self.prev_psum_m, self.current_psum_m, self.window_base, self.window_start_block,
                  self.lane_off, self.lane_next]
        new_fields, i = [], 0
        for f in fields:
            w = len(extract_mlir_values(f))
            new_fields.append(new_from_mlir_values(f, values[i:i + w]))
            i += w
        return MoeTileScheduler(self.bm, self.grid_dim_x, *new_fields)

    @dsl_user_op
    @cute.jit
    def get_next_block(self, offsets: cute.Tensor, *, loc=None, ip=None):
        self.current_iter = self.current_iter + Int32(1)
        nbi = self.current_iter * self.grid_dim_x + self.block_idx_x
        target_m_block = nbi // self.num_n_blocks
        n_block = nbi % self.num_n_blocks
        lane = cute.arch.lane_idx()
        wbase, wstart = self.window_base, self.window_start_block
        loff, lnext = self.lane_off, self.lane_next
        cg, po, cp = self.current_group_idx, self.prev_psum_m, self.current_psum_m
        m_block, has, done = Int32(0), Boolean(False), Int32(0)
        while done == Int32(0):
            my_blocks = ceil_div(lnext - loff, self.bm)
            incl = my_blocks
            for i in cutlass.range_constexpr(5):
                d = 1 << i
                up = cute.arch.shuffle_sync_up(incl, Int32(d), mask_and_clamp=0)
                incl = incl + up * (Int32(1) + ((lane - Int32(d)) >> Int32(31)))
            window_total = cute.arch.shuffle_sync(incl, Int32(31))
            local = target_m_block - wstart
            owners = cute.arch.vote_ballot_sync(incl > local)
            src = cute.arch.popc((owners & (Int32(0) - owners)) - Int32(1))
            group_start = cute.arch.shuffle_sync(incl - my_blocks, src)
            landed = target_m_block < wstart + window_total
            exhausted = (wbase + Int32(32)) >= self.num_groups
            if landed:
                cg = wbase + src
                po = cute.arch.shuffle_sync(loff, src)
                cp = cute.arch.shuffle_sync(lnext, src)
                m_block = local - group_start
                has = Boolean(True)
                done = Int32(1)
            else:
                if exhausted:
                    has = Boolean(False)
                    done = Int32(1)
                else:
                    wstart = wstart + window_total
                    wbase = wbase + Int32(32)
                    g2 = wbase + lane
                    loff = offsets[cute.min(g2, self.num_groups)]
                    lnext = offsets[cute.min(g2 + Int32(1), self.num_groups)]
        self.window_base, self.window_start_block = wbase, wstart
        self.lane_off, self.lane_next = loff, lnext
        self.current_group_idx, self.prev_psum_m, self.current_psum_m = cg, po, cp
        return has, cg, m_block, n_block, po, cp
