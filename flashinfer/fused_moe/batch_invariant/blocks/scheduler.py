# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Group scheduler block: which CTA computes which (expert, m-block, n-tile) cluster tile.

Each CTA scans ``counts[E]`` once into two shared-memory tables (row offsets of the groups and the prefix of per-group
cluster-row-block counts) and walks the linear cluster-tile indices ``cluster_id, cluster_id + num_clusters, ...`` up to
the device-side total. A linear index decodes to its group by binary search over the prefix table. The group offsets may
be padded to a multiple of ``align`` rows (the block-scaled classes: a group's first row must start a 128-row scale-factor
atom); the store mask is the group's real end ``off_e + counts[e]`` either way. Everything here is a placement: it
decides *which* rows a tile holds, never how an element is folded (every tile restarts its accumulator). The tile count
is never read on the host, so routing-dependent launches are CUDA-graph safe and the grid size is a launch parameter
only.

The scheduler and its work tile implement the DSL's loop-carried value protocol (``__extract_mlir_values__`` /
``__new_from_mlir_values__``) and the method surface the dense ``mma_loop`` expects (``initial_work_tile_info``,
``advance_to_next_work``, ``get_current_work``, ``WorkTile.is_valid_tile``), so the MMA block is reused unchanged.
"""

from __future__ import annotations

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values

MAX_E = 1024  # experts per launch (static smem tables of MAX_E + 1 ints; a whole number of lanes)


def warp_roles(epi_warpgroups: int) -> Tuple[Tuple[int, ...], int, int, int]:
    """``(EPILOGUE_WARPS, MMA_WARP, TMA_WARP, THREADS)`` of a grouped kernel with ``epi_warpgroups`` epilogue warpgroups.
    The epilogue warpgroups split a tile's epilogue subtiles round-robin (warp ``w`` reads TMEM lanes ``32 (w % 4) ..``:
    every warpgroup covers all 128 lanes), then come the MMA warp and the TMA warp. A placement: each output element's
    chain is computed by exactly one thread from the same accumulator value, in the same fp32 sequence, whichever
    warpgroup owns its subtile."""
    epi = tuple(range(4 * epi_warpgroups))
    return epi, 4 * epi_warpgroups, 4 * epi_warpgroups + 1, 32 * (len(epi) + 2)


EPI_WARPGROUPS_DENSE = 2  # the bf16 kernels: the gated chains are the epilogue leg once the stores are vectors
PER_LANE = MAX_E // 32  # table entries scanned per lane
TABLE_SMEM_BYTES = (
    2 * (MAX_E + 1) * 4
)  # what the grouped kernels reserve for the two tables (StaticPlan extra_smem_bytes)


class GroupWorkTile:
    """One CTA tile: expert ``e``, the first row of this *cluster* row block (``row0``), the real end row of the group
    (``row_end = off_e + counts[e]``, the store predicate), the CTA's global n-tile index."""

    def __init__(self, is_valid_tile, e, row0, row_end, n_tile):
        self._is_valid_tile = Boolean(is_valid_tile)
        self.e, self.row0, self.row_end, self.n_tile = (
            Int32(e),
            Int32(row0),
            Int32(row_end),
            Int32(n_tile),
        )

    def __extract_mlir_values__(self):
        vals = []
        for v in (self._is_valid_tile, self.e, self.row0, self.row_end, self.n_tile):
            vals.extend(extract_mlir_values(v))
        return vals

    def __new_from_mlir_values__(self, values):
        assert len(values) == 5, len(values)
        return GroupWorkTile(
            new_from_mlir_values(self._is_valid_tile, [values[0]]),
            new_from_mlir_values(self.e, [values[1]]),
            new_from_mlir_values(self.row0, [values[2]]),
            new_from_mlir_values(self.row_end, [values[3]]),
            new_from_mlir_values(self.n_tile, [values[4]]),
        )

    @property
    def is_valid_tile(self):
        return self._is_valid_tile

    @property
    def tile_idx(
        self,
    ):  # the dense blocks' name for the coordinates (not used by the grouped loops)
        return (self.row0, self.n_tile, self.e)


@cute.jit
def build_group_tables(
    counts: cute.Tensor,
    E: Int32,
    cluster_rows: cutlass.Constexpr,
    s_off: cute.Tensor,
    s_pre: cute.Tensor,
    align: cutlass.Constexpr = 1,
):
    """Warp-level scan (all 32 lanes of one warp): ``s_off[e]`` = exclusive prefix of ``counts`` rounded up to ``align``
    rows (the group's first row; ``align = 1``: the dense packing), ``s_pre[e]`` = exclusive prefix of
    ``ceil(counts[e] / cluster_rows)`` (the group's first cluster-row block); entries ``e > E`` hold the totals. Integer
    sums: exact in every order."""
    lane = cute.arch.lane_idx()
    loc_c = cute.make_rmem_tensor((PER_LANE,), Int32)
    loc_b = cute.make_rmem_tensor((PER_LANE,), Int32)
    # the lane's counts first, all loads in flight together (one memory latency per CTA instead of one per entry), then the sums
    for i in cutlass.range_constexpr(PER_LANE):
        e = lane * PER_LANE + i
        c = Int32(0)
        if e < E:
            c = counts[e]
        loc_c[i] = c
    sum_c = Int32(0)
    sum_b = Int32(0)
    for i in cutlass.range_constexpr(PER_LANE):
        c = loc_c[i]
        loc_c[i] = sum_c
        loc_b[i] = sum_b
        if cutlass.const_expr(align > 1):
            sum_c = sum_c + ((c + Int32(align - 1)) // Int32(align)) * Int32(align)
        else:
            sum_c = sum_c + c
        sum_b = sum_b + (c + Int32(cluster_rows - 1)) // Int32(cluster_rows)
    # exclusive warp scan of the per-lane totals
    inc_c = sum_c
    inc_b = sum_b
    for sh in cutlass.range_constexpr(5):
        oc = cute.arch.shuffle_sync_up(inc_c, offset=1 << sh)
        ob = cute.arch.shuffle_sync_up(inc_b, offset=1 << sh)
        if lane >= (1 << sh):
            inc_c = inc_c + oc
            inc_b = inc_b + ob
    base_c = inc_c - sum_c
    base_b = inc_b - sum_b
    for i in cutlass.range_constexpr(PER_LANE):
        e = lane * PER_LANE + i
        s_off[e] = base_c + loc_c[i]
        s_pre[e] = base_b + loc_b[i]
    if lane == 31:
        s_off[MAX_E] = inc_c
        s_pre[MAX_E] = inc_b
    cute.arch.sync_warp()


class GroupScheduler:
    """Persistent walk over the cluster tiles of all groups. ``cluster_rows`` = CTA tile rows x cluster M (the rows one
    cluster tile covers), ``cn`` = cluster N, ``raster_m``: m-blocks fastest inside a group (consecutive clusters share
    the expert's B tile) or n-tiles fastest."""

    def __init__(
        self,
        s_off,
        s_pre,
        counts,
        E,
        n_tiles,
        total,
        stride,
        linear,
        cluster_rows,
        cn,
        cta_m_in_cluster,
        cta_n_in_cluster,
        raster_m,
    ):
        self.s_off, self.s_pre, self.counts = s_off, s_pre, counts
        self.E, self.n_tiles, self.total, self.stride, self.linear = (
            Int32(E),
            Int32(n_tiles),
            Int32(total),
            Int32(stride),
            Int32(linear),
        )
        self.cluster_rows, self.cn, self.raster_m = cluster_rows, cn, raster_m
        self.cta_m_in_cluster, self.cta_n_in_cluster = (
            cta_m_in_cluster,
            cta_n_in_cluster,
        )

    # -- loop-carried protocol: the smem tables and static knobs come from the enclosing scope
    def __extract_mlir_values__(self):
        vals = []
        for v in (self.E, self.n_tiles, self.total, self.stride, self.linear):
            vals.extend(extract_mlir_values(v))
        return vals

    def __new_from_mlir_values__(self, values):
        assert len(values) == 5, len(values)
        E, n_tiles, total, stride, linear = [
            new_from_mlir_values(a, [v])
            for a, v in zip(
                (self.E, self.n_tiles, self.total, self.stride, self.linear),
                values,
                strict=True,
            )
        ]
        return GroupScheduler(
            self.s_off,
            self.s_pre,
            self.counts,
            E,
            n_tiles,
            total,
            stride,
            linear,
            self.cluster_rows,
            self.cn,
            self.cta_m_in_cluster,
            self.cta_n_in_cluster,
            self.raster_m,
        )

    @staticmethod
    @cute.jit
    def create(
        s_off,
        s_pre,
        counts,
        E: Int32,
        N: Int32,
        cluster_rows: cutlass.Constexpr,
        tn: cutlass.Constexpr,
        cn: cutlass.Constexpr,
        raster_m: cutlass.Constexpr,
        cta_m_in_cluster,
        cta_n_in_cluster,
    ):
        """After ``build_group_tables`` and a CTA barrier. The linear index starts at the cluster's index in the grid and
        advances by the number of clusters. ``counts`` (global) gives the group's real end for the store mask (``s_off``
        may be padded)."""
        n_tiles = (N + Int32(tn * cn - 1)) // Int32(tn * cn)
        total = s_pre[MAX_E] * n_tiles
        cid = cute.arch.cluster_idx()
        cdim = cute.arch.cluster_dim()
        linear = cid[0] + cdim[0] * (cid[1] + cdim[1] * cid[2])
        stride = cdim[0] * cdim[1] * cdim[2]
        return GroupScheduler(
            s_off,
            s_pre,
            counts,
            E,
            n_tiles,
            total,
            stride,
            linear,
            cluster_rows,
            cn,
            cta_m_in_cluster,
            cta_n_in_cluster,
            raster_m,
        )

    @cute.jit
    def get_current_work(self) -> GroupWorkTile:
        valid = self.linear < self.total
        e = Int32(0)
        row0 = Int32(0)
        row_end = Int32(0)
        n_tile = Int32(0)
        # no cluster tile at all (every count zero: every slot dropped, or an expert-parallel rank without a local hit):
        # the tile is invalid before the decode, whose last group would have no row blocks to divide by
        if self.total > Int32(0):
            L = cutlass.min(self.linear, self.total - Int32(1))
            blk = L // self.n_tiles
            # largest e with s_pre[e] <= blk, e in [0, E)
            lo = Int32(0)
            hi = self.E - Int32(1)
            while lo < hi:
                mid = (lo + hi + Int32(1)) // Int32(2)
                if self.s_pre[mid] <= blk:
                    lo = mid
                else:
                    hi = mid - Int32(1)
            e = lo
            local = L - self.s_pre[e] * self.n_tiles
            mb = self.s_pre[e + Int32(1)] - self.s_pre[e]
            m_block = Int32(0)
            n_block = Int32(0)
            if cutlass.const_expr(self.raster_m):
                m_block = local % mb
                n_block = local // mb
            else:
                n_block = local % self.n_tiles
                m_block = local // self.n_tiles
            row0 = self.s_off[e] + m_block * Int32(self.cluster_rows)
            row_end = (
                self.s_off[e] + self.counts[e]
            )  # the real end: rows past it belong to the padding or to other groups
            n_tile = n_block * Int32(self.cn) + self.cta_n_in_cluster
        return GroupWorkTile(valid, e, row0, row_end, n_tile)

    @cute.jit
    def initial_work_tile_info(self) -> GroupWorkTile:
        return self.get_current_work()

    @cute.jit
    def advance_to_next_work(self, advance_count: int = 1):
        self.linear = self.linear + Int32(advance_count) * self.stride
