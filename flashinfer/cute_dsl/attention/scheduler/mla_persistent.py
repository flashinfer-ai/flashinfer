# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Persistent tile scheduler for MLA decode kernels.

Manages work distribution across CTAs for split-KV MLA attention,
supporting both persistent and non-persistent kernel modes.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils


class MLAStaticTileSchedulerParams:
    def __init__(
        self,
        is_persistent: bool,
        problem_shape_b: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        split_kv: cutlass.Int32,
        *,
        loc=None,
        ip=None,
    ):
        """The static tile scheduler parameters prepared for MLA static tile scheduler.

        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param problem_shape_b: The shape of the problem
        :type problem_shape_b: cute.Int32
        :param cluster_shape_mnk: The shape of the cluster
        :type cluster_shape_mnk: cute.Shape
        :param split_kv: The scalar factor for split KV
        """
        self.is_persistent = is_persistent
        self.problem_shape_b = problem_shape_b
        self.cluster_shape_mnk = cluster_shape_mnk
        self.split_kv = split_kv
        self.loc = loc
        self.ip = ip

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_b)
        values += cutlass.extract_mlir_values(self.split_kv)
        return values

    def __new_from_mlir_values__(self, values):
        problem_shape_b = cutlass.new_from_mlir_values(
            self.problem_shape_b, (values[0],)
        )
        split_kv = cutlass.new_from_mlir_values(self.split_kv, (values[1],))
        return MLAStaticTileSchedulerParams(
            self.is_persistent,
            problem_shape_b,
            self.cluster_shape_mnk,
            split_kv,
            loc=self.loc,
        )


def create_mla_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_b: cute.Int32,
    cluster_shape_mnk: cute.Shape,
    split_kv: cutlass.Int32,
) -> MLAStaticTileSchedulerParams:
    return MLAStaticTileSchedulerParams(
        is_persistent, problem_shape_b, cluster_shape_mnk, split_kv
    )


class MLAStaticTileScheduler:
    def __init__(
        self,
        params: MLAStaticTileSchedulerParams,
        current_work_linear_idx: cutlass.Int32,
        blk_coord: cute.Coord,
        grid_shape: cute.Shape,
        *,
        is_valid: bool = True,
        loc=None,
        ip=None,
    ):
        """The static tile scheduler for MLA split kv kernel.
        Based on `is_persistent`, it provides 2 modes for use:
        - Persistent mode: Launch fixed blocks and reschedule the data blocks.
        - Non-persistent mode: Launch dynamic blocks and exit when the current work is done.

        :param params: The static tile scheduler parameters
        :type params: MLAStaticTileSchedulerParams
        :param current_work_linear_idx: The linear index of the current work
        :type current_work_linear_idx: cutlass.Int32
        :param blk_coord: The coordinate of the current work
        :type blk_coord: cute.Coord
        :param grid_shape: The shape of the grid
        :type grid_shape: cute.Shape
        :param is_valid: Whether the current work is valid
        :type is_valid: bool
        """
        self.params = params
        self.blk_coord = blk_coord
        self.grid_shape = grid_shape
        self.current_work_linear_idx = current_work_linear_idx
        if params.is_persistent:
            self.persistent_blk_layout = cute.make_layout(
                (
                    params.cluster_shape_mnk[0],
                    1,
                    params.problem_shape_b,
                    params.split_kv,
                ),
                loc=loc,
                ip=ip,
            )
            self.num_blocks = cute.size(self.persistent_blk_layout, loc=loc, ip=ip)
            # Used for persistent scheduling
            self.num_persistent_sm = cute.size(grid_shape, loc=loc, ip=ip)
        else:
            self.is_valid = is_valid
        self.loc = loc
        self.ip = ip

    @staticmethod
    def get_grid_shape(
        params: MLAStaticTileSchedulerParams,
        max_active_clusters: int,
        *,
        loc=None,
        ip=None,
    ) -> cute.Shape:
        # called by host
        grid_shape = (
            params.cluster_shape_mnk[0],
            params.problem_shape_b,
            params.split_kv,
        )
        if params.is_persistent:
            return (
                cutlass.min(
                    max_active_clusters * cute.size(params.cluster_shape_mnk),
                    cute.size(grid_shape, loc=loc, ip=ip),
                ),
                1,
                1,
            )
        else:
            return grid_shape

    def get_current_work(self, *, loc=None, ip=None) -> utils.WorkTileInfo:
        is_valid = (
            self.current_work_linear_idx < self.num_blocks
            if self.params.is_persistent
            else self.is_valid
        )

        if self.params.is_persistent:
            blk_coord = self.persistent_blk_layout.get_hier_coord(
                self.current_work_linear_idx, loc=loc, ip=ip
            )
        else:
            blk_coord = (self.blk_coord[0], 0, self.blk_coord[1], self.blk_coord[2])

        return utils.WorkTileInfo(blk_coord, is_valid)

    def initial_work_tile_info(self, *, loc=None, ip=None):
        return self.get_current_work(loc=loc, ip=ip)

    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None):
        if self.params.is_persistent:
            self.current_work_linear_idx += advance_count * self.num_persistent_sm
        else:
            self.is_valid = False

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.params)
        values.extend(cutlass.extract_mlir_values(self.current_work_linear_idx))
        values.extend(cutlass.extract_mlir_values(self.blk_coord))
        values.extend(cutlass.extract_mlir_values(self.grid_shape))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 9
        new_params = cutlass.new_from_mlir_values(self.params, values[0:2])
        new_current_work_linear_idx = cutlass.new_from_mlir_values(
            self.current_work_linear_idx, [values[2]]
        )
        new_blk_coord = cutlass.new_from_mlir_values(self.blk_coord, values[3:6])
        new_grid_shape = cutlass.new_from_mlir_values(self.grid_shape, values[6:])
        return MLAStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


def create_mla_static_tile_scheduler(
    params: MLAStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> MLAStaticTileScheduler:
    return MLAStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


def mla_get_k_tile_count(
    mma_qk_tiler_n,
    is_var_split_kv,
    split_kv,
    cache_seqs,
    block_split_kvs,
    blk_coord,
):
    """Device-side tile range computation for MLA split-KV.

    Returns (k_index, k_tile_count, local_split_kv) for the current block coordinate.
    """
    K = cache_seqs[blk_coord[2]]
    if cutlass.const_expr(is_var_split_kv):
        split_kv = block_split_kvs[blk_coord[2]]

    k_tile_total = cute.ceil_div(K, mma_qk_tiler_n)
    k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
    k_index = blk_coord[3] * k_tile_per_cta
    k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
    return k_index, k_tile_count, split_kv


def mla_get_split_kv(
    B: int, K: int, mma_qk_tiler_mn: tuple, max_active_blocks: int
) -> int:
    """Compute the split-KV factor for MLA decode based on problem size and hardware occupancy.

    :param B: Batch size
    :param K: Sequence length
    :param mma_qk_tiler_mn: MMA QK tile shape (M, N)
    :param max_active_blocks: Maximum number of active blocks on the GPU
    :return: Optimal split-KV factor
    """
    def _ceil_div(a, b):
        return (a + b - 1) // b

    max_splits = _ceil_div(K, mma_qk_tiler_mn[1])
    blocks_per_batch = max(1, max_active_blocks // B)
    split_heur = min(max_splits, blocks_per_batch)
    k_waves = _ceil_div(max_splits, split_heur)
    split_wave_aware = _ceil_div(max_splits, k_waves)
    return split_wave_aware
