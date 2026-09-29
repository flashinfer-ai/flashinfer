# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils


class MLAStaticTileSchedulerParams:
    def __init__(
        self,
        is_persistent: bool,
        problem_shape_b: cute.Int32,
        problem_shape_s: cute.Int32,
        cluster_shape_mnk: cute.Shape,
        split_kv: cutlass.Int32,
        *,
        problem_shape_b_fdd: cute.FastDivmodDivisor = None,
        problem_shape_s_fdd: cute.FastDivmodDivisor = None,
        split_kv_fdd: cute.FastDivmodDivisor = None,
        loc=None,
        ip=None,
    ):
        """The static tile scheduler parameters prepared for MLA static tile scheduler.

        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param problem_shape_b: The shape of the problem
        :type problem_shape_b: cute.Int32
        :param problem_shape_s: The shape of the problem in sequence length Q dimension
        :type problem_shape_s: cute.Int32
        :param cluster_shape_mnk: The shape of the cluster
        :type cluster_shape_mnk: cute.Shape
        :param split_kv: The scalar factor for split KV
        """
        self.is_persistent = is_persistent
        self.problem_shape_b = problem_shape_b
        self.problem_shape_s = problem_shape_s
        self.problem_shape_b_fdd = problem_shape_b_fdd
        self.problem_shape_s_fdd = problem_shape_s_fdd
        self.cluster_shape_mnk = cluster_shape_mnk
        self.split_kv = split_kv
        self.split_kv_fdd = split_kv_fdd
        if cutlass.const_expr(problem_shape_b_fdd is None):
            self.problem_shape_b_fdd = cute.fast_divmod_create_divisor(
                problem_shape_b, loc=loc, ip=ip
            )
        if cutlass.const_expr(problem_shape_s_fdd is None):
            self.problem_shape_s_fdd = cute.fast_divmod_create_divisor(
                problem_shape_s, loc=loc, ip=ip
            )
        if cutlass.const_expr(split_kv_fdd is None):
            self.split_kv_fdd = cute.fast_divmod_create_divisor(
                split_kv, loc=loc, ip=ip
            )
        self.loc = loc
        self.ip = ip

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.problem_shape_b)
        values += cutlass.extract_mlir_values(self.problem_shape_s)
        values += cutlass.extract_mlir_values(self.split_kv)
        values += cutlass.extract_mlir_values(self.problem_shape_b_fdd)
        values += cutlass.extract_mlir_values(self.problem_shape_s_fdd)
        values += cutlass.extract_mlir_values(self.split_kv_fdd)
        return values

    def __new_from_mlir_values__(self, values):
        problem_shape_b = cutlass.new_from_mlir_values(
            self.problem_shape_b, (values[0],)
        )
        problem_shape_s = cutlass.new_from_mlir_values(
            self.problem_shape_s, (values[1],)
        )
        split_kv = cutlass.new_from_mlir_values(self.split_kv, (values[2],))
        problem_shape_b_fdd = cutlass.new_from_mlir_values(
            self.problem_shape_b_fdd, (values[3],)
        )
        problem_shape_s_fdd = cutlass.new_from_mlir_values(
            self.problem_shape_s_fdd, (values[4],)
        )
        split_kv_fdd = cutlass.new_from_mlir_values(self.split_kv_fdd, (values[5],))
        return MLAStaticTileSchedulerParams(
            self.is_persistent,
            problem_shape_b,
            problem_shape_s,
            self.cluster_shape_mnk,
            split_kv,
            problem_shape_b_fdd=problem_shape_b_fdd,
            problem_shape_s_fdd=problem_shape_s_fdd,
            split_kv_fdd=split_kv_fdd,
            loc=self.loc,
        )


def create_mla_static_tile_scheduler_params(
    is_persistent: bool,
    problem_shape_b: cute.Int32,
    problem_shape_s: cute.Int32,
    cluster_shape_mnk: cute.Shape,
    split_kv: cutlass.Int32,
) -> MLAStaticTileSchedulerParams:
    return MLAStaticTileSchedulerParams(
        is_persistent, problem_shape_b, problem_shape_s, cluster_shape_mnk, split_kv
    )


class WorkTileInfo:
    def __init__(self, blk_coord: cute.Coord, is_valid: bool):
        self.blk_coord = blk_coord
        self.is_valid = cutlass.Boolean(is_valid)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.blk_coord)
        values += cutlass.extract_mlir_values(self.is_valid)
        return values

    def __new_from_mlir_values__(self, values):
        new_tile_idx = cutlass.new_from_mlir_values(self.blk_coord, values[:-1])
        new_is_valid_tile = cutlass.new_from_mlir_values(self.is_valid, [values[-1]])
        return WorkTileInfo(new_tile_idx, new_is_valid_tile)

    @property
    def is_valid_tile(self) -> cutlass.Boolean:
        return self.is_valid

    @property
    def tile_idx(self) -> cute.Coord:
        return self.blk_coord


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
                    params.problem_shape_s,
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
            params.problem_shape_b * params.problem_shape_s,
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

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        is_valid = (
            self.current_work_linear_idx < self.num_blocks
            if self.params.is_persistent
            else self.is_valid
        )

        if self.params.is_persistent:
            current_work_cluster_batch, cluster_idx = (
                self.current_work_linear_idx // self.params.cluster_shape_mnk[0],
                self.current_work_linear_idx % self.params.cluster_shape_mnk[0],
            )
            current_work_s_batch, s_idx = divmod(
                current_work_cluster_batch, self.params.problem_shape_s_fdd
            )
            current_work_b_batch, b_idx = divmod(
                current_work_s_batch, self.params.problem_shape_b_fdd
            )
            _, split_kv_idx = divmod(current_work_b_batch, self.params.split_kv_fdd)

            blk_coord = (cluster_idx, s_idx, b_idx, split_kv_idx)
        else:
            b_idx, s_idx = divmod(self.blk_coord[1], self.params.problem_shape_s_fdd)
            blk_coord = (self.blk_coord[0], s_idx, b_idx, self.blk_coord[2])

        return WorkTileInfo(blk_coord, is_valid)

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
        assert len(values) == 13
        new_params = cutlass.new_from_mlir_values(self.params, values[0:6])
        new_current_work_linear_idx = cutlass.new_from_mlir_values(
            self.current_work_linear_idx, [values[6]]
        )
        new_blk_coord = cutlass.new_from_mlir_values(self.blk_coord, values[7:10])
        new_grid_shape = cutlass.new_from_mlir_values(self.grid_shape, values[10:])
        return MLAStaticTileScheduler(
            new_params, new_current_work_linear_idx, new_blk_coord, new_grid_shape
        )


def create_mla_static_tile_scheduler(
    params: MLAStaticTileSchedulerParams,
    blk_coord: cute.Coord,
    grid_shape: cute.Shape,
) -> MLAStaticTileScheduler:
    return MLAStaticTileScheduler(params, blk_coord[0], blk_coord, grid_shape)


LOG2_E = 1.4426950408889634074
# avoid register indexing on array.
MAX_SPLITS = 256


@cute.jit
def get_variable_query_tile_info(
    num_heads: cutlass.Constexpr,
    q_tile_idx: cutlass.Int32,
    batch_idx: cutlass.Int32,
    cum_seq_lens_q: cute.Tensor,
    m_tile: cutlass.Constexpr,
) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
    """Return request-local metadata for one compact token/head tile.

    Compact Q uses ``row = token * num_heads + head``. Returns
    ``(q_begin, q_len, valid_rows)``; ``valid_rows == 0`` identifies a
    rectangular schedule slot beyond the request. Such a slot must remain
    scheduler-valid and become a no-op rather than terminate a persistent
    scheduler iteration.
    """
    # Every caller supplies a CTA-uniform request index. Marking the two indptr
    # loads warp-uniform avoids issuing the same scalar load in every lane,
    # especially in the many small standalone-reducer CTAs.
    q_begin = cute.arch.make_warp_uniform(cum_seq_lens_q[batch_idx])
    q_end = cute.arch.make_warp_uniform(cum_seq_lens_q[batch_idx + 1])
    q_len = q_end - q_begin

    remaining_rows = q_len * num_heads - q_tile_idx * m_tile
    valid_rows = cutlass.max(
        cutlass.Int32(0),
        cutlass.min(cutlass.Int32(m_tile), remaining_rows),
    )
    return q_begin, q_len, valid_rows


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compute_q_tile_layout(
    num_heads: int, seq_len_q: int, m_tile: int = 128
) -> tuple[int, int, int]:
    """Return ``(total_rows, num_tiles, tail_rows)`` for flat query packing.

    Query-token and head modes are one affine row space ordered as
    ``flat_row = q_token * num_heads + q_head``.  Consecutive M tiles may
    therefore cross token boundaries; only the final tile can be partial.
    ``tail_rows`` is always in ``[1, m_tile]`` and equals ``m_tile`` when the
    flattened row count exactly fills the final tile.

    This host-side helper is shared by launch/workspace selection and both
    kernel variants so their split-KV geometry cannot drift apart.
    """
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if seq_len_q <= 0:
        raise ValueError(f"seq_len_q must be positive, got {seq_len_q}")
    if m_tile <= 0:
        raise ValueError(f"m_tile must be positive, got {m_tile}")
    if num_heads > m_tile:
        raise ValueError(
            f"num_heads ({num_heads}) must not exceed the MMA M tile ({m_tile})"
        )

    total_rows = num_heads * seq_len_q
    num_tiles = ceil_div(total_rows, m_tile)
    tail_rows = total_rows - (num_tiles - 1) * m_tile
    return total_rows, num_tiles, tail_rows


class MLAReducerMixin:
    """Split-KV reduction kernel shared by the FP8 and FP16/BF16 MLA decode kernels.

    Requires the kernel attributes set before ``_init_reducer``: ``latent_dim``,
    ``reducer_d_tiles``, ``reducer_d_tile``, ``threads_per_warp`` and
    ``num_compute_warps``.
    """

    def _init_reducer(self, reducer_max_splits: int) -> None:
        # Each reducer thread owns ``reducer_vec`` adjacent columns of its band
        # and keeps a batch of splits' partials in registers (<= 128 floats).
        self.reducer_vec = self.reducer_d_tile // (
            self.threads_per_warp * self.num_compute_warps
        )
        self.reducer_accumulate_group = 4
        reducer_padded_splits = (
            ceil_div(reducer_max_splits, self.reducer_accumulate_group)
            * self.reducer_accumulate_group
        )
        self.reducer_batch_splits = min(
            reducer_padded_splits, 64, 128 // self.reducer_vec
        )
        self.reducer_num_sets = ceil_div(
            reducer_padded_splits, self.reducer_batch_splits
        )
        self.reducer_scale_slots = (
            ceil_div(reducer_max_splits, self.threads_per_warp) * self.threads_per_warp
        )

    @cute.kernel
    def reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        cum_seq_lens_q: Optional[cute.Tensor],
        block_split_kvs: cute.Tensor,
        lse_scale: cutlass.Float32,
    ):
        """The reduction kernel for Multi-Head Latent Attention (MLA) that combines intermediate results
        from multiple split_kv blocks into final outputs.

        :param mO: Output tensor for storing final results
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor for storing final LSE values
        :type mLSE: cute.Tensor
        :param mAccO: Accumulated output tensor from split_kv blocks
        :type mAccO: cute.Tensor
        :param mAccLSE: Accumulated LSE tensor from split_kv blocks
        :type mAccLSE: cute.Tensor
        :param split_kv: Number of split_kv blocks
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor (for variable split_kv)
        :type block_split_kvs: cute.Tensor
        :param lse_scale: Multiplier applied to the stored LSE. This kernel
            accumulates LSE in base 2, so ``1 / log2(e)`` yields natural-log
            values and ``1.0`` keeps base 2. Plumbed from
            ``return_lse_base`` on the public MLA decode API.
        :type lse_scale: cutlass.Float32
        """
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        d_tile_idx = bidx % self.reducer_d_tiles
        head_idx = bidx // self.reducer_d_tiles
        q_begin = cutlass.Int32(0)
        q_len = cutlass.Int32(self.seq_len_q)
        is_active = True
        if cutlass.const_expr(self.is_var_q):
            q_begin, q_len, _ = get_variable_query_tile_info(
                self.num_heads,
                cutlass.Int32(0),
                bidz,
                cum_seq_lens_q,
                self.mma_qk_tiler[0],
            )
            is_active = bidy < q_len

        # Inactive compact rows still complete the PDL protocol, but do not
        # touch workspace or enter the active-row shared-memory barrier.
        smem = utils.SmemAllocator()
        storage = smem.allocate(
            self.reducer_scale_slots * self.acc_dtype.width // 8, 16
        )
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.acc_dtype)
        smem_lse_scale = cute.make_tensor(
            lse_scale_ptr, cute.make_layout(self.reducer_scale_slots)
        )

        if cutlass.const_expr(self.enable_pdl):
            # The next kernel's own griddepcontrol.wait still orders its reads.
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
        if is_active:
            flat_q_row = bidy * self.num_heads + head_idx
            q_tile = flat_q_row >> 7
            q_tile_row = flat_q_row & 127
            local_split_kv = split_kv
            if cutlass.const_expr(self.is_var_split_kv):
                local_split_kv = block_split_kvs[bidz]
            k_tile_total = cute.ceil_div(cache_seqs[bidz], self.mma_qk_tiler[1])
            if cutlass.const_expr(self.enable_dcp):
                k_tile_per_cta = cutlass.max(
                    cute.ceil_div(k_tile_total, local_split_kv), cutlass.Int32(1)
                )
                local_split_kv = cutlass.max(
                    cute.ceil_div(k_tile_total, k_tile_per_cta), cutlass.Int32(1)
                )
            else:
                k_tile_per_cta = cute.ceil_div(k_tile_total, local_split_kv)
                local_split_kv = cute.ceil_div(k_tile_total, k_tile_per_cta)

            # Issue the first set of partial loads before the LSE phase so
            # their latency overlaps it; later sets only if the row needs them.
            gAccO = mAccO[q_tile_row, None, None, q_tile, bidz]
            band_ptr = cute.make_ptr(
                self.acc_dtype,
                (gAccO.iterator + d_tile_idx * self.reducer_d_tile).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            partial_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.acc_dtype,
                num_bits_per_copy=self.reducer_vec * self.acc_dtype.width,
            )
            partial_thr_copy = cute.make_tiled_copy_tv(
                partial_copy_atom,
                cute.make_ordered_layout(
                    (1, self.threads_per_warp * self.num_compute_warps), order=(1, 0)
                ),
                cute.make_layout((1, self.reducer_vec)),
            ).get_slice(tidx)
            partial_set = self._reducer_issue_set(
                band_ptr,
                gAccO.stride[0],
                partial_copy_atom,
                partial_thr_copy,
                local_split_kv,
                0,
            )

            gLSE = mAccLSE[q_tile_row, None, q_tile, bidz]
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 0:
                lse_per_thread = self.reducer_scale_slots // self.threads_per_warp
                local_lse = cute.make_rmem_tensor(
                    cute.make_layout(lse_per_thread), self.lse_dtype
                )
                lse_max = -self.lse_dtype.inf
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_kv_idx = tidx + i * self.threads_per_warp
                    local_lse[i] = (
                        gLSE[split_kv_idx]
                        if cute.elem_less(split_kv_idx, local_split_kv)
                        else -self.lse_dtype.inf
                    )
                    lse_max = cute.arch.fmax(lse_max, local_lse[i])
                lse_max = cute.arch.warp_reduction_max(lse_max)
                if cutlass.const_expr(self.enable_dcp):
                    has_valid_lse = lse_max != -self.lse_dtype.inf
                    lse_max = lse_max if has_valid_lse else 0.0
                else:
                    lse_max = lse_max if lse_max != -self.lse_dtype.inf else 0.0
                sum_lse = 0.0
                for i in cutlass.range_constexpr(lse_per_thread):
                    sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
                sum_lse = cute.arch.warp_reduction_sum(sum_lse)
                if cutlass.const_expr(self.enable_dcp):
                    global_lse = (
                        lse_max + cute.math.log2(sum_lse, fastmath=True)
                        if has_valid_lse
                        else -self.lse_dtype.inf
                    )
                else:
                    global_lse = (
                        lse_max + cute.math.log2(sum_lse, fastmath=True)
                        if sum_lse != self.lse_dtype(0.0) or sum_lse != sum_lse
                        else self.lse_dtype.inf
                    )
                if d_tile_idx == 0:
                    if tidx == 0:
                        if cutlass.const_expr(self.is_var_q):
                            mLSE[head_idx, q_begin + bidy] = global_lse * lse_scale
                        else:
                            mLSE[head_idx, bidy, bidz] = global_lse * lse_scale
                # Every slot is written (zero past local_split_kv) so whole
                # accumulate groups can be read without a per-split bound.
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_kv_idx = tidx + i * self.threads_per_warp
                    if cutlass.const_expr(self.enable_dcp):
                        smem_lse_scale[split_kv_idx] = (
                            cute.math.exp2(local_lse[i] - global_lse, fastmath=True)
                            if has_valid_lse
                            else self.acc_dtype(0.0)
                        )
                    else:
                        smem_lse_scale[split_kv_idx] = cute.math.exp2(
                            local_lse[i] - global_lse, fastmath=True
                        )

            pipeline.sync(barrier_id=4)

            rAccO = cute.make_rmem_tensor(
                cute.make_layout(self.reducer_vec), self.acc_dtype
            )
            rAccO.fill(0.0)
            self._reducer_accumulate_set(
                partial_set, smem_lse_scale, local_split_kv, 0, rAccO
            )
            for s in cutlass.range_constexpr(1, self.reducer_num_sets):
                first = s * self.reducer_batch_splits
                if local_split_kv > first:
                    partial_set = self._reducer_issue_set(
                        band_ptr,
                        gAccO.stride[0],
                        partial_copy_atom,
                        partial_thr_copy,
                        local_split_kv,
                        first,
                    )
                    self._reducer_accumulate_set(
                        partial_set, smem_lse_scale, local_split_kv, first, rAccO
                    )
            rO = cute.make_rmem_tensor(cute.make_layout(self.reducer_vec), self.o_dtype)
            rO.store(rAccO.load().to(self.o_dtype))
            for j in cutlass.range_constexpr(self.reducer_vec):
                element_idx = (
                    d_tile_idx * self.reducer_d_tile + tidx * self.reducer_vec + j
                )
                if cutlass.const_expr(self.is_var_q):
                    mO[head_idx, element_idx, q_begin + bidy] = rO[j]
                else:
                    mO[head_idx, element_idx, bidy, bidz] = rO[j]
        return

    @cute.jit
    def _reducer_issue_set(
        self,
        band_ptr: cute.Pointer,
        split_stride: cutlass.Int32,
        copy_atom: cute.CopyAtom,
        thr_copy: cute.TiledCopy,
        local_split_kv: cutlass.Int32,
        first: int,
    ) -> cute.Tensor:
        """Predicated loads of this thread's partial columns for splits [first, first + batch)."""
        batch = self.reducer_batch_splits
        gSet = cute.make_tensor(
            band_ptr + first * split_stride,
            cute.make_layout((batch, self.reducer_d_tile), stride=(split_stride, 1)),
        )
        tPgP = thr_copy.partition_S(gSet)
        tPrP = cute.make_fragment_like(tPgP)
        tPpP = cute.make_rmem_tensor(cute.make_layout((1, batch, 1)), cutlass.Boolean)
        for k in cutlass.range_constexpr(batch):
            tPpP[0, k, 0] = cute.elem_less(first + k, local_split_kv)
        tPrP.fill(0.0)
        cute.copy(copy_atom, tPgP, tPrP, pred=tPpP)
        return tPrP

    @cute.jit
    def _reducer_accumulate_set(
        self,
        tPrP: cute.Tensor,
        smem_lse_scale: cute.Tensor,
        local_split_kv: cutlass.Int32,
        first: int,
        rAccO: cute.Tensor,
    ):
        """Accumulate splits [first, first + batch) in split order, skipping empty groups."""
        group = self.reducer_accumulate_group
        for g in cutlass.range_constexpr(self.reducer_batch_splits // group):
            if local_split_kv > first + g * group:
                for k in cutlass.range_constexpr(group):
                    split = g * group + k
                    weight = smem_lse_scale[first + split]
                    for v in cutlass.range_constexpr(self.reducer_vec):
                        rAccO[v] += tPrP[v, split, 0] * weight
