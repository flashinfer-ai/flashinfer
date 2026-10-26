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


from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils

from .mla_helpers import (
    MAX_SPLITS,
    ceil_div,
    compute_q_tile_layout,
    get_variable_query_tile_info,
)


class MLAReducer:
    """Split-KV reducer shared by FP8 and FP16/BF16 MLA decode.

    Configuration describes the producer's workspace tiles and logical query
    rows. Output dtype is taken from the output tensor at compilation time.
    Runtime split counts must not exceed ``max_splits``. Partial output uses
    modes (M, split, D512, query_tile, batch), with contiguous D; partial LSE
    uses (M, split, query_tile, batch) and stores base-2 logarithms.
    """

    def __init__(
        self,
        *,
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        qk_tile_shape: Tuple[int, int],
        num_heads: int,
        seq_len_q: int,
        max_splits: int = MAX_SPLITS,
        d_tiles: int = 1,
        is_var_q: bool = False,
        is_var_split_kv: bool = False,
        enable_dcp: bool = False,
        enable_pdl: bool = False,
    ):
        if not 1 <= max_splits <= MAX_SPLITS:
            raise ValueError(
                f"max_splits must be in [1, {MAX_SPLITS}], got {max_splits}"
            )
        if is_var_split_kv and max_splits != MAX_SPLITS:
            raise ValueError("variable split-KV requires the generic reducer capacity")
        if d_tiles not in (1, 2, 4):
            raise ValueError(f"unsupported d_tiles={d_tiles}")
        compute_q_tile_layout(num_heads, seq_len_q, qk_tile_shape[0])
        if qk_tile_shape[1] <= 0:
            raise ValueError("QK tile N must be positive")
        self.acc_dtype = acc_dtype
        self.lse_dtype = lse_dtype
        self.qk_tile_shape = qk_tile_shape
        self.num_heads = num_heads
        self.seq_len_q = seq_len_q
        self.max_splits = max_splits
        self.d_tiles = d_tiles
        self.d_tile = 512 // d_tiles
        self.threads_per_warp = 32
        self.num_threads = 128
        self.is_var_q = is_var_q
        self.is_var_split_kv = is_var_split_kv
        self.enable_dcp = enable_dcp
        self.enable_pdl = enable_pdl
        # Each reducer thread owns ``vec`` adjacent columns of its band
        # and keeps a batch of splits' partials in registers (<= 128 floats).
        self.vec = self.d_tile // self.num_threads
        self.accumulate_group = 4
        reducer_padded_splits = (
            ceil_div(max_splits, self.accumulate_group) * self.accumulate_group
        )
        self.batch_splits = min(reducer_padded_splits, 64, 128 // self.vec)
        self.num_sets = ceil_div(reducer_padded_splits, self.batch_splits)
        self.scale_slots = (
            ceil_div(max_splits, self.threads_per_warp) * self.threads_per_warp
        )

    @cute.jit
    def __call__(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        cum_seq_lens_q: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        lse_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.reduction_kernel(
            mO,
            mLSE,
            mAccO,
            mAccLSE,
            split_kv,
            cache_seqs,
            cum_seq_lens_q,
            block_split_kvs,
            lse_scale,
        ).launch(
            grid=(self.num_heads * self.d_tiles, self.seq_len_q, cache_seqs.shape[0]),
            block=[self.num_threads, 1, 1],
            smem=self.scale_slots * self.acc_dtype.width // 8,
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
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
        block_split_kvs: Optional[cute.Tensor],
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
        :type block_split_kvs: Optional[cute.Tensor]
        :param lse_scale: Multiplier applied to the stored LSE. This kernel
            accumulates LSE in base 2, so ``1 / log2(e)`` yields natural-log
            values and ``1.0`` keeps base 2. Plumbed from
            ``return_lse_base`` on the public MLA decode API.
        :type lse_scale: cutlass.Float32
        """
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        d_tile_idx = bidx % self.d_tiles
        head_idx = bidx // self.d_tiles
        q_begin = cutlass.Int32(0)
        q_len = cutlass.Int32(self.seq_len_q)
        is_active = True
        if cutlass.const_expr(self.is_var_q):
            q_begin, q_len, _ = get_variable_query_tile_info(
                self.num_heads,
                cutlass.Int32(0),
                bidz,
                cum_seq_lens_q,
                self.qk_tile_shape[0],
            )
            is_active = bidy < q_len

        # Inactive compact rows still complete the PDL protocol, but do not
        # touch workspace or enter the active-row shared-memory barrier.
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.scale_slots * self.acc_dtype.width // 8, 16)
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.acc_dtype)
        smem_lse_scale = cute.make_tensor(
            lse_scale_ptr, cute.make_layout(self.scale_slots)
        )

        if cutlass.const_expr(self.enable_pdl):
            # The next kernel's own griddepcontrol.wait still orders its reads.
            cute.arch.griddepcontrol_launch_dependents()
            cute.arch.griddepcontrol_wait()
        if is_active:
            flat_q_row = bidy * self.num_heads + head_idx
            q_tile = flat_q_row // self.qk_tile_shape[0]
            q_tile_row = flat_q_row % self.qk_tile_shape[0]
            local_split_kv = split_kv
            if cutlass.const_expr(self.is_var_split_kv):
                local_split_kv = block_split_kvs[bidz]
            k_tile_total = cute.ceil_div(cache_seqs[bidz], self.qk_tile_shape[1])
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
                (gAccO.iterator + d_tile_idx * self.d_tile).toint(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            partial_copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.acc_dtype,
                num_bits_per_copy=self.vec * self.acc_dtype.width,
            )
            partial_thr_copy = cute.make_tiled_copy_tv(
                partial_copy_atom,
                cute.make_ordered_layout((1, self.num_threads), order=(1, 0)),
                cute.make_layout((1, self.vec)),
            ).get_slice(tidx)
            partial_set = self._issue_set(
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
                lse_per_thread = self.scale_slots // self.threads_per_warp
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

            rAccO = cute.make_rmem_tensor(cute.make_layout(self.vec), self.acc_dtype)
            rAccO.fill(0.0)
            self._accumulate_set(partial_set, smem_lse_scale, local_split_kv, 0, rAccO)
            for s in cutlass.range_constexpr(1, self.num_sets):
                first = s * self.batch_splits
                if local_split_kv > first:
                    partial_set = self._issue_set(
                        band_ptr,
                        gAccO.stride[0],
                        partial_copy_atom,
                        partial_thr_copy,
                        local_split_kv,
                        first,
                    )
                    self._accumulate_set(
                        partial_set, smem_lse_scale, local_split_kv, first, rAccO
                    )
            rO = cute.make_rmem_tensor(cute.make_layout(self.vec), mO.element_type)
            rO.store(rAccO.load().to(mO.element_type))
            for j in cutlass.range_constexpr(self.vec):
                element_idx = d_tile_idx * self.d_tile + tidx * self.vec + j
                if cutlass.const_expr(self.is_var_q):
                    mO[head_idx, element_idx, q_begin + bidy] = rO[j]
                else:
                    mO[head_idx, element_idx, bidy, bidz] = rO[j]
        return

    @cute.jit
    def _issue_set(
        self,
        band_ptr: cute.Pointer,
        split_stride: cutlass.Int32,
        copy_atom: cute.CopyAtom,
        thr_copy: cute.TiledCopy,
        local_split_kv: cutlass.Int32,
        first: int,
    ) -> cute.Tensor:
        """Predicated loads of this thread's partial columns for splits [first, first + batch)."""
        batch = self.batch_splits
        gSet = cute.make_tensor(
            band_ptr + first * split_stride,
            cute.make_layout((batch, self.d_tile), stride=(split_stride, 1)),
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
    def _accumulate_set(
        self,
        tPrP: cute.Tensor,
        smem_lse_scale: cute.Tensor,
        local_split_kv: cutlass.Int32,
        first: int,
        rAccO: cute.Tensor,
    ):
        """Accumulate splits [first, first + batch) in split order, skipping empty groups."""
        group = self.accumulate_group
        for g in cutlass.range_constexpr(self.batch_splits // group):
            if local_split_kv > first + g * group:
                for k in cutlass.range_constexpr(group):
                    split = g * group + k
                    weight = smem_lse_scale[first + split]
                    for v in cutlass.range_constexpr(self.vec):
                        rAccO[v] += tPrP[v, split, 0] * weight
