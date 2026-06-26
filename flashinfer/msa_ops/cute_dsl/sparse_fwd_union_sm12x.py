"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

Union-tile Minimax Sparse Attention forward kernel for SM120/SM121.

The query-tile counterpart of the KV-block-major ``sparse_fwd_sm12x``. Work is
distributed over (batch, query-tile, kv-head): each CTA owns ``tokens_per_tile``
consecutive query tokens (x ``group_size`` GQA heads = the ``m_block`` MMA rows)
and loops over the **union** of the KV blocks those tokens selected, running an
**in-kernel online softmax** over that union and writing the **final normalized
output directly** -- no per-(query, block) GMEM partials and no separate combine
kernel (contrast ``sparse_fwd_sm12x`` + ``sparse_combine_sm12x``).

A query token attends only the blocks it actually selected: a per-(union-block)
``tokens_per_tile``-bit membership mask (built by the union-metadata pass) gates
each row, so a token's scores for a block it didn't pick are forced to -inf.
This is the b12x ``msa_union_tile`` design ported to the SM12x warp-MMA + cp.async
mainloop; on sm12x (memory-bound, no partials round-trip) it removes the ~19x
partial write-amplification of the KV-major path.

Compute core: warp-level mma.sync (m16n8k16) + cp.async. bf16/fp16 flat K/V.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp


class SparseAttentionUnionFwdSm12x:
    """Union-tile (query-major) sparse attention forward for SM12x.

    Static config: head_dim (128), n_block (128 = blk_kv), m tile (64 rows =
    tokens_per_tile x group_size), GQA group size, causal flag. bf16/fp16, flat
    K/V. Writes the final output + optional log2-domain LSE directly.
    """

    def __init__(
        self,
        head_dim: int = 128,
        m_block_size: int = 64,
        n_block_size: int = 128,
        group_size: int = 1,
        num_threads: int = 128,
        is_causal: bool = True,
        return_softmax_lse: bool = False,
    ):
        if head_dim != 128 or n_block_size != 128:
            raise ValueError("only head_dim == n_block_size == 128 is supported")
        if m_block_size % group_size != 0:
            raise ValueError("group_size must divide m_block_size")
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._group_size = group_size
        self._tokens_per_tile = m_block_size // group_size
        self._num_threads = num_threads
        self._is_causal = is_causal
        self._return_softmax_lse = return_softmax_lse
        self._q_smem_stride = head_dim + 8
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    def _make_skv_layout(self):
        """Bank-swizzled K/V SMEM layout (same as the KV-major bf16 path)."""
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        return cute.tile_to_shape(atom, (self._n_block_size, self._head_dim), (0, 1))

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, Hq, d)
        mK: cute.Tensor,  # (total_k, Hkv, d) flat
        mV: cute.Tensor,  # (total_k, Hkv, d) flat
        mO: cute.Tensor,  # (total_q, Hq, d) final output
        mLse: cute.Tensor,  # (Hq, total_q) f32 LSE (dummy (1,1) if off)
        mUnionBlocks: cute.Tensor,  # (capacity, max_union) int32 kv-block ids
        mUnionMasks: cute.Tensor,  # (capacity, max_union) int32 tpt-bit masks
        mUnionCount: cute.Tensor,  # (capacity,) int32 blocks in the union
        mWorkMeta: cute.Tensor,  # (capacity, 3) int32 {batch, q_tile, kv_head}
        mWorkCount: cute.Tensor,  # (1,) int32
        mCuQ: cute.Tensor,  # (B + 1,) int32
        mCuK: cute.Tensor,  # (B + 1,) int32
        mQOffset: cute.Tensor,  # (B,) int32 causal offset (MSA q_offset)
        softmax_scale: cutlass.Float32,
        work_capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (mQ.element_type == mK.element_type == mV.element_type)
        ):
            raise TypeError("Q/K/V must have the same data type")
        if cutlass.const_expr(
            not (
                mQ.element_type == cutlass.Float16
                or mQ.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype = mQ.element_type

        skv_cosize = cute.cosize(self._make_skv_layout())
        sQ_layout = cute.make_layout(
            (self._m_block_size, self._head_dim),
            stride=(self._q_smem_stride, 1),
        )

        @cute.struct
        class SharedStorage:
            sK: cute.struct.Align[cute.struct.MemRange[self._dtype, skv_cosize], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[self._dtype, skv_cosize], 1024]
            sQ: cute.struct.Align[
                cute.struct.MemRange[
                    self._dtype, self._m_block_size * self._q_smem_stride
                ],
                1024,
            ]

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tKV_shape_dim_1 = 64 // async_copy_elems
        tKV_layout = cute.make_layout(
            (self._num_threads // tKV_shape_dim_1, tKV_shape_dim_1),
            stride=(tKV_shape_dim_1, 1),
        )
        vKV_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_KV = cute.make_tiled_copy_tv(
            atom_async_copy, tKV_layout, vKV_layout
        )

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLse,
            mUnionBlocks,
            mUnionMasks,
            mUnionCount,
            mWorkMeta,
            mWorkCount,
            mCuQ,
            mCuK,
            mQOffset,
            softmax_scale_log2,
            sQ_layout,
            gmem_tiled_copy_KV,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=(work_capacity, 1, 1),
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLse: cute.Tensor,
        mUnionBlocks: cute.Tensor,
        mUnionMasks: cute.Tensor,
        mUnionCount: cute.Tensor,
        mWorkMeta: cute.Tensor,
        mWorkCount: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.Layout,
        gmem_tiled_copy_KV: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        work_idx, _, _ = cute.arch.block_idx()

        has_work = cutlass.Boolean(False)
        batch_idx = cutlass.Int32(0)
        q_tile = cutlass.Int32(0)
        kv_head = cutlass.Int32(0)
        union_count = cutlass.Int32(0)
        if work_idx < mWorkCount[0]:
            batch_idx = mWorkMeta[work_idx, 0]
            q_tile = mWorkMeta[work_idx, 1]
            kv_head = mWorkMeta[work_idx, 2]
            union_count = mUnionCount[work_idx]
            has_work = union_count > 0

        if has_work:
            G = self._group_size
            tpt = self._tokens_per_tile
            q_start = mCuQ[batch_idx]
            seqlen_q = mCuQ[batch_idx + 1] - q_start
            k_start = mCuK[batch_idx]
            seqlen_k = mCuK[batch_idx + 1] - k_start
            tile_first = q_tile * tpt
            q_off = mQOffset[batch_idx]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sK = storage.sK.get_tensor(self._make_skv_layout())
            sV = storage.sV.get_tensor(self._make_skv_layout())
            sQ = storage.sQ.get_tensor(sQ_layout)
            sVt = cute.composition(
                sV,
                cute.make_layout(
                    (self._head_dim, self._n_block_size),
                    stride=(self._n_block_size, 1),
                ),
            )

            # ---- gather the query tile once: token i -> G GQA-head rows ----
            zero_chunk = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
            zero_chunk.fill(0)
            chunks_per_row = self._head_dim // 8
            total_chunks = self._m_block_size * chunks_per_row
            for it in cutlass.range_constexpr(total_chunks // self._num_threads):
                chunk = tidx + it * self._num_threads
                m = chunk // chunks_per_row
                c8 = chunk % chunks_per_row
                tok = m // G
                g = m % G
                q_loc = tile_first + tok
                sChunk = cute.local_tile(sQ[m, None], (8,), (c8,))
                if cute.elem_less(q_loc, seqlen_q):
                    gQ_row = mQ[q_start + q_loc, kv_head * G + g, None]
                    cute.autovec_copy(cute.local_tile(gQ_row, (8,), (c8,)), sChunk)
                else:
                    cute.autovec_copy(zero_chunk, sChunk)

            # ---- MMA partitions / staging (Q fragments built once) ----
            thr_mma = tiled_mma.get_slice(tidx)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
            acc_shape_S = thr_mma.partition_shape_C(
                (self._m_block_size, self._n_block_size)
            )
            acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
            acc_shape_O = thr_mma.partition_shape_C(
                (self._m_block_size, self._head_dim)
            )
            acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)

            smem_copy_atom_V = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
            )
            smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
            smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
            tOsVt = smem_thr_copy_V.partition_S(sVt)
            tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

            smem_copy_atom_QK = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)

            cS = cute.make_identity_tensor((self._m_block_size, self._n_block_size))
            tScS = thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)
            acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            n_rows = cute.size(acc_O_mn.shape[0])

            # Q fragments (loaded once; K/V reload per union block)
            self.cta_sync_barrier.arrive_and_wait()
            for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, k],
                    tSrQ_copy_view[None, None, k],
                )

            # ---- online-softmax running state ----
            row_max = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
            row_sum = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
            for r in cutlass.range_constexpr(n_rows):
                row_max[r] = -cutlass.Float32.inf
                row_sum[r] = cutlass.Float32.zero
            acc_O.fill(0.0)

            gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(tidx)
            tKsK = gmem_thr_copy_KV.partition_D(sK)
            tVsV = gmem_thr_copy_KV.partition_D(sV)
            cKV = cute.make_identity_tensor((self._n_block_size, self._head_dim))
            tKVcKV = gmem_thr_copy_KV.partition_S(cKV)
            mK_h = cute.domain_offset((k_start, 0), mK[None, kv_head, None])
            mV_h = cute.domain_offset((k_start, 0), mV[None, kv_head, None])
            gK_all = cute.local_tile(
                mK_h, (self._n_block_size, self._head_dim), (None, 0)
            )
            gV_all = cute.local_tile(
                mV_h, (self._n_block_size, self._head_dim), (None, 0)
            )
            tKgK = gmem_thr_copy_KV.partition_S(gK_all)
            tVgV = gmem_thr_copy_KV.partition_S(gV_all)

            # ///////////////////////////////////////////////////////////////
            # Loop over the union of KV blocks selected by this tile's tokens
            # ///////////////////////////////////////////////////////////////
            for u in cutlass.range(union_count):
                kv_block = mUnionBlocks[work_idx, u]
                mask_word = mUnionMasks[work_idx, u]
                base = kv_block * self._n_block_size

                # previous block's K/V fragment reads must be done first
                self.cta_sync_barrier.arrive_and_wait()
                for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                    if cute.elem_less(base + tKVcKV[0, n, 0][0], seqlen_k):
                        cute.copy(
                            gmem_tiled_copy_KV,
                            tKgK[None, n, None, kv_block],
                            tKsK[None, n, None],
                        )
                        cute.copy(
                            gmem_tiled_copy_KV,
                            tVgV[None, n, None, kv_block],
                            tVsV[None, n, None],
                        )
                    else:
                        tKsK[None, n, None].fill(0)
                        tVsV[None, n, None].fill(0)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                self.cta_sync_barrier.arrive_and_wait()

                for k in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                    cute.copy(
                        smem_tiled_copy_K,
                        tSsK[None, None, k],
                        tSrK_copy_view[None, None, k],
                    )
                for k in cutlass.range_constexpr(cute.size(tOsVt.shape[2])):
                    cute.copy(
                        smem_tiled_copy_V,
                        tOsVt[None, None, k],
                        tOrVt_copy_view[None, None, k],
                    )

                # S = Q K^T
                acc_S.fill(0.0)
                for k in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):
                    cute.gemm(
                        tiled_mma,
                        acc_S,
                        tSrQ[None, None, k],
                        tSrK[None, None, k],
                        acc_S,
                    )

                # mask (union membership + causal/bounds) + online update
                for r in cutlass.range_constexpr(n_rows):
                    m_local = tScS_mn[r, 0][0]
                    tok = m_local // G
                    q_loc = tile_first + tok
                    if cutlass.const_expr(self._is_causal):
                        col_limit = cutlass.min(q_loc + q_off + 1, seqlen_k)
                    else:
                        col_limit = seqlen_k
                    # union membership: token attends this block only if its bit is
                    # set. Fold the bit into col_limit (bit==0 -> col_limit 0 -> the
                    # same causal test masks the whole row) instead of a separate
                    # branch, so a token's scores for an unselected block are -inf.
                    bit = (mask_word >> tok) & cutlass.Int32(1)
                    col_limit = col_limit * bit
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        k_pos = base + tScS_mn[0, c][1]
                        if cute.elem_less(col_limit, k_pos + 1):
                            acc_S_mn[r, c] = -cutlass.Float32.inf

                    acc_S_row = acc_S_mn[r, None].load()
                    rmax = acc_S_row.reduce(
                        cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
                    )
                    rmax = self._threadquad_reduce_max(rmax)
                    rmax_prev = row_max[r]
                    rmax = cute.arch.fmax(rmax_prev, rmax)
                    rmax_safe = 0.0 if rmax == -cutlass.Float32.inf else rmax
                    p_row = cute.math.exp2(
                        acc_S_row * softmax_scale_log2 - rmax_safe * softmax_scale_log2,
                        fastmath=True,
                    )
                    rsum = p_row.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
                    prev_scale = cute.math.exp2(
                        rmax_prev * softmax_scale_log2 - rmax_safe * softmax_scale_log2,
                        fastmath=True,
                    )
                    row_sum[r] = rsum + row_sum[r] * prev_scale
                    acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_scale
                    row_max[r] = rmax
                    acc_S_mn[r, None] = p_row

                # O += P V
                rP = cute.make_fragment_like(acc_S, self._dtype)
                rP.store(acc_S.load().to(self._dtype))
                rP_div = cute.logical_divide(rP.layout, (None, None, 2))
                tOrS = cute.make_tensor(
                    rP.iterator,
                    cute.make_layout(
                        (
                            (rP_div.shape[0], rP_div.shape[2][0]),
                            rP_div.shape[1],
                            rP_div.shape[2][1],
                        ),
                        stride=(
                            (rP_div.stride[0], rP_div.stride[2][0]),
                            rP_div.stride[1],
                            rP_div.stride[2][1],
                        ),
                    ),
                )
                for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                    cute.gemm(
                        tiled_mma,
                        acc_O,
                        tOrS[None, None, k],
                        tOrVt[None, None, k],
                        acc_O,
                    )

            # ///////////////////////////////////////////////////////////////
            # Epilogue: normalize and store the final output + LSE directly
            # ///////////////////////////////////////////////////////////////
            for r in cutlass.range_constexpr(n_rows):
                m_local = tScS_mn[r, 0][0]
                tok = m_local // G
                g = m_local % G
                q_loc = tile_first + tok
                if cute.elem_less(q_loc, seqlen_q):
                    rs = self._threadquad_reduce_sum(row_sum[r])
                    rmax_final = row_max[r]
                    rmax_safe = (
                        0.0 if rmax_final == -cutlass.Float32.inf else rmax_final
                    )
                    inv = 0.0 if (rs == 0.0 or rs != rs) else cute.arch.rcp_approx(rs)
                    q_global = q_start + q_loc
                    hq = kv_head * G + g
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        d_pos = tScS_mn[0, c][1]
                        mO[q_global, hq, d_pos] = (acc_O_mn[r, c] * inv).to(
                            mO.element_type
                        )
                    if cutlass.const_expr(self._return_softmax_lse):
                        mLse[hq, q_global] = (
                            rmax_safe * softmax_scale_log2
                            + cute.math.log2(rs, fastmath=True)
                        )

    @cute.jit
    def _threadquad_reduce_max(self, val):
        val = cute.arch.fmax(
            val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31)
        )
        val = cute.arch.fmax(
            val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31)
        )
        return val

    @cute.jit
    def _threadquad_reduce_sum(self, val):
        val = val + cute.arch.shuffle_sync_bfly(
            val, offset=2, mask=-1, mask_and_clamp=31
        )
        val = val + cute.arch.shuffle_sync_bfly(
            val, offset=1, mask=-1, mask_and_clamp=31
        )
        return val

    def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
        acc_layout_col_major = cute.make_layout(acc.layout.shape)
        acc_layout_mn = cute.make_layout(
            (
                (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
            ),
            stride=(
                (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
            ),
        )
        acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
        return cute.make_tensor(acc.iterator, acc_layout_mn)
