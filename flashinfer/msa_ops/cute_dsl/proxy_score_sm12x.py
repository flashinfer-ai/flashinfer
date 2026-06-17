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

MSA dense proxy-score kernel for SM120/SM121 (the SM12x equivalent of MSA's
``SparseAttnMode::OnlyScore`` dense FMHA variant).

Computes, for every (query head, KV block, query token):

    max_score[h, t, q] = max over the 128 tokens of KV block t of the
                         UNSCALED, post-mask QK^T logits

matching MSA's contract (sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp:
the per-tile max is taken before ``scale_softmax_log2`` is applied, after
masking). KV blocks beyond a sequence's valid range — and blocks entirely
above the causal limit — produce ``-inf``, which is what
:func:`msa_topk_select` expects for invalid tiles.

There is no softmax, no PV matmul and no output tensor: this is a QK GEMM
with a fused per-block row-max epilogue. Structurally it is the sparse
attention forward with the selection map, online softmax and V machinery
removed, and the KV loop made dense.
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp


class MsaProxyScoreSm12x:
    def __init__(
        self,
        head_dim: int = 128,
        m_block_size: int = 64,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = True,
        paged: bool = False,
        kv_fp8: bool = False,
    ):
        if head_dim != 128 or n_block_size != 128:
            raise ValueError("only head_dim == n_block_size == 128 is supported")
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._num_threads = num_threads
        self._is_causal = is_causal
        self._paged = paged
        self._kv_fp8 = kv_fp8
        self._pad_stride = head_dim + 8

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    def _make_sk_layout(self):
        """K SMEM layout (built in-kernel for region isolation): swizzled for
        the cp.async path, plain padded for the fp8 convert path."""
        if self._kv_fp8:
            return cute.make_layout(
                (self._n_block_size, self._head_dim),
                stride=(self._pad_stride, 1),
            )
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        return cute.tile_to_shape(atom, (self._n_block_size, self._head_dim), (0, 1))

    def _make_sq_layout(self):
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        return cute.tile_to_shape(atom, (self._m_block_size, self._head_dim), (0, 1))

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, Hq, d)
        mK: cute.Tensor,  # (total_k, Hkv, d) flat | (pages, Hkv, 128, d) paged
        mPageTable: cute.Tensor,  # (B, max_pages) int32 (dummy (1,1) if flat)
        mMaxScore: cute.Tensor,  # (Hq, max_k_tiles, total_q) f32
        mCuQ: cute.Tensor,  # (B + 1,) int32
        mCuK: cute.Tensor,  # (B + 1,) int32
        mQOffset: cute.Tensor,  # (B,) int32 causal offset (MSA q_offset)
        max_seqlen_q: cutlass.Int32,
        batch_size: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        max_k_tiles: cutlass.Int32,
        num_splits: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (
                mQ.element_type == cutlass.Float16
                or mQ.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 q is supported")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type

        # split-K: fold a kv-block split factor into grid x. At long context with
        # low batch the base grid (1, batch, heads) is < #SMs (e.g. B8 q1 -> 32
        # CTAs) so the GPU starves; splitting each sequence's kv-block range across
        # CTAs fills it. The output is per-(head, kv_block, query), so the splits
        # write disjoint columns and need no reduction. The host passes
        # num_splits==1 when the base grid already fills the SMs.
        self.kernel(
            mQ,
            mK,
            mPageTable,
            mMaxScore,
            mCuQ,
            mCuK,
            mQOffset,
            max_k_tiles,
            num_splits,
        ).launch(
            grid=(
                cute.ceil_div(max_seqlen_q, self._m_block_size) * num_splits,
                batch_size,
                num_qo_heads,
            ),
            block=[self._num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPageTable: cute.Tensor,
        mMaxScore: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
        max_k_tiles: cutlass.Int32,
        num_splits: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, batch_idx, qo_head = cute.arch.block_idx()
        # grid x packs (m_block, split): splits of one m_block are adjacent.
        m_block = bx // num_splits
        split_idx = bx % num_splits

        q_start = mCuQ[batch_idx]
        seqlen_q = mCuQ[batch_idx + 1] - q_start
        k_start = mCuK[batch_idx]
        seqlen_k = mCuK[batch_idx + 1] - k_start
        num_kv_blocks = cute.ceil_div(seqlen_k, self._n_block_size)
        group_size = mQ.shape[1] // mK.shape[1]
        kv_head = qo_head // group_size

        if m_block * self._m_block_size < seqlen_q:
            sQ_layout = self._make_sq_layout()
            sK_layout = self._make_sk_layout()

            @cute.struct
            class SharedStorage:
                sQ: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)],
                    1024,
                ]
                sK: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)],
                    1024,
                ]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)

            # ///////////////////////////////////////////////////////////////
            # GMEM tiled copy (Q always; K only on the non-fp8 path)
            # ///////////////////////////////////////////////////////////////
            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self._dtype.width
            atom_async_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self._dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            t_dim1 = 64 // async_copy_elems
            t_layout = cute.make_layout(
                (self._num_threads // t_dim1, t_dim1), stride=(t_dim1, 1)
            )
            v_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy = cute.make_tiled_copy_tv(
                atom_async_copy, t_layout, v_layout
            )
            gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)

            mQ_h = cute.domain_offset((q_start, 0), mQ[None, qo_head, None])
            gQ = cute.local_tile(
                mQ_h, (self._m_block_size, self._head_dim), (m_block, 0)
            )
            tQgQ = gmem_thr_copy.partition_S(gQ)
            tQsQ = gmem_thr_copy.partition_D(sQ)
            cQ = cute.make_identity_tensor((self._m_block_size, self._head_dim))
            tQcQ = gmem_thr_copy.partition_S(cQ)

            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if (m_block * self._m_block_size + tQcQ[0, m, 0][0]) < seqlen_q:
                    cute.copy(
                        gmem_tiled_copy,
                        tQgQ[None, m, None],
                        tQsQ[None, m, None],
                    )
                else:
                    tQsQ[None, m, None].fill(0)
            cute.arch.cp_async_commit_group()

            # ///////////////////////////////////////////////////////////////
            # MMA setup (S = Q K^T only; no V)
            # ///////////////////////////////////////////////////////////////
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
                (self._num_threads // 32, 1, 1),
                permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            acc_S = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)),
                cutlass.Float32,
            )

            smem_copy_atom = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self._dtype,
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom, tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)

            cS = cute.make_identity_tensor((self._m_block_size, self._n_block_size))
            tScS_mn = self._make_acc_tensor_mn_view(thr_mma.partition_C(cS))
            acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
            n_rows = cute.size(acc_S_mn.shape[0])

            # K gmem views (block mode indexed after partitioning so copy tile
            # shapes stay static)
            if cutlass.const_expr(not self._kv_fp8):
                if cutlass.const_expr(self._paged):
                    pass  # per-block page lookup happens inside the loop
                else:
                    mK_h = cute.domain_offset((k_start, 0), mK[None, kv_head, None])
                    gK = cute.local_tile(
                        mK_h, (self._n_block_size, self._head_dim), (None, 0)
                    )
                    tKgK_flat = gmem_thr_copy.partition_S(gK)
                tKsK = gmem_thr_copy.partition_D(sK)
                cKV = cute.make_identity_tensor((self._n_block_size, self._head_dim))
                tKVcKV = gmem_thr_copy.partition_S(cKV)

            # wait for Q, preload its fragments once
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, kk],
                    tSrQ_copy_view[None, None, kk],
                )

            # ///////////////////////////////////////////////////////////////
            # split-K: this CTA owns kv_block = split_idx, split_idx + num_splits,
            # ... < max_k_tiles (disjoint across splits, union covers every column
            # once, no reduction). num_splits==1 -> the original [0, max_k_tiles).
            # ///////////////////////////////////////////////////////////////
            n_iter = cute.ceil_div(max_k_tiles, num_splits)
            for it in cutlass.range(n_iter):
                kv_block = split_idx + it * num_splits
                if kv_block < num_kv_blocks:
                    # previous iteration's K fragment reads must be complete
                    self.cta_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(self._kv_fp8):
                        # fp8 K: vectorized load -> upconvert -> SMEM
                        if cutlass.const_expr(self._paged):
                            page8 = mPageTable[batch_idx, kv_block]
                            mK_h8 = mK[page8, kv_head, None, None]
                            row_off8 = cutlass.Int32(0)
                        else:
                            mK_h8 = cute.domain_offset(
                                (k_start, 0), mK[None, kv_head, None]
                            )
                            row_off8 = kv_block * self._n_block_size
                        chunks_per_row = self._head_dim // 8
                        total_chunks = self._n_block_size * chunks_per_row
                        cvt_frag = cute.make_rmem_tensor(
                            cute.make_layout(8), self._dtype
                        )
                        for kv_it in cutlass.range_constexpr(
                            total_chunks // self._num_threads
                        ):
                            kv_chunk = tidx + kv_it * self._num_threads
                            kv_m = kv_chunk // chunks_per_row
                            kv_c8 = kv_chunk % chunks_per_row
                            sK_chunk = cute.local_tile(sK[kv_m, None], (8,), (kv_c8,))
                            if (kv_block * self._n_block_size + kv_m) < seqlen_k:
                                gKc = cute.local_tile(
                                    mK_h8[row_off8 + kv_m, None], (8,), (kv_c8,)
                                )
                                cvt_frag.store(
                                    gKc.load()
                                    .to(cutlass.Float16)
                                    .to(cutlass.Float32)
                                    .to(self._dtype)
                                )
                            else:
                                cvt_frag.fill(0)
                            cute.autovec_copy(cvt_frag, sK_chunk)
                    else:
                        # branch bodies keep their own tensor names: aliasing
                        # an outer tensor inside this dynamic loop would make
                        # it loop-carried (see project DSL notes)
                        if cutlass.const_expr(self._paged):
                            page = mPageTable[batch_idx, kv_block]
                            gK_pg = cute.local_tile(
                                mK[page, kv_head, None, None],
                                (self._n_block_size, self._head_dim),
                                (None, 0),
                            )
                            tKgK_pg = gmem_thr_copy.partition_S(gK_pg)
                            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                                if (
                                    kv_block * self._n_block_size + tKVcKV[0, n, 0][0]
                                ) < seqlen_k:
                                    cute.copy(
                                        gmem_tiled_copy,
                                        tKgK_pg[None, n, None, 0],
                                        tKsK[None, n, None],
                                    )
                                else:
                                    tKsK[None, n, None].fill(0)
                        else:
                            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                                if (
                                    kv_block * self._n_block_size + tKVcKV[0, n, 0][0]
                                ) < seqlen_k:
                                    cute.copy(
                                        gmem_tiled_copy,
                                        tKgK_flat[None, n, None, kv_block],
                                        tKsK[None, n, None],
                                    )
                                else:
                                    tKsK[None, n, None].fill(0)
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(0)
                    self.cta_sync_barrier.arrive_and_wait()

                    # S = Q K^T (unscaled)
                    acc_S.fill(0.0)
                    for kk in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                        cute.copy(
                            smem_tiled_copy_K,
                            tSsK[None, None, kk],
                            tSrK_copy_view[None, None, kk],
                        )
                    for kk in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                        cute.gemm(
                            tiled_mma,
                            acc_S,
                            tSrQ[None, None, kk],
                            tSrK[None, None, kk],
                            acc_S,
                        )

                    # mask (bounds + causal), per-block row max, store
                    for r in cutlass.range_constexpr(n_rows):
                        row_local = tScS_mn[r, 0][0]
                        q_loc = m_block * self._m_block_size + row_local
                        if cutlass.const_expr(self._is_causal):
                            col_limit = cutlass.min(
                                q_loc + mQOffset[batch_idx] + 1, seqlen_k
                            )
                        else:
                            col_limit = seqlen_k
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            k_pos = kv_block * self._n_block_size + tScS_mn[0, c][1]
                            if cute.elem_less(col_limit, k_pos + 1):
                                acc_S_mn[r, c] = -cutlass.Float32.inf
                        tile_max = (
                            acc_S_mn[r, None]
                            .load()
                            .reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
                        )
                        tile_max = self._threadquad_reduce_max(tile_max)
                        if q_loc < seqlen_q:
                            # all 4 quad threads hold the same value; the
                            # duplicate stores are idempotent
                            mMaxScore[qo_head, kv_block, q_start + q_loc] = tile_max
                elif kv_block < max_k_tiles:
                    # padding tile in [num_kv_blocks, max_k_tiles) -> -inf. Bounded
                    # by max_k_tiles because split-K can stride kv_block past the
                    # last output column (those overshoot blocks write nothing).
                    for r in cutlass.range_constexpr(n_rows):
                        row_local2 = tScS_mn[r, 0][0]
                        q_loc2 = m_block * self._m_block_size + row_local2
                        if q_loc2 < seqlen_q:
                            mMaxScore[
                                qo_head, kv_block, q_start + q_loc2
                            ] = -cutlass.Float32.inf

    @cute.jit
    def _threadquad_reduce_max(self, val):
        val = cute.arch.fmax(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = cute.arch.fmax(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
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
