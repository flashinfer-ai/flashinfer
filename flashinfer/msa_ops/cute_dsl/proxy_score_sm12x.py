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

MSA dense proxy-score kernels (bf16/fp16) for SM120/SM121: per (head, KV block,
query) max of the unscaled post-mask QK^T logits; masked blocks yield -inf.
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp


class MsaProxyScoreSm12x:
    """General bf16/fp16 proxy schedule (any q length, fp8 K, flat or paged):
    every query head scores against its kv_head's index-K independently."""

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

        # Split-K: fold a kv-block split factor into grid x to fill the SMs when the
        # base grid (1, batch, heads) starves at low batch. Output is per-(head,
        # kv_block, query) so splits write disjoint columns, no reduction.
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
        # grid x packs (m_block, split): splits of one m_block are adjacent so
        # they share the same Q tile in L2 (only the kv-block subset differs).
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

            # K is in this tiled copy only on the non-fp8 path (fp8 K loads separately)
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

            # K gmem views; the block mode is indexed after partitioning so the
            # copy tile shapes stay static.
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

            # sQ is reused across all kv-blocks, so Q fragments load once here
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, kk],
                    tSrQ_copy_view[None, None, kk],
                )

            # Split-K CTAs stride kv_block by num_splits. Causal skip: kv-blocks
            # entirely above the q-tile's causal limit get their -inf written directly,
            # with no K load or MMA (masking after the MMA would waste the work).
            if cutlass.const_expr(self._is_causal):
                causal_last = (
                    m_block * self._m_block_size
                    + (self._m_block_size - 1)
                    + mQOffset[batch_idx]
                ) // self._n_block_size
                last_block = cutlass.min(causal_last, num_kv_blocks - 1)
            else:
                last_block = num_kv_blocks - 1

            n_iter = cute.ceil_div(max_k_tiles, num_splits)
            for it in cutlass.range(n_iter):
                kv_block = split_idx + it * num_splits
                if kv_block <= last_block:
                    self.cta_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(self._kv_fp8):
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
                        # Branch bodies keep their own tensor names: aliasing an outer
                        # tensor in this dynamic loop would make it loop-carried.
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
                            # All 4 quad threads hold the same value; the
                            # duplicate stores are idempotent.
                            mMaxScore[qo_head, kv_block, q_start + q_loc] = tile_max
                elif kv_block < max_k_tiles:
                    # Blocks past last_block (causally future or padding) get -inf, no
                    # MMA. Bounded by max_k_tiles so split-K overshoot writes nothing.
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


class MsaProxyScoreDecodePackedSm12x(MsaProxyScoreSm12x):
    """Head-fused packed-decode bf16/fp16 schedule: packs qhead_per_kv heads x
    pack_q_len tokens into the M rows so the shared index-K is read once per kv_head."""

    def __init__(
        self,
        head_dim: int = 128,
        m_block_size: int = 64,
        n_block_size: int = 128,
        num_threads: int = 128,
        is_causal: bool = True,
        paged: bool = False,
        kv_fp8: bool = False,
        qhead_per_kv: int = 4,
        pack_q_len: int = 16,
    ):
        super().__init__(
            head_dim=head_dim,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            num_threads=num_threads,
            is_causal=is_causal,
            paged=paged,
            kv_fp8=kv_fp8,
        )
        if qhead_per_kv * pack_q_len != m_block_size:
            raise ValueError(
                f"qhead_per_kv * pack_q_len must equal the {m_block_size}-row MMA "
                f"tile, got {qhead_per_kv} x {pack_q_len}"
            )
        # These attrs are constexpr-baked into the gather/epilogue, so each
        # factorization compiles to its own kernel.
        self._qhead_per_kv = qhead_per_kv
        self._pack_q_len = pack_q_len

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPageTable: cute.Tensor,
        mMaxScore: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
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

        # One CTA per (split, batch, kv_head): all qhead_per_kv heads of a kv_head
        # are packed into the single m_block_size-row tile, so the index-K is read
        # once per kv_head (qhead_per_kv x less K traffic than the per-head schedule).
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
                num_splits,
                batch_size,
                num_qo_heads // self._qhead_per_kv,
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
        split_idx, batch_idx, kv_head = cute.arch.block_idx()

        q_start = mCuQ[batch_idx]
        seqlen_q = mCuQ[batch_idx + 1] - q_start
        k_start = mCuK[batch_idx]
        seqlen_k = mCuK[batch_idx + 1] - k_start
        num_kv_blocks = cute.ceil_div(seqlen_k, self._n_block_size)

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

        # Tiled copy covers K only; Q is gathered per-head below
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
        gmem_tiled_copy = cute.make_tiled_copy_tv(atom_async_copy, t_layout, v_layout)
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)

        self._gather_packed_q(mQ, sQ, q_start, kv_head, seqlen_q, tidx)

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

        # K gmem views, as in the general kernel.
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

        # Gather stores are synchronous; order them before the Q fragment reads,
        # then preload the Q fragments once (sQ reused across all kv-blocks).
        self.cta_sync_barrier.arrive_and_wait()
        for kk in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
            cute.copy(
                smem_tiled_copy_Q,
                tSsQ[None, None, kk],
                tSrQ_copy_view[None, None, kk],
            )

        # Split-K CTAs stride kv_block by num_splits, as in the general kernel.
        # No causal skip here: packed decode attends every kv-block and masks
        # the per-token causal tail in the epilogue.
        n_iter = cute.ceil_div(max_k_tiles, num_splits)
        for it in cutlass.range(n_iter):
            kv_block = split_idx + it * num_splits
            if kv_block < num_kv_blocks:
                self.cta_sync_barrier.arrive_and_wait()
                if cutlass.const_expr(self._kv_fp8):
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
                    cvt_frag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
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

                # A packed row maps to (local_head, token); a packed query's causal
                # position is its token index (the decode q within this step).
                for r in cutlass.range_constexpr(n_rows):
                    row = tScS_mn[r, 0][0]
                    token = row % self._pack_q_len
                    if cutlass.const_expr(self._is_causal):
                        col_limit = cutlass.min(
                            token + mQOffset[batch_idx] + 1, seqlen_k
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
                    local_head = row // self._pack_q_len
                    head = kv_head * self._qhead_per_kv + local_head
                    if cute.elem_less(token, seqlen_q):
                        # All 4 quad threads hold the same value; the duplicate
                        # stores are idempotent.
                        mMaxScore[head, kv_block, q_start + token] = tile_max
            elif kv_block < max_k_tiles:
                # Padding tile in [num_kv_blocks, max_k_tiles) -> -inf. Bounded by
                # max_k_tiles because split-K can stride kv_block past the last
                # output column (those overshoot blocks write nothing).
                for r in cutlass.range_constexpr(n_rows):
                    row2 = tScS_mn[r, 0][0]
                    token2 = row2 % self._pack_q_len
                    local_head2 = row2 // self._pack_q_len
                    head2 = kv_head * self._qhead_per_kv + local_head2
                    if cute.elem_less(token2, seqlen_q):
                        mMaxScore[
                            head2, kv_block, q_start + token2
                        ] = -cutlass.Float32.inf

    @cute.jit
    def _gather_packed_q(self, mQ, sQ, q_start, kv_head, seqlen_q, tidx):
        """Gather the packed Q tile into swizzled smem; rows past seqlen_q are
        zero-filled (epilogue masks them)."""
        elems = 128 // self._dtype.width
        chunks = self._head_dim // elems
        total = self._m_block_size * chunks
        frag = cute.make_rmem_tensor(cute.make_layout(elems), self._dtype)
        for it in cutlass.range_constexpr(total // self._num_threads):
            e = tidx + it * self._num_threads
            r = e // chunks
            c8 = e % chunks
            local_head = r // self._pack_q_len
            token = r % self._pack_q_len
            head = kv_head * self._qhead_per_kv + local_head
            sQ_chunk = cute.local_tile(sQ[r, None], (elems,), (c8,))
            if cute.elem_less(token, seqlen_q):
                gQ_chunk = cute.local_tile(
                    mQ[q_start + token, head, None], (elems,), (c8,)
                )
                frag.store(gQ_chunk.load())
                cute.autovec_copy(frag, sQ_chunk)
            else:
                sQ_chunk.fill(0)


class MsaProxyScoreDecodeStreamSm12x:
    """Single-token decode schedule: at one query token the score tile is only
    group_size rows, too skinny for the MMA atom, so K streams straight from
    GMEM to registers (no smem, no tensor cores) and dim-parallel lanes reduce
    each key's dot products with warp shuffles.

    Deliberately q_len == 1 only: a pack_q > 1 (MTP) variant reusing the K
    registers across tokens measured 0.67-0.99x of the packed-MMA schedule
    (5080, Sq 2-8) -- the shuffle+FMA phase scales with Sq while the K loads
    don't, and the rising arithmetic intensity at m = Sq * group_size rows is
    exactly what starts paying for the tensor cores. MTP stays packed-MMA."""

    _NUM_WARPS = 8
    _KEYS_PER_ITER = 2  # 32 lanes = 2 keys x 16 dim-slice lanes

    def __init__(
        self,
        head_dim: int = 128,
        group_size: int = 4,
        is_causal: bool = True,
        paged: bool = False,
        qoff_default: bool = True,
    ):
        if head_dim != 128:
            raise ValueError("only head_dim == 128 is supported")
        # Q and the running maxes live in registers per lane; past 8 heads the
        # footprint approaches the spill point, so larger groups keep the
        # packed MMA schedule.
        if not 1 <= group_size <= 8:
            raise ValueError("group_size must be in [1, 8]")
        self._head_dim = head_dim
        self._G = group_size
        self._is_causal = is_causal
        self._paged = paged
        # Right-aligned decode (q_offset=None): the single query sits at
        # seqlen_k - 1, so the causal limit is just seqlen_k and no offset
        # tensor is needed.
        self._qoff_default = qoff_default
        self._blk_kv = 128
        self._num_threads = self._NUM_WARPS * 32
        self._keys_per_warp = self._blk_kv // self._NUM_WARPS
        self._num_iters = self._keys_per_warp // self._KEYS_PER_ITER

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, Hq, d)
        mK: cute.Tensor,  # (total_k, Hkv, d) flat | (pages, Hkv, 128, d) paged
        mPageTable: cute.Tensor,  # (B, max_pages) int32 (dummy if flat)
        mMaxScore: cute.Tensor,  # (Hq, max_k_tiles, total_q) f32
        mCuQ: cute.Tensor,  # (B + 1,) int32 (unused: one q token per request)
        mCuK: cute.Tensor,  # (B + 1,) int32
        mQOffset: cute.Tensor,  # (B,) int32 causal offset (MSA q_offset)
        max_seqlen_q: cutlass.Int32,  # 1 (kept for signature parity)
        batch_size: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        max_k_tiles: cutlass.Int32,
        num_splits: cutlass.Int32,  # unused: the grid is already per KV block
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
        self.kernel(mQ, mK, mPageTable, mMaxScore, mCuK, mQOffset).launch(
            grid=(max_k_tiles, batch_size, num_qo_heads // self._G),
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
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
    ):
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        tile, qi, kv_head = cute.arch.block_idx()

        G = self._G
        k_start = mCuK[qi]
        seqlen_k = mCuK[qi + 1] - k_start
        if cutlass.const_expr(self._is_causal and not self._qoff_default):
            col_limit = cutlass.min(mQOffset[qi] + 1, seqlen_k)
        else:
            col_limit = seqlen_k
        tile_base = tile * self._blk_kv

        lane_key = lane >> 4  # which of the warp's two keys this half-warp owns
        lane_grp = lane & 15  # this lane's 8-dim slice index within its key

        mfrag = cute.make_rmem_tensor((G,), cutlass.Float32)
        mfrag.fill(-cutlass.Float32.inf)

        # tile/qi are CTA-uniform, so the whole block skips dead tiles together
        # (their output stays -inf) and the shuffles below stay full-warp.
        if tile_base < col_limit:
            qfrags = [
                cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
                for _ in range(G)
            ]
            for g in cutlass.range_constexpr(G):
                q_chunk = cute.local_tile(
                    mQ[qi, kv_head * G + g, None], (8,), (lane_grp,)
                )
                cute.autovec_copy(q_chunk, qfrags[g])
            qs = [[cutlass.Float32(qfrags[g][i]) for i in range(8)] for g in range(G)]

            if cutlass.const_expr(self._paged):
                page = mPageTable[qi, tile]
                mK_blk = mK[page, kv_head, None, None]  # (128, d)
            else:
                mK_h = mK[None, kv_head, None]  # (total_k, d)

            # Keep this loop separate from the compute loop below: issuing all
            # loads before any consumption is the latency hiding.
            kfrag = cute.make_rmem_tensor(
                cute.make_layout(self._num_iters * 8), self._dtype
            )
            for it in cutlass.range_constexpr(self._num_iters):
                off = warp * self._keys_per_warp + it * self._KEYS_PER_ITER + lane_key
                # Tail keys clamp to the tile's first row (always in range for a
                # live tile) so loads stay unpredicated; the max step masks them.
                ld_off = off if tile_base + off < col_limit else 0
                if cutlass.const_expr(self._paged):
                    k_row = mK_blk[ld_off, None]
                else:
                    k_row = mK_h[k_start + tile_base + ld_off, None]
                k_chunk = cute.local_tile(k_row, (8,), (lane_grp,))
                dst = cute.make_tensor(kfrag.iterator + it * 8, cute.make_layout(8))
                cute.autovec_copy(k_chunk, dst)

            for it in cutlass.range_constexpr(self._num_iters):
                ks = [cutlass.Float32(kfrag[it * 8 + i]) for i in range(8)]
                es = [
                    qs[g][0] * ks[0]
                    + qs[g][1] * ks[1]
                    + qs[g][2] * ks[2]
                    + qs[g][3] * ks[3]
                    + qs[g][4] * ks[4]
                    + qs[g][5] * ks[5]
                    + qs[g][6] * ks[6]
                    + qs[g][7] * ks[7]
                    for g in range(G)
                ]
                # Butterfly-add over the 16 dim-slice lanes of this key.
                for s in cutlass.range_constexpr(4):
                    off_r = 8 >> s
                    es = [
                        es[g] + cute.arch.shuffle_sync_bfly(es[g], off_r)
                        for g in range(G)
                    ]
                pos = (
                    tile_base
                    + warp * self._keys_per_warp
                    + it * self._KEYS_PER_ITER
                    + lane_key
                )
                if pos < col_limit:
                    for g in cutlass.range_constexpr(G):
                        mfrag[g] = cute.arch.fmax(mfrag[g], es[g])

            # Max-combine the two half-warps' keys.
            for g in cutlass.range_constexpr(G):
                mfrag[g] = cute.arch.fmax(
                    mfrag[g], cute.arch.shuffle_sync_bfly(mfrag[g], 16)
                )

        @cute.struct
        class SharedStorage:
            s_max: cute.struct.MemRange[cutlass.Float32, self._NUM_WARPS * G]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_max = storage.s_max.get_tensor(
            cute.make_layout((self._NUM_WARPS, G), stride=(G, 1))
        )
        if lane == 0:
            for g in cutlass.range_constexpr(G):
                s_max[warp, g] = mfrag[g]
        cute.arch.sync_threads()

        if warp == 0 and lane < G:
            r = s_max[0, lane]
            for w in cutlass.range_constexpr(1, self._NUM_WARPS):
                r = cute.arch.fmax(r, s_max[w, lane])
            mMaxScore[kv_head * G + lane, tile, qi] = r
