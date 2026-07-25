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

Union-tile MSA sparse attention forward (prefill) kernel for SM120/SM121,
plus the on-device union-metadata builder that feeds it.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp


class SparsePrefillSm12x:
    """Union-tile sparse prefill: a query tile processes the union of its
    tokens' selected KV blocks in one online-softmax pass, so each block loads
    once per tile and the final output is written directly."""

    def __init__(
        self,
        head_dim: int = 128,
        m_block_size: int = 64,
        n_block_size: int = 128,
        group_size: int = 1,
        num_threads: int = 128,
        is_causal: bool = True,
        return_softmax_lse: bool = False,
        return_temperature_lse: bool = False,
        paged: bool = False,
        kv_fp8: bool = False,
        kv_nvfp4: bool = False,
    ):
        if head_dim != 128 or n_block_size != 128:
            raise ValueError("only head_dim == n_block_size == 128 is supported")
        if m_block_size % group_size != 0:
            raise ValueError("group_size must divide m_block_size")
        if kv_fp8 and kv_nvfp4:
            raise ValueError("kv_fp8 and kv_nvfp4 are mutually exclusive")
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._group_size = group_size
        self._tokens_per_tile = m_block_size // group_size
        self._num_threads = num_threads
        self._is_causal = is_causal
        self._return_softmax_lse = return_softmax_lse
        self._return_temperature_lse = return_temperature_lse
        self._paged = paged
        # fp8/NVFP4 K/V are dequantized to the compute dtype on load (no native
        # fp8/fp4 MMA); the mainloop and swizzled SMEM layout are unchanged, only
        # the K/V HBM read shrinks to ~1 / ~0.5 byte per element.
        self._kv_fp8 = kv_fp8
        self._kv_nvfp4 = kv_nvfp4
        self._q_smem_stride = head_dim + 8
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    def _make_skv_layout(self):
        """Bank-swizzled K/V SMEM layout."""
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_layout((8, 64), stride=(64, 1)),
        )
        return cute.tile_to_shape(atom, (self._n_block_size, self._head_dim), (0, 1))

    @cute.jit
    def _load_kv_fp8(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: cute.Tensor,
        batch_idx: cutlass.Int32,
        kv_head: cutlass.Int32,
        k_start: cutlass.Int32,
        kv_block: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        """Dequant-on-load fp8 (E4M3) K/V into the swizzled compute-dtype SMEM the
        bf16 path fills. Rows past ``seqlen_k`` are zero-filled (epilogue masks)."""
        if cutlass.const_expr(self._paged):
            page = mPageTable[batch_idx, kv_block]
            mK_h8 = mK[page, kv_head, None, None]
            mV_h8 = mV[page, kv_head, None, None]
            row_off = cutlass.Int32(0)
        else:
            mK_h8 = cute.domain_offset((k_start, 0), mK[None, kv_head, None])
            mV_h8 = cute.domain_offset((k_start, 0), mV[None, kv_head, None])
            row_off = kv_block * self._n_block_size
        chunks_per_row = self._head_dim // 8
        total_chunks = self._n_block_size * chunks_per_row
        cvt_frag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
        for kv_it in cutlass.range_constexpr(total_chunks // self._num_threads):
            kv_chunk = tidx + kv_it * self._num_threads
            kv_m = kv_chunk // chunks_per_row
            kv_c8 = kv_chunk % chunks_per_row
            sK_chunk = cute.local_tile(sK[kv_m, None], (8,), (kv_c8,))
            sV_chunk = cute.local_tile(sV[kv_m, None], (8,), (kv_c8,))
            if (kv_block * self._n_block_size + kv_m) < seqlen_k:
                gKc = cute.local_tile(mK_h8[row_off + kv_m, None], (8,), (kv_c8,))
                gVc = cute.local_tile(mV_h8[row_off + kv_m, None], (8,), (kv_c8,))
                cvt_frag.store(
                    gKc.load().to(cutlass.Float16).to(cutlass.Float32).to(self._dtype)
                )
                cute.autovec_copy(cvt_frag, sK_chunk)
                cvt_frag.store(
                    gVc.load().to(cutlass.Float16).to(cutlass.Float32).to(self._dtype)
                )
                cute.autovec_copy(cvt_frag, sV_chunk)
            else:
                cvt_frag.fill(0)
                cute.autovec_copy(cvt_frag, sK_chunk)
                cute.autovec_copy(cvt_frag, sV_chunk)

    @cute.jit
    def _load_kv_nvfp4(
        self,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mKSf: cute.Tensor,
        mVSf: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        mPageTable: cute.Tensor,
        batch_idx: cutlass.Int32,
        kv_head: cutlass.Int32,
        k_start: cutlass.Int32,
        kv_block: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        """Dequant-on-load NVFP4 K/V into the swizzled compute-dtype SMEM. mK/mV are
        int32 word views (8 e2m1 each); global scales fold into softmax/out_scale."""
        from ...fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_fp4_helpers import (
            cvt_e4m3_to_f32_via_f16,
            fp4_decode_4bytes,
        )

        if cutlass.const_expr(self._paged):
            page = mPageTable[batch_idx, kv_block]
            mK_h4 = mK[page, kv_head, None, None]
            mV_h4 = mV[page, kv_head, None, None]
            row_off4 = cutlass.Int32(0)
            # Scale rows are flattened in (page, head, token) order (paged cache).
            sf_row_base = (page * mK.shape[1] + kv_head) * self._n_block_size
            sf_row_stride = cutlass.Int32(1)
        else:
            mK_h4 = cute.domain_offset((k_start, 0), mK[None, kv_head, None])
            mV_h4 = cute.domain_offset((k_start, 0), mV[None, kv_head, None])
            row_off4 = kv_block * self._n_block_size
            # Scale rows are flattened in (token, head) order (flat cache).
            sf_row_base = (k_start + kv_block * self._n_block_size) * mK.shape[
                1
            ] + kv_head
            sf_row_stride = mK.shape[1]
        kv_chunks_per_row = self._head_dim // 8
        kv_total_chunks = self._n_block_size * kv_chunks_per_row
        sf_tiles_n = (self._head_dim // 16 + 3) // 4
        cvt_frag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
        pair_frag = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
        pair_f16 = cute.make_tensor(
            cute.recast_ptr(pair_frag.iterator, dtype=cutlass.Float16),
            cute.make_layout(8),
        )
        for kv_it in cutlass.range_constexpr(kv_total_chunks // self._num_threads):
            kv_chunk = tidx + kv_it * self._num_threads
            kv_m = kv_chunk // kv_chunks_per_row
            kv_c8 = kv_chunk % kv_chunks_per_row
            sK_chunk = cute.local_tile(sK[kv_m, None], (8,), (kv_c8,))
            sV_chunk = cute.local_tile(sV[kv_m, None], (8,), (kv_c8,))
            if (kv_block * self._n_block_size + kv_m) < seqlen_k:
                # cuBLAS/cuDNN 128x4 tiled scale offset (an 8-elem chunk never
                # crosses a 16-elem scale block)
                srow = sf_row_base + kv_m * sf_row_stride
                scol = kv_c8 // 2
                srow_in = srow % 128
                sf_off = (
                    ((srow // 128) * sf_tiles_n + scol // 4) * 512
                    + (srow_in % 32) * 16
                    + (srow_in // 32) * 4
                    + scol % 4
                )
                k_word = mK_h4[row_off4 + kv_m, kv_c8]
                kp0, kp1, kp2, kp3 = fp4_decode_4bytes(cutlass.Uint32(k_word))
                pair_frag[0] = kp0
                pair_frag[1] = kp1
                pair_frag[2] = kp2
                pair_frag[3] = kp3
                k_sc = cvt_e4m3_to_f32_via_f16(cutlass.Uint32(mKSf[sf_off]))
                cvt_frag.store(
                    (pair_f16.load().to(cutlass.Float32) * k_sc).to(self._dtype)
                )
                cute.autovec_copy(cvt_frag, sK_chunk)
                v_word = mV_h4[row_off4 + kv_m, kv_c8]
                vp0, vp1, vp2, vp3 = fp4_decode_4bytes(cutlass.Uint32(v_word))
                pair_frag[0] = vp0
                pair_frag[1] = vp1
                pair_frag[2] = vp2
                pair_frag[3] = vp3
                v_sc = cvt_e4m3_to_f32_via_f16(cutlass.Uint32(mVSf[sf_off]))
                cvt_frag.store(
                    (pair_f16.load().to(cutlass.Float32) * v_sc).to(self._dtype)
                )
                cute.autovec_copy(cvt_frag, sV_chunk)
            else:
                cvt_frag.fill(0)
                cute.autovec_copy(cvt_frag, sK_chunk)
                cute.autovec_copy(cvt_frag, sV_chunk)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, Hq, d)
        mK: cute.Tensor,  # (total_k, Hkv, d) flat | (num_pages, Hkv, 128, d) paged
        mV: cute.Tensor,  # (total_k, Hkv, d) flat | (num_pages, Hkv, 128, d) paged
        mKSf: cute.Tensor,  # nvfp4: K block scales (flat uint8 128x4); dummy (1,) else
        mVSf: cute.Tensor,  # nvfp4: V block scales (flat uint8 128x4); dummy (1,) else
        mO: cute.Tensor,  # (total_q, Hq, d) final output
        mLse: cute.Tensor,  # (Hq, total_q) f32 LSE (dummy (1,1) if off)
        mLseT: cute.Tensor,  # (Hq, total_q) f32 temperature LSE (dummy (1,1) if off)
        mUnionBlocks: cute.Tensor,  # (capacity, max_union) int32 kv-block ids
        mUnionMasks: cute.Tensor,  # (capacity, max_union) int32 tpt-bit masks
        mUnionCount: cute.Tensor,  # (capacity,) int32 blocks in the union
        mWorkMeta: cute.Tensor,  # (capacity, 3) int32 {batch, q_tile, kv_head}
        mWorkCount: cute.Tensor,  # (1,) int32
        mCuQ: cute.Tensor,  # (B + 1,) int32
        mCuK: cute.Tensor,  # (B + 1,) int32 (paged: cumsum of seqused_k)
        mQOffset: cute.Tensor,  # (B,) int32 causal offset (MSA q_offset)
        mPageTable: cute.Tensor,  # (B, max_pages) int32 (dummy (1,1) if flat)
        softmax_scale: cutlass.Float32,  # nvfp4: K global scale pre-folded by host
        out_scale: cutlass.Float32,  # nvfp4: V global scale (1.0 otherwise); no combine
        lse_temperature_scale: cutlass.Float32,  # MSA temperature LSE scale
        work_capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(not (self._kv_fp8 or self._kv_nvfp4)):
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
            raise TypeError("Q must be Float16 or BFloat16 (the compute dtype)")
        if cutlass.const_expr(self._kv_fp8):
            if cutlass.const_expr(
                not (
                    mK.element_type == cutlass.Float8E4M3FN
                    and mV.element_type == cutlass.Float8E4M3FN
                )
            ):
                raise TypeError("kv_fp8 requires K/V to be Float8E4M3FN")
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
        lse_temp_scale_log2 = softmax_scale_log2 * lse_temperature_scale
        self.kernel(
            mQ,
            mK,
            mV,
            mKSf,
            mVSf,
            mO,
            mLse,
            mLseT,
            mUnionBlocks,
            mUnionMasks,
            mUnionCount,
            mWorkMeta,
            mWorkCount,
            mCuQ,
            mCuK,
            mQOffset,
            mPageTable,
            softmax_scale_log2,
            out_scale,
            lse_temp_scale_log2,
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
        mKSf: cute.Tensor,
        mVSf: cute.Tensor,
        mO: cute.Tensor,
        mLse: cute.Tensor,
        mLseT: cute.Tensor,
        mUnionBlocks: cute.Tensor,
        mUnionMasks: cute.Tensor,
        mUnionCount: cute.Tensor,
        mWorkMeta: cute.Tensor,
        mWorkCount: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
        mPageTable: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        out_scale: cutlass.Float32,
        lse_temp_scale_log2: cutlass.Float32,
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

            # Gather the query tile once: token i -> G GQA-head rows
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

            # MMA partitions / staging
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

            row_max = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
            row_sum = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
            row_sum_t = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
            for r in cutlass.range_constexpr(n_rows):
                row_max[r] = -cutlass.Float32.inf
                row_sum[r] = cutlass.Float32.zero
                if cutlass.const_expr(self._return_temperature_lse):
                    row_sum_t[r] = cutlass.Float32.zero
            acc_O.fill(0.0)

            gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(tidx)
            tKsK = gmem_thr_copy_KV.partition_D(sK)
            tVsV = gmem_thr_copy_KV.partition_D(sV)
            cKV = cute.make_identity_tensor((self._n_block_size, self._head_dim))
            tKVcKV = gmem_thr_copy_KV.partition_S(cKV)
            if cutlass.const_expr(not self._paged):
                # Flat: one batch-contiguous K/V tile, blocks indexed at read time.
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

            for u in cutlass.range(union_count):
                kv_block = mUnionBlocks[work_idx, u]
                mask_word = mUnionMasks[work_idx, u]
                base = kv_block * self._n_block_size

                self.cta_sync_barrier.arrive_and_wait()
                if cutlass.const_expr(self._kv_fp8):
                    # fp8/NVFP4 load+dequant is synchronous (no cp.async group to
                    # commit); only the bf16/fp16 branch below pipelines with cp.async.
                    self._load_kv_fp8(
                        mK,
                        mV,
                        sK,
                        sV,
                        mPageTable,
                        batch_idx,
                        kv_head,
                        k_start,
                        kv_block,
                        seqlen_k,
                        tidx,
                    )
                elif cutlass.const_expr(self._kv_nvfp4):
                    self._load_kv_nvfp4(
                        mK,
                        mV,
                        mKSf,
                        mVSf,
                        sK,
                        sV,
                        mPageTable,
                        batch_idx,
                        kv_head,
                        k_start,
                        kv_block,
                        seqlen_k,
                        tidx,
                    )
                else:
                    # bf16/fp16: cp.async load. Paged remaps the block through the
                    # page table (block coord 0); flat reuses the batch tile.
                    if cutlass.const_expr(self._paged):
                        gK_b = cute.local_tile(
                            mK[mPageTable[batch_idx, kv_block], kv_head, None, None],
                            (self._n_block_size, self._head_dim),
                            (None, 0),
                        )
                        gV_b = cute.local_tile(
                            mV[mPageTable[batch_idx, kv_block], kv_head, None, None],
                            (self._n_block_size, self._head_dim),
                            (None, 0),
                        )
                        tKgK_b = gmem_thr_copy_KV.partition_S(gK_b)
                        tVgV_b = gmem_thr_copy_KV.partition_S(gV_b)
                        blk_coord = cutlass.Int32(0)
                    else:
                        tKgK_b = tKgK
                        tVgV_b = tVgV
                        blk_coord = kv_block
                    for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                        if cute.elem_less(base + tKVcKV[0, n, 0][0], seqlen_k):
                            cute.copy(
                                gmem_tiled_copy_KV,
                                tKgK_b[None, n, None, blk_coord],
                                tKsK[None, n, None],
                            )
                            cute.copy(
                                gmem_tiled_copy_KV,
                                tVgV_b[None, n, None, blk_coord],
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
                    # Union membership: fold the token's bit into col_limit (bit==0 ->
                    # col_limit 0 -> the causal test masks the whole row) so an
                    # unselected block's scores go -inf without a separate branch.
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
                    if cutlass.const_expr(self._return_temperature_lse):
                        # Same masked scores and max-shift as row_sum, but with the
                        # temperature-scaled exponent; acc_S_row is the raw masked
                        # score, read before acc_S_mn is overwritten with p_row.
                        p_row_t = cute.math.exp2(
                            acc_S_row * lse_temp_scale_log2
                            - rmax_safe * lse_temp_scale_log2,
                            fastmath=True,
                        )
                        rsum_t = p_row_t.reduce(
                            cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                        )
                        prev_scale_t = cute.math.exp2(
                            rmax_prev * lse_temp_scale_log2
                            - rmax_safe * lse_temp_scale_log2,
                            fastmath=True,
                        )
                        row_sum_t[r] = rsum_t + row_sum_t[r] * prev_scale_t
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
                    # out_scale folds in the NVFP4 V global scale (1.0 otherwise);
                    # unlike decode there is no combine kernel to apply it.
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        d_pos = tScS_mn[0, c][1]
                        mO[q_global, hq, d_pos] = (acc_O_mn[r, c] * inv * out_scale).to(
                            mO.element_type
                        )
                    if cutlass.const_expr(self._return_softmax_lse):
                        mLse[hq, q_global] = (
                            rmax_safe * softmax_scale_log2
                            + cute.math.log2(rs, fastmath=True)
                        )
                    if cutlass.const_expr(self._return_temperature_lse):
                        rs_t = self._threadquad_reduce_sum(row_sum_t[r])
                        mLseT[hq, q_global] = (
                            rmax_safe * lse_temp_scale_log2
                            + cute.math.log2(rs_t, fastmath=True)
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


# Sentinel head value for an exhausted / inactive lane in the k-way merge. KV
# block ids are bounded by seqlen/128, far below this, so it never collides.
_SENTINEL = (1 << 31) - 1


class BuildUnionMetaSm12x:
    """Warp-synchronous k-way merge of a tile's per-token top-k lists into
    union blocks plus membership masks; lane i owns token i's list, so the
    merge needs only shuffles and ballots (no shared memory)."""

    def __init__(self, topk: int, tokens_per_tile: int):
        if topk not in (4, 8, 16, 32):
            raise ValueError(f"topk must be 4, 8, 16, or 32, got {topk}")
        if not 1 <= tokens_per_tile <= 32:
            raise ValueError(
                f"tokens_per_tile must be in [1, 32], got {tokens_per_tile}"
            )
        self._topk = topk
        self._tpt = tokens_per_tile
        # Distinct blocks in a union are bounded by the total list length.
        self._max_union = tokens_per_tile * topk

    @cute.jit
    def __call__(
        self,
        mQ2k: cute.Tensor,  # (Hkv, total_q, topk) int32, ascending / -1 trailing
        mTileBatch: cute.Tensor,  # (total_tiles,) int32, batch of each global tile
        mTileT: cute.Tensor,  # (total_tiles,) int32, within-batch tile index
        mTileQBase: cute.Tensor,  # (total_tiles,) int32, global query idx of token 0
        mTileNtok: cute.Tensor,  # (total_tiles,) int32, valid tokens in the tile
        mUnionBlocks: cute.Tensor,  # (work_items, max_union) int32 (out)
        mUnionMasks: cute.Tensor,  # (work_items, max_union) int32 (out)
        mUnionCount: cute.Tensor,  # (work_items,) int32 (out)
        mWorkMeta: cute.Tensor,  # (work_items, 3) int32 {batch, q_tile, kv_head} (out)
        H: cutlass.Int32,
        total_tiles: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._k_build(
            mQ2k,
            mTileBatch,
            mTileT,
            mTileQBase,
            mTileNtok,
            mUnionBlocks,
            mUnionMasks,
            mUnionCount,
            mWorkMeta,
            H,
        ).launch(
            grid=(total_tiles, H, 1),
            block=(32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def _k_build(
        self,
        mQ2k: cute.Tensor,
        mTileBatch: cute.Tensor,
        mTileT: cute.Tensor,
        mTileQBase: cute.Tensor,
        mTileNtok: cute.Tensor,
        mUnionBlocks: cute.Tensor,
        mUnionMasks: cute.Tensor,
        mUnionCount: cute.Tensor,
        mWorkMeta: cute.Tensor,
        H: cutlass.Int32,
    ):
        lane, _, _ = cute.arch.thread_idx()
        gt, h, _ = cute.arch.block_idx()
        work_idx = gt * H + h

        batch_idx = mTileBatch[gt]
        q_tile = mTileT[gt]
        qbase = mTileQBase[gt]
        ntok = mTileNtok[gt]

        # Lane i owns tile-token i; lanes >= ntok are inactive (head == SENTINEL)
        # and only participate in the warp shuffles / ballots.
        active = lane < ntok
        sentinel = cutlass.Int32(_SENTINEL)
        p = cutlass.Int32(0)  # cursor into this lane's top-k list

        u = cutlass.Int32(0)
        # A rolled loop over the fixed union bound keeps the warp trip count
        # uniform without unrolling 128-256 bodies; exhausted iterations no-op.
        for _ in cutlass.range(self._max_union):
            head = sentinel
            if active:
                if p < self._topk:
                    head = mQ2k[h, qbase + lane, p]
            # Ascending lists pad with trailing -1, so a negative head means this
            # lane is exhausted; pin it to the sentinel so it never wins the min.
            if head < 0:
                head = sentinel

            m = head
            off = 16
            while off >= 1:
                nbr = cute.arch.shuffle_sync(m, lane ^ off)
                m = cutlass.min(m, nbr)
                off >>= 1

            # m is uniform across the warp, so this branch is uniform: every lane
            # either emits this iteration or skips it together (ballot is safe).
            if m != sentinel:
                eq = active and head == m
                mask = cute.arch.vote_ballot_sync(eq)
                if lane == 0:
                    mUnionBlocks[work_idx, u] = m
                    mUnionMasks[work_idx, u] = mask
                if eq:
                    p += 1
                u += 1

        if lane == 0:
            mUnionCount[work_idx] = u
            mWorkMeta[work_idx, 0] = batch_idx
            mWorkMeta[work_idx, 1] = q_tile
            mWorkMeta[work_idx, 2] = h
