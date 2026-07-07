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

Sparse decode attention kernels for SM120/SM121: split-KV and fused paths,
plus the LSE-weighted combine kernel that merges the split partials.
"""

from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import warp

_LN2 = 0.6931471805599453  # 1 / log2(e): converts a log2-domain LSE to natural log
_FLT_MAX = 3.4028234663852886e38


class SparseDecodeForwardSm12x:
    """Sparse decode forward. Parallelism comes from splitting each token's
    selected KV blocks into chunks; the split path writes per-chunk partials
    for the combine kernel, the fused path (all blocks in one chunk) writes
    the output directly."""

    def __init__(
        self,
        head_dim: int = 128,
        blk_kv: int = 128,
        group_size: int = 1,
        topk: int = 16,
        num_threads: int = 128,
        is_causal: bool = True,
        paged: bool = False,
        kv_fp8: bool = False,
        kv_nvfp4: bool = False,
        q_fp8: bool = False,
        fused: bool = False,
        qoff_default: bool = False,
    ):
        if head_dim != 128 or blk_kv != 128:
            raise ValueError("only head_dim=blk_kv=128 supported")
        if group_size > 16:
            raise ValueError("group_size must be <= 16")
        self._head_dim = head_dim
        self._blk_kv = blk_kv
        # bf16/fp16 split path cp.async-pipelines the KV loads; 32-token sub-blocks
        # keep the double buffer within the 2-CTA/SM smem budget. fp8/nvfp4 convert
        # in-kernel (no cp.async) and fused keep the plain 64-token stream.
        self._pipeline = not fused and not kv_fp8 and not kv_nvfp4
        self._sub_block = 32 if self._pipeline else 64
        self._n_sub = blk_kv // self._sub_block
        self._m_tile = 16
        self._group_size = group_size
        self._topk = topk
        self._num_threads = num_threads
        self._is_causal = is_causal
        self._paged = paged
        self._kv_fp8 = kv_fp8
        self._kv_nvfp4 = kv_nvfp4
        self._q_fp8 = q_fp8
        self._fused = fused
        # Right-aligned decode (q_offset=None): the offset is seqlen_k - seqlen_q,
        # computed in-kernel so the wrapper launches no helper kernels to build it.
        self._qoff_default = qoff_default
        if kv_fp8 and kv_nvfp4:
            raise ValueError("kv_fp8 and kv_nvfp4 are mutually exclusive")
        self._pad_stride = head_dim + 8  # 16B-aligned padded rows

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, Hq, d)
        mK: cute.Tensor,  # (total_k, Hkv, d) flat | (pages, Hkv, 128, d) paged
        mV: cute.Tensor,
        mPageTable: cute.Tensor,  # (B, max_pages) int32 (dummy if flat)
        mKSf: cute.Tensor,  # nvfp4: K block scales, flat uint8 (dummy if unused)
        mVSf: cute.Tensor,  # nvfp4: V block scales
        mQ2K: cute.Tensor,  # (Hkv, total_q, topk) int32
        mOp: cute.Tensor,  # (topk, total_q, Hq, d) partials (dummy if fused)
        mLse: cute.Tensor,  # (topk, total_q, Hq) f32, log2 domain (dummy if fused)
        mSplitCounts: cute.Tensor,  # (total_q, Hkv) int32 (dummy if fused)
        mOut: cute.Tensor,  # (total_q, Hq, d) final output (dummy if not fused)
        mLseOut: cute.Tensor,  # (total_q, Hq) f32 natural-log LSE (dummy if not fused)
        mCuK: cute.Tensor,  # (B + 1,) int32
        mQOffset: cute.Tensor,  # (B,) int32 causal offset (MSA q_offset)
        softmax_scale: cutlass.Float32,
        out_scale: cutlass.Float32,  # fused: output value scale (v_global_scale)
        seqlen_q: cutlass.Int32,
        total_q: cutlass.Int32,
        num_kv_heads: cutlass.Int32,
        num_chunks: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(self._q_fp8):
            if cutlass.const_expr(mQ.element_type != cutlass.Float8E4M3FN):
                raise TypeError("q_fp8 requires Q to be Float8E4M3FN")
            self._dtype = cutlass.BFloat16
        else:
            if cutlass.const_expr(
                not (
                    mQ.element_type == cutlass.Float16
                    or mQ.element_type == cutlass.BFloat16
                )
            ):
                raise TypeError("Only Float16 or BFloat16 q is supported")
            self._dtype = mQ.element_type

        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        # total_q rides grid x (2^31-1 limit); grid y/z are capped at 65535.
        if cutlass.const_expr(self._fused):
            self.kernel_fused(
                mQ,
                mK,
                mV,
                mPageTable,
                mKSf,
                mVSf,
                mQ2K,
                mOut,
                mLseOut,
                mCuK,
                mQOffset,
                softmax_scale_log2,
                out_scale,
                seqlen_q,
            ).launch(
                grid=(total_q, 1, num_kv_heads),
                block=[self._num_threads, 1, 1],
                stream=stream,
            )
        else:
            self.kernel(
                mQ,
                mK,
                mV,
                mPageTable,
                mKSf,
                mVSf,
                mQ2K,
                mOp,
                mLse,
                mSplitCounts,
                mCuK,
                mQOffset,
                softmax_scale_log2,
                seqlen_q,
                num_chunks,
            ).launch(
                grid=(total_q, num_chunks, num_kv_heads),
                block=[self._num_threads, 1, 1],
                stream=stream,
            )

    @cute.jit
    def _load_kv_nvfp4_sub(
        self,
        mK_blk: cute.Tensor,
        mV_blk: cute.Tensor,
        mKSf: cute.Tensor,
        mVSf: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sf_row_base: cutlass.Int32,
        sf_row_stride: cutlass.Int32,
        row0: cutlass.Constexpr,
        base: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        """Dequant one sub-block of NVFP4 K/V into compute-dtype SMEM. mK_blk/mV_blk
        are (128, words) int32 views of one KV block (8 e2m1 per word); global scales
        fold into softmax_scale/out_scale. row0 is the sub-block's first row within
        the block view; base is its absolute token position (used only for the
        seqlen mask) -- for paged caches the two differ."""
        from ...fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_fp4_helpers import (
            cvt_e4m3_to_f32_via_f16,
            fp4_decode_4bytes,
        )

        chunks_per_row = self._head_dim // 8
        kv_chunks = self._sub_block * chunks_per_row
        sf_tiles_n = (self._head_dim // 16 + 3) // 4
        kvfrag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
        pair_frag = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
        pair_f16 = cute.make_tensor(
            cute.recast_ptr(pair_frag.iterator, dtype=cutlass.Float16),
            cute.make_layout(8),
        )
        for kv_it in cutlass.range_constexpr(
            cute.ceil_div(kv_chunks, self._num_threads)
        ):
            kv_chunk = tidx + kv_it * self._num_threads
            if kv_chunk < kv_chunks:
                kv_m = kv_chunk // chunks_per_row
                kv_c8 = kv_chunk % chunks_per_row
                sK_chunk = cute.local_tile(sK[kv_m, None], (8,), (kv_c8,))
                sV_chunk = cute.local_tile(sV[kv_m, None], (8,), (kv_c8,))
                src_row = row0 + kv_m
                if (base + kv_m) < seqlen_k:
                    # cuBLAS/cuDNN 128x4 tiled scale offset (an 8-elem chunk never
                    # crosses a 16-elem scale block)
                    srow = sf_row_base + src_row * sf_row_stride
                    scol = kv_c8 // 2
                    srow_in = srow % 128
                    sf_off = (
                        ((srow // 128) * sf_tiles_n + scol // 4) * 512
                        + (srow_in % 32) * 16
                        + (srow_in // 32) * 4
                        + scol % 4
                    )
                    k_word = mK_blk[src_row, kv_c8]
                    kp0, kp1, kp2, kp3 = fp4_decode_4bytes(cutlass.Uint32(k_word))
                    pair_frag[0] = kp0
                    pair_frag[1] = kp1
                    pair_frag[2] = kp2
                    pair_frag[3] = kp3
                    k_sc = cvt_e4m3_to_f32_via_f16(cutlass.Uint32(mKSf[sf_off]))
                    kvfrag.store(
                        (pair_f16.load().to(cutlass.Float32) * k_sc).to(self._dtype)
                    )
                    cute.autovec_copy(kvfrag, sK_chunk)
                    v_word = mV_blk[src_row, kv_c8]
                    vp0, vp1, vp2, vp3 = fp4_decode_4bytes(cutlass.Uint32(v_word))
                    pair_frag[0] = vp0
                    pair_frag[1] = vp1
                    pair_frag[2] = vp2
                    pair_frag[3] = vp3
                    v_sc = cvt_e4m3_to_f32_via_f16(cutlass.Uint32(mVSf[sf_off]))
                    kvfrag.store(
                        (pair_f16.load().to(cutlass.Float32) * v_sc).to(self._dtype)
                    )
                    cute.autovec_copy(kvfrag, sV_chunk)
                else:
                    kvfrag.fill(0)
                    cute.autovec_copy(kvfrag, sK_chunk)
                    cute.autovec_copy(kvfrag, sV_chunk)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPageTable: cute.Tensor,
        mKSf: cute.Tensor,
        mVSf: cute.Tensor,
        mQ2K: cute.Tensor,
        mOp: cute.Tensor,
        mLse: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        seqlen_q: cutlass.Int32,
        num_chunks: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qi, chunk_idx, kv_head = cute.arch.block_idx()

        G = self._group_size

        # Number of valid (leading, contiguous) selected blocks: msa_topk_select
        # tail-pads with -1, so a -1 hole stops the count rather than skipping past.
        cnt = cutlass.Int32(0)
        in_prefix = cutlass.Boolean(True)
        for t in cutlass.range_constexpr(self._topk):
            if in_prefix and mQ2K[kv_head, qi, t] >= 0:
                cnt = cnt + 1
            else:
                in_prefix = cutlass.Boolean(False)

        # num_chunks == topk degenerates to one block per CTA (per-block split).
        chunk_size = (self._topk + num_chunks - 1) // num_chunks
        chunk_start = chunk_idx * chunk_size
        chunk_end = cutlass.min(chunk_start + chunk_size, cnt)

        # The chunk-0 CTA publishes the active-chunk count that combine reduces over.
        if chunk_idx == 0:
            if tidx == 0:
                mSplitCounts[qi, kv_head] = (cnt + chunk_size - 1) // chunk_size

        if chunk_start < cnt:
            batch_idx = qi // seqlen_q
            tok_in_req = qi - batch_idx * seqlen_q
            k_start = mCuK[batch_idx]
            seqlen_k = mCuK[batch_idx + 1] - k_start

            n_buf = 2 if cutlass.const_expr(self._pipeline) else 1
            sQ_layout = cute.make_layout(
                (self._m_tile, self._head_dim), stride=(self._pad_stride, 1)
            )
            sKV_all_layout = cute.make_layout(
                (n_buf, self._sub_block, self._head_dim),
                stride=(self._sub_block * self._pad_stride, self._pad_stride, 1),
            )

            @cute.struct
            class SharedStorage:
                sK: cute.struct.Align[
                    cute.struct.MemRange[
                        self._dtype, n_buf * self._sub_block * self._pad_stride
                    ],
                    1024,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[
                        self._dtype, n_buf * self._sub_block * self._pad_stride
                    ],
                    1024,
                ]
                sQ: cute.struct.Align[
                    cute.struct.MemRange[self._dtype, self._m_tile * self._pad_stride],
                    1024,
                ]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sK_all = storage.sK.get_tensor(sKV_all_layout)
            sV_all = storage.sV.get_tensor(sKV_all_layout)
            sQ = storage.sQ.get_tensor(sQ_layout)

            sK = sK_all[0, None, None]
            sV = sV_all[0, None, None]
            sVt = cute.composition(
                sV,
                cute.make_layout(
                    (self._head_dim, self._sub_block), stride=(self._sub_block, 1)
                ),
            )

            # Load Q (G rows of one token; pad rows zero-filled)
            chunks_per_row = self._head_dim // 8
            q_chunks = self._m_tile * chunks_per_row
            qfrag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
            for q_it in cutlass.range_constexpr(
                cute.ceil_div(q_chunks, self._num_threads)
            ):
                q_chunk = tidx + q_it * self._num_threads
                if q_chunk < q_chunks:
                    q_m = q_chunk // chunks_per_row
                    q_c8 = q_chunk % chunks_per_row
                    s_chunk = cute.local_tile(sQ[q_m, None], (8,), (q_c8,))
                    if q_m < G:
                        g_row = mQ[qi, kv_head * G + q_m, None]
                        g_chunk = cute.local_tile(g_row, (8,), (q_c8,))
                        if cutlass.const_expr(self._q_fp8):
                            qfrag.store(
                                g_chunk.load()
                                .to(cutlass.Float16)
                                .to(cutlass.Float32)
                                .to(self._dtype)
                            )
                            cute.autovec_copy(qfrag, s_chunk)
                        else:
                            cute.autovec_copy(g_chunk, s_chunk)
                    else:
                        qfrag.fill(0)
                        cute.autovec_copy(qfrag, s_chunk)

            # MMA setup (single compute warp)
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
                (1, 1, 1),
                permutation_mnk=(16, 16, 16),
            )
            thr_mma = tiled_mma.get_slice(tidx % 32)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
            acc_S = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self._m_tile, self._sub_block)),
                cutlass.Float32,
            )
            acc_O = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self._m_tile, self._head_dim)),
                cutlass.Float32,
            )
            acc_O.fill(0.0)

            smem_copy_atom_QK = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
            )
            smem_copy_atom_V = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma)
            smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx % 32)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx % 32)
            smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx % 32)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
            tOsVt = smem_thr_copy_V.partition_S(sVt)
            tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

            if cutlass.const_expr(self._pipeline):
                cpasync_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(), self._dtype, num_bits_per_copy=128
                )

            cS = cute.make_identity_tensor((self._m_tile, self._sub_block))
            tScS_mn = self._make_acc_tensor_mn_view(thr_mma.partition_C(cS))
            cO = cute.make_identity_tensor((self._m_tile, self._head_dim))
            tScO_mn = self._make_acc_tensor_mn_view(thr_mma.partition_C(cO))
            acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            n_rows = cute.size(acc_O_mn.shape[0])

            row_max = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
            row_sum = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)

            if cutlass.const_expr(self._is_causal):
                if cutlass.const_expr(self._qoff_default):
                    q_pos_limit = tok_in_req + (seqlen_k - seqlen_q) + 1
                else:
                    q_pos_limit = tok_in_req + mQOffset[batch_idx] + 1
                col_limit = cutlass.min(q_pos_limit, seqlen_k)
            else:
                col_limit = seqlen_k

            kv_chunks = self._sub_block * chunks_per_row
            kvfrag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)

            self.cta_sync_barrier.arrive_and_wait()
            if tidx < 32:
                for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                    cute.copy(
                        smem_tiled_copy_Q,
                        tSsQ[None, None, k],
                        tSrQ_copy_view[None, None, k],
                    )

            if cutlass.const_expr(self._pipeline):
                # cp.async double-buffer: prefetch sub-block s while computing s-1.
                num_sub = (chunk_end - chunk_start) * self._n_sub
                for s in cutlass.range(num_sub + 1):
                    if s < num_sub:
                        ld_it = chunk_start + s // self._n_sub
                        ld_h = s % self._n_sub
                        ld_blk = mQ2K[kv_head, qi, ld_it]
                        if cutlass.const_expr(self._paged):
                            ld_page = mPageTable[batch_idx, ld_blk]
                            ld_mK = mK[ld_page, kv_head, None, None]
                            ld_mV = mV[ld_page, kv_head, None, None]
                        else:
                            ld_mK = cute.domain_offset(
                                (k_start + ld_blk * self._blk_kv, 0),
                                mK[None, kv_head, None],
                            )
                            ld_mV = cute.domain_offset(
                                (k_start + ld_blk * self._blk_kv, 0),
                                mV[None, kv_head, None],
                            )
                        ld_base = ld_blk * self._blk_kv + ld_h * self._sub_block
                        sK_b = sK_all[s % 2, None, None]
                        sV_b = sV_all[s % 2, None, None]
                        for kv_it in cutlass.range_constexpr(
                            cute.ceil_div(kv_chunks, self._num_threads)
                        ):
                            kv_chunk = tidx + kv_it * self._num_threads
                            if kv_chunk < kv_chunks:
                                kv_m = kv_chunk // chunks_per_row
                                kv_c8 = kv_chunk % chunks_per_row
                                sK_chunk = cute.local_tile(
                                    sK_b[kv_m, None], (8,), (kv_c8,)
                                )
                                sV_chunk = cute.local_tile(
                                    sV_b[kv_m, None], (8,), (kv_c8,)
                                )
                                src_row = ld_h * self._sub_block + kv_m
                                if (ld_base + kv_m) < seqlen_k:
                                    gK_chunk = cute.local_tile(
                                        ld_mK[src_row, None], (8,), (kv_c8,)
                                    )
                                    gV_chunk = cute.local_tile(
                                        ld_mV[src_row, None], (8,), (kv_c8,)
                                    )
                                    cute.copy(cpasync_atom, gK_chunk, sK_chunk)
                                    cute.copy(cpasync_atom, gV_chunk, sV_chunk)
                                else:
                                    kvfrag.fill(0)
                                    cute.autovec_copy(kvfrag, sK_chunk)
                                    cute.autovec_copy(kvfrag, sV_chunk)
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)
                    self.cta_sync_barrier.arrive_and_wait()

                    if s >= 1:
                        base = (
                            mQ2K[kv_head, qi, chunk_start + (s - 1) // self._n_sub]
                            * self._blk_kv
                            + ((s - 1) % self._n_sub) * self._sub_block
                        )
                        if tidx < 32:
                            sK_b = sK_all[(s - 1) % 2, None, None]
                            sVt_b = cute.composition(
                                sV_all[(s - 1) % 2, None, None],
                                cute.make_layout(
                                    (self._head_dim, self._sub_block),
                                    stride=(self._sub_block, 1),
                                ),
                            )
                            tSsK_b = smem_thr_copy_K.partition_S(sK_b)
                            tOsVt_b = smem_thr_copy_V.partition_S(sVt_b)
                            acc_S.fill(0.0)
                            for k in cutlass.range_constexpr(
                                cute.size(tSsK_b.shape[2])
                            ):
                                cute.copy(
                                    smem_tiled_copy_K,
                                    tSsK_b[None, None, k],
                                    tSrK_copy_view[None, None, k],
                                )
                            for k in cutlass.range_constexpr(
                                cute.size(tSsK_b.shape[2])
                            ):
                                cute.gemm(
                                    tiled_mma,
                                    acc_S,
                                    tSrQ[None, None, k],
                                    tSrK[None, None, k],
                                    acc_S,
                                )
                            for r in cutlass.range_constexpr(n_rows):
                                for c in cutlass.range_constexpr(
                                    cute.size(tScS_mn.shape[1])
                                ):
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
                                rmax_safe = (
                                    0.0 if rmax == -cutlass.Float32.inf else rmax
                                )
                                p_row = cute.math.exp2(
                                    acc_S_row * softmax_scale_log2
                                    - rmax_safe * softmax_scale_log2,
                                    fastmath=True,
                                )
                                rsum = p_row.reduce(
                                    cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                                )
                                prev_scale = cute.math.exp2(
                                    rmax_prev * softmax_scale_log2
                                    - rmax_safe * softmax_scale_log2,
                                    fastmath=True,
                                )
                                row_sum[r] = rsum + row_sum[r] * prev_scale
                                acc_O_mn[r, None] = (
                                    acc_O_mn[r, None].load() * prev_scale
                                )
                                row_max[r] = rmax
                                acc_S_mn[r, None] = p_row

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
                            for k in cutlass.range_constexpr(
                                cute.size(tOsVt_b.shape[2])
                            ):
                                cute.copy(
                                    smem_tiled_copy_V,
                                    tOsVt_b[None, None, k],
                                    tOrVt_copy_view[None, None, k],
                                )
                            for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                                cute.gemm(
                                    tiled_mma,
                                    acc_O,
                                    tOrS[None, None, k],
                                    tOrVt[None, None, k],
                                    acc_O,
                                )
                    self.cta_sync_barrier.arrive_and_wait()

            else:
                # cnt/chunk bounds are thread-uniform, so the per-sub-block
                # barriers stay collective.
                for it in cutlass.range(chunk_start, chunk_end):
                    kv_block = mQ2K[kv_head, qi, it]
                    if cutlass.const_expr(self._paged):
                        page = mPageTable[batch_idx, kv_block]
                        mK_blk = mK[page, kv_head, None, None]  # (128, d)
                        mV_blk = mV[page, kv_head, None, None]
                    else:
                        mK_blk = cute.domain_offset(
                            (k_start + kv_block * self._blk_kv, 0),
                            mK[None, kv_head, None],
                        )
                        mV_blk = cute.domain_offset(
                            (k_start + kv_block * self._blk_kv, 0),
                            mV[None, kv_head, None],
                        )
                    if cutlass.const_expr(self._kv_nvfp4):
                        # Scale-row flattening: paged caches are quantized as
                        # (page, head, token) rows; flat as (token, head).
                        if cutlass.const_expr(self._paged):
                            sf_row_base = (page * mK.shape[1] + kv_head) * self._blk_kv
                            sf_row_stride = cutlass.Int32(1)
                        else:
                            sf_row_base = (
                                k_start + kv_block * self._blk_kv
                            ) * mK.shape[1] + kv_head
                            sf_row_stride = mK.shape[1]

                    for half in cutlass.range_constexpr(self._n_sub):
                        base = kv_block * self._blk_kv + half * self._sub_block
                        if cutlass.const_expr(self._kv_nvfp4):
                            self._load_kv_nvfp4_sub(
                                mK_blk,
                                mV_blk,
                                mKSf,
                                mVSf,
                                sK,
                                sV,
                                sf_row_base,
                                sf_row_stride,
                                half * self._sub_block,
                                base,
                                seqlen_k,
                                tidx,
                            )
                        else:
                            for kv_it in cutlass.range_constexpr(
                                cute.ceil_div(kv_chunks, self._num_threads)
                            ):
                                kv_chunk = tidx + kv_it * self._num_threads
                                if kv_chunk < kv_chunks:
                                    kv_m = kv_chunk // chunks_per_row
                                    kv_c8 = kv_chunk % chunks_per_row
                                    sK_chunk = cute.local_tile(
                                        sK[kv_m, None], (8,), (kv_c8,)
                                    )
                                    sV_chunk = cute.local_tile(
                                        sV[kv_m, None], (8,), (kv_c8,)
                                    )
                                    src_row = half * self._sub_block + kv_m
                                    if (base + kv_m) < seqlen_k:
                                        gK_chunk = cute.local_tile(
                                            mK_blk[src_row, None], (8,), (kv_c8,)
                                        )
                                        gV_chunk = cute.local_tile(
                                            mV_blk[src_row, None], (8,), (kv_c8,)
                                        )
                                        if cutlass.const_expr(self._kv_fp8):
                                            kvfrag.store(
                                                gK_chunk.load()
                                                .to(cutlass.Float16)
                                                .to(cutlass.Float32)
                                                .to(self._dtype)
                                            )
                                            cute.autovec_copy(kvfrag, sK_chunk)
                                            kvfrag.store(
                                                gV_chunk.load()
                                                .to(cutlass.Float16)
                                                .to(cutlass.Float32)
                                                .to(self._dtype)
                                            )
                                            cute.autovec_copy(kvfrag, sV_chunk)
                                        else:
                                            cute.autovec_copy(gK_chunk, sK_chunk)
                                            cute.autovec_copy(gV_chunk, sV_chunk)
                                    else:
                                        kvfrag.fill(0)
                                        cute.autovec_copy(kvfrag, sK_chunk)
                                        cute.autovec_copy(kvfrag, sV_chunk)
                        self.cta_sync_barrier.arrive_and_wait()

                        if tidx < 32:
                            acc_S.fill(0.0)
                            for k in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                                cute.copy(
                                    smem_tiled_copy_K,
                                    tSsK[None, None, k],
                                    tSrK_copy_view[None, None, k],
                                )
                            for k in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                                cute.gemm(
                                    tiled_mma,
                                    acc_S,
                                    tSrQ[None, None, k],
                                    tSrK[None, None, k],
                                    acc_S,
                                )

                            for r in cutlass.range_constexpr(n_rows):
                                for c in cutlass.range_constexpr(
                                    cute.size(tScS_mn.shape[1])
                                ):
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
                                rmax_safe = (
                                    0.0 if rmax == -cutlass.Float32.inf else rmax
                                )
                                p_row = cute.math.exp2(
                                    acc_S_row * softmax_scale_log2
                                    - rmax_safe * softmax_scale_log2,
                                    fastmath=True,
                                )
                                rsum = p_row.reduce(
                                    cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                                )
                                prev_scale = cute.math.exp2(
                                    rmax_prev * softmax_scale_log2
                                    - rmax_safe * softmax_scale_log2,
                                    fastmath=True,
                                )
                                row_sum[r] = rsum + row_sum[r] * prev_scale
                                acc_O_mn[r, None] = (
                                    acc_O_mn[r, None].load() * prev_scale
                                )
                                row_max[r] = rmax
                                acc_S_mn[r, None] = p_row

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
                            for k in cutlass.range_constexpr(cute.size(tOsVt.shape[2])):
                                cute.copy(
                                    smem_tiled_copy_V,
                                    tOsVt[None, None, k],
                                    tOrVt_copy_view[None, None, k],
                                )
                            for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                                cute.gemm(
                                    tiled_mma,
                                    acc_O,
                                    tOrS[None, None, k],
                                    tOrVt[None, None, k],
                                    acc_O,
                                )
                        self.cta_sync_barrier.arrive_and_wait()
            # PDL: let the combine grid start its preamble now; its griddepcontrol
            # wait still orders its reads after this kernel's stores complete.
            cute.arch.griddepcontrol_launch_dependents()
            # Store normalized partial + log2-domain LSE for this chunk
            if tidx < 32:
                for r in cutlass.range_constexpr(n_rows):
                    g = tScO_mn[r, 0][0]
                    if g < G:
                        rs = self._threadquad_reduce_sum(row_sum[r])
                        rmax_final = row_max[r]
                        rmax_safe = (
                            0.0 if rmax_final == -cutlass.Float32.inf else rmax_final
                        )
                        inv = (
                            0.0 if (rs == 0.0 or rs != rs) else cute.arch.rcp_approx(rs)
                        )
                        hq = kv_head * G + g
                        for c in cutlass.range_constexpr(cute.size(tScO_mn.shape[1])):
                            d_pos = tScO_mn[0, c][1]
                            mOp[chunk_idx, qi, hq, d_pos] = (acc_O_mn[r, c] * inv).to(
                                mOp.element_type
                            )
                        mLse[chunk_idx, qi, hq] = (
                            rmax_safe * softmax_scale_log2
                            + cute.math.log2(rs, fastmath=True)
                        )

    @cute.kernel
    def kernel_fused(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mPageTable: cute.Tensor,
        mKSf: cute.Tensor,
        mVSf: cute.Tensor,
        mQ2K: cute.Tensor,
        mOut: cute.Tensor,  # (total_q, Hq, d) final output
        mLseOut: cute.Tensor,  # (total_q, Hq) f32, natural-log LSE
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        out_scale: cutlass.Float32,
        seqlen_q: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        qi, _, kv_head = cute.arch.block_idx()  # grid y is 1 (no split slots)

        G = self._group_size

        batch_idx = qi // seqlen_q
        tok_in_req = qi - batch_idx * seqlen_q
        k_start = mCuK[batch_idx]
        seqlen_k = mCuK[batch_idx + 1] - k_start

        # Valid-block count, as in the split kernel above.
        cnt = cutlass.Int32(0)
        in_prefix = cutlass.Boolean(True)
        for t in cutlass.range_constexpr(self._topk):
            if in_prefix and mQ2K[kv_head, qi, t] >= 0:
                cnt = cnt + 1
            else:
                in_prefix = cutlass.Boolean(False)

        sQ_layout = cute.make_layout(
            (self._m_tile, self._head_dim), stride=(self._pad_stride, 1)
        )
        sKV_layout = cute.make_layout(
            (self._sub_block, self._head_dim), stride=(self._pad_stride, 1)
        )

        @cute.struct
        class SharedStorage:
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._sub_block * self._pad_stride],
                1024,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._sub_block * self._pad_stride],
                1024,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self._m_tile * self._pad_stride],
                1024,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sVt = cute.composition(
            sV,
            cute.make_layout(
                (self._head_dim, self._sub_block), stride=(self._sub_block, 1)
            ),
        )

        # Load Q (G rows of one token; pad rows zero-filled)
        chunks_per_row = self._head_dim // 8
        q_chunks = self._m_tile * chunks_per_row
        qfrag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
        for q_it in cutlass.range_constexpr(cute.ceil_div(q_chunks, self._num_threads)):
            q_chunk = tidx + q_it * self._num_threads
            if q_chunk < q_chunks:
                q_m = q_chunk // chunks_per_row
                q_c8 = q_chunk % chunks_per_row
                s_chunk = cute.local_tile(sQ[q_m, None], (8,), (q_c8,))
                if q_m < G:
                    g_row = mQ[qi, kv_head * G + q_m, None]
                    g_chunk = cute.local_tile(g_row, (8,), (q_c8,))
                    if cutlass.const_expr(self._q_fp8):
                        qfrag.store(
                            g_chunk.load()
                            .to(cutlass.Float16)
                            .to(cutlass.Float32)
                            .to(self._dtype)
                        )
                        cute.autovec_copy(qfrag, s_chunk)
                    else:
                        cute.autovec_copy(g_chunk, s_chunk)
                else:
                    qfrag.fill(0)
                    cute.autovec_copy(qfrag, s_chunk)

        # MMA setup (single compute warp)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (1, 1, 1),
            permutation_mnk=(16, 16, 16),
        )
        thr_mma = tiled_mma.get_slice(tidx % 32)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_S = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_tile, self._sub_block)),
            cutlass.Float32,
        )
        acc_O = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self._m_tile, self._head_dim)),
            cutlass.Float32,
        )
        acc_O.fill(0.0)

        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
        )
        smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma)
        smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma)
        smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
        smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx % 32)
        smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx % 32)
        smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx % 32)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)
        tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

        cS = cute.make_identity_tensor((self._m_tile, self._sub_block))
        tScS_mn = self._make_acc_tensor_mn_view(thr_mma.partition_C(cS))
        cO = cute.make_identity_tensor((self._m_tile, self._head_dim))
        tScO_mn = self._make_acc_tensor_mn_view(thr_mma.partition_C(cO))
        acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        n_rows = cute.size(acc_O_mn.shape[0])

        row_max = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
        row_sum = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)

        if cutlass.const_expr(self._is_causal):
            if cutlass.const_expr(self._qoff_default):
                q_pos_limit = tok_in_req + (seqlen_k - seqlen_q) + 1
            else:
                q_pos_limit = tok_in_req + mQOffset[batch_idx] + 1
            col_limit = cutlass.min(q_pos_limit, seqlen_k)
        else:
            col_limit = seqlen_k

        kv_chunks = self._sub_block * chunks_per_row
        kvfrag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)

        self.cta_sync_barrier.arrive_and_wait()
        if tidx < 32:
            for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, k],
                    tSrQ_copy_view[None, None, k],
                )

        # cnt is thread-uniform, so the barriers stay collective.
        for it in cutlass.range(cnt):
            kv_block = mQ2K[kv_head, qi, it]
            if cutlass.const_expr(self._paged):
                page = mPageTable[batch_idx, kv_block]
                mK_blk = mK[page, kv_head, None, None]  # (128, d)
                mV_blk = mV[page, kv_head, None, None]
            else:
                mK_blk = cute.domain_offset(
                    (k_start + kv_block * self._blk_kv, 0), mK[None, kv_head, None]
                )
                mV_blk = cute.domain_offset(
                    (k_start + kv_block * self._blk_kv, 0), mV[None, kv_head, None]
                )
            if cutlass.const_expr(self._kv_nvfp4):
                # Scale-row flattening as in the split kernel above.
                if cutlass.const_expr(self._paged):
                    sf_row_base = (page * mK.shape[1] + kv_head) * self._blk_kv
                    sf_row_stride = cutlass.Int32(1)
                else:
                    sf_row_base = (k_start + kv_block * self._blk_kv) * mK.shape[
                        1
                    ] + kv_head
                    sf_row_stride = mK.shape[1]

            for half in cutlass.range_constexpr(self._n_sub):
                base = kv_block * self._blk_kv + half * self._sub_block
                if cutlass.const_expr(self._kv_nvfp4):
                    self._load_kv_nvfp4_sub(
                        mK_blk,
                        mV_blk,
                        mKSf,
                        mVSf,
                        sK,
                        sV,
                        sf_row_base,
                        sf_row_stride,
                        half * self._sub_block,
                        base,
                        seqlen_k,
                        tidx,
                    )
                else:
                    for kv_it in cutlass.range_constexpr(
                        cute.ceil_div(kv_chunks, self._num_threads)
                    ):
                        kv_chunk = tidx + kv_it * self._num_threads
                        if kv_chunk < kv_chunks:
                            kv_m = kv_chunk // chunks_per_row
                            kv_c8 = kv_chunk % chunks_per_row
                            sK_chunk = cute.local_tile(sK[kv_m, None], (8,), (kv_c8,))
                            sV_chunk = cute.local_tile(sV[kv_m, None], (8,), (kv_c8,))
                            src_row = half * self._sub_block + kv_m
                            if (base + kv_m) < seqlen_k:
                                gK_chunk = cute.local_tile(
                                    mK_blk[src_row, None], (8,), (kv_c8,)
                                )
                                gV_chunk = cute.local_tile(
                                    mV_blk[src_row, None], (8,), (kv_c8,)
                                )
                                if cutlass.const_expr(self._kv_fp8):
                                    kvfrag.store(
                                        gK_chunk.load()
                                        .to(cutlass.Float16)
                                        .to(cutlass.Float32)
                                        .to(self._dtype)
                                    )
                                    cute.autovec_copy(kvfrag, sK_chunk)
                                    kvfrag.store(
                                        gV_chunk.load()
                                        .to(cutlass.Float16)
                                        .to(cutlass.Float32)
                                        .to(self._dtype)
                                    )
                                    cute.autovec_copy(kvfrag, sV_chunk)
                                else:
                                    cute.autovec_copy(gK_chunk, sK_chunk)
                                    cute.autovec_copy(gV_chunk, sV_chunk)
                            else:
                                kvfrag.fill(0)
                                cute.autovec_copy(kvfrag, sK_chunk)
                                cute.autovec_copy(kvfrag, sV_chunk)
                self.cta_sync_barrier.arrive_and_wait()

                if tidx < 32:
                    acc_S.fill(0.0)
                    for k in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                        cute.copy(
                            smem_tiled_copy_K,
                            tSsK[None, None, k],
                            tSrK_copy_view[None, None, k],
                        )
                    for k in cutlass.range_constexpr(cute.size(tSsK.shape[2])):
                        cute.gemm(
                            tiled_mma,
                            acc_S,
                            tSrQ[None, None, k],
                            tSrK[None, None, k],
                            acc_S,
                        )

                    for r in cutlass.range_constexpr(n_rows):
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
                            acc_S_row * softmax_scale_log2
                            - rmax_safe * softmax_scale_log2,
                            fastmath=True,
                        )
                        rsum = p_row.reduce(
                            cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                        )
                        prev_scale = cute.math.exp2(
                            rmax_prev * softmax_scale_log2
                            - rmax_safe * softmax_scale_log2,
                            fastmath=True,
                        )
                        row_sum[r] = rsum + row_sum[r] * prev_scale
                        acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_scale
                        row_max[r] = rmax
                        acc_S_mn[r, None] = p_row

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
                    for k in cutlass.range_constexpr(cute.size(tOsVt.shape[2])):
                        cute.copy(
                            smem_tiled_copy_V,
                            tOsVt[None, None, k],
                            tOrVt_copy_view[None, None, k],
                        )
                    for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                        cute.gemm(
                            tiled_mma,
                            acc_O,
                            tOrS[None, None, k],
                            tOrVt[None, None, k],
                            acc_O,
                        )
                self.cta_sync_barrier.arrive_and_wait()

        # The epilogue always runs: cnt == 0 yields zero output and -inf LSE.
        if tidx < 32:
            for r in cutlass.range_constexpr(n_rows):
                g = tScO_mn[r, 0][0]
                if g < G:
                    rs = self._threadquad_reduce_sum(row_sum[r])
                    rmax_final = row_max[r]
                    rmax_safe = (
                        0.0 if rmax_final == -cutlass.Float32.inf else rmax_final
                    )
                    inv = 0.0 if (rs == 0.0 or rs != rs) else cute.arch.rcp_approx(rs)
                    hq = kv_head * G + g
                    for c in cutlass.range_constexpr(cute.size(tScO_mn.shape[1])):
                        d_pos = tScO_mn[0, c][1]
                        mOut[qi, hq, d_pos] = (acc_O_mn[r, c] * inv * out_scale).to(
                            mOut.element_type
                        )
                    mLseOut[qi, hq] = (
                        rmax_safe * softmax_scale_log2
                        + cute.math.log2(rs, fastmath=True)
                    ) * _LN2

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


class SparseCombineSm12x:
    """LSE-weighted merge of the split-decode partials into the final output."""

    def __init__(
        self,
        head_dim: int,
        topk: int,
        partial_is_fp8: bool,
        has_lse_out: bool,
        has_lse_t: bool,
        num_threads: int = 128,
    ):
        if head_dim % num_threads != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be a multiple of num_threads "
                f"({num_threads})"
            )
        self._head_dim = head_dim
        self._topk = topk
        self._partial_is_fp8 = partial_is_fp8
        self._has_lse_out = has_lse_out
        self._has_lse_t = has_lse_t
        self._num_threads = num_threads
        self._channels_per_thread = head_dim // num_threads

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,  # (topk, total_q, Hq, d)
        mLse2: cute.Tensor,  # (topk, total_q, Hq) f32, log2 domain
        mSplitCounts: cute.Tensor,  # (total_q, Hkv) int32
        mOut: cute.Tensor,  # (total_q, Hq, d)
        mLseOut: cute.Tensor,  # (total_q, Hq) f32 or dummy
        mLseT2: cute.Tensor,  # (topk, total_q, Hq) f32 or dummy
        mLseTOut: cute.Tensor,  # (total_q, Hq) f32 or dummy
        out_scale: cutlass.Float32,
        total_q: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        group_size: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self._partial_dtype: Type[cutlass.Numeric] = mO_partial.element_type
        self.kernel(
            mO_partial,
            mLse2,
            mSplitCounts,
            mOut,
            mLseOut,
            mLseT2,
            mLseTOut,
            out_scale,
            group_size,
        ).launch(
            grid=(total_q, num_qo_heads, 1),
            block=[self._num_threads, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLse2: cute.Tensor,
        mSplitCounts: cute.Tensor,
        mOut: cute.Tensor,
        mLseOut: cute.Tensor,
        mLseT2: cute.Tensor,
        mLseTOut: cute.Tensor,
        out_scale: cutlass.Float32,
        group_size: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        q, h, _ = cute.arch.block_idx()

        # PDL: this kernel launches while the forward is still running; wait for
        # its partial stores before reading them.
        cute.arch.griddepcontrol_wait()
        hkv = h // group_size
        count = mSplitCounts[q, hkv]
        if count > self._topk:
            count = cutlass.Int32(self._topk)

        lse_t_slots = self._topk if self._has_lse_t else 1

        @cute.struct
        class SharedStorage:
            s_lse: cute.struct.MemRange[cutlass.Float32, self._topk]
            s_lse_t: cute.struct.MemRange[cutlass.Float32, lse_t_slots]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        s_lse = storage.s_lse.get_tensor(cute.make_layout(self._topk))
        s_lse_t = storage.s_lse_t.get_tensor(cute.make_layout(lse_t_slots))

        neg_inf = -cutlass.Float32.inf

        # count <= 0 needs no special case: no slot passes s < count, so every
        # weight stays 0.
        for it in cutlass.range_constexpr(cute.ceil_div(self._topk, self._num_threads)):
            slot = tidx + it * self._num_threads
            if slot < self._topk:
                v = neg_inf
                if slot < count:
                    v = mLse2[slot, q, h]
                s_lse[slot] = v
                if cutlass.const_expr(self._has_lse_t):
                    vt = neg_inf
                    if slot < count:
                        vt = mLseT2[slot, q, h]
                    s_lse_t[slot] = vt
        cute.arch.sync_threads()

        # Every thread rebuilds the LSE weights redundantly: at low batch the grid
        # is a sub-wave, and building them once on one thread would serialize it.
        m = neg_inf
        for s in cutlass.range_constexpr(self._topk):
            if s < count:
                m = cute.arch.fmax(m, s_lse[s])
        m_finite = m > neg_inf
        w_frag = cute.make_rmem_tensor(cute.make_layout(self._topk), cutlass.Float32)
        denom = cutlass.Float32(0.0)
        for s in cutlass.range_constexpr(self._topk):
            w = cutlass.Float32(0.0)
            if s < count and m_finite:
                w = cute.math.exp2(s_lse[s] - m, fastmath=True)
            w_frag[s] = w
            denom += w
        inv = cutlass.Float32(0.0)
        if denom > 0.0:
            inv = cutlass.Float32(1.0) / denom

        # Branch-free: all topk slots load unconditionally (they pipeline); invalid
        # slots get weight 0, but 0 * NaN = NaN, so clamp the garbage to finite first
        # (fmax drops a NaN operand). cutlass-dsl 4.5.2 has no fmin: use -fmax(-x, -c).
        for i in cutlass.range_constexpr(self._channels_per_thread):
            c = tidx + i * self._num_threads
            acc = cutlass.Float32(0.0)
            for s in cutlass.range_constexpr(self._topk):
                e = mO_partial[s, q, h, c]
                if cutlass.const_expr(self._partial_is_fp8):
                    ef = e.to(cutlass.Float16).to(cutlass.Float32)
                else:
                    ef = e.to(cutlass.Float32)
                ef = cute.arch.fmax(-cute.arch.fmax(-ef, -_FLT_MAX), -_FLT_MAX)
                acc += w_frag[s] * ef
            mOut[q, h, c] = (acc * inv * out_scale).to(mOut.element_type)

        if cutlass.const_expr(self._has_lse_out):
            if tidx == 0:
                lse = neg_inf
                if denom > 0.0:
                    lse = (m + cute.math.log2(denom)) * _LN2
                mLseOut[q, h] = lse
        if cutlass.const_expr(self._has_lse_t):
            if tidx == 0:
                mt = neg_inf
                for s in cutlass.range_constexpr(self._topk):
                    if s < count:
                        mt = cute.arch.fmax(mt, s_lse_t[s])
                dt = cutlass.Float32(0.0)
                if mt > neg_inf:
                    for s in cutlass.range_constexpr(self._topk):
                        if s < count:
                            dt += cute.math.exp2(s_lse_t[s] - mt, fastmath=True)
                lse_t = neg_inf
                if dt > 0.0:
                    lse_t = (mt + cute.math.log2(dt)) * _LN2
                mLseTOut[q, h] = lse_t
