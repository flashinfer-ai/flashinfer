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

Slim sparse decode kernel for SM120/SM121

Decode shape: one query token (x its GQA query heads) attending one selected
128-token KV block per CTA. Compared to running decode through the prefill
``msa_sparse_attention`` kernel, this kernel:

- uses a dense static grid ``(topk, total_q, Hkv)`` driven directly by
  ``q2k_indices`` — no schedule tensors at all (split counts are computed
  in-kernel by the slot-0 CTA);
- uses an M=16 tile (the <=16 query heads of one token) instead of a 64-row
  tile that is mostly padding; multiple query tokens (seqlen_q>1, e.g. MTP) are
  handled as separate tiles;
- stages the KV block as two 64-token sub-blocks with online softmax,
  halving SMEM (~39KB) so two CTAs fit per SM (the prefill kernel's 85KB
  forces one CTA/SM, leaving decode latency-bound);
- loads with all 4 warps while warp 0 does the (tiny) MMA + softmax, so row
  reductions stay warp-local.

Partial outputs and log2-domain LSE go to the same split-slot buffers as the
prefill kernel and are reduced by the same fused combine kernel.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import warp


class SparseDecodeForwardSm12x:
    def __init__(
        self,
        head_dim: int = 128,
        blk_kv: int = 128,
        sub_block: int = 64,
        group_size: int = 1,
        topk: int = 16,
        num_threads: int = 128,
        is_causal: bool = True,
        paged: bool = False,
        kv_fp8: bool = False,
        kv_nvfp4: bool = False,
        q_fp8: bool = False,
        fused: bool = False,
    ):
        if head_dim != 128 or blk_kv != 128 or sub_block != 64:
            raise ValueError("only head_dim=blk_kv=128, sub_block=64 supported")
        if group_size > 16:
            raise ValueError("group_size must be <= 16")
        # E12 adaptive-split: when the per-token grid (total_q x num_kv_heads)
        # already fills the SMs, one CTA per (token, kv_head) loops every selected
        # block with in-kernel online softmax and writes the final output directly,
        # dropping the GMEM partials + the mandatory combine pass. nvfp4 KV keeps
        # the per-block split path (its decode is the bandwidth-bound case where the
        # combine is relatively cheap, and it keeps this kernel small).
        if fused and kv_nvfp4:
            raise ValueError("fused decode does not support nvfp4 KV")
        self._head_dim = head_dim
        self._blk_kv = blk_kv
        self._sub_block = sub_block
        self._n_sub = blk_kv // sub_block
        self._m_tile = 16
        self._group_size = group_size
        self._topk = topk
        self._num_threads = num_threads
        self._is_causal = is_causal
        self._paged = paged
        self._kv_fp8 = kv_fp8
        # NVFP4: packed e2m1 K/V (int32 word views) + e4m3 128x4 block scales
        self._kv_nvfp4 = kv_nvfp4
        self._q_fp8 = q_fp8
        self._fused = fused
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
        if cutlass.const_expr(self._fused):
            # one CTA per (token, kv_head): loops every selected block with online
            # softmax and writes the final output + LSE directly (no combine).
            self.kernel_fused(
                mQ,
                mK,
                mV,
                mPageTable,
                mQ2K,
                mOut,
                mLseOut,
                mCuK,
                mQOffset,
                softmax_scale_log2,
                out_scale,
                seqlen_q,
            ).launch(
                grid=(1, total_q, num_kv_heads),
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
            ).launch(
                grid=(self._topk, total_q, num_kv_heads),
                block=[self._num_threads, 1, 1],
                stream=stream,
            )

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
    ):
        tidx, _, _ = cute.arch.thread_idx()
        slot, qi, kv_head = cute.arch.block_idx()

        G = self._group_size
        kv_block = mQ2K[kv_head, qi, slot]

        # slot-0 CTA publishes the split count: the leading run of valid blocks.
        # combine reads slots [0, cnt) and a slot is written only when kv_block
        # >= 0, so a -1 hole must stop the count rather than skip past it.
        if slot == 0:
            if tidx == 0:
                cnt = cutlass.Int32(0)
                in_prefix = cutlass.Boolean(True)
                for t in cutlass.range_constexpr(self._topk):
                    if in_prefix and mQ2K[kv_head, qi, t] >= 0:
                        cnt = cnt + 1
                    else:
                        in_prefix = cutlass.Boolean(False)
                mSplitCounts[qi, kv_head] = cnt

        if kv_block >= 0:
            batch_idx = qi // seqlen_q
            tok_in_req = qi - batch_idx * seqlen_q
            k_start = mCuK[batch_idx]
            seqlen_k = mCuK[batch_idx + 1] - k_start

            # ///////////////////////////////////////////////////////////////
            # SMEM (plain padded layouts)
            # ///////////////////////////////////////////////////////////////
            sQ_layout = cute.make_layout(
                (self._m_tile, self._head_dim), stride=(self._pad_stride, 1)
            )
            sKV_layout = cute.make_layout(
                (self._sub_block, self._head_dim), stride=(self._pad_stride, 1)
            )

            @cute.struct
            class SharedStorage:
                sK: cute.struct.Align[
                    cute.struct.MemRange[
                        self._dtype, self._sub_block * self._pad_stride
                    ],
                    1024,
                ]
                sV: cute.struct.Align[
                    cute.struct.MemRange[
                        self._dtype, self._sub_block * self._pad_stride
                    ],
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
            # transpose view of V for the PV mma
            sVt = cute.composition(
                sV,
                cute.make_layout(
                    (self._head_dim, self._sub_block),
                    stride=(self._sub_block, 1),
                ),
            )

            # ///////////////////////////////////////////////////////////////
            # Load Q (G rows of one token; pad rows zero-filled)
            # ///////////////////////////////////////////////////////////////
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

            # ///////////////////////////////////////////////////////////////
            # MMA setup (single compute warp)
            # ///////////////////////////////////////////////////////////////
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

            # gmem K/V views for this block
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

            if cutlass.const_expr(self._is_causal):
                # causal: query global position = q_offset[b] + tok_in_req
                q_pos_limit = tok_in_req + mQOffset[batch_idx] + 1
                col_limit = cutlass.min(q_pos_limit, seqlen_k)
            else:
                col_limit = seqlen_k

            kv_chunks = self._sub_block * chunks_per_row
            kvfrag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
            if cutlass.const_expr(self._kv_nvfp4):
                # scale-row flattening per MSA quantize.py: paged caches are
                # quantized as (page, head, token) rows; flat as (token, head)
                if cutlass.const_expr(self._paged):
                    sf_row_base = (page * mK.shape[1] + kv_head) * self._blk_kv
                    sf_row_stride = cutlass.Int32(1)
                else:
                    sf_row_base = (k_start + kv_block * self._blk_kv) * mK.shape[
                        1
                    ] + kv_head
                    sf_row_stride = mK.shape[1]
                sf_tiles_n = (self._head_dim // 16 + 3) // 4
                pair_frag = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Uint32)
                pair_f16 = cute.make_tensor(
                    cute.recast_ptr(pair_frag.iterator, dtype=cutlass.Float16),
                    cute.make_layout(8),
                )

            # wait for Q stores from all warps, then preload Q fragments
            self.cta_sync_barrier.arrive_and_wait()
            if tidx < 32:
                for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                    cute.copy(
                        smem_tiled_copy_Q,
                        tSsQ[None, None, k],
                        tSrQ_copy_view[None, None, k],
                    )

            # ///////////////////////////////////////////////////////////////
            # Two 64-token sub-blocks with online softmax
            # ///////////////////////////////////////////////////////////////
            for half in cutlass.range_constexpr(self._n_sub):
                base = kv_block * self._blk_kv + half * self._sub_block
                # all threads load this sub-block (upconverting if fp8)
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
                            if cutlass.const_expr(self._kv_nvfp4):
                                from ...fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_fp4_helpers import (
                                    cvt_e4m3_to_f32_via_f16,
                                    fp4_decode_4bytes,
                                )

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
                                kp0, kp1, kp2, kp3 = fp4_decode_4bytes(
                                    cutlass.Uint32(k_word)
                                )
                                pair_frag[0] = kp0
                                pair_frag[1] = kp1
                                pair_frag[2] = kp2
                                pair_frag[3] = kp3
                                k_sc = cvt_e4m3_to_f32_via_f16(
                                    cutlass.Uint32(mKSf[sf_off])
                                )
                                kvfrag.store(
                                    (pair_f16.load().to(cutlass.Float32) * k_sc).to(
                                        self._dtype
                                    )
                                )
                                cute.autovec_copy(kvfrag, sK_chunk)
                                v_word = mV_blk[src_row, kv_c8]
                                vp0, vp1, vp2, vp3 = fp4_decode_4bytes(
                                    cutlass.Uint32(v_word)
                                )
                                pair_frag[0] = vp0
                                pair_frag[1] = vp1
                                pair_frag[2] = vp2
                                pair_frag[3] = vp3
                                v_sc = cvt_e4m3_to_f32_via_f16(
                                    cutlass.Uint32(mVSf[sf_off])
                                )
                                kvfrag.store(
                                    (pair_f16.load().to(cutlass.Float32) * v_sc).to(
                                        self._dtype
                                    )
                                )
                                cute.autovec_copy(kvfrag, sV_chunk)
                                gK_chunk = None
                                gV_chunk = None
                            else:
                                gK_chunk = cute.local_tile(
                                    mK_blk[src_row, None], (8,), (kv_c8,)
                                )
                                gV_chunk = cute.local_tile(
                                    mV_blk[src_row, None], (8,), (kv_c8,)
                                )
                            if cutlass.const_expr(self._kv_fp8 and not self._kv_nvfp4):
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
                            elif cutlass.const_expr(not self._kv_nvfp4):
                                cute.autovec_copy(gK_chunk, sK_chunk)
                                cute.autovec_copy(gV_chunk, sV_chunk)
                        else:
                            kvfrag.fill(0)
                            cute.autovec_copy(kvfrag, sK_chunk)
                            cute.autovec_copy(kvfrag, sV_chunk)
                self.cta_sync_barrier.arrive_and_wait()

                if tidx < 32:
                    # S = Q K^T for this sub-block
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

                    # mask + online softmax update
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
                # protect sK/sV before the next sub-block overwrites them
                self.cta_sync_barrier.arrive_and_wait()

            # ///////////////////////////////////////////////////////////////
            # Store normalized partial + log2-domain LSE
            # ///////////////////////////////////////////////////////////////
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
                            mOp[slot, qi, hq, d_pos] = (acc_O_mn[r, c] * inv).to(
                                mOp.element_type
                            )
                        mLse[slot, qi, hq] = (
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
        mQ2K: cute.Tensor,
        mOut: cute.Tensor,  # (total_q, Hq, d) final output
        mLseOut: cute.Tensor,  # (total_q, Hq) f32, natural-log LSE
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        out_scale: cutlass.Float32,
        seqlen_q: cutlass.Int32,
    ):
        # E12 fused decode: one CTA per (token, kv_head) attends *all* the token's
        # selected blocks with a single in-kernel online softmax (the same scheme
        # the split kernel runs over two 64-token sub-blocks, extended over the
        # whole selected-block list) and writes the final normalized output + LSE
        # directly. The union-tile prefill uses the identical dynamic-loop online
        # softmax; this is its decode analogue. Restricted to non-nvfp4 KV.
        tidx, _, _ = cute.arch.thread_idx()
        _, qi, kv_head = cute.arch.block_idx()  # grid x is 1 (no split slots)

        G = self._group_size
        LN2 = 0.6931471805599453  # 1 / log2(e); log2-domain LSE -> natural log

        batch_idx = qi // seqlen_q
        tok_in_req = qi - batch_idx * seqlen_q
        k_start = mCuK[batch_idx]
        seqlen_k = mCuK[batch_idx + 1] - k_start

        # number of valid (leading, contiguous) selected blocks: msa_topk_select
        # tail-pads with -1, so a -1 hole stops the count rather than skipping past.
        cnt = cutlass.Int32(0)
        in_prefix = cutlass.Boolean(True)
        for t in cutlass.range_constexpr(self._topk):
            if in_prefix and mQ2K[kv_head, qi, t] >= 0:
                cnt = cnt + 1
            else:
                in_prefix = cutlass.Boolean(False)

        # ///////////////////////////////////////////////////////////////
        # SMEM (plain padded layouts; same as the split kernel)
        # ///////////////////////////////////////////////////////////////
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

        # ///////////////////////////////////////////////////////////////
        # Load Q (G rows of one token; pad rows zero-filled)
        # ///////////////////////////////////////////////////////////////
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

        # ///////////////////////////////////////////////////////////////
        # MMA setup (single compute warp; same as split)
        # ///////////////////////////////////////////////////////////////
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
            q_pos_limit = tok_in_req + mQOffset[batch_idx] + 1
            col_limit = cutlass.min(q_pos_limit, seqlen_k)
        else:
            col_limit = seqlen_k

        kv_chunks = self._sub_block * chunks_per_row
        kvfrag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)

        # wait for Q stores from all warps, then preload Q fragments
        self.cta_sync_barrier.arrive_and_wait()
        if tidx < 32:
            for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, k],
                    tSrQ_copy_view[None, None, k],
                )

        # ///////////////////////////////////////////////////////////////
        # Loop over every selected block (online softmax carried across all
        # of them). cnt is thread-uniform, so the barriers stay collective.
        # ///////////////////////////////////////////////////////////////
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

            for half in cutlass.range_constexpr(self._n_sub):
                base = kv_block * self._blk_kv + half * self._sub_block
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
                # protect sK/sV before the next sub-block overwrites them
                self.cta_sync_barrier.arrive_and_wait()

        # ///////////////////////////////////////////////////////////////
        # Epilogue: normalize, write final output + natural-log LSE.
        # Always runs (cnt == 0 -> zero output, -inf LSE).
        # ///////////////////////////////////////////////////////////////
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
                    ) * LN2

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
