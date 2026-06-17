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

KV-major Minimax Sparse Attention forward kernel for SM120/SM121

This is the SM12x port of MSA's actual design: work is distributed over
(kv-head, KV block) CSR rows instead of query tiles. Each CTA processes one
flat work item ``{kv_head, row, q_begin, q_count, batch, kv_block}`` emitted
by ``build_k2q_csr_schedule``:

  1. Load the work item's single 128-token K/V block into SMEM **once**.
  2. Loop over tiles of the queries that selected this block (gathered via
     the CSR q-list; GQA query heads of each token are packed into the M
     dimension).
  3. For each tile: one-shot softmax (a work item covers exactly one KV
     block, so no online rescaling is needed) and write the normalized
     partial output and a log2-domain LSE into the query's split slot.

A separate combine step (LSE-weighted reduction over each query's split
slots) produces the final output.

Compute core: warp-level mma.sync (m16n8k16) + cp.async
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp


class SparseAttentionForwardKvMajorSm12x:
    """KV-major sparse attention forward for SM12x.

    Static configuration: head_dim (128), n_block (128 = blk_kv), m tile
    (64), GQA group size, causal flag. Everything else is dynamic.
    """

    def __init__(
        self,
        head_dim: int = 128,
        m_block_size: int = 64,
        n_block_size: int = 128,
        group_size: int = 1,
        num_threads: int = 128,
        is_causal: bool = False,
        paged: bool = False,
        kv_fp8: bool = False,
        kv_nvfp4: bool = False,
        return_temperature_lse: bool = False,
        q_fp8: bool = False,
        qk_fp8_mma: bool = False,
        pv_fp8_mma: bool = False,
    ):
        if head_dim != 128 or n_block_size != 128:
            raise ValueError("only head_dim == n_block_size == 128 is supported")
        if m_block_size % group_size != 0:
            raise ValueError("group_size must divide m_block_size")
        if kv_fp8 and kv_nvfp4:
            raise ValueError("kv_fp8 and kv_nvfp4 are mutually exclusive")
        self._paged = paged
        # fp8 (E4M3) K/V cache: upconverted to the compute dtype on load
        self._kv_fp8 = kv_fp8
        # NVFP4 K/V cache: packed E2M1 + E4M3 block scales (cuBLAS 128x4
        # tiled layout, 1 scale per 16 head-dim elements), dequantized to
        # the compute dtype on load. K/V tensors arrive as int32 views of
        # the packed bytes (last dim = head_dim/8 words).
        self._kv_nvfp4 = kv_nvfp4
        self._return_temperature_lse = return_temperature_lse
        # fp8 (E4M3) Q: upconverted to bf16 during the gather (route-B M1)
        self._q_fp8 = q_fp8
        # route-B M2: native fp8 QK MMA (m16n8k32, f32 acc). Q and K stay
        # e4m3 in SMEM; PV remains in the compute dtype (V upconverted).
        # Requires q_fp8 and fp8 K/V inputs.
        self._qk_fp8_mma = qk_fp8_mma
        if qk_fp8_mma and not q_fp8:
            raise ValueError("qk_fp8_mma requires q_fp8")
        # fp8 PV: P quantized to e4m3 (x448, compensated in the normalizer),
        # V kept e4m3 and staged PRE-TRANSPOSED in SMEM (no .b8 ldmatrix.trans
        # exists; see fa2-mla fp8-PV notes). Requires fp8 V input.
        self._pv_fp8_mma = pv_fp8_mma
        if pv_fp8_mma and kv_nvfp4:
            raise ValueError("pv_fp8_mma is not supported with NVFP4 KV")
        # fp8 operand rows padded to 16B alignment
        self._fp8_pad_stride = head_dim + 16
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._group_size = group_size
        self._tokens_per_tile = m_block_size // group_size
        self._num_threads = num_threads
        self._is_causal = is_causal
        # padded row stride (elements) for the gathered Q tile in SMEM:
        # +8 keeps 16B alignment of rows while spreading banks
        self._q_smem_stride = head_dim + 8

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    def _make_skv_layout(self):
        """K/V SMEM layout from compile-time config (callable from host or
        kernel trace context; the layout must be built inside the kernel
        region to satisfy IR region isolation)."""
        if self._qk_fp8_mma:
            # native fp8 K: rows must stay 16B-aligned for the fp8 ldmatrix
            return cute.make_layout(
                (self._n_block_size, self._head_dim),
                stride=(self._fp8_pad_stride, 1),
            )
        if self._kv_fp8 or self._kv_nvfp4:
            # fp8/nvfp4 K/V are dequantized elementwise into SMEM; the
            # convert path writes 8-element chunks at dynamic rows, so use a
            # plain padded layout instead of a swizzle
            return cute.make_layout(
                (self._n_block_size, self._head_dim),
                stride=(self._head_dim + 8, 1),
            )
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
        mK: cute.Tensor,  # (total_k, Hkv, d) flat | (num_pages, Hkv, 128, d) paged
        mV: cute.Tensor,  # same as mK
        mPageTable: cute.Tensor,  # (B, max_pages) int32 (dummy (1,1) if flat)
        mKSf: cute.Tensor,  # nvfp4: K block scales, flat uint8 128x4 storage
        mVSf: cute.Tensor,  # nvfp4: V block scales (dummy (1,) if unused)
        mOp: cute.Tensor,  # (topk, total_q, Hq, d) partial O
        mLse: cute.Tensor,  # (topk, total_q, Hq) f32, log2-domain LSE
        mLseT: cute.Tensor,  # (topk, total_q, Hq) f32 temperature LSE (dummy if off)
        mRowPtr: cute.Tensor,  # (Hkv, total_rows + 1) int32
        mQSplit: cute.Tensor,  # (Hkv, total_q * topk) int32
        mSched: cute.Tensor,  # (capacity, 6) int32
        mWorkCount: cute.Tensor,  # (1,) int32
        mCuQ: cute.Tensor,  # (B + 1,) int32
        mCuK: cute.Tensor,  # (B + 1,) int32
        mQOffset: cute.Tensor,  # (B,) int32 causal offset (MSA q_offset)
        softmax_scale: cutlass.Float32,
        lse_temperature_scale: cutlass.Float32,
        work_capacity: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(not (self._kv_fp8 or self._kv_nvfp4)):
            if cutlass.const_expr(
                not (mQ.element_type == mK.element_type == mV.element_type)
            ):
                raise TypeError("Q/K/V must have the same data type")
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
                raise TypeError("Only Float16 or BFloat16 is supported")
            self._dtype = mQ.element_type
        if cutlass.const_expr(self._kv_fp8):
            if cutlass.const_expr(mK.element_type != cutlass.Float8E4M3FN):
                raise TypeError("kv_fp8 requires K/V to be Float8E4M3FN")

        if cutlass.const_expr(self._kv_fp8 or self._kv_nvfp4):
            skv_cosize = self._n_block_size * (self._head_dim + 8)
        else:
            skv_cosize = cute.cosize(self._make_skv_layout())
        if cutlass.const_expr(self._qk_fp8_mma):
            # native fp8 QK: K kept as e4m3 (plain padded), Q gathered as e4m3
            sk_dtype = cutlass.Float8E4M3FN
            sq_dtype = cutlass.Float8E4M3FN
            sk_cosize = self._n_block_size * self._fp8_pad_stride
            sq_cosize = self._m_block_size * self._fp8_pad_stride
        else:
            sk_dtype = self._dtype
            sq_dtype = self._dtype
            sk_cosize = skv_cosize
            sq_cosize = self._m_block_size * self._q_smem_stride
        if cutlass.const_expr(self._pv_fp8_mma):
            # V stored transposed (d-major) in e4m3; P round-trip buffer
            sv_dtype = cutlass.Float8E4M3FN
            sv_cosize = self._head_dim * self._fp8_pad_stride
            sp_cosize = self._m_block_size * self._fp8_pad_stride
        else:
            sv_dtype = self._dtype
            sv_cosize = skv_cosize
            sp_cosize = 16  # dummy
        # Plain padded layout for the gathered Q tile (token-gather writes
        # don't go through a tile copy, so no swizzle)
        sQ_layout = cute.make_layout(
            (self._m_block_size, self._head_dim),
            stride=(
                self._fp8_pad_stride if self._qk_fp8_mma else self._q_smem_stride,
                1,
            ),
        )

        @cute.struct
        class SharedStorage:
            sK: cute.struct.Align[cute.struct.MemRange[sk_dtype, sk_cosize], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[sv_dtype, sv_cosize], 1024]
            sP: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, sp_cosize], 1024
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[sq_dtype, sq_cosize],
                1024,
            ]
            sQTok: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self._tokens_per_tile], 16
            ]

        # GMEM tiled copy for K/V tiles
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tKV_shape_dim_1 = 64 // async_copy_elems  # 64 = swizzle atom width
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
        if cutlass.const_expr(self._qk_fp8_mma):
            # the m16n8 accumulator layout matches the bf16 atom's, so the
            # softmax / P-reshape / PV machinery downstream is unchanged
            tiled_mma_qk = cute.make_tiled_mma(
                warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32, (16, 8, 32)),
                (self._num_threads // 32, 1, 1),
                permutation_mnk=(self._num_threads // 32 * 16, 16, 32),
            )
        else:
            tiled_mma_qk = tiled_mma
        if cutlass.const_expr(self._pv_fp8_mma):
            tiled_mma_pv = cute.make_tiled_mma(
                warp.MmaFP8Op(cutlass.Float8E4M3FN, cutlass.Float32, (16, 8, 32)),
                (self._num_threads // 32, 1, 1),
                permutation_mnk=(self._num_threads // 32 * 16, 16, 32),
            )
        else:
            tiled_mma_pv = tiled_mma

        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        # MSA semantics: exp2(T * (s - m) * scale * log2e)
        lse_temp_scale_log2 = softmax_scale_log2 * lse_temperature_scale
        self.kernel(
            mQ,
            mK,
            mV,
            mPageTable,
            mKSf,
            mVSf,
            mOp,
            mLse,
            mLseT,
            mRowPtr,
            mQSplit,
            mSched,
            mWorkCount,
            mCuQ,
            mCuK,
            mQOffset,
            softmax_scale_log2,
            lse_temp_scale_log2,
            sQ_layout,
            gmem_tiled_copy_KV,
            tiled_mma_pv,
            tiled_mma_qk,
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
        mPageTable: cute.Tensor,
        mKSf: cute.Tensor,
        mVSf: cute.Tensor,
        mOp: cute.Tensor,
        mLse: cute.Tensor,
        mLseT: cute.Tensor,
        mRowPtr: cute.Tensor,
        mQSplit: cute.Tensor,
        mSched: cute.Tensor,
        mWorkCount: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        lse_temp_scale_log2: cutlass.Float32,
        sQ_layout: cute.Layout,
        gmem_tiled_copy_KV: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tiled_mma_qk: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        work_idx, _, _ = cute.arch.block_idx()

        has_work = cutlass.Boolean(False)
        kv_head = cutlass.Int32(0)
        row = cutlass.Int32(0)
        q_begin = cutlass.Int32(0)
        q_count = cutlass.Int32(0)
        batch_idx = cutlass.Int32(0)
        kv_block = cutlass.Int32(0)
        if work_idx < mWorkCount[0]:
            kv_head = mSched[work_idx, 0]
            row = mSched[work_idx, 1]
            q_begin = mSched[work_idx, 2]
            q_count = mSched[work_idx, 3]
            batch_idx = mSched[work_idx, 4]
            kv_block = mSched[work_idx, 5]
            # q_count == 0 marks a padding item (static decode schedules)
            has_work = q_count > 0

        if has_work:
            q_start = mCuQ[batch_idx]
            k_start = mCuK[batch_idx]
            seqlen_k = mCuK[batch_idx + 1] - k_start
            row_base = mRowPtr[kv_head, row] + q_begin

            G = self._group_size
            tokens_per_tile = self._tokens_per_tile

            # ///////////////////////////////////////////////////////////////
            # Shared memory
            # ///////////////////////////////////////////////////////////////
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sK = storage.sK.get_tensor(self._make_skv_layout())
            if cutlass.const_expr(self._pv_fp8_mma):
                # V stored TRANSPOSED (d-major) so the PV B operand can be
                # fetched with the non-transposed fp8 ldmatrix idiom
                sV = storage.sV.get_tensor(
                    cute.make_layout(
                        (self._head_dim, self._n_block_size),
                        stride=(self._fp8_pad_stride, 1),
                    )
                )
                sP = storage.sP.get_tensor(
                    cute.make_layout(
                        (self._m_block_size, self._n_block_size),
                        stride=(self._fp8_pad_stride, 1),
                    )
                )
            elif cutlass.const_expr(self._qk_fp8_mma):
                # V keeps the dequant-path bf16 layout even when K stays fp8
                sV = storage.sV.get_tensor(
                    cute.make_layout(
                        (self._n_block_size, self._head_dim),
                        stride=(self._head_dim + 8, 1),
                    )
                )
            else:
                sV = storage.sV.get_tensor(self._make_skv_layout())
            sQ = storage.sQ.get_tensor(sQ_layout)
            sQTok = storage.sQTok.get_tensor(cute.make_layout(self._tokens_per_tile))
            if cutlass.const_expr(self._pv_fp8_mma):
                sVt = sV  # already stored (head_dim, n_block)
            else:
                sVt = cute.composition(
                    sV,
                    cute.make_layout(
                        (self._head_dim, self._n_block_size),
                        stride=(self._n_block_size, 1),
                    ),
                )

            # ///////////////////////////////////////////////////////////////
            # Load the work item's K/V block (once)
            # ///////////////////////////////////////////////////////////////
            if cutlass.const_expr(self._kv_nvfp4):
                from ...fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_fp4_helpers import (
                    cvt_e4m3_to_f32_via_f16,
                    fp4_decode_4bytes,
                )

                # NVFP4 K/V: packed e2m1 (one int32 word = 8 values) + e4m3
                # block scales (cuBLAS 128x4 tiled, 1 scale / 16 elements).
                # Dequant contract (MSA quantize.py):
                #   x = e2m1 * e4m3_block_scale * global_scale
                # K's global scale is folded into softmax_scale on the host;
                # V's is applied in the combine kernel.
                if cutlass.const_expr(self._paged):
                    page = mPageTable[batch_idx, kv_block]
                    mK_h4 = mK[page, kv_head, None, None]  # (128, d/8 words)
                    mV_h4 = mV[page, kv_head, None, None]
                    row_off4 = cutlass.Int32(0)
                    # scale rows flatten (page, head, token): see quantize.py
                    sf_row_base = (page * mK.shape[1] + kv_head) * self._n_block_size
                    sf_row_stride = cutlass.Int32(1)
                else:
                    mK_h4 = cute.domain_offset((k_start, 0), mK[None, kv_head, None])
                    mV_h4 = cute.domain_offset((k_start, 0), mV[None, kv_head, None])
                    row_off4 = kv_block * self._n_block_size
                    # scale rows flatten (token, head)
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
                for kv_it in cutlass.range_constexpr(
                    kv_total_chunks // self._num_threads
                ):
                    kv_chunk = tidx + kv_it * self._num_threads
                    kv_m = kv_chunk // kv_chunks_per_row
                    kv_c8 = kv_chunk % kv_chunks_per_row
                    sK_chunk = cute.local_tile(sK[kv_m, None], (8,), (kv_c8,))
                    sV_chunk = cute.local_tile(sV[kv_m, None], (8,), (kv_c8,))
                    if (kv_block * self._n_block_size + kv_m) < seqlen_k:
                        # cuBLAS/cuDNN 128x4 tiled scale offset (one 8-elem
                        # chunk never crosses a 16-elem scale block)
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
            elif cutlass.const_expr(self._kv_fp8):
                # fp8 K/V: vectorized load -> upconvert -> SMEM store
                if cutlass.const_expr(self._paged):
                    page = mPageTable[batch_idx, kv_block]
                    mK_h8 = mK[page, kv_head, None, None]
                    mV_h8 = mV[page, kv_head, None, None]
                    row_off = cutlass.Int32(0)
                else:
                    mK_h8 = cute.domain_offset((k_start, 0), mK[None, kv_head, None])
                    mV_h8 = cute.domain_offset((k_start, 0), mV[None, kv_head, None])
                    row_off = kv_block * self._n_block_size
                kv_chunks_per_row = self._head_dim // 8
                kv_total_chunks = self._n_block_size * kv_chunks_per_row
                cvt_frag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
                kz_frag = cute.make_rmem_tensor(
                    cute.make_layout(8), cutlass.Float8E4M3FN
                )
                # variable names here must not collide with those assigned
                # inside the dynamic tile loop below (the DSL would treat
                # them as loop-carried values)
                for kv_it in cutlass.range_constexpr(
                    kv_total_chunks // self._num_threads
                ):
                    kv_chunk = tidx + kv_it * self._num_threads
                    kv_m = kv_chunk // kv_chunks_per_row
                    kv_c8 = kv_chunk % kv_chunks_per_row
                    sK_chunk = cute.local_tile(sK[kv_m, None], (8,), (kv_c8,))
                    if cutlass.const_expr(not self._pv_fp8_mma):
                        sV_chunk = cute.local_tile(sV[kv_m, None], (8,), (kv_c8,))
                    if (kv_block * self._n_block_size + kv_m) < seqlen_k:
                        gKc = cute.local_tile(
                            mK_h8[row_off + kv_m, None], (8,), (kv_c8,)
                        )
                        gVc = cute.local_tile(
                            mV_h8[row_off + kv_m, None], (8,), (kv_c8,)
                        )
                        if cutlass.const_expr(self._qk_fp8_mma):
                            # native fp8 QK: K stays e4m3 in SMEM
                            cute.autovec_copy(gKc, sK_chunk)
                        else:
                            # e4m3 -> f16 is the natively supported cvt; route
                            # bf16 through f32 (no direct e4m3->bf16 cvt)
                            cvt_frag.store(
                                gKc.load()
                                .to(cutlass.Float16)
                                .to(cutlass.Float32)
                                .to(self._dtype)
                            )
                            cute.autovec_copy(cvt_frag, sK_chunk)
                        if cutlass.const_expr(self._pv_fp8_mma):
                            # V stays e4m3, written transposed: column kv_m,
                            # rows = this chunk's 8 head-dim elements
                            vvals = gVc.load()
                            for vj in cutlass.range_constexpr(8):
                                sV[kv_c8 * 8 + vj, kv_m] = vvals[vj]
                        else:
                            cvt_frag.store(
                                gVc.load()
                                .to(cutlass.Float16)
                                .to(cutlass.Float32)
                                .to(self._dtype)
                            )
                            cute.autovec_copy(cvt_frag, sV_chunk)
                    else:
                        cvt_frag.fill(0)
                        if cutlass.const_expr(self._qk_fp8_mma):
                            kz_frag.fill(0)
                            cute.autovec_copy(kz_frag, sK_chunk)
                        else:
                            cute.autovec_copy(cvt_frag, sK_chunk)
                        if cutlass.const_expr(self._pv_fp8_mma):
                            kz_frag.fill(0)
                            for vj in cutlass.range_constexpr(8):
                                sV[kv_c8 * 8 + vj, kv_m] = kz_frag[vj]
                        else:
                            cute.autovec_copy(cvt_frag, sV_chunk)
            else:
                gmem_thr_copy_KV = gmem_tiled_copy_KV.get_slice(tidx)
                tKsK = gmem_thr_copy_KV.partition_D(sK)
                tVsV = gmem_thr_copy_KV.partition_D(sV)
                cKV = cute.make_identity_tensor((self._n_block_size, self._head_dim))
                tKVcKV = gmem_thr_copy_KV.partition_S(cKV)

                if cutlass.const_expr(self._paged):
                    # paged: (num_pages, Hkv, 128, d) -> the block IS one page
                    page = mPageTable[batch_idx, kv_block]
                    mK_h = mK[page, kv_head, None, None]
                    mV_h = mV[page, kv_head, None, None]
                    blk_coord = cutlass.Int32(0)
                else:
                    mK_h = cute.domain_offset((k_start, 0), mK[None, kv_head, None])
                    mV_h = cute.domain_offset((k_start, 0), mV[None, kv_head, None])
                    blk_coord = kv_block
                # (n_block, d, num_blocks); block mode indexed after
                # partitioning so the copy tile shapes stay static
                gK = cute.local_tile(
                    mK_h, (self._n_block_size, self._head_dim), (None, 0)
                )
                gV = cute.local_tile(
                    mV_h, (self._n_block_size, self._head_dim), (None, 0)
                )
                tKgK = gmem_thr_copy_KV.partition_S(gK)
                tVgV = gmem_thr_copy_KV.partition_S(gV)

                for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                    if (kv_block * self._n_block_size + tKVcKV[0, n, 0][0]) < seqlen_k:
                        cute.copy(
                            gmem_tiled_copy_KV,
                            tKgK[None, n, None, blk_coord],
                            tKsK[None, n, None],
                        )
                        cute.copy(
                            gmem_tiled_copy_KV,
                            tVgV[None, n, None, blk_coord],
                            tVsV[None, n, None],
                        )
                    else:
                        tKsK[None, n, None].fill(0)
                        tVsV[None, n, None].fill(0)
                cute.arch.cp_async_commit_group()

            # ///////////////////////////////////////////////////////////////
            # MMA partitions and K/V fragment preload
            # ///////////////////////////////////////////////////////////////
            thr_mma = tiled_mma.get_slice(tidx)
            thr_mma_qk = tiled_mma_qk.get_slice(tidx)
            tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
            tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK))
            tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
            acc_shape_S = thr_mma_qk.partition_shape_C(
                (self._m_block_size, self._n_block_size)
            )
            acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)
            acc_shape_O = thr_mma.partition_shape_C(
                (self._m_block_size, self._head_dim)
            )
            acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)

            if cutlass.const_expr(self._pv_fp8_mma):
                # V is pre-transposed in SMEM: plain (non-transposed) fp8
                # ldmatrix fetches the B fragments (no .b8 ldmatrix.trans)
                smem_copy_atom_V = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    cutlass.Float8E4M3FN,
                )
            else:
                smem_copy_atom_V = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                    self._dtype,
                )
            smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
            smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
            tOsVt = smem_thr_copy_V.partition_S(sVt)
            tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)
            # the 16b ldmatrix atom composes with fp8 operands too (see
            # benchmarks empirical_sol MMAFP8TputKernel): same staging
            # structure for both QK dtypes, against tiled_mma_qk
            if cutlass.const_expr(self._qk_fp8_mma):
                qk_op_dtype = cutlass.Float8E4M3FN
            else:
                qk_op_dtype = self._dtype
            smem_copy_atom_QK = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                qk_op_dtype,
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma_qk)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma_qk)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)

            if cutlass.const_expr(not (self._kv_fp8 or self._kv_nvfp4)):
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

            # identity over the local (m, n) tile (head_dim == n_block, so the
            # same view serves the S mask and the O store coordinates)
            if cutlass.const_expr(self._pv_fp8_mma):
                smem_copy_atom_P = cute.make_copy_atom(
                    warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                    cutlass.Float8E4M3FN,
                )
                smem_tiled_copy_P = cute.make_tiled_copy_A(smem_copy_atom_P, tiled_mma)
                smem_thr_copy_P = smem_tiled_copy_P.get_slice(tidx)
                tOsP = smem_thr_copy_P.partition_S(sP)
                tOrP8 = thr_mma.make_fragment_A(thr_mma.partition_A(sP))
                tOrP_copy_view = smem_thr_copy_P.retile(tOrP8)

            cS = cute.make_identity_tensor((self._m_block_size, self._n_block_size))
            tScS = thr_mma_qk.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)
            acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            n_rows = cute.size(acc_O_mn.shape[0])

            if cutlass.const_expr(self._qk_fp8_mma):
                zero_chunk = cute.make_rmem_tensor(
                    cute.make_layout(8), cutlass.Float8E4M3FN
                )
            else:
                zero_chunk = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)
            zero_chunk.fill(0)
            q_cvt_frag = cute.make_rmem_tensor(cute.make_layout(8), self._dtype)

            # ///////////////////////////////////////////////////////////////
            # Loop over query tiles of this work item
            # ///////////////////////////////////////////////////////////////
            n_tiles = cute.ceil_div(q_count, tokens_per_tile)
            for tile in cutlass.range(n_tiles):
                # previous iteration's reads of sQ / sQTok must be complete
                self.cta_sync_barrier.arrive_and_wait()

                # stage this tile's packed (q_local | slot << 24) entries
                if tidx < tokens_per_tile:
                    e = tile * tokens_per_tile + tidx
                    v = cutlass.Int32(-1)
                    if e < q_count:
                        v = mQSplit[kv_head, row_base + e]
                    sQTok[tidx] = v
                self.cta_sync_barrier.arrive_and_wait()

                # gather Q rows: token i contributes G rows (its GQA heads)
                chunks_per_row = self._head_dim // 8
                total_chunks = self._m_block_size * chunks_per_row
                for it in cutlass.range_constexpr(total_chunks // self._num_threads):
                    chunk = tidx + it * self._num_threads
                    m = chunk // chunks_per_row
                    c8 = chunk % chunks_per_row
                    tok = m // G
                    g = m % G
                    v = sQTok[tok]
                    sQ_row = sQ[m, None]
                    sChunk = cute.local_tile(sQ_row, (8,), (c8,))
                    if v >= 0:
                        q_loc = v & cutlass.Int32(0x00FFFFFF)
                        gQ_row = mQ[q_start + q_loc, kv_head * G + g, None]
                        gChunk = cute.local_tile(gQ_row, (8,), (c8,))
                        if cutlass.const_expr(self._q_fp8 and not self._qk_fp8_mma):
                            q_cvt_frag.store(
                                gChunk.load()
                                .to(cutlass.Float16)
                                .to(cutlass.Float32)
                                .to(self._dtype)
                            )
                            cute.autovec_copy(q_cvt_frag, sChunk)
                        else:
                            # raw copy: bf16/fp16 Q, or native fp8-QK mode
                            cute.autovec_copy(gChunk, sChunk)
                    else:
                        cute.autovec_copy(zero_chunk, sChunk)
                self.cta_sync_barrier.arrive_and_wait()

                # Q fragments for this tile
                for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                    cute.copy(
                        smem_tiled_copy_Q,
                        tSsQ[None, None, k],
                        tSrQ_copy_view[None, None, k],
                    )

                # S = Q K^T
                acc_S.fill(0.0)
                for k in cutlass.range_constexpr(cute.size(tSrQ.shape[2])):
                    cute.gemm(
                        tiled_mma_qk,
                        acc_S,
                        tSrQ[None, None, k],
                        tSrK[None, None, k],
                        acc_S,
                    )

                # one-shot softmax with bounds (+ causal) masking
                row_sum = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
                row_lse = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
                row_lse_t = cute.make_rmem_tensor((n_rows,), cutlass.Float32)
                for r in cutlass.range_constexpr(n_rows):
                    m_local = tScS_mn[r, 0][0]
                    tok = m_local // G
                    v = sQTok[tok]
                    q_loc = v & cutlass.Int32(0x00FFFFFF)
                    if cutlass.const_expr(self._is_causal):
                        # causal: query global position = q_offset[b] + q_loc
                        # (default offset seqlen_k - seqlen_q = right-aligned)
                        col_limit = q_loc + mQOffset[batch_idx] + 1
                        col_limit = cutlass.min(col_limit, seqlen_k)
                    else:
                        col_limit = seqlen_k
                    for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                        k_pos = kv_block * self._n_block_size + tScS_mn[0, c][1]
                        if cute.elem_less(col_limit, k_pos + 1):
                            acc_S_mn[r, c] = -cutlass.Float32.inf

                    acc_S_row = acc_S_mn[r, None].load()
                    row_max = acc_S_row.reduce(
                        cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
                    )
                    row_max = self._threadquad_reduce_max(row_max)
                    row_max_safe = 0.0 if row_max == -cutlass.Float32.inf else row_max
                    acc_S_row_exp = cute.math.exp2(
                        acc_S_row * softmax_scale_log2
                        - row_max_safe * softmax_scale_log2,
                        fastmath=True,
                    )
                    rs = acc_S_row_exp.reduce(
                        cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                    )
                    rs = self._threadquad_reduce_sum(rs)
                    row_sum[r] = rs
                    # log2-domain LSE: lse2 = m * scale * log2(e) + log2(sum)
                    row_lse[r] = row_max_safe * softmax_scale_log2 + cute.math.log2(
                        rs, fastmath=True
                    )
                    if cutlass.const_expr(self._return_temperature_lse):
                        # MSA temperature LSE: exponent scaled by T
                        p_row_t = cute.math.exp2(
                            acc_S_row * lse_temp_scale_log2
                            - row_max_safe * lse_temp_scale_log2,
                            fastmath=True,
                        )
                        rs_t = p_row_t.reduce(
                            cute.ReductionOp.ADD, cutlass.Float32.zero, 0
                        )
                        rs_t = self._threadquad_reduce_sum(rs_t)
                        row_lse_t[r] = (
                            row_max_safe * lse_temp_scale_log2
                            + cute.math.log2(rs_t, fastmath=True)
                        )
                    acc_S_mn[r, None] = acc_S_row_exp

                # O_tile = P V
                acc_O.fill(0.0)
                if cutlass.const_expr(self._pv_fp8_mma):
                    # Round-trip P through SMEM as e4m3 (x448; the C->A register
                    # identity from the bf16 path does not hold for the fp8 k=32
                    # atom; 1/448 is folded into the row normalizer at store). A
                    # register-direct variant was tried and measured slower.
                    for r in cutlass.range_constexpr(n_rows):
                        p_row_coord = tScS_mn[r, 0][0]
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            sP[p_row_coord, tScS_mn[0, c][1]] = (
                                acc_S_mn[r, c] * 448.0
                            ).to(cutlass.Float8E4M3FN)
                    self.cta_sync_barrier.arrive_and_wait()
                    for k in cutlass.range_constexpr(cute.size(tOsP.shape[2])):
                        cute.copy(
                            smem_tiled_copy_P,
                            tOsP[None, None, k],
                            tOrP_copy_view[None, None, k],
                        )
                    for k in cutlass.range_constexpr(cute.size(tOrP8.shape[2])):
                        cute.gemm(
                            tiled_mma,
                            acc_O,
                            tOrP8[None, None, k],
                            tOrVt[None, None, k],
                            acc_O,
                        )
                    # sP is overwritten next tile; fragment reads done above,
                    # and the loop-top barrier orders the next tile's writes
                else:
                    rP = cute.make_fragment_like(acc_S, self._dtype)
                    rP.store(acc_S.load().to(self._dtype))
                    rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
                    rP_mma_view = cute.make_layout(
                        (
                            (
                                rP_layout_divided.shape[0],
                                rP_layout_divided.shape[2][0],
                            ),
                            rP_layout_divided.shape[1],
                            rP_layout_divided.shape[2][1],
                        ),
                        stride=(
                            (
                                rP_layout_divided.stride[0],
                                rP_layout_divided.stride[2][0],
                            ),
                            rP_layout_divided.stride[1],
                            rP_layout_divided.stride[2][1],
                        ),
                    )
                    tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
                    for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                        cute.gemm(
                            tiled_mma,
                            acc_O,
                            tOrS[None, None, k],
                            tOrVt[None, None, k],
                            acc_O,
                        )

                # store normalized partials + lse at each query's split slot
                for r in cutlass.range_constexpr(n_rows):
                    m_local = tScS_mn[r, 0][0]
                    tok = m_local // G
                    g = m_local % G
                    v = sQTok[tok]
                    if v >= 0:
                        q_loc = v & cutlass.Int32(0x00FFFFFF)
                        slot = (v >> cutlass.Int32(24)) & cutlass.Int32(0xFF)
                        q_global = q_start + q_loc
                        hq = kv_head * G + g
                        rs = row_sum[r]
                        inv = (
                            0.0 if (rs == 0.0 or rs != rs) else cute.arch.rcp_approx(rs)
                        )
                        if cutlass.const_expr(self._pv_fp8_mma):
                            inv = inv * cutlass.Float32(1.0 / 448.0)
                        for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                            d_pos = tScS_mn[0, c][1]
                            mOp[slot, q_global, hq, d_pos] = (acc_O_mn[r, c] * inv).to(
                                mOp.element_type
                            )
                        mLse[slot, q_global, hq] = row_lse[r]
                        if cutlass.const_expr(self._return_temperature_lse):
                            mLseT[slot, q_global, hq] = row_lse_t[r]

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
