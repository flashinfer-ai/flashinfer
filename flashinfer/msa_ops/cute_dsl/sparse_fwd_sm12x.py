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

Minimax Sparse Attention forward kernel for SM120/SM121 (consumer Blackwell).

Q-major design (Phase 3a functional baseline): each CTA owns one tile of
``m_block_size`` query tokens for a single query head and iterates over the
KV blocks selected by any query in the tile (union of the tile's per-query
top-K lists).  Per-row membership masking inside the online softmax restores
exact per-query sparsity.

This differs from the SM100 MSA kernel (KV-major CSR + split-KV combine):
SM120/121 has no tcgen05 MMA, no TMEM, and ~99KB SMEM per CTA, so the
compute core is rebuilt on warp-level ``mma.sync`` (m16n8k16) with cp.async
loads, following the structure of CUTLASS's CuTe-DSL Ampere FA2 example.

Because each KV block is ``blk_kv=128`` *contiguous* tokens, K/V tile loads
remain ordinary cp.async tile copies at a dynamic block offset — only the
block *selection* is sparse, not the within-block access pattern.
"""

from types import SimpleNamespace
from typing import Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, warp


class SparseAttentionForwardSm12x:
    """Sparse attention forward (q-major, top-K KV block gather) for SM12x.

    Static configuration (compile-time): head_dim, m/n block sizes, topk,
    causal flag, number of threads, and the SMEM selection-map capacity.

    Dynamic (runtime): all tensor shapes, including total_q/total_k, batch
    size, head counts (GQA group size is derived as Hq // Hkv in-kernel).
    """

    def __init__(
        self,
        head_dim: int = 128,
        m_block_size: int = 64,
        n_block_size: int = 128,
        topk: int = 16,
        num_threads: int = 128,
        is_causal: bool = False,
        max_kv_blocks: int = 4096,
    ):
        self._head_dim = head_dim
        self._m_block_size = m_block_size
        self._n_block_size = n_block_size
        self._head_dim_padded = (head_dim + 31) // 32 * 32
        self._topk = topk
        self._num_threads = num_threads
        self._is_causal = is_causal
        # Capacity of the per-CTA KV-block selection map in SMEM.  Bounds the
        # number of KV blocks per sequence: max_seqlen_k <= max_kv_blocks * 128.
        self._max_kv_blocks = max_kv_blocks

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=num_threads
        )

    @staticmethod
    def can_implement(
        dtype, head_dim, m_block_size, n_block_size, topk, num_threads, max_kv_blocks
    ) -> bool:
        if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if (m_block_size * 2) % num_threads != 0:
            return False
        if n_block_size != 128:
            # KV block granularity of MSA selection (blk_kv) is 128.
            return False
        head_dim_padded = (head_dim + 31) // 32 * 32
        smem_usage = (
            (m_block_size + 2 * n_block_size) * head_dim_padded * 2  # Q, K, V
            + m_block_size * topk * 4  # selected-block indices per row
            + max_kv_blocks  # selection byte map
        )
        smem_capacity = cutlass.utils.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False
        return True

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, Hq, head_dim)
        mK: cute.Tensor,  # (total_k, Hkv, head_dim)
        mV: cute.Tensor,  # (total_k, Hkv, head_dim)
        mO: cute.Tensor,  # (total_q, Hq, head_dim)
        mIdx: cute.Tensor,  # (Hkv, total_q, topk) int32, ascending, -1 padded
        mCuQ: cute.Tensor,  # (B + 1,) int32
        mCuK: cute.Tensor,  # (B + 1,) int32
        softmax_scale: cutlass.Float32,
        max_seqlen_q: cutlass.Int32,
        batch_size: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(
            not (
                mQ.element_type == mK.element_type == mV.element_type == mO.element_type
            )
        ):
            raise TypeError("Q/K/V/O must have the same data type")
        if cutlass.const_expr(
            not (
                mQ.element_type == cutlass.Float16
                or mQ.element_type == cutlass.BFloat16
            )
        ):
            raise TypeError("Only Float16 or BFloat16 is supported")
        self._dtype: Type[cutlass.Numeric] = mQ.element_type

        # SMEM layouts (swizzled row-major atoms, as in the Ampere FA2 example)
        smem_k_block_size = 64 if self._head_dim_padded % 64 == 0 else 32
        swizzle_bits = 3 if smem_k_block_size == 64 else 2
        sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._m_block_size, self._head_dim_padded), (0, 1)
        )
        sKV_layout = cute.tile_to_shape(
            sQ_layout_atom, (self._n_block_size, self._head_dim_padded), (0, 1)
        )
        sO_layout = sQ_layout
        sIdx_layout = cute.make_layout(self._m_block_size * self._topk)
        sSel_layout = cute.make_layout(self._max_kv_blocks)

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sQ_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sKV_layout)], 1024
            ]
            sIdx: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self._m_block_size * self._topk], 16
            ]
            sSel: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, self._max_kv_blocks], 16
            ]

        # GMEM tiled copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self._dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self._dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tQKV_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        tQKV_layout = cute.make_layout(
            (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
            stride=(tQKV_shape_dim_1, 1),
        )
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(
            atom_async_copy, tQKV_layout, vQKV_layout
        )
        gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tQKV_layout, vQKV_layout
        )

        # Warp-level mma.sync tiled MMA (SM80-class tensor cores; valid on SM12x)
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
            (self._num_threads // 32, 1, 1),
            permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
        )

        # grid: (m_block, batch, qo_head)
        grid_dim = (
            cute.ceil_div(max_seqlen_q, self._m_block_size),
            batch_size,
            num_qo_heads,
        )
        LOG2_E = 1.4426950408889634074
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mIdx,
            mCuQ,
            mCuK,
            softmax_scale_log2,
            sQ_layout,
            sKV_layout,
            sO_layout,
            sIdx_layout,
            sSel_layout,
            gmem_tiled_copy_QKV,
            gmem_tiled_copy_O,
            tiled_mma,
            SharedStorage,
        ).launch(
            grid=grid_dim,
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
        mIdx: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sKV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sIdx_layout: cute.Layout,
        sSel_layout: cute.Layout,
        gmem_tiled_copy_QKV: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, batch_idx, qo_head = cute.arch.block_idx()

        # Per-sequence geometry
        q_start = mCuQ[batch_idx]
        seqlen_q = mCuQ[batch_idx + 1] - q_start
        k_start = mCuK[batch_idx]
        seqlen_k = mCuK[batch_idx + 1] - k_start
        num_kv_blocks = cute.ceil_div(seqlen_k, self._n_block_size)
        # GQA: query heads share the selection of their KV head
        group_size = mQ.shape[1] // mK.shape[1]
        kv_head = qo_head // group_size

        if m_block * self._m_block_size < seqlen_q:
            # ///////////////////////////////////////////////////////////////
            # Global tiles for this CTA (head-sliced, sequence-offset views)
            # ///////////////////////////////////////////////////////////////
            mQ_h = cute.domain_offset((q_start, 0), mQ[None, qo_head, None])
            mK_h = cute.domain_offset((k_start, 0), mK[None, kv_head, None])
            mV_h = cute.domain_offset((k_start, 0), mV[None, kv_head, None])
            mO_h = cute.domain_offset((q_start, 0), mO[None, qo_head, None])

            # (m_block_size, head_dim)
            gQ = cute.local_tile(
                mQ_h, (self._m_block_size, self._head_dim_padded), (m_block, 0)
            )
            # (n_block_size, head_dim, kv_block)
            gK = cute.local_tile(
                mK_h, (self._n_block_size, self._head_dim_padded), (None, 0)
            )
            gV = cute.local_tile(
                mV_h, (self._n_block_size, self._head_dim_padded), (None, 0)
            )

            # ///////////////////////////////////////////////////////////////
            # Shared memory
            # ///////////////////////////////////////////////////////////////
            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQ.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sKV_layout)
            sV = storage.sV.get_tensor(sKV_layout)
            sIdx = storage.sIdx.get_tensor(sIdx_layout)
            sSel = storage.sSel.get_tensor(sSel_layout)

            # V transposed view for the PV mma
            sVt = cute.composition(
                sV,
                cute.make_layout(
                    (self._head_dim_padded, self._n_block_size),
                    stride=(self._n_block_size, 1),
                ),
            )

            gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
            tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
            tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
            tKgK = gmem_thr_copy_QKV.partition_S(gK)
            tKsK = gmem_thr_copy_QKV.partition_D(sK)
            tVgV = gmem_thr_copy_QKV.partition_S(gV)
            tVsV = gmem_thr_copy_QKV.partition_D(sV)

            # ///////////////////////////////////////////////////////////////
            # MMA partitions
            # ///////////////////////////////////////////////////////////////
            thr_mma = tiled_mma.get_slice(tidx)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
            acc_shape_O = thr_mma.partition_shape_C(
                (self._m_block_size, self._head_dim_padded)
            )
            acc_O = cute.make_rmem_tensor(acc_shape_O, cutlass.Float32)
            acc_O.fill(0.0)
            acc_shape_S = thr_mma.partition_shape_C(
                (self._m_block_size, self._n_block_size)
            )
            acc_S = cute.make_rmem_tensor(acc_shape_S, cutlass.Float32)

            # ///////////////////////////////////////////////////////////////
            # SMEM -> RMEM copy atoms (ldmatrix)
            # ///////////////////////////////////////////////////////////////
            smem_copy_atom_QK = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
            )
            smem_copy_atom_V = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype
            )
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_QK, tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_QK, tiled_mma)
            smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)
            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
            tOsVt = smem_thr_copy_V.partition_S(sVt)
            tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

            # ///////////////////////////////////////////////////////////////
            # Predicates (local-tile identity tensors)
            # ///////////////////////////////////////////////////////////////
            cQ = cute.make_identity_tensor((self._m_block_size, self._head_dim_padded))
            cKV = cute.make_identity_tensor((self._n_block_size, self._head_dim_padded))
            tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
            tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
            # head-dim (column) predicates; rows are guarded dynamically
            tQpQ = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tQsQ.shape[0][1],
                        cute.size(tQsQ, mode=[1]),
                        cute.size(tQsQ, mode=[2]),
                    ),
                    stride=(cute.size(tQsQ, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            tKVpKV = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tKsK.shape[0][1],
                        cute.size(tKsK, mode=[1]),
                        cute.size(tKsK, mode=[2]),
                    ),
                    stride=(cute.size(tKsK, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
                for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                    tQpQ[rest_v, 0, rest_k] = cute.elem_less(
                        tQcQ[(0, rest_v), 0, rest_k][1], self._head_dim
                    )
            for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
                for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                    tKVpKV[rest_v, 0, rest_k] = cute.elem_less(
                        tKVcKV[(0, rest_v), 0, rest_k][1], self._head_dim
                    )

            # ///////////////////////////////////////////////////////////////
            # Prologue: async-load Q tile (row-guarded)
            # ///////////////////////////////////////////////////////////////
            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if (m_block * self._m_block_size + tQcQ[0, m, 0][0]) < seqlen_q:
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tQgQ[None, m, None],
                        tQsQ[None, m, None],
                        pred=tQpQ[None, m, None],
                    )
                else:
                    tQsQ[None, m, None].fill(0)
            cute.arch.cp_async_commit_group()

            # ///////////////////////////////////////////////////////////////
            # Selection metadata: load the tile's top-K lists into SMEM and
            # build the per-KV-block selection byte map.
            # ///////////////////////////////////////////////////////////////
            # zero the selection map
            for i in cutlass.range_constexpr(self._max_kv_blocks // self._num_threads):
                sSel[tidx + i * self._num_threads] = cutlass.Int8(0)
            self.cta_sync_barrier.arrive_and_wait()

            total_idx_elems = self._m_block_size * self._topk
            for i in cutlass.range_constexpr(
                cute.ceil_div(total_idx_elems, self._num_threads)
            ):
                e = tidx + i * self._num_threads
                if cutlass.const_expr(total_idx_elems % self._num_threads != 0):
                    if e >= total_idx_elems:
                        e = 0  # benign duplicate
                r = e // self._topk
                t = e % self._topk
                q_off = m_block * self._m_block_size + r
                v = cutlass.Int32(-1)
                if q_off < seqlen_q:
                    v = mIdx[kv_head, q_start + q_off, t]
                sIdx[e] = v
                if v >= 0:
                    if v < num_kv_blocks:
                        if v < self._max_kv_blocks:
                            sSel[v] = cutlass.Int8(1)

            # wait for Q tile; make sIdx/sSel visible CTA-wide
            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()

            # Preload all Q fragments (sQ is stable for the whole CTA lifetime)
            for k in cutlass.range_constexpr(cute.size(tSsQ.shape[2])):
                cute.copy(
                    smem_tiled_copy_Q,
                    tSsQ[None, None, k],
                    tSrQ_copy_view[None, None, k],
                )

            # ///////////////////////////////////////////////////////////////
            # Online softmax state
            # ///////////////////////////////////////////////////////////////
            row_max = cute.make_rmem_tensor(
                (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
            )
            row_sum = cute.make_rmem_tensor(
                (acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32
            )
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)

            # Identity over the local (m, n) tile for coordinate queries
            cS = cute.make_identity_tensor((self._m_block_size, self._n_block_size))
            tScS = thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)
            acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)

            mma_params = SimpleNamespace(
                thr_mma=thr_mma,
                tiled_mma=tiled_mma,
                tSrQ=tSrQ,
                tSrK=tSrK,
                tOrVt=tOrVt,
                acc_O=acc_O,
            )
            gmem_copy_params = SimpleNamespace(
                gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
                tKVcKV=tKVcKV,
                tKgK=tKgK,
                tKsK=tKsK,
                tVgV=tVgV,
                tVsV=tVsV,
                tKVpKV=tKVpKV,
            )
            smem_copy_params = SimpleNamespace(
                smem_tiled_copy_K=smem_tiled_copy_K,
                smem_tiled_copy_V=smem_tiled_copy_V,
                tSsK=tSsK,
                tSrK_copy_view=tSrK_copy_view,
                tOsVt=tOsVt,
                tOrVt_copy_view=tOrVt_copy_view,
            )

            # ///////////////////////////////////////////////////////////////
            # Main loop over this sequence's KV blocks, skipping unselected
            # ///////////////////////////////////////////////////////////////
            for kv_block in cutlass.range(num_kv_blocks):
                if sSel[kv_block] != 0:
                    self.process_kv_block(
                        kv_block,
                        m_block,
                        seqlen_q,
                        seqlen_k,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        acc_S,
                        acc_S_mn,
                        acc_O_mn,
                        tScS_mn,
                        sIdx,
                        row_max,
                        row_sum,
                        softmax_scale_log2,
                    )

            # ///////////////////////////////////////////////////////////////
            # Epilogue: normalize and store O
            # ///////////////////////////////////////////////////////////////
            self.normalize_softmax(acc_O, row_sum)
            rO = cute.make_fragment_like(acc_O, self._dtype)
            rO.store(acc_O.load().to(self._dtype))
            sO = cute.make_tensor(sQ.iterator, sO_layout)

            smem_copy_atom_O = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self._dtype
            )
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            # all threads have finished reading sQ fragments before the loop;
            # make sure no thread is still inside the last kv block's mma
            self.cta_sync_barrier.arrive_and_wait()
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

            gO = cute.local_tile(
                mO_h, (self._m_block_size, self._head_dim_padded), (m_block, 0)
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOrO = cute.make_fragment_like(tOgO, self._dtype)
            self.cta_sync_barrier.arrive_and_wait()
            cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

            cO = cute.make_identity_tensor((self._m_block_size, self._head_dim_padded))
            tOcO = gmem_thr_copy_O.partition_D(cO)
            tOpO = cute.make_rmem_tensor(
                cute.make_layout(
                    (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                    stride=(tOgO.shape[2], 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
                for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                    tOpO[rest_v, 0, rest_n] = cute.elem_less(
                        tOcO[(0, rest_v), 0, rest_n][1], self._head_dim
                    )
            for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
                if (m_block * self._m_block_size + tOcO[0, rest_m, 0][0]) < seqlen_q:
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None],
                    )

    @cute.jit
    def process_kv_block(
        self,
        kv_block: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        mma_params: SimpleNamespace,
        gmem_copy_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        acc_S: cute.Tensor,
        acc_S_mn: cute.Tensor,
        acc_O_mn: cute.Tensor,
        tScS_mn: cute.Tensor,
        sIdx: cute.Tensor,
        row_max: cute.Tensor,
        row_sum: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
    ):
        """Load one selected KV block, compute S = Q K^T, apply per-row
        membership + bounds (+ causal) masking, online softmax, O += P V."""
        # all threads must be done reading the previous block's K/V fragments
        self.cta_sync_barrier.arrive_and_wait()

        # K/V loads (row-guarded against the tail partial block)
        for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tKsK.shape[1])):
            if (
                kv_block * self._n_block_size + gmem_copy_params.tKVcKV[0, n, 0][0]
            ) < seqlen_k:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tKgK[None, n, None, kv_block],
                    gmem_copy_params.tKsK[None, n, None],
                    pred=gmem_copy_params.tKVpKV[None, n, None],
                )
            else:
                gmem_copy_params.tKsK[None, n, None].fill(0)
        for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
            if (
                kv_block * self._n_block_size + gmem_copy_params.tKVcKV[0, n, 0][0]
            ) < seqlen_k:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tVgV[None, n, None, kv_block],
                    gmem_copy_params.tVsV[None, n, None],
                    pred=gmem_copy_params.tKVpKV[None, n, None],
                )
            else:
                gmem_copy_params.tVsV[None, n, None].fill(0)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.cta_sync_barrier.arrive_and_wait()

        # S = Q K^T
        acc_S.fill(0.0)
        cute.copy(
            smem_copy_params.smem_tiled_copy_K,
            smem_copy_params.tSsK[None, None, 0],
            smem_copy_params.tSrK_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsK.shape[2])):
            k_next = (k + 1) % cute.size(smem_copy_params.tSsK.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, k_next],
                smem_copy_params.tSrK_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                acc_S,
                mma_params.tSrQ[None, None, k],
                mma_params.tSrK[None, None, k],
                acc_S,
            )

        # Masking + online softmax (uniform path; initial state row_max=-inf,
        # row_sum=0, acc_O=0 makes the "subsequent block" math exact for the
        # first selected block as well).
        for r in cutlass.range_constexpr(cute.size(row_max)):
            row_local = tScS_mn[r, 0][0]
            q_pos = m_block * self._m_block_size + row_local
            # membership: does this row select kv_block?
            selected = cutlass.Boolean(False)
            for t in cutlass.range_constexpr(self._topk):
                if sIdx[row_local * self._topk + t] == kv_block:
                    selected = cutlass.Boolean(True)
            if cutlass.const_expr(self._is_causal):
                # right-aligned causal: q_pos attends k positions
                # <= q_pos + (seqlen_k - seqlen_q)
                col_limit = q_pos + seqlen_k - seqlen_q + 1
                col_limit = cutlass.min(col_limit, seqlen_k)
            else:
                col_limit = seqlen_k
            if selected:
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    k_pos = kv_block * self._n_block_size + tScS_mn[0, c][1]
                    if cute.elem_less(col_limit, k_pos + 1):
                        acc_S_mn[r, c] = -cutlass.Float32.inf
            else:
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    acc_S_mn[r, c] = -cutlass.Float32.inf

            acc_S_row = acc_S_mn[r, None].load()
            row_max_cur = acc_S_row.reduce(
                cute.ReductionOp.MAX, -cutlass.Float32.inf, 0
            )
            row_max_cur = self._threadquad_reduce_max(row_max_cur)
            row_max_prev = row_max[r]
            row_max_cur = cute.arch.fmax(row_max_prev, row_max_cur)
            # guard against fully-masked rows: -inf - (-inf) would be NaN
            row_max_safe = 0.0 if row_max_cur == -cutlass.Float32.inf else row_max_cur

            acc_S_row_exp = cute.math.exp2(
                acc_S_row * softmax_scale_log2 - row_max_safe * softmax_scale_log2,
                fastmath=True,
            )
            acc_S_row_sum = acc_S_row_exp.reduce(
                cute.ReductionOp.ADD, cutlass.Float32.zero, 0
            )
            prev_scale = cute.math.exp2(
                row_max_prev * softmax_scale_log2 - row_max_safe * softmax_scale_log2,
                fastmath=True,
            )
            row_sum[r] = acc_S_row_sum + row_sum[r] * prev_scale
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_scale
            row_max[r] = row_max_cur
            acc_S_mn[r, None] = acc_S_row_exp

        # P (bf16/fp16) from softmaxed S
        rP = cute.make_fragment_like(acc_S, self._dtype)
        rP.store(acc_S.load().to(self._dtype))
        # reshape (4, MMA_M, MMA_N) -> ((4, 2), MMA_M, MMA_N / 2) for the PV mma
        rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
        rP_mma_view = cute.make_layout(
            (
                (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                rP_layout_divided.shape[1],
                rP_layout_divided.shape[2][1],
            ),
            stride=(
                (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                rP_layout_divided.stride[1],
                rP_layout_divided.stride[2][1],
            ),
        )
        tOrS = cute.make_tensor(rP.iterator, rP_mma_view)

        # O += P V
        cute.copy(
            smem_copy_params.smem_tiled_copy_V,
            smem_copy_params.tOsVt[None, None, 0],
            smem_copy_params.tOrVt_copy_view[None, None, 0],
        )
        for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
            k_next = (k + 1) % cute.size(tOrS.shape[2])
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, k_next],
                smem_copy_params.tOrVt_copy_view[None, None, k_next],
            )
            cute.gemm(
                mma_params.tiled_mma,
                mma_params.acc_O,
                tOrS[None, None, k],
                mma_params.tOrVt[None, None, k],
                mma_params.acc_O,
            )

    @cute.jit
    def normalize_softmax(self, acc_O: cute.Tensor, row_sum: cute.Tensor):
        acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
        for r in cutlass.range_constexpr(cute.size(row_sum)):
            row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
            # rows with no selected KV blocks have row_sum == 0 -> output 0
            row_sum_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
            scale = 1.0 if row_sum_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale

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

    def _threadquad_reduce(self, val, op):
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31),
        )
        val = op(
            val,
            cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31),
        )
        return val

    def _threadquad_reduce_max(self, val):
        return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

    def _threadquad_reduce_sum(self, val):
        return self._threadquad_reduce(val, lambda x, y: x + y)
