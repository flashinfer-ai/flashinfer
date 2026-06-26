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

NVFP4 MSA proxy-score kernel for SM120/SM121: the FP4 counterpart of
:mod:`proxy_score_sm12x` (porting MSA's ``fp4_indexer_block_scores``), reading the
index K at ~4 bits/elem. Same contract as the bf16 proxy (per-block max of the
unscaled post-mask QK^T, -inf for invalid blocks); inputs are pre-quantized (packed
e2m1 + e4m3 per-16 block scales + per-tensor fp32 global scales).

QK^T runs on the SM120 fp4 tensor cores (``MmaMXF4NVF4Op``); the SF smem layout and
mainloop are ported from ``dense_blockscaled_gemm_sm120_b12x`` with the proxy's own
block-max epilogue. Two schedules (flat + paged K, split-K): the general per-(q-tile,
batch, head) :class:`MsaProxyScoreFp4MmaSm12x` and a 16-head q_len<=8
:class:`MsaProxyScoreFp4MmaDecodePackedSm12x`.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm120_utils
from cutlass import Uint8, Uint32
from cutlass.cute.nvgpu import warp
from cutlass.cute.nvgpu.warp.mma import Field as WarpField

from flashinfer.cute_dsl.utils import (
    sm120_make_smem_layout_sfa,
    sm120_make_smem_layout_sfb,
)

# NVFP4: 16-element scale groups (head_dim 128 -> 8 groups/row).
_SF_VEC_SIZE = 16

# Launch bound: cap registers so >=2 CTAs/SM stay resident. 2 is optimal across the
# measured shapes (mb=1 loses on latency-bound shapes; mb=3 spills) -- see the
# register/occupancy analysis in MsaProxyScoreFp4MmaSm12x.kernel. The split-K host
# heuristic in proxy_score.py assumes this same value (target = 2 * num_sms).
_MIN_BLOCKS_PER_MP = 2


class MsaProxyScoreFp4MmaSm12x:
    """Warp-level NVFP4 tensor-core MMA proxy: the general per-(q-tile, batch, head)
    schedule.

    Fixed M=128 q-tile, head_dim == kv_block == 128. Handles flat (cu_seqlens) and
    paged K via ``paged`` (the only difference is a ``mPageTable[batch, kv_block]``
    lookup). The 16-head packed-decode schedule is its own subclass
    (:class:`MsaProxyScoreFp4MmaDecodePackedSm12x`).
    """

    _M = 128  # q-tile rows (== one cuBLAS 128-row SF block, so sfa_tiles_per_block==1)
    _N = 128  # kv-block
    _NUM_THREADS = 256  # 8 MMA warps for the (4,2,1) atom

    def __init__(
        self, head_dim: int = 128, is_causal: bool = True, paged: bool = False
    ):
        if head_dim != 128:
            raise ValueError("only head_dim == 128 is supported")
        self._head_dim = head_dim
        self._is_causal = is_causal
        self._paged = paged
        self._num_threads = self._NUM_THREADS
        self.sf_vec_size = _SF_VEC_SIZE
        self.acc_dtype = cutlass.Float32
        self.a_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.tile_shape_mnk = (self._M, self._N, head_dim)
        self._sf_per_row = head_dim // _SF_VEC_SIZE  # 8
        self._sf_tiles_n = (self._sf_per_row + 3) // 4  # 2
        self._chunks_u32 = head_dim // 8  # 16 Uint32 / packed-fp4 row
        # atom (4,2,1) over the m16n8k64 fp4 MMA -> group m64 n16 k64.
        self._atom_shape = (4, 2, 1)
        self.num_m_tiles = self._M // (16 * self._atom_shape[0])
        self.num_n_tiles = self._N // (8 * self._atom_shape[1])
        self.num_k_blocks = head_dim // 64
        self._min_blocks = _MIN_BLOCKS_PER_MP

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self._num_threads
        )

    @cute.jit
    def _build_mma(self):
        """fp4 tiled MMA + bare atom + SF smem layouts. Built inside the jit
        context (``make_layout`` / ``get_permutation_mnk`` need an MLIR context)."""
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype, self.acc_dtype, self.sf_dtype
        )
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(self._atom_shape),
            permutation_mnk=permutation_mnk,
        )
        mma_atom = cute.make_mma_atom(mma_op)
        sfa_layout = sm120_make_smem_layout_sfa(
            tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1
        )
        sfb_layout = sm120_make_smem_layout_sfb(
            tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1
        )
        return tiled_mma, mma_atom, sfa_layout, sfb_layout

    def _ab_layout(self, mode_n: int):
        # plain K-major fp4 (mode_n, head_dim).
        return cute.make_layout((mode_n, self._head_dim), stride=(self._head_dim, 1))

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, Hq, head_dim//2) packed e2m1, uint8
        mK: cute.Tensor,  # flat (total_k,Hkv,hd/2) | paged (pages,Hkv,128,hd/2) u8
        mQsf: cute.Tensor,  # uint8 e4m3 scales, cuBLAS 128x4 tiled
        mKsf: cute.Tensor,  # uint8 e4m3 scales, cuBLAS 128x4 tiled
        mPageTable: cute.Tensor,  # (B, max_pages) int32 paged; dummy (1,1) flat
        mMaxScore: cute.Tensor,  # (Hq, max_k_tiles, total_q) f32
        mCuQ: cute.Tensor,  # (B + 1,) int32
        mCuK: cute.Tensor,  # (B + 1,) int32
        mQOffset: cute.Tensor,  # (B,) int32 causal offset
        q_global_scale: cutlass.Float32,
        k_global_scale: cutlass.Float32,
        max_seqlen_q: cutlass.Int32,
        batch_size: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        max_k_tiles: cutlass.Int32,
        num_splits: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # split-K: fold a kv-block split factor into grid x. At long context with
        # low batch the base grid (1, batch, heads) is < #SMs (e.g. B8 q1 -> 64
        # CTAs on 84 SMs) so the GPU starves; splitting each sequence's kv-block
        # range across CTAs fills it. The proxy output is per-(head, kv_block,
        # query) so the splits write disjoint columns and need no reduction.
        # The host passes num_splits==1 when the base grid already fills the SMs.
        self.kernel(
            mQ,
            mK,
            mQsf,
            mKsf,
            mPageTable,
            mMaxScore,
            mCuQ,
            mCuK,
            mQOffset,
            q_global_scale * k_global_scale,
            max_k_tiles,
            num_splits,
        ).launch(
            grid=(
                cute.ceil_div(max_seqlen_q, self._M) * num_splits,
                batch_size,
                num_qo_heads,
            ),
            block=[self._num_threads, 1, 1],
            stream=stream,
            # cap registers so >=2 CTAs/SM fit (smem is only ~19KB; the warp
            # epilogue pushed regs to 168 -> reg-limited to 1 CTA otherwise).
            min_blocks_per_mp=self._min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mQsf: cute.Tensor,
        mKsf: cute.Tensor,
        mPageTable: cute.Tensor,
        mMaxScore: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
        global_scale: cutlass.Float32,  # q_global * k_global, applied at store
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
        num_kv_blocks = cute.ceil_div(seqlen_k, self._N)
        group_size = mQ.shape[1] // mK.shape[1]
        kv_head = qo_head // group_size

        if m_block * self._M < seqlen_q:
            tiled_mma, mma_atom, sfa_layout, sfb_layout = self._build_mma()
            sA_layout = self._ab_layout(self._M)
            sB_layout = self._ab_layout(self._N)

            @cute.struct
            class SharedStorage:
                sA: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, cute.cosize(sA_layout)], 1024
                ]
                sB: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, cute.cosize(sB_layout)], 1024
                ]
                sSFA: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_layout)], 1024
                ]
                sSFB: cute.struct.Align[
                    cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_layout)], 1024
                ]
                # warp-epilogue cross-warp scratch: one f32 per (q-row, N-warp).
                # 1KB vs the old 64KB [M,N] scatter -> smem stops capping occupancy.
                sRed: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self._M * 2], 1024
                ]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sA = storage.sA.get_tensor(sA_layout)
            sB = storage.sB.get_tensor(sB_layout)
            # sm120 SF layouts append a (size-1) stage mode; drop it for the MMA
            # partition. The compound layout's physical byte offsets for one
            # 128-row/K=128 tile equal the cuBLAS 128x4 layout (== _sf_offset), so
            # the fill writes physical bytes through a flat contiguous view.
            sSFA = storage.sSFA.get_tensor(sfa_layout)[None, None, 0]
            sSFB = storage.sSFB.get_tensor(sfb_layout)[None, None, 0]
            sfa_n = cute.cosize(sfa_layout)
            sfb_n = cute.cosize(sfb_layout)
            sSFA_u8 = cute.recast_tensor(
                storage.sSFA.get_tensor(cute.make_layout(sfa_n, stride=1)), Uint8
            )
            sSFB_u8 = cute.recast_tensor(
                storage.sSFB.get_tensor(cute.make_layout(sfb_n, stride=1)), Uint8
            )
            sRed = storage.sRed.get_tensor(
                cute.make_layout((self._M, 2), stride=(2, 1))
            )

            mQ_u32 = cute.recast_tensor(mQ, Uint32)
            mK_u32 = cute.recast_tensor(mK, Uint32)
            sA_u32 = cute.recast_tensor(sA, Uint32)
            sB_u32 = cute.recast_tensor(sB, Uint32)

            # ----- stage Q (packed fp4 + e4m3 SF) once into smem -----
            # Q is always flat (current query tokens, never paged): read absolute
            # row q_tile0+m, mask on the in-tile position m_block*M+m.
            q_tile0 = q_start + m_block * self._M
            q_pos0 = m_block * self._M
            self._load_packed(
                mQ_u32[None, qo_head, None], sA_u32, q_tile0, q_pos0, seqlen_q, tidx
            )
            self._load_sf(
                mQsf,
                sSFA_u8,
                q_pos0,
                seqlen_q,
                q_tile0 * mQ.shape[1] + qo_head,
                mQ.shape[1],
                tidx,
            )
            self.cta_sync_barrier.arrive_and_wait()

            # ----- MMA fragment / copy setup (ported from b12x) -----
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)
            tCrSFA = self._partition_fragment_SFA(sSFA, thr_mma, tidx)
            tCrSFB = self._partition_fragment_SFB(sSFB, thr_mma, tidx)
            cS = cute.make_identity_tensor((self._M, self._N))
            tCcC = thr_mma.partition_C(cS)
            acc = cute.make_rmem_tensor(tCcC.shape[:3], self.acc_dtype)

            ldm_A = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype
            )
            ldm_B = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype
            )
            tiled_copy_A = cute.make_tiled_copy_A(ldm_A, tiled_mma)
            tiled_copy_B = cute.make_tiled_copy_B(ldm_B, tiled_mma)
            cp_sf = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
            tiled_copy_SFA = cute.make_tiled_copy(
                cp_sf,
                self._get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            tiled_copy_SFB = cute.make_tiled_copy(
                cp_sf,
                self._get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            thr_A = tiled_copy_A.get_slice(tidx)
            thr_B = tiled_copy_B.get_slice(tidx)
            thr_SFA = tiled_copy_SFA.get_slice(tidx)
            thr_SFB = tiled_copy_SFB.get_slice(tidx)
            tCsA_cp = thr_A.partition_S(sA)
            tCrA_cp = thr_A.retile(tCrA)
            tCsB_cp = thr_B.partition_S(sB)
            tCrB_cp = thr_B.retile(tCrB)
            tCsSFA_cp = thr_SFA.partition_S(sSFA)
            tCrSFA_cp = thr_SFA.retile(tCrSFA)
            tCsSFB_cp = thr_SFB.partition_S(sSFB)
            tCrSFB_cp = thr_SFB.retile(tCrSFB)

            # Q fragments load once (sA / sSFA reused across all kv-blocks)
            for kb in cutlass.range_constexpr(self.num_k_blocks):
                cute.copy(
                    tiled_copy_A, tCsA_cp[None, None, kb], tCrA_cp[None, None, kb]
                )
            cute.copy(
                tiled_copy_SFA,
                cute.filter_zeros(tCsSFA_cp),
                cute.filter_zeros(tCrSFA_cp),
            )

            # split-K: this CTA owns kv_block = split_idx, split_idx + num_splits,
            # ... < max_k_tiles (disjoint across splits, union covers all columns
            # once, no reduction). num_splits==1 -> the original [0, max_k_tiles).
            #
            # cp.async K double-buffering was tried here and reverted: at
            # min_blocks_per_mp=2 the kernel is register-bound (128 regs/thread), so a
            # 2nd K stage just spills to local memory with no occupancy gain, and the
            # CTAs already hide the K-load latency. Sync load + high occupancy wins.
            n_iter = cute.ceil_div(max_k_tiles, num_splits)
            for it in cutlass.range(n_iter):
                kv_block = split_idx + it * num_splits
                if kv_block < num_kv_blocks:
                    self.cta_sync_barrier.arrive_and_wait()
                    k_pos0 = kv_block * self._N
                    if cutlass.const_expr(self._paged):
                        # paged: kv_block maps to a 128-row page via the page table.
                        # Read it page-local (read_base 0) and mask on the global
                        # position; SF row = (page*Hkv + kv_head)*N + token, stride 1.
                        page = mPageTable[batch_idx, kv_block]
                        gK_pg = cute.recast_tensor(
                            mK[page, kv_head, None, None], Uint32
                        )
                        self._load_packed(gK_pg, sB_u32, 0, k_pos0, seqlen_k, tidx)
                        self._load_sf(
                            mKsf,
                            sSFB_u8,
                            k_pos0,
                            seqlen_k,
                            (page * mK.shape[1] + kv_head) * self._N,
                            1,
                            tidx,
                        )
                    else:
                        # flat: read absolute row k_tile0+m, SF row = token*Hkv+kv_head.
                        k_tile0 = k_start + kv_block * self._N
                        self._load_packed(
                            mK_u32[None, kv_head, None],
                            sB_u32,
                            k_tile0,
                            k_pos0,
                            seqlen_k,
                            tidx,
                        )
                        self._load_sf(
                            mKsf,
                            sSFB_u8,
                            k_pos0,
                            seqlen_k,
                            k_tile0 * mK.shape[1] + kv_head,
                            mK.shape[1],
                            tidx,
                        )
                    self.cta_sync_barrier.arrive_and_wait()

                    for kb in cutlass.range_constexpr(self.num_k_blocks):
                        cute.copy(
                            tiled_copy_B,
                            tCsB_cp[None, None, kb],
                            tCrB_cp[None, None, kb],
                        )
                    cute.copy(
                        tiled_copy_SFB,
                        cute.filter_zeros(tCsSFB_cp),
                        cute.filter_zeros(tCrSFB_cp),
                    )

                    acc.fill(0.0)
                    for mt in cutlass.range_constexpr(self.num_m_tiles):
                        for nt in cutlass.range_constexpr(self.num_n_tiles):
                            for kb in cutlass.range_constexpr(self.num_k_blocks):
                                mma_atom.set(
                                    WarpField.SFA, tCrSFA[None, mt, kb].iterator
                                )
                                mma_atom.set(
                                    WarpField.SFB, tCrSFB[None, nt, kb].iterator
                                )
                                cute.gemm(
                                    mma_atom,
                                    acc[None, mt, nt],
                                    tCrA[None, mt, kb],
                                    tCrB[None, nt, kb],
                                    acc[None, mt, nt],
                                )

                    # warp-level block-max epilogue (no big scatter buffer): each
                    # thread reduces its own N columns + a thread-quad shuffle ->
                    # max over its warp's N for each q-row it owns; the 2 N-warps
                    # are then combined through sRed[q-row, n-warp] (1KB smem).
                    acc_mn = self._make_acc_tensor_mn_view(acc)
                    tScS_mn = self._make_acc_tensor_mn_view(tCcC)
                    n_rows = cute.size(acc_mn.shape[0])
                    thr_vmnk = tiled_mma.thr_layout_vmnk.get_flat_coord(tidx)
                    nwarp = thr_vmnk[2]
                    self.cta_sync_barrier.arrive_and_wait()
                    for r in cutlass.range_constexpr(n_rows):
                        row = tScS_mn[r, 0][0]
                        q_loc = m_block * self._M + row
                        if cutlass.const_expr(self._is_causal):
                            col_limit = cutlass.min(
                                q_loc + mQOffset[batch_idx] + 1, seqlen_k
                            )
                        else:
                            col_limit = seqlen_k
                        for c in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                            k_pos = kv_block * self._N + tScS_mn[0, c][1]
                            if cute.elem_less(col_limit, k_pos + 1):
                                acc_mn[r, c] = -cutlass.Float32.inf
                        tile_max = (
                            acc_mn[r, None]
                            .load()
                            .reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
                        )
                        tile_max = self._threadquad_reduce_max(tile_max)
                        sRed[row, nwarp] = tile_max
                    self.cta_sync_barrier.arrive_and_wait()
                    # combine the 2 N-warps; one writer per q-row (nwarp 0 owner)
                    if nwarp == 0:
                        for r in cutlass.range_constexpr(n_rows):
                            row = tScS_mn[r, 0][0]
                            q_loc = m_block * self._M + row
                            both = cute.arch.fmax(sRed[row, 0], sRed[row, 1])
                            if q_loc < seqlen_q:
                                mMaxScore[qo_head, kv_block, q_start + q_loc] = (
                                    both * global_scale
                                )
                elif kv_block < max_k_tiles:
                    if tidx < self._M:
                        q_loc2 = m_block * self._M + tidx
                        if q_loc2 < seqlen_q:
                            mMaxScore[
                                qo_head, kv_block, q_start + q_loc2
                            ] = -cutlass.Float32.inf

    @cute.jit
    def _sf_offset(self, srow, scol):
        """cuBLAS/cuDNN 128x4 tiled SF byte offset (the layout the index-K cache is
        quantized in). For one 128-row, K=128 tile this equals the SM120 SF smem
        layout, so we can fill ``sSFA``/``sSFB`` by this offset directly."""
        srow_in = srow % 128
        return (
            ((srow // 128) * self._sf_tiles_n + scol // 4) * 512
            + (srow_in % 32) * 16
            + (srow_in // 32) * 4
            + scol % 4
        )

    @cute.jit
    def _load_packed(self, gX_u32, sX_u32, read_base, pos_base, seqlen, tidx):
        """Stage a (M or N) x head_dim packed-fp4 tile into plain K-major fp4 smem.
        Reads source row ``read_base + m``; row ``m`` is valid iff its in-sequence
        position ``pos_base + m`` is < ``seqlen`` (else zero-filled, since its Q.K^T
        is masked out anyway). Decoupling the read row from the position lets the
        paged K path read a page-local tile (``read_base == 0``) while still masking
        on the global token position ``kv_block*N + m``."""
        rows = cute.size(sX_u32.shape[0])
        total = rows * self._chunks_u32
        for it in cutlass.range_constexpr(total // self._num_threads):
            e = tidx + it * self._num_threads
            m = e // self._chunks_u32
            c = e % self._chunks_u32
            if cute.elem_less(pos_base + m, seqlen):
                sX_u32[m, c] = gX_u32[read_base + m, c]
            else:
                sX_u32[m, c] = Uint32(0)

    @cute.jit
    def _load_sf(self, mXsf, sXsf_u8, pos_base, seqlen, sf_row_base, sf_stride, tidx):
        """Fill the SM120 SF smem (== cuBLAS 128x4 for one 128-row tile) from the
        e4m3 SF buffer. One byte per (row, 16-group); 128*8 bytes total. Row ``m``
        reads SF logical row ``sf_row_base + m*sf_stride`` and is valid iff its
        in-sequence position ``pos_base + m`` is < ``seqlen``. The flat path passes
        the token*heads+head row base with stride ``num_heads``; the paged path
        passes the page-relative row base ``(page*Hkv + kv_head)*N`` with stride 1."""
        total = self._M * self._sf_per_row
        for it in cutlass.range_constexpr(total // self._num_threads):
            e = tidx + it * self._num_threads
            m = e // self._sf_per_row
            g = e % self._sf_per_row
            dst = self._sf_offset(m, g)
            if cute.elem_less(pos_base + m, seqlen):
                sXsf_u8[dst] = mXsf[self._sf_offset(sf_row_base + m * sf_stride, g)]
            else:
                sXsf_u8[dst] = Uint8(0)

    # ---- SF fragment partition (ported verbatim from b12x) ----
    def _partition_fragment_SFA(self, sfa_tensor, thr_mma, tidx):
        thrfrg = self._thrfrg_SFA(sfa_tensor.layout, thr_mma)
        thr_tensor = cute.make_tensor(sfa_tensor.iterator, thrfrg)
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vmk = (thr_vmnk[0], (thr_vmnk[1], thr_vmnk[3]))
        part = thr_tensor[thr_vmk, (None, None)]
        part = cute.group_modes(cute.flatten(part), 0, 2)
        return cute.make_fragment_like(part)

    def _partition_fragment_SFB(self, sfb_tensor, thr_mma, tidx):
        thrfrg = self._thrfrg_SFB(sfb_tensor.layout, thr_mma)
        thr_tensor = cute.make_tensor(sfb_tensor.iterator, thrfrg)
        thr_vmnk = thr_mma.thr_layout_vmnk.get_flat_coord(tidx)
        thr_vnk = (thr_vmnk[0], (thr_vmnk[2], thr_vmnk[3]))
        part = thr_tensor[thr_vnk, (None, None)]
        part = cute.group_modes(cute.flatten(part), 0, 2)
        part = cute.group_modes(part, 1, 3)
        return cute.make_fragment_like(part)

    def _thrfrg_SFA(self, sfa_tensor, tiled_mma):
        atom_shape_mnk = tiled_mma.shape_mnk
        atom_sfa_layout = cute.make_layout(
            shape=((2, 2, 8), 64), stride=((8, 0, 1), 16)
        )
        permutation_mnk = tiled_mma.permutation_mnk
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk
        t_tile = (permutation_mnk[0], permutation_mnk[2])
        t_tensor = cute.logical_divide(sfa_tensor, t_tile)
        a_tile = (
            cute.make_layout((atom_shape_mnk[0])),
            cute.make_layout((atom_shape_mnk[2])),
        )
        a_tensor = cute.zipped_divide(t_tensor, a_tile)
        tv_tensor = cute.composition(a_tensor, (atom_sfa_layout, None))
        thr_tile = (
            None,
            (
                cute.make_layout(cute.size(thr_layout_vmnk[1])),
                cute.make_layout(cute.size(thr_layout_vmnk[3])),
            ),
        )
        return cute.zipped_divide(tv_tensor, thr_tile)

    def _thrfrg_SFB(self, sfb_tensor, tiled_mma):
        atom_shape_mnk = tiled_mma.shape_mnk
        atom_sfb_layout = cute.make_layout(shape=((4, 8), 64), stride=((0, 1), 8))
        permutation_mnk = tiled_mma.permutation_mnk
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk
        t_tile = (permutation_mnk[1], permutation_mnk[2])
        t_tensor = cute.logical_divide(sfb_tensor, t_tile)
        a_tile = (
            cute.make_layout((atom_shape_mnk[1])),
            cute.make_layout((atom_shape_mnk[2])),
        )
        a_tensor = cute.zipped_divide(t_tensor, a_tile)
        tv_tensor = cute.composition(a_tensor, (atom_sfb_layout, None))
        thr_tile = (
            None,
            (
                cute.make_layout(cute.size(thr_layout_vmnk[2])),
                cute.make_layout(cute.size(thr_layout_vmnk[3])),
            ),
        )
        return cute.zipped_divide(tv_tensor, thr_tile)

    def _get_layoutSFA_TV(self, tiled_mma):
        perm_m = tiled_mma.permutation_mnk[0]
        perm_k = tiled_mma.permutation_mnk[2]
        ref_A = cute.make_layout((cute.size(perm_m), cute.size(perm_k)))
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk
        atile = (
            None,
            (
                cute.make_layout(
                    shape=(
                        cute.size(thr_layout_vmnk[1]),
                        cute.size(thr_layout_vmnk[2]),
                    ),
                    stride=(1, 0),
                ),
                None,
            ),
        )
        thridx_2_thrid = cute.right_inverse(thr_layout_vmnk)
        thrfrg_sfa = self._thrfrg_SFA(ref_A, tiled_mma)
        layout_tv = cute.composition(thrfrg_sfa, (atile, None))
        return cute.composition(layout_tv, (thridx_2_thrid, None))

    def _get_layoutSFB_TV(self, tiled_mma):
        perm_n = tiled_mma.permutation_mnk[1]
        perm_k = tiled_mma.permutation_mnk[2]
        ref_B = cute.make_layout((cute.size(perm_n), cute.size(perm_k)))
        thr_layout_vmnk = tiled_mma.thr_layout_vmnk
        atile = (
            None,
            (
                cute.make_layout(
                    shape=(
                        cute.size(thr_layout_vmnk[1]),
                        cute.size(thr_layout_vmnk[2]),
                    ),
                    stride=(0, 1),
                ),
                None,
            ),
        )
        thridx_2_thrid = cute.right_inverse(thr_layout_vmnk)
        thrfrg_sfb = self._thrfrg_SFB(ref_B, tiled_mma)
        layout_tv = cute.composition(thrfrg_sfb, (atile, None))
        return cute.composition(layout_tv, (thridx_2_thrid, None))

    @cute.jit
    def _threadquad_reduce_max(self, val):
        # max across the 4 lanes of an m16n8 row-quad (the N lanes within a warp)
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


class MsaProxyScoreFp4MmaDecodePackedSm12x(MsaProxyScoreFp4MmaSm12x):
    """The head-fused packed-decode schedule on the fp4 tensor cores.

    The fp4-MMA tile is already 128 rows, exactly ``qhead_per_kv x pack_q_len``, so
    the packed schedule is a pure relabel of the parent fp4-MMA kernel: pack all
    ``qhead_per_kv`` query heads of one kv_head x ``pack_q_len`` q-tokens into the 128
    MMA rows so a single CTA scores every head of one ``(batch, kv_head)`` against
    each KV block, reading the shared index-K **once** (``qhead_per_kv``x less K
    traffic than the per-query-head general schedule). Only the Q gather, the grid,
    and the per-row output mapping differ from the parent; the fp4 mainloop, SF
    plumbing, and warp block-max epilogue are inherited verbatim.

    ``(qhead_per_kv, pack_q_len)`` is any factorization of the 128-row tile with
    ``qhead_per_kv == group_size`` (defaults to the 16-head ``q_len<=8`` deployment
    shape, ``16 x 8``; the MiniMax-M3 indexer is ``4 x 32`` -- group_size 4,
    ``q_len<=32``). Packing: row ``r = local_head * pack_q_len + token``, ``head =
    kv_head * qhead_per_kv + local_head``, ``q = q_start + token``. Grid is one CTA
    per ``(split, batch, kv_head)``; split-K applies as in the parent.
    """

    _PACK_Q_LEN = 8
    _QHEAD_PER_KV = 16

    def __init__(
        self,
        head_dim: int = 128,
        is_causal: bool = True,
        paged: bool = False,
        qhead_per_kv: int = _QHEAD_PER_KV,
        pack_q_len: int = _PACK_Q_LEN,
    ):
        super().__init__(head_dim=head_dim, is_causal=is_causal, paged=paged)
        if qhead_per_kv * pack_q_len != self._M:
            raise ValueError(
                f"qhead_per_kv * pack_q_len must equal the {self._M}-row MMA tile, "
                f"got {qhead_per_kv} x {pack_q_len}"
            )
        # instance attrs shadow the class defaults; they are constexpr-baked into the
        # gather/epilogue, so each factorization compiles to its own kernel.
        self._QHEAD_PER_KV = qhead_per_kv
        self._PACK_Q_LEN = pack_q_len

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (total_q, Hq, head_dim//2) packed e2m1, uint8
        mK: cute.Tensor,  # flat (total_k,Hkv,hd/2) | paged (pages,Hkv,128,hd/2) u8
        mQsf: cute.Tensor,  # uint8 e4m3 scales, cuBLAS 128x4 tiled
        mKsf: cute.Tensor,  # uint8 e4m3 scales, cuBLAS 128x4 tiled
        mPageTable: cute.Tensor,  # (B, max_pages) int32 paged; dummy (1,1) flat
        mMaxScore: cute.Tensor,  # (Hq, max_k_tiles, total_q) f32
        mCuQ: cute.Tensor,  # (B + 1,) int32
        mCuK: cute.Tensor,  # (B + 1,) int32
        mQOffset: cute.Tensor,  # (B,) int32 causal offset
        q_global_scale: cutlass.Float32,
        k_global_scale: cutlass.Float32,
        max_seqlen_q: cutlass.Int32,
        batch_size: cutlass.Int32,
        num_qo_heads: cutlass.Int32,
        max_k_tiles: cutlass.Int32,
        num_splits: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # one CTA per (split, batch, kv_head): all qhead_per_kv heads of a kv_head
        # are packed into the single 128-row tile, so the index-K is read once per
        # kv_head (qhead_per_kv x less K traffic than the per-head general schedule).
        self.kernel(
            mQ,
            mK,
            mQsf,
            mKsf,
            mPageTable,
            mMaxScore,
            mCuQ,
            mCuK,
            mQOffset,
            q_global_scale * k_global_scale,
            max_k_tiles,
            num_splits,
        ).launch(
            grid=(
                num_splits,
                batch_size,
                num_qo_heads // self._QHEAD_PER_KV,
            ),
            block=[self._num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=self._min_blocks,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mQsf: cute.Tensor,
        mKsf: cute.Tensor,
        mPageTable: cute.Tensor,
        mMaxScore: cute.Tensor,
        mCuQ: cute.Tensor,
        mCuK: cute.Tensor,
        mQOffset: cute.Tensor,
        global_scale: cutlass.Float32,  # q_global * k_global, applied at store
        max_k_tiles: cutlass.Int32,
        num_splits: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        split_idx, batch_idx, kv_head = cute.arch.block_idx()

        q_start = mCuQ[batch_idx]
        seqlen_q = mCuQ[batch_idx + 1] - q_start
        k_start = mCuK[batch_idx]
        seqlen_k = mCuK[batch_idx + 1] - k_start
        num_kv_blocks = cute.ceil_div(seqlen_k, self._N)
        num_qo_heads = mQ.shape[1]

        tiled_mma, mma_atom, sfa_layout, sfb_layout = self._build_mma()
        sA_layout = self._ab_layout(self._M)
        sB_layout = self._ab_layout(self._N)

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(sA_layout)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(sB_layout)], 1024
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_layout)], 1024
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_layout)], 1024
            ]
            sRed: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self._M * 2], 1024
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout)
        sB = storage.sB.get_tensor(sB_layout)
        sSFA = storage.sSFA.get_tensor(sfa_layout)[None, None, 0]
        sSFB = storage.sSFB.get_tensor(sfb_layout)[None, None, 0]
        sfa_n = cute.cosize(sfa_layout)
        sfb_n = cute.cosize(sfb_layout)
        sSFA_u8 = cute.recast_tensor(
            storage.sSFA.get_tensor(cute.make_layout(sfa_n, stride=1)), Uint8
        )
        sSFB_u8 = cute.recast_tensor(
            storage.sSFB.get_tensor(cute.make_layout(sfb_n, stride=1)), Uint8
        )
        sRed = storage.sRed.get_tensor(cute.make_layout((self._M, 2), stride=(2, 1)))

        mQ_u32 = cute.recast_tensor(mQ, Uint32)
        mK_u32 = cute.recast_tensor(mK, Uint32)
        sA_u32 = cute.recast_tensor(sA, Uint32)
        sB_u32 = cute.recast_tensor(sB, Uint32)

        # ----- gather the qhead_per_kv-head x pack_q_len-q packed Q tile into smem -
        self._load_packed_q(mQ_u32, sA_u32, q_start, kv_head, seqlen_q, tidx)
        self._load_packed_q_sf(
            mQsf, sSFA_u8, q_start, kv_head, num_qo_heads, seqlen_q, tidx
        )
        self.cta_sync_barrier.arrive_and_wait()

        # ----- MMA fragment / copy setup (inherited layouts) -----
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCrSFA = self._partition_fragment_SFA(sSFA, thr_mma, tidx)
        tCrSFB = self._partition_fragment_SFB(sSFB, thr_mma, tidx)
        cS = cute.make_identity_tensor((self._M, self._N))
        tCcC = thr_mma.partition_C(cS)
        acc = cute.make_rmem_tensor(tCcC.shape[:3], self.acc_dtype)

        ldm_A = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype
        )
        ldm_B = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self.a_dtype
        )
        tiled_copy_A = cute.make_tiled_copy_A(ldm_A, tiled_mma)
        tiled_copy_B = cute.make_tiled_copy_B(ldm_B, tiled_mma)
        cp_sf = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
        tiled_copy_SFA = cute.make_tiled_copy(
            cp_sf,
            self._get_layoutSFA_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[0]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        tiled_copy_SFB = cute.make_tiled_copy(
            cp_sf,
            self._get_layoutSFB_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[1]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        thr_A = tiled_copy_A.get_slice(tidx)
        thr_B = tiled_copy_B.get_slice(tidx)
        thr_SFA = tiled_copy_SFA.get_slice(tidx)
        thr_SFB = tiled_copy_SFB.get_slice(tidx)
        tCsA_cp = thr_A.partition_S(sA)
        tCrA_cp = thr_A.retile(tCrA)
        tCsB_cp = thr_B.partition_S(sB)
        tCrB_cp = thr_B.retile(tCrB)
        tCsSFA_cp = thr_SFA.partition_S(sSFA)
        tCrSFA_cp = thr_SFA.retile(tCrSFA)
        tCsSFB_cp = thr_SFB.partition_S(sSFB)
        tCrSFB_cp = thr_SFB.retile(tCrSFB)

        for kb in cutlass.range_constexpr(self.num_k_blocks):
            cute.copy(tiled_copy_A, tCsA_cp[None, None, kb], tCrA_cp[None, None, kb])
        cute.copy(
            tiled_copy_SFA,
            cute.filter_zeros(tCsSFA_cp),
            cute.filter_zeros(tCrSFA_cp),
        )

        n_iter = cute.ceil_div(max_k_tiles, num_splits)
        for it in cutlass.range(n_iter):
            kv_block = split_idx + it * num_splits
            if kv_block < num_kv_blocks:
                self.cta_sync_barrier.arrive_and_wait()
                k_pos0 = kv_block * self._N
                if cutlass.const_expr(self._paged):
                    # paged: same page-table indirection as the general kernel; the
                    # packed schedule only changes the Q gather, not the K load.
                    page = mPageTable[batch_idx, kv_block]
                    gK_pg = cute.recast_tensor(mK[page, kv_head, None, None], Uint32)
                    self._load_packed(gK_pg, sB_u32, 0, k_pos0, seqlen_k, tidx)
                    self._load_sf(
                        mKsf,
                        sSFB_u8,
                        k_pos0,
                        seqlen_k,
                        (page * mK.shape[1] + kv_head) * self._N,
                        1,
                        tidx,
                    )
                else:
                    k_tile0 = k_start + kv_block * self._N
                    self._load_packed(
                        mK_u32[None, kv_head, None],
                        sB_u32,
                        k_tile0,
                        k_pos0,
                        seqlen_k,
                        tidx,
                    )
                    self._load_sf(
                        mKsf,
                        sSFB_u8,
                        k_pos0,
                        seqlen_k,
                        k_tile0 * mK.shape[1] + kv_head,
                        mK.shape[1],
                        tidx,
                    )
                self.cta_sync_barrier.arrive_and_wait()

                for kb in cutlass.range_constexpr(self.num_k_blocks):
                    cute.copy(
                        tiled_copy_B,
                        tCsB_cp[None, None, kb],
                        tCrB_cp[None, None, kb],
                    )
                cute.copy(
                    tiled_copy_SFB,
                    cute.filter_zeros(tCsSFB_cp),
                    cute.filter_zeros(tCrSFB_cp),
                )

                acc.fill(0.0)
                for mt in cutlass.range_constexpr(self.num_m_tiles):
                    for nt in cutlass.range_constexpr(self.num_n_tiles):
                        for kb in cutlass.range_constexpr(self.num_k_blocks):
                            mma_atom.set(WarpField.SFA, tCrSFA[None, mt, kb].iterator)
                            mma_atom.set(WarpField.SFB, tCrSFB[None, nt, kb].iterator)
                            cute.gemm(
                                mma_atom,
                                acc[None, mt, nt],
                                tCrA[None, mt, kb],
                                tCrB[None, nt, kb],
                                acc[None, mt, nt],
                            )

                # warp block-max epilogue; packed row -> (local_head, token)
                acc_mn = self._make_acc_tensor_mn_view(acc)
                tScS_mn = self._make_acc_tensor_mn_view(tCcC)
                n_rows = cute.size(acc_mn.shape[0])
                thr_vmnk = tiled_mma.thr_layout_vmnk.get_flat_coord(tidx)
                nwarp = thr_vmnk[2]
                self.cta_sync_barrier.arrive_and_wait()
                for r in cutlass.range_constexpr(n_rows):
                    row = tScS_mn[r, 0][0]
                    token = row % self._PACK_Q_LEN
                    if cutlass.const_expr(self._is_causal):
                        col_limit = cutlass.min(
                            token + mQOffset[batch_idx] + 1, seqlen_k
                        )
                    else:
                        col_limit = seqlen_k
                    for c in cutlass.range_constexpr(cute.size(acc_mn.shape[1])):
                        k_pos = kv_block * self._N + tScS_mn[0, c][1]
                        if cute.elem_less(col_limit, k_pos + 1):
                            acc_mn[r, c] = -cutlass.Float32.inf
                    tile_max = (
                        acc_mn[r, None]
                        .load()
                        .reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
                    )
                    tile_max = self._threadquad_reduce_max(tile_max)
                    sRed[row, nwarp] = tile_max
                self.cta_sync_barrier.arrive_and_wait()
                if nwarp == 0:
                    for r in cutlass.range_constexpr(n_rows):
                        row = tScS_mn[r, 0][0]
                        local_head = row // self._PACK_Q_LEN
                        token = row % self._PACK_Q_LEN
                        head = kv_head * self._QHEAD_PER_KV + local_head
                        both = cute.arch.fmax(sRed[row, 0], sRed[row, 1])
                        if cute.elem_less(token, seqlen_q):
                            mMaxScore[head, kv_block, q_start + token] = (
                                both * global_scale
                            )
            elif kv_block < max_k_tiles:
                if tidx < self._M:
                    local_head2 = tidx // self._PACK_Q_LEN
                    token2 = tidx % self._PACK_Q_LEN
                    head2 = kv_head * self._QHEAD_PER_KV + local_head2
                    if cute.elem_less(token2, seqlen_q):
                        mMaxScore[
                            head2, kv_block, q_start + token2
                        ] = -cutlass.Float32.inf

    @cute.jit
    def _load_packed_q(self, mQ_u32, sX_u32, q_start, kv_head, seqlen_q, tidx):
        """Gather qhead_per_kv heads x pack_q_len q-tokens of packed fp4 into the
        128-row tile. Packed row ``r = local_head * pack_q_len + token``; ``head =
        kv_head * qhead_per_kv + local_head``; ``gi = q_start + token``. Rows with
        ``token >= seqlen_q`` are zero-filled (masked out in the epilogue anyway)."""
        total = self._M * self._chunks_u32
        for it in cutlass.range_constexpr(total // self._num_threads):
            e = tidx + it * self._num_threads
            r = e // self._chunks_u32
            c = e % self._chunks_u32
            local_head = r // self._PACK_Q_LEN
            token = r % self._PACK_Q_LEN
            head = kv_head * self._QHEAD_PER_KV + local_head
            if cute.elem_less(token, seqlen_q):
                sX_u32[r, c] = mQ_u32[q_start + token, head, c]
            else:
                sX_u32[r, c] = Uint32(0)

    @cute.jit
    def _load_packed_q_sf(
        self, mQsf, sXsf_u8, q_start, kv_head, num_qo_heads, seqlen_q, tidx
    ):
        """Fill the SF smem for the packed Q tile. SF logical row =
        ``(q_start + token) * num_qo_heads + head`` (the flat token*heads+head row
        order ``nvfp4_quantize`` produces from a ``(total, heads, d)`` reshape)."""
        total = self._M * self._sf_per_row
        for it in cutlass.range_constexpr(total // self._num_threads):
            e = tidx + it * self._num_threads
            r = e // self._sf_per_row
            g = e % self._sf_per_row
            dst = self._sf_offset(r, g)
            local_head = r // self._PACK_Q_LEN
            token = r % self._PACK_Q_LEN
            head = kv_head * self._QHEAD_PER_KV + local_head
            if cute.elem_less(token, seqlen_q):
                sXsf_u8[dst] = mQsf[
                    self._sf_offset((q_start + token) * num_qo_heads + head, g)
                ]
            else:
                sXsf_u8[dst] = Uint8(0)
