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
"""

"""BF16 noncausal block-sparse (VSA) multi-head attention forward for SM90 (H100).

CuTe DSL warp-specialized kernel:
  - one CTA per (head, 64-row query block); grid = (M/64, H) so concurrent CTAs
    stay inside one head and sweep sorted KV block lists in near-lockstep (L2 reuse)
  - warpgroup 0: TMA producer (Q once, then K/V gathered via dynamic
    block_indices coordinates in 32-token half-block stages; V aliases K's SMEM)
  - warpgroup 1: math (WGMMA QK^T -> online softmax in exp2 domain -> WGMMA PV
    with the P fragment in registers), then StMatrix + TMA-store epilogue

The implementation retains the reference's 96-token reduction order and
scalar rounding on normal workloads, with stable masking on padded tiles. Known
FA3 finite-mask compatibility exceptions are documented in the wrapper's plan()
docstring.

Only compiled code is cached here. The wrapper owns validated GPU descriptors;
Q/K/V and output are supplied afresh on every call.
"""


import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warpgroup as warpgroup
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack

LOG2_E = 1.4426950216293335


class VsaFwdSm90:
    def __init__(self, kv_stage=12, q_stage=1, epi_stage=2, output_dim=32):
        self.output_dim = output_dim
        self.q_stage = q_stage
        self.kv_stage = kv_stage  # 32-token half-block stages (8 KB each)
        self.epi_stage = epi_stage
        self.qk_acc_dtype = cutlass.Float32
        self.pv_acc_dtype = cutlass.Float32

        # (M, N, K=D) for QK^T over one 32-token fragment; (M, N=D, K) for PV
        self.qk_mma_tiler = (64, 32, 128)
        self.pv_mma_tiler = (64, output_dim, 32)
        self.epi_tile = (64, min(64, output_dim))

        self.cluster_shape_mnk = (1, 1, 1)
        self.atom_layout_mnk = (1, 1, 1)

        self.num_threads_per_warp_group = 128
        self.threads_per_cta = 256  # WG0 producer + WG1 math
        self.load_warp_group_id = 0
        self.math_warp_group_id = 1
        self.producer_warp_loadkv_id = 1  # warp 1 of WG0 issues TMA

        self.num_regs_load = 32
        self.num_regs_mma = 224
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # (M, D, H)  d-contiguous
        k: cute.Tensor,  # (N, D, H)  d-contiguous
        v: cute.Tensor,  # (D, N, H)  d-contiguous (MN-major B for PV)
        o: cute.Tensor,  # (M, D, H)  d-contiguous
        block_indices: cute.Tensor,  # (H, MB, K) int32
        block_counts: cute.Tensor,  # (H, MB) int32
        work_order: cute.Tensor,
        use_order: cutlass.Constexpr,
        scale_softmax_log2: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        self.q_layout = utils.LayoutEnum.from_tensor(q)
        self.k_layout = utils.LayoutEnum.from_tensor(k)
        self.v_layout = utils.LayoutEnum.from_tensor(v)
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        qk_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            self.q_layout.sm90_mma_major_mode(),
            self.k_layout.sm90_mma_major_mode(),
            self.qk_acc_dtype,
            self.atom_layout_mnk,
            self.qk_mma_tiler[:2],
        )
        pv_tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            cute.nvgpu.OperandMajorMode.K,  # P is K-major register operand
            self.v_layout.sm90_mma_major_mode(),
            self.pv_acc_dtype,
            self.atom_layout_mnk,
            self.pv_mma_tiler[:2],
            warpgroup.OperandSource.RMEM,
        )

        q_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.q_layout, (64, 128, 128), self.q_dtype, self.q_stage
        )
        k_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.k_layout, self.qk_mma_tiler, self.k_dtype, self.kv_stage
        )
        v_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.v_layout, (64, 128, 32), self.v_dtype, self.kv_stage
        )
        o_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.epi_stage
        )

        q_smem_layout = cute.slice_(q_smem_layout_staged, (None, None, 0))
        k_smem_layout = cute.slice_(k_smem_layout_staged, (None, None, 0))

        tma_atom_q, tma_tensor_q = self._make_tma_atoms_and_tensors(
            q, q_smem_layout_staged, (64, 128)
        )
        tma_atom_k, tma_tensor_k = self._make_tma_atoms_and_tensors(
            k, k_smem_layout_staged, (self.qk_mma_tiler[1], self.qk_mma_tiler[2])
        )
        tma_atom_v, tma_tensor_v = self._make_tma_atoms_and_tensors(
            v, v_smem_layout_staged, (128, 32)
        )

        o_smem_layout = cute.slice_(o_smem_layout_staged, (None, None, 0))
        tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            o,
            o_smem_layout,
            self.epi_tile,
        )

        self.tma_copy_q_bytes = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        self.tma_copy_kv_bytes = cute.size_in_bytes(self.k_dtype, k_smem_layout)

        @cute.struct
        class SharedStorage:
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.q_stage * 2]
            load_kv_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.kv_stage * 2]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        grid = ((q.shape[0] // 64) * q.shape[2], 1, 128 // self.output_dim)

        self.kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_o,
            tma_tensor_o,
            block_indices,
            block_counts,
            work_order,
            use_order,
            scale_softmax_log2,
            q_smem_layout_staged,
            k_smem_layout_staged,
            v_smem_layout_staged,
            o_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=2,
        )

    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        mIdx: cute.Tensor,
        mCnt: cute.Tensor,
        work_order: cute.Tensor,
        use_order: cutlass.Constexpr,
        scale_softmax_log2: cutlass.Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        linear, _, dpart = cute.arch.block_idx()
        if cutlass.const_expr(use_order):
            linear = work_order[linear]
        bidx = linear % (mQ_qdl.shape[0] // 64)
        bidy = linear // (mQ_qdl.shape[0] // 64)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = self.make_and_init_load_q_pipeline(
            storage.load_q_mbar_ptr.data_ptr()
        )
        load_kv_producer, load_kv_consumer = self.make_and_init_load_kv_pipeline(
            storage.load_kv_mbar_ptr.data_ptr()
        )
        tma_store_pipeline = self.make_and_init_tma_store_pipeline()

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )

        # SMEM views: V aliases K's bytes (interleaved on one pipeline);
        # O aliases Q's bytes (epilogue starts only after this WG's mainloop).
        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner),
            v_smem_layout_staged.outer,
        )
        sO = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, o_smem_layout_staged.inner, self.o_dtype),
            o_smem_layout_staged.outer,
        )

        qk_thr_mma = qk_tiled_mma.get_slice(tidx)
        pv_thr_mma = pv_tiled_mma.get_slice(tidx)

        gQ_qdl = cute.flat_divide(mQ_qdl, (64, 128))
        tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
        tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 2),
            cute.group_modes(tSgQ_qdl, 0, 3),
        )

        # K tiled in 32-token half-blocks: half-tile index = 2*block + h
        gK_kdl = cute.flat_divide(mK_kdl, cute.select(self.qk_mma_tiler, mode=[1, 2]))
        tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
        tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(tSgK_kdl, 0, 3),
        )

        # Keep the complete128-column TMA transfer and8KiB pipeline ledger.
        # Only the PV math and output are sliced into disjoint column ranges.
        gV_dkl = cute.flat_divide(mV_dkl, (128, 32))
        tOgV_dkl = gV_dkl
        tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v,
            0,
            cute.make_layout(1),
            cute.group_modes(sV, 0, 2),
            cute.group_modes(tOgV_dkl, 0, 2),
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        # ------------------------------------------------------------------
        # Producer: Q once, then per 96-token softmax tile its K half-blocks
        # followed by its V half-blocks — exactly the consumer's wait order.
        # Half-block token index h for slot s of a tile starting at half-block
        # position p: half = p + s; block = half // 2; h = half % 2.
        # ------------------------------------------------------------------
        if warp_group_idx == self.load_warp_group_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            producer_warp_role = warp_idx % 4
            if producer_warp_role == self.producer_warp_loadkv_id:
                cnt = cute.arch.make_warp_uniform(mCnt[bidy, bidx])

                q_handle = load_q_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_q,
                    tQgQ_qdl[(None, bidx, 0, bidy)],
                    tQsQ[(None, q_handle.index)],
                    tma_bar_ptr=q_handle.barrier,
                )

                # FA3-parity ordering: prefix tiles of three 32-token
                # half-blocks (partial tile last in memory) are PROCESSED in
                # reverse; each step consumes K of tile t then V of tile t+1
                # (deferred PV); the final V of tile 0 closes the stream.
                halves = 2 * cnt
                full_tiles = halves // 3
                rem = halves - 3 * full_tiles
                pbase = 3 * full_tiles

                if rem == 0:
                    for s in cutlass.range_constexpr(3):
                        blk = cute.arch.make_warp_uniform(
                            mIdx[bidy, bidx, (3 * (full_tiles - 1) + s) // 2]
                        )
                        hb = 2 * blk + ((3 * (full_tiles - 1) + s) % 2)
                        kh = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK_kdl[(None, hb, 0, bidy)],
                            tKsK[(None, kh.index)],
                            tma_bar_ptr=kh.barrier,
                        )
                    t = full_tiles - 2
                    while t >= 0:
                        for s in cutlass.range_constexpr(3):
                            blk = cute.arch.make_warp_uniform(
                                mIdx[bidy, bidx, (3 * t + s) // 2]
                            )
                            hb = 2 * blk + ((3 * t + s) % 2)
                            kh = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tKgK_kdl[(None, hb, 0, bidy)],
                                tKsK[(None, kh.index)],
                                tma_bar_ptr=kh.barrier,
                            )
                        for s in cutlass.range_constexpr(3):
                            blk = cute.arch.make_warp_uniform(
                                mIdx[bidy, bidx, (3 * (t + 1) + s) // 2]
                            )
                            hb = 2 * blk + ((3 * (t + 1) + s) % 2)
                            vh = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV_dkl[(None, 0, hb, bidy)],
                                tVsV[(None, vh.index)],
                                tma_bar_ptr=vh.barrier,
                            )
                        t -= 1
                    for s in cutlass.range_constexpr(3):
                        blk = cute.arch.make_warp_uniform(mIdx[bidy, bidx, (s) // 2])
                        hb = 2 * blk + ((s) % 2)
                        vh = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV_dkl[(None, 0, hb, bidy)],
                            tVsV[(None, vh.index)],
                            tma_bar_ptr=vh.barrier,
                        )
                if rem == 1:
                    blk = cute.arch.make_warp_uniform(mIdx[bidy, bidx, (pbase) // 2])
                    hb = 2 * blk + ((pbase) % 2)
                    kh = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK_kdl[(None, hb, 0, bidy)],
                        tKsK[(None, kh.index)],
                        tma_bar_ptr=kh.barrier,
                    )
                    for s in cutlass.range_constexpr(3):
                        blk = cute.arch.make_warp_uniform(
                            mIdx[bidy, bidx, (3 * (full_tiles - 1) + s) // 2]
                        )
                        hb = 2 * blk + ((3 * (full_tiles - 1) + s) % 2)
                        kh = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK_kdl[(None, hb, 0, bidy)],
                            tKsK[(None, kh.index)],
                            tma_bar_ptr=kh.barrier,
                        )
                    blk = cute.arch.make_warp_uniform(mIdx[bidy, bidx, (pbase) // 2])
                    hb = 2 * blk + ((pbase) % 2)
                    vh = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_v,
                        tVgV_dkl[(None, 0, hb, bidy)],
                        tVsV[(None, vh.index)],
                        tma_bar_ptr=vh.barrier,
                    )
                    t = full_tiles - 2
                    while t >= 0:
                        for s in cutlass.range_constexpr(3):
                            blk = cute.arch.make_warp_uniform(
                                mIdx[bidy, bidx, (3 * t + s) // 2]
                            )
                            hb = 2 * blk + ((3 * t + s) % 2)
                            kh = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tKgK_kdl[(None, hb, 0, bidy)],
                                tKsK[(None, kh.index)],
                                tma_bar_ptr=kh.barrier,
                            )
                        for s in cutlass.range_constexpr(3):
                            blk = cute.arch.make_warp_uniform(
                                mIdx[bidy, bidx, (3 * (t + 1) + s) // 2]
                            )
                            hb = 2 * blk + ((3 * (t + 1) + s) % 2)
                            vh = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV_dkl[(None, 0, hb, bidy)],
                                tVsV[(None, vh.index)],
                                tma_bar_ptr=vh.barrier,
                            )
                        t -= 1
                    for s in cutlass.range_constexpr(3):
                        blk = cute.arch.make_warp_uniform(mIdx[bidy, bidx, (s) // 2])
                        hb = 2 * blk + ((s) % 2)
                        vh = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV_dkl[(None, 0, hb, bidy)],
                            tVsV[(None, vh.index)],
                            tma_bar_ptr=vh.barrier,
                        )
                if rem == 2:
                    blk = cute.arch.make_warp_uniform(mIdx[bidy, bidx, (pbase) // 2])
                    hb = 2 * blk + ((pbase) % 2)
                    kh = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK_kdl[(None, hb, 0, bidy)],
                        tKsK[(None, kh.index)],
                        tma_bar_ptr=kh.barrier,
                    )
                    blk = cute.arch.make_warp_uniform(
                        mIdx[bidy, bidx, (pbase + 1) // 2]
                    )
                    hb = 2 * blk + ((pbase + 1) % 2)
                    kh = load_kv_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        tKgK_kdl[(None, hb, 0, bidy)],
                        tKsK[(None, kh.index)],
                        tma_bar_ptr=kh.barrier,
                    )
                    if full_tiles == 0:
                        blk = cute.arch.make_warp_uniform(
                            mIdx[bidy, bidx, (pbase) // 2]
                        )
                        hb = 2 * blk + ((pbase) % 2)
                        vh = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV_dkl[(None, 0, hb, bidy)],
                            tVsV[(None, vh.index)],
                            tma_bar_ptr=vh.barrier,
                        )
                        blk = cute.arch.make_warp_uniform(
                            mIdx[bidy, bidx, (pbase + 1) // 2]
                        )
                        hb = 2 * blk + ((pbase + 1) % 2)
                        vh = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV_dkl[(None, 0, hb, bidy)],
                            tVsV[(None, vh.index)],
                            tma_bar_ptr=vh.barrier,
                        )
                    else:
                        for s in cutlass.range_constexpr(3):
                            blk = cute.arch.make_warp_uniform(
                                mIdx[bidy, bidx, (3 * (full_tiles - 1) + s) // 2]
                            )
                            hb = 2 * blk + ((3 * (full_tiles - 1) + s) % 2)
                            kh = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_k,
                                tKgK_kdl[(None, hb, 0, bidy)],
                                tKsK[(None, kh.index)],
                                tma_bar_ptr=kh.barrier,
                            )
                        blk = cute.arch.make_warp_uniform(
                            mIdx[bidy, bidx, (pbase) // 2]
                        )
                        hb = 2 * blk + ((pbase) % 2)
                        vh = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV_dkl[(None, 0, hb, bidy)],
                            tVsV[(None, vh.index)],
                            tma_bar_ptr=vh.barrier,
                        )
                        blk = cute.arch.make_warp_uniform(
                            mIdx[bidy, bidx, (pbase + 1) // 2]
                        )
                        hb = 2 * blk + ((pbase + 1) % 2)
                        vh = load_kv_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV_dkl[(None, 0, hb, bidy)],
                            tVsV[(None, vh.index)],
                            tma_bar_ptr=vh.barrier,
                        )
                        t = full_tiles - 2
                        while t >= 0:
                            for s in cutlass.range_constexpr(3):
                                blk = cute.arch.make_warp_uniform(
                                    mIdx[bidy, bidx, (3 * t + s) // 2]
                                )
                                hb = 2 * blk + ((3 * t + s) % 2)
                                kh = load_kv_producer.acquire_and_advance()
                                cute.copy(
                                    tma_atom_k,
                                    tKgK_kdl[(None, hb, 0, bidy)],
                                    tKsK[(None, kh.index)],
                                    tma_bar_ptr=kh.barrier,
                                )
                            for s in cutlass.range_constexpr(3):
                                blk = cute.arch.make_warp_uniform(
                                    mIdx[bidy, bidx, (3 * (t + 1) + s) // 2]
                                )
                                hb = 2 * blk + ((3 * (t + 1) + s) % 2)
                                vh = load_kv_producer.acquire_and_advance()
                                cute.copy(
                                    tma_atom_v,
                                    tVgV_dkl[(None, 0, hb, bidy)],
                                    tVsV[(None, vh.index)],
                                    tma_bar_ptr=vh.barrier,
                                )
                            t -= 1
                        for s in cutlass.range_constexpr(3):
                            blk = cute.arch.make_warp_uniform(
                                mIdx[bidy, bidx, (s) // 2]
                            )
                            hb = 2 * blk + ((s) % 2)
                            vh = load_kv_producer.acquire_and_advance()
                            cute.copy(
                                tma_atom_v,
                                tVgV_dkl[(None, 0, hb, bidy)],
                                tVsV[(None, vh.index)],
                                tma_bar_ptr=vh.barrier,
                            )

        # ------------------------------------------------------------------
        # Math warpgroup
        # ------------------------------------------------------------------
        if warp_group_idx == self.math_warp_group_id:
            cute.arch.setmaxregister_increase(self.num_regs_mma)

            # ACCUMULATE stays on for every WGMMA; QK accumulators are
            # zero-filled explicitly (0 + x == x exactly). Setting this once
            # here keeps tiled-mma value mutation out of sibling regions.
            qk_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
            pv_tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

            cnt = cute.arch.make_warp_uniform(mCnt[bidy, bidx])

            tSsQ = qk_thr_mma.partition_A(sQ)
            tSsK = qk_thr_mma.partition_B(sK)
            tSrQ = qk_thr_mma.make_fragment_A(tSsQ)
            tSrK = qk_thr_mma.make_fragment_B(tSsK)
            sv_part = cute.local_tile(sV, (self.output_dim, 32), (dpart, 0, None))
            tOsV = pv_thr_mma.partition_B(sv_part)
            tOrV = pv_thr_mma.make_fragment_B(tOsV)

            q_handle = load_q_consumer.wait()

            pv_acc_shape = pv_thr_mma.partition_shape_C(
                (self.pv_mma_tiler[0], self.pv_mma_tiler[1])
            )
            acc_pv = pv_thr_mma.make_fragment_C(pv_acc_shape)
            qk_acc_shape = qk_thr_mma.partition_shape_C(
                (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
            )

            s_max_layout = cute.make_layout(
                cute.size(self.layout_acc_mn(pv_tiled_mma, acc_pv.layout), mode=[0])
            )
            s_max = cute.make_rmem_tensor_like(s_max_layout, self.qk_acc_dtype)
            a_sum = cute.make_rmem_tensor_like(s_max, cutlass.Float32)

            # Uniform online-softmax state: first-tile results are bit-identical
            # to a specialized first iteration (0+x==x, 0*0==0, max(-inf,x)==x).
            s_max.fill(-cutlass.Float32.inf)
            a_sum.fill(0.0)
            acc_pv.fill(0.0)

            # FA3-parity mainloop: prefix tiles of three 32-token fragments
            # (partial tile last in memory) processed in REVERSE, deferred PV:
            # acc = (acc + P_prev @ V_prev) * scale_cur each step.
            halves = 2 * cnt
            full_tiles = halves // 3
            rem = halves - 3 * full_tiles

            # P operand fragments, written in place each step
            a_frag_layout = self.convert_c_layout_to_a_layout(
                qk_thr_mma.make_fragment_C(qk_acc_shape).layout,
                pv_tiled_mma.tv_layout_A.shape[1],
            )
            p_a = cute.make_rmem_tensor_like(a_frag_layout, self.q_dtype)
            p_b = cute.make_rmem_tensor_like(a_frag_layout, self.q_dtype)
            p_c = cute.make_rmem_tensor_like(a_frag_layout, self.q_dtype)

            if rem == 0:
                load_kv_consumer, s_max, a_sum = self.prologue(
                    3,
                    qk_thr_mma,
                    acc_pv,
                    qk_tiled_mma,
                    pv_tiled_mma,
                    load_kv_consumer,
                    q_handle,
                    tSrQ,
                    tSrK,
                    s_max,
                    a_sum,
                    p_a,
                    p_b,
                    p_c,
                    scale_softmax_log2,
                    qk_acc_shape,
                )
                load_kv_consumer, s_max, a_sum = self.dstep_loop(
                    full_tiles - 1,
                    3,
                    qk_thr_mma,
                    acc_pv,
                    qk_tiled_mma,
                    pv_tiled_mma,
                    load_kv_consumer,
                    q_handle,
                    tSrQ,
                    tSrK,
                    s_max,
                    a_sum,
                    tOrV,
                    p_a,
                    p_b,
                    p_c,
                    scale_softmax_log2,
                    qk_acc_shape,
                )
                load_kv_consumer = self.final_pv(
                    3,
                    pv_tiled_mma,
                    load_kv_consumer,
                    acc_pv,
                    tOrV,
                    p_a,
                    p_b,
                    p_c,
                )
            if rem == 1:
                load_kv_consumer, s_max, a_sum = self.prologue(
                    1,
                    qk_thr_mma,
                    acc_pv,
                    qk_tiled_mma,
                    pv_tiled_mma,
                    load_kv_consumer,
                    q_handle,
                    tSrQ,
                    tSrK,
                    s_max,
                    a_sum,
                    p_a,
                    p_b,
                    p_c,
                    scale_softmax_log2,
                    qk_acc_shape,
                )
                if full_tiles == 0:
                    load_kv_consumer = self.final_pv(
                        1,
                        pv_tiled_mma,
                        load_kv_consumer,
                        acc_pv,
                        tOrV,
                        p_a,
                        p_b,
                        p_c,
                    )
                else:
                    load_kv_consumer, s_max, a_sum = self.dstep_loop(
                        cutlass.Int32(1),
                        1,
                        qk_thr_mma,
                        acc_pv,
                        qk_tiled_mma,
                        pv_tiled_mma,
                        load_kv_consumer,
                        q_handle,
                        tSrQ,
                        tSrK,
                        s_max,
                        a_sum,
                        tOrV,
                        p_a,
                        p_b,
                        p_c,
                        scale_softmax_log2,
                        qk_acc_shape,
                    )
                    load_kv_consumer, s_max, a_sum = self.dstep_loop(
                        full_tiles - 1,
                        3,
                        qk_thr_mma,
                        acc_pv,
                        qk_tiled_mma,
                        pv_tiled_mma,
                        load_kv_consumer,
                        q_handle,
                        tSrQ,
                        tSrK,
                        s_max,
                        a_sum,
                        tOrV,
                        p_a,
                        p_b,
                        p_c,
                        scale_softmax_log2,
                        qk_acc_shape,
                    )
                    load_kv_consumer = self.final_pv(
                        3,
                        pv_tiled_mma,
                        load_kv_consumer,
                        acc_pv,
                        tOrV,
                        p_a,
                        p_b,
                        p_c,
                    )
            if rem == 2:
                load_kv_consumer, s_max, a_sum = self.prologue(
                    2,
                    qk_thr_mma,
                    acc_pv,
                    qk_tiled_mma,
                    pv_tiled_mma,
                    load_kv_consumer,
                    q_handle,
                    tSrQ,
                    tSrK,
                    s_max,
                    a_sum,
                    p_a,
                    p_b,
                    p_c,
                    scale_softmax_log2,
                    qk_acc_shape,
                )
                if full_tiles == 0:
                    load_kv_consumer = self.final_pv(
                        2,
                        pv_tiled_mma,
                        load_kv_consumer,
                        acc_pv,
                        tOrV,
                        p_a,
                        p_b,
                        p_c,
                    )
                else:
                    load_kv_consumer, s_max, a_sum = self.dstep_loop(
                        cutlass.Int32(1),
                        2,
                        qk_thr_mma,
                        acc_pv,
                        qk_tiled_mma,
                        pv_tiled_mma,
                        load_kv_consumer,
                        q_handle,
                        tSrQ,
                        tSrK,
                        s_max,
                        a_sum,
                        tOrV,
                        p_a,
                        p_b,
                        p_c,
                        scale_softmax_log2,
                        qk_acc_shape,
                    )
                    load_kv_consumer, s_max, a_sum = self.dstep_loop(
                        full_tiles - 1,
                        3,
                        qk_thr_mma,
                        acc_pv,
                        qk_tiled_mma,
                        pv_tiled_mma,
                        load_kv_consumer,
                        q_handle,
                        tSrQ,
                        tSrK,
                        s_max,
                        a_sum,
                        tOrV,
                        p_a,
                        p_b,
                        p_c,
                        scale_softmax_log2,
                        qk_acc_shape,
                    )
                    load_kv_consumer = self.final_pv(
                        3,
                        pv_tiled_mma,
                        load_kv_consumer,
                        acc_pv,
                        tOrV,
                        p_a,
                        p_b,
                        p_c,
                    )

            # ---- tail: cross-thread sum reduction + normalize ----
            self.tail(s_max, a_sum, acc_pv, pv_tiled_mma)

            # ---- epilogue: R2S (StMatrix) -> TMA store ----
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.o_layout, elem_ty_d=self.o_dtype, elem_ty_acc=self.pv_acc_dtype
            )
            copy_atom_O = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(self.o_layout.is_m_major_c(), 4),
                self.o_dtype,
            )
            tiled_copy_O_Atom = cute.make_tiled_copy_C_atom(copy_atom_O, pv_tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_O_Atom)
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx % self.num_threads_per_warp_group
            )
            tRS_sD = thr_copy_r2s.partition_D(sO)
            tRS_rAcc = tiled_copy_r2s.retile(acc_pv)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sO))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor_like(tRS_rD_layout, self.pv_acc_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            gD = cute.local_tile(mO_qdl, (64, self.output_dim), (bidx, dpart, bidy))
            sepi_for_tma_partition = cute.group_modes(sO, 0, 2)
            tcgc_for_tma_partition = cute.zipped_divide(gD, self.epi_tile)
            bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                tma_atom_o,
                0,
                cute.make_layout(1),
                sepi_for_tma_partition,
                tcgc_for_tma_partition,
            )
            epi_tile_num = cute.size(tcgc_for_tma_partition, mode=[1])

            for epi_idx in cutlass.range(epi_tile_num, unroll_full=True):
                for epi_v in cutlass.range(size_tRS_rD, unroll_full=True):
                    tRS_rD[epi_v] = tRS_rAcc[epi_idx * size_tRS_rD + epi_v]

                tRS_rD_out = cute.make_rmem_tensor_like(tRS_rD_layout, self.o_dtype)
                acc_vec = tRS_rD.load()
                tRS_rD_out.store(acc_vec.to(self.o_dtype))

                epi_buffer = epi_idx % self.epi_stage
                cute.copy(
                    tiled_copy_r2s,
                    tRS_rD_out,
                    tRS_sD[(None, None, None, epi_buffer)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline.arrive_and_wait(
                    barrier_id=warp_group_idx,
                    num_threads=self.num_threads_per_warp_group,
                )
                if warp_idx == 4:
                    cute.copy(
                        tma_atom_o,
                        bSG_sD[(None, epi_buffer)],
                        bSG_gD[(None, epi_idx)],
                    )
                    tma_store_pipeline.producer_commit()
                    tma_store_pipeline.producer_acquire()
                pipeline.arrive_and_wait(
                    barrier_id=warp_group_idx,
                    num_threads=self.num_threads_per_warp_group,
                )
            # Wait until both TMA epilogue stores have finished reading SMEM.
            if warp_idx == 4:
                tma_store_pipeline.producer_tail()
        return

    @cute.jit
    def prologue(
        self,
        num_frags: cutlass.Constexpr,
        qk_thr_mma: cute.ThrMma,
        acc_pv: cute.Tensor,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        load_kv_consumer: pipeline.PipelineConsumer,
        q_handle: pipeline.PipelineConsumer.ImmutableResourceHandle,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        p_a: cute.Tensor,
        p_b: cute.Tensor,
        p_c: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        qk_acc_shape: cute.Shape,
    ):
        ps = [p_a, p_b, p_c][:num_frags]
        accs = [qk_thr_mma.make_fragment_C(qk_acc_shape) for _ in range(num_frags)]
        k_handles = [load_kv_consumer.wait_and_advance() for _ in range(num_frags)]

        for f in cutlass.range_constexpr(num_frags):
            accs[f].fill(0.0)
        cute.nvgpu.warpgroup.fence()
        for f in cutlass.range_constexpr(num_frags):
            cute.gemm(
                qk_tiled_mma,
                accs[f],
                tSrQ[(None, None, None, q_handle.index)],
                tSrK[(None, None, None, k_handles[f].index)],
                accs[f],
            )
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        for f in cutlass.range_constexpr(num_frags):
            k_handles[f].release()

        scale = self.softmax_scale_step(
            accs, qk_tiled_mma, s_max, a_sum, scale_softmax_log2
        )
        for f in cutlass.range_constexpr(num_frags):
            self.convert_acc_into(ps[f], accs[f])
        self.rescale_o(acc_pv, pv_tiled_mma, scale)

        return load_kv_consumer, s_max, a_sum

    @cute.jit
    def dstep_loop(
        self,
        step_count: cutlass.Int32,
        prev_num_frags: cutlass.Constexpr,
        qk_thr_mma: cute.ThrMma,
        acc_pv: cute.Tensor,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        load_kv_consumer: pipeline.PipelineConsumer,
        q_handle: pipeline.PipelineConsumer.ImmutableResourceHandle,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        tOrV: cute.Tensor,
        p_a: cute.Tensor,
        p_b: cute.Tensor,
        p_c: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
        qk_acc_shape: cute.Shape,
    ):
        """Deferred-PV steps: QK of tile t, PV of tile t+1 (P held in p_a..p_c),
        softmax of t, rescale acc after the PV add — FA3 fwd_step ordering."""
        prev_ps = [p_a, p_b, p_c][:prev_num_frags]
        while step_count > 0:
            step_count -= 1

            accs = [qk_thr_mma.make_fragment_C(qk_acc_shape) for _ in range(3)]
            k_handles = [load_kv_consumer.wait_and_advance() for _ in range(3)]

            for f in cutlass.range_constexpr(3):
                accs[f].fill(0.0)
            cute.nvgpu.warpgroup.fence()
            for f in cutlass.range_constexpr(3):
                cute.gemm(
                    qk_tiled_mma,
                    accs[f],
                    tSrQ[(None, None, None, q_handle.index)],
                    tSrK[(None, None, None, k_handles[f].index)],
                    accs[f],
                )
            cute.nvgpu.warpgroup.commit_group()

            v_handles = [
                load_kv_consumer.wait_and_advance() for _ in range(prev_num_frags)
            ]
            cute.nvgpu.warpgroup.fence()
            for f in cutlass.range_constexpr(prev_num_frags):
                cute.gemm(
                    pv_tiled_mma,
                    acc_pv,
                    prev_ps[f],
                    tOrV[(None, None, None, v_handles[f].index)],
                    acc_pv,
                )
            cute.nvgpu.warpgroup.commit_group()
            # QK retires first; keep PV in flight across the independent
            # max/exp/sum work. P and O remain untouched until the final wait.
            cute.nvgpu.warpgroup.wait_group(1)

            for f in cutlass.range_constexpr(3):
                k_handles[f].release()

            scale = self.softmax_scale_step(
                accs, qk_tiled_mma, s_max, a_sum, scale_softmax_log2
            )
            cute.nvgpu.warpgroup.wait_group(0)
            for f in cutlass.range_constexpr(prev_num_frags):
                v_handles[f].release()
            self.convert_acc_into(p_a, accs[0])
            self.convert_acc_into(p_b, accs[1])
            self.convert_acc_into(p_c, accs[2])
            self.rescale_o(acc_pv, pv_tiled_mma, scale)

        return load_kv_consumer, s_max, a_sum

    @cute.jit
    def final_pv(
        self,
        num_frags: cutlass.Constexpr,
        pv_tiled_mma: cute.TiledMma,
        load_kv_consumer: pipeline.PipelineConsumer,
        acc_pv: cute.Tensor,
        tOrV: cute.Tensor,
        p_a: cute.Tensor,
        p_b: cute.Tensor,
        p_c: cute.Tensor,
    ):
        ps = [p_a, p_b, p_c][:num_frags]
        v_handles = [load_kv_consumer.wait_and_advance() for _ in range(num_frags)]
        cute.nvgpu.warpgroup.fence()
        for f in cutlass.range_constexpr(num_frags):
            cute.gemm(
                pv_tiled_mma,
                acc_pv,
                ps[f],
                tOrV[(None, None, None, v_handles[f].index)],
                acc_pv,
            )
        cute.nvgpu.warpgroup.commit_group()
        cute.nvgpu.warpgroup.wait_group(0)
        for f in cutlass.range_constexpr(num_frags):
            v_handles[f].release()
        return load_kv_consumer

    @cute.jit
    def softmax_scale_step(
        self,
        accs,
        tiled_mma_qk: cute.TiledMma,
        s_max: cute.Tensor,
        a_sum: cute.Tensor,
        scale_softmax_log2: cutlass.Float32,
    ):
        """FA3 max_get_scale + online_softmax: updates s_max/a_sum, exp2s the
        fragments in place, and returns the per-row rescale factors. Does NOT
        touch the PV accumulator (the caller rescales after the deferred PV)."""
        mns = [
            cute.make_tensor(a.iterator, self.layout_acc_mn(tiled_mma_qk, a.layout))
            for a in accs
        ]
        reduction_target_qk = self.reduction_target_n(tiled_mma_qk)
        red_rank = cute.rank(reduction_target_qk)
        s_max_prev = cute.make_rmem_tensor_like(s_max, s_max._dtype)
        scale = cute.make_rmem_tensor_like(s_max, cutlass.Float32)

        n_cols = cute.size(mns[0], mode=[1])
        n_frags = len(mns)
        if scale_softmax_log2 < 0.0:
            for f in cutlass.range_constexpr(n_frags):
                for i in cutlass.range(cute.size(mns[0], mode=[0]), unroll_full=True):
                    for j in cutlass.range(n_cols, unroll_full=True):
                        mns[f][i, j] = -mns[f][i, j]
        scale_softmax_log2 = abs(scale_softmax_log2)
        for i in cutlass.range(cute.size(mns[0], mode=[0]), unroll_full=True):
            s_max_prev[i] = s_max[i]
            for f in cutlass.range_constexpr(n_frags):
                s_max[i] = (
                    mns[f][i, None].load().reduce(cute.ReductionOp.MAX, s_max[i], 0)
                )

            for r in cutlass.range_constexpr(red_rank):
                s_max[i] = cute.arch.warp_reduction_max(
                    s_max[i], threads_in_group=reduction_target_qk.shape[r]
                )

            scale[i] = cute.math.exp2(
                (s_max_prev[i] - s_max[i]) * scale_softmax_log2, fastmath=True
            )
            if s_max_prev[i] == -cutlass.Float32.inf:
                scale[i] = 0.0
            a_sum[i] *= scale[i]

            # FA3 softmax.h numerics: max_scaled = max*scale,
            # p = exp2(fma(x, scale, -max_scaled)), sequential row-sum adds
            neg_scale_max = -(s_max[i] * scale_softmax_log2)
            for f in cutlass.range_constexpr(n_frags):
                for j in cutlass.range(n_cols, unroll_full=True):
                    mns[f][i, j] = cute.math.exp2(
                        cute.fma(mns[f][i, j], scale_softmax_log2, neg_scale_max),
                        fastmath=True,
                    )
            for f in cutlass.range_constexpr(n_frags):
                for j in cutlass.range(n_cols, unroll_full=True):
                    a_sum[i] = a_sum[i] + mns[f][i, j]

        return scale

    @cute.jit
    def rescale_o(self, acc_pv, tiled_mma_pv, scale):
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator, self.layout_acc_mn(tiled_mma_pv, acc_pv.layout)
        )
        for i in cutlass.range(cute.size(acc_pv_mn, mode=[0]), unroll_full=True):
            for j in cutlass.range(cute.size(acc_pv_mn, mode=[1]), unroll_full=True):
                acc_pv_mn[i, j] *= scale[i]

    @cute.jit
    def convert_acc_into(self, operand, acc):
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        acc_vec = acc.load()
        operand_as_acc.store(acc_vec.to(operand.element_type))

    @cute.jit
    def tail(self, s_max, a_sum, acc_pv, tiled_mma_pv):
        acc_pv_mn = cute.make_tensor(
            acc_pv.iterator, self.layout_acc_mn(tiled_mma_pv, acc_pv.layout)
        )
        reduction_target = self.reduction_target_n(tiled_mma_pv)
        red_rank = cute.rank(reduction_target)
        for r in cutlass.range_constexpr(red_rank):
            for i in cutlass.range(cute.size(acc_pv_mn, mode=[0]), unroll_full=True):
                a_sum[i] = cute.arch.warp_reduction_sum(
                    a_sum[i], threads_in_group=reduction_target.shape[r]
                )

        for i in cutlass.range(cute.size(acc_pv_mn, mode=[0]), unroll_full=True):
            sum_i = a_sum[i]
            # FA3 builds with --use_fast_math: its 1.f/sum is an approximate
            # reciprocal, so rcp.approx is the parity-correct choice here.
            inv_sum = cute.arch.rcp_approx(sum_i)
            if sum_i == 0.0 or sum_i != sum_i:
                inv_sum = 0.0
            for j in cutlass.range(cute.size(acc_pv_mn, mode=[1]), unroll_full=True):
                acc_pv_mn[i, j] *= inv_sum

    @cute.jit
    def reduction_target_n(self, tiled_mma):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0],
            cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
            tiled_mma.tv_layout_C.stride[0],
        )
        return separated[1]

    @staticmethod
    def convert_c_layout_to_a_layout(c, a):
        return cute.make_layout(
            (a, c.shape[1], (c.shape[2], cute.size(c, mode=[0]) // cute.size(a))),
            stride=(
                c.stride[0],
                c.stride[1],
                (c.stride[2], cute.size(a, mode=[2]) * c.stride[0][2]),
            ),
        )

    @cute.jit
    def make_acc_into_op(self, acc, operand_layout_tv, Element):
        operand = cute.make_rmem_tensor_like(
            self.convert_c_layout_to_a_layout(acc.layout, operand_layout_tv.shape[1]),
            Element,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        acc_vec = acc.load()
        operand_as_acc.store(acc_vec.to(Element))
        return operand

    @staticmethod
    def layout_separate(thr, src, ref):
        lt = cute.make_layout(())
        ge = cute.make_layout(())
        for kk, vv in enumerate(ref):
            if cutlass.const_expr(vv < thr):
                lt = cute.append(lt, src[kk])
            else:
                ge = cute.append(ge, src[kk])
        r = None
        if cutlass.const_expr(cute.rank(lt) == 1):
            r = cute.append(lt, ge)
        else:
            r = cute.append(cute.append(cute.make_layout(()), lt), ge)
        return r

    @staticmethod
    @cute.jit
    def gemm_zero_acc(tiled_mma, A, B, C):
        rA = cute.rank(A)
        rB = cute.rank(B)
        rC = cute.rank(C)
        if cutlass.const_expr(rA == 3 and rB == 3 and rC == 3):
            for k_block_idx in range(cute.size(A, mode=[2]), unroll_full=True):
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, k_block_idx != 0)
                cute.gemm(
                    tiled_mma,
                    C,
                    A[None, None, k_block_idx],
                    B[None, None, k_block_idx],
                    C,
                )
        else:
            assert 0

    @cute.jit
    def layout_acc_mn(self, tiled_mma, acc):
        separated = self.layout_separate(
            tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1]
        )
        V_M = separated[0]
        V_N = separated[1]
        V_M1 = None
        V_N1 = None
        if cutlass.const_expr(cute.rank(V_M) == 1):
            V_M1 = cute.append(V_M, acc[1])
        else:
            V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])
        if cutlass.const_expr(cute.rank(V_N) == 1):
            V_N1 = cute.append(V_N, acc[2])
        else:
            V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
        r = None
        if cutlass.const_expr(cute.rank(V_M1) == 1):
            r = cute.append(V_M1, V_N1)
        else:
            r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
        return r

    def make_and_init_load_q_pipeline(self, load_q_mbar_ptr):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_q_mbar_ptr,
            num_stages=self.q_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_q_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_load_kv_pipeline(self, load_kv_mbar_ptr):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        return pipeline.PipelineTmaAsync.create(
            barrier_storage=load_kv_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=self.tma_copy_kv_bytes,
            defer_sync=True,
        ).make_participants()

    def make_and_init_tma_store_pipeline(self):
        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        return pipeline.PipelineTmaStore.create(
            num_stages=self.epi_stage,
            producer_group=producer_group,
        )

    @staticmethod
    def _make_tma_atoms_and_tensors(tensor, smem_layout_staged, smem_tile):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            smem_layout,
            smem_tile,
        )
        return tma_atom, tma_tensor


# ---------------------------------------------------------------------------
# Host layer
# ---------------------------------------------------------------------------

_compiled = {}  # Code only; descriptors are owned by the wrapper plan.


def run_prepared(q, k, v, idx_gpu, cnt_gpu, order, sm_scale, out):
    scale_log2 = float(sm_scale) * LOG2_E
    caller_stream = torch.cuda.current_stream(q.device)
    stream = cuda.CUstream(caller_stream.cuda_stream)
    qv = from_dlpack(q.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    kv = from_dlpack(k.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    vv = from_dlpack(v.permute(2, 1, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=0
    )
    ov = from_dlpack(out.permute(1, 2, 0), assumed_align=16).mark_layout_dynamic(
        leading_dim=1
    )
    iv = from_dlpack(idx_gpu, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    cv = from_dlpack(cnt_gpu, assumed_align=4).mark_layout_dynamic(leading_dim=1)

    ov_order = from_dlpack(order, assumed_align=8).mark_layout_dynamic(leading_dim=0)
    use_order = order.numel() > 0
    output_dim = 32 if q.shape[0] * (q.shape[1] // 64) <= 32 else 64
    compile_key = (q.device.index, use_order, output_dim)
    if compile_key not in _compiled:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the VSA kernel before CUDA graph capture")
        obj = VsaFwdSm90(output_dim=output_dim)
        _compiled[compile_key] = cute.compile(
            obj,
            qv,
            kv,
            vv,
            ov,
            iv,
            cv,
            ov_order,
            use_order,
            cutlass.Float32(scale_log2),
            stream,
        )

    _compiled[compile_key](
        qv, kv, vv, ov, iv, cv, ov_order, cutlass.Float32(scale_log2), stream
    )
