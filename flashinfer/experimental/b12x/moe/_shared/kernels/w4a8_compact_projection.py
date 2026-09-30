"""Independent compact W4A8 FC1 gate/up projection kernel."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    e2m1x8_to_qmma_e2m1x8,
    get_ptr_as_int64,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    mxfp8_mma_m16n8k32_f32_e2m1,
    shared_ptr_to_u32,
    st_shared_u32,
)


class W4A8CompactMicroProjectionKernel:
    """Project N128 channels, broadcasting one staged activation row into QMMA."""

    tile_m = 16
    tile_n = 128
    tile_k = 64
    stages = 4
    num_warps = 4
    threads_per_cta = 128

    def __init__(self, k: int, n: int, num_topk: int):
        if k <= 0 or n < 64 or k % 128 or n % 64:
            raise ValueError(
                "compact W4A8 projection requires K divisible by 128 and "
                "N divisible by 64"
            )
        if num_topk <= 0:
            raise ValueError("num_topk must be positive")
        self.k = int(k)
        self.n = int(n)
        self.num_topk = int(num_topk)
        self.n_tiles = (self.n + self.tile_n - 1) // self.tile_n
        self.n64_tail = self.n % self.tile_n == 64
        self.b_payload_bytes = self.tile_n * self.tile_k // 2
        self.sfb_bytes = (self.tile_n // 8) * 8 * 4
        self.a_offset = 0
        self.sfa_offset = 64
        self.b_offset = 128
        self.sfb_offset = self.b_offset + self.b_payload_bytes
        self.stage_bytes = self.sfb_offset + self.sfb_bytes
        self.shared_words = self.stages * self.stage_bytes // 4

    @cute.jit
    def __call__(
        self,
        a8: cute.Tensor,
        scale_rows: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
        max_active_clusters: Int32,
        stream: cuda.CUstream,
    ):
        a_words = cute.recast_tensor(a8, cutlass.Uint32)
        projected_flat = cute.make_tensor(
            projections.iterator,
            cute.make_layout((projections.shape[0] * projections.shape[1],)),
        )
        grid = num_pairs * Int32(2 * self.n_tiles)
        if max_active_clusters > Int32(0):
            grid = cutlass.min(grid, max_active_clusters * Int32(2))
        self.kernel(
            a_words,
            scale_rows,
            w13_rp,
            w13_sfb,
            projected_flat,
            topk_ids,
            alpha,
            input_scale,
            num_pairs,
        ).launch(
            grid=(grid, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.jit
    def _stage_b_k64(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        k_half: Int32,
        tid: Int32,
    ):
        # Prepared RP is [K32, N32-chunk, lane, n8-in-chunk].  A CTA owns
        # one N128 half, retaining coalesced four-word lane loads.
        for i in cutlass.range_constexpr(2):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(256):
                lane = idx & Int32(31)
                kc = idx >> Int32(5)
                chunk = kc % Int32(4)
                kb = kc // Int32(4) + k_half * Int32(2)
                src_word = (
                    tile_word_base
                    + Int64(kb * Int32(8 * 32 * 4))
                    + Int64((packed_half * Int32(4) + chunk) * Int32(32 * 4))
                    + Int64(lane * Int32(4))
                )
                cp_async4_shared_global(
                    dst_base + (idx << Int32(4)), get_ptr_as_int64(weights, src_word)
                )

    @cute.jit
    def _stage_sfb(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        tid: Int32,
    ):
        if tid < Int32(32):
            src_word = tile_word_base + Int64(packed_half * Int32(128) + tid * 4)
            cp_async4_shared_global(
                dst_base + (tid << Int32(4)), get_ptr_as_int64(scales, src_word)
            )

    @cute.jit
    def _stage_n64_b_k64(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        projection: Int32,
        output_tile: Int32,
        k64_slice: Int32,
        tid: Int32,
    ):
        n_rows = Int32(128)
        if output_tile == Int32(self.n_tiles - 1):
            n_rows = Int32(64)
        for i in cutlass.range_constexpr(2):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(256):
                lane = idx & Int32(31)
                tmp = idx >> Int32(5)
                n32 = tmp & Int32(3)
                kb = tmp >> Int32(2)
                dst_transfer = (kb * Int32(4) + n32) * Int32(32) + lane
                dst_addr = dst_base + (dst_transfer << Int32(4))
                if n32 < n_rows // Int32(32):
                    source_word = (
                        Int64(expert_idx) * Int64(self.n * (self.k // 32) * 8)
                        + Int64(projection) * Int64(self.n * (self.k // 32) * 4)
                        + Int64(output_tile) * Int64(128 * (self.k // 32) * 4)
                        + Int64(k64_slice * Int32(2) + kb)
                        * Int64(n_rows // Int32(32))
                        * Int64(128)
                        + Int64(n32 * Int32(128) + lane * Int32(4))
                    )
                    cp_async4_shared_global(
                        dst_addr,
                        get_ptr_as_int64(weights, source_word),
                    )
                else:
                    st_shared_u32(dst_addr, Uint32(0))
                    st_shared_u32(dst_addr + Int32(4), Uint32(0))
                    st_shared_u32(dst_addr + Int32(8), Uint32(0))
                    st_shared_u32(dst_addr + Int32(12), Uint32(0))

    @cute.jit
    def _stage_n64_sfb(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        projection: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        tid: Int32,
    ):
        n_rows = Int32(128)
        if output_tile == Int32(self.n_tiles - 1):
            n_rows = Int32(64)
        if tid < Int32(32):
            dst_addr = dst_base + (tid << Int32(4))
            if tid < n_rows // Int32(4):
                source_word = (
                    Int64(expert_idx) * Int64(self.n * (self.k // 32) // 2)
                    + Int64(projection) * Int64(self.n * (self.k // 32) // 4)
                    + Int64(output_tile) * Int64(128 * (self.k // 32) // 4)
                    + Int64(k128_slice * n_rows + tid * Int32(4))
                )
                cp_async4_shared_global(
                    dst_addr,
                    get_ptr_as_int64(scales, source_word),
                )
            else:
                st_shared_u32(dst_addr, Uint32(0))
                st_shared_u32(dst_addr + Int32(4), Uint32(0))
                st_shared_u32(dst_addr + Int32(8), Uint32(0))
                st_shared_u32(dst_addr + Int32(12), Uint32(0))

    @cute.jit
    def _stage_slice(
        self,
        a_words: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pair: Int32,
        expert: Int32,
        projection_tile: Int32,
        k64_slice: Int32,
    ):
        stage_base = smem_base + (k64_slice & Int32(3)) * Int32(self.stage_bytes)
        token = pair // Int32(self.num_topk)
        if tid < Int32(4):
            src_word = (
                Int64(token) * Int64(self.k // 4)
                + Int64(k64_slice) * Int64(16)
                + Int64(tid * 4)
            )
            cp_async4_shared_global(
                stage_base + Int32(self.a_offset) + tid * Int32(16),
                get_ptr_as_int64(a_words, src_word),
            )
        if tid == Int32(0):
            sf_src = Int64(token) * Int64(self.k // 32) + Int64(k64_slice >> 1) * Int64(
                4
            )
            cp_async_u32_shared_global(
                stage_base + Int32(self.sfa_offset),
                get_ptr_as_int64(scale_words, sf_src),
            )
        b_base = stage_base + Int32(self.b_offset)
        sfb_base = stage_base + Int32(self.sfb_offset)
        k128_slice = k64_slice >> Int32(1)
        if cutlass.const_expr(self.n64_tail):
            projection = projection_tile // Int32(self.n_tiles)
            output_tile = projection_tile % Int32(self.n_tiles)
            self._stage_n64_b_k64(
                w13_rp,
                b_base,
                expert,
                projection,
                output_tile,
                k64_slice,
                tid,
            )
            self._stage_n64_sfb(
                w13_sfb,
                sfb_base,
                expert,
                projection,
                output_tile,
                k128_slice,
                tid,
            )
        else:
            packed_tile = projection_tile // Int32(2)
            packed_half = projection_tile % Int32(2)
            packed_tile_index = (
                Int64(expert) * Int64(self.n // 128) + Int64(packed_tile)
            ) * Int64(self.k // 128) + Int64(k128_slice)
            tile_word_base = packed_tile_index * Int64(4096)
            self._stage_b_k64(
                w13_rp,
                b_base,
                tile_word_base,
                packed_half,
                k64_slice & Int32(1),
                tid,
            )
            scale_word_base = packed_tile_index * Int64(256)
            self._stage_sfb(
                w13_sfb,
                sfb_base,
                scale_word_base,
                packed_half,
                tid,
            )

    @cute.jit
    def _run_task(
        self,
        a_words: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        warp: Int32,
        pair: Int32,
        expert: Int32,
        projection_tile: Int32,
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)
        for preload in cutlass.range_constexpr(3):
            if cutlass.const_expr(preload < self.k // 64):
                self._stage_slice(
                    a_words,
                    scale_words,
                    w13_rp,
                    w13_sfb,
                    smem_base,
                    tid,
                    pair,
                    expert,
                    projection_tile,
                    Int32(preload),
                )
            cute.arch.cp_async_commit_group()

        acc = tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _ in range(4))
        for nt in cutlass.range_constexpr(4):
            acc[nt].fill(0.0)
        k64_slice = Int32(0)
        k64_tiles = Int32(self.k // 64)
        while k64_slice < k64_tiles:
            stage_base = smem_base + (k64_slice & Int32(3)) * Int32(self.stage_bytes)
            b_base = stage_base + Int32(self.b_offset)
            sfb_base = stage_base + Int32(self.sfb_offset)
            next_slice = k64_slice + Int32(3)
            if next_slice < k64_tiles:
                self._stage_slice(
                    a_words,
                    scale_words,
                    w13_rp,
                    w13_sfb,
                    smem_base,
                    tid,
                    pair,
                    expert,
                    projection_tile,
                    next_slice,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(3)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()
            scale_shift = Uint32(k64_slice & Int32(1)) * Uint32(16)
            asc = ld_shared_u32(stage_base + Int32(self.sfa_offset)) >> scale_shift
            for kb in cutlass.range_constexpr(2):
                # One routed token is replicated across the logical M16
                # operand. Load its lane fragment once, not sixteen SMEM rows.
                a_src = (
                    stage_base + Int32(self.a_offset) + Int32(kb * 32) + c * Int32(8)
                )
                a0, a2 = ld_shared_v2_u32(a_src)
                a1, a3 = a0, a2
                b_words = cute.make_rmem_tensor((4,), Uint32)
                bw0, bw1, bw2, bw3 = ld_shared_v4_u32(
                    b_base + (((Int32(kb * 4) + warp) * Int32(32) + lane) << Int32(4))
                )
                b_words[0] = bw0
                b_words[1] = bw1
                b_words[2] = bw2
                b_words[3] = bw3
                for nt in cutlass.range_constexpr(4):
                    b0, b1 = e2m1x8_to_qmma_e2m1x8(b_words[nt])
                    n8 = warp * Int32(4) + Int32(nt)
                    sfb = (
                        ld_shared_u32(sfb_base + ((n8 * Int32(8) + q) << Int32(2)))
                        >> scale_shift
                    )
                    fragment = acc[nt]
                    x0, x1, x2, x3 = mxfp8_mma_m16n8k32_f32_e2m1(
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        cutlass.Float32(0.0),
                        a0,
                        a1,
                        a2,
                        a3,
                        b0,
                        b1,
                        asc,
                        sfb,
                        bid_a=kb,
                        bid_b=kb,
                    )
                    fragment[0] = fragment[0] + x0
                    fragment[1] = fragment[1] + x1
                    fragment[2] = fragment[2] + x2
                    fragment[3] = fragment[3] + x3
            cute.arch.sync_threads()
            k64_slice += Int32(1)
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        scale_expert = expert
        if cutlass.const_expr(input_scale.shape[0] == 1):
            scale_expert = Int32(0)
        value_scale = alpha[expert].to(cutlass.Float32) * input_scale[
            Int64(scale_expert)
        ].to(cutlass.Float32)
        # The prepared RP stores up first.  The seam is canonical gate then up.
        source_tile = projection_tile % Int32(self.n_tiles)
        dst_base = source_tile * Int32(self.tile_n)
        if projection_tile < Int32(self.n_tiles):
            dst_base += Int32(self.n)
        if q == Int32(0):
            col_base = warp * Int32(32) + (c << Int32(1))
            for nt in cutlass.range_constexpr(4):
                tile_col = col_base + Int32(nt * 8)
                if source_tile * Int32(self.tile_n) + tile_col < Int32(self.n):
                    col = dst_base + tile_col
                    fragment = acc[nt]
                    projections[Int64(pair) * Int64(2 * self.n) + Int64(col)] = (
                        value_scale * fragment[0]
                    ).to(cutlass.BFloat16)
                    projections[
                        Int64(pair) * Int64(2 * self.n) + Int64(col + Int32(1))
                    ] = (value_scale * fragment[1]).to(cutlass.BFloat16)

    @cute.kernel
    def kernel(
        self,
        a_words: cute.Tensor,
        scale_words: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb: cute.Tensor,
        projections: cute.Tensor,
        topk_ids: cute.Tensor,
        alpha: cute.Tensor,
        input_scale: cute.Tensor,
        num_pairs: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        tid = Int32(tidx)
        task = Int32(bidx)
        projection_tiles = Int32(2 * self.n_tiles)
        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words], 1024
            ]

        storage = smem.allocate(Storage)
        while task < num_pairs * projection_tiles:
            pair = task // projection_tiles
            projection_tile = task % projection_tiles
            expert = topk_ids[Int64(pair)].to(Int32)
            if expert >= Int32(0) and expert < Int32(alpha.shape[0]):
                self._run_task(
                    a_words,
                    scale_words,
                    w13_rp,
                    w13_sfb,
                    projections,
                    alpha,
                    input_scale,
                    shared_ptr_to_u32(storage.words.data_ptr()),
                    tid,
                    cute.arch.make_warp_uniform(cute.arch.warp_idx()),
                    pair,
                    expert,
                    projection_tile,
                )
            task += Int32(gdimx)


__all__ = ["W4A8CompactMicroProjectionKernel"]
