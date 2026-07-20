# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/w4a8_phase2.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Standalone materialized-W4A8 FC2 kernel for dense prefill regimes.

The dynamic front-end and phase 1 publish an expert-major M64 or M128 source
domain in MXFP8 form.  Keeping FC2 in the routing kernel would union phase-A
and phase-B shared-memory/register requirements.  This kernel instead consumes
the caller-owned materialization workspace as M64xN128 tasks.  The launch is
fixed-capacity and stream ordered, so it remains safe for CUDA graph capture
and replay without host reads or allocations.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32, Int64, Uint32

from flashinfer.experimental.sm12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    e2m1x8_to_qmma_e2m1x8,
    get_ptr_as_int64,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    mxfp8_mma_m16n8k32_f32_e2m1,
    pack_f32x2_to_bfloat2,
    scatter_add_bf16x2,
    shared_ptr_to_u32,
    st_global_u32,
)


class W4A8MaterializedPhase2Kernel:
    """Consume materialization through compact M64xN128 FC2 CTAs."""

    tile_m = 64
    source_tile_m = 128
    tile_n = 128
    tile_k = 128
    num_warps = 4
    threads_per_cta = num_warps * 32
    stages = 2

    # Per stage: 64x128 E4M3 A, 64 packed A-scale words, N128xK128 FP4 B,
    # and 16x8 packed B-scale words.  Keep the large regions 1 KiB aligned.
    a_payload_bytes = tile_m * tile_k
    a_scale_bytes = tile_m * 4
    a_stage_bytes = a_payload_bytes + a_scale_bytes
    a_storage_bytes = stages * a_stage_bytes
    b_storage_offset = ((a_storage_bytes + 1023) // 1024) * 1024
    b_stage_bytes = tile_n * tile_k // 2
    b_storage_bytes = stages * b_stage_bytes
    sfb_storage_offset = b_storage_offset + b_storage_bytes
    sfb_stage_bytes = (tile_n // 8) * 8 * 4
    shared_bytes = sfb_storage_offset + stages * sfb_stage_bytes
    shared_words = (shared_bytes + 3) // 4

    def __init__(
        self,
        *,
        source_tile_m: int = 128,
        deterministic_output: bool = False,
    ):
        if source_tile_m not in (64, 128):
            raise ValueError(
                f"materialized phase 2 source_tile_m must be 64 or 128, got {source_tile_m}"
            )
        self.source_tile_m = int(source_tile_m)
        self.source_halves = self.source_tile_m // self.tile_m
        self.deterministic_output = bool(deterministic_output)

    @cute.jit
    def __call__(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        intermediate_tiles: cutlass.Int32,
        packed_output_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # Two compact phase-B CTAs can reside per SM.  The grid is a fixed
        # multiple of the preplanned resident-grid capacity; the device-side
        # tail controls useful work and never changes the captured launch.
        self.kernel(
            intermediate_u32,
            down_rp,
            down_sfb_rp,
            scatter_output,
            token_map,
            token_weights,
            task_expert,
            task_valid_rows,
            expert_tile_base,
            down_alpha,
            global_scale,
            intermediate_tiles,
            packed_output_tiles,
        ).launch(
            grid=(1, 1, max_active_clusters * Int32(2)),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.jit
    def _stage_slice(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        source_m_tile: Int32,
        m_half: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        intermediate_slice: Int32,
        rows_capacity: Int32,
        intermediate_tiles: Int32,
        packed_output_tiles: Int32,
    ):
        stage = intermediate_slice & Int32(1)
        a_base = smem_base + stage * Int32(self.a_stage_bytes)
        sfa_base = a_base + Int32(self.a_payload_bytes)
        b_base = (
            smem_base + Int32(self.b_storage_offset) + stage * Int32(self.b_stage_bytes)
        )
        sfb_base = (
            smem_base
            + Int32(self.sfb_storage_offset)
            + stage * Int32(self.sfb_stage_bytes)
        )

        words_per_row = intermediate_tiles * Int32(32)
        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(
            self.tile_m
        )

        # A payload: 64 rows x eight 16-byte vectors.  Materialization is
        # row-major; shared memory XORs vectors by row to match QMMA loads.
        for i in cutlass.range_constexpr(
            (self.tile_m * 8 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(self.tile_m * 8):
                row = idx >> Int32(3)
                vec = idx & Int32(7)
                physical_vec = vec ^ (row & Int32(7))
                src_word = (
                    (physical_row_base + row) * words_per_row
                    + intermediate_slice * Int32(32)
                    + (vec << Int32(2))
                )
                cp_async4_shared_global(
                    a_base + row * Int32(self.tile_k) + (physical_vec << Int32(4)),
                    get_ptr_as_int64(intermediate_u32, src_word),
                )

        # One packed K128 scale word per activation row.
        if tid < Int32(self.tile_m):
            sf_src = (
                rows_capacity * words_per_row
                + intermediate_slice * rows_capacity
                + physical_row_base
                + tid
            )
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(intermediate_u32, sf_src),
            )

        # Prepared weights are N256 tile-major.  Select the N128 half and
        # compact its four K32 chunks into the phase-B shared tile.
        packed_tile = output_tile >> Int32(1)
        packed_half = output_tile & Int32(1)
        b_tile = (
            expert_idx * packed_output_tiles + packed_tile
        ) * intermediate_tiles + intermediate_slice
        b_word_base = Int64(b_tile) * Int64(4096)
        for i in cutlass.range_constexpr(
            (4 * 4 * 32 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(4 * 4 * 32):
                lane = idx & Int32(31)
                kc = idx >> Int32(5)
                chunk = kc & Int32(3)
                kb = kc >> Int32(2)
                src_word = (
                    b_word_base
                    + Int64(kb * 8 * 32 * 4)
                    + Int64((packed_half * Int32(4) + chunk) * Int32(32 * 4))
                    + Int64(lane * Int32(4))
                )
                cp_async4_shared_global(
                    b_base + (idx << Int32(4)),
                    get_ptr_as_int64(down_rp, src_word),
                )

        # The N128 half contains 16 n8 scale rows, eight u32 words each.
        sfb_word_base = Int64(b_tile) * Int64(256)
        for i in cutlass.range_constexpr(
            ((16 * 8) // 4 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32((16 * 8) // 4):
                src_word = sfb_word_base + Int64(packed_half * Int32(16 * 8) + idx * 4)
                cp_async4_shared_global(
                    sfb_base + (idx << Int32(4)),
                    get_ptr_as_int64(down_sfb_rp, src_word),
                )

    @cute.jit
    def _run_task(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        warp_idx: Int32,
        source_m_tile: Int32,
        m_half: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        valid_rows: Int32,
        rows_capacity: Int32,
        intermediate_tiles: Int32,
        packed_output_tiles: Int32,
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)

        self._stage_slice(
            intermediate_u32,
            down_rp,
            down_sfb_rp,
            smem_base,
            tid,
            source_m_tile,
            m_half,
            expert_idx,
            output_tile,
            Int32(0),
            rows_capacity,
            intermediate_tiles,
            packed_output_tiles,
        )
        cute.arch.cp_async_commit_group()

        # Each of four warps owns M64xN32: four M16 blocks by four N8
        # fragments.  This is 64 FP32 accumulator registers per thread.
        facc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(4))
            for _blk in range(4)
        )
        for blk in cutlass.range_constexpr(4):
            for nt in cutlass.range_constexpr(4):
                facc[blk][nt].fill(0.0)

        intermediate_slice = Int32(0)
        while intermediate_slice < intermediate_tiles:
            stage = intermediate_slice & Int32(1)
            a_base = smem_base + stage * Int32(self.a_stage_bytes)
            sfa_base = a_base + Int32(self.a_payload_bytes)
            b_base = (
                smem_base
                + Int32(self.b_storage_offset)
                + stage * Int32(self.b_stage_bytes)
            )
            sfb_base = (
                smem_base
                + Int32(self.sfb_storage_offset)
                + stage * Int32(self.sfb_stage_bytes)
            )

            next_slice = intermediate_slice + Int32(1)
            if next_slice < intermediate_tiles:
                self._stage_slice(
                    intermediate_u32,
                    down_rp,
                    down_sfb_rp,
                    smem_base,
                    tid,
                    source_m_tile,
                    m_half,
                    expert_idx,
                    output_tile,
                    next_slice,
                    rows_capacity,
                    intermediate_tiles,
                    packed_output_tiles,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            asc = cute.make_rmem_tensor((4,), Uint32)
            for blk in cutlass.range_constexpr(4):
                sf_row = Int32(blk * 16) + q + ((lane & Int32(1)) << Int32(3))
                asc[blk] = ld_shared_u32(sfa_base + (sf_row << Int32(2)))

            for kb in cutlass.range_constexpr(4):
                u_phys = (Int32(kb * 2) + (c >> Int32(1))) ^ q
                a_frag = cute.make_rmem_tensor((4, 4), Uint32)
                for blk in cutlass.range_constexpr(4):
                    a_lo = (
                        a_base
                        + Int32(blk * 16 * self.tile_k)
                        + (q << Int32(7))
                        + (u_phys << Int32(4))
                        + ((c & Int32(1)) << Int32(3))
                    )
                    a0, a2 = ld_shared_v2_u32(a_lo)
                    a1, a3 = ld_shared_v2_u32(a_lo + Int32(8 * self.tile_k))
                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3

                w0, w1, w2, w3 = ld_shared_v4_u32(
                    b_base
                    + (((Int32(kb * 4) + warp_idx) * Int32(32) + lane) << Int32(4))
                )
                words = cute.make_rmem_tensor((4,), Uint32)
                words[0] = w0
                words[1] = w1
                words[2] = w2
                words[3] = w3
                for nt in cutlass.range_constexpr(4):
                    n8 = warp_idx * Int32(4) + Int32(nt)
                    b0, b1 = e2m1x8_to_qmma_e2m1x8(words[nt])
                    sfb_word = ld_shared_u32(
                        sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                    )
                    for blk in cutlass.range_constexpr(4):
                        fragment = facc[blk][nt]
                        d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e2m1(
                            fragment[0],
                            fragment[1],
                            fragment[2],
                            fragment[3],
                            a_frag[blk, 0],
                            a_frag[blk, 1],
                            a_frag[blk, 2],
                            a_frag[blk, 3],
                            b0,
                            b1,
                            asc[blk],
                            sfb_word,
                            bid_a=kb,
                            bid_b=kb,
                        )
                        fragment[0] = d0
                        fragment[1] = d1
                        fragment[2] = d2
                        fragment[3] = d3

            # No thread can recycle this parity until all four compute warps
            # have consumed it.
            cute.arch.sync_threads()
            intermediate_slice += Int32(1)

        scatter_n = Int32(scatter_output.shape[1])
        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(
            self.tile_m
        )
        down_scale = down_alpha[expert_idx].to(cutlass.Float32) * global_scale[
            expert_idx
        ].to(cutlass.Float32)
        col_base = (
            output_tile * Int32(self.tile_n) + warp_idx * Int32(32) + (c << Int32(1))
        )
        for nt in cutlass.range_constexpr(4):
            col = col_base + Int32(nt * 8)
            for blk in cutlass.range_constexpr(4):
                fragment = facc[blk][nt]
                row_lo = Int32(blk * 16) + q
                row_hi = row_lo + Int32(8)
                if row_lo < valid_rows:
                    physical_row = physical_row_base + row_lo
                    tok = token_map[physical_row].to(Int32)
                    scale = down_scale * token_weights[physical_row].to(cutlass.Float32)
                    if cutlass.const_expr(self.deterministic_output):
                        # The routing front-end stores the token-major pair
                        # index in token_map for deterministic specializations.
                        # Each pair/output-column location has one producer, so
                        # phase 2 can write it exactly once; the caller then
                        # reduces routes in fixed top-k order.
                        st_global_u32(
                            get_ptr_as_int64(scatter_output, tok * scatter_n + col),
                            pack_f32x2_to_bfloat2(
                                scale * fragment[0], scale * fragment[1]
                            ),
                        )
                    else:
                        scatter_add_bf16x2(
                            get_ptr_as_int64(scatter_output, tok * scatter_n + col),
                            scale * fragment[0],
                            scale * fragment[1],
                        )
                if row_hi < valid_rows:
                    physical_row = physical_row_base + row_hi
                    tok = token_map[physical_row].to(Int32)
                    scale = down_scale * token_weights[physical_row].to(cutlass.Float32)
                    if cutlass.const_expr(self.deterministic_output):
                        st_global_u32(
                            get_ptr_as_int64(scatter_output, tok * scatter_n + col),
                            pack_f32x2_to_bfloat2(
                                scale * fragment[2], scale * fragment[3]
                            ),
                        )
                    else:
                        scatter_add_bf16x2(
                            get_ptr_as_int64(scatter_output, tok * scatter_n + col),
                            scale * fragment[2],
                            scale * fragment[3],
                        )

    @cute.kernel
    def kernel(
        self,
        intermediate_u32: cute.Tensor,
        down_rp: cute.Tensor,
        down_sfb_rp: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        down_alpha: cute.Tensor,
        global_scale: cute.Tensor,
        intermediate_tiles: cutlass.Int32,
        packed_output_tiles: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        _, _, bidz = cute.arch.block_idx()
        _, _, gdimz = cute.arch.grid_dim()
        tid = Int32(tidx)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            words: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, self.shared_words],
                1024,
            ]

        storage = smem.allocate(Storage)
        smem_base = shared_ptr_to_u32(storage.words.data_ptr())

        rows_capacity = Int32(token_map.shape[0])
        num_experts = Int32(expert_tile_base.shape[0] - 1)
        output_tiles = packed_output_tiles * Int32(2)
        source_m_tiles = expert_tile_base[num_experts].to(Int32)
        task_tail = source_m_tiles * Int32(self.source_halves) * output_tiles
        task_slot = Int32(bidz)
        while task_slot < task_tail:
            output_tile = task_slot % output_tiles
            source_half = task_slot // output_tiles
            if cutlass.const_expr(self.source_halves == 2):
                m_half = source_half & Int32(1)
                source_m_tile = source_half >> Int32(1)
            else:
                m_half = Int32(0)
                source_m_tile = source_half
            phase1_meta = source_m_tile * intermediate_tiles
            expert_idx = task_expert[phase1_meta].to(Int32)
            valid_rows = task_valid_rows[phase1_meta].to(Int32) - m_half * Int32(
                self.tile_m
            )
            valid_rows = cutlass.min(valid_rows, Int32(self.tile_m))
            valid_rows = cutlass.max(valid_rows, Int32(0))
            if valid_rows > Int32(0):
                self._run_task(
                    intermediate_u32,
                    down_rp,
                    down_sfb_rp,
                    scatter_output,
                    token_map,
                    token_weights,
                    down_alpha,
                    global_scale,
                    smem_base,
                    tid,
                    warp_idx,
                    source_m_tile,
                    m_half,
                    expert_idx,
                    output_tile,
                    valid_rows,
                    rows_capacity,
                    intermediate_tiles,
                    packed_output_tiles,
                )
            task_slot += Int32(gdimz)


__all__ = ["W4A8MaterializedPhase2Kernel"]
