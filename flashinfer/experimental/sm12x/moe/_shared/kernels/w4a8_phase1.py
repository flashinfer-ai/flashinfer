# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/w4a8_phase1.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Standalone materialized-W4A8 FC1 kernel for dense prefill regimes.

The dynamic front-end still owns routing, input quantization, and publication
of a compact expert-major M64 or M128 source domain.  This kernel consumes that
domain as M64xN128 tasks, computes gate and up together in one K sweep, applies
SiLU, and writes the existing caller-owned MXFP8 intermediate workspace.

Keeping the compute body separate from routing and FC2 removes the monolithic
kernel's register/shared-memory union.  The launch has a fixed two-CTA-per-SM
capacity and uses only caller-owned storage, so it is safe to capture and
replay without host reads or allocations.
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
    fabs_f32,
    get_ptr_as_int64,
    ld_shared_bf16_to_f32,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    mxfp8_mma_m16n8k32_f32_e2m1,
    pack_f32x2_to_bfloat2,
    quantize_block_fp8_mx,
    shared_ptr_to_u32,
    st_shared_u32,
)


class W4A8MaterializedPhase1Kernel:
    """Compute one M64 chunk of each published FC1 source tile per task."""

    tile_m = 64
    source_tile_m = 128
    tile_n = 128
    tile_k = 64
    num_warps = 4
    threads_per_cta = num_warps * 32
    stages = 2

    # A retains a 128-byte row stride even though each stage advances K64.
    # The XOR vector mapping used by QMMA can address all eight 16-byte slots
    # for any row; the unused half is merely padding.  Each B half is a compact
    # N128xK64 FP4 tile.  Gate and up are staged side by side.
    a_payload_bytes = tile_m * 128
    a_scale_bytes = tile_m * 4
    b_payload_bytes = tile_n * tile_k // 2
    sfb_bytes = (tile_n // 8) * 8 * 4

    a_offset = 0
    sfa_offset = a_offset + a_payload_bytes
    gate_b_offset = sfa_offset + a_scale_bytes
    up_b_offset = gate_b_offset + b_payload_bytes
    gate_sfb_offset = up_b_offset + b_payload_bytes
    up_sfb_offset = gate_sfb_offset + sfb_bytes
    stage_bytes = up_sfb_offset + sfb_bytes
    pipeline_bytes = stages * stage_bytes

    # The post-MMA activation tile is smaller than the pipeline union and
    # reuses it after all asynchronous copies have drained.
    epilogue_bytes = tile_m * tile_n * 2
    shared_bytes = max(pipeline_bytes, epilogue_bytes)
    shared_words = (shared_bytes + 3) // 4

    def __init__(
        self,
        *,
        fast_math: bool = False,
        source_tile_m: int = 128,
        deterministic_output: bool = False,
        num_topk: int = 1,
    ):
        if source_tile_m not in (64, 128):
            raise ValueError(
                f"materialized phase 1 source_tile_m must be 64 or 128, got {source_tile_m}"
            )
        self.fast_math = bool(fast_math)
        self.source_tile_m = int(source_tile_m)
        self.source_halves = self.source_tile_m // self.tile_m
        self.deterministic_output = bool(deterministic_output)
        if int(num_topk) <= 0:
            raise ValueError(f"num_topk must be positive, got {num_topk}")
        self.num_topk = int(num_topk)

    @cute.jit
    def __call__(
        self,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb_rp: cute.Tensor,
        intermediate_u32: cute.Tensor,
        token_map: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        alpha: cute.Tensor,
        input_global_scale: cute.Tensor,
        input_k128_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        packed_w13_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cute.recast_tensor(packed_a_storage, cutlass.Uint32),
            scale_storage,
            w13_rp,
            w13_sfb_rp,
            intermediate_u32,
            token_map,
            task_expert,
            task_valid_rows,
            expert_tile_base,
            alpha,
            input_global_scale,
            input_k128_tiles,
            intermediate_tiles,
            packed_w13_tiles,
        ).launch(
            grid=(1, 1, max_active_clusters * Int32(2)),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.jit
    def _stage_b_half_k64(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        k_half: Int32,
        tid: Int32,
    ):
        # Prepared weights are [K32, N32-chunk, lane, n8-in-chunk].  Select
        # two K32 blocks and one N128 half from the enclosing N256xK128 tile.
        for i in cutlass.range_constexpr(
            (2 * 4 * 32 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(2 * 4 * 32):
                lane = idx & Int32(31)
                kc = idx >> Int32(5)
                chunk = kc & Int32(3)
                kb = (kc >> Int32(2)) + k_half * Int32(2)
                src_word = (
                    tile_word_base
                    + Int64(kb * Int32(8 * 32 * 4))
                    + Int64((packed_half * Int32(4) + chunk) * Int32(32 * 4))
                    + Int64(lane * Int32(4))
                )
                cp_async4_shared_global(
                    dst_base + (idx << Int32(4)),
                    get_ptr_as_int64(weights, src_word),
                )

    @cute.jit
    def _stage_sfb_half(
        self,
        scales: cute.Tensor,
        dst_base: Int32,
        tile_word_base: Int64,
        packed_half: Int32,
        tid: Int32,
    ):
        # Retain the complete packed K128 scale word.  The two K64 epochs use
        # different compile-time QMMA byte ids from this same representation.
        for i in cutlass.range_constexpr(
            ((16 * 8) // 4 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32((16 * 8) // 4):
                src_word = tile_word_base + Int64(packed_half * Int32(16 * 8) + idx * 4)
                cp_async4_shared_global(
                    dst_base + (idx << Int32(4)),
                    get_ptr_as_int64(scales, src_word),
                )

    @cute.jit
    def _stage_slice(
        self,
        packed_a_u32: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb_rp: cute.Tensor,
        token_map: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        source_m_tile: Int32,
        m_half: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        valid_rows: Int32,
        k64_slice: Int32,
        input_k128_tiles: Int32,
        intermediate_tiles: Int32,
        packed_w13_tiles: Int32,
    ):
        stage = k64_slice & Int32(1)
        stage_base = smem_base + stage * Int32(self.stage_bytes)
        a_base = stage_base + Int32(self.a_offset)
        sfa_base = stage_base + Int32(self.sfa_offset)
        gate_b_base = stage_base + Int32(self.gate_b_offset)
        up_b_base = stage_base + Int32(self.up_b_offset)
        gate_sfb_base = stage_base + Int32(self.gate_sfb_offset)
        up_sfb_base = stage_base + Int32(self.up_sfb_offset)

        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(
            self.tile_m
        )
        words_per_token = input_k128_tiles * Int32(32)

        # Gather the shared input representation through the published route
        # map.  Invalid tail rows may read token zero safely; no output is
        # published for them.
        for i in cutlass.range_constexpr(
            (self.tile_m * 4 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(self.tile_m * 4):
                row = idx >> Int32(2)
                vec = idx & Int32(3)
                tok = Int32(0)
                if row < valid_rows:
                    tok = token_map[physical_row_base + row].to(Int32)
                    if cutlass.const_expr(self.deterministic_output):
                        tok = tok // Int32(self.num_topk)
                physical_vec = vec ^ (row & Int32(7))
                src_word = (
                    tok * words_per_token + k64_slice * Int32(16) + (vec << Int32(2))
                )
                cp_async4_shared_global(
                    a_base + row * Int32(128) + (physical_vec << Int32(4)),
                    get_ptr_as_int64(packed_a_u32, src_word),
                )

        if tid < Int32(self.tile_m):
            tok = Int32(0)
            if tid < valid_rows:
                tok = token_map[physical_row_base + tid].to(Int32)
                if cutlass.const_expr(self.deterministic_output):
                    tok = tok // Int32(self.num_topk)
            sf_src = tok * input_k128_tiles * Int32(4) + (
                k64_slice >> Int32(1)
            ) * Int32(4)
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(scale_storage, sf_src),
            )

        k128_slice = k64_slice >> Int32(1)
        k_half = k64_slice & Int32(1)
        input_k128_count = input_k128_tiles

        up_packed_tile = output_tile >> Int32(1)
        up_packed_half = output_tile & Int32(1)
        gate_tile = output_tile + intermediate_tiles
        gate_packed_tile = gate_tile >> Int32(1)
        gate_packed_half = gate_tile & Int32(1)

        up_tile = (
            expert_idx * packed_w13_tiles + up_packed_tile
        ) * input_k128_count + k128_slice
        gate_tile_idx = (
            expert_idx * packed_w13_tiles + gate_packed_tile
        ) * input_k128_count + k128_slice

        self._stage_b_half_k64(
            w13_rp,
            gate_b_base,
            Int64(gate_tile_idx) * Int64(4096),
            gate_packed_half,
            k_half,
            tid,
        )
        self._stage_b_half_k64(
            w13_rp,
            up_b_base,
            Int64(up_tile) * Int64(4096),
            up_packed_half,
            k_half,
            tid,
        )
        self._stage_sfb_half(
            w13_sfb_rp,
            gate_sfb_base,
            Int64(gate_tile_idx) * Int64(256),
            gate_packed_half,
            tid,
        )
        self._stage_sfb_half(
            w13_sfb_rp,
            up_sfb_base,
            Int64(up_tile) * Int64(256),
            up_packed_half,
            tid,
        )

    @cute.jit
    def _activated_value(
        self,
        gate: cutlass.Float32,
        up: cutlass.Float32,
        alpha_value: cutlass.Float32,
    ) -> cutlass.Float32:
        gate = alpha_value * gate
        up = alpha_value * up
        sigmoid = cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math)
        )
        return gate * sigmoid * up

    @cute.jit
    def _run_task(
        self,
        packed_a_u32: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb_rp: cute.Tensor,
        intermediate_u32: cute.Tensor,
        token_map: cute.Tensor,
        alpha: cute.Tensor,
        input_global_scale: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        warp_idx: Int32,
        source_m_tile: Int32,
        m_half: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        valid_rows: Int32,
        rows_capacity: Int32,
        input_k128_tiles: Int32,
        intermediate_tiles: Int32,
        packed_w13_tiles: Int32,
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)

        self._stage_slice(
            packed_a_u32,
            scale_storage,
            w13_rp,
            w13_sfb_rp,
            token_map,
            smem_base,
            tid,
            source_m_tile,
            m_half,
            expert_idx,
            output_tile,
            valid_rows,
            Int32(0),
            input_k128_tiles,
            intermediate_tiles,
            packed_w13_tiles,
        )
        cute.arch.cp_async_commit_group()

        # Keep each MMA's four accumulator registers as an independent
        # fragment.  A single 4x4x4 mutable tensor makes the 4.6 lowering pack
        # and unpack the complete live accumulator set around loop-carried
        # values even though every MMA consumes exactly four adjacent values.
        gate_acc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(4))
            for _blk in range(4)
        )
        up_acc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(4))
            for _blk in range(4)
        )
        for blk in cutlass.range_constexpr(4):
            for nt in cutlass.range_constexpr(4):
                gate_acc[blk][nt].fill(0.0)
                up_acc[blk][nt].fill(0.0)

        input_k64_tiles = input_k128_tiles * Int32(2)
        k64_slice = Int32(0)
        while k64_slice < input_k64_tiles:
            stage = k64_slice & Int32(1)
            stage_base = smem_base + stage * Int32(self.stage_bytes)
            a_base = stage_base + Int32(self.a_offset)
            sfa_base = stage_base + Int32(self.sfa_offset)
            gate_b_base = stage_base + Int32(self.gate_b_offset)
            up_b_base = stage_base + Int32(self.up_b_offset)
            gate_sfb_base = stage_base + Int32(self.gate_sfb_offset)
            up_sfb_base = stage_base + Int32(self.up_sfb_offset)

            next_slice = k64_slice + Int32(1)
            if next_slice < input_k64_tiles:
                self._stage_slice(
                    packed_a_u32,
                    scale_storage,
                    w13_rp,
                    w13_sfb_rp,
                    token_map,
                    smem_base,
                    tid,
                    source_m_tile,
                    m_half,
                    expert_idx,
                    output_tile,
                    valid_rows,
                    next_slice,
                    input_k128_tiles,
                    intermediate_tiles,
                    packed_w13_tiles,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            scale_shift = Uint32(k64_slice & Int32(1)) * Uint32(16)
            asc = cute.make_rmem_tensor((4,), Uint32)
            for blk in cutlass.range_constexpr(4):
                sf_row = Int32(blk * 16) + q + ((lane & Int32(1)) << Int32(3))
                asc[blk] = ld_shared_u32(sfa_base + (sf_row << Int32(2))) >> scale_shift

            for kb in cutlass.range_constexpr(2):
                u_phys = (Int32(kb * 2) + (c >> Int32(1))) ^ q
                a_frag = cute.make_rmem_tensor((4, 4), Uint32)
                for blk in cutlass.range_constexpr(4):
                    a_lo = (
                        a_base
                        + Int32(blk * 16 * 128)
                        + (q << Int32(7))
                        + (u_phys << Int32(4))
                        + ((c & Int32(1)) << Int32(3))
                    )
                    a0, a2 = ld_shared_v2_u32(a_lo)
                    a1, a3 = ld_shared_v2_u32(a_lo + Int32(8 * 128))
                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3

                gw0, gw1, gw2, gw3 = ld_shared_v4_u32(
                    gate_b_base
                    + (((Int32(kb * 4) + warp_idx) * Int32(32) + lane) << Int32(4))
                )
                uw0, uw1, uw2, uw3 = ld_shared_v4_u32(
                    up_b_base
                    + (((Int32(kb * 4) + warp_idx) * Int32(32) + lane) << Int32(4))
                )
                gate_words = cute.make_rmem_tensor((4,), Uint32)
                up_words = cute.make_rmem_tensor((4,), Uint32)
                gate_words[0] = gw0
                gate_words[1] = gw1
                gate_words[2] = gw2
                gate_words[3] = gw3
                up_words[0] = uw0
                up_words[1] = uw1
                up_words[2] = uw2
                up_words[3] = uw3

                for nt in cutlass.range_constexpr(4):
                    n8 = warp_idx * Int32(4) + Int32(nt)
                    gb0, gb1 = e2m1x8_to_qmma_e2m1x8(gate_words[nt])
                    ub0, ub1 = e2m1x8_to_qmma_e2m1x8(up_words[nt])
                    gate_sfb = (
                        ld_shared_u32(gate_sfb_base + ((n8 * Int32(8) + q) << Int32(2)))
                        >> scale_shift
                    )
                    up_sfb = (
                        ld_shared_u32(up_sfb_base + ((n8 * Int32(8) + q) << Int32(2)))
                        >> scale_shift
                    )
                    for blk in cutlass.range_constexpr(4):
                        gate_fragment = gate_acc[blk][nt]
                        g0, g1, g2, g3 = mxfp8_mma_m16n8k32_f32_e2m1(
                            gate_fragment[0],
                            gate_fragment[1],
                            gate_fragment[2],
                            gate_fragment[3],
                            a_frag[blk, 0],
                            a_frag[blk, 1],
                            a_frag[blk, 2],
                            a_frag[blk, 3],
                            gb0,
                            gb1,
                            asc[blk],
                            gate_sfb,
                            bid_a=kb,
                            bid_b=kb,
                        )
                        gate_fragment[0] = g0
                        gate_fragment[1] = g1
                        gate_fragment[2] = g2
                        gate_fragment[3] = g3
                        up_fragment = up_acc[blk][nt]
                        u0, u1, u2, u3 = mxfp8_mma_m16n8k32_f32_e2m1(
                            up_fragment[0],
                            up_fragment[1],
                            up_fragment[2],
                            up_fragment[3],
                            a_frag[blk, 0],
                            a_frag[blk, 1],
                            a_frag[blk, 2],
                            a_frag[blk, 3],
                            ub0,
                            ub1,
                            asc[blk],
                            up_sfb,
                            bid_a=kb,
                            bid_b=kb,
                        )
                        up_fragment[0] = u0
                        up_fragment[1] = u1
                        up_fragment[2] = u2
                        up_fragment[3] = u3

            cute.arch.sync_threads()
            k64_slice += Int32(1)

        # The pipeline region aliases the activation staging tile below.  A
        # wait-group(1) is sufficient while consuming alternating stages, but
        # the final (possibly empty) committed group must be fully retired
        # before those shared addresses are repurposed by ordinary stores.
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        alpha_value = alpha[expert_idx].to(cutlass.Float32) * input_global_scale[
            expert_idx
        ].to(cutlass.Float32)
        epilogue_base = smem_base
        col_base = warp_idx * Int32(32) + (c << Int32(1))
        for nt in cutlass.range_constexpr(4):
            col = col_base + Int32(nt * 8)
            for blk in cutlass.range_constexpr(4):
                gate_fragment = gate_acc[blk][nt]
                up_fragment = up_acc[blk][nt]
                row_lo = Int32(blk * 16) + q
                row_hi = row_lo + Int32(8)
                act0 = self._activated_value(
                    gate_fragment[0], up_fragment[0], alpha_value
                )
                act1 = self._activated_value(
                    gate_fragment[1], up_fragment[1], alpha_value
                )
                act2 = self._activated_value(
                    gate_fragment[2], up_fragment[2], alpha_value
                )
                act3 = self._activated_value(
                    gate_fragment[3], up_fragment[3], alpha_value
                )
                st_shared_u32(
                    epilogue_base + (row_lo * Int32(self.tile_n) + col) * Int32(2),
                    pack_f32x2_to_bfloat2(act0, act1),
                )
                st_shared_u32(
                    epilogue_base + (row_hi * Int32(self.tile_n) + col) * Int32(2),
                    pack_f32x2_to_bfloat2(act2, act3),
                )

        cute.arch.sync_threads()

        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(
            self.tile_m
        )
        words_per_row = intermediate_tiles * Int32(32)
        if tid < valid_rows:
            scale_word = Uint32(0)
            for block in cutlass.range_constexpr(4):
                values = cute.make_rmem_tensor((32,), cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                for elem in cutlass.range_constexpr(32):
                    value = ld_shared_bf16_to_f32(
                        epilogue_base
                        + (tid * Int32(self.tile_n) + Int32(block * 32 + elem))
                        * Int32(2)
                    )
                    values[elem] = value
                    abs_value = fabs_f32(value)
                    if abs_value > block_max:
                        block_max = abs_value
                payload, scale_byte = quantize_block_fp8_mx(values, block_max)
                dst_word = (
                    (physical_row_base + tid) * words_per_row
                    + output_tile * Int32(32)
                    + Int32(block * 8)
                )
                for word in cutlass.range_constexpr(8):
                    intermediate_u32[dst_word + Int32(word)] = payload[word]
                scale_word = scale_word | (
                    (scale_byte & Uint32(0xFF)) << Uint32(block * 8)
                )

            sf_base = rows_capacity * words_per_row
            intermediate_u32[
                sf_base + output_tile * rows_capacity + physical_row_base + tid
            ] = scale_word

        # Only the first 64 threads perform the row-wise quantize/store.  The
        # other warps must not advance the persistent task loop and overwrite
        # the aliased shared activation tile while those threads still read it.
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        packed_a_u32: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb_rp: cute.Tensor,
        intermediate_u32: cute.Tensor,
        token_map: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        alpha: cute.Tensor,
        input_global_scale: cute.Tensor,
        input_k128_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        packed_w13_tiles: cutlass.Int32,
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
        source_m_tiles = expert_tile_base[num_experts].to(Int32)
        task_tail = source_m_tiles * Int32(self.source_halves) * intermediate_tiles
        task_slot = Int32(bidz)
        while task_slot < task_tail:
            output_tile = task_slot % intermediate_tiles
            source_half = task_slot // intermediate_tiles
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
            if valid_rows > Int32(self.tile_m):
                valid_rows = Int32(self.tile_m)
            if valid_rows > Int32(0):
                self._run_task(
                    packed_a_u32,
                    scale_storage,
                    w13_rp,
                    w13_sfb_rp,
                    intermediate_u32,
                    token_map,
                    alpha,
                    input_global_scale,
                    smem_base,
                    tid,
                    warp_idx,
                    source_m_tile,
                    m_half,
                    expert_idx,
                    output_tile,
                    valid_rows,
                    rows_capacity,
                    input_k128_tiles,
                    intermediate_tiles,
                    packed_w13_tiles,
                )
            task_slot += Int32(gdimz)


__all__ = ["W4A8MaterializedPhase1Kernel"]
