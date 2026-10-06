"""Standalone materialized-NVFP4 FC1 kernel for dense prefill regimes.

The dynamic front-end still owns routing, input quantization, and publication
of the compact expert-major packed-A (E2M1x2) + SFA (E4M3 K/16, F8_128x4
swizzle) domain.  This kernel consumes that domain as M64xN128 tasks,
computes gate and up together in one K sweep with the SM120 native NVFP4
block-scaled QMMA (``kind::mxf4nvf4`` ``scale_vec::4X`` m16n8k64), applies
SiLU(gate)*up with the monolithic kernel's alpha semantics, and requantizes
the intermediate to NVFP4 (per-16 E4M3 block scale, ``quantize_block_fp4``
semantics) into the caller-owned intermediate workspace.

The published weight contract is the original (non-repacked) swizzled
block-scale FP4 tensors: ``w13_b`` is the packed E2M1 payload
``[w1_n, K, E]`` (K-major rows) and ``sfb_w13`` the F8_128x4-swizzled E4M3
scale plane.  Keeping the compute body separate from routing and FC2 removes
the monolithic kernel's register/shared-memory union.  The launch has a
fixed two-CTA-per-SM capacity and uses only caller-owned storage, so it is
safe to capture and replay without host reads or allocations.

Intermediate workspace layout (caller-owned ``intermediate_u32``):

- Payload plane: row-major ``[physical_row][intermediate_tiles * 16]`` u32;
  K128 slice ``t`` of a row occupies words ``[16*t, 16*t + 16)`` as packed
  E2M1 bytes (64 bytes per slice).
- Scale plane: ``[intermediate_tiles][rows_capacity][2]`` u32 appended after
  ``rows_capacity * words_per_row`` words; word pair ``2*row + {0, 1}`` of
  slice ``t`` holds the eight E4M3 block scales of that row's slice, low
  word first (k16 blocks 0..3 / 4..7).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    fabs_f32,
    get_ptr_as_int64,
    ld_shared_bf16_to_f32,
    ld_shared_u32,
    nvfp4_mma_m16n8k64_f32_e2m1,
    pack_f32x2_to_bfloat2,
    quantize_block_fp4,
    shared_ptr_to_u32,
    st_global_u64,
    st_global_u32,
    st_shared_u32,
)


class Nvfp4MaterializedPhase1Kernel:
    """Compute one M64 chunk of each published FC1 source tile per task."""

    tile_m = 64
    source_tile_m = 128
    tile_n = 128
    num_warps = 4
    threads_per_cta = num_warps * 32
    stages = 2

    # Per-stage shared geometry.  Each stage holds one K64 FP4 slice:
    # - A payload: M64 rows x 32 packed bytes, row-major (stage = k64 slice).
    # - A scale: M64 packed K64 scale words (4 E4M3 bytes per row), gathered
    #   from the published F8_128x4-swizzled SFA plane.
    # - Gate and up B payloads: N128 weight rows x 32 packed bytes each.
    # - Gate and up SFB: N128 packed K64 scale words each.
    a_payload_bytes = tile_m * 32
    a_scale_bytes = tile_m * 4
    b_payload_bytes = tile_n * 32
    sfb_bytes = tile_n * 4

    a_offset = 0
    sfa_offset = a_offset + a_payload_bytes
    gate_b_offset = sfa_offset + a_scale_bytes
    up_b_offset = gate_b_offset + b_payload_bytes
    gate_sfb_offset = up_b_offset + b_payload_bytes
    up_sfb_offset = gate_sfb_offset + sfb_bytes
    stage_bytes = up_sfb_offset + sfb_bytes
    pipeline_bytes = stages * stage_bytes

    # The post-MMA activation tile reuses the pipeline union after all
    # asynchronous copies have drained (identical to the w4a8 phase 1).
    epilogue_bytes = tile_m * tile_n * 2
    shared_bytes = max(pipeline_bytes, epilogue_bytes)
    shared_words = (shared_bytes + 3) // 4

    def __init__(
        self,
        *,
        fast_math: bool = False,
        source_tile_m: int = 128,
        activation: str = "silu",
        swiglu_limit: float | None = None,
    ):
        if source_tile_m not in (64, 128):
            raise ValueError(
                f"materialized phase 1 source_tile_m must be 64 or 128, got {source_tile_m}"
            )
        if activation != "silu":
            raise ValueError(
                "materialized NVFP4 phase 1 requires silu, "
                f"got {activation!r}"
            )
        self.fast_math = bool(fast_math)
        self.source_tile_m = int(source_tile_m)
        self.source_halves = self.source_tile_m // self.tile_m
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else float(swiglu_limit)

    @cute.jit
    def __call__(
        self,
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_b: cute.Tensor,
        sfb_w13: cute.Tensor,
        intermediate_u32: cute.Tensor,
        token_map: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        alpha: cute.Tensor,
        global_scale: cute.Tensor,
        input_k128_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        w13_n128_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cute.recast_tensor(packed_a_storage, cutlass.Uint32),
            scale_storage,
            cute.recast_tensor(w13_b, cutlass.Uint32),
            sfb_w13,
            intermediate_u32,
            token_map,
            task_expert,
            task_valid_rows,
            expert_tile_base,
            alpha,
            global_scale,
            input_k128_tiles,
            intermediate_tiles,
            w13_n128_tiles,
        ).launch(
            grid=(1, 1, max_active_clusters * Int32(2)),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.jit
    def _stage_slice(
        self,
        packed_a_u32: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_b_u32: cute.Tensor,
        sfb_w13: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        source_m_tile: Int32,
        m_half: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        k64_slice: Int32,
        input_k128_tiles: Int32,
        intermediate_tiles: Int32,
        w13_n128_tiles: Int32,
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
        words_per_token = input_k128_tiles * Int32(16)

        # A payload: 64 rows x two 16-byte vectors per K64 slice.  Every row
        # within the tile (including tail rows past valid_rows) is read; the
        # front-end zero-pads the tile's physical rows through the 128-row
        # atom, so out-of-range reads stay in-bounds.  No output is published
        # for tail rows.  Pool-scaled physical-row offsets stay Int64.
        for i in cutlass.range_constexpr(
            (self.tile_m * 2 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(self.tile_m * 2):
                row = idx >> Int32(1)
                vec = idx & Int32(1)
                src_word = (
                    Int64(physical_row_base + row) * Int64(words_per_token)
                    + Int64(k64_slice * Int32(8))
                    + Int64(vec * Int32(4))
                )
                cp_async4_shared_global(
                    a_base + row * Int32(32) + (vec << Int32(4)),
                    get_ptr_as_int64(packed_a_u32, src_word),
                )

        # One packed K64 scale word per activation row, gathered from the
        # published expert-major F8_128x4-swizzled SFA plane (the front-end's
        # scale_storage writes, indexed by physical row).
        if tid < Int32(self.tile_m):
            phys_row = physical_row_base + tid
            sf_atom = phys_row >> Int32(7)
            sf_row = phys_row & Int32(127)
            sf_src = (
                Int64(sf_atom) * Int64(input_k128_tiles * Int32(2)) * Int64(512)
                + Int64(k64_slice * Int32(512))
                + Int64((sf_row % Int32(32)) * Int32(16))
                + Int64((sf_row >> Int32(5)) * Int32(4))
            )
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(scale_storage, sf_src),
            )

        # B payload: the published w13 payload is plain row-major
        # [E][w1_n][K/2 bytes]; the gated activation packs [up, gate] across
        # N, so the up half of output tile t is expert rows
        # [128t, 128t+128) and the gate half rows [128(t + intermediate_tiles),
        # +128).  Pool-scaled weight-row offsets stay Int64.
        expert_row_base = Int64(expert_idx) * Int64(w13_n128_tiles) * Int64(128)
        up_row_base = expert_row_base + Int64(output_tile) * Int64(self.tile_n)
        gate_row_base = expert_row_base + Int64(
            output_tile + intermediate_tiles
        ) * Int64(self.tile_n)
        w13_row_words = input_k128_tiles * Int32(16)
        for half in cutlass.range_constexpr(2):
            if cutlass.const_expr(half == 0):
                b_base = up_b_base
                row_base = up_row_base
            else:
                b_base = gate_b_base
                row_base = gate_row_base
            for i in cutlass.range_constexpr(
                (self.tile_n * 2 + self.threads_per_cta - 1) // self.threads_per_cta
            ):
                idx = tid + Int32(i * self.threads_per_cta)
                if idx < Int32(self.tile_n * 2):
                    row = idx >> Int32(1)
                    vec = idx & Int32(1)
                    # row_base is a weight-row index; scale it into words
                    # before combining with the intra-row word offsets.
                    src_word = (
                        row_base * Int64(w13_row_words)
                        + Int64(row) * Int64(w13_row_words)
                        + Int64(k64_slice * Int32(8))
                        + Int64(vec * Int32(4))
                    )
                    cp_async4_shared_global(
                        b_base + row * Int32(32) + (vec << Int32(4)),
                        get_ptr_as_int64(w13_b_u32, src_word),
                    )

        # One packed K64 scale word per weight row, gathered from the
        # published F8_128x4-swizzled sfb_w13 plane (one padded plane per
        # expert, byte offsets; the expert stride is the padded plane size).
        sfb_atom_bytes = Int64(input_k128_tiles * Int32(2)) * Int64(512)
        # Publisher planes are compact: one padded 128-row atom contributes
        # k4*512 bytes, and each expert holds w13_n128_tiles atoms.
        sfb_expert_stride = Int64(w13_n128_tiles) * sfb_atom_bytes
        for half in cutlass.range_constexpr(2):
            if cutlass.const_expr(half == 0):
                sfb_base = up_sfb_base
                n_local = output_tile * Int32(self.tile_n) + tid
            else:
                sfb_base = gate_sfb_base
                n_local = (
                    output_tile + intermediate_tiles
                ) * Int32(self.tile_n) + tid
            if tid < Int32(self.tile_n):
                sf_src = (
                    Int64(expert_idx) * sfb_expert_stride
                    + Int64(n_local >> Int32(7)) * sfb_atom_bytes
                    + Int64(k64_slice * Int32(512))
                    + Int64(((n_local & Int32(127)) % Int32(32)) * Int32(16))
                    + Int64(((n_local & Int32(127)) >> Int32(5)) * Int32(4))
                )
                cp_async_u32_shared_global(
                    sfb_base + (tid << Int32(2)),
                    get_ptr_as_int64(sfb_w13, sf_src),
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
        if cutlass.const_expr(self.has_swiglu_limit):
            limit = cutlass.Float32(self.swiglu_limit)
            neg_limit = cutlass.Float32(-self.swiglu_limit)
            if gate > limit:
                gate = limit
            if up > limit:
                up = limit
            if up < neg_limit:
                up = neg_limit
        sigmoid = cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math)
        )
        return gate * sigmoid * up

    @cute.jit
    def _run_task(
        self,
        packed_a_u32: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_b_u32: cute.Tensor,
        sfb_w13: cute.Tensor,
        intermediate_u32: cute.Tensor,
        alpha: cute.Tensor,
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
        input_k128_tiles: Int32,
        intermediate_tiles: Int32,
        w13_n128_tiles: Int32,
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)

        self._stage_slice(
            packed_a_u32,
            scale_storage,
            w13_b_u32,
            sfb_w13,
            smem_base,
            tid,
            source_m_tile,
            m_half,
            expert_idx,
            output_tile,
            Int32(0),
            input_k128_tiles,
            intermediate_tiles,
            w13_n128_tiles,
        )
        cute.arch.cp_async_commit_group()

        # Keep each MMA's four accumulator registers as an independent
        # fragment (avoids the 4.6 lowering's pack/unpack around loop-carried
        # values; identical to the w4a8 phase 1 structure).  Each warp owns
        # M64 (four M16 blocks) x N32 (four n8 groups).
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
                    w13_b_u32,
                    sfb_w13,
                    smem_base,
                    tid,
                    source_m_tile,
                    m_half,
                    expert_idx,
                    output_tile,
                    next_slice,
                    input_k128_tiles,
                    intermediate_tiles,
                    w13_n128_tiles,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            # QMMA fragment addressing (see nvfp4_mma_m16n8k64_f32_e2m1):
            # A reg r nib n: row = q + 8*(r%2), k = 32*(r//2) + 8*c + n, so
            # the lane reads byte 4*c (+16 for regs 2/3) of row
            # q (+8 for odd regs).
            # B reg j nib n: col = q, k = 32*j + 8*c + n, so the lane reads
            # byte 4*c (+16 for reg 1) of staged weight row q.  A staged
            # K64 row is exactly 32 linear bytes, so k maps directly to
            # byte 4*c + 16*(half) + 4*(n//2) -- no swizzle needed.
            # SFA word of lane L (q = L>>2, c = L&3) covers SF_A row
            # q + 8*(c&1): the hardware reads rows 0..7 from lanes with
            # c in {0, 2} and rows 8..15 from lanes with c in {1, 3}; the
            # c in {2, 3} words duplicate the c in {0, 1} rows and are
            # ignored by the instruction.  SFB word of lane L covers SF_B
            # col L//4.

            # B fragments: per warp-local n8 group g (cols 8*(4*warp + g)).
            # Each staged weight row r holds the K64 slice linearly; the
            # lane's c selects the 4-byte k group within the half and the
            # register selects the half.
            # B fragments: the m16n8k64 B operand of lane (q, c) for n8
            # group g is B row (8*(4*warp + g) + q), word c (+4 for the
            # k32-high register) of that row's 8-word staging — empirically
            # pinned against the QMMA probe (tests/moe fragment probes);
            # the row index is q, never the warp-level column.
            gate_b_lo = cute.make_rmem_tensor((4,), Uint32)
            gate_b_hi = cute.make_rmem_tensor((4,), Uint32)
            up_b_lo = cute.make_rmem_tensor((4,), Uint32)
            up_b_hi = cute.make_rmem_tensor((4,), Uint32)
            gate_sfb_w = cute.make_rmem_tensor((4,), Uint32)
            up_sfb_w = cute.make_rmem_tensor((4,), Uint32)
            for g in cutlass.range_constexpr(4):
                b_row = Int32(8) * (warp_idx * Int32(4) + Int32(g)) + (
                    lane >> Int32(2)
                )
                gate_b_lo[g] = ld_shared_u32(
                    gate_b_base + b_row * Int32(32) + Int32(4) * c
                )
                gate_b_hi[g] = ld_shared_u32(
                    gate_b_base + b_row * Int32(32) + Int32(4) * c + Int32(16)
                )
                up_b_lo[g] = ld_shared_u32(
                    up_b_base + b_row * Int32(32) + Int32(4) * c
                )
                up_b_hi[g] = ld_shared_u32(
                    up_b_base + b_row * Int32(32) + Int32(4) * c + Int32(16)
                )
                sfb_col = Int32(8) * (warp_idx * Int32(4) + Int32(g)) + (
                    lane >> Int32(2)
                )
                gate_sfb_w[g] = ld_shared_u32(gate_sfb_base + (sfb_col << Int32(2)))
                up_sfb_w[g] = ld_shared_u32(up_sfb_base + (sfb_col << Int32(2)))

            for blk in cutlass.range_constexpr(4):
                a_frag = cute.make_rmem_tensor((4,), Uint32)
                for r in cutlass.range_constexpr(4):
                    a_row = (
                        Int32(16) * Int32(blk)
                        + (lane >> Int32(2))
                        + Int32(8) * (r & Int32(1))
                    )
                    a_frag[r] = ld_shared_u32(
                        a_base
                        + a_row * Int32(32)
                        + Int32(4) * c
                        + Int32(16) * (r >> Int32(1))
                    )
                # The SFA word of lane 4q+c covers SF_A row q + 8*(c&1):
                # rows 0..7 ride lanes with c in {0, 2} and rows 8..15 ride
                # lanes with c in {1, 3}; the c in {2, 3} lanes re-read the
                # c in {0, 1} words (clamped to a live row) which the
                # hardware ignores for this atom.
                sf_row = Int32(16) * Int32(blk) + (lane >> Int32(2)) + Int32(
                    8
                ) * (lane & Int32(1))
                sfa_w = ld_shared_u32(sfa_base + (sf_row << Int32(2)))

                for g in cutlass.range_constexpr(4):
                    gate_fragment = gate_acc[blk][g]
                    g0, g1, g2, g3 = nvfp4_mma_m16n8k64_f32_e2m1(
                        gate_fragment[0],
                        gate_fragment[1],
                        gate_fragment[2],
                        gate_fragment[3],
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        gate_b_lo[g],
                        gate_b_hi[g],
                        sfa_w,
                        gate_sfb_w[g],
                    )
                    gate_fragment[0] = g0
                    gate_fragment[1] = g1
                    gate_fragment[2] = g2
                    gate_fragment[3] = g3
                    up_fragment = up_acc[blk][g]
                    u0, u1, u2, u3 = nvfp4_mma_m16n8k64_f32_e2m1(
                        up_fragment[0],
                        up_fragment[1],
                        up_fragment[2],
                        up_fragment[3],
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        up_b_lo[g],
                        up_b_hi[g],
                        sfa_w,
                        up_sfb_w[g],
                    )
                    up_fragment[0] = u0
                    up_fragment[1] = u1
                    up_fragment[2] = u2
                    up_fragment[3] = u3

            cute.arch.sync_threads()
            k64_slice += Int32(1)

        # The pipeline region aliases the activation staging tile below.  The
        # final (possibly empty) committed group must be fully retired before
        # those shared addresses are repurposed by ordinary stores.
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        # alpha_value = alpha[e]: the monolithic NVFP4 kernel applies only
        # the weight dequant alpha at FC1 (the a1 global scale is already
        # folded into the published SFA plane).
        alpha_value = alpha[expert_idx].to(cutlass.Float32)
        quant_gs = global_scale[expert_idx].to(cutlass.Float32)
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
        words_per_row = intermediate_tiles * Int32(16)
        if tid < valid_rows:
            phys_row = physical_row_base + tid
            scale_word_lo = Uint32(0)
            scale_word_hi = Uint32(0)
            for block in cutlass.range_constexpr(8):
                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                block_max = cutlass.Float32(0.0)
                for elem in cutlass.range_constexpr(16):
                    value = ld_shared_bf16_to_f32(
                        epilogue_base
                        + (tid * Int32(self.tile_n) + Int32(block * 16 + elem))
                        * Int32(2)
                    )
                    values[elem] = value
                    abs_value = fabs_f32(value)
                    if abs_value > block_max:
                        block_max = abs_value
                payload, scale_byte = quantize_block_fp4(
                    values, block_max, quant_gs
                )
                dst_word = (
                    Int64(phys_row) * Int64(words_per_row)
                    + Int64(output_tile * Int32(16) + Int32(block * 2))
                )
                st_global_u64(get_ptr_as_int64(intermediate_u32, dst_word), payload)
                if cutlass.const_expr(block < 4):
                    scale_word_lo = scale_word_lo | (
                        (scale_byte & Uint32(0xFF)) << Uint32(block * 8)
                    )
                else:
                    scale_word_hi = scale_word_hi | (
                        (scale_byte & Uint32(0xFF)) << Uint32((block - 4) * 8)
                    )

            sf_plane = Int64(rows_capacity) * Int64(words_per_row)
            sf_word = (
                sf_plane
                + Int64(output_tile) * Int64(rows_capacity) * Int64(2)
                + Int64(phys_row) * Int64(2)
            )
            st_global_u32(
                get_ptr_as_int64(intermediate_u32, sf_word), scale_word_lo
            )
            st_global_u32(
                get_ptr_as_int64(intermediate_u32, sf_word + Int64(1)),
                scale_word_hi,
            )

        # Only the first 64 threads perform the row-wise quantize/store.  The
        # other warps must not advance the persistent task loop and overwrite
        # the aliased shared activation tile while those threads still read it.
        cute.arch.sync_threads()

    @cute.kernel
    def kernel(
        self,
        packed_a_u32: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_b_u32: cute.Tensor,
        sfb_w13: cute.Tensor,
        intermediate_u32: cute.Tensor,
        token_map: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        alpha: cute.Tensor,
        global_scale: cute.Tensor,
        input_k128_tiles: cutlass.Int32,
        intermediate_tiles: cutlass.Int32,
        w13_n128_tiles: cutlass.Int32,
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
                    w13_b_u32,
                    sfb_w13,
                    intermediate_u32,
                    alpha,
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
                    input_k128_tiles,
                    intermediate_tiles,
                    w13_n128_tiles,
                )
            task_slot += Int32(gdimz)


__all__ = ["Nvfp4MaterializedPhase1Kernel"]
