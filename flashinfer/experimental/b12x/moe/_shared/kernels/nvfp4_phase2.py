"""Standalone materialized-NVFP4 FC2 kernel for dense prefill regimes.

The dynamic front-end and phase 1 publish an expert-major M64 or M128 source
domain in NVFP4 form (packed E2M1 payload + per-16 E4M3 block scales).  This
kernel consumes the caller-owned materialization workspace as M64xN128 tasks,
contracts the intermediate through the SM120 native NVFP4 block-scaled QMMA
(``kind::mxf4nvf4`` ``scale_vec::4X`` m16n8k64), and accumulates per-token
topk-weighted output into ``scatter_output`` through the atomic-add path
(the monolithic kernel's default scatter contract).

The published weight contract is the original (non-repacked) tensors:
``down_b`` is the packed E2M1 payload ``[K, I_tp, E]`` (I-major rows, so each
hidden output row's contraction bytes are contiguous) and ``sfb_down`` the
F8_128x4-swizzled E4M3 scale plane indexed by the hidden dim.  The launch is
fixed-capacity and stream ordered, so it remains safe for CUDA graph capture
and replay without host reads or allocations.

Intermediate workspace layout (produced by :class:
`Nvfp4MaterializedPhase1Kernel`):

- Payload plane: row-major ``[physical_row][intermediate_tiles * 16]`` u32.
- Scale plane: ``[intermediate_tiles][rows_capacity][2]`` u32 after
  ``rows_capacity * words_per_row`` words.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.intrinsics import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    ld_shared_u32,
    nvfp4_mma_m16n8k64_f32_e2m1,
    pack_f32x2_to_bfloat2,
    scatter_add_bf16x2,
    shared_ptr_to_u32,
    st_global_u32,
)


class Nvfp4MaterializedPhase2Kernel:
    """Consume materialization through compact M64xN128 FC2 CTAs."""

    tile_m = 64
    source_tile_m = 128
    tile_n = 128
    num_warps = 4
    threads_per_cta = num_warps * 32
    stages = 2

    # Per-stage shared geometry (one K64 FP4 slice of the intermediate):
    # A payload M64 x 32B + A scale M64 x 4B + B payload N128 x 32B +
    # B scale N128 x 4B.
    a_payload_bytes = tile_m * 32
    a_scale_bytes = tile_m * 4
    b_payload_bytes = tile_n * 32
    sfb_bytes = tile_n * 4

    a_offset = 0
    sfa_offset = a_offset + a_payload_bytes
    b_offset = sfa_offset + a_scale_bytes
    sfb_offset = b_offset + b_payload_bytes
    stage_bytes = sfb_offset + sfb_bytes
    pipeline_bytes = stages * stage_bytes
    shared_bytes = pipeline_bytes
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
        down_b: cute.Tensor,
        sfb_down: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        down_alpha: cute.Tensor,
        intermediate_tiles: cutlass.Int32,
        output_n128_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            cute.recast_tensor(intermediate_u32, cutlass.Uint32),
            cute.recast_tensor(down_b, cutlass.Uint32),
            sfb_down,
            scatter_output,
            token_map,
            token_weights,
            task_expert,
            task_valid_rows,
            expert_tile_base,
            down_alpha,
            intermediate_tiles,
            output_n128_tiles,
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
        down_b_u32: cute.Tensor,
        sfb_down: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        source_m_tile: Int32,
        m_half: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        k64_slice: Int32,
        rows_capacity: Int32,
        intermediate_tiles: Int32,
        output_n128_tiles: Int32,
    ):
        stage = k64_slice & Int32(1)
        stage_base = smem_base + stage * Int32(self.stage_bytes)
        a_base = stage_base + Int32(self.a_offset)
        sfa_base = stage_base + Int32(self.sfa_offset)
        b_base = stage_base + Int32(self.b_offset)
        sfb_base = stage_base + Int32(self.sfb_offset)

        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(
            self.tile_m
        )
        words_per_row = intermediate_tiles * Int32(16)

        # A payload: 64 rows x two 16-byte vectors per K64 slice, gathered
        # from the materialized row-major payload plane.  Pool-scaled
        # physical-row offsets stay Int64 (2^31/stride rows).
        for i in cutlass.range_constexpr(
            (self.tile_m * 2 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(self.tile_m * 2):
                row = idx >> Int32(1)
                vec = idx & Int32(1)
                src_word = (
                    Int64(physical_row_base + row) * Int64(words_per_row)
                    + Int64((k64_slice >> Int32(1)) * Int32(16))
                    + Int64((k64_slice & Int32(1)) * Int32(8))
                    + Int64(vec * Int32(4))
                )
                cp_async4_shared_global(
                    a_base + row * Int32(32) + (vec << Int32(4)),
                    get_ptr_as_int64(intermediate_u32, src_word),
                )

        # One packed K64 scale word per row, gathered from the materialized
        # scale plane [slice][row][2].
        if tid < Int32(self.tile_m):
            sf_plane = Int64(rows_capacity) * Int64(words_per_row)
            sf_word = (
                sf_plane
                + Int64(k64_slice >> Int32(1)) * Int64(rows_capacity) * Int64(2)
                + Int64(physical_row_base + tid) * Int64(2)
                + Int64(k64_slice & Int32(1))
            )
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(intermediate_u32, sf_word),
            )

        # B payload: the published down payload is [E][K][I/2 bytes] with the
        # contraction (intermediate) dim contiguous, so each hidden output
        # row's K64 slice is 32 contiguous bytes.  Pool-scaled weight-row
        # offsets stay Int64.
        expert_row_base = Int64(expert_idx) * Int64(output_n128_tiles) * Int64(
            self.tile_n
        )
        row_base = expert_row_base + Int64(output_tile) * Int64(self.tile_n)
        down_row_words = intermediate_tiles * Int32(16)
        for i in cutlass.range_constexpr(
            (self.tile_n * 2 + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(self.tile_n * 2):
                row = idx >> Int32(1)
                vec = idx & Int32(1)
                src_word = (
                    # row_base is a weight-row index; scale it into words
                    # before combining with the intra-row word offsets.
                    row_base * Int64(down_row_words)
                    + Int64(row) * Int64(down_row_words)
                    + Int64(k64_slice * Int32(8))
                    + Int64(vec * Int32(4))
                )
                cp_async4_shared_global(
                    b_base + row * Int32(32) + (vec << Int32(4)),
                    get_ptr_as_int64(down_b_u32, src_word),
                )

        # One packed K64 scale word per hidden row, gathered from the
        # published F8_128x4-swizzled sfb_down plane (one padded plane per
        # expert; the expert stride is the padded plane size).
        if tid < Int32(self.tile_n):
            sfb_atom_bytes = Int64(intermediate_tiles * Int32(2)) * Int64(512)
            # Publisher planes are compact: each expert holds
            # output_n128_tiles padded 128-row atoms of sfb_atom_bytes.
            sfb_expert_stride = Int64(output_n128_tiles) * sfb_atom_bytes
            n_local = output_tile * Int32(self.tile_n) + tid
            sf_src = (
                Int64(expert_idx) * sfb_expert_stride
                + Int64(n_local >> Int32(7)) * sfb_atom_bytes
                + Int64(k64_slice * Int32(512))
                + Int64(((n_local & Int32(127)) % Int32(32)) * Int32(16))
                + Int64(((n_local & Int32(127)) >> Int32(5)) * Int32(4))
            )
            cp_async_u32_shared_global(
                sfb_base + (tid << Int32(2)),
                get_ptr_as_int64(sfb_down, sf_src),
            )

    @cute.jit
    def _run_task(
        self,
        intermediate_u32: cute.Tensor,
        down_b_u32: cute.Tensor,
        sfb_down: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        down_alpha: cute.Tensor,
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
        output_n128_tiles: Int32,
    ):
        lane = tid & Int32(31)
        q = lane >> Int32(2)
        c = lane & Int32(3)

        self._stage_slice(
            intermediate_u32,
            down_b_u32,
            sfb_down,
            smem_base,
            tid,
            source_m_tile,
            m_half,
            expert_idx,
            output_tile,
            Int32(0),
            rows_capacity,
            intermediate_tiles,
            output_n128_tiles,
        )
        cute.arch.cp_async_commit_group()

        # Each warp owns M64 (four M16 blocks) x N32 (four n8 groups).
        facc = tuple(
            tuple(cute.make_rmem_tensor((4,), cutlass.Float32) for _nt in range(4))
            for _blk in range(4)
        )
        for blk in cutlass.range_constexpr(4):
            for nt in cutlass.range_constexpr(4):
                facc[blk][nt].fill(0.0)

        intermediate_k64_tiles = intermediate_tiles * Int32(2)
        k64_slice = Int32(0)
        while k64_slice < intermediate_k64_tiles:
            stage = k64_slice & Int32(1)
            stage_base = smem_base + stage * Int32(self.stage_bytes)
            a_base = stage_base + Int32(self.a_offset)
            sfa_base = stage_base + Int32(self.sfa_offset)
            b_base = stage_base + Int32(self.b_offset)
            sfb_base = stage_base + Int32(self.sfb_offset)

            next_slice = k64_slice + Int32(1)
            if next_slice < intermediate_k64_tiles:
                self._stage_slice(
                    intermediate_u32,
                    down_b_u32,
                    sfb_down,
                    smem_base,
                    tid,
                    source_m_tile,
                    m_half,
                    expert_idx,
                    output_tile,
                    next_slice,
                    rows_capacity,
                    intermediate_tiles,
                    output_n128_tiles,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            # QMMA fragment addressing (see nvfp4_mma_m16n8k64_f32_e2m1):
            # A reg r nib n: row = q + 8*(r%2), k = 32*(r//2) + 8*c + n.
            # B reg j nib n: row = q, k = 32*j + 8*c + n.  Staged K64 rows
            # are 32 linear bytes; the lane's c selects the 4-byte k group
            # within the half and the register selects the half.  The B row
            # for n8 group g is 8*(4*warp + g) + q — pinned by the QMMA
            # fragment probes.
            b_lo = cute.make_rmem_tensor((4,), Uint32)
            b_hi = cute.make_rmem_tensor((4,), Uint32)
            sfb_w = cute.make_rmem_tensor((4,), Uint32)
            for g in cutlass.range_constexpr(4):
                b_row = Int32(8) * (warp_idx * Int32(4) + Int32(g)) + (
                    lane >> Int32(2)
                )
                b_lo[g] = ld_shared_u32(b_base + b_row * Int32(32) + Int32(4) * c)
                b_hi[g] = ld_shared_u32(
                    b_base + b_row * Int32(32) + Int32(4) * c + Int32(16)
                )
                sfb_col = Int32(8) * (warp_idx * Int32(4) + Int32(g)) + (
                    lane >> Int32(2)
                )
                sfb_w[g] = ld_shared_u32(sfb_base + (sfb_col << Int32(2)))

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
                    fragment = facc[blk][g]
                    d0, d1, d2, d3 = nvfp4_mma_m16n8k64_f32_e2m1(
                        fragment[0],
                        fragment[1],
                        fragment[2],
                        fragment[3],
                        a_frag[0],
                        a_frag[1],
                        a_frag[2],
                        a_frag[3],
                        b_lo[g],
                        b_hi[g],
                        sfa_w,
                        sfb_w[g],
                    )
                    fragment[0] = d0
                    fragment[1] = d1
                    fragment[2] = d2
                    fragment[3] = d3

            cute.arch.sync_threads()
            k64_slice += Int32(1)

        # Drain the final (possibly empty) committed group before the task
        # loop reuses the shared pipeline region.
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        scatter_n = Int32(scatter_output.shape[1])
        physical_row_base = source_m_tile * Int32(self.source_tile_m) + m_half * Int32(
            self.tile_m
        )
        # down_scale = down_alpha[e]: the monolithic NVFP4 kernel applies only
        # the weight dequant alpha at FC2 (the a2 global scale is already
        # folded into the materialized SFA2 plane).
        down_scale = down_alpha[expert_idx].to(cutlass.Float32)
        col_base = (
            output_tile * Int32(self.tile_n)
            + warp_idx * Int32(32)
            + (c << Int32(1))
        )
        for nt in cutlass.range_constexpr(4):
            col = col_base + Int32(nt * 8)
            for blk in cutlass.range_constexpr(4):
                fragment = facc[blk][nt]
                row_lo = Int32(blk * 16) + q
                row_hi = row_lo + Int32(8)
                if row_lo < valid_rows:
                    phys_row = physical_row_base + row_lo
                    tok = token_map[phys_row].to(Int32)
                    scale = down_scale * token_weights[phys_row].to(
                        cutlass.Float32
                    )
                    if cutlass.const_expr(self.deterministic_output):
                        st_global_u32(
                            get_ptr_as_int64(
                                scatter_output, Int64(tok) * Int64(scatter_n) + Int64(col)
                            ),
                            pack_f32x2_to_bfloat2(scale * fragment[0], scale * fragment[1]),
                        )
                    else:
                        scatter_add_bf16x2(
                            get_ptr_as_int64(
                                scatter_output, Int64(tok) * Int64(scatter_n) + Int64(col)
                            ),
                            scale * fragment[0],
                            scale * fragment[1],
                        )
                if row_hi < valid_rows:
                    phys_row = physical_row_base + row_hi
                    tok = token_map[phys_row].to(Int32)
                    scale = down_scale * token_weights[phys_row].to(
                        cutlass.Float32
                    )
                    if cutlass.const_expr(self.deterministic_output):
                        st_global_u32(
                            get_ptr_as_int64(
                                scatter_output, Int64(tok) * Int64(scatter_n) + Int64(col)
                            ),
                            pack_f32x2_to_bfloat2(scale * fragment[2], scale * fragment[3]),
                        )
                    else:
                        scatter_add_bf16x2(
                            get_ptr_as_int64(
                                scatter_output, Int64(tok) * Int64(scatter_n) + Int64(col)
                            ),
                            scale * fragment[2],
                            scale * fragment[3],
                        )

    @cute.kernel
    def kernel(
        self,
        intermediate_u32: cute.Tensor,
        down_b_u32: cute.Tensor,
        sfb_down: cute.Tensor,
        scatter_output: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        down_alpha: cute.Tensor,
        intermediate_tiles: cutlass.Int32,
        output_n128_tiles: cutlass.Int32,
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
        task_tail = source_m_tiles * Int32(self.source_halves) * output_n128_tiles
        task_slot = Int32(bidz)
        while task_slot < task_tail:
            output_tile = task_slot % output_n128_tiles
            source_half = task_slot // output_n128_tiles
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
                    down_b_u32,
                    sfb_down,
                    scatter_output,
                    token_map,
                    token_weights,
                    down_alpha,
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
                    output_n128_tiles,
                )
            task_slot += Int32(gdimz)


__all__ = ["Nvfp4MaterializedPhase2Kernel"]
