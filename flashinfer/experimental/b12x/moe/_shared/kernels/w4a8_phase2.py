"""Standalone materialized-W4A8 FC2 kernel for dense prefill regimes.

The dynamic front-end and phase 1 publish an expert-major M64 or M128 source
domain in MXFP8 form. Keeping FC2 in the routing kernel would union phase-A
and phase-B shared-memory/register requirements. This kernel consumes
materialized source tiles as compact M16 or M64 compute chunks over N128.
The launch is fixed-capacity and stream ordered, so it remains safe for CUDA
graph capture and replay without host reads or allocations.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import Int32, Int64, Uint8, Uint32

from b12x._lib.intrinsics import (
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
    st_shared_u8,
)
from b12x._lib.intrinsics import (
    mxfp8_mma_m16n8k32_f32_e4m3,
    st_shared_u32,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_stage_trellis_b_tile,
    _w4a8_trellis_lane_geom,
    _w4a8_trellis_pair_words_both,
)


class W4A8MaterializedPhase2Kernel:
    """Consume materialized source tiles through compact FC2 CTAs."""

    tile_m = 64
    source_tile_m = 128
    tile_n = 128
    tile_k = 128
    num_warps = 4
    threads_per_cta = num_warps * 32
    stages = 2

    # Per stage: tile_m x K128 E4M3 A, tile_m packed A-scale words, tile_n x
    # K128 FP4 B, and tile_n/8 x 8 packed B-scale words. Keep the large
    # regions 1 KiB aligned.
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
        trellis_bits: int | None = None,
        trellis_direct_lut: bool = False,
        n64_repacked: bool = False,
        n64_tail: bool = False,
        direct_routes: bool = False,
    ):
        self.n64_repacked = bool(n64_repacked)
        self.n64_tail = bool(n64_tail)
        self.direct_routes = bool(direct_routes)
        if self.direct_routes:
            if source_tile_m != 1:
                raise ValueError(
                    "direct materialized phase 2 requires source_tile_m=1"
                )
        elif source_tile_m not in (16, 64, 128):
            raise ValueError(
                "materialized phase 2 source_tile_m must be 16, 64, or 128, "
                f"got {source_tile_m}"
            )
        self.tile_m = 16 if source_tile_m in (1, 16) else type(self).tile_m
        self.tile_n = type(self).tile_n
        self.stages = 4 if self.direct_routes else 2
        self.a_payload_bytes = (
            self.tile_k if self.direct_routes else self.tile_m * self.tile_k
        )
        self.a_scale_bytes = 128 if self.direct_routes else self.tile_m * 4
        self.a_stage_bytes = self.a_payload_bytes + self.a_scale_bytes
        self.a_storage_bytes = self.stages * self.a_stage_bytes
        self.b_storage_offset = ((self.a_storage_bytes + 1023) // 1024) * 1024
        self.b_stage_bytes = self.tile_n * self.tile_k // 2
        self.b_storage_bytes = self.stages * self.b_stage_bytes
        self.sfb_storage_offset = self.b_storage_offset + self.b_storage_bytes
        self.sfb_stage_bytes = (self.tile_n // 8) * 8 * 4
        self.shared_bytes = self.sfb_storage_offset + self.stages * self.sfb_stage_bytes
        self.shared_words = (self.shared_bytes + 3) // 4
        self.source_tile_m = int(source_tile_m)
        self.source_halves = (
            1 if self.direct_routes else self.source_tile_m // self.tile_m
        )
        self.deterministic_output = bool(deterministic_output)
        if trellis_bits is not None and trellis_bits not in (2, 3, 4):
            raise ValueError(
                f"trellis_bits must be 2, 3, or 4, got {trellis_bits!r}"
            )
        self.w4a8_trellis = trellis_bits is not None
        self.trellis_bits = 0 if trellis_bits is None else int(trellis_bits)
        # Direct-LUT decode gathers each byte from the rate-indexed 192 KiB
        # global table instead of reading a 4 KiB shared value table; the
        # shared region is then not allocated.
        self.trellis_direct_lut = bool(trellis_direct_lut) and self.w4a8_trellis
        if self.w4a8_trellis:
            self.trellis_lut_offset = self.shared_bytes
            if not self.trellis_direct_lut:
                self.shared_words = (self.shared_bytes + 4096 + 3) // 4

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
        trellis_lut: cute.Tensor,
        intermediate_tiles: cutlass.Int32,
        packed_output_tiles: cutlass.Int32,
        max_active_clusters: cutlass.Int32,
        num_pairs: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        grid_z = max_active_clusters * Int32(2)
        if cutlass.const_expr(self.direct_routes):
            grid_z = num_pairs * packed_output_tiles * Int32(256 // self.tile_n)
            if max_active_clusters > Int32(0):
                grid_z = cutlass.min(grid_z, max_active_clusters * Int32(2))
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
            trellis_lut,
            intermediate_tiles,
            packed_output_tiles,
            num_pairs,
        ).launch(
            grid=(1, 1, grid_z),
            block=[self.threads_per_cta, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    @cute.jit
    def _stage_n64_b(
        self,
        weights: cute.Tensor,
        dst_base: Int32,
        expert_idx: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        total_output_tiles: Int32,
        total_k32: cutlass.Constexpr,
        tid: Int32,
    ):
        transfers = 4 * 4 * 32
        total_output_n = total_output_tiles * Int32(128)
        expert_words = total_output_n * Int32(total_k32 * 4)
        output_base = output_tile * Int32(128 * total_k32 * 4)
        k32_count = Int32(4)
        if cutlass.const_expr(self.n64_tail):
            if k128_slice == Int32(total_k32 // 4):
                k32_count = Int32(2)
        for i in cutlass.range_constexpr(
            (transfers + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(transfers):
                lane = idx & Int32(31)
                tmp = idx >> Int32(5)
                n32 = tmp & Int32(3)
                kb = tmp >> Int32(2)
                dst_transfer = (kb * Int32(4) + n32) * Int32(32) + lane
                dst_addr = dst_base + (dst_transfer << Int32(4))
                if kb < k32_count:
                    source_word = (
                        Int64(expert_idx) * Int64(expert_words)
                        + Int64(output_base)
                        + Int64(k128_slice * Int32(2048))
                        + Int64(kb * Int32(512))
                        + Int64(n32 * Int32(128))
                        + Int64(lane * Int32(4))
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
        output_tile: Int32,
        k128_slice: Int32,
        total_output_tiles: Int32,
        total_k32: cutlass.Constexpr,
        tid: Int32,
    ):
        scales_u8 = cute.recast_tensor(scales, Uint8)
        copies = 4 * 128
        total_output_n = total_output_tiles * Int32(128)
        expert_bytes = total_output_n * Int32(total_k32)
        output_base = output_tile * Int32(128 * total_k32)
        tile_cols = Int32(4)
        if cutlass.const_expr(self.n64_tail):
            if k128_slice == Int32(total_k32 // 4):
                tile_cols = Int32(2)
        for i in cutlass.range_constexpr(
            (copies + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(copies):
                row = idx & Int32(127)
                kb = idx >> Int32(7)
                value = Uint8(0)
                if kb < tile_cols:
                    source_byte = (
                        Int64(expert_idx) * Int64(expert_bytes)
                        + Int64(output_base)
                        + Int64(k128_slice * Int32(512))
                        + Int64(row * tile_cols + kb)
                    )
                    value = scales_u8[source_byte]
                st_shared_u8(dst_base + row * Int32(4) + kb, value)

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
        stage = intermediate_slice & Int32(self.stages - 1)
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

        if cutlass.const_expr(self.direct_routes):
            # Broadcast one compact route row into the logical M16 operand.
            if tid < Int32(8):
                src_word = (
                    Int64(physical_row_base) * Int64(words_per_row)
                    + Int64(intermediate_slice) * Int64(32)
                    + Int64(tid * 4)
                )
                cp_async4_shared_global(
                    a_base + tid * Int32(16),
                    get_ptr_as_int64(intermediate_u32, src_word),
                )
            if tid == Int32(0):
                sf_src = (
                    Int64(rows_capacity) * Int64(words_per_row)
                    + Int64(intermediate_slice) * Int64(rows_capacity)
                    + Int64(physical_row_base)
                )
                cp_async_u32_shared_global(
                    sfa_base,
                    get_ptr_as_int64(intermediate_u32, sf_src),
                )
        else:
            # Grouped source tiles retain the XOR row layout.
            for i in cutlass.range_constexpr(
                (self.tile_m * 8 + self.threads_per_cta - 1)
                // self.threads_per_cta
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
                        a_base
                        + row * Int32(self.tile_k)
                        + (physical_vec << Int32(4)),
                        get_ptr_as_int64(intermediate_u32, src_word),
                    )
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

        if cutlass.const_expr(self.w4a8_trellis):
            # Expert-major [E][K16][N16] trellis windows over the down
            # projection (K = intermediate, N = hidden); stage the slice's
            # eight K16 rows for this N128 output tile. SFB stays unstaged.
            tr_n16_cnt = packed_output_tiles * Int32(16)
            tr_k16_stride = tr_n16_cnt * Int32(8 * self.trellis_bits)
            tr_eu = Int64(intermediate_tiles * Int32(8)) * Int64(tr_k16_stride)
            tr_base = (
                Int64(expert_idx) * tr_eu
                + Int64(intermediate_slice * Int32(8)) * Int64(tr_k16_stride)
                + Int64(output_tile * Int32(8))
                * Int64(8 * self.trellis_bits)
            )
            _w4a8_stage_trellis_b_tile(
                down_rp,
                b_base,
                tr_base,
                tr_k16_stride,
                self.trellis_bits,
                tid,
                self.threads_per_cta,
                8,
            )
        elif cutlass.const_expr(self.n64_repacked):
            total_output_tiles = packed_output_tiles * Int32(2)
            total_k32 = intermediate_tiles * Int32(4)
            if cutlass.const_expr(self.n64_tail):
                total_k32 -= Int32(2)
            self._stage_n64_b(
                down_rp,
                b_base,
                expert_idx,
                output_tile,
                intermediate_slice,
                total_output_tiles,
                total_k32,
                tid,
            )
            self._stage_n64_sfb(
                down_sfb_rp,
                sfb_base,
                expert_idx,
                output_tile,
                intermediate_slice,
                total_output_tiles,
                total_k32,
                tid,
            )
        else:
            # Prepared weights remain N256 tile-major. Compact this CTA's
            # consecutive N32 chunks into its shared tile.
            n32_chunks = self.tile_n // 32
            packed_tile = output_tile // Int32(256 // self.tile_n)
            packed_n32 = (output_tile % Int32(256 // self.tile_n)) * Int32(
                n32_chunks
            )
            b_tile = (
                expert_idx * packed_output_tiles + packed_tile
            ) * intermediate_tiles + intermediate_slice
            b_word_base = Int64(b_tile) * Int64(4096)
            for i in cutlass.range_constexpr(
                (4 * n32_chunks * 32 + self.threads_per_cta - 1)
                // self.threads_per_cta
            ):
                idx = tid + Int32(i * self.threads_per_cta)
                if idx < Int32(4 * n32_chunks * 32):
                    lane = idx & Int32(31)
                    kc = idx >> Int32(5)
                    chunk = kc % Int32(n32_chunks)
                    kb = kc // Int32(n32_chunks)
                    src_word = (
                        b_word_base
                        + Int64(kb * 8 * 32 * 4)
                        + Int64((packed_n32 + chunk) * Int32(32 * 4))
                        + Int64(lane * Int32(4))
                    )
                    cp_async4_shared_global(
                        b_base + (idx << Int32(4)),
                        get_ptr_as_int64(down_rp, src_word),
                    )

            # An N32 chunk has four N8 scale rows, each with eight K16 words.
            sfb_word_base = Int64(b_tile) * Int64(256)
            for i in cutlass.range_constexpr(
                (((self.tile_n // 8) * 8) // 4 + self.threads_per_cta - 1)
                // self.threads_per_cta
            ):
                idx = tid + Int32(i * self.threads_per_cta)
                if idx < Int32(((self.tile_n // 8) * 8) // 4):
                    src_word = sfb_word_base + Int64(
                        packed_n32 * Int32(4 * 8) + idx * 4
                    )
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
        trellis_lut: cute.Tensor,
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
        if cutlass.const_expr(self.w4a8_trellis):
            tr_ia, tr_ib, tr_s2 = _w4a8_trellis_lane_geom(
                lane, self.trellis_bits
            )
            trellis_lut_addr = Int64(
                smem_base + Int32(self.trellis_lut_offset)
            )
            if cutlass.const_expr(self.trellis_direct_lut):
                trellis_lut_addr = trellis_lut.iterator.toint()

        # Keep enough groups in flight to hide global-memory latency while the
        # current K128 slice is consumed. Each preload gets its own group,
        # including the guarded-empty tail groups, so wait_group below retains
        # the current slice's data.
        for preload_slice in cutlass.range_constexpr(self.stages - 1):
            if Int32(preload_slice) < intermediate_tiles:
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
                    Int32(preload_slice),
                    rows_capacity,
                    intermediate_tiles,
                    packed_output_tiles,
                )
            cute.arch.cp_async_commit_group()

        # Each warp owns tile_m x tile_n/4 and its tile_n/32 N8 fragments.
        m_blocks = self.tile_m // 16
        n_fragments = self.tile_n // 32
        facc = tuple(
            tuple(
                cute.make_rmem_tensor((4,), cutlass.Float32)
                for _nt in range(n_fragments)
            )
            for _blk in range(m_blocks)
        )
        for blk in cutlass.range_constexpr(m_blocks):
            for nt in cutlass.range_constexpr(n_fragments):
                facc[blk][nt].fill(0.0)

        intermediate_slice = Int32(0)
        while intermediate_slice < intermediate_tiles:
            stage = intermediate_slice & Int32(self.stages - 1)
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

            next_slice = intermediate_slice + Int32(self.stages - 1)
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
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            asc = cute.make_rmem_tensor((m_blocks,), Uint32)
            for blk in cutlass.range_constexpr(m_blocks):
                if cutlass.const_expr(self.direct_routes):
                    asc[blk] = ld_shared_u32(sfa_base)
                else:
                    sf_row = Int32(blk * 16) + q + (
                        (lane & Int32(1)) << Int32(3)
                    )
                    asc[blk] = ld_shared_u32(sfa_base + (sf_row << Int32(2)))

            for kb in cutlass.range_constexpr(4):
                u_phys = (Int32(kb * 2) + (c >> Int32(1))) ^ q
                a_frag = cute.make_rmem_tensor((m_blocks, 4), Uint32)
                for blk in cutlass.range_constexpr(m_blocks):
                    if cutlass.const_expr(self.direct_routes):
                        a0, a2 = ld_shared_v2_u32(
                            a_base + Int32(kb * 32) + c * Int32(8)
                        )
                        a1, a3 = a0, a2
                    else:
                        a_lo = (
                            a_base
                            + Int32(blk * 16 * self.tile_k)
                            + (q << Int32(7))
                            + (u_phys << Int32(4))
                            + ((c & Int32(1)) << Int32(3))
                        )
                        a0, a2 = ld_shared_v2_u32(a_lo)
                        a1, a3 = ld_shared_v2_u32(
                            a_lo + Int32(8 * self.tile_k)
                        )
                    a_frag[blk, 0] = a0
                    a_frag[blk, 1] = a1
                    a_frag[blk, 2] = a2
                    a_frag[blk, 3] = a3

                dn_b0 = cute.make_rmem_tensor((n_fragments,), Uint32)
                dn_b1 = cute.make_rmem_tensor((n_fragments,), Uint32)
                if cutlass.const_expr(self.w4a8_trellis):
                    for th in cutlass.range_constexpr(2):
                        tr_n16 = warp_idx * Int32(2) + Int32(th)
                        tr_b0 = (Int32(kb * 16) + tr_n16) * Int32(
                            8 * self.trellis_bits
                        )
                        d_lo0, d_lo1, d_hi0, d_hi1 = (
                            _w4a8_trellis_pair_words_both(
                                b_base,
                                lane,
                                tr_b0,
                                tr_b0 + Int32(64 * self.trellis_bits),
                                tr_ia,
                                tr_ib,
                                tr_s2,
                                self.trellis_bits,
                                trellis_lut_addr,
                                not self.trellis_direct_lut,
                                self.trellis_direct_lut,
                            )
                        )
                        dn_b0[th * 2] = d_lo0
                        dn_b1[th * 2] = d_lo1
                        dn_b0[th * 2 + 1] = d_hi0
                        dn_b1[th * 2 + 1] = d_hi1
                else:
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
                        wb0, wb1 = e2m1x8_to_qmma_e2m1x8(words[nt])
                        dn_b0[nt] = wb0
                        dn_b1[nt] = wb1
                for nt in cutlass.range_constexpr(n_fragments):
                    n8 = warp_idx * Int32(n_fragments) + Int32(nt)
                    b0 = dn_b0[nt]
                    b1 = dn_b1[nt]
                    sfb_word = Uint32(0x7F7F7F7F)
                    if cutlass.const_expr(not self.w4a8_trellis):
                        sfb_word = ld_shared_u32(
                            sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                        )
                    for blk in cutlass.range_constexpr(m_blocks):
                        fragment = facc[blk][nt]
                        if cutlass.const_expr(self.w4a8_trellis):
                            d0, d1, d2, d3 = mxfp8_mma_m16n8k32_f32_e4m3(
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
                        else:
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
            0 if cutlass.const_expr(global_scale.shape[0] == 1) else expert_idx
        ].to(cutlass.Float32)
        col_base = (
            output_tile * Int32(self.tile_n)
            + warp_idx * Int32(self.tile_n // 4)
            + (c << Int32(1))
        )
        for nt in cutlass.range_constexpr(n_fragments):
            col = col_base + Int32(nt * 8)
            for blk in cutlass.range_constexpr(m_blocks):
                fragment = facc[blk][nt]
                row_lo = Int32(blk * 16) + q
                row_hi = row_lo + Int32(8)
                if row_lo < valid_rows:
                    physical_row = physical_row_base + row_lo
                    tok = (
                        source_m_tile
                        if cutlass.const_expr(self.direct_routes)
                        else token_map[physical_row].to(Int32)
                    )
                    scale = down_scale * token_weights[physical_row].to(cutlass.Float32)
                    if cutlass.const_expr(self.deterministic_output):
                        # The routing front-end stores the token-major pair
                        # index in token_map for deterministic specializations.
                        # Each pair/output-column location has one producer, so
                        # phase 2 can write it exactly once; the caller then
                        # reduces routes in fixed top-k order.
                        st_global_u32(
                            get_ptr_as_int64(scatter_output, Int64(tok) * Int64(scatter_n) + Int64(col)),
                            pack_f32x2_to_bfloat2(
                                scale * fragment[0], scale * fragment[1]
                            ),
                        )
                    else:
                        scatter_add_bf16x2(
                            get_ptr_as_int64(scatter_output, Int64(tok) * Int64(scatter_n) + Int64(col)),
                            scale * fragment[0],
                            scale * fragment[1],
                        )
                if row_hi < valid_rows:
                    physical_row = physical_row_base + row_hi
                    tok = (
                        source_m_tile
                        if cutlass.const_expr(self.direct_routes)
                        else token_map[physical_row].to(Int32)
                    )
                    scale = down_scale * token_weights[physical_row].to(cutlass.Float32)
                    if cutlass.const_expr(self.deterministic_output):
                        st_global_u32(
                            get_ptr_as_int64(scatter_output, Int64(tok) * Int64(scatter_n) + Int64(col)),
                            pack_f32x2_to_bfloat2(
                                scale * fragment[2], scale * fragment[3]
                            ),
                        )
                    else:
                        scatter_add_bf16x2(
                            get_ptr_as_int64(scatter_output, Int64(tok) * Int64(scatter_n) + Int64(col)),
                            scale * fragment[2],
                            scale * fragment[3],
                        )

        # Drain guarded-empty tail groups before this persistent CTA reuses
        # shared staging storage for its next task.
        cute.arch.cp_async_wait_group(0)

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
        trellis_lut: cute.Tensor,
        intermediate_tiles: cutlass.Int32,
        packed_output_tiles: cutlass.Int32,
        num_pairs: cutlass.Int32,
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
        if cutlass.const_expr(
            self.w4a8_trellis and not self.trellis_direct_lut
        ):
            trellis_lut_u32 = cute.recast_tensor(trellis_lut, cutlass.Uint32)
            lut_copy_i = Int32(tidx)
            while lut_copy_i < Int32(1024):
                st_shared_u32(
                    smem_base
                    + Int32(self.trellis_lut_offset)
                    + (lut_copy_i << Int32(2)),
                    Uint32(trellis_lut_u32[lut_copy_i]),
                )
                lut_copy_i += Int32(self.threads_per_cta)
            cute.arch.sync_threads()

        rows_capacity = Int32(token_map.shape[0])
        output_tiles = packed_output_tiles * Int32(256 // self.tile_n)
        if cutlass.const_expr(self.direct_routes):
            task_tail = num_pairs * output_tiles
        else:
            num_experts = Int32(expert_tile_base.shape[0] - 1)
            source_m_tiles = expert_tile_base[num_experts].to(Int32)
            task_tail = source_m_tiles * Int32(self.source_halves) * output_tiles
        task_slot = Int32(bidz)
        while task_slot < task_tail:
            output_tile = task_slot % output_tiles
            source_half = task_slot // output_tiles
            if cutlass.const_expr(self.direct_routes):
                m_half = Int32(0)
                source_m_tile = source_half
                expert_idx = task_expert[source_m_tile].to(Int32)
                valid_rows = Int32(1)
                if expert_idx < Int32(0):
                    valid_rows = Int32(0)
                if expert_idx >= Int32(down_alpha.shape[0]):
                    valid_rows = Int32(0)
            else:
                m_half = source_half % Int32(self.source_halves)
                source_m_tile = source_half // Int32(self.source_halves)
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
                    trellis_lut,
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
            elif cutlass.const_expr(self.direct_routes):
                if tid < Int32(self.tile_n):
                    scatter_output[
                        source_m_tile,
                        output_tile * Int32(self.tile_n) + tid,
                    ] = cutlass.BFloat16(0.0)
            task_slot += Int32(gdimz)


__all__ = ["W4A8MaterializedPhase2Kernel"]
