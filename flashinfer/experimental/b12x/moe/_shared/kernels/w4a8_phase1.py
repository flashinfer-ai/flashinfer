"""Standalone materialized-W4A8 FC1 kernel for dense prefill regimes.

The dynamic front-end still owns routing, input quantization, and publication
of a compact expert-major M64 or M128 source domain. This kernel consumes that
domain in M64 compute chunks, computes gate and up together in one K
sweep, applies the selected gated activation, and writes the existing
caller-owned MXFP8 intermediate workspace.

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

from b12x._lib.intrinsics import (
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
    mxfp8_mma_m16n8k32_f32_e4m3,
    pack_f32x2_to_bfloat2,
    quantize_block_fp8_mx,
    shared_ptr_to_u32,
    st_shared_u32,
)
from b12x.moe._shared.kernels.w4a8_trellis_decode import (
    _w4a8_had128_quad,
    _w4a8_stage_trellis_b_tile,
    _w4a8_trellis_lane_geom,
    _w4a8_trellis_pair_words_both,
    _w4a8_trellis_permute_k32,
)
from b12x.moe._shared.kernels.activations import (
    SITU,
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
)


class W4A8MaterializedPhase1Kernel:
    """Compute one FC1 compute chunk of each published source tile per task."""

    source_tile_m = 128
    tile_n = 128
    tile_k = 64
    num_warps = 4
    threads_per_cta = num_warps * 32
    stages = 2

    def __init__(
        self,
        *,
        fast_math: bool = False,
        source_tile_m: int = 128,
        deterministic_output: bool = False,
        num_topk: int = 1,
        activation: str = "silu",
        swiglu_limit: float | None = None,
        trellis_bits: int | None = None,
        trellis_intermediate_hadamard: bool = False,
        trellis_direct_lut: bool = False,
        n64_repacked: bool = False,
        n64_tail: bool = False,
    ):
        self.fast_math = bool(fast_math)
        self.has_swiglu_limit = swiglu_limit is not None
        self.swiglu_limit = 0.0 if swiglu_limit is None else float(swiglu_limit)
        self.n64_repacked = bool(n64_repacked)
        self.n64_tail = bool(n64_tail)
        if source_tile_m not in (16, 64, 128):
            raise ValueError(
                "materialized phase 1 source_tile_m must be 16, 64, or 128, "
                f"got {source_tile_m}"
            )
        self.tile_m = 16 if source_tile_m == 16 else 64
        self.tile_n = 128
        self.stages = 2
        # A retains a 128-byte row stride even though each stage advances K64.
        # The XOR vector mapping used by QMMA can address all eight 16-byte
        # slots for any row; the unused half is merely padding. Each B half is
        # a compact NxK64 FP4 tile. Gate and up are staged side by side.
        self.a_payload_bytes = self.tile_m * 128
        self.a_scale_bytes = self.tile_m * 4
        self.b_payload_bytes = self.tile_n * self.tile_k // 2
        self.sfb_bytes = (self.tile_n // 8) * 8 * 4
        self.a_offset = 0
        self.sfa_offset = self.a_offset + self.a_payload_bytes
        self.gate_b_offset = self.sfa_offset + self.a_scale_bytes
        self.up_b_offset = self.gate_b_offset + self.b_payload_bytes
        self.gate_sfb_offset = self.up_b_offset + self.b_payload_bytes
        self.up_sfb_offset = self.gate_sfb_offset + self.sfb_bytes
        self.stage_bytes = self.up_sfb_offset + self.sfb_bytes
        self.pipeline_bytes = self.stages * self.stage_bytes
        self.epilogue_bytes = self.tile_m * self.tile_n * 2
        self.shared_bytes = max(self.pipeline_bytes, self.epilogue_bytes)
        self.shared_words = (self.shared_bytes + 3) // 4
        self.source_tile_m = int(source_tile_m)
        self.source_halves = self.source_tile_m // self.tile_m
        self.deterministic_output = bool(deterministic_output)
        if int(num_topk) <= 0:
            raise ValueError(f"num_topk must be positive, got {num_topk}")
        self.num_topk = int(num_topk)
        if activation not in {"silu", SITU}:
            raise ValueError(
                "materialized W4A8 phase 1 requires silu or situ, "
                f"got {activation!r}"
            )
        self.is_situ = activation == SITU
        if trellis_bits is not None and trellis_bits not in (2, 3, 4):
            raise ValueError(
                f"trellis_bits must be 2, 3, or 4, got {trellis_bits!r}"
            )
        self.w4a8_trellis = trellis_bits is not None
        self.trellis_bits = 0 if trellis_bits is None else int(trellis_bits)
        if trellis_intermediate_hadamard and not self.w4a8_trellis:
            raise ValueError("trellis_intermediate_hadamard requires trellis_bits")
        self.trellis_intermediate_hadamard = bool(trellis_intermediate_hadamard)
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
        packed_a_storage: cute.Tensor,
        scale_storage: cute.Tensor,
        w13_rp: cute.Tensor,
        w13_sfb_rp: cute.Tensor,
        intermediate_u32: cute.Tensor,
        token_map: cute.Tensor,
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        alpha: cute.Tensor,
        input_global_scale: cute.Tensor,
        trellis_lut: cute.Tensor,
        trellis_rotations: cute.Tensor,
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
            token_weights,
            task_expert,
            task_valid_rows,
            expert_tile_base,
            alpha,
            input_global_scale,
            trellis_lut,
            trellis_rotations,
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
        # two K32 blocks and one tile_n part from the enclosing N256xK128 tile.
        stage_chunk_count = self.tile_n // 32
        for i in cutlass.range_constexpr(
            (2 * stage_chunk_count * 32 + self.threads_per_cta - 1)
            // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(2 * stage_chunk_count * 32):
                lane = idx & Int32(31)
                kc = idx >> Int32(5)
                chunk = kc % Int32(stage_chunk_count)
                kb = kc // Int32(stage_chunk_count) + k_half * Int32(2)
                src_word = (
                    tile_word_base
                    + Int64(kb * Int32(8 * 32 * 4))
                    + Int64(
                        (packed_half * Int32(stage_chunk_count) + chunk)
                        * Int32(32 * 4)
                    )
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
        # Retain the packed K128 scale words needed by this tile.  The two K64
        # epochs use different compile-time QMMA byte ids from that data.
        scale_words = (self.tile_n // 8) * 8
        for i in cutlass.range_constexpr(
            (scale_words // 4 + self.threads_per_cta - 1)
            // self.threads_per_cta
        ):
            idx = tid + Int32(i * self.threads_per_cta)
            if idx < Int32(scale_words // 4):
                src_word = tile_word_base + Int64(
                    packed_half * Int32(scale_words) + idx * 4
                )
                cp_async4_shared_global(
                    dst_base + (idx << Int32(4)),
                    get_ptr_as_int64(scales, src_word),
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
        intermediate_tiles: Int32,
        total_k32: cutlass.Constexpr,
        tid: Int32,
    ):
        logical_n = intermediate_tiles * Int32(128)
        if cutlass.const_expr(self.n64_tail):
            logical_n -= Int32(64)
        n_rows = Int32(128)
        if cutlass.const_expr(self.n64_tail):
            if output_tile == intermediate_tiles - Int32(1):
                n_rows = Int32(64)
        transfers = 2 * 4 * 32
        expert_words = logical_n * Int32(total_k32 * 8)
        projection_words = logical_n * Int32(total_k32 * 4)
        tile_base = output_tile * Int32(128 * total_k32 * 4)
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
                if n32 < n_rows // Int32(32):
                    source_k32 = k64_slice * Int32(2) + kb
                    source_word = (
                        Int64(expert_idx) * Int64(expert_words)
                        + Int64(projection) * Int64(projection_words)
                        + Int64(tile_base)
                        + Int64(source_k32 * (n_rows // Int32(32)) * Int32(128))
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
        projection: Int32,
        output_tile: Int32,
        k128_slice: Int32,
        intermediate_tiles: Int32,
        total_k32: cutlass.Constexpr,
        tid: Int32,
    ):
        logical_n = intermediate_tiles * Int32(128)
        if cutlass.const_expr(self.n64_tail):
            logical_n -= Int32(64)
        n_rows = Int32(128)
        if cutlass.const_expr(self.n64_tail):
            if output_tile == intermediate_tiles - Int32(1):
                n_rows = Int32(64)
        transfers = 512 // 16
        expert_words = logical_n * Int32(total_k32) // Int32(2)
        projection_words = logical_n * Int32(total_k32) // Int32(4)
        tile_base = output_tile * Int32(128 * total_k32 // 4)
        valid_transfers = n_rows // Int32(4)
        for i in cutlass.range_constexpr(
            (transfers + self.threads_per_cta - 1) // self.threads_per_cta
        ):
            transfer = tid + Int32(i * self.threads_per_cta)
            if transfer < Int32(transfers):
                dst_addr = dst_base + (transfer << Int32(4))
                if transfer < valid_transfers:
                    source_word = (
                        Int64(expert_idx) * Int64(expert_words)
                        + Int64(projection) * Int64(projection_words)
                        + Int64(tile_base)
                        + Int64(k128_slice * n_rows)
                        + Int64(transfer * Int32(4))
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
        stage = k64_slice & Int32(self.stages - 1)
        stage_base = smem_base + stage * Int32(self.stage_bytes)
        a_base = stage_base + Int32(self.a_offset)
        sfa_base = stage_base + Int32(self.sfa_offset)
        gate_b_base = stage_base + Int32(self.gate_b_offset)
        up_b_base = stage_base + Int32(self.up_b_offset)
        gate_sfb_base = stage_base + Int32(self.gate_sfb_offset)
        up_sfb_base = stage_base + Int32(self.up_sfb_offset)

        physical_row_base = Int64(source_m_tile) * Int64(self.source_tile_m) + Int64(m_half * self.tile_m)
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
                    Int64(tok) * Int64(words_per_token) + Int64(k64_slice * 16 + vec * 4)
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
            sf_src = Int64(tok) * Int64(input_k128_tiles) * Int64(4) + Int64(k64_slice >> 1) * Int64(4)
            cp_async_u32_shared_global(
                sfa_base + (tid << Int32(2)),
                get_ptr_as_int64(scale_storage, sf_src),
            )

        k128_slice = k64_slice >> Int32(1)
        k_half = k64_slice & Int32(1)
        input_k128_count = input_k128_tiles

        packed_parts = Int32(256 // self.tile_n)
        n128_parts = Int32(128 // self.tile_n)
        up_packed_tile = output_tile // packed_parts
        up_packed_half = output_tile % packed_parts
        gate_tile = output_tile + intermediate_tiles * n128_parts
        gate_packed_tile = gate_tile // packed_parts
        gate_packed_half = gate_tile % packed_parts

        up_tile = (
            expert_idx * packed_w13_tiles + up_packed_tile
        ) * input_k128_count + k128_slice
        gate_tile_idx = (
            expert_idx * packed_w13_tiles + gate_packed_tile
        ) * input_k128_count + k128_slice

        if cutlass.const_expr(self.w4a8_trellis):
            # Projection-major [proj][E][K16][N16] trellis windows (the
            # prepared trellis layout, shared with the micro kernel); stage
            # the four K16 rows of this K64 epoch for gate (projection 0)
            # and up (projection 1). B is fully scaled E4M3 after decode,
            # so the SFB regions stay unstaged.
            tr_n16_cnt = intermediate_tiles * Int32(8)
            tr_k16_stride = tr_n16_cnt * Int32(8 * self.trellis_bits)
            tr_eu = Int64(input_k128_tiles * Int32(8)) * Int64(tr_k16_stride)
            tr_w13_half = Int64(w13_rp.shape[0]) >> Int64(1)
            tr_common = (
                Int64(expert_idx) * tr_eu
                + Int64(k64_slice * Int32(4)) * Int64(tr_k16_stride)
                + Int64(output_tile * Int32(8))
                * Int64(8 * self.trellis_bits)
            )
            _w4a8_stage_trellis_b_tile(
                w13_rp,
                gate_b_base,
                tr_common,
                tr_k16_stride,
                self.trellis_bits,
                tid,
                self.threads_per_cta,
                4,
            )
            _w4a8_stage_trellis_b_tile(
                w13_rp,
                up_b_base,
                tr_common + tr_w13_half,
                tr_k16_stride,
                self.trellis_bits,
                tid,
                self.threads_per_cta,
                4,
            )
        elif cutlass.const_expr(self.n64_repacked):
            total_k32 = input_k128_tiles * Int32(4)
            self._stage_n64_b_k64(
                w13_rp,
                gate_b_base,
                expert_idx,
                Int32(1),
                output_tile,
                k64_slice,
                intermediate_tiles,
                total_k32,
                tid,
            )
            self._stage_n64_b_k64(
                w13_rp,
                up_b_base,
                expert_idx,
                Int32(0),
                output_tile,
                k64_slice,
                intermediate_tiles,
                total_k32,
                tid,
            )
            self._stage_n64_sfb(
                w13_sfb_rp,
                gate_sfb_base,
                expert_idx,
                Int32(1),
                output_tile,
                k128_slice,
                intermediate_tiles,
                total_k32,
                tid,
            )
            self._stage_n64_sfb(
                w13_sfb_rp,
                up_sfb_base,
                expert_idx,
                Int32(0),
                output_tile,
                k128_slice,
                intermediate_tiles,
                total_k32,
                tid,
            )
        else:
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
        if cutlass.const_expr(self.has_swiglu_limit):
            limit = cutlass.Float32(self.swiglu_limit)
            gate = cutlass.min(gate, limit)
            up = cutlass.max(cutlass.min(up, limit), -limit)
        sigmoid = cute.arch.rcp_approx(
            cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=self.fast_math)
        )
        if cutlass.const_expr(self.is_situ):
            beta = cutlass.Float32(SITU_DEFAULT_BETA)
            linear_beta = cutlass.Float32(SITU_DEFAULT_LINEAR_BETA)
            situ_gate = (
                beta
                * cute.math.tanh(gate / beta, fastmath=self.fast_math)
                * sigmoid
            )
            situ_up = linear_beta * cute.math.tanh(
                up / linear_beta,
                fastmath=self.fast_math,
            )
            return situ_gate * situ_up
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
        token_weights: cute.Tensor,
        alpha: cute.Tensor,
        input_global_scale: cute.Tensor,
        trellis_lut: cute.Tensor,
        trellis_rotations: cute.Tensor,
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
        if cutlass.const_expr(self.w4a8_trellis):
            tr_ia, tr_ib, tr_s2 = _w4a8_trellis_lane_geom(
                lane, self.trellis_bits
            )
            trellis_lut_addr = Int64(
                smem_base + Int32(self.trellis_lut_offset)
            )
            if cutlass.const_expr(self.trellis_direct_lut):
                trellis_lut_addr = trellis_lut.iterator.toint()

        input_k64_tiles = input_k128_tiles * Int32(2)
        for preload_slice in cutlass.range_constexpr(self.stages - 1):
            k64_slice = Int32(preload_slice)
            if k64_slice < input_k64_tiles:
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
                    k64_slice,
                    input_k128_tiles,
                    intermediate_tiles,
                    packed_w13_tiles,
                )
            cute.arch.cp_async_commit_group()

        # Keep each MMA's four accumulator registers as an independent
        # fragment. A mutable M-fragment tuple avoids packing and unpacking
        # the complete live accumulator set around loop-carried values even
        # though every MMA consumes exactly four adjacent values.
        n_fragments = self.tile_n // 32
        gate_acc = tuple(
            tuple(
                cute.make_rmem_tensor((4,), cutlass.Float32)
                for _nt in range(n_fragments)
            )
            for _blk in range(self.tile_m // 16)
        )
        up_acc = tuple(
            tuple(
                cute.make_rmem_tensor((4,), cutlass.Float32)
                for _nt in range(n_fragments)
            )
            for _blk in range(self.tile_m // 16)
        )
        for blk in cutlass.range_constexpr(self.tile_m // 16):
            for nt in cutlass.range_constexpr(n_fragments):
                gate_acc[blk][nt].fill(0.0)
                up_acc[blk][nt].fill(0.0)
        k64_slice = Int32(0)
        while k64_slice < input_k64_tiles:
            stage = k64_slice & Int32(self.stages - 1)
            stage_base = smem_base + stage * Int32(self.stage_bytes)
            a_base = stage_base + Int32(self.a_offset)
            sfa_base = stage_base + Int32(self.sfa_offset)
            gate_b_base = stage_base + Int32(self.gate_b_offset)
            up_b_base = stage_base + Int32(self.up_b_offset)
            gate_sfb_base = stage_base + Int32(self.gate_sfb_offset)
            up_sfb_base = stage_base + Int32(self.up_sfb_offset)

            next_slice = k64_slice + Int32(self.stages - 1)
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
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.sync_threads()

            scale_shift = Uint32(k64_slice & Int32(1)) * Uint32(16)
            asc = cute.make_rmem_tensor((self.tile_m // 16,), Uint32)
            for blk in cutlass.range_constexpr(self.tile_m // 16):
                sf_row = Int32(blk * 16) + q + ((lane & Int32(1)) << Int32(3))
                asc[blk] = ld_shared_u32(sfa_base + (sf_row << Int32(2))) >> scale_shift

            for kb in cutlass.range_constexpr(2):
                u_phys = (Int32(kb * 2) + (c >> Int32(1))) ^ q
                a_frag = cute.make_rmem_tensor((self.tile_m // 16, 4), Uint32)
                for blk in cutlass.range_constexpr(self.tile_m // 16):
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

                gate_b0 = cute.make_rmem_tensor((n_fragments,), Uint32)
                gate_b1 = cute.make_rmem_tensor((n_fragments,), Uint32)
                up_b0 = cute.make_rmem_tensor((n_fragments,), Uint32)
                up_b1 = cute.make_rmem_tensor((n_fragments,), Uint32)
                if cutlass.const_expr(self.w4a8_trellis):
                    for th in cutlass.range_constexpr(2):
                        tr_n16 = warp_idx * Int32(2) + Int32(th)
                        tr_b0 = (Int32(kb * 16) + tr_n16) * Int32(
                            8 * self.trellis_bits
                        )
                        g_lo0, g_lo1, g_hi0, g_hi1 = (
                            _w4a8_trellis_pair_words_both(
                                gate_b_base, lane, tr_b0,
                                tr_b0 + Int32(64 * self.trellis_bits),
                                tr_ia, tr_ib, tr_s2, self.trellis_bits,
                                trellis_lut_addr, not self.trellis_direct_lut,
                                self.trellis_direct_lut,
                            )
                        )
                        gate_b0[th * 2] = g_lo0
                        gate_b1[th * 2] = g_lo1
                        gate_b0[th * 2 + 1] = g_hi0
                        gate_b1[th * 2 + 1] = g_hi1
                        u_lo0, u_lo1, u_hi0, u_hi1 = (
                            _w4a8_trellis_pair_words_both(
                                up_b_base, lane, tr_b0,
                                tr_b0 + Int32(64 * self.trellis_bits),
                                tr_ia, tr_ib, tr_s2, self.trellis_bits,
                                trellis_lut_addr, not self.trellis_direct_lut,
                                self.trellis_direct_lut,
                            )
                        )
                        up_b0[th * 2] = u_lo0
                        up_b1[th * 2] = u_lo1
                        up_b0[th * 2 + 1] = u_hi0
                        up_b1[th * 2 + 1] = u_hi1
                else:
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
                    for nt in cutlass.range_constexpr(n_fragments):
                        gate_b0[nt], gate_b1[nt] = e2m1x8_to_qmma_e2m1x8(
                            gate_words[nt]
                        )
                        up_b0[nt], up_b1[nt] = e2m1x8_to_qmma_e2m1x8(
                            up_words[nt]
                        )
                for nt in cutlass.range_constexpr(n_fragments):
                    n8 = warp_idx * Int32(n_fragments) + Int32(nt)
                    gb0 = gate_b0[nt]
                    gb1 = gate_b1[nt]
                    ub0 = up_b0[nt]
                    ub1 = up_b1[nt]
                    gate_sfb = Uint32(0x7F7F7F7F)
                    up_sfb = Uint32(0x7F7F7F7F)
                    if cutlass.const_expr(not self.w4a8_trellis):
                        gate_sfb = (
                            ld_shared_u32(
                                gate_sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                            )
                            >> scale_shift
                        )
                        up_sfb = (
                            ld_shared_u32(
                                up_sfb_base + ((n8 * Int32(8) + q) << Int32(2))
                            )
                            >> scale_shift
                        )
                    for blk in cutlass.range_constexpr(self.tile_m // 16):
                        gate_fragment = gate_acc[blk][nt]
                        if cutlass.const_expr(self.w4a8_trellis):
                            g0, g1, g2, g3 = mxfp8_mma_m16n8k32_f32_e4m3(
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
                        else:
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
                        if cutlass.const_expr(self.w4a8_trellis):
                            u0, u1, u2, u3 = mxfp8_mma_m16n8k32_f32_e4m3(
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
                        else:
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

        # The pipeline region aliases the activation staging tile below. The
        # final (possibly empty) committed group must be fully retired before
        # those shared addresses are repurposed by ordinary stores.
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()

        scale_expert = expert_idx
        if cutlass.const_expr(input_global_scale.shape[0] == 1):
            scale_expert = Int32(0)
        alpha_value = alpha[expert_idx].to(cutlass.Float32) * input_global_scale[
            scale_expert
        ].to(cutlass.Float32)
        epilogue_base = smem_base
        col_base = (
            warp_idx * Int32((self.tile_n // 32) * 8) + (c << Int32(1))
        )
        if cutlass.const_expr(self.w4a8_trellis):
            # Trellis activation boundary: raw alpha-scaled projections into
            # two shared tiles, per-row H128 + per-expert rotations + gated
            # activation + down scale + H128, then the shared requantization
            # below reads the rotated result in place.
            up_tile_base = smem_base + Int32(self.tile_m * self.tile_n * 2)
            for nt in cutlass.range_constexpr(n_fragments):
                col = col_base + Int32(nt * 8)
                for blk in cutlass.range_constexpr(self.tile_m // 16):
                    gate_fragment = gate_acc[blk][nt]
                    up_fragment = up_acc[blk][nt]
                    row_lo = Int32(blk * 16) + q
                    row_hi = row_lo + Int32(8)
                    st_shared_u32(
                        epilogue_base
                        + (row_lo * Int32(self.tile_n) + col) * Int32(2),
                        pack_f32x2_to_bfloat2(
                            alpha_value * gate_fragment[0],
                            alpha_value * gate_fragment[1],
                        ),
                    )
                    st_shared_u32(
                        epilogue_base
                        + (row_hi * Int32(self.tile_n) + col) * Int32(2),
                        pack_f32x2_to_bfloat2(
                            alpha_value * gate_fragment[2],
                            alpha_value * gate_fragment[3],
                        ),
                    )
                    st_shared_u32(
                        up_tile_base
                        + (row_lo * Int32(self.tile_n) + col) * Int32(2),
                        pack_f32x2_to_bfloat2(
                            alpha_value * up_fragment[0],
                            alpha_value * up_fragment[1],
                        ),
                    )
                    st_shared_u32(
                        up_tile_base
                        + (row_hi * Int32(self.tile_n) + col) * Int32(2),
                        pack_f32x2_to_bfloat2(
                            alpha_value * up_fragment[2],
                            alpha_value * up_fragment[3],
                        ),
                    )
            cute.arch.sync_threads()
            tr_isz = intermediate_tiles * Int32(128)
            tr_col = output_tile * Int32(128) + lane * Int32(4)
            tr_seg = Int32(6) if cutlass.const_expr(self.trellis_intermediate_hadamard) else Int32(3)
            tr_rot = expert_idx * (tr_seg * tr_isz) + tr_col
            rg0 = cutlass.Float32(trellis_rotations[tr_rot])
            rg1 = cutlass.Float32(trellis_rotations[tr_rot + Int32(1)])
            rg2 = cutlass.Float32(trellis_rotations[tr_rot + Int32(2)])
            rg3 = cutlass.Float32(trellis_rotations[tr_rot + Int32(3)])
            ru0 = cutlass.Float32(trellis_rotations[tr_rot + tr_isz])
            ru1 = cutlass.Float32(trellis_rotations[tr_rot + tr_isz + Int32(1)])
            ru2 = cutlass.Float32(trellis_rotations[tr_rot + tr_isz + Int32(2)])
            ru3 = cutlass.Float32(trellis_rotations[tr_rot + tr_isz + Int32(3)])
            sd_base = tr_rot + tr_isz + tr_isz
            sd0 = cutlass.Float32(trellis_rotations[sd_base])
            sd1 = cutlass.Float32(trellis_rotations[sd_base + Int32(1)])
            sd2 = cutlass.Float32(trellis_rotations[sd_base + Int32(2)])
            sd3 = cutlass.Float32(trellis_rotations[sd_base + Int32(3)])
            unit_alpha = cutlass.Float32(1.0)
            tr_chunk = lane >> Int32(3)
            tr_chunk_lane = lane & Int32(7)
            tr_slot = tr_chunk & Int32(1)
            for row_it in cutlass.range_constexpr(self.tile_m // self.num_warps):
                seam_row = warp_idx * Int32(self.tile_m // self.num_warps) + Int32(
                    row_it
                )
                row_addr = (
                    epilogue_base
                    + (seam_row * Int32(self.tile_n) + lane * Int32(4)) * Int32(2)
                )
                up_addr = (
                    up_tile_base
                    + (seam_row * Int32(self.tile_n) + lane * Int32(4)) * Int32(2)
                )
                if cutlass.const_expr(self.trellis_intermediate_hadamard):
                    aa0 = cutlass.Float32(0.0)
                    aa1 = cutlass.Float32(0.0)
                    bb0 = cutlass.Float32(0.0)
                    bb1 = cutlass.Float32(0.0)
                    for _pb in cutlass.range_constexpr(2):
                        tr_ci = (
                            Int32(_pb * 64)
                            + ((tr_chunk >> Int32(1)) << Int32(5))
                            + tr_chunk_lane * Int32(4)
                        )
                        src_base = (
                            epilogue_base
                            + (seam_row * Int32(self.tile_n) + tr_ci) * Int32(2)
                        )
                        if tr_slot != Int32(0):
                            src_base = (
                                up_tile_base
                                + (seam_row * Int32(self.tile_n) + tr_ci)
                                * Int32(2)
                            )
                        p0 = ld_shared_bf16_to_f32(src_base)
                        p1 = ld_shared_bf16_to_f32(src_base + Int32(2))
                        p2 = ld_shared_bf16_to_f32(src_base + Int32(4))
                        p3 = ld_shared_bf16_to_f32(src_base + Int32(6))
                        h0, h1, h2, h3 = _w4a8_had128_quad(p0, p1, p2, p3, lane)
                        tr_coord = output_tile * Int32(128) + tr_ci
                        tr_scale = (
                            expert_idx * (Int32(6) * tr_isz)
                            + tr_slot * tr_isz
                            + tr_coord
                        )
                        h0 = h0 * cutlass.Float32(trellis_rotations[tr_scale])
                        h1 = h1 * cutlass.Float32(
                            trellis_rotations[tr_scale + Int32(1)]
                        )
                        h2 = h2 * cutlass.Float32(
                            trellis_rotations[tr_scale + Int32(2)]
                        )
                        h3 = h3 * cutlass.Float32(
                            trellis_rotations[tr_scale + Int32(3)]
                        )
                        h0, h1, h2, h3 = _w4a8_had128_quad(h0, h1, h2, h3, lane)
                        tr_sign = (
                            expert_idx * (Int32(6) * tr_isz)
                            + Int32(3) * tr_isz
                            + output_tile * Int32(256)
                            + Int32(_pb * 128)
                            + lane * Int32(4)
                        )
                        h0 = h0 * cutlass.Float32(trellis_rotations[tr_sign])
                        h1 = h1 * cutlass.Float32(
                            trellis_rotations[tr_sign + Int32(1)]
                        )
                        h2 = h2 * cutlass.Float32(
                            trellis_rotations[tr_sign + Int32(2)]
                        )
                        h3 = h3 * cutlass.Float32(
                            trellis_rotations[tr_sign + Int32(3)]
                        )
                        s0 = self._activated_value(h0, h1, unit_alpha)
                        s1 = self._activated_value(h2, h3, unit_alpha)
                        if cutlass.const_expr(_pb == 0):
                            aa0 = s0
                            aa1 = s1
                        else:
                            bb0 = s0
                            bb1 = s1
                    src0 = (lane & Int32(15)) << Int32(1)
                    src1 = src0 + Int32(1)
                    av0 = cute.arch.shuffle_sync(aa0, src0)
                    av1 = cute.arch.shuffle_sync(aa1, src0)
                    av2 = cute.arch.shuffle_sync(aa0, src1)
                    av3 = cute.arch.shuffle_sync(aa1, src1)
                    bv0 = cute.arch.shuffle_sync(bb0, src0)
                    bv1 = cute.arch.shuffle_sync(bb1, src0)
                    bv2 = cute.arch.shuffle_sync(bb0, src1)
                    bv3 = cute.arch.shuffle_sync(bb1, src1)
                    o0 = av0
                    o1 = av1
                    o2 = av2
                    o3 = av3
                    if lane >= Int32(16):
                        o0 = bv0
                        o1 = bv1
                        o2 = bv2
                        o3 = bv3
                    tr_col = output_tile * Int32(128) + lane * Int32(4)
                    tr_ub = (
                        expert_idx * (Int32(6) * tr_isz)
                        + Int32(5) * tr_isz
                        + tr_col
                    )
                    o0 = o0 * cutlass.Float32(trellis_rotations[tr_ub])
                    o1 = o1 * cutlass.Float32(trellis_rotations[tr_ub + Int32(1)])
                    o2 = o2 * cutlass.Float32(trellis_rotations[tr_ub + Int32(2)])
                    o3 = o3 * cutlass.Float32(trellis_rotations[tr_ub + Int32(3)])
                    o0, o1, o2, o3 = _w4a8_had128_quad(o0, o1, o2, o3, lane)
                    tr_dn = (
                        expert_idx * (Int32(6) * tr_isz)
                        + Int32(2) * tr_isz
                        + tr_col
                    )
                    o0 = o0 * cutlass.Float32(trellis_rotations[tr_dn])
                    o1 = o1 * cutlass.Float32(trellis_rotations[tr_dn + Int32(1)])
                    o2 = o2 * cutlass.Float32(trellis_rotations[tr_dn + Int32(2)])
                    o3 = o3 * cutlass.Float32(trellis_rotations[tr_dn + Int32(3)])
                    o0, o1, o2, o3 = _w4a8_had128_quad(o0, o1, o2, o3, lane)
                    st_shared_u32(row_addr, pack_f32x2_to_bfloat2(o0, o1))
                    st_shared_u32(
                        row_addr + Int32(4), pack_f32x2_to_bfloat2(o2, o3)
                    )
                else:
                    g0 = ld_shared_bf16_to_f32(row_addr)
                    g1 = ld_shared_bf16_to_f32(row_addr + Int32(2))
                    g2 = ld_shared_bf16_to_f32(row_addr + Int32(4))
                    g3 = ld_shared_bf16_to_f32(row_addr + Int32(6))
                    u0 = ld_shared_bf16_to_f32(up_addr)
                    u1 = ld_shared_bf16_to_f32(up_addr + Int32(2))
                    u2 = ld_shared_bf16_to_f32(up_addr + Int32(4))
                    u3 = ld_shared_bf16_to_f32(up_addr + Int32(6))
                    gh0, gh1, gh2, gh3 = _w4a8_had128_quad(g0, g1, g2, g3, lane)
                    uh0, uh1, uh2, uh3 = _w4a8_had128_quad(u0, u1, u2, u3, lane)
                    a0 = self._activated_value(gh0 * rg0, uh0 * ru0, unit_alpha) * sd0
                    a1 = self._activated_value(gh1 * rg1, uh1 * ru1, unit_alpha) * sd1
                    a2 = self._activated_value(gh2 * rg2, uh2 * ru2, unit_alpha) * sd2
                    a3 = self._activated_value(gh3 * rg3, uh3 * ru3, unit_alpha) * sd3
                    o0, o1, o2, o3 = _w4a8_had128_quad(a0, a1, a2, a3, lane)
                    st_shared_u32(row_addr, pack_f32x2_to_bfloat2(o0, o1))
                    st_shared_u32(row_addr + Int32(4), pack_f32x2_to_bfloat2(o2, o3))
        else:
            for nt in cutlass.range_constexpr(n_fragments):
                col = col_base + Int32(nt * 8)
                for blk in cutlass.range_constexpr(self.tile_m // 16):
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

        physical_row_base = Int64(source_m_tile) * Int64(self.source_tile_m) + Int64(m_half * self.tile_m)
        words_per_row = intermediate_tiles * Int32(32)
        if tid < valid_rows:
            scale_word = Uint32(0)
            for block in cutlass.range_constexpr(self.tile_n // 32):
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
                if cutlass.const_expr(self.w4a8_trellis):
                    payload, scale_byte = quantize_block_fp8_mx(
                        _w4a8_trellis_permute_k32(values), block_max
                    )
                else:
                    payload, scale_byte = quantize_block_fp8_mx(values, block_max)
                dst_word = (
                    (physical_row_base + tid) * words_per_row
                    + output_tile * Int32(self.tile_n // 4)
                    + Int32(block * 8)
                )
                for word in cutlass.range_constexpr(8):
                    intermediate_u32[dst_word + Int32(word)] = payload[word]
                scale_word = scale_word | (
                    (scale_byte & Uint32(0xFF)) << Uint32(block * 8)
                )

            sf_base = Int64(rows_capacity) * Int64(words_per_row)
            intermediate_u32[
                sf_base + Int64(output_tile) * Int64(rows_capacity) + physical_row_base + tid
            ] = scale_word

        # Only row-owning threads perform the row-wise quantize/store. The
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
        token_weights: cute.Tensor,
        task_expert: cute.Tensor,
        task_valid_rows: cute.Tensor,
        expert_tile_base: cute.Tensor,
        alpha: cute.Tensor,
        input_global_scale: cute.Tensor,
        trellis_lut: cute.Tensor,
        trellis_rotations: cute.Tensor,
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
        output_tiles = intermediate_tiles
        num_experts = Int32(expert_tile_base.shape[0] - 1)
        source_m_tiles = expert_tile_base[num_experts].to(Int32)
        task_tail = source_m_tiles * Int32(self.source_halves) * output_tiles
        task_slot = Int32(bidz)
        while task_slot < task_tail:
            output_tile = task_slot % output_tiles
            source_half = task_slot // output_tiles
            m_half = source_half % Int32(self.source_halves)
            source_m_tile = source_half // Int32(self.source_halves)
            phase1_meta = source_m_tile * intermediate_tiles
            expert_idx = task_expert[phase1_meta].to(Int32)
            valid_rows = task_valid_rows[phase1_meta].to(Int32) - m_half * Int32(self.tile_m)
            valid_rows = cutlass.min(valid_rows, Int32(self.tile_m))
            if valid_rows > Int32(0):
                self._run_task(
                    packed_a_u32,
                    scale_storage,
                    w13_rp,
                    w13_sfb_rp,
                    intermediate_u32,
                    token_map,
                    token_weights,
                    alpha,
                    input_global_scale,
                    trellis_lut,
                    trellis_rotations,
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
