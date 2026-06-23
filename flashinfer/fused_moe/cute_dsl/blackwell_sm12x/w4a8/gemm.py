"""W4A8 grouped MoE GEMM: MXFP8 (E4M3 + UE8M0/K32) activations x MXFP4 weights.

One launch computes ``C[rows_padded, size_n] (bf16) = A_q @ B[expert(mb)]^T``
where each 16-row m-block is assigned one expert. The mainloop is the
probe-pinned QMMA pipeline of tests/test_w4a8_marlin_probe.py generalized to
128 threads (4 n-warps x 8 n8 tiles = tile_n 256) and the register-level
contract of tests/test_w4a8_fragment_probe.py:

- A: plain row-major e4m3 staged as 16 x 128B rows per k-tile; fragments
  a0/a2 = u32 @ (q*128 + kb*32 + (lane&3)*8)(+4), a1/a3 the same at row q+8.
- A scales: one u32 per (row, k-tile) ([rows, K/32] u8 viewed as u32); lane l
  supplies row (l>>2) + 8*(l&1), byte kb extracted per k32 step.
- B: host-repacked ``B_rp[E][n_tile][k_tile][n8][k32][lane] u32`` so each
  lane's per-(n8, k32) packed-FP4 word read is a coalesced u32; expanded
  in-register via e2m1x8_to_e4m3x8 (the weight scale byte rides the SFB
  operand -- no bf16 multiply).
- B scales: ``SFB[E][n_tile][k_tile][n8][col8] u32`` (4 e8m0 bytes, k32
  0..3); lane l supplies col l>>2, byte kb extracted per k32 step.
- Output fragment map (pinned by both probes and identical to the w4a16
  epilogue contract): col = warp*64 + nt*8 + 2c (+1), rows q / q+8 of the
  m-block, per n8 tile nt.

V1 SIMPLIFICATIONS (correctness milestone -- hardening backlog):
- FLAT GRID: one CTA per (m-block GROUP, n_tile), grid.x =
  ceil(num_m_blocks/_BLOCKS_PER_CTA) * n_tiles (use ``grid_x()``). No
  persistent slice loop, NO split-K (every CTA sweeps the full K for its
  tile), so no c_tmp/locks machinery.
- M-BLOCK GROUPS: when every routed block of the group shares one expert
  (the common case under expert-sorted routing) a single staged B tile
  feeds all blocks' QMMAs, amortizing B smem loads, FP4 expansion, and B
  L2 traffic over _BLOCKS_PER_CTA x the FLOPs. Mixed-expert groups fall
  back to one sweep per routed block (correct for arbitrary per-block
  expert ids, just slower at expert boundaries).
- STAGING: cp.async double-buffered (2 stages x 19,520B). Per k-tile the CTA
  issues stage kt+1 (A rows + contiguous B/Bsf as 16B cp.async.cg, strided
  Asf u32s as 4B cp.async.ca), commits, waits all-but-one group, syncs, and
  consumes stage kt; a tail sync fences the WAR on the buffer the next
  iteration overwrites.
- ROUTING: two compile-time addressing modes (``gather_a`` const_expr flag):
  * DENSE (default): A rows are assumed already gathered/sorted so m-block mb
    covers A rows [mb*16, mb*16+16). Routing input is a per-m-block expert-id
    tensor (i32, <0 skips the block like w4a16's padding blocks); stores are
    masked by row < active_m.
  * GATHER (``gather_a=True``): A row r of m-block mb is read through the
    w4a16 route packing -- ``idx = packed_route_indices[mb*16 + r]`` encodes
    token*topk + j, so the source A row is ``idx // topk`` (``topk`` is a
    compile-time constant). Slots with ``idx >= total_routes`` are padding
    (route_pack fills them with live_numel): their A reads are clamped to row
    0 and their C rows are STORED AS ZEROS (so a packed intermediate consumed
    by the activation stage is deterministic garbage-free). C rows stay at
    the PACKED positions; ``active_m`` should be rows_padded. CONTRACT: the
    route-index buffer must cover num_m_blocks*16 slots; slots beyond
    route_pack's written capacity must be prefilled (once) with a value
    >= total_routes -- staging touches every present block's slots, and a
    stale "valid" index would gather an out-of-bounds A row.
  Per-thread A/Asf source offsets are precomputed once before the k-loop in
  both modes, so the staging hot loop is identical for dense and gather.
- EPILOGUE: direct st.global of bf16x2 pairs through a u32 view of C (no
  smem-staged coalescing drain). No topk-weight multiply, no scatter,
  alphas == 1 (w4a8_mx semantics).
- MASKING: stores are predicated on row < active_m. A_q / A_sf must be
  ALLOCATED padded to rows_padded = num_m_blocks*16 rows (padded-row loads
  are in-bounds garbage that only feeds masked-out outputs).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack

from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from flashinfer.cute_dsl.fp4_common import (
    cp_async4_shared_global,
    cp_async_u32_shared_global,
    get_ptr_as_int64,
    ld_shared_u32,
    ld_shared_v2_u32,
    ld_shared_v4_u32,
    pack_f32x2_to_bfloat2,
    shared_ptr_to_u32,
)

_TILE_M = 16  # rows per m-block
_BLOCKS_PER_CTA = 3  # m-blocks computed against one staged B tile
_TILE_N = 256  # 4 warps x 8 n8 tiles x 8 cols
_TILE_K = 128  # 4 k32 QMMA steps per k-tile
_CTA_THREADS = 128  # 4 warps, all n-warps (no k-split)
_N8_PER_WARP = 8
_B_TILE_WORDS = 32 * 4 * 32  # [k32][n8c][lane][n8i] u32 per (n_tile, k_tile)
_BSF_TILE_WORDS = 32 * 8  # [n8][col8] u32 per (n_tile, k_tile)
_A_BLOCK_WORDS = _TILE_M * (_TILE_K // 4)  # 512 per m-block
_A_TILE_WORDS = _BLOCKS_PER_CTA * _A_BLOCK_WORDS

# cp.async double-buffer stage layout (u32 word offsets within a stage; every
# sub-region is 16B aligned).
_ASF_WORD_OFF = _A_TILE_WORDS
_B_WORD_OFF = _ASF_WORD_OFF + _BLOCKS_PER_CTA * _TILE_M
_BSF_WORD_OFF = _B_WORD_OFF + _B_TILE_WORDS
_STAGE_WORDS = _BSF_WORD_OFF + _BSF_TILE_WORDS
_STAGE_BYTES = _STAGE_WORDS * 4
_STAGES = 2
assert _B_WORD_OFF % 4 == 0 and _STAGE_WORDS % 4 == 0  # 16B sub-alignment


@dsl_user_op
def _nib8_to_e2m1_bytes(
    packed_u32: Uint32, *, loc=None, ip=None
) -> tuple[Uint32, Uint32]:
    """Spread 8 packed FP4 nibbles into 8 bytes (nibble i -> byte i bits 5:2).

    The QMMA ``kind::mxf8f6f4`` consumes ``.e2m1`` operands as one element
    per 8-bit container with the value in bits 5:2 (the tcgen05 f8f6f4
    container convention), so this replaces the 12-op e4m3 relabeling with
    a 6-op spread.
    """
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 s, t;
            shl.b32 s, $2, 2;
            shr.u32 t, $2, 2;
            prmt.b32 $0, s, t, 0x5140;
            prmt.b32 $1, s, t, 0x7362;
            and.b32 $0, $0, 0x3C3C3C3C;
            and.b32 $1, $1, 0x3C3C3C3C;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def _qmma_m16n8k32_f32_e4m3_e2m1(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> tuple[Float32, Float32, Float32, Float32]:
    """SM120 block-scaled QMMA, A=e4m3 bytes x B=e2m1-in-byte containers.

    Identical numerics to the e4m3/e4m3 wrapper (e2m1 values are exact in
    both encodings); byte/thread scale selectors are baked as immediates.
    """
    asm = f"""
        mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0
        {{$0, $1, $2, $3}},
        {{$4, $5, $6, $7}},
        {{$8, $9}},
        {{$0, $1, $2, $3}},
        {{$10}},
        {{{int(bid_a)}, {int(tid_a)}}},
        {{$11}},
        {{{int(bid_b)}, {int(tid_b)}}};
        """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Uint32(sfa).ir_value(loc=loc, ip=ip),
            Uint32(sfb).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,r,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


def repack_w4a8_weights(
    w_fp4: torch.Tensor, w_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repack plain packed-FP4 weights + e8m0 scales for the W4A8 GEMM.

    ``w_fp4 [E, N, K/2] u8`` (k-major nibbles) + ``w_sf [E, N, K/32] u8`` ->

    - ``B_rp [E, n_tiles, k_tiles, 4, 8, 32, 4] i32`` = [k32][n8 chunk][lane]
      [n8-in-chunk]: word (lane; n8 = chunk*4 + i, k32) is the 8 packed
      nibbles of B row ``n_tile*256 + n8*8 + (lane>>2)`` covering original k
      in ``[k_tile*128 + k32*32 + (lane&3)*8, +8)``. Grouping the 4 chunk
      words per lane contiguously lets the kernel fetch them with one
      conflict-free 16B smem load.
    - ``SFB [E, n_tiles, k_tiles, 32, 8] i32``: the 4 scale bytes (k32 0..3)
      of col ``n8*8 + col8``, byte kb = k-group ``k_tile*4 + kb``.
    """
    if w_fp4.dim() != 3 or w_sf.dim() != 3:
        raise ValueError("w_fp4 must be [E, N, K/2] and w_sf [E, N, K/32]")
    e, n, k_half = w_fp4.shape
    k = k_half * 2
    if w_sf.shape != (e, n, k // 32):
        raise ValueError(f"w_sf shape {tuple(w_sf.shape)} != {(e, n, k // 32)}")
    if n % _TILE_N != 0:
        raise ValueError(f"N must be divisible by {_TILE_N}, got {n}")
    if k % _TILE_K != 0:
        raise ValueError(f"K must be divisible by {_TILE_K}, got {k}")
    n_tiles = n // _TILE_N
    k_tiles = k // _TILE_K

    b_u32 = w_fp4.contiguous().view(torch.int32).reshape(e, n, k // 8)
    # rows: (n_tile, n8 chunk, n8-in-chunk, row-in-n8); cols: (k_tile, k32,
    # 8k-group).
    b = b_u32.reshape(e, n_tiles, 8, 4, 8, k_tiles, 4, 4)
    b_rp = (
        # [E, nt, kt, k32, n8c, r8, cgrp, n8i]; lane = r8*4 + cgrp.
        b.permute(0, 1, 5, 6, 2, 4, 7, 3)
        .reshape(e, n_tiles, k_tiles, 4, 8, 32, 4)
        .contiguous()
    )

    sf = w_sf.contiguous().reshape(e, n_tiles, 32, 8, k_tiles, 4)
    sfb = (
        sf.permute(0, 1, 4, 2, 3, 5)  # [E, nt, kt, n8, col8, kb-byte]
        .contiguous()
        .view(torch.int32)
        .reshape(e, n_tiles, k_tiles, 32, 8)
    )
    return b_rp, sfb


class W4A8GemmKernel:
    """V1 flat-grid grouped QMMA GEMM (see module docstring for the contract).

    Constexpr geometry lives on ``self``; the call takes flat u32 views:
    A_q [rows_padded, K] e4m3 -> u32, A_sf [rows_padded, K/32] u8 -> u32,
    B_rp / SFB from :func:`repack_w4a8_weights`, C bf16 [rows_padded, size_n]
    -> u32 (bf16x2 words), block_expert_ids i32 [num_m_blocks].
    """

    def __init__(
        self,
        *,
        size_n: int,
        size_k: int,
        num_experts: int,
        gather_a: bool = False,
        topk: int = 1,
        experts_per_group: bool = False,
    ):
        if size_n % _TILE_N != 0:
            raise ValueError(f"size_n must be divisible by {_TILE_N}, got {size_n}")
        if size_k % _TILE_K != 0:
            raise ValueError(f"size_k must be divisible by {_TILE_K}, got {size_k}")
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        self.size_n = int(size_n)
        self.size_k = int(size_k)
        self.num_experts = int(num_experts)
        self.gather_a = bool(gather_a)
        self.topk = int(topk)
        # experts_per_group: block_expert_ids carries ONE id per
        # _BLOCKS_PER_CTA-block GROUP (route packing padded expert runs to
        # group multiples), so every group is expert-uniform and the
        # mixed-expert slow path is compiled out entirely.
        self.experts_per_group = bool(experts_per_group)
        self.n_tiles = self.size_n // _TILE_N
        self.k_tiles = self.size_k // _TILE_K
        self.cta_threads = _CTA_THREADS
        self.blocks_per_cta = _BLOCKS_PER_CTA

    def grid_x(self, num_m_blocks: int) -> int:
        """grid.x for a launch covering ``num_m_blocks`` 16-row m-blocks."""
        groups = (num_m_blocks + _BLOCKS_PER_CTA - 1) // _BLOCKS_PER_CTA
        return groups * self.n_tiles

    @property
    def __cache_key__(self) -> tuple[object, ...]:
        return (
            self.size_n,
            self.size_k,
            self.num_experts,
            self.gather_a,
            self.topk,
            self.experts_per_group,
        )

    @cute.jit
    def __call__(
        self,
        a_q_flat: cute.Tensor,  # u32 view of [rows_padded, K] e4m3
        a_sf_flat: cute.Tensor,  # u32 view of [rows_padded, K/32] u8
        b_rp_flat: cute.Tensor,  # u32 [E*n_tiles*k_tiles*4096]
        b_sf_flat: cute.Tensor,  # u32 [E*n_tiles*k_tiles*256]
        c_u32_flat: cute.Tensor,  # u32 view of [rows_padded, size_n] bf16
        block_expert_ids: cute.Tensor,  # i32 [num_m_blocks]
        packed_route_indices: cute.Tensor,  # i32 [num_m_blocks*16] (gather_a)
        num_m_blocks: cutlass.Int32,  # 16-row m-blocks (= rows_padded / 16)
        active_m: cutlass.Int32,  # valid A rows (stores masked beyond)
        total_routes: cutlass.Int32,  # valid route slots = m*topk (gather_a)
        grid_x: cutlass.Int32,  # ceil(num_m_blocks / 3) * n_tiles
        stream: cuda.CUstream,
    ):
        self.kernel(
            a_q_flat,
            a_sf_flat,
            b_rp_flat,
            b_sf_flat,
            c_u32_flat,
            block_expert_ids,
            packed_route_indices,
            num_m_blocks,
            active_m,
            total_routes,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        a_q_flat: cute.Tensor,
        a_sf_flat: cute.Tensor,
        b_rp_flat: cute.Tensor,
        b_sf_flat: cute.Tensor,
        c_u32_flat: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_indices: cute.Tensor,
        num_m_blocks: cutlass.Int32,
        active_m: cutlass.Int32,
        total_routes: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        tid = Int32(tidx)
        cta = Int32(bidx)

        smem = cutlass.utils.SmemAllocator()

        @cute.struct
        class Storage:
            sData: cute.struct.Align[
                cute.struct.MemRange[cutlass.Uint32, _STAGES * _STAGE_WORDS], 16
            ]

        storage = smem.allocate(Storage)
        # Hoisted before any dynamic control flow (flattener trap).
        s_base = shared_ptr_to_u32(storage.sData.data_ptr())

        group = cta // Int32(self.n_tiles)
        output_n_tile = cta - group * Int32(self.n_tiles)
        mb0 = group * Int32(_BLOCKS_PER_CTA)
        n_present = Int32(0)
        for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
            if (mb0 + Int32(blk)) < num_m_blocks:
                n_present = n_present + Int32(1)

        if cutlass.const_expr(self.experts_per_group):
            # One expert id per group (route packing pads expert runs to
            # group multiples): every group is uniform, no slow path.
            e_grp = block_expert_ids[group].to(Int32)
            if e_grp >= Int32(0):
                en_mask = Int32(0)
                for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                    if Int32(blk) < n_present:
                        en_mask = en_mask | Int32(1 << blk)
                self._run_tile(
                    a_q_flat,
                    a_sf_flat,
                    b_rp_flat,
                    b_sf_flat,
                    c_u32_flat,
                    packed_route_indices,
                    s_base,
                    tid,
                    mb0,
                    e_grp,
                    output_n_tile,
                    active_m,
                    total_routes,
                    en_mask,
                    n_present,
                )
        else:
            # Absent blocks (beyond num_m_blocks; always a suffix) get id -2
            # so they can never alias a real expert or the -1 padding id;
            # their A rows must not be staged (they would be out of bounds).
            eids = cute.make_rmem_tensor((_BLOCKS_PER_CTA,), cutlass.Int32)
            for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                e = Int32(-2)
                if (mb0 + Int32(blk)) < num_m_blocks:
                    e = block_expert_ids[mb0 + Int32(blk)].to(Int32)
                eids[blk] = e
            # Leader = first routed block's expert; the fast path applies
            # when every routed block matches it (one B staging feeds them
            # all).
            lead = Int32(-1)
            mismatch = Int32(0)
            for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                e = Int32(eids[blk])
                if e >= Int32(0):
                    if lead < Int32(0):
                        lead = e
                    if e != lead:
                        mismatch = Int32(1)
            if mismatch == Int32(0):
                if lead >= Int32(0):
                    en_mask = Int32(0)
                    for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                        if Int32(eids[blk]) == lead:
                            en_mask = en_mask | Int32(1 << blk)
                    self._run_tile(
                        a_q_flat,
                        a_sf_flat,
                        b_rp_flat,
                        b_sf_flat,
                        c_u32_flat,
                        packed_route_indices,
                        s_base,
                        tid,
                        mb0,
                        lead,
                        output_n_tile,
                        active_m,
                        total_routes,
                        en_mask,
                        n_present,
                    )
            else:
                # Slow path (expert boundary): one sweep per routed block,
                # storing only that block's rows.
                for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                    e = Int32(eids[blk])
                    if e >= Int32(0):
                        self._run_tile(
                            a_q_flat,
                            a_sf_flat,
                            b_rp_flat,
                            b_sf_flat,
                            c_u32_flat,
                            packed_route_indices,
                            s_base,
                            tid,
                            mb0,
                            e,
                            output_n_tile,
                            active_m,
                            total_routes,
                            Int32(1 << blk),
                            n_present,
                        )

    @cute.jit
    def _issue_stage(
        self,
        a_q_flat: cute.Tensor,
        a_sf_flat: cute.Tensor,
        b_rp_flat: cute.Tensor,
        b_sf_flat: cute.Tensor,
        stage_base: Int32,  # smem byte address of the target stage
        tid: Int32,
        kt: Int32,
        a_src_off: cute.Tensor,  # rmem [_BLOCKS_PER_CTA]: thread's A src base
        asf_src_off: Int32,  # thread's Asf source row word base
        b_base: Int32,
        bsf_base: Int32,
        n_present: Int32,
    ):
        """cp.async the (A, Asf, B, Bsf) k-tile ``kt`` into ``stage_base``.

        No commit/wait/sync here -- the caller owns group discipline.
        Only the first ``n_present`` m-blocks' A/Asf rows are staged (rows
        of absent blocks would be out of bounds). A/Asf source addressing is
        precomputed per thread by the caller (dense or route-gathered).
        """
        # A: blocks x 16 rows x 128B; 128 16B transfers per block, one per
        # thread. The 16B unit within the row is xor-swizzled by row&7 so
        # that the fragment reads (whose q*128 strides alias all rows onto
        # the same banks) become conflict-free.
        row = tid >> Int32(3)
        v = tid & Int32(7)  # 16B unit within the row
        p = v ^ (row & Int32(7))  # swizzled unit
        for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
            if Int32(blk) < n_present:
                cp_async4_shared_global(
                    stage_base
                    + Int32(blk * _A_BLOCK_WORDS * 4)
                    + (row << Int32(7))
                    + (p << Int32(4)),
                    get_ptr_as_int64(a_q_flat, Int32(a_src_off[blk]) + kt * Int32(32)),
                )
        # Asf: one u32 per A row (gmem rows are strided -- 4B copies).
        if tid < (n_present << Int32(4)):
            cp_async_u32_shared_global(
                stage_base + Int32(_ASF_WORD_OFF * 4) + (tid << Int32(2)),
                get_ptr_as_int64(a_sf_flat, asf_src_off + kt),
            )
        # B: contiguous 16KB; 1024 16B transfers, 8 per thread.
        b_off = b_base + kt * Int32(_B_TILE_WORDS)
        for s in cutlass.range_constexpr(_B_TILE_WORDS // (4 * _CTA_THREADS)):
            i = Int32(s * _CTA_THREADS) + tid  # 16B transfer index
            cp_async4_shared_global(
                stage_base + Int32(_B_WORD_OFF * 4) + (i << Int32(4)),
                get_ptr_as_int64(b_rp_flat, b_off + (i << Int32(2))),
            )
        # Bsf: contiguous 1KB; 64 16B transfers.
        if tid < Int32(_BSF_TILE_WORDS // 4):
            cp_async4_shared_global(
                stage_base + Int32(_BSF_WORD_OFF * 4) + (tid << Int32(4)),
                get_ptr_as_int64(
                    b_sf_flat,
                    bsf_base + kt * Int32(_BSF_TILE_WORDS) + (tid << Int32(2)),
                ),
            )

    @cute.jit
    def _run_tile(
        self,
        a_q_flat: cute.Tensor,
        a_sf_flat: cute.Tensor,
        b_rp_flat: cute.Tensor,
        b_sf_flat: cute.Tensor,
        c_u32_flat: cute.Tensor,
        packed_route_indices: cute.Tensor,
        s_base: Int32,
        tid: Int32,
        m_block: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        active_m: Int32,
        total_routes: Int32,
        en_mask: Int32,
        n_present: Int32,
    ):
        warp = tid >> Int32(5)
        lane = tid & Int32(31)
        q = lane >> Int32(2)  # quad row: A rows q / q+8, SFB col q
        c = lane & Int32(3)  # quad column: 8-byte k group within k32

        a_row_base = m_block * Int32(_TILE_M)
        a_row_stride = Int32(self.size_k // 4)  # u32 per A row
        asf_row_stride = Int32(self.k_tiles)  # u32 per A-sf row
        tile_base = (expert_idx * Int32(self.n_tiles) + output_n_tile) * Int32(
            self.k_tiles
        )
        b_base = tile_base * Int32(_B_TILE_WORDS)
        bsf_base = tile_base * Int32(_BSF_TILE_WORDS)

        # ---- Per-thread A / Asf source base offsets (u32 words), computed
        # once before the k-loop so the staging hot loop is mode-agnostic.
        # Staging thread roles: A copy row = tid>>3 / 16B unit = tid&7 per
        # block; Asf copy row = tid (predicated tid < n_present*16).
        stage_row = tid >> Int32(3)
        stage_v = tid & Int32(7)
        a_src_off = cute.make_rmem_tensor((_BLOCKS_PER_CTA,), cutlass.Int32)
        asf_src_row = Int32(0)
        if cutlass.const_expr(self.gather_a):
            for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                src_row = Int32(0)
                if Int32(blk) < n_present:
                    idx = packed_route_indices[
                        a_row_base + Int32(blk * _TILE_M) + stage_row
                    ].to(Int32)
                    if idx < total_routes:
                        src_row = idx // Int32(self.topk)
                a_src_off[blk] = src_row * a_row_stride + (stage_v << Int32(2))
            if tid < (n_present << Int32(4)):
                aidx = packed_route_indices[a_row_base + tid].to(Int32)
                if aidx < total_routes:
                    asf_src_row = aidx // Int32(self.topk)
        else:
            for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                a_src_off[blk] = (
                    a_row_base + Int32(blk * _TILE_M) + stage_row
                ) * a_row_stride + (stage_v << Int32(2))
            asf_src_row = a_row_base + tid
        asf_src_off = asf_src_row * asf_row_stride

        # f32 accumulators: [m-block][n8 tile][4 fragment regs], carried
        # across the full K sweep.
        facc = cute.make_rmem_tensor(
            (_BLOCKS_PER_CTA, _N8_PER_WARP, 4), cutlass.Float32
        )
        facc.fill(0.0)

        # ---- cp.async double-buffer pipeline: prefetch k-tile 0.
        self._issue_stage(
            a_q_flat,
            a_sf_flat,
            b_rp_flat,
            b_sf_flat,
            s_base,
            tid,
            Int32(0),
            a_src_off,
            asf_src_off,
            b_base,
            bsf_base,
            n_present,
        )
        cute.arch.cp_async_commit_group()

        kt = Int32(0)
        k_tiles = Int32(self.k_tiles)
        while kt < k_tiles:
            # Issue k-tile kt+1 into the other stage (its previous reads were
            # fenced by the tail sync of iteration kt-1).
            nxt = kt + Int32(1)
            if nxt < k_tiles:
                self._issue_stage(
                    a_q_flat,
                    a_sf_flat,
                    b_rp_flat,
                    b_sf_flat,
                    s_base + (nxt & Int32(1)) * Int32(_STAGE_BYTES),
                    tid,
                    nxt,
                    a_src_off,
                    asf_src_off,
                    b_base,
                    bsf_base,
                    n_present,
                )
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(1)
            cute.arch.sync_threads()

            cur = s_base + (kt & Int32(1)) * Int32(_STAGE_BYTES)
            sa_base = cur
            sasf_base = cur + Int32(_ASF_WORD_OFF * 4)
            sb_base = cur + Int32(_B_WORD_OFF * 4)
            sbsf_base = cur + Int32(_BSF_WORD_OFF * 4)

            # ---- consume: 4 k32 steps x 8 n8 tiles x m-blocks ----
            # Lane l supplies the SFA byte for row q + 8*(l&1) (quad-cols 2/3
            # are unconsumed; the &1 keeps their loads in range).
            asf_row = q + ((lane & Int32(1)) << Int32(3))
            asc = cute.make_rmem_tensor((_BLOCKS_PER_CTA,), cutlass.Uint32)
            for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                asc[blk] = ld_shared_u32(
                    sasf_base + Int32(blk * _TILE_M * 4) + (asf_row << Int32(2))
                )
            for kb in cutlass.range_constexpr(4):
                # Swizzle-aware fragment loads: logical 16B unit kb*2 + c/2,
                # physical unit xor q (rows q and q+8 swizzle identically).
                u_phys = (Int32(kb * 2) + (c >> Int32(1))) ^ q
                a_lo = (
                    sa_base
                    + (q << Int32(7))
                    + (u_phys << Int32(4))
                    + ((c & Int32(1)) << Int32(3))
                )
                a_frag = cute.make_rmem_tensor((_BLOCKS_PER_CTA, 4), cutlass.Uint32)
                for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                    blk_off = Int32(blk * _A_BLOCK_WORDS * 4)
                    f0, f2 = ld_shared_v2_u32(a_lo + blk_off)
                    f1, f3 = ld_shared_v2_u32(a_lo + blk_off + Int32(8 * 128))
                    a_frag[blk, 0] = f0
                    a_frag[blk, 1] = f1
                    a_frag[blk, 2] = f2
                    a_frag[blk, 3] = f3
                for ch in cutlass.range_constexpr(_N8_PER_WARP // 4):
                    # One conflict-free 16B smem load fetches the lane's 4
                    # n8-chunk words for this k32 step.
                    w0, w1, w2, w3 = ld_shared_v4_u32(
                        sb_base
                        + (
                            (
                                (Int32(kb * 8) + warp * Int32(2) + Int32(ch))
                                * Int32(32)
                                + lane
                            )
                            << Int32(4)
                        )
                    )
                    w_rmem = cute.make_rmem_tensor((4,), cutlass.Uint32)
                    w_rmem[0] = w0
                    w_rmem[1] = w1
                    w_rmem[2] = w2
                    w_rmem[3] = w3
                    for i in cutlass.range_constexpr(4):
                        nt = ch * 4 + i
                        n8 = warp * Int32(_N8_PER_WARP) + Int32(nt)
                        # The expansion + SFB word amortize over all
                        # m-blocks' QMMAs.
                        b0, b1 = _nib8_to_e2m1_bytes(Uint32(w_rmem[i]))
                        sfb_word = ld_shared_u32(
                            sbsf_base + ((n8 * Int32(8) + q) << Int32(2))
                        )
                        # Scale byte kb is selected by the QMMA byte-id
                        # immediates -- no shift/mask extraction.
                        for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                            d0, d1, d2, d3 = _qmma_m16n8k32_f32_e4m3_e2m1(
                                facc[blk, nt, 0],
                                facc[blk, nt, 1],
                                facc[blk, nt, 2],
                                facc[blk, nt, 3],
                                Uint32(a_frag[blk, 0]),
                                Uint32(a_frag[blk, 1]),
                                Uint32(a_frag[blk, 2]),
                                Uint32(a_frag[blk, 3]),
                                b0,
                                b1,
                                Uint32(asc[blk]),
                                sfb_word,
                                bid_a=kb,
                                bid_b=kb,
                            )
                            facc[blk, nt, 0] = d0
                            facc[blk, nt, 1] = d1
                            facc[blk, nt, 2] = d2
                            facc[blk, nt, 3] = d3
            cute.arch.sync_threads()
            kt += Int32(1)

        # ---- epilogue: direct bf16x2 stores via the pinned fragment map.
        # col = output_n_tile*256 + warp*64 + nt*8 + 2c (+1); u32 (bf16x2)
        # column index = col/2 = output_n_tile*128 + warp*32 + nt*4 + c.
        # Block b stores only when enabled (en_mask) and below active_m; in
        # gather mode padding slots (route idx >= total_routes) store ZEROS
        # so the packed C buffer stays deterministic for downstream stages.
        c_row_stride = Int32(self.size_n // 2)
        col_half_base = output_n_tile * Int32(_TILE_N // 2) + warp * Int32(32) + c
        lim = cute.make_rmem_tensor((_BLOCKS_PER_CTA,), cutlass.Int32)
        for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
            blk_lim = Int32(0)
            if (en_mask & Int32(1 << blk)) != Int32(0):
                blk_lim = active_m
            lim[blk] = blk_lim
        if cutlass.const_expr(self.gather_a):
            zero_row = cute.make_rmem_tensor((_BLOCKS_PER_CTA, 2), cutlass.Int32)
            for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                z_lo = Int32(0)
                z_hi = Int32(0)
                if (en_mask & Int32(1 << blk)) != Int32(0):
                    pos_lo = a_row_base + Int32(blk * _TILE_M) + q
                    if packed_route_indices[pos_lo].to(Int32) >= total_routes:
                        z_lo = Int32(1)
                    if (
                        packed_route_indices[pos_lo + Int32(8)].to(Int32)
                        >= total_routes
                    ):
                        z_hi = Int32(1)
                zero_row[blk, 0] = z_lo
                zero_row[blk, 1] = z_hi
        for nt in cutlass.range_constexpr(_N8_PER_WARP):
            col_half = col_half_base + Int32(nt * 4)
            for blk in cutlass.range_constexpr(_BLOCKS_PER_CTA):
                p_lo = pack_f32x2_to_bfloat2(facc[blk, nt, 0], facc[blk, nt, 1])
                p_hi = pack_f32x2_to_bfloat2(facc[blk, nt, 2], facc[blk, nt, 3])
                if cutlass.const_expr(self.gather_a):
                    if Int32(zero_row[blk, 0]) != Int32(0):
                        p_lo = Uint32(0)
                    if Int32(zero_row[blk, 1]) != Int32(0):
                        p_hi = Uint32(0)
                row_lo = a_row_base + Int32(blk * _TILE_M) + q
                row_hi = row_lo + Int32(8)
                if row_lo < Int32(lim[blk]):
                    c_u32_flat[row_lo * c_row_stride + col_half] = p_lo
                if row_hi < Int32(lim[blk]):
                    c_u32_flat[row_hi * c_row_stride + col_half] = p_hi


def to_cute_u32(x: torch.Tensor) -> cute.Tensor:
    """Flat u32 cute view of a contiguous tensor (byte size % 4 == 0)."""
    tensor = from_dlpack(x.view(torch.int32).reshape(-1), assumed_align=16)
    tensor.element_type = cutlass.Uint32
    return tensor
