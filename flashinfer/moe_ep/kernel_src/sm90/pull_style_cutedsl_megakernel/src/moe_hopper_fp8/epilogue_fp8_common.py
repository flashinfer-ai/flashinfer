# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Stateless SM90 FP8 epilogue helpers shared by both operand orders."""

import dataclasses
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.typing import Float32
from cutlass.cutlass_dsl import Int64, T
from cutlass._mlir.dialects import llvm

from common.megamoe_constants import Log2E
from common.moe_utils import fmax, fmin
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase
from src.token_comm import TokenSrcMetadata


@cute.jit
def advance_pipeline_state(state, iterations):
    """Bulk-advance a pipeline state across one inactive task."""
    if iterations < state.stages and (
        state._index + iterations >= state.stages
    ):
        state._phase ^= 1
    if iterations >= state.stages and (
        ((state._index + iterations) // state.stages) % 2 == 1
    ):
        state._phase ^= 1
    state._index = (state._index + iterations) % state.stages
    state._count += iterations
    return state


@cute.jit
def advance_skipped_task_state(
    work_tile_info,
    ab_consumer_state,
    k_tile_cnt_fc1,
    k_tile_cnt_fc2,
):
    """Skip one producer task without touching its full/empty barriers."""
    k_tile_cnt = k_tile_cnt_fc1
    if work_tile_info.phase != cutlass.Int32(BlockPhase.Linear1):
        k_tile_cnt = k_tile_cnt_fc2
    return advance_pipeline_state(ab_consumer_state, k_tile_cnt)


@cute.jit
def consume_initial_pingpong_work(
    sched_consumer,
    warpgroup_idx,
    ab_consumer_state,
    k_tile_cnt_fc1,
    k_tile_cnt_fc2,
):
    work_tile_info = sched_consumer.consume_work()
    if warpgroup_idx == cutlass.Int32(1):
        if work_tile_info.is_valid_tile:
            ab_consumer_state = advance_skipped_task_state(
                work_tile_info,
                ab_consumer_state,
                k_tile_cnt_fc1,
                k_tile_cnt_fc2,
            )
            work_tile_info = sched_consumer.consume_work()
    return sched_consumer, work_tile_info, ab_consumer_state


@cute.jit
def consume_next_pingpong_work(
    sched_consumer,
    ab_consumer_state,
    k_tile_cnt_fc1,
    k_tile_cnt_fc2,
):
    work_tile_info = sched_consumer.consume_work()
    if work_tile_info.is_valid_tile:
        ab_consumer_state = advance_skipped_task_state(
            work_tile_info,
            ab_consumer_state,
            k_tile_cnt_fc1,
            k_tile_cnt_fc2,
        )
        work_tile_info = sched_consumer.consume_work()
    return sched_consumer, work_tile_info, ab_consumer_state


@cute.jit
def clamp_and_swiglu_sm90(
    t_swiglu: cute.Tensor,
    t_up: cute.Tensor,
    t_gate: cute.Tensor,
    glu_clamp,
    prob: Float32,
) -> None:
    """Apply the optional gate/up clamp and scalar SM90 SwiGLU."""
    if cutlass.const_expr(glu_clamp is not None):
        for i in cutlass.range_constexpr(cute.size(t_up)):
            t_gate[i] = fmin(t_gate[i], glu_clamp)
            t_up[i] = fmin(t_up[i], glu_clamp)
            t_up[i] = fmax(t_up[i], -glu_clamp)

    for i in cutlass.range_constexpr(cute.size(t_swiglu)):
        neg_gate_log2e = t_gate[i] * -Log2E
        exp_val = cute.math.exp2(neg_gate_log2e, fastmath=True)
        sigmoid = cute.arch.rcp_approx(exp_val + Float32(1.0))
        t_swiglu[i] = t_up[i] * t_gate[i] * sigmoid * prob


@dataclasses.dataclass(frozen=True)
class Fc2OutputDest:
    """Resolve direct or peer-mapped FC2 output rows."""

    tensor: cute.Tensor
    metadata: Optional[cute.Tensor] = None
    peer_rank_ptr_mapper: Any = None
    reduce_topk_in_kernel: bool = False

    def __post_init__(self) -> None:
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError(
                "Fc2OutputDest: ``metadata`` and ``peer_rank_ptr_mapper`` must be "
                "both None (direct mode) or both non-None (MegaMoE / indirect "
                "mode).  Got metadata="
                f"{'set' if self.metadata is not None else 'None'}, "
                "peer_rank_ptr_mapper="
                f"{'set' if self.peer_rank_ptr_mapper is not None else 'None'}."
            )

    @cute.jit
    def resolve_token_row(self, pool_token_global) -> cute.Tensor:
        if cutlass.const_expr(self.metadata is None):
            return cute.slice_(self.tensor, (pool_token_global, 0, None))

        md = TokenSrcMetadata.load(
            self.metadata.iterator.toint()
            + Int64(pool_token_global) * Int64(TokenSrcMetadata.nbytes)
        )
        src_rank = md.src_rank
        src_token = md.src_token
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            src_topk = cutlass.Int32(0)
        else:
            src_topk = md.src_topk
        local_row = cute.slice_(self.tensor, (src_token, src_topk, None))
        peer_iter = self.peer_rank_ptr_mapper.ptr_map_to_rank(
            local_row.iterator, src_rank,
        )
        return cute.make_tensor(peer_iter, local_row.layout)


@cute.jit
def pack_f32x2_to_bf16x2(lo: Float32, hi: Float32) -> cutlass.Uint32:
    """Round two FP32 values to BF16 and pack them into one 32-bit word.

    ``lo`` lands in the low half (the lower memory address), ``hi`` in the
    high half -- i.e. the pair (col, col + 1) of a row-major BF16 row.
    """
    packed = llvm.inline_asm(
        T.i32(),
        [hi.ir_value(), lo.ir_value()],
        "cvt.rn.bf16x2.f32 $0, $1, $2;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return cutlass.Uint32(packed)


@cute.jit
def bfly_transpose4_u32(
    w0, w1, w2, w3, hi_set, lo_set,
    hi_xor: cutlass.Constexpr, lo_xor: cutlass.Constexpr,
):
    """Two-level butterfly transpose of a 4x4 matrix of 32-bit words held by
    four lanes.

    The four lanes differ in two lane-id bits: ``hi_xor`` and ``lo_xor`` are
    the corresponding shfl.bfly masks and ``hi_set`` / ``lo_set`` tell the
    caller's lane whether each bit is set, so lane index ``q = 2*hi + lo``.
    Lane ``q`` enters with ``V[q][0..3] = (w0, w1, w2, w3)`` -- its word for
    each of four 16-byte chunks -- and leaves with ``V[0..3][q]``: the four
    lanes' words for chunk ``q`` in source-lane order, i.e. the chunk's 16
    bytes ready for one vector store.  All lanes must execute this (no
    predication around the shuffles).
    """
    # Level 1: lanes with the high bit clear keep chunks {0, 1} and send
    # {2, 3}; lanes with it set do the opposite.
    s0 = w0 if hi_set else w2
    s1 = w1 if hi_set else w3
    r0 = cute.arch.shuffle_sync_bfly(s0, hi_xor)
    r1 = cute.arch.shuffle_sync_bfly(s1, hi_xor)
    x0 = r0 if hi_set else w0  # (lane q&1,     kept chunk 0)
    x1 = r1 if hi_set else w1  # (lane q&1,     kept chunk 1)
    x2 = w2 if hi_set else r0  # (lane (q&1)|2, kept chunk 0)
    x3 = w3 if hi_set else r1  # (lane (q&1)|2, kept chunk 1)
    # Level 2: lanes with the low bit clear keep kept-chunk 0 and send 1.
    t0 = x0 if lo_set else x1
    t1 = x2 if lo_set else x3
    u0 = cute.arch.shuffle_sync_bfly(t0, lo_xor)
    u1 = cute.arch.shuffle_sync_bfly(t1, lo_xor)
    o0 = u0 if lo_set else x0  # source lane 0
    o1 = x1 if lo_set else u0  # source lane 1
    o2 = u1 if lo_set else x2  # source lane 2
    o3 = x3 if lo_set else u1  # source lane 3
    return o0, o1, o2, o3


@cute.jit
def stg_128b_bf16x8(
    g_c: cute.Tensor, o0, o1, o2, o3, row, col,
) -> None:
    """One 16-byte store of eight BF16 values (packed as four u32 words, low
    half first) to ``g_c[row, col:col+8]``; ``col`` must be a multiple of 8.

    The target pointer carries an explicit 16-byte alignment assumption:
    autovec on the plain BF16 view has no alignment fact and would split the
    store into 16-bit pieces.
    """
    o_u32 = cute.make_rmem_tensor(4, cutlass.Uint32)
    o_u32[0] = o0
    o_u32[1] = o1
    o_u32[2] = o2
    o_u32[3] = o3
    o_bf16 = cute.recast_tensor(o_u32, cutlass.BFloat16)
    g_chunk = cute.coalesce(
        cute.local_tile(g_c, (1, 8), (row, col // cutlass.Int32(8))),
    )
    dst_ptr = cute.make_ptr(
        cutlass.BFloat16,
        g_chunk.iterator.toint(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    cute.copy(
        cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16,
            num_bits_per_copy=128,
        ),
        o_bf16,
        cute.make_tensor(dst_ptr, cute.make_layout(8)),
    )


@cute.jit
def tma_store_fc1_output(
    sC,
    stage_idx,
    tma_atom_fc1_output: cute.CopyAtom,
    g_fc1_output_subtile_view: cute.Tensor,
    valid_tokens,
) -> None:
    """Issue one WG-private FC1 TMA store for a non-empty token tile."""
    sC_stage = cute.slice_(sC, (None, None, stage_idx))
    g_fc1_output_2d = cute.slice_(g_fc1_output_subtile_view, (None, None, 0))
    bSG_sC, bSG_g = cpasync.tma_partition(
        tma_atom_fc1_output,
        0,
        cute.make_layout(1),
        cute.group_modes(sC_stage, 0, 2),
        cute.group_modes(g_fc1_output_2d, 0, 2),
    )

    tile_has_valid = valid_tokens > cutlass.Int32(0)
    if tile_has_valid:
        with cute.arch.elect_one():
            cute.copy(tma_atom_fc1_output, bSG_sC, bSG_g)


@cute.jit
def tma_store_fc2_output(
    sD,
    stage_idx,
    tma_atom_fc2_output: cute.CopyAtom,
    g_fc2_output_subtile_view: cute.Tensor,
    valid_tokens,
) -> None:
    """Issue one WG-private BF16 FC2 TMA store for a non-empty token tile."""
    sD_stage = cute.slice_(sD, (None, None, stage_idx))
    g_fc2_output_2d = cute.slice_(g_fc2_output_subtile_view, (None, None, 0))
    bSG_sD, bSG_g = cpasync.tma_partition(
        tma_atom_fc2_output,
        0,
        cute.make_layout(1),
        cute.group_modes(sD_stage, 0, 2),
        cute.group_modes(g_fc2_output_2d, 0, 2),
    )

    if valid_tokens > cutlass.Int32(0):
        with cute.arch.elect_one():
            cute.copy(tma_atom_fc2_output, bSG_sD, bSG_g)


@cute.jit
def stg_fc1_block_scale_row(
    real_fc1_output_sf: cute.Tensor,
    scale_col_idx,
    token_idx,
    scale: Float32,
) -> None:
    """Store one FP32 blockwise FC1-output scale."""
    sf_base = cute.local_tile(
        real_fc1_output_sf,
        (1, 1, 1),
        (token_idx, scale_col_idx, cutlass.Int32(0)),
    )
    gmem_sf = cute.make_tensor(sf_base.iterator, cute.make_layout(1))
    rmem_sf = cute.make_rmem_tensor((1,), cutlass.Float32)
    rmem_sf[0] = scale
    cute.autovec_copy(rmem_sf, gmem_sf)
