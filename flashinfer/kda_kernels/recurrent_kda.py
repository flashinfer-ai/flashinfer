"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Recurrent KDA (Key-Driven Attention) decode kernels using CuTe DSL for SM100.

Supports single-token decode (T=1) and fused speculative decode
(T=1+num_spec_tokens) with per-key-dimension gating. The logical recurrent
state is S[K,V] and is stored transposed in the kernel as state[...,V,K] bf16.
Per token:
  S = exp(g)[..., None] * S + beta * outer(k, v - k @ S)
  o = q @ S

Standard decode inputs use q,k [B,1,H,K], v [B,1,HV,V], g [B,1,HV,K],
and beta [B,1,HV]. With cu_seqlens, tokens are packed on dim 1 as
[1,total_tokens,...], state is [N,HV,V,K], and ssm_state_indices maps
sequences/checkpoints to state slots.

Supports GQA (H != HV), cu_seqlens, paged state indices, speculative decode,
and compile-time gate modes (pre-computed, softplus, lower_bound * sigmoid).
"""

import functools
import math
import os
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import utils
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math
from cutlass._mlir.dialects import nvvm
import tvm_ffi  # noqa: F401 -- TVM FFI required for zero-overhead kernel dispatch

# ==============================================================================
# CONSTANTS
# ==============================================================================
# SMEM H padding for bank conflict avoidance.
# H_SMEM_STRIDE = HEAD_DIM + H_SMEM_PADDING (computed inside kernels from HEAD_DIM).
# Constraint: (stride * 2) must have >=16 as highest power-of-2 factor for cp.async 128-bit
# alignment.
#   HEAD_DIM=128: stride=136 -> 272 bytes -> 272 = 16*17 -> align<16> ok
#   HEAD_DIM=64:  stride=72  -> 144 bytes -> 144 = 16*9  -> align<16> ok
# Bank analysis: stride=136 and stride=72 both give 4-way conflicts (acceptable).
H_SMEM_PADDING = 8


# ==============================================================================
# SHARED HELPER FUNCTIONS
# ==============================================================================


@cute.jit
def write_h_chunk_to_smem(h_chunk_f32, h_sh_chunk, lane_idx, k_base):
    """Write F32 register H chunk to BF16 SMEM."""
    for i in cutlass.range_constexpr(32):
        h_sh_chunk[lane_idx, k_base + i] = h_chunk_f32[i].to(cutlass.BFloat16)


@cute.jit
def store_h_smem_to_gmem(
    h_sh_chunk,
    h_out,
    tidx,
    v_row_offset,
    HEAD_DIM: cutlass.Constexpr[int],
    V_TILE_ROWS: cutlass.Constexpr[int] = 32,
):
    """Store H from SMEM to GMEM using 128-bit stores.

    V_TILE_ROWS: number of V-rows in this chunk (see load_h_chunk_async).
    """
    copy_bits = 128
    copy_elems = copy_bits // cutlass.BFloat16.width

    from cutlass.cute.nvgpu import CopyUniversalOp

    if HEAD_DIM == 64:
        # 64 threads: (8, 8) thread layout, (8, 64) tiles, V_TILE_ROWS/8 row iters
        thr_layout = cute.make_layout((8, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_store = cute.make_copy_atom(
            CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=copy_bits
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_store, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(V_TILE_ROWS // 8):
            s_tile = cute.local_tile(h_sh_chunk, (8, 64), (row_iter, 0))
            g_tile = cute.local_tile(
                h_out, (8, 64), (row_iter + (v_row_offset // 8), 0)
            )
            tS = thr_copy.partition_S(s_tile)
            tD = thr_copy.partition_D(g_tile)
            cute.copy(atom_store, tS, tD)
    elif HEAD_DIM == 128:
        # 128 threads: (16, 8) thread layout, (16, 64) tiles, V_TILE_ROWS/16 row x 2 col iters
        thr_layout = cute.make_layout((16, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_store = cute.make_copy_atom(
            CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=copy_bits
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_store, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(V_TILE_ROWS // 16):
            for col_iter in cutlass.range_constexpr(2):
                s_tile = cute.local_tile(h_sh_chunk, (16, 64), (row_iter, col_iter))
                g_tile = cute.local_tile(
                    h_out, (16, 64), (row_iter + (v_row_offset // 16), col_iter)
                )
                tS = thr_copy.partition_S(s_tile)
                tD = thr_copy.partition_D(g_tile)
                cute.copy(atom_store, tS, tD)


@cute.jit
def load_h_chunk_async(
    h_sh_chunk,
    h_global,
    tidx,
    row_offset,
    HEAD_DIM: cutlass.Constexpr[int],
    V_TILE_ROWS: cutlass.Constexpr[int] = 32,
):
    """Load H chunk from GMEM to SMEM using async copy.

    V_TILE_ROWS: number of V-rows in this chunk. Default 32 matches the
    base kernel's ping-pong chunk size and vtile's standard 32-row slice.
    V_TILE_ROWS=16 enables the finer-grained vtile variant that doubles
    grid expansion at D=128 for very low batch.
    """
    copy_bits = 128
    copy_elems = copy_bits // cutlass.BFloat16.width

    if HEAD_DIM == 64:
        # 64 threads: (8, 8) thread layout, (8, 64) tiles, V_TILE_ROWS/8 row iters
        thr_layout = cute.make_layout((8, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            cutlass.BFloat16,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(V_TILE_ROWS // 8):
            g_tile = cute.local_tile(
                h_global, (8, 64), (row_iter + (row_offset // 8), 0)
            )
            s_tile = cute.local_tile(h_sh_chunk, (8, 64), (row_iter, 0))
            tS = thr_copy.partition_S(g_tile)
            tD = thr_copy.partition_D(s_tile)
            cute.copy(atom_async_copy, tS, tD)
    elif HEAD_DIM == 128:
        # 128 threads: (16, 8) thread layout, (16, 64) tiles, V_TILE_ROWS/16 row x 2 col iters
        thr_layout = cute.make_layout((16, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            cutlass.BFloat16,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_async_copy, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(V_TILE_ROWS // 16):
            for col_iter in cutlass.range_constexpr(2):
                g_tile = cute.local_tile(
                    h_global, (16, 64), (row_iter + (row_offset // 16), col_iter)
                )
                s_tile = cute.local_tile(h_sh_chunk, (16, 64), (row_iter, col_iter))
                tS = thr_copy.partition_S(g_tile)
                tD = thr_copy.partition_D(s_tile)
                cute.copy(atom_async_copy, tS, tD)


@cute.jit
def load_qkvg_async(
    q_raw_sh,
    k_raw_sh,
    v_raw_sh,
    g_raw_sh,
    q_head,
    k_head,
    v_head,
    g_head,
    tidx,
    HEAD_DIM: cutlass.Constexpr[int],
):
    """Issue cp.async for Q/K/V/G bf16 vectors in a single group.

    Each vector is 1D [HEAD_DIM] bf16 = HEAD_DIM*2 bytes. cute.nvgpu cpasync
    only supports 128-bit (cp.async.cg) copies, so each participating thread
    loads 8 bf16 (16 bytes). Thread split:
      D=128: 16 threads/vector * 4 vectors = 64 threads (warps 0-1).
             Warps 2-3 idle through the issue (no divergence: warp-aligned).
      D=64:   8 threads/vector * 4 vectors = 32 threads (warp 0 only).
             Warp 1 idle (warp-aligned).

    Replaces four per-token scalar `ld.global + st.shared` chains that NCU
    flagged as the top L1TEX scoreboard stall (~40% of warp-cycles/issue at
    B=1, H32-D128-lb). Caller MUST `cp_async_commit_group` + `wait_group(0)`
    + `sync_threads` before reading the `*_raw_sh` buffers.
    """
    copy_bits = 128
    copy_elems = copy_bits // cutlass.BFloat16.width  # 8
    N_THR_PER_VEC: cutlass.Constexpr[int] = HEAD_DIM // copy_elems
    N_ISSUE_THR: cutlass.Constexpr[int] = 4 * N_THR_PER_VEC

    atom_async = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(
            cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
        ),
        cutlass.BFloat16,
        num_bits_per_copy=copy_bits,
    )
    thr_layout = cute.make_layout((N_THR_PER_VEC,))
    val_layout = cute.make_layout((copy_elems,))
    tiled_copy = cute.make_tiled_copy_tv(atom_async, thr_layout, val_layout)

    # D=128 assigns one vector to each warp so all four warps contribute to
    # the producer phase before the block-wide publication barrier. D=64 keeps
    # the compact single-warp issue path.
    should_issue = tidx < N_ISSUE_THR
    vec_idx = tidx // N_THR_PER_VEC
    local_tidx = tidx % N_THR_PER_VEC
    if HEAD_DIM == 128:
        lane_idx = tidx % 32
        should_issue = lane_idx < N_THR_PER_VEC
        vec_idx = tidx // 32
        local_tidx = lane_idx

    if should_issue:
        thr_copy = tiled_copy.get_slice(local_tidx)

        if vec_idx == 0:
            cute.copy(
                atom_async,
                thr_copy.partition_S(q_head),
                thr_copy.partition_D(q_raw_sh),
            )
        elif vec_idx == 1:
            cute.copy(
                atom_async,
                thr_copy.partition_S(k_head),
                thr_copy.partition_D(k_raw_sh),
            )
        elif vec_idx == 2:
            cute.copy(
                atom_async,
                thr_copy.partition_S(v_head),
                thr_copy.partition_D(v_raw_sh),
            )
        else:
            cute.copy(
                atom_async,
                thr_copy.partition_S(g_head),
                thr_copy.partition_D(g_raw_sh),
            )


@cute.jit
def issue_qkvg_async_for_token(
    q_raw_sh,
    k_raw_sh,
    v_raw_sh,
    g_raw_sh,
    gQ,
    gK,
    gV,
    gG,
    batch_idx,
    token_offset,
    query_head_idx,
    value_head_idx,
    tidx,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
):
    """Issue the per-token Q/K/V/G async staging copies with token views scoped locally."""
    if cutlass.const_expr(USE_CU_SEQLENS == 1):
        q_head = gQ[(0, token_offset, query_head_idx, None)]
        k_head = gK[(0, token_offset, query_head_idx, None)]
        v_head = gV[(0, token_offset, value_head_idx, None)]
        g_head = gG[(0, token_offset, value_head_idx, None)]
    else:
        q_head = gQ[(batch_idx, 0, query_head_idx, None)]
        k_head = gK[(batch_idx, 0, query_head_idx, None)]
        v_head = gV[(batch_idx, 0, value_head_idx, None)]
        g_head = gG[(batch_idx, 0, value_head_idx, None)]

    load_qkvg_async(
        q_raw_sh,
        k_raw_sh,
        v_raw_sh,
        g_raw_sh,
        q_head,
        k_head,
        v_head,
        g_head,
        tidx,
        HEAD_DIM,
    )


@cute.jit
def process_vtile_token_chunk(
    gBeta: cute.Tensor,
    gO: cute.Tensor,
    batch_idx,
    token_offset,
    value_head_idx,
    h_sh,
    h_chunk,
    lane_idx,
    warp_idx,
    k_base,
    g_sh,
    k_sh,
    q_sh,
    v_sh,
    pred_sh,
    out_sh,
    v_offset,
    is_active,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
    V_TILE_ROWS: cutlass.Constexpr[int] = 32,
):
    """Resolve per-token beta/output views right at the chunk compute site."""
    if cutlass.const_expr(USE_CU_SEQLENS == 1):
        beta = gBeta[(0, token_offset, value_head_idx)].to(cutlass.Float32)
        o_head = gO[(0, token_offset, value_head_idx, None)]
    else:
        beta = gBeta[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
        o_head = gO[(batch_idx, 0, value_head_idx, None)]

    _process_v_chunk(
        h_sh,
        h_sh,
        h_chunk,
        lane_idx,
        warp_idx,
        k_base,
        g_sh,
        k_sh,
        q_sh,
        v_sh,
        pred_sh,
        out_sh,
        v_offset,
        o_head,
        beta,
        is_active,
        HEAD_DIM,
        V_TILE_ROWS,
    )


@cute.jit
def store_vtile_token_state(
    gH: cute.Tensor,
    seq_idx,
    value_head_idx,
    h_sh,
    tidx,
    v_offset,
    HEAD_DIM: cutlass.Constexpr[int],
    V_TILE_ROWS: cutlass.Constexpr[int] = 32,
):
    """Materialize the token's state output view only for the GMEM checkpoint store."""
    h_out = gH[(seq_idx, value_head_idx, None, None)]
    store_h_smem_to_gmem(h_sh, h_out, tidx, v_offset, HEAD_DIM, V_TILE_ROWS)


@cute.jit
def compute_gate_to_smem(
    g_sh,
    g_head,
    k_base,
    lane_idx,
    A_log_val,
    gDtBias,
    h_K_offset,
    lower_bound_val,
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
):
    """Compute exp(gate) and store to SMEM. Each lane writes one element.

    When USE_GATE_IN_KERNEL=0: g is pre-computed log-space gate, just exp it.
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=0:
        g_log = -exp(A_log) * softplus(g + dt_bias); return exp(g_log)
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=1:
        g_log = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)); return exp(g_log)

    Args:
        g_sh: output SMEM tensor [HEAD_DIM] Float32
        g_head: global gate tensor slice [K=HEAD_DIM]
        k_base: starting K index for this warp (warp_idx * 32)
        lane_idx: lane index within warp (0..31)
        A_log_val: exp(A_log) for this head (Float32), precomputed outside
        gDtBias: dt_bias tensor [H*K] Float32 (or dummy if HAS_DT_BIAS=0)
        h_K_offset: query_head_idx * K offset into dt_bias
        lower_bound_val: lower bound float (negative, e.g. -5.0)
        USE_GATE_IN_KERNEL: 0 = pre-computed gate, 1 = in-kernel gate
        HAS_DT_BIAS: 0 = no dt_bias, 1 = add dt_bias
        USE_LOWER_BOUND: 0 = softplus formula, 1 = lower_bound * sigmoid formula

    Note: the dt_bias ld.global is intentionally left inside the token loop.
    Hoisting it to CTA entry (measured 2026-04-17) regressed +2% kernel
    duration at H32-D128-lb B=1 -- the scalar load scheduled at CTA entry
    has nothing to overlap with, whereas inside the loop the scheduler
    overlaps it with other compute/SMEM work.
    """
    g_val = g_head[k_base + lane_idx].to(cutlass.Float32)
    if USE_GATE_IN_KERNEL == 1:
        if HAS_DT_BIAS == 1:
            g_val = g_val + gDtBias[h_K_offset + k_base + lane_idx].to(cutlass.Float32)
        if USE_LOWER_BOUND == 1:
            LOG2_E = cutlass.Float32(1.4426950408889634)
            neg_A_log2e = -A_log_val * LOG2_E
            lb_log2e = lower_bound_val * LOG2_E
            one = cutlass.Float32(1.0)
            fm = mlir_arith.FastMathFlags.fast
            one_ir = one.ir_value()
            neg_A_ir = neg_A_log2e.ir_value()
            lb_ir = lb_log2e.ir_value()
            ag = mlir_arith.mulf(neg_A_ir, g_val.ir_value(), fastmath=fm)
            exp_neg = mlir_math.exp2(ag, fastmath=fm)
            denom = mlir_arith.addf(one_ir, exp_neg, fastmath=fm)
            sig = mlir_arith.divf(one_ir, denom, fastmath=fm)
            ls = mlir_arith.mulf(lb_ir, sig, fastmath=fm)
            g_val = mlir_math.exp2(ls, fastmath=fm)
        else:
            exp_g = cute.exp(g_val, fastmath=True)
            one = cutlass.Float32(1.0)
            log2_v = cute.log2(one + exp_g, fastmath=True)
            g_val = cute.exp2(-A_log_val * log2_v, fastmath=True)
    else:
        g_val = cute.exp(g_val, fastmath=True)
    g_sh[k_base + lane_idx] = g_val


@cute.jit
def normalize_and_store_qk_to_smem(
    q_head,
    k_head,
    q_sh,
    k_sh,
    lane_idx,
    scale,
    eps,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[int],
):
    """Optionally L2-normalize Q/K vectors, then store to shared memory."""
    # ELEMS_PER_LANE = HEAD_DIM // 32 (2 for HD=64, 4 for HD=128)
    q_reg = cute.make_rmem_tensor((HEAD_DIM // 32,), cutlass.Float32)
    k_reg = cute.make_rmem_tensor((HEAD_DIM // 32,), cutlass.Float32)

    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        q_reg[i] = q_head[lane_idx + i * 32].to(cutlass.Float32)
        k_reg[i] = k_head[lane_idx + i * 32].to(cutlass.Float32)

    if USE_QK_L2NORM == 1:
        q_sum_sq = cutlass.Float32(0.0)
        k_sum_sq = cutlass.Float32(0.0)
        q_sum_sq2 = cutlass.Float32(0.0)
        k_sum_sq2 = cutlass.Float32(0.0)

        for i in cutlass.range_constexpr(0, HEAD_DIM // 32, 2):
            q_sum_sq, q_sum_sq2 = cute.arch.fma_packed_f32x2(
                src_a=(q_reg[i], q_reg[i + 1]),
                src_b=(q_reg[i], q_reg[i + 1]),
                src_c=(q_sum_sq, q_sum_sq2),
            )
            k_sum_sq, k_sum_sq2 = cute.arch.fma_packed_f32x2(
                src_a=(k_reg[i], k_reg[i + 1]),
                src_b=(k_reg[i], k_reg[i + 1]),
                src_c=(k_sum_sq, k_sum_sq2),
            )

        q_sum_sq = q_sum_sq + q_sum_sq2
        k_sum_sq = k_sum_sq + k_sum_sq2

        # Butterfly shuffle: always 5 rounds (32 lanes per warp, hardware constant)
        for i in cutlass.range_constexpr(5):
            q_sum_sq = q_sum_sq + cute.arch.shuffle_sync_bfly(
                q_sum_sq, offset=1 << i, mask=0xFFFFFFFF
            )
            k_sum_sq = k_sum_sq + cute.arch.shuffle_sync_bfly(
                k_sum_sq, offset=1 << i, mask=0xFFFFFFFF
            )

        q_norm = cute.rsqrt(q_sum_sq + eps, fastmath=True)
        k_norm = cute.rsqrt(k_sum_sq + eps, fastmath=True)
        q_scale_factor = q_norm * scale

        for i in cutlass.range_constexpr(HEAD_DIM // 32):
            q_sh[lane_idx + i * 32] = q_reg[i] * q_scale_factor
            k_sh[lane_idx + i * 32] = k_reg[i] * k_norm
    else:
        for i in cutlass.range_constexpr(HEAD_DIM // 32):
            q_sh[lane_idx + i * 32] = q_reg[i] * scale
            k_sh[lane_idx + i * 32] = k_reg[i]


# ==============================================================================
# SEQLEN=1 KERNEL
# ==============================================================================


@cute.jit
def _process_v_chunk(
    h_sh_src,
    h_sh_dst,
    h_chunk,
    lane_idx,
    warp_idx,
    k_base,
    g_sh,
    k_sh,
    q_sh,
    v_sh,
    pred_sh,
    out_sh,
    v_offset,
    o_head,
    beta,
    is_active,
    HEAD_DIM: cutlass.Constexpr[int],
    V_TILE_ROWS: cutlass.Constexpr[int] = 32,
):
    """Process one V-chunk: decay state, compute prediction, update state, write output.

    h_sh_src: SMEM buffer holding this chunk's state (read)
    h_sh_dst: SMEM buffer to write updated state back to (for GMEM store)
    h_chunk: register tensor [32] for this chunk's state slice
    v_offset: V-row offset for indexing v_sh and o_head.
    V_TILE_ROWS: rows owned by this CTA (32 or 16). At 16, lanes >=16 are idle;
        their h_chunk/pred/out values are garbage and MUST NOT write to h_sh_dst
        or o_head (both sized by V_TILE_ROWS or HEAD_DIM respectively).
        Caller pads v_sh by LANES_PER_WARP (=32) so idle-lane v_sh reads at
        v_offset + lane_idx stay in-bounds.
    """
    # Fused decay + pred
    pred = cutlass.Float32(0.0)
    pred2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(
                h_sh_src[lane_idx, k_base + i].to(cutlass.Float32),
                h_sh_src[lane_idx, k_base + i + 1].to(cutlass.Float32),
            ),
            src_b=(g_sh[k_base + i], g_sh[k_base + i + 1]),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )
        pred, pred2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(k_sh[k_base + i], k_sh[k_base + i + 1]),
            src_c=(pred, pred2),
        )
    pred = pred + pred2

    pred_sh[warp_idx, lane_idx] = pred
    cute.arch.sync_threads()
    pred_final = cutlass.Float32(0.0)
    if HEAD_DIM == 64:
        pred_final = pred_sh[0, lane_idx] + pred_sh[1, lane_idx]
    elif HEAD_DIM == 128:
        pred_final = (
            pred_sh[0, lane_idx]
            + pred_sh[1, lane_idx]
            + pred_sh[2, lane_idx]
            + pred_sh[3, lane_idx]
        )

    v_val = (v_sh[v_offset + lane_idx] - pred_final) * beta

    # Fused update + output
    out = cutlass.Float32(0.0)
    out2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(k_sh[k_base + i], k_sh[k_base + i + 1]),
            src_b=(v_val, v_val),
            src_c=(h_chunk[i], h_chunk[i + 1]),
        )
        out, out2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(q_sh[k_base + i], q_sh[k_base + i + 1]),
            src_c=(out, out2),
        )
    out = out + out2

    out_sh[warp_idx, lane_idx] = out
    cute.arch.sync_threads()
    out_final = cutlass.Float32(0.0)
    if HEAD_DIM == 64:
        out_final = out_sh[0, lane_idx] + out_sh[1, lane_idx]
    elif HEAD_DIM == 128:
        out_final = (
            out_sh[0, lane_idx]
            + out_sh[1, lane_idx]
            + out_sh[2, lane_idx]
            + out_sh[3, lane_idx]
        )

    # Lanes >= V_TILE_ROWS hold garbage (no V-row assigned). Gate h_sh writeback
    # and output write to keep them out of bounds. At V_TILE_ROWS=32 (default)
    # the guard is trivially true for all 32 lanes -- zero behavior change.
    if lane_idx < V_TILE_ROWS:
        write_h_chunk_to_smem(h_chunk, h_sh_dst, lane_idx, k_base)
        if is_active:
            if warp_idx == 0:
                o_head[v_offset + lane_idx] = out_final.to(cutlass.BFloat16)


@cute.kernel
def recurrent_kda_decode_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gG: cute.Tensor,  # [B, T, HV, K] log-space gate (or raw input if USE_GATE_IN_KERNEL)
    gBeta: cute.Tensor,  # [B, T, HV] pre-sigmoided
    gH: cute.Tensor,  # state: bf16 [N,HV,V,K] (modified in-place)
    gO: cute.Tensor,
    gALog: cute.Tensor,  # [H] float32 (A_log per query head)
    gDtBias: cute.Tensor,  # [H*K] float32 (dt_bias per head and K)
    gCuSeqlens: cute.Tensor,  # [N+1] int32 -- raw cu_seqlens
    gSsmStateIndices: cute.Tensor,  # [N*NUM_TOKENS] int32 -- flattened ssm_state_indices (2D [N,T] in spec mode, 1D [N] otherwise)
    gNumAcceptedTokens: cute.Tensor,  # [N] int32 -- per-sequence accepted token count for spec decode initial state selection
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
    NUM_TOKENS: cutlass.Constexpr[int],
):
    """Multi-token spec-decode kernel. One CTA per (batch, head) pair.

    For NUM_TOKENS==1: behavior identical to the original T=1 kernel (zero regression).
    For NUM_TOKENS>1: D=64 uses register-carry (h_chunk_v0/v1 persist across tokens);
    D=128 uses GMEM round-trip (reload from previous token's output slot each iteration).

    Thread mapping: HEAD_DIM threads = (HEAD_DIM // 32) warps x 32 lanes.
    State [V,K] tiled into V-chunks of 32 rows (2 for HD=64, 4 for HD=128).
    2 ping-pong H SMEM buffers; K/G/Q read from SMEM (broadcast).
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    HV = cutlass.Int32(gV.shape[2])
    H = cutlass.Int32(gQ.shape[2])

    batch_idx = bidx // HV
    value_head_idx = bidx % HV
    query_head_idx = value_head_idx // (HV // H)

    # cu_seqlens base offset (zero cost when USE_CU_SEQLENS=0)
    token_base_offset = cutlass.Int32(0)
    seq_len = cutlass.Int32(1)  # default: all entries valid when no cu_seqlens
    if USE_CU_SEQLENS == 1:
        token_base_offset = gCuSeqlens[batch_idx].to(cutlass.Int32)
        seq_len = gCuSeqlens[batch_idx + 1].to(cutlass.Int32) - token_base_offset

    # Precompute gate params (guarded by Constexpr, zero cost when unused)
    A_log_val = cutlass.Float32(0.0)
    h_K_offset = cutlass.Int32(0)
    lower_bound_val = lower_bound
    if USE_GATE_IN_KERNEL == 1:
        A_log_val = cute.exp(gALog[query_head_idx].to(cutlass.Float32), fastmath=True)
        h_K_offset = query_head_idx * HEAD_DIM

    smem = utils.SmemAllocator()

    # Allocate SMEM -- 2 ping-pong H buffers (D=128 reuses for chunks 2,3)
    h_sh_a = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_b = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )

    q_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    g_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    pred_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((HEAD_DIM // 32, 32))
    )
    out_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((HEAD_DIM // 32, 32))
    )
    v_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    warp_idx = tidx // 32
    lane_idx = tidx % 32
    k_base = warp_idx * 32

    # Register state for V-chunks.
    # D=64: h_chunk_v0/v1 carry state across tokens (register-carry).
    # D=128: h_chunk is reused per V-chunk (GMEM round-trip, no persistence).
    # All three allocated unconditionally to satisfy SCF loop-carried variable rules.
    h_chunk_v0 = cute.make_rmem_tensor((32,), cutlass.Float32)
    h_chunk_v1 = cute.make_rmem_tensor((32,), cutlass.Float32)
    h_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)

    # Pre-initialize ALL variables assigned inside the token loop body.
    # CuTe DSL SCF requirement: names assigned inside cutlass.range() must have
    # initial values before the loop, even inside Constexpr-dead branches.
    seq_idx = batch_idx
    init_seq_idx = (
        batch_idx  # SCF pre-init: used only at token_t==0 for nat-based initial state
    )
    is_active = seq_len > 0
    token_offset = token_base_offset
    beta = cutlass.Float32(0.0)
    raw_slot = cutlass.Int32(0)
    raw_slot_prev = cutlass.Int32(0)
    seq_idx_prev = batch_idx
    # Tensor views: SCF dummy inits (overwritten each iteration before use).
    # gH uses batch_idx (state pool has enough slots). gG/gQ/gK/gV/gO use 0
    # because in cu_seqlens mode dim-0 is 1 while batch_idx can exceed it.
    h_global_in = gH[(batch_idx, value_head_idx, None, None)]
    h_global_prev = gH[(batch_idx, value_head_idx, None, None)]
    h_global_in_c2 = gH[(batch_idx, value_head_idx, None, None)]
    h_global_in_c3 = gH[(batch_idx, value_head_idx, None, None)]
    g_head = gG[(0, 0, value_head_idx, None)]
    q_head = gQ[(0, 0, query_head_idx, None)]
    k_head = gK[(0, 0, query_head_idx, None)]
    v_head = gV[(0, 0, value_head_idx, None)]
    o_head = gO[(0, 0, value_head_idx, None)]
    h_out = gH[(batch_idx, value_head_idx, None, None)]

    # ========================================================================
    # TOKEN LOOP
    # ========================================================================
    for token_t in cutlass.range(NUM_TOKENS):
        # ----------------------------------------------------------------
        # Derive state slot for this token
        # ----------------------------------------------------------------
        if USE_CU_SEQLENS == 1:
            # 1D flat indexing: for NUM_TOKENS>1, ssm_state_indices is flattened [N*T]
            # so batch_idx * NUM_TOKENS + token_t gives the right element.
            # For NUM_TOKENS==1, this reduces to batch_idx * 1 + 0 = batch_idx.
            raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS + token_t].to(
                cutlass.Int32
            )
            is_active = raw_slot >= 0
            seq_idx = cutlass.Int32(0) if raw_slot < 0 else raw_slot

            # Derive init_seq_idx from num_accepted_tokens for initial state selection.
            # When num_accepted_tokens[n]=nat, the initial state is at
            # ssm_state_indices[n, nat-1] (the checkpoint after the last accepted token
            # from the previous spec decode round). This matches the FLA Triton kernel.
            # For nat<=1 or NUM_TOKENS==1, nat_offset=0 -> same as current behavior.
            # Guard with NUM_TOKENS > 1 (Constexpr) so dummy tensors are never accessed
            # in standard decode where gNumAcceptedTokens may be a 1-element dummy.
            if token_t == 0:
                if NUM_TOKENS > 1:
                    nat_raw = gNumAcceptedTokens[batch_idx].to(cutlass.Int32)
                    nat_offset = cutlass.Int32(0) if nat_raw <= 1 else (nat_raw - 1)
                    init_raw_slot = gSsmStateIndices[
                        batch_idx * NUM_TOKENS + nat_offset
                    ].to(cutlass.Int32)
                    init_seq_idx = (
                        cutlass.Int32(0) if init_raw_slot < 0 else init_raw_slot
                    )
                else:
                    init_seq_idx = seq_idx
        else:
            # No cu_seqlens: batch_idx is the state slot, always active
            seq_idx = batch_idx
            is_active = seq_len > 0

        token_offset = token_base_offset + token_t

        # ----------------------------------------------------------------
        # State loading
        # ----------------------------------------------------------------
        if token_t == 0:
            # First token: async load from initial state slot (nat-selected)
            h_global_in = gH[(init_seq_idx, value_head_idx, None, None)]
            load_h_chunk_async(h_sh_a, h_global_in, tidx, 0, HEAD_DIM)
            nvvm.cp_async_commit_group()
            load_h_chunk_async(h_sh_b, h_global_in, tidx, 32, HEAD_DIM)
            nvvm.cp_async_commit_group()
        else:
            if HEAD_DIM == 64:
                # D=64 register-carry: write carried registers to SMEM
                write_h_chunk_to_smem(h_chunk_v0, h_sh_a, lane_idx, k_base)
                write_h_chunk_to_smem(h_chunk_v1, h_sh_b, lane_idx, k_base)
                cute.arch.sync_threads()
            else:
                # D=128 GMEM round-trip: reload from previous token's output slot
                if USE_CU_SEQLENS == 1 and NUM_TOKENS > 1:
                    raw_slot_prev = gSsmStateIndices[
                        batch_idx * NUM_TOKENS + token_t - 1
                    ].to(cutlass.Int32)
                    seq_idx_prev = (
                        cutlass.Int32(0) if raw_slot_prev < 0 else raw_slot_prev
                    )
                else:
                    seq_idx_prev = seq_idx
                h_global_prev = gH[(seq_idx_prev, value_head_idx, None, None)]
                load_h_chunk_async(h_sh_a, h_global_prev, tidx, 0, HEAD_DIM)
                nvvm.cp_async_commit_group()
                load_h_chunk_async(h_sh_b, h_global_prev, tidx, 32, HEAD_DIM)
                nvvm.cp_async_commit_group()

        # ----------------------------------------------------------------
        # Per-token data: Q, K, V, G, beta, output pointer
        # ----------------------------------------------------------------
        if USE_CU_SEQLENS == 1:
            g_head = gG[(0, token_offset, value_head_idx, None)]
            beta = gBeta[(0, token_offset, value_head_idx)].to(cutlass.Float32)
            q_head = gQ[(0, token_offset, query_head_idx, None)]
            k_head = gK[(0, token_offset, query_head_idx, None)]
            v_head = gV[(0, token_offset, value_head_idx, None)]
            o_head = gO[(0, token_offset, value_head_idx, None)]
        else:
            g_head = gG[(batch_idx, 0, value_head_idx, None)]
            beta = gBeta[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
            q_head = gQ[(batch_idx, 0, query_head_idx, None)]
            k_head = gK[(batch_idx, 0, query_head_idx, None)]
            v_head = gV[(batch_idx, 0, value_head_idx, None)]
            o_head = gO[(batch_idx, 0, value_head_idx, None)]

        h_out = gH[(seq_idx, value_head_idx, None, None)]

        # Q/K L2 normalization (warp 0 only)
        if warp_idx == 0:
            normalize_and_store_qk_to_smem(
                q_head,
                k_head,
                q_sh,
                k_sh,
                lane_idx,
                scale,
                eps,
                HEAD_DIM,
                USE_QK_L2NORM,
            )

        cute.arch.sync_threads()

        # V load
        v_sh[tidx] = v_head[tidx].to(cutlass.Float32)

        # ====================================================================
        # CHUNK 0 (from h_sh_a)
        # ====================================================================
        # Wait for chunk 0 async load (D=64 t>0 has no pending async)
        if HEAD_DIM == 64 and token_t > 0:
            pass  # registers -> SMEM already done above + sync_threads
        else:
            nvvm.cp_async_wait_group(1)
            cute.arch.sync_threads()

        # Compute per-K gate to SMEM
        compute_gate_to_smem(
            g_sh,
            g_head,
            k_base,
            lane_idx,
            A_log_val,
            gDtBias,
            h_K_offset,
            lower_bound_val,
            USE_GATE_IN_KERNEL,
            HAS_DT_BIAS,
            USE_LOWER_BOUND,
        )
        cute.arch.sync_threads()

        if HEAD_DIM == 64:
            _process_v_chunk(
                h_sh_a,
                h_sh_a,
                h_chunk_v0,
                lane_idx,
                warp_idx,
                k_base,
                g_sh,
                k_sh,
                q_sh,
                v_sh,
                pred_sh,
                out_sh,
                0,
                o_head,
                beta,
                is_active,
                HEAD_DIM,
            )
        else:
            _process_v_chunk(
                h_sh_a,
                h_sh_a,
                h_chunk,
                lane_idx,
                warp_idx,
                k_base,
                g_sh,
                k_sh,
                q_sh,
                v_sh,
                pred_sh,
                out_sh,
                0,
                o_head,
                beta,
                is_active,
                HEAD_DIM,
            )

        # ====================================================================
        # CHUNK 1 (from h_sh_b)
        # ====================================================================
        # Wait for chunk 1 async load (D=64 t>0 has no pending async)
        if HEAD_DIM == 64 and token_t > 0:
            pass  # registers -> SMEM already done above
        else:
            nvvm.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # Store chunk 0; for D=128 also load chunk 2 into h_sh_a (ping-pong)
        if is_active:
            store_h_smem_to_gmem(h_sh_a, h_out, tidx, 0, HEAD_DIM)
        if HEAD_DIM == 128:
            if token_t == 0:
                h_global_in_c2 = gH[(init_seq_idx, value_head_idx, None, None)]
            else:
                if USE_CU_SEQLENS == 1 and NUM_TOKENS > 1:
                    raw_slot_prev = gSsmStateIndices[
                        batch_idx * NUM_TOKENS + token_t - 1
                    ].to(cutlass.Int32)
                    seq_idx_prev = (
                        cutlass.Int32(0) if raw_slot_prev < 0 else raw_slot_prev
                    )
                else:
                    seq_idx_prev = seq_idx
                h_global_in_c2 = gH[(seq_idx_prev, value_head_idx, None, None)]
            load_h_chunk_async(h_sh_a, h_global_in_c2, tidx, 64, HEAD_DIM)
            nvvm.cp_async_commit_group()

        if HEAD_DIM == 64:
            _process_v_chunk(
                h_sh_b,
                h_sh_b,
                h_chunk_v1,
                lane_idx,
                warp_idx,
                k_base,
                g_sh,
                k_sh,
                q_sh,
                v_sh,
                pred_sh,
                out_sh,
                32,
                o_head,
                beta,
                is_active,
                HEAD_DIM,
            )
        else:
            _process_v_chunk(
                h_sh_b,
                h_sh_b,
                h_chunk,
                lane_idx,
                warp_idx,
                k_base,
                g_sh,
                k_sh,
                q_sh,
                v_sh,
                pred_sh,
                out_sh,
                32,
                o_head,
                beta,
                is_active,
                HEAD_DIM,
            )

        # For HEAD_DIM=64: done after 2 chunks. Store chunk1 H and continue to next token.
        if HEAD_DIM == 64:
            cute.arch.sync_threads()
            if is_active:
                store_h_smem_to_gmem(h_sh_b, h_out, tidx, 32, HEAD_DIM)

        # ====================================================================
        # CHUNK 2 (HEAD_DIM=128 only, from h_sh_a via ping-pong)
        # ====================================================================
        if HEAD_DIM == 128:
            cute.arch.sync_threads()

            # Store chunk 1 (h_sh_b), load chunk 3 into h_sh_b (ping-pong)
            if is_active:
                store_h_smem_to_gmem(h_sh_b, h_out, tidx, 32, HEAD_DIM)
            if token_t == 0:
                h_global_in_c3 = gH[(init_seq_idx, value_head_idx, None, None)]
            else:
                if USE_CU_SEQLENS == 1 and NUM_TOKENS > 1:
                    raw_slot_prev = gSsmStateIndices[
                        batch_idx * NUM_TOKENS + token_t - 1
                    ].to(cutlass.Int32)
                    seq_idx_prev = (
                        cutlass.Int32(0) if raw_slot_prev < 0 else raw_slot_prev
                    )
                else:
                    seq_idx_prev = seq_idx
                h_global_in_c3 = gH[(seq_idx_prev, value_head_idx, None, None)]
            load_h_chunk_async(h_sh_b, h_global_in_c3, tidx, 96, HEAD_DIM)
            nvvm.cp_async_commit_group()

            nvvm.cp_async_wait_group(1)
            cute.arch.sync_threads()

            _process_v_chunk(
                h_sh_a,
                h_sh_a,
                h_chunk,
                lane_idx,
                warp_idx,
                k_base,
                g_sh,
                k_sh,
                q_sh,
                v_sh,
                pred_sh,
                out_sh,
                64,
                o_head,
                beta,
                is_active,
                HEAD_DIM,
            )

            # ================================================================
            # CHUNK 3 (HEAD_DIM=128 only, from h_sh_b via ping-pong)
            # ================================================================
            nvvm.cp_async_wait_group(0)
            cute.arch.sync_threads()

            if is_active:
                store_h_smem_to_gmem(h_sh_a, h_out, tidx, 64, HEAD_DIM)

            _process_v_chunk(
                h_sh_b,
                h_sh_b,
                h_chunk,
                lane_idx,
                warp_idx,
                k_base,
                g_sh,
                k_sh,
                q_sh,
                v_sh,
                pred_sh,
                out_sh,
                96,
                o_head,
                beta,
                is_active,
                HEAD_DIM,
            )

            cute.arch.sync_threads()
            if is_active:
                store_h_smem_to_gmem(h_sh_b, h_out, tidx, 96, HEAD_DIM)

        # Token boundary barrier (required for BOTH D=64 and D=128):
        # - D=64: prevents SMEM h_sh_b race between store_h_smem_to_gmem (reads h_sh_b)
        #   in this iteration and write_h_chunk_to_smem (writes h_sh_b) at the start
        #   of the next iteration's register-carry prelude.
        # - D=128: per bar.sync (PTX ISA) all pending memory transactions from all threads
        #   complete before any thread proceeds, ensuring st.global stores are visible to
        #   the subsequent cp.async reloads in the next iteration's GMEM round-trip.
        cute.arch.sync_threads()


# ==============================================================================
# D128 T=4 CHUNK-MAJOR BASE KERNEL
# ==============================================================================


@cute.kernel
def recurrent_kda_decode_chunk_major_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gG: cute.Tensor,
    gBeta: cute.Tensor,
    gH: cute.Tensor,
    gO: cute.Tensor,
    gALog: cute.Tensor,
    gDtBias: cute.Tensor,
    gCuSeqlens: cute.Tensor,
    gSsmStateIndices: cute.Tensor,
    gNumAcceptedTokens: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
    NUM_TOKENS: cutlass.Constexpr[int],
):
    """D128 T=4 base kernel with V-chunk-outer, token-inner traversal."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    HV = cutlass.Int32(gV.shape[2])
    H = cutlass.Int32(gQ.shape[2])
    batch_idx = bidx // HV
    value_head_idx = bidx % HV
    query_head_idx = value_head_idx // (HV // H)

    token_base_offset = cutlass.Int32(0)
    seq_len = cutlass.Int32(1)
    if USE_CU_SEQLENS == 1:
        token_base_offset = gCuSeqlens[batch_idx].to(cutlass.Int32)
        seq_len = gCuSeqlens[batch_idx + 1].to(cutlass.Int32) - token_base_offset

    init_seq_idx = batch_idx
    if USE_CU_SEQLENS == 1:
        nat_raw = gNumAcceptedTokens[batch_idx].to(cutlass.Int32)
        nat_offset = cutlass.Int32(0) if nat_raw <= 1 else (nat_raw - 1)
        init_raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS + nat_offset].to(
            cutlass.Int32
        )
        init_seq_idx = cutlass.Int32(0) if init_raw_slot < 0 else init_raw_slot

    A_log_val = cutlass.Float32(0.0)
    h_K_offset = cutlass.Int32(0)
    if USE_GATE_IN_KERNEL == 1:
        A_log_val = cute.exp(gALog[query_head_idx].to(cutlass.Float32), fastmath=True)
        h_K_offset = query_head_idx * HEAD_DIM

    smem = utils.SmemAllocator()
    h_sh_a = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_b = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    q_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    g_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    pred_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((HEAD_DIM // 32, 32))
    )
    out_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((HEAD_DIM // 32, 32))
    )
    v_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    warp_idx = tidx // 32
    lane_idx = tidx % 32
    k_base = warp_idx * 32
    h_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)
    seq_idx = batch_idx
    is_active = seq_len > 0
    token_offset = token_base_offset
    beta = cutlass.Float32(0.0)
    raw_slot = cutlass.Int32(0)
    g_head = gG[(0, 0, value_head_idx, None)]
    q_head = gQ[(0, 0, query_head_idx, None)]
    k_head = gK[(0, 0, query_head_idx, None)]
    v_head = gV[(0, 0, value_head_idx, None)]
    o_head = gO[(0, 0, value_head_idx, None)]
    h_out = gH[(batch_idx, value_head_idx, None, None)]
    h_global_in = gH[(init_seq_idx, value_head_idx, None, None)]

    for pair_idx in cutlass.range_constexpr(2):
        v_offset_a = pair_idx * 64
        v_offset_b = v_offset_a + 32
        load_h_chunk_async(h_sh_a, h_global_in, tidx, v_offset_a, HEAD_DIM)
        nvvm.cp_async_commit_group()
        load_h_chunk_async(h_sh_b, h_global_in, tidx, v_offset_b, HEAD_DIM)
        nvvm.cp_async_commit_group()

        for token_t in cutlass.range(NUM_TOKENS):
            if USE_CU_SEQLENS == 1:
                raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS + token_t].to(
                    cutlass.Int32
                )
                is_active = raw_slot >= 0
                seq_idx = cutlass.Int32(0) if raw_slot < 0 else raw_slot
            else:
                seq_idx = batch_idx
                is_active = seq_len > 0

            token_offset = token_base_offset + token_t
            if USE_CU_SEQLENS == 1:
                g_head = gG[(0, token_offset, value_head_idx, None)]
                beta = gBeta[(0, token_offset, value_head_idx)].to(cutlass.Float32)
                q_head = gQ[(0, token_offset, query_head_idx, None)]
                k_head = gK[(0, token_offset, query_head_idx, None)]
                v_head = gV[(0, token_offset, value_head_idx, None)]
                o_head = gO[(0, token_offset, value_head_idx, None)]
            else:
                g_head = gG[(batch_idx, 0, value_head_idx, None)]
                beta = gBeta[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
                q_head = gQ[(batch_idx, 0, query_head_idx, None)]
                k_head = gK[(batch_idx, 0, query_head_idx, None)]
                v_head = gV[(batch_idx, 0, value_head_idx, None)]
                o_head = gO[(batch_idx, 0, value_head_idx, None)]
            h_out = gH[(seq_idx, value_head_idx, None, None)]

            if warp_idx == 0:
                normalize_and_store_qk_to_smem(
                    q_head,
                    k_head,
                    q_sh,
                    k_sh,
                    lane_idx,
                    scale,
                    eps,
                    HEAD_DIM,
                    USE_QK_L2NORM,
                )
            cute.arch.sync_threads()
            v_sh[tidx] = v_head[tidx].to(cutlass.Float32)

            if token_t == 0:
                nvvm.cp_async_wait_group(1)
                cute.arch.sync_threads()

            compute_gate_to_smem(
                g_sh,
                g_head,
                k_base,
                lane_idx,
                A_log_val,
                gDtBias,
                h_K_offset,
                lower_bound,
                USE_GATE_IN_KERNEL,
                HAS_DT_BIAS,
                USE_LOWER_BOUND,
            )
            cute.arch.sync_threads()

            _process_v_chunk(
                h_sh_a,
                h_sh_a,
                h_chunk,
                lane_idx,
                warp_idx,
                k_base,
                g_sh,
                k_sh,
                q_sh,
                v_sh,
                pred_sh,
                out_sh,
                v_offset_a,
                o_head,
                beta,
                is_active,
                HEAD_DIM,
            )

            if token_t == 0:
                nvvm.cp_async_wait_group(0)
            cute.arch.sync_threads()
            if is_active:
                store_h_smem_to_gmem(h_sh_a, h_out, tidx, v_offset_a, HEAD_DIM)

            _process_v_chunk(
                h_sh_b,
                h_sh_b,
                h_chunk,
                lane_idx,
                warp_idx,
                k_base,
                g_sh,
                k_sh,
                q_sh,
                v_sh,
                pred_sh,
                out_sh,
                v_offset_b,
                o_head,
                beta,
                is_active,
                HEAD_DIM,
            )
            cute.arch.sync_threads()
            if is_active:
                store_h_smem_to_gmem(h_sh_b, h_out, tidx, v_offset_b, HEAD_DIM)
            cute.arch.sync_threads()


# ==============================================================================
# V-TILED KERNEL (experimental: one V-chunk per CTA)
# ==============================================================================
# Unlike `recurrent_kda_decode_kernel` which runs one CTA per (batch, head) and
# processes all V rows sequentially with a 2- or 4-way SMEM ping-pong, this
# variant runs one CTA per (batch, head, v_tile) with a single 32-row V-chunk
# per CTA. Expands the grid by HEAD_DIM/32 (2x for D=64, 4x for D=128) to help
# saturate SMs at small batch sizes where `grid = B * HV` under-subscribes the
# GPU (at B=4, HV=16, D=128: 64 CTAs on ~148 SMs, NCU measured 6% occupancy).
#
# Trade-offs:
# - Wins at small B (more CTAs land on more SMs).
# - Loses the intra-CTA async overlap between chunks (each CTA has only one).
# - Duplicates Q/K L2 norm, gate computation, and per-token metadata reads
#   across the NUM_V_TILES CTAs per (batch, head) -- small compared to the
#   state update work.
# - h register-carry applies uniformly for D=64 and D=128 here (each CTA owns
#   only 32 rows, fits cleanly in registers). D=128 gets a bonus: no GMEM
#   round-trip at token boundaries (the original kernel pays this cost).


@cute.kernel
def recurrent_kda_decode_vtile_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gG: cute.Tensor,
    gBeta: cute.Tensor,
    gH: cute.Tensor,
    gO: cute.Tensor,
    gALog: cute.Tensor,
    gDtBias: cute.Tensor,
    gCuSeqlens: cute.Tensor,
    gSsmStateIndices: cute.Tensor,
    gNumAcceptedTokens: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
    NUM_TOKENS: cutlass.Constexpr[int],
    NUM_V_TILES: cutlass.Constexpr[int],
    V_TILE_ROWS: cutlass.Constexpr[int],
):
    """V-tiled decode: one CTA per (batch, head, v_tile). Grid = B*HV*NUM_V_TILES.

    V_TILE_ROWS: 32 (standard, NUM_V_TILES=HEAD_DIM/32) or 16 (finer-grained,
        NUM_V_TILES=HEAD_DIM/16). V_TILE_ROWS=16 doubles grid at D=128 for
        very-low-batch shapes at the cost of 50% idle lanes in the state
        reductions (warps-per-CTA stays 4, so scheduler warp-parallelism
        is preserved -- the Round 2A ILP rejection concern doesn't apply).
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    HV = cutlass.Int32(gV.shape[2])
    H = cutlass.Int32(gQ.shape[2])

    # Decode block_idx -> (batch_idx, value_head_idx, v_tile_idx)
    v_tile_idx = bidx % NUM_V_TILES
    bh = bidx // NUM_V_TILES
    value_head_idx = bh % HV
    batch_idx = bh // HV
    query_head_idx = value_head_idx // (HV // H)

    # Absolute V-row offset of this CTA's chunk in the full V vector
    v_offset = v_tile_idx * V_TILE_ROWS

    # cu_seqlens base offset (zero cost when USE_CU_SEQLENS=0)
    token_base_offset = cutlass.Int32(0)
    seq_len = cutlass.Int32(1)
    if USE_CU_SEQLENS == 1:
        token_base_offset = gCuSeqlens[batch_idx].to(cutlass.Int32)
        seq_len = gCuSeqlens[batch_idx + 1].to(cutlass.Int32) - token_base_offset

    # Precompute gate params (guarded by Constexpr, zero cost when unused)
    A_log_val = cutlass.Float32(0.0)
    h_K_offset = cutlass.Int32(0)
    lower_bound_val = lower_bound
    if USE_GATE_IN_KERNEL == 1:
        A_log_val = cute.exp(gALog[query_head_idx].to(cutlass.Float32), fastmath=True)
        h_K_offset = query_head_idx * HEAD_DIM

    smem = utils.SmemAllocator()

    # Single SMEM h buffer -- no ping-pong, only one chunk per CTA.
    # Physical row count is always 32 (=LANES_PER_WARP), even at V_TILE_ROWS=16.
    # _process_v_chunk reads h_sh[lane_idx, k_base+i] for all 32 lanes (the
    # per-lane FMA loop is unpredicated so block-wide syncs stay well-formed);
    # idle lanes at V_TILE_ROWS=16 read rows 16-31 which would OOB past the
    # SMEM allocation (measured: IMA at D=64 V_TILE=16 due to OOB past dynamic
    # SMEM total). Allocating 32 rows always trades ~2KB SMEM for safety and
    # keeps the helpers parameterized on V_TILE_ROWS only for their valid-row
    # count (load/store stride by V_TILE_ROWS*16 rows, etc.).
    h_sh = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )

    q_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    g_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    pred_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((HEAD_DIM // 32, 32))
    )
    out_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((HEAD_DIM // 32, 32))
    )
    # v_sh padded by one warp width so idle lanes (lane_idx >= V_TILE_ROWS) can
    # read v_sh[v_offset + lane_idx] without going OOB. Max index at
    # V_TILE_ROWS=16: (NUM_V_TILES-1)*16 + 31 = HEAD_DIM+15; pad to HEAD_DIM+32.
    # Negligible SMEM cost (~128 bytes) and keeps _process_v_chunk branchless
    # on the v_val computation path (the block-wide syncs inside can't be
    # skipped by idle lanes).
    v_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM + 32)

    # T>1 ping-pongs raw inputs so token t+1 can be in flight while token t
    # computes. T=1 has one bank and the constexpr shape adds no extra storage.
    N_RAW_BANKS: cutlass.Constexpr[int] = 2 if NUM_TOKENS > 1 else 1
    raw_layout = cute.make_layout((N_RAW_BANKS, HEAD_DIM), stride=(HEAD_DIM, 1))
    q_raw_sh = smem.allocate_tensor(cutlass.BFloat16, raw_layout, byte_alignment=16)
    k_raw_sh = smem.allocate_tensor(cutlass.BFloat16, raw_layout, byte_alignment=16)
    v_raw_sh = smem.allocate_tensor(cutlass.BFloat16, raw_layout, byte_alignment=16)
    g_raw_sh = smem.allocate_tensor(cutlass.BFloat16, raw_layout, byte_alignment=16)

    warp_idx = tidx // 32
    lane_idx = tidx % 32
    k_base = warp_idx * 32

    # Single register-resident h chunk. Carried across tokens for both D=64 and
    # D=128 (the v-tiled structure lets D=128 carry cleanly since each CTA owns
    # only 32 rows, unlike the base kernel which must GMEM-round-trip at D=128).
    h_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)

    # SCF pre-inits (see CuTe DSL loop-carried variable rules in the base kernel)
    seq_idx = batch_idx
    init_seq_idx = batch_idx
    is_active = seq_len > 0
    token_offset = token_base_offset
    raw_slot = cutlass.Int32(0)

    for token_t in cutlass.range(NUM_TOKENS):
        # ----------------------------------------------------------------
        # Resolve state slot + initial state slot
        # ----------------------------------------------------------------
        if USE_CU_SEQLENS == 1:
            raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS + token_t].to(
                cutlass.Int32
            )
            is_active = raw_slot >= 0
            seq_idx = cutlass.Int32(0) if raw_slot < 0 else raw_slot

            # nat-based initial state (matches base kernel and FLA Triton semantics)
            if token_t == 0:
                if NUM_TOKENS > 1:
                    nat_raw = gNumAcceptedTokens[batch_idx].to(cutlass.Int32)
                    nat_offset = cutlass.Int32(0) if nat_raw <= 1 else (nat_raw - 1)
                    init_raw_slot = gSsmStateIndices[
                        batch_idx * NUM_TOKENS + nat_offset
                    ].to(cutlass.Int32)
                    init_seq_idx = (
                        cutlass.Int32(0) if init_raw_slot < 0 else init_raw_slot
                    )
                else:
                    init_seq_idx = seq_idx
        else:
            seq_idx = batch_idx
            is_active = seq_len > 0

        token_offset = token_base_offset + token_t
        raw_bank = token_t % N_RAW_BANKS
        q_raw = q_raw_sh[(raw_bank, None)]
        k_raw = k_raw_sh[(raw_bank, None)]
        v_raw = v_raw_sh[(raw_bank, None)]
        g_raw = g_raw_sh[(raw_bank, None)]

        # ----------------------------------------------------------------
        # Batched async gmem->smem issue.
        #   Token 0: H state slice + Q/K/V/G per-token vectors.
        #   Token >0: Q/K/V/G only (H is register-carried from prior iter).
        # All into one commit group so a single wait_group(0) drains them.
        # ----------------------------------------------------------------
        if token_t == 0:
            load_h_chunk_async(
                h_sh,
                gH[(init_seq_idx, value_head_idx, None, None)],
                tidx,
                v_offset,
                HEAD_DIM,
                V_TILE_ROWS,
            )
            issue_qkvg_async_for_token(
                q_raw,
                k_raw,
                v_raw,
                g_raw,
                gQ,
                gK,
                gV,
                gG,
                batch_idx,
                token_offset,
                query_head_idx,
                value_head_idx,
                tidx,
                HEAD_DIM,
                USE_CU_SEQLENS,
            )
            nvvm.cp_async_commit_group()

        # For token>0, h_sh already contains the prior token's BF16 checkpoint
        # written by _process_v_chunk. The cp.async destinations are disjoint.

        # Drain all async loads and publish the staging buffers in one
        # cross-thread sync.
        nvvm.cp_async_wait_group(0)
        cute.arch.sync_threads()

        if token_t + 1 < NUM_TOKENS:
            next_token_offset = token_base_offset + token_t + 1
            next_raw_bank = (token_t + 1) % N_RAW_BANKS
            issue_qkvg_async_for_token(
                q_raw_sh[(next_raw_bank, None)],
                k_raw_sh[(next_raw_bank, None)],
                v_raw_sh[(next_raw_bank, None)],
                g_raw_sh[(next_raw_bank, None)],
                gQ,
                gK,
                gV,
                gG,
                batch_idx,
                next_token_offset,
                query_head_idx,
                value_head_idx,
                tidx,
                HEAD_DIM,
                USE_CU_SEQLENS,
            )
            nvvm.cp_async_commit_group()

        # ----------------------------------------------------------------
        # Consume bf16 staging buffers. Reads are cross-warp (e.g., warp 3
        # wrote g_raw_sh, warp 0 reads g_raw_sh[0..31]); the sync above
        # makes them visible.
        # ----------------------------------------------------------------
        # Q/K L2 norm (warp 0 only; duplicated across v-tile CTAs but
        # negligible -- ~HEAD_DIM flops vs. state update's ~32*HEAD_DIM).
        if warp_idx == 0:
            normalize_and_store_qk_to_smem(
                q_raw,
                k_raw,
                q_sh,
                k_sh,
                lane_idx,
                scale,
                eps,
                HEAD_DIM,
                USE_QK_L2NORM,
            )

        # V: bf16 staging -> f32 working buffer. Same-thread read/write;
        # cross-warp visibility is resolved by the next sync_threads().
        v_sh[tidx] = v_raw[tidx].to(cutlass.Float32)

        # Per-K gate: each warp reads its own g_raw_sh[k_base..k_base+31]
        # slice. Writes go to g_sh at disjoint per-warp offsets.
        compute_gate_to_smem(
            g_sh,
            g_raw,
            k_base,
            lane_idx,
            A_log_val,
            gDtBias,
            h_K_offset,
            lower_bound_val,
            USE_GATE_IN_KERNEL,
            HAS_DT_BIAS,
            USE_LOWER_BOUND,
        )
        cute.arch.sync_threads()

        # Process this CTA's single V-chunk. h_chunk registers come out holding
        # the updated state for this chunk (used by next token's register-carry).
        process_vtile_token_chunk(
            gBeta,
            gO,
            batch_idx,
            token_offset,
            value_head_idx,
            h_sh,
            h_chunk,
            lane_idx,
            warp_idx,
            k_base,
            g_sh,
            k_sh,
            q_sh,
            v_sh,
            pred_sh,
            out_sh,
            v_offset,
            is_active,
            HEAD_DIM,
            USE_CU_SEQLENS,
            V_TILE_ROWS,
        )

        cute.arch.sync_threads()

        # Checkpoint: store updated state to GMEM slot for this token
        if is_active:
            store_vtile_token_state(
                gH,
                seq_idx,
                value_head_idx,
                h_sh,
                tidx,
                v_offset,
                HEAD_DIM,
                V_TILE_ROWS,
            )


# ==============================================================================
# LAUNCH WRAPPERS
# ==============================================================================


@cute.jit
def recurrent_kda_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mG: cute.Tensor,
    mBeta: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    mALog: cute.Tensor,
    mDtBias: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mSsmStateIndices: cute.Tensor,
    mNumAcceptedTokens: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
    NUM_TOKENS: cutlass.Constexpr[int],
    USE_CHUNK_MAJOR: cutlass.Constexpr[int],
):
    batch_size = mQ.shape[0]
    if USE_CU_SEQLENS == 1:
        batch_size = mCuSeqlens.shape[0] - 1
    HV = mV.shape[2]

    if cutlass.const_expr(USE_CHUNK_MAJOR == 1):
        kernel = recurrent_kda_decode_chunk_major_kernel
    else:
        kernel = recurrent_kda_decode_kernel

    kernel(
        mQ,
        mK,
        mV,
        mG,
        mBeta,
        mH,
        mO,
        mALog,
        mDtBias,
        mCuSeqlens,
        mSsmStateIndices,
        mNumAcceptedTokens,
        scale,
        eps,
        lower_bound,
        HEAD_DIM,
        USE_QK_L2NORM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
        NUM_TOKENS,
    ).launch(
        grid=[batch_size * HV, 1, 1],
        block=[HEAD_DIM, 1, 1],
        stream=stream,
    )


@cute.jit
def recurrent_kda_vtile_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mG: cute.Tensor,
    mBeta: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    mALog: cute.Tensor,
    mDtBias: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mSsmStateIndices: cute.Tensor,
    mNumAcceptedTokens: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
    NUM_TOKENS: cutlass.Constexpr[int],
    NUM_V_TILES: cutlass.Constexpr[int],
    V_TILE_ROWS: cutlass.Constexpr[int],
):
    batch_size = mQ.shape[0]
    if USE_CU_SEQLENS == 1:
        batch_size = mCuSeqlens.shape[0] - 1
    HV = mV.shape[2]

    recurrent_kda_decode_vtile_kernel(
        mQ,
        mK,
        mV,
        mG,
        mBeta,
        mH,
        mO,
        mALog,
        mDtBias,
        mCuSeqlens,
        mSsmStateIndices,
        mNumAcceptedTokens,
        scale,
        eps,
        lower_bound,
        HEAD_DIM,
        USE_QK_L2NORM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
        NUM_TOKENS,
        NUM_V_TILES,
        V_TILE_ROWS,
    ).launch(
        grid=[batch_size * HV * NUM_V_TILES, 1, 1],
        block=[HEAD_DIM, 1, 1],
        stream=stream,
    )


@cute.jit
def recurrent_kda_vtile_spec_decode_launch(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mG: cute.Tensor,
    mBeta: cute.Tensor,
    mH: cute.Tensor,
    mO: cute.Tensor,
    mALog: cute.Tensor,
    mDtBias: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mSsmStateIndices: cute.Tensor,
    mNumAcceptedTokens: cute.Tensor,
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
    NUM_TOKENS: cutlass.Constexpr[int],
    NUM_V_TILES: cutlass.Constexpr[int],
    V_TILE_ROWS: cutlass.Constexpr[int],
):
    """Dedicated compile surface for D128 multi-token vtile tuning."""
    recurrent_kda_vtile_launch(
        mQ,
        mK,
        mV,
        mG,
        mBeta,
        mH,
        mO,
        mALog,
        mDtBias,
        mCuSeqlens,
        mSsmStateIndices,
        mNumAcceptedTokens,
        scale,
        eps,
        lower_bound,
        stream,
        HEAD_DIM,
        USE_QK_L2NORM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
        NUM_TOKENS,
        NUM_V_TILES,
        V_TILE_ROWS,
    )


# ==============================================================================
# PUBLIC API
# ==============================================================================

_dummy_cache = {}  # device -> dict of pre-allocated dummy tensors
_num_sms_cache = {}  # device -> int, SM count (used by auto-dispatch)
# Keeps the D128 spec-decode vtile kernel within the tuned occupancy/register budget.
_VTILE_SPEC_D128_MAXRREGCOUNT = 72


def _get_num_sms(device):
    if device not in _num_sms_cache:
        _num_sms_cache[device] = torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    return _num_sms_cache[device]


@functools.cache
def _get_compiled_kernel(
    HEAD_DIM,
    USE_QK_L2NORM,
    USE_GATE_IN_KERNEL,
    HAS_DT_BIAS,
    USE_LOWER_BOUND,
    USE_CU_SEQLENS,
    NUM_TOKENS=1,
    USE_CHUNK_MAJOR=0,
):
    """Cache compiled kernel for given configuration."""
    B, H, HV, N = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    K, V = HEAD_DIM, HEAD_DIM

    def make_fake(shape, dtype=cute.BFloat16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=32,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    T_dim = cute.sym_int() if USE_CU_SEQLENS == 1 else 1
    HV_state, ALog_sym, HK_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()
    CuSeqlens_sym, SsiB_sym, Nat_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()

    # State tensor: use make_fake_tensor with a free symbolic stride[0] to support
    # vLLM's block-based cache where stride[0] = page_size_bytes // dtype_size
    # (includes conv_state padding, so it's NOT compact = HV * V * K).
    # stride[1:] are standard row-major within each block.
    S_batch = cute.sym_int64(divisibility=16)
    state_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(N, HV_state, V, K),
        stride=(S_batch, V * K, K, 1),
        assumed_align=32,
    )

    # Gate tensor: use make_fake_tensor with free symbolic strides for batch and
    # token dims to support non-contiguous gates from split()/view() on fused
    # projection outputs (e.g. vLLM [B, T, total_proj_dim].split(..., dim=-1)).
    # HV and K strides remain compact since split() only affects outer dims.
    G_batch_stride = cute.sym_int64(divisibility=16)
    G_token_stride = cute.sym_int64(divisibility=16)
    gate_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(B, T_dim, HV, K),
        stride=(G_batch_stride, G_token_stride, K, 1),
        assumed_align=32,
    )

    return cute.compile(
        recurrent_kda_launch,
        make_fake((B, T_dim, H, K)),  # q
        make_fake((B, T_dim, H, K)),  # k
        make_fake((B, T_dim, HV, V)),  # v
        gate_fake,  # g (free batch/token strides for non-contiguous views)
        make_fake((B, T_dim, HV)),  # beta
        state_fake,  # state (free stride[0] for block-based cache)
        make_fake((B, T_dim, HV, V)),  # output
        make_fake((ALog_sym,), dtype=cute.Float32),  # A_log
        make_fake((HK_sym,), dtype=cute.Float32),  # dt_bias
        make_fake((CuSeqlens_sym,), dtype=cute.Int32),  # cu_seqlens
        make_fake((SsiB_sym,), dtype=cute.Int32),  # ssm_state_indices (flattened 1D)
        make_fake((Nat_sym,), dtype=cute.Int32),  # num_accepted_tokens [N]
        cutlass.Float32(0.0),  # scale
        cutlass.Float32(0.0),  # eps
        cutlass.Float32(0.0),  # lower_bound
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        HEAD_DIM,
        USE_QK_L2NORM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
        NUM_TOKENS,
        USE_CHUNK_MAJOR,
        options="--enable-tvm-ffi --generate-line-info",
    )


@functools.cache
def _get_compiled_vtile_kernel(
    HEAD_DIM,
    USE_QK_L2NORM,
    USE_GATE_IN_KERNEL,
    HAS_DT_BIAS,
    USE_LOWER_BOUND,
    USE_CU_SEQLENS,
    NUM_TOKENS=1,
    V_TILE_ROWS=32,
):
    """Cache compiled v-tiled kernel. NUM_V_TILES = HEAD_DIM / V_TILE_ROWS.

    V_TILE_ROWS=32: standard (one CTA owns 32 V-rows).
    V_TILE_ROWS=16: finer-grained grid at D=128 for very-low-batch; 2x grid
        vs V_TILE_ROWS=32 at cost of 50% idle lanes in state reductions.
    """
    B, H, HV, N = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    K, V = HEAD_DIM, HEAD_DIM
    NUM_V_TILES = HEAD_DIM // V_TILE_ROWS

    def make_fake(shape, dtype=cute.BFloat16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=32,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    T_dim = cute.sym_int() if USE_CU_SEQLENS == 1 else 1
    HV_state, ALog_sym, HK_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()
    CuSeqlens_sym, SsiB_sym, Nat_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()

    S_batch = cute.sym_int64(divisibility=16)
    state_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(N, HV_state, V, K),
        stride=(S_batch, V * K, K, 1),
        assumed_align=32,
    )

    G_batch_stride = cute.sym_int64(divisibility=16)
    G_token_stride = cute.sym_int64(divisibility=16)
    gate_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(B, T_dim, HV, K),
        stride=(G_batch_stride, G_token_stride, K, 1),
        assumed_align=32,
    )

    return cute.compile(
        recurrent_kda_vtile_launch,
        make_fake((B, T_dim, H, K)),  # q
        make_fake((B, T_dim, H, K)),  # k
        make_fake((B, T_dim, HV, V)),  # v
        gate_fake,  # g
        make_fake((B, T_dim, HV)),  # beta
        state_fake,  # state
        make_fake((B, T_dim, HV, V)),  # output
        make_fake((ALog_sym,), dtype=cute.Float32),  # A_log
        make_fake((HK_sym,), dtype=cute.Float32),  # dt_bias
        make_fake((CuSeqlens_sym,), dtype=cute.Int32),  # cu_seqlens
        make_fake((SsiB_sym,), dtype=cute.Int32),  # ssm_state_indices (flattened)
        make_fake((Nat_sym,), dtype=cute.Int32),  # num_accepted_tokens
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        HEAD_DIM,
        USE_QK_L2NORM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
        NUM_TOKENS,
        NUM_V_TILES,
        V_TILE_ROWS,
        options="--enable-tvm-ffi --generate-line-info",
    )


@functools.cache
def _get_compiled_vtile_spec_decode_d128_kernel(
    USE_QK_L2NORM,
    USE_GATE_IN_KERNEL,
    HAS_DT_BIAS,
    USE_LOWER_BOUND,
    USE_CU_SEQLENS,
    NUM_TOKENS,
    V_TILE_ROWS=32,
):
    """Dedicated D128 multi-token vtile compile surface with a moderate register cap."""
    HEAD_DIM = 128
    B, H, HV, N = cute.sym_int(), cute.sym_int(), cute.sym_int(), cute.sym_int()
    K, V = HEAD_DIM, HEAD_DIM
    NUM_V_TILES = HEAD_DIM // V_TILE_ROWS

    def make_fake(shape, dtype=cute.BFloat16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=32,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    T_dim = cute.sym_int() if USE_CU_SEQLENS == 1 else 1
    HV_state, ALog_sym, HK_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()
    CuSeqlens_sym, SsiB_sym, Nat_sym = cute.sym_int(), cute.sym_int(), cute.sym_int()

    S_batch = cute.sym_int64(divisibility=16)
    state_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(N, HV_state, V, K),
        stride=(S_batch, V * K, K, 1),
        assumed_align=32,
    )

    G_batch_stride = cute.sym_int64(divisibility=16)
    G_token_stride = cute.sym_int64(divisibility=16)
    gate_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(B, T_dim, HV, K),
        stride=(G_batch_stride, G_token_stride, K, 1),
        assumed_align=32,
    )

    return cute.compile[
        cute.EnableTVMFFI,
        cute.GenerateLineInfo,
        cute.PtxasOptions(f"--maxrregcount {_VTILE_SPEC_D128_MAXRREGCOUNT}"),
    ](
        recurrent_kda_vtile_spec_decode_launch,
        make_fake((B, T_dim, H, K)),  # q
        make_fake((B, T_dim, H, K)),  # k
        make_fake((B, T_dim, HV, V)),  # v
        gate_fake,  # g
        make_fake((B, T_dim, HV)),  # beta
        state_fake,  # state
        make_fake((B, T_dim, HV, V)),  # output
        make_fake((ALog_sym,), dtype=cute.Float32),  # A_log
        make_fake((HK_sym,), dtype=cute.Float32),  # dt_bias
        make_fake((CuSeqlens_sym,), dtype=cute.Int32),  # cu_seqlens
        make_fake((SsiB_sym,), dtype=cute.Int32),  # ssm_state_indices (flattened)
        make_fake((Nat_sym,), dtype=cute.Int32),  # num_accepted_tokens
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        HEAD_DIM,
        USE_QK_L2NORM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
        NUM_TOKENS,
        NUM_V_TILES,
        V_TILE_ROWS,
    )


def run_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = False,
    lower_bound: Optional[float] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_spec_tokens: Optional[int] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Recurrent KDA (Key-Driven Attention) decode kernel.

    This implements the decode phase of KDA linear attention with per-key-dimension
    gating, processing one token at a time and updating the recurrent state in-place.
    The logical recurrent state ``S`` is ``[K, V]`` and is stored transposed as
    ``[V, K]`` in the kernel. Per token:
    ``S = exp(g)[..., None] * S + beta * outer(k, v - k @ S)`` and
    ``o = q @ S``.

    Args:
        q (torch.Tensor):
            Current query of shape ``[B, 1, H, K]``, or ``[1, total_tokens, H, K]``
            when using ``cu_seqlens``. Must be bfloat16.
        k (torch.Tensor):
            Current key of shape ``[B, 1, H, K]``. Must be bfloat16.
        v (torch.Tensor):
            Current value of shape ``[B, 1, HV, V]``. Must be bfloat16.
            GQA is applied when ``HV != H``.
        g (torch.Tensor):
            Per-K-dimension gate of shape ``[B, 1, HV, K]``. Must be bfloat16.
            Log-space if pre-computed, raw input if ``use_gate_in_kernel=True``.
        beta (torch.Tensor):
            Delta-rule learning rate of shape ``[B, 1, HV]``. Must be bfloat16.
            Pre-sigmoided.
        A_log (Optional[torch.Tensor]):
            Log decay parameter of shape ``[H]``. Must be float32.
            Required when ``use_gate_in_kernel=True``.
        dt_bias (Optional[torch.Tensor]):
            Per-head-K decay bias of shape ``[H*K]``. Must be float32.
        scale (Optional[float]):
            Scale factor for queries. If None, defaults to ``1 / sqrt(K)``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[N, HV, V, K]``. Must be bfloat16.
            If None, zero-initialized. Updated in-place. For batched spec decode
            without ``cu_seqlens``, ``N`` is the packed checkpoint-slot count
            ``B * (1 + num_spec_tokens)`` when ``ssm_state_indices`` is omitted.
        output_final_state (bool):
            Whether to return the final state. Default: ``False``.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2 normalization to Q and K. Default: ``True``.
        use_gate_in_kernel (bool):
            Whether to compute the gate inside the kernel from ``A_log`` and ``g``.
            Default: ``False``.
        lower_bound (Optional[float]):
            If set, uses ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
            gate formula instead of softplus. Must be negative.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape ``[N+1]``. Must be int32.
        ssm_state_indices (Optional[torch.Tensor]):
            State cache indices. Shape ``[N]`` int32 for standard decode, or
            ``[N, 1+S]`` int32 for spec decode (``num_spec_tokens`` must also be
            set). The wrapper flattens 2D indices to ``[N*(1+S)]`` before
            passing to the kernel.
        num_spec_tokens (Optional[int]):
            Number of speculative tokens (S). When set, processes 1+S tokens in a single
            fused kernel launch. If ``cu_seqlens`` is provided, requires 2D
            ``ssm_state_indices`` ``[N, 1+S]``. If ``cu_seqlens`` is not provided,
            the wrapper auto-builds the packed format from ``[B, 1+S, ...]`` inputs
            (batched spec-decode shim). Must be >= 1.
        num_accepted_tokens (Optional[torch.Tensor]):
            Per-sequence accepted token count from the previous spec decode round.
            Shape ``[N]`` int32. When provided, the kernel loads initial state from
            ``ssm_state_indices[n, num_accepted_tokens[n] - 1]`` instead of
            ``ssm_state_indices[n, 0]``, matching the FLA Triton kernel's behavior.
            Kernel still writes all T=1+S checkpoint slots. Caller contract:
            ``num_accepted_tokens >= 1`` (the bonus token is always accepted).
            If ``None``, initial state is loaded from ``ssm_state_indices[n, 0]``
            (backward compatible).
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor. Shape ``[B, 1, HV, V]`` for standard
            decode, ``[1, N*(1+S), HV, V]`` for spec decode with
            ``cu_seqlens``, or ``[B, 1+S, HV, V]`` for batched spec decode
            (``num_spec_tokens=S`` without ``cu_seqlens``). Must be
            pre-allocated for CUDA graph capture; auto-allocated if ``None``.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - output: Shape ``[B, 1, HV, V]`` (standard),
              ``[1, N*(1+S), HV, V]`` (spec decode with ``cu_seqlens``), or
              ``[B, 1+S, HV, V]`` (batched spec decode without
              ``cu_seqlens``). Padded sequence positions are zero-filled
              in spec mode.
            - state: Updated state of shape ``[N, HV, V, K]`` if
              ``output_final_state=True``, else ``None``. For batched spec
              decode without ``cu_seqlens``, this is the packed checkpoint
              state pool used by the shim.

    Note:
        - Requires SM100 (Blackwell) architecture
        - State is bfloat16 ``[N, HV, V, K]`` and updated in-place
        - HEAD_DIM (K=V) must be 64 or 128
        - When using ``cu_seqlens``, batch size ``B`` must be 1
        - Spec mode (``num_spec_tokens=S``): ``cu_seqlens`` must step by
          ``1+S`` for every row including padded sequences (which signal via
          ``ssm_state_indices == -1``). Guaranteed by vLLM's
          ``GDNAttentionMetadata``; not validated at runtime (CUDA graph compat).
        - ``KDA_USE_VTILE`` and ``KDA_V_TILE_ROWS`` are internal benchmarking
          overrides for the auto-dispatch heuristic.
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    device = q.device
    if K != V:
        raise ValueError(f"K must equal V, got K={K}, V={V}")
    if K not in (64, 128):
        raise ValueError(f"HEAD_DIM must be 64 or 128, got K={K}")
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16, got {tensor.dtype}")
    if HV < H or HV % H != 0:
        raise ValueError(f"HV must be a positive multiple of H, got H={H}, HV={HV}")

    if use_gate_in_kernel:
        if A_log is None:
            raise ValueError("A_log is required when use_gate_in_kernel=True")
        if A_log.dtype != torch.float32:
            raise TypeError(f"A_log must be float32, got {A_log.dtype}")
    if dt_bias is not None and dt_bias.dtype != torch.float32:
        raise TypeError(f"dt_bias must be float32, got {dt_bias.dtype}")
    if lower_bound is not None:
        if not use_gate_in_kernel:
            raise ValueError("lower_bound requires use_gate_in_kernel=True")
        if lower_bound >= 0.0:
            raise ValueError("lower_bound must be negative")

    if (
        num_spec_tokens is not None
        and cu_seqlens is not None
        and ssm_state_indices is None
    ):
        raise ValueError(
            "ssm_state_indices is required when num_spec_tokens is set with cu_seqlens"
        )

    # Batched spec-decode shim: auto-converts [B,T,...] to packed [1,B*T,...] format.
    _batched_spec_B = None
    if num_spec_tokens is not None and cu_seqlens is None:
        T_spec = 1 + num_spec_tokens
        if T_spec != T:
            raise ValueError(
                f"q.shape[1]={T} must equal 1+num_spec_tokens={T_spec} "
                f"when cu_seqlens is not provided"
            )
        if (
            initial_state is not None
            and ssm_state_indices is None
            and initial_state.shape[0] < B * T_spec
        ):
            raise ValueError(
                "initial_state must have at least B*(1+num_spec_tokens) slots "
                "when batched spec decode auto-generates ssm_state_indices"
            )
        _batched_spec_B = B  # save original B for output reshape
        # Reshape [B, T, ...] -> [1, B*T, ...] -- pure view, no copy
        q = q.reshape(1, B * T_spec, H, K)
        k = k.reshape(1, B * T_spec, H, K)
        v = v.reshape(1, B * T_spec, HV, V)
        g = g.reshape(1, B * T_spec, HV, K)
        beta = beta.reshape(1, B * T_spec, HV)
        # Auto-build cu_seqlens: [0, T, 2T, ..., B*T]
        cu_seqlens = torch.arange(
            0, B * T_spec + 1, step=T_spec, dtype=torch.int32, device=device
        )
        # Auto-build ssm_state_indices [B, T] if not provided
        if ssm_state_indices is None:
            ssm_state_indices = torch.arange(
                B * T_spec, dtype=torch.int32, device=device
            ).reshape(B, T_spec)
        # Output pre-allocation: reshape caller-provided output if shape matches
        if output is not None and output.shape == (B, T_spec, HV, V):
            output = output.reshape(1, B * T_spec, HV, V)
        # Update B, T to reflect packed format
        B, T = 1, B * T_spec

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(f"Batch size must be 1 with cu_seqlens, got B={B}")
        NUM_TOKENS = 1
        if num_spec_tokens is not None:
            if num_spec_tokens <= 0:
                raise ValueError(
                    f"num_spec_tokens must be >= 1, got {num_spec_tokens}. "
                    f"Use num_spec_tokens=None for standard decode."
                )
            NUM_TOKENS = 1 + num_spec_tokens
            N = cu_seqlens.shape[0] - 1
            if ssm_state_indices.ndim != 2 or ssm_state_indices.shape[1] != NUM_TOKENS:
                raise ValueError(
                    f"ssm_state_indices must be 2D [N, {NUM_TOKENS}] in spec mode, "
                    f"got shape {list(ssm_state_indices.shape)}"
                )
            if ssm_state_indices.shape[0] != N:
                raise ValueError(
                    f"ssm_state_indices.shape[0]={ssm_state_indices.shape[0]} must equal "
                    f"N={N} (cu_seqlens.shape[0]-1)"
                )
            if initial_state is not None and initial_state.shape[0] == 0:
                raise ValueError(
                    "initial_state must have at least 1 slot when cu_seqlens is used"
                )
        N = cu_seqlens.shape[0] - 1
        cu_seqlens_i32 = cu_seqlens.to(torch.int32)
        ssi = (
            ssm_state_indices.to(torch.int32).contiguous().view(-1)
            if ssm_state_indices is not None
            else torch.arange(N, dtype=torch.int32, device=device)
        )
        if initial_state is None:
            max_idx = (
                max(0, int(ssi[ssi >= 0].max().item())) + 1 if (ssi >= 0).any() else 1
            )
            state = torch.zeros(max_idx, HV, V, K, device=device, dtype=torch.bfloat16)
        else:
            state = initial_state
        if num_spec_tokens is not None:
            # Spec mode: zero-init output so padded positions are well-defined.
            # Reuse caller buffer when possible to avoid allocation (CUDA graph compat).
            if (
                output is not None
                and output.shape == (1, N * NUM_TOKENS, HV, V)
                and output.dtype == q.dtype
                and output.device == device
            ):
                output.zero_()
                out_buf = output
            else:
                out_buf = torch.zeros(
                    1, N * NUM_TOKENS, HV, V, device=device, dtype=q.dtype
                )
        else:
            if (
                output is not None
                and output.shape == v.shape
                and output.dtype == q.dtype
                and output.device == device
            ):
                output.zero_()
                out_buf = output
            else:
                out_buf = torch.zeros_like(v)
    else:
        if T != 1:
            raise ValueError(
                f"Decode only supports T=1 without cu_seqlens, got T={T}. "
                f"For multi-token decode, use cu_seqlens or num_spec_tokens."
            )
        if initial_state is not None and not initial_state.is_contiguous():
            raise ValueError(
                "non-contiguous initial_state requires cu_seqlens: without cu_seqlens "
                "the wrapper calls .contiguous() which copies the state, so in-place "
                "updates would be silently lost on the original tensor"
            )
        cu_seqlens_i32 = None
        ssi = None
        copy_back_indices = None
        if initial_state is None:
            state = torch.zeros(B, HV, V, K, device=device, dtype=torch.bfloat16)
        elif ssm_state_indices is not None:
            state = initial_state[ssm_state_indices].contiguous()
            copy_back_indices = ssm_state_indices
        else:
            state = initial_state.contiguous()
        if (
            output is not None
            and output.shape == (B, 1, HV, V)
            and output.dtype == q.dtype
            and output.device == device
        ):
            out_buf = output
        else:
            out_buf = torch.empty(B, 1, HV, V, device=device, dtype=q.dtype)

    # Compile kernel (cached by constexpr config)
    USE_QK_NORM = 1 if use_qk_l2norm_in_kernel else 0
    USE_GATE = 1 if use_gate_in_kernel else 0
    HAS_BIAS = 1 if dt_bias is not None else 0
    USE_LB = 1 if lower_bound is not None else 0
    USE_CU = 1 if cu_seqlens_i32 is not None else 0
    if cu_seqlens is None:
        NUM_TOKENS = 1
    grid_seqs = cu_seqlens_i32.shape[0] - 1 if cu_seqlens_i32 is not None else B
    base_grid = grid_seqs * HV
    # Dispatch between the base and v-tiled kernels.
    #   - base:  grid = B*HV, one CTA per (batch, head), SMEM ping-pong over V chunks.
    #            Wins when the grid is large enough to saturate SMs (larger B).
    #   - vtile: grid = B*HV*(HEAD_DIM/32), one CTA per (batch, head, v_tile),
    #            register-carry h, no ping-pong. Wins when the base grid
    #            under-subscribes the GPU (small B) by expanding the grid 2-4x.
    # Threshold mirrors upstream GDN's `_select_tile_v_for_batch` heuristic
    # (flashinfer gdn_kernels/gdn_decode_bf16_state.py): switch to vtile when
    # the base grid is below ~4 waves of SMs. Measured crossover on B200 matches
    # (B=16 break-even, B<=4 vtile 1.5-1.7x, B>=64 base 7-20% faster).
    # KDA_USE_VTILE=1/0 forces a specific path (for A/B tests, benchmarks).
    # A register-resident ILP MTP port was evaluated and rejected in the
    # original kda-cutedsl tuning notes.
    _vtile_env = os.environ.get("KDA_USE_VTILE")
    if _vtile_env == "1":
        use_vtile = True
    elif _vtile_env == "0":
        use_vtile = False
    else:
        use_vtile = base_grid < 4 * _get_num_sms(device)

    # V-row tile size inside vtile: 32 (standard) or 16 (finer-grained).
    # V_TILE_ROWS=16 doubles the vtile grid at D=128 (see the parameterized
    # load/store helpers + guards in _process_v_chunk). Measured on B200 at
    # H32-D128-lb B=1 (2026-04-17): grid 128->256, waves/SM 0.12->0.25, SM
    # Busy 3%->6.3%, but wall-time +3% (kernel stays latency-bound per CTA
    # and the 2nd block per SM doesn't fully overlap). Not enabled by
    # default; KDA_V_TILE_ROWS=16 forces it for A/B experiments.
    _v_tile_env = os.environ.get("KDA_V_TILE_ROWS")
    if _v_tile_env == "16":
        V_TILE_ROWS = 16
    else:
        V_TILE_ROWS = 32

    if use_vtile:
        if K == 128 and NUM_TOKENS > 1:
            compiled = _get_compiled_vtile_spec_decode_d128_kernel(
                USE_QK_NORM,
                USE_GATE,
                HAS_BIAS,
                USE_LB,
                USE_CU,
                NUM_TOKENS,
                V_TILE_ROWS,
            )
        else:
            compiled = _get_compiled_vtile_kernel(
                K,
                USE_QK_NORM,
                USE_GATE,
                HAS_BIAS,
                USE_LB,
                USE_CU,
                NUM_TOKENS,
                V_TILE_ROWS,
            )
    else:
        use_chunk_major = K == 128 and NUM_TOKENS == 4 and base_grid >= 2048
        compiled = _get_compiled_kernel(
            K,
            USE_QK_NORM,
            USE_GATE,
            HAS_BIAS,
            USE_LB,
            USE_CU,
            NUM_TOKENS,
            int(use_chunk_major),
        )

    # Dummy tensors for unused optional args (TVM FFI requires all args present)
    global _dummy_cache
    if device not in _dummy_cache:
        _dummy_cache[device] = {
            "f32_1": torch.zeros(1, device=device, dtype=torch.float32),
            "i32_1": torch.zeros(1, device=device, dtype=torch.int32),
        }
    dc = _dummy_cache[device]

    if num_accepted_tokens is not None:
        num_accepted_tokens_i32 = (
            num_accepted_tokens
            if num_accepted_tokens.dtype == torch.int32
            else num_accepted_tokens.to(torch.int32)
        )
    elif cu_seqlens_i32 is not None and NUM_TOKENS > 1:
        nat_key = f"i32_ones_{cu_seqlens_i32.shape[0] - 1}"
        if nat_key not in dc:
            dc[nat_key] = torch.ones(
                cu_seqlens_i32.shape[0] - 1, device=device, dtype=torch.int32
            )
        num_accepted_tokens_i32 = dc[nat_key]
    else:
        num_accepted_tokens_i32 = dc["i32_1"]

    compiled(
        q,
        k,
        v,
        g,
        beta,
        state,
        out_buf,
        A_log if A_log is not None else dc["f32_1"],
        dt_bias if dt_bias is not None else dc["f32_1"],
        cu_seqlens_i32 if cu_seqlens_i32 is not None else dc["i32_1"],
        ssi if ssi is not None else dc["i32_1"],
        num_accepted_tokens_i32,
        scale if scale is not None else 1.0 / math.sqrt(K),
        1e-6,
        lower_bound if lower_bound is not None else 0.0,
    )

    if cu_seqlens_i32 is None and copy_back_indices is not None:
        initial_state[copy_back_indices] = state

    # Reshape output back to [B, T, HV, V] for batched spec decode
    if _batched_spec_B is not None:
        T_spec = 1 + num_spec_tokens
        out_buf = out_buf.reshape(_batched_spec_B, T_spec, HV, V)

    return (out_buf, state if output_final_state else None)
