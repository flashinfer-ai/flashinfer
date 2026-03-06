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

Recurrent KDA (Key-Driven Attention) decode kernel using CuTe DSL for SM100.

Single-token (T=1) recurrent linear attention with per-key-dimension gating.
State S[V,K] is updated via: S = diag(g_exp) @ S + beta * k * (v - S^T k)
Output: o = S^T q

Inputs:  q,k [B,1,H,K]  v,g [B,1,HV,K]  beta [B,1,HV]  state [B,HV,V,K] bf16
Output:  o [B,1,HV,V]   state modified in-place

Supports GQA (H != HV), cu_seqlens for variable-length batches, and
compile-time gate modes (pre-computed, softplus, lower_bound * sigmoid).
"""

import functools
import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass import utils
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math
from cutlass._mlir.dialects import nvvm
import tvm_ffi  # noqa: F401 — TVM FFI required for zero-overhead kernel dispatch

# ==============================================================================
# CONSTANTS
# ==============================================================================
# SMEM H padding for bank conflict avoidance.
# H_SMEM_STRIDE = HEAD_DIM + H_SMEM_PADDING (computed inside kernels from HEAD_DIM).
# Constraint: (stride * 2) must have ≥16 as highest power-of-2 factor for cp.async 128-bit
# alignment.
#   HEAD_DIM=128: stride=136 → 272 bytes → 272 = 16*17 → align<16> ✓
#   HEAD_DIM=64:  stride=72  → 144 bytes → 144 = 16*9  → align<16> ✓
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
    h_sh_chunk, h_out, tidx, v_row_offset, HEAD_DIM: cutlass.Constexpr[int]
):
    """Store H from SMEM to GMEM using 128-bit stores."""
    copy_bits = 128
    copy_elems = copy_bits // cutlass.BFloat16.width

    from cutlass.cute.nvgpu import CopyUniversalOp

    if HEAD_DIM == 64:
        # 64 threads: use (8, 8) thread layout, (8, 64) tiles, 4 row iterations
        thr_layout = cute.make_layout((8, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_store = cute.make_copy_atom(
            CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=copy_bits
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_store, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(4):
            s_tile = cute.local_tile(h_sh_chunk, (8, 64), (row_iter, 0))
            g_tile = cute.local_tile(
                h_out, (8, 64), (row_iter + (v_row_offset // 8), 0)
            )
            tS = thr_copy.partition_S(s_tile)
            tD = thr_copy.partition_D(g_tile)
            cute.copy(atom_store, tS, tD)
    elif HEAD_DIM == 128:
        # 128 threads: use (16, 8) thread layout, (16, 64) tiles, 2×2 iterations
        thr_layout = cute.make_layout((16, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, copy_elems))
        atom_store = cute.make_copy_atom(
            CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=copy_bits
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_store, thr_layout, val_layout)
        thr_copy = tiled_copy.get_slice(tidx)
        for row_iter in cutlass.range_constexpr(2):
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
    h_sh_chunk, h_global, tidx, row_offset, HEAD_DIM: cutlass.Constexpr[int]
):
    """Load H chunk from GMEM to SMEM using async copy."""
    copy_bits = 128
    copy_elems = copy_bits // cutlass.BFloat16.width

    if HEAD_DIM == 64:
        # 64 threads: use (8, 8) thread layout, (8, 64) tiles, 4 row iterations
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
        for row_iter in cutlass.range_constexpr(4):
            g_tile = cute.local_tile(
                h_global, (8, 64), (row_iter + (row_offset // 8), 0)
            )
            s_tile = cute.local_tile(h_sh_chunk, (8, 64), (row_iter, 0))
            tS = thr_copy.partition_S(g_tile)
            tD = thr_copy.partition_D(s_tile)
            cute.copy(atom_async_copy, tS, tD)
    elif HEAD_DIM == 128:
        # 128 threads: use (16, 8) thread layout, (16, 64) tiles, 2×2 iterations
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
        for row_iter in cutlass.range_constexpr(2):
            for col_iter in cutlass.range_constexpr(2):
                g_tile = cute.local_tile(
                    h_global, (16, 64), (row_iter + (row_offset // 16), col_iter)
                )
                s_tile = cute.local_tile(h_sh_chunk, (16, 64), (row_iter, col_iter))
                tS = thr_copy.partition_S(g_tile)
                tD = thr_copy.partition_D(s_tile)
                cute.copy(atom_async_copy, tS, tD)


@cute.jit
def compute_gate_exp_chunk(
    g_exp_chunk,
    g_head,
    k_base,
    A_log_val,
    gDtBias,
    h_K_offset,
    lower_bound_val,
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
):
    """Load gate from global memory and compute exp(gate).

    When USE_GATE_IN_KERNEL=0: g is pre-computed log-space gate, just exp it.
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=0:
        g_log = -exp(A_log) * softplus(g + dt_bias); return exp(g_log)
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=1:
        g_log = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)); return exp(g_log)

    Args:
        g_exp_chunk: output register tensor (32,) Float32
        g_head: global gate tensor slice [K=HEAD_DIM]
        k_base: starting K index for this warp (warp_idx * 32)
        A_log_val: exp(A_log) for this head (Float32), precomputed outside
        gDtBias: dt_bias tensor [H*K] Float32 (or dummy if HAS_DT_BIAS=0)
        h_K_offset: query_head_idx * K offset into dt_bias
        lower_bound_val: lower bound float (negative, e.g. -5.0)
        USE_GATE_IN_KERNEL: 0 = pre-computed gate, 1 = in-kernel gate
        HAS_DT_BIAS: 0 = no dt_bias, 1 = add dt_bias
        USE_LOWER_BOUND: 0 = softplus formula, 1 = lower_bound * sigmoid formula
    """
    for i in cutlass.range_constexpr(0, 32, 2):
        g0 = g_head[k_base + i].to(cutlass.Float32)
        g1 = g_head[k_base + i + 1].to(cutlass.Float32)
        if USE_GATE_IN_KERNEL == 1:
            if HAS_DT_BIAS == 1:
                g0 = g0 + gDtBias[h_K_offset + k_base + i].to(cutlass.Float32)
                g1 = g1 + gDtBias[h_K_offset + k_base + i + 1].to(cutlass.Float32)
            if USE_LOWER_BOUND == 1:
                # Fused exp(L * sigmoid(A*g)) using exp2 to eliminate LOG2_E multiplies:
                # exp(-A*g) = exp2(-A*LOG2_E * g), exp(L*sig) = exp2(L*LOG2_E * sig)
                LOG2_E = cutlass.Float32(1.4426950408889634)
                neg_A_log2e = -A_log_val * LOG2_E
                lb_log2e = lower_bound_val * LOG2_E
                one = cutlass.Float32(1.0)
                fm = mlir_arith.FastMathFlags.fast
                # Inline sigmoid + outer exp as exp2 chain (all MLIR for fastmath)
                one_ir = one.ir_value()
                neg_A_ir = neg_A_log2e.ir_value()
                lb_ir = lb_log2e.ir_value()
                # exp2(-A*LOG2_E * g) = exp(-A*g)
                ag0 = mlir_arith.mulf(neg_A_ir, g0.ir_value(), fastmath=fm)
                ag1 = mlir_arith.mulf(neg_A_ir, g1.ir_value(), fastmath=fm)
                exp0 = mlir_math.exp2(ag0, fastmath=fm)
                exp1 = mlir_math.exp2(ag1, fastmath=fm)
                # sigmoid = 1 / (1 + exp(-A*g))
                denom0 = mlir_arith.addf(one_ir, exp0, fastmath=fm)
                denom1 = mlir_arith.addf(one_ir, exp1, fastmath=fm)
                sig0 = mlir_arith.divf(one_ir, denom0, fastmath=fm)
                sig1 = mlir_arith.divf(one_ir, denom1, fastmath=fm)
                # exp2(L*LOG2_E * sigmoid) = exp(L * sigmoid)
                ls0 = mlir_arith.mulf(lb_ir, sig0, fastmath=fm)
                ls1 = mlir_arith.mulf(lb_ir, sig1, fastmath=fm)
                g0 = mlir_math.exp2(ls0, fastmath=fm)
                g1 = mlir_math.exp2(ls1, fastmath=fm)
            else:
                # Fused softplus + exp using log2/exp2 to save 2 PTX ops per element:
                # exp(-A * log(1+exp(g))) = exp2(-A * log2(1+exp(g)))
                # (log2/exp2 pair cancels base conversion, no LN2/LOG2_E needed)
                exp_g0 = cute.exp(g0, fastmath=True)
                exp_g1 = cute.exp(g1, fastmath=True)
                one = cutlass.Float32(1.0)
                log2_0 = cute.log2(one + exp_g0, fastmath=True)
                log2_1 = cute.log2(one + exp_g1, fastmath=True)
                g0 = cute.exp2(-A_log_val * log2_0, fastmath=True)
                g1 = cute.exp2(-A_log_val * log2_1, fastmath=True)
        else:
            g0 = cute.exp(g0, fastmath=True)
            g1 = cute.exp(g1, fastmath=True)
        g_exp_chunk[i] = g0
        g_exp_chunk[i + 1] = g1


@cute.jit
def normalize_and_store_qk_to_smem(
    q_head, k_head, q_sh, k_sh, lane_idx, scale, eps, HEAD_DIM: cutlass.Constexpr[int]
):
    """L2-normalize Q and K vectors, then store to shared memory."""
    # ELEMS_PER_LANE = HEAD_DIM // 32 (2 for HD=64, 4 for HD=128)
    q_reg = cute.make_rmem_tensor((HEAD_DIM // 32,), cutlass.Float32)
    k_reg = cute.make_rmem_tensor((HEAD_DIM // 32,), cutlass.Float32)

    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        q_reg[i] = q_head[lane_idx + i * 32].to(cutlass.Float32)
        k_reg[i] = k_head[lane_idx + i * 32].to(cutlass.Float32)

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


@cute.jit
def load_v_to_smem(v_head, v_sh, tidx):
    """Load V values from GMEM to SMEM."""
    v_sh[tidx] = v_head[tidx].to(cutlass.Float32)


@cute.jit
def load_kq_chunk_from_smem(kq_sh, kq_chunk, k_base):
    """Load K or Q chunk from SMEM to registers."""
    for i in cutlass.range_constexpr(32):
        kq_chunk[i] = kq_sh[k_base + i]


@cute.jit
def decay_h_from_smem_and_compute_pred(
    h_sh_chunk, h_chunk, kq_chunk, g_exp_chunk, lane_idx, k_base
):
    """Load H from SMEM, apply decay, and compute pred = sum_k(h * k)."""
    pred = cutlass.Float32(0.0)
    pred2 = cutlass.Float32(0.0)

    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(
                h_sh_chunk[lane_idx, k_base + i].to(cutlass.Float32),
                h_sh_chunk[lane_idx, k_base + i + 1].to(cutlass.Float32),
            ),
            src_b=(g_exp_chunk[i], g_exp_chunk[i + 1]),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )

    for i in cutlass.range_constexpr(0, 32, 2):
        pred, pred2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(kq_chunk[i], kq_chunk[i + 1]),
            src_c=(pred, pred2),
        )

    pred = pred + pred2
    return pred


@cute.jit
def update_h_with_delta(h_chunk, kq_chunk, v_delta):
    """Update H with delta: h = h + k * v_delta."""
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(kq_chunk[i], kq_chunk[i + 1]),
            src_b=(v_delta, v_delta),
            src_c=(h_chunk[i], h_chunk[i + 1]),
        )


@cute.jit
def compute_output(h_chunk, kq_chunk):
    """Compute output = sum_k(h * q)."""
    out = cutlass.Float32(0.0)
    out2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        out, out2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(kq_chunk[i], kq_chunk[i + 1]),
            src_c=(out, out2),
        )
    out = out + out2
    return out


@cute.jit
def decay_h_in_place(h_chunk, g_exp_chunk):
    """Apply decay to H in place: h = h * g_exp."""
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(g_exp_chunk[i], g_exp_chunk[i + 1]),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )


@cute.jit
def cross_warp_reduce_single(
    reduce_sh, slot, warp_idx, lane_idx, value, NUM_WARPS: cutlass.Constexpr[int]
):
    """
    Cross-warp reduction for a single value using bank-conflict-free layout.
    Layout: [slot, lane_idx, warp_idx]
    """
    reduce_sh[slot, lane_idx, warp_idx] = value
    cute.arch.sync_threads()
    reduced_value = cutlass.Float32(0.0)
    if NUM_WARPS == 2:
        reduced_value = reduce_sh[slot, lane_idx, 0] + reduce_sh[slot, lane_idx, 1]
    elif NUM_WARPS == 4:
        reduced_value = (
            reduce_sh[slot, lane_idx, 0]
            + reduce_sh[slot, lane_idx, 1]
            + reduce_sh[slot, lane_idx, 2]
            + reduce_sh[slot, lane_idx, 3]
        )
    return reduced_value


@cute.jit
def cross_warp_reduce_two(
    reduce_sh,
    slot1,
    slot2,
    warp_idx,
    lane_idx,
    value1,
    value2,
    NUM_WARPS: cutlass.Constexpr[int],
):
    """
    Cross-warp reduction for two values simultaneously using bank-conflict-free layout.
    Layout: [slot, lane_idx, warp_idx]
    """
    reduce_sh[slot1, lane_idx, warp_idx] = value1
    reduce_sh[slot2, lane_idx, warp_idx] = value2
    cute.arch.sync_threads()
    reduced1 = cutlass.Float32(0.0)
    reduced2 = cutlass.Float32(0.0)
    if NUM_WARPS == 2:
        reduced1 = reduce_sh[slot1, lane_idx, 0] + reduce_sh[slot1, lane_idx, 1]
        reduced2 = reduce_sh[slot2, lane_idx, 0] + reduce_sh[slot2, lane_idx, 1]
    elif NUM_WARPS == 4:
        reduced1 = (
            reduce_sh[slot1, lane_idx, 0]
            + reduce_sh[slot1, lane_idx, 1]
            + reduce_sh[slot1, lane_idx, 2]
            + reduce_sh[slot1, lane_idx, 3]
        )
        reduced2 = (
            reduce_sh[slot2, lane_idx, 0]
            + reduce_sh[slot2, lane_idx, 1]
            + reduce_sh[slot2, lane_idx, 2]
            + reduce_sh[slot2, lane_idx, 3]
        )
    return reduced1, reduced2


# ==============================================================================
# SEQLEN=1 KERNEL (Persistent K Optimization)
# ==============================================================================


@cute.kernel
def recurrent_kda_decode_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gG: cute.Tensor,  # [B, T, HV, K] log-space gate (or raw input if USE_GATE_IN_KERNEL)
    gBeta: cute.Tensor,  # [B, T, HV] pre-sigmoided
    gH: cute.Tensor,  # state: bf16 [B,HV,V,K] (modified in-place)
    gO: cute.Tensor,
    gALog: cute.Tensor,  # [H] float32 (A_log per query head)
    gDtBias: cute.Tensor,  # [H*K] float32 (dt_bias per head and K)
    gCuSeqlens: cute.Tensor,  # [N+1] int32 — raw cu_seqlens
    gSsmStateIndices: cute.Tensor,  # [N] int32 — raw ssm_state_indices
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
):
    """T=1 decode kernel. One CTA per (batch, head) pair.

    Thread mapping: HEAD_DIM threads = (HEAD_DIM // 32) warps x 32 lanes.
    State [V,K] tiled into V-chunks of 32 rows (2 for HD=64, 4 for HD=128).
    K held persistently in registers across V-chunks.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    HV = cutlass.Int32(gV.shape[2])
    H = cutlass.Int32(gQ.shape[2])

    batch_idx = bidx // HV
    value_head_idx = bidx % HV
    query_head_idx = value_head_idx // (HV // H)

    # cu_seqlens offset computation (zero cost when USE_CU_SEQLENS=0)
    token_offset = cutlass.Int32(0)
    seq_idx = batch_idx
    if USE_CU_SEQLENS == 1:
        token_offset = gCuSeqlens[batch_idx].to(cutlass.Int32)
        seq_idx = gSsmStateIndices[batch_idx].to(cutlass.Int32)

    # Precompute gate params (guarded by Constexpr, zero cost when unused)
    A_log_val = cutlass.Float32(0.0)
    h_K_offset = cutlass.Int32(0)
    lower_bound_val = lower_bound
    if USE_GATE_IN_KERNEL == 1:
        A_log_val = cute.exp(gALog[query_head_idx].to(cutlass.Float32), fastmath=True)
        h_K_offset = query_head_idx * HEAD_DIM

    smem = utils.SmemAllocator()

    # Load gate and beta from global memory
    g_head = gG[(batch_idx, 0, value_head_idx, None)]  # [K=HEAD_DIM]
    beta = gBeta[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)
    if USE_CU_SEQLENS == 1:
        g_head = gG[(0, token_offset, value_head_idx, None)]
        beta = gBeta[(0, token_offset, value_head_idx)].to(cutlass.Float32)

    # Allocate SMEM — always 4 H chunk buffers for simplicity
    # (unused ones waste ~4.5KB for HEAD_DIM=64, trivial vs SM100's 228KB)
    h_sh_chunk0 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_chunk1 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_chunk2 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )
    h_sh_chunk3 = smem.allocate_tensor(
        cutlass.BFloat16,
        cute.make_layout((32, HEAD_DIM), stride=(HEAD_DIM + H_SMEM_PADDING, 1)),
        byte_alignment=128,
    )

    q_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    k_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)

    pred_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((HEAD_DIM // 32, 32))
    )
    out_sh = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((HEAD_DIM // 32, 32))
    )

    # Derive state view — bf16 [V,K]
    h_global = gH[(batch_idx, value_head_idx, None, None)]
    if USE_CU_SEQLENS == 1:
        h_global = gH[(seq_idx, value_head_idx, None, None)]

    warp_idx = tidx // 32
    lane_idx = tidx % 32

    # Load first 2 state chunks (async bf16 copy)
    load_h_chunk_async(h_sh_chunk0, h_global, tidx, 0, HEAD_DIM)
    nvvm.cp_async_commit_group()
    load_h_chunk_async(h_sh_chunk1, h_global, tidx, 32, HEAD_DIM)
    nvvm.cp_async_commit_group()

    # L2 normalization
    q_head = gQ[(batch_idx, 0, query_head_idx, None)]
    k_head = gK[(batch_idx, 0, query_head_idx, None)]
    if USE_CU_SEQLENS == 1:
        q_head = gQ[(0, token_offset, query_head_idx, None)]
        k_head = gK[(0, token_offset, query_head_idx, None)]

    # Use shared helper for Q/K normalization (only warp 0 does the work)
    if warp_idx == 0:
        normalize_and_store_qk_to_smem(
            q_head, k_head, q_sh, k_sh, lane_idx, scale, eps, HEAD_DIM
        )

    cute.arch.sync_threads()

    # Load V
    v_head = gV[(batch_idx, 0, value_head_idx, None)]
    if USE_CU_SEQLENS == 1:
        v_head = gV[(0, token_offset, value_head_idx, None)]
    v_sh = smem.allocate_tensor(cutlass.Float32, HEAD_DIM)
    v_sh[tidx] = v_head[tidx].to(cutlass.Float32)

    # Registers: h_chunk + k_chunk (persistent) + qk_temp (reused for Q)
    h_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)
    k_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)  # PERSISTENT K!
    qk_temp = cute.make_rmem_tensor((32,), cutlass.Float32)

    # Transient per-K gate array (recomputed per V-chunk)
    g_exp_chunk = cute.make_rmem_tensor((32,), cutlass.Float32)

    k_base = warp_idx * 32

    # Load K ONCE - keep for entire kernel
    for i in cutlass.range_constexpr(32):
        k_chunk[i] = k_sh[k_base + i]

    h_out = gH[(batch_idx, value_head_idx, None, None)]
    o_head = gO[(batch_idx, 0, value_head_idx, None)]
    if USE_CU_SEQLENS == 1:
        h_out = gH[(seq_idx, value_head_idx, None, None)]
        o_head = gO[(0, token_offset, value_head_idx, None)]

    # ========================================================================
    # CHUNK 0
    # ========================================================================
    nvvm.cp_async_wait_group(1)
    cute.arch.sync_threads()

    # Load/compute per-K gate (same for all V-chunks since T=1)
    compute_gate_exp_chunk(
        g_exp_chunk,
        g_head,
        k_base,
        A_log_val,
        gDtBias,
        h_K_offset,
        lower_bound_val,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
    )

    pred = cutlass.Float32(0.0)
    pred2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(
                h_sh_chunk0[lane_idx, k_base + i].to(cutlass.Float32),
                h_sh_chunk0[lane_idx, k_base + i + 1].to(cutlass.Float32),
            ),
            src_b=(g_exp_chunk[i], g_exp_chunk[i + 1]),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )
    for i in cutlass.range_constexpr(0, 32, 2):
        pred, pred2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(k_chunk[i], k_chunk[i + 1]),
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

    v_val = (v_sh[lane_idx] - pred_final) * beta

    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(k_chunk[i], k_chunk[i + 1]),
            src_b=(v_val, v_val),
            src_c=(h_chunk[i], h_chunk[i + 1]),
        )

    # Load Q for output computation
    for i in cutlass.range_constexpr(32):
        qk_temp[i] = q_sh[k_base + i]

    out = cutlass.Float32(0.0)
    out2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        out, out2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(qk_temp[i], qk_temp[i + 1]),
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

    write_h_chunk_to_smem(h_chunk, h_sh_chunk0, lane_idx, k_base)
    if warp_idx == 0:
        o_head[lane_idx] = out_final.to(cutlass.BFloat16)

    # ========================================================================
    # CHUNK 1
    # ========================================================================
    nvvm.cp_async_wait_group(0)
    cute.arch.sync_threads()

    if HEAD_DIM == 128:
        load_h_chunk_async(h_sh_chunk2, h_global, tidx, 64, HEAD_DIM)
        nvvm.cp_async_commit_group()
        load_h_chunk_async(h_sh_chunk3, h_global, tidx, 96, HEAD_DIM)
        nvvm.cp_async_commit_group()

    store_h_smem_to_gmem(h_sh_chunk0, h_out, tidx, 0, HEAD_DIM)

    pred = cutlass.Float32(0.0)
    pred2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(
                h_sh_chunk1[lane_idx, k_base + i].to(cutlass.Float32),
                h_sh_chunk1[lane_idx, k_base + i + 1].to(cutlass.Float32),
            ),
            src_b=(g_exp_chunk[i], g_exp_chunk[i + 1]),
            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
        )
    for i in cutlass.range_constexpr(0, 32, 2):
        pred, pred2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(k_chunk[i], k_chunk[i + 1]),
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

    v_val = (v_sh[32 + lane_idx] - pred_final) * beta

    for i in cutlass.range_constexpr(0, 32, 2):
        h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
            src_a=(k_chunk[i], k_chunk[i + 1]),
            src_b=(v_val, v_val),
            src_c=(h_chunk[i], h_chunk[i + 1]),
        )

    for i in cutlass.range_constexpr(32):
        qk_temp[i] = q_sh[k_base + i]

    out = cutlass.Float32(0.0)
    out2 = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(0, 32, 2):
        out, out2 = cute.arch.fma_packed_f32x2(
            src_a=(h_chunk[i], h_chunk[i + 1]),
            src_b=(qk_temp[i], qk_temp[i + 1]),
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

    write_h_chunk_to_smem(h_chunk, h_sh_chunk1, lane_idx, k_base)
    if warp_idx == 0:
        o_head[32 + lane_idx] = out_final.to(cutlass.BFloat16)

    # For HEAD_DIM=64: done after 2 chunks. Store chunk1 H and return.
    if HEAD_DIM == 64:
        cute.arch.sync_threads()
        store_h_smem_to_gmem(h_sh_chunk1, h_out, tidx, 32, HEAD_DIM)

    # ========================================================================
    # CHUNK 2 (HEAD_DIM=128 only)
    # ========================================================================
    if HEAD_DIM == 128:
        nvvm.cp_async_wait_group(1)
        cute.arch.sync_threads()

        store_h_smem_to_gmem(h_sh_chunk1, h_out, tidx, 32, HEAD_DIM)

        pred = cutlass.Float32(0.0)
        pred2 = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(0, 32, 2):
            h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
                src_a=(
                    h_sh_chunk2[lane_idx, k_base + i].to(cutlass.Float32),
                    h_sh_chunk2[lane_idx, k_base + i + 1].to(cutlass.Float32),
                ),
                src_b=(g_exp_chunk[i], g_exp_chunk[i + 1]),
                src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
            )
        for i in cutlass.range_constexpr(0, 32, 2):
            pred, pred2 = cute.arch.fma_packed_f32x2(
                src_a=(h_chunk[i], h_chunk[i + 1]),
                src_b=(k_chunk[i], k_chunk[i + 1]),
                src_c=(pred, pred2),
            )
        pred = pred + pred2

        pred_sh[warp_idx, lane_idx] = pred
        cute.arch.sync_threads()
        pred_final = (
            pred_sh[0, lane_idx]
            + pred_sh[1, lane_idx]
            + pred_sh[2, lane_idx]
            + pred_sh[3, lane_idx]
        )

        v_val = (v_sh[64 + lane_idx] - pred_final) * beta

        for i in cutlass.range_constexpr(0, 32, 2):
            h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
                src_a=(k_chunk[i], k_chunk[i + 1]),
                src_b=(v_val, v_val),
                src_c=(h_chunk[i], h_chunk[i + 1]),
            )

        for i in cutlass.range_constexpr(32):
            qk_temp[i] = q_sh[k_base + i]

        out = cutlass.Float32(0.0)
        out2 = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(0, 32, 2):
            out, out2 = cute.arch.fma_packed_f32x2(
                src_a=(h_chunk[i], h_chunk[i + 1]),
                src_b=(qk_temp[i], qk_temp[i + 1]),
                src_c=(out, out2),
            )
        out = out + out2

        out_sh[warp_idx, lane_idx] = out
        cute.arch.sync_threads()
        out_final = (
            out_sh[0, lane_idx]
            + out_sh[1, lane_idx]
            + out_sh[2, lane_idx]
            + out_sh[3, lane_idx]
        )

        write_h_chunk_to_smem(h_chunk, h_sh_chunk2, lane_idx, k_base)
        if warp_idx == 0:
            o_head[64 + lane_idx] = out_final.to(cutlass.BFloat16)

        # ====================================================================
        # CHUNK 3 (HEAD_DIM=128 only)
        # ====================================================================
        nvvm.cp_async_wait_group(0)
        cute.arch.sync_threads()

        store_h_smem_to_gmem(h_sh_chunk2, h_out, tidx, 64, HEAD_DIM)

        pred = cutlass.Float32(0.0)
        pred2 = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(0, 32, 2):
            h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
                src_a=(
                    h_sh_chunk3[lane_idx, k_base + i].to(cutlass.Float32),
                    h_sh_chunk3[lane_idx, k_base + i + 1].to(cutlass.Float32),
                ),
                src_b=(g_exp_chunk[i], g_exp_chunk[i + 1]),
                src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
            )
        for i in cutlass.range_constexpr(0, 32, 2):
            pred, pred2 = cute.arch.fma_packed_f32x2(
                src_a=(h_chunk[i], h_chunk[i + 1]),
                src_b=(k_chunk[i], k_chunk[i + 1]),
                src_c=(pred, pred2),
            )
        pred = pred + pred2

        pred_sh[warp_idx, lane_idx] = pred
        cute.arch.sync_threads()
        pred_final = (
            pred_sh[0, lane_idx]
            + pred_sh[1, lane_idx]
            + pred_sh[2, lane_idx]
            + pred_sh[3, lane_idx]
        )

        v_val = (v_sh[96 + lane_idx] - pred_final) * beta

        for i in cutlass.range_constexpr(0, 32, 2):
            h_chunk[i], h_chunk[i + 1] = cute.arch.fma_packed_f32x2(
                src_a=(k_chunk[i], k_chunk[i + 1]),
                src_b=(v_val, v_val),
                src_c=(h_chunk[i], h_chunk[i + 1]),
            )

        for i in cutlass.range_constexpr(32):
            qk_temp[i] = q_sh[k_base + i]

        out = cutlass.Float32(0.0)
        out2 = cutlass.Float32(0.0)
        for i in cutlass.range_constexpr(0, 32, 2):
            out, out2 = cute.arch.fma_packed_f32x2(
                src_a=(h_chunk[i], h_chunk[i + 1]),
                src_b=(qk_temp[i], qk_temp[i + 1]),
                src_c=(out, out2),
            )
        out = out + out2

        out_sh[warp_idx, lane_idx] = out
        cute.arch.sync_threads()
        out_final = (
            out_sh[0, lane_idx]
            + out_sh[1, lane_idx]
            + out_sh[2, lane_idx]
            + out_sh[3, lane_idx]
        )

        write_h_chunk_to_smem(h_chunk, h_sh_chunk3, lane_idx, k_base)
        if warp_idx == 0:
            o_head[96 + lane_idx] = out_final.to(cutlass.BFloat16)

        cute.arch.sync_threads()
        store_h_smem_to_gmem(h_sh_chunk3, h_out, tidx, 96, HEAD_DIM)


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
    scale: cutlass.Float32,
    eps: cutlass.Float32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
    HEAD_DIM: cutlass.Constexpr[int],
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
    USE_CU_SEQLENS: cutlass.Constexpr[int],
):
    batch_size = mQ.shape[0]
    if USE_CU_SEQLENS == 1:
        batch_size = mCuSeqlens.shape[0] - 1
    HV = mV.shape[2]

    recurrent_kda_decode_kernel(
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
        scale,
        eps,
        lower_bound,
        HEAD_DIM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
    ).launch(
        grid=[batch_size * HV, 1, 1],
        block=[HEAD_DIM, 1, 1],
        stream=stream,
    )


# ==============================================================================
# PUBLIC API
# ==============================================================================

_dummy_cache = {}  # device -> dict of pre-allocated dummy tensors


@functools.cache
def _get_compiled_kernel(
    HEAD_DIM, USE_GATE_IN_KERNEL, HAS_DT_BIAS, USE_LOWER_BOUND, USE_CU_SEQLENS
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
    CuSeqlens_sym, SsiB_sym = cute.sym_int(), cute.sym_int()

    return cute.compile(
        recurrent_kda_launch,
        make_fake((B, T_dim, H, K)),  # q
        make_fake((B, T_dim, H, K)),  # k
        make_fake((B, T_dim, HV, V)),  # v
        make_fake((B, T_dim, HV, K)),  # g
        make_fake((B, T_dim, HV)),  # beta
        make_fake((N, HV_state, V, K)),  # state
        make_fake((B, T_dim, HV, V)),  # output
        make_fake((ALog_sym,), dtype=cute.Float32),  # A_log
        make_fake((HK_sym,), dtype=cute.Float32),  # dt_bias
        make_fake((CuSeqlens_sym,), dtype=cute.Int32),  # cu_seqlens
        make_fake((SsiB_sym,), dtype=cute.Int32),  # ssm_state_indices
        cutlass.Float32(0.0),  # scale
        cutlass.Float32(0.0),  # eps
        cutlass.Float32(0.0),  # lower_bound
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        HEAD_DIM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
        options="--enable-tvm-ffi --generate-line-info",
    )


def recurrent_kda(
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
    output: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Recurrent KDA (Key-Driven Attention) decode kernel.

    This implements the decode phase of KDA linear attention with per-key-dimension
    gating, processing one token at a time and updating the recurrent state in-place.
    State update: ``S = diag(exp(g)) @ S + beta * k * (v - S^T k)``

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
            If None, zero-initialized. Updated in-place.
        output_final_state (bool):
            Whether to return the final state. Default: ``False``.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2 normalization to Q and K. Default: ``True``.
        use_gate_in_kernel (bool):
            Whether to compute the gate inside the kernel from ``A_log`` and ``g``.
            Default: ``False``.
        lower_bound (Optional[float]):
            If set, uses ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``
            gate formula instead of softplus.
        cu_seqlens (Optional[torch.Tensor]):
            Cumulative sequence lengths of shape ``[N+1]``. Must be int32.
        ssm_state_indices (Optional[torch.Tensor]):
            State cache indices of shape ``[N]``. Must be int32.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[B, 1, HV, V]``.
            If None, will be allocated automatically.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - output: Output tensor of shape ``[B, 1, HV, V]``
            - state: Updated state of shape ``[N, HV, V, K]`` if
              ``output_final_state=True``, else ``None``

    Note:
        - Requires SM100 (Blackwell) architecture
        - State is bfloat16 ``[N, HV, V, K]`` and updated in-place
        - HEAD_DIM (K=V) must be 64 or 128
        - When using ``cu_seqlens``, batch size ``B`` must be 1
    """
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    device = q.device
    assert K == V, f"K must equal V, got K={K}, V={V}"
    assert K in (64, 128), f"HEAD_DIM must be 64 or 128, got K={K}"
    assert q.dtype == torch.bfloat16, f"q must be bfloat16, got {q.dtype}"
    assert HV >= H and HV % H == 0, (
        f"HV must be a positive multiple of H, got H={H}, HV={HV}"
    )

    if use_gate_in_kernel:
        assert A_log is not None, "A_log is required when use_gate_in_kernel=True"
        assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"
    if lower_bound is not None:
        assert use_gate_in_kernel, "lower_bound requires use_gate_in_kernel=True"

    # Prepare state and cu_seqlens
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(f"Batch size must be 1 with cu_seqlens, got B={B}")
        N = cu_seqlens.shape[0] - 1
        cu_seqlens_i32 = cu_seqlens.to(torch.int32)
        ssi = (
            ssm_state_indices.to(torch.int32)
            if ssm_state_indices is not None
            else torch.arange(N, dtype=torch.int32, device=device)
        )
        if initial_state is None:
            max_idx = int(ssi.max().item()) + 1 if N > 0 else N
            state = torch.zeros(max_idx, HV, V, K, device=device, dtype=torch.bfloat16)
        else:
            state = initial_state
        if (
            output is not None
            and output.shape == v.shape
            and output.dtype == q.dtype
            and output.device == device
        ):
            out_buf = output
        else:
            out_buf = torch.empty_like(v)
    else:
        assert T == 1, f"Decode only supports T=1, got T={T}"
        cu_seqlens_i32 = None
        ssi = None
        if initial_state is None:
            state = torch.zeros(B, HV, V, K, device=device, dtype=torch.bfloat16)
        elif ssm_state_indices is not None:
            state = initial_state[ssm_state_indices].contiguous()
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
    USE_GATE = 1 if use_gate_in_kernel else 0
    HAS_BIAS = 1 if dt_bias is not None else 0
    USE_LB = 1 if lower_bound is not None else 0
    USE_CU = 1 if cu_seqlens_i32 is not None else 0
    compiled = _get_compiled_kernel(K, USE_GATE, HAS_BIAS, USE_LB, USE_CU)

    # Dummy tensors for unused optional args (TVM FFI requires all args present)
    global _dummy_cache
    if device not in _dummy_cache:
        _dummy_cache[device] = {
            "f32_1": torch.zeros(1, device=device, dtype=torch.float32),
            "i32_1": torch.zeros(1, device=device, dtype=torch.int32),
        }
    dc = _dummy_cache[device]

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
        scale if scale is not None else 1.0 / math.sqrt(K),
        1e-6,
        lower_bound if lower_bound is not None else 0.0,
    )

    return out_buf, state if output_final_state else None
