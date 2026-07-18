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

Supports single-token decode and fused speculative-token updates with
per-key-dimension gating.
Stored state S[V,K] is updated via:
S = S @ diag(g_exp) + beta * outer(v - S @ k, k)
Output: o = S @ q

Inputs:  q,k [B,T,H,K]  v,g [B,T,HV,K]  beta [B,T,HV]
State:   [N,HV,V,K] bf16, updated in-place with optional separate initial source
Output:  o [B,T,HV,V]

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
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import math as mlir_math
import tvm_ffi  # noqa: F401 -- TVM FFI required for zero-overhead kernel dispatch

# ==============================================================================
# CONSTANTS
# ==============================================================================
# One-warp dot-product evaluation schedules. Both are active specializations:
# the balanced tree minimizes dependency depth at low-grid row-8 shapes, while
# dual accumulators expose more instruction-level parallelism elsewhere.
DOT_REDUCTION_TREE = 0
DOT_REDUCTION_DUAL_ACCUM = 1


# ==============================================================================
# SHARED HELPER FUNCTIONS
# ==============================================================================


@cute.jit
def compute_gate_value(
    g_val,
    k_idx,
    A_log_val,
    gDtBias,
    h_K_offset,
    lower_bound_val,
    USE_GATE_IN_KERNEL: cutlass.Constexpr[int],
    HAS_DT_BIAS: cutlass.Constexpr[int],
    USE_LOWER_BOUND: cutlass.Constexpr[int],
):
    """Compute one exp(gate) value in registers.

    When USE_GATE_IN_KERNEL=0: g is pre-computed log-space gate, just exp it.
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=0:
        g_log = -exp(A_log) * softplus(g + dt_bias); return exp(g_log)
    When USE_GATE_IN_KERNEL=1, USE_LOWER_BOUND=1:
        g_log = lower_bound * sigmoid(exp(A_log) * (g + dt_bias)); return exp(g_log)

    Args:
        g_val: raw or log-space gate scalar
        k_idx: K-dimension index for the optional dt_bias load
        A_log_val: exp(A_log) for this head (Float32), precomputed outside
        gDtBias: dt_bias tensor [H*K] Float32 (or dummy if HAS_DT_BIAS=0)
        h_K_offset: query_head_idx * K offset into dt_bias
        lower_bound_val: lower bound float (negative, e.g. -5.0)
        USE_GATE_IN_KERNEL: 0 = pre-computed gate, 1 = in-kernel gate
        HAS_DT_BIAS: 0 = no dt_bias, 1 = add dt_bias
        USE_LOWER_BOUND: 0 = softplus formula, 1 = lower_bound * sigmoid formula

    The optional dt_bias load remains at the token-local gate computation site.
    """
    if USE_GATE_IN_KERNEL == 1:
        if HAS_DT_BIAS == 1:
            g_val = g_val + gDtBias[h_K_offset + k_idx].to(cutlass.Float32)
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
    return g_val


# ==============================================================================
# REGISTER-TILE KERNEL
# ==============================================================================


@cute.jit
def _dot8_row(lhs, row, rhs, reduction_schedule: cutlass.Constexpr[int]):
    """Evaluate an eight-element dot product with the selected dependency graph."""
    if cutlass.const_expr(reduction_schedule == DOT_REDUCTION_DUAL_ACCUM):
        even = lhs[row, 0] * rhs[0]
        odd = lhs[row, 1] * rhs[1]
        for pair in cutlass.range_constexpr(1, 4):
            i = 2 * pair
            even = lhs[row, i] * rhs[i] + even
            odd = lhs[row, i + 1] * rhs[i + 1] + odd
        return even + odd
    return (
        (lhs[row, 0] * rhs[0] + lhs[row, 1] * rhs[1])
        + (lhs[row, 2] * rhs[2] + lhs[row, 3] * rhs[3])
    ) + (
        (lhs[row, 4] * rhs[4] + lhs[row, 5] * rhs[5])
        + (lhs[row, 6] * rhs[6] + lhs[row, 7] * rhs[7])
    )


@cute.jit
def _reduce_k_group(value, HEAD_DIM: cutlass.Constexpr[int]):
    """Reduce one partial dot product across the lanes that partition K."""
    if cutlass.const_expr(HEAD_DIM == 64):
        for offset in [4, 2, 1]:
            value = value + cute.arch.shuffle_sync_bfly(
                value, offset=offset, mask=0xFFFFFFFF
            )
    else:
        for offset in [8, 4, 2, 1]:
            value = value + cute.arch.shuffle_sync_bfly(
                value, offset=offset, mask=0xFFFFFFFF
            )
    return value


@cute.kernel
def recurrent_kda_decode_kernel(
    gQ: cute.Tensor,
    gK: cute.Tensor,
    gV: cute.Tensor,
    gG: cute.Tensor,
    gBeta: cute.Tensor,
    gH: cute.Tensor,
    gInitialStateSource: cute.Tensor,
    gInitialStateIndices: cute.Tensor,
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
    TILE_ROWS: cutlass.Constexpr[int],
    DOT_REDUCTION_SCHEDULE: cutlass.Constexpr[int],
    HAS_INITIAL_STATE_SOURCE: cutlass.Constexpr[int],
    BETA_IS_LOGIT: cutlass.Constexpr[int],
    ZERO_PADDED_OUTPUT: cutlass.Constexpr[int],
    HAS_NUM_ACCEPTED_TOKENS: cutlass.Constexpr[int],
):
    """One warp owns a [TILE_ROWS V, HEAD_DIM K] tile in registers.

    Lanes partition K into contiguous eight-element vectors; the remaining lane
    groups partition V rows. State I/O uses 128-bit autovectorized copies.
    TILE_ROWS is 8, 16, or 32.
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    HV = cutlass.Int32(gV.shape[2])
    H = cutlass.Int32(gQ.shape[2])
    NUM_V_TILES = HEAD_DIM // TILE_ROWS
    v_tile_idx = bidx % NUM_V_TILES
    bh = bidx // NUM_V_TILES
    value_head_idx = bh % HV
    batch_idx = bh // HV
    query_head_idx = value_head_idx // (HV // H)
    v_offset = v_tile_idx * TILE_ROWS

    K_LANES = HEAD_DIM // 8
    V_LANES = 32 // K_LANES
    VALUES_PER_THREAD = HEAD_DIM // 32
    k_lane = tidx % K_LANES
    v_lane = tidx // K_LANES

    token_base_offset = cutlass.Int32(0)
    seq_len = cutlass.Int32(1)
    if USE_CU_SEQLENS == 1:
        token_base_offset = gCuSeqlens[batch_idx].to(cutlass.Int32)
        seq_len = gCuSeqlens[batch_idx + 1].to(cutlass.Int32) - token_base_offset
    if ZERO_PADDED_OUTPUT == 1 and tidx < TILE_ROWS:
        padded_o_head = gO[(0, batch_idx, value_head_idx, None)]
        padded_o_head[v_offset + tidx] = cutlass.BFloat16(0.0)

    init_seq_idx = batch_idx
    if USE_CU_SEQLENS == 1:
        init_raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS].to(cutlass.Int32)
        if NUM_TOKENS > 1 and HAS_NUM_ACCEPTED_TOKENS == 1:
            nat_raw = gNumAcceptedTokens[batch_idx].to(cutlass.Int32)
            nat_offset = nat_raw - 1
            nat_offset = cutlass.Int32(0) if nat_offset < 0 else nat_offset
            nat_offset = (
                cutlass.Int32(NUM_TOKENS - 1)
                if nat_offset >= NUM_TOKENS
                else nat_offset
            )
            init_raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS + nat_offset].to(
                cutlass.Int32
            )
        init_seq_idx = cutlass.Int32(0) if init_raw_slot < 0 else init_raw_slot

    h_read = gH[(init_seq_idx, value_head_idx, None, None)]
    if HAS_INITIAL_STATE_SOURCE == 1:
        source_raw_slot = gInitialStateIndices[batch_idx].to(cutlass.Int32)
        source_seq_idx = cutlass.Int32(0) if source_raw_slot < 0 else source_raw_slot
        h_read = gInitialStateSource[(source_seq_idx, value_head_idx, None, None)]

    h_layout = cute.make_layout((TILE_ROWS // V_LANES, 8), stride=(8, 1))
    vec8_layout = cute.make_layout((8,), stride=(1,))
    src_layout = cute.make_layout((VALUES_PER_THREAD,), stride=(1,))
    h_reg = cute.make_rmem_tensor(h_layout, cutlass.Float32)
    h_bf16 = cute.make_rmem_tensor(vec8_layout, cutlass.BFloat16)
    q_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    k_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    gate_src = cute.make_rmem_tensor(src_layout, cutlass.Float32)
    q_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    k_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    gate_bf16 = cute.make_rmem_tensor(src_layout, cutlass.BFloat16)
    q_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    k_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)
    gate_reg = cute.make_rmem_tensor(vec8_layout, cutlass.Float32)

    for j in cutlass.range_constexpr(TILE_ROWS // V_LANES):
        v_idx = v_offset + v_lane + V_LANES * j
        h_tile = cute.local_tile(h_read, (1, 8), (v_idx, k_lane))
        cute.autovec_copy(h_tile, h_bf16)
        for i in cutlass.range_constexpr(8):
            h_reg[j, i] = h_bf16[i].to(cutlass.Float32)

    A_log_val = cutlass.Float32(0.0)
    h_K_offset = query_head_idx * HEAD_DIM
    if USE_GATE_IN_KERNEL == 1:
        A_log_val = cute.exp(gALog[query_head_idx].to(cutlass.Float32), fastmath=True)

    seq_idx = batch_idx
    is_active = seq_len > 0
    raw_slot = cutlass.Int32(0)
    token_offset = token_base_offset
    beta = cutlass.Float32(0.0)
    v_loaded = cutlass.Float32(0.0)
    q_head = gQ[(0, 0, query_head_idx, None)]
    k_head = gK[(0, 0, query_head_idx, None)]
    gate_head = gG[(0, 0, value_head_idx, None)]
    v_head = gV[(0, 0, value_head_idx, None)]
    o_head = gO[(0, 0, value_head_idx, None)]
    h_out = gH[(batch_idx, value_head_idx, None, None)]

    for token_t in cutlass.range(NUM_TOKENS):
        token_offset = token_base_offset + token_t
        if USE_CU_SEQLENS == 1:
            raw_slot = gSsmStateIndices[batch_idx * NUM_TOKENS + token_t].to(
                cutlass.Int32
            )
            has_token = token_t < seq_len
            is_active = raw_slot >= 0 and has_token
            token_offset = token_offset if has_token else cutlass.Int32(0)
            seq_idx = cutlass.Int32(0) if raw_slot < 0 else raw_slot
        else:
            seq_idx = batch_idx
            is_active = seq_len > 0
        if USE_CU_SEQLENS == 1:
            q_head = gQ[(0, token_offset, query_head_idx, None)]
            k_head = gK[(0, token_offset, query_head_idx, None)]
            gate_head = gG[(0, token_offset, value_head_idx, None)]
            v_head = gV[(0, token_offset, value_head_idx, None)]
            o_head = gO[(0, token_offset, value_head_idx, None)]
            beta = gBeta[(0, token_offset, value_head_idx)].to(cutlass.Float32)
        else:
            q_head = gQ[(batch_idx, 0, query_head_idx, None)]
            k_head = gK[(batch_idx, 0, query_head_idx, None)]
            gate_head = gG[(batch_idx, 0, value_head_idx, None)]
            v_head = gV[(batch_idx, 0, value_head_idx, None)]
            o_head = gO[(batch_idx, 0, value_head_idx, None)]
            beta = gBeta[(batch_idx, 0, value_head_idx)].to(cutlass.Float32)

        if BETA_IS_LOGIT == 1:
            beta = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(-beta, fastmath=True)
            )

        q_tile = cute.local_tile(q_head, (VALUES_PER_THREAD,), (tidx,))
        k_tile = cute.local_tile(k_head, (VALUES_PER_THREAD,), (tidx,))
        gate_tile = cute.local_tile(gate_head, (VALUES_PER_THREAD,), (tidx,))
        cute.autovec_copy(q_tile, q_bf16)
        cute.autovec_copy(k_tile, k_bf16)
        cute.autovec_copy(gate_tile, gate_bf16)
        v_loaded = cutlass.Float32(0.0)
        # D128 row-16 hides V latency behind gate/norm work. Other measured
        # specializations are faster with the load left next to its consumer.
        if cutlass.const_expr(HEAD_DIM == 128 and TILE_ROWS == 16):
            if tidx < TILE_ROWS:
                v_loaded = v_head[v_offset + tidx].to(cutlass.Float32)

        for i in cutlass.range_constexpr(VALUES_PER_THREAD):
            k_idx = tidx * VALUES_PER_THREAD + i
            q_src[i] = q_bf16[i].to(cutlass.Float32)
            k_src[i] = k_bf16[i].to(cutlass.Float32)
            gate_src[i] = compute_gate_value(
                gate_bf16[i].to(cutlass.Float32),
                k_idx,
                A_log_val,
                gDtBias,
                h_K_offset,
                lower_bound,
                USE_GATE_IN_KERNEL,
                HAS_DT_BIAS,
                USE_LOWER_BOUND,
            )

        q_sum_sq = cutlass.Float32(0.0)
        k_sum_sq = cutlass.Float32(0.0)
        if HEAD_DIM == 64:
            q_sum_sq = q_src[0] * q_src[0] + q_src[1] * q_src[1]
            k_sum_sq = k_src[0] * k_src[0] + k_src[1] * k_src[1]
        elif cutlass.const_expr(DOT_REDUCTION_SCHEDULE == DOT_REDUCTION_DUAL_ACCUM):
            q_sum_even = q_src[0] * q_src[0]
            q_sum_odd = q_src[1] * q_src[1]
            q_sum_even = q_src[2] * q_src[2] + q_sum_even
            q_sum_odd = q_src[3] * q_src[3] + q_sum_odd
            q_sum_sq = q_sum_even + q_sum_odd
            k_sum_even = k_src[0] * k_src[0]
            k_sum_odd = k_src[1] * k_src[1]
            k_sum_even = k_src[2] * k_src[2] + k_sum_even
            k_sum_odd = k_src[3] * k_src[3] + k_sum_odd
            k_sum_sq = k_sum_even + k_sum_odd
        else:
            q_sum_sq = (q_src[0] * q_src[0] + q_src[1] * q_src[1]) + (
                q_src[2] * q_src[2] + q_src[3] * q_src[3]
            )
            k_sum_sq = (k_src[0] * k_src[0] + k_src[1] * k_src[1]) + (
                k_src[2] * k_src[2] + k_src[3] * k_src[3]
            )
        for offset in [16, 8, 4, 2, 1]:
            q_sum_sq = q_sum_sq + cute.arch.shuffle_sync_bfly(
                q_sum_sq, offset=offset, mask=0xFFFFFFFF
            )
            k_sum_sq = k_sum_sq + cute.arch.shuffle_sync_bfly(
                k_sum_sq, offset=offset, mask=0xFFFFFFFF
            )

        q_scale_factor = scale
        k_scale_factor = cutlass.Float32(1.0)
        if USE_QK_L2NORM == 1:
            q_scale_factor = cute.rsqrt(q_sum_sq + eps, fastmath=True) * scale
            k_scale_factor = cute.rsqrt(k_sum_sq + eps, fastmath=True)
        for i in cutlass.range_constexpr(8):
            source_lane = V_LANES * k_lane + i // VALUES_PER_THREAD
            source_value = i % VALUES_PER_THREAD
            q_reg[i] = (
                cute.arch.shuffle_sync(
                    q_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                )
                * q_scale_factor
            )
            k_reg[i] = (
                cute.arch.shuffle_sync(
                    k_src[source_value], offset=source_lane, mask=0xFFFFFFFF
                )
                * k_scale_factor
            )
            gate_reg[i] = cute.arch.shuffle_sync(
                gate_src[source_value], offset=source_lane, mask=0xFFFFFFFF
            )

        if cutlass.const_expr(HEAD_DIM != 128 or TILE_ROWS != 16):
            if tidx < TILE_ROWS:
                v_loaded = v_head[v_offset + tidx].to(cutlass.Float32)
        for j in cutlass.range_constexpr(TILE_ROWS // V_LANES):
            for i in cutlass.range_constexpr(8):
                h_reg[j, i] = h_reg[j, i] * gate_reg[i]
            pred = _reduce_k_group(
                _dot8_row(h_reg, j, k_reg, DOT_REDUCTION_SCHEDULE), HEAD_DIM
            )

            v_idx = v_offset + v_lane + V_LANES * j
            v_val = cute.arch.shuffle_sync(
                v_loaded, offset=v_lane + V_LANES * j, mask=0xFFFFFFFF
            )
            delta = (v_val - pred) * beta
            for i in cutlass.range_constexpr(8):
                h_reg[j, i] = k_reg[i] * delta + h_reg[j, i]
            out = _reduce_k_group(
                _dot8_row(h_reg, j, q_reg, DOT_REDUCTION_SCHEDULE), HEAD_DIM
            )

            if is_active:
                if k_lane == j:
                    o_head[v_idx] = out.to(cutlass.BFloat16)

        if is_active:
            h_out = gH[(seq_idx, value_head_idx, None, None)]
            for j in cutlass.range_constexpr(TILE_ROWS // V_LANES):
                v_idx = v_offset + v_lane + V_LANES * j
                for i in cutlass.range_constexpr(8):
                    h_bf16[i] = h_reg[j, i].to(cutlass.BFloat16)
                h_tile = cute.local_tile(h_out, (1, 8), (v_idx, k_lane))
                cute.autovec_copy(h_bf16, h_tile)


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
    mInitialStateSource: cute.Tensor,
    mInitialStateIndices: cute.Tensor,
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
    TILE_ROWS: cutlass.Constexpr[int],
    DOT_REDUCTION_SCHEDULE: cutlass.Constexpr[int],
    HAS_INITIAL_STATE_SOURCE: cutlass.Constexpr[int],
    BETA_IS_LOGIT: cutlass.Constexpr[int],
    ZERO_PADDED_OUTPUT: cutlass.Constexpr[int],
    HAS_NUM_ACCEPTED_TOKENS: cutlass.Constexpr[int],
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
        mInitialStateSource,
        mInitialStateIndices,
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
        TILE_ROWS,
        DOT_REDUCTION_SCHEDULE,
        HAS_INITIAL_STATE_SOURCE,
        BETA_IS_LOGIT,
        ZERO_PADDED_OUTPUT,
        HAS_NUM_ACCEPTED_TOKENS,
    ).launch(
        grid=[batch_size * HV * (HEAD_DIM // TILE_ROWS), 1, 1],
        block=[32, 1, 1],
        stream=stream,
    )


# ==============================================================================
# PUBLIC API
# ==============================================================================

_dummy_cache = {}  # device -> dict of pre-allocated dummy tensors


def _make_compile_inputs(HEAD_DIM, USE_CU_SEQLENS):
    """Build the shared symbolic runtime argument list for CuTe compilation."""
    B, H, HV, N, N0 = (
        cute.sym_int(),
        cute.sym_int(),
        cute.sym_int(),
        cute.sym_int(),
        cute.sym_int(),
    )
    K = V = HEAD_DIM
    T_dim = cute.sym_int() if USE_CU_SEQLENS == 1 else 1

    def make_compact(shape, dtype=cute.BFloat16):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            assumed_align=32,
            stride_order=tuple(reversed(range(len(shape)))),
        )

    # State pools permit a padded outer stride while retaining compact [HV,V,K]
    # blocks. This supports block-based caches without constraining their prefix.
    HV_state = cute.sym_int()
    state_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(N, HV_state, V, K),
        stride=(cute.sym_int64(divisibility=16), V * K, K, 1),
        assumed_align=32,
    )
    source_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(N0, HV_state, V, K),
        stride=(cute.sym_int64(divisibility=16), V * K, K, 1),
        assumed_align=32,
    )

    # Gate views may have non-compact batch/token strides after splitting a
    # fused projection; the head and K dimensions remain compact.
    gate_fake = cute.runtime.make_fake_tensor(
        cute.BFloat16,
        shape=(B, T_dim, HV, K),
        stride=(
            cute.sym_int64(divisibility=16),
            cute.sym_int64(divisibility=16),
            K,
            1,
        ),
        assumed_align=32,
    )

    return (
        make_compact((B, T_dim, H, K)),
        make_compact((B, T_dim, H, K)),
        make_compact((B, T_dim, HV, V)),
        gate_fake,
        make_compact((B, T_dim, HV)),
        state_fake,
        source_fake,
        make_compact((cute.sym_int(),), dtype=cute.Int32),
        make_compact((B, T_dim, HV, V)),
        make_compact((cute.sym_int(),), dtype=cute.Float32),
        make_compact((cute.sym_int(),), dtype=cute.Float32),
        make_compact((cute.sym_int(),), dtype=cute.Int32),
        make_compact((cute.sym_int(),), dtype=cute.Int32),
        make_compact((cute.sym_int(),), dtype=cute.Int32),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cutlass.Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )


@functools.cache
def _get_compiled_kernel(
    HEAD_DIM,
    USE_QK_L2NORM,
    USE_GATE_IN_KERNEL,
    HAS_DT_BIAS,
    USE_LOWER_BOUND,
    USE_CU_SEQLENS,
    HAS_INITIAL_STATE_SOURCE,
    BETA_IS_LOGIT,
    NUM_TOKENS,
    TILE_ROWS,
    DOT_REDUCTION_SCHEDULE,
    ZERO_PADDED_OUTPUT,
    HAS_NUM_ACCEPTED_TOKENS,
):
    """Compile a register-tile specialization."""
    return cute.compile(
        recurrent_kda_launch,
        *_make_compile_inputs(HEAD_DIM, USE_CU_SEQLENS),
        HEAD_DIM,
        USE_QK_L2NORM,
        USE_GATE_IN_KERNEL,
        HAS_DT_BIAS,
        USE_LOWER_BOUND,
        USE_CU_SEQLENS,
        NUM_TOKENS,
        TILE_ROWS,
        DOT_REDUCTION_SCHEDULE,
        HAS_INITIAL_STATE_SOURCE,
        BETA_IS_LOGIT,
        ZERO_PADDED_OUTPUT,
        HAS_NUM_ACCEPTED_TOKENS,
        options="--enable-tvm-ffi --generate-line-info",
    )


def _select_kernel_schedule(
    head_dim,
    num_tokens,
    use_gate,
    sequence_heads,
):
    """Select tile rows and dot-product reduction for the active kernel."""
    if head_dim == 64:
        if not use_gate and sequence_heads <= 256:
            tile_rows = 8
        elif sequence_heads >= 4096 or (sequence_heads >= 2048 and num_tokens >= 6):
            tile_rows = 32
        else:
            tile_rows = 16
    elif num_tokens == 2:
        tile_rows = 8 if sequence_heads <= 192 else 16
    elif (
        sequence_heads <= 176
        or 304 <= sequence_heads <= 368
        or 448 <= sequence_heads <= 560
        or sequence_heads >= 720
    ):
        tile_rows = 8
    elif 224 <= sequence_heads <= 288:
        tile_rows = 32
    else:
        tile_rows = 16

    if head_dim == 128 and use_gate and sequence_heads >= 224:
        tile_rows = 16
    # The measured 64-sequence-head T1 gate path amortizes its front end better
    # with row-16 despite the smaller grid and higher register allocation.
    if head_dim == 128 and use_gate and num_tokens == 1 and sequence_heads == 64:
        tile_rows = 16

    reduction_schedule = (
        DOT_REDUCTION_DUAL_ACCUM
        if tile_rows != 8 or sequence_heads >= 304
        else DOT_REDUCTION_TREE
    )
    return tile_rows, reduction_schedule


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
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    beta_is_logit: bool = False,
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
            Pre-sigmoided unless ``beta_is_logit=True``.
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
            ``ssm_state_indices[n, 0]``.
            Kernel still writes all T=1+S checkpoint slots. Caller contract:
            ``num_accepted_tokens >= 1`` (the bonus token is always accepted).
            Values above ``1+S`` are clamped to the final checkpoint slot.
            If ``None``, initial state is loaded from ``ssm_state_indices[n, 0]``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor. Shape ``[B, 1, HV, V]`` for standard
            decode, ``[1, N*(1+S), HV, V]`` for spec decode with
            ``cu_seqlens``, or ``[B, 1+S, HV, V]`` for batched spec decode
            (``num_spec_tokens=S`` without ``cu_seqlens``). Must be
            pre-allocated for CUDA graph capture; auto-allocated if ``None``.
        initial_state_source (Optional[torch.Tensor]):
            Optional read-only committed state pool ``[N0, HV, V, K]``. When
            provided, token 0 is loaded directly from this pool instead of from
            ``initial_state``, avoiding an external gather into speculative scratch.
        initial_state_indices (Optional[torch.Tensor]):
            Source slot per sequence, shape ``[N]`` int32. Required together
            with ``initial_state_source``.
        beta_is_logit (bool):
            If True, apply sigmoid to ``beta`` inside the recurrent kernel.
            Default False preserves the pre-sigmoided input contract.

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
          ``1+S`` for every row, including padded sequences (which signal via
          ``ssm_state_indices == -1``). This caller contract is not validated
          at runtime to preserve CUDA graph compatibility.
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

    if (initial_state_source is None) != (initial_state_indices is None):
        raise ValueError(
            "initial_state_source and initial_state_indices must be provided together"
        )
    if initial_state_source is not None:
        if initial_state_indices.ndim != 1:
            raise ValueError(
                "initial_state_indices must be 1D, "
                f"got shape {list(initial_state_indices.shape)}"
            )
        if initial_state_source.dtype != torch.bfloat16:
            raise ValueError(
                "initial_state_source must be bfloat16, "
                f"got {initial_state_source.dtype}"
            )
        if initial_state_source.shape[1:] != (HV, V, K):
            raise ValueError(
                "initial_state_source must have trailing shape "
                f"[{HV}, {V}, {K}], got {list(initial_state_source.shape)}"
            )

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

    zero_padded_output = False
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
        if initial_state_indices is not None and initial_state_indices.shape[0] < N:
            raise ValueError(
                "initial_state_indices must contain one entry per sequence, "
                f"got {initial_state_indices.shape[0]} entries for N={N}"
            )
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
            zero_padded_output = K == 128 and v.shape[1] == N
            if (
                output is not None
                and output.shape == v.shape
                and output.dtype == q.dtype
                and output.device == device
            ):
                # The D=128 one-warp kernel writes zero for inactive rows, so
                # dense CUDA-graph output is defined without a separate fill.
                if not zero_padded_output:
                    output.zero_()
                out_buf = output
            else:
                out_buf = (
                    torch.empty_like(v) if zero_padded_output else torch.zeros_like(v)
                )
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
        if initial_state_indices is not None and initial_state_indices.shape[0] < B:
            raise ValueError(
                "initial_state_indices must contain one entry per batch row, "
                f"got {initial_state_indices.shape[0]} entries for B={B}"
            )
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

    # With no packed tokens there is no safe address for predicated loads and
    # no recurrent update to perform. Output initialization above defines the
    # result while the caller-owned state remains unchanged.
    if cu_seqlens_i32 is not None and q.shape[1] == 0:
        return (out_buf, state if output_final_state else None)

    # Compile kernel (cached by constexpr config)
    USE_QK_NORM = 1 if use_qk_l2norm_in_kernel else 0
    USE_GATE = 1 if use_gate_in_kernel else 0
    HAS_BIAS = 1 if dt_bias is not None else 0
    USE_LB = 1 if lower_bound is not None else 0
    USE_CU = 1 if cu_seqlens_i32 is not None else 0
    HAS_SOURCE = 1 if initial_state_source is not None else 0
    BETA_LOGIT = 1 if beta_is_logit else 0
    ZERO_PADDED_OUTPUT = 1 if zero_padded_output else 0
    HAS_NAT = 1 if num_accepted_tokens is not None else 0
    if cu_seqlens is None:
        NUM_TOKENS = 1

    # Select the measured tile/reduction schedule for this workload.
    grid_seqs = cu_seqlens_i32.shape[0] - 1 if cu_seqlens_i32 is not None else B
    sequence_heads = grid_seqs * HV
    tile_rows, reduction_schedule = _select_kernel_schedule(
        K,
        NUM_TOKENS,
        use_gate_in_kernel,
        sequence_heads,
    )
    compiled = _get_compiled_kernel(
        K,
        USE_QK_NORM,
        USE_GATE,
        HAS_BIAS,
        USE_LB,
        USE_CU,
        HAS_SOURCE,
        BETA_LOGIT,
        NUM_TOKENS,
        tile_rows,
        reduction_schedule,
        ZERO_PADDED_OUTPUT,
        HAS_NAT,
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
        initial_state_source if initial_state_source is not None else state,
        initial_state_indices.to(torch.int32).contiguous()
        if initial_state_indices is not None
        else dc["i32_1"],
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
