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
"""

"""
Gated Delta Rule Decode Kernel - BF16 Hidden State
===================================================

CuTe DSL kernel for GDN decode with BF16 hidden state storage.
Provides both T=1 (single token) and MTP (multi-token prediction) variants.

Approach:
- Each warp processes ONE V-row at a time (4 warps = 4 V-rows per iteration)
- Each thread holds vec_size=4 K-elements, using warp-level shuffle reduction
- H state is loaded/stored as BF16, converted to FP32 in registers for compute
- cp.async pipeline with TILE_V=8 x TILE_K=128 tiles

Architecture:
- 128 threads (4 warps x 32 threads)
- TILE_V=8 rows of H loaded per pipeline stage
- TILE_K=128 (full K dimension)
- Each thread: 4 K-elements (lane_id * 4 to lane_id * 4 + 3)
- Warp shuffle reduction across 32 threads for dot products

Public API:
- gated_delta_rule(): T=1 single-token decode with BF16 state
- gated_delta_rule_mtp(): Multi-token prediction (T>=1) with BF16 state
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

# ==============================================================================
# CONSTANTS
# ==============================================================================
TILE_V = 8
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128
NUM_BLOCKS_PER_STATE = 8  # 8 CTAs per (batch, head) for small batch

# ==============================================================================
# CONSTANTS FOR ILP-OPTIMIZED KERNEL (large batch sizes)
# ==============================================================================
TILE_V_ILP = 128  # V-tile size: each block processes all 128 V-rows
TILE_K_ILP = 128  # Full K dimension
NUM_THREADS_ILP = 128  # 4 warps
VEC_SIZE_ILP = 4  # Elements per thread along K (changed dynamically)
ILP_ROWS = (
    8  # Process 8 V-rows simultaneously per group (optimal ILP for latency hiding)
)


# ==============================================================================
# FMA WRAPPER FUNCTIONS (SM90 Compatibility)
# ==============================================================================
# cute.arch.fma_packed_f32x2() generates F32x2 intrinsics NOT supported on SM90.
# These wrappers use scalar FMA operations that work on all SM90+ architectures.
# On SM100+ (Blackwell), use_packed_fma=True selects the native packed path.


@cute.jit
def fma_pair_mul(a1, a2, b1, b2):
    """Multiply two pairs: (a1*b1, a2*b2). SM90-compatible."""
    result1 = a1 * b1
    result2 = a2 * b2
    return result1, result2


@cute.jit
def fma_pair(a1, a2, b1, b2, c1, c2):
    """FMA two pairs: (a1*b1+c1, a2*b2+c2). SM90-compatible."""
    result1 = a1 * b1 + c1
    result2 = a2 * b2 + c2
    return result1, result2


# ==============================================================================
# KERNEL: T=1 with gdn_decode approach but BF16 state
# ==============================================================================


@cute.kernel
def gdn_decode_bf16state_cooprow_kernel(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,  # [B*HV, V, K] as BF16
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, 1, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, 1, H, K]
    k: cute.Tensor,  # [B, 1, H, K]
    v: cute.Tensor,  # [B, 1, HV, V]
    b: cute.Tensor,  # [B, 1, HV]
    o: cute.Tensor,  # [B, 1, HV, V] - output
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
):
    """
    T=1 GDN decode kernel using the 'different approach':
    - Pipeline loads TILE_V x TILE_K BF16 tiles of H from GMEM to SMEM
    - Each warp processes 1 V-row (4 warps = 4 rows per TILE_V=8 iteration)
    - Each thread: vec_size=4 K-elements with warp shuffle reduction
    - H stored as BF16, compute in FP32
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    block_idx, _, _ = cute.arch.block_idx()

    batch_idx = block_idx // NUM_BLOCKS_PER_STATE
    batch_inner = block_idx % NUM_BLOCKS_PER_STATE
    num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    smem = cutlass.utils.SmemAllocator()

    # Allocate shared memory for H tile pipeline (BF16)
    sData = smem.allocate_tensor(cutlass.BFloat16, smem_layout_staged, 128)

    # Allocate shared memory for output (V elements, BF16)
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    # Allocate shared memory for v values (V elements, FP32)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    # Register tensors for K, Q, and H (vec_size=4 per thread)
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # BF16 register tensors for vectorized loading
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    # Each thread's K-range: lane_id*4 .. lane_id*4+3
    k_start = lane_id * vec_size

    # Read gate values from GMEM early (hide latency during subsequent syncs)
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    cute.arch.barrier()

    # Global memory views
    gSrc_batch = h0_source[(batch_idx, None, None)]  # (V, K) in BF16
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    # V-direction tiles
    gSrc = cute.local_tile(
        gSrc_batch, (TILE_V, TILE_K), (None, 0)
    )  # (TILE_V, TILE_K, num_v_tiles)

    # Partition for async load
    thr_copy_load = tiled_copy_load.get_slice(tidx)

    # ===================================================================
    # Prefetch first pipeline stages
    # ===================================================================
    start_v_tiles = batch_inner * num_v_tiles_per_block
    prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles_per_block)
    for v_tiles in range(start_v_tiles, start_v_tiles + prefetch_count):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES

        gSrc_tile = gSrc[(None, None, v_tiles)]
        sData_stage = sData[(None, None, stage)]

        thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
        thr_sData = thr_copy_load.partition_D(sData_stage)

        cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
        cute.arch.cp_async_commit_group()

    # Load q, k as BF16, convert to FP32
    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)

    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(r_q_bf16[i])
        r_k[i] = cutlass.Float32(r_k_bf16[i])

    # Load v as BF16, convert to FP32, store to sV
    v_tile = cute.local_tile(v, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_id))
    cute.autovec_copy(v_tile, r_v_bf16)
    for i in cutlass.range_constexpr(vec_size):
        sV[k_start + i] = cutlass.Float32(r_v_bf16[i])

    cute.arch.barrier()

    # ===================================================================
    # Compute gate values: g_exp and beta
    # ===================================================================
    r_g = 0.0
    r_beta = 0.0
    if lane_id == 0:
        x = r_a + r_dt_bias
        beta_x = softplus_beta * x
        softplus_x = 0.0

        if beta_x <= softplus_threshold:
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            log_input = cutlass.Float32(1.0 + exp_beta_x)
            log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
            softplus_x = cutlass.Float32(
                (cutlass.Float32(1.0) / softplus_beta) * log_result
            )
        else:
            softplus_x = x

        r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
        r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))
        r_g = cute.exp(r_g_value, fastmath=True)

    r_g = cute.arch.shuffle_sync(r_g, 0)
    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    # ===================================================================
    # L2 normalization of Q and K (if enabled)
    # ===================================================================
    if use_qk_l2norm:
        sum_q = 0.0
        sum_k = 0.0
        for i in cutlass.range_constexpr(vec_size):
            sum_q += r_q[i] * r_q[i]
            sum_k += r_k[i] * r_k[i]
        for offset in [16, 8, 4, 2, 1]:
            sum_q += cute.arch.shuffle_sync_bfly(
                sum_q, offset=offset, mask=-1, mask_and_clamp=31
            )
            sum_k += cute.arch.shuffle_sync_bfly(
                sum_k, offset=offset, mask=-1, mask_and_clamp=31
            )

        inv_norm_q = cute.rsqrt(sum_q + 1e-6, fastmath=True)
        inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = r_q[i] * inv_norm_q
            r_k[i] = r_k[i] * inv_norm_k

    # Apply scale to Q
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    # ===================================================================
    # Main loop: process V tiles
    # ===================================================================
    end_v_tiles = start_v_tiles + num_v_tiles_per_block
    for v_tiles in range(start_v_tiles, end_v_tiles):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES

        # Wait for current stage
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Prefetch next tile
        next_v_tiles = v_tiles + prefetch_count
        if next_v_tiles < end_v_tiles:
            next_stage = (next_v_tiles - start_v_tiles) % NUM_STAGES

            gSrc_next = gSrc[(None, None, next_v_tiles)]
            sData_next = sData[(None, None, next_stage)]

            thr_gSrc = thr_copy_load.partition_S(gSrc_next)
            thr_sData = thr_copy_load.partition_D(sData_next)

            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        # Process TILE_V rows, 4 rows at a time (one per warp)
        for row in cutlass.range_constexpr(0, TILE_V, 4):
            row_offset = tidx // 32  # = warp_idx
            sum_hk = 0.0

            # Load H from BF16 SMEM, convert to FP32 in registers
            sData_tile = cute.local_tile(
                sData, (1, vec_size, 1), (row + row_offset, lane_id, stage)
            )
            # Manual load + convert BF16 -> FP32
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = cutlass.Float32(sData_tile[i])

            # Decay H and compute dot product: sum_hk = sum(h * k)
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = r_h[i] * r_g
                sum_hk += r_h[i] * r_k[i]

            # Warp-level reduction for sum_hk
            for offset in [16, 8, 4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                )

            # Delta update: v_delta = beta * (v - pred)
            v_new = sV[v_tiles * TILE_V + row + row_offset] - sum_hk
            v_new = v_new * r_beta

            # Update H and compute output dot product: sum_hq = sum(h * q)
            sum_hq = 0.0
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] += r_k[i] * v_new
                sum_hq += r_h[i] * r_q[i]

            # Write updated H back to GMEM as BF16 via gDst
            gDst_tile = cute.local_tile(
                gDst, (1, 1, vec_size, 1), (0, row + row_offset, lane_id, v_tiles)
            )
            for i in cutlass.range_constexpr(vec_size):
                gDst_tile[i] = cutlass.BFloat16(r_h[i])

            # Warp-level reduction for sum_hq
            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                )

            o_idx = v_tiles * TILE_V + row + row_offset
            if lane_id == 0 and o_idx < V:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)

    # ===================================================================
    # Final writeback: output from SMEM to GMEM
    # ===================================================================
    cute.arch.barrier()
    if tidx >= start_v_tiles * TILE_V and tidx < end_v_tiles * TILE_V:
        o[(i_n, i_t, i_hv, tidx)] = sOutput[tidx]


# ==============================================================================
# KERNEL: ILP-OPTIMIZED T=1 with direct GMEM->register loads, 8-row ILP
# ==============================================================================
# Architecture (matches MTP kernel pattern):
# - Grid: (B * HV * num_v_tiles, 1, 1) - each block handles one TILE_V chunk
# - 128 threads = 4 groups of 32 threads (full warps)
# - Each group processes TILE_V/4 V-rows total, 8 rows at a time (ILP=8)
# - H loaded directly from GMEM into registers via autovec_copy (128-bit BF16 loads)
# - No SMEM pipeline - ILP hides memory latency instead


@cute.kernel
def gdn_decode_bf16state_ilp_kernel(
    h0_source: cute.Tensor,  # [B*HV, V, K] as BF16 (K-last, autovec_copy compatible)
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, 1, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, 1, H, K]
    k: cute.Tensor,  # [B, 1, H, K]
    v: cute.Tensor,  # [B, 1, HV, V]
    b: cute.Tensor,  # [B, 1, HV]
    o: cute.Tensor,  # [B, 1, HV, V] - output
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
):
    """
    ILP-optimized T=1 GDN decode kernel with BF16 state.
    Direct GMEM->register loads with 8-row ILP for high memory throughput.
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # 4 groups (= 4 warps), each full warp of 32 threads
    threads_per_group: cutlass.Constexpr[int] = 32  # noqa: F841
    num_groups: cutlass.Constexpr[int] = 4
    group_idx = warp_idx
    lane_in_group = lane_id

    batch_idx, _, _ = cute.arch.block_idx()

    # Decode block index: (i_n, i_hv, i_v) from batch_idx
    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)
    i_t = 0

    # Load A_log and dt_bias once
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])

    # No shared memory needed for ILP kernel (direct GMEM access)

    # Register arrays for q, k, and h (8 rows of vec_size=4 each)
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((ILP_ROWS, vec_size), stride=(vec_size, 1)), cutlass.Float32
    )

    # BF16 register tensors for vectorized loading from BF16 state
    # We use 4 separate BF16 register tensors for ILP loads
    r_hb0 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb1 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb2 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb3 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb5 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb6 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb7 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    # BF16 register tensors for vectorized loading q, k
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    # BF16 register tensors for vectorized V load and output store (8 elements)
    r_v_bf16_vec = cute.make_rmem_tensor(
        cute.make_layout((ILP_ROWS,), stride=(1,)), cutlass.BFloat16
    )
    r_o_bf16_vec = cute.make_rmem_tensor(
        cute.make_layout((ILP_ROWS,), stride=(1,)), cutlass.BFloat16
    )

    # Compute gate values: only lane 0 computes, then broadcast
    r_a_val = cutlass.Float32(a[i_n, i_t, i_hv])
    r_b_val = cutlass.Float32(b[i_n, i_t, i_hv])

    r_g = 0.0
    r_beta = 0.0
    if lane_id == 0:
        x = r_a_val + r_dt_bias
        beta_x = softplus_beta * x
        softplus_x = 0.0
        if beta_x <= softplus_threshold:
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            log_input = cutlass.Float32(1.0 + exp_beta_x)
            log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
            softplus_x = cutlass.Float32(
                (cutlass.Float32(1.0) / softplus_beta) * log_result
            )
        else:
            softplus_x = x
        r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
        r_beta = 1.0 / (1.0 + cute.exp(-r_b_val, fastmath=True))
        r_g = cute.exp(r_g_value, fastmath=True)

    r_g = cute.arch.shuffle_sync(r_g, 0)
    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    # Load q, k as BF16, convert to FP32
    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group))
    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group))
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(r_q_bf16[i])
        r_k[i] = cutlass.Float32(r_k_bf16[i])

    # L2 normalization of Q and K
    if use_qk_l2norm:
        sum_q = 0.0
        sum_k = 0.0
        for i in cutlass.range_constexpr(vec_size):
            sum_q += r_q[i] * r_q[i]
            sum_k += r_k[i] * r_k[i]
        sum_q = cute.arch.warp_reduction_sum(sum_q)
        sum_k = cute.arch.warp_reduction_sum(sum_k)
        inv_norm_q = cute.rsqrt(sum_q + 1e-6, fastmath=True)
        inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
        for i in cutlass.range_constexpr(vec_size):
            r_q[i] = r_q[i] * inv_norm_q
            r_k[i] = r_k[i] * inv_norm_k

    # Apply scale to Q
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    # ===================================================================
    # Main loop: process V rows with 8-row ILP
    # ===================================================================
    flat_state_idx = i_n * HV + i_hv
    rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
    eighth_rows: cutlass.Constexpr[int] = rows_per_group // ILP_ROWS

    for row_oct in cutlass.range_constexpr(eighth_rows):
        v_base = i_v * tile_v + group_idx * rows_per_group + row_oct * ILP_ROWS
        v0 = v_base
        v1 = v_base + 1
        v2 = v_base + 2
        v3 = v_base + 3
        v4 = v_base + 4
        v5 = v_base + 5
        v6 = v_base + 6
        v7 = v_base + 7

        # Always true when tile_v=128, V=128, 4 groups * 8 ILP_ROWS * 4 iters = 128
        if True:
            # Load h for ALL 8 V-rows: GMEM BF16 -> BF16 regs (vectorized) -> FP32 regs
            ht0 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v0, lane_in_group)
            )
            ht1 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v1, lane_in_group)
            )
            ht2 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v2, lane_in_group)
            )
            ht3 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v3, lane_in_group)
            )
            ht4 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v4, lane_in_group)
            )
            ht5 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v5, lane_in_group)
            )
            ht6 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v6, lane_in_group)
            )
            ht7 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v7, lane_in_group)
            )
            # Vectorized BF16 loads (64-bit = 4 BF16 elements per load)
            cute.autovec_copy(ht0, r_hb0)
            cute.autovec_copy(ht1, r_hb1)
            cute.autovec_copy(ht2, r_hb2)
            cute.autovec_copy(ht3, r_hb3)
            cute.autovec_copy(ht4, r_hb4)
            cute.autovec_copy(ht5, r_hb5)
            cute.autovec_copy(ht6, r_hb6)
            cute.autovec_copy(ht7, r_hb7)

            # Convert BF16 -> FP32, apply decay, AND compute dot products h@k in single pass
            # Using fma_packed_f32x2 for paired FMA operations
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            s3 = 0.0
            s4 = 0.0
            s5 = 0.0
            s6 = 0.0
            s7 = 0.0
            s0b = 0.0
            s1b = 0.0
            s2b = 0.0
            s3b = 0.0
            s4b = 0.0
            s5b = 0.0
            s6b = 0.0
            s7b = 0.0
            for i in cutlass.range_constexpr(0, vec_size, 2):
                # Convert + decay for pairs of elements
                if cutlass.const_expr(use_packed_fma):
                    r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(
                            cutlass.Float32(r_hb0[i]),
                            cutlass.Float32(r_hb0[i + 1]),
                        ),
                        src_b=(r_g, r_g),
                        src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                    )
                    r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(
                            cutlass.Float32(r_hb1[i]),
                            cutlass.Float32(r_hb1[i + 1]),
                        ),
                        src_b=(r_g, r_g),
                        src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                    )
                    r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(
                            cutlass.Float32(r_hb2[i]),
                            cutlass.Float32(r_hb2[i + 1]),
                        ),
                        src_b=(r_g, r_g),
                        src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                    )
                    r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(
                            cutlass.Float32(r_hb3[i]),
                            cutlass.Float32(r_hb3[i + 1]),
                        ),
                        src_b=(r_g, r_g),
                        src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                    )
                    r_h[4, i], r_h[4, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(
                            cutlass.Float32(r_hb4[i]),
                            cutlass.Float32(r_hb4[i + 1]),
                        ),
                        src_b=(r_g, r_g),
                        src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                    )
                    r_h[5, i], r_h[5, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(
                            cutlass.Float32(r_hb5[i]),
                            cutlass.Float32(r_hb5[i + 1]),
                        ),
                        src_b=(r_g, r_g),
                        src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                    )
                    r_h[6, i], r_h[6, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(
                            cutlass.Float32(r_hb6[i]),
                            cutlass.Float32(r_hb6[i + 1]),
                        ),
                        src_b=(r_g, r_g),
                        src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                    )
                    r_h[7, i], r_h[7, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(
                            cutlass.Float32(r_hb7[i]),
                            cutlass.Float32(r_hb7[i + 1]),
                        ),
                        src_b=(r_g, r_g),
                        src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                    )
                else:
                    r_h[0, i], r_h[0, i + 1] = fma_pair_mul(
                        cutlass.Float32(r_hb0[i]),
                        cutlass.Float32(r_hb0[i + 1]),
                        r_g,
                        r_g,
                    )
                    r_h[1, i], r_h[1, i + 1] = fma_pair_mul(
                        cutlass.Float32(r_hb1[i]),
                        cutlass.Float32(r_hb1[i + 1]),
                        r_g,
                        r_g,
                    )
                    r_h[2, i], r_h[2, i + 1] = fma_pair_mul(
                        cutlass.Float32(r_hb2[i]),
                        cutlass.Float32(r_hb2[i + 1]),
                        r_g,
                        r_g,
                    )
                    r_h[3, i], r_h[3, i + 1] = fma_pair_mul(
                        cutlass.Float32(r_hb3[i]),
                        cutlass.Float32(r_hb3[i + 1]),
                        r_g,
                        r_g,
                    )
                    r_h[4, i], r_h[4, i + 1] = fma_pair_mul(
                        cutlass.Float32(r_hb4[i]),
                        cutlass.Float32(r_hb4[i + 1]),
                        r_g,
                        r_g,
                    )
                    r_h[5, i], r_h[5, i + 1] = fma_pair_mul(
                        cutlass.Float32(r_hb5[i]),
                        cutlass.Float32(r_hb5[i + 1]),
                        r_g,
                        r_g,
                    )
                    r_h[6, i], r_h[6, i + 1] = fma_pair_mul(
                        cutlass.Float32(r_hb6[i]),
                        cutlass.Float32(r_hb6[i + 1]),
                        r_g,
                        r_g,
                    )
                    r_h[7, i], r_h[7, i + 1] = fma_pair_mul(
                        cutlass.Float32(r_hb7[i]),
                        cutlass.Float32(r_hb7[i + 1]),
                        r_g,
                        r_g,
                    )
                # Dot product h@k using paired FMA
                if cutlass.const_expr(use_packed_fma):
                    s0, s0b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[0, i], r_h[0, i + 1]),
                        src_b=(r_k[i], r_k[i + 1]),
                        src_c=(s0, s0b),
                    )
                    s1, s1b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[1, i], r_h[1, i + 1]),
                        src_b=(r_k[i], r_k[i + 1]),
                        src_c=(s1, s1b),
                    )
                    s2, s2b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[2, i], r_h[2, i + 1]),
                        src_b=(r_k[i], r_k[i + 1]),
                        src_c=(s2, s2b),
                    )
                    s3, s3b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[3, i], r_h[3, i + 1]),
                        src_b=(r_k[i], r_k[i + 1]),
                        src_c=(s3, s3b),
                    )
                    s4, s4b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[4, i], r_h[4, i + 1]),
                        src_b=(r_k[i], r_k[i + 1]),
                        src_c=(s4, s4b),
                    )
                    s5, s5b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[5, i], r_h[5, i + 1]),
                        src_b=(r_k[i], r_k[i + 1]),
                        src_c=(s5, s5b),
                    )
                    s6, s6b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[6, i], r_h[6, i + 1]),
                        src_b=(r_k[i], r_k[i + 1]),
                        src_c=(s6, s6b),
                    )
                    s7, s7b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[7, i], r_h[7, i + 1]),
                        src_b=(r_k[i], r_k[i + 1]),
                        src_c=(s7, s7b),
                    )
                else:
                    s0, s0b = fma_pair(
                        r_h[0, i], r_h[0, i + 1], r_k[i], r_k[i + 1], s0, s0b
                    )
                    s1, s1b = fma_pair(
                        r_h[1, i], r_h[1, i + 1], r_k[i], r_k[i + 1], s1, s1b
                    )
                    s2, s2b = fma_pair(
                        r_h[2, i], r_h[2, i + 1], r_k[i], r_k[i + 1], s2, s2b
                    )
                    s3, s3b = fma_pair(
                        r_h[3, i], r_h[3, i + 1], r_k[i], r_k[i + 1], s3, s3b
                    )
                    s4, s4b = fma_pair(
                        r_h[4, i], r_h[4, i + 1], r_k[i], r_k[i + 1], s4, s4b
                    )
                    s5, s5b = fma_pair(
                        r_h[5, i], r_h[5, i + 1], r_k[i], r_k[i + 1], s5, s5b
                    )
                    s6, s6b = fma_pair(
                        r_h[6, i], r_h[6, i + 1], r_k[i], r_k[i + 1], s6, s6b
                    )
                    s7, s7b = fma_pair(
                        r_h[7, i], r_h[7, i + 1], r_k[i], r_k[i + 1], s7, s7b
                    )
            # Combine paired accumulators
            s0 = s0 + s0b
            s1 = s1 + s1b
            s2 = s2 + s2b
            s3 = s3 + s3b
            s4 = s4 + s4b
            s5 = s5 + s5b
            s6 = s6 + s6b
            s7 = s7 + s7b

            # Interleaved butterfly reduction for all 8 s-values (better ILP than sequential warp_reduction_sum)
            for offset in [16, 8, 4, 2, 1]:
                s0 += cute.arch.shuffle_sync_bfly(
                    s0, offset=offset, mask=-1, mask_and_clamp=31
                )
                s1 += cute.arch.shuffle_sync_bfly(
                    s1, offset=offset, mask=-1, mask_and_clamp=31
                )
                s2 += cute.arch.shuffle_sync_bfly(
                    s2, offset=offset, mask=-1, mask_and_clamp=31
                )
                s3 += cute.arch.shuffle_sync_bfly(
                    s3, offset=offset, mask=-1, mask_and_clamp=31
                )
                s4 += cute.arch.shuffle_sync_bfly(
                    s4, offset=offset, mask=-1, mask_and_clamp=31
                )
                s5 += cute.arch.shuffle_sync_bfly(
                    s5, offset=offset, mask=-1, mask_and_clamp=31
                )
                s6 += cute.arch.shuffle_sync_bfly(
                    s6, offset=offset, mask=-1, mask_and_clamp=31
                )
                s7 += cute.arch.shuffle_sync_bfly(
                    s7, offset=offset, mask=-1, mask_and_clamp=31
                )

            # Step 3: Delta rule update - vectorized V load (8 consecutive BF16 elements)
            vt_slice = cute.local_tile(
                v, (1, 1, 1, ILP_ROWS), (i_n, i_t, i_hv, v_base // ILP_ROWS)
            )
            cute.autovec_copy(vt_slice, r_v_bf16_vec)
            vn0 = (cutlass.Float32(r_v_bf16_vec[0]) - s0) * r_beta
            vn1 = (cutlass.Float32(r_v_bf16_vec[1]) - s1) * r_beta
            vn2 = (cutlass.Float32(r_v_bf16_vec[2]) - s2) * r_beta
            vn3 = (cutlass.Float32(r_v_bf16_vec[3]) - s3) * r_beta
            vn4 = (cutlass.Float32(r_v_bf16_vec[4]) - s4) * r_beta
            vn5 = (cutlass.Float32(r_v_bf16_vec[5]) - s5) * r_beta
            vn6 = (cutlass.Float32(r_v_bf16_vec[6]) - s6) * r_beta
            vn7 = (cutlass.Float32(r_v_bf16_vec[7]) - s7) * r_beta

            # Step 4: Rank-1 update + output dot products h@q using fma_packed_f32x2
            o0 = 0.0
            o1 = 0.0
            o2 = 0.0
            o3 = 0.0
            o4 = 0.0
            o5 = 0.0
            o6 = 0.0
            o7 = 0.0
            o0b = 0.0
            o1b = 0.0
            o2b = 0.0
            o3b = 0.0
            o4b = 0.0
            o5b = 0.0
            o6b = 0.0
            o7b = 0.0
            for i in cutlass.range_constexpr(0, vec_size, 2):
                # Rank-1 update: h += k * vn (paired FMA)
                if cutlass.const_expr(use_packed_fma):
                    r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_k[i], r_k[i + 1]),
                        src_b=(vn0, vn0),
                        src_c=(r_h[0, i], r_h[0, i + 1]),
                    )
                    r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_k[i], r_k[i + 1]),
                        src_b=(vn1, vn1),
                        src_c=(r_h[1, i], r_h[1, i + 1]),
                    )
                    r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_k[i], r_k[i + 1]),
                        src_b=(vn2, vn2),
                        src_c=(r_h[2, i], r_h[2, i + 1]),
                    )
                    r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_k[i], r_k[i + 1]),
                        src_b=(vn3, vn3),
                        src_c=(r_h[3, i], r_h[3, i + 1]),
                    )
                    r_h[4, i], r_h[4, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_k[i], r_k[i + 1]),
                        src_b=(vn4, vn4),
                        src_c=(r_h[4, i], r_h[4, i + 1]),
                    )
                    r_h[5, i], r_h[5, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_k[i], r_k[i + 1]),
                        src_b=(vn5, vn5),
                        src_c=(r_h[5, i], r_h[5, i + 1]),
                    )
                    r_h[6, i], r_h[6, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_k[i], r_k[i + 1]),
                        src_b=(vn6, vn6),
                        src_c=(r_h[6, i], r_h[6, i + 1]),
                    )
                    r_h[7, i], r_h[7, i + 1] = cute.arch.fma_packed_f32x2(
                        src_a=(r_k[i], r_k[i + 1]),
                        src_b=(vn7, vn7),
                        src_c=(r_h[7, i], r_h[7, i + 1]),
                    )
                else:
                    r_h[0, i], r_h[0, i + 1] = fma_pair(
                        r_k[i], r_k[i + 1], vn0, vn0, r_h[0, i], r_h[0, i + 1]
                    )
                    r_h[1, i], r_h[1, i + 1] = fma_pair(
                        r_k[i], r_k[i + 1], vn1, vn1, r_h[1, i], r_h[1, i + 1]
                    )
                    r_h[2, i], r_h[2, i + 1] = fma_pair(
                        r_k[i], r_k[i + 1], vn2, vn2, r_h[2, i], r_h[2, i + 1]
                    )
                    r_h[3, i], r_h[3, i + 1] = fma_pair(
                        r_k[i], r_k[i + 1], vn3, vn3, r_h[3, i], r_h[3, i + 1]
                    )
                    r_h[4, i], r_h[4, i + 1] = fma_pair(
                        r_k[i], r_k[i + 1], vn4, vn4, r_h[4, i], r_h[4, i + 1]
                    )
                    r_h[5, i], r_h[5, i + 1] = fma_pair(
                        r_k[i], r_k[i + 1], vn5, vn5, r_h[5, i], r_h[5, i + 1]
                    )
                    r_h[6, i], r_h[6, i + 1] = fma_pair(
                        r_k[i], r_k[i + 1], vn6, vn6, r_h[6, i], r_h[6, i + 1]
                    )
                    r_h[7, i], r_h[7, i + 1] = fma_pair(
                        r_k[i], r_k[i + 1], vn7, vn7, r_h[7, i], r_h[7, i + 1]
                    )
                # Output dot product: o += h * q (paired FMA)
                if cutlass.const_expr(use_packed_fma):
                    o0, o0b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[0, i], r_h[0, i + 1]),
                        src_b=(r_q[i], r_q[i + 1]),
                        src_c=(o0, o0b),
                    )
                    o1, o1b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[1, i], r_h[1, i + 1]),
                        src_b=(r_q[i], r_q[i + 1]),
                        src_c=(o1, o1b),
                    )
                    o2, o2b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[2, i], r_h[2, i + 1]),
                        src_b=(r_q[i], r_q[i + 1]),
                        src_c=(o2, o2b),
                    )
                    o3, o3b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[3, i], r_h[3, i + 1]),
                        src_b=(r_q[i], r_q[i + 1]),
                        src_c=(o3, o3b),
                    )
                    o4, o4b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[4, i], r_h[4, i + 1]),
                        src_b=(r_q[i], r_q[i + 1]),
                        src_c=(o4, o4b),
                    )
                    o5, o5b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[5, i], r_h[5, i + 1]),
                        src_b=(r_q[i], r_q[i + 1]),
                        src_c=(o5, o5b),
                    )
                    o6, o6b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[6, i], r_h[6, i + 1]),
                        src_b=(r_q[i], r_q[i + 1]),
                        src_c=(o6, o6b),
                    )
                    o7, o7b = cute.arch.fma_packed_f32x2(
                        src_a=(r_h[7, i], r_h[7, i + 1]),
                        src_b=(r_q[i], r_q[i + 1]),
                        src_c=(o7, o7b),
                    )
                else:
                    o0, o0b = fma_pair(
                        r_h[0, i], r_h[0, i + 1], r_q[i], r_q[i + 1], o0, o0b
                    )
                    o1, o1b = fma_pair(
                        r_h[1, i], r_h[1, i + 1], r_q[i], r_q[i + 1], o1, o1b
                    )
                    o2, o2b = fma_pair(
                        r_h[2, i], r_h[2, i + 1], r_q[i], r_q[i + 1], o2, o2b
                    )
                    o3, o3b = fma_pair(
                        r_h[3, i], r_h[3, i + 1], r_q[i], r_q[i + 1], o3, o3b
                    )
                    o4, o4b = fma_pair(
                        r_h[4, i], r_h[4, i + 1], r_q[i], r_q[i + 1], o4, o4b
                    )
                    o5, o5b = fma_pair(
                        r_h[5, i], r_h[5, i + 1], r_q[i], r_q[i + 1], o5, o5b
                    )
                    o6, o6b = fma_pair(
                        r_h[6, i], r_h[6, i + 1], r_q[i], r_q[i + 1], o6, o6b
                    )
                    o7, o7b = fma_pair(
                        r_h[7, i], r_h[7, i + 1], r_q[i], r_q[i + 1], o7, o7b
                    )
            # Combine paired accumulators
            o0 = o0 + o0b
            o1 = o1 + o1b
            o2 = o2 + o2b
            o3 = o3 + o3b
            o4 = o4 + o4b
            o5 = o5 + o5b
            o6 = o6 + o6b
            o7 = o7 + o7b

            # Interleaved butterfly reduction for all 8 o-values (better ILP than sequential warp_reduction_sum)
            for offset in [16, 8, 4, 2, 1]:
                o0 += cute.arch.shuffle_sync_bfly(
                    o0, offset=offset, mask=-1, mask_and_clamp=31
                )
                o1 += cute.arch.shuffle_sync_bfly(
                    o1, offset=offset, mask=-1, mask_and_clamp=31
                )
                o2 += cute.arch.shuffle_sync_bfly(
                    o2, offset=offset, mask=-1, mask_and_clamp=31
                )
                o3 += cute.arch.shuffle_sync_bfly(
                    o3, offset=offset, mask=-1, mask_and_clamp=31
                )
                o4 += cute.arch.shuffle_sync_bfly(
                    o4, offset=offset, mask=-1, mask_and_clamp=31
                )
                o5 += cute.arch.shuffle_sync_bfly(
                    o5, offset=offset, mask=-1, mask_and_clamp=31
                )
                o6 += cute.arch.shuffle_sync_bfly(
                    o6, offset=offset, mask=-1, mask_and_clamp=31
                )
                o7 += cute.arch.shuffle_sync_bfly(
                    o7, offset=offset, mask=-1, mask_and_clamp=31
                )

            # Write output: pack into BF16 reg tensor and vectorized store
            if lane_in_group == 0:
                r_o_bf16_vec[0] = cutlass.BFloat16(o0)
                r_o_bf16_vec[1] = cutlass.BFloat16(o1)
                r_o_bf16_vec[2] = cutlass.BFloat16(o2)
                r_o_bf16_vec[3] = cutlass.BFloat16(o3)
                r_o_bf16_vec[4] = cutlass.BFloat16(o4)
                r_o_bf16_vec[5] = cutlass.BFloat16(o5)
                r_o_bf16_vec[6] = cutlass.BFloat16(o6)
                r_o_bf16_vec[7] = cutlass.BFloat16(o7)
                ot_slice = cute.local_tile(
                    o, (1, 1, 1, ILP_ROWS), (i_n, i_t, i_hv, v_base // ILP_ROWS)
                )
                cute.autovec_copy(r_o_bf16_vec, ot_slice)

            # Write updated H back to GMEM: FP32 regs -> BF16 regs -> GMEM BF16 (vectorized)
            for i in cutlass.range_constexpr(vec_size):
                r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                r_hb3[i] = cutlass.BFloat16(r_h[3, i])
                r_hb4[i] = cutlass.BFloat16(r_h[4, i])
                r_hb5[i] = cutlass.BFloat16(r_h[5, i])
                r_hb6[i] = cutlass.BFloat16(r_h[6, i])
                r_hb7[i] = cutlass.BFloat16(r_h[7, i])
            cute.autovec_copy(r_hb0, ht0)
            cute.autovec_copy(r_hb1, ht1)
            cute.autovec_copy(r_hb2, ht2)
            cute.autovec_copy(r_hb3, ht3)
            cute.autovec_copy(r_hb4, ht4)
            cute.autovec_copy(r_hb5, ht5)
            cute.autovec_copy(r_hb6, ht6)
            cute.autovec_copy(r_hb7, ht7)


# ==============================================================================
# KERNEL: MTP (Multiple Token Processing) with BF16 state
# ==============================================================================
# Architecture (adapted from gdn_verify_kernel_mtp_original in gdn_decode.py):
# - Grid: (B * HV * num_v_tiles, 1, 1) - each block handles one TILE_V chunk
# - 128 threads = 4 groups of 32 threads (full warps)
# - Each group processes tile_v/4 V-rows
# - H loaded as BF16 from GMEM, computed in FP32, stored back as BF16
# - Processes T tokens sequentially, keeping h in FP32 registers
# - Optional: cache intermediate states, disable state update

MTP_TILE_K = 128
MTP_NUM_THREADS = 128
MTP_VEC_SIZE = 4  # 32 threads per group × 4 = 128 K elements
MTP_ILP_ROWS = 8  # Process 8 V-rows simultaneously per group iteration


@cute.kernel
def gdn_decode_bf16state_mtp_kernel(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] as BF16
    intermediate_states: cute.Tensor,  # [pool_size * T * HV, V, K] as BF16 (or dummy)
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, T, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    b: cute.Tensor,  # [B, T, HV]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices (read)
    h0_out_indices: cute.Tensor,  # [B] - output state indices (write)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
):
    """
    ILP-optimized MTP kernel for BF16 state: processes T tokens sequentially.
    Each block handles one tile_v chunk of V rows.
    H is loaded as BF16, computed in FP32, stored back as BF16.
    Uses 8-row ILP with fma_packed_f32x2 (Blackwell) / scalar FMA (Hopper) with compile-time dispatch.
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # 4 groups (= 4 warps), each full warp of 32 threads
    threads_per_group: cutlass.Constexpr[int] = 32  # noqa: F841
    num_groups: cutlass.Constexpr[int] = 4
    group_idx = warp_idx
    lane_in_group = lane_id

    batch_idx, _, _ = cute.arch.block_idx()

    # Decode block index: (i_n, i_hv, i_v) from batch_idx
    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    # Get initial state index for this batch
    cache_idx = h0_indices[i_n]

    # Load A_log and dt_bias once
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])

    # For T>1: shared SMEM for q/k (one copy, all warps read)
    # Precomputed in parallel: warp i handles token i (barrier before inner loop)
    # For T>2: also cache g/beta in SMEM (saves redundant exp/log across row_oct iterations)
    # For T=1: no SMEM needed (inline compute is faster)
    if cutlass.const_expr(T > 1):
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
        )
        sK = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
        )
        # Always allocate sGB (SMEM variable must exist for all T>1 paths)
        sGB = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, 2), stride=(2, 1)), 16
        )

    # Register arrays for computation - ILP=8 rows of vec_size=4 each
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((MTP_ILP_ROWS, vec_size), stride=(vec_size, 1)),
        cutlass.Float32,
    )
    # BF16 register tensors for vectorized loading q, k
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    # 8 separate BF16 register tensors for vectorized H loading (autovec_copy)
    r_hb0 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb1 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb2 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb3 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb5 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb6 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb7 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    # BF16 register tensors for vectorized V load and output store (8 elements)
    r_v_bf16_vec = cute.make_rmem_tensor(
        cute.make_layout((MTP_ILP_ROWS,), stride=(1,)), cutlass.BFloat16
    )
    r_o_bf16_vec = cute.make_rmem_tensor(
        cute.make_layout((MTP_ILP_ROWS,), stride=(1,)), cutlass.BFloat16
    )

    # Redirect padding entries (cache_idx < 0) to null buffer (slot 0)
    if cache_idx < 0:
        cache_idx = cutlass.Int32(0)

    # Process all batch entries (padding slots redirected to slot 0 above)
    if cache_idx >= 0:
        k_start = lane_in_group * vec_size

        # For T>1: parallel precompute q, k into shared SMEM
        # With 4 warps, each pass precomputes up to 4 tokens in parallel.
        # For T<=4: 1 pass. For T=5..8: 2 passes. General: ceil(T/4) passes.
        if cutlass.const_expr(T > 1):
            num_precompute_passes: cutlass.Constexpr[int] = (
                T + num_groups - 1
            ) // num_groups
            for pass_idx in cutlass.range_constexpr(num_precompute_passes):
                i_t_pre = pass_idx * num_groups + group_idx
                if i_t_pre < T:
                    q_tile_pre = cute.local_tile(
                        q, (1, 1, 1, vec_size), (i_n, i_t_pre, i_h, lane_in_group)
                    )
                    k_tile_pre = cute.local_tile(
                        k, (1, 1, 1, vec_size), (i_n, i_t_pre, i_h, lane_in_group)
                    )
                    cute.autovec_copy(q_tile_pre, r_q_bf16)
                    cute.autovec_copy(k_tile_pre, r_k_bf16)

                    for i in cutlass.range_constexpr(vec_size):
                        r_q[i] = cutlass.Float32(r_q_bf16[i])
                        r_k[i] = cutlass.Float32(r_k_bf16[i])

                    if cutlass.const_expr(use_qk_l2norm):
                        sum_q = 0.0
                        sum_k = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_q += r_q[i] * r_q[i]
                            sum_k += r_k[i] * r_k[i]
                        for offset in [16, 8, 4, 2, 1]:
                            sum_q += cute.arch.shuffle_sync_bfly(
                                sum_q, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_k += cute.arch.shuffle_sync_bfly(
                                sum_k, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        inv_norm_q_scaled = (
                            cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                        )
                        inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = r_q[i] * inv_norm_q_scaled
                            r_k[i] = r_k[i] * inv_norm_k
                    else:
                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = r_q[i] * scale

                    # Write to shared SMEM (all active warps write different token slots)
                    for i in cutlass.range_constexpr(vec_size):
                        sQ[(i_t_pre, k_start + i)] = r_q[i]
                        sK[(i_t_pre, k_start + i)] = r_k[i]

                    # Precompute g/beta for the assigned token - only for T>2
                    if cutlass.const_expr(T > 2):
                        r_a_pre = cutlass.Float32(a[i_n, i_t_pre, i_hv])
                        r_b_pre = cutlass.Float32(b[i_n, i_t_pre, i_hv])
                        x_pre = r_a_pre + r_dt_bias
                        beta_x_pre = softplus_beta * x_pre
                        exp_beta_x_pre = cute.exp(beta_x_pre, fastmath=True)
                        softplus_val_pre = (
                            cutlass.Float32(1.0) / softplus_beta
                        ) * cute.log(
                            cutlass.Float32(1.0) + exp_beta_x_pre, fastmath=True
                        )
                        use_softplus_pre = (
                            cutlass.Float32(1.0)
                            if beta_x_pre <= softplus_threshold
                            else cutlass.Float32(0.0)
                        )
                        softplus_x_pre = (
                            use_softplus_pre * softplus_val_pre
                            + (cutlass.Float32(1.0) - use_softplus_pre) * x_pre
                        )
                        r_g_value_pre = (
                            -cute.exp(r_A_log, fastmath=True) * softplus_x_pre
                        )
                        r_beta_pre = cutlass.Float32(1.0) / (
                            cutlass.Float32(1.0) + cute.exp(-r_b_pre, fastmath=True)
                        )
                        r_g_pre = cute.exp(r_g_value_pre, fastmath=True)
                        if lane_in_group == 0:
                            sGB[(i_t_pre, 0)] = r_g_pre
                            sGB[(i_t_pre, 1)] = r_beta_pre

                # Barrier after each pass: all warps must finish writing before next pass reads/writes
                cute.arch.barrier()

        # Each group handles tile_v/num_groups V rows, 8 at a time (ILP=8)
        flat_state_idx = cache_idx * HV + i_hv
        write_cache_idx = h0_out_indices[i_n]
        # Redirect negative write indices to null buffer (slot 0),
        # matching the read-side redirect above.
        if write_cache_idx < 0:
            write_cache_idx = cutlass.Int32(0)
        flat_write_idx = write_cache_idx * HV + i_hv
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        eighth_rows: cutlass.Constexpr[int] = rows_per_group // MTP_ILP_ROWS

        # Pre-declare loop-carried variables for dynamic loop compatibility (T>1)
        sum_q = cutlass.Float32(0.0)
        sum_k = cutlass.Float32(0.0)
        inv_norm_q_scaled = cutlass.Float32(1.0)
        inv_norm_k = cutlass.Float32(1.0)

        # For T>1: don't unroll row_oct loop (reduces code size for better icache)
        # For T=1: fully unroll row_oct loop (no code size issue, max performance)
        for row_oct in cutlass.range(eighth_rows, unroll=1, unroll_full=(T <= 1)):
            v_base = i_v * tile_v + group_idx * rows_per_group + row_oct * MTP_ILP_ROWS
            v0 = v_base
            v1 = v_base + 1
            v2 = v_base + 2
            v3 = v_base + 3
            v4 = v_base + 4
            v5 = v_base + 5
            v6 = v_base + 6
            v7 = v_base + 7

            # Load h for ALL 8 V-rows: GMEM BF16 -> BF16 regs (vectorized) -> FP32 regs
            ht0 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v0, lane_in_group)
            )
            ht1 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v1, lane_in_group)
            )
            ht2 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v2, lane_in_group)
            )
            ht3 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v3, lane_in_group)
            )
            ht4 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v4, lane_in_group)
            )
            ht5 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v5, lane_in_group)
            )
            ht6 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v6, lane_in_group)
            )
            ht7 = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, v7, lane_in_group)
            )
            cute.autovec_copy(ht0, r_hb0)
            cute.autovec_copy(ht1, r_hb1)
            cute.autovec_copy(ht2, r_hb2)
            cute.autovec_copy(ht3, r_hb3)
            cute.autovec_copy(ht4, r_hb4)
            cute.autovec_copy(ht5, r_hb5)
            cute.autovec_copy(ht6, r_hb6)
            cute.autovec_copy(ht7, r_hb7)

            # Convert BF16 -> FP32 for all 8 rows
            for i in cutlass.range_constexpr(vec_size):
                r_h[0, i] = cutlass.Float32(r_hb0[i])
                r_h[1, i] = cutlass.Float32(r_hb1[i])
                r_h[2, i] = cutlass.Float32(r_hb2[i])
                r_h[3, i] = cutlass.Float32(r_hb3[i])
                r_h[4, i] = cutlass.Float32(r_hb4[i])
                r_h[5, i] = cutlass.Float32(r_hb5[i])
                r_h[6, i] = cutlass.Float32(r_hb6[i])
                r_h[7, i] = cutlass.Float32(r_hb7[i])

            # Process all T time steps with h in FP32 registers
            # For T>1: use dynamic timestep loop to reduce code size (saves icache)
            # For T=1: fully unroll timestep loop (minimal overhead, no loop counter)
            for i_t in cutlass.range(T, unroll=1, unroll_full=(T <= 1)):
                # Load q, k, g, beta - conditionally from SMEM or inline
                if cutlass.const_expr(T > 1):
                    # T>1: read q,k from shared SMEM (pre-computed in parallel)
                    sQ_tile = cute.local_tile(sQ, (1, vec_size), (i_t, lane_in_group))
                    sK_tile = cute.local_tile(sK, (1, vec_size), (i_t, lane_in_group))
                    cute.autovec_copy(sQ_tile, r_q)
                    cute.autovec_copy(sK_tile, r_k)
                    if cutlass.const_expr(T > 2):
                        # T>2: read pre-computed g, beta from shared SMEM
                        r_g = sGB[(i_t, 0)]
                        r_beta = sGB[(i_t, 1)]
                    else:
                        # T=2: compute g, beta inline (avoids SMEM read latency)
                        r_a_val = cutlass.Float32(a[i_n, i_t, i_hv])
                        r_b_val = cutlass.Float32(b[i_n, i_t, i_hv])
                        x_val = r_a_val + r_dt_bias
                        beta_x_val = softplus_beta * x_val
                        exp_beta_x_val = cute.exp(beta_x_val, fastmath=True)
                        softplus_val_v = (
                            cutlass.Float32(1.0) / softplus_beta
                        ) * cute.log(
                            cutlass.Float32(1.0) + exp_beta_x_val, fastmath=True
                        )
                        use_softplus_v = (
                            cutlass.Float32(1.0)
                            if beta_x_val <= softplus_threshold
                            else cutlass.Float32(0.0)
                        )
                        softplus_x_v = (
                            use_softplus_v * softplus_val_v
                            + (cutlass.Float32(1.0) - use_softplus_v) * x_val
                        )
                        r_g_value_v = -cute.exp(r_A_log, fastmath=True) * softplus_x_v
                        r_beta = cutlass.Float32(1.0) / (
                            cutlass.Float32(1.0) + cute.exp(-r_b_val, fastmath=True)
                        )
                        r_g = cute.exp(r_g_value_v, fastmath=True)
                else:
                    # T=1: compute inline (no SMEM overhead)
                    q_tile_t = cute.local_tile(
                        q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
                    )
                    k_tile_t = cute.local_tile(
                        k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
                    )
                    cute.autovec_copy(q_tile_t, r_q_bf16)
                    cute.autovec_copy(k_tile_t, r_k_bf16)

                    for i in cutlass.range_constexpr(vec_size):
                        r_q[i] = cutlass.Float32(r_q_bf16[i])
                        r_k[i] = cutlass.Float32(r_k_bf16[i])

                    if cutlass.const_expr(use_qk_l2norm):
                        sum_q = cutlass.Float32(0.0)
                        sum_k = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(vec_size):
                            sum_q += r_q[i] * r_q[i]
                            sum_k += r_k[i] * r_k[i]
                        for offset in [16, 8, 4, 2, 1]:
                            sum_q += cute.arch.shuffle_sync_bfly(
                                sum_q, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_k += cute.arch.shuffle_sync_bfly(
                                sum_k, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        inv_norm_q_scaled = (
                            cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                        )
                        inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = r_q[i] * inv_norm_q_scaled
                            r_k[i] = r_k[i] * inv_norm_k
                    else:
                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = r_q[i] * scale

                    r_a_val = cutlass.Float32(a[i_n, i_t, i_hv])
                    r_b_val = cutlass.Float32(b[i_n, i_t, i_hv])
                    x_val = r_a_val + r_dt_bias
                    beta_x_val = softplus_beta * x_val
                    exp_beta_x_val = cute.exp(beta_x_val, fastmath=True)
                    softplus_val_v = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                        cutlass.Float32(1.0) + exp_beta_x_val, fastmath=True
                    )
                    use_softplus_v = (
                        cutlass.Float32(1.0)
                        if beta_x_val <= softplus_threshold
                        else cutlass.Float32(0.0)
                    )
                    softplus_x_v = (
                        use_softplus_v * softplus_val_v
                        + (cutlass.Float32(1.0) - use_softplus_v) * x_val
                    )
                    r_g_value_v = -cute.exp(r_A_log, fastmath=True) * softplus_x_v
                    r_beta = cutlass.Float32(1.0) / (
                        cutlass.Float32(1.0) + cute.exp(-r_b_val, fastmath=True)
                    )
                    r_g = cute.exp(r_g_value_v, fastmath=True)

                # Fused: decay h, dot product h@k with conditional dispatch
                s0 = 0.0
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                s4 = 0.0
                s5 = 0.0
                s6 = 0.0
                s7 = 0.0
                s0b = 0.0
                s1b = 0.0
                s2b = 0.0
                s3b = 0.0
                s4b = 0.0
                s5b = 0.0
                s6b = 0.0
                s7b = 0.0
                for i in cutlass.range_constexpr(0, vec_size, 2):
                    # Convert + decay for pairs of elements
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[0, i], r_h[0, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[1, i], r_h[1, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[2, i], r_h[2, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[3, i], r_h[3, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[4, i], r_h[4, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[4, i], r_h[4, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[5, i], r_h[5, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[5, i], r_h[5, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[6, i], r_h[6, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[6, i], r_h[6, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                        r_h[7, i], r_h[7, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[7, i], r_h[7, i + 1]),
                            src_b=(r_g, r_g),
                            src_c=(cutlass.Float32(0.0), cutlass.Float32(0.0)),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair_mul(
                            r_h[0, i], r_h[0, i + 1], r_g, r_g
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair_mul(
                            r_h[1, i], r_h[1, i + 1], r_g, r_g
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair_mul(
                            r_h[2, i], r_h[2, i + 1], r_g, r_g
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair_mul(
                            r_h[3, i], r_h[3, i + 1], r_g, r_g
                        )
                        r_h[4, i], r_h[4, i + 1] = fma_pair_mul(
                            r_h[4, i], r_h[4, i + 1], r_g, r_g
                        )
                        r_h[5, i], r_h[5, i + 1] = fma_pair_mul(
                            r_h[5, i], r_h[5, i + 1], r_g, r_g
                        )
                        r_h[6, i], r_h[6, i + 1] = fma_pair_mul(
                            r_h[6, i], r_h[6, i + 1], r_g, r_g
                        )
                        r_h[7, i], r_h[7, i + 1] = fma_pair_mul(
                            r_h[7, i], r_h[7, i + 1], r_g, r_g
                        )
                    # Dot product h@k using paired FMA
                    if cutlass.const_expr(use_packed_fma):
                        s0, s0b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[0, i], r_h[0, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s0, s0b),
                        )
                        s1, s1b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[1, i], r_h[1, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s1, s1b),
                        )
                        s2, s2b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[2, i], r_h[2, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s2, s2b),
                        )
                        s3, s3b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[3, i], r_h[3, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s3, s3b),
                        )
                        s4, s4b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[4, i], r_h[4, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s4, s4b),
                        )
                        s5, s5b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[5, i], r_h[5, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s5, s5b),
                        )
                        s6, s6b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[6, i], r_h[6, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s6, s6b),
                        )
                        s7, s7b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[7, i], r_h[7, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(s7, s7b),
                        )
                    else:
                        s0, s0b = fma_pair(
                            r_h[0, i], r_h[0, i + 1], r_k[i], r_k[i + 1], s0, s0b
                        )
                        s1, s1b = fma_pair(
                            r_h[1, i], r_h[1, i + 1], r_k[i], r_k[i + 1], s1, s1b
                        )
                        s2, s2b = fma_pair(
                            r_h[2, i], r_h[2, i + 1], r_k[i], r_k[i + 1], s2, s2b
                        )
                        s3, s3b = fma_pair(
                            r_h[3, i], r_h[3, i + 1], r_k[i], r_k[i + 1], s3, s3b
                        )
                        s4, s4b = fma_pair(
                            r_h[4, i], r_h[4, i + 1], r_k[i], r_k[i + 1], s4, s4b
                        )
                        s5, s5b = fma_pair(
                            r_h[5, i], r_h[5, i + 1], r_k[i], r_k[i + 1], s5, s5b
                        )
                        s6, s6b = fma_pair(
                            r_h[6, i], r_h[6, i + 1], r_k[i], r_k[i + 1], s6, s6b
                        )
                        s7, s7b = fma_pair(
                            r_h[7, i], r_h[7, i + 1], r_k[i], r_k[i + 1], s7, s7b
                        )
                # Combine paired accumulators
                s0 = s0 + s0b
                s1 = s1 + s1b
                s2 = s2 + s2b
                s3 = s3 + s3b
                s4 = s4 + s4b
                s5 = s5 + s5b
                s6 = s6 + s6b
                s7 = s7 + s7b

                # Interleaved butterfly reduction for 8 s-values
                for offset in [16, 8, 4, 2, 1]:
                    s0 += cute.arch.shuffle_sync_bfly(
                        s0, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s1 += cute.arch.shuffle_sync_bfly(
                        s1, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s2 += cute.arch.shuffle_sync_bfly(
                        s2, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s3 += cute.arch.shuffle_sync_bfly(
                        s3, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s4 += cute.arch.shuffle_sync_bfly(
                        s4, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s5 += cute.arch.shuffle_sync_bfly(
                        s5, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s6 += cute.arch.shuffle_sync_bfly(
                        s6, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    s7 += cute.arch.shuffle_sync_bfly(
                        s7, offset=offset, mask=-1, mask_and_clamp=31
                    )

                # Delta rule: v_new = (v - sum_hk) * beta - vectorized V load
                vt_slice = cute.local_tile(
                    v, (1, 1, 1, MTP_ILP_ROWS), (i_n, i_t, i_hv, v_base // MTP_ILP_ROWS)
                )
                cute.autovec_copy(vt_slice, r_v_bf16_vec)
                vn0 = (cutlass.Float32(r_v_bf16_vec[0]) - s0) * r_beta
                vn1 = (cutlass.Float32(r_v_bf16_vec[1]) - s1) * r_beta
                vn2 = (cutlass.Float32(r_v_bf16_vec[2]) - s2) * r_beta
                vn3 = (cutlass.Float32(r_v_bf16_vec[3]) - s3) * r_beta
                vn4 = (cutlass.Float32(r_v_bf16_vec[4]) - s4) * r_beta
                vn5 = (cutlass.Float32(r_v_bf16_vec[5]) - s5) * r_beta
                vn6 = (cutlass.Float32(r_v_bf16_vec[6]) - s6) * r_beta
                vn7 = (cutlass.Float32(r_v_bf16_vec[7]) - s7) * r_beta

                # Rank-1 update + output dot product h@q with conditional dispatch
                o0 = 0.0
                o1 = 0.0
                o2 = 0.0
                o3 = 0.0
                o4 = 0.0
                o5 = 0.0
                o6 = 0.0
                o7 = 0.0
                o0b = 0.0
                o1b = 0.0
                o2b = 0.0
                o3b = 0.0
                o4b = 0.0
                o5b = 0.0
                o6b = 0.0
                o7b = 0.0
                for i in cutlass.range_constexpr(0, vec_size, 2):
                    # Rank-1 update: h += k * vn (paired FMA)
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn0, vn0),
                            src_c=(r_h[0, i], r_h[0, i + 1]),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn1, vn1),
                            src_c=(r_h[1, i], r_h[1, i + 1]),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn2, vn2),
                            src_c=(r_h[2, i], r_h[2, i + 1]),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn3, vn3),
                            src_c=(r_h[3, i], r_h[3, i + 1]),
                        )
                        r_h[4, i], r_h[4, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn4, vn4),
                            src_c=(r_h[4, i], r_h[4, i + 1]),
                        )
                        r_h[5, i], r_h[5, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn5, vn5),
                            src_c=(r_h[5, i], r_h[5, i + 1]),
                        )
                        r_h[6, i], r_h[6, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn6, vn6),
                            src_c=(r_h[6, i], r_h[6, i + 1]),
                        )
                        r_h[7, i], r_h[7, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vn7, vn7),
                            src_c=(r_h[7, i], r_h[7, i + 1]),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vn0, vn0, r_h[0, i], r_h[0, i + 1]
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vn1, vn1, r_h[1, i], r_h[1, i + 1]
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vn2, vn2, r_h[2, i], r_h[2, i + 1]
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vn3, vn3, r_h[3, i], r_h[3, i + 1]
                        )
                        r_h[4, i], r_h[4, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vn4, vn4, r_h[4, i], r_h[4, i + 1]
                        )
                        r_h[5, i], r_h[5, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vn5, vn5, r_h[5, i], r_h[5, i + 1]
                        )
                        r_h[6, i], r_h[6, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vn6, vn6, r_h[6, i], r_h[6, i + 1]
                        )
                        r_h[7, i], r_h[7, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vn7, vn7, r_h[7, i], r_h[7, i + 1]
                        )
                    # Output dot product: o += h * q (paired FMA)
                    if cutlass.const_expr(use_packed_fma):
                        o0, o0b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[0, i], r_h[0, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o0, o0b),
                        )
                        o1, o1b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[1, i], r_h[1, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o1, o1b),
                        )
                        o2, o2b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[2, i], r_h[2, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o2, o2b),
                        )
                        o3, o3b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[3, i], r_h[3, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o3, o3b),
                        )
                        o4, o4b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[4, i], r_h[4, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o4, o4b),
                        )
                        o5, o5b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[5, i], r_h[5, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o5, o5b),
                        )
                        o6, o6b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[6, i], r_h[6, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o6, o6b),
                        )
                        o7, o7b = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[7, i], r_h[7, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(o7, o7b),
                        )
                    else:
                        o0, o0b = fma_pair(
                            r_h[0, i], r_h[0, i + 1], r_q[i], r_q[i + 1], o0, o0b
                        )
                        o1, o1b = fma_pair(
                            r_h[1, i], r_h[1, i + 1], r_q[i], r_q[i + 1], o1, o1b
                        )
                        o2, o2b = fma_pair(
                            r_h[2, i], r_h[2, i + 1], r_q[i], r_q[i + 1], o2, o2b
                        )
                        o3, o3b = fma_pair(
                            r_h[3, i], r_h[3, i + 1], r_q[i], r_q[i + 1], o3, o3b
                        )
                        o4, o4b = fma_pair(
                            r_h[4, i], r_h[4, i + 1], r_q[i], r_q[i + 1], o4, o4b
                        )
                        o5, o5b = fma_pair(
                            r_h[5, i], r_h[5, i + 1], r_q[i], r_q[i + 1], o5, o5b
                        )
                        o6, o6b = fma_pair(
                            r_h[6, i], r_h[6, i + 1], r_q[i], r_q[i + 1], o6, o6b
                        )
                        o7, o7b = fma_pair(
                            r_h[7, i], r_h[7, i + 1], r_q[i], r_q[i + 1], o7, o7b
                        )
                # Combine paired accumulators
                o0 = o0 + o0b
                o1 = o1 + o1b
                o2 = o2 + o2b
                o3 = o3 + o3b
                o4 = o4 + o4b
                o5 = o5 + o5b
                o6 = o6 + o6b
                o7 = o7 + o7b

                # Start FP32→BF16 conversion for intermediate state BEFORE shuffles
                # (overlaps conversion with shuffle pipeline)
                if cutlass.const_expr(cache_intermediate_states):
                    for i in cutlass.range_constexpr(vec_size):
                        r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                        r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                        r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                        r_hb3[i] = cutlass.BFloat16(r_h[3, i])
                        r_hb4[i] = cutlass.BFloat16(r_h[4, i])
                        r_hb5[i] = cutlass.BFloat16(r_h[5, i])
                        r_hb6[i] = cutlass.BFloat16(r_h[6, i])
                        r_hb7[i] = cutlass.BFloat16(r_h[7, i])

                # Write intermediate state BEFORE output shuffles (issue stores early to overlap with shuffles)
                if cutlass.const_expr(cache_intermediate_states):
                    flat_idx = cache_idx * T * HV + i_t * HV + i_hv
                    it0 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, v0, lane_in_group),
                    )
                    it1 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, v1, lane_in_group),
                    )
                    it2 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, v2, lane_in_group),
                    )
                    it3 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, v3, lane_in_group),
                    )
                    it4 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, v4, lane_in_group),
                    )
                    it5 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, v5, lane_in_group),
                    )
                    it6 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, v6, lane_in_group),
                    )
                    it7 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, v7, lane_in_group),
                    )
                    cute.autovec_copy(r_hb0, it0)
                    cute.autovec_copy(r_hb1, it1)
                    cute.autovec_copy(r_hb2, it2)
                    cute.autovec_copy(r_hb3, it3)
                    cute.autovec_copy(r_hb4, it4)
                    cute.autovec_copy(r_hb5, it5)
                    cute.autovec_copy(r_hb6, it6)
                    cute.autovec_copy(r_hb7, it7)

                # Interleaved butterfly reduction for 8 o-values
                for offset in [16, 8, 4, 2, 1]:
                    o0 += cute.arch.shuffle_sync_bfly(
                        o0, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o1 += cute.arch.shuffle_sync_bfly(
                        o1, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o2 += cute.arch.shuffle_sync_bfly(
                        o2, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o3 += cute.arch.shuffle_sync_bfly(
                        o3, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o4 += cute.arch.shuffle_sync_bfly(
                        o4, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o5 += cute.arch.shuffle_sync_bfly(
                        o5, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o6 += cute.arch.shuffle_sync_bfly(
                        o6, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    o7 += cute.arch.shuffle_sync_bfly(
                        o7, offset=offset, mask=-1, mask_and_clamp=31
                    )

                # Write output: vectorized BF16 store
                if lane_in_group == 0:
                    r_o_bf16_vec[0] = cutlass.BFloat16(o0)
                    r_o_bf16_vec[1] = cutlass.BFloat16(o1)
                    r_o_bf16_vec[2] = cutlass.BFloat16(o2)
                    r_o_bf16_vec[3] = cutlass.BFloat16(o3)
                    r_o_bf16_vec[4] = cutlass.BFloat16(o4)
                    r_o_bf16_vec[5] = cutlass.BFloat16(o5)
                    r_o_bf16_vec[6] = cutlass.BFloat16(o6)
                    r_o_bf16_vec[7] = cutlass.BFloat16(o7)
                    ot_slice = cute.local_tile(
                        o,
                        (1, 1, 1, MTP_ILP_ROWS),
                        (i_n, i_t, i_hv, v_base // MTP_ILP_ROWS),
                    )
                    cute.autovec_copy(r_o_bf16_vec, ot_slice)

            # Write final state back as BF16 (if not disabled)
            if cutlass.const_expr(not disable_state_update):
                for i in cutlass.range_constexpr(vec_size):
                    r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                    r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                    r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                    r_hb3[i] = cutlass.BFloat16(r_h[3, i])
                    r_hb4[i] = cutlass.BFloat16(r_h[4, i])
                    r_hb5[i] = cutlass.BFloat16(r_h[5, i])
                    r_hb6[i] = cutlass.BFloat16(r_h[6, i])
                    r_hb7[i] = cutlass.BFloat16(r_h[7, i])
                wt0 = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_write_idx, v0, lane_in_group)
                )
                wt1 = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_write_idx, v1, lane_in_group)
                )
                wt2 = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_write_idx, v2, lane_in_group)
                )
                wt3 = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_write_idx, v3, lane_in_group)
                )
                wt4 = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_write_idx, v4, lane_in_group)
                )
                wt5 = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_write_idx, v5, lane_in_group)
                )
                wt6 = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_write_idx, v6, lane_in_group)
                )
                wt7 = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_write_idx, v7, lane_in_group)
                )
                cute.autovec_copy(r_hb0, wt0)
                cute.autovec_copy(r_hb1, wt1)
                cute.autovec_copy(r_hb2, wt2)
                cute.autovec_copy(r_hb3, wt3)
                cute.autovec_copy(r_hb4, wt4)
                cute.autovec_copy(r_hb5, wt5)
                cute.autovec_copy(r_hb6, wt6)
                cute.autovec_copy(r_hb7, wt7)


# ==============================================================================
# LAUNCH WRAPPER (MTP version)
# ==============================================================================


@cute.jit
def run_gdn_decode_bf16state_mtp(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] BF16
    intermediate_states: cute.Tensor,  # [pool_size * T * HV, V, K] BF16 (or dummy)
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    h0_out_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v_param: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Launch the MTP kernel for BF16 state."""
    tile_v = tile_v_param
    vec_size = MTP_VEC_SIZE
    _, v_dim, _k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    grid_size = B * HV * num_v_tiles

    # SMEM: for T>1 include shared sQ/sK (1 copy) + sGB; T=1 needs minimal
    smem_bytes = 128  # alignment padding
    if T > 1:
        smem_bytes = (
            4 * T * (K + 8)  # sQ: T × (K+8) × 4 bytes (shared, one copy)
            + 4 * T * (K + 8)  # sK: same
            + 4 * T * 2  # sGB: T × 2 × 4 bytes (shared)
            + 128  # alignment padding
        )

    gdn_decode_bf16state_mtp_kernel(
        h0_source,
        intermediate_states,
        vec_size,
        num_v_tiles,
        tile_v,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
        h0_out_indices,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_qk_l2norm,
        disable_state_update,
        cache_intermediate_states,
        use_packed_fma,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[MTP_NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ==============================================================================
# LAUNCH WRAPPER (ILP version)
# ==============================================================================


@cute.jit
def run_gdn_decode_bf16state_ilp(
    h0_source: cute.Tensor,  # [B*HV, V, K] BF16
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    tile_v_param: cutlass.Constexpr[int],
    stream: cuda.CUstream,
):
    """Launch the ILP-optimized kernel for T=1 with large batch sizes."""
    tile_v = tile_v_param
    vec_size = VEC_SIZE_ILP
    _, v_dim, _k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    grid_size = B * HV * num_v_tiles

    # SMEM: minimal (direct GMEM access)
    smem_bytes = 128

    gdn_decode_bf16state_ilp_kernel(
        h0_source,
        vec_size,
        num_v_tiles,
        tile_v,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        H,
        K,
        V,
        use_qk_l2norm,
        use_packed_fma,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_ILP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ==============================================================================
# LAUNCH WRAPPER (original cp.async pipeline version)
# ==============================================================================


@cute.jit
def run_gdn_decode_bf16state_cooprow(
    h0_source: cute.Tensor,  # [B*HV, V, K] BF16
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Launch the diff-approach kernel for T=1."""
    batch_size, v_dim, _k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    # BF16 async copy: 128-bit = 8 BF16 elements per copy
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.BFloat16,
        num_bits_per_copy=128,
    )

    # Thread layout: 8 rows × 16 threads/row = 128 threads
    # 16 threads × 8 BF16 elements = 128 K-elements per row
    # 8 rows = TILE_V rows per copy (covers full tile in one shot)
    thread_layout = cute.make_layout(
        (8, 16),
        stride=(16, 1),
    )
    val_layout = cute.make_layout((1, 8))  # 8 BF16 elements per copy = 128 bits

    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)

    vec_size = TILE_K // 32  # = 4

    # SMEM layout: (TILE_V, TILE_K, NUM_STAGES) in BF16
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    # SMEM: sData (BF16) + sV (FP32) + sOutput (BF16)
    smem_bytes = (
        2 * TILE_V * TILE_K * NUM_STAGES  # sData: BF16
        + 4 * v_dim  # sV: FP32
        + 2 * v_dim  # sOutput: BF16
        + 128  # alignment padding
    )

    gdn_decode_bf16state_cooprow_kernel(
        tiled_copy_load,
        h0_source,
        smem_layout_staged,
        vec_size,
        num_v_tiles,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        H,
        K,
        V,
        use_qk_l2norm,
    ).launch(
        grid=(batch_size * NUM_BLOCKS_PER_STATE, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ==============================================================================
# PUBLIC API
# ==============================================================================
_compiled_kernels: dict = {}
_compiled_kernels_ilp: dict = {}

# Batch size threshold for ILP kernel dispatch
ILP_BATCH_THRESHOLD = 16  # Use ILP kernel for B >= 16

# Number of SMs on target GPU (detected dynamically)
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


def _select_tile_v_for_batch(B: int, HV: int, V: int) -> int:
    """Select optimal tile_v for the ILP kernel based on batch size.

    Goal: maximize GPU occupancy by ensuring enough blocks to fill all SMs.
    Each block handles tile_v V-rows, grid = B * HV * (V / tile_v).
    We want at least ~4 waves (4 * NUM_SMS blocks) for good occupancy,
    since register pressure limits per-SM occupancy.

    tile_v must be a multiple of 32 (4 groups * ILP_ROWS=8) and divide V=128.
    Valid values: 32, 64, 128.
    """
    for tv in [128, 64, 32]:
        num_v_tiles = V // tv
        grid_size = B * HV * num_v_tiles
        # Want at least 4 waves for good occupancy (register pressure limits to ~3 blocks/SM)
        if grid_size >= 4 * NUM_SMS:
            return tv
    return 32  # Minimum tile_v for maximum parallelism


def gated_delta_rule(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    GDN decode T=1 with BF16 state.

    Args:
        A_log: [HV] float32
        a: [B, 1, HV] bf16
        dt_bias: [HV] float32
        q: [B, 1, H, K] bf16
        k: [B, 1, H, K] bf16
        v: [B, 1, HV, V] bf16
        b: [B, 1, HV] bf16
        initial_state_source: [B, HV, V, K] bf16 (modified in-place)
        scale: Optional, default 1/sqrt(K)

    Returns:
        output: [B, 1, HV, V] bf16
    """
    global _compiled_kernels_ilp

    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    B, T, H, K = q.shape
    assert T == 1, f"This kernel only supports T=1, got T={T}"
    HV = v.shape[2]
    V = v.shape[3]
    assert K == 128 and V == 128, f"K and V must be 128, got K={K}, V={V}"
    assert initial_state_source.dtype == torch.bfloat16

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    # Small batch: route through MTP kernel (T=1 path) with identity indices.
    # The cooprow kernel has known correctness issues at small batch sizes (e.g. B=2).
    # The MTP kernel's T=1 path uses the same ILP-style computation and is well-tested.
    if B < ILP_BATCH_THRESHOLD:
        return gated_delta_rule_mtp(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
        )

    output = torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype)

    # Reshape state to [B*HV, V, K]
    h0_source = initial_state_source.reshape(B * HV, V, K)

    q_ = from_dlpack(q, assumed_align=32, enable_tvm_ffi=True)
    k_ = from_dlpack(k, assumed_align=32, enable_tvm_ffi=True)
    v_ = from_dlpack(v, assumed_align=32, enable_tvm_ffi=True)
    a_ = from_dlpack(a, assumed_align=32, enable_tvm_ffi=True)
    b_ = from_dlpack(b, assumed_align=32, enable_tvm_ffi=True)
    A_log_ = from_dlpack(A_log, assumed_align=32, enable_tvm_ffi=True)
    dt_bias_ = from_dlpack(dt_bias, assumed_align=32, enable_tvm_ffi=True)
    h_ = from_dlpack(h0_source, assumed_align=32, enable_tvm_ffi=True)
    o_ = from_dlpack(output, assumed_align=32, enable_tvm_ffi=True)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    major, _ = torch.cuda.get_device_capability(q.device)
    use_packed_fma = major >= 10

    # B >= ILP_BATCH_THRESHOLD (small B handled by MTP path above)
    tile_v = _select_tile_v_for_batch(B, HV, V)
    cache_key = (
        "ilp",
        B,
        H,
        HV,
        K,
        V,
        tile_v,
        scale,
        softplus_beta,
        softplus_threshold,
        use_packed_fma,
    )
    if cache_key not in _compiled_kernels_ilp:
        # Use maxrregcount=64 for smaller tile_v to improve occupancy
        # when grid size is small (fewer waves)
        if tile_v < 128:
            compile_opts = "--enable-tvm-ffi --generate-line-info --opt-level 3 --ptxas-options=-maxrregcount=64"
        else:
            compile_opts = "--enable-tvm-ffi --generate-line-info --opt-level 3"
        _compiled_kernels_ilp[cache_key] = cute.compile(
            run_gdn_decode_bf16state_ilp,
            h_,
            A_log_,
            a_,
            dt_bias_,
            q_,
            k_,
            v_,
            b_,
            o_,
            softplus_beta,
            softplus_threshold,
            scale,
            HV,
            B,
            H,
            K,
            V,
            use_qk_l2norm_in_kernel,
            use_packed_fma,
            tile_v,
            stream,
            options=compile_opts,
        )

    _compiled_kernels_ilp[cache_key](
        h_,
        A_log_,
        a_,
        dt_bias_,
        q_,
        k_,
        v_,
        b_,
        o_,
        stream,
    )

    return output


# ==============================================================================
# MTP PUBLIC API
# ==============================================================================
_compiled_kernels_mtp: dict = {}


def _select_tile_v_for_mtp(B: int, HV: int, V: int, T: int = 1) -> int:
    """Select optimal tile_v for the MTP BF16 kernel based on batch size and T.

    tile_v must be a multiple of MTP_ILP_ROWS * 4 (= 32) and divide V=128.
    Valid values: 32, 64, 128.
    With ILP=8, minimum tile_v = 4 * 8 = 32 (4 groups * 8 ILP_ROWS).

    For large batch sizes, use larger tile_v to reduce block count and overhead.
    """
    for tv in [128, 64, 32]:
        num_v_tiles = V // tv
        grid_size = B * HV * num_v_tiles
        # Want at least 4 waves for good occupancy
        if grid_size >= 4 * NUM_SMS:
            return tv
    return 32  # Minimum tile_v for maximum parallelism


def gated_delta_rule_mtp(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    q: Optional[torch.Tensor] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    b: Optional[torch.Tensor] = None,
    initial_state_source: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    GDN MTP (Multiple Token Processing) with BF16 state.
    Processes T tokens sequentially, keeping h in FP32 registers.
    H state loaded/stored as BF16.

    Args:
        A_log: [HV] float32
        a: [B, T, HV] bf16
        dt_bias: [HV] float32
        q: [B, T, H, K] bf16
        k: [B, T, H, K] bf16
        v: [B, T, HV, V] bf16
        b: [B, T, HV] bf16
        initial_state_source: [pool_size, HV, V, K] bf16
        initial_state_indices: [B] int32 - indices into state pool (read)
        output_state_indices: Optional [B] int32 - indices for writing updated state.
            Defaults to initial_state_indices when None.
        intermediate_states_buffer: Optional [pool_size, T, HV, V, K] bf16
        disable_state_update: bool - if True, don't update initial state
        scale: Optional, default 1/sqrt(K)
        output: Optional pre-allocated output tensor [B, T, HV, V] bf16

    Returns:
        output: [B, T, HV, V] bf16
    """
    global _compiled_kernels_mtp

    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    B, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    pool_size = initial_state_source.shape[0]
    assert K == 128 and V == 128, f"K and V must be 128, got K={K}, V={V}"
    assert initial_state_source.dtype == torch.bfloat16

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if initial_state_indices is None:
        initial_state_indices = torch.arange(B, dtype=torch.int32, device=q.device)

    # Default output indices to read indices
    if output_state_indices is None:
        output_state_indices = initial_state_indices
    elif output_state_indices.dtype != torch.int32:
        output_state_indices = output_state_indices.to(torch.int32)

    if output is None:
        output = torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype)

    # Reshape state to [pool_size * HV, V, K]
    h0_source = initial_state_source.reshape(pool_size * HV, V, K)

    # Handle intermediate states
    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]
        assert cache_steps >= T, (
            f"intermediate_states_buffer dim 1 ({cache_steps}) must be >= T={T}"
        )
        assert intermediate_states_buffer.dtype == torch.bfloat16
        intermediate_states = intermediate_states_buffer.reshape(
            buffer_size * cache_steps * HV, V, K
        )
        if not intermediate_states.is_contiguous():
            intermediate_states = intermediate_states.contiguous()
    else:
        intermediate_states = h0_source[
            :1, :1, :1
        ]  # Reuse existing allocation as dummy

    tile_v = _select_tile_v_for_mtp(B, HV, V, T)

    h_ = from_dlpack(h0_source, assumed_align=32, enable_tvm_ffi=True)
    inter_ = from_dlpack(intermediate_states, assumed_align=32, enable_tvm_ffi=True)
    q_ = from_dlpack(q, assumed_align=32, enable_tvm_ffi=True)
    k_ = from_dlpack(k, assumed_align=32, enable_tvm_ffi=True)
    v_ = from_dlpack(v, assumed_align=32, enable_tvm_ffi=True)
    a_ = from_dlpack(a, assumed_align=32, enable_tvm_ffi=True)
    b_ = from_dlpack(b, assumed_align=32, enable_tvm_ffi=True)
    A_log_ = from_dlpack(A_log, assumed_align=32, enable_tvm_ffi=True)
    dt_bias_ = from_dlpack(dt_bias, assumed_align=32, enable_tvm_ffi=True)
    o_ = from_dlpack(output, assumed_align=32, enable_tvm_ffi=True)
    h0_idx_ = from_dlpack(initial_state_indices, assumed_align=32, enable_tvm_ffi=True)
    h0_out_idx_ = from_dlpack(
        output_state_indices, assumed_align=32, enable_tvm_ffi=True
    )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    major, _ = torch.cuda.get_device_capability(q.device)
    use_packed_fma = major >= 10

    cache_key = (
        "mtp_bf16",
        B,
        T,
        H,
        HV,
        K,
        V,
        pool_size,
        tile_v,
        disable_state_update,
        cache_intermediate_states,
        use_qk_l2norm_in_kernel,
        scale,
        softplus_beta,
        softplus_threshold,
        use_packed_fma,
    )
    if cache_key not in _compiled_kernels_mtp:
        _compiled_kernels_mtp[cache_key] = cute.compile(
            run_gdn_decode_bf16state_mtp,
            h_,
            inter_,
            A_log_,
            a_,
            dt_bias_,
            q_,
            k_,
            v_,
            b_,
            o_,
            h0_idx_,
            h0_out_idx_,
            softplus_beta,
            softplus_threshold,
            scale,
            HV,
            B,
            T,
            H,
            K,
            V,
            tile_v,
            use_qk_l2norm_in_kernel,
            disable_state_update,
            cache_intermediate_states,
            use_packed_fma,
            stream,
            options="--enable-tvm-ffi --generate-line-info --opt-level 3",
        )

    _compiled_kernels_mtp[cache_key](
        h_,
        inter_,
        A_log_,
        a_,
        dt_bias_,
        q_,
        k_,
        v_,
        b_,
        o_,
        h0_idx_,
        h0_out_idx_,
        stream,
    )

    return output


# Backward-compatible aliases
gated_delta_rule_bf16state_cooprow = gated_delta_rule
gated_delta_rule_bf16state_cooprow_mtp = gated_delta_rule_mtp
