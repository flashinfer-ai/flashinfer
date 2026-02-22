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
Gated Delta Rule Decode Kernel - Single Token Generation

This file implements a fused CUDA kernel using CUTLASS CuTe DSL for executing
Sigmoid Gating Delta Rule updates during decode phase. Key features:

Architecture Design:
- Uses TMA (Tensor Memory Accelerator) for efficient Global Memory → Shared Memory transfers
- Employs 2-stage pipeline to overlap loading and computation, hiding memory latency
- Each block uses 128 threads (4 warps), with each warp processing one matrix row
- Tile size: 8x128 (TILE_V x TILE_K)

Computation Flow:
1. Warp 0 handles TMA prefetch, loading data from GMEM to SMEM
2. All warps compute in parallel: softplus, L2 normalization, delta rule updates
3. Each warp processes one row of data, completing h_new = g*h + k*(beta*(v - h@k))
4. Uses warp-level shuffle for efficient reduction operations
5. Results are vectorized and written back to Global Memory

Performance Optimizations:
- Vectorized memory access: each thread processes vec_size=4 elements (128-bit aligned)
- Warp-level collective operations: shuffle-based reduction
- Pipeline overlap: load stage N+1 while computing stage N
"""

import functools
from typing import Optional, Tuple
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

try:
    from .api_logging import flashinfer_api

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False

    # Fallback decorator for standalone usage
    def flashinfer_api(func):  # type: ignore[misc]
        return func


# GDN decode K-last bf16 state kernel (T=1..4, bf16 state, K-last layout) - optional backend
try:
    from .gdn_kernels.gdn_decode_bf16_state import (
        gated_delta_rule as _gated_delta_rule_gdn_decode_klast_bf16_state,
    )

    _GDN_DECODE_KLAST_BF16_STATE_AVAILABLE = True
except ImportError:
    _GDN_DECODE_KLAST_BF16_STATE_AVAILABLE = False
    _gated_delta_rule_gdn_decode_klast_bf16_state = None


# ============================================================================
# Global configuration for PRETRANSPOSE version ([B*HV, V, K])
# ============================================================================
TILE_V = 8
TILE_K = 128
NUM_STAGES = 2
NUM_THREADS = 128  # 4 warps
NUM_BLOCKS_PER_STATE = 8

# ============================================================================
# Global configuration for NONTRANSPOSE version ([pool, HV, K, V])
# ============================================================================
TILE_K_NT = 128
TILE_V_NT = 32
TILE_V_PADDED_NT = 36
TILE_V_SMALL_NT = 16
TILE_V_SMALL_PADDED_NT = 20
NUM_STAGES_NT = 2
NUM_THREADS_NT = 128
NUM_BLOCKS_PER_STATE_SMALL_NT = 8
NUM_THREADS_LARGE_NT = 256
NUM_WARPS_LARGE_NT = 8
V_PER_WARP_NT = 4
ROWS_PER_ITER_NT = 8
NUM_K_ITERS_NT = TILE_K_NT // ROWS_PER_ITER_NT
SMALL_BATCH_THRESHOLD_NT = 32

# ============================================================================
# Global configuration for MTP (Multiple Token Processing) version
# ============================================================================
# Dynamic kernel selection based on batch size:
# - Small batch (B <= threshold): Use smaller TILE_V for more parallelism
# Optimal TILE_V depends on batch size (empirically determined):
#   B=1-2: TILE_V=4  (more blocks, better parallelism for tiny batches)
#   B=4:   TILE_V=8  (intermediate parallelism)
#   B=8:   TILE_V=16 (balance between parallelism and efficiency)
#   B>=16: TILE_V=32 (fewer blocks, better efficiency for large batches)

TILE_K_MTP = 128  # Full K dimension (shared across all configs)
NUM_THREADS_MTP = 128  # 4 warps


def get_vec_size_mtp(batch_size: int, seq_len: int = 1) -> int:
    """Select vec_size for MTP kernel.

    Always use vec_size=4 (32 threads per group = full warp, 4 groups per block).
    Full warp shuffle is more efficient and achieves >= 1.0x speedup vs Triton.
    """
    return 4


def get_tile_v_mtp(batch_size: int, seq_len: int = 1) -> int:
    """Select optimal TILE_V for MTP kernel based on batch size and sequence length.

    With vec_size=4, num_groups=4, rows_per_group = tile_v / 4.
    Tuned via grid search for optimal performance.

    For BS < 8: Use original tile_v values (smaller tiles for better parallelism)
    For BS >= 8: Use optimized tile_v values (larger tiles for better efficiency)
    """
    if batch_size < 8:
        # Original tile_v values for small batches (used with original kernel)
        if batch_size <= 2:
            return 4  # Small batch needs max parallelism
        elif batch_size <= 4:
            return 8
        else:
            return 16  # BS=5-7
    # Optimized tile_v values for large batches (used with optimized kernel)
    elif batch_size <= 8:
        return 16
    elif batch_size <= 16:
        return 32
    else:
        return 64  # BS>=17: 4 groups × 16 rows = 64 rows per tile


def get_ilp_rows(
    batch_size: int, seq_len: int, disable_state_update: bool = False
) -> int:
    """Select number of ILP rows (1, 2, 4, or 8) for the MTP kernel.

    Higher ILP rows process more V-rows simultaneously, providing more memory
    request parallelism and instruction-level parallelism to hide latency.

    When state update is enabled (disable_state_update=False), the kernel is
    more memory-bound due to the 1.07GB state writeback. Higher ILP (8-row)
    provides 3-5% improvement by maximizing outstanding memory requests:
    - T=1 BS=512: ilp=8 → 323us vs ilp=4 → 325us (0.7%)
    - T=2 BS=512: ilp=8 → 368us vs ilp=4 → 381us (3.4%)

    Selection logic:
    - BS < 8: Use 1-row ILP (ILP overhead causes regression for small batches)
    - BS = 8 (tile_v=16): Use 4-row ILP except T=4 (compiler issue)
    - BS 9-16 (tile_v=32): Use 4-row for T<=3, 1-row for T>=4
    - BS >= 17 (tile_v>=64): Use 4-row ILP, or 8-row if state_update ON + T<=2
    """
    # Small batch sizes: ILP overhead causes regression (benchmarked BS<8)
    if batch_size < 8:
        return 1

    tile_v = get_tile_v_mtp(batch_size, seq_len)

    if tile_v >= 64:
        # BS >= 17: Large batches benefit from high ILP
        # State update ON + low T: 8-row ILP maximizes memory parallelism
        if not disable_state_update and seq_len <= 2:
            return 8
        return 4
    if tile_v == 32:
        # BS 9-16: 4-row for T<=3, 1-row for T>=4 (register pressure)
        return 4 if seq_len <= 3 else 1
    if tile_v == 16:
        # BS = 8: 4-row ILP except T=4 (compiler artifact)
        return 1 if seq_len == 4 else 4
    return 1  # Fallback


def get_use_smem_v(
    batch_size: int, seq_len: int, disable_state_update: bool = False
) -> bool:
    """Decide whether to preload v values into SMEM.

    SMEM v preloading eliminates repeated GMEM scalar loads of v values during
    the inner T-loop, replacing them with fast SMEM reads. This provides 4-7%
    speedup at large batch sizes (BS>=32, tile_v>=64).

    For small batch sizes (BS<8), ILP optimizations cause regression, so we
    disable SMEM v along with ILP rows to use the simpler kernel path.

    When state update is enabled with low T (T<=2), SMEM v provides no benefit
    (measured 0-1us difference) because the kernel is fully memory-bound from
    the state read/write. Disabling SMEM v reduces shared memory usage slightly.
    """
    # Small batch sizes: disable SMEM v along with ILP to avoid regression
    if batch_size < 8:
        return False
    tile_v = get_tile_v_mtp(batch_size, seq_len)
    if tile_v < 64:
        return False
    # State update ON + low T: SMEM v provides no benefit (memory-bound)
    if not disable_state_update and seq_len <= 2:
        return False
    return True


@cute.kernel
def gdn_decode_kernel_small_batch_pretranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, T, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    b: cute.Tensor,  # [B, T, HV]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices
    cu_seqlens: cute.Tensor,  # [B+1] - cumulative sequence lengths (for varlen)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
):
    """Each block uses pipeline to load one batch and vectorized writeback"""

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

    # ===================================================================
    # Allocate shared memory (using passed-in layout)
    # ===================================================================
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)

    # Allocate shared memory for output (size V) - use BFloat16 to match SGLang
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    # Allocate shared memory for v values (size K, to reduce register usage)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # r_v moved to shared memory (sV)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # BF16 register tensors for vectorized q, k, v loading
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    # Compute k_start for contiguous access pattern
    k_start = lane_id * vec_size

    # Read gate values from GMEM EARLY (before barrier, latency hidden during sync)
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    cute.arch.barrier()

    # Get current batch
    gSrc_batch = h0_source[(batch_idx, None, None)]  # (V, K)
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    # V 方向分 tiles
    gSrc = cute.local_tile(
        gSrc_batch, (TILE_V, TILE_K), (None, 0)
    )  # (TILE_V, TILE_K, num_v_tiles)

    # Partition for load
    thr_copy_load = tiled_copy_load.get_slice(tidx)

    # ===================================================================
    # Prefetch: All threads participate in cp.async load
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

    # Load q, k into BF16 registers using autovec_copy (contiguous pattern)
    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)

    # Convert BF16 to FP32
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(r_q_bf16[i])
        r_k[i] = cutlass.Float32(r_k_bf16[i])

    # Load v into BF16 registers using autovec_copy, convert to FP32, store to sV
    v_tile = cute.local_tile(v, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_id))
    cute.autovec_copy(v_tile, r_v_bf16)
    for i in cutlass.range_constexpr(vec_size):
        sV[k_start + i] = cutlass.Float32(r_v_bf16[i])

    cute.arch.barrier()  # Ensure all threads finish writing to sV

    # ===================================================================
    # Compute g and beta (scalar values)
    # ===================================================================
    r_g = 0.0
    r_beta = 0.0
    if lane_id == 0:
        x = r_a + r_dt_bias
        beta_x = softplus_beta * x
        softplus_x = 0.0

        if beta_x <= softplus_threshold:
            # softplus(x) = (1/beta) * log(1 + exp(beta*x))
            # Compute in Float32
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            log_input = cutlass.Float32(1.0 + exp_beta_x)
            log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
            softplus_x = cutlass.Float32(
                (cutlass.Float32(1.0) / softplus_beta) * log_result
            )
        else:
            softplus_x = x

        # Compute g = exp(A_log) * softplus_x
        r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x

        # Compute beta = 1 / (1 + exp(-b))
        r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))

        # Store to scalar (Float32)
        r_g = cute.exp(r_g_value, fastmath=True)

    r_g = cute.arch.shuffle_sync(r_g, 0)
    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    if use_qk_l2norm:
        # Compute L2 norm of q and k
        sum_q = 0.0
        sum_k = 0.0
        for i in cutlass.range_constexpr(vec_size):
            sum_q += r_q[i] * r_q[i]
            sum_k += r_k[i] * r_k[i]
        # Warp-level reduction using butterfly shuffle
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

    # Apply scaling in Float32
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    # ===================================================================
    # Mainloop: All threads participate
    # ===================================================================
    end_v_tiles = start_v_tiles + num_v_tiles_per_block
    for v_tiles in range(start_v_tiles, end_v_tiles):
        stage = (v_tiles - start_v_tiles) % NUM_STAGES

        # Step 1: Wait for current stage to complete
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Step 2: Issue async load for next tile (after compute)
        next_v_tiles = v_tiles + prefetch_count
        if next_v_tiles < end_v_tiles:
            next_stage = (next_v_tiles - start_v_tiles) % NUM_STAGES

            gSrc_next = gSrc[(None, None, next_v_tiles)]
            sData_next = sData[(None, None, next_stage)]

            thr_gSrc = thr_copy_load.partition_S(gSrc_next)
            thr_sData = thr_copy_load.partition_D(sData_next)

            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        # Step 3: Compute using data from current stage (contiguous access pattern)
        for row in cutlass.range_constexpr(0, TILE_V, 4):
            row_offset = tidx // 32
            sum_hk = 0.0

            # Load h from sData using 3D local_tile + autovec_copy (contiguous in K)
            sData_tile = cute.local_tile(
                sData, (1, vec_size, 1), (row + row_offset, lane_id, stage)
            )
            cute.autovec_copy(sData_tile, r_h)

            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = r_h[i] * r_g
                sum_hk += r_h[i] * r_k[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                )

            v_new = sV[v_tiles * TILE_V + row + row_offset] - sum_hk
            v_new = v_new * r_beta

            sum_hq = 0.0
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] += r_k[i] * v_new
                sum_hq += r_h[i] * r_q[i]

            # Write h to gDst using 4D local_tile + autovec_copy (contiguous in K)
            gDst_tile = cute.local_tile(
                gDst, (1, 1, vec_size, 1), (0, row + row_offset, lane_id, v_tiles)
            )
            cute.autovec_copy(r_h, gDst_tile)

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                )

            o_idx = v_tiles * TILE_V + row + row_offset
            if lane_id == 0 and o_idx < V:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)

    # ===================================================================
    # Final writeback: Copy output from shared memory to global memory
    # All threads write (V=128, NUM_THREADS=128)
    # ===================================================================
    cute.arch.barrier()  # Ensure all writes to sOutput are complete
    if tidx >= start_v_tiles * TILE_V and tidx < end_v_tiles * TILE_V:
        o[(i_n, i_t, i_hv, tidx)] = sOutput[tidx]


@cute.kernel
def gdn_decode_kernel_big_batch_pretranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, T, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    b: cute.Tensor,  # [B, T, HV]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices
    cu_seqlens: cute.Tensor,  # [B+1] - cumulative sequence lengths (for varlen)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
):
    """Each block uses pipeline to load one batch and vectorized writeback"""

    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    batch_idx, _, _ = cute.arch.block_idx()
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)
    i_t = 0

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_a = cutlass.Float32(a[i_n, i_t, i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])
    r_b = cutlass.Float32(b[i_n, i_t, i_hv])

    smem = cutlass.utils.SmemAllocator()

    # ===================================================================
    # Allocate shared memory (using passed-in layout)
    # ===================================================================
    sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)

    # Allocate shared memory for output (size V) - use BFloat16 to match SGLang
    sOutput = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout((V,)), 16)

    # Allocate shared memory for v values (size K, to reduce register usage)
    sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout((V,)), 16)

    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # r_v moved to shared memory (sV)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # BF16 register tensors for vectorized q, k, v loading
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_v_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    # Compute k_start for contiguous access pattern
    k_start = lane_id * vec_size

    cute.arch.barrier()

    # Get current batch
    gSrc_batch = h0_source[(batch_idx, None, None)]  # (V, K)
    gDst = cute.local_tile(h0_source, (1, TILE_V, TILE_K), (batch_idx, None, 0))

    # V 方向分 tiles
    gSrc = cute.local_tile(
        gSrc_batch, (TILE_V, TILE_K), (None, 0)
    )  # (TILE_V, TILE_K, num_v_tiles)

    # Partition for load
    thr_copy_load = tiled_copy_load.get_slice(tidx)

    # ===================================================================
    # Prefetch: All threads participate in cp.async load
    # ===================================================================
    prefetch_count = cutlass.min(NUM_STAGES - 1, num_v_tiles)
    for v_tiles in range(prefetch_count):
        stage = v_tiles % NUM_STAGES

        gSrc_tile = gSrc[(None, None, v_tiles)]
        sData_stage = sData[(None, None, stage)]

        thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
        thr_sData = thr_copy_load.partition_D(sData_stage)

        cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
        cute.arch.cp_async_commit_group()

    # Load q, k into BF16 registers using autovec_copy (contiguous pattern)
    q_tile = cute.local_tile(q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    k_tile = cute.local_tile(k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_id))
    cute.autovec_copy(q_tile, r_q_bf16)
    cute.autovec_copy(k_tile, r_k_bf16)

    # Convert BF16 to FP32
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = cutlass.Float32(r_q_bf16[i])
        r_k[i] = cutlass.Float32(r_k_bf16[i])

    # Load v into BF16 registers using autovec_copy, convert to FP32, store to sV
    v_tile = cute.local_tile(v, (1, 1, 1, vec_size), (i_n, i_t, i_hv, lane_id))
    cute.autovec_copy(v_tile, r_v_bf16)
    for i in cutlass.range_constexpr(vec_size):
        sV[k_start + i] = cutlass.Float32(r_v_bf16[i])

    cute.arch.barrier()  # Ensure all threads finish writing to sV

    # ===================================================================
    # Compute g and beta (scalar values)
    # ===================================================================
    r_g = 0.0
    r_beta = 0.0
    if lane_id == 0:
        x = r_a + r_dt_bias
        beta_x = softplus_beta * x
        softplus_x = 0.0

        if beta_x <= softplus_threshold:
            # softplus(x) = (1/beta) * log(1 + exp(beta*x))
            # Compute in Float32
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            log_input = cutlass.Float32(1.0 + exp_beta_x)
            log_result = cutlass.Float32(cute.log(log_input, fastmath=True))
            softplus_x = cutlass.Float32(
                (cutlass.Float32(1.0) / softplus_beta) * log_result
            )
        else:
            softplus_x = x

        # Compute g = exp(A_log) * softplus_x
        r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x

        # Compute beta = 1 / (1 + exp(-b))
        r_beta = 1.0 / (1.0 + cute.exp(-r_b, fastmath=True))

        # Store to scalar (Float32)
        r_g = cute.exp(r_g_value, fastmath=True)

    r_g = cute.arch.shuffle_sync(r_g, 0)
    r_beta = cute.arch.shuffle_sync(r_beta, 0)

    if use_qk_l2norm:
        # Compute L2 norm of q and k
        sum_q = 0.0
        sum_k = 0.0
        for i in cutlass.range_constexpr(vec_size):
            sum_q += r_q[i] * r_q[i]
            sum_k += r_k[i] * r_k[i]
        # Warp-level reduction using butterfly shuffle
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

    # Apply scaling in Float32
    for i in cutlass.range_constexpr(vec_size):
        r_q[i] = r_q[i] * scale

    # ===================================================================
    # Mainloop: All threads participate
    # ===================================================================
    for v_tiles in range(num_v_tiles):
        stage = v_tiles % NUM_STAGES

        # Step 1: Wait for current stage to complete
        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        # Step 2: Issue async load for next tile (after compute)
        next_v_tiles = v_tiles + prefetch_count
        if next_v_tiles < num_v_tiles:
            next_stage = next_v_tiles % NUM_STAGES

            gSrc_next = gSrc[(None, None, next_v_tiles)]
            sData_next = sData[(None, None, next_stage)]

            thr_gSrc = thr_copy_load.partition_S(gSrc_next)
            thr_sData = thr_copy_load.partition_D(sData_next)

            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        # Step 3: Compute using data from current stage (contiguous access pattern)
        for row in cutlass.range_constexpr(0, TILE_V, 4):
            row_offset = tidx // 32
            sum_hk = 0.0

            # Load h from sData using 3D local_tile + autovec_copy (contiguous in K)
            sData_tile = cute.local_tile(
                sData, (1, vec_size, 1), (row + row_offset, lane_id, stage)
            )
            cute.autovec_copy(sData_tile, r_h)

            for i in cutlass.range_constexpr(vec_size):
                r_h[i] = r_h[i] * r_g
                sum_hk += r_h[i] * r_k[i]

            for offset in [16, 8, 4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                )

            v_new = sV[v_tiles * TILE_V + row + row_offset] - sum_hk
            v_new = v_new * r_beta

            sum_hq = 0.0
            for i in cutlass.range_constexpr(vec_size):
                r_h[i] += r_k[i] * v_new
                sum_hq += r_h[i] * r_q[i]

            # Write h to gDst using 4D local_tile + autovec_copy (contiguous in K)
            gDst_tile = cute.local_tile(
                gDst, (1, 1, vec_size, 1), (0, row + row_offset, lane_id, v_tiles)
            )
            cute.autovec_copy(r_h, gDst_tile)

            for offset in [16, 8, 4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                )

            o_idx = v_tiles * TILE_V + row + row_offset
            if lane_id == 0 and o_idx < V:
                sOutput[o_idx] = cutlass.BFloat16(sum_hq)

    # ===================================================================
    # Final writeback: Copy output from shared memory to global memory
    # All threads write (V=128, NUM_THREADS=128)
    # ===================================================================
    cute.arch.barrier()  # Ensure all writes to sOutput are complete

    if tidx < V:
        o[(i_n, i_t, i_hv, tidx)] = sOutput[tidx]


@cute.jit
def run_gdn_decode_kernel_small_batch_pretranspose(
    h0_source: cute.Tensor,  # [B*HV, K, V]
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Launch original pipelined kernel for small batch pretranspose."""
    # h0_source: (B*HV, V, K)
    batch_size, v_dim, k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    # Create cp.async copy with cache-global mode (bypass L1)
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,  # 4 elements per copy
    )

    # Thread layout: 4 rows × 32 threads/row = 128 threads
    thread_layout = cute.make_layout(
        (4, 32),  # 4 rows, 32 threads/row
        stride=(32, 1),
    )
    val_layout = cute.make_layout((1, 4))  # Each thread handles 4 elements

    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)

    vec_size = (
        TILE_K // 32
    )  # Each thread in a warp processes this many elements (always 4 for TILE_K=128)

    # Create SMEM layout
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    # sData: TILE_V * TILE_K * NUM_STAGES * 4 bytes (Float32)
    # sV: K * 4 bytes (Float32)
    # sOutput: V * 2 bytes (BFloat16)
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 4 * k_dim + 2 * v_dim + 32

    gdn_decode_kernel_small_batch_pretranspose(
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
        h0_indices,
        cu_seqlens,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_initial_state,
        use_qk_l2norm,
        is_varlen,
    ).launch(
        grid=(batch_size * NUM_BLOCKS_PER_STATE, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@cute.jit
def run_gdn_decode_kernel_big_batch_pretranspose(
    h0_source: cute.Tensor,  # [B*HV, K, V]
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    # h0_source: (B*HV, V, K)
    batch_size, v_dim, k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    # Create cp.async copy with cache-global mode (bypass L1)
    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,  # 4 elements per copy
    )

    # Thread layout: 4 rows × 32 threads/row = 128 threads
    thread_layout = cute.make_layout(
        (4, 32),  # 4 rows, 32 threads/row
        stride=(32, 1),
    )
    val_layout = cute.make_layout((1, 4))  # Each thread handles 4 elements

    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)

    num_v_tiles = cute.ceil_div(v_dim, TILE_V)

    vec_size = (
        TILE_K // 32
    )  # Each thread in a warp processes this many elements (always 4 for TILE_K=128)

    # Create SMEM layout
    smem_layout_staged = cute.make_layout(
        (TILE_V, TILE_K, NUM_STAGES), stride=(TILE_K, 1, TILE_V * TILE_K)
    )

    # sData: TILE_V * TILE_K * NUM_STAGES * 4 bytes (Float32)
    # sV: K * 4 bytes (Float32)
    # sOutput: V * 2 bytes (BFloat16)
    smem_bytes = 4 * TILE_V * TILE_K * NUM_STAGES + 4 * k_dim + 2 * v_dim + 32

    gdn_decode_kernel_big_batch_pretranspose(
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
        h0_indices,
        cu_seqlens,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_initial_state,
        use_qk_l2norm,
        is_varlen,
    ).launch(
        grid=(batch_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ============================================================================
# FlashInfer API Layer
# ============================================================================


@functools.cache
def _get_compiled_decode_kernel(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    scale: float,
    use_qk_l2norm: bool,
):
    """Cache compiled kernel for given configuration (pretranspose version)."""
    # This will be populated on first call
    return {}


@functools.cache
def _get_compiled_decode_kernel_nontranspose(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    scale: float,
    use_qk_l2norm: bool,
):
    """Cache compiled kernel for given configuration (nontranspose version)."""
    # This will be populated on first call
    return {}


@flashinfer_api
def gated_delta_rule_decode_pretranspose(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: Optional[torch.Tensor],
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated Delta Rule Decode kernel for single-token generation.

    This implements the decode phase of gated delta rule linear attention,
    processing one token at a time and updating the recurrent state.

    Args:
        q (torch.Tensor):
            Current query of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        k (torch.Tensor):
            Current key of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        v (torch.Tensor):
            Current value of shape ``[B, 1, HV, V]``. Must be float16/bfloat16.
        state (Optional[torch.Tensor]):
            Current state of shape ``[B, HV, V, K]`` (v-major / K-last layout).
            Float32: legacy kernel (T=1 only).  Bfloat16: gdn_decode_klast_bf16_state backend
            when T in 1..4 and K=V=128. Will be updated in-place.
            Pass ``None`` when using ``initial_state`` / ``initial_state_indices`` instead.
        A_log (torch.Tensor):
            Log decay parameter of shape ``[HV]``. Must be float32.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        dt_bias (torch.Tensor):
            Decay bias of shape ``[HV]``. Must be bfloat16 or float32.
        b (torch.Tensor):
            Update gate (beta) input of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        scale (Optional[float]):
            Scale factor for queries. If None, defaults to ``1 / sqrt(K)``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[B, 1, HV, V]``.
            If None, will be allocated automatically.
        use_qk_l2norm (bool):
            Whether to apply L2 normalization to q and k. Default: ``True``.
        initial_state (Optional[torch.Tensor]):
            State pool of shape ``[pool_size, HV, V, K]`` (K-last / K-contiguous,
            same layout as the per-batch ``state`` argument).
            When provided, the kernel gathers directly from the pool using
            ``initial_state_indices`` and writes updates back in-place — eliminating
            the caller-side gather/scatter overhead.
            Requires bfloat16 state with T in 1..4 and K=V=128 (bf16 fast path).
        initial_state_indices (Optional[torch.Tensor]):
            Per-batch indices of shape ``[B]`` (int32 or int64) mapping each batch
            entry to its slot in ``initial_state``.  Required when ``initial_state``
            is provided.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output: Output tensor of shape ``[B, 1, HV, V]``
            - state or initial_state: Updated state (in-place).

    Note:
        - Requires SM90+ (Hopper, Blackwell, etc.)
        - State is always updated in-place; the pool path writes directly into
          ``initial_state`` memory (no separate scatter step needed)
        - State layout is v-major (K-last): [B, HV, V, K]. When state is bfloat16
          and T in 1..4 with K=V=128, the gdn_decode_klast_bf16_state kernel is used
          (supports both the direct ``state`` path and the pool+indices path).
        - pool+indices (``initial_state``/``initial_state_indices``) only supported
          via the bf16 fast path; float32 state raises an error.
        - Legacy path (float32 state, T=1): K and V must be multiples of 4.
    """
    # Validate input shapes
    B, T, H, K = q.shape
    _, _, HV, V = v.shape

    use_pool = initial_state is not None
    assert use_pool == (initial_state_indices is not None), (
        "initial_state and initial_state_indices must be provided together"
    )

    if use_pool:
        pool_size = initial_state.shape[0]
        assert initial_state.shape == (pool_size, HV, V, K), (
            f"Expected initial_state shape [pool_size={pool_size}, HV={HV}, V={V}, K={K}], "
            f"got {initial_state.shape}"
        )
    else:
        assert state is not None, "Either state or initial_state must be provided"
        # Validate state shape (K-last: [B, HV, V, K])
        assert state.shape == (B, HV, V, K), (
            f"Expected state shape [B={B}, HV={HV}, V={V}, K={K}], got {state.shape}"
        )

    # Backend: gdn_decode_klast_bf16_state when bf16 state, T<=4, K-last layout, K=V=128
    state_dtype = initial_state.dtype if use_pool else state.dtype
    use_gdn_decode_klast_bf16_state = (
        _GDN_DECODE_KLAST_BF16_STATE_AVAILABLE
        and state_dtype == torch.bfloat16
        and T in (1, 2, 3, 4)
        and K == 128
        and V == 128
    )
    if use_gdn_decode_klast_bf16_state:
        assert q.dtype in (torch.float16, torch.bfloat16), (
            f"q must be float16/bfloat16, got {q.dtype}"
        )
        assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"
        scale_val = K**-0.5 if scale is None else scale
        out = _gated_delta_rule_gdn_decode_klast_bf16_state(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state if use_pool else state,
            initial_state_indices=initial_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm,
            scale=scale_val,
        )
        output_provided = output is not None
        target_dtype = output.dtype if output_provided else q.dtype
        if output is not None:
            output.copy_(out)
        else:
            output = out
        if output.dtype != target_dtype:
            output = output.to(target_dtype)
        return_state = initial_state if use_pool else state
        return output, return_state

    # Legacy path: T=1 only, float32 state (no pool+indices support)
    assert not use_pool, (
        "pool+indices (initial_state/initial_state_indices) requires bfloat16 state "
        "with T in 1..4 and K=V=128 (the gdn_decode_klast_bf16_state fast path)"
    )
    assert T == 1, f"Decode only supports T=1, got T={T}"
    assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"

    # Validate K and V constraints
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % TILE_V == 0, (
        f"V must be divisible by {TILE_V} to prevent out-of-bounds access, got V={V}"
    )

    # Validate dtypes
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"

    # Set default scale
    if scale is None:
        scale = K**-0.5

    # Allocate output if not provided
    # Note: kernel outputs bfloat16, we'll convert to q.dtype if needed
    output_provided = output is not None
    target_dtype = output.dtype if output_provided else q.dtype

    if output is None:
        # Kernel outputs bfloat16, allocate in that dtype first
        output = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    # Convert state from [B, HV, V, K] to [B*HV, V, K] for kernel
    h0_source = state.reshape(B * HV, V, K)

    # Compile kernel with TVM FFI (cached)
    cache_key = (B, T, H, HV, K, V, q.dtype, scale, use_qk_l2norm)
    cache = _get_compiled_decode_kernel(*cache_key)

    # Get or create h0_indices and cu_seqlens (cached per config)
    if "h0_indices" not in cache or cache["h0_indices"].device != q.device:
        cache["h0_indices"] = torch.zeros(B, dtype=torch.int32, device=q.device)
        cache["cu_seqlens"] = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    h0_indices = cache["h0_indices"]
    cu_seqlens = cache["cu_seqlens"]

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # Convert tensors to CuTe format for compilation only
        h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
        A_log_tensor = from_dlpack(A_log, assumed_align=16)
        a_tensor = from_dlpack(a, assumed_align=16)
        dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
        q_tensor = from_dlpack(q, assumed_align=16)
        k_tensor = from_dlpack(k, assumed_align=16)
        v_tensor = from_dlpack(v, assumed_align=16)
        b_tensor = from_dlpack(b, assumed_align=16)
        o_tensor = from_dlpack(output, assumed_align=16)
        h0_indices_tensor = from_dlpack(h0_indices, assumed_align=16)
        cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

        # Always use 8-CTA architecture (benchmarks show it's better for all batch sizes)
        run_func = run_gdn_decode_kernel_small_batch_pretranspose

        # Use TVM FFI to reduce runtime overhead
        compiled = cute.compile(
            run_func,
            h0_source_tensor,
            A_log_tensor,
            a_tensor,
            dt_bias_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            b_tensor,
            o_tensor,
            h0_indices_tensor,
            cu_seqlens_tensor,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            HV=HV,
            B=B,
            T=T,
            H=H,
            K=K,
            V=V,
            use_initial_state=True,
            use_qk_l2norm=use_qk_l2norm,
            is_varlen=False,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled
    else:
        compiled = cache["compiled"]

    # Run kernel directly with PyTorch tensors (no from_dlpack needed)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        h0_source, A_log, a, dt_bias, q, k, v, b, output, h0_indices, cu_seqlens, stream
    )

    # Convert output to target dtype if needed (kernel outputs bfloat16)
    if output.dtype != target_dtype:
        output = output.to(target_dtype)

    # Copy state back only if state was not contiguous
    # (if contiguous, reshape returns a view and kernel updated state in-place)
    if not state.is_contiguous():
        state.copy_(h0_source.reshape(B, HV, V, K))
    return output, state


# ============================================================================
# NONTRANSPOSE Version Kernels - K-major layout [pool, HV, K, V]
# ============================================================================


@cute.kernel
def gdn_decode_kernel_small_batch_nontranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    num_v_tiles: cutlass.Constexpr[int],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
):
    """Small batch kernel for (N, 1, ...) format with K-major state layout."""
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    block_idx, _, _ = cute.arch.block_idx()

    NUM_WARPS_SMALL = 4
    V_PER_WARP_SMALL = TILE_V_SMALL_NT // NUM_WARPS_SMALL
    ROWS_PER_ITER_SMALL = 32 // V_PER_WARP_SMALL
    NUM_K_ITERS_SMALL = TILE_K_NT // ROWS_PER_ITER_SMALL

    batch_idx = block_idx // NUM_BLOCKS_PER_STATE_SMALL_NT
    batch_inner = block_idx % NUM_BLOCKS_PER_STATE_SMALL_NT
    num_v_tiles_per_block = num_v_tiles // NUM_BLOCKS_PER_STATE_SMALL_NT
    start_v_tile = batch_inner * num_v_tiles_per_block

    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)

    pool_idx = h0_indices[i_n]

    if pool_idx >= 0:
        k_local = in_warp_tid // V_PER_WARP_SMALL
        v_local = in_warp_tid % V_PER_WARP_SMALL
        v_base = warp_idx * V_PER_WARP_SMALL
        v_idx = v_base + v_local

        smem = cutlass.utils.SmemAllocator()
        sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
        smem_o_layout = cute.make_layout((TILE_V_SMALL_NT,), stride=(1,))
        smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
        smem_k_layout = cute.make_layout((TILE_K_NT,), stride=(1,))
        smem_q_layout = cute.make_layout((TILE_K_NT,), stride=(1,))
        sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
        sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

        if tidx < TILE_K_NT:
            sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
            sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

        # Compute flat index for flattened state [B*HV, K, V]
        flat_idx = pool_idx * HV + i_hv
        gSrc_batch = h0_source[(flat_idx, None, None)]
        gSrc = cute.local_tile(gSrc_batch, (TILE_K_NT, TILE_V_SMALL_NT), (0, None))
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        prefetch_count = cutlass.min(NUM_STAGES_NT - 1, num_v_tiles_per_block)
        for v_tile_offset in range(prefetch_count):
            v_tile = start_v_tile + v_tile_offset
            stage = v_tile_offset % NUM_STAGES_NT
            gSrc_tile = gSrc[(None, None, v_tile)]
            sData_stage = sData[(None, None, stage)]
            thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
            thr_sData = thr_copy_load.partition_D(sData_stage)
            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        r_A_log = cutlass.Float32(A_log[i_hv])
        r_dt_bias = cutlass.Float32(dt_bias[i_hv])
        r_a = cutlass.Float32(a[i_n, 0, i_hv])
        r_b = cutlass.Float32(b[i_n, 0, i_hv])

        r_g = 0.0
        r_beta = 0.0
        if in_warp_tid == 0:
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

        cute.arch.barrier()

        if use_qk_l2norm:
            sum_q_partial = 0.0
            sum_k_partial = 0.0
            if tidx < TILE_K_NT:
                q_val = sQ[tidx]
                k_val = sK[tidx]
                sum_q_partial = q_val * q_val
                sum_k_partial = k_val * k_val

            for offset in [16, 8, 4, 2, 1]:
                sum_q_partial += cute.arch.shuffle_sync_bfly(
                    sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                )
                sum_k_partial += cute.arch.shuffle_sync_bfly(
                    sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                )

            if in_warp_tid == 0:
                smem_o[warp_idx] = sum_q_partial
                smem_o[warp_idx + 4] = sum_k_partial
            cute.arch.barrier()

            inv_norm_q = 0.0
            inv_norm_k = 0.0
            if warp_idx == 0:
                local_sum_q = 0.0
                local_sum_k = 0.0
                if in_warp_tid < NUM_WARPS_SMALL:
                    local_sum_q = smem_o[in_warp_tid]
                    local_sum_k = smem_o[in_warp_tid + 4]
                for offset in [2, 1]:
                    local_sum_q += cute.arch.shuffle_sync_bfly(
                        local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    local_sum_k += cute.arch.shuffle_sync_bfly(
                        local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )
                if in_warp_tid == 0:
                    smem_o[0] = cute.rsqrt(local_sum_q + 1e-6, fastmath=True)
                    smem_o[1] = cute.rsqrt(local_sum_k + 1e-6, fastmath=True)
            cute.arch.barrier()

            inv_norm_q = smem_o[0]
            inv_norm_k = smem_o[1]

            if tidx < TILE_K_NT:
                sK[tidx] = sK[tidx] * inv_norm_k
                sQ[tidx] = sQ[tidx] * scale * inv_norm_q
            cute.arch.barrier()
        else:
            if tidx < TILE_K_NT:
                sQ[tidx] = sQ[tidx] * scale
            cute.arch.barrier()

        for v_tile_offset in range(num_v_tiles_per_block):
            v_tile = start_v_tile + v_tile_offset
            stage = v_tile_offset % NUM_STAGES_NT

            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            next_v_tile_offset = v_tile_offset + prefetch_count
            if next_v_tile_offset < num_v_tiles_per_block:
                next_v_tile = start_v_tile + next_v_tile_offset
                next_stage = next_v_tile_offset % NUM_STAGES_NT
                gSrc_next = gSrc[(None, None, next_v_tile)]
                sData_next = sData[(None, None, next_stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                thr_sData = thr_copy_load.partition_D(sData_next)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            v_global = v_tile * TILE_V_SMALL_NT + v_idx
            r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

            sum_hk = 0.0
            for k_iter in range(NUM_K_ITERS_SMALL, unroll=16):  # type: ignore[call-overload]
                k_base = k_iter * ROWS_PER_ITER_SMALL
                k_idx = k_base + k_local
                h_val = sData[(k_idx, v_idx, stage)] * r_g
                r_k_val = sK[k_idx]
                sum_hk += h_val * r_k_val

            for offset in [4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk,
                    offset=offset * V_PER_WARP_SMALL,
                    mask=-1,
                    mask_and_clamp=31,
                )

            v_new = (r_v - sum_hk) * r_beta
            v_new = cute.arch.shuffle_sync(v_new, v_local)

            sum_hq = 0.0
            for k_iter in range(NUM_K_ITERS_SMALL, unroll=16):  # type: ignore[call-overload]
                k_base = k_iter * ROWS_PER_ITER_SMALL
                k_idx = k_base + k_local
                h_old = sData[(k_idx, v_idx, stage)] * r_g
                r_k_val = sK[k_idx]
                r_q_val = sQ[k_idx]
                h_new = h_old + r_k_val * v_new
                sData[(k_idx, v_idx, stage)] = h_new
                sum_hq += h_new * r_q_val

            for offset in [4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq,
                    offset=offset * V_PER_WARP_SMALL,
                    mask=-1,
                    mask_and_clamp=31,
                )

            if k_local == 0:
                v_global_out = v_tile * TILE_V_SMALL_NT + v_idx
                o[(i_n, 0, i_hv, v_global_out)] = cutlass.BFloat16(sum_hq)

            cute.arch.barrier()

            for k_iter in cutlass.range_constexpr(NUM_K_ITERS_SMALL):
                flat_tid = tidx + k_iter * 128
                k_write = flat_tid // TILE_V_SMALL_NT
                v_write = flat_tid % TILE_V_SMALL_NT
                if k_write < TILE_K_NT:
                    h_val = sData[(k_write, v_write, stage)]
                    v_global_write = v_tile * TILE_V_SMALL_NT + v_write
                    # Use flat index for flattened state [B*HV, K, V]
                    h0_source[(flat_idx, k_write, v_global_write)] = h_val

            cute.arch.barrier()


@cute.kernel
def gdn_decode_kernel_big_batch_nontranspose(
    tiled_copy_load: cute.TiledCopy,
    h0_source: cute.Tensor,
    smem_layout_staged: cute.Layout,
    num_v_tiles: cutlass.Constexpr[int],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
):
    """Large batch kernel for (N, 1, ...) format with K-major state layout."""
    tidx, _, _ = cute.arch.thread_idx()
    in_warp_tid = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    batch_idx, _, _ = cute.arch.block_idx()
    i_n = batch_idx // HV
    i_hv = batch_idx % HV
    i_h = i_hv // (HV // H)

    pool_idx = h0_indices[i_n]

    if pool_idx >= 0:
        k_local = in_warp_tid // V_PER_WARP_NT
        v_local = in_warp_tid % V_PER_WARP_NT
        v_base = warp_idx * V_PER_WARP_NT
        v_idx = v_base + v_local

        smem = cutlass.utils.SmemAllocator()
        sData = smem.allocate_tensor(cutlass.Float32, smem_layout_staged, 128)
        smem_o_layout = cute.make_layout((TILE_V_NT,), stride=(1,))
        smem_o = smem.allocate_tensor(cutlass.Float32, smem_o_layout, 128)
        smem_k_layout = cute.make_layout((TILE_K_NT,), stride=(1,))
        smem_q_layout = cute.make_layout((TILE_K_NT,), stride=(1,))
        sK = smem.allocate_tensor(cutlass.Float32, smem_k_layout, 128)
        sQ = smem.allocate_tensor(cutlass.Float32, smem_q_layout, 128)

        if tidx < TILE_K_NT:
            sK[tidx] = cutlass.Float32(k[i_n, 0, i_h, tidx])
            sQ[tidx] = cutlass.Float32(q[i_n, 0, i_h, tidx])

        # Compute flat index for flattened state [B*HV, K, V]
        flat_idx = pool_idx * HV + i_hv
        gSrc_batch = h0_source[(flat_idx, None, None)]
        gSrc = cute.local_tile(gSrc_batch, (TILE_K_NT, TILE_V_NT), (0, None))
        thr_copy_load = tiled_copy_load.get_slice(tidx)

        prefetch_count = cutlass.min(NUM_STAGES_NT - 1, num_v_tiles)
        for v_tile in range(prefetch_count):
            stage = v_tile % NUM_STAGES_NT
            gSrc_tile = gSrc[(None, None, v_tile)]
            sData_stage = sData[(None, None, stage)]
            thr_gSrc = thr_copy_load.partition_S(gSrc_tile)
            thr_sData = thr_copy_load.partition_D(sData_stage)
            cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
            cute.arch.cp_async_commit_group()

        r_A_log = cutlass.Float32(A_log[i_hv])
        r_dt_bias = cutlass.Float32(dt_bias[i_hv])
        r_a = cutlass.Float32(a[i_n, 0, i_hv])
        r_b = cutlass.Float32(b[i_n, 0, i_hv])

        r_g = 0.0
        r_beta = 0.0
        if in_warp_tid == 0:
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

        cute.arch.barrier()

        if use_qk_l2norm:
            sum_q_partial = 0.0
            sum_k_partial = 0.0
            if tidx < TILE_K_NT:
                q_val = sQ[tidx]
                k_val = sK[tidx]
                sum_q_partial = q_val * q_val
                sum_k_partial = k_val * k_val

            for offset in [16, 8, 4, 2, 1]:
                sum_q_partial += cute.arch.shuffle_sync_bfly(
                    sum_q_partial, offset=offset, mask=-1, mask_and_clamp=31
                )
                sum_k_partial += cute.arch.shuffle_sync_bfly(
                    sum_k_partial, offset=offset, mask=-1, mask_and_clamp=31
                )

            if in_warp_tid == 0:
                smem_o[warp_idx] = sum_q_partial
                smem_o[warp_idx + 8] = sum_k_partial
            cute.arch.barrier()

            inv_norm_q = 0.0
            inv_norm_k = 0.0
            if warp_idx == 0:
                local_sum_q = 0.0
                local_sum_k = 0.0
                if in_warp_tid < NUM_WARPS_LARGE_NT:
                    local_sum_q = smem_o[in_warp_tid]
                    local_sum_k = smem_o[in_warp_tid + 8]
                for offset in [4, 2, 1]:
                    local_sum_q += cute.arch.shuffle_sync_bfly(
                        local_sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    local_sum_k += cute.arch.shuffle_sync_bfly(
                        local_sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )
                if in_warp_tid == 0:
                    smem_o[0] = cute.rsqrt(local_sum_q + 1e-6, fastmath=True)
                    smem_o[1] = cute.rsqrt(local_sum_k + 1e-6, fastmath=True)
            cute.arch.barrier()

            inv_norm_q = smem_o[0]
            inv_norm_k = smem_o[1]

            if tidx < TILE_K_NT:
                sK[tidx] = sK[tidx] * inv_norm_k
                sQ[tidx] = sQ[tidx] * scale * inv_norm_q
            cute.arch.barrier()
        else:
            if tidx < TILE_K_NT:
                sQ[tidx] = sQ[tidx] * scale
            cute.arch.barrier()

        for v_tile in range(num_v_tiles):
            stage = v_tile % NUM_STAGES_NT

            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            next_v_tile = v_tile + prefetch_count
            if next_v_tile < num_v_tiles:
                next_stage = next_v_tile % NUM_STAGES_NT
                gSrc_next = gSrc[(None, None, next_v_tile)]
                sData_next = sData[(None, None, next_stage)]
                thr_gSrc = thr_copy_load.partition_S(gSrc_next)
                thr_sData = thr_copy_load.partition_D(sData_next)
                cute.copy(tiled_copy_load, thr_gSrc, thr_sData)
                cute.arch.cp_async_commit_group()

            v_global = v_tile * TILE_V_NT + v_idx
            r_v = cutlass.Float32(v[i_n, 0, i_hv, v_global])

            sum_hk = 0.0
            for k_iter in range(NUM_K_ITERS_NT, unroll=8):  # type: ignore[call-overload]
                k_base = k_iter * ROWS_PER_ITER_NT
                k_idx = k_base + k_local
                h_val = sData[(k_idx, v_idx, stage)] * r_g
                r_k_val = sK[k_idx]
                sum_hk += h_val * r_k_val

            for offset in [4, 2, 1]:
                sum_hk += cute.arch.shuffle_sync_bfly(
                    sum_hk, offset=offset * V_PER_WARP_NT, mask=-1, mask_and_clamp=31
                )

            v_new = (r_v - sum_hk) * r_beta
            v_new = cute.arch.shuffle_sync(v_new, v_local)

            sum_hq = 0.0
            for k_iter in range(NUM_K_ITERS_NT, unroll=8):  # type: ignore[call-overload]
                k_base = k_iter * ROWS_PER_ITER_NT
                k_idx = k_base + k_local
                h_old = sData[(k_idx, v_idx, stage)] * r_g
                r_k_val = sK[k_idx]
                r_q_val = sQ[k_idx]
                h_new = h_old + r_k_val * v_new
                sData[(k_idx, v_idx, stage)] = h_new
                sum_hq += h_new * r_q_val

            for offset in [4, 2, 1]:
                sum_hq += cute.arch.shuffle_sync_bfly(
                    sum_hq, offset=offset * V_PER_WARP_NT, mask=-1, mask_and_clamp=31
                )

            if k_local == 0:
                v_global_out = v_tile * TILE_V_NT + v_idx
                o[(i_n, 0, i_hv, v_global_out)] = cutlass.BFloat16(sum_hq)

            cute.arch.barrier()

            for k_iter in cutlass.range_constexpr(NUM_K_ITERS_NT):
                flat_tid = tidx + k_iter * 256
                k_write = flat_tid // TILE_V_NT
                v_write = flat_tid % TILE_V_NT
                if k_write < TILE_K_NT:
                    h_val = sData[(k_write, v_write, stage)]
                    v_global_write = v_tile * TILE_V_NT + v_write
                    # Use flat index for flattened state [B*HV, K, V]
                    h0_source[(flat_idx, k_write, v_global_write)] = h_val

            cute.arch.barrier()


@cute.jit
def run_gdn_decode_kernel_small_batch_nontranspose(
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    h0_source: cute.Tensor,
    h0_indices: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    # h0_source is flattened to [B*HV, K, V] to ensure proper alignment for SIMT async copy
    batch_hv_dim, k_dim, v_dim = h0_source.layout.shape
    h0_indices.layout.shape[0]
    batch_size = batch_hv_dim  # batch_hv_dim = B * HV

    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    num_v_tiles_small = cute.ceil_div(v_dim, TILE_V_SMALL_NT)
    smem_layout_small = cute.make_layout(
        (TILE_K_NT, TILE_V_SMALL_NT, NUM_STAGES_NT),
        stride=(TILE_V_SMALL_PADDED_NT, 1, TILE_K_NT * TILE_V_SMALL_PADDED_NT),
    )
    thread_layout_small = cute.make_layout((32, 4), stride=(4, 1))
    val_layout_small = cute.make_layout((1, 4))
    tiled_copy_load_small = cute.make_tiled_copy_tv(
        copy_atom, thread_layout_small, val_layout_small
    )
    smem_bytes_small = (
        4 * TILE_K_NT * TILE_V_SMALL_PADDED_NT * NUM_STAGES_NT
        + 4 * TILE_V_SMALL_NT
        + 4 * TILE_K_NT * 2
        + 64
    )

    gdn_decode_kernel_small_batch_nontranspose(
        tiled_copy_load_small,
        h0_source,
        smem_layout_small,
        num_v_tiles_small,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        o,
        h0_indices,
        softplus_beta,
        softplus_threshold,
        scale,
        H,
        HV,
        use_qk_l2norm,
    ).launch(
        grid=(batch_size * NUM_BLOCKS_PER_STATE_SMALL_NT, 1, 1),
        block=[NUM_THREADS_NT, 1, 1],
        smem=smem_bytes_small,
        stream=stream,
    )


@cute.jit
def run_gdn_decode_kernel_big_batch_nontranspose(
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    a: cute.Tensor,
    b: cute.Tensor,
    A_log: cute.Tensor,
    dt_bias: cute.Tensor,
    h0_source: cute.Tensor,
    h0_indices: cute.Tensor,
    o: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    # h0_source is flattened to [B*HV, K, V] to ensure proper alignment for SIMT async copy
    batch_hv_dim, k_dim, v_dim = h0_source.layout.shape
    h0_indices.layout.shape[0]
    batch_size = batch_hv_dim  # batch_hv_dim = B * HV

    copy_atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
        cutlass.Float32,
        num_bits_per_copy=128,
    )
    num_v_tiles = cute.ceil_div(v_dim, TILE_V_NT)
    base_smem_layout = cute.make_layout(
        (TILE_K_NT, TILE_V_NT, NUM_STAGES_NT),
        stride=(TILE_V_PADDED_NT, 1, TILE_K_NT * TILE_V_PADDED_NT),
    )
    thread_layout = cute.make_layout((32, 8), stride=(8, 1))
    val_layout = cute.make_layout((1, 4))
    tiled_copy_load = cute.make_tiled_copy_tv(copy_atom, thread_layout, val_layout)
    smem_bytes = (
        4 * TILE_K_NT * TILE_V_PADDED_NT * NUM_STAGES_NT
        + 4 * TILE_V_NT
        + 4 * TILE_K_NT * 2
        + 64
    )

    gdn_decode_kernel_big_batch_nontranspose(
        tiled_copy_load,
        h0_source,
        base_smem_layout,
        num_v_tiles,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        o,
        h0_indices,
        softplus_beta,
        softplus_threshold,
        scale,
        H,
        HV,
        use_qk_l2norm,
    ).launch(
        grid=(batch_size, 1, 1),
        block=[NUM_THREADS_LARGE_NT, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ============================================================================
# FlashInfer API Layer - NONTRANSPOSE Version (Recommended)
# ============================================================================


@flashinfer_api
def gated_delta_rule_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Gated Delta Rule Decode kernel (K-major layout, no transpose needed).

    This implements the decode phase of gated delta rule linear attention,
    processing one token at a time and updating the recurrent state.
    This version uses K-major state layout [B, HV, K, V] which is more natural
    and doesn't require transposition.

    Args:
        q (torch.Tensor):
            Current query of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        k (torch.Tensor):
            Current key of shape ``[B, 1, H, K]``. Must be float16/bfloat16.
        v (torch.Tensor):
            Current value of shape ``[B, 1, HV, V]``. Must be float16/bfloat16.
        state (torch.Tensor):
            Current state of shape ``[B, HV, K, V]`` (k-major layout).
            Must be float32. Will be updated in-place.
        A_log (torch.Tensor):
            Log decay parameter of shape ``[HV]``. Must be float32.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        dt_bias (torch.Tensor):
            Decay bias of shape ``[HV]``. Must be bfloat16 or float32.
        b (torch.Tensor):
            Update gate (beta) input of shape ``[B, 1, HV]``. Must be float16/bfloat16.
        scale (Optional[float]):
            Scale factor for queries. If None, defaults to ``1 / sqrt(K)``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[B, 1, HV, V]``.
            If None, will be allocated automatically.
        use_qk_l2norm (bool):
            Whether to apply L2 normalization to q and k. Default: ``True``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output: Output tensor of shape ``[B, 1, HV, V]``
            - state: Updated state tensor of shape ``[B, HV, K, V]``

    Note:
        - Requires SM90 (Hopper) architecture
        - State is updated in-place
        - K and V must be multiples of 4 for vectorized loads
        - State layout is k-major: [B, HV, K, V] (no transpose needed)
    """
    # Validate input shapes
    B, T, H, K = q.shape
    assert T == 1, f"Decode only supports T=1, got T={T}"
    _, _, HV, V = v.shape

    # Validate state shape
    assert state.shape == (B, HV, K, V), (
        f"Expected state shape [B={B}, HV={HV}, K={K}, V={V}], got {state.shape}"
    )

    # Validate K and V constraints
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    # V must be divisible by tile size to prevent out-of-bounds access
    # For small batch: TILE_V_SMALL_NT=16, for large batch: TILE_V_NT=32
    # Use the more restrictive constraint (32) to cover both cases
    assert V % TILE_V_NT == 0, (
        f"V must be divisible by {TILE_V_NT} to prevent out-of-bounds access, got V={V}"
    )

    # Validate dtypes
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert state.dtype == torch.float32, f"state must be float32, got {state.dtype}"
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"

    # Set default scale
    if scale is None:
        scale = K**-0.5

    # Allocate output if not provided
    output_provided = output is not None
    target_dtype = output.dtype if output_provided else q.dtype

    if output is None:
        # Kernel outputs bfloat16, allocate in that dtype first
        output = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    # State is in K-major layout [B, HV, K, V]
    # Flatten to [B*HV, K, V] to ensure proper alignment for SIMT async copy
    # This avoids alignment issues when B=1 (zero strides cause alignment failures)
    state_contiguous = state.contiguous()
    h0_source = state_contiguous.view(B * HV, K, V)

    # Compile kernel with TVM FFI (cached)
    cache_key = (B, T, H, HV, K, V, q.dtype, scale, use_qk_l2norm)
    cache = _get_compiled_decode_kernel_nontranspose(*cache_key)

    # Get or create h0_indices and cu_seqlens (cached per config)
    if "h0_indices" not in cache or cache["h0_indices"].device != q.device:
        cache["h0_indices"] = torch.arange(B, dtype=torch.int32, device=q.device)
        cache["cu_seqlens"] = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    h0_indices = cache["h0_indices"]
    cu_seqlens = cache["cu_seqlens"]

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # Choose kernel based on batch size
        use_small_batch = B < SMALL_BATCH_THRESHOLD_NT

        if use_small_batch:
            run_func = run_gdn_decode_kernel_small_batch_nontranspose
        else:
            run_func = run_gdn_decode_kernel_big_batch_nontranspose

        # Convert tensors to CuTe format for compilation only
        h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
        A_log_tensor = from_dlpack(A_log, assumed_align=16)
        a_tensor = from_dlpack(a, assumed_align=16)
        dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
        q_tensor = from_dlpack(q, assumed_align=16)
        k_tensor = from_dlpack(k, assumed_align=16)
        v_tensor = from_dlpack(v, assumed_align=16)
        b_tensor = from_dlpack(b, assumed_align=16)
        o_tensor = from_dlpack(output, assumed_align=16)
        h0_indices_tensor = from_dlpack(h0_indices, assumed_align=16)
        cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

        # Use TVM FFI to reduce runtime overhead
        compiled = cute.compile(
            run_func,
            cu_seqlens_tensor,
            q_tensor,
            k_tensor,
            v_tensor,
            a_tensor,
            b_tensor,
            A_log_tensor,
            dt_bias_tensor,
            h0_source_tensor,
            h0_indices_tensor,
            o_tensor,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            scale=scale,
            B=B,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            use_initial_state=True,
            use_qk_l2norm=use_qk_l2norm,
            stream=stream,
            options="--enable-tvm-ffi",
        )
        cache["compiled"] = compiled
    else:
        compiled = cache["compiled"]

    # Run kernel directly with PyTorch tensors (no from_dlpack needed)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        cu_seqlens,
        q,
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        h0_source,
        h0_indices,
        output,
        stream,
    )

    # Copy state back only if state was not contiguous
    # (if contiguous, state_contiguous is state itself, so kernel updated state in-place)
    if state_contiguous.data_ptr() != state.data_ptr():
        state.copy_(state_contiguous)

    # Convert output to target dtype if needed (kernel outputs bfloat16)
    if output.dtype != target_dtype:
        output = output.to(target_dtype)

    return output, state


# ============================================================================
# MTP (Multiple Token Processing) Kernel - for T > 1 (verify mode)
# ============================================================================


# Original MTP kernel (without ILP optimizations) - used for BS < 8
@cute.kernel
def gdn_verify_kernel_mtp_original(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] - initial state pool (K-last)
    intermediate_states: cute.Tensor,  # [pool_size * T * HV, V, K] - intermediate state cache
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],  # TILE_V - configurable for batch size
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, T, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    b: cute.Tensor,  # [B, T, HV]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices
    cu_seqlens: cute.Tensor,  # [B+1] - cumulative sequence lengths (for varlen)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
):
    """
    Original MTP kernel without ILP optimizations.

    Used for small batch sizes (BS < 8) where ILP overhead exceeds benefit.
    Grid: (B * HV * num_v_tiles, 1, 1)
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # Compute thread grouping based on vec_size:
    # vec_size=8: 16 threads per group (half-warp), 8 groups per block
    # vec_size=4: 32 threads per group (full warp), 4 groups per block
    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 16 or 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group  # 2 or 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp  # 8 or 4

    # Lane position within group and group index
    lane_in_group = lane_id % threads_per_group
    group_in_warp = lane_id // threads_per_group
    group_idx = warp_idx * groups_per_warp + group_in_warp

    batch_idx, _, _ = cute.arch.block_idx()

    # Decode block index: (i_n, i_hv, i_v) from batch_idx
    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    # Get initial state index for this batch
    cache_idx = h0_indices[i_n]

    # Load A_log and dt_bias once (they don't vary with time)
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])

    # Allocate shared memory for pre-computed values (broadcast to all warps)
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sK = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)

    # Register arrays for computation
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # BF16 register tensors for vectorized q, k loading
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    # Only process valid batch entries (cache_idx >= 0)
    if cache_idx >= 0:
        # Compute k_start once (used for shared memory writes)
        k_start = lane_in_group * vec_size

        # Pre-compute q, k, g, beta for ALL time steps ONCE (shared across warps)
        for i_t in cutlass.range_constexpr(T):
            # Load q, k into BF16 registers using autovec_copy (coalesced)
            q_tile = cute.local_tile(
                q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
            )
            k_tile = cute.local_tile(
                k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
            )
            cute.autovec_copy(q_tile, r_q_bf16)
            cute.autovec_copy(k_tile, r_k_bf16)

            # Convert BF16 to FP32 for computation
            for i in cutlass.range_constexpr(vec_size):
                r_q[i] = cutlass.Float32(r_q_bf16[i])
                r_k[i] = cutlass.Float32(r_k_bf16[i])

            # Apply L2 normalization to q, k (with scale fused for q)
            if cutlass.const_expr(use_qk_l2norm):
                sum_q = 0.0
                sum_k = 0.0
                for i in cutlass.range_constexpr(vec_size):
                    sum_q += r_q[i] * r_q[i]
                    sum_k += r_k[i] * r_k[i]

                # Warp-level reduction (32 threads per group with vec_size=4)
                for offset in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(
                        sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k += cute.arch.shuffle_sync_bfly(
                        sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )

                # Fuse scale into q's normalization factor
                inv_norm_q_scaled = cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)

                for i in cutlass.range_constexpr(vec_size):
                    r_q[i] = r_q[i] * inv_norm_q_scaled
                    r_k[i] = r_k[i] * inv_norm_k
            else:
                # No L2 norm, just apply scale to q
                for i in cutlass.range_constexpr(vec_size):
                    r_q[i] = r_q[i] * scale

            # Store to shared memory (only first group writes) - contiguous layout
            if tidx < threads_per_group:
                for i in cutlass.range_constexpr(vec_size):
                    sQ[(i_t, k_start + i)] = r_q[i]
                    sK[(i_t, k_start + i)] = r_k[i]

            # Compute g, beta - all lanes compute (redundant but no divergence)
            r_a = cutlass.Float32(a[i_n, i_t, i_hv])
            r_b = cutlass.Float32(b[i_n, i_t, i_hv])

            x = r_a + r_dt_bias
            beta_x = softplus_beta * x

            # Branchless softplus
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            softplus_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                cutlass.Float32(1.0) + exp_beta_x, fastmath=True
            )
            use_softplus = (
                cutlass.Float32(1.0)
                if beta_x <= softplus_threshold
                else cutlass.Float32(0.0)
            )
            softplus_x = (
                use_softplus * softplus_val + (cutlass.Float32(1.0) - use_softplus) * x
            )

            r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
            r_beta = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(-r_b, fastmath=True)
            )
            r_g = cute.exp(r_g_value, fastmath=True)

            # Only thread 0 stores to shared memory
            if tidx == 0:
                sG[i_t] = r_g
                sBeta[i_t] = r_beta

        cute.arch.barrier()

        # Each group handles tile_v/num_groups V rows
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        for row_in_group in cutlass.range_constexpr(rows_per_group):
            v_idx = i_v * tile_v + group_idx * rows_per_group + row_in_group

            if v_idx < V:
                # Load h[v_idx, :] into registers using 3D local_tile + autovec_copy
                flat_state_idx = cache_idx * HV + i_hv
                h_tile = cute.local_tile(
                    h0_source, (1, 1, vec_size), (flat_state_idx, v_idx, lane_in_group)
                )
                cute.autovec_copy(h_tile, r_h)

                # Process all T time steps with h in registers
                for i_t in cutlass.range_constexpr(T):
                    # Load pre-computed q, k from shared memory using 2D local_tile
                    sQ_tile = cute.local_tile(sQ, (1, vec_size), (i_t, lane_in_group))
                    sK_tile = cute.local_tile(sK, (1, vec_size), (i_t, lane_in_group))
                    cute.autovec_copy(sQ_tile, r_q)
                    cute.autovec_copy(sK_tile, r_k)

                    r_g = sG[i_t]
                    r_beta = sBeta[i_t]

                    # Step 1: Apply decay to h
                    for i in cutlass.range_constexpr(vec_size):
                        r_h[i] = r_h[i] * r_g

                    # Step 2: Compute sum_hk = h @ k (group reduction)
                    sum_hk = 0.0
                    for i in cutlass.range_constexpr(vec_size):
                        sum_hk += r_h[i] * r_k[i]

                    # Warp-level reduction
                    for offset in [16, 8, 4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(
                            sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                        )

                    # Step 3: Load v for this v_idx and time step, apply delta rule
                    r_v = cutlass.Float32(v[i_n, i_t, i_hv, v_idx])
                    v_new = (r_v - sum_hk) * r_beta

                    # Step 4: Update h: h += k * v_new
                    for i in cutlass.range_constexpr(vec_size):
                        r_h[i] += r_k[i] * v_new

                    # Cache intermediate state if needed using 3D local_tile + autovec_copy
                    if cutlass.const_expr(cache_intermediate_states):
                        flat_idx = i_n * T * HV + i_t * HV + i_hv
                        inter_tile = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec_size),
                            (flat_idx, v_idx, lane_in_group),
                        )
                        cute.autovec_copy(r_h, inter_tile)

                    # Step 5: Compute output: sum_hq = h @ q (group reduction)
                    sum_hq = 0.0
                    for i in cutlass.range_constexpr(vec_size):
                        sum_hq += r_h[i] * r_q[i]

                    # Warp-level reduction
                    for offset in [16, 8, 4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(
                            sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                        )

                    # Write output (only lane 0 of each group)
                    if lane_in_group == 0:
                        o[(i_n, i_t, i_hv, v_idx)] = cutlass.BFloat16(sum_hq)

                # Write final state back (if not disabled) using 3D local_tile + autovec_copy
                if cutlass.const_expr(not disable_state_update):
                    h_tile_out = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_idx, lane_in_group),
                    )
                    cute.autovec_copy(r_h, h_tile_out)


@cute.jit
def run_gdn_verify_kernel_mtp_original(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    vec_size: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Launcher for original MTP kernel (BS < 8)."""
    _, v_dim, k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    num_v_tiles = cute.ceil_div(v_dim, tile_v)

    # Grid: (B * HV * num_v_tiles, 1, 1)
    grid_size = B * HV * num_v_tiles

    # Shared memory for pre-computed q, k, g, beta
    smem_bytes = (
        4 * T * (k_dim + 8)  # sQ
        + 4 * T * (k_dim + 8)  # sK
        + 4 * T  # sG
        + 4 * T  # sBeta
        + 128  # alignment
    )

    gdn_verify_kernel_mtp_original(
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
        cu_seqlens,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_initial_state,
        use_qk_l2norm,
        is_varlen,
        disable_state_update,
        cache_intermediate_states,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# Optimized MTP kernel with ILP rows and SMEM v caching - used for BS >= 8
@cute.kernel
def gdn_verify_kernel_mtp(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] - initial state pool (K-last)
    intermediate_states: cute.Tensor,  # [pool_size * T * HV, V, K] - intermediate state cache
    vec_size: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],  # TILE_V - configurable for batch size
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, T, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    b: cute.Tensor,  # [B, T, HV]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices
    cu_seqlens: cute.Tensor,  # [B+1] - cumulative sequence lengths (for varlen)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    ilp_rows: cutlass.Constexpr[
        int
    ],  # 1, 2, 4, or 8: number of V-rows processed simultaneously
    use_smem_v: cutlass.Constexpr[
        bool
    ],  # True: preload v into SMEM (large BS), False: GMEM reads
):
    """
    Parallel MTP kernel - each block handles one [TILE_V, TILE_K] tile.

    Grid: (B * HV * num_v_tiles, 1, 1)
    Each block:
    - Loads its v_tile of state into registers
    - Processes all T time steps with state in registers
    - Writes output and optionally updates state

    This matches Triton's parallelization strategy for better small-batch performance.
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # Compute thread grouping based on vec_size:
    # vec_size=8: 16 threads per group (half-warp), 8 groups per block
    # vec_size=4: 32 threads per group (full warp), 4 groups per block
    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 16 or 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group  # 2 or 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp  # 8 or 4

    # Lane position within group and group index
    lane_in_group = lane_id % threads_per_group
    group_in_warp = lane_id // threads_per_group
    group_idx = warp_idx * groups_per_warp + group_in_warp

    batch_idx, _, _ = cute.arch.block_idx()

    # Decode block index: (i_n, i_hv, i_v) from batch_idx
    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    # Get initial state index for this batch
    cache_idx = h0_indices[i_n]

    # Load A_log and dt_bias once (they don't vary with time)
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])

    # Allocate shared memory for pre-computed values (broadcast to all warps)
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sK = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)
    # Shared memory for preloaded v values [T, tile_v] — avoids repeated GMEM loads
    sVdata = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
    )
    # Shared memory for output accumulation [T, tile_v] — enables coalesced GMEM writeback
    sOutput = smem.allocate_tensor(
        cutlass.BFloat16, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
    )

    # Register arrays for computation
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h2 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h3 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h4 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h5 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h6 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h7 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h8 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # BF16 register tensors for vectorized q, k loading
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    # Only process valid batch entries (cache_idx >= 0)
    if cache_idx >= 0:
        # Compute k_start once (used for shared memory writes)
        k_start = lane_in_group * vec_size

        # Pre-compute q, k, g, beta for ALL time steps ONCE (shared across warps)
        for i_t in cutlass.range_constexpr(T):
            # Load q, k into BF16 registers using autovec_copy (coalesced)
            q_tile = cute.local_tile(
                q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
            )
            k_tile = cute.local_tile(
                k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
            )
            cute.autovec_copy(q_tile, r_q_bf16)
            cute.autovec_copy(k_tile, r_k_bf16)

            # Convert BF16 to FP32 for computation
            for i in cutlass.range_constexpr(vec_size):
                r_q[i] = cutlass.Float32(r_q_bf16[i])
                r_k[i] = cutlass.Float32(r_k_bf16[i])

            # Apply L2 normalization to q, k (with scale fused for q)
            if cutlass.const_expr(use_qk_l2norm):
                sum_q = 0.0
                sum_k = 0.0
                for i in cutlass.range_constexpr(vec_size):
                    sum_q += r_q[i] * r_q[i]
                    sum_k += r_k[i] * r_k[i]

                # Warp-level reduction (32 threads per group with vec_size=4)
                for offset in [16, 8, 4, 2, 1]:
                    sum_q += cute.arch.shuffle_sync_bfly(
                        sum_q, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sum_k += cute.arch.shuffle_sync_bfly(
                        sum_k, offset=offset, mask=-1, mask_and_clamp=31
                    )

                # Fuse scale into q's normalization factor
                inv_norm_q_scaled = cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)

                for i in cutlass.range_constexpr(vec_size):
                    r_q[i] = r_q[i] * inv_norm_q_scaled
                    r_k[i] = r_k[i] * inv_norm_k
            else:
                # No L2 norm, just apply scale to q
                for i in cutlass.range_constexpr(vec_size):
                    r_q[i] = r_q[i] * scale

            # Store to shared memory (only first group writes) - contiguous layout
            if tidx < threads_per_group:
                for i in cutlass.range_constexpr(vec_size):
                    sQ[(i_t, k_start + i)] = r_q[i]
                    sK[(i_t, k_start + i)] = r_k[i]

            # Compute g, beta - all lanes compute (redundant but no divergence)
            r_a = cutlass.Float32(a[i_n, i_t, i_hv])
            r_b = cutlass.Float32(b[i_n, i_t, i_hv])

            x = r_a + r_dt_bias
            beta_x = softplus_beta * x

            # Branchless softplus
            exp_beta_x = cute.exp(beta_x, fastmath=True)
            softplus_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                cutlass.Float32(1.0) + exp_beta_x, fastmath=True
            )
            use_softplus = (
                cutlass.Float32(1.0)
                if beta_x <= softplus_threshold
                else cutlass.Float32(0.0)
            )
            softplus_x = (
                use_softplus * softplus_val + (cutlass.Float32(1.0) - use_softplus) * x
            )

            r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
            r_beta = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(-r_b, fastmath=True)
            )
            r_g = cute.exp(r_g_value, fastmath=True)

            # Only thread 0 stores to shared memory
            if tidx == 0:
                sG[i_t] = r_g
                sBeta[i_t] = r_beta

            # Cooperatively preload v values for this timestep into SMEM
            if cutlass.const_expr(use_smem_v):
                v_tile_start = i_v * tile_v
                if tidx < tile_v:
                    v_global_idx = v_tile_start + tidx
                    if v_global_idx < V:
                        sVdata[(i_t, tidx)] = cutlass.Float32(
                            v[i_n, i_t, i_hv, v_global_idx]
                        )

        cute.arch.barrier()

        # Each group handles tile_v/num_groups V rows
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        if cutlass.const_expr(ilp_rows == 8):
            # === 8-ROW ILP PATH: Process 8 V-rows simultaneously ===
            eighth_rows: cutlass.Constexpr[int] = rows_per_group // 8

            for row_oct in cutlass.range_constexpr(eighth_rows):
                v_base = i_v * tile_v + group_idx * rows_per_group + row_oct * 8
                v0 = v_base
                v1 = v_base + 1
                v2 = v_base + 2
                v3 = v_base + 3
                v4 = v_base + 4
                v5 = v_base + 5
                v6 = v_base + 6
                v7 = v_base + 7

                if v7 < V:
                    # Load h for ALL 8 V-rows (8 independent load streams)
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
                    cute.autovec_copy(ht0, r_h)
                    cute.autovec_copy(ht1, r_h2)
                    cute.autovec_copy(ht2, r_h3)
                    cute.autovec_copy(ht3, r_h4)
                    cute.autovec_copy(ht4, r_h5)
                    cute.autovec_copy(ht5, r_h6)
                    cute.autovec_copy(ht6, r_h7)
                    cute.autovec_copy(ht7, r_h8)

                    for i_t in cutlass.range_constexpr(T):
                        sQ_tile = cute.local_tile(
                            sQ, (1, vec_size), (i_t, lane_in_group)
                        )
                        sK_tile = cute.local_tile(
                            sK, (1, vec_size), (i_t, lane_in_group)
                        )
                        cute.autovec_copy(sQ_tile, r_q)
                        cute.autovec_copy(sK_tile, r_k)

                        r_g = sG[i_t]
                        r_beta = sBeta[i_t]

                        # Step 1: Decay all 8 h vectors
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[i] = r_h[i] * r_g
                            r_h2[i] = r_h2[i] * r_g
                            r_h3[i] = r_h3[i] * r_g
                            r_h4[i] = r_h4[i] * r_g
                            r_h5[i] = r_h5[i] * r_g
                            r_h6[i] = r_h6[i] * r_g
                            r_h7[i] = r_h7[i] * r_g
                            r_h8[i] = r_h8[i] * r_g

                        # Step 2: Dot products h@k for all 8 rows
                        s0 = 0.0
                        s1 = 0.0
                        s2 = 0.0
                        s3 = 0.0
                        s4 = 0.0
                        s5 = 0.0
                        s6 = 0.0
                        s7 = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            s0 += r_h[i] * r_k[i]
                            s1 += r_h2[i] * r_k[i]
                            s2 += r_h3[i] * r_k[i]
                            s3 += r_h4[i] * r_k[i]
                            s4 += r_h5[i] * r_k[i]
                            s5 += r_h6[i] * r_k[i]
                            s6 += r_h7[i] * r_k[i]
                            s7 += r_h8[i] * r_k[i]

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

                        # Step 3: Load v, delta rule
                        if cutlass.const_expr(use_smem_v):
                            vl = v0 - i_v * tile_v
                            rv0 = sVdata[(i_t, vl)]
                            rv1 = sVdata[(i_t, vl + 1)]
                            rv2 = sVdata[(i_t, vl + 2)]
                            rv3 = sVdata[(i_t, vl + 3)]
                            rv4 = sVdata[(i_t, vl + 4)]
                            rv5 = sVdata[(i_t, vl + 5)]
                            rv6 = sVdata[(i_t, vl + 6)]
                            rv7 = sVdata[(i_t, vl + 7)]
                        else:
                            rv0 = cutlass.Float32(v[i_n, i_t, i_hv, v0])
                            rv1 = cutlass.Float32(v[i_n, i_t, i_hv, v1])
                            rv2 = cutlass.Float32(v[i_n, i_t, i_hv, v2])
                            rv3 = cutlass.Float32(v[i_n, i_t, i_hv, v3])
                            rv4 = cutlass.Float32(v[i_n, i_t, i_hv, v4])
                            rv5 = cutlass.Float32(v[i_n, i_t, i_hv, v5])
                            rv6 = cutlass.Float32(v[i_n, i_t, i_hv, v6])
                            rv7 = cutlass.Float32(v[i_n, i_t, i_hv, v7])
                        vn0 = (rv0 - s0) * r_beta
                        vn1 = (rv1 - s1) * r_beta
                        vn2 = (rv2 - s2) * r_beta
                        vn3 = (rv3 - s3) * r_beta
                        vn4 = (rv4 - s4) * r_beta
                        vn5 = (rv5 - s5) * r_beta
                        vn6 = (rv6 - s6) * r_beta
                        vn7 = (rv7 - s7) * r_beta

                        # Step 4: Rank-1 update all 8 h vectors
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[i] += r_k[i] * vn0
                            r_h2[i] += r_k[i] * vn1
                            r_h3[i] += r_k[i] * vn2
                            r_h4[i] += r_k[i] * vn3
                            r_h5[i] += r_k[i] * vn4
                            r_h6[i] += r_k[i] * vn5
                            r_h7[i] += r_k[i] * vn6
                            r_h8[i] += r_k[i] * vn7

                        # Cache intermediate state if needed
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            it0 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v0, lane_in_group),
                            )
                            cute.autovec_copy(r_h, it0)
                            it1 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v1, lane_in_group),
                            )
                            cute.autovec_copy(r_h2, it1)
                            it2 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v2, lane_in_group),
                            )
                            cute.autovec_copy(r_h3, it2)
                            it3 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v3, lane_in_group),
                            )
                            cute.autovec_copy(r_h4, it3)
                            it4 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v4, lane_in_group),
                            )
                            cute.autovec_copy(r_h5, it4)
                            it5 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v5, lane_in_group),
                            )
                            cute.autovec_copy(r_h6, it5)
                            it6 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v6, lane_in_group),
                            )
                            cute.autovec_copy(r_h7, it6)
                            it7 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v7, lane_in_group),
                            )
                            cute.autovec_copy(r_h8, it7)

                        # Step 5: Output dot products h@q for all 8 rows
                        o0 = 0.0
                        o1 = 0.0
                        o2 = 0.0
                        o3 = 0.0
                        o4 = 0.0
                        o5 = 0.0
                        o6 = 0.0
                        o7 = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            o0 += r_h[i] * r_q[i]
                            o1 += r_h2[i] * r_q[i]
                            o2 += r_h3[i] * r_q[i]
                            o3 += r_h4[i] * r_q[i]
                            o4 += r_h5[i] * r_q[i]
                            o5 += r_h6[i] * r_q[i]
                            o6 += r_h7[i] * r_q[i]
                            o7 += r_h8[i] * r_q[i]

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

                        if lane_in_group == 0:
                            if cutlass.const_expr(use_smem_v):
                                vl0 = v0 - i_v * tile_v
                                sOutput[(i_t, vl0)] = cutlass.BFloat16(o0)
                                sOutput[(i_t, vl0 + 1)] = cutlass.BFloat16(o1)
                                sOutput[(i_t, vl0 + 2)] = cutlass.BFloat16(o2)
                                sOutput[(i_t, vl0 + 3)] = cutlass.BFloat16(o3)
                                sOutput[(i_t, vl0 + 4)] = cutlass.BFloat16(o4)
                                sOutput[(i_t, vl0 + 5)] = cutlass.BFloat16(o5)
                                sOutput[(i_t, vl0 + 6)] = cutlass.BFloat16(o6)
                                sOutput[(i_t, vl0 + 7)] = cutlass.BFloat16(o7)
                            else:
                                o[(i_n, i_t, i_hv, v0)] = cutlass.BFloat16(o0)
                                o[(i_n, i_t, i_hv, v1)] = cutlass.BFloat16(o1)
                                o[(i_n, i_t, i_hv, v2)] = cutlass.BFloat16(o2)
                                o[(i_n, i_t, i_hv, v3)] = cutlass.BFloat16(o3)
                                o[(i_n, i_t, i_hv, v4)] = cutlass.BFloat16(o4)
                                o[(i_n, i_t, i_hv, v5)] = cutlass.BFloat16(o5)
                                o[(i_n, i_t, i_hv, v6)] = cutlass.BFloat16(o6)
                                o[(i_n, i_t, i_hv, v7)] = cutlass.BFloat16(o7)

                    # Write final state back for all 8 rows
                    if cutlass.const_expr(not disable_state_update):
                        ht_o0 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v0, lane_in_group),
                        )
                        cute.autovec_copy(r_h, ht_o0)
                        ht_o1 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v1, lane_in_group),
                        )
                        cute.autovec_copy(r_h2, ht_o1)
                        ht_o2 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v2, lane_in_group),
                        )
                        cute.autovec_copy(r_h3, ht_o2)
                        ht_o3 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v3, lane_in_group),
                        )
                        cute.autovec_copy(r_h4, ht_o3)
                        ht_o4 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v4, lane_in_group),
                        )
                        cute.autovec_copy(r_h5, ht_o4)
                        ht_o5 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v5, lane_in_group),
                        )
                        cute.autovec_copy(r_h6, ht_o5)
                        ht_o6 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v6, lane_in_group),
                        )
                        cute.autovec_copy(r_h7, ht_o6)
                        ht_o7 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v7, lane_in_group),
                        )
                        cute.autovec_copy(r_h8, ht_o7)
        elif cutlass.const_expr(ilp_rows == 4):
            # === 4-ROW ILP PATH: Process 4 V-rows simultaneously ===
            quarter_rows: cutlass.Constexpr[int] = rows_per_group // 4

            for row_quad in cutlass.range_constexpr(quarter_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_quad * 4
                v_idx_b = v_idx_a + 1
                v_idx_c = v_idx_a + 2
                v_idx_d = v_idx_a + 3

                if v_idx_d < V:
                    # Load h for ALL 4 V-rows simultaneously (ILP: 4 independent load streams)
                    h_tile_a = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_idx_a, lane_in_group),
                    )
                    h_tile_b = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_idx_b, lane_in_group),
                    )
                    h_tile_c = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_idx_c, lane_in_group),
                    )
                    h_tile_d = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_idx_d, lane_in_group),
                    )
                    cute.autovec_copy(h_tile_a, r_h)
                    cute.autovec_copy(h_tile_b, r_h2)
                    cute.autovec_copy(h_tile_c, r_h3)
                    cute.autovec_copy(h_tile_d, r_h4)

                    # Process all T time steps with all 4 h vectors in registers
                    for i_t in cutlass.range_constexpr(T):
                        # Load pre-computed q, k from shared memory (shared between all rows)
                        sQ_tile = cute.local_tile(
                            sQ, (1, vec_size), (i_t, lane_in_group)
                        )
                        sK_tile = cute.local_tile(
                            sK, (1, vec_size), (i_t, lane_in_group)
                        )
                        cute.autovec_copy(sQ_tile, r_q)
                        cute.autovec_copy(sK_tile, r_k)

                        r_g = sG[i_t]
                        r_beta = sBeta[i_t]

                        # Step 1: Apply decay to ALL 4 h vectors (ILP)
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[i] = r_h[i] * r_g
                            r_h2[i] = r_h2[i] * r_g
                            r_h3[i] = r_h3[i] * r_g
                            r_h4[i] = r_h4[i] * r_g

                        # Step 2: Compute dot products for ALL 4 rows (ILP)
                        sum_hk_a = 0.0
                        sum_hk_b = 0.0
                        sum_hk_c = 0.0
                        sum_hk_d = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hk_a += r_h[i] * r_k[i]
                            sum_hk_b += r_h2[i] * r_k[i]
                            sum_hk_c += r_h3[i] * r_k[i]
                            sum_hk_d += r_h4[i] * r_k[i]

                        # Warp-level reduction for ALL 4 (interleaved shuffles)
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hk_a += cute.arch.shuffle_sync_bfly(
                                sum_hk_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_b += cute.arch.shuffle_sync_bfly(
                                sum_hk_b, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_c += cute.arch.shuffle_sync_bfly(
                                sum_hk_c, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_d += cute.arch.shuffle_sync_bfly(
                                sum_hk_d, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Step 3: Load v for ALL 4 rows, apply delta rule
                        if cutlass.const_expr(use_smem_v):
                            v_local_a = v_idx_a - i_v * tile_v
                            r_v_a = sVdata[(i_t, v_local_a)]
                            r_v_b = sVdata[(i_t, v_local_a + 1)]
                            r_v_c = sVdata[(i_t, v_local_a + 2)]
                            r_v_d = sVdata[(i_t, v_local_a + 3)]
                        else:
                            r_v_a = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_b])
                            r_v_c = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_c])
                            r_v_d = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_d])
                        v_new_a = (r_v_a - sum_hk_a) * r_beta
                        v_new_b = (r_v_b - sum_hk_b) * r_beta
                        v_new_c = (r_v_c - sum_hk_c) * r_beta
                        v_new_d = (r_v_d - sum_hk_d) * r_beta

                        # Step 4: Update ALL 4 h vectors (ILP)
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[i] += r_k[i] * v_new_a
                            r_h2[i] += r_k[i] * v_new_b
                            r_h3[i] += r_k[i] * v_new_c
                            r_h4[i] += r_k[i] * v_new_d

                        # Cache intermediate state if needed
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_tile_a = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_a, lane_in_group),
                            )
                            cute.autovec_copy(r_h, inter_tile_a)
                            inter_tile_b = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_b, lane_in_group),
                            )
                            cute.autovec_copy(r_h2, inter_tile_b)
                            inter_tile_c = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_c, lane_in_group),
                            )
                            cute.autovec_copy(r_h3, inter_tile_c)
                            inter_tile_d = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_d, lane_in_group),
                            )
                            cute.autovec_copy(r_h4, inter_tile_d)

                        # Step 5: Compute output for ALL 4 rows (ILP)
                        sum_hq_a = 0.0
                        sum_hq_b = 0.0
                        sum_hq_c = 0.0
                        sum_hq_d = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hq_a += r_h[i] * r_q[i]
                            sum_hq_b += r_h2[i] * r_q[i]
                            sum_hq_c += r_h3[i] * r_q[i]
                            sum_hq_d += r_h4[i] * r_q[i]

                        # Warp-level reduction for ALL 4 (interleaved)
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hq_a += cute.arch.shuffle_sync_bfly(
                                sum_hq_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_b += cute.arch.shuffle_sync_bfly(
                                sum_hq_b, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_c += cute.arch.shuffle_sync_bfly(
                                sum_hq_c, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_d += cute.arch.shuffle_sync_bfly(
                                sum_hq_d, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Write output for ALL 4 rows (to SMEM for batched writeback)
                        if lane_in_group == 0:
                            if cutlass.const_expr(use_smem_v):
                                vla = v_idx_a - i_v * tile_v
                                sOutput[(i_t, vla)] = cutlass.BFloat16(sum_hq_a)
                                sOutput[(i_t, vla + 1)] = cutlass.BFloat16(sum_hq_b)
                                sOutput[(i_t, vla + 2)] = cutlass.BFloat16(sum_hq_c)
                                sOutput[(i_t, vla + 3)] = cutlass.BFloat16(sum_hq_d)
                            else:
                                o[(i_n, i_t, i_hv, v_idx_a)] = cutlass.BFloat16(
                                    sum_hq_a
                                )
                                o[(i_n, i_t, i_hv, v_idx_b)] = cutlass.BFloat16(
                                    sum_hq_b
                                )
                                o[(i_n, i_t, i_hv, v_idx_c)] = cutlass.BFloat16(
                                    sum_hq_c
                                )
                                o[(i_n, i_t, i_hv, v_idx_d)] = cutlass.BFloat16(
                                    sum_hq_d
                                )

                    # Write final state back for ALL 4 rows (if not disabled)
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out_a = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_a, lane_in_group),
                        )
                        cute.autovec_copy(r_h, h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        cute.autovec_copy(r_h2, h_tile_out_b)
                        h_tile_out_c = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_c, lane_in_group),
                        )
                        cute.autovec_copy(r_h3, h_tile_out_c)
                        h_tile_out_d = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_d, lane_in_group),
                        )
                        cute.autovec_copy(r_h4, h_tile_out_d)
        elif cutlass.const_expr(ilp_rows == 2):
            # === 2-ROW ILP PATH: Process 2 V-rows simultaneously ===
            half_rows: cutlass.Constexpr[int] = rows_per_group // 2

            for row_pair in cutlass.range_constexpr(half_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_pair * 2
                v_idx_b = v_idx_a + 1

                if v_idx_b < V:
                    # Load h for BOTH V-rows simultaneously (ILP: 2 independent load streams)
                    h_tile_a = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_idx_a, lane_in_group),
                    )
                    h_tile_b = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_idx_b, lane_in_group),
                    )
                    cute.autovec_copy(h_tile_a, r_h)
                    cute.autovec_copy(h_tile_b, r_h2)

                    # Process all T time steps with both h vectors in registers
                    for i_t in cutlass.range_constexpr(T):
                        # Load pre-computed q, k from shared memory (shared between both rows)
                        sQ_tile = cute.local_tile(
                            sQ, (1, vec_size), (i_t, lane_in_group)
                        )
                        sK_tile = cute.local_tile(
                            sK, (1, vec_size), (i_t, lane_in_group)
                        )
                        cute.autovec_copy(sQ_tile, r_q)
                        cute.autovec_copy(sK_tile, r_k)

                        r_g = sG[i_t]
                        r_beta = sBeta[i_t]

                        # Step 1: Apply decay to BOTH h vectors (ILP)
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[i] = r_h[i] * r_g
                            r_h2[i] = r_h2[i] * r_g

                        # Step 2: Compute dot products for BOTH rows (ILP)
                        sum_hk_a = 0.0
                        sum_hk_b = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hk_a += r_h[i] * r_k[i]
                            sum_hk_b += r_h2[i] * r_k[i]

                        # Warp-level reduction for BOTH (interleaved shuffles)
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hk_a += cute.arch.shuffle_sync_bfly(
                                sum_hk_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_b += cute.arch.shuffle_sync_bfly(
                                sum_hk_b, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Step 3: Load v for BOTH rows, apply delta rule
                        if cutlass.const_expr(use_smem_v):
                            v_local_a = v_idx_a - i_v * tile_v
                            r_v_a = sVdata[(i_t, v_local_a)]
                            r_v_b = sVdata[(i_t, v_local_a + 1)]
                        else:
                            r_v_a = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_a])
                            r_v_b = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_b])
                        v_new_a = (r_v_a - sum_hk_a) * r_beta
                        v_new_b = (r_v_b - sum_hk_b) * r_beta

                        # Step 4: Update BOTH h vectors (ILP)
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[i] += r_k[i] * v_new_a
                            r_h2[i] += r_k[i] * v_new_b

                        # Cache intermediate state if needed
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_tile_a = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_a, lane_in_group),
                            )
                            cute.autovec_copy(r_h, inter_tile_a)
                            inter_tile_b = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_b, lane_in_group),
                            )
                            cute.autovec_copy(r_h2, inter_tile_b)

                        # Step 5: Compute output for BOTH rows (ILP)
                        sum_hq_a = 0.0
                        sum_hq_b = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hq_a += r_h[i] * r_q[i]
                            sum_hq_b += r_h2[i] * r_q[i]

                        # Warp-level reduction for BOTH (interleaved)
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hq_a += cute.arch.shuffle_sync_bfly(
                                sum_hq_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_b += cute.arch.shuffle_sync_bfly(
                                sum_hq_b, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Write output for BOTH rows
                        if lane_in_group == 0:
                            if cutlass.const_expr(use_smem_v):
                                vla2 = v_idx_a - i_v * tile_v
                                sOutput[(i_t, vla2)] = cutlass.BFloat16(sum_hq_a)
                                sOutput[(i_t, vla2 + 1)] = cutlass.BFloat16(sum_hq_b)
                            else:
                                o[(i_n, i_t, i_hv, v_idx_a)] = cutlass.BFloat16(
                                    sum_hq_a
                                )
                                o[(i_n, i_t, i_hv, v_idx_b)] = cutlass.BFloat16(
                                    sum_hq_b
                                )

                    # Write final state back for BOTH rows (if not disabled)
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out_a = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_a, lane_in_group),
                        )
                        cute.autovec_copy(r_h, h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        cute.autovec_copy(r_h2, h_tile_out_b)
        else:
            # === 1-ROW PATH: Process 1 V-row at a time (lower register pressure) ===
            for row_idx in cutlass.range_constexpr(rows_per_group):
                v_idx = i_v * tile_v + group_idx * rows_per_group + row_idx

                if v_idx < V:
                    # Load h for this V-row
                    h_tile = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_idx, lane_in_group),
                    )
                    cute.autovec_copy(h_tile, r_h)

                    # Process all T time steps with h in registers
                    for i_t in cutlass.range_constexpr(T):
                        sQ_tile = cute.local_tile(
                            sQ, (1, vec_size), (i_t, lane_in_group)
                        )
                        sK_tile = cute.local_tile(
                            sK, (1, vec_size), (i_t, lane_in_group)
                        )
                        cute.autovec_copy(sQ_tile, r_q)
                        cute.autovec_copy(sK_tile, r_k)

                        r_g = sG[i_t]
                        r_beta = sBeta[i_t]

                        # Step 1: Apply decay
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[i] = r_h[i] * r_g

                        # Step 2: Dot product h @ k
                        sum_hk = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hk += r_h[i] * r_k[i]

                        for offset in [16, 8, 4, 2, 1]:
                            sum_hk += cute.arch.shuffle_sync_bfly(
                                sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Step 3: Delta rule
                        if cutlass.const_expr(use_smem_v):
                            v_local = v_idx - i_v * tile_v
                            r_v_val = sVdata[(i_t, v_local)]
                        else:
                            r_v_val = cutlass.Float32(v[i_n, i_t, i_hv, v_idx])
                        v_new = (r_v_val - sum_hk) * r_beta

                        # Step 4: Rank-1 update
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[i] += r_k[i] * v_new

                        # Cache intermediate state if needed
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_tile = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx, lane_in_group),
                            )
                            cute.autovec_copy(r_h, inter_tile)

                        # Step 5: Output dot product h @ q
                        sum_hq = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hq += r_h[i] * r_q[i]

                        for offset in [16, 8, 4, 2, 1]:
                            sum_hq += cute.arch.shuffle_sync_bfly(
                                sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                            )

                        # Write output
                        if lane_in_group == 0:
                            if cutlass.const_expr(use_smem_v):
                                vl1 = v_idx - i_v * tile_v
                                sOutput[(i_t, vl1)] = cutlass.BFloat16(sum_hq)
                            else:
                                o[(i_n, i_t, i_hv, v_idx)] = cutlass.BFloat16(sum_hq)

                    # Write final state back (if not disabled)
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx, lane_in_group),
                        )
                        cute.autovec_copy(r_h, h_tile_out)

        # === Cooperative output writeback from SMEM to GMEM (only if use_smem_v) ===
        if cutlass.const_expr(use_smem_v):
            cute.arch.barrier()  # Ensure all groups finished writing to sOutput
            v_tile_base = i_v * tile_v
            for t_idx in cutlass.range_constexpr(T):
                # 128 threads, tile_v values to write per timestep
                if tidx < tile_v:
                    v_global = v_tile_base + tidx
                    if v_global < V:
                        o[(i_n, t_idx, i_hv, v_global)] = sOutput[(t_idx, tidx)]


@cute.jit
def run_gdn_verify_kernel_mtp(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],  # TILE_V - configurable for batch size
    vec_size: cutlass.Constexpr[int],  # 4 for full warp, 8 for half-warp
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    ilp_rows: cutlass.Constexpr[int],
    use_smem_v: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    _, v_dim, k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    num_v_tiles = cute.ceil_div(v_dim, tile_v)

    # Grid: (B * HV * num_v_tiles, 1, 1) - parallelize across V dimension
    grid_size = B * HV * num_v_tiles

    # Shared memory for pre-computed q, k, g, beta, preloaded v data, and output
    smem_bytes = (
        4 * T * (k_dim + 8)  # sQ
        + 4 * T * (k_dim + 8)  # sK
        + 4 * T  # sG
        + 4 * T  # sBeta
        + 4 * T * tile_v  # sVdata (v values for all timesteps)
        + 2 * T * tile_v  # sOutput (output accumulation in BF16)
        + 128  # alignment
    )

    gdn_verify_kernel_mtp(
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
        cu_seqlens,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_initial_state,
        use_qk_l2norm,
        is_varlen,
        disable_state_update,
        cache_intermediate_states,
        ilp_rows,
        use_smem_v,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ============================================================================
# SMEM-Resident State MTP Kernel — higher occupancy via reduced register pressure
# ============================================================================
# Design rationale:
# - State tile [TILE_V, K] kept in shared memory instead of registers
# - V-row loop is runtime (not constexpr) → compiler reuses registers across rows
# - T-loop stays constexpr for optimal unrolling per row
# - Cooperative GMEM↔SMEM transfers using all threads
# - Target: ~40 regs/thread → 8+ CTAs/SM → much better latency hiding

TILE_V_SMEM = 64  # V-tile size for SMEM kernel (same as max dynamic selector)
NUM_THREADS_SMEM = 128  # 4 warps
VEC_SIZE_SMEM = 4  # Elements per thread for K dimension (32 threads × 4 = 128 = K)


@cute.kernel
def gdn_verify_kernel_mtp_smem(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] - initial state pool (K-last)
    intermediate_states: cute.Tensor,  # [pool_size * T * HV, V, K] - intermediate state cache
    tile_v: cutlass.Constexpr[int],  # TILE_V - configurable for batch size
    num_v_tiles: cutlass.Constexpr[int],
    A_log: cute.Tensor,  # [HV]
    a: cute.Tensor,  # [B, T, HV]
    dt_bias: cute.Tensor,  # [HV]
    q: cute.Tensor,  # [B, T, H, K]
    k: cute.Tensor,  # [B, T, H, K]
    v: cute.Tensor,  # [B, T, HV, V]
    b: cute.Tensor,  # [B, T, HV]
    o: cute.Tensor,  # [B, T, HV, V] - output
    h0_indices: cute.Tensor,  # [B] - initial state indices
    cu_seqlens: cute.Tensor,  # [B+1] - cumulative sequence lengths (for varlen)
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
):
    """
    SMEM-resident state MTP kernel — reduced register pressure for higher occupancy.

    Key differences from gdn_verify_kernel_mtp:
    - State tile [tile_v, K] kept in SMEM (not registers)
    - V-row loop is runtime (not constexpr) → register reuse across rows
    - Cooperative GMEM↔SMEM transfers for state loading/storing
    - Same correctness, much lower register pressure → more CTAs/SM → better perf

    Grid: (B * HV * num_v_tiles, 1, 1)
    Block: (128 threads = 4 warps)
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # Thread grouping: vec_size=4, 32 threads per group (full warp), 4 groups
    vec_size: cutlass.Constexpr[int] = VEC_SIZE_SMEM
    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group  # 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp  # 4

    lane_in_group = lane_id % threads_per_group
    group_in_warp = lane_id // threads_per_group
    group_idx = warp_idx * groups_per_warp + group_in_warp

    batch_idx, _, _ = cute.arch.block_idx()

    # Decode block index: (i_n, i_hv, i_v) from batch_idx
    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    # Get initial state index for this batch
    cache_idx = h0_indices[i_n]

    # Load A_log and dt_bias once (they don't vary with time)
    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])

    # ===================================================================
    # Allocate shared memory
    # ===================================================================
    smem = cutlass.utils.SmemAllocator()

    # State tile: [tile_v, K] in SMEM — the key change for occupancy
    # Use padded stride to avoid bank conflicts: stride = K + 4
    sH = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((tile_v, K), stride=(K + 4, 1)), 128
    )

    # Pre-computed q, k, gates in SMEM (same as original)
    sQ = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sK = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sG = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)
    sBeta = smem.allocate_tensor(cutlass.Float32, cute.make_layout((T,)), 16)

    # Register arrays for computation (small — only need one row at a time)
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    # BF16 staging registers
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )

    # Only process valid batch entries
    if cache_idx >= 0:
        k_start = lane_in_group * vec_size
        flat_state_idx = cache_idx * HV + i_hv

        # ===================================================================
        # Phase 0: Cooperative load of state tile GMEM → SMEM
        # Each thread loads multiple elements to cover tile_v × K
        # ===================================================================
        # Total elements = tile_v * K. Each thread loads (tile_v * K) / 128 elements
        # With 128 threads: each thread handles (tile_v * K) / 128 = (64*128)/128 = 64 elems
        # We use a strided pattern: thread t handles elements t, t+128, t+256, ...
        total_elems: cutlass.Constexpr[int] = tile_v * K
        elems_per_thread: cutlass.Constexpr[int] = total_elems // NUM_THREADS_SMEM

        for elem_iter in cutlass.range_constexpr(elems_per_thread):
            flat_idx = tidx + elem_iter * NUM_THREADS_SMEM
            v_row = flat_idx // K
            k_col = flat_idx % K
            v_global = i_v * tile_v + v_row
            if v_global < V:
                sH[(v_row, k_col)] = h0_source[(flat_state_idx, v_global, k_col)]

        cute.arch.barrier()

        # ===================================================================
        # Phase 1: Pre-compute q, k, g, beta for ALL time steps (same as original)
        # ===================================================================
        for i_t in cutlass.range_constexpr(T):
            q_tile = cute.local_tile(
                q, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
            )
            k_tile = cute.local_tile(
                k, (1, 1, 1, vec_size), (i_n, i_t, i_h, lane_in_group)
            )
            cute.autovec_copy(q_tile, r_q_bf16)
            cute.autovec_copy(k_tile, r_k_bf16)

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

                inv_norm_q_scaled = cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)

                for i in cutlass.range_constexpr(vec_size):
                    r_q[i] = r_q[i] * inv_norm_q_scaled
                    r_k[i] = r_k[i] * inv_norm_k
            else:
                for i in cutlass.range_constexpr(vec_size):
                    r_q[i] = r_q[i] * scale

            if tidx < threads_per_group:
                for i in cutlass.range_constexpr(vec_size):
                    sQ[(i_t, k_start + i)] = r_q[i]
                    sK[(i_t, k_start + i)] = r_k[i]

            # Compute g, beta — all lanes compute (redundant but no divergence)
            r_a = cutlass.Float32(a[i_n, i_t, i_hv])
            r_b = cutlass.Float32(b[i_n, i_t, i_hv])

            x = r_a + r_dt_bias
            beta_x = softplus_beta * x

            exp_beta_x = cute.exp(beta_x, fastmath=True)
            softplus_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                cutlass.Float32(1.0) + exp_beta_x, fastmath=True
            )
            use_softplus = (
                cutlass.Float32(1.0)
                if beta_x <= softplus_threshold
                else cutlass.Float32(0.0)
            )
            softplus_x = (
                use_softplus * softplus_val + (cutlass.Float32(1.0) - use_softplus) * x
            )

            r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
            r_beta = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(-r_b, fastmath=True)
            )
            r_g = cute.exp(r_g_value, fastmath=True)

            if tidx == 0:
                sG[i_t] = r_g
                sBeta[i_t] = r_beta

        cute.arch.barrier()

        # ===================================================================
        # Phase 2: State update loop — V-rows from SMEM (runtime loop!)
        # Each group handles tile_v/num_groups rows
        # ===================================================================
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        for row_in_group in range(rows_per_group):
            v_local = group_idx * rows_per_group + row_in_group
            v_idx = i_v * tile_v + v_local

            if v_idx < V:
                # Load h[v_local, :] from SMEM into r_h
                for i in cutlass.range_constexpr(vec_size):
                    r_h[i] = sH[(v_local, k_start + i)]

                # Process all T time steps with h in registers
                for i_t in cutlass.range_constexpr(T):
                    # Load pre-computed q, k from shared memory
                    sQ_tile = cute.local_tile(sQ, (1, vec_size), (i_t, lane_in_group))
                    sK_tile = cute.local_tile(sK, (1, vec_size), (i_t, lane_in_group))
                    cute.autovec_copy(sQ_tile, r_q)
                    cute.autovec_copy(sK_tile, r_k)

                    r_g = sG[i_t]
                    r_beta = sBeta[i_t]

                    # Step 1+2 fused: Decay h AND compute sum_hk = h @ k
                    sum_hk = 0.0
                    for i in cutlass.range_constexpr(vec_size):
                        r_h[i] = r_h[i] * r_g
                        sum_hk += r_h[i] * r_k[i]

                    for offset in [16, 8, 4, 2, 1]:
                        sum_hk += cute.arch.shuffle_sync_bfly(
                            sum_hk, offset=offset, mask=-1, mask_and_clamp=31
                        )

                    # Step 3: Delta rule
                    r_v = cutlass.Float32(v[i_n, i_t, i_hv, v_idx])
                    v_new = (r_v - sum_hk) * r_beta

                    # Step 4+5 fused: Update h AND compute output dot product
                    sum_hq = 0.0
                    for i in cutlass.range_constexpr(vec_size):
                        r_h[i] += r_k[i] * v_new
                        sum_hq += r_h[i] * r_q[i]

                    # Cache intermediate state if needed
                    if cutlass.const_expr(cache_intermediate_states):
                        flat_idx = i_n * T * HV + i_t * HV + i_hv
                        inter_tile = cute.local_tile(
                            intermediate_states,
                            (1, 1, vec_size),
                            (flat_idx, v_idx, lane_in_group),
                        )
                        cute.autovec_copy(r_h, inter_tile)

                    for offset in [16, 8, 4, 2, 1]:
                        sum_hq += cute.arch.shuffle_sync_bfly(
                            sum_hq, offset=offset, mask=-1, mask_and_clamp=31
                        )

                    if lane_in_group == 0:
                        o[(i_n, i_t, i_hv, v_idx)] = cutlass.BFloat16(sum_hq)

                # Store updated h back to SMEM
                for i in cutlass.range_constexpr(vec_size):
                    sH[(v_local, k_start + i)] = r_h[i]

        cute.arch.barrier()

        # ===================================================================
        # Phase 3: Cooperative store of state tile SMEM → GMEM (if state update enabled)
        # ===================================================================
        if cutlass.const_expr(not disable_state_update):
            for elem_iter in cutlass.range_constexpr(elems_per_thread):
                flat_idx = tidx + elem_iter * NUM_THREADS_SMEM
                v_row = flat_idx // K
                k_col = flat_idx % K
                v_global = i_v * tile_v + v_row
                if v_global < V:
                    h0_source[(flat_state_idx, v_global, k_col)] = sH[(v_row, k_col)]


@cute.jit
def run_gdn_verify_kernel_mtp_smem(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b: cute.Tensor,
    o: cute.Tensor,
    h0_indices: cute.Tensor,
    cu_seqlens: cute.Tensor,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
    scale: cutlass.Constexpr[float],
    HV: cutlass.Constexpr[int],
    B: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    K: cutlass.Constexpr[int],
    V: cutlass.Constexpr[int],
    tile_v: cutlass.Constexpr[int],
    use_initial_state: cutlass.Constexpr[bool],
    use_qk_l2norm: cutlass.Constexpr[bool],
    is_varlen: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    _, v_dim, k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    grid_size = B * HV * num_v_tiles

    # Shared memory: sH + sQ + sK + sG + sBeta + alignment
    smem_bytes = (
        4 * tile_v * (k_dim + 4)  # sH with padding
        + 4 * T * (k_dim + 8)  # sQ
        + 4 * T * (k_dim + 8)  # sK
        + 4 * T  # sG
        + 4 * T  # sBeta
        + 256  # alignment
    )

    gdn_verify_kernel_mtp_smem(
        h0_source,
        intermediate_states,
        tile_v,
        num_v_tiles,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        h0_indices,
        cu_seqlens,
        softplus_beta,
        softplus_threshold,
        scale,
        HV,
        B,
        T,
        H,
        K,
        V,
        use_initial_state,
        use_qk_l2norm,
        is_varlen,
        disable_state_update,
        cache_intermediate_states,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_SMEM, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


@functools.cache
def _get_compiled_mtp_smem_kernel(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    cache_steps: int,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    scale: float,
    use_qk_l2norm: bool,
    tile_v: int,
):
    """Cache compiled SMEM-resident MTP kernel."""
    return {}


@functools.cache
def _get_compiled_mtp_kernel_original(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    cache_steps: int,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    scale: float,
    use_qk_l2norm: bool,
    tile_v: int,
    vec_size: int,
):
    """Cache compiled original MTP kernel (BS < 8) for given configuration."""
    return {}


@functools.cache
def _get_compiled_mtp_kernel(
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    cache_steps: int,
    disable_state_update: bool,
    cache_intermediate_states: bool,
    scale: float,
    use_qk_l2norm: bool,
    tile_v: int,  # TILE_V - configurable for batch size
    vec_size: int,  # 4 for full warp, 8 for half-warp
    use_2row_ilp: bool = True,  # Whether to use 2-row ILP (legacy, kept for compat)
    ilp_rows: int = 4,  # Number of ILP rows (1, 2, or 4)
):
    """Cache compiled optimized MTP kernel (BS >= 8) for given configuration."""
    return {}


@flashinfer_api
def gated_delta_rule_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gated Delta Rule MTP Kernel (Multiple Token Processing).

    This function processes multiple tokens (T > 1) in sequence, typically used for
    speculative decoding verification. It supports intermediate state caching for
    potential rollback scenarios.

    Args:
        q (torch.Tensor):
            Query tensor of shape ``[B, T, H, K]``.
        k (torch.Tensor):
            Key tensor of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Value tensor of shape ``[B, T, HV, V]``.
        initial_state (torch.Tensor):
            Initial state tensor of shape ``[pool_size, HV, V, K]`` (K-last layout).
        initial_state_indices (torch.Tensor):
            Indices mapping each batch to its initial state, shape ``[B]``.
        A_log (torch.Tensor):
            Log decay parameter of shape ``[HV]``.
        a (torch.Tensor):
            Input-dependent decay of shape ``[B, T, HV]``.
        dt_bias (torch.Tensor):
            Decay bias of shape ``[HV]``.
        b (torch.Tensor):
            Update gate input of shape ``[B, T, HV]``.
        scale (Optional[float]):
            Scaling factor for queries. If None, uses ``1/sqrt(K)``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[B, T, HV, V]``.
        intermediate_states_buffer (Optional[torch.Tensor]):
            Buffer for caching intermediate states, shape ``[pool_size, T, HV, V, K]``.
            If None, intermediate states are not cached.
        disable_state_update (bool):
            If True, the initial state is not updated. Default: ``True``.
        use_qk_l2norm (bool):
            Whether to apply L2 normalization to q and k. Default: ``True``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - output: Output tensor of shape ``[B, T, HV, V]``
            - initial_state: Updated state tensor (unchanged if disable_state_update=True)

    Note:
        - Requires SM90 (Hopper) architecture
        - Supports T > 1 (multiple token processing)
        - State layout is K-last: [pool_size, HV, V, K]
        - Optimized for speculative decoding verification scenarios
    """
    # Validate input shapes
    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    pool_size = initial_state.shape[0]

    # Dynamic TILE_V and vec_size selection based on batch size and sequence length
    tile_v = get_tile_v_mtp(B, T)
    vec_size = get_vec_size_mtp(B, T)

    # Validate state shape
    assert initial_state.shape == (pool_size, HV, V, K), (
        f"Expected initial_state shape [pool_size={pool_size}, HV={HV}, V={V}, K={K}], got {initial_state.shape}"
    )

    # Validate K and V constraints
    assert K >= 128, f"K must be at least 128, got K={K}"
    assert V >= 128, f"V must be at least 128, got V={V}"
    assert V % tile_v == 0, (
        f"V must be divisible by {tile_v} to prevent out-of-bounds access, got V={V}"
    )

    # Validate dtypes
    assert q.dtype in (torch.float16, torch.bfloat16), (
        f"q must be float16/bfloat16, got {q.dtype}"
    )
    assert initial_state.dtype == torch.float32, (
        f"initial_state must be float32, got {initial_state.dtype}"
    )
    assert A_log.dtype == torch.float32, f"A_log must be float32, got {A_log.dtype}"

    # Set default scale
    if scale is None:
        scale = K**-0.5

    # Allocate output if not provided
    output_provided = output is not None
    target_dtype = output.dtype if output_provided else q.dtype

    if output is None:
        output = torch.zeros((B, T, HV, V), dtype=torch.bfloat16, device=q.device)

    # Reshape initial_state from [pool_size, HV, V, K] to [pool_size * HV, V, K]
    h0_source = initial_state.to(torch.float32).reshape(pool_size * HV, V, K)

    # Handle intermediate states
    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]

        # Validate buffer length matches query sequence length
        assert cache_steps >= T, (
            f"intermediate_states_buffer second dimension (cache_steps={cache_steps}) must be at least T={T} to prevent out-of-bounds indexing"
        )

        intermediate_states = (
            intermediate_states_buffer.to(torch.float32)
            .reshape(buffer_size * cache_steps * HV, V, K)
            .contiguous()
        )
    else:
        cache_steps = T
        intermediate_states = torch.zeros(1, 1, 1, dtype=torch.float32, device=q.device)

    # Select kernel variant based on batch size:
    # - BS < 8: Use original kernel (no ILP overhead)
    # - BS >= 8: Use optimized kernel (ILP rows + SMEM v caching)
    use_original_kernel = B < 8
    use_smem_kernel = False  # Disabled: SMEM kernel is slower (higher SMEM latency outweighs occupancy gains)

    if use_original_kernel:
        # Original kernel for small batch sizes (BS < 8) - no ILP overhead
        cache_key_orig = (
            B,
            T,
            H,
            HV,
            K,
            V,
            pool_size,
            cache_steps,
            disable_state_update,
            cache_intermediate_states,
            scale,
            use_qk_l2norm,
            tile_v,
            vec_size,
        )
        cache = _get_compiled_mtp_kernel_original(*cache_key_orig)

        if "cu_seqlens" not in cache or cache["cu_seqlens"].device != q.device:
            cache["cu_seqlens"] = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        cu_seqlens = cache["cu_seqlens"]

        if "compiled" not in cache:
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
            intermediate_states_tensor = from_dlpack(
                intermediate_states, assumed_align=16
            )
            A_log_tensor = from_dlpack(A_log, assumed_align=16)
            a_tensor = from_dlpack(a, assumed_align=16)
            dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
            q_tensor = from_dlpack(q, assumed_align=16)
            k_tensor = from_dlpack(k, assumed_align=16)
            v_tensor = from_dlpack(v, assumed_align=16)
            b_tensor = from_dlpack(b, assumed_align=16)
            o_tensor = from_dlpack(output, assumed_align=16)
            h0_indices_tensor = from_dlpack(initial_state_indices, assumed_align=16)
            cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

            compiled = cute.compile(
                run_gdn_verify_kernel_mtp_original,
                h0_source_tensor,
                intermediate_states_tensor,
                A_log_tensor,
                a_tensor,
                dt_bias_tensor,
                q_tensor,
                k_tensor,
                v_tensor,
                b_tensor,
                o_tensor,
                h0_indices_tensor,
                cu_seqlens_tensor,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                scale=scale,
                HV=HV,
                B=B,
                T=T,
                H=H,
                K=K,
                V=V,
                tile_v=tile_v,
                vec_size=vec_size,
                use_initial_state=True,
                use_qk_l2norm=use_qk_l2norm,
                is_varlen=False,
                disable_state_update=disable_state_update,
                cache_intermediate_states=cache_intermediate_states,
                stream=stream,
                options="--enable-tvm-ffi",
            )
            cache["compiled"] = compiled
        else:
            compiled = cache["compiled"]

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(
            h0_source,
            intermediate_states,
            A_log,
            a,
            dt_bias,
            q,
            k,
            v,
            b,
            output,
            initial_state_indices,
            cu_seqlens,
            stream,
        )
    elif use_smem_kernel:
        # SMEM-resident state kernel — higher occupancy
        cache_key_smem = (
            B,
            T,
            H,
            HV,
            K,
            V,
            pool_size,
            cache_steps,
            disable_state_update,
            cache_intermediate_states,
            scale,
            use_qk_l2norm,
            tile_v,
        )
        cache = _get_compiled_mtp_smem_kernel(*cache_key_smem)

        if "cu_seqlens" not in cache or cache["cu_seqlens"].device != q.device:
            cache["cu_seqlens"] = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        cu_seqlens = cache["cu_seqlens"]

        if "compiled" not in cache:
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
            intermediate_states_tensor = from_dlpack(
                intermediate_states, assumed_align=16
            )
            A_log_tensor = from_dlpack(A_log, assumed_align=16)
            a_tensor = from_dlpack(a, assumed_align=16)
            dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
            q_tensor = from_dlpack(q, assumed_align=16)
            k_tensor = from_dlpack(k, assumed_align=16)
            v_tensor = from_dlpack(v, assumed_align=16)
            b_tensor = from_dlpack(b, assumed_align=16)
            o_tensor = from_dlpack(output, assumed_align=16)
            h0_indices_tensor = from_dlpack(initial_state_indices, assumed_align=16)
            cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

            compiled = cute.compile(
                run_gdn_verify_kernel_mtp_smem,
                h0_source_tensor,
                intermediate_states_tensor,
                A_log_tensor,
                a_tensor,
                dt_bias_tensor,
                q_tensor,
                k_tensor,
                v_tensor,
                b_tensor,
                o_tensor,
                h0_indices_tensor,
                cu_seqlens_tensor,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                scale=scale,
                HV=HV,
                B=B,
                T=T,
                H=H,
                K=K,
                V=V,
                tile_v=tile_v,
                use_initial_state=True,
                use_qk_l2norm=use_qk_l2norm,
                is_varlen=False,
                disable_state_update=disable_state_update,
                cache_intermediate_states=cache_intermediate_states,
                stream=stream,
                options="--enable-tvm-ffi",
            )
            cache["compiled"] = compiled
        else:
            compiled = cache["compiled"]

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(
            h0_source,
            intermediate_states,
            A_log,
            a,
            dt_bias,
            q,
            k,
            v,
            b,
            output,
            initial_state_indices,
            cu_seqlens,
            stream,
        )
    else:
        # Original register-resident state kernel
        ilp_rows = get_ilp_rows(B, T, disable_state_update)
        use_smem_v = get_use_smem_v(B, T, disable_state_update)
        cache_key = (
            B,
            T,
            H,
            HV,
            K,
            V,
            pool_size,
            cache_steps,
            disable_state_update,
            cache_intermediate_states,
            scale,
            use_qk_l2norm,
            tile_v,
            vec_size,
            ilp_rows,
            use_smem_v,
        )
        cache = _get_compiled_mtp_kernel(*cache_key)

        if "cu_seqlens" not in cache or cache["cu_seqlens"].device != q.device:
            cache["cu_seqlens"] = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        cu_seqlens = cache["cu_seqlens"]

        if "compiled" not in cache:
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

            h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
            intermediate_states_tensor = from_dlpack(
                intermediate_states, assumed_align=16
            )
            A_log_tensor = from_dlpack(A_log, assumed_align=16)
            a_tensor = from_dlpack(a, assumed_align=16)
            dt_bias_tensor = from_dlpack(dt_bias, assumed_align=16)
            q_tensor = from_dlpack(q, assumed_align=16)
            k_tensor = from_dlpack(k, assumed_align=16)
            v_tensor = from_dlpack(v, assumed_align=16)
            b_tensor = from_dlpack(b, assumed_align=16)
            o_tensor = from_dlpack(output, assumed_align=16)
            h0_indices_tensor = from_dlpack(initial_state_indices, assumed_align=16)
            cu_seqlens_tensor = from_dlpack(cu_seqlens, assumed_align=16)

            compiled = cute.compile(
                run_gdn_verify_kernel_mtp,
                h0_source_tensor,
                intermediate_states_tensor,
                A_log_tensor,
                a_tensor,
                dt_bias_tensor,
                q_tensor,
                k_tensor,
                v_tensor,
                b_tensor,
                o_tensor,
                h0_indices_tensor,
                cu_seqlens_tensor,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                scale=scale,
                HV=HV,
                B=B,
                T=T,
                H=H,
                K=K,
                V=V,
                tile_v=tile_v,
                vec_size=vec_size,
                use_initial_state=True,
                use_qk_l2norm=use_qk_l2norm,
                is_varlen=False,
                disable_state_update=disable_state_update,
                cache_intermediate_states=cache_intermediate_states,
                ilp_rows=ilp_rows,
                use_smem_v=use_smem_v,
                stream=stream,
                options="--enable-tvm-ffi",
            )
            cache["compiled"] = compiled
        else:
            compiled = cache["compiled"]

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(
            h0_source,
            intermediate_states,
            A_log,
            a,
            dt_bias,
            q,
            k,
            v,
            b,
            output,
            initial_state_indices,
            cu_seqlens,
            stream,
        )

    # Copy state back if needed (no sync needed - PyTorch handles stream ordering)
    # Only copy if state update is enabled AND initial_state was not contiguous
    # (if contiguous, reshape returns a view and kernel updated state in-place)
    if not disable_state_update and not initial_state.is_contiguous():
        initial_state.copy_(h0_source.reshape(pool_size, HV, V, K))

    # Convert output to target dtype if needed
    if output.dtype != target_dtype:
        output = output.to(target_dtype)

    return output, initial_state
