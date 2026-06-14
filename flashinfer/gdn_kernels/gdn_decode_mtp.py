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
Gated Delta Rule MTP (Multiple Token Processing) Kernels.

FP32 hidden state, V-major (K-last) layout [pool_size * HV, V, K], T > 1.

This module contains the CuTe DSL kernel implementations for multi-token
processing in the Gated Delta Rule decode phase. These kernels process
T > 1 tokens sequentially, typically used for speculative decoding verification.

Kernel variants:
- gdn_verify_kernel_mtp: Warp-specialized kernel with ILP rows, SMEM precompute,
  and optional SMEM v caching. Used for BS >= 3.
- gdn_verify_kernel_mtp_inline: Inline/deferred-L2-norm kernel with register-resident
  q/k/g/beta (no SMEM precompute overhead). Used for BS <= 2.

v15 dispatch: inline for BS<=2, warp-specialized for BS>=3.
Key v15 optimization: BS=8-16 use tile_v=32 with ilp=4 for ALL T values,
eliminating the ilp=1 fallback at T>=4 that caused a ~33% per-step slowdown.

Each kernel has a corresponding launcher (run_*), compilation cache (_get_compiled_*),
and is dispatched via the public run_mtp_decode() function.
"""

import functools

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

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


def get_mtp_config(
    batch_size: int,
    seq_len: int,
    num_v_heads: int = 64,
    v_dim: int = 128,
    disable_state_update: bool = False,
) -> tuple:
    """Unified MTP config selection based on CTA work units.

    Returns (tile_v, vec_size, ilp_rows, use_smem_v) for the MTP kernel.

    Uses work_units = batch_size * num_v_heads (the tile_v-independent CTA factor)
    as the decision variable so the selection is model-independent. Thresholds
    derived from Qwen3.5 (HV=64) grid search on B200.

    | work_units | Qwen3.5 BS equiv | tile_v | ilp | smem_v |
    |-----------:|------------------:|-------:|----:|:------:|
    | ≤64        | BS≤1              | 8      | 2   | False  |
    | ≤128       | BS≤2              | 16     | 4   | False  |
    | ≤448, T≤2  | BS≤7              | 16     | 2   | False  |
    | ≤448, T≥3  | BS≤7              | 32     | 4   | False  |
    | ≤1024      | BS≤16             | 32     | 4   | False  |
    | >1024      | BS≥17             | 64     | 4/8 | True*  |

    *smem_v=False if state_update ON + T≤2; ilp=8 if tile_v≥64 + state_update ON + T≤2.
    """
    work_units = batch_size * num_v_heads
    vec_size = 4  # Always 4 (full warp shuffle)

    if work_units <= 64:
        tile_v, ilp_rows, use_smem_v = 8, 2, False
    elif work_units <= 128:
        tile_v, ilp_rows, use_smem_v = 16, 4, False
    elif work_units <= 448:
        if seq_len <= 2:
            tile_v, ilp_rows, use_smem_v = 16, 2, False
        else:
            tile_v, ilp_rows, use_smem_v = 32, 4, False
    elif work_units <= 1024:
        tile_v, ilp_rows, use_smem_v = 32, 4, False
    else:
        # Large batches: tile_v depends on T
        if seq_len <= 2:
            # Low T: tile_v=32 gives 2x CTAs, better DRAM page utilization
            # smem_v=False is fine since V reuse is only 2x
            tile_v = 32
            use_smem_v = False
            ilp_rows = 4
            # State update ON + low T: ilp=8 helps
            if not disable_state_update:
                ilp_rows = 8
        else:
            # High T: tile_v=64 with smem_v=True — V values cached in SMEM
            # and reused across T timesteps, amortizing the GMEM load
            tile_v = 64
            use_smem_v = True
            ilp_rows = 4

    # Clamp tile_v to v_dim (e.g. v_dim=64 models shouldn't use tile_v=128)
    tile_v = min(tile_v, v_dim)

    return tile_v, vec_size, ilp_rows, use_smem_v


def get_vec_size_mtp(batch_size: int, seq_len: int = 1) -> int:
    """Select vec_size for MTP kernel.

    Always use vec_size=4 (32 threads per group = full warp, 4 groups per block).
    Full warp shuffle is more efficient and achieves >= 1.0x speedup vs Triton.
    """
    return 4


def get_tile_v_mtp(
    batch_size: int,
    seq_len: int = 1,
    *,
    num_v_heads: int = 64,
    v_dim: int = 128,
) -> int:
    """Select optimal TILE_V for MTP kernel. Delegates to get_mtp_config()."""
    tile_v, _, _, _ = get_mtp_config(batch_size, seq_len, num_v_heads, v_dim)
    return tile_v


def get_ilp_rows(
    batch_size: int,
    seq_len: int,
    disable_state_update: bool = False,
    *,
    num_v_heads: int = 64,
    v_dim: int = 128,
) -> int:
    """Select number of ILP rows for the MTP kernel. Delegates to get_mtp_config()."""
    _, _, ilp_rows, _ = get_mtp_config(
        batch_size, seq_len, num_v_heads, v_dim, disable_state_update
    )
    return ilp_rows


def get_use_smem_v(
    batch_size: int,
    seq_len: int,
    disable_state_update: bool = False,
    *,
    num_v_heads: int = 64,
    v_dim: int = 128,
) -> bool:
    """Decide whether to preload v values into SMEM. Delegates to get_mtp_config()."""
    _, _, _, use_smem_v = get_mtp_config(
        batch_size, seq_len, num_v_heads, v_dim, disable_state_update
    )
    return use_smem_v


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
    use_packed_fma: cutlass.Constexpr[bool],
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

    # Thread grouping: vec_size=4, so 32 threads/group (full warp), 4 groups/block
    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group  # 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp  # 4

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
    # 2D register tensor for 8 V-rows processed in parallel (ILP)
    # Row-major layout: stride (vec_size, 1) so each row is contiguous
    r_h = cute.make_rmem_tensor(
        cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32
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

        # Pre-compute these before Phase 1 (needed for h-state prefetch)
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups
        flat_state_idx = cache_idx * HV + i_hv

        # === WARP SPECIALIZATION: Phase 1 ===
        # Warp 0: compute q/k/g/beta for all T timesteps, write to SMEM
        # Warps 1-3: prefetch h-state for first ILP set during Phase 1 window
        if warp_idx == 0:
            # Warp 0: Phase 1 — compute and broadcast q, k, g, beta via SMEM
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

                    # Full warp reduction (threads_per_group=32, vec_size=4)
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

                # Warp 0 writes to SMEM — with vec_size=8, only first 16 threads
                # cover full K (16*8=128). Threads 16-31 write redundantly (same values).
                for i in cutlass.range_constexpr(vec_size):
                    sQ[(i_t, k_start + i)] = r_q[i]
                    sK[(i_t, k_start + i)] = r_k[i]

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
                    use_softplus * softplus_val
                    + (cutlass.Float32(1.0) - use_softplus) * x
                )

                r_g_value = -cute.exp(r_A_log, fastmath=True) * softplus_x
                r_beta = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-r_b, fastmath=True)
                )
                r_g = cute.exp(r_g_value, fastmath=True)

                # All threads in warp 0 write same warp-uniform values
                sG[i_t] = r_g
                sBeta[i_t] = r_beta

                if cutlass.const_expr(use_smem_v):
                    v_tile_start = i_v * tile_v
                    if tidx < tile_v:
                        v_global_idx = v_tile_start + tidx
                        if v_global_idx < V:
                            sVdata[(i_t, tidx)] = cutlass.Float32(
                                v[i_n, i_t, i_hv, v_global_idx]
                            )
        else:
            # Warps 1-3: Prefetch h-state for first ILP set during Phase 1 window
            # This overlaps h-state DRAM latency with warp 0's Phase 1 compute
            v_base_prefetch = i_v * tile_v + group_idx * rows_per_group
            if cutlass.const_expr(ilp_rows >= 4):
                # Prefetch first 4 h-state rows into r_h[0:3]
                v_pf_d = v_base_prefetch + 3
                if v_pf_d < V:
                    pf_a = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch, lane_in_group),
                    )
                    pf_b = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch + 1, lane_in_group),
                    )
                    pf_c = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch + 2, lane_in_group),
                    )
                    pf_d = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch + 3, lane_in_group),
                    )
                    cute.autovec_copy(pf_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(pf_b, cute.slice_(r_h, (1, None)))
                    cute.autovec_copy(pf_c, cute.slice_(r_h, (2, None)))
                    cute.autovec_copy(pf_d, cute.slice_(r_h, (3, None)))
            elif cutlass.const_expr(ilp_rows == 2):
                # Prefetch 2 h-state rows
                v_pf_b = v_base_prefetch + 1
                if v_pf_b < V:
                    pf_a = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch, lane_in_group),
                    )
                    pf_b = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch + 1, lane_in_group),
                    )
                    cute.autovec_copy(pf_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(pf_b, cute.slice_(r_h, (1, None)))
            else:
                # Prefetch 1 h-state row
                if v_base_prefetch < V:
                    pf_a = cute.local_tile(
                        h0_source,
                        (1, 1, vec_size),
                        (flat_state_idx, v_base_prefetch, lane_in_group),
                    )
                    cute.autovec_copy(pf_a, cute.slice_(r_h, (0, None)))

            # Cooperatively preload v values if use_smem_v (warps 1-3 help too)
            if cutlass.const_expr(use_smem_v):
                for i_t in cutlass.range_constexpr(T):
                    v_tile_start = i_v * tile_v
                    if tidx < tile_v:
                        v_global_idx = v_tile_start + tidx
                        if v_global_idx < V:
                            sVdata[(i_t, tidx)] = cutlass.Float32(
                                v[i_n, i_t, i_hv, v_global_idx]
                            )

        cute.arch.barrier()

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
                    cute.autovec_copy(ht0, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(ht1, cute.slice_(r_h, (1, None)))
                    cute.autovec_copy(ht2, cute.slice_(r_h, (2, None)))
                    cute.autovec_copy(ht3, cute.slice_(r_h, (3, None)))
                    cute.autovec_copy(ht4, cute.slice_(r_h, (4, None)))
                    cute.autovec_copy(ht5, cute.slice_(r_h, (5, None)))
                    cute.autovec_copy(ht6, cute.slice_(r_h, (6, None)))
                    cute.autovec_copy(ht7, cute.slice_(r_h, (7, None)))

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
                            r_h[0, i] = r_h[0, i] * r_g
                            r_h[1, i] = r_h[1, i] * r_g
                            r_h[2, i] = r_h[2, i] * r_g
                            r_h[3, i] = r_h[3, i] * r_g
                            r_h[4, i] = r_h[4, i] * r_g
                            r_h[5, i] = r_h[5, i] * r_g
                            r_h[6, i] = r_h[6, i] * r_g
                            r_h[7, i] = r_h[7, i] * r_g

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
                            s0 += r_h[0, i] * r_k[i]
                            s1 += r_h[1, i] * r_k[i]
                            s2 += r_h[2, i] * r_k[i]
                            s3 += r_h[3, i] * r_k[i]
                            s4 += r_h[4, i] * r_k[i]
                            s5 += r_h[5, i] * r_k[i]
                            s6 += r_h[6, i] * r_k[i]
                            s7 += r_h[7, i] * r_k[i]

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
                            r_h[0, i] += r_k[i] * vn0
                            r_h[1, i] += r_k[i] * vn1
                            r_h[2, i] += r_k[i] * vn2
                            r_h[3, i] += r_k[i] * vn3
                            r_h[4, i] += r_k[i] * vn4
                            r_h[5, i] += r_k[i] * vn5
                            r_h[6, i] += r_k[i] * vn6
                            r_h[7, i] += r_k[i] * vn7

                        # Cache intermediate state if needed
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            it0 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v0, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (0, None)), it0)
                            it1 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v1, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (1, None)), it1)
                            it2 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v2, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (2, None)), it2)
                            it3 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v3, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (3, None)), it3)
                            it4 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v4, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (4, None)), it4)
                            it5 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v5, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (5, None)), it5)
                            it6 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v6, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (6, None)), it6)
                            it7 = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v7, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (7, None)), it7)

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
                            o0 += r_h[0, i] * r_q[i]
                            o1 += r_h[1, i] * r_q[i]
                            o2 += r_h[2, i] * r_q[i]
                            o3 += r_h[3, i] * r_q[i]
                            o4 += r_h[4, i] * r_q[i]
                            o5 += r_h[5, i] * r_q[i]
                            o6 += r_h[6, i] * r_q[i]
                            o7 += r_h[7, i] * r_q[i]

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
                        cute.autovec_copy(cute.slice_(r_h, (0, None)), ht_o0)
                        ht_o1 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v1, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (1, None)), ht_o1)
                        ht_o2 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v2, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (2, None)), ht_o2)
                        ht_o3 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v3, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (3, None)), ht_o3)
                        ht_o4 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v4, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (4, None)), ht_o4)
                        ht_o5 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v5, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (5, None)), ht_o5)
                        ht_o6 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v6, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (6, None)), ht_o6)
                        ht_o7 = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v7, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (7, None)), ht_o7)
        elif cutlass.const_expr(ilp_rows == 4):
            # === 4-ROW ILP PATH: Process 4 V-rows simultaneously ===
            quarter_rows: cutlass.Constexpr[int] = rows_per_group // 4

            for row_quad in cutlass.range_constexpr(quarter_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_quad * 4
                v_idx_b = v_idx_a + 1
                v_idx_c = v_idx_a + 2
                v_idx_d = v_idx_a + 3

                if v_idx_d < V:
                    # Load h for 4 V-rows.
                    # T>2: Warp 0 loads all quads, warps 1-3 skip first (prefetched in Phase 1)
                    if cutlass.const_expr(T > 6):
                        if warp_idx == 0 or row_quad > 0:
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
                            cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                            cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))
                            cute.autovec_copy(h_tile_c, cute.slice_(r_h, (2, None)))
                            cute.autovec_copy(h_tile_d, cute.slice_(r_h, (3, None)))

                    # T<=2: Only warp 0 first quad loads; software pipeline handles rest
                    if cutlass.const_expr(T <= 6):
                        if warp_idx == 0 and row_quad == 0:
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
                            cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                            cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))
                            cute.autovec_copy(h_tile_c, cute.slice_(r_h, (2, None)))
                            cute.autovec_copy(h_tile_d, cute.slice_(r_h, (3, None)))

                        # Software pipeline: issue LDGs for NEXT row_quad into r_h[4:7]
                        if row_quad < quarter_rows - 1:
                            next_v_a = v_idx_a + 4
                            if next_v_a + 3 < V:
                                h_pf_a = cute.local_tile(
                                    h0_source,
                                    (1, 1, vec_size),
                                    (flat_state_idx, next_v_a, lane_in_group),
                                )
                                h_pf_b = cute.local_tile(
                                    h0_source,
                                    (1, 1, vec_size),
                                    (flat_state_idx, next_v_a + 1, lane_in_group),
                                )
                                h_pf_c = cute.local_tile(
                                    h0_source,
                                    (1, 1, vec_size),
                                    (flat_state_idx, next_v_a + 2, lane_in_group),
                                )
                                h_pf_d = cute.local_tile(
                                    h0_source,
                                    (1, 1, vec_size),
                                    (flat_state_idx, next_v_a + 3, lane_in_group),
                                )
                                cute.autovec_copy(h_pf_a, cute.slice_(r_h, (4, None)))
                                cute.autovec_copy(h_pf_b, cute.slice_(r_h, (5, None)))
                                cute.autovec_copy(h_pf_c, cute.slice_(r_h, (6, None)))
                                cute.autovec_copy(h_pf_d, cute.slice_(r_h, (7, None)))

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

                        # Steps 1+2 FUSED: Decay + h@k using fma_packed_f32x2
                        # Process 2 elements at a time; Blackwell packs 2 FMA in 1 instruction
                        sum_hk_a = cutlass.Float32(0.0)
                        sum_hk_a2 = cutlass.Float32(0.0)
                        sum_hk_b = cutlass.Float32(0.0)
                        sum_hk_b2 = cutlass.Float32(0.0)
                        sum_hk_c = cutlass.Float32(0.0)
                        sum_hk_c2 = cutlass.Float32(0.0)
                        sum_hk_d = cutlass.Float32(0.0)
                        sum_hk_d2 = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(0, vec_size, 2):
                            r_h[0, i] = r_h[0, i] * r_g
                            r_h[0, i + 1] = r_h[0, i + 1] * r_g
                            r_h[1, i] = r_h[1, i] * r_g
                            r_h[1, i + 1] = r_h[1, i + 1] * r_g
                            r_h[2, i] = r_h[2, i] * r_g
                            r_h[2, i + 1] = r_h[2, i + 1] * r_g
                            r_h[3, i] = r_h[3, i] * r_g
                            r_h[3, i + 1] = r_h[3, i + 1] * r_g
                            if cutlass.const_expr(use_packed_fma):
                                sum_hk_a, sum_hk_a2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[0, i], r_h[0, i + 1]),
                                    src_b=(r_k[i], r_k[i + 1]),
                                    src_c=(sum_hk_a, sum_hk_a2),
                                )
                                sum_hk_b, sum_hk_b2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[1, i], r_h[1, i + 1]),
                                    src_b=(r_k[i], r_k[i + 1]),
                                    src_c=(sum_hk_b, sum_hk_b2),
                                )
                                sum_hk_c, sum_hk_c2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[2, i], r_h[2, i + 1]),
                                    src_b=(r_k[i], r_k[i + 1]),
                                    src_c=(sum_hk_c, sum_hk_c2),
                                )
                                sum_hk_d, sum_hk_d2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[3, i], r_h[3, i + 1]),
                                    src_b=(r_k[i], r_k[i + 1]),
                                    src_c=(sum_hk_d, sum_hk_d2),
                                )
                            else:
                                sum_hk_a, sum_hk_a2 = fma_pair(
                                    r_h[0, i],
                                    r_h[0, i + 1],
                                    r_k[i],
                                    r_k[i + 1],
                                    sum_hk_a,
                                    sum_hk_a2,
                                )
                                sum_hk_b, sum_hk_b2 = fma_pair(
                                    r_h[1, i],
                                    r_h[1, i + 1],
                                    r_k[i],
                                    r_k[i + 1],
                                    sum_hk_b,
                                    sum_hk_b2,
                                )
                                sum_hk_c, sum_hk_c2 = fma_pair(
                                    r_h[2, i],
                                    r_h[2, i + 1],
                                    r_k[i],
                                    r_k[i + 1],
                                    sum_hk_c,
                                    sum_hk_c2,
                                )
                                sum_hk_d, sum_hk_d2 = fma_pair(
                                    r_h[3, i],
                                    r_h[3, i + 1],
                                    r_k[i],
                                    r_k[i + 1],
                                    sum_hk_d,
                                    sum_hk_d2,
                                )
                        sum_hk_a = sum_hk_a + sum_hk_a2
                        sum_hk_b = sum_hk_b + sum_hk_b2
                        sum_hk_c = sum_hk_c + sum_hk_c2
                        sum_hk_d = sum_hk_d + sum_hk_d2

                        # Full warp reduction for ALL 4 h@k dot products
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

                        # Steps 4+5 FUSED: h-update + h@q using fma_packed_f32x2
                        sum_hq_a = cutlass.Float32(0.0)
                        sum_hq_a2 = cutlass.Float32(0.0)
                        sum_hq_b = cutlass.Float32(0.0)
                        sum_hq_b2 = cutlass.Float32(0.0)
                        sum_hq_c = cutlass.Float32(0.0)
                        sum_hq_c2 = cutlass.Float32(0.0)
                        sum_hq_d = cutlass.Float32(0.0)
                        sum_hq_d2 = cutlass.Float32(0.0)
                        for i in cutlass.range_constexpr(0, vec_size, 2):
                            if cutlass.const_expr(use_packed_fma):
                                r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                                    src_a=(r_k[i], r_k[i + 1]),
                                    src_b=(v_new_a, v_new_a),
                                    src_c=(r_h[0, i], r_h[0, i + 1]),
                                )
                                r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                                    src_a=(r_k[i], r_k[i + 1]),
                                    src_b=(v_new_b, v_new_b),
                                    src_c=(r_h[1, i], r_h[1, i + 1]),
                                )
                                r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                                    src_a=(r_k[i], r_k[i + 1]),
                                    src_b=(v_new_c, v_new_c),
                                    src_c=(r_h[2, i], r_h[2, i + 1]),
                                )
                                r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                                    src_a=(r_k[i], r_k[i + 1]),
                                    src_b=(v_new_d, v_new_d),
                                    src_c=(r_h[3, i], r_h[3, i + 1]),
                                )
                                sum_hq_a, sum_hq_a2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[0, i], r_h[0, i + 1]),
                                    src_b=(r_q[i], r_q[i + 1]),
                                    src_c=(sum_hq_a, sum_hq_a2),
                                )
                                sum_hq_b, sum_hq_b2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[1, i], r_h[1, i + 1]),
                                    src_b=(r_q[i], r_q[i + 1]),
                                    src_c=(sum_hq_b, sum_hq_b2),
                                )
                                sum_hq_c, sum_hq_c2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[2, i], r_h[2, i + 1]),
                                    src_b=(r_q[i], r_q[i + 1]),
                                    src_c=(sum_hq_c, sum_hq_c2),
                                )
                                sum_hq_d, sum_hq_d2 = cute.arch.fma_packed_f32x2(
                                    src_a=(r_h[3, i], r_h[3, i + 1]),
                                    src_b=(r_q[i], r_q[i + 1]),
                                    src_c=(sum_hq_d, sum_hq_d2),
                                )
                            else:
                                r_h[0, i], r_h[0, i + 1] = fma_pair(
                                    r_k[i],
                                    r_k[i + 1],
                                    v_new_a,
                                    v_new_a,
                                    r_h[0, i],
                                    r_h[0, i + 1],
                                )
                                r_h[1, i], r_h[1, i + 1] = fma_pair(
                                    r_k[i],
                                    r_k[i + 1],
                                    v_new_b,
                                    v_new_b,
                                    r_h[1, i],
                                    r_h[1, i + 1],
                                )
                                r_h[2, i], r_h[2, i + 1] = fma_pair(
                                    r_k[i],
                                    r_k[i + 1],
                                    v_new_c,
                                    v_new_c,
                                    r_h[2, i],
                                    r_h[2, i + 1],
                                )
                                r_h[3, i], r_h[3, i + 1] = fma_pair(
                                    r_k[i],
                                    r_k[i + 1],
                                    v_new_d,
                                    v_new_d,
                                    r_h[3, i],
                                    r_h[3, i + 1],
                                )
                                sum_hq_a, sum_hq_a2 = fma_pair(
                                    r_h[0, i],
                                    r_h[0, i + 1],
                                    r_q[i],
                                    r_q[i + 1],
                                    sum_hq_a,
                                    sum_hq_a2,
                                )
                                sum_hq_b, sum_hq_b2 = fma_pair(
                                    r_h[1, i],
                                    r_h[1, i + 1],
                                    r_q[i],
                                    r_q[i + 1],
                                    sum_hq_b,
                                    sum_hq_b2,
                                )
                                sum_hq_c, sum_hq_c2 = fma_pair(
                                    r_h[2, i],
                                    r_h[2, i + 1],
                                    r_q[i],
                                    r_q[i + 1],
                                    sum_hq_c,
                                    sum_hq_c2,
                                )
                                sum_hq_d, sum_hq_d2 = fma_pair(
                                    r_h[3, i],
                                    r_h[3, i + 1],
                                    r_q[i],
                                    r_q[i + 1],
                                    sum_hq_d,
                                    sum_hq_d2,
                                )
                        sum_hq_a = sum_hq_a + sum_hq_a2
                        sum_hq_b = sum_hq_b + sum_hq_b2
                        sum_hq_c = sum_hq_c + sum_hq_c2
                        sum_hq_d = sum_hq_d + sum_hq_d2

                        # Full warp reduction for ALL 4 h@q dot products
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

                        # Write output for ALL 4 rows
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

                        # Cache intermediate state LAST in timestep (fire-and-forget stores
                        # overlap with next timestep's compute)
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_tile_a = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_a, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (0, None)), inter_tile_a)
                            inter_tile_b = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_b, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (1, None)), inter_tile_b)
                            inter_tile_c = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_c, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (2, None)), inter_tile_c)
                            inter_tile_d = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_d, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (3, None)), inter_tile_d)

                    # Write final state back for ALL 4 rows (if not disabled)
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out_a = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_a, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (0, None)), h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (1, None)), h_tile_out_b)
                        h_tile_out_c = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_c, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (2, None)), h_tile_out_c)
                        h_tile_out_d = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_d, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (3, None)), h_tile_out_d)

                    # Software pipeline: move prefetched h-state to active slots.
                    # Only for T<=2 (matches the prefetch LDG gate above).
                    if cutlass.const_expr(T <= 6):
                        if row_quad < quarter_rows - 1:
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] = r_h[4, i]
                                r_h[1, i] = r_h[5, i]
                                r_h[2, i] = r_h[6, i]
                                r_h[3, i] = r_h[7, i]
        elif cutlass.const_expr(ilp_rows == 2):
            # === 2-ROW ILP PATH: Process 2 V-rows simultaneously ===
            half_rows: cutlass.Constexpr[int] = rows_per_group // 2

            for row_pair in cutlass.range_constexpr(half_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_pair * 2
                v_idx_b = v_idx_a + 1

                if v_idx_b < V:
                    # Load h for 2 V-rows.
                    # T>2: Warp 0 loads all pairs, warps 1-3 skip first (prefetched in Phase 1)
                    if cutlass.const_expr(T > 6):
                        if warp_idx == 0 or row_pair > 0:
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
                            cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                            cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))

                    # T<=2: Only warp 0 first pair loads; software pipeline handles rest
                    if cutlass.const_expr(T <= 6):
                        if warp_idx == 0 and row_pair == 0:
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
                            cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                            cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))

                        # Software pipeline: issue LDGs for NEXT row_pair into r_h[2:3]
                        if row_pair < half_rows - 1:
                            next_v_a = v_idx_a + 2
                            if next_v_a + 1 < V:
                                h_pf_a = cute.local_tile(
                                    h0_source,
                                    (1, 1, vec_size),
                                    (flat_state_idx, next_v_a, lane_in_group),
                                )
                                h_pf_b = cute.local_tile(
                                    h0_source,
                                    (1, 1, vec_size),
                                    (flat_state_idx, next_v_a + 1, lane_in_group),
                                )
                                cute.autovec_copy(h_pf_a, cute.slice_(r_h, (2, None)))
                                cute.autovec_copy(h_pf_b, cute.slice_(r_h, (3, None)))

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
                            r_h[0, i] = r_h[0, i] * r_g
                            r_h[1, i] = r_h[1, i] * r_g

                        # Step 2: Compute dot products for BOTH rows (ILP)
                        sum_hk_a = 0.0
                        sum_hk_b = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hk_a += r_h[0, i] * r_k[i]
                            sum_hk_b += r_h[1, i] * r_k[i]

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
                            r_h[0, i] += r_k[i] * v_new_a
                            r_h[1, i] += r_k[i] * v_new_b

                        # Cache intermediate state if needed
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_tile_a = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_a, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (0, None)), inter_tile_a)
                            inter_tile_b = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_b, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (1, None)), inter_tile_b)

                        # Step 5: Compute output for BOTH rows (ILP)
                        sum_hq_a = 0.0
                        sum_hq_b = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            sum_hq_a += r_h[0, i] * r_q[i]
                            sum_hq_b += r_h[1, i] * r_q[i]

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
                        cute.autovec_copy(cute.slice_(r_h, (0, None)), h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (1, None)), h_tile_out_b)

                    # Software pipeline: move prefetched h-state to active slots.
                    # Only for T<=2 (matches the prefetch LDG gate above).
                    if cutlass.const_expr(T <= 6):
                        if row_pair < half_rows - 1:
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] = r_h[2, i]
                                r_h[1, i] = r_h[3, i]
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
    use_packed_fma: cutlass.Constexpr[bool],
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
        use_packed_fma,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# v10-style inline MTP kernel — no sQ/sK/sG/sBeta SMEM, deferred L2 norm
# Best for BS <= 2 (up to +10pp SOL vs v9 SMEM precompute variant)
@cute.kernel
def gdn_verify_kernel_mtp_inline(
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
    use_packed_fma: cutlass.Constexpr[bool],
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

    # Thread grouping: vec_size=4, so 32 threads/group (full warp), 4 groups/block
    threads_per_group: cutlass.Constexpr[int] = K // vec_size  # 32
    groups_per_warp: cutlass.Constexpr[int] = 32 // threads_per_group  # 1
    num_groups: cutlass.Constexpr[int] = 4 * groups_per_warp  # 4

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

    # v10: No sQ/sK/sG/sBeta — q/k/g/β are inlined into T-loop (deferred L2 norm)
    # Only allocate sVdata (for use_smem_v) and sOutput (for coalesced writeback)
    smem = cutlass.utils.SmemAllocator()
    sVdata = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, tile_v), stride=(tile_v, 1)), 16
    )
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
    # 2D register tensor for 8 V-rows processed in parallel (ILP)
    r_h = cute.make_rmem_tensor(
        cute.make_layout((8, vec_size), stride=(vec_size, 1)), cutlass.Float32
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
        # v10: Mini pre-compute — only v preload for use_smem_v (q/k/g/β inlined)
        if cutlass.const_expr(use_smem_v):
            for i_t in cutlass.range_constexpr(T):
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

        # v10: Pre-compute g/β for ALL timesteps into register arrays (shared across V-rows)
        # This avoids redundant softplus/sigmoid computation in each V-row iteration.
        r_g_arr = cute.make_rmem_tensor(
            cute.make_layout((T,), stride=(1,)), cutlass.Float32
        )
        r_beta_arr = cute.make_rmem_tensor(
            cute.make_layout((T,), stride=(1,)), cutlass.Float32
        )
        for i_t in cutlass.range_constexpr(T):
            r_a_val = cutlass.Float32(a[i_n, i_t, i_hv])
            r_b_val = cutlass.Float32(b[i_n, i_t, i_hv])
            x_val = r_a_val + r_dt_bias
            beta_x_val = softplus_beta * x_val
            exp_beta_x_val = cute.exp(beta_x_val, fastmath=True)
            sp_val = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
                cutlass.Float32(1.0) + exp_beta_x_val, fastmath=True
            )
            use_sp = (
                cutlass.Float32(1.0)
                if beta_x_val <= softplus_threshold
                else cutlass.Float32(0.0)
            )
            sp_x = use_sp * sp_val + (cutlass.Float32(1.0) - use_sp) * x_val
            r_g_value = -cute.exp(r_A_log, fastmath=True) * sp_x
            r_g_arr[i_t] = cute.exp(r_g_value, fastmath=True)
            r_beta_arr[i_t] = cutlass.Float32(1.0) / (
                cutlass.Float32(1.0) + cute.exp(-r_b_val, fastmath=True)
            )

        if cutlass.const_expr(ilp_rows == 4):
            # === 4-ROW ILP PATH (v10: inline pre-compute + deferred L2 norm) ===
            quarter_rows: cutlass.Constexpr[int] = rows_per_group // 4

            for row_quad in cutlass.range_constexpr(quarter_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_quad * 4
                v_idx_b = v_idx_a + 1
                v_idx_c = v_idx_a + 2
                v_idx_d = v_idx_a + 3

                if v_idx_d < V:
                    # Issue h loads FIRST
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
                    cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))
                    cute.autovec_copy(h_tile_c, cute.slice_(r_h, (2, None)))
                    cute.autovec_copy(h_tile_d, cute.slice_(r_h, (3, None)))

                    # Prologue: load q[0], k[0]
                    q_tile = cute.local_tile(
                        q, (1, 1, 1, vec_size), (i_n, 0, i_h, lane_in_group)
                    )
                    k_tile = cute.local_tile(
                        k, (1, 1, 1, vec_size), (i_n, 0, i_h, lane_in_group)
                    )
                    cute.autovec_copy(q_tile, r_q_bf16)
                    cute.autovec_copy(k_tile, r_k_bf16)
                    for i in cutlass.range_constexpr(vec_size):
                        r_q[i] = cutlass.Float32(r_q_bf16[i])
                        r_k[i] = cutlass.Float32(r_k_bf16[i])
                    if cutlass.const_expr(not use_qk_l2norm):
                        for i in cutlass.range_constexpr(vec_size):
                            r_q[i] = r_q[i] * scale

                    # g/β for t=0 from pre-computed register arrays
                    r_g = r_g_arr[0]
                    r_beta = r_beta_arr[0]

                    for i_t in cutlass.range_constexpr(T):
                        # Step 1: Decay all 4 h vectors
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[0, i] = r_h[0, i] * r_g
                            r_h[1, i] = r_h[1, i] * r_g
                            r_h[2, i] = r_h[2, i] * r_g
                            r_h[3, i] = r_h[3, i] * r_g

                        # Step 2: h@k with deferred L2 norm
                        sum_hk_a = 0.0
                        sum_hk_b = 0.0
                        sum_hk_c = 0.0
                        sum_hk_d = 0.0
                        if cutlass.const_expr(use_qk_l2norm):
                            sum_sq_k = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                sum_hk_a += r_h[0, i] * r_k[i]
                                sum_hk_b += r_h[1, i] * r_k[i]
                                sum_hk_c += r_h[2, i] * r_k[i]
                                sum_hk_d += r_h[3, i] * r_k[i]
                                sum_sq_k += r_k[i] * r_k[i]
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
                                sum_sq_k += cute.arch.shuffle_sync_bfly(
                                    sum_sq_k, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            inv_norm_k = cute.rsqrt(sum_sq_k + 1e-6, fastmath=True)
                            sum_hk_a = sum_hk_a * inv_norm_k
                            sum_hk_b = sum_hk_b * inv_norm_k
                            sum_hk_c = sum_hk_c * inv_norm_k
                            sum_hk_d = sum_hk_d * inv_norm_k
                        else:
                            for i in cutlass.range_constexpr(vec_size):
                                sum_hk_a += r_h[0, i] * r_k[i]
                                sum_hk_b += r_h[1, i] * r_k[i]
                                sum_hk_c += r_h[2, i] * r_k[i]
                                sum_hk_d += r_h[3, i] * r_k[i]
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

                        # Step 3: Load v, delta rule
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

                        # Step 4: Rank-1 update
                        if cutlass.const_expr(use_qk_l2norm):
                            ks_a = inv_norm_k * v_new_a
                            ks_b = inv_norm_k * v_new_b
                            ks_c = inv_norm_k * v_new_c
                            ks_d = inv_norm_k * v_new_d
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] += r_k[i] * ks_a
                                r_h[1, i] += r_k[i] * ks_b
                                r_h[2, i] += r_k[i] * ks_c
                                r_h[3, i] += r_k[i] * ks_d
                        else:
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] += r_k[i] * v_new_a
                                r_h[1, i] += r_k[i] * v_new_b
                                r_h[2, i] += r_k[i] * v_new_c
                                r_h[3, i] += r_k[i] * v_new_d

                        # Cache intermediate state if needed
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_tile_a = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_a, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (0, None)), inter_tile_a)
                            inter_tile_b = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_b, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (1, None)), inter_tile_b)
                            inter_tile_c = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_c, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (2, None)), inter_tile_c)
                            inter_tile_d = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_d, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (3, None)), inter_tile_d)

                        # Step 5: h@q with deferred L2 norm
                        sum_hq_a = 0.0
                        sum_hq_b = 0.0
                        sum_hq_c = 0.0
                        sum_hq_d = 0.0
                        if cutlass.const_expr(use_qk_l2norm):
                            sum_sq_q = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                sum_hq_a += r_h[0, i] * r_q[i]
                                sum_hq_b += r_h[1, i] * r_q[i]
                                sum_hq_c += r_h[2, i] * r_q[i]
                                sum_hq_d += r_h[3, i] * r_q[i]
                                sum_sq_q += r_q[i] * r_q[i]
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
                                sum_sq_q += cute.arch.shuffle_sync_bfly(
                                    sum_sq_q, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            inv_norm_q_scaled = (
                                cute.rsqrt(sum_sq_q + 1e-6, fastmath=True) * scale
                            )
                            sum_hq_a = sum_hq_a * inv_norm_q_scaled
                            sum_hq_b = sum_hq_b * inv_norm_q_scaled
                            sum_hq_c = sum_hq_c * inv_norm_q_scaled
                            sum_hq_d = sum_hq_d * inv_norm_q_scaled
                        else:
                            for i in cutlass.range_constexpr(vec_size):
                                sum_hq_a += r_h[0, i] * r_q[i]
                                sum_hq_b += r_h[1, i] * r_q[i]
                                sum_hq_c += r_h[2, i] * r_q[i]
                                sum_hq_d += r_h[3, i] * r_q[i]
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

                        # Write output
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

                        # Prefetch q/k/g/β for next timestep
                        if cutlass.const_expr(i_t + 1 < T):
                            q_tile = cute.local_tile(
                                q,
                                (1, 1, 1, vec_size),
                                (i_n, i_t + 1, i_h, lane_in_group),
                            )
                            k_tile = cute.local_tile(
                                k,
                                (1, 1, 1, vec_size),
                                (i_n, i_t + 1, i_h, lane_in_group),
                            )
                            cute.autovec_copy(q_tile, r_q_bf16)
                            cute.autovec_copy(k_tile, r_k_bf16)
                            for i in cutlass.range_constexpr(vec_size):
                                r_q[i] = cutlass.Float32(r_q_bf16[i])
                                r_k[i] = cutlass.Float32(r_k_bf16[i])
                            if cutlass.const_expr(not use_qk_l2norm):
                                for i in cutlass.range_constexpr(vec_size):
                                    r_q[i] = r_q[i] * scale
                            r_g = r_g_arr[i_t + 1]
                            r_beta = r_beta_arr[i_t + 1]

                    # Write final state back for ALL 4 rows (if not disabled)
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out_a = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_a, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (0, None)), h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (1, None)), h_tile_out_b)
                        h_tile_out_c = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_c, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (2, None)), h_tile_out_c)
                        h_tile_out_d = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_d, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (3, None)), h_tile_out_d)
        elif cutlass.const_expr(ilp_rows == 2):
            # === 2-ROW ILP PATH (v10: batched q/k + pre-computed L2 norms + fused loops) ===
            half_rows: cutlass.Constexpr[int] = rows_per_group // 2

            r_q_all = cute.make_rmem_tensor(
                cute.make_layout((T, vec_size), stride=(vec_size, 1)), cutlass.Float32
            )
            r_k_all = cute.make_rmem_tensor(
                cute.make_layout((T, vec_size), stride=(vec_size, 1)), cutlass.Float32
            )
            inv_nk_arr = cute.make_rmem_tensor(
                cute.make_layout((T,), stride=(1,)), cutlass.Float32
            )
            inv_nq_arr = cute.make_rmem_tensor(
                cute.make_layout((T,), stride=(1,)), cutlass.Float32
            )

            for row_pair in cutlass.range_constexpr(half_rows):
                v_idx_a = i_v * tile_v + group_idx * rows_per_group + row_pair * 2
                v_idx_b = v_idx_a + 1

                if v_idx_b < V:
                    # Issue h loads FIRST
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
                    cute.autovec_copy(h_tile_a, cute.slice_(r_h, (0, None)))
                    cute.autovec_copy(h_tile_b, cute.slice_(r_h, (1, None)))

                    # Batch load ALL q/k + compute ALL L2 norms
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
                            r_q_all[i_t, i] = cutlass.Float32(r_q_bf16[i])
                            r_k_all[i_t, i] = cutlass.Float32(r_k_bf16[i])

                        if cutlass.const_expr(use_qk_l2norm):
                            sum_sq_q = 0.0
                            sum_sq_k = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                sum_sq_q += r_q_all[i_t, i] * r_q_all[i_t, i]
                                sum_sq_k += r_k_all[i_t, i] * r_k_all[i_t, i]
                            for offset in [16, 8, 4, 2, 1]:
                                sum_sq_q += cute.arch.shuffle_sync_bfly(
                                    sum_sq_q, offset=offset, mask=-1, mask_and_clamp=31
                                )
                                sum_sq_k += cute.arch.shuffle_sync_bfly(
                                    sum_sq_k, offset=offset, mask=-1, mask_and_clamp=31
                                )
                            inv_nk_arr[i_t] = cute.rsqrt(sum_sq_k + 1e-6, fastmath=True)
                            inv_nq_arr[i_t] = (
                                cute.rsqrt(sum_sq_q + 1e-6, fastmath=True) * scale
                            )
                        else:
                            for i in cutlass.range_constexpr(vec_size):
                                r_q_all[i_t, i] = r_q_all[i_t, i] * scale

                    # Pre-load all v values
                    r_va_all = cute.make_rmem_tensor(
                        cute.make_layout((T,), stride=(1,)), cutlass.Float32
                    )
                    r_vb_all = cute.make_rmem_tensor(
                        cute.make_layout((T,), stride=(1,)), cutlass.Float32
                    )
                    if cutlass.const_expr(not use_smem_v):
                        for i_t in cutlass.range_constexpr(T):
                            r_va_all[i_t] = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_a])
                            r_vb_all[i_t] = cutlass.Float32(v[i_n, i_t, i_hv, v_idx_b])

                    # T-LOOP: zero q/k GMEM loads, fused loops
                    for i_t in cutlass.range_constexpr(T):
                        r_g = r_g_arr[i_t]
                        r_beta = r_beta_arr[i_t]

                        # Fused decay + h@k
                        sum_hk_a = 0.0
                        sum_hk_b = 0.0
                        for i in cutlass.range_constexpr(vec_size):
                            r_h[0, i] = r_h[0, i] * r_g
                            r_h[1, i] = r_h[1, i] * r_g
                            sum_hk_a += r_h[0, i] * r_k_all[i_t, i]
                            sum_hk_b += r_h[1, i] * r_k_all[i_t, i]
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hk_a += cute.arch.shuffle_sync_bfly(
                                sum_hk_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hk_b += cute.arch.shuffle_sync_bfly(
                                sum_hk_b, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        if cutlass.const_expr(use_qk_l2norm):
                            inv_nk = inv_nk_arr[i_t]
                            sum_hk_a = sum_hk_a * inv_nk
                            sum_hk_b = sum_hk_b * inv_nk

                        # v from pre-loaded registers
                        if cutlass.const_expr(use_smem_v):
                            v_local_a = v_idx_a - i_v * tile_v
                            r_v_a = sVdata[(i_t, v_local_a)]
                            r_v_b = sVdata[(i_t, v_local_a + 1)]
                        else:
                            r_v_a = r_va_all[i_t]
                            r_v_b = r_vb_all[i_t]
                        v_new_a = (r_v_a - sum_hk_a) * r_beta
                        v_new_b = (r_v_b - sum_hk_b) * r_beta

                        # Fused update + h@q
                        if cutlass.const_expr(use_qk_l2norm):
                            ks_a = inv_nk * v_new_a
                            ks_b = inv_nk * v_new_b
                            sum_hq_a = 0.0
                            sum_hq_b = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] += r_k_all[i_t, i] * ks_a
                                r_h[1, i] += r_k_all[i_t, i] * ks_b
                                sum_hq_a += r_h[0, i] * r_q_all[i_t, i]
                                sum_hq_b += r_h[1, i] * r_q_all[i_t, i]
                        else:
                            sum_hq_a = 0.0
                            sum_hq_b = 0.0
                            for i in cutlass.range_constexpr(vec_size):
                                r_h[0, i] += r_k_all[i_t, i] * v_new_a
                                r_h[1, i] += r_k_all[i_t, i] * v_new_b
                                sum_hq_a += r_h[0, i] * r_q_all[i_t, i]
                                sum_hq_b += r_h[1, i] * r_q_all[i_t, i]

                        # Cache intermediate state
                        if cutlass.const_expr(cache_intermediate_states):
                            flat_idx = i_n * T * HV + i_t * HV + i_hv
                            inter_tile_a = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_a, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (0, None)), inter_tile_a)
                            inter_tile_b = cute.local_tile(
                                intermediate_states,
                                (1, 1, vec_size),
                                (flat_idx, v_idx_b, lane_in_group),
                            )
                            cute.autovec_copy(cute.slice_(r_h, (1, None)), inter_tile_b)

                        # h@q reduction
                        for offset in [16, 8, 4, 2, 1]:
                            sum_hq_a += cute.arch.shuffle_sync_bfly(
                                sum_hq_a, offset=offset, mask=-1, mask_and_clamp=31
                            )
                            sum_hq_b += cute.arch.shuffle_sync_bfly(
                                sum_hq_b, offset=offset, mask=-1, mask_and_clamp=31
                            )
                        if cutlass.const_expr(use_qk_l2norm):
                            inv_nq = inv_nq_arr[i_t]
                            sum_hq_a = sum_hq_a * inv_nq
                            sum_hq_b = sum_hq_b * inv_nq

                        # Write output
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

                    # Write final state back
                    if cutlass.const_expr(not disable_state_update):
                        h_tile_out_a = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_a, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (0, None)), h_tile_out_a)
                        h_tile_out_b = cute.local_tile(
                            h0_source,
                            (1, 1, vec_size),
                            (flat_state_idx, v_idx_b, lane_in_group),
                        )
                        cute.autovec_copy(cute.slice_(r_h, (1, None)), h_tile_out_b)

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
def run_gdn_verify_kernel_mtp_inline(
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
    use_packed_fma: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    _, v_dim, _ = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    num_v_tiles = cute.ceil_div(v_dim, tile_v)

    # Grid: (B * HV * num_v_tiles, 1, 1) - parallelize across V dimension
    grid_size = B * HV * num_v_tiles

    # v10: No sQ/sK/sG/sBeta — only sVdata + sOutput
    smem_bytes = (
        4 * T * tile_v  # sVdata (v values for all timesteps)
        + 2 * T * tile_v  # sOutput (output accumulation in BF16)
        + 128  # alignment
    )

    gdn_verify_kernel_mtp_inline(
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
        use_packed_fma,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS_MTP, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


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
    tile_v: int,
    vec_size: int,
    ilp_rows: int = 4,
    use_smem_v: bool = False,
    use_packed_fma: bool = True,
):
    """Cache compiled optimized MTP kernel for given configuration."""
    return {}


@functools.cache
def _get_compiled_mtp_kernel_inline(
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
    ilp_rows: int = 4,
    use_smem_v: bool = False,
    use_packed_fma: bool = True,
):
    """Cache compiled inline MTP kernel (BS <= 2) for given configuration."""
    return {}


def run_mtp_decode(
    h0_source: torch.Tensor,
    intermediate_states: torch.Tensor,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    output: torch.Tensor,
    initial_state_indices: torch.Tensor,
    B: int,
    T: int,
    H: int,
    HV: int,
    K: int,
    V: int,
    pool_size: int,
    cache_steps: int,
    tile_v: int,
    vec_size: int,
    scale: float,
    use_qk_l2norm: bool,
    disable_state_update: bool,
    cache_intermediate_states: bool,
):
    """Execute the appropriate MTP kernel based on batch size.

    Kernel selection:
    - BS <= 2: Inline kernel (register-resident q/k/g/beta, no SMEM precompute)
    - BS >= 3: Warp-specialized kernel (SMEM precompute + h-state prefetch)

    Args:
        h0_source: Reshaped initial state [pool_size * HV, V, K].
        intermediate_states: Reshaped intermediate state cache, or dummy tensor.
        A_log: Log decay parameter [HV].
        a: Input-dependent decay [B, T, HV].
        dt_bias: Decay bias [HV].
        q: Query tensor [B, T, H, K].
        k: Key tensor [B, T, H, K].
        v: Value tensor [B, T, HV, V].
        b: Update gate input [B, T, HV].
        output: Output tensor [B, T, HV, V].
        initial_state_indices: Batch-to-state mapping [B].
        B, T, H, HV, K, V: Dimension sizes.
        pool_size: Number of states in the pool.
        cache_steps: Number of intermediate state cache slots.
        tile_v: Tile size for V dimension.
        vec_size: Vector size for K dimension loads.
        scale: Query scaling factor.
        use_qk_l2norm: Whether to apply L2 normalization to q and k.
        disable_state_update: If True, do not write back updated state.
        cache_intermediate_states: If True, cache intermediate states.
    """
    # Dispatch between inline kernel and warp-specialized kernel based on CTA work units
    _, _, ilp_rows, use_smem_v = get_mtp_config(B, T, HV, V, disable_state_update)
    use_inline_kernel = (B * HV) <= 128
    major, _ = torch.cuda.get_device_capability(q.device)
    use_packed_fma = major >= 10  # SM100+ (Blackwell) supports packed F32x2

    if use_inline_kernel:
        inline_cache_key = (
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
            use_packed_fma,
        )
        cache = _get_compiled_mtp_kernel_inline(*inline_cache_key)
    else:
        warp_cache_key = (
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
            use_packed_fma,
        )
        cache = _get_compiled_mtp_kernel(*warp_cache_key)

    if "cu_seqlens" not in cache or cache["cu_seqlens"].device != q.device:
        cache["cu_seqlens"] = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    cu_seqlens = cache["cu_seqlens"]

    if "compiled" not in cache:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        h0_source_tensor = from_dlpack(h0_source, assumed_align=16)
        intermediate_states_tensor = from_dlpack(intermediate_states, assumed_align=16)
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

        if use_inline_kernel:
            compiled = cute.compile(
                run_gdn_verify_kernel_mtp_inline,
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
                use_packed_fma=use_packed_fma,
                stream=stream,
                options="--enable-tvm-ffi --generate-line-info",
            )
        else:
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
                use_packed_fma=use_packed_fma,
                stream=stream,
                options="--enable-tvm-ffi --generate-line-info",
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
