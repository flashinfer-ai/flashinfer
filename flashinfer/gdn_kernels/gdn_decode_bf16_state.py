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

CuTe DSL kernels for GDN decode with BF16 hidden state storage. Pool mode
only (each batch element reads/writes its slot in a shared
``[pool_size, HV, V, K]`` state pool, indexed by ``initial_state_indices``).
Split-pool writes (``output_state_indices != initial_state_indices``,
PR #2905) are supported natively by both kernels. ``K = V = 128`` is
required.

Public API:
- ``gated_delta_rule()``: T=1 single-token decode with BF16 state.
- ``gated_delta_rule_mtp()``: multi-token prediction (T>=1) with BF16 state.

Both entries dispatch to one of:
- ``gdn_wide_vec_kernel`` — the fast path (LDG.E.128 / STG.E.128). Covers
  T=1 with ``B*HV >= 512`` and T>=2 with ``B*HV >= 128``, single-pool or
  split-pool.
- ``gdn_decode_bf16state_mtp_ilp4_kernel`` (ILP=4) — higher-occupancy
  fallback for the low-throughput tail (B=1 at HV=64; T=1 small batch
  with ``tile_v < 64``). Also single-pool or split-pool.
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass.cute.runtime import from_dlpack

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


MTP_NUM_THREADS = 128
MTP_VEC_SIZE = 4  # 32 threads per group x 4 = 128 K elements

# ===== Wide-vec layout constants =====
# LDG.128 over 16-thread subgroups, 8 subgroups per CTA. tile_v is passed as a
# constexpr at compile time; the kernel decodes (i_n, i_hv, i_v) from the
# linear block_idx.
LANES_PER_ROW = 16  # 16 threads cooperate on one V-row's K=128 BF16
ELEMS_PER_LANE = 8  # 8 BF16 = LDG.128
NUM_WARPS = 4
NUM_THREADS = NUM_WARPS * 32  # 128
NUM_GROUPS = NUM_THREADS // LANES_PER_ROW  # 8 groups of 16 threads
ILP_ROWS = 4  # 4 V-rows held in regs per thread per iter


# ==============================================================================
# KERNEL: MTP (ILP=4) — higher-occupancy variant for small `work_units = B*HV`
# ==============================================================================
# Processes 4 V-rows per group iteration (vs the original ILP=8 design).
# ILP=4 uses ~48 regs/thread → ~62% occupancy, which covers the T=2 inline
# g/beta recompute stall and the small-batch latency tail. Dispatched when
# wide_vec gates out — i.e. work_units <= 128 (B=1 at HV=64) or T=1 small
# batch with tile_v < 64.
#
# Supports split-pool writes via ``h0_out_indices``: when the dispatcher
# passes a separate write-indices tensor (output_state_indices !=
# initial_state_indices), the read uses h0_indices and the final-state
# writeback targets h0_out_indices. Single-pool callers reuse the same
# indices tensor for both, which costs nothing extra (cute.local_tile is
# metadata-only and the writeback hits the same slot).

MTP_ILP4_ROWS = 4


@cute.kernel
def gdn_decode_bf16state_mtp_ilp4_kernel(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] as BF16
    intermediate_states: cute.Tensor,  # [B * T * HV, V, K] as BF16 (or dummy)
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
    h0_indices: cute.Tensor,  # [B] - state pool slots to READ from
    h0_out_indices: cute.Tensor,  # [B] - state pool slots to WRITE final H to
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
    same_pool: cutlass.Constexpr[bool],
):
    """MTP kernel (ILP=4) for BF16 state — higher occupancy at small batch.

    Read uses h0_indices, final-state writeback uses h0_out_indices.
    For single-pool callers, the dispatcher passes the same tensor for both
    AND sets ``same_pool=True``; the kernel then aliases write-side
    addressing to the read side at compile time, eliding the extra LDG +
    IMAD + local_tile instructions in SASS.
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    threads_per_group: cutlass.Constexpr[int] = 32  # noqa: F841
    num_groups: cutlass.Constexpr[int] = 4
    group_idx = warp_idx
    lane_in_group = lane_id

    batch_idx, _, _ = cute.arch.block_idx()

    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]
    if cutlass.const_expr(same_pool):
        # Single-pool: alias write to read; nvcc DCEs the write-side LDG /
        # IMAD / local_tile entirely in this compile path.
        write_cache_idx = cache_idx
    else:
        write_cache_idx = h0_out_indices[i_n]
        if write_cache_idx < 0:
            write_cache_idx = cutlass.Int32(0)

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])

    if cutlass.const_expr(T > 1):
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
        )
        sK = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
        )
        sGB = smem.allocate_tensor(
            cutlass.Float32, cute.make_layout((T, 2), stride=(2, 1)), 16
        )

    ILP4: cutlass.Constexpr[int] = 4
    r_q = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_k = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.Float32
    )
    r_h = cute.make_rmem_tensor(
        cute.make_layout((ILP4, vec_size), stride=(vec_size, 1)),
        cutlass.Float32,
    )
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4_0 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4_1 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4_2 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_hb4_3 = cute.make_rmem_tensor(
        cute.make_layout((vec_size,), stride=(1,)), cutlass.BFloat16
    )
    r_o4_bf16 = cute.make_rmem_tensor(
        cute.make_layout((ILP4,), stride=(1,)), cutlass.BFloat16
    )
    r_v4_bf16 = cute.make_rmem_tensor(
        cute.make_layout((ILP4,), stride=(1,)), cutlass.BFloat16
    )

    if cache_idx < 0:
        cache_idx = cutlass.Int32(0)

    if cache_idx >= 0:
        k_start = lane_in_group * vec_size

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

                    for i in cutlass.range_constexpr(vec_size):
                        sQ[(i_t_pre, k_start + i)] = r_q[i]
                        sK[(i_t_pre, k_start + i)] = r_k[i]

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

                cute.arch.barrier()

        flat_state_idx = cache_idx * HV + i_hv
        if cutlass.const_expr(same_pool):
            flat_write_state_idx = flat_state_idx
        else:
            flat_write_state_idx = write_cache_idx * HV + i_hv
        rows_per_group: cutlass.Constexpr[int] = tile_v // num_groups

        sum_q = cutlass.Float32(0.0)
        sum_k = cutlass.Float32(0.0)
        inv_norm_q_scaled = cutlass.Float32(1.0)
        inv_norm_k = cutlass.Float32(1.0)

        quarter_rows: cutlass.Constexpr[int] = rows_per_group // ILP4

        for row_quad in cutlass.range(quarter_rows, unroll=1, unroll_full=(T <= 1)):
            vb4 = i_v * tile_v + group_idx * rows_per_group + row_quad * ILP4
            va = vb4
            vb = vb4 + 1
            vc = vb4 + 2
            vd = vb4 + 3

            # Read tiles at the source slot.
            hta = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, va, lane_in_group)
            )
            htb = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, vb, lane_in_group)
            )
            htc = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, vc, lane_in_group)
            )
            htd = cute.local_tile(
                h0_source, (1, 1, vec_size), (flat_state_idx, vd, lane_in_group)
            )
            # Write tiles. In single-pool (same_pool=True), they alias the
            # read tiles — nvcc DCEs the write-side base-pointer arithmetic.
            if cutlass.const_expr(same_pool):
                hta_w = hta
                htb_w = htb
                htc_w = htc
                htd_w = htd
            else:
                hta_w = cute.local_tile(
                    h0_source,
                    (1, 1, vec_size),
                    (flat_write_state_idx, va, lane_in_group),
                )
                htb_w = cute.local_tile(
                    h0_source,
                    (1, 1, vec_size),
                    (flat_write_state_idx, vb, lane_in_group),
                )
                htc_w = cute.local_tile(
                    h0_source,
                    (1, 1, vec_size),
                    (flat_write_state_idx, vc, lane_in_group),
                )
                htd_w = cute.local_tile(
                    h0_source,
                    (1, 1, vec_size),
                    (flat_write_state_idx, vd, lane_in_group),
                )
            cute.autovec_copy(hta, r_hb4_0)
            cute.autovec_copy(htb, r_hb4_1)
            cute.autovec_copy(htc, r_hb4_2)
            cute.autovec_copy(htd, r_hb4_3)

            for i in cutlass.range_constexpr(vec_size):
                r_h[0, i] = cutlass.Float32(r_hb4_0[i])
                r_h[1, i] = cutlass.Float32(r_hb4_1[i])
                r_h[2, i] = cutlass.Float32(r_hb4_2[i])
                r_h[3, i] = cutlass.Float32(r_hb4_3[i])

            for i_t in cutlass.range(T, unroll=1, unroll_full=(T <= 1)):
                if cutlass.const_expr(T > 1):
                    sQ_tile = cute.local_tile(sQ, (1, vec_size), (i_t, lane_in_group))
                    sK_tile = cute.local_tile(sK, (1, vec_size), (i_t, lane_in_group))
                    cute.autovec_copy(sQ_tile, r_q)
                    cute.autovec_copy(sK_tile, r_k)
                    if cutlass.const_expr(T > 2):
                        r_g = sGB[(i_t, 0)]
                        r_beta = sGB[(i_t, 1)]
                    else:
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

                sa = 0.0
                sb = 0.0
                sc = 0.0
                sd = 0.0
                sa2 = 0.0
                sb2 = 0.0
                sc2 = 0.0
                sd2 = 0.0
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
                        sa, sa2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[0, i], r_h[0, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(sa, sa2),
                        )
                        sb, sb2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[1, i], r_h[1, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(sb, sb2),
                        )
                        sc, sc2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[2, i], r_h[2, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(sc, sc2),
                        )
                        sd, sd2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[3, i], r_h[3, i + 1]),
                            src_b=(r_k[i], r_k[i + 1]),
                            src_c=(sd, sd2),
                        )
                    else:
                        sa, sa2 = fma_pair(
                            r_h[0, i], r_h[0, i + 1], r_k[i], r_k[i + 1], sa, sa2
                        )
                        sb, sb2 = fma_pair(
                            r_h[1, i], r_h[1, i + 1], r_k[i], r_k[i + 1], sb, sb2
                        )
                        sc, sc2 = fma_pair(
                            r_h[2, i], r_h[2, i + 1], r_k[i], r_k[i + 1], sc, sc2
                        )
                        sd, sd2 = fma_pair(
                            r_h[3, i], r_h[3, i + 1], r_k[i], r_k[i + 1], sd, sd2
                        )
                sa = sa + sa2
                sb = sb + sb2
                sc = sc + sc2
                sd = sd + sd2

                for offset in [16, 8, 4, 2, 1]:
                    sa += cute.arch.shuffle_sync_bfly(
                        sa, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sb += cute.arch.shuffle_sync_bfly(
                        sb, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sc += cute.arch.shuffle_sync_bfly(
                        sc, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    sd += cute.arch.shuffle_sync_bfly(
                        sd, offset=offset, mask=-1, mask_and_clamp=31
                    )

                vt4_slice = cute.local_tile(
                    v, (1, 1, 1, ILP4), (i_n, i_t, i_hv, vb4 // ILP4)
                )
                cute.autovec_copy(vt4_slice, r_v4_bf16)
                vna = (cutlass.Float32(r_v4_bf16[0]) - sa) * r_beta
                vnb = (cutlass.Float32(r_v4_bf16[1]) - sb) * r_beta
                vnc = (cutlass.Float32(r_v4_bf16[2]) - sc) * r_beta
                vnd = (cutlass.Float32(r_v4_bf16[3]) - sd) * r_beta

                oa = 0.0
                ob = 0.0
                oc = 0.0
                od = 0.0
                oa2 = 0.0
                ob2 = 0.0
                oc2 = 0.0
                od2 = 0.0
                for i in cutlass.range_constexpr(0, vec_size, 2):
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vna, vna),
                            src_c=(r_h[0, i], r_h[0, i + 1]),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vnb, vnb),
                            src_c=(r_h[1, i], r_h[1, i + 1]),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vnc, vnc),
                            src_c=(r_h[2, i], r_h[2, i + 1]),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(r_k[i], r_k[i + 1]),
                            src_b=(vnd, vnd),
                            src_c=(r_h[3, i], r_h[3, i + 1]),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vna, vna, r_h[0, i], r_h[0, i + 1]
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vnb, vnb, r_h[1, i], r_h[1, i + 1]
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vnc, vnc, r_h[2, i], r_h[2, i + 1]
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair(
                            r_k[i], r_k[i + 1], vnd, vnd, r_h[3, i], r_h[3, i + 1]
                        )
                    if cutlass.const_expr(use_packed_fma):
                        oa, oa2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[0, i], r_h[0, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(oa, oa2),
                        )
                        ob, ob2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[1, i], r_h[1, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(ob, ob2),
                        )
                        oc, oc2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[2, i], r_h[2, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(oc, oc2),
                        )
                        od, od2 = cute.arch.fma_packed_f32x2(
                            src_a=(r_h[3, i], r_h[3, i + 1]),
                            src_b=(r_q[i], r_q[i + 1]),
                            src_c=(od, od2),
                        )
                    else:
                        oa, oa2 = fma_pair(
                            r_h[0, i], r_h[0, i + 1], r_q[i], r_q[i + 1], oa, oa2
                        )
                        ob, ob2 = fma_pair(
                            r_h[1, i], r_h[1, i + 1], r_q[i], r_q[i + 1], ob, ob2
                        )
                        oc, oc2 = fma_pair(
                            r_h[2, i], r_h[2, i + 1], r_q[i], r_q[i + 1], oc, oc2
                        )
                        od, od2 = fma_pair(
                            r_h[3, i], r_h[3, i + 1], r_q[i], r_q[i + 1], od, od2
                        )
                oa = oa + oa2
                ob = ob + ob2
                oc = oc + oc2
                od = od + od2

                if cutlass.const_expr(cache_intermediate_states):
                    for i in cutlass.range_constexpr(vec_size):
                        r_hb4_0[i] = cutlass.BFloat16(r_h[0, i])
                        r_hb4_1[i] = cutlass.BFloat16(r_h[1, i])
                        r_hb4_2[i] = cutlass.BFloat16(r_h[2, i])
                        r_hb4_3[i] = cutlass.BFloat16(r_h[3, i])

                if cutlass.const_expr(cache_intermediate_states):
                    # The intermediate_states buffer is sized [B, T, HV, V, K]
                    # (batch-scoped, NOT pool-scoped), so this index uses i_n
                    # (the per-call batch index) and not cache_idx (the pool
                    # slot). Using cache_idx here writes OOB whenever
                    # initial_state_indices points at slots >= B (i.e. any
                    # realistic pool_size > B serving config). Fix mirrors
                    # upstream PR #3145.
                    flat_idx = i_n * T * HV + i_t * HV + i_hv
                    ita = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, va, lane_in_group),
                    )
                    itb = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, vb, lane_in_group),
                    )
                    itc = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, vc, lane_in_group),
                    )
                    itd = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec_size),
                        (flat_idx, vd, lane_in_group),
                    )
                    cute.autovec_copy(r_hb4_0, ita)
                    cute.autovec_copy(r_hb4_1, itb)
                    cute.autovec_copy(r_hb4_2, itc)
                    cute.autovec_copy(r_hb4_3, itd)

                for offset in [16, 8, 4, 2, 1]:
                    oa += cute.arch.shuffle_sync_bfly(
                        oa, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    ob += cute.arch.shuffle_sync_bfly(
                        ob, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    oc += cute.arch.shuffle_sync_bfly(
                        oc, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    od += cute.arch.shuffle_sync_bfly(
                        od, offset=offset, mask=-1, mask_and_clamp=31
                    )

                if lane_in_group == 0:
                    r_o4_bf16[0] = cutlass.BFloat16(oa)
                    r_o4_bf16[1] = cutlass.BFloat16(ob)
                    r_o4_bf16[2] = cutlass.BFloat16(oc)
                    r_o4_bf16[3] = cutlass.BFloat16(od)
                    ot4_slice = cute.local_tile(
                        o,
                        (1, 1, 1, ILP4),
                        (i_n, i_t, i_hv, vb4 // ILP4),
                    )
                    cute.autovec_copy(r_o4_bf16, ot4_slice)

            if cutlass.const_expr(not disable_state_update):
                if cutlass.const_expr(not cache_intermediate_states):
                    for i in cutlass.range_constexpr(vec_size):
                        r_hb4_0[i] = cutlass.BFloat16(r_h[0, i])
                        r_hb4_1[i] = cutlass.BFloat16(r_h[1, i])
                        r_hb4_2[i] = cutlass.BFloat16(r_h[2, i])
                        r_hb4_3[i] = cutlass.BFloat16(r_h[3, i])
                cute.autovec_copy(r_hb4_0, hta_w)
                cute.autovec_copy(r_hb4_1, htb_w)
                cute.autovec_copy(r_hb4_2, htc_w)
                cute.autovec_copy(r_hb4_3, htd_w)


# ==============================================================================
# KERNEL: wide_vec — LDG.E.128 / STG.E.128 fast path
# ==============================================================================
# 128 threads/CTA = 4 warps organised as 8 groups of 16 threads, vec=8 BF16
# (LDG.E.128 / STG.E.128) on H. ILP_ROWS=4 V-rows held in registers per thread
# per iter. Supports split-pool writes via h0_out_indices.


@cute.kernel
def gdn_wide_vec_kernel(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b_gate: cute.Tensor,
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
    tile_v: cutlass.Constexpr[int],
    num_v_tiles: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
):
    tidx, _, _ = cute.arch.thread_idx()
    lane_in_warp = tidx % 32
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)

    # 8 groups of 16 threads. Within a CTA:
    #   group_idx in [0..8)
    #   lane_in_group in [0..16) — owns K[lane_in_group*8 : +8]
    group_idx = tidx // LANES_PER_ROW
    lane_in_group = tidx % LANES_PER_ROW
    k_start = lane_in_group * ELEMS_PER_LANE

    ROWS_PER_GROUP: cutlass.Constexpr[int] = tile_v // NUM_GROUPS
    ITERS_PER_GROUP: cutlass.Constexpr[int] = ROWS_PER_GROUP // ILP_ROWS

    batch_idx, _, _ = cute.arch.block_idx()
    # Grid: (num_v_tiles × HV × B) linearized. Decode (i_n, i_hv, i_v) so each
    # CTA handles one V-tile of one (n, hv) pair.
    i_v = batch_idx % num_v_tiles
    tmp = batch_idx // num_v_tiles
    i_hv = tmp % HV
    i_n = tmp // HV
    i_h = i_hv // (HV // H)

    cache_idx = h0_indices[i_n]

    r_A_log = cutlass.Float32(A_log[i_hv])
    r_dt_bias = cutlass.Float32(dt_bias[i_hv])

    # ----- SMEM -----
    smem = cutlass.utils.SmemAllocator()
    sQ = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sK = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, K), stride=(K + 8, 1)), 16
    )
    sGB = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((T, 3), stride=(3, 1)), 16
    )

    # ----- registers -----
    vec: cutlass.Constexpr[int] = ELEMS_PER_LANE  # 8
    r_h = cute.make_rmem_tensor(
        cute.make_layout((ILP_ROWS, vec), stride=(vec, 1)), cutlass.Float32
    )
    r_q = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_k = cute.make_rmem_tensor(cute.make_layout((vec,), stride=(1,)), cutlass.Float32)
    r_q_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_k_bf16 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb0 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb1 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb2 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )
    r_hb3 = cute.make_rmem_tensor(
        cute.make_layout((vec,), stride=(1,)), cutlass.BFloat16
    )

    if cache_idx < 0:
        cache_idx = cutlass.Int32(0)

    # Split-pool write index: distinct slot to write the updated H state.
    # When same_pool=True (compile-time, set by the dispatcher whenever the
    # caller's read and write indices alias), nvcc DCEs the LDG +
    # negative-redirect compare. When False, the kernel reads the
    # write-indices tensor and applies the same null-slot redirect as the
    # read side.
    if cutlass.const_expr(same_pool):
        write_cache_idx = cache_idx
    else:
        write_cache_idx = h0_out_indices[i_n]
        if write_cache_idx < 0:
            write_cache_idx = cutlass.Int32(0)

    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv
        if cutlass.const_expr(same_pool):
            flat_write_state_idx = flat_state_idx
        else:
            flat_write_state_idx = write_cache_idx * HV + i_hv

        # ==================================================================
        # Phase 0: precompute q/k/g/beta/kq into SMEM (all 4 warps)
        # Each warp handles one token per pass; 4 warps -> ceil(T/4) passes.
        # Within a warp, threads cooperate by K-lane (16 threads cover K=128).
        # ==================================================================
        # Use the same 16-thread-per-row layout for precompute. Only the first
        # 16 lanes of each warp participate; the second 16 lanes sit idle for
        # q/k loads but contribute identical work for reduction consistency.
        # To keep it simple: all 32 threads of a warp co-load via 2 groups of 16.
        # Here lane_in_warp < 16 loads q/k for the warp's assigned token; the
        # upper 16 lanes redundantly load/compute (their writes land in the same
        # SMEM slots, idempotent).
        num_precompute_passes: cutlass.Constexpr[int] = (T + NUM_WARPS - 1) // NUM_WARPS
        member_pre = lane_in_warp % LANES_PER_ROW
        k_start_pre = member_pre * ELEMS_PER_LANE
        for pass_idx in cutlass.range_constexpr(num_precompute_passes):
            i_t_pre = pass_idx * NUM_WARPS + warp_idx
            if i_t_pre < T:
                q_tile_pre = cute.local_tile(
                    q, (1, 1, 1, vec), (i_n, i_t_pre, i_h, member_pre)
                )
                k_tile_pre = cute.local_tile(
                    k, (1, 1, 1, vec), (i_n, i_t_pre, i_h, member_pre)
                )
                cute.autovec_copy(q_tile_pre, r_q_bf16)
                cute.autovec_copy(k_tile_pre, r_k_bf16)
                for i in cutlass.range_constexpr(vec):
                    r_q[i] = cutlass.Float32(r_q_bf16[i])
                    r_k[i] = cutlass.Float32(r_k_bf16[i])

                if cutlass.const_expr(use_qk_l2norm):
                    sum_q = cutlass.Float32(0.0)
                    sum_k = cutlass.Float32(0.0)
                    for i in cutlass.range_constexpr(vec):
                        sum_q += r_q[i] * r_q[i]
                        sum_k += r_k[i] * r_k[i]
                    # 4-stage butterfly within 16-thread subgroup
                    for offset in [8, 4, 2, 1]:
                        sum_q += cute.arch.shuffle_sync_bfly(
                            sum_q, offset=offset, mask=-1, mask_and_clamp=31
                        )
                        sum_k += cute.arch.shuffle_sync_bfly(
                            sum_k, offset=offset, mask=-1, mask_and_clamp=31
                        )
                    inv_norm_q_scaled = cute.rsqrt(sum_q + 1e-6, fastmath=True) * scale
                    inv_norm_k = cute.rsqrt(sum_k + 1e-6, fastmath=True)
                    for i in cutlass.range_constexpr(vec):
                        r_q[i] = r_q[i] * inv_norm_q_scaled
                        r_k[i] = r_k[i] * inv_norm_k
                else:
                    for i in cutlass.range_constexpr(vec):
                        r_q[i] = r_q[i] * scale

                # Write q, k to SMEM (both 16-thread subgroups write same data)
                for i in cutlass.range_constexpr(vec):
                    sQ[(i_t_pre, k_start_pre + i)] = r_q[i]
                    sK[(i_t_pre, k_start_pre + i)] = r_k[i]

                # kq partial and reduce within 16-thread subgroup
                kq_partial = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(vec):
                    kq_partial += r_k[i] * r_q[i]
                for offset in [8, 4, 2, 1]:
                    kq_partial += cute.arch.shuffle_sync_bfly(
                        kq_partial, offset=offset, mask=-1, mask_and_clamp=31
                    )

                # g, beta, kq to sGB (lane 0 of warp writes)
                r_a_pre = cutlass.Float32(a[i_n, i_t_pre, i_hv])
                r_b_pre = cutlass.Float32(b_gate[i_n, i_t_pre, i_hv])
                x_pre = r_a_pre + r_dt_bias
                beta_x_pre = softplus_beta * x_pre
                exp_beta_x_pre = cute.exp(beta_x_pre, fastmath=True)
                softplus_val_pre = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
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
                r_g_pre = cute.exp(
                    -cute.exp(r_A_log, fastmath=True) * softplus_x_pre, fastmath=True
                )
                r_beta_pre = cutlass.Float32(1.0) / (
                    cutlass.Float32(1.0) + cute.exp(-r_b_pre, fastmath=True)
                )

                if lane_in_warp == 0:
                    sGB[(i_t_pre, 0)] = r_g_pre
                    sGB[(i_t_pre, 1)] = r_beta_pre
                    sGB[(i_t_pre, 2)] = kq_partial

            cute.arch.barrier()

        # ==================================================================
        # Phase 1: main compute loop
        # Each group of 16 threads owns ROWS_PER_GROUP=16 V-rows.
        # With ILP_ROWS=4, iterate 4 times. Each iter holds 4 V-rows in r_h.
        # ==================================================================
        for iter_idx in cutlass.range_constexpr(ITERS_PER_GROUP):
            v_base = i_v * tile_v + group_idx * ROWS_PER_GROUP + iter_idx * ILP_ROWS
            v0 = v_base + 0
            v1 = v_base + 1
            v2 = v_base + 2
            v3 = v_base + 3

            # Load 4 V-rows of h (LDG.128 each) into r_h from the read slot.
            ht0 = cute.local_tile(
                h0_source, (1, 1, vec), (flat_state_idx, v0, lane_in_group)
            )
            ht1 = cute.local_tile(
                h0_source, (1, 1, vec), (flat_state_idx, v1, lane_in_group)
            )
            ht2 = cute.local_tile(
                h0_source, (1, 1, vec), (flat_state_idx, v2, lane_in_group)
            )
            ht3 = cute.local_tile(
                h0_source, (1, 1, vec), (flat_state_idx, v3, lane_in_group)
            )
            # Write-side tiles. In single-pool (same_pool=True), they alias
            # the read tiles — nvcc DCEs the write-side base-pointer
            # arithmetic (the source of the +5-7 % T=1 large-B regression).
            # In split-pool (same_pool=False), separate STG.128 destinations
            # at the split-pool write slot.
            if cutlass.const_expr(same_pool):
                ht_w0 = ht0
                ht_w1 = ht1
                ht_w2 = ht2
                ht_w3 = ht3
            else:
                ht_w0 = cute.local_tile(
                    h0_source, (1, 1, vec), (flat_write_state_idx, v0, lane_in_group)
                )
                ht_w1 = cute.local_tile(
                    h0_source, (1, 1, vec), (flat_write_state_idx, v1, lane_in_group)
                )
                ht_w2 = cute.local_tile(
                    h0_source, (1, 1, vec), (flat_write_state_idx, v2, lane_in_group)
                )
                ht_w3 = cute.local_tile(
                    h0_source, (1, 1, vec), (flat_write_state_idx, v3, lane_in_group)
                )
            cute.autovec_copy(ht0, r_hb0)
            cute.autovec_copy(ht1, r_hb1)
            cute.autovec_copy(ht2, r_hb2)
            cute.autovec_copy(ht3, r_hb3)
            for i in cutlass.range_constexpr(vec):
                r_h[0, i] = cutlass.Float32(r_hb0[i])
                r_h[1, i] = cutlass.Float32(r_hb1[i])
                r_h[2, i] = cutlass.Float32(r_hb2[i])
                r_h[3, i] = cutlass.Float32(r_hb3[i])

            # Process each token sequentially (state is carried in registers).
            # Non-fused form for numerical robustness: compute s = h_decayed @ k,
            # then update h, then compute o = h_new @ q. Two reductions per token
            # instead of one, but matches baseline's accumulation order.
            for i_t in cutlass.range(T, unroll=1, unroll_full=(T <= 1)):
                r_g = sGB[(i_t, 0)]
                r_beta = sGB[(i_t, 1)]

                # Decay + h @ k (in one K-loop)
                s0 = cutlass.Float32(0.0)
                s1 = cutlass.Float32(0.0)
                s2 = cutlass.Float32(0.0)
                s3 = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(0, vec, 2):
                    kv0 = sK[(i_t, k_start + i)]
                    kv1 = sK[(i_t, k_start + i + 1)]
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
                    s0 = s0 + r_h[0, i] * kv0 + r_h[0, i + 1] * kv1
                    s1 = s1 + r_h[1, i] * kv0 + r_h[1, i + 1] * kv1
                    s2 = s2 + r_h[2, i] * kv0 + r_h[2, i + 1] * kv1
                    s3 = s3 + r_h[3, i] * kv0 + r_h[3, i + 1] * kv1

                # Butterfly reduce s across 16-thread subgroup (4 stages)
                for offset in [8, 4, 2, 1]:
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

                # Delta rule: v_new = (v - s) * beta
                v_val0 = cutlass.Float32(v[(i_n, i_t, i_hv, v0)])
                v_val1 = cutlass.Float32(v[(i_n, i_t, i_hv, v1)])
                v_val2 = cutlass.Float32(v[(i_n, i_t, i_hv, v2)])
                v_val3 = cutlass.Float32(v[(i_n, i_t, i_hv, v3)])
                vn0 = (v_val0 - s0) * r_beta
                vn1 = (v_val1 - s1) * r_beta
                vn2 = (v_val2 - s2) * r_beta
                vn3 = (v_val3 - s3) * r_beta

                # Rank-1 update + h @ q (in one K-loop)
                o0 = cutlass.Float32(0.0)
                o1 = cutlass.Float32(0.0)
                o2 = cutlass.Float32(0.0)
                o3 = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(0, vec, 2):
                    qv0 = sQ[(i_t, k_start + i)]
                    qv1 = sQ[(i_t, k_start + i + 1)]
                    kv0 = sK[(i_t, k_start + i)]
                    kv1 = sK[(i_t, k_start + i + 1)]
                    if cutlass.const_expr(use_packed_fma):
                        r_h[0, i], r_h[0, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn0, vn0),
                            src_c=(r_h[0, i], r_h[0, i + 1]),
                        )
                        r_h[1, i], r_h[1, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn1, vn1),
                            src_c=(r_h[1, i], r_h[1, i + 1]),
                        )
                        r_h[2, i], r_h[2, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn2, vn2),
                            src_c=(r_h[2, i], r_h[2, i + 1]),
                        )
                        r_h[3, i], r_h[3, i + 1] = cute.arch.fma_packed_f32x2(
                            src_a=(kv0, kv1),
                            src_b=(vn3, vn3),
                            src_c=(r_h[3, i], r_h[3, i + 1]),
                        )
                    else:
                        r_h[0, i], r_h[0, i + 1] = fma_pair(
                            kv0, kv1, vn0, vn0, r_h[0, i], r_h[0, i + 1]
                        )
                        r_h[1, i], r_h[1, i + 1] = fma_pair(
                            kv0, kv1, vn1, vn1, r_h[1, i], r_h[1, i + 1]
                        )
                        r_h[2, i], r_h[2, i + 1] = fma_pair(
                            kv0, kv1, vn2, vn2, r_h[2, i], r_h[2, i + 1]
                        )
                        r_h[3, i], r_h[3, i + 1] = fma_pair(
                            kv0, kv1, vn3, vn3, r_h[3, i], r_h[3, i + 1]
                        )
                    # h_new @ q with updated r_h
                    o0 = o0 + r_h[0, i] * qv0 + r_h[0, i + 1] * qv1
                    o1 = o1 + r_h[1, i] * qv0 + r_h[1, i + 1] * qv1
                    o2 = o2 + r_h[2, i] * qv0 + r_h[2, i + 1] * qv1
                    o3 = o3 + r_h[3, i] * qv0 + r_h[3, i + 1] * qv1

                # Butterfly reduce o
                for offset in [8, 4, 2, 1]:
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

                # Output scalar write (lane_in_group==0 only)
                if lane_in_group == 0:
                    o[(i_n, i_t, i_hv, v0)] = cutlass.BFloat16(o0)
                    o[(i_n, i_t, i_hv, v1)] = cutlass.BFloat16(o1)
                    o[(i_n, i_t, i_hv, v2)] = cutlass.BFloat16(o2)
                    o[(i_n, i_t, i_hv, v3)] = cutlass.BFloat16(o3)

                # Intermediate write (for every token when caching)
                if cutlass.const_expr(cache_intermediate_states):
                    for i in cutlass.range_constexpr(vec):
                        r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                        r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                        r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                        r_hb3[i] = cutlass.BFloat16(r_h[3, i])
                    # The intermediate_states buffer is sized [B, T, HV, V, K]
                    # (batch-scoped, NOT pool-scoped), so this index uses i_n
                    # (the per-call batch index) and not cache_idx (the pool
                    # slot). Using cache_idx here writes OOB whenever
                    # initial_state_indices points at slots >= B (i.e. any
                    # realistic pool_size > B serving config). Fix mirrors
                    # upstream PR #3145.
                    flat_idx = i_n * T * HV + i_t * HV + i_hv
                    it0 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v0, lane_in_group),
                    )
                    it1 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v1, lane_in_group),
                    )
                    it2 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v2, lane_in_group),
                    )
                    it3 = cute.local_tile(
                        intermediate_states,
                        (1, 1, vec),
                        (flat_idx, v3, lane_in_group),
                    )
                    cute.autovec_copy(r_hb0, it0)
                    cute.autovec_copy(r_hb1, it1)
                    cute.autovec_copy(r_hb2, it2)
                    cute.autovec_copy(r_hb3, it3)

            # Final state write-back to the split-pool WRITE slot. Skipped when
            # caching is enabled (inter[T-1] already holds the final state).
            if cutlass.const_expr(
                not disable_state_update and not cache_intermediate_states
            ):
                for i in cutlass.range_constexpr(vec):
                    r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                    r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                    r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                    r_hb3[i] = cutlass.BFloat16(r_h[3, i])
                cute.autovec_copy(r_hb0, ht_w0)
                cute.autovec_copy(r_hb1, ht_w1)
                cute.autovec_copy(r_hb2, ht_w2)
                cute.autovec_copy(r_hb3, ht_w3)


# ==============================================================================
# LAUNCH WRAPPER (MTP ILP=4 version)
# ==============================================================================


@cute.jit
def run_gdn_decode_bf16state_mtp_ilp4(
    h0_source: cute.Tensor,  # [pool_size * HV, V, K] BF16
    intermediate_states: cute.Tensor,  # [B * T * HV, V, K] BF16 (or dummy)
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
    same_pool: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    """Launch the MTP kernel (ILP=4) for BF16 state."""
    tile_v = tile_v_param
    vec_size = MTP_VEC_SIZE
    _, v_dim, _k_dim = (
        h0_source.layout.shape[0],
        h0_source.layout.shape[1],
        h0_source.layout.shape[2],
    )

    num_v_tiles = cute.ceil_div(v_dim, tile_v)
    grid_size = B * HV * num_v_tiles

    smem_bytes = 128
    if T > 1:
        smem_bytes = 4 * T * (K + 8) + 4 * T * (K + 8) + 4 * T * 2 + 128

    gdn_decode_bf16state_mtp_ilp4_kernel(
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
        same_pool,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[MTP_NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ==============================================================================
# LAUNCH WRAPPER (wide_vec)
# ==============================================================================


@cute.jit
def _run_wide_vec(
    h0_source: cute.Tensor,
    intermediate_states: cute.Tensor,
    A_log: cute.Tensor,
    a: cute.Tensor,
    dt_bias: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    b_gate: cute.Tensor,
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
    tile_v: cutlass.Constexpr[int],
    use_qk_l2norm: cutlass.Constexpr[bool],
    disable_state_update: cutlass.Constexpr[bool],
    cache_intermediate_states: cutlass.Constexpr[bool],
    use_packed_fma: cutlass.Constexpr[bool],
    same_pool: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    num_v_tiles: cutlass.Constexpr[int] = V // tile_v
    grid_size = B * HV * num_v_tiles
    smem_bytes = (
        4 * T * (K + 8)  # sQ FP32
        + 4 * T * (K + 8)  # sK FP32
        + 4 * T * 3  # sGB
        + 256
    )
    gdn_wide_vec_kernel(
        h0_source,
        intermediate_states,
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b_gate,
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
        tile_v,
        num_v_tiles,
        use_qk_l2norm,
        disable_state_update,
        cache_intermediate_states,
        use_packed_fma,
        same_pool,
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


# ==============================================================================
# PUBLIC API
# ==============================================================================
# Number of SMs on target GPU (detected dynamically)
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count

# GPU architecture detected once at import time — avoids per-call
# torch.cuda.get_device_capability() in the hot path.
_GPU_MAJOR, _ = torch.cuda.get_device_capability(0)
_USE_PACKED_FMA = _GPU_MAJOR >= 10


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
    initial_state_indices: Optional[torch.Tensor] = None,
    output_state_indices: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    GDN decode T=1 with BF16 state (pool mode, K=V=128 only).

    Dispatches to wide_vec when work_units is large enough
    (`B * HV >= 512`, i.e. tile_v >= 64 at T=1); otherwise falls through
    to the MTP T=1 path which picks tile_v via ``_get_bf16_mtp_config``
    and runs ``gdn_decode_bf16state_mtp_ilp4_kernel``. Both kernels
    handle split-pool natively (``output_state_indices`` !=
    ``initial_state_indices``).

    Args:
        A_log: [HV] float32
        a: [B, 1, HV] bf16
        dt_bias: [HV] float32
        q: [B, 1, H, K] bf16
        k: [B, 1, H, K] bf16
        v: [B, 1, HV, V] bf16
        b: [B, 1, HV] bf16
        initial_state_source: [pool_size, HV, V, K] bf16 — shared state pool
            (modified in-place at the slots given by indices).
        initial_state_indices: [B] int32 — pool slots to read.
            Negative entries redirect to slot 0 (null buffer). REQUIRED.
        output_state_indices: Optional [B] int32 — pool slots to write.
            Defaults to initial_state_indices when None. Forwarded to the
            kernel so split-pool is supported on either dispatch path.
        output: Optional pre-allocated [B, 1, HV, V] bf16 output
        scale: Optional, default 1/sqrt(K)

    Returns:
        output: [B, 1, HV, V] bf16
    """
    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    B, T, H, K = q.shape
    assert T == 1, f"This kernel only supports T=1, got T={T}"
    HV = v.shape[2]
    V = v.shape[3]
    assert K == 128 and V == 128, f"K and V must be 128, got K={K}, V={V}"
    assert initial_state_source.dtype == torch.bfloat16
    assert initial_state_indices is not None, (
        "Pool mode is required: pass initial_state_indices. "
        "Non-pool mode is no longer supported by the BF16 GDN kernels."
    )

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if output_state_indices is not None and output_state_indices.dtype != torch.int32:
        output_state_indices = output_state_indices.to(torch.int32)

    # Wide_vec T=1 fast path. Wide_vec uses LDG.E.128 / STG.E.128 on H, halving
    # LSU instruction count vs the baseline ILP=4 kernel. SMEM-precompute phase
    # runs ceil(T/NUM_WARPS)=1 pass at T=1, so wide_vec degenerates gracefully.
    #
    # Gate: tile_v >= 64. At T=1 the wide_vec Phase 0 precompute overhead is
    # fixed per CTA while the main loop shrinks with tile_v; tile_v=32 gives
    # only 1 ILP iter per subgroup, insufficient to amortize Phase 0.
    # Measured at HV=64: tile_v=32 regresses at B=4 (0.91x); tile_v=64 wins
    # at B=8 (1.05x). Split-pool writes are now natively supported by
    # wide_vec so no longer gated on `output_state_indices is None`.
    wv_tile_v = _select_wide_vec_tile_v(B, HV)
    if wv_tile_v is not None and wv_tile_v < 64:
        wv_tile_v = None  # tile_v=32 at T=1 loses to MTP fallback
    if wv_tile_v is not None:
        return gated_delta_rule_mtp_wide_vec(
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
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
            intermediate_states_buffer=None,  # T=1 has no cache
            disable_state_update=False,  # T=1 default: write final state
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            output=output,
            tile_v=wv_tile_v,
        )

    # Wide_vec didn't fire (B*HV too small at T=1, i.e. tile_v < 64).
    # Route through the MTP T=1 path which dispatches to mtp_ilp4_kernel
    # via _get_bf16_mtp_config.
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
        initial_state_indices=initial_state_indices,
        output_state_indices=output_state_indices,
        output=output,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        scale=scale,
    )


# ==============================================================================
# MTP PUBLIC API
# ==============================================================================
_compiled_kernels_mtp: dict = {}
_compiled_kernels_wide_vec: dict = {}


def _select_tile_v_for_mtp(B: int, HV: int, V: int, T: int = 1) -> int:
    """Select optimal tile_v for the MTP BF16 kernel based on batch size and T.

    tile_v must be a multiple of 4 * MTP_ILP4_ROWS (= 16) and divide V=128.
    Valid values: 32, 64, 128.

    For large batch sizes, use larger tile_v to reduce block count and overhead.
    """
    for tv in [128, 64, 32]:
        num_v_tiles = V // tv
        grid_size = B * HV * num_v_tiles
        # Want at least 4 waves for good occupancy
        if grid_size >= 4 * NUM_SMS:
            return tv
    return 32  # Minimum tile_v for maximum parallelism


def _get_bf16_mtp_config(
    batch_size: int, seq_len: int, num_v_heads: int, v_dim: int
) -> tuple:
    """Select ``(tile_v, ilp_rows)`` for the BF16 MTP kernel.

    Smaller tile_v + lower ILP gives more CTAs (better SM utilization at small
    batch) at the cost of register pressure reduction → higher occupancy.

    With ILP=4: ~48 regs/thread → ~62% occupancy.

    Wide_vec now covers every shape where ILP=8 was historically a win
    (B*HV >= 128 at T>=2; B>=8 at T=1 with HV=64). The MTP fallback is
    only reached at low work_units / T=1 small-batch redirect, where
    ILP=4's higher occupancy beats ILP=8's larger per-CTA work amount.

    Returns ``(tile_v, ilp_rows)`` with ``ilp_rows == 4``.
    """
    work_units = batch_size * num_v_heads
    if work_units <= 128:
        # Tiny grid: small tile_v gives more CTAs to fill SMs.
        return min(16, v_dim), 4
    return _select_tile_v_for_mtp(batch_size, num_v_heads, v_dim, seq_len), 4


# Threshold above which `gated_delta_rule_mtp` dispatches to the wide_vec
# kernel. Exposed at module scope so benchmarks can raise it to bypass the
# dispatcher and measure the baseline path alone. See
# results/bf16_mtp_optimization_apr18/wide_vec_design.md for derivation.
# Kept for external callers / benchmark monkey-patching; the actual tile_v
# picking is done by `_select_wide_vec_tile_v` below.
_WIDE_VEC_WORK_UNITS_THRESHOLD = 128


def _select_wide_vec_tile_v(B: int, HV: int) -> Optional[int]:
    """Pick a wide_vec tile_v by `work_units = B * HV`, or return None to
    indicate "no wide_vec — use the baseline ILP=4/8 path instead."

    K = V = 128 is required by callers (asserted at the public API entry).

    Thresholds derived from the (B, T) sweep in
    `results/bf16_mtp_optimization_apr21/` (B200, HV=64, T=2):

    ==========================  ===========  ==========================
    work_units = B * HV         tile_v       where this picks
    ==========================  ===========  ==========================
    >= 1024                     128          B >= 16 at HV=64
    >= 512                      64           B =  8 at HV=64 (~1.10× over baseline)
    >= 128                      32           B = 2..4 at HV=64 (~1.17× at B=4)
    <  128                      None         baseline ILP=4/8
    ==========================  ===========  ==========================
    """
    work_units = B * HV
    if work_units >= 1024:
        return 128
    if work_units >= 512:
        return 64
    if work_units >= _WIDE_VEC_WORK_UNITS_THRESHOLD:
        return 32
    return None


# ==============================================================================
# PYTHON ENTRY (wide_vec) — called from gated_delta_rule and gated_delta_rule_mtp
# ==============================================================================


def gated_delta_rule_mtp_wide_vec(
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
    tile_v: int = 128,
) -> torch.Tensor:
    """Wide-vector BF16 GDN MTP decode.

    Prefer calling via the production entry point `gated_delta_rule_mtp`
    (T>=2) or `gated_delta_rule` (T=1), which auto-dispatch to this
    kernel when `B * HV >= 128` (tile_v=32) up through `>= 1024`
    (tile_v=128). Call this symbol directly only when you know your
    work size hits the fast path.

    When `intermediate_states_buffer is not None`, skips the final state
    writeback; caller must read the final state from `buffer[:, T-1]`.

    `output_state_indices` enables split-pool semantics: when non-None and
    different from `initial_state_indices`, the kernel reads from the read
    slots and writes the updated H state to the write slots. When None
    (or pointing at the same tensor as `initial_state_indices`), the
    kernel reads and writes the same slot (single-pool).
    """
    global _compiled_kernels_wide_vec

    assert q is not None and k is not None and v is not None
    assert b is not None and initial_state_source is not None

    B_val, T_val, H_val, K_val = q.shape
    HV_val = v.shape[2]
    V_val = v.shape[3]
    pool_size = initial_state_source.shape[0]
    assert K_val == 128 and V_val == 128
    assert initial_state_source.dtype == torch.bfloat16
    assert tile_v in (32, 64, 128), f"tile_v must be 32/64/128, got {tile_v}"
    assert V_val % tile_v == 0 and (tile_v // NUM_GROUPS) % ILP_ROWS == 0, (
        f"tile_v={tile_v} incompatible with 8 groups × ILP=4 layout"
    )

    if scale is None:
        scale = 1.0 / math.sqrt(K_val)

    h0_source = initial_state_source.reshape(pool_size * HV_val, V_val, K_val)

    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        # The cache buffer is BATCH-scoped: shape [B, T, HV, V, K]. The kernel
        # indexes it by i_n (the per-call batch index), NOT by cache_idx (the
        # pool slot), so a pool_size-sized buffer would be OOB-prone. Fix
        # mirrors upstream PR #3145.
        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]
        assert buffer_size == B_val, (
            f"intermediate_states_buffer dim 0 ({buffer_size}) must equal "
            f"batch size B={B_val}; the buffer is batch-scoped, not pool-scoped"
        )
        assert cache_steps >= T_val
        assert intermediate_states_buffer.dtype == torch.bfloat16
        intermediate_states = intermediate_states_buffer.reshape(
            B_val * cache_steps * HV_val, V_val, K_val
        )
        if not intermediate_states.is_contiguous():
            intermediate_states = intermediate_states.contiguous()
        # Skip the redundant final writeback when caching is on.
        effective_disable_final = True
    else:
        intermediate_states = h0_source[:1, :1, :1]
        effective_disable_final = disable_state_update

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_packed_fma = _USE_PACKED_FMA
    # Single-pool callers either pass output_state_indices=None (defaults to
    # initial_state_indices below) or pass the same tensor for both. In both
    # cases the kernel can elide write-side base-pointer arithmetic via the
    # same_pool Constexpr; nvcc DCEs the dead branch in the compiled cubin.
    same_pool = (
        output_state_indices is None or output_state_indices is initial_state_indices
    )

    cache_key = (
        "v3_mtp_bf16_tiled",
        B_val,
        T_val,
        H_val,
        HV_val,
        K_val,
        V_val,
        pool_size,
        tile_v,
        effective_disable_final,
        cache_intermediate_states,
        use_qk_l2norm_in_kernel,
        scale,
        softplus_beta,
        softplus_threshold,
        use_packed_fma,
        same_pool,
    )
    if cache_key not in _compiled_kernels_wide_vec:
        default_indices = torch.arange(B_val, dtype=torch.int32, device=q.device)
        default_output = torch.empty(
            B_val, T_val, HV_val, V_val, device=q.device, dtype=q.dtype
        )

        if initial_state_indices is None:
            initial_state_indices = default_indices
        if output is None:
            output = default_output

        # Compile-time indices template: any [B] int32 on the right device.
        # Both read and write index tensors share the same shape/dtype so
        # the same dlpack handle works for both slots.
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
        h0_idx_ = from_dlpack(
            initial_state_indices, assumed_align=32, enable_tvm_ffi=True
        )
        h0_out_idx_ = from_dlpack(
            initial_state_indices, assumed_align=32, enable_tvm_ffi=True
        )

        _compiled_kernels_wide_vec[cache_key] = {
            "compiled": cute.compile(
                _run_wide_vec,
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
                HV_val,
                B_val,
                T_val,
                H_val,
                K_val,
                V_val,
                tile_v,
                use_qk_l2norm_in_kernel,
                effective_disable_final,
                cache_intermediate_states,
                use_packed_fma,
                same_pool,
                stream,
                options="--enable-tvm-ffi --generate-line-info --opt-level 3",
            ),
            "default_indices": default_indices,
            "output": default_output,
        }

    cache = _compiled_kernels_wide_vec[cache_key]
    if initial_state_indices is None:
        initial_state_indices = cache["default_indices"]
    if output_state_indices is None:
        # Single-pool: read==write. Reuse the same indices tensor — no extra
        # allocation, kernel still produces the same address for both slots.
        output_state_indices = initial_state_indices
    if output is None:
        output = cache["output"]

    cache["compiled"](
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
        output_state_indices,
        stream,
    )
    return output


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
        intermediate_states_buffer: Optional [B, T, HV, V, K] bf16. Note: this
            buffer is BATCH-scoped, not pool-scoped — the kernel indexes it by
            the per-call batch index (i_n), not by the pool slot. Sizing it
            larger than B silently wastes memory; sizing it smaller than B
            triggers an assertion (see the OOB fix mirroring upstream PR #3145).
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
    assert initial_state_indices is not None, (
        "Pool mode is required: pass initial_state_indices. "
        "Non-pool mode is no longer supported by the BF16 GDN MTP kernels."
    )

    if scale is None:
        scale = 1.0 / math.sqrt(K)

    if output_state_indices is not None and output_state_indices.dtype != torch.int32:
        output_state_indices = output_state_indices.to(torch.int32)

    # Reshape state to [pool_size * HV, V, K]
    h0_source = initial_state_source.reshape(pool_size * HV, V, K)

    # Handle intermediate states. The cache buffer is BATCH-scoped: shape
    # [B, T, HV, V, K]. The kernel indexes it by i_n (per-call batch index),
    # NOT by cache_idx (pool slot), so a pool_size-sized buffer would be
    # OOB-prone. Fix mirrors upstream PR #3145.
    cache_intermediate_states = intermediate_states_buffer is not None
    if cache_intermediate_states:
        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]
        assert buffer_size == B, (
            f"intermediate_states_buffer dim 0 ({buffer_size}) must equal "
            f"batch size B={B}; the buffer is batch-scoped, not pool-scoped"
        )
        assert cache_steps >= T, (
            f"intermediate_states_buffer dim 1 ({cache_steps}) must be >= T={T}"
        )
        assert intermediate_states_buffer.dtype == torch.bfloat16
        intermediate_states = intermediate_states_buffer.reshape(
            B * cache_steps * HV, V, K
        )
        if not intermediate_states.is_contiguous():
            intermediate_states = intermediate_states.contiguous()
    else:
        intermediate_states = h0_source[
            :1, :1, :1
        ]  # Reuse existing allocation as dummy

    # Dispatch to the wide_vec kernel when work_units (B*HV) amortizes its
    # lower per-CTA parallelism. ``_select_wide_vec_tile_v`` picks tile_v
    # ∈ {32, 64, 128} so wide_vec covers ``B*HV >= 128`` at T>=2; below
    # that it returns None and we fall back to mtp_ilp4. Wide_vec
    # supports split-pool natively (PR #2905); ``output_state_indices``
    # is forwarded to the kernel. T=1 dispatches via gated_delta_rule
    # (different gate, requires tile_v >= 64).
    wv_tile_v = _select_wide_vec_tile_v(B, HV) if T >= 2 else None
    if wv_tile_v is not None:
        return gated_delta_rule_mtp_wide_vec(
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
            initial_state_indices=initial_state_indices,
            output_state_indices=output_state_indices,
            intermediate_states_buffer=intermediate_states_buffer,
            disable_state_update=disable_state_update,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            scale=scale,
            output=output,
            tile_v=wv_tile_v,
        )

    # Wide_vec didn't fire (work_units < 128 at T>=2, or T=1 small batch
    # redirected here). Falls to the ILP=4 MTP path
    # (mtp_ilp4_kernel), which natively supports both single- and
    # split-pool, so the config picker is independent of pool mode.
    tile_v, ilp_rows = _get_bf16_mtp_config(B, T, HV, V)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    use_packed_fma = _USE_PACKED_FMA
    # Set same_pool=True when reads and writes alias (single-pool); the
    # kernel then DCEs write-side base-pointer arithmetic.
    same_pool = (
        output_state_indices is None or output_state_indices is initial_state_indices
    )

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
        ilp_rows,
        disable_state_update,
        cache_intermediate_states,
        use_qk_l2norm_in_kernel,
        scale,
        softplus_beta,
        softplus_threshold,
        use_packed_fma,
        same_pool,
    )
    if cache_key not in _compiled_kernels_mtp:
        # First call for this shape: allocate default indices/output and do
        # dlpack conversions once for compilation. Steady-state calls pass
        # torch tensors straight to the compiled callable (tvm-ffi accepts
        # either) and reuse these cached defaults when the caller doesn't
        # provide their own.
        default_indices = torch.arange(B, dtype=torch.int32, device=q.device)
        default_output = torch.empty(B, T, HV, V, device=q.device, dtype=q.dtype)

        h_ = from_dlpack(h0_source, assumed_align=32, enable_tvm_ffi=True)
        inter_ = from_dlpack(intermediate_states, assumed_align=32, enable_tvm_ffi=True)
        q_ = from_dlpack(q, assumed_align=32, enable_tvm_ffi=True)
        k_ = from_dlpack(k, assumed_align=32, enable_tvm_ffi=True)
        v_ = from_dlpack(v, assumed_align=32, enable_tvm_ffi=True)
        a_ = from_dlpack(a, assumed_align=32, enable_tvm_ffi=True)
        b_ = from_dlpack(b, assumed_align=32, enable_tvm_ffi=True)
        A_log_ = from_dlpack(A_log, assumed_align=32, enable_tvm_ffi=True)
        dt_bias_ = from_dlpack(dt_bias, assumed_align=32, enable_tvm_ffi=True)
        o_ = from_dlpack(default_output, assumed_align=32, enable_tvm_ffi=True)
        h0_idx_ = from_dlpack(default_indices, assumed_align=32, enable_tvm_ffi=True)
        h0_out_idx_ = from_dlpack(
            default_indices, assumed_align=32, enable_tvm_ffi=True
        )

        _compiled_kernels_mtp[cache_key] = {
            "compiled": cute.compile(
                run_gdn_decode_bf16state_mtp_ilp4,
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
                same_pool,
                stream,
                options="--enable-tvm-ffi --generate-line-info --opt-level 3",
            ),
            "default_indices": default_indices,
            "output": default_output,
        }

    cache = _compiled_kernels_mtp[cache_key]

    if output_state_indices is None:
        output_state_indices = initial_state_indices
    if output is None:
        output = cache["output"]

    cache["compiled"](
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
        output_state_indices,
        stream,
    )

    return output


# Backward-compatible aliases
gated_delta_rule_bf16state_cooprow = gated_delta_rule
gated_delta_rule_bf16state_cooprow_mtp = gated_delta_rule_mtp
