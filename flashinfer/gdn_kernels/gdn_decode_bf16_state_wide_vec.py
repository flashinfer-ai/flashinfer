"""
Wide-vector BF16 GDN MTP decode kernel.

Speed advantage over the production ILP=8 kernel comes from widening the
per-thread vector width from 4 BF16 (LDG.E.64 / STG.E.64) to 8 BF16
(LDG.E.128 / STG.E.128) for the H state tensor, which halves LSU instruction
count at equal data throughput. Uses **no TMA**, **no persistent CTAs**, and
**no warp specialization**.

Thread / data layout (K=V=128 fast path):
  - 128 threads/CTA = 4 warps organised as **8 groups of 16 threads**
    (baseline: 4 groups of 32 threads)
  - **vec = 8 BF16** per thread → LDG.E.128 / STG.E.128
    (baseline: vec = 4 BF16 → LDG.E.64 / STG.E.64)
  - **ILP = 4** V-rows held in registers per thread per iter
    (baseline: ILP = 8; register state 4 × 8 = 32 FP32 matches baseline's 8 × 4)
  - **tile_v ∈ {32, 64, 128}** — configurable per call; grid gains a V-tile
    dimension so the kernel can target small `work_units = B * HV` sizes
    without starving SMs
  - 4-stage butterfly shuffle within 16-thread subgroups
    (baseline: 5-stage within full 32-thread warps)
  - Grid = B × HV × (V / tile_v)

tile_v picker (invoked from `gated_delta_rule_mtp`):
  - work_units ≥ 1024 → tile_v=128 (e.g. B ≥ 16 at HV=64)
  - work_units ≥  512 → tile_v=64  (e.g. B = 8 at HV=64, ~1.10× over baseline)
  - work_units ≥  128 → tile_v=32  (e.g. B = 2..4 at HV=64, ~1.17× over baseline at B=4)
  - below              → dispatcher falls back to baseline ILP=4/8

Skip-final-write: when `intermediate_states_buffer` is provided, the last
cached slot `[:, T-1]` already holds the final state, so the final writeback
to `initial_state_source` is skipped to save ~67 MB / call at HV=64 K=V=128.
"""

import math
from typing import Optional

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass.cute.runtime import from_dlpack

# Reuse reference impl from horiz for testing
from .gdn_decode_bf16_state_tma_horiz import _reference_gdn_mtp  # noqa: F401


# ===== constants =====
# Layout is identical to gdn_decode_bf16_state_wide_vec.py — LDG.128 over
# 16-thread subgroups, 8 subgroups per CTA — but this variant adds a V-tile
# dimension to the grid so smaller work_units (B * HV) can still fill the
# machine. tile_v is passed as a constexpr at compile time; the kernel decodes
# (i_n, i_hv, i_v) from the linear block_idx.
LANES_PER_ROW = 16  # 16 threads cooperate on one V-row's K=128 BF16
ELEMS_PER_LANE = 8  # 8 BF16 = LDG.128
TILE_K = 128
NUM_WARPS = 4
NUM_THREADS = NUM_WARPS * 32  # 128
NUM_GROUPS = NUM_THREADS // LANES_PER_ROW  # 8 groups of 16 threads
ILP_ROWS = 4  # 4 V-rows held in regs per thread per iter


@cute.jit
def fma_pair_mul(a1, a2, b1, b2):
    return a1 * b1, a2 * b2


@cute.jit
def fma_pair(a1, a2, b1, b2, c1, c2):
    return a1 * b1 + c1, a2 * b2 + c2


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

    if cache_idx >= 0:
        flat_state_idx = cache_idx * HV + i_hv

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

            # Load 4 V-rows of h (LDG.128 each) into r_h
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
                    flat_idx = cache_idx * T * HV + i_t * HV + i_hv
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

            # Final state write-back: skip when caching (inter[T-1] already has it)
            if cutlass.const_expr(
                not disable_state_update and not cache_intermediate_states
            ):
                for i in cutlass.range_constexpr(vec):
                    r_hb0[i] = cutlass.BFloat16(r_h[0, i])
                    r_hb1[i] = cutlass.BFloat16(r_h[1, i])
                    r_hb2[i] = cutlass.BFloat16(r_h[2, i])
                    r_hb3[i] = cutlass.BFloat16(r_h[3, i])
                cute.autovec_copy(r_hb0, ht0)
                cute.autovec_copy(r_hb1, ht1)
                cute.autovec_copy(r_hb2, ht2)
                cute.autovec_copy(r_hb3, ht3)


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
    ).launch(
        grid=(grid_size, 1, 1),
        block=[NUM_THREADS, 1, 1],
        smem=smem_bytes,
        stream=stream,
    )


_compiled_kernels_wide_vec: dict = {}
NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
_GPU_MAJOR, _ = torch.cuda.get_device_capability(0)
_USE_PACKED_FMA = _GPU_MAJOR >= 10


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
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    disable_state_update: bool = False,
    use_qk_l2norm_in_kernel: bool = True,
    scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
    tile_v: int = 128,
) -> torch.Tensor:
    """Wide-vector BF16 GDN MTP decode.

    Prefer calling via the production entry point `gated_delta_rule_mtp`,
    which auto-dispatches to this kernel when `B * HV >= 1024`. Call this
    symbol directly only when you know your work size hits the fast path.

    When `intermediate_states_buffer is not None`, skips the final state
    writeback; caller must read the final state from `buffer[:, T-1]`.
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
        buffer_size = intermediate_states_buffer.shape[0]
        cache_steps = intermediate_states_buffer.shape[1]
        assert cache_steps >= T_val
        assert intermediate_states_buffer.dtype == torch.bfloat16
        intermediate_states = intermediate_states_buffer.reshape(
            buffer_size * cache_steps * HV_val, V_val, K_val
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
                stream,
                options="--enable-tvm-ffi --generate-line-info --opt-level 3",
            ),
            "default_indices": default_indices,
            "output": default_output,
        }

    cache = _compiled_kernels_wide_vec[cache_key]
    if initial_state_indices is None:
        initial_state_indices = cache["default_indices"]
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
        stream,
    )
    return output
