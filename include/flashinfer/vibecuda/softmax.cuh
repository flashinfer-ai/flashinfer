/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// VibeCUDA fused row-wise FP32 softmax for NVIDIA SM100-class GPUs
// (B200 / B300, compute capability 10.x).  Backend name: "vibecuda".
//
// Numerically-stable softmax:  m = max_j x_j ;  e_j = exp(x_j - m) ;
// y_j = e_j / sum_k e_k.
//
// Two kernel families, dispatched by vocab-width bands:
//
//   * narrow/medium rows (vocab <= 64000): register-resident cluster softmax.
//     The whole row lives in per-thread float4 registers (8/thread), one
//     global read + one write.  CTA width is chosen so several small CTAs
//     stay resident per SM and memory phases overlap reduction phases.
//   * wide rows (vocab > 64000): cp.async SMEM-streaming cluster softmax.
//     R CTAs of a thread cluster own contiguous slabs of the row; each CTA
//     streams its slab through a ring of 16 KB shared-memory chunks using
//     LDGSTS (cp.async.cg) so DRAM depth lives in the copy engine instead of
//     the register file.  ~32 regs/thread -> four 512-thread CTAs per SM
//     with ~192 KB of copies in flight per SM.  Pass 2 rereads the row from
//     L2 (per-wave working set is a few tens of MB) and writes the
//     normalized output.  An up-front full-slab staging variant wins on
//     thin/mid grids (zero pass-2 reread) and is band-scoped by the same
//     occupancy crossover.
//
// Cross-CTA (cluster) max/sum combine, shared by both families:
//   * each warp builds its own (max, sum) online-softmax pair by shuffles,
//   * lanes r < R mirror the pair into the pair pool of EVERY cluster CTA
//     via distributed shared memory (one relaxed vector store per target),
//   * ONE cluster.sync() rendezvous,
//   * warp 0 of each CTA then reads all P = R*NW pairs of its own pool,
//     merges them into the row (M, S), publishes the pair through one
//     shared broadcast slot, and one __syncthreads() releases the CTA into
//     the normalization phase.  Merging in ONE warp (instead of redundantly
//     in every warp) removes ~9-25% of the shared-load wavefronts at NCU
//     and measured a small net latency win across the 40-shape suite.
//   A generation-stamped polling variant of this protocol (per-slot
//   acquire spins instead of cluster.sync) measured 20-40% SLOWER on
//   SM100; a device-scope global-atomic rendezvous lost 30-50%.  Do not
//   retry either.
//
// Cross-row soft pipeline (multi-wave grids only): each cluster loops over
// successive rows (stride = launched clusters).  After posting a row's pair
// the cluster barrier is SPLIT into barrier_arrive / barrier_wait; between
// the two the CTA issues a cp.async prefetch of its next row into a per-CTA
// SMEM slab, so the DSM rendezvous window carries DRAM traffic.  Pair pools
// alternate by iteration parity.  Engaged only when the row grid exceeds
// one resident wave (occupancy-queried); per-row launches are used below.
//
// Temperature scaling: out = softmax(logits / temperature), with
// temperature either a scalar (temperature_val) or a per-row tensor
// (temperature_arr).  inv_t = 1 / temperature folds into every exp
// argument: exp((x - m) * inv_t) == exp(x / t - m / t), and max is
// invariant to positive scaling, so all online (max, sum) accumulators
// stay in raw-logit space.  The packed f32x2 exp paths absorb inv_t into
// their loop-invariant vector constants at zero marginal instruction cost;
// at temperature == 1 the result is bit-identical to the untempered
// kernels (x * 1.0f and fmaf(x, 1.0f, c) round to exactly x - m).

#ifndef FLASHINFER_VIBECUDA_SOFTMAX_CUH_
#define FLASHINFER_VIBECUDA_SOFTMAX_CUH_

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cstdint>

namespace flashinfer {
namespace vibecuda {

namespace cg = cooperative_groups;

#define VIBECUDA_FULL_MASK 0xffffffffu
#define VIBECUDA_NEG_INF (-CUDART_INF_F)

// Leader-warp pair-pool reduction is the shipped protocol: only warp 0 of
// each CTA merges the P pool pairs, publishes the row (M, S) through one
// shared broadcast slot, and one __syncthreads releases the CTA (pairs
// posted through DSM).  The extra float2 slot is the broadcast slot.
#define VIBECUDA_LW_EXTRA 1

// Phase-3 output store policy of the register kernels: st.global.wt
// (write-through).  NCU sector attribution shows the SM->L2 store side is
// already exactly ideal (1.00x output bytes), and a replicated A/B of the
// store policies gave write-through +1.4..+3.1% on the 64K band with zero
// losses; the visible lever is wall time, not sector traffic.

__device__ __forceinline__ void st4_out(float4* p, float4 v) { __stwt(p, v); }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(VIBECUDA_FULL_MASK, v, o));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(VIBECUDA_FULL_MASK, v, o);
  return v;
}

__device__ __forceinline__ float4 f4_scale(float4 a, float s) {
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

// out = exp((x - m) * inv_t) for one float4; equals exp(x / t - m / t).
__device__ __forceinline__ float4 f4_exp(float4 a, float m, float inv_t) {
  const float nmt = -m * inv_t;
  float4 e;
  e.x = __expf(fmaf(a.x, inv_t, nmt));
  e.y = __expf(fmaf(a.y, inv_t, nmt));
  e.z = __expf(fmaf(a.z, inv_t, nmt));
  e.w = __expf(fmaf(a.w, inv_t, nmt));
  return e;
}

__device__ __forceinline__ float f4_max(float4 a) {
  return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w));
}

__device__ __forceinline__ float f4_sum(float4 a) { return (a.x + a.y) + (a.z + a.w); }

// Blackwell SM10x packed-f32x2 arithmetic: one instruction operates on the
// two lanes of a b64 register (round-toward-zero, flush-to-zero; both
// acceptable for exp-argument prep and output scaling at fp32 tolerance).
__device__ __forceinline__ unsigned long long pack2(float v) {
  unsigned long long u = (unsigned long long)__float_as_uint(v);
  return u | (u << 32);
}

__device__ __forceinline__ float2 f2_packed_fma(float2 a, unsigned long long b2,
                                                unsigned long long c2) {
  float2 r;
  asm("fma.rz.ftz.f32x2 %0, %1, %2, %3;"
      : "=l"(*reinterpret_cast<unsigned long long*>(&r))
      : "l"(*reinterpret_cast<const unsigned long long*>(&a)), "l"(b2), "l"(c2));
  return r;
}

__device__ __forceinline__ float2 f2_packed_mul(float2 a, unsigned long long b2) {
  float2 r;
  asm("mul.rz.ftz.f32x2 %0, %1, %2;"
      : "=l"(*reinterpret_cast<unsigned long long*>(&r))
      : "l"(*reinterpret_cast<const unsigned long long*>(&a)), "l"(b2));
  return r;
}

__device__ __forceinline__ float2 f2_exp2(float2 t) {
  t.x = exp2f(t.x);
  t.y = exp2f(t.y);
  return t;
}

// out = exp((x - m) * inv_t) * s for one float4 (2 packed FMA + 4 MUFU + 2
// packed MUL instead of 4 FMA + 4 MUFU + 4 MUL).  The constants are
// l2e2 = pack2(log2e * inv_t), negml2e2 = pack2(-m * log2e * inv_t),
// invs2 = pack2(s); inv_t is pre-folded by the caller at zero marginal
// instruction cost.
__device__ __forceinline__ float4 f4_expn_pack(float4 v, unsigned long long l2e2,
                                               unsigned long long negml2e2,
                                               unsigned long long invs2) {
  float2 t0 = f2_packed_fma(make_float2(v.x, v.y), l2e2, negml2e2);
  float2 t1 = f2_packed_fma(make_float2(v.z, v.w), l2e2, negml2e2);
  t0 = f2_packed_mul(f2_exp2(t0), invs2);
  t1 = f2_packed_mul(f2_exp2(t1), invs2);
  return make_float4(t0.x, t0.y, t1.x, t1.y);
}

// out = exp((x - m) * inv_t) for one float4: 2 packed FMA + 4 MUFU.
__device__ __forceinline__ float4 f4_exp_pack(float4 v, unsigned long long l2e2,
                                              unsigned long long negml2e2) {
  float2 t0 = f2_packed_fma(make_float2(v.x, v.y), l2e2, negml2e2);
  float2 t1 = f2_packed_fma(make_float2(v.z, v.w), l2e2, negml2e2);
  t0 = f2_exp2(t0);
  t1 = f2_exp2(t1);
  return make_float4(t0.x, t0.y, t1.x, t1.y);
}

// online-softmax pair merge: (m1, s1) . (m2, s2) -> (max, sum rescaled).
// Accumulator sums live in exp((x - m) * inv_t) space, so the rescale
// factors carry the same inv_t.
__device__ __forceinline__ float2 pair_merge(float2 p, float2 q, float inv_t) {
  const float mn = fmaxf(p.x, q.x);
  float ca = __expf((p.x - mn) * inv_t);
  float cb = __expf((q.x - mn) * inv_t);
  if (mn == VIBECUDA_NEG_INF) {
    ca = 0.f;
    cb = 0.f;
  }
  return make_float2(mn, p.y * ca + q.y * cb);
}

// fold a single value into a running online-softmax accumulator
__device__ __forceinline__ void online_add(float x, float& m, float& s, float inv_t) {
  const float mn = fmaxf(m, x);
  float c = __expf((m - mn) * inv_t);
  float e = __expf((x - mn) * inv_t);
  // all -inf so far: (-inf) - (-inf) is NaN; the elements contribute nothing
  if (mn == VIBECUDA_NEG_INF) {
    c = 0.f;
    e = 0.f;
  }
  s = s * c + e;
  m = mn;
}

// merge all P pool pairs within one warp: lane-strided reads + shuffle tree
template <int P, int K>
__device__ __forceinline__ float2 pool_merge_warp(const float2* pool_ms, int lane, float inv_t) {
  float2 p = make_float2(VIBECUDA_NEG_INF, 0.f);
#pragma unroll
  for (int k = 0; k < K; ++k) {
    const int i = lane + k * 32;
    if (i < P) p = pair_merge(p, pool_ms[i], inv_t);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float2 q;
    q.x = __shfl_xor_sync(VIBECUDA_FULL_MASK, p.x, o);
    q.y = __shfl_xor_sync(VIBECUDA_FULL_MASK, p.y, o);
    p = pair_merge(p, q, inv_t);
  }
  return p;
}

// DSM write of this warp's (m,s) pair into CTA `rank`'s pool slot.  Ordering
// vs. the readers is provided by the cluster.sync() that follows all posts.
__device__ __forceinline__ void dsm_post_pair(float2* pool_ms, int slot, unsigned rank, float m,
                                              float s) {
  const unsigned ms_off = (unsigned)__cvta_generic_to_shared(pool_ms + slot);
  unsigned ms_dst;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ms_dst) : "r"(ms_off), "r"(rank));
  asm volatile("st.shared::cluster.v2.f32 [%0], {%1, %2};" ::"r"(ms_dst), "f"(m), "f"(s)
               : "memory");
}

// ---------------------------------------------------------------------------
// Register-resident cluster softmax.  R CTAs (a thread cluster) cooperate on
// one row; each CTA has BT threads; each thread owns up to 8 float4 values
// in registers laid out with stride T = R*BT for full coalescing.
// ---------------------------------------------------------------------------
// PK2 (packed-f32x2 exp-argument path) is OFF by default: the 512-thread
// (64K-band) arms below launch the 3-argument template form and must stay on
// the scalar phase-2 loop, matching the measured dispatch this design was
// accepted with.  The 128/256-thread arms opt in explicitly via
// detail::kPk2Reg at every launch site.
template <int R, int BT, int LV, bool PK2 = false>
__global__ void __launch_bounds__(BT, 1024 / BT) __cluster_dims__(R, 1, 1)
    softmax_cluster_kernel(const float* __restrict__ x, float* __restrict__ y, int n,
                           const float* __restrict__ t_arr, float t_val) {
  static_assert(BT == 512 || BT == 256 || BT == 128, "BT must divide the warp budget");
  static_assert(LV >= 1 && LV <= 8, "float4 registers per thread");
  cg::cluster_group cluster = cg::this_cluster();
  const int row = blockIdx.x / R;
  const int crank = (int)cluster.block_rank();
  const float* __restrict__ xr = x + (long long)row * n;
  float* __restrict__ yr = y + (long long)row * n;

  // out = softmax(logits / t); accept scalar or per-row temperature
  const float t_row = (t_arr == nullptr) ? t_val : t_arr[row];
  const float inv_t = (t_row == 0.f) ? 0.f : 1.0f / t_row;

  const int T = R * BT;
  const int t = crank * BT + (int)threadIdx.x;
  const int V = n >> 2;    // number of full float4 vectors
  const int tail = n & 3;  // scalar leftovers at row end
  const float4* xv = reinterpret_cast<const float4*>(xr);
  float4* yv = reinterpret_cast<float4*>(yr);

  // PDL: index math above runs ahead of the previous grid's tail; all global
  // accesses wait for full visibility of the previous grid's stores.
  cudaGridDependencySynchronize();

  constexpr int NW = BT / 32;
  constexpr int P = R * NW;         // cluster pair pool size (<= 256)
  constexpr int K = (P + 31) / 32;  // strided pool polls per lane
  __shared__ float2 pool_ms[P];

  // ---- phase 1: issue every outstanding load before first use -------------
  float4 v[LV];
#pragma unroll
  for (int j = 0; j < LV; ++j) {
    int idx = t + j * T;
    if (idx < V) {
      v[j] = __ldcs(xv + idx);
    } else {
      v[j] = make_float4(VIBECUDA_NEG_INF, VIBECUDA_NEG_INF, VIBECUDA_NEG_INF, VIBECUDA_NEG_INF);
    }
  }
  // scalar tail handled by global lane t == 0 (it always exists)
  float sc[3];
  if (t == 0) {
    const float* xs = xr + (V << 2);
#pragma unroll
    for (int k = 0; k < 3; ++k) sc[k] = (k < tail) ? xs[k] : VIBECUDA_NEG_INF;
  }
  float m = VIBECUDA_NEG_INF;
#pragma unroll
  for (int j = 0; j < LV; ++j) m = fmaxf(m, f4_max(v[j]));
  if (t == 0) {
#pragma unroll
    for (int k = 0; k < 3; ++k) m = (k < tail) ? fmaxf(m, sc[k]) : m;
  }

  // ---- phase 2: warp-local online softmax ---------------------------------
  const float m_w = warp_max(m);
  float s = 0.f;
  if constexpr (PK2) {
    // packed exp-argument prep: 2 f32x2 FMA instead of 4 scalar FMA per
    // float4; inv_t folds into the loop-invariant vector constants
    const float l2e_t = 1.4426950408889634f * inv_t;
    const unsigned long long r2_l2e = pack2(l2e_t);
    const unsigned long long r2_negm = pack2(-m_w * l2e_t);
#pragma unroll
    for (int j = 0; j < LV; ++j) {
      v[j] = f4_exp_pack(v[j], r2_l2e, r2_negm);
      s += f4_sum(v[j]);
    }
  } else {
#pragma unroll
    for (int j = 0; j < LV; ++j) {
      v[j] = f4_exp(v[j], m_w, inv_t);
      s += f4_sum(v[j]);
    }
  }
  if (t == 0) {
    const float nmwt = -m_w * inv_t;
#pragma unroll
    for (int k = 0; k < 3; ++k) sc[k] = __expf(fmaf(sc[k], inv_t, nmwt));
  }
  // threads with no live elements: s may be NaN (exp(-inf - -inf))
  s = warp_sum((m == VIBECUDA_NEG_INF) ? 0.f : s + (t == 0 ? sc[0] + sc[1] + sc[2] : 0.f));

  // DSM mirror: lane r of each warp deposits this warp's pair into CTA r.
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (lane < R) {
    dsm_post_pair(pool_ms, crank * NW + warp, (unsigned)lane, m_w, s);
  }

  // one cluster-wide rendezvous makes every posted pair visible everywhere
  cluster.sync();

  // row (M, S) from the cluster pair pool (leader-warp merge + broadcast)
  __shared__ float2 bc_ms;
  if (warp == 0) {
    const float2 pw = pool_merge_warp<P, K>(pool_ms, lane, inv_t);
    if (lane == 0) bc_ms = pw;
  }
  __syncthreads();
  const float2 prow = bc_ms;
  const float M_g = prow.x;
  const float S_g = prow.y;

  // ---- phase 3: normalize + store ------------------------------------------
  // registers hold exp((x_j - m_w) * inv_t); the rescale folds into the
  // store scale: y_j = exp((x_j - m_w)inv_t) * exp((m_w - M)inv_t) / S
  const float scale = __expf((m_w - M_g) * inv_t) / S_g;
#pragma unroll
  for (int j = 0; j < LV; ++j) {
    int idx = t + j * T;
    if (idx < V) st4_out(yv + idx, f4_scale(v[j], scale));
  }
  if (t == 0) {
    float* ys = yr + (V << 2);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (k < tail) ys[k] = sc[k] * scale;
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

// ---------------------------------------------------------------------------
// Cross-row soft-pipelined register cluster softmax.  Same per-row math as
// softmax_cluster_kernel, but each cluster loops over rows with stride =
// launched clusters.  Row i's DSM rendezvous is split into barrier_arrive /
// barrier_wait; in between, the CTA issues a cp.async prefetch of row
// i+stride into its private SMEM slab, so the reduce/merge window carries
// DRAM traffic instead of idling.  Pair pools alternate by iteration parity:
// iteration k+1 posts to pool[(k+1)&1] in every cluster CTA simultaneously,
// while leader warps may still be merging pool[k&1] -- remote posts for
// iteration k+2 reach pool[k&1] only after this CTA's arrive(k+1), which is
// sequenced after its merge(k) completes, so two pools suffice.  Per-thread
// slab slots are private (same tid writes and reads), so a single
// per-thread cp.async.wait_group orders them; no __syncthreads on that path.
// ---------------------------------------------------------------------------
template <int R, int BT, int LV, bool PK2 = false>
__global__ void __launch_bounds__(BT, 1024 / BT) __cluster_dims__(R, 1, 1)
    softmax_cluster_xr_kernel(const float* __restrict__ x, float* __restrict__ y, int n, int batch,
                              const float* __restrict__ t_arr, float t_val) {
  static_assert(LV >= 1 && LV <= 8, "float4 registers per thread");
  cg::cluster_group cluster = cg::this_cluster();
  const int crank = (int)cluster.block_rank();
  const int gstride = (int)(gridDim.x / R);  // clusters launched (row stride)

  const int T = R * BT;
  const int t = crank * BT + (int)threadIdx.x;
  const int V = n >> 2;    // number of full float4 vectors per row
  const int tail = n & 3;  // scalar leftovers at row end

  constexpr int NW = BT / 32;
  constexpr int P = R * NW;                             // cluster pair pool size (<= 256)
  constexpr int K = (P + 31) / 32;                      // strided pool polls per lane
  __shared__ float2 pool_ms[2][P + VIBECUDA_LW_EXTRA];  // parity pools + broadcast
  extern __shared__ float4 smem4[];                     // prefetch slab: BT*LV float4

  cudaGridDependencySynchronize();

  // per-thread slab slots inside the prefetch region: (tid + j*BT)
  const unsigned slot0 =
      (unsigned)__cvta_generic_to_shared(smem4) + (unsigned)((int)threadIdx.x * 16);

  int crow = (int)(blockIdx.x / R);  // linear cluster slot
  int row = crow;

  // iteration 0: direct global loads into registers (registers are free)
  float4 v[LV];
  {
    const float4* xv = reinterpret_cast<const float4*>(x + (long long)row * n);
#pragma unroll
    for (int j = 0; j < LV; ++j) {
      const int idx = t + j * T;
      v[j] = (idx < V) ? __ldcs(xv + idx)
                       : make_float4(VIBECUDA_NEG_INF, VIBECUDA_NEG_INF, VIBECUDA_NEG_INF,
                                     VIBECUDA_NEG_INF);
    }
  }

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int parity = 0;
  for (; crow < batch; crow += gstride, parity ^= 1) {
    const bool has_next = (crow + gstride < batch);
    row = crow;
    const float* __restrict__ xr = x + (long long)row * n;
    float* __restrict__ yr = y + (long long)row * n;
    float4* yv = reinterpret_cast<float4*>(yr);
    float2* pool = pool_ms[parity];

    const float t_row = (t_arr == nullptr) ? t_val : t_arr[row];
    const float inv_t = (t_row == 0.f) ? 0.f : 1.0f / t_row;

    // ---- phase 1+2: warp-local online softmax from registers ---------------
    float sc[3];
    if (t == 0) {
      const float* xs = xr + (V << 2);
#pragma unroll
      for (int k = 0; k < 3; ++k) sc[k] = (k < tail) ? __ldg(xs + k) : VIBECUDA_NEG_INF;
    }
    float m = VIBECUDA_NEG_INF;
#pragma unroll
    for (int j = 0; j < LV; ++j) m = fmaxf(m, f4_max(v[j]));
    if (t == 0) {
#pragma unroll
      for (int k = 0; k < 3; ++k) m = (k < tail) ? fmaxf(m, sc[k]) : m;
    }
    const float m_w = warp_max(m);
    float s = 0.f;
    if constexpr (PK2) {
      const float l2e_t = 1.4426950408889634f * inv_t;
      const unsigned long long r2_l2e = pack2(l2e_t);
      const unsigned long long r2_negm = pack2(-m_w * l2e_t);
#pragma unroll
      for (int j = 0; j < LV; ++j) {
        v[j] = f4_exp_pack(v[j], r2_l2e, r2_negm);
        s += f4_sum(v[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < LV; ++j) {
        v[j] = f4_exp(v[j], m_w, inv_t);
        s += f4_sum(v[j]);
      }
    }
    if (t == 0) {
      const float nmwt = -m_w * inv_t;
#pragma unroll
      for (int k = 0; k < 3; ++k) sc[k] = __expf(fmaf(sc[k], inv_t, nmwt));
    }
    // threads with no live elements: s may be NaN (exp(-inf - -inf))
    s = warp_sum((m == VIBECUDA_NEG_INF) ? 0.f : s + (t == 0 ? sc[0] + sc[1] + sc[2] : 0.f));

    // DSM mirror: lane r of each warp deposits this warp's pair into CTA r.
    if (lane < R) {
      dsm_post_pair(pool, crank * NW + warp, (unsigned)lane, m_w, s);
    }

    // ---- split cluster rendezvous: prefetch row i+stride while waiting -----
    cluster.barrier_arrive();

    if (has_next) {
      const int nrow = crow + gstride;
      const float4* xn = reinterpret_cast<const float4*>(x + (long long)nrow * n);
#pragma unroll
      for (int j = 0; j < LV; ++j) {
        const int idx = t + j * T;
        if (idx < V) {
          const unsigned a = slot0 + (unsigned)(j * (BT * 16));
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(xn + idx));
        }
      }
    }
    asm volatile("cp.async.commit_group;");

    cluster.barrier_wait();

    // row (M, S) from this iteration's pool (leader-warp merge + broadcast)
    if (warp == 0) {
      const float2 pw = pool_merge_warp<P, K>(pool, lane, inv_t);
      if (lane == 0) pool[P] = pw;
    }
    __syncthreads();
    const float2 prow = pool[P];
    const float M_g = prow.x;
    const float S_g = prow.y;

    // ---- phase 3: normalize + store current row from registers -------------
    const float scale = __expf((m_w - M_g) * inv_t) / S_g;
#pragma unroll
    for (int j = 0; j < LV; ++j) {
      const int idx = t + j * T;
      if (idx < V) st4_out(yv + idx, f4_scale(v[j], scale));
    }
    if (t == 0) {
      float* ys = yr + (V << 2);
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        if (k < tail) ys[k] = sc[k] * scale;
      }
    }

    // ---- next row: retire prefetch, stage slab into registers --------------
    if (has_next) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      const float4* stg = smem4 + (int)threadIdx.x;
#pragma unroll
      for (int j = 0; j < LV; ++j) {
        const int idx = t + j * T;
        v[j] = (idx < V) ? stg[j * BT]
                         : make_float4(VIBECUDA_NEG_INF, VIBECUDA_NEG_INF, VIBECUDA_NEG_INF,
                                       VIBECUDA_NEG_INF);
      }
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

// ---------------------------------------------------------------------------
// cp.async SMEM-streaming cluster softmax for wide rows.
//
// R CTAs (thread cluster) per row; CTA `crank` owns the contiguous float4
// slab [crank*SLAB4, min(V4, (crank+1)*SLAB4)).  The slab streams through a
// ring of NST chunk buffers in shared memory via LDGSTS (cp.async.cg), so
// deep DRAM pipelining lives in the copy engine, not the register file.
// Each thread copies and later reads its own strided float4 slots, so the
// phase-1 (online max/sum) loop needs no cross-thread visibility.
//
// pass 1: for every landed chunk, fold its elements into a per-thread
// online (max, sum) accumulator (chunk-local max keeps it at 1 MUFU/elt).
// pass 2 (after the cluster-wide pair merge): reread the slab via
// vectorized global loads -- still L2-resident -- and write
// exp((x - M)inv_t) / S.
// ---------------------------------------------------------------------------
template <int R, bool PK2P = true>
__global__ void __launch_bounds__(512, 4) __cluster_dims__(R, 1, 1)
    softmax_pipe_kernel(const float* __restrict__ x, float* __restrict__ y, int n,
                        const float* __restrict__ t_arr, float t_val) {
  constexpr int BT = 512;
  constexpr int CH4 = 1024;  // float4 per chunk (16 KB)
  constexpr int NST = 3;     // ring stages
  constexpr int NW = BT / 32;
  constexpr int P = R * NW;  // pair pool (<= 256)
  constexpr int K = (P + 31) / 32;
  extern __shared__ float4 smem4[];
  // AoS float2 pair pool.  NOTE: the pool merge reads pay ~2x ideal
  // wavefronts on SM100 in EVERY layout tried -- AoS LDS.64 (4.03), SoA
  // scalar LDS.32 (2.18, also +2% latency), AoS vectorized LDS.128 (8.06) --
  // while identical-width ring reads run at ideal, so the recorded bank
  // conflicts are intrinsic to the 16-warp redundant pool read, not the
  // element layout.  Layout variants measured and rejected; do not retry.
  float2* pool_ms = reinterpret_cast<float2*>(smem4 + NST * CH4);

  cg::cluster_group cluster = cg::this_cluster();
  const int row = blockIdx.x / R;
  const int crank = (int)cluster.block_rank();
  const float* __restrict__ xr = x + (long long)row * n;
  float* __restrict__ yr = y + (long long)row * n;

  const float t_row = (t_arr == nullptr) ? t_val : t_arr[row];
  const float inv_t = (t_row == 0.f) ? 0.f : 1.0f / t_row;

  const int V4 = n >> 2;
  const int tail = n & 3;
  const int SLAB4 = (V4 + R - 1) / R;
  const int v0 = crank * SLAB4;
  const int v1 = min(V4, v0 + SLAB4);
  const int n4 = v1 - v0;  // may be <= 0 for the last CTA
  const int NC = n4 > 0 ? (n4 + CH4 - 1) / CH4 : 0;

  const float4* __restrict__ src = reinterpret_cast<const float4*>(xr) + v0;
  float4* __restrict__ dst = reinterpret_cast<float4*>(yr) + v0;
  const int tid = threadIdx.x;

  // PDL: slab math above overlaps the previous grid's tail; global accesses
  // below wait for full visibility of the previous grid's stores.
  cudaGridDependencySynchronize();

  // per-thread SMEM slots inside a stage: (tid + j*BT), j in {0,1}
  unsigned slot0;
  {
    const unsigned base = (unsigned)__cvta_generic_to_shared(smem4);
    slot0 = base + (unsigned)(tid * 16);
  }

  // issue chunk c into stage (c % NST): 2 float4 copies per thread
  auto issue_chunk = [&](int c) {
    const unsigned sbase = slot0 + (unsigned)((c % NST) * (CH4 * 16));
    const float4* g = src + (long long)c * CH4 + tid;
#pragma unroll
    for (int j = 0; j < CH4 / BT; ++j) {
      if (c * CH4 + tid + j * BT < n4) {
        const unsigned a = sbase + (unsigned)(j * (BT * 16));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(g + j * BT));
      }
    }
    asm volatile("cp.async.commit_group;");
  };

#pragma unroll
  for (int st = 0; st < NST; ++st) {
    if (st < NC)
      issue_chunk(st);
    else
      asm volatile("cp.async.commit_group;");
  }

  // ---- phase 1: stream chunks through SMEM, online (max, sum) -------------
  // NOTE: no __syncthreads here -- each thread copies into and reads back
  // only its own strided SMEM slots, so per-thread cp.async.wait_group
  // ordering is sufficient (NCU: barrier stalls were 21% of warp time;
  // removing them bought +3-5% on every pipe shape).
  float m = VIBECUDA_NEG_INF, s = 0.f;
  const float l2e_t = 1.4426950408889634f * inv_t;
  const unsigned long long p1_l2e = PK2P ? pack2(l2e_t) : 0ull;
  for (int c = 0; c < NC; ++c) {
    asm volatile("cp.async.wait_group %0;" ::"n"(NST - 1));
    const float4* stg = smem4 + (c % NST) * CH4 + tid;
    float4 a, b;
    float tm = VIBECUDA_NEG_INF;
    if (c * CH4 + tid < n4) {
      a = stg[0];
      tm = f4_max(a);
      if (c * CH4 + tid + BT < n4) {
        b = stg[BT];
        tm = fmaxf(tm, f4_max(b));
        const float mn = fmaxf(m, tm);
        // all -inf so far (m and this chunk): (-inf) - (-inf) is NaN; the
        // chunk contributes nothing and the accumulator stays at (-inf, s*0).
        if (mn != VIBECUDA_NEG_INF) {
          float csc = __expf((m - mn) * inv_t);
          if (m == VIBECUDA_NEG_INF) csc = 0.f;
          float4 ea, eb;
          if constexpr (PK2P) {
            // packed exp-argument prep: the vector constant is loop-invariant,
            // the -mn*log2e*inv_t pair is hoisted once per chunk; 2 f32x2 FMA
            // per float4 instead of 4 scalar FADD + 4 scalar FMUL.
            const unsigned long long p1_negmn = pack2(-mn * l2e_t);
            ea = f4_exp_pack(a, p1_l2e, p1_negmn);
            eb = f4_exp_pack(b, p1_l2e, p1_negmn);
          } else {
            ea = f4_exp(a, mn, inv_t);
            eb = f4_exp(b, mn, inv_t);
          }
          s = s * csc + (f4_sum(ea) + f4_sum(eb));
        } else {
          s = 0.f;
        }
        m = mn;
      } else {
        online_add(a.x, m, s, inv_t);
        online_add(a.y, m, s, inv_t);
        online_add(a.z, m, s, inv_t);
        online_add(a.w, m, s, inv_t);
      }
    }
    if (c + NST < NC)
      issue_chunk(c + NST);
    else
      asm volatile("cp.async.commit_group;");
  }

  // scalar row tail folded into the last CTA's thread 0
  if (crank == R - 1 && tid == 0) {
    const float* xs = xr + (V4 << 2);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (k < tail) online_add(__ldg(xs + k), m, s, inv_t);
    }
  }

  // pass-2 re-read prefetch: stream the slab through the ring again (same
  // per-thread slots, same zero-cross-thread sync discipline); the L2-hot
  // re-read latency overlaps the pair merge + cluster.sync window below.
  // (A direct __ldg pass-2 re-read was benchmarked for the one-wave band:
  // it re-exposes the L2 latency after the merge and loses 2-6%.)
#pragma unroll
  for (int st = 0; st < NST; ++st) {
    if (st < NC)
      issue_chunk(st);
    else
      asm volatile("cp.async.commit_group;");
  }

  // ---- warp pair -> DSM pool -> cluster.sync -> row (M, S) -----------------
  float2 p = make_float2(m, s);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float2 q;
    q.x = __shfl_xor_sync(VIBECUDA_FULL_MASK, p.x, o);
    q.y = __shfl_xor_sync(VIBECUDA_FULL_MASK, p.y, o);
    p = pair_merge(p, q, inv_t);
  }
  const int warp = tid >> 5, lane = tid & 31;
  if (lane < R) {
    dsm_post_pair(pool_ms, crank * NW + warp, (unsigned)lane, p.x, p.y);
  }
  cluster.sync();

  float2* bc_ms = pool_ms + P;  // leader-warp broadcast slot
  if (warp == 0) {
    const float2 pw = pool_merge_warp<P, K>(pool_ms, lane, inv_t);
    if (lane == 0) *bc_ms = pw;
  }
  __syncthreads();
  const float2 prow = *bc_ms;
  const float M_g = prow.x;
  const float invS = 1.0f / prow.y;
  const unsigned long long q_l2e = pack2(l2e_t);
  const unsigned long long q_negm = pack2(-M_g * l2e_t);
  const unsigned long long q_invs = pack2(invS);

  // ---- phase 2: consume the re-read ring + normalize + store ---------------
  for (int c = 0; c < NC; ++c) {
    asm volatile("cp.async.wait_group %0;" ::"n"(NST - 1));
    const float4* stg = smem4 + (c % NST) * CH4 + tid;
    const int i0 = c * CH4 + tid;
#pragma unroll
    for (int j = 0; j < CH4 / BT; ++j) {
      const int idx = i0 + j * BT;
      if (idx < n4) {
        const float4 v = stg[j * BT];
        __stcs(dst + idx, f4_expn_pack(v, q_l2e, q_negm, q_invs));
      }
    }
    if (c + NST < NC)
      issue_chunk(c + NST);
    else
      asm volatile("cp.async.commit_group;");
  }
  if (crank == R - 1 && tid == 0) {
    float* ys = yr + (V4 << 2);
    const float* xs = xr + (V4 << 2);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (k < tail) ys[k] = __expf((__ldg(xs + k) - M_g) * inv_t) * invS;
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

// ---------------------------------------------------------------------------
// Full-slab SMEM staging cluster softmax for wide rows.  Each CTA's
// contiguous slab (<= 4096 float4 = 64 KiB; the 128K/R=8 and 256K/R=16
// bands both land at ~4000) is cp.async'd into shared memory ONCE, up front
// -- NC <= 4 commit groups of 16 KB, i.e. ~192 KB of copies in flight per
// SM at 3 CTAs/SM, the same DRAM depth as the 4 x 3-stage pipe ring.
// Phase 1 folds (max, sum) out of SMEM as chunks land (thread-local slots
// only, so per-thread cp.async.wait_group ordering suffices -- no
// __syncthreads); after the pair rendezvous phase 2 normalizes directly
// from SMEM: zero pass-2 global/L2 reread.
// ---------------------------------------------------------------------------
template <int R, bool PK2P = true>
__global__ void __launch_bounds__(512, 3) __cluster_dims__(R, 1, 1)
    softmax_slab_kernel(const float* __restrict__ x, float* __restrict__ y, int n,
                        const float* __restrict__ t_arr, float t_val) {
  constexpr int BT = 512;
  constexpr int CH4 = 1024;        // float4 per commit group (16 KB)
  constexpr int SLAB_MAX4 = 4096;  // 64 KiB staging cap
  constexpr int NW = BT / 32;
  constexpr int P = R * NW;  // pair pool (<= 256)
  constexpr int K = (P + 31) / 32;
  extern __shared__ float4 smem4[];
  float2* pool_ms = reinterpret_cast<float2*>(smem4 + SLAB_MAX4);

  cg::cluster_group cluster = cg::this_cluster();
  const int row = blockIdx.x / R;
  const int crank = (int)cluster.block_rank();
  const float* __restrict__ xr = x + (long long)row * n;
  float* __restrict__ yr = y + (long long)row * n;

  const float t_row = (t_arr == nullptr) ? t_val : t_arr[row];
  const float inv_t = (t_row == 0.f) ? 0.f : 1.0f / t_row;

  const int V4 = n >> 2;
  const int tail = n & 3;
  const int SLAB4 = (V4 + R - 1) / R;
  const int v0 = crank * SLAB4;
  const int v1 = min(V4, v0 + SLAB4);
  const int n4 = v1 - v0;                            // may be <= 0 for the last CTA
  const int NC = n4 > 0 ? (n4 + CH4 - 1) / CH4 : 0;  // <= 4 by dispatch gate

  const float4* __restrict__ src = reinterpret_cast<const float4*>(xr) + v0;
  float4* __restrict__ dst = reinterpret_cast<float4*>(yr) + v0;
  const int tid = threadIdx.x;

  cudaGridDependencySynchronize();

  unsigned slot0;
  {
    const unsigned base = (unsigned)__cvta_generic_to_shared(smem4);
    slot0 = base + (unsigned)(tid * 16);
  }

  // issue chunk c into slab slot c: 2 float4 copies per thread, one group
  auto issue_chunk = [&](int c) {
    const unsigned sbase = slot0 + (unsigned)(c * (CH4 * 16));
    const float4* g = src + (long long)c * CH4 + tid;
#pragma unroll
    for (int j = 0; j < CH4 / BT; ++j) {
      if (c * CH4 + tid + j * BT < n4) {
        const unsigned a = sbase + (unsigned)(j * (BT * 16));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(g + j * BT));
      }
    }
    asm volatile("cp.async.commit_group;");
  };

  // issue the whole slab up front: NC commit groups (<= 4)
  for (int c = 0; c < NC; ++c) issue_chunk(c);

  // chunk c has landed once at most NC-1-c groups are still pending
  auto wait_chunk = [&](int c) {
    switch (NC - 1 - c) {
      case 0:
        asm volatile("cp.async.wait_group 0;");
        break;
      case 1:
        asm volatile("cp.async.wait_group 1;");
        break;
      case 2:
        asm volatile("cp.async.wait_group 2;");
        break;
      default:
        asm volatile("cp.async.wait_group 3;");
        break;
    }
  };

  // ---- phase 1: fold (max, sum) as chunks land -----------------------------
  float m = VIBECUDA_NEG_INF, s = 0.f;
  const float l2e_t = 1.4426950408889634f * inv_t;
  const unsigned long long p1_l2e = PK2P ? pack2(l2e_t) : 0ull;
  for (int c = 0; c < NC; ++c) {
    wait_chunk(c);
    const float4* stg = smem4 + c * CH4 + tid;
    float4 a, b;
    float tm = VIBECUDA_NEG_INF;
    if (c * CH4 + tid < n4) {
      a = stg[0];
      tm = f4_max(a);
      if (c * CH4 + tid + BT < n4) {
        b = stg[BT];
        tm = fmaxf(tm, f4_max(b));
        const float mn = fmaxf(m, tm);
        // all -inf so far (m and this chunk): (-inf) - (-inf) is NaN; the
        // chunk contributes nothing and the accumulator stays at (-inf, s*0).
        if (mn != VIBECUDA_NEG_INF) {
          float csc = __expf((m - mn) * inv_t);
          if (m == VIBECUDA_NEG_INF) csc = 0.f;
          float4 ea, eb;
          if constexpr (PK2P) {
            const unsigned long long p1_negmn = pack2(-mn * l2e_t);
            ea = f4_exp_pack(a, p1_l2e, p1_negmn);
            eb = f4_exp_pack(b, p1_l2e, p1_negmn);
          } else {
            ea = f4_exp(a, mn, inv_t);
            eb = f4_exp(b, mn, inv_t);
          }
          s = s * csc + (f4_sum(ea) + f4_sum(eb));
        } else {
          s = 0.f;
        }
        m = mn;
      } else {
        online_add(a.x, m, s, inv_t);
        online_add(a.y, m, s, inv_t);
        online_add(a.z, m, s, inv_t);
        online_add(a.w, m, s, inv_t);
      }
    }
  }

  // scalar row tail folded into the last CTA's thread 0
  if (crank == R - 1 && tid == 0) {
    const float* xs = xr + (V4 << 2);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (k < tail) online_add(__ldg(xs + k), m, s, inv_t);
    }
  }

  // ---- warp pair -> DSM pool -> cluster.sync -> row (M, S) -----------------
  float2 p = make_float2(m, s);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    float2 q;
    q.x = __shfl_xor_sync(VIBECUDA_FULL_MASK, p.x, o);
    q.y = __shfl_xor_sync(VIBECUDA_FULL_MASK, p.y, o);
    p = pair_merge(p, q, inv_t);
  }
  const int warp = tid >> 5, lane = tid & 31;
  if (lane < R) {
    dsm_post_pair(pool_ms, crank * NW + warp, (unsigned)lane, p.x, p.y);
  }
  cluster.sync();

  float2* bc_ms = pool_ms + P;  // leader-warp broadcast slot
  if (warp == 0) {
    const float2 pw = pool_merge_warp<P, K>(pool_ms, lane, inv_t);
    if (lane == 0) *bc_ms = pw;
  }
  __syncthreads();
  const float2 prow = *bc_ms;
  const float M_g = prow.x;
  const float invS = 1.0f / prow.y;
  const unsigned long long q_l2e = pack2(l2e_t);
  const unsigned long long q_negm = pack2(-M_g * l2e_t);
  const unsigned long long q_invs = pack2(invS);

  // ---- phase 2: normalize + store straight from the staged slab ------------
  for (int c = 0; c < NC; ++c) {
    const float4* stg = smem4 + c * CH4 + tid;
    const int i0 = c * CH4 + tid;
#pragma unroll
    for (int j = 0; j < CH4 / BT; ++j) {
      const int idx = i0 + j * BT;
      if (idx < n4) {
        const float4 v = stg[j * BT];
        __stcs(dst + idx, f4_expn_pack(v, q_l2e, q_negm, q_invs));
      }
    }
  }
  if (crank == R - 1 && tid == 0) {
    float* ys = yr + (V4 << 2);
    const float* xs = xr + (V4 << 2);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
      if (k < tail) ys[k] = __expf((__ldg(xs + k) - M_g) * inv_t) * invS;
    }
  }
  cudaTriggerProgrammaticLaunchCompletion();
}

// ---------------------------------------------------------------------------
// Streaming fallback for oversized or unaligned rows: one CTA per row,
// block-strided online (max, sum) pass, then a re-read pass that writes
// exp((x - m)inv_t) / s.  Correct for any n; only used outside the fast
// bands.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(1024, 1)
    softmax_stream_kernel(const float* __restrict__ x, float* __restrict__ y, long long n,
                          const float* __restrict__ t_arr, float t_val) {
  const float* __restrict__ xr = x + (long long)blockIdx.x * n;
  float* __restrict__ yr = y + (long long)blockIdx.x * n;

  __shared__ float red[32];
  __shared__ float bc[1];
  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;

  const float t_row = (t_arr == nullptr) ? t_val : t_arr[blockIdx.x];
  const float inv_t = (t_row == 0.f) ? 0.f : 1.0f / t_row;

  cudaGridDependencySynchronize();

  // pass 1: running max
  float m = VIBECUDA_NEG_INF;
  for (long long i = tid; i < n; i += 1024) m = fmaxf(m, __ldg(xr + i));
  {
    float wm = warp_max(m);
    if (lane == 0) red[warp] = wm;
    __syncthreads();
    if (warp == 0) {
      float w2 = (lane < 32) ? red[lane] : VIBECUDA_NEG_INF;
      w2 = warp_max(w2);
      if (lane == 0) bc[0] = w2;
    }
    __syncthreads();
  }
  const float M = bc[0];

  // pass 2: sum of exp((x - M) * inv_t)
  float s = 0.f;
  for (long long i = tid; i < n; i += 1024) s += __expf((__ldg(xr + i) - M) * inv_t);
  {
    float ws = warp_sum(s);
    if (lane == 0) red[warp] = ws;
    __syncthreads();
    if (warp == 0) {
      float w2 = (lane < 32) ? red[lane] : 0.f;
      w2 = warp_sum(w2);
      if (lane == 0) bc[0] = w2;
    }
    __syncthreads();
  }
  const float invS = 1.0f / bc[0];

  // pass 3: normalize + store
  for (long long i = tid; i < n; i += 1024) yr[i] = __expf((__ldg(xr + i) - M) * inv_t) * invS;
  cudaTriggerProgrammaticLaunchCompletion();
}

namespace detail {

// Packed-f32x2 exp-argument prep is the shipped configuration on both the
// register kernel's phase 2 and the pipe/slab kernel's phase-1 chunk loops
// (regression A/Bs on the full 40-shape suite; do not reopen without new
// measurements).
constexpr bool kPk2Reg = true;
constexpr bool kPk2Pipe = true;
constexpr bool kPk8 = true;
// Full-slab SMEM staging arm for the wide bands (band-scoped vs the pipe by
// grid width; gates below).
constexpr bool kSlab = true;
// Cross-row soft-pipelined register kernel on multi-wave grids.
constexpr bool kXR = true;

// dynamic SMEM sizes for the slab kernel: 64 KiB staging + pair pool
constexpr int SM_SLAB8 = 4096 * 16 + (8 * 16 + VIBECUDA_LW_EXTRA) * 8;
constexpr int SM_SLAB16 = 4096 * 16 + (16 * 16 + VIBECUDA_LW_EXTRA) * 8;

inline bool slab_smem_ready(int r) {
  cudaError_t e;
  if (r == 8) {
    e = cudaFuncSetAttribute((const void*)softmax_slab_kernel<8, kPk8>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SLAB8);
  } else {
    e = cudaFuncSetAttribute((const void*)softmax_slab_kernel<16, kPk2Pipe>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SLAB16);
    if (e == cudaSuccess) {
      e = cudaFuncSetAttribute((const void*)softmax_slab_kernel<16, kPk2Pipe>,
                               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    }
  }
  return e == cudaSuccess;
}

// dynamic prefetch-slab SMEM sizes for the cross-row register kernel
constexpr int SM_XR_128 = 128 * 8 * 16;  // BT=128, LV=8 -> 16 KiB
constexpr int SM_XR_256 = 256 * 8 * 16;  // BT=256, LV=8 -> 32 KiB
constexpr int SM_XR_512 = 512 * 8 * 16;  // BT=512, LV=8 -> 64 KiB

inline int device_sms() {
  static const int nsm = [] {
    int n = 0, dev = 0;
    cudaGetDevice(&dev);
    if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0) {
      return 0;
    }
    return n;
  }();
  return nsm;
}

// resident CTA capacity of one XR instantiation from the occupancy API;
// computed once per process.  Returns 0 on query failure (XR disengaged).
template <typename K>
inline long long xr_ctas_resident(K kernel, int bt, int smem) {
  int per_sm = 0;
  if (smem > 48 * 1024) {
    if (cudaFuncSetAttribute((const void*)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem) != cudaSuccess) {
      return 0;
    }
  }
  const int nsm = device_sms();
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, bt, smem) != cudaSuccess ||
      per_sm <= 0 || nsm <= 0) {
    return (long long)0;
  }
  return (long long)per_sm * nsm;
}

// resident CTA capacity for the pipe geometry (CTAs/SM from the occupancy
// API times SM count); used for wave-fit dispatch, computed once per process
inline long long pipe_ctas_resident() {
  static const long long resident = [] {
    constexpr int SM_PIPE16_BYTES = 3 * 1024 * 16 + (256 + VIBECUDA_LW_EXTRA) * 8;
    int per_sm = 0, nsm = 0, dev = 0;
    cudaGetDevice(&dev);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, (const void*)softmax_pipe_kernel<16, kPk2Pipe>, 512, SM_PIPE16_BYTES) !=
            cudaSuccess ||
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        per_sm <= 0 || nsm <= 0) {
      return (long long)0;
    }
    return (long long)per_sm * nsm;
  }();
  return resident;
}

inline bool pipe_smem_ready(int r) {
  constexpr int SM_BYTES_8 = 3 * 1024 * 16 + (8 * 16 + VIBECUDA_LW_EXTRA) * 8;
  constexpr int SM_BYTES_16 = 3 * 1024 * 16 + (16 * 16 + VIBECUDA_LW_EXTRA) * 8;
  cudaError_t e;
  if (r == 8) {
    e = cudaFuncSetAttribute((const void*)softmax_pipe_kernel<8, kPk8>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SM_BYTES_8);
  } else {
    e = cudaFuncSetAttribute((const void*)softmax_pipe_kernel<16, kPk2Pipe>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, SM_BYTES_16);
    if (e == cudaSuccess) {
      e = cudaFuncSetAttribute((const void*)softmax_pipe_kernel<16, kPk2Pipe>,
                               cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    }
  }
  return e == cudaSuccess;
}

// Launch helper: when enable_pdl is set, attaches the
// ProgrammaticStreamSerialization attribute so the next back-to-back call
// can run its sync-gated prologue while the previous grid drains, hiding
// grid-launch and schedule setup latency.  The in-kernel
// cudaGridDependencySynchronize / cudaTriggerProgrammaticLaunchCompletion
// calls are no-ops without the attribute, so the same device code serves
// both modes.
template <typename K, typename... Args>
inline cudaError_t launch_softmax(K kernel, dim3 grid, dim3 block, unsigned smem, bool enable_pdl,
                                  cudaStream_t stream, Args... args) {
  if (enable_pdl) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    return cudaLaunchKernelEx(&cfg, kernel, args...);
  }
  void* arg_ptrs[] = {(void*)&args...};
  return cudaLaunchKernel((const void*)kernel, grid, block, arg_ptrs, smem, stream);
}

}  // namespace detail

// VibeCUDA fused row-wise softmax host entry point.
//
// out[i, :] = softmax(logits[i, :] / t_i) with t_i = temperature_arr[i] when
// temperature_arr != nullptr, else t_i = temperature_val.  logits/output are
// contiguous FP32 [batch_size, d] tensors on the current device.  Requires
// SM90+ for the fallback path and SM100 for the tuned cluster paths; callers
// must gate the architecture (the csrc op does) so selecting this backend on
// an unsupported architecture fails loudly rather than silently degrading.
inline cudaError_t Softmax(const float* logits, float* output, uint32_t batch_size, uint32_t d,
                           const float* temperature_arr, float temperature_val, bool enable_pdl,
                           cudaStream_t stream = 0) {
  if (batch_size == 0 || d == 0) return cudaSuccess;

  const float* xp = logits;
  float* yp = output;
  const long long n = d;
  const long long batch = batch_size;

  const bool aligned = ((n & 3) == 0) && ((reinterpret_cast<uintptr_t>(xp) & 15) == 0) &&
                       ((reinterpret_cast<uintptr_t>(yp) & 15) == 0);

  if (aligned && n <= 262144) {
    // one-time capability probes
    static const bool c16_reg_ok = [] {
      const bool wide =
          cudaFuncSetAttribute((const void*)softmax_cluster_kernel<16, 512, 8>,
                               cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess;
      const bool narrow =
          cudaFuncSetAttribute((const void*)softmax_cluster_kernel<16, 256, 8, detail::kPk2Reg>,
                               cudaFuncAttributeNonPortableClusterSizeAllowed, 1) == cudaSuccess;
      return wide && narrow;
    }();
    static const bool pipe8_ok = detail::pipe_smem_ready(8);
    static const bool pipe16_ok = detail::pipe_smem_ready(16);
    static const bool slab8_ok = !detail::kSlab || detail::slab_smem_ready(8);
    static const bool slab16_ok = !detail::kSlab || detail::slab_smem_ready(16);
    constexpr int SM_PIPE8 = 3 * 1024 * 16 + (128 + VIBECUDA_LW_EXTRA) * 8;
    constexpr int SM_PIPE16 = 3 * 1024 * 16 + (256 + VIBECUDA_LW_EXTRA) * 8;

    const long long V = n >> 2;  // float4 count in a row
    cudaError_t status = cudaSuccess;
    if (V <= 8192) {
      // 32K-class rows: cluster of 8 x 128-thr CTAs per row, 8 vec4 registers
      // per thread; eight independent CTAs per SM interleave memory phases.
      // (Measured: single-CTA-per-row 1024-thr and cluster<9,128,8> variants
      // tie only isolated batches and lose everywhere else; warp-specialized,
      // multi-row, and dual-bank rendezvous-amortization variants all lose at
      // occupancy parity -- the band is DRAM-critical-path bound.  Closed.)
      // cross-row arm engages once the row grid exceeds one resident wave
      // (occupancy-queried): below that there is no next row to pipeline and
      // shrinking the grid only lengthens the per-row critical path.
      static const long long xr_res32 =
          !detail::kXR ? 0
                       : detail::xr_ctas_resident(
                             (const void*)softmax_cluster_xr_kernel<8, 128, 8, detail::kPk2Reg>,
                             128, detail::SM_XR_128);
      if (detail::kXR && batch * 8 > xr_res32 && xr_res32 > 0) {
        const long long rescl = xr_res32 / 8;
        const long long gridrows = std::min(batch, std::max(rescl, (batch + 1) / 2));
        status = detail::launch_softmax(softmax_cluster_xr_kernel<8, 128, 8, detail::kPk2Reg>,
                                        dim3((unsigned)(gridrows * 8)), dim3(128),
                                        detail::SM_XR_128, enable_pdl, stream, xp, yp, (int)n,
                                        (int)batch, temperature_arr, temperature_val);
      } else {
        status = detail::launch_softmax(softmax_cluster_kernel<8, 128, 8, detail::kPk2Reg>,
                                        dim3((unsigned)(batch * 8)), dim3(128), 0, enable_pdl,
                                        stream, xp, yp, (int)n, temperature_arr, temperature_val);
      }
    } else if (V <= 16384) {
      // 64K-class rows: register-resident, cluster of 8 x 256-thr CTAs.
      // Narrower CTAs keep four resident per SM at 64 regs and beat the
      // 4 x 512-thr decomposition by 1.3-4.5% across the whole band.
      // (Measured: two-CTA 1024-thr cluster<2,1024,8> collapses occupancy to
      // 1 CTA/SM and loses 7-22% at both thin and wide batches; big-CTA
      // register-file collapse family closed.  A cp.async pipe for this band
      // also loses: the register path already interleaves 8 CTAs/SM.)
      static const long long xr_res64 =
          !detail::kXR ? 0
                       : detail::xr_ctas_resident(
                             (const void*)softmax_cluster_xr_kernel<8, 256, 8, detail::kPk2Reg>,
                             256, detail::SM_XR_256);
      if (detail::kXR && batch * 8 > xr_res64 && xr_res64 > 0) {
        const long long rescl = xr_res64 / 8;
        const long long gridrows = std::min(batch, std::max(rescl, (batch + 1) / 2));
        status = detail::launch_softmax(softmax_cluster_xr_kernel<8, 256, 8, detail::kPk2Reg>,
                                        dim3((unsigned)(gridrows * 8)), dim3(256),
                                        detail::SM_XR_256, enable_pdl, stream, xp, yp, (int)n,
                                        (int)batch, temperature_arr, temperature_val);
      } else {
        status = detail::launch_softmax(softmax_cluster_kernel<8, 256, 8, detail::kPk2Reg>,
                                        dim3((unsigned)(batch * 8)), dim3(256), 0, enable_pdl,
                                        stream, xp, yp, (int)n, temperature_arr, temperature_val);
      }
    } else if (V <= 32768) {
      // 128K-class rows: cp.async SMEM pipeline, cluster of 8 x 512-thr CTAs.
      // The L2-reread pipeline pays off once the grid is wide enough to
      // overlap phases (~2+ waves of resident CTAs); below that the register
      // path's single global read wins.  Measured: extending the pipe to
      // batch 32-128 ties at 64/128 rows and loses 9% at 32 rows, so the
      // gate stays at 1280.  Slab staging wins in the thin/mid-grid classes
      // (batch*8 in 1280..4096, +1.7-4.5% vs the pipe) where the zero pass-2
      // reread shortens the critical path; beyond 4096 the slab's 3-CTA/SM
      // occupancy loses DRAM depth to the 4-CTA/SM pipe.
      if (detail::kSlab && slab8_ok && batch * 8 >= 1280 && batch * 8 <= 4096) {
        status = detail::launch_softmax(
            softmax_slab_kernel<8, detail::kPk8>, dim3((unsigned)(batch * 8)), dim3(512),
            detail::SM_SLAB8, enable_pdl, stream, xp, yp, (int)n, temperature_arr, temperature_val);
      } else if (pipe8_ok && batch * 8 >= 1280) {
        status = detail::launch_softmax(
            softmax_pipe_kernel<8, detail::kPk8>, dim3((unsigned)(batch * 8)), dim3(512), SM_PIPE8,
            enable_pdl, stream, xp, yp, (int)n, temperature_arr, temperature_val);
      } else if (c16_reg_ok && batch <= 16) {
        // 128K-class rows on thin grids: narrow cluster-16 CTAs engage more
        // SMs per row (measured -13% at 16 rows, tie at <= 8 rows, loss at
        // >= 32 rows, so keep the 8 x 512-thr path there).
        status = detail::launch_softmax(softmax_cluster_kernel<16, 256, 8, detail::kPk2Reg>,
                                        dim3((unsigned)(batch * 16)), dim3(256), 0, enable_pdl,
                                        stream, xp, yp, (int)n, temperature_arr, temperature_val);
      } else {
        static const long long xr_res128 =
            !detail::kXR
                ? 0
                : detail::xr_ctas_resident((const void*)softmax_cluster_xr_kernel<8, 512, 8>, 512,
                                           detail::SM_XR_512);
        if (detail::kXR && batch * 8 > xr_res128 && xr_res128 > 0) {
          const long long rescl = xr_res128 / 8;
          const long long gridrows = std::min(batch, std::max(rescl, (batch + 1) / 2));
          status = detail::launch_softmax(softmax_cluster_xr_kernel<8, 512, 8>,
                                          dim3((unsigned)(gridrows * 8)), dim3(512),
                                          detail::SM_XR_512, enable_pdl, stream, xp, yp, (int)n,
                                          (int)batch, temperature_arr, temperature_val);
        } else {
          status = detail::launch_softmax(softmax_cluster_kernel<8, 512, 8>,
                                          dim3((unsigned)(batch * 8)), dim3(512), 0, enable_pdl,
                                          stream, xp, yp, (int)n, temperature_arr, temperature_val);
        }
      }
    } else if (!c16_reg_ok) {
      // cluster-16 unavailable: streaming fallback for 256K-class rows
      status =
          detail::launch_softmax(softmax_stream_kernel, dim3((unsigned)batch), dim3(1024), 0,
                                 enable_pdl, stream, xp, yp, n, temperature_arr, temperature_val);
    } else if (pipe16_ok && batch >= 16) {
      // 256K-class rows: cp.async SMEM pipeline once the grid carries at
      // least two CTAs of real streaming work per row-pair; below ~17 rows
      // the register path's single global read wins (measured ties at 4-8
      // rows, losses at 1 row).
      // NOTE: pipe<8> at this band measured WORSE at batch >= 128 (NCU: the
      // doubled per-CTA slab doubles the rows-in-flight reread working set,
      // partially evicting it from L2 -> DRAM refetch).  The one occupancy
      // class where it wins: when the 16-CTA grid spills a partial tail wave
      // (batch*16 > resident) while the 8-CTA grid fits in a single wave
      // (batch*8 <= resident) -- measured +7% in that class.
      const long long resident = detail::pipe_ctas_resident();
      dim3 block(512);
      // cluster-12 was benchmarked for the one-wave batch 16-48 class and
      // loses to cluster-16 (-8% at 32 rows, -21% at 48 rows: fewer CTAs per
      // row cuts SM-parallel DRAM streaming more than the narrower DSM
      // rendezvous saves), so this band stays on cluster-16/R8 wave-fit.
      if (pipe8_ok && batch * 8 <= resident && batch * 16 > resident) {
        // wave-fit pipe<8> class: an R=8 slab at 256K rows is 8000 float4
        // (125 KB) and would break the 64 KiB staging cap, so this class
        // always stays on the pipe kernel.
        status = detail::launch_softmax(softmax_pipe_kernel<8, detail::kPk8>,
                                        dim3((unsigned)(batch * 8)), block, SM_PIPE8, enable_pdl,
                                        stream, xp, yp, (int)n, temperature_arr, temperature_val);
      } else if (detail::kSlab && slab16_ok && batch <= 128) {
        // same occupancy crossover as the 128K band: slab wins batch <= 128
        // (+0.5-3.0%), pipe wins batch >= 256 (-0.8..-2.1%), both repeatable.
        status = detail::launch_softmax(softmax_slab_kernel<16, detail::kPk2Pipe>,
                                        dim3((unsigned)(batch * 16)), block, detail::SM_SLAB16,
                                        enable_pdl, stream, xp, yp, (int)n, temperature_arr,
                                        temperature_val);
      } else {
        status = detail::launch_softmax(softmax_pipe_kernel<16, detail::kPk2Pipe>,
                                        dim3((unsigned)(batch * 16)), block, SM_PIPE16, enable_pdl,
                                        stream, xp, yp, (int)n, temperature_arr, temperature_val);
      }
    } else {
      status = detail::launch_softmax(softmax_cluster_kernel<16, 512, 8>,
                                      dim3((unsigned)(batch * 16)), dim3(512), 0, enable_pdl,
                                      stream, xp, yp, (int)n, temperature_arr, temperature_val);
    }
    if (status != cudaSuccess) return status;
    return cudaGetLastError();
  }

  // fallback path: oversized or unaligned rows
  const cudaError_t status =
      detail::launch_softmax(softmax_stream_kernel, dim3((unsigned)batch), dim3(1024), 0,
                             enable_pdl, stream, xp, yp, n, temperature_arr, temperature_val);
  if (status != cudaSuccess) return status;
  return cudaGetLastError();
}

}  // namespace vibecuda
}  // namespace flashinfer

#endif  // FLASHINFER_VIBECUDA_SOFTMAX_CUH_
