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
#ifndef FLASHINFER_TOP_P_RENORM_SPECIALIZED_CUH_
#define FLASHINFER_TOP_P_RENORM_SPECIALIZED_CUH_

// Top-p renormalisation of float32 probability rows with one thread-block cluster (8 or 16 CTAs)
// per row: one read of the row and one write of the output, in a single launch.
//
// Semantics: keep {p >= threshold}, where threshold is the smallest value of the shortest
// descending prefix whose mass reaches top_p * M, M being the row's positive mass; ties at the
// threshold are kept; the kept entries are divided by their sum (left as they are if it is
// <= 1e-8), every other entry is 0; a NaN entry has no mass and is never kept (its output is 0).
// This is the rule of the AIR path of top_p_renorm_probs (its target is blockSum * p, relative to
// the row's own mass, so a row that does not sum to 1 keeps the same entries), evaluated with
// 64-bit fixed-point masses instead of AIR's float32 sums. Each entry is converted to whole units
// of 2^-q of the row's scale (2^-56 on the head paths, whose entries are all >= 2^-8; on the radix
// path q follows the row's mass, which is at least 2^58 units), every sum is an exact integer sum,
// and the target is top_p times the fixed-point mass rounded up to a whole unit. An entry of at
// least 2^-35 of the row's mass (2^23 units) converts exactly; a smaller one is rounded to the
// nearest unit, so the fixed-point masses lie within V/2 units of the exact ones: below 2^-41 of
// the row's mass for V <= 2^18, where float32 rounds at 2^-24. The kept set is therefore what
// exact arithmetic keeps unless the exact mass of a prefix lies within that bound of the target,
// and it never depends on a summation order. AIR's float32 sums round in an order that depends
// on its launch (float atomics across CTAs); where a prefix mass lies within that rounding of the
// target the two can keep different entries. Example: float32 [0.5, 0.3, 0.2] at top_p 0.5 has an
// exact mass of 1 + 2^-26, so the target is just above 0.5 and this kernel keeps {0.5, 0.3}
// (output [0.625, 0.375]) like a float64 evaluation; AIR's float32 row sum rounds to 1.0 and
// keeps 0.5 alone.
//
// Fast path: every CTA streams its slice of the row once, writes zeros to the matching output
// slice, adds the slice up in float32 and collects the entries >= 2^-8 (at most 256 per row when
// the row sums to 1; a row with more takes the radix path). The float32 row mass bounds M within
// its rounding error, and so bounds the target. If the head entries' mass reaches the target's
// upper bound, the nucleus lies entirely inside that set: the cluster gathers these head entries
// through distributed shared memory, resolves the exact threshold with 2^-56 fixed-point prefix
// sums (exact for every float >= 2^-8, so the kept set does not depend on the summation order)
// and each CTA scatters its kept entries. A head prefix sum inside the target's bounds (the kept
// set would depend on M's last bits) sends the row to the radix path. Serving rows are peaked
// (nucleus of 1-2 tokens at top_p 0.95), so this path serves almost every row; up to 32 head
// entries are resolved by one warp with shuffles, and a single head entry that reaches the target
// on its own exits right after the gather (its output is exactly 1).
// Otherwise a radix select over the float bits (12/11/9-bit levels, 64-bit fixed-point bin masses)
// reduced across the cluster through distributed shared memory finds the threshold, and a masked
// pass rewrites the slice. Its first level also sums the entries below the level's range, which
// gives the row's fixed-point M, and so its target, before any decision.
//
// Compared with the AIR radix top-p (air_top_p.cuh: five kernels, one 1024-thread CTA per row in
// the apply pass), this keeps the whole GPU busy for small batches (16 CTAs per row) and reads
// the row once on the fast path. Accumulation is 64-bit fixed point: 2^-56 on the head paths
// (exact for their entries, >= 2^-8), and 2^-q on the radix path, q chosen per row from its
// float32 mass so that the mass lies in [2^58, 2^61) units (a row of mass in [8, 64) keeps 2^-56):
// the resolution follows the row's own mass, whatever its scale, so a vanishing row (entries below
// 2^-57, down to subnormals) or a heavy one (mass 64 or more; the head paths are off for it) is
// selected like a probability row. For rows that sum to ~1 this is finer than float64 summation
// (at most V/2 units of 2^-59 in all, against one float64 rounding per addition). The one
// discontinuous decision on the mass (a kept mass <= 1e-8 is left unnormalised) is re-checked in
// float64 near its cutoff. No fast-math is used (the JIT module is built without
// -use_fast_math, so the renormalising division is the IEEE division).
//
// Requires sm_90+ (thread-block clusters, distributed shared memory).

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace flashinfer {
namespace sampling {
namespace top_p_renorm_specialized {

namespace cg = cooperative_groups;

constexpr int NBINS = 2048;
constexpr int LBITS = 0x37800000;  // bits of 2^-16: fallback level-0 filter
constexpr int CBITS = 0x3B800000;  // bits of 2^-8: fast-path head-entry cutoff
constexpr int CAP = 256;           // head entries per row (mass <= 1 => at most 256)
constexpr float FIX_SCALE = 72057594037927936.0f;  // 2^56
constexpr float BIG_MASS = 64.0f;                  // a row this heavy never takes the head paths

typedef unsigned long long u64;

// Fixed-point mass of p in units of 2^-q, 2^q = scale * scale2 (two factors, so that q can exceed
// a float's exponent range for a row of tiny entries; each product is exact); 0 for p <= 0 (the
// conversion saturates at 0). A NaN converts to 2^63, not 0: the callers drop NaN entries first
// (head_fix, hist_add, tail_fix).
__device__ __forceinline__ u64 to_fix(float p, float scale = FIX_SCALE, float scale2 = 1.f) {
  return __float2ull_rn(p * scale * scale2);
}

// 2^-56 fixed-point mass of a gathered head entry (p >= 2^-8 or NaN): 0 for a NaN, which has no
// mass and is never kept (x >= threshold is false for it).
__device__ __forceinline__ u64 head_fix(float p) {
  return (__float_as_uint(p) <= 0x7f800000u) ? to_fix(p) : 0ULL;
}

// 2^k as a float, for k in [-126, 127].
__device__ __forceinline__ float exp2i(int k) { return __int_as_float((k + 127) << 23); }

// max(1, ceil(top_p * mass)) in exact integer arithmetic: AIR's target blockSum * p for a
// fixed-point row mass (top_p >= 1 gives the whole mass), rounded up, so that a prefix (a whole
// number of units) reaches it exactly when it reaches top_p * mass. The minimum of one unit (top_p
// <= 0 or NaN) keeps the row maximum, as AIR does: the nucleus always holds at least one entry.
__device__ __forceinline__ u64 row_target(float tp, u64 mass) {
  u64 r = 0;
  if (tp >= 1.f) {
    r = mass;
  } else if (tp > 0.f) {
    const unsigned b = __float_as_uint(tp);
    const int e = (int)(b >> 23);  // biased exponent, <= 126 since tp < 1
    const int sh = 150 - e;        // tp = m * 2^-sh, sh >= 24
    const u64 m = (b & 0x7FFFFFu) | 0x800000u;
    const u64 lo = mass * m, hi = __umul64hi(mass, m);  // the 88-bit product
    if (e == 0 || sh >= 128) {
      r = 0;  // below 2^-126 * 2^64 units, so below one unit: the minimum applies
    } else if (sh < 64) {
      r = ((lo >> sh) | (hi << (64 - sh))) + ((lo & ((1ULL << sh) - 1)) != 0);
    } else {
      const u64 low_bits = (sh == 64) ? 0ULL : (hi & ((1ULL << (sh - 64)) - 1));
      r = (hi >> (sh - 64)) + ((lo | low_bits) != 0);
    }
  }
  return r ? r : 1ULL;
}

template <typename T>
__device__ __forceinline__ T warp_sum(T x) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) x += __shfl_xor_sync(0xffffffffu, x, off);
  return x;
}
__device__ __forceinline__ void cl_arrive() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cl_wait() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

// mode 0: p >= 2^-16, key bits>>20      mode 1: 0 < p < 2^-16, key bits>>20
// mode 2: bits>>20 == prefix, key (bits>>9)&0x7FF     mode 3: bits>>9 == prefix, key bits&0x1FF
__device__ __forceinline__ void hist_add(float x, int mode, int prefix, float scale, float scale2,
                                         u64* lmass, unsigned* lcnt) {
  int bits = __float_as_int(x);
  if (bits <= 0 || bits > 0x7f800000) return;  // zero, negative or NaN: no mass, never kept
  bool f;
  int key;
  if (mode == 0) {
    f = bits >= LBITS;
    key = bits >> 20;
  } else if (mode == 1) {
    f = bits < LBITS;
    key = bits >> 20;
  } else if (mode == 2) {
    f = (bits >> 20) == prefix;
    key = (bits >> 9) & 0x7FF;
  } else {
    f = (bits >> 9) == prefix;
    key = bits & 0x1FF;
  }
  if (f) {
    atomicAdd(&lmass[key], to_fix(x, scale, scale2));
    atomicAdd(&lcnt[key], 1u);
  }
}

// The radix path's rewrite predicate: x is at or above the threshold's bit prefix and is not a NaN.
// A NaN has no mass anywhere in this kernel (it is never a head entry or in a histogram bin, and
// the float32 slice sums add fmaxf(NaN, 0) = 0) and is never kept, as on the head paths (x >= thr
// is false for it); its bit pattern sorts above every finite value, so the prefix test alone would
// keep it.
__device__ __forceinline__ bool radix_keep(float x, int kshift, int kprefix) {
  const int bits = __float_as_int(x);
  return bits <= 0x7f800000 && (bits >> kshift) >= kprefix;
}

// A finite positive entry kept by the rewrite predicate, as float64 (0 otherwise).
__device__ __forceinline__ double kept_f64(float x, int kshift, int kprefix) {
  const int bits = __float_as_int(x);
  return (bits > 0 && bits < 0x7f800000 && (bits >> kshift) >= kprefix) ? (double)x : 0.0;
}

// Fixed-point mass of an entry below the fallback's level-0 range (0 for p >= 2^-16, p <= 0 and
// NaN, including a NaN with the sign bit set).
__device__ __forceinline__ u64 tail_fix(float x, float scale, float scale2) {
  return (__float_as_uint(x) - 1u < (unsigned)(LBITS - 1)) ? to_fix(x, scale, scale2) : 0ULL;
}

// A head entry: bits >= those of 2^-8. A +NaN passes this test (its bits sort above every finite
// value) and is neutralised after the gather (head_fix), which keeps the streaming loop as it is.
__device__ __forceinline__ void cand_add(float x, int gi, int* s_n, float* s_cv, int* s_ci) {
  if (__float_as_int(x) >= CBITS) {
    int k = atomicAdd(s_n, 1);
    if (k < CAP) {
      s_cv[k] = x;
      s_ci[k] = gi;
    }
  }
}

template <int C, bool VEC, int NT, int MINB, int UNR>
__global__ void __launch_bounds__(NT, MINB)
    topp_kernel(const float* __restrict__ probs, const float* __restrict__ top_p,
                float* __restrict__ out, int V, int chunk) {
  constexpr int BPT = NBINS / NT;
  constexpr int S = NBINS / C;  // reduced bins owned by each CTA (fallback)
  constexpr int LOGC = (C == 1) ? 0 : (C == 2) ? 1 : (C == 4) ? 2 : (C == 8) ? 3 : 4;
  __shared__ u64 lmass[NBINS];
  __shared__ unsigned lcnt[NBINS];
  __shared__ u64 gmass[4 * S];
  __shared__ unsigned gcnt[4 * S];
  __shared__ float s_cv[CAP];  // this CTA's head entries (value, row index)
  __shared__ int s_ci[CAP];
  __shared__ float s_all[CAP];  // all head entries of the row (gathered)
  __shared__ int s_n;
  __shared__ int s_off[C + 1];
  __shared__ u64 s_wsum[NT / 32];
  __shared__ float s_wmass[NT / 32];  // per-warp float32 slice mass
  __shared__ float s_rmass;           // this CTA's float32 slice mass (read by the cluster)
  __shared__ float s_fmass;           // the row's float32 mass
  __shared__ u64 s_tlo, s_thi;        // bounds of the row's target (head paths)
  __shared__ int s_big;               // the row's mass is >= BIG_MASS
  __shared__ int s_amb;               // a head prefix sum lies within the target's bounds
  __shared__ u64 s_rsum, s_tail;      // fallback level 0: this CTA's / the row's tail mass
  __shared__ int s_best;
  __shared__ u64 s_above, s_bmass;
  __shared__ unsigned s_bcnt;
  __shared__ float s_thr;
  __shared__ u64 s_ksum;
  __shared__ int s_small;
  __shared__ double s_dsum[NT / 32 + 2];  // float64 kept-mass recheck: warps, CTA, row

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = (int)cluster.block_rank();
  const int row = blockIdx.x / C;
  const int t = threadIdx.x;
  const unsigned lane = t & 31, w = t >> 5;
  const int cstart = rank * chunk;
  const int len = max(0, min(chunk, V - cstart));
  const float* psrc = probs + (long long)row * V + cstart;
  float* odst = out + (long long)row * V + cstart;

  if (t == 0) s_n = 0;
  const float tp = top_p[row];
  __syncthreads();

  // ---- single streaming pass: collect head entries, add up the slice and zero the output slice
  // The output is a fresh (uninitialised) buffer: every element is written here, the kept
  // entries are overwritten by the scatter below (ordered by the barriers in between).
  float fsum = 0.f;  // positive mass (NaN and p <= 0 add 0, as in to_fix)
  if (VEC) {
    const float4* p4 = reinterpret_cast<const float4*>(psrc);
    float4* o4 = reinterpret_cast<float4*>(odst);
    const int len4 = len >> 2;
    const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int base = 0; base < len4; base += NT * UNR) {
      float4 x[UNR];
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const int i = base + u * NT + t;
        if (i < len4) {
          x[u] = __ldg(p4 + i);
          o4[i] = z;
        } else {
          x[u] = z;
        }
      }
#pragma unroll
      for (int u = 0; u < UNR; ++u) {
        const int gi = cstart + 4 * (base + u * NT + t);
        cand_add(x[u].x, gi + 0, &s_n, s_cv, s_ci);
        fsum += fmaxf(x[u].x, 0.f);
        cand_add(x[u].y, gi + 1, &s_n, s_cv, s_ci);
        fsum += fmaxf(x[u].y, 0.f);
        cand_add(x[u].z, gi + 2, &s_n, s_cv, s_ci);
        fsum += fmaxf(x[u].z, 0.f);
        cand_add(x[u].w, gi + 3, &s_n, s_cv, s_ci);
        fsum += fmaxf(x[u].w, 0.f);
      }
    }
  } else {
    for (int i = t; i < len; i += NT) {
      float x = __ldg(psrc + i);
      odst[i] = 0.f;
      cand_add(x, cstart + i, &s_n, s_cv, s_ci);
      fsum += fmaxf(x, 0.f);
    }
  }
  fsum = warp_sum(fsum);
  if (lane == 0) s_wmass[w] = fsum;
  __syncthreads();
  if (t < 32) {
    const float f = warp_sum((t < NT / 32) ? s_wmass[t] : 0.f);
    if (t == 0) s_rmass = f;
  }
  cluster.sync();

  // ---- gather the row's head entries (one DSMEM exchange for the counts and slice masses, one
  // for the values) and bound the target ----
  if (t < 32) {
    int n = (t < C) ? *cluster.map_shared_rank(&s_n, t) : 0;
    // the same values in the same order in every CTA: the row's float32 mass is uniform
    const float f = warp_sum((t < C) ? *cluster.map_shared_rank(&s_rmass, t) : 0.f);
    int incl = n;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      int x = __shfl_up_sync(0xffffffffu, incl, off);
      if ((int)lane >= off) incl += x;
    }
    if (t < C) s_off[t] = incl - n;
    if (t == C - 1) s_off[C] = incl;
    const bool heavy = !(f < BIG_MASS);  // also an inf entry
    if (t == 0) {
      s_fmass = f;
      s_big = heavy;
    }
    if (t < 2) {
      u64 tg = ~0ULL;  // a heavy row never takes the head paths
      if (!heavy) {
        // f is a sum tree of positive terms of height <= per_thread + 15, so it is within
        // (per_thread + 24) * 2^-24 of the exact sum of the entries; to_fix rounds each entry by
        // at most half a unit. M (the fixed-point mass the radix path sums) lies in
        // [mlo - V, mhi + V]; lane 0 bounds the target from below, lane 1 from above.
        const int per_thread = VEC ? 4 * ((chunk / 4 + NT - 1) / NT) : (chunk + NT - 1) / NT;
        const float d = (float)(per_thread + 24) * 5.9604645e-8f;
        const u64 mlo = __float2ull_rd(__fmul_rd(__fmul_rd(f, __fsub_rd(1.f, d)), FIX_SCALE));
        const u64 mhi = __float2ull_ru(__fmul_ru(__fmul_ru(f, __fadd_ru(1.f, d)), FIX_SCALE));
        tg = row_target(tp, t == 0 ? (mlo > (u64)V ? mlo - (u64)V : 0ULL) : mhi + (u64)V);
      }
      if (t == 0) {
        s_tlo = tg;
      } else {
        s_thi = tg;
      }
    }
  }
  __syncthreads();
  const int ntot = s_off[C];
  const bool big = s_big;  // uniform across the cluster, like every decision derived from f
  // The exact target lies in [tlo, thi]: an entry whose prefix sum is below tlo is in the
  // nucleus, one at or above thi is out, one in between makes the row ambiguous on the head path.
  const u64 tlo = s_tlo, thi = s_thi;
  bool fast = !big && ntot <= CAP;
  if (!big && ntot <= 32) {
    // Small head set (serving rows hold 1-4): warp 0 gathers and resolves it with shuffles
    // while the rest of the block waits at the barrier below.
    if (t < 32) {
      float v = 0.f;
      if (t < ntot) {
        int r = 0;
#pragma unroll
        for (int q = 1; q < C; ++q) r += (t >= s_off[q]);
        v = *(cluster.map_shared_rank(s_cv, r) + (t - s_off[r]));
        s_all[t] = v;
      }
      if (ntot >= 2) {
        const u64 fv = (t < ntot) ? head_fix(v) : 0ULL;
        u64 wtot = fv, excl = 0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) wtot += __shfl_xor_sync(0xffffffffu, wtot, off);
        for (int j = 0; j < ntot; ++j) {
          const float y = __shfl_sync(0xffffffffu, v, j);
          const u64 fy = __shfl_sync(0xffffffffu, fv, j);
          if (y > v || (y == v && j < t)) excl += fy;
        }
        const int cb = (t < ntot && excl < tlo) ? __float_as_int(v) : 0x7fffffff;
        const int bst = __reduce_min_sync(0xffffffffu, cb);
        const float thr = __int_as_float(bst);
        const bool amb = __any_sync(0xffffffffu, t < ntot && excl >= tlo && excl < thi);
        u64 ks = (t < ntot && v >= thr) ? fv : 0ULL;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) ks += __shfl_xor_sync(0xffffffffu, ks, off);
        if (t == 0) {
          s_small = (wtot >= thi && !amb) ? 1 : 0;
          s_thr = thr;
          s_ksum = ks;
        }
      }
    }
  } else if (fast && t < ntot) {
    int r = 0;
#pragma unroll
    for (int q = 1; q < C; ++q) r += (t >= s_off[q]);
    s_all[t] = *(cluster.map_shared_rank(s_cv, r) + (t - s_off[r]));
  }
  __syncthreads();
  cl_arrive();  // DSMEM reads of the fast path are done (waited on before any further DSMEM use)

  // One-entry nucleus (the common serving case): every CTA holds the same sole head entry,
  // so its exact fixed-point mass decides without further block reductions or atomics.
  if (!big && ntot == 1 && head_fix(s_all[0]) >= thi) {
    if (s_n == 1 && t == 0) {
      const float x = s_cv[0];
      out[(long long)row * V + s_ci[0]] = x / x;
    }
    cl_wait();
    return;
  }
  if (!big && ntot >= 2 && ntot <= 32 && s_small) {
    const float thr = s_thr;
    const float sumf = __ull2float_rn(s_ksum) * (1.0f / FIX_SCALE);
    const int n = min(s_n, CAP);
    for (int i = t; i < n; i += NT) {
      const float x = s_cv[i];
      if (x >= thr) out[(long long)row * V + s_ci[i]] = x / sumf;
    }
    cl_wait();
    return;
  }

  // ---- exact resolve: element e is in the prefix iff (sum of strictly-earlier entries) < target
  u64 tot = 0;
  if (fast) {
    float v = 0.f;
    u64 excl = 0;
    if (t == 0) {
      s_ksum = 0;
      s_best = 0x7fffffff;
      s_amb = 0;
    }
    __syncthreads();
    if (t < ntot) {
      v = s_all[t];
      for (int j = 0; j < ntot; ++j) {
        const float y = s_all[j];
        const u64 fy = head_fix(y);
        tot += fy;
        if (y > v || (y == v && j < t)) excl += fy;
      }
      if (excl < tlo) {
        atomicMin(&s_best, __float_as_int(v));
      } else if (excl < thi) {
        s_amb = 1;
      }
    }
    __syncthreads();
    // total mass of the head entries (identical in every thread that computed it; thread 0 always
    // does)
    if (t == 0) s_above = tot;
    __syncthreads();
    tot = s_above;
    fast = (ntot > 0) && (tot >= thi) && !s_amb;
    if (fast) {
      // Positive float bit patterns are monotonically ordered, including threshold ties.
      const float thr = __int_as_float(s_best);
      if (t < ntot && v >= thr) atomicAdd(&s_ksum, to_fix(v));
      __syncthreads();
      const float sumf = __ull2float_rn(s_ksum) * (1.0f / FIX_SCALE);
      const int n = min(s_n, CAP);
      for (int i = t; i < n; i += NT) {
        const float x = s_cv[i];
        if (x >= thr) out[(long long)row * V + s_ci[i]] = x / sumf;
      }
      cl_wait();
      return;
    }
  }
  cl_wait();  // uniform across the cluster: every CTA reached the same decision

  // ---- exact fallback: radix histogram over float bits, reduced through DSMEM ----
  // Initialize fallback-only storage only for rows that actually need radix selection.
  for (int i = t; i < 4 * S; i += NT) {
    gmass[i] = 0;
    gcnt[i] = 0;
  }
  for (int i = t; i < NBINS; i += NT) {
    lmass[i] = 0;
    lcnt[i] = 0;
  }
  cluster.sync();
  u64 above = 0, sumfix = 0;
  int mode = 0, prefix = 0, kshift = 0, kprefix = 0;
  const float4* sd4 = reinterpret_cast<const float4*>(psrc);
  const int len4 = len >> 2;
  // The radix path's fixed point is 2^-q per row, q chosen from the row's float32 mass f = 2^e *
  // [1, 2) so that its exact mass lies in [2^58, 2^61) units (below 2^(e+2) allowing for f's
  // rounding): the resolution follows the row's own mass (an entry of at least 2^-35 of it converts
  // exactly, a smaller one to the nearest unit), whatever the row's scale, so a vanishing row (all
  // entries below 2^-57) is selected like any other instead of having zero mass. A row the head
  // paths may resolve (mass < 64) keeps q >= 56, their fixed point, so both paths resolve the same
  // target. An all-zero row (f = 0, ilogbf = INT_MIN) has no mass at any scale. (A row of mass in
  // [8, 64) keeps 2^-56; computing the scale only in the branch below keeps ptxas from spilling
  // more in the streaming loop of the 40-register launch configurations.)
  int q = 56;
  float scale = FIX_SCALE, scale2 = 1.f;  // 2^q = scale * scale2, each factor a normal float
  if (big || s_fmass < 8.f) {
    const int e = min(max(ilogbf(s_fmass), -149), 185);
    q = 59 - e;  // in [-126, 53] for a heavy row, [57, 208] otherwise
    const int q1 = min(q, 127);
    scale = exp2i(q1);
    scale2 = exp2i(q - q1);
  }
  u64 target = 1;  // the exact target, set on level 0
  for (int lvl = 0; lvl < 4; ++lvl) {
    if (lvl > 0) {
      for (int i = t; i < NBINS; i += NT) {
        lmass[i] = 0;
        lcnt[i] = 0;
      }
    }
    __syncthreads();
    // Level 0 bins the entries >= 2^-16 and also sums the rest: the row's exact mass is then the
    // level's bin total plus that tail.
    u64 tail = 0;
    if (VEC) {
      for (int i = t; i < len4; i += NT) {
        float4 x = sd4[i];
        hist_add(x.x, mode, prefix, scale, scale2, lmass, lcnt);
        hist_add(x.y, mode, prefix, scale, scale2, lmass, lcnt);
        hist_add(x.z, mode, prefix, scale, scale2, lmass, lcnt);
        hist_add(x.w, mode, prefix, scale, scale2, lmass, lcnt);
        if (lvl == 0) {
          tail += tail_fix(x.x, scale, scale2) + tail_fix(x.y, scale, scale2) +
                  tail_fix(x.z, scale, scale2) + tail_fix(x.w, scale, scale2);
        }
      }
    } else {
      for (int i = t; i < len; i += NT) {
        const float x = psrc[i];
        hist_add(x, mode, prefix, scale, scale2, lmass, lcnt);
        if (lvl == 0) tail += tail_fix(x, scale, scale2);
      }
    }
    if (lvl == 0) {
      tail = warp_sum(tail);
      if (lane == 0) s_wsum[w] = tail;
    }
    __syncthreads();
    if (lvl == 0 && t < 32) {
      const u64 m = warp_sum((t < NT / 32) ? s_wsum[t] : 0ULL);
      if (t == 0) s_rsum = m;  // read by the cluster after the cluster barrier below
    }
    u64* gm = gmass + lvl * S;
    unsigned* gc = gcnt + lvl * S;
    for (int b = t; b < NBINS; b += NT) {
      unsigned c = lcnt[b];
      if (c) {
        const int o = b & (C - 1), idx = b >> LOGC;
        atomicAdd(cluster.map_shared_rank(gm, o) + idx, lmass[b]);
        atomicAdd(cluster.map_shared_rank(gc, o) + idx, c);
      }
    }
    cluster.sync();

    const int nb = (mode == 3) ? 512 : NBINS;
    u64 m[BPT];
    unsigned c[BPT];
    u64 tsum = 0;
#pragma unroll
    for (int i = 0; i < BPT; ++i) {
      const int b = BPT * t + i;
      if (b < nb) {
        const int o = b & (C - 1), idx = b >> LOGC;
        m[i] = *(cluster.map_shared_rank(gm, o) + idx);
        c[i] = *(cluster.map_shared_rank(gc, o) + idx);
      } else {
        m[i] = 0;
        c[i] = 0;
      }
      tsum += m[i];
    }
    u64 incl = tsum;  // inclusive suffix sum within the warp (higher lanes = higher bins)
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      u64 x = __shfl_down_sync(0xffffffffu, incl, off);
      if (lane + off < 32) incl += x;
    }
    if (lane == 0) s_wsum[w] = incl;
    if (t == 0) s_best = -1;
    if (lvl == 0 && t < 32) {
      const u64 m = warp_sum((t < C) ? *cluster.map_shared_rank(&s_rsum, t) : 0ULL);
      if (t == 0) s_tail = m;
    }
    __syncthreads();
    u64 wabove = 0, ftot = 0;
#pragma unroll
    for (int ww = 0; ww < NT / 32; ++ww) {
      u64 x = s_wsum[ww];
      ftot += x;
      if (ww > (int)w) wabove += x;
    }
    if (lvl == 0) target = row_target(tp, ftot + s_tail);
    u64 cum = above + wabove + (incl - tsum);
    int mybest = -1;
    u64 myabove = 0, mym = 0;
    unsigned myc = 0;
#pragma unroll
    for (int i = BPT - 1; i >= 0; --i) {
      cum += m[i];
      if (mybest < 0 && c[i] && cum >= target) {
        mybest = BPT * t + i;
        myabove = cum - m[i];
        mym = m[i];
        myc = c[i];
      }
    }
    if (mybest >= 0) atomicMax(&s_best, mybest);
    __syncthreads();
    const int best = s_best;
    if (best >= 0 && mybest == best) {
      s_above = myabove;
      s_bmass = mym;
      s_bcnt = myc;
    }
    __syncthreads();

    if (best < 0) {
      if (mode == 0) {
        above += ftot;
        mode = 1;
        continue;
      }
      kshift = 0;
      kprefix = 0;
      sumfix = above + ftot;  // whole positive mass < target: keep all
      break;
    }
    above = s_above;
    const int shift = (mode <= 1) ? 20 : (mode == 2 ? 9 : 0);
    const int fullprefix = (mode <= 1) ? best : ((prefix << (mode == 2 ? 11 : 9)) | best);
    if (s_bcnt == 1 || mode == 3) {
      kshift = shift;
      kprefix = fullprefix;
      sumfix = above + s_bmass;
      break;
    }
    prefix = fullprefix;
    mode = (mode <= 1) ? 2 : 3;
  }

  // Like the stock AIR apply pass, a row whose kept mass is <= 1e-8 is left unnormalised (norm
  // 1.0): an all-zero (or vanishing) row stays as it is instead of becoming 0/0 = NaN or x/0 = inf.
  // The head paths above never get here with such a row (their entries are >= 2^-8).
  // The cutoff 1e-8f is exactly KEEP_CUT units of 2^-56, a multiple of 2^6, so exactly
  // KEEP_CUT * 2^(q - 56) units of 2^-q for q in [50, 90]. A row with q < 50 has a mass of at least
  // 2^9 (kept mass >= its maximum >= 2^9 / V: always normalised), one with q > 90 a mass below
  // 2^-29 (never normalised). sumfix carries up to 1/2 unit of rounding per kept entry (at most V
  // of them), so a sum within V units of the cutoff can land on the wrong side of it: such a row
  // re-sums its kept entries in float64 (relative error below V * 2^-53, in a fixed order) and
  // decides on that. A probability row never gets here (its kept mass is at least its maximum,
  // >= 1/V), so the served rows pay one integer comparison.
  constexpr u64 KEEP_CUT = 720575936ULL;
  const u64 cut = (q < 50 || q > 90) ? 0ULL
                  : (q >= 56)        ? (KEEP_CUT << (q - 56))
                                     : (KEEP_CUT >> (56 - q));
  bool normalise = (q < 50) || (q <= 90 && sumfix > cut);
  float sumf = (q <= 90) ? __ull2float_rn(sumfix) * exp2i(-q) : 1.0f;
  // q and sumfix are identical in every CTA of the cluster, so this branch (and its barrier) is
  // uniform.
  if (q >= 50 && q <= 90 && (sumfix > cut ? sumfix - cut : cut - sumfix) < (u64)V) {
    double part = 0.0;
    if (VEC) {
      for (int i = t; i < len4; i += NT) {
        const float4 x = sd4[i];
        part += kept_f64(x.x, kshift, kprefix) + kept_f64(x.y, kshift, kprefix) +
                kept_f64(x.z, kshift, kprefix) + kept_f64(x.w, kshift, kprefix);
      }
    } else {
      for (int i = t; i < len; i += NT) part += kept_f64(psrc[i], kshift, kprefix);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) part += __shfl_xor_sync(0xffffffffu, part, off);
    if (lane == 0) s_dsum[w] = part;
    __syncthreads();
    if (t == 0) {
      double cta = 0.0;
      for (int ww = 0; ww < NT / 32; ++ww) cta += s_dsum[ww];
      s_dsum[NT / 32] = cta;
    }
    cluster.sync();
    if (t == 0) {
      double row_sum = 0.0;  // the same values in the same order in every CTA
      for (int r = 0; r < C; ++r) row_sum += *cluster.map_shared_rank(&s_dsum[NT / 32], r);
      s_dsum[NT / 32 + 1] = row_sum;
    }
    __syncthreads();
    const double row_sum = s_dsum[NT / 32 + 1];
    normalise = row_sum > (double)1e-8f;
    sumf = (float)row_sum;
  }
  cl_arrive();  // all DSMEM traffic of this CTA is done
  if (!normalise) sumf = 1.0f;
  if (VEC) {
    float4* o4 = reinterpret_cast<float4*>(odst);
    for (int i = t; i < len4; i += NT) {
      float4 x = sd4[i], o = make_float4(0.f, 0.f, 0.f, 0.f);
      if (radix_keep(x.x, kshift, kprefix)) o.x = x.x / sumf;
      if (radix_keep(x.y, kshift, kprefix)) o.y = x.y / sumf;
      if (radix_keep(x.z, kshift, kprefix)) o.z = x.z / sumf;
      if (radix_keep(x.w, kshift, kprefix)) o.w = x.w / sumf;
      o4[i] = o;
    }
  } else {
    for (int i = t; i < len; i += NT) {
      float x = psrc[i];
      odst[i] = radix_keep(x, kshift, kprefix) ? x / sumf : 0.f;
    }
  }
  cl_wait();  // keep this CTA's shared memory alive until the cluster is done with it
}

template <int C, bool VEC, int NT, int MINB, int UNR>
inline cudaError_t launch_t(const float* probs, const float* top_p, float* out, int B, int V,
                            int chunk, cudaStream_t stream) {
  auto kernel = topp_kernel<C, VEC, NT, MINB, UNR>;
  // Cluster size 16 exceeds the portable limit of 8. Set on every launch (a host-side attribute
  // of the current device, legal during stream capture) rather than memoised once per process,
  // so every device of a multi-GPU process gets it.
  cudaError_t status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (status != cudaSuccess) return status;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(B * C, 1, 1);
  cfg.blockDim = dim3(NT, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = C;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, probs, top_p, out, V, chunk);
}

// Launch configurations: cluster size, threads per CTA, minimum CTAs per SM (launch bounds) and
// float4 loads in flight per thread. Tuned on B300 for the serving shapes; plan_launch picks one.
struct LaunchConfig {
  int cluster_size;
  int num_threads;
  int min_blocks;
  int unroll;
};
inline constexpr LaunchConfig kLaunchConfigs[] = {
    {8, 256, 6, 8},    // 0: short rows (< 64K entries; the static fallback buffers need C >= 8)
    {16, 1024, 1, 4},  // 1: B <= 9,  V > 200000
    {16, 1024, 1, 3},  // 2: B <= 9,  V <= 200000
    {16, 512, 3, 8},   // 3: B <= 20, V > 200000
    {16, 512, 3, 5},   // 4: B <= 20, V <= 200000
    {16, 256, 4, 10},  // 5: B <= 30, V <= 200000
    {8, 512, 3, 8},    // 6: B > 30,  V <= 200000
    {8, 512, 2, 16},   // 7: B <= 40, V > 200000
    {8, 512, 4, 8},    // 8: B > 40,  V > 200000
};
inline constexpr int kNumLaunchConfigs = sizeof(kLaunchConfigs) / sizeof(kLaunchConfigs[0]);

// Index into kLaunchConfigs for a (B, V) problem. Single source of truth for the launcher and the
// binding's introspection export.
inline int plan_launch(int B, int V) {
  if ((V + 15) / 16 < 4096) return 0;
  if (B <= 9) return V > 200000 ? 1 : 2;
  if (B <= 20) return V > 200000 ? 3 : 4;
  if (V <= 200000) return B <= 30 ? 5 : 6;
  return B <= 40 ? 7 : 8;
}

template <bool VEC, int I>
inline cudaError_t launch_config(const float* probs, const float* top_p, float* out, int B, int V,
                                 cudaStream_t s) {
  constexpr LaunchConfig k = kLaunchConfigs[I];
  const int chunk = (((V + k.cluster_size - 1) / k.cluster_size) + 3) & ~3;
  return launch_t<k.cluster_size, VEC, k.num_threads, k.min_blocks, k.unroll>(probs, top_p, out, B,
                                                                              V, chunk, s);
}

template <bool VEC>
inline cudaError_t dispatch(int config, const float* probs, const float* top_p, float* out, int B,
                            int V, cudaStream_t s) {
  static_assert(kNumLaunchConfigs == 9, "dispatch must cover every launch configuration");
  switch (config) {
    case 0:
      return launch_config<VEC, 0>(probs, top_p, out, B, V, s);
    case 1:
      return launch_config<VEC, 1>(probs, top_p, out, B, V, s);
    case 2:
      return launch_config<VEC, 2>(probs, top_p, out, B, V, s);
    case 3:
      return launch_config<VEC, 3>(probs, top_p, out, B, V, s);
    case 4:
      return launch_config<VEC, 4>(probs, top_p, out, B, V, s);
    case 5:
      return launch_config<VEC, 5>(probs, top_p, out, B, V, s);
    case 6:
      return launch_config<VEC, 6>(probs, top_p, out, B, V, s);
    case 7:
      return launch_config<VEC, 7>(probs, top_p, out, B, V, s);
    default:
      return launch_config<VEC, 8>(probs, top_p, out, B, V, s);
  }
}

// probs, out: [B, V] float32, contiguous; top_p: [B] float32. `out` need not be initialised.
inline cudaError_t topp_renorm_launch(const float* probs, const float* top_p, float* out, int B,
                                      int V, cudaStream_t stream) {
  if (B <= 0 || V <= 0) return cudaSuccess;
  const int config = plan_launch(B, V);
  // The float4 path needs 16-byte aligned rows: V % 4 == 0 and 16-byte aligned bases.
  const bool aligned = ((reinterpret_cast<uintptr_t>(probs) | reinterpret_cast<uintptr_t>(out)) &
                        uintptr_t(15)) == 0;
  if ((V & 3) == 0 && aligned) return dispatch<true>(config, probs, top_p, out, B, V, stream);
  return dispatch<false>(config, probs, top_p, out, B, V, stream);
}

}  // namespace top_p_renorm_specialized
}  // namespace sampling
}  // namespace flashinfer

#endif  // FLASHINFER_TOP_P_RENORM_SPECIALIZED_CUH_
