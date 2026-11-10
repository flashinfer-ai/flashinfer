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

// mHC pre-map big-fuse with fused output RMSNorm (H = 4096, HC = 4), decode regime.
// SM100-family only (thread-block clusters, distributed shared memory, st.async).
//
// One CL-CTA thread-block cluster per token (CL = 4, or 2 for T > 24); CTA `rank` owns channels
// [rank * CH, (rank + 1) * CH).
//   warps 1..NWC   : pre sigmoids (all 4 in every lane), 4-stream collapse (16 B bf16 loads),
//                    bf16 rounding of ol, sum(ol^2) partial, DSMEM all-gather of the partials
//                    (st.async + mbarrier; the cluster wait is taken during the DRAM loads),
//                    RMSNorm x weight, store.
//   warp 0, rank 1 : softmax + 20-iteration division-free Sinkhorn (rolled loop) -> comb_mix.
//   warp 0, rank 2 : post sigmoid -> post_mix.  Warp 0 of the other ranks is idle (warp 0 of
//                    block 0 is routinely the last warp of the grid to start, so it carries no
//                    work).
// Every warp issues its loads first (one DRAM round trip): the logit partials are spread over the
// lanes (lane group g = lane >> 2 loads split-K partial s = g, reduced later by an xor butterfly),
// so the code in front of the first load is a few dozen instructions.  All input loads are plain
// ld.global (an .nc load gets hoisted by ptxas above griddepcontrol.wait).
//
// Numerics (no fast-math): libdevice expf, IEEE division for the post sigmoid and a Newton-refined
// reciprocal for the pre sigmoids (rcp.approx seed, e = fma(-d, y, 1), y' = fma(y, e, y)),
// rsqrt.approx.ftz for rsqrt (arguments >= 1e-5, never subnormal).  The Sinkhorn half-steps are
// division-free (exact cofactor products + power-of-two rescaling, see hs_lam and rebalance); the
// one reciprocal (final 1/G) is a Newton-refined rcp.approx seed.  Every Sinkhorn quantity stays
// within proven bounds far inside the normal fp32 range for every finite input (see rebalance).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "tvm_ffi_utils.h"

namespace flashinfer::mhc::pre_norm_cluster {
namespace {

constexpr int H = 4096;
constexpr int HC = 4;
constexpr int MIX = 24;
constexpr int MIX4 = MIX / 4;       // float4 per (s, t) row of gemm_out_mul
constexpr int VEC = 8;              // bf16 per 16 B
constexpr int NCT = 64;             // collapse threads per CTA
constexpr int NWC = NCT / 32;       // collapse warps per CTA
constexpr int NTHREADS = NCT + 32;  // + Sinkhorn / post warp
constexpr int CL_MAX = 8;           // largest cluster (CTAs per token)
constexpr int PU = 8;               // NS partials loaded per lane per pass

constexpr float RMS_EPS = 1e-5f;
constexpr float HC_PRE_EPS = 1e-6f;
constexpr float SK_EPS = 1e-6f;
constexpr float POST_MULT = 2.0f;
constexpr float NORM_EPS = 1e-5f;
constexpr float INV_RMS_NUMEL = 1.0f / 16384.0f;  // exact power of two: x * 2^-14 == x / 16384
constexpr float INV_H = 1.0f / 4096.0f;           // exact power of two

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}

// MUFU reciprocal seed. NEVER used directly: every use below is followed by a Newton step.
__device__ __forceinline__ float rcp_seed(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// Newton-refined reciprocal: e = fma(-x, y, 1); y' = fma(y, e, y).
__device__ __forceinline__ float rcp_refined(float x) {
  const float y = rcp_seed(x);
  const float e = fmaf(-x, y, 1.0f);
  return fmaf(y, e, y);
}

// MUFU.RSQ (as torch.rsqrt); the arguments are >= 1e-5, so the ftz form needs no subnormal guard.
__device__ __forceinline__ float rsqrt_ftz(float x) {
  float y;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// IEEE sigmoid (rcp.rn, libdevice expf), as torch.sigmoid.
__device__ __forceinline__ float sigmoid_ieee(float x) { return 1.0f / (1.0f + expf(-x)); }

// Coherent global loads (ld.global, not .nc): ptxas hoists read-only .nc loads above
// griddepcontrol.wait, which would break the PDL contract (no input read before the wait).
__device__ __forceinline__ float ld_f32(const float* p) {
  float v;
  asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float4 ld_f32x4(const float4* p) {
  float4 v;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
__device__ __forceinline__ uint4 ld_u32x4(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

// sigmoid with a Newton-refined reciprocal (rcp.approx seed, e = fma(-d, y, 1), y' = fma(y, e, y))
// and libdevice expf.  d = 1 + expf(-x) >= 1 is clamped to 2^126 so the seed and the residual stay
// finite (for d >= 2^126 the IEEE result is <= 2^-126, below every eps it is added to).
__device__ __forceinline__ float sigmoid_nr(float x) {
  return rcp_refined(fminf(1.0f + expf(-x), 0x1p126f));
}

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ float4 add4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 shfl_xor4(float4 v, int m) {
  v.x = __shfl_xor_sync(0xffffffffu, v.x, m);
  v.y = __shfl_xor_sync(0xffffffffu, v.y, m);
  v.z = __shfl_xor_sync(0xffffffffu, v.z, m);
  v.w = __shfl_xor_sync(0xffffffffu, v.w, m);
  return v;
}
__device__ __forceinline__ float sel4(float4 v, int j) {
  return j == 0 ? v.x : (j == 1 ? v.y : (j == 2 ? v.z : v.w));
}

// Division-free Sinkhorn in diagonal-scaling form.
//   c[i][j] = r[i] * K[i][j] * s[j] / G,   K = S_i * (softmax + eps) (fixed),  E = eps * G.
// A half-step that normalises axis k (z_k = sum_k(c) + eps) is, exactly in real arithmetic,
//   W_k = (scale_k * matvec_k) + E      (= G * z_k)
//   1/z_k = G * Q_k / P,  Q_k = prod_{l != k} W_l,  P = prod_l W_l     (batch-inverted reciprocal)
//   scale_k <- scale_k * kappa * Q_k,  G <- kappa * P,  E <- eps * G
// i.e. the common denominator P of the batch-inverted reciprocal is accumulated into G and the
// division by G is done once, at the end, with a Newton-refined reciprocal.  kappa is an exact
// power of two taken from the exponent bits of G (keeps every value in range, exact).
// No reciprocal, division or approximate instruction is executed inside the 39 half-steps.
// The scale vectors carry a free power-of-two gauge that rebalance() fixes once per (rows, columns)
// pair, which keeps every carried quantity at a provable scale.

// kappa = 2^(-e(P)) (first half-step only: resets the scale from the softmax product).
__device__ __forceinline__ float kappa_direct(float P) {
  const uint32_t eb = __float_as_uint(P) & 0x7f800000u;
  return __uint_as_float((254u << 23) - eb);
}

// Same half-step with the power-of-two rescale lambda = kappa entering through lambda*b1 and
// lambda*E only (applied as exponent-field integer adds): W1' = lambda*W1 exactly, so W13' =
// lambda*W1*W3 and every cofactor product carries exactly one factor lambda (one multiply level
// shorter than scaling the cofactor pairs).  The summation / statement order is the one that gave
// ptxas's loop allocation the fewest register-bank conflicts.
template <bool COL>
__device__ __forceinline__ void hs_lam(const float K[4][4], const float a[4], float b[4], float& G,
                                       float& E) {
#define KM(i, k) (COL ? K[i][k] : K[k][i])
  // lambda = 2^(-4 e(G)) applied as an exact exponent-field add: bits(x * lambda) ==
  // bits(x) + 4 * (~bits(G) & expmask)  (mod 2^32) while x * lambda is normal  (LOP3 + 2 LEA).
  const uint32_t lm = ~__float_as_uint(G) & 0x7f800000u;
  const float v2 =
      fmaf(KM(1, 2), a[1], fmaf(KM(2, 2), a[2], fmaf(KM(3, 2), a[3], KM(0, 2) * a[0])));
  const float v0 =
      fmaf(KM(1, 0), a[1], fmaf(KM(2, 0), a[2], fmaf(KM(3, 0), a[3], KM(0, 0) * a[0])));
  const float v3 =
      fmaf(KM(1, 3), a[1], fmaf(KM(2, 3), a[2], fmaf(KM(3, 3), a[3], KM(0, 3) * a[0])));
  const float v1 =
      fmaf(KM(1, 1), a[1], fmaf(KM(2, 1), a[2], fmaf(KM(3, 1), a[3], KM(0, 1) * a[0])));
#undef KM
  uint32_t lb1b, lEb;
  asm("mad.lo.u32 %0, %1, 4, %2;" : "=r"(lb1b) : "r"(lm), "r"(__float_as_uint(b[1])));
  asm("mad.lo.u32 %0, %1, 4, %2;" : "=r"(lEb) : "r"(lm), "r"(__float_as_uint(E)));
  const float lb1 = __uint_as_float(lb1b), lE = __uint_as_float(lEb);
  const float W3 = fmaf(b[3], v3, E);
  const float W1 = fmaf(lb1, v1, lE);
  const float W0 = fmaf(b[0], v0, E);
  const float W2 = fmaf(b[2], v2, E);
  const float W02 = W0 * W2, W13 = W1 * W3;
  b[1] = (lb1 * W3) * W02;
  b[3] = (b[3] * W1) * W02;
  b[0] = (b[0] * W2) * W13;
  b[2] = (b[2] * W0) * W13;
  G = W02 * W13;
  E = SK_EPS * G;
}

// Exact power-of-two rebalance of the two scale vectors.  c = r K s / G is invariant under
// (r, s) -> (r * 2^-k, s * 2^k), a free gauge that lambda (which renormalises only G) does not fix.
// Once per (rows, columns) pair, k = (e(r[1]) - e(s[1])) >> 1 (biased exponents, arithmetic shift)
// is moved from r to s as integer adds on the exponent fields; afterwards |e(r[1]) - e(s[1])| <= 1.
// Power-of-two scaling is exact and fp32 rounding is scale-equivariant in the normal range, so c, G
// and every later rounding (hence comb_mix) are bit-identical to the un-rebalanced recursion.  What
// changes is that r and s stay at a provable scale: r_i * s_j = c_ij * G / K_ij is bounded for
// every input, the rebalance splits r_1 * s_1 evenly between the two vectors, and one half-step
// moves a vector by at most 12 binades, so every r and s stays within (2^-64, 2^50) (normal fp32
// range: [2^-126, 2^128)).
//
// s[1] is the last value of the column half-step, so k is formed from it with one subtract and one
// shift: with r[1]'s mantissa forced to all ones, (bits(r1) | 0x7fffff) - bits(s1) =
// (e(r1) - e(s1)) * 2^23 + delta with delta in [0, 2^23), and the arithmetic shift by 24 floors it
// to (e(r1) - e(s1)) >> 1 (r, s > 0: no sign bits).  The exponent adds are mad.lo.u32 (one
// shift-add each, like lambda in hs_lam).
__device__ __forceinline__ void rebalance(float r[4], float s[4]) {
  const int k = (int)((__float_as_uint(r[1]) | 0x7fffffu) - __float_as_uint(s[1])) >> 24;
  const uint32_t kp = (uint32_t)k, kn = (uint32_t)(-k);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t rb, sb;
    asm("mad.lo.u32 %0, %1, 0x800000, %2;" : "=r"(rb) : "r"(kn), "r"(__float_as_uint(r[i])));
    asm("mad.lo.u32 %0, %1, 0x800000, %2;" : "=r"(sb) : "r"(kp), "r"(__float_as_uint(s[i])));
    r[i] = __uint_as_float(rb);
    s[i] = __uint_as_float(sb);
  }
}

// NS split-K partials of one float4 logit row (+ the sqrsum partials).  Lane group g = lane >> 2
// loads partial s = g (one 16 B row + one float): a single load per lane, issued right after the
// kernel prologue (the code in front of the first load is fetched cold when the L2 is cold, so it
// is kept short). NS > 8 also issues s = g + 8k (k < PU) up front and a tail loop covers NS > 64;
// reduce_partials adds the per-lane partials and then the 8 lane groups with an xor butterfly.
struct Partials {
  float4 v[PU];
  float w[PU];
};
__device__ __forceinline__ void issue_partials(Partials& p, const float4* m0, const float* s0,
                                               int mstride, int T, int NS, int g) {
  p.v[0] = make_float4(0.f, 0.f, 0.f, 0.f);
  p.w[0] = 0.f;
  if (g < NS) {
    p.v[0] = ld_f32x4(m0 + g * mstride);
    p.w[0] = ld_f32(s0 + g * T);
  }
}
// NS > 8 only: s = g + 8k, k = 1..PU-1, issued by each role after its main loads.
__device__ __forceinline__ void issue_tail(Partials& p, const float4* m0, const float* s0,
                                           int mstride, int T, int NS, int g) {
  if (__builtin_expect(NS > 8, 0)) {
#pragma unroll
    for (int k = 1; k < PU; ++k) {
      const int s = g + 8 * k;
      p.v[k] = make_float4(0.f, 0.f, 0.f, 0.f);
      p.w[k] = 0.f;
      if (s < NS) {
        p.v[k] = ld_f32x4(m0 + s * mstride);
        p.w[k] = ld_f32(s0 + s * T);
      }
    }
  }
}
// -> logits x[0..3] of the row: (sum * scale) * rms + base
__device__ __forceinline__ float4 reduce_partials(Partials& p, const float4* m0, const float* s0,
                                                  int mstride, int T, int NS, int g, float scale,
                                                  float4 bias) {
  float4 acc = p.v[0];
  float sq = p.w[0];
  if (__builtin_expect(NS > 8, 0)) {
#pragma unroll
    for (int h = PU / 2; h >= 1; h >>= 1) {
#pragma unroll
      for (int k = 0; k < h; ++k) {
        p.v[k + 1] =
            add4(p.v[k + 1], h + k + 1 < PU ? p.v[k + 1 + h] : make_float4(0.f, 0.f, 0.f, 0.f));
        p.w[k + 1] = p.w[k + 1] + (h + k + 1 < PU ? p.w[k + 1 + h] : 0.f);
      }
    }
    acc = add4(acc, p.v[1]);
    sq = sq + p.w[1];
#pragma unroll 1
    for (int s = g + 8 * PU; s < NS; s += 8) {  // NS > 64 only
      acc = add4(acc, ld_f32x4(m0 + s * mstride));
      sq = sq + ld_f32(s0 + s * T);
    }
  }
#pragma unroll
  for (int m = 4; m <= 16; m <<= 1) {
    acc = add4(acc, shfl_xor4(acc, m));
    sq = sq + __shfl_xor_sync(0xffffffffu, sq, m);
  }
  const float rms = rsqrt_ftz(sq * INV_RMS_NUMEL + RMS_EPS);
  return make_float4(fmaf(acc.x * scale, rms, bias.x), fmaf(acc.y * scale, rms, bias.y),
                     fmaf(acc.z * scale, rms, bias.z), fmaf(acc.w * scale, rms, bias.w));
}

// CL CTAs per token (thread-block cluster); each collapse thread owns NV 16-byte chunks per stream.
template <int CL>
__global__ __launch_bounds__(NTHREADS, 1) void __cluster_dims__(CL, 1, 1)
    mhc_pre_cluster_kernel(const float* __restrict__ gemm_out_mul,
                           const float* __restrict__ gemm_out_sqrsum,
                           const float* __restrict__ hc_scale, const float* __restrict__ hc_base,
                           const __nv_bfloat16* __restrict__ residual,
                           const __nv_bfloat16* __restrict__ norm_weight,
                           float* __restrict__ post_mix, float* __restrict__ comb_mix,
                           __nv_bfloat16* __restrict__ layer_input, int T, int NS) {
  constexpr int CH = H / CL;            // channels per CTA
  constexpr int NV = CH / (NCT * VEC);  // 16 B chunks per stream per collapse thread
  constexpr int NSLOT = CL * NWC;       // sum(ol^2) partials per token
  static_assert(CL <= CL_MAX && NV * NCT * VEC == CH, "bad cluster size");
  __shared__ float red[NSLOT];
  __shared__ __align__(8) uint64_t mbar;
  __shared__ float4 sK[4];  // Sinkhorn warp: K rows
  __shared__ float4 sS;     // Sinkhorn warp: softmax row sums

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int j4 = lane & 3;
  const int g = lane >> 2;
  const int t = blockIdx.y;
  const uint32_t rank = cluster_ctarank();
  // warp 0: control (rank 1 = comb / Sinkhorn, rank 2 = post, ranks 0 and 3 idle); warps 1..NWC:
  // collapse.  Warp 0 of block 0 is routinely the last warp of the grid to start, so it carries no
  // work.
  constexpr uint32_t POST_RANK = CL > 2 ? 2 : 0;  // CL = 2: post on rank 0 (short path)
  const bool collapse = warp > 0;
  // idle warp-0s leave after the shared load prologue (an early exit would serialize the parameter
  // loads behind the S2R latency of the role test)
  const bool idle = !collapse && rank != 1 && rank != POST_RANK;
  const bool sink = !collapse && rank == 1;
  // logit row of this warp's role: pre (collapse), comb row j4 (rank 1 control), post
  // (POST_RANK control)
  const int fidx = collapse ? 0 : (sink ? 2 + j4 : 1);
  const float4* mrow = reinterpret_cast<const float4*>(gemm_out_mul) + t * MIX4 + fidx;
  const float* sp = gemm_out_sqrsum + t;
  const int mstride = T * MIX4;  // float4 units
  // collapse chunk c of this thread: channels [ch0 + c*NCT*VEC, +VEC)
  const size_t ch0 = (size_t)rank * CH + (size_t)(tid & (NCT - 1)) * VEC;

  // PDL: no global reads before the producer grid is complete.
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // ---- logit partials of every role first, straight after the prologue (one DRAM round trip) ----
  Partials p;
  issue_partials(p, mrow, sp, mstride, T, NS, g);
  const float scale = ld_f32(hc_scale + (collapse ? 0 : (sink ? 2 : 1)));
  const float4 bias = ld_f32x4(reinterpret_cast<const float4*>(hc_base) + fidx);
  if (idle) return;
  if (!collapse) {
    // ================= control warp: rank 1 = comb (Sinkhorn), rank 2 = post =================
    issue_tail(p, mrow, sp, mstride, T, NS, g);
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    if (sink) {
      // ---------------- comb: softmax (lane j4 = row) + division-free Sinkhorn ----------------
      const float4 x = reduce_partials(p, mrow, sp, mstride, T, NS, g, scale, bias);
      const float m = fmaxf(fmaxf(x.x, x.y), fmaxf(x.z, x.w));
      const float e0 = expf(x.x - m), e1 = expf(x.y - m), e2 = expf(x.z - m), e3 = expf(x.w - m);
      const float Sr = (e0 + e1) + (e2 + e3);
      // K = S * (softmax + eps) = e + eps * S   (softmax denominator S deferred into r and G)
      const float k0 = fmaf(SK_EPS, Sr, e0), k1 = fmaf(SK_EPS, Sr, e1);
      const float k2 = fmaf(SK_EPS, Sr, e2), k3 = fmaf(SK_EPS, Sr, e3);
      if (lane < 4) {
        sK[lane] = make_float4(k0, k1, k2, k3);
        reinterpret_cast<float*>(&sS)[lane] = Sr;
      }
      __syncwarp();
      float K[4][4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float4 q = sK[r];
        K[r][0] = q.x;
        K[r][1] = q.y;
        K[r][2] = q.z;
        K[r][3] = q.w;
      }
      const float4 S = sS;
      // r_i = prod_{l != i} S_l, G = prod S: c = r K s / G = softmax + eps (s = 1)
      const float S01 = S.x * S.y, S23 = S.z * S.w;
      float rv[4] = {S.y * S23, S.x * S23, S.w * S01, S.z * S01};
      float sv[4];
      float G, E;
      {
        // half-step 0 (columns), s = 1; scale reset from P itself (kappa_direct)
        const float E0 = SK_EPS * (S01 * S23);
        float W[4];
#pragma unroll
        for (int k = 0; k < 4; ++k)
          W[k] = fmaf(K[3][k], rv[3],
                      fmaf(K[2][k], rv[2], fmaf(K[1][k], rv[1], fmaf(K[0][k], rv[0], E0))));
        const float W02 = W[0] * W[2], W13 = W[1] * W[3];
        const float kap = kappa_direct(W02 * W13);
        const float kW02 = kap * W02, kW13 = kap * W13;
        sv[0] = W[2] * kW13;
        sv[2] = W[0] * kW13;
        sv[1] = W[3] * kW02;
        sv[3] = W[1] * kW02;
        G = kW02 * W13;
        E = SK_EPS * G;
      }
      // 19 x (rows, columns): 39 half-steps in total, no early exit (1 peeled + 9 x 2)
      hs_lam<false>(K, sv, rv, G, E);
      hs_lam<true>(K, rv, sv, G, E);
      rebalance(rv, sv);
#pragma unroll 1
      for (int it = 0; it < 9; ++it) {
        hs_lam<false>(K, sv, rv, G, E);
        hs_lam<true>(K, rv, sv, G, E);
        rebalance(rv, sv);
        hs_lam<false>(K, sv, rv, G, E);
        hs_lam<true>(K, rv, sv, G, E);
        rebalance(rv, sv);
      }
      // c = r K s / G: one Newton-refined reciprocal; lane i < 4 writes row i (its own K row).
      // The numerators (r_i K_ij) s_j are formed while the reciprocal is in flight.
      const float rr = (lane & 2) ? ((lane & 1) ? rv[3] : rv[2]) : ((lane & 1) ? rv[1] : rv[0]);
      const float n0 = (rr * k0) * sv[0], n1 = (rr * k1) * sv[1];
      const float n2 = (rr * k2) * sv[2], n3 = (rr * k3) * sv[3];
      const float invG = rcp_refined(G);
      if (lane < 4)
        reinterpret_cast<float4*>(comb_mix + (size_t)t * 16)[lane] =
            make_float4(n0 * invG, n1 * invG, n2 * invG, n3 * invG);
    } else {
      // ---------------- post (rank 2): lane j4 -> post_mix[t, j4] ----------------
      const float4 x = reduce_partials(p, mrow, sp, mstride, T, NS, g, scale, bias);
      const float pv = sigmoid_ieee(sel4(x, j4)) * POST_MULT;
      if (lane < 4) post_mix[(size_t)t * 4 + lane] = pv;
    }
  } else {
    // ================= collapse warps: CH channels of this CTA =================
    uint4 rv[NV][HC], wv[NV];
#pragma unroll
    for (int c = 0; c < NV; ++c) {
#pragma unroll
      for (int hc = 0; hc < HC; ++hc)
        rv[c][hc] = ld_u32x4(residual + ((size_t)t * HC + hc) * H + ch0 + c * NCT * VEC);
      wv[c] = ld_u32x4(norm_weight + ch0 + c * NCT * VEC);
    }
    issue_tail(p, mrow, sp, mstride, T, NS, g);
    // mbarrier for the sum(ol^2) all-gather; its init is published to the cluster by the fence +
    // the cluster barrier that every sender waits on before its first st.async.
    const uint32_t mbar_addr = smem_u32(&mbar);
    if (tid == 32) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mbar_addr) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mbar_addr),
                   "r"((uint32_t)(NSLOT * sizeof(float)))
                   : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    // The cluster wait (every remote mbarrier initialised before our first st.async) is taken here,
    // while the loads are still in flight, instead of on the critical path after the collapse.
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    // every lane holds the whole pre row after the butterfly: 4 independent sigmoids, no shuffles
    const float4 x = reduce_partials(p, mrow, sp, mstride, T, NS, g, scale, bias);
    const float pre0 = sigmoid_nr(x.x) + HC_PRE_EPS;
    const float pre1 = sigmoid_nr(x.y) + HC_PRE_EPS;
    const float pre2 = sigmoid_nr(x.z) + HC_PRE_EPS;
    const float pre3 = sigmoid_nr(x.w) + HC_PRE_EPS;

    float ob[NV][VEC];
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < NV; ++c) {
      const uint32_t* r0 = reinterpret_cast<const uint32_t*>(&rv[c][0]);
      const uint32_t* r1 = reinterpret_cast<const uint32_t*>(&rv[c][1]);
      const uint32_t* r2 = reinterpret_cast<const uint32_t*>(&rv[c][2]);
      const uint32_t* r3 = reinterpret_cast<const uint32_t*>(&rv[c][3]);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        float o_lo = pre0 * bf_lo(r0[k]);
        o_lo = fmaf(pre1, bf_lo(r1[k]), o_lo);
        o_lo = fmaf(pre2, bf_lo(r2[k]), o_lo);
        o_lo = fmaf(pre3, bf_lo(r3[k]), o_lo);
        float o_hi = pre0 * bf_hi(r0[k]);
        o_hi = fmaf(pre1, bf_hi(r1[k]), o_hi);
        o_hi = fmaf(pre2, bf_hi(r2[k]), o_hi);
        o_hi = fmaf(pre3, bf_hi(r3[k]), o_hi);
        ss = fmaf(o_lo, o_lo, ss);
        ss = fmaf(o_hi, o_hi, ss);
        const uint32_t pb = pack_bf16x2(o_lo, o_hi);  // bf16(ol); statistic stays fp32
        ob[c][2 * k] = bf_lo(pb);
        ob[c][2 * k + 1] = bf_hi(pb);
      }
    }
#pragma unroll
    for (int m = 16; m >= 1; m >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, m);

    // bf16(ol) * weight, formed while the all-gather is in flight (layer = (ob * w) * r)
#pragma unroll
    for (int c = 0; c < NV; ++c) {
      const uint32_t* wp = reinterpret_cast<const uint32_t*>(&wv[c]);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        ob[c][2 * k] *= bf_lo(wp[k]);
        ob[c][2 * k + 1] *= bf_hi(wp[k]);
      }
    }
    // all-gather of the NSLOT partials over DSMEM
    if (lane < CL) {
      const uint32_t dst = mapa(smem_u32(&red[rank * NWC + warp - 1]), (uint32_t)lane);
      const uint32_t dbar = mapa(mbar_addr, (uint32_t)lane);
      asm volatile(
          "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(dst),
          "r"(__float_as_uint(ss)), "r"(dbar)
          : "memory");
    }
    {
      uint32_t done = 0;
      do {
        asm volatile(
            "{\n .reg .pred p;\n"
            " mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
            " selp.u32 %0, 1, 0, p;\n}"
            : "=r"(done)
            : "r"(mbar_addr)
            : "memory");
      } while (!done);
    }
    float part[NSLOT];
#pragma unroll
    for (int k = 0; k < NSLOT; ++k) part[k] = red[k];
#pragma unroll
    for (int h = NSLOT / 2; h >= 1; h >>= 1)
#pragma unroll
      for (int k = 0; k < h; ++k) part[k] = part[k] + part[k + h];
    const float r = rsqrt_ftz(part[0] * INV_H + NORM_EPS);

#pragma unroll
    for (int c = 0; c < NV; ++c) {
      uint4 out;
      uint32_t* op = reinterpret_cast<uint32_t*>(&out);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) op[k] = pack_bf16x2(ob[c][2 * k] * r, ob[c][2 * k + 1] * r);
      *reinterpret_cast<uint4*>(layer_input + (size_t)t * H + ch0 + c * NCT * VEC) = out;
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

// One launch per call on the caller's stream. The cluster shape is compiled into the kernel
// (__cluster_dims__), so only the programmatic-dependent-launch attribute is set here: the kernel
// waits (griddepcontrol.wait) before its first global read and triggers its dependents after its
// last global write, so the attribute is safe either way.
template <int CL>
void launch_cl(const float* gemm_out_mul, const float* gemm_out_sqrsum, const float* hc_scale,
               const float* hc_base, const __nv_bfloat16* residual,
               const __nv_bfloat16* norm_weight, float* post_mix, float* comb_mix,
               __nv_bfloat16* layer_input, int T, int NS, bool enable_pdl, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CL, T, 1);
  cfg.blockDim = dim3(NTHREADS, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, mhc_pre_cluster_kernel<CL>, gemm_out_mul, gemm_out_sqrsum, hc_scale,
                     hc_base, residual, norm_weight, post_mix, comb_mix, layer_input, T, NS);
}

void launch_cluster_kernel(const float* gemm_out_mul, const float* gemm_out_sqrsum,
                           const float* hc_scale, const float* hc_base,
                           const __nv_bfloat16* residual, const __nv_bfloat16* norm_weight,
                           float* post_mix, float* comb_mix, __nv_bfloat16* layer_input, int T,
                           int NS, bool enable_pdl, cudaStream_t stream) {
  // 4 CTAs (one thread-block cluster) per token while that is at most 96 CTAs; beyond that
  // 2 CTAs per token (fewer clusters to place at T = 32, equal time at T = 4..24).
  if (4 * T <= 96) {
    launch_cl<4>(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, norm_weight, post_mix,
                 comb_mix, layer_input, T, NS, enable_pdl, stream);
  } else {
    launch_cl<2>(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, norm_weight, post_mix,
                 comb_mix, layer_input, T, NS, enable_pdl, stream);
  }
}

}  // namespace

// Destination-passing entry point. Shapes (all contiguous, one CUDA device):
//   post_mix fp32 [T, 4], comb_mix fp32 [T, 16] or [T, 4, 4], layer_input bf16 [T, 4096],
//   gemm_out_mul fp32 [NS, T, 24], gemm_out_sqrsum fp32 [NS, T], hc_scale fp32 [3],
//   hc_base fp32 [24], residual bf16 [T, 4, 4096], norm_weight bf16 [4096].
// gemm_out_mul, hc_base, residual, norm_weight, comb_mix and layer_input are accessed with
// 16-byte vectors. The epsilons / post multiplier / Sinkhorn iteration count are compile-time
// constants of this kernel; the Python support check only dispatches calls whose values match.
void mhc_pre_big_fuse_with_norm_cluster(TensorView post_mix, TensorView comb_mix,
                                        TensorView layer_input, TensorView gemm_out_mul,
                                        TensorView gemm_out_sqrsum, TensorView hc_scale,
                                        TensorView hc_base, TensorView residual,
                                        TensorView norm_weight, bool enable_pdl) {
  CHECK_INPUT_AND_TYPE(post_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(comb_mix, dl_float32);
  CHECK_INPUT_AND_TYPE(layer_input, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(gemm_out_mul, dl_float32);
  CHECK_INPUT_AND_TYPE(gemm_out_sqrsum, dl_float32);
  CHECK_INPUT_AND_TYPE(hc_scale, dl_float32);
  CHECK_INPUT_AND_TYPE(hc_base, dl_float32);
  CHECK_INPUT_AND_TYPE(residual, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(norm_weight, dl_bfloat16);
  CHECK_DEVICE(residual, post_mix);
  CHECK_DEVICE(residual, comb_mix);
  CHECK_DEVICE(residual, layer_input);
  CHECK_DEVICE(residual, gemm_out_mul);
  CHECK_DEVICE(residual, gemm_out_sqrsum);
  CHECK_DEVICE(residual, hc_scale);
  CHECK_DEVICE(residual, hc_base);
  CHECK_DEVICE(residual, norm_weight);
  CHECK_DIM(3, gemm_out_mul);
  CHECK_DIM(2, gemm_out_sqrsum);
  CHECK_DIM(3, residual);
  CHECK_DIM(1, norm_weight);
  CHECK_DIM(1, hc_scale);
  CHECK_DIM(1, hc_base);
  CHECK_DIM(2, post_mix);
  CHECK_DIM(2, layer_input);

  const int64_t NS = gemm_out_mul.size(0);
  const int64_t T = gemm_out_mul.size(1);
  TVM_FFI_ICHECK_EQ(gemm_out_mul.size(2), MIX) << "gemm_out_mul.shape[2] must be 24";
  TVM_FFI_ICHECK_GE(NS, 1) << "gemm_out_mul.shape[0] (split count) must be >= 1";
  TVM_FFI_ICHECK_EQ(gemm_out_sqrsum.size(0), NS) << "gemm_out_sqrsum.shape[0] must match";
  TVM_FFI_ICHECK_EQ(gemm_out_sqrsum.size(1), T) << "gemm_out_sqrsum.shape[1] must match";
  TVM_FFI_ICHECK_EQ(residual.size(0), T) << "residual.shape[0] must match";
  TVM_FFI_ICHECK_EQ(residual.size(1), HC) << "residual.shape[1] must be 4";
  TVM_FFI_ICHECK_EQ(residual.size(2), H) << "residual.shape[2] must be 4096";
  TVM_FFI_ICHECK_EQ(norm_weight.size(0), H) << "norm_weight must have shape [4096]";
  TVM_FFI_ICHECK_EQ(hc_scale.size(0), 3) << "hc_scale must have shape [3]";
  TVM_FFI_ICHECK_EQ(hc_base.size(0), MIX) << "hc_base must have shape [24]";
  TVM_FFI_ICHECK_EQ(post_mix.size(0), T) << "post_mix.shape[0] must match";
  TVM_FFI_ICHECK_EQ(post_mix.size(1), HC) << "post_mix.shape[1] must be 4";
  TVM_FFI_ICHECK_EQ(comb_mix.numel(), T * HC * HC) << "comb_mix must hold T * 16 elements";
  TVM_FFI_ICHECK_EQ(comb_mix.size(0), T) << "comb_mix.shape[0] must match";
  TVM_FFI_ICHECK_EQ(layer_input.size(0), T) << "layer_input.shape[0] must match";
  TVM_FFI_ICHECK_EQ(layer_input.size(1), H) << "layer_input.shape[1] must be 4096";
  TVM_FFI_ICHECK_LE(T, 65535) << "token count exceeds the grid limit";

  if (T == 0) {
    return;
  }

  ffi::CUDADeviceGuard device_guard(residual.device().device_id);
  auto stream = get_stream(residual.device());
  launch_cluster_kernel(
      static_cast<const float*>(gemm_out_mul.data_ptr()),
      static_cast<const float*>(gemm_out_sqrsum.data_ptr()),
      static_cast<const float*>(hc_scale.data_ptr()), static_cast<const float*>(hc_base.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr()),
      static_cast<float*>(post_mix.data_ptr()), static_cast<float*>(comb_mix.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(layer_input.data_ptr()), static_cast<int>(T),
      static_cast<int>(NS), enable_pdl, stream);
  cudaError_t status = cudaPeekAtLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "mhc_pre_big_fuse_with_norm_cluster kernel launch failed with error code "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    mhc_pre_big_fuse_with_norm_cluster,
    flashinfer::mhc::pre_norm_cluster::mhc_pre_big_fuse_with_norm_cluster);

}  // namespace flashinfer::mhc::pre_norm_cluster
