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
#pragma once

// Fused mHC post -> pre (+ input RMSNorm) for hidden size 4096 and four residual streams,
// one-pass thread-block-cluster kernels for SM103.
//
// Loads/stores of the big tiles use TMA tensor maps (one box per tensor per tile).
// One cluster of 16 CTAs owns a tile of up to 16 tokens; CTA `r` owns hidden slice
// [r*256, r*256+256) of all 4 streams (1024 of the 16384 projection columns).  Its slice
// of `fn` (24 x 1024 fp32 = 96 KB) lives in registers as TF32 MMA B-fragments for the whole
// persistent kernel.  Per tile:
//   A : post-map (fp32 FMA) -> bf16 residual_cur in smem (bulk-stored to HBM), bf16 sum-sq,
//       TF32 mma.sync projection of the bf16 values -> per-CTA partial mixes -> st.async to
//       the token owner CTA (cluster rank == token index inside the tile).
//   F0: owner reduces the 16 partials in fixed order, RMS-scales, pre-mix sigmoid, broadcasts
//       pre_mix to all 16 CTAs, writes post_mix and queues comb logits for the Sinkhorn warp.
//   B1: pre-mix weighted sum of the bf16 streams (fp32), sum of squares of the pre-round fp32
//       value -> all-to-all partials.
//   B2: full RMS statistic -> layer_input = bf16(bf16(sum) * rsqrt * w).
//   F1: 20-iteration Sinkhorn (4 lanes per token) off the critical path.
// The decode kernel runs A(k), B2(k-2), B1(k-1) so every cross-CTA wait is covered by work;
// the stream kernels (several tiles per cluster) are warp-specialized (see each .cu file).
//
// Numerics: no fast-math. expf is the accurate libdevice expf, every division is the IEEE
// div.rn.f32 result (either '/' or the identical MUFU.RCP + Newton + remainder-correction
// sequence inside its verified operand range), rsqrtf is rsqrt.approx.f32, the projection is
// TF32 (fn rounded to nearest, bf16 residual_cur exact) with FP32 accumulation.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "mhc_fused_post_pre.h"

namespace flashinfer {
namespace mhc_fpp {

constexpr int H = 4096;
constexpr int NMIX = 24;
constexpr int CL = 16;     // cluster size (non-portable)
constexpr int W = H / CL;  // 256 hidden per CTA
constexpr int TMAX = 16;   // tokens per tile
constexpr int S = 1;       // decode owns exactly one tile / stage
constexpr int NCW = 8;     // compute warps
constexpr int NCT = NCW * 32;
constexpr int WARP_PROD = 8;
constexpr int WARP_F0 = 9;
constexpr int WARP_F1 = 10;
constexpr int NTHREADS = 11 * 32;

constexpr int ROWB = W * 2;  // 512 B per (token, stream) row
// Stage layout (dense TMA boxes): x [16][256] bf16, residual [16][4][256] bf16, plm, comb.
constexpr int X_OFF = 0;
constexpr int R_OFF = X_OFF + TMAX * ROWB;
constexpr int PLM_OFF = R_OFF + TMAX * 4 * ROWB;
constexpr int COMB_OFF = PLM_OFF + TMAX * 16;
constexpr int STAGEB = ((COMB_OFF + TMAX * 64 + 1023) / 1024) * 1024;

constexpr int G1_REC = 28;  // floats per (sender) record: 24 mixes + sumsq
constexpr int OFF_G1 = S * STAGEB;
constexpr int OFF_PM = OFF_G1 + S * CL * G1_REC * 4;
constexpr int OFF_G2 = OFF_PM + S * TMAX * 4 * 4;
constexpr int OFF_RED = OFF_G2 + S * TMAX * CL * 4;
constexpr int OFF_SSQ = OFF_RED + NCW * TMAX * NMIX * 4;
constexpr int QN = 32;
constexpr int DQN = 1;
constexpr int QREC = 20;
constexpr int OFF_Q = OFF_SSQ + 2 * TMAX * 4;
constexpr int OFF_BAR = OFF_Q + DQN * QREC * 4;
constexpr int B_FULL = 0, B_EMPTY = S, B_G1 = 2 * S, B_PM = 3 * S, B_G2 = 4 * S;
constexpr int B_QF = 5 * S, B_QE = 5 * S + DQN;
constexpr int NBAR = 5 * S + 2 * DQN;
constexpr int SMEM_BYTES = OFF_BAR + NBAR * 8;
static_assert(SMEM_BYTES <= 227 * 1024, "smem");
static_assert(OFF_G1 % 16 == 0 && OFF_PM % 16 == 0 && OFF_G2 % 16 == 0 && OFF_RED % 16 == 0 &&
                  OFF_Q % 16 == 0 && OFF_BAR % 8 == 0,
              "align");

constexpr float RMS_EPS = 1.0e-5f;
constexpr float NORM_EPS = 1.0e-5f;
constexpr float HC_PRE_EPS = 1.0e-6f;
constexpr float HC_SINKHORN_EPS = 1.0e-6f;
constexpr float HC_POST_MULT = 2.0f;
constexpr int SINKHORN_REPEAT = 20;

struct Params {
  const __nv_bfloat16* x;
  const __nv_bfloat16* res;
  const float* plm;
  const float* comb;
  const float* fn;
  const float* hc_scale;
  const float* hc_base;
  const __nv_bfloat16* nw;
  __nv_bfloat16* rc_out;
  float* post_out;
  float* comb_out;
  __nv_bfloat16* li_out;
  int M;
  int TM;
  int ntiles;
};

// ------------------------------------------------------------------------------------------
// PTX helpers
// ------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
               "r"(tx)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
// CTA-scope acquire: every object we wait for (TMA tiles, st.async partials from peer CTAs) lands
// in this CTA's shared memory, which L1 never caches, so a cluster-scope acquire (which costs an
// L1 invalidate, CCTL.IVALL, per wait) buys nothing here.
__device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}\n"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_try_wait(bar, parity)) {
  }
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
      "selp.u32 %0, 1, 0, p;\n\t}\n"
      : "=r"(ok)
      : "r"(bar), "r"(parity)
      : "memory");
  return ok != 0;
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
__device__ __forceinline__ void cluster_sync_init() {
  asm volatile(
      "barrier.cluster.arrive.relaxed.aligned;\n\t"
      "barrier.cluster.wait.aligned;" ::
          : "memory");
}
__device__ __forceinline__ void cluster_arrive() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ void st_async_v4(uint32_t raddr, float a, float b, float c, float d,
                                            uint32_t rbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, "
      "[%5];" ::"r"(raddr),
      "r"(__float_as_uint(a)), "r"(__float_as_uint(b)), "r"(__float_as_uint(c)),
      "r"(__float_as_uint(d)), "r"(rbar)
      : "memory");
}
__device__ __forceinline__ void st_async_b32(uint32_t raddr, float a, uint32_t rbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];" ::"r"(raddr),
      "r"(__float_as_uint(a)), "r"(rbar)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes,
                                         uint32_t bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
      "[%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tm, int c0, int c1,
                                            uint32_t bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_"
      "hint "
      "[%0], [%1, {%2, %3}], [%4], %5;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(c0), "r"(c1), "r"(bar), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* tm, int c0, int c1,
                                            int c2, uint32_t bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_"
      "hint "
      "[%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tm)), "r"(c0), "r"(c1), "r"(c2), "r"(bar), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_store_3d(const CUtensorMap* tm, int c0, int c1, int c2,
                                             uint32_t src) {
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%1, %2, %3}], [%4];" ::"l"(
          reinterpret_cast<uint64_t>(tm)),
      "r"(c0), "r"(c1), "r"(c2), "r"(src)
      : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dst), "r"(src),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void bulk_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void bulk_wait_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
template <int N>
__device__ __forceinline__ void bulk_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}
__device__ __forceinline__ void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
__device__ __forceinline__ void prefetch_tmap(const CUtensorMap* tm) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(tm)) : "memory");
}
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void compute_bar() {
  asm volatile("bar.sync 1, %0;" ::"n"(NCT) : "memory");
}
__device__ __forceinline__ uint32_t f2tf32(float x) {
  // Round-to-nearest (ties away) onto the 10-bit TF32 mantissa, same result as cvt.rna.tf32.
  return (__float_as_uint(x) + 0x1000u) & 0xffffe000u;
}
__device__ __forceinline__ void mma_tf32(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 256-bit read-only global load (8 consecutive fp32, 32 B aligned).
__device__ __forceinline__ void ldg_v8(const float* src, uint32_t (&d)[8]) {
  asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]),
                 "=r"(d[7])
               : "l"(src));
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack_bf2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float sigmoidf_ref(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float2 bf2_to_f2(uint32_t w) { return make_float2(bf_lo(w), bf_hi(w)); }
__device__ __forceinline__ float2 f2s(float v) { return make_float2(v, v); }

// IEEE-754 round-to-nearest division x / y.  This is the exact FMA sequence nvcc emits for
// div.rn.f32 (MUFU.RCP, one Newton step on the reciprocal, quotient + one remainder correction);
// nvcc only leaves this path (FCHK -> slow-path call) for denormal / huge-exponent operands.
// Sharing the refined reciprocal of one denominator across several numerators is bit-identical
// to dividing each numerator separately.  Callers verify the operand range with div_ok() and
// fall back to the compiler's '/' if any operand is outside it.
__device__ __forceinline__ float rcp_refined(float y) {
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(y));
  const float e = __fmaf_rn(-y, r, 1.0f);
  return __fmaf_rn(r, e, r);
}
__device__ __forceinline__ float div_rn_rcp(float x, float y, float r) {
  const float q = __fmaf_rn(x, r, 0.0f);
  const float rem = __fmaf_rn(-y, q, x);
  return __fmaf_rn(r, rem, q);
}
// |v| in [2^-60, 2^60): far from every FCHK slow-path case (denormals, inf/nan, exponent
// overflow of the quotient or of the reciprocal).
__device__ __forceinline__ bool div_ok(float v) {
  return ((__float_as_uint(v) >> 23) & 0xffu) - 67u < 120u;
}

// 1 / (1 + expf(-x)) with the division done by the exact div.rn FMA sequence when the
// denominator is in the verified range (always, unless x < -41), '/' otherwise.
__device__ __forceinline__ float sigmoid_exact(float x) {
  const float d = 1.0f + expf(-x);
  if (div_ok(d)) return div_rn_rcp(1.0f, d, rcp_refined(d));
  return 1.0f / d;
}

// 20-iteration Sinkhorn on a 4x4 matrix; 4 consecutive lanes hold one matrix, lane = row.
template <bool kExact, bool kShareColRcp>
__device__ __forceinline__ void sinkhorn_rows(float (&c)[4], bool& bad) {
  {
    const float mx = fmaxf(fmaxf(c[0], c[1]), fmaxf(c[2], c[3]));
    float e[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) e[kk] = expf(c[kk] - mx);
    const float sm = (e[0] + e[1]) + (e[2] + e[3]);
    if (kExact) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) c[kk] = e[kk] / sm + HC_SINKHORN_EPS;
    } else {
      const float r = rcp_refined(sm);
      bad |= !div_ok(sm);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        bad |= !div_ok(e[kk]);
        c[kk] = div_rn_rcp(e[kk], sm, r) + HC_SINKHORN_EPS;
      }
    }
  }
  auto colnorm = [&]() {
    // Keep the two-step per-column shuffle/add tree in every row lane.  One lane then
    // owns each column's identical refined reciprocal and broadcasts it to the subgroup.
    // This keeps every denominator and corrected quotient bit-identical while cutting the
    // reciprocal refinement work 4x.
    float den[4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      float cs = c[kk] + __shfl_xor_sync(0xffffffffu, c[kk], 1);
      cs = cs + __shfl_xor_sync(0xffffffffu, cs, 2);
      den[kk] = cs + HC_SINKHORN_EPS;
    }
    // No per-step range checks: after the softmax + eps step every entry is in
    // [1e-6, 1 + 1e-6] and every denominator in [1e-6, 4 + 5e-6], so a normalization shrinks
    // an entry by at most ~4x; over the 39 normalizations every operand stays a normal float
    // >= 2^-98 and div_rn_rcp is the exact IEEE quotient.  The softmax division (exp may
    // underflow) keeps its div_ok() guard and '/' fallback.
    const int col = threadIdx.x & 3;
    const float owned_den =
        (col == 0) ? den[0] : ((col == 1) ? den[1] : ((col == 2) ? den[2] : den[3]));
    float owned_rcp = 0.0f;
    if (!kExact && kShareColRcp) owned_rcp = rcp_refined(owned_den);
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      if (kExact) {
        c[kk] = c[kk] / den[kk];
      } else if (kShareColRcp) {
        const float r = __shfl_sync(0xffffffffu, owned_rcp, kk, 4);
        c[kk] = div_rn_rcp(c[kk], den[kk], r);
      } else {
        c[kk] = div_rn_rcp(c[kk], den[kk], rcp_refined(den[kk]));
      }
    }
  };
  colnorm();
#pragma unroll 1
  for (int rep = 0; rep < SINKHORN_REPEAT - 1; ++rep) {
    const float den = ((c[0] + c[1]) + (c[2] + c[3])) + HC_SINKHORN_EPS;
    if (kExact) {
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) c[kk] = c[kk] / den;
    } else {
      const float r = rcp_refined(den);
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) c[kk] = div_rn_rcp(c[kk], den, r);
    }
    colnorm();
  }
}

// Single-token decode path: one lane owns each matrix element.  The row/column
// trees match sinkhorn_rows(), but the warp issues one quotient per step instead
// of four serial quotients in each of four row lanes.
template <bool kExact, unsigned MASK16 = 0x0000ffffu>
__device__ __forceinline__ float sinkhorn_element(float c, bool& bad) {
  // lane = 4 * row + col.  Every reduction gathers its three partners with independent
  // shuffles and adds (own + xor1) + (xor2 + xor3) (resp. xor 4/8/12 for columns): the same
  // tree, hence the same bits, as the two-step butterfly, at one shuffle latency per step.
  auto rsum = [&](float v) {
    const float a = __shfl_xor_sync(MASK16, v, 1);
    const float b = __shfl_xor_sync(MASK16, v, 2);
    const float d = __shfl_xor_sync(MASK16, v, 3);
    return (v + a) + (b + d);
  };
  auto csum = [&](float v) {
    const float a = __shfl_xor_sync(MASK16, v, 4);
    const float b = __shfl_xor_sync(MASK16, v, 8);
    const float d = __shfl_xor_sync(MASK16, v, 12);
    return (v + a) + (b + d);
  };
  {
    const float a = __shfl_xor_sync(MASK16, c, 1);
    const float b = __shfl_xor_sync(MASK16, c, 2);
    const float d = __shfl_xor_sync(MASK16, c, 3);
    const float mx = fmaxf(fmaxf(c, a), fmaxf(b, d));
    const float e = expf(c - mx);
    const float sm = rsum(e);
    if (kExact) {
      c = e / sm + HC_SINKHORN_EPS;
    } else {
      const float r = rcp_refined(sm);
      bad |= !div_ok(sm) || !div_ok(e);
      c = div_rn_rcp(e, sm, r) + HC_SINKHORN_EPS;
    }
  }
  // After the softmax + eps step the 39 normalizations keep every operand a normal float
  // >= 2^-98 (see sinkhorn_rows), so they need no range check.
  auto colnorm = [&]() {
    const float den = csum(c) + HC_SINKHORN_EPS;
    if (kExact)
      c = c / den;
    else
      c = div_rn_rcp(c, den, rcp_refined(den));
  };
  colnorm();
#pragma unroll 1
  for (int rep = 0; rep < SINKHORN_REPEAT - 1; ++rep) {
    const float den = rsum(c) + HC_SINKHORN_EPS;
    if (kExact)
      c = c / den;
    else
      c = div_rn_rcp(c, den, rcp_refined(den));
    colnorm();
  }
  return c;
}

// ------------------------------------------------------------------------------------------
// Host helpers shared by the per-regime translation units
// ------------------------------------------------------------------------------------------
static_assert(kHidden == H && kClusterSize == CL && kSliceWidth == W && kTileTokens == TMAX &&
                  kMixWidth == NMIX,
              "host and device geometry disagree");

inline Params make_params(const LaunchArgs& a) {
  Params p;
  p.x = static_cast<const __nv_bfloat16*>(a.x);
  p.res = static_cast<const __nv_bfloat16*>(a.residual);
  p.plm = a.post_layer_mix;
  p.comb = a.comb_res_mix;
  p.fn = a.fn;
  p.hc_scale = a.hc_scale;
  p.hc_base = a.hc_base;
  p.nw = static_cast<const __nv_bfloat16*>(a.norm_weight);
  p.rc_out = static_cast<__nv_bfloat16*>(a.residual_cur);
  p.post_out = a.post_mix_cur;
  p.comb_out = a.comb_mix_cur;
  p.li_out = static_cast<__nv_bfloat16*>(a.layer_input_cur);
  p.M = a.num_tokens;
  p.TM = a.tile_tokens;
  p.ntiles = a.num_tiles;
  return p;
}

// Opt a cluster kernel into its dynamic shared memory and the non-portable 16-CTA cluster,
// then (optionally) query how many such clusters can be co-resident on the current device.
inline cudaError_t configure_cluster_kernel(const void* kernel, int threads, int smem_bytes,
                                            int* max_clusters) {
  cudaError_t err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (err != cudaSuccess || max_clusters == nullptr) return err;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CL * 64, 1, 1);
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = smem_bytes;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CL;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int n = 0;
  err = cudaOccupancyMaxActiveClusters(&n, kernel, &cfg);
  if (err != cudaSuccess) return err;
  if (n <= 0) return cudaErrorInvalidConfiguration;
  *max_clusters = n;
  return cudaSuccess;
}

// One cluster launch (no other launch attributes) on `stream`.
template <typename Kernel>
inline cudaError_t launch_cluster_kernel(Kernel kernel, int threads, int smem_bytes,
                                         int num_clusters, const Params& p, const TensorMaps& maps,
                                         cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(CL * num_clusters, 1, 1);
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = smem_bytes;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CL;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return cudaLaunchKernelEx(&cfg, kernel, p, maps.x, maps.residual, maps.residual_cur);
}

}  // namespace mhc_fpp
}  // namespace flashinfer
