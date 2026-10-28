/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// MNNVL one-shot all-reduce with a fused mHC post-map + split-K pre-norm partial epilogue.
//
// Replaces the decode-time pair
//   oneshotAllreduceFusionKernel<8, bf16, RMSNORM=false>  ->  mHC post-map + pre-norm GEMM
// at one mHC sublayer boundary (world size 8, hidden 4096, hc_mult 4, bf16, <= 32 tokens).
//
// All-reduce: the one-shot Lamport protocol of flashinfer/comm/trtllm_mnnvl_allreduce.cuh on the
// same workspace (same buffer layout [token][rank][hidden], same flags, same triple-buffer
// rotation and dirty-buffer clearing), so these kernels interleave freely with the stock one-shot /
// two-shot kernels on one workspace. The reduction is the stock deterministic one: fp32
// accumulation in rank order 0..7 of the sanitized bf16 contributions, one bf16 RNE rounding.
//
// Epilogue (per token, on the reduced bf16 x):
//   new_r[j, h]   = post[j] * x[h] + sum_k comb[k, j] * residual[k, h]   (fp32)
//   residual_out  = bf16_rne(new_r)
//   yp[s, m, n]   = sum_{j, h in split s} fn[n, j * H + h] * new_r[j, h]
//   rp[s, m]      = sum_{j, h in split s} new_r[j, h]^2
// yp and rp are fp32, from the unrounded new_r, over 8 contiguous hidden splits of 512: the
// split-K partials the mHC pre-norm finalize consumes.
//
// Four geometries, selected by token count (arMhcPostPrenormDispatch):
//  * row-split kernel (arMhcPostRowSplitKernel, 1-2 tokens): split s of a token is served by
//    24 / NPC CTAs; each reduces the whole 512-wide split slice (all 8 ranks, all 4 streams) and
//    owns NPC fn rows, so its yp entries have a single writer and no cross-CTA combine is needed.
//  * logit-split cluster kernel (arMhcPostLogitSplitKernel, 3 tokens): (24 / NPC) x GH CTAs per
//    (token, split); CTA (c, g) reduces hidden sub-slice g and projects fn rows
//    [c NPC, c NPC + NPC); the GH partials of (s, c) are summed by cluster rank 0 in fixed order.
//  * row-group kernel (arMhcPostRowGroupKernel, 4 tokens): the row split with 6 fn rows per CTA.
//  * hidden-split cluster kernel (arMhcPostPrenormKernel, 5-32 tokens): one 4-CTA thread-block
//    cluster per (token tile, hidden split); CTA `crank` owns hidden [512 s + 128 crank, + 128) of
//    every stream: it broadcasts its slice of the local shard to every rank, polls the 8 rank
//    contributions of that slice, applies the post-map, projects onto its fn sub-slice (held in
//    registers) and the per-CTA partials are combined inside the cluster through distributed
//    shared memory.
// In the row-split, row-group and logit-split kernels several CTAs of one split read the same
// Lamport slots (the slice is broadcast once, each slot is polled by every CTA that consumes it).
// No grid-wide barrier besides the stock Lamport arrival counter; every grid is small enough to
// be co-resident, so Lamport polling never waits on a CTA that is not yet scheduled.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "flashinfer/comm/trtllm_mnnvl_allreduce.cuh"

namespace flashinfer {
namespace trtllm_mnnvl_ar_mhc_post {

constexpr int kWorldSize = 8;
constexpr int kHidden = 4096;
constexpr int kHcMult = 4;
constexpr int kMix = kHcMult * (2 + kHcMult);  // 24
constexpr int kNumSplits = 8;
constexpr int kSplitWidth = kHidden / kNumSplits;  // 512
constexpr int kMaxTokens = 32;

struct ArMhcPostParams {
  __nv_bfloat16 const* input;     // [M, H] local (un-reduced) shard
  __nv_bfloat16 const* residual;  // [M, HC, H]
  float const* post;              // [M, HC]
  float const* comb;              // [M, HC, HC], comb[m, k, j]
  float const* fn;                // [MIX, HC * H]
  float* yp;                      // [NS, M, MIX]
  float* rp;                      // [NS, M]
  __nv_bfloat16* residualOut;     // [M, HC, H]
  void* multicastPtr;             // MNNVL workspace multicast base
  void* bufferPtrLocal;           // MNNVL workspace unicast base of this rank
  uint32_t* bufferFlags;          // MNNVL workspace flags
  int rank;
  int numTokens;
};

namespace detail {

constexpr uint32_t kLamportSentinel = trtllm_mnnvl_allreduce::utils::kNEGZERO_FP32;

__device__ __forceinline__ uint32_t smem_u32(void const* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ uint32_t cluster_ctarank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ uint32_t mapa_shared(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}

__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void st_async_v4(uint32_t addr, float4 v, uint32_t mbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, "
      "[%5];" ::"r"(addr),
      "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void st_async_f32(uint32_t addr, float v, uint32_t mbar) {
  asm volatile(
      "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];" ::"r"(addr),
      "f"(v), "r"(mbar)
      : "memory");
}

__device__ __forceinline__ void mbar_arrive_local(uint32_t addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr) : "memory");
}

__device__ __forceinline__ void mbar_wait_parity_cluster(uint32_t addr, uint32_t parity) {
  uint32_t done;
  do {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, p;\n\t}"
        : "=r"(done)
        : "r"(addr), "r"(parity)
        : "memory");
  } while (!done);
}

// Plain (weak, coherent) loads for operands that are not produced by peers.
__device__ __forceinline__ float4 ldg_f4(float const* p) {
  float4 v;
  asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ float2 ldg_f2(float const* p) {
  float2 v;
  asm volatile("ld.global.v2.f32 {%0, %1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p));
  return v;
}

// VE consecutive bf16 as VE / 2 packed words; VE in {2, 4, 8}.
template <int VE>
__device__ __forceinline__ void ldg_words(void const* p, uint32_t (&w)[VE / 2]) {
  if constexpr (VE == 2) {
    asm volatile("ld.global.b32 %0, [%1];" : "=r"(w[0]) : "l"(p));
  } else if constexpr (VE == 4) {
    asm volatile("ld.global.v2.b32 {%0, %1}, [%2];" : "=r"(w[0]), "=r"(w[1]) : "l"(p));
  } else {
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                 : "l"(p));
  }
}

// Lamport poll loads: volatile, as the stock one-shot kernel issues them.
template <int VE>
__device__ __forceinline__ void ld_volatile_words(void const* p, uint32_t (&w)[VE / 2]) {
  if constexpr (VE == 2) {
    asm volatile("ld.volatile.global.b32 %0, [%1];" : "=r"(w[0]) : "l"(p) : "memory");
  } else if constexpr (VE == 4) {
    asm volatile("ld.volatile.global.v2.b32 {%0, %1}, [%2];"
                 : "=r"(w[0]), "=r"(w[1])
                 : "l"(p)
                 : "memory");
  } else {
    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3])
                 : "l"(p)
                 : "memory");
  }
}

template <int VE>
__device__ __forceinline__ void stg_words(__nv_bfloat16* p, uint32_t const (&w)[VE / 2]) {
  if constexpr (VE == 2) {
    *reinterpret_cast<unsigned int*>(p) = w[0];
  } else if constexpr (VE == 4) {
    *reinterpret_cast<uint2*>(p) = make_uint2(w[0], w[1]);
  } else {
    *reinterpret_cast<uint4*>(p) = make_uint4(w[0], w[1], w[2], w[3]);
  }
}

template <int VE>
__device__ __forceinline__ void sts_floats(float* p, float const (&v)[VE]) {
#pragma unroll
  for (int i = 0; i < VE; i += 4) {
    if constexpr (VE == 2) {
      *reinterpret_cast<float2*>(p) = make_float2(v[0], v[1]);
    } else {
      *reinterpret_cast<float4*>(p + i) = make_float4(v[i], v[i + 1], v[i + 2], v[i + 3]);
    }
  }
}

__device__ __forceinline__ float bf_lo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bf_hi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

// Lamport wire sanitization of one packed bf16 pair: -0.0 -> +0.0 per half (the stock
// sanitizeLamportPayload), so no payload word can equal the fp32 -0.0 sentinel.
__device__ __forceinline__ uint32_t sanitize_bf16x2(uint32_t v) {
  if ((v & 0xffffu) == 0x8000u) v &= 0xffff0000u;
  if ((v >> 16) == 0x8000u) v &= 0x0000ffffu;
  return v;
}

// (lo, hi) bf16 halves of v converted exactly to fp32 and added to (c0, c1) with RNE.
__device__ __forceinline__ void add_bf16x2_f32(uint32_t v, float& c0, float& c1) {
  asm("{\n\t.reg .b16 l, h;\n\t"
      "mov.b32 {l, h}, %2;\n\t"
      "add.rn.f32.bf16 %0, l, %0;\n\t"
      "add.rn.f32.bf16 %1, h, %1;\n\t}"
      : "+f"(c0), "+f"(c1)
      : "r"(v));
}

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 const p = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t const*>(&p);
}

// d = a * b + c on (x, y) pairs, each lane rounded once (fp32 RNE).
__device__ __forceinline__ float2 ffma2(float2 a, float2 b, float2 c) {
  float2 d;
  asm("{\n\t.reg .b64 ra, rb, rc, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\t"
      "mov.b64 rb, {%4, %5};\n\t"
      "mov.b64 rc, {%6, %7};\n\t"
      "fma.rn.f32x2 rd, ra, rb, rc;\n\t"
      "mov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y), "f"(c.x), "f"(c.y));
  return d;
}

// d = a * b on (x, y) pairs, each lane rounded once (fp32 RNE, no flush to zero): the same result
// as two scalar __fmul_rn.
__device__ __forceinline__ float2 fmul2(float2 a, float2 b) {
  float2 d;
  asm("{\n\t.reg .b64 ra, rb, rd;\n\t"
      "mov.b64 ra, {%2, %3};\n\t"
      "mov.b64 rb, {%4, %5};\n\t"
      "mul.rn.f32x2 rd, ra, rb;\n\t"
      "mov.b64 {%0, %1}, rd;\n\t}"
      : "=f"(d.x), "=f"(d.y)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y));
  return d;
}

// Lamport arrival (the stock arrival counter at bufferFlags[8]).
__device__ __forceinline__ void lamport_arrive(uint32_t* bufferFlags) {
  uint32_t* counter = &bufferFlags[8];
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  asm volatile("red.async.release.global.gpu.add.u32 [%0], %1;" ::"l"(counter), "r"(1) : "memory");
#else
  asm volatile("red.release.global.gpu.add.u32 [%0], %1;" ::"l"(counter), "r"(1) : "memory");
#endif
}

// Broadcast 8 consecutive bf16 of the local shard (token m, hidden h) to every rank's current
// Lamport buffer, sanitized (16 B multicast store).
__device__ __forceinline__ void lamport_broadcast16(ArMhcPostParams const& params,
                                                    __nv_bfloat16* stageMcast, int m, int h) {
  uint4 v = *reinterpret_cast<uint4 const*>(params.input + static_cast<size_t>(m) * kHidden + h);
  v.x = sanitize_bf16x2(v.x);
  v.y = sanitize_bf16x2(v.y);
  v.z = sanitize_bf16x2(v.z);
  v.w = sanitize_bf16x2(v.w);
  *reinterpret_cast<uint4*>(stageMcast + static_cast<size_t>(m) * kHidden * kWorldSize +
                            static_cast<size_t>(params.rank) * kHidden + h) = v;
}

// Poll the 7 peers' contributions of VE consecutive bf16 (token m, hidden h) until none is the
// sentinel; the local contribution `own` (already sanitized) fills slot `rank`.
template <int VE>
__device__ __forceinline__ void lamport_poll(__nv_bfloat16 const* stageLocal, int rank, int m,
                                             int h, uint32_t const (&own)[VE / 2],
                                             uint32_t (&xr)[kWorldSize][VE / 2]) {
  bool ready;
  do {
    ready = true;
#pragma unroll
    for (int r = 0; r < kWorldSize; ++r) {
      if (r == rank) {
#pragma unroll
        for (int w = 0; w < VE / 2; ++w) xr[r][w] = own[w];
      } else {
        ld_volatile_words<VE>(stageLocal + static_cast<size_t>(m) * kHidden * kWorldSize +
                                  static_cast<size_t>(r) * kHidden + h,
                              xr[r]);
#pragma unroll
        for (int w = 0; w < VE / 2; ++w) ready &= xr[r][w] != kLamportSentinel;
      }
    }
  } while (!ready);
}

// Deterministic all-reduce of VE bf16: fp32 accumulation in rank order 0..7, one bf16 RNE
// rounding (bit-identical to the stock one-shot kernel); returns the reduced bf16 as fp32.
template <int VE>
__device__ __forceinline__ void rank_order_reduce(uint32_t const (&xr)[kWorldSize][VE / 2],
                                                  float (&x)[VE]) {
  float a[VE];
#pragma unroll
  for (int e = 0; e < VE; ++e) a[e] = 0.f;
#pragma unroll
  for (int r = 0; r < kWorldSize; ++r)
#pragma unroll
    for (int w = 0; w < VE / 2; ++w) add_bf16x2_f32(xr[r][w], a[2 * w], a[2 * w + 1]);
#pragma unroll
  for (int w = 0; w < VE / 2; ++w) {
    uint32_t const xb = pack_bf16x2(a[2 * w], a[2 * w + 1]);
    x[2 * w] = bf_lo(xb);
    x[2 * w + 1] = bf_hi(xb);
  }
}

// Warp reduce-scatter of V values: at every xor offset O >= OMIN where the live count is even,
// each lane keeps one half (and adds the partner's copy of it); otherwise all live values are
// xor-reduced. On return lane holds the sums (over the lanes that agree with it on every bit below
// OMIN) of value indices [base, base + kOut); lanes with (lane & dup) != 0 hold duplicates.
template <int V, int O, int OMIN = 1>
struct WarpReduceScatter {
  using Next =
      WarpReduceScatter<(O >= OMIN && V % 2 == 0) ? V / 2 : V, (O >= OMIN) ? O / 2 : 0, OMIN>;
  static constexpr int kOut = (O < OMIN) ? V : Next::kOut;
  static __device__ __forceinline__ void run(float* a, int lane, int& base, int& dup) {
    if constexpr (O >= OMIN) {
      if constexpr (V % 2 == 0) {
        constexpr int HF = V / 2;
        bool const up = (lane & O) != 0;
#pragma unroll
        for (int i = 0; i < HF; ++i) {
          float const send = up ? a[i] : a[i + HF];
          float const keep = up ? a[i + HF] : a[i];
          a[i] = keep + __shfl_xor_sync(0xffffffffu, send, O);
        }
        if (up) base += HF;
      } else {
#pragma unroll
        for (int i = 0; i < V; ++i) a[i] += __shfl_xor_sync(0xffffffffu, a[i], O);
        dup |= O;
      }
      Next::run(a, lane, base, dup);
    }
  }
};
template <int V, int OMIN>
struct WarpReduceScatter<V, 0, OMIN> {
  static constexpr int kOut = V;
  static __device__ __forceinline__ void run(float*, int, int&, int&) {}
};

}  // namespace detail

// ---------------------------------------------------------------------------------------------
// Hidden-split cluster kernel (5-32 tokens): clusters of C = 4 CTAs (hidden sub-slices of one
// split), 256 threads per CTA.
// TM: tokens per cluster. VE: hidden elements per post-map item.
// G: projection warp roles. G == 1: warp w projects fn rows {w, w + 8, w + 16} for all TM tokens.
//    G == 2: the 8 warps form 2 token groups x 4 row groups; warp (g, r) projects the 6 rows
//    {r + 4 i} for tokens [g TM / 2, (g + 1) TM / 2), so every new_r vector read from shared
//    memory feeds 6 rows instead of 3 (half the shared-memory traffic at large TM).
// FIN: order of the cluster combine. 0: rank-order chain, 1: pairwise tree (G == 1 only).
// ---------------------------------------------------------------------------------------------
constexpr int kClusterCtas = 4;
constexpr int kClusterThreads = 256;

template <int TM, int VE, int G = 1, int FIN = 0>
__global__ void __launch_bounds__(kClusterThreads, 1)
    arMhcPostPrenormKernel(ArMhcPostParams params) {
  using namespace detail;
  constexpr int H = kHidden;
  constexpr int C = kClusterCtas;
  constexpr int NT = kClusterThreads;
  constexpr int W = kSplitWidth / C;  // hidden elements per CTA per stream
  constexpr int HV = W / 4;           // 4-element hidden vectors per stream
  constexpr int KV = kHcMult * HV;    // float4 K-vectors per CTA
  constexpr int NW = NT / 32;
  constexpr int RPW = G * kMix / NW;             // fn rows per warp
  constexpr int TMG = TM / G;                    // tokens per warp
  constexpr int KPL = KV / 32;                   // K-vectors per lane
  constexpr int HE = W / VE;                     // post-map items per token
  constexpr int NWD = VE / 2;                    // packed bf16x2 words per load
  constexpr int NI = TM * HE;                    // post-map items (each covers all 4 streams)
  constexpr int kBcastVec = 8;                   // bf16 per 16 B broadcast store
  constexpr int kBcastPerToken = W / kBcastVec;  // broadcast threads per token
  // G == 1: warp w projects fn rows {w, w + 8, w + 16} (and warp 0 the square sum); row n of the
  // split is finalised by cluster rank n % 4: warps w and w + 4 both report to rank w % 4 (slot
  // groups 0 and 1).
  // G == 2: warp (g, r) owns rows {r + 4 i}, all finalised by cluster rank r, and the square sum
  // of token g TM / 2 + r; its per-token partials travel as two 16 B messages (6 rows + square
  // sum + pad) to rank r.
  constexpr int SG = 8 / C;  // 16 B slot groups per (source rank, token)
  constexpr bool kTokenGroups2 = (G == 2);
  static_assert(KV % 32 == 0, "K-vectors must tile the warp");
  static_assert(kMix % NW == 0, "rows must tile the warps");
  static_assert(NI <= NT, "one post-map item per thread");
  static_assert(W % kBcastVec == 0 && TM * kBcastPerToken <= NT, "broadcast mapping");
  static_assert(NW == 8 && RPW == 3 * G && TMG * G == TM && (G == 1 || G == 2) &&
                    (FIN == 0 || (FIN == 1 && G == 1)),
                "unsupported cluster / warp layout");

  __shared__ __align__(16) float sR[TM][kHcMult][W];
  // sRed[source rank][token][slot group * 4 + row slot] (row slot 3 = square sum, rank 0 only).
  __shared__ __align__(16) float sRed[C][TM][4 * SG];
  __shared__ __align__(8) uint64_t sBar;

  int const M = params.numTokens;
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  uint32_t const crank = cluster_ctarank();
  int const s = blockIdx.y;
  int const m0 = static_cast<int>(blockIdx.x / C) * TM;
  int const hbase = s * kSplitWidth + static_cast<int>(crank) * W;
  uint32_t const bar = smem_u32(&sBar);

  int const wg = (G == 1) ? 0 : (warp >> 2);    // token group
  int const wr = (G == 1) ? warp : (warp & 3);  // row group: rows wr + (NW / G) i

  // ---- fn sub-slice -> registers (weights: independent of the upstream producer) ----
  float4 f[RPW][KPL];
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    int const n = wr + r * (NW / G);
#pragma unroll
    for (int q = 0; q < KPL; ++q) {
      int const kv = lane + 32 * q;
      int const j = kv / HV;
      int const hv = kv % HV;
      f[r][q] = ldg_f4(params.fn + static_cast<size_t>(n) * (kHcMult * H) + j * H + hbase + hv * 4);
    }
  }

  // PDL: everything below depends on the upstream producer (and on the previous collective's
  // flag update).
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // One-shot Lamport stage of this call (1 stage), exactly as the stock one-shot kernel.
  trtllm_mnnvl_allreduce::LamportFlags<float4, true> flag(params.bufferFlags, 1);
  auto* stageMcast =
      reinterpret_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.multicastPtr, 0));
  auto* stageLocal =
      reinterpret_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.bufferPtrLocal, 0));
  int const rank = params.rank;

  // ---- broadcast this CTA's slice of the local shard to every rank (16 B per thread) ----
  if (tid < TM * kBcastPerToken) {
    int const mb = tid / kBcastPerToken;
    int const mgb = m0 + mb;
    if (mgb < M) {
      int const hb = hbase + (tid % kBcastPerToken) * kBcastVec;
      uint4 v = *reinterpret_cast<uint4 const*>(params.input + static_cast<size_t>(mgb) * H + hb);
      v.x = sanitize_bf16x2(v.x);
      v.y = sanitize_bf16x2(v.y);
      v.z = sanitize_bf16x2(v.z);
      v.w = sanitize_bf16x2(v.w);
      *reinterpret_cast<uint4*>(stageMcast + static_cast<size_t>(mgb) * H * kWorldSize +
                                static_cast<size_t>(rank) * H + hb) = v;
    }
  }

  // ---- post-map item of this thread: VE hidden elements of one token, all 4 streams ----
  int const it = tid;
  int const he = it % HE;
  int const m = it / HE;
  int const mg = m0 + m;
  int const hl = he * VE;  // local hidden offset in [0, W)
  int const h = hbase + hl;
  bool const active = it < NI;
  bool const valid = active && mg < M;
  uint32_t rs[kHcMult][NWD];
  uint32_t own[NWD];
  float4 pv = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 cb[kHcMult];
  if (valid) {
    ldg_words<VE>(params.input + static_cast<size_t>(mg) * H + h, own);
#pragma unroll
    for (int k = 0; k < kHcMult; ++k)
      ldg_words<VE>(params.residual + (static_cast<size_t>(mg) * kHcMult + k) * H + h, rs[k]);
    pv = ldg_f4(params.post + mg * kHcMult);
#pragma unroll
    for (int k = 0; k < kHcMult; ++k)
      cb[k] = ldg_f4(params.comb + mg * kHcMult * kHcMult + k * kHcMult);
  }

  // Cluster-exchange setup (after the operand loads are issued).
  if (tid == 0) {
    // Arrivals: this expect_tx arrival + SG local arrivals per token (this CTA's own partials,
    // written by its warps that report to `crank`); the other C - 1 CTAs deliver TM * SG 16 B
    // messages each.
    constexpr uint32_t kCount = 1 + TM * SG;
    constexpr uint32_t kBytes = (C - 1) * TM * SG * 16;
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(kCount) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(kBytes)
                 : "memory");
  }
  cluster_arrive_relaxed();

  // ---- clear the previous call's (dirty) Lamport buffer; every CTA of the grid participates ----
  flag.clearDirtyLamportBuf(params.bufferPtrLocal, -1);

  // ---- all-reduce: poll the 7 peers' contributions, reduce in rank order (fp32, one bf16
  //      rounding), then the post-map ----
  if (active) {
    float o[kHcMult][VE];
    if (valid) {
#pragma unroll
      for (int w = 0; w < NWD; ++w) own[w] = sanitize_bf16x2(own[w]);
      uint32_t xr[kWorldSize][NWD];
      bool ready;
      do {
        ready = true;
#pragma unroll
        for (int r = 0; r < kWorldSize; ++r) {
          if (r == rank) {
#pragma unroll
            for (int w = 0; w < NWD; ++w) xr[r][w] = own[w];
          } else {
            ld_volatile_words<VE>(stageLocal + static_cast<size_t>(mg) * H * kWorldSize +
                                      static_cast<size_t>(r) * H + h,
                                  xr[r]);
#pragma unroll
            for (int w = 0; w < NWD; ++w) ready &= xr[r][w] != kLamportSentinel;
          }
        }
      } while (!ready);

      float a[VE];
#pragma unroll
      for (int e = 0; e < VE; ++e) a[e] = 0.f;
#pragma unroll
      for (int r = 0; r < kWorldSize; ++r)
#pragma unroll
        for (int w = 0; w < NWD; ++w) add_bf16x2_f32(xr[r][w], a[2 * w], a[2 * w + 1]);
      float x[VE];
#pragma unroll
      for (int w = 0; w < NWD; ++w) {
        uint32_t const xb = pack_bf16x2(a[2 * w], a[2 * w + 1]);
        x[2 * w] = bf_lo(xb);
        x[2 * w + 1] = bf_hi(xb);
      }
      float res[kHcMult][VE];
#pragma unroll
      for (int k = 0; k < kHcMult; ++k)
#pragma unroll
        for (int w = 0; w < NWD; ++w) {
          res[k][2 * w] = bf_lo(rs[k][w]);
          res[k][2 * w + 1] = bf_hi(rs[k][w]);
        }
      float const pj[kHcMult] = {pv.x, pv.y, pv.z, pv.w};
      float const cbk[kHcMult][kHcMult] = {{cb[0].x, cb[0].y, cb[0].z, cb[0].w},
                                           {cb[1].x, cb[1].y, cb[1].z, cb[1].w},
                                           {cb[2].x, cb[2].y, cb[2].z, cb[2].w},
                                           {cb[3].x, cb[3].y, cb[3].z, cb[3].w}};
#pragma unroll
      for (int j = 0; j < kHcMult; ++j) {
        float pjj, cj[kHcMult];
        pjj = pj[j];
#pragma unroll
        for (int k = 0; k < kHcMult; ++k) cj[k] = cbk[k][j];
#pragma unroll
        for (int e = 0; e < VE; ++e) {
          float t = __fmul_rn(x[e], pjj);
#pragma unroll
          for (int k = 0; k < kHcMult; ++k) t = __fmaf_rn(res[k][e], cj[k], t);
          o[j][e] = t;
        }
        uint32_t packed[NWD];
#pragma unroll
        for (int w = 0; w < NWD; ++w) packed[w] = pack_bf16x2(o[j][2 * w], o[j][2 * w + 1]);
        stg_words<VE>(params.residualOut + (static_cast<size_t>(mg) * kHcMult + j) * H + h, packed);
      }
    } else {
#pragma unroll
      for (int j = 0; j < kHcMult; ++j)
#pragma unroll
        for (int e = 0; e < VE; ++e) o[j][e] = 0.f;
    }
#pragma unroll
    for (int j = 0; j < kHcMult; ++j) sts_floats<VE>(&sR[m][j][hl], o[j]);
  }
  __syncthreads();

  // ---- split-K projection partials (fp32 FMA on (x,y)/(z,w) pairs, fn in registers) ----
  float2 acc[TMG][RPW];
  float2 accr[TMG];
#pragma unroll
  for (int mi = 0; mi < TMG; ++mi) {
    accr[mi] = make_float2(0.f, 0.f);
#pragma unroll
    for (int r = 0; r < RPW; ++r) acc[mi][r] = make_float2(0.f, 0.f);
  }
  // Padded tokens (m0 + mm >= M) hold zeros in sR.
#pragma unroll
  for (int mi = 0; mi < TMG; ++mi) {
    int const mm = wg * TMG + mi;
#pragma unroll
    for (int q = 0; q < KPL; ++q) {
      int const kv = lane + 32 * q;
      int const j = kv / HV;
      int const hv = kv % HV;
      float4 const rv = *reinterpret_cast<float4 const*>(&sR[mm][j][hv * 4]);
      float2 const rxy = make_float2(rv.x, rv.y);
      float2 const rzw = make_float2(rv.z, rv.w);
#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        acc[mi][r] = ffma2(make_float2(f[r][q].x, f[r][q].y), rxy, acc[mi][r]);
        acc[mi][r] = ffma2(make_float2(f[r][q].z, f[r][q].w), rzw, acc[mi][r]);
      }
      if (kTokenGroups2 ? (mi % 4 == wr) : (warp == 0)) {
        accr[mi] = ffma2(rxy, rxy, accr[mi]);
        accr[mi] = ffma2(rzw, rzw, accr[mi]);
      }
    }
  }

  // ---- warp reduction: recursive halving down to 4 consecutive values per lane ----
  // Every warp emits TM 16 B messages; message p = values [4 p, 4 p + 4).
  // G == 1: message m = token m: slots 0..2 = rows warp + 8 k, slot 3 = warp 0's square sum.
  // G == 2: messages 2 i, 2 i + 1 = token wg TM / 2 + i: rows wr + 4 k (k < 6), then the square
  // sum (row group i % 4 only) and a zero pad.
  constexpr int V = 4 * TM;
  constexpr int LH = TM <= 1 ? 0 : TM <= 2 ? 1 : TM <= 4 ? 2 : 3;
  static_assert(V == (4 << LH) && TM <= 8, "token tile must be a power of two <= 8");
  float vals[V];
#pragma unroll
  for (int mi = 0; mi < TMG; ++mi) {
    if constexpr (kTokenGroups2) {
#pragma unroll
      for (int k = 0; k < 6; ++k) vals[8 * mi + k] = acc[mi][k].x + acc[mi][k].y;
      vals[8 * mi + 6] = accr[mi].x + accr[mi].y;
      vals[8 * mi + 7] = 0.f;
    } else {
#pragma unroll
      for (int r = 0; r < RPW; ++r) vals[4 * mi + r] = acc[mi][r].x + acc[mi][r].y;
      vals[4 * mi + 3] = accr[mi].x + accr[mi].y;  // zero outside warp 0
    }
  }
#pragma unroll
  for (int st = 0; st < LH; ++st) {
    int const off = 16 >> st;
    int const half = V >> (st + 1);
    bool const upper = (lane & off) != 0;
#pragma unroll
    for (int i = 0; i < half; ++i) {
      float const send = upper ? vals[i] : vals[i + half];
      float const keep = upper ? vals[i + half] : vals[i];
      vals[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
    }
  }
#pragma unroll
  for (int off = 16 >> LH; off > 0; off >>= 1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) vals[i] += __shfl_xor_sync(0xffffffffu, vals[i], off);
  }
  bool const publisher = (lane & ((1 << (5 - LH)) - 1)) == 0;
  int const pmsg = lane >> (5 - LH);  // message held by this lane
  // token of this lane's 4 values, its finalising cluster rank and the slot group there
  int const pm = kTokenGroups2 ? wg * TMG + (pmsg >> 1) : pmsg;
  int const pdst = kTokenGroups2 ? wr : warp % C;
  int const pslot = kTokenGroups2 ? 4 * (pmsg & 1) : 4 * (warp / C);

  // Every CTA of the cluster has initialised its mbarrier and read the Lamport flags.
  cluster_wait_acquire();
  if (crank == 0 && tid == 0) lamport_arrive(params.bufferFlags);

  if (publisher) {
    float4 const v4 = make_float4(vals[0], vals[1], vals[2], vals[3]);
    if (pdst == static_cast<int>(crank)) {
      *reinterpret_cast<float4*>(&sRed[crank][pm][pslot]) = v4;
      mbar_arrive_local(bar);
    } else {
      st_async_v4(mapa_shared(smem_u32(&sRed[crank][pm][pslot]), pdst), v4, mapa_shared(bar, pdst));
    }
  }

  // G == 1: rank c finalises rows {c + C g + 8 k} (+ the square sum on rank 0) of its token tile.
  // G == 2: rank c finalises rows {c + 4 k} (k < 6) and the square sums of the tokens m with
  // (m % (TM / 2)) % 4 == c. The C source partials are summed in fixed order. Token m0 is always
  // valid, so thread 0 of every CTA waits for the full phase: no CTA exits while a peer's message
  // to it is in flight.
  if (tid < TM * 4 * SG) {
    int const mm = tid / (4 * SG);
    int const sl = tid % (4 * SG);
    int const g = sl >> 2;
    int const k = sl & 3;
    int const mgg = m0 + mm;
    if constexpr (kTokenGroups2) {
      if ((sl < 6 || (sl == 6 && (mm % TMG) % 4 == static_cast<int>(crank))) && mgg < M) {
        mbar_wait_parity_cluster(bar, 0);
        float sum = sRed[0][mm][sl];
#pragma unroll
        for (int c = 1; c < C; ++c) sum += sRed[c][mm][sl];
        if (sl < 6)
          params.yp[(static_cast<size_t>(s) * M + mgg) * kMix + static_cast<int>(crank) + 4 * sl] =
              sum;
        else
          params.rp[static_cast<size_t>(s) * M + mgg] = sum;
      }
    } else if ((k < 3 || (crank == 0 && g == 0)) && mgg < M) {
      mbar_wait_parity_cluster(bar, 0);
      float sum;
      if constexpr (FIN == 1) {
        float const p01 = __fadd_rn(sRed[0][mm][sl], sRed[1][mm][sl]);
        float const p23 = __fadd_rn(sRed[2][mm][sl], sRed[3][mm][sl]);
        sum = __fadd_rn(p01, p23);
      } else {
        sum = sRed[0][mm][sl];
#pragma unroll
        for (int c = 1; c < C; ++c) sum += sRed[c][mm][sl];
      }
      if (k < 3)
        params.yp[(static_cast<size_t>(s) * M + mgg) * kMix + static_cast<int>(crank) + C * g +
                  8 * k] = sum;
      else
        params.rp[static_cast<size_t>(s) * M + mgg] = sum;
    }
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // Flip the Lamport buffer once every cluster has arrived (grid thread 0 only).
  flag.waitAndUpdate({static_cast<uint32_t>(M * H * kWorldSize * sizeof(__nv_bfloat16)), 0, 0, 0});
}

// ---------------------------------------------------------------------------------------------
// Row-split kernel (exactly FIXED_M tokens; used at 1 and 2): no cross-CTA combine. Grid
// (CPS x 8 splits, FIXED_M), CPS = 24 / NPC, blockIdx.x = c * 8 + s (adjacent CTAs read
// contiguous 512-wide segments of the same fn rows).
// CTA c of split s broadcasts hidden [512 s + c * 512 / CPS, + 512 / CPS) of the local shard,
// then polls and reduces the WHOLE split slice of its token (thread = 2 hidden x 4 streams, one
// 4 B Lamport slot per rank), keeps x / new_r in registers and projects onto fn rows
// [c NPC, c NPC + NPC) (single writer of those yp entries). CTA c also writes residual_out for
// hidden pairs [c * 256 / CPS, (c + 1) * 256 / CPS) of the split; CTA 0 writes rp.
// FIXED_M: yp / rp strides are compile-time; the launcher refuses any other token count. The
//   Lamport clear size always uses the runtime count.
// The post-map runs as fmul2 / ffma2 on the thread's (lo, hi) hidden pair: per lane the same
//   operations in the same order as the scalar __fmul_rn / __fmaf_rn chain of the other kernels,
//   so residual_out is bit-identical across geometries.
// TAIL: layout of the 8 warp partials, both summed by the same fixed pairwise tree.
//   1: [warp][value]; 2: value-major [value][warp] (two float4 reads per value).
// ---------------------------------------------------------------------------------------------
constexpr int kRowSplitThreads = kSplitWidth / 2;  // 256

template <int NPC, int FIXED_M, int TAIL>
__global__ void __launch_bounds__(kRowSplitThreads)
    arMhcPostRowSplitKernel(ArMhcPostParams params) {
  using namespace detail;
  constexpr int H = kHidden;
  constexpr int NT = kRowSplitThreads;
  constexpr int CPS = kMix / NPC;  // CTAs per split
  constexpr int NV = NPC + 1;      // NPC logits + square sum
  constexpr int NW = NT / 32;
  constexpr int kBcastPerCta = kSplitWidth / CPS / 8;  // 16 B broadcast stores per CTA
  static_assert(kMix % NPC == 0 && kSplitWidth % (CPS * 8) == 0 && FIXED_M > 0,
                "row-split mapping");
  static_assert(NV <= NT, "finaliser threads");
  static_assert((TAIL == 1 || TAIL == 2) && NW == 8, "tail order");
  __shared__ __align__(16) float sPart[TAIL == 2 ? NV : NW][TAIL == 2 ? NW : NV];

  constexpr int M = FIXED_M;
  int const tid = threadIdx.x;
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const s = static_cast<int>(blockIdx.x) % kNumSplits;
  int const c = static_cast<int>(blockIdx.x) / kNumSplits;
  int const m = blockIdx.y;
  int const hb = s * kSplitWidth + 2 * tid;

  // ---- fn rows of this CTA (2 hidden x 4 streams per thread) -> registers ----
  float2 f[NPC][kHcMult];
#pragma unroll
  for (int i = 0; i < NPC; ++i)
#pragma unroll
    for (int j = 0; j < kHcMult; ++j)
      f[i][j] = ldg_f2(params.fn + static_cast<size_t>(c * NPC + i) * (kHcMult * H) + j * H + hb);

  asm volatile("griddepcontrol.wait;" ::: "memory");

  // Per-CTA arrivals (no cluster combine in this kernel).
  trtllm_mnnvl_allreduce::LamportFlags<float4, false> flag(params.bufferFlags, 1);
  auto* stageMcast =
      reinterpret_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.multicastPtr, 0));
  auto* stageLocal =
      reinterpret_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.bufferPtrLocal, 0));
  int const rank = params.rank;

  if (tid < kBcastPerCta) {
    lamport_broadcast16(params, stageMcast, m, s * kSplitWidth + c * (kSplitWidth / CPS) + tid * 8);
  }

  uint32_t own[1];
  uint32_t rv[kHcMult][1];
  ldg_words<2>(params.input + static_cast<size_t>(m) * H + hb, own);
#pragma unroll
  for (int k = 0; k < kHcMult; ++k)
    ldg_words<2>(params.residual + (static_cast<size_t>(m) * kHcMult + k) * H + hb, rv[k]);
  float4 const pv = ldg_f4(params.post + m * kHcMult);
  float4 cv[kHcMult];
#pragma unroll
  for (int k = 0; k < kHcMult; ++k) cv[k] = ldg_f4(params.comb + (m * kHcMult + k) * kHcMult);

  flag.clearDirtyLamportBuf(params.bufferPtrLocal, -1);

  own[0] = sanitize_bf16x2(own[0]);
  uint32_t xr[kWorldSize][1];
  lamport_poll<2>(stageLocal, rank, m, hb, own, xr);
  float x[2];
  rank_order_reduce<2>(xr, x);

  float const pj[kHcMult] = {pv.x, pv.y, pv.z, pv.w};
  float const ck[kHcMult][kHcMult] = {{cv[0].x, cv[0].y, cv[0].z, cv[0].w},
                                      {cv[1].x, cv[1].y, cv[1].z, cv[1].w},
                                      {cv[2].x, cv[2].y, cv[2].z, cv[2].w},
                                      {cv[3].x, cv[3].y, cv[3].z, cv[3].w}};
  float nr[kHcMult][2];
#pragma unroll
  for (int j = 0; j < kHcMult; ++j) {
    float2 u = fmul2(make_float2(pj[j], pj[j]), make_float2(x[0], x[1]));
#pragma unroll
    for (int k = 0; k < kHcMult; ++k)
      u = ffma2(make_float2(ck[k][j], ck[k][j]), make_float2(bf_lo(rv[k][0]), bf_hi(rv[k][0])), u);
    nr[j][0] = u.x;
    nr[j][1] = u.y;
  }
  if ((tid * CPS) / NT == c) {
#pragma unroll
    for (int j = 0; j < kHcMult; ++j)
      *reinterpret_cast<uint32_t*>(params.residualOut + (static_cast<size_t>(m) * kHcMult + j) * H +
                                   hb) = pack_bf16x2(nr[j][0], nr[j][1]);
  }
  float v[NV];
  {
    float2 r2 = make_float2(0.f, 0.f);
#pragma unroll
    for (int j = 0; j < kHcMult; ++j) {
      float2 const nv = make_float2(nr[j][0], nr[j][1]);
      r2 = ffma2(nv, nv, r2);
    }
    v[NPC] = r2.x + r2.y;
  }
#pragma unroll
  for (int i = 0; i < NPC; ++i) {
    float2 acc = make_float2(0.f, 0.f);
#pragma unroll
    for (int j = 0; j < kHcMult; ++j) acc = ffma2(f[i][j], make_float2(nr[j][0], nr[j][1]), acc);
    v[i] = acc.x + acc.y;
  }

  // Warp reduce-scatter, then a fixed pairwise sum of the 8 warp partials.
  int base = 0, dup = 0;
  WarpReduceScatter<NV, 16>::run(v, lane, base, dup);
  if ((lane & dup) == 0) {
#pragma unroll
    for (int i = 0; i < WarpReduceScatter<NV, 16>::kOut; ++i) {
      if constexpr (TAIL == 2)
        sPart[base + i][warp] = v[i];
      else
        sPart[warp][base + i] = v[i];
    }
  }
  __syncthreads();
  // Every thread of this CTA has read the Lamport flags.
  if (tid == 0) lamport_arrive(params.bufferFlags);
  if (tid < NV) {
    float sum;
    if constexpr (TAIL == 2) {
      float4 const q03 = *reinterpret_cast<float4 const*>(&sPart[tid][0]);
      float4 const q47 = *reinterpret_cast<float4 const*>(&sPart[tid][4]);
      sum = __fadd_rn(__fadd_rn(__fadd_rn(q03.x, q03.y), __fadd_rn(q03.z, q03.w)),
                      __fadd_rn(__fadd_rn(q47.x, q47.y), __fadd_rn(q47.z, q47.w)));
    } else {
      sum = __fadd_rn(__fadd_rn(__fadd_rn(sPart[0][tid], sPart[1][tid]),
                                __fadd_rn(sPart[2][tid], sPart[3][tid])),
                      __fadd_rn(__fadd_rn(sPart[4][tid], sPart[5][tid]),
                                __fadd_rn(sPart[6][tid], sPart[7][tid])));
    }
    if (tid < NPC)
      params.yp[(static_cast<size_t>(s) * M + m) * kMix + c * NPC + tid] = sum;
    else if (c == 0)
      params.rp[static_cast<size_t>(s) * M + m] = sum;
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // Clear size of this call's buffer from the runtime token count.
  flag.waitAndUpdate(
      {static_cast<uint32_t>(params.numTokens * H * kWorldSize * sizeof(__nv_bfloat16)), 0, 0, 0});
}

// ---------------------------------------------------------------------------------------------
// Logit-split cluster kernel (3 tokens). Grid (CPS x GH, 8 splits x M), clusters of GH CTAs
// along x, CPS = 24 / NPC. CTA (c, g) owns hidden sub-slice g (W = 512 / GH, all 4 streams, HPT
// hidden per thread) of split s for one token: it broadcasts 1 / CPS of that sub-slice, polls and
// reduces the whole sub-slice, applies the post-map in registers and projects onto fn rows
// [c NPC, c NPC + NPC). Each warp's reduce-scatter stops above xor offset OMIN and every lane
// st.async's its partials straight to cluster rank 0, which sums the GH x warps x OMIN partials of
// every output in a fixed pairwise tree. Only the CTAs of logit group 0 (the rp writers)
// accumulate the square sum; the others contribute 0 to an output nobody reads.
// ---------------------------------------------------------------------------------------------
template <int NPC, int GH, int HPT, int OMIN>
__global__ void __launch_bounds__(kSplitWidth / GH / HPT)
    arMhcPostLogitSplitKernel(ArMhcPostParams params) {
  using namespace detail;
  constexpr int H = kHidden;
  constexpr int W = kSplitWidth / GH;  // hidden elements per CTA per stream
  constexpr int NT = W / HPT;
  constexpr int NWARP = NT / 32;
  constexpr int CPS = kMix / NPC;            // logit groups per split
  constexpr int NV = NPC + 1;                // NPC logits + square sum
  constexpr int NVP = NV + (NV & 1);         // padded to even for the reduce-scatter
  constexpr int kBcastPerCta = W / CPS / 8;  // 16 B broadcast stores per CTA
  using RS = WarpReduceScatter<NVP, 16, OMIN>;
  static_assert(HPT == 2, "one packed bf16x2 Lamport slot per thread and rank");
  static_assert(GH > 1 && NT % 32 == 0 && kMix % NPC == 0, "logit-split mapping");
  static_assert(W % (CPS * 8) == 0, "broadcast mapping");
  static_assert(NV <= NT, "finaliser threads");

  __shared__ float sRx[GH][NWARP][NVP][OMIN];
  __shared__ __align__(8) uint64_t sBar;

  int const M = params.numTokens;
  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  int const g = static_cast<int>(cluster_ctarank());
  int const c = blockIdx.x / GH;
  int const s = blockIdx.y % kNumSplits;
  int const m = blockIdx.y / kNumSplits;
  int const h = s * kSplitWidth + g * W + tid * HPT;
  uint32_t const bar = smem_u32(&sBar);

  // ---- fn rows of this CTA -> registers (weights: independent of the upstream producer) ----
  float2 fv[NPC][kHcMult];
#pragma unroll
  for (int i = 0; i < NPC; ++i)
#pragma unroll
    for (int j = 0; j < kHcMult; ++j)
      fv[i][j] = ldg_f2(params.fn + static_cast<size_t>(c * NPC + i) * (kHcMult * H) + j * H + h);

  asm volatile("griddepcontrol.wait;" ::: "memory");

  trtllm_mnnvl_allreduce::LamportFlags<float4, true> flag(params.bufferFlags, 1);
  auto* stageMcast =
      reinterpret_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.multicastPtr, 0));
  auto* stageLocal =
      reinterpret_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.bufferPtrLocal, 0));
  int const rank = params.rank;

  if (tid < kBcastPerCta) {
    lamport_broadcast16(params, stageMcast, m, s * kSplitWidth + g * W + c * (W / CPS) + tid * 8);
  }

  uint32_t own[1];
  uint32_t rw[kHcMult][1];
  ldg_words<2>(params.input + static_cast<size_t>(m) * H + h, own);
#pragma unroll
  for (int k = 0; k < kHcMult; ++k)
    ldg_words<2>(params.residual + (static_cast<size_t>(m) * kHcMult + k) * H + h, rw[k]);
  float4 const pv = ldg_f4(params.post + m * kHcMult);
  float4 cv[kHcMult];
#pragma unroll
  for (int k = 0; k < kHcMult; ++k) cv[k] = ldg_f4(params.comb + (m * kHcMult + k) * kHcMult);

  if (g == 0 && tid == 0) {
    // One expect_tx arrival; every lane of every warp of the GH CTAs delivers RS::kOut floats.
    constexpr uint32_t kBytes = GH * NWARP * NVP * OMIN * 4;
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(bar) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(kBytes)
                 : "memory");
  }
  cluster_arrive_relaxed();

  flag.clearDirtyLamportBuf(params.bufferPtrLocal, -1);

  own[0] = sanitize_bf16x2(own[0]);
  uint32_t xr[kWorldSize][1];
  lamport_poll<2>(stageLocal, rank, m, h, own, xr);
  float x[2];
  rank_order_reduce<2>(xr, x);

  float const pj[kHcMult] = {pv.x, pv.y, pv.z, pv.w};
  float const ck[kHcMult][kHcMult] = {{cv[0].x, cv[0].y, cv[0].z, cv[0].w},
                                      {cv[1].x, cv[1].y, cv[1].z, cv[1].w},
                                      {cv[2].x, cv[2].y, cv[2].z, cv[2].w},
                                      {cv[3].x, cv[3].y, cv[3].z, cv[3].w}};
  float nr[kHcMult][2];
  float r2 = 0.f;
#pragma unroll
  for (int j = 0; j < kHcMult; ++j)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float u = __fmul_rn(pj[j], x[e]);
#pragma unroll
      for (int k = 0; k < kHcMult; ++k)
        u = __fmaf_rn(ck[k][j], e == 0 ? bf_lo(rw[k][0]) : bf_hi(rw[k][0]), u);
      nr[j][e] = u;
      if (c == 0) r2 = __fmaf_rn(u, u, r2);
    }
  if ((tid * CPS) / NT == c) {
#pragma unroll
    for (int j = 0; j < kHcMult; ++j)
      *reinterpret_cast<uint32_t*>(params.residualOut + (static_cast<size_t>(m) * kHcMult + j) * H +
                                   h) = pack_bf16x2(nr[j][0], nr[j][1]);
  }
  float v[NVP];
#pragma unroll
  for (int i = 0; i < NPC; ++i) {
    float acc0 = 0.f, acc1 = 0.f;  // streams {0, 1} and {2, 3}
#pragma unroll
    for (int j = 0; j < kHcMult / 2; ++j)
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float const f0 = e == 0 ? fv[i][j].x : fv[i][j].y;
        float const f1 = e == 0 ? fv[i][j + 2].x : fv[i][j + 2].y;
        acc0 = __fmaf_rn(f0, nr[j][e], acc0);
        acc1 = __fmaf_rn(f1, nr[j + 2][e], acc1);
      }
    v[i] = acc0 + acc1;
  }
  v[NPC] = r2;
#pragma unroll
  for (int i = NV; i < NVP; ++i) v[i] = 0.f;

  int base = 0, dup = 0;
  RS::run(v, lane, base, dup);
  // Rank 0's mbarrier is initialised and visible; every CTA of the cluster has read the flags.
  cluster_wait_acquire();
  if (g == 0 && tid == 0) lamport_arrive(params.bufferFlags);
  if ((lane & dup) == 0) {
#pragma unroll
    for (int i = 0; i < RS::kOut; ++i)
      st_async_f32(mapa_shared(smem_u32(&sRx[g][warp][base + i][lane & (OMIN - 1)]), 0), v[i],
                   mapa_shared(bar, 0));
  }
  if (g != 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    return;
  }
  // Thread 0 is a finaliser, so rank 0 stays resident until every message has landed.
  if (tid < NV) {
    mbar_wait_parity_cluster(bar, 0);
    // Pairwise tree over the GH x NWARP x OMIN partials (fixed order).
    constexpr int NP = GH * NWARP * OMIN;
    float part[NP];
#pragma unroll
    for (int gg = 0; gg < GH; ++gg)
#pragma unroll
      for (int w = 0; w < NWARP; ++w)
#pragma unroll
        for (int q = 0; q < OMIN; ++q) part[(gg * NWARP + w) * OMIN + q] = sRx[gg][w][tid][q];
#pragma unroll
    for (int width = 1; width < NP; width *= 2)
#pragma unroll
      for (int i = 0; i + width < NP; i += 2 * width) part[i] += part[i + width];
    if (tid < NPC)
      params.yp[(static_cast<size_t>(s) * M + m) * kMix + c * NPC + tid] = part[0];
    else if (c == 0)
      params.rp[static_cast<size_t>(s) * M + m] = part[0];
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  flag.waitAndUpdate({static_cast<uint32_t>(M * H * kWorldSize * sizeof(__nv_bfloat16)), 0, 0, 0});
}

// ---------------------------------------------------------------------------------------------
// Row-group kernel (exactly FIXED_M tokens; used at 4): the row split with NPC fn rows per CTA and
// a reduction tree sized for NPC + 1 > 4 outputs. Grid (CPS x 8 splits, FIXED_M), blockIdx.x =
// c * 8 + s, CPS = 24 / NPC. CTA (c, s) of token m broadcasts hidden [512 s + c * 512 / CPS,
// + 512 / CPS) of the local shard, polls and reduces the whole split slice (thread = 2 hidden x 4
// streams), applies the post-map, writes residual_out for hidden pairs [c * 256 / CPS, + 256 /
// CPS) and projects onto fn rows [c NPC, c NPC + NPC) (+ the square sum, written by c == 0).
// Per thread the NVP = pow2(NPC + 1) values are reduced by recursive halving to one value per
// lane, the PPW lane partials per value and warp go to shared memory and are summed by a fixed
// pairwise tree over the NW x PPW partials.
// ---------------------------------------------------------------------------------------------
template <int NPC, int FIXED_M>
__global__ void __launch_bounds__(kRowSplitThreads)
    arMhcPostRowGroupKernel(ArMhcPostParams params) {
  using namespace detail;
  constexpr int H = kHidden;
  constexpr int NT = kRowSplitThreads;
  constexpr int CPS = kMix / NPC;              // CTAs per split
  constexpr int NVP = (NPC + 1 <= 4) ? 4 : 8;  // NPC logits + square sum, padded to a power of 2
  constexpr int LV = NVP == 4 ? 2 : 3;         // halving steps
  constexpr int L = LV < 3 ? 3 : LV;           // + xor steps up to 3 levels
  constexpr int PPW = 1 << (5 - L);            // partials per value per warp
  constexpr int NW = NT / 32;
  constexpr int P = NW * PPW;                          // partials per value
  constexpr int PS = P + 4;                            // padded row
  constexpr int kBcastPerCta = kSplitWidth / CPS / 8;  // 16 B broadcast stores per CTA
  static_assert(kMix % NPC == 0 && NPC + 1 <= 8 && kSplitWidth % (CPS * 8) == 0 && FIXED_M > 0,
                "row-group mapping");
  __shared__ __align__(16) float sPart[NVP][PS];

  int const tid = threadIdx.x;
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const c = static_cast<int>(blockIdx.x) / kNumSplits;
  int const s = static_cast<int>(blockIdx.x) % kNumSplits;
  int const m = blockIdx.y;
  int const hb = s * kSplitWidth + 2 * tid;

  // ---- fn rows of this CTA (2 hidden x 4 streams per thread) -> registers ----
  float2 f[NPC][kHcMult];
#pragma unroll
  for (int i = 0; i < NPC; ++i)
#pragma unroll
    for (int j = 0; j < kHcMult; ++j)
      f[i][j] = ldg_f2(params.fn + static_cast<size_t>(c * NPC + i) * (kHcMult * H) + j * H + hb);

  asm volatile("griddepcontrol.wait;" ::: "memory");

  // Per-CTA arrivals (the cluster dimension of the launch is a placement hint only).
  trtllm_mnnvl_allreduce::LamportFlags<float4, false> flag(params.bufferFlags, 1);
  auto* stageMcast =
      reinterpret_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.multicastPtr, 0));
  auto* stageLocal =
      reinterpret_cast<__nv_bfloat16*>(flag.getCurLamportBuf(params.bufferPtrLocal, 0));
  int const rank = params.rank;

  if (tid < kBcastPerCta) {
    lamport_broadcast16(params, stageMcast, m, s * kSplitWidth + c * (kSplitWidth / CPS) + tid * 8);
  }

  uint32_t own[1];
  uint32_t rv[kHcMult][1];
  ldg_words<2>(params.input + static_cast<size_t>(m) * H + hb, own);
#pragma unroll
  for (int k = 0; k < kHcMult; ++k)
    ldg_words<2>(params.residual + (static_cast<size_t>(m) * kHcMult + k) * H + hb, rv[k]);
  float4 const pv = ldg_f4(params.post + m * kHcMult);
  float4 cv[kHcMult];
#pragma unroll
  for (int k = 0; k < kHcMult; ++k) cv[k] = ldg_f4(params.comb + (m * kHcMult + k) * kHcMult);

  flag.clearDirtyLamportBuf(params.bufferPtrLocal, -1);

  own[0] = sanitize_bf16x2(own[0]);
  uint32_t xr[kWorldSize][1];
  lamport_poll<2>(stageLocal, rank, m, hb, own, xr);
  float x[2];
  rank_order_reduce<2>(xr, x);

  float const pj[kHcMult] = {pv.x, pv.y, pv.z, pv.w};
  float const ck[kHcMult][kHcMult] = {{cv[0].x, cv[0].y, cv[0].z, cv[0].w},
                                      {cv[1].x, cv[1].y, cv[1].z, cv[1].w},
                                      {cv[2].x, cv[2].y, cv[2].z, cv[2].w},
                                      {cv[3].x, cv[3].y, cv[3].z, cv[3].w}};
  float nr[kHcMult][2];
#pragma unroll
  for (int j = 0; j < kHcMult; ++j)
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      float u = __fmul_rn(pj[j], x[e]);
#pragma unroll
      for (int k = 0; k < kHcMult; ++k)
        u = __fmaf_rn(ck[k][j], e == 0 ? bf_lo(rv[k][0]) : bf_hi(rv[k][0]), u);
      nr[j][e] = u;
    }
  if ((tid * CPS) / NT == c) {
#pragma unroll
    for (int j = 0; j < kHcMult; ++j)
      *reinterpret_cast<uint32_t*>(params.residualOut + (static_cast<size_t>(m) * kHcMult + j) * H +
                                   hb) = pack_bf16x2(nr[j][0], nr[j][1]);
  }
  float vals[NVP];
  {
    float2 r2 = make_float2(0.f, 0.f);
#pragma unroll
    for (int j = 0; j < kHcMult; ++j) {
      float2 const nv = make_float2(nr[j][0], nr[j][1]);
      r2 = ffma2(nv, nv, r2);
    }
    vals[NPC] = r2.x + r2.y;
  }
#pragma unroll
  for (int i = 0; i < NPC; ++i) {
    float2 acc = make_float2(0.f, 0.f);
#pragma unroll
    for (int j = 0; j < kHcMult; ++j) acc = ffma2(f[i][j], make_float2(nr[j][0], nr[j][1]), acc);
    vals[i] = acc.x + acc.y;
  }
#pragma unroll
  for (int i = NPC + 1; i < NVP; ++i) vals[i] = 0.f;

  // Recursive halving: after step st (xor offset 16 >> st) the lane keeps the upper / lower half;
  // lane l ends with value l >> (5 - LV), summed over the lanes that differ in the low bits.
#pragma unroll
  for (int st = 0; st < LV; ++st) {
    int const off = 16 >> st;
    int const half = NVP >> (st + 1);
    bool const up = (lane & off) != 0;
#pragma unroll
    for (int i = 0; i < half; ++i) {
      float const send = up ? vals[i] : vals[i + half];
      float const keep = up ? vals[i + half] : vals[i];
      vals[i] = keep + __shfl_xor_sync(0xffffffffu, send, off);
    }
  }
#pragma unroll
  for (int st = LV; st < L; ++st) vals[0] += __shfl_xor_sync(0xffffffffu, vals[0], 16 >> st);
  {
    int const q = lane >> (5 - LV);                      // value held by this lane
    int const xbits = ((1 << (L - LV)) - 1) << (5 - L);  // lanes duplicated by the xor steps
    if ((lane & xbits) == 0) sPart[q][warp * PPW + (lane & (PPW - 1))] = vals[0];
  }
  __syncthreads();
  // Every thread of this CTA has read the Lamport flags.
  if (tid == 0) lamport_arrive(params.bufferFlags);
  if (tid < NPC + 1) {
    float tv[P];
#pragma unroll
    for (int k = 0; k < P / 4; ++k) {
      float4 const p4 = *reinterpret_cast<float4 const*>(&sPart[tid][4 * k]);
      tv[4 * k] = p4.x;
      tv[4 * k + 1] = p4.y;
      tv[4 * k + 2] = p4.z;
      tv[4 * k + 3] = p4.w;
    }
#pragma unroll
    for (int w = P / 2; w > 0; w >>= 1)
#pragma unroll
      for (int k = 0; k < w; ++k) tv[k] += tv[k + w];
    if (tid < NPC)
      params.yp[(static_cast<size_t>(s) * FIXED_M + m) * kMix + c * NPC + tid] = tv[0];
    else if (c == 0)
      params.rp[static_cast<size_t>(s) * FIXED_M + m] = tv[0];
  }
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // Clear size of this call's buffer from the runtime token count.
  flag.waitAndUpdate(
      {static_cast<uint32_t>(params.numTokens * H * kWorldSize * sizeof(__nv_bfloat16)), 0, 0, 0});
}

// ---------------------------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------------------------
namespace detail {

template <typename Kernel>
inline cudaError_t launchWithCluster(Kernel kfn, dim3 grid, int threads, int cluster,
                                     ArMhcPostParams const& params, cudaStream_t stream,
                                     bool launchWithPdl) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = cluster;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[1].val.programmaticStreamSerializationAllowed = launchWithPdl ? 1 : 0;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;
  return cudaLaunchKernelEx(&cfg, kfn, params);
}

}  // namespace detail

template <int TM, int VE, int G = 1, int FIN = 0>
inline cudaError_t launchArMhcPostPrenorm(ArMhcPostParams const& params, cudaStream_t stream,
                                          bool launchWithPdl) {
  return detail::launchWithCluster(
      arMhcPostPrenormKernel<TM, VE, G, FIN>,
      dim3(kClusterCtas * ((params.numTokens + TM - 1) / TM), kNumSplits, 1), kClusterThreads,
      kClusterCtas, params, stream, launchWithPdl);
}

// The cluster dimension (CL CTAs along x) is a placement hint only: the kernel does not
// communicate across CTAs and every CTA arrives on the Lamport counter itself.
template <int NPC, int CL, int FIXED_M, int TAIL>
inline cudaError_t launchArMhcPostRowSplit(ArMhcPostParams const& params, cudaStream_t stream,
                                           bool launchWithPdl) {
  if (params.numTokens != FIXED_M) return cudaErrorInvalidValue;
  return detail::launchWithCluster(arMhcPostRowSplitKernel<NPC, FIXED_M, TAIL>,
                                   dim3(kNumSplits * (kMix / NPC), FIXED_M, 1), kRowSplitThreads,
                                   CL, params, stream, launchWithPdl);
}

template <int NPC, int GH, int HPT, int OMIN>
inline cudaError_t launchArMhcPostLogitSplit(ArMhcPostParams const& params, cudaStream_t stream,
                                             bool launchWithPdl) {
  return detail::launchWithCluster(arMhcPostLogitSplitKernel<NPC, GH, HPT, OMIN>,
                                   dim3((kMix / NPC) * GH, kNumSplits * params.numTokens, 1),
                                   kSplitWidth / GH / HPT, GH, params, stream, launchWithPdl);
}

// Placement-only 2-CTA cluster dimension, as for the row split.
template <int NPC, int FIXED_M>
inline cudaError_t launchArMhcPostRowGroup(ArMhcPostParams const& params, cudaStream_t stream,
                                           bool launchWithPdl) {
  if (params.numTokens != FIXED_M) return cudaErrorInvalidValue;
  return detail::launchWithCluster(arMhcPostRowGroupKernel<NPC, FIXED_M>,
                                   dim3((kMix / NPC) * kNumSplits, FIXED_M, 1), kRowSplitThreads, 2,
                                   params, stream, launchWithPdl);
}

// Token-count dispatch over the full T <= 32 decode range (7 kernel instantiations). Grids:
// <= 128 CTAs of 256 threads (row split, row group, hidden-split clusters) or <= 192 CTAs of 128
// threads (logit split), so every grid is co-resident and Lamport polling never waits on a CTA
// that is not yet scheduled on its own GPU.
//   1 token: row split, 8-CTA placement clusters, pairwise warp-partial tree;
//   2 tokens: the same with 2-CTA placement clusters and a value-major partial layout;
//   3 tokens: logit split; 4 tokens: row group (6 fn rows per CTA);
//   5-8 / 9-16 tokens: hidden-split clusters with 2 / 4-token tiles (pairwise combine at 2);
//   17-32 tokens: hidden-split clusters, 8-token tiles, 2 token groups x 4 row groups.
inline cudaError_t arMhcPostPrenormDispatch(ArMhcPostParams const& params, cudaStream_t stream,
                                            bool launchWithPdl) {
  int const M = params.numTokens;
  if (M <= 0 || M > kMaxTokens) return cudaErrorInvalidValue;
  if (M == 1) return launchArMhcPostRowSplit<3, 8, 1, 1>(params, stream, launchWithPdl);
  if (M == 2) return launchArMhcPostRowSplit<3, 2, 2, 2>(params, stream, launchWithPdl);
  if (M == 3) return launchArMhcPostLogitSplit<6, 2, 2, 8>(params, stream, launchWithPdl);
  if (M == 4) return launchArMhcPostRowGroup<6, 4>(params, stream, launchWithPdl);
  if (M <= 8) return launchArMhcPostPrenorm<2, 2, 1, 1>(params, stream, launchWithPdl);
  if (M <= 16) return launchArMhcPostPrenorm<4, 2>(params, stream, launchWithPdl);
  return launchArMhcPostPrenorm<8, 4, 2>(params, stream, launchWithPdl);
}

}  // namespace trtllm_mnnvl_ar_mhc_post
}  // namespace flashinfer
