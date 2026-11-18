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

// Specialized SM90 batch-decode (MTP/spec-verify) attention for an fp8
// (e4m3) query over an fp8 (e4m3) paged KV cache: GQA 8 query heads over
// 1 KV head, head_dim 128, page size 16, NHD layout, causal, bf16 output,
// UNIFORM q_len == 6 per request (the multi-token-prediction verify step
// this kernel is compiled for: QL is a compile-time constant). Per-sequence
// float32 Q/K/V descales ((batch, num_kv_heads) ABI; Q and K descales fold
// into the softmax scale, the V descale folds into the final merge).
//
// Algorithm: a split-KV fp8 tensor-core kernel. Each request's KV range is
// partitioned into whole BN=48-key tiles shared proportionally across a
// one-wave CTA budget (`share_of`, identical arithmetic in producer and
// reducer); QK^T runs as fp8 mma (m16n8k32 e4m3, fp32 accumulate) on
// cp.async-staged K pages; softmax statistics are fp32 online-softmax; the
// P.V product accumulates in fp16 (m16n8k16 on power-of-two prescaled
// values, exact for every finite e4m3 input, bounded by the 48-key x
// 24-tile x 448 range cap); per-split partial outputs (fp16) and LSEs
// (fp32) land in a caller-owned workspace and a second kernel (`reduce_u4`)
// merges the splits and applies the V descale, writing bf16. Both launches
// use programmatic stream serialization (PDL); the whole sequence is two
// kernels, no host round trips.
//
// Host contract of the device code (asserted by the Python dispatcher at
// plan time and by `run` below, never read back from the device):
//  * q / out are contiguous [batch * 6, 8, 128] (uniform q_len 6);
//  * k_cache / v_cache are contiguous [num_pages, 16, 1, 128] e4m3
//    (2048-byte pages);
//  * block_table is a contiguous int32 [batch, max_pages_per_seq] table of
//    physical page ids, one leading run per request;
//  * seq_lens (kv length) >= 6 per request (verify step: the 6 query rows
//    are the last 6 positions of the sequence);
//  * the caller-owned uint8 workspace holds at least
//    `plan_workspace_bytes(batch, max_pages_per_seq)` bytes (split partial
//    outputs + LSEs + per-request share metadata + the long-context carry
//    buffer when the page-table width requires it).
//
// `run` performs no allocation, no synchronisation and no attribute queries
// (the kernels use static shared memory only): it launches on the caller's
// current stream and is CUDA-graph capturable.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

#define HQ 8
#define DD 128
#define PAGE 16
#define QL 6
#define MROWS 48
#define BN 48
#define NTHREAD 128
#define TMAX 512
#define GMB 660
#define SMAX 660

__device__ __forceinline__ int sw8(int row, int byte) {
  int c = byte >> 4, o = byte & 15;
  return row * 128 + (((c ^ (row & 7))) << 4) + o;
}
__device__ __forceinline__ int sw16(int row, int h) {
  int byte = h << 1;
  int c = byte >> 4, o = byte & 15;
  return row * 256 + (((c ^ (row & 7))) << 4) + o;
}
__device__ __forceinline__ uint32_t cvt8x2(uint32_t x) {
  uint32_t r;
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)(x & 0xffffu)));
  // A power-of-two transform is exact for every finite e4m3 value.
  // 48 keys * 24 tiles * 448 / 16 stays below the FP16 accumulator range.
  __half2 h = __hmul2(*reinterpret_cast<__half2*>(&r), __float2half2_rn(0.0625f));
  return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ uint32_t smaddr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void async_copy16(void* dst, const void* src, bool valid) {
  uint32_t n = valid ? 16u : 0u;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(smaddr(dst)), "l"(src),
               "r"(n));
}
__device__ __forceinline__ void async_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void async_wait() { asm volatile("cp.async.wait_group 0;"); }
__device__ __forceinline__ void async_wait1() { asm volatile("cp.async.wait_group 1;"); }
#define LDM4(r0, r1, r2, r3, a)                                                \
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" \
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)                        \
               : "r"(a))
#define LDM4T(r0, r1, r2, r3, a)                                                     \
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" \
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)                              \
               : "r"(a))
#define STM4(a, r0, r1, r2, r3)                                                               \
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], {%1,%2,%3,%4};" ::"r"(a), \
               "r"(r0), "r"(r1), "r"(r2), "r"(r3)                                             \
               : "memory")
#define MMA_FP8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)                                    \
  asm volatile(                                                                            \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, " \
      "{%8,%9}, {%0,%1,%2,%3};"                                                            \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)                                             \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define MMA_F16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1)                                           \
  asm volatile(                                                                                   \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, " \
      "{%0,%1,%2,%3};"                                                                            \
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)                                                    \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define MMA_F16H(d0, d1, a0, a1, a2, a3, b0, b1)                                            \
  asm volatile(                                                                             \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, " \
      "{%0,%1};"                                                                            \
      : "+r"(d0), "+r"(d1)                                                                  \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

#define WGMMA_PV_N64(D, O, A0, A1, A2, A3, B)                                                \
  asm volatile(                                                                              \
      "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "                                  \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, "                            \
      "{%16,%17,%18,%19}, %20, 1, 1, 1, 1;"                                                  \
      : "+r"(D[O + 0]), "+r"(D[O + 1]), "+r"(D[O + 2]), "+r"(D[O + 3]), "+r"(D[O + 4]),      \
        "+r"(D[O + 5]), "+r"(D[O + 6]), "+r"(D[O + 7]), "+r"(D[O + 8]), "+r"(D[O + 9]),      \
        "+r"(D[O + 10]), "+r"(D[O + 11]), "+r"(D[O + 12]), "+r"(D[O + 13]), "+r"(D[O + 14]), \
        "+r"(D[O + 15])                                                                      \
      : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "l"(B))

__device__ __forceinline__ uint64_t wgmma_desc(const void* p) {
  uint64_t a = (uint64_t)smaddr(p);
  return ((a & 0x3ffffull) >> 4) | (64ull << 32) | (1ull << 62);
}

#define WGMMA_FP8_64x48(D, A, B)                                                                  \
  asm volatile(                                                                                   \
      "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "                                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, " \
      "%24, %25, 1, 1, 1;"                                                                        \
      : "+f"(D[0][0]), "+f"(D[0][1]), "+f"(D[0][2]), "+f"(D[0][3]), "+f"(D[1][0]), "+f"(D[1][1]), \
        "+f"(D[1][2]), "+f"(D[1][3]), "+f"(D[2][0]), "+f"(D[2][1]), "+f"(D[2][2]), "+f"(D[2][3]), \
        "+f"(D[3][0]), "+f"(D[3][1]), "+f"(D[3][2]), "+f"(D[3][3]), "+f"(D[4][0]), "+f"(D[4][1]), \
        "+f"(D[4][2]), "+f"(D[4][3]), "+f"(D[5][0]), "+f"(D[5][1]), "+f"(D[5][2]), "+f"(D[5][3])  \
      : "l"(A), "l"(B))

#define WGMMA_FP8_64x48_ZERO_D(D, A, B)                                                           \
  asm volatile(                                                                                   \
      "wgmma.mma_async.sync.aligned.m64n48k32.f32.e4m3.e4m3 "                                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23}, " \
      "%24, %25, 0, 1, 1;"                                                                        \
      : "+f"(D[0][0]), "+f"(D[0][1]), "+f"(D[0][2]), "+f"(D[0][3]), "+f"(D[1][0]), "+f"(D[1][1]), \
        "+f"(D[1][2]), "+f"(D[1][3]), "+f"(D[2][0]), "+f"(D[2][1]), "+f"(D[2][2]), "+f"(D[2][3]), \
        "+f"(D[3][0]), "+f"(D[3][1]), "+f"(D[3][2]), "+f"(D[3][3]), "+f"(D[4][0]), "+f"(D[4][1]), \
        "+f"(D[4][2]), "+f"(D[4][3]), "+f"(D[5][0]), "+f"(D[5][1]), "+f"(D[5][2]), "+f"(D[5][3])  \
      : "l"(A), "l"(B))

__device__ __forceinline__ uint32_t hmul2_bits(uint32_t x, __half2 y) {
  __half2 z = __hmul2(*reinterpret_cast<__half2*>(&x), y);
  return *reinterpret_cast<uint32_t*>(&z);
}

// Proportional whole-tile partitioning.  Each request receives a share of
// the GMB-CTA launch proportional to its own BN-tile count (cumulative
// floor keeps the shares summing to GMB), then its tiles are spread evenly
// over that share.  The old quantize-key-chunk-up-to-BN scheme rounded the
// per-CTA tile count up twice -- once for the chunk and once for the tile
// loop -- so the longest CTA carried a whole extra tile on every multi-
// request class.  GMB == 132 SMs * 4 resident CTAs, so the launch exactly
// fills the machine in one wave.
__device__ __forceinline__ int tcount(int len) { return (len + BN - 1) / BN; }

// Exact floor(tj * rem / remT) without a dynamic 64-bit integer divide.
// The quotient is bounded by the 528-CTA budget, so the float estimate is
// within one integer over the full int32 input range; the multiply-only
// correction restores exact integer-floor semantics.
__device__ __forceinline__ int fast_share_floor(int tj, int rem, int remT) {
  long long num = (long long)tj * rem;
  int q = __float2int_rd(__fdividef((float)num, (float)remT));
  if ((long long)(q + 1) * remT <= num) ++q;
  if ((long long)q * remT > num) --q;
  return q;
}

// Share of the launch owned by request i, and the first CTA index of that
// share.  Identical arithmetic in the producer and the reducer.
__device__ __forceinline__ int share_of(const int* __restrict__ sl, int B, int i, int& base) {
  int remT = 0;
#pragma unroll 1
  for (int j = 0; j < B; ++j) remT += tcount(sl[j]);
  int rem = B > GMB ? B : GMB, bs = 0, si = 1;
#pragma unroll 1
  for (int j = 0; j <= i; ++j) {
    int tj = tcount(sl[j]);
    si = fast_share_floor(tj, rem, remT);
    if (si < 1) si = 1;
    if (si > tj) si = tj;
    int keep = rem - (B - 1 - j);  // leave one CTA for each later request
    if (si > keep) si = keep;
    if (j < i) bs += si;
    rem -= si;
    remT -= tj;
  }
  base = bs;
  return si;
}

template <bool ASYNC_Q, bool SINGLE>
__global__ __launch_bounds__(NTHREAD, SINGLE ? 4 : 5) void attn_kernel(
    const uint8_t* __restrict__ qg, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const int* __restrict__ bt, const int* __restrict__ sl, const float* __restrict__ qd,
    const float* __restrict__ kd, __half* __restrict__ po, float* __restrict__ plse,
    int2* __restrict__ pmeta, float2* __restrict__ carry, int B, int NB, int GSTRIDE) {
  __shared__ __align__(128) uint8_t qs[64 * 128];
  // Each stage's raw K/V pair is contiguous.  Once QK has consumed K and
  // the V bytes have been loaded into registers, reuse the same 12 KiB as
  // the two f16 V halves for PV and, finally, as the partial-output tile.
  __shared__ __align__(128) uint8_t kv[2][2][BN * 128];

  const int tid = threadIdx.x;
  const int w = tid >> 5, lane = tid & 31;

  int bid = blockIdx.x, r = -1, s = 0, sbase = 0, sshare = 0;
  int t0 = 0, nt = 0;
  if constexpr (SINGLE) {
    // Distribute whole BN-tiles evenly over the launch instead of
    // quantizing a per-split key chunk up to a multiple of BN: that
    // rounding left ~11% of the grid idle while the active CTAs each
    // carried a full extra tile.  Now every CTA runs, and the tile
    // counts differ by at most one.
    int tt = tcount(sl[0]);
    int sp = tt < TMAX ? tt : TMAX;
    if (bid >= sp) return;
    int btile = tt / sp, rm = tt - btile * sp;
    t0 = bid * btile + (bid < rm ? bid : rm);
    nt = btile + (bid < rm ? 1 : 0);
    r = 0;
    s = bid;
  } else {
    // Water-fill the 528-CTA budget request by request: each request takes
    // a share of the CTAs still free in proportion to the tiles still
    // unassigned, so a request whose proportional share rounds below one
    // borrows its single CTA from the budget instead of pushing the launch
    // past the machine's 528 resident slots.  Overshooting by even one CTA
    // costs a second wave and roughly a third of the kernel's runtime.
    int remT = 0;
#pragma unroll 1
    for (int i = 0; i < B; ++i) remT += tcount(sl[i]);
    int rem = B > GMB ? B : GMB;
#pragma unroll 1
    for (int i = 0, base = 0; i < B; ++i) {
      int ti = tcount(sl[i]);
      int si = fast_share_floor(ti, rem, remT);
      if (si < 1) si = 1;
      if (si > ti) si = ti;
      int keep = rem - (B - 1 - i);
      if (si > keep) si = keep;
      if (bid < base + si) {
        r = i;
        s = bid - base;
        sbase = base;
        sshare = si;
        int btile = ti / si, rm = ti - btile * si;
        t0 = s * btile + (s < rm ? s : rm);
        nt = btile + (s < rm ? 1 : 0);
        break;
      }
      base += si;
      rem -= si;
      remT -= ti;
    }
    if (r < 0) return;
    if (tid == 0 && s == 0) pmeta[r] = make_int2(sbase, sshare);
  }

  const int kvl = sl[r];
  const int n0 = t0 * BN;
  int n1 = (t0 + nt) * BN;
  if (n1 > kvl) n1 = kvl;
  const int q0 = r * QL;

  // The first three warps retain the 48-row copy ownership.  The fourth
  // warp participates in WGMMA and zero-pads its 16 dummy query rows.
  // the chunk lane is invariant.  Derive both instead of keeping two
  // four-element index arrays live across the entire attention loop.
  const int copy_row0 = tid >> 3;
  const int copy_c = tid & 7;

  const int npg0 = (kvl + 15) >> 4;
  const int* btr0 = bt + (size_t)r * NB;
  if constexpr (ASYNC_Q) {
    const uint8_t* src = qg + (size_t)q0 * 1024;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int cr = copy_row0 + 16 * j;
      bool valid = cr < MROWS;
      const uint8_t* qsrc = valid ? src + cr * 128 + copy_c * 16 : qg;
      async_copy16(qs + sw8(cr, copy_c * 16), qsrc, valid);
    }
    async_commit();
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      int cr = copy_row0 + 16 * j;
      int key = n0 + cr;
      int pi = key >> 4;
      bool valid = key < n1 && pi < npg0;
      int pg = valid ? btr0[pi] : 0;
      size_t off = (size_t)pg * 2048 + (key & 15) * 128 + copy_c * 16;
      async_copy16(kv[0][0] + sw8(cr, copy_c * 16), kc + off, valid);
      async_copy16(kv[0][1] + sw8(cr, copy_c * 16), vc + off, valid);
    }
    async_commit();
    async_wait1();
  } else {
    const uint8_t* src = qg + (size_t)q0 * 1024;
    if (tid < 96) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int cr = copy_row0 + 12 * j;
        uint4 v = *reinterpret_cast<const uint4*>(src + cr * 128 + copy_c * 16);
        *reinterpret_cast<uint4*>(qs + sw8(cr, copy_c * 16)) = v;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int cr = 48 + (lane >> 1);
        int dc = (lane & 1) * 64 + j * 16;
        *reinterpret_cast<uint4*>(qs + sw8(cr, dc)) = make_uint4(0, 0, 0, 0);
      }
    }
  }

  const float scale2 = 0.08838834764831845f * qd[r] * kd[r] * 1.4426950408889634f;
  __syncthreads();

  uint32_t oa[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) oa[i] = 0u;
  float m_a = -1e30f, m_b = -1e30f, l_a = 0.f, l_b = 0.f;
  float cm_a = -1e30f, cm_b = -1e30f;

  const int ra = 16 * w + (lane >> 2);
  const int rb = ra + 8;
  int lim_a = kvl - (QL - 1) + (ra >> 3);
  if (lim_a > n1) lim_a = n1;
  int lim_b = kvl - (QL - 1) + (rb >> 3);
  if (lim_b > n1) lim_b = n1;
  const int key0 = (lane & 3) * 2;

  const int ntile = (n1 - n0 + BN - 1) / BN;

  const int npg = npg0;
  const int* btr = btr0;
  if constexpr (!ASYNC_Q) {
    if (tid < 96) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        int cr = copy_row0 + 12 * j;
        int key = n0 + cr;
        int pi = key >> 4;
        bool valid = key < n1 && pi < npg;
        int pg = valid ? btr[pi] : 0;
        size_t off = (size_t)pg * 2048 + (key & 15) * 128 + copy_c * 16;
        async_copy16(kv[0][0] + sw8(cr, copy_c * 16), kc + off, valid);
        async_copy16(kv[0][1] + sw8(cr, copy_c * 16), vc + off, valid);
      }
    }
    async_commit();
  }

  int stage = 0;
  for (int t = 0; t < ntile; ++t) {
    const int nb = n0 + t * BN;
    async_wait();
    __syncthreads();
    // The preceding tile's PV can overlap the next stage's cp.async
    // arrival.  Drain it before issuing copies into that aliased stage.
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    // The main loop retains the original FP16 register footprint. Only
    // long splits drain a 24-tile subtotal to FP32 scratch, with its own
    // softmax exponent so later maxima can rescale it at the next drain.
    if (t && t % 24 == 0 && w < 3) {
      const float ca = exp2f(cm_a - m_a), cb = exp2f(cm_b - m_b);
#pragma unroll
      for (int i = 0; i < 32; ++i) {
        const size_t ix = ((size_t)bid * 32 + i) * 96 + tid;
        float2 v = __half22float2(*reinterpret_cast<__half2*>(&oa[i]));
        if (t > 24) {
          float2 old = carry[ix];
          float scale = (i & 1) ? cb : ca;
          v.x = fmaf(old.x, scale, v.x);
          v.y = fmaf(old.y, scale, v.y);
        }
        carry[ix] = v;
        oa[i] = 0;
      }
      cm_a = m_a;
      cm_b = m_b;
    }
    if (t + 1 < ntile) {
      int nn = nb + BN;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        int cr = copy_row0 + 16 * j;
        int key = nn + cr;
        int pi = key >> 4;
        bool valid = key < n1 && pi < npg;
        int pg = valid ? btr[pi] : 0;
        size_t off = (size_t)pg * 2048 + (key & 15) * 128 + copy_c * 16;
        async_copy16(kv[stage ^ 1][0] + sw8(cr, copy_c * 16), kc + off, valid);
        async_copy16(kv[stage ^ 1][1] + sw8(cr, copy_c * 16), vc + off, valid);
      }
      async_commit();
    }
    float sm[6][4];
    // The first QK WGMMA overwrites D in the B1 specialization.  Keep
    // the explicit zero initialization for the multi-request path,
    // whose instruction stream is retained byte-for-byte.
    if constexpr (!SINGLE) {
#pragma unroll
      for (int i = 0; i < 6; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) sm[i][j] = 0.f;
    }

    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      uint64_t da = wgmma_desc(qs + kb * 32);
      uint64_t db = wgmma_desc(kv[stage][0] + kb * 32);
      if constexpr (SINGLE) {
        if (kb == 0)
          WGMMA_FP8_64x48_ZERO_D(sm, da, db);
        else
          WGMMA_FP8_64x48(sm, da, db);
      } else {
        WGMMA_FP8_64x48(sm, da, db);
      }
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

    // Stage V as f16 after the QK MMAs have issued: the ldmatrix/MMA
    // stream covers the LDS->cvt->STS latency, and the barrier that makes
    // converted V visible is deferred to just before PV instead of stalling here.
    uint2 vlo[4], vhi[4];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      int cr = copy_row0 + 16 * j;
      // Each lane owns chunk cg in each 64-byte half.  The matching
      // f16 stores below therefore touch consecutive 16-byte chunks
      // across the eight lanes of a row instead of every other chunk.
      vlo[j] = *reinterpret_cast<const uint2*>(kv[stage][1] + sw8(cr, copy_c * 8));
      vhi[j] = *reinterpret_cast<const uint2*>(kv[stage][1] + sw8(cr, 64 + copy_c * 8));
    }
    // Every logical row is owned by one 8-lane subgroup inside a single
    // warp.  Publish completion of all raw-V reads before lanes 4..7
    // overwrite that same row with the expanded second f16 half.
    __syncwarp();
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      int cr = copy_row0 + 16 * j;
      uint2 a = vlo[j], b = vhi[j];
      uint4 h0, h1;
      h0.x = cvt8x2(a.x);
      h0.y = cvt8x2(a.x >> 16);
      h0.z = cvt8x2(a.y);
      h0.w = cvt8x2(a.y >> 16);
      h1.x = cvt8x2(b.x);
      h1.y = cvt8x2(b.x >> 16);
      h1.z = cvt8x2(b.y);
      h1.w = cvt8x2(b.y >> 16);
      *reinterpret_cast<uint4*>((char*)kv[stage][0] + sw8(cr, copy_c * 16)) = h0;
      *reinterpret_cast<uint4*>((char*)kv[stage][1] + sw8(cr, copy_c * 16)) = h1;
    }
    // The f16 V tile above is written through the generic proxy, while the
    // PV WGMMAs read it through the async proxy via shared-memory
    // descriptors; wgmma.fence only orders register accesses.  Publish the
    // generic stores to the async proxy here, before the __syncthreads()
    // below makes them visible across the warpgroup.
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    // Scores stay in RAW (unscaled) units through the max reduction:
    // scale2 > 0 so the argmax is unchanged, and folding the scale into
    // the exp2 argument turns FMUL+FSUB into a single FFMA per element.
    // The causal/chunk mask is only live in the final tiles, so the
    // interior fast path drops 8 compare+select ops per key pair.
    if (w < 3) {
      float ma = -1e30f, mb = -1e30f;
      if (nb + BN <= lim_a) {
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          ma = fmaxf(ma, fmaxf(sm[j][0], sm[j][1]));
          mb = fmaxf(mb, fmaxf(sm[j][2], sm[j][3]));
        }
      } else {
#pragma unroll
        for (int j = 0; j < 6; ++j) {
          int kk = nb + key0 + j * 8;
          if (kk >= lim_a) sm[j][0] = -1e30f;
          if (kk + 1 >= lim_a) sm[j][1] = -1e30f;
          if (kk >= lim_b) sm[j][2] = -1e30f;
          if (kk + 1 >= lim_b) sm[j][3] = -1e30f;
          ma = fmaxf(ma, fmaxf(sm[j][0], sm[j][1]));
          mb = fmaxf(mb, fmaxf(sm[j][2], sm[j][3]));
        }
      }
      ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 1));
      ma = fmaxf(ma, __shfl_xor_sync(0xffffffffu, ma, 2));
      mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
      mb = fmaxf(mb, __shfl_xor_sync(0xffffffffu, mb, 2));
      float mna = fmaxf(fmaxf(m_a, ma * scale2), -1e20f);
      float mnb = fmaxf(fmaxf(m_b, mb * scale2), -1e20f);
      bool rescale = (mna != m_a) || (mnb != m_b);
      if (rescale) {
        float ca = exp2f(m_a - mna), cb = exp2f(m_b - mnb);
        l_a *= ca;
        l_b *= cb;
        __half2 ca2 = __float2half2_rn(ca), cb2 = __float2half2_rn(cb);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          __half2 ra2 = __hmul2(*reinterpret_cast<__half2*>(&oa[2 * i]), ca2);
          __half2 rb2 = __hmul2(*reinterpret_cast<__half2*>(&oa[2 * i + 1]), cb2);
          oa[2 * i] = *reinterpret_cast<uint32_t*>(&ra2);
          oa[2 * i + 1] = *reinterpret_cast<uint32_t*>(&rb2);
        }
      }
      m_a = mna;
      m_b = mnb;
      float sa = 0.f, sb = 0.f;
      const float nega = -mna, negb = -mnb;
#pragma unroll
      for (int j = 0; j < 6; ++j) {
        sm[j][0] = exp2f(fmaf(sm[j][0], scale2, nega));
        sm[j][1] = exp2f(fmaf(sm[j][1], scale2, nega));
        sm[j][2] = exp2f(fmaf(sm[j][2], scale2, negb));
        sm[j][3] = exp2f(fmaf(sm[j][3], scale2, negb));
        if (nb + BN > lim_a) {
          int kk = nb + key0 + j * 8;
          if (kk >= lim_a) sm[j][0] = 0.f;
          if (kk + 1 >= lim_a) sm[j][1] = 0.f;
          if (kk >= lim_b) sm[j][2] = 0.f;
          if (kk + 1 >= lim_b) sm[j][3] = 0.f;
        }
        sa += sm[j][0] + sm[j][1];
        sb += sm[j][2] + sm[j][3];
      }
      l_a += sa;
      l_b += sb;
    }

    __syncthreads();
    // PV as one warpgroup WGMMA per 64-dim half: A (the probabilities)
    // comes straight out of the softmax registers in the warpgroup RS
    // layout, B is the f16 V tile read MN-major from shared memory.  This
    // removes every ldmatrix from the inner loop -- each of the three
    // compute warps used to re-read the whole 12 KiB V tile through the
    // LSU -- and lets all four subcores issue the PV math.
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int kstep = 0; kstep < 3; ++kstep) {
      __half2 t0 = __floats2half2_rn(sm[2 * kstep][0], sm[2 * kstep][1]);
      __half2 t1 = __floats2half2_rn(sm[2 * kstep][2], sm[2 * kstep][3]);
      __half2 t2 = __floats2half2_rn(sm[2 * kstep + 1][0], sm[2 * kstep + 1][1]);
      __half2 t3 = __floats2half2_rn(sm[2 * kstep + 1][2], sm[2 * kstep + 1][3]);
      uint32_t a0 = *reinterpret_cast<uint32_t*>(&t0);
      uint32_t a1 = *reinterpret_cast<uint32_t*>(&t1);
      uint32_t a2 = *reinterpret_cast<uint32_t*>(&t2);
      uint32_t a3 = *reinterpret_cast<uint32_t*>(&t3);
      uint64_t b0 = wgmma_desc((char*)kv[stage][0] + kstep * (16 * 128));
      uint64_t b1 = wgmma_desc((char*)kv[stage][1] + kstep * (16 * 128));
      WGMMA_PV_N64(oa, 0, a0, a1, a2, a3, b0);
      WGMMA_PV_N64(oa, 16, a0, a1, a2, a3, b1);
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
    stage ^= 1;
  }
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");

  // Undo only the exact power-of-two PV prescale here. The per-sequence V
  // descale is applied by the reducer in FP32: folding it into this FP16
  // normalization would overflow the stored partials for a large descale
  // (448 * 256 > 65504) and underflow them for a small one.
  const float vsc = 16.f;
  float la = l_a, lb = l_b, inva = 0.f, invb = 0.f;
  if (w < 3) {
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    inva = (la > 0.f) ? (vsc / la) : 0.f;
    invb = (lb > 0.f) ? (vsc / lb) : 0.f;
  }
  __half2 inva2 = __float2half2_rn(inva), invb2 = __float2half2_rn(invb);

  // Long splits have an FP32 subtotal. A large inverse also needs FP32
  // normalization so scaling V cannot introduce an FP16 inverse overflow.
  if (w < 3 && (ntile > 24 || fabsf(inva) > 65504.f || fabsf(invb) > 65504.f)) {
    const float ca = exp2f(cm_a - m_a), cb = exp2f(cm_b - m_b);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      float2 v = __half22float2(*reinterpret_cast<__half2*>(&oa[i]));
      if (ntile > 24) {
        const size_t ix = ((size_t)bid * 32 + i) * 96 + tid;
        float2 old = carry[ix];
        float scale = (i & 1) ? cb : ca;
        v.x = fmaf(old.x, scale, v.x);
        v.y = fmaf(old.y, scale, v.y);
      }
      float inv = (i & 1) ? invb : inva;
      __half2 h = __floats2half2_rn(v.x * inv, v.y * inv);
      oa[i] = *reinterpret_cast<uint32_t*>(&h);
    }
    // The accumulated partial has already been normalized in FP32.
    inva2 = __float2half2_rn(1.f);
    invb2 = inva2;
  }

  // Let the dependent reducer grid begin scheduling while this CTA drains
  // the coalesced partial-output and LSE epilogue.
  cudaTriggerProgrammaticLaunchCompletion();

  __half* pob = po + (size_t)bid * (MROWS * 128);
  __syncthreads();
  char* ob = (char*)kv[stage ^ 1];
#pragma unroll
  for (int dt = 0; dt < 16; dt += 2) {
    if (w < 3) {
      oa[2 * dt] = hmul2_bits(oa[2 * dt], inva2);
      oa[2 * dt + 1] = hmul2_bits(oa[2 * dt + 1], invb2);
      oa[2 * dt + 2] = hmul2_bits(oa[2 * dt + 2], inva2);
      oa[2 * dt + 3] = hmul2_bits(oa[2 * dt + 3], invb2);
      int row = 16 * w + (lane & 15);
      int col = (dt + (lane >> 4)) * 8;
      uint32_t ad = smaddr(ob + sw16(row, col));
      STM4(ad, oa[2 * dt], oa[2 * dt + 1], oa[2 * dt + 2], oa[2 * dt + 3]);
    }
  }
  __syncthreads();
  if (tid == 0) {
    // The online-softmax fragments are already staged in a complete
    // 12 KiB shared tile.  Store that tile with Hopper bulk copy and
    // retain its XOR swizzle in scratch; the reducer addresses the same
    // logical 8-half chunks through sw16().
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 12288;" ::"l"(pob),
                 "r"(smaddr(ob))
                 : "memory");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
  }
  if (w < 3 && (lane & 3) == 0) {
    // LSE row-major over splits: the merge streams a contiguous run of S
    // floats instead of touching one 32 B sector per split.
    plse[(size_t)ra * GSTRIDE + bid] = (la > 0.f) ? (m_a + log2f(la)) : -3e30f;
    plse[(size_t)rb * GSTRIDE + bid] = (lb > 0.f) ? (m_b + log2f(lb)) : -3e30f;
  }
  if (tid == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void acc8(float* A, uint4 v, float w) {
  __half2 h0 = *reinterpret_cast<__half2*>(&v.x);
  __half2 h1 = *reinterpret_cast<__half2*>(&v.y);
  __half2 h2 = *reinterpret_cast<__half2*>(&v.z);
  __half2 h3 = *reinterpret_cast<__half2*>(&v.w);
  A[0] = fmaf(w, __low2float(h0), A[0]);
  A[1] = fmaf(w, __high2float(h0), A[1]);
  A[2] = fmaf(w, __low2float(h1), A[2]);
  A[3] = fmaf(w, __high2float(h1), A[3]);
  A[4] = fmaf(w, __low2float(h2), A[4]);
  A[5] = fmaf(w, __high2float(h2), A[5]);
  A[6] = fmaf(w, __low2float(h3), A[6]);
  A[7] = fmaf(w, __high2float(h3), A[7]);
}

// Split-merge.  Each thread owns 8 contiguous output dims as one uint4 of
// halves and keeps 4 splits in flight, so the memory-level parallelism is
// 16x the old one-__half2-per-thread form; the merge is pure latency-bound
// streaming over the partial buffer.
template <int NTH, int NSLICE, bool SGL>
__global__ __launch_bounds__(NTH, 2) void reduce_u4(
    const __half* __restrict__ po, const float* __restrict__ plse, __nv_bfloat16* __restrict__ out,
    const int* __restrict__ sl, const float* __restrict__ vd, const int2* __restrict__ pmeta, int B,
    int GSTRIDE) {
  constexpr int DPC = 128 / NSLICE;
  constexpr int LANES = DPC / 8;
  constexpr int GROUPS = NTH / LANES;
  constexpr int NW = NTH / 32;
  constexpr int LPT = (SMAX + NTH - 1) / NTH;

  int gg = blockIdx.x;
  int ds = 0, g = gg;
  if constexpr (NSLICE > 1) {
    ds = gg & (NSLICE - 1);
    g = gg / NSLICE;
  }
  // pmeta below is written by the producer grid, and under programmatic
  // dependent launch this grid may start before the producer completes:
  // synchronize before the first read of producer state. The SGL
  // specialization derives its split metadata from sl alone and keeps the
  // full preamble overlap until its own synchronize below.
  if constexpr (!SGL) cudaGridDependencySynchronize();
  // No producer owns an empty request. Bypass its unwritten split metadata.
  int request = SGL ? 0 : g / MROWS;
  if (sl[request] == 0) {
    if constexpr (SGL) cudaGridDependencySynchronize();
    for (int d = threadIdx.x; d < DPC; d += NTH)
      out[(size_t)g * 128 + ds * DPC + d] = __float2bfloat16(0.f);
    return;
  }
  int r, m, base = 0, S;
  if constexpr (SGL) {
    r = 0;
    m = g;
    int tt = tcount(sl[0]);
    S = tt < TMAX ? tt : TMAX;
  } else {
    r = g / MROWS;
    m = g - r * MROWS;
    int2 md = pmeta[r];
    base = md.x;
    S = md.y;
  }

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  const int dl = (tid & (LANES - 1)) * 8;
  const int grp = tid / LANES;
  const int d = ds * DPC + dl;

  const char* pb =
      reinterpret_cast<const char*>(po) + (size_t)base * MROWS * 128 * sizeof(__half) + sw16(m, d);
  const float* lb = plse + (size_t)m * GSTRIDE + base;

  __shared__ float sW[SMAX];
  __shared__ float sRed[NW];
  __shared__ float sMax, sInv;
  __shared__ float sA[NW * DPC];

  // Single-request index setup above overlaps the producer tail: wait only
  // at the first read of state written by the attention grid. The
  // multi-request path already synchronized before loading pmeta.
  if constexpr (SGL) cudaGridDependencySynchronize();
  // Pull the first eight split vectors into registers before the LSE
  // reduction.  Their memory latency is then hidden by the max/exp/sum
  // chain; for B1 this covers the complete 512-split accumulation loop.
  constexpr size_t ST = (size_t)MROWS * 128 * sizeof(__half);
  constexpr int PFD = 8;
  int npf = (S > grp) ? ((S - 1 - grp) / GROUPS + 1) : 0;
  if (npf > PFD) npf = PFD;
  uint4 pv[PFD];
#pragma unroll
  for (int i = 0; i < PFD; ++i) {
    int s = grp + i * GROUPS;
    if (i < npf) pv[i] = *reinterpret_cast<const uint4*>(pb + (size_t)s * ST);
  }
  float lv[LPT];
  float x = -3e30f;
#pragma unroll
  for (int i = 0; i < LPT; ++i) {
    int idx = tid + i * NTH;
    lv[i] = (idx < S) ? lb[idx] : -3e30f;
    x = fmaxf(x, lv[i]);
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, off));
  if (lane == 0) sRed[warp] = x;
  __syncthreads();
  if (tid == 0) {
    float y = sRed[0];
#pragma unroll
    for (int i = 1; i < NW; ++i) y = fmaxf(y, sRed[i]);
    sMax = y;
  }
  __syncthreads();
  const float mx = sMax;
  float ws = 0.f;
#pragma unroll
  for (int i = 0; i < LPT; ++i) {
    int idx = tid + i * NTH;
    float w = (idx < S) ? exp2f(lv[i] - mx) : 0.f;
    if (idx < S) sW[idx] = w;
    ws += w;
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) ws += __shfl_down_sync(0xffffffffu, ws, off);
  if (lane == 0) sRed[warp] = ws;
  __syncthreads();
  if (tid == 0) {
    float z = 0.f;
#pragma unroll
    for (int i = 0; i < NW; ++i) z += sRed[i];
    sInv = 1.f / z;
  }
  // Publish both the per-split weights and normalization to every warp.
  // The preceding barrier only makes sRed visible to thread 0; without this
  // one, consumers can race the sW stores or thread 0's sInv store.
  __syncthreads();

  float A[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) A[i] = 0.f;
#pragma unroll
  for (int i = 0; i < PFD; ++i)
    if (i < npf) acc8(A, pv[i], sW[grp + i * GROUPS]);
  int s = grp + PFD * GROUPS;
  for (; s + 3 * GROUPS < S; s += 4 * GROUPS) {
    uint4 v0 = *reinterpret_cast<const uint4*>(pb + (size_t)s * ST);
    uint4 v1 = *reinterpret_cast<const uint4*>(pb + (size_t)(s + GROUPS) * ST);
    uint4 v2 = *reinterpret_cast<const uint4*>(pb + (size_t)(s + 2 * GROUPS) * ST);
    uint4 v3 = *reinterpret_cast<const uint4*>(pb + (size_t)(s + 3 * GROUPS) * ST);
    float w0 = sW[s], w1 = sW[s + GROUPS], w2 = sW[s + 2 * GROUPS], w3 = sW[s + 3 * GROUPS];
    acc8(A, v0, w0);
    acc8(A, v1, w1);
    acc8(A, v2, w2);
    acc8(A, v3, w3);
  }
  for (; s < S; s += GROUPS) {
    uint4 v0 = *reinterpret_cast<const uint4*>(pb + (size_t)s * ST);
    acc8(A, v0, sW[s]);
  }

#pragma unroll
  for (int off = LANES; off < 32; off <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; ++i) A[i] += __shfl_xor_sync(0xffffffffu, A[i], off);
  }
  if (lane < LANES) {
#pragma unroll
    for (int i = 0; i < 8; ++i) sA[warp * DPC + dl + i] = A[i];
  }
  __syncthreads();
  if (tid < DPC) {
    float acc = 0.f;
#pragma unroll
    for (int w2 = 0; w2 < NW; ++w2) acc += sA[w2 * DPC + tid];
    size_t od = (size_t)g * 128 + ds * DPC + tid;
    // The per-sequence V descale lands here, in FP32, so it can neither
    // overflow nor underflow the FP16 split partials.
    out[od] = __float2bfloat16(acc * sInv * vd[r]);
  }
}

// ---------------------------------------------------------------------------
// Host side. The grid/split budget and the two-kernel launch sequence are
// carried verbatim from the measured kernel's
// same-SKU re-time; only the workspace ownership moved out of the launcher
// (plan-time allocation by the Python dispatcher — the measured launcher
// kept a per-stream cudaMallocAsync pool) so that `run` never allocates and
// is CUDA-graph capturable with a stable workspace pointer.
// ---------------------------------------------------------------------------

namespace {

struct GqaDecodeWsLayout {
  int grid;
  int lstride;
  size_t partial_bytes;
  size_t lse_bytes;
  size_t carry_offset;
  bool needs_carry;
  size_t total_bytes;
};

// Verbatim workspace arithmetic of the measured launcher: one-wave 528-CTA
// budget for the captured single-request class, at least one CTA per
// request over the unbounded-batch scope, carry storage only when the
// page-table width can push a split past 24 tiles.
GqaDecodeWsLayout gqa_decode_fp8_paged_sm90_ws_layout(int B, int NB) {
  GqaDecodeWsLayout L;
  L.grid = (B == 1) ? TMAX : (B > GMB ? B : GMB);
  L.lstride = (L.grid + 31) & ~31;
  L.partial_bytes = (size_t)L.grid * MROWS * 128 * sizeof(__half);
  L.lse_bytes = (size_t)MROWS * L.lstride * sizeof(float);
  L.carry_offset = L.partial_bytes + L.lse_bytes + ((B * sizeof(int2) + 127) & ~size_t(127));
  const size_t max_tiles = ((size_t)NB * PAGE + BN - 1) / BN;
  L.needs_carry = B == 1 ? max_tiles > size_t(TMAX) * 24 : max_tiles > 24;
  L.total_bytes =
      L.carry_offset + (L.needs_carry ? (size_t)L.grid * MROWS * 128 * sizeof(float) : 0);
  return L;
}

}  // namespace

// Workspace bytes the dispatcher must plan-allocate for a (batch,
// max_pages_per_seq) plan.
int64_t gqa_decode_fp8_paged_sm90_plan_workspace_bytes(int64_t batch_size,
                                                       int64_t max_pages_per_seq) {
  TVM_FFI_ICHECK_GE(batch_size, 1);
  TVM_FFI_ICHECK_GE(max_pages_per_seq, 1);
  return (int64_t)gqa_decode_fp8_paged_sm90_ws_layout(static_cast<int>(batch_size),
                                                      static_cast<int>(max_pages_per_seq))
      .total_bytes;
}

// Launch the measured two-kernel sequence on the caller's current stream:
// attn_kernel (split-KV fp8 attention over the proportional whole-tile
// partition) then reduce_u4 (split merge + V descale) under programmatic
// stream serialization. No allocation, no sync, no attribute queries:
// CUDA-graph capturable. The workspace is plan-allocated by the Python
// dispatcher.
void gqa_decode_fp8_paged_sm90_run(TensorView q, TensorView k_cache, TensorView v_cache,
                                   TensorView block_table, TensorView seq_lens,
                                   TensorView q_descale, TensorView k_descale, TensorView v_descale,
                                   TensorView out, TensorView workspace) {
  const int B = static_cast<int>(block_table.size(0));
  const int NB = static_cast<int>(block_table.size(1));
  const int TQ = static_cast<int>(q.size(0));

  CHECK_INPUT_AND_TYPE(q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_DIM(3, q);
  CHECK_DIM(3, out);
  TVM_FFI_ICHECK_EQ(q.size(1), HQ);
  TVM_FFI_ICHECK_EQ(q.size(2), DD);
  TVM_FFI_ICHECK_EQ(TQ, (int64_t)B * QL)
      << "this kernel serves uniform q_len == " << QL << " verify batches";
  TVM_FFI_ICHECK_EQ(out.size(0), TQ);
  TVM_FFI_ICHECK_EQ(out.size(1), HQ);
  TVM_FFI_ICHECK_EQ(out.size(2), DD);
  CHECK_CUDA(k_cache);
  CHECK_CUDA(v_cache);
  CHECK_INPUT_TYPE(k_cache, dl_float8_e4m3fn);
  CHECK_INPUT_TYPE(v_cache, dl_float8_e4m3fn);
  CHECK_DIM(4, k_cache);
  CHECK_DIM(4, v_cache);
  for (const TensorView* cache : {&k_cache, &v_cache}) {
    TVM_FFI_ICHECK_EQ(cache->size(1), PAGE) << "page size must be 16";
    TVM_FFI_ICHECK_EQ(cache->size(2), 1) << "exactly one KV head";
    TVM_FFI_ICHECK_EQ(cache->size(3), DD) << "KV head_dim must be 128";
    TVM_FFI_ICHECK_EQ(cache->stride(3), 1);
    TVM_FFI_ICHECK_EQ(cache->stride(2), DD);
    TVM_FFI_ICHECK_EQ(cache->stride(1), DD);
    TVM_FFI_ICHECK_EQ(cache->stride(0), PAGE * DD) << "KV pages must be 2048 contiguous bytes";
  }
  // The kernel issues 16-byte cp.async loads from the Q/K/V base pointers;
  // PTX requires natural alignment, and contiguity alone does not establish
  // it (a view can start mid-storage).
  for (const TensorView* t : {&q, &k_cache, &v_cache}) {
    TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16, 0)
        << "q/k_cache/v_cache base pointers must be 16-byte aligned";
  }
  CHECK_INPUT_AND_TYPE(block_table, dl_int32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(q_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(k_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(v_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_DIM(2, block_table);
  TVM_FFI_ICHECK_GE(seq_lens.numel(), B);
  TVM_FFI_ICHECK_GE(q_descale.numel(), B);
  TVM_FFI_ICHECK_GE(k_descale.numel(), B);
  TVM_FFI_ICHECK_GE(v_descale.numel(), B);

  const GqaDecodeWsLayout L = gqa_decode_fp8_paged_sm90_ws_layout(B, NB);
  TVM_FFI_ICHECK_GE((size_t)workspace.numel(), L.total_bytes)
      << "workspace too small: plan with plan_workspace_bytes(" << B << ", " << NB << ")";

  CHECK_DEVICE(q, k_cache);
  CHECK_DEVICE(q, v_cache);
  CHECK_DEVICE(q, out);
  CHECK_DEVICE(q, block_table);
  CHECK_DEVICE(q, seq_lens);
  CHECK_DEVICE(q, q_descale);
  CHECK_DEVICE(q, k_descale);
  CHECK_DEVICE(q, v_descale);
  CHECK_DEVICE(q, workspace);

  ffi::CUDADeviceGuard device_guard(q.device().device_id);
  const cudaStream_t stream = get_stream(q.device());

  char* scratch = static_cast<char*>(workspace.data_ptr());
  __half* po = reinterpret_cast<__half*>(scratch);
  float* plse = reinterpret_cast<float*>(scratch + L.partial_bytes);
  int2* pmeta = reinterpret_cast<int2*>(scratch + L.partial_bytes + L.lse_bytes);
  float2* carry = reinterpret_cast<float2*>(scratch + L.carry_offset);

  const uint8_t* qd = static_cast<const uint8_t*>(q.data_ptr());
  const uint8_t* kd = static_cast<const uint8_t*>(k_cache.data_ptr());
  const uint8_t* vd = static_cast<const uint8_t*>(v_cache.data_ptr());
  const int* bt = static_cast<const int*>(block_table.data_ptr());
  const int* slp = static_cast<const int*>(seq_lens.data_ptr());
  const float* qds = static_cast<const float*>(q_descale.data_ptr());
  const float* kds = static_cast<const float*>(k_descale.data_ptr());
  const float* vds = static_cast<const float*>(v_descale.data_ptr());
  __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out.data_ptr());

  if (B == 1) {
    attn_kernel<true, true><<<L.grid, NTHREAD, 0, stream>>>(qd, kd, vd, bt, slp, qds, kds, po, plse,
                                                            pmeta, carry, B, NB, L.lstride);
  } else {
    attn_kernel<true, false><<<L.grid, NTHREAD, 0, stream>>>(qd, kd, vd, bt, slp, qds, kds, po,
                                                             plse, pmeta, carry, B, NB, L.lstride);
  }
  cudaLaunchAttribute attribute[1] = {};
  attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attribute[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config = {};
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = attribute;
  config.numAttrs = 1;
  cudaError_t launch_status;
  if (B >= 6) {
    config.gridDim = B * MROWS;
    config.blockDim = 256;
    launch_status = cudaLaunchKernelEx(&config, reduce_u4<256, 1, false>, po, plse, op, slp, vds,
                                       pmeta, B, L.lstride);
  } else if (B == 1) {
    config.gridDim = MROWS * 2;
    config.blockDim = 512;
    launch_status = cudaLaunchKernelEx(&config, reduce_u4<512, 2, true>, po, plse, op, slp, vds,
                                       pmeta, B, L.lstride);
  } else if (B == 2) {
    config.gridDim = B * MROWS * 2;
    config.blockDim = 512;
    launch_status = cudaLaunchKernelEx(&config, reduce_u4<512, 2, false>, po, plse, op, slp, vds,
                                       pmeta, B, L.lstride);
  } else {
    config.gridDim = B * MROWS;
    config.blockDim = 512;
    launch_status = cudaLaunchKernelEx(&config, reduce_u4<512, 1, false>, po, plse, op, slp, vds,
                                       pmeta, B, L.lstride);
  }
  TVM_FFI_ICHECK(launch_status == cudaSuccess)
      << "gqa_decode_fp8_paged_sm90 reduce launch failed: " << cudaGetErrorString(launch_status);
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "gqa_decode_fp8_paged_sm90 launch failed: " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gqa_decode_fp8_paged_sm90_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_workspace_bytes, gqa_decode_fp8_paged_sm90_plan_workspace_bytes);
