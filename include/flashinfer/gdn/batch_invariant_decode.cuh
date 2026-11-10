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
#ifndef FLASHINFER_GDN_BATCH_INVARIANT_DECODE_CUH_
#define FLASHINFER_GDN_BATCH_INVARIANT_DECODE_CUH_

// Decode step of the class gdn.bf16.c64; build without fast math and with -fmad=false.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace flashinfer {
namespace gdn {
namespace bi_decode {

constexpr int D = 128;
constexpr int CH = 64;
constexpr int NTHR = 256;
constexpr int MINB = 4;
constexpr int SD = D + 8;
constexpr int SC = CH + 8;
constexpr int XS = CH;
constexpr int FS = 68;
constexpr size_t kSmemBytes = 56640;
// phase-count cell values: bit 8 asks the step to refresh the bf16 state copy, negative marks rows
// not built
constexpr int kStale = 256;
constexpr int kPending = -1;

struct Acc4 {
  float c[4];
};

__device__ __forceinline__ void mma_bf16(Acc4& d, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d.c[0]), "+f"(d.c[1]), "+f"(d.c[2]), "+f"(d.c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ld32(const uint16_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

__device__ __forceinline__ float bf2f(uint32_t x) { return __uint_as_float((x & 0xffffu) << 16); }
__device__ __forceinline__ uint32_t f2bf_x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ uint32_t f2bf(float f) { return f2bf_x2(f, 0.0f) & 0xffffu; }
__device__ __forceinline__ float lg2a(float x) {
  float y;
  asm("lg2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float ex2a(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rsqa(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ size_t rup16(size_t x) { return (x + 15) & ~(size_t)15; }

__device__ __forceinline__ void cp_async16(void* sdst, const void* gsrc) {
  const unsigned s = (unsigned)__cvta_generic_to_shared(sdst);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ void mma_k8(Acc4& d, uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d.c[0]), "+f"(d.c[1]), "+f"(d.c[2]), "+f"(d.c[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

struct AReg {
  uint32_t a[32];
};
struct BReg {
  uint32_t b[16];
};
struct TReg {
  uint32_t a[16];
};
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                        const uint16_t* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x4t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
                                         const uint16_t* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldsm_x2t(uint32_t& r0, uint32_t& r1, const uint16_t* p) {
  const unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}
// A fragments of a 16-row group for 8 k-steps of a row-major tile (row stride `stride`).
__device__ __forceinline__ void load_a(AReg& r, const uint16_t* base, int stride, int lane) {
  const uint16_t* p =
      base + (size_t)((lane & 7) + 8 * ((lane >> 3) & 1)) * stride + 8 * (lane >> 4);
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
    ldsm_x4(r.a[4 * ks], r.a[4 * ks + 1], r.a[4 * ks + 2], r.a[4 * ks + 3], p + 16 * ks);
}
// B fragments for 8 k-steps: `row` is this lane's column vector.
__device__ __forceinline__ void load_b(BReg& r, const uint16_t* row, int lane) {
  const uint16_t* p = row + 8 * ((lane >> 3) & 1) + 16 * (lane >> 4);
#pragma unroll
  for (int x = 0; x < 4; ++x)
    ldsm_x4(r.b[4 * x], r.b[4 * x + 1], r.b[4 * x + 2], r.b[4 * x + 3], p + 32 * x);
}
__device__ __forceinline__ Acc4 mma128(const AReg& A, const BReg& B) {
  Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
    mma_bf16(d, A.a[4 * ks], A.a[4 * ks + 1], A.a[4 * ks + 2], A.a[4 * ks + 3], B.b[2 * ks],
             B.b[2 * ks + 1]);
  return d;
}
// A fragments A[v][j] of a 16-value group over the tokens, from a [token][value] tile through
// transposing loads.
__device__ __forceinline__ void load_at(TReg& r, const uint16_t* tile, int v16, int lane, int nks) {
  const uint16_t* pa =
      tile + (size_t)((lane & 7) + 8 * (lane >> 4)) * SD + v16 + 8 * ((lane >> 3) & 1);
#pragma unroll
  for (int js = 0; js < 4; ++js)
    if (js < nks)
      ldsm_x4t(r.a[4 * js], r.a[4 * js + 1], r.a[4 * js + 2], r.a[4 * js + 3],
               pa + (size_t)(16 * js) * SD);
}

// normalize_qk's operation order for one 128-wide row (16 lanes of 8 elements).
__device__ __forceinline__ uint4 norm_row(uint4 raw) {
  const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
  float x[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) x[i] = bf2f(w[i >> 1] >> (16 * (i & 1)));
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    ss = ss + x[i] * x[i];
  }
  for (int off = 8; off > 0; off >>= 1) ss = ss + __shfl_xor_sync(0xffffffffu, ss, off);
  const float inv = rsqa(ss + 1e-6f);
  uint4 o;
  o.x = f2bf_x2(x[0] * inv, x[1] * inv);
  o.y = f2bf_x2(x[2] * inv, x[3] * inv);
  o.z = f2bf_x2(x[4] * inv, x[5] * inv);
  o.w = f2bf_x2(x[6] * inv, x[7] * inv);
  return o;
}

// Hillis-Steele inclusive scan of the log2 gates over 64 positions (lane owns positions lane and
// lane + 32).
__device__ __forceinline__ void gate_scan(float& g0, float& g1, int lane) {
  for (int off = 1; off < 32; off <<= 1) {
    const float n0 = __shfl_up_sync(0xffffffffu, g0, off),
                n1 = __shfl_up_sync(0xffffffffu, g1, off);
    if (lane >= off) {
      g0 = g0 + n0;
      g1 = g1 + n1;
    }
  }
  g1 = g1 + __shfl_sync(0xffffffffu, g0, 31);
}

// One CTA per (request, v-head); shared memory is reused within a step so four CTAs fit an SM.
__global__ void __launch_bounds__(NTHR, MINB)
    DecodeStepKernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                     const uint16_t* __restrict__ v, const float* __restrict__ gl,
                     const float* __restrict__ gb, const int* __restrict__ idx,
                     uint16_t* __restrict__ out, float* state, uint16_t* state16, int* cnt,
                     float* ring_l, float* ring_b, uint16_t* Kn, uint16_t* Mc, uint16_t* Xc,
                     uint16_t* Vc, uint16_t* Nc, uint16_t* hq, uint16_t* hk, uint16_t* hv,
                     float* hg, float* hb, float scale, int HQ, int HV, int num_slots) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int h = blockIdx.x % HV, b = blockIdx.x / HV;
  const int slot = idx[b];
  const bool no_slot = slot < 0 || slot >= num_slots;
  const int kh = h / (HV / HQ);
  const bool owner = (h % (HV / HQ)) == 0;
  int* cnt_cell = cnt + (size_t)(no_slot ? 0 : slot) * HV + h;
  const int c0 = no_slot ? -1 : *reinterpret_cast<volatile int*>(cnt_cell);
  if (c0 < 0) {
    // no such slot, or its cached rows are not built (kPending): a NaN row, never another
    // sequence's bits
    if (tid < D) out[((size_t)b * HV + h) * D + tid] = 0x7fc0;
    return;
  }
  const int p = c0 & 63;
  const bool stale = (c0 >> 8) & 1;
  const size_t bt = b;

  size_t off = 0;
  uint16_t* Ssm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)D * SD * 2;
  uint16_t* Ksm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)CH * SD * 2;
  uint16_t* Qsm = reinterpret_cast<uint16_t*>(smem + off);
  off += rup16(SD * 2);
  uint16_t* knrow = reinterpret_cast<uint16_t*>(smem + off);
  off += rup16(SD * 2);
  uint16_t* xrow = reinterpret_cast<uint16_t*>(smem + off);
  off += rup16(SC * 2);
  uint16_t* ainv = reinterpret_cast<uint16_t*>(smem + off);
  off += rup16(SC * 2);
  uint16_t* wrow = reinterpret_cast<uint16_t*>(smem + off);
  off += rup16(SC * 2);
  uint16_t* vrow = reinterpret_cast<uint16_t*>(smem + off);
  off += D * 2;
  float* lsm = reinterpret_cast<float*>(smem + off);
  off += CH * 4;
  float* bsm = reinterpret_cast<float*>(smem + off);
  off += CH * 4;
  float* cssm = reinterpret_cast<float*>(smem + off);
  off += CH * 4;
  float* Tsm = reinterpret_cast<float*>(smem + off);
  off += CH * 4;
  float* ABsm = reinterpret_cast<float*>(smem + off);
  off += 2 * D * 4;
  float* KQsm = reinterpret_cast<float*>(smem + off);
  off += 2 * CH * 4;
  float* nvsm = reinterpret_cast<float*>(smem + off);
  off += D * 4;
  float* misc = reinterpret_cast<float*>(smem + off);
  uint16_t* const Vsm = Ssm;
  uint16_t* const Nsm = Ssm + (size_t)CH * SD;
  uint16_t* const Xsm = Ksm;
  uint16_t* const Msm = Ksm + (size_t)(CH / 2) * SD;

  const size_t sh = (size_t)slot * HV + h;
  uint16_t* S16 = state16 + sh * D * D;
  float* S32 = state + sh * D * D;
  uint16_t* Kpool = Kn + ((size_t)slot * CH * HQ + kh) * D;  // row j at + j * HQ * D
  uint16_t* Mpool = Mc + sh * CH * CH;
  uint16_t* Xpool = Xc + sh * CH * CH;
  uint16_t* Vpool = Vc + sh * CH * D;
  uint16_t* Npool = Nc + sh * CH * D;
  float* Lpool = ring_l + sh * CH;
  float* Bpool = ring_b + sh * CH;
  const size_t hrow = (size_t)slot * CH + p;
  const int mlo = p >= 32 ? 32 : (p >= 16 ? 16 : (p >= 8 ? 8 : 0)),
            xhi = p == CH - 1 ? CH - 1 : (p & ~7);
#define MEL(r, c) Msm[(size_t)((r) - mlo) * SC + (c)]
#define XEL(r, c) Xsm[(size_t)(r) * XS + (c)]

  // ---- loads, all issued before the first dependent use
  if (!stale)
    for (int u = tid; u < D * 16; u += NTHR)
      cp_async16(Ssm + (u >> 4) * SD + (u & 15) * 8, S16 + (size_t)u * 8);
  for (int u = tid; u < p * 16; u += NTHR) {
    const int j = u >> 4, c8 = (u & 15) * 8;
    cp_async16(Ksm + j * SD + c8, Kpool + (size_t)j * HQ * D + c8);
  }
  cp_async_commit();
  float lj = 0.0f, bj = 0.0f;
  if (tid < p) {
    lj = Lpool[tid];
    bj = Bpool[tid];
  }
  uint4 raw = make_uint4(0, 0, 0, 0);
  if (warp == 0) raw = *reinterpret_cast<const uint4*>(k + (bt * HQ + kh) * D + 8 * (lane & 15));
  if (warp == 1) raw = *reinterpret_cast<const uint4*>(q + (bt * HQ + kh) * D + 8 * (lane & 15));
  if (warp == 2) raw = *reinterpret_cast<const uint4*>(v + (bt * HV + h) * D + 8 * (lane & 15));
  float graw = 0.0f, braw = 0.0f;
  if (tid == 96) {
    graw = gl[bt * HV + h];
    braw = gb[bt * HV + h];
  }
  if (stale)
    for (int e = tid; e < D * D; e += NTHR) {
      const uint16_t x = (uint16_t)f2bf(S32[e]);
      Ssm[(e >> 7) * SD + (e & 127)] = x;
      S16[e] = x;
    }
  if (tid < p) {
    lsm[tid] = lj;
    bsm[tid] = bj;
  }
  if (warp == 0) {
    const uint4 kn = norm_row(raw);
    if (lane < 16) {
      *reinterpret_cast<uint4*>(Ksm + (size_t)p * SD + 8 * lane) = kn;
      *reinterpret_cast<uint4*>(knrow + 8 * lane) = kn;
      if (owner) *reinterpret_cast<uint4*>(Kpool + (size_t)p * HQ * D + 8 * lane) = kn;
      if (owner) {
        *reinterpret_cast<uint4*>(hk + (hrow * HQ + kh) * D + 8 * lane) = raw;
      }
    }
    if (tid < SC / 2) reinterpret_cast<uint32_t*>(xrow)[tid] = 0u;
  }
  if (warp == 1) {
    const uint4 qn = norm_row(raw);
    if (lane < 16) {
      *reinterpret_cast<uint4*>(Qsm + 8 * lane) = qn;
      if (owner) *reinterpret_cast<uint4*>(hq + (hrow * HQ + kh) * D + 8 * lane) = raw;
    }
  }
  if (warp == 2 && lane < 16) {
    *reinterpret_cast<uint4*>(vrow + 8 * lane) = raw;
    *reinterpret_cast<uint4*>(hv + (hrow * HV + h) * D + 8 * lane) = raw;
  }
  if (tid == 96) {
    const float alpha = expf(graw);
    const float lp = lg2a(alpha + 1e-10f);
    lsm[p] = lp;
    bsm[p] = braw;
    Lpool[p] = lp;
    Bpool[p] = braw;
    hg[hrow * HV + h] = graw;
    hb[hrow * HV + h] = braw;
  }
  cp_async_wait_all();
  __syncthreads();

  // ---- gate scan (warp 0), then GEMMs 1-4
  if (warp == 0) {
    float g0 = lane <= p ? lsm[lane] : 0.0f, g1 = lane + 32 <= p ? lsm[lane + 32] : 0.0f;
    gate_scan(g0, g1, lane);
    const float c0p = __shfl_sync(0xffffffffu, g0, p & 31),
                c1p = __shfl_sync(0xffffffffu, g1, p & 31);
    const float csp = p < 32 ? c0p : c1p;
    cssm[lane] = g0;
    cssm[lane + 32] = g1;
    Tsm[lane] = lane <= p ? ex2a(csp - g0) : 0.0f;
    Tsm[lane + 32] = lane + 32 <= p ? ex2a(csp - g1) : 0.0f;
    if (lane == 0) misc[0] = ex2a(csp);
  }
  {
    BReg B;
    load_b(B, (lane & 7) == 0 ? Ksm + (size_t)p * SD : Qsm, lane);
    {
      AReg A;
      load_a(A, Ssm + (size_t)(16 * warp) * SD, SD, lane);
      {
        // the warp's state rows are in registers: their bytes take 16 rows of the VKS (warps 0-3)
        // or NV (4-7) tile
        const uint16_t* src = warp < 4 ? Vpool : Npool;
        const int j0 = 16 * (warp & 3);
        uint16_t* dst = Ssm + (size_t)(16 * warp) * SD;
        const int nr = p - j0 < 16 ? p - j0 : 16;
        for (int u = lane; u < nr * 16; u += 32)
          cp_async16(dst + (u >> 4) * SD + (u & 15) * 8,
                     src + (size_t)(j0 + (u >> 4)) * D + (u & 15) * 8);
        cp_async_commit();
        if ((p >> 4) == (warp & 3))
          for (int e = lane; e < (15 - (p & 15)) * (SD / 2); e += 32)
            reinterpret_cast<uint32_t*>(dst + (size_t)((p & 15) + 1) * SD)[e] = 0u;
      }
      const Acc4 d = mma128(A, B);
      if (t4 == 0) {
        const int r = 16 * warp + g;
        ABsm[r] = d.c[0];
        ABsm[D + r] = d.c[1];
        ABsm[r + 8] = d.c[2];
        ABsm[D + r + 8] = d.c[3];
      }
    }
    if (warp < ((p + 16) >> 4)) {
      const int j = 16 * warp + g;
      AReg A2;
      load_a(A2, Ksm + (size_t)(16 * warp) * SD, SD, lane);
      const Acc4 d = mma128(A2, B);
      if (t4 == 0) {
        KQsm[2 * j] = d.c[0];
        KQsm[2 * j + 1] = d.c[1];
        KQsm[2 * (j + 8)] = d.c[2];
        KQsm[2 * (j + 8) + 1] = d.c[3];
      }
    }
  }
  __syncthreads();
  // the K tile is consumed: X rows of the earlier blocks and the M rows of p's block take its place
  for (int u = tid; u < xhi * 8; u += NTHR)
    cp_async16(Xsm + (u >> 3) * XS + (u & 7) * 8, Xpool + (size_t)u * 8);
  for (int u = tid + mlo * 8; u < p * 8; u += NTHR)
    cp_async16(Msm + (size_t)((u >> 3) - mlo) * SC + (u & 7) * 8, Mpool + (size_t)u * 8);
  cp_async_commit();

  // ---- epilogues: M row p, W row p, VKS row p, scaled QS
  const float prodp = misc[0], betap = bsm[p];
  if (tid < CH) {
    const float T = Tsm[tid];
    const uint16_t m = tid < p ? (uint16_t)f2bf((KQsm[2 * tid] * T) * betap) : (uint16_t)0;
    const uint16_t w = tid <= p ? (uint16_t)f2bf((KQsm[2 * tid + 1] * T) * scale) : (uint16_t)0;
    MEL(p, tid) = m;
    Mpool[(size_t)p * CH + tid] = m;
    wrow[tid] = w;
  }
  if (tid >= CH && tid < SC) wrow[tid] = 0;
  if (tid >= D) {
    const int vv = tid - D;
    const float ks = ABsm[vv] * prodp;
    const uint16_t x = (uint16_t)f2bf(bf2f(vrow[vv]) - bf2f(f2bf(ks)));
    Vsm[(size_t)p * SD + vv] = x;
    Vpool[(size_t)p * D + vv] = x;
    ABsm[D + vv] = (ABsm[D + vv] * prodp) * scale;
  }
  cp_async_wait_all();
  __syncthreads();

  // ---- row p of the blocked unit-lower-triangular inverse (warp 0): 8x8 Gauss-Jordan, then the 16
  // / 32 / 64 stages
  if (warp == 0) {
    {
      const int b8 = p & ~7, r = lane & 7, ri = b8 + r;
      float row[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) row[i] = (ri <= p && i < r) ? bf2f(MEL(ri, b8 + i)) : 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i) row[i] = (r == i) ? 1.0f : row[i];
#pragma unroll
      for (int src = 0; src < 7; ++src) {
        const float rs = -row[src];
#pragma unroll
        for (int i = 0; i < src; ++i) {
          const float sv = __shfl_sync(0xffffffffu, row[i], src, 8);
          row[i] = r > src ? __fmaf_rn(rs, sv, row[i]) : row[i];
        }
        row[src] = r > src ? rs : row[src];
      }
      if (lane == (p & 7)) {
#pragma unroll
        for (int i = 0; i < 8; ++i)
          if (i <= r) xrow[b8 + i] = (uint16_t)f2bf(row[i]);
      }
    }
    if ((p & 15) >= 8) {
      const int b16 = p & ~15, r0 = b16 + 8 + 2 * t4;
      uint32_t a0 = g == 0 ? ld32(xrow + b16 + 8 + 2 * t4) : 0u;
      uint32_t b0 = (r0 <= p ? (uint32_t)MEL(r0, b16 + g) : 0u) |
                    ((r0 + 1 <= p ? (uint32_t)MEL(r0 + 1, b16 + g) : 0u) << 16);
      Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
      mma_k8(d, a0, 0u, b0);
      a0 = g == 0 ? f2bf_x2(0.0f - d.c[0], 0.0f - d.c[1]) : 0u;
      b0 = (uint32_t)XEL(b16 + 2 * t4, b16 + g) | ((uint32_t)XEL(b16 + 2 * t4 + 1, b16 + g) << 16);
      Acc4 e = {{0.f, 0.f, 0.f, 0.f}};
      mma_k8(e, a0, 0u, b0);
      if (g == 0) *reinterpret_cast<uint32_t*>(xrow + b16 + 2 * t4) = f2bf_x2(e.c[0], e.c[1]);
    }
    if ((p & 31) >= 16) {
      const int b32 = p & ~31, db = b32 + 16;
      uint32_t a0 = g == 0 ? ld32(xrow + db + 2 * t4) : 0u,
               a2 = g == 0 ? ld32(xrow + db + 8 + 2 * t4) : 0u;
      uint32_t dc[2];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int col = b32 + 8 * nt + g;
        uint32_t bb[2];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int r0 = db + 8 * hh + 2 * t4;
          bb[hh] = (r0 <= p ? (uint32_t)MEL(r0, col) : 0u) |
                   ((r0 + 1 <= p ? (uint32_t)MEL(r0 + 1, col) : 0u) << 16);
        }
        Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
        mma_bf16(d, a0, 0u, a2, 0u, bb[0], bb[1]);
        dc[nt] = f2bf_x2(-d.c[0], -d.c[1]);
      }
      a0 = g == 0 ? dc[0] : 0u;
      a2 = g == 0 ? dc[1] : 0u;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const int col = b32 + 8 * nt + g;
        uint32_t bb[2];
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const int r0 = b32 + 8 * hh + 2 * t4;
          bb[hh] = (uint32_t)XEL(r0, col) | ((uint32_t)XEL(r0 + 1, col) << 16);
        }
        Acc4 e = {{0.f, 0.f, 0.f, 0.f}};
        mma_bf16(e, a0, 0u, a2, 0u, bb[0], bb[1]);
        if (g == 0)
          *reinterpret_cast<uint32_t*>(xrow + b32 + 8 * nt + 2 * t4) = f2bf_x2(e.c[0], e.c[1]);
      }
    }
    if (p >= 32) {
      uint32_t aa[4];
#pragma unroll
      for (int x = 0; x < 4; ++x) aa[x] = g == 0 ? ld32(xrow + 32 + 8 * x + 2 * t4) : 0u;
      uint32_t dc[4];
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = 8 * nt + g;
        Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          uint32_t bb[2];
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int r0 = 32 + 16 * s + 8 * hh + 2 * t4;
            bb[hh] = (r0 <= p ? (uint32_t)MEL(r0, col) : 0u) |
                     ((r0 + 1 <= p ? (uint32_t)MEL(r0 + 1, col) : 0u) << 16);
          }
          mma_bf16(d, aa[2 * s], 0u, aa[2 * s + 1], 0u, bb[0], bb[1]);
        }
        dc[nt] = f2bf_x2(-d.c[0], -d.c[1]);
      }
#pragma unroll
      for (int x = 0; x < 4; ++x) aa[x] = g == 0 ? dc[x] : 0u;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = 8 * nt + g;
        Acc4 e = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          uint32_t bb[2];
#pragma unroll
          for (int hh = 0; hh < 2; ++hh) {
            const int r0 = 16 * s + 8 * hh + 2 * t4;
            bb[hh] = (uint32_t)XEL(r0, col) | ((uint32_t)XEL(r0 + 1, col) << 16);
          }
          mma_bf16(e, aa[2 * s], 0u, aa[2 * s + 1], 0u, bb[0], bb[1]);
        }
        if (g == 0) *reinterpret_cast<uint32_t*>(xrow + 8 * nt + 2 * t4) = f2bf_x2(e.c[0], e.c[1]);
      }
    }
    // A_inv row p = bf16(f32(X[p][j]) * beta[j]); X row p to the pool
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int j = lane + 32 * c;
      const uint16_t x = xrow[j];
      ainv[j] = j <= p ? (uint16_t)f2bf(bf2f(x) * bsm[j]) : (uint16_t)0;
      Xpool[(size_t)p * CH + j] = x;
    }
    if (lane < SC - CH) ainv[CH + lane] = 0;
  }
  __syncthreads();

  // ---- GEMM 5: NV row p = sum_i VKS[i] * A_inv[p][i]
  const int nks = (p >> 4) + 1;
  {
    TReg A;
    load_at(A, Vsm, 16 * warp, lane, nks);
    Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int js = 0; js < 4; ++js)
      if (js < nks) {
        const uint32_t b0 = g == 0 ? ld32(ainv + 16 * js + 2 * t4) : 0u,
                       b1 = g == 0 ? ld32(ainv + 16 * js + 8 + 2 * t4) : 0u;
        mma_bf16(d, A.a[4 * js], A.a[4 * js + 1], A.a[4 * js + 2], A.a[4 * js + 3], b0, b1);
      }
    if (t4 == 0) {
      nvsm[16 * warp + g] = d.c[0];
      nvsm[16 * warp + g + 8] = d.c[2];
    }
  }
  __syncthreads();
  if (tid < D) {
    const uint16_t x = (uint16_t)f2bf(nvsm[tid]);
    Nsm[(size_t)p * SD + tid] = x;
    Npool[(size_t)p * D + tid] = x;
  }
  __syncthreads();

  // ---- GEMM 6: O row p = scaled QS + sum_j NV_bf16[j] * W[p][j]
  {
    TReg A;
    load_at(A, Nsm, 16 * warp, lane, nks);
    Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
    if (t4 == 0) {
      d.c[0] = ABsm[D + 16 * warp + g];
      d.c[2] = ABsm[D + 16 * warp + g + 8];
    }
#pragma unroll
    for (int js = 0; js < 4; ++js)
      if (js < nks) {
        const uint32_t b0 = g == 0 ? ld32(wrow + 16 * js + 2 * t4) : 0u,
                       b1 = g == 0 ? ld32(wrow + 16 * js + 8 + 2 * t4) : 0u;
        mma_bf16(d, A.a[4 * js], A.a[4 * js + 1], A.a[4 * js + 2], A.a[4 * js + 3], b0, b1);
      }
    if (t4 == 0) {
      uint16_t* o = out + (bt * HV + h) * D + 16 * warp + g;
      o[0] = (uint16_t)f2bf(d.c[0]);
      o[8] = (uint16_t)f2bf(d.c[2]);
    }
  }
  if (p + 1 < CH) {
    if (tid == 0) *cnt_cell = p + 1;
    return;
  }

  // ---- chunk completion: NV refolded from the VKS rows and A_inv, then the state update S' =
  // S_decayed + decayV^T K
  __syncthreads();
  TReg AV;
  load_at(AV, Vsm, 16 * warp, lane, 4);
  if (tid < CH) Tsm[tid] = ex2a(cssm[CH - 1] + (-cssm[tid]));
  if (warp == 1) {
    float g0 = lg2a(1.0f + 1e-10f), g1 = g0;
    gate_scan(g0, g1, lane);
    if (lane == 31) {
      misc[1] = ex2a(g1);
      misc[2] = ex2a(cssm[CH - 1]);
    }
  }
  __syncthreads();
  float* const blk = (warp < 4 ? reinterpret_cast<float*>(Ssm) : reinterpret_cast<float*>(Ksm)) +
                     (size_t)(warp & 3) * 16 * FS;
  const float* Srow = S32 + (size_t)(16 * warp) * D;
  if (warp < 4) {
    for (int u = lane; u < 256; u += 32)
      cp_async16(blk + (u >> 4) * FS + (u & 15) * 4, Srow + (size_t)(u >> 4) * D + (u & 15) * 4);
    cp_async_commit();
  }
  uint16_t* const Asm = Nsm;
  for (int e = tid; e < CH * CH; e += NTHR) {
    const int j = e >> 6, i = e & 63;
    const uint16_t x = j == CH - 1 ? xrow[i] : XEL(j, i);
    Asm[(size_t)j * SC + i] = i <= j ? (uint16_t)f2bf(bf2f(x) * bsm[i]) : (uint16_t)0;
  }
  __syncthreads();
  if (warp >= 4) {
    for (int u = lane; u < 256; u += 32)
      cp_async16(blk + (u >> 4) * FS + (u & 15) * 4, Srow + (size_t)(u >> 4) * D + (u & 15) * 4);
    cp_async_commit();
  }
  TReg ADV;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const uint16_t* pr =
        Asm + (size_t)(8 * nt + (lane & 7)) * SC + 8 * ((lane >> 3) & 1) + 16 * (lane >> 4);
    uint32_t bb[8];
    ldsm_x4(bb[0], bb[1], bb[2], bb[3], pr);
    ldsm_x4(bb[4], bb[5], bb[6], bb[7], pr + 32);
    Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int js = 0; js < 4; ++js)
      mma_bf16(d, AV.a[4 * js], AV.a[4 * js + 1], AV.a[4 * js + 2], AV.a[4 * js + 3], bb[2 * js],
               bb[2 * js + 1]);
    const float e0 = Tsm[8 * nt + 2 * t4], e1 = Tsm[8 * nt + 2 * t4 + 1];
    ADV.a[4 * (nt >> 1) + 2 * (nt & 1)] = f2bf_x2(d.c[0] * e0, d.c[1] * e1);
    ADV.a[4 * (nt >> 1) + 2 * (nt & 1) + 1] = f2bf_x2(d.c[2] * e0, d.c[3] * e1);
  }
  __syncthreads();
  for (int u = tid; u < (CH - 1) * 16; u += NTHR)
    cp_async16(Nsm + (u >> 4) * SD + (u & 15) * 8,
               Kpool + (size_t)(u >> 4) * HQ * D + (u & 15) * 8);
  cp_async_commit();
  if (tid < 16)
    *reinterpret_cast<uint4*>(Nsm + (size_t)(CH - 1) * SD + 8 * tid) =
        *reinterpret_cast<const uint4*>(knrow + 8 * tid);
  cp_async_wait_all();
  __syncthreads();
  {
    const float cpt = misc[2], c2 = misc[1];
#pragma unroll 1
    for (int hf = 0; hf < 2; ++hf) {
#pragma unroll 1
      for (int kq = 0; kq < 8; ++kq) {
        const int kg = 8 * hf + kq, kc = 8 * kq + 2 * t4;
        float2* plo = reinterpret_cast<float2*>(blk + (size_t)g * FS + kc);
        float2* phi = reinterpret_cast<float2*>(blk + (size_t)(g + 8) * FS + kc);
        const float2 lo = *plo, hi = *phi;
        Acc4 d;
        d.c[0] = lo.x * cpt;
        d.c[1] = lo.y * cpt;
        d.c[2] = hi.x * cpt;
        d.c[3] = hi.y * cpt;
        const uint16_t* pb = Nsm + 8 * kg + (size_t)((lane & 7) + 8 * ((lane >> 3) & 1)) * SD;
        uint32_t bb[8];
#pragma unroll
        for (int js = 0; js < 4; ++js)
          ldsm_x2t(bb[2 * js], bb[2 * js + 1], pb + (size_t)(16 * js) * SD);
#pragma unroll
        for (int js = 0; js < 4; ++js)
          mma_bf16(d, ADV.a[4 * js], ADV.a[4 * js + 1], ADV.a[4 * js + 2], ADV.a[4 * js + 3],
                   bb[2 * js], bb[2 * js + 1]);
        *plo = make_float2((d.c[0] * c2) + 0.0f, (d.c[1] * c2) + 0.0f);
        *phi = make_float2((d.c[2] * c2) + 0.0f, (d.c[3] * c2) + 0.0f);
      }
      __syncwarp();
      for (int u = lane; u < 256; u += 32)
        *reinterpret_cast<uint4*>(S32 + (size_t)(16 * warp + (u >> 4)) * D + 64 * hf +
                                  (u & 15) * 4) =
            *reinterpret_cast<const uint4*>(blk + (u >> 4) * FS + (u & 15) * 4);
      for (int u = lane; u < 128; u += 32) {
        const float* f = blk + (u >> 3) * FS + (u & 7) * 8;
        const float4 x = *reinterpret_cast<const float4*>(f),
                     y = *reinterpret_cast<const float4*>(f + 4);
        *reinterpret_cast<uint4*>(S16 + (size_t)(16 * warp + (u >> 3)) * D + 64 * hf +
                                  (u & 7) * 8) =
            make_uint4(f2bf_x2(x.x, x.y), f2bf_x2(x.z, x.w), f2bf_x2(y.x, y.y), f2bf_x2(y.z, y.w));
      }
      if (hf == 0) {
        __syncwarp();
        for (int u = lane; u < 256; u += 32)
          cp_async16(blk + (u >> 4) * FS + (u & 15) * 4,
                     Srow + (size_t)(u >> 4) * D + 64 + (u & 15) * 4);
        cp_async_commit();
        cp_async_wait_all();
        __syncwarp();
      }
    }
  }
  if (tid == 0) *cnt_cell = 0;
}
#undef MEL
#undef XEL

// Launches one step for B requests: grid B * HV, one row per (request, v-head).
inline cudaError_t DecodeStep(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                              const float* g, const float* beta, const int* slots, uint16_t* out,
                              float* state, uint16_t* state_bf16, int* cnt, float* ring_l,
                              float* ring_b, uint16_t* kn, uint16_t* m_rows, uint16_t* x_rows,
                              uint16_t* vks_rows, uint16_t* nv_rows, uint16_t* hist_q,
                              uint16_t* hist_k, uint16_t* hist_v, float* hist_g, float* hist_beta,
                              float scale, int batch, int num_q_heads, int num_v_heads,
                              int num_slots, cudaStream_t stream) {
  cudaError_t status = cudaFuncSetAttribute(
      DecodeStepKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes));
  if (status != cudaSuccess) return status;
  DecodeStepKernel<<<batch * num_v_heads, NTHR, kSmemBytes, stream>>>(
      q, k, v, g, beta, slots, out, state, state_bf16, cnt, ring_l, ring_b, kn, m_rows, x_rows,
      vks_rows, nv_rows, hist_q, hist_k, hist_v, hist_g, hist_beta, scale, num_q_heads, num_v_heads,
      num_slots);
  return cudaGetLastError();
}

}  // namespace bi_decode
}  // namespace gdn
}  // namespace flashinfer

#endif  // FLASHINFER_GDN_BATCH_INVARIANT_DECODE_CUH_
