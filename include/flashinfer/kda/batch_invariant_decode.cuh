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
#ifndef FLASHINFER_KDA_BATCH_INVARIANT_DECODE_CUH_
#define FLASHINFER_KDA_BATCH_INVARIANT_DECODE_CUH_

// Decode step of the batch-invariant KDA class kda.bf16.c16: row p of the BT = 16 chunked prefill
// from cached rows of the open chunk. Build without fast math and with -fmad=false: the fused and
// unfused operations below are the prefill's own.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace flashinfer {
namespace kda {
namespace bi_decode {

constexpr int D = 128;
constexpr int CH = 16;
constexpr int NTHR = 256;
constexpr int MINB = 4;
constexpr int SD = D + 8;
constexpr int FS = 68;
constexpr int STAMP = 8;
constexpr size_t kSmemBytes = (size_t)D * SD * 2 + 3 * (size_t)CH * SD * 2 + 2 * SD * 2 +
                              CH * CH * 2 + 2 * CH * 2 + 4 * D * 4 + 64;
// the span kernel also stages the q / k / v / g rows of the current and the next token
constexpr size_t kSpanSmemBytes = kSmemBytes + 2 * 4 * D * 2;

struct Acc4 {
  float c[4];
};

// one mma.sync m16n8k16 bf16 step, fp32 accumulation; lane 4 * g + t holds A rows g, g + 8 and B
// column g
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
__device__ __forceinline__ uint32_t f2h_x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ float h2f(uint32_t x) {
  float y;
  const uint16_t s = (uint16_t)x;
  asm("cvt.f32.f16 %0, %1;\n" : "=f"(y) : "h"(s));
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
__device__ __forceinline__ float rcpa(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float tanha(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t sub_bf(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("sub.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t mul_bf(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
  return d;
}
__device__ __forceinline__ uint32_t movm(uint32_t a) {
  uint32_t d;
  asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}
__device__ __forceinline__ void mma_f16(Acc4& d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d.c[0]), "+f"(d.c[1]), "+f"(d.c[2]), "+f"(d.c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
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

struct AReg {
  uint32_t a[32];
};
struct Frag {
  uint32_t r[4];
};
struct Tile {
  Acc4 n0, n1;
};

// lhs @ rhs of two 16 x 16 fp16 tiles held as A-layout fragments (the engine's pairwise product)
__device__ __forceinline__ Tile pair_mul(const Frag& lhs, const Frag& rhs) {
  const uint32_t b0 = movm(rhs.r[0]), b1 = movm(rhs.r[1]), b2 = movm(rhs.r[2]), b3 = movm(rhs.r[3]);
  Tile o = {{{0.f, 0.f, 0.f, 0.f}}, {{0.f, 0.f, 0.f, 0.f}}};
  mma_f16(o.n0, lhs.r[0], lhs.r[1], lhs.r[2], lhs.r[3], b0, b1);
  mma_f16(o.n1, lhs.r[0], lhs.r[1], lhs.r[2], lhs.r[3], b2, b3);
  return o;
}
__device__ __forceinline__ Frag pack_h(const Tile& t) {
  Frag f;
  f.r[0] = f2h_x2(t.n0.c[0], t.n0.c[1]);
  f.r[1] = f2h_x2(t.n0.c[2], t.n0.c[3]);
  f.r[2] = f2h_x2(t.n1.c[0], t.n1.c[1]);
  f.r[3] = f2h_x2(t.n1.c[2], t.n1.c[3]);
  return f;
}
__device__ __forceinline__ float rnd_h(float x) { return h2f(f2h_x2(x, 0.0f)); }
// inv += inv @ pw with inv rounded through fp16 first
__device__ __forceinline__ void inv_update(Tile& inv, const Frag& pw) {
  const Tile p = pair_mul(pack_h(inv), pw);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    inv.n0.c[i] = rnd_h(inv.n0.c[i]) + p.n0.c[i];
    inv.n1.c[i] = rnd_h(inv.n1.c[i]) + p.n1.c[i];
  }
}

// SPAN = false: one token per block, row b of the inputs. SPAN = true: block (b, h) appends rows
// [start, start + len) of the packed inputs to its slot in one launch, committing every chunk it
// completes.
template <bool SPAN>
__global__ void __launch_bounds__(NTHR, MINB)
    DecodeKernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                 const uint16_t* __restrict__ v, const uint16_t* __restrict__ gt,
                 const uint16_t* __restrict__ bt, const int* __restrict__ idx,
                 const int* __restrict__ starts, const int* __restrict__ lens, int stride,
                 uint16_t* __restrict__ out, float* state, uint16_t* state16, int* cnt, float* pref,
                 uint16_t* Kc, float* Kf, uint16_t* Rc, uint16_t* Uc, uint16_t* Lc, float* stamp,
                 const float* __restrict__ a_log, const float* __restrict__ dt_bias, float scale,
                 float half_scale, int H) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int h = blockIdx.x % H, b = blockIdx.x / H;
  const int slot = idx[b];
  if (slot < 0) return;
  const int n = SPAN ? (starts ? lens[b] : stride) : 1;
  if (n <= 0) return;
  const size_t row0 = SPAN ? (starts ? (size_t)starts[b] : (size_t)b * stride) : (size_t)b;
  const size_t sh = (size_t)slot * H + h;
  size_t bh = row0 * H + h;
  int* cnt_cell = cnt + sh;

  size_t off = 0;
  uint16_t* Ssm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)D * SD * 2;
  uint16_t* Ksm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)CH * SD * 2;
  uint16_t* Rsm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)CH * SD * 2;
  uint16_t* Usm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)CH * SD * 2;
  uint16_t* kdrow = reinterpret_cast<uint16_t*>(smem + off);
  off += SD * 2;
  uint16_t* qdrow = reinterpret_cast<uint16_t*>(smem + off);
  off += SD * 2;
  uint16_t* Lsm = reinterpret_cast<uint16_t*>(smem + off);
  off += CH * CH * 2;
  uint16_t* ainv = reinterpret_cast<uint16_t*>(smem + off);
  off += CH * 2;
  uint16_t* qkrow = reinterpret_cast<uint16_t*>(smem + off);
  off += CH * 2;
  float* egsm = reinterpret_cast<float*>(smem + off);
  off += D * 4;
  float* SKsm = reinterpret_cast<float*>(smem + off);
  off += D * 4;
  float* SQsm = reinterpret_cast<float*>(smem + off);
  off += D * 4;
  float* kfrow = reinterpret_cast<float*>(smem + off);
  off += D * 4;
  float* misc = reinterpret_cast<float*>(smem + off);
  off += 64;
  uint16_t* tok = reinterpret_cast<uint16_t*>(smem + off);

  uint16_t* S16 = state16 + sh * D * D;
  float* S32 = state + sh * D * D;
  uint16_t* Kpool = Kc + sh * CH * D;
  float* Fpool = Kf + sh * CH * D;
  uint16_t* Rpool = Rc + sh * CH * D;
  uint16_t* Upool = Uc + sh * CH * D;
  uint16_t* Lpool = Lc + sh * CH * CH;
  float* Ppool = pref + sh * D;

  // phase-independent loads first: state tile, raw rows, gate constants. The last warp copies no
  // state: it copies the cached rows, which the first barrier of a step then need not wait for
  if (warp < NTHR / 32 - 1)
    for (int u = tid; u < D * 16; u += NTHR - 32)
      cp_async16(Ssm + (u >> 4) * SD + (u & 15) * 8, S16 + (size_t)u * 8);
  cp_async_commit();
  const int L = lane & 7, isq = (lane >> 3) & 1;
  uint4 r0 = make_uint4(0, 0, 0, 0), r1 = r0;
  uint32_t vp = 0u;
  uint16_t braw = 0, bnext = 0;
  float gx = 0.0f, araw = 0.0f, gbias = 0.0f;
  if (SPAN) {
    if (tid < 64) {
      const uint16_t* src = (tid < 32 ? (tid < 16 ? q : k) : (tid < 48 ? v : gt)) + bh * D;
      cp_async16(tok + (tid >> 4) * D + (tid & 15) * 8, src + (tid & 15) * 8);
    }
    cp_async_commit();
    if (tid == D) bnext = bt[bh];
    if (tid < D) {
      araw = a_log[h];
      gbias = dt_bias[(size_t)h * D + tid];
    }
  } else {
    if (warp == 0 && lane < 16) {
      const uint16_t* src = (isq ? q : k) + bh * D;
      r0 = *reinterpret_cast<const uint4*>(src + 8 * L);
      r1 = *reinterpret_cast<const uint4*>(src + 64 + 8 * L);
    }
    if (tid < D / 2) vp = *reinterpret_cast<const uint32_t*>(v + bh * D + 2 * tid);
    // the uses stay behind the loads: a use would hold back the loads issued after it
    if (tid < D) {
      const float al = ex2a(a_log[h] * 1.4426950408889634f);
      gx = (al * 0.5f) * (bf2f(gt[bh * D + tid]) + dt_bias[(size_t)h * D + tid]);
    }
    if (tid == D) braw = bt[bh];
  }
  float pprev = 0.0f;
  if (tid < D) pprev = Ppool[tid];
  const uint16_t lraw = Lpool[tid];

  // a state whose first values differ from the stamp was replaced by the caller: new chunk
  const int c0 = *cnt_cell;
  float* Stamp = stamp + sh * STAMP;
  const uint4 cur0 = *reinterpret_cast<const uint4*>(S32),
              cur1 = *reinterpret_cast<const uint4*>(S32 + 4);
  const uint4 old0 = *reinterpret_cast<const uint4*>(Stamp),
              old1 = *reinterpret_cast<const uint4*>(Stamp + 4);
  const bool moved =
      ((cur0.x ^ old0.x) | (cur0.y ^ old0.y) | (cur0.z ^ old0.z) | (cur0.w ^ old0.w) |
       (cur1.x ^ old1.x) | (cur1.y ^ old1.y) | (cur1.z ^ old1.z) | (cur1.w ^ old1.w)) != 0u;
  int p = moved ? 0 : (c0 & (CH - 1));
  if (warp == NTHR / 32 - 1)
    for (int u = lane; u < p * 16; u += 32) {
      const int j = u >> 4, c8 = (u & 15) * 8;
      cp_async16(Ksm + j * SD + c8, Kpool + (size_t)j * D + c8);
      cp_async16(Rsm + j * SD + c8, Rpool + (size_t)j * D + c8);
      cp_async16(Usm + j * SD + c8, Upool + (size_t)j * D + c8);
    }
  cp_async_commit();
  Lsm[tid] = (tid >> 4) < p ? lraw : (uint16_t)0;
  const float ghalf = ex2a(araw * 1.4426950408889634f) * 0.5f;
#pragma unroll 1
  for (int j = 0; j < n; ++j) {
    if (SPAN) {
      // token j is staged (with the state tile and the cached rows when j = 0); stage token j + 1
      cp_async_wait_all();
      __syncthreads();
      const uint16_t* cur = tok + (j & 1) * 4 * D;
      if (warp == 0 && lane < 16) {
        r0 = *reinterpret_cast<const uint4*>(cur + (isq ? 0 : D) + 8 * L);
        r1 = *reinterpret_cast<const uint4*>(cur + (isq ? 0 : D) + 64 + 8 * L);
      }
      if (tid < D / 2) vp = *reinterpret_cast<const uint32_t*>(cur + 2 * D + 2 * tid);
      if (tid < D) gx = ghalf * (bf2f(cur[3 * D + tid]) + gbias);
      bh = (row0 + j) * H + h;
      braw = bnext;
      if (j + 1 < n) {
        if (tid < 64) {
          const uint16_t* src =
              (tid < 32 ? (tid < 16 ? q : k) : (tid < 48 ? v : gt)) + (bh + H) * D;
          cp_async16(tok + ((j + 1) & 1) * 4 * D + (tid >> 4) * D + (tid & 15) * 8,
                     src + (tid & 15) * 8);
        }
        if (tid == D) bnext = bt[bh + H];
      }
      cp_async_commit();
    }
    if (tid < D) {
      const float th = tanha(gx);
      // the prefill computes this increment as one fused multiply-add
      const float inc = __fmaf_rn(th, half_scale, half_scale);
      const float pv = (p ? pprev : 0.0f) + inc;
      egsm[tid] = ex2a(pv);
      Ppool[tid] = pv;
      pprev = pv;
    }
    if (tid == D) misc[0] = tanha(bf2f(braw) * 0.5f) * 0.5f + 0.5f;

    // q / k l2 norm (warp 0: lanes 0-7 k, lanes 8-15 q), then the decay operands of row p
    float x[16];
    float inv = 0.0f;
    if (warp == 0) {
      float ss = 0.0f;
      const uint32_t w[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        x[i] = bf2f(w[i >> 1] >> (16 * (i & 1)));
        ss = ss + x[i] * x[i];
      }
      ss = ss + __shfl_xor_sync(0xffffffffu, ss, 4);
      ss = ss + __shfl_xor_sync(0xffffffffu, ss, 2);
      ss = ss + __shfl_xor_sync(0xffffffffu, ss, 1);
      inv = rsqa(ss + 1e-6f);
    }
    if (!SPAN && warp < NTHR / 32 - 1) cp_async_wait_all();
    __syncthreads();
    for (int e = tid; e < (CH - 1 - p) * (SD / 2); e += NTHR) {
      reinterpret_cast<uint32_t*>(Rsm + (size_t)(p + 1) * SD)[e] = 0u;
      reinterpret_cast<uint32_t*>(Usm + (size_t)(p + 1) * SD)[e] = 0u;
    }
    if (moved && j == 0) {
      // two 8-value pieces per round: their loads are issued before the first one is used
      for (int e = tid; e < D * D / 8; e += 2 * NTHR) {
        float4 x[2], y[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          x[i] = *reinterpret_cast<const float4*>(S32 + 8 * (e + i * NTHR));
          y[i] = *reinterpret_cast<const float4*>(S32 + 8 * (e + i * NTHR) + 4);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const int e8 = 8 * (e + i * NTHR);
          const uint4 w = make_uint4(f2bf_x2(x[i].x, x[i].y), f2bf_x2(x[i].z, x[i].w),
                                     f2bf_x2(y[i].x, y[i].y), f2bf_x2(y[i].z, y[i].w));
          *reinterpret_cast<uint4*>(Ssm + (e8 >> 7) * SD + (e8 & 127)) = w;
          *reinterpret_cast<uint4*>(S16 + e8) = w;
        }
      }
      if (tid < STAMP) Stamp[tid] = S32[tid];
    }
    if (warp == 0 && lane < 16) {
#pragma unroll
      for (int hf = 0; hf < 2; ++hf) {
        const int d0 = 64 * hf + 8 * L;
        uint32_t pd[4], pi[4];
        float kf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float e0 = egsm[d0 + 2 * i], e1 = egsm[d0 + 2 * i + 1];
          const float y0 = x[8 * hf + 2 * i] * inv, y1 = x[8 * hf + 2 * i + 1] * inv;
          pd[i] = f2bf_x2(y0 * e0, y1 * e1);
          kf[2 * i] = y0 * rcpa(e0);
          kf[2 * i + 1] = y1 * rcpa(e1);
          pi[i] = f2bf_x2(kf[2 * i], kf[2 * i + 1]);
        }
        const uint4 dv = make_uint4(pd[0], pd[1], pd[2], pd[3]);
        if (isq) {
          *reinterpret_cast<uint4*>(qdrow + d0) = dv;
        } else {
          const uint4 iv = make_uint4(pi[0], pi[1], pi[2], pi[3]);
          *reinterpret_cast<uint4*>(kdrow + d0) = dv;
          *reinterpret_cast<uint4*>(Ksm + (size_t)p * SD + d0) = iv;
          *reinterpret_cast<uint4*>(Kpool + (size_t)p * D + d0) = iv;
          *reinterpret_cast<float4*>(kfrow + d0) = make_float4(kf[0], kf[1], kf[2], kf[3]);
          *reinterpret_cast<float4*>(kfrow + d0 + 4) = make_float4(kf[4], kf[5], kf[6], kf[7]);
          if (p + 1 < CH) {
            *reinterpret_cast<float4*>(Fpool + (size_t)p * D + d0) =
                make_float4(kf[0], kf[1], kf[2], kf[3]);
            *reinterpret_cast<float4*>(Fpool + (size_t)p * D + d0 + 4) =
                make_float4(kf[4], kf[5], kf[6], kf[7]);
          }
        }
      }
    }
    // the cached rows are first read behind this barrier
    if (!SPAN && warp == NTHR / 32 - 1) cp_async_wait_all();
    __syncthreads();

    // state_k / state_q = bf16(S) . [k_decay_p, q_decay_p]; key s ^ 8 at k-position s as in the
    // prefill
    {
      uint32_t B[16];
      {
        const uint16_t* row =
            ((lane & 7) == 0 ? kdrow : qdrow) + 8 * ((lane >> 3) & 1) + 16 * (lane >> 4);
#pragma unroll
        for (int x = 0; x < 4; ++x)
          ldsm_x4(B[4 * x], B[4 * x + 1], B[4 * x + 2], B[4 * x + 3], row + 32 * x);
      }
      AReg A;
      {
        const uint16_t* pa =
            Ssm + (size_t)(16 * warp + (lane & 7) + 8 * ((lane >> 3) & 1)) * SD + 8 * (lane >> 4);
#pragma unroll
        for (int ks = 0; ks < 8; ++ks)
          ldsm_x4(A.a[4 * ks], A.a[4 * ks + 1], A.a[4 * ks + 2], A.a[4 * ks + 3], pa + 16 * ks);
      }
      Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks)
        mma_bf16(d, A.a[4 * ks + 2], A.a[4 * ks + 3], A.a[4 * ks], A.a[4 * ks + 1], B[2 * ks + 1],
                 B[2 * ks]);
      if (t4 == 0) {
        const int r = 16 * warp + g;
        SKsm[r] = d.c[0];
        SQsm[r] = d.c[1];
        SKsm[r + 8] = d.c[2];
        SQsm[r + 8] = d.c[3];
      }
    }
    // KK and QK row p with the prefill's operand sides (warp 0): L row p, causal QK row p
    if (warp == 0) {
      const uint16_t* pa = ((lane & 15) == 0 ? kdrow : qdrow) + 8 * (lane >> 4);
      const uint16_t* pb =
          Ksm + (size_t)((lane & 7) + 8 * (lane >> 4)) * SD + 8 * ((lane >> 3) & 1);
      Acc4 n0 = {{0.f, 0.f, 0.f, 0.f}}, n1 = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        ldsm_x4(a0, a1, a2, a3, pa + 16 * ks);
        ldsm_x4(b0, b1, b2, b3, pb + 16 * ks);
        mma_bf16(n0, a0, a1, a2, a3, b0, b1);
        mma_bf16(n1, a0, a1, a2, a3, b2, b3);
      }
      const float beta = misc[0];
      const int j0 = 2 * t4, j1 = 8 + 2 * t4;
      if (g == 0) {
        const uint32_t l0 =
            f2bf_x2((j0 < p ? n0.c[0] : 0.0f) * beta, (j0 + 1 < p ? n0.c[1] : 0.0f) * beta);
        const uint32_t l1 =
            f2bf_x2((j1 < p ? n1.c[0] : 0.0f) * beta, (j1 + 1 < p ? n1.c[1] : 0.0f) * beta);
        *reinterpret_cast<uint32_t*>(Lsm + p * CH + j0) = l0;
        *reinterpret_cast<uint32_t*>(Lsm + p * CH + j1) = l1;
        *reinterpret_cast<uint32_t*>(Lpool + p * CH + j0) = l0;
        *reinterpret_cast<uint32_t*>(Lpool + p * CH + j1) = l1;
      }
      if (g == 1) {
        *reinterpret_cast<uint32_t*>(qkrow + j0) =
            f2bf_x2(j0 <= p ? n0.c[0] : 0.0f, j0 + 1 <= p ? n0.c[1] : 0.0f);
        *reinterpret_cast<uint32_t*>(qkrow + j1) =
            f2bf_x2(j1 <= p ? n1.c[0] : 0.0f, j1 + 1 <= p ? n1.c[1] : 0.0f);
      }
    }
    __syncthreads();
    // the state tile is done: on the 16th row its room takes the first fp32 half-state of the
    // commit now
    float* const blk = reinterpret_cast<float*>(Ssm) + (size_t)warp * 16 * FS;
    const float* Srow = S32 + (size_t)(16 * warp) * D;
    if (p == CH - 1) {
      for (int u = lane; u < 256; u += 32)
        cp_async16(blk + (u >> 4) * FS + (u & 15) * 4, Srow + (size_t)(u >> 4) * D + (u & 15) * 4);
      cp_async_commit();
    }

    // rhs row p = bf16(beta) * (v - bf16(state_k)) in bf16 arithmetic; the 16 x 16 inverse (warp 2)
    if (tid < D / 2) {
      const uint32_t r =
          mul_bf(f2bf_x2(misc[0], misc[0]), sub_bf(vp, f2bf_x2(SKsm[2 * tid], SKsm[2 * tid + 1])));
      *reinterpret_cast<uint32_t*>(Rsm + (size_t)p * SD + 2 * tid) = r;
      *reinterpret_cast<uint32_t*>(Rpool + (size_t)p * D + 2 * tid) = r;
    }
    if (warp == 2) {
      float l[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(Lsm + (g + 8 * (i & 1)) * CH +
                                                              8 * (i >> 1) + 2 * t4);
        l[2 * i] = bf2f(w);
        l[2 * i + 1] = bf2f(w >> 16);
      }
      Frag dfr;
      dfr.r[0] = f2h_x2(l[0], l[1]);
      dfr.r[1] = 0u;
      dfr.r[2] = 0u;
      dfr.r[3] = f2h_x2(l[6], l[7]);
      Tile inv;
      inv.n0.c[0] = (g == 2 * t4 ? 1.0f : 0.0f) - l[0];
      inv.n0.c[1] = (g == 2 * t4 + 1 ? 1.0f : 0.0f) - l[1];
      inv.n0.c[2] = 0.0f;
      inv.n0.c[3] = 0.0f;
      inv.n1.c[0] = 0.0f - l[4];
      inv.n1.c[1] = 0.0f - l[5];
      inv.n1.c[2] = (g == 2 * t4 ? 1.0f : 0.0f) - l[6];
      inv.n1.c[3] = (g == 2 * t4 + 1 ? 1.0f : 0.0f) - l[7];
      const Frag d2 = pack_h(pair_mul(dfr, dfr));
      inv_update(inv, d2);
      const Frag d4 = pack_h(pair_mul(d2, d2));
      inv_update(inv, d4);
      Frag binv;
      binv.r[0] = f2h_x2(inv.n0.c[0], inv.n0.c[1]);
      binv.r[1] = 0u;
      binv.r[2] = 0u;
      binv.r[3] = f2h_x2(inv.n1.c[2], inv.n1.c[3]);
      Frag a21;
      a21.r[0] = 0u;
      a21.r[1] = f2h_x2(l[2], l[3]);
      a21.r[2] = 0u;
      a21.r[3] = 0u;
      const Tile t1 = pair_mul(binv, a21);
      Frag t1n;
      t1n.r[0] = 0u;
      t1n.r[1] = f2h_x2(0.0f - t1.n0.c[2], 0.0f - t1.n0.c[3]);
      t1n.r[2] = 0u;
      t1n.r[3] = 0u;
      const Tile x21 = pair_mul(t1n, binv);
      inv.n0.c[2] = x21.n0.c[2];
      inv.n0.c[3] = x21.n0.c[3];
      if (g == (p & 7)) {
        const int o = p < 8 ? 0 : 2;
        *reinterpret_cast<uint32_t*>(ainv + 2 * t4) = f2bf_x2(inv.n0.c[o], inv.n0.c[o + 1]);
        *reinterpret_cast<uint32_t*>(ainv + 8 + 2 * t4) = f2bf_x2(inv.n1.c[o], inv.n1.c[o + 1]);
      }
    }
    __syncthreads();

    // update column p = rhs . A_inv[p]; o = bf16((state_q + bf16(update) . QK[p]) * scale); token s
    // ^ 8 at k s
    {
      const size_t aoff =
          (size_t)((lane & 7) + 8 * (lane >> 4)) * SD + 16 * warp + 8 * ((lane >> 3) & 1);
      uint32_t a0, a1, a2, a3;
      ldsm_x4t(a0, a1, a2, a3, Rsm + aoff);
      Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
      mma_bf16(d, a2, a3, a0, a1, g == 0 ? ld32(ainv + 8 + 2 * t4) : 0u,
               g == 0 ? ld32(ainv + 2 * t4) : 0u);
      if (t4 == 0) {
        const int r = 16 * warp + g;
        const uint16_t u0 = (uint16_t)(f2bf_x2(d.c[0], 0.0f) & 0xffffu),
                       u1 = (uint16_t)(f2bf_x2(d.c[2], 0.0f) & 0xffffu);
        Usm[(size_t)p * SD + r] = u0;
        Usm[(size_t)p * SD + r + 8] = u1;
        Upool[(size_t)p * D + r] = u0;
        Upool[(size_t)p * D + r + 8] = u1;
      }
      __syncwarp();
      ldsm_x4t(a0, a1, a2, a3, Usm + aoff);
      Acc4 e = {{0.f, 0.f, 0.f, 0.f}};
      if (t4 == 0) {
        e.c[0] = SQsm[16 * warp + g];
        e.c[2] = SQsm[16 * warp + g + 8];
      }
      mma_bf16(e, a2, a3, a0, a1, g == 0 ? ld32(qkrow + 8 + 2 * t4) : 0u,
               g == 0 ? ld32(qkrow + 2 * t4) : 0u);
      if (t4 == 0) {
        uint16_t* o = out + bh * D + 16 * warp + g;
        o[0] = (uint16_t)(f2bf_x2(e.c[0] * scale, 0.0f) & 0xffffu);
        o[8] = (uint16_t)(f2bf_x2(e.c[2] * scale, 0.0f) & 0xffffu);
      }
      if (p + 1 < CH) {
        p += 1;
        continue;
      }

      // chunk completion: S' = S * exp2(g_prefix[15]) + bf16(update) @ k_restore, through
      // warp-private blocks
      __syncthreads();
      // every load is issued before the first one is used
      float2 kf4[CH * (D / 2) / NTHR];
#pragma unroll
      for (int i = 0; i < CH * (D / 2) / NTHR; ++i) {
        const int e2 = tid + i * NTHR, j = e2 >> 6, dd = 2 * (e2 & 63);
        kf4[i] = j == CH - 1 ? *reinterpret_cast<const float2*>(kfrow + dd)
                             : *reinterpret_cast<const float2*>(Fpool + (size_t)j * D + dd);
      }
#pragma unroll
      for (int i = 0; i < CH * (D / 2) / NTHR; ++i) {
        const int e2 = tid + i * NTHR, j = e2 >> 6, dd = 2 * (e2 & 63);
        *reinterpret_cast<uint32_t*>(Ksm + (size_t)j * SD + dd) =
            f2bf_x2(kf4[i].x * egsm[dd], kf4[i].y * egsm[dd + 1]);
      }
      cp_async_wait_all();
      __syncthreads();
#pragma unroll 1
      for (int hf = 0; hf < 2; ++hf) {
#pragma unroll 1
        for (int kq = 0; kq < 8; ++kq) {
          const int kc = 8 * kq + 2 * t4, kk = 64 * hf + kc;
          float2* plo = reinterpret_cast<float2*>(blk + (size_t)g * FS + kc);
          float2* phi = reinterpret_cast<float2*>(blk + (size_t)(g + 8) * FS + kc);
          const float2 lo = *plo, hi = *phi;
          const float e0 = egsm[kk], e1 = egsm[kk + 1];
          Acc4 dd;
          dd.c[0] = lo.x * e0;
          dd.c[1] = lo.y * e1;
          dd.c[2] = hi.x * e0;
          dd.c[3] = hi.y * e1;
          uint32_t b0, b1;
          ldsm_x2t(b0, b1,
                   Ksm + 64 * hf + 8 * kq + (size_t)((lane & 7) + 8 * ((lane >> 3) & 1)) * SD);
          mma_bf16(dd, a2, a3, a0, a1, b1, b0);
          *plo = make_float2(dd.c[0], dd.c[1]);
          *phi = make_float2(dd.c[2], dd.c[3]);
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
                                    (u & 7) * 8) = make_uint4(f2bf_x2(x.x, x.y), f2bf_x2(x.z, x.w),
                                                              f2bf_x2(y.x, y.y), f2bf_x2(y.z, y.w));
        }
        if (hf == 0) {
          __syncwarp();
          if (tid < STAMP) Stamp[tid] = blk[tid];
          for (int u = lane; u < 256; u += 32)
            cp_async16(blk + (u >> 4) * FS + (u & 15) * 4,
                       Srow + (size_t)(u >> 4) * D + 64 + (u & 15) * 4);
          cp_async_commit();
          cp_async_wait_all();
          __syncwarp();
        }
      }
      p = 0;
      if (SPAN && j + 1 < n) {
        // the next chunk of the span starts from the state just committed
        __syncthreads();
        for (int u = tid; u < D * 16; u += NTHR)
          cp_async16(Ssm + (u >> 4) * SD + (u & 15) * 8, S16 + (size_t)u * 8);
        cp_async_commit();
        Lsm[tid] = (uint16_t)0;
      }
    }
  }
  if (tid == 0) *cnt_cell = p;
}

// The slots' fp32 states become chunk boundaries (bf16 copy, stamp, count); CTA = (slot, head).
__global__ void BeginKernel(const int* __restrict__ idx, const float* __restrict__ state,
                            uint16_t* state16, int* cnt, float* stamp, int H) {
  const int tid = threadIdx.x, h = blockIdx.x % H;
  const int slot = idx[blockIdx.x / H];
  if (slot < 0) return;
  const size_t sh = (size_t)slot * H + h;
  const float* S32 = state + sh * D * D;
  uint16_t* S16 = state16 + sh * D * D;
  // four 8-value pieces per round: their loads are issued before the first one is used
  for (int e = tid; e < D * D / 8; e += 4 * NTHR) {
    float4 x[4], y[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      x[i] = *reinterpret_cast<const float4*>(S32 + 8 * (e + i * NTHR));
      y[i] = *reinterpret_cast<const float4*>(S32 + 8 * (e + i * NTHR) + 4);
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
      *reinterpret_cast<uint4*>(S16 + 8 * (e + i * NTHR)) =
          make_uint4(f2bf_x2(x[i].x, x[i].y), f2bf_x2(x[i].z, x[i].w), f2bf_x2(y[i].x, y[i].y),
                     f2bf_x2(y[i].z, y[i].w));
  }
  if (tid < STAMP) stamp[sh * STAMP + tid] = S32[tid];
  if (tid == 0) cnt[sh] = 0;
}

inline cudaError_t Begin(const int* slots, const float* state, uint16_t* state_bf16, int* cnt,
                         float* stamp, int batch, int num_heads, cudaStream_t stream) {
  BeginKernel<<<batch * num_heads, NTHR, 0, stream>>>(slots, state, state_bf16, cnt, stamp,
                                                      num_heads);
  return cudaGetLastError();
}

inline cudaError_t DecodeStep(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                              const uint16_t* g, const uint16_t* beta, const int* slots,
                              uint16_t* out, float* state, uint16_t* state_bf16, int* cnt,
                              float* prefix, uint16_t* k_inv_rows, float* k_inv_f32_rows,
                              uint16_t* rhs_rows, uint16_t* update_rows, uint16_t* l_rows,
                              float* stamp, const float* a_log, const float* dt_bias, float scale,
                              float half_gate_scale_log2, int batch, int num_heads,
                              cudaStream_t stream) {
  cudaError_t status =
      cudaFuncSetAttribute(DecodeKernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(kSmemBytes));
  if (status != cudaSuccess) return status;
  DecodeKernel<false><<<batch * num_heads, NTHR, kSmemBytes, stream>>>(
      q, k, v, g, beta, slots, nullptr, nullptr, 0, out, state, state_bf16, cnt, prefix, k_inv_rows,
      k_inv_f32_rows, rhs_rows, update_rows, l_rows, stamp, a_log, dt_bias, scale,
      half_gate_scale_log2, num_heads);
  return cudaGetLastError();
}

// Sequence i appends rows [starts[i], starts[i] + lens[i]) of the packed inputs to slot slots[i];
// with starts == nullptr every sequence appends `stride` rows from row i * stride. One launch.
inline cudaError_t DecodeSpan(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                              const uint16_t* g, const uint16_t* beta, const int* slots,
                              const int* starts, const int* lens, int stride, uint16_t* out,
                              float* state, uint16_t* state_bf16, int* cnt, float* prefix,
                              uint16_t* k_inv_rows, float* k_inv_f32_rows, uint16_t* rhs_rows,
                              uint16_t* update_rows, uint16_t* l_rows, float* stamp,
                              const float* a_log, const float* dt_bias, float scale,
                              float half_gate_scale_log2, int batch, int num_heads,
                              cudaStream_t stream) {
  cudaError_t status =
      cudaFuncSetAttribute(DecodeKernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(kSpanSmemBytes));
  if (status != cudaSuccess) return status;
  DecodeKernel<true><<<batch * num_heads, NTHR, kSpanSmemBytes, stream>>>(
      q, k, v, g, beta, slots, starts, lens, stride, out, state, state_bf16, cnt, prefix,
      k_inv_rows, k_inv_f32_rows, rhs_rows, update_rows, l_rows, stamp, a_log, dt_bias, scale,
      half_gate_scale_log2, num_heads);
  return cudaGetLastError();
}

}  // namespace bi_decode
}  // namespace kda
}  // namespace flashinfer

#endif  // FLASHINFER_KDA_BATCH_INVARIANT_DECODE_CUH_
