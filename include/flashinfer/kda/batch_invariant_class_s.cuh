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
#ifndef FLASHINFER_KDA_BATCH_INVARIANT_CLASS_S_CUH_
#define FLASHINFER_KDA_BATCH_INVARIANT_CLASS_S_CUH_

// The kernels of the KDA class kda.bf16.s16; compile with -fmad=false (the term's fmas are
// explicit).

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace flashinfer {
namespace kda {
namespace bi_class_s {

constexpr int C = 16;
constexpr int D = 128;
constexpr int NWARP = 8;
constexpr int NTHR = 256;
// row stride of operand tiles: every lane of a fragment load has its own bank
constexpr int SD = D + 8;
constexpr int STAMP = 2;

struct Acc4 {
  float c[4];
};

// one mma.sync m16n8k16 step, bf16 operands, fp32 accumulate
__device__ __forceinline__ void mma_bf16(Acc4& d, uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d.c[0]), "+f"(d.c[1]), "+f"(d.c[2]), "+f"(d.c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float bf2f(uint32_t x) { return __uint_as_float((x & 0xffffu) << 16); }
// two fp32 -> one packed bf16 pair (lo | hi << 16), round to nearest even
__device__ __forceinline__ uint32_t f2bf_x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ uint16_t f2bf(float f) { return (uint16_t)f2bf_x2(f, f); }
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

// the hardware's elementary functions: one instruction each, part of the class's term
__device__ __forceinline__ float ex2a(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float rsqrta(float x) {
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
constexpr float kLog2E = 1.4426950408889634f;

struct AReg {
  uint32_t a[32];
};
struct BReg {
  uint32_t b[16];
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
// A fragments of a 16-row group for 8 k-steps from a shared tile with row stride `stride`
__device__ __forceinline__ void load_a(AReg& r, const uint16_t* base, int stride, int lane) {
  const uint16_t* p =
      base + (size_t)((lane & 7) + 8 * ((lane >> 3) & 1)) * stride + 8 * (lane >> 4);
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
    ldsm_x4(r.a[4 * ks], r.a[4 * ks + 1], r.a[4 * ks + 2], r.a[4 * ks + 3], p + 16 * ks);
}
// B fragments for 8 k-steps: `row` is this lane's column vector (column lane & 7 of the group)
__device__ __forceinline__ void load_b(BReg& r, const uint16_t* row, int lane) {
  const uint16_t* p = row + 8 * ((lane >> 3) & 1) + 16 * (lane >> 4);
#pragma unroll
  for (int x = 0; x < 4; ++x)
    ldsm_x4(r.b[4 * x], r.b[4 * x + 1], r.b[4 * x + 2], r.b[4 * x + 3], p + 32 * x);
}
// MMA128: the 128-term bf16 dot fold, 8 mma steps in ascending k from a zero accumulator
__device__ __forceinline__ Acc4 mma128(const AReg& A, const BReg& B) {
  Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
    mma_bf16(d, A.a[4 * ks], A.a[4 * ks + 1], A.a[4 * ks + 2], A.a[4 * ks + 3], B.b[2 * ks],
             B.b[2 * ks + 1]);
  return d;
}
// two MMA128 folds that share A, interleaved (each element's instruction sequence is mma128's)
__device__ __forceinline__ void mma128x2(const AReg& A, const BReg& B0, const BReg& B1, Acc4& d0,
                                         Acc4& d1) {
  d0 = {{0.f, 0.f, 0.f, 0.f}};
  d1 = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    mma_bf16(d0, A.a[4 * ks], A.a[4 * ks + 1], A.a[4 * ks + 2], A.a[4 * ks + 3], B0.b[2 * ks],
             B0.b[2 * ks + 1]);
    mma_bf16(d1, A.a[4 * ks], A.a[4 * ks + 1], A.a[4 * ks + 2], A.a[4 * ks + 3], B1.b[2 * ks],
             B1.b[2 * ks + 1]);
  }
}

// The fold of one 8-column group of the lane's state pairs x / y (rows g, g + 8).
__device__ __forceinline__ void fold_tile(uint32_t& x, uint32_t& y, float dlo, float dhi,
                                          const uint32_t* fa, const uint16_t* kcol, int lane) {
  Acc4 d;
  d.c[0] = dlo * __uint_as_float(x << 16);
  d.c[1] = dhi * __uint_as_float(x & 0xffff0000u);
  d.c[2] = dlo * __uint_as_float(y << 16);
  d.c[3] = dhi * __uint_as_float(y & 0xffff0000u);
  uint32_t b0, b1;
  ldsm_x2t(b0, b1, kcol + (size_t)((lane & 7) + 8 * ((lane >> 3) & 1)) * SD);
  mma_bf16(d, fa[0], fa[1], fa[2], fa[3], b0, b1);
  x = f2bf_x2(d.c[0], d.c[1]);
  y = f2bf_x2(d.c[2], d.c[3]);
}

// Per-channel pieces of one token (4 channels per lane).
struct Ch4 {
  float x[4];
};
__device__ __forceinline__ Ch4 unpack4(uint64_t raw) {
  Ch4 r;
  r.x[0] = bf2f((uint32_t)raw);
  r.x[1] = bf2f((uint32_t)(raw >> 16));
  r.x[2] = bf2f((uint32_t)(raw >> 32));
  r.x[3] = bf2f((uint32_t)(raw >> 48));
  return r;
}
__device__ __forceinline__ uint64_t pack4(float a, float b, float c, float d) {
  return (uint64_t)f2bf_x2(a, b) | ((uint64_t)f2bf_x2(c, d) << 32);
}
// rsqrt(sum of squares + 1e-6) of 128 values held 4 per lane
__device__ __forceinline__ float warp_rnorm(const Ch4& v) {
  float s = v.x[0] * v.x[0];
  s = s + v.x[1] * v.x[1];
  s = s + v.x[2] * v.x[2];
  s = s + v.x[3] * v.x[3];
  for (int off = 16; off > 0; off >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, off);
  return rsqrta(s + 1e-6f);
}
// log2 of one channel's decay; ea = exp(A_log) / 2, hs = lower_bound * log2(e) / 2
__device__ __forceinline__ float log2_decay(float g, float dtb, float ea, float hs) {
  return __fmaf_rn(tanha(ea * (g + dtb)), hs, hs);
}
__device__ __forceinline__ float half_exp_a(float a_log) { return ex2a(a_log * kLog2E) * 0.5f; }
__device__ __forceinline__ float beta_of(uint32_t braw) {
  return __fmaf_rn(tanha(0.5f * bf2f(braw)), 0.5f, 0.5f);
}

// shared memory of a prefill block, in bytes
constexpr int kRawRow = D * 2;
constexpr int kTile = C * SD * 2;
// row stride of the per-row product table
constexpr int SA = D + 4;
constexpr int kPrefillSmemBytes =
    2 * 4 * C * kRawRow + 5 * kTile + 2 * C * D * 4 + 2 * C * SA * 4 + C * C * 8 + 2 * D * 4 + 64;

// A block is one (sequence, head): it loops over the chunks with the state in registers.
__global__ void __launch_bounds__(NTHR)
    PrefillKernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                  const uint16_t* __restrict__ v, const uint16_t* __restrict__ gch,
                  const uint16_t* __restrict__ gb, const float* __restrict__ A_log,
                  const float* __restrict__ dt_bias, uint16_t* pool, const int* __restrict__ idx,
                  const int* __restrict__ cu, uint16_t* __restrict__ out, uint16_t* Kh, float* Uh,
                  float* Lh, int* cnt, uint64_t* stamp, float scale, float hs, int H, int num_slots,
                  int fresh) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int h = blockIdx.x % H, b = blockIdx.x / H;
  const int slot = idx[b];
  if ((unsigned)slot >= (unsigned)num_slots) return;
  const int tok_begin = cu[b], Ttot = cu[b + 1] - cu[b];
  if (Ttot <= 0) return;
  const size_t sh = (size_t)slot * H + h;
  uint16_t* Spool = pool + sh * D * D;
  uint64_t* stamp_cell = stamp + sh * STAMP;
  const uint64_t* shead = reinterpret_cast<const uint64_t*>(Spool);
  // a continuing call drops the open rows of a state the caller replaced
  const bool keep = !fresh && stamp_cell[0] == shead[0] && stamp_cell[1] == shead[1];
  const int n0 = keep ? cnt[sh] : 0;

  size_t off = 0;
  uint16_t* Raw =
      reinterpret_cast<uint16_t*>(smem + off);  // [2 buffers][k, q, g, v][C][D] raw rows
  off += (size_t)2 * 4 * C * kRawRow;
  uint16_t* Ktsm = reinterpret_cast<uint16_t*>(smem + off);  // kt of the call's tokens [C][SD]
  off += kTile;
  uint16_t* Qtsm = reinterpret_cast<uint16_t*>(smem + off);  // qt [C][SD]
  off += kTile;
  uint16_t* Ksm = reinterpret_cast<uint16_t*>(smem + off);  // ki by chunk position [C][SD]
  off += kTile;
  uint16_t* Kbsm = reinterpret_cast<uint16_t*>(smem + off);  // ki decayed to the chunk end [C][SD]
  off += kTile;
  float* Lsm = reinterpret_cast<float*>(smem + off);  // log2-decay prefix rows [C][D]
  off += (size_t)C * D * 4;
  float* Usm = reinterpret_cast<float*>(smem + off);  // u rows [C][D]
  off += (size_t)C * D * 4;
  uint16_t* Utsm = reinterpret_cast<uint16_t*>(smem + off);  // bf16(u) of a folding chunk [C][SD]
  off += kTile;
  float* ABsm =
      reinterpret_cast<float*>(smem + off);  // S0 . kt_i and S0 . qt_i by state row [2 C][SA]
  off += (size_t)2 * C * SA * 4;
  float* CEsm =
      reinterpret_cast<float*>(smem + off);  // [position j][token i]{ki_j . kt_i, ki_j . qt_i}
  off += (size_t)C * C * 8;
  float* dbsm = reinterpret_cast<float*>(smem + off);  // dt_bias row
  off += D * 4;
  float* dcsm = reinterpret_cast<float*>(smem + off);  // decay to the chunk end
  off += D * 4;
  float* betasm = reinterpret_cast<float*>(smem + off);

  uint16_t* Kpool = Kh + sh * C * D;
  float* Upool = Uh + sh * C * D;
  float* Lpool = Lh + sh * C * D;
  const float ea = half_exp_a(A_log[h]);
  const int row0 = 16 * warp + g, row1 = row0 + 8;

  uint32_t b_next = 0;
  auto stage = [&](int buf, size_t tokA, int mm) {
    uint16_t* dst = Raw + (size_t)buf * 4 * C * D;
    for (int u = tid; u < mm * 16; u += NTHR) {
      const int row = u >> 4, c8 = (u & 15) * 8;
      const size_t src = ((tokA + row) * H + h) * D + c8;
      cp_async16(dst + row * D + c8, k + src);
      cp_async16(dst + (C + row) * D + c8, q + src);
      cp_async16(dst + (2 * C + row) * D + c8, gch + src);
      cp_async16(dst + (3 * C + row) * D + c8, v + src);
    }
    cp_async_commit();
    if (tid < mm) b_next = gb[(tokA + tid) * H + h];
  };

  stage(0, (size_t)tok_begin, (C - n0) < Ttot ? (C - n0) : Ttot);
  // the state as A fragments: rows row0 / row1, the lane's two column pairs of every k-step
  AReg S;
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    const int c = 16 * ks + 2 * t4;
    S.a[4 * ks] = *reinterpret_cast<const uint32_t*>(Spool + (size_t)row0 * D + c);
    S.a[4 * ks + 1] = *reinterpret_cast<const uint32_t*>(Spool + (size_t)row1 * D + c);
    S.a[4 * ks + 2] = *reinterpret_cast<const uint32_t*>(Spool + (size_t)row0 * D + c + 8);
    S.a[4 * ks + 3] = *reinterpret_cast<const uint32_t*>(Spool + (size_t)row1 * D + c + 8);
  }
  for (int u = tid; u < n0 * 16; u += NTHR)
    cp_async16(Ksm + (u >> 4) * SD + (u & 15) * 8, Kpool + (size_t)u * 8);
  cp_async_commit();
  for (int e = tid; e < n0 * D; e += NTHR) Usm[e] = Upool[e];
  if (tid < D) {
    dbsm[tid] = dt_bias[(size_t)h * D + tid];
    if (n0 > 0) Lsm[(size_t)(n0 - 1) * D + tid] = Lpool[(size_t)(n0 - 1) * D + tid];
  }
  bool folded = false, crossed = false;

  int n = n0, t = 0, cur = 0;
  while (t < Ttot) {
    const int p0 = n;
    const int m = (C - p0) < (Ttot - t) ? (C - p0) : (Ttot - t);
    const bool fold = p0 + m == C;
    const size_t tok0 = (size_t)tok_begin + t;
    const uint16_t* Rk = Raw + (size_t)cur * 4 * C * D;
    const uint16_t* Rq = Rk + C * D;
    const uint16_t* Rg = Rq + C * D;
    const uint16_t* Rv = Rg + C * D;

    cp_async_wait_all();
    const uint32_t b_cur = b_next;
    {
      const int rem = Ttot - t - m;
      if (rem > 0) stage(cur ^ 1, tok0 + m, rem < C ? rem : C);
    }
    if (tid < m) betasm[tid] = beta_of(b_cur);
    __syncthreads();
    // per token (two per warp): norms and the per-channel log2-decay
    uint64_t kraw[2], qraw[2];
    float rk[2], rq[2];
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int i = warp + 8 * it;
      if (i < m) {
        kraw[it] = *reinterpret_cast<const uint64_t*>(Rk + (size_t)i * D + 4 * lane);
        qraw[it] = *reinterpret_cast<const uint64_t*>(Rq + (size_t)i * D + 4 * lane);
        const Ch4 gr = unpack4(*reinterpret_cast<const uint64_t*>(Rg + (size_t)i * D + 4 * lane));
        rk[it] = warp_rnorm(unpack4(kraw[it]));
        rq[it] = warp_rnorm(unpack4(qraw[it])) * scale;
        const float4 db = *reinterpret_cast<const float4*>(dbsm + 4 * lane);
        *reinterpret_cast<float4*>(Lsm + (size_t)(p0 + i) * D + 4 * lane) =
            make_float4(log2_decay(gr.x[0], db.x, ea, hs), log2_decay(gr.x[1], db.y, ea, hs),
                        log2_decay(gr.x[2], db.z, ea, hs), log2_decay(gr.x[3], db.w, ea, hs));
      }
    }
    __syncthreads();
    // prefix over the chunk, per channel, sequential in the token index
    if (tid < D) {
      float acc = p0 ? Lsm[(size_t)(p0 - 1) * D + tid] : 0.0f;
      for (int i = 0; i < m; ++i) {
        const float l = Lsm[(size_t)(p0 + i) * D + tid];
        acc = (p0 + i) ? acc + l : l;
        Lsm[(size_t)(p0 + i) * D + tid] = acc;
      }
    }
    __syncthreads();
    // operands kt, ki, qt, and ki decayed to the chunk end for the fold
    float4 lend = make_float4(0.f, 0.f, 0.f, 0.f);
    if (fold) lend = *reinterpret_cast<const float4*>(Lsm + (size_t)(C - 1) * D + 4 * lane);
    const float de[4] = {ex2a(lend.x), ex2a(lend.y), ex2a(lend.z), ex2a(lend.w)};
    if (fold && warp == 0)
      *reinterpret_cast<float4*>(dcsm + 4 * lane) = make_float4(de[0], de[1], de[2], de[3]);
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const int i = warp + 8 * it;
      if (i < m) {
        const Ch4 kr = unpack4(kraw[it]), qr = unpack4(qraw[it]);
        const float4 lam4 = *reinterpret_cast<const float4*>(Lsm + (size_t)(p0 + i) * D + 4 * lane);
        const float lam[4] = {lam4.x, lam4.y, lam4.z, lam4.w};
        float kd[4], ki[4], qd[4];
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          const float dec = ex2a(lam[x]);
          const float kn = kr.x[x] * rk[it];
          kd[x] = kn * dec;
          ki[x] = kn * rcpa(dec);
          qd[x] = (qr.x[x] * rq[it]) * dec;
        }
        const uint64_t kip = pack4(ki[0], ki[1], ki[2], ki[3]);
        *reinterpret_cast<uint64_t*>(Ktsm + (size_t)i * SD + 4 * lane) =
            pack4(kd[0], kd[1], kd[2], kd[3]);
        *reinterpret_cast<uint64_t*>(Qtsm + (size_t)i * SD + 4 * lane) =
            pack4(qd[0], qd[1], qd[2], qd[3]);
        *reinterpret_cast<uint64_t*>(Ksm + (size_t)(p0 + i) * SD + 4 * lane) = kip;
        if (fold) {
          const Ch4 kq = unpack4(kip);
          *reinterpret_cast<uint64_t*>(Kbsm + (size_t)(p0 + i) * SD + 4 * lane) =
              pack4(kq.x[0] * de[0], kq.x[1] * de[1], kq.x[2] * de[2], kq.x[3] * de[3]);
        }
      }
    }
    if (fold)
      for (int j = warp; j < p0; j += NWARP) {
        const Ch4 kq = unpack4(*reinterpret_cast<const uint64_t*>(Ksm + (size_t)j * SD + 4 * lane));
        *reinterpret_cast<uint64_t*>(Kbsm + (size_t)j * SD + 4 * lane) =
            pack4(kq.x[0] * de[0], kq.x[1] * de[1], kq.x[2] * de[2], kq.x[3] * de[3]);
      }
    __syncthreads();

    // P = S0 . kt and O = S0 . qt for this lane's rows and token columns
    Acc4 P[2] = {}, O[2] = {};
#pragma unroll
    for (int cg = 0; cg < 2; ++cg) {
      if (8 * cg < m) {
        int col = 8 * cg + (lane & 7);
        col = col < m ? col : m - 1;
        BReg Bk, Bq;
        load_b(Bk, Ktsm + (size_t)col * SD, lane);
        load_b(Bq, Qtsm + (size_t)col * SD, lane);
        mma128x2(S, Bk, Bq, P[cg], O[cg]);
      }
    }
    // the chunk's key products ki_j . kt_i and ki_j . qt_i: one column group per warp
    if (warp < 4 && 8 * (warp & 1) < m) {
      int col = 8 * (warp & 1) + (lane & 7);
      col = col < m ? col : m - 1;
      AReg A;
      BReg B;
      load_a(A, Ksm, SD, lane);
      load_b(B, (warp < 2 ? Ktsm : Qtsm) + (size_t)col * SD, lane);
      const Acc4 d = mma128(A, B);
      float* ce = CEsm + (8 * (warp & 1) + 2 * t4) * 2 + (warp >> 1);
      ce[(size_t)g * C * 2] = d.c[0];
      ce[(size_t)g * C * 2 + 2] = d.c[1];
      ce[(size_t)(g + 8) * C * 2] = d.c[2];
      ce[(size_t)(g + 8) * C * 2 + 2] = d.c[3];
    }
    __syncthreads();

    // hand the products to one thread per state row
#pragma unroll
    for (int cg = 0; cg < 2; ++cg) {
      float* ab = ABsm + (size_t)(8 * cg + 2 * t4) * SA;
      ab[row0] = P[cg].c[0];
      ab[SA + row0] = P[cg].c[1];
      ab[row1] = P[cg].c[2];
      ab[SA + row1] = P[cg].c[3];
      ab[C * SA + row0] = O[cg].c[0];
      ab[(C + 1) * SA + row0] = O[cg].c[1];
      ab[C * SA + row1] = O[cg].c[2];
      ab[(C + 1) * SA + row1] = O[cg].c[3];
    }
    if (fold)
      for (int e = tid; e < p0 * D; e += NTHR) Utsm[(size_t)(e / D) * SD + (e % D)] = f2bf(Usm[e]);
    __syncthreads();

    // substitution, sequential in the chunk position: thread = state row
    if (tid < D) {
      float pred[C], o[C];
#pragma unroll
      for (int i = 0; i < C; ++i) {
        pred[i] = ABsm[(size_t)i * SA + tid];
        o[i] = ABsm[(size_t)(C + i) * SA + tid];
      }
      for (int j = 0; j < p0; ++j) {
        const float u = Usm[(size_t)j * D + tid];
#pragma unroll
        for (int i = 0; i < C; i += 2) {
          const float4 c = *reinterpret_cast<const float4*>(CEsm + ((size_t)j * C + i) * 2);
          pred[i] = __fmaf_rn(c.x, u, pred[i]);
          o[i] = __fmaf_rn(c.y, u, o[i]);
          pred[i + 1] = __fmaf_rn(c.z, u, pred[i + 1]);
          o[i + 1] = __fmaf_rn(c.w, u, o[i + 1]);
        }
      }
#pragma unroll
      for (int i = 0; i < C; ++i) {
        if (i < m) {
          const int j = p0 + i;
          const float* ce = CEsm + (size_t)j * C * 2;
          const float u = betasm[i] * (bf2f(Rv[(size_t)i * D + tid]) - pred[i]);
          if (fold)
            Utsm[(size_t)j * SD + tid] = f2bf(u);
          else
            Usm[(size_t)j * D + tid] = u;
          o[i] = __fmaf_rn(ce[2 * i + 1], u, o[i]);
          out[((tok0 + i) * H + h) * D + tid] = f2bf(o[i]);
          if ((i & 1) == 0) {
            const float2 c = *reinterpret_cast<const float2*>(ce + 2 * (i + 1));
            pred[i + 1] = __fmaf_rn(c.x, u, pred[i + 1]);
            o[i + 1] = __fmaf_rn(c.y, u, o[i + 1]);
          }
#pragma unroll
          for (int i2 = (i + 2) & ~1; i2 < C; i2 += 2) {
            const float4 c = *reinterpret_cast<const float4*>(ce + 2 * i2);
            pred[i2] = __fmaf_rn(c.x, u, pred[i2]);
            o[i2] = __fmaf_rn(c.y, u, o[i2]);
            pred[i2 + 1] = __fmaf_rn(c.z, u, pred[i2 + 1]);
            o[i2 + 1] = __fmaf_rn(c.w, u, o[i2 + 1]);
          }
        }
      }
    }
    t += m;
    n += m;
    cur ^= 1;
    if (fold) {
      __syncthreads();
      uint32_t fa[4];
      ldsm_x4t(
          fa[0], fa[1], fa[2], fa[3],
          Utsm + (size_t)((lane & 7) + 8 * (lane >> 4)) * SD + 16 * warp + 8 * ((lane >> 3) & 1));
#pragma unroll
      for (int kg = 0; kg < 16; ++kg) {
        const float2 dc = *reinterpret_cast<const float2*>(dcsm + 8 * kg + 2 * t4);
        const int r = 4 * (kg >> 1) + 2 * (kg & 1);
        fold_tile(S.a[r], S.a[r + 1], dc.x, dc.y, fa, Kbsm + 8 * kg, lane);
      }
      n = 0;
      folded = true;
      crossed = true;
    }
    __syncthreads();
  }

  if (folded) {
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int c = 16 * ks + 2 * t4;
      *reinterpret_cast<uint32_t*>(Spool + (size_t)row0 * D + c) = S.a[4 * ks];
      *reinterpret_cast<uint32_t*>(Spool + (size_t)row1 * D + c) = S.a[4 * ks + 1];
      *reinterpret_cast<uint32_t*>(Spool + (size_t)row0 * D + c + 8) = S.a[4 * ks + 2];
      *reinterpret_cast<uint32_t*>(Spool + (size_t)row1 * D + c + 8) = S.a[4 * ks + 3];
    }
  }
  const int w0 = crossed ? 0 : n0;
  if (n > w0) {
    for (int u = tid + w0 * 16; u < n * 16; u += NTHR)
      *reinterpret_cast<uint4*>(Kpool + (size_t)u * 8) =
          *reinterpret_cast<const uint4*>(Ksm + (u >> 4) * SD + (u & 15) * 8);
    for (int e = tid + w0 * D; e < n * D; e += NTHR) {
      Upool[e] = Usm[e];
      Lpool[e] = Lsm[e];
    }
  }
  if (tid == 0) cnt[sh] = n;
  // the stamp is the first 8 values of the state's row 0, held by lanes 0..3 of warp 0
  if (tid < 4) reinterpret_cast<uint32_t*>(stamp_cell)[tid] = S.a[0];
}

// One token per sequence; a block is one (sequence, head).
__global__ void __launch_bounds__(NTHR)
    DecodeKernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                 const uint16_t* __restrict__ v, const uint16_t* __restrict__ gch,
                 const uint16_t* __restrict__ gb, const float* __restrict__ A_log,
                 const float* __restrict__ dt_bias, uint16_t* pool, const int* __restrict__ idx,
                 uint16_t* __restrict__ out, uint16_t* Kh, float* Uh, float* Lh, int* cnt,
                 uint64_t* stamp, float scale, float hs, int H, int num_slots) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int h = blockIdx.x % H, b = blockIdx.x / H;
  const int slot = idx[b];
  if ((unsigned)slot >= (unsigned)num_slots) return;
  const size_t tok = b;
  const size_t sh = (size_t)slot * H + h;
  const int c0 = cnt[sh];
  uint64_t* stamp_cell = stamp + sh * STAMP;
  const uint64_t* shead = reinterpret_cast<const uint64_t*>(pool + sh * D * D);
  const uint64_t st0 = stamp_cell[0], st1 = stamp_cell[1], sh0 = shead[0], sh1 = shead[1];

  size_t off = 0;
  uint16_t* Ssm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)D * SD * 2;
  uint16_t* Ksm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)C * SD * 2;
  float* Usm = reinterpret_cast<float*>(smem + off);
  off += (size_t)C * D * 4;
  uint16_t* Ktsm = reinterpret_cast<uint16_t*>(smem + off);  // k * D
  off += SD * 2;
  uint16_t* Qsm = reinterpret_cast<uint16_t*>(smem + off);  // q * D
  off += SD * 2;
  float* dcsm = reinterpret_cast<float*>(smem + off);  // decay prefix D_p per channel
  off += D * 4;
  float* betasm = reinterpret_cast<float*>(smem + off);
  off += 16;
  float* ABsm = reinterpret_cast<float*>(smem + off);
  off += 2 * D * 4;
  float* CEsm = reinterpret_cast<float*>(smem + off);
  off += 2 * C * 4;
  uint16_t* Utsm = reinterpret_cast<uint16_t*>(smem + off);

  uint16_t* Spool = pool + sh * D * D;
  for (int u = tid; u < D * 16; u += NTHR)
    cp_async16(Ssm + (u >> 4) * SD + (u & 15) * 8, Spool + (size_t)u * 8);
  uint64_t rawkq = 0, rawg = 0;
  float4 dtb = make_float4(0.f, 0.f, 0.f, 0.f);
  float alog = 0.0f;
  if (warp < 2) {
    rawkq = *reinterpret_cast<const uint64_t*>((warp == 0 ? k : q) + (tok * H + h) * D + 4 * lane);
    rawg = *reinterpret_cast<const uint64_t*>(gch + (tok * H + h) * D + 4 * lane);
    dtb = *reinterpret_cast<const float4*>(dt_bias + (size_t)h * D + 4 * lane);
    alog = A_log[h];
  }
  uint32_t vraw = 0, braw = 0;
  if (tid < D) vraw = v[(tok * H + h) * D + tid];
  if (tid == 0) braw = gb[tok * H + h];

  // a state the caller replaced since the class last wrote it starts a new chunk
  const bool keep = st0 == sh0 && st1 == sh1;
  const int p = keep ? c0 : 0;
  uint16_t* Kpool = Kh + sh * C * D;
  float* Upool = Uh + sh * C * D;
  float* Lpool = Lh + sh * C * D;
  float4 lprev = make_float4(0.f, 0.f, 0.f, 0.f);
  if (warp < 2 && p > 0)
    lprev = *reinterpret_cast<const float4*>(Lpool + (size_t)(p - 1) * D + 4 * lane);
  for (int u = tid; u < p * 16; u += NTHR)
    cp_async16(Ksm + (u >> 4) * SD + (u & 15) * 8, Kpool + (size_t)u * 8);
  for (int u = tid; u < p * 32; u += NTHR) cp_async16(Usm + (size_t)u * 4, Upool + (size_t)u * 4);
  cp_async_commit();
  if (warp < 2) {
    const Ch4 xr = unpack4(rawkq);
    const Ch4 gr = unpack4(rawg);
    const float ea = half_exp_a(alog);
    const float rn = warp == 0 ? warp_rnorm(xr) : warp_rnorm(xr) * scale;
    const float db[4] = {dtb.x, dtb.y, dtb.z, dtb.w}, lp[4] = {lprev.x, lprev.y, lprev.z, lprev.w};
    float lam[4], xd[4], xi[4];
#pragma unroll
    for (int x = 0; x < 4; ++x) {
      const float l = log2_decay(gr.x[x], db[x], ea, hs);
      lam[x] = p ? lp[x] + l : l;
      const float dec = ex2a(lam[x]);
      const float xn = xr.x[x] * rn;
      xd[x] = xn * dec;
      xi[x] = xn * rcpa(dec);
      if (warp == 0) dcsm[4 * lane + x] = dec;
    }
    if (warp == 0) {
      *reinterpret_cast<uint64_t*>(Ktsm + 4 * lane) = pack4(xd[0], xd[1], xd[2], xd[3]);
      const uint64_t ki = pack4(xi[0], xi[1], xi[2], xi[3]);
      *reinterpret_cast<uint64_t*>(Ksm + (size_t)p * SD + 4 * lane) = ki;
      *reinterpret_cast<uint64_t*>(Kpool + (size_t)p * D + 4 * lane) = ki;
      *reinterpret_cast<float4*>(Lpool + (size_t)p * D + 4 * lane) =
          make_float4(lam[0], lam[1], lam[2], lam[3]);
    } else {
      *reinterpret_cast<uint64_t*>(Qsm + 4 * lane) = pack4(xd[0], xd[1], xd[2], xd[3]);
    }
  }
  if (tid == 0) betasm[0] = beta_of(braw);
  cp_async_wait_all();
  __syncthreads();

  BReg B;
  load_b(B, (lane & 7) == 0 ? Ktsm : Qsm, lane);
  {
    AReg A;
    load_a(A, Ssm + (size_t)(16 * warp) * SD, SD, lane);
    const Acc4 d = mma128(A, B);
    if (t4 == 0) {
      const int r = 16 * warp + g;
      ABsm[r] = d.c[0];
      ABsm[D + r] = d.c[1];
      ABsm[r + 8] = d.c[2];
      ABsm[D + r + 8] = d.c[3];
    }
  }
  if (warp == 0) {
    const int j = 16 * warp + g;
    AReg A2;
    load_a(A2, Ksm + (size_t)(16 * warp) * SD, SD, lane);
    const Acc4 d = mma128(A2, B);
    if (t4 == 0) {
      CEsm[2 * j] = d.c[0];
      CEsm[2 * j + 1] = d.c[1];
      CEsm[2 * (j + 8)] = d.c[2];
      CEsm[2 * (j + 8) + 1] = d.c[3];
    }
  }
  __syncthreads();
  if (tid < D) {
    float pred = ABsm[tid];
    float o = ABsm[D + tid];
    const float* up = Usm + tid;
    for (int j = 0; j < p; ++j) {
      const float u = up[(size_t)j * D];
      pred = __fmaf_rn(CEsm[2 * j], u, pred);
      o = __fmaf_rn(CEsm[2 * j + 1], u, o);
    }
    const float u = betasm[0] * (bf2f(vraw) - pred);
    Usm[(size_t)p * D + tid] = u;
    Upool[(size_t)p * D + tid] = u;
    o = __fmaf_rn(CEsm[2 * p + 1], u, o);
    out[(tok * H + h) * D + tid] = f2bf(o);
  }
  if (tid == 0) cnt[sh] = (p + 1 == C) ? 0 : p + 1;
  if (!keep && tid < STAMP) stamp_cell[tid] = tid ? sh1 : sh0;
  if (p + 1 == C) {
    __syncthreads();
    for (int e = tid; e < D * C; e += NTHR) {
      Utsm[(size_t)(e / D) * SD + (e % D)] = f2bf(Usm[e]);
      uint16_t* kp = Ksm + (size_t)(e / D) * SD + (e % D);
      *kp = f2bf(bf2f(*kp) * dcsm[e % D]);
    }
    __syncthreads();
    const int r = 16 * warp + g;
    uint32_t fa[4];
    ldsm_x4t(
        fa[0], fa[1], fa[2], fa[3],
        Utsm + (size_t)((lane & 7) + 8 * (lane >> 4)) * SD + 16 * warp + 8 * ((lane >> 3) & 1));
#pragma unroll
    for (int kg = 0; kg < 16; ++kg) {
      const int kc = 8 * kg + 2 * t4;
      uint32_t x = *reinterpret_cast<uint32_t*>(Ssm + (size_t)r * SD + kc);
      uint32_t y = *reinterpret_cast<uint32_t*>(Ssm + (size_t)(r + 8) * SD + kc);
      fold_tile(x, y, dcsm[kc], dcsm[kc + 1], fa, Ksm + 8 * kg, lane);
      *reinterpret_cast<uint32_t*>(Ssm + (size_t)r * SD + kc) = x;
      *reinterpret_cast<uint32_t*>(Ssm + (size_t)(r + 8) * SD + kc) = y;
    }
    __syncthreads();
    for (int u = tid; u < D * 16; u += NTHR)
      *reinterpret_cast<uint4*>(Spool + (size_t)u * 8) =
          *reinterpret_cast<const uint4*>(Ssm + (u >> 4) * SD + (u & 15) * 8);
    if (tid < STAMP) stamp_cell[tid] = *reinterpret_cast<const uint64_t*>(Ssm + 4 * tid);
  }
}

constexpr int kDecodeSmemBytes = D * SD * 2 + C * SD * 2 + C * D * 4 + 2 * SD * 2 + D * 4 + 16 +
                                 2 * D * 4 + 2 * C * 4 + C * SD * 2;

// Sequence b appends packed rows [cu[b], cu[b + 1]) to slot idx[b].
inline cudaError_t Prefill(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                           const uint16_t* g, const uint16_t* beta, const float* a_log,
                           const float* dt_bias, uint16_t* pool, const int* idx, const int* cu,
                           uint16_t* out, uint16_t* key_rows, float* u_rows, float* prefix_rows,
                           int* cnt, uint64_t* stamp, float scale, float half_gate_scale_log2,
                           int batch, int num_heads, int num_slots, bool fresh,
                           cudaStream_t stream) {
  cudaError_t status = cudaFuncSetAttribute(
      PrefillKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kPrefillSmemBytes);
  if (status != cudaSuccess) return status;
  PrefillKernel<<<batch * num_heads, NTHR, kPrefillSmemBytes, stream>>>(
      q, k, v, g, beta, a_log, dt_bias, pool, idx, cu, out, key_rows, u_rows, prefix_rows, cnt,
      stamp, scale, half_gate_scale_log2, num_heads, num_slots, fresh ? 1 : 0);
  return cudaGetLastError();
}

inline cudaError_t DecodeStep(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                              const uint16_t* g, const uint16_t* beta, const float* a_log,
                              const float* dt_bias, uint16_t* pool, const int* idx, uint16_t* out,
                              uint16_t* key_rows, float* u_rows, float* prefix_rows, int* cnt,
                              uint64_t* stamp, float scale, float half_gate_scale_log2, int batch,
                              int num_heads, int num_slots, cudaStream_t stream) {
  cudaError_t status = cudaFuncSetAttribute(
      DecodeKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kDecodeSmemBytes);
  if (status != cudaSuccess) return status;
  DecodeKernel<<<batch * num_heads, NTHR, kDecodeSmemBytes, stream>>>(
      q, k, v, g, beta, a_log, dt_bias, pool, idx, out, key_rows, u_rows, prefix_rows, cnt, stamp,
      scale, half_gate_scale_log2, num_heads, num_slots);
  return cudaGetLastError();
}

}  // namespace bi_class_s
}  // namespace kda
}  // namespace flashinfer

#endif  // FLASHINFER_KDA_BATCH_INVARIANT_CLASS_S_CUH_
