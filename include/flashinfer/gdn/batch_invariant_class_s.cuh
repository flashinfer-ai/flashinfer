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
#ifndef FLASHINFER_GDN_BATCH_INVARIANT_CLASS_S_CUH_
#define FLASHINFER_GDN_BATCH_INVARIANT_CLASS_S_CUH_

// Kernels of the class gdn.bf16.s16; build without fast math and with -fmad=false.

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace flashinfer {
namespace gdn {
namespace bi_class_s {

constexpr int C = 16;
constexpr int D = 128;
constexpr int NWARP = 8;
constexpr int NTHR = 256;
// operand tiles use row strides padded by 8 elements so a fragment load hits 32 distinct banks
constexpr int SD = D + 8;
constexpr uint16_t kNaN = 0x7fc0;

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

__device__ __forceinline__ float bf2f(uint32_t x) { return __uint_as_float((x & 0xffffu) << 16); }
// two fp32 -> one packed bf16 pair (lo | hi << 16), round to nearest even
__device__ __forceinline__ uint32_t f2bf_x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ uint32_t f2bf(float f) { return f2bf_x2(f, 0.0f) & 0xffffu; }
__host__ __device__ __forceinline__ size_t rup16(size_t x) { return (x + 15) & ~(size_t)15; }

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

// l2-norm of 128 bf16 values by one warp: 4 squares per lane in order, then an xor butterfly.
__device__ __forceinline__ uint64_t warp_norm(uint64_t raw, float post) {
  const float x0 = bf2f((uint32_t)raw), x1 = bf2f((uint32_t)(raw >> 16)),
              x2 = bf2f((uint32_t)(raw >> 32)), x3 = bf2f((uint32_t)(raw >> 48));
  float s = x0 * x0;
  s = s + x1 * x1;
  s = s + x2 * x2;
  s = s + x3 * x3;
  for (int off = 16; off > 0; off >>= 1) s = s + __shfl_xor_sync(0xffffffffu, s, off);
  const float r = (1.0f / sqrtf(s + 1e-6f)) * post;
  return (uint64_t)f2bf_x2(x0 * r, x1 * r) | ((uint64_t)f2bf_x2(x2 * r, x3 * r) << 32);
}

// k and q normalised together, each as warp_norm computes it.
__device__ __forceinline__ void warp_norm2(uint64_t kr, uint64_t qr, float scale, uint64_t& ko,
                                           uint64_t& qo) {
  const float k0 = bf2f((uint32_t)kr), k1 = bf2f((uint32_t)(kr >> 16)),
              k2 = bf2f((uint32_t)(kr >> 32)), k3 = bf2f((uint32_t)(kr >> 48));
  const float q0 = bf2f((uint32_t)qr), q1 = bf2f((uint32_t)(qr >> 16)),
              q2 = bf2f((uint32_t)(qr >> 32)), q3 = bf2f((uint32_t)(qr >> 48));
  float sk = k0 * k0, sq = q0 * q0;
  sk = sk + k1 * k1;
  sq = sq + q1 * q1;
  sk = sk + k2 * k2;
  sq = sq + q2 * q2;
  sk = sk + k3 * k3;
  sq = sq + q3 * q3;
  for (int off = 16; off > 0; off >>= 1) {
    const float tk = __shfl_xor_sync(0xffffffffu, sk, off),
                tq = __shfl_xor_sync(0xffffffffu, sq, off);
    sk = sk + tk;
    sq = sq + tq;
  }
  const float rk = (1.0f / sqrtf(sk + 1e-6f)) * 1.0f;
  const float rq = (1.0f / sqrtf(sq + 1e-6f)) * scale;
  ko = (uint64_t)f2bf_x2(k0 * rk, k1 * rk) | ((uint64_t)f2bf_x2(k2 * rk, k3 * rk) << 32);
  qo = (uint64_t)f2bf_x2(q0 * rq, q1 * rq) | ((uint64_t)f2bf_x2(q2 * rq, q3 * rq) << 32);
}

__device__ __forceinline__ void gate(uint32_t araw, uint32_t braw, float alog, float dtb, float& gl,
                                     float& beta) {
  const float x = bf2f(araw) + dtb;
  const float sp = x > 20.0f ? x : log1pf(expf(x));
  gl = -expf(alog) * sp;
  beta = 1.0f / (1.0f + expf(-bf2f(braw)));
}

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
// A fragments of a 16-row group for 8 k-steps from a shared tile of row stride `stride`.
__device__ __forceinline__ void load_a(AReg& r, const uint16_t* base, int stride, int lane) {
  const uint16_t* p =
      base + (size_t)((lane & 7) + 8 * ((lane >> 3) & 1)) * stride + 8 * (lane >> 4);
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
    ldsm_x4(r.a[4 * ks], r.a[4 * ks + 1], r.a[4 * ks + 2], r.a[4 * ks + 3], p + 16 * ks);
}
// B fragments for 8 k-steps: row = this lane's column vector (column lane & 7 of the group), k = 0.
__device__ __forceinline__ void load_b(BReg& r, const uint16_t* row, int lane) {
  const uint16_t* p = row + 8 * ((lane >> 3) & 1) + 16 * (lane >> 4);
#pragma unroll
  for (int x = 0; x < 4; ++x)
    ldsm_x4(r.b[4 * x], r.b[4 * x + 1], r.b[4 * x + 2], r.b[4 * x + 3], p + 32 * x);
}
// 128-term bf16 dot fold: 8 mma steps, ascending k, accumulator from +0.
__device__ __forceinline__ Acc4 mma128(const AReg& A, const BReg& B) {
  Acc4 d = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int ks = 0; ks < 8; ++ks)
    mma_bf16(d, A.a[4 * ks], A.a[4 * ks + 1], A.a[4 * ks + 2], A.a[4 * ks + 3], B.b[2 * ks],
             B.b[2 * ks + 1]);
  return d;
}

// Fold operand A (scaled updates ut[j][v], row stride sv) of one 16-row group, transposing load.
struct FoldA {
  uint32_t a[C / 4];
};
__device__ __forceinline__ void fold_a(FoldA& r, const uint16_t* ut, int sv, int v16, int lane) {
  const uint16_t* pa =
      ut + (size_t)((lane & 7) + 8 * (lane >> 4)) * sv + v16 + 8 * ((lane >> 3) & 1);
#pragma unroll
  for (int js = 0; js < C / 16; ++js)
    ldsm_x4t(r.a[4 * js], r.a[4 * js + 1], r.a[4 * js + 2], r.a[4 * js + 3],
             pa + (size_t)(16 * js) * sv);
}
// One 8-column group of the fold: acc = d * S, one mma step over the chunk rows, bf16 round.
__device__ __forceinline__ void fold_tile(uint32_t x, uint32_t y, float dC, const FoldA& A,
                                          const uint16_t* kcol, int lane, uint32_t& xo,
                                          uint32_t& yo) {
  Acc4 d;
  d.c[0] = dC * __uint_as_float(x << 16);
  d.c[1] = dC * __uint_as_float(x & 0xffff0000u);
  d.c[2] = dC * __uint_as_float(y << 16);
  d.c[3] = dC * __uint_as_float(y & 0xffff0000u);
  const uint16_t* pb = kcol + (size_t)((lane & 7) + 8 * ((lane >> 3) & 1)) * SD;
  uint32_t b[C / 8];
#pragma unroll
  for (int js = 0; js < C / 16; ++js)
    ldsm_x2t(b[2 * js], b[2 * js + 1], pb + (size_t)(16 * js) * SD);
#pragma unroll
  for (int js = 0; js < C / 16; ++js)
    mma_bf16(d, A.a[4 * js], A.a[4 * js + 1], A.a[4 * js + 2], A.a[4 * js + 3], b[2 * js],
             b[2 * js + 1]);
  xo = f2bf_x2(d.c[0], d.c[1]);
  yo = f2bf_x2(d.c[2], d.c[3]);
}

__global__ void __launch_bounds__(NTHR)
    PrefillKernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                  const uint16_t* __restrict__ v, const uint16_t* __restrict__ ga,
                  const uint16_t* __restrict__ gb, const float* __restrict__ A_log,
                  const float* __restrict__ dt_bias, uint16_t* pool, const int* __restrict__ idx,
                  const int* __restrict__ cu, uint16_t* __restrict__ out, uint16_t* Kh, float* Uh,
                  float* Lh, int* cnt, float scale, int H, int HV, int NT, int mmax, int num_slots,
                  const int* __restrict__ prefix, int reset, int tokens) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int VT = D / NT, R = VT / 16;
  int bi = blockIdx.x;
  const int tile = bi % NT;
  bi /= NT;
  const int h = bi % HV, b = bi / HV;
  const int tok_begin = cu[b], Ttot = cu[b + 1] - cu[b];
  if (Ttot <= 0) return;
  const int slot = idx[b];
  const int kh = h / (HV / H), v0 = tile * VT;
  bool bad = slot < 0 || slot >= num_slots || tok_begin < 0 || tok_begin + Ttot > tokens;
  bad |= mmax < C && Ttot > mmax;
  int* cnt_cell = cnt + ((size_t)(bad ? 0 : slot) * HV + h) * 8 + tile * R;
  const int c0 = bad ? 0 : cnt_cell[0];
  int n0 = c0 & 0xff;
  const int bank = (c0 >> 8) & 1;
  if (prefix != nullptr) {
    const int want = prefix[b] & (C - 1);
    if (want == 0) {
      n0 = 0;
    } else {
      bad |= want != n0;
    }
  } else if (reset) {
    n0 = 0;
  }
  // a bad slot, offset or prefix: NaN rows, slot untouched
  if (bad) {
    for (int e = tid; e < Ttot * VT; e += NTHR) {
      const long long row = (long long)tok_begin + e / VT;
      if (row >= 0 && row < tokens) out[((size_t)row * HV + h) * D + v0 + e % VT] = kNaN;
    }
    return;
  }
  const bool multi = (n0 + Ttot > C);

  size_t off = 0;
  uint16_t* Ssm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)VT * SD * 2;
  uint16_t* Ksm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)C * SD * 2;
  float* Usm = reinterpret_cast<float*>(smem + off);
  off += (size_t)C * VT * 4;
  float* Lsm = reinterpret_cast<float*>(smem + off);
  off += rup16(C * 4);
  float* wsm = reinterpret_cast<float*>(smem + off);
  off += rup16(C * 4);
  float* betasm = reinterpret_cast<float*>(smem + off);
  off += rup16(mmax * 4);
  float* dsm = reinterpret_cast<float*>(smem + off);
  off += rup16(mmax * 4);
  uint16_t* Kst = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)2 * mmax * D * 2;
  uint16_t* Qst = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)2 * mmax * SD * 2;
  uint16_t* Vst = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)2 * mmax * VT * 2;
  float* ABsm = reinterpret_cast<float*>(smem + off);
  off += (size_t)8 * mmax * VT;
  uint16_t* Utsm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)C * (VT + 8) * 2;
  float* CEsm = reinterpret_cast<float*>(smem + off);
  const int SV = VT + 8, nwr = NWARP / R, wrg = warp % R, wpart = warp / R;

  uint16_t* Spool = pool + ((size_t)slot * HV + h) * D * D + (size_t)v0 * D;
  uint16_t* Kpool = Kh + (((size_t)slot * 2 + bank) * H + kh) * C * D;
  float* Upool = Uh + (((size_t)slot * 2 + bank) * HV + h) * C * D + v0;
  float* Lpool = Lh + (((size_t)slot * 2 + bank) * HV + h) * C;
  uint16_t* const Sp = Ssm;

  uint32_t a_next = 0, b_next = 0;
  auto stage = [&](int buf, size_t tokA, int mm) {
    uint16_t* kd = Kst + (size_t)buf * mmax * D;
    uint16_t* qd = Qst + (size_t)buf * mmax * SD;
    uint16_t* vd = Vst + (size_t)buf * mmax * VT;
    for (int u = tid; u < mm * 16; u += NTHR) {
      const int row = u >> 4, c8 = (u & 15) * 8;
      cp_async16(kd + row * D + c8, k + ((tokA + row) * H + kh) * D + c8);
      cp_async16(qd + row * SD + c8, q + ((tokA + row) * H + kh) * D + c8);
    }
    const int vu = VT / 8;
    for (int u = tid; u < mm * vu; u += NTHR) {
      const int row = u / vu, c8 = (u % vu) * 8;
      cp_async16(vd + row * VT + c8, v + ((tokA + row) * HV + h) * D + v0 + c8);
    }
    cp_async_commit();
    if (tid < mm) {
      a_next = ga[(tokA + tid) * HV + h];
      b_next = gb[(tokA + tid) * HV + h];
    }
  };

  {
    const int m0 = (C - n0) < Ttot ? (C - n0) : Ttot;
    stage(0, (size_t)tok_begin, m0);
  }
  if (tid < n0) Lsm[tid] = Lpool[tid];
  for (int u = tid; u < VT * 16; u += NTHR)
    cp_async16(Ssm + (u >> 4) * SD + (u & 15) * 8, Spool + (size_t)u * 8);
  for (int u = tid; u < n0 * 16; u += NTHR)
    cp_async16(Ksm + (u >> 4) * SD + (u & 15) * 8, Kpool + (size_t)u * 8);
  cp_async_commit();
  for (int e = tid; e < n0 * VT; e += NTHR) Usm[e] = Upool[(e / VT) * D + (e % VT)];
  bool folded = false;

  int n = n0, t = 0, cur = 0;
  while (t < Ttot) {
    const int p0 = n;
    const int m = (C - p0) < (Ttot - t) ? (C - p0) : (Ttot - t);
    const size_t tok0 = (size_t)tok_begin + t;
    uint16_t* Kraw = Kst + (size_t)cur * mmax * D;
    uint16_t* Qsm = Qst + (size_t)cur * mmax * SD;
    const uint16_t* Vsm = Vst + (size_t)cur * mmax * VT;

    cp_async_wait_all();
    const uint32_t a_cur = a_next, b_cur = b_next;
    {
      const int rem = Ttot - t - m;
      const int mn = rem < C ? rem : C;
      if (mn > 0) stage(cur ^ 1, tok0 + m, mn);
    }
    if (tid < m) {
      float gl, beta;
      gate(a_cur, b_cur, A_log[h], dt_bias[h], gl, beta);
      Lsm[p0 + tid] = gl;
      betasm[tid] = beta;
    }
    __syncthreads();
    if (tid == 0) {
      float acc = p0 ? Lsm[p0 - 1] : 0.0f;
      for (int i = 0; i < m; ++i) {
        acc = (p0 + i) ? acc + Lsm[p0 + i] : Lsm[p0 + i];
        Lsm[p0 + i] = acc;
      }
    }
    for (int i = warp; i < m; i += NWARP) {
      const uint64_t kr = *reinterpret_cast<const uint64_t*>(Kraw + (size_t)i * D + 4 * lane);
      const uint64_t qr = *reinterpret_cast<const uint64_t*>(Qsm + (size_t)i * SD + 4 * lane);
      uint64_t ko, qo;
      warp_norm2(kr, qr, scale, ko, qo);
      *reinterpret_cast<uint64_t*>(Ksm + (size_t)(p0 + i) * SD + 4 * lane) = ko;
      *reinterpret_cast<uint64_t*>(Qsm + (size_t)i * SD + 4 * lane) = qo;
    }
    __syncthreads();
    if (tid < m) dsm[tid] = expf(Lsm[p0 + tid]);

    const int ncol = 2 * m, ncg = (ncol + 7) >> 3;
    {
      AReg A;
      load_a(A, Sp + (size_t)(16 * wrg) * SD, SD, lane);
      const int r = 16 * wrg + g;
      for (int cg = wpart; cg < ncg; cg += nwr) {
        int col = 8 * cg + (lane & 7);
        col = col < ncol ? col : ncol - 1;
        const uint16_t* bp = col < m ? Ksm + (size_t)(p0 + col) * SD : Qsm + (size_t)(col - m) * SD;
        BReg B;
        load_b(B, bp, lane);
        const Acc4 d = mma128(A, B);
        const int cA = 8 * cg + 2 * t4;
        if (cA < ncol) {
          ABsm[cA * VT + r] = d.c[0];
          ABsm[cA * VT + r + 8] = d.c[2];
        }
        if (cA + 1 < ncol) {
          ABsm[(cA + 1) * VT + r] = d.c[1];
          ABsm[(cA + 1) * VT + r + 8] = d.c[3];
        }
      }
    }
    const int njg = (p0 + m + 15) >> 4;
    for (int job = warp; job < njg * ncg; job += NWARP) {
      const int jg = job % njg, cg = job / njg;
      int col = 8 * cg + (lane & 7);
      col = col < ncol ? col : ncol - 1;
      const uint16_t* bp = col < m ? Ksm + (size_t)(p0 + col) * SD : Qsm + (size_t)(col - m) * SD;
      const int j = 16 * jg + g;
      AReg A;
      BReg B;
      load_a(A, Ksm + (size_t)(16 * jg) * SD, SD, lane);
      load_b(B, bp, lane);
      const Acc4 d = mma128(A, B);
      const int cA = 8 * cg + 2 * t4;
      if (cA < ncol) {
        const int i = cA < m ? cA : cA - m, w = cA >= m;
        CEsm[((size_t)j * mmax + i) * 2 + w] = d.c[0];
        CEsm[((size_t)(j + 8) * mmax + i) * 2 + w] = d.c[2];
      }
      if (cA + 1 < ncol) {
        const int i = (cA + 1) < m ? (cA + 1) : (cA + 1) - m, w = (cA + 1) >= m;
        CEsm[((size_t)j * mmax + i) * 2 + w] = d.c[1];
        CEsm[((size_t)(j + 8) * mmax + i) * 2 + w] = d.c[3];
      }
    }
    __syncthreads();

    for (int e = tid; e < m * C; e += NTHR) {
      const int i = e % m, j = e / m, p = p0 + i;
      if (j <= p) {
        const float w = expf(Lsm[p] - Lsm[j]);
        float* ce = CEsm + ((size_t)j * mmax + i) * 2;
        ce[0] = w * ce[0];
        ce[1] = w * ce[1];
      }
    }
    __syncthreads();

    if (tid < VT) {
      const int vv = tid;
      float* up = Usm + vv;
      if (p0 == 0 && m == C) {
        // right-looking form: every accumulator still receives its terms in ascending j
        for (int blk = 0; blk < C; blk += 16) {
          float pred[16], o[16];
#pragma unroll
          for (int x = 0; x < 16; ++x) {
            const float dd = dsm[blk + x];
            pred[x] = dd * ABsm[(blk + x) * VT + vv];
            o[x] = dd * ABsm[(C + blk + x) * VT + vv];
          }
          for (int j = 0; j < blk; ++j) {
            const float u = up[(size_t)j * VT];
            float2 c[16];
#pragma unroll
            for (int x = 0; x < 16; ++x)
              c[x] = *reinterpret_cast<const float2*>(CEsm + ((size_t)j * C + blk + x) * 2);
#pragma unroll
            for (int x = 0; x < 16; ++x) {
              pred[x] = __fmaf_rn(c[x].x, u, pred[x]);
              o[x] = __fmaf_rn(c[x].y, u, o[x]);
            }
          }
          float vn = bf2f(Vsm[blk * VT + vv]), bn = betasm[blk];
#pragma unroll
          for (int jj = 0; jj < 16; ++jj) {
            const int j = blk + jj;
            const float u = bn * (vn - pred[jj]);
            const float cjj = CEsm[((size_t)j * C + j) * 2 + 1];
            if (jj + 1 < 16) {
              vn = bf2f(Vsm[(j + 1) * VT + vv]);
              bn = betasm[j + 1];
            }
            up[(size_t)j * VT] = u;
            o[jj] = __fmaf_rn(cjj, u, o[jj]);
            out[((tok0 + j) * HV + h) * D + v0 + vv] = (uint16_t)f2bf(o[jj]);
#pragma unroll
            for (int i0 = jj + 1; i0 < 16; i0 += 8) {
              float2 c[8];
#pragma unroll
              for (int x = 0; x < 8; ++x)
                if (i0 + x < 16)
                  c[x] =
                      *reinterpret_cast<const float2*>(CEsm + ((size_t)j * C + blk + i0 + x) * 2);
#pragma unroll
              for (int x = 0; x < 8; ++x)
                if (i0 + x < 16) {
                  pred[i0 + x] = __fmaf_rn(c[x].x, u, pred[i0 + x]);
                  o[i0 + x] = __fmaf_rn(c[x].y, u, o[i0 + x]);
                }
            }
          }
        }
      } else {
        for (int i = 0; i < m; ++i) {
          const int p = p0 + i;
          const size_t tok = tok0 + i;
          const float dd = dsm[i];
          float pred = dd * ABsm[i * VT + vv];
          float o = dd * ABsm[(m + i) * VT + vv];
          for (int j = 0; j < p; ++j) {
            const float u = up[(size_t)j * VT];
            const float* ce = CEsm + ((size_t)j * mmax + i) * 2;
            pred = __fmaf_rn(ce[0], u, pred);
            o = __fmaf_rn(ce[1], u, o);
          }
          const float u = betasm[i] * (bf2f(Vsm[i * VT + vv]) - pred);
          up[(size_t)p * VT] = u;
          o = __fmaf_rn(CEsm[((size_t)p * mmax + i) * 2 + 1], u, o);
          out[(tok * HV + h) * D + v0 + vv] = (uint16_t)f2bf(o);
        }
      }
    }
    __syncthreads();

    t += m;
    n += m;
    cur ^= 1;
    if (n == C) {
      if (tid < C) wsm[tid] = expf(Lsm[C - 1] - Lsm[tid]);
      __syncthreads();
      for (int e = tid; e < VT * C; e += NTHR)
        Utsm[(size_t)(e / VT) * SV + (e % VT)] = (uint16_t)f2bf(wsm[e / VT] * Usm[e]);
      __syncthreads();
      const float dC = dsm[m - 1];
      {
        FoldA A;
        fold_a(A, Utsm, SV, 16 * wrg, lane);
        const int r = 16 * wrg + g;
        for (int kg = wpart; kg < 16; kg += nwr) {
          const int kc = 8 * kg + 2 * t4;
          uint32_t* slo = reinterpret_cast<uint32_t*>(Sp + (size_t)r * SD + kc);
          uint32_t* shi = reinterpret_cast<uint32_t*>(Sp + (size_t)(r + 8) * SD + kc);
          uint32_t xo, yo;
          fold_tile(*slo, *shi, dC, A, Ksm + 8 * kg, lane, xo, yo);
          *slo = xo;
          *shi = yo;
        }
      }
      __syncthreads();
      n = 0;
      folded = true;
    }
  }

  const int nb = multi ? (bank ^ 1) : bank;
  const int w0 = multi ? 0 : n0;
  if (folded)
    for (int u = tid; u < VT * 16; u += NTHR)
      *reinterpret_cast<uint4*>(Spool + (size_t)u * 8) =
          *reinterpret_cast<const uint4*>(Ssm + (u >> 4) * SD + (u & 15) * 8);
  if (n > w0) {
    uint16_t* Ko = Kh + (((size_t)slot * 2 + nb) * H + kh) * C * D;
    float* Uo = Uh + (((size_t)slot * 2 + nb) * HV + h) * C * D + v0;
    float* Lo = Lh + (((size_t)slot * 2 + nb) * HV + h) * C;
    for (int u = tid + w0 * 16; u < n * 16; u += NTHR)
      *reinterpret_cast<uint4*>(Ko + (size_t)u * 8) =
          *reinterpret_cast<const uint4*>(Ksm + (u >> 4) * SD + (u & 15) * 8);
    for (int e = tid + w0 * VT; e < n * VT; e += NTHR) Uo[(e / VT) * D + (e % VT)] = Usm[e];
    if (tid >= w0 && tid < n) Lo[tid] = Lsm[tid];
  }
  if (tid < R) cnt_cell[tid] = n | (nb << 8);
}

// Decode: one CTA per (sequence, v-head); EARLY orders the work for launches of few blocks per SM.
template <bool EARLY>
__global__ void __launch_bounds__(NTHR)
    DecodeKernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                 const uint16_t* __restrict__ v, const uint16_t* __restrict__ ga,
                 const uint16_t* __restrict__ gb, const float* __restrict__ A_log,
                 const float* __restrict__ dt_bias, uint16_t* pool, const int* __restrict__ idx,
                 uint16_t* __restrict__ out, uint16_t* Kh, float* Uh, float* Lh, int* cnt,
                 float scale, int H, int HV, int num_slots) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int h = blockIdx.x % HV, b = blockIdx.x / HV;
  const int slot = idx[b];
  const size_t tok = b;
  const int kh = h / (HV / H);
  uint64_t rawkq = 0, nrm = 0;
  uint32_t vraw = 0, araw = 0, braw = 0;
  float gate_l = 0.0f, gate_b = 0.0f;
  auto load_rows = [&]() {
    if (warp == 0) rawkq = *reinterpret_cast<const uint64_t*>(k + (tok * H + kh) * D + 4 * lane);
    if (warp == 1) rawkq = *reinterpret_cast<const uint64_t*>(q + (tok * H + kh) * D + 4 * lane);
    if (tid < D) vraw = v[(tok * HV + h) * D + tid];
    if (tid == 0) {
      araw = ga[tok * HV + h];
      braw = gb[tok * HV + h];
    }
  };
  auto norm_and_gate = [&]() {
    if (warp < 2) nrm = warp_norm(rawkq, warp ? scale : 1.0f);
    if (tid == 0) gate(araw, braw, A_log[h], dt_bias[h], gate_l, gate_b);
  };
  // EARLY: what does not depend on the slot runs while the slot index arrives
  if (EARLY) {
    load_rows();
    norm_and_gate();
  }
  if (slot < 0 || slot >= num_slots) {
    if (tid < D) out[(tok * HV + h) * D + tid] = kNaN;
    return;
  }
  int* cnt_cell = cnt + ((size_t)slot * HV + h) * 8;
  const int c0 = cnt_cell[0];

  size_t off = 0;
  uint16_t* Ssm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)D * SD * 2;
  uint16_t* Ksm = reinterpret_cast<uint16_t*>(smem + off);
  off += (size_t)C * SD * 2;
  float* Usm = reinterpret_cast<float*>(smem + off);
  off += (size_t)C * D * 4;
  float* Lsm = reinterpret_cast<float*>(smem + off);
  off += rup16(C * 4);
  float* wsm = reinterpret_cast<float*>(smem + off);
  off += rup16(C * 4);
  uint16_t* Qsm = reinterpret_cast<uint16_t*>(smem + off);
  off += SD * 2;
  float* misc = reinterpret_cast<float*>(smem + off);
  off += 16;
  float* ABsm = reinterpret_cast<float*>(smem + off);
  off += 2 * D * 4;
  float* CEsm = reinterpret_cast<float*>(smem + off);
  off += 2 * C * 4;
  uint16_t* Utsm = reinterpret_cast<uint16_t*>(smem + off);

  uint16_t* Spool = pool + ((size_t)slot * HV + h) * D * D;
  for (int u = tid; u < D * 16; u += NTHR)
    cp_async16(Ssm + (u >> 4) * SD + (u & 15) * 8, Spool + (size_t)u * 8);
  if (!EARLY) load_rows();
  const int p = c0 & 0xff, bank = (c0 >> 8) & 1;
  uint16_t* Kpool = Kh + (((size_t)slot * 2 + bank) * H + kh) * C * D;
  float* Upool = Uh + (((size_t)slot * 2 + bank) * HV + h) * C * D;
  float* Lpool = Lh + (((size_t)slot * 2 + bank) * HV + h) * C;
  float lj = 0.0f;
  if (tid < p) lj = Lpool[tid];
  for (int u = tid; u < p * 16; u += NTHR)
    cp_async16(Ksm + (u >> 4) * SD + (u & 15) * 8, Kpool + (size_t)u * 8);
  for (int u = tid; u < p * 32; u += NTHR) cp_async16(Usm + (size_t)u * 4, Upool + (size_t)u * 4);
  cp_async_commit();
  if (tid < p) Lsm[tid] = lj;
  if (!EARLY) norm_and_gate();
  if (warp == 0) {
    *reinterpret_cast<uint64_t*>(Ksm + (size_t)p * SD + 4 * lane) = nrm;
    *reinterpret_cast<uint64_t*>(Kpool + (size_t)p * D + 4 * lane) = nrm;
  }
  if (warp == 1) *reinterpret_cast<uint64_t*>(Qsm + 4 * lane) = nrm;
  if (tid == 0) {
    misc[0] = gate_l;
    misc[1] = gate_b;
  }
  cp_async_wait_all();
  __syncthreads();

  float w = 0.0f;
  if (EARLY) {
    // the tensor-core folds are issued first; the exponentials run while they complete
    BReg B;
    load_b(B, (lane & 7) == 0 ? Ksm + (size_t)p * SD : Qsm, lane);
    AReg A, A2;
    load_a(A, Ssm + (size_t)(16 * warp) * SD, SD, lane);
    if (warp == 0) load_a(A2, Ksm, SD, lane);
    const Acc4 d = mma128(A, B);
    Acc4 d2 = {{0.f, 0.f, 0.f, 0.f}};
    if (warp == 0) d2 = mma128(A2, B);
    const float gl = misc[0];
    const float Lp = p ? Lsm[p - 1] + gl : gl;
    if (tid <= p) {
      const float Lj = (tid == p) ? Lp : lj;
      w = expf(Lp - Lj);
      wsm[tid] = w;
    }
    if (tid == 0) {
      Lsm[p] = Lp;
      Lpool[p] = Lp;
      misc[2] = expf(Lp);
    }
    if (t4 == 0) {
      const int r = 16 * warp + g;
      ABsm[r] = d.c[0];
      ABsm[D + r] = d.c[1];
      ABsm[r + 8] = d.c[2];
      ABsm[D + r + 8] = d.c[3];
    }
    // the open chunk's Gram row is written and scaled in warp 0 alone
    if (warp == 0 && t4 == 0) {
      CEsm[2 * g] = d2.c[0];
      CEsm[2 * g + 1] = d2.c[1];
      CEsm[2 * (g + 8)] = d2.c[2];
      CEsm[2 * (g + 8) + 1] = d2.c[3];
    }
    __syncwarp();
    if (tid <= p) {
      CEsm[2 * tid] = w * CEsm[2 * tid];
      CEsm[2 * tid + 1] = w * CEsm[2 * tid + 1];
    }
    __syncthreads();
  } else {
    const float gl = misc[0];
    const float Lp = p ? Lsm[p - 1] + gl : gl;
    if (tid <= p) {
      const float Lj = (tid == p) ? Lp : lj;
      w = expf(Lp - Lj);
      wsm[tid] = w;
    }
    if (tid == 0) {
      Lsm[p] = Lp;
      Lpool[p] = Lp;
      misc[2] = expf(Lp);
    }
    BReg B;
    load_b(B, (lane & 7) == 0 ? Ksm + (size_t)p * SD : Qsm, lane);
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
      AReg A2;
      load_a(A2, Ksm, SD, lane);
      const Acc4 d = mma128(A2, B);
      if (t4 == 0) {
        CEsm[2 * g] = d.c[0];
        CEsm[2 * g + 1] = d.c[1];
        CEsm[2 * (g + 8)] = d.c[2];
        CEsm[2 * (g + 8) + 1] = d.c[3];
      }
    }
    __syncwarp();
    if (tid <= p) {
      CEsm[2 * tid] = w * CEsm[2 * tid];
      CEsm[2 * tid + 1] = w * CEsm[2 * tid + 1];
    }
    __syncthreads();
  }
  const float dd = misc[2];
  if (tid < D) {
    float pred = dd * ABsm[tid];
    float o = dd * ABsm[D + tid];
    const float* up = Usm + tid;
    for (int j = 0; j < p; ++j) {
      const float u = up[(size_t)j * D];
      pred = __fmaf_rn(CEsm[2 * j], u, pred);
      o = __fmaf_rn(CEsm[2 * j + 1], u, o);
    }
    const float u = misc[1] * (bf2f(vraw) - pred);
    Usm[(size_t)p * D + tid] = u;
    Upool[(size_t)p * D + tid] = u;
    o = __fmaf_rn(CEsm[2 * p + 1], u, o);
    out[(tok * HV + h) * D + tid] = (uint16_t)f2bf(o);
  }
  if (tid < 8) cnt_cell[tid] = ((p + 1 == C) ? 0 : p + 1) | (bank << 8);
  if (p + 1 == C) {
    __syncthreads();
    for (int e = tid; e < D * C; e += NTHR)
      Utsm[(size_t)(e / D) * SD + (e % D)] = (uint16_t)f2bf(wsm[e / D] * Usm[e]);
    __syncthreads();
    const int r = 16 * warp + g;
    FoldA FA;
    fold_a(FA, Utsm, SD, 16 * warp, lane);
#pragma unroll
    for (int kg = 0; kg < 16; ++kg) {
      const int kc = 8 * kg + 2 * t4;
      uint32_t* slo = reinterpret_cast<uint32_t*>(Ssm + (size_t)r * SD + kc);
      uint32_t* shi = reinterpret_cast<uint32_t*>(Ssm + (size_t)(r + 8) * SD + kc);
      uint32_t xo, yo;
      fold_tile(*slo, *shi, dd, FA, Ksm + 8 * kg, lane, xo, yo);
      *slo = xo;
      *shi = yo;
    }
    __syncthreads();
    for (int u = tid; u < D * 16; u += NTHR)
      *reinterpret_cast<uint4*>(Spool + (size_t)u * 8) =
          *reinterpret_cast<const uint4*>(Ssm + (u >> 4) * SD + (u & 15) * 8);
  }
}

constexpr int SAB = 4;
constexpr int NTHR_CHAIN = 512;
constexpr int CHAIN_AHEAD = 4;
constexpr int ST = 20;
constexpr int SU = 24;

__device__ __forceinline__ void cp_async_wait_group1() {
  asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}

// Four independent 128-term folds on one A operand, interleaved.
__device__ __forceinline__ void mma128x4(const AReg& A, const BReg& B0, const BReg& B1,
                                         const BReg& B2, const BReg& B3, Acc4& d0, Acc4& d1,
                                         Acc4& d2, Acc4& d3) {
  d0 = {{0.f, 0.f, 0.f, 0.f}};
  d1 = d0;
  d2 = d0;
  d3 = d0;
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    const uint32_t a0 = A.a[4 * ks], a1 = A.a[4 * ks + 1], a2 = A.a[4 * ks + 2],
                   a3 = A.a[4 * ks + 3];
    mma_bf16(d0, a0, a1, a2, a3, B0.b[2 * ks], B0.b[2 * ks + 1]);
    mma_bf16(d1, a0, a1, a2, a3, B1.b[2 * ks], B1.b[2 * ks + 1]);
    mma_bf16(d2, a0, a1, a2, a3, B2.b[2 * ks], B2.b[2 * ks + 1]);
    mma_bf16(d3, a0, a1, a2, a3, B3.b[2 * ks], B3.b[2 * ks + 1]);
  }
}

// Prefill of long runs: nmw chain warps own 16 state rows each, eight helper warps work ahead.
__global__ void __launch_bounds__(NTHR_CHAIN, 1)
    ChainKernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                const uint16_t* __restrict__ v, const uint16_t* __restrict__ ga,
                const uint16_t* __restrict__ gb, const float* __restrict__ A_log,
                const float* __restrict__ dt_bias, uint16_t* pool, const int* __restrict__ idx,
                const int* __restrict__ cu, uint16_t* __restrict__ out, uint16_t* Kh, float* Uh,
                float* Lh, int* cnt, float scale, int H, int HV, int num_slots,
                const int* __restrict__ prefix, int reset, int NT, int tokens) {
  extern __shared__ __align__(16) unsigned char smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t4 = lane & 3;
  const int nmw = 8 / NT, mt = 32 * nmw, nthr = mt + 256, VT = 16 * nmw;
  const bool ism = warp < nmw;
  const int hw = warp - nmw, ht = tid - mt;
  int bi = blockIdx.x;
  const int v0 = (bi % NT) * VT;
  bi /= NT;
  const int h = bi % HV, b = bi / HV;
  const int tok_begin = cu[b], Ttot = cu[b + 1] - cu[b];
  if (Ttot <= 0) return;
  const int slot = idx[b];
  const int kh = h / (HV / H);
  bool bad = slot < 0 || slot >= num_slots || tok_begin < 0 || tok_begin + Ttot > tokens;
  int* cnt_cell = cnt + ((size_t)(bad ? 0 : slot) * HV + h) * 8 + (v0 >> 4);
  const int c0 = bad ? 0 : cnt_cell[0];
  int n0 = c0 & 0xff;
  const int bank = (c0 >> 8) & 1;
  if (prefix != nullptr) {
    const int want = prefix[b] & (C - 1);
    if (want == 0) {
      n0 = 0;
    } else {
      bad |= want != n0;
    }
  } else if (reset) {
    n0 = 0;
  }
  if (bad) {
    for (int e = tid; e < Ttot * VT; e += nthr) {
      const long long row = (long long)tok_begin + e / VT;
      if (row >= 0 && row < tokens) out[((size_t)row * HV + h) * D + v0 + e % VT] = kNaN;
    }
    return;
  }
  const bool multi = (n0 + Ttot > C);
  const int SV = VT + 8, SA = VT + SAB;

  size_t off = 0;
  auto take = [&](size_t bytes) {
    unsigned char* p = smem + off;
    off += bytes;
    return p;
  };
  uint16_t* Ssm = reinterpret_cast<uint16_t*>(take((size_t)VT * SD * 2));
  uint16_t* Kt = reinterpret_cast<uint16_t*>(take((size_t)CHAIN_AHEAD * C * SD * 2));
  uint16_t* Qt = reinterpret_cast<uint16_t*>(take((size_t)CHAIN_AHEAD * C * SD * 2));
  uint16_t* Vt = reinterpret_cast<uint16_t*>(take((size_t)CHAIN_AHEAD * C * VT * 2));
  float* CEt = reinterpret_cast<float*>(take((size_t)CHAIN_AHEAD * C * C * 8));
  float* Gt = reinterpret_cast<float*>(take((size_t)CHAIN_AHEAD * 4 * C * 4));
  uint16_t* Kg = reinterpret_cast<uint16_t*>(take((size_t)C * SD * 2));
  uint16_t* Qg = reinterpret_cast<uint16_t*>(take((size_t)C * SD * 2));
  uint16_t* Kraw = reinterpret_cast<uint16_t*>(take((size_t)C * D * 2));
  uint16_t* Vg = reinterpret_cast<uint16_t*>(take((size_t)C * VT * 2));
  float* CEg = reinterpret_cast<float*>(take((size_t)C * C * 8));
  float* Usm = reinterpret_cast<float*>(take((size_t)C * VT * 4));
  float* Lsm = reinterpret_cast<float*>(take(C * 4));
  float* wsm = reinterpret_cast<float*>(take(C * 4));
  float* betasm = reinterpret_cast<float*>(take(C * 4));
  float* dsm = reinterpret_cast<float*>(take(C * 4));
  // the long-run loop keeps both per row: [half][row][p] (stride ST) and [row][j] (stride SU)
  float* ABsm = reinterpret_cast<float*>(take((size_t)2 * VT * ST * 4));
  uint16_t* Utsm = reinterpret_cast<uint16_t*>(take((size_t)VT * SU * 2));

  uint16_t* Spool = pool + ((size_t)slot * HV + h) * D * D + (size_t)v0 * D;
  uint16_t* Kpool = Kh + (((size_t)slot * 2 + bank) * H + kh) * C * D;
  float* Upool = Uh + (((size_t)slot * 2 + bank) * HV + h) * C * D + v0;
  float* Lpool = Lh + (((size_t)slot * 2 + bank) * HV + h) * C;

  if (ism) {
    if (tid < n0) Lsm[tid] = Lpool[tid];
    for (int u = tid; u < VT * 16; u += mt)
      cp_async16(Ssm + (u >> 4) * SD + (u & 15) * 8, Spool + (size_t)u * 8);
    for (int u = tid; u < n0 * 16; u += mt)
      cp_async16(Kg + (u >> 4) * SD + (u & 15) * 8, Kpool + (size_t)u * 8);
    cp_async_commit();
    for (int e = tid; e < n0 * VT; e += mt) Usm[e] = Upool[(e / VT) * D + (e % VT)];
    cp_async_wait_all();
  }
  __syncthreads();
  AReg S;
  if (ism) load_a(S, Ssm + (size_t)(16 * warp) * SD, SD, lane);
  bool folded = false;
  const int r = 16 * warp + g;

  // a chunk with history or an open tail: rows [tok0, tok0 + m) at chunk rows p0 ..
  auto generic = [&](int p0, int m, size_t tok0) {
    uint32_t araw = 0, braw = 0;
    if (ism) {
      for (int u = tid; u < m * 16; u += mt) {
        const int row = u >> 4, c8 = (u & 15) * 8;
        cp_async16(Kraw + row * D + c8, k + ((tok0 + row) * H + kh) * D + c8);
        cp_async16(Qg + row * SD + c8, q + ((tok0 + row) * H + kh) * D + c8);
      }
      const int vu = VT / 8;
      for (int u = tid; u < m * vu; u += mt) {
        const int row = u / vu, c8 = (u % vu) * 8;
        cp_async16(Vg + row * VT + c8, v + ((tok0 + row) * HV + h) * D + v0 + c8);
      }
      cp_async_commit();
      if (tid < m) {
        araw = ga[(tok0 + tid) * HV + h];
        braw = gb[(tok0 + tid) * HV + h];
      }
      cp_async_wait_all();
      if (tid < m) {
        float gl, beta;
        gate(araw, braw, A_log[h], dt_bias[h], gl, beta);
        Lsm[p0 + tid] = gl;
        betasm[tid] = beta;
      }
    }
    __syncthreads();
    if (tid == 0) {
      float acc = p0 ? Lsm[p0 - 1] : 0.0f;
      for (int i = 0; i < m; ++i) {
        acc = (p0 + i) ? acc + Lsm[p0 + i] : Lsm[p0 + i];
        Lsm[p0 + i] = acc;
      }
    }
    if (ism) {
      for (int i = warp; i < m; i += nmw) {
        const uint64_t kr = *reinterpret_cast<const uint64_t*>(Kraw + (size_t)i * D + 4 * lane);
        const uint64_t qr = *reinterpret_cast<const uint64_t*>(Qg + (size_t)i * SD + 4 * lane);
        uint64_t ko, qo;
        warp_norm2(kr, qr, scale, ko, qo);
        *reinterpret_cast<uint64_t*>(Kg + (size_t)(p0 + i) * SD + 4 * lane) = ko;
        *reinterpret_cast<uint64_t*>(Qg + (size_t)i * SD + 4 * lane) = qo;
      }
    }
    __syncthreads();
    const int ncol = 2 * m, ncg = (ncol + 7) >> 3;
    if (ism) {
      if (tid < m) dsm[tid] = expf(Lsm[p0 + tid]);
      for (int cg = 0; cg < ncg; ++cg) {
        int col = 8 * cg + (lane & 7);
        col = col < ncol ? col : ncol - 1;
        const uint16_t* bp = col < m ? Kg + (size_t)(p0 + col) * SD : Qg + (size_t)(col - m) * SD;
        BReg B;
        load_b(B, bp, lane);
        const Acc4 d = mma128(S, B);
        const int cA = 8 * cg + 2 * t4;
        if (cA < ncol) {
          ABsm[cA * SA + r] = d.c[0];
          ABsm[cA * SA + r + 8] = d.c[2];
        }
        if (cA + 1 < ncol) {
          ABsm[(cA + 1) * SA + r] = d.c[1];
          ABsm[(cA + 1) * SA + r + 8] = d.c[3];
        }
      }
    }
    if (!ism && hw < ncg) {
      {
        const int cg = hw;
        int col = 8 * cg + (lane & 7);
        col = col < ncol ? col : ncol - 1;
        const uint16_t* bp = col < m ? Kg + (size_t)(p0 + col) * SD : Qg + (size_t)(col - m) * SD;
        AReg A;
        BReg B;
        load_a(A, Kg, SD, lane);
        load_b(B, bp, lane);
        const Acc4 d = mma128(A, B);
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          const int j = g + 8 * (x >> 1), cA = 8 * cg + 2 * t4 + (x & 1);
          const int w = cA >= m, p = p0 + (w ? cA - m : cA);
          if (cA < ncol && j <= p) CEg[(j * C + (p - j)) * 2 + w] = d.c[x];
        }
      }
    }
    __syncthreads();
    if (!ism) {
      const int j = ht / C, p = j + ht % C;
      if (p >= p0 && p < p0 + m) {
        const float w = expf(Lsm[p] - Lsm[j]);
        CEg[2 * ht] = w * CEg[2 * ht];
        CEg[2 * ht + 1] = w * CEg[2 * ht + 1];
      }
    }
    if (p0 + m == C && tid < C) wsm[tid] = expf(Lsm[C - 1] - Lsm[tid]);
    __syncthreads();
    if (tid < VT) {
      float* up = Usm + tid;
      for (int i = 0; i < m; ++i) {
        const int p = p0 + i;
        const float dd = dsm[i];
        float pred = dd * ABsm[i * SA + tid];
        float o = dd * ABsm[(m + i) * SA + tid];
        for (int j = 0; j < p; ++j) {
          const float u = up[(size_t)j * VT];
          const float* ce = CEg + ((size_t)j * C + (p - j)) * 2;
          pred = __fmaf_rn(ce[0], u, pred);
          o = __fmaf_rn(ce[1], u, o);
        }
        const float u = betasm[i] * (bf2f(Vg[i * VT + tid]) - pred);
        up[(size_t)p * VT] = u;
        o = __fmaf_rn(CEg[((size_t)p * C) * 2 + 1], u, o);
        out[((tok0 + i) * HV + h) * D + v0 + tid] = (uint16_t)f2bf(o);
      }
    }
    __syncthreads();
    if (p0 + m == C) {
      for (int e = tid; e < VT * C; e += nthr)
        Utsm[(size_t)(e / VT) * SV + (e % VT)] = (uint16_t)f2bf(wsm[e / VT] * Usm[e]);
      __syncthreads();
      if (ism) {
        const float dC = dsm[m - 1];
        FoldA FA;
        fold_a(FA, Utsm, SV, 16 * warp, lane);
#pragma unroll
        for (int kg = 0; kg < 16; ++kg)
          fold_tile(S.a[2 * kg], S.a[2 * kg + 1], dC, FA, Kg + 8 * kg, lane, S.a[2 * kg],
                    S.a[2 * kg + 1]);
      }
      __syncthreads();
      folded = true;
    }
  };

  int n = n0, t = 0;
  if (n0 > 0) {
    const int m = (C - n0) < Ttot ? (C - n0) : Ttot;
    generic(n0, m, (size_t)tok_begin);
    t = m;
    n = (n0 + m) & (C - 1);
  }
  const int nwhole = (Ttot - t) / C;
  const size_t tokw = (size_t)tok_begin + t;
  if (nwhole > 0) {
    // helper state carried across iterations: raw rows and gate inputs of the chunk normalised next
    uint64_t raw[4] = {0, 0, 0, 0};
    uint32_t araw = 0, braw = 0;
    const float post = hw >= 4 ? scale : 1.0f;
    for (int it = -CHAIN_AHEAD; it < nwhole; ++it) {
      if (!ism) {
        // tables of chunk it + 1 from its Gram rows and its log-decay prefix
        if (it + 1 >= 0 && it + 1 < nwhole) {
          float* CE = CEt + (size_t)((it + 1) & 3) * C * C * 2;
          float* G = Gt + (size_t)((it + 1) & 3) * 4 * C;
          const int j = ht / C, p = j + ht % C;
          if (p < C) {
            const float w = expf(G[p] - G[j]);
            CE[ht] = w * CE[ht];
            CE[C * C + ht] = w * CE[C * C + ht];
          }
          if (ht < C) G[C + ht] = expf(G[ht]);
          if (ht >= 32 && ht < 32 + C) G[2 * C + 2 * (ht - 32) + 1] = expf(G[C - 1] - G[ht - 32]);
        }
        // Gram rows and log-decay prefix of chunk it + 2
        if (it + 2 >= 0 && it + 2 < nwhole) {
          const uint16_t* Kc = Kt + (size_t)((it + 2) & 3) * C * SD;
          const uint16_t* Qc = Qt + (size_t)((it + 2) & 3) * C * SD;
          float* CE = CEt + (size_t)((it + 2) & 3) * C * C * 2;
          float* G = Gt + (size_t)((it + 2) & 3) * 4 * C;
          if (hw < 4) {
            const int w = hw >> 1, pc0 = 8 * (hw & 1);
            AReg A;
            BReg B;
            load_a(A, Kc, SD, lane);
            load_b(B, (w ? Qc : Kc) + (size_t)(pc0 + (lane & 7)) * SD, lane);
            const Acc4 d = mma128(A, B);
#pragma unroll
            for (int x = 0; x < 4; ++x) {
              const int j = g + 8 * (x >> 1), dd = pc0 + 2 * t4 + (x & 1) - j;
              if (dd >= 0) CE[w * C * C + j * C + dd] = d.c[x];
            }
          }
          if (ht == 5 * 32) {
            float l[C];
#pragma unroll
            for (int i = 0; i < C; ++i) l[i] = G[i];
#pragma unroll
            for (int i = 1; i < C; ++i) l[i] = l[i - 1] + l[i];
#pragma unroll
            for (int i = 1; i < C; ++i) G[i] = l[i];
          }
        }
        // norms and gates of chunk it + 3: four rows per helper warp, k then q
        if (it + 3 >= 0 && it + 3 < nwhole) {
          uint16_t* Tc = (hw >= 4 ? Qt : Kt) + (size_t)((it + 3) & 3) * C * SD;
          float* G = Gt + (size_t)((it + 3) & 3) * 4 * C;
          // 16 lanes per row: a lane holds partial sums l and l + 16, the first level is in-lane
          const int l16 = lane & 15;
          float xs[2][8], sx[2];
#pragma unroll
          for (int x = 0; x < 2; ++x) {
#pragma unroll
            for (int y = 0; y < 2; ++y) {
              const uint64_t w = raw[2 * x + y];
              xs[x][4 * y] = bf2f((uint32_t)w);
              xs[x][4 * y + 1] = bf2f((uint32_t)(w >> 16));
              xs[x][4 * y + 2] = bf2f((uint32_t)(w >> 32));
              xs[x][4 * y + 3] = bf2f((uint32_t)(w >> 48));
            }
            float sa = xs[x][0] * xs[x][0];
            sa = sa + xs[x][1] * xs[x][1];
            sa = sa + xs[x][2] * xs[x][2];
            sa = sa + xs[x][3] * xs[x][3];
            float sb = xs[x][4] * xs[x][4];
            sb = sb + xs[x][5] * xs[x][5];
            sb = sb + xs[x][6] * xs[x][6];
            sb = sb + xs[x][7] * xs[x][7];
            sx[x] = sa + sb;
          }
          for (int o = 8; o > 0; o >>= 1) {
#pragma unroll
            for (int x = 0; x < 2; ++x) sx[x] = sx[x] + __shfl_xor_sync(0xffffffffu, sx[x], o);
          }
          // the four reciprocal square roots in four lanes of one instruction
          float rv = 0.0f;
          if (l16 < 2) rv = (1.0f / sqrtf((l16 ? sx[1] : sx[0]) + 1e-6f)) * post;
#pragma unroll
          for (int x = 0; x < 2; ++x) {
            const float rr = __shfl_sync(0xffffffffu, rv, (lane & 16) | x);
            uint16_t* dst = Tc + (size_t)((hw & 3) * 4 + 2 * x + (lane >> 4)) * SD + 4 * l16;
#pragma unroll
            for (int y = 0; y < 2; ++y)
              *reinterpret_cast<uint64_t*>(dst + 64 * y) =
                  (uint64_t)f2bf_x2(xs[x][4 * y] * rr, xs[x][4 * y + 1] * rr) |
                  ((uint64_t)f2bf_x2(xs[x][4 * y + 2] * rr, xs[x][4 * y + 3] * rr) << 32);
          }
          if (hw == 6 && lane < C) {
            float gl, beta;
            gate(araw, braw, A_log[h], dt_bias[h], gl, beta);
            G[lane] = gl;
            G[2 * C + 2 * lane] = beta;
          }
        }
        // loads for chunk it + 4, value rows of chunk it + 2
        if (it + 4 < nwhole) {
          const size_t tk = tokw + (size_t)(it + 4) * C;
          const uint16_t* src = hw >= 4 ? q : k;
#pragma unroll
          for (int x = 0; x < 4; ++x)
            raw[x] = *reinterpret_cast<const uint64_t*>(
                src + ((tk + (hw & 3) * 4 + (x & 2) + (lane >> 4)) * H + kh) * D + 64 * (x & 1) +
                4 * (lane & 15));
          if (hw == 6 && lane < C) {
            araw = ga[(tk + lane) * HV + h];
            braw = gb[(tk + lane) * HV + h];
          }
        }
        if (it + 2 >= 0 && it + 2 < nwhole && ht < 2 * VT) {
          const size_t tk = tokw + (size_t)(it + 2) * C;
          uint16_t* Vc = Vt + (size_t)((it + 2) & 3) * C * VT;
          const int vu = VT / 8;
          cp_async16(Vc + ht * 8, v + ((tk + ht / vu) * HV + h) * D + v0 + (ht % vu) * 8);
        }
        cp_async_commit();
        cp_async_wait_group1();
      } else if (it >= 0) {
        const uint16_t* Kc = Kt + (size_t)(it & 3) * C * SD;
        const uint16_t* Qc = Qt + (size_t)(it & 3) * C * SD;
        const uint16_t* Vc = Vt + (size_t)(it & 3) * C * VT;
        const float* CE = CEt + (size_t)(it & 3) * C * C * 2;
        const float* G = Gt + (size_t)(it & 3) * 4 * C;
        const size_t tok0 = tokw + (size_t)it * C;
        // state products of this warp's 16 rows with the chunk's 16 keys and 16 queries
        {
          BReg B0, B1, B2, B3;
          load_b(B0, Kc + (size_t)(lane & 7) * SD, lane);
          load_b(B1, Kc + (size_t)(8 + (lane & 7)) * SD, lane);
          load_b(B2, Qc + (size_t)(lane & 7) * SD, lane);
          load_b(B3, Qc + (size_t)(8 + (lane & 7)) * SD, lane);
          Acc4 d0, d1, d2, d3;
          mma128x4(S, B0, B1, B2, B3, d0, d1, d2, d3);
          float* ab = ABsm + (size_t)r * ST + 2 * t4;
          float* ab8 = ab + 8 * ST;
          ab[0] = d0.c[0];
          ab[1] = d0.c[1];
          ab8[0] = d0.c[2];
          ab8[1] = d0.c[3];
          ab[8] = d1.c[0];
          ab[9] = d1.c[1];
          ab8[8] = d1.c[2];
          ab8[9] = d1.c[3];
          ab[VT * ST] = d2.c[0];
          ab[VT * ST + 1] = d2.c[1];
          ab8[VT * ST] = d2.c[2];
          ab8[VT * ST + 1] = d2.c[3];
          ab[VT * ST + 8] = d3.c[0];
          ab[VT * ST + 9] = d3.c[1];
          ab8[VT * ST + 8] = d3.c[2];
          ab8[VT * ST + 9] = d3.c[3];
          __syncwarp();
        }
        // substitution: lane l < 16 carries pred of row l, lane l + 16 its output
        {
          const int lo = lane < 16, half = lane >> 4, row = 16 * warp + (lane & 15);
          const float* ce = CE + half * C * C;
          float acc[C];
#pragma unroll
          for (int x = 0; x < C; x += 4) {
            const float4 a4 =
                *reinterpret_cast<const float4*>(ABsm + (size_t)(half * VT + row) * ST + x);
            const float4 d4 = *reinterpret_cast<const float4*>(G + C + x);
            acc[x] = d4.x * a4.x;
            acc[x + 1] = d4.y * a4.y;
            acc[x + 2] = d4.z * a4.z;
            acc[x + 3] = d4.w * a4.w;
          }
          float cn[C];
#pragma unroll
          for (int x = 0; x < C; x += 4) {
            const float4 c4 = *reinterpret_cast<const float4*>(ce + x);
            cn[x] = c4.x;
            cn[x + 1] = c4.y;
            cn[x + 2] = c4.z;
            cn[x + 3] = c4.w;
          }
          float vn = bf2f(Vc[row]);
          float2 bw = *reinterpret_cast<const float2*>(G + 2 * C);
#pragma unroll
          for (int jj = 0; jj < C; ++jj) {
            float u = bw.x * (vn - acc[jj]);
            u = __shfl_sync(0xffffffffu, u, lane & 15);
            const float ut = bw.y * u, oj = __fmaf_rn(cn[0], u, acc[jj]);
            acc[jj] = lo ? ut : oj;
            if (jj + 1 < C) {
              vn = bf2f(Vc[(jj + 1) * VT + row]);
              bw = *reinterpret_cast<const float2*>(G + 2 * C + 2 * (jj + 1));
            }
#pragma unroll
            for (int x = jj + 1; x < C; ++x) acc[x] = __fmaf_rn(cn[x - jj], u, acc[x]);
            // row jj + 1 of the coefficient band: dd = 0 .. 14 - jj
#pragma unroll
            for (int x = 0; x < C - 1 - jj; x += 4) {
              const float4 c4 = *reinterpret_cast<const float4*>(ce + (jj + 1) * C + x);
              cn[x] = c4.x;
              cn[x + 1] = c4.y;
              cn[x + 2] = c4.z;
              cn[x + 3] = c4.w;
            }
          }
          uint32_t pk[C / 2];
#pragma unroll
          for (int x = 0; x < C / 2; ++x) pk[x] = f2bf_x2(acc[2 * x], acc[2 * x + 1]);
          if (lo) {
            uint4* ut = reinterpret_cast<uint4*>(Utsm + (size_t)row * SU);
            ut[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
            ut[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
          } else {
            uint16_t* op = out + (tok0 * HV + h) * D + v0 + row;
#pragma unroll
            for (int x = 0; x < C / 2; ++x) {
              op[(size_t)(2 * x) * HV * D] = (uint16_t)pk[x];
              op[(size_t)(2 * x + 1) * HV * D] = (uint16_t)(pk[x] >> 16);
            }
          }
          __syncwarp();
        }
        {
          const float dC = G[2 * C - 1];
          FoldA FA;
          ldsm_x4(FA.a[0], FA.a[1], FA.a[2], FA.a[3],
                  Utsm + (size_t)(16 * warp + (lane & 7) + 8 * ((lane >> 3) & 1)) * SU +
                      8 * (lane >> 4));
#pragma unroll
          for (int kg = 0; kg < 16; ++kg)
            fold_tile(S.a[2 * kg], S.a[2 * kg + 1], dC, FA, Kc + 8 * kg, lane, S.a[2 * kg],
                      S.a[2 * kg + 1]);
        }
      }
      __syncthreads();
    }
    folded = true;
    t += nwhole * C;
  }
  if (t < Ttot) {
    generic(0, Ttot - t, (size_t)tok_begin + t);
    n = Ttot - t;
  }

  const int nb = multi ? (bank ^ 1) : bank;
  const int w0 = multi ? 0 : n0;
  if (folded) {
    if (ism) {
#pragma unroll
      for (int kg = 0; kg < 16; ++kg) {
        const int kc = 8 * kg + 2 * t4;
        *reinterpret_cast<uint32_t*>(Ssm + (size_t)r * SD + kc) = S.a[2 * kg];
        *reinterpret_cast<uint32_t*>(Ssm + (size_t)(r + 8) * SD + kc) = S.a[2 * kg + 1];
      }
    }
    __syncthreads();
    for (int u = tid; u < VT * 16; u += nthr)
      *reinterpret_cast<uint4*>(Spool + (size_t)u * 8) =
          *reinterpret_cast<const uint4*>(Ssm + (u >> 4) * SD + (u & 15) * 8);
  }
  if (n > w0) {
    uint16_t* Ko = Kh + (((size_t)slot * 2 + nb) * H + kh) * C * D;
    float* Uo = Uh + (((size_t)slot * 2 + nb) * HV + h) * C * D + v0;
    float* Lo = Lh + (((size_t)slot * 2 + nb) * HV + h) * C;
    for (int u = tid + w0 * 16; u < n * 16; u += nthr)
      *reinterpret_cast<uint4*>(Ko + (size_t)u * 8) =
          *reinterpret_cast<const uint4*>(Kg + (u >> 4) * SD + (u & 15) * 8);
    for (int e = tid + w0 * VT; e < n * VT; e += nthr) Uo[(e / VT) * D + (e % VT)] = Usm[e];
    if (tid >= w0 && tid < n) Lo[tid] = Lsm[tid];
  }
  if (tid < nmw) cnt_cell[tid] = n | (nb << 8);
}

inline size_t ChainSmemBytes(int num_tiles) {
  const size_t VT = D / num_tiles;
  return VT * SD * 2 + CHAIN_AHEAD * (2 * C * SD * 2 + C * VT * 2 + C * C * 8 + 4 * C * 4) +
         2 * C * SD * 2 + C * D * 2 + C * VT * 2 + C * C * 8 + C * VT * 4 + 4 * C * 4 +
         2 * VT * ST * 4 + VT * SU * 2;
}

inline cudaError_t LaunchPrefillChain(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                                      const uint16_t* ga, const uint16_t* gb, const float* A_log,
                                      const float* dt_bias, uint16_t* pool, const int* slots,
                                      const int* cu, const int* prefix, uint16_t* out, uint16_t* Kh,
                                      float* Uh, float* Lh, int* cnt, float scale, int batch, int H,
                                      int HV, int num_slots, int num_tiles, int reset,
                                      int num_tokens, cudaStream_t stream) {
  const size_t smem = ChainSmemBytes(num_tiles);
  cudaError_t status = cudaFuncSetAttribute(
      ChainKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (status != cudaSuccess) return status;
  ChainKernel<<<batch * HV * num_tiles, 256 / num_tiles + 256, smem, stream>>>(
      q, k, v, ga, gb, A_log, dt_bias, pool, slots, cu, out, Kh, Uh, Lh, cnt, scale, H, HV,
      num_slots, prefix, reset, num_tiles, num_tokens);
  return cudaGetLastError();
}

inline size_t PrefillSmemBytes(int num_tiles, int mmax) {
  const size_t VT = D / num_tiles;
  size_t n = VT * SD * 2 + C * SD * 2 + C * VT * 4 + 2 * rup16(C * 4) + 2 * rup16(mmax * 4);
  n += 2 * mmax * D * 2 + 2 * mmax * SD * 2 + 2 * mmax * VT * 2;
  return n + 8 * mmax * VT + C * (VT + 8) * 2 + 8 * C * mmax;
}

constexpr size_t kDecodeSmemBytes = D * SD * 2 + C * SD * 2 + C * D * 4 + 2 * (C * 4) + SD * 2 +
                                    16 + 2 * D * 4 + 2 * C * 4 + C * SD * 2;

// One CTA per (sequence, v-head, V-tile); max_tokens bounds the longest sequence of the call.
inline cudaError_t LaunchPrefill(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                                 const uint16_t* ga, const uint16_t* gb, const float* A_log,
                                 const float* dt_bias, uint16_t* pool, const int* slots,
                                 const int* cu, const int* prefix, uint16_t* out, uint16_t* Kh,
                                 float* Uh, float* Lh, int* cnt, float scale, int batch, int H,
                                 int HV, int num_slots, int num_tiles, int max_tokens, int reset,
                                 int num_tokens, cudaStream_t stream) {
  const int mmax = max_tokens < C ? max_tokens : C;
  const size_t smem = PrefillSmemBytes(num_tiles, mmax);
  cudaError_t status = cudaFuncSetAttribute(
      PrefillKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (status != cudaSuccess) return status;
  PrefillKernel<<<batch * HV * num_tiles, NTHR, smem, stream>>>(
      q, k, v, ga, gb, A_log, dt_bias, pool, slots, cu, out, Kh, Uh, Lh, cnt, scale, H, HV,
      num_tiles, mmax, num_slots, prefix, reset, num_tokens);
  return cudaGetLastError();
}

inline cudaError_t LaunchDecode(const uint16_t* q, const uint16_t* k, const uint16_t* v,
                                const uint16_t* ga, const uint16_t* gb, const float* A_log,
                                const float* dt_bias, uint16_t* pool, const int* slots,
                                uint16_t* out, uint16_t* Kh, float* Uh, float* Lh, int* cnt,
                                float scale, int batch, int H, int HV, int num_slots, int early,
                                cudaStream_t stream) {
  auto kernel = early ? DecodeKernel<true> : DecodeKernel<false>;
  cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            static_cast<int>(kDecodeSmemBytes));
  if (status != cudaSuccess) return status;
  kernel<<<batch * HV, NTHR, kDecodeSmemBytes, stream>>>(
      q, k, v, ga, gb, A_log, dt_bias, pool, slots, out, Kh, Uh, Lh, cnt, scale, H, HV, num_slots);
  return cudaGetLastError();
}

}  // namespace bi_class_s
}  // namespace gdn
}  // namespace flashinfer

#endif  // FLASHINFER_GDN_BATCH_INVARIANT_CLASS_S_CUH_
