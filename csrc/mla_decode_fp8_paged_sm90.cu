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

// Specialized SM90 batch-decode MLA attention for an fp8 (e4m3) packed query
// over an fp8 (e4m3) packed paged compressed-KV cache: 8 query heads,
// head_dim_ckv 512 + head_dim_kpe 64 (576-byte packed rows), page size 64,
// causal, bf16 output, per-tensor float32 Q and KV descales.
//
// Algorithm: an all-fp8 split-KV kernel. QK^T runs as fp8 mma (m16n8k32
// e4m3, fp32 accumulate) over cp.async-staged 128-key (two-page) tiles with
// a rolling refill; softmax statistics are fp32 online-softmax with a
// unit-max probability quantisation on long splits (packed-fp16 P.V below
// overflow at the same softmax ratio); the P.V product accumulates against
// the same staged fp8 pages and per-split partial outputs (fp8) with fp32
// M/L land in a caller-owned workspace, merged by `combine_kernel` with the
// KV descale folded into the final bf16 write. The split plan (tiles cover
// two pages; one-wave budget against the device SM count; a warp-group
// variant for the deepest multi-query tiles) is carried verbatim from the
// measured launcher.
//
// PRECISION (part of the API contract, disclosed in the dispatcher): this
// is an all-fp8-compute kernel (fp8 q x fp8 KV end-to-end, the FlashMLA
// dense-fp8 class) — NOT the stock wrapper's bf16-q-over-dequantised-KV
// arithmetic. It is deterministic (atomics-free, bit-stable); its measured
// record (per-element error profile on the allowance row, fresh seeds,
// magnitude perturbations, bit-determinism) accompanies the change.
//
// `init` must run once per device before the first `run` (SM-count query +
// dynamic smem opt-in), outside CUDA-graph capture. `run` performs no
// allocation, no synchronisation and no attribute queries: it launches on
// the caller's current stream and is CUDA-graph capturable.

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "tvm_ffi_utils.h"

namespace {

constexpr int DIMQ = 576;
constexpr int CKV  = 512;
constexpr int PAGE = 64;
constexpr int NH   = 8;
constexpr int TT   = 128;   // tokens per kv tile == 2 pages
constexpr int KSTR = 592;   // linear fp8 byte stride of a Q/K row (592%128==80)
constexpr int PSTR = 144;   // fp8 byte stride of a P row (128 slots + pad)
constexpr float PLOG = 3.0f;      // log2 P pre-scale vs the lagged reference max
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ uint32_t prm(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(s));
  return r;
}
__device__ __forceinline__ uint32_t sptr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mma32(float* d, uint32_t a0, uint32_t a1,
                                      uint32_t a2, uint32_t a3,
                                      uint32_t b0, uint32_t b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma32h(uint32_t* d, uint32_t a0, uint32_t a1,
                                       uint32_t a2, uint32_t a3,
                                       uint32_t b0, uint32_t b1) {
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
    "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
    : "+r"(d[0]), "+r"(d[1])
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldm4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void sts16(uint32_t a, unsigned short v) {
  asm volatile("st.shared.u16 [%0], %1;" :: "r"(a), "h"(v) : "memory");
}
__device__ __forceinline__ void row_group_sync(int id) {
  asm volatile("bar.sync %0, 128;" :: "r"(id) : "memory");
}

template <int MT, bool HALF_ACC>
__global__ __launch_bounds__(128 * MT, 1)
void mla_kernel(const uint8_t* __restrict__ qg, const uint8_t* __restrict__ kvg,
                const int* __restrict__ btab, const int* __restrict__ slen,
                const float* __restrict__ dqp, const float* __restrict__ dkp,
                float sm_scale, __nv_bfloat16* __restrict__ outp,
                uint8_t* __restrict__ Op, float* __restrict__ Mp, float* __restrict__ Lp,
                int B, int QL, int pps, int ns, int pps_split, int QPC) {
  constexpr int NTHR = 128 * MT;
  constexpr int ROWS = 16 * MT;
  constexpr int NLD  = (PAGE * 36) / NTHR;

  extern __shared__ char smraw[];
  uint8_t* Qs = reinterpret_cast<uint8_t*>(smraw);
  uint8_t* Ks = Qs + ROWS * KSTR;
  uint8_t* Ps = Ks + 2 * TT * KSTR;
  float*   red = reinterpret_cast<float*>(Ps + ROWS * PSTR);   // [ROWS][4] max
  float*   red2 = red + ROWS * 4;                              // [ROWS][4] sum

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int m = warp >> 2, s = warp & 3;
  const int qc = blockIdx.z;
  int b = 0, sp = blockIdx.x;
  for (; b < B; ++b) {
    const int ppr_b = (slen[b] + PAGE - 1) / PAGE;
    const int nsp_b = (ppr_b + pps_split - 1) / pps_split;
    if (sp < nsp_b) break;
    sp -= nsp_b;
  }
  if (b == B) return;

  const int L = slen[b];
  const int ppr = (L + PAGE - 1) / PAGE;
  const int pg0 = sp * pps_split;
  const int pg1 = min(ppr, pg0 + pps_split);
  const int q0 = qc * QPC;
  const int vrows = min(ROWS, (QL - q0) * NH);
  const int rows_tot = QL * NH;

  if (pg0 >= pg1) {
    for (int r = tid; r < vrows; r += NTHR) {
      int gi = (b * ns + sp) * rows_tot + q0 * NH + r;
      Mp[gi] = -INFINITY; Lp[gi] = 0.f;
    }
    return;
  }

  // ---- load Q tile (raw fp8, linear stride) ----
  {
    const uint8_t* qb = qg + (size_t)(b * QL + q0) * NH * DIMQ;
    for (int j = tid; j < ROWS * 36; j += NTHR) {
      int row = j / 36, c = j % 36;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (row < vrows) v = *reinterpret_cast<const uint4*>(qb + (size_t)j * 16);
      *reinterpret_cast<uint4*>(Qs + row * KSTR + (c << 4)) = v;
    }
  }

  const float qk_scale = dqp[0] * dkp[0] * sm_scale * LOG2E;
  const float o_scale = dkp[0];
  // Long split ranges use unit-max P quantization so packed-FP16 PV has the
  // same headroom as the old FP32 path.  The scale cancels through lrun.
  const float pshift = (HALF_ACC && pps_split > 28) ? 0.f : PLOG;

  constexpr int OREGS = HALF_ACC ? 2 : 4;
  uint32_t oacc[16][OREGS];
#pragma unroll
  for (int i = 0; i < 16; ++i)
#pragma unroll
    for (int k = 0; k < OREGS; ++k) oacc[i][k] = 0;
  float mrun0 = -INFINITY, mrun1 = -INFINITY, lrun0 = 0.f, lrun1 = 0.f;

  const int rA = m * 16 + (lane >> 2);
  const int rB = rA + 8;
  const int gA = q0 * NH + rA, gB = q0 * NH + rB;
  const int ptop = pg1 * PAGE;
  const int limA = min(max(1, L - QL + (gA / NH) + 1), ptop);
  const int limB = min(max(1, L - QL + (gB / NH) + 1), ptop);

  auto loadpair = [&](int pg, int buf) {
    uint8_t* dst = Ks + buf * TT * KSTR;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int pgh = pg + h;
      if (pgh >= pg1) pgh = pg1 - 1;      // clamp: duplicate data is masked out
      const uint8_t* kb = kvg + (size_t)btab[b * pps + pgh] * PAGE * DIMQ;
      uint8_t* d2 = dst + h * PAGE * KSTR;
#pragma unroll
      for (int i = 0; i < NLD; ++i) {
        int j = tid + i * NTHR, row = j / 36, c = j % 36;
        uint32_t a = sptr(d2 + row * KSTR + (c << 4));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(a), "l"(kb + (size_t)j * 16));
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  const int mi = lane >> 3, l7 = lane & 7;
  const int qrow = m * 16 + ((mi & 1) << 3) + l7;
  const int bcA  = (mi >> 1) << 4;
  const int krow = s * 32 + ((mi >> 1) << 3) + l7;
  const int bcB  = (mi & 1) << 4;
  const int prow = m * 16 + ((mi & 1) << 3) + l7;
  const int pbc  = (mi >> 1) << 4;

  // --- page-invariant base addresses (linear layout: base + immediate) ---
  const uint32_t qA = sptr(Qs) + qrow * KSTR + bcA;
  const uint32_t kA = krow * KSTR + bcB;        // offset within one K buffer
  const uint32_t Ksb = sptr(Ks);
  const uint32_t pA = sptr(Ps) + prow * PSTR + pbc;
  const uint32_t vofs = lane * KSTR + s * 128;
  // P store addresses (tile invariant)
  uint32_t psA[4];
  uint32_t psB[4];
  const uint32_t Psb = sptr(Ps);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int col = s * 32 + ((j >> 1) << 4) + ((j & 1) << 3) + 2 * (lane & 3);
    int t16 = col & 15;
    int slot = (col >> 5) * 32 + (((col >> 4) & 1) << 4)
             + 4 * ((t16 & 7) >> 1) + 2 * (t16 >> 3);
    psA[j] = Psb + rA * PSTR + slot;
    psB[j] = Psb + rB * PSTR + slot;
  }
  const int cbase = s * 32 + 2 * (lane & 3);   // tile-local token col for j=0

  loadpair(pg0, 0);
  int cur = 0;

  for (int pg = pg0; pg < pg1; pg += 2) {
    const bool more = (pg + 2) < pg1;
    if (more) loadpair(pg + 2, cur ^ 1);
    if (more) asm volatile("cp.async.wait_group 1;\n" ::);
    else      asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    const uint32_t Kbase = Ksb + cur * (TT * KSTR);
    const int tok0 = pg * PAGE;

    float sacc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) sacc[j][k] = 0.f;

    {
      uint32_t qp = qA, kp = Kbase + kA;
#pragma unroll 2
      for (int kk = 0; kk < 18; ++kk) {
        uint32_t a[4], b0[4], b1[4];
        ldm4(a, qp); ldm4(b0, kp); ldm4(b1, kp + 16 * KSTR);
        mma32(sacc[0], a[0], a[1], a[2], a[3], b0[0], b0[1]);
        mma32(sacc[1], a[0], a[1], a[2], a[3], b0[2], b0[3]);
        mma32(sacc[2], a[0], a[1], a[2], a[3], b1[0], b1[1]);
        mma32(sacc[3], a[0], a[1], a[2], a[3], b1[2], b1[3]);
        qp += 32; kp += 32;
      }
    }

    float mxA = -INFINITY, mxB = -INFINITY;
    const int remA = limA - tok0, remB = limB - tok0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int c0 = cbase + ((j >> 1) << 4) + ((j & 1) << 3);
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float xa = (c0 + e < remA) ? sacc[j][e] * qk_scale : -INFINITY;
        float xb = (c0 + e < remB) ? sacc[j][2 + e] * qk_scale : -INFINITY;
        sacc[j][e] = xa; sacc[j][2 + e] = xb;
        mxA = fmaxf(mxA, xa); mxB = fmaxf(mxB, xb);
      }
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, d));
      mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, d));
    }
    // first tile of the split: establish the reference max exactly
    if (pg == pg0) {
      if ((lane & 3) == 0) { red[rA * 4 + s] = mxA; red[rB * 4 + s] = mxB; }
      row_group_sync(m + 1);
      float4 ia = *reinterpret_cast<const float4*>(red + rA * 4);
      float4 ib = *reinterpret_cast<const float4*>(red + rB * 4);
      mrun0 = fmaxf(fmaxf(ia.x, ia.y), fmaxf(ia.z, ia.w));
      mrun1 = fmaxf(fmaxf(ib.x, ib.y), fmaxf(ib.z, ib.w));
      row_group_sync(m + 1);
    }
    // lagged reference max: quantize P against the previous tiles' running max
    const float eA = mrun0 - pshift, eB = mrun1 - pshift;
    float sA = 0.f, sB = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float pa[2], pb[2];
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        float xa = (eA == -INFINITY) ? 0.f : exp2f(sacc[j][e] - eA);
        float xb = (eB == -INFINITY) ? 0.f : exp2f(sacc[j][2 + e] - eB);
        sA += xa; sB += xb;
        pa[e] = xa; pb[e] = xb;
      }
      unsigned short qa = __nv_cvt_float2_to_fp8x2(make_float2(pa[0], pa[1]),
                                                   __NV_SATFINITE, __NV_E4M3);
      unsigned short qb2 = __nv_cvt_float2_to_fp8x2(make_float2(pb[0], pb[1]),
                                                    __NV_SATFINITE, __NV_E4M3);
      sts16(psA[j], qa);
      sts16(psB[j], qb2);
    }
#pragma unroll
    for (int d = 1; d < 4; d <<= 1) {
      sA += __shfl_xor_sync(0xffffffff, sA, d);
      sB += __shfl_xor_sync(0xffffffff, sB, d);
    }
    if ((lane & 3) == 0) {
      red[rA * 4 + s] = mxA; red[rB * 4 + s] = mxB;
      red2[rA * 4 + s] = sA; red2[rB * 4 + s] = sB;
    }
    row_group_sync(m + 1);
    float4 fa = *reinterpret_cast<const float4*>(red + rA * 4);
    float4 fb = *reinterpret_cast<const float4*>(red + rB * 4);
    float4 ga = *reinterpret_cast<const float4*>(red2 + rA * 4);
    float4 gb = *reinterpret_cast<const float4*>(red2 + rB * 4);
    float mnA = fmaxf(mrun0, fmaxf(fmaxf(fa.x, fa.y), fmaxf(fa.z, fa.w)));
    float mnB = fmaxf(mrun1, fmaxf(fmaxf(fb.x, fb.y), fmaxf(fb.z, fb.w)));
    float corrA = (mnA == -INFINITY) ? 1.f : exp2f(mrun0 - mnA);
    float corrB = (mnB == -INFINITY) ? 1.f : exp2f(mrun1 - mnB);
    mrun0 = mnA; mrun1 = mnB;
    lrun0 = (lrun0 + ((ga.x + ga.y) + (ga.z + ga.w))) * corrA;
    lrun1 = (lrun1 + ((gb.x + gb.y) + (gb.z + gb.w))) * corrB;

    // ---- PV (fp8 m16n8k32, V^T fragments from ldmatrix.trans + prmt) ----
    {
      const uint32_t vb0 = Kbase + vofs;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t a[4];
        ldm4(a, pA + kk * 32);
        const uint32_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
        const uint32_t vb = vb0 + kk * (32 * KSTR);
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          uint32_t r[4];
          ldm4t(r, vb + (uint32_t)(g << 4));
          uint32_t be0 = prm(r[0], r[1], 0x6420u), be1 = prm(r[2], r[3], 0x6420u);
          if constexpr (HALF_ACC)
            mma32h(oacc[g * 2], a0, a1, a2, a3, be0, be1);
          else
            mma32(reinterpret_cast<float*>(oacc[g * 2]), a0, a1, a2, a3, be0, be1);
          uint32_t bo0 = prm(r[0], r[1], 0x7531u), bo1 = prm(r[2], r[3], 0x7531u);
          if constexpr (HALF_ACC)
            mma32h(oacc[g * 2 + 1], a0, a1, a2, a3, bo0, bo1);
          else
            mma32(reinterpret_cast<float*>(oacc[g * 2 + 1]), a0, a1, a2, a3, bo0, bo1);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if constexpr (HALF_ACC) {
        __half2 ha = *reinterpret_cast<__half2*>(&oacc[i][0]);
        __half2 hb = *reinterpret_cast<__half2*>(&oacc[i][1]);
        ha = __hmul2(ha, __float2half2_rn(corrA));
        hb = __hmul2(hb, __float2half2_rn(corrB));
        oacc[i][0] = *reinterpret_cast<uint32_t*>(&ha);
        oacc[i][1] = *reinterpret_cast<uint32_t*>(&hb);
      } else {
        float* of = reinterpret_cast<float*>(oacc[i]);
        of[0] *= corrA; of[1] *= corrA;
        of[2] *= corrB; of[3] *= corrB;
      }
    }

    cur ^= 1;
    __syncthreads();
  }

  // ---- epilogue ----
  const bool single = (ns == 1);
  float rAi = (lrun0 > 0.f) ? (single ? o_scale : 1.f) / lrun0 : 0.f;
  float rBi = (lrun1 > 0.f) ? (single ? o_scale : 1.f) / lrun1 : 0.f;
  if (!single) {
    int giA = (b * ns + sp) * rows_tot + gA, giB = (b * ns + sp) * rows_tot + gB;
    if (s == 0 && (lane & 3) == 0) {
      if (rA < vrows) { Mp[giA] = mrun0; Lp[giA] = lrun0; }
      if (rB < vrows) { Mp[giB] = mrun1; Lp[giB] = lrun1; }
    }
  }
#pragma unroll
  for (int g = 0; g < 8; ++g) {
    int col = s * 128 + g * 16 + 4 * (lane & 3);
    float a0, a1, a2, a3, b0, b1, b2, b3;
    if constexpr (HALF_ACC) {
      float2 av0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[2 * g][0]));
      float2 av1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[2 * g + 1][0]));
      float2 bv0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[2 * g][1]));
      float2 bv1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[2 * g + 1][1]));
      a0 = av0.x; a1 = av1.x; a2 = av0.y; a3 = av1.y;
      b0 = bv0.x; b1 = bv1.x; b2 = bv0.y; b3 = bv1.y;
    } else {
      float* oa0 = reinterpret_cast<float*>(oacc[2 * g]);
      float* oa1 = reinterpret_cast<float*>(oacc[2 * g + 1]);
      a0 = oa0[0]; a1 = oa1[0]; a2 = oa0[1]; a3 = oa1[1];
      b0 = oa0[2]; b1 = oa1[2]; b2 = oa0[3]; b3 = oa1[3];
    }
    if (single) {
      if (rA < vrows) {
        __nv_bfloat162 t0 = __floats2bfloat162_rn(a0 * rAi, a1 * rAi);
        __nv_bfloat162 t1 = __floats2bfloat162_rn(a2 * rAi, a3 * rAi);
        __nv_bfloat162* d = reinterpret_cast<__nv_bfloat162*>(
            outp + (size_t)(b * QL * NH + gA) * CKV + col);
        d[0] = t0; d[1] = t1;
      }
      if (rB < vrows) {
        __nv_bfloat162 t0 = __floats2bfloat162_rn(b0 * rBi, b1 * rBi);
        __nv_bfloat162 t1 = __floats2bfloat162_rn(b2 * rBi, b3 * rBi);
        __nv_bfloat162* d = reinterpret_cast<__nv_bfloat162*>(
            outp + (size_t)(b * QL * NH + gB) * CKV + col);
        d[0] = t0; d[1] = t1;
      }
    } else {
      size_t base = (size_t)((b * ns + sp) * rows_tot) * CKV;
      if (rA < vrows) {
        __nv_fp8x4_e4m3 t(make_float4(a0 * rAi, a1 * rAi, a2 * rAi, a3 * rAi));
        *reinterpret_cast<uint32_t*>(Op + base + (size_t)gA * CKV + col) = t.__x;
      }
      if (rB < vrows) {
        __nv_fp8x4_e4m3 t(make_float4(b0 * rBi, b1 * rBi, b2 * rBi, b3 * rBi));
        *reinterpret_cast<uint32_t*>(Op + base + (size_t)gB * CKV + col) = t.__x;
      }
    }
  }
}

// ============ wgmma (sm_90a) 48-row kernel, 2 warpgroups, 256 thr ============
constexpr int CMS = 144;          // LBO: byte stride between 16B core-matrix cols
constexpr int SBOX = 36 * CMS;    // 5184: stride between 8-row groups
constexpr int PGB = 8 * SBOX;     // 41472: bytes for 64 rows (one page)

__device__ __forceinline__ uint64_t gdesc(uint32_t addr) {
  return ((uint64_t)((addr & 0x3FFFFu) >> 4))
       | ((uint64_t)((uint32_t)CMS >> 4) << 16)
       | ((uint64_t)((uint32_t)SBOX >> 4) << 32);
}

#define WGMMA64(SD)                                                            \
  asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %34, 0;\n"                \
    "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "                    \
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"                                         \
    : "+f"(sa[0]),"+f"(sa[1]),"+f"(sa[2]),"+f"(sa[3]),"+f"(sa[4]),"+f"(sa[5]),"+f"(sa[6]),"+f"(sa[7]),"+f"(sa[8]),"+f"(sa[9]),"+f"(sa[10]),"+f"(sa[11]),"+f"(sa[12]),"+f"(sa[13]),"+f"(sa[14]),"+f"(sa[15]),"+f"(sa[16]),"+f"(sa[17]),"+f"(sa[18]),"+f"(sa[19]),"+f"(sa[20]),"+f"(sa[21]),"+f"(sa[22]),"+f"(sa[23]),"+f"(sa[24]),"+f"(sa[25]),"+f"(sa[26]),"+f"(sa[27]),"+f"(sa[28]),"+f"(sa[29]),"+f"(sa[30]),"+f"(sa[31]) : "l"(da), "l"(db), "n"(SD))

#define WGMMA64_INIT                                                           \
  asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, 0, 0;\n"                  \
    "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "                    \
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n" \
    : "=f"(sa[0]),"=f"(sa[1]),"=f"(sa[2]),"=f"(sa[3]),"=f"(sa[4]),"=f"(sa[5]),"=f"(sa[6]),"=f"(sa[7]),"=f"(sa[8]),"=f"(sa[9]),"=f"(sa[10]),"=f"(sa[11]),"=f"(sa[12]),"=f"(sa[13]),"=f"(sa[14]),"=f"(sa[15]),"=f"(sa[16]),"=f"(sa[17]),"=f"(sa[18]),"=f"(sa[19]),"=f"(sa[20]),"=f"(sa[21]),"=f"(sa[22]),"=f"(sa[23]),"=f"(sa[24]),"=f"(sa[25]),"=f"(sa[26]),"=f"(sa[27]),"=f"(sa[28]),"=f"(sa[29]),"=f"(sa[30]),"=f"(sa[31]) : "l"(da), "l"(db))

template <bool HALF_ACC>
__global__ __launch_bounds__(256, 1)
void mla_kernel_wg(const uint8_t* __restrict__ qg, const uint8_t* __restrict__ kvg,
                const int* __restrict__ btab, const int* __restrict__ slen,
                const float* __restrict__ dqp, const float* __restrict__ dkp,
                float sm_scale, __nv_bfloat16* __restrict__ outp,
                uint8_t* __restrict__ Op, float* __restrict__ Mp, float* __restrict__ Lp,
                int B, int QL, int pps, int ns, int pps_split, int QPC) {
  constexpr int NTHR = 256;
  constexpr int ROWS = 48;

  extern __shared__ char smraw[];
  uint8_t* Qs = reinterpret_cast<uint8_t*>(smraw);   // 64 rows
  uint8_t* Ks = Qs + PGB;                            // 2 buffers x 2 pages
  uint8_t* Ps = Ks + 4 * PGB;                        // 48 x PSTR
  float*   tmx = reinterpret_cast<float*>(Ps + ROWS * PSTR);   // [48][2]
  float*   tsm = tmx + 2 * ROWS;                               // [48][2]

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wg = warp >> 2, w = warp & 3;
  const int qc = blockIdx.z;
  int b = 0, sp = blockIdx.x;
  for (; b < B; ++b) {
    const int ppr_b = (slen[b] + PAGE - 1) / PAGE;
    const int nsp_b = (ppr_b + pps_split - 1) / pps_split;
    if (sp < nsp_b) break;
    sp -= nsp_b;
  }
  if (b == B) return;

  const int L = slen[b];
  const int ppr = (L + PAGE - 1) / PAGE;
  const int pg0 = sp * pps_split;
  const int pg1 = min(ppr, pg0 + pps_split);
  const int q0 = qc * QPC;
  const int vrows = min(ROWS, (QL - q0) * NH);
  const int rows_tot = QL * NH;

  if (pg0 >= pg1) {
    for (int r = tid; r < vrows; r += NTHR) {
      int gi = (b * ns + sp) * rows_tot + q0 * NH + r;
      Mp[gi] = -INFINITY; Lp[gi] = 0.f;
    }
    return;
  }

  // ---- load Q tile into core-matrix interleaved layout (64 rows, zero pad) ----
  {
    const uint8_t* qb = qg + (size_t)(b * QL + q0) * NH * DIMQ;
    for (int j = tid; j < 64 * 36; j += NTHR) {
      int row = j / 36, c = j % 36;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (row < vrows) v = *reinterpret_cast<const uint4*>(qb + (size_t)j * 16);
      *reinterpret_cast<uint4*>(Qs + (row >> 3) * SBOX + c * CMS + ((row & 7) << 4)) = v;
    }
  }
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

  const float qk_scale = dqp[0] * dkp[0] * sm_scale * LOG2E;
  const float o_scale = dkp[0];
  const float pshift = (HALF_ACC && pps_split > 28) ? 0.f : PLOG;

  constexpr int OREGS = HALF_ACC ? 2 : 4;
  uint32_t oacc[3][8][OREGS];
#pragma unroll
  for (int rg = 0; rg < 3; ++rg)
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
      for (int k = 0; k < OREGS; ++k) oacc[rg][i][k] = 0;
  float mrun[3][2], lrun[3][2];
#pragma unroll
  for (int rg = 0; rg < 3; ++rg) {
    mrun[rg][0] = -INFINITY; mrun[rg][1] = -INFINITY;
    lrun[rg][0] = 0.f;       lrun[rg][1] = 0.f;
  }
  float mq0 = -INFINITY, mq1 = -INFINITY;
  float corr[3][2];
#pragma unroll
  for (int rg = 0; rg < 3; ++rg) { corr[rg][0] = 1.f; corr[rg][1] = 1.f; }

  // QK rows (row group = w, 3 of the 4 warps cover the 48 real rows)
  const int qrA = w * 16 + (lane >> 2);
  const int qrB = qrA + 8;
  const int ptop = pg1 * PAGE;
  const int limA = min(max(1, L - QL + ((q0 * NH + qrA) / NH) + 1), ptop);
  const int limB = min(max(1, L - QL + ((q0 * NH + qrB) / NH) + 1), ptop);

  auto loadpair = [&](int pg, int buf) {
    uint8_t* dst = Ks + buf * (2 * PGB);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      int pgh = pg + h;
      if (pgh >= pg1) pgh = pg1 - 1;
      const uint8_t* kb = kvg + (size_t)btab[b * pps + pgh] * PAGE * DIMQ;
      uint8_t* d2 = dst + h * PGB;
#pragma unroll
      for (int i = 0; i < 9; ++i) {
        int j = tid + i * NTHR, row = j / 36, c = j % 36;
        uint32_t a = sptr(d2 + (row >> 3) * SBOX + c * CMS + ((row & 7) << 4));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(a), "l"(kb + (size_t)j * 16));
      }
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };

  // PV operand addressing: every warp owns all 48 rows and 64 of the 512 cols
  const int mi = lane >> 3, l7 = lane & 7;
  const int pbc  = (mi >> 1) << 4;
  const uint32_t Psb = sptr(Ps);
  uint32_t pA[3];
#pragma unroll
  for (int rg = 0; rg < 3; ++rg)
    pA[rg] = Psb + (rg * 16 + ((mi & 1) << 3) + l7) * PSTR + pbc;
  const uint32_t vofs = (uint32_t)((lane >> 3) * SBOX + ((lane & 7) << 4) + warp * (4 * CMS));
  const uint32_t Ksb = sptr(Ks);

  // P store slots for the wgmma accumulator columns
  uint32_t psA[8], psB[8];
  const int cbase = wg * 64 + 2 * (lane & 3);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    int col = cbase + 8 * j;
    int t16 = col & 15;
    int slot = (col >> 5) * 32 + (((col >> 4) & 1) << 4)
             + 4 * ((t16 & 7) >> 1) + 2 * (t16 >> 3);
    psA[j] = Psb + qrA * PSTR + slot;
    psB[j] = Psb + qrB * PSTR + slot;
  }

  const uint64_t dqbase = gdesc(sptr(Qs));
  loadpair(pg0, 0);
  if (pg0 + 2 < pg1) loadpair(pg0 + 2, 1);
  int cur = 0;

  for (int pg = pg0; pg < pg1; pg += 2) {
    const bool more = (pg + 2) < pg1;
    if (more) asm volatile("cp.async.wait_group 1;\n" ::);
    else      asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    const int tok0 = pg * PAGE;
    float sa[32];
    float mxA = -INFINITY, mxB = -INFINITY, sA = 0.f, sB = 0.f;

    {
      uint64_t da = dqbase;
      uint64_t db = gdesc(Ksb + cur * (2 * PGB) + wg * PGB);
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      WGMMA64_INIT; da += 18; db += 18;
#pragma unroll
      for (int kk = 1; kk < 18; ++kk) { WGMMA64(1); da += 18; db += 18; }
      asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
      if (pg != pg0) {
#pragma unroll
        for (int rg = 0; rg < 3; ++rg) {
          const __half2 hA = __float2half2_rn(corr[rg][0]);
          const __half2 hB = __float2half2_rn(corr[rg][1]);
          const float cA = corr[rg][0], cB = corr[rg][1];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            if constexpr (HALF_ACC) {
              __half2 ha = __hmul2(*reinterpret_cast<__half2*>(&oacc[rg][i][0]), hA);
              __half2 hb = __hmul2(*reinterpret_cast<__half2*>(&oacc[rg][i][1]), hB);
              oacc[rg][i][0] = *reinterpret_cast<uint32_t*>(&ha);
              oacc[rg][i][1] = *reinterpret_cast<uint32_t*>(&hb);
            } else {
              float* of = reinterpret_cast<float*>(oacc[rg][i]);
              of[0] *= cA; of[1] *= cA;
              of[2] *= cB; of[3] *= cB;
            }
          }
        }
      }
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }

    if (w < 3) {
      const int remA = limA - tok0, remB = limB - tok0;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        int c0 = cbase + 8 * j;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float xa = (c0 + e < remA) ? sa[4 * j + e] * qk_scale : -INFINITY;
          float xb = (c0 + e < remB) ? sa[4 * j + 2 + e] * qk_scale : -INFINITY;
          sa[4 * j + e] = xa; sa[4 * j + 2 + e] = xb;
          mxA = fmaxf(mxA, xa); mxB = fmaxf(mxB, xb);
        }
      }
#pragma unroll
      for (int d = 1; d < 4; d <<= 1) {
        mxA = fmaxf(mxA, __shfl_xor_sync(0xffffffff, mxA, d));
        mxB = fmaxf(mxB, __shfl_xor_sync(0xffffffff, mxB, d));
      }
    }

    if (pg == pg0) {
      if (w < 3 && (lane & 3) == 0) {
        tmx[qrA * 2 + wg] = mxA; tmx[qrB * 2 + wg] = mxB;
      }
      __syncthreads();
      if (w < 3) {
        mq0 = fmaxf(tmx[qrA * 2], tmx[qrA * 2 + 1]);
        mq1 = fmaxf(tmx[qrB * 2], tmx[qrB * 2 + 1]);
      }
      __syncthreads();
    }

    if (w < 3) {
      const float eA = mq0 - pshift, eB = mq1 - pshift;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float pa[2], pb[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          float xa = (eA == -INFINITY) ? 0.f : exp2f(sa[4 * j + e] - eA);
          float xb = (eB == -INFINITY) ? 0.f : exp2f(sa[4 * j + 2 + e] - eB);
          sA += xa; sB += xb;
          pa[e] = xa; pb[e] = xb;
        }
        sts16(psA[j], __nv_cvt_float2_to_fp8x2(make_float2(pa[0], pa[1]),
                                               __NV_SATFINITE, __NV_E4M3));
        sts16(psB[j], __nv_cvt_float2_to_fp8x2(make_float2(pb[0], pb[1]),
                                               __NV_SATFINITE, __NV_E4M3));
      }
#pragma unroll
      for (int d = 1; d < 4; d <<= 1) {
        sA += __shfl_xor_sync(0xffffffff, sA, d);
        sB += __shfl_xor_sync(0xffffffff, sB, d);
      }
      if ((lane & 3) == 0) {
        tmx[qrA * 2 + wg] = mxA; tmx[qrB * 2 + wg] = mxB;
        tsm[qrA * 2 + wg] = sA; tsm[qrB * 2 + wg] = sB;
      }
    }
    __syncthreads();

#pragma unroll
    for (int rg = 0; rg < 3; ++rg) {
      const int r0 = rg * 16 + (lane >> 2), r1 = r0 + 8;
      float2 ta = *reinterpret_cast<const float2*>(tmx + r0 * 2);
      float2 tb = *reinterpret_cast<const float2*>(tmx + r1 * 2);
      float2 ua = *reinterpret_cast<const float2*>(tsm + r0 * 2);
      float2 ub = *reinterpret_cast<const float2*>(tsm + r1 * 2);
      float mnA = fmaxf(mrun[rg][0], fmaxf(ta.x, ta.y));
      float mnB = fmaxf(mrun[rg][1], fmaxf(tb.x, tb.y));
      float cA = (mrun[rg][0] == -INFINITY) ? 1.f : exp2f(mrun[rg][0] - mnA);
      float cB = (mrun[rg][1] == -INFINITY) ? 1.f : exp2f(mrun[rg][1] - mnB);
      mrun[rg][0] = mnA; mrun[rg][1] = mnB;
      lrun[rg][0] = (lrun[rg][0] + (ua.x + ua.y)) * cA;
      lrun[rg][1] = (lrun[rg][1] + (ub.x + ub.y)) * cB;
      corr[rg][0] = cA; corr[rg][1] = cB;
    }
    if (w < 3) {
      float2 qa = *reinterpret_cast<const float2*>(tmx + qrA * 2);
      float2 qb = *reinterpret_cast<const float2*>(tmx + qrB * 2);
      mq0 = fmaxf(mq0, fmaxf(qa.x, qa.y));
      mq1 = fmaxf(mq1, fmaxf(qb.x, qb.y));
    }

    // ---- PV (fp8 m16n8k32): V fragments shared by all three row groups ----
    {
      const uint32_t vb0 = Ksb + cur * (2 * PGB) + vofs;
#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t ap[3][4];
#pragma unroll
        for (int rg = 0; rg < 3; ++rg) ldm4(ap[rg], pA[rg] + kk * 32);
        const uint32_t vb = vb0 + kk * (4 * SBOX);
#pragma unroll
        for (int g = 0; g < 4; ++g) {
          uint32_t r[4];
          ldm4t(r, vb + (uint32_t)(g * CMS));
          uint32_t be0 = prm(r[0], r[1], 0x6420u), be1 = prm(r[2], r[3], 0x6420u);
          uint32_t bo0 = prm(r[0], r[1], 0x7531u), bo1 = prm(r[2], r[3], 0x7531u);
#pragma unroll
          for (int rg = 0; rg < 3; ++rg) {
            if constexpr (HALF_ACC) {
              mma32h(oacc[rg][g * 2], ap[rg][0], ap[rg][1], ap[rg][2], ap[rg][3], be0, be1);
              mma32h(oacc[rg][g * 2 + 1], ap[rg][0], ap[rg][1], ap[rg][2], ap[rg][3], bo0, bo1);
            } else {
              mma32(reinterpret_cast<float*>(oacc[rg][g * 2]), ap[rg][0], ap[rg][1], ap[rg][2], ap[rg][3], be0, be1);
              mma32(reinterpret_cast<float*>(oacc[rg][g * 2 + 1]), ap[rg][0], ap[rg][1], ap[rg][2], ap[rg][3], bo0, bo1);
            }
          }
        }
        __syncwarp();
        if (pg + 4 < pg1) {
          const int h = kk >> 1;
          const int pgh = min(pg + 4 + h, pg1 - 1);
          const uint8_t* kb = kvg + (size_t)btab[b * pps + pgh] * PAGE * DIMQ;
          uint8_t* d2 = Ks + cur * (2 * PGB) + h * PGB;
#pragma unroll
          for (int rr = 0; rr < 4; ++rr) {
            const int row = (kk & 1) * 32 + rr * 8 + (lane >> 2);
            const int c = warp * 4 + (lane & 3);
            const uint32_t a = sptr(d2 + (row >> 3) * SBOX + c * CMS + ((row & 7) << 4));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(a), "l"(kb + (size_t)row * DIMQ + c * 16));
          }
          if (warp < 4) {
            const int row = (kk & 1) * 32 + warp * 8 + (lane >> 2);
            const int c = 32 + (lane & 3);
            const uint32_t a = sptr(d2 + (row >> 3) * SBOX + c * CMS + ((row & 7) << 4));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(a), "l"(kb + (size_t)row * DIMQ + c * 16));
          }
        }
      }
      if (pg + 4 < pg1) asm volatile("cp.async.commit_group;\n" ::);
    }

    cur ^= 1;
    __syncthreads();
  }

  // ---- epilogue ----
  const bool single = (ns == 1);
#pragma unroll
  for (int rg = 0; rg < 3; ++rg) {
    const int rA = rg * 16 + (lane >> 2), rB = rA + 8;
    const int gA = q0 * NH + rA, gB = q0 * NH + rB;
    const float rAi = (lrun[rg][0] > 0.f) ? corr[rg][0] * (single ? o_scale : 1.f) / lrun[rg][0] : 0.f;
    const float rBi = (lrun[rg][1] > 0.f) ? corr[rg][1] * (single ? o_scale : 1.f) / lrun[rg][1] : 0.f;
    if (!single && warp == 0 && (lane & 3) == 0) {
      int giA = (b * ns + sp) * rows_tot + gA, giB = (b * ns + sp) * rows_tot + gB;
      if (rA < vrows) { Mp[giA] = mrun[rg][0]; Lp[giA] = lrun[rg][0]; }
      if (rB < vrows) { Mp[giB] = mrun[rg][1]; Lp[giB] = lrun[rg][1]; }
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      int col = warp * 64 + g * 16 + 4 * (lane & 3);
      float a0, a1, a2, a3, b0, b1, b2, b3;
      if constexpr (HALF_ACC) {
        float2 av0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[rg][2 * g][0]));
        float2 av1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[rg][2 * g + 1][0]));
        float2 bv0 = __half22float2(*reinterpret_cast<__half2*>(&oacc[rg][2 * g][1]));
        float2 bv1 = __half22float2(*reinterpret_cast<__half2*>(&oacc[rg][2 * g + 1][1]));
        a0 = av0.x; a1 = av1.x; a2 = av0.y; a3 = av1.y;
        b0 = bv0.x; b1 = bv1.x; b2 = bv0.y; b3 = bv1.y;
      } else {
        float* oa0 = reinterpret_cast<float*>(oacc[rg][2 * g]);
        float* oa1 = reinterpret_cast<float*>(oacc[rg][2 * g + 1]);
        a0 = oa0[0]; a1 = oa1[0]; a2 = oa0[1]; a3 = oa1[1];
        b0 = oa0[2]; b1 = oa1[2]; b2 = oa0[3]; b3 = oa1[3];
      }
      if (single) {
        if (rA < vrows) {
          __nv_bfloat162 t0 = __floats2bfloat162_rn(a0 * rAi, a1 * rAi);
          __nv_bfloat162 t1 = __floats2bfloat162_rn(a2 * rAi, a3 * rAi);
          __nv_bfloat162* d = reinterpret_cast<__nv_bfloat162*>(
              outp + (size_t)(b * QL * NH + gA) * CKV + col);
          d[0] = t0; d[1] = t1;
        }
        if (rB < vrows) {
          __nv_bfloat162 t0 = __floats2bfloat162_rn(b0 * rBi, b1 * rBi);
          __nv_bfloat162 t1 = __floats2bfloat162_rn(b2 * rBi, b3 * rBi);
          __nv_bfloat162* d = reinterpret_cast<__nv_bfloat162*>(
              outp + (size_t)(b * QL * NH + gB) * CKV + col);
          d[0] = t0; d[1] = t1;
        }
      } else {
        size_t base = (size_t)((b * ns + sp) * rows_tot) * CKV;
        if (rA < vrows) {
          __nv_fp8x4_e4m3 t(make_float4(a0 * rAi, a1 * rAi, a2 * rAi, a3 * rAi));
          *reinterpret_cast<uint32_t*>(Op + base + (size_t)gA * CKV + col) = t.__x;
        }
        if (rB < vrows) {
          __nv_fp8x4_e4m3 t(make_float4(b0 * rBi, b1 * rBi, b2 * rBi, b3 * rBi));
          *reinterpret_cast<uint32_t*>(Op + base + (size_t)gB * CKV + col) = t.__x;
        }
      }
    }
  }
}

#define CWARPS 16
template <int GROUP>
__global__ __launch_bounds__(32 * CWARPS, 1)
void combine_kernel(const uint8_t* __restrict__ Op, const float* __restrict__ Mp,
                    const float* __restrict__ Lp, const float* __restrict__ dkp,
                    const int* __restrict__ slen, __nv_bfloat16* __restrict__ outp,
                    int ns, int rows_tot, int pps_split) {
  constexpr int NT = 32 * CWARPS;
  constexpr int PARTS = CWARPS / GROUP;
  const int b = blockIdx.z;
  const int r = blockIdx.x;
  const int row = b * rows_tot + r;
  const int tid = threadIdx.x;
  const int lane = tid & 31, wp = tid >> 5;
  const int chunk = blockIdx.y * GROUP + wp / PARTS;
  const int part = wp % PARTS;

  int nsb = 1;
  if (lane == 0) {
    const int ppr = (slen[b] + PAGE - 1) / PAGE;
    nsb = (ppr + pps_split - 1) / pps_split;
    if (nsb > ns) nsb = ns;
    if (nsb < 1) nsb = 1;
  }
  nsb = __shfl_sync(0xffffffffu, nsb, 0);

  extern __shared__ float sw[];            // [ns] split weights
  __shared__ float sred[CWARPS][128];      // per-warp partials, 128 cols
  __shared__ float sstat[2];

  float lm = -INFINITY;
  for (int s = tid; s < nsb; s += NT) lm = fmaxf(lm, Mp[(b * ns + s) * rows_tot + r]);
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) lm = fmaxf(lm, __shfl_xor_sync(0xffffffffu, lm, d));
  if (lane == 0) sred[wp][0] = lm;
  __syncthreads();
  if (tid == 0) {
    float x = -INFINITY;
#pragma unroll
    for (int w = 0; w < CWARPS; ++w) x = fmaxf(x, sred[w][0]);
    sstat[0] = x;
  }
  __syncthreads();
  const float mmax = sstat[0];

  float ls = 0.f;
  for (int s = tid; s < nsb; s += NT) {
    float m = Mp[(b * ns + s) * rows_tot + r];
    float w = (m > -INFINITY) ? exp2f(m - mmax) * Lp[(b * ns + s) * rows_tot + r] : 0.f;
    sw[s] = w; ls += w;
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) ls += __shfl_xor_sync(0xffffffffu, ls, d);
  if (lane == 0) sred[wp][1] = ls;
  __syncthreads();
  if (tid == 0) {
    float x = 0.f;
#pragma unroll
    for (int w = 0; w < CWARPS; ++w) x += sred[w][1];
    sstat[1] = x;
  }
  __syncthreads();
  const float den = sstat[1];
  const float inv = (den > 0.f) ? dkp[0] / den : 0.f;

  const int c = chunk * 128 + lane * 4;
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  const uint8_t* base = Op + (size_t)(b * ns) * rows_tot * CKV + (size_t)r * CKV + c;
  const size_t stride = (size_t)rows_tot * CKV;
#pragma unroll 4
  for (int s = part; s < nsb; s += PARTS) {
    float w = sw[s];
    __nv_fp8x4_e4m3 v;
    v.__x = *reinterpret_cast<const uint32_t*>(base + (size_t)s * stride);
    float4 a = static_cast<float4>(v);
    acc.x = fmaf(w, a.x, acc.x); acc.y = fmaf(w, a.y, acc.y);
    acc.z = fmaf(w, a.z, acc.z); acc.w = fmaf(w, a.w, acc.w);
  }
  __syncthreads();
  *reinterpret_cast<float4*>(&sred[wp][lane * 4]) = acc;
  __syncthreads();
  if (tid < 128 * GROUP) {
    const int out_chunk = blockIdx.y * GROUP + tid / 128;
    const int out_col = tid & 127;
    const int first_warp = (tid / 128) * PARTS;
    float o = 0.f;
#pragma unroll
    for (int w = 0; w < PARTS; ++w) o += sred[first_warp + w][out_col];
    outp[(size_t)row * CKV + out_chunk * 128 + out_col] = __float2bfloat16(o * inv);
  }
}



}  // namespace

// ---------------------------------------------------------------------------
// Host side. The split-plan arithmetic and the kernel launch sequence are
// carried verbatim from the measured launcher; only the workspace ownership
// (caller-owned plan-time buffer instead of a per-call torch allocation),
// the SM-count query and the dynamic-smem opt-ins (moved to `init`, outside
// CUDA-graph capture) changed, so `run` never allocates and is capturable.
// ---------------------------------------------------------------------------

namespace {

constexpr int kMaxDevices = 128;
int g_sm_count[kMaxDevices] = {0};

struct MlaSplitPlan {
  int MT;
  int QPC;
  int nqc;
  int ns;
  int pps_split;
  int nblocks;
  int rows_tot;
};

// Verbatim split arithmetic of the measured launcher (`run`), parameterised
// on the plan-time facts it read from its tensors.
MlaSplitPlan mla_decode_fp8_split_plan(int B, int QL, int pps, int total_pages,
                                       int sm_count) {
  MlaSplitPlan P;
  const int maxsp = (pps + 1) / 2;  // tiles cover 2 pages
  int MT = (QL + 1) / 2;
  if (MT > 3) MT = 3;
  if (MT < 1) MT = 1;
  if (B == 1 && maxsp * ((QL + 1) / 2) <= 2 * sm_count) MT = 1;
  P.MT = MT;
  P.QPC = 2 * MT;
  P.nqc = (QL + P.QPC - 1) / P.QPC;
  int ns, pps_split;
  if (B == 1) {
    ns = sm_count;
    if (ns > maxsp) ns = maxsp;
    if (ns > 256) ns = 256;
    if (ns < 1) ns = 1;
    pps_split = (((pps + ns - 1) / ns) + 1) & ~1;
    ns = (pps + pps_split - 1) / pps_split;
    P.nblocks = ns;
  } else {
    const int target = std::max(1, sm_count - B + 1);
    pps_split = (std::max(1, (total_pages + target - 1) / target) + 1) & ~1;
    ns = (pps + pps_split - 1) / pps_split;
    if (ns > 256) {
      ns = 256;
      pps_split = (((pps + ns - 1) / ns) + 1) & ~1;
      ns = (pps + pps_split - 1) / pps_split;
    }
    // Launch the full logical (sequence, split) rectangle. Split width may
    // use physical cache pages as a scheduling heuristic, but launch
    // coverage must not assume page ids are unique across sequences.
    P.nblocks = B * ns;
  }
  P.ns = ns;
  P.pps_split = pps_split;
  P.rows_tot = QL * NH;
  return P;
}

size_t mla_decode_fp8_ws_bytes(const MlaSplitPlan& P, int B) {
  if (P.ns <= 1) return 0;
  const size_t nrow = (size_t)B * P.ns * P.rows_tot;
  return nrow * CKV + 2 * nrow * sizeof(float);
}

}  // namespace

// One-time per-device SM-count query + dynamic shared memory opt-in for
// every kernel instantiation `run` can launch. Outside CUDA-graph capture,
// before the first `run` on the device.
void mla_decode_fp8_paged_sm90_init(int64_t device_id) {
  TVM_FFI_ICHECK_GE(device_id, 0);
  TVM_FFI_ICHECK_LT(device_id, kMaxDevices);
  ffi::CUDADeviceGuard device_guard(static_cast<int>(device_id));
  int sm = 0;
  cudaError_t st =
      cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(device_id));
  TVM_FFI_ICHECK(st == cudaSuccess && sm > 0)
      << "mla_decode_fp8_paged_sm90: SM-count query failed: "
      << cudaGetErrorString(st);
  g_sm_count[device_id] = sm;
  const size_t smw = (size_t)5 * PGB + (size_t)48 * PSTR + (size_t)4 * 48 * 4;
  for (int MT = 1; MT <= 2; ++MT) {
    const int ROWS = 16 * MT;
    const size_t smem = (size_t)ROWS * KSTR + (size_t)2 * TT * KSTR +
                        (size_t)ROWS * PSTR + (size_t)8 * ROWS * 4;
    cudaError_t s1, s2;
    if (MT == 1) {
      s1 = cudaFuncSetAttribute(mla_kernel<1, true>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem);
      s2 = cudaFuncSetAttribute(mla_kernel<1, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem);
    } else {
      s1 = cudaFuncSetAttribute(mla_kernel<2, true>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem);
      s2 = cudaFuncSetAttribute(mla_kernel<2, false>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem);
    }
    TVM_FFI_ICHECK(s1 == cudaSuccess && s2 == cudaSuccess)
        << "mla_decode_fp8_paged_sm90: cudaFuncSetAttribute failed";
  }
  cudaError_t s3 = cudaFuncSetAttribute(
      mla_kernel_wg<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smw);
  cudaError_t s4 = cudaFuncSetAttribute(
      mla_kernel_wg<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      (int)smw);
  TVM_FFI_ICHECK(s3 == cudaSuccess && s4 == cudaSuccess)
      << "mla_decode_fp8_paged_sm90: cudaFuncSetAttribute (wg) failed";
}

// Workspace bytes the dispatcher must plan-allocate for a
// (batch, q_len, pages_per_seq, total_pages) plan on `device_id`.
int64_t mla_decode_fp8_paged_sm90_plan_workspace_bytes(int64_t batch_size,
                                                       int64_t q_len,
                                                       int64_t pages_per_seq,
                                                       int64_t total_pages,
                                                       int64_t device_id) {
  TVM_FFI_ICHECK_GE(device_id, 0);
  TVM_FFI_ICHECK_LT(device_id, kMaxDevices);
  TVM_FFI_ICHECK_GT(g_sm_count[device_id], 0)
      << "mla_decode_fp8_paged_sm90: init(device_id) must run before "
         "plan_workspace_bytes";
  const MlaSplitPlan P = mla_decode_fp8_split_plan(
      static_cast<int>(batch_size), static_cast<int>(q_len),
      static_cast<int>(pages_per_seq), static_cast<int>(total_pages),
      g_sm_count[device_id]);
  return (int64_t)mla_decode_fp8_ws_bytes(P, static_cast<int>(batch_size));
}

// Launch the measured kernel sequence on the caller's current stream:
// mla_kernel<MT>/mla_kernel_wg over the verbatim split plan, then
// combine_kernel when the plan splits. No allocation, no sync, no attribute
// queries: CUDA-graph capturable.
void mla_decode_fp8_paged_sm90_run(TensorView q, TensorView kv_cache,
                                   TensorView block_table, TensorView seq_lens,
                                   TensorView q_descale, TensorView k_descale,
                                   double sm_scale, TensorView out,
                                   TensorView workspace) {
  CHECK_INPUT_AND_TYPE(q, dl_float8_e4m3fn);
  CHECK_INPUT_AND_TYPE(out, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(block_table, dl_int32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(q_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(k_descale, dl_float32);
  CHECK_INPUT_AND_TYPE(workspace, dl_uint8);
  CHECK_CUDA(kv_cache);
  CHECK_INPUT_TYPE(kv_cache, dl_float8_e4m3fn);
  CHECK_DIM(4, q);
  CHECK_DIM(4, out);
  CHECK_DIM(2, block_table);

  const int B = static_cast<int>(q.size(0));
  const int QL = static_cast<int>(q.size(1));
  const int pps = static_cast<int>(block_table.size(1));

  TVM_FFI_ICHECK_EQ(q.size(2), NH);
  TVM_FFI_ICHECK_EQ(q.size(3), DIMQ);
  TVM_FFI_ICHECK_EQ(out.size(0), B);
  TVM_FFI_ICHECK_EQ(out.size(1), QL);
  TVM_FFI_ICHECK_EQ(out.size(2), NH);
  TVM_FFI_ICHECK_EQ(out.size(3), CKV);
  TVM_FFI_ICHECK_EQ(block_table.size(0), B);
  TVM_FFI_ICHECK_GE(seq_lens.numel(), B);
  TVM_FFI_ICHECK_GE(q_descale.numel(), 1);
  TVM_FFI_ICHECK_GE(k_descale.numel(), 1);
  // packed paged cache: (num_pages, 64, 576) or (num_pages, 64, 1, 576),
  // 576-byte contiguous rows, page-contiguous
  const int kv_dim = kv_cache.dim();
  TVM_FFI_ICHECK(kv_dim == 3 || kv_dim == 4)
      << "kv_cache must be (pages, 64, [1,] 576)";
  TVM_FFI_ICHECK_EQ(kv_cache.size(1), PAGE);
  TVM_FFI_ICHECK_EQ(kv_cache.size(kv_dim - 1), DIMQ);
  if (kv_dim == 4) TVM_FFI_ICHECK_EQ(kv_cache.size(2), 1);
  TVM_FFI_ICHECK_EQ(kv_cache.stride(kv_dim - 1), 1);
  TVM_FFI_ICHECK_EQ(kv_cache.stride(1), DIMQ);
  TVM_FFI_ICHECK_EQ(kv_cache.stride(0), (int64_t)PAGE * DIMQ)
      << "kv pages must be 64x576 contiguous bytes";
  const int total_pages = static_cast<int>(kv_cache.size(0));

  const int device_id = q.device().device_id;
  TVM_FFI_ICHECK_GE(device_id, 0);
  TVM_FFI_ICHECK_LT(device_id, kMaxDevices);
  const int sm_count = g_sm_count[device_id];
  TVM_FFI_ICHECK_GT(sm_count, 0)
      << "mla_decode_fp8_paged_sm90: init(device_id) must run before run";

  const MlaSplitPlan P =
      mla_decode_fp8_split_plan(B, QL, pps, total_pages, sm_count);
  const size_t need = mla_decode_fp8_ws_bytes(P, B);
  TVM_FFI_ICHECK_GE((size_t)workspace.numel(), need)
      << "workspace too small: plan with plan_workspace_bytes(" << B << ", "
      << QL << ", " << pps << ", " << total_pages << ", " << device_id << ")";

  CHECK_DEVICE(q, kv_cache);
  CHECK_DEVICE(q, block_table);
  CHECK_DEVICE(q, seq_lens);
  CHECK_DEVICE(q, q_descale);
  CHECK_DEVICE(q, k_descale);
  CHECK_DEVICE(q, out);
  CHECK_DEVICE(q, workspace);

  ffi::CUDADeviceGuard device_guard(device_id);
  const cudaStream_t st = get_stream(q.device());

  uint8_t* Opp = nullptr;
  float* Mpp = nullptr;
  float* Lpp = nullptr;
  if (P.ns > 1) {
    const size_t nrow = (size_t)B * P.ns * P.rows_tot;
    const size_t obytes = nrow * CKV;
    char* p = static_cast<char*>(workspace.data_ptr());
    Opp = reinterpret_cast<uint8_t*>(p);
    Mpp = reinterpret_cast<float*>(p + obytes);
    Lpp = Mpp + nrow;
  }

  const int ROWS = 16 * P.MT;
  const size_t smem = (size_t)ROWS * KSTR + (size_t)2 * TT * KSTR +
                      (size_t)ROWS * PSTR + (size_t)8 * ROWS * 4;
  dim3 grid(P.nblocks, 1, P.nqc);
  const int nthr = 128 * P.MT;

  const uint8_t* qp = static_cast<const uint8_t*>(q.data_ptr());
  const uint8_t* kp = static_cast<const uint8_t*>(kv_cache.data_ptr());
  const int* btp = static_cast<const int*>(block_table.data_ptr());
  const int* slp = static_cast<const int*>(seq_lens.data_ptr());
  const float* dqp = static_cast<const float*>(q_descale.data_ptr());
  const float* dkp = static_cast<const float*>(k_descale.data_ptr());
  __nv_bfloat16* op = static_cast<__nv_bfloat16*>(out.data_ptr());

#define LAUNCH(M, H)                                                          \
  mla_kernel<M, H><<<grid, nthr, smem, st>>>(                                 \
      qp, kp, btp, slp, dqp, dkp, (float)sm_scale, op, Opp, Mpp, Lpp, B, QL,  \
      pps, P.ns, P.pps_split, P.QPC)
#define LAUNCHW(H)                                                            \
  mla_kernel_wg<H><<<grid, 256,                                               \
                     (size_t)5 * PGB + (size_t)48 * PSTR + (size_t)4 * 48 * 4,\
                     st>>>(qp, kp, btp, slp, dqp, dkp, (float)sm_scale, op,   \
                           Opp, Mpp, Lpp, B, QL, pps, P.ns, P.pps_split,      \
                           P.QPC)
  // Unit-max probability quantization on long splits keeps packed-FP16 PV
  // below overflow while preserving the same softmax ratio in the epilogue.
  const bool half_acc = true;
  if (half_acc) {
    if (P.MT == 1)
      LAUNCH(1, true);
    else if (P.MT == 2)
      LAUNCH(2, true);
    else
      LAUNCHW(true);
  } else {
    if (P.MT == 1)
      LAUNCH(1, false);
    else if (P.MT == 2)
      LAUNCH(2, false);
    else
      LAUNCHW(false);
  }
#undef LAUNCHW
#undef LAUNCH

  if (P.ns > 1) {
    const int nrows = B * P.rows_tot;
    const int group = (nrows >= sm_count) ? 4 : 1;
    dim3 cgrid(P.rows_tot, (CKV / 128) / group, B);
    if (group == 4)
      combine_kernel<4><<<cgrid, 32 * CWARPS, P.ns * sizeof(float), st>>>(
          Opp, Mpp, Lpp, dkp, slp, op, P.ns, P.rows_tot, P.pps_split);
    else
      combine_kernel<1><<<cgrid, 32 * CWARPS, P.ns * sizeof(float), st>>>(
          Opp, Mpp, Lpp, dkp, slp, op, P.ns, P.rows_tot, P.pps_split);
  }
  cudaError_t status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "mla_decode_fp8_paged_sm90 launch failed: "
      << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mla_decode_fp8_paged_sm90_run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, mla_decode_fp8_paged_sm90_init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan_workspace_bytes,
                              mla_decode_fp8_paged_sm90_plan_workspace_bytes);
