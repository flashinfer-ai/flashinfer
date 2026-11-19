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
 *
 * SM90 W4A16 latent-MoE routed-expert block (mxfp4 group-32 weights, bf16
 * activations), JIT module fused_moe_w4a16_mxfp4_latent_sm90.
 *
 * Fixed-geometry specialist for the latent routed-expert block
 * E_local = 56, top_k = 16, latent hidden L = 3584, expert intermediate
 * I = 3072, gate|up-stacked fc1, "situ" activation
 * (4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)), global expert ids consumed
 * as-is (ids >= E_local contribute nothing; the output is the rank-partial
 * combine).  Weights are mxfp4-pack-quantized in the compressed-tensors
 * convention: two e2m1 nibbles per uint8 with the LOW nibble at the even
 * K index, and ue8m0 uint8 group-32 scales (scale = 2**(u8 - 127)).
 *
 * Two launch paths picked per call from the token count T:
 *   T <= 64  : decode path, one cooperative kernel
 *              (cudaLaunchCooperativeKernel, occupancy-derived grid);
 *   T  > 64  : prefill path, routing kernel + two tensor-core GEMM stages.
 * The guard, the CUDA-graph rules and the kill switch live in
 * flashinfer/fused_moe/w4a16_mxfp4_moe.py; this file validates nothing.
 * The device code and the decode/prefill launch heuristic are kept verbatim
 * from the measured artifact; only the host binding differs.
 */
//
// Framework-independent device code: the kernels, device helpers and their
// launch-geometry constants/macros, relocated verbatim from the former csrc
// kernel TU. Included by exactly one binding TU,
// csrc/fused_moe_w4a16_mxfp4_latent_sm90.cu,
// which keeps the host-side launch glue and dispatch. File-scope geometry
// macros/constants are by design: never include two of these headers in one
// translation unit.
#pragma once
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define EL 56
#define KTOP 16
#define LH 3584
#define IH 3072
#define NT1 64
#define NT2 64
#define PNT2 128
#define BN1 2
#define NTILES1 (IH / NT1)
#define NTILES2 (LH / NT2)
#define PNTILES2 (LH / PNT2)
#define NGRP1 (NTILES1 / BN1)
#define DGRID 264
#define PMT 8

__device__ __forceinline__ uint32_t prm(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ uint32_t prm1404(uint32_t a) {
  uint32_t d;
  asm("prmt.b32 %0, %1, 0, 0x1404;" : "=r"(d) : "r"(a));
  return d;
}

__device__ __forceinline__ uint32_t prm3424(uint32_t a) {
  uint32_t d;
  asm("prmt.b32 %0, %1, 0, 0x3424;" : "=r"(d) : "r"(a));
  return d;
}

__device__ __forceinline__ uint32_t lop3ea(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ float ue8m0(uint32_t e) {
  // dq8 contributes 2^-14 and staged activations contribute 2^7, so absorb
  // the remaining 2^7 here.  This also keeps the e==0 case normal.
  uint32_t b;
  asm("mad.lo.u32 %0, %1, 8388608, 58720256;" : "=r"(b) : "r"(e));
  return __uint_as_float(b);
}

// Eight E2M1 nibbles -> four half2 values scaled by 2^-14.  The emitted
// order is (n0,n2),(n4,n6),(n1,n3),(n5,n7), matched by cvt8 below.
__device__ __forceinline__ void dq8(uint32_t w, __half2& a, __half2& b, __half2& c, __half2& d) {
  uint32_t t = w & 0x0F0F0F0Fu, s = w & 0x08080808u, be, m;
  asm("mad.lo.u32 %0, %1, 14, 0;" : "=r"(m) : "r"(s));
  asm("mad.lo.u32 %0, %1, 2, %2;" : "=r"(be) : "r"(t), "r"(m));
  uint32_t bo = lop3ea(w, 0x80808080u, (w >> 3) & 0x0E0E0E0Eu);
  uint32_t r;
  r = prm1404(be);
  a = *reinterpret_cast<__half2*>(&r);
  r = prm3424(be);
  b = *reinterpret_cast<__half2*>(&r);
  r = prm1404(bo);
  c = *reinterpret_cast<__half2*>(&r);
  r = prm3424(bo);
  d = *reinterpret_cast<__half2*>(&r);
}

// Eight BF16 activations -> eight FP16 values scaled by 2^7 and permuted to
// the pairing emitted by dq8.  The scaled values are saturated to the FP16
// finite range: an activation with |x| >= 512 would otherwise convert to inf
// (512 * 2^7 = 65536 > 65504) and poison the fp16 dot products with NaN
// through inf * 0 weights.
__device__ __forceinline__ uint4 cvt8(uint4 v) {
  const __nv_bfloat162* bp = reinterpret_cast<const __nv_bfloat162*>(&v);
  uint32_t t[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float2 f = __bfloat1622float2(bp[i]);
    f.x = fminf(fmaxf(f.x * 128.f, -65504.f), 65504.f);
    f.y = fminf(fmaxf(f.y * 128.f, -65504.f), 65504.f);
    __half2 h = __float22half2_rn(f);
    t[i] = *reinterpret_cast<uint32_t*>(&h);
  }
  uint4 o;
  o.x = prm(t[0], t[1], 0x5410u);
  o.y = prm(t[2], t[3], 0x5410u);
  o.z = prm(t[0], t[1], 0x7632u);
  o.w = prm(t[2], t[3], 0x7632u);
  return o;
}

__device__ __forceinline__ int hperm(int c) { return (c & ~7) | ((c & 1) << 2) | ((c & 7) >> 1); }

// Transpose each 64-uint4 K slice from [lane][half] to [half][lane].  This
// makes the two LDS.128 warp reads below conflict-free on Hopper shared memory.
__device__ __forceinline__ int xswz(int i) { return (i & ~63) | ((i & 1) << 5) | ((i & 63) >> 1); }

template <bool ZERO_OUT>
__global__ void route_k(const int* __restrict__ ids, const float* __restrict__ tw, int T, int MT,
                        int* __restrict__ eoff, int* __restrict__ moff, int* __restrict__ mex,
                        int* __restrict__ ptok, float* __restrict__ pw,
                        __nv_bfloat16* __restrict__ out) {
  __shared__ int cnt[64];
  __shared__ int cur[64];
  int tid = threadIdx.x;
  if (ZERO_OUT) {
    uint4* out4 = reinterpret_cast<uint4*>(out);
    int n4 = T * (LH / 8);
    for (int i = tid; i < n4; i += blockDim.x) out4[i] = make_uint4(0, 0, 0, 0);
  }
  if (tid < 64) cnt[tid] = 0;
  __syncthreads();
  int n = T * KTOP;
  for (int i = tid; i < n; i += blockDim.x) {
    int e = ids[i];
    if ((unsigned)e < (unsigned)EL) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int s = 0, ms = 0;
    for (int e = 0; e < EL; ++e) {
      int c = cnt[e];
      int nm = (c + MT - 1) / MT;
      eoff[e] = s;
      cur[e] = s;
      moff[e] = ms;
      for (int q = 0; q < nm; ++q) mex[ms + q] = e;
      s += c;
      ms += nm;
    }
    eoff[EL] = s;
    moff[EL] = ms;
  }
  __syncthreads();
  for (int i = tid; i < n; i += blockDim.x) {
    int e = ids[i];
    if ((unsigned)e < (unsigned)EL) {
      int p = atomicAdd(&cur[e], 1);
      ptok[p] = i >> 4;
      pw[p] = tw[i];
    }
  }
}

// Decode hits are independent, so expert-grouped offsets are unnecessary.
// Compact local slots in one pass and publish only the total hit count.
__global__ void route_dec_k(const int* __restrict__ ids, const float* __restrict__ tw, int T,
                            int* __restrict__ moff, int* __restrict__ mex, float* __restrict__ pw) {
  __shared__ int count;
  const int tid = threadIdx.x;
  if (tid == 0) count = 0;
  __syncthreads();
  const int n = T * KTOP;
  for (int i = tid; i < n; i += blockDim.x) {
    const int e = ids[i];
    if ((unsigned)e < (unsigned)EL) {
      const int p = atomicAdd(&count, 1);
      mex[p] = e | ((i >> 4) << 6);
      pw[p] = tw[i];
    }
  }
  __syncthreads();
  if (tid == 0) moff[EL] = count;
}

template <int MT>
__global__ __launch_bounds__(512) void fc1_k(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ wp,
    const uint8_t* __restrict__ ws, const int* __restrict__ eoff, const int* __restrict__ moff,
    const int* __restrict__ mex, const int* __restrict__ ptok, __half* __restrict__ hbuf) {
  extern __shared__ char smem[];
  __half* xs = reinterpret_cast<__half*>(smem);
  __shared__ int smoff[64], seoff[64];
  __shared__ float gs[NT1][MT], us[NT1][MT];

  int tid = threadIdx.x;
  if (tid <= EL) {
    smoff[tid] = moff[tid];
    seoff[tid] = eoff[tid];
  }
  __syncthreads();
  int Wtot = smoff[EL] * NGRP1;
  int lane = tid & 31, warp = tid >> 5;

  for (int wi = blockIdx.x; wi < Wtot; wi += gridDim.x) {
    int mg = wi / NGRP1;
    int ng = wi - mg * NGRP1;
    int e = mex[mg];
    int m0 = (mg - smoff[e]) * MT;
    int cnte = seoff[e + 1] - seoff[e];
    int rows = cnte - m0;
    if (rows > MT) rows = MT;
    int p0 = seoff[e] + m0;

    __syncthreads();
    for (int r = 0; r < rows; ++r) {
      const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)ptok[p0 + r] * LH);
      uint4* dst = reinterpret_cast<uint4*>(xs + r * LH);
      for (int i = tid; i < LH / 8; i += 512) dst[i] = cvt8(src[i]);
    }
    __syncthreads();

    // Two adjacent output tiles share the same gathered activation staging.
#pragma unroll
    for (int bn = 0; bn < BN1; ++bn) {
      int n0 = (ng * BN1 + bn) * NT1;
      for (int task = warp; task < 2 * NT1; task += 16) {
        int row = task < NT1 ? n0 + task : IH + n0 + task - NT1;
        const uint32_t* wrow =
            reinterpret_cast<const uint32_t*>(wp + ((size_t)e * (2 * IH) + row) * (LH / 2));
        const uint8_t* srow = ws + ((size_t)e * (2 * IH) + row) * (LH / 32);
        float acc[MT];
#pragma unroll
        for (int m = 0; m < MT; ++m) acc[m] = 0.f;
#pragma unroll 2
        for (int it = 0; it < LH / 256; ++it) {
          int j = lane + 32 * it;
          uint32_t wv = wrow[j];
          float sc = ue8m0(srow[j >> 2]);
          __half2 d0, d1, d2, d3;
          dq8(wv, d0, d1, d2, d3);
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            if (m < rows) {
              const __half2* xp = reinterpret_cast<const __half2*>(xs + m * LH + j * 8);
              __half2 p = __hmul2(d0, xp[0]);
              p = __hfma2(d1, xp[1], p);
              p = __hfma2(d2, xp[2], p);
              p = __hfma2(d3, xp[3], p);
              acc[m] = fmaf(sc, __low2float(p) + __high2float(p), acc[m]);
            }
          }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_down_sync(0xffffffffu, acc[m], o);
        }
        if (lane == 0) {
#pragma unroll
          for (int m = 0; m < MT; ++m) {
            if (task < NT1)
              gs[task][m] = acc[m];
            else
              us[task - NT1][m] = acc[m];
          }
        }
      }
      __syncthreads();
      for (int idx = tid; idx < NT1 * MT; idx += 512) {
        int m = idx / NT1, c = idx - m * NT1;
        if (m < rows) {
          float g = gs[c][m], u = us[c][m];
          float a = 4.f * tanhf(g * 0.25f) * (1.f / (1.f + __expf(-g)));
          float b = 25.f * tanhf(u * 0.04f);
          float h = __bfloat162float(__float2bfloat16(a * b));
          hbuf[(size_t)(p0 + m) * IH + hperm(n0 + c)] = __float2half(h * 128.f);
        }
      }
      if (bn + 1 < BN1) __syncthreads();
    }
  }
}

template <int MT>
__global__ __launch_bounds__(512) void fc2_k(
    const __half* __restrict__ hbuf, const uint8_t* __restrict__ wp, const uint8_t* __restrict__ ws,
    const int* __restrict__ eoff, const int* __restrict__ moff, const int* __restrict__ mex,
    const int* __restrict__ ptok, const float* __restrict__ pw, __nv_bfloat16* __restrict__ out) {
  extern __shared__ char smem[];
  __half* xs = reinterpret_cast<__half*>(smem);
  __shared__ int smoff[64], seoff[64];

  int tid = threadIdx.x;
  if (tid <= EL) {
    smoff[tid] = moff[tid];
    seoff[tid] = eoff[tid];
  }
  __syncthreads();
  int Wtot = smoff[EL] * PNTILES2;
  int lane = tid & 31, warp = tid >> 5;

  for (int wi = blockIdx.x; wi < Wtot; wi += gridDim.x) {
    int mg = wi / PNTILES2;
    int nt = wi - mg * PNTILES2;
    int e = mex[mg];
    int m0 = (mg - smoff[e]) * MT;
    int cnte = seoff[e + 1] - seoff[e];
    int rows = cnte - m0;
    if (rows > MT) rows = MT;
    int p0 = seoff[e] + m0;
    int n0 = nt * PNT2;

    __syncthreads();
    for (int r = 0; r < rows; ++r) {
      const uint4* src = reinterpret_cast<const uint4*>(hbuf + (size_t)(p0 + r) * IH);
      uint4* dst = reinterpret_cast<uint4*>(xs + r * IH);
      for (int i = tid; i < IH / 8; i += 512) dst[i] = src[i];
    }
    __syncthreads();

    for (int task = warp; task < PNT2; task += 16) {
      int row = n0 + task;
      const uint32_t* wrow =
          reinterpret_cast<const uint32_t*>(wp + ((size_t)e * LH + row) * (IH / 2));
      const uint8_t* srow = ws + ((size_t)e * LH + row) * (IH / 32);
      float acc[MT];
#pragma unroll
      for (int m = 0; m < MT; ++m) acc[m] = 0.f;
#pragma unroll 2
      for (int it = 0; it < IH / 256; ++it) {
        int j = lane + 32 * it;
        uint32_t wv = wrow[j];
        float sc = ue8m0(srow[j >> 2]);
        __half2 d0, d1, d2, d3;
        dq8(wv, d0, d1, d2, d3);
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          if (m < rows) {
            const __half2* xp = reinterpret_cast<const __half2*>(xs + m * IH + j * 8);
            __half2 p = __hmul2(d0, xp[0]);
            p = __hfma2(d1, xp[1], p);
            p = __hfma2(d2, xp[2], p);
            p = __hfma2(d3, xp[3], p);
            acc[m] = fmaf(sc, __low2float(p) + __high2float(p), acc[m]);
          }
        }
      }
#pragma unroll
      for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[m] += __shfl_down_sync(0xffffffffu, acc[m], o);
      }
      if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MT; ++m) {
          if (m < rows) {
            float v = acc[m] * pw[p0 + m];
            atomicAdd(out + (size_t)ptok[p0 + m] * LH + row, __float2bfloat16(v));
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Prefill tensor-core path.  At prefill the per-expert M is tens-to-hundreds of
// rows, so the FFMA/shared-memory shape used for decode is shared-bandwidth
// bound (one LDS.128 per 8 MACs per row).  Here each warp keeps its C tile in
// registers and feeds mma.m16n8k16 from ldmatrix fragments: weights are
// dequantized straight to BF16 in registers and staged once per K chunk.
// ---------------------------------------------------------------------------
#define MSTR 72  // shared row stride in bf16 (64 + 8 pad)
#define MMM 64   // M rows per CTA tile
#define MNT 128  // N rows per CTA tile (per gate/up half for fc1)
#define MNW 16   // N rows per warp

__device__ __forceinline__ uint32_t smaddr(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void ldm4(uint32_t a, uint32_t* r) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mmaf(float* d, const uint32_t* a, const uint32_t* b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t mulbf2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// 32 e2m1 nibbles (one ue8m0 group) -> 16 packed bf16x2 registers.  The
// magnitude lookup is two PRMTs against the eight E2M1 bit patterns; the sign
// bit is re-inserted with a third PRMT and the group scale folds into one
// bf16x2 multiply (exact: e2m1 needs 2 mantissa bits, the scale is 2^k).
__device__ __forceinline__ void dq32(const uint4& w, uint32_t su, uint32_t* o) {
  const uint32_t sc2 = su * 0x00800080u;
  const uint32_t* wv = reinterpret_cast<const uint32_t*>(&w);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint32_t v = wv[i];
    uint32_t m = v & 0x77777777u;
    uint32_t hiA = prm(0x3F3F3F00u, 0x40404040u, m);
    uint32_t loA = prm(0xC0800000u, 0xC0804000u, m);
    uint32_t m2 = m >> 16;
    uint32_t hiB = prm(0x3F3F3F00u, 0x40404040u, m2);
    uint32_t loB = prm(0xC0800000u, 0xC0804000u, m2);
    uint32_t v4 = v << 4;
    o[4 * i + 0] = mulbf2(prm(loA, hiA, 0x5140u) | (prm(v4, v, 0x4000u) & 0x80008000u), sc2);
    o[4 * i + 1] = mulbf2(prm(loA, hiA, 0x7362u) | (prm(v4, v, 0x5010u) & 0x80008000u), sc2);
    o[4 * i + 2] = mulbf2(prm(loB, hiB, 0x5140u) | (prm(v4, v, 0x6020u) & 0x80008000u), sc2);
    o[4 * i + 3] = mulbf2(prm(loB, hiB, 0x7362u) | (prm(v4, v, 0x7030u) & 0x80008000u), sc2);
  }
}

__global__ __launch_bounds__(256) void fc1_m_k(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ wp,
    const uint8_t* __restrict__ ws, const int* __restrict__ eoff, const int* __restrict__ moff,
    const int* __restrict__ mex, const int* __restrict__ ptok, __nv_bfloat16* __restrict__ hbuf) {
  extern __shared__ char smem[];
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Bs = As + MMM * MSTR;  // 2*MNT rows (gate then up)
  __shared__ int sptok[MMM];
  __shared__ int smoff[64], seoff[64];

  const int tid = threadIdx.x;
  if (tid <= EL) {
    smoff[tid] = moff[tid];
    seoff[tid] = eoff[tid];
  }
  __syncthreads();
  const int NT = IH / MNT;
  const int Wtot = smoff[EL] * NT;
  const int lane = tid & 31, warp = tid >> 5;
  const int nbase = warp * MNW;
  const int q = lane >> 3, l7 = lane & 7;
  // ldmatrix source offsets (constant across the K loop apart from kk/mb).
  const int arow = (q & 1) * 8 + l7;
  const int acol = ((q >> 1) & 1) * 8;
  const int brow = ((q >> 1) & 1) * 8 + l7;
  const int bcol = (q & 1) * 8;

  for (int wi = blockIdx.x; wi < Wtot; wi += gridDim.x) {
    const int mg = wi / NT, nt = wi - mg * NT;
    const int e = mex[mg];
    const int m0 = (mg - smoff[e]) * MMM;
    const int cnte = seoff[e + 1] - seoff[e];
    int rows = cnte - m0;
    if (rows > MMM) rows = MMM;
    const int p0 = seoff[e] + m0;
    const int n0 = nt * MNT;

    __syncthreads();
    if (tid < MMM) sptok[tid] = ptok[p0 + (tid < rows ? tid : rows - 1)];

    const int grow = (tid < MNT) ? (n0 + tid) : (IH + n0 + tid - MNT);
    const uint8_t* wrow = wp + ((size_t)e * (2 * IH) + grow) * (LH / 2);
    const uint8_t* srow = ws + ((size_t)e * (2 * IH) + grow) * (LH / 32);
    __nv_bfloat16* bdst = Bs + tid * MSTR;

    float ag[4][2][4], au[4][2][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          ag[i][j][c] = 0.f;
          au[i][j][c] = 0.f;
        }

    const int mblk = (rows + 15) >> 4;
    const int ar0 = tid >> 3, ac0 = (tid & 7) * 8;
    const int ar1 = (tid + 256) >> 3, ac1 = ((tid + 256) & 7) * 8;
    __syncthreads();

    // One K chunk of weights/activations is held in registers so the global
    // loads for chunk i+1 issue before the MMAs of chunk i.
    uint4 pw0, pw1, pa0, pa1;
    uint32_t ps0, ps1;
    {
      const uint4* wsrc = reinterpret_cast<const uint4*>(wrow);
      pw0 = wsrc[0];
      pw1 = wsrc[1];
      ps0 = srow[0];
      ps1 = srow[1];
      pa0 = *reinterpret_cast<const uint4*>(x + (size_t)sptok[ar0] * LH + ac0);
      pa1 = *reinterpret_cast<const uint4*>(x + (size_t)sptok[ar1] * LH + ac1);
    }

    for (int k0 = 0; k0 < LH; k0 += 64) {
      {
        __align__(16) uint32_t o[16];
        dq32(pw0, ps0, o);
        uint4* d0 = reinterpret_cast<uint4*>(bdst);
#pragma unroll
        for (int i = 0; i < 4; ++i) d0[i] = reinterpret_cast<const uint4*>(o)[i];
        dq32(pw1, ps1, o);
        uint4* d1 = reinterpret_cast<uint4*>(bdst + 32);
#pragma unroll
        for (int i = 0; i < 4; ++i) d1[i] = reinterpret_cast<const uint4*>(o)[i];
      }
      *reinterpret_cast<uint4*>(As + ar0 * MSTR + ac0) = pa0;
      *reinterpret_cast<uint4*>(As + ar1 * MSTR + ac1) = pa1;
      __syncthreads();
      const int kn = k0 + 64;
      if (kn < LH) {
        const uint4* wsrc = reinterpret_cast<const uint4*>(wrow + (kn >> 1));
        pw0 = wsrc[0];
        pw1 = wsrc[1];
        ps0 = srow[kn >> 5];
        ps1 = srow[(kn >> 5) + 1];
        pa0 = *reinterpret_cast<const uint4*>(x + (size_t)sptok[ar0] * LH + kn + ac0);
        pa1 = *reinterpret_cast<const uint4*>(x + (size_t)sptok[ar1] * LH + kn + ac1);
      }

#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t bg[4], bu[4];
        ldm4(smaddr(Bs + (nbase + brow) * MSTR + kk * 16 + bcol), bg);
        ldm4(smaddr(Bs + (MNT + nbase + brow) * MSTR + kk * 16 + bcol), bu);
#pragma unroll
        for (int mb = 0; mb < 4; ++mb) {
          if (mb < mblk) {
            uint32_t af[4];
            ldm4(smaddr(As + (mb * 16 + arow) * MSTR + kk * 16 + acol), af);
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
              mmaf(ag[mb][nb], af, bg + nb * 2);
              mmaf(au[mb][nb], af, bu + nb * 2);
            }
          }
        }
      }
      __syncthreads();
    }

    // situ epilogue: both halves live in the same registers, no exchange.
#pragma unroll
    for (int mb = 0; mb < 4; ++mb) {
      if (mb < mblk) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = mb * 16 + (lane >> 2) + h * 8;
          if (m < rows) {
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
              const int n = n0 + nbase + nb * 8 + (lane & 3) * 2;
              float g0 = ag[mb][nb][h * 2], g1 = ag[mb][nb][h * 2 + 1];
              float u0 = au[mb][nb][h * 2], u1 = au[mb][nb][h * 2 + 1];
              float a0 = 4.f * tanhf(g0 * 0.25f) * (1.f / (1.f + __expf(-g0)));
              float a1 = 4.f * tanhf(g1 * 0.25f) * (1.f / (1.f + __expf(-g1)));
              float b0 = 25.f * tanhf(u0 * 0.04f);
              float b1 = 25.f * tanhf(u1 * 0.04f);
              __nv_bfloat162 hv = __floats2bfloat162_rn(a0 * b0, a1 * b1);
              *reinterpret_cast<__nv_bfloat162*>(hbuf + (size_t)(p0 + m) * IH + n) = hv;
            }
          }
        }
      }
    }
  }
}

__global__ __launch_bounds__(256) void fc2_m_k(
    const __nv_bfloat16* __restrict__ hbuf, const uint8_t* __restrict__ wp,
    const uint8_t* __restrict__ ws, const int* __restrict__ eoff, const int* __restrict__ moff,
    const int* __restrict__ mex, const int* __restrict__ ptok, const float* __restrict__ pw,
    __nv_bfloat16* __restrict__ out) {
  extern __shared__ char smem[];
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* Bs = As + MMM * MSTR;  // 2*MNT rows
  __shared__ int smoff[64], seoff[64];

  const int tid = threadIdx.x;
  if (tid <= EL) {
    smoff[tid] = moff[tid];
    seoff[tid] = eoff[tid];
  }
  __syncthreads();
  const int NT = LH / (2 * MNT);
  const int Wtot = smoff[EL] * NT;
  const int lane = tid & 31, warp = tid >> 5;
  const int nbase = warp * MNW;
  const int q = lane >> 3, l7 = lane & 7;
  const int arow = (q & 1) * 8 + l7;
  const int acol = ((q >> 1) & 1) * 8;
  const int brow = ((q >> 1) & 1) * 8 + l7;
  const int bcol = (q & 1) * 8;
  const int ar0 = tid >> 3, ac0 = (tid & 7) * 8;
  const int ar1 = (tid + 256) >> 3, ac1 = ((tid + 256) & 7) * 8;

  for (int wi = blockIdx.x; wi < Wtot; wi += gridDim.x) {
    const int mg = wi / NT, nt = wi - mg * NT;
    const int e = mex[mg];
    const int m0 = (mg - smoff[e]) * MMM;
    const int cnte = seoff[e + 1] - seoff[e];
    int rows = cnte - m0;
    if (rows > MMM) rows = MMM;
    const int p0 = seoff[e] + m0;
    const int n0 = nt * (2 * MNT);

    const uint8_t* wrow = wp + ((size_t)e * LH + n0 + tid) * (IH / 2);
    const uint8_t* srow = ws + ((size_t)e * LH + n0 + tid) * (IH / 32);
    __nv_bfloat16* bdst = Bs + tid * MSTR;
    const int arc0 = ar0 < rows ? ar0 : rows - 1;
    const int arc1 = ar1 < rows ? ar1 : rows - 1;

    float a0[4][2][4], a1[4][2][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          a0[i][j][c] = 0.f;
          a1[i][j][c] = 0.f;
        }

    const int mblk = (rows + 15) >> 4;
    __syncthreads();

    uint4 pw0, pw1, pa0, pa1;
    uint32_t ps0, ps1;
    {
      const uint4* wsrc = reinterpret_cast<const uint4*>(wrow);
      pw0 = wsrc[0];
      pw1 = wsrc[1];
      ps0 = srow[0];
      ps1 = srow[1];
      pa0 = *reinterpret_cast<const uint4*>(hbuf + (size_t)(p0 + arc0) * IH + ac0);
      pa1 = *reinterpret_cast<const uint4*>(hbuf + (size_t)(p0 + arc1) * IH + ac1);
    }

    for (int k0 = 0; k0 < IH; k0 += 64) {
      {
        __align__(16) uint32_t o[16];
        dq32(pw0, ps0, o);
        uint4* d0 = reinterpret_cast<uint4*>(bdst);
#pragma unroll
        for (int i = 0; i < 4; ++i) d0[i] = reinterpret_cast<const uint4*>(o)[i];
        dq32(pw1, ps1, o);
        uint4* d1 = reinterpret_cast<uint4*>(bdst + 32);
#pragma unroll
        for (int i = 0; i < 4; ++i) d1[i] = reinterpret_cast<const uint4*>(o)[i];
      }
      *reinterpret_cast<uint4*>(As + ar0 * MSTR + ac0) = pa0;
      *reinterpret_cast<uint4*>(As + ar1 * MSTR + ac1) = pa1;
      __syncthreads();
      const int kn = k0 + 64;
      if (kn < IH) {
        const uint4* wsrc = reinterpret_cast<const uint4*>(wrow + (kn >> 1));
        pw0 = wsrc[0];
        pw1 = wsrc[1];
        ps0 = srow[kn >> 5];
        ps1 = srow[(kn >> 5) + 1];
        pa0 = *reinterpret_cast<const uint4*>(hbuf + (size_t)(p0 + arc0) * IH + kn + ac0);
        pa1 = *reinterpret_cast<const uint4*>(hbuf + (size_t)(p0 + arc1) * IH + kn + ac1);
      }

#pragma unroll
      for (int kk = 0; kk < 4; ++kk) {
        uint32_t b0[4], b1[4];
        ldm4(smaddr(Bs + (nbase + brow) * MSTR + kk * 16 + bcol), b0);
        ldm4(smaddr(Bs + (MNT + nbase + brow) * MSTR + kk * 16 + bcol), b1);
#pragma unroll
        for (int mb = 0; mb < 4; ++mb) {
          if (mb < mblk) {
            uint32_t af[4];
            ldm4(smaddr(As + (mb * 16 + arow) * MSTR + kk * 16 + acol), af);
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
              mmaf(a0[mb][nb], af, b0 + nb * 2);
              mmaf(a1[mb][nb], af, b1 + nb * 2);
            }
          }
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int mb = 0; mb < 4; ++mb) {
      if (mb < mblk) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int m = mb * 16 + (lane >> 2) + h * 8;
          if (m < rows) {
            const float sw = pw[p0 + m];
            const size_t ob = (size_t)ptok[p0 + m] * LH;
#pragma unroll
            for (int nb = 0; nb < 2; ++nb) {
              const int n = n0 + nbase + nb * 8 + (lane & 3) * 2;
              __nv_bfloat162 v0 =
                  __floats2bfloat162_rn(a0[mb][nb][h * 2] * sw, a0[mb][nb][h * 2 + 1] * sw);
              atomicAdd(reinterpret_cast<__nv_bfloat162*>(out + ob + n), v0);
              __nv_bfloat162 v1 =
                  __floats2bfloat162_rn(a1[mb][nb][h * 2] * sw, a1[mb][nb][h * 2 + 1] * sw);
              atomicAdd(reinterpret_cast<__nv_bfloat162*>(out + ob + n + MNT), v1);
            }
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 4x32 transpose within each 1024-element chunk.  Each fc2 lane then reads
// one naturally-coalesced uint4 from each quarter of a group-32 slice.
__device__ __forceinline__ int hpos(int c) {
  const int j = (c & 1023) >> 3;
  return (c & ~1023) + (((j & 3) * 32 + (j >> 2)) << 3) + (((c & 1) << 2) | ((c & 7) >> 1));
}
// Four-way transpose for the three full 1024-element fc1 chunks, followed by
// the original two-way layout for the final 512-element tail.
__device__ __forceinline__ int xswz4(int i) {
  if (i < 384) return (i & ~127) | ((i & 3) << 5) | ((i & 127) >> 2);
  const int k = i - 384;
  return 384 + ((k & 1) << 5) + (k >> 1);
}
// ---------------------------------------------------------------------------
// Decode specializations (T <= 64).  These dominate the weighted score.
//
// The round-2 shape issued one 32-bit weight load per lane per step, which left
// only ~1 KB of weight traffic in flight per warp and showed up in NCU as 36%
// of cycles with no eligible warp (long-scoreboard on the weight loads).  Here
// every lane pulls a uint2 (8 B = 16 e2m1 values), four independent rows per
// warp are advanced together, and the next K step is prefetched into registers
// while the current one is dequantized -- 8 outstanding 256 B warp requests.
// ---------------------------------------------------------------------------

// Dot of 16 e2m1 weights (one uint2) against 16 staged fp16 activations.
__device__ __forceinline__ float dot16(uint2 w, const uint4& xa, const uint4& xb) {
  const __half2* pa = reinterpret_cast<const __half2*>(&xa);
  const __half2* pb = reinterpret_cast<const __half2*>(&xb);
  __half2 d0, d1, d2, d3;
  dq8(w.x, d0, d1, d2, d3);
  __half2 p = __hmul2(d0, pa[0]);
  p = __hfma2(d1, pa[1], p);
  p = __hfma2(d2, pa[2], p);
  p = __hfma2(d3, pa[3], p);
  dq8(w.y, d0, d1, d2, d3);
  p = __hfma2(d0, pb[0], p);
  p = __hfma2(d1, pb[1], p);
  p = __hfma2(d2, pb[2], p);
  p = __hfma2(d3, pb[3], p);
  return __low2float(p) + __high2float(p);
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem), "l"(src) : "memory");
}

#define RCAP 1024  // decode path is T <= 64, so T*KTOP <= 1024

// Deterministic in-CTA compaction of the local routing slots.  Every CTA
// rebuilds the identical compact list (stable, slot-index order) from a block
// scan, which removes the separate single-CTA routing launch entirely.  The
// id stream is a few hundred bytes and lands in L2 after the first reader.
template <bool NEEDW>
__device__ __forceinline__ int route_build(const int* __restrict__ ids,
                                           const float* __restrict__ tw, int n, int* smeta,
                                           float* sw, int* wsum, int* ranges) {
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  int base = 0;
  for (int i0 = 0; i0 < n; i0 += 512) {
    const int i = i0 + tid;
    int e = 0;
    bool p = false;
    if (i < n) {
      e = ids[i];
      p = ((unsigned)e < (unsigned)EL);
    }
    const unsigned msk = __ballot_sync(0xffffffffu, p);
    if (lane == 0) wsum[warp] = __popc(msk);
    __syncthreads();
    int pre = 0, tot = 0;
    const int nw = min(16, (n - i0 + 31) >> 5);
    for (int w = 0; w < nw; ++w) {
      const int c = wsum[w];
      if (w < warp) pre += c;
      tot += c;
    }
    if (p) {
      const int pos = base + pre + __popc(msk & ((1u << lane) - 1u));
      smeta[pos] = e | ((i >> 4) << 6);
      if (NEEDW) sw[pos] = tw[i];
    }
    base += tot;
    if (ranges && i0 + 512 >= n && tid == 0) {
      const int w1 = base * 96;
      const int w2 = base * 56;
      ranges[0] = (int)blockIdx.x * w1 / (int)gridDim.x;
      ranges[1] = ((int)blockIdx.x + 1) * w1 / (int)gridDim.x;
      ranges[2] = (int)blockIdx.x * w2 / (int)gridDim.x;
      ranges[3] = ((int)blockIdx.x + 1) * w2 / (int)gridDim.x;
    }
    __syncthreads();
  }
  return base;
}

#define C1TILE 32           // fc1 columns per block (16 warps x 2 columns)
#define NT1D (IH / C1TILE)  // 96 column tiles
#define N2TILE 64           // fc2 rows per block (16 warps x 4 rows)
#define NT2D (LH / N2TILE)  // 56 row tiles

// fc1: warp owns columns c0 and c0+16, i.e. four weight rows (two gate, two
// up).  One staged activation fragment feeds all four dequant chains.  Each
// CTA takes a CONTIGUOUS span of (hit, column-tile) work so consecutive tiles
// almost always share one hit and the activation staging is paid once.
__global__ __launch_bounds__(512, 2) void fc1_dec_k(const __nv_bfloat16* __restrict__ x,
                                                    const uint8_t* __restrict__ wp,
                                                    const uint8_t* __restrict__ ws,
                                                    const int* __restrict__ ids,
                                                    __half* __restrict__ hbuf, int T,
                                                    __nv_bfloat16* __restrict__ out) {
  extern __shared__ char smem_raw[];
  __half* xs = reinterpret_cast<__half*>(smem_raw);
  int* smeta = reinterpret_cast<int*>(smem_raw + LH * 2);
  __shared__ int wsum[16];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  constexpr int WOFF[4] = {0, 16 * (LH / 2) / 8, IH * (LH / 2) / 8, (IH + 16) * (LH / 2) / 8};
  constexpr int SOFF[4] = {0, 16 * (LH / 32), IH * (LH / 32), (IH + 16) * (LH / 32)};

  uint4* out4 = reinterpret_cast<uint4*>(out);
  const int n4 = T * (LH / 8);
  for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += gridDim.x * blockDim.x)
    out4[i] = make_uint4(0, 0, 0, 0);

  const int nh = route_build<false>(ids, nullptr, T * KTOP, smeta, nullptr, wsum, nullptr);
  const int Wtot = nh * NT1D;
  const int begin = (int)(((long long)blockIdx.x * Wtot) / gridDim.x);
  const int end = (int)(((long long)(blockIdx.x + 1) * Wtot) / gridDim.x);
  int cur = -1;

  for (int wi = begin; wi < end; ++wi) {
    const int p0 = wi / NT1D;
    const int nt = wi - p0 * NT1D;
    const int meta = smeta[p0];
    const int e = meta & 63;
    const int c0 = nt * C1TILE + warp;

    if (p0 != cur) {
      __syncthreads();
      const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)(meta >> 6) * LH);
      uint4* dst = reinterpret_cast<uint4*>(xs);
      for (int i = tid; i < LH / 8; i += 512) dst[xswz(i)] = cvt8(src[i]);
      __syncthreads();
      cur = p0;
    }

    const size_t rb = (size_t)e * (2 * IH) + c0;
    const uint2* W = reinterpret_cast<const uint2*>(wp + rb * (LH / 2));
    const uint8_t* S = ws + rb * (LH / 32);

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    uint2 wq[2][4];
    uint32_t sq[2][4];
#pragma unroll
    for (int p = 0; p < 2; ++p)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        wq[p][q] = W[WOFF[q] + lane + 32 * p];
        sq[p][q] = S[SOFF[q] + 16 * p + (lane >> 1)];
      }
#pragma unroll
    for (int it = 0; it < LH / 512; ++it) {
      const uint4* xp = reinterpret_cast<const uint4*>(xs + it * 512);
      uint4 xa = xp[lane], xb = xp[32 + lane];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc[q] = fmaf(ue8m0(sq[it & 1][q]), dot16(wq[it & 1][q], xa, xb), acc[q]);
        if (it + 2 < LH / 512) {
          wq[it & 1][q] = W[WOFF[q] + lane + 32 * (it + 2)];
          sq[it & 1][q] = S[SOFF[q] + 16 * (it + 2) + (lane >> 1)];
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[q] += __shfl_down_sync(0xffffffffu, acc[q], o);
    }
    if (lane == 0) {
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        float g = acc[q], u = acc[q + 2];
        float a = 4.f * tanhf(g * 0.25f) * (1.f / (1.f + __expf(-g)));
        float b = 25.f * tanhf(u * 0.04f);
        float h = __bfloat162float(__float2bfloat16(a * b));
        hbuf[(size_t)p0 * IH + hperm(c0 + 16 * q)] = __float2half(h * 128.f);
      }
    }
  }
}

// fc2: warp owns four adjacent output rows inside a 64-row tile.  Adjacent
// ownership permits two aligned bf16x2 atomics in the combine epilogue.
__global__ __launch_bounds__(512, 2) void fc2_dec_k(const __half* __restrict__ hbuf,
                                                    const uint8_t* __restrict__ wp,
                                                    const uint8_t* __restrict__ ws,
                                                    const int* __restrict__ ids,
                                                    const float* __restrict__ tw, int T,
                                                    __nv_bfloat16* __restrict__ out) {
  extern __shared__ char smem_raw2[];
  __half* xs = reinterpret_cast<__half*>(smem_raw2);
  int* smeta = reinterpret_cast<int*>(smem_raw2 + IH * 2);
  float* sw = reinterpret_cast<float*>(smem_raw2 + IH * 2 + RCAP * 4);
  __shared__ int wsum[16];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;
  constexpr int WOFF = (IH / 2) / 8;
  constexpr int SOFF = IH / 32;

  const int nh = route_build<true>(ids, tw, T * KTOP, smeta, sw, wsum, nullptr);
  const int Wtot = nh * NT2D;
  const int begin = (int)(((long long)blockIdx.x * Wtot) / gridDim.x);
  const int end = (int)(((long long)(blockIdx.x + 1) * Wtot) / gridDim.x);

  for (int wi = begin; wi < end; ++wi) {
    const int p0 = wi / NT2D;
    const int nt = wi - p0 * NT2D;
    const int meta = smeta[p0];
    const int e = meta & 63;
    const int row0 = nt * N2TILE + 4 * warp;

    const size_t rb = (size_t)e * LH + row0;
    const uint2* W = reinterpret_cast<const uint2*>(wp + rb * (IH / 2));
    const uint8_t* S = ws + rb * (IH / 32);

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    uint2 wq[2][4];
    uint32_t sq[2][4];
#pragma unroll
    for (int p = 0; p < 2; ++p)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        wq[p][q] = W[q * WOFF + lane + 32 * p];
        sq[p][q] = S[q * SOFF + 16 * p + (lane >> 1)];
      }
#pragma unroll
    for (int it = 0; it < IH / 512; ++it) {
      const uint4* xp = reinterpret_cast<const uint4*>(hbuf + (size_t)p0 * IH + it * 512);
      uint4 xa = xp[2 * lane], xb = xp[2 * lane + 1];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        acc[q] = fmaf(ue8m0(sq[it & 1][q]), dot16(wq[it & 1][q], xa, xb), acc[q]);
        if (it + 2 < IH / 512) {
          wq[it & 1][q] = W[q * WOFF + lane + 32 * (it + 2)];
          sq[it & 1][q] = S[q * SOFF + 16 * (it + 2) + (lane >> 1)];
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[q] += __shfl_down_sync(0xffffffffu, acc[q], o);
    }
    if (lane == 0) {
      const float w = sw[p0];
      const size_t ob = (size_t)(meta >> 6) * LH + row0;
      __nv_bfloat162 v01 = __floats2bfloat162_rn(acc[0] * w, acc[1] * w);
      __nv_bfloat162 v23 = __floats2bfloat162_rn(acc[2] * w, acc[3] * w);
      atomicAdd(reinterpret_cast<__nv_bfloat162*>(out + ob), v01);
      atomicAdd(reinterpret_cast<__nv_bfloat162*>(out + ob + 2), v23);
    }
  }
}

// Fused decode step: route once in-CTA, FC1 for every (hit, column tile),
// one cooperative grid barrier, then FC2.  Removes a kernel launch and the
// second redundant route build; hbuf stays resident in L2 across the barrier.
__global__ __launch_bounds__(512, 2) void moe_dec_k(
    const __nv_bfloat16* __restrict__ x, const uint8_t* __restrict__ wp1,
    const uint8_t* __restrict__ ws1, const uint8_t* __restrict__ wp2,
    const uint8_t* __restrict__ ws2, const int* __restrict__ ids, const float* __restrict__ tw,
    __half* __restrict__ hbuf, int T, __nv_bfloat16* __restrict__ out) {
  extern __shared__ char smem_raw[];
  __half* xs = reinterpret_cast<__half*>(smem_raw);
  int* smeta = reinterpret_cast<int*>(smem_raw + LH * 2);
  float* sw = reinterpret_cast<float*>(smem_raw + LH * 2 + RCAP * 4);
  __shared__ int wsum[16];
  __shared__ int ranges[4];
  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;

  const int nh = route_build<true>(ids, tw, T * KTOP, smeta, sw, wsum, ranges);
  {
    constexpr int WOFF[4] = {0, 16 * (LH / 2) / 16, IH * (LH / 2) / 16, (IH + 16) * (LH / 2) / 16};
    constexpr int WOF2[4] = {0, 16 * (LH / 2) / 8, IH * (LH / 2) / 8, (IH + 16) * (LH / 2) / 8};
    constexpr int SOFF[4] = {0, 16 * (LH / 32), IH * (LH / 32), (IH + 16) * (LH / 32)};
    const uint8_t* __restrict__ wp = wp1;
    const uint8_t* __restrict__ ws = ws1;
    const int Wtot = nh * NT1D;
    const int begin = ranges[0];
    const int end = ranges[1];
    int cur = -1;

    for (int wi = begin; wi < end; ++wi) {
      const int p0 = wi / NT1D;
      const int nt = wi - p0 * NT1D;
      const int meta = smeta[p0];
      const int e = meta & 63;
      const int c0 = nt * C1TILE + warp;

      if (p0 != cur) {
        __syncthreads();
        const uint4* src = reinterpret_cast<const uint4*>(x + (size_t)(meta >> 6) * LH);
        uint4* dst = reinterpret_cast<uint4*>(xs);
        for (int i = tid; i < LH / 8; i += 512) dst[xswz4(i)] = cvt8(src[i]);
        __syncthreads();
        cur = p0;
      }

      const size_t rb = (size_t)e * (2 * IH) + c0;
      const uint4* W = reinterpret_cast<const uint4*>(wp + rb * (LH / 2));
      const uint2* W2 = reinterpret_cast<const uint2*>(wp + rb * (LH / 2));
      const uint8_t* S = ws + rb * (LH / 32);

      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      uint4 wq[2][4];
      float sq[2][4];
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          wq[p][q] = W[WOFF[q] + lane + 32 * p];
          sq[p][q] = ue8m0(S[SOFF[q] + 32 * p + lane]);
        }
#pragma unroll
      for (int it = 0; it < 3; ++it) {
        const uint4* xp = reinterpret_cast<const uint4*>(xs + it * 1024);
        uint4 xa = xp[lane], xb = xp[32 + lane];
        uint4 xc = xp[64 + lane], xd = xp[96 + lane];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 w = wq[it & 1][q];
          const float d = dot16(make_uint2(w.x, w.y), xa, xb) + dot16(make_uint2(w.z, w.w), xc, xd);
          acc[q] = fmaf(sq[it & 1][q], d, acc[q]);
          if (it + 2 < 3) {
            wq[it & 1][q] = W[WOFF[q] + lane + 32 * (it + 2)];
            sq[it & 1][q] = ue8m0(S[SOFF[q] + 32 * (it + 2) + lane]);
          }
        }
      }
      {
        const uint4* xp = reinterpret_cast<const uint4*>(xs + 3072);
        uint4 xa = xp[lane], xb = xp[32 + lane];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint2 w = W2[WOF2[q] + 192 + lane];
          const uint32_t sc = S[SOFF[q] + 96 + (lane >> 1)];
          acc[q] = fmaf(ue8m0(sc), dot16(w, xa, xb), acc[q]);
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[q] += __shfl_down_sync(0xffffffffu, acc[q], o);
      }
      if (lane == 0) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          float g = acc[q], u = acc[q + 2];
          float a = 4.f * tanhf(g * 0.25f) * (1.f / (1.f + __expf(-g)));
          float b = 25.f * tanhf(u * 0.04f);
          float h = __bfloat162float(__float2bfloat16(a * b));
          hbuf[(size_t)p0 * IH + hpos(c0 + 16 * q)] = __float2half(h * 128.f);
        }
      }
    }
  }
  {
    uint4* out4 = reinterpret_cast<uint4*>(out);
    const int n4 = T * (LH / 8);
    for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += gridDim.x * blockDim.x)
      out4[i] = make_uint4(0, 0, 0, 0);
  }
  cooperative_groups::this_grid().sync();
  {
    constexpr int WOFF = (IH / 2) / 16;
    constexpr int SOFF = IH / 32;
    const uint8_t* __restrict__ wp = wp2;
    const uint8_t* __restrict__ ws = ws2;
    const int Wtot = nh * NT2D;
    const int begin = ranges[2];
    const int end = ranges[3];

    for (int wi = begin; wi < end; ++wi) {
      const int p0 = wi / NT2D;
      const int nt = wi - p0 * NT2D;
      const int meta = smeta[p0];
      const int e = meta & 63;
      const int row0 = nt * N2TILE + 4 * warp;

      const size_t rb = (size_t)e * LH + row0;
      const uint4* W = reinterpret_cast<const uint4*>(wp + rb * (IH / 2));
      const uint8_t* S = ws + rb * (IH / 32);

      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      uint4 wq[2][4];
      float sq[2][4];
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          wq[p][q] = W[q * WOFF + lane + 32 * p];
          sq[p][q] = ue8m0(S[q * SOFF + 32 * p + lane]);
        }
#pragma unroll
      for (int it = 0; it < IH / 1024; ++it) {
        const uint4* xp = reinterpret_cast<const uint4*>(hbuf + (size_t)p0 * IH + it * 1024);
        uint4 xa = xp[lane], xb = xp[32 + lane];
        uint4 xc = xp[64 + lane], xd = xp[96 + lane];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          const uint4 w = wq[it & 1][q];
          const float d = dot16(make_uint2(w.x, w.y), xa, xb) + dot16(make_uint2(w.z, w.w), xc, xd);
          acc[q] = fmaf(sq[it & 1][q], d, acc[q]);
          if (it + 2 < IH / 1024) {
            wq[it & 1][q] = W[q * WOFF + lane + 32 * (it + 2)];
            sq[it & 1][q] = ue8m0(S[q * SOFF + 32 * (it + 2) + lane]);
          }
        }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[q] += __shfl_down_sync(0xffffffffu, acc[q], o);
      }
      if (lane == 0) {
        const float w = sw[p0];
        const size_t ob = (size_t)(meta >> 6) * LH + row0;
        __nv_bfloat162 v01 = __floats2bfloat162_rn(acc[0] * w, acc[1] * w);
        __nv_bfloat162 v23 = __floats2bfloat162_rn(acc[2] * w, acc[3] * w);
        atomicAdd(reinterpret_cast<__nv_bfloat162*>(out + ob), v01);
        atomicAdd(reinterpret_cast<__nv_bfloat162*>(out + ob + 2), v23);
      }
    }
  }
}
