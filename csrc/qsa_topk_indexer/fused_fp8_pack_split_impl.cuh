// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
// QSA stage-1 token top-k indexer, SM107 (Rubin)
//
// Algebraic short-circuit: nb = min(vis, 512).  When vis <= 512 every visible
// block is selected, so the selected set is {0..vis-1} regardless of score and
// the ascending packed row degenerates to 0,1,...,pos (count = pos+1).  Only
// tokens with vis > 512 (pos >= 2051) need scoring + selection, and the host
// can bound how many such tokens exist from NUM_CPAGES alone.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

#ifndef DEVICE_ONLY
namespace ffi = tvm::ffi;
#endif

#define CR 4
#define KTOP 512
#define PWIDTH 2052
#define ROW_V4 513
#define CPAGE 64
#define IDXH 4
#define IDXD 128
#define FASTPOS 2051
#define NTS 256      /* threads, store kernel                    */
#define NTK 256      /* threads, eight scoring/selection warps   */
#define TM 16        /* tokens per top-k CTA                     */
#define TN 128       /* blocks per MMA tile                      */
#define QPAD 272     /* padded fp16 row stride in smem (bytes)   */
#define SCSTR PWIDTH /* scores temporarily occupy output rows    */
#define VCAP 2048    /* max vis served by the smem score buffer  */

typedef unsigned char u8;

// streaming store: output rows are never read back, keep them evict-first in L2
__device__ __forceinline__ void st_cs(int4* p, int4 v) {
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z),
               "r"(v.w)
               : "memory");
}

__device__ __forceinline__ int findReqWarp(const int* __restrict__ cu, int B, int t) {
  int lo = 0, hi = B - 1;
  const int lane = threadIdx.x & 31;
  while (hi > lo) {
    int step = (hi - lo) / 32 + 1;
    int idx = lo + lane * step;
    if (idx > hi) idx = hi;
    unsigned b = __ballot_sync(0xFFFFFFFFu, cu[idx] <= t);
    if (b == 0u) break;
    int last = 31 - __clz(b);
    int nlo = lo + last * step;
    if (nlo > hi) nlo = hi;
    int nhi = nlo + step - 1;
    if (nhi > hi) nhi = hi;
    lo = nlo;
    hi = nhi;
  }
  return lo;
}

// ------------------------------------------------------- store kernel ------
extern "C" __global__ void __launch_bounds__(NTS)
    qsa_store(const int* __restrict__ cu, int* __restrict__ packed, int T, int B, int chunkLen,
              int rowsPerCta) {
  const int nt = blockDim.x;
  const int tid = threadIdx.x;
  const int row0 = blockIdx.x * rowsPerCta;
  if (row0 >= T) return;
  int jb = blockIdx.y * chunkLen;
  int je = jb + chunkLen;
  if (je > ROW_V4) je = ROW_V4;
  if (jb >= je) return;

  int r = findReqWarp(cu, B, row0);
  int cbeg = cu[r], cend = cu[r + 1];
  for (int q = 0; q < rowsPerCta; ++q) {
    const int row = row0 + q;
    if (row >= T) break;
    while (row >= cend && r < B - 1) {
      ++r;
      cbeg = cend;
      cend = cu[r + 1];
    }
    const int cnt = row - cbeg + 1;
    if (cnt > FASTPOS) continue;  // top-k kernel owns this row
    int4* out = reinterpret_cast<int4*>(packed + (size_t)row * PWIDTH);
    for (int j = jb + tid; j < je; j += nt) {
      const int b = j << 2;
      int4 v;
      v.x = (b + 0 < cnt) ? (b + 0) : -1;
      v.y = (b + 1 < cnt) ? (b + 1) : -1;
      v.z = (b + 2 < cnt) ? (b + 2) : -1;
      v.w = (b + 3 < cnt) ? (b + 3) : -1;
      if (j == ROW_V4 - 1) v.w = cnt;
      st_cs(out + j, v);
    }
  }
}

// -------------------------------------------------------- top-k kernel -----
__device__ __forceinline__ void mma_f16(float* d, unsigned a0, unsigned a1, unsigned a2,
                                        unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_fp8(float& d0, float& d1, float& d2, float& d3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldm4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                     const void* p) {
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(a));
}
__device__ __forceinline__ void ldm2(unsigned& r0, unsigned& r1, const void* p) {
  unsigned a = (unsigned)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}

// 16 fp8 bytes -> 16 fp16 (32 bytes), exact for e4m3
__device__ __forceinline__ void cvt16(const uint4& raw, uint4& o0, uint4& o1) {
  const unsigned short* p = reinterpret_cast<const unsigned short*>(&raw);
  unsigned* o = reinterpret_cast<unsigned*>(&o0);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)p[i], __NV_E4M3);
    union {
      __half2_raw h;
      unsigned u;
    } c;
    c.h = hr;
    if (i < 4)
      o[i] = c.u;
    else
      reinterpret_cast<unsigned*>(&o1)[i - 4] = c.u;
  }
}

__device__ __forceinline__ void dec2(unsigned short src, float& a, float& b) {
  __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)src, __NV_E4M3);
  float2 f = __half22float2(__half2(hr));
  a = f.x;
  b = f.y;
}

// fallback (vis > VCAP): recompute one (token, block) score from global fp8
__device__ float scoreG(const u8* __restrict__ qp, const u8* __restrict__ kp) {
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  for (int d0 = 0; d0 < IDXD; d0 += 8) {
    uint2 kr = *reinterpret_cast<const uint2*>(kp + d0);
    float kf[8];
    // Extract the halfwords by shift/mask instead of an unsigned short view
    // of the uint2 (not type-accessible under the aliasing rules); the
    // little-endian halfword order is unchanged.
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned int kword = (i < 2) ? kr.x : kr.y;
      dec2(static_cast<unsigned short>(kword >> (16 * (i & 1))), kf[2 * i], kf[2 * i + 1]);
    }
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      uint2 qr = *reinterpret_cast<const uint2*>(qp + h * IDXD + d0);
      float qf[8];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const unsigned int qword = (i < 2) ? qr.x : qr.y;
        dec2(static_cast<unsigned short>(qword >> (16 * (i & 1))), qf[2 * i], qf[2 * i + 1]);
      }
      float s = (h == 0) ? a0 : (h == 1) ? a1 : (h == 2) ? a2 : a3;
#pragma unroll
      for (int j = 0; j < 8; ++j) s = __fmaf_rn(qf[j], kf[j], s);
      if (h == 0)
        a0 = s;
      else if (h == 1)
        a1 = s;
      else if (h == 2)
        a2 = s;
      else
        a3 = s;
    }
  }
  return qsa_relu_exact(a0) + qsa_relu_exact(a1) + qsa_relu_exact(a2) + qsa_relu_exact(a3);
}

// Generic fallback for tiles whose shared-score footprint would exceed VCAP.
// A separate fast-only kernel owns all smaller tiles.
extern "C" __global__ void __launch_bounds__(NTK, 2)
    qsa_topk_slow(const u8* __restrict__ qi, const u8* __restrict__ ck, const int* __restrict__ cu,
                  const int* __restrict__ cpb, int* __restrict__ packed, int B) {
  // The extra uint4 per row pads the FP8 query stride to 528 bytes, distributing
  // the four MMA A-fragment loads across all shared-memory banks.
  __shared__ __align__(16) uint4 qtile[TM][33];
  __shared__ int hist[TM * 256];

  const int tid = threadIdx.x;
  const int lane = tid & 31, warp = tid >> 5;

  int total = 0;
  for (int rr = 0; rr < B; ++rr) {
    int n = cu[rr + 1] - cu[rr] - FASTPOS;
    if (n > 0) total += (n + TM - 1) / TM;
  }
  if ((int)blockIdx.x >= total) return;
  int rem = total - 1 - (int)blockIdx.x, r = -1, tile = 0;
  for (int rr = 0; rr < B; ++rr) {
    int n = cu[rr + 1] - cu[rr] - FASTPOS;
    if (n <= 0) continue;
    int ntl = (n + TM - 1) / TM;
    if (rem < ntl) {
      r = rr;
      tile = rem;
      break;
    }
    rem -= ntl;
  }
  if (r < 0) return;
  const int cu0 = cu[r], len = cu[r + 1] - cu0;
  const int pos0 = FASTPOS + tile * TM;
  const int t0 = cu0 + pos0;
  int nTok = len - pos0;
  if (nTok > TM) nTok = TM;
  const int visMax = (pos0 + nTok) >> 2;
  if (visMax <= VCAP) return;
  const u8* kbase = ck + (size_t)cpb[r] * CPAGE * IDXD;
  const bool fast = visMax <= VCAP;
  float* sc = reinterpret_cast<float*>(packed) + (size_t)t0 * SCSTR;

  for (int i = tid; i < TM * 256; i += NTK) hist[i] = 0;
  __syncthreads();
  if (fast) {
    for (int v = tid; v < TM * 32; v += NTK) {
      const int m = v >> 5, vec = v & 31;
      uint4 x = make_uint4(0u, 0u, 0u, 0u);
      if (m < nTok)
        x = *reinterpret_cast<const uint4*>(qi + (size_t)(t0 + m) * IDXH * IDXD + vec * 16);
      qtile[m][vec] = x;
    }
    __syncthreads();

    const int group = lane >> 2, quad = lane & 3;
    const int row0 = group, row1 = group + 8;
    const u8* q0base = reinterpret_cast<const u8*>(qtile[row0]);
    const u8* q1base = reinterpret_cast<const u8*>(qtile[row1]);
    for (int blockBase = warp * 8; blockBase < visMax; blockBase += 64) {
      const int keyCol = blockBase + group;
      const bool keyValid = keyCol < visMax;
      const u8* key = keyValid ? kbase + (size_t)keyCol * IDXD : kbase;
      float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f, sum3 = 0.f;
#pragma unroll
      for (int h = 0; h < IDXH; ++h) {
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
          const int k0 = kt * 32 + quad * 4;
          const u8* q0 = q0base + h * IDXD;
          const u8* q1 = q1base + h * IDXD;
          unsigned a0 = *reinterpret_cast<const unsigned*>(q0 + k0);
          unsigned a1 = *reinterpret_cast<const unsigned*>(q1 + k0);
          unsigned a2 = *reinterpret_cast<const unsigned*>(q0 + k0 + 16);
          unsigned a3 = *reinterpret_cast<const unsigned*>(q1 + k0 + 16);
          unsigned b0 = keyValid ? *reinterpret_cast<const unsigned*>(key + k0) : 0u;
          unsigned b1 = keyValid ? *reinterpret_cast<const unsigned*>(key + k0 + 16) : 0u;
          mma_fp8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
        }
        sum0 += qsa_relu_exact(d0);
        sum1 += qsa_relu_exact(d1);
        sum2 += qsa_relu_exact(d2);
        sum3 += qsa_relu_exact(d3);
      }
      const int col = blockBase + quad * 2;
      const int vis0 = (pos0 + row0 + 1) >> 2;
      const int vis1 = (pos0 + row1 + 1) >> 2;
      if (row0 < nTok && col < vis0) {
        sc[(size_t)row0 * SCSTR + col] = sum0;
        atomicAdd(&hist[row0 * 256 + (qsa_score_key_exact(sum0) >> 24)], 1);
      }
      if (row0 < nTok && col + 1 < vis0) {
        sc[(size_t)row0 * SCSTR + col + 1] = sum1;
        atomicAdd(&hist[row0 * 256 + (qsa_score_key_exact(sum1) >> 24)], 1);
      }
      if (row1 < nTok && col < vis1) {
        sc[(size_t)row1 * SCSTR + col] = sum2;
        atomicAdd(&hist[row1 * 256 + (qsa_score_key_exact(sum2) >> 24)], 1);
      }
      if (row1 < nTok && col + 1 < vis1) {
        sc[(size_t)row1 * SCSTR + col + 1] = sum3;
        atomicAdd(&hist[row1 * 256 + (qsa_score_key_exact(sum3) >> 24)], 1);
      }
    }
    __syncthreads();
  }

  // ---- per-warp exact top-512: 2 tokens per warp (TM=16, 8 warps) ----
  const unsigned FULL = 0xFFFFFFFFu;
  const unsigned ltmask = (1u << lane) - 1u;
  for (int m = warp; m < nTok; m += 8) {
    const int pos = pos0 + m;
    const int vis = (pos + 1) >> 2;
    const float* scm = sc + (size_t)m * SCSTR;
    const u8* qg = qi + (size_t)(t0 + m) * (IDXH * IDXD);
    int* hw = hist + m * 256;

    unsigned tau = 0u;
    int krem = KTOP;
    for (int shift = 24; shift >= 0; shift -= 8) {
      unsigned hmask = (shift == 24) ? 0u : (0xFFFFFFFFu << (shift + 8));
      if (shift != 24 || !fast) {
        if (shift != 24) {
          for (int d = lane; d < 256; d += 32) hw[d] = 0;
          __syncwarp();
        }
        if (fast) {
          const int nv4 = vis >> 2;
          const int ni = (nv4 + 31) >> 5;  // uniform across the warp
          const float4* s4 = reinterpret_cast<const float4*>(scm);
          for (int it = 0; it < ni; ++it) {
            const int i = (it << 5) + lane;
            float4 f = make_float4(0.f, 0.f, 0.f, 0.f);
            if (i < nv4) f = s4[i];
            const float fv[4] = {f.x, f.y, f.z, f.w};
#pragma unroll
            for (int u = 0; u < 4; ++u) {
              unsigned bin = 0x100u;
              if (i < nv4) {
                unsigned b = qsa_score_key_exact(fv[u]);
                if ((b & hmask) == tau) bin = (b >> shift) & 0xFFu;
              }
              unsigned peers = __match_any_sync(FULL, bin);
              if (bin != 0x100u && (__ffs(peers) - 1) == lane) atomicAdd(&hw[bin], __popc(peers));
            }
          }
          for (int v = (nv4 << 2) + lane; v < (nv4 << 2) + 32; v += 32) {
            unsigned bin = 0x100u;
            if (v < vis) {
              unsigned b = qsa_score_key_exact(scm[v]);
              if ((b & hmask) == tau) bin = (b >> shift) & 0xFFu;
            }
            unsigned peers = __match_any_sync(FULL, bin);
            if (bin != 0x100u && (__ffs(peers) - 1) == lane) atomicAdd(&hw[bin], __popc(peers));
          }
        } else {
          for (int vv = lane; vv < vis + lane; vv += 32) {
            unsigned bin = 0x100u;
            if (vv < vis) {
              unsigned b = qsa_score_key_exact(scoreG(qg, kbase + (size_t)vv * IDXD));
              if ((b & hmask) == tau) bin = (b >> shift) & 0xFFu;
            }
            unsigned peers = __match_any_sync(FULL, bin);
            if (bin != 0x100u && (__ffs(peers) - 1) == lane) atomicAdd(&hw[bin], __popc(peers));
          }
        }
        __syncwarp();
      }
      int g = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) g += hw[lane * 8 + i];
      int suf = g;
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        int y = __shfl_down_sync(FULL, suf, off);
        if (lane + off < 32) suf += y;
      }
      const int above = suf - g;
      const bool owner = (above < krem) && (above + g >= krem);
      const int src = __ffs(__ballot_sync(FULL, owner)) - 1;
      int dsel = 0, acc = above;
      if (owner) {
        for (dsel = lane * 8 + 7; dsel >= lane * 8; --dsel) {
          if (acc + hw[dsel] >= krem) break;
          acc += hw[dsel];
        }
      }
      int bucket = owner ? hw[dsel] : 0;
      dsel = __shfl_sync(FULL, dsel, src);
      acc = __shfl_sync(FULL, acc, src);
      bucket = __shfl_sync(FULL, bucket, src);
      tau |= ((unsigned)dsel) << shift;
      krem -= acc;
      __syncwarp();
      if (bucket == krem) break;
    }
    const int need = krem;

    int4* out = reinterpret_cast<int4*>(packed + (size_t)(t0 + m) * PWIDTH);
    // Scores occupy this destination row, so first materialize the exact
    // selection flags into the no-longer-needed radix histogram.  Only after
    // every score has been consumed do we compact flags into the row.
    int runEq = 0;
    for (int base = 0; base < vis; base += 32) {
      const int v = base + lane;
      int gt = 0, eq = 0;
      if (v < vis) {
        float s = fast ? scm[v] : scoreG(qg, kbase + (size_t)v * IDXD);
        unsigned b = qsa_score_key_exact(s);
        gt = (b > tau);
        eq = (b == tau);
      }
      unsigned mg = __ballot_sync(FULL, gt), me = __ballot_sync(FULL, eq);
      const int peq = __popc(me & ltmask);
      const bool take = gt || (eq && runEq + peq < need);
      const unsigned mt = __ballot_sync(FULL, take);
      if (lane == 0) hw[base >> 5] = static_cast<int>(mt);
      runEq += __popc(me);
    }
    __syncwarp();

    int rank = 0;
    for (int base = 0; base < vis; base += 32) {
      const unsigned mt = static_cast<unsigned>(hw[base >> 5]);
      if (mt & (1u << lane)) {
        const int p = rank + __popc(mt & ltmask);
        const int b0 = (base + lane) << 2;
        st_cs(out + p, make_int4(b0, b0 + 1, b0 + 2, b0 + 3));
      }
      rank += __popc(mt);
    }
    if (lane == 0) {
      const int tailStart = vis << 2;
      const int tailCnt = (pos + 1) & 3;
      int4 v;
      v.x = (0 < tailCnt) ? (tailStart + 0) : -1;
      v.y = (1 < tailCnt) ? (tailStart + 1) : -1;
      v.z = (2 < tailCnt) ? (tailStart + 2) : -1;
      v.w = KTOP * CR + tailCnt;
      st_cs(out + ROW_V4 - 1, v);
    }
    __syncwarp();
  }
}

// -------------------------------------------- dedicated identity packers ---
// For T <= FASTPOS every request row has at most KTOP complete blocks, so the
// exact packed result is the ascending interval [0, pos], followed by -1 pad
// and the valid count in the final column.
__device__ __forceinline__ void pack_identity_row(int* __restrict__ out, int pos) {
  const int tid = static_cast<int>(threadIdx.x);
  for (int v = tid; v < ROW_V4; v += NTS) {
    const int j = v << 2;
    reinterpret_cast<int4*>(out)[v] =
        make_int4(j <= pos ? j : -1, j + 1 <= pos ? j + 1 : -1, j + 2 <= pos ? j + 2 : -1,
                  j == PWIDTH - 4 ? pos + 1 : (j + 3 <= pos ? j + 3 : -1));
  }
}

__global__ __launch_bounds__(NTS, 4) void pack_rows_kernel(const int* __restrict__ cu,
                                                           int* __restrict__ packed, int T, int B) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= T) return;
  const int req = B == 1 ? 0 : findReqWarp(cu, B, token);
  const int pos = B == 1 ? token : token - cu[req];
  pack_identity_row(packed + static_cast<size_t>(token) * PWIDTH, pos);
}

__global__ __launch_bounds__(NTS, 4) void pack_by_request_kernel(const int* __restrict__ cu,
                                                                 int* __restrict__ packed, int B) {
  const int req = static_cast<int>(blockIdx.x);
  if (req >= B) return;
  const int begin = cu[req];
  const int end = cu[req + 1];
  for (int token = begin; token < end; ++token) {
    const int pos = token - begin;
    pack_identity_row(packed + static_cast<size_t>(token) * PWIDTH, pos);
  }
}

// ------------------------------------------------------------------ host ---
#ifndef DEVICE_ONLY
void Run(ffi::TensorView qi, ffi::TensorView ck, ffi::TensorView cu, ffi::TensorView cpb,
         ffi::TensorView packed) {
  const int T = (int)qi.shape()[0];
  const int B = (int)cpb.shape()[0];
  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(kDLCUDA, packed.device().device_id);

  if (T <= FASTPOS) {
    if (B >= 64) {
      pack_by_request_kernel<<<B, NTS, 0, stream>>>((const int*)cu.data_ptr(),
                                                    (int*)packed.data_ptr(), B);
    } else {
      pack_rows_kernel<<<T, NTS, 0, stream>>>((const int*)cu.data_ptr(), (int*)packed.data_ptr(), T,
                                              B);
    }
    return;
  }

  int R = (T <= 1024) ? 1 : (T <= 2048) ? 2 : (T <= 4096) ? 4 : 8;
  int nrow = (T + R - 1) / R;
  int nch = 1;
  if (nrow < 448) {
    nch = (448 + nrow - 1) / nrow;
    if (nch > 4) nch = 4;
  }
  int chunkLen = (ROW_V4 + nch - 1) / nch;

  qsa_store<<<dim3((unsigned)nrow, (unsigned)nch, 1u), NTS, 0, stream>>>(
      (const int*)cu.data_ptr(), (int*)packed.data_ptr(), T, B, chunkLen, R);

  // No request can contain a slow row when T <= FASTPOS.  Otherwise launch a
  // metadata-independent upper bound; excess CTAs return after reading cu.
  // This remains correct for arbitrary in-scope request partitions.
  if (T > FASTPOS) {
    int ntiles = (T + TM - 1) / TM + B;
    qsa_topk<<<(unsigned)ntiles, NTK, 0, stream>>>(
        (const u8*)qi.data_ptr(), (const u8*)ck.data_ptr(), (const int*)cu.data_ptr(),
        (const int*)cpb.data_ptr(), (int*)packed.data_ptr(), B);
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
#endif
