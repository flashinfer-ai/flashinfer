// fp8 blockwise-scaled dense linear for decode-regime M (SM90 / H200)
//   out[m,n] = sum_kg ws[n/128,kg] * xs[m,kg] * sum_{k in kg} xq[m,k]*w[n,k]
// The 1x128 activation quant is fused into the kernel prologue.  SM90 has no
// native fp8 mma.sync (ptxas expands it into 12 unpack + 2 HMMA), so the e4m3
// operands are widened to f16 explicitly - exactly, and with the A-side widening
// shared across all M sub-tiles - and fed to native m16n8k16 HMMA.  Split-K
// partials are pushed straight into the owning CTA through cluster DSMEM.

//
// Framework-independent device code: the kernels, device helpers and their
// launch-geometry constants, relocated verbatim from the former csrc kernel
// TU. Included by exactly one binding TU, csrc/fp8bs_gemm_decodem_sm90_hmma.cu, which owns the
// host-side launch glue and the exported entry point. The geometry macros
// are file-scope by design: never include two of these headers in the same
// translation unit.
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <cstring>

namespace cg = cooperative_groups;

#define BN 128
#define WN 8  // warps along n (16 rows each) -> BN = 128
#define SMEM_BUDGET (128 * 1024)
#define XROW 256  // bytes per (row, k-group) of f16 activations

__device__ __forceinline__ void cluster_arrive_relaxed() {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void mma_f16(float& d0, float& d1, float& d2, float& d3, uint32_t a0,
                                        uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t e4m3x2_f16x2(uint16_t e) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(e));
  return r;
}
__device__ __forceinline__ uint32_t cvt_lo(uint32_t a) {
  return e4m3x2_f16x2((uint16_t)(a & 0xffffu));
}
__device__ __forceinline__ uint32_t cvt_hi(uint32_t a) { return e4m3x2_f16x2((uint16_t)(a >> 16)); }
// quantize two fp32 to e4m3 (RN, saturating) and widen back to an f16 pair
__device__ __forceinline__ uint32_t q2(float a, float b) {
  __nv_fp8x2_storage_t e = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
  return e4m3x2_f16x2((uint16_t)e);
}
__device__ __forceinline__ float blo(uint32_t v) { return __uint_as_float(v << 16); }
__device__ __forceinline__ float bhi(uint32_t v) { return __uint_as_float(v & 0xffff0000u); }

template <int MT, int NWARP, int KSHIFT>
__global__ __launch_bounds__(NWARP * 32) void fp8bs_kernel(const __nv_bfloat16* __restrict__ X,
                                                           const uint8_t* __restrict__ W,
                                                           const float* __restrict__ WS,
                                                           __nv_bfloat16* __restrict__ OUT, int M,
                                                           int N, int K, int Sk, float eps,
                                                           int nbuf) {
  static_assert(KSHIFT >= 0 && KSHIFT <= 3);
  constexpr int MP = MT * 8;
  constexpr int WK = NWARP / WN;
  constexpr int NTHREAD = NWARP * 32;
  constexpr int ksplit = 1 << KSHIFT;
  extern __shared__ __align__(16) char smem_raw[];
  uint8_t* xqb = (uint8_t*)smem_raw;                   // [nbuf][MP][128] f16
  float* xsb = (float*)(smem_raw + nbuf * MP * XROW);  // [nbuf][MP]
  // Split-K partials stay FP32 through the DSM reduction: BF16-packed
  // partials of opposite sign can cancel to a rounded residual that differs
  // from the exact sum by the partial's full rounding error.
  float2* rcv = (float2*)(smem_raw + nbuf * MP * (XROW + 4));

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  cg::cluster_group cluster = cg::this_cluster();
  if constexpr (KSHIFT != 0) cluster_arrive_relaxed();
  const int wn = warp & (WN - 1);
  const int wk = warp / WN;
  const int gid = lane >> 2;
  const int tig = lane & 3;
  const int krank = blockIdx.x;
  const int ntile = blockIdx.y;
  const int nbase = ntile * BN;
  const int m0 = blockIdx.z * MP;
  const int Mloc = min(M - m0, MP);

  const int kg_beg = (Sk * krank) >> KSHIFT;
  const int kg_end = (Sk * (krank + 1)) >> KSHIFT;

  const int nw = nbase + wn * 16;
  int rr0 = nw + gid, rr1 = nw + gid + 8;
  rr0 = rr0 < N ? rr0 : (N - 1);
  rr1 = rr1 < N ? rr1 : (N - 1);
  const uint8_t* wA = W + (size_t)rr0 * (size_t)K + tig * 16;
  const uint8_t* wB = W + (size_t)rr1 * (size_t)K + tig * 16;
  const float* wsp = WS + (size_t)ntile * (size_t)Sk;

  const int qt = tid & 7;
  const int qg = tid >> 3;
  const int qs = qg & 7;
  const int qq = warp >> 1;  // 0..NWARP/2-1, warp-uniform

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;

  uint4 c0, c1, c2, c3, n0, n1, n2, n3;
  int pf = kg_beg + wk;
#define LOADW(d0, d1, d2, d3, kg)                 \
  {                                               \
    const uint8_t* pa_ = wA + (size_t)(kg) * 128; \
    const uint8_t* pb_ = wB + (size_t)(kg) * 128; \
    d0 = __ldg((const uint4*)pa_);                \
    d1 = __ldg((const uint4*)pb_);                \
    d2 = __ldg((const uint4*)(pa_ + 64));         \
    d3 = __ldg((const uint4*)(pb_ + 64));         \
  }
  if (pf < kg_end) {
    LOADW(c0, c1, c2, c3, pf);
    pf += WK;
  }
  if (pf < kg_end) {
    LOADW(n0, n1, n2, n3, pf);
    pf += WK;
  }

  for (int cb = kg_beg; cb < kg_end; cb += nbuf) {
    const int cn = min(nbuf, kg_end - cb);

    // ---------------- activation quantization for this chunk ----------------
    for (int i = qq; i < cn; i += (NWARP / 2)) {
      const int kg = cb + i;
      uint8_t* dst = xqb + (size_t)i * MP * XROW;
      float* dsc = xsb + i * MP;
      uint4 h0[MT], h1[MT];
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        const int m = qs + 8 * r;
        h0[r] = make_uint4(0, 0, 0, 0);
        h1[r] = make_uint4(0, 0, 0, 0);
        if (m < Mloc) {
          const __nv_bfloat16* xp = X + (size_t)(m0 + m) * (size_t)K + (size_t)kg * 128 + 8 * qt;
          h0[r] = __ldg((const uint4*)xp);
          h1[r] = __ldg((const uint4*)(xp + 64));
        }
      }
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        const int m = qs + 8 * r;
        uint4 v0 = h0[r], v1 = h1[r];
        uint32_t am = 0u;
        const uint32_t* vv = (const uint32_t*)&v0;
        const uint32_t* ww = (const uint32_t*)&v1;
#pragma unroll
        for (int t = 0; t < 4; ++t) {
          uint32_t a = vv[t] & 0x7fff7fffu, b = ww[t] & 0x7fff7fffu;
          uint32_t x0 = a & 0xffffu, x1 = a >> 16, x2 = b & 0xffffu, x3 = b >> 16;
          am = x0 > am ? x0 : am;
          am = x1 > am ? x1 : am;
          am = x2 > am ? x2 : am;
          am = x3 > am ? x3 : am;
        }
#pragma unroll
        for (int s = 1; s < 8; s <<= 1) {
          uint32_t o = __shfl_xor_sync(0xffffffffu, am, s);
          am = o > am ? o : am;
        }
        float amax = __uint_as_float(am << 16);
        const float inv = amax == 0.0f ? 1.0f : __fdiv_rn(448.0f, amax);
        if (qt == 0) dsc[m] = (m < Mloc) ? __frcp_rn(inv) : 0.0f;
        const int sw = m & 1;
        uint4 s0, s1;
        s0.x = q2(blo(v0.x) * inv, bhi(v0.x) * inv);
        s0.y = q2(blo(v0.y) * inv, bhi(v0.y) * inv);
        s0.z = q2(blo(v0.z) * inv, bhi(v0.z) * inv);
        s0.w = q2(blo(v0.w) * inv, bhi(v0.w) * inv);
        s1.x = q2(blo(v1.x) * inv, bhi(v1.x) * inv);
        s1.y = q2(blo(v1.y) * inv, bhi(v1.y) * inv);
        s1.z = q2(blo(v1.z) * inv, bhi(v1.z) * inv);
        s1.w = q2(blo(v1.w) * inv, bhi(v1.w) * inv);
        *(uint4*)(dst + m * XROW + (((qt ^ sw)) << 4)) = s0;
        *(uint4*)(dst + m * XROW + (((8 + qt) ^ sw) << 4)) = s1;
      }
    }
    __syncthreads();

    // ------------------------------- GEMM -------------------------------
    for (int i = wk; i < cn; i += WK) {
      const uint8_t* xb = xqb + (size_t)i * MP * XROW;
      const float* sb = xsb + i * MP;
      float g[MT][4];
#pragma unroll
      for (int j = 0; j < MT; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) g[j][q] = 0.0f;
#pragma unroll
      for (int sp = 0; sp < 2; ++sp) {
        const uint32_t* al = (const uint32_t*)(sp ? &c2 : &c0);
        const uint32_t* ah = (const uint32_t*)(sp ? &c3 : &c1);
        uint32_t f0[4], f1[4], f2[4], f3[4];
#pragma unroll
        for (int s = 0; s < 4; ++s) {
          f0[s] = cvt_lo(al[s]);
          f2[s] = cvt_hi(al[s]);
          f1[s] = cvt_lo(ah[s]);
          f3[s] = cvt_hi(ah[s]);
        }
#pragma unroll
        for (int j = 0; j < MT; ++j) {
          const int m = gid + 8 * j;
          const int cbk = (sp * 8 + tig * 2) ^ (m & 1);
          uint4 b0 = *(const uint4*)(xb + m * XROW + (cbk << 4));
          uint4 b1 = *(const uint4*)(xb + m * XROW + ((cbk ^ 1) << 4));
          mma_f16(g[j][0], g[j][1], g[j][2], g[j][3], f0[0], f1[0], f2[0], f3[0], b0.x, b0.y);
          mma_f16(g[j][0], g[j][1], g[j][2], g[j][3], f0[1], f1[1], f2[1], f3[1], b0.z, b0.w);
          mma_f16(g[j][0], g[j][1], g[j][2], g[j][3], f0[2], f1[2], f2[2], f3[2], b1.x, b1.y);
          mma_f16(g[j][0], g[j][1], g[j][2], g[j][3], f0[3], f1[3], f2[3], f3[3], b1.z, b1.w);
        }
      }

      // A prefetched group may belong to the next chunk, but not the next rank.
      if (cb + i + WK < kg_end) {
        c0 = n0;
        c1 = n1;
        c2 = n2;
        c3 = n3;
      }
      if (pf < kg_end) {
        LOADW(n0, n1, n2, n3, pf);
        pf += WK;
      }

      const float wsc = __ldg(wsp + (cb + i));
#pragma unroll
      for (int j = 0; j < MT; ++j) {
        float s0 = sb[tig * 2 + 8 * j] * wsc;
        float s1 = sb[tig * 2 + 1 + 8 * j] * wsc;
        acc[j][0] = fmaf(s0, g[j][0], acc[j][0]);
        acc[j][1] = fmaf(s1, g[j][1], acc[j][1]);
        acc[j][2] = fmaf(s0, g[j][2], acc[j][2]);
        acc[j][3] = fmaf(s1, g[j][3], acc[j][3]);
      }
    }
    // Only synchronize when a later chunk will overwrite xqb/xsb.  The final
    // chunk proceeds to the disjoint DSM reduction buffer.
    if (cb + cn < kg_end) __syncthreads();
  }

  // ------- epilogue: push partials into the owning CTA, reduce locally -------
  constexpr int wsh = 7 - KSHIFT;
  constexpr int wdt = 1 << wsh;
  const int rsv = wdt + 4;
  const int peer = (wn * 16) >> wsh;
  const int ncol = wn * 16 - (peer << wsh);
  if constexpr (KSHIFT != 0) cluster_wait_acquire();
  float2* dstb;
  if constexpr (KSHIFT != 0)
    dstb = cluster.map_shared_rank(rcv, peer);
  else
    dstb = rcv;
  float2* slot = dstb + (size_t)(krank * WK + wk) * (MP / 2) * rsv;
#pragma unroll
  for (int j = 0; j < MT; ++j) {
    const int mr = tig * 2 + 8 * j;
    const int pr = mr >> 1;
    const int nn = ncol + gid;
    slot[pr * rsv + nn] = make_float2(acc[j][0], acc[j][1]);
    slot[pr * rsv + nn + 8] = make_float2(acc[j][2], acc[j][3]);
  }
  if constexpr (KSHIFT != 0)
    cluster.sync();
  else
    __syncthreads();

  const int lo = krank * wdt;
  const int tot = ((Mloc + 1) >> 1) << wsh;
  for (int idx = tid; idx < tot; idx += NTHREAD) {
    const int mpair = idx >> wsh;
    const int nn = idx & (wdt - 1);
    float2 sum = rcv[mpair * rsv + nn];
#pragma unroll
    for (int c = 1; c < ksplit * WK; ++c) {
      const float2 v = rcv[(c * (MP / 2) + mpair) * rsv + nn];
      sum.x += v.x;
      sum.y += v.y;
    }
    const int ng = nbase + lo + nn;
    const int mm = m0 + 2 * mpair;
    if (ng < N && mm < M) OUT[(size_t)mm * (size_t)N + ng] = __float2bfloat16(sum.x);
    if (ng < N && mm + 1 < M) OUT[(size_t)(mm + 1) * (size_t)N + ng] = __float2bfloat16(sum.y);
  }
}
