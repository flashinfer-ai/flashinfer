// fp8 e4m3 blockwise-scaled NT GEMM, sm120 decode shapes -- "xp" family.
//   Y[m,n] = sum_g As[m,g] * Bs[n/128,g] * sum_{k in group g} A[m,k]*B[n,k]
//
// Same direct-register mma.sync.m16n8k32 core as the rest of the hybrid: one
// 256-bit load supplies a whole fragment because the reduction order inside a
// 128-wide k group is free, so k is permuted identically on both operands.
//
// Differences from the older c00/d26 tiles:
//   * the cross-warp mailbox carries fp32 pairs (float2) instead of packed
//     half2, so no partial can overflow, and the plane stride is padded to a
//     bank-conflict-free 4*odd so the wider store still costs one wavefront
//     pair -- same shared traffic as the packed form, full fp32 range;
//   * every epilogue store is a 2-wide bf16x2 (half the store instructions,
//     full 4-byte sectors) whenever N is even;
//   * all global addressing is mul.wide.s32 (one IMAD.WIDE) instead of a
//     64x64 multiply.
//
// Framework-independent device code: the kernels, device helpers and their
// launch-geometry constants/macros, relocated verbatim from the former csrc
// kernel TU. Included by exactly one binding TU,
// csrc/specialized_gemm_fp8_blockwise_sm120/xp.cu,
// which keeps the host-side launch glue and dispatch. File-scope geometry
// macros/constants are by design: never include two of these headers in one
// translation unit.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace xp_impl {

struct F8 {
  uint32_t v[8];
};

__device__ __forceinline__ F8 ldB(const void* p) {
  F8 r;
  asm("ld.global.v8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]),
        "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ F8 ldA(const void* p) {
  F8 r;
  asm("ld.global.nc.L1::evict_last.v8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]),
        "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ float ldf(const float* p) {
  float r;
  asm("ld.global.nc.L1::evict_last.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}

// Y is never read back, so the stores are kept out of the caches the weight
// stream wants.
template <bool WT>
__device__ __forceinline__ void stPair(__nv_bfloat16* p, float a, float b) {
  const __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  unsigned u;
  __builtin_memcpy(&u, &h, 4);
  if constexpr (WT)
    asm volatile("st.global.wt.b32 [%0], %1;" ::"l"(p), "r"(u) : "memory");
  else
    *reinterpret_cast<unsigned*>(p) = u;
}
template <bool WT>
__device__ __forceinline__ void stOne(__nv_bfloat16* p, float a) {
  const __nv_bfloat16 h = __float2bfloat16_rn(a);
  unsigned short u;
  __builtin_memcpy(&u, &h, 2);
  if constexpr (WT)
    asm volatile("st.global.wt.b16 [%0], %1;" ::"l"(p), "h"(u) : "memory");
  else
    *reinterpret_cast<unsigned short*>(p) = u;
}

#define MMA(d, a0, a1, a2, a3, b0, b1)                          \
  asm volatile(                                                 \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "    \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])          \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// NW warps per CTA split NWN ways along n and NWK = NW/NWN ways along k.
// KI > 0 pins the per-warp k-group trip count so the loop control folds away.
// SWAPG puts the m chunk on the fast grid axis, so CTAs sharing a weight
// column launch back to back.
template <int NW, int NWN, int NT, int MR, int KI, int MINB, bool SWAPG, bool WT>
__global__ void __launch_bounds__(NW * 32, MINB)
    xpk(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bq, const float* __restrict__ As,
        const float* __restrict__ Bs, __nv_bfloat16* __restrict__ Y, int M, int N, int K, int KB) {
  constexpr int NWK = NW / NWN;
  constexpr int MT = (MR <= 8) ? 1 : (MR / 16);
  constexpr int HH = (MR <= 8) ? 1 : 2;
  constexpr int NR = MT * HH;
  constexpr int CN = NWN * NT * 8;
  constexpr int CP = CN / 2;
  // plane stride padded to 4*odd float2 slots: the 16 threads of a store phase
  // then cover 32 distinct banks.
  constexpr int Q = (CP + 3) / 4;
  constexpr int S2 = (Q & 1) ? (4 * Q) : (4 * Q + 4);
  constexpr int RSZ = (NWK > 1) ? (NWK * MR * S2) : 1;
  __shared__ float2 sm[RSZ];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int ng = (NWN == 1) ? 0 : (wid / NWK);
  const int wk = wid - ng * NWK;
  const int g = lane >> 2;
  const int c = lane & 3;

  const int nc = (int)(SWAPG ? blockIdx.y : blockIdx.x) * CN;
  const int n0 = nc + ng * (NT * 8);
  const int m0 = (int)(SWAPG ? blockIdx.x : blockIdx.y) * MR;

  // Out-of-range rows/cols read a live address and are dropped at store time:
  // an mma output element depends only on its own A row and B column.
  const uint8_t* bp[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int r = n0 + t * 8 + g;
    bp[t] = Bq + (long long)(r < N ? r : 0) * K + (c * 32) + (wk * 128);
  }
  const uint8_t* ap[NR];
  const float* asp[NR];
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    const int r = m0 + (q / HH) * 16 + (q % HH) * 8 + g;
    const int rc = (r < M) ? r : 0;
    ap[q] = A + (long long)rc * K + (c * 32) + (wk * 128);
    asp[q] = As + (long long)rc * KB + wk;
  }
  const float* bsp = Bs + (long long)(nc >> 7) * KB + wk;

  float acc[MT][NT][4];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[t][i][r] = 0.f;

  auto body = [&]() {
    F8 bf[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      bf[t] = ldB(bp[t]);
      bp[t] += NWK * 128;
    }
    const float bsv = ldf(bsp);
    bsp += NWK;
    F8 af[NR];
    float sc[NR];
#pragma unroll
    for (int q = 0; q < NR; ++q) {
      af[q] = ldA(ap[q]);
      sc[q] = ldf(asp[q]) * bsv;
      ap[q] += NWK * 128;
      asp[q] += NWK;
    }
#pragma unroll
    for (int t = 0; t < MT; ++t) {
      const F8& F0 = af[t * HH];
      const F8& F1 = af[t * HH + HH - 1];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        float d[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t z = 0u;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          MMA(d, F0.v[2 * j], (HH == 2) ? F1.v[2 * j] : z, F0.v[2 * j + 1],
              (HH == 2) ? F1.v[2 * j + 1] : z, bf[i].v[2 * j], bf[i].v[2 * j + 1]);
        }
        acc[t][i][0] += sc[t * HH] * d[0];
        acc[t][i][1] += sc[t * HH] * d[1];
        if (HH == 2) {
          acc[t][i][2] += sc[t * HH + 1] * d[2];
          acc[t][i][3] += sc[t * HH + 1] * d[3];
        }
      }
    }
  };

  if (KI > 0) {
#pragma unroll
    for (int it = 0; it < KI; ++it) body();
  } else {
    for (int kg = wk; kg < KB; kg += NWK) body();
  }

  const bool pairOK = ((N & 1) == 0);

  if (NWK == 1) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int col = n0 + i * 8 + c * 2;
        const int r0 = m0 + t * 16 + g;
        if (r0 < M && col < N) {
          __nv_bfloat16* p = Y + (long long)r0 * N + col;
          if (pairOK && col + 1 < N)
            stPair<WT>(p, acc[t][i][0], acc[t][i][1]);
          else {
            stOne<WT>(p, acc[t][i][0]);
            if (col + 1 < N) stOne<WT>(p + 1, acc[t][i][1]);
          }
        }
        if (HH == 2 && r0 + 8 < M && col < N) {
          __nv_bfloat16* p = Y + (long long)(r0 + 8) * N + col;
          if (pairOK && col + 1 < N)
            stPair<WT>(p, acc[t][i][2], acc[t][i][3]);
          else {
            stOne<WT>(p, acc[t][i][2]);
            if (col + 1 < N) stOne<WT>(p + 1, acc[t][i][3]);
          }
        }
      }
    return;
  }

  float2* sw = sm + wk * (MR * S2);
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int pb = ng * (NT * 4) + i * 4 + c;
      const int r0 = t * 16 + g;
      sw[r0 * S2 + pb] = make_float2(acc[t][i][0], acc[t][i][1]);
      if (HH == 2) sw[(r0 + 8) * S2 + pb] = make_float2(acc[t][i][2], acc[t][i][3]);
    }
  __syncthreads();
  constexpr int TOT = MR * CP;
  for (int idx = tid; idx < TOT; idx += NW * 32) {
    // keep the (row, 4-pair) decomposition the stores used so the loads land
    // on the same conflict-free bank pattern
    const int q4 = idx >> 2;
    const int cc = idx & 3;
    const int row = q4 % MR;
    const int pb = (q4 / MR) * 4 + cc;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int w = 0; w < NWK; ++w) {
      const float2 v = sm[(w * MR + row) * S2 + pb];
      s0 += v.x;
      s1 += v.y;
    }
    const int gr = m0 + row, gc = nc + pb * 2;
    if (gr < M && gc < N) {
      __nv_bfloat16* p = Y + (long long)gr * N + gc;
      if (pairOK && gc + 1 < N)
        stPair<WT>(p, s0, s1);
      else {
        stOne<WT>(p, s0);
        if (gc + 1 < N) stOne<WT>(p + 1, s1);
      }
    }
  }
}

}  // namespace xp_impl
