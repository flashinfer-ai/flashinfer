// fp8 e4m3 blockwise-scaled NT GEMM for sm120 decode shapes.
//   Y[m,n] = sum_g As[m,g] * Bs[n/128,g] * sum_{k in group g} A[m,k]*B[n,k]
// A (M,K) row-major fp8, B (N,K) row-major fp8, scales fp32, Y (M,N) bf16.
//
// One templated kernel: a CTA owns CN = NW_N*NT*8 output columns and all of K.
// The W = NW_N*NW_K warps split K; each warp keeps KU k-groups in flight so a
// single round of 256-bit loads covers several groups.  Inside a 128-wide k
// group the reduction order is free, so k is permuted such that both operands
// arrive as 32 contiguous bytes straight into mma fragment registers -- no smem
// staging and no ldmatrix on the streamed weight operand.
//
// Out-of-range rows/columns are handled by clamping the *address* to a live
// row and dropping the value at store time: an mma output element depends only
// on its own A row and B column, so garbage never reaches a stored element.
//
// Framework-independent device code: the kernels, device helpers and their
// launch-geometry constants/macros, relocated verbatim from the former csrc
// kernel TU. Included by exactly one binding TU,
// csrc/specialized_gemm_fp8_blockwise_sm120/table_kernel.cu,
// which keeps the host-side launch glue and dispatch. File-scope geometry
// macros/constants are by design: never include two of these headers in one
// translation unit.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace table_impl {

#define MMA16832(d, a0, a1, a2, a3, b0, b1)                     \
  asm volatile(                                                 \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "    \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])          \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

struct Frag8 {
  uint32_t v[8];
};

__device__ __forceinline__ Frag8 ld_stream(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::evict_first.L2::256B.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]),
        "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ Frag8 ld_keep(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::evict_last.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]),
        "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ float ld_keepf(const float* p) {
  float r;
  asm("ld.global.nc.L1::evict_last.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ void store2(__nv_bfloat16* p, float v0, float v1, int col, int N) {
  if (col + 1 < N) {
    __nv_bfloat162 v = __floats2bfloat162_rn(v0, v1);
    if ((reinterpret_cast<uintptr_t>(p) & 3u) == 0u) {
      *reinterpret_cast<__nv_bfloat162*>(p) = v;
    } else {
      p[0] = v.x;
      p[1] = v.y;
    }
  } else if (col < N) {
    p[0] = __float2bfloat16_rn(v0);
  }
}

// NW_N: warp groups along N.  NW_K: warps splitting K.  NT: n8 subtiles per
// warp.  MR: compile-time row bound (8/16/32).  KU: k-groups in flight.
template <int NW_N, int NW_K, int NT, int MR, int KU, int ITERS, int MINB = 1>
__global__ void __launch_bounds__(NW_N* NW_K * 32, MINB)
    gemm_bw(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bq,
            const float* __restrict__ As, const float* __restrict__ Bs,
            __nv_bfloat16* __restrict__ Y, int M, int N, int K, int KB) {
  constexpr int W = NW_N * NW_K;
  constexpr int CN = NW_N * NT * 8;
  constexpr int MT = (MR + 15) / 16;     // 16-row m-tiles
  constexpr int HH = (MR <= 8) ? 1 : 2;  // 8-row halves actually live
  constexpr int NR = MT * HH;            // A row slots per thread
  constexpr int ACT = HH * 2;            // accumulators per (mtile, ntile)
  constexpr int NACC = MT * NT * ACT;
  // The cross-warp reduction scratch lives in DYNAMIC shared memory: the
  // largest tilings need more than the 48 KB that ptxas allows a sm_120a
  // kernel to declare statically (CUDA 12.9 rejects the build outright).
  // Every launch passes red_smem_bytes<...>() and opts in through
  // cudaFuncAttributeMaxDynamicSharedMemorySize when that exceeds 48 KB.
  extern __shared__ float red[];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int ng = (NW_N == 1) ? 0 : (warp % NW_N);
  const int wk = (NW_N == 1) ? warp : (warp / NW_N);
  const int g = lane >> 2;
  const int c = lane & 3;
  const int nc = blockIdx.x * CN;
  const int n0 = nc + ng * (NT * 8);
  const int mb = blockIdx.y * MR;
  const int kstep = NW_K * KU * 128;

  const uint8_t* bp[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int nq = n0 + i * 8 + g;
    bp[i] = Bq + (size_t)(nq < N ? nq : 0) * (size_t)K + (size_t)(wk * KU * 128 + c * 32);
  }
  const uint8_t* ap[NR];
  const float* asp[NR];
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    const int r = mb + (q / HH) * 16 + (q % HH) * 8 + g;
    const int rc = (r < M) ? r : 0;
    ap[q] = A + (size_t)rc * (size_t)K + (size_t)(wk * KU * 128 + c * 32);
    asp[q] = As + (size_t)rc * (size_t)KB + wk * KU;
  }
  int nblk = n0 >> 7;
  const int nbmax = (N + 127) >> 7;
  if (nblk >= nbmax) nblk = nbmax - 1;
  const float* bsp = Bs + (size_t)nblk * (size_t)KB + wk * KU;

  float acc[NACC];
#pragma unroll
  for (int a = 0; a < NACC; ++a) acc[a] = 0.f;

  // One k step = KU groups per warp.  The dispatcher only picks a k split
  // that divides KB, so the trip count is exact; ITERS pins it at compile time
  // for the common one- and two-step decode schedules.
  auto kstep_body = [&]() {
    Frag8 bf[KU][NT];
    Frag8 af[KU][NR];
    float sc[KU][NR];
#pragma unroll
    for (int u = 0; u < KU; ++u) {
#pragma unroll
      for (int i = 0; i < NT; ++i) bf[u][i] = ld_stream(bp[i] + u * 128);
    }
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      const float bsv = ld_keepf(bsp + u);
#pragma unroll
      for (int q = 0; q < NR; ++q) {
        af[u][q] = ld_keep(ap[q] + u * 128);
        sc[u][q] = ld_keepf(asp[q] + u) * bsv;
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) bp[i] += kstep;
#pragma unroll
    for (int q = 0; q < NR; ++q) {
      ap[q] += kstep;
      asp[q] += NW_K * KU;
    }
    bsp += NW_K * KU;

#pragma unroll
    for (int u = 0; u < KU; ++u) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          const Frag8& F0 = af[u][t * HH];
          const Frag8& F1 = af[u][t * HH + HH - 1];  // == F0 when HH == 1
          const uint32_t z = 0u;
          float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t a0 = F0.v[2 * j], a2 = F0.v[2 * j + 1];
            const uint32_t a1 = (HH == 2) ? F1.v[2 * j] : z;
            const uint32_t a3 = (HH == 2) ? F1.v[2 * j + 1] : z;
            MMA16832(d, a0, a1, a2, a3, bf[u][i].v[2 * j], bf[u][i].v[2 * j + 1]);
          }
          const int b = (t * NT + i) * ACT;
          acc[b + 0] += sc[u][t * HH] * d[0];
          acc[b + 1] += sc[u][t * HH] * d[1];
          if (HH == 2) {
            acc[b + 2] += sc[u][t * HH + 1] * d[2];
            acc[b + 3] += sc[u][t * HH + 1] * d[3];
          }
        }
      }
    }
  };

  if constexpr (ITERS > 0) {
#pragma unroll
    for (int it = 0; it < ITERS; ++it) kstep_body();
  } else {
    const int iters = KB / (NW_K * KU);
    for (int it = 0; it < iters; ++it) kstep_body();
  }

  if (NW_K == 1) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int b = (t * NT + i) * ACT;
        const int col = n0 + i * 8 + c * 2;
        const int r0 = mb + t * 16 + g;
        if (r0 < M) store2(Y + (size_t)r0 * (size_t)N + col, acc[b + 0], acc[b + 1], col, N);
        if (HH == 2 && r0 + 8 < M)
          store2(Y + (size_t)(r0 + 8) * (size_t)N + col, acc[b + 2], acc[b + 3], col, N);
      }
    return;
  }

  // Cross-warp k reduction: every warp parks its accumulators, then all
  // threads cooperatively sum the NW_K partials for a slice of the tile.
  float* sw = red + (size_t)warp * NACC * 32;
#pragma unroll
  for (int a = 0; a < NACC; ++a) sw[a * 32 + lane] = acc[a];
  __syncthreads();
  // Accumulator slots come in adjacent column pairs (sl, sl+1) for the same
  // row, so each thread reduces both and leaves one 4-byte bf16x2 store
  // instead of two 2-byte scattered ones.
  constexpr int PAIRS = NW_N * (NACC / 2) * 32;
  constexpr int THREADS = W * 32;
  for (int idx = threadIdx.x; idx < PAIRS; idx += THREADS) {
    const int ngi = idx / ((NACC / 2) * 32);
    const int rest = idx - ngi * ((NACC / 2) * 32);
    const int a0 = (rest >> 5) * 2;
    const int l = rest & 31;
    float s0 = 0.f, s1 = 0.f;
#pragma unroll
    for (int w = 0; w < NW_K; ++w) {
      const size_t b = (size_t)(w * NW_N + ngi) * NACC * 32;
      s0 += red[b + (size_t)a0 * 32 + l];
      s1 += red[b + (size_t)(a0 + 1) * 32 + l];
    }
    const int t = a0 / (NT * ACT);
    const int rem = a0 - t * (NT * ACT);
    const int i = rem / ACT;
    const int sl = rem - i * ACT;
    const int row = mb + t * 16 + ((sl >= 2) ? 8 : 0) + (l >> 2);
    const int col = nc + ngi * (NT * 8) + i * 8 + (l & 3) * 2;
    if (row < M) store2(Y + (size_t)row * (size_t)N + col, s0, s1, col, N);
  }
}
// Dynamic-smem byte count of gemm_bw's cross-warp reduction scratch (zero for
// the NW_K == 1 forms, which never touch it).
template <int NW_N, int NW_K, int NT, int MR>
constexpr int red_smem_bytes() {
  const int MT = (MR + 15) / 16;
  const int HH = (MR <= 8) ? 1 : 2;
  const int NACC = MT * NT * (HH * 2);
  return (NW_K > 1) ? NW_N * NW_K * NACC * 32 * static_cast<int>(sizeof(float)) : 0;
}

}  // namespace table_impl
