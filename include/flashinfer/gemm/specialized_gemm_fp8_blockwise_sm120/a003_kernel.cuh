// fp8 e4m3 blockwise-scaled NT GEMM for sm120 decode shapes.
//   Y[m,n] = sum_g As[m,g] * Bs[n/128,g] * sum_{k in group g} A[m,k]*B[n,k]
// A (M,K) row-major fp8, B (N,K) row-major fp8, scales fp32, Y (M,N) bf16.
//
// Warp-level mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.  Inside one
// 128-wide k group the reduction order is free, so k is permuted such that both
// operands arrive as 16B contiguous uint4 loads straight into fragment
// registers -- no smem staging and no ldmatrix on the streamed weight operand.
//
// Out-of-range rows/columns are handled by clamping the *address* to a live row
// and dropping the value at store time: an mma output element depends only on
// its own A row and B column, so garbage never reaches a stored element.  The
// k loop is therefore completely branch-free.
//
// Framework-independent device code: the kernels, device helpers and their
// launch-geometry constants/macros, relocated verbatim from the former csrc
// kernel TU. Included by exactly one binding TU,
// csrc/specialized_gemm_fp8_blockwise_sm120/a003_kernel.cu,
// which keeps the host-side launch glue and dispatch. File-scope geometry
// macros/constants are by design: never include two of these headers in one
// translation unit.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#define FLASHINFER_SM120_A003_MMA(d, a0, a1, a2, a3, b0, b1)    \
  asm volatile(                                                 \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "    \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])          \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// The weight stream is reused only inside its own 128B line (the two 16B
// fragment halves), while A / As are re-read by every block (they are only M*K
// bytes -- with M=16 A sends twice as many bytes to L2 as the weights do).
// Tag the weights evict-first and the activations evict-last so the streaming
// weights stop flushing A out of L1.
// One 256-bit load per row per 128-wide k group.  k is permuted so the eight
// fragment words a thread needs are 32 *contiguous* bytes, which both halves
// the load instruction count and makes a warp-wide access consume whole 128B
// lines instead of half of each.
struct Frag8 {
  uint32_t v[8];
};
__device__ __forceinline__ Frag8 ld_stream(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::no_allocate.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]),
        "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ Frag8 ld_stream_l2(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
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
__device__ __forceinline__ float ld_uniformf(const float* p) {
  float r;
  asm("ldu.global.f32 %0, [%1];" : "=f"(r) : "l"(p));
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

// W: warps per block, splitting the k-group loop; they reduce through smem.
// MR: compile-time row bound (8/16/32).  MSPLIT: m rows are chunked over
// grid.y -- it costs one extra weight pass per chunk, but on shapes with too
// few n-tiles to fill the SMs and a small weight stream it is the only way to
// buy blocks (N=64 has just 8 n-tiles for 188 SMs).  KITER fixes the number of
// scale groups owned by each warp so the loop control and pointer bumps fold.
template <int W, int MR, bool MSPLIT, int KITER, bool UNIFORM_BS = false, bool L2_B = false>
__global__ void __launch_bounds__(W * 32)
    gemm_fp8_bw_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bq,
                       const float* __restrict__ As, const float* __restrict__ Bs,
                       __nv_bfloat16* __restrict__ Y, int M, int N, int K, int KB) {
  constexpr int MT = (MR + 15) / 16;     // 16-row m-tiles
  constexpr int HH = (MR <= 8) ? 1 : 2;  // 8-row halves actually live
  constexpr int NR = MT * HH;            // A row slots per thread
  constexpr int ACT = HH * 2;            // accumulators per m-tile
  constexpr int NT = 1;
  constexpr int NACC = MT * NT * ACT;
  constexpr bool VRED = W == 4 && MR == 16 && !MSPLIT;
  constexpr int VPAIRS = NACC / 2;
  constexpr int RSZ = (W > 1 && !VRED) ? (W - 1) * NACC * 32 : 1;
  // MR=32 keeps 32 fragment registers live; a second in-flight k group spills.
  constexpr int UNR = 1;
  constexpr bool BFIRST = MR == 16 && !MSPLIT && KITER == 0 && (W == 4 || W == 16);
  __shared__ float red[RSZ];
  __shared__ float2 red2[VRED ? W * VPAIRS * 32 : 1];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int c = lane & 3;
  const int n0 = blockIdx.x * (NT * 8);
  const int mb = MSPLIT ? (int)blockIdx.y * MR : 0;
  const int kstep = W * 128;

  const uint8_t* bp[NT];
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    const int nq = n0 + i * 8 + g;
    bp[i] = Bq + (size_t)(nq < N ? nq : 0) * (size_t)K + (size_t)(warp * 128 + c * 32);
  }
  const uint8_t* ap[NR];
  const float* asp[NR];
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    const int r = mb + (q / HH) * 16 + (q % HH) * 8 + g;
    const int rc = (r < M && (!MSPLIT || g < MR)) ? r : 0;
    ap[q] = A + (size_t)rc * (size_t)K + (size_t)(warp * 128 + c * 32);
    asp[q] = As + (size_t)rc * (size_t)KB + warp;
  }
  const float* bsp = Bs + (size_t)(n0 >> 7) * (size_t)KB + warp;

  float acc[NACC];
#pragma unroll
  for (int a = 0; a < NACC; ++a) acc[a] = 0.f;

  auto kbody = [&]() {
    float bsv;
    if constexpr (UNIFORM_BS)
      bsv = ld_uniformf(bsp);
    else
      bsv = ld_keepf(bsp);
    bsp += W;
    Frag8 bf[NT];
    if constexpr (BFIRST) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        if constexpr (L2_B)
          bf[i] = ld_stream_l2(bp[i]);
        else
          bf[i] = ld_stream(bp[i]);
        bp[i] += kstep;
      }
    }
    Frag8 af[NR];
    float sc[NR];
#pragma unroll
    for (int q = 0; q < NR; ++q) {
      af[q] = ld_keep(ap[q]);
      if constexpr (BFIRST && W == 4) {
        float asv = (c == 0) ? ld_keepf(asp[q]) : 0.f;
        asv = __shfl_sync(0xffffffffu, asv, g * 4);
        sc[q] = asv * bsv;
      } else {
        sc[q] = ld_keepf(asp[q]) * bsv;
      }
      ap[q] += kstep;
      asp[q] += W;
    }
    if constexpr (!BFIRST) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        if constexpr (L2_B)
          bf[i] = ld_stream_l2(bp[i]);
        else
          bf[i] = ld_stream(bp[i]);
        bp[i] += kstep;
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) {
#pragma unroll
      for (int t = 0; t < MT; ++t) {
        const Frag8& F0 = af[t * HH];
        const Frag8& F1 = af[t * HH + HH - 1];  // == F0 when HH == 1
        const uint32_t z = 0u;
        float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t a0 = F0.v[2 * j], a2 = F0.v[2 * j + 1];
          const uint32_t a1 = (HH == 2) ? F1.v[2 * j] : z;
          const uint32_t a3 = (HH == 2) ? F1.v[2 * j + 1] : z;
          FLASHINFER_SM120_A003_MMA(d, a0, a1, a2, a3, bf[i].v[2 * j], bf[i].v[2 * j + 1]);
        }
        const int b = (t * NT + i) * ACT;
        acc[b + 0] += sc[t * HH] * d[0];
        acc[b + 1] += sc[t * HH] * d[1];
        if (HH == 2) {
          acc[b + 2] += sc[t * HH + 1] * d[2];
          acc[b + 3] += sc[t * HH + 1] * d[3];
        }
      }
    }
  };

  if (KITER > 0) {
#pragma unroll
    for (int it = 0; it < KITER; ++it) kbody();
  } else {
#pragma unroll UNR
    for (int kg = warp; kg < KB; kg += W) kbody();
  }

  if (W > 1) {
    if constexpr (VRED) {
      static_assert(NACC == 4);
#pragma unroll
      for (int p = 0; p < VPAIRS; ++p)
        red2[(warp * VPAIRS + p) * 32 + lane] = make_float2(acc[2 * p], acc[2 * p + 1]);
      __syncthreads();
      if (warp < VPAIRS) {
        float2 s = make_float2(0.f, 0.f);
#pragma unroll
        for (int w = 0; w < W; ++w) {
          const float2 v = red2[(w * VPAIRS + warp) * 32 + lane];
          s.x += v.x;
          s.y += v.y;
        }
        const int row = mb + warp * 8 + g;
        const int col = n0 + c * 2;
        if (row < M) store2(Y + (size_t)row * N + col, s.x, s.y, col, N);
      }
      return;
    }
    if (warp > 0) {
#pragma unroll
      for (int a = 0; a < NACC; ++a) red[((warp - 1) * NACC + a) * 32 + lane] = acc[a];
    }
    __syncthreads();
    if (warp == 0) {
      for (int w = 0; w < W - 1; ++w) {
#pragma unroll
        for (int a = 0; a < NACC; ++a) acc[a] += red[(w * NACC + a) * 32 + lane];
      }
    }
  }
  if (warp != 0) return;

#pragma unroll
  for (int t = 0; t < MT; ++t) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int b = (t * NT + i) * ACT;
      const int col = n0 + i * 8 + c * 2;
      const int r0 = mb + t * 16 + g;
      if (r0 < M && (!MSPLIT || g < MR))
        store2(Y + (size_t)r0 * (size_t)N + col, acc[b + 0], acc[b + 1], col, N);
      if (HH == 2 && r0 + 8 < M)
        store2(Y + (size_t)(r0 + 8) * (size_t)N + col, acc[b + 2], acc[b + 3], col, N);
    }
  }
}

#undef FLASHINFER_SM120_A003_MMA
