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
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

namespace ffi = tvm::ffi;

void LaunchWide(const uint8_t* a, const uint8_t* b, const float* as, const float* bs,
                __nv_bfloat16* y, int M, int N, int K, int KB, cudaStream_t stream);
void LaunchAsyncM32N12288(const uint8_t* a, const uint8_t* b, const float* as, const float* bs,
                          __nv_bfloat16* y, int M, int N, int K, cudaStream_t stream);

#define MMA16832(d, a0, a1, a2, a3, b0, b1)                     \
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
          MMA16832(d, a0, a1, a2, a3, bf[i].v[2 * j], bf[i].v[2 * j + 1]);
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

#define LAUNCH(Wv, MRv, SPv, K1)       \
  gemm_fp8_bw_kernel<Wv, MRv, SPv, K1> \
      <<<grid, Wv * 32, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, mm, N, K, KB)

#define DISP_SP(Wv, MRv, K1)      \
  do {                            \
    if (msplit) {                 \
      LAUNCH(Wv, MRv, true, K1);  \
    } else {                      \
      LAUNCH(Wv, MRv, false, K1); \
    }                             \
  } while (0)

#define DISP_MR1(Wv, K1)         \
  do {                           \
    if (MR == 8) {               \
      DISP_SP(Wv, 8, K1);        \
    } else if (MR == 16) {       \
      DISP_SP(Wv, 16, K1);       \
    } else {                     \
      LAUNCH(Wv, 32, false, K1); \
    }                            \
  } while (0)

#define DISP_MR(Wv)          \
  do {                       \
    if (kiter == 1) {        \
      DISP_MR1(Wv, 1);       \
    } else if (kiter == 2) { \
      DISP_MR1(Wv, 2);       \
    } else {                 \
      DISP_MR1(Wv, 0);       \
    }                        \
  } while (0)

#define DISP_W()     \
  do {               \
    switch (W) {     \
      case 1:        \
        DISP_MR(1);  \
        break;       \
      case 2:        \
        DISP_MR(2);  \
        break;       \
      case 4:        \
        DISP_MR(4);  \
        break;       \
      case 8:        \
        DISP_MR(8);  \
        break;       \
      default:       \
        DISP_MR(16); \
        break;       \
    }                \
  } while (0)

void RunSpecial(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
                ffi::TensorView Y) {
  const int M = static_cast<int>(A.shape()[0]);
  const int K = static_cast<int>(A.shape()[1]);
  const int N = static_cast<int>(B.shape()[0]);
  const int KB = K >> 7;
  if (M <= 0 || N <= 0 || KB <= 0) return;

  const uint8_t* Abase = static_cast<const uint8_t*>(A.data_ptr());
  const uint8_t* Bp = static_cast<const uint8_t*>(B.data_ptr());
  const float* Asbase = static_cast<const float*>(As.data_ptr());
  const float* Bsp = static_cast<const float*>(Bs.data_ptr());
  __nv_bfloat16* Ybase = static_cast<__nv_bfloat16*>(Y.data_ptr());
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, A.device().device_id));

  if (M == 32 && N == 12288 && K == 2048) {
    LaunchAsyncM32N12288(Abase, Bp, Asbase, Bsp, Ybase, M, N, K, stream);
    return;
  }

  const bool use_wide = (M <= 8 && M != 4 && N == 2048 && K == 4096) ||
                        (M >= 2 && M <= 8 && N == 2048 && K == 512) || (M == 8 && N == 9216) ||
                        (M == 16 && ((N == 2048 && K == 4096) || (N == 2048 && K == 512))) ||
                        (M == 32 && (N == 1024 || N == 2048 || N == 9216));
  if (use_wide) {
    LaunchWide(Abase, Bp, Asbase, Bsp, Ybase, M, N, K, KB, stream);
    return;
  }

  if (M == 8 && N == 64 && K == 2048) {
    const dim3 grid(N / 8, M, 1u);
    gemm_fp8_bw_kernel<16, 1, true, 1>
        <<<grid, 512, 0, stream>>>(Abase, Bp, Asbase, Bsp, Ybase, M, N, K, KB);
    return;
  }

  if (M == 16 && N == 64 && K == 2048) {
    const dim3 grid(N / 8, M / 2, 1u);
    gemm_fp8_bw_kernel<16, 2, true, 1>
        <<<grid, 512, 0, stream>>>(Abase, Bp, Asbase, Bsp, Ybase, M, N, K, KB);
    return;
  }

  if (M == 16 && N == 9216 && K == 2048) {
    const dim3 grid((N + 7) >> 3, 1, 1);
    gemm_fp8_bw_kernel<4, 16, false, 0, true, true>
        <<<grid, 128, 0, stream>>>(Abase, Bp, Asbase, Bsp, Ybase, M, N, K, KB);
    return;
  }

  if (M == 16 && N == 12288 && K == 2048) {
    const dim3 grid((N + 7) >> 3, 1, 1);
    gemm_fp8_bw_kernel<4, 16, false, 0, true>
        <<<grid, 128, 0, stream>>>(Abase, Bp, Asbase, Bsp, Ybase, M, N, K, KB);
    return;
  }

  const int ntiles = (N + 7) >> 3;
  const int blocks = ntiles;
  int W = 1;
  const int wt = (9024 + blocks - 1) / blocks;
  while (W < wt && W < 16) W <<= 1;
  while (W > 1 && W > KB) W >>= 1;
  if ((M == 2 || M == 4 || M == 8) && N == 12288 && K == 2048) W = 4;
  if (M == 16 && N == 9216 && K == 2048) W = 4;
  if (M == 16 && N == 12288 && K == 2048) W = 4;

  int MR = (M <= 8) ? 8 : ((M <= 16) ? 16 : 32);
  // Halving the m chunk doubles the block count at the price of re-reading the
  // weights; only worth it while that stream is small.
  const bool msplit = (MR > 8) && (blocks < 376) && ((long long)N * (long long)K <= (1536LL << 10));
  if (msplit) MR >>= 1;

  // Fixed one- and two-group forms remove loop control on the common decode
  // schedules.  The unsplit MR16 one-group form measured slower, so it keeps
  // the generic loop.
  int kiter = 0;
  const bool kok = (MR != 16) || msplit || (W <= 8);
  if (W == KB && kok)
    kiter = 1;
  else if (2 * W == KB && kok)
    kiter = 2;
  if (M == 4 && N == 2048 && K == 4096) kiter = 0;

  if (msplit) {
    const dim3 grid(blocks, (unsigned)((M + MR - 1) / MR), 1);
    const uint8_t* Ap = Abase;
    const float* Asp = Asbase;
    __nv_bfloat16* Yp = Ybase;
    const int mm = M;
    DISP_W();
  } else {
    for (int m0 = 0; m0 < M; m0 += 32) {
      const int mm = (M - m0 < 32) ? (M - m0) : 32;
      MR = (mm <= 8) ? 8 : ((mm <= 16) ? 16 : 32);
      const dim3 grid(blocks, 1u, 1u);
      const uint8_t* Ap = Abase + (size_t)m0 * (size_t)K;
      const float* Asp = Asbase + (size_t)m0 * (size_t)KB;
      __nv_bfloat16* Yp = Ybase + (size_t)m0 * (size_t)N;
      DISP_W();
    }
  }
}
