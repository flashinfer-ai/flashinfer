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
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

namespace a003_tuned_impl {

#define MMA16832(d, a0, a1, a2, a3, b0, b1)                                 \
  asm volatile(                                                             \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "                \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"             \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                      \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

struct Frag8 { uint32_t v[8]; };

__device__ __forceinline__ Frag8 ld_stream(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::evict_first.L2::256B.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]),
        "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ Frag8 ld_keep(const void* p) {
  Frag8 r;
  asm("ld.global.nc.L1::evict_last.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]),
        "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ float ld_keepf(const float* p) {
  float r;
  asm("ld.global.nc.L1::evict_last.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ void store2(__nv_bfloat16* p, float v0, float v1,
                                       int col, int N) {
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
template <int NW_N, int NW_K, int NT, int MR, int KU, int ITERS, int MINB = 1,
          bool ASB = false>
__global__ void __launch_bounds__(NW_N * NW_K * 32, MINB) gemm_bw(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bq,
    const float* __restrict__ As, const float* __restrict__ Bs,
    __nv_bfloat16* __restrict__ Y, int M, int N, int K, int KB) {
  constexpr int W = NW_N * NW_K;
  constexpr int CN = NW_N * NT * 8;
  constexpr int MT = (MR + 15) / 16;     // 16-row m-tiles
  constexpr int HH = (MR <= 8) ? 1 : 2;  // 8-row halves actually live
  constexpr int NR = MT * HH;            // A row slots per thread
  constexpr int ACT = HH * 2;            // accumulators per (mtile, ntile)
  constexpr int NACC = MT * NT * ACT;
  constexpr int RSZ = (NW_K > 1) ? (W * NACC * 32) : 1;
  __shared__ float red[RSZ];

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
    bp[i] = Bq + (size_t)(nq < N ? nq : 0) * (size_t)K +
            (size_t)(wk * KU * 128 + c * 32);
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
        if constexpr (ASB) {
          float asv = (c == 0) ? ld_keepf(asp[q] + u) : 0.f;
          asv = __shfl_sync(0xffffffffu, asv, g * 4);
          sc[u][q] = asv * bsv;
        } else {
          sc[u][q] = ld_keepf(asp[q] + u) * bsv;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) bp[i] += kstep;
#pragma unroll
    for (int q = 0; q < NR; ++q) { ap[q] += kstep; asp[q] += NW_K * KU; }
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
        if (r0 < M)
          store2(Y + (size_t)r0 * (size_t)N + col, acc[b + 0], acc[b + 1], col, N);
        if (HH == 2 && r0 + 8 < M)
          store2(Y + (size_t)(r0 + 8) * (size_t)N + col, acc[b + 2], acc[b + 3],
                 col, N);
      }
    return;
  }

  // Cross-warp k reduction: every warp parks its accumulators, then all
  // threads cooperatively sum the NW_K partials for a slice of the tile.
  float* sw = red + (size_t)warp * NACC * 32;
#pragma unroll
  for (int a = 0; a < NACC; ++a) sw[a * 32 + lane] = acc[a];
  __syncthreads();
  constexpr int OUTS = NW_N * NACC * 32;  // (ng, acc, lane) triples
  constexpr int THREADS = W * 32;
  for (int idx = threadIdx.x; idx < OUTS; idx += THREADS) {
    const int ngi = idx / (NACC * 32);
    const int rest = idx - ngi * (NACC * 32);
    const int a = rest >> 5;
    const int l = rest & 31;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NW_K; ++w)
      s += red[((size_t)(w * NW_N + ngi) * NACC + a) * 32 + l];
    const int t = a / (NT * ACT);
    const int rem = a - t * (NT * ACT);
    const int i = rem / ACT;
    const int sl = rem - i * ACT;
    const int row = mb + t * 16 + ((sl >= 2) ? 8 : 0) + (l >> 2);
    const int col = nc + ngi * (NT * 8) + i * 8 + (l & 3) * 2 + (sl & 1);
    if (row < M && col < N) Y[(size_t)row * (size_t)N + col] = __float2bfloat16_rn(s);
  }
}

// ---------------------------------------------------------------- dispatch
template <int NW_N, int NW_K, int NT, int KU, int ITERS, int MINB>
static inline void go2(const uint8_t* A, const uint8_t* B, const float* As,
                       const float* Bs, __nv_bfloat16* Y, int M, int N, int K,
                       int KB, int MR, cudaStream_t s) {
  constexpr int CN = NW_N * NT * 8;
  constexpr int TH = NW_N * NW_K * 32;
  const dim3 grid((N + CN - 1) / CN, (unsigned)((M + MR - 1) / MR), 1);
  if (MR == 8)
    gemm_bw<NW_N, NW_K, NT, 8, KU, ITERS, MINB><<<grid, TH, 0, s>>>(A, B, As, Bs, Y, M, N, K, KB);
  else if (MR == 16)
    gemm_bw<NW_N, NW_K, NT, 16, KU, ITERS, MINB><<<grid, TH, 0, s>>>(A, B, As, Bs, Y, M, N, K, KB);
  else
    gemm_bw<NW_N, NW_K, NT, 32, KU, ITERS, MINB><<<grid, TH, 0, s>>>(A, B, As, Bs, Y, M, N, K, KB);
}

template <int NW_N, int NW_K, int NT, int KU, int MINB>
static inline void go(const uint8_t* A, const uint8_t* B, const float* As,
                      const float* Bs, __nv_bfloat16* Y, int M, int N, int K,
                      int KB, int MR, cudaStream_t s) {
  const int iters = KB / (NW_K * KU);
  if (iters == 1)
    go2<NW_N, NW_K, NT, KU, 1, MINB>(A, B, As, Bs, Y, M, N, K, KB, MR, s);
  else if (iters == 2)
    go2<NW_N, NW_K, NT, KU, 2, MINB>(A, B, As, Bs, Y, M, N, K, KB, MR, s);
  else
    go2<NW_N, NW_K, NT, KU, 0, MINB>(A, B, As, Bs, Y, M, N, K, KB, MR, s);
}

//      id, NW_N, NW_K, NT, KU, MINB
#define CFG_LIST(X)        \
  X(0,  1,  1, 1, 1, 1)    \
  X(1,  1,  2, 1, 1, 1)    \
  X(2,  1,  4, 1, 1, 1)    \
  X(3,  1,  8, 1, 1, 1)    \
  X(4,  1, 16, 1, 1, 1)    \
  X(5,  1,  8, 1, 2, 1)    \
  X(8,  1,  8, 2, 1, 1)    \
  X(9,  1, 16, 2, 1, 1)    \
  X(12, 1,  8, 2, 2, 1)    \
  X(16, 1,  2, 1, 2, 1)    \
  X(17, 4,  4, 1, 1, 1)    \
  X(19, 1,  4, 2, 1, 1)    \
  X(20, 2,  8, 2, 1, 1)    \
  X(21, 1, 16, 4, 1, 1)    \
  X(22, 1,  4, 4, 1, 1)    \
  X(23, 1,  2, 2, 1, 1)    \
  X(24, 1,  1, 2, 1, 1)    \
  X(25, 2,  4, 2, 1, 1)    \
  X(26, 1,  4, 1, 2, 1)    \
  X(27, 1,  2, 4, 1, 1)    \
  X(28, 1, 32, 1, 1, 1)    \
  X(29, 2, 16, 1, 1, 1)    \
  X(30, 1, 16, 1, 2, 1)    \
  X(32, 2,  8, 2, 2, 1)    \
  X(33, 1,  8, 1, 4, 1)    \
  X(34, 4,  8, 1, 1, 1)    \
  X(35, 1, 16, 2, 2, 1)    \
  X(36, 1,  8, 2, 1, 3)    \
  X(37, 1,  2, 1, 1, 12)

#define CFG_MAX 38
static const unsigned char CFG_NWK[CFG_MAX] = {
  1,2,4,8,16,8,0,0,8,16,0,0,8,0,0,0,2,4,0,4,8,16,4,2,1,4,4,2,
  32,16,16,0,8,8,8,16,8,2};
static const unsigned char CFG_KU[CFG_MAX] = {
  1,1,1,1,1,2,0,0,1,1,0,0,2,0,0,0,2,1,0,1,1,1,1,1,1,1,2,1,
  1,1,2,0,2,4,1,2,1,1};

#ifndef VARIANT
#define VARIANT 0
#endif

struct Sel { int id, mr; };

// Best known tiling per (shape class, m class): {config id, m-split shift}.
// A CTA owns CN = NW_N*NT*8 columns and all of K; NW_K warps split the k
// groups with KU of them in flight.  The m-split shift halves the rows per CTA
// to buy blocks, which only pays where the weight stream is small.
static const signed char BEST[7][6][2] = {
  {{4,1},{4,1},{4,1},{4,1},{5,1},{3,2}},
  {{4,0},{4,2},{9,0},{9,0},{9,1},{8,1}},
  {{17,2},{17,0},{17,0},{17,0},{17,1},{17,1}},
  {{35,0},{35,0},{20,0},{20,0},{9,0},{25,0}},
  {{0,0},{0,0},{0,0},{37,0},{12,0},{0,0}},
  {{8,0},{8,0},{8,0},{8,0},{12,0},{24,0}},
  {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}},
};

#if VARIANT > 0
// Experiment slots: VARIANT v applies ALT[cls][v-1] to every m class.
static const signed char ALT[7][8][2] = {
  /* N=64    */ {{5,1},{4,1},{3,1},{5,2},{3,2},{3,2},{3,2},{3,2}},
  /* N=1024  */ {{12,1},{19,0},{23,0},{24,0},{9,0},{8,0},{4,1},{12,0}},
  /* N=2048/512 */ {{22,1},{19,1},{27,0},{25,1},{17,0},{2,1},{23,1},{25,0}},
  /* N=2048/4096*/ {{35,0},{30,0},{21,0},{3,0},{9,0},{12,0},{5,0},{20,0}},
  /* N=9216  */ {{5,0},{3,0},{16,0},{19,0},{12,0},{8,0},{1,0},{2,0}},
  /* N=12288 */ {{5,0},{3,0},{16,0},{19,0},{12,0},{8,0},{1,0},{2,0}},
  /* generic */ {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}},
};
#endif

static inline int shape_class(int N, int K) {
  if (K == 2048) {
    if (N == 64) return 0;
    if (N == 1024) return 1;
    if (N == 9216) return 4;
    if (N == 12288) return 5;
  }
  if (N == 2048) {
    if (K == 512) return 2;
    if (K == 4096) return 3;
  }
  return 6;
}

static inline Sel pick(int M, int N, int K, int KB) {
  int mr = (M <= 8) ? 8 : ((M <= 16) ? 16 : 32);
  const int cls = shape_class(N, K);
  const int mc = (M <= 1) ? 0 : (M <= 2) ? 1 : (M <= 4) ? 2
               : (M <= 8) ? 3 : (M <= 16) ? 4 : 5;
  int id = BEST[cls][mc][0];
  int shift = BEST[cls][mc][1];
#if VARIANT > 0
  id = ALT[cls][VARIANT - 1][0];
  shift = ALT[cls][VARIANT - 1][1];
#endif
  if (cls == 6 || M > 32) {
    int ntiles = (N + 7) / 8, nwk = 1;
    while (nwk < 16 && ntiles * nwk * 8 < 9024) nwk <<= 1;
    while (nwk > 1 && (nwk > KB || KB % nwk != 0)) nwk >>= 1;
    id = (nwk == 1) ? 0 : (nwk == 2) ? 1 : (nwk == 4) ? 2 : (nwk == 8) ? 3 : 4;
    shift = 0;
  }
  while (shift > 0 && mr > 8) { mr >>= 1; --shift; }
  const int nwk = CFG_NWK[id], ku = CFG_KU[id];
  // The k loop runs a fixed number of KU-group steps, so the split must
  // divide KB exactly; id 0 (one warp, one group per step) always does.
  if (!nwk || KB % (nwk * ku) != 0) id = 0;
  return Sel{id, mr};
}

void Run(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As,
         ffi::TensorView Bs, ffi::TensorView Y) {
  const int M = static_cast<int>(A.shape()[0]);
  const int K = static_cast<int>(A.shape()[1]);
  const int N = static_cast<int>(B.shape()[0]);
  const int KB = K >> 7;
  if (M <= 0 || N <= 0 || KB <= 0) return;

  const uint8_t* Ap = static_cast<const uint8_t*>(A.data_ptr());
  const uint8_t* Bp = static_cast<const uint8_t*>(B.data_ptr());
  const float* Asp = static_cast<const float*>(As.data_ptr());
  const float* Bsp = static_cast<const float*>(Bs.data_ptr());
  __nv_bfloat16* Yp = static_cast<__nv_bfloat16*>(Y.data_ptr());
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, A.device().device_id));

  if (M == 16 && N == 12288 && K == 2048) {
    const dim3 grid(768, 1, 1);
    gemm_bw<1, 8, 2, 16, 2, 1, 1, true>
        <<<grid, 256, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);
    return;
  }
  if (M == 16 && N == 2048 && K == 4096) {
    const dim3 grid(64, 1, 1);
    gemm_bw<2, 8, 2, 16, 1, 4, 1, true>
        <<<grid, 512, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);
    return;
  }

  const Sel s = pick(M, N, K, KB);
  switch (s.id) {
#define X(ID, NWN, NWK, NT, KU, MINB)                                        \
  case ID:                                                                   \
    go<NWN, NWK, NT, KU, MINB>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, s.mr,      \
                               stream);                                      \
    break;
    CFG_LIST(X)
#undef X
    default:
      go<1, 1, 1, 1, 1>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, s.mr, stream);
      break;
  }
}

}  // namespace a003_tuned_impl

void RunTuned(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As,
              ffi::TensorView Bs, ffi::TensorView Y) {
  a003_tuned_impl::Run(A, B, As, Bs, Y);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run_tuned, RunTuned);




