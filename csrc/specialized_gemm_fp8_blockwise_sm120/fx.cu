// fx.cu -- low-latency fp8 e4m3 blockwise-scaled NT GEMM for sm120 decode rows.
//   Y[m,n] = sum_g As[m,g] * Bs[n/128,g] * sum_{k in group g} A[m,k]*B[n,k]
//
// Same direct-to-register mma formulation as the round-2 lineage (k permuted
// inside each 128-wide group so both operands land straight in fragments), but
// parameterised so a warp can hold KU k-groups in flight.  The decode rows are
// bound by one cold DRAM round trip plus the epilogue, so the interesting axis
// is "how few warps can still issue every load at once": raising KU and
// dropping NWK shortens (or removes) the cross-warp reduction without losing
// memory-level parallelism.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

namespace fx {

#define FXMMA(d, a0, a1, a2, a3, b0, b1)                                    \
  asm volatile(                                                             \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "                \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"             \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                      \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

struct F8 { uint32_t v[8]; };

__device__ __forceinline__ F8 ldb(const uint8_t* p) {
  F8 r;
  asm("ld.global.nc.L1::evict_first.L2::256B.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]),
        "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ F8 lda(const uint8_t* p) {
  F8 r;
  asm("ld.global.nc.L1::evict_last.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]),
        "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ float lds(const float* p) {
  float r;
  asm("ld.global.nc.L1::evict_last.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}

__device__ __forceinline__ void st2(__nv_bfloat16* p, float v0, float v1,
                                    int col, int N) {
  if (col + 1 < N) {
    __nv_bfloat162 v = __floats2bfloat162_rn(v0, v1);
    if ((reinterpret_cast<uintptr_t>(p) & 3u) == 0u)
      *reinterpret_cast<__nv_bfloat162*>(p) = v;
    else { p[0] = v.x; p[1] = v.y; }
  } else if (col < N) {
    p[0] = __float2bfloat16_rn(v0);
  }
}

template <int NWN, int NWK, int NT, int MR, int KU, int ITERS, int MINB>
__global__ void __launch_bounds__(NWN * NWK * 32, MINB) fxk(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bq,
    const float* __restrict__ As, const float* __restrict__ Bs,
    __nv_bfloat16* __restrict__ Y, int M, int N, int K, int KB) {
  constexpr int W = NWN * NWK;
  constexpr int CN = NWN * NT * 8;
  constexpr int MT = (MR + 15) / 16;
  constexpr int HH = (MR <= 8) ? 1 : 2;
  constexpr int NR = MT * HH;
  constexpr int ACT = HH * 2;
  constexpr int NACC = MT * NT * ACT;
  constexpr int THREADS = W * 32;
  constexpr int OUTS = NWN * NACC * 32;
  constexpr int RSZ = (NWK > 1) ? (W * NACC * 32) : 1;
  __shared__ float red[RSZ];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int ng = (NWN == 1) ? 0 : (warp % NWN);
  const int wk = (NWN == 1) ? warp : (warp / NWN);
  const int g = lane >> 2;
  const int c = lane & 3;
  const int nc = blockIdx.x * CN;
  const int n0 = nc + ng * (NT * 8);
  const int mb = blockIdx.y * MR;
  const int k0 = wk * KU * 128 + c * 32;
  const int kstep = NWK * KU * 128;

  // B addresses first so the DRAM round trip starts before anything touches A.
  const int nq0 = min(n0 + g, N - 1);
  const uint8_t* bp[NT];
  bp[0] = Bq + (size_t)nq0 * (size_t)K + k0;
#pragma unroll
  for (int i = 1; i < NT; ++i) {
    const int nqi = min(n0 + i * 8 + g, N - 1);
    bp[i] = bp[0] + (size_t)((nqi - nq0) * K);
  }

  const int rc0 = min(mb + g, M - 1);
  const uint8_t* ap[NR];
  const float* asp[NR];
  ap[0] = A + (size_t)rc0 * (size_t)K + k0;
  asp[0] = As + (size_t)rc0 * (size_t)KB + (wk * KU);
#pragma unroll
  for (int q = 1; q < NR; ++q) {
    const int rc = min(mb + (q / HH) * 16 + (q % HH) * 8 + g, M - 1);
    ap[q] = ap[0] + (size_t)((rc - rc0) * K);
    asp[q] = asp[0] + (size_t)((rc - rc0) * KB);
  }
  int nblk = n0 >> 7;
  const int nbmax = (N + 127) >> 7;
  if (nblk >= nbmax) nblk = nbmax - 1;
  const float* bsp = Bs + (size_t)nblk * (size_t)KB + (wk * KU);

  float acc[NACC];
#pragma unroll
  for (int a = 0; a < NACC; ++a) acc[a] = 0.f;

  auto body = [&]() {
    F8 bf[KU][NT];
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int i = 0; i < NT; ++i) bf[u][i] = ldb(bp[i] + u * 128);
    F8 af[KU][NR];
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int q = 0; q < NR; ++q) af[u][q] = lda(ap[q] + u * 128);
    float sc[KU][NR];
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      const float bv = lds(bsp + u);
#pragma unroll
      for (int q = 0; q < NR; ++q) sc[u][q] = lds(asp[q] + u) * bv;
    }
#pragma unroll
    for (int i = 0; i < NT; ++i) bp[i] += kstep;
#pragma unroll
    for (int q = 0; q < NR; ++q) { ap[q] += kstep; asp[q] += NWK * KU; }
    bsp += NWK * KU;

#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int t = 0; t < MT; ++t) {
          const F8& F0 = af[u][t * HH];
          const F8& F1 = af[u][t * HH + HH - 1];
          const uint32_t z = 0u;
          float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t a0 = F0.v[2 * j], a2 = F0.v[2 * j + 1];
            const uint32_t a1 = (HH == 2) ? F1.v[2 * j] : z;
            const uint32_t a3 = (HH == 2) ? F1.v[2 * j + 1] : z;
            FXMMA(d, a0, a1, a2, a3, bf[u][i].v[2 * j], bf[u][i].v[2 * j + 1]);
          }
          const int b = (t * NT + i) * ACT;
          acc[b + 0] += sc[u][t * HH] * d[0];
          acc[b + 1] += sc[u][t * HH] * d[1];
          if (HH == 2) {
            acc[b + 2] += sc[u][t * HH + 1] * d[2];
            acc[b + 3] += sc[u][t * HH + 1] * d[3];
          }
        }
  };

  if constexpr (ITERS > 0) {
#pragma unroll
    for (int it = 0; it < ITERS; ++it) body();
  } else {
    const int iters = KB / (NWK * KU);
#pragma unroll 1
    for (int it = 0; it < iters; ++it) body();
  }

  if (NWK == 1) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int b = (t * NT + i) * ACT;
        const int col = n0 + i * 8 + c * 2;
        const int r0 = mb + t * 16 + g;
        if (r0 < M)
          st2(Y + (size_t)r0 * (size_t)N + col, acc[b + 0], acc[b + 1], col, N);
        if (HH == 2 && r0 + 8 < M)
          st2(Y + (size_t)(r0 + 8) * (size_t)N + col, acc[b + 2], acc[b + 3],
              col, N);
      }
    return;
  }

  float* sw = red + warp * (NACC * 32);
#pragma unroll
  for (int a = 0; a < NACC; ++a) sw[a * 32 + lane] = acc[a];
  __syncthreads();

  constexpr int ROUNDS = (OUTS + THREADS - 1) / THREADS;
#pragma unroll
  for (int rd = 0; rd < ROUNDS; ++rd) {
    const int idx = tid + rd * THREADS;
    if (idx < OUTS) {
      const int ngi = (NWN == 1) ? 0 : (idx / (NACC * 32));
      const int rest = (NWN == 1) ? idx : (idx - ngi * (NACC * 32));
      const int a = rest >> 5;
      const int l = rest & 31;
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < NWK; ++w)
        s += red[(w * NWN + ngi) * (NACC * 32) + a * 32 + l];
      const int t = a / (NT * ACT);
      const int rem = a - t * (NT * ACT);
      const int i = rem / ACT;
      const int sl = rem - i * ACT;
      const int row = mb + t * 16 + ((sl >= 2) ? 8 : 0) + (l >> 2);
      const int col = nc + ngi * (NT * 8) + i * 8 + (l & 3) * 2 + (sl & 1);
      if (row < M && col < N)
        Y[(size_t)row * (size_t)N + col] = __float2bfloat16_rn(s);
    }
  }
}

// ------------------------------------------------------------------ dispatch
//      id, NWN, NWK, NT, MR, KU, ITERS, MINB
#define FX_CFGS(X)                   \
  X( 0,  1, 16, 1,  8, 1,  1, 1)     \
  X( 1,  1,  8, 1,  8, 2,  1, 1)     \
  X( 2,  1,  4, 1,  8, 4,  1, 1)     \
  X( 3,  1,  2, 1,  8, 8,  1, 1)     \
  X( 4,  1, 16, 2,  8, 1,  1, 1)     \
  X( 5,  1,  8, 2,  8, 2,  1, 1)     \
  X( 6,  1,  8, 2,  8, 1,  2, 1)     \
  X( 7,  1,  4, 2,  8, 2,  2, 1)     \
  X( 8,  1,  1, 1,  8, 1, 16, 1)     \
  X( 9,  1,  2, 1,  8, 1,  8, 1)     \
  X(10,  1,  4, 1,  8, 1,  4, 1)     \
  X(11,  1,  1, 2,  8, 1, 16, 1)     \
  X(12,  1,  8, 1,  8, 1,  2, 1)     \
  X(13,  1,  2, 2,  8, 2,  4, 1)     \
  X(14,  1, 16, 2, 16, 1,  1, 1)     \
  X(15,  1,  8, 2, 16, 2,  1, 1)     \
  X(16,  1,  8, 2, 16, 1,  2, 1)     \
  X(17,  1,  4, 2, 16, 2,  2, 1)     \
  X(18,  1, 16, 1, 16, 1,  1, 1)     \
  X(19,  1,  4, 1, 16, 4,  1, 1)     \
  X(20,  1,  1, 1, 32, 1, 16, 1)     \
  X(21,  1,  1, 2, 32, 1, 16, 1)     \
  X(22,  1,  8, 2, 32, 2,  1, 1)     \
  X(23,  1,  4, 2, 32, 2,  2, 1)     \
  X(24,  1,  8, 1, 32, 1,  2, 1)     \
  X(25,  4,  4, 1,  8, 1,  1, 1)     \
  X(26,  1,  1, 1,  8, 4,  1, 1)     \
  X(27,  2,  1, 1,  8, 4,  1, 1)     \
  X(28,  4,  1, 1,  8, 4,  1, 1)     \
  X(29,  1,  2, 1,  8, 2,  1, 1)     \
  X(30,  1,  4, 1,  8, 1,  1, 1)     \
  X(31,  2,  4, 1,  8, 1,  1, 1)     \
  X(32,  2,  2, 1,  8, 2,  1, 1)     \
  X(33,  4,  4, 1, 16, 1,  1, 1)     \
  X(34,  1,  1, 1, 16, 4,  1, 1)     \
  X(35,  2,  4, 1, 16, 1,  1, 1)     \
  X(36,  1,  1, 1, 32, 4,  1, 1)     \
  X(37,  2,  4, 1, 32, 1,  1, 1)     \
  X(38,  1, 16, 2,  8, 2,  1, 1)     \
  X(39,  2,  8, 2,  8, 1,  4, 1)     \
  X(40,  1,  8, 2,  8, 4,  1, 1)     \
  X(41,  1,  8, 1,  8, 4,  1, 1)     \
  X(42,  1, 16, 1,  8, 2,  1, 1)     \
  X(43,  1,  8, 2,  8, 2,  2, 1)     \
  X(44,  1, 16, 2, 16, 1,  2, 1)     \
  X(45,  1,  8, 2, 16, 2,  2, 1)     \
  X(46,  2,  4, 2, 32, 1,  8, 1)     \
  X(47,  1,  8, 2, 32, 2,  2, 1)     \
  X(48,  1,  1, 1,  8, 1,  0, 1)     \
  X(49,  1,  1, 1, 16, 1,  0, 1)     \
  X(50,  1,  1, 1, 32, 1,  0, 1)     \
  X(51,  1,  4, 1,  8, 1,  0, 1)     \
  X(52,  1,  4, 1, 16, 1,  0, 1)     \
  X(53,  1,  4, 1, 32, 1,  0, 1)     \
  X(54,  1, 16, 1,  8, 1,  0, 1)     \
  X(55,  1, 16, 1, 16, 1,  0, 1)     \
  X(56,  1, 16, 1, 32, 1,  0, 1)     \
  X(57,  1,  2, 1,  8, 1,  0, 1)     \
  X(58,  1,  2, 1, 16, 1,  0, 1)     \
  X(59,  1,  8, 1,  8, 1,  0, 1)     \
  X(60,  1,  8, 1, 16, 1,  0, 1)     \
  X(61,  1,  8, 1, 32, 1,  0, 1)     \
  X(62,  1,  2, 1, 32, 1,  0, 1)     \
  X(63,  8,  4, 1,  8, 1,  1, 1)     \
  X(64,  4,  4, 2,  8, 1,  1, 1)     \
  X(65,  1, 16, 2,  8, 1,  2, 1)     \
  X(66,  1, 16, 1,  8, 1,  2, 1)     \
  X(67,  1,  8, 2,  8, 1,  4, 1)     \
  X(68,  1, 16, 1, 16, 1,  2, 1)     \
  X(69,  1, 16, 2, 16, 1,  1, 1)     \
  X(70,  2,  8, 1,  8, 1,  0, 1)     \
  X(71,  2,  8, 1, 16, 1,  0, 1)

#define FX_NCFG 72
static const unsigned char FX_NWK[FX_NCFG] = {
  16,  8,  4,  2, 16,  8,  8,  4,  1,  2,  4,  1,  8,  2, 16,  8,  8,  4, 16,  4,
   1,  1,  8,  4,  8,  4,  1,  1,  1,  2,  4,  4,  2,  4,  1,  4,  1,  4, 16,  8,
   8,  8, 16,  8, 16,  8,  4,  8,  1,  1,  1,  4,  4,  4, 16, 16, 16,  2,  2,  8,
   8,  8,  2,  4,  4, 16, 16,  8, 16, 16,  8,  8};
static const unsigned char FX_IT[FX_NCFG] = {
   1,  1,  1,  1,  1,  1,  2,  2, 16,  8,  4, 16,  2,  4,  1,  1,  2,  2,  1,  1,
  16, 16,  1,  2,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  4,
   1,  1,  1,  2,  2,  2,  8,  2,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  1,  1,  2,  2,  4,  2,  1,  0,  0};
static const unsigned char FX_KU[FX_NCFG] = {
   1,  2,  4,  8,  1,  2,  1,  2,  1,  1,  1,  1,  1,  2,  1,  2,  1,  2,  1,  4,
   1,  1,  2,  2,  1,  1,  4,  4,  4,  2,  1,  1,  2,  1,  4,  1,  4,  1,  2,  1,
   4,  4,  2,  2,  1,  2,  1,  2,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
   1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1};

// [shape class][m class] -> config id.
//  cls: 0 N64K2048, 1 N1024K2048, 2 N2048K512, 3 N2048K4096,
//       4 N9216K2048, 5 N12288K2048, 6 generic
#ifndef FX_TABLE_V
#define FX_TABLE_V 1
#endif

#if FX_TABLE_V == 1
static const signed char FX_BEST[7][6] = {
  { 0,  0,  0,  0,  0,  0},
  { 0,  0,  0,  0,  0,  0},
  {25, 25, 25, 25, 25, 25},
  {65, 65, 65, 65, 44, 46},
  { 8,  8,  8, 59, 52, 53},
  { 6,  6,  6,  6, 52, 53},
  {-1, -1, -1, -1, -1, -1},
};
#endif

static inline int fx_cls(int N, int K) {
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

template <int NWN, int NWK, int NT, int MR, int KU, int ITERS, int MINB>
static inline void fx_go(const uint8_t* A, const uint8_t* B, const float* As,
                         const float* Bs, __nv_bfloat16* Y, int M, int N, int K,
                         int KB, cudaStream_t s) {
  constexpr int CN = NWN * NT * 8;
  const dim3 grid((N + CN - 1) / CN, (unsigned)((M + MR - 1) / MR), 1);
  fxk<NWN, NWK, NT, MR, KU, ITERS, MINB>
      <<<grid, NWN * NWK * 32, 0, s>>>(A, B, As, Bs, Y, M, N, K, KB);
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

  const int cls = fx_cls(N, K);
  const int mc = (M <= 1) ? 0 : (M <= 2) ? 1 : (M <= 4) ? 2
               : (M <= 8) ? 3 : (M <= 16) ? 4 : 5;
  const int mr = (M <= 8) ? 0 : ((M <= 16) ? 1 : 2);
  int id = (cls == 6 || M > 32) ? -1 : (int)FX_BEST[cls][mc];
  if (id < 0) {
    int ntiles = (N + 7) / 8, nwk = 1;
    while (nwk < 16 && ntiles * nwk * 8 < 9024) nwk <<= 1;
    while (nwk > 1 && (nwk > KB || KB % nwk != 0)) nwk >>= 1;
    id = (nwk >= 16 ? 54 : (nwk >= 4 ? 51 : 48)) + mr;
  }
  const int tot = FX_NWK[id] * FX_KU[id];
  if (KB % tot != 0 || (FX_IT[id] != 0 && FX_IT[id] != KB / tot)) id = 48 + mr;

  switch (id) {
#define X(ID, NWN, NWK, NT, MR, KU, ITERS, MINB)                             \
  case ID:                                                                   \
    fx_go<NWN, NWK, NT, MR, KU, ITERS, MINB>(Ap, Bp, Asp, Bsp, Yp, M, N, K,  \
                                             KB, stream);                    \
    break;
    FX_CFGS(X)
#undef X
    default:
      fx_go<1, 1, 1, 8, 1, 0, 1>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, stream);
      break;
  }
}

}  // namespace fx

void RunFX(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As,
           ffi::TensorView Bs, ffi::TensorView Y) {
  fx::Run(A, B, As, Bs, Y);
}


