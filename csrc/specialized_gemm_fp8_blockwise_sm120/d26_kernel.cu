// fp8 e4m3 blockwise-scaled NT GEMM, sm120 decode shapes.
//   Y[m,n] = sum_g As[m,g] * Bs[n/128,g] * sum_{k in group g} A[m,k]*B[n,k]
//
// Warp-level mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32.  Inside one
// 128-wide k group the reduction order is free, so k is permuted so that both
// operands land in fragment registers from one 256-bit contiguous load -- no
// shared staging and no ldmatrix in the k loop.
//
// Every decode shape here is weight-bandwidth bound behind a fixed ~1.8us
// launch cost, so the tile is picked to keep the SMs saturated with in-flight
// weight loads while holding the activation re-reads down (NT n8-tiles per
// warp amortise the A fragment loads over more weight columns).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

namespace ffi = tvm::ffi;

namespace d26_impl {

struct F8 {
  uint32_t v[8];
};

__device__ __forceinline__ F8 ldB(const void* p) {
  F8 r;
  asm("ld.global.nc.L1::no_allocate.v8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]),
        "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ F8 ldB_global(const void* p) {
  F8 r;
  asm("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]), "=r"(r.v[5]),
        "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ F8 ldA(const void* p) {
  F8 r;
  asm("ld.global.nc.L1::evict_last.L2::256B.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
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

// The result is never read back by this kernel, so stream it past the caches
// and leave L2 to the weight and activation streams.
__device__ __forceinline__ void stY(__nv_bfloat16* p, float v) {
  const __nv_bfloat16 b = __float2bfloat16_rn(v);
  unsigned short u;
  __builtin_memcpy(&u, &b, 2);
  __stwt(reinterpret_cast<unsigned short*>(p), u);
}

#define MMA(d, a0, a1, a2, a3, b0, b1)                          \
  asm volatile(                                                 \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "    \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])          \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// NW warps per CTA, NWN groups along n, NWK = NW/NWN along k.  KI > 0 pins the
// per-warp k-group trip count so the loop control and pointer bumps fold away.
template <int NW, int NWN, int NT, int MR, int KI, int MINB, bool SWAPG, int ACTIVE_M = MR,
          bool COHERENT_B = false, bool RED4 = false>
__global__ void __launch_bounds__(NW * 32, MINB)
    gk(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bq, const float* __restrict__ As,
       const float* __restrict__ Bs, __nv_bfloat16* __restrict__ Y, int M, int N, int K, int KB) {
  constexpr int NWK = NW / NWN;
  constexpr int MT = (MR <= 8) ? 1 : (MR / 16);
  constexpr int HH = (MR <= 8) ? 1 : 2;
  constexpr int NR = MT * HH;
  constexpr int CN = NWN * NT * 8;
  constexpr int CNP = CN + 4;
  constexpr int RSZ = (NWK > 1) ? (NWK * MR * CNP) : 1;
  __shared__ float sm[RSZ];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int ng = (NWN == 1) ? 0 : (wid / NWK);
  const int wk = wid - ng * NWK;
  const int g = lane >> 2;
  const int c = lane & 3;

  // When m is chunked, the chunks index the fast grid axis so the CTAs that
  // re-read the same weight columns launch back to back and share them while
  // they are still in L2.  A single chunk keeps n on the fast axis: a grid of
  // width 1 distributes measurably worse.
  const int nc = (int)(SWAPG ? blockIdx.y : blockIdx.x) * CN;
  const int n0 = nc + ng * (NT * 8);
  const int m0 = (int)(SWAPG ? blockIdx.x : blockIdx.y) * ACTIVE_M;

  // Out-of-range rows/cols read a live address and are dropped at store time:
  // an mma output element depends only on its own A row and B column.
  const uint8_t* bp[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int r = n0 + t * 8 + g;
    bp[t] = Bq + (size_t)(r < N ? r : 0) * (size_t)K + (size_t)(c * 32) + (size_t)wk * 128;
  }
  const uint8_t* ap[NR];
  const float* asp[NR];
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    const int lr = (q / HH) * 16 + (q % HH) * 8 + g;
    const int r = m0 + lr;
    const int rc = (lr < ACTIVE_M && r < M) ? r : 0;
    ap[q] = A + (size_t)rc * (size_t)K + (size_t)(c * 32) + (size_t)wk * 128;
    asp[q] = As + (size_t)rc * (size_t)KB + wk;
  }
  const float* bsp = Bs + (size_t)(nc >> 7) * (size_t)KB + wk;

  float acc[MT][NT][4];
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[t][i][r] = 0.f;

  auto body = [&]() {
    const float bsv = ldf(bsp);
    bsp += NWK;
    F8 bf[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      if constexpr (COHERENT_B)
        bf[t] = ldB_global(bp[t]);
      else
        bf[t] = ldB(bp[t]);
      bp[t] += NWK * 128;
    }
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

  if (NWK == 1) {
#pragma unroll
    for (int t = 0; t < MT; ++t)
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        const int col = n0 + i * 8 + c * 2;
        const int r0 = m0 + t * 16 + g;
        if (t * 16 + g < ACTIVE_M && r0 < M && col < N) {
          stY(Y + (size_t)r0 * N + col, acc[t][i][0]);
          if (col + 1 < N) stY(Y + (size_t)r0 * N + col + 1, acc[t][i][1]);
        }
        if (HH == 2 && t * 16 + g + 8 < ACTIVE_M && r0 + 8 < M && col < N) {
          stY(Y + (size_t)(r0 + 8) * N + col, acc[t][i][2]);
          if (col + 1 < N) stY(Y + (size_t)(r0 + 8) * N + col + 1, acc[t][i][3]);
        }
      }
    return;
  }

  float* sw = sm + (size_t)wk * MR * CNP;
#pragma unroll
  for (int t = 0; t < MT; ++t)
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int col = ng * (NT * 8) + i * 8 + c * 2;
      const int r0 = t * 16 + g;
      sw[r0 * CNP + col] = acc[t][i][0];
      sw[r0 * CNP + col + 1] = acc[t][i][1];
      if (HH == 2) {
        sw[(r0 + 8) * CNP + col] = acc[t][i][2];
        sw[(r0 + 8) * CNP + col + 1] = acc[t][i][3];
      }
    }
  __syncthreads();
  for (int idx = tid; idx < MR * CN; idx += NW * 32) {
    const int row = idx / CN;
    const int col = idx - row * CN;
    float s;
    if constexpr (RED4) {
      static_assert((NWK & 3) == 0);
      float s0 = sm[(0 * MR + row) * CNP + col];
      float s1 = sm[(1 * MR + row) * CNP + col];
      float s2 = sm[(2 * MR + row) * CNP + col];
      float s3 = sm[(3 * MR + row) * CNP + col];
#pragma unroll
      for (int w = 4; w < NWK; w += 4) {
        s0 += sm[((w + 0) * MR + row) * CNP + col];
        s1 += sm[((w + 1) * MR + row) * CNP + col];
        s2 += sm[((w + 2) * MR + row) * CNP + col];
        s3 += sm[((w + 3) * MR + row) * CNP + col];
      }
      s = (s0 + s1) + (s2 + s3);
    } else {
      s = 0.f;
#pragma unroll
      for (int w = 0; w < NWK; ++w) s += sm[(w * MR + row) * CNP + col];
    }
    const int gr = m0 + row, gc = nc + col;
    if (row < ACTIVE_M && gr < M && gc < N) stY(Y + (size_t)gr * N + gc, s);
  }
}

// Exact M16/N64/K2048 path.  Two live rows per CTA double the grid to 64 CTAs;
// fixed strides and a dense [k-warp][row][column] mailbox remove the generic
// bounds and padded-reduction overhead on this strongly underfilled shape.
template <int ACTIVE_ROWS, bool COHERENT_B, bool SCALE_BCAST = false>
__global__ void __launch_bounds__(256, 2)
    gk_m16_n64(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bq,
               const float* __restrict__ As, const float* __restrict__ Bs,
               __nv_bfloat16* __restrict__ Y) {
  __shared__ float sm[8 * ACTIVE_ROWS * 8];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wk = tid >> 5;
  const int g = lane >> 2;
  const int c = lane & 3;
  const int m0 = (int)blockIdx.x * ACTIVE_ROWS;
  const int n0 = (int)blockIdx.y * 8;
  const int ar = m0 + (g & (ACTIVE_ROWS - 1));

  float a0 = 0.f, a1 = 0.f;
#pragma unroll
  for (int it = 0; it < 2; ++it) {
    const int kg = wk + it * 8;
    const uint8_t* bp = Bq + (size_t)(n0 + g) * 2048 + (size_t)c * 32 + (size_t)kg * 128;
    const F8 bf = COHERENT_B ? ldB_global(bp) : ldB(bp);
    const F8 af = ldA(A + (size_t)ar * 2048 + (size_t)c * 32 + (size_t)kg * 128);
    float asv, bsv;
    if constexpr (SCALE_BCAST) {
      const int as_src = (g & (ACTIVE_ROWS - 1)) * 4;
      asv = (lane == as_src) ? ldf(As + (size_t)ar * 16 + kg) : 0.f;
      asv = __shfl_sync(0xffffffffu, asv, as_src);
      bsv = (lane == 0) ? ldf(Bs + kg) : 0.f;
      bsv = __shfl_sync(0xffffffffu, bsv, 0);
    } else {
      asv = ldf(As + (size_t)ar * 16 + kg);
      bsv = ldf(Bs + kg);
    }
    const float sc = asv * bsv;
    float d[4] = {0.f, 0.f, 0.f, 0.f};
    const uint32_t z = 0u;
#pragma unroll
    for (int j = 0; j < 4; ++j)
      MMA(d, af.v[2 * j], z, af.v[2 * j + 1], z, bf.v[2 * j], bf.v[2 * j + 1]);
    a0 += sc * d[0];
    a1 += sc * d[1];
  }

  if (g < ACTIVE_ROWS) {
    sm[(wk * ACTIVE_ROWS + g) * 8 + c * 2] = a0;
    sm[(wk * ACTIVE_ROWS + g) * 8 + c * 2 + 1] = a1;
  }
  __syncthreads();

  if (tid < ACTIVE_ROWS * 8) {
    const int row = tid >> 3;
    const int col = tid & 7;
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) s += sm[(w * ACTIVE_ROWS + row) * 8 + col];
    stY(Y + (size_t)(m0 + row) * 64 + n0 + col, s);
  }
}

#define LAUNCH(NWv, NWNv, NTv, MRv, KIv, MINBv)                               \
  do {                                                                        \
    const unsigned nt = (N + (NWNv * NTv * 8) - 1) / (NWNv * NTv * 8);        \
    const unsigned mc = (unsigned)((M + MRv - 1) / MRv);                      \
    if (mc > 1) {                                                             \
      const dim3 grid(mc, nt, 1);                                             \
      gk<NWv, NWNv, NTv, MRv, KIv, MINBv, true>                               \
          <<<grid, NWv * 32, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB); \
    } else {                                                                  \
      const dim3 grid(nt, 1, 1);                                              \
      gk<NWv, NWNv, NTv, MRv, KIv, MINBv, false>                              \
          <<<grid, NWv * 32, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB); \
    }                                                                         \
    return;                                                                   \
  } while (0)

#define LAUNCHN(NWv, NWNv, NTv, MRv, KIv, MINBv)                            \
  do {                                                                      \
    const dim3 grid((N + (NWNv * NTv * 8) - 1) / (NWNv * NTv * 8),          \
                    (unsigned)((M + MRv - 1) / MRv), 1);                    \
    gk<NWv, NWNv, NTv, MRv, KIv, MINBv, false>                              \
        <<<grid, NWv * 32, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB); \
    return;                                                                 \
  } while (0)

#define BYMRN(NWv, NWNv, NTv, KIv, MINBv)      \
  do {                                         \
    if (mr == 8) {                             \
      LAUNCHN(NWv, NWNv, NTv, 8, KIv, MINBv);  \
    } else if (mr == 16) {                     \
      LAUNCHN(NWv, NWNv, NTv, 16, KIv, MINBv); \
    } else {                                   \
      LAUNCHN(NWv, NWNv, NTv, 32, KIv, MINBv); \
    }                                          \
  } while (0)

#define BYMR(NWv, NWNv, NTv, KIv, MINBv)      \
  do {                                        \
    if (mr == 8) {                            \
      LAUNCH(NWv, NWNv, NTv, 8, KIv, MINBv);  \
    } else if (mr == 16) {                    \
      LAUNCH(NWv, NWNv, NTv, 16, KIv, MINBv); \
    } else {                                  \
      LAUNCH(NWv, NWNv, NTv, 32, KIv, MINBv); \
    }                                         \
  } while (0)

void RunImpl(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
             ffi::TensorView Y) {
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
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, A.device().device_id));

  if (M <= 2 && N == 1024 && K == 2048) {
    const dim3 grid(128, 1, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 4;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    cudaLaunchKernelEx(&config, gk<16, 1, 1, 8, 1, 1, false, 8, false, true>, Ap, Bp, Asp, Bsp, Yp,
                       M, N, K, KB);
    return;
  }

  const long long bbytes = (long long)N * (long long)K;
  // A small weight stream stays resident in L2, so splitting m across CTAs to
  // buy parallelism costs a cheap re-read rather than DRAM traffic.
  const bool tiny = bbytes <= (3LL << 19);  // <= 1.5 MB of weights
  const int mr = (M <= 8) ? 8 : ((M <= 16) ? (tiny ? 8 : 16) : (tiny ? 16 : 32));

  if (M == 1 && N == 64 && K == 2048) {
    const dim3 grid(8, 1, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 4;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    cudaLaunchKernelEx(&config, gk<16, 1, 1, 8, 1, 1, false, 8, true>, Ap, Bp, Asp, Bsp, Yp, M, N,
                       K, KB);
    return;
  }

  if (M == 16 && N == 64 && K == 2048) {
    const dim3 grid(8, 8, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    cudaLaunchKernelEx(&config, gk_m16_n64<2, true, true>, Ap, Bp, Asp, Bsp, Yp);
    return;
  }

  if (M == 32 && N == 64 && K == 2048) {
    const dim3 grid(8, 8, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(256, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 4;
    attrs[0].val.clusterDim.y = 2;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    cudaLaunchKernelEx(&config, gk_m16_n64<4, true>, Ap, Bp, Asp, Bsp, Yp);
    return;
  }

  if (M == 8 && N == 64 && K == 2048) {
    const dim3 grid(M / 4, 8, 1);
    gk<8, 1, 1, 8, 2, 2, true, 4><<<grid, 256, 0, stream>>>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);
    return;
  }

  if (N <= 128) {
    // 8 n-tiles only: 16 warps keep the single k round-trip short at tiny M,
    // but past 4 live rows the 16-way shared-memory reduction costs more than
    // the second k group it saves.
    if (KB == 16 && M <= 4) LAUNCH(16, 1, 1, 8, 1, 1);
    if (KB == 16) LAUNCH(8, 1, 1, 8, 2, 2);
    LAUNCH(8, 1, 1, 8, 0, 2);
  }

  // Activation re-reads cost (N / CTA-n-width) * M * K bytes of L2 traffic, so
  // the CTA n-width grows with the live row count.  NWN n-groups of warps let a
  // CTA cover 2x the columns while the duplicate A request merges in L1.
  if (bbytes >= (12LL << 20)) {  // 9216 / 12288 columns
    if (mr == 32) {
      if (KB == 16) BYMR(8, 1, 4, 2, 2);
      BYMR(8, 1, 4, 0, 2);
    }
    if (M >= 16 && N < 12288) {
      if (KB == 16) BYMR(8, 2, 2, 4, 2);
      BYMR(8, 2, 2, 0, 2);
    }
    if (N >= 12288 && M >= 16) {
      if (KB == 16) BYMR(16, 2, 2, 2, 1);
      BYMR(16, 2, 2, 0, 1);
    }
    if (N >= 12288) {
      if (KB == 16) BYMR(8, 1, 2, 2, 2);
      BYMR(8, 1, 2, 0, 2);
    }
    if (KB == 16) BYMR(8, 1, 1, 2, 2);
    BYMR(8, 1, 1, 0, 2);
  }
  if (bbytes >= (4LL << 20)) {  // 2048 x 4096
    if (M > 16) {
      // a 32-row tile makes the activation stream as large as the weight
      // stream; two 16-row CTAs halve it and double the CTA count
      if (KB == 32) LAUNCH(16, 2, 2, 16, 4, 1);
      LAUNCH(16, 2, 2, 16, 0, 1);
    }
    if (M > 8) {
      if (KB == 32) LAUNCH(16, 4, 1, 8, 8, 1);
      LAUNCH(16, 4, 1, 8, 0, 1);
    }
    if (M >= 8) {
      if (KB == 32) BYMR(16, 2, 2, 4, 1);
      BYMR(16, 2, 2, 0, 1);
    }
    if (KB == 32) BYMR(16, 1, 2, 2, 1);
    BYMR(16, 1, 2, 0, 1);
  }
  if (KB <= 4) {  // 2048 x 512
    if (KB == 4) BYMRN(8, 2, 2, 1, 2);
    BYMRN(8, 2, 2, 0, 2);
  }
  if (M > 16) {  // 1024 x 2048
    if (KB == 16) LAUNCH(16, 2, 1, 16, 2, 1);
    LAUNCH(16, 2, 1, 16, 0, 1);
  }
  if (M > 8) {
    if (KB == 16) LAUNCH(16, 2, 1, 8, 2, 1);
    LAUNCH(16, 2, 1, 8, 0, 1);
  }
  // At 1-4 live rows the activation stream is negligible, so trade CTA n-width
  // for twice the CTA count and keep every SM busy.
  if (M <= 4) {
    if (KB == 16) BYMR(16, 1, 1, 1, 1);
    BYMR(16, 1, 1, 0, 1);
  }
  if (KB == 16) BYMR(16, 1, 2, 1, 1);
  BYMR(16, 1, 2, 0, 1);
}

}  // namespace d26_impl

void LaunchExactN64Rows1(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                         __nv_bfloat16* Y, int M, cudaStream_t stream) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((unsigned)M, 8, 1);
  config.blockDim = dim3(256, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = (M == 2) ? 2u : 1u;
  attrs[0].val.clusterDim.y = (M == 2) ? 2u : 1u;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
  config.attrs = attrs;
  config.numAttrs = 2;
  cudaLaunchKernelEx(&config, d26_impl::gk_m16_n64<1, true>, A, B, As, Bs, Y);
}

void LaunchExactN64Rows2(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                         __nv_bfloat16* Y, int M, cudaStream_t stream) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3((unsigned)(M / 2), 8, 1);
  config.blockDim = dim3(256, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[2]{};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 1;
  attrs[0].val.clusterDim.y = 2;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
  attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
  config.attrs = attrs;
  config.numAttrs = 2;
  cudaLaunchKernelEx(&config, d26_impl::gk_m16_n64<2, true, true>, A, B, As, Bs, Y);
}

void RunD26(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
            ffi::TensorView Y) {
  d26_impl::RunImpl(A, B, As, Bs, Y);
}
