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
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

namespace peer_impl {

struct F8 {
  uint32_t v[8];
};

__device__ __forceinline__ F8 ldB(const void* p) {
  F8 r;
  asm("ld.global.nc.L1::no_allocate.v8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]),
        "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ F8 ldA(const void* p) {
  F8 r;
  asm("ld.global.nc.L1::evict_last.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]), "=r"(r.v[4]),
        "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}
__device__ __forceinline__ float ldf(const float* p) {
  float r;
  asm("ld.global.nc.L1::evict_last.f32 %0, [%1];" : "=f"(r) : "l"(p));
  return r;
}

#define MMA(d, a0, a1, a2, a3, b0, b1)                                      \
  asm volatile(                                                             \
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "                \
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"             \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                      \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// NW warps per CTA, NWN groups along n, NWK = NW/NWN along k.  KI > 0 pins the
// per-warp k-group trip count so the loop control and pointer bumps fold away.
template <int NW, int NWN, int NT, int MR, int KI, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB) gk(
    const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bq,
    const float* __restrict__ As, const float* __restrict__ Bs,
    __nv_bfloat16* __restrict__ Y, int M, int N, int K, int KB) {
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

  const int nc = blockIdx.x * CN;
  const int n0 = nc + ng * (NT * 8);
  const int m0 = blockIdx.y * MR;

  // Out-of-range rows/cols read a live address and are dropped at store time:
  // an mma output element depends only on its own A row and B column.
  const uint8_t* bp[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int r = n0 + t * 8 + g;
    bp[t] = Bq + (size_t)(r < N ? r : 0) * (size_t)K + (size_t)(c * 32) +
            (size_t)wk * 128;
  }
  const uint8_t* ap[NR];
  const float* asp[NR];
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    const int r = m0 + (q / HH) * 16 + (q % HH) * 8 + g;
    const int rc = (r < M) ? r : 0;
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
              (HH == 2) ? F1.v[2 * j + 1] : z, bf[i].v[2 * j],
              bf[i].v[2 * j + 1]);
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
        if (r0 < M && col < N) {
          Y[(size_t)r0 * N + col] = __float2bfloat16_rn(acc[t][i][0]);
          if (col + 1 < N)
            Y[(size_t)r0 * N + col + 1] = __float2bfloat16_rn(acc[t][i][1]);
        }
        if (HH == 2 && r0 + 8 < M && col < N) {
          Y[(size_t)(r0 + 8) * N + col] = __float2bfloat16_rn(acc[t][i][2]);
          if (col + 1 < N)
            Y[(size_t)(r0 + 8) * N + col + 1] =
                __float2bfloat16_rn(acc[t][i][3]);
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
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NWK; ++w) s += sm[(w * MR + row) * CNP + col];
    const int gr = m0 + row, gc = nc + col;
    if (gr < M && gc < N) Y[(size_t)gr * N + gc] = __float2bfloat16_rn(s);
  }
}

#define LAUNCH(NWv, NWNv, NTv, MRv, KIv, MINBv)                              \
  do {                                                                       \
    const dim3 grid((N + (NWNv * NTv * 8) - 1) / (NWNv * NTv * 8),           \
                    (unsigned)((M + MRv - 1) / MRv), 1);                     \
    gk<NWv, NWNv, NTv, MRv, KIv, MINBv><<<grid, NWv * 32, 0, stream>>>(      \
        Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);                                  \
    return;                                                                  \
  } while (0)

#define BYMR(NWv, NWNv, NTv, KIv, MINBv)                                     \
  do {                                                                       \
    if (mr == 8) { LAUNCH(NWv, NWNv, NTv, 8, KIv, MINBv); }                  \
    else if (mr == 16) { LAUNCH(NWv, NWNv, NTv, 16, KIv, MINBv); }           \
    else { LAUNCH(NWv, NWNv, NTv, 32, KIv, MINBv); }                         \
  } while (0)

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

  const long long bbytes = (long long)N * (long long)K;
  // A small weight stream stays resident in L2, so splitting m across CTAs to
  // buy parallelism costs a cheap re-read rather than DRAM traffic.
  const bool tiny = bbytes <= (3LL << 19);   // <= 1.5 MB of weights
  const int mr = (M <= 8) ? 8
                          : ((M <= 16) ? (tiny ? 8 : 16) : (tiny ? 16 : 32));

  if (M == 32 && N == 2048 && K == 4096) {
    const dim3 grid(64, 2, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference =
        cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    cudaLaunchKernelEx(&config, gk<16, 2, 2, 16, 4, 1>, Ap, Bp, Asp, Bsp,
                       Yp, M, N, K, KB);
    return;
  }

  if (N <= 128 && KB == 16) LAUNCH(16, 1, 1, 8, 1, 1);
  if (N <= 128) BYMR(8, 1, 1, 0, 2);

  // Activation re-reads cost (N / CTA-n-width) * M * K bytes of L2 traffic, so
  // the CTA n-width grows with the live row count.  NWN n-groups of warps let a
  // CTA cover 2x the columns while the duplicate A request merges in L1.
  if (bbytes >= (12LL << 20)) {            // 9216 / 12288 columns
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
  if (bbytes >= (4LL << 20)) {             // 2048 x 4096
    if (M >= 8) {
      if (KB == 32) BYMR(16, 2, 2, 4, 1);
      BYMR(16, 2, 2, 0, 1);
    }
    if (KB == 32) BYMR(16, 1, 2, 2, 1);
    BYMR(16, 1, 2, 0, 1);
  }
  if (KB <= 4) {                           // 2048 x 512
    if (KB == 4) BYMR(8, 2, 2, 1, 2);
    BYMR(8, 2, 2, 0, 2);
  }
  if (M == 32 && N == 1024 && KB == 16)
    LAUNCH(16, 2, 1, 16, 2, 1);
  if (M > 16) {                            // 1024 x 2048
    if (KB == 16) BYMR(16, 2, 1, 2, 1);
    BYMR(16, 2, 1, 0, 1);
  }
  if (M > 8) {
    if (KB == 16) LAUNCH(16, 1, 2, 8, 1, 1);
    LAUNCH(16, 1, 2, 8, 0, 1);
  }
  if (KB == 16) BYMR(16, 1, 2, 1, 1);
  BYMR(16, 1, 2, 0, 1);
}

}  // namespace peer_impl

void RunPeer(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As,
             ffi::TensorView Bs, ffi::TensorView Y) {
  peer_impl::Run(A, B, As, Bs, Y);
}



