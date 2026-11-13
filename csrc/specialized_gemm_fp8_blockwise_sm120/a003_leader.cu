// SM120 fp8 e4m3 blockwise-scaled NT GEMM.  The k coordinates inside each
// 128-wide scale group are permuted identically for A and B so a single
// 256-bit global load lands directly in each mma.sync fragment.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

void LaunchAsyncM32N12288(const uint8_t* a, const uint8_t* b,
                          const float* as, const float* bs,
                          __nv_bfloat16* y, int M, int N, int K,
                          cudaStream_t stream);
void RunSpecial(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As,
                ffi::TensorView Bs, ffi::TensorView Y);

namespace a003_leader_impl {

struct F8 {
  uint32_t v[8];
};

__device__ __forceinline__ F8 ldB(const void* p) {
  F8 r;
  asm("ld.global.nc.L1::no_allocate.L2::256B.v8.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]),
        "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(p));
  return r;
}

__device__ __forceinline__ F8 ldA(const void* p) {
  F8 r;
  asm("ld.global.nc.L1::evict_last.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]),
        "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
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
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                    \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

// NWN warp groups own independent n ranges; NWK=NW/NWN warps cooperate over
// k.  KI pins the groups per warp when profitable.  The reduction is laid out
// by output element so every CTA thread participates in the epilogue.
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

  const uint8_t* bp[NT];
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int r = n0 + t * 8 + g;
    bp[t] = Bq + (size_t)(r < N ? r : 0) * (size_t)K + (size_t)c * 32 +
            (size_t)wk * 128;
  }
  const uint8_t* ap[NR];
  const float* asp[NR];
#pragma unroll
  for (int q = 0; q < NR; ++q) {
    const int r = m0 + (q / HH) * 16 + (q % HH) * 8 + g;
    const int rc = (r < M) ? r : 0;
    ap[q] = A + (size_t)rc * (size_t)K + (size_t)c * 32 + (size_t)wk * 128;
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
          MMA(d, F0.v[2 * j], (HH == 2) ? F1.v[2 * j] : z,
              F0.v[2 * j + 1], (HH == 2) ? F1.v[2 * j + 1] : z,
              bf[i].v[2 * j], bf[i].v[2 * j + 1]);
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
    const int gr = m0 + row;
    const int gc = nc + col;
    if (gr < M && gc < N) Y[(size_t)gr * N + gc] = __float2bfloat16_rn(s);
  }
}

#define LAUNCH(NWv, NWNv, NTv, MRv, KIv, MINBv)                              \
  do {                                                                       \
    const dim3 grid((N + NWNv * NTv * 8 - 1) / (NWNv * NTv * 8),             \
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

  if (M == 32 && N == 12288 && K == 2048) {
    LaunchAsyncM32N12288(Ap, Bp, Asp, Bsp, Yp, M, N, K, stream);
    return;
  }

  if ((N == 12288 && K == 2048 && (M == 4 || M == 8 || M == 16)) ||
      (M == 16 && N == 9216 && K == 2048) ||
      (M == 4 && N == 2048 && K == 4096)) {
    RunSpecial(A, B, As, Bs, Y);
    return;
  }

  const long long bbytes = (long long)N * (long long)K;
  const bool tiny = bbytes <= (3LL << 19);
  const int mr = (M <= 8) ? 8
                          : ((M <= 16) ? (tiny ? 8 : 16) : (tiny ? 16 : 32));

  if (N <= 128 && KB == 16) LAUNCH(16, 1, 1, 8, 1, 1);
  if (N <= 128) BYMR(8, 1, 1, 0, 2);

  if (bbytes >= (12LL << 20)) {
    if (mr == 32) {
      if (KB == 16) BYMR(8, 1, 4, 2, 2);
      BYMR(8, 1, 4, 0, 2);
    }
    if (M >= 16 && N < 12288) {
      if (KB == 16) BYMR(8, 2, 2, 4, 2);
      BYMR(8, 2, 2, 0, 2);
    }
    if (N >= 12288) {
      if (KB == 16) BYMR(8, 1, 2, 2, 2);
      BYMR(8, 1, 2, 0, 2);
    }
    if (KB == 16) BYMR(8, 1, 1, 2, 2);
    BYMR(8, 1, 1, 0, 2);
  }
  if (bbytes >= (4LL << 20)) {
    if (M == 16 && N == 2048 && KB == 32)
      LAUNCH(16, 2, 2, 8, 4, 1);
    if (M >= 8) {
      if (KB == 32) BYMR(16, 2, 2, 4, 1);
      BYMR(16, 2, 2, 0, 1);
    }
    if (KB == 32) BYMR(16, 1, 2, 2, 1);
    BYMR(16, 1, 2, 0, 1);
  }
  if (KB <= 4) {
    if (KB == 4) BYMR(8, 2, 2, 1, 2);
    BYMR(8, 2, 2, 0, 2);
  }
  if (M > 16) {
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

}  // namespace a003_leader_impl

void RunA003(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As,
             ffi::TensorView Bs, ffi::TensorView Y) {
  a003_leader_impl::Run(A, B, As, Bs, Y);
}



