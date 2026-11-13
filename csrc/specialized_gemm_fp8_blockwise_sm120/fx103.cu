// Exact candidate fx config 103 for M=16, N=2048, K=4096.
// FP32 cross-warp reduction; B fragments are issued before A fragments.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

namespace fx103 {

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

template <int NWN, int NWK, int NT, int MR, int KU, int ITERS, int MINB, bool SBC>
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
  constexpr int RSZ2 = (NWK > 1) ? (W * (NACC / 2) * 32) : 1;
  static_assert(NWN == 2 && NWK == 16 && NT == 2 && MR == 16 && KU == 1 &&
                    ITERS == 2,
                "fx103 is fixed to M16 N2048 K4096");
  constexpr int FM = 16;
  constexpr int FN = 2048;
  constexpr int FK = 4096;
  constexpr int FKB = 32;
  __shared__ float2 red[RSZ2];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int ng = (NWN == 1) ? 0 : (warp % NWN);
  const int wk = (NWN == 1) ? warp : (warp / NWN);
  const int g = lane >> 2;
  const int c = lane & 3;
  const int nc = blockIdx.x * CN;
  const int n0 = nc + ng * (NT * 8);
  constexpr int mb = 0;
  const int k0 = wk * KU * 128 + c * 32;
  const int kstep = NWK * KU * 128;

  // B addresses first so the DRAM round trip starts before anything touches A.
  const int nq0 = n0 + g;
  const uint8_t* bp[NT];
  bp[0] = Bq + (size_t)nq0 * (size_t)FK + k0;
#pragma unroll
  for (int i = 1; i < NT; ++i) {
    const int nqi = n0 + i * 8 + g;
    bp[i] = bp[0] + (size_t)((nqi - nq0) * FK);
  }

  const int rc0 = g;
  const uint8_t* ap[NR];
  const float* asp[NR];
  ap[0] = A + (size_t)rc0 * (size_t)FK + k0;
  asp[0] = As + (size_t)rc0 * (size_t)FKB + (wk * KU);
#pragma unroll
  for (int q = 1; q < NR; ++q) {
    const int rc = (q / HH) * 16 + (q % HH) * 8 + g;
    ap[q] = ap[0] + (size_t)((rc - rc0) * FK);
    asp[q] = asp[0] + (size_t)((rc - rc0) * FKB);
  }
  const int nblk = n0 >> 7;
  const float* bsp = Bs + (size_t)nblk * (size_t)FKB + (wk * KU);

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
      for (int q = 0; q < NR; ++q) {
        if constexpr (SBC) {
          // One lane per 4-lane group fetches As; the other three take it by
          // shuffle, which cuts the redundant scalar requests by 4x.
          float av = (c == 0) ? lds(asp[q] + u) : 0.f;
          av = __shfl_sync(0xffffffffu, av, g * 4);
          sc[u][q] = av * bv;
        } else {
          sc[u][q] = lds(asp[q] + u) * bv;
        }
      }
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

  float2* sw = red + warp * ((NACC / 2) * 32);
#pragma unroll
  for (int p = 0; p < NACC / 2; ++p)
    sw[p * 32 + lane] = make_float2(acc[2 * p], acc[2 * p + 1]);
  __syncthreads();

  // Accumulator slots come in adjacent-column pairs (sl, sl+1), so reduce and
  // store them together: 4-byte stores instead of two 2-byte partial ones.
  constexpr int PAIRS = NWN * (NACC / 2) * 32;
  constexpr int ROUNDS = (PAIRS + THREADS - 1) / THREADS;
#pragma unroll
  for (int rd = 0; rd < ROUNDS; ++rd) {
    const int idx = tid + rd * THREADS;
    if (idx < PAIRS) {
      const int ngi = (NWN == 1) ? 0 : (idx / ((NACC / 2) * 32));
      const int rest = (NWN == 1) ? idx : (idx - ngi * ((NACC / 2) * 32));
      const int a = (rest >> 5) * 2;
      const int l = rest & 31;
      float s0 = 0.f, s1 = 0.f;
#pragma unroll
      for (int w = 0; w < NWK; ++w) {
        const float2 v =
            red[((w * NWN + ngi) * (NACC / 2) + (a >> 1)) * 32 + l];
        s0 += v.x;
        s1 += v.y;
      }
      const int t = a / (NT * ACT);
      const int rem = a - t * (NT * ACT);
      const int i = rem / ACT;
      const int sl = rem - i * ACT;
      const int row = t * 16 + ((sl >= 2) ? 8 : 0) + (l >> 2);
      const int col = nc + ngi * (NT * 8) + i * 8 + (l & 3) * 2;
      st2(Y + (size_t)row * (size_t)FN + col, s0, s1, col, FN);
    }
  }
}

}  // namespace fx103

void RunFX103(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As,
              ffi::TensorView Bs, ffi::TensorView Y) {
  const int M = static_cast<int>(A.shape()[0]);
  const int K = static_cast<int>(A.shape()[1]);
  const int N = static_cast<int>(B.shape()[0]);
  const int KB = K >> 7;
  const uint8_t* Ap = static_cast<const uint8_t*>(A.data_ptr());
  const uint8_t* Bp = static_cast<const uint8_t*>(B.data_ptr());
  const float* Asp = static_cast<const float*>(As.data_ptr());
  const float* Bsp = static_cast<const float*>(Bs.data_ptr());
  __nv_bfloat16* Yp = static_cast<__nv_bfloat16*>(Y.data_ptr());
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(kDLCUDA, A.device().device_id));

  const dim3 grid(64, 1, 1);
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = dim3(1024, 1, 1);
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
  cudaLaunchKernelEx(&config, fx103::fxk<2, 16, 2, 16, 1, 2, 1, true>, Ap,
                     Bp, Asp, Bsp, Yp, M, N, K, KB);
}


