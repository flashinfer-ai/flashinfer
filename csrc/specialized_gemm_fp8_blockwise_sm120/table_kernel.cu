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
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

namespace ffi = tvm::ffi;

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

// ---------------------------------------------------------------- dispatch
// Dynamic-smem byte count of gemm_bw's cross-warp reduction scratch (zero for
// the NW_K == 1 forms, which never touch it).
template <int NW_N, int NW_K, int NT, int MR>
constexpr int red_smem_bytes() {
  const int MT = (MR + 15) / 16;
  const int HH = (MR <= 8) ? 1 : 2;
  const int NACC = MT * NT * (HH * 2);
  return (NW_K > 1) ? NW_N * NW_K * NACC * 32 * static_cast<int>(sizeof(float)) : 0;
}

template <int NW_N, int NW_K, int NT, int MR, int KU, int ITERS, int MINB>
static inline void launch_bw(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                             __nv_bfloat16* Y, int M, int N, int K, int KB, const dim3& grid,
                             cudaStream_t s) {
  constexpr int TH = NW_N * NW_K * 32;
  constexpr int SMEM = red_smem_bytes<NW_N, NW_K, NT, MR>();
  if constexpr (SMEM > 48 * 1024) {
    cudaFuncSetAttribute(gemm_bw<NW_N, NW_K, NT, MR, KU, ITERS, MINB>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
  }
  gemm_bw<NW_N, NW_K, NT, MR, KU, ITERS, MINB><<<grid, TH, SMEM, s>>>(A, B, As, Bs, Y, M, N, K, KB);
}

template <int NW_N, int NW_K, int NT, int KU, int ITERS, int MINB>
static inline void go2(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                       __nv_bfloat16* Y, int M, int N, int K, int KB, int MR, cudaStream_t s) {
  constexpr int CN = NW_N * NT * 8;
  const dim3 grid((N + CN - 1) / CN, (unsigned)((M + MR - 1) / MR), 1);
  if (MR == 8)
    launch_bw<NW_N, NW_K, NT, 8, KU, ITERS, MINB>(A, B, As, Bs, Y, M, N, K, KB, grid, s);
  else if (MR == 16)
    launch_bw<NW_N, NW_K, NT, 16, KU, ITERS, MINB>(A, B, As, Bs, Y, M, N, K, KB, grid, s);
  else
    launch_bw<NW_N, NW_K, NT, 32, KU, ITERS, MINB>(A, B, As, Bs, Y, M, N, K, KB, grid, s);
}

template <int NW_N, int NW_K, int NT, int KU, int MINB>
static inline void go(const uint8_t* A, const uint8_t* B, const float* As, const float* Bs,
                      __nv_bfloat16* Y, int M, int N, int K, int KB, int MR, cudaStream_t s) {
  const int iters = KB / (NW_K * KU);
  if (iters == 1)
    go2<NW_N, NW_K, NT, KU, 1, MINB>(A, B, As, Bs, Y, M, N, K, KB, MR, s);
  else if (iters == 2)
    go2<NW_N, NW_K, NT, KU, 2, MINB>(A, B, As, Bs, Y, M, N, K, KB, MR, s);
  else
    go2<NW_N, NW_K, NT, KU, 0, MINB>(A, B, As, Bs, Y, M, N, K, KB, MR, s);
}

void Run(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
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

  if (M <= 8 && N == 2048 && K == 512) {
    const dim3 grid(64, 1, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = red_smem_bytes<4, 4, 1, 8>();
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
    cudaLaunchKernelEx(&config, gemm_bw<4, 4, 1, 8, 1, 1, 1>, Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);
    return;
  }

  if ((M == 32 || (M >= 17 && M <= 31)) && N == 2048 && K == 512) {
    // M=32 uses a 2-CTA cluster; M 17..31 the same tiling without one (the
    // measured route; combo_entry sends M 9..16 of this class to c00).
    if (M != 32) {
      const dim3 grid(64, (unsigned)((M + 15) / 16), 1);
      launch_bw<4, 4, 1, 16, 1, 1, 1>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, grid, stream);
      return;
    }
    const dim3 grid(64, 2, 1);
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = dim3(512, 1, 1);
    config.dynamicSmemBytes = red_smem_bytes<4, 4, 1, 16>();
    config.stream = stream;
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
    config.attrs = attrs;
    config.numAttrs = 2;
    cudaLaunchKernelEx(&config, gemm_bw<4, 4, 1, 16, 1, 1, 1>, Ap, Bp, Asp, Bsp, Yp, M, N, K, KB);
    return;
  }

  // combo_entry sends (N=9216, M 5..8) here; measured best is eight k-warps,
  // two n8 subtiles, two fixed k steps, two blocks per SM.
  if (N == 9216 && K == 2048 && M >= 5 && M <= 8) {
    go<1, 8, 2, 1, 2>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, 8, stream);
    return;
  }

  // Remaining reachable cell (N=9216, M=1) measured best on the generic
  // single-warp-K route; it also serves as the dispatcher's safety net
  // (correct for any shape).
  const int mr = (M <= 8) ? 8 : ((M <= 16) ? 16 : 32);
  go<1, 1, 1, 1, 1>(Ap, Bp, Asp, Bsp, Yp, M, N, K, KB, mr, stream);
}

}  // namespace table_impl

void RunTable(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
              ffi::TensorView Y) {
  table_impl::Run(A, B, As, Bs, Y);
}
