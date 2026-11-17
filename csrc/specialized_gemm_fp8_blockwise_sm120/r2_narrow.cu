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

namespace r2_narrow {

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
  asm("ld.global.nc.L1::no_allocate.L2::256B.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
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

// W: warps per block, splitting the k-group loop; they reduce through smem.
// MR: compile-time row bound (8/16/32).  MSPLIT: m rows are chunked over
// grid.y -- it costs one extra weight pass per chunk, but on shapes with too
// few n-tiles to fill the SMs and a small weight stream it is the only way to
// buy blocks (N=64 has just 8 n-tiles for 188 SMs).  KITER fixes the number of
// scale groups owned by each warp so the loop control and pointer bumps fold.
template <int W, int MR, bool MSPLIT, int KITER, int NT = 1, bool PREFETCH2 = false,
          bool PAIRRED = false>
__global__ void __launch_bounds__(W * 32)
    gemm_fp8_bw_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ Bq,
                       const float* __restrict__ As, const float* __restrict__ Bs,
                       __nv_bfloat16* __restrict__ Y, int M, int N, int K, int KB) {
  constexpr int MT = (MR + 15) / 16;     // 16-row m-tiles
  constexpr int HH = (MR <= 8) ? 1 : 2;  // 8-row halves actually live
  constexpr int NR = MT * HH;            // A row slots per thread
  constexpr int ACT = HH * 2;            // accumulators per m-tile
  constexpr int NACC = MT * NT * ACT;
  constexpr int RSZ = (W > 1 && !PAIRRED) ? (W * NACC * 32) : 1;
  constexpr int RSZ2 = (W > 1 && PAIRRED) ? (W * (NACC / 2) * 32) : 1;
  // MR=32 keeps 32 fragment registers live; a second in-flight k group spills.
  constexpr int UNR = 1;
  constexpr bool BFIRST = W == 16 && MR == 16 && !MSPLIT && KITER == 0;
  __shared__ float red[RSZ];
  __shared__ float2 red2[RSZ2];

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
    const int rc = (r < M) ? r : 0;
    ap[q] = A + (size_t)rc * (size_t)K + (size_t)(warp * 128 + c * 32);
    asp[q] = As + (size_t)rc * (size_t)KB + warp;
  }
  const float* bsp = Bs + (size_t)(n0 >> 7) * (size_t)KB + warp;

  float acc[NACC];
#pragma unroll
  for (int a = 0; a < NACC; ++a) acc[a] = 0.f;

  auto consume = [&](const Frag8* bf, const Frag8* af, const float* sc) {
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

  auto kbody = [&]() {
    const float bsv = ld_keepf(bsp);
    bsp += W;
    Frag8 bf[NT];
    if constexpr (BFIRST) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        bf[i] = ld_stream(bp[i]);
        bp[i] += kstep;
      }
    }
    Frag8 af[NR];
    float sc[NR];
#pragma unroll
    for (int q = 0; q < NR; ++q) {
      af[q] = ld_keep(ap[q]);
      sc[q] = ld_keepf(asp[q]) * bsv;
      ap[q] += kstep;
      asp[q] += W;
    }
    if constexpr (!BFIRST) {
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        bf[i] = ld_stream(bp[i]);
        bp[i] += kstep;
      }
    }
    consume(bf, af, sc);
  };

  if constexpr (PREFETCH2) {
    static_assert(KITER == 2 && !BFIRST);
    Frag8 bf[2][NT];
    Frag8 af[2][NR];
    float sc[2][NR];
#pragma unroll
    for (int it = 0; it < 2; ++it) {
      const float bsv = ld_keepf(bsp);
      bsp += W;
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        bf[it][i] = ld_stream(bp[i]);
        bp[i] += kstep;
      }
#pragma unroll
      for (int q = 0; q < NR; ++q) {
        af[it][q] = ld_keep(ap[q]);
        sc[it][q] = ld_keepf(asp[q]) * bsv;
        ap[q] += kstep;
        asp[q] += W;
      }
    }
#pragma unroll
    for (int it = 0; it < 2; ++it) consume(bf[it], af[it], sc[it]);
  } else if (KITER > 0) {
#pragma unroll
    for (int it = 0; it < KITER; ++it) kbody();
  } else {
#pragma unroll UNR
    for (int kg = warp; kg < KB; kg += W) kbody();
  }

  // Cross-warp k reduction.  Every warp parks its accumulators, then all
  // threads cooperatively sum the W partials for a slice of the tile and store
  // it.  Walking the partials in warp 0 alone leaves a serial tail of
  // (W-1)*NACC shared-memory rounds, which is pure latency on the schedules
  // that give each warp a single k group.
  if (W > 1) {
    if constexpr (PAIRRED) {
      static_assert((NACC & 1) == 0);
      float2* sw = red2 + warp * (NACC / 2) * 32;
#pragma unroll
      for (int p = 0; p < NACC / 2; ++p)
        sw[p * 32 + lane] = make_float2(acc[2 * p], acc[2 * p + 1]);
      __syncthreads();
      constexpr int PAIRS = (NACC / 2) * 32;
      for (int idx = threadIdx.x; idx < PAIRS; idx += W * 32) {
        const int p = idx >> 5;
        const int l = idx & 31;
        float2 s = make_float2(0.f, 0.f);
#pragma unroll
        for (int w = 0; w < W; ++w) {
          const float2 v = red2[(w * (NACC / 2) + p) * 32 + l];
          s.x += v.x;
          s.y += v.y;
        }
        const int a = 2 * p;
        const int t = a / (NT * ACT);
        const int rem = a - t * (NT * ACT);
        const int i = rem / ACT;
        const int sl = rem - i * ACT;
        const int row = mb + t * 16 + ((sl >= 2) ? 8 : 0) + (l >> 2);
        const int col = n0 + i * 8 + (l & 3) * 2;
        if (row < M && col + 1 < N) store2(Y + (size_t)row * (size_t)N + col, s.x, s.y, col, N);
      }
      return;
    }
    float* sw = red + warp * NACC * 32;
#pragma unroll
    for (int a = 0; a < NACC; ++a) sw[a * 32 + lane] = acc[a];
    __syncthreads();
    constexpr int OUTS = NACC * 32;
    constexpr int THREADS = W * 32;
    for (int idx = threadIdx.x; idx < OUTS; idx += THREADS) {
      const int a = idx >> 5;
      const int l = idx & 31;
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < W; ++w) s += red[(w * NACC + a) * 32 + l];
      const int t = a / (NT * ACT);
      const int rem = a - t * (NT * ACT);
      const int i = rem / ACT;
      const int sl = rem - i * ACT;
      const int row = mb + t * 16 + ((sl >= 2) ? 8 : 0) + (l >> 2);
      const int col = n0 + i * 8 + (l & 3) * 2 + (sl & 1);
      if (row < M && col < N) Y[(size_t)row * (size_t)N + col] = __float2bfloat16_rn(s);
    }
    return;
  }

#pragma unroll
  for (int t = 0; t < MT; ++t) {
#pragma unroll
    for (int i = 0; i < NT; ++i) {
      const int b = (t * NT + i) * ACT;
      const int col = n0 + i * 8 + c * 2;
      const int r0 = mb + t * 16 + g;
      if (r0 < M) store2(Y + (size_t)r0 * (size_t)N + col, acc[b + 0], acc[b + 1], col, N);
      if (HH == 2 && r0 + 8 < M)
        store2(Y + (size_t)(r0 + 8) * (size_t)N + col, acc[b + 2], acc[b + 3], col, N);
    }
  }
}

}  // namespace r2_narrow

void RunR2Narrow(ffi::TensorView A, ffi::TensorView B, ffi::TensorView As, ffi::TensorView Bs,
                 ffi::TensorView Y) {
  const int M = static_cast<int>(A.shape()[0]);
  const int K = static_cast<int>(A.shape()[1]);
  const int N = static_cast<int>(B.shape()[0]);
  const int KB = K >> 7;
  const uint8_t* Ap = static_cast<const uint8_t*>(A.data_ptr());
  const uint8_t* Bp = static_cast<const uint8_t*>(B.data_ptr());
  const float* Asp = static_cast<const float*>(As.data_ptr());
  const float* Bsp = static_cast<const float*>(Bs.data_ptr());
  __nv_bfloat16* Yp = static_cast<__nv_bfloat16*>(Y.data_ptr());
  cudaStream_t stream =
      static_cast<cudaStream_t>(TVMFFIEnvGetStream(kDLCUDA, A.device().device_id));
  const dim3 grid((N + 15) / 16, 1, 1);
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
  attrs[1].val.clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyLoadBalancing;
  config.attrs = attrs;
  config.numAttrs = 2;
  // The paired path wins at two and eight live rows; four rows remains on the
  // lean path after an isolated A/B was flat.
  if (M == 2 || M == 4 || M == 8) {
    cudaLaunchKernelEx(&config, r2_narrow::gemm_fp8_bw_kernel<16, 8, false, 2, 2, true, true>, Ap,
                       Bp, Asp, Bsp, Yp, M, N, K, KB);
  } else {
    cudaLaunchKernelEx(&config, r2_narrow::gemm_fp8_bw_kernel<16, 8, false, 2, 2>, Ap, Bp, Asp, Bsp,
                       Yp, M, N, K, KB);
  }
}
