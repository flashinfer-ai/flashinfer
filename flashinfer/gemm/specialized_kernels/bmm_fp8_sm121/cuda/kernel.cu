// FP8 GEMM kernel for batched per-tensor scaled FP8 matmul on SM121.
// A: (batch, M, K) e4m3 row-major
// B: (batch, K, N) e4m3 stored as transpose of (batch, N, K) row-major
//    => B is K-major (K contiguous, stride K between N columns)
//    => effectively (N, K) row-major in memory
// Out: (batch, M, N) bf16 row-major
// scale = A_scale * B_scale (FP32, applied at epilogue)
//
// Split-K writes per-slice fp32 partial sums into a workspace, then a
// separate reduce kernel sums and casts to bf16.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t cvta_to_shared(const void* ptr) {
  uint32_t addr;
  asm("{\n"
      "  .reg .u64 t;\n"
      "  cvta.to.shared.u64 t, %1;\n"
      "  cvt.u32.u64 %0, t;\n"
      "}"
      : "=r"(addr)
      : "l"(ptr));
  return addr;
}

__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4_a(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                              uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k32_e4m3(float& d0, float& d1, float& d2, float& d3,
                                                  uint32_t a0, uint32_t a1, uint32_t a2,
                                                  uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, "
      "{%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Core GEMM kernel - templated on whether output is bf16 (regular) or fp32 (split-K).
// For split-K: blockIdx.z = split_id, K range is [k_start, k_end)
//   output goes to workspace[split_id * M*N + row*N + col] as raw fp32 partial sum (no scale).
template <int BM, int BN, int BK, int WARP_M, int WARP_N, int STAGES, bool SPLIT_K,
          int MIN_BLOCKS = 1>
__global__ __launch_bounds__(WARP_M* WARP_N * 32, MIN_BLOCKS) void fp8_gemm_kernel(
    const __nv_fp8_e4m3* __restrict__ A, const __nv_fp8_e4m3* __restrict__ B,
    void* __restrict__ Out,  // bf16* if !SPLIT_K else fp32*
    const float* __restrict__ A_scale_ptr, const float* __restrict__ B_scale_ptr, int M, int N,
    int K,
    int K_per_split  // ignored if !SPLIT_K
) {
  constexpr int NWARPS = WARP_M * WARP_N;
  constexpr int WBM = BM / WARP_M;
  constexpr int WBN = BN / WARP_N;
  constexpr int M_FRAGS = WBM / 16;
  constexpr int N_FRAGS = WBN / 8;
  constexpr int PAD = 16;
  constexpr int LDA = BK + PAD;
  constexpr int LDB = BK + PAD;
  constexpr int NTHREADS = NWARPS * 32;

  const int tid = threadIdx.x;
  const int wid = tid / 32;
  const int lid = tid % 32;
  const int gid = lid >> 2;
  const int tig = lid & 3;
  const int warp_m = wid / WARP_N;
  const int warp_n = wid % WARP_N;

  const int block_m = blockIdx.y * BM;
  const int block_n = blockIdx.x * BN;

  int k_start = 0;
  int k_end = K;
  int split_id = 0;
  if (SPLIT_K) {
    split_id = blockIdx.z;
    k_start = split_id * K_per_split;
    if (k_start >= K) return;
    k_end = k_start + K_per_split;
    if (k_end > K) k_end = K;
  }
  const int num_k_local = (k_end - k_start) / BK + ((k_end - k_start) % BK != 0 ? 1 : 0);

  extern __shared__ __align__(16) uint8_t smem_buf[];
  auto sA = reinterpret_cast<__nv_fp8_e4m3*>(smem_buf);
  auto sB = sA + STAGES * BM * LDA;

  auto A_at = [&](int stage, int m, int k) -> __nv_fp8_e4m3& {
    return sA[stage * BM * LDA + m * LDA + k];
  };
  auto B_at = [&](int stage, int n, int k) -> __nv_fp8_e4m3& {
    return sB[stage * BN * LDB + n * LDB + k];
  };

  // Scale (only used for non-split-K direct write; split-K reduce kernel applies scale)
  const float a_scale = SPLIT_K ? 1.f : *A_scale_ptr;
  const float b_scale = SPLIT_K ? 1.f : *B_scale_ptr;
  const float scale = a_scale * b_scale;

  float acc[M_FRAGS][N_FRAGS][4];
#pragma unroll
  for (int m = 0; m < M_FRAGS; m++) {
#pragma unroll
    for (int n = 0; n < N_FRAGS; n++) {
#pragma unroll
      for (int i = 0; i < 4; i++) acc[m][n][i] = 0.f;
    }
  }

  constexpr int A_LDS = (BM * BK) / 16;
  constexpr int B_LDS = (BN * BK) / 16;
  constexpr int CHUNKS_PER_ROW = BK / 16;

  auto load_A = [&](int stage, int k0) {
#pragma unroll
    for (int i = tid; i < A_LDS; i += NTHREADS) {
      int row = i / CHUNKS_PER_ROW;
      int col_chunk = i % CHUNKS_PER_ROW;
      int gm = block_m + row;
      int gk = k0 + col_chunk * 16;
      uint32_t smem_addr = cvta_to_shared(&A_at(stage, row, col_chunk * 16));
      if (gm < M && gk < K) {
        cp_async_16(smem_addr, &A[gm * K + gk]);
      } else {
        asm volatile("st.shared.v4.b32 [%0], {0, 0, 0, 0};\n" ::"r"(smem_addr));
      }
    }
  };

  auto load_B = [&](int stage, int k0) {
#pragma unroll
    for (int i = tid; i < B_LDS; i += NTHREADS) {
      int n_row = i / CHUNKS_PER_ROW;
      int col_chunk = i % CHUNKS_PER_ROW;
      int gn = block_n + n_row;
      int gk = k0 + col_chunk * 16;
      uint32_t smem_addr = cvta_to_shared(&B_at(stage, n_row, col_chunk * 16));
      if (gn < N && gk < K) {
        cp_async_16(smem_addr, &B[gn * K + gk]);
      } else {
        asm volatile("st.shared.v4.b32 [%0], {0, 0, 0, 0};\n" ::"r"(smem_addr));
      }
    }
  };

  // Pipeline prefill
  int read_stage = 0;
  int write_stage = 0;
  int k0 = k_start;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (k0 < k_end) {
      load_A(write_stage, k0);
      load_B(write_stage, k0);
      cp_async_commit();
      k0 += BK;
      write_stage = (write_stage + 1) % STAGES;
    } else {
      cp_async_commit();
    }
  }

  // Main K loop
  for (int k_iter = 0; k_iter < num_k_local; k_iter++) {
    if (k0 < k_end) {
      load_A(write_stage, k0);
      load_B(write_stage, k0);
      cp_async_commit();
      k0 += BK;
      write_stage = (write_stage + 1) % STAGES;
    } else {
      cp_async_commit();
    }

    cp_async_wait_group<STAGES - 1>();
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 32) {
      uint32_t a_regs[M_FRAGS][4];
#pragma unroll
      for (int mf = 0; mf < M_FRAGS; mf++) {
        int m_base = warp_m * WBM + mf * 16;
        int row_in_blk = (lid & 15);
        int col_off = (lid >> 4) * 16;
        uint32_t a_addr = cvta_to_shared(&A_at(read_stage, m_base + row_in_blk, kk + col_off));
        ldmatrix_x4_a(a_regs[mf][0], a_regs[mf][1], a_regs[mf][2], a_regs[mf][3], a_addr);
      }

#pragma unroll
      for (int nf = 0; nf < N_FRAGS; nf++) {
        int n_base = warp_n * WBN + nf * 8;
        uint32_t b_regs[2];
        b_regs[0] =
            *reinterpret_cast<const uint32_t*>(&B_at(read_stage, n_base + gid, kk + tig * 4));
        b_regs[1] =
            *reinterpret_cast<const uint32_t*>(&B_at(read_stage, n_base + gid, kk + tig * 4 + 16));

#pragma unroll
        for (int mf = 0; mf < M_FRAGS; mf++) {
          mma_m16n8k32_e4m3(acc[mf][nf][0], acc[mf][nf][1], acc[mf][nf][2], acc[mf][nf][3],
                            a_regs[mf][0], a_regs[mf][1], a_regs[mf][2], a_regs[mf][3], b_regs[0],
                            b_regs[1]);
        }
      }
    }

    read_stage = (read_stage + 1) % STAGES;
  }

  // Epilogue
  if (SPLIT_K) {
    float* ws = reinterpret_cast<float*>(Out) + (size_t)split_id * M * N;
#pragma unroll
    for (int mf = 0; mf < M_FRAGS; mf++) {
      int m_base = warp_m * WBM + mf * 16;
#pragma unroll
      for (int nf = 0; nf < N_FRAGS; nf++) {
        int n_base = warp_n * WBN + nf * 8;
        int row0 = block_m + m_base + gid;
        int row1 = block_m + m_base + gid + 8;
        int col_base = block_n + n_base + tig * 2;

        if (row0 < M && col_base < N) {
          ws[row0 * N + col_base + 0] = acc[mf][nf][0];
          if (col_base + 1 < N) ws[row0 * N + col_base + 1] = acc[mf][nf][1];
        }
        if (row1 < M && col_base < N) {
          ws[row1 * N + col_base + 0] = acc[mf][nf][2];
          if (col_base + 1 < N) ws[row1 * N + col_base + 1] = acc[mf][nf][3];
        }
      }
    }
  } else {
    __nv_bfloat16* Out_bf = reinterpret_cast<__nv_bfloat16*>(Out);
#pragma unroll
    for (int mf = 0; mf < M_FRAGS; mf++) {
      int m_base = warp_m * WBM + mf * 16;
#pragma unroll
      for (int nf = 0; nf < N_FRAGS; nf++) {
        int n_base = warp_n * WBN + nf * 8;
        int row0 = block_m + m_base + gid;
        int row1 = block_m + m_base + gid + 8;
        int col_base = block_n + n_base + tig * 2;

        __nv_bfloat162 v01, v23;
        v01.x = __float2bfloat16(acc[mf][nf][0] * scale);
        v01.y = __float2bfloat16(acc[mf][nf][1] * scale);
        v23.x = __float2bfloat16(acc[mf][nf][2] * scale);
        v23.y = __float2bfloat16(acc[mf][nf][3] * scale);

        if (row0 < M && col_base < N) {
          *reinterpret_cast<__nv_bfloat162*>(&Out_bf[row0 * N + col_base]) = v01;
        }
        if (row1 < M && col_base < N) {
          *reinterpret_cast<__nv_bfloat162*>(&Out_bf[row1 * N + col_base]) = v23;
        }
      }
    }
  }
}

// Reduce + scale + cast: sum partial fp32 sums across S splits, multiply by scale, cast to bf16.
// Vectorized: each thread processes 4 fp32 elements (float4 load) and writes 4 bf16 (one
// 8-byte aligned store packing two bf162). S is templated so the per-split loop can fully unroll.
template <int S_FIXED>
__global__ void reduce_cast_kernel_vec(const float* __restrict__ workspace,
                                       __nv_bfloat16* __restrict__ out, int total,
                                       const float* A_scale_ptr, const float* B_scale_ptr) {
  const int idx4_base = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int total4 = total / 4;
  const float scale = (*A_scale_ptr) * (*B_scale_ptr);

  for (int i4 = idx4_base; i4 < total4; i4 += stride) {
    float4 sum;
    sum.x = sum.y = sum.z = sum.w = 0.f;
#pragma unroll
    for (int s = 0; s < S_FIXED; s++) {
      const float4* p = reinterpret_cast<const float4*>(workspace + (size_t)s * total) + i4;
      float4 v = *p;
      sum.x += v.x;
      sum.y += v.y;
      sum.z += v.z;
      sum.w += v.w;
    }
    __nv_bfloat162 lo = __floats2bfloat162_rn(sum.x * scale, sum.y * scale);
    __nv_bfloat162 hi = __floats2bfloat162_rn(sum.z * scale, sum.w * scale);
    struct alignas(8) Pair {
      __nv_bfloat162 a, b;
    };
    Pair packed{lo, hi};
    *reinterpret_cast<Pair*>(out + 4 * i4) = packed;
  }
  // Tail (when total is not divisible by 4)
  int remainder_start = total4 * 4;
  int remainder = total - remainder_start;
  if (remainder > 0) {
    int t = idx4_base;
    if (t < remainder) {
      int idx = remainder_start + t;
      float sum = 0.f;
#pragma unroll
      for (int s = 0; s < S_FIXED; s++) {
        sum += workspace[(size_t)s * total + idx];
      }
      out[idx] = __float2bfloat16(sum * scale);
    }
  }
}

// Fallback (non-vectorized) for unusual shapes
__global__ void reduce_cast_kernel(const float* __restrict__ workspace,
                                   __nv_bfloat16* __restrict__ out, int S, int total,
                                   const float* A_scale_ptr, const float* B_scale_ptr) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const float scale = (*A_scale_ptr) * (*B_scale_ptr);
  for (int i = idx; i < total; i += stride) {
    float sum = 0.f;
#pragma unroll
    for (int s = 0; s < 8; s++) {
      if (s < S) sum += workspace[(size_t)s * total + i];
    }
    out[i] = __float2bfloat16(sum * scale);
  }
}

template <int BM, int BN, int BK, int STAGES, int PAD = 16>
constexpr int smem_size() {
  return STAGES * (BM + BN) * (BK + PAD) * sizeof(__nv_fp8_e4m3);
}

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int STAGES, int MIN_BLOCKS = 1>
void launch_kernel_normal(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, __nv_bfloat16* Out,
                          const float* A_scale, const float* B_scale, int M, int N, int K,
                          cudaStream_t stream) {
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
  dim3 block(WARP_M * WARP_N * 32);
  int smem = smem_size<BM, BN, BK, STAGES>();
  auto kfn = fp8_gemm_kernel<BM, BN, BK, WARP_M, WARP_N, STAGES, /*SPLIT_K=*/false, MIN_BLOCKS>;
  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  kfn<<<grid, block, smem, stream>>>(A, B, (void*)Out, A_scale, B_scale, M, N, K, 0);
}

template <int BM, int BN, int BK, int WARP_M, int WARP_N, int STAGES, int MIN_BLOCKS = 1>
void launch_kernel_splitk(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, float* workspace,
                          const float* A_scale, const float* B_scale, int M, int N, int K,
                          int splits, cudaStream_t stream) {
  int K_per_split = (K + splits - 1) / splits;
  K_per_split = (K_per_split + BK - 1) / BK * BK;  // align up to BK
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, splits);
  dim3 block(WARP_M * WARP_N * 32);
  int smem = smem_size<BM, BN, BK, STAGES>();
  auto kfn = fp8_gemm_kernel<BM, BN, BK, WARP_M, WARP_N, STAGES, /*SPLIT_K=*/true, MIN_BLOCKS>;
  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }
  kfn<<<grid, block, smem, stream>>>(A, B, (void*)workspace, A_scale, B_scale, M, N, K,
                                     K_per_split);
}

void launch_reduce_cast(const float* workspace, __nv_bfloat16* out, int splits, int M, int N,
                        const float* A_scale, const float* B_scale, cudaStream_t stream) {
  int total = M * N;
  int block = 128;
  int total4 = total / 4;
  int grid = (total4 + block - 1) / block;
  if (grid > 512) grid = 512;
  if (grid < 1) grid = 1;

  // Use vectorized template if N is multiple of 4 (the common case)
  if (N % 4 == 0) {
    switch (splits) {
      case 2:
        reduce_cast_kernel_vec<2>
            <<<grid, block, 0, stream>>>(workspace, out, total, A_scale, B_scale);
        break;
      case 3:
        reduce_cast_kernel_vec<3>
            <<<grid, block, 0, stream>>>(workspace, out, total, A_scale, B_scale);
        break;
      case 4:
        reduce_cast_kernel_vec<4>
            <<<grid, block, 0, stream>>>(workspace, out, total, A_scale, B_scale);
        break;
      case 5:
        reduce_cast_kernel_vec<5>
            <<<grid, block, 0, stream>>>(workspace, out, total, A_scale, B_scale);
        break;
      case 6:
        reduce_cast_kernel_vec<6>
            <<<grid, block, 0, stream>>>(workspace, out, total, A_scale, B_scale);
        break;
      case 7:
        reduce_cast_kernel_vec<7>
            <<<grid, block, 0, stream>>>(workspace, out, total, A_scale, B_scale);
        break;
      case 8:
        reduce_cast_kernel_vec<8>
            <<<grid, block, 0, stream>>>(workspace, out, total, A_scale, B_scale);
        break;
      default:
        int g = (total + block - 1) / block;
        if (g > 1024) g = 1024;
        reduce_cast_kernel<<<g, block, 0, stream>>>(workspace, out, splits, total, A_scale,
                                                    B_scale);
    }
  } else {
    int g = (total + block - 1) / block;
    if (g > 1024) g = 1024;
    reduce_cast_kernel<<<g, block, 0, stream>>>(workspace, out, splits, total, A_scale, B_scale);
  }
}

// Scalar dot-product kernel for very small problems (M<=16, N<=64, K<=2048).
// 1 CTA per (m, n) output. 32 threads cooperatively reduce K dimension.
// Each thread loads 16 fp8 (uint4) at a time per K-iteration.
// This avoids tensor-core M-padding waste and split-K overhead for tiny shapes.
__global__ __launch_bounds__(32, 8) void scalar_dot_kernel(const __nv_fp8_e4m3* __restrict__ A,
                                                           const __nv_fp8_e4m3* __restrict__ B,
                                                           __nv_bfloat16* __restrict__ Out,
                                                           const float* __restrict__ A_scale_ptr,
                                                           const float* __restrict__ B_scale_ptr,
                                                           int M, int N, int K) {
  int m = blockIdx.y;
  int n = blockIdx.x;
  int tid = threadIdx.x;

  if (m >= M || n >= N) return;

  const __nv_fp8_e4m3* A_row = A + m * K;
  const __nv_fp8_e4m3* B_row = B + n * K;

  float sum = 0.0f;
// Each thread takes 16 fp8 per pass, 32 threads × 16 = 512 elements per warp pass
// For K=2048: 4 passes. For K=1024: 2 passes. For K=512: 1 pass (with bounds).
#pragma unroll 4
  for (int kk = 0; kk < K; kk += 32 * 16) {
    int k = kk + tid * 16;
    if (k + 16 <= K) {
      uint4 a4 = *reinterpret_cast<const uint4*>(A_row + k);
      uint4 b4 = *reinterpret_cast<const uint4*>(B_row + k);
      const __nv_fp8_e4m3* a8 = reinterpret_cast<const __nv_fp8_e4m3*>(&a4);
      const __nv_fp8_e4m3* b8 = reinterpret_cast<const __nv_fp8_e4m3*>(&b4);
#pragma unroll
      for (int j = 0; j < 16; j++) {
        sum = fmaf(float(a8[j]), float(b8[j]), sum);
      }
    }
  }

// Warp shuffle reduce (32 lanes -> lane 0)
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }

  if (tid == 0) {
    float scale = (*A_scale_ptr) * (*B_scale_ptr);
    Out[m * N + n] = __float2bfloat16(sum * scale);
  }
}

// Multi-warp scalar dot for wider K (up to 8192). NWARPS warps cooperatively
// reduce K-dimension. Each warp covers K/NWARPS elements.
// 1 CTA = NWARPS*32 threads = 1 (m,n) output.
template <int NWARPS>
__global__ __launch_bounds__(NWARPS * 32, 1024 / (NWARPS * 32)) void scalar_dot_mw_kernel(
    const __nv_fp8_e4m3* __restrict__ A, const __nv_fp8_e4m3* __restrict__ B,
    __nv_bfloat16* __restrict__ Out, const float* __restrict__ A_scale_ptr,
    const float* __restrict__ B_scale_ptr, int M, int N, int K) {
  int m = blockIdx.y;
  int n = blockIdx.x;
  int wid = threadIdx.x / 32;
  int lane = threadIdx.x & 31;
  int tid = threadIdx.x;

  if (m >= M || n >= N) return;

  const __nv_fp8_e4m3* A_row = A + m * K;
  const __nv_fp8_e4m3* B_row = B + n * K;

  float sum = 0.0f;
  // Each thread loads 16 fp8 per pass; warp covers 32*16=512 K elements per pass.
  // CTA covers NWARPS*512 K elements per outer iteration.
  constexpr int K_PER_CTA_PASS = NWARPS * 32 * 16;
#pragma unroll 2
  for (int kk = 0; kk < K; kk += K_PER_CTA_PASS) {
    int k = kk + tid * 16;
    if (k + 16 <= K) {
      uint4 a4 = *reinterpret_cast<const uint4*>(A_row + k);
      uint4 b4 = *reinterpret_cast<const uint4*>(B_row + k);
      const __nv_fp8_e4m3* a8 = reinterpret_cast<const __nv_fp8_e4m3*>(&a4);
      const __nv_fp8_e4m3* b8 = reinterpret_cast<const __nv_fp8_e4m3*>(&b4);
#pragma unroll
      for (int j = 0; j < 16; j++) {
        sum = fmaf(float(a8[j]), float(b8[j]), sum);
      }
    }
  }

// Warp shuffle reduce
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xffffffff, sum, offset);
  }

  // Cross-warp reduce via shared memory
  __shared__ float warp_sums[NWARPS];
  if (lane == 0) warp_sums[wid] = sum;
  __syncthreads();

  if (wid == 0 && lane < NWARPS) {
    float s = warp_sums[lane];
// Reduce across NWARPS lanes
#pragma unroll
    for (int offset = NWARPS / 2; offset > 0; offset >>= 1) {
      s += __shfl_xor_sync((1u << NWARPS) - 1, s, offset);
    }
    if (lane == 0) {
      float scale = (*A_scale_ptr) * (*B_scale_ptr);
      Out[m * N + n] = __float2bfloat16(s * scale);
    }
  }
}

void launch_scalar_dot(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, __nv_bfloat16* Out,
                       const float* A_scale, const float* B_scale, int M, int N, int K,
                       cudaStream_t stream) {
  dim3 grid(N, M);
  dim3 block(32);
  scalar_dot_kernel<<<grid, block, 0, stream>>>(A, B, Out, A_scale, B_scale, M, N, K);
}

template <int NWARPS>
void launch_scalar_dot_mw(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, __nv_bfloat16* Out,
                          const float* A_scale, const float* B_scale, int M, int N, int K,
                          cudaStream_t stream) {
  dim3 grid(N, M);
  dim3 block(NWARPS * 32);
  scalar_dot_mw_kernel<NWARPS><<<grid, block, 0, stream>>>(A, B, Out, A_scale, B_scale, M, N, K);
}

// GEMV-style kernel: 1 CTA per N output, computes MBLK m-rows at once.
// Dispatcher ensures M == MBLK, so no runtime M check needed.
template <int MBLK, int NWARPS>
__global__ __launch_bounds__(NWARPS * 32, 4) void gemv_kernel(const __nv_fp8_e4m3* __restrict__ A,
                                                              const __nv_fp8_e4m3* __restrict__ B,
                                                              __nv_bfloat16* __restrict__ Out,
                                                              const float* __restrict__ A_scale_ptr,
                                                              const float* __restrict__ B_scale_ptr,
                                                              int M, int N, int K) {
  int n = blockIdx.x;
  int tid = threadIdx.x;
  int wid = tid / 32;
  int lane = tid & 31;

  if (n >= N) return;

  const __nv_fp8_e4m3* B_row = B + (size_t)n * K;

  float acc[MBLK];
#pragma unroll
  for (int m = 0; m < MBLK; m++) acc[m] = 0.0f;

  constexpr int K_PER_PASS = NWARPS * 32 * 16;
  for (int kk = 0; kk < K; kk += K_PER_PASS) {
    int k = kk + tid * 16;
    if (k + 16 <= K) {
      uint4 b4 = *reinterpret_cast<const uint4*>(B_row + k);
      const __nv_fp8_e4m3* b8 = reinterpret_cast<const __nv_fp8_e4m3*>(&b4);

#pragma unroll
      for (int m = 0; m < MBLK; m++) {
        uint4 a4 = *reinterpret_cast<const uint4*>(A + (size_t)m * K + k);
        const __nv_fp8_e4m3* a8 = reinterpret_cast<const __nv_fp8_e4m3*>(&a4);
#pragma unroll
        for (int j = 0; j < 16; j++) {
          acc[m] = fmaf(float(a8[j]), float(b8[j]), acc[m]);
        }
      }
    }
  }

// Warp shuffle reduce per m
#pragma unroll
  for (int m = 0; m < MBLK; m++) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
    }
  }

  if constexpr (NWARPS == 1) {
    if (tid == 0) {
      float scale = (*A_scale_ptr) * (*B_scale_ptr);
#pragma unroll
      for (int m = 0; m < MBLK; m++) {
        Out[(size_t)m * N + n] = __float2bfloat16(acc[m] * scale);
      }
    }
  } else {
    // Cross-warp reduce via shared memory
    __shared__ float warp_sums[NWARPS][MBLK];
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < MBLK; m++) {
        warp_sums[wid][m] = acc[m];
      }
    }
    __syncthreads();

    if (wid == 0 && lane < NWARPS) {
      float scale = (*A_scale_ptr) * (*B_scale_ptr);
#pragma unroll
      for (int m = 0; m < MBLK; m++) {
        float s = warp_sums[lane][m];
#pragma unroll
        for (int offset = NWARPS / 2; offset > 0; offset >>= 1) {
          s += __shfl_xor_sync((1u << NWARPS) - 1, s, offset);
        }
        if (lane == 0) {
          Out[(size_t)m * N + n] = __float2bfloat16(s * scale);
        }
      }
    }
  }
}

template <int MBLK, int NWARPS>
void launch_gemv(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, __nv_bfloat16* Out,
                 const float* A_scale, const float* B_scale, int M, int N, int K,
                 cudaStream_t stream) {
  dim3 grid(N);
  dim3 block(NWARPS * 32);
  gemv_kernel<MBLK, NWARPS><<<grid, block, 0, stream>>>(A, B, Out, A_scale, B_scale, M, N, K);
}

// Wide gemv: each CTA produces NBLK N-outputs × MBLK m-outputs.
// Reduces total CTA count by NBLK, amortizes A loads across NBLK n's.
// Dispatch ensures M == MBLK and (N % NBLK == 0 || handles tail).
template <int MBLK, int NBLK, int NWARPS>
__global__ __launch_bounds__(NWARPS * 32, 4) void gemv_wide_kernel(
    const __nv_fp8_e4m3* __restrict__ A, const __nv_fp8_e4m3* __restrict__ B,
    __nv_bfloat16* __restrict__ Out, const float* __restrict__ A_scale_ptr,
    const float* __restrict__ B_scale_ptr, int M, int N, int K) {
  int n_block = blockIdx.x;
  int n_base = n_block * NBLK;
  int tid = threadIdx.x;
  int wid = tid / 32;
  int lane = tid & 31;

  if (n_base >= N) return;

  float acc[MBLK][NBLK];
#pragma unroll
  for (int m = 0; m < MBLK; m++) {
#pragma unroll
    for (int n = 0; n < NBLK; n++) {
      acc[m][n] = 0.0f;
    }
  }

  constexpr int K_PER_PASS = NWARPS * 32 * 16;
  for (int kk = 0; kk < K; kk += K_PER_PASS) {
    int k = kk + tid * 16;
    if (k + 16 <= K) {
      // Load NBLK B uint4s (one per n)
      uint4 b4[NBLK];
#pragma unroll
      for (int nf = 0; nf < NBLK; nf++) {
        int nn = n_base + nf;
        if (nn < N) {
          b4[nf] = *reinterpret_cast<const uint4*>(B + (size_t)nn * K + k);
        }
      }
      // Load MBLK A uint4s
      uint4 a4[MBLK];
#pragma unroll
      for (int mf = 0; mf < MBLK; mf++) {
        a4[mf] = *reinterpret_cast<const uint4*>(A + (size_t)mf * K + k);
      }
// FMAs: MBLK * NBLK * 16
#pragma unroll
      for (int j = 0; j < 16; j++) {
#pragma unroll
        for (int mf = 0; mf < MBLK; mf++) {
          float av = float(reinterpret_cast<const __nv_fp8_e4m3*>(&a4[mf])[j]);
#pragma unroll
          for (int nf = 0; nf < NBLK; nf++) {
            if (n_base + nf < N) {
              float bv = float(reinterpret_cast<const __nv_fp8_e4m3*>(&b4[nf])[j]);
              acc[mf][nf] = fmaf(av, bv, acc[mf][nf]);
            }
          }
        }
      }
    }
  }

// Warp shuffle reduce per (m, n)
#pragma unroll
  for (int m = 0; m < MBLK; m++) {
#pragma unroll
    for (int n = 0; n < NBLK; n++) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        acc[m][n] += __shfl_xor_sync(0xffffffff, acc[m][n], offset);
      }
    }
  }

  if constexpr (NWARPS == 1) {
    if (tid == 0) {
      float scale = (*A_scale_ptr) * (*B_scale_ptr);
#pragma unroll
      for (int m = 0; m < MBLK; m++) {
#pragma unroll
        for (int nf = 0; nf < NBLK; nf++) {
          int nn = n_base + nf;
          if (nn < N) {
            Out[(size_t)m * N + nn] = __float2bfloat16(acc[m][nf] * scale);
          }
        }
      }
    }
  } else {
    // Cross-warp reduce via shared memory
    __shared__ float warp_sums[NWARPS][MBLK][NBLK];
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < MBLK; m++) {
#pragma unroll
        for (int nf = 0; nf < NBLK; nf++) {
          warp_sums[wid][m][nf] = acc[m][nf];
        }
      }
    }
    __syncthreads();

    if (wid == 0 && lane < NWARPS) {
      float scale = (*A_scale_ptr) * (*B_scale_ptr);
#pragma unroll
      for (int m = 0; m < MBLK; m++) {
#pragma unroll
        for (int nf = 0; nf < NBLK; nf++) {
          float s = warp_sums[lane][m][nf];
#pragma unroll
          for (int offset = NWARPS / 2; offset > 0; offset >>= 1) {
            s += __shfl_xor_sync((1u << NWARPS) - 1, s, offset);
          }
          int nn = n_base + nf;
          if (lane == 0 && nn < N) {
            Out[(size_t)m * N + nn] = __float2bfloat16(s * scale);
          }
        }
      }
    }
  }
}

template <int MBLK, int NBLK, int NWARPS>
void launch_gemv_wide(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* B, __nv_bfloat16* Out,
                      const float* A_scale, const float* B_scale, int M, int N, int K,
                      cudaStream_t stream) {
  int n_blocks = (N + NBLK - 1) / NBLK;
  dim3 grid(n_blocks);
  dim3 block(NWARPS * 32);
  gemv_wide_kernel<MBLK, NBLK, NWARPS>
      <<<grid, block, 0, stream>>>(A, B, Out, A_scale, B_scale, M, N, K);
}

}  // namespace

extern "C" void launch_fp8_gemm(const void* A, const void* B, void* Out, const void* A_scale,
                                const void* B_scale, int M, int N, int K,
                                void* workspace,  // float*, may be nullptr
                                int splits,       // 1 = no split-K
                                cudaStream_t stream) {
  auto Ap = (const __nv_fp8_e4m3*)A;
  auto Bp = (const __nv_fp8_e4m3*)B;
  auto Op = (__nv_bfloat16*)Out;
  auto Asp = (const float*)A_scale;
  auto Bsp = (const float*)B_scale;
  auto WSp = (float*)workspace;

  // Scalar dot path for very small N with bounded K.
  if (M <= 16 && N <= 64 && K <= 2048 && K >= 512 && (K % 16 == 0)) {
    launch_scalar_dot(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
    return;
  }

  // Extended scalar_dot for tiny M=1-3, N up to 512, K=2048.
  if (M <= 3 && N <= 512 && K == 2048 && (K % 16 == 0)) {
    launch_scalar_dot(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
    return;
  }

  // M=1 dispatch.
  if (M == 1 && N >= 4096 && (N % 4 == 0) && (K % 16 == 0) && K >= 512 && K <= 8192) {
    if (K <= 2048) {
      launch_gemv_wide<1, 4, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else if (K <= 4096) {
      launch_gemv_wide<1, 4, 2>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else {
      launch_gemv_wide<1, 4, 4>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    }
  }
  if (M == 1 && N >= 64 && N <= 4096 && (K % 16 == 0) && K >= 512 && K <= 8192) {
    if (K <= 2048) {
      launch_gemv<1, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else if (K <= 4096) {
      launch_gemv<1, 2>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else {
      launch_gemv<1, 4>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    }
  }
  // M=2 dispatch.
  if (M == 2 && N >= 4096 && N <= 12288 && (N % 4 == 0) && (K % 16 == 0) && K >= 512 && K <= 4096) {
    if (K <= 2048) {
      launch_gemv_wide<2, 4, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else {
      launch_gemv_wide<2, 4, 2>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    }
  }
  if (M == 2 && N >= 64 && N <= 2048 && (K % 16 == 0) && K >= 512 && K <= 8192) {
    if (K <= 2048) {
      launch_gemv<2, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else if (K <= 4096) {
      launch_gemv<2, 2>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else {
      launch_gemv<2, 4>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    }
  }
  // M=3 dispatch.
  if (M == 3 && N >= 4096 && N <= 12288 && (N % 4 == 0) && (K % 16 == 0) && K >= 512 && K <= 4096) {
    if (K <= 2048) {
      launch_gemv_wide<3, 4, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else {
      launch_gemv_wide<3, 4, 2>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    }
  }
  if (M == 3 && N >= 64 && N <= 2048 && (K % 16 == 0) && K >= 512 && K <= 8192) {
    if (K <= 2048) {
      launch_gemv<3, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else if (K <= 4096) {
      launch_gemv<3, 2>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    } else {
      launch_gemv<3, 4>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      return;
    }
  }

  if (splits > 1 && WSp != nullptr) {
    // Split-K path. Tile sizes match the regular dispatch
    // so the workspace-size estimate in compute_splits() is accurate.
    if (M <= 16) {
      if (N <= 64) {
        launch_kernel_splitk<16, 64, 64, 1, 2, 6, 1>(Ap, Bp, WSp, Asp, Bsp, M, N, K, splits,
                                                     stream);
      } else if (N <= 512) {
        launch_kernel_splitk<16, 64, 64, 1, 4, 6, 1>(Ap, Bp, WSp, Asp, Bsp, M, N, K, splits,
                                                     stream);
      } else {
        // BK=128 path.
        launch_kernel_splitk<16, 128, 128, 1, 4, 4, 1>(Ap, Bp, WSp, Asp, Bsp, M, N, K, splits,
                                                       stream);
      }
    } else if (M <= 128) {
      // Mid-M split-K path.
      if (N <= 1024) {
        launch_kernel_splitk<64, 64, 128, 2, 2, 3, 1>(Ap, Bp, WSp, Asp, Bsp, M, N, K, splits,
                                                      stream);
      } else {
        launch_kernel_splitk<64, 128, 128, 1, 4, 3, 1>(Ap, Bp, WSp, Asp, Bsp, M, N, K, splits,
                                                       stream);
      }
    } else {
      // Large-M split-K
      launch_kernel_splitk<128, 128, 128, 2, 4, 2, 1>(Ap, Bp, WSp, Asp, Bsp, M, N, K, splits,
                                                      stream);
    }
    launch_reduce_cast(WSp, Op, splits, M, N, Asp, Bsp, stream);
  } else {
    // Regular path
    if (M <= 16) {
      // Small-M dispatch.
      if (N >= 1024) {
        // BN=128 path.
        launch_kernel_normal<16, 128, 128, 1, 4, 4, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      } else {
        // BN=64 path.
        launch_kernel_normal<16, 64, 128, 1, 2, 4, 2>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      }
    } else if (M <= 32) {
      // Small-medium M path.
      launch_kernel_normal<32, 128, 128, 1, 4, 4, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
    } else if (M <= 64) {
      if (N <= 1024) {
        // BM=64, BN=64 path.
        launch_kernel_normal<64, 64, 128, 2, 2, 4, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      } else {
        launch_kernel_normal<64, 128, 128, 1, 4, 3, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      }
    } else if (M <= 128) {
      // Mid-M dispatch.
      if (N <= 4096) {
        // BM=64, BN=64 path.
        launch_kernel_normal<64, 64, 128, 2, 2, 4, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      } else {
        launch_kernel_normal<128, 128, 128, 2, 4, 2, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      }
    } else if (M >= 1024) {
      // Large-M dispatch.
      launch_kernel_normal<192, 128, 128, 3, 4, 2, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
    } else {
      // M=129-1023. N-based dispatch:
      // - N<=1024: BM=64 BN=64.
      // - N>=5120: BM=192 BN=128.
      // - otherwise: BM=128 BN=128.
      if (N <= 1024) {
        launch_kernel_normal<64, 64, 128, 2, 2, 4, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      } else if (N >= 5120) {
        launch_kernel_normal<192, 128, 128, 3, 4, 2, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      } else {
        launch_kernel_normal<128, 128, 128, 2, 4, 2, 1>(Ap, Bp, Op, Asp, Bsp, M, N, K, stream);
      }
    }
  }
}
