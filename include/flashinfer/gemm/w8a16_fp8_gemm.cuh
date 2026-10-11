/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
#define FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_

// W8A16 GEMM: FP8 e4m3 weight with one per-tensor fp32 scale, 16-bit
// activation, output in the activation dtype, optional per-N bias.
//
//   out[m, n] = sum_k act[m, k] * (float(w[n, k]) * scale)  (+ bias[n])
//
// The weight is dequantized on the fly; the 16-bit copy TRT-LLM's cast-copy
// path materialises never exists.  The weight is [N, K] row-major (the
// torch Linear layout, "NT" GEMM), act is [M, K] row-major.
//
// Rung 2 of the optimization ladder: shared-memory tiling.  Rung 1 streamed
// both operands from L2/DRAM with no reuse beyond L1 (ncu: 1.5 TB/s of
// traffic for 3 TFLOP/s, 44 of 47 issue cycles long-scoreboard).  Here a
// 256-thread block computes a 16x16 output tile: for each BK=64 slab of K it
// stages act[16][64] (bf16) and w[16][64] (fp8) in shared memory with
// coalesced loads along K (each thread moves 4 act and 4 w elements), then
// every thread computes one output from the two staged tiles.  Each staged
// act element now feeds 16 outputs and each weight element 16 outputs
// instead of one.  The smem rows are padded (66 bf16 / 68 fp8 = 33 / 17
// words) so the 16 rows a warp reads at one k land in different banks.
// Still one output per thread, scalar loads, SIMT fp32 FMA.

#include <cuda_fp8.h>

#include <cstdint>

#include "../utils.cuh"

namespace flashinfer {
namespace gemm {

constexpr int kW8A16TileM = 16;
constexpr int kW8A16TileN = 16;
constexpr int kW8A16TileK = 64;
constexpr int kW8A16Threads = kW8A16TileM * kW8A16TileN;  // one output per thread

template <typename T>
__global__ void __launch_bounds__(kW8A16Threads)
    W8A16Fp8GemmSmemTiledKernel(const T* __restrict__ act, const __nv_fp8_e4m3* __restrict__ w,
                                const float* __restrict__ scale, const T* __restrict__ bias,
                                T* __restrict__ out, int M, int N, int K) {
  __shared__ T a_s[kW8A16TileM][kW8A16TileK + 2];
  __shared__ __nv_fp8_e4m3 w_s[kW8A16TileN][kW8A16TileK + 4];

  const int tx = threadIdx.x & (kW8A16TileN - 1);  // n within the tile
  const int ty = threadIdx.x / kW8A16TileN;        // m within the tile
  const int m0 = blockIdx.y * kW8A16TileM;
  const int n0 = blockIdx.x * kW8A16TileN;
  const int m = m0 + ty;
  const int n = n0 + tx;

  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += kW8A16TileK) {
    // Stage: 16 rows x 64 columns of each operand, 4 elements per thread,
    // consecutive threads take consecutive k so the global reads coalesce.
#pragma unroll
    for (int i = 0; i < (kW8A16TileM * kW8A16TileK) / kW8A16Threads; ++i) {
      const int idx = threadIdx.x + i * kW8A16Threads;
      const int r = idx / kW8A16TileK;
      const int c = idx % kW8A16TileK;
      const int gm = m0 + r;
      const int gn = n0 + r;
      const int gk = k0 + c;
      a_s[r][c] = (gm < M && gk < K) ? act[static_cast<int64_t>(gm) * K + gk] : T(0.f);
      w_s[r][c] = (gn < N && gk < K) ? w[static_cast<int64_t>(gn) * K + gk] : __nv_fp8_e4m3(0.f);
    }
    __syncthreads();
#pragma unroll 16
    for (int kk = 0; kk < kW8A16TileK; ++kk) {
      acc = fmaf(static_cast<float>(a_s[ty][kk]), static_cast<float>(w_s[tx][kk]), acc);
    }
    __syncthreads();
  }

  if (m < M && n < N) {
    const float b = bias != nullptr ? static_cast<float>(bias[n]) : 0.f;
    out[static_cast<int64_t>(m) * N + n] = static_cast<T>(acc * (*scale) + b);
  }
}

template <typename T>
cudaError_t W8A16Fp8Gemm(const T* act, const __nv_fp8_e4m3* w, const float* scale, const T* bias,
                         T* out, int M, int N, int K, cudaStream_t stream) {
  const unsigned tiles_m = ceil_div(M, kW8A16TileM);
  if (tiles_m > 65535u) return cudaErrorInvalidValue;  // gridDim.y limit
  const dim3 grid(ceil_div(N, kW8A16TileN), tiles_m);
  W8A16Fp8GemmSmemTiledKernel<T>
      <<<grid, kW8A16Threads, 0, stream>>>(act, w, scale, bias, out, M, N, K);
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
