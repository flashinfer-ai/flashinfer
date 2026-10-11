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
// Rung 3 of the optimization ladder: register blocking.  Rung 2 computed one
// output per thread from the smem tiles, i.e. two smem loads plus two
// conversions per FMA, and ncu showed it issue-bound (82% issue slots busy,
// IPC 3.3, 93% L1/TEX pipe) at 6.9 TFLOP/s.  Here a 256-thread block owns a
// 64x64 output tile and every thread accumulates a 4x4 register block: per k
// it loads 4 act and 4 weight values from smem and does 16 FMAs.  A thread's
// rows / columns are strided by 16 inside the tile (ty + 16 i, tx + 16 j) so
// the 16 weight rows a warp reads at one k are 9 words apart (conflict-free)
// and the 2-byte output stores along n stay coalesced.  Still scalar loads,
// still a per-element fp8 -> float conversion, still SIMT fp32 FMA.

#include <cuda_fp8.h>

#include <cstdint>

#include "../utils.cuh"

namespace flashinfer {
namespace gemm {

constexpr int kW8A16TileM = 64;
constexpr int kW8A16TileN = 64;
constexpr int kW8A16TileK = 32;
constexpr int kW8A16ThreadM = 4;  // outputs per thread along m
constexpr int kW8A16ThreadN = 4;  // outputs per thread along n
constexpr int kW8A16Threads = (kW8A16TileM / kW8A16ThreadM) * (kW8A16TileN / kW8A16ThreadN);

template <typename T>
__global__ void __launch_bounds__(kW8A16Threads)
    W8A16Fp8GemmRegBlockedKernel(const T* __restrict__ act, const __nv_fp8_e4m3* __restrict__ w,
                                 const float* __restrict__ scale, const T* __restrict__ bias,
                                 T* __restrict__ out, int M, int N, int K) {
  constexpr int kThreadsN = kW8A16TileN / kW8A16ThreadN;  // 16
  constexpr int kThreadsM = kW8A16TileM / kW8A16ThreadM;  // 16
  // Row strides of 34 bf16 (17 words) and 36 fp8 (9 words): both odd, so
  // rows 16 apart never share a bank.
  __shared__ T a_s[kW8A16TileM][kW8A16TileK + 2];
  __shared__ __nv_fp8_e4m3 w_s[kW8A16TileN][kW8A16TileK + 4];

  const int tx = threadIdx.x % kThreadsN;
  const int ty = threadIdx.x / kThreadsN;
  const int m0 = blockIdx.y * kW8A16TileM;
  const int n0 = blockIdx.x * kW8A16TileN;

  float acc[kW8A16ThreadM][kW8A16ThreadN];
#pragma unroll
  for (int i = 0; i < kW8A16ThreadM; ++i)
#pragma unroll
    for (int j = 0; j < kW8A16ThreadN; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += kW8A16TileK) {
    // Stage 64 rows x 32 columns of each operand; consecutive threads take
    // consecutive k so the global reads coalesce.
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
#pragma unroll 8
    for (int kk = 0; kk < kW8A16TileK; ++kk) {
      float a_frag[kW8A16ThreadM];
      float w_frag[kW8A16ThreadN];
#pragma unroll
      for (int i = 0; i < kW8A16ThreadM; ++i) {
        a_frag[i] = static_cast<float>(a_s[ty + i * kThreadsM][kk]);
      }
#pragma unroll
      for (int j = 0; j < kW8A16ThreadN; ++j) {
        w_frag[j] = static_cast<float>(w_s[tx + j * kThreadsN][kk]);
      }
#pragma unroll
      for (int i = 0; i < kW8A16ThreadM; ++i)
#pragma unroll
        for (int j = 0; j < kW8A16ThreadN; ++j) acc[i][j] = fmaf(a_frag[i], w_frag[j], acc[i][j]);
    }
    __syncthreads();
  }

  const float s = *scale;
#pragma unroll
  for (int j = 0; j < kW8A16ThreadN; ++j) {
    const int n = n0 + tx + j * kThreadsN;
    if (n >= N) continue;
    const float b = bias != nullptr ? static_cast<float>(bias[n]) : 0.f;
#pragma unroll
    for (int i = 0; i < kW8A16ThreadM; ++i) {
      const int m = m0 + ty + i * kThreadsM;
      if (m < M) out[static_cast<int64_t>(m) * N + n] = static_cast<T>(acc[i][j] * s + b);
    }
  }
}

template <typename T>
cudaError_t W8A16Fp8Gemm(const T* act, const __nv_fp8_e4m3* w, const float* scale, const T* bias,
                         T* out, int M, int N, int K, cudaStream_t stream) {
  const unsigned tiles_m = ceil_div(M, kW8A16TileM);
  if (tiles_m > 65535u) return cudaErrorInvalidValue;  // gridDim.y limit
  const dim3 grid(ceil_div(N, kW8A16TileN), tiles_m);
  W8A16Fp8GemmRegBlockedKernel<T>
      <<<grid, kW8A16Threads, 0, stream>>>(act, w, scale, bias, out, M, N, K);
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
