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
// Rung 1 of the optimization ladder: coalesced loads.  Rung 0 gave each
// thread one output and walked K serially, so the 32 lanes of a warp read 32
// weight rows K bytes apart (uncoalesced; ncu: 97% L1/TEX throughput at 0.5
// TFLOP/s, 74% of issue cycles in LG throttle).  Here a warp owns one output
// and its lanes stride along K: lane l reads act[m, l + 32 i] and
// w[n, l + 32 i], so every warp-wide load touches 64 contiguous bytes of act
// and 32 of w, and the partial sums are combined with a shuffle reduction.
// The 8 warps of a block share one m and own 8 consecutive n, so the act row
// is reused through L1.  Still plain SIMT fp32 FMA, still no shared memory,
// still scalar 1-byte / 2-byte loads.

#include <cuda_fp8.h>

#include <cstdint>

#include "../utils.cuh"

namespace flashinfer {
namespace gemm {

constexpr int kW8A16WarpsPerBlock = 8;

template <typename T>
__global__ void W8A16Fp8GemmCoalescedKernel(const T* __restrict__ act,
                                            const __nv_fp8_e4m3* __restrict__ w,
                                            const float* __restrict__ scale,
                                            const T* __restrict__ bias, T* __restrict__ out, int M,
                                            int N, int K) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int n = blockIdx.x * kW8A16WarpsPerBlock + warp;
  if (n >= N) return;
  const float s = *scale;
  const float b = bias != nullptr ? static_cast<float>(bias[n]) : 0.f;
  const __nv_fp8_e4m3* w_row = w + static_cast<int64_t>(n) * K;
  // gridDim.y is capped at 65535; rows beyond that are strided over.
  for (int m = blockIdx.y; m < M; m += gridDim.y) {
    const T* a_row = act + static_cast<int64_t>(m) * K;
    float acc = 0.f;
    for (int k = lane; k < K; k += 32) {
      acc = fmaf(static_cast<float>(a_row[k]), static_cast<float>(w_row[k]), acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
      out[static_cast<int64_t>(m) * N + n] = static_cast<T>(acc * s + b);
    }
  }
}

template <typename T>
cudaError_t W8A16Fp8Gemm(const T* act, const __nv_fp8_e4m3* w, const float* scale, const T* bias,
                         T* out, int M, int N, int K, cudaStream_t stream) {
  constexpr int kThreads = 32 * kW8A16WarpsPerBlock;
  const dim3 grid(ceil_div(N, kW8A16WarpsPerBlock), static_cast<unsigned>(M < 65535 ? M : 65535));
  W8A16Fp8GemmCoalescedKernel<T><<<grid, kThreads, 0, stream>>>(act, w, scale, bias, out, M, N, K);
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
