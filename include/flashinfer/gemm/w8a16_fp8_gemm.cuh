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
// Rung 0 of the optimization ladder: deliberately naive.  One output element
// per thread, plain SIMT fp32 FMA, every operand read straight from global
// memory, no shared memory.  Adjacent threads own adjacent n, so adjacent
// threads read weight rows K bytes apart: the weight loads are uncoalesced
// by construction.  This is the baseline every later rung is measured
// against; it is kept correct, not fast.

#include <cuda_fp8.h>

#include <cstdint>

#include "../utils.cuh"

namespace flashinfer {
namespace gemm {

template <typename T>
__global__ void W8A16Fp8GemmNaiveKernel(const T* __restrict__ act,
                                        const __nv_fp8_e4m3* __restrict__ w,
                                        const float* __restrict__ scale, const T* __restrict__ bias,
                                        T* __restrict__ out, int M, int N, int K) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float s = *scale;
  const float b = bias != nullptr ? static_cast<float>(bias[n]) : 0.f;
  // gridDim.y is capped at 65535; rows beyond that are strided over.
  for (int m = blockIdx.y; m < M; m += gridDim.y) {
    const T* a_row = act + static_cast<int64_t>(m) * K;
    const __nv_fp8_e4m3* w_row = w + static_cast<int64_t>(n) * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      acc = fmaf(static_cast<float>(a_row[k]), static_cast<float>(w_row[k]), acc);
    }
    out[static_cast<int64_t>(m) * N + n] = static_cast<T>(acc * s + b);
  }
}

template <typename T>
cudaError_t W8A16Fp8Gemm(const T* act, const __nv_fp8_e4m3* w, const float* scale, const T* bias,
                         T* out, int M, int N, int K, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const dim3 grid(ceil_div(N, kThreads), static_cast<unsigned>(M < 65535 ? M : 65535));
  W8A16Fp8GemmNaiveKernel<T><<<grid, kThreads, 0, stream>>>(act, w, scale, bias, out, M, N, K);
  return cudaGetLastError();
}

}  // namespace gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GEMM_W8A16_FP8_GEMM_CUH_
