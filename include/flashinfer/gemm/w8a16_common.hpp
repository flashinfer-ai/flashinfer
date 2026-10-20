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
#ifndef FLASHINFER_GEMM_W8A16_COMMON_HPP_
#define FLASHINFER_GEMM_W8A16_COMMON_HPP_

// Architecture-independent pieces of the W8A16 GEMM (fp8 e4m3 weight, 16-bit activation):
// the decomposition selector, the activation type mapping, the split-K reduce and the
// tile-count arithmetic shared by the Hopper (w8a16_sm90/) and Blackwell (w8a16_sm100/)
// variants.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

#include "cutlass/numeric_types.h"

namespace flashinfer::gemm {

// How a kernel variant covers the reduction dimension.
//   kPlain:   one CTA (or 2SM cluster) per output tile walks the whole K.
//   kSplitK:  the GEMM runs as a batch of `splits` K-slices in one launch, each writing an
//             fp32 partial; W8A16SplitKReduceKernel sums the partials and applies scale and
//             bias once.
//   kStreamK: CUTLASS's stream-K scheduler cuts the tiles of the partial last wave(s) along
//             K across the resident CTAs and reduces the peers through its workspace.
enum class W8A16Decomposition { kPlain, kSplitK, kStreamK };

// Activation dtype of the public op -> the CUTLASS element type.
template <typename T>
struct W8A16CutlassType;
template <>
struct W8A16CutlassType<__nv_bfloat16> {
  using type = cutlass::bfloat16_t;
};
template <>
struct W8A16CutlassType<half> {
  using type = cutlass::half_t;
};

// Tile count along one dimension.  The launchers admit dimensions up to INT32_MAX, so
// `(dim + tile - 1)` is formed in 64 bits: as int it overflows for dim > INT32_MAX - tile.
inline long long W8A16Tiles(long long dim, long long tile) { return (dim + tile - 1) / tile; }

// CUTLASS workspace sizes are rounded up to 256 bytes before the split-K partials follow.
inline size_t W8A16Align256(size_t bytes) { return (bytes + 255) / 256 * 256; }

// Split-K reduce: out[i] = T(sum_s ws[s][i] * scale + bias[i % N]), 8 elements per thread
// (two 16 B loads per split, one 16 B store).  Requires N % 8 == 0 (the launcher checks it)
// and hence M * N % 8 == 0.
template <typename T>
__global__ void W8A16SplitKReduceKernel(const float* __restrict__ ws, int splits,
                                        const float* __restrict__ scale, const T* __restrict__ bias,
                                        T* __restrict__ out, int64_t mn, int N) {
  const int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (i >= mn) return;
  float acc[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) acc[e] = 0.f;
  for (int sp = 0; sp < splits; ++sp) {
    const float4* src = reinterpret_cast<const float4*>(ws + static_cast<int64_t>(sp) * mn + i);
    const float4 a = src[0], b = src[1];
    acc[0] += a.x;
    acc[1] += a.y;
    acc[2] += a.z;
    acc[3] += a.w;
    acc[4] += b.x;
    acc[5] += b.y;
    acc[6] += b.z;
    acc[7] += b.w;
  }
  const float sc = *scale;
  const int n = static_cast<int>(i % N);
  alignas(16) T v[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float bb = bias != nullptr ? static_cast<float>(bias[n + e]) : 0.f;
    v[e] = static_cast<T>(acc[e] * sc + bb);
  }
  *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<const uint4*>(v);
}

// Launches the split-K reduce over `splits` fp32 partials of M x N at `partials`.
template <typename T>
cudaError_t W8A16SplitKReduce(const float* partials, int splits, const float* scale, const T* bias,
                              T* out, int M, int N, cudaStream_t stream) {
  const int64_t mn = static_cast<int64_t>(M) * N;
  const int threads = 256;
  const int64_t blocks = (mn / 8 + threads - 1) / threads;
  W8A16SplitKReduceKernel<T><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(
      partials, splits, scale, bias, out, mn, N);
  return cudaGetLastError();
}

}  // namespace flashinfer::gemm

#endif  // FLASHINFER_GEMM_W8A16_COMMON_HPP_
