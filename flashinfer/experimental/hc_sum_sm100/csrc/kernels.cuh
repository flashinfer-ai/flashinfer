// SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
// SPDX-License-Identifier: Apache-2.0
// Derived from SGLang hc_apply_sum.cuh and hc_raw_transition.cuh at
// 9cdd6624fe115979b0a0e6c588d9de5ddfeaafe2.
// Modified 2026-11-04: FlashInfer vectors and framework-independent launch parameters.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <flashinfer/vec_dtypes.cuh>

namespace flashinfer::hc_sum {

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int mask = 16; mask; mask >>= 1) x += __shfl_xor_sync(0xffffffffu, x, mask);
  return x;
}

template <typename T>
struct ApplyParams {
  const T* block_output;
  const T* residual;
  const float* alpha;
  T* output;
  float* sums;
};

template <int H, typename T, bool Partials>
__global__ __launch_bounds__(160) void hc_apply_sum_kernel(
    const ApplyParams<T> __grid_constant__ p) {
  static_assert(H > 0 && H % 512 == 0 && H <= 16384);
  const unsigned token = blockIdx.x;
  const unsigned branch = blockIdx.y / 2;
  const unsigned half = blockIdx.y % 2;
  const unsigned lane = threadIdx.x % 32;
  constexpr unsigned vecs_per_half = H / 16;
  constexpr unsigned iterations = (vecs_per_half + 159) / 160;
  const int64_t offset = (int64_t(token) * 4 + branch) * H;
  const float alpha = p.alpha[int64_t(token) * 4 + branch];
  float sum = 0.0f;
#pragma unroll
  for (unsigned j = 0; j < iterations; ++j) {
    const unsigned local = threadIdx.x + j * 160;
    if (local < vecs_per_half) {
      const unsigned vec = half * vecs_per_half + local;
      vec_t<T, 8> r, y, out;
      r.load(p.residual + offset + vec * 8);
      y.load(p.block_output + int64_t(token) * H + vec * 8);
#pragma unroll
      for (unsigned i = 0; i < 4; ++i) {
        const float ux = float(r[2 * i]) + alpha * float(y[2 * i]);
        const float uy = float(r[2 * i + 1]) + alpha * float(y[2 * i + 1]);
        sum += ux * ux + uy * uy;
        out[2 * i] = T(ux);
        out[2 * i + 1] = T(uy);
      }
      out.store(p.output + offset + vec * 8);
    }
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    const int64_t group = int64_t(token) * 4 + branch;
    if constexpr (Partials) {
      p.sums[group * 10 + half * 5 + threadIdx.x / 32] = sum;
    } else {
      atomicAdd(p.sums + group, sum);
    }
  }
}

static __global__ __launch_bounds__(128) void hc_sum_partials_reduce_kernel(const float* partials,
                                                                            float* sums) {
  const unsigned lane = threadIdx.x % 32;
  const int64_t group = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
  float sum = lane < 10 ? partials[group * 10 + lane] : 0.0f;
  sum = warp_sum(sum);
  if (lane == 0) sums[group] = sum;
}

template <int H, typename T>
__global__ __launch_bounds__(H / 16) void hc_bootstrap_kernel(const T* residual, float* sums) {
  constexpr unsigned warps = H / 16 / 32;
  const int64_t group = blockIdx.x;
  const unsigned lane = threadIdx.x % 32;
  const unsigned warp = threadIdx.x / 32;
  __shared__ float partial[32];
  vec_t<T, 16> r;
  r.load(residual + group * H + threadIdx.x * 16);
  float sum = 0.0f;
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    const float ux = float(r[2 * i]), uy = float(r[2 * i + 1]);
    sum += ux * ux + uy * uy;
  }
  sum = warp_sum(sum);
  if (lane == 0) partial[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    float value = warp_sum(lane < warps ? partial[lane] : 0.0f);
    if (lane == 0) sums[group] = value;
  }
}
}  // namespace flashinfer::hc_sum
