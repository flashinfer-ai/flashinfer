/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_NORM_CUH_
#define FLASHINFER_NORM_CUH_

#include "flashinfer/utils.cuh"
#include "math.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace norm {

template <typename T>
__global__ void RMSNormKernel(T* __restrict__ x, T* __restrict__ w, T* __restrict__ y,
                              const uint32_t d, float eps) {
  const uint32_t bx = blockIdx.x;
  const uint32_t tx = threadIdx.x, ty = threadIdx.y;
  constexpr uint32_t warp_size = 32;
  const uint32_t num_warps = blockDim.y;
  // NOTE(Zihao): it's guaranteed that num_warps should be smaller than 32
  const uint32_t thread_id = tx + ty * warp_size;
  const uint32_t num_threads = num_warps * warp_size;
  constexpr uint32_t vec_size = 16 / sizeof(T);
  const uint32_t rounds = d / vec_size;
  extern __shared__ float smem[];

  float sum_sq = 0.f;

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, vec_size> x_vec;
    x_vec.fill(0);
    if ((i * num_threads + thread_id) * vec_size < d) {
      x_vec.load(x + bx * d + i * num_threads * vec_size + thread_id * vec_size);
    }
#pragma unroll
    for (uint32_t j = 0; j < vec_size; j++) {
      sum_sq += float(x_vec[j]) * float(x_vec[j]);
    }
  }

  // first, warp reduce sum
#pragma unroll
  for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
    sum_sq += math::shfl_xor_sync(sum_sq, offset);
  }

  smem[ty] = sum_sq;
  __syncthreads();
  // then, cross warp reduce sum using only the first warp
  if (ty == 0) {
    sum_sq = (tx < num_warps) ? smem[tx] : 0.f;
#pragma unroll
    for (uint32_t offset = warp_size / 2; offset > 0; offset /= 2) {
      sum_sq += math::shfl_xor_sync(sum_sq, offset);
    }
    smem[0] = sum_sq;
  }
  __syncthreads();

  float rms_rcp = math::rsqrt(smem[0] / float(d) + eps);

  for (uint32_t i = 0; i < rounds; i++) {
    vec_t<T, vec_size> x_vec;
    vec_t<T, vec_size> w_vec;
    vec_t<T, vec_size> y_vec;
    x_vec.fill(0);
    w_vec.fill(0);
    if ((i * num_threads + thread_id) * vec_size < d) {
      x_vec.load(x + bx * d + i * num_threads * vec_size + thread_id * vec_size);
      w_vec.load(w + i * num_threads * vec_size + thread_id * vec_size);
    }
#pragma unroll
    for (uint32_t j = 0; j < vec_size; j++) {
      y_vec[j] = float(x_vec[j]) * rms_rcp * float(w_vec[j]);
    }
    if ((i * num_threads + thread_id) * vec_size < d) {
      y_vec.store(y + bx * d + i * num_threads * vec_size + thread_id * vec_size);
    }
  }
}

template <typename T>
cudaError_t RMSNorm(T* x, T* w, T* y, uint32_t batch_size, uint32_t d, float eps = 1e-5,
                    cudaStream_t stream = 0) {
  constexpr uint32_t vec_size = 16 / sizeof(T);
  if (d % vec_size != 0) {
    return cudaErrorInvalidValue;
  }
  const uint32_t block_size = std::min<uint32_t>(1024, d / vec_size);
  const uint32_t num_warps = ceil_div(block_size, 32);
  dim3 nblks(batch_size);
  dim3 nthrs(32, num_warps);
  const uint32_t smem_size = num_warps * sizeof(float);
  auto kernel = RMSNormKernel<T>;
  void* args[] = {&x, &w, &y, &d, &eps};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  return cudaSuccess;
}

}  // namespace norm

}  // namespace flashinfer

#endif  // FLASHINFER_NORM_CUH_
