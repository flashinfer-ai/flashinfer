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

#ifndef FLASHINFER_ACTIVATION_CUH_
#define FLASHINFER_ACTIVATION_CUH_

#include "math.cuh"
#include "utils.cuh"
#include "vec_dtypes.cuh"

namespace flashinfer {

namespace activation {

// https://github.com/NVIDIA/FasterTransformer/blob/d21dc02bc5f70bc7dc0d18ba5801ae263565e68e/src/fastertransformer/kernels/activation_kernels.cu#L126-L133
__device__ __forceinline__ float silu_kernel(const float& val) {
  // NOTE(Zihao): use __fdividef might be faster, at the cost of precision
  return val / (1.0f + __expf(-val));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& val) {
  const float cdf =
      0.5f * (1.0f + math::tanh((0.7978845608028654f * (val + 0.044715f * val * val * val))));
  return val * cdf;
}

template <typename T, float (*Activation)(const float&)>
__global__ void act_and_mul_kernel(T* __restrict__ out, const T* __restrict__ input, const int d) {
  constexpr uint32_t vec_size = 16 / sizeof(T);
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;

#pragma unroll 1
  for (uint32_t idx = thread_idx; idx < d / vec_size; idx += stride) {
    vec_t<float, vec_size> x_vec, y_vec, out_vec;
    x_vec.cast_load(input + offset + idx * vec_size);
    y_vec.cast_load(input + offset + d + idx * vec_size);
#pragma unroll
    for (uint32_t i = 0; i < vec_size; ++i) {
      out_vec[i] = Activation(x_vec[i]) * y_vec[i];
    }
    out_vec.cast_store(out + token_idx * d + idx * vec_size);
  }

  const int64_t remaining_offset = d - d % (stride * vec_size);
  // process the remaining elements
#pragma unroll 1
  for (int64_t idx = thread_idx; idx < d % (stride * vec_size); idx += stride) {
    float x = input[offset + remaining_offset + idx],
          y = input[offset + remaining_offset + d + idx];
    out[token_idx * d + remaining_offset + idx] = Activation(x) * y;
  }
}

}  // namespace activation
}  // namespace flashinfer

#endif  // FLASHINFER_ACTIVATION_CUH_
