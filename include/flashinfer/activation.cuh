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

// vec_size must divide d: every vectorized access below (the x half at
// offset, the y half at offset + d, and the output row at token_idx * d) is
// then aligned to vec_size * sizeof(T) bytes. The launcher picks the largest
// power of two that satisfies this, so d = 3420 (Qwen2.5-VL) runs with
// vec_size = 4 instead of faulting on a 16-byte load at an 8-byte address.
template <typename T, float (*Activation)(const float&), uint32_t vec_size>
__global__ __launch_bounds__(256) void act_and_mul_kernel(T* __restrict__ out,
                                                          const T* __restrict__ input,
                                                          const int d) {
  const int64_t token_idx = blockIdx.x;
  const int64_t thread_idx = threadIdx.x;
  const int64_t stride = blockDim.x;
  const int64_t offset = token_idx * 2 * d;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

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

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// Same computation as act_and_mul_kernel for launches where one block per row
// leaves the GPU underused. Each row is split across `split` consecutive blocks
// (blockIdx.x = row * split + chunk), and each thread issues kUnroll independent
// loads per half before computing, instead of one dependent round trip per loop
// iteration. Same vec_size alignment requirement as above.
template <typename T, float (*Activation)(const float&), uint32_t vec_size, uint32_t kUnroll>
__global__ __launch_bounds__(256) void act_and_mul_split_kernel(T* __restrict__ out,
                                                                const T* __restrict__ input,
                                                                const int d, const uint32_t split) {
  const uint32_t row = blockIdx.x / split;
  const uint32_t num_vecs = d / vec_size;
  const uint32_t stride = blockDim.x * split;
  uint32_t idx = (blockIdx.x - row * split) * blockDim.x + threadIdx.x;
  const T* x_ptr = input + int64_t(row) * 2 * d;
  const T* y_ptr = x_ptr + d;
  T* out_ptr = out + int64_t(row) * d;

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

#pragma unroll 1
  for (; idx + (kUnroll - 1) * stride < num_vecs; idx += kUnroll * stride) {
    // Hold the loaded vectors in the input type until all loads are issued;
    // converting to float first would double the registers they occupy.
    vec_t<T, vec_size> x_raw[kUnroll], y_raw[kUnroll];
#pragma unroll
    for (uint32_t k = 0; k < kUnroll; ++k) {
      x_raw[k].load(x_ptr + (idx + k * stride) * vec_size);
      y_raw[k].load(y_ptr + (idx + k * stride) * vec_size);
    }
#pragma unroll
    for (uint32_t k = 0; k < kUnroll; ++k) {
      vec_t<float, vec_size> x_vec, y_vec, out_vec;
      x_vec.cast_from(x_raw[k]);
      y_vec.cast_from(y_raw[k]);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        out_vec[i] = Activation(x_vec[i]) * y_vec[i];
      }
      out_vec.cast_store(out_ptr + (idx + k * stride) * vec_size);
    }
  }
  if constexpr (kUnroll > 1) {
#pragma unroll 1
    for (; idx < num_vecs; idx += stride) {
      vec_t<float, vec_size> x_vec, y_vec, out_vec;
      x_vec.cast_load(x_ptr + idx * vec_size);
      y_vec.cast_load(y_ptr + idx * vec_size);
#pragma unroll
      for (uint32_t i = 0; i < vec_size; ++i) {
        out_vec[i] = Activation(x_vec[i]) * y_vec[i];
      }
      out_vec.cast_store(out_ptr + idx * vec_size);
    }
  }

#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

}  // namespace activation
}  // namespace flashinfer

#endif  // FLASHINFER_ACTIVATION_CUH_
