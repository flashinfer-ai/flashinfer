/*
 * Copyright (c) 2023 by FlashInfer team.
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
#ifndef FLASHINFER_UTILS_CUH_
#define FLASHINFER_UTILS_CUH_
#include <cuda_runtime.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// macro to turn off fp16 qk reduction to reduce binary
#ifndef FLASHINFER_ALWAYS_DISALLOW_FP16_QK_REDUCTION
#define FLASHINFER_ALWAYS_DISALLOW_FP16_QK_REDUCTION 0
#endif

#ifndef NDEBUG
#define FLASHINFER_CUDA_CALL(func, ...)                                                     \
  {                                                                                         \
    cudaError_t e = (func);                                                                 \
    if (e != cudaSuccess) {                                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << STR(func) << std::endl;      \
      return e;                                                                             \
    }                                                                                       \
  }
#else
#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }
#endif

#define DISPATCH_SPLIT_QO_INDPTR(split_qo_indptr, SPLIT_QO_INDPTR, ...) \
  if (split_qo_indptr) {                                                \
    constexpr bool SPLIT_QO_INDPTR = true;                              \
    __VA_ARGS__                                                         \
  } else {                                                              \
    constexpr bool SPLIT_QO_INDPTR = false;                             \
    __VA_ARGS__                                                         \
  }

#define DISPATCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, ...) \
  if (num_frags_x == 1) {                                   \
    constexpr size_t NUM_FRAGS_X = 1;                       \
    __VA_ARGS__                                             \
  } else if (num_frags_x == 2) {                            \
    constexpr size_t NUM_FRAGS_X = 2;                       \
    __VA_ARGS__                                             \
  } else {                                                  \
    std::ostringstream err_msg;                             \
    err_msg << "Unsupported num_frags_x: " << num_frags_x;  \
    throw std::invalid_argument(err_msg.str());             \
  }

#define DISPATCH_NUM_FRAGS_Z(max_frags_z, NUM_FRAGS_Z, ...) \
  if (max_frags_z >= 4) {                                   \
    constexpr size_t NUM_FRAGS_Z = 4;                       \
    __VA_ARGS__                                             \
  } else if (max_frags_z >= 2) {                            \
    constexpr size_t NUM_FRAGS_Z = 2;                       \
    __VA_ARGS__                                             \
  } else if (max_frags_z >= 1) {                            \
    constexpr size_t NUM_FRAGS_Z = 1;                       \
    __VA_ARGS__                                             \
  } else {                                                  \
    std::ostringstream err_msg;                             \
    err_msg << "Unsupported max_frags_z: " << max_frags_z;  \
    throw std::invalid_argument(err_msg.str());             \
  }

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      throw std::invalid_argument(err_msg.str());                          \
    }                                                                      \
  }

namespace flashinfer {

inline bool is_device_ptr(const void* ptr) {
  cudaPointerAttributes attrs;
  FLASHINFER_CUDA_CALL(cudaPointerGetAttributes(&attrs, ptr));
  return attrs.type == cudaMemoryTypeDevice;
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

template <typename IdType>
std::tuple<IdType, IdType, std::vector<IdType>, std::vector<IdType>> split_qo_indptr(
    IdType* qo_indptr, uint32_t batch_size, uint32_t gqa_group_size, uint32_t head_dim,
    cudaStream_t stream = nullptr) {
  constexpr uint32_t num_warps = 4;
  std::vector<IdType> qo_indptr_h(batch_size + 1), request_indices, tile_indices;
  if (is_device_ptr((void*)qo_indptr)) {
    cudaMemcpyAsync(qo_indptr_h.data(), qo_indptr, sizeof(IdType) * (batch_size + 1),
                    cudaMemcpyDeviceToHost, stream);
  } else {
    qo_indptr_h.assign(qo_indptr, qo_indptr + batch_size + 1);
  }

  const uint32_t total_q_len = qo_indptr_h[batch_size];
  const bool avg_len_greater_than_64 = total_q_len * gqa_group_size > 64 * batch_size;
  const uint32_t num_frags_x = (head_dim < 256 && avg_len_greater_than_64) ? 2 : 1;
  const uint32_t num_rows_per_cta = num_frags_x * num_warps * 16;
  uint32_t num_qo_tiles = 0;

  for (uint32_t i = 0; i < batch_size; ++i) {
    for (uint32_t j = qo_indptr_h[i] * gqa_group_size; j < qo_indptr_h[i + 1] * gqa_group_size;
         j += num_rows_per_cta) {
      request_indices.push_back(i);
      tile_indices.push_back((j - qo_indptr_h[i] * gqa_group_size) / num_rows_per_cta);
      ++num_qo_tiles;
    }
  }

  return {num_frags_x, num_qo_tiles, std::move(request_indices), std::move(tile_indices)};
}

}  // namespace flashinfer

#endif  // FLASHINFER_UTILS_CUH_
