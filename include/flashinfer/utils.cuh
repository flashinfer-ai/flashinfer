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

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

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

#define SWITCH_SPLIT_QO_INDPTR(split_qo_indptr, SPLIT_QO_INDPTR, ...) \
  if (split_qo_indptr) {                                              \
    constexpr bool SPLIT_QO_INDPTR = true;                            \
    __VA_ARGS__                                                       \
  } else {                                                            \
    constexpr bool SPLIT_QO_INDPTR = false;                           \
    __VA_ARGS__                                                       \
  }

#define SWITCH_ALLOW_FP16_QK_REDUCTION(allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, ...) \
  if (allow_fp16_qk_reduction) {                                                              \
    constexpr bool ALLOW_FP16_QK_REDUCTION = true;                                            \
    __VA_ARGS__                                                                               \
  } else {                                                                                    \
    constexpr bool ALLOW_FP16_QK_REDUCTION = false;                                           \
    __VA_ARGS__                                                                               \
  }

#define SWITCH_PAGE_SIZE(page_size, PAGE_SIZE, ...) \
  if (page_size == 1) {                             \
    constexpr size_t PAGE_SIZE = 1;                 \
    __VA_ARGS__                                     \
  } else if (page_size == 8) {                      \
    constexpr size_t PAGE_SIZE = 8;                 \
    __VA_ARGS__                                     \
  } else if (page_size == 16) {                     \
    constexpr size_t PAGE_SIZE = 16;                \
    __VA_ARGS__                                     \
  } else if (page_size == 32) {                     \
    constexpr size_t PAGE_SIZE = 32;                \
    __VA_ARGS__                                     \
  } else {                                          \
    constexpr size_t PAGE_SIZE = 0;                 \
    __VA_ARGS__                                     \
  }

#define SWITCH_NUM_FRAGS_X(num_frags_x, NUM_FRAGS_X, ...)                 \
  if (num_frags_x == 1) {                                                 \
    constexpr size_t NUM_FRAGS_X = 1;                                     \
    __VA_ARGS__                                                           \
  } else if (num_frags_x == 2) {                                          \
    constexpr size_t NUM_FRAGS_X = 2;                                     \
    __VA_ARGS__                                                           \
  } else {                                                                \
    std::cerr << "Unsupported num_frags_x: " << num_frags_x << std::endl; \
  }

#define SWITCH_NUM_FRAGS_Z(max_frags_z, NUM_FRAGS_Z, ...)                 \
  if (max_frags_z == 4) {                                                 \
    constexpr size_t NUM_FRAGS_Z = 4;                                     \
    __VA_ARGS__                                                           \
  } else if (max_frags_z == 2) {                                          \
    constexpr size_t NUM_FRAGS_Z = 2;                                     \
    __VA_ARGS__                                                           \
  } else {                                                                \
    std::cerr << "Unsupported max_frags_z: " << max_frags_z << std::endl; \
  }

#define SWITCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...)              \
  if (group_size == 1) {                                                \
    constexpr size_t GROUP_SIZE = 1;                                    \
    __VA_ARGS__                                                         \
  } else if (group_size == 4) {                                         \
    constexpr size_t GROUP_SIZE = 4;                                    \
    __VA_ARGS__                                                         \
  } else if (group_size == 8) {                                         \
    constexpr size_t GROUP_SIZE = 8;                                    \
    __VA_ARGS__                                                         \
  } else {                                                              \
    std::cerr << "Unsupported group_size: " << group_size << std::endl; \
  }

#define SWITCH_CAUSAL(causal, CAUSAL, ...) \
  if (causal) {                            \
    constexpr bool CAUSAL = true;          \
    __VA_ARGS__                            \
  } else {                                 \
    constexpr bool CAUSAL = false;         \
    __VA_ARGS__                            \
  }

#define SWITCH_LAYOUT(layout, LAYOUT, ...)              \
  switch (layout) {                                     \
    case QKVLayout::kNHD: {                             \
      constexpr QKVLayout LAYOUT = QKVLayout::kNHD;     \
      __VA_ARGS__                                       \
      break;                                            \
    }                                                   \
    case QKVLayout::kHND: {                             \
      constexpr QKVLayout LAYOUT = QKVLayout::kHND;     \
      __VA_ARGS__                                       \
      break;                                            \
    }                                                   \
    default: {                                          \
      std::ostringstream err_msg;                       \
      err_msg << "Unsupported layout: " << int(layout); \
      throw std::invalid_argument(err_msg.str());       \
    }                                                   \
  }

#define SWITCH_HEAD_DIM(head_dim, HEAD_DIM, ...)       \
  switch (head_dim) {                                  \
    case 64: {                                         \
      constexpr size_t HEAD_DIM = 64;                  \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 128: {                                        \
      constexpr size_t HEAD_DIM = 128;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 256: {                                        \
      constexpr size_t HEAD_DIM = 256;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    default: {                                         \
      std::ostringstream err_msg;                      \
      err_msg << "Unsupported head_dim: " << head_dim; \
      throw std::invalid_argument(err_msg.str());      \
    }                                                  \
  }

#define SWITCH_HEAD_DIM_PREFILL(head_dim, HEAD_DIM, ...) \
  switch (head_dim) {                                    \
    case 64: {                                           \
      constexpr size_t HEAD_DIM = 64;                    \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    case 128: {                                          \
      constexpr size_t HEAD_DIM = 128;                   \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    default: {                                           \
      std::ostringstream err_msg;                        \
      err_msg << "Unsupported head_dim: " << head_dim;   \
      throw std::invalid_argument(err_msg.str());        \
    }                                                    \
  }

#define SWITCH_ROTARY_MODE(rotary_mode, ROTARY_MODE, ...)         \
  switch (rotary_mode) {                                          \
    case RotaryMode::kNone: {                                     \
      constexpr RotaryMode ROTARY_MODE = RotaryMode::kNone;       \
      __VA_ARGS__                                                 \
      break;                                                      \
    }                                                             \
    case RotaryMode::kLlama: {                                    \
      constexpr RotaryMode ROTARY_MODE = RotaryMode::kLlama;      \
      __VA_ARGS__                                                 \
      break;                                                      \
    }                                                             \
    default: {                                                    \
      std::ostringstream err_msg;                                 \
      err_msg << "Unsupported rotary_mode: " << int(rotary_mode); \
      throw std::invalid_argument(err_msg.str());                 \
    }                                                             \
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

}  // namespace flashinfer

#endif  // FLASHINFER_UTILS_CUH_
