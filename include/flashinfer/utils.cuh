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

#define DISPATCH_ALLOW_FP16_QK_REDUCTION(allow_fp16_qk_reduction, ALLOW_FP16_QK_REDUCTION, ...) \
  if (allow_fp16_qk_reduction) {                                                                \
    throw std::runtime_error("FP16_QK_REDUCTION disabled at compile time");                     \
  } else {                                                                                      \
    constexpr bool ALLOW_FP16_QK_REDUCTION = false;                                             \
    __VA_ARGS__                                                                                 \
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
  if (max_frags_z >= 8) {                                   \
    constexpr size_t NUM_FRAGS_Z = 8;                       \
    __VA_ARGS__                                             \
  } else if (max_frags_z >= 4) {                            \
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

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 2) {                              \
    constexpr size_t GROUP_SIZE = 2;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 4) {                              \
    constexpr size_t GROUP_SIZE = 4;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else {                                                   \
    std::ostringstream err_msg;                              \
    err_msg << "Unsupported group_size: " << group_size;     \
    throw std::invalid_argument(err_msg.str());              \
  }

#define DISPATCH_MASK_MODE(mask_mode, MASK_MODE, ...)         \
  switch (mask_mode) {                                        \
    case MaskMode::kNone: {                                   \
      constexpr MaskMode MASK_MODE = MaskMode::kNone;         \
      __VA_ARGS__                                             \
      break;                                                  \
    }                                                         \
    case MaskMode::kCausal: {                                 \
      constexpr MaskMode MASK_MODE = MaskMode::kCausal;       \
      __VA_ARGS__                                             \
      break;                                                  \
    }                                                         \
    case MaskMode::kCustom: {                                 \
      constexpr MaskMode MASK_MODE = MaskMode::kCustom;       \
      __VA_ARGS__                                             \
      break;                                                  \
    }                                                         \
    default: {                                                \
      std::ostringstream err_msg;                             \
      err_msg << "Unsupported mask_mode: " << int(mask_mode); \
      throw std::invalid_argument(err_msg.str());             \
    }                                                         \
  }

#define DISPATCH_LOGITS_POST_HOOK(logits_soft_cap, LOGITS_POST_HOOK, ...)       \
  if (logits_soft_cap > 0.f) {                                                  \
    constexpr LogitsPostHook LOGITS_POST_HOOK = LogitsPostHook::kSoftCap;       \
    __VA_ARGS__                                                                 \
  } else if (logits_soft_cap == 0.f) {                                          \
    constexpr LogitsPostHook LOGITS_POST_HOOK = LogitsPostHook::kNone;          \
    __VA_ARGS__                                                                 \
  } else {                                                                      \
    std::ostringstream err_msg;                                                 \
    err_msg << "Invalid logits_soft_cap (should be >= 0): " << logits_soft_cap; \
    throw std::invalid_argument(err_msg.str());                                 \
  }

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)     \
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

#define DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, ...)    \
  switch (pos_encoding_mode) {                                                   \
    case PosEncodingMode::kNone: {                                               \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;      \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kRoPELlama: {                                          \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kRoPELlama; \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kALiBi: {                                              \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kALiBi;     \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    default: {                                                                   \
      std::ostringstream err_msg;                                                \
      err_msg << "Unsupported pos_encoding_mode: " << int(pos_encoding_mode);    \
      throw std::invalid_argument(err_msg.str());                                \
    }                                                                            \
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

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

template <typename T>
inline void DebugPrintCUDAArray(T* device_ptr, size_t size, std::string prefix = "") {
  std::vector<T> host_array(size);
  std::cout << prefix;
  cudaMemcpy(host_array.data(), device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < size; ++i) {
    std::cout << host_array[i] << " ";
  }
  std::cout << std::endl;
}

/*!
 * \brief Return x - y if x > y, otherwise return 0.
 */
__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x, uint32_t y) {
  return (x > y) ? x - y : 0U;
}

__device__ __forceinline__ void swap(uint32_t& a, uint32_t& b) {
  uint32_t tmp = a;
  a = b;
  b = tmp;
}

}  // namespace flashinfer

#endif  // FLASHINFER_UTILS_CUH_
