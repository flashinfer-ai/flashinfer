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
#pragma once
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include "generated/dispatch.inc"
#ifdef FLASHINFER_ENABLE_BF16
#include <cuda_bf16.h>
#endif
#ifdef FLASHINFER_ENABLE_FP8
#include <cuda_fp8.h>
#endif

#ifdef FLASHINFER_ENABLE_BF16
#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...) \
  [&]() -> bool {                                                   \
    switch (pytorch_dtype) {                                        \
      case at::ScalarType::Half: {                                  \
        using c_type = nv_half;                                     \
        return __VA_ARGS__();                                       \
      }                                                             \
      case at::ScalarType::BFloat16: {                              \
        using c_type = nv_bfloat16;                                 \
        return __VA_ARGS__();                                       \
      }                                                             \
      default:                                                      \
        return false;                                               \
    }                                                               \
  }()
#else
#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...) \
  [&]() -> bool {                                                   \
    switch (pytorch_dtype) {                                        \
      case at::ScalarType::Half: {                                  \
        using c_type = nv_half;                                     \
        return __VA_ARGS__();                                       \
      }                                                             \
      default:                                                      \
        return false;                                               \
    }                                                               \
  }()
#endif

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(pytorch_dtype, c_type, ...) \
  [&]() -> bool {                                                       \
    switch (pytorch_dtype) {                                            \
      case at::ScalarType::Float8_e4m3fn: {                             \
        using c_type = __nv_fp8_e4m3;                                   \
        return __VA_ARGS__();                                           \
      }                                                                 \
      case at::ScalarType::Float8_e5m2: {                               \
        using c_type = __nv_fp8_e5m2;                                   \
        return __VA_ARGS__();                                           \
      }                                                                 \
      default:                                                          \
        return false;                                                   \
    }                                                                   \
  }()

#define _DISPATCH_SWITCH(cond, ...) \
  [&]() -> bool {                   \
    switch (cond) {                 \
      __VA_ARGS__                   \
      default:                      \
        return false;               \
    }                               \
  }()

#define _DISPATCH_CASE(case_expr, var, ...) \
  case case_expr: {                         \
    constexpr auto var = case_expr;         \
    return __VA_ARGS__();                   \
  }

#define DISPATCH_group_size(expr, ...) \
  _DISPATCH_SWITCH(expr, _DISPATCH_CASES_group_size(__VA_ARGS__))

#define DISPATCH_page_size(expr, ...) _DISPATCH_SWITCH(expr, _DISPATCH_CASES_page_size(__VA_ARGS__))

#define DISPATCH_head_dim(expr, ...) _DISPATCH_SWITCH(expr, _DISPATCH_CASES_head_dim(__VA_ARGS__))

inline void check_shape(const torch::Tensor& a, const torch::Tensor& b, const char* a_name,
                        const char* b_name) {
  TORCH_CHECK(a.dim() == b.dim(), a_name, ".dim() != ", b_name, ".dim(). ", a.dim(), " vs ",
              b.dim());
  for (int i = 0; i < a.dim(); ++i) {
    TORCH_CHECK(a.size(i) == b.size(i), a_name, ".size(", i, ") != ", b_name, ".size(", i, ")");
  }
}

inline constexpr uint32_t pack_u16(uint16_t a, uint16_t b) {
  return (uint32_t(a) << 16) | uint32_t(b);
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

inline bool is_float8_tensor(const torch::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Float8_e4m3fn ||
         tensor.scalar_type() == at::ScalarType::Float8_e5m2;
}
