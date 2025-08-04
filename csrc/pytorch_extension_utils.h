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
#include <Python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#ifdef FLASHINFER_ENABLE_BF16
#include <cuda_bf16.h>
#endif

#ifdef FLASHINFER_ENABLE_F16
#include <cuda_fp16.h>
#endif

#if defined(FLASHINFER_ENABLE_FP8_E4M3) || defined(FLASHINFER_ENABLE_FP8_E5M2) || \
    defined(FLASHINFER_ENABLE_FP8_E8M0)
#include <cuda_fp8.h>
#endif

#if defined(FLASHINFER_ENABLE_FP4_E2M1)
#include <cuda_fp4.h>
#endif

#ifndef FLASHINFER_EXT_MODULE_INITED
#define FLASHINFER_EXT_MODULE_INITED

// To expand macros in #name
#define FLASHINFER_EXT_MODULE_INIT_EXPAND(name) FLASHINFER_EXT_MODULE_INIT(name)

/* Creates a dummy empty module that can be imported from Python.
   The import from Python will load the .so consisting of the file
   in this extension, so that the TORCH_LIBRARY_FRAGMENT static initializers
   are run. */
#define FLASHINFER_EXT_MODULE_INIT(name)                                  \
  extern "C" {                                                            \
  __attribute__((weak)) PyObject* PyInit_##name(void) {                   \
    static struct PyModuleDef module_def = {                              \
        PyModuleDef_HEAD_INIT,                                            \
        #name, /* name of module */                                       \
        NULL,  /* module documentation, may be NULL */                    \
        -1,    /* size of per-interpreter state of the module,            \
                  or -1 if the module keeps state in global variables. */ \
        NULL,  /* methods */                                              \
        NULL,  /* slots */                                                \
        NULL,  /* traverse */                                             \
        NULL,  /* clear */                                                \
        NULL,  /* free */                                                 \
    };                                                                    \
    return PyModule_Create(&module_def);                                  \
  }                                                                       \
  }

FLASHINFER_EXT_MODULE_INIT_EXPAND(TORCH_EXTENSION_NAME)

#undef FLASHINFER_EXT_MODULE_INIT
#undef FLASHINFER_EXT_MODULE_INIT_EXPAND

#endif

#define _DISPATCH_CASE_I32(c_type, ...) \
  case at::ScalarType::Int: {           \
    using c_type = int32_t;             \
    return __VA_ARGS__();               \
  }

#define _DISPATCH_CASE_I64(c_type, ...) \
  case at::ScalarType::Long: {          \
    using c_type = int64_t;             \
    return __VA_ARGS__();               \
  }

#define DISPATCH_PYTORCH_IDTYPE_TO_CTYPE(pytorch_dtype, c_type, ...)                  \
  [&]() -> bool {                                                                     \
    switch (pytorch_dtype) {                                                          \
      _DISPATCH_CASE_I32(c_type, __VA_ARGS__)                                         \
      _DISPATCH_CASE_I64(c_type, __VA_ARGS__)                                         \
      default:                                                                        \
        std::ostringstream oss;                                                       \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch idtype " << pytorch_dtype; \
        TORCH_CHECK(false, oss.str());                                                \
        return false;                                                                 \
    }                                                                                 \
  }()

#define _DISPATCH_CASE_F32(c_type, ...) \
  case at::ScalarType::Float: {         \
    using c_type = float;               \
    return __VA_ARGS__();               \
  }

#ifdef FLASHINFER_ENABLE_F16
#define _DISPATCH_CASE_F16(c_type, ...) \
  case at::ScalarType::Half: {          \
    using c_type = nv_half;             \
    return __VA_ARGS__();               \
  }
#else
#define _DISPATCH_CASE_F16(c_type, ...)
#endif

#ifdef FLASHINFER_ENABLE_BF16
#define _DISPATCH_CASE_BF16(c_type, ...) \
  case at::ScalarType::BFloat16: {       \
    using c_type = nv_bfloat16;          \
    return __VA_ARGS__();                \
  }
#else
#define _DISPATCH_CASE_BF16(c_type, ...)
#endif

#ifdef FLASHINFER_ENABLE_FP8_E4M3
#define _DISPATCH_CASE_FP8_E4M3(c_type, ...) \
  case at::ScalarType::Float8_e4m3fn: {      \
    using c_type = __nv_fp8_e4m3;            \
    return __VA_ARGS__();                    \
  }
#else
#define _DISPATCH_CASE_FP8_E4M3(c_type, ...)
#endif

#ifdef FLASHINFER_ENABLE_FP8_E5M2
#define _DISPATCH_CASE_FP8_E5M2(c_type, ...) \
  case at::ScalarType::Float8_e5m2: {        \
    using c_type = __nv_fp8_e5m2;            \
    return __VA_ARGS__();                    \
  }
#else
#define _DISPATCH_CASE_FP8_E5M2(c_type, ...)
#endif

// Should not be used together with _DISPATCH_SF_CASE_FP8_E8M0
#ifdef FLASHINFER_ENABLE_FP4_E2M1
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
#define _DISPATCH_CASE_FP4_E2M1(c_type, ...) \
  case at::ScalarType::Byte: {               \
    using c_type = __nv_fp4_e2m1;            \
    return __VA_ARGS__();                    \
  }
#else
#define _DISPATCH_CASE_FP4_E2M1(c_type, ...)                               \
  case at::ScalarType::Byte: {                                             \
    static_assert(false, "FP4 E2M1 support requires CUDA 12.8 or newer."); \
    break;                                                                 \
  }
#endif
#else
#define _DISPATCH_CASE_FP4_E2M1(c_type, ...)
#endif

// Should not be used together with _DISPATCH_CASE_FP4_E2M1
#ifdef FLASHINFER_ENABLE_FP8_E8M0
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
#define _DISPATCH_SF_CASE_FP8_E8M0(c_type, ...) \
  case at::ScalarType::Byte: {                  \
    using c_type = __nv_fp8_e8m0;               \
    return __VA_ARGS__();                       \
  }
#else
#define _DISPATCH_SF_CASE_FP8_E8M0(c_type, ...)                            \
  case at::ScalarType::Byte: {                                             \
    static_assert(false, "FP8 E8M0 support requires CUDA 12.8 or newer."); \
    break;                                                                 \
  }
#endif
#else
#define _DISPATCH_SF_CASE_FP8_E8M0(c_type, ...)
#endif

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)                 \
  [&]() -> bool {                                                                        \
    switch (pytorch_dtype) {                                                             \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                            \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                           \
      default:                                                                           \
        std::ostringstream oss;                                                          \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype; \
        TORCH_CHECK(false, oss.str());                                                   \
        return false;                                                                    \
    }                                                                                    \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP8(pytorch_dtype, c_type, ...)                      \
  [&]() -> bool {                                                                            \
    switch (pytorch_dtype) {                                                                 \
      _DISPATCH_CASE_FP8_E4M3(c_type, __VA_ARGS__)                                           \
      _DISPATCH_CASE_FP8_E5M2(c_type, __VA_ARGS__)                                           \
      default:                                                                               \
        std::ostringstream oss;                                                              \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch fp8 data type " << pytorch_dtype; \
        TORCH_CHECK(false, oss.str());                                                       \
        return false;                                                                        \
    }                                                                                        \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_SF(pytorch_dtype, c_type, ...)                \
  [&]() -> bool {                                                                     \
    switch (pytorch_dtype) {                                                          \
      _DISPATCH_CASE_F32(c_type, __VA_ARGS__)                                         \
      _DISPATCH_SF_CASE_FP8_E8M0(c_type, __VA_ARGS__)                                 \
      default:                                                                        \
        std::ostringstream oss;                                                       \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch scaling factor data type " \
            << pytorch_dtype;                                                         \
        TORCH_CHECK(false, oss.str());                                                \
        return false;                                                                 \
    }                                                                                 \
  }()

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE(pytorch_dtype, c_type, ...)                      \
  [&]() -> bool {                                                                        \
    switch (pytorch_dtype) {                                                             \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                            \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                           \
      _DISPATCH_CASE_FP8_E4M3(c_type, __VA_ARGS__)                                       \
      _DISPATCH_CASE_FP8_E5M2(c_type, __VA_ARGS__)                                       \
      _DISPATCH_CASE_FP4_E2M1(c_type, __VA_ARGS__)                                       \
      default:                                                                           \
        std::ostringstream oss;                                                          \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch data type " << pytorch_dtype; \
        TORCH_CHECK(false, oss.str());                                                   \
        return false;                                                                    \
    }                                                                                    \
  }()

#define _DISPATCH_SWITCH(var_name, cond, ...)                                           \
  [&]() -> bool {                                                                       \
    switch (cond) {                                                                     \
      __VA_ARGS__                                                                       \
      default:                                                                          \
        std::ostringstream oss;                                                         \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch " var_name " " << int(cond); \
        TORCH_CHECK(false, oss.str());                                                  \
        return false;                                                                   \
    }                                                                                   \
  }()

#define _DISPATCH_SWITCH_U16x2(var1_name, var2_name, cond1, cond2, ...)                       \
  [&]() -> bool {                                                                             \
    switch (pack_u16(cond1, cond2)) {                                                         \
      __VA_ARGS__                                                                             \
      default:                                                                                \
        std::ostringstream oss;                                                               \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch (" var1_name ", " var2_name "): (" \
            << int(cond1) << ", " << int(cond2) << ")";                                       \
        TORCH_CHECK(false, oss.str());                                                        \
        return false;                                                                         \
    }                                                                                         \
  }()

#define _DISPATCH_CASE(case_expr, case_var, ...) \
  case case_expr: {                              \
    constexpr auto case_var = case_expr;         \
    return __VA_ARGS__();                        \
  }

#define _DISPATCH_CASE_U16x2(case_expr1, case_expr2, case_var1, case_var2, ...) \
  case pack_u16(case_expr1, case_expr2): {                                      \
    constexpr auto case_var1 = case_expr1;                                      \
    constexpr auto case_var2 = case_expr2;                                      \
    return __VA_ARGS__();                                                       \
  }

#define DISPATCH_BOOL(expr, const_expr, ...) \
  [&]() -> bool {                            \
    if (expr) {                              \
      constexpr bool const_expr = true;      \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr bool const_expr = false;     \
      return __VA_ARGS__();                  \
    }                                        \
  }()

inline void check_shape(const at::Tensor& a, const at::Tensor& b, const char* a_name,
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

#define CHECK_GQA_HEAD_DIVISIBLE(num_qo_heads, num_kv_heads)                   \
  TORCH_CHECK(num_qo_heads % num_kv_heads == 0, "num_qo_heads(", num_qo_heads, \
              ") must be divisible by num_kv_heads(", num_kv_heads, ")")

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_LAST_DIM_CONTIGUOUS(x) \
  TORCH_CHECK(x.strides()[x.strides().size() - 1] == 1, #x "must be contiguous at last dimension")

#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_TYPE(x, st) \
  TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_INPUT_AND_TYPE(x, st) \
  CHECK_CUDA(x);                    \
  CHECK_CONTIGUOUS(x);              \
  CHECK_INPUT_TYPE(x, st)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CUDA(x);                           \
  CHECK_LAST_DIM_CONTIGUOUS(x)

#define CHECK_DIM(d, x) TORCH_CHECK(x.dim() == d, #x " must be a " #d "D tensor")

#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)

#define CHECK_EQ(a, b) TORCH_CHECK((a) == (b), "CHECK_EQ(" #a ", " #b ") failed. ", a, " vs ", b)

#define CHECK_GE(a, b) TORCH_CHECK((a) >= (b), "CHECK_GE(" #a ", " #b ") failed. ", a, " vs ", b)

inline bool is_float8_tensor(const at::Tensor& tensor) {
  return tensor.scalar_type() == at::ScalarType::Float8_e4m3fn ||
         tensor.scalar_type() == at::ScalarType::Float8_e5m2;
}
