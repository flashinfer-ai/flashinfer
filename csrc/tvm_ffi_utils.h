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
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include "dlpack/dlpack.h"

inline constexpr int64_t encode_dlpack_dtype(DLDataType dtype) {
  return (dtype.code << 16) | (dtype.bits << 8) | dtype.lanes;
}

constexpr int64_t float16_code = encode_dlpack_dtype(DLDataType{kDLFloat, 16, 1});
constexpr int64_t bfloat16_code = encode_dlpack_dtype(DLDataType{kDLBfloat, 16, 1});
constexpr int64_t float32_code = encode_dlpack_dtype(DLDataType{kDLFloat, 32, 1});
constexpr int64_t int32_code = encode_dlpack_dtype(DLDataType{kDLInt, 32, 1});
constexpr int64_t int64_code = encode_dlpack_dtype(DLDataType{kDLInt, 64, 1});
constexpr int64_t float8e4m3fn_code = encode_dlpack_dtype(DLDataType{kDLFloat8_e4m3fn, 32, 1});
constexpr int64_t float8e5m2_code = encode_dlpack_dtype(DLDataType{kDLFloat8_e5m2, 64, 1});

#ifdef FLASHINFER_ENABLE_F16
#define _DISPATCH_CASE_F16(c_type, ...) \
  case float16_code: {                  \
    using c_type = nv_half;             \
    return __VA_ARGS__();               \
  }
#else
#define _DISPATCH_CASE_F16(c_type, ...)
#endif

#ifdef FLASHINFER_ENABLE_BF16
#define _DISPATCH_CASE_BF16(c_type, ...) \
  case bfloat16_code: {                  \
    using c_type = nv_bfloat16;          \
    return __VA_ARGS__();                \
  }
#else
#define _DISPATCH_CASE_BF16(c_type, ...)
#endif

#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dlpack_dtype, c_type, ...)                   \
  [&]() -> bool {                                                                        \
    switch (encode_dlpack_dtype(dlpack_dtype)) {                                         \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                            \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                           \
      default:                                                                           \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch data type " \
                              << (dlpack_dtype).code << " " << (dlpack_dtype).bits;      \
        return false;                                                                    \
    }                                                                                    \
  }()

#define _DISPATCH_CASE_I32(c_type, ...) \
  case int32_code: {                    \
    using c_type = int32_t;             \
    return __VA_ARGS__();               \
  }

#define _DISPATCH_CASE_I64(c_type, ...) \
  case int64_code: {                    \
    using c_type = int64_t;             \
    return __VA_ARGS__();               \
  }

#define DISPATCH_DLPACK_IDTYPE_TO_CTYPE(dlpack_dtype, c_type, ...)                       \
  [&]() -> bool {                                                                        \
    switch (encode_dlpack_dtype(dlpack_dtype)) {                                         \
      _DISPATCH_CASE_I32(c_type, __VA_ARGS__)                                            \
      _DISPATCH_CASE_I64(c_type, __VA_ARGS__)                                            \
      default:                                                                           \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch data type " \
                              << (dlpack_dtype).code << " " << (dlpack_dtype).bits;      \
        return false;                                                                    \
    }                                                                                    \
  }()

inline void check_shape(const tvm::ffi::Tensor& a, const tvm::ffi::Tensor& b, const char* a_name,
                        const char* b_name) {
  TVM_FFI_ICHECK_EQ(a->ndim, b->ndim) << a_name << "->ndim and " << b_name << "->ndim mismatch";
  for (int i = 0; i < a->ndim; ++i) {
    TVM_FFI_ICHECK_EQ(a->shape[i], b->shape[i])
        << a_name << "->shape[" << i << "] and " << b_name << "->shape[" << i << "] mismatch";
  }
}

#define CHECK_CUDA(x) \
  TVM_FFI_ICHECK_EQ(x->device.device_type, kDLCUDA) << #x " must be a CUDA tensor";
#define CHECK_CONTIGUOUS(x) TVM_FFI_ICHECK(x.IsContiguous()) << #x " must be contiguous";
#define CHECK_LAST_DIM_CONTIGUOUS(x)            \
  TVM_FFI_ICHECK_EQ(x->strides[x->ndim - 1], 1) \
  #x "must be contiguous at last dimension";
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_TYPE(x, st) TORCH_CHECK(x->type == st) << "Inconsistency of Tensor type: " #x;
#define CHECK_INPUT_AND_TYPE(x, st) \
  CHECK_CUDA(x);                    \
  CHECK_CONTIGUOUS(x);              \
  CHECK_INPUT_TYPE(x, st)
#define CHECK_LAST_DIM_CONTIGUOUS_INPUT(x) \
  CHECK_CUDA(x);                           \
  CHECK_LAST_DIM_CONTIGUOUS(x)
#define CHECK_DIM(d, x) TVM_FFI_ICHECK_EQ(x->ndim, d) << #x " must be a " #d "D tensor";
#define CHECK_SHAPE(a, b) check_shape(a, b, #a, #b)
#define CHECK_DEVICE(a, b)                                         \
  TVM_FFI_ICHECK_EQ(a->device.device_type, b->device.device_type); \
  TVM_FFI_ICHECK_EQ(a->device.device_id, b->device.device_id);
