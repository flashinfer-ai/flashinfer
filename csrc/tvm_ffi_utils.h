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
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#define _DISPATCH_CASE_F16(c_type, ...)  \
  case (kDLFloat << 16 | 16 << 8 | 1): { \
    using c_type = nv_half;              \
    return __VA_ARGS__();                \
  }
#define _DISPATCH_CASE_BF16(c_type, ...)  \
  case (kDLBfloat << 16 | 16 << 8 | 1): { \
    using c_type = nv_bfloat16;           \
    return __VA_ARGS__();                 \
  }

#define DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(dlpack_dtype, c_type, ...)                         \
  [&]() -> bool {                                                                              \
    switch (((dlpack_dtype).code << 16) | ((dlpack_dtype).bits << 8) | (dlpack_dtype).lanes) { \
      _DISPATCH_CASE_F16(c_type, __VA_ARGS__)                                                  \
      _DISPATCH_CASE_BF16(c_type, __VA_ARGS__)                                                 \
      default:                                                                                 \
        TVM_FFI_ICHECK(false) << __PRETTY_FUNCTION__ << " failed to dispatch data type "       \
                              << (dlpack_dtype).code << " " << (dlpack_dtype).bits;            \
        return false;                                                                          \
    }                                                                                          \
  }()
