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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <flashinfer/exception.h>

#include <sstream>

using namespace flashinfer;

enum class TypeCode {
  kFloat64 = 0,
  kFloat32 = 1,
  kFloat16 = 2,
  kBFloat16 = 3,
  kFloat8_e4m3fn = 4,
  kFloat8_e5m2 = 5,
  kInt64 = 100,
  kUInt64 = 101,
  kInt32 = 102,
  kUInt32 = 103,
  kInt16 = 104,
  kUInt16 = 105,
  kInt8 = 106,
  kUInt8 = 107,
};

#ifdef FLASHINFER_ENABLE_BF16
#define DISPATCH_TYPE_CODE_TO_CTYPE_FP16(type_code, c_type, ...)                     \
  [&]() -> bool {                                                                    \
    switch (TypeCode(type_code)) {                                                   \
      case TypeCode::kFloat16: {                                                     \
        using c_type = nv_half;                                                      \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      case TypeCode::kBFloat16: {                                                    \
        using c_type = nv_bfloat16;                                                  \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      default:                                                                       \
        std::ostringstream oss;                                                      \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch type code " << type_code; \
        FLASHINFER_ERROR(oss.str());                                                 \
        return false;                                                                \
    }                                                                                \
  }()
#else
#define DISPATCH_TYPE_CODE_TO_CTYPE_FP16(type_code, c_type, ...)                     \
  [&]() -> bool {                                                                    \
    switch (TypeCode(type_code)) {                                                   \
      case TypeCode::kFloat16: {                                                     \
        using c_type = nv_half;                                                      \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      default:                                                                       \
        std::ostringstream oss;                                                      \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch type code " << type_code; \
        FLASHINFER_ERROR(oss.str());                                                 \
        return false;                                                                \
    }                                                                                \
  }()
#endif

#ifdef FLASHINFER_ENABLE_FP8
#define DISPATCH_TYPE_CODE_TO_CTYPE_FP8(type_code, c_type, ...)                      \
  [&]() -> bool {                                                                    \
    switch (TypeCode(type_code)) {                                                   \
      case TypeCode::kFloat8_e4m3fn: {                                               \
        using c_type = __nv_fp8_e4m3;                                                \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      case TypeCode::kFloat8_e5m2: {                                                 \
        using c_type = __nv_fp8_e5m2;                                                \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      default:                                                                       \
        std::ostringstream oss;                                                      \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch type code " << type_code; \
        FLASHINFER_ERROR(oss.str());                                                 \
        return false;                                                                \
    }                                                                                \
  }()
#else
#define DISPATCH_TYPE_CODE_TO_CTYPE_FP8(type_code, c_type, ...)                  \
  [&]() -> bool {                                                                \
    std::ostringstream oss;                                                      \
    oss << __PRETTY_FUNCTION__ << " failed to dispatch type code " << type_code; \
    FLASHINFER_ERROR(oss.str());                                                 \
    return false;                                                                \
  }()
#endif

#if defined(FLASHINFER_ENABLE_BF16) && defined(FLASHINFER_ENABLE_FP8)
#define DISPATCH_TYPE_CODE_TO_CTYPE(type_code, c_type, ...)                          \
  [&]() -> bool {                                                                    \
    switch (TypeCode(type_code)) {                                                   \
      case TypeCode::kFloat16: {                                                     \
        using c_type = nv_half;                                                      \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      case TypeCode::kBFloat16: {                                                    \
        using c_type = nv_bfloat16;                                                  \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      case TypeCode::kFloat8_e4m3fn: {                                               \
        using c_type = __nv_fp8_e4m3;                                                \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      case TypeCode::kFloat8_e5m2: {                                                 \
        using c_type = __nv_fp8_e5m2;                                                \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      default:                                                                       \
        std::ostringstream oss;                                                      \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch type code " << type_code; \
        FLASHINFER_ERROR(oss.str());                                                 \
        return false;                                                                \
    }                                                                                \
  }()
#elif defined(FLASHINFER_ENABLE_BF16)
#define DISPATCH_TYPE_CODE_TO_CTYPE(type_code, c_type, ...)                          \
  [&]() -> bool {                                                                    \
    switch (TypeCode(type_code)) {                                                   \
      case TypeCode::kFloat16: {                                                     \
        using c_type = nv_half;                                                      \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      case TypeCode::kBFloat16: {                                                    \
        using c_type = nv_bfloat16;                                                  \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      default:                                                                       \
        std::ostringstream oss;                                                      \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch type code " << type_code; \
        FLASHINFER_ERROR(oss.str());                                                 \
        return false;                                                                \
    }                                                                                \
  }()
#elif defined(FLASHINFER_ENABLE_FP8)
#define DISPATCH_TYPE_CODE_TO_CTYPE(type_code, c_type, ...)                          \
  [&]() -> bool {                                                                    \
    switch (TypeCode(type_code)) {                                                   \
      case TypeCode::kFloat16: {                                                     \
        using c_type = nv_half;                                                      \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      case TypeCode::kFloat8_e4m3fn: {                                               \
        using c_type = __nv_fp8_e4m3;                                                \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      case TypeCode::kFloat8_e5m2: {                                                 \
        using c_type = __nv_fp8_e5m2;                                                \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      default:                                                                       \
        std::ostringstream oss;                                                      \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch type code " << type_code; \
        FLASHINFER_ERROR(oss.str());                                                 \
        return false;                                                                \
    }                                                                                \
  }()
#else
#define DISPATCH_TYPE_CODE_TO_CTYPE(type_code, c_type, ...)                          \
  [&]() -> bool {                                                                    \
    switch (TypeCode(type_code)) {                                                   \
      case TypeCode::kFloat16: {                                                     \
        using c_type = nv_half;                                                      \
        return __VA_ARGS__();                                                        \
      }                                                                              \
      default:                                                                       \
        std::ostringstream oss;                                                      \
        oss << __PRETTY_FUNCTION__ << " failed to dispatch type code " << type_code; \
        FLASHINFER_ERROR(oss.str());                                                 \
        return false;                                                                \
    }                                                                                \
  }()
#endif
