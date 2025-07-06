/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

// #include "tensorrt_llm/common/cudaBf16Wrapper.h"
// #include "tensorrt_llm/common/cudaDriverWrapper.h"
// #include "tensorrt_llm/common/cudaFp8Utils.h"
// #include "tensorrt_llm/common/logger.h"
// #include "tensorrt_llm/common/tllmException.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_types.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
// #ifndef _WIN32 // Linux
// #include <sys/sysinfo.h>
// #endif         // not WIN32
// #include <vector>
// #ifdef _WIN32  // Windows
// #include <windows.h>
// #undef ERROR   // A Windows header file defines ERROR as 0, but it's used in our logger.h enum.
// Logging breaks without
//                // this undef.
// #endif         // WIN32

namespace tensorrt_llm::common {

// // workspace for cublas gemm : 32MB
// #define CUBLAS_WORKSPACE_SIZE 33554432

// typedef struct __align__(4)
// {
//     half x, y, z, w;
// }

// half4;

// /* **************************** type definition ***************************** */

// enum CublasDataType
// {
//     FLOAT_DATATYPE = 0,
//     HALF_DATATYPE = 1,
//     BFLOAT16_DATATYPE = 2,
//     INT8_DATATYPE = 3,
//     FP8_DATATYPE = 4
// };

// enum TRTLLMCudaDataType
// {
//     FP32 = 0,
//     FP16 = 1,
//     BF16 = 2,
//     INT8 = 3,
//     FP8 = 4
// };

// enum class OperationType
// {
//     FP32,
//     FP16,
//     BF16,
//     INT8,
//     FP8
// };

/* **************************** debug tools ********************************* */

inline std::optional<bool> isCudaLaunchBlocking() {
  thread_local bool firstCall = true;
  thread_local std::optional<bool> result = std::nullopt;
  if (!firstCall) {
    char const* env = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (env != nullptr && std::string(env) == "1") {
      result = true;
    } else {
      result = false;
    }
    firstCall = false;
  }
  return result;
}

inline std::optional<bool> isCapturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status;
  TORCH_CHECK(cudaStreamIsCapturing(stream, &status) == cudaSuccess,
              "CUDA error in cudaStreamIsCapturing");
  return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive;
}

inline bool doCheckError(cudaStream_t stream) {
  auto const cudaLaunchBlocking = isCudaLaunchBlocking();
  if (cudaLaunchBlocking.has_value() && cudaLaunchBlocking.value()) {
    return !isCapturing(stream);
  }

#ifndef NDEBUG
  // Debug builds will sync when we're not capturing unless explicitly
  // disabled.
  bool const checkError = cudaLaunchBlocking.value_or(!isCapturing(stream));
#else
  bool const checkError = cudaLaunchBlocking.value_or(false);
#endif

  return checkError;
}

inline void syncAndCheck(cudaStream_t stream, char const* const file, int const line) {
  if (doCheckError(stream)) {
    cudaStreamSynchronize(stream);
    auto error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, "CUDA error in %s: %s", file, cudaGetErrorString(error));
  }
}

#define sync_check_cuda_error(stream) tensorrt_llm::common::syncAndCheck(stream, __FILE__, __LINE__)

template <typename T1, typename T2>
inline size_t divUp(T1 const& a, T2 const& b) {
  auto const tmp_a = static_cast<size_t>(a);
  auto const tmp_b = static_cast<size_t>(b);
  return (tmp_a + tmp_b - 1) / tmp_b;
}

inline int roundUp(int a, int b) { return divUp(a, b) * b; }

template <typename T, typename U, typename = std::enable_if_t<std::is_integral<T>::value>,
          typename = std::enable_if_t<std::is_integral<U>::value>>
auto constexpr ceilDiv(T numerator, U denominator) {
  return (numerator + denominator - 1) / denominator;
}

}  // namespace tensorrt_llm::common
