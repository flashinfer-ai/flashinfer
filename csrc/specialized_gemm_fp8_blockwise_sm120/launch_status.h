// Copyright (c) 2026 by FlashInfer team.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>
#include <tvm/ffi/tvm_ffi.h>

namespace sm120_launch {

inline void check(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "specialized sm120 fp8 gemm " << operation << " failed: " << cudaGetErrorString(status);
}

// Check immediately after <<<...>>>. Do not clear a pending error before launch:
// CUDA can report an earlier asynchronous fault here, and it must reach the caller.
inline void check_submission() { check(cudaGetLastError(), "launch"); }

template <typename Kernel, typename... Args>
inline void launch_ex(const cudaLaunchConfig_t* config, Kernel kernel, Args... args) {
  check(cudaLaunchKernelEx(config, kernel, args...), "cudaLaunchKernelEx");
}

}  // namespace sm120_launch
