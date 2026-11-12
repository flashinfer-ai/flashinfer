/*
 * Copyright (c) 2026 by FlashInfer team.
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

// Shim for MOSCA translation units built on the vendored deep_gemm tree. Include it before the
// MOSCA kernel header and any vendored `deep_gemm/` header.
//
// The vendored layout and scheduler helpers are `__host__ __device__` and
// check their arguments with `DG_UNIFIED_ASSERT`, which the vendored
// `common/exception.cuh` maps to `DG_DEVICE_ASSERT` (printf + PTX `trap`)
// whenever nvcc compiles the file, host pass included; upstream's host build
// never compiles them with nvcc. MOSCA host code (the MegaMoE planner: buffer
// layout, ring sizing) calls them on the host, so the macro is defined here
// first: in device passes it expands to exactly the vendored default, leaving
// device code unchanged; in the host pass it raises a Python-visible error.

#pragma once

#include "tvm_ffi_utils.h"

[[noreturn]] inline void mosca_deep_gemm_host_assert_fail(const char* cond, const char* file,
                                                          int line) {
  TVM_FFI_THROW(RuntimeError) << "MOSCA host check failed (vendored deep_gemm): " << cond << " ("
                              << file << ":" << line << ")";
  __builtin_unreachable();
}

#ifndef DG_UNIFIED_ASSERT
#if defined(__CUDA_ARCH__)
#define DG_UNIFIED_ASSERT(cond) DG_DEVICE_ASSERT(cond)
#else
#define DG_UNIFIED_ASSERT(cond)                                               \
  do {                                                                        \
    if (!(cond)) mosca_deep_gemm_host_assert_fail(#cond, __FILE__, __LINE__); \
  } while (0)
#endif
#endif
