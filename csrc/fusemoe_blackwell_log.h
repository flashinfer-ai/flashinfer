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
// Shared logging + cudaMalloc helpers for FuseMoE Blackwell.
//
// Progress messages (e.g. "[tcgen05] TMA: A=[…]") are routed through
// FUSEMOE_LOG_INFO() and only print when FUSEMOE_VERBOSE=1 is in the
// environment. Errors print unconditionally via FUSEMOE_LOG_ERR().
//
// FUSEMOE_CUDA_MALLOC wraps cudaMalloc, prints a diagnostic on failure,
// nulls out the destination pointer, and returns the cudaError_t so
// callers can short-circuit.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace flashinfer_fusemoe_blackwell {

inline bool verbose() {
  static const bool v = [] {
    const char* e = std::getenv("FUSEMOE_VERBOSE");
    return e != nullptr && e[0] != '\0' && e[0] != '0';
  }();
  return v;
}

inline cudaError_t cuda_malloc_checked(void** ptr, size_t bytes, const char* what, const char* file,
                                       int line) {
  cudaError_t e = cudaMalloc(ptr, bytes);
  if (e != cudaSuccess) {
    fprintf(stderr, "[fusemoe_blackwell] cudaMalloc(%s, %zu bytes) at %s:%d failed: %s\n",
            what ? what : "?", bytes, file, line, cudaGetErrorString(e));
    *ptr = nullptr;
  }
  return e;
}

}  // namespace flashinfer_fusemoe_blackwell

// Require callers to supply an explicit format string first; this lets
// -Wformat / -Wformat-security catch CWE-134 misuse at call sites and
// keeps the zero-extra-args case working via the GNU ##__VA_ARGS__ token.
#define FUSEMOE_LOG_INFO(fmt, ...)                 \
  do {                                             \
    if (flashinfer_fusemoe_blackwell::verbose()) { \
      fprintf(stderr, fmt, ##__VA_ARGS__);         \
    }                                              \
  } while (0)

#define FUSEMOE_LOG_ERR(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)

#define FUSEMOE_CUDA_MALLOC(ptr, bytes)                                                        \
  flashinfer_fusemoe_blackwell::cuda_malloc_checked(reinterpret_cast<void**>(&(ptr)), (bytes), \
                                                    #ptr, __FILE__, __LINE__)
