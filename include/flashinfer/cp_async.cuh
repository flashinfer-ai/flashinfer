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
#ifndef FLASHINFER_CP_ASYNC_CUH_
#define FLASHINFER_CP_ASYNC_CUH_

#include <cuda_runtime.h>

namespace flashinfer {

namespace cp_async {

__device__ __forceinline__ void commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && \
    (__CUDACC_VER_MAJOR__ >= 11)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <size_t n>
__device__ __forceinline__ void wait_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && \
    (__CUDACC_VER_MAJOR__ >= 11)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

template <bool prefetch, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && \
    (__CUDACC_VER_MAJOR__ >= 11)

  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (prefetch) {
    asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
            smem_int_ptr),
        "l"(gmem_ptr), "n"(16), "r"(16));
  } else {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
        "l"(gmem_ptr), "n"(16), "r"(16));
  }
#else
  *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
#endif
}

template <bool prefetch, bool fill_zero, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr,
                                               bool predicate) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && \
    (__CUDACC_VER_MAJOR__ >= 11)

  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_zero) {
    int src_in_bytes = predicate ? 16 : 0;
    if constexpr (prefetch) {
      asm volatile(
          "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
              smem_int_ptr),
          "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    } else {
      asm volatile(
          "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
          "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    }
  } else {
    if constexpr (prefetch) {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    } else {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    }
  }
#else
  if (predicate) {
    *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
  } else {
    if constexpr (fill_zero) {
      *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
#endif
}

template <size_t num_bits, bool prefetch, typename T>
__device__ __forceinline__ void load(T* smem_ptr, const T* gmem_ptr) {
  static_assert(num_bits == 128 || num_bits == 256,
                "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    load_128b<prefetch>(smem_ptr, gmem_ptr);
  } else {
    load_128b<prefetch>(smem_ptr, gmem_ptr);
    load_128b<prefetch>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T));
  }
}

template <size_t num_bits, bool prefetch, typename T>
__device__ __forceinline__ void pred_load(T* smem_ptr, const T* gmem_ptr,
                                          bool predicate) {
  static_assert(num_bits == 128 || num_bits == 256,
                "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    pred_load_128b<prefetch>(smem_ptr, gmem_ptr, predicate);
  } else {
    pred_load_128b<prefetch>(smem_ptr, gmem_ptr, predicate);
    pred_load_128b<prefetch>(smem_ptr + 16 / sizeof(T),
                             gmem_ptr + 16 / sizeof(T), predicate);
  }
}

}  // namespace cp_async

}  // namespace flashinfer

#endif  // FLASHINFER_CP_ASYNC_CUH_
