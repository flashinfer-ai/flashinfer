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
#ifndef FLASHINFER_CP_ASYNC_CUH_
#define FLASHINFER_CP_ASYNC_CUH_

#include <hip/hip_runtime.h>

#include <cstdint>



namespace flashinfer::cp_async {

enum class SharedMemFillMode {
  kFillZero,  // Fill zero to shared memory when predicate is false
  kNoFill     // Do not fill zero to shared memory when predicate is false
};

enum class PrefetchMode {
  kNoPrefetch,  // Do not fetch additional data from global memory to L2
  kPrefetch     // Fetch additional data from global memory to L2
};


/// @brief orders memory accesses for all threads within a thread block.
__device__ __forceinline__ void commit_group() {
  __threadfence_block();
}


/// @brief Wrapper of PTX cp.async.wait_group instruction
/// @param n Wait till most recent n groups are committed
template <size_t n>
__device__ __forceinline__ void wait_group() {
  __syncthreads();
}


/// @brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from global memory to shared memory
/// @tparam prefetch_mode Whether to fetch additional data from global memory to L2
/// @tparam T Data type
/// @param smem_ptr Pointer to shared memory
/// @param gmem_ptr Pointer to global memory
template <PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
  *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
}

/// @brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from global memory to shared memory with predicate.
/// @tparam prefetch_mode Whether to fetch additional data from global memory to L2
/// @tparam fill_mode Whether to fill zero to shared memory when predicate is false
/// @tparam T Data type
/// @param smem_ptr Pointer to shared memory
/// @param gmem_ptr Pointer to global memory
/// @param predicate Predicate value
/// @note fill zero is slower than not fill zero
template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  if (predicate) {
    *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
  } else {
    if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
      *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
}

/// @brief Load specified number of bits per thread from global memory to shared memory
/// @tparam num_bits Number of bits to load, must be 128 or 256
/// @tparam prefetch_mode Whether to fetch additional data from global memory to L2
/// @tparam T Data type
/// @param smem_ptr Pointer to shared memory
/// @param gmem_ptr Pointer to global memory
template <size_t num_bits, PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load(T* smem_ptr, const T* gmem_ptr) {
  static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
  } else {
    load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
    load_128b<prefetch_mode>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T));
  }
}

/// @brief Load specified number of bits per thread from global memory to shared memory with predicate
/// @tparam num_bits Number of bits to load, must be 128 or 256
/// @tparam prefetch_mode Whether to fetch additional data from global memory to L2
/// @tparam fill_mode Whether to fill zero to shared memory when predicate is false
/// @tparam T Data type
/// @param smem_ptr Pointer to shared memory
/// @param gmem_ptr Pointer to global memory
/// @param predicate Predicate value
/// @note fill zero is slower than not fill zero
template <size_t num_bits, PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  // static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
  } else {
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T),
                                             predicate);
  }
}

}  // namespace flashinfer::cp_async

#endif  // FLASHINFER_CP_ASYNC_CUH_
