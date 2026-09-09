/*
 * Copyright 2025-2026 NVIDIA
 * Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
 * Copyright (c) 2026 by FlashInfer team.
 * Modifications Copyright (c) 2026 by the pcie_collectives contributors.
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
 *
 * The system-scope release/acquire primitives follow FlashInfer's Apache-2.0
 * pcie_ipc_all_reduce implementation. The device-selected double-buffer slot
 * protocol is derived from b12x's Apache-2.0 PCIe two-shot implementation:
 *   https://github.com/flashinfer-ai/flashinfer/pull/4393
 *   https://github.com/local-inference-lab/b12x
 */
#ifndef FLASHINFER_COMM_PCIE_IPC_COMMON_CUH_
#define FLASHINFER_COMM_PCIE_IPC_COMMON_CUH_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace flashinfer {
namespace comm {
namespace pcie_ipc {
namespace common {

constexpr int kMaxWorldSize = 8;
constexpr int kMaxBlocks = 64;
constexpr int kMaxThreads = 512;
constexpr int kFlagStrideWords = 32;  // one live word per 128-byte record
constexpr size_t kPackBytes = sizeof(uint4);
constexpr size_t kSignalAlignment = 128;

inline bool valid_element_size(int element_size) {
  return element_size > 0 && kPackBytes % static_cast<size_t>(element_size) == 0;
}

__host__ __device__ __forceinline__ constexpr size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Host-side workspace layout helpers. Keep overflow handling framework-free so
// every binding can translate the same boolean result into its own error type.
inline bool checked_add_size(size_t left, size_t right, size_t* result) {
  if (result == nullptr || left > std::numeric_limits<size_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

inline bool checked_mul_size(size_t left, size_t right, size_t* result) {
  if (result == nullptr || (right != 0 && left > std::numeric_limits<size_t>::max() / right)) {
    return false;
  }
  *result = left * right;
  return true;
}

inline bool checked_align_up_size(size_t value, size_t alignment, size_t* result) {
  if (result == nullptr || alignment == 0) {
    return false;
  }
  const size_t remainder = value % alignment;
  if (remainder == 0) {
    *result = value;
    return true;
  }
  return checked_add_size(value, alignment - remainder, result);
}

__device__ __forceinline__ uint32_t load_global_u32(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ void store_global_u32(uint32_t* address, uint32_t value) {
  asm volatile("st.global.u32 [%0], %1;" : : "l"(address), "r"(value));
}

__device__ __forceinline__ uint32_t load_relaxed_gpu_u32(const uint32_t* address) {
  uint32_t value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(value) : "l"(address));
#endif
  return value;
}

__device__ __forceinline__ uint32_t load_acquire_system_u32(const uint32_t* address) {
  uint32_t value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(address));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(value) : "l"(address));
#endif
  return value;
}

__device__ __forceinline__ void store_release_system_u32(uint32_t* address, uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%0], %1;" : : "l"(address), "r"(value));
#else
  asm volatile("membar.sys; st.volatile.global.u32 [%0], %1;" : : "l"(address), "r"(value));
#endif
}

// True while `observed` is behind `expected` on a wrapping uint32_t epoch.
// This is valid as long as ranks never differ by 2^31 collective calls.
__host__ __device__ __forceinline__ constexpr bool generation_pending(uint32_t observed,
                                                                      uint32_t expected) {
  return static_cast<int32_t>(observed - expected) < 0;
}

static_assert(!generation_pending(0, UINT32_MAX), "zero follows UINT32_MAX");
static_assert(generation_pending(UINT32_MAX, 0), "UINT32_MAX precedes zero");

__device__ __forceinline__ void fence_system() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("fence.sc.sys;");
#else
  asm volatile("membar.sys;");
#endif
}

__device__ __forceinline__ uint4 load_global_nc_u4(const uint4* address) {
  uint4 value;
  asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address));
  return value;
}

__device__ __forceinline__ uint4 load_global_volatile_u4(const uint4* address) {
  uint4 value;
  asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(address));
  return value;
}

__device__ __forceinline__ void store_global_volatile_u4(uint4* address, uint4 value) {
  asm volatile("st.volatile.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "l"(address));
}

__device__ __forceinline__ void store_global_u4(uint4* address, uint4 value) {
  asm volatile("st.global.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "l"(address));
}

// Retire one call-level staging generation after every CTA has selected its slot.
// The counter uses atomicInc so changing gridDim between calls is safe.
__device__ __forceinline__ void retire_call_epoch(uint32_t* self_signal, size_t call_epoch_word,
                                                  size_t call_blocks_arrived_word) {
  if (threadIdx.x == blockDim.x - 1) {
    uint32_t* arrived = self_signal + call_blocks_arrived_word;
    const uint32_t last = static_cast<uint32_t>(gridDim.x - 1);
    const uint32_t prior = atomicInc(arrived, last);
    if (prior == last) {
      uint32_t* call_epoch = self_signal + call_epoch_word;
      store_global_u32(call_epoch, load_relaxed_gpu_u32(call_epoch) + 1U);
    }
  }
}

__device__ __forceinline__ uint32_t select_staging_slot(uint32_t* self_signal,
                                                        size_t call_epoch_word,
                                                        size_t call_blocks_arrived_word) {
  __shared__ uint32_t slot;
  if (threadIdx.x == 0) {
    slot = load_relaxed_gpu_u32(self_signal + call_epoch_word) & 1U;
  }
  __syncthreads();
  retire_call_epoch(self_signal, call_epoch_word, call_blocks_arrived_word);
  return slot;
}

}  // namespace common
}  // namespace pcie_ipc
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_PCIE_IPC_COMMON_CUH_
