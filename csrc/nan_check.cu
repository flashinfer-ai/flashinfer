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
#include "nan_check.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>

namespace flashinfer {
namespace nan_check {

bool IsEnabled() {
  static const bool enabled = []() {
    const char* env = std::getenv("FLASHINFER_NAN_CHECK");
    return env != nullptr && (std::string(env) == "1" || std::string(env) == "true");
  }();
  return enabled;
}

bool ShouldTrap() {
  static const bool trap = []() {
    const char* env = std::getenv("FLASHINFER_NAN_CHECK_TRAP");
    return env != nullptr && (std::string(env) == "1" || std::string(env) == "true");
  }();
  return trap;
}

static int* GetFlagBuffer(cudaStream_t stream) {
  static int* d_flag = nullptr;
  if (d_flag == nullptr) {
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemsetAsync(d_flag, 0, sizeof(int), stream);
  }
  return d_flag;
}

static const char* GetDeviceLabel(const char* host_label, cudaStream_t stream) {
  static std::unordered_map<std::string, char*> label_cache;
  std::string key(host_label);
  auto it = label_cache.find(key);
  if (it != label_cache.end()) {
    return it->second;
  }
  size_t len = key.size() + 1;
  char* d_label = nullptr;
  cudaMalloc(&d_label, len);
  cudaMemcpyAsync(d_label, host_label, len, cudaMemcpyHostToDevice, stream);
  label_cache[key] = d_label;
  return d_label;
}

constexpr int kBlockSize = 256;

template <typename T>
__device__ __forceinline__ bool is_nan_val(T val) {
  return isnan(static_cast<float>(val));
}

template <>
__device__ __forceinline__ bool is_nan_val<half>(half val) {
  return __hisnan(val);
}

template <>
__device__ __forceinline__ bool is_nan_val<__nv_bfloat16>(__nv_bfloat16 val) {
  return __hisnan(val);
}

template <>
__device__ __forceinline__ bool is_nan_val<uint8_t>(uint8_t val) {
  return (val & 0x7F) == 0x7F;
}

template <typename T>
__global__ void nan_check_kernel(const T* __restrict__ data, int64_t size, const char* label,
                                 int* flag, bool do_trap) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size && is_nan_val(data[idx])) {
    int prev = atomicExch(flag, 1);
    if (prev == 0) {
      printf("[FLASHINFER_NAN_CHECK] NaN detected in \"%s\" at index %lld (of %lld)\n", label,
             static_cast<long long>(idx), static_cast<long long>(size));
    }
    if (do_trap) {
      __trap();
    }
  }
}

template <typename T>
static void LaunchNanCheckTyped(const void* data, int64_t numel, const char* label,
                                cudaStream_t stream) {
  if (!IsEnabled() || numel <= 0) return;
  int* d_flag = GetFlagBuffer(stream);
  cudaMemsetAsync(d_flag, 0, sizeof(int), stream);
  const char* d_label = GetDeviceLabel(label, stream);
  bool do_trap = ShouldTrap();
  int grid = static_cast<int>((numel + kBlockSize - 1) / kBlockSize);
  nan_check_kernel<T><<<grid, kBlockSize, 0, stream>>>(static_cast<const T*>(data), numel,
                                                        d_label, d_flag, do_trap);
}

void LaunchNanCheckFloat(const void* data, int64_t numel, const char* label, cudaStream_t stream) {
  LaunchNanCheckTyped<float>(data, numel, label, stream);
}

void LaunchNanCheckHalf(const void* data, int64_t numel, const char* label, cudaStream_t stream) {
  LaunchNanCheckTyped<half>(data, numel, label, stream);
}

void LaunchNanCheckBFloat16(const void* data, int64_t numel, const char* label,
                            cudaStream_t stream) {
  LaunchNanCheckTyped<__nv_bfloat16>(data, numel, label, stream);
}

void LaunchNanCheckFp8Bytes(const void* data, int64_t num_bytes, const char* label,
                            cudaStream_t stream) {
  LaunchNanCheckTyped<uint8_t>(data, num_bytes, label, stream);
}

}  // namespace nan_check
}  // namespace flashinfer
