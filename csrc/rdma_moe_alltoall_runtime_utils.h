/*
 * Derived from DeepEP; SM120 adaptation starting-point revision:
 * be2582ffe69b5e7d61c3bc9bf7a5316bc48261f9 (fork revision).
 * Copyright (c) 2025 DeepSeek. SPDX-License-Identifier: MIT.
 * See LICENSE.deepep.txt and include/flashinfer/comm/rdma_moe_alltoall/VENDOR.md.
 */
#pragma once

#include <cuda_runtime.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tvm_ffi_utils.h"

namespace deep_ep {
using tvm::ffi::Any;
using tvm::ffi::Array;
using tvm::ffi::Map;
using tvm::ffi::Optional;
using tvm::ffi::String;
using Info = Map<String, Any>;

template <typename... Args>
void check(bool condition, Args&&... args) {
  if (!condition) {
    std::ostringstream message;
    (message << ... << std::forward<Args>(args));
    TVM_FFI_THROW(ValueError) << message.str();
  }
}
#define LL_CHECK(condition, ...) ::deep_ep::check(bool(condition), __VA_ARGS__)
#define LL_CUDA(call)                                                         \
  do {                                                                        \
    auto status = (call);                                                     \
    LL_CHECK(status == cudaSuccess, #call, ": ", cudaGetErrorString(status)); \
  } while (0)

template <typename T>
T* pointer(TensorView value) {
  return static_cast<T*>(value.data_ptr());
}
inline bool same_shape(TensorView a, TensorView b) {
  if (a.ndim() != b.ndim()) return false;
  for (int i = 0; i < a.ndim(); ++i)
    if (a.size(i) != b.size(i)) return false;
  return true;
}
inline bool overlaps(TensorView a, TensorView b) {
  if (!a.numel() || !b.numel()) return false;
  const auto begin_a = reinterpret_cast<uintptr_t>(a.data_ptr());
  const auto begin_b = reinterpret_cast<uintptr_t>(b.data_ptr());
  const size_t bytes_a = size_t(a.numel()) * get_element_size(a);
  const size_t bytes_b = size_t(b.numel()) * get_element_size(b);
  return begin_a < begin_b + bytes_b && begin_b < begin_a + bytes_a;
}
inline void check_tensor(TensorView tensor, const char* name, DLDataType dtype, int ndim,
                         int device) {
  LL_CHECK(tensor.device().device_type == kDLCUDA && tensor.device().device_id == device, name,
           " must be on this buffer's CUDA device");
  // Empty BF16 scales and zero-token inputs have no elements to address.
  // Torch's empty-tensor strides need not satisfy TVM-FFI's contiguous test.
  LL_CHECK((!tensor.numel() || tensor.IsContiguous()) && tensor.ndim() == ndim &&
               encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype),
           name, " has an unsupported shape, stride, or dtype");
  LL_CHECK(!tensor.numel() ||
               reinterpret_cast<uintptr_t>(tensor.data_ptr()) % get_element_size(tensor) == 0,
           name, " must have an element-aligned data pointer");
}
inline void check_vector_alignment(TensorView tensor, const char* name) {
  LL_CHECK(!tensor.numel() || reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, name,
           " must have a 16-byte aligned data pointer");
}
inline void check_phases(int phases) {
  LL_CHECK(phases == 1 || phases == 2 || phases == 3,
           "phases must be send(1), receive(2), or both(3)");
}
inline void check_dispatch_overlap(const std::vector<TensorView>& outputs,
                                   const std::vector<TensorView>& inputs) {
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (size_t j = i + 1; j < outputs.size(); ++j)
      LL_CHECK(!overlaps(outputs[i], outputs[j]), "dispatch outputs must not overlap");
    for (auto input : inputs)
      LL_CHECK(!overlaps(outputs[i], input), "dispatch outputs must not overlap inputs");
  }
}
inline Any optional_int(int value) { return value < 0 ? Any(nullptr) : Any(int64_t(value)); }
inline Array<int64_t> encode_bytes(const void* pointer, size_t bytes) {
  const auto* data = static_cast<const uint8_t*>(pointer);
  Array<int64_t> result;
  for (size_t i = 0; i < bytes; ++i) result.push_back(int64_t(data[i]));
  return result;
}
inline std::string decode_bytes(const Array<int64_t>& input, size_t bytes) {
  LL_CHECK(input.size() == bytes, "invalid transport handle byte length");
  std::string result(bytes, '\0');
  for (size_t i = 0; i < bytes; ++i) {
    LL_CHECK(input[i] >= 0 && input[i] <= 255, "transport handle bytes must be in [0,255]");
    result[i] = static_cast<char>(input[i]);
  }
  return result;
}
// Opaque IDs reject stale/invalid handles instead of dereferencing integers.
// Python serializes operations and close on each buffer; the shared reference
// protects the native object while a bound call is executing.
template <typename T>
class RuntimeRegistry {
 public:
  int64_t insert(std::shared_ptr<T> runtime) {
    std::lock_guard<std::mutex> lock(mutex_);
    LL_CHECK(next_ < std::numeric_limits<int64_t>::max(), "native handle space exhausted");
    const auto handle = next_++;
    values_.emplace(handle, std::move(runtime));
    return handle;
  }
  std::shared_ptr<T> get(int64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = values_.find(handle);
    LL_CHECK(found != values_.end(), "invalid or closed native buffer handle");
    return found->second;
  }
  void erase(int64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    values_.erase(handle);
  }

 private:
  std::mutex mutex_;
  std::unordered_map<int64_t, std::shared_ptr<T>> values_;
  int64_t next_ = 1;
};
}  // namespace deep_ep
