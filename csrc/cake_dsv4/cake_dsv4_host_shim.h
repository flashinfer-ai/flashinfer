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

// Shared host-side helpers of the Cake DSv4 TVM-FFI launchers (one copy for every binding).
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tvm_ffi_utils.h"

namespace cake_dsv4_host_shim {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device_id) {
    cudaError_t error = cudaGetDevice(&previous_device_);
    TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
        << "cudaGetDevice failed before host-shim launch: cudaError=" << static_cast<int>(error);
    if (previous_device_ != device_id) {
      error = cudaSetDevice(device_id);
      TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
          << "cudaSetDevice failed before host-shim launch for cuda:" << device_id
          << ": cudaError=" << static_cast<int>(error);
      restore_ = true;
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  ~ScopedCudaDevice() noexcept {
    if (restore_) {
      (void)cudaSetDevice(previous_device_);
    }
  }

 private:
  int previous_device_ = -1;
  bool restore_ = false;
};

inline void CheckCudaTensor(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << (int)t.device().device_type;
}

inline void CheckSameCudaDevice(const TensorView& t, const TensorView& reference, const char* name,
                                const char* reference_name) {
  TVM_FFI_CHECK(t.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as " << reference_name
      << ": got cuda:" << t.device().device_id << " versus cuda:" << reference.device().device_id;
}

inline void CheckCurrentCudaDevice(const TensorView& reference, const char* reference_name) {
  int current_device = -1;
  cudaError_t error = cudaGetDevice(&current_device);
  TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
      << "cudaGetDevice failed while validating " << reference_name
      << ": cudaError=" << static_cast<int>(error);
  TVM_FFI_CHECK(current_device == reference.device().device_id, ValueError)
      << "current CUDA device must match " << reference_name << ": current=cuda:" << current_device
      << ", tensor=cuda:" << reference.device().device_id;
}

inline void CheckContiguous(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.IsContiguous(), ValueError) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& t, const char* name, int code, int bits, int lanes) {
  DLDataType d = t.dtype();
  TVM_FFI_CHECK((int)d.code == code && (int)d.bits == bits && (int)d.lanes == lanes, TypeError)
      << name << " dtype mismatch: expected DLDataType(code=" << code << ", bits=" << bits
      << ", lanes=" << lanes << "), got (code=" << (int)d.code << ", bits=" << (int)d.bits
      << ", lanes=" << (int)d.lanes << ")";
}

// A logical axis.outer(trailing) folds every source dim above the trailing
// dimensions. Shape products are independent of physical strides, so verify
// the leading dimensions form one dense row-major chain instead of inventing
// a "folded stride". The descriptor reads its exact adjacent physical step
// separately through stride[-(trailing + 1)].
inline void CheckDenseLeadingFold(const TensorView& t, int trailing, const char* name) {
  TVM_FFI_CHECK(trailing > 0 && t.ndim() >= trailing, ValueError)
      << name << " cannot fold leading dimensions above " << trailing
      << " trailing dims from ndim=" << t.ndim();
  int outer_last = t.ndim() - trailing - 1;
  if (outer_last <= 0) {
    return;
  }
  int64_t step = t.stride(outer_last);
  TVM_FFI_CHECK(step > 0, ValueError) << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis) << ", expected "
          << expected;
    }
  }
}

// Explicit caller descriptor storage. The owner retains the allocation and
// never writes it; this function alone sets its bytes: the first time
// synchronously (no launch has read the storage yet, matching the production
// TmaDeviceSlot path), afterwards only when the descriptors of the call differ,
// in order on the launching stream so every earlier launch on that stream
// finished reading the previous descriptors. Neither write may happen inside
// CUDA Graph capture; the host keeps a storage captured by a graph unchanged.
template <size_t N>
static inline void PrepareImmutableCallerTmaWorkspace(const tvm::ffi::Tensor& retained_workspace,
                                                      const CUtensorMap (&maps)[N],
                                                      cudaStream_t stream) {
  void* workspace = retained_workspace.data_ptr();
  CUcontext context = nullptr;
  CUresult result = cuCtxGetCurrent(&context);
  TVM_FFI_CHECK(result == CUDA_SUCCESS && context != nullptr, RuntimeError)
      << "immutable TMA workspace requires an active CUDA context";
  CUcontext allocation_context = nullptr;
  result = cuPointerGetAttribute(&allocation_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                                 reinterpret_cast<CUdeviceptr>(workspace));
  TVM_FFI_CHECK(result == CUDA_SUCCESS && allocation_context == context, RuntimeError)
      << "immutable TMA workspace must belong to the active CUDA context";
  unsigned long long buffer_id = 0;
  result = cuPointerGetAttribute(&buffer_id, CU_POINTER_ATTRIBUTE_BUFFER_ID,
                                 reinterpret_cast<CUdeviceptr>(workspace));
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "querying immutable TMA allocation identity failed";
  std::string key = std::to_string(reinterpret_cast<uintptr_t>(context));
  key += ":" + std::to_string(reinterpret_cast<uintptr_t>(workspace));
  key += ":" + std::to_string(buffer_id);
  std::string descriptor_bytes(reinterpret_cast<const char*>(maps), sizeof(maps));
  struct InitializedWorkspace {
    tvm::ffi::Tensor owner;
    std::string descriptor_bytes;
  };
  static std::mutex mu;
  // Keep the actual Tensor allocation alive: a caching allocator may reuse a
  // freed suballocation without changing its enclosing CUDA buffer identity.
  static auto* initialized = new std::unordered_map<std::string, InitializedWorkspace>();
  std::lock_guard<std::mutex> lock(mu);
  auto it = initialized->find(key);
  if (it != initialized->end() && it->second.descriptor_bytes == descriptor_bytes) {
    return;
  }
  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  result = cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "querying TMA descriptor storage capture state failed";
  TVM_FFI_CHECK(capture_status == CU_STREAM_CAPTURE_STATUS_NONE, RuntimeError)
      << "TMA descriptor storage must be initialized before CUDA Graph capture";
  if (it == initialized->end()) {
    // Synchronous host-to-device initialization completes before publishing the
    // slot to any thread/stream, matching TmaDeviceSlot. The consumer retains its
    // generated tensor-map acquire; the mutable upload's release fence is unchanged.
    result = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(workspace), maps, sizeof(maps));
    TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
        << "initializing caller TMA descriptors failed";
    initialized->emplace(std::move(key),
                         InitializedWorkspace{retained_workspace, std::move(descriptor_bytes)});
    return;
  }
  // The host reassigned this storage to another descriptor set: rewrite it in
  // stream order, after every earlier launch on this stream read the old set.
  // A pageable source is staged before the call returns.
  result = cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(workspace), maps, sizeof(maps),
                             reinterpret_cast<CUstream>(stream));
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError) << "rewriting caller TMA descriptors failed";
  it->second.descriptor_bytes = std::move(descriptor_bytes);
}

static inline void* CallerTmaWorkspaceSlot(const tvm::ffi::TensorView& workspace, size_t slot) {
  return static_cast<char*>(workspace.data_ptr()) + slot * sizeof(CUtensorMap);
}

}  // namespace cake_dsv4_host_shim
