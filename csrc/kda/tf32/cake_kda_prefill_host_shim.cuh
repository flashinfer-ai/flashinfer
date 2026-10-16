/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

// Shared tvm-ffi host-shim helpers for the generated Cake KDA prefill bindings in
// csrc/kda/tf32 and csrc/kda/bf16: device guard, tensor checks, caller-owned TMA
// descriptor upload, the dynamic shared-memory opt-in and every distinct TMA
// descriptor encoder. Bindings include this header and keep only their own
// argument checks, descriptor wiring and launch.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace cake_kda_prefill_shim {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

class ScopedCudaDevice {
 public:
  explicit ScopedCudaDevice(int device_id) {
    cudaError_t error = cudaGetDevice(&previous_device_);
    TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
        << "cudaGetDevice failed before host-shim launch: cudaError="
        << static_cast<int>(error);
    if (previous_device_ != device_id) {
      error = cudaSetDevice(device_id);
      TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
          << "cudaSetDevice failed before host-shim launch for cuda:"
          << device_id << ": cudaError=" << static_cast<int>(error);
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

inline int64_t CakeDeviceMultiprocessorCount(int device_id) {
  constexpr int kMaxCachedCudaDevices = 64;
  TVM_FFI_CHECK(device_id >= 0 && device_id < kMaxCachedCudaDevices, RuntimeError)
      << "physical-SM-count cache does not cover cuda:" << device_id;
  static std::atomic<int> count_by_device[kMaxCachedCudaDevices]{};
  int cached = count_by_device[device_id].load(std::memory_order_acquire);
  if (cached > 0) return cached;

  int count = 0;
  cudaError_t error = cudaDeviceGetAttribute(
      &count, cudaDevAttrMultiProcessorCount, device_id);
  TVM_FFI_CHECK(error == cudaSuccess && count > 0, RuntimeError)
      << "querying multiProcessorCount failed for cuda:" << device_id
      << ": cudaError=" << static_cast<int>(error) << ", count=" << count;
  // Concurrent first launches may repeat the immutable device query, but all
  // publication is atomic and every later hot-path lookup is one acquire load.
  count_by_device[device_id].store(count, std::memory_order_release);
  return count;
}

inline void CheckCudaTensor(const TensorView& t, const char* name) {
  TVM_FFI_CHECK(t.device().device_type == kDLCUDA, ValueError)
      << name << " must be a CUDA tensor, got device_type=" << (int)t.device().device_type;
}

inline void CheckSameCudaDevice(
    const TensorView& t,
    const TensorView& reference,
    const char* name,
    const char* reference_name) {
  TVM_FFI_CHECK(t.device().device_id == reference.device().device_id, ValueError)
      << name << " must be on the same CUDA device as " << reference_name
      << ": got cuda:" << t.device().device_id
      << " versus cuda:" << reference.device().device_id;
}

inline void CheckCurrentCudaDevice(
    const TensorView& reference,
    const char* reference_name) {
  int current_device = -1;
  cudaError_t error = cudaGetDevice(&current_device);
  TVM_FFI_CHECK(error == cudaSuccess, RuntimeError)
      << "cudaGetDevice failed while validating " << reference_name
      << ": cudaError=" << static_cast<int>(error);
  TVM_FFI_CHECK(current_device == reference.device().device_id, ValueError)
      << "current CUDA device must match " << reference_name
      << ": current=cuda:" << current_device
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
  TVM_FFI_CHECK(step > 0, ValueError)
      << name << " physical strides must be positive";
  int64_t expected = step;
  for (int axis = outer_last - 1; axis >= 0; --axis) {
    expected *= t.size(axis + 1);
    if (t.size(axis) > 1) {
      TVM_FFI_CHECK(t.stride(axis) == expected, ValueError)
          << name << " leading dims are not physically foldable above " << trailing
          << " trailing dims: stride(" << axis << ")=" << t.stride(axis)
          << ", expected " << expected;
    }
  }
}

// Caller-owned storage is mutable scratch, including when its address and
// underlying CUDA allocation stay unchanged. Refresh all descriptors on the
// launch stream every time; allocation identity cannot establish their contents.
// By-value kernel parameters make the upload independent of host-buffer lifetime
// and record the descriptor bytes in CUDA Graphs for every replay.
template <size_t N>
struct CallerTmaDescriptorBatch {
  uint4 words[N * sizeof(CUtensorMap) / sizeof(uint4)];
};

template <size_t N>
__global__ void UploadCallerTmaDescriptors(
    uint4* workspace, CallerTmaDescriptorBatch<N> descriptors) {
  for (size_t i = threadIdx.x;
       i < N * sizeof(CUtensorMap) / sizeof(uint4); i += blockDim.x) {
    workspace[i] = descriptors.words[i];
  }
  // Generated pointer-ABI consumers acquire the tensor-map proxy before use.
  asm volatile("fence.proxy.tensormap::generic.release.sys;" ::: "memory");
}

template <size_t N>
static inline void UploadCallerTmaWorkspace(
    void* workspace, const CUtensorMap (&maps)[N], cudaStream_t stream) {
  CallerTmaDescriptorBatch<N> descriptors;
  static_assert(sizeof(descriptors) == sizeof(maps), "tensor-map batch ABI");
  std::memcpy(&descriptors, maps, sizeof(maps));
  void* args[] = {&workspace, &descriptors};
  cudaError_t result = cudaLaunchKernel(
      reinterpret_cast<const void*>(UploadCallerTmaDescriptors<N>),
      dim3(1), dim3(32), args, 0, stream);
  TVM_FFI_CHECK(result == cudaSuccess, RuntimeError)
      << "uploading caller-owned TMA descriptors failed: "
      << cudaGetErrorString(result);
}

static inline void* CallerTmaWorkspaceSlot(
    const tvm::ffi::TensorView& workspace,
    size_t slot) {
  const uintptr_t base = reinterpret_cast<uintptr_t>(workspace.data_ptr());
  const uintptr_t address = base + slot * sizeof(CUtensorMap);
  return reinterpret_cast<void*>(address);
}

// Opt a linked kernel into its dynamic shared-memory budget once per device.
// The caller owns the per-kernel mutex and status map (function-local statics in
// its own translation unit); a failed opt-in is remembered and reported on every call.
inline cudaError_t EnsureMaxDynamicSharedMemory(
    std::mutex& mu,
    std::unordered_map<int, cudaError_t>& status_by_device,
    int device_id,
    const void* kernel,
    int bytes) {
  cudaError_t smem_status = cudaSuccess;
  std::lock_guard<std::mutex> lock(mu);
  auto it = status_by_device.find(device_id);
  if (it == status_by_device.end()) {
    int current_device = -1;
    smem_status = cudaGetDevice(&current_device);
    if (smem_status == cudaSuccess && current_device != device_id) {
      smem_status = cudaErrorInvalidDevice;
    }
    if (smem_status == cudaSuccess) {
      smem_status = cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    }
    status_by_device.emplace(device_id, smem_status);
  } else {
    smem_status = it->second;
  }
  return smem_status;
}

// 2D TMA descriptor for buffer 'beta_logits_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_beta_logits_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'beta_logits_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'beta_logits_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'beta_logits_tma' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "beta_logits_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'beta_logits_tma' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'beta_logits_tma' resolved a non-positive global dim";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'beta_logits_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'beta_logits_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'beta_logits_tma' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
  };
  uint32_t box_dim[2] = {8u, 32u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'beta_logits_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'beta_logits_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_beta_logits_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'beta_logits_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'beta_logits_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'beta_logits_tma' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "beta_logits_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'beta_logits_tma' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'beta_logits_tma' resolved a non-positive global dim";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'beta_logits_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'beta_logits_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'beta_logits_tma' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
  };
  uint32_t box_dim[2] = {8u, 16u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'beta_logits_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'beta_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_beta_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'beta_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'beta_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'beta_tma' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "beta_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'beta_tma' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved a non-positive global dim";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
  };
  uint32_t box_dim[2] = {8u, 32u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'beta_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'beta_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_beta_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'beta_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'beta_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'beta_tma' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "beta_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'beta_tma' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved a non-positive global dim";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
  };
  uint32_t box_dim[2] = {8u, 16u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'beta_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'beta_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_beta_tma_2(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'beta_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'beta_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'beta_tma' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "beta_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'beta_tma' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(8u <= global_dim[0] && 32u <= global_dim[1], ValueError)
      << "TMA box (8, 32) exceeds resolved global dims for 'beta_tma'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
  };
  uint32_t box_dim[2] = {8u, 32u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'beta_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'beta_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_beta_tma_3(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'beta_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'beta_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'beta_tma' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "beta_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'beta_tma' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved a non-positive global dim";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'beta_tma' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
  };
  uint32_t box_dim[2] = {24u, 17u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'beta_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'carry_hi_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_carry_hi_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'carry_hi_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'carry_hi_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'carry_hi_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'carry_hi_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 128, 1, 2) exceeds resolved global dims for 'carry_hi_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_hi_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 128u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'carry_hi_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'carry_lo_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_carry_lo_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'carry_lo_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'carry_lo_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'carry_lo_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'carry_lo_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 128, 1, 2) exceeds resolved global dims for 'carry_lo_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_lo_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 128u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'carry_lo_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'carry_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_carry_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'carry_tma' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'carry_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'carry_tma' trailing dims must be positive";
  TVM_FFI_CHECK(d1 % 32 == 0, ValueError)
      << "TMA source 'carry_tma' extent " << d1
      << " must divide exactly by " << 32;
  uint64_t global_dim[5] = {(uint64_t)(32), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)((d1 / 32)), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'carry_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2] && 2u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (32, 128, 1, 2, 1) exceeds resolved global dims for 'carry_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 32;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 3 to a non-whole-byte offset";
  int64_t carrier_stride_3 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_3 >= 0, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 4 negative";
  TVM_FFI_CHECK(carrier_stride_3 != 0 || global_dim[4] == 1, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 4 zero while global dimension 4 is not 1";
  TVM_FFI_CHECK((carrier_stride_3 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'carry_tma' resolved global stride 4 to a non-whole-byte offset";
  uint64_t global_strides[4] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
      (uint64_t)((carrier_stride_2 * 32) / 8),
      (uint64_t)((carrier_stride_3 * 32) / 8),
  };
  uint32_t box_dim[5] = {32u, 128u, 1u, 2u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'carry_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'coefficients_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_coefficients_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'coefficients_tma' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'coefficients_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'coefficients_tma' trailing dims must be positive";
  TVM_FFI_CHECK(d1 % 32 == 0, ValueError)
      << "TMA source 'coefficients_tma' extent " << d1
      << " must divide exactly by " << 32;
  uint64_t global_dim[5] = {(uint64_t)(32), (uint64_t)(d3), (uint64_t)(d2), (uint64_t)((d1 / 32)), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (32, 128, 1, 2, 1) exceeds resolved global dims for 'coefficients_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 32;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 3 to a non-whole-byte offset";
  int64_t carrier_stride_3 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_3 >= 0, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 4 negative";
  TVM_FFI_CHECK(carrier_stride_3 != 0 || global_dim[4] == 1, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 4 zero while global dimension 4 is not 1";
  TVM_FFI_CHECK((carrier_stride_3 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'coefficients_tma' resolved global stride 4 to a non-whole-byte offset";
  uint64_t global_strides[4] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
      (uint64_t)((carrier_stride_2 * 32) / 8),
      (uint64_t)((carrier_stride_3 * 32) / 8),
  };
  uint32_t box_dim[5] = {32u, 128u, 1u, 2u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'coefficients_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'final_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_final_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'final_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'final_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'final_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'final_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (32, 128, 1) exceeds resolved global dims for 'final_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'final_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'final_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'final_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'final_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'final_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'final_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
  };
  uint32_t box_dim[3] = {32u, 128u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'final_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'g_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_g_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'g_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'g_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'g_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "g_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'g_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'g_tma' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1, 16) exceeds resolved global dims for 'g_tma'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s3;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {128u, 1u, 16u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'g_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'g_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_g_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'g_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'g_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'g_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "g_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'g_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'g_tma' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1, 32) exceeds resolved global dims for 'g_tma'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s3;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'g_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {128u, 1u, 32u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'g_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'k_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_k_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'k_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'k_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'k_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "k_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'k_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'k_tma' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'k_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 2) exceeds resolved global dims for 'k_tma'";
  int64_t carrier_stride_0 = s3;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s2;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'k_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'k_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_k_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'k_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'k_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'k_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'k_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 16, 1, 2) exceeds resolved global dims for 'k_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 16u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'k_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'k_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_k_tma_2(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'k_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'k_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'k_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'k_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 1) exceeds resolved global dims for 'k_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'k_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'k_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_k_tma_3(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'k_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'k_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'k_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'k_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 2) exceeds resolved global dims for 'k_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'k_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'k_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_k_tma_4(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'k_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'k_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'k_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "k_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'k_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'k_tma' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'k_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 16, 1, 2) exceeds resolved global dims for 'k_tma'";
  int64_t carrier_stride_0 = s3;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s2;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'k_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 16u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'k_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'map_final_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_map_final_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'map_final_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'map_final_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'map_final_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'map_final_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 128, 1, 2) exceeds resolved global dims for 'map_final_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 128u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'map_final_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'map_final_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_map_final_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'map_final_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'map_final_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'map_final_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (32, 128, 1) exceeds resolved global dims for 'map_final_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'map_final_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {32u, 128u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'map_final_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'maps_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_maps_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'maps_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'maps_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'maps_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'maps_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 128, 1, 2) exceeds resolved global dims for 'maps_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 128u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'maps_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'maps_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_maps_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'maps_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'maps_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'maps_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (32, 128, 1) exceeds resolved global dims for 'maps_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'maps_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {32u, 128u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'maps_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'out_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_out_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'out_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'out_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'out_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'out_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 16, 1, 2) exceeds resolved global dims for 'out_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 16u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'out_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'out_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_out_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'out_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'out_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'out_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'out_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 1) exceeds resolved global dims for 'out_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'out_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'out_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_out_tma_2(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'out_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'out_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'out_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'out_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 2) exceeds resolved global dims for 'out_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'out_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'out_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_out_tma_3(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'out_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'out_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'out_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[1] && 32u <= global_dim[2], ValueError)
      << "TMA box (64, 1, 32) exceeds resolved global dims for 'out_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 1u, 32u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'out_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'out_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_out_tma_4(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'out_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'out_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'out_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'out_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 16, 1, 1) exceeds resolved global dims for 'out_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'out_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 16u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'out_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'owner_packet_tail_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_owner_packet_tail_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'owner_packet_tail_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'owner_packet_tail_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'owner_packet_tail_tma' trailing dims must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'owner_packet_tail_tma' resolved a non-positive global dim";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'owner_packet_tail_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'owner_packet_tail_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'owner_packet_tail_tma' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
  };
  uint32_t box_dim[2] = {128u, 10u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'owner_packet_tail_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'owner_packet_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_owner_packet_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'owner_packet_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'owner_packet_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'owner_packet_tma' trailing dims must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'owner_packet_tma' resolved a non-positive global dim";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'owner_packet_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'owner_packet_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'owner_packet_tma' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
  };
  uint32_t box_dim[2] = {128u, 146u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'owner_packet_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 2D TMA descriptor for buffer 'owner_packet_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_owner_packet_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'owner_packet_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'owner_packet_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'owner_packet_tma' trailing dims must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(d2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'owner_packet_tma' resolved a non-positive global dim";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'owner_packet_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'owner_packet_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'owner_packet_tma' resolved global stride 1 to a non-whole-byte offset";
  uint64_t global_strides[1] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
  };
  uint32_t box_dim[2] = {128u, 104u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'owner_packet_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'pair_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_pair_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'pair_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'pair_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'pair_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (32, 128, 1) exceeds resolved global dims for 'pair_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {32u, 128u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'pair_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'pair_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_pair_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'pair_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'pair_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'pair_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'pair_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 128, 1, 2) exceeds resolved global dims for 'pair_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'pair_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 128u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'pair_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'q_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_q_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'q_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'q_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'q_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "q_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'q_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'q_tma' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'q_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 2) exceeds resolved global dims for 'q_tma'";
  int64_t carrier_stride_0 = s3;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s2;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'q_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'q_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_q_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'q_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'q_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'q_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'q_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 2) exceeds resolved global dims for 'q_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'q_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'q_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_q_tma_2(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'q_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'q_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'q_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "q_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'q_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'q_tma' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'q_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 16, 1, 2) exceeds resolved global dims for 'q_tma'";
  int64_t carrier_stride_0 = s3;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s2;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 16u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'q_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'q_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_q_tma_3(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'q_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'q_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'q_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'q_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 16, 1, 2) exceeds resolved global dims for 'q_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 16u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'q_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'q_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_q_tma_4(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'q_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'q_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'q_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'q_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 1) exceeds resolved global dims for 'q_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'q_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'q_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'raw_gate_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_raw_gate_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'raw_gate_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'raw_gate_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'raw_gate_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "raw_gate_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'raw_gate_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'raw_gate_tma' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1, 32) exceeds resolved global dims for 'raw_gate_tma'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s3;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {128u, 1u, 32u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'raw_gate_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'raw_gate_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_raw_gate_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'raw_gate_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'raw_gate_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'raw_gate_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "raw_gate_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'raw_gate_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'raw_gate_tma' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1, 16) exceeds resolved global dims for 'raw_gate_tma'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s3;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'raw_gate_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {128u, 1u, 16u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'raw_gate_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'rows_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_rows_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'rows_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'rows_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'rows_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'rows_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (32, 128, 1) exceeds resolved global dims for 'rows_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'rows_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'rows_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'rows_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'rows_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'rows_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'rows_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
  };
  uint32_t box_dim[3] = {32u, 128u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'rows_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'state_checkpoints_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_state_checkpoints_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'state_checkpoints_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'state_checkpoints_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'state_checkpoints_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (32, 128, 1) exceeds resolved global dims for 'state_checkpoints_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
  };
  uint32_t box_dim[3] = {32u, 128u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'state_checkpoints_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'state_checkpoints_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_state_checkpoints_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'state_checkpoints_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'state_checkpoints_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source 'state_checkpoints_tma' trailing dims must be positive";
  int64_t outer3 = t.numel() / (d1 * d2 * d3);
  uint64_t global_dim[4] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)(outer3)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 128, 1, 1) exceeds resolved global dims for 'state_checkpoints_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 128u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'state_checkpoints_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'state_checkpoints_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_state_checkpoints_tma_2(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'state_checkpoints_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'state_checkpoints_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'state_checkpoints_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (32, 128, 1) exceeds resolved global dims for 'state_checkpoints_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'state_checkpoints_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {32u, 128u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'state_checkpoints_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'v_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_v_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'v_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'v_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'v_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (64, 1, 16) exceeds resolved global dims for 'v_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d1 * d2);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 1u, 16u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'v_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'v_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_v_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'v_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'v_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'v_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1, 32) exceeds resolved global dims for 'v_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d1 * d2);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {128u, 1u, 32u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'v_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'v_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_v_tma_2(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'v_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'v_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'v_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "v_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'v_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'v_tma' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (64, 1, 32) exceeds resolved global dims for 'v_tma'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s3;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 1u, 32u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'v_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'v_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_v_tma_3(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'v_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'v_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'v_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "v_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'v_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'v_tma' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'v_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d2), (uint64_t)(outer2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[1] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 1, 16, 1) exceeds resolved global dims for 'v_tma'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s3;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 1u, 16u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'v_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'v_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_v_tma_4(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'v_tma' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'v_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'v_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'v_tma' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(outer2), (uint64_t)(d2), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 2u <= global_dim[3], ValueError)
      << "TMA box (64, 32, 1, 2) exceeds resolved global dims for 'v_tma'";
  int64_t carrier_stride_0 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = d1;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 64;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
      (uint64_t)((carrier_stride_2 * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 32u, 1u, 2u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'v_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'v_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_v_tma_5(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source 'v_tma' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'v_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'v_tma' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  CheckDenseLeadingFold(t, 2, "v_tma");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'v_tma' physical strides must be positive";
  int64_t s3 = t.stride(t.ndim() - 3) * 1;
  TVM_FFI_CHECK(s3 > 0, ValueError)
      << "TMA source 'v_tma' physical strides must be positive";
  uint64_t global_dim[3] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (128, 1, 32) exceeds resolved global dims for 'v_tma'";
  int64_t carrier_stride_0 = s2;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = s3;
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 16) % 8 == 0, ValueError)
      << "TMA descriptor for 'v_tma' resolved global stride 2 to a non-whole-byte offset";
  uint64_t global_strides[2] = {
      (uint64_t)((carrier_stride_0 * 16) / 8),
      (uint64_t)((carrier_stride_1 * 16) / 8),
  };
  uint32_t box_dim[3] = {128u, 1u, 32u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'v_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'ws_diag_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_ws_diag_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'ws_diag_tma' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'ws_diag_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'ws_diag_tma' trailing dims must be positive";
  uint64_t global_dim[4] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 1u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (128, 1, 1, 1) exceeds resolved global dims for 'ws_diag_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_diag_tma' resolved global stride 3 to a non-whole-byte offset";
  uint64_t global_strides[3] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
      (uint64_t)((carrier_stride_2 * 32) / 8),
  };
  uint32_t box_dim[4] = {128u, 1u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'ws_diag_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'ws_kd_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_ws_kd_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'ws_kd_tma' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'ws_kd_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'ws_kd_tma' trailing dims must be positive";
  TVM_FFI_CHECK(d1 % 32 == 0, ValueError)
      << "TMA source 'ws_kd_tma' extent " << d1
      << " must divide exactly by " << 32;
  uint64_t global_dim[5] = {(uint64_t)(32), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)((d1 / 32)), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 16u <= global_dim[1] && 1u <= global_dim[2] && 4u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (32, 16, 1, 4, 1) exceeds resolved global dims for 'ws_kd_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 32;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 3 to a non-whole-byte offset";
  int64_t carrier_stride_3 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_3 >= 0, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 4 negative";
  TVM_FFI_CHECK(carrier_stride_3 != 0 || global_dim[4] == 1, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 4 zero while global dimension 4 is not 1";
  TVM_FFI_CHECK((carrier_stride_3 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_kd_tma' resolved global stride 4 to a non-whole-byte offset";
  uint64_t global_strides[4] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
      (uint64_t)((carrier_stride_2 * 32) / 8),
      (uint64_t)((carrier_stride_3 * 32) / 8),
  };
  uint32_t box_dim[5] = {32u, 16u, 1u, 4u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'ws_kd_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'ws_qd_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_ws_qd_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 4, ValueError)
      << "TMA source 'ws_qd_tma' must have at least 4 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'ws_qd_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0, ValueError)
      << "TMA source 'ws_qd_tma' trailing dims must be positive";
  TVM_FFI_CHECK(d1 % 32 == 0, ValueError)
      << "TMA source 'ws_qd_tma' extent " << d1
      << " must divide exactly by " << 32;
  uint64_t global_dim[5] = {(uint64_t)(32), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)((d1 / 32)), (uint64_t)(d4)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(32u <= global_dim[0] && 16u <= global_dim[1] && 1u <= global_dim[2] && 4u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (32, 16, 1, 4, 1) exceeds resolved global dims for 'ws_qd_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = 32;
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 3 to a non-whole-byte offset";
  int64_t carrier_stride_3 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_3 >= 0, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 4 negative";
  TVM_FFI_CHECK(carrier_stride_3 != 0 || global_dim[4] == 1, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 4 zero while global dimension 4 is not 1";
  TVM_FFI_CHECK((carrier_stride_3 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_qd_tma' resolved global stride 4 to a non-whole-byte offset";
  uint64_t global_strides[4] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
      (uint64_t)((carrier_stride_2 * 32) / 8),
      (uint64_t)((carrier_stride_3 * 32) / 8),
  };
  uint32_t box_dim[5] = {32u, 16u, 1u, 4u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'ws_qd_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'ws_qk_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_ws_qk_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 5, ValueError)
      << "TMA source 'ws_qk_tma' must have at least 5 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'ws_qk_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  int64_t d5 = t.size(t.ndim() - 5);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0 && d5 > 0, ValueError)
      << "TMA source 'ws_qk_tma' trailing dims must be positive";
  uint64_t global_dim[5] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)(d4), (uint64_t)(d5)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(16u <= global_dim[0] && 16u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (16, 16, 1, 1, 1) exceeds resolved global dims for 'ws_qk_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 3 to a non-whole-byte offset";
  int64_t carrier_stride_3 = (((d4 * d3) * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_3 >= 0, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 4 negative";
  TVM_FFI_CHECK(carrier_stride_3 != 0 || global_dim[4] == 1, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 4 zero while global dimension 4 is not 1";
  TVM_FFI_CHECK((carrier_stride_3 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_qk_tma' resolved global stride 4 to a non-whole-byte offset";
  uint64_t global_strides[4] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
      (uint64_t)((carrier_stride_2 * 32) / 8),
      (uint64_t)((carrier_stride_3 * 32) / 8),
  };
  uint32_t box_dim[5] = {16u, 16u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'ws_qk_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'ws_w_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_ws_w_tma_0(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 5, ValueError)
      << "TMA source 'ws_w_tma' must have at least 5 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'ws_w_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  int64_t d5 = t.size(t.ndim() - 5);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0 && d5 > 0, ValueError)
      << "TMA source 'ws_w_tma' trailing dims must be positive";
  uint64_t global_dim[5] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)(d4), (uint64_t)(d5)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(16u <= global_dim[0] && 128u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (16, 128, 1, 1, 1) exceeds resolved global dims for 'ws_w_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 3 to a non-whole-byte offset";
  int64_t carrier_stride_3 = (((d4 * d3) * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_3 >= 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 4 negative";
  TVM_FFI_CHECK(carrier_stride_3 != 0 || global_dim[4] == 1, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 4 zero while global dimension 4 is not 1";
  TVM_FFI_CHECK((carrier_stride_3 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 4 to a non-whole-byte offset";
  uint64_t global_strides[4] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
      (uint64_t)((carrier_stride_2 * 32) / 8),
      (uint64_t)((carrier_stride_3 * 32) / 8),
  };
  uint32_t box_dim[5] = {16u, 128u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'ws_w_tma') failed: CUresult=" << (int)r;
  return tm;
}

// 5D TMA descriptor for buffer 'ws_w_tma' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_ws_w_tma_1(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 5, ValueError)
      << "TMA source 'ws_w_tma' must have at least 5 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'ws_w_tma' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  int64_t d5 = t.size(t.ndim() - 5);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0 && d4 > 0 && d5 > 0, ValueError)
      << "TMA source 'ws_w_tma' trailing dims must be positive";
  uint64_t global_dim[5] = {(uint64_t)(d1), (uint64_t)(d2), (uint64_t)(d3), (uint64_t)(d4), (uint64_t)(d5)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0 && global_dim[4] > 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved a non-positive global dim";
  TVM_FFI_CHECK(16u <= global_dim[0] && 64u <= global_dim[1] && 1u <= global_dim[2] && 1u <= global_dim[3] && 1u <= global_dim[4], ValueError)
      << "TMA box (16, 64, 1, 1, 1) exceeds resolved global dims for 'ws_w_tma'";
  int64_t carrier_stride_0 = d1;
  TVM_FFI_CHECK(carrier_stride_0 >= 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 1 negative";
  TVM_FFI_CHECK(carrier_stride_0 != 0 || global_dim[1] == 1, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 1 zero while global dimension 1 is not 1";
  TVM_FFI_CHECK((carrier_stride_0 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 1 to a non-whole-byte offset";
  int64_t carrier_stride_1 = (d2 * d1);
  TVM_FFI_CHECK(carrier_stride_1 >= 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 2 negative";
  TVM_FFI_CHECK(carrier_stride_1 != 0 || global_dim[2] == 1, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 2 zero while global dimension 2 is not 1";
  TVM_FFI_CHECK((carrier_stride_1 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 2 to a non-whole-byte offset";
  int64_t carrier_stride_2 = ((d3 * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_2 >= 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 3 negative";
  TVM_FFI_CHECK(carrier_stride_2 != 0 || global_dim[3] == 1, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 3 zero while global dimension 3 is not 1";
  TVM_FFI_CHECK((carrier_stride_2 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 3 to a non-whole-byte offset";
  int64_t carrier_stride_3 = (((d4 * d3) * d2) * d1);
  TVM_FFI_CHECK(carrier_stride_3 >= 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 4 negative";
  TVM_FFI_CHECK(carrier_stride_3 != 0 || global_dim[4] == 1, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 4 zero while global dimension 4 is not 1";
  TVM_FFI_CHECK((carrier_stride_3 * 32) % 8 == 0, ValueError)
      << "TMA descriptor for 'ws_w_tma' resolved global stride 4 to a non-whole-byte offset";
  uint64_t global_strides[4] = {
      (uint64_t)((carrier_stride_0 * 32) / 8),
      (uint64_t)((carrier_stride_1 * 32) / 8),
      (uint64_t)((carrier_stride_2 * 32) / 8),
      (uint64_t)((carrier_stride_3 * 32) / 8),
  };
  uint32_t box_dim[5] = {16u, 64u, 1u, 1u, 1u};
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  const void* tensor_base = static_cast<const char*>(t.data_ptr()) + 0u;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, const_cast<void*>(tensor_base), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (5D, 'ws_w_tma') failed: CUresult=" << (int)r;
  return tm;
}

}  // namespace cake_kda_prefill_shim
