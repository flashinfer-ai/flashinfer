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
#pragma once

// Launch helpers shared by the SM110 GQA decode bindings: the TMA descriptors
// of the fixed FP16 operands (Q viewed as [batch, 4, 8, 128], K and V as
// [batch, 8, capacity, 128]) and the one-time dynamic shared memory opt-in.
// Tensor-metadata checks, the device guard and the stream come from
// tvm_ffi_utils.h; every binding is a launcher over these helpers.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace sm110_gqa_decode {

using tvm::ffi::TensorView;

// Host-side carrier of the 64-byte aligned tensor map the kernels take by value.
struct alignas(64) TensorMap64 {
  uint64_t opaque[16];
};
static_assert(sizeof(TensorMap64) == 128, "64-aligned tensor-map ABI size");

inline void CheckGrid(int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y << ", "
      << grid_z << ")";
}

inline void CheckStridedFp16(const TensorView& t, const char* name, int dims) {
  TVM_FFI_CHECK(t.ndim() >= dims, ValueError)
      << "TMA source '" << name << "' must have at least " << dims
      << " dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  for (int axis = 1; axis <= dims; ++axis) {
    TVM_FFI_CHECK(t.size(t.ndim() - axis) > 0, ValueError)
        << "TMA source '" << name << "' trailing dims must be positive";
  }
  for (int axis = 2; axis <= dims; ++axis) {
    TVM_FFI_CHECK(t.stride(t.ndim() - axis) > 0, ValueError)
        << "TMA source '" << name << "' physical strides must be positive";
  }
}

inline void CheckStride(int64_t stride, int index, int64_t dim, const char* name) {
  TVM_FFI_CHECK(stride >= 0, ValueError)
      << "TMA descriptor for '" << name << "' resolved global stride " << index << " negative";
  TVM_FFI_CHECK(stride != 0 || dim == 1, ValueError)
      << "TMA descriptor for '" << name << "' resolved global stride " << index
      << " zero while global dimension " << index << " is not 1";
}

inline CUtensorMap Encode(const char* name, uint32_t rank, const void* base,
                          const uint64_t* global_dim, const uint64_t* global_strides,
                          const uint32_t* box_dim) {
  uint32_t elem_strides[5] = {1u, 1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, rank, const_cast<void*>(base),
                                      global_dim, global_strides, box_dim, elem_strides,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (" << rank << "D, '" << name << "') failed: CUresult=" << (int)r;
  return tm;
}

// 5D descriptor of the grouped query view [batch, 4, 8, 128] (box 64 x 64 rows):
// dims (head_dim, kv_head, 1, head_in_group, batch).
inline CUtensorMap EncodeQ(const TensorView& t) {
  CheckStridedFp16(t, "Q", 4);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  int64_t s2 = t.stride(t.ndim() - 2);
  int64_t s3 = t.stride(t.ndim() - 3);
  int64_t s4 = t.stride(t.ndim() - 4);
  uint64_t global_dim[5] = {(uint64_t)d1, (uint64_t)d3, 1u, (uint64_t)d2, (uint64_t)d4};
  TVM_FFI_CHECK(64u <= global_dim[0], ValueError)
      << "TMA box (64, 64, 1, 1, 1) exceeds resolved global dims for 'Q'";
  int64_t strides[4] = {s3, s2, s2, s4};
  for (int i = 0; i < 4; ++i) CheckStride(strides[i], i + 1, (int64_t)global_dim[i + 1], "Q");
  uint64_t global_strides[4] = {(uint64_t)(s3 * 2), (uint64_t)(s2 * 2), (uint64_t)(s2 * 2), (uint64_t)(s4 * 2)};
  uint32_t box_dim[5] = {64u, 64u, 1u, 1u, 1u};
  return Encode("Q", 5, t.data_ptr(), global_dim, global_strides, box_dim);
}

// 4D descriptor of one KV plane [batch, 8, capacity, 128] with a box of
// ``box_tokens`` tokens x 64 head-dim elements.
inline CUtensorMap EncodeKV(const TensorView& t, const char* name, uint32_t box_tokens) {
  CheckStridedFp16(t, name, 4);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  int64_t d3 = t.size(t.ndim() - 3);
  int64_t d4 = t.size(t.ndim() - 4);
  int64_t s2 = t.stride(t.ndim() - 2);
  int64_t s3 = t.stride(t.ndim() - 3);
  int64_t s4 = t.stride(t.ndim() - 4);
  uint64_t global_dim[4] = {(uint64_t)d1, (uint64_t)d2, (uint64_t)d3, (uint64_t)d4};
  TVM_FFI_CHECK(64u <= global_dim[0] && box_tokens <= global_dim[1], ValueError)
      << "TMA box (64, " << box_tokens << ", 1, 1) exceeds resolved global dims for '" << name << "'";
  int64_t strides[3] = {s2, s3, s4};
  for (int i = 0; i < 3; ++i) CheckStride(strides[i], i + 1, (int64_t)global_dim[i + 1], name);
  uint64_t global_strides[3] = {(uint64_t)(s2 * 2), (uint64_t)(s3 * 2), (uint64_t)(s4 * 2)};
  uint32_t box_dim[4] = {64u, box_tokens, 1u, 1u};
  return Encode(name, 4, t.data_ptr(), global_dim, global_strides, box_dim);
}

// Opt the linked kernel into its dynamic shared memory once for every device
// that accepts it. Function-local static initialization serializes the first
// call; no lock or per-device cache is needed.
inline bool SetMaxDynamicSharedMemory(const void* symbol, int smem_bytes) {
  cudaKernel_t kernel = nullptr;
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetKernel(&kernel, symbol));
  int device_count = 0;
  TVM_FFI_CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));
  int configured = 0;
  for (int device = 0; device < device_count; ++device) {
    if (cudaKernelSetAttributeForDevice(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_bytes, device) == cudaSuccess) {
      ++configured;
    } else {
      (void)cudaGetLastError();
    }
  }
  TVM_FFI_CHECK(configured > 0, RuntimeError)
      << "no CUDA device accepts " << smem_bytes << " B of dynamic shared memory";
  return true;
}

inline void Launch(const void* symbol, const char* name, dim3 grid, dim3 block, unsigned smem_bytes,
                   cudaStream_t stream, void** kargs) {
  cudaError_t status = cudaLaunchKernel(symbol, grid, block, kargs, smem_bytes, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernel for " << name << " failed: " << cudaGetErrorString(status);
}

}  // namespace sm110_gqa_decode
