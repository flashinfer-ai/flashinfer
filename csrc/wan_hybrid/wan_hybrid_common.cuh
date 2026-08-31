/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace wan_hybrid {

constexpr int64_t kBatch = 1;
constexpr int64_t kSequence = 4800;
constexpr int64_t kHeads = 40;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kPaddedSequence = 5120;
constexpr int64_t kLogicalBlocks = 38;
constexpr int64_t kPhysicalBlocks = 40;
constexpr int64_t kValueRows = kBatch * kHeads * kHeadDim;
constexpr int64_t kPackedValueRows = 2 * kValueRows;
constexpr int64_t kPackedValueColumns = kPaddedSequence / 2;
constexpr int64_t kScaleRows = 25'600;
constexpr int64_t kScaleColumns = 32;
constexpr int64_t kValueScaleRows = 2 * kScaleRows;
constexpr int64_t kTensorMapCount = 6;
constexpr int64_t kTensorMapBytes = 128;
constexpr uintptr_t kTensorMapGlobalBaseAlignment = 16;
constexpr int64_t kAttentionThreads = 512;
constexpr int64_t kMaximumTiles = 147;
constexpr size_t kAttentionDynamicSmemBytes = 231'424;

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK_EQ(status, cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckDriver(CUresult status, const char* operation) {
  TVM_FFI_ICHECK_EQ(status, CUDA_SUCCESS)
      << operation << " failed with CUresult=" << static_cast<int>(status);
}

inline void CheckExactTensor(TensorView tensor, const char* name, DLDataType dtype,
                             std::initializer_list<int64_t> shape, int32_t device_id) {
  CHECK_INPUT(tensor);
  TVM_FFI_ICHECK_EQ(encode_dlpack_dtype(tensor.dtype()), encode_dlpack_dtype(dtype))
      << name << " has the wrong dtype";
  TVM_FFI_ICHECK_EQ(tensor.ndim(), static_cast<int32_t>(shape.size()))
      << name << " has the wrong rank";
  int32_t axis = 0;
  for (int64_t extent : shape) {
    TVM_FFI_ICHECK_EQ(tensor.size(axis), extent)
        << name << " has the wrong extent at axis " << axis;
    ++axis;
  }
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, device_id)
      << name << " must be on the same CUDA device as q";
}

inline void CheckTarget(int32_t device_id, const char* component) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK_EQ(major, 10) << "wan_hybrid " << component << " requires compute capability 10.x";
  TVM_FFI_ICHECK_EQ(minor, FLASHINFER_WAN_HYBRID_TARGET_MINOR)
      << "wan_hybrid " << component << " module target does not match the CUDA device";
}

inline CUtensorMap EncodeTensorMap(const void* address, CUtensorMapDataType data_type,
                                   uint32_t rank, const uint64_t* global_dims,
                                   const uint64_t* global_strides, const uint32_t* box_dims,
                                   CUtensorMapSwizzle swizzle, const char* name) {
  std::array<uint32_t, 5> element_strides{1, 1, 1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, data_type, rank, const_cast<void*>(address), global_dims, global_strides,
      box_dims, element_strides.data(), CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  CheckDriver(result, name);
  return tensor_map;
}

inline CUtensorMap EncodeNHD(const TensorView& tensor, uint32_t rows_per_box, const char* name) {
  TVM_FFI_ICHECK_EQ(reinterpret_cast<uintptr_t>(tensor.data_ptr()) %
                        kTensorMapGlobalBaseAlignment,
                    0)
      << name << " global address must be 16-byte aligned";
  constexpr uint64_t global_dims[5] = {64, kSequence, kHeads, kBatch, 2};
  constexpr uint64_t global_strides[4] = {
      kHeads * kHeadDim * sizeof(__nv_bfloat16),
      kHeadDim * sizeof(__nv_bfloat16),
      kSequence * kHeads * kHeadDim * sizeof(__nv_bfloat16),
      64 * sizeof(__nv_bfloat16),
  };
  const uint32_t box_dims[5] = {64, rows_per_box, 1, 1, 2};
  return EncodeTensorMap(tensor.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, global_dims,
                         global_strides, box_dims, CU_TENSOR_MAP_SWIZZLE_128B, name);
}

inline CUtensorMap Encode2D(const TensorView& tensor, uint64_t columns, uint64_t rows,
                            uint32_t box_columns, uint32_t box_rows, CUtensorMapSwizzle swizzle,
                            const char* name) {
  const uint64_t global_dims[2] = {columns, rows};
  const uint64_t global_strides[1] = {columns};
  const uint32_t box_dims[2] = {box_columns, box_rows};
  return EncodeTensorMap(tensor.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, global_dims,
                         global_strides, box_dims, swizzle, name);
}

inline void PrepareTensorMaps(const TensorView& q, const TensorView& k, const TensorView& vt,
                              const TensorView& sfvt_lo, const TensorView& sfvt_hi,
                              const TensorView& out, const TensorView& descriptor_storage) {
  std::array<CUtensorMap, kTensorMapCount> maps{
      EncodeNHD(q, 128, "cuTensorMapEncodeTiled(q)"),
      EncodeNHD(k, 128, "cuTensorMapEncodeTiled(k)"),
      Encode2D(vt, kPackedValueColumns, kPackedValueRows, 64, 128, CU_TENSOR_MAP_SWIZZLE_64B,
               "cuTensorMapEncodeTiled(vt)"),
      Encode2D(sfvt_lo, kScaleColumns, kValueScaleRows, 32, 16, CU_TENSOR_MAP_SWIZZLE_NONE,
               "cuTensorMapEncodeTiled(sfvt_lo)"),
      Encode2D(sfvt_hi, kScaleColumns, kValueScaleRows, 32, 16, CU_TENSOR_MAP_SWIZZLE_NONE,
               "cuTensorMapEncodeTiled(sfvt_hi)"),
      EncodeNHD(out, 128, "cuTensorMapEncodeTiled(out)"),
  };
  CheckCuda(
      cudaMemcpy(descriptor_storage.data_ptr(), maps.data(), sizeof(maps), cudaMemcpyHostToDevice),
      "cudaMemcpy(wan_hybrid tensor maps)");
}

}  // namespace wan_hybrid
}  // namespace flashinfer
