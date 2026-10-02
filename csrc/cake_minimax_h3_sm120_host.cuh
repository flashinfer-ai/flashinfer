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
// clang-format off
// Host-side helpers shared by the MiniMax-H3 SM120 (GB202) operators: tensor checks, tensor-map encoders,
// the per-device configuration cache and the persistent-grid launch plan.  Host code only -- including this
// header adds no device code to a translation unit.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <utility>
#include <vector>

#include "tvm_ffi_utils.h"

namespace minimax_h3_sm120 {

// ``tensor`` is a 16-byte aligned CUDA tensor on the device of the operator's ``anchor`` argument.
inline void CheckDevice(const TensorView& tensor, const char* name, DLDevice device, const char* anchor) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as " << anchor;
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, ValueError)
      << name << " must be 16-byte aligned";
}

// Dense tensor of ``dtype`` with exactly ``shape`` (unit-sized dimensions may carry any stride).
inline void CheckTensor(const TensorView& tensor, const char* name, DLDevice device, DLDataType dtype,
                        std::initializer_list<int64_t> shape, const char* anchor) {
  TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA, ValueError) << name << " must be a CUDA tensor";
  TVM_FFI_CHECK(tensor.device().device_id == device.device_id, ValueError)
      << name << " must be on the same CUDA device as " << anchor;
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " has the wrong dtype";
  TVM_FFI_CHECK(tensor.ndim() == static_cast<int>(shape.size()), ValueError)
      << name << " must have " << shape.size() << " dimensions";
  int64_t expected_stride = 1;
  int dim = tensor.ndim() - 1;
  for (auto it = std::rbegin(shape); it != std::rend(shape); ++it, --dim) {
    TVM_FFI_CHECK(tensor.size(dim) == *it, ValueError) << name << " has the wrong shape (dimension " << dim << ")";
    TVM_FFI_CHECK(tensor.size(dim) == 1 || tensor.stride(dim) == expected_stride, ValueError)
        << name << " must be contiguous";
    expected_stride *= *it;
  }
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0, ValueError)
      << name << " must be 16-byte aligned";
}

// Packed-THD operand: contiguous bfloat16 ``[tokens, heads, head_dim]``.
inline void CheckThd(const TensorView& tensor, const char* name, int64_t tokens, int64_t heads, int64_t head_dim,
                     DLDevice device, const char* anchor) {
  CheckDevice(tensor, name, device, anchor);
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dl_bfloat16), ValueError)
      << name << " must be bfloat16";
  TVM_FFI_CHECK(tensor.ndim() == 3 && tensor.size(0) == tokens && tensor.size(1) == heads &&
                    tensor.size(2) == head_dim,
                ValueError)
      << name << " must have shape [tokens, heads, " << head_dim << "]";
  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError) << name << " must be contiguous";
}

// Contiguous 1-D buffer of ``dtype`` with at least ``min_numel`` elements (workspaces and plan tables).
inline void CheckFlat(const TensorView& tensor, const char* name, DLDataType dtype, const char* dtype_name,
                      int64_t min_numel, DLDevice device, const char* anchor) {
  CheckDevice(tensor, name, device, anchor);
  TVM_FFI_CHECK(encode_dlpack_dtype(tensor.dtype()) == encode_dlpack_dtype(dtype), ValueError)
      << name << " must be " << dtype_name;
  TVM_FFI_CHECK(tensor.ndim() == 1 && tensor.IsContiguous() && tensor.size(0) >= min_numel, ValueError)
      << name << " must be a contiguous 1-D " << dtype_name << " tensor with at least " << min_numel
      << " elements";
}

// Tiled UINT8 tensor map with unit element strides and TMA zero fill outside the tensor.
inline CUtensorMap EncodeUint8Tiled(const void* base, uint32_t rank, const uint64_t* global_dim,
                                    const uint64_t* global_strides, const uint32_t* box_dim,
                                    CUtensorMapSwizzle swizzle, const char* name) {
  uint32_t element_strides[3] = {1, 1, 1};
  CUtensorMap descriptor{};
  CUresult result = cuTensorMapEncodeTiled(
      &descriptor, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, const_cast<void*>(base), global_dim, global_strides,
      box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(result == CUDA_SUCCESS, RuntimeError)
      << "failed to encode the " << name << " tensor map: CUresult=" << static_cast<int>(result);
  return descriptor;
}

// 2-D byte tile map: rows of ``inner_bytes`` contiguous bytes, box = ``box_rows`` x ``box_inner`` bytes.
// Rows beyond the tensor are zero-filled by TMA (partial M tail tiles).
inline CUtensorMap EncodeByteTile(const void* base, int64_t inner_bytes, int64_t rows, uint32_t box_inner,
                                  uint32_t box_rows, CUtensorMapSwizzle swizzle, const char* name) {
  uint64_t global_dim[2] = {static_cast<uint64_t>(inner_bytes), static_cast<uint64_t>(rows)};
  uint64_t global_strides[1] = {static_cast<uint64_t>(inner_bytes)};
  uint32_t box_dim[2] = {box_inner, box_rows};
  return EncodeUint8Tiled(base, 2, global_dim, global_strides, box_dim, swizzle, name);
}

// One head's [``row_bytes`` x ``box_rows`` tokens] tile of a [tokens, heads, row_bytes] byte tensor, addressed
// as the 3-D view (row bytes, tokens, heads).  The token box may run past the tensor: TMA zero-fills those
// rows, the kernel masks those keys / never stores those query rows.
inline CUtensorMap EncodeHeadRowsTile(const void* base, int64_t row_bytes, int64_t tokens, int64_t heads,
                                      uint32_t box_rows, CUtensorMapSwizzle swizzle, const char* name) {
  uint64_t global_dim[3] = {static_cast<uint64_t>(row_bytes), static_cast<uint64_t>(tokens),
                            static_cast<uint64_t>(heads)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(heads * row_bytes), static_cast<uint64_t>(row_bytes)};
  uint32_t box_dim[3] = {static_cast<uint32_t>(row_bytes), box_rows, 1};
  return EncodeUint8Tiled(base, 3, global_dim, global_strides, box_dim, swizzle, name);
}

// One head's [``head_dim`` channel rows x ``box_inner`` bytes] tile of a transposed V^T byte tensor
// [heads, head_dim, row_bytes]: 3-D view (row bytes, channels, heads).
inline CUtensorMap EncodeTransposedHeadTile(const void* base, int64_t row_bytes, int64_t head_dim, int64_t heads,
                                            uint32_t box_inner, CUtensorMapSwizzle swizzle, const char* name) {
  uint64_t global_dim[3] = {static_cast<uint64_t>(row_bytes), static_cast<uint64_t>(head_dim),
                            static_cast<uint64_t>(heads)};
  uint64_t global_strides[2] = {static_cast<uint64_t>(row_bytes), static_cast<uint64_t>(row_bytes * head_dim)};
  uint32_t box_dim[3] = {box_inner, static_cast<uint32_t>(head_dim), 1};
  return EncodeUint8Tiled(base, 3, global_dim, global_strides, box_dim, swizzle, name);
}

// Dynamic shared memory opt-in of one kernel (above the 48 KB default).
template <typename Kernel>
inline void OptInDynamicSmem(Kernel* kernel, int bytes) {
  cudaError_t status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "failed to opt in to dynamic shared memory: " << cudaGetErrorString(status);
}

// Per-device configuration, resolved once per (module, device): the GB202 check, the module's dynamic
// shared memory opt-ins and whatever ``init`` derives from the device properties (SM count, occupancy).
// ``Tag`` is a type private to the including translation unit, so every module owns its own cache: the
// opt-ins are per module and must not be skipped because another module configured the device.
template <class Tag, class Info>
struct DeviceConfig {
  template <class Init>
  static Info Get(const char* what, Init&& init) {
    static std::mutex mutex;
    static std::vector<std::pair<int, Info>> configured;
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
        << "failed to get the active CUDA device: " << cudaGetErrorString(status);
    std::lock_guard<std::mutex> lock(mutex);
    for (const auto& entry : configured) {
      if (entry.first == device) return entry.second;
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
        << "failed to query CUDA device properties: " << cudaGetErrorString(status);
    TVM_FFI_CHECK(properties.major == 12, RuntimeError)
        << what << " requires compute capability 12.x (GB202); got " << properties.major << "." << properties.minor;
    Info info = init(properties);
    configured.emplace_back(device, info);
    return info;
  }
};

// Persistent GEMM kernel table entry: the instantiation and its dynamic shared memory.
template <typename Kernel>
struct GemmVariant {
  Kernel kernel;
  int dynamic_smem_bytes;
};

// Grids of the quantization launch (``quant_ctas_per_sm`` CTAs per SM, at most one per row) and of the
// persistent GEMM (one CTA per SM, at most one per tile).  Mirrors the Python launch_plan().
struct GemmLaunchPlan {
  int quant_grid;
  int gemm_grid;
  int num_m_tiles;
  int total_tiles;
};

inline GemmLaunchPlan MakeGemmLaunchPlan(int64_t rows, int num_sms, int block_m, int n_tiles,
                                         int quant_ctas_per_sm) {
  const int64_t num_m_tiles = (rows + block_m - 1) / block_m;
  const int64_t total_tiles = num_m_tiles * n_tiles;
  GemmLaunchPlan plan{};
  plan.quant_grid = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(rows, static_cast<int64_t>(quant_ctas_per_sm) * num_sms)));
  plan.gemm_grid = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(total_tiles, num_sms)));
  plan.num_m_tiles = static_cast<int>(num_m_tiles);
  plan.total_tiles = static_cast<int>(total_tiles);
  return plan;
}

}  // namespace minimax_h3_sm120
// clang-format on
