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

// Tensor-map encoders of the Cake DSv4 NVFP4 prefill kernels; the launch helpers come from
// cake_dsv4_nvfp4_host_common.cuh.
#pragma once
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#include <cstdint>

namespace cake_host_shim_common {

using tvm::ffi::TensorView;

// Tensor-map encoders of the Cake DSv4 NVFP4 prefill kernels (nvfp4_h64_prefill, nvfp4_h128_prefill).
// Each follows the descriptor record of the generated kernel: the trailing dims of the source tensor
// are the map's extents, the leading dims fold into one axis where the map has one.

inline void* PrefillTensorBase(const TensorView& t, const char* name) {
  void* address = static_cast<char*>(t.data_ptr()) + t.byte_offset();
  TVM_FFI_CHECK(reinterpret_cast<uintptr_t>(address) % 16 == 0, ValueError)
      << "TMA source '" << name << "' must be 16-byte aligned";
  return address;
}

inline int64_t PrefillNumelAbove(const TensorView& t, int trailing) {
  int64_t product = 1;
  for (int axis = 0; axis < t.ndim() - trailing; ++axis) product *= t.size(axis);
  return product;
}

inline CUtensorMap PrefillEncodeTiled(const char* name, CUtensorMapDataType dtype, cuuint32_t rank,
                                      void* address, const cuuint64_t* global_dim,
                                      const cuuint64_t* global_strides, const cuuint32_t* box_dim,
                                      CUtensorMapSwizzle swizzle) {
  for (cuuint32_t i = 0; i < rank; ++i) {
    TVM_FFI_CHECK(global_dim[i] > 0, ValueError)
        << "TMA descriptor for '" << name << "' resolved a non-positive global dim";
  }
  for (cuuint32_t i = 0; i + 1 < rank; ++i) {
    TVM_FFI_CHECK(global_strides[i] % 16 == 0, ValueError)
        << "the strides of descriptor '" << name << "' must be multiples of 16 bytes";
  }
  cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap map{};
  CUresult r = cuTensorMapEncodeTiled(&map, dtype, rank, address, global_dim, global_strides, box_dim,
                                      elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (" << rank << "D, '" << name << "') failed: CUresult=" << (int)r;
  return map;
}

// The packed NVFP4 query [tokens, heads, 384] (uint8): a 3-D map of (bytes, heads, tokens).
inline CUtensorMap PrefillEncodeQueryRows(const TensorView& t, const char* name, cuuint32_t box_bytes,
                                          CUtensorMapSwizzle swizzle) {
  TVM_FFI_CHECK(t.ndim() >= 3, ValueError)
      << "TMA source '" << name << "' must have at least 3 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(-1);
  int64_t d2 = t.size(-2);
  int64_t d3 = t.size(-3);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0 && d3 > 0, ValueError)
      << "TMA source '" << name << "' trailing dims must be positive";
  cuuint64_t global_dim[3] = {(cuuint64_t)d1, (cuuint64_t)d2, (cuuint64_t)d3};
  cuuint64_t global_strides[2] = {(cuuint64_t)d1, (cuuint64_t)(d1 * d2)};
  cuuint32_t box_dim[3] = {box_bytes, 64u, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[2] <= global_dim[2], ValueError)
      << "the box of descriptor '" << name << "' exceeds its source extents";
  return PrefillEncodeTiled(name, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, PrefillTensorBase(t, name), global_dim,
                            global_strides, box_dim, swizzle);
}

inline CUtensorMap EncodeTmaPrefill_tmap_q(const TensorView& t) {
  return PrefillEncodeQueryRows(t, "tmap_q", 128u, CU_TENSOR_MAP_SWIZZLE_128B);
}

inline CUtensorMap EncodeTmaPrefill_tmap_q_sf(const TensorView& t) {
  return PrefillEncodeQueryRows(t, "tmap_q_sf", 32u, CU_TENSOR_MAP_SWIZZLE_NONE);
}

// A pool row view [rows, row_bytes] on a 32-byte pitch (the 352-byte data rows or the 32-byte scale rows of
// a packed NVFP4 pool): a 2-D map of (row_bytes, rows).
inline CUtensorMap PrefillEncodePoolRows(const TensorView& t, const char* name, cuuint32_t box_bytes,
                                         CUtensorMapSwizzle swizzle) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source '" << name << "' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  check_dense_leading_dims(t, 1, name);
  int64_t d1 = t.size(-1);
  int64_t outer1 = PrefillNumelAbove(t, 1);
  int64_t s2 = t.stride(-2);
  TVM_FFI_CHECK(d1 > 0 && outer1 > 0 && s2 > 0, ValueError)
      << "TMA source '" << name << "' extents and row pitch must be positive";
  cuuint64_t global_dim[2] = {(cuuint64_t)d1, (cuuint64_t)outer1};
  cuuint64_t global_strides[1] = {(cuuint64_t)s2};
  cuuint32_t box_dim[2] = {box_bytes, 1u};
  TVM_FFI_CHECK(box_dim[0] <= global_dim[0] && box_dim[1] <= global_dim[1], ValueError)
      << "the box of descriptor '" << name << "' exceeds its source extents";
  return PrefillEncodeTiled(name, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, PrefillTensorBase(t, name), global_dim,
                            global_strides, box_dim, swizzle);
}

inline CUtensorMap EncodeTmaPrefill_tmap_swa_kv(const TensorView& t) {
  return PrefillEncodePoolRows(t, "tmap_swa_kv", 128u, CU_TENSOR_MAP_SWIZZLE_128B);
}

inline CUtensorMap EncodeTmaPrefill_tmap_compressed_kv(const TensorView& t) {
  return PrefillEncodePoolRows(t, "tmap_compressed_kv", 128u, CU_TENSOR_MAP_SWIZZLE_128B);
}

inline CUtensorMap EncodeTmaPrefill_tmap_swa_sf(const TensorView& t) {
  return PrefillEncodePoolRows(t, "tmap_swa_sf", 32u, CU_TENSOR_MAP_SWIZZLE_NONE);
}

inline CUtensorMap EncodeTmaPrefill_tmap_compressed_sf(const TensorView& t) {
  return PrefillEncodePoolRows(t, "tmap_compressed_sf", 32u, CU_TENSOR_MAP_SWIZZLE_NONE);
}

// The BF16 output rows [..., 512]: a 3-D store map of (32 elements, rows, 512 / 32 column blocks),
// 64-byte swizzle, one 32 x 32 box per store.
inline CUtensorMap EncodeTmaPrefill_tmap_o(const TensorView& t) {
  const char* name = "tmap_o";
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source '" << name << "' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source '" << name << "' must have unit innermost stride, got " << t.stride(-1);
  check_dense_leading_dims(t, 1, name);
  int64_t d1 = t.size(-1);
  int64_t outer1 = PrefillNumelAbove(t, 1);
  int64_t s2 = t.stride(-2) * 2;
  TVM_FFI_CHECK(d1 > 0 && d1 % 32 == 0 && outer1 > 0 && s2 > 0, ValueError)
      << "TMA source '" << name << "' needs a positive row count and a row length that is a multiple of 32";
  cuuint64_t global_dim[3] = {32u, (cuuint64_t)outer1, (cuuint64_t)(d1 / 32)};
  cuuint64_t global_strides[2] = {(cuuint64_t)s2, 64u};
  cuuint32_t box_dim[3] = {32u, 32u, 1u};
  TVM_FFI_CHECK(box_dim[1] <= global_dim[1], ValueError)
      << "the box of descriptor '" << name << "' exceeds its source extents";
  return PrefillEncodeTiled(name, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, PrefillTensorBase(t, name), global_dim,
                            global_strides, box_dim, CU_TENSOR_MAP_SWIZZLE_64B);
}

}  // namespace cake_host_shim_common
