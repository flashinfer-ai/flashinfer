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

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "tvm_ffi_utils.h"

namespace flashinfer {
namespace flash_kda {

constexpr int64_t kHeadDim = 128;
constexpr size_t kTensorMapCount = 6;
constexpr size_t kTensorMapAlignment = 64;
static_assert(sizeof(CUtensorMap) == 128);
constexpr size_t kDescriptorStorageBytes = kTensorMapCount * sizeof(CUtensorMap);

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckCudaTensor(const TensorView& tensor, const char* name, int32_t device_id) {
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.device().device_id == device_id)
      << name << " must be on CUDA device " << device_id << ", got " << tensor.device().device_id;
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

inline void CheckDtype(const TensorView& tensor, const char* name, DLDataType expected) {
  const DLDataType actual = tensor.dtype();
  TVM_FFI_ICHECK(actual.code == expected.code && actual.bits == expected.bits &&
                 actual.lanes == expected.lanes)
      << name << " has wrong dtype: expected (code=" << int(expected.code)
      << ", bits=" << int(expected.bits) << ", lanes=" << int(expected.lanes)
      << "), got (code=" << int(actual.code) << ", bits=" << int(actual.bits)
      << ", lanes=" << int(actual.lanes) << ")";
}

struct TensorByteRange {
  uintptr_t begin;
  uintptr_t end;
};

inline TensorByteRange GetTensorByteRange(const TensorView& tensor, const char* name) {
  const DLDataType dtype = tensor.dtype();
  const uint64_t bits = static_cast<uint64_t>(dtype.bits) * static_cast<uint64_t>(dtype.lanes);
  TVM_FFI_ICHECK(bits > 0 && bits % 8 == 0) << name << " has a non-byte-addressable dtype";
  const uint64_t bytes = static_cast<uint64_t>(tensor.numel()) * (bits / 8);
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  TVM_FFI_ICHECK(bytes <= std::numeric_limits<uintptr_t>::max() - begin)
      << name << " byte range overflows uintptr_t";
  return {begin, begin + static_cast<uintptr_t>(bytes)};
}

inline void CheckNoOverlap(const TensorView& lhs, const char* lhs_name, const TensorView& rhs,
                           const char* rhs_name) {
  const TensorByteRange lhs_range = GetTensorByteRange(lhs, lhs_name);
  const TensorByteRange rhs_range = GetTensorByteRange(rhs, rhs_name);
  const bool overlaps = lhs_range.begin < rhs_range.end && rhs_range.begin < lhs_range.end;
  TVM_FFI_ICHECK(!overlaps) << lhs_name << " must not overlap " << rhs_name
                            << ": the frozen kernel uses __restrict__ pointers";
}

inline void CheckNoPartialOverlapOrExactAlias(const TensorView& lhs, const char* lhs_name,
                                              const TensorView& rhs, const char* rhs_name) {
  const TensorByteRange lhs_range = GetTensorByteRange(lhs, lhs_name);
  const TensorByteRange rhs_range = GetTensorByteRange(rhs, rhs_name);
  const bool overlaps = lhs_range.begin < rhs_range.end && rhs_range.begin < lhs_range.end;
  const bool exact_alias = lhs_range.begin == rhs_range.begin && lhs_range.end == rhs_range.end;
  TVM_FFI_ICHECK(!overlaps || exact_alias)
      << lhs_name << " and " << rhs_name
      << " must either be disjoint or exactly alias the same state storage";
}

inline void CheckExactSm100a(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == 0)
      << "FlashKDA frozen kernels require exact compute capability 10.0 "
         "(sm_100a), got "
      << major << "." << minor;
}

inline int64_t CheckCommonInputs(const TensorView& q, const TensorView& k, const TensorView& v,
                                 const TensorView& g, const TensorView& beta,
                                 const TensorView& beta_tma, const TensorView& A_log,
                                 const TensorView& dt_bias, const TensorView& cu_seqlens,
                                 const TensorView& seq_order, const TensorView& initial_state,
                                 const TensorView& out, const TensorView& final_state,
                                 const TensorView& descriptor_storage, int64_t prepare_descriptors,
                                 int64_t num_heads, int64_t use_initial_state,
                                 int64_t store_final_state, double scale, double lower_bound) {
  TVM_FFI_ICHECK(prepare_descriptors == 0 || prepare_descriptors == 1)
      << "prepare_descriptors must be 0 or 1, got " << prepare_descriptors;
  TVM_FFI_ICHECK(num_heads > 0 && num_heads <= std::numeric_limits<int32_t>::max())
      << "num_heads must be in the positive int32 range, got " << num_heads;
  TVM_FFI_ICHECK(use_initial_state == 0 || use_initial_state == 1)
      << "use_initial_state must be 0 or 1, got " << use_initial_state;
  TVM_FFI_ICHECK(store_final_state == 0 || store_final_state == 1)
      << "store_final_state must be 0 or 1, got " << store_final_state;
  TVM_FFI_ICHECK(std::isfinite(scale) && std::isfinite(static_cast<float>(scale)))
      << "scale must be finite and representable as float32, got " << scale;
  TVM_FFI_ICHECK(std::isfinite(lower_bound) && lower_bound < 0.0 &&
                 std::isfinite(static_cast<float>(lower_bound)))
      << "lower_bound must be finite, negative, and representable as "
         "float32, got "
      << lower_bound;

  const int32_t device_id = q.device().device_id;
  CheckCudaTensor(q, "q", device_id);
  CheckCudaTensor(k, "k", device_id);
  CheckCudaTensor(v, "v", device_id);
  CheckCudaTensor(g, "g", device_id);
  CheckCudaTensor(beta, "beta", device_id);
  CheckCudaTensor(beta_tma, "beta_tma", device_id);
  CheckCudaTensor(A_log, "A_log", device_id);
  CheckCudaTensor(dt_bias, "dt_bias", device_id);
  CheckCudaTensor(cu_seqlens, "cu_seqlens", device_id);
  CheckCudaTensor(seq_order, "seq_order", device_id);
  CheckCudaTensor(initial_state, "initial_state", device_id);
  CheckCudaTensor(out, "out", device_id);
  CheckCudaTensor(final_state, "final_state", device_id);
  CheckCudaTensor(descriptor_storage, "descriptor_storage", device_id);

  CheckDtype(q, "q", dl_bfloat16);
  CheckDtype(k, "k", dl_bfloat16);
  CheckDtype(v, "v", dl_bfloat16);
  CheckDtype(g, "g", dl_bfloat16);
  CheckDtype(beta, "beta", dl_bfloat16);
  CheckDtype(beta_tma, "beta_tma", dl_bfloat16);
  CheckDtype(A_log, "A_log", dl_float32);
  CheckDtype(dt_bias, "dt_bias", dl_float32);
  CheckDtype(cu_seqlens, "cu_seqlens", dl_int64);
  CheckDtype(seq_order, "seq_order", dl_int32);
  CheckDtype(initial_state, "initial_state", dl_bfloat16);
  CheckDtype(out, "out", dl_bfloat16);
  CheckDtype(final_state, "final_state", dl_bfloat16);
  CheckDtype(descriptor_storage, "descriptor_storage", dl_uint8);

  TVM_FFI_ICHECK(descriptor_storage.numel() >= static_cast<int64_t>(kDescriptorStorageBytes))
      << "descriptor_storage must contain at least " << kDescriptorStorageBytes << " bytes";
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(descriptor_storage.data_ptr()) % kTensorMapAlignment ==
                 0)
      << "descriptor_storage must be aligned to " << kTensorMapAlignment << " bytes";

  TVM_FFI_ICHECK(q.ndim() >= 3) << "q must have trailing [H, 128] dimensions";
  TVM_FFI_ICHECK(q.size(q.ndim() - 1) == kHeadDim && q.size(q.ndim() - 2) == num_heads)
      << "q must have trailing shape [" << num_heads << ", 128]";
  const int64_t token_count = q.numel() / (num_heads * kHeadDim);
  TVM_FFI_ICHECK(token_count > 0) << "q must contain at least one token";

  for (const auto& named : {std::pair<const TensorView*, const char*>(&k, "k"),
                            std::pair<const TensorView*, const char*>(&v, "v"),
                            std::pair<const TensorView*, const char*>(&g, "g"),
                            std::pair<const TensorView*, const char*>(&out, "out")}) {
    const TensorView& tensor = *named.first;
    TVM_FFI_ICHECK(tensor.ndim() >= 3 && tensor.size(tensor.ndim() - 1) == kHeadDim &&
                   tensor.size(tensor.ndim() - 2) == num_heads && tensor.numel() == q.numel())
        << named.second << " must match q's flattened [tokens, H, 128] shape";
  }

  TVM_FFI_ICHECK(beta.ndim() >= 2 && beta.size(beta.ndim() - 1) == num_heads &&
                 beta.numel() == token_count * num_heads)
      << "beta must match flattened [tokens, H]";
  const int64_t beta_tma_heads = std::max<int64_t>(num_heads, 8);
  TVM_FFI_ICHECK(beta_tma.ndim() >= 2 && beta_tma.size(beta_tma.ndim() - 1) == beta_tma_heads &&
                 beta_tma.numel() % beta_tma_heads == 0 &&
                 beta_tma.numel() / beta_tma_heads >= std::max<int64_t>(token_count, 32))
      << "beta_tma must have at least [max(tokens, 32), max(H, 8)] "
         "storage";
  TVM_FFI_ICHECK(A_log.numel() == num_heads) << "A_log must contain H elements";
  TVM_FFI_ICHECK(dt_bias.numel() == num_heads * kHeadDim)
      << "dt_bias must contain H * 128 elements";
  TVM_FFI_ICHECK(cu_seqlens.ndim() == 1 && cu_seqlens.numel() >= 2)
      << "cu_seqlens must be a one-dimensional tensor with at least two "
         "elements";

  const int64_t num_seqs = cu_seqlens.numel() - 1;
  TVM_FFI_ICHECK(seq_order.ndim() == 1 && seq_order.numel() == num_seqs)
      << "seq_order must contain one int32 entry per sequence";
  const int64_t state_numel = num_seqs * num_heads * kHeadDim * kHeadDim;
  if (use_initial_state != 0) {
    TVM_FFI_ICHECK(initial_state.numel() == state_numel)
        << "initial_state must have flattened [N, H, 128, 128] size";
  }
  if (store_final_state != 0) {
    TVM_FFI_ICHECK(final_state.numel() == state_numel)
        << "final_state must have flattened [N, H, 128, 128] size";
  }
  CheckNoOverlap(out, "out", q, "q");
  CheckNoOverlap(out, "out", k, "k");
  CheckNoOverlap(out, "out", v, "v");
  CheckNoOverlap(out, "out", g, "g");
  CheckNoOverlap(out, "out", beta, "beta");
  CheckNoOverlap(out, "out", beta_tma, "beta_tma");
  CheckNoOverlap(out, "out", A_log, "A_log");
  CheckNoOverlap(out, "out", dt_bias, "dt_bias");
  CheckNoOverlap(out, "out", cu_seqlens, "cu_seqlens");
  CheckNoOverlap(out, "out", seq_order, "seq_order");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", q, "q");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", k, "k");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", v, "v");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", g, "g");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", beta, "beta");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", beta_tma, "beta_tma");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", A_log, "A_log");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", dt_bias, "dt_bias");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", cu_seqlens, "cu_seqlens");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", seq_order, "seq_order");
  CheckNoOverlap(descriptor_storage, "descriptor_storage", out, "out");
  if (use_initial_state != 0) {
    CheckNoOverlap(out, "out", initial_state, "initial_state");
    CheckNoOverlap(descriptor_storage, "descriptor_storage", initial_state, "initial_state");
  }
  if (store_final_state != 0) {
    CheckNoOverlap(out, "out", final_state, "final_state");
    CheckNoOverlap(descriptor_storage, "descriptor_storage", final_state, "final_state");
    CheckNoOverlap(final_state, "final_state", q, "q");
    CheckNoOverlap(final_state, "final_state", k, "k");
    CheckNoOverlap(final_state, "final_state", v, "v");
    CheckNoOverlap(final_state, "final_state", g, "g");
    CheckNoOverlap(final_state, "final_state", beta, "beta");
    CheckNoOverlap(final_state, "final_state", beta_tma, "beta_tma");
    CheckNoOverlap(final_state, "final_state", A_log, "A_log");
    CheckNoOverlap(final_state, "final_state", dt_bias, "dt_bias");
    CheckNoOverlap(final_state, "final_state", cu_seqlens, "cu_seqlens");
    CheckNoOverlap(final_state, "final_state", seq_order, "seq_order");
    if (use_initial_state != 0) {
      CheckNoPartialOverlapOrExactAlias(initial_state, "initial_state", final_state, "final_state");
    }
  }
  return num_seqs;
}

inline CUtensorMap EncodeQkTma(const TensorView& tensor, const char* name) {
  TVM_FFI_ICHECK(tensor.ndim() >= 2) << name << " must have at least two dimensions";
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  TVM_FFI_ICHECK(d1 > 0 && d2 > 0 && d1 % 64 == 0) << name << " has invalid trailing dimensions";
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(outer2), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(d1 / 64)};
  TVM_FFI_ICHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] >= 1 && global_dim[3] >= 2)
      << name << " cannot encode the (64, 32, 1, 2) TMA box";
  uint64_t global_strides[3] = {static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, 32, 1, 2};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(),
                             global_dim, global_strides, box_dim, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for " << name << " with CUresult=" << int(result);
  return tensor_map;
}

template <int ValueRows>
inline CUtensorMap EncodeValueTma(const TensorView& tensor) {
  static_assert(ValueRows == 64 || ValueRows == 128);
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  uint64_t global_dim[3] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(outer2)};
  TVM_FFI_ICHECK(global_dim[0] >= ValueRows && global_dim[1] >= 1 && global_dim[2] > 0)
      << "v cannot encode the (" << ValueRows << ", 1, 32) TMA box";
  uint64_t global_strides[2] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[3] = {ValueRows, 1, 32};
  uint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap tensor_map{};
  constexpr CUtensorMapSwizzle swizzle =
      ValueRows == 64 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE;
  const CUresult result = cuTensorMapEncodeTiled(
      &tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, tensor.data_ptr(), global_dim,
      global_strides, box_dim, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for v with CUresult=" << int(result);
  return tensor_map;
}

inline CUtensorMap EncodeGateTma(const TensorView& tensor) {
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  uint64_t global_dim[3] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(outer2)};
  TVM_FFI_ICHECK(global_dim[0] >= 128 && global_dim[1] >= 1 && global_dim[2] > 0)
      << "g cannot encode the (128, 1, 32) TMA box";
  uint64_t global_strides[2] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * d2 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[3] = {128, 1, 32};
  uint32_t elem_strides[3] = {1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, tensor.data_ptr(),
                             global_dim, global_strides, box_dim, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for g with CUresult=" << int(result);
  return tensor_map;
}

inline CUtensorMap EncodeBetaTma(const TensorView& tensor) {
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t outer1 = tensor.numel() / d1;
  uint64_t global_dim[2] = {static_cast<uint64_t>(d1), static_cast<uint64_t>(outer1)};
  TVM_FFI_ICHECK(global_dim[0] >= 8 && global_dim[1] >= 32)
      << "beta_tma cannot encode the (8, 32) TMA box";
  uint64_t global_strides[1] = {static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[2] = {8, 32};
  uint32_t elem_strides[2] = {1, 1};
  CUtensorMap tensor_map{};
  const CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, tensor.data_ptr(),
                             global_dim, global_strides, box_dim, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for beta_tma with CUresult=" << int(result);
  return tensor_map;
}

template <int ValueRows>
inline CUtensorMap EncodeOutputTma(const TensorView& tensor) {
  static_assert(ValueRows == 64 || ValueRows == 128);
  const int64_t d1 = tensor.size(tensor.ndim() - 1);
  const int64_t d2 = tensor.size(tensor.ndim() - 2);
  const int64_t outer2 = tensor.numel() / (d1 * d2);
  TVM_FFI_ICHECK(d1 > 0 && d2 > 0 && d1 % 64 == 0) << "out has invalid trailing dimensions";
  uint64_t global_dim[4] = {64, static_cast<uint64_t>(outer2), static_cast<uint64_t>(d2),
                            static_cast<uint64_t>(d1 / 64)};
  constexpr uint32_t value_splits = ValueRows / 64;
  TVM_FFI_ICHECK(global_dim[0] >= 64 && global_dim[1] > 0 && global_dim[2] >= 1 &&
                 global_dim[3] >= value_splits)
      << "out cannot encode the (64, 32, 1, " << value_splits << ") TMA box";
  uint64_t global_strides[3] = {static_cast<uint64_t>(d2 * d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(d1 * sizeof(__nv_bfloat16)),
                                static_cast<uint64_t>(64 * sizeof(__nv_bfloat16))};
  uint32_t box_dim[4] = {64, 32, 1, value_splits};
  uint32_t elem_strides[4] = {1, 1, 1, 1};
  CUtensorMap tensor_map{};
  const CUresult result =
      cuTensorMapEncodeTiled(&tensor_map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, tensor.data_ptr(),
                             global_dim, global_strides, box_dim, elem_strides,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for out with CUresult=" << int(result);
  return tensor_map;
}

struct TmaPointers {
  void* q;
  void* k;
  void* v;
  void* g;
  void* beta;
  void* out;
};

struct TensorMapWords {
  static constexpr size_t kWordCount = kDescriptorStorageBytes / sizeof(uint64_t);
  uint64_t words[kWordCount];
};

static __global__ void PublishTensorMaps(uint64_t* destination, TensorMapWords source) {
  const uint32_t index = threadIdx.x;
  if (index < TensorMapWords::kWordCount) {
    destination[index] = source.words[index];
  }
}

template <int ValueRows>
inline TmaPointers EncodeTmaPointers(const TensorView& q, const TensorView& k, const TensorView& v,
                                     const TensorView& g, const TensorView& beta_tma,
                                     const TensorView& out, const TensorView& descriptor_storage,
                                     int64_t prepare_descriptors, cudaStream_t stream) {
  if (prepare_descriptors != 0) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CheckCuda(cudaStreamIsCapturing(stream, &capture_status),
              "cudaStreamIsCapturing(TMA descriptor preparation)");
    TVM_FFI_ICHECK(capture_status == cudaStreamCaptureStatusNone)
        << "prepare_descriptors must be 0 during CUDA graph capture; warm "
           "this exact workspace and tensor signature before capture";

    const std::array<CUtensorMap, kTensorMapCount> host_maps = {
        EncodeQkTma(q, "q"), EncodeQkTma(k, "k"),     EncodeValueTma<ValueRows>(v),
        EncodeGateTma(g),    EncodeBetaTma(beta_tma), EncodeOutputTma<ValueRows>(out),
    };
    static_assert(sizeof(host_maps) == kDescriptorStorageBytes);
    TensorMapWords words{};
    std::memcpy(words.words, host_maps.data(), sizeof(host_maps));
    PublishTensorMaps<<<1, 128, 0, stream>>>(
        reinterpret_cast<uint64_t*>(descriptor_storage.data_ptr()), words);
    CheckCuda(cudaGetLastError(), "PublishTensorMaps launch");
  }

  auto* bytes = static_cast<unsigned char*>(descriptor_storage.data_ptr());
  constexpr size_t stride = sizeof(CUtensorMap);
  return {
      bytes + 0 * stride, bytes + 1 * stride, bytes + 2 * stride,
      bytes + 3 * stride, bytes + 4 * stride, bytes + 5 * stride,
  };
}

inline void CheckDynamicSmemCapacity(int32_t device_id, int32_t smem_bytes) {
  int max_optin = 0;
  CheckCuda(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id),
            "cudaDeviceGetAttribute(max dynamic shared memory)");
  TVM_FFI_ICHECK(max_optin >= smem_bytes)
      << "device exposes only " << max_optin << " bytes of opt-in shared memory; FlashKDA requires "
      << smem_bytes;
}

}  // namespace flash_kda
}  // namespace flashinfer
