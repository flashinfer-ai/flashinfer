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

#include <cuda.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

#include "tvm_ffi_utils.h"

#define uint8_t blackwell_msa_generated_uint8_t
#define uint16_t blackwell_msa_generated_uint16_t
#define uint32_t blackwell_msa_generated_uint32_t
#define uint64_t blackwell_msa_generated_uint64_t
#define int32_t blackwell_msa_generated_int32_t
#define int16_t blackwell_msa_generated_int16_t
#define CUtensorMap BlackwellMsaGeneratedTensorMap
#include "blackwell_msa_long_prefill_paged_bf16_gqa8_sm103.cu"
#undef CUtensorMap
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

static_assert(sizeof(BlackwellMsaGeneratedTensorMap) == sizeof(CUtensorMap));

namespace flashinfer::blackwell_msa {

using tvm::ffi::TensorView;

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

#if !defined(FLASHINFER_BLACKWELL_MSA_TARGET_MINOR)
#error "the exact Blackwell MSA target minor must be defined"
#endif

inline void CheckBlackwellMsaTarget(int32_t device_id) {
  int major = 0;
  int minor = 0;
  cudaError_t status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaDeviceGetAttribute(major) failed: " << cudaGetErrorString(status);
  status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaDeviceGetAttribute(minor) failed: " << cudaGetErrorString(status);
  TVM_FFI_CHECK(major == 10 && minor == FLASHINFER_BLACKWELL_MSA_TARGET_MINOR, RuntimeError)
      << "this Blackwell MSA module requires compute capability 10."
      << FLASHINFER_BLACKWELL_MSA_TARGET_MINOR << ", got " << major << "." << minor;
}

inline int64_t HostCheckedExtentValue(int64_t value, const char* formula) {
  TVM_FFI_CHECK(value >= 0, ValueError)
      << "host extent " << formula << " must resolve inside [0, INT64_MAX], got " << value;
  return value;
}

inline int64_t HostCheckedExtentMul(int64_t lhs, int64_t rhs, const char* formula) {
  TVM_FFI_CHECK(lhs >= 0 && rhs >= 0 && (rhs == 0 || lhs <= INT64_MAX / rhs), ValueError)
      << "host extent overflow while evaluating " << formula;
  return lhs * rhs;
}

inline int64_t HostCheckedExtentAdd(int64_t lhs, int64_t rhs, const char* formula) {
  TVM_FFI_CHECK(lhs >= 0 && rhs >= 0 && lhs <= INT64_MAX - rhs, ValueError)
      << "host extent overflow while evaluating " << formula;
  return lhs + rhs;
}

// 2D TMA descriptor for buffer 'q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'q' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'q' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "q");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'q' physical strides must be positive";
  uint64_t global_dim[2] = {(uint64_t)(d1), (uint64_t)(outer1)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0, ValueError)
      << "TMA descriptor for 'q' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[1], ValueError)
      << "TMA box (64, 1) exceeds resolved global dims for 'q'";
  uint64_t global_strides[1] = {
      (uint64_t)((s2 * 16) / 8),
  };
  uint32_t box_dim[2] = {64u, 1u};
  uint32_t elem_strides[2] = {1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (2D, 'q') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'k' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_k(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'k' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'k' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'k' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'k' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d2), (uint64_t)((d1 / 64)), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'k' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 64, 1, 1) exceeds resolved global dims for 'k'";
  uint64_t global_strides[3] = {
      (uint64_t)((d1 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
      (uint64_t)(((d2 * d1) * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'k') failed: CUresult=" << (int)r;
  return tm;
}

// 4D TMA descriptor for buffer 'v' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_v(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'v' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'v' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'v' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'v' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[4] = {(uint64_t)(64), (uint64_t)(d2), (uint64_t)((d1 / 64)), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0 && global_dim[3] > 0, ValueError)
      << "TMA descriptor for 'v' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 1u <= global_dim[2] && 1u <= global_dim[3], ValueError)
      << "TMA box (64, 64, 1, 1) exceeds resolved global dims for 'v'";
  uint64_t global_strides[3] = {
      (uint64_t)((d1 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
      (uint64_t)(((d2 * d1) * 16) / 8),
  };
  uint32_t box_dim[4] = {64u, 64u, 1u, 1u};
  uint32_t elem_strides[4] = {1u, 1u, 1u, 1u};
  CUtensorMap tm{};
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (4D, 'v') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_q, TensorView arg_k, TensorView arg_v, TensorView arg_scheduler_metadata, TensorView arg_k2q_row_ptr, TensorView arg_k2q_qsplit_indices, TensorView arg_partial_o, TensorView arg_partial_scale, TensorView arg_partial_lse, TensorView arg_partial_temperature_lse, TensorView arg_out, TensorView arg_cu_seqlens_q, TensorView arg_cu_seqlens_k, TensorView arg_q_offsets, TensorView arg_kv_lens, TensorView arg_page_table, int64_t arg_q_group_segment_end_128, int64_t arg_q_group_segment_end_64, int64_t arg_q_group_segment_end_32, int64_t arg_q_group_segment_end_16, int64_t arg_q_group_segment_end_8, int64_t arg_q_group_segment_end_4, int64_t arg_q_group_segment_end_2, int64_t arg_total_q, int64_t arg_num_q_heads, int64_t arg_num_kv_heads, int64_t arg_total_rows, int64_t arg_nnz_per_head, int64_t arg_work_capacity, int64_t arg_num_work_items, int64_t arg_topk, int64_t arg_max_pages, int64_t arg_causal, int64_t arg_derive_q_offset, double arg_softmax_scale_log2, double arg_lse_temperature_scale, int64_t arg_return_temperature_lse, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_CHECK(cuda_stream >= 0, ValueError) << "cuda_stream must be non-negative";
  CheckCudaTensor(arg_q, "q");
  ffi::CUDADeviceGuard device_guard(arg_q.device().device_id);
  CheckBlackwellMsaTarget(arg_q.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckDtype(arg_q, "q", 4, 16, 1);
  CheckCudaTensor(arg_k, "k");
  CheckDtype(arg_k, "k", 4, 16, 1);
  CheckContiguous(arg_k, "k");
  CheckCudaTensor(arg_v, "v");
  CheckDtype(arg_v, "v", 4, 16, 1);
  CheckContiguous(arg_v, "v");
  CheckCudaTensor(arg_scheduler_metadata, "scheduler_metadata");
  CheckDtype(arg_scheduler_metadata, "scheduler_metadata", 0, 32, 1);
  CheckContiguous(arg_scheduler_metadata, "scheduler_metadata");
  CheckCudaTensor(arg_k2q_row_ptr, "k2q_row_ptr");
  CheckDtype(arg_k2q_row_ptr, "k2q_row_ptr", 0, 32, 1);
  CheckContiguous(arg_k2q_row_ptr, "k2q_row_ptr");
  CheckCudaTensor(arg_k2q_qsplit_indices, "k2q_qsplit_indices");
  CheckDtype(arg_k2q_qsplit_indices, "k2q_qsplit_indices", 0, 32, 1);
  CheckContiguous(arg_k2q_qsplit_indices, "k2q_qsplit_indices");
  CheckCudaTensor(arg_partial_o, "partial_o");
  CheckDtype(arg_partial_o, "partial_o", 1, 8, 1);
  CheckContiguous(arg_partial_o, "partial_o");
  CheckCudaTensor(arg_partial_scale, "partial_scale");
  CheckDtype(arg_partial_scale, "partial_scale", 4, 16, 1);
  CheckContiguous(arg_partial_scale, "partial_scale");
  CheckCudaTensor(arg_partial_lse, "partial_lse");
  CheckDtype(arg_partial_lse, "partial_lse", 2, 32, 1);
  CheckContiguous(arg_partial_lse, "partial_lse");
  CheckCudaTensor(arg_partial_temperature_lse, "partial_temperature_lse");
  CheckDtype(arg_partial_temperature_lse, "partial_temperature_lse", 2, 32, 1);
  CheckContiguous(arg_partial_temperature_lse, "partial_temperature_lse");
  CheckCudaTensor(arg_out, "out");
  CheckDtype(arg_out, "out", 4, 16, 1);
  CheckContiguous(arg_out, "out");
  CheckCudaTensor(arg_cu_seqlens_q, "cu_seqlens_q");
  CheckDtype(arg_cu_seqlens_q, "cu_seqlens_q", 0, 32, 1);
  CheckContiguous(arg_cu_seqlens_q, "cu_seqlens_q");
  CheckCudaTensor(arg_cu_seqlens_k, "cu_seqlens_k");
  CheckDtype(arg_cu_seqlens_k, "cu_seqlens_k", 0, 32, 1);
  CheckContiguous(arg_cu_seqlens_k, "cu_seqlens_k");
  CheckCudaTensor(arg_q_offsets, "q_offsets");
  CheckDtype(arg_q_offsets, "q_offsets", 0, 32, 1);
  CheckContiguous(arg_q_offsets, "q_offsets");
  CheckCudaTensor(arg_kv_lens, "kv_lens");
  CheckDtype(arg_kv_lens, "kv_lens", 0, 32, 1);
  CheckContiguous(arg_kv_lens, "kv_lens");
  CheckCudaTensor(arg_page_table, "page_table");
  CheckDtype(arg_page_table, "page_table", 0, 32, 1);
  CheckContiguous(arg_page_table, "page_table");
  TVM_FFI_CHECK(arg_q_group_segment_end_128 >= -2147483648LL && arg_q_group_segment_end_128 <= 2147483647LL, ValueError)
      << "scalar 'q_group_segment_end_128' value " << arg_q_group_segment_end_128
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_q_group_segment_end_64 >= -2147483648LL && arg_q_group_segment_end_64 <= 2147483647LL, ValueError)
      << "scalar 'q_group_segment_end_64' value " << arg_q_group_segment_end_64
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_q_group_segment_end_32 >= -2147483648LL && arg_q_group_segment_end_32 <= 2147483647LL, ValueError)
      << "scalar 'q_group_segment_end_32' value " << arg_q_group_segment_end_32
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_q_group_segment_end_16 >= -2147483648LL && arg_q_group_segment_end_16 <= 2147483647LL, ValueError)
      << "scalar 'q_group_segment_end_16' value " << arg_q_group_segment_end_16
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_q_group_segment_end_8 >= -2147483648LL && arg_q_group_segment_end_8 <= 2147483647LL, ValueError)
      << "scalar 'q_group_segment_end_8' value " << arg_q_group_segment_end_8
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_q_group_segment_end_4 >= -2147483648LL && arg_q_group_segment_end_4 <= 2147483647LL, ValueError)
      << "scalar 'q_group_segment_end_4' value " << arg_q_group_segment_end_4
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_q_group_segment_end_2 >= -2147483648LL && arg_q_group_segment_end_2 <= 2147483647LL, ValueError)
      << "scalar 'q_group_segment_end_2' value " << arg_q_group_segment_end_2
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_q >= -2147483648LL && arg_total_q <= 2147483647LL, ValueError)
      << "scalar 'total_q' value " << arg_total_q
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_q_heads >= -2147483648LL && arg_num_q_heads <= 2147483647LL, ValueError)
      << "scalar 'num_q_heads' value " << arg_num_q_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_kv_heads >= -2147483648LL && arg_num_kv_heads <= 2147483647LL, ValueError)
      << "scalar 'num_kv_heads' value " << arg_num_kv_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_rows >= -2147483648LL && arg_total_rows <= 2147483647LL, ValueError)
      << "scalar 'total_rows' value " << arg_total_rows
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_nnz_per_head >= -2147483648LL && arg_nnz_per_head <= 2147483647LL, ValueError)
      << "scalar 'nnz_per_head' value " << arg_nnz_per_head
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_work_capacity >= -2147483648LL && arg_work_capacity <= 2147483647LL, ValueError)
      << "scalar 'work_capacity' value " << arg_work_capacity
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_work_items >= -2147483648LL && arg_num_work_items <= 2147483647LL, ValueError)
      << "scalar 'num_work_items' value " << arg_num_work_items
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_topk >= -2147483648LL && arg_topk <= 2147483647LL, ValueError)
      << "scalar 'topk' value " << arg_topk
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_max_pages >= -2147483648LL && arg_max_pages <= 2147483647LL, ValueError)
      << "scalar 'max_pages' value " << arg_max_pages
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_causal >= -2147483648LL && arg_causal <= 2147483647LL, ValueError)
      << "scalar 'causal' value " << arg_causal
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_derive_q_offset >= -2147483648LL && arg_derive_q_offset <= 2147483647LL, ValueError)
      << "scalar 'derive_q_offset' value " << arg_derive_q_offset
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_return_temperature_lse >= -2147483648LL && arg_return_temperature_lse <= 2147483647LL, ValueError)
      << "scalar 'return_temperature_lse' value " << arg_return_temperature_lse
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_k, arg_q, "k", "q");
  CheckSameCudaDevice(arg_v, arg_q, "v", "q");
  CheckSameCudaDevice(arg_scheduler_metadata, arg_q, "scheduler_metadata", "q");
  CheckSameCudaDevice(arg_k2q_row_ptr, arg_q, "k2q_row_ptr", "q");
  CheckSameCudaDevice(arg_k2q_qsplit_indices, arg_q, "k2q_qsplit_indices", "q");
  CheckSameCudaDevice(arg_partial_o, arg_q, "partial_o", "q");
  CheckSameCudaDevice(arg_partial_scale, arg_q, "partial_scale", "q");
  CheckSameCudaDevice(arg_partial_lse, arg_q, "partial_lse", "q");
  CheckSameCudaDevice(arg_partial_temperature_lse, arg_q, "partial_temperature_lse", "q");
  CheckSameCudaDevice(arg_out, arg_q, "out", "q");
  CheckSameCudaDevice(arg_cu_seqlens_q, arg_q, "cu_seqlens_q", "q");
  CheckSameCudaDevice(arg_cu_seqlens_k, arg_q, "cu_seqlens_k", "q");
  CheckSameCudaDevice(arg_q_offsets, arg_q, "q_offsets", "q");
  CheckSameCudaDevice(arg_kv_lens, arg_q, "kv_lens", "q");
  CheckSameCudaDevice(arg_page_table, arg_q, "page_table", "q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";
  TVM_FFI_CHECK(arg_q.ndim() == 3, ValueError)
      << "q must have rank 3, got " << arg_q.ndim();
  int64_t host_extent_0 = 128;
  TVM_FFI_CHECK(arg_q.size(2) == host_extent_0, ValueError)
      << "q dimension 2 must be equal " << host_extent_0      << ", got " << arg_q.size(2);
  TVM_FFI_CHECK(arg_k.ndim() == 4, ValueError)
      << "k must have rank 4, got " << arg_k.ndim();
  int64_t host_extent_1 = 128;
  TVM_FFI_CHECK(arg_k.size(2) >= host_extent_1, ValueError)
      << "k dimension 2 must be at least " << host_extent_1      << ", got " << arg_k.size(2);
  int64_t host_extent_2 = 128;
  TVM_FFI_CHECK(arg_k.size(3) >= host_extent_2, ValueError)
      << "k dimension 3 must be at least " << host_extent_2      << ", got " << arg_k.size(3);
  TVM_FFI_CHECK(arg_v.ndim() == 4, ValueError)
      << "v must have rank 4, got " << arg_v.ndim();
  int64_t host_extent_3 = 128;
  TVM_FFI_CHECK(arg_v.size(2) >= host_extent_3, ValueError)
      << "v dimension 2 must be at least " << host_extent_3      << ", got " << arg_v.size(2);
  int64_t host_extent_4 = 128;
  TVM_FFI_CHECK(arg_v.size(3) >= host_extent_4, ValueError)
      << "v dimension 3 must be at least " << host_extent_4      << ", got " << arg_v.size(3);
  int64_t host_extent_5 = 128;
  host_extent_5 = HostCheckedExtentMul(host_extent_5, HostCheckedExtentValue(static_cast<int64_t>(arg_topk), "topk"), "128 * topk * total_q * num_q_heads");
  host_extent_5 = HostCheckedExtentMul(host_extent_5, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "128 * topk * total_q * num_q_heads");
  host_extent_5 = HostCheckedExtentMul(host_extent_5, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "128 * topk * total_q * num_q_heads");
  TVM_FFI_CHECK(arg_partial_o.numel() >= host_extent_5, ValueError)
      << "partial_o requires at least " << (host_extent_5)      << " TensorView storage elements, got " << arg_partial_o.numel();
  int64_t host_extent_6 = 1;
  host_extent_6 = HostCheckedExtentMul(host_extent_6, HostCheckedExtentValue(static_cast<int64_t>(arg_topk), "topk"), "topk * total_q * num_q_heads");
  host_extent_6 = HostCheckedExtentMul(host_extent_6, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "topk * total_q * num_q_heads");
  host_extent_6 = HostCheckedExtentMul(host_extent_6, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "topk * total_q * num_q_heads");
  TVM_FFI_CHECK(arg_partial_lse.numel() >= host_extent_6, ValueError)
      << "partial_lse requires at least " << (host_extent_6)      << " TensorView storage elements, got " << arg_partial_lse.numel();
  int64_t host_extent_7 = 1;
  host_extent_7 = HostCheckedExtentMul(host_extent_7, HostCheckedExtentValue(static_cast<int64_t>(arg_topk), "topk"), "topk * total_q * num_q_heads");
  host_extent_7 = HostCheckedExtentMul(host_extent_7, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "topk * total_q * num_q_heads");
  host_extent_7 = HostCheckedExtentMul(host_extent_7, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "topk * total_q * num_q_heads");
  TVM_FFI_CHECK(arg_partial_temperature_lse.numel() >= host_extent_7, ValueError)
      << "partial_temperature_lse requires at least " << (host_extent_7)      << " TensorView storage elements, got " << arg_partial_temperature_lse.numel();
  int64_t host_extent_8 = 1;
  host_extent_8 = HostCheckedExtentMul(host_extent_8, HostCheckedExtentValue(static_cast<int64_t>(arg_num_kv_heads), "num_kv_heads"), "num_kv_heads * total_rows");
  host_extent_8 = HostCheckedExtentMul(host_extent_8, HostCheckedExtentValue(static_cast<int64_t>(arg_total_rows), "total_rows"), "num_kv_heads * total_rows");
  TVM_FFI_CHECK(arg_k2q_row_ptr.numel() >= host_extent_8, ValueError)
      << "k2q_row_ptr requires at least " << (host_extent_8)      << " TensorView storage elements, got " << arg_k2q_row_ptr.numel();
  int64_t host_extent_9 = 4;
  host_extent_9 = HostCheckedExtentMul(host_extent_9, HostCheckedExtentValue(static_cast<int64_t>(arg_topk), "topk"), "4 * topk * total_q * num_q_heads");
  host_extent_9 = HostCheckedExtentMul(host_extent_9, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "4 * topk * total_q * num_q_heads");
  host_extent_9 = HostCheckedExtentMul(host_extent_9, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "4 * topk * total_q * num_q_heads");
  TVM_FFI_CHECK(arg_partial_scale.numel() >= host_extent_9, ValueError)
      << "partial_scale requires at least " << (host_extent_9)      << " TensorView storage elements, got " << arg_partial_scale.numel();
  TVM_FFI_CHECK(arg_max_pages >= 1, ValueError)
      << "max_pages must be >= " << 1      << ", got " << arg_max_pages;
  TVM_FFI_CHECK(arg_total_q >= 1, ValueError)
      << "total_q must be >= " << 1      << ", got " << arg_total_q;
  TVM_FFI_CHECK(arg_num_q_heads >= 1, ValueError)
      << "num_q_heads must be >= " << 1      << ", got " << arg_num_q_heads;
  TVM_FFI_CHECK(arg_num_kv_heads > 0 && arg_num_q_heads % arg_num_kv_heads == 0, ValueError)
      << "num_q_heads must be divisible by num_kv_heads";
  TVM_FFI_CHECK(arg_num_kv_heads >= 1, ValueError)
      << "num_kv_heads must be >= " << 1      << ", got " << arg_num_kv_heads;
  TVM_FFI_CHECK(arg_topk >= 1, ValueError)
      << "topk must be >= " << 1      << ", got " << arg_topk;
  TVM_FFI_CHECK(arg_topk <= 16, ValueError)
      << "topk must be <= " << 16      << ", got " << arg_topk;
  TVM_FFI_CHECK(arg_total_rows >= 1, ValueError)
      << "total_rows must be >= " << 1      << ", got " << arg_total_rows;


  CUtensorMap p_q = EncodeTma_q(arg_q);
  CUtensorMap p_k = EncodeTma_k(arg_k);
  CUtensorMap p_v = EncodeTma_v(arg_v);
  void* p_scheduler_metadata = arg_scheduler_metadata.data_ptr();
  void* p_k2q_row_ptr = arg_k2q_row_ptr.data_ptr();
  void* p_k2q_qsplit_indices = arg_k2q_qsplit_indices.data_ptr();
  void* p_partial_o = arg_partial_o.data_ptr();
  void* p_partial_scale = arg_partial_scale.data_ptr();
  void* p_partial_lse = arg_partial_lse.data_ptr();
  void* p_partial_temperature_lse = arg_partial_temperature_lse.data_ptr();
  void* p_out = arg_out.data_ptr();
  void* p_cu_seqlens_q = arg_cu_seqlens_q.data_ptr();
  void* p_cu_seqlens_k = arg_cu_seqlens_k.data_ptr();
  void* p_q_offsets = arg_q_offsets.data_ptr();
  void* p_kv_lens = arg_kv_lens.data_ptr();
  void* p_page_table = arg_page_table.data_ptr();
  int32_t v_q_group_segment_end_128 = (int32_t)arg_q_group_segment_end_128;
  int32_t v_q_group_segment_end_64 = (int32_t)arg_q_group_segment_end_64;
  int32_t v_q_group_segment_end_32 = (int32_t)arg_q_group_segment_end_32;
  int32_t v_q_group_segment_end_16 = (int32_t)arg_q_group_segment_end_16;
  int32_t v_q_group_segment_end_8 = (int32_t)arg_q_group_segment_end_8;
  int32_t v_q_group_segment_end_4 = (int32_t)arg_q_group_segment_end_4;
  int32_t v_q_group_segment_end_2 = (int32_t)arg_q_group_segment_end_2;
  int32_t v_total_q = (int32_t)arg_total_q;
  int32_t v_num_q_heads = (int32_t)arg_num_q_heads;
  int32_t v_num_kv_heads = (int32_t)arg_num_kv_heads;
  int32_t v_total_rows = (int32_t)arg_total_rows;
  int32_t v_nnz_per_head = (int32_t)arg_nnz_per_head;
  int32_t v_work_capacity = (int32_t)arg_work_capacity;
  int32_t v_num_work_items = (int32_t)arg_num_work_items;
  int32_t v_topk = (int32_t)arg_topk;
  int32_t v_max_pages = (int32_t)arg_max_pages;
  int32_t v_causal = (int32_t)arg_causal;
  int32_t v_derive_q_offset = (int32_t)arg_derive_q_offset;
  float v_softmax_scale_log2 = (float)arg_softmax_scale_log2;
  float v_lse_temperature_scale = (float)arg_lse_temperature_scale;
  int32_t v_return_temperature_lse = (int32_t)arg_return_temperature_lse;
  void* kargs[] = {&p_q, &p_k, &p_v, &p_scheduler_metadata, &p_k2q_row_ptr, &p_k2q_qsplit_indices, &p_partial_o, &p_partial_scale, &p_partial_lse, &p_partial_temperature_lse, &p_out, &p_cu_seqlens_q, &p_cu_seqlens_k, &p_q_offsets, &p_kv_lens, &p_page_table, &v_q_group_segment_end_128, &v_q_group_segment_end_64, &v_q_group_segment_end_32, &v_q_group_segment_end_16, &v_q_group_segment_end_8, &v_q_group_segment_end_4, &v_q_group_segment_end_2, &v_total_q, &v_num_q_heads, &v_num_kv_heads, &v_total_rows, &v_nnz_per_head, &v_work_capacity, &v_num_work_items, &v_topk, &v_max_pages, &v_causal, &v_derive_q_offset, &v_softmax_scale_log2, &v_lse_temperature_scale, &v_return_temperature_lse};

  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(512u, 1u, 1u);

  cudaError_t status = cudaFuncSetAttribute(
      kernel_minimax_sparse_reverse_prefill_paged_bf16_gqa4_qload4_nobar_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, 148480);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(kernel_minimax_sparse_reverse_prefill_paged_bf16_gqa4_qload4_nobar_sm100) failed: " << cudaGetErrorString(status);
  status = cudaLaunchKernel(reinterpret_cast<const void*>(kernel_minimax_sparse_reverse_prefill_paged_bf16_gqa4_qload4_nobar_sm100), grid, block, kargs,
                            148480u, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "kernel_minimax_sparse_reverse_prefill_paged_bf16_gqa4_qload4_nobar_sm100 launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::blackwell_msa

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::blackwell_msa::Run);
