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

#include <cstdint>

#include "tvm_ffi_utils.h"

#define uint8_t blackwell_msa_generated_uint8_t
#define uint16_t blackwell_msa_generated_uint16_t
#define uint32_t blackwell_msa_generated_uint32_t
#define uint64_t blackwell_msa_generated_uint64_t
#define int32_t blackwell_msa_generated_int32_t
#define int16_t blackwell_msa_generated_int16_t
#define CUtensorMap BlackwellMsaGeneratedTensorMap
#include "blackwell_msa_decode_q1_bf16_query_fp8_kv_xform2_paged.cu"
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

// A logical axis.outer(trailing) folds every source dim above the trailing
// dimensions. Shape products are independent of physical strides, so verify
// the leading dimensions form one dense row-major chain instead of inventing
// a "folded stride". The descriptor reads its exact adjacent physical step
// separately through stride[-(trailing + 1)].

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

// 3D TMA descriptor for buffer 'Q' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_Q(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'Q' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'Q' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  TVM_FFI_CHECK(d1 > 0, ValueError)
      << "TMA source 'Q' trailing dims must be positive";
  int64_t outer1 = t.numel() / (d1);
  CheckDenseLeadingFold(t, 1, "Q");
  int64_t s2 = t.stride(t.ndim() - 2) * 1;
  TVM_FFI_CHECK(s2 > 0, ValueError)
      << "TMA source 'Q' physical strides must be positive";
  TVM_FFI_CHECK(d1 % 64 == 0, ValueError)
      << "TMA source 'Q' extent " << d1
      << " must divide exactly by " << 64;
  uint64_t global_dim[3] = {(uint64_t)(64), (uint64_t)(outer1), (uint64_t)((d1 / 64))};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'Q' resolved a non-positive global dim";
  TVM_FFI_CHECK(64u <= global_dim[0] && 16u <= global_dim[1] && 2u <= global_dim[2], ValueError)
      << "TMA box (64, 16, 2) exceeds resolved global dims for 'Q'";
  uint64_t global_strides[2] = {
      (uint64_t)((s2 * 16) / 8),
      (uint64_t)((64 * 16) / 8),
  };
  uint32_t box_dim[3] = {64u, 16u, 2u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'Q') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'K' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_K(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'K' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'K' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'K' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(128), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'K' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 64u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (128, 64, 1) exceeds resolved global dims for 'K'";
  uint64_t global_strides[2] = {
      (uint64_t)((d1 * 8) / 8),
      (uint64_t)(((d2 * d1) * 8) / 8),
  };
  uint32_t box_dim[3] = {128u, 64u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'K') failed: CUresult=" << (int)r;
  return tm;
}

// 3D TMA descriptor for buffer 'V' — compiled from the
// descriptor's std.Expr global_dim/global_strides/checks record.
inline CUtensorMap EncodeTma_V(const TensorView& t) {
  TVM_FFI_CHECK(t.ndim() >= 2, ValueError)
      << "TMA source 'V' must have at least 2 dimensions, got ndim=" << t.ndim();
  TVM_FFI_CHECK(t.stride(-1) == 1, ValueError)
      << "TMA source 'V' must have unit innermost stride, got " << t.stride(-1);
  int64_t d1 = t.size(t.ndim() - 1);
  int64_t d2 = t.size(t.ndim() - 2);
  TVM_FFI_CHECK(d1 > 0 && d2 > 0, ValueError)
      << "TMA source 'V' trailing dims must be positive";
  int64_t outer2 = t.numel() / (d1 * d2);
  uint64_t global_dim[3] = {(uint64_t)(128), (uint64_t)(d2), (uint64_t)(outer2)};
  TVM_FFI_CHECK(global_dim[0] > 0 && global_dim[1] > 0 && global_dim[2] > 0, ValueError)
      << "TMA descriptor for 'V' resolved a non-positive global dim";
  TVM_FFI_CHECK(128u <= global_dim[0] && 64u <= global_dim[1] && 1u <= global_dim[2], ValueError)
      << "TMA box (128, 64, 1) exceeds resolved global dims for 'V'";
  uint64_t global_strides[2] = {
      (uint64_t)((d1 * 8) / 8),
      (uint64_t)(((d2 * d1) * 8) / 8),
  };
  uint32_t box_dim[3] = {128u, 64u, 1u};
  uint32_t elem_strides[3] = {1u, 1u, 1u};
  CUtensorMap tm;
  CUresult r = cuTensorMapEncodeTiled(
      &tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, t.data_ptr(), global_dim, global_strides, box_dim, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TVM_FFI_CHECK(r == CUDA_SUCCESS, RuntimeError)
      << "cuTensorMapEncodeTiled (3D, 'V') failed: CUresult=" << (int)r;
  return tm;
}

void Run(TensorView arg_Q, TensorView arg_K, TensorView arg_V, TensorView arg_O, TensorView arg_msa_lse, TensorView arg_kv_indices, TensorView arg_kv_indptr, TensorView arg_task_kind, TensorView arg_task_request, TensorView arg_task_kv_head, int64_t arg_num_requests, int64_t arg_num_q_heads, int64_t arg_num_kv_heads, double arg_softmax_scale_log2, int64_t arg_msa_max_pages, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_CHECK(cuda_stream >= 0, ValueError) << "cuda_stream must be non-negative";
  ffi::CUDADeviceGuard device_guard(arg_Q.device().device_id);
  CheckBlackwellMsaTarget(arg_Q.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckCudaTensor(arg_Q, "Q");
  CheckDtype(arg_Q, "Q", 4, 16, 1);
  CheckCudaTensor(arg_K, "K");
  CheckDtype(arg_K, "K", 1, 8, 1);
  CheckContiguous(arg_K, "K");
  CheckCudaTensor(arg_V, "V");
  CheckDtype(arg_V, "V", 1, 8, 1);
  CheckContiguous(arg_V, "V");
  CheckCudaTensor(arg_O, "O");
  CheckDtype(arg_O, "O", 4, 16, 1);
  CheckContiguous(arg_O, "O");
  CheckCudaTensor(arg_msa_lse, "msa_lse");
  CheckDtype(arg_msa_lse, "msa_lse", 2, 32, 1);
  CheckContiguous(arg_msa_lse, "msa_lse");
  CheckCudaTensor(arg_kv_indices, "kv_indices");
  CheckDtype(arg_kv_indices, "kv_indices", 0, 32, 1);
  CheckContiguous(arg_kv_indices, "kv_indices");
  CheckCudaTensor(arg_kv_indptr, "kv_indptr");
  CheckDtype(arg_kv_indptr, "kv_indptr", 0, 32, 1);
  CheckContiguous(arg_kv_indptr, "kv_indptr");
  CheckCudaTensor(arg_task_kind, "task_kind");
  CheckDtype(arg_task_kind, "task_kind", 0, 32, 1);
  CheckContiguous(arg_task_kind, "task_kind");
  CheckCudaTensor(arg_task_request, "task_request");
  CheckDtype(arg_task_request, "task_request", 0, 32, 1);
  CheckContiguous(arg_task_request, "task_request");
  CheckCudaTensor(arg_task_kv_head, "task_kv_head");
  CheckDtype(arg_task_kv_head, "task_kv_head", 0, 32, 1);
  CheckContiguous(arg_task_kv_head, "task_kv_head");
  TVM_FFI_CHECK(arg_num_requests >= -2147483648LL && arg_num_requests <= 2147483647LL, ValueError)
      << "scalar 'num_requests' value " << arg_num_requests
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_q_heads >= -2147483648LL && arg_num_q_heads <= 2147483647LL, ValueError)
      << "scalar 'num_q_heads' value " << arg_num_q_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_kv_heads >= -2147483648LL && arg_num_kv_heads <= 2147483647LL, ValueError)
      << "scalar 'num_kv_heads' value " << arg_num_kv_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_msa_max_pages >= -2147483648LL && arg_msa_max_pages <= 2147483647LL, ValueError)
      << "scalar 'msa_max_pages' value " << arg_msa_max_pages
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_K, arg_Q, "K", "Q");
  CheckSameCudaDevice(arg_V, arg_Q, "V", "Q");
  CheckSameCudaDevice(arg_O, arg_Q, "O", "Q");
  CheckSameCudaDevice(arg_msa_lse, arg_Q, "msa_lse", "Q");
  CheckSameCudaDevice(arg_kv_indices, arg_Q, "kv_indices", "Q");
  CheckSameCudaDevice(arg_kv_indptr, arg_Q, "kv_indptr", "Q");
  CheckSameCudaDevice(arg_task_kind, arg_Q, "task_kind", "Q");
  CheckSameCudaDevice(arg_task_request, arg_Q, "task_request", "Q");
  CheckSameCudaDevice(arg_task_kv_head, arg_Q, "task_kv_head", "Q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  CUtensorMap p_Q = EncodeTma_Q(arg_Q);
  CUtensorMap p_K = EncodeTma_K(arg_K);
  CUtensorMap p_V = EncodeTma_V(arg_V);
  void* p_O = arg_O.data_ptr();
  void* p_msa_lse = arg_msa_lse.data_ptr();
  void* p_kv_indices = arg_kv_indices.data_ptr();
  void* p_kv_indptr = arg_kv_indptr.data_ptr();
  void* p_task_kind = arg_task_kind.data_ptr();
  void* p_task_request = arg_task_request.data_ptr();
  void* p_task_kv_head = arg_task_kv_head.data_ptr();
  int32_t v_num_requests = (int32_t)arg_num_requests;
  int32_t v_num_q_heads = (int32_t)arg_num_q_heads;
  int32_t v_num_kv_heads = (int32_t)arg_num_kv_heads;
  float v_softmax_scale_log2 = (float)arg_softmax_scale_log2;
  int32_t v_msa_max_pages = (int32_t)arg_msa_max_pages;
  void* kargs[] = {&p_Q, &p_K, &p_V, &p_O, &p_msa_lse, &p_kv_indices, &p_kv_indptr, &p_task_kind, &p_task_request, &p_task_kv_head, &v_num_requests, &v_num_q_heads, &v_num_kv_heads, &v_softmax_scale_log2, &v_msa_max_pages};

  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(512u, 1u, 1u);

  cudaError_t status = cudaFuncSetAttribute(
      kernel_blackwell_batch_attention_msa_decode_q1_fp8_paged_xform2_v1, cudaFuncAttributeMaxDynamicSharedMemorySize, 216704);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(kernel_blackwell_batch_attention_msa_decode_q1_fp8_paged_xform2_v1) failed: " << cudaGetErrorString(status);
  status = cudaLaunchKernel(reinterpret_cast<const void*>(kernel_blackwell_batch_attention_msa_decode_q1_fp8_paged_xform2_v1), grid, block, kargs,
                            216704u, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "kernel_blackwell_batch_attention_msa_decode_q1_fp8_paged_xform2_v1 launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::blackwell_msa

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::blackwell_msa::Run);
