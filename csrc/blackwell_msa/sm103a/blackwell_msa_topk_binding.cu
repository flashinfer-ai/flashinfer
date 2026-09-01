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
#include "blackwell_msa_topk.cu"
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

void Run(TensorView arg_max_score, TensorView arg_output, int64_t arg_num_heads, int64_t arg_max_k_tiles, int64_t arg_total_q, int64_t arg_num_valid_pages, int64_t arg_force_begin_blocks, int64_t arg_force_end_blocks, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_CHECK(cuda_stream >= 0, ValueError) << "cuda_stream must be non-negative";
  CheckCudaTensor(arg_max_score, "max_score");
  ffi::CUDADeviceGuard device_guard(arg_max_score.device().device_id);
  CheckBlackwellMsaTarget(arg_max_score.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckDtype(arg_max_score, "max_score", 2, 32, 1);
  CheckContiguous(arg_max_score, "max_score");
  CheckCudaTensor(arg_output, "output");
  CheckDtype(arg_output, "output", 0, 32, 1);
  CheckContiguous(arg_output, "output");
  TVM_FFI_CHECK(arg_num_heads >= -2147483648LL && arg_num_heads <= 2147483647LL, ValueError)
      << "scalar 'num_heads' value " << arg_num_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_max_k_tiles >= -2147483648LL && arg_max_k_tiles <= 2147483647LL, ValueError)
      << "scalar 'max_k_tiles' value " << arg_max_k_tiles
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_total_q >= -2147483648LL && arg_total_q <= 2147483647LL, ValueError)
      << "scalar 'total_q' value " << arg_total_q
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_valid_pages >= -2147483648LL && arg_num_valid_pages <= 2147483647LL, ValueError)
      << "scalar 'num_valid_pages' value " << arg_num_valid_pages
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_force_begin_blocks >= -2147483648LL && arg_force_begin_blocks <= 2147483647LL, ValueError)
      << "scalar 'force_begin_blocks' value " << arg_force_begin_blocks
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_force_end_blocks >= -2147483648LL && arg_force_end_blocks <= 2147483647LL, ValueError)
      << "scalar 'force_end_blocks' value " << arg_force_end_blocks
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_output, arg_max_score, "output", "max_score");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";
  int64_t host_extent_0 = 1;
  host_extent_0 = HostCheckedExtentMul(host_extent_0, HostCheckedExtentValue(static_cast<int64_t>(arg_num_heads), "num_heads"), "num_heads * max_k_tiles * total_q");
  host_extent_0 = HostCheckedExtentMul(host_extent_0, HostCheckedExtentValue(static_cast<int64_t>(arg_max_k_tiles), "max_k_tiles"), "num_heads * max_k_tiles * total_q");
  host_extent_0 = HostCheckedExtentMul(host_extent_0, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "num_heads * max_k_tiles * total_q");
  TVM_FFI_CHECK(arg_max_score.numel() >= host_extent_0, ValueError)
      << "max_score requires at least " << (host_extent_0)      << " TensorView storage elements, got " << arg_max_score.numel();
  int64_t host_extent_1 = 16;
  host_extent_1 = HostCheckedExtentMul(host_extent_1, HostCheckedExtentValue(static_cast<int64_t>(grid_x), "grid_x"), "16 * grid_x");
  TVM_FFI_CHECK(arg_output.numel() >= host_extent_1, ValueError)
      << "output requires at least " << (host_extent_1)      << " TensorView storage elements, got " << arg_output.numel();
  TVM_FFI_CHECK(arg_num_heads >= 1, ValueError)
      << "num_heads must be >= " << 1      << ", got " << arg_num_heads;
  TVM_FFI_CHECK(arg_max_k_tiles >= 1, ValueError)
      << "max_k_tiles must be >= " << 1      << ", got " << arg_max_k_tiles;
  TVM_FFI_CHECK(arg_total_q >= 1, ValueError)
      << "total_q must be >= " << 1      << ", got " << arg_total_q;
  TVM_FFI_CHECK(arg_num_valid_pages >= 1, ValueError)
      << "num_valid_pages must be >= " << 1      << ", got " << arg_num_valid_pages;
  int64_t host_extent_2 = 1;
  host_extent_2 = HostCheckedExtentMul(host_extent_2, HostCheckedExtentValue(static_cast<int64_t>(arg_max_k_tiles), "max_k_tiles"), "max_k_tiles");
  TVM_FFI_CHECK(arg_num_valid_pages <= host_extent_2, ValueError)
      << "num_valid_pages must be <= " << host_extent_2      << ", got " << arg_num_valid_pages;
  TVM_FFI_CHECK(arg_force_begin_blocks >= 0, ValueError)
      << "force_begin_blocks must be >= " << 0      << ", got " << arg_force_begin_blocks;
  TVM_FFI_CHECK(arg_force_end_blocks >= 0, ValueError)
      << "force_end_blocks must be >= " << 0      << ", got " << arg_force_end_blocks;
  int64_t host_extent_3 = 1;
  host_extent_3 = HostCheckedExtentMul(host_extent_3, HostCheckedExtentValue(static_cast<int64_t>(arg_force_begin_blocks), "force_begin_blocks"), "force_begin_blocks");
  int64_t host_extent_4 = 1;
  host_extent_4 = HostCheckedExtentMul(host_extent_4, HostCheckedExtentValue(static_cast<int64_t>(arg_force_end_blocks), "force_end_blocks"), "force_end_blocks");
  int64_t host_extent_5 = host_extent_3;
  host_extent_5 = HostCheckedExtentAdd(host_extent_5, host_extent_4, "(force_begin_blocks + force_end_blocks)");
  int64_t host_extent_6 = 1;
  host_extent_6 = HostCheckedExtentMul(host_extent_6, HostCheckedExtentValue(static_cast<int64_t>(arg_num_valid_pages), "num_valid_pages"), "num_valid_pages");
  TVM_FFI_CHECK(host_extent_5 <= host_extent_6, ValueError)
      << "force_begin_blocks + force_end_blocks" << " must be <= " << host_extent_6      << ", got " << host_extent_5;
  int64_t host_extent_7 = 1;
  host_extent_7 = HostCheckedExtentMul(host_extent_7, HostCheckedExtentValue(static_cast<int64_t>(arg_force_begin_blocks), "force_begin_blocks"), "force_begin_blocks");
  int64_t host_extent_8 = 1;
  host_extent_8 = HostCheckedExtentMul(host_extent_8, HostCheckedExtentValue(static_cast<int64_t>(arg_force_end_blocks), "force_end_blocks"), "force_end_blocks");
  int64_t host_extent_9 = host_extent_7;
  host_extent_9 = HostCheckedExtentAdd(host_extent_9, host_extent_8, "(force_begin_blocks + force_end_blocks)");
  TVM_FFI_CHECK(host_extent_9 <= 16, ValueError)
      << "forced Top-K blocks" << " must be <= " << 16      << ", got " << host_extent_9;
  int64_t host_extent_10 = 1;
  host_extent_10 = HostCheckedExtentMul(host_extent_10, HostCheckedExtentValue(static_cast<int64_t>(arg_num_heads), "num_heads"), "num_heads * total_q");
  host_extent_10 = HostCheckedExtentMul(host_extent_10, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "num_heads * total_q");
  TVM_FFI_CHECK(grid_x == host_extent_10, ValueError)
      << "grid_x must equal " << host_extent_10      << ", got " << grid_x;
  int64_t host_extent_11 = 1;
  TVM_FFI_CHECK(grid_y == host_extent_11, ValueError)
      << "grid_y must equal " << host_extent_11      << ", got " << grid_y;
  int64_t host_extent_12 = 1;
  TVM_FFI_CHECK(grid_z == host_extent_12, ValueError)
      << "grid_z must equal " << host_extent_12      << ", got " << grid_z;


  void* p_max_score = arg_max_score.data_ptr();
  void* p_output = arg_output.data_ptr();
  int32_t v_num_heads = (int32_t)arg_num_heads;
  int32_t v_max_k_tiles = (int32_t)arg_max_k_tiles;
  int32_t v_total_q = (int32_t)arg_total_q;
  int32_t v_num_valid_pages = (int32_t)arg_num_valid_pages;
  int32_t v_force_begin_blocks = (int32_t)arg_force_begin_blocks;
  int32_t v_force_end_blocks = (int32_t)arg_force_end_blocks;
  void* kargs[] = {&p_max_score, &p_output, &v_num_heads, &v_max_k_tiles, &v_total_q, &v_num_valid_pages, &v_force_begin_blocks, &v_force_end_blocks};

  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(256u, 1u, 1u);

  cudaError_t status = cudaFuncSetAttribute(
      kernel_minimax_sparse_topk_select_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, 128);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(kernel_minimax_sparse_topk_select_sm100) failed: " << cudaGetErrorString(status);
  status = cudaLaunchKernel(reinterpret_cast<const void*>(kernel_minimax_sparse_topk_select_sm100), grid, block, kargs,
                            128u, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "kernel_minimax_sparse_topk_select_sm100 launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::blackwell_msa

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::blackwell_msa::Run);
