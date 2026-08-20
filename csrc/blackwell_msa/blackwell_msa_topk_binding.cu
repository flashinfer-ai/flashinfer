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
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "tvm_ffi_utils.h"

#define uint8_t blackwell_msa_generated_uint8_t
#define uint16_t blackwell_msa_generated_uint16_t
#define uint32_t blackwell_msa_generated_uint32_t
#define uint64_t blackwell_msa_generated_uint64_t
#define int32_t blackwell_msa_generated_int32_t
#define int16_t blackwell_msa_generated_int16_t
#include "blackwell_msa_topk.cu"
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer::blackwell_msa {

using tvm::ffi::TensorView;

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

#if defined(FLASHINFER_BLACKWELL_MSA_TARGET_MINOR) == defined(FLASHINFER_BLACKWELL_MSA_TARGET_FAMILY)
#error "exactly one Blackwell MSA target must be defined"
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
#if defined(FLASHINFER_BLACKWELL_MSA_TARGET_MINOR)
  static_assert(FLASHINFER_BLACKWELL_MSA_TARGET_MINOR == 0);
  TVM_FFI_CHECK(major == 10 && minor == 0, RuntimeError)
      << "this Blackwell MSA module requires compute capability 10.0, got " << major << "." << minor;
#else
  static_assert(FLASHINFER_BLACKWELL_MSA_TARGET_FAMILY == 100);
  TVM_FFI_CHECK(major == 10 && (minor == 0 || minor == 3), RuntimeError)
      << "this Blackwell MSA module supports compute capability 10.0 or 10.3, got " << major << "."
      << minor;
#endif
}

void Run(TensorView arg_max_score, TensorView arg_output, int64_t arg_num_heads,
         int64_t arg_max_k_tiles, int64_t arg_total_q, int64_t arg_num_valid_pages,
         int64_t arg_force_begin_blocks, int64_t arg_force_end_blocks, int64_t grid_x,
         int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_CHECK(cuda_stream >= 0, ValueError) << "cuda_stream must be non-negative";
  ffi::CUDADeviceGuard device_guard(arg_max_score.device().device_id);
  CheckBlackwellMsaTarget(arg_max_score.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  CheckCudaTensor(arg_max_score, "max_score");
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
  TVM_FFI_CHECK(arg_num_valid_pages >= -2147483648LL && arg_num_valid_pages <= 2147483647LL,
                ValueError)
      << "scalar 'num_valid_pages' value " << arg_num_valid_pages
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_force_begin_blocks >= -2147483648LL && arg_force_begin_blocks <= 2147483647LL,
                ValueError)
      << "scalar 'force_begin_blocks' value " << arg_force_begin_blocks
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_force_end_blocks >= -2147483648LL && arg_force_end_blocks <= 2147483647LL,
                ValueError)
      << "scalar 'force_end_blocks' value " << arg_force_end_blocks
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_output, arg_max_score, "output", "max_score");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y << ", "
      << grid_z << ")";

  void* p_max_score = arg_max_score.data_ptr();
  void* p_output = arg_output.data_ptr();
  int32_t v_num_heads = (int32_t)arg_num_heads;
  int32_t v_max_k_tiles = (int32_t)arg_max_k_tiles;
  int32_t v_total_q = (int32_t)arg_total_q;
  int32_t v_num_valid_pages = (int32_t)arg_num_valid_pages;
  int32_t v_force_begin_blocks = (int32_t)arg_force_begin_blocks;
  int32_t v_force_end_blocks = (int32_t)arg_force_end_blocks;
  void* kargs[] = {&p_max_score, &p_output,          &v_num_heads,          &v_max_k_tiles,
                   &v_total_q,   &v_num_valid_pages, &v_force_begin_blocks, &v_force_end_blocks};

  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(256u, 1u, 1u);

  cudaError_t status =
      cudaFuncSetAttribute(kernel_blackwell_msa_topk, cudaFuncAttributeMaxDynamicSharedMemorySize, 128);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(kernel_blackwell_msa_topk) failed: " << cudaGetErrorString(status);
  status = cudaLaunchKernel(reinterpret_cast<const void*>(kernel_blackwell_msa_topk), grid, block, kargs,
                            128u, stream);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "kernel_blackwell_msa_topk launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::blackwell_msa

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::blackwell_msa::Run);
