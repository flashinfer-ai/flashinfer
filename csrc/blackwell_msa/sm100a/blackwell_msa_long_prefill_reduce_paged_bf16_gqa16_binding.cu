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
#include "blackwell_msa_long_prefill_reduce_paged_bf16_gqa16.cu"
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

void Run(TensorView arg_partial_o, TensorView arg_partial_scale, TensorView arg_partial_lse, TensorView arg_partial_temperature_lse, TensorView arg_split_counts, TensorView arg_out, TensorView arg_lse, TensorView arg_temperature_lse, int64_t arg_total_q, int64_t arg_num_q_heads, int64_t arg_num_kv_heads, int64_t arg_qhead_per_kv, int64_t arg_topk, int64_t arg_return_softmax_lse, int64_t arg_return_temperature_lse, int64_t grid_x, int64_t grid_y, int64_t grid_z, int64_t cuda_stream) {
  TVM_FFI_CHECK(cuda_stream >= 0, ValueError) << "cuda_stream must be non-negative";
  CheckCudaTensor(arg_partial_o, "partial_o");
  ffi::CUDADeviceGuard device_guard(arg_partial_o.device().device_id);
  CheckBlackwellMsaTarget(arg_partial_o.device().device_id);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
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
  CheckCudaTensor(arg_split_counts, "split_counts");
  CheckDtype(arg_split_counts, "split_counts", 0, 32, 1);
  CheckContiguous(arg_split_counts, "split_counts");
  CheckCudaTensor(arg_out, "out");
  CheckDtype(arg_out, "out", 4, 16, 1);
  CheckContiguous(arg_out, "out");
  CheckCudaTensor(arg_lse, "lse");
  CheckDtype(arg_lse, "lse", 2, 32, 1);
  CheckContiguous(arg_lse, "lse");
  CheckCudaTensor(arg_temperature_lse, "temperature_lse");
  CheckDtype(arg_temperature_lse, "temperature_lse", 2, 32, 1);
  CheckContiguous(arg_temperature_lse, "temperature_lse");
  TVM_FFI_CHECK(arg_total_q >= -2147483648LL && arg_total_q <= 2147483647LL, ValueError)
      << "scalar 'total_q' value " << arg_total_q
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_q_heads >= -2147483648LL && arg_num_q_heads <= 2147483647LL, ValueError)
      << "scalar 'num_q_heads' value " << arg_num_q_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_num_kv_heads >= -2147483648LL && arg_num_kv_heads <= 2147483647LL, ValueError)
      << "scalar 'num_kv_heads' value " << arg_num_kv_heads
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_qhead_per_kv >= -2147483648LL && arg_qhead_per_kv <= 2147483647LL, ValueError)
      << "scalar 'qhead_per_kv' value " << arg_qhead_per_kv
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_topk >= -2147483648LL && arg_topk <= 2147483647LL, ValueError)
      << "scalar 'topk' value " << arg_topk
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_return_softmax_lse >= -2147483648LL && arg_return_softmax_lse <= 2147483647LL, ValueError)
      << "scalar 'return_softmax_lse' value " << arg_return_softmax_lse
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_return_temperature_lse >= -2147483648LL && arg_return_temperature_lse <= 2147483647LL, ValueError)
      << "scalar 'return_temperature_lse' value " << arg_return_temperature_lse
      << " is outside i32 range [-2147483648, 2147483647]";
  CheckSameCudaDevice(arg_partial_scale, arg_partial_o, "partial_scale", "partial_o");
  CheckSameCudaDevice(arg_partial_lse, arg_partial_o, "partial_lse", "partial_o");
  CheckSameCudaDevice(arg_partial_temperature_lse, arg_partial_o, "partial_temperature_lse", "partial_o");
  CheckSameCudaDevice(arg_split_counts, arg_partial_o, "split_counts", "partial_o");
  CheckSameCudaDevice(arg_out, arg_partial_o, "out", "partial_o");
  CheckSameCudaDevice(arg_lse, arg_partial_o, "lse", "partial_o");
  CheckSameCudaDevice(arg_temperature_lse, arg_partial_o, "temperature_lse", "partial_o");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";
  int64_t host_extent_0 = 128;
  host_extent_0 = HostCheckedExtentMul(host_extent_0, HostCheckedExtentValue(static_cast<int64_t>(arg_topk), "topk"), "128 * topk * total_q * num_q_heads");
  host_extent_0 = HostCheckedExtentMul(host_extent_0, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "128 * topk * total_q * num_q_heads");
  host_extent_0 = HostCheckedExtentMul(host_extent_0, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "128 * topk * total_q * num_q_heads");
  TVM_FFI_CHECK(arg_partial_o.numel() >= host_extent_0, ValueError)
      << "partial_o requires at least " << (host_extent_0)      << " TensorView storage elements, got " << arg_partial_o.numel();
  int64_t host_extent_1 = 1;
  host_extent_1 = HostCheckedExtentMul(host_extent_1, HostCheckedExtentValue(static_cast<int64_t>(arg_topk), "topk"), "topk * total_q * num_q_heads");
  host_extent_1 = HostCheckedExtentMul(host_extent_1, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "topk * total_q * num_q_heads");
  host_extent_1 = HostCheckedExtentMul(host_extent_1, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "topk * total_q * num_q_heads");
  TVM_FFI_CHECK(arg_partial_lse.numel() >= host_extent_1, ValueError)
      << "partial_lse requires at least " << (host_extent_1)      << " TensorView storage elements, got " << arg_partial_lse.numel();
  int64_t host_extent_2 = 1;
  host_extent_2 = HostCheckedExtentMul(host_extent_2, HostCheckedExtentValue(static_cast<int64_t>(arg_topk), "topk"), "topk * total_q * num_q_heads");
  host_extent_2 = HostCheckedExtentMul(host_extent_2, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "topk * total_q * num_q_heads");
  host_extent_2 = HostCheckedExtentMul(host_extent_2, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "topk * total_q * num_q_heads");
  TVM_FFI_CHECK(arg_partial_temperature_lse.numel() >= host_extent_2, ValueError)
      << "partial_temperature_lse requires at least " << (host_extent_2)      << " TensorView storage elements, got " << arg_partial_temperature_lse.numel();
  int64_t host_extent_3 = 1;
  host_extent_3 = HostCheckedExtentMul(host_extent_3, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "total_q * num_kv_heads");
  host_extent_3 = HostCheckedExtentMul(host_extent_3, HostCheckedExtentValue(static_cast<int64_t>(arg_num_kv_heads), "num_kv_heads"), "total_q * num_kv_heads");
  TVM_FFI_CHECK(arg_split_counts.numel() >= host_extent_3, ValueError)
      << "split_counts requires at least " << (host_extent_3)      << " TensorView storage elements, got " << arg_split_counts.numel();
  int64_t host_extent_4 = 128;
  host_extent_4 = HostCheckedExtentMul(host_extent_4, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "128 * total_q * num_q_heads");
  host_extent_4 = HostCheckedExtentMul(host_extent_4, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "128 * total_q * num_q_heads");
  TVM_FFI_CHECK(arg_out.numel() >= host_extent_4, ValueError)
      << "out requires at least " << (host_extent_4)      << " TensorView storage elements, got " << arg_out.numel();
  int64_t host_extent_5 = 1;
  host_extent_5 = HostCheckedExtentMul(host_extent_5, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "total_q * num_q_heads");
  host_extent_5 = HostCheckedExtentMul(host_extent_5, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "total_q * num_q_heads");
  TVM_FFI_CHECK(arg_lse.numel() >= host_extent_5, ValueError)
      << "lse requires at least " << (host_extent_5)      << " TensorView storage elements, got " << arg_lse.numel();
  int64_t host_extent_6 = 1;
  host_extent_6 = HostCheckedExtentMul(host_extent_6, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "total_q * num_q_heads");
  host_extent_6 = HostCheckedExtentMul(host_extent_6, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "total_q * num_q_heads");
  TVM_FFI_CHECK(arg_temperature_lse.numel() >= host_extent_6, ValueError)
      << "temperature_lse requires at least " << (host_extent_6)      << " TensorView storage elements, got " << arg_temperature_lse.numel();
  int64_t host_extent_7 = 4;
  host_extent_7 = HostCheckedExtentMul(host_extent_7, HostCheckedExtentValue(static_cast<int64_t>(arg_topk), "topk"), "4 * topk * total_q * num_q_heads");
  host_extent_7 = HostCheckedExtentMul(host_extent_7, HostCheckedExtentValue(static_cast<int64_t>(arg_total_q), "total_q"), "4 * topk * total_q * num_q_heads");
  host_extent_7 = HostCheckedExtentMul(host_extent_7, HostCheckedExtentValue(static_cast<int64_t>(arg_num_q_heads), "num_q_heads"), "4 * topk * total_q * num_q_heads");
  TVM_FFI_CHECK(arg_partial_scale.numel() >= host_extent_7, ValueError)
      << "partial_scale requires at least " << (host_extent_7)      << " TensorView storage elements, got " << arg_partial_scale.numel();
  TVM_FFI_CHECK(arg_total_q >= 1, ValueError)
      << "total_q must be >= " << 1      << ", got " << arg_total_q;
  TVM_FFI_CHECK(arg_num_q_heads >= 1, ValueError)
      << "num_q_heads must be >= " << 1      << ", got " << arg_num_q_heads;
  int64_t host_extent_8 = 1;
  host_extent_8 = HostCheckedExtentMul(host_extent_8, HostCheckedExtentValue(static_cast<int64_t>(arg_qhead_per_kv), "qhead_per_kv"), "qhead_per_kv * num_kv_heads");
  host_extent_8 = HostCheckedExtentMul(host_extent_8, HostCheckedExtentValue(static_cast<int64_t>(arg_num_kv_heads), "num_kv_heads"), "qhead_per_kv * num_kv_heads");
  TVM_FFI_CHECK(arg_num_q_heads == host_extent_8, ValueError)
      << "num_q_heads must equal " << host_extent_8      << ", got " << arg_num_q_heads;
  TVM_FFI_CHECK(arg_num_kv_heads > 0 && arg_num_q_heads % arg_num_kv_heads == 0, ValueError)
      << "num_q_heads must be divisible by num_kv_heads";
  TVM_FFI_CHECK(arg_num_kv_heads >= 1, ValueError)
      << "num_kv_heads must be >= " << 1      << ", got " << arg_num_kv_heads;
  TVM_FFI_CHECK(arg_topk >= 1, ValueError)
      << "topk must be >= " << 1      << ", got " << arg_topk;
  TVM_FFI_CHECK(arg_topk <= 16, ValueError)
      << "topk must be <= " << 16      << ", got " << arg_topk;
  TVM_FFI_CHECK(arg_qhead_per_kv >= 1, ValueError)
      << "qhead_per_kv must be >= " << 1      << ", got " << arg_qhead_per_kv;


  void* p_partial_o = arg_partial_o.data_ptr();
  void* p_partial_scale = arg_partial_scale.data_ptr();
  void* p_partial_lse = arg_partial_lse.data_ptr();
  void* p_partial_temperature_lse = arg_partial_temperature_lse.data_ptr();
  void* p_split_counts = arg_split_counts.data_ptr();
  void* p_out = arg_out.data_ptr();
  void* p_lse = arg_lse.data_ptr();
  void* p_temperature_lse = arg_temperature_lse.data_ptr();
  int32_t v_total_q = (int32_t)arg_total_q;
  int32_t v_num_q_heads = (int32_t)arg_num_q_heads;
  int32_t v_num_kv_heads = (int32_t)arg_num_kv_heads;
  int32_t v_qhead_per_kv = (int32_t)arg_qhead_per_kv;
  int32_t v_topk = (int32_t)arg_topk;
  int32_t v_return_softmax_lse = (int32_t)arg_return_softmax_lse;
  int32_t v_return_temperature_lse = (int32_t)arg_return_temperature_lse;
  void* kargs[] = {&p_partial_o, &p_partial_scale, &p_partial_lse, &p_partial_temperature_lse, &p_split_counts, &p_out, &p_lse, &p_temperature_lse, &v_total_q, &v_num_q_heads, &v_num_kv_heads, &v_qhead_per_kv, &v_topk, &v_return_softmax_lse, &v_return_temperature_lse};

  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(256u, 1u, 1u);

  cudaError_t status = cudaFuncSetAttribute(
      kernel_minimax_sparse_reverse_prefill_combine_topk16_fp8partial_bf16_sm100, cudaFuncAttributeMaxDynamicSharedMemorySize, 36864);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaFuncSetAttribute(kernel_minimax_sparse_reverse_prefill_combine_topk16_fp8partial_bf16_sm100) failed: " << cudaGetErrorString(status);
  cudaLaunchAttribute attrs[1]{};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 36864u;
  config.stream = stream;
  config.attrs = attrs;
  config.numAttrs = 1;
  status = cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(kernel_minimax_sparse_reverse_prefill_combine_topk16_fp8partial_bf16_sm100), kargs);
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "kernel_minimax_sparse_reverse_prefill_combine_topk16_fp8partial_bf16_sm100 launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::blackwell_msa

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::blackwell_msa::Run);
