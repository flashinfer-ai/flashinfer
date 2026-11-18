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

// Shared tvm-ffi host binding of the `long_bf16_reverse_reduce` argument plan: a hand fold of the exporter's per-program
// bindings, pending exporter regeneration.  The record's compile_flags select the program:
//   -DCAKE_PROGRAM_KERNEL (the kernel symbol its device source defines)
//   -DCAKE_BIND_SMEM (dynamic shared memory bytes)
// The preprocessed text equals the exporter's binding of that program apart from the file / line of the host checks.
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "tvm_ffi_utils.h"

#include <cstdint>

#ifndef CAKE_PROGRAM_KERNEL
#error "CAKE_PROGRAM_KERNEL (the kernel symbol the record's device source defines) must be defined on the compile line"
#endif
#ifndef CAKE_BIND_SMEM
#error "CAKE_BIND_SMEM (dynamic shared memory bytes of the launch) must be defined on the compile line"
#endif
// Stringise / paste the expansion of a macro argument (the kernel name string and the binding
// namespace take the value the exporter's per-program binding spelled).
#define CAKE_STRINGIZE_(x) #x
#define CAKE_STRINGIZE(x) CAKE_STRINGIZE_(x)
#define CAKE_PASTE_(a, b) a##b
#define CAKE_PASTE(a, b) CAKE_PASTE_(a, b)
#define CAKE_BINDING_NS CAKE_PASTE(cake_binding_, CAKE_PROGRAM_KERNEL)

extern "C" __global__ void CAKE_PROGRAM_KERNEL(uint8_t* __restrict__ partial_o, __nv_bfloat16* __restrict__ partial_scale, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ split_counts, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, int total_q, int num_q_heads, int num_kv_heads, int qhead_per_kv, int topk, int return_softmax_lse, int return_temperature_lse);

namespace CAKE_BINDING_NS {

using tvm::ffi::TensorView;

constexpr const char* KERNEL_NAME = CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL);

void CheckCuda(cudaError_t status, const char* what) {
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << KERNEL_NAME << ": " << what << " failed: " << cudaGetErrorString(status);
}

void* DataPtr(const TensorView& t) { return static_cast<char*>(t.data_ptr()) + t.byte_offset(); }

void Run(TensorView a_partial_o, TensorView a_partial_scale, TensorView a_partial_lse, TensorView a_partial_temperature_lse, TensorView a_split_counts, TensorView a_out, TensorView a_lse, TensorView a_temperature_lse, int64_t a_total_q, int64_t a_num_q_heads, int64_t a_num_kv_heads, int64_t a_qhead_per_kv, int64_t a_topk, int64_t a_return_softmax_lse, int64_t a_return_temperature_lse, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = a_partial_o.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(a_partial_o, "partial_o");
  check_dtype(a_partial_o, DLDataType{kDLUInt, 8, 1}, "partial_o");
  check_cuda_tensor(a_partial_scale, "partial_scale");
  check_dtype(a_partial_scale, DLDataType{kDLBfloat, 16, 1}, "partial_scale");
  check_cuda_tensor(a_partial_lse, "partial_lse");
  check_dtype(a_partial_lse, DLDataType{kDLFloat, 32, 1}, "partial_lse");
  check_cuda_tensor(a_partial_temperature_lse, "partial_temperature_lse");
  check_dtype(a_partial_temperature_lse, DLDataType{kDLFloat, 32, 1}, "partial_temperature_lse");
  check_cuda_tensor(a_split_counts, "split_counts");
  check_dtype(a_split_counts, DLDataType{kDLInt, 32, 1}, "split_counts");
  check_cuda_tensor(a_out, "out");
  check_dtype(a_out, DLDataType{kDLBfloat, 16, 1}, "out");
  check_cuda_tensor(a_lse, "lse");
  check_dtype(a_lse, DLDataType{kDLFloat, 32, 1}, "lse");
  check_cuda_tensor(a_temperature_lse, "temperature_lse");
  check_dtype(a_temperature_lse, DLDataType{kDLFloat, 32, 1}, "temperature_lse");
  TVM_FFI_CHECK(a_total_q >= -2147483648LL && a_total_q <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'total_q' = " << a_total_q
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_q_heads >= -2147483648LL && a_num_q_heads <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_q_heads' = " << a_num_q_heads
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_kv_heads >= -2147483648LL && a_num_kv_heads <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_kv_heads' = " << a_num_kv_heads
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_qhead_per_kv >= -2147483648LL && a_qhead_per_kv <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'qhead_per_kv' = " << a_qhead_per_kv
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_topk >= -2147483648LL && a_topk <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'topk' = " << a_topk
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_return_softmax_lse >= -2147483648LL && a_return_softmax_lse <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'return_softmax_lse' = " << a_return_softmax_lse
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_return_temperature_lse >= -2147483648LL && a_return_temperature_lse <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'return_temperature_lse' = " << a_return_temperature_lse
      << " is outside the i32 range [-2147483648, 2147483647]";
  check_contiguous(a_partial_o, "partial_o");
  check_contiguous(a_partial_scale, "partial_scale");
  check_same_device(a_partial_scale, a_partial_o, "partial_scale", "partial_o");
  check_contiguous(a_partial_lse, "partial_lse");
  check_same_device(a_partial_lse, a_partial_o, "partial_lse", "partial_o");
  check_contiguous(a_partial_temperature_lse, "partial_temperature_lse");
  check_same_device(a_partial_temperature_lse, a_partial_o, "partial_temperature_lse", "partial_o");
  check_contiguous(a_split_counts, "split_counts");
  check_same_device(a_split_counts, a_partial_o, "split_counts", "partial_o");
  check_contiguous(a_out, "out");
  check_same_device(a_out, a_partial_o, "out", "partial_o");
  check_contiguous(a_lse, "lse");
  check_same_device(a_lse, a_partial_o, "lse", "partial_o");
  check_contiguous(a_temperature_lse, "temperature_lse");
  check_same_device(a_temperature_lse, a_partial_o, "temperature_lse", "partial_o");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << KERNEL_NAME << ": the grid (" << grid_x << ", " << grid_y << ", " << grid_z << ") must be positive";
  TVM_FFI_CHECK(grid_x <= 2147483647 && grid_y <= 65535 && grid_z <= 65535, ValueError)
      << KERNEL_NAME << ": the grid (" << grid_x << ", " << grid_y << ", " << grid_z << ") exceeds CUDA's limits";
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  void* v_partial_o = DataPtr(a_partial_o);
  void* v_partial_scale = DataPtr(a_partial_scale);
  void* v_partial_lse = DataPtr(a_partial_lse);
  void* v_partial_temperature_lse = DataPtr(a_partial_temperature_lse);
  void* v_split_counts = DataPtr(a_split_counts);
  void* v_out = DataPtr(a_out);
  void* v_lse = DataPtr(a_lse);
  void* v_temperature_lse = DataPtr(a_temperature_lse);
  int32_t v_total_q = static_cast<int32_t>(a_total_q);
  int32_t v_num_q_heads = static_cast<int32_t>(a_num_q_heads);
  int32_t v_num_kv_heads = static_cast<int32_t>(a_num_kv_heads);
  int32_t v_qhead_per_kv = static_cast<int32_t>(a_qhead_per_kv);
  int32_t v_topk = static_cast<int32_t>(a_topk);
  int32_t v_return_softmax_lse = static_cast<int32_t>(a_return_softmax_lse);
  int32_t v_return_temperature_lse = static_cast<int32_t>(a_return_temperature_lse);
  void* kernel_args[] = {&v_partial_o, &v_partial_scale, &v_partial_lse, &v_partial_temperature_lse, &v_split_counts, &v_out, &v_lse, &v_temperature_lse, &v_total_q, &v_num_q_heads, &v_num_kv_heads, &v_qhead_per_kv, &v_topk, &v_return_softmax_lse, &v_return_temperature_lse};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), static_cast<unsigned>(grid_z));
  config.blockDim = dim3(256, 1, 1);
  config.dynamicSmemBytes = CAKE_BIND_SMEM;
  config.stream = stream;
  cudaLaunchAttribute attrs[1] = {};
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = 1;
  CheckCuda(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kernel_args),
            "cudaLaunchKernelExC");
}

}  // namespace CAKE_BINDING_NS

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_BINDING_NS::Run);
