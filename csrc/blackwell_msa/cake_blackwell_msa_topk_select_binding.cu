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

// Shared tvm-ffi host binding of the `topk_select` argument plan: a hand fold of the exporter's per-program
// bindings, pending exporter regeneration.  The record's compile_flags select the program:
//   -DCAKE_PROGRAM_KERNEL (the kernel symbol its device source defines)
//   -DCAKE_BIND_SMEM (dynamic shared memory bytes)
// The preprocessed text equals the exporter's binding of that program apart from the file / line of the host checks.
#include <cuda_runtime.h>

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

extern "C" __global__ void CAKE_PROGRAM_KERNEL(float* __restrict__ max_score, int* __restrict__ output, unsigned long long* __restrict__ partials, unsigned int* __restrict__ counters, unsigned int* __restrict__ block_max, int num_heads, int max_k_tiles, int total_q, int num_valid_pages, int force_begin_blocks, int force_end_blocks, int num_qtiles, int num_slices, int slice_len, int index_bits);

namespace CAKE_BINDING_NS {

using tvm::ffi::TensorView;

constexpr const char* KERNEL_NAME = CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL);

void CheckCuda(cudaError_t status, const char* what) {
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << KERNEL_NAME << ": " << what << " failed: " << cudaGetErrorString(status);
}

void* DataPtr(const TensorView& t) { return static_cast<char*>(t.data_ptr()) + t.byte_offset(); }

void Run(TensorView a_max_score, TensorView a_output, TensorView a_partials, TensorView a_counters, TensorView a_block_max, int64_t a_num_heads, int64_t a_max_k_tiles, int64_t a_total_q, int64_t a_num_valid_pages, int64_t a_force_begin_blocks, int64_t a_force_end_blocks, int64_t a_num_qtiles, int64_t a_num_slices, int64_t a_slice_len, int64_t a_index_bits, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = a_max_score.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(a_max_score, "max_score");
  check_dtype(a_max_score, DLDataType{kDLFloat, 32, 1}, "max_score");
  check_cuda_tensor(a_output, "output");
  check_dtype(a_output, DLDataType{kDLInt, 32, 1}, "output");
  check_cuda_tensor(a_partials, "partials");
  check_dtype(a_partials, DLDataType{kDLUInt, 64, 1}, "partials");
  check_cuda_tensor(a_counters, "counters");
  check_dtype(a_counters, DLDataType{kDLUInt, 32, 1}, "counters");
  check_cuda_tensor(a_block_max, "block_max");
  check_dtype(a_block_max, DLDataType{kDLUInt, 32, 1}, "block_max");
  TVM_FFI_CHECK(a_num_heads >= -2147483648LL && a_num_heads <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_heads' = " << a_num_heads
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_max_k_tiles >= -2147483648LL && a_max_k_tiles <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'max_k_tiles' = " << a_max_k_tiles
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_total_q >= -2147483648LL && a_total_q <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'total_q' = " << a_total_q
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_valid_pages >= -2147483648LL && a_num_valid_pages <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_valid_pages' = " << a_num_valid_pages
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_force_begin_blocks >= -2147483648LL && a_force_begin_blocks <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'force_begin_blocks' = " << a_force_begin_blocks
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_force_end_blocks >= -2147483648LL && a_force_end_blocks <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'force_end_blocks' = " << a_force_end_blocks
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_qtiles >= -2147483648LL && a_num_qtiles <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_qtiles' = " << a_num_qtiles
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_num_slices >= -2147483648LL && a_num_slices <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'num_slices' = " << a_num_slices
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_slice_len >= -2147483648LL && a_slice_len <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'slice_len' = " << a_slice_len
      << " is outside the i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(a_index_bits >= -2147483648LL && a_index_bits <= 2147483647LL, ValueError)
      << KERNEL_NAME << ": 'index_bits' = " << a_index_bits
      << " is outside the i32 range [-2147483648, 2147483647]";
  check_contiguous(a_max_score, "max_score");
  check_contiguous(a_output, "output");
  check_same_device(a_output, a_max_score, "output", "max_score");
  check_contiguous(a_partials, "partials");
  check_same_device(a_partials, a_max_score, "partials", "max_score");
  check_contiguous(a_counters, "counters");
  check_same_device(a_counters, a_max_score, "counters", "max_score");
  check_contiguous(a_block_max, "block_max");
  check_same_device(a_block_max, a_max_score, "block_max", "max_score");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << KERNEL_NAME << ": the grid (" << grid_x << ", " << grid_y << ", " << grid_z << ") must be positive";
  TVM_FFI_CHECK(grid_x <= 2147483647 && grid_y <= 65535 && grid_z <= 65535, ValueError)
      << KERNEL_NAME << ": the grid (" << grid_x << ", " << grid_y << ", " << grid_z << ") exceeds CUDA's limits";
  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(dev.device_type, dev.device_id));
  {
    // Once per thread per device: the kernel's function attributes hold per device.
    static thread_local uint64_t prepared_devices = 0;
    const uint64_t device_bit = dev.device_id < 64 ? uint64_t{1} << dev.device_id : uint64_t{0};
    if (device_bit == 0 || (prepared_devices & device_bit) == 0) {
      CheckCuda(cudaFuncSetAttribute(reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), cudaFuncAttributeMaxDynamicSharedMemorySize, CAKE_BIND_SMEM), "cudaFuncSetAttribute");
      prepared_devices |= device_bit;
    }
  }
  void* v_max_score = DataPtr(a_max_score);
  void* v_output = DataPtr(a_output);
  void* v_partials = DataPtr(a_partials);
  void* v_counters = DataPtr(a_counters);
  void* v_block_max = DataPtr(a_block_max);
  int32_t v_num_heads = static_cast<int32_t>(a_num_heads);
  int32_t v_max_k_tiles = static_cast<int32_t>(a_max_k_tiles);
  int32_t v_total_q = static_cast<int32_t>(a_total_q);
  int32_t v_num_valid_pages = static_cast<int32_t>(a_num_valid_pages);
  int32_t v_force_begin_blocks = static_cast<int32_t>(a_force_begin_blocks);
  int32_t v_force_end_blocks = static_cast<int32_t>(a_force_end_blocks);
  int32_t v_num_qtiles = static_cast<int32_t>(a_num_qtiles);
  int32_t v_num_slices = static_cast<int32_t>(a_num_slices);
  int32_t v_slice_len = static_cast<int32_t>(a_slice_len);
  int32_t v_index_bits = static_cast<int32_t>(a_index_bits);
  void* kernel_args[] = {&v_max_score, &v_output, &v_partials, &v_counters, &v_block_max, &v_num_heads, &v_max_k_tiles, &v_total_q, &v_num_valid_pages, &v_force_begin_blocks, &v_force_end_blocks, &v_num_qtiles, &v_num_slices, &v_slice_len, &v_index_bits};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), static_cast<unsigned>(grid_z));
  config.blockDim = dim3(256, 1, 1);
  config.dynamicSmemBytes = CAKE_BIND_SMEM;
  config.stream = stream;
  CheckCuda(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kernel_args),
            "cudaLaunchKernelExC");
}

}  // namespace CAKE_BINDING_NS

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_BINDING_NS::Run);
