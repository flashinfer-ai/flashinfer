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

// Shared launcher: the kernel symbol and host-shim namespace come from
// CAKE_PROGRAM_KERNEL / CAKE_PROGRAM_SHIM; see the build.json entries that build this file.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cake_ffi_checks.h"

#include <cstdint>

#include "cake_dsa_indexer_topk_host_common.cuh"

// Program selection.  Programs whose exported binding text differs from this file's
// only in the kernel symbol, the host-shim namespace, the block size and the dynamic
// shared memory build this translation unit with -DCAKE_PROGRAM_KERNEL /
// -DCAKE_PROGRAM_SHIM naming their own and the -DTHREADS / -DSMEM_TOTAL their
// build.json entry passes to the kernel; the defaults select the program this file was
// exported for.
#ifndef CAKE_PROGRAM_KERNEL
#define CAKE_PROGRAM_KERNEL kernel_cake_dsa_indexer_topk_e1e316281190b2a3203b
#endif
#ifndef CAKE_PROGRAM_SHIM
#define CAKE_PROGRAM_SHIM cake_host_shim_b06dca111e0ce4d9
#endif
#ifndef SMEM_TOTAL
#define SMEM_TOTAL 4096
#endif
#ifndef THREADS
#define THREADS 32
#endif

extern "C" __global__ void CAKE_PROGRAM_KERNEL(int* __restrict__ Indices, float* __restrict__ Scores, int top_k, int key_bits);


namespace CAKE_PROGRAM_SHIM {

using namespace cake_host_shim_common;
using tvm::ffi::TensorView;

void Run(TensorView arg_Indices, TensorView arg_Scores, int64_t arg_top_k, int64_t arg_key_bits, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_Indices.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  check_cuda_tensor(arg_Indices, "Indices");
  check_dtype(arg_Indices, DLDataType{kDLInt, 32, 1}, "Indices");
  check_contiguous(arg_Indices, "Indices");
  check_cuda_tensor(arg_Scores, "Scores");
  check_dtype(arg_Scores, DLDataType{kDLFloat, 32, 1}, "Scores");
  check_contiguous(arg_Scores, "Scores");
  TVM_FFI_CHECK(arg_top_k >= -2147483648LL && arg_top_k <= 2147483647LL, ValueError)
      << "scalar 'top_k' value " << arg_top_k
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_key_bits >= -2147483648LL && arg_key_bits <= 2147483647LL, ValueError)
      << "scalar 'key_bits' value " << arg_key_bits
      << " is outside i32 range [-2147483648, 2147483647]";
  check_same_device(arg_Scores, arg_Indices, "Scores", "Indices");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  int* p_Indices = static_cast<int*>(arg_Indices.data_ptr());
  float* p_Scores = static_cast<float*>(arg_Scores.data_ptr());
  int32_t v_top_k = (int32_t)arg_top_k;
  int32_t v_key_bits = (int32_t)arg_key_bits;
  void* kargs[] = {&p_Indices, &p_Scores, &v_top_k, &v_key_bits};


  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(CAKE_UNSIGNED(THREADS), 1u, 1u);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = CAKE_UNSIGNED(SMEM_TOTAL);
  config.stream = stream;
  cudaError_t launch_status = cudaLaunchKernelExC(
      &config, reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), kargs);
  TVM_FFI_CHECK(launch_status == cudaSuccess, RuntimeError)
      << "cudaLaunchKernelExC for " CAKE_STRINGIZE(CAKE_PROGRAM_KERNEL) " failed: "
      << cudaGetErrorString(launch_status);
}

}  // namespace CAKE_PROGRAM_SHIM

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, CAKE_PROGRAM_SHIM::Run);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_cuda_version, cake_ffi_build_cuda_version);
