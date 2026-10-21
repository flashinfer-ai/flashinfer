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

// Co-resident CTA capacity of one Cake fused norm-combine kernel.
//
// The persistent (wide) kernel pipelines tokens across a grid whose CTAs wait
// on each other, so the host may launch at most as many CTAs as the driver
// reports resident at once. That number is a property of the compiled kernel,
// so it is queried here, next to the kernel symbol, and cached per device by
// the Python loader, which sizes the grid from it. This translation unit is
// compiled into the same module as the kernel it names; the module's
// `-DCAKE_FUSED_NORM_COMBINE_KERNEL=<symbol>` flag selects that kernel.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "tvm_ffi_utils.h"

#ifndef CAKE_FUSED_NORM_COMBINE_KERNEL
#error "CAKE_FUSED_NORM_COMBINE_KERNEL must name the kernel symbol of this module"
#endif

extern "C" __global__ void CAKE_FUSED_NORM_COMBINE_KERNEL(
    __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ norm_out,
    __nv_bfloat16* __restrict__ residual_out, __nv_bfloat16* __restrict__ collective_out,
    unsigned long long* __restrict__ workspace, int rank, int tokens, float epsilon);

namespace {

int64_t MaxActiveBlocksPerSm(int64_t device_id, int64_t block_threads, int64_t dynamic_smem_bytes) {
  TVM_FFI_CHECK(device_id >= 0, ValueError) << "device_id must be a CUDA ordinal, got " << device_id;
  TVM_FFI_CHECK(block_threads > 0 && dynamic_smem_bytes >= 0, ValueError)
      << "block_threads must be positive and dynamic_smem_bytes non-negative, got " << block_threads
      << " / " << dynamic_smem_bytes;
  tvm::ffi::CUDADeviceGuard device_guard(static_cast<int>(device_id));
  int blocks = 0;
  cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, CAKE_FUSED_NORM_COMBINE_KERNEL, static_cast<int>(block_threads),
      static_cast<size_t>(dynamic_smem_bytes));
  TVM_FFI_CHECK(status == cudaSuccess, RuntimeError)
      << "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed: " << cudaGetErrorString(status);
  return blocks;
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(max_active_blocks_per_sm, MaxActiveBlocksPerSm);
