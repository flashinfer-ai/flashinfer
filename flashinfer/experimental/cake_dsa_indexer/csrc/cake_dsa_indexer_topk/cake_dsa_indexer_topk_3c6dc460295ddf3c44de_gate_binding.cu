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
// Gate-warpgroup launch form (block 512, dynamic smem 221184) of the same launcher for the sm_100a / sm_103a wide
// programs; the 384-thread form stays in cake_dsa_indexer_topk_3c6dc460295ddf3c44de_binding.cu for the programs that keep it.
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cake_ffi_checks.h"

#include <cstdint>

#include "cake_dsa_indexer_topk_host_common.cuh"

// Program selection.  Programs whose exported binding text differs from this file's
// only in the kernel symbol and the host-shim namespace build this translation unit
// with -DCAKE_PROGRAM_KERNEL / -DCAKE_PROGRAM_SHIM naming their own; the defaults
// select the program this file was exported for.
#ifndef CAKE_PROGRAM_KERNEL
#define CAKE_PROGRAM_KERNEL kernel_cake_dsa_indexer_topk_3c6dc460295ddf3c44de
#endif
#ifndef CAKE_PROGRAM_SHIM
#define CAKE_PROGRAM_SHIM cake_host_shim_01b7464a4c0d2d6c
#endif

extern "C" __global__ void CAKE_PROGRAM_KERNEL(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap W, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, long long* __restrict__ q_offsets, int* __restrict__ Indices, float* __restrict__ Scores, long long* __restrict__ Cand, int num_segments, int top_k, int ratio, int has_offsets, int cand_cap, int first_cap, int sample_tiles_max, int sample_shift_permille, int far_sample, int check_period, int grid_ctas, float softmax_scale, int n_split);


namespace CAKE_PROGRAM_SHIM {

using namespace cake_host_shim_common;
using tvm::ffi::TensorView;

// The TMA descriptor encoders are the same text in every scan binding of this box
// shape; cake_dsa_indexer_topk_host_common.cuh holds them once (CAKE_HOST_COMMON_PART).
#define CAKE_HOST_COMMON_PART 3  // EncodeTma_Q, box {64u, 256u, 2u}
#include "cake_dsa_indexer_topk_host_common.cuh"

#define CAKE_HOST_COMMON_PART 5  // EncodeTma_K, box {64u, 128u, 2u}
#include "cake_dsa_indexer_topk_host_common.cuh"

#define CAKE_HOST_COMMON_PART 8  // EncodeTma_W, box {32u, 8u}
#include "cake_dsa_indexer_topk_host_common.cuh"

void Run(TensorView arg_Q, TensorView arg_K, TensorView arg_W, TensorView arg_cu_seqlens_q, TensorView arg_cu_seqlens_k, TensorView arg_q_offsets, TensorView arg_Indices, TensorView arg_Scores, TensorView arg_Cand, int64_t arg_num_segments, int64_t arg_top_k, int64_t arg_ratio, int64_t arg_has_offsets, int64_t arg_cand_cap, int64_t arg_first_cap, int64_t arg_sample_tiles_max, int64_t arg_sample_shift_permille, int64_t arg_far_sample, int64_t arg_check_period, int64_t arg_grid_ctas, double arg_softmax_scale, int64_t arg_n_split, int64_t grid_x, int64_t grid_y, int64_t grid_z) {
  DLDevice dev = arg_Q.device();
  tvm::ffi::CUDADeviceGuard device_guard(dev.device_id);
  {
    // Bind the primary context for the tensor-map encoders once per thread and device.
    static thread_local int encoder_context_device = -1;
    if (encoder_context_device != dev.device_id) {
      TVM_FFI_CHECK_CUDA_ERROR(cudaSetDevice(dev.device_id));
      encoder_context_device = dev.device_id;
    }
  }
  check_cuda_tensor(arg_Q, "Q");
  check_dtype(arg_Q, DLDataType{kDLBfloat, 16, 1}, "Q");
  check_cuda_tensor(arg_K, "K");
  check_dtype(arg_K, DLDataType{kDLBfloat, 16, 1}, "K");
  check_cuda_tensor(arg_W, "W");
  check_dtype(arg_W, DLDataType{kDLFloat, 32, 1}, "W");
  check_cuda_tensor(arg_cu_seqlens_q, "cu_seqlens_q");
  check_dtype(arg_cu_seqlens_q, DLDataType{kDLInt, 32, 1}, "cu_seqlens_q");
  check_contiguous(arg_cu_seqlens_q, "cu_seqlens_q");
  check_cuda_tensor(arg_cu_seqlens_k, "cu_seqlens_k");
  check_dtype(arg_cu_seqlens_k, DLDataType{kDLInt, 32, 1}, "cu_seqlens_k");
  check_contiguous(arg_cu_seqlens_k, "cu_seqlens_k");
  check_cuda_tensor(arg_q_offsets, "q_offsets");
  check_dtype(arg_q_offsets, DLDataType{kDLInt, 64, 1}, "q_offsets");
  check_contiguous(arg_q_offsets, "q_offsets");
  check_cuda_tensor(arg_Indices, "Indices");
  check_dtype(arg_Indices, DLDataType{kDLInt, 32, 1}, "Indices");
  check_contiguous(arg_Indices, "Indices");
  check_cuda_tensor(arg_Scores, "Scores");
  check_dtype(arg_Scores, DLDataType{kDLFloat, 32, 1}, "Scores");
  check_contiguous(arg_Scores, "Scores");
  check_cuda_tensor(arg_Cand, "Cand");
  check_dtype(arg_Cand, DLDataType{kDLInt, 64, 1}, "Cand");
  check_contiguous(arg_Cand, "Cand");
  TVM_FFI_CHECK(arg_num_segments >= -2147483648LL && arg_num_segments <= 2147483647LL, ValueError)
      << "scalar 'num_segments' value " << arg_num_segments
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_top_k >= -2147483648LL && arg_top_k <= 2147483647LL, ValueError)
      << "scalar 'top_k' value " << arg_top_k
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_ratio >= -2147483648LL && arg_ratio <= 2147483647LL, ValueError)
      << "scalar 'ratio' value " << arg_ratio
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_has_offsets >= -2147483648LL && arg_has_offsets <= 2147483647LL, ValueError)
      << "scalar 'has_offsets' value " << arg_has_offsets
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_cand_cap >= -2147483648LL && arg_cand_cap <= 2147483647LL, ValueError)
      << "scalar 'cand_cap' value " << arg_cand_cap
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_first_cap >= -2147483648LL && arg_first_cap <= 2147483647LL, ValueError)
      << "scalar 'first_cap' value " << arg_first_cap
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_sample_tiles_max >= -2147483648LL && arg_sample_tiles_max <= 2147483647LL, ValueError)
      << "scalar 'sample_tiles_max' value " << arg_sample_tiles_max
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_sample_shift_permille >= -2147483648LL && arg_sample_shift_permille <= 2147483647LL, ValueError)
      << "scalar 'sample_shift_permille' value " << arg_sample_shift_permille
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_far_sample >= -2147483648LL && arg_far_sample <= 2147483647LL, ValueError)
      << "scalar 'far_sample' value " << arg_far_sample
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_check_period >= -2147483648LL && arg_check_period <= 2147483647LL, ValueError)
      << "scalar 'check_period' value " << arg_check_period
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_grid_ctas >= -2147483648LL && arg_grid_ctas <= 2147483647LL, ValueError)
      << "scalar 'grid_ctas' value " << arg_grid_ctas
      << " is outside i32 range [-2147483648, 2147483647]";
  TVM_FFI_CHECK(arg_n_split >= -2147483648LL && arg_n_split <= 2147483647LL, ValueError)
      << "scalar 'n_split' value " << arg_n_split
      << " is outside i32 range [-2147483648, 2147483647]";
  check_same_device(arg_K, arg_Q, "K", "Q");
  check_same_device(arg_W, arg_Q, "W", "Q");
  check_same_device(arg_cu_seqlens_q, arg_Q, "cu_seqlens_q", "Q");
  check_same_device(arg_cu_seqlens_k, arg_Q, "cu_seqlens_k", "Q");
  check_same_device(arg_q_offsets, arg_Q, "q_offsets", "Q");
  check_same_device(arg_Indices, arg_Q, "Indices", "Q");
  check_same_device(arg_Scores, arg_Q, "Scores", "Q");
  check_same_device(arg_Cand, arg_Q, "Cand", "Q");
  TVM_FFI_CHECK(grid_x > 0 && grid_y > 0 && grid_z > 0, ValueError)
      << "launch grid dimensions must be positive, got (" << grid_x << ", " << grid_y
      << ", " << grid_z << ")";

  cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(dev.device_type, dev.device_id);
  CUtensorMap p_Q = EncodeTma_Q(arg_Q);
  CUtensorMap p_K = EncodeTma_K(arg_K);
  CUtensorMap p_W = EncodeTma_W(arg_W);
  int* p_cu_seqlens_q = static_cast<int*>(arg_cu_seqlens_q.data_ptr());
  int* p_cu_seqlens_k = static_cast<int*>(arg_cu_seqlens_k.data_ptr());
  long long* p_q_offsets = static_cast<long long*>(arg_q_offsets.data_ptr());
  int* p_Indices = static_cast<int*>(arg_Indices.data_ptr());
  float* p_Scores = static_cast<float*>(arg_Scores.data_ptr());
  long long* p_Cand = static_cast<long long*>(arg_Cand.data_ptr());
  int32_t v_num_segments = (int32_t)arg_num_segments;
  int32_t v_top_k = (int32_t)arg_top_k;
  int32_t v_ratio = (int32_t)arg_ratio;
  int32_t v_has_offsets = (int32_t)arg_has_offsets;
  int32_t v_cand_cap = (int32_t)arg_cand_cap;
  int32_t v_first_cap = (int32_t)arg_first_cap;
  int32_t v_sample_tiles_max = (int32_t)arg_sample_tiles_max;
  int32_t v_sample_shift_permille = (int32_t)arg_sample_shift_permille;
  int32_t v_far_sample = (int32_t)arg_far_sample;
  int32_t v_check_period = (int32_t)arg_check_period;
  int32_t v_grid_ctas = (int32_t)arg_grid_ctas;
  float v_softmax_scale = (float)arg_softmax_scale;
  int32_t v_n_split = (int32_t)arg_n_split;
  void* kargs[] = {&p_Q, &p_K, &p_W, &p_cu_seqlens_q, &p_cu_seqlens_k, &p_q_offsets, &p_Indices, &p_Scores, &p_Cand, &v_num_segments, &v_top_k, &v_ratio, &v_has_offsets, &v_cand_cap, &v_first_cap, &v_sample_tiles_max, &v_sample_shift_permille, &v_far_sample, &v_check_period, &v_grid_ctas, &v_softmax_scale, &v_n_split};

  static const bool smem_ready = CakeSetMaxDynamicSmem(
      reinterpret_cast<const void*>(CAKE_PROGRAM_KERNEL), 221312);
  (void)smem_ready;
  dim3 grid((uint32_t)grid_x, (uint32_t)grid_y, (uint32_t)grid_z);
  dim3 block(512u, 1u, 1u);

  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 221312u;
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
