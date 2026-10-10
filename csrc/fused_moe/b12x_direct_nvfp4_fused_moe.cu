/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstdint>

#include "flashinfer/fused_moe/b12x_direct_quantized/nvfp4_fused_moe.cuh"
#include "tvm_ffi_utils.h"

namespace flashinfer::fused_moe {

using tvm::ffi::TensorView;

void B12xDirectNVFP4FusedMoe(TensorView hidden_states, TensorView topk_ids, TensorView topk_weights,
                             TensorView gemm1_weights, TensorView gemm1_scales,
                             TensorView gemm2_weights, TensorView gemm2_scales,
                             TensorView expert_map, TensorView hidden_quantized,
                             TensorView hidden_scales, TensorView intermediate_quantized,
                             TensorView intermediate_scales, TensorView output,
                             int64_t outputs_per_warp, int64_t num_threads,
                             double hidden_global_encode_scale,
                             double intermediate_global_encode_scale, int64_t run_down) {
  CHECK_INPUT_AND_TYPE(hidden_states, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(topk_ids, dl_int32);
  CHECK_INPUT_AND_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_AND_TYPE(gemm1_weights, dl_uint8);
  CHECK_INPUT_AND_TYPE(gemm1_scales, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(gemm2_weights, dl_uint8);
  CHECK_INPUT_AND_TYPE(gemm2_scales, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(expert_map, dl_int32);
  CHECK_INPUT_AND_TYPE(hidden_quantized, dl_uint8);
  CHECK_INPUT_AND_TYPE(hidden_scales, dl_uint8);
  CHECK_INPUT_AND_TYPE(intermediate_quantized, dl_uint8);
  CHECK_INPUT_AND_TYPE(intermediate_scales, dl_bfloat16);
  CHECK_INPUT_AND_TYPE(output, dl_bfloat16);

  CHECK_DIM(2, hidden_states);
  CHECK_DIM(2, topk_ids);
  CHECK_DIM(2, topk_weights);
  CHECK_DIM(3, gemm1_weights);
  CHECK_DIM(3, gemm1_scales);
  CHECK_DIM(3, gemm2_weights);
  CHECK_DIM(3, gemm2_scales);
  CHECK_DIM(1, expert_map);
  CHECK_DIM(2, hidden_quantized);
  CHECK_DIM(2, hidden_scales);
  CHECK_DIM(2, intermediate_quantized);
  CHECK_DIM(2, intermediate_scales);
  CHECK_DIM(2, output);

  CHECK_DEVICE(topk_ids, hidden_states);
  CHECK_DEVICE(topk_weights, hidden_states);
  CHECK_DEVICE(gemm1_weights, hidden_states);
  CHECK_DEVICE(gemm1_scales, hidden_states);
  CHECK_DEVICE(gemm2_weights, hidden_states);
  CHECK_DEVICE(gemm2_scales, hidden_states);
  CHECK_DEVICE(expert_map, hidden_states);
  CHECK_DEVICE(hidden_quantized, hidden_states);
  CHECK_DEVICE(hidden_scales, hidden_states);
  CHECK_DEVICE(intermediate_quantized, hidden_states);
  CHECK_DEVICE(intermediate_scales, hidden_states);
  CHECK_DEVICE(output, hidden_states);

  const int64_t num_tokens = hidden_states.size(0);
  const int64_t hidden_size = hidden_states.size(1);
  const int64_t topk = topk_ids.size(1);
  const int64_t num_local_experts = gemm1_weights.size(0);
  const int64_t intermediate_size = gemm2_weights.size(2) * 2;
  const int64_t expert_map_items = expert_map.numel();
  const int64_t routed_rows = num_tokens * topk;

  TVM_FFI_ICHECK(num_tokens >= 1 && num_tokens <= 8)
      << "num_tokens must be in [1, 8] for the SM120 direct decode path";
  TVM_FFI_ICHECK(topk >= 1 && topk <= 8) << "topk must be in [1, 8]";
  TVM_FFI_ICHECK(num_local_experts >= 1) << "at least one local expert is required";
  TVM_FFI_ICHECK(hidden_size >= 16 && hidden_size <= 8192 && hidden_size % 16 == 0)
      << "hidden_size must be a multiple of 16 in [16, 8192]";
  TVM_FFI_ICHECK(intermediate_size >= 16 && intermediate_size <= 1024 &&
                 intermediate_size % 16 == 0)
      << "intermediate_size must be a multiple of 16 in [16, 1024]";
  TVM_FFI_ICHECK(outputs_per_warp == 1 || outputs_per_warp == 2 || outputs_per_warp == 4 ||
                 outputs_per_warp == 8)
      << "outputs_per_warp must be one of 1, 2, 4, or 8";
  TVM_FFI_ICHECK(num_threads >= 64 && num_threads <= 1024 && num_threads % 32 == 0)
      << "num_threads must be a warp multiple in [64, 1024]";
  TVM_FFI_ICHECK(hidden_global_encode_scale > 0.0 && intermediate_global_encode_scale > 0.0)
      << "activation global encode scales must be positive";

  TVM_FFI_ICHECK_EQ(topk_ids.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(topk_weights.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(topk_weights.size(1), topk);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(1), 2 * intermediate_size);
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(2), hidden_size / 2);
  TVM_FFI_ICHECK_EQ(gemm1_scales.size(0), num_local_experts);
  TVM_FFI_ICHECK_EQ(gemm1_scales.size(1), 2 * intermediate_size);
  TVM_FFI_ICHECK_EQ(gemm1_scales.size(2), hidden_size / 16);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(0), num_local_experts);
  TVM_FFI_ICHECK_EQ(gemm2_weights.size(1), hidden_size);
  TVM_FFI_ICHECK_EQ(gemm2_scales.size(0), num_local_experts);
  TVM_FFI_ICHECK_EQ(gemm2_scales.size(1), hidden_size);
  TVM_FFI_ICHECK_EQ(gemm2_scales.size(2), intermediate_size / 16);
  TVM_FFI_ICHECK_EQ(hidden_quantized.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(hidden_quantized.size(1), hidden_size / 2);
  TVM_FFI_ICHECK_EQ(hidden_scales.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(hidden_scales.size(1), hidden_size / 16);
  TVM_FFI_ICHECK_EQ(intermediate_quantized.size(0), routed_rows);
  TVM_FFI_ICHECK_EQ(intermediate_quantized.size(1), intermediate_size / 2);
  TVM_FFI_ICHECK_EQ(intermediate_scales.size(0), routed_rows);
  TVM_FFI_ICHECK_EQ(intermediate_scales.size(1), intermediate_size / 16);
  TVM_FFI_ICHECK_EQ(output.size(0), num_tokens);
  TVM_FFI_ICHECK_EQ(output.size(1), hidden_size);
  TVM_FFI_ICHECK(expert_map_items == 0 || expert_map_items >= num_local_experts)
      << "expert_map must be empty or a global-to-local map";

  ffi::CUDADeviceGuard device_guard(hidden_states.device().device_id);
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
  TVM_FFI_ICHECK(properties.major == 12 && properties.minor == 0)
      << "b12x_direct_nvfp4_fused_moe requires compute capability 12.0";

  B12xDirectNVFP4FusedMoeParams params{
      static_cast<const uint8_t*>(gemm1_weights.data_ptr()),
      static_cast<const __nv_bfloat16*>(gemm1_scales.data_ptr()),
      static_cast<const uint8_t*>(gemm2_weights.data_ptr()),
      static_cast<const __nv_bfloat16*>(gemm2_scales.data_ptr()),
      static_cast<const int32_t*>(topk_ids.data_ptr()),
      static_cast<const int32_t*>(expert_map.data_ptr()),
      static_cast<const float*>(topk_weights.data_ptr()),
      static_cast<uint8_t*>(hidden_quantized.data_ptr()),
      static_cast<uint8_t*>(hidden_scales.data_ptr()),
      static_cast<uint8_t*>(intermediate_quantized.data_ptr()),
      static_cast<__nv_bfloat16*>(intermediate_scales.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()),
      static_cast<int32_t>(num_tokens),
      static_cast<int32_t>(topk),
      static_cast<int32_t>(num_local_experts),
      static_cast<int32_t>(expert_map_items),
      static_cast<int32_t>(hidden_size),
      static_cast<int32_t>(intermediate_size),
      static_cast<int32_t>(outputs_per_warp),
      static_cast<int32_t>(num_threads),
      static_cast<float>(1.0 / hidden_global_encode_scale),
      static_cast<float>(intermediate_global_encode_scale),
  };
  status = LaunchB12xDirectNVFP4FusedMoe(params, get_stream(hidden_states.device()), run_down != 0);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "b12x_direct_nvfp4_fused_moe kernel launch failed: " << cudaGetErrorString(status);
}

}  // namespace flashinfer::fused_moe
