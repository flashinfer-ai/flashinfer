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

// Test-only binding for the finalize step of the CUTLASS fused MoE (top-k unpermute + weighted
// reduction). It runs either the production launcher (finalizeMoeRoutingKernelLauncher, which picks
// finalizeMoeRoutingWarpKernel when it can) or the block-per-row kernels it falls back to, so that
// tests/moe/test_cutlass_moe_finalize.py can compare their outputs bit for bit.

#include "fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh"
#include "tvm_ffi_utils.h"

namespace {

using namespace tensorrt_llm::kernels::cutlass_kernels;
using tvm::ffi::Optional;

template <class T>
void RunFinalize(TensorView gemm_output, TensorView output, Optional<TensorView> bias,
                 Optional<TensorView> final_scales, TensorView unpermuted_row_to_permuted_row,
                 TensorView permuted_row_to_unpermuted_row, TensorView token_selected_experts,
                 TensorView expert_first_token_offset, int64_t num_rows, int64_t padded_cols,
                 int64_t unpadded_cols, int64_t experts_per_token, int64_t num_experts_per_node,
                 MOEParallelismConfig parallelism_config, bool enable_alltoall, bool enable_pdl,
                 bool block_kernels, cudaStream_t stream) {
  T const* bias_ptr = bias.has_value() ? static_cast<T const*>(bias.value().data_ptr()) : nullptr;
  float const* scales_ptr = final_scales.has_value()
                                ? static_cast<float const*>(final_scales.value().data_ptr())
                                : nullptr;
  auto const launcher = block_kernels ? &finalizeMoeRoutingBlockKernelLauncher<T, T, T>
                                      : &finalizeMoeRoutingKernelLauncher<T, T, T>;
  launcher(static_cast<T const*>(gemm_output.data_ptr()), static_cast<T*>(output.data_ptr()),
           bias_ptr, scales_ptr, static_cast<int const*>(unpermuted_row_to_permuted_row.data_ptr()),
           static_cast<int const*>(permuted_row_to_unpermuted_row.data_ptr()),
           static_cast<int const*>(token_selected_experts.data_ptr()),
           static_cast<int64_t const*>(expert_first_token_offset.data_ptr()), num_rows, padded_cols,
           unpadded_cols, experts_per_token, num_experts_per_node, parallelism_config,
           enable_alltoall, enable_pdl, stream);
}

void FinalizeMoeRouting(TensorView gemm_output, TensorView output, Optional<TensorView> bias,
                        Optional<TensorView> final_scales,
                        TensorView unpermuted_row_to_permuted_row,
                        TensorView permuted_row_to_unpermuted_row,
                        TensorView token_selected_experts, TensorView expert_first_token_offset,
                        int64_t num_rows, int64_t padded_cols, int64_t unpadded_cols,
                        int64_t experts_per_token, int64_t num_experts_per_node, int64_t tp_size,
                        int64_t tp_rank, int64_t ep_size, int64_t ep_rank, bool enable_alltoall,
                        bool enable_pdl, bool block_kernels) {
  TVM_FFI_ICHECK(gemm_output.dtype() == output.dtype())
      << "gemm_output and output must have the same dtype";
  TVM_FFI_ICHECK(!bias.has_value() || bias.value().dtype() == output.dtype())
      << "bias must have the output dtype";
  ffi::CUDADeviceGuard device_guard(output.device().device_id);
  cudaStream_t const stream = get_stream(output.device());
  MOEParallelismConfig const parallelism_config(
      static_cast<int>(tp_size), static_cast<int>(tp_rank), static_cast<int>(ep_size),
      static_cast<int>(ep_rank));
  auto const run = [&](auto type_tag) {
    using T = decltype(type_tag);
    RunFinalize<T>(gemm_output, output, bias, final_scales, unpermuted_row_to_permuted_row,
                   permuted_row_to_unpermuted_row, token_selected_experts,
                   expert_first_token_offset, num_rows, padded_cols, unpadded_cols,
                   experts_per_token, num_experts_per_node, parallelism_config, enable_alltoall,
                   enable_pdl, block_kernels, stream);
  };
  switch (encode_dlpack_dtype(output.dtype())) {
    case float16_code:
      run(half{});
      break;
    case bfloat16_code:
      run(__nv_bfloat16{});
      break;
    case float32_code:
      run(float{});
      break;
    default:
      TVM_FFI_THROW(TypeError) << "Unsupported output dtype for the MoE finalize test binding";
  }
  cudaError_t const status = cudaGetLastError();
  TVM_FFI_ICHECK(status == cudaSuccess) << "finalize launch failed: " << cudaGetErrorString(status);
}

// Whether finalizeMoeRoutingKernelLauncher runs finalizeMoeRoutingWarpKernel for these arguments.
bool UsesWarpKernel(TensorView gemm_output, TensorView output, Optional<TensorView> bias,
                    int64_t padded_cols, int64_t unpadded_cols, int64_t experts_per_token) {
  auto const check = [&](auto type_tag) {
    using T = decltype(type_tag);
    T const* bias_ptr = bias.has_value() ? static_cast<T const*>(bias.value().data_ptr()) : nullptr;
    return canUseFinalizeMoeRoutingWarpKernel(static_cast<T const*>(gemm_output.data_ptr()),
                                              static_cast<T const*>(output.data_ptr()), bias_ptr,
                                              padded_cols, unpadded_cols, experts_per_token);
  };
  switch (encode_dlpack_dtype(output.dtype())) {
    case float16_code:
      return check(half{});
    case bfloat16_code:
      return check(__nv_bfloat16{});
    case float32_code:
      return check(float{});
    default:
      TVM_FFI_THROW(TypeError) << "Unsupported output dtype for the MoE finalize test binding";
  }
  return false;
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(finalize_moe_routing, FinalizeMoeRouting);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(uses_warp_kernel, UsesWarpKernel);
