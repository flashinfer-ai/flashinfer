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

// Standalone entry points of the Cake StepFun FC1 stage. The fused-MoE operations of
// this module run the same kernels through MoE::Runner; these operations expose the
// FC1 stage alone over trtllm-gen routing metadata so the exported kernels can be
// validated and timed against their Cake source programs on identical operands.

#ifndef CAKE_STEPFUN_FC1
#error "cake_stepfun_moe_binding.cu is part of the fused_moe_cake_stepfun_* module only"
#endif

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "flashinfer/trtllm/fused_moe/runner.h"
#include "generated/cake_stepfun_generated_manifest.cuh"
#include "tvm_ffi_utils.h"

namespace {

using tvm::ffi::Array;
using tvm::ffi::TensorView;
namespace moe = tensorrt_llm::kernels::trtllmgen_moe;
namespace btg = batchedGemm::trtllm::gen;
namespace generated = flashinfer::cake_stepfun::generated;

void checkTensor(TensorView const& tensor, char const* name, DLDataType dtype, int64_t ndim,
                 DLDevice device) {
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype) << "cake_stepfun_fc1_nvfp4: " << name
                                            << " has an unexpected dtype.";
  TVM_FFI_ICHECK_EQ(tensor.ndim(), ndim) << "cake_stepfun_fc1_nvfp4: " << name << " must be "
                                          << ndim << "-D.";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << "cake_stepfun_fc1_nvfp4: " << name
                                        << " must be contiguous.";
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA &&
                 tensor.device().device_id == device.device_id)
      << "cake_stepfun_fc1_nvfp4: " << name << " must live on the launch device.";
}

}  // namespace

/** Tile sizes (tokens per CTA) served by the exported Cake StepFun FC1 kernels. */
Array<int64_t> cake_stepfun_fc1_tiles() {
  Array<int64_t> tiles;
  for (size_t index = 0; index < generated::kFc1KernelCount; ++index) {
    tiles.push_back(generated::kFc1Kernels[index].tile_n);
  }
  return tiles;
}

/**
 * Run the Cake StepFun NVFP4 FC1 stage over trtllm-gen routing metadata.
 *
 * The operands are the fused-MoE FC1 inputs: packed E2m1 activations with linear E4M3
 * block scales, trtllm-shuffled MajorK E2m1 weights with 128x4 block scales, the three
 * per-expert FP32 vectors (output scale, gate scale, raw step limit) and the routing
 * arrays produced by the module's routing operation for ``tile_tokens_dim``. The result
 * is the GEMM2 input: E2m1 ``[max_padded_tokens, intermediate_size / 2]`` with 8x4
 * block scales in ``gemm1_output_scale``.
 */
void cake_stepfun_fc1_nvfp4(TensorView const& hidden_states, TensorView const& hidden_states_scale,
                            TensorView const& gemm1_weights, TensorView const& gemm1_weights_scale,
                            TensorView const& output1_scale_scalar,
                            TensorView const& output1_scale_gate_scalar,
                            TensorView const& gemm1_clamp_limit,
                            TensorView const& permuted_idx_to_token_idx,
                            TensorView const& cta_idx_xy_to_batch_idx,
                            TensorView const& cta_idx_xy_to_mn_limit,
                            TensorView const& num_non_exiting_ctas,
                            TensorView const& total_num_padded_tokens, TensorView const& gemm1_output,
                            TensorView const& gemm1_output_scale, int64_t top_k,
                            int64_t tile_tokens_dim, bool enable_pdl) {
  DLDevice const device = hidden_states.device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA)
      << "cake_stepfun_fc1_nvfp4: hidden_states must be a CUDA tensor.";
  checkTensor(hidden_states, "hidden_states", dl_uint8, 2, device);
  checkTensor(hidden_states_scale, "hidden_states_scale", dl_float8_e4m3fn, 2, device);
  checkTensor(gemm1_weights, "gemm1_weights", dl_uint8, 3, device);
  checkTensor(gemm1_weights_scale, "gemm1_weights_scale", dl_float8_e4m3fn, 3, device);
  checkTensor(output1_scale_scalar, "output1_scale_scalar", dl_float32, 1, device);
  checkTensor(output1_scale_gate_scalar, "output1_scale_gate_scalar", dl_float32, 1, device);
  checkTensor(gemm1_clamp_limit, "gemm1_clamp_limit", dl_float32, 1, device);
  checkTensor(permuted_idx_to_token_idx, "permuted_idx_to_token_idx", dl_int32, 1, device);
  checkTensor(cta_idx_xy_to_batch_idx, "cta_idx_xy_to_batch_idx", dl_int32, 1, device);
  checkTensor(cta_idx_xy_to_mn_limit, "cta_idx_xy_to_mn_limit", dl_int32, 1, device);
  checkTensor(num_non_exiting_ctas, "num_non_exiting_ctas", dl_int32, 1, device);
  checkTensor(total_num_padded_tokens, "total_num_padded_tokens", dl_int32, 1, device);
  checkTensor(gemm1_output, "gemm1_output", dl_uint8, 2, device);
  checkTensor(gemm1_output_scale, "gemm1_output_scale", dl_uint8, 1, device);

  int64_t const num_tokens = hidden_states.size(0);
  int64_t const hidden_size = hidden_states.size(1) * 2;
  int64_t const num_experts = gemm1_weights.size(0);
  int64_t const intermediate_size = gemm1_weights.size(1) / 2;
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(1), 2 * intermediate_size)
      << "cake_stepfun_fc1_nvfp4: gemm1_weights rows must be 2 * intermediate_size.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.size(2) * 2, hidden_size)
      << "cake_stepfun_fc1_nvfp4: gemm1_weights columns must be hidden_size / 2.";
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(0), num_tokens)
      << "cake_stepfun_fc1_nvfp4: hidden_states_scale rows must match num_tokens.";
  TVM_FFI_ICHECK_EQ(hidden_states_scale.size(1) * 16, hidden_size)
      << "cake_stepfun_fc1_nvfp4: hidden_states_scale columns must be hidden_size / 16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(0), num_experts)
      << "cake_stepfun_fc1_nvfp4: gemm1_weights_scale experts must match gemm1_weights.";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(1), 2 * intermediate_size)
      << "cake_stepfun_fc1_nvfp4: gemm1_weights_scale rows must be 2 * intermediate_size.";
  TVM_FFI_ICHECK_EQ(gemm1_weights_scale.size(2) * 16, hidden_size)
      << "cake_stepfun_fc1_nvfp4: gemm1_weights_scale columns must be hidden_size / 16.";
  for (TensorView const* vector : {&output1_scale_scalar, &output1_scale_gate_scalar,
                                   &gemm1_clamp_limit}) {
    TVM_FFI_ICHECK_EQ(vector->size(0), num_experts)
        << "cake_stepfun_fc1_nvfp4: per-expert vectors must hold one value per expert.";
  }
  TVM_FFI_ICHECK_EQ(num_non_exiting_ctas.size(0), 1)
      << "cake_stepfun_fc1_nvfp4: num_non_exiting_ctas must hold one value.";
  TVM_FFI_ICHECK_EQ(total_num_padded_tokens.size(0), 1)
      << "cake_stepfun_fc1_nvfp4: total_num_padded_tokens must hold one value.";
  TVM_FFI_ICHECK_EQ(gemm1_output.size(1) * 2, intermediate_size)
      << "cake_stepfun_fc1_nvfp4: gemm1_output columns must be intermediate_size / 2.";
  int64_t const max_padded_tokens = moe::Routing::getMaxPermutedPaddedCount(
      static_cast<int32_t>(num_tokens), static_cast<int32_t>(top_k),
      static_cast<int32_t>(num_experts), static_cast<int32_t>(tile_tokens_dim));
  TVM_FFI_ICHECK_EQ(gemm1_output.size(0), max_padded_tokens)
      << "cake_stepfun_fc1_nvfp4: gemm1_output rows must be the maximum padded token count "
      << max_padded_tokens << ".";
  TVM_FFI_ICHECK_EQ(gemm1_output_scale.size(0), max_padded_tokens * intermediate_size / 16)
      << "cake_stepfun_fc1_nvfp4: gemm1_output_scale must hold max_padded_tokens * "
         "intermediate_size / 16 bytes.";
  int64_t const max_ctas = moe::Routing::getMaxNumCtasInBatchDim(
      static_cast<int32_t>(num_tokens), static_cast<int32_t>(top_k),
      static_cast<int32_t>(num_experts), static_cast<int32_t>(tile_tokens_dim));
  TVM_FFI_ICHECK_GE(cta_idx_xy_to_batch_idx.size(0), max_ctas)
      << "cake_stepfun_fc1_nvfp4: cta_idx_xy_to_batch_idx is shorter than the CTA grid.";
  TVM_FFI_ICHECK_GE(cta_idx_xy_to_mn_limit.size(0), max_ctas)
      << "cake_stepfun_fc1_nvfp4: cta_idx_xy_to_mn_limit is shorter than the CTA grid.";
  TVM_FFI_ICHECK_GE(permuted_idx_to_token_idx.size(0), max_padded_tokens)
      << "cake_stepfun_fc1_nvfp4: permuted_idx_to_token_idx is shorter than the padded count.";

  moe::cake_stepfun::Fc1Runner runner(
      btg::Dtype::E2m1, btg::Dtype::E2m1, btg::Dtype::E2m1, /*useDeepSeekFp8=*/false,
      static_cast<int>(tile_tokens_dim), moe::MoE::ActivationType::SwigluStep,
      /*useShuffledMatrix=*/true, batchedGemm::gemm::MatrixLayout::MajorK,
      batchedGemm::gemm::BiasType::None, /*usePerTokenScaling=*/false,
      /*usePerChannelScaling=*/false);
  TVM_FFI_ICHECK(runner.usesCakeKernels())
      << "cake_stepfun_fc1_nvfp4: no exported Cake kernel serves tile_tokens_dim "
      << tile_tokens_dim << ".";
  int32_t const config_index = runner.getDefaultValidConfigIndex(
      static_cast<int32_t>(top_k), static_cast<int32_t>(hidden_size),
      static_cast<int32_t>(intermediate_size), static_cast<int32_t>(num_experts),
      static_cast<int32_t>(num_tokens));
  TVM_FFI_ICHECK_GE(config_index, 0)
      << "cake_stepfun_fc1_nvfp4: no exported Cake kernel accepts hidden_size " << hidden_size
      << " and intermediate_size " << intermediate_size << ".";

  cudaStream_t const stream = get_stream(device);
  runner.run(hidden_states.data_ptr(), hidden_states_scale.data_ptr(), gemm1_weights.data_ptr(),
             gemm1_weights_scale.data_ptr(), /*perTokenScales=*/nullptr,
             /*perChannelScales=*/nullptr, static_cast<float*>(output1_scale_scalar.data_ptr()),
             static_cast<float*>(output1_scale_gate_scalar.data_ptr()), /*ptrBias=*/nullptr,
             /*ptrGatedActAlpha=*/nullptr, /*ptrGatedActBeta=*/nullptr,
             static_cast<float*>(gemm1_clamp_limit.data_ptr()),
             /*permutedIdxToBiasRowIdx=*/nullptr, gemm1_output.data_ptr(),
             gemm1_output_scale.data_ptr(), static_cast<int32_t>(top_k),
             static_cast<int32_t>(hidden_size), static_cast<int32_t>(intermediate_size),
             static_cast<int32_t>(num_experts), static_cast<int32_t>(num_tokens),
             static_cast<int32_t*>(permuted_idx_to_token_idx.data_ptr()),
             static_cast<int32_t*>(num_non_exiting_ctas.data_ptr()),
             static_cast<int32_t*>(total_num_padded_tokens.data_ptr()),
             static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr()),
             static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr()), /*bmm1Workspace=*/nullptr,
             /*useRoutingScalesOnInput=*/false, device.device_id, stream, config_index,
             enable_pdl);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc1_tiles, cake_stepfun_fc1_tiles);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc1_nvfp4, cake_stepfun_fc1_nvfp4);
