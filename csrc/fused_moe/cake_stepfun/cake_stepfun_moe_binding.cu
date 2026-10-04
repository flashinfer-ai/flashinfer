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
#include <string>
#include <vector>

#include "flashinfer/trtllm/fused_moe/runner.h"
#include "generated/cake_stepfun_generated_manifest.cuh"
#include "tvm_ffi_utils.h"

namespace {

using tvm::ffi::Array;
using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::ffi::TensorView;
namespace tgm = tensorrt_llm::kernels::trtllmgen_moe;
namespace btg = batchedGemm::trtllm::gen;
namespace generated = flashinfer::cake_stepfun::generated;

struct FamilySpec {
  char const* name;
  int index;
  btg::Dtype dtypeAct;
  btg::Dtype dtypeWeights;
  btg::Dtype dtypeOutput;
  batchedGemm::gemm::MatrixLayout weightLayout;
  bool perToken;
  // Activation elements per activation storage byte (2 for packed E2m1).
  int actPerByte;
};

constexpr FamilySpec kFamilies[] = {
    {"nvfp4", generated::kFc1Nvfp4, btg::Dtype::E2m1, btg::Dtype::E2m1, btg::Dtype::E2m1,
     batchedGemm::gemm::MatrixLayout::MajorK, false, 2},
    {"nvfp4_bf16tok", generated::kFc1Nvfp4PerToken, btg::Dtype::E2m1, btg::Dtype::E2m1,
     btg::Dtype::Bfloat16, batchedGemm::gemm::MatrixLayout::MajorK, true, 2},
    {"bf16", generated::kFc1Bf16, btg::Dtype::Bfloat16, btg::Dtype::Bfloat16, btg::Dtype::Bfloat16,
     batchedGemm::gemm::MatrixLayout::BlockMajorK, false, 1},
    {"fp8", generated::kFc1Fp8PerTensor, btg::Dtype::E4m3, btg::Dtype::E4m3, btg::Dtype::E4m3,
     batchedGemm::gemm::MatrixLayout::MajorK, false, 1},
    {"mxfp8", generated::kFc1MxFp8, btg::Dtype::MxE4m3, btg::Dtype::MxE4m3, btg::Dtype::MxE4m3,
     batchedGemm::gemm::MatrixLayout::MajorK, false, 1},
};

FamilySpec const& familySpec(String const& family) {
  std::string const name(family.data(), family.size());
  for (auto const& spec : kFamilies) {
    if (name == spec.name) return spec;
  }
  TVM_FFI_ICHECK(false) << "cake_stepfun_fc1: unknown family '" << name
                        << "' (nvfp4, nvfp4_bf16tok, bf16, fp8, mxfp8).";
  return kFamilies[0];
}

void checkTensor(TensorView const& tensor, char const* name, int64_t ndim, DLDevice device) {
  TVM_FFI_ICHECK_EQ(tensor.ndim(), ndim) << "cake_stepfun_fc1: " << name << " must be " << ndim
                                          << "-D.";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << "cake_stepfun_fc1: " << name << " must be contiguous.";
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA &&
                 tensor.device().device_id == device.device_id)
      << "cake_stepfun_fc1: " << name << " must live on the launch device.";
}

void checkDtype(TensorView const& tensor, char const* name, DLDataType dtype) {
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype) << "cake_stepfun_fc1: " << name
                                            << " has an unexpected dtype.";
}

void* optionalPtr(Optional<TensorView> const& tensor) {
  return tensor.has_value() ? tensor.value().data_ptr() : nullptr;
}

float* optionalFloatPtr(Optional<TensorView> const& tensor, char const* name, int64_t experts,
                        DLDevice device) {
  if (!tensor.has_value()) return nullptr;
  checkTensor(tensor.value(), name, 1, device);
  checkDtype(tensor.value(), name, dl_float32);
  TVM_FFI_ICHECK_EQ(tensor.value().size(0), experts)
      << "cake_stepfun_fc1: " << name << " must hold one value per expert.";
  return static_cast<float*>(tensor.value().data_ptr());
}

}  // namespace

/** Exported Cake StepFun FC1 families. */
Array<String> cake_stepfun_fc1_families() {
  Array<String> families;
  for (auto const& spec : kFamilies) {
    for (size_t index = 0; index < generated::kFc1KernelCount; ++index) {
      if (generated::kFc1Kernels[index].family == spec.index) {
        families.push_back(String(spec.name));
        break;
      }
    }
  }
  return families;
}

/** Tile sizes (tokens per CTA) served by the exported Cake StepFun FC1 kernels of ``family``. */
Array<int64_t> cake_stepfun_fc1_tiles(String const& family) {
  FamilySpec const& spec = familySpec(family);
  Array<int64_t> tiles;
  for (size_t index = 0; index < generated::kFc1KernelCount; ++index) {
    if (generated::kFc1Kernels[index].family == spec.index) {
      tiles.push_back(generated::kFc1Kernels[index].tile_n);
    }
  }
  return tiles;
}

/**
 * Run the Cake StepFun FC1 stage of ``family`` over trtllm-gen routing metadata.
 *
 * The operands are the fused-MoE FC1 inputs of the family exactly as MoE::Runner passes them
 * (``hidden_states`` [num_tokens, hidden storage], optional linear activation block scales,
 * the trtllm-prepared ``gemm1_weights`` with optional block scales, optional per-expert FP32
 * output / gate scales, the per-expert clamp limit, the optional fp32 per-token activation
 * scales of the ``nvfp4_bf16tok`` family) and the routing arrays produced by the module's
 * routing operation for ``tile_tokens_dim``. ``gemm1_output`` is the GEMM2 input of the family
 * ([max_padded_tokens, intermediate storage]); ``gemm1_output_scale`` receives its block scales
 * where the family has them.
 */
void cake_stepfun_fc1(String const& family, TensorView const& hidden_states,
                      Optional<TensorView> const& hidden_states_scale,
                      TensorView const& gemm1_weights,
                      Optional<TensorView> const& gemm1_weights_scale,
                      Optional<TensorView> const& output1_scale_scalar,
                      Optional<TensorView> const& output1_scale_gate_scalar,
                      TensorView const& gemm1_clamp_limit,
                      Optional<TensorView> const& per_token_scale,
                      TensorView const& permuted_idx_to_token_idx,
                      TensorView const& cta_idx_xy_to_batch_idx,
                      TensorView const& cta_idx_xy_to_mn_limit,
                      TensorView const& num_non_exiting_ctas,
                      TensorView const& total_num_padded_tokens, TensorView const& gemm1_output,
                      Optional<TensorView> const& gemm1_output_scale, int64_t top_k,
                      int64_t tile_tokens_dim, bool enable_pdl) {
  FamilySpec const& spec = familySpec(family);
  DLDevice const device = hidden_states.device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA)
      << "cake_stepfun_fc1: hidden_states must be a CUDA tensor.";
  checkTensor(hidden_states, "hidden_states", 2, device);
  checkTensor(gemm1_weights, "gemm1_weights", gemm1_weights.ndim(), device);
  checkTensor(gemm1_clamp_limit, "gemm1_clamp_limit", 1, device);
  checkDtype(gemm1_clamp_limit, "gemm1_clamp_limit", dl_float32);
  checkTensor(permuted_idx_to_token_idx, "permuted_idx_to_token_idx", 1, device);
  checkTensor(cta_idx_xy_to_batch_idx, "cta_idx_xy_to_batch_idx", 1, device);
  checkTensor(cta_idx_xy_to_mn_limit, "cta_idx_xy_to_mn_limit", 1, device);
  checkTensor(num_non_exiting_ctas, "num_non_exiting_ctas", 1, device);
  checkTensor(total_num_padded_tokens, "total_num_padded_tokens", 1, device);
  checkTensor(gemm1_output, "gemm1_output", 2, device);
  for (TensorView const* array : {&permuted_idx_to_token_idx, &cta_idx_xy_to_batch_idx,
                                  &cta_idx_xy_to_mn_limit, &num_non_exiting_ctas,
                                  &total_num_padded_tokens}) {
    checkDtype(*array, "routing arrays", dl_int32);
  }
  if (hidden_states_scale.has_value()) {
    checkTensor(hidden_states_scale.value(), "hidden_states_scale", 2, device);
  }
  if (gemm1_weights_scale.has_value()) {
    checkTensor(gemm1_weights_scale.value(), "gemm1_weights_scale",
                gemm1_weights_scale.value().ndim(), device);
  }
  if (gemm1_output_scale.has_value()) {
    checkTensor(gemm1_output_scale.value(), "gemm1_output_scale", 1, device);
  }
  TVM_FFI_ICHECK_EQ(num_non_exiting_ctas.size(0), 1)
      << "cake_stepfun_fc1: num_non_exiting_ctas must hold one value.";
  TVM_FFI_ICHECK_EQ(total_num_padded_tokens.size(0), 1)
      << "cake_stepfun_fc1: total_num_padded_tokens must hold one value.";

  int64_t const num_tokens = hidden_states.size(0);
  int64_t const hidden_size = hidden_states.size(1) * spec.actPerByte;
  int64_t const num_experts = gemm1_weights.size(0);
  int64_t const intermediate_size = gemm1_output.size(1) * spec.actPerByte;
  TVM_FFI_ICHECK_EQ(gemm1_clamp_limit.size(0), num_experts)
      << "cake_stepfun_fc1: gemm1_clamp_limit must hold one value per expert.";
  float* scale_c = optionalFloatPtr(output1_scale_scalar, "output1_scale_scalar", num_experts, device);
  float* scale_gate =
      optionalFloatPtr(output1_scale_gate_scalar, "output1_scale_gate_scalar", num_experts, device);
  float* token_scales = nullptr;
  if (per_token_scale.has_value()) {
    checkTensor(per_token_scale.value(), "per_token_scale", 1, device);
    checkDtype(per_token_scale.value(), "per_token_scale", dl_float32);
    TVM_FFI_ICHECK_GE(per_token_scale.value().size(0), num_tokens)
        << "cake_stepfun_fc1: per_token_scale must hold one value per token.";
    token_scales = static_cast<float*>(per_token_scale.value().data_ptr());
  }
  int64_t const max_padded_tokens = tgm::Routing::getMaxPermutedPaddedCount(
      static_cast<int32_t>(num_tokens), static_cast<int32_t>(top_k),
      static_cast<int32_t>(num_experts), static_cast<int32_t>(tile_tokens_dim));
  TVM_FFI_ICHECK_EQ(gemm1_output.size(0), max_padded_tokens)
      << "cake_stepfun_fc1: gemm1_output rows must be the maximum padded token count "
      << max_padded_tokens << ".";
  int64_t const max_ctas = tgm::Routing::getMaxNumCtasInBatchDim(
      static_cast<int32_t>(num_tokens), static_cast<int32_t>(top_k),
      static_cast<int32_t>(num_experts), static_cast<int32_t>(tile_tokens_dim));
  TVM_FFI_ICHECK_GE(cta_idx_xy_to_batch_idx.size(0), max_ctas)
      << "cake_stepfun_fc1: cta_idx_xy_to_batch_idx is shorter than the CTA grid.";
  TVM_FFI_ICHECK_GE(cta_idx_xy_to_mn_limit.size(0), max_ctas)
      << "cake_stepfun_fc1: cta_idx_xy_to_mn_limit is shorter than the CTA grid.";
  TVM_FFI_ICHECK_GE(permuted_idx_to_token_idx.size(0), max_padded_tokens)
      << "cake_stepfun_fc1: permuted_idx_to_token_idx is shorter than the padded count.";

  tgm::cake_stepfun::Fc1Runner runner(
      spec.dtypeAct, spec.dtypeWeights, spec.dtypeOutput, /*useDeepSeekFp8=*/false,
      static_cast<int>(tile_tokens_dim), tgm::MoE::ActivationType::SwigluStep,
      /*useShuffledMatrix=*/true, spec.weightLayout, batchedGemm::gemm::BiasType::None,
      /*usePerTokenScaling=*/spec.perToken, /*usePerChannelScaling=*/false);
  TVM_FFI_ICHECK(runner.usesCakeKernels())
      << "cake_stepfun_fc1: no exported Cake " << spec.name << " kernel serves tile_tokens_dim "
      << tile_tokens_dim << ".";
  int32_t const config_index = runner.getDefaultValidConfigIndex(
      static_cast<int32_t>(top_k), static_cast<int32_t>(hidden_size),
      static_cast<int32_t>(intermediate_size), static_cast<int32_t>(num_experts),
      static_cast<int32_t>(num_tokens));
  TVM_FFI_ICHECK_GE(config_index, 0)
      << "cake_stepfun_fc1: no exported Cake kernel accepts hidden_size " << hidden_size
      << " and intermediate_size " << intermediate_size << ".";

  cudaStream_t const stream = get_stream(device);
  runner.run(hidden_states.data_ptr(), optionalPtr(hidden_states_scale), gemm1_weights.data_ptr(),
             optionalPtr(gemm1_weights_scale), /*perTokenScales=*/token_scales,
             /*perChannelScales=*/nullptr, scale_c, scale_gate, /*ptrBias=*/nullptr,
             /*ptrGatedActAlpha=*/nullptr, /*ptrGatedActBeta=*/nullptr,
             static_cast<float*>(gemm1_clamp_limit.data_ptr()),
             /*permutedIdxToBiasRowIdx=*/nullptr, gemm1_output.data_ptr(),
             optionalPtr(gemm1_output_scale), static_cast<int32_t>(top_k),
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

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc1_families, cake_stepfun_fc1_families);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc1_tiles, cake_stepfun_fc1_tiles);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc1, cake_stepfun_fc1);
