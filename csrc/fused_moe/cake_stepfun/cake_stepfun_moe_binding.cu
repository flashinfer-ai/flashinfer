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

// Standalone entry points of the Cake StepFun stages. The fused-MoE operations of this
// module run the same kernels through MoE::Runner and the launchers; these operations
// expose the FC1 stage (every module build) and, on the full Cake path
// (-DCAKE_STEPFUN_FULL), the FC2, requantization and finalize stages alone over
// trtllm-gen routing metadata so the exported kernels can be validated and timed against
// their reference implementations on identical operands. Routing alone is reachable
// through the module's trtllm_moe_run_routing* operations, which the full path serves
// with the Cake router.

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
#ifdef CAKE_STEPFUN_FULL
#include "tensorrt_llm/kernels/nvfp4Recipe.h"
#endif

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
  TVM_FFI_ICHECK_EQ(tensor.ndim(), ndim)
      << "cake_stepfun_fc1: " << name << " must be " << ndim << "-D.";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << "cake_stepfun_fc1: " << name << " must be contiguous.";
  TVM_FFI_ICHECK(tensor.device().device_type == kDLCUDA &&
                 tensor.device().device_id == device.device_id)
      << "cake_stepfun_fc1: " << name << " must live on the launch device.";
}

void checkDtype(TensorView const& tensor, char const* name, DLDataType dtype) {
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dtype)
      << "cake_stepfun_fc1: " << name << " has an unexpected dtype.";
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

/** Index of the trtllm-gen batched-GEMM configuration named ``function_name`` in the metainfo
 * table this module was built with -- the FC1 / FC2 coordinate space of the native runners'
 * ``trtllm_get_valid_moe_factorizations`` -- or -1 when the artifact has no such kernel. The
 * lookup lives in the batched-GEMM runner translation unit, the one TU of the module that
 * instantiates the metainfo table. */
int64_t cake_stepfun_native_bmm_config_index(String const& function_name) {
  std::string const name(function_name.data(), function_name.size());
  return tensorrt_llm::kernels::getBatchedGemmConfigIndexByName(name);
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
void cake_stepfun_fc1(
    String const& family, TensorView const& hidden_states,
    Optional<TensorView> const& hidden_states_scale, TensorView const& gemm1_weights,
    Optional<TensorView> const& gemm1_weights_scale,
    Optional<TensorView> const& output1_scale_scalar,
    Optional<TensorView> const& output1_scale_gate_scalar, TensorView const& gemm1_clamp_limit,
    Optional<TensorView> const& per_token_scale, TensorView const& permuted_idx_to_token_idx,
    TensorView const& cta_idx_xy_to_batch_idx, TensorView const& cta_idx_xy_to_mn_limit,
    TensorView const& num_non_exiting_ctas, TensorView const& total_num_padded_tokens,
    TensorView const& gemm1_output, Optional<TensorView> const& gemm1_output_scale, int64_t top_k,
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
  for (TensorView const* array :
       {&permuted_idx_to_token_idx, &cta_idx_xy_to_batch_idx, &cta_idx_xy_to_mn_limit,
        &num_non_exiting_ctas, &total_num_padded_tokens}) {
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
  // The GEMM2 input has the family's activation storage ([max_padded, I / actPerByte] bytes),
  // except for the per-token family whose FC1 output is bf16 [max_padded, I].
  int64_t const intermediate_size =
      spec.perToken ? gemm1_output.size(1) : gemm1_output.size(1) * spec.actPerByte;
  TVM_FFI_ICHECK_EQ(gemm1_clamp_limit.size(0), num_experts)
      << "cake_stepfun_fc1: gemm1_clamp_limit must hold one value per expert.";
  float* scale_c =
      optionalFloatPtr(output1_scale_scalar, "output1_scale_scalar", num_experts, device);
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
  TVM_FFI_ICHECK(runner.hasKernels())
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
             /*useRoutingScalesOnInput=*/false, device.device_id, stream, config_index, enable_pdl);
}

/** True when this module build runs the full Cake path (routing, FC2, requantization, finalize). */
bool cake_stepfun_full_path() {
#ifdef CAKE_STEPFUN_FULL
  return true;
#else
  return false;
#endif
}

/** Pipeline stages this module build serves with exported Cake kernels, in pipeline order. */
Array<String> cake_stepfun_stages() {
  Array<String> stages;
#ifdef CAKE_STEPFUN_FULL
  stages.push_back(String("routing"));
#endif
  stages.push_back(String("fc1"));
#ifdef CAKE_STEPFUN_FULL
  stages.push_back(String("requant"));
  stages.push_back(String("fc2"));
  stages.push_back(String("finalize"));
#endif
  return stages;
}

#ifdef CAKE_STEPFUN_FULL
namespace {

generated::SfLayout sfLayoutFromName(String const& layout) {
  std::string const name(layout.data(), layout.size());
  if (name == "linear") return generated::SfLayout::kLinear;
  if (name == "r8c4") return generated::SfLayout::kR8c4;
  if (name == "r128c4") return generated::SfLayout::kR128c4;
  TVM_FFI_ICHECK(false) << "cake_stepfun: unknown block-scale layout '" << name
                        << "' (linear, r8c4, r128c4).";
  return generated::SfLayout::kNone;
}

char const* sfLayoutName(generated::SfLayout layout) {
  switch (layout) {
    case generated::SfLayout::kLinear:
      return "linear";
    case generated::SfLayout::kR8c4:
      return "r8c4";
    case generated::SfLayout::kR128c4:
      return "r128c4";
    default:
      return "none";
  }
}

}  // namespace

/** Tile sizes (tokens per CTA) served by the exported Cake StepFun FC2 kernels of ``family``. */
Array<int64_t> cake_stepfun_fc2_tiles(String const& family) {
  FamilySpec const& spec = familySpec(family);
  Array<int64_t> tiles;
  for (size_t index = 0; index < generated::kFc2KernelCount; ++index) {
    if (generated::kFc2Kernels[index].family == spec.index) {
      tiles.push_back(generated::kFc2Kernels[index].tile_n);
    }
  }
  return tiles;
}

/** Block-scale layout the Cake FC2 kernel of ``family`` / ``tile_tokens_dim`` reads for its
 * activations. */
String cake_stepfun_fc2_activation_sf_layout(String const& family, int64_t tile_tokens_dim) {
  FamilySpec const& spec = familySpec(family);
  for (size_t index = 0; index < generated::kFc2KernelCount; ++index) {
    auto const& kernel = generated::kFc2Kernels[index];
    if (kernel.family == spec.index && kernel.tile_n == tile_tokens_dim) {
      return String(sfLayoutName(kernel.sf_layout_a));
    }
  }
  TVM_FFI_ICHECK(false) << "cake_stepfun_fc2: no exported Cake " << spec.name
                        << " FC2 kernel serves tile_tokens_dim " << tile_tokens_dim << ".";
  return String("none");
}

/** Device symbol of the Cake StepFun FC2 kernel at ``config_index`` of the generated FC2 table
 * (the FC2 coordinate of a full-path tactic in ``trtllm_get_valid_moe_factorizations``). */
String cake_stepfun_fc2_kernel_symbol(int64_t config_index) {
  TVM_FFI_ICHECK(config_index >= 0 &&
                 config_index < static_cast<int64_t>(generated::kFc2KernelCount))
      << "cake_stepfun_fc2: FC2 config index " << config_index
      << " is outside the generated table [0, " << generated::kFc2KernelCount << ").";
  return String(generated::kFc2Kernels[config_index].symbol);
}

/**
 * Run the Cake StepFun FC2 stage of ``family`` over trtllm-gen routing metadata.
 *
 * ``gemm2_input`` is the FC1 output of the family in permuted order ([max_padded_tokens,
 * intermediate storage]; the requantized E2m1 rows for ``nvfp4_bf16tok``) with its optional
 * block scales, ``gemm2_weights`` the trtllm-prepared weights with optional block scales,
 * ``output2_scale_scalar`` the optional per-expert FP32 output scales and ``per_token_scale``
 * the fp32 per-token scales of the ``nvfp4_bf16tok`` family. ``gemm2_output`` receives bf16
 * rows in permuted order ([max_padded_tokens, hidden_size]).
 */
void cake_stepfun_fc2(String const& family, TensorView const& gemm2_input,
                      Optional<TensorView> const& gemm2_input_scale,
                      TensorView const& gemm2_weights,
                      Optional<TensorView> const& gemm2_weights_scale,
                      Optional<TensorView> const& output2_scale_scalar,
                      Optional<TensorView> const& per_token_scale,
                      TensorView const& cta_idx_xy_to_batch_idx,
                      TensorView const& cta_idx_xy_to_mn_limit,
                      TensorView const& num_non_exiting_ctas,
                      TensorView const& total_num_padded_tokens, TensorView const& gemm2_output,
                      int64_t num_tokens, int64_t top_k, int64_t tile_tokens_dim, bool enable_pdl) {
  FamilySpec const& spec = familySpec(family);
  DLDevice const device = gemm2_input.device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA)
      << "cake_stepfun_fc2: gemm2_input must be a CUDA tensor.";
  checkTensor(gemm2_input, "gemm2_input", 2, device);
  checkTensor(gemm2_weights, "gemm2_weights", gemm2_weights.ndim(), device);
  checkTensor(cta_idx_xy_to_batch_idx, "cta_idx_xy_to_batch_idx", 1, device);
  checkTensor(cta_idx_xy_to_mn_limit, "cta_idx_xy_to_mn_limit", 1, device);
  checkTensor(num_non_exiting_ctas, "num_non_exiting_ctas", 1, device);
  checkTensor(total_num_padded_tokens, "total_num_padded_tokens", 1, device);
  checkTensor(gemm2_output, "gemm2_output", 2, device);
  checkDtype(gemm2_output, "gemm2_output", dl_bfloat16);
  for (TensorView const* array : {&cta_idx_xy_to_batch_idx, &cta_idx_xy_to_mn_limit,
                                  &num_non_exiting_ctas, &total_num_padded_tokens}) {
    checkDtype(*array, "routing arrays", dl_int32);
  }
  if (gemm2_input_scale.has_value()) {
    checkTensor(gemm2_input_scale.value(), "gemm2_input_scale", gemm2_input_scale.value().ndim(),
                device);
  }
  if (gemm2_weights_scale.has_value()) {
    checkTensor(gemm2_weights_scale.value(), "gemm2_weights_scale",
                gemm2_weights_scale.value().ndim(), device);
  }
  int64_t const num_experts = gemm2_weights.size(0);
  int64_t const hidden_size = gemm2_output.size(1);
  int64_t const intermediate_size = gemm2_input.size(1) * spec.actPerByte;
  int64_t const max_padded_tokens = tgm::Routing::getMaxPermutedPaddedCount(
      static_cast<int32_t>(num_tokens), static_cast<int32_t>(top_k),
      static_cast<int32_t>(num_experts), static_cast<int32_t>(tile_tokens_dim));
  TVM_FFI_ICHECK_EQ(gemm2_input.size(0), max_padded_tokens)
      << "cake_stepfun_fc2: gemm2_input rows must be the maximum padded token count "
      << max_padded_tokens << ".";
  TVM_FFI_ICHECK_EQ(gemm2_output.size(0), max_padded_tokens)
      << "cake_stepfun_fc2: gemm2_output rows must be the maximum padded token count "
      << max_padded_tokens << ".";
  int64_t const max_ctas = tgm::Routing::getMaxNumCtasInBatchDim(
      static_cast<int32_t>(num_tokens), static_cast<int32_t>(top_k),
      static_cast<int32_t>(num_experts), static_cast<int32_t>(tile_tokens_dim));
  TVM_FFI_ICHECK_GE(cta_idx_xy_to_batch_idx.size(0), max_ctas)
      << "cake_stepfun_fc2: cta_idx_xy_to_batch_idx is shorter than the CTA grid.";
  TVM_FFI_ICHECK_GE(cta_idx_xy_to_mn_limit.size(0), max_ctas)
      << "cake_stepfun_fc2: cta_idx_xy_to_mn_limit is shorter than the CTA grid.";
  float* scale_c =
      optionalFloatPtr(output2_scale_scalar, "output2_scale_scalar", num_experts, device);
  float* token_scales = nullptr;
  if (per_token_scale.has_value()) {
    checkTensor(per_token_scale.value(), "per_token_scale", 1, device);
    checkDtype(per_token_scale.value(), "per_token_scale", dl_float32);
    TVM_FFI_ICHECK_GE(per_token_scale.value().size(0), max_padded_tokens)
        << "cake_stepfun_fc2: per_token_scale must hold one value per padded token.";
    token_scales = static_cast<float*>(per_token_scale.value().data_ptr());
  }

  tgm::cake_stepfun::Fc2Runner runner(spec.dtypeAct, spec.dtypeWeights, btg::Dtype::Bfloat16,
                                      /*useDeepSeekFp8=*/false, static_cast<int>(tile_tokens_dim),
                                      /*useShuffledMatrix=*/true, spec.weightLayout,
                                      /*usePerTokenScaling=*/spec.perToken,
                                      /*usePerChannelScaling=*/false);
  TVM_FFI_ICHECK(runner.hasKernels())
      << "cake_stepfun_fc2: no exported Cake " << spec.name << " FC2 kernel serves tile_tokens_dim "
      << tile_tokens_dim << ".";
  int32_t const config_index = runner.getDefaultValidConfigIndex(
      static_cast<int32_t>(top_k), static_cast<int32_t>(hidden_size),
      static_cast<int32_t>(intermediate_size), static_cast<int32_t>(num_experts),
      static_cast<int32_t>(num_tokens));
  cudaStream_t const stream = get_stream(device);
  runner.run(gemm2_input.data_ptr(), optionalPtr(gemm2_input_scale), gemm2_weights.data_ptr(),
             optionalPtr(gemm2_weights_scale), /*perTokenScales=*/token_scales,
             /*perChannelScales=*/nullptr, scale_c, /*ptrBias=*/nullptr, gemm2_output.data_ptr(),
             /*outputScale=*/nullptr, static_cast<int32_t>(top_k),
             static_cast<int32_t>(hidden_size), static_cast<int32_t>(intermediate_size),
             static_cast<int32_t>(num_experts), static_cast<int32_t>(num_tokens),
             static_cast<int32_t*>(num_non_exiting_ctas.data_ptr()),
             static_cast<int32_t*>(total_num_padded_tokens.data_ptr()),
             static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr()),
             static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr()),
             /*bmm2Workspace=*/nullptr, device.device_id, stream, config_index, enable_pdl);
}

/**
 * Requantize the bf16 FC1 output of the ``nvfp4_bf16tok`` family into the FC2 input.
 *
 * ``gemm1_output`` is bf16 [max_padded_tokens, intermediate_size]; ``output`` receives packed
 * E2m1 [max_padded_tokens, intermediate_size / 2], ``output_scale`` the E4m3 block scales in
 * ``sf_layout`` (``linear``, ``r8c4`` or ``r128c4``; what the consuming FC2 kernel reads, see
 * cake_stepfun_fc2_activation_sf_layout) and ``per_token_scale`` the fp32 per-token scales.
 * The NVFP4 recipe is resolved exactly as the fused-MoE forward resolves it.
 */
void cake_stepfun_requant(TensorView const& gemm1_output,
                          TensorView const& expanded_idx_to_permuted_idx, TensorView const& output,
                          TensorView const& output_scale, TensorView const& per_token_scale,
                          String const& sf_layout, bool enable_pdl) {
  DLDevice const device = gemm1_output.device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA)
      << "cake_stepfun_requant: gemm1_output must be a CUDA tensor.";
  checkTensor(gemm1_output, "gemm1_output", 2, device);
  checkDtype(gemm1_output, "gemm1_output", dl_bfloat16);
  checkTensor(expanded_idx_to_permuted_idx, "expanded_idx_to_permuted_idx", 1, device);
  checkDtype(expanded_idx_to_permuted_idx, "expanded_idx_to_permuted_idx", dl_int32);
  checkTensor(output, "output", 2, device);
  checkDtype(output, "output", dl_uint8);
  checkTensor(output_scale, "output_scale", output_scale.ndim(), device);
  checkDtype(output_scale, "output_scale", dl_uint8);
  checkTensor(per_token_scale, "per_token_scale", 1, device);
  checkDtype(per_token_scale, "per_token_scale", dl_float32);
  int64_t const max_padded_tokens = gemm1_output.size(0);
  int64_t const intermediate_size = gemm1_output.size(1);
  TVM_FFI_ICHECK_EQ(output.size(0), max_padded_tokens)
      << "cake_stepfun_requant: output rows must match gemm1_output.";
  TVM_FFI_ICHECK_EQ(output.size(1) * 2, intermediate_size)
      << "cake_stepfun_requant: output must hold intermediate_size / 2 packed bytes per row.";
  TVM_FFI_ICHECK_GE(per_token_scale.size(0), max_padded_tokens)
      << "cake_stepfun_requant: per_token_scale must hold one value per padded token.";
  auto const recipe =
      tensorrt_llm::kernels::resolveNVFP4Recipe(tensorrt_llm::kernels::kNVFP44Over6FromEnv);
  tgm::cake_stepfun::requant::run(
      static_cast<int32_t>(expanded_idx_to_permuted_idx.size(0)),
      static_cast<int32_t>(intermediate_size),
      static_cast<__nv_bfloat16 const*>(gemm1_output.data_ptr()), recipe.globalScaleInv(),
      static_cast<float>(recipe.e4m3Max),
      static_cast<int32_t const*>(expanded_idx_to_permuted_idx.data_ptr()),
      static_cast<uint8_t*>(output.data_ptr()), static_cast<uint8_t*>(output_scale.data_ptr()),
      static_cast<float*>(per_token_scale.data_ptr()), sfLayoutFromName(sf_layout),
      get_stream(device), enable_pdl);
}

/** Routing input kinds (``scores``, ``topk_ids``) with an exported Cake routing kernel. */
Array<String> cake_stepfun_routing_inputs() {
  Array<String> inputs;
  for (auto input : {generated::RoutingInput::kScores, generated::RoutingInput::kTopKIds}) {
    for (size_t index = 0; index < generated::kRoutingKernelCount; ++index) {
      if (generated::kRoutingKernels[index].input == input) {
        inputs.push_back(String(input == generated::RoutingInput::kScores ? "scores" : "topk_ids"));
        break;
      }
    }
  }
  return inputs;
}

/**
 * Run the exported Cake routing on caller-owned tables: the launch ``RoutingRunner`` performs for
 * the fused-MoE forward, without the forward's allocations. Renormalize top-k over
 * ``routing_logits`` ([num_tokens, num_experts] bfloat16 or float32; ``topk_packed`` and bf16
 * ``topk_weights`` [num_tokens, top_k] are written) or the permutation tables of pre-computed
 * ``topk_ids`` ([num_tokens, top_k] int32; ``topk_weights`` is the caller's input and
 * ``topk_packed`` is not written). The other tensors are the fused forward's buffers of the same
 * names: ``expert_count_histogram`` (>= 2 * num_experts int32), ``total_num_padded_tokens`` [1],
 * ``expanded_idx_to_permuted_idx`` [num_tokens * top_k], ``permuted_idx_to_token_idx``
 * (>= Routing::getMaxPermutedPaddedCount), ``cta_idx_xy_to_batch_idx`` / ``cta_idx_xy_to_mn_limit``
 * (>= Routing::getMaxNumCtasInBatchDim), ``num_non_exiting_ctas`` [1] and the optional
 * ``num_tokens_per_expert`` [num_experts].
 */
void cake_stepfun_routing(
    Optional<TensorView> const& routing_logits, Optional<TensorView> const& topk_ids,
    TensorView const& topk_packed, TensorView const& topk_weights,
    TensorView const& expert_count_histogram, TensorView const& total_num_padded_tokens,
    TensorView const& expanded_idx_to_permuted_idx, TensorView const& permuted_idx_to_token_idx,
    TensorView const& cta_idx_xy_to_batch_idx, TensorView const& cta_idx_xy_to_mn_limit,
    TensorView const& num_non_exiting_ctas, Optional<TensorView> const& num_tokens_per_expert,
    int64_t num_experts, int64_t top_k, int64_t local_expert_offset, int64_t local_num_experts,
    int64_t tile_tokens_dim, bool enable_pdl) {
  TVM_FFI_ICHECK(routing_logits.has_value() != topk_ids.has_value())
      << "cake_stepfun_routing: pass exactly one of routing_logits (scores path) or topk_ids "
         "(pre-computed path).";
  DLDevice const device = topk_packed.device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA)
      << "cake_stepfun_routing: topk_packed must be a CUDA tensor.";
  int64_t num_tokens = 0;
  btg::Dtype dtype_logits = btg::Dtype::Bfloat16;
  if (routing_logits.has_value()) {
    TensorView const& logits = routing_logits.value();
    checkTensor(logits, "routing_logits", 2, device);
    TVM_FFI_ICHECK(logits.dtype() == dl_bfloat16 || logits.dtype() == dl_float32)
        << "cake_stepfun_routing: routing_logits must be bfloat16 or float32.";
    TVM_FFI_ICHECK_EQ(logits.size(1), num_experts)
        << "cake_stepfun_routing: routing_logits columns must equal num_experts.";
    num_tokens = logits.size(0);
    dtype_logits = logits.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  } else {
    TensorView const& ids = topk_ids.value();
    checkTensor(ids, "topk_ids", 2, device);
    checkDtype(ids, "topk_ids", dl_int32);
    TVM_FFI_ICHECK_EQ(ids.size(1), top_k)
        << "cake_stepfun_routing: topk_ids columns must equal top_k.";
    num_tokens = ids.size(0);
  }
  TVM_FFI_ICHECK(num_tokens > 0) << "cake_stepfun_routing: num_tokens must be positive.";
  TVM_FFI_ICHECK(top_k > 0 && top_k <= num_experts)
      << "cake_stepfun_routing: top_k must be between one and num_experts.";
  TVM_FFI_ICHECK(local_num_experts > 0 && local_expert_offset >= 0 &&
                 local_expert_offset + local_num_experts <= num_experts)
      << "cake_stepfun_routing: the local expert range must lie within num_experts.";
  TVM_FFI_ICHECK_GT(tile_tokens_dim, 0)
      << "cake_stepfun_routing: tile_tokens_dim must be positive.";
  checkTensor(topk_packed, "topk_packed", 2, device);
  checkDtype(topk_packed, "topk_packed", dl_int32);
  TVM_FFI_ICHECK(topk_packed.size(0) == num_tokens && topk_packed.size(1) == top_k)
      << "cake_stepfun_routing: topk_packed must be [num_tokens, top_k].";
  checkTensor(topk_weights, "topk_weights", 2, device);
  checkDtype(topk_weights, "topk_weights", dl_bfloat16);
  TVM_FFI_ICHECK(topk_weights.size(0) == num_tokens && topk_weights.size(1) == top_k)
      << "cake_stepfun_routing: topk_weights must be [num_tokens, top_k].";
  int64_t const max_padded_tokens = tgm::Routing::getMaxPermutedPaddedCount(
      static_cast<int32_t>(num_tokens), static_cast<int32_t>(top_k),
      static_cast<int32_t>(num_experts), static_cast<int32_t>(tile_tokens_dim));
  int64_t const max_num_ctas = tgm::Routing::getMaxNumCtasInBatchDim(
      static_cast<int32_t>(num_tokens), static_cast<int32_t>(top_k),
      static_cast<int32_t>(num_experts), static_cast<int32_t>(tile_tokens_dim));
  struct Table {
    TensorView const* tensor;
    char const* name;
    int64_t min_size;
  };
  Table const tables[] = {
      {&expert_count_histogram, "expert_count_histogram", 2 * num_experts},
      {&total_num_padded_tokens, "total_num_padded_tokens", 1},
      {&expanded_idx_to_permuted_idx, "expanded_idx_to_permuted_idx", num_tokens * top_k},
      {&permuted_idx_to_token_idx, "permuted_idx_to_token_idx", max_padded_tokens},
      {&cta_idx_xy_to_batch_idx, "cta_idx_xy_to_batch_idx", max_num_ctas},
      {&cta_idx_xy_to_mn_limit, "cta_idx_xy_to_mn_limit", max_num_ctas},
      {&num_non_exiting_ctas, "num_non_exiting_ctas", 1},
  };
  for (Table const& table : tables) {
    TVM_FFI_ICHECK(table.tensor->IsContiguous() && table.tensor->device().device_type == kDLCUDA &&
                   table.tensor->device().device_id == device.device_id)
        << "cake_stepfun_routing: " << table.name
        << " must be a contiguous tensor on the launch device.";
    checkDtype(*table.tensor, table.name, dl_int32);
    TVM_FFI_ICHECK_GE(table.tensor->numel(), table.min_size)
        << "cake_stepfun_routing: " << table.name << " must hold at least " << table.min_size
        << " entries.";
  }
  int32_t* num_tokens_per_expert_ptr = nullptr;
  if (num_tokens_per_expert.has_value()) {
    checkTensor(num_tokens_per_expert.value(), "num_tokens_per_expert", 1, device);
    checkDtype(num_tokens_per_expert.value(), "num_tokens_per_expert", dl_int32);
    TVM_FFI_ICHECK_GE(num_tokens_per_expert.value().size(0), num_experts)
        << "cake_stepfun_routing: num_tokens_per_expert must hold one count per expert.";
    num_tokens_per_expert_ptr = static_cast<int32_t*>(num_tokens_per_expert.value().data_ptr());
  }
  tgm::cake_stepfun::RoutingRunner runner(static_cast<int32_t>(tile_tokens_dim));
  runner.run(routing_logits.has_value() ? routing_logits.value().data_ptr() : nullptr,
             /*routingBias=*/nullptr, static_cast<int32_t>(num_tokens),
             static_cast<int32_t>(num_experts), static_cast<int32_t>(top_k),
             /*numFusedSharedExpert=*/0, /*nGroup=*/0, /*topkGroup=*/0,
             static_cast<int32_t>(local_expert_offset), static_cast<int32_t>(local_num_experts),
             /*routedScalingFactor=*/1.0f, static_cast<int32_t*>(topk_packed.data_ptr()),
             static_cast<int32_t*>(expert_count_histogram.data_ptr()),
             static_cast<int32_t*>(total_num_padded_tokens.data_ptr()),
             static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr()),
             /*permutedIdxToExpandedIdx=*/nullptr,
             static_cast<int32_t*>(permuted_idx_to_token_idx.data_ptr()),
             topk_ids.has_value() ? static_cast<int32_t*>(topk_ids.value().data_ptr()) : nullptr,
             topk_weights.data_ptr(), num_tokens_per_expert_ptr,
             static_cast<int32_t*>(cta_idx_xy_to_batch_idx.data_ptr()),
             static_cast<int32_t*>(cta_idx_xy_to_mn_limit.data_ptr()),
             static_cast<int32_t*>(num_non_exiting_ctas.data_ptr()), btg::Dtype::Bfloat16,
             btg::Dtype::Bfloat16, /*useRoutingScalesOnInput=*/false, /*useDeepSeekFp8=*/false,
             tgm::Routing::RoutingMethodType::Renormalize, get_stream(device), dtype_logits,
             /*normTopkProb=*/true, /*routing_replay_out=*/nullptr, enable_pdl);
}

/** Expert-weight dtypes (``float32``, ``bfloat16``) with an exported Cake finalize kernel. */
Array<String> cake_stepfun_finalize_weight_dtypes() {
  Array<String> dtypes;
  for (int dtype : {0, 1}) {
    for (size_t index = 0; index < generated::kFinalizeKernelCount; ++index) {
      if (generated::kFinalizeKernels[index].expert_weights_dtype == dtype) {
        dtypes.push_back(String(dtype == 0 ? "float32" : "bfloat16"));
        break;
      }
    }
  }
  return dtypes;
}

/**
 * Run the Cake finalize stage: unpermute the bf16 FC2 output and reduce the top-k experts of
 * every token with ``expert_weights`` (bf16 or fp32 [num_tokens, top_k]) into ``output``
 * (bf16 [num_tokens, hidden_size]).
 */
void cake_stepfun_finalize(TensorView const& gemm2_output, TensorView const& expert_weights,
                           TensorView const& expanded_idx_to_permuted_idx,
                           TensorView const& total_num_padded_tokens, TensorView const& output,
                           int64_t num_experts, bool enable_pdl) {
  DLDevice const device = gemm2_output.device();
  TVM_FFI_ICHECK(device.device_type == kDLCUDA)
      << "cake_stepfun_finalize: gemm2_output must be a CUDA tensor.";
  checkTensor(gemm2_output, "gemm2_output", 2, device);
  checkDtype(gemm2_output, "gemm2_output", dl_bfloat16);
  checkTensor(expert_weights, "expert_weights", 2, device);
  TVM_FFI_ICHECK(expert_weights.dtype() == dl_bfloat16 || expert_weights.dtype() == dl_float32)
      << "cake_stepfun_finalize: expert_weights must be bfloat16 or float32.";
  checkTensor(expanded_idx_to_permuted_idx, "expanded_idx_to_permuted_idx", 1, device);
  checkDtype(expanded_idx_to_permuted_idx, "expanded_idx_to_permuted_idx", dl_int32);
  checkTensor(total_num_padded_tokens, "total_num_padded_tokens", 1, device);
  checkDtype(total_num_padded_tokens, "total_num_padded_tokens", dl_int32);
  checkTensor(output, "output", 2, device);
  checkDtype(output, "output", dl_bfloat16);
  int64_t const num_tokens = output.size(0);
  int64_t const top_k = expert_weights.size(1);
  TVM_FFI_ICHECK_EQ(expert_weights.size(0), num_tokens)
      << "cake_stepfun_finalize: expert_weights rows must match output rows.";
  TVM_FFI_ICHECK_EQ(expanded_idx_to_permuted_idx.size(0), num_tokens * top_k)
      << "cake_stepfun_finalize: expanded_idx_to_permuted_idx must hold num_tokens * top_k "
         "entries.";
  TVM_FFI_ICHECK_LE(output.size(1), gemm2_output.size(1))
      << "cake_stepfun_finalize: output width exceeds the FC2 output row stride.";
  moe::dev::finalize::Data data{};
  data.mDtypeElt = btg::Dtype::Bfloat16;
  data.mDtypeExpW = expert_weights.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  data.mUsePdl = enable_pdl;
  data.mUseDeepSeekFp8 = false;
  data.inPtr = gemm2_output.data_ptr();
  data.outPtr = output.data_ptr();
  data.expertWeightsPtr = expert_weights.data_ptr();
  data.expandedIdxToPermutedIdx = static_cast<int32_t*>(expanded_idx_to_permuted_idx.data_ptr());
  data.numTokens = static_cast<int32_t>(num_tokens);
  data.numExperts = static_cast<int32_t>(num_experts);
  data.topK = static_cast<int32_t>(top_k);
  data.hiddenDim = static_cast<int32_t>(output.size(1));
  data.hiddenDimPadded = static_cast<int32_t>(gemm2_output.size(1));
  data.totalNumPaddedTokens = static_cast<int32_t const*>(total_num_padded_tokens.data_ptr());
  tgm::cake_stepfun::finalize::run(data, get_stream(device));
}
#endif  // CAKE_STEPFUN_FULL

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc1_families, cake_stepfun_fc1_families);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc1_tiles, cake_stepfun_fc1_tiles);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_native_bmm_config_index,
                              cake_stepfun_native_bmm_config_index);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc1, cake_stepfun_fc1);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_full_path, cake_stepfun_full_path);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_stages, cake_stepfun_stages);
#ifdef CAKE_STEPFUN_FULL
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_routing_inputs, cake_stepfun_routing_inputs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_routing, cake_stepfun_routing);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_finalize_weight_dtypes,
                              cake_stepfun_finalize_weight_dtypes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc2_tiles, cake_stepfun_fc2_tiles);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc2_kernel_symbol, cake_stepfun_fc2_kernel_symbol);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc2_activation_sf_layout,
                              cake_stepfun_fc2_activation_sf_layout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_fc2, cake_stepfun_fc2);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_requant, cake_stepfun_requant);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cake_stepfun_finalize, cake_stepfun_finalize);
#endif
