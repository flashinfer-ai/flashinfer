/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#if defined(USING_OSS_CUTLASS_MOE_GEMM)
#include "moe_kernels.h"
#else
#include "moe_gemm_kernels.h"
#include "moe_kernels.h"
#endif
// Always include the public header for moe_gemm_kernels.h

#include <tvm/ffi/extra/module.h>

#include "../../tvm_ffi_utils.h"
#include "cutlass_kernel_selector.h"
#include "moe_gemm_kernels.h"
#include "tensorrt_llm/common/workspace.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_gemm.h"

namespace common = tensorrt_llm::common;
namespace kernels = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE;
using ActivationParams = CUTLASS_MOE_GEMM_NAMESPACE::ActivationParams;
using ActivationType = CUTLASS_MOE_GEMM_NAMESPACE::ActivationType;
// Always use public header as it is just utility functions and types
using TmaWarpSpecializedGroupedGemmInput =
    tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;
using profiler_backend = CUTLASS_MOE_GEMM_KERNELS_NAMESPACE::GemmProfilerBackend;

using tvm::ffi::Array;
using tvm::ffi::DLDataTypeToString;
using tvm::ffi::Function;
using tvm::ffi::Optional;
constexpr DLDataType dl_uint4x2 = DLDataType{kDLUInt, 4, 2};

class DtypeUtils {
 public:
  static nvinfer1::DataType dataType(DLDataType dtype) {
    switch (encode_dlpack_dtype(dtype)) {
      case float32_code:
        return nvinfer1::DataType::kFLOAT;
      case float16_code:
        return nvinfer1::DataType::kHALF;
      case encode_dlpack_dtype(dl_int8):
        return nvinfer1::DataType::kINT8;
      case encode_dlpack_dtype(dl_uint8):
        return nvinfer1::DataType::kUINT8;
      case int32_code:
        return nvinfer1::DataType::kINT32;
      case int64_code:
        return nvinfer1::DataType::kINT64;
      case encode_dlpack_dtype(dl_bool):
        return nvinfer1::DataType::kBOOL;
      case float8_e4m3fn_code:
        return nvinfer1::DataType::kFP8;
      case bfloat16_code:
        return nvinfer1::DataType::kBF16;
      case encode_dlpack_dtype(dl_uint4x2):
        return nvinfer1::DataType::kINT4;
      default:
        TVM_FFI_ICHECK(false) << "unsupported data type";
    }
  }

 private:
  DtypeUtils() = default;
};

class FusedMoeRunner : public tvm::ffi::ModuleObj {
 public:
  template <typename TypeAct, typename TypeWeight, bool NeedQuant = false>
  std::unique_ptr<kernels::CutlassMoeFCRunnerInterface> switch_output_type(DLDataType output_type) {
    switch (encode_dlpack_dtype(output_type)) {
      case int64_code:  // INT64 == FP4
      case float8_e4m3fn_code:
        // TODO We need an atomic FP8 reduction for the finalize fusions
        TVM_FFI_LOG_AND_THROW(NotImplementedError)
            << "Outputting " << DLDataTypeToString(output_type)
            << " directly is not currently supported";
        // return std::make_unique<kernels::CutlassMoeFCRunner<Type, Type>>();
      case float16_code:
        if constexpr (NeedQuant) {
          return std::make_unique<kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, half>>();
        } else {
          return std::make_unique<
              kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, TypeAct>>();
        }
#ifdef ENABLE_BF16
      case bfloat16_code:
        if constexpr (NeedQuant) {
          return std::make_unique<
              kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, __nv_bfloat16>>();
        } else {
          return std::make_unique<
              kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, TypeAct>>();
        }
#endif
      default:
        TVM_FFI_ICHECK(false) << "Invalid output type " << DLDataTypeToString(output_type)
                              << " specified for " << DLDataTypeToString(mActivationDtype);
    }
  };

  FusedMoeRunner(DLDataType activation_dtype, DLDataType weight_dtype, DLDataType output_dtype,
                 bool use_deepseek_fp8_block_scale, bool use_w4_group_scaling,
                 bool use_mxfp8_act_scaling) {
    mActivationDtype = activation_dtype;
    mWeightDtype = weight_dtype;
    mOutputDtype = output_dtype;
    mUseDeepSeekFP8BlockScaling = use_deepseek_fp8_block_scale;
    mUseW4GroupScaling = use_w4_group_scaling;
    mUseMxfp8ActScaling = use_mxfp8_act_scaling;
    mInnerDimMultiplier = 1;

    // keep consistent with cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp
    if (mActivationDtype == dl_float16 && mWeightDtype == dl_float16) {
      mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, half>>();
    }
#ifdef ENABLE_BF16
    else if (mActivationDtype == dl_bfloat16 && mWeightDtype == dl_bfloat16) {
      mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>>();
    }
#ifdef ENABLE_FP8
    else if (mActivationDtype == dl_bfloat16 && mWeightDtype == dl_float8_e4m3fn) {
      mKernelRunner = std::make_unique<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3>>();
    }
#endif
#endif

#ifdef ENABLE_FP8
    if (isFp8Quant()) {
      mKernelRunner = switch_output_type<__nv_fp8_e4m3, __nv_fp8_e4m3>(mOutputDtype);
    }
#endif
#ifdef ENABLE_FP4
    if (isWMxfp4AMxfp8Quant() || isWMxfp4AFp8Quant()) {
      mInnerDimMultiplier = 16;  // 16 FP4 -> 1 LONG
      mKernelRunner = switch_output_type<__nv_fp8_e4m3, __nv_fp4_e2m1>(mOutputDtype);
    }

    if (isNvfp4Quant()) {
      mInnerDimMultiplier = 16;
      switch (encode_dlpack_dtype(mActivationDtype)) {
        case float16_code:
#ifdef ENABLE_BF16
        case bfloat16_code:
#endif
          mKernelRunner = switch_output_type<__nv_fp4_e2m1, __nv_fp4_e2m1, true>(mOutputDtype);
          break;
        default:
          mKernelRunner = switch_output_type<__nv_fp4_e2m1, __nv_fp4_e2m1, false>(mOutputDtype);
      }
    }

    if (isWFP4A16Quant()) {
      mInnerDimMultiplier = 2;
      if (mActivationDtype == dl_float16) {
        mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, __nv_fp4_e2m1>>();
      }
#ifdef ENABLE_BF16
      else if (mActivationDtype == dl_bfloat16) {
        mKernelRunner =
            std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, __nv_fp4_e2m1>>();
      }
#endif
    }

#endif
    if (isInt4Quant()) {
      mInnerDimMultiplier = 2;
      if (mActivationDtype == dl_float16) {
#ifdef ENABLE_FP8
        if (mUseW4GroupScaling) {
          mKernelRunner = std::make_unique<
              kernels::CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t, half, half>>();
        } else {
          mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, cutlass::uint4b_t>>();
        }
#else
        mKernelRunner = std::make_shared<kernels::CutlassMoeFCRunner<half, cutlass::uint4b_t>>();
#endif
      }
#ifdef ENABLE_BF16
      else if (mActivationDtype == dl_bfloat16) {
#ifdef ENABLE_FP8
        if (mUseW4GroupScaling) {
          mKernelRunner =
              std::make_unique<kernels::CutlassMoeFCRunner<__nv_fp8_e4m3, cutlass::uint4b_t,
                                                           __nv_bfloat16, __nv_bfloat16>>();
        } else {
          mKernelRunner =
              std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>>();
        }
#else
        mKernelRunner =
            std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>>();
#endif
      }
#endif
    }
    if (!mKernelRunner) {
      TVM_FFI_ICHECK(false)
          << "Could not construct fused moe op with the requested input combination Activation: "
          << DLDataTypeToString(mActivationDtype)
          << ", Weight: " << DLDataTypeToString(mWeightDtype)
          << ", Output: " << DLDataTypeToString(mOutputDtype);
    }

    mProfiler = std::make_shared<kernels::GemmProfilerBackend>();
    mAllProfiles = mKernelRunner->getTactics();
    TVM_FFI_ICHECK(!mAllProfiles.empty())
        << "No valid tactics available for fused moe op with the requested input combination "
           "Activation: "
        << DLDataTypeToString(mActivationDtype) << ", Weight: " << DLDataTypeToString(mWeightDtype)
        << ", Output: " << DLDataTypeToString(mOutputDtype);
  }

  void runMoe(Tensor output, Tensor input, Tensor token_selected_experts,
              Optional<Tensor> token_final_scales, Tensor fc1_expert_weights,
              Optional<Tensor> fc1_expert_biases, Tensor fc2_expert_weights,
              Optional<Tensor> fc2_expert_biases, Optional<Array<Tensor>> quant_scales,
              Optional<Tensor> input_sf, Optional<Tensor> swiglu_alpha,
              Optional<Tensor> swiglu_beta, Optional<Tensor> swiglu_limit, int64_t tp_size,
              int64_t tp_rank, int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
              int64_t cluster_rank, bool enable_alltoall, bool min_latency_mode,
              Optional<Array<int64_t>> profile_ids, bool enable_pdl) {
    std::lock_guard<std::mutex> lock(mMutex);

    TVM_FFI_ICHECK(cluster_size == 1 && cluster_rank == 0)
        << "smart_router is supported in min_latency mode";

    CHECK_INPUT_TYPE(input, mActivationDtype)
    CHECK_INPUT_TYPE(token_selected_experts, dl_int32)
    if (token_final_scales) {
      CHECK_INPUT_TYPE(token_final_scales.value(), dl_float32)
    }
    if (mWeightDtype == dl_uint4x2) {
      // Since dlpack does not support uint4x2, here uses uint8 to bypass
      CHECK_INPUT_TYPE(fc1_expert_weights, dl_uint8);
      CHECK_INPUT_TYPE(fc2_expert_weights, dl_uint8)
    } else {
      CHECK_INPUT_TYPE(fc1_expert_weights, mWeightDtype);
      CHECK_INPUT_TYPE(fc2_expert_weights, mWeightDtype)
    }

    CHECK_DIM(2, input);
    CHECK_DIM(2, token_selected_experts);

    CHECK_DIM(3, fc1_expert_weights);
    CHECK_DIM(3, fc2_expert_weights);

    if (fc1_expert_biases.has_value() || fc2_expert_biases.has_value()) {
      CHECK_INPUT_TYPE(fc1_expert_biases.value(), mOutputDtype);
      CHECK_INPUT_TYPE(fc2_expert_biases.value(), mOutputDtype);

      CHECK_DIM(2, fc1_expert_biases.value());
      CHECK_DIM(2, fc2_expert_biases.value());
      TVM_FFI_ICHECK_EQ(fc1_expert_weights->shape[0], fc1_expert_biases.value()->shape[0])
          << "fc1_expert_weights and fc1_expert_biases must have the same number of experts.";
      TVM_FFI_ICHECK_EQ(fc2_expert_weights->shape[0], fc2_expert_biases.value()->shape[0])
          << "fc2_expert_weights and fc2_expert_biases must have the same number of experts.";
      TVM_FFI_ICHECK_EQ(fc1_expert_biases.value()->shape[1], fc1_expert_weights->shape[1])
          << "fc1_expert_biases should match fc1_expert_weights output shape.";
      TVM_FFI_ICHECK_EQ(fc2_expert_biases.value()->shape[1], fc2_expert_weights->shape[1])
          << "fc2_expert_biases should match fc2_expert_weights output shape.";
    }

    TVM_FFI_ICHECK_EQ(input->shape[0], token_selected_experts->shape[0])
        << "input and token_selected_experts must have the same num tokens.";
    if (token_final_scales.has_value()) {
      CHECK_DIM(2, token_final_scales.value());
      TVM_FFI_ICHECK_EQ(input->shape[0], token_final_scales.value()->shape[0])
          << "input and token_selected_experts_probs must have the same num tokens.";
      TVM_FFI_ICHECK_EQ(token_selected_experts->shape[1], token_final_scales.value()->shape[1])
          << "token_selected_experts and token_final_scales must have the same number of "
             "experts per token.";
    }
    TVM_FFI_ICHECK_EQ(fc1_expert_weights->shape[0], fc2_expert_weights->shape[0])
        << "fc1_expert_weights and fc2_expert_weights must have the same number of experts.";
    TVM_FFI_ICHECK_EQ(fc1_expert_weights->shape[1],
                      fc2_expert_weights->shape[2] * mInnerDimMultiplier * 2)
        << "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.";

    int experts_per_token = token_selected_experts->shape[1];
    int64_t num_rows = input->shape[0];
    int64_t hidden_size = fc2_expert_weights->shape[1];
    int64_t inter_size = fc2_expert_weights->shape[2] * mInnerDimMultiplier;

    if (isWMxfp4AMxfp8Quant() || isWMxfp4AFp8Quant()) {
      // MXFP4 weights are required to bealigned to 128 bytes
      TVM_FFI_ICHECK_EQ(hidden_size % 128, 0)
          << "hidden_size must be divisible by 128 for MXFP4 weights";
      TVM_FFI_ICHECK_EQ(inter_size % 128, 0)
          << "inter_size must be divisible by 128 for MXFP4 weights";
    } else {
      // TMA requires at least 128 bit alignment
      auto min_alignment = 128 / (8 * std::min(mActivationDtype.bits * mActivationDtype.lanes / 8,
                                               mWeightDtype.bits * mWeightDtype.lanes / 8));
      TVM_FFI_ICHECK_EQ(hidden_size % min_alignment, 0)
          << "hidden_size " << hidden_size << " must be divisible by " << min_alignment
          << " for weights";
      TVM_FFI_ICHECK_EQ(inter_size % min_alignment, 0)
          << "inter_size " << inter_size << " must be divisible by " << min_alignment
          << " for weights";
    }

    int const num_experts_on_rank = fc2_expert_weights->shape[0];
    auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
    ActivationType base_activation_type = ActivationType::Swiglu;
    if (swiglu_alpha.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_alpha.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_alpha.value()->shape[0], num_experts_on_rank)
          << "swiglu_alpha must have num_experts_on_rank elements.";
      base_activation_type = ActivationType::SwigluBias;
    }
    if (swiglu_beta.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_beta.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_beta.value()->shape[0], num_experts_on_rank)
          << "swiglu_beta must have num_experts_on_rank elements.";
      base_activation_type = ActivationType::SwigluBias;
    }
    if (swiglu_limit.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_limit.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_limit.value()->shape[0], num_experts_on_rank)
          << "swiglu_limit must have num_experts_on_rank elements.";
      base_activation_type = ActivationType::SwigluBias;
    }
    auto activation_params = ActivationParams(
        base_activation_type,
        reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value()->data
                                                                : nullptr),
        reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value()->data
                                                               : nullptr),
        reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value()->data
                                                                : nullptr));

    setRunnerProfiles(profile_ids);

    auto stream = get_stream(input->device);

    WorkspaceInfo workspace_info = getWorkspaceInfo(
        num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
        base_activation_type, parallelism_config, min_latency_mode);

    auto const quant_params =
        getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);
    kernels::MoeMinLatencyParams min_latency_params{};

    // TODO: support lora in the future
    ::tensorrt_llm::kernels::LoraParams lora_params{};
#ifdef USING_OSS_CUTLASS_MOE_GEMM
    mKernelRunner->runMoe(input->data, input_sf.has_value() ? input_sf.value()->data : nullptr,
                          reinterpret_cast<int const*>(token_selected_experts->data),
                          token_final_scales.has_value()
                              ? reinterpret_cast<float const*>(token_final_scales.value()->data)
                              : nullptr,
                          fc1_expert_weights->data,
                          fc1_expert_biases.has_value() ? fc1_expert_biases.value()->data : nullptr,
                          activation_params, fc2_expert_weights->data,
                          fc2_expert_biases.has_value() ? fc2_expert_biases.value()->data : nullptr,
                          quant_params, num_rows, hidden_size, inter_size, num_experts_total,
                          static_cast<int>(experts_per_token),
                          static_cast<char*>(workspace_info.workspace->data), output->data,
                          static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config,
                          enable_alltoall, false, lora_params, mUseDeepSeekFP8BlockScaling,
                          min_latency_mode, min_latency_params, enable_pdl, stream);
#else
    mKernelRunner->runMoe(
        input->data, input_sf.has_value() ? input_sf.value()->data : nullptr,
        reinterpret_cast<int const*>(token_selected_experts->data),
        token_final_scales.has_value()
            ? reinterpret_cast<float const*>(token_final_scales.value()->data)
            : nullptr,
        fc1_expert_weights->data,
        fc1_expert_biases.has_value() ? fc1_expert_biases.value()->data : nullptr,
        activation_params, fc2_expert_weights->data,
        fc2_expert_biases.has_value() ? fc2_expert_biases.value()->data : nullptr, quant_params,
        num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
        static_cast<char*>(workspace_info.workspace), output->data,
        static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, false, lora_params,
        mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, enable_pdl, stream);
#endif
  }

  void runMoeMinLantency(Tensor output, Tensor input, Tensor token_selected_experts,
                         Optional<Tensor> token_final_scales, Tensor fc1_expert_weights,
                         Optional<Tensor> fc1_expert_biases, Tensor fc2_expert_weights,
                         Optional<Tensor> fc2_expert_biases, Optional<Array<Tensor>> quant_scales,
                         Optional<Tensor> input_sf, Optional<Tensor> swiglu_alpha,
                         Optional<Tensor> swiglu_beta, Optional<Tensor> swiglu_limit,
                         Tensor num_active_experts_per_node, Tensor experts_to_token_score,
                         Tensor active_expert_global_ids, int64_t tp_size, int64_t tp_rank,
                         int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
                         int64_t cluster_rank, bool enable_alltoall, bool min_latency_mode,
                         Optional<Array<int64_t>> profile_ids, bool enable_pdl) {
    std::lock_guard<std::mutex> lock(mMutex);

    CHECK_INPUT_TYPE(input, mActivationDtype)
    CHECK_INPUT_TYPE(token_selected_experts, dl_int32)
    if (token_final_scales) {
      CHECK_INPUT_TYPE(token_final_scales.value(), dl_float32)
    }
    if (mWeightDtype == dl_uint4x2) {
      // Since dlpack does not support uint4x2, here uses uint8 to bypass
      CHECK_INPUT_TYPE(fc1_expert_weights, dl_uint8);
      CHECK_INPUT_TYPE(fc2_expert_weights, dl_uint8)
    } else {
      CHECK_INPUT_TYPE(fc1_expert_weights, mWeightDtype);
      CHECK_INPUT_TYPE(fc2_expert_weights, mWeightDtype)
    }

    CHECK_DIM(2, input);
    CHECK_DIM(2, token_selected_experts);

    CHECK_DIM(3, fc1_expert_weights);
    CHECK_DIM(3, fc2_expert_weights);

    if (fc1_expert_biases.has_value() || fc2_expert_biases.has_value()) {
      CHECK_INPUT_TYPE(fc1_expert_biases.value(), mOutputDtype);
      CHECK_INPUT_TYPE(fc2_expert_biases.value(), mOutputDtype);
      CHECK_DIM(2, fc1_expert_biases.value());
      CHECK_DIM(2, fc2_expert_biases.value());
      TVM_FFI_ICHECK_EQ(fc1_expert_weights->shape[0], fc1_expert_biases.value()->shape[0])
          << "fc1_expert_weights and fc1_expert_biases must have the same number of experts.";
      TVM_FFI_ICHECK_EQ(fc2_expert_weights->shape[0], fc2_expert_biases.value()->shape[0])
          << "fc2_expert_weights and fc2_expert_biases must have the same number of experts.";
      TVM_FFI_ICHECK_EQ(fc1_expert_biases.value()->shape[1], fc1_expert_weights->shape[1])
          << "fc1_expert_biases should match fc1_expert_weights output shape.";
      TVM_FFI_ICHECK_EQ(fc2_expert_biases.value()->shape[1], fc2_expert_weights->shape[1])
          << "fc2_expert_biases should match fc2_expert_weights output shape.";
    }

    TVM_FFI_ICHECK_EQ(input->shape[0], token_selected_experts->shape[0])
        << "input and token_selected_experts must have the same num tokens.";
    if (token_final_scales) {
      CHECK_DIM(2, token_final_scales.value());
      TVM_FFI_ICHECK_EQ(input->shape[0], token_final_scales.value()->shape[0])
          << "input and token_selected_experts_probs must have the same num tokens.";
      TVM_FFI_ICHECK_EQ(token_selected_experts->shape[1], token_final_scales.value()->shape[1])
          << "token_selected_experts and token_final_scales must have the same number of "
             "experts per token.";
    }
    TVM_FFI_ICHECK_EQ(fc1_expert_weights->shape[0], fc2_expert_weights->shape[0])
        << "fc1_expert_weights and fc2_expert_weights must have the same number of experts.";
    TVM_FFI_ICHECK_EQ(fc1_expert_weights->shape[1],
                      fc2_expert_weights->shape[2] * mInnerDimMultiplier * 2)
        << "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.";

    TVM_FFI_ICHECK(!input_sf.has_value() || isWMxfp4AMxfp8Quant() || isNvfp4Quant())
        << "Block-scaling factors provided for non block-scaling quantization";

    int experts_per_token = token_selected_experts->shape[1];
    int64_t num_rows = input->shape[0];
    int64_t hidden_size = fc2_expert_weights->shape[1];
    int64_t inter_size = fc2_expert_weights->shape[2] * mInnerDimMultiplier;

    int const num_experts_on_rank = fc2_expert_weights->shape[0];
    auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
    ActivationType base_activation_type = ActivationType::Swiglu;
    if (swiglu_alpha.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_alpha.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_alpha.value()->shape[0], num_experts_on_rank)
          << "swiglu_alpha must have num_experts_on_rank elements.";
      base_activation_type = ActivationType::SwigluBias;
    }
    if (swiglu_beta.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_beta.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_beta.value()->shape[0], num_experts_on_rank)
      "swiglu_beta must have num_experts_on_rank elements.";
      base_activation_type = ActivationType::SwigluBias;
    }
    if (swiglu_limit.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_limit.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_limit.value()->shape[0], num_experts_on_rank)
          << "swiglu_limit must have num_experts_on_rank elements.";
      base_activation_type = ActivationType::SwigluBias;
    }
    auto activation_params = ActivationParams(
        base_activation_type,
        reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value()->data
                                                                : nullptr),
        reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value()->data
                                                               : nullptr),
        reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value()->data
                                                                : nullptr));

    setRunnerProfiles(profile_ids);

    auto stream = get_stream(input->device);

    CHECK_DIM(1, num_active_experts_per_node);
    CHECK_INPUT_TYPE(num_active_experts_per_node, dl_int32);
    TVM_FFI_ICHECK_EQ(num_active_experts_per_node->shape[0], 1);

    CHECK_DIM(2, experts_to_token_score);
    CHECK_INPUT_TYPE(experts_to_token_score, dl_float32);
    TVM_FFI_ICHECK_EQ(experts_to_token_score->shape[0], num_experts_on_rank);
    TVM_FFI_ICHECK_EQ(experts_to_token_score->shape[1], num_rows);

    CHECK_DIM(1, active_expert_global_ids);
    CHECK_INPUT_TYPE(active_expert_global_ids, dl_int32);
    TVM_FFI_ICHECK_EQ(active_expert_global_ids->shape[0], num_experts_on_rank);

    kernels::MoeMinLatencyParams min_latency_params{};
    min_latency_params.num_active_experts_per_node =
        static_cast<int*>(num_active_experts_per_node->data);
    min_latency_params.experts_to_token_score = static_cast<float*>(experts_to_token_score->data);
    min_latency_params.active_expert_global_ids = static_cast<int*>(active_expert_global_ids->data);

    WorkspaceInfo workspace_info = getWorkspaceInfo(
        num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
        base_activation_type, parallelism_config, min_latency_mode);

    auto const quant_params =
        getQuantParams(num_experts_on_rank, hidden_size, inter_size, quant_scales);

    // TODO: support lora in the future
    ::tensorrt_llm::kernels::LoraParams lora_params{};
#ifdef USING_OSS_CUTLASS_MOE_GEMM
    mKernelRunner->runMoe(input->data, input_sf.has_value() ? input_sf.value()->data : nullptr,
                          reinterpret_cast<int const*>(token_selected_experts->data),
                          token_final_scales.has_value()
                              ? reinterpret_cast<float const*>(token_final_scales.value()->data)
                              : nullptr,
                          fc1_expert_weights->data,
                          fc1_expert_biases.has_value() ? fc1_expert_biases.value()->data : nullptr,
                          activation_params, fc2_expert_weights->data,
                          fc2_expert_biases.has_value() ? fc2_expert_biases.value()->data : nullptr,
                          quant_params, num_rows, hidden_size, inter_size, num_experts_total,
                          static_cast<int>(experts_per_token),
                          static_cast<char*>(workspace_info.workspace->data), output->data,
                          static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config,
                          enable_alltoall, false, lora_params, mUseDeepSeekFP8BlockScaling,
                          min_latency_mode, min_latency_params, enable_pdl, stream);
#else
    mKernelRunner->runMoe(
        input->data, input_sf.has_value() ? input_sf.value()->data : nullptr,
        reinterpret_cast<int const*>(token_selected_experts->data),
        token_final_scales.has_value()
            ? reinterpret_cast<float const*>(token_final_scales.value()->data)
            : nullptr,
        fc1_expert_weights->data,
        fc1_expert_biases.has_value() ? fc1_expert_biases.value()->data : nullptr,
        activation_params, fc2_expert_weights->data,
        fc2_expert_biases.has_value() ? fc2_expert_biases.value()->data : nullptr, quant_params,
        num_rows, hidden_size, inter_size, num_experts_total, static_cast<int>(experts_per_token),
        static_cast<char*>(workspace_info.workspace), output->data,
        static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config, false, lora_params,
        mUseDeepSeekFP8BlockScaling, min_latency_mode, min_latency_params, enable_pdl, stream);
#endif
  }

  int64_t getTacticNum() {
    std::lock_guard<std::mutex> lock(mMutex);
    return mAllProfiles.size();
  }

  void runGemmProfile(Tensor input, Tensor fc1_expert_weights, Optional<Tensor> fc1_expert_biases,
                      Tensor fc2_expert_weights, Optional<Tensor> fc2_expert_biases, int64_t top_k,
                      int64_t tp_size, int64_t tp_rank, int64_t ep_size, int64_t ep_rank,
                      int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
                      bool min_latency_mode, int64_t gemm_idx, int64_t profile_id,
                      bool do_preparation, bool enable_pdl) {
    std::lock_guard<std::mutex> lock(mMutex);

    // TODO: support profiling under fp8 block scaling in the future
    if (mUseDeepSeekFP8BlockScaling) {
      return;
    }

    int64_t num_rows = input->shape[0];
    int64_t hidden_size = fc2_expert_weights->shape[1];
    int64_t inter_size = fc2_expert_weights->shape[2] * mInnerDimMultiplier;
    int64_t group_size_ =
        isInt4Quant() ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size
                      : -1;
    int64_t group_size =
        isWFP4A16Quant()
            ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size
            : group_size_;
    int const num_experts = static_cast<int>(fc2_expert_weights->shape[0] * ep_size);

    // Get specific profile configs according to the profile_id.
    // Fallback tactic is set to be 0
    // TODO: use the best tactic id found offline for a better default inference perf
    auto profile = profile_id == -1 ? mAllProfiles.front() : mAllProfiles[profile_id];

    auto stream = get_stream(input->device);

    auto const* expert_weights_ptr =
        (gemm_idx == 1) ? fc1_expert_weights->data : fc2_expert_weights->data;

    // Preparation phase, only enabled during autotuning warmup phase.
    if (do_preparation) {
      // Set profiled gemm idx
      mProfiler->mGemmToProfile = (gemm_idx == 1) ? profiler_backend::GemmToProfile::GEMM_1
                                                  : profiler_backend::GemmToProfile::GEMM_2;

      // mProfiler init
      auto parallelism_config = kernels::MOEParallelismConfig(
          static_cast<int>(tp_size), static_cast<int>(tp_rank), static_cast<int>(ep_size),
          static_cast<int>(ep_rank), static_cast<int>(cluster_size),
          static_cast<int>(cluster_rank));

      bool USE_BIAS = fc1_expert_biases.has_value() || fc2_expert_biases.has_value();
      bool USE_LORA = false;
      auto activation_dtype =
          (mUseW4GroupScaling && !isWFP4A16Quant()) ? dl_float8_e4m3fn : mActivationDtype;
      activation_dtype = isNvfp4Quant() ? dl_int64 : activation_dtype;
#ifdef USING_OSS_CUTLASS_MOE_GEMM
      mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
                      DtypeUtils::dataType(activation_dtype), DtypeUtils::dataType(mWeightDtype),
                      DtypeUtils::dataType(mOutputDtype), num_experts, static_cast<int>(top_k),
                      hidden_size, inter_size, group_size, ActivationType::Swiglu, USE_BIAS,
                      USE_LORA, min_latency_mode,
                      /*need_weights*/ false, parallelism_config, enable_alltoall);
#else
      mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
                      DtypeUtils::dataType(activation_dtype), DtypeUtils::dataType(mWeightDtype),
                      DtypeUtils::dataType(mOutputDtype), num_experts, static_cast<int>(top_k),
                      hidden_size, inter_size, group_size, ActivationType::Swiglu, USE_BIAS,
                      USE_LORA, min_latency_mode,
                      /*need_weights*/ false, parallelism_config);
#endif

      size_t profile_workspace_size = mProfiler->getWorkspaceSize(num_rows);
      int device_id;
      cudaGetDevice(&device_id);
      mProfileWorkspace = alloc_tensor({static_cast<int64_t>(profile_workspace_size)}, dl_int8,
                                       DLDevice{kDLCUDA, device_id});

      mProfiler->prepare(num_rows, static_cast<char*>(mProfileWorkspace->data), expert_weights_ptr,
                         enable_pdl, stream);
    }

    // Profile specific tactic. Assuming at least one preparation phase has been executed already.
    mProfiler->runProfiler(num_rows, profile, static_cast<char*>(mProfileWorkspace->data),
                           expert_weights_ptr, enable_pdl, stream);
  }

  const char* kind() const final { return "fused_moe_runner"; }
  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "run_gemm_profile") {
      return Function::FromTyped(
          [this](Tensor input, Tensor fc1_expert_weights, Optional<Tensor> fc1_expert_biases,
                 Tensor fc2_expert_weights, Optional<Tensor> fc2_expert_biases, int64_t top_k,
                 int64_t tp_size, int64_t tp_rank, int64_t ep_size, int64_t ep_rank,
                 int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
                 bool min_latency_mode, int64_t gemm_idx, int64_t profile_id, bool do_preparation,
                 bool enable_pdl) {
            runGemmProfile(input, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights,
                           fc2_expert_biases, top_k, tp_size, tp_rank, ep_size, ep_rank,
                           cluster_size, cluster_rank, enable_alltoall, min_latency_mode, gemm_idx,
                           profile_id, do_preparation, enable_pdl);
          });
    } else if (name == "get_tactic_num") {
      return Function::FromTyped([this]() -> int64_t { return getTacticNum(); });
    } else if (name == "run_moe") {
      return Function::FromTyped(
          [this](Tensor output, Tensor input, Tensor token_selected_experts,
                 Optional<Tensor> token_final_scales, Tensor fc1_expert_weights,
                 Optional<Tensor> fc1_expert_biases, Tensor fc2_expert_weights,
                 Optional<Tensor> fc2_expert_biases, Optional<Array<Tensor>> quant_scales,
                 Optional<Tensor> input_sf, Optional<Tensor> swiglu_alpha,
                 Optional<Tensor> swiglu_beta, Optional<Tensor> swiglu_limit, int64_t tp_size,
                 int64_t tp_rank, int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
                 int64_t cluster_rank, bool enable_alltoall, bool min_latency_mode,
                 Optional<Array<int64_t>> profile_ids, bool enable_pdl) {
            runMoe(output, input, token_selected_experts, token_final_scales, fc1_expert_weights,
                   fc1_expert_biases, fc2_expert_weights, fc2_expert_biases, quant_scales, input_sf,
                   swiglu_alpha, swiglu_beta, swiglu_limit, tp_size, tp_rank, ep_size, ep_rank,
                   cluster_size, cluster_rank, enable_alltoall, min_latency_mode, profile_ids,
                   enable_pdl);
          });
    } else if (name == "run_moe_min_latency") {
      return Function::FromTyped(
          [this](Tensor output, Tensor input, Tensor token_selected_experts,
                 Optional<Tensor> token_final_scales, Tensor fc1_expert_weights,
                 Optional<Tensor> fc1_expert_biases, Tensor fc2_expert_weights,
                 Optional<Tensor> fc2_expert_biases, Optional<Array<Tensor>> quant_scales,
                 Optional<Tensor> input_sf, Optional<Tensor> swiglu_alpha,
                 Optional<Tensor> swiglu_beta, Optional<Tensor> swiglu_limit,
                 Tensor num_active_experts_per_node, Tensor experts_to_token_score,
                 Tensor active_expert_global_ids, int64_t tp_size, int64_t tp_rank, int64_t ep_size,
                 int64_t ep_rank, int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
                 bool min_latency_mode, Optional<Array<int64_t>> profile_ids, bool enable_pdl) {
            runMoeMinLantency(output, input, token_selected_experts, token_final_scales,
                              fc1_expert_weights, fc1_expert_biases, fc2_expert_weights,
                              fc2_expert_biases, quant_scales, input_sf, swiglu_alpha, swiglu_beta,
                              swiglu_limit, num_active_experts_per_node, experts_to_token_score,
                              active_expert_global_ids, tp_size, tp_rank, ep_size, ep_rank,
                              cluster_size, cluster_rank, enable_alltoall, min_latency_mode,
                              profile_ids, enable_pdl);
          });
    } else {
      return Function(nullptr);
    }
  }

 private:
  struct WorkspaceInfo {
    Tensor workspace{};
    void* src_to_dest_map{};
  };

  std::mutex mMutex;
  std::shared_ptr<kernels::CutlassMoeFCRunnerInterface> mKernelRunner;
  std::shared_ptr<kernels::GemmProfilerBackend> mProfiler;
  DLDataType mActivationDtype;
  DLDataType mWeightDtype;
  DLDataType mOutputDtype;
  // number of elements packed into the inner dimension of a matrix
  // e.g. 16 nvfp4 elements are packed into a single int64 element
  int64_t mInnerDimMultiplier;
  Tensor mProfileWorkspace;

  bool mUseDeepSeekFP8BlockScaling = false;
  bool mUseW4GroupScaling = false;
  bool mUseMxfp8ActScaling = false;

  using Profile = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
  std::vector<Profile> mAllProfiles;

  void setRunnerProfiles(Optional<Array<int64_t>> profile_ids) {
    if (mUseDeepSeekFP8BlockScaling) {
      auto config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig(
          tensorrt_llm::cutlass_extensions::CutlassTileConfigSM90::CtaShape128x16x128B,
          tensorrt_llm::cutlass_extensions::MainloopScheduleType::AUTO,
          tensorrt_llm::cutlass_extensions::EpilogueScheduleType::AUTO,
          tensorrt_llm::cutlass_extensions::ClusterShape::ClusterShape_1x1x1);
      mKernelRunner->setTactic(config, config);
      return;
    }

    auto best_gemm1_profile = mAllProfiles.front();
    auto best_gemm2_profile = mAllProfiles.front();
    if (profile_ids.has_value()) {
      TVM_FFI_ICHECK_EQ(profile_ids.value().size(), 2) << "Expecting 2 profile ids";
      best_gemm1_profile = profile_ids.value()[0] == -1 ? best_gemm1_profile
                                                        : mAllProfiles.at(profile_ids.value()[0]);
      best_gemm2_profile = profile_ids.value()[1] == -1 ? best_gemm2_profile
                                                        : mAllProfiles.at(profile_ids.value()[1]);
    }
    mKernelRunner->setTactic(best_gemm1_profile, best_gemm2_profile);
  }

  WorkspaceInfo getWorkspaceInfo(int64_t num_rows, int64_t hidden_size, int64_t inter_size,
                                 int num_experts, int experts_per_token,
                                 ActivationType activation_type,
                                 kernels::MOEParallelismConfig parallelismConfig,
                                 bool min_latency_mode) {
    size_t moe_workspace_size = mKernelRunner->getWorkspaceSize(
        num_rows, hidden_size, inter_size, num_experts, experts_per_token, activation_type,
        parallelismConfig, /* use_lora */ false, mUseDeepSeekFP8BlockScaling, min_latency_mode,
        mUseW4GroupScaling);
    size_t src_to_dest_map_size = experts_per_token * num_rows * sizeof(int);

    std::vector<size_t> workspaces{moe_workspace_size, src_to_dest_map_size};

    size_t total_workspace_size =
        common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());

    WorkspaceInfo info{};
    int device_id;
    cudaGetDevice(&device_id);
    info.workspace = alloc_tensor({static_cast<int64_t>(total_workspace_size)}, dl_int8,
                                  DLDevice{kDLCUDA, device_id});
    info.src_to_dest_map =
        common::nextWorkspacePtr(static_cast<int8_t*>(info.workspace->data), moe_workspace_size);

    return info;
  }

  kernels::QuantParams getQuantParams(int64_t num_experts_on_rank, int64_t hidden_size,
                                      int64_t inter_size,
                                      Optional<Array<Tensor>> quant_scales) const {
    if (isFp8Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value()) << "Expecting quant scales for fp8 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 4)
          << "Expecting 4 quant scales for fp8 quantization";

      auto const fc1_dequant = quant_scales.value()[0];
      auto const fc2_quant = quant_scales.value()[1];
      auto const fc2_dequant = quant_scales.value()[2];
      auto const fc1_input_dequant = quant_scales.value()[3];

      // Check types
      CHECK_INPUT_TYPE(fc1_dequant, dl_float32);
      CHECK_INPUT_TYPE(fc2_quant, dl_float32);
      CHECK_INPUT_TYPE(fc2_dequant, dl_float32);
      CHECK_INPUT_TYPE(fc1_input_dequant, dl_float32);
      // Check ranks
      CHECK_DIM(1, fc1_dequant);
      TVM_FFI_ICHECK_LE(fc2_quant->ndim, 1) << "fc2 quant must be a scalar or 1-D tensor";
      CHECK_DIM(1, fc2_dequant);
      CHECK_DIM(0, fc1_input_dequant);
      // Check shapes
      TVM_FFI_ICHECK_EQ(fc1_dequant->shape[0], num_experts_on_rank)
          << "fc1 dequant size must be (num_experts_on_rank,)";
      TVM_FFI_ICHECK(fc2_quant->ndim == 0 || fc2_quant->shape[0] == num_experts_on_rank)
          << "fc2 quant must be scalar or (num_experts_on_rank,)";
      TVM_FFI_ICHECK_EQ(fc2_dequant->shape[0], num_experts_on_rank)
          << "fc2 dequant size must be (num_experts_on_rank,)";

      return kernels::QuantParams::FP8(
          static_cast<float const*>(fc1_dequant->data), static_cast<float const*>(fc2_quant->data),
          static_cast<float const*>(fc2_dequant->data),
          /* fp8 output quant scale */ nullptr, static_cast<float const*>(fc1_input_dequant->data),
          fc2_quant->ndim == 1);
    } else if (isWMxfp4AFp8Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value())
          << "Expecting quant scales for W4A8_MXFP4_MXF8 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 5)
          << "Expecting 5 quant scales for W4A8_MXFP4_FP8 quantization";

      auto const fc1_weight_block = quant_scales.value()[0];
      auto const fc1_global = quant_scales.value()[1];
      auto const fc2_act_global = quant_scales.value()[2];
      auto const fc2_weight_block = quant_scales.value()[3];
      auto const fc2_global = quant_scales.value()[4];

      // The input for scale fc1_weight_block / fc2_weight_block is packed into INT32
      constexpr int FP8_PER_INT32 = 4;
      // Check types
      CHECK_INPUT_TYPE(fc1_weight_block, dl_int32);
      CHECK_INPUT_TYPE(fc1_global, dl_float32);
      CHECK_INPUT_TYPE(fc2_act_global, dl_float32);
      CHECK_INPUT_TYPE(fc2_weight_block, dl_int32);
      CHECK_INPUT_TYPE(fc2_global, dl_float32);
      // Check ranks
      CHECK_DIM(3, fc1_weight_block);
      CHECK_DIM(1, fc1_global);
      TVM_FFI_ICHECK_LE(fc2_act_global->ndim, 1) << "fc2 act global must be a scalar or 1-D tensor";
      CHECK_DIM(3, fc2_weight_block);
      CHECK_DIM(1, fc2_global);
      // Check shapes
      TVM_FFI_ICHECK(
          fc1_weight_block->shape[0] == num_experts_on_rank &&
          fc1_weight_block->shape[1] ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) *
                  2 &&
          fc1_weight_block->shape[2] * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 "
             "// block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc1_global->shape[0], num_experts_on_rank)
          << "fc1 global size must be (num_experts_on_rank,)";
      TVM_FFI_ICHECK(fc2_act_global->ndim == 0 || fc2_act_global->shape[0] == num_experts_on_rank)
          << "fc2 act global must be scalar or (num_experts_on_rank,)";
      TVM_FFI_ICHECK(
          fc2_weight_block->shape[0] == num_experts_on_rank &&
          fc2_weight_block->shape[1] ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) &&
          fc2_weight_block->shape[2] * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
             "block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc2_global->shape[0], num_experts_on_rank)
          << "fc2 global size must be (num_experts_on_rank,)";

      return kernels::QuantParams::FP8MXFP4(
          nullptr,
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block->data),
          static_cast<float const*>(fc1_global->data),
          static_cast<float const*>(fc2_act_global->data),
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block->data),
          static_cast<float const*>(fc2_global->data), false, fc2_act_global->ndim == 1);
    } else if (isWMxfp4AMxfp8Quant()) {
#ifdef USING_OSS_CUTLASS_MOE_GEMM
      TVM_FFI_ICHECK(quant_scales.has_value())
          << "Expecting quant scales for W4A8_MXFP4_MXFP8 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 4)
      "Expecting 4 quant scales for W4A8_MXFP4_MXFP8 quantization";

      Tensor fc1_weight_block = quant_scales.value()[0];
      Tensor fc1_global = quant_scales.value()[1];
      Tensor fc2_weight_block = quant_scales.value()[2];
      Tensor fc2_global = quant_scales.value()[3];

      // The input for scale fc1_weight_block / fc2_weight_block is packed into INT32
      constexpr int FP8_PER_INT32 = 4;
      CHECK_INPUT_TYPE(fc1_weight_block, dl_int32);
      CHECK_INPUT_TYPE(fc1_global, dl_float32);
      CHECK_INPUT_TYPE(fc2_weight_block, dl_int32);
      CHECK_INPUT_TYPE(fc2_global, dl_float32);
      CHECK_DIM(3, fc1_weight_block);
      CHECK_DIM(1, fc1_global);
      CHECK_DIM(3, fc2_weight_block);
      CHECK_DIM(1, fc2_global);
      TVM_FFI_ICHECK(
          fc1_weight_block->shape[0] == num_experts_on_rank &&
          fc1_weight_block->shape[1] ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) *
                  2 &&
          fc1_weight_block->shape[2] * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 "
             "// block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc1_global->shape[0], num_experts_on_rank)
          << "fc1 global size must be (num_experts_on_rank,)";
      TVM_FFI_ICHECK(
          fc2_weight_block->shape[0] == num_experts_on_rank &&
          fc2_weight_block->shape[1] ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) &&
          fc2_weight_block->shape[2] * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
             "block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc2_global->shape[0], num_experts_on_rank)
          << "fc2 global size must be (num_experts_on_rank,)";

      return kernels::QuantParams::MXFP8MXFP4(
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block->data),
          static_cast<float const*>(fc1_global->data),
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block->data),
          static_cast<float const*>(fc2_global->data));
#else
      TVM_FFI_ICHECK(false)
          << "MXFP8 x MXFP4 quantization is not supported in OSS Cutlass Moe Gemm";
#endif
    }

    else if (isNvfp4Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value()) << "Expecting quant scales for nvfp4 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 6)
          << "Expecting 6 quant scales for nvfp4 quantization";

      Tensor fc1_act_global = quant_scales.value()[0];
      Tensor fc1_weight_block = quant_scales.value()[1];
      Tensor fc1_global = quant_scales.value()[2];
      Tensor fc2_act_global = quant_scales.value()[3];
      Tensor fc2_weight_block = quant_scales.value()[4];
      Tensor fc2_global = quant_scales.value()[5];

      // The input for scale fc1_weight_block / fc2_weight_block is packed into INT32
      constexpr int FP8_PER_INT32 = 4;
      // Check types
      CHECK_INPUT_TYPE(fc1_act_global, dl_float32);
      CHECK_INPUT_TYPE(fc1_weight_block, dl_int32);
      CHECK_INPUT_TYPE(fc1_global, dl_float32);
      CHECK_INPUT_TYPE(fc2_act_global, dl_float32);
      CHECK_INPUT_TYPE(fc2_weight_block, dl_int32);
      CHECK_INPUT_TYPE(fc2_global, dl_float32);
      // Check ranks
      TVM_FFI_ICHECK_LE(fc1_act_global->ndim, 1) << "fc1 act global must be a scalar or 1-D tensor";
      CHECK_DIM(3, fc1_weight_block);
      CHECK_DIM(1, fc1_global);
      TVM_FFI_ICHECK_LE(fc2_act_global->ndim, 1) << "fc2 act global must be a scalar or 1-D tensor";
      CHECK_DIM(3, fc2_weight_block);
      CHECK_DIM(1, fc2_global);
      // Check shapes
      TVM_FFI_ICHECK(fc1_act_global->ndim == 0 || fc1_act_global->shape[0] == num_experts_on_rank)
          << "fc1 act global must be scalar or (num_experts_on_rank,)";
      TVM_FFI_ICHECK(
          fc1_weight_block->shape[0] == num_experts_on_rank &&
          fc1_weight_block->shape[1] ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4) *
                  2 &&
          fc1_weight_block->shape[2] * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4))
          << "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // 4 "
             "// block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc1_global->shape[0], num_experts_on_rank)
          << "fc1 global size must be (num_experts_on_rank,)";
      TVM_FFI_ICHECK(fc2_act_global->ndim == 0 || fc2_act_global->shape[0] == num_experts_on_rank)
          << "fc2 act global must be scalar or (num_experts_on_rank,)";
      TVM_FFI_ICHECK(
          fc2_weight_block->shape[0] == num_experts_on_rank &&
          fc2_weight_block->shape[1] ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4) &&
          fc2_weight_block->shape[2] * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4))
          << "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
             "block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc2_global->shape[0], num_experts_on_rank)
          << "fc2 global size must be (num_experts_on_rank,)";

      return kernels::QuantParams::FP4(
          static_cast<float const*>(fc1_act_global->data),
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block->data),
          static_cast<float const*>(fc1_global->data),
          static_cast<float const*>(fc2_act_global->data),
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block->data),
          static_cast<float const*>(fc2_global->data), fc1_act_global->ndim == 1,
          fc2_act_global->ndim == 1);
    } else if (mUseDeepSeekFP8BlockScaling) {
      Tensor fc1_scales = quant_scales.value()[0];
      Tensor fc2_scales = quant_scales.value()[1];
      return kernels::QuantParams::FP8BlockScaling(static_cast<float const*>(fc1_scales->data),
                                                   static_cast<float const*>(fc2_scales->data));
    } else if (isWFP4A16Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value()) << "Expecting quant scales for W4 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 2)
          << "Expecting 2 quant scales for W4A16 quantization";

      Tensor fc1_weight_scales = quant_scales.value()[0];
      Tensor fc2_weight_scales = quant_scales.value()[1];
      int group_size = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size;
      return kernels::QuantParams::GroupWise(group_size,
                                             static_cast<void const*>(fc1_weight_scales->data),
                                             static_cast<void const*>(fc2_weight_scales->data),
                                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    } else if (isInt4Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value()) << "Expecting quant scales for INT4 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 8)
          << "Expecting 8 quant scales for INT4 quantization";
      Tensor fc1_weight_scales = quant_scales.value()[0];
      Tensor fc2_weight_scales = quant_scales.value()[1];
      Tensor fc1_act_scales = quant_scales.value()[2];
      Tensor fc2_act_scales = quant_scales.value()[3];
      Tensor fc1_weight_zeros = quant_scales.value()[4];
      Tensor fc2_weight_zeros = quant_scales.value()[5];
      Tensor fc1_alpha = quant_scales.value()[6];
      Tensor fc2_alpha = quant_scales.value()[7];
      int group_size = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size;
      return kernels::QuantParams::GroupWise(
          group_size, static_cast<void const*>(fc1_weight_scales->data),
          static_cast<void const*>(fc2_weight_scales->data),
          static_cast<void const*>(get_numel(fc1_act_scales) > 0 ? fc1_act_scales->data : nullptr),
          static_cast<void const*>(get_numel(fc2_act_scales) > 0 ? fc2_act_scales->data : nullptr),
          static_cast<void const*>(get_numel(fc1_weight_zeros) > 0 ? fc1_weight_zeros->data
                                                                   : nullptr),
          static_cast<void const*>(get_numel(fc2_weight_zeros) > 0 ? fc2_weight_zeros->data
                                                                   : nullptr),
          static_cast<float const*>(get_numel(fc1_alpha) > 0 ? fc1_alpha->data : nullptr),
          static_cast<float const*>(get_numel(fc2_alpha) > 0 ? fc2_alpha->data : nullptr));
    } else {
      return kernels::QuantParams{};
    }
  }

  bool isFp8Quant() const {
    return !mUseDeepSeekFP8BlockScaling && mActivationDtype == dl_float8_e4m3fn &&
           mWeightDtype == dl_float8_e4m3fn;
  }

  bool isNvfp4Quant() const {
    return mWeightDtype == dl_int64 &&
           mActivationDtype != dl_float8_e4m3fn;  // FP8 activation does not use FP4
  }

  bool isWFP4A16Quant() const { return mUseW4GroupScaling && mWeightDtype == dl_uint8; }

  bool isInt4Quant() const { return mWeightDtype == dl_uint4x2; }

  bool isW4AFp8Quant() const { return mActivationDtype == dl_float8_e4m3fn && isInt4Quant(); }

  bool isWMxfp4AFp8Quant() const {
    return mActivationDtype == dl_float8_e4m3fn && mWeightDtype == dl_int64 && !mUseMxfp8ActScaling;
  }

  bool isWMxfp4AMxfp8Quant() const {
    return mActivationDtype == dl_float8_e4m3fn && mWeightDtype == dl_int64 && mUseMxfp8ActScaling;
  }
};

tvm::ffi::Module init(DLDataType activation_dtype, DLDataType weight_dtype, DLDataType output_dtype,
                      bool use_deepseek_fp8_block_scale, bool use_w4_group_scaling,
                      bool use_mxfp8_act_scaling) {
  auto ptr = tvm::ffi::make_object<FusedMoeRunner>(activation_dtype, weight_dtype, output_dtype,
                                                   use_deepseek_fp8_block_scale,
                                                   use_w4_group_scaling, use_mxfp8_act_scaling);
  return tvm::ffi::Module(ptr);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);
