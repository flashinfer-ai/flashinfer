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
#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_mixed_utils.h"

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

    return nvinfer1::DataType::kFLOAT;  // supress compiler warning
  }

 private:
  DtypeUtils() = default;
};

class FusedMoeRunner : public tvm::ffi::ModuleObj {
 public:
  template <
      typename TypeAct, typename TypeWeight, bool NeedQuant = false, bool IsMXFPX = false,
      kernels::Sm90Wfp4Afp8ScaleMode Sm90Wfp4Afp8Mode = kernels::Sm90Wfp4Afp8ScaleMode::kDisabled>
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
          return std::make_unique<kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, half, half,
                                                              IsMXFPX, Sm90Wfp4Afp8Mode>>();
        } else {
          return std::make_unique<kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, half, TypeAct,
                                                              half, IsMXFPX, Sm90Wfp4Afp8Mode>>();
        }
#ifdef ENABLE_BF16
      case bfloat16_code:
        if constexpr (NeedQuant) {
          return std::make_unique<
              kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, __nv_bfloat16,
                                          __nv_bfloat16, IsMXFPX, Sm90Wfp4Afp8Mode>>();
        } else {
          return std::make_unique<
              kernels::CutlassMoeFCRunner<TypeAct, TypeWeight, __nv_bfloat16, TypeAct,
                                          __nv_bfloat16, IsMXFPX, Sm90Wfp4Afp8Mode>>();
        }
#endif
      default:
        TVM_FFI_ICHECK(false) << "Invalid output type " << DLDataTypeToString(output_type)
                              << " specified for " << DLDataTypeToString(mActivationDtype);
    }

    return nullptr;  // supress compiler warning
  };

  FusedMoeRunner(DLDataType activation_dtype, DLDataType weight_dtype, DLDataType output_dtype,
                 bool use_deepseek_fp8_block_scale, bool use_w4_group_scaling,
                 bool use_mxfp8_act_scaling, bool use_packed_weights, bool use_fused_finalize,
                 bool use_wfp4afp8_humming) {
    mActivationDtype = activation_dtype;
    mWeightDtype = weight_dtype;
    mUsePackedWeights = use_packed_weights;
    mOutputDtype = output_dtype;
    mUseDeepSeekFP8BlockScaling = use_deepseek_fp8_block_scale;
    mUseW4GroupScaling = use_w4_group_scaling;
    mUseMxfp8ActScaling = use_mxfp8_act_scaling;
    mUseFusedFinalize = use_fused_finalize;
    mUseWfp4Afp8Humming = use_wfp4afp8_humming;
    mSm90Wfp4Afp8Mode = kernels::Sm90Wfp4Afp8ScaleMode::kDisabled;
    mInnerDimMultiplier = 1;

    auto make_humming_runner = [&] {
      mInnerDimMultiplier = 2;
      mSm90Wfp4Afp8Mode = kernels::Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0;
      TVM_FFI_ICHECK(mActivationDtype == dl_float16 || mActivationDtype == dl_bfloat16)
          << "Humming-style MXFP4 x FP8 requires FP16/BF16 inputs and online FP8 activation "
             "quantization.";
      TVM_FFI_ICHECK(mActivationDtype == mOutputDtype)
          << "Humming-style MXFP4 x FP8 online activation quantization currently requires "
             "activation dtype and output dtype to match.";
      mKernelRunner =
          switch_output_type<__nv_fp8_e4m3, kernels::Fp4Type, true, false,
                             kernels::Sm90Wfp4Afp8ScaleMode::kHummingPreMmaE8M0>(mOutputDtype);
    };

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
    if (isWMxfp8AMxfp8Quant()) {
      mKernelRunner = switch_output_type<__nv_fp8_e4m3, __nv_fp8_e4m3, false, true>(mOutputDtype);
    } else if (isFp8Quant()) {
      mKernelRunner = switch_output_type<__nv_fp8_e4m3, __nv_fp8_e4m3>(mOutputDtype);
    }
#endif
#ifdef ENABLE_FP4
    int const sm = common::getSMVersion();
    if (sm >= 100 && (isWMxfp4AMxfp8Quant() || isWMxfp4AFp8Quant())) {
      mInnerDimMultiplier = 16;  // 16 FP4 -> 1 LONG
      mKernelRunner = switch_output_type<__nv_fp8_e4m3, kernels::Fp4Type>(mOutputDtype);
    }

#if 0
    // PHASE3_POST_MMA_PLACEHOLDER: future SM90 post-MMA MXFP4 paths should use
    // uint8-packed MXFP4 storage and select their Sm90Wfp4Afp8ScaleMode explicitly.
    // Enabling these paths also requires updating the shared predicates for
    // the SM90 uint8-packed input contract.
    if (sm == 90 && isWMxfp4AFp8Quant()) {
      mInnerDimMultiplier = 2;
      mSm90Wfp4Afp8Mode = kernels::Sm90Wfp4Afp8ScaleMode::kPostMmaFp8Act;
      mKernelRunner = switch_output_type<
          __nv_fp8_e4m3, kernels::Fp4Type, false, false,
          kernels::Sm90Wfp4Afp8ScaleMode::kPostMmaFp8Act>(mOutputDtype);
    }

    if (sm == 90 && isWMxfp4AMxfp8Quant()) {
      mInnerDimMultiplier = 2;
      mSm90Wfp4Afp8Mode = kernels::Sm90Wfp4Afp8ScaleMode::kPostMmaMxfp8Act;
      mKernelRunner = switch_output_type<
          __nv_fp8_e4m3, kernels::Fp4Type, false, false,
          kernels::Sm90Wfp4Afp8ScaleMode::kPostMmaMxfp8Act>(mOutputDtype);
    }
#endif

    if (isWMxfp4AFp8HummingQuant()) {
      TVM_FFI_ICHECK_EQ(sm, 90) << "Humming-style MXFP4 x FP8 is only supported on SM90.";
      make_humming_runner();
    }

    if (isNvfp4Quant()) {
      mInnerDimMultiplier = 16;
      switch (encode_dlpack_dtype(mActivationDtype)) {
        case float16_code:
#ifdef ENABLE_BF16
        case bfloat16_code:
#endif
          mKernelRunner =
              switch_output_type<kernels::Fp4Type, kernels::Fp4Type, true>(mOutputDtype);
          break;
        default:
          mKernelRunner =
              switch_output_type<kernels::Fp4Type, kernels::Fp4Type, false>(mOutputDtype);
      }
    }

    if (isWFP4A16Quant()) {
      TVM_FFI_ICHECK_EQ(mActivationDtype, dl_bfloat16)
          << "SM90 MXFP4 W4A16 supports BF16 activations only; FP16 is incompatible with the "
             "interleaved weight layout.";
      mInnerDimMultiplier = 2;
#ifdef ENABLE_BF16
      mKernelRunner =
          std::make_shared<kernels::CutlassMoeFCRunner<__nv_bfloat16, kernels::Fp4Type>>();
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

    // Must be set before enumerating tactics below: it gates whether GEMM2 finalize-fusion
    // tactics are produced (mayHaveFinalizeFused) and the corresponding workspace sizing.
    mKernelRunner->use_fused_finalize_ = mUseFusedFinalize;

    mProfiler = std::make_shared<kernels::GemmProfilerBackend>();
    // Get tactics for both GEMM1 and GEMM2, combine them
    auto gemm1_tactics = mKernelRunner->getTactics(kernels::MoeGemmId::GEMM_1);
    auto gemm2_tactics = mKernelRunner->getTactics(kernels::MoeGemmId::GEMM_2);
    mGemm1TacticCount = static_cast<int64_t>(gemm1_tactics.size());
    mGemm2TacticCount = static_cast<int64_t>(gemm2_tactics.size());
    mAllProfiles = gemm1_tactics;
    mAllProfiles.insert(mAllProfiles.end(), gemm2_tactics.begin(), gemm2_tactics.end());
    TVM_FFI_ICHECK(!mAllProfiles.empty())
        << "No valid tactics available for fused moe op with the requested input combination "
           "Activation: "
        << DLDataTypeToString(mActivationDtype) << ", Weight: " << DLDataTypeToString(mWeightDtype)
        << ", Output: " << DLDataTypeToString(mOutputDtype);
  }

  void runMoe(TensorView output, TensorView input, TensorView token_selected_experts,
              Optional<TensorView> token_final_scales, TensorView fc1_expert_weights,
              Optional<TensorView> fc1_expert_biases, TensorView fc2_expert_weights,
              Optional<TensorView> fc2_expert_biases, Optional<Array<Tensor>> quant_scales,
              Optional<TensorView> input_sf, Optional<TensorView> swiglu_alpha,
              Optional<TensorView> swiglu_beta, Optional<TensorView> swiglu_limit,
              bool swizzled_input_sf, int64_t tp_size, int64_t tp_rank, int64_t ep_size,
              int64_t ep_rank, int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
              bool min_latency_mode, Optional<Array<int64_t>> profile_ids, bool enable_pdl,
              ActivationType base_activation_type = ActivationType::Swiglu,
              Optional<TensorView> workspace_buffer = Optional<TensorView>{}) {
    std::lock_guard<std::mutex> lock(mMutex);
    ffi::CUDADeviceGuard device_guard(input.device().device_id);

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
      TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(0), fc1_expert_biases.value().size(0))
          << "fc1_expert_weights and fc1_expert_biases must have the same number of experts.";
      TVM_FFI_ICHECK_EQ(fc2_expert_weights.size(0), fc2_expert_biases.value().size(0))
          << "fc2_expert_weights and fc2_expert_biases must have the same number of experts.";
      TVM_FFI_ICHECK_EQ(fc1_expert_biases.value().size(1), fc1_expert_weights.size(1))
          << "fc1_expert_biases should match fc1_expert_weights output shape.";
      TVM_FFI_ICHECK_EQ(fc2_expert_biases.value().size(1), fc2_expert_weights.size(1))
          << "fc2_expert_biases should match fc2_expert_weights output shape.";
    }

    TVM_FFI_ICHECK_EQ(input.size(0), token_selected_experts.size(0))
        << "input and token_selected_experts must have the same num tokens.";
    if (token_final_scales.has_value()) {
      CHECK_DIM(2, token_final_scales.value());
      TVM_FFI_ICHECK_EQ(input.size(0), token_final_scales.value().size(0))
          << "input and token_selected_experts_probs must have the same num tokens.";
      TVM_FFI_ICHECK_EQ(token_selected_experts.size(1), token_final_scales.value().size(1))
          << "token_selected_experts and token_final_scales must have the same number of "
             "experts per token.";
    }
    TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(0), fc2_expert_weights.size(0))
        << "fc1_expert_weights and fc2_expert_weights must have the same number of experts.";
    if (isGatedActivation(base_activation_type)) {
      TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(1),
                        fc2_expert_weights.size(2) * mInnerDimMultiplier * 2)
          << "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.";
    } else {
      TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(1),
                        fc2_expert_weights.size(2) * mInnerDimMultiplier)
          << "fc1_expert_weights inter size must be equal to fc2_expert_weights inter size.";
    }

    int experts_per_token = token_selected_experts.size(1);
    int64_t num_rows = input.size(0);
    int64_t hidden_size = fc2_expert_weights.size(1);
    int64_t inter_size = fc2_expert_weights.size(2) * mInnerDimMultiplier;

    if (isWMxfp4AMxfp8Quant() || isWMxfp4AFp8Quant() || isWMxfp4AFp8HummingQuant()) {
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

    int const num_experts_on_rank = fc2_expert_weights.size(0);
    auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
    if (swiglu_alpha.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_alpha.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_alpha.value().size(0), num_experts_on_rank)
          << "swiglu_alpha must have num_experts_on_rank elements.";
    }
    if (swiglu_beta.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_beta.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_beta.value().size(0), num_experts_on_rank)
          << "swiglu_beta must have num_experts_on_rank elements.";
    }
    if (swiglu_limit.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_limit.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_limit.value().size(0), num_experts_on_rank)
          << "swiglu_limit must have num_experts_on_rank elements.";
    }
    // Swiglu + swiglu_alpha/beta/limit selects the SwigluBias kernel; other gated activations
    // (e.g. SwigluStep) keep their own kernel.
    if (base_activation_type == ActivationType::Swiglu &&
        (swiglu_alpha.has_value() || swiglu_beta.has_value() || swiglu_limit.has_value())) {
      base_activation_type = ActivationType::SwigluBias;
    }
    auto activation_params = ActivationParams(
        base_activation_type,
        reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value().data_ptr()
                                                                : nullptr),
        reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value().data_ptr()
                                                               : nullptr),
        reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value().data_ptr()
                                                                : nullptr));

    setRunnerProfiles(profile_ids);

    auto stream = get_stream(input.device());

    WorkspaceInfo workspace_info =
        getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
                         static_cast<int>(experts_per_token), base_activation_type,
                         parallelism_config, min_latency_mode, input.device(), workspace_buffer);

    int64_t const routed_tokens = input.size(0) * token_selected_experts.size(1);
    auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size,
                                             routed_tokens, quant_scales, base_activation_type);
    kernels::MoeMinLatencyParams min_latency_params{};

    // TODO: support lora in the future
    ::tensorrt_llm::kernels::LoraParams lora_params{};
    // HACK Define default values for parameters we don't have good values for
    int64_t const unpadded_hidden_size = hidden_size;  // Assume no padding by default
    bool const use_lora = false;                       // No lora support yet
#ifdef USING_OSS_CUTLASS_MOE_GEMM
    mKernelRunner->runMoe(
        input.data_ptr(), input_sf.has_value() ? input_sf.value().data_ptr() : nullptr,
        swizzled_input_sf, reinterpret_cast<int const*>(token_selected_experts.data_ptr()),
        token_final_scales.has_value()
            ? reinterpret_cast<float const*>(token_final_scales.value().data_ptr())
            : nullptr,
        fc1_expert_weights.data_ptr(),
        fc1_expert_biases.has_value() ? fc1_expert_biases.value().data_ptr() : nullptr,
        activation_params, fc2_expert_weights.data_ptr(),
        fc2_expert_biases.has_value() ? fc2_expert_biases.value().data_ptr() : nullptr,
        quant_params, num_rows, hidden_size, unpadded_hidden_size, inter_size, num_experts_total,
        static_cast<int>(experts_per_token), static_cast<char*>(workspace_info.workspace_ptr),
        output.data_ptr(), static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config,
        enable_alltoall, use_lora, lora_params, mUseDeepSeekFP8BlockScaling, mUseMxfp8ActScaling,
        min_latency_mode, min_latency_params, enable_pdl, stream);
#else
    mKernelRunner->runMoe(
        input.data_ptr(), input_sf.has_value() ? input_sf.value().data_ptr() : nullptr,
        swizzled_input_sf, reinterpret_cast<int const*>(token_selected_experts.data_ptr()),
        token_final_scales.has_value()
            ? reinterpret_cast<float const*>(token_final_scales.value().data_ptr())
            : nullptr,
        fc1_expert_weights.data_ptr(),
        fc1_expert_biases.has_value() ? fc1_expert_biases.value().data_ptr() : nullptr,
        activation_params, fc2_expert_weights.data_ptr(),
        fc2_expert_biases.has_value() ? fc2_expert_biases.value().data_ptr() : nullptr,
        quant_params, num_rows, hidden_size, inter_size, num_experts_total,
        static_cast<int>(experts_per_token), static_cast<char*>(workspace_info.workspace_ptr),
        output.data_ptr(), static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config,
        false, lora_params, mUseDeepSeekFP8BlockScaling, mUseMxfp8ActScaling, min_latency_mode,
        min_latency_params, enable_pdl, stream);
#endif
  }

  void runMoeMinLantency(TensorView output, TensorView input, TensorView token_selected_experts,
                         Optional<TensorView> token_final_scales, TensorView fc1_expert_weights,
                         Optional<TensorView> fc1_expert_biases, TensorView fc2_expert_weights,
                         Optional<TensorView> fc2_expert_biases,
                         Optional<Array<Tensor>> quant_scales, Optional<TensorView> input_sf,
                         Optional<TensorView> swiglu_alpha, Optional<TensorView> swiglu_beta,
                         Optional<TensorView> swiglu_limit, bool swizzled_input_sf,
                         TensorView num_active_experts_per_node, TensorView experts_to_token_score,
                         TensorView active_expert_global_ids, int64_t tp_size, int64_t tp_rank,
                         int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
                         int64_t cluster_rank, bool enable_alltoall, bool min_latency_mode,
                         Optional<Array<int64_t>> profile_ids, bool enable_pdl,
                         ActivationType base_activation_type = ActivationType::Swiglu,
                         Optional<TensorView> workspace_buffer = Optional<TensorView>{}) {
    std::lock_guard<std::mutex> lock(mMutex);
    ffi::CUDADeviceGuard device_guard(input.device().device_id);

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
      TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(0), fc1_expert_biases.value().size(0))
          << "fc1_expert_weights and fc1_expert_biases must have the same number of experts.";
      TVM_FFI_ICHECK_EQ(fc2_expert_weights.size(0), fc2_expert_biases.value().size(0))
          << "fc2_expert_weights and fc2_expert_biases must have the same number of experts.";
      TVM_FFI_ICHECK_EQ(fc1_expert_biases.value().size(1), fc1_expert_weights.size(1))
          << "fc1_expert_biases should match fc1_expert_weights output shape.";
      TVM_FFI_ICHECK_EQ(fc2_expert_biases.value().size(1), fc2_expert_weights.size(1))
          << "fc2_expert_biases should match fc2_expert_weights output shape.";
    }

    TVM_FFI_ICHECK_EQ(input.size(0), token_selected_experts.size(0))
        << "input and token_selected_experts must have the same num tokens.";
    if (token_final_scales) {
      CHECK_DIM(2, token_final_scales.value());
      TVM_FFI_ICHECK_EQ(input.size(0), token_final_scales.value().size(0))
          << "input and token_selected_experts_probs must have the same num tokens.";
      TVM_FFI_ICHECK_EQ(token_selected_experts.size(1), token_final_scales.value().size(1))
          << "token_selected_experts and token_final_scales must have the same number of "
             "experts per token.";
    }
    TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(0), fc2_expert_weights.size(0))
        << "fc1_expert_weights and fc2_expert_weights must have the same number of experts.";
    if (isGatedActivation(base_activation_type)) {
      TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(1),
                        fc2_expert_weights.size(2) * mInnerDimMultiplier * 2)
          << "fc1_expert_weights inter size must be 2 times fc2_expert_weights inter size.";
    } else {
      TVM_FFI_ICHECK_EQ(fc1_expert_weights.size(1),
                        fc2_expert_weights.size(2) * mInnerDimMultiplier)
          << "fc1_expert_weights inter size must be equal to fc2_expert_weights inter size.";
    }

    TVM_FFI_ICHECK(!input_sf.has_value() || isMxfp8ActScalingQuant() || isNvfp4Quant())
        << "Block-scaling factors provided for non block-scaling quantization";
    TVM_FFI_ICHECK(!isMxfp8ActScalingQuant() || input_sf.has_value())
        << "input_sf must be provided when use_mxfp8_act_scaling=True";

    int experts_per_token = token_selected_experts.size(1);
    int64_t num_rows = input.size(0);
    int64_t hidden_size = fc2_expert_weights.size(1);
    int64_t inter_size = fc2_expert_weights.size(2) * mInnerDimMultiplier;

    int const num_experts_on_rank = fc2_expert_weights.size(0);
    auto const num_experts_total = static_cast<int>(num_experts_on_rank * ep_size);
    auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
    if (swiglu_alpha.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_alpha.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_alpha.value().size(0), num_experts_on_rank)
          << "swiglu_alpha must have num_experts_on_rank elements.";
    }
    if (swiglu_beta.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_beta.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_beta.value().size(0), num_experts_on_rank)
          << "swiglu_beta must have num_experts_on_rank elements.";
    }
    if (swiglu_limit.has_value()) {
      CHECK_INPUT_AND_TYPE(swiglu_limit.value(), dl_float32);
      TVM_FFI_ICHECK_EQ(swiglu_limit.value().size(0), num_experts_on_rank)
          << "swiglu_limit must have num_experts_on_rank elements.";
    }
    // Swiglu + swiglu_alpha/beta/limit selects the SwigluBias kernel; other gated activations
    // (e.g. SwigluStep) keep their own kernel.
    if (base_activation_type == ActivationType::Swiglu &&
        (swiglu_alpha.has_value() || swiglu_beta.has_value() || swiglu_limit.has_value())) {
      base_activation_type = ActivationType::SwigluBias;
    }
    auto activation_params = ActivationParams(
        base_activation_type,
        reinterpret_cast<float const*>(swiglu_alpha.has_value() ? swiglu_alpha.value().data_ptr()
                                                                : nullptr),
        reinterpret_cast<float const*>(swiglu_beta.has_value() ? swiglu_beta.value().data_ptr()
                                                               : nullptr),
        reinterpret_cast<float const*>(swiglu_limit.has_value() ? swiglu_limit.value().data_ptr()
                                                                : nullptr));

    setRunnerProfiles(profile_ids);

    auto stream = get_stream(input.device());

    CHECK_DIM(1, num_active_experts_per_node);
    CHECK_INPUT_TYPE(num_active_experts_per_node, dl_int32);
    TVM_FFI_ICHECK_EQ(num_active_experts_per_node.size(0), 1);

    CHECK_DIM(2, experts_to_token_score);
    CHECK_INPUT_TYPE(experts_to_token_score, dl_float32);
    TVM_FFI_ICHECK_EQ(experts_to_token_score.size(0), num_experts_on_rank);
    TVM_FFI_ICHECK_EQ(experts_to_token_score.size(1), num_rows);

    CHECK_DIM(1, active_expert_global_ids);
    CHECK_INPUT_TYPE(active_expert_global_ids, dl_int32);
    TVM_FFI_ICHECK_EQ(active_expert_global_ids.size(0), num_experts_on_rank);

    kernels::MoeMinLatencyParams min_latency_params{};
    min_latency_params.num_active_experts_per_node =
        static_cast<int*>(num_active_experts_per_node.data_ptr());
    min_latency_params.experts_to_token_score =
        static_cast<float*>(experts_to_token_score.data_ptr());
    min_latency_params.active_expert_global_ids =
        static_cast<int*>(active_expert_global_ids.data_ptr());

    WorkspaceInfo workspace_info =
        getWorkspaceInfo(num_rows, hidden_size, inter_size, num_experts_total,
                         static_cast<int>(experts_per_token), base_activation_type,
                         parallelism_config, min_latency_mode, input.device(), workspace_buffer);

    int64_t const routed_tokens = input.size(0) * token_selected_experts.size(1);
    auto const quant_params = getQuantParams(num_experts_on_rank, hidden_size, inter_size,
                                             routed_tokens, quant_scales, base_activation_type);

    // TODO: support lora in the future
    ::tensorrt_llm::kernels::LoraParams lora_params{};
    // HACK Define default values for parameters we don't have good values for
    int64_t const unpadded_hidden_size_ml = hidden_size;  // Assume no padding by default
    bool const use_lora_ml = false;                       // No lora support yet
#ifdef USING_OSS_CUTLASS_MOE_GEMM
    mKernelRunner->runMoe(
        input.data_ptr(), input_sf.has_value() ? input_sf.value().data_ptr() : nullptr,
        swizzled_input_sf, reinterpret_cast<int const*>(token_selected_experts.data_ptr()),
        token_final_scales.has_value()
            ? reinterpret_cast<float const*>(token_final_scales.value().data_ptr())
            : nullptr,
        fc1_expert_weights.data_ptr(),
        fc1_expert_biases.has_value() ? fc1_expert_biases.value().data_ptr() : nullptr,
        activation_params, fc2_expert_weights.data_ptr(),
        fc2_expert_biases.has_value() ? fc2_expert_biases.value().data_ptr() : nullptr,
        quant_params, num_rows, hidden_size, unpadded_hidden_size_ml, inter_size, num_experts_total,
        static_cast<int>(experts_per_token), static_cast<char*>(workspace_info.workspace_ptr),
        output.data_ptr(), static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config,
        enable_alltoall, use_lora_ml, lora_params, mUseDeepSeekFP8BlockScaling, mUseMxfp8ActScaling,
        min_latency_mode, min_latency_params, enable_pdl, stream);
#else
    mKernelRunner->runMoe(
        input.data_ptr(), input_sf.has_value() ? input_sf.value().data_ptr() : nullptr,
        swizzled_input_sf, reinterpret_cast<int const*>(token_selected_experts.data_ptr()),
        token_final_scales.has_value()
            ? reinterpret_cast<float const*>(token_final_scales.value().data_ptr())
            : nullptr,
        fc1_expert_weights.data_ptr(),
        fc1_expert_biases.has_value() ? fc1_expert_biases.value().data_ptr() : nullptr,
        activation_params, fc2_expert_weights.data_ptr(),
        fc2_expert_biases.has_value() ? fc2_expert_biases.value().data_ptr() : nullptr,
        quant_params, num_rows, hidden_size, unpadded_hidden_size_ml, inter_size, num_experts_total,
        static_cast<int>(experts_per_token), static_cast<char*>(workspace_info.workspace_ptr),
        output.data_ptr(), static_cast<int*>(workspace_info.src_to_dest_map), parallelism_config,
        false, use_lora_ml, lora_params, mUseDeepSeekFP8BlockScaling, mUseMxfp8ActScaling,
        min_latency_mode, min_latency_params, enable_pdl, stream);
#endif
  }

  int64_t getTacticNum() {
    std::lock_guard<std::mutex> lock(mMutex);
    return mAllProfiles.size();
  }

  void runGemmProfile(TensorView input, TensorView fc1_expert_weights,
                      Optional<TensorView> fc1_expert_biases, TensorView fc2_expert_weights,
                      Optional<TensorView> fc2_expert_biases, int64_t top_k, int64_t tp_size,
                      int64_t tp_rank, int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
                      int64_t cluster_rank, bool enable_alltoall, bool min_latency_mode,
                      int64_t gemm_idx, int64_t profile_id, bool do_preparation, bool enable_pdl,
                      ActivationType activation_type) {
    std::lock_guard<std::mutex> lock(mMutex);
    ffi::CUDADeviceGuard device_guard(input.device().device_id);

    // TODO: support profiling under fp8 block scaling in the future
    if (mUseDeepSeekFP8BlockScaling) {
      return;
    }

    int64_t num_rows = input.size(0);
    int64_t hidden_size = fc2_expert_weights.size(1);
    int64_t inter_size = fc2_expert_weights.size(2) * mInnerDimMultiplier;
    int64_t group_size_ =
        isInt4Quant() ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size
                      : -1;
    int64_t group_size =
        (isWFP4A16Quant() || isWMxfp4AFp8HummingQuant())
            ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size
            : group_size_;
    int const num_experts = static_cast<int>(fc2_expert_weights.size(0) * ep_size);

    // Get specific profile configs according to the profile_id.
    // Fallback tactic is set to be 0
    // TODO: use the best tactic id found offline for a better default inference perf
    auto profile = profile_id == -1 ? mAllProfiles.front() : mAllProfiles[profile_id];

    auto stream = get_stream(input.device());

    auto const* expert_weights_ptr =
        (gemm_idx == 1) ? fc1_expert_weights.data_ptr() : fc2_expert_weights.data_ptr();

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
      int64_t const unpadded_hidden_size_profiler = hidden_size;  // HACK no padding by default
#ifdef USING_OSS_CUTLASS_MOE_GEMM
      mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
                      DtypeUtils::dataType(activation_dtype), DtypeUtils::dataType(mWeightDtype),
                      DtypeUtils::dataType(mOutputDtype), num_experts, static_cast<int>(top_k),
                      hidden_size, unpadded_hidden_size_profiler, inter_size, group_size,
                      activation_type, USE_BIAS, USE_LORA, min_latency_mode,
                      /*need_weights*/ false, parallelism_config, enable_alltoall,
                      mUseMxfp8ActScaling, mSm90Wfp4Afp8Mode);
#else
      mProfiler->init(*mKernelRunner.get(), mProfiler->mGemmToProfile,
                      DtypeUtils::dataType(activation_dtype), DtypeUtils::dataType(mWeightDtype),
                      DtypeUtils::dataType(mOutputDtype), num_experts, static_cast<int>(top_k),
                      hidden_size, unpadded_hidden_size_profiler, inter_size, group_size,
                      activation_type, USE_BIAS, USE_LORA, min_latency_mode,
                      /*need_weights*/ false, parallelism_config,
                      /*enable_alltoall*/ false,
                      /*use_mxfp8_act_scaling*/ false, mSm90Wfp4Afp8Mode);
#endif

      size_t profile_workspace_size = mProfiler->getWorkspaceSize(num_rows);
      mProfileWorkspace =
          alloc_tensor({static_cast<int64_t>(profile_workspace_size)}, dl_int8, input.device());

      mProfiler->prepare(num_rows, static_cast<char*>(mProfileWorkspace.data_ptr()),
                         expert_weights_ptr, enable_pdl, stream);
    }

    // Profile specific tactic. Assuming at least one preparation phase has been executed already.
    mProfiler->runProfiler(num_rows, profile, static_cast<char*>(mProfileWorkspace.data_ptr()),
                           expert_weights_ptr, enable_pdl, stream);
  }

  const char* kind() const final { return "fused_moe_runner"; }
  Optional<Function> GetFunction(const tvm::ffi::String& name) final {
    if (name == "run_gemm_profile") {
      return Function::FromTyped(
          [this](TensorView input, TensorView fc1_expert_weights,
                 Optional<TensorView> fc1_expert_biases, TensorView fc2_expert_weights,
                 Optional<TensorView> fc2_expert_biases, int64_t top_k, int64_t tp_size,
                 int64_t tp_rank, int64_t ep_size, int64_t ep_rank, int64_t cluster_size,
                 int64_t cluster_rank, bool enable_alltoall, bool min_latency_mode,
                 int64_t gemm_idx, int64_t profile_id, bool do_preparation, bool enable_pdl,
                 int64_t activation_type) {
            runGemmProfile(input, fc1_expert_weights, fc1_expert_biases, fc2_expert_weights,
                           fc2_expert_biases, top_k, tp_size, tp_rank, ep_size, ep_rank,
                           cluster_size, cluster_rank, enable_alltoall, min_latency_mode, gemm_idx,
                           profile_id, do_preparation, enable_pdl,
                           static_cast<ActivationType>(activation_type));
          });
    } else if (name == "get_tactic_num") {
      return Function::FromTyped([this]() -> int64_t { return getTacticNum(); });
    } else if (name == "get_gemm1_tactic_count") {
      return Function::FromTyped([this]() -> int64_t { return mGemm1TacticCount; });
    } else if (name == "get_gemm2_tactic_count") {
      return Function::FromTyped([this]() -> int64_t { return mGemm2TacticCount; });
    } else if (name == "get_tactic_occupancy") {
      // Returns the max active blocks per SM for the given tactic index.
      // Returns 0 if the tactic is not supported on the current device (e.g., SM89 tile
      // configs with insufficient shared memory when running on SM120 Blackwell).
      return Function::FromTyped([this](int64_t tactic_id) -> int64_t {
        std::lock_guard<std::mutex> lock(mMutex);
        if (tactic_id < 0 || tactic_id >= static_cast<int64_t>(mAllProfiles.size())) {
          return 0;
        }
        return static_cast<int64_t>(
            mKernelRunner->queryOccupancyForConfig(mAllProfiles[tactic_id]));
      });
    } else if (name == "get_valid_tactics_for_shape") {
      return Function::FromTyped(
          [this](int64_t stage, int64_t gemm_n, int64_t gemm_k) -> Array<int64_t> {
            std::lock_guard<std::mutex> lock(mMutex);
            return getValidTacticsForShape(stage, gemm_n, gemm_k);
          });
    } else if (name == "run_moe") {
      return Function::FromTyped(
          [this](TensorView output, TensorView input, TensorView token_selected_experts,
                 Optional<TensorView> token_final_scales, TensorView fc1_expert_weights,
                 Optional<TensorView> fc1_expert_biases, TensorView fc2_expert_weights,
                 Optional<TensorView> fc2_expert_biases, Optional<Array<Tensor>> quant_scales,
                 Optional<TensorView> input_sf, Optional<TensorView> swiglu_alpha,
                 Optional<TensorView> swiglu_beta, Optional<TensorView> swiglu_limit,
                 bool swizzled_input_sf, int64_t tp_size, int64_t tp_rank, int64_t ep_size,
                 int64_t ep_rank, int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
                 bool min_latency_mode, Optional<Array<int64_t>> profile_ids, bool enable_pdl,
                 int64_t base_activation_type, Optional<TensorView> workspace_buffer) {
            runMoe(output, input, token_selected_experts, token_final_scales, fc1_expert_weights,
                   fc1_expert_biases, fc2_expert_weights, fc2_expert_biases, quant_scales, input_sf,
                   swiglu_alpha, swiglu_beta, swiglu_limit, swizzled_input_sf, tp_size, tp_rank,
                   ep_size, ep_rank, cluster_size, cluster_rank, enable_alltoall, min_latency_mode,
                   profile_ids, enable_pdl, static_cast<ActivationType>(base_activation_type),
                   workspace_buffer);
          });
    } else if (name == "run_moe_min_latency") {
      return Function::FromTyped(
          [this](TensorView output, TensorView input, TensorView token_selected_experts,
                 Optional<TensorView> token_final_scales, TensorView fc1_expert_weights,
                 Optional<TensorView> fc1_expert_biases, TensorView fc2_expert_weights,
                 Optional<TensorView> fc2_expert_biases, Optional<Array<Tensor>> quant_scales,
                 Optional<TensorView> input_sf, Optional<TensorView> swiglu_alpha,
                 Optional<TensorView> swiglu_beta, Optional<TensorView> swiglu_limit,
                 bool swizzled_input_sf, TensorView num_active_experts_per_node,
                 TensorView experts_to_token_score, TensorView active_expert_global_ids,
                 int64_t tp_size, int64_t tp_rank, int64_t ep_size, int64_t ep_rank,
                 int64_t cluster_size, int64_t cluster_rank, bool enable_alltoall,
                 bool min_latency_mode, Optional<Array<int64_t>> profile_ids, bool enable_pdl,
                 int64_t base_activation_type, Optional<TensorView> workspace_buffer) {
            runMoeMinLantency(output, input, token_selected_experts, token_final_scales,
                              fc1_expert_weights, fc1_expert_biases, fc2_expert_weights,
                              fc2_expert_biases, quant_scales, input_sf, swiglu_alpha, swiglu_beta,
                              swiglu_limit, swizzled_input_sf, num_active_experts_per_node,
                              experts_to_token_score, active_expert_global_ids, tp_size, tp_rank,
                              ep_size, ep_rank, cluster_size, cluster_rank, enable_alltoall,
                              min_latency_mode, profile_ids, enable_pdl,
                              static_cast<ActivationType>(base_activation_type), workspace_buffer);
          });
    } else if (name == "get_workspace_size") {
      return Function::FromTyped([this](int64_t num_rows, int64_t hidden_size, int64_t inter_size,
                                        int64_t num_experts_total, int64_t experts_per_token,
                                        int64_t tp_size, int64_t tp_rank, int64_t ep_size,
                                        int64_t ep_rank, bool min_latency_mode,
                                        int64_t base_activation_type) -> int64_t {
        std::lock_guard<std::mutex> lock(mMutex);
        auto parallelism_config = kernels::MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank);
        auto [moe_ws, src_map] = getWorkspaceSizes(
            num_rows, hidden_size, inter_size, static_cast<int>(num_experts_total),
            static_cast<int>(experts_per_token), static_cast<ActivationType>(base_activation_type),
            parallelism_config, min_latency_mode);
        std::vector<size_t> ws{moe_ws, src_map};
        return static_cast<int64_t>(common::calculateTotalWorkspaceSize(ws.data(), ws.size()));
      });
    } else {
      return Function(nullptr);
    }
  }

 private:
  struct WorkspaceInfo {
    Tensor workspace{};     // owns allocation when caller does not provide a buffer
    void* workspace_ptr{};  // raw pointer valid in both cases
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
  bool mUseWfp4Afp8Humming = false;
  bool mUsePackedWeights = false;
  bool mUseFusedFinalize = true;
  kernels::Sm90Wfp4Afp8ScaleMode mSm90Wfp4Afp8Mode = kernels::Sm90Wfp4Afp8ScaleMode::kDisabled;

  using Profile = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;
  std::vector<Profile> mAllProfiles;
  int64_t mGemm1TacticCount{0};
  int64_t mGemm2TacticCount{0};

  bool isProfileShapeSupported(Profile const& profile, int64_t gemm_n, int64_t gemm_k) const {
    int64_t tile_m = 0;
    int64_t tile_n = 0;
    int64_t tile_k = 0;
    if (profile.sm_version == 90) {
      auto const [m, n, k] =
          tensorrt_llm::cutlass_extensions::enum_to_shape_tuple(profile.tile_config_sm90);
      tile_m = m;
      tile_n = n;
      tile_k = k;
    } else if (profile.sm_version == 100) {
      auto const [m, n, k] =
          tensorrt_llm::cutlass_extensions::enum_to_shape_tuple(profile.tile_config_sm100);
      tile_m = m;
      tile_n = n;
      tile_k = k;
    } else if (profile.sm_version == 120) {
      auto const [m, n, k] =
          tensorrt_llm::cutlass_extensions::enum_to_shape_tuple(profile.tile_config_sm120);
      tile_m = m;
      tile_n = n;
      tile_k = k;
    }

    if (tile_m <= 0 || tile_n <= 0 || tile_k <= 0 || gemm_n <= 0 || gemm_k <= 0) {
      return false;
    }
    if (gemm_k < tile_k || gemm_k % tile_k != 0) {
      return false;
    }
    if (gemm_n < tile_n) {
      return false;
    }
    if (mUseW4GroupScaling && gemm_n % tile_m != 0) {
      return false;
    }
    bool const is_single_warpgroup =
        profile.mainloop_schedule ==
            tensorrt_llm::cutlass_extensions::MainloopScheduleType::SINGLE_WARPGROUP_PREFILL ||
        profile.mainloop_schedule ==
            tensorrt_llm::cutlass_extensions::MainloopScheduleType::SINGLE_WARPGROUP_ROLLING;
    if (is_single_warpgroup) {
      if (!mUseWfp4Afp8Humming || profile.sm_version != 90 || tile_m != 128 || tile_k != 128 ||
          (tile_n != 8 && tile_n != 16 && tile_n != 32 && tile_n != 40) ||
          profile.cluster_shape !=
              tensorrt_llm::cutlass_extensions::ClusterShape::ClusterShape_1x1x1 ||
          gemm_n % 128 != 0) {
        return false;
      }
      if (profile.mainloop_schedule ==
          tensorrt_llm::cutlass_extensions::MainloopScheduleType::SINGLE_WARPGROUP_PREFILL) {
        return gemm_k <= 384;
      }
      return gemm_k > 384;
    }
    if (isWFP4A16Quant()) {
      if (tile_k == 256 && ((tile_m == 128 && tile_n == 256) || (tile_m == 256 && tile_n == 128))) {
        return false;
      }
      if (tile_k == 512 && tile_n >= 128) {
        return false;
      }
    }
    return true;
  }

  Array<int64_t> getValidTacticsForShape(int64_t stage, int64_t gemm_n, int64_t gemm_k) const {
    int64_t begin = 0;
    int64_t end = static_cast<int64_t>(mAllProfiles.size());
    if (stage == 1) {
      end = mGemm1TacticCount;
    } else if (stage == 2) {
      begin = mGemm1TacticCount;
      end = mGemm1TacticCount + mGemm2TacticCount;
    }

    int64_t const total = static_cast<int64_t>(mAllProfiles.size());
    if (begin < 0) {
      begin = 0;
    }
    if (begin > total) {
      begin = total;
    }
    if (end < begin) {
      end = begin;
    }
    if (end > total) {
      end = total;
    }

    Array<int64_t> tactics;
    for (int64_t tactic_id = begin; tactic_id < end; ++tactic_id) {
      if (!mUseW4GroupScaling || isProfileShapeSupported(mAllProfiles[tactic_id], gemm_n, gemm_k)) {
        tactics.push_back(tactic_id);
      }
    }
    return tactics;
  }

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
    // Default GEMM2 profile should come from the GEMM2 subrange if present
    auto best_gemm2_profile =
        (mGemm2TacticCount > 0 && mAllProfiles.size() > static_cast<size_t>(mGemm1TacticCount))
            ? mAllProfiles.at(mGemm1TacticCount)
            : mAllProfiles.front();
    if (profile_ids.has_value()) {
      TVM_FFI_ICHECK_EQ(profile_ids.value().size(), 2) << "Expecting 2 profile ids";
      // GEMM1 index: accept absolute index and raise error if out of GEMM1 range
      auto id1 = profile_ids.value()[0];
      if (id1 != -1) {
        TVM_FFI_ICHECK(id1 >= 0 && id1 < mGemm1TacticCount) << "Invalid gemm1 profile id: " << id1;
        best_gemm1_profile = mAllProfiles.at(id1);
      }

      // GEMM2 profiles use absolute indices in the combined profile array.
      auto id2 = profile_ids.value()[1];
      if (id2 != -1) {
        TVM_FFI_ICHECK(id2 >= mGemm1TacticCount && id2 < mGemm1TacticCount + mGemm2TacticCount)
            << "Invalid gemm2 profile id: " << id2;
        best_gemm2_profile = mAllProfiles.at(id2);
      }
    }
    mKernelRunner->setTactic(best_gemm1_profile, best_gemm2_profile);
  }

  // Returns {moe_workspace_size, src_to_dest_map_size}.
  std::pair<size_t, size_t> getWorkspaceSizes(int64_t num_rows, int64_t hidden_size,
                                              int64_t inter_size, int num_experts,
                                              int experts_per_token, ActivationType activation_type,
                                              kernels::MOEParallelismConfig parallelism_config,
                                              bool min_latency_mode) {
    size_t moe_ws = mKernelRunner->getWorkspaceSize(
        num_rows, hidden_size, inter_size, num_experts, experts_per_token, activation_type,
        parallelism_config, /*use_lora=*/false, mUseDeepSeekFP8BlockScaling, mUseMxfp8ActScaling,
        min_latency_mode, mUseW4GroupScaling);
    size_t src_map =
        static_cast<size_t>(experts_per_token) * static_cast<size_t>(num_rows) * sizeof(int);
    return {moe_ws, src_map};
  }

  WorkspaceInfo getWorkspaceInfo(int64_t num_rows, int64_t hidden_size, int64_t inter_size,
                                 int num_experts, int experts_per_token,
                                 ActivationType activation_type,
                                 kernels::MOEParallelismConfig parallelismConfig,
                                 bool min_latency_mode, DLDevice expected_device,
                                 Optional<TensorView> provided_workspace = Optional<TensorView>{}) {
    auto [moe_workspace_size, src_to_dest_map_size] =
        getWorkspaceSizes(num_rows, hidden_size, inter_size, num_experts, experts_per_token,
                          activation_type, parallelismConfig, min_latency_mode);

    std::vector<size_t> workspaces{moe_workspace_size, src_to_dest_map_size};
    size_t total_workspace_size =
        common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());

    WorkspaceInfo info{};
    if (provided_workspace.has_value()) {
      auto const& ws = provided_workspace.value();
      // dtype: int8 or uint8 (1 byte per element so size(0) == nbytes)
      TVM_FFI_ICHECK(ws.dtype() == dl_int8 || ws.dtype() == dl_uint8)
          << "workspace_buffer dtype must be int8 or uint8, got " << DLDataTypeToString(ws.dtype());
      // must be 1-D
      TVM_FFI_ICHECK_EQ(ws.ndim(), 1) << "workspace_buffer must be 1-D, got ndim=" << ws.ndim();
      // must be on CUDA, same device as the input
      TVM_FFI_ICHECK_EQ(ws.device().device_type, kDLCUDA)
          << "workspace_buffer must be a CUDA tensor";
      TVM_FFI_ICHECK_EQ(ws.device().device_id, expected_device.device_id)
          << "workspace_buffer device (" << ws.device().device_id
          << ") does not match input device (" << expected_device.device_id << ")";
      // sufficient bytes
      TVM_FFI_ICHECK(static_cast<size_t>(ws.size(0)) >= total_workspace_size)
          << "workspace_buffer too small: need " << total_workspace_size << " bytes, got "
          << ws.size(0);
      // must be contiguous (non-contiguous strides corrupt the workspace layout)
      TVM_FFI_ICHECK(ws.IsContiguous()) << "workspace_buffer must be contiguous";
      // 128-byte alignment required by nextWorkspacePtr
      TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(ws.data_ptr()) % common::kCudaMemAlign == 0)
          << "workspace_buffer data pointer must be " << common::kCudaMemAlign
          << "-byte aligned (torch.empty on CUDA satisfies this by default)";
      info.workspace_ptr = ws.data_ptr();
    } else {
      info.workspace =
          alloc_tensor({static_cast<int64_t>(total_workspace_size)}, dl_int8, expected_device);
      info.workspace_ptr = info.workspace.data_ptr();
    }
    info.src_to_dest_map =
        common::nextWorkspacePtr(static_cast<int8_t*>(info.workspace_ptr), moe_workspace_size);

    return info;
  }

  kernels::QuantParams getQuantParams(
      int64_t num_experts_on_rank, int64_t hidden_size, int64_t inter_size, int64_t routed_tokens,
      Optional<Array<Tensor>> quant_scales,
      ActivationType base_activation_type = ActivationType::Swiglu) const {
    if (isWMxfp8AMxfp8Quant()) {
#ifdef USING_OSS_CUTLASS_MOE_GEMM
      TVM_FFI_ICHECK(quant_scales.has_value())
          << "Expecting quant scales for MXFP8xMXFP8 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 4)
          << "Expecting 4 quant scales for MXFP8xMXFP8 quantization";

      TensorView fc1_weight_block = quant_scales.value()[0];
      TensorView fc1_global = quant_scales.value()[1];
      TensorView fc2_weight_block = quant_scales.value()[2];
      TensorView fc2_global = quant_scales.value()[3];

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
      int const fc1_n_mult = isGatedActivation(base_activation_type) ? 2 : 1;
      TVM_FFI_ICHECK(
          fc1_weight_block.size(0) == num_experts_on_rank &&
          fc1_weight_block.size(1) ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) *
                  fc1_n_mult &&
          fc1_weight_block.size(2) * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc1 weight block size must be (num_experts_on_rank, inter_size"
          << (fc1_n_mult == 2 ? " * 2" : "") << ", hidden_size // 4 // block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc1_global.size(0), num_experts_on_rank)
          << "fc1 global size must be (num_experts_on_rank,)";
      TVM_FFI_ICHECK(
          fc2_weight_block.size(0) == num_experts_on_rank &&
          fc2_weight_block.size(1) ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) &&
          fc2_weight_block.size(2) * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
             "block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc2_global.size(0), num_experts_on_rank)
          << "fc2 global size must be (num_experts_on_rank,)";

      return kernels::QuantParams::MXFP8MXFP8(
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
          static_cast<float const*>(fc1_global.data_ptr()),
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()),
          static_cast<float const*>(fc2_global.data_ptr()));
#else
      TVM_FFI_ICHECK(false)
          << "MXFP8 x MXFP8 quantization is not supported in OSS Cutlass Moe Gemm";
#endif
    } else if (isFp8Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value()) << "Expecting quant scales for fp8 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 4)
          << "Expecting 4 quant scales for fp8 quantization";

      auto const fc1_dequant = quant_scales.value()[0];
      auto const fc2_quant = quant_scales.value()[1];
      auto const fc2_dequant = quant_scales.value()[2];
      auto const fc1_input_dequant = quant_scales.value()[3];

      TVM_FFI_ICHECK(fc1_dequant.GetDLTensorPtr() != nullptr)
          << "Expecting fc1_dequant to be non null";
      TVM_FFI_ICHECK(fc2_quant.GetDLTensorPtr() != nullptr) << "Expecting fc2_quant to be non null";
      TVM_FFI_ICHECK(fc2_dequant.GetDLTensorPtr() != nullptr)
          << "Expecting fc2_dequant to be non null";
      TVM_FFI_ICHECK(fc1_input_dequant.GetDLTensorPtr() != nullptr)
          << "Expecting fc1_input_dequant to be non null";

      // Check types
      CHECK_INPUT_TYPE(fc1_dequant, dl_float32);
      CHECK_INPUT_TYPE(fc2_quant, dl_float32);
      CHECK_INPUT_TYPE(fc2_dequant, dl_float32);
      CHECK_INPUT_TYPE(fc1_input_dequant, dl_float32);
      // Check ranks
      CHECK_DIM(1, fc1_dequant);
      TVM_FFI_ICHECK_LE(fc2_quant.ndim(), 1) << "fc2 quant must be a scalar or 1-D tensor";
      CHECK_DIM(1, fc2_dequant);
      CHECK_DIM(0, fc1_input_dequant);
      // Check shapes
      TVM_FFI_ICHECK_EQ(fc1_dequant.size(0), num_experts_on_rank)
          << "fc1 dequant size must be (num_experts_on_rank,)";
      TVM_FFI_ICHECK(fc2_quant.ndim() == 0 || fc2_quant.size(0) == num_experts_on_rank)
          << "fc2 quant must be scalar or (num_experts_on_rank,)";
      TVM_FFI_ICHECK_EQ(fc2_dequant.size(0), num_experts_on_rank)
          << "fc2 dequant size must be (num_experts_on_rank,)";

      return kernels::QuantParams::FP8(static_cast<float const*>(fc1_dequant.data_ptr()),
                                       static_cast<float const*>(fc2_quant.data_ptr()),
                                       static_cast<float const*>(fc2_dequant.data_ptr()),
                                       /* fp8 output quant scale */ nullptr,
                                       static_cast<float const*>(fc1_input_dequant.data_ptr()),
                                       fc2_quant.ndim() == 1);
    } else if (isWMxfp4AFp8Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value())
          << "Expecting quant scales for post-MMA MXFP4 x FP8 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 5)
          << "Expecting 5 quant scales for post-MMA MXFP4 x FP8 quantization";

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
      TVM_FFI_ICHECK_LE(fc2_act_global.ndim(), 1)
          << "fc2 act global must be a scalar or 1-D tensor";
      CHECK_DIM(3, fc2_weight_block);
      CHECK_DIM(1, fc2_global);
      // Check shapes
      int const fc1_n_mult = isGatedActivation(base_activation_type) ? 2 : 1;
      TVM_FFI_ICHECK(
          fc1_weight_block.size(0) == num_experts_on_rank &&
          fc1_weight_block.size(1) ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) *
                  fc1_n_mult &&
          fc1_weight_block.size(2) * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc1 weight block size must be (num_experts_on_rank, inter_size"
          << (fc1_n_mult == 2 ? " * 2" : "") << ", hidden_size // 4 // block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc1_global.size(0), num_experts_on_rank)
          << "fc1 global size must be (num_experts_on_rank,)";
      TVM_FFI_ICHECK(fc2_act_global.ndim() == 0 || fc2_act_global.size(0) == num_experts_on_rank)
          << "fc2 act global must be scalar or (num_experts_on_rank,)";
      TVM_FFI_ICHECK(
          fc2_weight_block.size(0) == num_experts_on_rank &&
          fc2_weight_block.size(1) ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) &&
          fc2_weight_block.size(2) * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
             "block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc2_global.size(0), num_experts_on_rank)
          << "fc2 global size must be (num_experts_on_rank,)";

      return kernels::QuantParams::FP8MXFP4(
          nullptr,
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
          static_cast<float const*>(fc1_global.data_ptr()),
          static_cast<float const*>(fc2_act_global.data_ptr()),
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()),
          static_cast<float const*>(fc2_global.data_ptr()), false, fc2_act_global.ndim() == 1);
    } else if (isWMxfp4AFp8HummingQuant()) {
      TVM_FFI_ICHECK(quant_scales.has_value())
          << "Expecting quant scales for Humming-style MXFP4 x FP8 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 5)
          << "Expecting 5 quant scales for Humming-style MXFP4 x FP8 quantization";

      auto const fc1_weight_block = quant_scales.value()[0];
      auto const fc1_token_scale = quant_scales.value()[1];
      auto const fc2_act_global = quant_scales.value()[2];
      auto const fc2_weight_block = quant_scales.value()[3];
      auto const fc2_token_scale = quant_scales.value()[4];

      CHECK_INPUT_TYPE(fc1_weight_block, dl_int32);
      CHECK_INPUT_TYPE(fc1_token_scale, dl_float32);
      CHECK_INPUT_TYPE(fc2_act_global, dl_float32);
      CHECK_INPUT_TYPE(fc2_weight_block, dl_int32);
      CHECK_INPUT_TYPE(fc2_token_scale, dl_float32);
      CHECK_DIM(5, fc1_weight_block);
      CHECK_DIM(1, fc1_token_scale);
      TVM_FFI_ICHECK_LE(fc2_act_global.ndim(), 1)
          << "fc2 act global must be a scalar or 1-D tensor";
      CHECK_DIM(5, fc2_weight_block);
      CHECK_DIM(1, fc2_token_scale);
      int const fc1_n_mult = isGatedActivation(base_activation_type) ? 2 : 1;
      TVM_FFI_ICHECK(fc1_weight_block.size(0) == num_experts_on_rank &&
                     fc1_weight_block.size(1) * 64 == inter_size * fc1_n_mult &&
                     fc1_weight_block.size(2) * 128 == hidden_size &&
                     fc1_weight_block.size(3) == 16 && fc1_weight_block.size(4) == 4)
          << "fc1 Humming-style folded weight scale must be "
             "(num_experts_on_rank, inter_size"
          << (fc1_n_mult == 2 ? " * 2" : "") << " / 64, hidden_size / 128, 16, 4)";
      TVM_FFI_ICHECK_EQ(fc1_token_scale.size(0), routed_tokens)
          << "fc1 token scale must have one element per routed token";
      TVM_FFI_ICHECK(fc2_act_global.ndim() == 0 || fc2_act_global.size(0) == num_experts_on_rank)
          << "fc2 act global must be scalar or (num_experts_on_rank,)";
      TVM_FFI_ICHECK(fc2_weight_block.size(0) == num_experts_on_rank &&
                     fc2_weight_block.size(1) * 64 == hidden_size &&
                     fc2_weight_block.size(2) * 128 == inter_size &&
                     fc2_weight_block.size(3) == 16 && fc2_weight_block.size(4) == 4)
          << "fc2 Humming-style folded weight scale must be "
             "(num_experts_on_rank, hidden_size / 64, inter_size / 128, 16, 4)";
      TVM_FFI_ICHECK_EQ(fc2_token_scale.size(0), routed_tokens)
          << "fc2 token scale must have one element per routed token";

      return kernels::QuantParams::FP8MXFP4(
          nullptr,
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
          static_cast<float const*>(fc1_token_scale.data_ptr()), nullptr,
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()),
          static_cast<float const*>(fc2_token_scale.data_ptr()), false, false);
    } else if (isWMxfp4AMxfp8Quant()) {
#ifdef USING_OSS_CUTLASS_MOE_GEMM
      TVM_FFI_ICHECK(quant_scales.has_value())
          << "Expecting quant scales for W4A8_MXFP4_MXFP8 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 4)
          << "Expecting 4 quant scales for W4A8_MXFP4_MXFP8 quantization";

      auto const& fc1_weight_block = quant_scales.value()[0];
      auto const& fc1_global = quant_scales.value()[1];
      auto const& fc2_weight_block = quant_scales.value()[2];
      auto const& fc2_global = quant_scales.value()[3];

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
      int const fc1_n_mult = isGatedActivation(base_activation_type) ? 2 : 1;
      TVM_FFI_ICHECK(
          fc1_weight_block.size(0) == num_experts_on_rank &&
          fc1_weight_block.size(1) ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) *
                  fc1_n_mult &&
          fc1_weight_block.size(2) * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc1 weight block size must be (num_experts_on_rank, inter_size"
          << (fc1_n_mult == 2 ? " * 2" : "") << ", hidden_size // 4 // block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc1_global.size(0), num_experts_on_rank)
          << "fc1 global size must be (num_experts_on_rank,)";
      TVM_FFI_ICHECK(
          fc2_weight_block.size(0) == num_experts_on_rank &&
          fc2_weight_block.size(1) ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX) &&
          fc2_weight_block.size(2) * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX))
          << "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
             "block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc2_global.size(0), num_experts_on_rank)
          << "fc2 global size must be (num_experts_on_rank,)";

      return kernels::QuantParams::MXFP8MXFP4(
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
          static_cast<float const*>(fc1_global.data_ptr()),
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()),
          static_cast<float const*>(fc2_global.data_ptr()));
#else
      TVM_FFI_ICHECK(false)
          << "MXFP8 x MXFP4 quantization is not supported in OSS Cutlass Moe Gemm";
#endif
    }

    else if (isNvfp4Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value()) << "Expecting quant scales for nvfp4 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 6)
          << "Expecting 6 quant scales for nvfp4 quantization";

      auto const& fc1_act_global = quant_scales.value()[0];
      auto const& fc1_weight_block = quant_scales.value()[1];
      auto const& fc1_global = quant_scales.value()[2];
      auto const& fc2_act_global = quant_scales.value()[3];
      auto const& fc2_weight_block = quant_scales.value()[4];
      auto const& fc2_global = quant_scales.value()[5];

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
      TVM_FFI_ICHECK_LE(fc1_act_global.ndim(), 1)
          << "fc1 act global must be a scalar or 1-D tensor";
      CHECK_DIM(3, fc1_weight_block);
      CHECK_DIM(1, fc1_global);
      TVM_FFI_ICHECK_LE(fc2_act_global.ndim(), 1)
          << "fc2 act global must be a scalar or 1-D tensor";
      CHECK_DIM(3, fc2_weight_block);
      CHECK_DIM(1, fc2_global);
      // Check shapes
      TVM_FFI_ICHECK(fc1_act_global.ndim() == 0 || fc1_act_global.size(0) == num_experts_on_rank)
          << "fc1 act global must be scalar or (num_experts_on_rank,)";
      if (isGatedActivation(base_activation_type)) {
        TVM_FFI_ICHECK(
            fc1_weight_block.size(0) == num_experts_on_rank &&
            fc1_weight_block.size(1) ==
                TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4) *
                    2 &&
            fc1_weight_block.size(2) * FP8_PER_INT32 *
                    TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize ==
                TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4))
            << "fc1 weight block size must be (num_experts_on_rank, inter_size * 2, hidden_size // "
               "4 "
               "// block_scale_vector_size)";
      } else {
        TVM_FFI_ICHECK(
            fc1_weight_block.size(0) == num_experts_on_rank &&
            fc1_weight_block.size(1) ==
                TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4) &&
            fc1_weight_block.size(2) * FP8_PER_INT32 *
                    TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize ==
                TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                    hidden_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4))
            << "fc1 weight block size must be (num_experts_on_rank, inter_size, hidden_size // 4 "
               "// block_scale_vector_size)";
      }

      TVM_FFI_ICHECK_EQ(fc1_global.size(0), num_experts_on_rank)
          << "fc1 global size must be (num_experts_on_rank,)";
      TVM_FFI_ICHECK(fc2_act_global.ndim() == 0 || fc2_act_global.size(0) == num_experts_on_rank)
          << "fc2 act global must be scalar or (num_experts_on_rank,)";
      TVM_FFI_ICHECK(
          fc2_weight_block.size(0) == num_experts_on_rank &&
          fc2_weight_block.size(1) ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  hidden_size, TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4) &&
          fc2_weight_block.size(2) * FP8_PER_INT32 *
                  TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize ==
              TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
                  inter_size, TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4))
          << "fc2 weight block size must be (num_experts_on_rank, hidden_size, inter_size // 4 // "
             "block_scale_vector_size)";
      TVM_FFI_ICHECK_EQ(fc2_global.size(0), num_experts_on_rank)
          << "fc2 global size must be (num_experts_on_rank,)";

      return kernels::QuantParams::FP4(
          static_cast<float const*>(fc1_act_global.data_ptr()),
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc1_weight_block.data_ptr()),
          static_cast<float const*>(fc1_global.data_ptr()),
          static_cast<float const*>(fc2_act_global.data_ptr()),
          static_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(fc2_weight_block.data_ptr()),
          static_cast<float const*>(fc2_global.data_ptr()), fc1_act_global.ndim() == 1,
          fc2_act_global.ndim() == 1);
    } else if (mUseDeepSeekFP8BlockScaling) {
      auto const& fc1_scales = quant_scales.value()[0];
      auto const& fc2_scales = quant_scales.value()[1];
      return kernels::QuantParams::FP8BlockScaling(
          static_cast<float const*>(fc1_scales.data_ptr()),
          static_cast<float const*>(fc2_scales.data_ptr()));
    } else if (isWFP4A16Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value()) << "Expecting quant scales for W4 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 2)
          << "Expecting 2 quant scales for W4A16 quantization";

      auto const& fc1_weight_scales = quant_scales.value()[0];
      auto const& fc2_weight_scales = quant_scales.value()[1];
      int group_size = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size;
      return kernels::QuantParams::GroupWise(group_size,
                                             static_cast<void const*>(fc1_weight_scales.data_ptr()),
                                             static_cast<void const*>(fc2_weight_scales.data_ptr()),
                                             nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    } else if (isInt4Quant()) {
      TVM_FFI_ICHECK(quant_scales.has_value()) << "Expecting quant scales for INT4 quantization";
      TVM_FFI_ICHECK_EQ(quant_scales.value().size(), 8)
          << "Expecting 8 quant scales for INT4 quantization";
      auto const& fc1_weight_scales = quant_scales.value()[0];
      auto const& fc2_weight_scales = quant_scales.value()[1];
      auto const& fc1_act_scales = quant_scales.value()[2];
      auto const& fc2_act_scales = quant_scales.value()[3];
      auto const& fc1_weight_zeros = quant_scales.value()[4];
      auto const& fc2_weight_zeros = quant_scales.value()[5];
      auto const& fc1_alpha = quant_scales.value()[6];
      auto const& fc2_alpha = quant_scales.value()[7];
      if (fc1_act_scales.numel() > 0) {
        TVM_FFI_ICHECK_EQ(fc1_act_scales.numel(), hidden_size)
            << "INT4xFP8 FC1 prequant scale must be shared across experts with shape [hidden_size]";
      }
      if (fc2_act_scales.numel() > 0) {
        TVM_FFI_ICHECK_EQ(fc2_act_scales.numel(), inter_size)
            << "INT4xFP8 FC2 prequant scale must be shared across experts with shape [inter_size]";
      }
      int group_size = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size;
      return kernels::QuantParams::GroupWise(
          group_size, static_cast<void const*>(fc1_weight_scales.data_ptr()),
          static_cast<void const*>(fc2_weight_scales.data_ptr()),
          static_cast<void const*>(fc1_act_scales.numel() > 0 ? fc1_act_scales.data_ptr()
                                                              : nullptr),
          static_cast<void const*>(fc2_act_scales.numel() > 0 ? fc2_act_scales.data_ptr()
                                                              : nullptr),
          static_cast<void const*>(fc1_weight_zeros.numel() > 0 ? fc1_weight_zeros.data_ptr()
                                                                : nullptr),
          static_cast<void const*>(fc2_weight_zeros.numel() > 0 ? fc2_weight_zeros.data_ptr()
                                                                : nullptr),
          static_cast<float const*>(fc1_alpha.numel() > 0 ? fc1_alpha.data_ptr() : nullptr),
          static_cast<float const*>(fc2_alpha.numel() > 0 ? fc2_alpha.data_ptr() : nullptr));
    } else {
      return kernels::QuantParams{};
    }
  }

  bool isFp8Quant() const {
    return !mUseDeepSeekFP8BlockScaling && mActivationDtype == dl_float8_e4m3fn &&
           mWeightDtype == dl_float8_e4m3fn && !mUseMxfp8ActScaling;
  }

  bool isWMxfp8AMxfp8Quant() const {
    return !mUseDeepSeekFP8BlockScaling && mActivationDtype == dl_float8_e4m3fn &&
           mWeightDtype == dl_float8_e4m3fn && mUseMxfp8ActScaling;
  }

  bool isMxfp8ActScalingQuant() const { return isWMxfp8AMxfp8Quant() || isWMxfp4AMxfp8Quant(); }

  bool isNvfp4Quant() const {
    return mWeightDtype == dl_int64 &&
           mActivationDtype != dl_float8_e4m3fn;  // FP8 activation does not use FP4
  }

  bool isWFP4A16Quant() const {
    return mUseW4GroupScaling &&
           (mActivationDtype == dl_float16 || mActivationDtype == dl_bfloat16) &&
           mWeightDtype == dl_uint8 && !mUsePackedWeights && !mUseWfp4Afp8Humming;
  }

  bool isInt4Quant() const { return mWeightDtype == dl_uint8 && mUsePackedWeights; }

  bool isW4AFp8Quant() const { return mActivationDtype == dl_float8_e4m3fn && isInt4Quant(); }

  bool isWMxfp4AFp8HummingQuant() const {
    bool const supported_activation =
        mActivationDtype == dl_float16 || mActivationDtype == dl_bfloat16;
    return mUseWfp4Afp8Humming && mUseW4GroupScaling && supported_activation &&
           mWeightDtype == dl_uint8 && !mUsePackedWeights && !mUseMxfp8ActScaling;
  }

  bool isWMxfp4AFp8Quant() const {
    return mActivationDtype == dl_float8_e4m3fn && mWeightDtype == dl_int64 && !mUseMxfp8ActScaling;
  }

  bool isWMxfp4AMxfp8Quant() const {
    return mActivationDtype == dl_float8_e4m3fn && mWeightDtype == dl_int64 && mUseMxfp8ActScaling;
  }
};

tvm::ffi::Module init(DLDataType activation_dtype, DLDataType weight_dtype, DLDataType output_dtype,
                      bool use_deepseek_fp8_block_scale, bool use_w4_group_scaling,
                      bool use_mxfp8_act_scaling, bool use_packed_weights, bool use_fused_finalize,
                      bool use_wfp4afp8_humming) {
  auto ptr = tvm::ffi::make_object<FusedMoeRunner>(
      activation_dtype, weight_dtype, output_dtype, use_deepseek_fp8_block_scale,
      use_w4_group_scaling, use_mxfp8_act_scaling, use_packed_weights, use_fused_finalize,
      use_wfp4afp8_humming);
  return tvm::ffi::Module(ptr);
}

// Interleave a 4-bit packed weight tensor into the layout required by the
// SM90 mixed-input MoE GEMM. Expected input shape (num_experts, n,
// k / 2) uint8 on CUDA. Writes into an output tensor of the same shape.
// quant_type: 0 for INT4 (W4A8), 1 for FP4 (W4A16 / MXFP4 BF16),
// 2 for FP4 consumed by FP8/Humming-style pre-MMA scaling.
void interleave_moe_weights_for_sm90_mixed_gemm(TensorView weight, TensorView weight_interleaved,
                                                int64_t quant_type) {
  CHECK_INPUT_TYPE(weight, dl_uint8);
  CHECK_INPUT_TYPE(weight_interleaved, dl_uint8);
  CHECK_CONTIGUOUS(weight);
  CHECK_CONTIGUOUS(weight_interleaved);
  CHECK_DIM(3, weight);
  CHECK_DIM(3, weight_interleaved);
  TVM_FFI_ICHECK_EQ(weight.size(0), weight_interleaved.size(0))
      << "weight and weight_interleaved must share num_experts dim";
  TVM_FFI_ICHECK_EQ(weight.size(1), weight_interleaved.size(1))
      << "weight and weight_interleaved must share n dim";
  TVM_FFI_ICHECK_EQ(weight.size(2), weight_interleaved.size(2))
      << "weight and weight_interleaved must share packed-k dim";
  TVM_FFI_ICHECK(quant_type == 0 || quant_type == 1 || quant_type == 2)
      << "quant_type must be 0 (INT4), 1 (FP4), or 2 (FP4 for FP8), got " << quant_type;

  int64_t const num_experts = weight.size(0);
  int64_t const n = weight.size(1);
  int64_t const k = weight.size(2) * 2;
  TVM_FFI_ICHECK_EQ(n % 16, 0)
      << "weight n dimension must be divisible by 16 for SM90 mixed-gemm interleave";
  TVM_FFI_ICHECK_EQ(k % 64, 0)
      << "logical K dimension must be divisible by 64 for SM90 mixed-gemm interleave";
  int64_t const per_expert_bytes = n * (k / 2);

  auto stream = get_stream(weight.device());
  auto* src = static_cast<uint8_t*>(weight.data_ptr());
  auto* dst = static_cast<uint8_t*>(weight_interleaved.data_ptr());
  for (int64_t e = 0; e < num_experts; ++e) {
    uint8_t* src_e = src + e * per_expert_bytes;
    uint8_t* dst_e = dst + e * per_expert_bytes;
    if (quant_type == 1) {
      tensorrt_llm::kernels::cutlass_kernels::interleave_fp4_weights_for_sm90_mixed_gemm(
          src_e, dst_e, static_cast<int>(n), static_cast<int>(k), stream);
    } else if (quant_type == 2) {
      tensorrt_llm::kernels::cutlass_kernels::interleave_fp4_fp8_weights_for_sm90_mixed_gemm(
          src_e, dst_e, static_cast<int>(n), static_cast<int>(k), stream);
    } else {
      tensorrt_llm::kernels::cutlass_kernels::interleave_int4_weights_for_sm90_mixed_gemm(
          src_e, dst_e, static_cast<int>(n), static_cast<int>(k), stream);
    }
  }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(interleave_moe_weights_for_sm90_mixed_gemm,
                              interleave_moe_weights_for_sm90_mixed_gemm);
