/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <iostream>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/batched_gemm/KernelRunner.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/SfLayoutDecl.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/quantization.h"

namespace tensorrt_llm {
namespace kernels {
namespace trtllmgen_moe {

namespace btg = batchedGemm::trtllm::gen;

namespace PermuteGemm1 {

using tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType;
using tensorrt_llm::kernels::trtllmgen_moe::MoE::isGatedActivation;
using tensorrt_llm::kernels::trtllmgen_moe::MoE::serializeActivationType;

static inline ActType activationTypeToGatedActType(ActivationType actType) {
  switch (actType) {
    case ActivationType::Swiglu:
      return ActType::SwiGlu;
    case ActivationType::Geglu:
      return ActType::GeGlu;
    default:
      FLASHINFER_CHECK(false, "Unsupported gated activation type ",
                       serializeActivationType(actType), " of enum ",
                       static_cast<int64_t>(actType));
  }
  return ActType::SwiGlu;
}

static inline EltwiseActType activationTypeToEltwiseActType(ActivationType actType) {
  switch (actType) {
    case ActivationType::Relu2:
      return EltwiseActType::Relu2;
    case ActivationType::Identity:
      return EltwiseActType::None;
    default:
      FLASHINFER_CHECK(false, "Unsupported eltwise activation type ",
                       serializeActivationType(actType), " of enum ",
                       static_cast<int64_t>(actType));
  }
  return EltwiseActType::None;
}

tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOutput, int32_t tileTokensDim,
    bool useDeepSeekFp8, ActivationType activationType, bool useShuffledMatrix,
    batchedGemm::gemm::MatrixLayout weightLayout, batchedGemm::gemm::BiasType biasType,
    bool usePerTokenScaling, bool usePerChannelScaling) {
  int64_t actTypeInt = static_cast<int64_t>(activationType);
  FLASHINFER_CHECK(
      0 <= actTypeInt && actTypeInt < static_cast<int64_t>(ActivationType::InvalidType),
      "Unknown activation type", serializeActivationType(activationType), "of enum", actTypeInt);
  bool isGatedAct = isGatedActivation(activationType);
  bool useBiasMn = biasType == batchedGemm::gemm::BiasType::Mn;
  // ReorderAndShuffle is only supported on fused-act (gated) paths in trtllm-gen.
  // DSFp8 uses non-fused activation, so it must use Shuffle mode for biasMn.
  auto fusedBiasShuffleMode =
      useBiasMn ? (useDeepSeekFp8 ? batchedGemm::gemm::FusedBiasShuffleMode::Shuffle
                                  : batchedGemm::gemm::FusedBiasShuffleMode::ReorderAndShuffle)
                : batchedGemm::gemm::FusedBiasShuffleMode::None;
  auto const biasDtype = batchedGemm::trtllm::gen::Dtype::Bfloat16;
  if (useBiasMn) {
    // These checks are because trtllm-gen only exports a subset of the bias types and modes
    FLASHINFER_CHECK(isGatedAct,
                     "PermuteGemm1 BiasType::Mn requires a gated activation (SwiGlu/GeGlu)");
    FLASHINFER_CHECK(useShuffledMatrix,
                     "PermuteGemm1 BiasType::Mn requires useShuffledMatrix=true");
  }
  if (isGatedAct) {
    ActType actType = activationTypeToGatedActType(activationType);
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options = {
        // Swap A and B dtypes because transposeMmaOutput is hardcoded to true
        .dtypeA = dtypeWeights,
        .dtypeB = dtypeAct,
        .dtypeC = dtypeOutput,
        .actType = actType,
        .deepSeekFp8 = useDeepSeekFp8,
        .fusedAct = !useDeepSeekFp8,
        .routeAct = true,
        .staticBatch = false,
        .transposeMmaOutput = true,
        .tileSize = tileTokensDim,
        .epilogueTileM = useDeepSeekFp8 ? 64 : 128,
        .useShuffledMatrix = useShuffledMatrix,
        .weightLayout = weightLayout,
        .biasType = biasType,
        .fusedBiasShuffleMode = fusedBiasShuffleMode,
        .biasDtype = biasDtype,
        .usePerTokenScaling = usePerTokenScaling,
        .usePerChannelScaling = usePerChannelScaling,
    };
    return options;
  } else {
    EltwiseActType actType = activationTypeToEltwiseActType(activationType);
    tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options = {
        // Swap A and B dtypes because transposeMmaOutput is hardcoded to true
        .dtypeA = dtypeWeights,
        .dtypeB = dtypeAct,
        .dtypeC = dtypeOutput,
        .eltwiseActType = actType,
        .deepSeekFp8 = useDeepSeekFp8,
        .fusedAct = false,
        .routeAct = true,
        .staticBatch = false,
        .transposeMmaOutput = true,
        .tileSize = tileTokensDim,
        .epilogueTileM = 128,
        .useShuffledMatrix = useShuffledMatrix,
        .weightLayout = weightLayout,
        .biasType = biasType,
        .fusedBiasShuffleMode = fusedBiasShuffleMode,
        .biasDtype = biasDtype,
        .usePerTokenScaling = usePerTokenScaling,
        .usePerChannelScaling = usePerChannelScaling};
    return options;
  }
}

Runner::Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOutput,
               bool useDeepSeekFp8, int tileTokensDim, ActivationType activationType,
               bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
               batchedGemm::gemm::BiasType biasType, bool usePerTokenScaling,
               bool usePerChannelScaling)
    : mDtypeAct(dtypeAct),
      mDtypeWeights(dtypeWeights),
      mDtypeOutput(dtypeOutput),
      mTileTokensDim(tileTokensDim),
      mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(getOptions(
          mDtypeAct, mDtypeWeights, mDtypeOutput, mTileTokensDim, useDeepSeekFp8, activationType,
          useShuffledMatrix, weightLayout, biasType, usePerTokenScaling, usePerChannelScaling))),
      mActType(activationType),
      mBiasType(biasType) {}

void Runner::run(void* hiddenState, void* hiddenStateScale, void* weights, void* weightsScale,
                 void* perTokenScales, void* perChannelScales, float* outputScalesScalar,
                 float* outputScalesGateScalar, void* ptrBias, float* ptrAlpha, float* ptrBeta,
                 float* ptrClampLimit, int32_t* permutedIdxToBiasRowIdx, void* output,
                 void* outputScale, int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                 int32_t numExperts, int32_t numTokens, int32_t* permutedIdxToTokenIdx,
                 int32_t* ptrNumNonExitingCtas, int32_t* ptrTotalNumPaddedTokens,
                 int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit, void* bmm1Workspace,
                 bool useRoutingScalesOnInput, int device, cudaStream_t stream, int32_t configIndex,
                 bool enable_pdl) {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int32_t intermediateSizeFactor = (isGatedActivation(mActType) ? 2 : 1);
  if (mBiasType == batchedGemm::gemm::BiasType::Mn) {
    FLASHINFER_CHECK(ptrBias != nullptr,
                     "PermuteGemm1 configured with BiasType::Mn requires a non-null bias pointer");
    FLASHINFER_CHECK(
        permutedIdxToBiasRowIdx != nullptr,
        "PermuteGemm1 configured with BiasType::Mn requires a non-null permutedIdxToBiasRowIdx");
  }
  mRunner.run(numTokens, intermediateSizeFactor * intermediateSize, hiddenSize, {}, numTokens,
              numExperts, maxNumCtasInBatchDim, hiddenState, hiddenStateScale, weights,
              weightsScale, perTokenScales, perChannelScales, outputScalesScalar,
              outputScalesGateScalar, reinterpret_cast<float const*>(ptrBias), ptrAlpha, ptrBeta,
              ptrClampLimit, output, outputScale, permutedIdxToTokenIdx, ptrTotalNumPaddedTokens,
              ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit, ptrNumNonExitingCtas,
              permutedIdxToBiasRowIdx, bmm1Workspace, stream, device, configIndex, enable_pdl);
}

size_t Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                                       int32_t numExperts, int32_t numTokens,
                                       int32_t configIndex) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int32_t intermediateSizeFactor = (isGatedActivation(mActType) ? 2 : 1);
  return mRunner.getWorkspaceSizeInBytes(numTokens, intermediateSizeFactor * intermediateSize,
                                         hiddenSize, {}, numTokens, numExperts,
                                         maxNumCtasInBatchDim, configIndex);
}

int32_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                           int32_t intermediateSize, int32_t numExperts,
                                           int32_t numTokens) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int32_t intermediateSizeFactor = (isGatedActivation(mActType) ? 2 : 1);
  return mRunner.getDefaultValidConfigIndex(numTokens, intermediateSizeFactor * intermediateSize,
                                            hiddenSize, {}, numTokens, numExperts,
                                            maxNumCtasInBatchDim);
}

bool Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                int32_t intermediateSize, int32_t numExperts,
                                int32_t numTokens) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);

  int32_t intermediateSizeFactor = (isGatedActivation(mActType) ? 2 : 1);
  auto const isValid =
      mRunner.isValidConfigIndex(configIndex, numTokens, intermediateSizeFactor * intermediateSize,
                                 hiddenSize, {}, numTokens, numExperts, maxNumCtasInBatchDim);

  return isValid;
}

std::vector<int64_t> Runner::getPassingConfigIndices() const {
  return mRunner.getPassingConfigIndices();
}
}  // namespace PermuteGemm1

namespace Gemm2 {
tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions getOptions(
    btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut, int32_t tileTokensDim,
    bool useDeepSeekFp8, bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
    bool usePerTokenScaling, bool usePerChannelScaling) {
  tensorrt_llm::kernels::TrtllmGenBatchedGemmRunnerOptions options = {
      // Swap A and B dtypes because transposeMmaOutput is hardcoded to true
      .dtypeA = dtypeWeights,
      .dtypeB = dtypeAct,
      .dtypeC = dtypeOut,
      .eltwiseActType = EltwiseActType::None,
      .deepSeekFp8 = useDeepSeekFp8,
      .fusedAct = false,
      .routeAct = false,
      .staticBatch = false,
      .transposeMmaOutput = true,
      .tileSize = tileTokensDim,
      .epilogueTileM = useDeepSeekFp8 ? 64 : 128,
      .useShuffledMatrix = useShuffledMatrix,
      .weightLayout = weightLayout,
      .usePerTokenScaling = usePerTokenScaling,
      .usePerChannelScaling = usePerChannelScaling};
  return options;
}

Runner::Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOut,
               bool useDeepSeekFp8, int tileTokensDim, bool useShuffledMatrix,
               batchedGemm::gemm::MatrixLayout weightLayout, bool usePerTokenScaling,
               bool usePerChannelScaling)
    : mDtypeAct(dtypeAct),
      mDtypeWeights(dtypeWeights),
      mDtypeOut(dtypeOut),
      mTileTokensDim(tileTokensDim),
      mRunner(tensorrt_llm::kernels::TrtllmGenBatchedGemmRunner(
          getOptions(dtypeAct, dtypeWeights, dtypeOut, tileTokensDim, useDeepSeekFp8,
                     useShuffledMatrix, weightLayout, usePerTokenScaling, usePerChannelScaling))) {}

void Runner::run(void* permutedHiddenState, void* permutedHiddenStateScale, void* weights,
                 void* weightsScale, void* perTokenScales, void* perChannelScales,
                 float* outputScalesScalar, float* ptrBias, void* output, void* outputScale,
                 int32_t topK, int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts,
                 int32_t numTokens, int32_t* ptrNumNonExitingCtas, int32_t* ptrTotalNumPaddedTokens,
                 int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit, void* bmm2Workspace,
                 int device, cudaStream_t stream, int32_t configIndex, bool enable_pdl) {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  mRunner.run(
      numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim,
      permutedHiddenState, permutedHiddenStateScale, weights, weightsScale,
      /* perTokensSfA */ perTokenScales,
      /* perTokensSfB */ perChannelScales, outputScalesScalar, /* outputScalesGateScalar */ nullptr,
      ptrBias,
      /* ptrAlpha */ nullptr, /* ptrBeta */ nullptr, /* clampLimit */ nullptr, output, outputScale,
      /* permutedIdxToTokenIdx */ nullptr, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx,
      ptrCtaIdxXyToMnLimit, ptrNumNonExitingCtas, /* permutedIdxToBiasRowIdx */ nullptr,
      bmm2Workspace, stream, device, configIndex, enable_pdl);
}

size_t Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                                       int32_t numExperts, int32_t numTokens,
                                       int32_t configIndex) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  return mRunner.getWorkspaceSizeInBytes(numTokens, hiddenSize, intermediateSize, {}, numTokens,
                                         numExperts, maxNumCtasInBatchDim, configIndex);
}

int32_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                           int32_t intermediateSize, int32_t numExperts,
                                           int32_t numTokens) const {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  return mRunner.getDefaultValidConfigIndex(numTokens, hiddenSize, intermediateSize, {}, numTokens,
                                            numExperts, maxNumCtasInBatchDim);
}

bool Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                int32_t intermediateSize, int32_t numExperts,
                                int32_t numTokens) const {
  auto const maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);

  auto const isValid =
      mRunner.isValidConfigIndex(configIndex, numTokens, hiddenSize, intermediateSize, {},
                                 numTokens, numExperts, maxNumCtasInBatchDim);

  return isValid;
}

std::vector<int64_t> Runner::getPassingConfigIndices() const {
  return mRunner.getPassingConfigIndices();
}
}  // namespace Gemm2

namespace MoE {
Runner::Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, bool useDeepSeekFp8,
               int32_t tileTokensDim, ActivationType activationType, bool useShuffledMatrix,
               batchedGemm::gemm::MatrixLayout weightLayout,
               batchedGemm::gemm::BiasType gemm1BiasType, bool usePerTokenScalingGemm1,
               bool usePerTokenScalingGemm2, bool usePerChannelScalingGemm1,
               bool usePerChannelScalingGemm2)
    : mUsePerTokenScalingGemm1(usePerTokenScalingGemm1),
      mUsePerTokenScalingGemm2(usePerTokenScalingGemm2),
      mUsePerChannelScalingGemm1(usePerChannelScalingGemm1),
      mUsePerChannelScalingGemm2(usePerChannelScalingGemm2),
      mPermuteGemm1(PermuteGemm1::Runner(
          dtypeAct, dtypeWeights, usePerTokenScalingGemm2 ? btg::Dtype::Bfloat16 : dtypeAct,
          useDeepSeekFp8, tileTokensDim, activationType, useShuffledMatrix, weightLayout,
          gemm1BiasType, usePerTokenScalingGemm1, usePerChannelScalingGemm1)),
      mGemm2(Gemm2::Runner(dtypeAct, dtypeWeights, btg::Dtype::Bfloat16, useDeepSeekFp8,
                           tileTokensDim, useShuffledMatrix, weightLayout, usePerTokenScalingGemm2,
                           usePerChannelScalingGemm2)) {
  auto const& gemm1PassingIndices = mPermuteGemm1.getPassingConfigIndices();
  auto const& gemm2PassingIndices = mGemm2.getPassingConfigIndices();

  auto const totalPassingIndices = gemm1PassingIndices.size() * gemm2PassingIndices.size();
  mPassingConfigs.reserve(totalPassingIndices);

  for (auto const& indexGemm1 : gemm1PassingIndices) {
    for (auto const& indexGemm2 : gemm2PassingIndices) {
      mPassingConfigs.push_back(MoEConfig{indexGemm1, indexGemm2});
    }
  }
  FLASHINFER_CHECK(!mPassingConfigs.empty(),
                   "No compatible configs found for the fp8 block scale MoE runner.");
}

Runner::Runner(btg::Dtype dtypeElt, bool useDeepSeekFp8, int32_t tileTokensDim,
               bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
               bool usePerTokenScalingGemm1, bool usePerTokenScalingGemm2,
               bool usePerChannelScalingGemm1, bool usePerChannelScalingGemm2)
    : Runner(dtypeElt, dtypeElt, useDeepSeekFp8, tileTokensDim, ActivationType::Swiglu,
             useShuffledMatrix, weightLayout, batchedGemm::gemm::BiasType::None,
             usePerTokenScalingGemm1, usePerTokenScalingGemm2, usePerChannelScalingGemm1,
             usePerChannelScalingGemm2) {}

void Runner::setOpsData(MoERunnerArgs const& args, MoEWorkspace const& workspace,
                        bool const enablePdl, moe::dev::convertsf::Data& convertSfData,
                        moe::dev::activation::Data& activationData,
                        moe::dev::finalize::Data& finalizeData) {
  // Setup sf conversion data if needed
  convertSfData.inSfPtr = args.hidden_states_scale;
  convertSfData.outSfPtr = workspace.hidden_states_scale_linear;
  convertSfData.hiddenDimSf = args.hidden_size / 16;
  convertSfData.numTokens = args.num_tokens;
  convertSfData.sfLayoutSrc = btg::SfLayout::R128c4;
  convertSfData.sfLayoutDst = btg::SfLayout::Linear;
  convertSfData.mUsePdl = enablePdl;

  int32_t const totalNumExperts = args.num_experts + args.num_fused_shared_experts;
  int32_t const totalExpertsPerToken = args.top_k + args.num_fused_shared_experts;

  // Setup activation data
  activationData.mDtypeElt = args.mDtypeElt;
  activationData.mUsePdl = enablePdl;
  activationData.mUseDeepSeekFp8 = true;
  activationData.inPtr = workspace.gemm1_output;
  activationData.outPtr = workspace.activation_output;
  activationData.inDqSfsPtr = workspace.gemm1_output_scale;
  activationData.outDqSfsPtr = workspace.activation_output_scale;
  activationData.innerDim =
      args.intermediate_size * (isGatedActivation(args.activation_type) ? 2 : 1);
  activationData.topK = totalExpertsPerToken;
  activationData.numTokens = args.num_tokens;
  activationData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;

  activationData.totalNumPaddedTokens = workspace.total_num_padded_tokens;

  // Setup finalize data
  if (args.do_finalize) {
    // Setup finalize data
    finalizeData.mDtypeElt = args.mDtypeOut;
    finalizeData.mDtypeExpW = args.mDtypeExpW;
    finalizeData.mUsePdl = enablePdl;
    finalizeData.mUseDeepSeekFp8 = false;
    finalizeData.inPtr = workspace.gemm2_output;
    finalizeData.outPtr = args.output;
    finalizeData.inDqSfsPtr = workspace.gemm2_output_scale;
    finalizeData.outDqSfsPtr = args.output_scale;
    if (args.mUseRoutingScalesOnInput) {
      finalizeData.expertWeightsPtr = nullptr;
    } else {
      finalizeData.expertWeightsPtr = workspace.expert_weights;
    }
    finalizeData.expandedIdxToPermutedIdx = workspace.expanded_idx_to_permuted_idx;
    finalizeData.numTokens = args.num_tokens;
    finalizeData.numExperts = totalNumExperts;
    finalizeData.topK = totalExpertsPerToken;
    // We want to fuse unpadding into the finalize kernel, so we need to use the output hidden size.
    finalizeData.hiddenDim = args.hidden_size_output.value_or(args.hidden_size);
    finalizeData.hiddenDimPadded = args.hidden_size;
    finalizeData.totalNumPaddedTokens = workspace.total_num_padded_tokens;
  }
}

std::tuple<int32_t, int32_t> Runner::getWorkspaceSizeInBytes(MoERunnerArgs const& args,
                                                             int64_t configIndex) const {
  FLASHINFER_CHECK(configIndex >= 0 && configIndex < static_cast<int64_t>(mPassingConfigs.size()),
                   "Invalid MoE config index ", configIndex, ", valid range is [0, ",
                   static_cast<int64_t>(mPassingConfigs.size()) - 1, "].");
  int32_t const totalLocalExperts = args.local_num_experts + args.num_fused_shared_experts;
  int32_t const totalExpertsPerToken = args.top_k + args.num_fused_shared_experts;

  auto const& config = mPassingConfigs[configIndex];

  auto workspace_size_fc1 = static_cast<int32_t>(mPermuteGemm1.getWorkspaceSizeInBytes(
      totalExpertsPerToken, args.hidden_size, args.intermediate_size, totalLocalExperts,
      args.num_tokens, config.gemm1Config));
  auto workspace_size_fc2 = static_cast<int32_t>(
      mGemm2.getWorkspaceSizeInBytes(totalExpertsPerToken, args.hidden_size, args.intermediate_size,
                                     totalLocalExperts, args.num_tokens, config.gemm2Config));
  return std::make_tuple(workspace_size_fc1, workspace_size_fc2);
}

std::vector<int64_t> Runner::getValidConfigIndices(int32_t topK, int32_t hiddenSize,
                                                   int32_t intermediateSize,
                                                   int32_t numLocalExperts,
                                                   int32_t numTokens) const {
  std::vector<int64_t> validIndices;

  for (int i = 0; i < mPassingConfigs.size(); ++i) {
    auto const& config = mPassingConfigs[i];

    if (mPermuteGemm1.isValidConfigIndex(config.gemm1Config, topK, hiddenSize, intermediateSize,
                                         numLocalExperts, numTokens) &&
        mGemm2.isValidConfigIndex(config.gemm2Config, topK, hiddenSize, intermediateSize,
                                  numLocalExperts, numTokens)) {
      validIndices.push_back(i);
    }
  }

  return validIndices;
}

int64_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                           int32_t intermediateSize, int32_t numLocalExperts,
                                           int32_t numTokens) const {
  int32_t indexGemm1 = mPermuteGemm1.getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize,
                                                                numLocalExperts, numTokens);
  int32_t indexGemm2 = mGemm2.getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize,
                                                         numLocalExperts, numTokens);

  auto it = std::find_if(mPassingConfigs.begin(), mPassingConfigs.end(),
                         [indexGemm1, indexGemm2](MoEConfig cfg) {
                           return (cfg.gemm1Config == indexGemm1 && cfg.gemm2Config == indexGemm2);
                         });
  FLASHINFER_CHECK(it != mPassingConfigs.end(),
                   "No compatible configs found for the block scale MoE runner.");
  return std::distance(mPassingConfigs.begin(), it);
}

void Runner::run(MoERunnerArgs const& args, MoEWorkspace const& workspace, int device,
                 cudaStream_t stream, int64_t configIndex, bool enable_pdl) {
  FLASHINFER_CHECK(configIndex >= 0 && configIndex < static_cast<int64_t>(mPassingConfigs.size()),
                   "Invalid MoE config index ", configIndex, ", valid range is [0, ",
                   static_cast<int64_t>(mPassingConfigs.size()) - 1, "].");
  FLASHINFER_CHECK(!mUsePerChannelScalingGemm1 && !mUsePerChannelScalingGemm2,
                   "Per-channel scaling is currently not supported.");
  // Setup all operation data
  moe::dev::activation::Data activationData;
  moe::dev::finalize::Data finalizeData;
  moe::dev::convertsf::Data convertSfData;
  sync_check_cuda_error(stream);
  setOpsData(args, workspace, enable_pdl, convertSfData, activationData, finalizeData);

  void* hidden_states_scale_linear{args.hidden_states_scale};

  auto const& config = mPassingConfigs[configIndex];

  int32_t const totalLocalExperts = args.local_num_experts + args.num_fused_shared_experts;
  int32_t const totalExpertsPerToken = args.top_k + args.num_fused_shared_experts;

  int32_t* permutedIdxToBiasRowIdx = args.gemm1_bias_type == batchedGemm::gemm::BiasType::Mn
                                         ? workspace.permuted_idx_to_expanded_idx
                                         : nullptr;
  mPermuteGemm1.run(
      args.hidden_states, hidden_states_scale_linear, args.gemm1_weights, args.gemm1_weights_scale,
      workspace.token_scales, /* perChannelScales */ nullptr, args.output1_scales_scalar,
      args.output1_scales_gate_scalar, args.gemm1_bias, args.gemm1_alpha, args.gemm1_beta,
      args.gemm1_clamp_limit, permutedIdxToBiasRowIdx, workspace.gemm1_output,
      workspace.gemm1_output_scale, totalExpertsPerToken, args.hidden_size, args.intermediate_size,
      totalLocalExperts, args.num_tokens, workspace.permuted_idx_to_token_idx,
      workspace.num_non_exiting_ctas, workspace.total_num_padded_tokens,
      workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit, workspace.bmm1_workspace,
      args.mUseRoutingScalesOnInput, device, stream, config.gemm1Config, enable_pdl);

  // We do not fuse activation with FC1 for DeepSeek FP8 due to the weights shuffling constraint.
  void* gemm2_input = workspace.gemm1_output;
  void* gemm2_input_scale = workspace.gemm1_output_scale;
  // We do activation only for DeepSeek FP8, as cubins do not have fused activation.
  if (args.mDtypeElt == btg::Dtype::E4m3 && args.mUseDeepSeekFp8) {
    // Run activation
    moe::dev::activation::run(activationData, stream);
    gemm2_input = workspace.activation_output;
    gemm2_input_scale = workspace.activation_output_scale;
  } else if (mUsePerTokenScalingGemm2) {
    // TODO(siyuan): currently only support per-token nvfp4 quantization
    FLASHINFER_CHECK(
        mPermuteGemm1.mDtypeOutput == btg::Dtype::Bfloat16,
        "When using explicit quantization, PermuteGemm1 output dtype must be Bfloat16.");
    FLASHINFER_CHECK(mGemm2.mDtypeAct == btg::Dtype::E2m1,
                     "Currently only support NvFP4 when using explicit quantization.");
    FLASHINFER_CHECK(
        workspace.token_scales_fc2 != nullptr,
        "workspace.token_scales_fc2 must be provided When using explicit quantization.");
    // FIXME(siyuan): Detect from the kernel config. Currently only tile size >= 128 will use R128c4
    auto sfLayout = mGemm2.mTileTokensDim >= 128 ? QuantizationSFLayout::SWIZZLED_128x4
                                                 : QuantizationSFLayout::SWIZZLED_8x4;

    float globalScaleInv = 1.f / (448.f * 6.f);
    if (tensorrt_llm::common::getEnvNVFP4Use4Over6() &&
        tensorrt_llm::common::getEnvNVFP44Over6E4M3Use256()) {
      globalScaleInv = 1.f / (256.f * 6.f);
    }
    invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16>(
        args.num_tokens * totalExpertsPerToken, args.intermediate_size,
        reinterpret_cast<__nv_bfloat16 const*>(workspace.gemm1_output), globalScaleInv,
        workspace.expanded_idx_to_permuted_idx,
        reinterpret_cast<uint8_t*>(workspace.activation_output),
        reinterpret_cast<uint8_t*>(workspace.activation_output_scale),
        reinterpret_cast<float*>(workspace.token_scales_fc2), sfLayout, stream);

    gemm2_input = workspace.activation_output;
    gemm2_input_scale = workspace.activation_output_scale;
  }

  // Run gemm2
  mGemm2.run(gemm2_input, gemm2_input_scale, args.gemm2_weights, args.gemm2_weights_scale,
             workspace.token_scales_fc2, /*perChannelScales*/ nullptr, args.output2_scales_scalar,
             args.gemm2_bias, workspace.gemm2_output, workspace.gemm2_output_scale,
             totalExpertsPerToken, args.hidden_size, args.intermediate_size, totalLocalExperts,
             args.num_tokens, workspace.num_non_exiting_ctas, workspace.total_num_padded_tokens,
             workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit,
             workspace.bmm2_workspace, device, stream, config.gemm2Config, enable_pdl);

  // Run finalize
  if (args.do_finalize) {
    // Run finalize
    moe::dev::finalize::run(finalizeData, stream);
    sync_check_cuda_error(stream);
  }
}
}  // namespace MoE

}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
