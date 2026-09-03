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

#include <algorithm>
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

namespace {
btg::Dtype getPerTokenScaleDtype(btg::Dtype dtypeAct, bool usePerTokenScaling,
                                 bool usePerChannelScaling) {
  if (usePerChannelScaling) {
    return btg::Dtype::Fp32;
  }
  if (!usePerTokenScaling) {
    return btg::Dtype::Void;
  }
  return dtypeAct == btg::Dtype::E4m3 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;
}

// trtllm-gen turns validM/validN/validK into the TMA descriptor's globalDim (see
// makeTmaShapeStrideAbc() in trtllmGen_bmm_export/KernelParams.h: the *shape* comes from the
// valid dims while the *stride* comes from the padded dims). cuTensorMapEncodeTiled therefore
// enforces the kernel's dimension constraints on the valid dims directly, and a bad value
// surfaces only as an opaque "Failed to initialize the TMA descriptor" from the driver.
//
// GemmOptions.h documents what those constraints are:
//   1. outputDim % (4 * sfBlockSize) == 0; as 4x SFs are packed into 4 bytes
//   2. MxFp4 x Fp8 mmaType requires bespoke TMA load which requires hiddenDim % 128 == 0
//   3. TMA requires 16B alignment for each row
// 128 satisfies all three for every dtype this MoE runner supports (sfBlockSize is 32 for
// mx* and 16 for nvfp4, so 4*sfBlockSize is 128 or 64; the 16B row rule needs at most 16
// elements; the shuffled-matrix and DeepSeek-FP8 rules also want 128). Rather than encode a
// per-dtype table that has to track trtllm-gen, use the single conservative value: at most
// 127 extra elements are contracted, which is negligible next to the padding we skip.
constexpr int32_t kValidDimAlignment = 128;

// Round a valid dim up to the alignment the TMA descriptor requires, clamped to the padded
// dim. Rounding *up* (rather than rejecting) is safe: the region between the caller's valid
// dim and the padded dim is already required to be zero-filled, because without valid dims at
// all the kernel contracts over the *entire* padded extent. Rounding up therefore relies on
// strictly less zero-padding than the valid-dims-disabled path does.
// Returns -1 unchanged (meaning "no valid dim supplied").
int32_t alignValidDim(int32_t validDim, int32_t paddedDim) {
  if (validDim < 0) {
    return -1;
  }
  int32_t const aligned =
      ((validDim + kValidDimAlignment - 1) / kValidDimAlignment) * kValidDimAlignment;
  return std::min(aligned, paddedDim);
}

}  // namespace

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
    case ActivationType::Situ:
      return ActType::SiTuGlu;
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
        .perTokenSfDtype =
            getPerTokenScaleDtype(dtypeAct, usePerTokenScaling, usePerChannelScaling),
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
        .perTokenSfDtype =
            getPerTokenScaleDtype(dtypeAct, usePerTokenScaling, usePerChannelScaling),
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
                 bool enable_pdl, int32_t validHiddenSize, int32_t validIntermediateSize) {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int32_t intermediateSizeFactor = (isGatedActivation(mActType) ? 2 : 1);
  // Align the intermediate before scaling by the gate factor, so that both the per-gate half
  // and the full N dimension land on the alignment boundary.
  int32_t const alignedValidIntermediate = alignValidDim(validIntermediateSize, intermediateSize);
  int32_t validN =
      (alignedValidIntermediate >= 0) ? intermediateSizeFactor * alignedValidIntermediate : -1;
  int32_t validK = alignValidDim(validHiddenSize, hiddenSize);
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
              permutedIdxToBiasRowIdx, bmm1Workspace, stream, device, configIndex, enable_pdl,
              /* validM */ -1, validN, validK);
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
      .perTokenSfDtype = getPerTokenScaleDtype(dtypeAct, usePerTokenScaling, usePerChannelScaling),
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
                 int device, cudaStream_t stream, int32_t configIndex, bool enable_pdl,
                 int32_t validIntermediateSize, int32_t validHiddenSize) {
  auto maxNumCtasInBatchDim =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  // GEMM2: [numTokens, intermediateSize] @ [intermediateSize, hiddenSize] -> [numTokens,
  // hiddenSize] For batched GEMM with transposeMmaOutput: M=numTokens, N=hiddenSize,
  // K=intermediateSize validN = validHiddenSize, validK = validIntermediateSize
  // Note hiddenSize here is the GEMM2 N the MoE runner selected, which is the *output* hidden
  // size (hidden_size_output) only when the caller supplied valid dims.
  int32_t validN = alignValidDim(validHiddenSize, hiddenSize);
  int32_t validK = alignValidDim(validIntermediateSize, intermediateSize);
  mRunner.run(
      numTokens, hiddenSize, intermediateSize, {}, numTokens, numExperts, maxNumCtasInBatchDim,
      permutedHiddenState, permutedHiddenStateScale, weights, weightsScale,
      /* perTokensSfA */ perTokenScales,
      /* perTokensSfB */ perChannelScales, outputScalesScalar, /* outputScalesGateScalar */ nullptr,
      ptrBias,
      /* ptrAlpha */ nullptr, /* ptrBeta */ nullptr, /* clampLimit */ nullptr, output, outputScale,
      /* permutedIdxToTokenIdx */ nullptr, ptrTotalNumPaddedTokens, ptrCtaIdxXyToBatchIdx,
      ptrCtaIdxXyToMnLimit, ptrNumNonExitingCtas, /* permutedIdxToBiasRowIdx */ nullptr,
      bmm2Workspace, stream, device, configIndex, enable_pdl, /* validM */ -1, validN, validK);
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

namespace {

// GEMM2's N dimension, i.e. the row width of the gemm2_output workspace.
//
// hidden_size_output narrows the GEMM only when the caller opted in by also supplying valid dims:
// that is the contract under which the FC2 weights are laid out for the narrower N. Without valid
// dims a caller may still hand in an `output` narrower than hidden_size (the pre-existing NVFP4
// contract), while the weights still describe the full padded hidden_size. Shrinking N there would
// reinterpret a row-shuffled weight matrix as having fewer rows -- a scrambled subset, not a
// prefix -- so GEMM2 keeps the full N and finalize truncates the rows instead.
//
// With valid dims hidden_size_output is roundUp(valid_hidden_size, 128) (the launcher computes it,
// matching TRT-LLM's args.output_hidden_size), which is *not* the caller-visible output row width:
// that one is valid_hidden_size, and only finalize sees it. See setOpsData.
int32_t getGemm2OutputHiddenSize(MoERunnerArgs const& args) {
  return args.valid_hidden_size.has_value() ? args.hidden_size_output.value_or(args.hidden_size)
                                            : args.hidden_size;
}

}  // namespace

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

  // SwiGLU OAI controls. The fused-epilogue paths get these through the FC1 GEMM instead; this
  // kernel only runs for DeepSeek FP8, where FC1 has no fused activation to carry them.
  activationData.gatedActAlphaPtr = args.gemm1_alpha;
  activationData.gatedActBetaPtr = args.gemm1_beta;
  activationData.gatedActClampLimitPtr = args.gemm1_clamp_limit;
  activationData.ctaIdxXyToBatchIdx = workspace.cta_idx_xy_to_batch_idx;
  activationData.tileTokensDim = workspace.ProjUpTileN;

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
    // Fuse unpadding into finalize: hiddenDim is the caller-visible `output` row width, while
    // hiddenDimPadded is GEMM2's row stride, i.e. its N (see getGemm2OutputHiddenSize).
    //
    // With valid dims the output row is exactly valid_hidden_size wide, matching TRT-LLM's
    // contract (mxFp4BlockScaleMoe.cpp allocates {num_tokens, valid_hidden_size}), while GEMM2
    // computed roundUp(valid_hidden_size, 128) columns because that is the width the FC2 weights
    // are laid out for. Finalize therefore writes *every* column of the output -- no
    // uninitialized memory can reach the caller -- and the surplus computed columns
    // [valid_hidden_size, roundUp(valid_hidden_size, 128)) simply stay behind in the
    // gemm2_output workspace.
    //
    // Without valid dims GEMM2 keeps the full padded N (see getGemm2OutputHiddenSize), and this
    // is the only place a narrower hidden_size_output takes effect: finalize reads the full
    // hidden_size stride and writes just the leading hidden_size_output columns.
    //
    // MoERunnerArgs carries no separate output-width field: the width is fully determined by the
    // dims above, so derive it here rather than duplicating it into another field that could
    // disagree with them.
    finalizeData.hiddenDim =
        args.valid_hidden_size.value_or(args.hidden_size_output.value_or(args.hidden_size));
    finalizeData.hiddenDimPadded = getGemm2OutputHiddenSize(args);
    FLASHINFER_CHECK(finalizeData.hiddenDim <= finalizeData.hiddenDimPadded,
                     "Finalize output width ", finalizeData.hiddenDim,
                     " exceeds the GEMM2 output row stride ", finalizeData.hiddenDimPadded, ".");
    // finalizeKernelVecLoad reinterprets each output row as 128-bit vectors (its numElemsInCol is
    // hiddenDim / eltsPer16B, and row `t` starts at t * hiddenDim), so a row width that is not a
    // whole number of 16B chunks would drop the trailing columns and misalign the stores. The
    // kernel only asserts this, which is a no-op in release builds; the aligned widths this path
    // used to see made it unreachable, but the output width is now the caller's raw
    // valid_hidden_size, so check it here.
    //
    // Scoped to the valid-dims path on purpose. Without valid dims a caller may still hand in a
    // narrow `output` (the pre-existing #2217 contract) whose width was never constrained, and
    // small problems there dispatch to the non-vectorized finalizeKernel, which handles any width.
    // Applying the check unconditionally would reject widths that work today.
    if (args.valid_hidden_size.has_value()) {
      int32_t const outputEltBits = btg::dtypeGetNumBits(args.mDtypeOut);
      FLASHINFER_CHECK(static_cast<int64_t>(finalizeData.hiddenDim) * outputEltBits % 128 == 0,
                       "MoE output row width ", finalizeData.hiddenDim,
                       " is not 16B-aligned for a ", outputEltBits,
                       "-bit output dtype; it must be a multiple of ", 128 / outputEltBits, ".");
    }
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
  // Must match the N that Runner::run actually gives GEMM2.
  int32_t const gemm2HiddenSize = getGemm2OutputHiddenSize(args);
  auto workspace_size_fc2 = static_cast<int32_t>(
      mGemm2.getWorkspaceSizeInBytes(totalExpertsPerToken, gemm2HiddenSize, args.intermediate_size,
                                     totalLocalExperts, args.num_tokens, config.gemm2Config));
  return std::make_tuple(workspace_size_fc1, workspace_size_fc2);
}

std::vector<int64_t> Runner::getValidConfigIndices(int32_t topK, int32_t hiddenSize,
                                                   int32_t intermediateSize,
                                                   int32_t numLocalExperts, int32_t numTokens,
                                                   int32_t hiddenSizeOutput) const {
  std::vector<int64_t> validIndices;
  hiddenSizeOutput = hiddenSizeOutput > 0 ? hiddenSizeOutput : hiddenSize;

  for (int i = 0; i < mPassingConfigs.size(); ++i) {
    auto const& config = mPassingConfigs[i];

    if (mPermuteGemm1.isValidConfigIndex(config.gemm1Config, topK, hiddenSize, intermediateSize,
                                         numLocalExperts, numTokens) &&
        mGemm2.isValidConfigIndex(config.gemm2Config, topK, hiddenSizeOutput, intermediateSize,
                                  numLocalExperts, numTokens)) {
      validIndices.push_back(i);
    }
  }

  return validIndices;
}

MoEConfig Runner::getConfigComponents(int64_t configIndex) const {
  FLASHINFER_CHECK(configIndex >= 0 && configIndex < static_cast<int64_t>(mPassingConfigs.size()),
                   "Invalid MoE config index ", configIndex, ", valid range is [0, ",
                   static_cast<int64_t>(mPassingConfigs.size()) - 1, "].");
  return mPassingConfigs[configIndex];
}

bool Runner::isValidConfigIndex(int64_t configIndex, int32_t topK, int32_t hiddenSize,
                                int32_t intermediateSize, int32_t numLocalExperts,
                                int32_t numTokens, int32_t hiddenSizeOutput) const {
  if (configIndex < 0 || configIndex >= static_cast<int64_t>(mPassingConfigs.size())) {
    return false;
  }
  hiddenSizeOutput = hiddenSizeOutput > 0 ? hiddenSizeOutput : hiddenSize;

  auto const& config = mPassingConfigs[configIndex];
  return mPermuteGemm1.isValidConfigIndex(static_cast<int32_t>(config.gemm1Config), topK,
                                          hiddenSize, intermediateSize, numLocalExperts,
                                          numTokens) &&
         mGemm2.isValidConfigIndex(static_cast<int32_t>(config.gemm2Config), topK, hiddenSizeOutput,
                                   intermediateSize, numLocalExperts, numTokens);
}

int64_t Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                           int32_t intermediateSize, int32_t numLocalExperts,
                                           int32_t numTokens, int32_t hiddenSizeOutput) const {
  hiddenSizeOutput = hiddenSizeOutput > 0 ? hiddenSizeOutput : hiddenSize;
  int32_t indexGemm1 = mPermuteGemm1.getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize,
                                                                numLocalExperts, numTokens);
  int32_t indexGemm2 = mGemm2.getDefaultValidConfigIndex(topK, hiddenSizeOutput, intermediateSize,
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

  // Pass valid dimensions: validHiddenSize (K for GEMM1), validIntermediateSize (N factor for
  // GEMM1)
  int32_t validHiddenSize = args.valid_hidden_size.value_or(-1);
  int32_t validIntermediateSize = args.valid_intermediate_size.value_or(-1);

  int32_t* permutedIdxToBiasRowIdx = args.gemm1_bias_type == batchedGemm::gemm::BiasType::Mn
                                         ? workspace.permuted_idx_to_expanded_idx
                                         : nullptr;
  // DeepSeek FP8 activates in a separate kernel (see below), which owns the SwiGLU OAI controls.
  // Keep them out of the FC1 GEMM there so they can only ever be applied once.
  bool const useUnfusedActivation = args.mDtypeElt == btg::Dtype::E4m3 && args.mUseDeepSeekFp8;
  float* const gemm1Alpha = useUnfusedActivation ? nullptr : args.gemm1_alpha;
  float* const gemm1Beta = useUnfusedActivation ? nullptr : args.gemm1_beta;
  float* const gemm1ClampLimit = useUnfusedActivation ? nullptr : args.gemm1_clamp_limit;
  mPermuteGemm1.run(
      args.hidden_states, hidden_states_scale_linear, args.gemm1_weights, args.gemm1_weights_scale,
      args.gemm1_per_channel_weight_scale == nullptr ? workspace.token_scales
                                                     : args.hidden_states_scale,
      args.gemm1_per_channel_weight_scale, args.output1_scales_scalar,
      args.output1_scales_gate_scalar, args.gemm1_bias, gemm1Alpha, gemm1Beta, gemm1ClampLimit,
      permutedIdxToBiasRowIdx, workspace.gemm1_output, workspace.gemm1_output_scale,
      totalExpertsPerToken, args.hidden_size, args.intermediate_size, totalLocalExperts,
      args.num_tokens, workspace.permuted_idx_to_token_idx, workspace.num_non_exiting_ctas,
      workspace.total_num_padded_tokens, workspace.cta_idx_xy_to_batch_idx,
      workspace.cta_idx_xy_to_mn_limit, workspace.bmm1_workspace, args.mUseRoutingScalesOnInput,
      device, stream, config.gemm1Config, enable_pdl, validHiddenSize, validIntermediateSize);

  // We do not fuse activation with FC1 for DeepSeek FP8 due to the weights shuffling constraint.
  void* gemm2_input = workspace.gemm1_output;
  void* gemm2_input_scale = workspace.gemm1_output_scale;
  // We do activation only for DeepSeek FP8, as cubins do not have fused activation.
  if (useUnfusedActivation) {
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
    auto const sfLayoutB = mGemm2.mRunner.getSfLayoutB(config.gemm2Config);
    auto sfLayout = QuantizationSFLayout::LINEAR;
    switch (sfLayoutB) {
      case btg::SfLayout::R8c4:
        sfLayout = QuantizationSFLayout::SWIZZLED_8x4;
        break;
      case btg::SfLayout::R128c4:
        sfLayout = QuantizationSFLayout::SWIZZLED_128x4;
        break;
      default:
        FLASHINFER_CHECK(false, "Unsupported FC2 block scale layout ",
                         btg::sfLayoutToString(sfLayoutB));
    }

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
  // Pass valid dimensions: validIntermediateSize (K for GEMM2), validHiddenSize (N for GEMM2)
  int32_t const gemm2HiddenSize = getGemm2OutputHiddenSize(args);
  mGemm2.run(gemm2_input, gemm2_input_scale, args.gemm2_weights, args.gemm2_weights_scale,
             args.gemm2_per_channel_weight_scale == nullptr ? workspace.token_scales_fc2
                                                            : gemm2_input_scale,
             args.gemm2_per_channel_weight_scale, args.output2_scales_scalar, args.gemm2_bias,
             workspace.gemm2_output, workspace.gemm2_output_scale, totalExpertsPerToken,
             gemm2HiddenSize, args.intermediate_size, totalLocalExperts, args.num_tokens,
             workspace.num_non_exiting_ctas, workspace.total_num_padded_tokens,
             workspace.cta_idx_xy_to_batch_idx, workspace.cta_idx_xy_to_mn_limit,
             workspace.bmm2_workspace, device, stream, config.gemm2Config, enable_pdl,
             validIntermediateSize, validHiddenSize);

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
