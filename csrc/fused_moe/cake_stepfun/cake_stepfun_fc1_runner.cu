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

#ifndef CAKE_STEPFUN_FC1
#error "cake_stepfun_fc1_runner.cu is part of the Cake StepFun fused-MoE module (-DCAKE_STEPFUN_FC1)"
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "generated/cake_stepfun_generated_manifest.cuh"

namespace tensorrt_llm {
namespace kernels {
namespace trtllmgen_moe {
namespace cake_stepfun {

namespace btg = batchedGemm::trtllm::gen;
namespace generated = flashinfer::cake_stepfun::generated;

namespace {

generated::TensorLayout denseLayout(void const* data, std::initializer_list<int64_t> dims) {
  generated::TensorLayout layout{};
  layout.data = data;
  layout.rank = static_cast<int>(dims.size());
  int64_t stride = 1;
  int index = layout.rank;
  for (auto it = std::rbegin(dims); it != std::rend(dims); ++it) {
    --index;
    layout.dimensions[index] = *it;
    layout.strides[index] = stride;
    stride *= *it;
  }
  layout.elements = stride;
  return layout;
}

bool isCakeSelectable(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOutput,
                      bool useDeepSeekFp8, MoE::ActivationType activationType,
                      bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
                      batchedGemm::gemm::BiasType biasType, bool usePerTokenScaling,
                      bool usePerChannelScaling) {
  // The exported inventory is the NVFP4 fused-activation FC1: E2m1 activations and weights, E2m1
  // output with block scales, trtllm-shuffled MajorK weights, no bias, no per-token/channel scales.
  return dtypeAct == btg::Dtype::E2m1 && dtypeWeights == btg::Dtype::E2m1 &&
         dtypeOutput == btg::Dtype::E2m1 && !useDeepSeekFp8 &&
         activationType == MoE::ActivationType::SwigluStep && useShuffledMatrix &&
         weightLayout == batchedGemm::gemm::MatrixLayout::MajorK &&
         biasType == batchedGemm::gemm::BiasType::None && !usePerTokenScaling &&
         !usePerChannelScaling;
}

}  // namespace

Fc1Runner::Fc1Runner(btg::Dtype dtypeAct, btg::Dtype dtypeWeights, btg::Dtype dtypeOutput,
                     bool useDeepSeekFp8, int tileTokensDim, MoE::ActivationType activationType,
                     bool useShuffledMatrix, batchedGemm::gemm::MatrixLayout weightLayout,
                     batchedGemm::gemm::BiasType biasType, bool usePerTokenScaling,
                     bool usePerChannelScaling)
    : mDtypeAct(dtypeAct),
      mDtypeWeights(dtypeWeights),
      mDtypeOutput(dtypeOutput),
      mTileTokensDim(tileTokensDim),
      mActType(activationType),
      mBiasType(biasType) {
  if (isCakeSelectable(dtypeAct, dtypeWeights, dtypeOutput, useDeepSeekFp8, activationType,
                       useShuffledMatrix, weightLayout, biasType, usePerTokenScaling,
                       usePerChannelScaling)) {
    for (size_t index = 0; index < generated::kFc1KernelCount; ++index) {
      if (generated::kFc1Kernels[index].tile_n == tileTokensDim) {
        mKernels.push_back(static_cast<int32_t>(index));
      }
    }
  }
  mSmemConfigured.assign(generated::kFc1KernelCount, false);
  if (mKernels.empty()) {
    mNative.emplace(dtypeAct, dtypeWeights, dtypeOutput, useDeepSeekFp8, tileTokensDim,
                    activationType, useShuffledMatrix, weightLayout, biasType, usePerTokenScaling,
                    usePerChannelScaling);
  }
}

bool Fc1Runner::shapeSupported(int32_t configIndex, int32_t hiddenSize,
                               int32_t intermediateSize) const {
  if (std::find(mKernels.begin(), mKernels.end(), configIndex) == mKernels.end()) {
    return false;
  }
  auto const& spec = generated::kFc1Kernels[configIndex];
  // K streams in whole BLOCK_K tiles and the weight block scales in 64-column groups; the
  // interleaved up/gate output rows tile by output_rows_per_cta (128 physical weight rows).
  return hiddenSize > 0 && hiddenSize % spec.block_k == 0 && intermediateSize > 0 &&
         intermediateSize % spec.output_rows_per_cta == 0;
}

size_t Fc1Runner::getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize,
                                          int32_t intermediateSize, int32_t numExperts,
                                          int32_t numTokens, int32_t configIndex) const {
  if (mNative) {
    return mNative->getWorkspaceSizeInBytes(topK, hiddenSize, intermediateSize, numExperts,
                                            numTokens, configIndex);
  }
  return 0;
}

int32_t Fc1Runner::getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                              int32_t intermediateSize, int32_t numExperts,
                                              int32_t numTokens) const {
  if (mNative) {
    return mNative->getDefaultValidConfigIndex(topK, hiddenSize, intermediateSize, numExperts,
                                               numTokens);
  }
  for (int32_t index : mKernels) {
    if (shapeSupported(index, hiddenSize, intermediateSize)) {
      return index;
    }
  }
  FLASHINFER_CHECK(false, "No Cake StepFun FC1 kernel for tile_N=", mTileTokensDim,
                   " accepts hidden_size=", hiddenSize, ", intermediate_size=", intermediateSize,
                   " (hidden_size must be a multiple of the kernel K tile and intermediate_size a "
                   "multiple of its output rows per CTA).");
  return -1;
}

bool Fc1Runner::isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                   int32_t intermediateSize, int32_t numExperts,
                                   int32_t numTokens) const {
  if (mNative) {
    return mNative->isValidConfigIndex(configIndex, topK, hiddenSize, intermediateSize,
                                       numExperts, numTokens);
  }
  return shapeSupported(configIndex, hiddenSize, intermediateSize);
}

std::vector<int64_t> Fc1Runner::getPassingConfigIndices() const {
  if (mNative) {
    return mNative->getPassingConfigIndices();
  }
  return std::vector<int64_t>(mKernels.begin(), mKernels.end());
}

void Fc1Runner::run(void* hiddenState, void* hiddenStateScale, void* weight, void* weightScale,
                    void* perTokenScales, void* perChannelScales, float* outputScalesScalar,
                    float* outputScalesGateScalar, void* ptrBias, float* ptrGatedActAlpha,
                    float* ptrGatedActBeta, float* ptrClampLimit,
                    int32_t* permutedIdxToBiasRowIdx, void* output, void* outputScale,
                    int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                    int32_t numExperts, int32_t numTokens, int32_t* permutedIdxToTokenIdx,
                    int32_t* ptrNumNonExitingCtas, int32_t* ptrTotalNumPaddedTokens,
                    int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit,
                    void* bmm1Workspace, bool useRoutingScalesOnInput, int device,
                    cudaStream_t stream, int32_t configIndex, bool enable_pdl,
                    int32_t validHiddenSize, int32_t validIntermediateSize) {
  if (mNative) {
    mNative->run(hiddenState, hiddenStateScale, weight, weightScale, perTokenScales,
                 perChannelScales, outputScalesScalar, outputScalesGateScalar, ptrBias,
                 ptrGatedActAlpha, ptrGatedActBeta, ptrClampLimit, permutedIdxToBiasRowIdx, output,
                 outputScale, topK, hiddenSize, intermediateSize, numExperts, numTokens,
                 permutedIdxToTokenIdx, ptrNumNonExitingCtas, ptrTotalNumPaddedTokens,
                 ptrCtaIdxXyToBatchIdx, ptrCtaIdxXyToMnLimit, bmm1Workspace,
                 useRoutingScalesOnInput, device, stream, configIndex, enable_pdl,
                 validHiddenSize, validIntermediateSize);
    return;
  }
  FLASHINFER_CHECK(shapeSupported(configIndex, hiddenSize, intermediateSize),
                   "Invalid Cake StepFun FC1 config index ", configIndex, " for tile_N=",
                   mTileTokensDim, ", hidden_size=", hiddenSize,
                   ", intermediate_size=", intermediateSize);
  FLASHINFER_CHECK(hiddenStateScale != nullptr && weightScale != nullptr,
                   "Cake StepFun FC1 requires NVFP4 activation and weight block scales");
  FLASHINFER_CHECK(outputScalesScalar != nullptr && outputScalesGateScalar != nullptr,
                   "Cake StepFun FC1 requires per-expert output1 scales");
  FLASHINFER_CHECK(ptrClampLimit != nullptr,
                   "Cake StepFun FC1 requires an explicit per-expert clamp limit "
                   "(gemm1_clamp_limit in raw units: limit / output1_scales_gate_scalar)");
  FLASHINFER_CHECK(ptrGatedActAlpha == nullptr && ptrGatedActBeta == nullptr,
                   "SwigluStep accepts gemm1_clamp_limit only; gemm1_alpha / gemm1_beta must be "
                   "absent");
  FLASHINFER_CHECK(ptrBias == nullptr && permutedIdxToBiasRowIdx == nullptr,
                   "Cake StepFun FC1 does not consume a GEMM1 bias");
  FLASHINFER_CHECK(perTokenScales == nullptr && perChannelScales == nullptr,
                   "Cake StepFun FC1 does not consume per-token or per-channel scales");
  FLASHINFER_CHECK(!useRoutingScalesOnInput,
                   "Cake StepFun FC1 does not apply routing scales on the input");
  FLASHINFER_CHECK((validHiddenSize < 0 || validHiddenSize == hiddenSize) &&
                       (validIntermediateSize < 0 || validIntermediateSize == intermediateSize),
                   "Cake StepFun FC1 does not support valid (unpadded) dimensions smaller than "
                   "the padded hidden_size / intermediate_size");
  FLASHINFER_CHECK(ptrNumNonExitingCtas != nullptr && permutedIdxToTokenIdx != nullptr &&
                       ptrCtaIdxXyToBatchIdx != nullptr && ptrCtaIdxXyToMnLimit != nullptr,
                   "Cake StepFun FC1 requires the complete trtllm-gen routing arrays");

  auto const& spec = generated::kFc1Kernels[configIndex];
  int32_t const gridM = intermediateSize / spec.output_rows_per_cta;
  int32_t const gridN =
      Routing::getMaxNumCtasInBatchDim(numTokens, topK, numExperts, mTileTokensDim);
  int32_t const maxPaddedTokens =
      Routing::getMaxPermutedPaddedCount(numTokens, topK, numExperts, mTileTokensDim);
  int32_t const kTiles = hiddenSize / spec.block_k;

  generated::Fc1Args args{};
  // Weights: [numExperts, 2 * intermediateSize, hiddenSize / 2] packed E2m1 bytes.
  auto const weightLayout = denseLayout(
      weight, {int64_t{numExperts}, int64_t{2} * intermediateSize, int64_t{hiddenSize} / 2});
  FLASHINFER_CHECK(spec.encode_a(&args.A, weightLayout),
                   "Cake StepFun FC1: failed to encode the weight tensor map");
  // Weight block scales in the 128x4 interleaved layout, viewed per 128-row weight tile:
  // [numExperts * gridM, hiddenSize / 64, 2, 256] bytes.
  auto const weightScaleLayout = denseLayout(
      weightScale, {int64_t{numExperts} * gridM, int64_t{hiddenSize} / 64, int64_t{2}, int64_t{256}});
  FLASHINFER_CHECK(spec.encode_sfa(&args.SFA, weightScaleLayout),
                   "Cake StepFun FC1: failed to encode the weight block-scale tensor map");
  // Output: [maxPaddedTokens, intermediateSize / 2] packed E2m1 bytes (row stride is what the
  // kernel's descriptor consumes; the padded row extent is published by routing on device).
  auto const outputLayout =
      denseLayout(output, {int64_t{maxPaddedTokens}, int64_t{intermediateSize} / 2});
  FLASHINFER_CHECK(spec.encode_c(&args.C, outputLayout),
                   "Cake StepFun FC1: failed to encode the output tensor map");
  args.B = static_cast<uint8_t*>(hiddenState);
  args.SFB = static_cast<uint8_t*>(hiddenStateScale);
  args.SFC = static_cast<uint8_t*>(outputScale);
  args.route_map = permutedIdxToTokenIdx;
  args.tile_expert = ptrCtaIdxXyToBatchIdx;
  args.tile_mn_limit = ptrCtaIdxXyToMnLimit;
  args.scale_c = outputScalesScalar;
  args.scale_gate = outputScalesGateScalar;
  args.clamp_limit = ptrClampLimit;
  // The SwiGLU-Step epilogue forces alpha = 1 and beta = 0; the exported kernel reads neither.
  args.act_alpha = nullptr;
  args.act_beta = nullptr;
  args.M_out = intermediateSize;
  args.K = hiddenSize;
  args.grid_m = gridM;
  args.grid_n = gridN;
  args.K_tiles = kTiles;
  args.total_tiles = ptrNumNonExitingCtas;

  if (!mSmemConfigured[configIndex]) {
    cudaError_t const configured = spec.configure(spec.dynamic_smem_bytes);
    FLASHINFER_CHECK(configured == cudaSuccess,
                     "Cake StepFun FC1: cudaFuncSetAttribute(MaxDynamicSharedMemorySize=",
                     spec.dynamic_smem_bytes, ") failed for ", spec.symbol, ": ",
                     cudaGetErrorString(configured));
    mSmemConfigured[configIndex] = true;
  }

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(gridM), static_cast<unsigned>(gridN), 1u);
  config.blockDim = dim3(spec.block[0], spec.block[1], spec.block[2]);
  config.dynamicSmemBytes = spec.dynamic_smem_bytes;
  config.stream = stream;
  std::array<cudaLaunchAttribute, 1> attributes{};
  // Diagnostic override: CAKE_STEPFUN_FC1_PDL=0 launches the FC1 kernel without the programmatic
  // stream-serialization attribute, =1 launches it with the attribute, while the rest of the
  // pipeline keeps the caller's PDL setting; unset follows enable_pdl.
  static int const fc1PdlOverride = [] {
    char const* value = std::getenv("CAKE_STEPFUN_FC1_PDL");
    if (value == nullptr || value[0] == '\0' || value[1] != '\0') return -1;
    return value[0] == '0' ? 0 : value[0] == '1' ? 1 : -1;
  }();
  bool const fc1Pdl = fc1PdlOverride < 0 ? enable_pdl : fc1PdlOverride == 1;
  if (fc1Pdl) {
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = 1;
    config.attrs = attributes.data();
    config.numAttrs = 1;
  }
  cudaError_t const launched = spec.submit(&config, args);
  FLASHINFER_CHECK(launched == cudaSuccess, "Cake StepFun FC1 launch failed for ", spec.symbol,
                   " grid=(", gridM, ",", gridN, ") : ", cudaGetErrorString(launched));
}

}  // namespace cake_stepfun
}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
