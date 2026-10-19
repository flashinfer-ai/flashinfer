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
#pragma once

// Cake StepFun full-path stage runners (module built with -DCAKE_STEPFUN_FULL): routing, FC2, the
// NVFP4 per-token requantization and finalize over exported Cake kernels, mirroring the trtllm-gen
// host entry points they replace (Routing::Runner, Gemm2::Runner, invokeNvfp4QuantAndPerTokenScale,
// moe::dev::finalize::run) argument for argument so MoE::Runner and the launchers change only the
// type they name. The stage kernels come from the generated manifest tables declared through
// cake_stepfun_abi.cuh; a module whose inventory lacks a stage is never built with this macro (the
// JIT loader refuses it by stage name).
//
// This header is included from include/flashinfer/trtllm/fused_moe/runner.h after the Gemm2
// declarations it mirrors.

#ifndef CAKE_STEPFUN_FULL
#error "cake_stepfun_stages.cuh is part of the full-path Cake StepFun module (-DCAKE_STEPFUN_FULL)"
#endif

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "fused_moe/cake_stepfun/cake_stepfun_abi.cuh"

namespace tensorrt_llm {
namespace kernels {
namespace trtllmgen_moe {
namespace cake_stepfun {

// Routing stage. Same constructor and run() argument list as Routing::Runner; serves
// RoutingMethodType::Renormalize from float32 / bfloat16 logits or from pre-computed top-k ids
// (+ weights, the trtllm-gen launcher's unpacked pre-routed protocol) and rejects (does not fall
// back for) every other method, fused shared experts, routing replay output and the GEMM1 Mn bias
// row map. A variant is one kernel, or the two-kernel large-token sequence (histogram-scores
// kernel, then the cooperative kernel on the device's cooperative SM budget).
class RoutingRunner {
 public:
  explicit RoutingRunner(int32_t tileTokensDim) : mTileTokensDim(tileTokensDim) {}

  void run(void* routingLogits, void* routingBias, int32_t numTokens, int32_t numExperts,
           int32_t topK, int32_t numFusedSharedExpert, int32_t nGroup, int32_t topkGroup,
           int32_t localExpertOffset, int32_t localNumExperts, float routedScalingFactor,
           int32_t* routingExpertIndexes, int32_t* expertCountHistogram, int32_t* permutedIdxSize,
           int32_t* expandedIdxToPermutedIdx, int32_t* permutedIdxToExpandedIdx,
           int32_t* permutedIdxToTokenIdx, int32_t* expertIds, void* expertWeights,
           int32_t* numTokensPerExpert, int32_t* ctaIdxXyToBatchIdx, int32_t* ctaIdxXyToMnLimit,
           int32_t* numNonExitingCtas, batchedGemm::trtllm::gen::Dtype dtypeElt,
           batchedGemm::trtllm::gen::Dtype dtypeBias, bool useRoutingScalesOnInput,
           bool useDeepSeekFp8, Routing::RoutingMethodType routingMethodType, cudaStream_t stream,
           batchedGemm::trtllm::gen::Dtype dtypeLogits, bool normTopkProb,
           int16_t* routing_replay_out, bool enable_pdl);

 private:
  int32_t mTileTokensDim;
};

// FC2 stage. Same constructor, config-index queries and run() argument list as Gemm2::Runner.
class Fc2Runner {
 public:
  explicit Fc2Runner(batchedGemm::trtllm::gen::Dtype dtypeAct,
                     batchedGemm::trtllm::gen::Dtype dtypeWeights,
                     batchedGemm::trtllm::gen::Dtype outputDtype, bool useDeepSeekFp8,
                     int tileTokensDim, bool useShuffledMatrix,
                     batchedGemm::gemm::MatrixLayout weightLayout, bool usePerTokenScaling,
                     bool usePerChannelScaling);

  size_t getWorkspaceSizeInBytes(int32_t topK, int32_t hiddenSize, int32_t intermediateSize,
                                 int32_t numExperts, int32_t numTokens, int32_t configIndex) const;

  [[nodiscard]] int32_t getDefaultValidConfigIndex(int32_t topK, int32_t hiddenSize,
                                                   int32_t intermediateSize, int32_t numExperts,
                                                   int32_t numTokens) const;

  [[nodiscard]] bool isValidConfigIndex(int32_t configIndex, int32_t topK, int32_t hiddenSize,
                                        int32_t intermediateSize, int32_t numExperts,
                                        int32_t numTokens) const;

  [[nodiscard]] std::vector<int64_t> getPassingConfigIndices() const;

  // True when an exported Cake FC2 kernel serves this (family, tile).
  [[nodiscard]] bool hasKernels() const { return !mKernels.empty(); }

  // Generated-manifest family index of this runner (-1 when the dtype combination has no family).
  [[nodiscard]] int family() const { return mFamily; }

  // Block-scale layout the FC2 kernel of configIndex reads for its activation operand (what the
  // requantization stage must write for the per-token NVFP4 family).
  [[nodiscard]] flashinfer::cake_stepfun::generated::SfLayout sfLayoutA(int32_t configIndex) const;

  void run(void* permutedHiddenState, void* permutedHiddenStateScale, void* weight,
           void* weightScale, void* perTokenScales, void* perChannelScales,
           float* outputScalesScalar, float* ptrBias, void* output, void* outputScale, int32_t topK,
           int32_t hiddenSize, int32_t intermediateSize, int32_t numExperts, int32_t numTokens,
           int32_t* ptrNumNonExitingCtas, int32_t* ptrTotalNumPaddedTokens,
           int32_t* ptrCtaIdxXyToBatchIdx, int32_t* ptrCtaIdxXyToMnLimit, void* bmm2Workspace,
           int device, cudaStream_t stream, int32_t configIndex, bool enable_pdl,
           int32_t validIntermediateSize = -1, int32_t validHiddenSize = -1);

  // Read by MoE::Runner::run exactly as Gemm2::Runner's members of the same names.
  batchedGemm::trtllm::gen::Dtype mDtypeAct;
  batchedGemm::trtllm::gen::Dtype mDtypeWeights;
  batchedGemm::trtllm::gen::Dtype mDtypeOut;
  int32_t mTileTokensDim;

 private:
  bool shapeSupported(int32_t configIndex, int32_t hiddenSize, int32_t intermediateSize) const;

  int mFamily{-1};
  std::vector<int32_t> mKernels;
  mutable std::vector<bool> mSmemConfigured;
};

namespace requant {
// NVFP4 per-token requantization of the bf16 FC1 output into the FC2 input of the per-token
// family; same operands as invokeNvfp4QuantAndPerTokenScale<__nv_bfloat16> plus the block-scale
// layout the FC2 kernel reads and the recipe constants.
void run(int32_t numExpanded, int32_t innerDim, __nv_bfloat16 const* input, float globalScaleInv,
         float e4m3Max, int32_t const* expandedIdxToPermutedIdx, uint8_t* output,
         uint8_t* outputScale, float* perTokenScaleOut,
         flashinfer::cake_stepfun::generated::SfLayout layout, cudaStream_t stream, bool enablePdl);
}  // namespace requant

namespace finalize {
// Unpermute + top-k weighted sum; same Data as moe::dev::finalize::run and the same kernel-variant
// selection rule (scalar kernel below 1184 CTAs, vector-load kernel otherwise).
void run(moe::dev::finalize::Data const& data, cudaStream_t stream);
}  // namespace finalize

}  // namespace cake_stepfun
}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
