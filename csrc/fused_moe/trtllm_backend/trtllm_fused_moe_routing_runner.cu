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

// Routing::Runner dispatcher, split out of trtllm_fused_moe_runner.cu so it can
// be linked into the standalone trtllm_gen_routing module without pulling in
// the batched-GEMM runner (MoE::Runner) and its cubin dependencies.

#include <string>

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"

namespace tensorrt_llm {
namespace kernels {
namespace trtllmgen_moe {

namespace btg = batchedGemm::trtllm::gen;

namespace Routing {
namespace {
inline int32_t computeLog2(int32_t val, std::string const& name = "") {
  int32_t n = val;
  int32_t out = 0;
  while (n >>= 1) {
    ++out;
  }
  if ((1 << out) != val) {
    out = -1;
  }
  return out;
}
}  // namespace

Runner::Runner() {}

Runner::Runner(int32_t tileTokensDim) : mTileTokensDim(tileTokensDim) {}

void Runner::run(void* routingLogits, void* routingBias, int32_t numTokens, int32_t numExperts,
                 int32_t topK, int32_t numFusedSharedExpert, int32_t nGroup, int32_t topkGroup,
                 int32_t localExpertOffset, int32_t localNumExperts, float routedScalingFactor,
                 int32_t* routingExpertIndexes, int32_t* expertCountHistogram,
                 int32_t* permutedIdxSize, int32_t* expandedIdxToPermutedIdx,
                 int32_t* permutedIdxToExpandedIdx, int32_t* permutedIdxToTokenIdx,
                 int32_t* expertIds, void* expertWeights, int32_t* numTokensPerExpert,
                 int32_t* ctaIdxXyToBatchIdx, int32_t* ctaIdxXyToMnLimit,
                 int32_t* numNonExitingCtas, btg::Dtype dtypeElt, btg::Dtype dtypeBias,
                 bool useRoutingScalesOnInput, bool useDeepSeekFp8,
                 RoutingMethodType routingMethodType, cudaStream_t stream, btg::Dtype dtypeLogits,
                 bool normTopkProb, int16_t* routing_replay_out, bool enable_pdl) {
  if (routingMethodType == RoutingMethodType::DeepSeekV3 && nGroup <= 1) {
    // DeepSeek no-groups case: use routingCustom with SigmoidBias preprocess
    // and ScaledSumNormalize postprocess. This is more efficient than the full DeepSeek
    // kernel because it uses the warp-level routingTopKExperts flow.
    moe::dev::routing::routingCustom::Data routingData;

    routingData.mDtypeOutput = btg::Dtype::Bfloat16;
    routingData.mDtypeInput = dtypeLogits;
    routingData.mUsePdl = enable_pdl;
    routingData.mPreprocessType = moe::dev::routing::RoutingPreprocessType::SigmoidBias;
    routingData.mPostprocessType = moe::dev::routing::RoutingPostprocessType::ScaledSumNormalize;
    routingData.mPtrRoutingBias = routingBias;
    routingData.mDtypeBias = dtypeBias;
    routingData.mRouteScale = routedScalingFactor;
    // Floor the renorm denominator. ScaledSumNormalizePostprocess computes
    // sigmoid * routeScale / (sum + mSumEpsilon). If a token's top-K selected experts all have
    // strongly negative pre-bias logits, their sigmoids underflow to exactly 0.0 (bf16) so sum ==
    // 0, and mSumEpsilon's 0.0f default makes this 0/0 = NaN across the token's whole output row.
    // 1e-20f matches DeepSeek-V3's reference gate (modeling_deepseek.py: sum + 1e-20) and the
    // MiniMax2 branch.
    routingData.mSumEpsilon = 1e-20f;

    routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
    routingData.mPtrTopKIds = expertIds;
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mPtrRoutingReplayOut = routing_replay_out;

    moe::dev::routing::routingCustom::run(routingData, stream);
  } else if (routingMethodType == RoutingMethodType::MiniMax2) {
    // MiniMaxM2: sigmoid(logit) + bias → topK → renormalize un-biased sigmoid scores.
    // Similar to DeepSeek no-groups but with routeScale = 1.0 and epsilon = 1e-20
    // to match the Python reference: weight / (sum + 1e-20).
    moe::dev::routing::routingCustom::Data routingData;

    routingData.mDtypeOutput = btg::Dtype::Bfloat16;
    routingData.mDtypeInput = dtypeLogits;
    routingData.mUsePdl = enable_pdl;
    routingData.mPreprocessType = moe::dev::routing::RoutingPreprocessType::SigmoidBias;
    routingData.mPostprocessType = moe::dev::routing::RoutingPostprocessType::ScaledSumNormalize;
    routingData.mPtrRoutingBias = routingBias;
    routingData.mDtypeBias = dtypeBias;
    routingData.mRouteScale = routedScalingFactor;
    routingData.mSumEpsilon = 1e-20f;

    routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
    routingData.mPtrTopKIds = expertIds;
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mPtrRoutingReplayOut = routing_replay_out;

    moe::dev::routing::routingCustom::run(routingData, stream);
  } else if (routingMethodType == RoutingMethodType::DeepSeekV3) {
    FLASHINFER_CHECK(topK <= 22, "For DeepSeek routing method, must have topK <= 22");
    FLASHINFER_CHECK(topkGroup <= 4, "For DeepSeek routing method, must have topkGroup <= 4");
    moe::dev::routing::routingDeepSeek::Data routingData;
    routingData.mDtypeOutput =
        btg::Dtype::Bfloat16;               // for DeepSeek, the expW is currently always bfloat16
    routingData.mDtypeInput = dtypeLogits;  // routing logits can be bfloat16 or fp32
    routingData.mDtypeBias = dtypeBias;     // for DeepSeek, the bias can be bfloat16 or fp32
    routingData.mUsePdl = enable_pdl;

    int32_t const totalExpertsPerToken = topK + numFusedSharedExpert;

    // output:
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    // input:
    routingData.mPtrRoutingBias = routingBias;
    // Pre-computed routing support: when expertIds is provided, use it directly
    routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
    routingData.mPtrTopKIds = expertIds;
    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mNumFusedSharedExperts = numFusedSharedExpert;
    routingData.mNumExpertGroups = nGroup;
    routingData.mNumLimitedGroups = topkGroup;
    routingData.mTopK = topK;
    routingData.mTotalExpertsPerToken = totalExpertsPerToken;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mRouteScale = routedScalingFactor;
    routingData.mUseRoutingSoftmax = false;

    int32_t const numDevices = (localNumExperts > 0) ? numExperts / localNumExperts : 1;
    int32_t const deviceIndex = (localNumExperts > 0) ? localExpertOffset / localNumExperts : 0;
    int32_t const baseTokensPerDevice = numTokens / numDevices;
    int32_t const remainingTokens = numTokens % numDevices;

    if (deviceIndex < remainingTokens) {
      routingData.mSharedExpertTokenOffset = (baseTokensPerDevice + 1) * deviceIndex;
      routingData.mSharedExpertNumTokens = baseTokensPerDevice + 1;
    } else {
      routingData.mSharedExpertTokenOffset = remainingTokens + deviceIndex * baseTokensPerDevice;
      routingData.mSharedExpertNumTokens = baseTokensPerDevice;
    }
    routingData.mPtrRoutingReplayOut = routing_replay_out;
    moe::dev::routing::routingDeepSeek::run(routingData, stream);
  } else if (routingMethodType == RoutingMethodType::Llama4) {
    FLASHINFER_CHECK(numFusedSharedExpert == 0,
                     "Llama routing method does not support fusing shared expert");
    FLASHINFER_CHECK(topK == 1, "For Llama routing method, must have topK == 1");
    if (nGroup > 0 || topkGroup > 0) {
      FLASHINFER_WARN("For Llama routing method, nGroup/topkGroup is ignored, got ", nGroup, "/",
                      topkGroup);
    }
    moe::dev::routing::routingLlama4::Data routingData;
    routingData.mDtypeOutput = btg::Dtype::Bfloat16;
    routingData.mDtypeInput = dtypeLogits;  // routing logits can be bfloat16 or fp32
    routingData.mUsePdl = enable_pdl;

    // output:
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    // input:
    // Pre-computed routing support: when expertIds is provided, use it directly
    routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
    routingData.mPtrTopKIds = expertIds;
    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mPtrRoutingReplayOut = routing_replay_out;
    moe::dev::routing::routingLlama4::run(routingData, stream);
  } else if (routingMethodType == RoutingMethodType::Default        /* Softmax -> TopK */
             || routingMethodType == RoutingMethodType::Renormalize /* TopK -> Softmax */
             || routingMethodType ==
                    RoutingMethodType::RenormalizeNaive      /* Softmax -> TopK -> Renormalize */
             || routingMethodType == RoutingMethodType::TopK /* TopK only (no softmax) */
             || routingMethodType ==
                    RoutingMethodType::SigmoidRenorm /* Sigmoid -> TopK -> Renormalize */
             || routingMethodType == RoutingMethodType::Sigmoid /* Sigmoid -> TopK */) {
    FLASHINFER_CHECK(numFusedSharedExpert == 0,
                     "routingCustom method does not support fusing shared expert");
    using namespace moe::dev::routing;
    routingCustom::Data routingData;

    //
    // Config
    //

    routingData.mDtypeOutput = btg::Dtype::Bfloat16;
    routingData.mDtypeInput = dtypeLogits;  // routing logits can be bfloat16 or fp32
    routingData.mUsePdl = enable_pdl;

    // Map routing method types to policy-based routing:
    // Note: RenormalizeNaive (Softmax → TopK → SumNormalize) is mathematically equivalent
    // to Renormalize (TopK → Softmax), because taking softmax over all experts, selecting
    // top-K, and dividing by their sum produces the same result as applying softmax only
    // over the top-K values. We therefore use the same Renormalize implementation for both.
    if (routingMethodType == RoutingMethodType::Default) {
      // Softmax -> TopK (softmax on all scores, then select top-K)
      routingData.mPreprocessType = RoutingPreprocessType::Softmax;
      routingData.mPostprocessType = RoutingPostprocessType::None;
    } else if (routingMethodType == RoutingMethodType::SigmoidRenorm) {
      // Sigmoid -> TopK -> SumNormalize (renormalize)
      routingData.mPreprocessType = RoutingPreprocessType::Sigmoid;
      routingData.mPostprocessType = RoutingPostprocessType::SumNormalize;
      routingData.mNormTopkProb = normTopkProb;
    } else if (routingMethodType == RoutingMethodType::Sigmoid) {
      // Sigmoid -> TopK (no renormalization)
      routingData.mPreprocessType = RoutingPreprocessType::Sigmoid;
      routingData.mPostprocessType = RoutingPostprocessType::SumNormalize;
      routingData.mNormTopkProb = false;
    } else if (routingMethodType == RoutingMethodType::Renormalize ||
               routingMethodType == RoutingMethodType::RenormalizeNaive) {
      // TopK -> Softmax (also used for RenormalizeNaive, see comment above)
      routingData.mPreprocessType = RoutingPreprocessType::None;
      routingData.mPostprocessType = RoutingPostprocessType::Softmax;
    } else {
      // TopK only (no softmax or renormalize)
      routingData.mPreprocessType = RoutingPreprocessType::None;
      routingData.mPostprocessType = RoutingPostprocessType::None;
    }

    // Pre-computed routing support: when expertIds is provided, use it directly
    routingData.mPtrScores = expertIds == nullptr ? routingLogits : nullptr;
    routingData.mPtrTopKIds = expertIds;

    //
    // Outputs
    //
    routingData.mPtrTopKPacked = routingExpertIndexes;
    routingData.mPtrExpertCounts = expertCountHistogram;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = permutedIdxToExpandedIdx;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;

    //
    // Grouped Gemm Launch Config Buffers
    //
    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;

    //
    // Inputs
    //
    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = computeLog2(mTileTokensDim);
    routingData.mTileTokensDim = mTileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;
    routingData.mPtrRoutingReplayOut = routing_replay_out;

    routingCustom::run(routingData, stream);
  } else {
    FLASHINFER_CHECK(false, "Unimplemented routing method ",
                     serializeMoeRoutingMethodType(routingMethodType), " of enum ",
                     (int)routingMethodType);
  }
}
}  // namespace Routing

}  // namespace trtllmgen_moe
}  // namespace kernels
}  // namespace tensorrt_llm
