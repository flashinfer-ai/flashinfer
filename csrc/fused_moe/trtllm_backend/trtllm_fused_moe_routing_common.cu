/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"

namespace moe::dev::routing {
namespace routingCustom {
// Forward declarations of launch functions
void launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream);
void launchDynBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream);
void launchClusterKernel(Data const& data, void* stream);
void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream);
void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream);
void launchHistogramKernel(Data const& data, int numBlocksHistogram, uint32_t numThreadsHist,
                           void* stream);
void launchOffsetsKernel(Data const& data, int numBlocksOffsets, uint32_t numThreadsHist,
                         void* stream);
}  // namespace routingCustom

////////////////////////////////////////////////////////////////////////////////////////////////////

// Implementation of shared post-topK pipeline for all routing methods.
// When topK is already computed (mPtrTopKIds or mPtrTopKPacked), we don't need
// routing-method-specific logic, so all methods can use the same workflow.
// This function handles all path selection: single-block, single-cluster, coop, multi-kernel.
template <typename DataType>
void runPostTopKPipeline(DataType const& data, void* stream) {
  // Convert to routingCustom::Data for launching (kernels are shared)
  routingCustom::Data customData;
  // Copy base fields
  static_cast<DataBase&>(customData) = static_cast<DataBase const&>(data);
  // Set routingCustom-specific defaults (not needed for utility kernels)
  customData.mDtypeOutput = data.mDtypeOutput;
  // The post-TopK kernels don't read routing logits (mPtrInput), only mPtrTopKPacked.
  // Set mDtypeInput = mDtypeOutput so the dispatched template is <OutputT, OutputT>,
  // avoiding an unnecessary mixed-type instantiation.
  customData.mDtypeInput = data.mDtypeOutput;
  // Softmax is chosen for its broad tier coverage, not because we need softmax.
  // The TopKIds/TopKPacked branches never call ExpertSelectPolicy::apply(),
  // so the postprocess is never executed.  Using Softmax avoids extra template
  // instantiations by reusing tiers already compiled for other models.
  customData.mPreprocessType = RoutingPreprocessType::None;
  customData.mPostprocessType = RoutingPostprocessType::Softmax;

  // Recompute numThreadsHist using routingCustom's expert tiers, since we launch custom kernels.
  // Different routing methods (DeepSeek, Llama4) may have different expert tier thresholds
  // that don't match routingCustom's tiers (128, 512, 2048).
  uint32_t const numThreadsHist =
      std::min(1024u, static_cast<uint32_t>(routingCustom::getMaxNumExperts(data.mNumExperts)));

  // Determine which path to use based on token count
  static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;
  bool const useStaticBlock = data.mNumTokens <= routingCustom::BlockKernelMaxNumTokens;
  // Use the dispatched tier size (not raw mNumExperts).
  // Example: 512 experts with topK=22 skips Tier<512,8> and lands on
  // Tier<1024,32>, so queryDispatchedMaxExperts() returns 1024 while
  // mNumExperts is 512.  The dynblock kernel sizes smem proportional to
  // maxExperts; using the raw count would exceed the smem budget.
  // Use customData (routingCustom::Data) since queryDispatchedMaxExperts
  // requires routingCustom::Data, not the template DataType.
  int32_t const dispatchedMaxExperts = routingCustom::queryDispatchedMaxExperts(customData);
  bool const useDynBlock = !useStaticBlock &&
                           data.mNumTokens <= routingCustom::DynBlockKernelMaxNumTokens &&
                           dispatchedMaxExperts <= routingCustom::DynBlockKernelMaxNumExperts;
  // runPostTopKPipeline only handles pre-computed topK (mPtrTopKIds or mPtrTopKPacked),
  // never raw scores. The cluster kernel's routingPermutation uses thread-per-expanded-index
  // for both input types (LoadExpertIdxFromGlobal=true), so the capacity is
  // NumBlocksPerCluster * NumThreads = 8192 tokens.
  bool const useSingleCluster =
      (smMajor >= 9) && (data.mNumTokens <= routingCustom::MaxNumTokensSingleCluster);

  if (useDynBlock) {
    routingCustom::launchDynBlockKernel(customData, numThreadsHist, stream);
  } else if (useStaticBlock) {
    routingCustom::launchBlockKernel(customData, numThreadsHist, stream);
  } else if (useSingleCluster) {
    routingCustom::launchClusterKernel(customData, stream);
  } else {
    // Check if we can use the coop path (more efficient for medium token counts)
    // Requires SM90+ (grid-sync), numExperts <= 1024.
    // Note: NumTop8Experts is used for template instantiation but does NOT limit runtime topK —
    // the coop kernel uses hardcoded MaxExpandedIdxPerThread=64 and runtime params.mTopK.
    bool const canUseCoop = (smMajor >= 9) && (data.mNumExperts <= 1024) &&
                            (data.mPtrPermutedIdxSize != nullptr) &&
                            (data.mPtrExpertCounts != nullptr);
    bool useCoop = false;
    int numBlocksCoop = 0;

    if (canUseCoop) {
      // Number of blocks we can use in the cooperative kernel
      static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
      // WAR: Reserve 8 SMs for overlapping kernels.
      numBlocksCoop = smCount - kReservedSMsForOverlapping;
      // Maximum number of tokens supported by the kernel using a cooperative launch.
      // The number of blocks must be:
      //   >= ⌈(numTokens * topK) / (MaxExpandedIdxPerThread * NumThreads)⌉
      // MaxExpandedIdxPerThread = 64 (from coop kernel)
      int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;
      useCoop = (data.mNumTokens <= maxTokensCoop);
    }

    if (useCoop) {
      // Coop path: cooperative launch fuses histogram + offsets (more efficient).
      // The coop kernel atomicAdds to mPtrExpertCounts, so we must zero it first.
      routingCustom::launchInitExpertCounts(customData, numThreadsHist, stream);
      routingCustom::launchCoopKernel(customData, numBlocksCoop, numThreadsHist, stream);
    } else {
      // Large-token path: multi-kernel pipeline
      FLASHINFER_CHECK(data.mPtrExpertCounts != nullptr,
                       "When #tokens is large, `mPtrExpertCounts` is a required input.");

      // Step 1: Reset expert counts
      routingCustom::launchInitExpertCounts(customData, numThreadsHist, stream);

      // Step 2-3: Histogram + Offsets
      int32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
      int32_t const histogramEltsPerBlock = 8 * numThreadsHist;
      int32_t const offsetEltsPerBlock = routing::NumEltsPerOffsetTilePerThread * numThreadsHist;
      int32_t const maxNumBlocks = 1024;

      int const numBlocksHistogram = std::min(
          (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
      int const numBlocksOffsets =
          std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

      routingCustom::launchHistogramKernel(customData, numBlocksHistogram, numThreadsHist, stream);
      routingCustom::launchOffsetsKernel(customData, numBlocksOffsets, numThreadsHist, stream);
    }
  }
}

// Explicit instantiations for the three routing method Data types
template void runPostTopKPipeline<routingCustom::Data>(routingCustom::Data const&, void*);
template void runPostTopKPipeline<routingDeepSeek::Data>(routingDeepSeek::Data const&, void*);
template void runPostTopKPipeline<routingLlama4::Data>(routingLlama4::Data const&, void*);

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace moe::dev::routing
