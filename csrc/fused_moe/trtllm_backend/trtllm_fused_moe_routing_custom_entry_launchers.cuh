/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

// Implementation header included by exactly one routing entry translation unit in each JIT module.
// It owns the launchers shared by raw-score and precomputed routing, while score selection and the
// full routing entry point remain in trtllm_fused_moe_routing_custom.cuh.

#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
#include "tvm_ffi_utils.h"

namespace moe::dev::routing::routingCustom {

bool launchClusterKernelBlockDim256(Data const& data, void* stream);
bool launchClusterKernelBlockDim512(Data const& data, void* stream);
bool launchClusterKernelBlockDim1024(Data const& data, void* stream);

// Returns whether a compiled tier covered the runtime (numExperts, topK) and the
// kernel was actually launched (see launchBlockKernel).
bool launchClusterKernel(Data const& data, void* stream) {
  // Use the wider cluster only for permutation-only launches in the bounded high-expert/high-TopK
  // range. Score-to-TopK fused clusters retain the general capacity-based dispatch.
  bool const useWidePermutationCluster =
      data.mPtrScores == nullptr &&
      topk::isInHighExpertLaneOwnedTopKRange(data.mNumExperts, data.mTopK) &&
      data.mNumTokens >= 32 && data.mNumTokens <= 64;
  if (useWidePermutationCluster) {
    return launchClusterKernelBlockDim512(data, stream);
  }

  // Each warp owns one token, so the reduced-thread cluster variants have lower token capacity.
  // Use them only where the requested token count fits; otherwise keep the original 1024-thread
  // launch.
  constexpr int MaxNumTokensClusterScores256 =
      NumBlocksPerCluster * (NumExperts256Experts / WarpSize);
  constexpr int MaxNumTokensClusterScores512 =
      NumBlocksPerCluster * (NumExperts512Experts / WarpSize);
  if (data.mNumTokens <= MaxNumTokensClusterScores256) {
    return launchClusterKernelBlockDim256(data, stream);
  }
  if (data.mNumTokens <= MaxNumTokensClusterScores512) {
    return launchClusterKernelBlockDim512(data, stream);
  }
  return launchClusterKernelBlockDim1024(data, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Cooperative and multi-kernel post-TopK launchers.
//
// The coop kernel only performs the post-TopK permutation pipeline (histogram, prefix-scan,
// index writes). It does not compute TopK; it reads pre-computed results from mPtrTopKPacked
// or mPtrTopKIds. The expert tier sizes shared memory and determines the thread count. Most
// tiers use NumTop8Experts and generic 64-entry per-thread state. Bounded high-expert/high-TopK
// tiers can use NumTop16Experts with four entries per thread when that state covers
// mNumTokens * mTopK.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumExpertsTier>
void launchCoopKernelTier(Data const& data, int numBlocksCoop, uint32_t numThreadsHist,
                          void* stream) {
  bool useBoundedState = false;
  if constexpr (topk::isInHighExpertLaneOwnedTopKRange(NumExpertsTier, NumTop16Experts)) {
    int64_t const expandedIdxSize = int64_t{data.mNumTokens} * int64_t{data.mTopK};
    int64_t const boundedCapacity = int64_t{4} * int64_t{numBlocksCoop} * int64_t{numThreadsHist};
    useBoundedState = data.mTopK > NumTop8Experts && data.mTopK <= NumTop16Experts &&
                      expandedIdxSize <= boundedCapacity;
  }

  if (useBoundedState) {
    LAUNCH_ROUTING_WITH_POLICIES(
        data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
        /*smemSize=*/0, stream, NoOpPreprocess, NoOpPostprocess, NumExpertsTier, NumTop16Experts);
  } else {
    LAUNCH_ROUTING_WITH_POLICIES(
        data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop, numThreadsHist,
        /*smemSize=*/0, stream, NoOpPreprocess, NoOpPostprocess, NumExpertsTier, NumTop8Experts);
  }
}

void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream) {
  if (data.mNumExperts <= NumExperts128Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts128Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts160Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts160Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts256Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts256Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts384Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts384Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts512Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts512Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts576Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts576Experts, NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts896Experts) {
    launchCoopKernelTier<NumExperts896Experts>(data, numBlocksCoop, numThreadsHist, stream);
  } else if (data.mNumExperts <= NumExperts1024Experts) {
    launchCoopKernelTier<NumExperts1024Experts>(data, numBlocksCoop, numThreadsHist, stream);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "Coop kernel does not support numExperts > " << NumExperts1024Experts << ", got "
        << data.mNumExperts;
  }
}

void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingInitExpertCounts,
                                  (2 * data.mNumExperts - 1) / numThreadsHist + 1, numThreadsHist,
                                  /*smemSize=*/0,  // No dynamic smem
                                  stream);
}

void launchHistogramKernel(Data const& data, int numBlocksHistogram, uint32_t numThreadsHist,
                           void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingIndicesHistogramKernel, numBlocksHistogram,
                                  numThreadsHist,
                                  /*smemSize=*/0,  // No dynamic smem
                                  stream);
}

void launchOffsetsKernel(Data const& data, int numBlocksOffsets, uint32_t numThreadsHist,
                         void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingIndicesOffsetsKernel, numBlocksOffsets,
                                  numThreadsHist,
                                  /*smemSize=*/0,  // No dynamic smem
                                  stream);
}

}  // namespace moe::dev::routing::routingCustom
