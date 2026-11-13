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
#include <algorithm>

#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"

namespace moe::dev::routing {
namespace routingCustom {
// Forward declarations of launch functions.
// Block/DynBlock/Cluster return whether a compiled tier covered (numExperts, topK)
// and the kernel was actually launched; the definitions live in the per-family
// routing translation units. Keep these signatures in sync (the return type is
// not part of the mangled name, so a mismatch would silently be UB).
bool launchBlockKernel(Data const& data, void* stream);
bool launchDynBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream);
bool launchClusterKernel(Data const& data, void* stream);
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
    bool const launched = routingCustom::launchDynBlockKernel(customData, numThreadsHist, stream);
    FLASHINFER_CHECK(
        launched, "runPostTopKPipeline: no compiled tier covers numExperts=", data.mNumExperts,
        " topK=", data.mTopK,
        " for the post-topK permutation (dyn-block path). Add a matching Tier<E, K> to "
        "PolicyTraits<NoOpPreprocess, SoftmaxPostprocess> in RoutingCustomPolicy.cuh.");
  } else if (useStaticBlock) {
    bool const launched = routingCustom::launchBlockKernel(customData, stream);
    FLASHINFER_CHECK(
        launched, "runPostTopKPipeline: no compiled tier covers numExperts=", data.mNumExperts,
        " topK=", data.mTopK,
        " for the post-topK permutation (static-block path). Add a matching Tier<E, K> "
        "to PolicyTraits<NoOpPreprocess, SoftmaxPostprocess> in RoutingCustomPolicy.cuh.");
  } else if (useSingleCluster) {
    bool const launched = routingCustom::launchClusterKernel(customData, stream);
    FLASHINFER_CHECK(launched,
                     "runPostTopKPipeline: no compiled tier covers numExperts=", data.mNumExperts,
                     " topK=", data.mTopK,
                     " for the post-topK permutation (single-cluster path). Add a matching "
                     "Tier<E, K> to (Cluster)PolicyTraits<NoOpPreprocess, SoftmaxPostprocess> in "
                     "RoutingCustomPolicy.cuh.");
  } else {
    // Check if we can use the coop path (more efficient for medium token counts)
    // Requires SM90+ (grid-sync), numExperts <= 1024.
    // Note: NumTop8Experts is used for template instantiation but does NOT limit runtime topK —
    // the coop kernel uses hardcoded MaxExpandedIdxPerThread=64 and runtime params.mTopK.
    bool const canUseCoop = (smMajor >= 9) && (data.mNumExperts <= 1024) &&
                            (data.mPtrPermutedIdxSize != nullptr) &&
                            (data.mPtrExpertCounts != nullptr);
    bool useCoop = false;
    CoopLaunchSMCounts coopLaunchSMCounts{0, 0};
    int numBlocksCoop = 0;

    if (canUseCoop) {
      // Number of blocks we can use in the cooperative kernel
      static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
      coopLaunchSMCounts = getCoopLaunchSMCounts(smCount);
      numBlocksCoop = coopLaunchSMCounts.moeSms;
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
      logCoopLaunchSMCounts(coopLaunchSMCounts);
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

namespace routingPrecomputed {

namespace {

constexpr int32_t kExpertTier256 = 256;
constexpr int32_t kExpertTier384 = 384;
constexpr int32_t kExpertTier512 = 512;
constexpr int32_t kMaxSupportedTopK = 8;

/// Return the compiled expert tier covering one routing problem, or zero when unsupported.
int32_t getMaxNumExpertsTier(int32_t numExperts) {
  if (numExperts <= topk::MaxNumExpertsUnit) {
    return topk::MaxNumExpertsUnit;
  }
  if (numExperts <= kExpertTier256) {
    return kExpertTier256;
  }
  if (numExperts <= kExpertTier384) {
    return kExpertTier384;
  }
  if (numExperts <= kExpertTier512) {
    return kExpertTier512;
  }
  FLASHINFER_WARN("Unsupported numExperts");
  return 0;
}

/// Value object passed by value to one fused launch for all tile-specific metadata outputs.
template <typename KernelParams, typename ExpertId, int MaxTiles = kMaxRoutingMetadataTiles>
struct MultiTileKernelArgs {
  // Fixed-capacity per-tile descriptors copied by value into the graph-stable launch.
  KernelParams params[MaxTiles];
  // Shared precomputed expert-ID input; null only for conventional packed entries.
  ExpertId const* expertIds;
  // Runtime prefix of params populated for this fused routing invocation.
  int32_t numTiles;
};

/// Run the permutation topology once per tile inside a single clustered CUDA kernel.
template <typename KernelParams, typename ExpertId>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1)
    __launch_bounds__(KernelParams::MaxNumExperts)
        routingIndicesMultiTileClusterKernel(MultiTileKernelArgs<KernelParams, ExpertId> args) {
  using OutputT = typename KernelParams::OutputT;

  int32_t const tileIdx = blockIdx.x / NumBlocksPerCluster;
  if (tileIdx >= args.numTiles) {
    return;
  }

  // Keep the divisor and power-of-two property tile-local so arbitrary tile mixtures retain
  // the existing routingPermutation arithmetic.
  KernelParams params = args.params[tileIdx];
  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const clusterBlockRank = blockIdx.x - tileIdx * NumBlocksPerCluster;

  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
  // Preserve the ordinary routing permutation implementation within each independent tile slice.
  routingPermutation<KernelParams, OutputT, KernelParams::MaxNumExperts,
                     KernelParams::MaxNumExperts / WarpSize, KernelParams::MaxNumTopExperts,
                     /*LoadExpertIdxFromGlobal=*/true, ExpertId>(params, nullptr, warpIdx,
                                                                 clusterBlockRank, args.expertIds);
}
#else
__global__ void routingIndicesMultiTileClusterKernel(MultiTileKernelArgs<KernelParams, ExpertId>) {
  assert(false && "routingIndicesMultiTileClusterKernel is only supported on SM90+ architectures");
}
#endif

/// Pack host-side tile descriptors and issue the extended clustered-kernel launch.
template <typename KernelParams, typename ExpertId>
void launchMultiTileClusterKernel(Data* data, int32_t numTiles, int32_t numBlocks,
                                  int32_t numThreads, int32_t smemSize, void* stream) {
  MultiTileKernelArgs<KernelParams, ExpertId> args{};
  args.numTiles = numTiles;
  args.expertIds = static_cast<ExpertId const*>(data[0].mPtrPrecomputedExpertIds);
  for (int32_t i = 0; i < numTiles; ++i) {
    args.params[i] = KernelParams::setKernelParams(data[i]);
  }

  cudaLaunchConfig_t config{};
  config.gridDim = numBlocks;
  config.blockDim = numThreads;
  config.dynamicSmemBytes = smemSize;
  config.stream = reinterpret_cast<cudaStream_t>(stream);

  cudaLaunchAttribute attributes[2] = {};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = int(data[0].mUsePdl);
  attributes[1].id = cudaLaunchAttributeCooperative;
  attributes[1].val.cooperative = 0;
  config.attrs = attributes;
  config.numAttrs = 2;

  auto kernelTyped = routingIndicesMultiTileClusterKernel<KernelParams, ExpertId>;
  if (smemSize > 48 * 1024) {
    CHECK_CUDA_ERROR(
        cudaFuncSetAttribute(kernelTyped, cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize));
  }
  CHECK_CUDA_ERROR(cudaLaunchKernelEx(&config, kernelTyped, args));
}

/// Launch one expert-count tier for either supported precomputed weight type.
template <int MaxNumExperts, typename ExpertId>
void launchMultiTileClusterTier(Data* data, int32_t numTiles, int32_t numBlocks, int32_t numThreads,
                                int32_t smemSize, void* stream) {
  if (data[0].mDtypeOutput == tg::Dtype::Bfloat16) {
    using Params = KernelParams<__nv_bfloat16, MaxNumExperts, kMaxSupportedTopK>;
    launchMultiTileClusterKernel<Params, ExpertId>(data, numTiles, numBlocks, numThreads, smemSize,
                                                   stream);
  } else {
    using Params = KernelParams<float, MaxNumExperts, kMaxSupportedTopK>;
    launchMultiTileClusterKernel<Params, ExpertId>(data, numTiles, numBlocks, numThreads, smemSize,
                                                   stream);
  }
}

/// Launch one expert tier with the exact precomputed expert-ID storage type.
template <int MaxNumExperts>
void launchMultiTileClusterExpertIds(Data* data, int32_t numTiles, int32_t numBlocks,
                                     int32_t numThreads, int32_t smemSize, void* stream) {
  if (data[0].mExpertIdType == ExpertIdType::Int16) {
    launchMultiTileClusterTier<MaxNumExperts, int16_t>(data, numTiles, numBlocks, numThreads,
                                                       smemSize, stream);
  } else {
    launchMultiTileClusterTier<MaxNumExperts, int32_t>(data, numTiles, numBlocks, numThreads,
                                                       smemSize, stream);
  }
}

/// Dispatch the common expert tiers used by fused multi-tile precomputed routing.
void launchMultiTileCluster(Data* data, int32_t numTiles, int32_t numBlocks, int32_t numThreads,
                            int32_t smemSize, void* stream) {
  FLASHINFER_CHECK(data[0].mNumExperts <= kExpertTier512,
                   "multi-tile precomputed routing currently supports numExperts <= ",
                   kExpertTier512, ", got ", data[0].mNumExperts);
  FLASHINFER_CHECK(data[0].mTopK <= kMaxSupportedTopK,
                   "multi-tile precomputed routing currently supports topK <= ", kMaxSupportedTopK,
                   ", got ", data[0].mTopK);
  FLASHINFER_CHECK(
      data[0].mDtypeOutput == tg::Dtype::Bfloat16 || data[0].mDtypeOutput == tg::Dtype::Fp32,
      "multi-tile precomputed routing supports BF16 or FP32 weights");
  FLASHINFER_CHECK(!data[0].mUsePdl,
                   "multi-tile precomputed routing captures as a normal graph node");

  if (data[0].mNumExperts <= topk::MaxNumExpertsUnit) {
    launchMultiTileClusterExpertIds<topk::MaxNumExpertsUnit>(data, numTiles, numBlocks, numThreads,
                                                             smemSize, stream);
  } else if (data[0].mNumExperts <= kExpertTier256) {
    launchMultiTileClusterExpertIds<kExpertTier256>(data, numTiles, numBlocks, numThreads, smemSize,
                                                    stream);
  } else if (data[0].mNumExperts <= kExpertTier384) {
    launchMultiTileClusterExpertIds<kExpertTier384>(data, numTiles, numBlocks, numThreads, smemSize,
                                                    stream);
  } else {
    launchMultiTileClusterExpertIds<kExpertTier512>(data, numTiles, numBlocks, numThreads, smemSize,
                                                    stream);
  }
}

}  // namespace

/// Return the token bound implied by the eight-block cluster and expert-count specialization.
int32_t maxTokensMultiTileCluster(int32_t numExperts) {
  return NumBlocksPerCluster * getMaxNumExpertsTier(numExperts);
}

/// Validate the common precomputed-routing shape and launch every tile as one fused kernel.
void runMultiTileCluster(Data* data, int32_t numTiles, void* stream) {
  // Validate the first tile as the shared input contract and establish launch bounds.
  FLASHINFER_CHECK(data != nullptr, "runMultiTileCluster requires non-null data");
  FLASHINFER_CHECK(numTiles > 0 && numTiles <= kMaxRoutingMetadataTiles, "numTiles must be in [1, ",
                   kMaxRoutingMetadataTiles, "], got ", numTiles);

  Data const& first = data[0];
  int32_t const maxTokens = maxTokensMultiTileCluster(first.mNumExperts);
  FLASHINFER_CHECK(first.mNumTokens <= maxTokens, "runMultiTileCluster supports up to ", maxTokens,
                   " tokens (NumBlocksPerCluster * MaxNumExperts), got ", first.mNumTokens);
  FLASHINFER_CHECK(first.mPtrTopKPacked != nullptr || first.mPtrPrecomputedExpertIds != nullptr,
                   "runMultiTileCluster requires precomputed top-k ids");
  FLASHINFER_CHECK(first.mPtrPermutedIdxSize != nullptr,
                   "runMultiTileCluster requires routing metadata output buffers");
  FLASHINFER_CHECK(first.mTileTokensDim > 0,
                   "multi-tile routing entries must have positive tile sizes");

  // Every fused tile may change tile-N and outputs, but must describe the same live routing input.
  for (int32_t i = 1; i < numTiles; ++i) {
    FLASHINFER_CHECK(data[i].mTileTokensDim > 0,
                     "multi-tile routing entries must have positive tile sizes");
    FLASHINFER_CHECK(data[i].mNumTokens == first.mNumTokens,
                     "all multi-tile routing entries must have the same num_tokens");
    FLASHINFER_CHECK(data[i].mNumExperts == first.mNumExperts,
                     "all multi-tile routing entries must have the same num_experts");
    FLASHINFER_CHECK(data[i].mTopK == first.mTopK,
                     "all multi-tile routing entries must have the same top_k");
    FLASHINFER_CHECK(data[i].mDtypeOutput == first.mDtypeOutput,
                     "all multi-tile routing entries must have the same weight dtype");
    FLASHINFER_CHECK(data[i].mExpertIdType == first.mExpertIdType,
                     "all multi-tile routing entries must have the same expert-ID dtype");
    FLASHINFER_CHECK(data[i].mPtrPrecomputedExpertIds == first.mPtrPrecomputedExpertIds,
                     "all multi-tile routing entries must share one expert-ID input");
    FLASHINFER_CHECK(data[i].mLocalExpertsStartIdx == first.mLocalExpertsStartIdx,
                     "all multi-tile routing entries must have the same local expert offset");
    FLASHINFER_CHECK(data[i].mLocalExpertsStrideLog2 == first.mLocalExpertsStrideLog2,
                     "all multi-tile routing entries must have the same local expert stride");
    FLASHINFER_CHECK(data[i].mNumLocalExperts == first.mNumLocalExperts,
                     "all multi-tile routing entries must have the same local expert count");
  }

  int32_t const numThreads = getMaxNumExpertsTier(first.mNumExperts);
  int32_t const numBlocks = NumBlocksPerCluster * numTiles;
  launchMultiTileCluster(data, numTiles, numBlocks, numThreads, /*smemSize=*/0, stream);
}

}  // namespace routingPrecomputed

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace moe::dev::routing
