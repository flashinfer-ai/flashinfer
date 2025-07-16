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

#include "flashinfer/trtllm/fused_moe/RoutingKernel.cuh"

namespace moe::dev::routing {

namespace routingDeepSeek {

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 256;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int NumTopGroupScores = 2;
static constexpr int MaxNumTopExperts = 8;
static constexpr int MaxNumTopGroups = 4;

template <typename KernelParams>
__global__ void routingMainKernel(KernelParams params) {
  // declare types
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;

  // declare shared memory structure
  // number of experts is bounded by number of threads
  __shared__ float __attribute((aligned(128))) smemScoreSigmoid[NumThreads];
  __shared__ float __attribute((aligned(128))) smemScoreBias[NumThreads];
  // number of expert groups is bounded by number of warps
  __shared__ float __attribute((aligned(128))) smemGroupScores[NumWarps];

  // needed for warp reduce
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);
  // for the final reduction of weight norm, only some lanes need to participate
  int32_t laneIdx = threadIdx.x % WarpSize;
  int32_t warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  // warps outside the range of expert groups do not participate
  if constexpr (KernelParams::UseGroups) {
    if (warpIdx >= params.mNumExpertGroups) {
      return;
    }
  }

  // note that for invalid scores, we simply use a negative value:
  // they work well even with the compacted format used in topK, and
  // sigmoid / bias activated scores cannot be negative
  static constexpr float invalidScoreFloat = -1.F;
  const OutputT invalidScore = OutputT{invalidScoreFloat};

  // load bias already; each warp represents one expert group
  auto threadExpert = threadIdx.x;
  bool expertSelected = threadExpert < params.mNumExperts;
  if constexpr (KernelParams::UseGroups) {
    threadExpert = warpIdx * params.mNumExpertsPerGroup + laneIdx;
    expertSelected = laneIdx < params.mNumExpertsPerGroup;
  }
  auto scoreIdx = int64_t{blockIdx.x} * int64_t{params.mNumExperts} + threadExpert;
  auto biasVal = expertSelected ? params.mPtrRoutingBias[threadExpert] : invalidScore;

  // initialize the mPtrExpertCounts
  if (params.mPtrExpertCounts) {
    int32_t globalThreadIdx = blockIdx.x * NumThreads + threadIdx.x;
    int32_t globalThreadStride = gridDim.x * NumThreads;
    int32_t expertCountsNum = 2 * params.mNumExperts;
    initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  // trigger the secondary kernel when using PDL, then wait on primary
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
    cudaGridDependencySynchronize();
  }
#endif

  // get our assigned thread score; each warp represents one expert group
  float score =
      expertSelected ? static_cast<float>(params.mPtrScores[scoreIdx]) : invalidScoreFloat;
  // get the sigmoid score
  // note that for invalid values, we simply use a negative value:
  // sigmoig scores are always strictly positive
  auto scoreSigmoid = sigmoid_accurate(score);
  // write the sigmoid score to shared for later use
  if (expertSelected) {
    smemScoreSigmoid[threadExpert] = scoreSigmoid;
  }
  // get the score with bias
  // note that with invalid values, because sigmoid is < 1 and bias is -1,
  // we must get a negative value, which is smaller than any valid value
  auto scoreBias = float{scoreSigmoid + float{biasVal}};

  if (expertSelected) {
    smemScoreBias[threadExpert] = scoreBias;
  }

  // registers for top group score reduction
  float topExpGroupScores[NumTopGroupScores];
  [[maybe_unused]] int32_t topExpGroupIdx[NumTopGroupScores];
  float topGroups[MaxNumTopGroups];  // bound of params.mNumLimitedGroups
  int32_t topGroupIdx[MaxNumTopGroups];
  float expertScoreGroup[MaxNumTopGroups];
  int32_t expertIdxGroup[MaxNumTopGroups];
  float topScores[MaxNumTopExperts];  // bound of params.mTopK
  int32_t topExperts[MaxNumTopExperts];

  if constexpr (KernelParams::UseGroups) {
    topk::reduceTopK(warp, topExpGroupScores, topExpGroupIdx, scoreBias, threadExpert,
                     /* minValue */ invalidScoreFloat);

    // get the final group score and write it to shared
    if (cute::elect_one_sync()) {
      auto groupScore = topExpGroupScores[0] + topExpGroupScores[1];
      smemGroupScores[warpIdx] = groupScore;
    }
  }

  // make group scores available to all warps
  __syncthreads();

  auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
  if (warpIdx == 0) {
    // a single warp performs the selection of top groups, and goes on to select the final experts
    if constexpr (KernelParams::UseGroups) {
      float groupScore =
          laneIdx < params.mNumExpertGroups ? smemGroupScores[laneIdx] : invalidScoreFloat;

      topk::reduceTopK(warp, topGroups, topGroupIdx, groupScore, laneIdx,
                       /* minValue */ invalidScoreFloat);

      // final expert selection: get relevant indexes and scores from shared

#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {  // bound of params.mNumLimitedGroups
        auto groupIdx = topGroupIdx[ii];
        expertIdxGroup[ii] = groupIdx * params.mNumExpertsPerGroup + laneIdx;
        // note: expertSelected implies laneIdx < params.mNumExpertsPerGroup.
        // we have params.mNumExpertsPerGroup == params.mNumExperts / params.mNumExpertGroups,
        // thus groupIdx <= params.mNumExpertGroups - 1 =>
        // groupIdx * params.mNumExpertsPerGroup <= params.mNumExperts - params.mNumExpertsPerGroup
        // => expertIdxGroup[ii] < params.mNumExperts <= NumThreads,
        // so the access is safe here
        expertScoreGroup[ii] = groupIdx < params.mNumExpertGroups && expertSelected
                                   ? smemScoreBias[expertIdxGroup[ii]]
                                   : invalidScoreFloat;
      }
    } else {
      // without groups, each thread just takes `MaxNumTopGroups` experts

#pragma unroll
      for (int ii = 0; ii < MaxNumTopGroups; ++ii) {
        auto expertIdx = ii * WarpSize + laneIdx;
        expertIdxGroup[ii] = expertIdx;
        expertScoreGroup[ii] =
            expertIdx < params.mNumExperts ? smemScoreBias[expertIdx] : invalidScoreFloat;
      }
    }

    topk::reduceTopK(warp, topScores, topExperts, expertScoreGroup, expertIdxGroup,
                     /* minValue */ invalidScoreFloat, params.mTopK);

    // determine our lane's expert index and write to output
    int32_t expertIdx = 0;
#pragma unroll
    for (int ii = 0; ii < params.mTopK; ++ii) {  // bound of params.mTopK
      expertIdx = laneIdx == ii ? topExperts[ii] : expertIdx;
    }
    // determine whether our expert is local to this GPU
    auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                         (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;

    float scoreNorm = laneIdx < params.mTopK ? smemScoreSigmoid[expertIdx] : 0.F;
    auto redNorm = cg::reduce(warp, scoreNorm, cg::plus<float>{});
    auto finalScore = OutputT{scoreNorm * params.mRouteScale / redNorm};

    // write expert idx out already
    auto idxTopK = blockIdx.x * params.mTopK + laneIdx;
    if (laneIdx < params.mTopK && params.mPtrExpertIdx != nullptr) {
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(finalScore),
                                          static_cast<int16_t>(expertIdx)};
      params.mPtrExpertIdx[idxTopK] = packedScore;
    }

    if (laneIdx < params.mTopK && params.mPtrExpertWeights != nullptr) {
      params.mPtrExpertWeights[idxTopK] = finalScore;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const clusterBlockRank = blockIdx.x;

  //@todo: try to move it into routingPermutation
  // then wait on primary grid
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }

  routingPermutation<KernelParams, OutputT, NumThreads, NumWarps, MaxNumTopExperts,
                     /*LoadExpertIdxFromGlobal=*/true>(params, nullptr, warpIdx, clusterBlockRank);
}
#else
__global__ void routingIndicesClusterKernel(KernelParams params) {
  assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(NumThreads) routingIndicesCoopKernel(KernelParams params) {
  // number of experts is bounded by number of threads
  __shared__ int32_t __attribute((aligned(128))) smemExpertCount[NumThreads];
  __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[NumThreads];
  // needed for the exclusive sum of token offsets
  using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename Scan::TempStorage tempStorage;
  // 64 elements -> 128+ registers. Above that we may start to see spilling to local memory.
  static constexpr int MaxExpandedIdxPerThread = 64;

  // Initialize grid.
  cg::grid_group grid = cg::this_grid();
  // Note: the following is more efficient than grid.block_index() because we don't use y and z.
  int32_t const gridBlockIdx = blockIdx.x;
  int32_t const gridThreadIdx = NumThreads * gridBlockIdx + threadIdx.x;
  int32_t const numBlocks = gridDim.x;
  int32_t const numThreadsPerGrid = numBlocks * NumThreads;

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

  auto expandedIdxSize = params.mNumTokens * params.mTopK;

  // pre-fill the counts with 0
  smemExpertCount[threadIdx.x] = 0;
  __syncthreads();

  // then wait on primary grid
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }

  // each thread keeps has some number of "expanded indexes" assigned to it
  // for each of these, we keep the associated expert and offset within expert in registers
  int32_t expertIndexes[MaxExpandedIdxPerThread];
  int32_t expertOffsets[MaxExpandedIdxPerThread];
  auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
  // In order to avoid a serialization LDG-ATOMS-LDG-ATOMS-..., we skip multiple iterations at a
  // time, and branch between a fast path without bound checks and a slow path with bound checks.
  int constexpr IterStride = 4;
  static_assert(MaxExpandedIdxPerThread % IterStride == 0);

  // Define a lambda to avoid code duplication in both branches.
  auto loopBody = [&](int ii, int expandedIdx) {
    int32_t expertIdx = params.mPtrExpertIdx[expandedIdx].idx;
    expertIndexes[ii] = expertIdx;
    // check whether this expert is local to our GPU at all and ignore if not
    auto localExpertIdx = expertIdx - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                         (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
    expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + expertIdx, 1) : 0;
  };

#pragma unroll
  for (int32_t ii0 = 0; ii0 < MaxExpandedIdxPerThread; ii0 += IterStride) {
    // Whether it's safe to do multiple iterations without bound checks.
    bool const takeFastPath = (ii0 + IterStride) * numThreadsPerGrid <= expandedIdxSize;
    if (takeFastPath) {
#pragma unroll
      for (int32_t jj = 0; jj < IterStride; jj++) {
        int const ii = ii0 + jj;
        auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
        loopBody(ii, expandedIdx);
      }
    } else {
      bool doBreak = false;
#pragma unroll
      for (int32_t jj = 0; jj < IterStride; jj++) {
        int const ii = ii0 + jj;
        auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
        if (expandedIdx >= expandedIdxSize) {
          doBreak = true;
          break;
        }
        loopBody(ii, expandedIdx);
      }
      if (doBreak) {
        break;
      }
    }
  }

  // Make histogram (token counts per expert) available to all threads in the block.
  __syncthreads();

  //
  // Each thread now represents one expert
  //

  // Add the local bin count to the common bin count and get a per-CTA offset.
  int32_t const localExpertCount = smemExpertCount[threadIdx.x];

  int32_t blockExpertOffset = 0;
  if (threadIdx.x < params.mNumExperts) {
    blockExpertOffset = atomicAdd(&params.mPtrExpertCounts[threadIdx.x], localExpertCount);
  }

  // Sync to wait for completion of the histogram reduction.
  grid.sync();

  // Get total count for this expert.
  int32_t count = (threadIdx.x < params.mNumExperts) ? params.mPtrExpertCounts[threadIdx.x] : 0;

  // Note: the scan is redundant in all CTAs, but doing it in only 1 CTA would be worse for latency.

  // Compute the runtime config for projections
  // Whether or not an expert is local is taken into account when smemExpertCount is computed
  // so we do not need to take it into account here.
  const int32_t numCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
  int32_t ctaOffset;
  int32_t numNonExitingCtas;
  Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

  for (int32_t cta = gridBlockIdx; cta < numCta; cta += numBlocks) {
    const int32_t localExpertIdx =
        (threadIdx.x - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
    params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = localExpertIdx;
    params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] =
        min(mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2),
            mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + count);
  }

  // get the padded offset associated with this expert
  const int32_t offset = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
  const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);

  // write out padded count
  if (gridBlockIdx == 0 && warpIdx == NumWarps - 1 && cute::elect_one_sync()) {
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

  // write expert offsets to shared
  smemExpertOffset[threadIdx.x] = offset + blockExpertOffset;

  // make expert offsets available to all threads
  __syncthreads();

  // trigger the secondary kernel when using PDL
  // We can't do it earlier because FC1 depends on the mPtrCtaIdxXyToBatchIdx,
  // mPtrCtaIdxXyToMnLimit, mPtrNumNonExitingCtas and mPtrTotalNumPaddedTokens
  // TODO: this is not sufficient to ensure visibility in the next kernel!
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }

// each thread has the same "expanded indexes" assigned to it as above
// at this point, we know the final offsets of experts and the offsets within
// experts, which allows writing the final index values
#pragma unroll
  for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ++ii) {
    auto expandedIdx = static_cast<int32_t>(gridThreadIdx) + ii * numThreadsPerGrid;
    if (expandedIdx >= expandedIdxSize) {
      break;
    }
    auto expertIdx = expertIndexes[ii];
    // check whether this expert is local to our GPU at all
    auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                         (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
    auto tokenIdx = expandedIdx / params.mTopK;
    auto permutedIdx =
        isLocalExpert ? int32_t{smemExpertOffset[expertIdx]} + expertOffsets[ii] : int32_t{-1};
    if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
      params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
    }
    if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert) {
      params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
    }
  }
}
#else
__global__ void routingIndicesCoopKernel(KernelParams params) {
  assert(false && "routingIndicesCoopKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data& data, void* stream) {
  TORCH_CHECK(data.mPtrExpertIdx != nullptr || data.mPtrPermutedIdxSize != nullptr ||
                  data.mPtrExpertWeights != nullptr,
              "Routing kernel requires at least one output parameter");
  if (data.mPtrExpandedIdxToPermutedIdx != nullptr || data.mPtrPermutedIdxToTokenIdx != nullptr) {
    TORCH_CHECK(data.mPtrExpertIdx != nullptr && data.mPtrPermutedIdxSize,
                "If permuted index is required, `mPtrExpertIdx` is also required");
  }
  TORCH_CHECK(!data.mUseRoutingSoftmax, "Routing with softmax not implemented yet");
  TORCH_CHECK(data.mNumLimitedGroups <= MaxNumTopGroups,
              "Routing kernel expects <= %d top groups, got %d", MaxNumTopGroups,
              data.mNumLimitedGroups);
  TORCH_CHECK(data.mTopK <= MaxNumTopExperts, "Routing kernel expects topK experts <= %d, got %d",
              MaxNumTopExperts, data.mTopK);
  TORCH_CHECK(data.mTopK <= WarpSize, "Routing kernel expects top K <= warp size, got %d",
              data.mTopK);
  TORCH_CHECK(data.mTopK * data.mNumLimitedGroups <= WarpSize,
              "Routing kernel expects top K * top groups <= warp size (for now), got %d * %d",
              data.mTopK, data.mNumLimitedGroups);
  TORCH_CHECK(data.mNumExperts >= MaxNumTopExperts,
              "Routing kernel expects %d to be at most #experts %d", MaxNumTopExperts,
              data.mNumExperts);
  TORCH_CHECK(data.mNumExperts <= NumThreads, "Routing kernel expects #experts %d  <= #threads %d",
              data.mNumExperts, NumThreads);
  TORCH_CHECK(data.mNumExpertGroups >= data.mNumLimitedGroups,
              "Routing kernel expects top groups %d to be limited by #expert groups %d",
              data.mNumLimitedGroups, data.mNumExpertGroups);
  if (data.mNumExpertGroups > 1) {
    TORCH_CHECK(data.mNumExpertGroups <= NumWarps,
                "Routing kernel expects #experts groups %d to be <= #warps %d",
                data.mNumExpertGroups, NumWarps);
    TORCH_CHECK(data.mNumExperts % data.mNumExpertGroups == 0,
                "Routing kernel expects #experts %d to be a multiple of #expert groups %d",
                data.mNumExperts, data.mNumExpertGroups);
    TORCH_CHECK(data.mNumExperts / data.mNumExpertGroups <= WarpSize,
                "Routing kernel expects #experts per group <= warp size, got %d",
                data.mNumExperts / data.mNumExpertGroups);
  } else {
    TORCH_CHECK(data.mNumExperts <= WarpSize * MaxNumTopGroups,
                "Routing kernel expects #experts %d <= WarpSize * MaxNumTopGroups %d",
                data.mNumExperts, WarpSize * MaxNumTopGroups);
    TORCH_CHECK(data.mTopK <= NumWarps, "Routing kernel expects top K %d to be <= #warps %d",
                data.mTopK, NumWarps);
  }
  TORCH_CHECK(data.mNumExperts % 4 == 0,
              "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);
  TORCH_CHECK(data.mPaddingLog2 < 8, "Routing kernel expects padding log2 < 8, got %d",
              data.mPaddingLog2);
  int const numBlocks = data.mNumTokens;

  bool const useSingleCluster = data.mNumTokens <= 1024;
  if (!useSingleCluster) {
    // Reset the global histograms (not used in single-cluster code path).
    // Cover both for the cooperative and two-kernel code paths.
    TORCH_CHECK(data.mPtrExpertCounts != nullptr,
                "When #tokens is large, `mPtrExpertCounts` is a required input.");
  } else {
    data.mPtrExpertCounts =
        nullptr;  // Set it to nullptr for single-cluster code path, as it won't be used
  }

  // Number of blocks we can use in the cooperative kernel
  // The number of blocks must be:
  //   >= ⌈(numTokens * topK) / (MaxExpandedIdxPerThread * NumThreads)⌉
  //   <= numSms, assuming an occupancy of 1 block/SM
  //
  // If too small for the given numTokens, fall back to the less performant two-step method.
  //
  // The upper bound is a strict requirement. The number of blocks should be determined by querying
  // the device properties, or conservatively low.
  // /!\ The following number is not portable!! (but works on H100 and B200)
  int const numBlocksCoop = 128;

  // Maximum number of tokens supported by the kernel using a cooperative launch.
  int const maxTokensCoop = (numBlocksCoop * NumThreads * 64) / data.mTopK;
  LAUNCH_ROUTING_WITH_EXTRA_FLAG(data,
                                 /*coopLaunch=*/false, routingMainKernel, numBlocks, NumThreads,
                                 /*smemSize=*/0,  // No dynamic smem
                                 stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);

  if (data.mPtrPermutedIdxSize != nullptr) {
    if (useSingleCluster) {
      LAUNCH_ROUTING_WITH_EXTRA_FLAG(data,
                                     /*coopLaunch=*/false, routingIndicesClusterKernel,
                                     NumBlocksPerCluster, NumThreads,
                                     /*smemSize=*/0,  // No dynamic smem
                                     stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
    } else if (data.mNumTokens <= maxTokensCoop) {
      LAUNCH_ROUTING_WITH_EXTRA_FLAG(data,
                                     /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                     NumThreads,
                                     /*smemSize=*/0,  // No dynamic smem
                                     stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
    } else {
      const int32_t expandedIdxSize = data.mNumTokens * data.mTopK;

      const int32_t histogramEltsPerBlock = 8 * NumThreads;
      const int32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * NumThreads;

      // Limit grid size (both kernels use a grid-stride loop).
      const int32_t maxNumBlocks = 1024;

      int const numBlocksHistogram = std::min(
          (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
      int const numBlocksOffsets =
          std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

      LAUNCH_ROUTING_WITH_EXTRA_FLAG(data,
                                     /*coopLaunch=*/false, routingIndicesHistogramKernel,
                                     numBlocksHistogram, NumThreads,
                                     /*smemSize=*/0,  // No dynamic smem
                                     stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
      LAUNCH_ROUTING_WITH_EXTRA_FLAG(data,
                                     /*coopLaunch=*/false, routingIndicesOffsetsKernel,
                                     numBlocksOffsets, NumThreads,
                                     /*smemSize=*/0,  // No dynamic smem
                                     stream, data.mNumExpertGroups > 1, /*forceFloatInput=*/true);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingDeepSeek

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingLlama4 {

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int MaxNumTopExperts = 1;
static constexpr int MaxNumExperts = 128;
static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;
static constexpr int WarpKernelSmemStride = 33;
// with further optimization to `routingIndicesWarpKernel`, this limit may
// increase. For now, it is a good cut-off point for when the block-wise
// operations are more efficient end-to-end.
static constexpr int WarpKernelMaxNumTokens = 4;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType, int VecSize>
__forceinline__ __device__ void routingTopKExperts(cg::thread_block_tile<WarpSize> const& warp,
                                                   DataType (&warpMaxScore)[MaxNumTopExperts],
                                                   int32_t (&warpMaxExpertIdx)[MaxNumTopExperts],
                                                   int32_t const laneIdx, int32_t const numExperts,
                                                   DataType const* ptrScores) {
  DataType minScore = DataType{-INFINITY};
  DataType maxScore = minScore;
  int32_t maxExpertIdx{-1};
  using DataTypeVec = std::conditional_t<sizeof(DataType) == 2, float2, float4>;

  // Non-vectorized loading: directly access ptrScores with expertIdx
  for (int i = 0; i < VecSize; ++i) {
    auto expertIdx = i * WarpSize + laneIdx;
    auto newScore = expertIdx < numExperts ? ptrScores[expertIdx] : minScore;
    // note: use `>=` s.t. highest index always wins, just like in `reduceTopK`
    if (newScore > maxScore) {
      maxScore = newScore;
      maxExpertIdx = expertIdx;
    }
  }

  topk::reduceTopK(warp, warpMaxScore, warpMaxExpertIdx, maxScore, maxExpertIdx, minScore);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(WarpSize) routingIndicesWarpKernel(KernelParams params) {
  // types used in this kernel
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using TypePacked = PackedScoreIdx<OutputT>;
  // use the default cub warp-scan, with shfl
  using Scan = cub::WarpScan<int32_t>;
  __shared__ typename Scan::TempStorage tempStorage;

  // each thread encodes 4 experts in one `int32_t`. The assumption is that
  // we don't have more than 127 tokens, but `WarpKernelMaxNumTokens` must be
  // smaller than that because other approaches will be more efficient for
  // 127 tokens.
  static constexpr int ExpertsPerThread = sizeof(int32_t);
  static_assert(WarpKernelMaxNumTokens <= 127);
  // this is a full table of which token is routed to which expert.
  // the assumption here is that there are no more than 128 experts.
  // we use a stride of 33 instead of 32 to avoid shared memory bank conflicts.
  __shared__ int32_t __attribute((
      aligned(128))) smemExpertTokenCountFull[WarpKernelMaxNumTokens][WarpKernelSmemStride];
  static_assert(WarpKernelSmemStride == WarpSize + 1);
  static_assert(MaxNumExperts / sizeof(int32_t) <= WarpSize);

  // values needed for the top-1 reduction, if required
  InputT minScore = InputT{-INFINITY};
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

#pragma unroll
  for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens; ++tokenIdx) {
    // reset full shared memory field to 0
    smemExpertTokenCountFull[tokenIdx][threadIdx.x] = 0;
  }
  __syncwarp();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // then wait on primary grid
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  if (params.mPtrScores != nullptr) {
    // if we use `mPtrScores` as input, we need to perform the top-1 reduction
    // for each token, we load the scores then use `reduceTopK` for this.
    // each thread works on 4 experts, so a local reduction is done before
    for (int tokenIdx = 0; tokenIdx < params.mNumTokens; ++tokenIdx) {
      auto scoreOffset = tokenIdx * params.mNumExperts;
      int32_t warpMaxExpertIdx[MaxNumTopExperts];
      InputT warpMaxScore[MaxNumTopExperts];

      // Use routingTopKExperts function instead of inline logic
      routingTopKExperts<InputT, ExpertsPerThread>(warp, warpMaxScore, warpMaxExpertIdx,
                                                   threadIdx.x, params.mNumExperts,
                                                   params.mPtrScores + scoreOffset);

      if (cute::elect_one_sync()) {
        // one thread updates the count linking token to chosen expert
        auto expertTokenCount = 0;
        setBits</* IsZero= */ true>(expertTokenCount, 1, warpMaxExpertIdx[0] % ExpertsPerThread);
        smemExpertTokenCountFull[tokenIdx][warpMaxExpertIdx[0] / ExpertsPerThread] =
            expertTokenCount;
        // we also compute the final score here and write it out if required
        auto finalScore = OutputT{sigmoid_accurate(float{warpMaxScore[0]})};
        if (params.mPtrExpertWeights != nullptr) {
          params.mPtrExpertWeights[tokenIdx] = finalScore;
        }
      }
    }
  } else {
    // if we do not have `mPtrScores` as input, we expect that `mPtrExpertWeights`
    // contains the top-1 packed score and index already.
    // Each thread represents a token here, and we extract the relevant score
    // The assumption is that the #tokens is limited by warp-size
    static_assert(WarpKernelMaxNumTokens <= WarpSize);
    TypePacked scoreIdx =
        threadIdx.x < params.mNumTokens ? params.mPtrExpertIdx[threadIdx.x] : TypePacked{};
    int32_t expertTokenCount = 0;
    setBits</* IsZero= */ true>(expertTokenCount, 1, scoreIdx.idx % ExpertsPerThread);
    if (threadIdx.x < params.mNumTokens) {
      smemExpertTokenCountFull[threadIdx.x][scoreIdx.idx / ExpertsPerThread] = expertTokenCount;
    }
    // we also compute the final score here and write it out if required
    auto finalScore = OutputT{sigmoid_accurate(float{scoreIdx.score})};
    if (params.mPtrExpertWeights != nullptr && threadIdx.x < params.mNumTokens) {
      params.mPtrExpertWeights[threadIdx.x] = finalScore;
    }
  }

  // make the full table available to all threads
  __syncwarp();

  // at this point, each thread keeps a count of its 4 assigned experts in
  // `expertCount`, as well as the offsets for all tokens w.r.t. these 4 experts
  // in `expertOffset`.
  int32_t expertCount = 0;
  int32_t expertOffset[WarpKernelMaxNumTokens + 1];
#pragma unroll
  for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens + 1; ++tokenIdx) {
    if (tokenIdx > params.mNumTokens) break;
    // simple reduction for `expertCount`, and scan for `expertOffset`
    auto expertTokenCount =
        tokenIdx < params.mNumTokens ? smemExpertTokenCountFull[tokenIdx][threadIdx.x] : 0;
    expertOffset[tokenIdx] = expertCount;
    expertCount += expertTokenCount;
  }

  // at this point, we are ready for the scan across all experts to get the
  // thread-wise offsets across experts
  // first, we need to reduce across our 4 experts into `numCta`
  int32_t numCta = 0;
#pragma unroll
  for (int ii = 0; ii < ExpertsPerThread; ++ii) {
    auto count = getBits(expertCount, ii);
    numCta += divUpLog2<int32_t>(count, params.mPaddingLog2);
  }
  // second, we perform the exclusive sum across the warp
  int32_t ctaOffset;
  int32_t numNonExitingCtas;
  Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

  // finally, we perform a scan across our local experts, starting with the
  // warp-wide scan result (`ctaOffset`)
  auto ctaOffsetExp = ctaOffset;
#pragma unroll
  for (int ii = 0; ii < ExpertsPerThread; ++ii) {
    auto count = getBits(expertCount, ii);
    auto finalNumCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
    auto expertIdx = threadIdx.x * ExpertsPerThread + ii;
    // during the scan for expert offsets, we can already write out
    // both `mPtrCtaIdxXyToBatchIdx` and `mPtrCtaIdxXyToMnLimit`
    for (int cta = 0; cta < finalNumCta; ++cta) {
      params.mPtrCtaIdxXyToBatchIdx[ctaOffsetExp + cta] = expertIdx;
      params.mPtrCtaIdxXyToMnLimit[ctaOffsetExp + cta] =
          min(mulLog2<int32_t>(ctaOffsetExp + cta + 1, params.mPaddingLog2),
              mulLog2<int32_t>(ctaOffsetExp, params.mPaddingLog2) + count);
    }
    ctaOffsetExp += finalNumCta;
  }

  // at this point, we can write out padded count from the warp-aggregate
  if (cute::elect_one_sync()) {
    const int32_t permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
#if !defined(PDL_PROFILE) || PDL_PROFILE == 0
  // we can trigger the next kernel at this point
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
#endif

  // at this point, all values for offsets are ready, except the final offsets
  // within the padded index (`permutedIdx`)
  // for this, we perform a scan similar to the one directly after the warp-scan:
  // here, we keep the local offset for each of the thread's experts in a field
  // of registers
  auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
  int32_t finalExpertOffset[ExpertsPerThread];
  finalExpertOffset[0] = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
#pragma unroll
  for (int ii = 1; ii < ExpertsPerThread; ++ii) {
    finalExpertOffset[ii] =
        finalExpertOffset[ii - 1] +
        divUpMulLog2<int32_t>(getBits(expertCount, ii - 1), params.mPaddingLog2);
  }

#pragma unroll
  for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens; ++tokenIdx) {
    // at this point, we can calculate the final index:
    // we simply loop over all tokens, and all experts assigned to this thread.
    // For each pair, we determine whether that token was routed to that expert
    // based on whether the offset for that token changed.
    // we can then easily compute the final `expertIdx` and `permutedIdx` relative
    // to this token and expert, and write them out.
    if (tokenIdx >= params.mNumTokens) break;

#pragma unroll
    for (int ii = 0; ii < ExpertsPerThread; ++ii) {
      // determine whether the offset for this expert and token changes
      auto localOffsetToken = getBits(expertOffset[tokenIdx], ii);
      auto isTokenRouted = getBits(expertOffset[tokenIdx + 1], ii) > localOffsetToken;
      // the expert index of this expert
      auto expertIdx = threadIdx.x * ExpertsPerThread + ii;
      auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
      auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                           (localExpertIdx & params.mLocalExpertsStrideLog2) == 0;
      // the permuted index: we add the local offset relative to this expert and token
      // to the global offset from the scan for this expert
      auto permutedIdx = isLocalExpert ? finalExpertOffset[ii] + localOffsetToken : int32_t{-1};
      // write out `mPtrExpandedIdxToPermutedIdx` if required
      if (params.mPtrExpandedIdxToPermutedIdx != nullptr && isTokenRouted) {
        params.mPtrExpandedIdxToPermutedIdx[tokenIdx] = permutedIdx;
      }
      // write out `mPtrPermutedIdxToTokenIdx` if required
      if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert && isTokenRouted) {
        params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
      }
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params) {
  // number of tokens/expanded idx is bounded by total number of warps
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using TypePacked = PackedScoreIdx<OutputT>;
  __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[NumWarps];

  uint32_t const clusterBlockRank = blockIdx.x;
  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();

  // TODO(mjoux): expand to more tokens (possibly)
  auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
  auto scoreOffset = warpTokenIdx * params.mNumExperts;
  bool validToken = warpTokenIdx < params.mNumTokens;
  InputT minScore = InputT{-INFINITY};

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  // then wait on primary grid
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }

  if (params.mPtrScores != nullptr) {
    // in this case, each warp represents a token
    // we then exchange all token max scores, s.t. afterwards, each thread
    // represents a token
    InputT warpMaxScore[MaxNumTopExperts];
    int32_t warpMaxExpertIdx[MaxNumTopExperts];

    if (validToken) {
      routingTopKExperts<InputT, MaxNumExperts / WarpSize>(warp, warpMaxScore, warpMaxExpertIdx,
                                                           laneIdx, params.mNumExperts,
                                                           params.mPtrScores + scoreOffset);
      if (cute::elect_one_sync()) {
        auto finalScore = OutputT{sigmoid_accurate(float{warpMaxScore[0]})};
        TypePacked packedScore{finalScore, static_cast<int16_t>(warpMaxExpertIdx[0])};
        smemPackedScoreIdx[warpIdx] = packedScore;
      }
    }
    // make packed scores available to all threads in cluster
    __cluster_barrier_arrive();
    __cluster_barrier_wait();
  }

  routingPermutation<KernelParams, OutputT, NumThreads, NumWarps, MaxNumTopExperts,
                     /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx,
                                                        clusterBlockRank);
}
#else
__global__ void routingIndicesClusterKernel(KernelParams params) {
  assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

// this kernel is needed in case we have scores as input for the histogram kernel
template <typename KernelParams>
__global__ void __launch_bounds__(NumThreadsHist)
    routingIndicesHistogramScoresKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using TypePacked = PackedScoreIdx<OutputT>;
  static constexpr int VecSize = MaxNumExperts / WarpSize;
  //  we assume that #experts is a multiple of 4, so VecSize must be 4.
  static_assert(VecSize == 4);

  int32_t const laneIdx = cutlass::arch::LaneId();
  int32_t const warpIdx = threadIdx.x / WarpSize;
  int32_t const globalWarpIdx = blockIdx.x * NumWarpsHist + warpIdx;
  int32_t const globalWarpStride = gridDim.x * NumWarpsHist;
  InputT minScore = InputT{-INFINITY};
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  // initialize the mPtrExpertCounts
  int32_t expertCountsNum = 2 * params.mNumExperts;
  int32_t globalThreadIdx = blockIdx.x * NumThreads + threadIdx.x;
  int32_t globalThreadStride = gridDim.x * NumThreads;
  initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Wait on primary grid and trigger secondary kernel.
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  // in this case, each warp represents a token, and we use a grid-stride loop
  // over all warps/tokens
  for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride) {
    auto scoreOffset = tokenIdx * params.mNumExperts;
    int32_t warpMaxExpertIdx[MaxNumTopExperts];
    InputT warpMaxScore[MaxNumTopExperts];

    routingTopKExperts<InputT, MaxNumExperts / WarpSize>(warp, warpMaxScore, warpMaxExpertIdx,
                                                         laneIdx, params.mNumExperts,
                                                         params.mPtrScores + scoreOffset);

    if (cute::elect_one_sync()) {
      auto finalScore = OutputT{sigmoid_accurate(float{warpMaxScore[0]})};
      TypePacked packedScore{finalScore, static_cast<int16_t>(warpMaxExpertIdx[0])};
      params.mPtrExpertIdx[tokenIdx] = packedScore;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  TORCH_CHECK(data.mPtrExpertIdx != nullptr || data.mPtrScores != nullptr,
              "Routing kernel requires at least one input parameter");
  TORCH_CHECK(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr &&
                  data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
              "Llama4 routing kernel expects permuted idx and grouped Gemm launch config buffers");
  TORCH_CHECK(data.mTopK <= MaxNumTopExperts, "Routing kernel expects topK experts <= %d, got %d",
              MaxNumTopExperts, data.mTopK);
  TORCH_CHECK(data.mNumExperts <= MaxNumExperts,
              "Routing kernel expects #experts %d to be at most max #experts %d", data.mNumExperts,
              MaxNumExperts);
  static_assert(MaxNumExperts <= NumThreads, "#experts must be bounded by #threads");
  static_assert(MaxNumExperts <= NumThreadsHist, "#experts must be bounded by #threads");
  TORCH_CHECK(data.mNumExperts % 4 == 0,
              "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);
  TORCH_CHECK(data.mPaddingLog2 < 8, "Routing kernel expects padding log2 < 8, got %d",
              data.mPaddingLog2);

  bool const useSingleWarp =
      (data.mPtrScores == nullptr && data.mNumTokens <= WarpKernelMaxNumTokens) ||
      data.mNumTokens < WarpKernelMaxNumTokens;
  bool const useSingleCluster =
      data.mNumTokens <=
      (data.mPtrScores != nullptr ? MaxNumTokensSingleClusterScores : MaxNumTokensSingleCluster);
  if (!useSingleCluster) {
    TORCH_CHECK(data.mPtrExpertIdx != nullptr,
                "When #tokens is large, `mPtrExpertIdx` is a required input.");
    TORCH_CHECK(data.mPtrExpertCounts != nullptr,
                "When #tokens is large, `mPtrExpertCounts` is a required input.");
  }

  if (useSingleWarp) {
    LAUNCH_ROUTING(data,
                   /*coopLaunch=*/false, routingIndicesWarpKernel, 1, WarpSize,
                   /*smemSize=*/0,  // No dynamic smem
                   stream);
  } else if (useSingleCluster) {
    LAUNCH_ROUTING(data,
                   /*coopLaunch=*/false, routingIndicesClusterKernel, NumBlocksPerCluster,
                   NumThreads,
                   /*smemSize=*/0,  // No dynamic smem
                   stream);
  } else {
    const uint32_t expandedIdxSize = data.mNumTokens * data.mTopK;

    const uint32_t histogramEltsPerBlock = 8 * NumThreadsHist;
    const uint32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * NumThreadsHist;

    // Limit grid size (all kernels use a grid-stride loop).
    const uint32_t maxNumBlocks = 1024;

    int const numBlocksHistogram = std::min(
        (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
    int const numBlocksOffsets =
        std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

    if (data.mPtrScores != nullptr) {
      LAUNCH_ROUTING(data,
                     /*coopLaunch=*/false, routingIndicesHistogramScoresKernel, maxNumBlocks,
                     NumThreadsHist,
                     /*smemSize=*/0,  // No dynamic smem
                     stream);
    } else {
      // Reset the global histograms.
      TORCH_CHECK(cudaMemsetAsync(data.mPtrExpertCounts, 0,
                                  static_cast<size_t>(2 * NumThreads) * sizeof(int32_t),
                                  (cudaStream_t)stream));
    }
    LAUNCH_ROUTING(data,
                   /*coopLaunch=*/false, routingIndicesHistogramKernel, numBlocksHistogram,
                   NumThreadsHist,
                   /*smemSize=*/0,  // No dynamic smem
                   stream);
    LAUNCH_ROUTING(data,
                   /*coopLaunch=*/false, routingIndicesOffsetsKernel, numBlocksOffsets,
                   NumThreadsHist,
                   /*smemSize=*/0,  // No dynamic smem
                   stream);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingLlama4

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace routingRenormalize {
////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int MaxNumTopExperts = 8;
static constexpr int MaxNumExperts = 128;
static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;

template <typename DataType, typename InputType, int VecSize, bool DoSoftmaxBeforeTopK>
__forceinline__ __device__ void routingTopKExperts(
    cg::thread_block_tile<WarpSize> const& warp, DataType (&score)[VecSize],
    int32_t (&idx)[VecSize], DataType (&warpTopKScore)[MaxNumTopExperts],
    int32_t (&warpTopKExpertIdx)[MaxNumTopExperts], int32_t const laneIdx, int32_t const numExperts,
    int32_t topK, InputType const* ptrScores, bool const normTopkProb) {
  DataType minScore = DataType{-INFINITY};

  for (int i = 0; i < VecSize; i++) {
    auto expertIdx = i * WarpSize + laneIdx;
    auto newScore = expertIdx < numExperts ? static_cast<DataType>(ptrScores[expertIdx]) : minScore;
    score[i] = newScore;
    idx[i] = expertIdx;
  }
  if constexpr (DoSoftmaxBeforeTopK) {
    calcSoftmax(warp, score);
  }

  // Get the top-k scores and their corresponding expert indices
  topk::reduceTopK(warp, warpTopKScore, warpTopKExpertIdx, score, idx, minScore, topK);

  // Normalize the scores
  if constexpr (DoSoftmaxBeforeTopK) {
    float sum = float{1.f};
    if (normTopkProb) {
      sum = static_cast<float>(laneIdx < topK ? warpTopKScore[laneIdx] : 0);
      sum = cg::reduce(warp, sum, cg::plus<float>());
    }
    if (laneIdx < topK) {
      warpTopKScore[laneIdx] = warpTopKScore[laneIdx] / sum;
    }
  } else {
    auto softmaxScore =
        calcSoftmax(warp, laneIdx < topK ? warpTopKScore[laneIdx] : minScore, laneIdx, topK);
    if (laneIdx < topK) {
      warpTopKScore[laneIdx] = softmaxScore;
    }
  }
}

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params) {
  // number of tokens/expanded idx is bounded by total number of warps
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;

  using BaseType = std::conditional_t<KernelParams::DoSoftmaxBeforeTopK, float, InputT>;
  using TypePacked = PackedScoreIdx<BaseType>;

  static constexpr int VecSize = MaxNumExperts / WarpSize;
  // we assume that #experts is a multiple of 4, so VecSize must be 4.
  static_assert(VecSize == 4);

  __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[NumWarps * MaxNumTopExperts];

  uint32_t const clusterBlockRank = blockIdx.x;

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();

  auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
  auto scoreOffset = warpTokenIdx * params.mNumExperts;
  bool validToken = warpTokenIdx < params.mNumTokens;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  // then wait on primary grid
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }

  if (params.mPtrScores != nullptr) {
    // in this case, each warp represents a token
    BaseType score[VecSize];
    int32_t idx[VecSize];

    BaseType warpTopKScore[MaxNumTopExperts];
    int32_t warpTopKExpertIdx[MaxNumTopExperts];

    BaseType minScore = BaseType{-INFINITY};
    if (validToken) {
      routingTopKExperts<BaseType, InputT, VecSize, KernelParams::DoSoftmaxBeforeTopK>(
          warp, score, idx, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts,
          params.mTopK, params.mPtrScores + scoreOffset, params.mNormTopkProb);

      if (laneIdx < params.mTopK) {
        smemPackedScoreIdx[warpIdx * params.mTopK + laneIdx] =
            TypePacked{warpTopKScore[laneIdx], static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
      }
    }  // end if (validToken)

    // make packed scores available to all threads in cluster
    __cluster_barrier_arrive();
    __cluster_barrier_wait();
  }

  routingPermutation<KernelParams, BaseType, NumThreads, NumWarps, MaxNumTopExperts,
                     /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx,
                                                        clusterBlockRank);
}
#else
__global__ void __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams /* params */) {
  assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
////////////////////////////////////////////////////////////////////////////////////////////////////

// this kernel is needed in case we have scores as input for the histogram kernel
template <typename KernelParams>
__global__ void __launch_bounds__(NumThreadsHist)
    routingIndicesHistogramScoresKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = std::conditional_t<KernelParams::DoSoftmaxBeforeTopK, float, InputT>;

  static constexpr int VecSize = MaxNumExperts / WarpSize;
  // we assume that #experts is a multiple of 4, so VecSize must be 4.
  static_assert(VecSize == 4);

  int32_t const laneIdx = cutlass::arch::LaneId();
  int32_t const warpIdx = threadIdx.x / WarpSize;
  int32_t const globalWarpIdx = blockIdx.x * NumWarpsHist + warpIdx;
  int32_t const globalWarpStride = gridDim.x * NumWarpsHist;
  BaseType minScore = BaseType{-INFINITY};
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Wait on primary grid.
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

  // initialize the mPtrExpertCounts
  int32_t expertCountsNum = 2 * params.mNumExperts;
  int32_t globalThreadIdx = blockIdx.x * NumThreads + threadIdx.x;
  int32_t globalThreadStride = gridDim.x * NumThreads;
  initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Trigger secondary kernel.
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

  // in this case, each warp represents a token, and we use a grid-stride loop
  // over all warps/tokens
  BaseType allScores[VecSize];
  int32_t allExpertIdx[VecSize];
  BaseType warpTopKScore[MaxNumTopExperts];
  int32_t warpTopKExpertIdx[MaxNumTopExperts];
  for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride) {
    auto scoreOffset = tokenIdx * params.mNumExperts;

    routingTopKExperts<BaseType, InputT, VecSize, KernelParams::DoSoftmaxBeforeTopK>(
        warp, allScores, allExpertIdx, warpTopKScore, warpTopKExpertIdx, laneIdx,
        params.mNumExperts, params.mTopK, params.mPtrScores + scoreOffset, params.mNormTopkProb);

    if (laneIdx < params.mTopK) {
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(warpTopKScore[laneIdx]),
                                          static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
      params.mPtrExpertIdx[tokenIdx * params.mTopK + laneIdx] = packedScore;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  TORCH_CHECK(data.mPtrExpertIdx != nullptr || data.mPtrScores != nullptr,
              "Routing kernel requires at least one input parameter");
  TORCH_CHECK(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr &&
                  data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
              "Llama4 routing kernel expects permuted idx and grouped Gemm launch config buffers");
  TORCH_CHECK(data.mTopK <= MaxNumTopExperts, "Routing kernel expects topK experts <= %d, got %d",
              MaxNumTopExperts, data.mTopK);
  TORCH_CHECK(data.mNumExperts <= MaxNumExperts,
              "Routing kernel expects #experts %d to be at most max #experts %d", data.mNumExperts,
              MaxNumExperts);
  static_assert(MaxNumExperts <= NumThreads, "#experts must be bounded by #threads");
  static_assert(MaxNumExperts <= NumThreadsHist, "#experts must be bounded by #threads");
  TORCH_CHECK(data.mNumExperts % 4 == 0,
              "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);
  TORCH_CHECK(data.mPaddingLog2 < 8, "Routing kernel expects padding log2 < 8, got %d",
              data.mPaddingLog2);

  bool const useSingleCluster =
      data.mNumTokens <=
      (data.mPtrScores != nullptr ? MaxNumTokensSingleClusterScores : MaxNumTokensSingleCluster);

  if (!useSingleCluster) {
    TORCH_CHECK(data.mPtrExpertIdx != nullptr,
                "When #tokens is large, `mPtrExpertIdx` is a required input.");
    TORCH_CHECK(data.mPtrExpertCounts != nullptr,
                "When #tokens is large, `mPtrExpertCounts` is a required input.");
  }

  if (useSingleCluster) {
    LAUNCH_ROUTING_WITH_EXTRA_FLAG(data, false, routingIndicesClusterKernel, NumBlocksPerCluster,
                                   NumThreads,
                                   /*smemSize=*/0,  // No dynamic smem
                                   stream, data.mDoSoftmaxBeforeTopK, /*forceFloatInput=*/false);
  } else {
    uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;

    uint32_t const histogramEltsPerBlock = 8 * NumThreadsHist;
    uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * NumThreadsHist;

    // Limit grid size (all kernels use a grid-stride loop).
    uint32_t const maxNumBlocks = 1024;

    int const numBlocksHistogram = std::min(
        (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
    int const numBlocksOffsets =
        std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

    if (data.mPtrScores != nullptr) {
      LAUNCH_ROUTING_WITH_EXTRA_FLAG(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks,
                                     NumThreadsHist,
                                     /*smemSize=*/0,  // No dynamic smem
                                     stream, data.mDoSoftmaxBeforeTopK, /*forceFloatInput=*/false);
    } else {
      // Reset the global histograms.
      TORCH_CHECK(cudaMemsetAsync(data.mPtrExpertCounts, 0,
                                  static_cast<size_t>(2 * NumThreads) * sizeof(int32_t),
                                  (cudaStream_t)stream));
    }
    LAUNCH_ROUTING_WITH_EXTRA_FLAG(data, false, routingIndicesHistogramKernel, numBlocksHistogram,
                                   NumThreadsHist,
                                   /*smemSize=*/0,  // No dynamic smem
                                   stream, data.mDoSoftmaxBeforeTopK, /*forceFloatInput=*/false);
    LAUNCH_ROUTING_WITH_EXTRA_FLAG(data, false, routingIndicesOffsetsKernel, numBlocksOffsets,
                                   NumThreadsHist,
                                   /*smemSize=*/0,  // No dynamic smem
                                   stream, data.mDoSoftmaxBeforeTopK, /*forceFloatInput=*/false);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingRenormalize
}  // namespace moe::dev::routing
