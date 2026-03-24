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

#include "flashinfer/exception.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.cuh"

namespace moe::dev::routing {
namespace routingLlama4 {

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int NumThreads = 1024;
static constexpr int NumWarps = NumThreads / WarpSize;
static constexpr int MaxNumTopExperts = 1;
static constexpr int MaxSupportedExperts = 128;
static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
static constexpr int MaxNumTokensSingleClusterScores = NumBlocksPerCluster * NumWarps;
static constexpr int WarpKernelSmemStride = 33;
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
  int32_t maxExpertIdx{0};
  using DataTypeVec = std::conditional_t<sizeof(DataType) == 2, float2, float4>;

  for (int i = 0; i < VecSize; ++i) {
    auto expertIdx = i * WarpSize + laneIdx;
    auto newScore = expertIdx < numExperts ? ptrScores[expertIdx] : minScore;
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
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using TypePacked = PackedScoreIdx<OutputT>;
  using Scan = cub::WarpScan<int32_t>;
  __shared__ typename Scan::TempStorage tempStorage;

  static constexpr int ExpertsPerThread = sizeof(int32_t);
  static_assert(WarpKernelMaxNumTokens <= 127);
  __shared__ int32_t __attribute((
      aligned(128))) smemExpertTokenCountFull[WarpKernelMaxNumTokens][WarpKernelSmemStride];
  static_assert(WarpKernelSmemStride == WarpSize + 1);
  static_assert(KernelParams::MaxNumExperts / sizeof(int32_t) <= WarpSize);

  InputT minScore = InputT{-INFINITY};
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

#pragma unroll
  for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens; ++tokenIdx) {
    smemExpertTokenCountFull[tokenIdx][threadIdx.x] = 0;
  }
  __syncwarp();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  if (params.mPtrScores != nullptr && params.mPtrTopKIds == nullptr) {
    for (int tokenIdx = 0; tokenIdx < params.mNumTokens; ++tokenIdx) {
      auto scoreOffset = tokenIdx * params.mNumExperts;
      int32_t warpMaxExpertIdx[MaxNumTopExperts];
      InputT warpMaxScore[MaxNumTopExperts];

      routingTopKExperts<InputT, ExpertsPerThread>(warp, warpMaxScore, warpMaxExpertIdx,
                                                   threadIdx.x, params.mNumExperts,
                                                   params.mPtrScores + scoreOffset);

      if (cute::elect_one_sync()) {
        auto expertTokenCount = 0;
        setBits</* IsZero= */ true>(expertTokenCount, 1, warpMaxExpertIdx[0] % ExpertsPerThread);
        smemExpertTokenCountFull[tokenIdx][warpMaxExpertIdx[0] / ExpertsPerThread] =
            expertTokenCount;
        auto finalScore = OutputT{sigmoid_accurate(float{warpMaxScore[0]})};
        if (params.mPtrTopKWeights != nullptr) {
          params.mPtrTopKWeights[tokenIdx] = finalScore;
        }
      }
    }
  } else {
    static_assert(WarpKernelMaxNumTokens <= WarpSize);
    TypePacked scoreIdx = TypePacked{};
    if (params.mPtrTopKIds != nullptr) {
      if (threadIdx.x < params.mNumTokens) {
        scoreIdx = TypePacked{static_cast<OutputT>(params.mPtrTopKWeights[threadIdx.x]),
                              static_cast<int16_t>(params.mPtrTopKIds[threadIdx.x])};
      }
    } else {
      if (threadIdx.x < params.mNumTokens) {
        scoreIdx = TypePacked{static_cast<OutputT>(params.mPtrTopKPacked[threadIdx.x].score),
                              static_cast<int16_t>(params.mPtrTopKPacked[threadIdx.x].idx)};
        if (params.mPtrTopKWeights != nullptr) {
          auto finalScore = OutputT{sigmoid_accurate(float{scoreIdx.score})};
          params.mPtrTopKWeights[threadIdx.x] = finalScore;
        }
      }
    }

    int32_t expertTokenCount = 0;
    setBits</* IsZero= */ true>(expertTokenCount, 1, scoreIdx.idx % ExpertsPerThread);
    if (threadIdx.x < params.mNumTokens) {
      smemExpertTokenCountFull[threadIdx.x][scoreIdx.idx / ExpertsPerThread] = expertTokenCount;
    }
  }

  __syncwarp();

  int32_t expertCount = 0;
  int32_t expertOffset[WarpKernelMaxNumTokens + 1];
#pragma unroll
  for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens + 1; ++tokenIdx) {
    if (tokenIdx > params.mNumTokens) break;
    auto expertTokenCount =
        tokenIdx < params.mNumTokens ? smemExpertTokenCountFull[tokenIdx][threadIdx.x] : 0;
    expertOffset[tokenIdx] = expertCount;
    expertCount += expertTokenCount;
  }

  int32_t numCta = 0;
#pragma unroll
  for (int ii = 0; ii < ExpertsPerThread; ++ii) {
    auto count = getBits(expertCount, ii);
    int32_t num;
    if constexpr (KernelParams::isPow2) {
      num = divUpLog2<int32_t>(count, params.mPaddingLog2);
    } else {
      num = divUpTileN<int32_t>(count, params.mTileTokensDim);
    }
    numCta += num;
  }
  int32_t ctaOffset;
  int32_t numNonExitingCtas;
  Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

  auto ctaOffsetExp = ctaOffset;
#pragma unroll
  for (int ii = 0; ii < ExpertsPerThread; ++ii) {
    auto count = getBits(expertCount, ii);
    int32_t finalNumCta;
    if constexpr (KernelParams::isPow2) {
      finalNumCta = divUpLog2<int32_t>(count, params.mPaddingLog2);
    } else {
      finalNumCta = divUpTileN<int32_t>(count, params.mTileTokensDim);
    }
    auto expertIdx = threadIdx.x * ExpertsPerThread + ii;
    for (int cta = 0; cta < finalNumCta; ++cta) {
      params.mPtrCtaIdxXyToBatchIdx[ctaOffsetExp + cta] = expertIdx;
      int32_t mnLimit1;
      int32_t mnLimit2;
      if constexpr (KernelParams::isPow2) {
        mnLimit1 = mulLog2<int32_t>(ctaOffsetExp + cta + 1, params.mPaddingLog2);
        mnLimit2 = mulLog2<int32_t>(ctaOffsetExp, params.mPaddingLog2) + count;
      } else {
        mnLimit1 = mulTileN<int32_t>(ctaOffsetExp + cta + 1, params.mTileTokensDim);
        mnLimit2 = mulTileN<int32_t>(ctaOffsetExp, params.mTileTokensDim) + count;
      }
      params.mPtrCtaIdxXyToMnLimit[ctaOffsetExp + cta] = min(mnLimit1, mnLimit2);
    }
    ctaOffsetExp += finalNumCta;
  }

  if (cute::elect_one_sync()) {
    int32_t permutedIdxSize;
    if constexpr (KernelParams::isPow2) {
      permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    } else {
      permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
    }
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
  int32_t finalExpertOffset[ExpertsPerThread];
  if constexpr (KernelParams::isPow2) {
    finalExpertOffset[0] = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2);
  } else {
    finalExpertOffset[0] = mulTileN<int32_t>(ctaOffset, params.mTileTokensDim);
  }
#pragma unroll
  for (int ii = 1; ii < ExpertsPerThread; ++ii) {
    int32_t tmp;
    if constexpr (KernelParams::isPow2) {
      tmp = divUpMulLog2<int32_t>(getBits(expertCount, ii - 1), params.mPaddingLog2);
    } else {
      tmp = divUpMulTileN<int32_t>(getBits(expertCount, ii - 1), params.mTileTokensDim);
    }
    finalExpertOffset[ii] = finalExpertOffset[ii - 1] + tmp;
  }

#pragma unroll
  for (int tokenIdx = 0; tokenIdx < WarpKernelMaxNumTokens; ++tokenIdx) {
    if (tokenIdx >= params.mNumTokens) break;

#pragma unroll
    for (int ii = 0; ii < ExpertsPerThread; ++ii) {
      auto localOffsetToken = getBits(expertOffset[tokenIdx], ii);
      auto isTokenRouted = getBits(expertOffset[tokenIdx + 1], ii) > localOffsetToken;
      auto expertIdx = threadIdx.x * ExpertsPerThread + ii;
      auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
      auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                           (localExpertIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
      auto permutedIdx = isLocalExpert ? finalExpertOffset[ii] + localOffsetToken : int32_t{-1};
      if (params.mPtrExpandedIdxToPermutedIdx != nullptr && isTokenRouted) {
        params.mPtrExpandedIdxToPermutedIdx[tokenIdx] = permutedIdx;
      }
      if (params.mPtrPermutedIdxToExpandedIdx != nullptr && isLocalExpert) {
        params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = tokenIdx;
      }
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
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using TypePacked = PackedScoreIdx<OutputT>;
  __shared__ TypePacked __attribute((aligned(128))) smemPackedScoreIdx[NumWarps];

  uint32_t const clusterBlockRank = blockIdx.x;
  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();

  auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
  auto scoreOffset = warpTokenIdx * params.mNumExperts;
  bool validToken = warpTokenIdx < params.mNumTokens;
  InputT minScore = InputT{-INFINITY};

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }

  if (params.mPtrTopKIds != nullptr) {
    if (validToken) {
      TypePacked packedScore{static_cast<OutputT>(params.mPtrTopKWeights[warpTokenIdx]),
                             static_cast<int16_t>(params.mPtrTopKIds[warpTokenIdx])};
      smemPackedScoreIdx[warpIdx] = packedScore;
    }
  } else if (params.mPtrScores != nullptr) {
    InputT warpMaxScore[MaxNumTopExperts];
    int32_t warpMaxExpertIdx[MaxNumTopExperts];

    if (validToken) {
      routingTopKExperts<InputT, KernelParams::MaxNumExperts / WarpSize>(
          warp, warpMaxScore, warpMaxExpertIdx, laneIdx, params.mNumExperts,
          params.mPtrScores + scoreOffset);
      if (cute::elect_one_sync()) {
        auto finalScore = OutputT{sigmoid_accurate(float{warpMaxScore[0]})};
        TypePacked packedScore{finalScore, static_cast<int16_t>(warpMaxExpertIdx[0])};
        smemPackedScoreIdx[warpIdx] = packedScore;
      }
    }
  } else {
    if (validToken) {
      smemPackedScoreIdx[warpIdx] = params.mPtrTopKPacked[warpTokenIdx];
    }
  }

  __cluster_barrier_arrive();
  __cluster_barrier_wait();

  if (params.mPtrTopKIds != nullptr || params.mPtrScores != nullptr) {
    routingPermutation<KernelParams, OutputT, NumThreads, NumWarps, MaxNumTopExperts,
                       /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx,
                                                          clusterBlockRank);
  } else {
    routingPermutation<KernelParams, OutputT, NumThreads, NumWarps, MaxNumTopExperts,
                       /*LoadExpertIdxFromGlobal=*/true>(params, smemPackedScoreIdx, warpIdx,
                                                         clusterBlockRank);
  }
}
#else
__global__ void routingIndicesClusterKernel(KernelParams params) {
  assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts)
    routingIndicesHistogramScoresKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using TypePacked = PackedScoreIdx<OutputT>;
  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;
  static_assert(VecSize == 4);

  int32_t const laneIdx = cutlass::arch::LaneId();
  int32_t const warpIdx = threadIdx.x / WarpSize;
  int32_t const globalWarpIdx = blockIdx.x * KernelParams::MaxNumExperts / WarpSize + warpIdx;
  int32_t const globalWarpStride = gridDim.x * KernelParams::MaxNumExperts / WarpSize;
  InputT minScore = InputT{-INFINITY};
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  int32_t expertCountsNum = 2 * params.mNumExperts;
  int32_t globalThreadIdx = blockIdx.x * KernelParams::MaxNumExperts + threadIdx.x;
  int32_t globalThreadStride = gridDim.x * KernelParams::MaxNumExperts;
  initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride) {
    auto scoreOffset = tokenIdx * params.mNumExperts;
    int32_t warpMaxExpertIdx[MaxNumTopExperts];
    InputT warpMaxScore[MaxNumTopExperts];

    if (params.mPtrTopKIds != nullptr) {
      if (laneIdx < MaxNumTopExperts) {
        warpMaxExpertIdx[laneIdx] = params.mPtrTopKIds[tokenIdx];
        warpMaxScore[laneIdx] = static_cast<InputT>(params.mPtrTopKWeights[tokenIdx]);
      }
    } else if (params.mPtrScores != nullptr) {
      routingTopKExperts<InputT, KernelParams::MaxNumExperts / WarpSize>(
          warp, warpMaxScore, warpMaxExpertIdx, laneIdx, params.mNumExperts,
          params.mPtrScores + scoreOffset);
    } else {
      if (laneIdx < MaxNumTopExperts) {
        warpMaxExpertIdx[laneIdx] = params.mPtrTopKPacked[tokenIdx].idx;
        warpMaxScore[laneIdx] = params.mPtrTopKPacked[tokenIdx].score;
      }
    }

    if (cute::elect_one_sync()) {
      auto finalScore = OutputT{sigmoid_accurate(float{warpMaxScore[0]})};
      TypePacked packedScore{finalScore, static_cast<int16_t>(warpMaxExpertIdx[0])};
      params.mPtrTopKPacked[tokenIdx] = packedScore;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int constexpr getMaxNumExperts(int32_t numExperts) {
  if (numExperts <= topk::MaxNumExpertsUnit) {
    return topk::MaxNumExpertsUnit;
  } else {
    TLLM_LOG_ERROR("Unsupported numExperts");
    return 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  FLASHINFER_CHECK(
      data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr,
      "Routing kernel requires at least one input parameter");
  if (data.mPtrTopKIds != nullptr) {
    FLASHINFER_CHECK(
        data.mPtrTopKWeights != nullptr,
        "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for Llama4 routing.");
  }
  FLASHINFER_CHECK(
      data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr &&
          data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr,
      "Llama4 routing kernel expects permuted idx and grouped Gemm launch config buffers");
  FLASHINFER_CHECK(data.mTopK <= MaxNumTopExperts,
                   "Routing kernel expects topK experts <= %d, got %d", MaxNumTopExperts,
                   data.mTopK);
  FLASHINFER_CHECK(data.mNumExperts <= MaxSupportedExperts,
                   "Routing kernel expects #experts %d to be no more than %d", data.mNumExperts,
                   MaxSupportedExperts);
  FLASHINFER_CHECK(data.mNumExperts % 4 == 0,
                   "Routing kernel expects #experts %d to be a multiple of 4.", data.mNumExperts);

  bool const useSingleWarp =
      (data.mPtrScores == nullptr && data.mNumTokens <= WarpKernelMaxNumTokens) ||
      data.mNumTokens < WarpKernelMaxNumTokens;
  bool const useSingleCluster =
      data.mNumTokens <= ((data.mPtrScores != nullptr || data.mPtrTopKIds != nullptr)
                              ? MaxNumTokensSingleClusterScores
                              : MaxNumTokensSingleCluster);
  if (!useSingleCluster) {
    FLASHINFER_CHECK(
        (data.mPtrTopKPacked != nullptr || data.mPtrTopKIds != nullptr),
        "When #tokens is large, `mPtrTopKPacked` or `mPtrTopKIds` is a required input.");
    FLASHINFER_CHECK(data.mPtrExpertCounts != nullptr,
                     "When #tokens is large, `mPtrExpertCounts` is a required input.");
  }

  int const numThreadsHist = getMaxNumExperts(data.mNumExperts);
  if (useSingleWarp) {
    LAUNCH_ROUTING_LLAMA4(data,
                          /*coopLaunch=*/false, routingIndicesWarpKernel, 1, WarpSize,
                          /*smemSize=*/0, stream);
  } else if (useSingleCluster) {
    LAUNCH_ROUTING_LLAMA4(data,
                          /*coopLaunch=*/false, routingIndicesClusterKernel, NumBlocksPerCluster,
                          NumThreads,
                          /*smemSize=*/0, stream);
  } else {
    const uint32_t expandedIdxSize = data.mNumTokens * data.mTopK;
    const uint32_t histogramEltsPerBlock = 8 * numThreadsHist;
    const uint32_t offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;
    const uint32_t maxNumBlocks = 1024;

    int const numBlocksHistogram = std::min(
        (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
    int const numBlocksOffsets =
        std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

    if (data.mPtrScores != nullptr && data.mPtrTopKIds == nullptr) {
      LAUNCH_ROUTING_LLAMA4(data,
                            /*coopLaunch=*/false, routingIndicesHistogramScoresKernel, maxNumBlocks,
                            numThreadsHist,
                            /*smemSize=*/0, stream);
    } else {
      LAUNCH_ROUTING_LLAMA4(data, false, routingInitExpertCounts,
                            (2 * data.mNumExperts - 1) / numThreadsHist + 1, numThreadsHist,
                            /*smemSize=*/0, stream);
    }
    LAUNCH_ROUTING_LLAMA4(data,
                          /*coopLaunch=*/false, routingIndicesHistogramKernel, numBlocksHistogram,
                          numThreadsHist,
                          /*smemSize=*/0, stream);
    LAUNCH_ROUTING_LLAMA4(data,
                          /*coopLaunch=*/false, routingIndicesOffsetsKernel, numBlocksOffsets,
                          numThreadsHist,
                          /*smemSize=*/0, stream);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingLlama4
}  // namespace moe::dev::routing
