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
#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cutlass/arch/arch.h>

#include <cub/cub.cuh>
#include <cute/arch/cluster_sm90.hpp>
#include <type_traits>

#include "DevKernel.h"
#include "RoutingKernel.h"
#include "RoutingKernelTopK.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace moe::dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace routing {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int WarpSize = 32;
static constexpr int NumBlocksPerCluster = 8;
// Performance tuning knob.
static constexpr int NumEltsPerOffsetTilePerThread = 8;

////////////////////////////////////////////////////////////////////////////////////////////////////

static __device__ inline float sigmoid_accurate(float x) { return 0.5f * tanhf(0.5f * x) + 0.5f; }

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__ constexpr T mulLog2(T a, T bLog2) {
  return a << bLog2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__ constexpr T divUpLog2(T a, T bLog2) {
  return ((a + (1 << bLog2) - 1) >> bLog2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__host__ __device__ constexpr T divUpMulLog2(T a, T bLog2) {
  return mulLog2<T>(divUpLog2<T>(a, bLog2), bLog2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ __device__ constexpr T mulTileN(T a, T tileN) {
  return a * tileN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ __device__ constexpr T divUpTileN(T a, T tileN) {
  return (a + tileN - 1) / tileN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
__host__ __device__ constexpr T divUpMulTileN(T a, T tileN) {
  return divUpTileN(a, tileN) * tileN;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ constexpr int32_t getBits(int32_t value, int idx) {
  int mask = idx == 0 ? 0x000000FF : idx == 1 ? 0x0000FF00 : idx == 2 ? 0x00FF0000 : 0xFF000000;
  return (value & mask) >> (idx * 8);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool IsZero = false>
__host__ __device__ constexpr void setBits(int32_t& value, int32_t newBits, int idx) {
  if constexpr (!IsZero) {
    int mask = idx == 0 ? 0xFFFFFF00 : idx == 1 ? 0xFFFF00FF : idx == 2 ? 0xFF00FFFF : 0x00FFFFFF;
    value &= mask;
  }
  value |= (newBits << (idx * 8));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
__device__ void initArr(int startIdx, int numElts, int stride, DataType* arr, DataType value) {
  if (arr != nullptr) {
    for (int i = startIdx; i < numElts; i += stride) {
      arr[i] = value;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename DataType, int VecSize>
__device__ void calcSoftmax(cg::thread_block_tile<WarpSize> const& warp,
                            DataType (&scores)[VecSize]) {
  // Compute in float to support half/bfloat16 inputs safely.
  float maxScore = -INFINITY;
  float sumScore = 0.f;
  // Get the max score for each token
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    float si = static_cast<float>(scores[i]);
    maxScore = si >= maxScore ? si : maxScore;
  }
  maxScore = cg::reduce(warp, maxScore, cg::greater<float>());

  // Get the summation of scores for each token
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    float si = static_cast<float>(scores[i]);
    float e = expf(si - maxScore);
    scores[i] = static_cast<DataType>(e);
    sumScore += e;
  }
  sumScore = cg::reduce(warp, sumScore, cg::plus<float>());

  // Normalize the scores
#pragma unroll
  for (int i = 0; i < VecSize; ++i) {
    float si = static_cast<float>(scores[i]) / sumScore;
    scores[i] = static_cast<DataType>(si);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename DataType>
__device__ DataType calcSoftmax(cg::thread_block_tile<WarpSize> const& warp, DataType score,
                                int32_t laneIdx, int32_t NumTopExperts) {
  // Compute max in float to support half/bfloat16 inputs safely.
  // cg::reduce with cg::greater<T> only supports float/double and integer types;
  // using __nv_bfloat16 or __half directly can generate unsupported redux.sync.max instructions.
  float maxScore = -INFINITY;
  if (laneIdx < NumTopExperts) {
    float si = static_cast<float>(score);
    maxScore = si >= maxScore ? si : maxScore;
  }
  maxScore = cg::reduce(warp, maxScore, cg::greater<float>());

  float sumScore = float{0.f};
  float newScore = 0.f;
  if (laneIdx < NumTopExperts) {
    newScore = static_cast<float>(score) - maxScore;
    newScore = expf(newScore);
    sumScore += newScore;
  }
  sumScore = cg::reduce(warp, sumScore, cg::plus<float>());

  if (laneIdx < NumTopExperts) {
    score = static_cast<DataType>(newScore / sumScore);
  }

  return score;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename KernelParams, typename BaseType, int NumThreads, int NumWarps,
          int MaxNumTopExperts, bool LoadExpertIdxFromGlobal = false>
__device__ void routingPermutation(KernelParams params,
                                   PackedScoreIdx<BaseType>* smemPackedScoreIdx,
                                   int32_t const warpIdx, uint32_t const clusterBlockRank) {
  using OutputT = typename KernelParams::OutputT;
  using TypePacked = PackedScoreIdx<BaseType>;

  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
  static constexpr int ExpertsPerThread =
      MaxNumExperts <= NumThreads ? 1 : MaxNumExperts / NumThreads;
  static_assert(MaxNumExperts <= NumThreads || MaxNumExperts % NumThreads == 0,
                "MaxNumExperts must be <= NumThreads or a multiple of NumThreads");

  static constexpr int MaxNumTokensSingleCluster = NumBlocksPerCluster * NumThreads;
  static constexpr int NumThreadsPerCluster = NumThreads * NumBlocksPerCluster;
  static constexpr int MaxExpandedIdxPerThread =
      (MaxNumTokensSingleCluster * MaxNumTopExperts + NumThreadsPerCluster - 1) /
      NumThreadsPerCluster;

  using Scan = cub::BlockScan<int32_t, NumThreads, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename Scan::TempStorage tempStorage;

  uint32_t const clusterThreadIdx = NumThreads * clusterBlockRank + threadIdx.x;
  auto expandedIdxSize = params.mNumTokens * params.mTopK;

  __shared__ int32_t __attribute((aligned(128))) smemExpertCount[MaxNumExperts];
  __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[MaxNumExperts];

#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    if (expert < params.mNumExperts) {
      smemExpertCount[expert] = 0;
    }
  }
  __syncthreads();

  int32_t expertIndexes[MaxExpandedIdxPerThread];
  int32_t expertOffsets[MaxExpandedIdxPerThread];
  auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

  auto loopBody = [&](int ii, int expandedIdx) {
    TypePacked scoreIdx;
    if constexpr (LoadExpertIdxFromGlobal) {
      if (params.mPtrTopKIds != nullptr) {
        scoreIdx = TypePacked{static_cast<BaseType>(params.mPtrTopKWeights[expandedIdx]),
                              static_cast<int16_t>(params.mPtrTopKIds[expandedIdx])};
      } else {
        scoreIdx = TypePacked{static_cast<BaseType>(params.mPtrTopKPacked[expandedIdx].score),
                              static_cast<int16_t>(params.mPtrTopKPacked[expandedIdx].idx)};
      }
    } else {
      TypePacked const* remoteSmem = cg::cluster_group::map_shared_rank(
          smemPackedScoreIdx, expandedIdx / (NumWarps * params.mTopK));
      scoreIdx = remoteSmem[expandedIdx % (NumWarps * params.mTopK)];
    }

    expertIndexes[ii] = scoreIdx.idx;
    auto localExpertIdx = scoreIdx.idx - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                         (localExpertIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
    expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + scoreIdx.idx, 1) : 0;
    if (params.mPtrTopKWeights != nullptr && params.mPtrTopKIds == nullptr) {
      params.mPtrTopKWeights[expandedIdx] = OutputT{scoreIdx.score};
    }
  };

  int constexpr IterStride = 4;
#pragma unroll
  for (int32_t ii0 = 0; ii0 < MaxExpandedIdxPerThread; ii0 += IterStride) {
    bool const takeFastPath = (ii0 + IterStride) * NumThreadsPerCluster <= expandedIdxSize;
    if (takeFastPath) {
#pragma unroll
      for (int32_t jj = 0; jj < IterStride; jj++) {
        int const ii = ii0 + jj;
        auto expandedIdx = static_cast<int32_t>(clusterThreadIdx) + ii * NumThreadsPerCluster;
        loopBody(ii, expandedIdx);
      }
    } else {
      bool doBreak = false;
#pragma unroll
      for (int32_t jj = 0; jj < IterStride; jj++) {
        int const ii = ii0 + jj;
        auto expandedIdx = static_cast<int32_t>(clusterThreadIdx) + ii * NumThreadsPerCluster;
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
  __cluster_barrier_arrive();
  __cluster_barrier_wait();

  int32_t count[ExpertsPerThread];
  int32_t blockExpertOffset[ExpertsPerThread];

#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    count[e] = 0;
    blockExpertOffset[e] = 0;

    if (expert < params.mNumExperts) {
      int32_t expertCounts[NumBlocksPerCluster];
#pragma unroll
      for (int rank = 0; rank < NumBlocksPerCluster; rank++) {
        int32_t const* remoteSmem = cg::cluster_group::map_shared_rank(smemExpertCount, rank);
        expertCounts[rank] = rank * NumWarps < params.mNumTokens ? remoteSmem[expert] : 0;
      }

#pragma unroll
      for (int rank = 0; rank < NumBlocksPerCluster; rank++) {
        if (rank == clusterBlockRank) {
          blockExpertOffset[e] = count[e];
        }
        count[e] += expertCounts[rank];
      }
    }
  }

  __cluster_barrier_arrive();

  int32_t numCta[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    if constexpr (KernelParams::isPow2) {
      numCta[e] = divUpLog2<int32_t>(count[e], params.mPaddingLog2);
    } else {
      numCta[e] = divUpTileN<int32_t>(count[e], params.mTileTokensDim);
    }
  }

  int32_t ctaOffset[ExpertsPerThread];
  int32_t numNonExitingCtas;
  Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    if (expert < params.mNumExperts) {
      for (int32_t cta = clusterBlockRank; cta < numCta[e]; cta += NumBlocksPerCluster) {
        const int32_t localExpertIdx =
            (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffset[e] + cta] = localExpertIdx;
        int32_t mnLimit1;
        int32_t mnLimit2;
        if constexpr (KernelParams::isPow2) {
          mnLimit1 = mulLog2<int32_t>(ctaOffset[e] + cta + 1, params.mPaddingLog2);
          mnLimit2 = mulLog2<int32_t>(ctaOffset[e], params.mPaddingLog2) + count[e];
        } else {
          mnLimit1 = mulTileN<int32_t>(ctaOffset[e] + cta + 1, params.mTileTokensDim);
          mnLimit2 = mulTileN<int32_t>(ctaOffset[e], params.mTileTokensDim) + count[e];
        }
        params.mPtrCtaIdxXyToMnLimit[ctaOffset[e] + cta] = min(mnLimit1, mnLimit2);
      }

      int32_t offset;
      if constexpr (KernelParams::isPow2) {
        offset = mulLog2<int32_t>(ctaOffset[e], params.mPaddingLog2);
      } else {
        offset = mulTileN<int32_t>(ctaOffset[e], params.mTileTokensDim);
      }
      smemExpertOffset[expert] = offset + blockExpertOffset[e];
    }
  }

  if (clusterBlockRank == 0 && warpIdx == NumWarps - 1 && cute::elect_one_sync()) {
    int32_t permutedIdxSize;
    if constexpr (KernelParams::isPow2) {
      permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    } else {
      permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
    }
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

  __syncthreads();
  __cluster_barrier_wait();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

#pragma unroll
  for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ++ii) {
    auto expandedIdx = static_cast<int32_t>(clusterThreadIdx) + ii * NumThreadsPerCluster;
    if (expandedIdx >= expandedIdxSize) {
      break;
    }
    auto expertIdx = expertIndexes[ii];
    auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                         (localExpertIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
    auto tokenIdx = expandedIdx / params.mTopK;
    auto permutedIdx =
        isLocalExpert ? int32_t{smemExpertOffset[expertIdx]} + expertOffsets[ii] : int32_t{-1};
    if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
      params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
    }
    if (params.mPtrPermutedIdxToExpandedIdx != nullptr && isLocalExpert) {
      params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
    }
    if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert) {
      params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts
                                                                      : 1024)
    routingIndicesHistogramKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
  static constexpr int NumThreadsBlock = MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
  static constexpr int ExpertsPerThread = MaxNumExperts / NumThreadsBlock;
  static_assert(MaxNumExperts % NumThreadsBlock == 0,
                "MaxNumExperts must be a multiple of NumThreadsBlock");

  __shared__ int32_t __attribute((aligned(128))) smemExpertCount[MaxNumExperts];

  uint32_t constexpr NumEltsPerThread = 8;

#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    if (expert < params.mNumExperts) {
      smemExpertCount[expert] = 0;
    }
  }
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif

  uint32_t const expandedIdxSize = params.mNumTokens * params.mTopK;
  uint32_t const localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

  uint32_t const gridBlockOffset = blockIdx.x * NumThreadsBlock;
  uint32_t const gridStride = gridDim.x * NumThreadsBlock;

  auto loopBody = [&](int expandedIdx) {
    PackedScoreIdx<OutputT> scoreIdx;
    int idx;
    if (params.mPtrTopKIds != nullptr) {
      idx = params.mPtrTopKIds[expandedIdx];
    } else {
      if (params.mPtrTopKWeights != nullptr) {
        scoreIdx = params.mPtrTopKPacked[expandedIdx];
        idx = scoreIdx.idx;
        params.mPtrTopKWeights[expandedIdx] = static_cast<OutputT>(scoreIdx.score);
      }
    }
    auto localExpertIdx = idx - params.mLocalExpertsStartIdx;
    auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                         (localExpertIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
    if (isLocalExpert) {
      atomicAdd(&smemExpertCount[idx], 1);
    }
  };

  for (uint32_t expandedIdx0 = gridBlockOffset * NumEltsPerThread; expandedIdx0 < expandedIdxSize;
       expandedIdx0 += gridStride * NumEltsPerThread) {
    if (expandedIdx0 + NumEltsPerThread * NumThreadsBlock <= expandedIdxSize) {
#pragma unroll
      for (uint32_t ii = 0; ii < NumEltsPerThread; ii++) {
        uint32_t expandedIdx = expandedIdx0 + ii * NumThreadsBlock + threadIdx.x;
        loopBody(expandedIdx);
      }
    } else {
      for (uint32_t expandedIdx = expandedIdx0 + threadIdx.x; expandedIdx < expandedIdxSize;
           expandedIdx += NumThreadsBlock) {
        loopBody(expandedIdx);
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    if (expert < params.mNumExperts) {
      int32_t const localExpertCount = smemExpertCount[expert];
      atomicAdd(&params.mPtrExpertCounts[expert], localExpertCount);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts
                                                                      : 1024)
    routingIndicesOffsetsKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
  static constexpr int NumThreadsBlock = MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
  static constexpr int ExpertsPerThread = MaxNumExperts / NumThreadsBlock;
  static_assert(MaxNumExperts % NumThreadsBlock == 0,
                "MaxNumExperts must be a multiple of NumThreadsBlock");

  __shared__ int32_t __attribute((aligned(128))) smemExpertOffset[MaxNumExperts];
  __shared__ int32_t __attribute((aligned(128))) smemExpertCount[MaxNumExperts];
  __shared__ int32_t __attribute((aligned(128))) smemExpertTileOffset[MaxNumExperts];
  using Scan = cub::BlockScan<int32_t, NumThreadsBlock, cub::BLOCK_SCAN_WARP_SCANS>;
  __shared__ typename Scan::TempStorage tempStorage;
  static constexpr int MaxExpandedIdxPerThread = NumEltsPerOffsetTilePerThread;
  static constexpr int MaxExpandedIdxPerBlock = NumThreadsBlock * MaxExpandedIdxPerThread;

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);

  uint32_t const expandedIdxSize = params.mNumTokens * params.mTopK;
  uint32_t const numTiles =
      (expandedIdxSize + MaxExpandedIdxPerBlock - 1) / (MaxExpandedIdxPerBlock);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  int32_t count[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    count[e] = (expert < params.mNumExperts) ? params.mPtrExpertCounts[expert] : 0;
  }

  int32_t numCta[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    if constexpr (KernelParams::isPow2) {
      numCta[e] = divUpLog2<int32_t>(count[e], params.mPaddingLog2);
    } else {
      numCta[e] = divUpTileN<int32_t>(count[e], params.mTileTokensDim);
    }
  }
  int32_t ctaOffset[ExpertsPerThread];
  int32_t numNonExitingCtas;
  Scan(tempStorage).ExclusiveSum(numCta, ctaOffset, numNonExitingCtas);

#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    if (expert < params.mNumExperts) {
      int32_t offset;
      if constexpr (KernelParams::isPow2) {
        offset = mulLog2<int32_t>(ctaOffset[e], params.mPaddingLog2);
      } else {
        offset = mulTileN<int32_t>(ctaOffset[e], params.mTileTokensDim);
      }
      smemExpertOffset[expert] = offset;
    }
  }

  __syncthreads();

  if (blockIdx.x == 0 && warpIdx == NumThreadsBlock / WarpSize - 1 && cute::elect_one_sync()) {
    int32_t permutedIdxSize;
    if constexpr (KernelParams::isPow2) {
      permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    } else {
      permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
    }
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    if (expert < params.mNumExperts) {
      for (int32_t cta = blockIdx.x; cta < numCta[e]; cta += gridDim.x) {
        const int32_t localExpertIdx =
            (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffset[e] + cta] = localExpertIdx;
        int32_t mnLimit1;
        int32_t mnLimit2;
        if constexpr (KernelParams::isPow2) {
          mnLimit1 = mulLog2<int32_t>(ctaOffset[e] + cta + 1, params.mPaddingLog2);
          mnLimit2 = mulLog2<int32_t>(ctaOffset[e], params.mPaddingLog2) + count[e];
        } else {
          mnLimit1 = mulTileN<int32_t>(ctaOffset[e] + cta + 1, params.mTileTokensDim);
          mnLimit2 = mulTileN<int32_t>(ctaOffset[e], params.mTileTokensDim) + count[e];
        }
        params.mPtrCtaIdxXyToMnLimit[ctaOffset[e] + cta] = min(mnLimit1, mnLimit2);
      }
    }
  }

  for (uint32_t tileIdx = blockIdx.x; tileIdx < numTiles; tileIdx += gridDim.x) {
    if (tileIdx > 0) {
      __syncthreads();
    }

#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      int expert = threadIdx.x * ExpertsPerThread + e;
      if (expert < params.mNumExperts) {
        smemExpertCount[expert] = 0;
      }
    }
    __syncthreads();

    int32_t expertIndexes[MaxExpandedIdxPerThread];
    int32_t expertOffsets[MaxExpandedIdxPerThread];
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;

    auto loopBody = [&](int ii, int expandedIdx) {
      expertIndexes[ii] = params.mPtrTopKIds ? params.mPtrTopKIds[expandedIdx]
                                             : params.mPtrTopKPacked[expandedIdx].idx;
      auto localExpertIdx = expertIndexes[ii] - params.mLocalExpertsStartIdx;
      auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                           (localExpertIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
      expertOffsets[ii] = isLocalExpert ? atomicAdd(smemExpertCount + expertIndexes[ii], 1) : 0;
    };

    if (tileIdx < numTiles - 1) {
#pragma unroll
      for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1) {
        auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsBlock + threadIdx.x;
        loopBody(ii, expandedIdx);
      }
    } else {
      int constexpr IterStride = 4;
      static_assert(MaxExpandedIdxPerThread % IterStride == 0);

#pragma unroll
      for (int32_t ii0 = 0; ii0 < MaxExpandedIdxPerThread; ii0 += IterStride) {
        bool const takeFastPath =
            tileIdx * MaxExpandedIdxPerBlock + (ii0 + IterStride) * NumThreadsBlock <=
            expandedIdxSize;
        if (takeFastPath) {
#pragma unroll
          for (int32_t jj = 0; jj < IterStride; jj++) {
            int const ii = ii0 + jj;
            auto expandedIdx =
                tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsBlock + threadIdx.x;
            loopBody(ii, expandedIdx);
          }
        } else {
          bool doBreak = false;
#pragma unroll
          for (int32_t jj = 0; jj < IterStride; jj++) {
            int const ii = ii0 + jj;
            auto expandedIdx =
                tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsBlock + threadIdx.x;
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
    }

    __syncthreads();

#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      int expert = threadIdx.x * ExpertsPerThread + e;
      if (expert < params.mNumExperts) {
        int32_t const localExpertCount = smemExpertCount[expert];
        int32_t const tileExpertOffset =
            atomicAdd(&params.mPtrExpertCounts[params.mNumExperts + expert], localExpertCount);
        smemExpertTileOffset[expert] = tileExpertOffset + smemExpertOffset[expert];
      }
    }
    __syncthreads();

    auto storeLoopBody = [&](int ii, int expandedIdx) {
      int32_t expertIdx = expertIndexes[ii];
      auto localExpertIdx = static_cast<int32_t>(expertIdx) - params.mLocalExpertsStartIdx;
      auto isLocalExpert = localExpertIdx >= 0 && localExpertIdx < localExpertExtent &&
                           (localExpertIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
      auto tokenIdx = expandedIdx / params.mTopK;
      auto permutedIdx =
          isLocalExpert ? (expertOffsets[ii] + smemExpertTileOffset[expertIdx]) : int32_t{-1};
      if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
        params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
      }
      if (params.mPtrPermutedIdxToExpandedIdx != nullptr && isLocalExpert) {
        params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
      }
      if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocalExpert) {
        params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
      }
    };
    if (tileIdx < numTiles - 1) {
#pragma unroll
      for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1) {
        auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsBlock + threadIdx.x;
        storeLoopBody(ii, expandedIdx);
      }
    } else {
#pragma unroll
      for (int32_t ii = 0; ii < MaxExpandedIdxPerThread; ii += 1) {
        auto expandedIdx = tileIdx * MaxExpandedIdxPerBlock + ii * NumThreadsBlock + threadIdx.x;
        if (expandedIdx >= expandedIdxSize) {
          break;
        }
        storeLoopBody(ii, expandedIdx);
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts
                                                                      : 1024)
    routingInitExpertCounts(KernelParams params) {
  static constexpr int NumThreadsBlock =
      KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024;

  int32_t expertCountsNum = 2 * params.mNumExperts;
  int32_t globalThreadIdx = blockIdx.x * NumThreadsBlock + threadIdx.x;
  int32_t globalThreadStride = gridDim.x * NumThreadsBlock;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (KernelParams::UsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (KernelParams::UsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}
}  // namespace routing
}  // namespace moe::dev
