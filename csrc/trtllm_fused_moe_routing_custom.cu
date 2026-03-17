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

// Custom routing: entry point, kernel definitions, and launch wrappers.
//
// Kernel inventory:
//   1. routingIndicesBlockKernel      — single-block fused kernel (≤4 tokens)
//   2. routingIndicesClusterKernel    — single-cluster fused kernel (≤256 tokens, SM90+)
//   3. routingIndicesHistogramScoresKernel — TopK + histogram from raw scores
//   4. routingIndicesCoopKernel       — cooperative histogram + offsets (defined in RoutingKernel.cuh)
//   5. routingInitExpertCounts        — zero expert counts (defined in RoutingKernel.cuh)
//   6. routingIndicesHistogramKernel  — histogram from packed TopK (defined in RoutingKernel.cuh)
//   7. routingIndicesOffsetsKernel    — prefix-scan + permutation (defined in RoutingKernel.cuh)

#include "flashinfer/trtllm/fused_moe/RoutingCustomPolicy.cuh"
#include "tvm_ffi_utils.h"

namespace moe::dev::routing {
namespace routingCustom {

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 1. Block kernel — single-block fused kernel for ≤4 tokens.
//    Fuses TopK, histogram, prefix-scan, and permutation in one block.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void
    __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024)
        routingIndicesBlockKernel(KernelParams params) {
  // types used in this kernel
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  using TypePacked = PackedScoreIdx<BaseType>;
  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
  // When MaxNumExperts > 1024, cap actual thread count at 1024 and let each thread handle
  // multiple experts. This is needed because CUDA blocks support at most 1024 threads.
  static constexpr int NumThreadsBlock = MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
  static constexpr int ExpertsPerThread = MaxNumExperts / NumThreadsBlock;
  static_assert(MaxNumExperts % NumThreadsBlock == 0,
                "MaxNumExperts must be a multiple of NumThreadsBlock");

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();
  auto scoreOffset = warpIdx * params.mNumExperts;
  bool validToken = warpIdx < params.mNumTokens;

  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;
  static constexpr int totalExpertCounts = BlockKernelMaxNumTokens * MaxNumExperts;
  __shared__ int8_t __attribute((aligned(128))) smemOffset[totalExpertCounts];
  __shared__ int8_t __attribute((aligned(128))) smemKIdx[totalExpertCounts];

  using Scan = cub::BlockScan<int32_t, NumThreadsBlock>;
  __shared__ typename Scan::TempStorage tempStorage;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  for (int i = threadIdx.x; i < totalExpertCounts; i += blockDim.x) {
    smemOffset[i] = int8_t{-1};
    smemKIdx[i] = int8_t{-1};
  }
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // then wait on primary grid
  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  if (params.mPtrTopKIds != nullptr) {
    if (validToken) {
      if (laneIdx < params.mTopK) {
        auto expertIdx = params.mPtrTopKIds[warpIdx * params.mTopK + laneIdx];
        if (expertIdx > -1 && expertIdx < params.mNumExperts) {
          int offset = warpIdx * MaxNumExperts + expertIdx;
          smemKIdx[offset] = static_cast<int8_t>(laneIdx);
        } else {
          params.mPtrExpandedIdxToPermutedIdx[warpIdx * params.mTopK + laneIdx] = int32_t{-1};
        }
      }
    }
  } else if (params.mPtrScores != nullptr) {
    // in this case, each warp represents a token
    BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
    int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];

    if (validToken) {
      KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                       KernelParams::MaxNumTopExperts>(
          warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
          params.mPtrScores + scoreOffset, params);

      if (laneIdx < params.mTopK) {
        int offset = warpIdx * MaxNumExperts + warpTopKExpertIdx[laneIdx];
        smemKIdx[offset] = static_cast<int8_t>(laneIdx);
        if (params.mPtrTopKWeights != nullptr) {
          params.mPtrTopKWeights[warpIdx * params.mTopK + laneIdx] =
              OutputT{warpTopKScore[laneIdx]};
        }
      }
    }  // end if (validToken)
  } else if (params.mPtrTopKPacked != nullptr) {
    if (validToken) {
      if (laneIdx < params.mTopK) {
        auto const expandedIdx = warpIdx * params.mTopK + laneIdx;
        auto const scoreIdx = params.mPtrTopKPacked[expandedIdx];
        int offset = warpIdx * MaxNumExperts + static_cast<int>(scoreIdx.idx);
        smemKIdx[offset] = static_cast<int8_t>(laneIdx);
        if (params.mPtrTopKWeights != nullptr) {
          params.mPtrTopKWeights[expandedIdx] = static_cast<OutputT>(scoreIdx.score);
        }
      }
    }
  }
  __syncthreads();

  // Each thread handles ExpertsPerThread contiguous experts.
  // Thread i handles experts [i * ExpertsPerThread, (i+1) * ExpertsPerThread).
  // Contiguous assignment ensures prefix sum ordering is correct.
  int accExpertCount[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    auto localExpIdx = expert - params.mLocalExpertsStartIdx;
    auto isLocal = localExpIdx >= 0 && localExpIdx < params.mNumLocalExperts &&
                   (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

    // Get the count of each expert and the offset for each token
    accExpertCount[e] = 0;
    if (isLocal) {
      int offset = expert;
      for (int j = 0; j < BlockKernelMaxNumTokens; j++) {
        if (smemKIdx[offset] >= 0) {
          smemOffset[offset] = static_cast<int8_t>(accExpertCount[e]);
          accExpertCount[e]++;
        }
        offset += MaxNumExperts;
      }
    }
  }
  __syncthreads();

  // Get the number of CTAs and the offset for each CTA.
  // Use cub::BlockScan's array overload: each thread holds ExpertsPerThread items,
  // and ExclusiveSum computes the prefix sum across all NumThreadsBlock * ExpertsPerThread
  // items in thread order — exactly matching our contiguous expert assignment.
  int32_t numCtaPerExpert[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    if (params.mIsPow2) {
      numCtaPerExpert[e] = divUpLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
    } else {
      numCtaPerExpert[e] = divUpTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
    }
    // Expand from CGA count to CTA count to keep the semantic stable with downstream kernels
    numCtaPerExpert[e] *= params.mClusterSizeInBatchDim;
  }
  int32_t ctaOffsetPerExpert[ExpertsPerThread];
  int32_t numNonExitingCtas;
  Scan(tempStorage).ExclusiveSum(numCtaPerExpert, ctaOffsetPerExpert, numNonExitingCtas);
  __syncthreads();  // Required barrier before reusing TempStorage for the next BlockScan

  // Compute padded expert scan counts (same array-overload pattern)
  int32_t tmpCountPerExpert[ExpertsPerThread];
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    if (params.mIsPow2) {
      tmpCountPerExpert[e] = divUpMulLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
    } else {
      tmpCountPerExpert[e] = divUpMulTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
    }
  }
  int32_t expertScanCountsPerExpert[ExpertsPerThread];
  Scan(tempStorage).ExclusiveSum(tmpCountPerExpert, expertScanCountsPerExpert);
  __syncthreads();

  // Write CTA configs for each expert this thread handles
#pragma unroll
  for (int e = 0; e < ExpertsPerThread; e++) {
    int expert = threadIdx.x * ExpertsPerThread + e;
    auto localExpIdx = expert - params.mLocalExpertsStartIdx;
    auto isLocal = localExpIdx >= 0 && localExpIdx < params.mNumLocalExperts &&
                   (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

    if (isLocal) {
      for (int cta = 0; cta < numCtaPerExpert[e]; ++cta) {
        int32_t const mappedLocalIdx =
            (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        params.mPtrCtaIdxXyToBatchIdx[ctaOffsetPerExpert[e] + cta] = mappedLocalIdx;
        // Write CTA-level MnLimits using ctaTile = cgaTile / clusterSize
        int32_t mnLimit1;
        int32_t mnLimit2;
        if (params.mIsPow2) {
          int32_t ctaPaddingLog2 = params.mPaddingLog2 - params.mClusterSizeLog2;
          mnLimit1 =
              mulLog2<int32_t>(ctaOffsetPerExpert[e] + cta + 1, ctaPaddingLog2);
          mnLimit2 = mulLog2<int32_t>(ctaOffsetPerExpert[e], ctaPaddingLog2) +
                     accExpertCount[e];
        } else {
          int32_t ctaTile = params.mTileTokensDim / params.mClusterSizeInBatchDim;
          mnLimit1 = (ctaOffsetPerExpert[e] + cta + 1) * ctaTile;
          mnLimit2 = ctaOffsetPerExpert[e] * ctaTile + accExpertCount[e];
        }
        params.mPtrCtaIdxXyToMnLimit[ctaOffsetPerExpert[e] + cta] = min(mnLimit1, mnLimit2);
      }
    }
  }

  // at this point, we can write out padded count
  if (threadIdx.x == 0) {
    int32_t permutedIdxSize;
    if (params.mIsPow2) {
      permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas >> params.mClusterSizeLog2, params.mPaddingLog2);
    } else {
      permutedIdxSize = (numNonExitingCtas / params.mClusterSizeInBatchDim) * params.mTileTokensDim;
    }
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

  for (int tokenIdx = 0; tokenIdx < params.mNumTokens; tokenIdx++) {
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      int expert = threadIdx.x * ExpertsPerThread + e;
      int offset = tokenIdx * MaxNumExperts + expert;
      if (smemKIdx[offset] >= 0) {
        auto localExpIdx = expert - params.mLocalExpertsStartIdx;
        auto isLocal = localExpIdx >= 0 && localExpIdx < params.mNumLocalExperts &&
                       (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

        int const expandedIdx = tokenIdx * params.mTopK + smemKIdx[offset];
        int const offsetWithinExpert = static_cast<int>(smemOffset[offset]);
        int const offsetForExpert = expertScanCountsPerExpert[e];
        int const permutedIdx =
            isLocal ? offsetForExpert + offsetWithinExpert : int32_t{-1};

        if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
        }
        if (params.mPtrPermutedIdxToExpandedIdx != nullptr && isLocal) {
          params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
        }
        if (params.mPtrPermutedIdxToTokenIdx != nullptr && isLocal) {
          params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
        }
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Trigger the secondary kernel AFTER all global memory writes (including permutation indices).
  // The downstream kernels depend on all routing outputs being visible.
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

void launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesBlockKernel, 1, numThreadsHist,
                        /*smemSize=*/0,  // No dynamic smem
                        stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 2. Cluster kernel — single-cluster fused kernel for ≤256 tokens (SM90+).
//    Uses distributed shared memory across 8 blocks in a cluster.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  using TypePacked = PackedScoreIdx<BaseType>;
  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

  __shared__ TypePacked __attribute((aligned(128)))
      smemPackedScoreIdx[NumWarps * KernelParams::MaxNumTopExperts];

  uint32_t const clusterBlockRank = blockIdx.x;
  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();
  auto warpTokenIdx = clusterBlockRank * NumWarps + warpIdx;
  auto scoreOffset = warpTokenIdx * params.mNumExperts;
  bool validToken = warpTokenIdx < params.mNumTokens;
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }

  if (params.mPtrScores != nullptr) {
    BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
    int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
    if (validToken) {
      KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                       KernelParams::MaxNumTopExperts>(
          warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
          params.mPtrScores + scoreOffset, params);
      if (laneIdx < params.mTopK) {
        smemPackedScoreIdx[warpIdx * params.mTopK + laneIdx] =
            TypePacked{warpTopKScore[laneIdx],
                       static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
      }
    }
  }

  __cluster_barrier_arrive();
  __cluster_barrier_wait();

  if (params.mPtrScores != nullptr) {
    routingPermutation<KernelParams, BaseType, NumThreads, NumWarps,
                       KernelParams::MaxNumTopExperts,
                       /*LoadExpertIdxFromGlobal=*/false>(params, smemPackedScoreIdx, warpIdx,
                                                          clusterBlockRank);
  } else {
    routingPermutation<KernelParams, BaseType, NumThreads, NumWarps,
                       KernelParams::MaxNumTopExperts,
                       /*LoadExpertIdxFromGlobal=*/true>(params, smemPackedScoreIdx, warpIdx,
                                                         clusterBlockRank);
  }
}
#else
__global__ void __launch_bounds__(NumThreads)
    routingIndicesClusterKernel(KernelParams /* params */) {
  assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

void launchClusterKernel(Data const& data, void* stream) {
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesClusterKernel, NumBlocksPerCluster,
                        NumThreads,
                        /*smemSize=*/0,  // No dynamic smem
                        stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 3. HistogramScores kernel — computes TopK from raw scores and initializes expert counts.
//    Used as step 1 of the multi-kernel pipeline when input is raw logits.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void
    __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024)
        routingIndicesHistogramScoresKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  // Cap actual thread count at 1024 when MaxNumExperts > 1024.
  static constexpr int NumThreadsBlock =
      KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts : 1024;

  // VecSize stays based on MaxNumExperts — each warp still processes all experts for one token.
  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

  int32_t const laneIdx = cutlass::arch::LaneId();
  int32_t const warpIdx = threadIdx.x / WarpSize;
  // Use NumThreadsBlock (actual thread count) for grid-stride warp/thread addressing
  int32_t const globalWarpIdx = blockIdx.x * NumThreadsBlock / WarpSize + warpIdx;
  int32_t const globalWarpStride = gridDim.x * NumThreadsBlock / WarpSize;
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Wait on primary grid.
  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

  // initialize the mPtrExpertCounts — use NumThreadsBlock for grid-stride
  int32_t expertCountsNum = 2 * params.mNumExperts;
  int32_t globalThreadIdx = blockIdx.x * NumThreadsBlock + threadIdx.x;
  int32_t globalThreadStride = gridDim.x * NumThreadsBlock;
  initArr(globalThreadIdx, expertCountsNum, globalThreadStride, params.mPtrExpertCounts, 0);

  // in this case, each warp represents a token, and we use a grid-stride loop
  // over all warps/tokens
  BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
  int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
  for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens;
       tokenIdx += globalWarpStride) {
    auto scoreOffset = tokenIdx * params.mNumExperts;

    KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                     KernelParams::MaxNumTopExperts>(
        warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
        params.mPtrScores + scoreOffset, params);

    if (laneIdx < params.mTopK) {
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(warpTopKScore[laneIdx]),
                                          static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
      params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packedScore;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Trigger secondary kernel AFTER writing all packed scores, so the next kernel
  // (routingIndicesHistogramKernel) sees the completed mPtrTopKPacked writes.
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
}

void launchHistogramScoresKernel(Data const& data, uint32_t maxNumBlocks,
                                 uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesHistogramScoresKernel, maxNumBlocks,
                        numThreadsHist,
                        /*smemSize=*/0,  // No dynamic smem
                        stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 4. Coop kernel — cooperative histogram + offsets via grid-sync.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream) {
  if (data.mNumExperts <= NumExperts128Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel,
                                 numBlocksCoop, numThreadsHist, /*smemSize=*/0, stream,
                                 NoOpPreprocess, NoOpPostprocess, NumExperts128Experts,
                                 NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts160Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel,
                                 numBlocksCoop, numThreadsHist, /*smemSize=*/0, stream,
                                 NoOpPreprocess, NoOpPostprocess, NumExperts160Experts,
                                 NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts256Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel,
                                 numBlocksCoop, numThreadsHist, /*smemSize=*/0, stream,
                                 NoOpPreprocess, NoOpPostprocess, NumExperts256Experts,
                                 NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts384Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel,
                                 numBlocksCoop, numThreadsHist, /*smemSize=*/0, stream,
                                 NoOpPreprocess, NoOpPostprocess, NumExperts384Experts,
                                 NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts512Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel,
                                 numBlocksCoop, numThreadsHist, /*smemSize=*/0, stream,
                                 NoOpPreprocess, NoOpPostprocess, NumExperts512Experts,
                                 NumTop8Experts);
  } else if (data.mNumExperts <= NumExperts576Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel,
                                 numBlocksCoop, numThreadsHist, /*smemSize=*/0, stream,
                                 NoOpPreprocess, NoOpPostprocess, NumExperts576Experts,
                                 NumTop8Experts);
  } else {
    FLASHINFER_WARN("Coop kernel does not support numExperts > %d", NumExperts576Experts);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 5-7. Launch wrappers for shared kernels (defined in RoutingKernel.cuh):
//      - InitExpertCounts (zero expert counts)
//      - Histogram kernel (histogram from packed TopK)
//      - Offsets kernel (prefix-scan + permutation)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingInitExpertCounts,
                                  (2 * data.mNumExperts - 1) / numThreadsHist + 1,
                                  numThreadsHist,
                                  /*smemSize=*/0,  // No dynamic smem
                                  stream);
}

void launchHistogramKernel(Data const& data, int numBlocksHistogram, uint32_t numThreadsHist,
                           void* stream) {
  LAUNCH_ROUTING_CUSTOM_NO_POLICY(data, false, routingIndicesHistogramKernel,
                                  numBlocksHistogram, numThreadsHist,
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

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Entry point
//
////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream) {
  TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr ||
                 data.mPtrTopKIds != nullptr)
      << "Routing kernel requires at least one input parameter";

  // When topK is already computed (mPtrTopKIds or mPtrTopKPacked without scores),
  // delegate to the shared post-topK pipeline which handles all path selection
  // (single-block, single-cluster, coop, multi-kernel) automatically.
  // No routing-method-specific logic needed.
  if (data.mPtrTopKIds != nullptr ||
      (data.mPtrTopKPacked != nullptr && data.mPtrScores == nullptr)) {
    if (data.mPtrTopKIds != nullptr) {
      TVM_FFI_ICHECK(data.mPtrTopKWeights != nullptr)
          << "When mPtrTopKIds is provided, mPtrTopKWeights must also be provided for "
             "custom routing.";
    }
    uint32_t const numThreadsHist =
        std::min(1024u, static_cast<uint32_t>(getMaxNumExperts(data.mNumExperts)));
    runPostTopKPipeline(data, numThreadsHist, stream);
    return;
  }

  // After this point, input is mPtrScores (raw logits that need topK computation).
  TVM_FFI_ICHECK(data.mPtrScores != nullptr) << "Expected mPtrScores to be non-null at this "
                                                 "point.";
  TVM_FFI_ICHECK(data.mPtrPermutedIdxSize != nullptr &&
                 data.mPtrCtaIdxXyToBatchIdx != nullptr &&
                 data.mPtrCtaIdxXyToMnLimit != nullptr &&
                 data.mPtrNumNonExitingCtas != nullptr)
      << "Custom routing kernel expects permuted idx and grouped Gemm launch config buffers";
  TVM_FFI_ICHECK_LE(data.mTopK, static_cast<int32_t>(MaxSupportedTopExperts))
      << "Routing kernel expects topK experts <= " << MaxSupportedTopExperts << ", got "
      << data.mTopK;
  TVM_FFI_ICHECK_LE(data.mNumExperts, static_cast<int32_t>(MaxSupportedExperts))
      << "Routing kernel expects #experts " << data.mNumExperts << " to be no more than "
      << MaxSupportedExperts << ".";
  TVM_FFI_ICHECK_EQ(data.mNumExperts % 4, 0)
      << "Routing kernel expects #experts " << data.mNumExperts
      << " to be a multiple of 4.";

  bool const useSingleBlock = data.mNumTokens <= BlockKernelMaxNumTokens;
  bool const useSingleCluster = data.mNumTokens <= MaxNumTokensSingleClusterScores;

  if (!useSingleCluster && !useSingleBlock) {
    TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr)
        << "When #tokens is large, `mPtrTopKPacked` is a required input.";
    TVM_FFI_ICHECK(data.mPtrExpertCounts != nullptr)
        << "When #tokens is large, `mPtrExpertCounts` is a required input.";
  }

  uint32_t const numThreadsHist =
      std::min(1024u, static_cast<uint32_t>(getMaxNumExperts(data.mNumExperts)));

  // PDL overlap control: intermediate routing kernels allow the next routing kernel to overlap
  // (mPdlOverlapWithNext = mUsePdl). The LAST routing kernel disables overlap so the consumer
  // GEMM (which may not have cudaGridDependencySynchronize for routing data) can't start early.
  // We need a mutable copy since `data` is const.
  Data mutableData = data;
  bool const pdl = data.mUsePdl;

  if (useSingleBlock) {
    //@TODO: For now we use the single block kernel for cases with token number no larger than 4.
    // We will future tune this threshold based on the performance.
    mutableData.mPdlOverlapWithNext = false;  // Last kernel — don't let consumer overlap
    launchBlockKernel(mutableData, numThreadsHist, stream);
  } else if (useSingleCluster) {
    mutableData.mPdlOverlapWithNext = false;  // Last kernel — don't let consumer overlap
    launchClusterKernel(mutableData, stream);
  } else {
    // mPtrScores path: compute topK first via fused scores+histogram kernel,
    // then use coop or multi-kernel pipeline for histogram + offsets.
    uint32_t const maxNumBlocks = 1024;

    // Step 1: Compute topK from raw scores and write packed results to mPtrTopKPacked.
    mutableData.mPdlOverlapWithNext = pdl;  // Intermediate — allow next routing kernel to
                                            // overlap
    launchHistogramScoresKernel(mutableData, maxNumBlocks, numThreadsHist, stream);

    // Step 2+3: Histogram + Offsets — try coop path first, fall back to multi-kernel.
    // Coop kernel fuses histogram + offsets into a single cooperative launch.
    // Requires SM90+ (kernel uses grid-sync), numExperts <= 1024, and enough SM capacity.
    static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;
    bool const canUseCoop = (smMajor >= 9) && (data.mNumExperts <= 1024) &&
                            (data.mPtrPermutedIdxSize != nullptr);
    bool useCoop = false;
    int numBlocksCoop = 0;

    if (canUseCoop) {
      static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
      numBlocksCoop = smCount - 8;  // Reserve 8 SMs for overlapping kernels
      int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;
      useCoop = (data.mNumTokens <= maxTokensCoop);
    }

    if (useCoop) {
      // Coop path: 2 kernels (scores+topK → coop histogram+offsets) instead of 3.
      mutableData.mPdlOverlapWithNext =
          pdl;  // Intermediate — allow next routing kernel to overlap
      launchInitExpertCounts(mutableData, numThreadsHist, stream);
      mutableData.mPdlOverlapWithNext = false;  // Last kernel — don't let consumer overlap
      launchCoopKernel(mutableData, numBlocksCoop, numThreadsHist, stream);
    } else {
      // Multi-kernel path: 3 kernels (scores+topK → histogram → offsets).
      // Note: histogramScoresKernel already zeroes expert counts, so no initExpertCounts
      // needed.
      uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
      uint32_t const histogramEltsPerBlock = 8 * numThreadsHist;
      uint32_t const offsetEltsPerBlock =
          NumEltsPerOffsetTilePerThread * numThreadsHist;

      int const numBlocksHistogram = std::min(
          (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
      int const numBlocksOffsets = std::min(
          (expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

      mutableData.mPdlOverlapWithNext =
          pdl;  // Intermediate — allow next routing kernel to overlap
      launchHistogramKernel(mutableData, numBlocksHistogram, numThreadsHist, stream);
      mutableData.mPdlOverlapWithNext = false;  // Last kernel — don't let consumer overlap
      launchOffsetsKernel(mutableData, numBlocksOffsets, numThreadsHist, stream);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingCustom
}  // namespace moe::dev::routing
