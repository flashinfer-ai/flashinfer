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
//   4. routingIndicesCoopKernel       — cooperative histogram + offsets (defined in
//   RoutingKernel.cuh)
//   5. routingInitExpertCounts        — zero expert counts (defined in RoutingKernel.cuh)
//   6. routingIndicesHistogramKernel  — histogram from packed TopK (defined in RoutingKernel.cuh)
//   7. routingIndicesOffsetsKernel    — prefix-scan + permutation (defined in RoutingKernel.cuh)

#include <cstdlib>
#include <string>
#include <type_traits>

#include "flashinfer/trtllm/common/cudaUtils.h"
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
__global__ void __launch_bounds__(KernelParams::MaxNumExperts <= 1024 ? KernelParams::MaxNumExperts
                                                                      : 1024)
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
        auto const expandedIdx = warpIdx * params.mTopK + laneIdx;
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = int32_t{-1};
        }
        auto expertIdx = params.mPtrTopKIds[expandedIdx];
        if (expertIdx > -1 && expertIdx < params.mNumExperts) {
          int offset = warpIdx * MaxNumExperts + expertIdx;
          smemKIdx[offset] = static_cast<int8_t>(laneIdx);
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
        // Routing replay: record selected expert IDs. Layout: [num_tokens, topK]
        // -- same indexing as mPtrTopKPacked / mPtrTopKWeights.
        if (params.mPtrRoutingReplayOut != nullptr) {
          params.mPtrRoutingReplayOut[warpIdx * params.mTopK + laneIdx] =
              static_cast<int16_t>(warpTopKExpertIdx[laneIdx]);
        }
      }
    }  // end if (validToken)
  } else if (params.mPtrTopKPacked != nullptr) {
    if (validToken) {
      if (laneIdx < params.mTopK) {
        auto const expandedIdx = warpIdx * params.mTopK + laneIdx;
        // Pre-initialize to -1: when duplicate expert IDs appear, multiple
        // lanes race on the same smemKIdx slot and only one wins.  Losing
        // lanes would skip the else-branch below, leaving their entry as
        // uninitialized garbage.  Writing -1 up front makes every entry safe.
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = int32_t{-1};
        }
        auto const scoreIdx = params.mPtrTopKPacked[expandedIdx];
        int const expertIdx = static_cast<int>(scoreIdx.idx);
        if (expertIdx >= 0 && expertIdx < params.mNumExperts) {
          int const offset = warpIdx * MaxNumExperts + expertIdx;
          smemKIdx[offset] = static_cast<int8_t>(laneIdx);
          if (params.mPtrTopKWeights != nullptr) {
            params.mPtrTopKWeights[expandedIdx] = static_cast<OutputT>(scoreIdx.score);
          }
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
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    auto isLocal = localExpIdx >= 0 && localExpIdx < localExpertExtent &&
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
    auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
    auto isLocal = localExpIdx >= 0 && localExpIdx < localExpertExtent &&
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
          mnLimit1 = mulLog2<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mPaddingLog2);
          mnLimit2 =
              mulLog2<int32_t>(ctaOffsetPerExpert[e], params.mPaddingLog2) + accExpertCount[e];
        } else {
          mnLimit1 = mulTileN<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mTileTokensDim);
          mnLimit2 =
              mulTileN<int32_t>(ctaOffsetPerExpert[e], params.mTileTokensDim) + accExpertCount[e];
        }
        params.mPtrCtaIdxXyToMnLimit[ctaOffsetPerExpert[e] + cta] = min(mnLimit1, mnLimit2);
      }
    }
  }

  // at this point, we can write out padded count
  if (threadIdx.x == 0) {
    int32_t permutedIdxSize;
    if (params.mIsPow2) {
      permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    } else {
      permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
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
        auto localExpertExtent = params.mNumLocalExperts << params.mLocalExpertsStrideLog2;
        auto isLocal = localExpIdx >= 0 && localExpIdx < localExpertExtent &&
                       (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

        int const expandedIdx = tokenIdx * params.mTopK + smemKIdx[offset];
        // Only load smemOffset for local experts; the histogram phase only
        // writes it for local experts, so remote entries are uninitialized.
        int const permutedIdx =
            isLocal ? expertScanCountsPerExpert[e] + static_cast<int>(smemOffset[offset])
                    : int32_t{-1};

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

// Returns whether a compiled tier covered the runtime (numExperts, topK) and the
// kernel was actually launched. A false return means no routing output was written;
// the caller (run) must not proceed to the downstream pipeline in that case.
bool launchBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream) {
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesBlockKernel, 1, numThreadsHist,
                        /*smemSize=*/0,  // No dynamic smem
                        stream);
  return queryPolicyHasCompiledTier(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Warp-level exclusive scan for the dynamic block kernel.
// Computes dual prefix sums across all threads in the block using a two-level scan:
// first within each warp, then across warps.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NumExpertWarps>
__device__ __forceinline__ void warpExclusiveScan(int32_t val1, int32_t val2, int32_t laneIdx,
                                                  int32_t warpIdx, int32_t* warpTotals1,
                                                  int32_t* warpTotals2, int32_t& prefix1,
                                                  int32_t& prefix2, int32_t& totalSum1) {
  static_assert(NumExpertWarps <= WarpSize,
                "NumExpertWarps must fit in one warp for the cross-warp scan");

  int32_t inc1 = val1, inc2 = val2;
#pragma unroll
  for (int j = 1; j < WarpSize; j *= 2) {
    int32_t n1 = __shfl_up_sync(0xffffffff, inc1, j);
    int32_t n2 = __shfl_up_sync(0xffffffff, inc2, j);
    if (laneIdx >= j) {
      inc1 += n1;
      inc2 += n2;
    }
  }

  if (warpIdx < NumExpertWarps && laneIdx == WarpSize - 1) {
    warpTotals1[warpIdx] = inc1;
    warpTotals2[warpIdx] = inc2;
  }
  __syncthreads();

  if (warpIdx == 0) {
    int32_t wt1 = (laneIdx < NumExpertWarps) ? warpTotals1[laneIdx] : 0;
    int32_t wt2 = (laneIdx < NumExpertWarps) ? warpTotals2[laneIdx] : 0;
#pragma unroll
    for (int j = 1; j < NumExpertWarps; j *= 2) {
      int32_t n1 = __shfl_up_sync(0xffffffff, wt1, j);
      int32_t n2 = __shfl_up_sync(0xffffffff, wt2, j);
      if (laneIdx >= j) {
        wt1 += n1;
        wt2 += n2;
      }
    }
    if (laneIdx < NumExpertWarps) {
      warpTotals1[laneIdx] = wt1;
      warpTotals2[laneIdx] = wt2;
    }
  }
  __syncthreads();

  totalSum1 = warpTotals1[NumExpertWarps - 1];
  int32_t wp1 = (warpIdx > 0 && warpIdx < NumExpertWarps) ? warpTotals1[warpIdx - 1] : 0;
  int32_t wp2 = (warpIdx > 0 && warpIdx < NumExpertWarps) ? warpTotals2[warpIdx - 1] : 0;
  prefix1 = inc1 - val1 + wp1;
  prefix2 = inc2 - val2 + wp2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 1b. Dynamic block kernel — single-block kernel with dynamic shared memory.
//     Handles ≤DynBlockKernelMaxNumTokens tokens and ≤DynBlockKernelMaxNumExperts experts.
//     Extends the static block kernel to more tokens by using dynamic smem and loop-based
//     warp-per-token processing instead of fixed warpIdx mapping.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void routingIndicesDynBlockKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  using TypePacked = PackedScoreIdx<BaseType>;
  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
  static constexpr int NumThreadsExperts = MaxNumExperts <= 1024 ? MaxNumExperts : 1024;
  static constexpr int ExpertsPerThread = MaxNumExperts / NumThreadsExperts;
  static constexpr int NumExpertWarps = NumThreadsExperts / WarpSize;
  static constexpr int VecSize = MaxNumExperts / WarpSize;

  static_assert(MaxNumExperts % WarpSize == 0);
  static_assert(MaxNumExperts % NumThreadsExperts == 0);

  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();
  int32_t const numWarps = blockDim.x / WarpSize;

  extern __shared__ char dynSmem[];
  int const numSlots = params.mNumTokens * MaxNumExperts;
  int8_t* smemKIdx = reinterpret_cast<int8_t*>(dynSmem);
  int16_t* smemOffset = reinterpret_cast<int16_t*>(dynSmem + numSlots);
  char* warpBase = dynSmem + numSlots + numSlots * 2;
  warpBase = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(warpBase) + 127) & ~127);
  int32_t* warpTotals = reinterpret_cast<int32_t*>(warpBase);
  int32_t* warpTotals2 = warpTotals + NumExpertWarps;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  for (int i = threadIdx.x; i < numSlots; i += blockDim.x) {
    smemKIdx[i] = int8_t{-1};
  }
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  // Phase 1: TopK — one warp per token (loop when numTokens > numWarps)
  for (int tokenIdx = warpIdx; tokenIdx < params.mNumTokens; tokenIdx += numWarps) {
    if (params.mPtrTopKIds != nullptr) {
      if (laneIdx < params.mTopK) {
        auto const expandedIdx = tokenIdx * params.mTopK + laneIdx;
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = int32_t{-1};
        }
        auto expertIdx = params.mPtrTopKIds[expandedIdx];
        if (expertIdx > -1 && expertIdx < params.mNumExperts) {
          smemKIdx[tokenIdx * MaxNumExperts + expertIdx] = static_cast<int8_t>(laneIdx);
        }
      }
    } else if (params.mPtrScores != nullptr) {
      BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
      int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];

      auto scoreOff = tokenIdx * params.mNumExperts;
      KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                       KernelParams::MaxNumTopExperts>(
          warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
          params.mPtrScores + scoreOff, params);

      if (laneIdx < params.mTopK) {
        smemKIdx[tokenIdx * MaxNumExperts + warpTopKExpertIdx[laneIdx]] =
            static_cast<int8_t>(laneIdx);
        if (params.mPtrTopKWeights != nullptr) {
          params.mPtrTopKWeights[tokenIdx * params.mTopK + laneIdx] =
              OutputT{warpTopKScore[laneIdx]};
        }
        // Routing replay: record selected expert IDs. Layout: [num_tokens, topK]
        // -- same indexing as mPtrTopKWeights.
        if (params.mPtrRoutingReplayOut != nullptr) {
          params.mPtrRoutingReplayOut[tokenIdx * params.mTopK + laneIdx] =
              static_cast<int16_t>(warpTopKExpertIdx[laneIdx]);
        }
      }
    } else if (params.mPtrTopKPacked != nullptr) {
      if (laneIdx < params.mTopK) {
        auto const expandedIdx = tokenIdx * params.mTopK + laneIdx;
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
          params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = int32_t{-1};
        }
        auto scoreIdx = params.mPtrTopKPacked[expandedIdx];
        int const expertIdx = static_cast<int>(scoreIdx.idx);
        if (expertIdx >= 0 && expertIdx < params.mNumExperts) {
          smemKIdx[tokenIdx * MaxNumExperts + expertIdx] = static_cast<int8_t>(laneIdx);
          if (params.mPtrTopKWeights != nullptr) {
            params.mPtrTopKWeights[expandedIdx] = static_cast<OutputT>(scoreIdx.score);
          }
        }
      }
    }
  }
  __syncthreads();

  // Phase 2: Histogram — count tokens per expert
  int accExpertCount[ExpertsPerThread];
  if (threadIdx.x < NumThreadsExperts) {
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      int expert = threadIdx.x * ExpertsPerThread + e;
      auto localExpIdx = expert - params.mLocalExpertsStartIdx;
      auto isLocal = localExpIdx >= 0 &&
                     localExpIdx < (params.mNumLocalExperts << params.mLocalExpertsStrideLog2) &&
                     (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
      accExpertCount[e] = 0;
      if (isLocal) {
        int offset = expert;
        for (int j = 0; j < params.mNumTokens; j++) {
          if (smemKIdx[offset] >= 0) {
            smemOffset[offset] = static_cast<int16_t>(accExpertCount[e]);
            accExpertCount[e]++;
          }
          offset += MaxNumExperts;
        }
      }
    }
  } else {
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      accExpertCount[e] = 0;
    }
  }

  // Phase 3: Prefix-scan (merged dual warp-level scan)
  int32_t numCtaPerExpert[ExpertsPerThread];
  int32_t tmpCountPerExpert[ExpertsPerThread];
  int32_t ctaOffsetPerExpert[ExpertsPerThread];
  int32_t expertScanCountsPerExpert[ExpertsPerThread];
  int32_t numNonExitingCtas;
  {
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      if (threadIdx.x < NumThreadsExperts) {
        if (params.mIsPow2) {
          numCtaPerExpert[e] = divUpLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
          tmpCountPerExpert[e] = divUpMulLog2<int32_t>(accExpertCount[e], params.mPaddingLog2);
        } else {
          numCtaPerExpert[e] = divUpTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
          tmpCountPerExpert[e] = divUpMulTileN<int32_t>(accExpertCount[e], params.mTileTokensDim);
        }
      } else {
        numCtaPerExpert[e] = 0;
        tmpCountPerExpert[e] = 0;
      }
    }

    int32_t localPrefix1[ExpertsPerThread], localPrefix2[ExpertsPerThread];
    int32_t threadTotal1 = 0, threadTotal2 = 0;
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      localPrefix1[e] = threadTotal1;
      localPrefix2[e] = threadTotal2;
      threadTotal1 += numCtaPerExpert[e];
      threadTotal2 += tmpCountPerExpert[e];
    }

    int32_t threadPrefix1, threadPrefix2;
    warpExclusiveScan<NumExpertWarps>(threadTotal1, threadTotal2, laneIdx, warpIdx, warpTotals,
                                      warpTotals2, threadPrefix1, threadPrefix2, numNonExitingCtas);

#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      ctaOffsetPerExpert[e] = threadPrefix1 + localPrefix1[e];
      expertScanCountsPerExpert[e] = threadPrefix2 + localPrefix2[e];
    }
  }

  // Phase 4: CTA configs
  if (threadIdx.x < NumThreadsExperts) {
#pragma unroll
    for (int e = 0; e < ExpertsPerThread; e++) {
      int expert = threadIdx.x * ExpertsPerThread + e;
      auto localExpIdx = expert - params.mLocalExpertsStartIdx;
      auto isLocal = localExpIdx >= 0 &&
                     localExpIdx < (params.mNumLocalExperts << params.mLocalExpertsStrideLog2) &&
                     (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
      if (isLocal) {
        for (int cta = 0; cta < numCtaPerExpert[e]; ++cta) {
          int32_t const mappedLocalIdx =
              (expert - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
          params.mPtrCtaIdxXyToBatchIdx[ctaOffsetPerExpert[e] + cta] = mappedLocalIdx;
          int32_t mnLimit1, mnLimit2;
          if (params.mIsPow2) {
            mnLimit1 = mulLog2<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mPaddingLog2);
            mnLimit2 =
                mulLog2<int32_t>(ctaOffsetPerExpert[e], params.mPaddingLog2) + accExpertCount[e];
          } else {
            mnLimit1 = mulTileN<int32_t>(ctaOffsetPerExpert[e] + cta + 1, params.mTileTokensDim);
            mnLimit2 =
                mulTileN<int32_t>(ctaOffsetPerExpert[e], params.mTileTokensDim) + accExpertCount[e];
          }
          params.mPtrCtaIdxXyToMnLimit[ctaOffsetPerExpert[e] + cta] = min(mnLimit1, mnLimit2);
        }
      }
    }
  }

  if (threadIdx.x == 0) {
    int32_t permutedIdxSize;
    if (params.mIsPow2) {
      permutedIdxSize = mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2);
    } else {
      permutedIdxSize = mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
    }
    params.mPtrPermutedIdxSize[0] = permutedIdxSize;
    params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
  }

  // Phase 5: Permutation
  if (threadIdx.x < NumThreadsExperts) {
    for (int tokenIdx = 0; tokenIdx < params.mNumTokens; tokenIdx++) {
#pragma unroll
      for (int e = 0; e < ExpertsPerThread; e++) {
        int expert = threadIdx.x * ExpertsPerThread + e;
        int offset = tokenIdx * MaxNumExperts + expert;
        if (smemKIdx[offset] >= 0) {
          auto localExpIdx = expert - params.mLocalExpertsStartIdx;
          auto isLocal =
              localExpIdx >= 0 &&
              localExpIdx < (params.mNumLocalExperts << params.mLocalExpertsStrideLog2) &&
              (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;

          int const expandedIdx = tokenIdx * params.mTopK + smemKIdx[offset];
          // Only load smemOffset for local experts; Phase 2 only writes it
          // for local experts, so remote entries are uninitialized.
          int const permutedIdx =
              isLocal ? expertScanCountsPerExpert[e] + static_cast<int>(smemOffset[offset])
                      : int32_t{-1};

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
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

// Returns whether a compiled tier covered the runtime (numExperts, topK) and the
// kernel was actually launched (see launchBlockKernel).
bool launchDynBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream) {
  int32_t const maxExperts = queryDispatchedMaxExperts(data);
  int const numSlots = data.mNumTokens * maxExperts;
  int const smemSize = numSlots + numSlots * 2 + 128 +
                       2 * (maxExperts / WarpSize) * static_cast<int>(sizeof(int32_t));
  int const threads =
      std::min(std::max(data.mNumTokens * static_cast<int>(WarpSize), maxExperts), 1024);

  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesDynBlockKernel, 1, threads, smemSize, stream);
  return queryPolicyHasCompiledTier(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 2. Cluster kernel — single-cluster fused kernel for ≤256 tokens (SM90+).
//    Uses distributed shared memory across 8 blocks in a cluster.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

static constexpr int ClusterBlockDim256 = NumExperts256Experts;
static constexpr int ClusterBlockDim512 = NumExperts512Experts;
static constexpr int ClusterBlockDim1024 = NumThreads;
static constexpr int MaxNumTokensClusterScores256 =
    NumBlocksPerCluster * (ClusterBlockDim256 / WarpSize);
static constexpr int MaxNumTokensClusterScores512 =
    NumBlocksPerCluster * (ClusterBlockDim512 / WarpSize);

template <typename TierT, typename TierListT>
struct PrependTier;

template <typename TierT, typename... Tiers>
struct PrependTier<TierT, TierList<Tiers...>> {
  using type = TierList<TierT, Tiers...>;
};

template <int ClusterBlockDim, typename TierListT>
struct FilterClusterTiers;

template <int ClusterBlockDim>
struct FilterClusterTiers<ClusterBlockDim, TierList<>> {
  using type = TierList<>;
};

template <int ClusterBlockDim, typename First, typename... Rest>
struct FilterClusterTiers<ClusterBlockDim, TierList<First, Rest...>> {
  using Tail = typename FilterClusterTiers<ClusterBlockDim, TierList<Rest...>>::type;
  static constexpr bool IsValid =
      First::kExperts <= ClusterBlockDim || First::kExperts % ClusterBlockDim == 0;
  using type = std::conditional_t<IsValid, typename PrependTier<First, Tail>::type, Tail>;
};

template <int ClusterBlockDim, typename PreProc, typename PostProc>
struct ClusterPolicyTraits {
  using Pairs = typename FilterClusterTiers<ClusterBlockDim,
                                            typename PolicyTraits<PreProc, PostProc>::Pairs>::type;
};

template <typename KernelParams, typename BaseType, int ClusterBlockDim, int ClusterNumWarps>
__device__ __forceinline__ void routingIndicesClusterKernelBody(
    KernelParams params, PackedScoreIdx<BaseType>* smemPackedScoreIdx) {
  using InputT = typename KernelParams::InputT;
  using TypePacked = PackedScoreIdx<BaseType>;
  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;

  uint32_t const clusterBlockRank = blockIdx.x;
  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();
  auto warpTokenIdx = clusterBlockRank * ClusterNumWarps + warpIdx;
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
            TypePacked{warpTopKScore[laneIdx], static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
        // Routing replay: record selected expert IDs. Layout: [num_tokens, topK]
        // -- warpTokenIdx is the global token index for this warp.
        if (params.mPtrRoutingReplayOut != nullptr) {
          params.mPtrRoutingReplayOut[warpTokenIdx * params.mTopK + laneIdx] =
              static_cast<int16_t>(warpTopKExpertIdx[laneIdx]);
        }
      }
    }
  }

  __cluster_barrier_arrive();
  __cluster_barrier_wait();

  if (params.mPtrScores != nullptr) {
    routingPermutation<KernelParams, BaseType, ClusterBlockDim, ClusterNumWarps,
                       KernelParams::MaxNumTopExperts, /*LoadExpertIdxFromGlobal=*/false>(
        params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
  } else {
    routingPermutation<KernelParams, BaseType, ClusterBlockDim, ClusterNumWarps,
                       KernelParams::MaxNumTopExperts, /*LoadExpertIdxFromGlobal=*/true>(
        params, smemPackedScoreIdx, warpIdx, clusterBlockRank);
  }
}

template <typename KernelParams, int ClusterBlockDim = NumThreads>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __cluster_dims__(NumBlocksPerCluster, 1, 1) __launch_bounds__(ClusterBlockDim)
    routingIndicesClusterKernel(KernelParams params) {
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  using TypePacked = PackedScoreIdx<BaseType>;
  static constexpr int NumWarpsBlock = ClusterBlockDim / WarpSize;
  static_assert(ClusterBlockDim % WarpSize == 0);
  static_assert(ClusterBlockDim <= NumThreads);
  __shared__ TypePacked __attribute((
      aligned(128))) smemPackedScoreIdx[NumWarpsBlock * KernelParams::MaxNumTopExperts];
  routingIndicesClusterKernelBody<KernelParams, BaseType, ClusterBlockDim, NumWarpsBlock>(
      params, smemPackedScoreIdx);
}
#else
__global__ void __launch_bounds__(ClusterBlockDim)
    routingIndicesClusterKernel(KernelParams /* params */) {
  assert(false && "routingIndicesClusterKernel is only supported on SM90+ architectures");
}
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

template <typename ParamsT, int ClusterBlockDim>
void launchClusterKernelInstance(Data const& data, void* stream) {
  static_assert(ClusterBlockDim % WarpSize == 0);
  static_assert(ClusterBlockDim <= NumThreads);

  cudaLaunchConfig_t config{};
  config.gridDim = NumBlocksPerCluster;
  config.blockDim = ClusterBlockDim;
  config.dynamicSmemBytes = 0;
  config.stream = (cudaStream_t)stream;

  cudaLaunchAttribute attributes[2] = {};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = int(data.mUsePdl);
  attributes[1].id = cudaLaunchAttributeCooperative;
  attributes[1].val.cooperative = 0;
  config.attrs = attributes;
  config.numAttrs = 2;

  auto params = ParamsT::setKernelParams(data);
  auto kernelTyped = routingIndicesClusterKernel<ParamsT, ClusterBlockDim>;
  CHECK_CUDA_ERROR(cudaLaunchKernelEx(&config, kernelTyped, params));
}

template <int ClusterBlockDim, typename PreProc, typename PostProc, int MaxNumExperts,
          int MaxNumTopExperts>
void launchClusterKernelForTier(Data const& data, void* stream) {
  using ExpertSelect = TopKExpertSelect<PreProc, PostProc>;
  if (data.mDtypeOutput == tg::Dtype::Fp32) {
    using ParamsT = KernelParams<float, float, MaxNumExperts, MaxNumTopExperts, ExpertSelect>;
    launchClusterKernelInstance<ParamsT, ClusterBlockDim>(data, stream);
  } else if (data.mDtypeOutput == tg::Dtype::Bfloat16 && data.mDtypeInput == tg::Dtype::Fp32) {
    using ParamsT =
        KernelParams<float, __nv_bfloat16, MaxNumExperts, MaxNumTopExperts, ExpertSelect>;
    launchClusterKernelInstance<ParamsT, ClusterBlockDim>(data, stream);
  } else if (data.mDtypeOutput == tg::Dtype::Bfloat16) {
    using ParamsT =
        KernelParams<__nv_bfloat16, __nv_bfloat16, MaxNumExperts, MaxNumTopExperts, ExpertSelect>;
    launchClusterKernelInstance<ParamsT, ClusterBlockDim>(data, stream);
  } else {
    FLASHINFER_WARN("Unsupported dtypeOutput");
  }
}

// Returns whether the cluster tier dispatch found a covering tier (ClusterPolicyTraits
// is a filtered subset of PolicyTraits, so it may reject a config PolicyTraits covers).
template <int ClusterBlockDim, typename PreProc, typename PostProc>
bool launchClusterKernelForPolicy(Data const& data, void* stream) {
  using Pairs = typename ClusterPolicyTraits<ClusterBlockDim, PreProc, PostProc>::Pairs;
  bool dispatched =
      dispatchTierPairs(static_cast<Pairs*>(nullptr), data, [&](auto eTag, auto kTag) {
        launchClusterKernelForTier<ClusterBlockDim, PreProc, PostProc, decltype(eTag)::value,
                                   decltype(kTag)::value>(data, stream);
      });
  if (!dispatched) {
    FLASHINFER_WARN("No tier covers numExperts=%d topK=%d", data.mNumExperts, data.mTopK);
  }
  return dispatched;
}

template <int ClusterBlockDim>
bool launchClusterKernelForBlockDim(Data const& data, void* stream) {
  bool dispatched = false;
  dispatchRoutingPolicy(data, [&](auto preProc, auto postProc, char const* /*policyName*/) {
    dispatched =
        launchClusterKernelForPolicy<ClusterBlockDim, decltype(preProc), decltype(postProc)>(
            data, stream);
  });
  return dispatched;
}

// Returns whether a compiled tier covered the runtime (numExperts, topK) and the
// kernel was actually launched (see launchBlockKernel).
bool launchClusterKernel(Data const& data, void* stream) {
  // Each warp owns one token, so the reduced-thread cluster variants have lower token capacity.
  // Use them only where the requested token count fits; otherwise keep the original 1024-thread
  // launch.
  if (data.mNumTokens <= MaxNumTokensClusterScores256) {
    return launchClusterKernelForBlockDim<ClusterBlockDim256>(data, stream);
  }
  if (data.mNumTokens <= MaxNumTokensClusterScores512) {
    return launchClusterKernelForBlockDim<ClusterBlockDim512>(data, stream);
  }

  bool const useNoOpSoftmaxScores = data.mPtrScores != nullptr &&
                                    data.mPreprocessType == RoutingPreprocessType::None &&
                                    data.mPostprocessType == RoutingPostprocessType::Softmax;
  if (useNoOpSoftmaxScores) {
    return launchClusterKernelForPolicy<ClusterBlockDim1024, NoOpPreprocess, SoftmaxPostprocess>(
        data, stream);
  }

  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesClusterKernel, NumBlocksPerCluster, NumThreads,
                        /*smemSize=*/0,  // No dynamic smem
                        stream);
  return queryPolicyHasCompiledTier(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 3. HistogramScores kernel — computes TopK from raw scores and initializes expert counts.
//    Used as step 1 of the multi-kernel pipeline when input is raw logits.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MaxNumExperts, int MaxNumTopExperts>
struct HistogramScoresLaunchConfig : DefaultRoutingLaunchConfig<MaxNumExperts, MaxNumTopExperts> {
  static constexpr int DefaultBlockDim =
      DefaultRoutingLaunchConfig<MaxNumExperts, MaxNumTopExperts>::BlockDim;
  static constexpr int HistogramScoresBlockDim = NumExperts256Experts;

  // This kernel uses one warp per token and keeps per-warp arrays sized by both the expert tier and
  // the topK tier. The 256-expert tier already launches with 256 threads; for larger tiers, fewer
  // warps per CTA gives each thread more register headroom while preserving total warp-level
  // parallelism by scaling the grid cap below.
  static constexpr bool UseHistogramScoresBlockDim = DefaultBlockDim > HistogramScoresBlockDim;
  static constexpr int BlockDim =
      UseHistogramScoresBlockDim ? HistogramScoresBlockDim : DefaultBlockDim;

  static_assert(BlockDim % WarpSize == 0);
  static_assert(BlockDim <= NumThreads);

  static int blockDim(Data const& /*data*/, int /*numThreads*/) { return BlockDim; }

  static int gridDim(Data const& data, int numBlocks, int /*blockDim*/) {
    if constexpr (UseHistogramScoresBlockDim) {
      static constexpr int NumWarpsBlock = BlockDim / WarpSize;
      static constexpr int MaxBlockScale = (DefaultBlockDim + BlockDim - 1) / BlockDim;
      int const tokenBlocks =
          (static_cast<int>(data.mNumTokens) + NumWarpsBlock - 1) / NumWarpsBlock;
      int const scaledMaxBlocks = numBlocks * MaxBlockScale;
      int const selectedBlocks = tokenBlocks < scaledMaxBlocks ? tokenBlocks : scaledMaxBlocks;
      return selectedBlocks > 0 ? selectedBlocks : 1;
    } else {
      return DefaultRoutingLaunchConfig<MaxNumExperts, MaxNumTopExperts>::gridDim(data, numBlocks,
                                                                                  DefaultBlockDim);
    }
  }
};

template <typename ExpertSelect, int MaxNumExperts, int MaxNumTopExperts>
struct HistogramScoresKernelConfig : HistogramScoresLaunchConfig<MaxNumExperts, MaxNumTopExperts> {
};

template <typename PreProc, typename PostProc>
struct HistogramScoresPolicyTraits : PolicyTraits<PreProc, PostProc> {};

template <>
struct HistogramScoresPolicyTraits<NoOpPreprocess, SoftmaxPostprocess> {
  using Pairs = TierList<Tier<128, 4>, Tier<128, 8>, Tier<160, 8>, Tier<256, 8>, Tier<256, 16>,
                         Tier<512, 8>, Tier<512, 16>, Tier<512, 22>, Tier<512, 32>, Tier<576, 8>,
                         Tier<768, 32>, Tier<1024, 32>, Tier<1536, 32>, Tier<2048, 32>>;
};

template <typename KernelParams>
__global__ void __launch_bounds__(
    HistogramScoresKernelConfig<typename KernelParams::ExpertSelectPolicy,
                                KernelParams::MaxNumExperts,
                                KernelParams::MaxNumTopExperts>::BlockDim)
    routingIndicesHistogramScoresKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using BaseType = typename KernelParams::ExpertSelectPolicy::template BaseType<InputT>;
  static constexpr int NumThreadsBlock =
      HistogramScoresKernelConfig<typename KernelParams::ExpertSelectPolicy,
                                  KernelParams::MaxNumExperts,
                                  KernelParams::MaxNumTopExperts>::BlockDim;

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
  for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride) {
    auto scoreOffset = tokenIdx * params.mNumExperts;

    KernelParams::ExpertSelectPolicy::template apply<BaseType, InputT, VecSize,
                                                     KernelParams::MaxNumTopExperts>(
        warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
        params.mPtrScores + scoreOffset, params);

    if (laneIdx < params.mTopK) {
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(warpTopKScore[laneIdx]),
                                          static_cast<int16_t>(warpTopKExpertIdx[laneIdx])};
      params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packedScore;
      // Routing replay: record selected expert IDs. Layout: [num_tokens, topK]
      // -- same indexing as mPtrTopKPacked.
      if (params.mPtrRoutingReplayOut != nullptr) {
        params.mPtrRoutingReplayOut[tokenIdx * params.mTopK + laneIdx] =
            static_cast<int16_t>(warpTopKExpertIdx[laneIdx]);
      }
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

// Returns whether a compiled tier covered the runtime (numExperts, topK) and the
// kernel was actually launched (see launchBlockKernel).
bool launchHistogramScoresKernel(Data const& data, uint32_t maxNumBlocks, uint32_t numThreadsHist,
                                 void* stream) {
  bool launched = false;
  dispatchRoutingPolicy(data, [&](auto preProc, auto postProc, char const* policyName) {
    using PreProc = decltype(preProc);
    using PostProc = decltype(postProc);
    using Pairs = typename HistogramScoresPolicyTraits<PreProc, PostProc>::Pairs;
    bool dispatched =
        dispatchTierPairs(static_cast<Pairs*>(nullptr), data, [&](auto eTag, auto kTag) {
          using ExpertSelect = TopKExpertSelect<PreProc, PostProc>;
          using LaunchConfig = HistogramScoresKernelConfig<ExpertSelect, decltype(eTag)::value,
                                                           decltype(kTag)::value>;
          int const effectiveThreads =
              LaunchConfig::blockDim(data, static_cast<int>(numThreadsHist));
          int const effectiveBlocks =
              LaunchConfig::gridDim(data, static_cast<int>(maxNumBlocks), effectiveThreads);
          LAUNCH_ROUTING_WITH_POLICIES(data, false, routingIndicesHistogramScoresKernel,
                                       effectiveBlocks, effectiveThreads,
                                       /*smemSize=*/0, stream, PreProc, PostProc,
                                       decltype(eTag)::value, decltype(kTag)::value);
        });
    launched = dispatched;
    if (!dispatched) {
      FLASHINFER_WARN(
          "No tier covers numExperts=%d topK=%d for policy %s in "
          "launchHistogramScoresKernel",
          data.mNumExperts, data.mTopK, policyName);
    }
  });
  return launched;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 3b. BlockScores kernel — one block per token, block-parallel preprocess + warp-0 sort-based topK.
//
// Motivation: for small numTokens with high topK (e.g. Nemotron Super V3:
// BS<=256, E=512, K=22), the per-warp-per-token `routingIndicesHistogramScoresKernel`
// suffers from (a) high register pressure — *every* warp in a block carries
// the K-sized topK arrays (~99 regs/thread → 25% occupancy), and (b) low
// arithmetic intensity per warp during preprocess (each lane redundantly
// processes VecSize=MaxNumExperts/32 experts into registers).
//
// This kernel instead mirrors the design used by the no-groups path of
// TRT-LLM's `deepseek_v3_topk_kernel`:
//   Phase 1: parallelise the preprocess across all threads via a grid-stride
//            loop (PreprocessPolicy::applyToSmem), writing per-expert topK
//            key (and optional aux data) into smem.
//   Phase 2: warp 0 alone does the topK via a sort-based reduceTopK with
//            N = ceil(MaxNumExperts / WarpSize) elements per lane.
//   Phase 3: warp 0 applies the postprocess (PostprocessPolicy::applyWithAux),
//            in-place modifying topScores.
// Only warp 0 carries the K-sized register arrays, so per-block register
// pressure stays low (~64 regs/thread) and occupancy rises to 2 blocks/SM.
//
// The kernel is generic in (PreprocessPolicy, PostprocessPolicy).  The
// auxiliary smem array is only allocated when the policy pair requires it
// (today: only ScaledSumNormalizePostprocess — see PolicyPairNeedsAux trait
// in RoutingCustomPolicy.cuh).  Otherwise, the "aux" pointer aliases the
// "biased" pointer and no extra smem is reserved.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

// Block dimension for routingIndicesBlockScoresKernel.  Chosen so that:
//   - __launch_bounds__ can reserve the per-thread register budget
//     (each SM then fits 2 blocks at 64 regs/thread, the measured sweet spot).
//   - SoftmaxPreprocess::applyToSmem<kBlockDim> can size its cub::BlockReduce
//     temp storage at compile time.
// The value is kept as a compile-time constant (rather than a runtime
// parameter) so the kernel and its launcher share a single source of truth;
// the internal Softmax path would not function correctly if the actual
// blockDim.x disagreed with this value.
constexpr int kBlockScoresKernelBlockDim = 256;

template <typename KernelParams>
__global__ void __launch_bounds__(kBlockScoresKernelBlockDim)
    routingIndicesBlockScoresKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using ExpertSelect = typename KernelParams::ExpertSelectPolicy;
  using PreProc = typename ExpertSelect::PreprocessPolicy;
  using PostProc = typename ExpertSelect::PostprocessPolicy;
  using BaseType = typename ExpertSelect::template BaseType<InputT>;

  static constexpr int MaxNumExperts = KernelParams::MaxNumExperts;
  static constexpr int MaxNumTopExperts = KernelParams::MaxNumTopExperts;
  // One chunk per warp lane — each lane of warp 0 holds NumChunks experts.
  // For E=512 this is 16; for E=1024 it's 32; for E=2048 it's 64 — which hits
  // Sort<N>'s 64-element cap (the static_assert inside topk::reduceTopK).
  static constexpr int NumChunks = (MaxNumExperts + WarpSize - 1) / WarpSize;

  // Compile-time opt-out: the macro dispatch instantiates this kernel across
  // every (PreProc, PostProc, tier) combination, but we only emit a real
  // kernel body for policy pairs that implement the block-per-token interface
  // (PolicyPairSupportsBlockPerToken<PreProc, PostProc>::value).  For the
  // other instantiations we emit an empty kernel body.  These instantiations
  // are never reached at runtime because `run()` gates the host-side launch
  // on the same trait check, but compiling them as no-ops keeps this kernel
  // generic across the whole policy matrix.
  //
  // The expert count is bounded by Sort<N>'s N ≤ 64 cap, i.e. MaxNumExperts
  // ≤ 64 * WarpSize = 2048.  This matches `MaxSupportedExperts` in
  // RoutingCustomPolicy.cuh, so every tier declared in `PolicyTraits` fits
  // and no extra guard is needed here.
  static constexpr bool kSupported = PolicyPairSupportsBlockPerToken<PreProc, PostProc>::value;
  static_assert(NumChunks <= 64,
                "routingIndicesBlockScoresKernel: MaxNumExperts must be <= 64 * WarpSize = 2048 "
                "(Sort<N>'s upper bound inside topk::reduceTopK)");
  if constexpr (!kSupported) {
    return;
  } else {
    // Allocate smemAux only when the postprocess actually reads it.  For every
    // other (pre, post) combination we save MaxNumExperts × 4 B of smem by
    // letting `auxPtr` alias `smemBiased`.
    static constexpr bool kNeedsAux = PolicyPairNeedsAux<PreProc, PostProc>::value;
    static constexpr int kAuxSize = kNeedsAux ? MaxNumExperts : 1;

    static constexpr float invalidScoreFloat = -INFINITY;

    // Per-expert smem arrays:
    //   smemBiased[e] = topK selection key for expert e
    //   smemAux[e]    = auxiliary data for expert e (only used / written when
    //                   PolicyPairNeedsAux<PreProc, PostProc>::value is true)
    __shared__ BaseType __attribute((aligned(128))) smemBiased[MaxNumExperts];
    __shared__ BaseType __attribute((aligned(128))) smemAuxStorage[kAuxSize];
    BaseType* auxPtr = kNeedsAux ? smemAuxStorage : smemBiased;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    int32_t const laneIdx = cutlass::arch::LaneId();
    // blockDim.x is always kBlockScoresKernelBlockDim (a multiple of WarpSize),
    // so `threadIdx.x / WarpSize` is already uniform within a warp — no
    // `__shfl_sync` broadcast needed.
    int32_t const warpIdx = threadIdx.x / WarpSize;
    int32_t const tokenIdx = blockIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (params.mUsePdl) {
      cudaGridDependencySynchronize();
    }
#endif

    // Reset `mPtrExpertCounts` (histogram needed by the downstream coop /
    // multi-kernel permutation paths).  Block 0 does this across its threads;
    // other blocks skip to avoid redundant writes.  Size is 2*numExperts (the
    // array is used both for the final histogram and the tile-offset histogram).
    if (blockIdx.x == 0 && params.mPtrExpertCounts != nullptr) {
      int32_t const expertCountsNum = 2 * params.mNumExperts;
      for (int i = threadIdx.x; i < expertCountsNum; i += blockDim.x) {
        params.mPtrExpertCounts[i] = 0;
      }
    }

    // Phase 1: block-parallel preprocess.  Dispatches on PreProc so the kernel
    // works for all registered preprocess policies (single-pass NoOp / Sigmoid /
    // SigmoidBias, two-pass Softmax).
    int64_t const scoreBase = int64_t{tokenIdx} * int64_t{params.mNumExperts};
    if constexpr (std::is_same_v<PreProc, SoftmaxPreprocess>) {
      // Softmax needs block-level reductions; pass the block size as a template
      // parameter so it can construct cub::BlockReduce with the right size.
      // We rely on kBlockScoresKernelBlockDim matching the actual launch
      // blockDim.x — enforced by the launcher below.
      PreProc::template applyToSmem<kBlockScoresKernelBlockDim>(
          block, params.mPtrScores + scoreBase, params.mNumExperts, smemBiased, auxPtr,
          params.mExpertSelectParams.mPreprocessParams);
    } else {
      PreProc::applyToSmem(block, params.mPtrScores + scoreBase, params.mNumExperts, smemBiased,
                           auxPtr, params.mExpertSelectParams.mPreprocessParams);
    }

    __syncthreads();

    // Phase 2: warp-0 sort-based topK over NumChunks elements per lane.
    // Warps 1..N-1 are done and can exit; only warp 0 carries the K-sized
    // register arrays from here on.
    if (warpIdx != 0) {
      return;
    }

    BaseType localScores[NumChunks];
    int32_t localIdx[NumChunks];
#pragma unroll
    for (int ii = 0; ii < NumChunks; ++ii) {
      int const eIdx = ii * WarpSize + laneIdx;
      localIdx[ii] = eIdx;
      localScores[ii] = eIdx < params.mNumExperts ? smemBiased[eIdx] : BaseType{invalidScoreFloat};
    }

    BaseType topScores[MaxNumTopExperts];
    int32_t topExperts[MaxNumTopExperts];
    topk::reduceTopK(warp, topScores, topExperts, localScores, localIdx,
                     /*minValue=*/BaseType{invalidScoreFloat}, params.mTopK);

    // Phase 3: postprocess.  Reads the per-expert aux data (if the policy needs
    // it) and in-place modifies topScores.
    PostProc::applyWithAux(warp, topScores, topExperts, laneIdx, params.mTopK, auxPtr,
                           params.mExpertSelectParams.mPostprocessParams);

    // Phase 4: write packed (score, expertIdx) for the downstream permutation stage.
    if (laneIdx < params.mTopK) {
      int32_t const expertIdx = topExperts[laneIdx];
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(topScores[laneIdx]),
                                          static_cast<int16_t>(expertIdx)};
      params.mPtrTopKPacked[int64_t{tokenIdx} * int64_t{params.mTopK} + laneIdx] = packedScore;
      // Routing replay: record selected expert IDs. Layout: [num_tokens, topK]
      // -- same indexing as mPtrTopKPacked.
      if (params.mPtrRoutingReplayOut != nullptr) {
        params.mPtrRoutingReplayOut[int64_t{tokenIdx} * int64_t{params.mTopK} + laneIdx] =
            static_cast<int16_t>(expertIdx);
      }
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (params.mUsePdl) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
  }  // end `if constexpr (kSupported)` else branch
}

// Returns whether a compiled tier covered the runtime (numExperts, topK) and the
// kernel was actually launched (see launchBlockKernel).
bool launchBlockScoresKernel(Data const& data, void* stream) {
  // Custom dispatch that does NOT clamp blockDim to the dispatched tier's
  // expert count (unlike `LAUNCH_ROUTING_CUSTOM`).  The kernel's layout —
  // kBlockScoresKernelBlockDim threads with a grid-stride loop over experts —
  // is decoupled from MaxNumExperts; oversizing blockDim would waste
  // registers and shrink occupancy.  The blockDim is intentionally fixed to
  // the kernel-side constant so host and device agree (SoftmaxPreprocess
  // sizes its cub::BlockReduce with the same value at compile time).
  bool launched = false;
  dispatchRoutingPolicy(data, [&](auto preProc_, auto postProc_, char const* policyName_) {
    using PreProc_ = decltype(preProc_);
    using PostProc_ = decltype(postProc_);
    using Pairs_ = typename PolicyTraits<PreProc_, PostProc_>::Pairs;
    bool dispatched_ =
        dispatchTierPairs(static_cast<Pairs_*>(nullptr), data, [&](auto eTag_, auto kTag_) {
          LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/false, routingIndicesBlockScoresKernel,
                                       /*gridDim=*/data.mNumTokens,
                                       /*blockDim=*/kBlockScoresKernelBlockDim,
                                       /*smemSize=*/0, stream, PreProc_, PostProc_,
                                       decltype(eTag_)::value, decltype(kTag_)::value);
        });
    launched = dispatched_;
    if (!dispatched_) {
      FLASHINFER_WARN(
          "No compiled tier covers numExperts=%d topK=%d for policy %s in "
          "launchBlockScoresKernel.",
          data.mNumExperts, data.mTopK, policyName_);
    }
  });
  return launched;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// 4. Coop kernel — cooperative histogram + offsets via grid-sync.
//
// The coop kernel only performs the post-topK permutation pipeline (histogram, prefix-scan,
// index writes). It does NOT compute topK — it reads pre-computed results from mPtrTopKPacked
// or mPtrTopKIds. Therefore, only the NumExperts template tier matters (it sizes shared memory
// arrays and determines the thread count). The NumTopExperts tier is fixed at NumTop8Experts
// because the kernel uses a hardcoded MaxExpandedIdxPerThread=64 and processes all expanded
// indices (mNumTokens * mTopK) via a grid-stride loop with runtime bounds checking, regardless
// of the compile-time MaxNumTopExperts value.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

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
  } else if (data.mNumExperts <= NumExperts1024Experts) {
    LAUNCH_ROUTING_WITH_POLICIES(data, /*coopLaunch=*/true, routingIndicesCoopKernel, numBlocksCoop,
                                 numThreadsHist, /*smemSize=*/0, stream, NoOpPreprocess,
                                 NoOpPostprocess, NumExperts1024Experts, NumTop8Experts);
  } else {
    TVM_FFI_LOG_AND_THROW(NotImplementedError)
        << "Coop kernel does not support numExperts > " << NumExperts1024Experts << ", got "
        << data.mNumExperts;
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
    runPostTopKPipeline(data, stream);
    return;
  }

  // After this point, input is mPtrScores (raw logits that need topK computation).
  TVM_FFI_ICHECK(data.mPtrScores != nullptr) << "Expected mPtrScores to be non-null at this "
                                                "point.";
  TVM_FFI_ICHECK(data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr &&
                 data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr)
      << "Custom routing kernel expects permuted idx and grouped Gemm launch config buffers";
  TVM_FFI_ICHECK_LE(data.mTopK, static_cast<int32_t>(MaxSupportedTopExperts))
      << "Routing kernel expects topK experts <= " << MaxSupportedTopExperts << ", got "
      << data.mTopK;
  TVM_FFI_ICHECK_LE(data.mNumExperts, static_cast<int32_t>(MaxSupportedExperts))
      << "Routing kernel expects #experts " << data.mNumExperts << " to be no more than "
      << MaxSupportedExperts << ".";

  static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;
  bool const useStaticBlock = data.mNumTokens <= BlockKernelMaxNumTokens;
  // Gate on the dispatched tier size, not the raw expert count.
  // Example: a model with 512 experts and topK=22 skips Tier<512,8> (topK too
  // large) and falls through to Tier<1024,32>.  queryDispatchedMaxExperts()
  // returns 1024 while mNumExperts is only 512.  The dynblock kernel sizes
  // shared memory proportional to maxExperts, so using the raw count (512 <=
  // 512, passes) would let it enter a 1024-expert specialization that exceeds
  // the smem budget.
  int32_t const dispatchedMaxExperts = queryDispatchedMaxExperts(data);
  bool const useDynBlock = !useStaticBlock && data.mNumTokens <= DynBlockKernelMaxNumTokens &&
                           dispatchedMaxExperts <= DynBlockKernelMaxNumExperts;
  bool const useSingleBlock = useStaticBlock || useDynBlock;
  bool const useSingleCluster =
      (smMajor >= 9) && (data.mNumTokens <= MaxNumTokensSingleClusterScores);

  // Split-topK path: block-per-token scores kernel + permutation-only cluster
  // kernel, the two overlapping via PDL.  This replaces the fused single-
  // cluster kernel for configs where it underperforms:
  //
  //   - Register pressure / spills: the fused kernel has every cluster warp
  //     carry K-sized per-thread arrays across the cluster barrier.  For
  //     Nemotron Super V3 (K=22) on B200 this yields 1088 spill slots and
  //     caps occupancy to 1 block/SM.
  //   - Idle-warp barrier stalls: at BS < cluster capacity the cluster
  //     still launches 256 warps, but only ~BS of them carry a token; the
  //     rest wait at __cluster_barrier_wait, dominating the profile (~75%
  //     of stalls at BS=64 on Nemotron).
  //   - No PDL overlap: everything happens in one kernel.
  //
  // Split-path structure:
  //   * routingIndicesBlockScoresKernel writes mPtrTopKPacked
  //     (1 block / token, 256 threads, only warp 0 carries K-sized regs).
  //   * runPostTopKPipeline selects the cluster kernel with
  //     LoadExpertIdxFromGlobal=true → permutation only.
  // The two kernels overlap via cudaTriggerProgrammaticLaunchCompletion()
  // and cudaGridDependencySynchronize().
  //
  // Dispatch rule (derived from a policy × (E, K) × BS sweep, see
  // `bench_routing_sweep.py` + `bench_routing_analyze.py`):
  //
  //   useSplit = useSingleCluster && !useSingleBlock && numExperts >= 160
  //
  // Why this rule (measured on B200 across {Default, Renormalize,
  // DeepSeekV3_nGrp1, MiniMax2} policies, BS ∈ [17, 256]):
  //   - At `numExperts >= 160` split wins on average for every policy:
  //     1.17× (Renormalize 160/8) up to 3.62× (DeepSeekV3 1024/32).
  //   - At `numExperts = 128`, split's two-kernel overhead (launch + PDL
  //     handoff) approximately cancels the single-cluster savings.  Default
  //     128/8 is ~1.0×; DeepSeekV3 / MiniMax2 128/8 show ~1.11× but
  //     Renormalize 128/8 is ~1.06× — not worth the complexity.
  //   - At numTokens < 17 the block/dynblock kernels already handle small
  //     BS optimally; this branch only fires when the cluster path would.
  //
  // Env-var override for benchmarking, applies to *both* the single-cluster
  // split path and the large-BS topK kernel choice:
  //   `FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN`
  //     unset / "auto" / "" → use the heuristic above (default)
  //     "1" / "on"          → force the block-per-token topK kernel wherever
  //                           applicable (ignores the numExperts guard, but
  //                           still requires the policy to opt in via
  //                           PolicyPairSupportsBlockPerToken)
  //     "0" / "off"         → force the fused single-cluster kernel for small
  //                           BS, and HistogramScoresKernel (warp-per-token)
  //                           for large BS
  //   Read once per process via a static local; invalid values silently fall
  //   back to "auto".
  enum class ForceMode { kAuto, kOn, kOff };
  static ForceMode const forceMode = [] {
    char const* raw = std::getenv("FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN");
    if (raw == nullptr) return ForceMode::kAuto;
    std::string v = raw;
    if (v == "1" || v == "on" || v == "ON") return ForceMode::kOn;
    if (v == "0" || v == "off" || v == "OFF") return ForceMode::kOff;
    return ForceMode::kAuto;
  }();

  // Does the currently-active policy pair implement the block-per-token
  // interface?  Queried via PolicyPairSupportsBlockPerToken — policies that
  // don't opt in (no applyToSmem / applyWithAux specialisation) will force
  // this branch to false and fall back to the fused single-cluster kernel.
  bool const policySupportsBlockPerToken = queryPolicySupportsBlockPerToken(data);
  if (forceMode == ForceMode::kOn && !policySupportsBlockPerToken) {
    FLASHINFER_WARN(
        "FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN is set but the active routing policy does not "
        "support block-per-token; the request is ignored.");
  }

  bool useSplitTopKPath = useSingleCluster && !useSingleBlock && policySupportsBlockPerToken &&
                          (data.mNumExperts >= NumExperts160Experts);
  if (forceMode == ForceMode::kOn && useSingleCluster && !useSingleBlock &&
      policySupportsBlockPerToken) {
    useSplitTopKPath = true;
  } else if (forceMode == ForceMode::kOff) {
    useSplitTopKPath = false;
  }
  if (!useSingleCluster && !useSingleBlock) {
    TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr)
        << "When #tokens is large, `mPtrTopKPacked` is a required input.";
    TVM_FFI_ICHECK(data.mPtrExpertCounts != nullptr)
        << "When #tokens is large, `mPtrExpertCounts` is a required input.";
  } else if (useSplitTopKPath) {
    // BlockScoresKernel unconditionally writes topK results to mPtrTopKPacked
    // (the downstream runPostTopKPipeline then reads it as pre-computed topK).
    // Unlike the large-BS path above, this branch is gated on useSingleCluster,
    // so we need a separate check here.  mPtrExpertCounts has its own
    // `nullptr` guard inside the kernel (histogram reset), so it remains
    // optional and doesn't need a check.
    TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr)
        << "The block-per-token split path requires `mPtrTopKPacked` to be non-null "
           "(BlockScoresKernel writes the topK result into it).";
  }

  uint32_t const numThreadsHist =
      std::min(1024u, static_cast<uint32_t>(getMaxNumExperts(data.mNumExperts)));

  // We need a mutable copy since `data` is const.
  Data mutableData = data;

  if (useSplitTopKPath) {
    // Step 1: scores → mPtrTopKPacked via a block-per-token kernel that
    // mirrors TRT-LLM's deepseek_v3_topk_kernel (no-groups path) layout:
    //   - One block per token, kBlockScoresKernelBlockDim threads per block.
    //   - Block-parallel preprocess (via PreprocessPolicy::applyToSmem).
    //   - Warp 0 alone does the sort-based reduceTopK<K, N=ceil(E/32)> plus
    //     the postprocess (via PostprocessPolicy::applyWithAux).
    //   - Only warp 0 holds the K-sized register arrays, so per-block
    //     register pressure stays low and occupancy high.
    //
    // Why 256 threads (kBlockScoresKernelBlockDim): larger blocks (e.g. 512)
    // would reserve the ~99-register budget across more threads and limit
    // occupancy to 1 block/SM.  256 threads gives 2 blocks/SM with the same
    // per-warp workload, doubling arithmetic throughput during the preprocess
    // phase.  For E=512 the grid-stride loop iterates 2× per thread; for
    // E<=256 it iterates once.
    //
    // The kernel is generic in (PreprocessPolicy, PostprocessPolicy) — any
    // registered policy pair in RoutingCustomPolicy.cuh works here.
    bool const launched = launchBlockScoresKernel(mutableData, stream);
    FLASHINFER_CHECK(
        launched, "routingCustom::run: no compiled tier covers numExperts=", data.mNumExperts,
        " topK=", data.mTopK,
        " for the active routing policy (block-per-token split path; see preceding "
        "warning). Add a matching Tier<E, K> to PolicyTraits in RoutingCustomPolicy.cuh.");

    // Step 2: delegate the permutation pipeline to runPostTopKPipeline, which
    //   will pick the cluster kernel (LoadExpertIdxFromGlobal=true branch) for
    //   small numTokens.  The cluster kernel in that branch reads topK from
    //   mPtrTopKPacked instead of doing topK itself, so it does not carry the
    //   K-sized register arrays or suffer from idle-warp barrier waits.
    //
    //   Clear mPtrScores so runPostTopKPipeline takes the pre-computed-topK
    //   path (we've already written mPtrTopKPacked above).
    mutableData.mPtrScores = nullptr;
    runPostTopKPipeline(mutableData, stream);
    return;
  }

  if (useDynBlock) {
    bool const launched = launchDynBlockKernel(mutableData, numThreadsHist, stream);
    FLASHINFER_CHECK(launched,
                     "routingCustom::run: no compiled tier covers numExperts=", data.mNumExperts,
                     " topK=", data.mTopK,
                     " for the active routing policy (dyn-block path; see preceding warning). "
                     "Add a matching Tier<E, K> to PolicyTraits in RoutingCustomPolicy.cuh.");
  } else if (useStaticBlock) {
    bool const launched = launchBlockKernel(mutableData, numThreadsHist, stream);
    FLASHINFER_CHECK(launched,
                     "routingCustom::run: no compiled tier covers numExperts=", data.mNumExperts,
                     " topK=", data.mTopK,
                     " for the active routing policy (static-block path; see preceding warning). "
                     "Add a matching Tier<E, K> to PolicyTraits in RoutingCustomPolicy.cuh.");
  } else if (useSingleCluster) {
    bool const launched = launchClusterKernel(mutableData, stream);
    FLASHINFER_CHECK(
        launched, "routingCustom::run: no compiled tier covers numExperts=", data.mNumExperts,
        " topK=", data.mTopK,
        " for the active routing policy (single-cluster path; see preceding warning). "
        "Add a matching Tier<E, K> to (Cluster)PolicyTraits in RoutingCustomPolicy.cuh.");
  } else {
    uint32_t const maxNumBlocks = 1024;

    // TopK kernel selection for the large-BS path.
    //
    // The default is `routingIndicesHistogramScoresKernel` (warp-per-token,
    // grid-stride over numTokens): well-suited to the large-BS regime where
    // there's naturally one warp's worth of work per token.
    //
    // For policies that support the block-per-token interface and have E >= 256,
    // `routingIndicesBlockScoresKernel` (1 block/token, warp-0 sort-based topK)
    // can be faster because:
    //   - Only warp 0 carries the K-sized topK register arrays (saves ~40
    //     regs/thread for K=22, avoids spills even at large E).
    //   - The bitonic-sort reduceTopK over N=ceil(E/32) elements per lane
    //     beats the K sequential warp reductions a warp-per-token layout
    //     does for high-K configs like Nemotron.
    //
    // The trade-off is one block per token — at BS = 4096 that's 4096 blocks
    // vs ~128 blocks for warp-per-token.  Above BS ≈ 1024 block-per-token
    // starts oversubscribing SMs and loses to the warp-per-token layout.
    //
    // Dispatch rule (derived from the same sweep as the cluster path):
    //
    //   useBlockScores = (E >= NumExperts1024Experts)
    //                 || (E >= NumExperts256Experts && BS <= 1024)
    //
    // The E=1024/K=32 tier is a special corner: block-per-token wins at
    // every BS (1.95×–5.17×) because fused warp-per-token topK explodes on
    // register pressure when K=32.  For E ∈ [256, 576], speedups are
    // 1.01×–1.75× at BS <= 1024 but collapse to 0.77×–1.03× at BS >= 2048,
    // hence the BS cap.
    bool useBlockScoresForTopK =
        policySupportsBlockPerToken &&
        ((data.mNumExperts >= NumExperts1024Experts) ||
         (data.mNumExperts >= NumExperts256Experts && data.mNumTokens <= 1024));
    if (forceMode == ForceMode::kOn && policySupportsBlockPerToken) {
      useBlockScoresForTopK = true;
    } else if (forceMode == ForceMode::kOff) {
      useBlockScoresForTopK = false;
    }
    bool launched = false;
    if (useBlockScoresForTopK) {
      launched = launchBlockScoresKernel(mutableData, stream);
    } else {
      launched = launchHistogramScoresKernel(mutableData, maxNumBlocks, numThreadsHist, stream);
    }
    FLASHINFER_CHECK(
        launched,
        "routingCustom::run: no compiled score-to-topK tier covers numExperts=", data.mNumExperts,
        " topK=", data.mTopK,
        " for the active routing policy (large-batch path; see preceding warning). "
        "Add a matching Tier<E, K> to (Histogram)PolicyTraits in RoutingCustomPolicy.cuh.");

    bool const canUseCoop =
        (smMajor >= 9) && (data.mNumExperts <= 1024) && (data.mPtrPermutedIdxSize != nullptr);
    bool useCoop = false;
    CoopLaunchSMCounts coopLaunchSMCounts{0, 0};
    int numBlocksCoop = 0;

    if (canUseCoop) {
      static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
      coopLaunchSMCounts = getCoopLaunchSMCounts(smCount);
      numBlocksCoop = coopLaunchSMCounts.moeSms;
      int const maxTokensCoop = (numBlocksCoop * numThreadsHist * 64) / data.mTopK;
      useCoop = (data.mNumTokens <= maxTokensCoop);
    }

    if (useCoop) {
      logCoopLaunchSMCounts(coopLaunchSMCounts);
      launchInitExpertCounts(mutableData, numThreadsHist, stream);
      launchCoopKernel(mutableData, numBlocksCoop, numThreadsHist, stream);
    } else {
      uint32_t const expandedIdxSize = data.mNumTokens * data.mTopK;
      uint32_t const histogramEltsPerBlock = 8 * numThreadsHist;
      uint32_t const offsetEltsPerBlock = NumEltsPerOffsetTilePerThread * numThreadsHist;

      int const numBlocksHistogram = std::min(
          (expandedIdxSize + histogramEltsPerBlock - 1) / histogramEltsPerBlock, maxNumBlocks);
      int const numBlocksOffsets =
          std::min((expandedIdxSize + offsetEltsPerBlock - 1) / offsetEltsPerBlock, maxNumBlocks);

      launchHistogramKernel(mutableData, numBlocksHistogram, numThreadsHist, stream);
      launchOffsetsKernel(mutableData, numBlocksOffsets, numThreadsHist, stream);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingCustom
}  // namespace moe::dev::routing
