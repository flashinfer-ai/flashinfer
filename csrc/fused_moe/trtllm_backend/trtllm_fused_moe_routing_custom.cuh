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

// Custom routing implementation included by the per-family translation units.
//
// Kernel inventory:
//   1. routingIndicesBlockKernel      — single-block fused kernel (≤4 tokens)
//   1c. routingIndicesSmallBatchKernel / routingIndicesSmallBatchClusterKernel — single-launch
//       kernels for 256 experts / top-8, ≤64 tokens (SM90+)
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

#if (defined(FLASHINFER_ROUTING_CUSTOM_BLOCK_GROUP) +   \
     defined(FLASHINFER_ROUTING_CUSTOM_CLUSTER_SMALL) + \
     defined(FLASHINFER_ROUTING_CUSTOM_CLUSTER_LARGE) + \
     defined(FLASHINFER_ROUTING_CUSTOM_ENTRY)) != 1
#error "Define exactly one FLASHINFER_ROUTING_CUSTOM_* translation-unit selector"
#endif

bool launchBlockKernel(Data const& data, void* stream);
bool launchDynBlockKernel(Data const& data, uint32_t numThreadsHist, void* stream);
void launchSmallBatchKernel(Data const& data, void* stream);
bool launchClusterKernelBlockDim256(Data const& data, void* stream);
bool launchClusterKernelBlockDim512(Data const& data, void* stream);
bool launchClusterKernelBlockDim1024(Data const& data, void* stream);
bool launchClusterKernel(Data const& data, void* stream);
bool launchHistogramScoresKernel(Data const& data, uint32_t maxNumBlocks, uint32_t numThreadsHist,
                                 void* stream);
bool launchBlockScoresKernel(Data const& data, void* stream);
void launchCoopKernel(Data const& data, int numBlocksCoop, uint32_t numThreadsHist, void* stream);
void launchInitExpertCounts(Data const& data, uint32_t numThreadsHist, void* stream);
void launchHistogramKernel(Data const& data, int numBlocksHistogram, uint32_t numThreadsHist,
                           void* stream);
void launchOffsetsKernel(Data const& data, int numBlocksOffsets, uint32_t numThreadsHist,
                         void* stream);

// Small-batch kernels (section 1c): 256 experts, top-8, up to 64 tokens.
static constexpr int SmallBatchKernelNumExperts = NumExperts256Experts;
static constexpr int SmallBatchKernelTopK = NumTop8Experts;
static constexpr int SmallBatchKernelMaxNumTokens = 64;
static constexpr int SmallBatchKernelMaxThreads = 1024;
// Largest batch for the single-CTA kernel; larger batches use the 2-CTA cluster kernel. Small
// batches keep the policy's own top-k: a faster one lets the routed FC1 GEMM take the SMs ahead
// of a shared expert running on a side stream.
static constexpr int SmallBatchKernelSingleCtaMaxNumTokens = 18;

#if defined(FLASHINFER_ROUTING_CUSTOM_BLOCK_GROUP)

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
    // Downstream kernels wait on this grid before reading any routing output (the batched
    // GEMMs guard every routing-dependent load with a grid dependency), so they can launch
    // and run their prologue while this kernel computes the routing.
    cudaTriggerProgrammaticLaunchCompletion();
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
        initRoutingTilePadding(params.mPtrPermutedIdxToTokenIdx, min(mnLimit1, mnLimit2), mnLimit1);
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
}

// Returns whether a compiled tier covered the runtime (numExperts, topK) and the
// kernel was actually launched. A false return means no routing output was written;
// the caller (run) must not proceed to the downstream pipeline in that case.
bool launchBlockKernel(Data const& data, void* stream) {
  uint32_t const numThreadsBlock =
      std::min(1024u, static_cast<uint32_t>(queryDispatchedMaxExperts(data)));
  LAUNCH_ROUTING_CUSTOM(data, false, routingIndicesBlockKernel, 1, numThreadsBlock,
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
    // Downstream kernels wait on this grid before reading any routing output (the batched
    // GEMMs guard every routing-dependent load with a grid dependency), so they can launch
    // and run their prologue while this kernel computes the routing.
    cudaTriggerProgrammaticLaunchCompletion();
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
          initRoutingTilePadding(params.mPtrPermutedIdxToTokenIdx, min(mnLimit1, mnLimit2),
                                 mnLimit1);
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
// 1c. Small-batch kernels — 256 experts / top-8, up to 64 tokens, one launch. Input is logits
//     with SigmoidBias + ScaledSumNormalize routing (DeepSeek-V3 style, no expert groups) or
//     precomputed top-k. Outputs match the block and split kernels bit for bit; inside an expert,
//     tokens are placed in ascending order.
//
//     Each expert keeps a bitset of the tokens routed to it:
//       A. one warp per token: top-k, weights, and the token's bit in each selected expert's set;
//       B. one thread per expert: token count (popc), CTA count, exclusive scan, CTA maps, padding;
//       C. one thread per (token, k): permuted row = expert's first row + popc(earlier tokens).
//     routingIndicesSmallBatchKernel: one CTA, using the policy's applyForLane top-k.
//     routingIndicesSmallBatchClusterKernel: two CTAs with half of the tokens each, reading the
//     other CTA's sets through DSMEM; its top-k gives each lane 8 contiguous experts.
//
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class SmallBatchInput { Scores, TopKIds, TopKPacked };

template <typename KernelParams, SmallBatchInput Input>
__global__ void __launch_bounds__(SmallBatchKernelMaxThreads, 1)
    routingIndicesSmallBatchKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using ExpertSelect = typename KernelParams::ExpertSelectPolicy;
  using BaseType = typename ExpertSelect::template BaseType<InputT>;
  static constexpr int NumExperts = SmallBatchKernelNumExperts;
  static constexpr int TopK = SmallBatchKernelTopK;
  static constexpr int MaxNumTokens = SmallBatchKernelSingleCtaMaxNumTokens;
  static constexpr int VecSize = NumExperts / WarpSize;
  static constexpr int NumExpertWarps = NumExperts / WarpSize;
  static_assert(KernelParams::MaxNumExperts == NumExperts &&
                KernelParams::MaxNumTopExperts == TopK);
  static_assert(MaxNumTokens <= WarpSize, "one 32-bit token set per expert");
  static_assert(TopK <= WarpSize && NumExperts <= SmallBatchKernelMaxThreads);

  // Tokens that selected each expert: bit t = token t.
  __shared__ uint32_t smemTokenSet[NumExperts];
  // Expert of each (token, k) pair, -1 when the pair is not routed (invalid or duplicate id).
  __shared__ int32_t smemPairExpert[MaxNumTokens * TopK];
  __shared__ int32_t smemExpertRow[NumExperts];
  __shared__ int32_t smemWarpCtas[NumExpertWarps];

  int32_t const threadIdxX = static_cast<int32_t>(threadIdx.x);
  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdxX / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();
  int32_t const numTokens = params.mNumTokens;
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  auto const isLocalExpert = [&](int32_t expert) {
    auto const localExpIdx = expert - params.mLocalExpertsStartIdx;
    return localExpIdx >= 0 &&
           localExpIdx < (params.mNumLocalExperts << params.mLocalExpertsStrideLog2) &&
           (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
  };

  // Before the PDL wait: shared memory only.
  if (threadIdxX < NumExperts) {
    smemTokenSet[threadIdxX] = 0u;
  }
  __syncthreads();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  // A. One warp per token.
  if (warpIdx < numTokens) {
    int32_t const tokenIdx = warpIdx;
    int32_t const pairIdx = tokenIdx * TopK + laneIdx;
    if constexpr (Input == SmallBatchInput::Scores) {
      BaseType laneTopKScore;
      int32_t laneTopKExpertIdx;
      ExpertSelect::template applyForLane<BaseType, InputT, VecSize, TopK>(
          warp, laneTopKScore, laneTopKExpertIdx, laneIdx, NumExperts, TopK,
          params.mPtrScores + tokenIdx * NumExperts, params);
      if (laneIdx < TopK) {
        if (params.mPtrTopKWeights != nullptr) {
          params.mPtrTopKWeights[pairIdx] = OutputT{laneTopKScore};
        }
        if (params.mPtrRoutingReplayOut != nullptr) {
          params.mPtrRoutingReplayOut[pairIdx] = static_cast<int16_t>(laneTopKExpertIdx);
        }
        smemPairExpert[pairIdx] = laneTopKExpertIdx;
        atomicOr(&smemTokenSet[laneTopKExpertIdx], 1u << tokenIdx);
      }
    } else {
      int32_t expertIdx = -1;
      bool valid = false;
      if (laneIdx < TopK) {
        if constexpr (Input == SmallBatchInput::TopKIds) {
          expertIdx = params.mPtrTopKIds[pairIdx];
          valid = expertIdx > -1 && expertIdx < NumExperts;
        } else {
          auto const scoreIdx = params.mPtrTopKPacked[pairIdx];
          expertIdx = static_cast<int32_t>(scoreIdx.idx);
          valid = expertIdx >= 0 && expertIdx < NumExperts;
          if (valid && params.mPtrTopKWeights != nullptr) {
            params.mPtrTopKWeights[pairIdx] = static_cast<OutputT>(scoreIdx.score);
          }
        }
      }
      // An expert listed twice for a token is routed once, from its lowest k; other copies get -1.
      uint32_t const peers = __match_any_sync(0xffffffff, valid ? expertIdx : -1 - laneIdx);
      bool const routed = valid && (__ffs(peers) - 1 == laneIdx);
      if (laneIdx < TopK) {
        smemPairExpert[pairIdx] = routed ? expertIdx : int32_t{-1};
        if (routed) {
          atomicOr(&smemTokenSet[expertIdx], 1u << tokenIdx);
        }
      }
    }
  }
  __syncthreads();

  // B. One thread per expert: token count, CTA count, exclusive scan of the CTA counts (warp
  // scan, then across warps).
  int32_t count = 0;
  int32_t numCta = 0;
  int32_t ctaInclusive = 0;
  bool isLocal = false;
  if (threadIdxX < NumExperts) {
    isLocal = isLocalExpert(threadIdxX);
    count = isLocal ? __popc(smemTokenSet[threadIdxX]) : 0;
    numCta = params.mIsPow2 ? divUpLog2<int32_t>(count, params.mPaddingLog2)
                            : divUpTileN<int32_t>(count, params.mTileTokensDim);
    ctaInclusive = numCta;
#pragma unroll
    for (int offset = 1; offset < WarpSize; offset *= 2) {
      int32_t const other = __shfl_up_sync(0xffffffff, ctaInclusive, offset);
      if (laneIdx >= offset) {
        ctaInclusive += other;
      }
    }
    if (laneIdx == WarpSize - 1) {
      smemWarpCtas[warpIdx] = ctaInclusive;
    }
  }
  __syncthreads();

  if (threadIdxX < NumExperts) {
    int32_t warpPrefix = 0;
    int32_t numNonExitingCtas = 0;
#pragma unroll
    for (int w = 0; w < NumExpertWarps; ++w) {
      int32_t const warpCtas = smemWarpCtas[w];
      warpPrefix += w < warpIdx ? warpCtas : 0;
      numNonExitingCtas += warpCtas;
    }
    int32_t const ctaOffset = warpPrefix + ctaInclusive - numCta;
    smemExpertRow[threadIdxX] = params.mIsPow2
                                    ? mulLog2<int32_t>(ctaOffset, params.mPaddingLog2)
                                    : mulTileN<int32_t>(ctaOffset, params.mTileTokensDim);
    if (isLocal) {
      int32_t const mappedLocalIdx =
          (threadIdxX - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
      for (int cta = 0; cta < numCta; ++cta) {
        params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = mappedLocalIdx;
        int32_t mnLimit1;
        int32_t mnLimit2;
        if (params.mIsPow2) {
          mnLimit1 = mulLog2<int32_t>(ctaOffset + cta + 1, params.mPaddingLog2);
          mnLimit2 = mulLog2<int32_t>(ctaOffset, params.mPaddingLog2) + count;
        } else {
          mnLimit1 = mulTileN<int32_t>(ctaOffset + cta + 1, params.mTileTokensDim);
          mnLimit2 = mulTileN<int32_t>(ctaOffset, params.mTileTokensDim) + count;
        }
        params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] = min(mnLimit1, mnLimit2);
        initRoutingTilePadding(params.mPtrPermutedIdxToTokenIdx, min(mnLimit1, mnLimit2), mnLimit1);
      }
    }
    if (params.mPtrNumTokensPerExpert != nullptr) {
      params.mPtrNumTokensPerExpert[threadIdxX] = count;
    }
    if (threadIdxX == 0) {
      params.mPtrPermutedIdxSize[0] =
          params.mIsPow2 ? mulLog2<int32_t>(numNonExitingCtas, params.mPaddingLog2)
                         : mulTileN<int32_t>(numNonExitingCtas, params.mTileTokensDim);
      params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
    }
  }
  __syncthreads();

  // C. One thread per (token, k): permuted row = expert's first row + its earlier tokens.
  if (int32_t const pairIdx = threadIdxX; pairIdx < numTokens * TopK) {
    int32_t const tokenIdx = pairIdx / TopK;
    int32_t const expertIdx = smemPairExpert[pairIdx];
    int32_t permutedIdx = -1;
    if (expertIdx >= 0 && isLocalExpert(expertIdx)) {
      uint32_t const earlier = smemTokenSet[expertIdx] & ((1u << tokenIdx) - 1u);
      permutedIdx = smemExpertRow[expertIdx] + __popc(earlier);
      if (params.mPtrPermutedIdxToExpandedIdx != nullptr) {
        params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = pairIdx;
      }
      if (params.mPtrPermutedIdxToTokenIdx != nullptr) {
        params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
      }
    }
    if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
      params.mPtrExpandedIdxToPermutedIdx[pairIdx] = permutedIdx;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // Trigger after the last global store, so the next GEMM does not take SMs while routing runs.
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <typename T>
__device__ __forceinline__ void smallBatchCompareSwapDesc(T& a, T& b) {
  T const hi = a > b ? a : b;
  T const lo = a > b ? b : a;
  a = hi;
  b = lo;
}

// Descending sort of 8 unique keys: Batcher's odd-even merge network (19 compare-exchanges).
template <typename T>
__device__ __forceinline__ void smallBatchSort8Desc(T (&k)[8]) {
  smallBatchCompareSwapDesc(k[0], k[1]);
  smallBatchCompareSwapDesc(k[2], k[3]);
  smallBatchCompareSwapDesc(k[4], k[5]);
  smallBatchCompareSwapDesc(k[6], k[7]);
  smallBatchCompareSwapDesc(k[0], k[2]);
  smallBatchCompareSwapDesc(k[1], k[3]);
  smallBatchCompareSwapDesc(k[4], k[6]);
  smallBatchCompareSwapDesc(k[5], k[7]);
  smallBatchCompareSwapDesc(k[1], k[2]);
  smallBatchCompareSwapDesc(k[5], k[6]);
  smallBatchCompareSwapDesc(k[0], k[4]);
  smallBatchCompareSwapDesc(k[1], k[5]);
  smallBatchCompareSwapDesc(k[2], k[6]);
  smallBatchCompareSwapDesc(k[3], k[7]);
  smallBatchCompareSwapDesc(k[2], k[4]);
  smallBatchCompareSwapDesc(k[3], k[5]);
  smallBatchCompareSwapDesc(k[1], k[2]);
  smallBatchCompareSwapDesc(k[3], k[4]);
  smallBatchCompareSwapDesc(k[5], k[6]);
}

// Loads 8 contiguous logits as float. VecLoad: 16-byte aligned, so one or two vector loads.
template <typename InputT, bool VecLoad>
__device__ __forceinline__ void smallBatchLoadLogits(InputT const* src, float (&x)[8]) {
  if constexpr (VecLoad && std::is_same_v<InputT, float>) {
    float4 const a = reinterpret_cast<float4 const*>(src)[0];
    float4 const b = reinterpret_cast<float4 const*>(src)[1];
    x[0] = a.x, x[1] = a.y, x[2] = a.z, x[3] = a.w;
    x[4] = b.x, x[5] = b.y, x[6] = b.z, x[7] = b.w;
  } else if constexpr (VecLoad) {
    static_assert(sizeof(InputT) == 2, "16-bit logits");
    uint4 const v = *reinterpret_cast<uint4 const*>(src);
    InputT const* h = reinterpret_cast<InputT const*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      x[i] = static_cast<float>(h[i]);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      x[i] = static_cast<float>(src[i]);
    }
  }
}

static constexpr int SmallBatchClusterKernelNumCtas = 2;

template <typename KernelParams, SmallBatchInput Input, bool VecLoad>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
__global__ void __launch_bounds__(SmallBatchKernelMaxThreads, 1)
    routingIndicesSmallBatchClusterKernel(KernelParams params) {
  using OutputT = typename KernelParams::OutputT;
  using InputT = typename KernelParams::InputT;
  using ExpertSelect = typename KernelParams::ExpertSelectPolicy;
  using BaseType = typename ExpertSelect::template BaseType<InputT>;
  static constexpr int NumExperts = SmallBatchKernelNumExperts;
  static constexpr int TopK = SmallBatchKernelTopK;
  static constexpr int NumCtas = SmallBatchClusterKernelNumCtas;
  static constexpr int ExpertsPerLane = NumExperts / WarpSize;
  static constexpr int NumExpertWarps = NumExperts / WarpSize;
  static_assert(KernelParams::MaxNumExperts == NumExperts &&
                KernelParams::MaxNumTopExperts == TopK);
  static_assert(SmallBatchKernelMaxNumTokens <= NumCtas * WarpSize,
                "at most 32 tokens per CTA: one 32-bit token set per expert and CTA");
  static_assert(TopK <= WarpSize && NumExperts <= SmallBatchKernelMaxThreads);
  static_assert(ExpertsPerLane == 8, "the per-lane sort network is written for 8 candidates");

  // Selection bias, staged before the PDL wait.
  __shared__ __align__(16) float smemBias[Input == SmallBatchInput::Scores ? NumExperts : 1];
  // This CTA's tokens that selected each expert: bit j = the CTA's j-th token.
  __shared__ uint32_t smemTokenSet[NumExperts];
  // Expert of each of this CTA's (token, k) pairs, -1 when not routed (invalid or duplicate id).
  __shared__ int32_t smemPairExpert[WarpSize * TopK];
  __shared__ int32_t smemExpertRow[NumExperts];
  __shared__ int32_t smemWarpCtas[NumExpertWarps];

  int32_t const threadIdxX = static_cast<int32_t>(threadIdx.x);
  int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdxX / WarpSize, 0);
  int32_t const laneIdx = cutlass::arch::LaneId();
  int32_t const numTokens = params.mNumTokens;
  // CTA 0 takes the first ceil(T / 2) tokens, CTA 1 the rest.
  int32_t const tokensPerCta = (numTokens + NumCtas - 1) / NumCtas;
  int32_t const ctaRank = static_cast<int32_t>(cg::this_cluster().block_rank());
  int32_t const firstToken = ctaRank * tokensPerCta;
  int32_t const ctaTokens = min(tokensPerCta, numTokens - firstToken);
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WarpSize>(block);

  auto const isLocalExpert = [&](int32_t expert) {
    auto const localExpIdx = expert - params.mLocalExpertsStartIdx;
    return localExpIdx >= 0 &&
           localExpIdx < (params.mNumLocalExperts << params.mLocalExpertsStrideLog2) &&
           (localExpIdx & ((1 << params.mLocalExpertsStrideLog2) - 1)) == 0;
  };

  // Before the PDL wait: clear the token sets and stage the bias (a weight, not written by the
  // preceding kernels).
  if (threadIdxX < NumExperts) {
    if constexpr (Input == SmallBatchInput::Scores) {
      auto const& pre = params.mExpertSelectParams.mPreprocessParams;
      smemBias[threadIdxX] = loadScalar(pre.ptrRoutingBias, threadIdxX, pre.dtypeBias);
    }
    smemTokenSet[threadIdxX] = 0u;
  }
  __syncthreads();

  if (params.mUsePdl) {
    cudaGridDependencySynchronize();
  }

  // A. One warp per token of this CTA.
  if (warpIdx < ctaTokens) {
    int32_t const tokenIdx = firstToken + warpIdx;
    int32_t const pairIdx = tokenIdx * TopK + laneIdx;
    int32_t const localPair = warpIdx * TopK + laneIdx;
    if constexpr (Input == SmallBatchInput::Scores) {
      // Same keys and result as TopKExpertSelect<SigmoidBias, ScaledSumNormalize>::applyForLane
      // (sigmoid(logit) + bias, ties to the lower expert), with 8 contiguous experts per lane.
      using RedType = topk::TopKRedType<float>;
      using TypeCmp = typename RedType::TypeCmp;
      using PostProc = typename ExpertSelect::PostprocessPolicy;
      InputT const* row = params.mPtrScores + tokenIdx * NumExperts;
      int32_t const firstExpert = laneIdx * ExpertsPerLane;
      float logit[ExpertsPerLane];
      smallBatchLoadLogits<InputT, VecLoad>(row + firstExpert, logit);
      float4 const bias0 = reinterpret_cast<float4 const*>(smemBias + firstExpert)[0];
      float4 const bias1 = reinterpret_cast<float4 const*>(smemBias + firstExpert)[1];
      float const bias[ExpertsPerLane] = {bias0.x, bias0.y, bias0.z, bias0.w,
                                          bias1.x, bias1.y, bias1.z, bias1.w};
      TypeCmp keys[ExpertsPerLane];
#pragma unroll
      for (int i = 0; i < ExpertsPerLane; ++i) {
        float const s = sigmoid_accurate(logit[i]);
        keys[i] = RedType::makeCmpVal(s + bias[i], firstExpert + i);
      }
      smallBatchSort8Desc(keys);
      // Round kk: the warp max of the lanes' heads; the lane holding it pops its head, and lane kk
      // keeps the key. With 8 candidates per lane, a head is never the 0 filler.
      TypeCmp laneKey = 0;
#pragma unroll
      for (int kk = 0; kk < TopK; ++kk) {
        RedType head;
        head.compVal = keys[0];
        TypeCmp const best = head.reduce(warp);
        laneKey = laneIdx == kk ? best : laneKey;
        bool const popped = keys[0] == best;
#pragma unroll
        for (int i = 0; i < ExpertsPerLane - 1; ++i) {
          keys[i] = popped ? keys[i + 1] : keys[i];
        }
        keys[ExpertsPerLane - 1] = popped ? TypeCmp{0} : keys[ExpertsPerLane - 1];
      }
      BaseType laneTopKScore = 0.f;
      int32_t laneTopKExpertIdx = -1;
      if (laneIdx < TopK) {
        RedType::unpack(laneTopKScore, laneTopKExpertIdx, laneKey);
      }
      PostProc::template applyForLaneWithScores<BaseType, InputT>(
          warp, laneTopKScore, laneTopKExpertIdx, laneIdx, TopK, row,
          params.mExpertSelectParams.mPostprocessParams);
      if (laneIdx < TopK) {
        if (params.mPtrTopKWeights != nullptr) {
          params.mPtrTopKWeights[pairIdx] = OutputT{laneTopKScore};
        }
        if (params.mPtrRoutingReplayOut != nullptr) {
          params.mPtrRoutingReplayOut[pairIdx] = static_cast<int16_t>(laneTopKExpertIdx);
        }
        smemPairExpert[localPair] = laneTopKExpertIdx;
        atomicOr(&smemTokenSet[laneTopKExpertIdx], 1u << warpIdx);
      }
    } else {
      int32_t expertIdx = -1;
      bool valid = false;
      if (laneIdx < TopK) {
        if constexpr (Input == SmallBatchInput::TopKIds) {
          expertIdx = params.mPtrTopKIds[pairIdx];
          valid = expertIdx > -1 && expertIdx < NumExperts;
        } else {
          auto const scoreIdx = params.mPtrTopKPacked[pairIdx];
          expertIdx = static_cast<int32_t>(scoreIdx.idx);
          valid = expertIdx >= 0 && expertIdx < NumExperts;
          if (valid && params.mPtrTopKWeights != nullptr) {
            params.mPtrTopKWeights[pairIdx] = static_cast<OutputT>(scoreIdx.score);
          }
        }
      }
      // Duplicates: routed once, from the lowest k (as in routingIndicesSmallBatchKernel).
      uint32_t const peers = __match_any_sync(0xffffffff, valid ? expertIdx : -1 - laneIdx);
      bool const routed = valid && (__ffs(peers) - 1 == laneIdx);
      if (laneIdx < TopK) {
        smemPairExpert[localPair] = routed ? expertIdx : int32_t{-1};
        if (routed) {
          atomicOr(&smemTokenSet[expertIdx], 1u << warpIdx);
        }
      }
    }
  }
  // Publish this CTA's token sets to the other CTA.
  __cluster_barrier_arrive();
  __cluster_barrier_wait();

  // B. One thread per expert in both CTAs: token count over both CTAs' sets, CTA count, exclusive
  // scan. Only CTA 0 writes the CTA maps. Multiply / divide by the tile size covers power-of-two
  // tiles too, so one code path serves both.
  int32_t const tile = params.mTileTokensDim;
  int32_t count = 0;
  int32_t numCta = 0;
  int32_t ctaInclusive = 0;
  int32_t earlierCount = 0;  // the first CTA's tokens for this thread's expert (second CTA only)
  if (threadIdxX < NumExperts) {
    int32_t const ownCount = __popc(smemTokenSet[threadIdxX]);
    uint32_t const* otherSets =
        cg::this_cluster().map_shared_rank(smemTokenSet, static_cast<unsigned>(ctaRank ^ 1));
    int32_t const otherCount = __popc(otherSets[threadIdxX]);
    earlierCount = ctaRank == 1 ? otherCount : 0;
    count = isLocalExpert(threadIdxX) ? ownCount + otherCount : 0;
    numCta = (count + tile - 1) / tile;
    ctaInclusive = numCta;
#pragma unroll
    for (int offset = 1; offset < WarpSize; offset *= 2) {
      int32_t const other = __shfl_up_sync(0xffffffff, ctaInclusive, offset);
      if (laneIdx >= offset) {
        ctaInclusive += other;
      }
    }
    if (laneIdx == WarpSize - 1) {
      smemWarpCtas[warpIdx] = ctaInclusive;
    }
  }
  // Done reading the other CTA's sets; the wait at exit keeps ours alive until it is done too.
  __cluster_barrier_arrive();
  __syncthreads();

  if (threadIdxX < NumExperts) {
    int32_t warpPrefix = 0;
    int32_t numNonExitingCtas = 0;
#pragma unroll
    for (int w = 0; w < NumExpertWarps; ++w) {
      int32_t const warpCtas = smemWarpCtas[w];
      warpPrefix += w < warpIdx ? warpCtas : 0;
      numNonExitingCtas += warpCtas;
    }
    int32_t const ctaOffset = warpPrefix + ctaInclusive - numCta;
    int32_t const firstRow = ctaOffset * tile;
    // This CTA's tokens of the expert start after the first CTA's.
    smemExpertRow[threadIdxX] = firstRow + earlierCount;
    if (ctaRank == 0) {
      if (numCta > 0) {
        int32_t const mappedLocalIdx =
            (threadIdxX - params.mLocalExpertsStartIdx) >> params.mLocalExpertsStrideLog2;
        int32_t const rowsEnd = firstRow + count;
        // Kept rolled, as the loop below, to keep the kernel body small.
#pragma unroll 1
        for (int cta = 0; cta < numCta; ++cta) {
          params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = mappedLocalIdx;
          params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta] = min(firstRow + (cta + 1) * tile, rowsEnd);
        }
        // Unused rows of the expert's last tile: -1 keeps the GEMM's routed gathers off them.
        if (params.mPtrPermutedIdxToTokenIdx != nullptr) {
#pragma unroll 1
          for (int32_t row = rowsEnd; row < firstRow + numCta * tile; ++row) {
            params.mPtrPermutedIdxToTokenIdx[row] = -1;
          }
        }
      }
      if (params.mPtrNumTokensPerExpert != nullptr) {
        params.mPtrNumTokensPerExpert[threadIdxX] = count;
      }
      if (threadIdxX == 0) {
        params.mPtrPermutedIdxSize[0] = numNonExitingCtas * tile;
        params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
      }
    }
  }
  __syncthreads();

  // C. One thread per (token, k) of this CTA: permuted row = expert's first row for this CTA +
  // this CTA's earlier tokens of the expert.
  if (int32_t const localPair = threadIdxX; localPair < ctaTokens * TopK) {
    int32_t const localToken = localPair / TopK;
    int32_t const tokenIdx = firstToken + localToken;
    int32_t const pairIdx = firstToken * TopK + localPair;
    int32_t const expertIdx = smemPairExpert[localPair];
    int32_t permutedIdx = -1;
    if (expertIdx >= 0 && isLocalExpert(expertIdx)) {
      uint32_t const earlier = smemTokenSet[expertIdx] & ((1u << localToken) - 1u);
      permutedIdx = smemExpertRow[expertIdx] + __popc(earlier);
      if (params.mPtrPermutedIdxToExpandedIdx != nullptr) {
        params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = pairIdx;
      }
      if (params.mPtrPermutedIdxToTokenIdx != nullptr) {
        params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
      }
    }
    if (params.mPtrExpandedIdxToPermutedIdx != nullptr) {
      params.mPtrExpandedIdxToPermutedIdx[pairIdx] = permutedIdx;
    }
  }

  // After the last global store (see routingIndicesSmallBatchKernel).
  if (params.mUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
  // Do not exit while the other CTA may still read this CTA's token sets.
  __cluster_barrier_wait();
}
#else
__global__ void __launch_bounds__(SmallBatchKernelMaxThreads, 1)
    routingIndicesSmallBatchClusterKernel(KernelParams /* params */) {
  assert(false && "routingIndicesSmallBatchClusterKernel is only supported on SM90+ architectures");
}
#endif  // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))

// One CTA up to SmallBatchKernelSingleCtaMaxNumTokens tokens: at least 256 threads (one per
// expert) and one warp per token.
template <typename InputT, SmallBatchInput Input, typename ExpertSelect>
void launchSmallBatchKernelInstance(Data const& data, void* stream) {
  using ParamsT = KernelParams<InputT, __nv_bfloat16, SmallBatchKernelNumExperts,
                               SmallBatchKernelTopK, ExpertSelect>;
  int const numThreads =
      std::max(data.mNumTokens * static_cast<int>(WarpSize), SmallBatchKernelNumExperts);

  cudaLaunchConfig_t config{};
  config.gridDim = 1;
  config.blockDim = numThreads;
  config.dynamicSmemBytes = 0;
  config.stream = (cudaStream_t)stream;
  cudaLaunchAttribute attributes[1] = {};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = int(data.mUsePdl);
  config.attrs = attributes;
  config.numAttrs = 1;

  auto params = ParamsT::setKernelParams(data);
  CHECK_CUDA_ERROR(
      cudaLaunchKernelEx(&config, routingIndicesSmallBatchKernel<ParamsT, Input>, params));
}

// A cluster of two CTAs, each with at least 256 threads and one warp per token of its half.
template <typename InputT, SmallBatchInput Input, typename ExpertSelect, bool VecLoad = false>
void launchSmallBatchClusterKernelInstance(Data const& data, void* stream) {
  using ParamsT = KernelParams<InputT, __nv_bfloat16, SmallBatchKernelNumExperts,
                               SmallBatchKernelTopK, ExpertSelect>;
  int const tokensPerCta =
      (data.mNumTokens + SmallBatchClusterKernelNumCtas - 1) / SmallBatchClusterKernelNumCtas;
  int const numThreads =
      std::max(tokensPerCta * static_cast<int>(WarpSize), SmallBatchKernelNumExperts);

  cudaLaunchConfig_t config{};
  config.gridDim = SmallBatchClusterKernelNumCtas;
  config.blockDim = numThreads;
  config.dynamicSmemBytes = 0;
  config.stream = (cudaStream_t)stream;
  cudaLaunchAttribute attributes[2] = {};
  attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[0].val.programmaticStreamSerializationAllowed = int(data.mUsePdl);
  attributes[1].id = cudaLaunchAttributeClusterDimension;
  attributes[1].val.clusterDim.x = SmallBatchClusterKernelNumCtas;
  attributes[1].val.clusterDim.y = 1;
  attributes[1].val.clusterDim.z = 1;
  config.attrs = attributes;
  config.numAttrs = 2;

  auto params = ParamsT::setKernelParams(data);
  CHECK_CUDA_ERROR(cudaLaunchKernelEx(
      &config, routingIndicesSmallBatchClusterKernel<ParamsT, Input, VecLoad>, params));
}

// Precondition: canUseSmallBatchKernel(data) (entry point below).
void launchSmallBatchKernel(Data const& data, void* stream) {
  using NoOpSelect = TopKExpertSelect<NoOpPreprocess, NoOpPostprocess>;
  using DeepSeekSelect = TopKExpertSelect<SigmoidBiasPreprocess, ScaledSumNormalizePostprocess>;
  bool const fp32Logits = data.mDtypeInput == tg::Dtype::Fp32;
  if (data.mNumTokens <= SmallBatchKernelSingleCtaMaxNumTokens) {
    if (data.mPtrTopKIds != nullptr) {
      // Precomputed ids and weights: the routing policy is not applied.
      launchSmallBatchKernelInstance<__nv_bfloat16, SmallBatchInput::TopKIds, NoOpSelect>(data,
                                                                                          stream);
    } else if (data.mPtrScores == nullptr) {
      launchSmallBatchKernelInstance<__nv_bfloat16, SmallBatchInput::TopKPacked, NoOpSelect>(
          data, stream);
    } else if (fp32Logits) {
      launchSmallBatchKernelInstance<float, SmallBatchInput::Scores, DeepSeekSelect>(data, stream);
    } else {
      launchSmallBatchKernelInstance<__nv_bfloat16, SmallBatchInput::Scores, DeepSeekSelect>(
          data, stream);
    }
    return;
  }
  if (data.mPtrTopKIds != nullptr) {
    launchSmallBatchClusterKernelInstance<__nv_bfloat16, SmallBatchInput::TopKIds, NoOpSelect>(
        data, stream);
    return;
  }
  if (data.mPtrScores == nullptr) {
    launchSmallBatchClusterKernelInstance<__nv_bfloat16, SmallBatchInput::TopKPacked, NoOpSelect>(
        data, stream);
    return;
  }
  // Rows are 256 logits, so every row is 16-byte aligned when the tensor is.
  bool const vecLoad = reinterpret_cast<uintptr_t>(data.mPtrScores) % 16 == 0;
  if (fp32Logits && vecLoad) {
    launchSmallBatchClusterKernelInstance<float, SmallBatchInput::Scores, DeepSeekSelect, true>(
        data, stream);
  } else if (fp32Logits) {
    launchSmallBatchClusterKernelInstance<float, SmallBatchInput::Scores, DeepSeekSelect, false>(
        data, stream);
  } else if (vecLoad) {
    launchSmallBatchClusterKernelInstance<__nv_bfloat16, SmallBatchInput::Scores, DeepSeekSelect,
                                          true>(data, stream);
  } else {
    launchSmallBatchClusterKernelInstance<__nv_bfloat16, SmallBatchInput::Scores, DeepSeekSelect,
                                          false>(data, stream);
  }
}

#endif  // defined(FLASHINFER_ROUTING_CUSTOM_BLOCK_GROUP)

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

#if defined(FLASHINFER_ROUTING_CUSTOM_CLUSTER_SMALL) || \
    defined(FLASHINFER_ROUTING_CUSTOM_CLUSTER_LARGE)

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
  // The ceil-divided scan supports a bounded zero-padded tail. Limit the extra scan work to
  // four warps and only enable it for the optimized high-expert/high-TopK tier family.
  static constexpr bool IsValid =
      First::kExperts <= ClusterBlockDim || First::kExperts % ClusterBlockDim == 0 ||
      (topk::isInHighExpertLaneOwnedTopKRange(First::kExperts, First::kTopK) &&
       ((First::kExperts + ClusterBlockDim - 1) / ClusterBlockDim * ClusterBlockDim -
        First::kExperts) <= 4 * WarpSize);
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

// Returns whether the cluster tier dispatch found a covering tier and launched a kernel.
#if defined(FLASHINFER_ROUTING_CUSTOM_CLUSTER_SMALL)
bool launchClusterKernelBlockDim256(Data const& data, void* stream) {
  return launchClusterKernelForBlockDim<ClusterBlockDim256>(data, stream);
}

bool launchClusterKernelBlockDim512(Data const& data, void* stream) {
  return launchClusterKernelForBlockDim<ClusterBlockDim512>(data, stream);
}
#endif  // defined(FLASHINFER_ROUTING_CUSTOM_CLUSTER_SMALL)

#if defined(FLASHINFER_ROUTING_CUSTOM_CLUSTER_LARGE)
bool launchClusterKernelBlockDim1024(Data const& data, void* stream) {
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
#endif  // defined(FLASHINFER_ROUTING_CUSTOM_CLUSTER_LARGE)

#endif  // FLASHINFER_ROUTING_CUSTOM_CLUSTER_SMALL || FLASHINFER_ROUTING_CUSTOM_CLUSTER_LARGE

#if defined(FLASHINFER_ROUTING_CUSTOM_ENTRY)

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
  using ExpertSelect = typename KernelParams::ExpertSelectPolicy;
  using PostProc = typename ExpertSelect::PostprocessPolicy;
  using BaseType = typename ExpertSelect::template BaseType<InputT>;
  static constexpr int NumThreadsBlock =
      HistogramScoresKernelConfig<typename KernelParams::ExpertSelectPolicy,
                                  KernelParams::MaxNumExperts,
                                  KernelParams::MaxNumTopExperts>::BlockDim;

  // VecSize stays based on MaxNumExperts — each warp still processes all experts for one token.
  static constexpr int VecSize = KernelParams::MaxNumExperts / WarpSize;
  static constexpr bool UseLaneOwnedTopK =
      topk::isInHighExpertLaneOwnedTopKRange(KernelParams::MaxNumExperts,
                                             KernelParams::MaxNumTopExperts) &&
      PostprocessSupportsLaneOwnedTopK<PostProc>::value;

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
  for (int tokenIdx = globalWarpIdx; tokenIdx < params.mNumTokens; tokenIdx += globalWarpStride) {
    auto scoreOffset = tokenIdx * params.mNumExperts;

    BaseType laneTopKScore;
    int32_t laneTopKExpertIdx;
    if constexpr (UseLaneOwnedTopK) {
      ExpertSelect::template applyForLane<BaseType, InputT, VecSize,
                                          KernelParams::MaxNumTopExperts>(
          warp, laneTopKScore, laneTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
          params.mPtrScores + scoreOffset, params);
    } else {
      BaseType warpTopKScore[KernelParams::MaxNumTopExperts];
      int32_t warpTopKExpertIdx[KernelParams::MaxNumTopExperts];
      ExpertSelect::template apply<BaseType, InputT, VecSize, KernelParams::MaxNumTopExperts>(
          warp, warpTopKScore, warpTopKExpertIdx, laneIdx, params.mNumExperts, params.mTopK,
          params.mPtrScores + scoreOffset, params);
      laneTopKScore = laneIdx < params.mTopK ? warpTopKScore[laneIdx] : BaseType{-INFINITY};
      laneTopKExpertIdx = laneIdx < params.mTopK ? warpTopKExpertIdx[laneIdx] : -1;
    }

    if (laneIdx < params.mTopK) {
      PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(laneTopKScore),
                                          static_cast<int16_t>(laneTopKExpertIdx)};
      params.mPtrTopKPacked[tokenIdx * params.mTopK + laneIdx] = packedScore;
      // Routing replay: record selected expert IDs. Layout: [num_tokens, topK]
      // -- same indexing as mPtrTopKPacked.
      if (params.mPtrRoutingReplayOut != nullptr) {
        params.mPtrRoutingReplayOut[tokenIdx * params.mTopK + laneIdx] =
            static_cast<int16_t>(laneTopKExpertIdx);
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
// 3b. BlockScores kernel — one block per token with block-parallel preprocess and topK.
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
//   Phase 2: the generic path uses warp 0 for sort-based topK. Bounded
//            high-expert/high-topK tiers use worker warps over 128-expert
//            partitions, then warp 0 merges their candidates.
//   Phase 3: warp 0 applies the postprocess (PostprocessPolicy::applyWithAux),
//            in-place modifying topScores.
// Only the generic fallback carries final K-sized register arrays. The
// bounded path keeps one result per lane and uses all required expert warps.
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

  // Enable the hierarchical specialization for the shared high-expert/high-TopK range.
  // dispatchTierPairs still determines which concrete tiers are instantiated.
  static constexpr int LaneTopKValuesPerWorkerLane = 4;
  static constexpr int LaneTopKExpertsPerWorkerWarp = LaneTopKValuesPerWorkerLane * WarpSize;
  static constexpr int LaneTopKNumWorkerWarps =
      (MaxNumExperts + LaneTopKExpertsPerWorkerWarp - 1) / LaneTopKExpertsPerWorkerWarp;
  static constexpr int LaneTopKNumIntermediate = LaneTopKNumWorkerWarps * MaxNumTopExperts;
  static constexpr int LaneTopKMergeValuesPerLane =
      (LaneTopKNumIntermediate + WarpSize - 1) / WarpSize;
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

    // Phase 2a: bounded hierarchical lane-owned TopK. Worker warps each
    // reduce a 128-expert partition; warp 0 merges their scalar lane-owned
    // outputs. No K-element output array is dynamically indexed per thread.
    if constexpr (topk::isInHighExpertLaneOwnedTopKRange(MaxNumExperts, MaxNumTopExperts) &&
                  LaneTopKNumWorkerWarps <= kBlockScoresKernelBlockDim / WarpSize &&
                  LaneTopKMergeValuesPerLane <= 64 &&
                  PostprocessSupportsLaneOwnedTopK<PostProc>::value) {
      __shared__ BaseType
          __attribute((aligned(128))) smemIntermediateScores[LaneTopKNumIntermediate];
      __shared__ int32_t
          __attribute((aligned(128))) smemIntermediateExperts[LaneTopKNumIntermediate];

      if (warpIdx < LaneTopKNumWorkerWarps) {
        BaseType localScores[LaneTopKValuesPerWorkerLane];
        int32_t localIdx[LaneTopKValuesPerWorkerLane];
#pragma unroll
        for (int ii = 0; ii < LaneTopKValuesPerWorkerLane; ++ii) {
          int const expertIdx = warpIdx * LaneTopKExpertsPerWorkerWarp + ii * WarpSize + laneIdx;
          localIdx[ii] = expertIdx;
          localScores[ii] =
              expertIdx < params.mNumExperts ? smemBiased[expertIdx] : BaseType{invalidScoreFloat};
        }

        BaseType laneScore;
        int32_t laneExpert;
        topk::reduceTopKForLane<MaxNumTopExperts>(
            warp, laneScore, laneExpert, localScores, localIdx,
            /*minValue=*/BaseType{invalidScoreFloat}, laneIdx);
        if (laneIdx < MaxNumTopExperts) {
          int const intermediateIdx = warpIdx * MaxNumTopExperts + laneIdx;
          bool const valid = laneIdx < params.mTopK;
          smemIntermediateScores[intermediateIdx] = valid ? laneScore : BaseType{invalidScoreFloat};
          smemIntermediateExperts[intermediateIdx] = valid ? laneExpert : MaxNumExperts;
        }
      }
      __syncthreads();

      if (warpIdx != 0) {
        return;
      }

      BaseType intermediateScores[LaneTopKMergeValuesPerLane];
      int32_t intermediateExperts[LaneTopKMergeValuesPerLane];
#pragma unroll
      for (int ii = 0; ii < LaneTopKMergeValuesPerLane; ++ii) {
        int const intermediateIdx = ii * WarpSize + laneIdx;
        bool const valid = intermediateIdx < LaneTopKNumIntermediate;
        intermediateScores[ii] =
            valid ? smemIntermediateScores[intermediateIdx] : BaseType{invalidScoreFloat};
        intermediateExperts[ii] = valid ? smemIntermediateExperts[intermediateIdx] : MaxNumExperts;
      }

      BaseType laneTopKScore;
      int32_t laneTopKExpertIdx;
      topk::reduceTopKForLane<MaxNumTopExperts>(warp, laneTopKScore, laneTopKExpertIdx,
                                                intermediateScores, intermediateExperts,
                                                /*minValue=*/BaseType{invalidScoreFloat}, laneIdx);
      PostProc::applyForLaneWithAux(warp, laneTopKScore, laneTopKExpertIdx, laneIdx, params.mTopK,
                                    auxPtr, params.mExpertSelectParams.mPostprocessParams);

      if (laneIdx < params.mTopK) {
        PackedScoreIdx<OutputT> packedScore{static_cast<OutputT>(laneTopKScore),
                                            static_cast<int16_t>(laneTopKExpertIdx)};
        int64_t const outputIdx = int64_t{tokenIdx} * int64_t{params.mTopK} + laneIdx;
        params.mPtrTopKPacked[outputIdx] = packedScore;
        if (params.mPtrRoutingReplayOut != nullptr) {
          params.mPtrRoutingReplayOut[outputIdx] = static_cast<int16_t>(laneTopKExpertIdx);
        }
      }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
      if (params.mUsePdl) {
        cudaTriggerProgrammaticLaunchCompletion();
      }
#endif
      return;
    }

    // Phase 2b: generic fallback. Warp 0 sorts NumChunks candidates per lane.
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

// FLASHINFER_ROUTING_SMALL_BATCH_KERNEL=0 (or "off") disables the small-batch kernels. Read on
// every eligible call, so a test can compare both paths in one process.
inline bool smallBatchKernelEnabled() {
  char const* raw = std::getenv("FLASHINFER_ROUTING_SMALL_BATCH_KERNEL");
  if (raw == nullptr) return true;
  std::string const v = raw;
  return !(v == "0" || v == "off" || v == "OFF");
}

// Whether the small-batch kernels (section 1c) serve this call: SM90+, 256 experts, top-8,
// 1-64 tokens, BF16 weights, and either precomputed top-k or logits with DeepSeek-V3-style
// SigmoidBias + ScaledSumNormalize routing.
inline bool canUseSmallBatchKernel(Data const& data) {
  static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;
  bool const fromScores = data.mPtrTopKIds == nullptr && data.mPtrScores != nullptr;
  bool const shapeOk = smMajor >= 9 && data.mNumExperts == SmallBatchKernelNumExperts &&
                       data.mTopK == SmallBatchKernelTopK && data.mNumTokens >= 1 &&
                       data.mNumTokens <= SmallBatchKernelMaxNumTokens &&
                       data.mNumFusedSharedExperts == 0 && data.mDtypeOutput == tg::Dtype::Bfloat16;
  bool const outputsOk =
      data.mPtrPermutedIdxSize != nullptr && data.mPtrCtaIdxXyToBatchIdx != nullptr &&
      data.mPtrCtaIdxXyToMnLimit != nullptr && data.mPtrNumNonExitingCtas != nullptr;
  bool const policyOk =
      !fromScores ||
      (data.mPreprocessType == RoutingPreprocessType::SigmoidBias &&
       data.mPostprocessType == RoutingPostprocessType::ScaledSumNormalize &&
       data.mPtrRoutingBias != nullptr &&
       (data.mDtypeInput == tg::Dtype::Fp32 || data.mDtypeInput == tg::Dtype::Bfloat16) &&
       (data.mDtypeBias == tg::Dtype::Fp32 || data.mDtypeBias == tg::Dtype::Bfloat16));
  return shapeOk && outputsOk && policyOk && smallBatchKernelEnabled();
}

void run(Data const& data, void* stream) {
  TVM_FFI_ICHECK(data.mPtrTopKPacked != nullptr || data.mPtrScores != nullptr ||
                 data.mPtrTopKIds != nullptr)
      << "Routing kernel requires at least one input parameter";
  TVM_FFI_ICHECK_LE(data.mTopK, data.mNumExperts)
      << "Routing kernel expects topK (" << data.mTopK << ") to be <= numExperts ("
      << data.mNumExperts << ").";

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
    if (canUseSmallBatchKernel(data)) {
      launchSmallBatchKernel(data, stream);
      return;
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

  // One launch for DeepSeek-V3-style routing at decode batch sizes (section 1c).
  if (canUseSmallBatchKernel(data)) {
    launchSmallBatchKernel(data, stream);
    return;
  }

  static int const smMajor = tensorrt_llm::common::getSMVersion() / 10;
  // Gate on the dispatched tier size, not the raw expert count.
  // Example: a model with 512 experts and topK=22 skips Tier<512,8> (topK too
  // large) and falls through to Tier<1024,32>.  queryDispatchedMaxExperts()
  // returns 1024 while mNumExperts is only 512.  The dynblock kernel sizes
  // shared memory proportional to maxExperts, so using the raw count (512 <=
  // 512, passes) would let it enter a 1024-expert specialization that exceeds
  // the smem budget.
  int32_t const dispatchedMaxExperts = queryDispatchedMaxExperts(data);

  // Does the currently-active policy pair implement the block-per-token interface?
  bool const policySupportsBlockPerToken = queryPolicySupportsBlockPerToken(data);

  // At the static-block boundary, high-E/high-K TopK has enough parallel work to benefit from
  // one score block per token. The downstream post-TopK pipeline still uses its single-block
  // kernel, so this only splits the score selection from permutation. Keep the generic static
  // kernel for smaller batches, expert counts at or below the dyn-block limit, other K ranges,
  // no-op preprocessing, and policy pairs without block-per-token support.
  bool const useSplitStaticBoundary =
      smMajor >= 9 && data.mNumTokens == BlockKernelMaxNumTokens &&
      data.mNumExperts > DynBlockKernelMaxNumExperts &&
      topk::isInHighExpertLaneOwnedTopKRange(data.mNumExperts, data.mTopK) &&
      policySupportsBlockPerToken && data.mPreprocessType != RoutingPreprocessType::None;
  bool const useStaticBlock = data.mNumTokens <= BlockKernelMaxNumTokens && !useSplitStaticBoundary;
  bool const useDynBlock = !useStaticBlock && data.mNumTokens <= DynBlockKernelMaxNumTokens &&
                           dispatchedMaxExperts <= DynBlockKernelMaxNumExperts;
  bool const useSingleBlock = useStaticBlock || useDynBlock;
  bool const useSingleCluster =
      (smMajor >= 9) && (data.mNumTokens <= MaxNumTokensSingleClusterScores);

  // Split-topK path: block-per-token scores kernel + the shared post-TopK pipeline, with PDL
  // between them. For cluster-sized batches this replaces the fused score-and-permutation
  // cluster kernel where it underperforms:
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
  //   * routingIndicesBlockScoresKernel writes mPtrTopKPacked with block-parallel preprocessing
  //     and either generic warp-0 TopK or the bounded worker-warp specialization.
  //   * runPostTopKPipeline consumes mPtrTopKPacked and selects the existing block, cluster,
  //     cooperative, or multi-kernel permutation path for the batch size.
  // The two kernels overlap via cudaTriggerProgrammaticLaunchCompletion()
  // and cudaGridDependencySynchronize().
  //
  // Dispatch rule (derived from a routing-only sweep of every routingCustom policy over
  // E in {128..1024}, K in {4..22}, BS in [1, 256] on SM100, SM103 and SM107):
  //
  //   Few experts (E < 160): split when K >= 8 and BS * K >= 1536, where the fused kernel's
  //   per-thread K-sized state dominates (up to 2.4x for E=128, K=22).
  //   Sigmoid+bias policies (DeepSeekV3 nGroup=1, MiniMax2) with E >= 160: always split.
  //   Other policies with E >= 160: split when BS > 64, or when E >= 512, 10 <= K <= 16 and
  //   BS >= 32. Below that the fused single-cluster kernel is faster than the scores kernel
  //   plus the PDL handoff.
  //   plus the bounded high-E/high-K static-block boundary selected above.
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

  if (forceMode == ForceMode::kOn && !policySupportsBlockPerToken) {
    FLASHINFER_WARN(
        "FLASHINFER_ROUTING_FORCE_BLOCK_PER_TOKEN is set but the active routing policy does not "
        "support block-per-token; the request is ignored.");
  }

  bool const splitBeatsFusedCluster = [&] {
    if (data.mNumExperts < NumExperts160Experts) {
      return data.mTopK >= 8 && data.mNumTokens * data.mTopK >= 1536;
    }
    if (data.mPreprocessType == RoutingPreprocessType::SigmoidBias) {
      return true;
    }
    return data.mNumTokens > 64 || (data.mNumExperts >= 512 && data.mTopK >= 10 &&
                                    data.mTopK <= 16 && data.mNumTokens >= 32);
  }();
  bool useSplitTopKPath =
      useSingleCluster && !useSingleBlock && policySupportsBlockPerToken && splitBeatsFusedCluster;
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
    //   - Generic tiers use warp 0 for sort-based TopK and postprocessing.
    //   - Bounded high-expert/high-TopK tiers partition candidates across worker warps,
    //     then merge to one final result per lane.
    //
    // The fixed 256-thread block provides eight worker warps while preserving occupancy for
    // the generic path; each preprocess pass covers experts with a grid-stride loop.
    //
    // The kernel remains generic across policy pairs that opt into the block-per-token interface.
    bool const launched = launchBlockScoresKernel(mutableData, stream);
    FLASHINFER_CHECK(
        launched, "routingCustom::run: no compiled tier covers numExperts=", data.mNumExperts,
        " topK=", data.mTopK,
        " for the active routing policy (block-per-token split path; see preceding "
        "warning). Add a matching Tier<E, K> to PolicyTraits in RoutingCustomPolicy.cuh.");

    // Step 2: delegate permutation to runPostTopKPipeline. It consumes mPtrTopKPacked without
    // recomputing TopK, then selects the single-block path at the static boundary or the cluster
    // path for larger small batches. This keeps K-sized score state out of the permutation kernel.
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
    bool const launched = launchBlockKernel(mutableData, stream);
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
    // routingIndicesBlockScoresKernel (1 block/token) can be faster because it parallelizes
    // preprocessing across a block. The generic path confines K-sized output arrays to warp 0;
    // bounded high-expert/high-TopK tiers use worker warps and lane-owned final results.
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
      // Both score-to-TopK kernels above zero the full 2*numExperts histogram.
      // Stream ordering (or the PDL dependency) makes that state visible to
      // the cooperative permutation kernel, so a separate init launch would
      // only repeat the same writes.
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

#endif  // defined(FLASHINFER_ROUTING_CUSTOM_ENTRY)

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace routingCustom
}  // namespace moe::dev::routing
