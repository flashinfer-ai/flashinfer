/***************************************************************************************************
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are not permit-
 * ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace flashinfer {
namespace prims_ts {

// Packed device ABI consumed only by the balanced PrimsTS task queues and reducer.
struct alignas(16) BalancedWorkDescriptor {
  int32_t reqIdx;
  int32_t startBlock;
  int32_t endBlock;
  int32_t splitInfo;
};
static_assert(sizeof(BalancedWorkDescriptor) == 16);

static constexpr int32_t kMaxSplitGlobalIdx = 0xFFF;
static constexpr int32_t kMaxNumPieces = 0x7F;
static constexpr int32_t kMaxSplitBeginIdx = 0xFFF;

__host__ __device__ inline int32_t packSplitInfo(int32_t isSplit, int32_t splitGlobalIdx,
                                                 int32_t numPieces, int32_t splitBeginIdx) {
  return (isSplit & 1) | ((splitGlobalIdx & kMaxSplitGlobalIdx) << 1) |
         ((numPieces & kMaxNumPieces) << 13) | ((splitBeginIdx & kMaxSplitBeginIdx) << 20);
}

struct alignas(16) BalancedCombineDescriptor {
  int32_t reqIdx;
  int32_t numPieces;
  int32_t splitBeginIdx;
  int32_t reserved;
};
static_assert(sizeof(BalancedCombineDescriptor) == 16);

struct BalancedSchedParams {
  int32_t batchSize;
  int32_t blockSizeN;
  int32_t numSmParts;
  int32_t const* seqLensKvPtr;
  bool seqLensOnHost = false;
  int32_t workDescriptorCapacity = 0;
  int32_t combineDescriptorCapacity = 0;
  BalancedWorkDescriptor* workDescriptorPtr;
  int32_t* workDescriptorOffsetsPtr;
  BalancedCombineDescriptor* combineDescriptorPtr;
  int32_t* numCombineDescriptorsDevicePtr;
  cudaStream_t stream;
  int32_t costPerBlock = 0;
  int32_t fixedPieceCost = 0;
  int32_t splitFixedCost = 0;
  int32_t splitPieceCost = 0;
  int32_t combineFixedCost = 0;
  int32_t combinePieceCost = 0;
  // Positive only for offline calibration. Production planning leaves this
  // zero and selects the target from the cost model.
  int32_t forcedTargetPieceTiles = 0;
  int32_t* planMetadataHostPtr = nullptr;
};

void runBalancedSchedHost(BalancedSchedParams const& params, int32_t* numCombineDescDev,
                          int32_t maxTotalSplits);

}  // namespace prims_ts
}  // namespace flashinfer
