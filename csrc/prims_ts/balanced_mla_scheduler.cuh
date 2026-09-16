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

#include <cstddef>
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

// Device scheduler metadata layout. Callers get asynchronous status and the selected calibrated
// cost-model bucket in addition to descriptor counts and the selected target.
enum class BalancedSchedMetadata : int32_t {
  kDescriptorCount = 0,
  kTargetPieceTiles = 1,
  kCombineDescriptorCount = 2,
  kStatus = 3,
  kCostBucket = 4,
  kCount = 5,
};

enum class BalancedSchedStatus : int32_t {
  kSuccess = 0,
  kInvalidSequenceLength = 1,
  kTargetOverflow = 2,
  kWorkDescriptorOverflow = 3,
  kPartialOverflow = 4,
  kCombineDescriptorOverflow = 5,
  kSplitInfoOverflow = 6,
};

// Seven rows of six int32 coefficients. Rows correspond to the replay-time bucket classification
// used by balanced_scheduler/cost_model.py; an explicit cost model may repeat one row seven times.
static constexpr int32_t kBalancedDeviceCostBucketCount = 7;
static constexpr int32_t kBalancedDeviceCostCoefficientCount = 6;

// Fixed upper bound used by the Python plan for graph-stable device scratch. The implementation
// deliberately exposes a simple formula rather than its internal layout so that the layout may be
// changed without changing the public scheduler ABI.
inline size_t getBalancedSchedDeviceWorkspaceSize(int32_t batchSize, int32_t numSmParts) {
  return 80ULL * (static_cast<size_t>(batchSize) + static_cast<size_t>(numSmParts)) + 4096ULL;
}

struct BalancedSchedDeviceParams {
  int32_t batchSize;
  int32_t blockSizeN;
  // One producer CTA advances this many base KV tiles per mainloop step. Descriptor coordinates
  // remain expressed in base KV tiles; only schedule partitioning uses this larger work unit.
  int32_t tilesPerWorkUnit;
  int32_t numSmParts;
  int32_t maxSeqLen;
  int32_t const* seqLensKvPtr;
  int32_t workDescriptorCapacity;
  int32_t combineDescriptorCapacity;
  BalancedWorkDescriptor* workDescriptorPtr;
  int32_t* workDescriptorOffsetsPtr;
  BalancedCombineDescriptor* combineDescriptorPtr;
  int32_t* numCombineDescriptorsDevicePtr;
  int32_t* planMetadataDevicePtr;
  int64_t* predictedCostDevicePtr;
  int32_t const* costModelTablePtr;
  int32_t const* evaluationCostModelPtr;
  bool selectCostModelOnDevice;
  bool useOptimizedSchedule;
  bool computePredictedCost;
  bool useEvaluationCostModel;
  void* workspacePtr;
  size_t workspaceBytes;
  cudaStream_t stream;
  int32_t forcedTargetPieceTiles = 0;
};

// Launch a fixed, graph-capturable scheduling pipeline that reads live device sequence lengths and
// publishes the compact descriptor ABI. The exact policy uses prepare/order/score/emit kernels;
// the optimized policy uses a lower-overhead semantics-preserving placement. The pipeline never
// synchronizes its stream.
void runBalancedSchedDevice(BalancedSchedDeviceParams const& params);

}  // namespace prims_ts
}  // namespace flashinfer
