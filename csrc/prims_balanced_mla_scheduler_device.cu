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
// Graph-capturable balanced PrimsTS MLA scheduler. Candidate targets are evaluated by independent
// CTAs between a deterministic prepare stage and a deterministic descriptor-emission stage. The
// exact policy preserves the original deterministic placement and descriptor ABI; the optimized
// policy emits a lower-overhead rank-and-fold placement from fixed-capacity global scratch.

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>

#include "flashinfer/exception.h"
#include "prims_balanced_mla_scheduler.cuh"

namespace flashinfer {
namespace prims_ts {
namespace {

constexpr int32_t kSingleBucket = 0;
constexpr int32_t kSparseSmallBucket = 1;
constexpr int32_t kSparseUniformSmallBucket = 2;
constexpr int32_t kSparseBucket = 3;
constexpr int32_t kDenseSmallBucket = 4;
constexpr int32_t kDenseLargeBucket = 5;
constexpr int32_t kLegacyBucket = 6;
constexpr int32_t kMaxCandidates = 128;
constexpr int32_t kScoreThreads = 32;
constexpr int32_t kFoldThreads = 256;

struct DeviceCost {
  int32_t cB, cR, gamma0, gamma1, comb0, comb1;
};

struct DeviceChunk {
  int32_t reqIdx;
  int32_t startBlock;
  int32_t endBlock;
  int32_t numPieces;
  int32_t localPieceIdx;
  int32_t partitionIdx;
  int32_t isSplit;
};

struct DevicePlannerState {
  int64_t totalBlocks;
  int32_t activeCount;
  int32_t maxBlocks;
  int32_t bucket;
  int32_t candidateCount;
  int32_t target;
  int32_t usesCostSearch;
  int32_t useAdditiveCombine;
  int32_t noSplitFastPath;
  int32_t failed;
};

struct DeviceWorkspace {
  DevicePlannerState* state;
  int32_t* blocks;
  int32_t* lptOrder;
  int32_t* longOrder;
  int32_t* piecesPerRequest;
  int32_t* splitBegin;
  int32_t* chunkBegin;
  int32_t* chunkCount;
  int32_t* candidates;
  int32_t* candidateDescriptors;
  int32_t* selected;
  int32_t* partitionCounts;
  int64_t* candidateCosts;
  int64_t* run;
  DeviceChunk* chunks;
  int64_t* chunkCosts;
  int32_t* chunkRanks;
};

__device__ __forceinline__ char* alignWorkspaceCursor(char* cursor, size_t alignment) {
  uintptr_t const value = reinterpret_cast<uintptr_t>(cursor);
  return reinterpret_cast<char*>((value + alignment - 1) & ~(alignment - 1));
}

__device__ __forceinline__ DeviceWorkspace getWorkspace(BalancedSchedDeviceParams const& p) {
  int32_t const N = p.batchSize;
  int32_t const P = p.numSmParts;
  char* cursor = static_cast<char*>(p.workspacePtr);
  DeviceWorkspace ws;
  ws.state = reinterpret_cast<DevicePlannerState*>(cursor);
  cursor += sizeof(DevicePlannerState);
  ws.blocks = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(N) * sizeof(int32_t);
  ws.lptOrder = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(N) * sizeof(int32_t);
  ws.longOrder = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(N) * sizeof(int32_t);
  ws.piecesPerRequest = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(N) * sizeof(int32_t);
  ws.splitBegin = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(N) * sizeof(int32_t);
  ws.chunkBegin = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(N) * sizeof(int32_t);
  ws.chunkCount = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(N) * sizeof(int32_t);
  ws.candidates = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(kMaxCandidates) * sizeof(int32_t);
  ws.candidateDescriptors = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(kMaxCandidates) * sizeof(int32_t);
  ws.selected = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(P) * sizeof(int32_t);
  ws.partitionCounts = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(P) * sizeof(int32_t);
  cursor = alignWorkspaceCursor(cursor, alignof(int64_t));
  ws.candidateCosts = reinterpret_cast<int64_t*>(cursor);
  cursor += static_cast<size_t>(kMaxCandidates) * sizeof(int64_t);
  ws.run = reinterpret_cast<int64_t*>(cursor);
  cursor += static_cast<size_t>(P) * sizeof(int64_t);
  ws.chunks = reinterpret_cast<DeviceChunk*>(cursor);
  cursor += static_cast<size_t>(N + P) * sizeof(DeviceChunk);
  cursor = alignWorkspaceCursor(cursor, alignof(int64_t));
  ws.chunkCosts = reinterpret_cast<int64_t*>(cursor);
  cursor += static_cast<size_t>(N + P) * sizeof(int64_t);
  ws.chunkRanks = reinterpret_cast<int32_t*>(cursor);
  return ws;
}

__device__ __forceinline__ int32_t ceilDivI32(int32_t value, int32_t divisor) {
  return static_cast<int32_t>((static_cast<int64_t>(value) + divisor - 1) / divisor);
}

__device__ __forceinline__ int64_t pieceCost(int32_t blocks, bool split, int32_t numPieces,
                                             DeviceCost const& cm) {
  int64_t cost = static_cast<int64_t>(cm.cR) + static_cast<int64_t>(blocks) * cm.cB;
  if (split) cost += static_cast<int64_t>(cm.gamma1) + cm.gamma0 / (numPieces > 0 ? numPieces : 1);
  return cost;
}

__device__ __forceinline__ int64_t combineCost(int32_t maxPieces, DeviceCost const& cm) {
  if (maxPieces < 2) return 0;
  return static_cast<int64_t>(cm.comb0) + static_cast<int64_t>(cm.comb1) * maxPieces;
}

__device__ __forceinline__ bool isSelected(int32_t pid, int32_t const* selected,
                                           int32_t selectedCount) {
  for (int32_t i = 0; i < selectedCount; ++i)
    if (selected[i] == pid) return true;
  return false;
}

__device__ void sortSelected(int32_t* selected, int32_t count) {
  for (int32_t i = 1; i < count; ++i) {
    int32_t value = selected[i], j = i;
    while (j > 0 && value < selected[j - 1]) {
      selected[j] = selected[j - 1];
      --j;
    }
    selected[j] = value;
  }
}

__device__ int32_t leastLoadedPartitionWarp(int64_t const* run, int32_t P, int32_t const* selected,
                                            int32_t selectedCount) {
  int32_t const lane = threadIdx.x % warpSize;
  int64_t bestLoad = INT64_MAX;
  int32_t bestPartition = INT_MAX;
  for (int32_t pid = lane; pid < P; pid += warpSize) {
    if (isSelected(pid, selected, selectedCount)) continue;
    int64_t const load = run[pid];
    if (load < bestLoad || (load == bestLoad && pid < bestPartition)) {
      bestLoad = load;
      bestPartition = pid;
    }
  }
  constexpr uint32_t kFullWarp = 0xffffffffU;
  for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
    int64_t const otherLoad = __shfl_down_sync(kFullWarp, bestLoad, offset);
    int32_t const otherPartition = __shfl_down_sync(kFullWarp, bestPartition, offset);
    if (otherLoad < bestLoad || (otherLoad == bestLoad && otherPartition < bestPartition)) {
      bestLoad = otherLoad;
      bestPartition = otherPartition;
    }
  }
  return __shfl_sync(kFullWarp, bestPartition, 0);
}

__device__ void assignShortsForCandidateWarp(int64_t* run, int32_t const* lptOrder,
                                             int32_t activeCount, int32_t const* blocks,
                                             DeviceCost const& cm, int32_t P, int32_t target) {
  for (int32_t orderIdx = 0; orderIdx < activeCount; ++orderIdx) {
    int32_t const req = lptOrder[orderIdx];
    if (blocks[req] > target) continue;
    int32_t const pid = leastLoadedPartitionWarp(run, P, nullptr, 0);
    if (threadIdx.x % warpSize == 0) run[pid] += pieceCost(blocks[req], false, 0, cm);
    __syncwarp();
  }
}

__device__ void assignLongsForCandidateWarp(int64_t* run, int32_t const* longOrder,
                                            int32_t activeCount, int32_t const* blocks,
                                            DeviceCost const& cm, int32_t P, int32_t target,
                                            bool chargeSplitPenalty, int32_t* selected) {
  for (int32_t orderIdx = 0; orderIdx < activeCount; ++orderIdx) {
    int32_t const req = longOrder[orderIdx];
    int32_t const blockCount = blocks[req];
    if (blockCount <= target) continue;
    int32_t pieces = ceilDivI32(blockCount, target);
    if (pieces > 127) pieces = 127;
    if (pieces > P) pieces = P;
    for (int32_t j = 0; j < pieces; ++j) {
      int32_t const pid = leastLoadedPartitionWarp(run, P, selected, j);
      if (threadIdx.x % warpSize == 0) selected[j] = pid;
      __syncwarp();
    }
    if (threadIdx.x % warpSize == 0) sortSelected(selected, pieces);
    __syncwarp();
    int32_t const base = blockCount / pieces;
    int32_t const remainder = blockCount - base * pieces;
    for (int32_t j = threadIdx.x % warpSize; j < pieces; j += warpSize) {
      int32_t const pid = selected[j];
      int32_t const pieceBlocks = base + (j < remainder ? 1 : 0);
      run[pid] += pieceCost(pieceBlocks, chargeSplitPenalty && pieces > 1, pieces, cm);
    }
    __syncwarp();
  }
}

__device__ int32_t selectCostBucket(int32_t activeCount, int64_t totalBlocks, int32_t maxBlocks,
                                    int32_t P) {
  if (activeCount <= 1) return kSingleBucket;
  int64_t const averageBlocks = (totalBlocks + P - 1) / P;
  if (averageBlocks < 32) {
    if (activeCount > 64 && averageBlocks < 16) return kLegacyBucket;
    int64_t const averageActiveBlocks = (totalBlocks + activeCount - 1) / activeCount;
    if (activeCount <= 8 && maxBlocks >= 4 * averageActiveBlocks) return kSparseSmallBucket;
    if (activeCount <= 8) {
      if (maxBlocks >= 2 * averageActiveBlocks) return kLegacyBucket;
      return kSparseUniformSmallBucket;
    }
    return kSparseBucket;
  }
  return activeCount <= 32 ? kDenseSmallBucket : kDenseLargeBucket;
}

__device__ DeviceCost loadCost(int32_t const* table, int32_t bucket) {
  int32_t const* row = table + bucket * kBalancedDeviceCostCoefficientCount;
  return {row[0], row[1], row[2], row[3], row[4], row[5]};
}

__device__ void failPlan(BalancedSchedDeviceParams const& p, BalancedSchedStatus status,
                         int32_t bucket) {
  p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kStatus)] =
      static_cast<int32_t>(status);
  p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kCostBucket)] = bucket;
}

__device__ __forceinline__ int32_t blockExclusiveScanI32(int32_t value, int32_t* warpScratch) {
  int32_t const lane = threadIdx.x & (warpSize - 1);
  int32_t const warp = threadIdx.x / warpSize;
  int32_t inclusive = value;
  constexpr uint32_t kFullWarp = 0xffffffffU;
  for (int32_t offset = 1; offset < warpSize; offset *= 2) {
    int32_t const other = __shfl_up_sync(kFullWarp, inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == warpSize - 1) warpScratch[warp] = inclusive;
  __syncthreads();
  if (warp == 0) {
    int32_t warpInclusive = lane < blockDim.x / warpSize ? warpScratch[lane] : 0;
    for (int32_t offset = 1; offset < warpSize; offset *= 2) {
      int32_t const other = __shfl_up_sync(kFullWarp, warpInclusive, offset);
      if (lane >= offset) warpInclusive += other;
    }
    if (lane < blockDim.x / warpSize) warpScratch[lane] = warpInclusive;
  }
  __syncthreads();
  int32_t const warpPrefix = warp == 0 ? 0 : warpScratch[warp - 1];
  int32_t const exclusive = warpPrefix + inclusive - value;
  __syncthreads();
  return exclusive;
}

__global__ void balancedSchedPrepareKernel(BalancedSchedDeviceParams p) {
  int32_t const N = p.batchSize;
  int32_t const P = p.numSmParts;
  DeviceWorkspace const ws = getWorkspace(p);

  __shared__ int32_t activeCount;
  __shared__ int32_t maxBlocks;
  __shared__ int32_t minBlocks;
  __shared__ int32_t invalidLength;
  __shared__ unsigned long long totalBlocks;
  __shared__ int32_t scanScratch[8];
  __shared__ unsigned long long warpTotals[8];
  __shared__ int32_t warpMaxBlocks[8];
  __shared__ int32_t warpMinBlocks[8];
  __shared__ int32_t warpInvalid[8];
  if (threadIdx.x == 0) {
    *p.numCombineDescriptorsDevicePtr = 0;
    ws.state->activeCount = 0;
    ws.state->totalBlocks = 0;
    ws.state->maxBlocks = 0;
    ws.state->bucket = kSingleBucket;
    ws.state->candidateCount = 0;
    ws.state->target = 0;
    ws.state->usesCostSearch = 0;
    ws.state->useAdditiveCombine = 0;
    ws.state->noSplitFastPath = 0;
    ws.state->failed = 0;
    activeCount = 0;
    maxBlocks = 0;
    minBlocks = INT_MAX;
    invalidLength = 0;
    totalBlocks = 0;
  }
  for (int32_t pid = threadIdx.x; pid <= P; pid += blockDim.x) p.workDescriptorOffsetsPtr[pid] = 0;
  for (int32_t i = threadIdx.x; i < static_cast<int32_t>(BalancedSchedMetadata::kCount);
       i += blockDim.x)
    p.planMetadataDevicePtr[i] = 0;
  __syncthreads();
  if (threadIdx.x == 0)
    p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kCostBucket)] = -1;

  if (N <= blockDim.x) {
    int32_t const req = threadIdx.x;
    int32_t const seqLen = req < N ? p.seqLensKvPtr[req] : 0;
    int32_t const isInvalid = req < N && (seqLen < 0 || seqLen > p.maxSeqLen);
    int32_t const blockCount =
        req < N && seqLen > 0 && seqLen <= p.maxSeqLen ? ceilDivI32(seqLen, p.blockSizeN) : 0;
    if (req < N) {
      ws.blocks[req] = blockCount;
      ws.piecesPerRequest[req] = 0;
      ws.splitBegin[req] = 0;
      ws.chunkBegin[req] = -1;
      ws.chunkCount[req] = 0;
    }
    int32_t const isActive = blockCount > 0 ? 1 : 0;
    int32_t const activeIdx = blockExclusiveScanI32(isActive, scanScratch);
    if (isActive) {
      ws.lptOrder[activeIdx] = req;
      ws.longOrder[activeIdx] = req;
    }
    if (threadIdx.x == blockDim.x - 1) activeCount = activeIdx + isActive;

    int32_t const lane = threadIdx.x & (warpSize - 1);
    int32_t const warp = threadIdx.x / warpSize;
    unsigned long long warpTotal = static_cast<unsigned long long>(blockCount);
    int32_t warpMax = blockCount;
    int32_t warpMin = isActive ? blockCount : INT_MAX;
    int32_t warpHasInvalid = isInvalid;
    constexpr uint32_t kFullWarp = 0xffffffffU;
    for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
      warpTotal += __shfl_down_sync(kFullWarp, warpTotal, offset);
      int32_t const otherMax = __shfl_down_sync(kFullWarp, warpMax, offset);
      int32_t const otherMin = __shfl_down_sync(kFullWarp, warpMin, offset);
      int32_t const otherInvalid = __shfl_down_sync(kFullWarp, warpHasInvalid, offset);
      if (otherMax > warpMax) warpMax = otherMax;
      if (otherMin < warpMin) warpMin = otherMin;
      warpHasInvalid |= otherInvalid;
    }
    if (lane == 0) {
      warpTotals[warp] = warpTotal;
      warpMaxBlocks[warp] = warpMax;
      warpMinBlocks[warp] = warpMin;
      warpInvalid[warp] = warpHasInvalid;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int32_t warpIdx = 0; warpIdx < blockDim.x / warpSize; ++warpIdx) {
        totalBlocks += warpTotals[warpIdx];
        if (warpMaxBlocks[warpIdx] > maxBlocks) maxBlocks = warpMaxBlocks[warpIdx];
        if (warpMinBlocks[warpIdx] < minBlocks) minBlocks = warpMinBlocks[warpIdx];
        invalidLength |= warpInvalid[warpIdx];
      }
    }
  } else {
    for (int32_t req = threadIdx.x; req < N; req += blockDim.x) {
      int32_t const seqLen = p.seqLensKvPtr[req];
      if (seqLen < 0 || seqLen > p.maxSeqLen) atomicExch(&invalidLength, 1);
      int32_t const blockCount =
          seqLen > 0 && seqLen <= p.maxSeqLen ? ceilDivI32(seqLen, p.blockSizeN) : 0;
      ws.blocks[req] = blockCount;
      ws.piecesPerRequest[req] = 0;
      ws.splitBegin[req] = 0;
      ws.chunkBegin[req] = -1;
      ws.chunkCount[req] = 0;
      if (blockCount > 0) {
        int32_t const activeIdx = atomicAdd(&activeCount, 1);
        ws.lptOrder[activeIdx] = req;
        ws.longOrder[activeIdx] = req;
        atomicAdd(&totalBlocks, static_cast<unsigned long long>(blockCount));
        atomicMax(&maxBlocks, blockCount);
        atomicMin(&minBlocks, blockCount);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  if (invalidLength) {
    ws.state->failed = 1;
    failPlan(p, BalancedSchedStatus::kInvalidSequenceLength, -1);
    return;
  }

  int32_t const bucket =
      p.selectCostModelOnDevice
          ? selectCostBucket(activeCount, static_cast<int64_t>(totalBlocks), maxBlocks, P)
          : kSingleBucket;
  ws.state->activeCount = activeCount;
  ws.state->totalBlocks = static_cast<int64_t>(totalBlocks);
  ws.state->maxBlocks = maxBlocks;
  ws.state->bucket = bucket;
  p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kCostBucket)] =
      p.selectCostModelOnDevice ? bucket : -1;
  DeviceCost const cm = loadCost(p.costModelTablePtr, bucket);
  if (activeCount == 0) return;

  int64_t const yAverage = (static_cast<int64_t>(totalBlocks) + P - 1) / P;
  int64_t const yBreak =
      cm.cB > 0 ? (static_cast<int64_t>(cm.gamma0) + 2LL * cm.gamma1 + cm.cB - 1) / cm.cB + 3
                : 1000000000LL;
  int64_t const positiveCB = cm.cB > 0 ? cm.cB : 1;
  int64_t const yMinimum =
      (static_cast<int64_t>(cm.cR) + cm.gamma1 + cm.gamma0 / 2 + positiveCB - 1) / positiveCB + 1;
  int64_t const yCap = (static_cast<int64_t>(maxBlocks) + 125) / 126;
  int64_t baseTarget64 = yAverage + yBreak;
  if (yMinimum > baseTarget64) baseTarget64 = yMinimum;
  if (yCap > baseTarget64) baseTarget64 = yCap;
  if (baseTarget64 < 1) baseTarget64 = 1;
  if (baseTarget64 > INT_MAX) {
    ws.state->failed = 1;
    failPlan(p, BalancedSchedStatus::kTargetOverflow, p.selectCostModelOnDevice ? bucket : -1);
    return;
  }

  int32_t const baseTarget = static_cast<int32_t>(baseTarget64);
  bool const usesForcedTarget = p.forcedTargetPieceTiles > 0;
  bool const usesCostSearch = !usesForcedTarget && activeCount <= 2 * P;
  bool const useAdditiveCombine = cm.comb0 != 0 || cm.comb1 != 0;
  ws.state->target = usesForcedTarget ? p.forcedTargetPieceTiles : baseTarget;
  ws.state->usesCostSearch = usesCostSearch;
  ws.state->useAdditiveCombine = useAdditiveCombine;

  if (usesCostSearch) {
    int32_t candidateCount = 0;
    ws.candidates[candidateCount++] = baseTarget;
    int32_t const kMax = P < 127 ? P : 127;
    for (int32_t k = 1; k <= kMax; ++k) {
      int32_t const floorTarget =
          static_cast<int64_t>(k) * activeCount <= P ? 1 : static_cast<int32_t>(yMinimum);
      int32_t candidate = ceilDivI32(maxBlocks, k);
      if (floorTarget > candidate) candidate = floorTarget;
      if (yCap > candidate) candidate = static_cast<int32_t>(yCap);
      if (candidate < 1) candidate = 1;
      ws.candidates[candidateCount++] = candidate;
    }
    ws.state->candidateCount = candidateCount;
  }

  bool const spreadUniform = static_cast<int64_t>(maxBlocks) - minBlocks <= yBreak;
  bool const enoughRequests = 2LL * activeCount >= P;
  ws.state->noSplitFastPath =
      spreadUniform && enoughRequests && !usesCostSearch && !usesForcedTarget;
}

__global__ void balancedSchedSortOrdersKernel(BalancedSchedDeviceParams p) {
  DeviceWorkspace const ws = getWorkspace(p);
  if (ws.state->failed || ws.state->activeCount == 0) return;

  int32_t const activeCount = ws.state->activeCount;
  DeviceCost const cm = loadCost(p.costModelTablePtr, ws.state->bucket);
  for (int32_t index = threadIdx.x; index < activeCount; index += blockDim.x) {
    int32_t const req = ws.lptOrder[index];
    int64_t const reqCost = pieceCost(ws.blocks[req], false, 0, cm);
    int32_t lptRank = 0;
    int32_t longRank = 0;
    for (int32_t otherIndex = 0; otherIndex < activeCount; ++otherIndex) {
      int32_t const otherReq = ws.lptOrder[otherIndex];
      int64_t const otherCost = pieceCost(ws.blocks[otherReq], false, 0, cm);
      if (otherCost > reqCost || (otherCost == reqCost && otherReq < req)) ++lptRank;
      if (ws.blocks[otherReq] > ws.blocks[req] ||
          (ws.blocks[otherReq] == ws.blocks[req] && otherReq < req))
        ++longRank;
    }
    ws.chunkBegin[lptRank] = req;
    ws.chunkCount[longRank] = req;
  }
  __syncthreads();
  for (int32_t index = threadIdx.x; index < activeCount; index += blockDim.x) {
    ws.lptOrder[index] = ws.chunkBegin[index];
    ws.longOrder[index] = ws.chunkCount[index];
  }
  __syncthreads();
  for (int32_t req = threadIdx.x; req < p.batchSize; req += blockDim.x) {
    ws.chunkBegin[req] = -1;
    ws.chunkCount[req] = 0;
  }
}

__global__ void balancedSchedScoreKernel(BalancedSchedDeviceParams p) {
  int32_t const candidateIdx = blockIdx.x;
  if (candidateIdx >= kMaxCandidates) return;

  DeviceWorkspace const ws = getWorkspace(p);
  if (threadIdx.x == 0) ws.candidateCosts[candidateIdx] = INT64_MAX;
  if (threadIdx.x == 0) ws.candidateDescriptors[candidateIdx] = INT_MAX;
  if (ws.state->failed || !ws.state->usesCostSearch || candidateIdx >= ws.state->candidateCount)
    return;

  int32_t const N = p.batchSize;
  int32_t const P = p.numSmParts;
  int32_t const activeCount = ws.state->activeCount;
  int32_t const target = ws.candidates[candidateIdx];
  DeviceCost const cm = loadCost(p.costModelTablePtr, ws.state->bucket);
  bool const useAdditiveCombine = ws.state->useAdditiveCombine;

  extern __shared__ int64_t sharedRun[];
  int32_t* selected = reinterpret_cast<int32_t*>(sharedRun + P);

  int64_t descriptorCount = 0;
  int32_t maxPieces = 0;
  for (int32_t req = 0; req < N; ++req) {
    int32_t const blockCount = ws.blocks[req];
    int32_t requestPieces = 0;
    if (blockCount > 0) {
      requestPieces = blockCount > target ? ceilDivI32(blockCount, target) : 1;
      if (requestPieces > 127) requestPieces = 127;
      if (requestPieces > P) requestPieces = P;
    }
    descriptorCount += requestPieces;
    if (requestPieces > maxPieces) maxPieces = requestPieces;
  }
  if (descriptorCount > static_cast<int64_t>(activeCount) + P) return;
  if (threadIdx.x == 0) ws.candidateDescriptors[candidateIdx] = descriptorCount;

  for (int32_t pid = threadIdx.x; pid < P; pid += warpSize) {
    sharedRun[pid] = 0;
  }
  __syncwarp();
  assignShortsForCandidateWarp(sharedRun, ws.lptOrder, activeCount, ws.blocks, cm, P, target);
  assignLongsForCandidateWarp(sharedRun, ws.longOrder, activeCount, ws.blocks, cm, P, target,
                              !useAdditiveCombine, selected);
  if (threadIdx.x == 0) {
    int64_t makespan = 0;
    for (int32_t pid = 0; pid < P; ++pid)
      if (sharedRun[pid] > makespan) makespan = sharedRun[pid];
    ws.candidateCosts[candidateIdx] =
        makespan + (useAdditiveCombine ? combineCost(maxPieces, cm) : 0);
  }
}

__device__ __forceinline__ int32_t foldedPartition(int32_t rank, int32_t P) {
  int32_t const row = rank / P;
  int32_t const column = rank - row * P;
  return (row & 1) ? P - 1 - column : column;
}

// Score one target per CTA with a parallel, order-independent approximation of LPT. Chunks are
// ranked by cost and successive rows are folded over the partitions in alternating directions.
// Equal-cost chunk identity does not affect the resulting per-partition load, so the atomic chunk
// allocation below need not reproduce exact-policy descriptor order.
__global__ void balancedSchedFoldPrepareScoreKernel(BalancedSchedDeviceParams p) {
  int32_t const candidateIdx = blockIdx.x;
  if (candidateIdx >= kMaxCandidates) return;

  int32_t const N = p.batchSize;
  int32_t const P = p.numSmParts;
  int32_t const capacity = N + P;
  DeviceWorkspace const ws = getWorkspace(p);

  extern __shared__ char sharedStorage[];
  char* cursor = sharedStorage;
  int32_t* sharedBlocks = reinterpret_cast<int32_t*>(cursor);
  cursor += static_cast<size_t>(N) * sizeof(int32_t);
  cursor = alignWorkspaceCursor(cursor, alignof(int64_t));
  int64_t* chunkCosts = reinterpret_cast<int64_t*>(cursor);
  cursor += static_cast<size_t>(capacity) * sizeof(int64_t);
  cursor = alignWorkspaceCursor(cursor, alignof(int64_t));
  int64_t* partitionLoads = reinterpret_cast<int64_t*>(cursor);

  __shared__ int32_t warpActive[8];
  __shared__ unsigned long long warpTotal[8];
  __shared__ int32_t warpMaximum[8];
  __shared__ int32_t warpMinimum[8];
  __shared__ int32_t warpInvalid[8];
  __shared__ int32_t activeCount;
  __shared__ unsigned long long totalBlocks;
  __shared__ int32_t maxBlocks;
  __shared__ int32_t minBlocks;
  __shared__ int32_t bucket;
  __shared__ int32_t candidateCount;
  __shared__ int32_t target;
  __shared__ int32_t usesCostSearch;
  __shared__ int32_t useAdditiveCombine;
  __shared__ int32_t noSplitFastPath;
  __shared__ int32_t failureStatus;
  __shared__ int32_t totalChunks;
  __shared__ int32_t maxPieces;

  if (candidateIdx == 0) {
    for (int32_t pid = threadIdx.x; pid <= P; pid += blockDim.x)
      p.workDescriptorOffsetsPtr[pid] = 0;
    for (int32_t index = threadIdx.x; index < static_cast<int32_t>(BalancedSchedMetadata::kCount);
         index += blockDim.x)
      p.planMetadataDevicePtr[index] = 0;
    if (threadIdx.x == 0) {
      *p.numCombineDescriptorsDevicePtr = 0;
      p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kCostBucket)] = -1;
    }
  }

  int32_t localActive = 0;
  unsigned long long localTotal = 0;
  int32_t localMaximum = 0;
  int32_t localMinimum = INT_MAX;
  int32_t localInvalid = 0;
  for (int32_t req = threadIdx.x; req < N; req += blockDim.x) {
    int32_t const seqLen = p.seqLensKvPtr[req];
    bool const valid = seqLen >= 0 && seqLen <= p.maxSeqLen;
    int32_t const blockCount = valid && seqLen > 0 ? ceilDivI32(seqLen, p.blockSizeN) : 0;
    sharedBlocks[req] = blockCount;
    if (candidateIdx == 0) ws.blocks[req] = blockCount;
    localInvalid |= !valid;
    if (blockCount > 0) {
      ++localActive;
      localTotal += static_cast<unsigned long long>(blockCount);
      if (blockCount > localMaximum) localMaximum = blockCount;
      if (blockCount < localMinimum) localMinimum = blockCount;
    }
  }
  int32_t const lane = threadIdx.x & (warpSize - 1);
  int32_t const warp = threadIdx.x / warpSize;
  constexpr uint32_t kFullWarp = 0xffffffffU;
  for (int32_t offset = warpSize / 2; offset > 0; offset /= 2) {
    localActive += __shfl_down_sync(kFullWarp, localActive, offset);
    localTotal += __shfl_down_sync(kFullWarp, localTotal, offset);
    int32_t const otherMaximum = __shfl_down_sync(kFullWarp, localMaximum, offset);
    int32_t const otherMinimum = __shfl_down_sync(kFullWarp, localMinimum, offset);
    localInvalid |= __shfl_down_sync(kFullWarp, localInvalid, offset);
    if (otherMaximum > localMaximum) localMaximum = otherMaximum;
    if (otherMinimum < localMinimum) localMinimum = otherMinimum;
  }
  if (lane == 0) {
    warpActive[warp] = localActive;
    warpTotal[warp] = localTotal;
    warpMaximum[warp] = localMaximum;
    warpMinimum[warp] = localMinimum;
    warpInvalid[warp] = localInvalid;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    activeCount = 0;
    totalBlocks = 0;
    maxBlocks = 0;
    minBlocks = INT_MAX;
    failureStatus = 0;
    for (int32_t warpIdx = 0; warpIdx < blockDim.x / warpSize; ++warpIdx) {
      activeCount += warpActive[warpIdx];
      totalBlocks += warpTotal[warpIdx];
      if (warpMaximum[warpIdx] > maxBlocks) maxBlocks = warpMaximum[warpIdx];
      if (warpMinimum[warpIdx] < minBlocks) minBlocks = warpMinimum[warpIdx];
      if (warpInvalid[warpIdx])
        failureStatus = static_cast<int32_t>(BalancedSchedStatus::kInvalidSequenceLength);
    }
    bucket = p.selectCostModelOnDevice
                 ? selectCostBucket(activeCount, static_cast<int64_t>(totalBlocks), maxBlocks, P)
                 : kSingleBucket;
    DeviceCost const aggregateCost = loadCost(p.costModelTablePtr, bucket);
    int64_t const yAverage = (static_cast<int64_t>(totalBlocks) + P - 1) / P;
    int64_t const yBreak = aggregateCost.cB > 0 ? (static_cast<int64_t>(aggregateCost.gamma0) +
                                                   2LL * aggregateCost.gamma1 + aggregateCost.cB -
                                                   1) / aggregateCost.cB +
                                                      3
                                                : 1000000000LL;
    int64_t const positiveCB = aggregateCost.cB > 0 ? aggregateCost.cB : 1;
    int64_t const yMinimum = (static_cast<int64_t>(aggregateCost.cR) + aggregateCost.gamma1 +
                              aggregateCost.gamma0 / 2 + positiveCB - 1) /
                                 positiveCB +
                             1;
    int64_t const yCap = (static_cast<int64_t>(maxBlocks) + 125) / 126;
    int64_t baseTarget = yAverage + yBreak;
    if (yMinimum > baseTarget) baseTarget = yMinimum;
    if (yCap > baseTarget) baseTarget = yCap;
    if (baseTarget < 1) baseTarget = 1;
    if (baseTarget > INT_MAX) {
      failureStatus = static_cast<int32_t>(BalancedSchedStatus::kTargetOverflow);
      baseTarget = 1;
    }
    bool const usesForcedTarget = p.forcedTargetPieceTiles > 0;
    usesCostSearch = !usesForcedTarget && activeCount <= 2 * P;
    useAdditiveCombine = aggregateCost.comb0 != 0 || aggregateCost.comb1 != 0;
    candidateCount = usesCostSearch ? 1 + (P < 127 ? P : 127) : 0;
    if (candidateIdx == 0) {
      target = usesForcedTarget ? p.forcedTargetPieceTiles : static_cast<int32_t>(baseTarget);
    } else {
      int32_t const floorTarget = static_cast<int64_t>(candidateIdx) * activeCount <= P
                                      ? 1
                                      : static_cast<int32_t>(yMinimum);
      int32_t candidate = ceilDivI32(maxBlocks, candidateIdx);
      if (floorTarget > candidate) candidate = floorTarget;
      if (yCap > candidate) candidate = static_cast<int32_t>(yCap);
      if (candidate < 1) candidate = 1;
      target = candidate;
    }
    bool const spreadUniform = static_cast<int64_t>(maxBlocks) - minBlocks <= yBreak;
    bool const enoughRequests = 2LL * activeCount >= P;
    noSplitFastPath = spreadUniform && enoughRequests && !usesCostSearch && !usesForcedTarget;
    totalChunks = 0;
    maxPieces = 0;
    ws.candidateCosts[candidateIdx] = INT64_MAX;
    ws.candidateDescriptors[candidateIdx] = INT_MAX;
    if (candidateIdx < candidateCount) ws.candidates[candidateIdx] = target;
    if (candidateIdx == 0) {
      ws.state->activeCount = activeCount;
      ws.state->totalBlocks = static_cast<int64_t>(totalBlocks);
      ws.state->maxBlocks = maxBlocks;
      ws.state->bucket = bucket;
      ws.state->candidateCount = candidateCount;
      ws.state->target = target;
      ws.state->usesCostSearch = usesCostSearch;
      ws.state->useAdditiveCombine = useAdditiveCombine;
      ws.state->noSplitFastPath = noSplitFastPath;
      ws.state->failed = failureStatus != 0;
      p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kCostBucket)] =
          p.selectCostModelOnDevice ? bucket : -1;
      if (failureStatus != 0)
        failPlan(p, static_cast<BalancedSchedStatus>(failureStatus),
                 p.selectCostModelOnDevice ? bucket : -1);
    }
  }
  __syncthreads();

  if (failureStatus != 0 || activeCount == 0 || !usesCostSearch || candidateIdx >= candidateCount)
    return;
  DeviceCost const cm = loadCost(p.costModelTablePtr, bucket);

  for (int32_t req = threadIdx.x; req < N; req += blockDim.x) {
    int32_t const blockCount = sharedBlocks[req];
    if (blockCount <= 0) continue;
    int32_t requestPieces = blockCount > target ? ceilDivI32(blockCount, target) : 1;
    if (requestPieces > 127) requestPieces = 127;
    if (requestPieces > P) requestPieces = P;
    int32_t const chunkBase = atomicAdd(&totalChunks, requestPieces);
    atomicMax(&maxPieces, requestPieces);
    int32_t const base = blockCount / requestPieces;
    int32_t const remainder = blockCount - base * requestPieces;
    for (int32_t j = 0; j < requestPieces; ++j) {
      int32_t const chunkIdx = chunkBase + j;
      if (chunkIdx >= capacity) continue;
      int32_t const pieceBlocks = base + (j < remainder ? 1 : 0);
      chunkCosts[chunkIdx] =
          pieceCost(pieceBlocks, !useAdditiveCombine && requestPieces > 1, requestPieces, cm);
    }
  }
  __syncthreads();
  if (totalChunks > activeCount + P || totalChunks > capacity) return;
  if (threadIdx.x == 0) ws.candidateDescriptors[candidateIdx] = totalChunks;

  for (int32_t pid = threadIdx.x; pid < P; pid += blockDim.x) partitionLoads[pid] = 0;
  __syncthreads();
  for (int32_t chunkIdx = threadIdx.x; chunkIdx < totalChunks; chunkIdx += blockDim.x) {
    int64_t const cost = chunkCosts[chunkIdx];
    int32_t rank = 0;
    for (int32_t other = 0; other < totalChunks; ++other) {
      int64_t const otherCost = chunkCosts[other];
      if (otherCost > cost || (otherCost == cost && other < chunkIdx)) ++rank;
    }
    int32_t const pid = foldedPartition(rank, P);
    atomicAdd(reinterpret_cast<unsigned long long*>(partitionLoads + pid),
              static_cast<unsigned long long>(cost));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int64_t makespan = 0;
    for (int32_t pid = 0; pid < P; ++pid)
      if (partitionLoads[pid] > makespan) makespan = partitionLoads[pid];
    ws.candidateCosts[candidateIdx] =
        makespan + (useAdditiveCombine ? combineCost(maxPieces, cm) : 0);
  }
}

// Emit the optimized descriptor set for the winning folded target. Within every
// partition descriptors are ordered by descending modeled cost. The rank-to-row mapping gives
// every chunk its output slot directly, avoiding a global sort or atomic scatter.
__global__ void balancedSchedFoldEmitKernel(BalancedSchedDeviceParams p) {
  int32_t const N = p.batchSize;
  int32_t const P = p.numSmParts;
  DeviceWorkspace const ws = getWorkspace(p);
  if (ws.state->failed || ws.state->activeCount == 0) return;

  int32_t const activeCount = ws.state->activeCount;
  int32_t const bucket = ws.state->bucket;
  DeviceCost const cm = loadCost(p.costModelTablePtr, bucket);
  bool const useAdditiveCombine = ws.state->useAdditiveCombine;
  bool const noSplitFastPath = ws.state->noSplitFastPath;

  __shared__ int64_t selectionCosts[kFoldThreads];
  __shared__ int32_t selectionTargets[kFoldThreads];
  __shared__ int64_t bestCandidateCost;
  __shared__ int32_t bestCandidateTarget;
  __shared__ int32_t scanScratch[8];
  if (ws.state->usesCostSearch) {
    int32_t const candidateIdx = threadIdx.x;
    if (candidateIdx < ws.state->candidateCount) {
      selectionCosts[candidateIdx] = ws.candidateCosts[candidateIdx];
      selectionTargets[candidateIdx] = ws.candidates[candidateIdx];
    } else {
      selectionCosts[candidateIdx] = INT64_MAX;
      selectionTargets[candidateIdx] = -1;
    }
    __syncthreads();
    for (int32_t offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        int64_t const otherCost = selectionCosts[threadIdx.x + offset];
        int32_t const otherTarget = selectionTargets[threadIdx.x + offset];
        if (otherCost < selectionCosts[threadIdx.x] ||
            (otherCost == selectionCosts[threadIdx.x] &&
             otherTarget > selectionTargets[threadIdx.x])) {
          selectionCosts[threadIdx.x] = otherCost;
          selectionTargets[threadIdx.x] = otherTarget;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      bestCandidateCost = selectionCosts[0];
      bestCandidateTarget = selectionTargets[0];
    }
    __syncthreads();
    // The calibrated linear model cannot represent the producer's long-piece latency cliff.
    // Back off to the nearest smaller candidate when it is within 5% of optimal. Taking only one
    // candidate step avoids trading a noticeably longer producer loop for a marginal modeled win
    // without over-splitting requests.
    int32_t backedOffTarget = -1;
    if (candidateIdx < ws.state->candidateCount) {
      int64_t const cost = ws.candidateCosts[candidateIdx];
      int32_t const candidate = ws.candidates[candidateIdx];
      if (cost <= bestCandidateCost + bestCandidateCost / 20 && candidate < bestCandidateTarget)
        backedOffTarget = candidate;
    }
    selectionTargets[threadIdx.x] = backedOffTarget;
    __syncthreads();
    for (int32_t offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset &&
          selectionTargets[threadIdx.x + offset] > selectionTargets[threadIdx.x]) {
        selectionTargets[threadIdx.x] = selectionTargets[threadIdx.x + offset];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      ws.state->target = selectionTargets[0] > 0 ? selectionTargets[0] : bestCandidateTarget;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && ws.state->usesCostSearch) {
    int32_t const currentTarget = ws.state->target;
    int32_t currentDescriptors = INT_MAX;
    int64_t currentCost = INT64_MAX;
    for (int32_t candidateIdx = 0; candidateIdx < ws.state->candidateCount; ++candidateIdx) {
      if (ws.candidates[candidateIdx] == currentTarget) {
        if (ws.candidateCosts[candidateIdx] < currentCost)
          currentCost = ws.candidateCosts[candidateIdx];
        if (ws.candidateDescriptors[candidateIdx] < currentDescriptors)
          currentDescriptors = ws.candidateDescriptors[candidateIdx];
      }
    }

    int32_t correctedTarget = currentTarget;
    // A lightly populated second producer wave is a discontinuity that the linear piece model
    // cannot represent. If a one-wave candidate has no worse folded objective, retain the finest
    // candidate that fits in one wave. The small occupancy margin avoids moving schedules that
    // only exceed P because of one or two harmless descriptors.
    if (currentDescriptors >= P + P / 12) {
      int64_t bestOneWaveCost = INT64_MAX;
      int32_t finestOneWaveTarget = INT_MAX;
      for (int32_t candidateIdx = 0; candidateIdx < ws.state->candidateCount; ++candidateIdx) {
        if (ws.candidateDescriptors[candidateIdx] <= P) {
          if (ws.candidateCosts[candidateIdx] < bestOneWaveCost)
            bestOneWaveCost = ws.candidateCosts[candidateIdx];
          if (ws.candidates[candidateIdx] < finestOneWaveTarget)
            finestOneWaveTarget = ws.candidates[candidateIdx];
        }
      }
      if (bestOneWaveCost <= currentCost && finestOneWaveTarget < INT_MAX)
        correctedTarget = finestOneWaveTarget;
    }

    // The calibrated BF16 dense-large model has a high fixed piece cost. Combined with the fold's
    // nearly ideal modeled balance, that can favor a sparse second wave of pieces that are much
    // longer than the average partition. The producer has a measured long-piece latency cliff.
    // Correct only that narrow signature, using request concentration to choose the nearest safe
    // side of the average. This scans existing candidate metadata and does not add a graph node.
    int64_t const totalBlocks = ws.state->totalBlocks;
    int32_t const activeCount = ws.state->activeCount;
    int32_t const maxBlocks = ws.state->maxBlocks;
    bool const highFixedPieceCost =
        static_cast<int64_t>(cm.cR) >= 32LL * static_cast<int64_t>(cm.cB);
    bool const sparseSecondWave =
        currentDescriptors > P && currentDescriptors <= (4 * P + 2) / 3 + 1;
    bool const targetAboveAverage = 4LL * currentTarget * P >= 5LL * totalBlocks;
    if (highFixedPieceCost && sparseSecondWave && targetAboveAverage) {
      bool const concentrationBelowFour =
          static_cast<int64_t>(maxBlocks) * activeCount < 4LL * totalBlocks;
      bool const concentrationAtLeastFive =
          static_cast<int64_t>(maxBlocks) * activeCount >= 5LL * totalBlocks;
      if (concentrationBelowFour) {
        int32_t cappedTarget = -1;
        for (int32_t candidateIdx = 0; candidateIdx < ws.state->candidateCount; ++candidateIdx) {
          int32_t const candidate = ws.candidates[candidateIdx];
          if (ws.candidateDescriptors[candidateIdx] != INT_MAX &&
              5LL * candidate * P <= 4LL * totalBlocks && candidate > cappedTarget)
            cappedTarget = candidate;
        }
        if (cappedTarget > 0) correctedTarget = cappedTarget;
      } else if (concentrationAtLeastFive && totalBlocks < 160LL * P) {
        int64_t bestDistance = INT64_MAX;
        int32_t nearestTarget = currentTarget;
        for (int32_t candidateIdx = 0; candidateIdx < ws.state->candidateCount; ++candidateIdx) {
          if (ws.candidateDescriptors[candidateIdx] == INT_MAX) continue;
          int32_t const candidate = ws.candidates[candidateIdx];
          int64_t distance = static_cast<int64_t>(candidate) * P - totalBlocks;
          if (distance < 0) distance = -distance;
          if (distance < bestDistance || (distance == bestDistance && candidate > nearestTarget)) {
            bestDistance = distance;
            nearestTarget = candidate;
          }
        }
        correctedTarget = nearestTarget;
      }
    }
    ws.state->target = correctedTarget;
  }
  __syncthreads();
  int32_t const target = ws.state->target;

  for (int32_t req = threadIdx.x; req < N; req += blockDim.x) {
    int32_t requestPieces = 0;
    int32_t const blockCount = ws.blocks[req];
    if (blockCount > 0) {
      requestPieces = 1;
      if (!noSplitFastPath && blockCount > target) {
        requestPieces = ceilDivI32(blockCount, target);
        if (requestPieces > 127) requestPieces = 127;
        if (requestPieces > P) requestPieces = P;
      }
    }
    ws.piecesPerRequest[req] = requestPieces;
  }
  __syncthreads();

  __shared__ int32_t descriptorCount;
  __shared__ int32_t combineCount;
  __shared__ int32_t splitPartialCount;
  if (N <= blockDim.x) {
    int32_t const pieces = threadIdx.x < N ? ws.piecesPerRequest[threadIdx.x] : 0;
    int32_t const descriptorPrefix = blockExclusiveScanI32(pieces, scanScratch);
    int32_t const splitPieces = pieces >= 2 ? pieces : 0;
    int32_t const splitPrefix = blockExclusiveScanI32(splitPieces, scanScratch);
    int32_t const isCombine = pieces >= 2 ? 1 : 0;
    int32_t const combinePrefix = blockExclusiveScanI32(isCombine, scanScratch);
    if (threadIdx.x < N) {
      int32_t const req = threadIdx.x;
      ws.chunkBegin[req] = pieces > 0 ? descriptorPrefix : -1;
      ws.chunkCount[req] = pieces;
      ws.splitBegin[req] = pieces >= 2 ? splitPrefix : 0;
      ws.lptOrder[req] = pieces >= 2 ? combinePrefix : -1;
    }
    if (threadIdx.x == blockDim.x - 1) {
      descriptorCount = descriptorPrefix + pieces;
      splitPartialCount = splitPrefix + splitPieces;
      combineCount = combinePrefix + isCombine;
    }
  } else if (threadIdx.x == 0) {
    descriptorCount = 0;
    splitPartialCount = 0;
    combineCount = 0;
    for (int32_t req = 0; req < N; ++req) {
      int32_t const pieces = ws.piecesPerRequest[req];
      ws.chunkBegin[req] = pieces > 0 ? descriptorCount : -1;
      ws.chunkCount[req] = pieces;
      descriptorCount += pieces;
      if (pieces >= 2) {
        ws.splitBegin[req] = splitPartialCount;
        splitPartialCount += pieces;
        ws.lptOrder[req] = combineCount++;
      } else {
        ws.splitBegin[req] = 0;
        ws.lptOrder[req] = -1;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (descriptorCount > activeCount + P || descriptorCount > p.workDescriptorCapacity) {
      ws.state->failed = 1;
      failPlan(p, BalancedSchedStatus::kWorkDescriptorOverflow,
               p.selectCostModelOnDevice ? bucket : -1);
    } else {
      int32_t const partialCapacity = static_cast<int32_t>(
          (static_cast<int64_t>(N) + P < 2LL * P) ? static_cast<int64_t>(N) + P : 2LL * P);
      if (splitPartialCount > partialCapacity) {
        ws.state->failed = 1;
        failPlan(p, BalancedSchedStatus::kPartialOverflow, p.selectCostModelOnDevice ? bucket : -1);
      } else if (combineCount > p.combineDescriptorCapacity) {
        ws.state->failed = 1;
        failPlan(p, BalancedSchedStatus::kCombineDescriptorOverflow,
                 p.selectCostModelOnDevice ? bucket : -1);
      } else if (splitPartialCount > 4096) {
        ws.state->failed = 1;
        failPlan(p, BalancedSchedStatus::kSplitInfoOverflow,
                 p.selectCostModelOnDevice ? bucket : -1);
      }
    }
  }
  __syncthreads();
  if (ws.state->failed) return;

  for (int32_t req = threadIdx.x; req < N; req += blockDim.x) {
    int32_t const requestPieces = ws.piecesPerRequest[req];
    if (requestPieces == 0) continue;
    int32_t const blockCount = ws.blocks[req];
    int32_t const base = blockCount / requestPieces;
    int32_t const remainder = blockCount - base * requestPieces;
    int32_t const chunkBase = ws.chunkBegin[req];
    for (int32_t j = 0; j < requestPieces; ++j) {
      int32_t const pieceBlocks = base + (j < remainder ? 1 : 0);
      int32_t const blockOffset = j * base + (j < remainder ? j : remainder);
      int32_t const chunkIdx = chunkBase + j;
      ws.chunks[chunkIdx] = {req, blockOffset, blockOffset + pieceBlocks, requestPieces,
                             j,   -1,          requestPieces > 1 ? 1 : 0};
      ws.chunkCosts[chunkIdx] =
          pieceCost(pieceBlocks, !useAdditiveCombine && requestPieces > 1, requestPieces, cm);
    }
    if (requestPieces >= 2) {
      p.combineDescriptorPtr[ws.lptOrder[req]] = {req, requestPieces, ws.splitBegin[req], 0};
    }
  }
  for (int32_t pid = threadIdx.x; pid < P; pid += blockDim.x) {
    ws.partitionCounts[pid] = 0;
    selectionCosts[pid] = 0;
  }
  __syncthreads();

  int32_t const fullRowDescriptorCount = descriptorCount / P * P;
  bool const balancePartialRow =
      descriptorCount > fullRowDescriptorCount && fullRowDescriptorCount >= 2 * P;
  for (int32_t chunkIdx = threadIdx.x; chunkIdx < descriptorCount; chunkIdx += blockDim.x) {
    int64_t const cost = ws.chunkCosts[chunkIdx];
    int32_t rank = 0;
    for (int32_t other = 0; other < descriptorCount; ++other) {
      int64_t const otherCost = ws.chunkCosts[other];
      if (otherCost > cost || (otherCost == cost && other < chunkIdx)) ++rank;
    }
    ws.chunkRanks[chunkIdx] = rank;
    if (rank < fullRowDescriptorCount || !balancePartialRow) {
      int32_t const pid = foldedPartition(rank, P);
      ws.chunks[chunkIdx].partitionIdx = pid;
      atomicAdd(ws.partitionCounts + pid, 1);
      if (balancePartialRow && rank < fullRowDescriptorCount) {
        atomicAdd(reinterpret_cast<unsigned long long*>(selectionCosts + pid),
                  static_cast<unsigned long long>(cost));
      }
    }
  }
  __syncthreads();

  // A snake fold pairs complete rows well, but putting every final partial row back at partition
  // zero can stack the largest remainder on the hottest pair. Rank the complete-row loads once and
  // place the remaining descending chunks on the least-loaded partitions. This is a monotone
  // improvement to the folded objective and leaves target selection unchanged.
  if (balancePartialRow) {
    for (int32_t pid = threadIdx.x; pid < P; pid += blockDim.x) {
      int32_t loadRank = 0;
      int64_t const load = selectionCosts[pid];
      for (int32_t other = 0; other < P; ++other) {
        int64_t const otherLoad = selectionCosts[other];
        if (otherLoad < load || (otherLoad == load && other < pid)) ++loadRank;
      }
      selectionTargets[loadRank] = pid;
    }
    __syncthreads();
    for (int32_t chunkIdx = threadIdx.x; chunkIdx < descriptorCount; chunkIdx += blockDim.x) {
      int32_t const rank = ws.chunkRanks[chunkIdx];
      if (rank >= fullRowDescriptorCount) {
        int32_t const pid = selectionTargets[rank - fullRowDescriptorCount];
        ws.chunks[chunkIdx].partitionIdx = pid;
        atomicAdd(ws.partitionCounts + pid, 1);
      }
    }
    __syncthreads();
  }

  if (P <= blockDim.x) {
    int32_t const count = threadIdx.x < P ? ws.partitionCounts[threadIdx.x] : 0;
    int32_t const offset = blockExclusiveScanI32(count, scanScratch);
    if (threadIdx.x < P) p.workDescriptorOffsetsPtr[threadIdx.x] = offset;
    if (threadIdx.x == blockDim.x - 1) p.workDescriptorOffsetsPtr[P] = offset + count;
  } else if (threadIdx.x == 0) {
    int32_t cursor = 0;
    for (int32_t pid = 0; pid < P; ++pid) {
      p.workDescriptorOffsetsPtr[pid] = cursor;
      cursor += ws.partitionCounts[pid];
    }
    p.workDescriptorOffsetsPtr[P] = cursor;
  }
  __syncthreads();

  for (int32_t chunkIdx = threadIdx.x; chunkIdx < descriptorCount; chunkIdx += blockDim.x) {
    DeviceChunk const chunk = ws.chunks[chunkIdx];
    int32_t const rank = ws.chunkRanks[chunkIdx];
    int32_t const outputIdx = p.workDescriptorOffsetsPtr[chunk.partitionIdx] + rank / P;
    int32_t const splitInfo =
        chunk.isSplit ? packSplitInfo(1, ws.splitBegin[chunk.reqIdx] + chunk.localPieceIdx,
                                      chunk.numPieces, ws.splitBegin[chunk.reqIdx])
                      : 0;
    p.workDescriptorPtr[outputIdx] = {chunk.reqIdx, chunk.startBlock, chunk.endBlock, splitInfo};
  }
  if (threadIdx.x == 0) {
    *p.numCombineDescriptorsDevicePtr = combineCount;
    p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kDescriptorCount)] =
        descriptorCount;
    p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kTargetPieceTiles)] =
        target;
    p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kCombineDescriptorCount)] =
        combineCount;
  }
}

__global__ void balancedSchedEmitKernel(BalancedSchedDeviceParams p) {
  int32_t const N = p.batchSize;
  int32_t const P = p.numSmParts;
  int32_t const workCapacity = p.workDescriptorCapacity;
  DeviceWorkspace const ws = getWorkspace(p);
  if (ws.state->failed || ws.state->activeCount == 0) return;

  int32_t const activeCount = ws.state->activeCount;
  int32_t const bucket = ws.state->bucket;
  DeviceCost const cm = loadCost(p.costModelTablePtr, bucket);
  bool const useAdditiveCombine = ws.state->useAdditiveCombine;
  bool const noSplitFastPath = ws.state->noSplitFastPath;

  if (threadIdx.x == 0 && ws.state->usesCostSearch) {
    int64_t bestCost = INT64_MAX;
    int32_t bestTarget = ws.state->target;
    for (int32_t candidateIdx = 0; candidateIdx < ws.state->candidateCount; ++candidateIdx) {
      int64_t const cost = ws.candidateCosts[candidateIdx];
      int32_t const candidate = ws.candidates[candidateIdx];
      if (cost < bestCost || (cost == bestCost && candidate > bestTarget)) {
        bestCost = cost;
        bestTarget = candidate;
      }
    }
    ws.state->target = bestTarget;
  }
  __syncwarp();
  int32_t const target = ws.state->target;

  if (threadIdx.x == 0) {
    int32_t descriptorDemand = 0, splitPartialCount = 0, combineDemand = 0;
    for (int32_t req = 0; req < N; ++req) {
      int32_t requestPieces = 0;
      if (ws.blocks[req] > 0) {
        requestPieces = 1;
        if (!noSplitFastPath && ws.blocks[req] > target) {
          requestPieces = ceilDivI32(ws.blocks[req], target);
          if (requestPieces > 127) requestPieces = 127;
          if (requestPieces > P) requestPieces = P;
        }
      }
      descriptorDemand += requestPieces;
      if (requestPieces >= 2) {
        splitPartialCount += requestPieces;
        ++combineDemand;
      }
    }
    if (descriptorDemand > activeCount + P || descriptorDemand > workCapacity) {
      ws.state->failed = 1;
      failPlan(p, BalancedSchedStatus::kWorkDescriptorOverflow,
               p.selectCostModelOnDevice ? bucket : -1);
    } else {
      int32_t const partialCapacity = static_cast<int32_t>(
          (static_cast<int64_t>(N) + P < 2LL * P) ? static_cast<int64_t>(N) + P : 2LL * P);
      if (splitPartialCount > partialCapacity) {
        ws.state->failed = 1;
        failPlan(p, BalancedSchedStatus::kPartialOverflow, p.selectCostModelOnDevice ? bucket : -1);
      } else if (combineDemand > p.combineDescriptorCapacity) {
        ws.state->failed = 1;
        failPlan(p, BalancedSchedStatus::kCombineDescriptorOverflow,
                 p.selectCostModelOnDevice ? bucket : -1);
      } else if (splitPartialCount > 4096) {
        ws.state->failed = 1;
        failPlan(p, BalancedSchedStatus::kSplitInfoOverflow,
                 p.selectCostModelOnDevice ? bucket : -1);
      }
    }
  }
  __syncwarp();
  if (ws.state->failed) return;

  for (int32_t pid = threadIdx.x; pid < P; pid += warpSize) {
    ws.run[pid] = 0;
    ws.partitionCounts[pid] = 0;
  }
  __shared__ int32_t chunkWrite;
  __shared__ int32_t chunkBase;
  __shared__ int32_t descriptorCount;
  if (threadIdx.x == 0) chunkWrite = 0;
  __syncwarp();

  // Materialize the same shorts-first, longs-second placement as placeForTarget.
  for (int32_t orderIdx = 0; orderIdx < activeCount; ++orderIdx) {
    int32_t const req = ws.lptOrder[orderIdx];
    if (!noSplitFastPath && ws.blocks[req] > target) continue;
    int32_t const pid = leastLoadedPartitionWarp(ws.run, P, nullptr, 0);
    if (threadIdx.x == 0) {
      ws.run[pid] += pieceCost(ws.blocks[req], false, 0, cm);
      ws.chunkBegin[req] = chunkWrite;
      ws.chunkCount[req] = 1;
      ws.piecesPerRequest[req] = 1;
      ws.chunks[chunkWrite++] = {req, 0, ws.blocks[req], 1, 0, pid, 0};
      ++ws.partitionCounts[pid];
    }
    __syncwarp();
  }
  if (!noSplitFastPath) {
    for (int32_t orderIdx = 0; orderIdx < activeCount; ++orderIdx) {
      int32_t const req = ws.longOrder[orderIdx];
      int32_t const blockCount = ws.blocks[req];
      if (blockCount <= target) continue;
      int32_t requestPieces = ceilDivI32(blockCount, target);
      if (requestPieces > 127) requestPieces = 127;
      if (requestPieces > P) requestPieces = P;
      for (int32_t j = 0; j < requestPieces; ++j) {
        int32_t const pid = leastLoadedPartitionWarp(ws.run, P, ws.selected, j);
        if (threadIdx.x == 0) ws.selected[j] = pid;
        __syncwarp();
      }
      if (threadIdx.x == 0) {
        sortSelected(ws.selected, requestPieces);
        chunkBase = chunkWrite;
        chunkWrite += requestPieces;
        ws.chunkBegin[req] = chunkBase;
        ws.chunkCount[req] = requestPieces;
        ws.piecesPerRequest[req] = requestPieces;
      }
      __syncwarp();
      int32_t const base = blockCount / requestPieces;
      int32_t const remainder = blockCount - base * requestPieces;
      for (int32_t j = threadIdx.x; j < requestPieces; j += warpSize) {
        int32_t const pid = ws.selected[j];
        int32_t const pieceBlocks = base + (j < remainder ? 1 : 0);
        int32_t const blockOffset = j * base + (j < remainder ? j : remainder);
        ws.run[pid] +=
            pieceCost(pieceBlocks, !useAdditiveCombine && requestPieces > 1, requestPieces, cm);
        ws.chunks[chunkBase + j] = {req, blockOffset, blockOffset + pieceBlocks, requestPieces,
                                    j,   pid,         requestPieces > 1 ? 1 : 0};
        ++ws.partitionCounts[pid];
      }
      __syncwarp();
    }
  }

  if (threadIdx.x == 0) {
    int32_t splitCursor = 0;
    for (int32_t req = 0; req < N; ++req) {
      if (ws.piecesPerRequest[req] >= 2) {
        ws.splitBegin[req] = splitCursor;
        splitCursor += ws.piecesPerRequest[req];
      }
    }

    int32_t descriptorCursor = 0;
    for (int32_t pid = 0; pid < P; ++pid) {
      p.workDescriptorOffsetsPtr[pid] = descriptorCursor;
      descriptorCursor += ws.partitionCounts[pid];
    }
    p.workDescriptorOffsetsPtr[P] = descriptorCursor;
    descriptorCount = descriptorCursor;
  }
  __syncwarp();

  // Each lane emits complete partitions. Scanning requests in coordinate order reproduces
  // emitDescriptors' per-partition sort without a global device sort.
  for (int32_t pid = threadIdx.x; pid < P; pid += warpSize) {
    int32_t outputIdx = p.workDescriptorOffsetsPtr[pid];
    for (int32_t req = 0; req < N; ++req) {
      int32_t const begin = ws.chunkBegin[req];
      for (int32_t j = 0; j < ws.chunkCount[req]; ++j) {
        DeviceChunk const chunk = ws.chunks[begin + j];
        if (chunk.partitionIdx != pid) continue;
        int32_t const splitInfo = chunk.isSplit
                                      ? packSplitInfo(1, ws.splitBegin[req] + chunk.localPieceIdx,
                                                      chunk.numPieces, ws.splitBegin[req])
                                      : 0;
        p.workDescriptorPtr[outputIdx++] = {req, chunk.startBlock, chunk.endBlock, splitInfo};
      }
    }
  }
  __syncwarp();

  if (threadIdx.x == 0) {
    int32_t combineCursor = 0;
    for (int32_t req = 0; req < N; ++req) {
      if (ws.piecesPerRequest[req] >= 2)
        p.combineDescriptorPtr[combineCursor++] = {req, ws.piecesPerRequest[req],
                                                   ws.splitBegin[req], 0};
    }
    *p.numCombineDescriptorsDevicePtr = combineCursor;
    p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kDescriptorCount)] =
        descriptorCount;
    p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kTargetPieceTiles)] =
        target;
    p.planMetadataDevicePtr[static_cast<int32_t>(BalancedSchedMetadata::kCombineDescriptorCount)] =
        combineCursor;
  }
}

}  // namespace

void runBalancedSchedDevice(BalancedSchedDeviceParams const& params) {
  FLASHINFER_CHECK(params.workspaceBytes >=
                       getBalancedSchedDeviceWorkspaceSize(params.batchSize, params.numSmParts),
                   "Balanced device scheduler workspace is too small");
  if (params.useOptimizedSchedule) {
    size_t const sharedBytes =
        static_cast<size_t>(params.batchSize) * sizeof(int32_t) + sizeof(int64_t) +
        static_cast<size_t>(params.batchSize + params.numSmParts) * sizeof(int64_t) +
        static_cast<size_t>(params.numSmParts) * sizeof(int64_t);
    balancedSchedFoldPrepareScoreKernel<<<kMaxCandidates, kFoldThreads, sharedBytes,
                                          params.stream>>>(params);
    cudaError_t status = cudaGetLastError();
    FLASHINFER_CHECK(
        status == cudaSuccess,
        "balanced device fold scheduler prepare/score launch failed:", cudaGetErrorString(status));
    balancedSchedFoldEmitKernel<<<1, kFoldThreads, 0, params.stream>>>(params);
    status = cudaGetLastError();
    FLASHINFER_CHECK(status == cudaSuccess, "balanced device fold scheduler emit launch failed:",
                     cudaGetErrorString(status));
    return;
  }
  balancedSchedPrepareKernel<<<1, kFoldThreads, 0, params.stream>>>(params);
  cudaError_t status = cudaGetLastError();
  FLASHINFER_CHECK(status == cudaSuccess,
                   "balanced device scheduler prepare launch failed:", cudaGetErrorString(status));
  balancedSchedSortOrdersKernel<<<1, 128, 0, params.stream>>>(params);
  status = cudaGetLastError();
  FLASHINFER_CHECK(status == cudaSuccess,
                   "balanced device scheduler sort launch failed:", cudaGetErrorString(status));
  size_t const sharedBytes =
      static_cast<size_t>(params.numSmParts) * (sizeof(int64_t) + sizeof(int32_t));
  balancedSchedScoreKernel<<<kMaxCandidates, kScoreThreads, sharedBytes, params.stream>>>(params);
  status = cudaGetLastError();
  FLASHINFER_CHECK(status == cudaSuccess,
                   "balanced device scheduler score launch failed:", cudaGetErrorString(status));
  balancedSchedEmitKernel<<<1, 32, 0, params.stream>>>(params);
  status = cudaGetLastError();
  FLASHINFER_CHECK(status == cudaSuccess,
                   "balanced device scheduler emit launch failed:", cudaGetErrorString(status));
}

}  // namespace prims_ts
}  // namespace flashinfer
