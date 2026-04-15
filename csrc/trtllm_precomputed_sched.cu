/*
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

// Precomputed scheduler metadata generator.
// Adapted from trtllm-gen FmhaPrecomputedSched.cpp — takes host seqLens directly (no D2H sync).

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include <flashinfer/trtllm/fmha/kernelParams.h>
#include "tvm/ffi/error.h"
#include "tvm_ffi_utils.h"

namespace flashinfer {

static int hostCeilDiv(int a, int b) { return (a + b - 1) / b; }

// CPU metadata generator for the precomputed scheduler.
// Algorithm: variance-aware greedy bin-packing of KV blocks across SM partitions.
//
// Takes host-side seqLens directly (no D2H copy needed when called from vllm/flashinfer
// where seq_lens_cpu is already available).
//
// Outputs:
//   workDescriptorsOut[]: dense PrecomputedWorkDescriptor array (H2D copied to device)
//   workDescriptorOffsetsOut[]: per-partition offsets [numSmParts+1] (H2D copied to device)
static void runPrecomputedSchedFromHost(
    int32_t const* seqLensHost,               // [batchSize] on host
    PrecomputedWorkDescriptor* workDescPtrD,   // [maxDescs] on device, output
    int32_t* workDescOffsetsPtrD,              // [numSmParts+1] on device, output
    int32_t batchSize,
    int32_t numSmParts,
    int32_t blockSizeN,
    int32_t attentionWindowSize,
    int32_t tileSizeKv,
    cudaStream_t stream) {
  int const N = batchSize;
  int const P = numSmParts;

  if (N == 0) {
    cudaMemsetAsync(workDescOffsetsPtrD, 0, (P + 1) * sizeof(int32_t), stream);
    return;
  }

  // ---- Step 1: Compute numBlocks + statistics ----
  static constexpr int kFixedOverhead = 1;

  std::vector<int> numBlocks(N);
  int totalWeightedBlocks = 0;
  int maxBlocks = 0, minBlocks = INT32_MAX;

  for (int i = 0; i < N; ++i) {
    int seqLen = seqLensHost[i];
    if (attentionWindowSize > 0 && seqLen > attentionWindowSize) {
      int skippedSeqLen = std::max(0, seqLen - attentionWindowSize - 1);
      int numSkippedTokens = (skippedSeqLen / tileSizeKv) * tileSizeKv;
      seqLen = seqLen - numSkippedTokens;
    }
    numBlocks[i] = (seqLen > 0) ? hostCeilDiv(seqLen, blockSizeN) : 1;
    totalWeightedBlocks += numBlocks[i] + kFixedOverhead;
    maxBlocks = std::max(maxBlocks, numBlocks[i]);
    minBlocks = std::min(minBlocks, numBlocks[i]);
  }

  // ---- Step 1.5: Compute effective partition count ----
  // Port of espUpperBound from fmhaKernels.cuh computeCtaAndClusterConfig().
  // Prevents over-splitting when total work is small relative to numSmParts.
  // The kernel always launches P (= rawNumSmParts) CTAs for stable grid Z;
  // partitions P_active..P-1 receive empty descriptor ranges and exit immediately.
  static constexpr int kMinBlocksPerPartition = 6;
  int64_t totalWeightForEsp =
      static_cast<int64_t>(N) * (maxBlocks + kMinBlocksPerPartition);
  int espUpperBound =
      std::max(1, static_cast<int>(totalWeightForEsp / kMinBlocksPerPartition));
  int const P_active = std::min(P, espUpperBound);

  // ---- Step 2: Payload selection (uses P_active for work distribution) ----
  int const basePayload = hostCeilDiv(totalWeightedBlocks, P_active) + kFixedOverhead;
  int const maxPiecesPayload = hostCeilDiv(maxBlocks, kMaxNumPieces) + kFixedOverhead;
  int const maxCost = maxBlocks + kFixedOverhead;
  int const minCost = minBlocks + kFixedOverhead;

  bool const isNearUniform = (maxCost <= 2 * minCost);
  int payload;
  if (isNearUniform) {
    int const reqsPerPart = hostCeilDiv(N, P_active);
    int const avgCost = hostCeilDiv(totalWeightedBlocks, N);
    payload = reqsPerPart * avgCost + kFixedOverhead;
  } else {
    payload = std::max(basePayload, maxPiecesPayload);
  }

  // ---- Step 3: Two-pass greedy scan ----
  struct PartBound {
    int beginReq, endReq;
    int beginBlock, endBlock;
    int beginSplitIdx;
  };
  std::vector<PartBound> partBounds(P_active);
  std::vector<int> numSplitsPrefix(N + 1, 0);

  // Pass 1: greedy scan → partition boundaries + numSplitsPrefix.
  {
    int curReq = 0, curBlock = 0, curSplitIdx = 0, cumSplits = 0;

    for (int p = 0; p < P_active; ++p) {
      PartBound& pb = partBounds[p];
      pb.beginReq = curReq;
      pb.beginBlock = curBlock;
      pb.beginSplitIdx = curSplitIdx;

      int remainPayload = payload;

      while (curReq < N) {
        int remainBlocks = numBlocks[curReq] - curBlock;

        if (remainPayload >= remainBlocks + kFixedOverhead) {
          cumSplits += (curSplitIdx > 0) ? (curSplitIdx + 1) : 0;
          numSplitsPrefix[curReq + 1] = cumSplits;
          remainPayload -= remainBlocks + kFixedOverhead;
          curReq++;
          curBlock = 0;
          curSplitIdx = 0;
        } else {
          int blocksFit = remainPayload - kFixedOverhead;
          if (blocksFit > 0) {
            blocksFit = std::min(blocksFit, remainBlocks);
            curBlock += blocksFit;
            curSplitIdx++;
          }
          break;
        }
      }

      if (curReq >= N && curReq == pb.beginReq && curBlock == 0) {
        pb.endReq = pb.beginReq - 1;
        pb.endBlock = 0;
      } else if (curBlock > 0 && curReq < N) {
        pb.endReq = curReq;
        pb.endBlock = curBlock;
      } else if (curReq > pb.beginReq) {
        pb.endReq = curReq - 1;
        pb.endBlock = numBlocks[curReq - 1];
      } else {
        pb.endReq = pb.beginReq;
        pb.endBlock = pb.beginBlock;
      }
    }
  }

  // Pass 2: generate descriptors (numSplitsPrefix is now complete).
  int const maxDescs = N + P;
  std::vector<PrecomputedWorkDescriptor> descsHost;
  descsHost.reserve(maxDescs);
  std::vector<int32_t> offsetsHost(P + 1, 0);

  for (int p = 0; p < P_active; ++p) {
    offsetsHost[p] = static_cast<int32_t>(descsHost.size());

    PartBound const& pb = partBounds[p];
    if (pb.beginReq > pb.endReq)
      continue;

    for (int req = pb.beginReq; req <= pb.endReq; ++req) {
      int startBlock, endBlock;
      bool isSplit;
      int localPieceIdx = 0;

      if (req == pb.beginReq && req == pb.endReq) {
        startBlock = pb.beginBlock;
        endBlock = pb.endBlock;
        isSplit = (pb.beginBlock > 0) || (pb.endBlock < numBlocks[req]);
        localPieceIdx = pb.beginSplitIdx;
      } else if (req == pb.beginReq) {
        startBlock = pb.beginBlock;
        endBlock = numBlocks[req];
        isSplit = (pb.beginBlock > 0);
        localPieceIdx = pb.beginSplitIdx;
      } else if (req == pb.endReq) {
        startBlock = 0;
        endBlock = pb.endBlock;
        isSplit = (pb.endBlock < numBlocks[req]);
        localPieceIdx = 0;
      } else {
        startBlock = 0;
        endBlock = numBlocks[req];
        isSplit = false;
        localPieceIdx = 0;
      }

      int32_t si = 0;
      if (isSplit) {
        int splitBeginIdx = numSplitsPrefix[req];
        int numPieces = numSplitsPrefix[req + 1] - splitBeginIdx;
        int splitGlobalIdx = splitBeginIdx + localPieceIdx;
        si = packSplitInfo(1, splitGlobalIdx, numPieces, splitBeginIdx);
      }

      PrecomputedWorkDescriptor desc;
      desc.reqIdx = req;
      desc.startBlock = startBlock;
      desc.endBlock = endBlock;
      desc.splitInfo = si;
      descsHost.push_back(desc);
    }
  }
  // Fill offsets for inactive partitions (P_active..P): all point to end of descriptors.
  int32_t totalDescsVal = static_cast<int32_t>(descsHost.size());
  for (int p = P_active; p <= P; ++p) {
    offsetsHost[p] = totalDescsVal;
  }

  // ---- Step 4: H2D copy results (async on stream, no sync needed) ----
  int totalDescs = static_cast<int>(descsHost.size());
  cudaMemcpyAsync(workDescPtrD, descsHost.data(),
                  totalDescs * sizeof(PrecomputedWorkDescriptor), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(workDescOffsetsPtrD, offsetsHost.data(), (P + 1) * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream);
}

// TVM-FFI entry point for precomputed metadata computation.
void trtllm_compute_precomputed_metadata(TensorView seq_lens_cpu, TensorView work_descriptors_out,
                                         TensorView work_descriptor_offsets_out,
                                         int64_t batch_size, int64_t block_size_n,
                                         int64_t num_sm_parts, int64_t attention_window_size,
                                         int64_t tile_size_kv) {
  TVM_FFI_ICHECK_EQ(seq_lens_cpu.device().device_type, kDLCPU)
      << "seq_lens_cpu must be a CPU tensor";
  TVM_FFI_ICHECK_EQ(work_descriptors_out.device().device_type, kDLCUDA)
      << "work_descriptors_out must be a CUDA tensor";

  auto stream = get_stream(work_descriptors_out.device());

  runPrecomputedSchedFromHost(
      static_cast<int32_t const*>(seq_lens_cpu.data_ptr()),
      reinterpret_cast<PrecomputedWorkDescriptor*>(work_descriptors_out.data_ptr()),
      static_cast<int32_t*>(work_descriptor_offsets_out.data_ptr()),
      static_cast<int32_t>(batch_size), static_cast<int32_t>(num_sm_parts),
      static_cast<int32_t>(block_size_n), static_cast<int32_t>(attention_window_size),
      static_cast<int32_t>(tile_size_kv), stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(trtllm_compute_precomputed_metadata,
                               trtllm_compute_precomputed_metadata);

}  // namespace flashinfer
