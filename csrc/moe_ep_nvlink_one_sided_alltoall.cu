/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// TVM-FFI bindings of the NVLink one-sided MoE all-to-all used by
// flashinfer.moe_ep.NVLinkOneSidedAlltoAll. Collective steps (workspace
// barriers and the CFT logical-endpoint exchange) are driven from Python over
// the EP group's communicator; these ops only touch local state.

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "tensorrt_llm/common/tllmDataType.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllCftManager.h"
#include "tensorrt_llm/kernels/moe/communication/moeAlltoAllKernels.h"
#include "tensorrt_llm/thop/moe/communication/moeAlltoAllMeta.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;
using tvm::ffi::String;
using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
using tvm::ffi::Tuple;

namespace {

namespace tl = tensorrt_llm::kernels::moe_comm;
namespace meta = tensorrt_llm::torch_ext::moe_comm;

constexpr size_t kCachelineAlignment = 128;
constexpr int64_t kMaxTimeoutSec = 24 * 60 * 60;

inline size_t alignOffset(size_t offset, size_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

int64_t timeoutCycles(int64_t timeoutSec) {
  TVM_FFI_ICHECK(timeoutSec > 0 && timeoutSec <= kMaxTimeoutSec)
      << "MoE all-to-all timeout must be in 1.." << kMaxTimeoutSec << " seconds";
  return timeoutSec * tl::kAssumedClockHz;
}

meta::MoeA2AWorkspaceLayout readWorkspaceLayout(TensorView metainfo) {
  CHECK_CPU(metainfo);
  CHECK_CONTIGUOUS(metainfo);
  CHECK_INPUT_TYPE(metainfo, dl_int64);
  TVM_FFI_ICHECK(metainfo.ndim() == 1 && metainfo.size(0) == meta::NUM_METAINFO_FIELDS)
      << "metainfo must contain the complete MoE A2A workspace layout";
  meta::MoeA2AWorkspaceLayout layout{};
  auto const* src = static_cast<int64_t const*>(metainfo.data_ptr());
  std::copy(src, src + meta::NUM_METAINFO_FIELDS, layout.begin());
  return layout;
}

uint8_t* rankWorkspacePtr(TensorView workspace, int64_t epRank) {
  return static_cast<uint8_t*>(workspace.data_ptr()) + epRank * workspace.stride(0);
}

inline bool hasActiveRankMask(Optional<TensorView> const& maskTensor) {
  return maskTensor.has_value();
}

// Resolve a provided rank-mask tensor into a fixed-width uint64 array. On failure
// (wrong dtype / device / shape), throw at the Python op boundary rather than launch.
inline void resolveActiveRankMask(Optional<TensorView> const& maskTensor, int64_t epRank,
                                  uint64_t (&out)[tl::kRankMaskWords]) {
  TVM_FFI_ICHECK(epRank >= 0 && epRank < tl::kMaxRanks)
      << "epRank must be in the range [0, " << tl::kMaxRanks << ") for active_rank_mask";
  TVM_FFI_ICHECK(hasActiveRankMask(maskTensor)) << "active_rank_mask must be defined";
  TensorView const& t = maskTensor.value();
  CHECK_CPU(t);
  CHECK_CONTIGUOUS(t);
  CHECK_INPUT_TYPE(t, dl_uint64);
  TVM_FFI_ICHECK_EQ(t.ndim(), 1) << "active_rank_mask must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(t.size(0), tl::kRankMaskWords)
      << "active_rank_mask must have exactly " << tl::kRankMaskWords << " uint64 elements";
  auto const* src = static_cast<uint64_t const*>(t.data_ptr());
  for (int w = 0; w < tl::kRankMaskWords; ++w) {
    out[w] = src[w];
  }
  // Local rank's bit must be set; otherwise the kernel would be running on a "dead" rank.
  TVM_FFI_ICHECK((out[epRank >> 6] >> (epRank & 63)) & 1ULL)
      << "active_rank_mask must mark the local ep_rank (" << epRank << ") as active";
}

void resolveRankMask(bool enableRankMask, Optional<TensorView> const& activeRankMask,
                     int64_t epRank, bool& enabledOut, uint64_t (&maskOut)[tl::kRankMaskWords]) {
  enabledOut = enableRankMask;
  if (enableRankMask) {
    resolveActiveRankMask(activeRankMask, epRank, maskOut);
  } else {
    TVM_FFI_ICHECK(!hasActiveRankMask(activeRankMask))
        << "active_rank_mask requires enable_rank_mask=True";
  }
}

// All offsets and capacities are per rank. Payload capacities are padded so
// the following phase's control buffers remain aligned.
meta::MoeA2AWorkspaceLayout calculateWorkspaceLayout(int64_t epSize, int64_t maxNumTokens,
                                                     int64_t topK, int64_t dispatchBytes,
                                                     int64_t combineInputBytes,
                                                     int64_t combineRecvBytes,
                                                     int64_t eplbStatsNumExperts, bool canUseCft) {
  using namespace meta;
  TVM_FFI_ICHECK(epSize > 0 && epSize <= tl::kMaxRanks) << "Invalid EP size: " << epSize;
  TVM_FFI_ICHECK(maxNumTokens > 0 && maxNumTokens <= std::numeric_limits<int32_t>::max())
      << "Invalid allocation-time token capacity: " << maxNumTokens;
  TVM_FFI_ICHECK(topK > 0 && topK <= tl::kMaxTopK) << "Invalid top_k: " << topK;
  TVM_FFI_ICHECK(eplbStatsNumExperts >= 0 &&
                 eplbStatsNumExperts <= std::numeric_limits<int32_t>::max())
      << "Invalid EPLB expert capacity: " << eplbStatsNumExperts;
  TVM_FFI_ICHECK(canUseCft || combineRecvBytes == 0)
      << "A combine receive payload requires CFT support";

  auto alignedBytes = [](int64_t bytes) {
    TVM_FFI_ICHECK(bytes >= 0 && bytes <= std::numeric_limits<int64_t>::max() - kWorkspaceAlignment)
        << "Invalid workspace payload capacity: " << bytes;
    return static_cast<int64_t>(alignOffset(bytes, kWorkspaceAlignment));
  };
  dispatchBytes = alignedBytes(dispatchBytes);
  combineInputBytes = alignedBytes(combineInputBytes);
  combineRecvBytes = alignedBytes(combineRecvBytes);

  MoeA2AWorkspaceLayout layout{};
  int64_t offset = 0;
  auto reserve = [&](MoeA2AMetaInfoIndex index, int64_t bytes,
                     int64_t alignment = sizeof(int32_t)) {
    TVM_FFI_ICHECK(offset <= std::numeric_limits<int64_t>::max() - alignment)
        << "Workspace layout alignment overflows int64";
    offset = static_cast<int64_t>(alignOffset(offset, alignment));
    TVM_FFI_ICHECK(bytes >= 0 && bytes <= std::numeric_limits<int64_t>::max() - offset)
        << "Workspace layout size overflows int64";
    layout[index] = offset;
    offset += bytes;
  };
  int64_t const rankCountsBytes = epSize * sizeof(int32_t);
  int64_t const combineSlots = epSize * maxNumTokens;

  reserve(FLAG_VAL_OFFSET_INDEX, sizeof(uint32_t));

  // Dispatch writes these counters and routes; combine reuses the completed routes.
  reserve(LOCAL_TOKEN_COUNTER_OFFSET_INDEX, sizeof(int32_t));
  reserve(SEND_COUNTERS_OFFSET_INDEX, rankCountsBytes);
  reserve(RECV_COUNTERS_OFFSET_INDEX, 2 * rankCountsBytes);
  reserve(DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX, rankCountsBytes, kCachelineAlignment);
  if (canUseCft) {
    reserve(DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX, epSize * tl::kCftCounterStride,
            tl::kCftCounterStride);
    reserve(DISPATCH_COUNTER_BASELINE_OFFSET_INDEX, epSize * sizeof(uint64_t), kCachelineAlignment);
  }
  reserve(TOPK_TARGET_RANKS_OFFSET_INDEX, maxNumTokens * topK * sizeof(int32_t),
          kCachelineAlignment);
  reserve(TOPK_TARGET_INDICES_OFFSET_INDEX, maxNumTokens * topK * sizeof(int32_t),
          kCachelineAlignment);
  reserve(EPLB_GATHERED_STATS_OFFSET_INDEX, epSize * eplbStatsNumExperts * sizeof(int32_t),
          kCachelineAlignment);
  reserve(DISPATCH_PAYLOAD_OFFSET_INDEX, dispatchBytes, kWorkspaceAlignment);
  layout[DISPATCH_PAYLOAD_SIZE_INDEX] = dispatchBytes;

  // Fence pulls from combine input; CFT pushes into the separate receive inbox.
  reserve(COMBINE_COMPLETION_FLAGS_OFFSET_INDEX, rankCountsBytes, kCachelineAlignment);
  if (canUseCft) {
    reserve(COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX, combineSlots * tl::kCftCounterStride,
            tl::kCftCounterStride);
    reserve(COMBINE_COUNTER_BASELINE_OFFSET_INDEX, combineSlots * sizeof(uint64_t),
            kCachelineAlignment);
  }
  reserve(COMBINE_INPUT_OFFSET_INDEX, combineInputBytes, kWorkspaceAlignment);
  layout[COMBINE_INPUT_SIZE_INDEX] = combineInputBytes;
  if (canUseCft) {
    reserve(COMBINE_RECV_OFFSET_INDEX, combineRecvBytes, kWorkspaceAlignment);
    layout[COMBINE_RECV_SIZE_INDEX] = combineRecvBytes;
  }

  layout[MAX_NUM_TOKENS_INDEX] = maxNumTokens;
  layout[TOP_K_INDEX] = topK;
  layout[EP_SIZE_INDEX] = epSize;
  layout[EPLB_STATS_NUM_EXPERTS_INDEX] = eplbStatsNumExperts;
  layout[CFT_ENABLED_INDEX] = canUseCft;
  layout[WORKSPACE_SIZE_INDEX] = offset;
  return layout;
}

// Write the workspace layout into metainfo, a CPU int64 tensor of NUM_METAINFO_FIELDS.
void moeA2AGetWorkspaceLayoutOp(int64_t epSize, int64_t maxNumTokens, int64_t topK,
                                int64_t dispatchBytes, int64_t combineInputBytes,
                                int64_t combineRecvBytes, int64_t eplbStatsNumExperts,
                                bool canUseCft, TensorView metainfo) {
  CHECK_CPU(metainfo);
  CHECK_CONTIGUOUS(metainfo);
  CHECK_INPUT_TYPE(metainfo, dl_int64);
  TVM_FFI_ICHECK(metainfo.ndim() == 1 && metainfo.size(0) == meta::NUM_METAINFO_FIELDS)
      << "metainfo must have NUM_METAINFO_FIELDS elements";
  auto const layout =
      calculateWorkspaceLayout(epSize, maxNumTokens, topK, dispatchBytes, combineInputBytes,
                               combineRecvBytes, eplbStatsNumExperts, canUseCft);
  std::copy(layout.begin(), layout.end(), static_cast<int64_t*>(metainfo.data_ptr()));
}

// Reset this rank's control state after allocation. The caller must barrier the EP
// group before any peer launches against the workspace.
void moeA2AInitializeOp(TensorView workspace, TensorView metainfo, int64_t epRank, int64_t epSize) {
  using namespace meta;
  CHECK_CUDA(workspace);
  CHECK_INPUT_TYPE(workspace, dl_uint8);
  TVM_FFI_ICHECK(workspace.ndim() == 2 && workspace.size(0) == epSize && workspace.stride(1) == 1)
      << "workspace must have shape [ep_size, bytes_per_rank] with contiguous rank slices";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "Invalid EP rank";
  auto const layout = readWorkspaceLayout(metainfo);
  TVM_FFI_ICHECK_EQ(layout[EP_SIZE_INDEX], epSize) << "Workspace layout EP size mismatch";
  auto const expected =
      calculateWorkspaceLayout(epSize, layout[MAX_NUM_TOKENS_INDEX], layout[TOP_K_INDEX],
                               layout[DISPATCH_PAYLOAD_SIZE_INDEX],
                               layout[COMBINE_INPUT_SIZE_INDEX], layout[COMBINE_RECV_SIZE_INDEX],
                               layout[EPLB_STATS_NUM_EXPERTS_INDEX], layout[CFT_ENABLED_INDEX]);
  TVM_FFI_ICHECK(layout == expected) << "Workspace layout metadata is inconsistent";
  TVM_FFI_ICHECK(layout[DISPATCH_PAYLOAD_SIZE_INDEX] > 0 && layout[COMBINE_INPUT_SIZE_INDEX] > 0)
      << "Dispatch and combine input payload capacities must be positive";
  TVM_FFI_ICHECK(!layout[CFT_ENABLED_INDEX] || layout[COMBINE_RECV_SIZE_INDEX] > 0)
      << "CFT requires a combine receive payload capacity";
  TVM_FFI_ICHECK(workspace.size(1) >= layout[WORKSPACE_SIZE_INDEX])
      << "Workspace needs " << layout[WORKSPACE_SIZE_INDEX] << " bytes per rank, got "
      << workspace.size(1);

  auto stream = get_current_stream();
  uint8_t* rankWorkspace = rankWorkspacePtr(workspace, epRank);
  TVM_FFI_ICHECK(cudaMemsetAsync(rankWorkspace, 0, layout[WORKSPACE_SIZE_INDEX], stream) ==
                 cudaSuccess)
      << "cudaMemsetAsync failed";
  TVM_FFI_ICHECK(cudaMemsetAsync(rankWorkspace + layout[RECV_COUNTERS_OFFSET_INDEX], 0xFF,
                                 2 * static_cast<size_t>(epSize) * sizeof(int32_t),
                                 stream) == cudaSuccess)
      << "cudaMemsetAsync failed";
  auto err = cudaStreamSynchronize(stream);
  TVM_FFI_ICHECK(err == cudaSuccess) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);
}

// ============================================================================
// CFT Handle-Based Counted Writes Initialization
// ============================================================================

// One CFT binding per process, released before its backing workspace is freed.
std::unique_ptr<tl::CftLeManager> g_cft_manager;

int64_t moeA2ACftHandleBytesOp() { return static_cast<int64_t>(tl::CftLeManager::handleBytes()); }

// Create this rank's logical endpoint bound to its workspace slice and export it
// into handleOut. Returns false when the driver or device cannot provide one; the
// EP group must then agree to fall back before exchanging endpoints.
//
// Args:
//   workspaceMemHandle: CUmemGenericAllocationHandle (as int64) backing this rank's slice
//   workspaceSizePerRank: size of the workspace per rank in bytes
//   handleOut: CPU uint8 tensor of moe_a2a_cft_handle_bytes() bytes
bool moeA2ACftCreateEndpointOp(TensorView workspace, int64_t workspaceMemHandle,
                               int64_t workspaceSizePerRank, int64_t epRank, int64_t epSize,
                               TensorView handleOut) {
  CHECK_CUDA(workspace);
  CHECK_INPUT_TYPE(workspace, dl_uint8);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2)
      << "workspace must be a 2D tensor of shape [epSize, sizePerRank]";
  TVM_FFI_ICHECK_EQ(workspace.size(0), epSize) << "workspace first dimension must equal epSize";
  TVM_FFI_ICHECK(epSize > 0 && epSize <= tl::kMaxRanks)
      << "epSize must be in the range (0, " << tl::kMaxRanks << "]";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "epRank must be in the range [0, epSize)";
  TVM_FFI_ICHECK(workspaceSizePerRank > 0 && workspaceSizePerRank <= workspace.stride(0))
      << "workspaceSizePerRank must be in the range (0, workspace.stride(0)]";
  CHECK_CPU(handleOut);
  CHECK_CONTIGUOUS(handleOut);
  CHECK_INPUT_TYPE(handleOut, dl_uint8);
  TVM_FFI_ICHECK_EQ(handleOut.numel(), static_cast<int64_t>(tl::CftLeManager::handleBytes()))
      << "handle_out must hold exactly one exported endpoint handle";
  TVM_FFI_ICHECK(!g_cft_manager)
      << "CFT logical endpoints are already bound to a workspace. Only one workspace per "
         "process may use CFT counted writes.";

  auto manager = std::make_unique<tl::CftLeManager>();
  if (!manager->loadApis()) {
    return false;
  }
  int localDevIdx = -1;
  TVM_FFI_ICHECK(cudaGetDevice(&localDevIdx) == cudaSuccess)
      << "cudaGetDevice failed during CFT initialization";
  auto const workspaceRankPtr = reinterpret_cast<CUdeviceptr>(rankWorkspacePtr(workspace, epRank));
  if (!manager->createEndpointExternal(
          localDevIdx, static_cast<CUmemGenericAllocationHandle>(workspaceMemHandle),
          workspaceRankPtr, static_cast<size_t>(workspaceSizePerRank), static_cast<int>(epRank),
          static_cast<int>(epSize))) {
    return false;
  }
  if (!manager->exportEndpoint(handleOut.data_ptr())) {
    return false;
  }
  g_cft_manager = std::move(manager);
  return true;
}

// Import every rank's exported endpoint (allHandles: CPU uint8 [epSize, handle bytes],
// ordered by rank). Returns false when an import fails or the endpoints never become ready.
bool moeA2ACftImportEndpointsOp(TensorView workspace, int64_t epRank, TensorView allHandles) {
  CHECK_CPU(allHandles);
  CHECK_CONTIGUOUS(allHandles);
  CHECK_INPUT_TYPE(allHandles, dl_uint8);
  TVM_FFI_ICHECK(g_cft_manager) << "moe_a2a_cft_create_endpoint must succeed first";
  TVM_FFI_ICHECK(g_cft_manager->getLocalBackingPtr() ==
                 reinterpret_cast<CUdeviceptr>(rankWorkspacePtr(workspace, epRank)))
      << "CFT endpoints were created for a different workspace";
  TVM_FFI_ICHECK_EQ(allHandles.numel(),
                    workspace.size(0) * static_cast<int64_t>(tl::CftLeManager::handleBytes()))
      << "all_handles must hold one exported endpoint handle per rank";
  if (!g_cft_manager->importEndpoints(allHandles.data_ptr())) {
    return false;
  }
  auto err = cudaDeviceSynchronize();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "cudaDeviceSynchronize after CFT initialization failed: " << cudaGetErrorString(err);
  return true;
}

// All ranks must finish using the workspace before releasing their local binding.
void moeA2ACftDestroyOp(TensorView workspace, int64_t epRank) {
  if (!g_cft_manager) {
    return;
  }
  TVM_FFI_ICHECK(g_cft_manager->getLocalBackingPtr() ==
                 reinterpret_cast<CUdeviceptr>(rankWorkspacePtr(workspace, epRank)))
      << "Cannot destroy CFT endpoints bound to a different workspace";
  TVM_FFI_ICHECK(cudaDeviceSynchronize() == cudaSuccess)
      << "CUDA synchronization failed before CFT endpoint release";
  g_cft_manager.reset();
}

// MoE All-to-All Dispatch Operation
// Dispatches tokens and their payloads to the ranks that own their experts.
//
// Inputs:
//   - tokenSelectedExperts: [local_num_tokens, top_k] int32 expert indices
//   - inputPayloads: tensors of shape [local_num_tokens, elements_per_token] to dispatch
//   - workspace: [ep_size, size_per_rank] uint8 symmetric workspace
//   - metainfo: CPU int64 workspace layout from moe_a2a_get_workspace_layout
//   - runtimeMaxTokensPerRank: maximum local batch over the EP group for this round
//   - expertIdPayloadIndex: index of the int32 expert-id payload whose padding rows the CFT
//     path fills with invalidTokenExpertId, or -1 to leave sanitization to the caller
//
// Returns (recv_offsets, combine_payload_offset, eplb_gathered_stats_offset), all byte
// offsets from this rank's workspace base. recv_offsets[i] addresses payload i's
// [ep_size, runtimeMaxTokensPerRank, elements_per_token] receive buffer;
// eplb_gathered_stats_offset is -1 when eplbLocalStats is not given.
//
// Note: token_selected_experts is used for routing but is NOT automatically included as a
// payload. To dispatch it, include it explicitly in inputPayloads.
Tuple<Array<int64_t>, int64_t, int64_t> moeA2ADispatchOp(
    TensorView tokenSelectedExperts, Array<Tensor> inputPayloads, TensorView workspace,
    TensorView metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize,
    int64_t topK, int64_t numExperts, Optional<TensorView> eplbLocalStats, bool useCftCountedWrites,
    int64_t expertIdPayloadIndex, int64_t invalidTokenExpertId, bool enableRankMask,
    Optional<TensorView> activeRankMask, int64_t timeoutSec, bool enablePdl) {
  using namespace meta;

  // Validate inputs
  CHECK_INPUT(tokenSelectedExperts);
  CHECK_INPUT_TYPE(tokenSelectedExperts, dl_int32);
  TVM_FFI_ICHECK_EQ(tokenSelectedExperts.ndim(), 2) << "tokenSelectedExperts must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(tokenSelectedExperts.size(1), topK)
      << "tokenSelectedExperts must have topK columns";

  auto const offsets = readWorkspaceLayout(metainfo);

  int64_t const localNumTokens = tokenSelectedExperts.size(0);
  int const numPayloads = static_cast<int>(inputPayloads.size());
  TVM_FFI_ICHECK(runtimeMaxTokensPerRank > 0) << "runtimeMaxTokensPerRank must be positive";
  TVM_FFI_ICHECK(runtimeMaxTokensPerRank <= offsets[MAX_NUM_TOKENS_INDEX])
      << "runtimeMaxTokensPerRank exceeds the allocation-time token capacity";
  TVM_FFI_ICHECK(epSize > 0 && epSize <= tl::kMaxRanks)
      << "epSize must be in the range (0, " << tl::kMaxRanks << "]";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "epRank must be in the range [0, epSize)";
  TVM_FFI_ICHECK(topK > 0 && topK <= tl::kMaxTopK) << "topK must be in the range (0, kMaxTopK]";
  TVM_FFI_ICHECK(numPayloads > 0) << "inputPayloads must not be empty";
  TVM_FFI_ICHECK(numPayloads <= tl::kMaxPayloads) << "Too many input payloads";
  TVM_FFI_ICHECK(numExperts >= epSize) << "numExperts must be greater than or equal to epSize";
  // numExperts does not need to be divisible by epSize: the kernel performs
  // ceil/floor contiguous partitioning so ranks [0, numExperts % epSize)
  // own (numExperts / epSize + 1) experts and the rest own (numExperts / epSize).

  bool const enableEplb = eplbLocalStats.has_value();
  int64_t eplbStatsNumExperts = 0;
  if (enableEplb) {
    TensorView const& stats = eplbLocalStats.value();
    CHECK_INPUT(stats);
    CHECK_INPUT_TYPE(stats, dl_int32);
    TVM_FFI_ICHECK_EQ(stats.ndim(), 1) << "eplb_local_stats must be a 1D tensor";
    eplbStatsNumExperts = stats.size(0);
    TVM_FFI_ICHECK(eplbStatsNumExperts > 0) << "eplb_local_stats must not be empty";
    TVM_FFI_ICHECK(eplbStatsNumExperts <= numExperts)
        << "eplb_local_stats size must be <= numExperts (slots)";
  }

  // Record the cacheline aligned start offset for each payload's recv buffer.
  // 1. We assume the base workspace ptr of each rank is aligned (checked in this OP)
  // 2. offsets[DISPATCH_PAYLOAD_OFFSET_INDEX] is aligned (fixed by the workspace layout)
  // 3. We align the currentOffset during update.
  // In this way, it is guaranteed that the recv buffer is (over-)aligned, sufficient for 128bit
  // vectorized ld/st.
  std::vector<int> payloadElementSizes;
  std::vector<int> payloadElementsPerToken;
  std::vector<size_t> payloadRecvBufferOffsets;
  size_t currentOffset = static_cast<size_t>(offsets[DISPATCH_PAYLOAD_OFFSET_INDEX]);
  for (Tensor const& payload : inputPayloads) {
    CHECK_INPUT(payload);
    CHECK_DEVICE(payload, tokenSelectedExperts);
    TVM_FFI_ICHECK_EQ(payload.ndim(), 2) << "payload must be a 2D tensor";
    TVM_FFI_ICHECK_EQ(payload.size(0), localNumTokens)
        << "payload must have the same first dimension as tokenSelectedExperts";
    // Unlike recv buffer for payloads, payload itself is not allocated by us and we cannot
    // control its alignment. We only make sure the payload start offset is 16-byte aligned,
    // while the actual vectorized ld/st width is dynamically determined based on bytes per
    // token of this payload.
    TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0)
        << "payload must be 16-byte aligned";

    int const elementsPerToken = static_cast<int>(payload.size(1));
    int const elementSize = static_cast<int>(get_element_size(payload));
    // Each payload buffer stores data from ALL ranks
    int64_t const bytesPerPayload =
        epSize * runtimeMaxTokensPerRank * elementsPerToken * elementSize;

    payloadElementSizes.push_back(elementSize);
    payloadElementsPerToken.push_back(elementsPerToken);
    payloadRecvBufferOffsets.push_back(currentOffset);

    // Update offset and align to cacheline boundary for the next payload recv buffer.
    currentOffset = alignOffset(currentOffset + bytesPerPayload, kCachelineAlignment);
  }

  bool const sanitizeExpertIds = expertIdPayloadIndex >= 0;
  if (sanitizeExpertIds) {
    TVM_FFI_ICHECK(expertIdPayloadIndex < numPayloads) << "expert_id_payload_index out of range";
    Tensor const& expertIdPayload = inputPayloads[expertIdPayloadIndex];
    CHECK_INPUT_TYPE(expertIdPayload, dl_int32);
    TVM_FFI_ICHECK_EQ(expertIdPayload.size(1), topK) << "expert-id payload must have topK columns";
  }

  CHECK_CUDA(workspace);
  CHECK_INPUT_TYPE(workspace, dl_uint8);
  // Don't check contiguous - MnnvlMemory creates strided tensors for multi-GPU
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2)
      << "workspace must be a 2D tensor of shape [epSize, sizePerRank]";
  TVM_FFI_ICHECK_EQ(workspace.size(0), epSize) << "workspace first dimension must equal epSize";
  TVM_FFI_ICHECK(epSize == offsets[EP_SIZE_INDEX] && topK == offsets[TOP_K_INDEX])
      << "Dispatch EP size/top_k differs from its workspace layout";
  TVM_FFI_ICHECK(localNumTokens <= runtimeMaxTokensPerRank)
      << "Local token count exceeds the runtime capacity";
  TVM_FFI_ICHECK(eplbStatsNumExperts <= offsets[EPLB_STATS_NUM_EXPERTS_INDEX])
      << "EPLB statistics exceed their workspace capacity";
  TVM_FFI_ICHECK(!useCftCountedWrites || offsets[CFT_ENABLED_INDEX])
      << "Workspace was allocated without CFT support";
  TVM_FFI_ICHECK(workspace.size(1) >= offsets[WORKSPACE_SIZE_INDEX])
      << "Workspace is smaller than its layout";
  int64_t const payloadCapacity = offsets[DISPATCH_PAYLOAD_SIZE_INDEX];
  TVM_FFI_ICHECK(currentOffset <=
                 static_cast<size_t>(offsets[DISPATCH_PAYLOAD_OFFSET_INDEX] + payloadCapacity))
      << "Dispatch payload exceeds its workspace capacity: need "
      << currentOffset - offsets[DISPATCH_PAYLOAD_OFFSET_INDEX] << " bytes, capacity "
      << payloadCapacity;

  // Get base workspace pointer
  uint8_t* workspacePtr = static_cast<uint8_t*>(workspace.data_ptr());
  uint8_t* rankWorkSpacePtr = rankWorkspacePtr(workspace, epRank);
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(rankWorkSpacePtr) % kCachelineAlignment == 0)
      << "rankWorkSpacePtr must be " << kCachelineAlignment << "-byte aligned";

  // Setup dispatch parameters
  tl::MoeA2ADispatchParams params{};
  params.ep_size = static_cast<int>(epSize);
  params.ep_rank = static_cast<int>(epRank);
  params.num_experts = static_cast<int>(numExperts);
  params.local_num_tokens = static_cast<int>(localNumTokens);
  params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
  params.top_k = static_cast<int>(topK);
  params.enable_eplb = enableEplb;
  params.eplb_stats_num_experts = static_cast<int>(eplbStatsNumExperts);

  params.token_selected_experts = static_cast<int32_t const*>(tokenSelectedExperts.data_ptr());

  params.num_payloads = numPayloads;
  for (int i = 0; i < numPayloads; i++) {
    params.payloads[i].src_data = inputPayloads[i].data_ptr();
    params.payloads[i].element_size = payloadElementSizes[i];
    params.payloads[i].elements_per_token = payloadElementsPerToken[i];
  }

  params.flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);
  params.local_token_counter =
      reinterpret_cast<int*>(rankWorkSpacePtr + offsets[LOCAL_TOKEN_COUNTER_OFFSET_INDEX]);
  params.send_counters =
      reinterpret_cast<int*>(rankWorkSpacePtr + offsets[SEND_COUNTERS_OFFSET_INDEX]);
  params.topk_target_ranks =
      reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_RANKS_OFFSET_INDEX]);
  params.topk_target_indices =
      reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_INDICES_OFFSET_INDEX]);

  for (int targetRank = 0; targetRank < epSize; targetRank++) {
    uint8_t* targetWorkSpacePtr = workspacePtr + targetRank * workspace.stride(0);

    params.recv_counters[targetRank] =
        reinterpret_cast<int*>(targetWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);
    params.completion_flags[targetRank] = reinterpret_cast<uint32_t*>(
        targetWorkSpacePtr + offsets[DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX]);
    params.le_dispatch_counters[targetRank] = reinterpret_cast<uint64_t*>(
        targetWorkSpacePtr + offsets[DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX]);
    params.eplb_gathered_stats[targetRank] =
        enableEplb
            ? reinterpret_cast<int*>(targetWorkSpacePtr + offsets[EPLB_GATHERED_STATS_OFFSET_INDEX])
            : nullptr;

    for (int payloadIdx = 0; payloadIdx < numPayloads; payloadIdx++) {
      // Store pointer for current payload using pre-calculated aligned offset
      params.recv_buffers[targetRank][payloadIdx] =
          targetWorkSpacePtr + payloadRecvBufferOffsets[payloadIdx];
    }
  }

  params.eplb_local_stats =
      enableEplb ? static_cast<int32_t const*>(eplbLocalStats.value().data_ptr()) : nullptr;

  // CFT requires all payloads to be 16B-aligned (fabric.try_put.counted operates on 16B chunks).
  if (useCftCountedWrites) {
    for (int i = 0; i < numPayloads; i++) {
      int const bytesPerToken = payloadElementSizes[i] * payloadElementsPerToken[i];
      TVM_FFI_ICHECK(bytesPerToken % 16 == 0)
          << "CFT dispatch payload " << i << " has " << bytesPerToken
          << " bytes per token; CFT counted writes require 16-byte alignment";
    }
  }
  params.use_cft_counted_writes = useCftCountedWrites;
  // Fused sanitization is a CFT dispatch optimisation; the fence path uses the standalone
  // moe_a2a_sanitize_expert_ids op instead, so these options are ignored without CFT.
  params.sanitize_expert_ids = useCftCountedWrites && sanitizeExpertIds;
  params.expert_id_payload_index = static_cast<int>(expertIdPayloadIndex);
  params.invalid_expert_id = static_cast<int32_t>(invalidTokenExpertId);

  // CFT handle-based counted writes
  if (useCftCountedWrites) {
    TVM_FFI_ICHECK(g_cft_manager && g_cft_manager->isInitialized())
        << "CFT counted writes requested but the CFT endpoints are not initialized";
    TVM_FFI_ICHECK(g_cft_manager->getLocalBackingPtr() ==
                   reinterpret_cast<CUdeviceptr>(rankWorkSpacePtr))
        << "CFT endpoints are bound to a different workspace";

    // Fill peer LE IDs
    auto const* leIds = g_cft_manager->getAllLeIds();
    for (int i = 0; i < static_cast<int>(epSize); i++) {
      params.cft_peer_le_ids[i] = leIds[i];
    }

    // LE payload offsets = workspace payload offsets (LE IS the workspace).
    // No separate LE layout — fabric.try_put.counted writes directly into workspace
    // recv_buffers.
    for (int i = 0; i < numPayloads; i++) {
      params.cft_le_payload_offsets[i] = payloadRecvBufferOffsets[i];
    }
    params.cft_le_counter_base = offsets[DISPATCH_COUNTED_WRITE_COUNTERS_OFFSET_INDEX];

    // recv_buffers and le_dispatch_counters already point to the workspace (set above).
    // No override needed — workspace IS the LE backing store.

    params.cft_dispatch_counter_baseline = reinterpret_cast<uint64_t*>(
        rankWorkSpacePtr + offsets[DISPATCH_COUNTER_BASELINE_OFFSET_INDEX]);
  }

  // Resolve the optional active-rank mask. Default (no mask) = all bits set, which
  // exactly reproduces the pre-fault-tolerance kernel behavior.
  resolveRankMask(enableRankMask, activeRankMask, epRank, params.enable_rank_mask,
                  params.active_rank_mask);

  params.stream = get_current_stream();
  params.timeout_cycles = timeoutCycles(timeoutSec);
  params.enable_pdl = enablePdl;

  // Prepare for dispatch (zero counters/indices and increment flag_val)
  tl::moe_a2a_prepare_dispatch_launch(params);

  // Launch the dispatch kernel
  tl::moe_a2a_dispatch_launch(params);

  cudaError_t result = cudaGetLastError();
  TVM_FFI_ICHECK(result == cudaSuccess)
      << "moe_a2a_dispatch kernel launch failed: " << cudaGetErrorString(result);

  Array<int64_t> recvOffsets;
  for (size_t offset : payloadRecvBufferOffsets) {
    recvOffsets.push_back(static_cast<int64_t>(offset));
  }
  int64_t const eplbGatheredStatsOffset =
      enableEplb ? offsets[EPLB_GATHERED_STATS_OFFSET_INDEX] : -1;
  return Tuple(recvOffsets, static_cast<int64_t>(offsets[COMBINE_INPUT_OFFSET_INDEX]),
               eplbGatheredStatsOffset);
}

tensorrt_llm::DataType toTllmDataType(DLDataType dtype) {
  if (dtype == dl_float16) {
    return tensorrt_llm::DataType::kHALF;
  }
  if (dtype == dl_bfloat16) {
    return tensorrt_llm::DataType::kBF16;
  }
  if (dtype == dl_float32) {
    return tensorrt_llm::DataType::kFLOAT;
  }
  TVM_FFI_LOG_AND_THROW(TypeError) << "Unsupported data type for payload";
  return tensorrt_llm::DataType::kFLOAT;
}

// MoE All-to-All Combine Operation
// Combine the per-rank expert outputs into the originating tokens' buffers on the local rank.
//
// The payload may be external or a view of the combine input region. Recognize the input
// region by its address; payloadInWorkspace=true additionally requires that zero-copy path.
// Other sources, including dispatch payload views, are staged when needed.
// Fence combine reads from 'combinePayloadOffset'. CFT combine stages the local slice and
// receives peer slices in a dedicated counted-write region before reduction.
// output: [local_num_tokens, elements_per_token] in the payload dtype.
void moeA2ACombineOp(TensorView payload, int64_t localNumTokens, TensorView workspace,
                     TensorView metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank,
                     int64_t epSize, int64_t topK, int64_t combinePayloadOffset,
                     bool payloadInWorkspace, bool useLowPrecision, bool useCftCountedWrites,
                     bool enableRankMask, Optional<TensorView> activeRankMask, TensorView output,
                     int64_t timeoutSec, bool enablePdl) {
  using namespace meta;

  // Validate inputs
  CHECK_INPUT(payload);
  TVM_FFI_ICHECK_EQ(payload.ndim(), 3)
      << "payload must be a 3D tensor [ep_size, max_tokens_per_rank, elements_per_token]";
  TVM_FFI_ICHECK_EQ(payload.size(0), epSize) << "payload first dimension must equal epSize";
  TVM_FFI_ICHECK(runtimeMaxTokensPerRank > 0) << "runtimeMaxTokensPerRank must be positive";
  TVM_FFI_ICHECK_EQ(payload.size(1), runtimeMaxTokensPerRank)
      << "payload second dimension must equal runtimeMaxTokensPerRank";
  // We only make sure the payload start offset is 16-byte aligned, while the actual vectorized
  // ld/st width is dynamically determined based on bytes per token of this payload.
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(payload.data_ptr()) % 16 == 0)
      << "payload must be 16-byte aligned";
  int64_t const elementsPerToken = payload.size(2);
  TVM_FFI_ICHECK(elementsPerToken > 0) << "elementsPerToken must be positive";
  TVM_FFI_ICHECK(epSize > 0 && epSize <= tl::kMaxRanks)
      << "epSize must be in the range (0, " << tl::kMaxRanks << "]";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "epRank must be in the range [0, epSize)";
  TVM_FFI_ICHECK(topK > 0 && topK <= tl::kMaxTopK) << "topK must be in the range (0, kMaxTopK]";
  TVM_FFI_ICHECK(localNumTokens >= 0) << "localNumTokens must be non-negative";

  tensorrt_llm::DataType const nvDtype = toTllmDataType(payload.dtype());
  int64_t const elementSize = get_element_size(payload);

  // Output dtype always matches the payload dtype: low-precision accumulates FP8 back to it.
  CHECK_INPUT(output);
  CHECK_DEVICE(payload, output);
  TVM_FFI_ICHECK(output.dtype() == payload.dtype()) << "output dtype must match the payload dtype";
  TVM_FFI_ICHECK(output.ndim() == 2 && output.size(0) == localNumTokens &&
                 output.size(1) == elementsPerToken)
      << "output must be a [local_num_tokens, elements_per_token] tensor";

  auto const offsets = readWorkspaceLayout(metainfo);

  // Validate workspace and set synchronization pointers
  CHECK_CUDA(workspace);
  CHECK_INPUT_TYPE(workspace, dl_uint8);
  TVM_FFI_ICHECK(workspace.ndim() == 2 && workspace.size(0) == epSize)
      << "workspace must be [ep_size, size_per_rank]";
  uint8_t* workspacePtr = static_cast<uint8_t*>(workspace.data_ptr());
  int64_t const sizePerRank = workspace.size(1);
  uint8_t* rankWorkSpacePtr = rankWorkspacePtr(workspace, epRank);
  TVM_FFI_ICHECK(epSize == offsets[EP_SIZE_INDEX] && topK == offsets[TOP_K_INDEX])
      << "Combine EP size/top_k differs from its workspace layout";
  TVM_FFI_ICHECK(sizePerRank >= offsets[WORKSPACE_SIZE_INDEX])
      << "Workspace is smaller than its layout";
  TVM_FFI_ICHECK(!useCftCountedWrites || offsets[CFT_ENABLED_INDEX])
      << "Workspace was allocated without CFT support";
  int64_t const regionSize = offsets[COMBINE_INPUT_SIZE_INDEX];
  TVM_FFI_ICHECK_EQ(combinePayloadOffset, offsets[COMBINE_INPUT_OFFSET_INDEX])
      << "combinePayloadOffset must address the fixed combine source region";
  TVM_FFI_ICHECK(runtimeMaxTokensPerRank <= offsets[MAX_NUM_TOKENS_INDEX])
      << "runtimeMaxTokensPerRank exceeds the allocation-time token capacity";
  uint8_t* combinePayloadPtr = rankWorkSpacePtr + combinePayloadOffset;
  // If the caller claims the payload is in the workspace, ensure it really is: a mismatch would
  // otherwise silently fall back to staging and lose the zero-copy path the caller asked for.
  bool const inputIsWorkspace = payload.data_ptr() == combinePayloadPtr;
  TVM_FFI_ICHECK(!payloadInWorkspace || inputIsWorkspace)
      << "payload_in_workspace is true but payload does not address the combine input region";
  int64_t const payloadSize = payload.numel() * elementSize;
  TVM_FFI_ICHECK(payloadSize <= regionSize)
      << "Combine payload exceeds its fixed workspace region: need " << payloadSize
      << " bytes, capacity " << regionSize;

  // Setup combine parameters
  tl::MoeA2ACombineParams params{};
  params.ep_size = static_cast<int>(epSize);
  params.ep_rank = static_cast<int>(epRank);
  params.local_num_tokens = static_cast<int>(localNumTokens);
  params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
  params.top_k = static_cast<int>(topK);
  params.source_payload = payload.data_ptr();
  params.output_data = output.data_ptr();
  params.elements_per_token = static_cast<int>(elementsPerToken);
  params.dtype = nvDtype;
  params.use_low_precision = useLowPrecision;
  params.source_stride_per_token = static_cast<int>(elementsPerToken * elementSize);
  params.wire_bytes_per_token =
      static_cast<int>(elementsPerToken) * (useLowPrecision ? 1 : static_cast<int>(elementSize));
  params.workspace_stride_per_token = useLowPrecision && !inputIsWorkspace
                                          ? params.wire_bytes_per_token
                                          : params.source_stride_per_token;

  params.flag_val = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);
  params.topk_target_ranks =
      reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_RANKS_OFFSET_INDEX]);
  params.topk_target_indices =
      reinterpret_cast<int*>(rankWorkSpacePtr + offsets[TOPK_TARGET_INDICES_OFFSET_INDEX]);
  params.recv_counters =
      reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);

  for (int targetRank = 0; targetRank < epSize; targetRank++) {
    uint8_t* targetWorkspacePtr = workspacePtr + targetRank * workspace.stride(0);
    params.completion_flags[targetRank] = reinterpret_cast<uint32_t*>(
        targetWorkspacePtr + offsets[COMBINE_COMPLETION_FLAGS_OFFSET_INDEX]);
    params.combine_input_buffers[targetRank] = targetWorkspacePtr + combinePayloadOffset;
  }

  // CFT requires the payload to be 16B-aligned (fabric.try_put.counted operates on 16B chunks).
  if (useCftCountedWrites) {
    TVM_FFI_ICHECK(params.wire_bytes_per_token % 16 == 0)
        << "CFT combine payload has " << params.wire_bytes_per_token
        << " bytes per token; CFT counted writes require 16-byte alignment";
  }

  // CFT receives peer pushes and the local contribution into a dedicated local inbox.
  // Fence combine instead reads the peer combine input buffers directly.
  params.use_cft_for_combine = useCftCountedWrites;
  if (useCftCountedWrites) {
    TVM_FFI_ICHECK(g_cft_manager && g_cft_manager->isInitialized())
        << "CFT counted writes requested but the CFT endpoints are not initialized";
    TVM_FFI_ICHECK(g_cft_manager->getLocalBackingPtr() ==
                   reinterpret_cast<CUdeviceptr>(rankWorkSpacePtr))
        << "CFT endpoints are bound to a different workspace";
    auto const* leIds = g_cft_manager->getAllLeIds();
    for (int i = 0; i < static_cast<int>(epSize); i++) {
      params.cft_peer_le_ids[i] = leIds[i];
    }

    // Dedicated combine receive region: prepare writes the local slice and fabric pushes write
    // peer slices.
    int64_t const combineRecvRegionOffset = offsets[COMBINE_RECV_OFFSET_INDEX];
    int64_t const receiveBytes = epSize * runtimeMaxTokensPerRank * params.wire_bytes_per_token;
    TVM_FFI_ICHECK(receiveBytes <= offsets[COMBINE_RECV_SIZE_INDEX])
        << "CFT combine receive payload exceeds its workspace capacity: need " << receiveBytes
        << " bytes, capacity " << offsets[COMBINE_RECV_SIZE_INDEX];
    params.cft_combine_recv_offset = static_cast<uint64_t>(combineRecvRegionOffset);
    params.cft_le_combine_counter_base = offsets[COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX];
    params.cft_le_combine_counters = reinterpret_cast<uint64_t*>(
        rankWorkSpacePtr + offsets[COMBINE_COUNTED_WRITE_COUNTERS_OFFSET_INDEX]);
    params.cft_combine_recv_payload = rankWorkSpacePtr + combineRecvRegionOffset;
    params.combine_counter_ep_stride = static_cast<int>(offsets[MAX_NUM_TOKENS_INDEX]);
    params.cft_combine_counter_baseline = reinterpret_cast<uint64_t*>(
        rankWorkSpacePtr + offsets[COMBINE_COUNTER_BASELINE_OFFSET_INDEX]);
  }

  // Resolve the optional active-rank mask. Default (no mask) = all bits set.
  resolveRankMask(enableRankMask, activeRankMask, epRank, params.enable_rank_mask,
                  params.active_rank_mask);

  // Resolve the complete payload plan once. Prepare always launches at least one block to
  // advance flag_val, but prepare_num_tokens=0 performs no payload work.
  params.prepare_first_token = 0;
  if (params.use_low_precision) {
    params.prepare_num_tokens = params.ep_size * params.max_tokens_per_rank;
  } else if (params.use_cft_for_combine) {
    params.prepare_first_token = params.ep_rank * params.max_tokens_per_rank;
    params.prepare_num_tokens = params.max_tokens_per_rank;
  } else {
    params.prepare_num_tokens = inputIsWorkspace ? 0 : params.ep_size * params.max_tokens_per_rank;
  }

  params.cft_push_payload = params.use_low_precision ? combinePayloadPtr : params.source_payload;
  params.cft_push_stride_per_token =
      params.use_low_precision ? params.workspace_stride_per_token : params.source_stride_per_token;
  params.reduce_stride_per_token =
      params.use_cft_for_combine ? params.wire_bytes_per_token : params.workspace_stride_per_token;

  params.stream = get_current_stream();
  params.timeout_cycles = timeoutCycles(timeoutSec);
  params.enable_pdl = enablePdl;

  tl::moe_a2a_prepare_combine_launch(params);

  // CFT combine push: processing rank pushes results back to originating rank's LE.
  if (params.use_cft_for_combine) {
    tl::moe_a2a_cft_combine_push_launch(params);
  }

  // Launch the combine kernel.
  tl::moe_a2a_combine_launch(params);
  cudaError_t result = cudaGetLastError();
  TVM_FFI_ICHECK(result == cudaSuccess)
      << "moe_a2a_combine kernel launch failed: " << cudaGetErrorString(result);
}

// Fill the expert ids of received padding rows (beyond each source rank's token count)
// with invalidExpertId. expertIds: [ep_size, runtime_max_tokens_per_rank, top_k] int32.
void moeA2ASanitizeExpertIdsOp(TensorView expertIds, TensorView workspace, TensorView metainfo,
                               int64_t epRank, int64_t invalidExpertId, bool enablePdl) {
  using namespace meta;
  CHECK_INPUT(expertIds);
  CHECK_INPUT_TYPE(expertIds, dl_int32);
  TVM_FFI_ICHECK_EQ(expertIds.ndim(), 3)
      << "expert_ids must be [ep_size, runtime_max_tokens_per_rank, top_k]";

  int const epSize = static_cast<int>(expertIds.size(0));
  int const runtimeMaxTokensPerRank = static_cast<int>(expertIds.size(1));
  int const topK = static_cast<int>(expertIds.size(2));

  auto const offsets = readWorkspaceLayout(metainfo);
  CHECK_INPUT_TYPE(workspace, dl_uint8);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2);
  uint8_t* rankWorkSpacePtr = rankWorkspacePtr(workspace, epRank);
  auto* recvCounters =
      reinterpret_cast<int*>(rankWorkSpacePtr + offsets[RECV_COUNTERS_OFFSET_INDEX]);
  auto* flagVal = reinterpret_cast<uint32_t*>(rankWorkSpacePtr + offsets[FLAG_VAL_OFFSET_INDEX]);

  tl::moe_a2a_sanitize_expert_ids_launch(static_cast<int32_t*>(expertIds.data_ptr()), recvCounters,
                                         flagVal, static_cast<int32_t>(invalidExpertId), epSize,
                                         runtimeMaxTokensPerRank, topK, get_current_stream(),
                                         enablePdl);
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "moe_a2a_sanitize_expert_ids launch failed: " << cudaGetErrorString(err);
}

// Expose metainfo indices and kernel limits as (names, values).
Tuple<Array<String>, Array<int64_t>> getMoeA2AMetaInfoIndexPairsOp() {
  Array<String> names;
  Array<int64_t> values;
  for (auto const& pair : meta::getMoeA2AMetaInfoIndexPairs()) {
    names.push_back(pair.first);
    values.push_back(pair.second);
  }
  return Tuple{names, values};
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_get_workspace_layout, moeA2AGetWorkspaceLayoutOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_initialize, moeA2AInitializeOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_cft_handle_bytes, moeA2ACftHandleBytesOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_cft_create_endpoint, moeA2ACftCreateEndpointOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_cft_import_endpoints, moeA2ACftImportEndpointsOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_cft_destroy, moeA2ACftDestroyOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_dispatch, moeA2ADispatchOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_combine, moeA2ACombineOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_sanitize_expert_ids, moeA2ASanitizeExpertIdsOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_get_metainfo_index_pairs, getMoeA2AMetaInfoIndexPairsOp);
