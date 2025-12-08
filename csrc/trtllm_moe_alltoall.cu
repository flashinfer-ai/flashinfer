/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved.
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

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "flashinfer/utils.cuh"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/thop/moeAlltoAllMeta.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Shape;
using tvm::ffi::String;
using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
using tvm::ffi::Tuple;

namespace {

namespace tl_throughput = tensorrt_llm::kernels::moe_alltoall;
namespace fi_throughput = torch_ext::moe_alltoall;

constexpr size_t kCachelineAlignment = 128;
constexpr size_t kInt32Bytes = sizeof(int32_t);

inline size_t alignOffset(size_t offset, size_t alignment = kCachelineAlignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

fi_throughput::MoeA2ADataOffsets calculateOffsets(int epSize, int maxNumTokens) {
  fi_throughput::MoeA2ADataOffsets offsets{};
  size_t offset = 0;

  offsets[fi_throughput::FLAG_VAL_OFFSET_INDEX] = offset;
  offset += kInt32Bytes;

  offsets[fi_throughput::LOCAL_TOKEN_COUNTER_OFFSET_INDEX] = offset;
  offset += kInt32Bytes;

  offsets[fi_throughput::SEND_COUNTERS_OFFSET_INDEX] = offset;
  offset += static_cast<size_t>(epSize) * kInt32Bytes;

  offsets[fi_throughput::RECV_COUNTERS_OFFSET_INDEX] = offset;
  offset += static_cast<size_t>(epSize) * kInt32Bytes;

  offset = alignOffset(offset);
  offsets[fi_throughput::DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX] = offset;
  offset += static_cast<size_t>(epSize) * kInt32Bytes;

  offset = alignOffset(offset);
  offsets[fi_throughput::COMBINE_COMPLETION_FLAGS_OFFSET_INDEX] = offset;
  offset += static_cast<size_t>(epSize) * kInt32Bytes;

  offset = alignOffset(offset);
  offsets[fi_throughput::TOPK_TARGET_RANKS_OFFSET_INDEX] = offset;
  offset += static_cast<size_t>(maxNumTokens) * tl_throughput::kMaxTopK * kInt32Bytes;

  offset = alignOffset(offset);
  offsets[fi_throughput::TOPK_SEND_INDICES_OFFSET_INDEX] = offset;
  offset += static_cast<size_t>(maxNumTokens) * tl_throughput::kMaxTopK * kInt32Bytes;

  offset = alignOffset(offset);
  offsets[fi_throughput::PAYLOAD_DATA_OFFSET_INDEX] = offset;
  return offsets;
}

int64_t getMoeA2AAuxDataSize(int64_t epSize, int64_t maxNumTokens) {
  return calculateOffsets(static_cast<int>(epSize),
                          static_cast<int>(maxNumTokens))[fi_throughput::PAYLOAD_DATA_OFFSET_INDEX];
}

Tensor moeA2AInitializeOp(TensorView workspace, int64_t epRank, int64_t epSize,
                          int64_t maxNumTokens) {
  CHECK_INPUT_TYPE(workspace, dl_uint8);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2) << "workspace must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(workspace.size(0), epSize) << "workspace first dim must equal ep_size";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "epRank out of range";

  auto stream = get_current_stream();
  auto* basePtr = static_cast<uint8_t*>(workspace.data_ptr());
  auto* rankPtr = basePtr + epRank * workspace.stride(0);
  auto result = cudaMemsetAsync(rankPtr, 0, workspace.size(1), stream);
  TVM_FFI_ICHECK(result == cudaSuccess) << "cudaMemsetAsync failed";

  auto offsets = calculateOffsets(static_cast<int>(epSize), static_cast<int>(maxNumTokens));
  Tensor metainfo = alloc_tensor({fi_throughput::NUM_METAINFO_FIELDS}, dl_int64, cpu);
  auto* metaPtr = static_cast<int64_t*>(metainfo.data_ptr());
  std::copy(offsets.begin(), offsets.end(), metaPtr);

  auto err = cudaStreamSynchronize(stream);
  TVM_FFI_ICHECK(err == cudaSuccess) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);

  return metainfo;
}

Tuple<Array<int64_t>, Array<int64_t>, int64_t> moeA2ADispatchOp(
    TensorView tokenSelectedExperts, Array<Tensor> inputPayloads, TensorView workspace,
    TensorView metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize,
    int64_t topK, int64_t numExperts) {
  using tl_throughput::PayloadDescriptor;

  CHECK_INPUT(tokenSelectedExperts);
  CHECK_INPUT_TYPE(tokenSelectedExperts, dl_int32);
  TVM_FFI_ICHECK_EQ(tokenSelectedExperts.ndim(), 2) << "token_selected_experts must be 2D";
  TVM_FFI_ICHECK_EQ(tokenSelectedExperts.size(1), topK) << "token_selected_experts shape mismatch";

  int numPayloads = static_cast<int>(inputPayloads.size());
  TVM_FFI_ICHECK(numPayloads > 0) << "At least one payload is required";
  TVM_FFI_ICHECK(numPayloads <= tl_throughput::kMaxPayloads)
      << "Too many payloads: " << numPayloads << " > " << tl_throughput::kMaxPayloads;

  auto localNumTokens = static_cast<int>(tokenSelectedExperts.size(0));
  TVM_FFI_ICHECK(localNumTokens > 0) << "local_num_tokens must be positive";

  // Validate all payloads and calculate sizes
  for (int i = 0; i < numPayloads; ++i) {
    auto const& payload = inputPayloads[i];
    CHECK_INPUT(payload);
    TVM_FFI_ICHECK_EQ(payload.ndim(), 2) << "payload " << i << " must be 2D";
    TVM_FFI_ICHECK_EQ(payload.size(0), localNumTokens)
        << "payload " << i << " first dimension must match local_num_tokens";
  }

  CHECK_CPU(metainfo);
  CHECK_INPUT_TYPE(metainfo, dl_int64);
  TVM_FFI_ICHECK_EQ(metainfo.ndim(), 1);
  TVM_FFI_ICHECK_EQ(metainfo.size(0), fi_throughput::NUM_METAINFO_FIELDS);
  auto const* offsetsPtr = static_cast<int64_t const*>(metainfo.data_ptr());
  fi_throughput::MoeA2ADataOffsets offsets{};
  std::copy(offsetsPtr, offsetsPtr + fi_throughput::NUM_METAINFO_FIELDS, offsets.begin());

  CHECK_INPUT_TYPE(workspace, dl_uint8);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2);
  TVM_FFI_ICHECK_EQ(workspace.size(0), epSize);
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize);
  TVM_FFI_ICHECK(runtimeMaxTokensPerRank > 0);
  TVM_FFI_ICHECK(numExperts >= epSize && numExperts % epSize == 0)
      << "num_experts must be divisible by ep_size";
  TVM_FFI_ICHECK(topK > 0 && topK <= tl_throughput::kMaxTopK);

  // Calculate payload descriptors and sizes from input tensors
  std::vector<PayloadDescriptor> payloadDescriptors(numPayloads);
  std::vector<int64_t> payloadByteSizes(numPayloads);
  int64_t totalBytesNeeded = 0;

  for (int i = 0; i < numPayloads; ++i) {
    auto const& payload = inputPayloads[i];
    int elementsPerToken = static_cast<int>(payload.size(1));
    int elementSize = static_cast<int>(get_element_size(payload));

    payloadDescriptors[i].src_data = payload.data_ptr();
    payloadDescriptors[i].element_size = elementSize;
    payloadDescriptors[i].elements_per_token = elementsPerToken;

    int64_t bytesPerPayload =
        static_cast<int64_t>(epSize) * runtimeMaxTokensPerRank * elementsPerToken * elementSize;
    payloadByteSizes[i] = bytesPerPayload;
    totalBytesNeeded += bytesPerPayload;

    TVM_FFI_ICHECK(totalBytesNeeded % elementSize == 0)
        << "Misaligned payload buffer " << i << " with element size " << elementSize
        << ". Consider reordering payloads by largest to smallest element size";
  }

  auto* workspaceBase = static_cast<uint8_t*>(workspace.data_ptr());
  auto strideBytes = workspace.stride(0);
  size_t rankWorkspaceOffset = epRank * strideBytes;
  auto* rankWorkspacePtr = workspaceBase + rankWorkspaceOffset;
  int64_t sizePerRank = workspace.size(1);

  int64_t requiredSize = offsets[fi_throughput::PAYLOAD_DATA_OFFSET_INDEX] + totalBytesNeeded;
  TVM_FFI_ICHECK(sizePerRank >= requiredSize) << "workspace size per rank insufficient, need "
                                              << requiredSize << " bytes but has " << sizePerRank;

  tl_throughput::MoeA2ADispatchParams params{};
  params.one_block_per_token = tensorrt_llm::common::getEnvMoeA2AOneBlockPerToken();
  params.ep_size = static_cast<int>(epSize);
  params.ep_rank = static_cast<int>(epRank);
  params.num_experts_per_rank = static_cast<int>(numExperts / epSize);
  params.local_num_tokens = localNumTokens;
  params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
  params.top_k = static_cast<int>(topK);
  params.token_selected_experts = static_cast<int32_t const*>(tokenSelectedExperts.data_ptr());
  params.num_payloads = numPayloads;
  std::copy(payloadDescriptors.begin(), payloadDescriptors.end(), params.payloads);

  params.flag_val =
      reinterpret_cast<uint32_t*>(rankWorkspacePtr + offsets[fi_throughput::FLAG_VAL_OFFSET_INDEX]);
  params.local_token_counter = reinterpret_cast<int*>(
      rankWorkspacePtr + offsets[fi_throughput::LOCAL_TOKEN_COUNTER_OFFSET_INDEX]);
  params.send_counters =
      reinterpret_cast<int*>(rankWorkspacePtr + offsets[fi_throughput::SEND_COUNTERS_OFFSET_INDEX]);
  params.topk_target_ranks = reinterpret_cast<int*>(
      rankWorkspacePtr + offsets[fi_throughput::TOPK_TARGET_RANKS_OFFSET_INDEX]);
  params.topk_send_indices = reinterpret_cast<int*>(
      rankWorkspacePtr + offsets[fi_throughput::TOPK_SEND_INDICES_OFFSET_INDEX]);

  for (int targetRank = 0; targetRank < epSize; ++targetRank) {
    auto* targetWorkspacePtr = workspaceBase + targetRank * strideBytes;
    params.recv_counters[targetRank] = reinterpret_cast<int*>(
        targetWorkspacePtr + offsets[fi_throughput::RECV_COUNTERS_OFFSET_INDEX]);
    params.completion_flags[targetRank] = reinterpret_cast<uint32_t*>(
        targetWorkspacePtr + offsets[fi_throughput::DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX]);

    size_t offset = static_cast<size_t>(offsets[fi_throughput::PAYLOAD_DATA_OFFSET_INDEX]);
    for (int payloadIdx = 0; payloadIdx < numPayloads; ++payloadIdx) {
      params.recv_buffers[targetRank][payloadIdx] = targetWorkspacePtr + offset;
      offset += payloadByteSizes[payloadIdx];
    }
  }

  params.stream = get_current_stream();

  tl_throughput::moe_a2a_prepare_dispatch_launch(params);
  tl_throughput::moe_a2a_dispatch_launch(params);
  auto launchErr = cudaGetLastError();
  TVM_FFI_ICHECK(launchErr == cudaSuccess)
      << "moe_a2a_dispatch launch failed: " << cudaGetErrorString(launchErr);

  Array<int64_t> recvOffsets;
  Array<int64_t> recvByteSizes;
  recvOffsets.reserve(numPayloads);
  size_t localOffset = static_cast<size_t>(offsets[fi_throughput::PAYLOAD_DATA_OFFSET_INDEX]);
  for (auto payloadByteSize : payloadByteSizes) {
    recvOffsets.push_back(rankWorkspaceOffset + localOffset);
    recvByteSizes.push_back(payloadByteSize);
    localOffset += payloadByteSize;
  }

  int64_t combinePayloadOffset = static_cast<int64_t>(alignOffset(localOffset));
  return Tuple(recvOffsets, recvByteSizes, combinePayloadOffset);
}

nvinfer1::DataType toNvDataType(DLDataType dtype) {
  auto code = encode_dlpack_dtype(dtype);
  if (code == float16_code) {
    return nvinfer1::DataType::kHALF;
  }
  if (code == bfloat16_code) {
    return nvinfer1::DataType::kBF16;
  }
  if (code == float32_code) {
    return nvinfer1::DataType::kFLOAT;
  }
  TVM_FFI_LOG_AND_THROW(TypeError) << "Unsupported dtype for MoE combine";
  return nvinfer1::DataType::kFLOAT;
}

Tensor moeA2ACombineOp(TensorView payload, int64_t localNumTokens, TensorView workspace,
                       TensorView metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank,
                       int64_t epSize, int64_t topK, int64_t combinePayloadOffset,
                       bool payloadInWorkspace) {
  using tl_throughput::MoeA2ACombineParams;
  CHECK_INPUT(payload);
  TVM_FFI_ICHECK_EQ(payload.ndim(), 3)
      << "payload must be [ep_size, runtime_max_tokens_per_rank, hidden]";
  TVM_FFI_ICHECK_EQ(payload.size(0), epSize);
  TVM_FFI_ICHECK_EQ(payload.size(1), runtimeMaxTokensPerRank);
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize);
  TVM_FFI_ICHECK(topK > 0 && topK <= tl_throughput::kMaxTopK);
  TVM_FFI_ICHECK(localNumTokens > 0);

  CHECK_CPU(metainfo);
  CHECK_INPUT_TYPE(metainfo, dl_int64);
  TVM_FFI_ICHECK_EQ(metainfo.ndim(), 1);
  TVM_FFI_ICHECK_EQ(metainfo.size(0), fi_throughput::NUM_METAINFO_FIELDS);
  auto const* offsetsPtr = static_cast<int64_t const*>(metainfo.data_ptr());
  fi_throughput::MoeA2ADataOffsets offsets{};
  std::copy(offsetsPtr, offsetsPtr + fi_throughput::NUM_METAINFO_FIELDS, offsets.begin());

  CHECK_INPUT_TYPE(workspace, dl_uint8);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2);
  TVM_FFI_ICHECK_EQ(workspace.size(0), epSize);
  auto* workspaceBase = static_cast<uint8_t*>(workspace.data_ptr());
  auto strideBytes = workspace.stride(0);
  auto* rankWorkspacePtr = workspaceBase + epRank * strideBytes;
  int64_t sizePerRank = workspace.size(1);

  int64_t elementsPerToken = payload.size(2);
  int64_t payloadBytes =
      payload.numel() *
      get_element_size(payload);  // includes all ranks * runtime_max_tokens_per_rank
  TVM_FFI_ICHECK(combinePayloadOffset >= 0 && combinePayloadOffset + payloadBytes <= sizePerRank)
      << "workspace insufficient for combine payload region";

  if (payloadInWorkspace) {
    auto* expectedPtr = rankWorkspacePtr + combinePayloadOffset;
    TVM_FFI_ICHECK(payload.data_ptr() == expectedPtr)
        << "payload_in_workspace is True but tensor pointer mismatch: " << (void*)payload.data_ptr()
        << " != " << (void*)expectedPtr;
  }

  Tensor output =
      alloc_tensor({localNumTokens, elementsPerToken}, payload.dtype(), payload.device());

  MoeA2ACombineParams params{};
  params.one_block_per_token = tensorrt_llm::common::getEnvMoeA2AOneBlockPerToken();
  params.ep_size = static_cast<int>(epSize);
  params.ep_rank = static_cast<int>(epRank);
  params.local_num_tokens = static_cast<int>(localNumTokens);
  params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
  params.top_k = static_cast<int>(topK);
  params.prepare_payload = payloadInWorkspace ? nullptr : payload.data_ptr();
  params.output_data = output.data_ptr();
  params.elements_per_token = static_cast<int>(elementsPerToken);
  params.dtype = toNvDataType(payload.dtype());

  params.flag_val =
      reinterpret_cast<uint32_t*>(rankWorkspacePtr + offsets[fi_throughput::FLAG_VAL_OFFSET_INDEX]);
  params.topk_target_ranks = reinterpret_cast<int*>(
      rankWorkspacePtr + offsets[fi_throughput::TOPK_TARGET_RANKS_OFFSET_INDEX]);
  params.topk_send_indices = reinterpret_cast<int*>(
      rankWorkspacePtr + offsets[fi_throughput::TOPK_SEND_INDICES_OFFSET_INDEX]);
  params.recv_counters =
      reinterpret_cast<int*>(rankWorkspacePtr + offsets[fi_throughput::RECV_COUNTERS_OFFSET_INDEX]);

  for (int targetRank = 0; targetRank < epSize; ++targetRank) {
    auto* targetWorkspacePtr = workspaceBase + targetRank * strideBytes;
    params.completion_flags[targetRank] = reinterpret_cast<uint32_t*>(
        targetWorkspacePtr + offsets[fi_throughput::COMBINE_COMPLETION_FLAGS_OFFSET_INDEX]);
    params.recv_buffers[targetRank] = targetWorkspacePtr + combinePayloadOffset;
  }
  params.stream = get_current_stream();

  tl_throughput::moe_a2a_prepare_combine_launch(params);
  tl_throughput::moe_a2a_combine_launch(params);
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "moe_a2a_combine launch failed: " << cudaGetErrorString(err);
  return output;
}

void moeA2ASanitizeExpertIdsOp(TensorView expertIds, TensorView workspace, TensorView metainfo,
                               int64_t epRank, int64_t invalidExpertId) {
  CHECK_INPUT(expertIds);
  CHECK_INPUT_TYPE(expertIds, dl_int32);
  TVM_FFI_ICHECK_EQ(expertIds.ndim(), 3);
  int64_t epSize = expertIds.size(0);
  int64_t runtimeMaxTokensPerRank = expertIds.size(1);
  int64_t topK = expertIds.size(2);

  CHECK_CPU(metainfo);
  CHECK_INPUT_TYPE(metainfo, dl_int64);
  TVM_FFI_ICHECK_EQ(metainfo.ndim(), 1);
  TVM_FFI_ICHECK_EQ(metainfo.size(0), fi_throughput::NUM_METAINFO_FIELDS);
  auto const* offsetsPtr = static_cast<int64_t const*>(metainfo.data_ptr());
  fi_throughput::MoeA2ADataOffsets offsets{};
  std::copy(offsetsPtr, offsetsPtr + fi_throughput::NUM_METAINFO_FIELDS, offsets.begin());

  CHECK_INPUT_TYPE(workspace, dl_uint8);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2);
  auto* workspaceBase = static_cast<uint8_t*>(workspace.data_ptr());
  auto* rankWorkspacePtr = workspaceBase + epRank * workspace.stride(0);
  auto* recvCounters =
      reinterpret_cast<int*>(rankWorkspacePtr + offsets[fi_throughput::RECV_COUNTERS_OFFSET_INDEX]);

  tl_throughput::moe_a2a_sanitize_expert_ids_launch(
      static_cast<int32_t*>(expertIds.data_ptr()), recvCounters,
      static_cast<int32_t>(invalidExpertId), static_cast<int>(epSize),
      static_cast<int>(runtimeMaxTokensPerRank), static_cast<int>(topK), get_current_stream());

  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "moe_a2a_sanitize_expert_ids launch failed: " << cudaGetErrorString(err);
}

// Expose metainfo index constants for Python access
// Returns a tuple of (names, values) for all metainfo constants
Tuple<Array<String>, Array<int64_t>> getMoeA2AMetaInfoIndexPairs() {
  auto pairs = fi_throughput::getMoeA2AMetaInfoIndexPairs();

  Array<String> names;
  Array<int64_t> values;

  for (const auto& pair : pairs) {
    names.push_back(pair.first);
    values.push_back(pair.second);
  }

  return Tuple{names, values};
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_get_aux_data_size, getMoeA2AAuxDataSize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_initialize, moeA2AInitializeOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_dispatch, moeA2ADispatchOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_combine, moeA2ACombineOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_sanitize_expert_ids, moeA2ASanitizeExpertIdsOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_get_metainfo_index_pairs, getMoeA2AMetaInfoIndexPairs);
