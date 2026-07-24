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
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"
#include "tensorrt_llm/thop/moeAlltoAllMeta.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;
using tvm::ffi::Optional;
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

inline bool hasActiveRankMask(Optional<TensorView> const& maskTensor) {
  return maskTensor.has_value();
}

// Resolve a provided rank-mask tensor into a fixed-width uint64 array. Only called when
// enable_rank_mask is true; the caller must supply a mask in that case (no silent default),
// since a caller that opted into rank-mask mode without specifying which ranks are alive is
// almost certainly a bug.
inline void resolveActiveRankMask(Optional<TensorView> maskTensor, int64_t epRank,
                                  uint64_t (&out)[tl_throughput::kRankMaskWords]) {
  using tl_throughput::kMaxRanks;
  using tl_throughput::kRankMaskWords;
  TVM_FFI_ICHECK(epRank >= 0 && epRank < kMaxRanks)
      << "epRank must be in the range [0, " << kMaxRanks << ") for active_rank_mask";
  TVM_FFI_ICHECK(hasActiveRankMask(maskTensor))
      << "active_rank_mask must be defined when enable_rank_mask=True";
  TensorView const& t = maskTensor.value();
  CHECK_CPU(t);
  CHECK_CONTIGUOUS(t);
  CHECK_INPUT_TYPE(t, dl_uint64);
  TVM_FFI_ICHECK_EQ(t.ndim(), 1) << "active_rank_mask must be a 1D tensor";
  TVM_FFI_ICHECK_EQ(t.size(0), kRankMaskWords)
      << "active_rank_mask must have exactly " << kRankMaskWords << " uint64 elements";
  auto const* src = static_cast<uint64_t const*>(t.data_ptr());
  for (int w = 0; w < kRankMaskWords; ++w) {
    out[w] = src[w];
  }
  // Local rank's bit must be set; otherwise the kernel would be running on a "dead" rank.
  TVM_FFI_ICHECK((out[epRank >> 6] >> (epRank & 63)) & 1ULL)
      << "active_rank_mask must mark the local ep_rank (" << epRank << ") as active";
}

fi_throughput::MoeA2ADataOffsets calculateOffsets(int epSize, int maxNumTokens,
                                                  int eplbStatsNumExperts) {
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
  offsets[fi_throughput::EPLB_GATHERED_STATS_OFFSET_INDEX] = offset;
  offset += static_cast<size_t>(epSize) * static_cast<size_t>(eplbStatsNumExperts) * kInt32Bytes;

  offset = alignOffset(offset);
  offsets[fi_throughput::PAYLOAD_DATA_OFFSET_INDEX] = offset;
  return offsets;
}

int64_t getMoeA2AAuxDataSize(int64_t epSize, int64_t maxNumTokens, int64_t eplbStatsNumExperts) {
  return calculateOffsets(
      static_cast<int>(epSize), static_cast<int>(maxNumTokens),
      static_cast<int>(eplbStatsNumExperts))[fi_throughput::PAYLOAD_DATA_OFFSET_INDEX];
}

Tensor moeA2AInitializeOp(TensorView workspace, int64_t epRank, int64_t epSize,
                          int64_t maxNumTokens, int64_t eplbStatsNumExperts) {
  CHECK_INPUT_TYPE(workspace, dl_uint8);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2) << "workspace must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(workspace.size(0), epSize) << "workspace first dim must equal ep_size";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize) << "epRank out of range";

  auto stream = get_current_stream();
  auto* basePtr = static_cast<uint8_t*>(workspace.data_ptr());
  auto* rankPtr = basePtr + epRank * workspace.stride(0);
  auto result = cudaMemsetAsync(rankPtr, 0, workspace.size(1), stream);
  TVM_FFI_ICHECK(result == cudaSuccess) << "cudaMemsetAsync failed";

  TVM_FFI_ICHECK(eplbStatsNumExperts >= 0) << "eplb_stats_num_experts must be non-negative";
  auto offsets = calculateOffsets(static_cast<int>(epSize), static_cast<int>(maxNumTokens),
                                  static_cast<int>(eplbStatsNumExperts));
  Tensor metainfo = alloc_tensor({fi_throughput::NUM_METAINFO_FIELDS}, dl_int64, cpu);
  auto* metaPtr = static_cast<int64_t*>(metainfo.data_ptr());
  std::copy(offsets.begin(), offsets.end(), metaPtr);

  auto err = cudaStreamSynchronize(stream);
  TVM_FFI_ICHECK(err == cudaSuccess) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);

  return metainfo;
}

// Returns (recv_offsets, recv_byte_sizes, combine_payload_offset, eplb_gathered_stats_offset,
// eplb_stats_num_experts); the last two are -1 and 0 when EPLB is disabled.
Tuple<Array<int64_t>, Array<int64_t>, int64_t, int64_t, int64_t> moeA2ADispatchOp(
    TensorView tokenSelectedExperts, Array<Tensor> inputPayloads, TensorView workspace,
    TensorView metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank, int64_t epSize,
    int64_t topK, int64_t numExperts, bool enablePdl, Optional<TensorView> eplbLocalStats,
    bool enableRankMask, Optional<TensorView> activeRankMask) {
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
  TVM_FFI_ICHECK(localNumTokens >= 0) << "local_num_tokens must be non-negative";

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
  TVM_FFI_ICHECK(epSize > 0 && epSize <= tl_throughput::kMaxRanks)
      << "epSize must be in the range (0, " << tl_throughput::kMaxRanks << "]";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize);
  TVM_FFI_ICHECK(runtimeMaxTokensPerRank > 0);
  // Non-divisible num_experts % ep_size is supported via ceil/floor expert-to-rank partitioning.
  TVM_FFI_ICHECK(numExperts >= epSize) << "num_experts must be >= ep_size";
  TVM_FFI_ICHECK(topK > 0 && topK <= tl_throughput::kMaxTopK);

  // Optional EPLB stats: the dispatch kernel all-gathers each rank's eplb_local_stats.
  bool const enableEplb = eplbLocalStats.has_value();
  int64_t eplbStatsNumExperts = 0;
  if (enableEplb) {
    auto const& localStats = eplbLocalStats.value();
    CHECK_INPUT(localStats);
    CHECK_INPUT_TYPE(localStats, dl_int32);
    TVM_FFI_ICHECK_EQ(localStats.ndim(), 1) << "eplb_local_stats must be a 1D tensor";
    eplbStatsNumExperts = localStats.size(0);
    TVM_FFI_ICHECK(eplbStatsNumExperts > 0) << "eplb_local_stats must not be empty";
    TVM_FFI_ICHECK(eplbStatsNumExperts <= numExperts)
        << "eplb_local_stats size must be <= num_experts";
    // Must fit in the space reserved at initialize time (not overflow into the payload region).
    int64_t gatheredEnd = offsets[fi_throughput::EPLB_GATHERED_STATS_OFFSET_INDEX] +
                          epSize * eplbStatsNumExperts * static_cast<int64_t>(kInt32Bytes);
    TVM_FFI_ICHECK(gatheredEnd <= offsets[fi_throughput::PAYLOAD_DATA_OFFSET_INDEX])
        << "eplb_local_stats size (" << eplbStatsNumExperts
        << ") exceeds the eplb_stats_num_experts reserved at initialize time";
  }

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
  params.enable_pdl = enablePdl;
  params.ep_size = static_cast<int>(epSize);
  params.ep_rank = static_cast<int>(epRank);
  params.num_experts = static_cast<int>(numExperts);
  params.local_num_tokens = localNumTokens;
  params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
  params.top_k = static_cast<int>(topK);
  params.enable_eplb = enableEplb;
  params.eplb_stats_num_experts = static_cast<int>(eplbStatsNumExperts);
  params.eplb_local_stats =
      enableEplb ? static_cast<int32_t const*>(eplbLocalStats.value().data_ptr()) : nullptr;
  params.token_selected_experts = static_cast<int32_t const*>(tokenSelectedExperts.data_ptr());
  params.num_payloads = numPayloads;
  std::copy(payloadDescriptors.begin(), payloadDescriptors.end(), params.payloads);

  params.enable_rank_mask = enableRankMask;
  if (params.enable_rank_mask) {
    resolveActiveRankMask(activeRankMask, epRank, params.active_rank_mask);
  } else {
    TVM_FFI_ICHECK(!hasActiveRankMask(activeRankMask))
        << "active_rank_mask requires enable_rank_mask=True";
  }

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
    params.eplb_gathered_stats[targetRank] =
        enableEplb
            ? reinterpret_cast<int*>(targetWorkspacePtr +
                                     offsets[fi_throughput::EPLB_GATHERED_STATS_OFFSET_INDEX])
            : nullptr;

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

  // Absolute workspace offset (this rank) of the gathered-stats region, or -1 when disabled.
  int64_t eplbGatheredStatsOffset =
      enableEplb ? static_cast<int64_t>(rankWorkspaceOffset +
                                        offsets[fi_throughput::EPLB_GATHERED_STATS_OFFSET_INDEX])
                 : -1;
  return Tuple(recvOffsets, recvByteSizes, combinePayloadOffset, eplbGatheredStatsOffset,
               eplbStatsNumExperts);
}

nvinfer1::DataType toNvDataType(DLDataType dtype) {
  auto code = encode_dlpack_dtype(dtype);
  if (code == float8_e4m3fn_code) {
    return nvinfer1::DataType::kFP8;
  }
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

void moeA2ACombineIntoOp(TensorView payload, int64_t localNumTokens, TensorView workspace,
                         TensorView metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank,
                         int64_t epSize, int64_t topK, int64_t combinePayloadOffset,
                         bool payloadInWorkspace, Optional<DLDataType> outputDtype_,
                         Optional<TensorView> outputScales, double outputScalarScale,
                         int64_t sfLayout, bool useLowPrecision, bool enablePdl,
                         bool enableRankMask, Optional<TensorView> activeRankMask,
                         TensorView output) {
  using tl_throughput::MoeA2ACombineParams;
  using tl_throughput::MoeA2ACombineQuantMode;
  using tl_throughput::MoeA2ACombineSwizzleSFMode;
  CHECK_INPUT(payload);
  TVM_FFI_ICHECK_EQ(payload.ndim(), 3)
      << "payload must be [ep_size, runtime_max_tokens_per_rank, hidden]";
  TVM_FFI_ICHECK_EQ(payload.size(0), epSize);
  TVM_FFI_ICHECK_EQ(payload.size(1), runtimeMaxTokensPerRank);
  TVM_FFI_ICHECK(epSize > 0 && epSize <= tl_throughput::kMaxRanks)
      << "epSize must be in the range (0, " << tl_throughput::kMaxRanks << "]";
  TVM_FFI_ICHECK(epRank >= 0 && epRank < epSize);
  TVM_FFI_ICHECK(topK > 0 && topK <= tl_throughput::kMaxTopK);
  TVM_FFI_ICHECK(localNumTokens >= 0);

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

  auto stream = get_current_stream();

  // Output dtype precedence:
  //   - explicit outputDtype_ (e.g. MXFP8/FP4 output quantization) wins;
  //   - else low-precision combine upcasts the FP8 recv buffers to BF16;
  //   - else matches the payload dtype.
  DLDataType outputDtype = outputDtype_.has_value()
                               ? outputDtype_.value()
                               : (useLowPrecision ? dl_bfloat16 : payload.dtype());
  // FP4 output packs two e2m1 values per byte, so its row is half as wide.
  auto const output_shape = outputDtype == dl_uint8
                                ? tvm::ffi::Shape{localNumTokens, elementsPerToken / 2}
                                : tvm::ffi::Shape{localNumTokens, elementsPerToken};
  CHECK_INPUT(output);
  CHECK_DEVICE(payload, output);
  TVM_FFI_ICHECK_EQ(output.ndim(), 2) << "output must be a 2D tensor";
  TVM_FFI_ICHECK_EQ(output.size(0), output_shape[0]);
  TVM_FFI_ICHECK_EQ(output.size(1), output_shape[1]);
  TVM_FFI_ICHECK(output.dtype() == outputDtype)
      << "output dtype must match "
      << (outputDtype_.has_value() ? "output_dtype"
          : useLowPrecision        ? "bf16 (use_low_precision)"
                                   : "payload dtype")
      << " (expected " << outputDtype << ", got " << output.dtype() << ")";

  MoeA2ACombineParams params{};
  params.enable_pdl = enablePdl;
  params.ep_size = static_cast<int>(epSize);
  params.ep_rank = static_cast<int>(epRank);
  params.local_num_tokens = static_cast<int>(localNumTokens);
  params.max_tokens_per_rank = static_cast<int>(runtimeMaxTokensPerRank);
  params.top_k = static_cast<int>(topK);
  params.use_low_precision = useLowPrecision;
  params.prepare_payload = payloadInWorkspace ? nullptr : payload.data_ptr();
  params.output_data = output.data_ptr();
  params.elements_per_token = static_cast<int>(elementsPerToken);
  params.dtype = toNvDataType(payload.dtype());
  params.swizzle_mode = static_cast<MoeA2ACombineSwizzleSFMode>(sfLayout);
  params.output_scalar_scale = static_cast<float>(outputScalarScale);

  // Handle quantization parameters if output scales are provided
  if (outputScales.has_value()) {
    // Quantized combine (MXFP8/MXFP4/NVFP4) relies on Blackwell-only conversion instructions.
    auto const sm_version = tensorrt_llm::common::getSMVersion();
    TVM_FFI_ICHECK(sm_version >= 100)
        << "Quantized moe_a2a_combine requires SM>=100 (Blackwell), but got SM" << sm_version;
    TVM_FFI_ICHECK(payload.dtype() == dl_bfloat16 || payload.dtype() == dl_float16)
        << "Quantization only supports for fp16 or bf16 inputs";
    params.output_scales = outputScales.value().data_ptr();

    if (output.dtype() == dl_float8_e4m3fn) {
      // TODO(siyuan): currently only support MXFP8 quantization
      CHECK_INPUT_AND_TYPE(outputScales.value(), dl_uint8);
      params.quant_mode = MoeA2ACombineQuantMode::MXFP8;
    } else if (output.dtype() == dl_uint8) {
      // packed fp4
      params.quant_mode = outputScales.value().dtype() == dl_uint8 ? MoeA2ACombineQuantMode::MXFP4
                                                                   : MoeA2ACombineQuantMode::NVFP4;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "Quantization not supported for output dtype: " << output.dtype();
    }
  } else if (useLowPrecision) {
    // Low-precision combine upcasts the FP8 recv buffers to a BF16 output; no output scales.
    TVM_FFI_ICHECK(output.dtype() == dl_bfloat16)
        << "low-precision combine must produce a bf16 output";
    params.quant_mode = MoeA2ACombineQuantMode::NONE;
  } else {
    TVM_FFI_ICHECK(output.dtype() == payload.dtype())
        << "output_dtype without output_scales must match payload dtype";
    params.quant_mode = MoeA2ACombineQuantMode::NONE;
  }

  if (params.quant_mode != MoeA2ACombineQuantMode::NVFP4 && outputScalarScale != 1.0) {
    TLLM_LOG_WARNING(
        "moe_a2a_combine: output_scalar_scale=%f is ignored unless output quantization is NVFP4",
        outputScalarScale);
  }

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

  params.enable_rank_mask = enableRankMask;
  if (params.enable_rank_mask) {
    resolveActiveRankMask(activeRankMask, epRank, params.active_rank_mask);
  } else {
    TVM_FFI_ICHECK(!hasActiveRankMask(activeRankMask))
        << "active_rank_mask requires enable_rank_mask=True";
  }

  params.stream = stream;

  tl_throughput::moe_a2a_prepare_combine_launch(params);
  tl_throughput::moe_a2a_combine_launch(params);
  auto err = cudaGetLastError();
  TVM_FFI_ICHECK(err == cudaSuccess)
      << "moe_a2a_combine launch failed: " << cudaGetErrorString(err);
}

Tensor moeA2ACombineOp(TensorView payload, int64_t localNumTokens, TensorView workspace,
                       TensorView metainfo, int64_t runtimeMaxTokensPerRank, int64_t epRank,
                       int64_t epSize, int64_t topK, int64_t combinePayloadOffset,
                       bool payloadInWorkspace, Optional<DLDataType> outputDtype_,
                       Optional<TensorView> outputScales, double outputScalarScale,
                       int64_t sfLayout, bool useLowPrecision, bool enablePdl, bool enableRankMask,
                       Optional<TensorView> activeRankMask) {
  CHECK_INPUT(payload);
  TVM_FFI_ICHECK_EQ(payload.ndim(), 3)
      << "payload must be [ep_size, runtime_max_tokens_per_rank, hidden]";
  int64_t const elementsPerToken = payload.size(2);
  DLDataType const outputDtype = outputDtype_.has_value()
                                     ? outputDtype_.value()
                                     : (useLowPrecision ? dl_bfloat16 : payload.dtype());
  auto const outputShape = outputDtype == dl_uint8
                               ? tvm::ffi::Shape{localNumTokens, elementsPerToken / 2}
                               : tvm::ffi::Shape{localNumTokens, elementsPerToken};
  Tensor output = alloc_tensor(outputShape, outputDtype, payload.device());
  moeA2ACombineIntoOp(payload, localNumTokens, workspace, metainfo, runtimeMaxTokensPerRank, epRank,
                      epSize, topK, combinePayloadOffset, payloadInWorkspace, outputDtype_,
                      outputScales, outputScalarScale, sfLayout, useLowPrecision, enablePdl,
                      enableRankMask, activeRankMask, output);
  return output;
}

void moeA2ASanitizeExpertIdsOp(TensorView expertIds, TensorView workspace, TensorView metainfo,
                               int64_t epRank, int64_t invalidExpertId, bool enablePdl) {
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
      static_cast<int>(runtimeMaxTokensPerRank), static_cast<int>(topK), get_current_stream(),
      enablePdl);

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
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_combine_into, moeA2ACombineIntoOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_sanitize_expert_ids, moeA2ASanitizeExpertIdsOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(moe_a2a_get_metainfo_index_pairs, getMoeA2AMetaInfoIndexPairs);
