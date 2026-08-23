/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "flashinfer/exception.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h"

// These entry points are frozen CUDA sources compiled as part of the same JIT
// module. Keep this launch-only translation unit free of framework tensor
// types: the public TVM-FFI boundary owns validation and allocation.
extern "C" __global__ void kernel_flashinfer_mnnvl_moe_alltoall_prepare_dispatch(
    int*, int*, int, unsigned int*, bool);

extern "C" __global__ void kernel_flashinfer_mnnvl_moe_alltoall_dispatch(
    int32_t* token_selected_experts, uint8_t* payload_0, uint8_t* payload_1,
    uint8_t* payload_2, uint8_t* payload_3, uint8_t* payload_4, uint8_t* payload_5,
    uint8_t* workspace, int* eplb_local_stats, unsigned long long workspace_stride_bytes,
    unsigned long long flag_val_offset, unsigned long long local_token_counter_offset,
    unsigned long long send_counters_offset, unsigned long long recv_counters_offset,
    unsigned long long completion_flags_offset, unsigned long long topk_target_ranks_offset,
    unsigned long long topk_send_indices_offset, unsigned long long eplb_gathered_stats_offset,
    unsigned long long payload_0_offset, unsigned long long payload_1_offset,
    unsigned long long payload_2_offset, unsigned long long payload_3_offset,
    unsigned long long payload_4_offset, unsigned long long payload_5_offset,
    int payload_0_bytes, int payload_1_bytes, int payload_2_bytes, int payload_3_bytes,
    int payload_4_bytes, int payload_5_bytes, int num_payloads, int max_tokens_per_rank,
    int local_num_tokens, int ep_rank, int ep_size, int num_experts, int top_k,
    int eplb_stats_num_experts, bool enable_pdl, bool enable_eplb, bool enable_rank_mask,
    unsigned long long active_rank_mask);

extern "C" __global__ void kernel_flashinfer_mnnvl_moe_alltoall_stage_combine(
    uint8_t*, uint8_t*, unsigned long long, unsigned long long, unsigned long long,
    unsigned long long, int, bool);

extern "C" __global__ void kernel_flashinfer_mnnvl_moe_alltoall_publish_combine(
    uint8_t*, unsigned long long, unsigned long long, unsigned long long, int, int, bool, bool,
    unsigned long long);

#define DECLARE_COMBINE_TOP_K(TOP_K)                                                     \
  extern "C" __global__ void kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_##TOP_K( \
      uint8_t*, uint8_t*, unsigned long long, unsigned long long, unsigned long long,    \
      unsigned long long, int, int, int, int, int, int, int, bool, bool);

DECLARE_COMBINE_TOP_K(1)
DECLARE_COMBINE_TOP_K(2)
DECLARE_COMBINE_TOP_K(4)
DECLARE_COMBINE_TOP_K(6)
DECLARE_COMBINE_TOP_K(8)
DECLARE_COMBINE_TOP_K(10)
DECLARE_COMBINE_TOP_K(12)
DECLARE_COMBINE_TOP_K(14)
DECLARE_COMBINE_TOP_K(16)
DECLARE_COMBINE_TOP_K(18)
DECLARE_COMBINE_TOP_K(22)

#undef DECLARE_COMBINE_TOP_K

extern "C" __global__ void kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk8(
    uint8_t*, uint8_t*, unsigned long long, unsigned long long, unsigned long long,
    unsigned long long, int, int, int, int, int, int, int, bool, bool);

extern "C" __global__ void kernel_flashinfer_mnnvl_moe_alltoall_quantize_combine(
    uint8_t*, uint8_t*, uint8_t*, uint8_t*, uint8_t*, int, int, int, int, int, float, int, int,
    bool);

extern "C" __global__ void kernel_flashinfer_mnnvl_moe_alltoall_sanitize_expert_ids(
    int*, int*, int, int, int, int, int);

namespace tensorrt_llm::kernels::moe_alltoall {
namespace {

using tensorrt_llm::common::launchWithPdlWhenEnabled;

constexpr int kDispatchThreads = 256;
constexpr int kDispatchSharedBytes = 256;
constexpr int kCombineThreads = 256;
constexpr int kPublicationThreads = 64;
constexpr int kQuantThreads = 32;
constexpr int kSanitizeThreads = 256;
constexpr int kDTypeBFloat16 = 0;
constexpr int kDTypeFloat16 = 1;
constexpr int kDTypeFloat8E4M3 = 2;
constexpr int kDTypeFloat32 = 3;

template <typename T>
int ceilDiv(T numerator, T denominator) {
  return static_cast<int>((numerator + denominator - 1) / denominator);
}

uint64_t byteOffset(void const* pointer, uint8_t const* base) {
  auto const address = reinterpret_cast<uintptr_t>(pointer);
  auto const base_address = reinterpret_cast<uintptr_t>(base);
  FLASHINFER_CHECK(address >= base_address, "workspace pointer precedes allocation base");
  return static_cast<uint64_t>(address - base_address);
}

template <typename KernelFn>
void preloadKernel(char const* name, KernelFn kernel_fn) {
  cudaFuncAttributes attributes{};
  cudaError_t const error = cudaFuncGetAttributes(&attributes, kernel_fn);
  FLASHINFER_CHECK(error == cudaSuccess, "cudaFuncGetAttributes (", name,
                   ") failed: ", cudaGetErrorString(error));
}

int dtypeBytes(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kHALF:
    case nvinfer1::DataType::kBF16:
      return 2;
    case nvinfer1::DataType::kFP8:
      return 1;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    default:
      FLASHINFER_CHECK(false, "Unsupported dtype for moe_a2a_combine");
      return 0;
  }
}

int dtypeCode(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kBF16:
      return kDTypeBFloat16;
    case nvinfer1::DataType::kHALF:
      return kDTypeFloat16;
    case nvinfer1::DataType::kFP8:
      return kDTypeFloat8E4M3;
    case nvinfer1::DataType::kFLOAT:
      return kDTypeFloat32;
    default:
      FLASHINFER_CHECK(false, "Unsupported dtype for moe_a2a_combine");
      return 0;
  }
}

void preloadCombineKernel(int top_k) {
#define PRELOAD_COMBINE_TOP_K(TOP_K)                                                    \
  case TOP_K:                                                                           \
    preloadKernel("mnnvl_moe_alltoall_combine_top_k_" #TOP_K,                           \
                  kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_##TOP_K);           \
    return
  switch (top_k) {
    PRELOAD_COMBINE_TOP_K(1);
    PRELOAD_COMBINE_TOP_K(2);
    PRELOAD_COMBINE_TOP_K(4);
    PRELOAD_COMBINE_TOP_K(6);
    PRELOAD_COMBINE_TOP_K(8);
    PRELOAD_COMBINE_TOP_K(10);
    PRELOAD_COMBINE_TOP_K(12);
    PRELOAD_COMBINE_TOP_K(14);
    PRELOAD_COMBINE_TOP_K(16);
    PRELOAD_COMBINE_TOP_K(18);
    PRELOAD_COMBINE_TOP_K(22);
    default:
      FLASHINFER_CHECK(false, "unsupported top_k for moe_a2a_combine: ", top_k);
  }
#undef PRELOAD_COMBINE_TOP_K
}

bool useBf16TopK8Combine(MoeA2ACombineParams const& params) {
  return params.top_k == 8 && params.dtype == nvinfer1::DataType::kBF16 &&
         params.elements_per_token % 8 == 0;
}

uint8_t* localWorkspace(uint8_t* workspace, uint64_t stride, int rank) {
  return workspace + static_cast<uint64_t>(rank) * stride;
}

}  // namespace

void moe_a2a_prepare_dispatch_launch(MoeA2ADispatchParams const& params) {
  FLASHINFER_CHECK(params.workspace != nullptr, "workspace must be defined");
  launchWithPdlWhenEnabled(
      "mnnvl_moe_alltoall_prepare_dispatch", params.enable_pdl,
      kernel_flashinfer_mnnvl_moe_alltoall_prepare_dispatch, 1, params.ep_size, 0, params.stream,
      params.send_counters, params.local_token_counter, params.ep_size, params.flag_val,
      params.enable_pdl);
}

void moe_a2a_dispatch_launch(MoeA2ADispatchParams const& params) {
  FLASHINFER_CHECK(params.workspace != nullptr, "workspace must be defined");
  FLASHINFER_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK, "top_k is out of range");
  FLASHINFER_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks, "ep_size is out of range");
  FLASHINFER_CHECK(params.ep_rank >= 0 && params.ep_rank < params.ep_size,
                   "ep_rank is out of range");
  FLASHINFER_CHECK(params.num_payloads > 0 && params.num_payloads <= kMaxPayloads,
                   "num_payloads is out of range");

  auto* rank_workspace =
      localWorkspace(params.workspace, params.workspace_stride_bytes, params.ep_rank);
  uint8_t* payloads[kMaxPayloads];
  uint64_t payload_offsets[kMaxPayloads]{};
  int32_t payload_bytes[kMaxPayloads]{};
  for (int index = 0; index < kMaxPayloads; ++index) {
    if (index < params.num_payloads) {
      payloads[index] = static_cast<uint8_t*>(const_cast<void*>(params.payloads[index].src_data));
      payload_offsets[index] =
          byteOffset(params.recv_buffers[params.ep_rank][index], rank_workspace);
      payload_bytes[index] =
          params.payloads[index].element_size * params.payloads[index].elements_per_token;
    } else {
      payloads[index] = params.workspace;
    }
  }
  auto* eplb_stats = params.enable_eplb ? const_cast<int32_t*>(params.eplb_local_stats)
                                         : params.send_counters;
  uint64_t const eplb_offset =
      params.enable_eplb ? byteOffset(params.eplb_gathered_stats[params.ep_rank], rank_workspace)
                         : 0;

  int const grid = std::max(params.local_num_tokens, 1);
  launchWithPdlWhenEnabled(
      "mnnvl_moe_alltoall_dispatch", params.enable_pdl,
      kernel_flashinfer_mnnvl_moe_alltoall_dispatch, grid, kDispatchThreads,
      kDispatchSharedBytes, params.stream,
      const_cast<int32_t*>(params.token_selected_experts), payloads[0], payloads[1], payloads[2],
      payloads[3],
      payloads[4], payloads[5], params.workspace, eplb_stats, params.workspace_stride_bytes,
      byteOffset(params.flag_val, rank_workspace),
      byteOffset(params.local_token_counter, rank_workspace),
      byteOffset(params.send_counters, rank_workspace),
      byteOffset(params.recv_counters[params.ep_rank], rank_workspace),
      byteOffset(params.completion_flags[params.ep_rank], rank_workspace),
      byteOffset(params.topk_target_ranks, rank_workspace),
      byteOffset(params.topk_send_indices, rank_workspace), eplb_offset, payload_offsets[0],
      payload_offsets[1], payload_offsets[2], payload_offsets[3], payload_offsets[4],
      payload_offsets[5], payload_bytes[0], payload_bytes[1], payload_bytes[2], payload_bytes[3],
      payload_bytes[4], payload_bytes[5], params.num_payloads, params.max_tokens_per_rank,
      params.local_num_tokens, params.ep_rank, params.ep_size, params.num_experts, params.top_k,
      params.eplb_stats_num_experts, params.enable_pdl, params.enable_eplb,
      params.enable_rank_mask, params.active_rank_mask[0]);
}

void moe_a2a_prepare_combine_launch(MoeA2ACombineParams const& params) {
  FLASHINFER_CHECK(params.workspace != nullptr, "workspace must be defined");
  // CUDA lazy module loading may synchronize the device. Load every downstream
  // function before publication can enter its cross-rank wait.
  preloadKernel("mnnvl_moe_alltoall_stage_combine",
                kernel_flashinfer_mnnvl_moe_alltoall_stage_combine);
  preloadKernel("mnnvl_moe_alltoall_publish_combine",
                kernel_flashinfer_mnnvl_moe_alltoall_publish_combine);
  if (useBf16TopK8Combine(params)) {
    preloadKernel("mnnvl_moe_alltoall_combine_bf16_topk8",
                  kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk8);
  } else {
    preloadCombineKernel(params.top_k);
  }
  if (params.quant_mode != MoeA2ACombineQuantMode::NONE) {
    preloadKernel("mnnvl_moe_alltoall_quantize_combine",
                  kernel_flashinfer_mnnvl_moe_alltoall_quantize_combine);
  }
  auto* rank_workspace =
      localWorkspace(params.workspace, params.workspace_stride_bytes, params.ep_rank);
  uint64_t const payload_bytes =
      params.prepare_payload == nullptr
          ? 0
          : static_cast<uint64_t>(params.ep_size) * params.max_tokens_per_rank *
                params.elements_per_token * dtypeBytes(params.dtype);
  int const grid = payload_bytes == 0
                       ? 1
                       : std::min(128, ceilDiv(payload_bytes,
                                              uint64_t{kCombineThreads * 16}));
  launchWithPdlWhenEnabled(
      "mnnvl_moe_alltoall_stage_combine", params.enable_pdl,
      kernel_flashinfer_mnnvl_moe_alltoall_stage_combine, grid, kCombineThreads, 0, params.stream,
      static_cast<uint8_t*>(const_cast<void*>(params.prepare_payload)), params.workspace,
      params.workspace_stride_bytes, byteOffset(params.flag_val, rank_workspace),
      byteOffset(params.recv_buffers[params.ep_rank], params.workspace), payload_bytes,
      params.ep_rank, params.enable_pdl);
}

void moe_a2a_combine_launch(MoeA2ACombineParams const& params) {
  FLASHINFER_CHECK(params.workspace != nullptr, "workspace must be defined");
  FLASHINFER_CHECK(params.top_k > 0 && params.top_k <= kMaxTopK, "top_k is out of range");
  FLASHINFER_CHECK(params.ep_size > 0 && params.ep_size <= kMaxRanks, "ep_size is out of range");
  FLASHINFER_CHECK(params.ep_rank >= 0 && params.ep_rank < params.ep_size,
                   "ep_rank is out of range");

  auto* rank_workspace =
      localWorkspace(params.workspace, params.workspace_stride_bytes, params.ep_rank);
  uint64_t const flag_offset = byteOffset(params.flag_val, rank_workspace);
  uint64_t const completion_offset =
      byteOffset(params.completion_flags[params.ep_rank], rank_workspace);

  launchWithPdlWhenEnabled(
      "mnnvl_moe_alltoall_publish_combine", params.enable_pdl,
      kernel_flashinfer_mnnvl_moe_alltoall_publish_combine, 1, kPublicationThreads, 0,
      params.stream, params.workspace, params.workspace_stride_bytes, flag_offset,
      completion_offset, params.ep_rank, params.ep_size, params.enable_pdl,
      params.enable_rank_mask, params.active_rank_mask[0]);

  bool const quantized = params.quant_mode != MoeA2ACombineQuantMode::NONE;
  void* accumulation = params.local_num_tokens == 0
                           ? static_cast<void*>(params.workspace)
                           : (quantized ? params.accumulation_data : params.output_data);
  FLASHINFER_CHECK(accumulation != nullptr, "combine accumulation output must be defined");
  int const output_dtype_code =
      params.use_low_precision ? kDTypeBFloat16 : dtypeCode(params.dtype);
  int const grid = std::max(params.local_num_tokens, 1);
  if (useBf16TopK8Combine(params)) {
    launchWithPdlWhenEnabled(
        "mnnvl_moe_alltoall_combine_bf16_topk8", params.enable_pdl,
        kernel_flashinfer_mnnvl_moe_alltoall_combine_bf16_topk8, grid, kCombineThreads, 0,
        params.stream, params.workspace, static_cast<uint8_t*>(accumulation),
        params.workspace_stride_bytes, byteOffset(params.topk_target_ranks, rank_workspace),
        byteOffset(params.topk_send_indices, rank_workspace),
        byteOffset(params.recv_buffers[params.ep_rank], rank_workspace),
        params.max_tokens_per_rank, params.local_num_tokens, params.elements_per_token,
        dtypeBytes(params.dtype), dtypeCode(params.dtype), output_dtype_code, params.ep_rank,
        params.use_low_precision, params.enable_pdl);
  } else {
#define LAUNCH_COMBINE_TOP_K(TOP_K)                                                       \
  case TOP_K:                                                                             \
    launchWithPdlWhenEnabled(                                                              \
        "mnnvl_moe_alltoall_combine_top_k_" #TOP_K, params.enable_pdl,                    \
        kernel_flashinfer_mnnvl_moe_alltoall_combine_top_k_##TOP_K, grid,                 \
        kCombineThreads, 0, params.stream, params.workspace,                               \
        static_cast<uint8_t*>(accumulation), params.workspace_stride_bytes,                \
        byteOffset(params.topk_target_ranks, rank_workspace),                              \
        byteOffset(params.topk_send_indices, rank_workspace),                              \
        byteOffset(params.recv_buffers[params.ep_rank], rank_workspace),                   \
        params.max_tokens_per_rank, params.local_num_tokens, params.elements_per_token,    \
        dtypeBytes(params.dtype), dtypeCode(params.dtype), output_dtype_code,               \
        params.ep_rank, params.use_low_precision, params.enable_pdl);                       \
    break
  switch (params.top_k) {
    LAUNCH_COMBINE_TOP_K(1);
    LAUNCH_COMBINE_TOP_K(2);
    LAUNCH_COMBINE_TOP_K(4);
    LAUNCH_COMBINE_TOP_K(6);
    LAUNCH_COMBINE_TOP_K(8);
    LAUNCH_COMBINE_TOP_K(10);
    LAUNCH_COMBINE_TOP_K(12);
    LAUNCH_COMBINE_TOP_K(14);
    LAUNCH_COMBINE_TOP_K(16);
    LAUNCH_COMBINE_TOP_K(18);
    LAUNCH_COMBINE_TOP_K(22);
    default:
      FLASHINFER_CHECK(false, "unsupported top_k for moe_a2a_combine: ", params.top_k);
  }
#undef LAUNCH_COMBINE_TOP_K
  }

  if (!quantized || params.local_num_tokens == 0) {
    return;
  }

  int const quant_mode = static_cast<int>(params.quant_mode);
  int const block_size = params.quant_mode == MoeA2ACombineQuantMode::NVFP4 ? 16 : 32;
  int const blocks_per_row = ceilDiv(params.elements_per_token, block_size);
  int const padded_scale_cols =
      params.swizzle_mode == MoeA2ACombineSwizzleSFMode::LINEAR
          ? blocks_per_row
          : ceilDiv(blocks_per_row, 4) * 4;
  int const quant_grid = params.local_num_tokens * blocks_per_row;
  auto* output_bytes = static_cast<uint8_t*>(params.output_data);
  auto* scale_bytes = static_cast<uint8_t*>(params.output_scales);
  launchWithPdlWhenEnabled(
      "mnnvl_moe_alltoall_quantize_combine", params.enable_pdl,
      kernel_flashinfer_mnnvl_moe_alltoall_quantize_combine, quant_grid, kQuantThreads, 0,
      params.stream, static_cast<uint8_t*>(params.accumulation_data), output_bytes, output_bytes,
      scale_bytes, scale_bytes, params.elements_per_token, dtypeBytes(params.dtype),
      dtypeCode(params.dtype), quant_mode, static_cast<int>(params.swizzle_mode),
      params.output_scalar_scale, blocks_per_row, padded_scale_cols, params.enable_pdl);
}

void moe_a2a_sanitize_expert_ids_launch(int32_t* expert_ids, int32_t const* recv_counters,
                                        int32_t invalid_id, int ep_size,
                                        int max_tokens_per_rank, int top_k, cudaStream_t stream,
                                        bool enable_pdl) {
  int const total_tokens = ep_size * max_tokens_per_rank;
  int const grid = ceilDiv(total_tokens, kSanitizeThreads);
  int const pdl = enable_pdl ? 1 : 0;
  launchWithPdlWhenEnabled(
      "mnnvl_moe_alltoall_sanitize_expert_ids", enable_pdl,
      kernel_flashinfer_mnnvl_moe_alltoall_sanitize_expert_ids, grid, kSanitizeThreads, 0, stream,
      expert_ids, const_cast<int32_t*>(recv_counters), ep_size, max_tokens_per_rank, top_k,
      invalid_id, pdl);
}

}  // namespace tensorrt_llm::kernels::moe_alltoall
