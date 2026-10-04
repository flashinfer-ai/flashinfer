/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 *
 * TVM-FFI binding for the generic-shape Cake BGMV MoE bundle: one compile-time
 * LoRA rank (CAKE_BGMV_MOE_RANK), any hidden size that is a multiple of 8 at
 * run time. The consuming JIT spec defines:
 *   CAKE_BGMV_MOE_BODY_FILE           generated body to include
 *   CAKE_BGMV_MOE_RANK                8, 16, 32 or 64
 *   CAKE_BGMV_MOE_INPUT_DTYPE         dl_bfloat16 or dl_float16
 *   CAKE_BGMV_MOE_CC_MAJOR / _MINOR   compute capability this module was built for
 *   CAKE_BGMV_MOE_SHRINK_DECODE       generated decode shrink kernel (PPB=4, 3 stages)
 *   CAKE_BGMV_MOE_SHRINK_PREFILL      generated prefill shrink kernel (PPB=1, 2 stages)
 *   CAKE_BGMV_MOE_EXPAND_T64          generated 64-lane token-owned expand kernel
 *   CAKE_BGMV_MOE_EXPAND_T128         generated 128-lane token-owned expand kernel
 */
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

#include CAKE_BGMV_MOE_BODY_FILE

namespace flashinfer {
namespace cake_bgmv_moe_generic {

constexpr int32_t kRank = CAKE_BGMV_MOE_RANK;
constexpr int32_t kRankTile = 8;
constexpr int32_t kVec = 8;
constexpr int32_t kShrinkThreads = 128;
constexpr int32_t kShrinkTileElements = 128 * kVec;  // one vec8 per lane per tile
constexpr int32_t kShrinkDecodePairsPerBlock = 4;
// Dynamic shared memory per launch; the generated body records the values its
// kernels were scheduled with (x/weight cp.async rings plus FP32 partials for
// the shrink kernels, routed activations plus the route list for expand).
constexpr int32_t kShrinkDecodeSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_DECODE;
constexpr int32_t kShrinkPrefillSmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_SHRINK_PREFILL;
constexpr int32_t kExpandT64SmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T64;
constexpr int32_t kExpandT128SmemBytes = CAKE_BGMV_MOE_GENERIC_SMEM_EXPAND_T128;
static_assert(kShrinkDecodeSmemBytes == 221824, "decode shrink smem layout changed");
static_assert(kShrinkPrefillSmemBytes == 37120, "prefill shrink smem layout changed");
static_assert(kRank % kRankTile == 0, "rank must be a multiple of the 8-row shrink tile");
// Token->pair route index published by the shrink kernels and consumed by the
// expand kernels for arbitrary pair order (u32 words): a 4-word header (launch
// counter, parity used by the current shrink), per token a monotonic route
// count, two launch-parity base counts and kRouteIndexMaxRoutes pair slots.
// The plan allocates it zeroed once; the kernels never reset it.
constexpr int32_t kRouteIndexMaxRoutes = 16;
constexpr int32_t kRouteIndexHeaderWords = 4;
constexpr int32_t kRouteIndexWordsPerToken = 3 + kRouteIndexMaxRoutes;
// Hidden-split shrink workspace appended to the route index (see
// flashinfer/jit/cake_bgmv_moe.py): FP32 partials [split][pair][64] as raw
// 32-bit words, then one arrival counter per (pair block, rank block).
constexpr int32_t kShrinkSplitMax = 8;
constexpr int32_t kShrinkSplitMaxPairs = 128;
constexpr int32_t kShrinkSplitPartialWords = kShrinkSplitMax * kShrinkSplitMaxPairs * 64;
constexpr int32_t kShrinkSplitCounterWords = kShrinkSplitMaxPairs * (64 / kRankTile);

enum class Schedule : int32_t {
  kTokenOwnedT64 = 0,
  kTokenOwned = 1,
};

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

// Each module is compiled for exactly one target (sm_90a for H100/H200,
// sm_100a for B200/GB200, sm_103a for B300/GB300). The device must match it;
// anything else fails closed instead of silently running another cubin.
inline void CheckCompiledArch(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == CAKE_BGMV_MOE_CC_MAJOR && minor == CAKE_BGMV_MOE_CC_MINOR)
      << "Cake BGMV MoE generic module was compiled for compute capability "
      << CAKE_BGMV_MOE_CC_MAJOR << "." << CAKE_BGMV_MOE_CC_MINOR << ", got " << major << "."
      << minor;
}

// Called once per loaded module: the device must match the compiled target and
// the decode shrink needs its opt-in dynamic shared memory.
void Configure() {
  int32_t device_id = 0;
  CheckCuda(cudaGetDevice(&device_id), "cudaGetDevice");
  CheckCompiledArch(device_id);

  int32_t max_dynamic_smem = 0;
  CheckCuda(
      cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id),
      "cudaDeviceGetAttribute(max opt-in shared memory)");
  TVM_FFI_ICHECK(max_dynamic_smem >= kShrinkDecodeSmemBytes)
      << "Cake BGMV MoE decode shrink requires " << kShrinkDecodeSmemBytes
      << " bytes of dynamic shared memory, but device " << device_id << " supports "
      << max_dynamic_smem;
  CheckCuda(
      cudaFuncSetAttribute(CAKE_BGMV_MOE_SHRINK_DECODE, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           kShrinkDecodeSmemBytes),
      "cudaFuncSetAttribute(Cake BGMV MoE decode shrink)");
}

inline void CheckCompact(const TensorView& tensor, const char* name) {
  CHECK_CONTIGUOUS(tensor);
  TVM_FFI_ICHECK(tensor.numel() <= std::numeric_limits<int32_t>::max())
      << name << " exceeds the generated kernel's int32 index range";
}

// The expand kernels take the output row stride at runtime, so the FP32
// accumulator may be a column slice of a wider row-major buffer: unit column
// stride, row stride at least the hidden size, and every row offset within
// the kernels' int32 index range.
inline int32_t OutputRowStride(const TensorView& y_accum, int64_t num_tokens, int64_t hidden) {
  TVM_FFI_ICHECK(y_accum.stride(1) == 1) << "y_accum must be contiguous along hidden";
  const int64_t row_stride = y_accum.stride(0);
  TVM_FFI_ICHECK(row_stride >= hidden)
      << "y_accum row stride " << row_stride << " is smaller than hidden " << hidden;
  TVM_FFI_ICHECK((num_tokens - 1) * row_stride + hidden <= std::numeric_limits<int32_t>::max())
      << "y_accum exceeds the generated kernel's int32 index range";
  return static_cast<int32_t>(row_stride);
}

void Run(TensorView y_accum, TensorView shrink_out, TensorView x, TensorView lora_a,
         TensorView lora_b, TensorView sorted_token_ids, TensorView expert_ids,
         TensorView lora_indices, TensorView topk_weights, TensorView route_index,
         int64_t schedule_value, int64_t shrink_decode, int64_t shrink_splits,
         int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  CHECK_CUDA(x);
  // The device/module match is checked once in Configure (module load); the
  // Python side already routes each device to the module compiled for it.
  ffi::CUDADeviceGuard device_guard(x.device().device_id);

  CHECK_CUDA(y_accum);
  CHECK_CUDA(shrink_out);
  CHECK_CUDA(lora_a);
  CHECK_CUDA(lora_b);
  CHECK_CUDA(sorted_token_ids);
  CHECK_CUDA(expert_ids);
  CHECK_CUDA(lora_indices);
  CHECK_CUDA(topk_weights);
  CHECK_CUDA(route_index);
  CHECK_DEVICE(x, y_accum);
  CHECK_DEVICE(x, shrink_out);
  CHECK_DEVICE(x, lora_a);
  CHECK_DEVICE(x, lora_b);
  CHECK_DEVICE(x, sorted_token_ids);
  CHECK_DEVICE(x, expert_ids);
  CHECK_DEVICE(x, lora_indices);
  CHECK_DEVICE(x, topk_weights);
  CHECK_DEVICE(x, route_index);

  CHECK_INPUT_TYPE(x, CAKE_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(shrink_out, CAKE_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(lora_a, CAKE_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(lora_b, CAKE_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(y_accum, dl_float32);
  CHECK_INPUT_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_TYPE(sorted_token_ids, dl_int64);
  CHECK_INPUT_TYPE(expert_ids, dl_int64);
  CHECK_INPUT_TYPE(lora_indices, dl_int64);
  CHECK_INPUT_TYPE(route_index, dl_int32);

  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(0) > 0 && x.size(1) > 0 && x.size(1) % kVec == 0)
      << "x must have shape [num_tokens, hidden] with hidden a positive multiple of 8";
  const int32_t num_tokens = static_cast<int32_t>(x.size(0));
  const int32_t hidden = static_cast<int32_t>(x.size(1));
  TVM_FFI_ICHECK(sorted_token_ids.ndim() == 1 && sorted_token_ids.size(0) > 0)
      << "sorted_token_ids must be a non-empty rank-1 tensor";
  const int32_t num_pairs = static_cast<int32_t>(sorted_token_ids.size(0));
  TVM_FFI_ICHECK(expert_ids.ndim() == 1 && expert_ids.size(0) == num_pairs)
      << "expert_ids must have shape [num_pairs]";
  TVM_FFI_ICHECK(topk_weights.ndim() == 1 && topk_weights.size(0) == num_pairs)
      << "topk_weights must have shape [num_pairs]";
  TVM_FFI_ICHECK(lora_indices.ndim() == 1 && lora_indices.size(0) == num_tokens)
      << "lora_indices must have shape [num_tokens]";
  TVM_FFI_ICHECK(shrink_out.ndim() == 3 && shrink_out.size(0) == 1 &&
                 shrink_out.size(1) == num_pairs && shrink_out.size(2) == kRank)
      << "shrink_out must have shape [1, num_pairs, " << kRank << "]";
  TVM_FFI_ICHECK(y_accum.ndim() == 2 && y_accum.size(0) == num_tokens && y_accum.size(1) == hidden)
      << "y_accum must have shape [num_tokens, " << hidden << "]";
  TVM_FFI_ICHECK(lora_a.ndim() == 4 && lora_a.size(0) > 0 && lora_a.size(1) > 0 &&
                 lora_a.size(2) == kRank && lora_a.size(3) == hidden)
      << "lora_a must have shape [num_loras, num_experts, " << kRank << ", " << hidden << "]";
  const int32_t num_experts = static_cast<int32_t>(lora_a.size(1));
  TVM_FFI_ICHECK(lora_b.ndim() == 4 && lora_b.size(0) == lora_a.size(0) &&
                 lora_b.size(1) == num_experts && lora_b.size(2) == hidden &&
                 lora_b.size(3) == kRank)
      << "lora_b must have shape [num_loras, num_experts, " << hidden << ", " << kRank << "]";

  CheckCompact(x, "x");
  CheckCompact(shrink_out, "shrink_out");
  CheckCompact(lora_a, "lora_a");
  CheckCompact(lora_b, "lora_b");
  CheckCompact(sorted_token_ids, "sorted_token_ids");
  CheckCompact(expert_ids, "expert_ids");
  CheckCompact(lora_indices, "lora_indices");
  CheckCompact(topk_weights, "topk_weights");
  CheckCompact(route_index, "route_index");
  const int64_t route_words =
      kRouteIndexHeaderWords + static_cast<int64_t>(num_tokens) * kRouteIndexWordsPerToken;
  TVM_FFI_ICHECK(route_index.ndim() == 1 && route_index.size(0) >= route_words +
                                                                       kShrinkSplitPartialWords +
                                                                       kShrinkSplitCounterWords)
      << "route_index must hold at least " << kRouteIndexHeaderWords << " + num_tokens * "
      << kRouteIndexWordsPerToken << " + " << (kShrinkSplitPartialWords + kShrinkSplitCounterWords)
      << " int32 words";

  TVM_FFI_ICHECK(schedule_value >= static_cast<int64_t>(Schedule::kTokenOwnedT64) &&
                 schedule_value <= static_cast<int64_t>(Schedule::kTokenOwned))
      << "invalid Cake BGMV MoE generic schedule id: " << schedule_value;
  const auto schedule = static_cast<Schedule>(schedule_value);
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  auto* y_ptr = static_cast<float*>(y_accum.data_ptr());
  auto* shrink_ptr = static_cast<unsigned short*>(shrink_out.data_ptr());
  auto* x_ptr = static_cast<unsigned short*>(x.data_ptr());
  auto* a_ptr = static_cast<unsigned short*>(lora_a.data_ptr());
  auto* b_ptr = static_cast<unsigned short*>(lora_b.data_ptr());
  auto* token_ptr = static_cast<long long*>(sorted_token_ids.data_ptr());
  auto* expert_ptr = static_cast<long long*>(expert_ids.data_ptr());
  auto* lora_ptr = static_cast<long long*>(lora_indices.data_ptr());
  auto* weight_ptr = static_cast<float*>(topk_weights.data_ptr());
  auto* route_ptr = static_cast<unsigned int*>(route_index.data_ptr());
  constexpr int32_t kRouteBuild = 1;
  constexpr int32_t kRouteLookup = 1;
  constexpr int32_t kRouteAdvance = 1;

  const int32_t num_tiles = (hidden + kShrinkTileElements - 1) / kShrinkTileElements;
  TVM_FFI_ICHECK(shrink_splits >= 1 && shrink_splits <= kShrinkSplitMax &&
                 shrink_splits <= num_tiles)
      << "shrink_splits must be in [1, min(" << kShrinkSplitMax << ", num_tiles=" << num_tiles
      << ")], got " << shrink_splits;
  TVM_FFI_ICHECK(shrink_splits == 1 || num_pairs <= kShrinkSplitMaxPairs)
      << "hidden-split shrink supports at most " << kShrinkSplitMaxPairs << " pairs, got "
      << num_pairs;
  TVM_FFI_ICHECK(shrink_decode == 0 || num_pairs <= 32)
      << "the decode shrink kernel supports at most 32 pairs, got " << num_pairs;
  const int32_t splits = static_cast<int32_t>(shrink_splits);
  auto* split_partials = reinterpret_cast<float*>(route_ptr + route_words);
  auto* split_counters = route_ptr + route_words + kShrinkSplitPartialWords;
  const dim3 shrink_block(kShrinkThreads, 1, 1);
  if (shrink_decode != 0) {
    const dim3 shrink_grid(
        (num_pairs + kShrinkDecodePairsPerBlock - 1) / kShrinkDecodePairsPerBlock,
        kRank / kRankTile, splits);
    CAKE_BGMV_MOE_SHRINK_DECODE<<<shrink_grid, shrink_block, kShrinkDecodeSmemBytes, stream>>>(
        shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
        num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
        splits);
  } else {
    const dim3 shrink_grid(num_pairs, kRank / kRankTile, splits);
    CAKE_BGMV_MOE_SHRINK_PREFILL<<<shrink_grid, shrink_block, kShrinkPrefillSmemBytes, stream>>>(
        shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
        num_tokens, route_ptr, kRouteBuild, hidden, num_tiles, split_partials, split_counters,
        splits);
  }
  CheckCuda(cudaGetLastError(), "Cake BGMV MoE generic shrink launch");

  const int32_t output_stride = OutputRowStride(y_accum, num_tokens, hidden);
  const int32_t output_offset = 0;
  if (schedule == Schedule::kTokenOwnedT64) {
    const dim3 grid(num_tokens, (hidden + 63) / 64, 1);
    CAKE_BGMV_MOE_EXPAND_T64<<<grid, 64, kExpandT64SmemBytes, stream>>>(
        y_ptr, shrink_ptr, b_ptr, token_ptr, expert_ptr, lora_ptr, weight_ptr, num_pairs,
        num_experts, num_tokens, output_stride, output_offset, route_ptr, kRouteLookup,
        kRouteAdvance, hidden);
  } else {
    const dim3 grid(num_tokens, (hidden + 127) / 128, 1);
    CAKE_BGMV_MOE_EXPAND_T128<<<grid, 128, kExpandT128SmemBytes, stream>>>(
        y_ptr, shrink_ptr, b_ptr, token_ptr, expert_ptr, lora_ptr, weight_ptr, num_pairs,
        num_experts, num_tokens, output_stride, output_offset, route_ptr, kRouteLookup,
        kRouteAdvance, hidden);
  }
  CheckCuda(cudaGetLastError(), "Cake BGMV MoE generic expand launch");
}

}  // namespace cake_bgmv_moe_generic
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure, flashinfer::cake_bgmv_moe_generic::Configure);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::cake_bgmv_moe_generic::Run);
