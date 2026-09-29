/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef BLACKWELL_BGMV_MOE_BODY_FILE
#error "BLACKWELL_BGMV_MOE_BODY_FILE must name one generated BGMV MoE body"
#endif
#ifndef BLACKWELL_BGMV_MOE_HIDDEN
#error "BLACKWELL_BGMV_MOE_HIDDEN must describe the generated hidden size"
#endif
#ifndef BLACKWELL_BGMV_MOE_INPUT_DTYPE
#error "BLACKWELL_BGMV_MOE_INPUT_DTYPE must describe the generated input dtype"
#endif
#ifndef BLACKWELL_BGMV_MOE_SHRINK_DECODE
#error "BLACKWELL_BGMV_MOE_SHRINK_DECODE must name the generated kernel symbol"
#endif
#ifndef BLACKWELL_BGMV_MOE_SHRINK_PREFILL
#error "BLACKWELL_BGMV_MOE_SHRINK_PREFILL must name the generated kernel symbol"
#endif
#ifndef BLACKWELL_BGMV_MOE_EXPAND_TOKEN_T64
#error "BLACKWELL_BGMV_MOE_EXPAND_TOKEN_T64 must name the generated kernel symbol"
#endif
#ifndef BLACKWELL_BGMV_MOE_EXPAND_TOKEN
#error "BLACKWELL_BGMV_MOE_EXPAND_TOKEN must name the generated kernel symbol"
#endif
#ifndef BLACKWELL_BGMV_MOE_EXPAND_TOKEN_DUAL
#error "BLACKWELL_BGMV_MOE_EXPAND_TOKEN_DUAL must name the generated kernel symbol"
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

#include "tvm_ffi_utils.h"

// The generated source owns private fixed-width aliases and a tensor-map
// stand-in. Rename them at the include boundary to avoid CUDA-header clashes.
#define uint8_t blackwell_bgmv_generated_uint8_t
#define uint16_t blackwell_bgmv_generated_uint16_t
#define uint32_t blackwell_bgmv_generated_uint32_t
#define uint64_t blackwell_bgmv_generated_uint64_t
#define int32_t blackwell_bgmv_generated_int32_t
#define int16_t blackwell_bgmv_generated_int16_t
#define BlackwellTensorMap blackwell_bgmv_generated_BlackwellTensorMap
#define BlackwellTensorMapPack blackwell_bgmv_generated_BlackwellTensorMapPack
#define CUtensorMap blackwell_bgmv_generated_CUtensorMap
#include BLACKWELL_BGMV_MOE_BODY_FILE
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t
#undef BlackwellTensorMap
#undef BlackwellTensorMapPack
#undef CUtensorMap

namespace flashinfer {
namespace blackwell_bgmv_moe {

constexpr int32_t kHidden = BLACKWELL_BGMV_MOE_HIDDEN;
constexpr int32_t kRank = 32;
constexpr int32_t kShrinkThreads = 128;
constexpr int32_t kShrinkDecodePairsPerBlock = 4;
constexpr int32_t kShrinkDecodeSmemBytes = 221696;
constexpr int32_t kShrinkPrefillSmemBytes = 36992;
constexpr int32_t kExpandSmemBytes = 128;

enum class Schedule : int32_t {
  kTokenOwnedT64 = 0,
  kTokenOwned = 1,
  kTokenOwnedDualCol = 2,
};

inline void CheckCuda(cudaError_t status, const char* operation) {
  TVM_FFI_ICHECK(status == cudaSuccess) << operation << " failed: " << cudaGetErrorString(status);
}

inline void CheckExactSM100(int32_t device_id) {
  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == 0)
      << "Blackwell BGMV MoE requires exact compute capability 10.0, got " << major << "." << minor;
}

void Configure() {
  int32_t device_id = 0;
  CheckCuda(cudaGetDevice(&device_id), "cudaGetDevice");
  CheckExactSM100(device_id);

  int32_t max_dynamic_smem = 0;
  CheckCuda(
      cudaDeviceGetAttribute(&max_dynamic_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id),
      "cudaDeviceGetAttribute(max opt-in shared memory)");
  TVM_FFI_ICHECK(max_dynamic_smem >= kShrinkDecodeSmemBytes)
      << "Blackwell BGMV MoE decode shrink requires " << kShrinkDecodeSmemBytes
      << " bytes of dynamic shared memory, but device " << device_id << " supports "
      << max_dynamic_smem;
  CheckCuda(
      cudaFuncSetAttribute(BLACKWELL_BGMV_MOE_SHRINK_DECODE,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kShrinkDecodeSmemBytes),
      "cudaFuncSetAttribute(Blackwell BGMV MoE decode shrink)");
}

inline void CheckCompact(const TensorView& tensor, const char* name) {
  CHECK_CONTIGUOUS(tensor);
  TVM_FFI_ICHECK(tensor.numel() <= std::numeric_limits<int32_t>::max())
      << name << " exceeds the generated kernel's int32 index range";
}

void Run(TensorView y_accum, TensorView shrink_out, TensorView x, TensorView lora_a,
         TensorView lora_b, TensorView sorted_token_ids, TensorView expert_ids,
         TensorView lora_indices, TensorView topk_weights, int64_t schedule_value,
         int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  CHECK_CUDA(x);
  const int32_t device_id = x.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckExactSM100(device_id);

  CHECK_CUDA(y_accum);
  CHECK_CUDA(shrink_out);
  CHECK_CUDA(lora_a);
  CHECK_CUDA(lora_b);
  CHECK_CUDA(sorted_token_ids);
  CHECK_CUDA(expert_ids);
  CHECK_CUDA(lora_indices);
  CHECK_CUDA(topk_weights);
  CHECK_DEVICE(x, y_accum);
  CHECK_DEVICE(x, shrink_out);
  CHECK_DEVICE(x, lora_a);
  CHECK_DEVICE(x, lora_b);
  CHECK_DEVICE(x, sorted_token_ids);
  CHECK_DEVICE(x, expert_ids);
  CHECK_DEVICE(x, lora_indices);
  CHECK_DEVICE(x, topk_weights);

  CHECK_INPUT_TYPE(x, BLACKWELL_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(shrink_out, BLACKWELL_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(lora_a, BLACKWELL_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(lora_b, BLACKWELL_BGMV_MOE_INPUT_DTYPE);
  CHECK_INPUT_TYPE(y_accum, dl_float32);
  CHECK_INPUT_TYPE(topk_weights, dl_float32);
  CHECK_INPUT_TYPE(sorted_token_ids, dl_int64);
  CHECK_INPUT_TYPE(expert_ids, dl_int64);
  CHECK_INPUT_TYPE(lora_indices, dl_int64);

  TVM_FFI_ICHECK(x.ndim() == 2 && x.size(0) > 0 && x.size(1) == kHidden)
      << "x must have shape [num_tokens, " << kHidden << "]";
  const int32_t num_tokens = static_cast<int32_t>(x.size(0));
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
      << "shrink_out must have shape [1, num_pairs, 32]";
  TVM_FFI_ICHECK(y_accum.ndim() == 2 && y_accum.size(0) == num_tokens && y_accum.size(1) == kHidden)
      << "y_accum must have shape [num_tokens, " << kHidden << "]";
  TVM_FFI_ICHECK(lora_a.ndim() == 4 && lora_a.size(0) > 0 && lora_a.size(1) > 0 &&
                 lora_a.size(2) == kRank && lora_a.size(3) == kHidden)
      << "lora_a must have shape [num_loras, num_experts, 32, " << kHidden << "]";
  const int32_t num_experts = static_cast<int32_t>(lora_a.size(1));
  TVM_FFI_ICHECK(lora_b.ndim() == 4 && lora_b.size(0) == lora_a.size(0) &&
                 lora_b.size(1) == num_experts && lora_b.size(2) == kHidden &&
                 lora_b.size(3) == kRank)
      << "lora_b must have shape [num_loras, num_experts, " << kHidden << ", 32]";

  CheckCompact(x, "x");
  CheckCompact(y_accum, "y_accum");
  CheckCompact(shrink_out, "shrink_out");
  CheckCompact(lora_a, "lora_a");
  CheckCompact(lora_b, "lora_b");
  CheckCompact(sorted_token_ids, "sorted_token_ids");
  CheckCompact(expert_ids, "expert_ids");
  CheckCompact(lora_indices, "lora_indices");
  CheckCompact(topk_weights, "topk_weights");

  TVM_FFI_ICHECK(schedule_value >= static_cast<int64_t>(Schedule::kTokenOwnedT64) &&
                 schedule_value <= static_cast<int64_t>(Schedule::kTokenOwnedDualCol))
      << "invalid Blackwell BGMV MoE schedule id: " << schedule_value;
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

  const dim3 shrink_block(kShrinkThreads, 1, 1);
  if (num_pairs <= 32) {
    const dim3 shrink_grid(
        (num_pairs + kShrinkDecodePairsPerBlock - 1) / kShrinkDecodePairsPerBlock, kRank / 8, 1);
    BLACKWELL_BGMV_MOE_SHRINK_DECODE<<<shrink_grid, shrink_block, kShrinkDecodeSmemBytes, stream>>>(
        shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr, lora_ptr, num_pairs, num_experts,
        num_tokens);
  } else {
    const dim3 shrink_grid(num_pairs, kRank / 8, 1);
    BLACKWELL_BGMV_MOE_SHRINK_PREFILL<<<shrink_grid, shrink_block, kShrinkPrefillSmemBytes,
                                        stream>>>(shrink_ptr, x_ptr, a_ptr, token_ptr, expert_ptr,
                                                  lora_ptr, num_pairs, num_experts, num_tokens);
  }
  CheckCuda(cudaGetLastError(), "Blackwell BGMV MoE shrink launch");

  const int32_t output_stride = kHidden;
  const int32_t output_offset = 0;
  if (schedule == Schedule::kTokenOwnedT64) {
    const dim3 grid(num_tokens, (kHidden + 63) / 64, 1);
    BLACKWELL_BGMV_MOE_EXPAND_TOKEN_T64<<<grid, 64, kExpandSmemBytes, stream>>>(
        y_ptr, shrink_ptr, b_ptr, token_ptr, expert_ptr, lora_ptr, weight_ptr, num_pairs,
        num_experts, num_tokens, output_stride, output_offset);
  } else if (schedule == Schedule::kTokenOwned) {
    const dim3 grid(num_tokens, (kHidden + 127) / 128, 1);
    BLACKWELL_BGMV_MOE_EXPAND_TOKEN<<<grid, 128, kExpandSmemBytes, stream>>>(
        y_ptr, shrink_ptr, b_ptr, token_ptr, expert_ptr, lora_ptr, weight_ptr, num_pairs,
        num_experts, num_tokens, output_stride, output_offset);
  } else {
    const dim3 grid(num_tokens, (kHidden + 255) / 256, 1);
    BLACKWELL_BGMV_MOE_EXPAND_TOKEN_DUAL<<<grid, 128, kExpandSmemBytes, stream>>>(
        y_ptr, shrink_ptr, b_ptr, token_ptr, expert_ptr, lora_ptr, weight_ptr, num_pairs,
        num_experts, num_tokens, output_stride, output_offset);
  }
  CheckCuda(cudaGetLastError(), "Blackwell BGMV MoE expand launch");
}

}  // namespace blackwell_bgmv_moe
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(configure, flashinfer::blackwell_bgmv_moe::Configure);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::blackwell_bgmv_moe::Run);
