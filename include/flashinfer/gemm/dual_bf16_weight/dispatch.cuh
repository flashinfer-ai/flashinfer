/*
 * Copyright (c) 2026 by FlashInfer team.
 *
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

#pragma once

#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>

#include <cstddef>

namespace flashinfer::gemm::dual_bf16_weight {

using Input = cutlass::bfloat16_t;

constexpr int kReductionTile = 128;
constexpr int kSplitKMaximumTokenCount = 256;
constexpr int kTwoSmMinimumTokenCount = 1024;
constexpr float kLowScale = 1.0f / 256.0f;

enum class OutputType {
  kFloat32,
  kBFloat16,
};

enum class KernelKind {
  kSplitK1Sm,
  kPersistent1Sm,
  kCluster2Sm,
};

struct ProblemShape {
  int token_count;
  int output_channel_count;
  int reduction_size;
};

struct Arguments {
  void* output;
  Input const* activation;
  Input const* weight_high;
  Input const* weight_low;
  int token_count;
  int output_channel_count;
  int reduction_size;
  OutputType output_type = OutputType::kFloat32;
};

struct KernelConfig {
  ProblemShape shape{};
  KernelKind kind = KernelKind::kSplitK1Sm;
  int output_channel_tile = 0;
  int token_tile = 0;
  int reduction_tile = kReductionTile;
  int split_k = 1;
  int stage_count = 0;
  int grid_size = 0;
  int shared_memory_bytes = 0;
  std::size_t partial_workspace_bytes = 0;
  std::size_t counter_workspace_bytes = 0;
  std::size_t workspace_bytes = 0;
  bool used_compatibility_fallback = false;
};

char const* kernel_kind_name(KernelKind kind);
char const* output_type_name(OutputType output_type);

// Select a dispatch configuration for the current CUDA device. The current
// implementation intentionally supports only exact SM100 (compute capability
// 10.0), because these kernels use SM100a-specific TMA/UMMA features.
cudaError_t select_kernel_config(ProblemShape const& shape, KernelConfig* config);

// Launch with caller-owned workspace. For split-K, the workspace contains FP32
// partial outputs followed by one int32 completion counter per output tile.
// Counters are cleared on stream before every launch, which makes a fresh
// caller-provided buffer safe and keeps CUDA Graph capture deterministic.
cudaError_t launch(Arguments const& arguments, KernelConfig const& config, void* workspace,
                   std::size_t workspace_bytes, cudaStream_t stream = nullptr);

}  // namespace flashinfer::gemm::dual_bf16_weight
