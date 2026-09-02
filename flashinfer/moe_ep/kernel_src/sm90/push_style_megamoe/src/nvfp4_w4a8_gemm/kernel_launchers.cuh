// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "decode.cuh"
#include "scheduler.cuh"

namespace flashinfer {
namespace sm90_w4a8 {

struct W4A8KernelLaunchParams {
  void* output;
  const float* activation_scales;
  const float* alpha;
  const int32_t* expert_mapping;
  const int64_t* source_offsets;
  const int64_t* tile_prefix;
  unsigned long long* task_counter;
  int64_t row_capacity;
  int32_t logical_n;
  int32_t padded_n;
  int32_t padded_k;
  int32_t launch_n_tiles;
  int32_t n_tile_begin;
  int32_t bucket_experts;
  int32_t total_experts;
  int64_t activation_scale_stride;
  bool alpha_per_expert;
  const float* group_scales;
  CUtensorMap activation_map;
  CUtensorMap payload_map;
  CUtensorMap residual_map;
};

struct W4A8KernelResources {
  int32_t blocks_per_sm;
  int32_t num_regs;
  size_t local_memory_bytes;
};

using W4A8ConfigureKernel = cudaError_t (*)(int, W4A8KernelResources*, W4A8KernelResources*);
using W4A8LaunchKernel = cudaError_t (*)(bool, int32_t, cudaStream_t,
                                         const W4A8KernelLaunchParams&);

struct W4A8KernelVariant {
  int32_t block_m;
  int32_t block_n;
  int32_t group_size;
  ResidualScheme residual_scheme;
  int32_t threads;
  int32_t pipeline_stages;
  int32_t min_blocks_per_sm;
  int32_t producer_registers;
  int32_t consumer_register_cap;
  int32_t register_footprint_target;
  size_t dynamic_smem_bytes;
  W4A8ConfigureKernel configure;
  W4A8LaunchKernel launch;
};

constexpr int kW4A8VariantsPerMN = 6;

const W4A8KernelVariant* get_w4a8_m64_n64_variants();
const W4A8KernelVariant* get_w4a8_m64_n128_variants();
const W4A8KernelVariant* get_w4a8_m128_n64_variants();
const W4A8KernelVariant* get_w4a8_m128_n128_variants();
const W4A8KernelVariant* get_w4a8_m64_n64_alternate_stage_variants();
const W4A8KernelVariant* get_w4a8_m64_n128_alternate_stage_variants();
const W4A8KernelVariant* get_w4a8_m128_n64_alternate_stage_variants();
const W4A8KernelVariant* get_w4a8_m128_n128_alternate_stage_variants();

template <int BlockM, int BlockN>
inline const W4A8KernelVariant* get_w4a8_kernel_variant_table() {
  static_assert(BlockM == 64 || BlockM == 128);
  static_assert(BlockN == 64 || BlockN == 128);
  if constexpr (BlockM == 64 && BlockN == 64) {
    return get_w4a8_m64_n64_variants();
  } else if constexpr (BlockM == 64 && BlockN == 128) {
    return get_w4a8_m64_n128_variants();
  } else if constexpr (BlockM == 128 && BlockN == 64) {
    return get_w4a8_m128_n64_variants();
  } else {
    return get_w4a8_m128_n128_variants();
  }
}

template <int BlockM, int BlockN>
inline const W4A8KernelVariant* get_w4a8_alternate_stage_variant_table() {
  static_assert(BlockM == 64 || BlockM == 128);
  static_assert(BlockN == 64 || BlockN == 128);
  if constexpr (BlockM == 64 && BlockN == 64) {
    return get_w4a8_m64_n64_alternate_stage_variants();
  } else if constexpr (BlockM == 64 && BlockN == 128) {
    return get_w4a8_m64_n128_alternate_stage_variants();
  } else if constexpr (BlockM == 128 && BlockN == 64) {
    return get_w4a8_m128_n64_alternate_stage_variants();
  } else {
    return get_w4a8_m128_n128_alternate_stage_variants();
  }
}

template <int BlockM>
constexpr int default_w4a8_pipeline_stages() {
  static_assert(BlockM == 64 || BlockM == 128);
  return BlockM == 64 ? 3 : 4;
}

template <int BlockM>
constexpr int alternate_w4a8_pipeline_stages() {
  static_assert(BlockM == 64 || BlockM == 128);
  return BlockM == 64 ? 2 : 3;
}

template <int GroupSize>
constexpr int w4a8_group_variant_index() {
  static_assert(GroupSize == 32 || GroupSize == 64 || GroupSize == 128);
  if constexpr (GroupSize == 32) {
    return 0;
  } else if constexpr (GroupSize == 64) {
    return 1;
  } else {
    return 2;
  }
}

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme>
inline const W4A8KernelVariant& get_w4a8_kernel_variant() {
  static_assert(Scheme == ResidualScheme::kGeneric || Scheme == ResidualScheme::kPow2);
  constexpr int kSchemeOffset = Scheme == ResidualScheme::kGeneric ? 0 : 3;
  constexpr int kVariantIndex = kSchemeOffset + w4a8_group_variant_index<GroupSize>();
  static_assert(kVariantIndex >= 0 && kVariantIndex < kW4A8VariantsPerMN);
  return get_w4a8_kernel_variant_table<BlockM, BlockN>()[kVariantIndex];
}

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme, int PipelineStages>
inline const W4A8KernelVariant& get_w4a8_kernel_variant() {
  static_assert(PipelineStages == default_w4a8_pipeline_stages<BlockM>() ||
                PipelineStages == alternate_w4a8_pipeline_stages<BlockM>());
  constexpr int kSchemeOffset = Scheme == ResidualScheme::kGeneric ? 0 : 3;
  constexpr int kVariantIndex = kSchemeOffset + w4a8_group_variant_index<GroupSize>();
  if constexpr (PipelineStages == default_w4a8_pipeline_stages<BlockM>()) {
    return get_w4a8_kernel_variant_table<BlockM, BlockN>()[kVariantIndex];
  } else {
    return get_w4a8_alternate_stage_variant_table<BlockM, BlockN>()[kVariantIndex];
  }
}

inline const W4A8KernelVariant* find_w4a8_kernel_variant(int32_t block_m, int32_t block_n,
                                                         int32_t group_size, ResidualScheme scheme,
                                                         int32_t pipeline_stages) {
  const int group_offset = group_size == 32 ? 0 : group_size == 64 ? 1 : group_size == 128 ? 2 : -1;
  if (group_offset < 0 || (scheme != ResidualScheme::kGeneric && scheme != ResidualScheme::kPow2)) {
    return nullptr;
  }
  const int variant_index = (scheme == ResidualScheme::kGeneric ? 0 : 3) + group_offset;
  const W4A8KernelVariant* table = nullptr;
  if (block_m == 64 && block_n == 64) {
    table = pipeline_stages == 3   ? get_w4a8_m64_n64_variants()
            : pipeline_stages == 2 ? get_w4a8_m64_n64_alternate_stage_variants()
                                   : nullptr;
  } else if (block_m == 64 && block_n == 128) {
    table = pipeline_stages == 3   ? get_w4a8_m64_n128_variants()
            : pipeline_stages == 2 ? get_w4a8_m64_n128_alternate_stage_variants()
                                   : nullptr;
  } else if (block_m == 128 && block_n == 64) {
    table = pipeline_stages == 4   ? get_w4a8_m128_n64_variants()
            : pipeline_stages == 3 ? get_w4a8_m128_n64_alternate_stage_variants()
                                   : nullptr;
  } else if (block_m == 128 && block_n == 128) {
    table = pipeline_stages == 4   ? get_w4a8_m128_n128_variants()
            : pipeline_stages == 3 ? get_w4a8_m128_n128_alternate_stage_variants()
                                   : nullptr;
  }
  return table == nullptr ? nullptr : table + variant_index;
}

}  // namespace sm90_w4a8
}  // namespace flashinfer
