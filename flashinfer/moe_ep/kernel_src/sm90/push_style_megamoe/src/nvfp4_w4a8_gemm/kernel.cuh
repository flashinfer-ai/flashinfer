// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>

#include "decode.cuh"
#include "nv_internal/tensorrt_llm/deep_gemm/mma_utils.cuh"
#include "nv_internal/tensorrt_llm/deep_gemm/nvrtc_cutlass.cuh"
#include "scheduler.cuh"

#ifndef W4A8_OVERLAP
#define W4A8_OVERLAP 0
#endif

#if W4A8_OVERLAP != 0 && W4A8_OVERLAP != 1
#error "W4A8_OVERLAP must be 0 or 1"
#endif

#ifndef W4A8_SINGLE_READY
#define W4A8_SINGLE_READY 0
#endif

#ifndef W4A8_RESIDUAL_TMA
#define W4A8_RESIDUAL_TMA 1
#endif

#ifndef W4A8_EMPTY_FAMILY_EARLY_EXIT
#define W4A8_EMPTY_FAMILY_EARLY_EXIT 0
#endif

#ifndef W4A8_CROSS_STAGE_RETIRE
#define W4A8_CROSS_STAGE_RETIRE 0
#endif

#ifndef W4A8_SINGLE_PARTIAL
#define W4A8_SINGLE_PARTIAL 1
#endif

#ifndef W4A8_SPLIT_M64_TAIL
#define W4A8_SPLIT_M64_TAIL 1
#endif

#ifndef W4A8_PAYLOAD_V4
#define W4A8_PAYLOAD_V4 1
#endif

#if W4A8_SINGLE_READY != 0 && W4A8_SINGLE_READY != 1
#error "W4A8_SINGLE_READY must be 0 or 1"
#endif

#if W4A8_RESIDUAL_TMA != 0 && W4A8_RESIDUAL_TMA != 1
#error "W4A8_RESIDUAL_TMA must be 0 or 1"
#endif

#if W4A8_EMPTY_FAMILY_EARLY_EXIT != 0 && W4A8_EMPTY_FAMILY_EARLY_EXIT != 1
#error "W4A8_EMPTY_FAMILY_EARLY_EXIT must be 0 or 1"
#endif

#if W4A8_CROSS_STAGE_RETIRE != 0 && W4A8_CROSS_STAGE_RETIRE != 1
#error "W4A8_CROSS_STAGE_RETIRE must be 0 or 1"
#endif

#if W4A8_SINGLE_PARTIAL != 0 && W4A8_SINGLE_PARTIAL != 1
#error "W4A8_SINGLE_PARTIAL must be 0 or 1"
#endif

#if W4A8_SPLIT_M64_TAIL != 0 && W4A8_SPLIT_M64_TAIL != 1
#error "W4A8_SPLIT_M64_TAIL must be 0 or 1"
#endif

#if W4A8_PAYLOAD_V4 != 0 && W4A8_PAYLOAD_V4 != 1
#error "W4A8_PAYLOAD_V4 must be 0 or 1"
#endif

#if W4A8_SINGLE_PARTIAL && W4A8_CROSS_STAGE_RETIRE
#error "W4A8_SINGLE_PARTIAL requires per-stage retirement"
#endif

namespace flashinfer {
namespace sm90_w4a8 {

constexpr int kSm90OptInSharedMemoryBytes = 232448;
constexpr int kProducerNamedBarrier = 0;

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme, int PipelineStages>
struct alignas(1024) W4A8SharedStorage {
  static_assert(BlockM == 64 || BlockM == 128);
  static_assert(BlockN == 64 || BlockN == 128);
  static_assert(GroupSize == 32 || GroupSize == 64 || GroupSize == 128);
  using ResidualStorage = typename ResidualDecoder<Scheme>::Storage;
  static constexpr int kGroupsPerStage = kBlockK / GroupSize;
  static constexpr int kStages = W4A8LaunchTraits<BlockM, BlockN, PipelineStages>::kPipelineStages;

  alignas(1024) uint8_t activation[kStages][BlockM * kBlockK];
  alignas(1024) uint8_t raw_payload[kStages][BlockN * kBlockK / 2];
#if W4A8_RESIDUAL_TMA
#if W4A8_PAYLOAD_V4
  alignas(1024) ResidualStorage residual[kStages][BlockN][kBlockK / kV3ResidualBlockK];
#else
  alignas(1024) ResidualStorage
      residual[kStages][kBlockK / kV3PayloadTileK][BlockN][kV3ResidualsPerPayloadTile];
#endif
#endif
  alignas(1024) uint8_t decoded_weight[kStages][BlockN * kBlockK];
  alignas(8) uint64_t raw_full[kStages];
  alignas(8) uint64_t decoded_ready[kStages];
  alignas(8) uint64_t empty[kStages];
#if W4A8_CROSS_STAGE_RETIRE
  int32_t last_group[kStages];
#endif
  GroupedTask task;
};

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme, int PipelineStages>
__host__ __device__ constexpr size_t w4a8_smem_bytes() {
  return sizeof(W4A8SharedStorage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>);
}

static_assert(w4a8_smem_bytes<64, 64, 32, ResidualScheme::kGeneric, 3>() <=
              kSm90OptInSharedMemoryBytes);
static_assert(2 * w4a8_smem_bytes<64, 64, 32, ResidualScheme::kGeneric, 3>() <=
              kSm90OptInSharedMemoryBytes);
static_assert(w4a8_smem_bytes<64, 128, 32, ResidualScheme::kGeneric, 3>() <=
              kSm90OptInSharedMemoryBytes);
static_assert(w4a8_smem_bytes<128, 64, 32, ResidualScheme::kGeneric, 4>() <=
              kSm90OptInSharedMemoryBytes);
static_assert(w4a8_smem_bytes<128, 128, 32, ResidualScheme::kGeneric, 4>() <=
              kSm90OptInSharedMemoryBytes);

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tensor_map, uint64_t* barrier,
                                            void* destination, int32_t coordinate0,
                                            int32_t coordinate1) {
  const uint64_t descriptor = reinterpret_cast<uint64_t>(tensor_map);
  const uint32_t smem_destination = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  const uint32_t smem_barrier = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"(smem_destination), "l"(descriptor), "r"(coordinate0), "r"(coordinate1),
        "r"(smem_barrier)
      : "memory");
}

__device__ __forceinline__ void tma_load_3d(const CUtensorMap* tensor_map, uint64_t* barrier,
                                            void* destination, int32_t coordinate0,
                                            int32_t coordinate1, int32_t coordinate2) {
  const uint64_t descriptor = reinterpret_cast<uint64_t>(tensor_map);
  const uint32_t smem_destination = static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  const uint32_t smem_barrier = static_cast<uint32_t>(__cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3, %4}], [%5];\n"
      :
      : "r"(smem_destination), "l"(descriptor), "r"(coordinate0), "r"(coordinate1),
        "r"(coordinate2), "r"(smem_barrier)
      : "memory");
}

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme, int PipelineStages>
__device__ __forceinline__ void issue_tma_stage(
    W4A8SharedStorage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>& storage,
    const GroupedTask& task, int k_stage, int32_t padded_n, int32_t padded_k,
    const CUtensorMap& activation_map, const CUtensorMap& payload_map
#if W4A8_RESIDUAL_TMA
    ,
    const CUtensorMap& residual_map
#endif
) {
  using Storage = W4A8SharedStorage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>;
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kStages = Storage::kStages;
  constexpr int kGroupsPerStage = Storage::kGroupsPerStage;
  constexpr int kActivationStageBytes = BlockM * kBlockK;
  constexpr int kRawStageBytes = BlockN * kBlockK / 2;
#if W4A8_RESIDUAL_TMA
  constexpr int kResidualStageBytes =
      BlockN * (kBlockK / kV3ResidualBlockK) * sizeof(typename Storage::ResidualStorage);
#else
  constexpr int kResidualStageBytes = 0;
#endif
  constexpr int kExpectedBytes = kActivationStageBytes + kRawStageBytes + kResidualStageBytes;
  const int stage = k_stage % kStages;
  const int weight_n_tiles = padded_n / kV3PayloadTileN;
#if W4A8_PAYLOAD_V4
  const int weight_k_stages = padded_k / kBlockK;
#else
  const int weight_k_tiles = padded_k / kV3PayloadTileK;
#endif
#if W4A8_CROSS_STAGE_RETIRE
  storage.last_group[stage] = (k_stage + 1) * kGroupsPerStage - 1;
#endif
  tma_load_2d(&activation_map, &storage.raw_full[stage], storage.activation[stage],
              k_stage * kBlockK, static_cast<int32_t>(task.m_begin));
#if W4A8_PAYLOAD_V4
  const int stage_cell = task.bucket_expert * weight_k_stages + k_stage;
  tma_load_3d(&payload_map, &storage.raw_full[stage], storage.raw_payload[stage], 0, task.n_begin,
              stage_cell);
#if W4A8_RESIDUAL_TMA
  constexpr int kResidualRowsPerTmaRow = Scheme == ResidualScheme::kPow2 ? 2 : 1;
  tma_load_3d(&residual_map, &storage.raw_full[stage], &storage.residual[stage][0][0], 0,
              task.n_begin / kResidualRowsPerTmaRow, stage_cell);
#endif
#else
#pragma unroll
  for (int k32 = 0; k32 < kBlockK / kV3PayloadTileK; ++k32) {
    const int global_k_tile = k_stage * (kBlockK / kV3PayloadTileK) + k32;
#pragma unroll
    for (int n64 = 0; n64 < BlockN / kV3PayloadTileN; ++n64) {
      const int cell = (task.bucket_expert * weight_k_tiles + global_k_tile) * weight_n_tiles +
                       task.n_begin / kV3PayloadTileN + n64;
      tma_load_3d(&payload_map, &storage.raw_full[stage],
                  storage.raw_payload[stage] + k32 * BlockN * kV3PackedBytesPerRow +
                      n64 * kV3PayloadTileN * kV3PackedBytesPerRow,
                  0, 0, cell);
#if W4A8_RESIDUAL_TMA
      tma_load_2d(&residual_map, &storage.raw_full[stage],
                  &storage.residual[stage][k32][n64 * kV3PayloadTileN][0], 0, cell);
#endif
    }
  }
#endif
  reinterpret_cast<FullBarrier*>(&storage.raw_full[stage])->arrive_and_expect_tx(kExpectedBytes);
}

template <typename Output>
__device__ __forceinline__ void store_output_value(Output* output, int64_t index, float value);

template <>
__device__ __forceinline__ void store_output_value(__nv_bfloat16* output, int64_t index,
                                                   float value) {
  output[index] = __float2bfloat16_rn(value);
}

template <>
__device__ __forceinline__ void store_output_value(float* output, int64_t index, float value) {
  output[index] = value;
}

template <int BlockN>
__host__ __device__ constexpr int decode_tasks_per_stage() {
#if W4A8_DECODE_VECTOR
  return BlockN * (kBlockK / kV3PayloadTileK);
#else
  return BlockN * (kBlockK / kV3ResidualBlockK);
#endif
}

#if !W4A8_SINGLE_READY
template <int BlockN>
__host__ __device__ constexpr int decoded_writer_threads() {
  constexpr int kTasks = decode_tasks_per_stage<BlockN>();
  return kTasks < kProducerThreads ? kTasks : kProducerThreads;
}
#endif

template <int BlockN, ResidualScheme Scheme>
__device__ __forceinline__ bool producer_decode_stage(
    const uint8_t* raw_payload, const typename ResidualDecoder<Scheme>::Storage* staged_residual,
    uint8_t* decoded_weight, int producer_thread) {
#if W4A8_DECODE_VECTOR
  constexpr int kDecodeTasks = decode_tasks_per_stage<BlockN>();
  for (int task = producer_thread; task < kDecodeTasks; task += kProducerThreads) {
    const int k32_in_stage = task / BlockN;
    const int n_local = task % BlockN;
#if W4A8_PAYLOAD_V4
    const int raw_index = (n_local * 4 + k32_in_stage) * kV3PackedBytesPerRow;
    const int residual_index = n_local * 8 + k32_in_stage * kV3ResidualsPerPayloadTile;
#else
    const int raw_index = (k32_in_stage * BlockN + n_local) * kV3PackedBytesPerRow;
    const int residual_index = (k32_in_stage * BlockN + n_local) * kV3ResidualsPerPayloadTile;
#endif
    const int k_local = k32_in_stage * kV3PayloadTileK;
    run_vector_task<Scheme>(
        raw_payload + raw_index, decoded_weight + wgmma_swizzle_128b_offset(n_local, k_local),
        decoded_weight + wgmma_swizzle_128b_offset(n_local, k_local + kV3ResidualBlockK),
        staged_residual[residual_index], staged_residual[residual_index + 1]);
  }
  return producer_thread < kDecodeTasks;
#else
  constexpr int kDecodeTasks = decode_tasks_per_stage<BlockN>();
  for (int task = producer_thread; task < kDecodeTasks; task += kProducerThreads) {
    const int n_local = task / (kBlockK / kV3ResidualBlockK);
    const int residual_in_stage = task % (kBlockK / kV3ResidualBlockK);
    const int k32_in_stage = residual_in_stage / kV3ResidualsPerPayloadTile;
    const int residual_in_k32 = residual_in_stage % kV3ResidualsPerPayloadTile;
#if W4A8_PAYLOAD_V4
    const int raw_index = (n_local * 4 + k32_in_stage) * kV3PackedBytesPerRow +
                          residual_in_k32 * (kV3ResidualBlockK / 2);
    const int residual_index = n_local * 8 + residual_in_stage;
#else
    const int raw_index = (k32_in_stage * BlockN + n_local) * kV3PackedBytesPerRow +
                          residual_in_k32 * (kV3ResidualBlockK / 2);
    const int residual_index =
        (k32_in_stage * BlockN + n_local) * kV3ResidualsPerPayloadTile + residual_in_k32;
#endif
    run_scalar_task<Scheme>(
        raw_payload + raw_index,
        decoded_weight + wgmma_swizzle_128b_offset(n_local, residual_in_stage * kV3ResidualBlockK),
        staged_residual[residual_index]);
  }
  return producer_thread < kDecodeTasks;
#endif
}

template <int BlockN, ResidualScheme Scheme>
__device__ __forceinline__ bool producer_decode_global_stage(
    const GroupedTask& task, const uint8_t* raw_payload, uint8_t* decoded_weight, int k_stage,
    int producer_thread, int32_t padded_k, int32_t padded_n,
    const typename ResidualDecoder<Scheme>::Storage* residual) {
  const int k_tiles = padded_k / kV3PayloadTileK;
  const int n_tiles = padded_n / kV3PayloadTileN;
#if W4A8_DECODE_VECTOR
  constexpr int kDecodeTasks = decode_tasks_per_stage<BlockN>();
  for (int decode_task = producer_thread; decode_task < kDecodeTasks;
       decode_task += kProducerThreads) {
    const int k32_in_stage = decode_task / BlockN;
    const int n_local = decode_task % BlockN;
    const int global_n = task.n_begin + n_local;
    const int k_tile = k_stage * (kBlockK / kV3PayloadTileK) + k32_in_stage;
#if W4A8_PAYLOAD_V4
    const int64_t residual_index =
        ((static_cast<int64_t>(task.bucket_expert) * (padded_k / kBlockK) + k_stage) * padded_n +
         global_n) *
            8 +
        k32_in_stage * kV3ResidualsPerPayloadTile;
    const int raw_index = (n_local * 4 + k32_in_stage) * kV3PackedBytesPerRow;
#else
    const int64_t residual_index =
        v3_residual_offset(task.bucket_expert, k_tile, global_n / kV3PayloadTileN,
                           global_n % kV3PayloadTileN, 0, k_tiles, n_tiles);
    const int raw_index = (k32_in_stage * BlockN + n_local) * kV3PackedBytesPerRow;
#endif
    const int k_local = k32_in_stage * kV3PayloadTileK;
    run_vector_task<Scheme>(
        raw_payload + raw_index, decoded_weight + wgmma_swizzle_128b_offset(n_local, k_local),
        decoded_weight + wgmma_swizzle_128b_offset(n_local, k_local + kV3ResidualBlockK),
        residual[residual_index], residual[residual_index + 1]);
  }
  return producer_thread < kDecodeTasks;
#else
  constexpr int kDecodeTasks = decode_tasks_per_stage<BlockN>();
  for (int decode_task = producer_thread; decode_task < kDecodeTasks;
       decode_task += kProducerThreads) {
    const int n_local = decode_task / (kBlockK / kV3ResidualBlockK);
    const int residual_in_stage = decode_task % (kBlockK / kV3ResidualBlockK);
    const int k32_in_stage = residual_in_stage / kV3ResidualsPerPayloadTile;
    const int residual_in_k32 = residual_in_stage % kV3ResidualsPerPayloadTile;
    const int global_n = task.n_begin + n_local;
    const int k_tile = k_stage * (kBlockK / kV3PayloadTileK) + k32_in_stage;
#if W4A8_PAYLOAD_V4
    const int64_t residual_index =
        ((static_cast<int64_t>(task.bucket_expert) * (padded_k / kBlockK) + k_stage) * padded_n +
         global_n) *
            8 +
        residual_in_stage;
    const int raw_index = (n_local * 4 + k32_in_stage) * kV3PackedBytesPerRow +
                          residual_in_k32 * (kV3ResidualBlockK / 2);
#else
    const int64_t residual_index =
        v3_residual_offset(task.bucket_expert, k_tile, global_n / kV3PayloadTileN,
                           global_n % kV3PayloadTileN, residual_in_k32, k_tiles, n_tiles);
    const int raw_index = (k32_in_stage * BlockN + n_local) * kV3PackedBytesPerRow +
                          residual_in_k32 * (kV3ResidualBlockK / 2);
#endif
    run_scalar_task<Scheme>(
        raw_payload + raw_index,
        decoded_weight + wgmma_swizzle_128b_offset(n_local, residual_in_stage * kV3ResidualBlockK),
        residual[residual_index]);
  }
  return producer_thread < kDecodeTasks;
#endif
}

__device__ __forceinline__ void fence_decoded_writer() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme, int PipelineStages>
__device__ __forceinline__ const float* quant_group_scales(
    W4A8SharedStorage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>& storage,
    const GroupedTask& task, int global_group, int stage, int local_group, int32_t padded_n,
    int32_t padded_k, const float* group_scales) {
  (void)storage;
  (void)stage;
  (void)local_group;
  const int weight_n_tiles = padded_n / kV3PayloadTileN;
  const int weight_k_groups = padded_k / GroupSize;
  const int64_t offset =
      v3_group_scale_offset(task.bucket_expert, global_group, task.n_begin / kV3PayloadTileN, 0,
                            weight_k_groups, weight_n_tiles);
  return group_scales + offset;
}

template <int BlockN, int GroupSize>
__device__ __forceinline__ void accumulate_scaled_partial(float* final_accum, const float* partial,
                                                          float activation_scale0,
                                                          float activation_scale1,
                                                          const float* group_scales, int lane) {
  using WGMMA = typename deep_gemm::FP8MMASelector<BlockN>::type;
  static_assert(GroupSize == 32 || GroupSize == 64 || GroupSize == 128);
#pragma unroll
  for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
    const int column0 = i * 8 + (lane & 3) * 2;
    const int column1 = column0 + 1;
    float weight_scale0 = 0.0F;
    float weight_scale1 = 0.0F;
    if (lane < 4) {
      weight_scale0 = __ldg(group_scales + column0);
      weight_scale1 = __ldg(group_scales + column1);
    }
    weight_scale0 = __shfl_sync(0xffffffffU, weight_scale0, lane & 3);
    weight_scale1 = __shfl_sync(0xffffffffU, weight_scale1, lane & 3);
    final_accum[i * 4 + 0] += activation_scale0 * weight_scale0 * partial[i * 4 + 0];
    final_accum[i * 4 + 1] += activation_scale0 * weight_scale1 * partial[i * 4 + 1];
    final_accum[i * 4 + 2] += activation_scale1 * weight_scale0 * partial[i * 4 + 2];
    final_accum[i * 4 + 3] += activation_scale1 * weight_scale1 * partial[i * 4 + 3];
  }
}

template <int BlockN, int GroupSize>
__device__ __forceinline__ void retire_quant_group(float* final_accum, const float* partial,
                                                   float activation_scale0, float activation_scale1,
                                                   const float* group_scales, int retired_group,
                                                   int stage_last_group, uint64_t* empty_storage,
                                                   int lane) {
  accumulate_scaled_partial<BlockN, GroupSize>(final_accum, partial, activation_scale0,
                                               activation_scale1, group_scales, lane);
  if (retired_group == stage_last_group && lane == 0) {
    reinterpret_cast<cutlass::arch::ClusterBarrier*>(empty_storage)->arrive();
  }
}

template <typename Output, int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme,
          int PipelineStages>
__device__ __forceinline__ void grouped_w4a8_kernel_body(
    W4A8SharedStorage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>& storage, Output* output,
    const float* activation_scales, const float* alpha, const int32_t* expert_mapping,
    const int64_t* source_offsets, const int64_t* tile_prefix, unsigned long long* task_counter,
    int64_t row_capacity, int32_t logical_n, int32_t padded_n, int32_t padded_k,
    int32_t launch_n_tiles, int32_t n_tile_begin, int32_t bucket_experts,
    int64_t activation_scale_stride, bool alpha_per_expert, const CUtensorMap& activation_map,
    const CUtensorMap& payload_map,
#if W4A8_RESIDUAL_TMA
    const CUtensorMap& residual_map,
#else
    const typename ResidualDecoder<Scheme>::Storage* residual,
#endif
    const float* group_scales) {
  static_assert(std::is_same_v<Output, __nv_bfloat16> || std::is_same_v<Output, float>);
  static_assert(BlockM == 64 || BlockM == 128);
  static_assert(BlockN == 64 || BlockN == 128);
  static_assert(GroupSize == 32 || GroupSize == 64 || GroupSize == 128);
  static_assert(kBlockK % GroupSize == 0);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
  using WGMMA = typename deep_gemm::FP8MMASelector<BlockN>::type;
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using Barrier = cutlass::arch::ClusterBarrier;
  using Traits = W4A8LaunchTraits<BlockM, BlockN, PipelineStages>;
  constexpr int kStages = Traits::kPipelineStages;
  constexpr int kConsumerThreads = kConsumerThreadsFor<BlockM>;
  constexpr int kConsumerWarps = kConsumerThreads / 32;
  constexpr int kGroupsPerStage = kBlockK / GroupSize;
  constexpr int kMmaPerGroup = GroupSize / WGMMA::K;
  constexpr int kConsumerRegisters = Traits::kConsumerRegisters;

  const int lane = static_cast<int>(deep_gemm::get_lane_id());
  const int warp = static_cast<int>(threadIdx.x) / 32;
  const bool is_consumer = static_cast<int>(threadIdx.x) < kConsumerThreads;

#if W4A8_EMPTY_FAMILY_EARLY_EXIT
  if (tile_prefix[bucket_experts] == 0) {
    return;
  }
#endif

  if (threadIdx.x == kConsumerThreads) {
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&activation_map));
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&payload_map));
#if W4A8_RESIDUAL_TMA
    cute::prefetch_tma_descriptor(reinterpret_cast<const cute::TmaDescriptor*>(&residual_map));
#endif
  }

  if constexpr (!std::is_same_v<Output, float>) {
    if (is_consumer) {
      cutlass::arch::warpgroup_reg_alloc<kConsumerRegisters>();
    } else {
      cutlass::arch::warpgroup_reg_dealloc<Traits::kProducerRegisters>();
    }
  }

  while (true) {
    if (threadIdx.x == 0) {
      const uint64_t task_index = atomicAdd(task_counter, 1ULL);
      storage.task = map_grouped_task<BlockM, BlockN>(task_index, source_offsets, expert_mapping,
                                                      tile_prefix, bucket_experts, launch_n_tiles,
                                                      n_tile_begin, row_capacity);
    }
    __syncthreads();
    if (!storage.task.valid) {
      return;
    }

    if (threadIdx.x == kConsumerThreads) {
#pragma unroll
      for (int stage = 0; stage < kStages; ++stage) {
        reinterpret_cast<FullBarrier*>(&storage.raw_full[stage])->init(1);
#if W4A8_SINGLE_READY
        reinterpret_cast<Barrier*>(&storage.decoded_ready[stage])->init(1);
#else
        reinterpret_cast<Barrier*>(&storage.decoded_ready[stage])
            ->init(decoded_writer_threads<BlockN>());
#endif
        reinterpret_cast<Barrier*>(&storage.empty[stage])->init(kConsumerWarps);
      }
      cutlass::arch::fence_view_async_shared();
    }
    __syncthreads();

    const int k_stages = padded_k / kBlockK;
    if (!is_consumer) {
      const int producer_thread = static_cast<int>(threadIdx.x) - kConsumerThreads;
#if W4A8_OVERLAP
      if (producer_thread == 0) {
        reinterpret_cast<Barrier*>(&storage.empty[0])->wait(1);
        issue_tma_stage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
            storage, storage.task, 0, padded_n, padded_k, activation_map, payload_map
#if W4A8_RESIDUAL_TMA
            ,
            residual_map
#endif
        );
      }
#endif
      for (int k_stage = 0; k_stage < k_stages; ++k_stage) {
        const int stage = k_stage % kStages;
        const int generation = k_stage / kStages;
        auto* raw_full = reinterpret_cast<FullBarrier*>(&storage.raw_full[stage]);
        auto* decoded_ready = reinterpret_cast<Barrier*>(&storage.decoded_ready[stage]);
#if !W4A8_OVERLAP
        reinterpret_cast<Barrier*>(&storage.empty[stage])->wait((generation + 1) & 1);
        if (producer_thread == 0) {
          issue_tma_stage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
              storage, storage.task, k_stage, padded_n, padded_k, activation_map, payload_map
#if W4A8_RESIDUAL_TMA
              ,
              residual_map
#endif
          );
        }
#endif
        raw_full->wait(generation & 1);
#if W4A8_OVERLAP
        if (producer_thread == 0 && k_stage + 1 < k_stages) {
          const int next_k_stage = k_stage + 1;
          const int next_stage = next_k_stage % kStages;
          const int next_generation = next_k_stage / kStages;
          reinterpret_cast<Barrier*>(&storage.empty[next_stage])->wait((next_generation + 1) & 1);
          issue_tma_stage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
              storage, storage.task, next_k_stage, padded_n, padded_k, activation_map, payload_map
#if W4A8_RESIDUAL_TMA
              ,
              residual_map
#endif
          );
        }
#endif
        const bool wrote =
#if W4A8_RESIDUAL_TMA
#if W4A8_PAYLOAD_V4
            producer_decode_stage<BlockN, Scheme>(storage.raw_payload[stage],
                                                  &storage.residual[stage][0][0],
                                                  storage.decoded_weight[stage], producer_thread);
#else
            producer_decode_stage<BlockN, Scheme>(storage.raw_payload[stage],
                                                  &storage.residual[stage][0][0][0],
                                                  storage.decoded_weight[stage], producer_thread);
#endif
#else
            producer_decode_global_stage<BlockN, Scheme>(
                storage.task, storage.raw_payload[stage], storage.decoded_weight[stage], k_stage,
                producer_thread, padded_k, padded_n, residual);
#endif
        if (wrote) {
          fence_decoded_writer();
        }
        cutlass::arch::NamedBarrier(kProducerThreads, kProducerNamedBarrier).sync();
#if W4A8_SINGLE_READY
        if (producer_thread == 0) {
          decoded_ready->arrive();
        }
#else
        if (wrote) {
          decoded_ready->arrive();
        }
#endif
      }
    } else {
      float final_accum[WGMMA::kNumAccum] = {0.0F};
#if W4A8_SINGLE_PARTIAL
      float partial[WGMMA::kNumAccum];
#else
      float partial[2][WGMMA::kNumAccum];
#endif
#if W4A8_CROSS_STAGE_RETIRE
      float pending_activation_scale0 = 0.0F;
      float pending_activation_scale1 = 0.0F;
#endif
      const int math_wg = static_cast<int>(threadIdx.x) / 128;
      const int row0 = warp * 16 + lane / 4;
      const int row1 = row0 + 8;
      const int64_t global_row0 = storage.task.m_begin + row0;
      const int64_t global_row1 = storage.task.m_begin + row1;
      const bool row0_valid = global_row0 < storage.task.m_end;
      const bool row1_valid = global_row1 < storage.task.m_end;

#if W4A8_CROSS_STAGE_RETIRE
      for (int k_stage = 0; k_stage < k_stages; ++k_stage) {
        const int stage = k_stage % kStages;
        const int generation = k_stage / kStages;
        reinterpret_cast<Barrier*>(&storage.decoded_ready[stage])->wait(generation & 1);

#pragma unroll
        for (int group = 0; group < kGroupsPerStage; ++group) {
          const int global_group = k_stage * kGroupsPerStage + group;
          const int current_slot = global_group & 1;
          // Mark the slot defined without emitting instructions; the group's
          // first WGMMA runs with scale_d=false and overwrites every value.
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum; ++i) {
            asm volatile("" : "=f"(partial[current_slot][i]));
          }
          deep_gemm::warpgroup_arrive();
#pragma unroll
          for (int mma = 0; mma < kMmaPerGroup; ++mma) {
            const int k_local = group * GroupSize + mma * WGMMA::K;
            const auto desc_a = deep_gemm::make_smem_desc(
                storage.activation[stage] + math_wg * WGMMA::M * kBlockK + k_local, 1);
            const auto desc_b =
                deep_gemm::make_smem_desc(storage.decoded_weight[stage] + k_local, 1);
            WGMMA::wgmma(desc_a, desc_b, partial[current_slot], mma != 0);
          }
          deep_gemm::warpgroup_commit_batch();
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum; ++i) {
            deep_gemm::warpgroup_fence_operand(partial[current_slot][i]);
          }

          if (global_group != 0) {
            deep_gemm::warpgroup_wait<1>();
            const int retired_slot = current_slot ^ 1;
#pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum; ++i) {
              deep_gemm::warpgroup_fence_operand(partial[retired_slot][i]);
            }
            const int retired_group = global_group - 1;
            const int retired_k_stage = retired_group / kGroupsPerStage;
            const int retired_stage = retired_k_stage % kStages;
            const int retired_local_group = retired_group % kGroupsPerStage;
            const float* retired_group_scales =
                quant_group_scales<BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
                    storage, storage.task, retired_group, retired_stage, retired_local_group,
                    padded_n, padded_k, group_scales);
            retire_quant_group<BlockN, GroupSize>(
                final_accum, partial[retired_slot], pending_activation_scale0,
                pending_activation_scale1, retired_group_scales, retired_group,
                storage.last_group[retired_stage], &storage.empty[retired_stage], lane);
          }

          if (group == 0) {
            const int64_t activation_scale_base =
                static_cast<int64_t>(k_stage) * activation_scale_stride +
                storage.task.padded_m_begin;
            pending_activation_scale0 =
                row0_valid ? __ldg(activation_scales + activation_scale_base + row0) : 0.0F;
            pending_activation_scale1 =
                row1_valid ? __ldg(activation_scales + activation_scale_base + row1) : 0.0F;
          }
        }
      }

      deep_gemm::warpgroup_wait<0>();
      const int final_group = k_stages * kGroupsPerStage - 1;
      const int final_k_stage = final_group / kGroupsPerStage;
      const int final_stage = final_k_stage % kStages;
      const int final_local_group = final_group % kGroupsPerStage;
#pragma unroll
      for (int i = 0; i < WGMMA::kNumAccum; ++i) {
        deep_gemm::warpgroup_fence_operand(partial[final_group & 1][i]);
      }
      const float* final_group_scales =
          quant_group_scales<BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
              storage, storage.task, final_group, final_stage, final_local_group, padded_n,
              padded_k, group_scales);
      retire_quant_group<BlockN, GroupSize>(
          final_accum, partial[final_group & 1], pending_activation_scale0,
          pending_activation_scale1, final_group_scales, final_group,
          storage.last_group[final_stage], &storage.empty[final_stage], lane);
#else
      for (int k_stage = 0; k_stage < k_stages; ++k_stage) {
        const int stage = k_stage % kStages;
        const int generation = k_stage / kStages;
        reinterpret_cast<Barrier*>(&storage.decoded_ready[stage])->wait(generation & 1);
        const int64_t activation_scale_base =
            static_cast<int64_t>(k_stage) * activation_scale_stride + storage.task.padded_m_begin;
        const float activation_scale0 =
            row0_valid ? __ldg(activation_scales + activation_scale_base + row0) : 0.0F;
        const float activation_scale1 =
            row1_valid ? __ldg(activation_scales + activation_scale_base + row1) : 0.0F;

#pragma unroll
        for (int group = 0; group < kGroupsPerStage; ++group) {
          const int global_group = k_stage * kGroupsPerStage + group;
#if W4A8_SINGLE_PARTIAL
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum; ++i) {
            asm volatile("" : "=f"(partial[i]));
          }
#else
          const int current_slot = group & 1;
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum; ++i) {
            asm volatile("" : "=f"(partial[current_slot][i]));
          }
#endif
          deep_gemm::warpgroup_arrive();
#pragma unroll
          for (int mma = 0; mma < kMmaPerGroup; ++mma) {
            const int k_local = group * GroupSize + mma * WGMMA::K;
            const auto desc_a = deep_gemm::make_smem_desc(
                storage.activation[stage] + math_wg * WGMMA::M * kBlockK + k_local, 1);
            const auto desc_b =
                deep_gemm::make_smem_desc(storage.decoded_weight[stage] + k_local, 1);
#if W4A8_SINGLE_PARTIAL
            WGMMA::wgmma(desc_a, desc_b, partial, mma != 0);
#else
            WGMMA::wgmma(desc_a, desc_b, partial[current_slot], mma != 0);
#endif
          }
          deep_gemm::warpgroup_commit_batch();

#if W4A8_SINGLE_PARTIAL
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum; ++i) {
            deep_gemm::warpgroup_fence_operand(partial[i]);
          }
          deep_gemm::warpgroup_wait<0>();
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum; ++i) {
            deep_gemm::warpgroup_fence_operand(partial[i]);
          }
          const float* current_group_scales =
              quant_group_scales<BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
                  storage, storage.task, global_group, stage, group, padded_n, padded_k,
                  group_scales);
          retire_quant_group<BlockN, GroupSize>(
              final_accum, partial, activation_scale0, activation_scale1, current_group_scales,
              global_group, (k_stage + 1) * kGroupsPerStage - 1, &storage.empty[stage], lane);
#else
#pragma unroll
          for (int i = 0; i < WGMMA::kNumAccum; ++i) {
            deep_gemm::warpgroup_fence_operand(partial[current_slot][i]);
          }
          if (group != 0) {
            deep_gemm::warpgroup_wait<1>();
            const int retired_slot = current_slot ^ 1;
#pragma unroll
            for (int i = 0; i < WGMMA::kNumAccum; ++i) {
              deep_gemm::warpgroup_fence_operand(partial[retired_slot][i]);
            }
            const int retired_group = global_group - 1;
            const int retired_local_group = group - 1;
            const float* retired_group_scales =
                quant_group_scales<BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
                    storage, storage.task, retired_group, stage, retired_local_group, padded_n,
                    padded_k, group_scales);
            accumulate_scaled_partial<BlockN, GroupSize>(final_accum, partial[retired_slot],
                                                         activation_scale0, activation_scale1,
                                                         retired_group_scales, lane);
          }
#endif
        }

#if !W4A8_SINGLE_PARTIAL
        deep_gemm::warpgroup_wait<0>();
        constexpr int kFinalLocalGroup = kGroupsPerStage - 1;
        const int final_slot = kFinalLocalGroup & 1;
#pragma unroll
        for (int i = 0; i < WGMMA::kNumAccum; ++i) {
          deep_gemm::warpgroup_fence_operand(partial[final_slot][i]);
        }
        const int final_group = (k_stage + 1) * kGroupsPerStage - 1;
        const float* final_group_scales =
            quant_group_scales<BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
                storage, storage.task, final_group, stage, kFinalLocalGroup, padded_n, padded_k,
                group_scales);
        retire_quant_group<BlockN, GroupSize>(final_accum, partial[final_slot], activation_scale0,
                                              activation_scale1, final_group_scales, final_group,
                                              final_group, &storage.empty[stage], lane);
#endif
      }
#endif

      const float global_alpha = __ldg(alpha + (alpha_per_expert ? storage.task.bucket_expert : 0));
#pragma unroll
      for (int i = 0; i < WGMMA::kNumAccum / 4; ++i) {
        const int column0 = i * 8 + (lane & 3) * 2;
        const int column1 = column0 + 1;
        const int64_t output_row0 = storage.task.m_begin + row0;
        const int64_t output_row1 = storage.task.m_begin + row1;
        const int output_column0 = storage.task.n_begin + column0;
        const int output_column1 = storage.task.n_begin + column1;
        if (output_row0 < storage.task.m_end && output_column0 < logical_n) {
          store_output_value(output, output_row0 * logical_n + output_column0,
                             final_accum[i * 4 + 0] * global_alpha);
        }
        if (output_row0 < storage.task.m_end && output_column1 < logical_n) {
          store_output_value(output, output_row0 * logical_n + output_column1,
                             final_accum[i * 4 + 1] * global_alpha);
        }
        if (output_row1 < storage.task.m_end && output_column0 < logical_n) {
          store_output_value(output, output_row1 * logical_n + output_column0,
                             final_accum[i * 4 + 2] * global_alpha);
        }
        if (output_row1 < storage.task.m_end && output_column1 < logical_n) {
          store_output_value(output, output_row1 * logical_n + output_column1,
                             final_accum[i * 4 + 3] * global_alpha);
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == kConsumerThreads) {
#pragma unroll
      for (int stage = 0; stage < kStages; ++stage) {
        FullBarrier::invalidate(&storage.raw_full[stage]);
        Barrier::invalidate(&storage.decoded_ready[stage]);
        Barrier::invalidate(&storage.empty[stage]);
      }
    }
    __syncthreads();
  }
#else
  if (threadIdx.x == 0) {
    asm volatile("trap;\n");
  }
#endif
}

#if W4A8_RESIDUAL_TMA
#define FLASHINFER_SM90_W4A8_RESIDUAL_PARAMETER __grid_constant__ const CUtensorMap residual_map
#define FLASHINFER_SM90_W4A8_RESIDUAL_ARGUMENT residual_map
#else
#define FLASHINFER_SM90_W4A8_RESIDUAL_PARAMETER \
  const typename ResidualDecoder<Scheme>::Storage* residual
#define FLASHINFER_SM90_W4A8_RESIDUAL_ARGUMENT residual
#endif

#define FLASHINFER_SM90_W4A8_KERNEL_PARAMETERS(OutputType)                                         \
  OutputType *output, const float *activation_scales, const float *alpha,                          \
      const int32_t *expert_mapping, const int64_t *source_offsets, const int64_t *tile_prefix,    \
      unsigned long long *task_counter, int64_t row_capacity, int32_t logical_n, int32_t padded_n, \
      int32_t padded_k, int32_t launch_n_tiles, int32_t n_tile_begin, int32_t bucket_experts,      \
      int64_t activation_scale_stride, bool alpha_per_expert,                                      \
      __grid_constant__ const CUtensorMap activation_map,                                          \
      __grid_constant__ const CUtensorMap payload_map, FLASHINFER_SM90_W4A8_RESIDUAL_PARAMETER,    \
      const float *group_scales

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme, int PipelineStages>
__global__ __launch_bounds__(
    W4A8LaunchTraits<BlockM, BlockN, PipelineStages>::kThreads,
    W4A8LaunchTraits<BlockM, BlockN, PipelineStages>::
        kMinBlocksPerSm) void grouped_w4a8_bf16_kernel(FLASHINFER_SM90_W4A8_KERNEL_PARAMETERS(__nv_bfloat16)) {
  extern __shared__ __align__(1024) uint8_t dynamic_shared[];
  auto& storage =
      *reinterpret_cast<W4A8SharedStorage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>*>(
          dynamic_shared);
  grouped_w4a8_kernel_body<__nv_bfloat16, BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
      storage, output, activation_scales, alpha, expert_mapping, source_offsets, tile_prefix,
      task_counter, row_capacity, logical_n, padded_n, padded_k, launch_n_tiles, n_tile_begin,
      bucket_experts, activation_scale_stride, alpha_per_expert, activation_map, payload_map,
      FLASHINFER_SM90_W4A8_RESIDUAL_ARGUMENT, group_scales);
}

template <int BlockM, int BlockN, int GroupSize, ResidualScheme Scheme, int PipelineStages>
__global__ __launch_bounds__(
    W4A8LaunchTraits<BlockM, BlockN, PipelineStages>::kThreads,
    W4A8LaunchTraits<BlockM, BlockN, PipelineStages>::
        kDebugMinBlocksPerSm) void grouped_w4a8_fp32_debug_kernel(FLASHINFER_SM90_W4A8_KERNEL_PARAMETERS(float)) {
  extern __shared__ __align__(1024) uint8_t dynamic_shared[];
  auto& storage =
      *reinterpret_cast<W4A8SharedStorage<BlockM, BlockN, GroupSize, Scheme, PipelineStages>*>(
          dynamic_shared);
  grouped_w4a8_kernel_body<float, BlockM, BlockN, GroupSize, Scheme, PipelineStages>(
      storage, output, activation_scales, alpha, expert_mapping, source_offsets, tile_prefix,
      task_counter, row_capacity, logical_n, padded_n, padded_k, launch_n_tiles, n_tile_begin,
      bucket_experts, activation_scale_stride, alpha_per_expert, activation_map, payload_map,
      FLASHINFER_SM90_W4A8_RESIDUAL_ARGUMENT, group_scales);
}

#undef FLASHINFER_SM90_W4A8_KERNEL_PARAMETERS
#undef FLASHINFER_SM90_W4A8_RESIDUAL_ARGUMENT
#undef FLASHINFER_SM90_W4A8_RESIDUAL_PARAMETER

struct alignas(1024) DebugDecodeSharedStorage {
  alignas(1024) uint8_t raw_payload[kV3PayloadTileN * kBlockK / 2];
  alignas(1024) uint8_t decoded_weight[kV3PayloadTileN * kBlockK];
  GroupedTask task;
};

__host__ __device__ constexpr size_t debug_decode_smem_bytes() {
  return sizeof(DebugDecodeSharedStorage);
}

static_assert(debug_decode_smem_bytes() <= kSm90OptInSharedMemoryBytes);

template <ResidualScheme Scheme>
__global__ __launch_bounds__(kProducerThreads, 1) void debug_decode_v3_kernel(
    uint8_t* decoded, const uint8_t* packed,
    const typename ResidualDecoder<Scheme>::Storage* residual, int32_t experts, int32_t k_tiles,
    int32_t n_tiles) {
  extern __shared__ __align__(1024) uint8_t dynamic_shared[];
  auto& storage = *reinterpret_cast<DebugDecodeSharedStorage*>(dynamic_shared);
  if (experts <= 0 || n_tiles <= 0 || k_tiles <= 0 || k_tiles % 4 != 0) {
    return;
  }

  const int32_t k_stages = k_tiles / (kBlockK / kV3PayloadTileK);
  const int64_t tasks = static_cast<int64_t>(experts) * k_stages * n_tiles;
  for (int64_t task_index = blockIdx.x; task_index < tasks; task_index += gridDim.x) {
    int64_t coordinate = task_index;
    const int32_t n_tile = coordinate % n_tiles;
    coordinate /= n_tiles;
    const int32_t k_stage = coordinate % k_stages;
    const int32_t expert = coordinate / k_stages;

    constexpr int kRawBytes = kV3PayloadTileN * kBlockK / 2;
    for (int raw = static_cast<int>(threadIdx.x); raw < kRawBytes; raw += kProducerThreads) {
#if W4A8_PAYLOAD_V4
      const int32_t n_in_tile = raw / (kBlockK / 2);
      const int32_t packed_k = raw % (kBlockK / 2);
      const int64_t packed_index =
          ((static_cast<int64_t>(expert) * k_stages + k_stage) * (n_tiles * kV3PayloadTileN) +
           n_tile * kV3PayloadTileN + n_in_tile) *
              (kBlockK / 2) +
          packed_k;
      storage.raw_payload[raw] = packed[packed_index];
#else
      const int32_t k32_in_stage = raw / (kV3PayloadTileN * kV3PackedBytesPerRow);
      const int32_t in_k32 = raw % (kV3PayloadTileN * kV3PackedBytesPerRow);
      const int32_t n_in_tile = in_k32 / kV3PackedBytesPerRow;
      const int32_t packed_k = in_k32 % kV3PackedBytesPerRow;
      storage.raw_payload[raw] = packed[v3_payload_offset(
          expert, k_stage * 4 + k32_in_stage, n_tile, n_in_tile, packed_k, k_tiles, n_tiles)];
#endif
    }
    if (threadIdx.x == 0) {
      storage.task = GroupedTask{expert, expert, n_tile * kV3PayloadTileN, 0, 0, 0, true};
    }
    __syncthreads();

    producer_decode_global_stage<kV3PayloadTileN, Scheme>(
        storage.task, storage.raw_payload, storage.decoded_weight, k_stage,
        static_cast<int>(threadIdx.x), k_tiles * kV3PayloadTileK, n_tiles * kV3PayloadTileN,
        residual);
    __syncthreads();

    constexpr int kDecodedBytes = kV3PayloadTileN * kBlockK;
    for (int logical = static_cast<int>(threadIdx.x); logical < kDecodedBytes;
         logical += kProducerThreads) {
      const int32_t n_in_tile = logical / kBlockK;
      const int32_t k_in_stage = logical % kBlockK;
      const int32_t k32_in_stage = k_in_stage / kV3PayloadTileK;
      const int32_t k_in_tile = k_in_stage % kV3PayloadTileK;
      const int32_t k_tile = k_stage * 4 + k32_in_stage;
      const int64_t output_index =
          v3_debug_operand_offset(expert, k_tile, n_tile, n_in_tile, k_in_tile, k_tiles, n_tiles);
      decoded[output_index] =
          storage.decoded_weight[wgmma_swizzle_128b_offset(n_in_tile, k_in_stage)];
    }
    __syncthreads();
  }
}

}  // namespace sm90_w4a8
}  // namespace flashinfer
