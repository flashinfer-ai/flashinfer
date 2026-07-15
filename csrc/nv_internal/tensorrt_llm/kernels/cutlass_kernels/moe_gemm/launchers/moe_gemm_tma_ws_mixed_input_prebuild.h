/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <cstdint>
#include <type_traits>

#include "../../include/moe_gemm_kernels.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass_extensions/gemm/kernel/sm90_tile_scheduler_group_precomputed.hpp"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels_oss {
namespace detail {

using namespace cute;

using TmaWarpSpecializedGroupedGemmInput =
    tensorrt_llm::kernels::cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;
using PrecomputedWorkTileCodec = cutlass::gemm::kernel::detail::PrecomputedGroupWorkTile;

static constexpr int kPrecomputedSchedulerThreads = 128;
static constexpr int kPrecomputedSchedulerMaxSwizzle = 2;
static constexpr uint64_t kPrecomputedSchedulerSentinelTiles = 4096;

CUTLASS_HOST_DEVICE uint64_t div_round_up(uint64_t value, uint64_t divisor) {
  return (value + divisor - 1) / divisor;
}

CUTLASS_HOST_DEVICE uint64_t round_up_to_multiple(uint64_t value, uint64_t multiple) {
  return div_round_up(value, multiple) * multiple;
}

inline size_t align_bytes(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

inline int log_swizzle_size(uint64_t problem_blocks_m, uint64_t problem_blocks_n,
                            int max_swizzle_size) {
  int const min_cta_dim =
      static_cast<int>(problem_blocks_m < problem_blocks_n ? problem_blocks_m : problem_blocks_n);
  if (max_swizzle_size >= 8 && min_cta_dim >= 6) {
    return 3;
  }
  if (max_swizzle_size >= 4 && min_cta_dim >= 3) {
    return 2;
  }
  if (max_swizzle_size >= 2 && min_cta_dim >= 2) {
    return 1;
  }
  return 0;
}

template <int TileShapeM, int TileShapeN, int ClusterShapeM, int ClusterShapeN>
inline uint64_t max_work_tiles_from_total_tokens(int num_experts, int64_t total_routed_tokens,
                                                 int64_t channels, int max_swizzle_size) {
  if (num_experts <= 0 || total_routed_tokens <= 0 || channels <= 0) {
    return 0;
  }

  int const swizzle_log =
      log_swizzle_size(static_cast<uint64_t>(total_routed_tokens), 1, max_swizzle_size);
  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const channel_multiple = swizzle * uint64_t(ClusterShapeM);
  uint64_t const token_tile_group = swizzle * uint64_t(ClusterShapeN);
  uint64_t const tokens_per_padded_group = uint64_t(TileShapeN) * token_tile_group;
  uint64_t const channel_tiles =
      div_round_up(static_cast<uint64_t>(channels), uint64_t(TileShapeM));
  uint64_t const padded_channel_tiles = round_up_to_multiple(channel_tiles, channel_multiple);
  uint64_t const total_tokens = static_cast<uint64_t>(total_routed_tokens);
  uint64_t const nonempty_experts =
      total_tokens < uint64_t(num_experts) ? total_tokens : uint64_t(num_experts);
  uint64_t const extra_tokens = total_tokens - nonempty_experts;
  uint64_t const max_token_tile_rows =
      token_tile_group * (nonempty_experts + extra_tokens / tokens_per_padded_group);

  return padded_channel_tiles * max_token_tile_rows;
}

inline size_t precomputed_scheduler_workspace_size(int num_experts, int64_t total_routed_tokens,
                                                   int64_t max_channels) {
  uint64_t const regular_max_work_tiles = max_work_tiles_from_total_tokens<64, 16, 2, 2>(
      num_experts, total_routed_tokens, max_channels, kPrecomputedSchedulerMaxSwizzle);
  uint64_t const single_warpgroup_max_work_tiles = max_work_tiles_from_total_tokens<128, 8, 1, 1>(
      num_experts, total_routed_tokens, max_channels, kPrecomputedSchedulerMaxSwizzle);
  uint64_t const max_work_tiles = regular_max_work_tiles > single_warpgroup_max_work_tiles
                                      ? regular_max_work_tiles
                                      : single_warpgroup_max_work_tiles;
  // Chunk-major storage rounds every worker chunk up independently and appends
  // one sentinel per worker. Its capacity is strictly below max_tiles + 2 * workers.
  uint64_t const work_tile_capacity = max_work_tiles + 2 * kPrecomputedSchedulerSentinelTiles;

  size_t bytes = 0;
  bytes += align_bytes(size_t(work_tile_capacity) * sizeof(uint64_t), 128);
  bytes += align_bytes(sizeof(cute::TmaDescriptor), 128);
  bytes +=
      align_bytes(size_t(num_experts > 0 ? num_experts : 1) * sizeof(cute::TmaDescriptor), 128);
  return bytes;
}

struct PrecomputedSchedulerWorkspace {
  uint64_t* work_tiles = nullptr;
  cute::TmaDescriptor* prebuilt_tma_desc_A = nullptr;
  cute::TmaDescriptor* prebuilt_tma_desc_B = nullptr;
  size_t required_bytes = 0;
  dim3 gemm_grid_shape = dim3(1, 1, 1);
  uint32_t work_tiles_per_worker = 0;
};

template <int TileShapeM, int TileShapeN, int ClusterShapeM, int ClusterShapeN,
          bool ChunkMajorWorkMap = false>
inline PrecomputedSchedulerWorkspace partition_precomputed_scheduler_workspace(
    TmaWarpSpecializedGroupedGemmInput const& hopper_inputs, int num_experts,
    int64_t total_routed_tokens, int64_t channels, int sm_count) {
  using ProblemShape = TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt;
  using Scheduler =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90GroupPrecomputed<ProblemShape, 8,
                                                                                 ChunkMajorWorkMap>;
  using SchedulerParams = typename Scheduler::Params;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count = sm_count;
  cutlass::gemm::GemmCoord cluster_shape(ClusterShapeM, ClusterShapeN, 1);
  dim3 const problem_blocks =
      SchedulerParams::get_tiled_cta_shape_mnl(cluster_shape, static_cast<uint32_t>(sm_count), 1);
  dim3 const gemm_grid_shape = SchedulerParams::get_grid_shape(
      problem_blocks, cluster_shape, hw_info, kPrecomputedSchedulerMaxSwizzle,
      SchedulerParams::RasterOrderOptions::AlongM, true);

  uint64_t const max_work_tiles =
      max_work_tiles_from_total_tokens<TileShapeM, TileShapeN, ClusterShapeM, ClusterShapeN>(
          num_experts, total_routed_tokens, channels, kPrecomputedSchedulerMaxSwizzle);
  uint64_t const sentinel_count =
      uint64_t(gemm_grid_shape.x) * uint64_t(gemm_grid_shape.y) * uint64_t(gemm_grid_shape.z);
  TLLM_CHECK_WITH_INFO(sentinel_count > 0,
                       "Precomputed scheduler requires at least one logical worker.");
  TLLM_CHECK_WITH_INFO(gemm_grid_shape.z == 1,
                       "Precomputed grouped scheduler work-map requires a 2D launch grid.");
  uint32_t work_tiles_per_worker = 0;
  uint64_t work_tile_capacity = max_work_tiles + sentinel_count;
  if constexpr (ChunkMajorWorkMap) {
    TLLM_CHECK_WITH_INFO(
        sentinel_count <= kPrecomputedSchedulerSentinelTiles,
        "Single-warpgroup logical worker count exceeds precomputed scheduler workspace bound.");
    uint64_t const work_tiles_per_worker_u64 =
        (max_work_tiles + sentinel_count - 1) / sentinel_count + 1;
    TLLM_CHECK_WITH_INFO(work_tiles_per_worker_u64 <= uint64_t(0xffffffffu),
                         "Precomputed scheduler worker chunk exceeds uint32_t index range.");
    work_tiles_per_worker = static_cast<uint32_t>(work_tiles_per_worker_u64);
    work_tile_capacity = sentinel_count * uint64_t(work_tiles_per_worker);
  }
  TLLM_CHECK_WITH_INFO(work_tile_capacity <= uint64_t(0xffffffffu),
                       "Precomputed scheduler work-map exceeds uint32_t index range.");

  size_t const work_tiles_bytes = align_bytes(size_t(work_tile_capacity) * sizeof(uint64_t), 128);
  size_t const prebuilt_a_bytes = align_bytes(sizeof(cute::TmaDescriptor), 128);
  size_t const prebuilt_b_bytes =
      align_bytes(size_t(num_experts > 0 ? num_experts : 1) * sizeof(cute::TmaDescriptor), 128);
  size_t const required_bytes = work_tiles_bytes + prebuilt_a_bytes + prebuilt_b_bytes;

  TLLM_CHECK_WITH_INFO(
      hopper_inputs.precomputed_scheduler_workspace != nullptr,
      "Precomputed scheduler workspace must be configured for mixed dtype TMA WS GEMM.");
  TLLM_CHECK_WITH_INFO(
      required_bytes <= hopper_inputs.precomputed_scheduler_workspace_size,
      "Precomputed scheduler workspace is too small for selected mixed dtype TMA WS GEMM config.");
  TLLM_CHECK_WITH_INFO(num_experts <= int(PrecomputedWorkTileCodec::ExpertMask + 1),
                       "Precomputed scheduler work-map expert index exceeds packed limit.");

  auto* base = hopper_inputs.precomputed_scheduler_workspace;
  PrecomputedSchedulerWorkspace workspace;
  workspace.work_tiles = reinterpret_cast<uint64_t*>(base);
  workspace.prebuilt_tma_desc_A = reinterpret_cast<cute::TmaDescriptor*>(base + work_tiles_bytes);
  workspace.prebuilt_tma_desc_B =
      reinterpret_cast<cute::TmaDescriptor*>(base + work_tiles_bytes + prebuilt_a_bytes);
  workspace.required_bytes = required_bytes;
  workspace.gemm_grid_shape = gemm_grid_shape;
  workspace.work_tiles_per_worker = work_tiles_per_worker;
  return workspace;
}

template <int ClusterShapeM, int ClusterShapeN>
__device__ __forceinline__ uint64_t make_work_tile_static(uint64_t global_linear_idx,
                                                          uint64_t local_linear_idx, int group_idx,
                                                          uint64_t problem_blocks_m,
                                                          int swizzle_log, int gemm_grid_x,
                                                          int gemm_grid_y) {
  uint64_t const cluster_shape_major = uint64_t(ClusterShapeM);
  uint64_t const cluster_shape_minor = uint64_t(ClusterShapeN);
  uint64_t const total_grid_size = uint64_t(gemm_grid_x) * uint64_t(gemm_grid_y);
  uint64_t const worker_id = total_grid_size == 0 ? 0 : global_linear_idx % total_grid_size;
  uint64_t const cluster_minor_offset = worker_id % uint64_t(gemm_grid_y);

  uint64_t const blk_per_grid_dim = local_linear_idx / cluster_shape_minor;
  uint64_t const cluster_id = blk_per_grid_dim / cluster_shape_major;
  uint64_t const cluster_major_offset = blk_per_grid_dim % cluster_shape_major;

  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const offset = cluster_id & (swizzle - 1);
  uint64_t const extra = cluster_id >> swizzle_log;
  uint64_t const curr_group_cluster_blk_major = problem_blocks_m / cluster_shape_major;
  uint64_t const cluster_idx_minor_div_swizzle = extra / curr_group_cluster_blk_major;
  uint64_t const cluster_idx_major = extra % curr_group_cluster_blk_major;
  uint64_t const cluster_idx_minor = cluster_idx_minor_div_swizzle * swizzle + offset;

  uint64_t const minor_work_idx = cluster_idx_minor * cluster_shape_minor + cluster_minor_offset;
  uint64_t const major_work_idx = cluster_idx_major * cluster_shape_major + cluster_major_offset;
  return PrecomputedWorkTileCodec::pack(major_work_idx, minor_work_idx, uint64_t(group_idx));
}

struct PrecomputedGroupInfo {
  uint64_t problem_blocks_m = 0;
  uint64_t group_tiles = 0;
};

template <int TileShapeM, int TileShapeN, int ClusterShapeM, int ClusterShapeN, class Problem>
__device__ __forceinline__ PrecomputedGroupInfo get_group_info_static(Problem const& problem,
                                                                      int swizzle_log) {
  uint64_t const ctas_along_m =
      (uint64_t(cute::get<0>(problem)) + uint64_t(TileShapeM) - 1) / uint64_t(TileShapeM);
  uint64_t const ctas_along_n =
      (uint64_t(cute::get<1>(problem)) + uint64_t(TileShapeN) - 1) / uint64_t(TileShapeN);
  uint64_t const swizzle = uint64_t(1) << swizzle_log;
  uint64_t const m_multiple = swizzle * uint64_t(ClusterShapeM);
  uint64_t const n_multiple = swizzle * uint64_t(ClusterShapeN);
  uint64_t const problem_blocks_m = round_up_to_multiple(ctas_along_m, m_multiple);
  uint64_t const problem_blocks_n = round_up_to_multiple(ctas_along_n, n_multiple);

  if (problem_blocks_m > uint64_t(PrecomputedWorkTileCodec::ChannelMask + 1) ||
      problem_blocks_n > uint64_t(PrecomputedWorkTileCodec::TokenMask + 1)) {
    asm volatile("trap;");
  }

  return {problem_blocks_m, problem_blocks_m * problem_blocks_n};
}

static constexpr int kPrebuiltTmaDescriptorScratchCount = 2;
static constexpr size_t kPrebuiltTmaDescriptorScratchBytes =
    kPrebuiltTmaDescriptorScratchCount * sizeof(cute::TmaDescriptor);
static constexpr int kPrebuiltTmaDescriptorSlotA = 0;
static constexpr int kPrebuiltTmaDescriptorSlotB = 1;
static constexpr int kPrebuiltTmaDescriptorWarpA = 0;
static constexpr int kPrebuiltTmaDescriptorWarpB = 1;

CUTE_DEVICE void publish_prebuilt_tma_descriptor(cute::TmaDescriptor const* gmem_desc_ptr,
                                                 cute::TmaDescriptor& smem_desc,
                                                 int publisher_warp) {
  if ((threadIdx.x >> 5) == publisher_warp) {
    __syncwarp();
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    cute::tma_descriptor_cp_fence_release(gmem_desc_ptr, smem_desc);
    __syncwarp();
  }
}

template <class MainloopParams, class Problem>
__device__ __forceinline__ void build_prebuilt_tma_descriptors(
    MainloopParams const& mainloop_params, Problem const& problem, int group,
    cute::TmaDescriptor* smem_tma_desc, cute::TmaDescriptor* prebuilt_tma_desc_A,
    cute::TmaDescriptor* prebuilt_tma_desc_B) {
  if (group == 0) {
    cute::TmaDescriptor& smem_desc = smem_tma_desc[kPrebuiltTmaDescriptorSlotA];
    if (threadIdx.x == kPrebuiltTmaDescriptorWarpA * 32) {
      constexpr int MaxTensorRank = 5;
      cute::array<uint32_t, MaxTensorRank> prob_shape_A = {1, 1, 1, 1, 1};
      cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0, 0, 0, 0, 0};
      using PtrA = std::remove_reference_t<decltype(mainloop_params.ptr_A[group])>;
      PtrA ptr_A = nullptr;
      uint32_t const M = static_cast<uint32_t>(cute::get<0>(problem));
      uint32_t const K = static_cast<uint32_t>(cute::get<2>(problem));
      auto dA_group = mainloop_params.ptr_dA[group];
      auto stride_m = cute::get<0>(dA_group);
      auto stride_k = cute::get<1>(dA_group);
      int64_t const term_m = static_cast<int64_t>(M) * static_cast<int64_t>(stride_m);
      int64_t const term_k = static_cast<int64_t>(K) * static_cast<int64_t>(stride_k);
      int64_t const stride_l = term_m > term_k ? term_m : term_k;
      auto full_layout =
          make_layout(make_shape(M, K, static_cast<uint32_t>(mainloop_params.num_groups)),
                      cute::make_stride(stride_m, stride_k, stride_l));
      Tensor tensor_a = make_tensor(ptr_A, full_layout);

      smem_desc = *mainloop_params.tma_load_a.get_tma_descriptor();
      cute::tma_descriptor_replace_addr_in_shared_mem(smem_desc, mainloop_params.ptr_A[0]);
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_a, tensor_a, prob_shape_A,
                                               prob_stride_A);

      using ElementA = std::remove_cv_t<std::remove_pointer_t<PtrA>>;
      for (uint64_t& stride : prob_stride_A) {
        stride = (stride * cutlass::sizeof_bits<ElementA>::value) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(smem_desc, prob_shape_A,
                                                              prob_stride_A);
    }
    publish_prebuilt_tma_descriptor(&prebuilt_tma_desc_A[0], smem_desc,
                                    kPrebuiltTmaDescriptorWarpA);
  }

  if (cute::get<1>(problem) == 0) {
    return;
  }

  {
    cute::TmaDescriptor& smem_desc = smem_tma_desc[kPrebuiltTmaDescriptorSlotB];
    if (threadIdx.x == kPrebuiltTmaDescriptorWarpB * 32) {
      constexpr int MaxTensorRank = 5;
      cute::array<uint32_t, MaxTensorRank> prob_shape_B = {1, 1, 1, 1, 1};
      cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0, 0, 0, 0, 0};
      using PtrB = std::remove_reference_t<decltype(mainloop_params.ptr_B[group])>;
      PtrB ptr_B = nullptr;
      uint32_t const N = static_cast<uint32_t>(cute::get<1>(problem));
      uint32_t const K = static_cast<uint32_t>(cute::get<2>(problem));
      auto dB_group = mainloop_params.ptr_dB[group];
      auto stride_n = cute::get<0>(dB_group);
      auto stride_k = cute::get<1>(dB_group);
      auto full_layout = make_layout(make_shape(N, K, uint32_t(1)),
                                     cute::make_stride(stride_n, stride_k, int64_t(0)));
      Tensor tensor_b = make_tensor(ptr_B, full_layout);

      smem_desc = *mainloop_params.tma_load_b.get_tma_descriptor();
      cute::tma_descriptor_replace_addr_in_shared_mem(smem_desc, mainloop_params.ptr_B[group]);
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b, prob_shape_B,
                                               prob_stride_B);

      using ElementB = std::remove_cv_t<std::remove_pointer_t<PtrB>>;
      for (uint64_t& stride : prob_stride_B) {
        stride = (stride * cutlass::sizeof_bits<ElementB>::value) / 8;
      }
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(smem_desc, prob_shape_B,
                                                              prob_stride_B);
    }
    publish_prebuilt_tma_descriptor(&prebuilt_tma_desc_B[group], smem_desc,
                                    kPrebuiltTmaDescriptorWarpB);
  }
}

template <int TileShapeM, int TileShapeN, int ClusterShapeM, int ClusterShapeN,
          bool ChunkMajorWorkMap, class Problem, class MainloopParams>
__global__ void build_precomputed_work_tile_map_kernel(
    Problem const* problem_shapes, int groups, int swizzle_log, int gemm_grid_x, int gemm_grid_y,
    uint32_t work_tiles_per_worker, uint64_t* work_tiles, MainloopParams mainloop_params,
    cute::TmaDescriptor* prebuilt_tma_desc_A, cute::TmaDescriptor* prebuilt_tma_desc_B) {
  int const tid = threadIdx.x;
  uint64_t const total_grid_size = uint64_t(gemm_grid_x) * uint64_t(gemm_grid_y);

  if (groups <= 0) {
    if (blockIdx.x == 0) {
      for (uint64_t i = uint64_t(tid); i < total_grid_size; i += uint64_t(blockDim.x)) {
        uint64_t const storage_idx = ChunkMajorWorkMap ? i * uint64_t(work_tiles_per_worker) : i;
        work_tiles[storage_idx] = PrecomputedWorkTileCodec::Invalid;
      }
    }
    return;
  }

  int const group = int(blockIdx.x);
  if (group >= groups) {
    return;
  }

  extern __shared__ __align__(64) unsigned char shared_storage[];
  cute::TmaDescriptor* smem_tma_desc = reinterpret_cast<cute::TmaDescriptor*>(shared_storage);
  unsigned long long* prefix_partials =
      reinterpret_cast<unsigned long long*>(shared_storage + kPrebuiltTmaDescriptorScratchBytes);
  unsigned long long* total_partials = nullptr;
  unsigned long long* group_info_storage = nullptr;
  if constexpr (ChunkMajorWorkMap) {
    total_partials = prefix_partials + blockDim.x;
    group_info_storage = total_partials + blockDim.x;
  } else {
    group_info_storage = prefix_partials + blockDim.x;
  }

  if (tid == 0) {
    PrecomputedGroupInfo const info =
        get_group_info_static<TileShapeM, TileShapeN, ClusterShapeM, ClusterShapeN>(
            problem_shapes[group], swizzle_log);
    group_info_storage[0] = static_cast<unsigned long long>(info.problem_blocks_m);
    group_info_storage[1] = static_cast<unsigned long long>(info.group_tiles);
  }

  build_prebuilt_tma_descriptors(mainloop_params, problem_shapes[group], group, smem_tma_desc,
                                 prebuilt_tma_desc_A, prebuilt_tma_desc_B);

  uint64_t prefix_sum = 0;
  if constexpr (ChunkMajorWorkMap) {
    uint64_t total_sum = 0;
    for (int scan_group = tid; scan_group < groups; scan_group += blockDim.x) {
      PrecomputedGroupInfo const info =
          get_group_info_static<TileShapeM, TileShapeN, ClusterShapeM, ClusterShapeN>(
              problem_shapes[scan_group], swizzle_log);
      total_sum += info.group_tiles;
      if (scan_group < group) {
        prefix_sum += info.group_tiles;
      }
    }
    total_partials[tid] = static_cast<unsigned long long>(total_sum);
  } else {
    for (int prefix_group = tid; prefix_group < group; prefix_group += blockDim.x) {
      PrecomputedGroupInfo const info =
          get_group_info_static<TileShapeM, TileShapeN, ClusterShapeM, ClusterShapeN>(
              problem_shapes[prefix_group], swizzle_log);
      prefix_sum += info.group_tiles;
    }
  }

  prefix_partials[tid] = static_cast<unsigned long long>(prefix_sum);
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      prefix_partials[tid] += prefix_partials[tid + offset];
      if constexpr (ChunkMajorWorkMap) {
        total_partials[tid] += total_partials[tid + offset];
      }
    }
    __syncthreads();
  }

  uint64_t const group_start = static_cast<uint64_t>(prefix_partials[0]);
  uint64_t const problem_blocks_m = static_cast<uint64_t>(group_info_storage[0]);
  uint64_t const group_tiles = static_cast<uint64_t>(group_info_storage[1]);
  uint64_t total_tiles = 0;
  uint64_t tiles_per_worker = 0;
  if constexpr (ChunkMajorWorkMap) {
    total_tiles = static_cast<uint64_t>(total_partials[0]);
    tiles_per_worker = total_tiles == 0 ? 1 : (total_tiles + total_grid_size - 1) / total_grid_size;
  }

  for (uint64_t local_tile = uint64_t(tid); local_tile < group_tiles;
       local_tile += uint64_t(blockDim.x)) {
    uint64_t const global_tile = group_start + local_tile;
    uint64_t storage_idx = global_tile;
    if constexpr (ChunkMajorWorkMap) {
      uint64_t const worker_idx = global_tile / tiles_per_worker;
      uint64_t const worker_tile_idx = global_tile % tiles_per_worker;
      storage_idx = worker_idx * uint64_t(work_tiles_per_worker) + worker_tile_idx;
    }
    work_tiles[storage_idx] = make_work_tile_static<ClusterShapeM, ClusterShapeN>(
        global_tile, local_tile, group, problem_blocks_m, swizzle_log, gemm_grid_x, gemm_grid_y);
  }

  if (group == groups - 1) {
    [[maybe_unused]] uint64_t const sentinel_start = group_start + group_tiles;
    for (uint64_t i = uint64_t(tid); i < total_grid_size; i += uint64_t(blockDim.x)) {
      if constexpr (ChunkMajorWorkMap) {
        uint64_t const worker_start = i * tiles_per_worker;
        uint64_t const worker_tile_count =
            worker_start < total_tiles
                ? ((total_tiles - worker_start < tiles_per_worker) ? total_tiles - worker_start
                                                                   : tiles_per_worker)
                : 0;
        work_tiles[i * uint64_t(work_tiles_per_worker) + worker_tile_count] =
            PrecomputedWorkTileCodec::Invalid;
      } else {
        work_tiles[sentinel_start + i] = PrecomputedWorkTileCodec::Invalid;
      }
    }
  }
}

template <int TileShapeM, int TileShapeN, int ClusterShapeM, int ClusterShapeN,
          bool ChunkMajorWorkMap = false, class Problem, class MainloopParams>
inline void build_precomputed_work_tile_map(PrecomputedSchedulerWorkspace const& workspace,
                                            Problem const* problem_shapes, int groups,
                                            int64_t total_routed_tokens, int64_t channels,
                                            MainloopParams const& mainloop_params,
                                            cudaStream_t stream) {
  uint64_t const max_work_tiles =
      max_work_tiles_from_total_tokens<TileShapeM, TileShapeN, ClusterShapeM, ClusterShapeN>(
          groups, total_routed_tokens, channels, kPrecomputedSchedulerMaxSwizzle);
  int const swizzle_log = log_swizzle_size(max_work_tiles, 1, kPrecomputedSchedulerMaxSwizzle);
  dim3 const scheduler_grid(groups > 0 ? groups : 1);
  size_t const scheduler_smem =
      kPrebuiltTmaDescriptorScratchBytes +
      size_t((ChunkMajorWorkMap ? kPrecomputedSchedulerThreads * 2 : kPrecomputedSchedulerThreads) +
             2) *
          sizeof(unsigned long long);
  build_precomputed_work_tile_map_kernel<TileShapeM, TileShapeN, ClusterShapeM, ClusterShapeN,
                                         ChunkMajorWorkMap, Problem, MainloopParams>
      <<<scheduler_grid, kPrecomputedSchedulerThreads, scheduler_smem, stream>>>(
          problem_shapes, groups, swizzle_log, workspace.gemm_grid_shape.x,
          workspace.gemm_grid_shape.y, workspace.work_tiles_per_worker, workspace.work_tiles,
          mainloop_params, workspace.prebuilt_tma_desc_A, workspace.prebuilt_tma_desc_B);
  TLLM_CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace cutlass_kernels_oss
}  // namespace kernels
}  // namespace tensorrt_llm
