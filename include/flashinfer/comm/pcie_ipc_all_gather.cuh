/*
 * Copyright 2025-2026 NVIDIA
 * Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/)
 * Copyright (c) 2026 by FlashInfer team.
 * Modifications Copyright (c) 2026 by the pcie_collectives contributors.
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
 *
 * This AllGather adaptation uses the CUDA-IPC signaling and device-selected
 * double-buffer slot protocol derived from b12x's Apache-2.0 PCIe two-shot
 * implementation, and the system-scope release/acquire protocol from
 * FlashInfer's Apache-2.0 pcie_ipc_all_reduce implementation:
 *   https://github.com/local-inference-lab/b12x
 *   https://github.com/flashinfer-ai/flashinfer/pull/4393
 */
#ifndef FLASHINFER_COMM_PCIE_IPC_ALL_GATHER_CUH_
#define FLASHINFER_COMM_PCIE_IPC_ALL_GATHER_CUH_

// Out-of-place AllGather for single-node PCIe systems without NVLink.
//
// Each rank contributes one contiguous shard. The output is rank-major and
// contains world_size consecutive shards. All communication storage lives in
// one caller-owned CUDA IPC slab per rank. The caller must collectively
// exchange the slab pointers and keep the slabs alive until every user (and
// every captured CUDA graph) has finished.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "flashinfer/comm/pcie_ipc_common.cuh"

namespace flashinfer {
namespace comm {
namespace pcie_ipc {
namespace all_gather {

namespace ipc = common;

constexpr int kCopyEngineWorldSize = 8;
constexpr int kCopyEngineTransportStages = 3;
constexpr int kCopyEngineRetireStage = 3;
constexpr int kCopyEngineSignalStages = 4;

// Values are part of the FFI contract and must remain stable.
enum class Variant : int {
  kFlatPush = 0,
  kRecursiveDoubling = 1,
  kCopyEngine = 2,
};

constexpr int kVariantCount = 3;

// Byte and signal offsets within one rank's CUDA IPC slab. The two SM staging
// regions are call-level double buffers. A topology-verified TP8 workspace
// additionally owns a dedicated CE region because a CE materialization can
// outlive the SM staging protocol.
struct WorkspaceLayout {
  int world_size;
  int element_size;
  int max_blocks;
  int64_t max_numel;
  int64_t pack_stride;

  size_t counter_words;
  size_t flag_base_word;
  size_t call_epoch_word;
  size_t call_blocks_arrived_word;
  size_t copy_engine_counter_base_word;
  size_t copy_engine_flag_base_word;

  size_t signal_bytes;
  size_t max_payload_bytes;
  size_t staging_slot_bytes;
  size_t staging0_offset;
  size_t staging1_offset;
  size_t copy_engine_offset;
  size_t copy_engine_bytes;
  size_t total_bytes;
};

inline bool try_compute_workspace_layout(int world_size, int64_t max_numel, int element_size,
                                         int max_blocks, bool enable_copy_engine,
                                         WorkspaceLayout* result) {
  if (result == nullptr || (world_size != 2 && world_size != 4 && world_size != 8) ||
      max_numel <= 0 || !ipc::valid_element_size(element_size) || max_blocks <= 0 ||
      max_blocks > ipc::kMaxBlocks || (enable_copy_engine && world_size != kCopyEngineWorldSize) ||
      static_cast<uint64_t>(max_numel) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return false;
  }

  WorkspaceLayout layout{};
  layout.world_size = world_size;
  layout.element_size = element_size;
  layout.max_blocks = max_blocks;
  layout.max_numel = max_numel;

  size_t payload_bytes = 0;
  if (!ipc::checked_mul_size(static_cast<size_t>(max_numel), static_cast<size_t>(element_size),
                             &payload_bytes) ||
      !ipc::checked_align_up_size(payload_bytes, ipc::kSignalAlignment,
                                  &layout.max_payload_bytes)) {
    return false;
  }

  const size_t world_size_value = static_cast<size_t>(world_size);
  const size_t max_blocks_value = static_cast<size_t>(max_blocks);
  if (!ipc::checked_mul_size(max_blocks_value, world_size_value, &layout.counter_words)) {
    return false;
  }
  layout.flag_base_word = layout.counter_words;
  size_t flag_words = 0;
  if (!ipc::checked_mul_size(2, max_blocks_value, &flag_words) ||
      !ipc::checked_mul_size(flag_words, world_size_value, &flag_words) ||
      !ipc::checked_mul_size(flag_words, ipc::kFlagStrideWords, &flag_words) ||
      !ipc::checked_add_size(layout.flag_base_word, flag_words, &layout.call_epoch_word) ||
      !ipc::checked_add_size(layout.call_epoch_word, 1, &layout.call_blocks_arrived_word) ||
      !ipc::checked_add_size(layout.call_blocks_arrived_word, 1,
                             &layout.copy_engine_counter_base_word)) {
    return false;
  }

  const size_t copy_engine_counter_words =
      enable_copy_engine ? static_cast<size_t>(kCopyEngineSignalStages) : 0;
  if (!ipc::checked_add_size(layout.copy_engine_counter_base_word, copy_engine_counter_words,
                             &layout.copy_engine_flag_base_word)) {
    return false;
  }

  size_t copy_engine_flag_words = 0;
  if (enable_copy_engine && !ipc::checked_mul_size(kCopyEngineWorldSize, kCopyEngineSignalStages,
                                                   &copy_engine_flag_words)) {
    return false;
  }
  size_t signal_words = 0;
  size_t unaligned_signal_bytes = 0;
  if (!ipc::checked_add_size(layout.copy_engine_flag_base_word, copy_engine_flag_words,
                             &signal_words) ||
      !ipc::checked_mul_size(signal_words, sizeof(uint32_t), &unaligned_signal_bytes) ||
      !ipc::checked_align_up_size(unaligned_signal_bytes, ipc::kSignalAlignment,
                                  &layout.signal_bytes)) {
    return false;
  }

  if (!ipc::checked_mul_size(world_size_value, layout.max_payload_bytes,
                             &layout.staging_slot_bytes)) {
    return false;
  }
  layout.staging0_offset = layout.signal_bytes;
  if (!ipc::checked_add_size(layout.staging0_offset, layout.staging_slot_bytes,
                             &layout.staging1_offset) ||
      !ipc::checked_add_size(layout.staging1_offset, layout.staging_slot_bytes,
                             &layout.copy_engine_offset)) {
    return false;
  }
  layout.copy_engine_bytes = enable_copy_engine ? layout.staging_slot_bytes : 0;
  if (!ipc::checked_add_size(layout.copy_engine_offset, layout.copy_engine_bytes,
                             &layout.total_bytes) ||
      layout.total_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      layout.max_payload_bytes / sizeof(uint4) >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }

  layout.pack_stride = static_cast<int64_t>(layout.max_payload_bytes / sizeof(uint4));
  *result = layout;
  return true;
}

inline bool try_workspace_size(int world_size, int64_t max_numel, int element_size, int max_blocks,
                               bool enable_copy_engine, int64_t* result) {
  WorkspaceLayout layout{};
  if (result == nullptr || !try_compute_workspace_layout(world_size, max_numel, element_size,
                                                         max_blocks, enable_copy_engine, &layout)) {
    return false;
  }
  *result = static_cast<int64_t>(layout.total_bytes);
  return true;
}

struct PeerViews {
  uint64_t signal_ptrs[ipc::kMaxWorldSize]{};
  uint64_t staging0_ptrs[ipc::kMaxWorldSize]{};
  uint64_t staging1_ptrs[ipc::kMaxWorldSize]{};
  uint64_t copy_engine_ptrs[ipc::kMaxWorldSize]{};
};

inline PeerViews make_peer_views(const int64_t* ipc_ptrs, int world_size,
                                 const WorkspaceLayout& layout) {
  PeerViews views{};
  for (int peer = 0; peer < world_size; ++peer) {
    const uint64_t base = static_cast<uint64_t>(ipc_ptrs[peer]);
    views.signal_ptrs[peer] = base;
    views.staging0_ptrs[peer] = base + layout.staging0_offset;
    views.staging1_ptrs[peer] = base + layout.staging1_offset;
    views.copy_engine_ptrs[peer] =
        layout.copy_engine_bytes == 0 ? 0 : base + layout.copy_engine_offset;
  }
  return views;
}

namespace detail {

struct KernelParams {
  uint64_t signal_ptrs[ipc::kMaxWorldSize];
  uint64_t staging0_ptrs[ipc::kMaxWorldSize];
  uint64_t staging1_ptrs[ipc::kMaxWorldSize];
  const void* input;
  void* output;
  int64_t shard_packs;
  int64_t pack_stride;
  int max_blocks;
  int world_size;
  size_t flag_base_word;
  size_t call_epoch_word;
  size_t call_blocks_arrived_word;
};

struct CopyEngineBarrierParams {
  uint64_t signal_ptrs[kCopyEngineWorldSize];
  int rank;
  size_t counter_base_word;
  size_t flag_base_word;
};

__device__ __forceinline__ uint32_t* signal_base(const KernelParams& params, int rank) {
  return reinterpret_cast<uint32_t*>(params.signal_ptrs[rank]);
}

__device__ __forceinline__ uint4* staging_base(const KernelParams& params, uint32_t slot,
                                               int rank) {
  const uint64_t address = slot == 0 ? params.staging0_ptrs[rank] : params.staging1_ptrs[rank];
  return reinterpret_cast<uint4*>(address);
}

__device__ __forceinline__ size_t barrier_record(const KernelParams& params, uint32_t flag_slot,
                                                 int block, int sender) {
  return ((static_cast<size_t>(flag_slot) * params.max_blocks + block) * params.world_size +
          sender) *
         ipc::kFlagStrideWords;
}

template <int WorldSize, int StaticRank>
__device__ __forceinline__ void block_all_rank_barrier(const KernelParams& params) {
  static_assert(WorldSize == 2 || WorldSize == 4 || WorldSize == 8);
  static_assert(StaticRank >= 0 && StaticRank < WorldSize);
  __syncthreads();

  if (threadIdx.x < WorldSize) {
    const int peer = static_cast<int>(threadIdx.x);
    if (peer != StaticRank) {
      // Publish all CTA payload stores before notifying the matching peer.
      ipc::fence_system();
      uint32_t* self = signal_base(params, StaticRank);
      uint32_t* self_counter = self + blockIdx.x * params.world_size + peer;
      const uint32_t generation = ipc::load_global_u32(self_counter) + 1U;
      ipc::store_global_u32(self_counter, generation);
      const uint32_t flag_slot = generation & 1U;

      uint32_t* remote = signal_base(params, peer);
      ipc::store_release_system_u32(remote + params.flag_base_word +
                                        barrier_record(params, flag_slot, blockIdx.x, StaticRank),
                                    generation);
      const uint32_t* local_flag =
          self + params.flag_base_word + barrier_record(params, flag_slot, blockIdx.x, peer);
      while (ipc::generation_pending(ipc::load_acquire_system_u32(local_flag), generation)) {
      }
    }
  }
  __syncthreads();
}

template <int WorldSize, int StaticRank>
__device__ __forceinline__ void block_peer_barrier(const KernelParams& params, int peer) {
  static_assert(WorldSize == 2 || WorldSize == 4 || WorldSize == 8);
  static_assert(StaticRank >= 0 && StaticRank < WorldSize);
  __syncthreads();

  if (threadIdx.x == 0) {
    ipc::fence_system();
    uint32_t* self = signal_base(params, StaticRank);
    uint32_t* self_counter = self + blockIdx.x * params.world_size + peer;
    const uint32_t generation = ipc::load_global_u32(self_counter) + 1U;
    ipc::store_global_u32(self_counter, generation);
    const uint32_t flag_slot = generation & 1U;

    uint32_t* remote = signal_base(params, peer);
    ipc::store_release_system_u32(
        remote + params.flag_base_word + barrier_record(params, flag_slot, blockIdx.x, StaticRank),
        generation);
    const uint32_t* local_flag =
        self + params.flag_base_word + barrier_record(params, flag_slot, blockIdx.x, peer);
    while (ipc::generation_pending(ipc::load_acquire_system_u32(local_flag), generation)) {
    }
  }
  __syncthreads();
}

__device__ __forceinline__ uint32_t* copy_engine_signal_base(const CopyEngineBarrierParams& params,
                                                             int rank) {
  return reinterpret_cast<uint32_t*>(params.signal_ptrs[rank]);
}

__device__ __forceinline__ size_t copy_engine_flag_word(const CopyEngineBarrierParams& params,
                                                        int sender, int stage) {
  return params.flag_base_word + sender * kCopyEngineSignalStages + stage;
}

__global__ void copy_engine_pair_barrier_kernel(CopyEngineBarrierParams params, int peer,
                                                int stage) {
  if (threadIdx.x != 0) {
    return;
  }
  uint32_t* self = copy_engine_signal_base(params, params.rank);
  uint32_t* counter = self + params.counter_base_word + stage;
  const uint32_t generation = ipc::load_global_u32(counter) + 1U;
  ipc::store_global_u32(counter, generation);

  // This kernel is stream-ordered after the preceding CE copy.
  __threadfence_system();
  uint32_t* remote = copy_engine_signal_base(params, peer);
  ipc::store_release_system_u32(remote + copy_engine_flag_word(params, params.rank, stage),
                                generation);
  while (ipc::generation_pending(
      ipc::load_acquire_system_u32(self + copy_engine_flag_word(params, peer, stage)),
      generation)) {
  }
}

__global__ void copy_engine_retire_kernel(CopyEngineBarrierParams params) {
  __shared__ uint32_t generation;
  uint32_t* self = copy_engine_signal_base(params, params.rank);
  if (threadIdx.x == 0) {
    uint32_t* counter = self + params.counter_base_word + kCopyEngineRetireStage;
    generation = ipc::load_global_u32(counter) + 1U;
    ipc::store_global_u32(counter, generation);
  }
  __syncthreads();

  const int peer = static_cast<int>(threadIdx.x);
  if (peer < kCopyEngineWorldSize && peer != params.rank) {
    // The kernel starts after every local staging-to-output copy, so its
    // release retires every read of the shared CE slot.
    __threadfence_system();
    uint32_t* remote = copy_engine_signal_base(params, peer);
    ipc::store_release_system_u32(
        remote + copy_engine_flag_word(params, params.rank, kCopyEngineRetireStage), generation);
  }
  __syncthreads();

  if (peer < kCopyEngineWorldSize && peer != params.rank) {
    while (ipc::generation_pending(
        ipc::load_acquire_system_u32(self +
                                     copy_engine_flag_word(params, peer, kCopyEngineRetireStage)),
        generation)) {
    }
  }
}

template <int WorldSize, int StaticRank>
__global__ __launch_bounds__(ipc::kMaxThreads, 1) void flat_push_kernel(KernelParams params) {
  uint32_t* self_signal = signal_base(params, StaticRank);
  const uint32_t slot = ipc::select_staging_slot(self_signal, params.call_epoch_word,
                                                 params.call_blocks_arrived_word);
  const uint4* input = reinterpret_cast<const uint4*>(params.input);
  uint4* output = reinterpret_cast<uint4*>(params.output);
  const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t pack = start; pack < params.shard_packs; pack += stride) {
    const uint4 value = ipc::load_global_nc_u4(input + pack);
    ipc::store_global_u4(output + static_cast<int64_t>(StaticRank) * params.shard_packs + pack,
                         value);
#pragma unroll
    for (int offset = 1; offset < WorldSize; ++offset) {
      const int destination = (StaticRank + offset) % WorldSize;
      uint4* destination_region = staging_base(params, slot, destination) +
                                  static_cast<int64_t>(StaticRank) * params.pack_stride;
      ipc::store_global_volatile_u4(destination_region + pack, value);
    }
  }

  block_all_rank_barrier<WorldSize, StaticRank>(params);

  const uint4* local_staging = staging_base(params, slot, StaticRank);
  for (int64_t pack = start; pack < params.shard_packs; pack += stride) {
#pragma unroll
    for (int offset = 1; offset < WorldSize; ++offset) {
      const int source_rank = (StaticRank + offset) % WorldSize;
      const uint4 value = ipc::load_global_volatile_u4(
          local_staging + static_cast<int64_t>(source_rank) * params.pack_stride + pack);
      ipc::store_global_u4(output + static_cast<int64_t>(source_rank) * params.shard_packs + pack,
                           value);
    }
  }
}

template <int WorldSize, int StaticRank>
__global__ __launch_bounds__(ipc::kMaxThreads,
                             1) void recursive_doubling_kernel(KernelParams params) {
  static_assert((WorldSize & (WorldSize - 1)) == 0,
                "recursive doubling requires a power-of-two world size");
  uint32_t* self_signal = signal_base(params, StaticRank);
  const uint32_t slot = ipc::select_staging_slot(self_signal, params.call_epoch_word,
                                                 params.call_blocks_arrived_word);
  const uint4* input = reinterpret_cast<const uint4*>(params.input);
  uint4* output = reinterpret_cast<uint4*>(params.output);
  uint4* local_staging = staging_base(params, slot, StaticRank);
  const int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  // Highest xor bit first. For an ordered 4+4 TP8 placement this is the only
  // cross-island exchange; xor 2 and xor 1 then relay within each island.
  constexpr int first_mask = WorldSize / 2;
  constexpr int first_peer = StaticRank ^ first_mask;
  uint4* local_source = local_staging + static_cast<int64_t>(StaticRank) * params.pack_stride;
  uint4* first_remote = staging_base(params, slot, first_peer) +
                        static_cast<int64_t>(StaticRank) * params.pack_stride;
  for (int64_t pack = start; pack < params.shard_packs; pack += stride) {
    const uint4 value = ipc::load_global_nc_u4(input + pack);
    ipc::store_global_volatile_u4(local_source + pack, value);
    ipc::store_global_volatile_u4(first_remote + pack, value);
    ipc::store_global_u4(output + static_cast<int64_t>(StaticRank) * params.shard_packs + pack,
                         value);
  }
  block_peer_barrier<WorldSize, StaticRank>(params, first_peer);

  int known_mask = first_mask;
  int newly_received_mask = first_mask;
#pragma unroll
  for (int mask = first_mask >> 1; mask >= 1; mask >>= 1) {
    const int peer = StaticRank ^ mask;
    uint4* remote_staging = staging_base(params, slot, peer);
#pragma unroll
    for (int source_rank = 0; source_rank < WorldSize; ++source_rank) {
      if (((source_rank ^ StaticRank) & ~known_mask) == 0) {
        const bool copy_to_output =
            source_rank != StaticRank && ((source_rank ^ StaticRank) & newly_received_mask) != 0;
        const uint4* source_region =
            local_staging + static_cast<int64_t>(source_rank) * params.pack_stride;
        uint4* destination_region =
            remote_staging + static_cast<int64_t>(source_rank) * params.pack_stride;
        uint4* output_region = output + static_cast<int64_t>(source_rank) * params.shard_packs;
        for (int64_t pack = start; pack < params.shard_packs; pack += stride) {
          const uint4 value = ipc::load_global_volatile_u4(source_region + pack);
          ipc::store_global_volatile_u4(destination_region + pack, value);
          if (copy_to_output) {
            ipc::store_global_u4(output_region + pack, value);
          }
        }
      }
    }
    block_peer_barrier<WorldSize, StaticRank>(params, peer);
    known_mask |= mask;
    newly_received_mask = mask;
  }

  // The last acquire contributes one final half that has not yet been copied
  // to the rank-major output.
#pragma unroll
  for (int source_rank = 0; source_rank < WorldSize; ++source_rank) {
    if (source_rank != StaticRank && ((source_rank ^ StaticRank) & newly_received_mask) != 0) {
      const uint4* source_region =
          local_staging + static_cast<int64_t>(source_rank) * params.pack_stride;
      uint4* output_region = output + static_cast<int64_t>(source_rank) * params.shard_packs;
      for (int64_t pack = start; pack < params.shard_packs; pack += stride) {
        ipc::store_global_u4(output_region + pack,
                             ipc::load_global_volatile_u4(source_region + pack));
      }
    }
  }
}

inline KernelParams make_kernel_params(const void* input, void* output, int64_t numel,
                                       const PeerViews& views, const WorkspaceLayout& layout) {
  KernelParams params{};
  for (int peer = 0; peer < layout.world_size; ++peer) {
    params.signal_ptrs[peer] = views.signal_ptrs[peer];
    params.staging0_ptrs[peer] = views.staging0_ptrs[peer];
    params.staging1_ptrs[peer] = views.staging1_ptrs[peer];
  }
  params.input = input;
  params.output = output;
  params.shard_packs = numel * layout.element_size / static_cast<int64_t>(ipc::kPackBytes);
  params.pack_stride = layout.pack_stride;
  params.max_blocks = layout.max_blocks;
  params.world_size = layout.world_size;
  params.flag_base_word = layout.flag_base_word;
  params.call_epoch_word = layout.call_epoch_word;
  params.call_blocks_arrived_word = layout.call_blocks_arrived_word;
  return params;
}

inline CopyEngineBarrierParams make_copy_engine_barrier_params(const PeerViews& views, int rank,
                                                               const WorkspaceLayout& layout) {
  CopyEngineBarrierParams params{};
  for (int peer = 0; peer < kCopyEngineWorldSize; ++peer) {
    params.signal_ptrs[peer] = views.signal_ptrs[peer];
  }
  params.rank = rank;
  params.counter_base_word = layout.copy_engine_counter_base_word;
  params.flag_base_word = layout.copy_engine_flag_base_word;
  return params;
}

inline int bit_reverse_3(int rank) { return ((rank & 1) << 2) | (rank & 2) | ((rank & 4) >> 2); }

inline cudaError_t launch_copy_engine(const void* input, void* output, int64_t numel,
                                      const PeerViews& views, int rank,
                                      const WorkspaceLayout& layout, cudaStream_t stream) {
  const size_t shard_bytes = static_cast<size_t>(numel) * static_cast<size_t>(layout.element_size);
  const int slot = bit_reverse_3(rank);
  const uint64_t local_staging = views.copy_engine_ptrs[rank];
  const CopyEngineBarrierParams barrier_params =
      make_copy_engine_barrier_params(views, rank, layout);

  cudaError_t error = cudaMemcpyAsync(
      reinterpret_cast<void*>(local_staging + static_cast<uint64_t>(slot) * shard_bytes), input,
      shard_bytes, cudaMemcpyDefault, stream);
  if (error != cudaSuccess) {
    return error;
  }

  constexpr int masks[kCopyEngineTransportStages] = {4, 2, 1};
  constexpr int shards[kCopyEngineTransportStages] = {1, 2, 4};
  for (int stage = 0; stage < kCopyEngineTransportStages; ++stage) {
    const int peer = rank ^ masks[stage];
    const int first_slot = (slot / shards[stage]) * shards[stage];
    const size_t byte_offset = static_cast<size_t>(first_slot) * shard_bytes;
    error = cudaMemcpyAsync(reinterpret_cast<void*>(views.copy_engine_ptrs[peer] + byte_offset),
                            reinterpret_cast<const void*>(local_staging + byte_offset),
                            static_cast<size_t>(shards[stage]) * shard_bytes, cudaMemcpyDefault,
                            stream);
    if (error != cudaSuccess) {
      return error;
    }
    copy_engine_pair_barrier_kernel<<<1, 32, 0, stream>>>(barrier_params, peer, stage);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
      return error;
    }
  }

  for (int source_rank = 0; source_rank < kCopyEngineWorldSize; ++source_rank) {
    const int source_slot = bit_reverse_3(source_rank);
    error =
        cudaMemcpyAsync(static_cast<char*>(output) + static_cast<size_t>(source_rank) * shard_bytes,
                        reinterpret_cast<const void*>(
                            local_staging + static_cast<uint64_t>(source_slot) * shard_bytes),
                        shard_bytes, cudaMemcpyDefault, stream);
    if (error != cudaSuccess) {
      return error;
    }
  }
  copy_engine_retire_kernel<<<1, 32, 0, stream>>>(barrier_params);
  return cudaGetLastError();
}

template <int WorldSize, Variant Algorithm, int CandidateRank = 0>
inline cudaError_t launch_static_rank(const KernelParams& params, int rank, int blocks, int threads,
                                      cudaStream_t stream) {
  static_assert(CandidateRank >= 0 && CandidateRank < WorldSize);
  if (rank == CandidateRank) {
    if constexpr (Algorithm == Variant::kFlatPush) {
      if constexpr (WorldSize == 4) {
        flat_push_kernel<WorldSize, CandidateRank><<<blocks, threads, 0, stream>>>(params);
      } else {
        return cudaErrorInvalidValue;
      }
    } else {
      recursive_doubling_kernel<WorldSize, CandidateRank><<<blocks, threads, 0, stream>>>(params);
    }
    return cudaGetLastError();
  }
  if constexpr (CandidateRank + 1 < WorldSize) {
    return launch_static_rank<WorldSize, Algorithm, CandidateRank + 1>(params, rank, blocks,
                                                                       threads, stream);
  }
  return cudaErrorInvalidValue;
}

template <int WorldSize>
inline cudaError_t launch_sm(const KernelParams& params, int rank, int blocks, int threads,
                             Variant variant, cudaStream_t stream) {
  if (variant == Variant::kFlatPush) {
    return launch_static_rank<WorldSize, Variant::kFlatPush>(params, rank, blocks, threads, stream);
  }
  if (variant == Variant::kRecursiveDoubling) {
    return launch_static_rank<WorldSize, Variant::kRecursiveDoubling>(params, rank, blocks, threads,
                                                                      stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace detail

inline cudaError_t all_gather(const void* input, void* output, int64_t numel,
                              const PeerViews& views, int rank, const WorkspaceLayout& layout,
                              int blocks, int threads, Variant variant, cudaStream_t stream) {
  if (variant == Variant::kCopyEngine) {
    return detail::launch_copy_engine(input, output, numel, views, rank, layout, stream);
  }

  const detail::KernelParams params =
      detail::make_kernel_params(input, output, numel, views, layout);
  switch (layout.world_size) {
    case 2:
      return detail::launch_sm<2>(params, rank, blocks, threads, variant, stream);
    case 4:
      return detail::launch_sm<4>(params, rank, blocks, threads, variant, stream);
    case 8:
      return detail::launch_sm<8>(params, rank, blocks, threads, variant, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace all_gather
}  // namespace pcie_ipc
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_PCIE_IPC_ALL_GATHER_CUH_
