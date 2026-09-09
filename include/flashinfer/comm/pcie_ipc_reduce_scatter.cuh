/*
 * Copyright (c) 2026 by FlashInfer team.
 * Copyright (c) 2026 by the pcie_reduce_scatter_flat_safe_static contributors.
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
 *
 * The staged PCIe schedules and 4+4 decomposition are derived from
 * FlashInfer's Apache-2.0 pcie_ipc_all_reduce implementation. The block-pair
 * signaling and device-selected double-buffer slot protocol are derived from b12x's
 * Apache-2.0 PCIe two-shot implementation:
 *   https://github.com/flashinfer-ai/flashinfer/pull/4393
 *   https://github.com/local-inference-lab/b12x
 */
#ifndef FLASHINFER_COMM_PCIE_IPC_REDUCE_SCATTER_CUH_
#define FLASHINFER_COMM_PCIE_IPC_REDUCE_SCATTER_CUH_

// SUM reduce-scatter for CUDA-IPC-connected PCIe GPUs.
//
// Input is laid out as world_size consecutive output shards. Each source rank
// pushes a destination's shard into that destination's IPC slab; the owner
// performs the only reduction and accumulates in FP32. The TP8 variants use a
// 4+4 island decomposition so only one four-rank partial per shard crosses the
// inter-island links.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "flashinfer/comm/pcie_ipc_common.cuh"

namespace flashinfer {
namespace comm {
namespace pcie_ipc {
namespace reduce_scatter {

namespace ipc = common;

using ipc::kFlagStrideWords;
using ipc::kMaxBlocks;
using ipc::kMaxThreads;
using ipc::kMaxWorldSize;

// Values cross the FFI boundary and are therefore explicit and append-only.
enum class Variant : int {
  kFlatCyclic = 0,
  kFlatOnePack = 1,
  kTopologyCyclic = 2,
  kTopologyOnePack = 3,
};

constexpr int kVariantCount = 4;

// One CUDA-IPC allocation per rank:
//
//   [ signal metadata | staging slot 0 | staging slot 1 ]
//
// Each call-selected staging region has world_size source slots. TP8 alone
// adds one extra slot for the cross-island partial. Offsets use
// max_output_numel and never move between calls, so a small replay cannot
// overwrite a larger call still being drained.
struct WorkspaceLayout {
  int world_size;
  int element_size;
  int max_blocks;
  int64_t max_output_numel;
  int64_t pack_stride;

  size_t counter_words;
  size_t flag_base_word;
  size_t call_epoch_word;
  size_t call_blocks_arrived_word;

  size_t signal_bytes;
  size_t max_payload_bytes;
  size_t staging_slot_bytes;
  size_t staging0_offset;
  size_t staging1_offset;
  size_t total_bytes;
};

namespace detail {

using ipc::load_global_nc_u4;
using ipc::load_global_volatile_u4;
using ipc::store_global_u4;
using ipc::store_global_volatile_u4;

template <typename T>
struct DTypeTraits;

template <typename T>
struct DTypePack {
  static_assert(ipc::kPackBytes % sizeof(T) == 0);
  static constexpr int kElements = ipc::kPackBytes / sizeof(T);
};

template <>
struct DTypeTraits<float> : DTypePack<float> {
  __device__ __forceinline__ static float to_float(float value) { return value; }
  __device__ __forceinline__ static float from_float(float value) { return value; }
};

template <>
struct DTypeTraits<half> : DTypePack<half> {
  __device__ __forceinline__ static float to_float(half value) { return __half2float(value); }
  __device__ __forceinline__ static half from_float(float value) { return __float2half_rn(value); }
};

template <>
struct DTypeTraits<nv_bfloat16> : DTypePack<nv_bfloat16> {
  __device__ __forceinline__ static float to_float(nv_bfloat16 value) {
    return __bfloat162float(value);
  }
  __device__ __forceinline__ static nv_bfloat16 from_float(float value) {
    return __float2bfloat16_rn(value);
  }
};

template <typename T>
__device__ __forceinline__ void zero_accumulator(float* accumulator) {
#pragma unroll
  for (int lane = 0; lane < DTypeTraits<T>::kElements; ++lane) {
    accumulator[lane] = 0.0F;
  }
}

template <typename T>
__device__ __forceinline__ void accumulate_pack(float* accumulator, uint4 raw) {
  const T* values = reinterpret_cast<const T*>(&raw);
#pragma unroll
  for (int lane = 0; lane < DTypeTraits<T>::kElements; ++lane) {
    accumulator[lane] += DTypeTraits<T>::to_float(values[lane]);
  }
}

template <typename T>
__device__ __forceinline__ uint4 pack(const float* accumulator) {
  uint4 raw;
  T* values = reinterpret_cast<T*>(&raw);
#pragma unroll
  for (int lane = 0; lane < DTypeTraits<T>::kElements; ++lane) {
    values[lane] = DTypeTraits<T>::from_float(accumulator[lane]);
  }
  return raw;
}

}  // namespace detail

inline bool try_compute_workspace_layout(int world_size, int64_t max_output_numel, int element_size,
                                         int max_blocks, WorkspaceLayout* result) {
  if (result == nullptr || (world_size != 2 && world_size != 4 && world_size != 8) ||
      max_output_numel <= 0 || !ipc::valid_element_size(element_size) || max_blocks <= 0 ||
      max_blocks > kMaxBlocks ||
      static_cast<uint64_t>(max_output_numel) >
          static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return false;
  }

  WorkspaceLayout layout{};
  layout.world_size = world_size;
  layout.element_size = element_size;
  layout.max_blocks = max_blocks;
  layout.max_output_numel = max_output_numel;

  size_t payload_bytes = 0;
  if (!ipc::checked_mul_size(static_cast<size_t>(max_output_numel),
                             static_cast<size_t>(element_size), &payload_bytes) ||
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
      !ipc::checked_mul_size(flag_words, kFlagStrideWords, &flag_words) ||
      !ipc::checked_add_size(layout.flag_base_word, flag_words, &layout.call_epoch_word) ||
      !ipc::checked_add_size(layout.call_epoch_word, 1, &layout.call_blocks_arrived_word)) {
    return false;
  }

  size_t signal_words = 0;
  size_t unaligned_signal_bytes = 0;
  if (!ipc::checked_add_size(layout.call_blocks_arrived_word, 1, &signal_words) ||
      !ipc::checked_mul_size(signal_words, sizeof(uint32_t), &unaligned_signal_bytes) ||
      !ipc::checked_align_up_size(unaligned_signal_bytes, ipc::kSignalAlignment,
                                  &layout.signal_bytes)) {
    return false;
  }

  size_t staging_shards = world_size_value;
  if (world_size == kMaxWorldSize && !ipc::checked_add_size(staging_shards, 1, &staging_shards)) {
    return false;
  }
  if (!ipc::checked_mul_size(staging_shards, layout.max_payload_bytes,
                             &layout.staging_slot_bytes)) {
    return false;
  }
  layout.staging0_offset = layout.signal_bytes;
  if (!ipc::checked_add_size(layout.staging0_offset, layout.staging_slot_bytes,
                             &layout.staging1_offset) ||
      !ipc::checked_add_size(layout.staging1_offset, layout.staging_slot_bytes,
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

inline bool try_workspace_size(int world_size, int64_t max_output_numel, int element_size,
                               int max_blocks, int64_t* result) {
  WorkspaceLayout layout{};
  if (result == nullptr || !try_compute_workspace_layout(world_size, max_output_numel, element_size,
                                                         max_blocks, &layout)) {
    return false;
  }
  *result = static_cast<int64_t>(layout.total_bytes);
  return true;
}

struct PeerViews {
  uint64_t signal[kMaxWorldSize];
  uint64_t staging0[kMaxWorldSize];
  uint64_t staging1[kMaxWorldSize];
};

inline PeerViews make_peer_views(const int64_t* ipc_ptrs, int world_size,
                                 const WorkspaceLayout& layout) {
  PeerViews views{};
  for (int peer = 0; peer < world_size; ++peer) {
    const uint64_t base = static_cast<uint64_t>(ipc_ptrs[peer]);
    views.signal[peer] = base;
    views.staging0[peer] = base + layout.staging0_offset;
    views.staging1[peer] = base + layout.staging1_offset;
  }
  return views;
}

struct KernelParams {
  uint64_t signal_ptrs[kMaxWorldSize];
  uint64_t staging0_ptrs[kMaxWorldSize];
  uint64_t staging1_ptrs[kMaxWorldSize];
  const void* input;
  void* output;
  int64_t output_packs;
  int64_t pack_stride;
  int max_blocks;
  int world_size;
  int rank;
  size_t flag_base_word;
  size_t call_epoch_word;
  size_t call_blocks_arrived_word;
};

namespace detail {

__device__ __forceinline__ uint32_t* signal_base(const KernelParams& params, int rank) {
  return reinterpret_cast<uint32_t*>(params.signal_ptrs[rank]);
}

__device__ __forceinline__ uint4* staging_base(const KernelParams& params, uint32_t slot,
                                               int rank) {
  const uint64_t address = slot == 0 ? params.staging0_ptrs[rank] : params.staging1_ptrs[rank];
  return reinterpret_cast<uint4*>(address);
}

// Every CTA observes the current staging slot before contributing to its
// retirement. Same-stream launch ordering prevents the next collective from
// reusing that slot until every CTA in this launch has exited.
template <int Rank>
__device__ __forceinline__ uint32_t select_staging_slot(const KernelParams& params) {
  uint32_t* self_signal = signal_base(params, Rank);
  return ipc::select_staging_slot(self_signal, params.call_epoch_word,
                                  params.call_blocks_arrived_word);
}

__device__ __forceinline__ size_t barrier_record(const KernelParams& params, uint32_t flag_slot,
                                                 int block, int sender) {
  return ((static_cast<size_t>(flag_slot) * params.max_blocks + block) * params.world_size +
          sender) *
         kFlagStrideWords;
}

// CTA-scoped rendezvous among the ranks selected by participant_mask. The
// protocol intentionally includes the self peer, matching the measured
// production variants and keeping all signalling lanes uniform.
template <int WorldSize, int Rank>
__device__ __forceinline__ void block_mask_barrier(const KernelParams& params,
                                                   uint32_t participant_mask) {
  static_assert(WorldSize == 2 || WorldSize == 4 || WorldSize == 8);
  static_assert(Rank >= 0 && Rank < WorldSize);
  __syncthreads();

  if ((participant_mask & (1U << Rank)) != 0U && threadIdx.x < WorldSize) {
    const int peer = static_cast<int>(threadIdx.x);
    if ((participant_mask & (1U << peer)) != 0U) {
      ipc::fence_system();

      uint32_t* self = signal_base(params, Rank);
      uint32_t* self_counter = self + blockIdx.x * params.world_size + peer;
      const uint32_t value = ipc::load_global_u32(self_counter) + 1U;
      ipc::store_global_u32(self_counter, value);
      const uint32_t flag_slot = value & 1U;

      uint32_t* peer_signal = signal_base(params, peer);
      ipc::store_release_system_u32(
          peer_signal + params.flag_base_word + barrier_record(params, flag_slot, blockIdx.x, Rank),
          value);
      const uint32_t* self_flag =
          self + params.flag_base_word + barrier_record(params, flag_slot, blockIdx.x, peer);
      while (ipc::generation_pending(ipc::load_acquire_system_u32(self_flag), value)) {
      }
    }
  }
  __syncthreads();
}

template <typename T, int WorldSize, int Rank>
__global__ __launch_bounds__(kMaxThreads, 1) void flat_cyclic_kernel(KernelParams params) {
  const uint32_t slot = select_staging_slot<Rank>(params);
  const int64_t first_pack = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pack_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const uint4* input = reinterpret_cast<const uint4*>(params.input);

  // Rank-staggered traversal gives each pass one outbound and one inbound
  // stream per rank. All passes overlap, followed by one publish barrier.
#pragma unroll 1
  for (int pass = 1; pass < WorldSize; ++pass) {
    const int destination = (Rank + pass) % WorldSize;
    const uint4* source = input + static_cast<int64_t>(destination) * params.output_packs;
    uint4* destination_slot =
        staging_base(params, slot, destination) + static_cast<int64_t>(Rank) * params.pack_stride;
    for (int64_t pack = first_pack; pack < params.output_packs; pack += pack_stride) {
      store_global_volatile_u4(destination_slot + pack, load_global_nc_u4(source + pack));
    }
  }

  block_mask_barrier<WorldSize, Rank>(params, (1U << WorldSize) - 1U);

  const uint4* local_staging = staging_base(params, slot, Rank);
  uint4* output = reinterpret_cast<uint4*>(params.output);
  for (int64_t pack = first_pack; pack < params.output_packs; pack += pack_stride) {
    float accumulator[DTypeTraits<T>::kElements];
    zero_accumulator<T>(accumulator);
#pragma unroll
    for (int source_rank = 0; source_rank < WorldSize; ++source_rank) {
      const uint4 value =
          source_rank == Rank
              ? load_global_nc_u4(input + static_cast<int64_t>(Rank) * params.output_packs + pack)
              : load_global_volatile_u4(
                    local_staging + static_cast<int64_t>(source_rank) * params.pack_stride + pack);
      accumulate_pack<T>(accumulator, value);
    }
    store_global_u4(output + pack, detail::pack<T>(accumulator));
  }
}

template <typename T, int WorldSize, int Rank>
__global__ __launch_bounds__(kMaxThreads, 1) void flat_one_pack_kernel(KernelParams params) {
  const uint32_t slot = select_staging_slot<Rank>(params);
  const int64_t thread_pack = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t batch_packs = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const uint4* input = reinterpret_cast<const uint4*>(params.input);
  uint4* output = reinterpret_cast<uint4*>(params.output);

  for (int64_t batch_begin = 0; batch_begin < params.output_packs; batch_begin += batch_packs) {
    const int64_t pack = batch_begin + thread_pack;
    const bool active = pack < params.output_packs;
    float accumulator[DTypeTraits<T>::kElements];
    zero_accumulator<T>(accumulator);
    if (active) {
      // Keep the local term in registers across the publish barrier.
      accumulate_pack<T>(
          accumulator,
          load_global_nc_u4(input + static_cast<int64_t>(Rank) * params.output_packs + pack));
    }

#pragma unroll 1
    for (int pass = 1; pass < WorldSize; ++pass) {
      if (active) {
        const int destination = (Rank + pass) % WorldSize;
        const uint4* source = input + static_cast<int64_t>(destination) * params.output_packs;
        uint4* destination_slot = staging_base(params, slot, destination) +
                                  static_cast<int64_t>(Rank) * params.pack_stride;
        store_global_volatile_u4(destination_slot + pack, load_global_nc_u4(source + pack));
      }
    }

    block_mask_barrier<WorldSize, Rank>(params, (1U << WorldSize) - 1U);

    if (active) {
      const uint4* local_staging = staging_base(params, slot, Rank);
#pragma unroll
      for (int source_rank = 0; source_rank < WorldSize; ++source_rank) {
        if (source_rank != Rank) {
          accumulate_pack<T>(
              accumulator,
              load_global_volatile_u4(
                  local_staging + static_cast<int64_t>(source_rank) * params.pack_stride + pack));
        }
      }
      store_global_u4(output + pack, detail::pack<T>(accumulator));
    }
  }
}

template <typename T, int Rank>
__global__ __launch_bounds__(kMaxThreads,
                             1) void topology_4plus4_cyclic_kernel(KernelParams params) {
  constexpr int WorldSize = 8;
  const uint32_t slot = select_staging_slot<Rank>(params);
  const int island_base = Rank < 4 ? 0 : 4;
  const int local_position = Rank & 3;
  const uint32_t island_mask = Rank < 4 ? 0x0FU : 0xF0U;
  const int64_t first_pack = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pack_step = static_cast<int64_t>(gridDim.x) * blockDim.x;
  const uint4* input = reinterpret_cast<const uint4*>(params.input);

  // Within an island, a destination owns global chunks l and l+4. Slots
  // 2*s and 2*s+1 hold the pair contributed by source-local position s.
#pragma unroll 1
  for (int pass = 1; pass < 4; ++pass) {
    const int destination_local = (local_position + pass) & 3;
    const int destination = island_base + destination_local;
    uint4* destination_staging = staging_base(params, slot, destination);
    uint4* destination_low =
        destination_staging + static_cast<int64_t>(2 * local_position) * params.pack_stride;
    uint4* destination_high = destination_low + params.pack_stride;
    const uint4* source_low = input + static_cast<int64_t>(destination_local) * params.output_packs;
    const uint4* source_high = source_low + 4 * params.output_packs;
    for (int64_t pack = first_pack; pack < params.output_packs; pack += pack_step) {
      store_global_volatile_u4(destination_low + pack, load_global_nc_u4(source_low + pack));
      store_global_volatile_u4(destination_high + pack, load_global_nc_u4(source_high + pack));
    }
  }

  block_mask_barrier<WorldSize, Rank>(params, island_mask);

  uint4* local_staging = staging_base(params, slot, Rank);
  uint4* cross_receive = local_staging + static_cast<int64_t>(WorldSize) * params.pack_stride;
  const int cross_owner = Rank ^ 4;
  uint4* cross_destination = staging_base(params, slot, cross_owner) +
                             static_cast<int64_t>(WorldSize) * params.pack_stride;
  const int side_to_send = Rank < 4 ? 1 : 0;
  const int chunk_to_send = local_position + 4 * side_to_send;

  // Reduce the opposite side locally and send exactly one dtype-T partial across
  // the 4+4 boundary.
  for (int64_t pack = first_pack; pack < params.output_packs; pack += pack_step) {
    float cross_accumulator[DTypeTraits<T>::kElements];
    zero_accumulator<T>(cross_accumulator);
#pragma unroll
    for (int source_local = 0; source_local < 4; ++source_local) {
      const uint4 value =
          source_local == local_position
              ? load_global_nc_u4(input +
                                  static_cast<int64_t>(chunk_to_send) * params.output_packs + pack)
              : load_global_volatile_u4(local_staging +
                                        static_cast<int64_t>(2 * source_local + side_to_send) *
                                            params.pack_stride +
                                        pack);
      accumulate_pack<T>(cross_accumulator, value);
    }
    store_global_volatile_u4(cross_destination + pack, detail::pack<T>(cross_accumulator));
  }

  block_mask_barrier<WorldSize, Rank>(params, (1U << Rank) | (1U << cross_owner));

  uint4* output = reinterpret_cast<uint4*>(params.output);
  const int owned_side = Rank < 4 ? 0 : 1;
  const int owned_chunk = local_position + 4 * owned_side;
  for (int64_t pack = first_pack; pack < params.output_packs; pack += pack_step) {
    float accumulator[DTypeTraits<T>::kElements];
    zero_accumulator<T>(accumulator);
#pragma unroll
    for (int source_local = 0; source_local < 4; ++source_local) {
      const uint4 value =
          source_local == local_position
              ? load_global_nc_u4(input + static_cast<int64_t>(owned_chunk) * params.output_packs +
                                  pack)
              : load_global_volatile_u4(local_staging +
                                        static_cast<int64_t>(2 * source_local + owned_side) *
                                            params.pack_stride +
                                        pack);
      accumulate_pack<T>(accumulator, value);
    }
    accumulate_pack<T>(accumulator, load_global_volatile_u4(cross_receive + pack));
    store_global_u4(output + pack, detail::pack<T>(accumulator));
  }
}

template <typename T, int Rank>
__global__ __launch_bounds__(kMaxThreads,
                             1) void topology_4plus4_one_pack_kernel(KernelParams params) {
  constexpr int WorldSize = 8;
  const uint32_t slot = select_staging_slot<Rank>(params);
  const int island_base = Rank < 4 ? 0 : 4;
  const int local_position = Rank & 3;
  const uint32_t island_mask = Rank < 4 ? 0x0FU : 0xF0U;
  const int owned_side = Rank < 4 ? 0 : 1;
  const int side_to_send = owned_side ^ 1;
  const int owned_chunk = local_position + 4 * owned_side;
  const int chunk_to_send = local_position + 4 * side_to_send;
  const int cross_owner = Rank ^ 4;
  const uint4* input = reinterpret_cast<const uint4*>(params.input);
  uint4* output = reinterpret_cast<uint4*>(params.output);
  uint4* local_staging = staging_base(params, slot, Rank);
  uint4* cross_receive = local_staging + static_cast<int64_t>(WorldSize) * params.pack_stride;
  uint4* cross_destination = staging_base(params, slot, cross_owner) +
                             static_cast<int64_t>(WorldSize) * params.pack_stride;
  const int64_t thread_pack = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t batch_packs = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t batch_begin = 0; batch_begin < params.output_packs; batch_begin += batch_packs) {
    const int64_t pack = batch_begin + thread_pack;
    const bool active = pack < params.output_packs;

#pragma unroll 1
    for (int pass = 1; pass < 4; ++pass) {
      if (active) {
        const int destination_local = (local_position + pass) & 3;
        const int destination = island_base + destination_local;
        uint4* destination_staging = staging_base(params, slot, destination);
        uint4* destination_low =
            destination_staging + static_cast<int64_t>(2 * local_position) * params.pack_stride;
        uint4* destination_high = destination_low + params.pack_stride;
        const uint4* source_low =
            input + static_cast<int64_t>(destination_local) * params.output_packs;
        const uint4* source_high = source_low + 4 * params.output_packs;
        store_global_volatile_u4(destination_low + pack, load_global_nc_u4(source_low + pack));
        store_global_volatile_u4(destination_high + pack, load_global_nc_u4(source_high + pack));
      }
    }

    block_mask_barrier<WorldSize, Rank>(params, island_mask);

    float owned_accumulator[DTypeTraits<T>::kElements];
    float cross_accumulator[DTypeTraits<T>::kElements];
    zero_accumulator<T>(owned_accumulator);
    zero_accumulator<T>(cross_accumulator);
    if (active) {
#pragma unroll
      for (int source_local = 0; source_local < 4; ++source_local) {
        uint4 owned_value;
        uint4 cross_value;
        if (source_local == local_position) {
          owned_value = load_global_nc_u4(
              input + static_cast<int64_t>(owned_chunk) * params.output_packs + pack);
          cross_value = load_global_nc_u4(
              input + static_cast<int64_t>(chunk_to_send) * params.output_packs + pack);
        } else {
          const uint4* source_pair =
              local_staging + static_cast<int64_t>(2 * source_local) * params.pack_stride;
          owned_value =
              load_global_volatile_u4(source_pair + owned_side * params.pack_stride + pack);
          cross_value =
              load_global_volatile_u4(source_pair + side_to_send * params.pack_stride + pack);
        }
        accumulate_pack<T>(owned_accumulator, owned_value);
        accumulate_pack<T>(cross_accumulator, cross_value);
      }
      store_global_volatile_u4(cross_destination + pack, detail::pack<T>(cross_accumulator));
    }

    block_mask_barrier<WorldSize, Rank>(params, (1U << Rank) | (1U << cross_owner));

    if (active) {
      accumulate_pack<T>(owned_accumulator, load_global_volatile_u4(cross_receive + pack));
      store_global_u4(output + pack, detail::pack<T>(owned_accumulator));
    }
  }
}

template <typename T, int WorldSize, int CandidateRank = 0>
inline cudaError_t launch_static_rank(const KernelParams& params, int blocks, int threads,
                                      Variant variant, cudaStream_t stream) {
  static_assert(WorldSize == 2 || WorldSize == 4 || WorldSize == 8);
  static_assert(CandidateRank >= 0 && CandidateRank < WorldSize);
  if (params.rank == CandidateRank) {
    switch (variant) {
      case Variant::kFlatCyclic:
        if constexpr (WorldSize == 2 || WorldSize == 4) {
          flat_cyclic_kernel<T, WorldSize, CandidateRank><<<blocks, threads, 0, stream>>>(params);
        } else {
          return cudaErrorInvalidValue;
        }
        break;
      case Variant::kFlatOnePack:
        if constexpr (WorldSize == 2 || WorldSize == 4) {
          flat_one_pack_kernel<T, WorldSize, CandidateRank><<<blocks, threads, 0, stream>>>(params);
        } else {
          return cudaErrorInvalidValue;
        }
        break;
      case Variant::kTopologyCyclic:
        if constexpr (WorldSize == 8) {
          topology_4plus4_cyclic_kernel<T, CandidateRank><<<blocks, threads, 0, stream>>>(params);
        } else {
          return cudaErrorInvalidValue;
        }
        break;
      case Variant::kTopologyOnePack:
        if constexpr (WorldSize == 8) {
          topology_4plus4_one_pack_kernel<T, CandidateRank><<<blocks, threads, 0, stream>>>(params);
        } else {
          return cudaErrorInvalidValue;
        }
        break;
      default:
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
  }
  if constexpr (CandidateRank + 1 < WorldSize) {
    return launch_static_rank<T, WorldSize, CandidateRank + 1>(params, blocks, threads, variant,
                                                               stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace detail

// Preconditions are checked by the TVM-FFI binding: contiguous tensors;
// input_numel == world_size * output_numel; output_numel and max_output_numel
// contain complete 16-byte packs; 0 < blocks <= max_blocks; warp-aligned threads in
// [32, 512]; flat variants are selected only for world_size 2 or 4; and
// topology variants are selected only for world_size 8.
template <typename T>
inline cudaError_t reduce_scatter(const T* input, T* output, int64_t output_numel,
                                  const PeerViews& views, int rank, const WorkspaceLayout& layout,
                                  int blocks, int threads, Variant variant, cudaStream_t stream) {
  KernelParams params{};
  for (int peer = 0; peer < layout.world_size; ++peer) {
    params.signal_ptrs[peer] = views.signal[peer];
    params.staging0_ptrs[peer] = views.staging0[peer];
    params.staging1_ptrs[peer] = views.staging1[peer];
  }
  params.input = input;
  params.output = output;
  params.output_packs = output_numel / detail::DTypeTraits<T>::kElements;
  params.pack_stride = layout.pack_stride;
  params.max_blocks = layout.max_blocks;
  params.world_size = layout.world_size;
  params.rank = rank;
  params.flag_base_word = layout.flag_base_word;
  params.call_epoch_word = layout.call_epoch_word;
  params.call_blocks_arrived_word = layout.call_blocks_arrived_word;

  switch (layout.world_size) {
    case 2:
      return detail::launch_static_rank<T, 2>(params, blocks, threads, variant, stream);
    case 4:
      return detail::launch_static_rank<T, 4>(params, blocks, threads, variant, stream);
    case 8:
      return detail::launch_static_rank<T, 8>(params, blocks, threads, variant, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace reduce_scatter
}  // namespace pcie_ipc
}  // namespace comm
}  // namespace flashinfer

#endif  // FLASHINFER_COMM_PCIE_IPC_REDUCE_SCATTER_CUH_
