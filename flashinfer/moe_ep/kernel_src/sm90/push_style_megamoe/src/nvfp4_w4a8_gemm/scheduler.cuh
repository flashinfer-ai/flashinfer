// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#ifndef SM90_PUSH_W4A8_PRODUCER_REGS
// Keep production at 40 so N64 variants remain spill-free while preserving
// the consumer register budget.
#define SM90_PUSH_W4A8_PRODUCER_REGS 40
#endif

#if SM90_PUSH_W4A8_PRODUCER_REGS != 40 && SM90_PUSH_W4A8_PRODUCER_REGS != 64 && \
    SM90_PUSH_W4A8_PRODUCER_REGS != 80
#error "SM90_PUSH_W4A8_PRODUCER_REGS must be 40, 64, or 80"
#endif

namespace flashinfer {
namespace sm90_w4a8 {

constexpr int kBlockK = 128;
constexpr int kProducerThreads = 128;

template <int BlockM>
inline constexpr int kConsumerThreadsFor = (BlockM / 64) * 128;

template <int BlockM>
inline constexpr int kThreadsFor = kConsumerThreadsFor<BlockM> + kProducerThreads;

template <int BlockM, int BlockN, int PipelineStages = 0>
struct W4A8LaunchTraits {
  static_assert(BlockM == 64 || BlockM == 128);
  static_assert(BlockN == 64 || BlockN == 128);
  static_assert((BlockM == 64 && (PipelineStages == 2 || PipelineStages == 3)) ||
                (BlockM == 128 && (PipelineStages == 3 || PipelineStages == 4)));
  static constexpr int kThreads = kThreadsFor<BlockM>;
  static constexpr int kPipelineStages = PipelineStages;
  static constexpr int kMinBlocksPerSm = BlockM == 64 && BlockN == 64 ? 2 : 1;
  static constexpr int kDebugMinBlocksPerSm = 1;
  static constexpr int kProducerRegisters = SM90_PUSH_W4A8_PRODUCER_REGS;
  static constexpr int kConsumerThreads = kConsumerThreadsFor<BlockM>;
  static constexpr int kConsumerRegisterCap =
      ((65536 - kProducerThreads * kProducerRegisters) / kConsumerThreads / 8) * 8;
  static_assert(kConsumerRegisterCap >= 152,
                "producer register budget leaves too few consumer registers");
  static constexpr int kUncappedConsumerRegisters = BlockM == 64 && BlockN == 64 ? 128 : 232;
  static constexpr int kConsumerRegisters = kUncappedConsumerRegisters < kConsumerRegisterCap
                                                ? kUncappedConsumerRegisters
                                                : kConsumerRegisterCap;
  static constexpr int kRegisterFootprintTarget = kConsumerRegisters;
};

template <int BlockM, int BlockN>
struct W4A8LaunchTraits<BlockM, BlockN, 0>
    : W4A8LaunchTraits<BlockM, BlockN, BlockM == 64 ? 3 : 4> {
  static constexpr int kPipelineStages = BlockM == 64 ? 3 : 4;
};

enum class MTileFamily : int32_t {
  kM64 = 0,
  kM128 = 1,
};

constexpr int kNumMTileFamilies = 2;

enum class NTileFamily : int32_t {
  kN64 = 0,
  kN128 = 1,
};

constexpr int kNumNTileFamilies = 2;
constexpr int kNumTaskCountersPerBank = kNumMTileFamilies * kNumNTileFamilies;
constexpr int kNumCounterBanks = 2;
constexpr int kNumTaskCounters = kNumCounterBanks * kNumTaskCountersPerBank;
constexpr size_t kScheduleWorkspaceBytesPerExpert = kNumMTileFamilies * sizeof(int64_t);
constexpr size_t kScheduleWorkspaceFixedBytes =
    kNumTaskCounters * sizeof(unsigned long long) + kNumMTileFamilies * sizeof(int64_t);
static_assert(kScheduleWorkspaceBytesPerExpert == 16);
static_assert(kScheduleWorkspaceFixedBytes == 80);

struct GroupedTask {
  int32_t bucket_expert;
  int32_t source_expert;
  int32_t n_begin;
  int64_t m_begin;
  int64_t m_end;
  int64_t padded_m_begin;
  bool valid;
};

__host__ __device__ constexpr int64_t ceil_div_nonnegative(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

// Dispatch activation scales pad every source-expert segment to 32 rows.  The
// source expert, rather than its position in a promoted-weight bucket, selects
// that segment.
__host__ __device__ constexpr int64_t padded_offset(int64_t offset, int32_t source_expert) {
  return (offset + static_cast<int64_t>(source_expert) * 31) / 32 * 32;
}

__host__ __device__ constexpr int m_tile_family_index(MTileFamily family) {
  return static_cast<int>(family);
}

__host__ __device__ constexpr int n_tile_family_index(NTileFamily family) {
  return static_cast<int>(family);
}

template <int BlockM>
__host__ __device__ constexpr MTileFamily m_tile_family() {
  static_assert(BlockM == 64 || BlockM == 128);
  if constexpr (BlockM == 64) {
    return MTileFamily::kM64;
  } else {
    return MTileFamily::kM128;
  }
}

template <int BlockN>
__host__ __device__ constexpr NTileFamily n_tile_family() {
  static_assert(BlockN == 64 || BlockN == 128);
  if constexpr (BlockN == 64) {
    return NTileFamily::kN64;
  } else {
    return NTileFamily::kN128;
  }
}

__host__ __device__ constexpr int64_t m64_tile_count(int64_t rows) {
  const int64_t remainder = rows % 128;
  return remainder > 0 && remainder <= 64 ? 1 : 0;
}

__host__ __device__ constexpr int64_t m128_tile_count(int64_t rows) {
  const int64_t remainder = rows % 128;
  return rows / 128 + (remainder > 64 ? 1 : 0);
}

__host__ __device__ constexpr size_t schedule_workspace_size(int32_t bucket_experts) {
  return kScheduleWorkspaceFixedBytes +
         static_cast<size_t>(bucket_experts) * kScheduleWorkspaceBytesPerExpert;
}

__host__ __device__ inline unsigned long long* schedule_task_counters(void* workspace) {
  return static_cast<unsigned long long*>(workspace);
}

__host__ __device__ inline unsigned long long* schedule_task_counter_bank(void* workspace,
                                                                          int counter_bank) {
  return schedule_task_counters(workspace) + counter_bank * kNumTaskCountersPerBank;
}

__host__ __device__ inline unsigned long long* schedule_task_counter(void* workspace,
                                                                     int counter_bank,
                                                                     MTileFamily m_family,
                                                                     NTileFamily n_family) {
  const int index =
      m_tile_family_index(m_family) * kNumNTileFamilies + n_tile_family_index(n_family);
  return schedule_task_counter_bank(workspace, counter_bank) + index;
}

__host__ __device__ inline int64_t* schedule_tile_prefixes(void* workspace) {
  return reinterpret_cast<int64_t*>(static_cast<uint8_t*>(workspace) +
                                    kNumTaskCounters * sizeof(unsigned long long));
}

__host__ __device__ inline int64_t* schedule_tile_prefix(void* workspace, MTileFamily family,
                                                         int32_t bucket_experts) {
  return schedule_tile_prefixes(workspace) + static_cast<size_t>(m_tile_family_index(family)) *
                                                 (static_cast<size_t>(bucket_experts) + 1);
}

__device__ __forceinline__ void trap_invalid_schedule(int code) {
  printf("sm90_w4a8_gemm: invalid offsets or expert mapping, code=%d\n", code);
  asm volatile("trap;");
}

__global__ void prepare_grouped_schedule_kernel(
    const int64_t* source_offsets, const int32_t* expert_mapping, int32_t bucket_experts,
    int32_t total_experts, int64_t row_capacity, unsigned long long* task_counters,
    int64_t* tile_prefix_m64, int64_t* tile_prefix_m128, bool trusted_offsets);

template <int BlockM, int BlockN>
__device__ __forceinline__ GroupedTask
map_grouped_task(uint64_t task_index, const int64_t* source_offsets, const int32_t* expert_mapping,
                 const int64_t* tile_prefix, int32_t bucket_experts, int32_t n_tiles,
                 int32_t n_tile_begin, int32_t total_experts, int64_t row_capacity) {
  static_assert(BlockM == 64 || BlockM == 128);
  static_assert(BlockN == 64 || BlockN == 128);
  GroupedTask result{-1, -1, 0, 0, 0, 0, false};
  if (n_tiles <= 0 || bucket_experts <= 0 || tile_prefix[0] < 0) {
    return result;
  }

  const uint64_t row_task = task_index / static_cast<uint64_t>(n_tiles);
  if (row_task >= static_cast<uint64_t>(tile_prefix[bucket_experts])) {
    return result;
  }

  // upper_bound(prefix, row_task) - 1 also handles empty experts, whose two
  // adjacent prefix entries are equal.
  int32_t low = 0;
  int32_t high = bucket_experts;
  while (low < high) {
    const int32_t middle = low + (high - low) / 2;
    if (static_cast<uint64_t>(tile_prefix[middle + 1]) <= row_task) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }

  const int32_t bucket_expert = low;
  const uint64_t local_m_tile = row_task - static_cast<uint64_t>(tile_prefix[bucket_expert]);
  const uint64_t local_n_tile = task_index % static_cast<uint64_t>(n_tiles);
  const int32_t source_expert = expert_mapping[bucket_expert];
  if (source_expert < 0 || source_expert >= total_experts) {
    return result;
  }
  const int64_t source_begin = source_offsets[source_expert];
  const int64_t source_end = source_offsets[source_expert + 1];
  if (source_begin < 0 || source_end < source_begin || source_end > row_capacity) {
    return result;
  }

  int64_t local_m_begin = 0;
  if constexpr (BlockM == 64) {
    if (local_m_tile != 0) {
      return result;
    }
    local_m_begin = (source_end - source_begin) / 128 * 128;
  } else {
    local_m_begin = static_cast<int64_t>(local_m_tile) * BlockM;
  }
  result.bucket_expert = bucket_expert;
  result.source_expert = source_expert;
  result.n_begin = (n_tile_begin + static_cast<int32_t>(local_n_tile)) * BlockN;
  result.m_begin = source_begin + local_m_begin;
  result.m_end = source_end;
  result.padded_m_begin = padded_offset(source_begin, source_expert) + local_m_begin;
  result.valid = true;
  return result;
}

}  // namespace sm90_w4a8
}  // namespace flashinfer
