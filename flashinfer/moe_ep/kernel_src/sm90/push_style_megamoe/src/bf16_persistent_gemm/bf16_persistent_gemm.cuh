// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0
//
// Archived experimental engine, disabled by default because the phase-locked
// pipeline and single-thread K16 TMA issue path limit throughput.

#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <flashinfer/attention/hopper.cuh>
#include <limits>
#include <type_traits>

#include "nv_internal/tensorrt_llm/deep_gemm/mma_utils.cuh"
#include "nv_internal/tensorrt_llm/deep_gemm/tma_utils.cuh"

#ifndef SM90_PUSH_BF16_SHAPE_N
#define SM90_PUSH_BF16_SHAPE_N 7168
#endif

#ifndef SM90_PUSH_BF16_SHAPE_K
#define SM90_PUSH_BF16_SHAPE_K 2048
#endif

#ifndef SM90_PUSH_BF16_FAMILY_MASK
#define SM90_PUSH_BF16_FAMILY_MASK 3
#endif

#ifndef SM90_PUSH_BF16_M64_BLOCK_N
#define SM90_PUSH_BF16_M64_BLOCK_N 128
#endif

#ifndef SM90_PUSH_BF16_M64_BLOCK_K
#define SM90_PUSH_BF16_M64_BLOCK_K 128
#endif

#ifndef SM90_PUSH_BF16_M64_STAGES
#define SM90_PUSH_BF16_M64_STAGES 3
#endif

#ifndef SM90_PUSH_BF16_M64_CLUSTER_M
#define SM90_PUSH_BF16_M64_CLUSTER_M 1
#endif

#ifndef SM90_PUSH_BF16_M64_SCHEDULE
#define SM90_PUSH_BF16_M64_SCHEDULE 0
#endif

#ifndef SM90_PUSH_BF16_M128_BLOCK_N
#define SM90_PUSH_BF16_M128_BLOCK_N 128
#endif

#ifndef SM90_PUSH_BF16_M128_BLOCK_K
#define SM90_PUSH_BF16_M128_BLOCK_K 128
#endif

#ifndef SM90_PUSH_BF16_M128_STAGES
#define SM90_PUSH_BF16_M128_STAGES 3
#endif

#ifndef SM90_PUSH_BF16_M128_CLUSTER_M
#define SM90_PUSH_BF16_M128_CLUSTER_M 2
#endif

#ifndef SM90_PUSH_BF16_M128_SCHEDULE
#define SM90_PUSH_BF16_M128_SCHEDULE 1
#endif

#ifndef SM90_PUSH_BF16_SWAP_AB
#define SM90_PUSH_BF16_SWAP_AB 0
#endif

namespace flashinfer::sm90_push_bf16_persistent {

namespace cg = cooperative_groups;

constexpr int kShapeN = SM90_PUSH_BF16_SHAPE_N;
constexpr int kShapeK = SM90_PUSH_BF16_SHAPE_K;
constexpr int kFamilyMask = SM90_PUSH_BF16_FAMILY_MASK;
constexpr bool kEnableM64 = (kFamilyMask & 1) != 0;
constexpr bool kEnableM128 = (kFamilyMask & 2) != 0;
constexpr bool kSwapAB = SM90_PUSH_BF16_SWAP_AB != 0;
constexpr int kMmaK = 16;
constexpr int kTmaThreads = 128;
constexpr int kMathThreadsPerGroup = 128;

static_assert(kShapeN > 0 && kShapeK > 0);
static_assert(kShapeN % 64 == 0 && kShapeK % kMmaK == 0);
static_assert(kFamilyMask >= 1 && kFamilyMask <= 3);
static_assert(!kSwapAB || kFamilyMask != 3);

enum class MTileFamily : int32_t {
  kM64 = 0,
  kM128 = 1,
};

template <int BlockM, int BlockN, int BlockK, int Stages, int ClusterM, int Schedule, bool SwapAB>
struct PersistentTraits {
  static_assert(BlockM == 64 || BlockM == 128);
  static_assert(BlockN == 64 || BlockN == 128);
  static_assert(BlockK == 64 || BlockK == 128);
  static_assert(BlockK % kMmaK == 0);
  static_assert(Stages >= 2 && Stages <= 4);
  static_assert(ClusterM == 1 || ClusterM == 2);
  static_assert(Schedule == 0 || Schedule == 1);
  static_assert(!SwapAB || BlockM == 64 || BlockM == 128);
  static constexpr int kBlockM = BlockM;
  static constexpr int kBlockN = BlockN;
  static constexpr int kBlockK = BlockK;
  static constexpr int kStages = Stages;
  static constexpr int kClusterM = ClusterM;
  static constexpr int kSchedule = Schedule;
  static constexpr bool kSwapAB = SwapAB;
  static constexpr int kWgmmaM = SwapAB ? BlockN : BlockM;
  static constexpr int kWgmmaN = SwapAB ? BlockM : BlockN;
  static constexpr int kMathGroups = kWgmmaM / 64;
  static constexpr int kMathThreads = kMathGroups * kMathThreadsPerGroup;
  static constexpr int kThreads = kMathThreads + kTmaThreads;
  static constexpr int kOperandARows = kWgmmaM;
  static constexpr int kOperandBRows = kWgmmaN;
  static constexpr int kOperandAElements = kOperandARows * kBlockK;
  static constexpr int kOperandBElements = kOperandBRows * kBlockK;
  static constexpr int kOutputElements = kWgmmaM * kWgmmaN;
  static_assert(kWgmmaM == 64 || kWgmmaM == 128);
  static_assert(kWgmmaN == 64 || kWgmmaN == 128);
};

using M64Traits = PersistentTraits<64, SM90_PUSH_BF16_M64_BLOCK_N, SM90_PUSH_BF16_M64_BLOCK_K,
                                   SM90_PUSH_BF16_M64_STAGES, SM90_PUSH_BF16_M64_CLUSTER_M,
                                   SM90_PUSH_BF16_M64_SCHEDULE, kSwapAB>;
using M128Traits = PersistentTraits<128, SM90_PUSH_BF16_M128_BLOCK_N, SM90_PUSH_BF16_M128_BLOCK_K,
                                    SM90_PUSH_BF16_M128_STAGES, SM90_PUSH_BF16_M128_CLUSTER_M,
                                    SM90_PUSH_BF16_M128_SCHEDULE, kSwapAB>;

static_assert(!kEnableM64 || kShapeN % M64Traits::kBlockN == 0);
static_assert(!kEnableM64 || kShapeK % M64Traits::kBlockK == 0);
static_assert(!kEnableM128 || kShapeN % M128Traits::kBlockN == 0);
static_assert(!kEnableM128 || kShapeK % M128Traits::kBlockK == 0);

struct RowFamilySelection {
  int64_t m128_rows;
  int64_t m64_begin;
  int64_t m64_rows;
};

__host__ __device__ constexpr RowFamilySelection select_row_families(int64_t begin, int64_t rows) {
  if constexpr (kEnableM64 && !kEnableM128) {
    return RowFamilySelection{0, begin, rows};
  }
  if constexpr (!kEnableM64 && kEnableM128) {
    return RowFamilySelection{rows, begin + rows, 0};
  }
  int64_t const full_m128_rows = rows / 128 * 128;
  int64_t const remainder = rows - full_m128_rows;
  return RowFamilySelection{full_m128_rows + (remainder > 64 ? remainder : 0),
                            begin + full_m128_rows,
                            remainder > 0 && remainder <= 64 ? remainder : 0};
}

__host__ __device__ constexpr uint64_t ceil_div_nonnegative(int64_t value, int64_t divisor) {
  return static_cast<uint64_t>((value + divisor - 1) / divisor);
}

struct GroupedTask {
  int32_t expert;
  int32_t n_begin;
  int64_t m_begin;
  int64_t m_end;
  bool valid;
  bool active;
};

template <typename Traits>
__device__ __forceinline__ GroupedTask map_cluster_task(uint64_t task_index, uint32_t cluster_rank,
                                                        int64_t const* offsets, int32_t num_experts,
                                                        int64_t row_capacity) {
  GroupedTask result{-1, 0, 0, 0, false, false};
  uint64_t task_base = 0;
  constexpr int block_m = Traits::kBlockM;
  constexpr int block_n = Traits::kBlockN;
  constexpr int cluster_m = Traits::kClusterM;
  constexpr int n_tiles = kShapeN / block_n;
  for (int32_t expert = 0; expert < num_experts; ++expert) {
    int64_t const group_begin = __ldg(offsets + expert);
    int64_t const group_end = __ldg(offsets + expert + 1);
    if (group_begin < 0 || group_end < group_begin || group_end > row_capacity) {
      return result;
    }
    RowFamilySelection const selection = select_row_families(group_begin, group_end - group_begin);
    int64_t const family_begin = block_m == 64 ? selection.m64_begin : group_begin;
    int64_t const family_rows = block_m == 64 ? selection.m64_rows : selection.m128_rows;
    uint64_t const m_tiles = ceil_div_nonnegative(family_rows, block_m);
    uint64_t const clustered_m_tiles = (m_tiles + cluster_m - 1) / cluster_m;
    uint64_t task_count;
    if constexpr (Traits::kSwapAB) {
      uint64_t const clustered_n_tiles = (uint64_t{n_tiles} + cluster_m - 1) / cluster_m;
      task_count = m_tiles * clustered_n_tiles;
    } else {
      task_count = clustered_m_tiles * uint64_t{n_tiles};
    }
    if (task_count > std::numeric_limits<uint64_t>::max() - task_base) {
      return result;
    }
    if (task_index < task_base + task_count) {
      uint64_t const local = task_index - task_base;
      uint64_t m_tile;
      uint64_t n_tile;
      if constexpr (Traits::kSwapAB) {
        uint64_t const clustered_n_tiles = (uint64_t{n_tiles} + cluster_m - 1) / cluster_m;
        m_tile = local / clustered_n_tiles;
        n_tile = (local % clustered_n_tiles) * cluster_m + cluster_rank;
      } else {
        m_tile = (local / uint64_t{n_tiles}) * cluster_m + cluster_rank;
        n_tile = local % uint64_t{n_tiles};
      }
      result.expert = expert;
      result.n_begin = static_cast<int32_t>(n_tile * block_n);
      result.m_begin = family_begin + static_cast<int64_t>(m_tile) * block_m;
      result.m_end = family_begin + family_rows;
      result.valid = true;
      result.active = m_tile < m_tiles && n_tile < uint64_t{n_tiles};
      if (!result.active) {
        result.n_begin = 0;
        result.m_begin = 0;
        result.m_end = 0;
      }
      return result;
    }
    task_base += task_count;
  }
  return result;
}

__global__ void offsets_preflight_kernel(int64_t const* offsets, int32_t num_experts,
                                         int64_t row_capacity) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int64_t previous = offsets[0];
  bool invalid = previous != 0;
  for (int32_t expert = 1; expert <= num_experts && !invalid; ++expert) {
    int64_t const current = offsets[expert];
    invalid = current < previous || current > row_capacity;
    previous = current;
  }
  if (invalid) {
    printf("sm90_push_bf16_persistent: offsets must be monotonic and within row capacity\n");
    asm volatile("trap;");
  }
}

template <typename Traits>
struct alignas(1024) PersistentOperandStorage {
  alignas(1024) __nv_bfloat16 operand_a[Traits::kStages][Traits::kOperandAElements];
  alignas(1024) __nv_bfloat16 operand_b[Traits::kStages][Traits::kOperandBElements];
};

template <typename Traits>
union alignas(1024) PersistentDataStorage {
  PersistentOperandStorage<Traits> operands;
  alignas(1024) __nv_bfloat16 output[Traits::kOutputElements];
};

template <typename Traits>
struct alignas(1024) PersistentSharedStorage {
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  PersistentDataStorage<Traits> data;
  alignas(8) Barrier full_barriers[Traits::kStages];
  alignas(8) uint64_t task_index;
  GroupedTask task;
};

template <typename Traits>
constexpr size_t persistent_dynamic_smem_bytes() {
  return sizeof(PersistentSharedStorage<Traits>);
}

template <typename Traits>
struct Wgmma
    : flashinfer::WGMMA_ASYNC_SS<__nv_bfloat16, float, 64, Traits::kWgmmaN, kMmaK,
                                 cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::K,
                                 cute::SM90::GMMA::ScaleIn::One, cute::SM90::GMMA::ScaleIn::One> {
  static constexpr int kNumAccum = 64 * Traits::kWgmmaN / 128;
  static_assert(kNumAccum % 8 == 0);
};

template <typename Traits>
__device__ __forceinline__ void sync_execution_scope() {
  if constexpr (Traits::kClusterM == 2) {
    cg::this_cluster().sync();
  } else {
    __syncthreads();
  }
}

template <typename Traits>
__device__ __forceinline__ uint32_t cluster_rank() {
  if constexpr (Traits::kClusterM == 2) {
    return cg::this_cluster().block_rank();
  }
  return 0;
}

template <typename Traits>
__device__ __forceinline__ void initialize_barriers(PersistentSharedStorage<Traits>& storage) {
  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < Traits::kStages; ++stage) storage.full_barriers[stage].init(1);
    cutlass::arch::fence_view_async_shared();
    if constexpr (Traits::kClusterM == 2) cutlass::arch::fence_barrier_init();
  }
  sync_execution_scope<Traits>();
}

template <typename Traits>
__device__ __forceinline__ void invalidate_barriers(PersistentSharedStorage<Traits>& storage) {
  sync_execution_scope<Traits>();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < Traits::kStages; ++stage) {
      cutlass::arch::ClusterBarrier::invalidate(
          reinterpret_cast<uint64_t const*>(&storage.full_barriers[stage]));
    }
  }
}

template <typename Traits>
__device__ __forceinline__ void acquire_task(PersistentSharedStorage<Traits>& storage,
                                             unsigned long long* task_counter,
                                             int64_t const* offsets, int32_t num_experts,
                                             int64_t row_capacity) {
  uint32_t const rank = cluster_rank<Traits>();
  if (threadIdx.x == 0 && rank == 0) storage.task_index = atomicAdd(task_counter, 1ULL);
  sync_execution_scope<Traits>();
  if constexpr (Traits::kClusterM == 2) {
    if (threadIdx.x == 0 && rank != 0) {
      storage.task_index = *cg::this_cluster().map_shared_rank(&storage.task_index, 0);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    storage.task =
        map_cluster_task<Traits>(storage.task_index, rank, offsets, num_experts, row_capacity);
  }
  __syncthreads();
}

template <typename Traits>
__device__ __forceinline__ void issue_stage(PersistentSharedStorage<Traits>& storage, int stage,
                                            int k_block, GroupedTask const& task,
                                            CUtensorMap const& activation_map,
                                            CUtensorMap const& weight_map) {
  uint32_t const rank = cluster_rank<Traits>();
  if (threadIdx.x != Traits::kMathThreads) return;
  auto& barrier = storage.full_barriers[stage];
  int32_t const k_begin = k_block * Traits::kBlockK;
  int32_t const activation_row = static_cast<int32_t>(task.m_begin);
  int32_t const weight_row = task.expert * kShapeN + task.n_begin;
  constexpr uint64_t cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
#pragma unroll
  for (int inner = 0; inner < Traits::kBlockK / kMmaK; ++inner) {
    int32_t const inner_k = k_begin + inner * kMmaK;
    if constexpr (Traits::kSwapAB) {
      deep_gemm::tma_copy(
          &weight_map, reinterpret_cast<uint64_t*>(&barrier),
          storage.data.operands.operand_a[stage] + inner * Traits::kOperandARows * kMmaK, inner_k,
          weight_row);
      if constexpr (Traits::kClusterM == 2) {
        if (rank == 0) {
          cute::SM90_TMA_LOAD_MULTICAST_2D::copy(
              &activation_map, reinterpret_cast<uint64_t*>(&barrier), uint16_t{0x3}, cache_hint,
              storage.data.operands.operand_b[stage] + inner * Traits::kOperandBRows * kMmaK,
              inner_k, activation_row);
        }
      } else {
        deep_gemm::tma_copy(
            &activation_map, reinterpret_cast<uint64_t*>(&barrier),
            storage.data.operands.operand_b[stage] + inner * Traits::kOperandBRows * kMmaK, inner_k,
            activation_row);
      }
    } else {
      deep_gemm::tma_copy(
          &activation_map, reinterpret_cast<uint64_t*>(&barrier),
          storage.data.operands.operand_a[stage] + inner * Traits::kOperandARows * kMmaK, inner_k,
          activation_row);
      if constexpr (Traits::kClusterM == 2) {
        if (rank == 0) {
          cute::SM90_TMA_LOAD_MULTICAST_2D::copy(
              &weight_map, reinterpret_cast<uint64_t*>(&barrier), uint16_t{0x3}, cache_hint,
              storage.data.operands.operand_b[stage] + inner * Traits::kOperandBRows * kMmaK,
              inner_k, weight_row);
        }
      } else {
        deep_gemm::tma_copy(
            &weight_map, reinterpret_cast<uint64_t*>(&barrier),
            storage.data.operands.operand_b[stage] + inner * Traits::kOperandBRows * kMmaK, inner_k,
            weight_row);
      }
    }
  }
  barrier.arrive_and_expect_tx(static_cast<uint32_t>(
      (Traits::kOperandAElements + Traits::kOperandBElements) * sizeof(__nv_bfloat16)));
}

template <typename Traits>
__device__ __forceinline__ int next_prefetch_block(int k_block) {
  return k_block + Traits::kStages;
}

template <typename Traits>
__device__ __forceinline__ void wait_for_stage(
    typename PersistentSharedStorage<Traits>::Barrier const& barrier, uint32_t phase) {
  if constexpr (Traits::kSchedule == 0) {
    barrier.wait(phase);
  } else {
    while (!barrier.try_wait(phase)) {
      asm volatile("nanosleep.u32 64;");
    }
  }
}

template <typename Traits>
__device__ __forceinline__ deep_gemm::GmmaDescriptor make_operand_descriptor(
    __nv_bfloat16* pointer) {
  constexpr int stride_bytes = 8 * kMmaK * sizeof(__nv_bfloat16);
  return deep_gemm::make_smem_desc(pointer, 3, 0, stride_bytes);
}

template <typename Traits>
__device__ __forceinline__ void store_accumulator(PersistentSharedStorage<Traits>& storage,
                                                  float const* accumulator, int math_group) {
  int32_t const thread = static_cast<int32_t>(threadIdx.x);
  int32_t const warp = (thread % kMathThreadsPerGroup) / 32;
  int32_t const lane = thread % 32;
#pragma unroll
  for (int index = 0; index < Wgmma<Traits>::kNumAccum / 8; ++index) {
    auto const value_0 =
        __float22bfloat162_rn({accumulator[index * 8 + 0], accumulator[index * 8 + 1]});
    auto const value_1 =
        __float22bfloat162_rn({accumulator[index * 8 + 2], accumulator[index * 8 + 3]});
    auto const value_2 =
        __float22bfloat162_rn({accumulator[index * 8 + 4], accumulator[index * 8 + 5]});
    auto const value_3 =
        __float22bfloat162_rn({accumulator[index * 8 + 6], accumulator[index * 8 + 7]});
    if constexpr (!Traits::kSwapAB) {
      deep_gemm::SM90_U32x4_STSM_N<nv_bfloat162>::copy(
          value_0, value_1, value_2, value_3,
          storage.data.output + (math_group * 64 + warp * 16 + lane % 16) * Traits::kWgmmaN +
              index * 16 + 8 * (lane / 16));
    } else {
      int32_t tid;
      if (lane < 8) {
        tid = lane * Traits::kWgmmaM;
      } else if (lane < 16) {
        tid = (lane - 8) * Traits::kWgmmaM + 8;
      } else if (lane < 24) {
        tid = (lane - 8) * Traits::kWgmmaM;
      } else {
        tid = (lane - 16) * Traits::kWgmmaM + 8;
      }
      deep_gemm::SM90_U32x4_STSM_T<nv_bfloat162>::copy(
          value_0, value_1, value_2, value_3,
          storage.data.output + math_group * 64 + warp * 16 + index * 16 * Traits::kWgmmaM + tid);
    }
  }
}

template <typename Traits>
__device__ __forceinline__ void store_output(PersistentSharedStorage<Traits>& storage,
                                             GroupedTask const& task, __nv_bfloat16* output) {
  int32_t const thread = static_cast<int32_t>(threadIdx.x);
  if constexpr (!Traits::kSwapAB) {
    constexpr int vectors_per_row = Traits::kBlockN / 8;
    constexpr int vector_count = Traits::kBlockM * vectors_per_row;
    auto const* shared_vectors = reinterpret_cast<int4 const*>(storage.data.output);
    for (int index = thread; index < vector_count; index += Traits::kMathThreads) {
      int32_t const local_m = index / vectors_per_row;
      int32_t const local_vector = index % vectors_per_row;
      int64_t const global_m = task.m_begin + local_m;
      if (task.active && global_m < task.m_end) {
        auto* global_vectors = reinterpret_cast<int4*>(
            output + global_m * static_cast<int64_t>(kShapeN) + task.n_begin);
        global_vectors[local_vector] = shared_vectors[index];
      }
    }
  } else {
    constexpr int elements = Traits::kBlockN * Traits::kBlockM;
    for (int index = thread; index < elements; index += Traits::kMathThreads) {
      int32_t const local_n = index / Traits::kBlockM;
      int32_t const local_m = index % Traits::kBlockM;
      int64_t const global_m = task.m_begin + local_m;
      if (task.active && global_m < task.m_end) {
        output[global_m * static_cast<int64_t>(kShapeN) + task.n_begin + local_n] =
            storage.data.output[local_m * Traits::kBlockN + local_n];
      }
    }
  }
}

template <typename Traits>
__global__ __launch_bounds__(Traits::kThreads, 1) void grouped_bf16_persistent_offsets_kernel(
    __nv_bfloat16* output, __nv_bfloat16 const* activation, __nv_bfloat16 const* weights,
    int64_t const* offsets, unsigned long long* task_counter, int64_t row_capacity,
    int32_t num_experts, __grid_constant__ CUtensorMap const activation_map,
    __grid_constant__ CUtensorMap const weight_map) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 900
  (void)activation;
  (void)weights;
  extern __shared__ __align__(1024) uint8_t shared_buffer[];
  auto& storage = *reinterpret_cast<PersistentSharedStorage<Traits>*>(shared_buffer);
  int32_t const thread = static_cast<int32_t>(threadIdx.x);
  if (thread == Traits::kMathThreads) {
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&activation_map));
    cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&weight_map));
  }
  constexpr int k_blocks = (kShapeK + Traits::kBlockK - 1) / Traits::kBlockK;
  while (true) {
    acquire_task(storage, task_counter, offsets, num_experts, row_capacity);
    if (!storage.task.valid) break;
    initialize_barriers(storage);

#pragma unroll
    for (int stage = 0; stage < Traits::kStages; ++stage) {
      if (stage < k_blocks) {
        issue_stage(storage, stage, stage, storage.task, activation_map, weight_map);
      }
    }

    if (thread < Traits::kMathThreads) {
      float accumulator[Wgmma<Traits>::kNumAccum] = {};
      int32_t const math_group = thread / kMathThreadsPerGroup;
      for (int k_block = 0; k_block < k_blocks; ++k_block) {
        int const stage = k_block % Traits::kStages;
        uint32_t const phase = static_cast<uint32_t>(k_block / Traits::kStages) & 1;
        wait_for_stage<Traits>(storage.full_barriers[stage], phase);
#pragma unroll
        for (int index = 0; index < Wgmma<Traits>::kNumAccum; ++index) {
          deep_gemm::warpgroup_fence_operand(accumulator[index]);
        }
        deep_gemm::warpgroup_arrive();
        int const valid_k = min(Traits::kBlockK, kShapeK - k_block * Traits::kBlockK);
        int const valid_inner = valid_k / kMmaK;
#pragma unroll
        for (int inner = 0; inner < Traits::kBlockK / kMmaK; ++inner) {
          if (inner >= valid_inner) break;
          auto const desc_a = make_operand_descriptor<Traits>(
              storage.data.operands.operand_a[stage] + inner * Traits::kOperandARows * kMmaK +
              math_group * 64 * kMmaK);
          auto const desc_b = make_operand_descriptor<Traits>(
              storage.data.operands.operand_b[stage] + inner * Traits::kOperandBRows * kMmaK);
          if (k_block == 0 && inner == 0) {
            Wgmma<Traits>::template op<true>(desc_a, desc_b, accumulator);
          } else {
            Wgmma<Traits>::template op<false>(desc_a, desc_b, accumulator);
          }
        }
        deep_gemm::warpgroup_commit_batch();
#pragma unroll
        for (int index = 0; index < Wgmma<Traits>::kNumAccum; ++index) {
          deep_gemm::warpgroup_fence_operand(accumulator[index]);
        }
        deep_gemm::warpgroup_wait<0>();

        sync_execution_scope<Traits>();
        int const next_block = next_prefetch_block<Traits>(k_block);
        if (next_block < k_blocks) {
          issue_stage(storage, stage, next_block, storage.task, activation_map, weight_map);
        }
        sync_execution_scope<Traits>();
      }
      store_accumulator(storage, accumulator, math_group);
      cutlass::arch::NamedBarrier(Traits::kMathThreads).sync();
      store_output(storage, storage.task, output);
      cutlass::arch::NamedBarrier(Traits::kMathThreads).sync();
    } else {
      for (int k_block = 0; k_block < k_blocks; ++k_block) {
        sync_execution_scope<Traits>();
        int const next_block = next_prefetch_block<Traits>(k_block);
        if (next_block < k_blocks) {
          issue_stage(storage, k_block % Traits::kStages, next_block, storage.task, activation_map,
                      weight_map);
        }
        sync_execution_scope<Traits>();
      }
    }
    sync_execution_scope<Traits>();
    invalidate_barriers(storage);
    sync_execution_scope<Traits>();
  }
#else
  if (blockIdx.x == 0 && threadIdx.x == 0) asm volatile("trap;");
#endif
}

struct KernelResources {
  int32_t blocks_per_sm;
  int32_t num_regs;
  int32_t local_memory_bytes;
  int32_t dynamic_smem_bytes;
};

template <typename Traits>
inline cudaError_t configure_and_query_kernel(KernelResources* resources) {
  auto kernel = grouped_bf16_persistent_offsets_kernel<Traits>;
  int32_t const dynamic_smem = static_cast<int32_t>(persistent_dynamic_smem_bytes<Traits>());
  cudaError_t status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem);
  if (status != cudaSuccess) return status;
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) return status;
  int32_t blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, Traits::kThreads,
                                                         dynamic_smem);
  if (status != cudaSuccess) return status;
  resources->blocks_per_sm = blocks_per_sm;
  resources->num_regs = attributes.numRegs;
  resources->local_memory_bytes = static_cast<int32_t>(attributes.localSizeBytes);
  resources->dynamic_smem_bytes = dynamic_smem;
  return cudaSuccess;
}

template <typename Traits>
inline cudaError_t launch_persistent_family(__nv_bfloat16* output, __nv_bfloat16 const* activation,
                                            __nv_bfloat16 const* weights, int64_t const* offsets,
                                            unsigned long long* task_counter, int64_t rows,
                                            int32_t num_experts, int32_t sm_count,
                                            KernelResources const& resources,
                                            CUtensorMap const& activation_map,
                                            CUtensorMap const& weight_map, cudaStream_t stream) {
  uint64_t const n_tiles = static_cast<uint64_t>(kShapeN / Traits::kBlockN);
  uint64_t const m_tiles_upper =
      ceil_div_nonnegative(rows, Traits::kBlockM) + static_cast<uint64_t>(num_experts);
  uint64_t cluster_tasks_upper;
  if constexpr (Traits::kSwapAB) {
    uint64_t const n_clusters = (n_tiles + Traits::kClusterM - 1) / Traits::kClusterM;
    cluster_tasks_upper = m_tiles_upper * n_clusters;
  } else {
    cluster_tasks_upper = ((m_tiles_upper + Traits::kClusterM - 1) / Traits::kClusterM) * n_tiles;
  }
  uint64_t const resident_blocks = static_cast<uint64_t>(std::max(sm_count, 1)) *
                                   static_cast<uint64_t>(std::max(resources.blocks_per_sm, 1));
  uint64_t resident_clusters = resident_blocks / Traits::kClusterM;
  resident_clusters = std::max<uint64_t>(resident_clusters, 1);
  uint64_t const grid_clusters =
      std::max<uint64_t>(1, std::min(cluster_tasks_upper, resident_clusters));
  int32_t const grid_blocks = static_cast<int32_t>(grid_clusters * Traits::kClusterM);

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(grid_blocks, 1, 1);
  config.blockDim = dim3(Traits::kThreads, 1, 1);
  config.dynamicSmemBytes = persistent_dynamic_smem_bytes<Traits>();
  config.stream = stream;
  cudaLaunchAttribute cluster_attribute{};
  cluster_attribute.id = cudaLaunchAttributeClusterDimension;
  cluster_attribute.val.clusterDim.x = Traits::kClusterM;
  cluster_attribute.val.clusterDim.y = 1;
  cluster_attribute.val.clusterDim.z = 1;
  config.attrs = &cluster_attribute;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, grouped_bf16_persistent_offsets_kernel<Traits>, output,
                            activation, weights, offsets, task_counter, rows, num_experts,
                            activation_map, weight_map);
}

}  // namespace flashinfer::sm90_push_bf16_persistent
