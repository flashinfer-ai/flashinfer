/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_NAMED_BARRIERS_CUH_
#define FLASHINFER_ATTENTION_HOPPER_NAMED_BARRIERS_CUH_

#include <cuda_runtime.h>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"

namespace flashinfer {

// Enumerates the reserved named barriers to avoid potential conflicts

enum class NamedBarriers {
  kQueryEmpty = 0,
  kValueEmpty = 1,
  kWarpSchedulerWG1 = 2,
  kWarpSchedulerWG2 = 3,
  kWarpSchedulerWG3 = 4,
  kPrefetchIndices = 5,
};

__device__ __forceinline__ int get_warp_group_barrier_idx(int warp_group_idx) {
  return static_cast<int>(NamedBarriers::kWarpSchedulerWG1) + warp_group_idx - 1;
}

template <int num_consumer_warp_groups>
__device__ __forceinline__ int get_next_consumer_warp_group_idx() {
  static_assert(num_consumer_warp_groups == 2 || num_consumer_warp_groups == 3);
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  if constexpr (num_consumer_warp_groups == 2) {
    // 1 -> 2, 2 -> 1
    return 3 - warp_group_idx;
  } else {
    // num_consumer_warp_groups == 3
    // 1 -> 2, 2 -> 3, 3 -> 1
    return (warp_group_idx % 3) + 1;
  }
}

template <int num_consumer_warp_groups>
__device__ __forceinline__ int get_prev_consumer_warp_group_idx() {
  static_assert(num_consumer_warp_groups == 2 || num_consumer_warp_groups == 3);
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  if constexpr (num_consumer_warp_groups == 2) {
    // 1 -> 2, 2 -> 1
    return 3 - warp_group_idx;
  } else {
    // num_consumer_warp_groups == 3
    // 1 -> 3, 2 -> 1, 3 -> 2
    return ((warp_group_idx + 1) % 3) + 1;
  }
}

template <typename Ktraits, bool UseSchedulerBarrier>
struct WarpScheduler {
  constexpr static int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static CUTLASS_DEVICE void barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      cutlass::arch::NamedBarrier::sync(
          NUM_MMA_THREADS, get_warp_group_barrier_idx(cutlass::canonical_warp_group_idx()));
    }
  }

  static CUTLASS_DEVICE void barrier_arrive() {
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup ||
                  NUM_MMA_THREADS == 3 * cutlass::NumThreadsPerWarpGroup);
    if constexpr (NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup) {
      cutlass::arch::NamedBarrier::arrive(
          NUM_MMA_THREADS, get_warp_group_barrier_idx(get_next_consumer_warp_group_idx<2>()));
    } else {
      cutlass::arch::NamedBarrier::arrive(
          NUM_MMA_THREADS, get_warp_group_barrier_idx(get_next_consumer_warp_group_idx<3>()));
      cutlass::arch::NamedBarrier::arrive(
          NUM_MMA_THREADS, get_warp_group_barrier_idx(get_prev_consumer_warp_group_idx<3>()));
    }
  }

  static CUTLASS_DEVICE void mma_init() {
    // Tell producer (warp 0) that smem_q is ready
    cutlass::arch::NamedBarrier::arrive(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                        /*id=*/static_cast<int>(NamedBarriers::kQueryEmpty));
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NUM_MMA_THREADS == 2 * cutlass::NumThreadsPerWarpGroup ||
                  NUM_MMA_THREADS == 3 * cutlass::NumThreadsPerWarpGroup);
    if (cutlass::canonical_warp_group_idx() > 1) {
      cutlass::arch::NamedBarrier::arrive(
          NUM_MMA_THREADS, /*id=*/static_cast<int>(NamedBarriers::kWarpSchedulerWG1));
    }
    if constexpr (NUM_MMA_THREADS == 3 * cutlass::NumThreadsPerWarpGroup) {
      if (cutlass::canonical_warp_group_idx() > 2) {
        cutlass::arch::NamedBarrier::arrive(
            NUM_MMA_THREADS, /*id=*/static_cast<int>(NamedBarriers::kWarpSchedulerWG2));
      }
    }
  }

};  // struct WarpScheduler

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_NAMED_BARRIERS_CUH_
