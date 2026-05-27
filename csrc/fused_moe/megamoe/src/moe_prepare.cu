
#pragma once
#ifndef MOE_PREPARE_CU
#define MOE_PREPARE_CU

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cstdint>

#include "moe_internal.h"

#ifndef __SIZEOF_INT128__
static_assert(false,
              "This module currently needs int128. You're host compiler does "
              "not support it.")
#endif

#define FULL_MASK 0xFFFFFFFFU

    namespace moe_monokernel {

  // We use an uint128 to store 16 uint8
  typedef __uint128_t uint8x16_t;

  /**
   * @brief 16-byte allreduce summation within a warp
   */
  __device__ static inline uint8x16_t allreduce_sum_across_warp(uint8x16_t val) {
    uint64_t val_lo = val & 0xFFFFFFFFFFFFFFFFU;
    uint64_t val_hi = val >> 64;
    for (int offset = 16; offset > 0; offset /= 2) {
      val_lo += __shfl_xor_sync(FULL_MASK, val_lo, offset, 32);
      val_hi += __shfl_xor_sync(FULL_MASK, val_hi, offset, 32);
    }
    return val_lo | ((uint8x16_t)val_hi << 64);
  }

  /**
   * @brief Prefix sum of all uint8s in an uint8x16_t
   */
  __device__ static inline uint8x16_t prefix_sum_over_bytes(uint8x16_t val) {
    val += val << 8;
    val += val << 16;
    val += val << 32;
    val += val << 64;
    return val;
  }

  /**
   * @brief Prepares the top-K single-pass MoE computation for BS > 8.
   *
   * Reads @c shmem->topk_ids_flat and @c shmem->topk_weights_flat (filled by
   * topK_BS64) and builds the sorted data structures for the single-pass
   * up/down projection:
   *
   *  - @c shmem->experts[]          — unique experts with their sorted ranges
   *  - @c shmem->expert_count       — number of unique experts
   *  - @c shmem->path.bs64.token_indexes_topk — original token index per sorted
   * slot
   *  - @c shmem->path.bs64.token_weights      — routing weight per sorted slot
   *
   * @param num_tokens  Number of active tokens (not virtual rows).
   * @param top_k       Number of experts per token.
   * @param shmem       Shared memory struct to read from and write to.
   */
  template <typename Dims>
  __device__ static void prepare_moe_topk_BSx_Ey(std::uint32_t num_tokens, std::uint32_t top_k,
                                                 MoE_SHM<Dims>* __restrict__ shmem) {
    using CoreDims = MoECoreDims<Dims>;
    constexpr uint32_t MAX_TOPK = MoE_SHM<Dims>::MAX_TOPK;

    const std::uint32_t virtual_batch = num_tokens * top_k;

    typename MoE_SHM<Dims>::U::SortData* shm = &shmem->u.sorting;
    auto& counters = shm->counters;
    auto& total_counts = shm->total_counts;

    if (threadIdx.x < CoreDims::THREADS_PER_WARP) {
      for (unsigned e = 0; e < Dims::NUM_EXPERTS; ++e) counters[e][threadIdx.x] = 0;

      // topk_ids_flat is laid out as [token * MAX_TOPK + k] with valid
      // entries only for k < top_k.  Iterate using the strided layout
      // so we never read the uninitialised slots at k >= top_k.
      for (unsigned i = threadIdx.x; i < virtual_batch; i += CoreDims::THREADS_PER_WARP) {
        unsigned tok = i / top_k;
        unsigned k = i % top_k;
        counters[shmem->topk_ids_flat[tok * MAX_TOPK + k]][threadIdx.x]++;
      }

      __syncwarp();

      for (unsigned e = threadIdx.x; e < Dims::NUM_EXPERTS; e += CoreDims::THREADS_PER_WARP) {
        std::uint32_t sum = 0;
        for (unsigned i = 0; i < CoreDims::THREADS_PER_WARP; ++i) {
          std::uint32_t prior = sum;
          sum += counters[e][(i + threadIdx.x) % CoreDims::THREADS_PER_WARP];
          counters[e][(i + threadIdx.x) % CoreDims::THREADS_PER_WARP] = prior;
        }
        total_counts[e] = sum;
      }

      __syncwarp();

      if (threadIdx.x == 0) {
        std::uint32_t sum = 0;
        std::uint32_t expert_count = 0;
        for (unsigned e = 0; e < Dims::NUM_EXPERTS; ++e) {
          std::uint32_t local_count = total_counts[e];
          if (local_count > 0) {
            std::uint32_t prior = sum;
            total_counts[e] = prior;
            sum += local_count;
            shmem->experts[expert_count].first_token = prior;
            shmem->experts[expert_count].last_token = sum;
            shmem->experts[expert_count].id = e;
            expert_count++;
          }
        }
        shmem->expert_count = expert_count;
      }

      __syncwarp();

      for (unsigned i = threadIdx.x; i < virtual_batch; i += CoreDims::THREADS_PER_WARP) {
        unsigned tok = i / top_k;
        unsigned k = i % top_k;
        std::uint32_t e = shmem->topk_ids_flat[tok * MAX_TOPK + k];
        unsigned offset = counters[e][threadIdx.x];
        unsigned index = total_counts[e] + offset;
        counters[e][threadIdx.x] = offset + 1;

        shmem->path.bs64.token_indexes_topk[index] = (std::uint16_t)tok;
        shmem->path.bs64.token_weights[index] = shmem->topk_weights_flat[tok * MAX_TOPK + k];
      }
    }
  }

}  // namespace moe_monokernel

#endif
