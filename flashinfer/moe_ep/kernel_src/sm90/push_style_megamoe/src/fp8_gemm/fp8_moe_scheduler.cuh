/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-License-Identifier: MIT
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * SPDX-FileCopyrightText: Copyright (c) 2026 FlashInfer team.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <deep_gemm/scheduler.cuh>

namespace flashinfer::sm90_push_fp8 {

using Fp8MoeSchedulerInput = deep_gemm::GroupedWithOffsetSchedulerInput;

// Each scheduled task owns one M tile and one adjacent gate/up N-tile pair.
template <uint32_t SHAPE_N, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups,
          uint32_t kNumTMAMulticast, uint32_t kNumNBlocks = ceil_div(SHAPE_N, BLOCK_N),
          uint32_t kNumNBlocksPerGroup = 16>
struct Fp8MoeFc1Scheduler {
  DG_STATIC_ASSERT(SHAPE_N % (2 * BLOCK_N) == 0,
                   "Interleaved gate/up weights must contain complete tile pairs");
  static constexpr bool kIsFp8MoeFc1Scheduler = true;
  static constexpr uint32_t kNumPairs = kNumNBlocks / 2;

  int current_iter = -1;
  uint32_t curr_group_idx = 0;
  uint32_t curr_cumsum = 0;
  int64_t m_offset = 0;
  int64_t m_padded_4_offset = 0;
  int64_t m_boundary = 0;
  int64_t* problem_m_offsets = nullptr;

  using Input = Fp8MoeSchedulerInput;

  __device__ __forceinline__ Fp8MoeFc1Scheduler() = default;

  __device__ __forceinline__ explicit Fp8MoeFc1Scheduler(Input& input)
      : problem_m_offsets(input.problem_m_offsets) {}

  __device__ __forceinline__ uint32_t get_global_m_idx(uint32_t block_idx) const {
    return static_cast<uint32_t>(m_offset) + block_idx * BLOCK_M;
  }

  __device__ __forceinline__ uint32_t get_global_n_idx_phase(uint32_t pair_idx,
                                                             uint32_t phase) const {
    return curr_group_idx * SHAPE_N + (2 * pair_idx + phase) * BLOCK_N;
  }

  __device__ __forceinline__ uint32_t get_global_scales_a_idx(uint32_t block_idx) const {
    return static_cast<uint32_t>(m_padded_4_offset) + block_idx * BLOCK_M;
  }

  __device__ __forceinline__ uint32_t get_scales_b_row_gate(uint32_t pair_idx) const {
    return curr_group_idx * (SHAPE_N / 128) + 2 * pair_idx;
  }

  __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& pair_idx) {
    ++current_iter;
    auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
    uint32_t num_m_blocks;
    while (true) {
      if (curr_group_idx == kNumGroups) return false;
      m_offset = __ldg(problem_m_offsets + curr_group_idx);
      m_boundary = __ldg(problem_m_offsets + curr_group_idx + 1);
      m_padded_4_offset = deep_gemm::compute_padded_offset(m_offset, curr_group_idx);
      auto const m = m_boundary - m_offset;
      num_m_blocks = m > 0 ? static_cast<uint32_t>(ceil_div(m, static_cast<int64_t>(BLOCK_M))) : 0u;
      auto const current_m_block_cumsum = curr_cumsum + num_m_blocks;
      if (next_block_idx < current_m_block_cumsum * kNumPairs) break;
      ++curr_group_idx;
      curr_cumsum = current_m_block_cumsum;
    }

    deep_gemm::get_swizzled_block_idx<kNumTMAMulticast, kNumPairs, kNumNBlocksPerGroup>(
        num_m_blocks, next_block_idx - curr_cumsum * kNumPairs, m_block_idx, pair_idx);
    return true;
  }
};

}  // namespace flashinfer::sm90_push_fp8
