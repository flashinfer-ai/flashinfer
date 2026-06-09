/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <cuda_runtime.h>

#include <cstdint>

namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled {

enum class GemmType {
  Normal,
  Batched,
  MGroupedContiguous,
  MGroupedMasked,
  MGroupedContiguousWithPsumLayout,
  MGroupedContiguousWithZeroPadding,
};

// Flat = tma_store-compatible D layout (per-batch D, no per-row expert indirection):
// Normal / Batched / MGroupedMasked. Non-flat (MGroupedContiguous{,WithPsumLayout}) uses
// flat A across experts via grouped_layout indirection → R2G epilogue with smem_O union.
constexpr bool is_flat_gemm(GemmType t) {
  return t == GemmType::Normal || t == GemmType::Batched || t == GemmType::MGroupedMasked;
}

// Compile-time SM count hint for L2 swizzle group size selection.
// Override at build time via -DSM120_DEFAULT_NUM_SMS=N (CMakeLists auto-detects
// host GPU SM count via nvidia-smi; falls back to 188 / RTX PRO 6000 default).
#ifndef SM120_DEFAULT_NUM_SMS
#define SM120_DEFAULT_NUM_SMS 188
#endif
constexpr int kDefaultNumSMs = SM120_DEFAULT_NUM_SMS;

// L2 tile-swizzle group size selection: candidate (8 or 16) minimizing per-group L2 footprint:
//   usage = candidate * BlockM + ceil(kNumSMs / candidate) * BlockN
template <int BlockM, int BlockN, int kNumSMs = kDefaultNumSMs>
constexpr int get_num_1d_blocks_per_group() {
  constexpr int usage8 = 8 * BlockM + ((kNumSMs + 7) / 8) * BlockN;
  constexpr int usage16 = 16 * BlockM + ((kNumSMs + 15) / 16) * BlockN;
  return (usage8 <= usage16) ? 8 : 16;
}

template <GemmType kGemmType, int BlockM, int BlockN>
struct Scheduler {
  static_assert(
      kGemmType == GemmType::Normal || kGemmType == GemmType::Batched ||
          kGemmType == GemmType::MGroupedContiguous ||
          kGemmType == GemmType::MGroupedContiguousWithPsumLayout ||
          kGemmType == GemmType::MGroupedContiguousWithZeroPadding ||
          kGemmType == GemmType::MGroupedMasked,
      "Scheduler supports Normal, Batched, MGroupedContiguous, "
      "MGroupedContiguousWithPsumLayout, MGroupedContiguousWithZeroPadding, MGroupedMasked.");

  static constexpr int kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<BlockM, BlockN>();

  int32_t shape_m;
  int32_t shape_n;
  int32_t num_groups;
  int32_t num_n_blocks;
  int32_t const* grouped_layout;

  int32_t current_iter = -1;
  int32_t current_group_idx = 0;
  int32_t current_psum_m = 0;
  int32_t prev_psum_m = 0;
  int32_t current_m_block_cumsum = 0;
  int32_t num_m_blocks = 0;

  __device__ __forceinline__ explicit Scheduler(int shape_m_, int shape_n_, int num_groups_,
                                                int32_t const* grouped_layout_ = nullptr)
      : shape_m(shape_m_),
        shape_n(shape_n_),
        num_groups(num_groups_),
        num_n_blocks((shape_n_ + BlockN - 1) / BlockN),
        grouped_layout(grouped_layout_) {
    if constexpr (kGemmType == GemmType::Normal || kGemmType == GemmType::Batched ||
                  kGemmType == GemmType::MGroupedContiguous) {
      num_m_blocks = (shape_m_ + BlockM - 1) / BlockM;
    } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
      current_psum_m = grouped_layout[0];
      num_m_blocks = (current_psum_m + BlockM - 1) / BlockM;
    }
  }

  __device__ __forceinline__ void get_swizzled_block_idx_local(int32_t block_idx,
                                                               int32_t num_m_blocks_local,
                                                               int32_t& m_block_idx,
                                                               int32_t& n_block_idx) const {
    constexpr int K = kNum1DBlocksPerGroup;
    int32_t num_blocks_per_group = num_n_blocks * K;
    int32_t group_idx = block_idx / num_blocks_per_group;
    int32_t first_block_idx = group_idx * K;
    int32_t in_group_idx = block_idx % num_blocks_per_group;
    int32_t num_blocks_in_group = cute::min(K, num_m_blocks_local - first_block_idx);
    m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
    n_block_idx = in_group_idx / num_blocks_in_group;
  }

  __device__ __forceinline__ bool get_next_block(int32_t& m_block_idx, int32_t& n_block_idx) {
    int64_t next_block_idx = static_cast<int64_t>(++current_iter) * gridDim.x + blockIdx.x;

    if constexpr (kGemmType == GemmType::Normal) {
      int32_t total_blocks = num_m_blocks * num_n_blocks;
      if (next_block_idx >= total_blocks) return false;
      get_swizzled_block_idx_local(next_block_idx, num_m_blocks, m_block_idx, n_block_idx);
      return true;
    } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
      int32_t total_blocks = num_m_blocks * num_n_blocks;
      while (next_block_idx < total_blocks) {
        get_swizzled_block_idx_local(next_block_idx, num_m_blocks, m_block_idx, n_block_idx);
        if (grouped_layout[m_block_idx * BlockM] >= 0) return true;
        next_block_idx = static_cast<int64_t>(++current_iter) * gridDim.x + blockIdx.x;
      }
      return false;
    } else if constexpr (kGemmType == GemmType::Batched) {
      int32_t blocks_per_batch = num_m_blocks * num_n_blocks;
      int32_t total_blocks = blocks_per_batch * num_groups;
      if (next_block_idx >= total_blocks) return false;
      current_group_idx = next_block_idx / blocks_per_batch;
      int32_t in_batch_idx = next_block_idx % blocks_per_batch;
      get_swizzled_block_idx_local(in_batch_idx, num_m_blocks, m_block_idx, n_block_idx);
      return true;
    } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {
      while (true) {
        if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks) break;
        if (++current_group_idx == num_groups) return false;

        prev_psum_m = current_psum_m;
        current_psum_m = grouped_layout[current_group_idx];
        current_m_block_cumsum += num_m_blocks;
        int32_t diff = current_psum_m - prev_psum_m;
        num_m_blocks = (diff > 0) ? (diff + BlockM - 1) / BlockM : 0;
      }

      auto remain_blocks = next_block_idx - current_m_block_cumsum * num_n_blocks;
      get_swizzled_block_idx_local(remain_blocks, num_m_blocks, m_block_idx, n_block_idx);
      return true;
    } else if constexpr (kGemmType == GemmType::MGroupedContiguousWithZeroPadding) {
      // ZeroPadding MoE: unpadded raw grouped_layout interpreted as
      // token_offset[E+1] int32 cumsum WITH leading 0 (= [0, m0, m0+m1, ..., M_total]),
      // plain row-major (no L2 swizzle).
      while (true) {
        if (current_group_idx >= num_groups) return false;

        prev_psum_m = grouped_layout[current_group_idx];
        current_psum_m = grouped_layout[current_group_idx + 1];
        int32_t shape_m_cur = current_psum_m - prev_psum_m;
        num_m_blocks = (shape_m_cur + BlockM - 1) / BlockM;

        int32_t next_cumsum = current_m_block_cumsum + num_m_blocks;
        if (next_block_idx < next_cumsum * num_n_blocks) break;

        ++current_group_idx;
        current_m_block_cumsum = next_cumsum;
      }

      auto remain_blocks = next_block_idx - current_m_block_cumsum * num_n_blocks;
      m_block_idx = remain_blocks / num_n_blocks;
      n_block_idx = remain_blocks % num_n_blocks;
      return true;
    } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
      while (true) {
        // End of the task
        if (current_group_idx == num_groups) return false;

        // Within current group
        current_psum_m = grouped_layout[current_group_idx];
        num_m_blocks = (current_psum_m + BlockM - 1) / BlockM;
        if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks) break;

        // Move to check the next group
        current_group_idx++;
        current_m_block_cumsum += num_m_blocks;
      }

      auto remain_blocks = next_block_idx - current_m_block_cumsum * num_n_blocks;
      get_swizzled_block_idx_local(remain_blocks, num_m_blocks, m_block_idx, n_block_idx);
      return true;
    }
  }

  __device__ __forceinline__ int32_t get_m_offset() const {
    if constexpr (kGemmType == GemmType::Normal || kGemmType == GemmType::Batched ||
                  kGemmType == GemmType::MGroupedContiguous)
      return 0;
    else
      return prev_psum_m;
  }
  __device__ __forceinline__ int32_t get_m_boundary() const {
    if constexpr (kGemmType == GemmType::Normal || kGemmType == GemmType::Batched ||
                  kGemmType == GemmType::MGroupedContiguous)
      return shape_m;
    else
      return current_psum_m;
  }
  __device__ __forceinline__ int32_t get_expert_idx(int32_t m_block_idx = 0) const {
    if constexpr (kGemmType == GemmType::MGroupedContiguous) {
      return grouped_layout[m_block_idx * BlockM];
    }
    return current_group_idx;
  }
};

}  // namespace flashinfer::gemm::mxfp8_cute_sm120::sm120_blockscaled
