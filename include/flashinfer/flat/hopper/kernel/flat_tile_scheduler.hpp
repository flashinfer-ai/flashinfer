/*
 * Copyright (c) 2025 by FlashInfer team.
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
 */
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/kernel_hardware_info.h"

namespace flat::kernel {

using namespace cute;

struct GQATag {};  //         num_q_heads == ratio * num_k_heads == ratio * num_v_heads
struct GVATag {};  // ratio * num_q_heads == ratio * num_k_heads ==         num_v_heads

template <typename GroupingTag = GQATag>
struct WorkDesc {
  // coord
  int32_t seq_idx;
  int32_t private_q_head_idx;
  int32_t private_v_head_idx;
  int64_t tok_offset;  // offset to the start of the start

  // shape
  int64_t seq_len;

  // update by mainloop
  int32_t tile_idx = 0;

  template <typename Params>
  CUTE_DEVICE bool is_valid(Params const& params) {
    return seq_idx >= 0 && seq_idx < params.num_seqs;
  }

  CUTE_DEVICE int32_t q_head_idx() const { return private_q_head_idx; }

  CUTE_DEVICE int32_t k_head_idx() const {
    if constexpr (std::is_same_v<GroupingTag, GQATag>) {
      return private_v_head_idx;
    } else if constexpr (std::is_same_v<GroupingTag, GVATag>) {
      return private_q_head_idx;
    } else {
      static_assert(dependent_false<GroupingTag>, "unknown grouping relation");
    }
  }

  CUTE_DEVICE int32_t v_head_idx() const { return private_v_head_idx; }

  CUTE_DEVICE int32_t o_head_idx() const {
    if constexpr (std::is_same_v<GroupingTag, GQATag>) {
      return private_q_head_idx;
    } else if constexpr (std::is_same_v<GroupingTag, GVATag>) {
      return private_v_head_idx;
    } else {
      static_assert(dependent_false<GroupingTag>, "unknown grouping relation");
    }
  }
};

template <typename GroupingTag = GQATag>
struct IndividualTileScheduler {
  struct Params {
    dim3 grid;
    int32_t num_seqs;
    int32_t num_q_heads;
    int32_t num_v_heads;
  };

  bool scheduled = false;  // a once flag

  CUTE_DEVICE
  IndividualTileScheduler(Params const& params) {}

  template <typename ProblemSize, typename ClusterShape, typename TileShape>
  static Params to_underlying_arguments(ProblemSize const& problem_size,
                                        cutlass::KernelHardwareInfo const& hw_info,
                                        ClusterShape const& cluster_shape,
                                        TileShape const& tile_shape) {
    dim3 grid(0, 1, 1);
    if constexpr (std::is_same_v<GroupingTag, GQATag>) {
      grid.x = problem_size.num_seqs * problem_size.num_q_heads;
    } else if constexpr (std::is_same_v<GroupingTag, GVATag>) {
      grid.x = problem_size.num_seqs * problem_size.num_v_heads;
    } else {
      static_assert(dependent_false<GroupingTag>, "unknown grouping relation");
    }
    DPRINTF(
        "to_underlying_arguments: grid:{.x:%d, .y:%d, .z:%d}, num_seqs:%d, num_q_heads:%d, "
        "num_v_heads:%d\n",
        grid.x, grid.y, grid.z, problem_size.num_seqs, problem_size.num_q_heads,
        problem_size.num_v_heads);
    return {
        .grid = grid,
        .num_seqs = problem_size.num_seqs,
        .num_q_heads = problem_size.num_q_heads,
        .num_v_heads = problem_size.num_v_heads,
    };
  }

  static dim3 get_grid_shape(Params const& params) { return params.grid; }

  template <typename ProblemSize>
  CUTE_DEVICE WorkDesc<GroupingTag> get_next_work(Params params, ProblemSize const& problem_size) {
    int32_t seq_idx;
    ;
    int32_t q_head_idx;
    int32_t v_head_idx;
    if constexpr (std::is_same_v<GroupingTag, GQATag>) {
      seq_idx = blockIdx.x / params.num_q_heads;
      q_head_idx = blockIdx.x % params.num_q_heads;
      v_head_idx = q_head_idx / (params.num_q_heads / params.num_v_heads);
    } else if constexpr (std::is_same_v<GroupingTag, GVATag>) {
      seq_idx = blockIdx.x / params.num_v_heads;
      v_head_idx = blockIdx.x % params.num_v_heads;
      q_head_idx = v_head_idx / (params.num_v_heads / params.num_q_heads);
    } else {
      static_assert(dependent_false<GroupingTag>, "unknown grouping relation");
    }

    int64_t s = problem_size.cu_seqlens[seq_idx];
    int64_t e = problem_size.cu_seqlens[seq_idx + 1];
    int64_t seq_len = e - s;

    if (scheduled) {
      seq_idx = -1;
    } else {
      scheduled = true;
      DPRINTF0_W(
          "get_next_work: this_work={seq_idx:%d q_head_idx:%d v_head_idx:%d tok_offset:%lld "
          "seq_len:%lld}\n",
          seq_idx, q_head_idx, v_head_idx, s, seq_len);
    }

    return {
        .seq_idx = seq_idx,
        .private_q_head_idx = q_head_idx,
        .private_v_head_idx = v_head_idx,
        .tok_offset = s,
        .seq_len = seq_len,
    };
  }
};

}  // namespace flat::kernel
