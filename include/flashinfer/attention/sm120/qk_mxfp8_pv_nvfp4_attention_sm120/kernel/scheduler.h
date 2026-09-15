/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * This code is based on FlashAttention-3:
 * https://github.com/Dao-AILab/flash-attention
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang,
 * Vijay Thakkar, Pradeep Ramani, Tri Dao.
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

#include "cute/tensor.hpp"
#include "cutlass/fast_math.h"

namespace qk_mxfp8_pv_nvfp4_attention {

class SingleTileScheduler {
 public:
  struct Arguments {
    int num_blocks_m;
    int num_heads;
    int batch_size;
  };

  struct Params {
    int total_blocks;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.num_blocks_m * args.num_heads * args.batch_size};
  }

  static dim3 get_grid_dim(Arguments const& args, [[maybe_unused]] int num_sms) {
    return {uint32_t(args.num_blocks_m), uint32_t(args.num_heads), uint32_t(args.batch_size)};
  }

  struct WorkTileInfo {
    int m_block;
    int head_idx;
    int batch_idx;
    bool valid;

    CUTLASS_DEVICE bool is_valid([[maybe_unused]] Params const& params) const { return valid; }

    CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t> get_block_coord(
        [[maybe_unused]] Params const& params) const {
      return {m_block, head_idx, batch_idx};
    }
  };

  CUTLASS_DEVICE WorkTileInfo get_initial_work() const {
    return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
  }

  CUTLASS_DEVICE WorkTileInfo
  get_next_work([[maybe_unused]] Params const& params,
                [[maybe_unused]] WorkTileInfo const& current_work) const {
    return {-1, -1, -1, false};
  }
};

class StaticPersistentTileScheduler {
 public:
  struct Arguments {
    int num_blocks_m;
    int num_heads;
    int batch_size;
  };

  struct Params {
    int total_blocks;
    cutlass::FastDivmod m_block_divmod;
    cutlass::FastDivmod head_divmod;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.num_blocks_m * args.num_heads * args.batch_size,
            cutlass::FastDivmod(args.num_blocks_m), cutlass::FastDivmod(args.num_heads)};
  }

  static dim3 get_grid_dim(Arguments const& args, int num_ctas) {
    int const total_blocks = args.num_blocks_m * args.num_heads * args.batch_size;
    return {uint32_t(cute::min(num_ctas, total_blocks))};
  }

  struct WorkTileInfo {
    int tile_idx;

    CUTLASS_DEVICE bool is_valid(Params const& params) const {
      return tile_idx < params.total_blocks;
    }

    CUTLASS_DEVICE cute::tuple<int32_t, int32_t, int32_t> get_block_coord(
        Params const& params) const {
      int m_block, head_idx, batch_idx;
      batch_idx =
          params.head_divmod.divmod(head_idx, params.m_block_divmod.divmod(m_block, tile_idx));
      return {m_block, head_idx, batch_idx};
    }
  };

  CUTLASS_DEVICE WorkTileInfo get_initial_work() const { return {int(blockIdx.x)}; }

  CUTLASS_DEVICE WorkTileInfo get_next_work([[maybe_unused]] Params const& params,
                                            WorkTileInfo const& current_work) const {
    return {current_work.tile_idx + int(gridDim.x)};
  }
};

}  // namespace qk_mxfp8_pv_nvfp4_attention
