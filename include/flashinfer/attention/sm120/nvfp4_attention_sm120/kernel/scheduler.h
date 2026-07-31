/*
 * Copyright (c) 2025 by SageAttention team.
 *
 * This code is based on code from FlashAttention3, https://github.com/Dao-AILab/flash-attention
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
 * except in compliance with the License. You may obtain a copy of the License at
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

namespace nvfp4_attention {

class SingleTileScheduler {
 public:
  struct Arguments {
    int const num_blocks_m;
    int const num_head;
    int const num_batch;
    int const* tile_count_semaphore = nullptr;
    bool const is_causal = false;
  };

  struct Params {};

  static Params to_underlying_arguments(Arguments const& args) { return {}; }

  static dim3 get_grid_dim(Arguments const& args, int num_sm) {
    return {uint32_t(args.num_blocks_m), uint32_t(args.num_head), uint32_t(args.num_batch)};
  }

  struct WorkTileInfo {
    int M_idx = 0;
    int H_idx = 0;
    int B_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const { return is_valid_tile; }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t> get_block_coord(Params const& params) const {
      return {M_idx, H_idx, B_idx};
    }

    CUTLASS_DEVICE
    WorkTileInfo get_next_work(Params const& params) const { return {-1, -1, -1, false}; }
  };

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work() const {
    return {int(blockIdx.x), int(blockIdx.y), int(blockIdx.z), true};
  }

  CUTLASS_DEVICE
  WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {-1, -1, -1, false};
  }
};

class StaticPersistentTileScheduler {
 public:
  struct Arguments {
    int const num_blocks_m;
    int const num_head;
    int const num_batch;
    int const* tile_count_semaphore = nullptr;
    bool const is_causal = false;
  };

  struct Params {
    int total_blocks;
    int num_blocks_m;
    cutlass::FastDivmod m_block_divmod;
    cutlass::FastDivmod head_divmod;
    bool is_causal;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.num_blocks_m * args.num_head * args.num_batch, args.num_blocks_m,
            cutlass::FastDivmod(args.num_blocks_m), cutlass::FastDivmod(args.num_head),
            args.is_causal};
  }

  static dim3 get_grid_dim(Arguments const& args, int num_sm) { return {uint32_t(num_sm)}; }

  struct WorkTileInfo {
    int tile_idx;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const { return tile_idx < params.total_blocks; }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t, int32_t> get_block_coord(Params const& params) const {
      int m_block, bidh, bidb;
      bidb = params.head_divmod.divmod(bidh, params.m_block_divmod.divmod(m_block, tile_idx));
      if (params.is_causal) {
        m_block = params.num_blocks_m - 1 - m_block;
      }
      return {m_block, bidh, bidb};
    }
  };

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work() const { return {int(blockIdx.x)}; }

  CUTLASS_DEVICE
  WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {current_work.tile_idx + int(gridDim.x)};
  }
};

class DynamicPersistentTileScheduler {
 public:
  struct Arguments {
    int const num_blocks_m;
    int const num_head;
    int const num_batch;
    int const* tile_count_semaphore;
    bool const is_causal = false;
  };

  struct Params {
    int const total_blocks;
    int const num_blocks_m;
    cutlass::FastDivmod const m_block_divmod;
    cutlass::FastDivmod const head_divmod;
    bool const is_causal;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.num_blocks_m * args.num_head * args.num_batch, args.num_blocks_m,
            cutlass::FastDivmod(args.num_blocks_m), cutlass::FastDivmod(args.num_head),
            args.is_causal};
  }

  static dim3 get_grid_dim(Arguments const& args, int num_sm) { return {uint32_t(num_sm)}; }

  using WorkTileInfo = StaticPersistentTileScheduler::WorkTileInfo;

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work() const { return {int(blockIdx.x)}; }

  CUTLASS_DEVICE
  WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {current_work.tile_idx + int(gridDim.x)};
  }
};

class StaticPersistentTileSchedulerOld {
 private:
  int current_work_linear_idx_;
  cutlass::FastDivmod m_block_divmod, head_divmod;
  int const total_blocks;

 public:
  struct WorkTileInfo {
    int M_idx = 0;
    int H_idx = 0;
    int B_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_HOST_DEVICE
    bool is_valid() const { return is_valid_tile; }

    CUTLASS_HOST_DEVICE
    static WorkTileInfo invalid_work_tile() { return {-1, -1, -1, false}; }
  };

 public:
  CUTLASS_DEVICE explicit StaticPersistentTileSchedulerOld(
      cutlass::FastDivmod const& m_block_divmod_, cutlass::FastDivmod const& head_divmod_,
      int const total_blocks_)
      : m_block_divmod(m_block_divmod_), head_divmod(head_divmod_), total_blocks(total_blocks_) {
#if defined(__CUDA_ARCH__)
    current_work_linear_idx_ = blockIdx.x;
#else
    CUTLASS_ASSERT(false && "This line should never be reached");
#endif
  }

  CUTLASS_DEVICE
  WorkTileInfo get_current_work() const {
    return get_current_work_for_linear_idx(current_work_linear_idx_);
  }

  CUTLASS_DEVICE
  WorkTileInfo get_current_work_for_linear_idx(int linear_idx) const {
    if (linear_idx >= total_blocks) {
      return WorkTileInfo::invalid_work_tile();
    }

    int M_idx, H_idx, B_idx;
    int quotient = m_block_divmod.divmod(M_idx, linear_idx);
    B_idx = head_divmod.divmod(H_idx, quotient);
    return {M_idx, H_idx, B_idx, true};
  }

  CUTLASS_DEVICE
  void advance_to_next_work() { current_work_linear_idx_ += int(gridDim.x); }

  CUTLASS_DEVICE
  WorkTileInfo fetch_next_work() {
    WorkTileInfo new_work_tile_info;
    advance_to_next_work();
    new_work_tile_info = get_current_work();
    return new_work_tile_info;
  }
};

}  // namespace nvfp4_attention
