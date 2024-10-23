/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 *Dao.
 ******************************************************************************/

#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"
#include "named_barrier.cuh"

namespace flashinfer {

///////////////////////////////////////////////////////////////////////////////

struct SingleTileScheduler {
 public:
  // Host side kernel arguments
  struct Arguments {
    int const num_qo_tiles, num_head;
  };

  // Device side kernel params
  struct Params {};

  static Params to_underlying_arguments(Arguments const& args) { return {}; }

  static dim3 get_grid_dim(Arguments const& args, int num_sm) {
    return {uint32_t(args.num_qo_tiles), uint32_t(args.num_head)};
  }

  struct WorkTileInfo {
    int q_tile_idx = 0;
    int head_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const { return is_valid_tile; }

    CUTLASS_DEVICE
    cute::tuple<int32_t, int32_t> get_block_coord(Params const& params) const {
      return {q_tile_idx, head_idx};
    }
  };

  CUTLASS_DEVICE
  SingleTileScheduler() {}

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work() const { return {int(blockIdx.x), int(blockIdx.y), true}; }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

  CUTLASS_DEVICE
  void broadcast_next_work(WorkTileInfo& current_work) const {}

  template <bool IsProducer = false>
  CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params,
                                            WorkTileInfo const& current_work) const {
    return {-1, -1, false};
  }
};

}  // namespace flashinfer
