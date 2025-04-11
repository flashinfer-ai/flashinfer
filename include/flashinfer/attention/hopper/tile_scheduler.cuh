/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_TILE_SCHEDULER_CUH_
#define FLASHINFER_ATTENTION_HOPPER_TILE_SCHEDULER_CUH_

#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"
#include "named_barrier.cuh"

namespace flashinfer {

struct SingleTileScheduler {
 public:
  // Host side kernel arguments
  struct Arguments {
    int const num_qo_tiles, num_qo_heads, qo_len, kv_len;
    cutlass::FastDivmod group_size_fastdiv;
  };

  // Device side kernel params
  struct Params {
    int const qo_len, kv_len;
    cutlass::FastDivmod group_size_fastdiv;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.qo_len, args.kv_len, args.group_size_fastdiv};
  }

  static dim3 get_grid_dim(Arguments const& args, int num_sm) {
    return {uint32_t(args.num_qo_tiles), uint32_t(args.num_qo_heads)};
  }

  struct WorkTileInfo {
    int q_tile_idx = 0;
    int qo_head_idx = 0;
    int kv_head_idx = 0;
    bool is_valid_tile = false;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const { return is_valid_tile; }

    CUTLASS_DEVICE
    auto get_block_coord(Params const& params) const {
      return cute::tuple{q_tile_idx,      qo_head_idx,   kv_head_idx,   /*qo_indptr=*/0,
                         /*kv_indptr=*/0, params.qo_len, params.kv_len, /*batch_idx=*/0};
    }
  };

  CUTLASS_DEVICE
  SingleTileScheduler() {}

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const& params) const {
    int qo_head_idx = blockIdx.y;
    int kv_head_idx = params.group_size_fastdiv.divide(qo_head_idx);
    return {/*q_tile_idx=*/int(blockIdx.x), qo_head_idx, kv_head_idx, /*is_valid_tile*/ true};
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

  CUTLASS_DEVICE
  void broadcast_next_work(WorkTileInfo& current_work) const {}

  template <bool is_producer = false>
  CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params,
                                            WorkTileInfo const& current_work) const {
    return {-1, -1, false};
  }
};

template <typename IdType>
struct BatchPrefillPersistentTileScheduler {
 public:
  // Host side kernel arguments
  struct Arguments {
    IdType *work_indptr, *head_indices, *qo_tile_indices, *qo_indptr, *kv_indptr, *qo_lens,
        *kv_lens, *batch_indices;
    cutlass::FastDivmod group_size_fastdiv;
    int num_qo_heads;  // placeholder
  };

  // Device side kernel params
  struct Params {
    IdType *work_indptr, *head_indices, *qo_tile_indices, *qo_indptr, *kv_indptr, *qo_lens,
        *kv_lens, *batch_indices;
    cutlass::FastDivmod group_size_fastdiv;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.work_indptr, args.head_indices,  args.qo_tile_indices,
            args.qo_indptr,   args.kv_indptr,     args.qo_lens,
            args.kv_lens,     args.batch_indices, args.group_size_fastdiv};
  }

  static dim3 get_grid_dim(Arguments const& args, int num_sm) { return {(unsigned)num_sm}; }

  struct WorkTileInfo {
    int q_tile_idx = 0;
    int qo_head_idx = 0;
    int kv_head_idx = 0;
    int qo_indptr = 0;
    int kv_indptr = 0;
    int qo_len = 0;
    int kv_len = 0;
    int batch_idx = 0;
    int counter = 0;
    int ptr_begin = 0;
    int ptr_end = 0;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const { return counter + ptr_begin < ptr_end; }

    CUTLASS_DEVICE
    auto get_block_coord(Params const& params) const {
      return cute::tuple{q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr,
                         kv_indptr,  qo_len,      kv_len,      batch_idx};
    }
  };

  CUTLASS_DEVICE
  BatchPrefillPersistentTileScheduler() {}

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const& params) const {
    int ptr_begin = params.work_indptr[blockIdx.x];
    int ptr_end = params.work_indptr[blockIdx.x + 1];
    if (ptr_begin < ptr_end) {
      int work_idx = ptr_begin;
      int qo_head_idx = params.head_indices[work_idx];
      int kv_head_idx = params.group_size_fastdiv.divide(qo_head_idx);
      return {params.qo_tile_indices[work_idx],
              qo_head_idx,
              kv_head_idx,
              params.qo_indptr[work_idx],
              params.kv_indptr[work_idx],
              params.qo_lens[work_idx],
              params.kv_lens[work_idx],
              params.batch_indices[work_idx],
              /*counter=*/0,
              ptr_begin,
              ptr_end};
    } else {
      return {-1, -1, -1, -1, -1, -1, -1, 0, ptr_begin, ptr_end};
    }
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

  CUTLASS_DEVICE
  void broadcast_next_work(WorkTileInfo& current_work) const {}

  template <bool is_producer = false>
  CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params,
                                            WorkTileInfo const& current_work) const {
    int work_idx = current_work.ptr_begin + current_work.counter + 1;
    if (work_idx < current_work.ptr_end) {
      int qo_head_idx = params.head_indices[work_idx];
      int kv_head_idx = params.group_size_fastdiv.divide(qo_head_idx);
      return {params.qo_tile_indices[work_idx],
              qo_head_idx,
              kv_head_idx,
              params.qo_indptr[work_idx],
              params.kv_indptr[work_idx],
              params.qo_lens[work_idx],
              params.kv_lens[work_idx],
              params.batch_indices[work_idx],
              current_work.counter + 1,
              current_work.ptr_begin,
              current_work.ptr_end};
    } else {
      return {-1,
              -1,
              -1,
              -1,
              -1,
              -1,
              -1,
              current_work.counter + 1,
              current_work.ptr_begin,
              current_work.ptr_end};
    }
  }
};

/*!
 * \brief Tile scheduler that maps q/o head to blockIdx.y
 */
template <typename IdType>
struct BatchPrefillTileScheduler {
 public:
  // Host side kernel arguments
  struct Arguments {
    IdType *work_indptr, *head_indices, *qo_tile_indices, *qo_indptr, *kv_indptr, *qo_lens,
        *kv_lens, *batch_indices;  // head_indices is a placeholder
    cutlass::FastDivmod group_size_fastdiv;
    int num_qo_heads;
  };

  // Device side kernel params
  struct Params {
    IdType *work_indptr, *qo_tile_indices, *qo_indptr, *kv_indptr, *qo_lens, *kv_lens,
        *batch_indices;
    cutlass::FastDivmod group_size_fastdiv;
    int num_qo_heads;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {args.work_indptr, args.qo_tile_indices, args.qo_indptr,     args.kv_indptr,
            args.qo_lens,     args.kv_lens,         args.batch_indices, args.group_size_fastdiv,
            args.num_qo_heads};
  }

  static dim3 get_grid_dim(Arguments const& args, int num_sm) {
    return {(unsigned)num_sm, (unsigned)args.num_qo_heads};
  }

  struct WorkTileInfo {
    int q_tile_idx = 0;
    int qo_head_idx = 0;
    int kv_head_idx = 0;
    int qo_indptr = 0;
    int kv_indptr = 0;
    int qo_len = 0;
    int kv_len = 0;
    int batch_idx = 0;
    int counter = 0;
    int ptr_begin = 0;
    int ptr_end = 0;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const { return counter + ptr_begin < ptr_end; }

    CUTLASS_DEVICE
    auto get_block_coord(Params const& params) const {
      return cute::tuple{q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr,
                         kv_indptr,  qo_len,      kv_len,      batch_idx};
    }
  };

  CUTLASS_DEVICE
  BatchPrefillTileScheduler() {}

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const& params) const {
    int ptr_begin = params.work_indptr[blockIdx.x];
    int ptr_end = params.work_indptr[blockIdx.x + 1];
    if (ptr_begin < ptr_end) {
      int work_idx = ptr_begin;
      int qo_head_idx = blockIdx.y;
      int kv_head_idx = params.group_size_fastdiv.divide(qo_head_idx);
      return {params.qo_tile_indices[work_idx],
              /*qo_head_idx=*/qo_head_idx,
              /*kv_head_idx=*/kv_head_idx,
              params.qo_indptr[work_idx],
              params.kv_indptr[work_idx],
              params.qo_lens[work_idx],
              params.kv_lens[work_idx],
              params.batch_indices[work_idx],
              /*counter=*/0,
              ptr_begin,
              ptr_end};
    } else {
      return {-1, -1, -1, -1, -1, -1, -1, 0, ptr_begin, ptr_end};
    }
  }

  CUTLASS_DEVICE
  void init_consumer() const {}

  CUTLASS_DEVICE
  void prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

  CUTLASS_DEVICE
  void broadcast_next_work(WorkTileInfo& current_work) const {}

  template <bool is_producer = false>
  CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params,
                                            WorkTileInfo const& current_work) const {
    int work_idx = current_work.ptr_begin + current_work.counter + 1;
    if (work_idx < current_work.ptr_end) {
      return {params.qo_tile_indices[work_idx],
              current_work.qo_head_idx,
              current_work.kv_head_idx,
              params.qo_indptr[work_idx],
              params.kv_indptr[work_idx],
              params.qo_lens[work_idx],
              params.kv_lens[work_idx],
              params.batch_indices[work_idx],
              current_work.counter + 1,
              current_work.ptr_begin,
              current_work.ptr_end};
    } else {
      return {-1,
              -1,
              -1,
              -1,
              -1,
              -1,
              -1,
              current_work.counter + 1,
              current_work.ptr_begin,
              current_work.ptr_end};
    }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_TILE_SCHEDULER_CUH_
