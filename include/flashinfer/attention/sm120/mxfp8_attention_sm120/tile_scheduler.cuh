/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_SM120_MXFP8_TILE_SCHEDULER_CUH_
#define FLASHINFER_ATTENTION_SM120_MXFP8_TILE_SCHEDULER_CUH_

#include "cutlass/arch/barrier.h"
#include "cutlass/fast_math.h"

namespace flashinfer {
namespace mxfp8_attention_sm120 {

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

// Persistent grid-stride scheduler for a SINGLE dense problem (no batch indptr).
// grid = num_sm; each CTA processes tiles {blockIdx.x, +gridDim.x, +2*gridDim.x, ...}.
// get_next_work is deterministic (no atomics, no smem) so the warp-specialized
// producer and consumer groups compute the IDENTICAL tile sequence independently --
// drop-in for the current kernel with no coordination. For causal, the grid stride
// pairs low and high m_blocks on the same CTA, improving load balance when
// num_tiles > num_sm. Arguments layout matches SingleTileScheduler so the host can
// swap schedulers without touching the Arguments initializer.
struct StaticPersistentScheduler {
  struct Arguments {
    int const num_qo_tiles, num_qo_heads, qo_len, kv_len;
    cutlass::FastDivmod group_size_fastdiv;
    int const* tile_count_semaphore =
        nullptr;  // unused here; keeps a common ctor with the dynamic one
  };
  struct Params {
    int total_tiles, qo_len, kv_len;
    cutlass::FastDivmod num_qo_tiles_fastdiv;
    cutlass::FastDivmod group_size_fastdiv;
  };

  static Params to_underlying_arguments(Arguments const& a) {
    return {a.num_qo_tiles * a.num_qo_heads, a.qo_len, a.kv_len,
            cutlass::FastDivmod(a.num_qo_tiles), a.group_size_fastdiv};
  }

  static dim3 get_grid_dim(Arguments const& a, int num_sm) { return {uint32_t(num_sm)}; }

  struct WorkTileInfo {
    int tile_idx = 0;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const { return tile_idx < params.total_tiles; }

    CUTLASS_DEVICE
    auto get_block_coord(Params const& params) const {
      int q_tile_idx;
      int qo_head_idx = params.num_qo_tiles_fastdiv.divmod(
          q_tile_idx, tile_idx);  // tile = head*num_tiles + q_tile
      int kv_head_idx = params.group_size_fastdiv.divide(qo_head_idx);
      return cute::tuple{q_tile_idx,      qo_head_idx,   kv_head_idx,   /*qo_indptr=*/0,
                         /*kv_indptr=*/0, params.qo_len, params.kv_len, /*batch_idx=*/0};
    }
  };

  CUTLASS_DEVICE StaticPersistentScheduler() {}

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const& params) const { return {int(blockIdx.x)}; }

  CUTLASS_DEVICE void init_consumer() const {}
  CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}
  CUTLASS_DEVICE void broadcast_next_work(WorkTileInfo&) const {}

  template <bool is_producer = false>
  CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& w) const {
    return {w.tile_idx + int(gridDim.x)};
  }
};

// Persistent grid-stride over a HOST-SORTED tile order (heaviest-first = LPT). Same
// zero-coordination property as StaticPersistentScheduler (deterministic get_next, and
// both the WS producer & consumer read tile_order[] from gmem so they agree on the
// tile sequence with no smem handshake). The host sorts q-tiles by cost (e.g. kv_len)
// DESCENDING so heavy tiles are spread evenly across the persistent CTAs -- the one
// thing the HW block scheduler (fixed blockIdx dispatch order) cannot do. Measured to
// recover 10-20% on variable-length load in bench/sched_varlen.cu.
struct LPTPersistentScheduler {
  struct Arguments {
    int const num_qo_tiles, num_qo_heads, qo_len, kv_len;
    cutlass::FastDivmod group_size_fastdiv;
    int const*
        tile_order;  // [num_qo_tiles*num_qo_heads] host-sorted physical tile for each logical slot
  };
  struct Params {
    int total_tiles, qo_len, kv_len;
    cutlass::FastDivmod num_qo_tiles_fastdiv;
    cutlass::FastDivmod group_size_fastdiv;
    int const* tile_order;
  };

  static Params to_underlying_arguments(Arguments const& a) {
    return {a.num_qo_tiles * a.num_qo_heads,
            a.qo_len,
            a.kv_len,
            cutlass::FastDivmod(a.num_qo_tiles),
            a.group_size_fastdiv,
            a.tile_order};
  }

  static dim3 get_grid_dim(Arguments const& a, int num_sm) { return {uint32_t(num_sm)}; }

  struct WorkTileInfo {
    int slot = 0;

    CUTLASS_DEVICE
    bool is_valid(Params const& params) const { return slot < params.total_tiles; }

    CUTLASS_DEVICE
    auto get_block_coord(Params const& params) const {
      int phys = params.tile_order[slot];  // host-sorted (heaviest-first) -> physical tile index
      int q_tile_idx;
      int qo_head_idx = params.num_qo_tiles_fastdiv.divmod(q_tile_idx, phys);
      int kv_head_idx = params.group_size_fastdiv.divide(qo_head_idx);
      return cute::tuple{q_tile_idx,      qo_head_idx,   kv_head_idx,   /*qo_indptr=*/0,
                         /*kv_indptr=*/0, params.qo_len, params.kv_len, /*batch_idx=*/0};
    }
  };

  CUTLASS_DEVICE LPTPersistentScheduler() {}

  CUTLASS_DEVICE
  WorkTileInfo get_initial_work(Params const& params) const { return {int(blockIdx.x)}; }

  CUTLASS_DEVICE void init_consumer() const {}
  CUTLASS_DEVICE void prefetch_next_work(Params const&, WorkTileInfo&) const {}
  CUTLASS_DEVICE void broadcast_next_work(WorkTileInfo&) const {}

  template <bool is_producer = false>
  CUTLASS_DEVICE WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& w) const {
    return {w.slot + int(gridDim.x)};
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
      return {-1, -1, -1, -1, -1, -1, -1, /*batch_idx=*/-1, /*counter=*/0, ptr_begin, ptr_end};
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
              /*batch_idx=*/-1,
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
      return {-1, -1, -1, -1, -1, -1, -1, /*batch_idx=*/-1, /*counter=*/0, ptr_begin, ptr_end};
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
              /*batch_idx=*/-1,
              current_work.counter + 1,
              current_work.ptr_begin,
              current_work.ptr_end};
    }
  }
};

}  // namespace mxfp8_attention_sm120
}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_SM120_MXFP8_TILE_SCHEDULER_CUH_
