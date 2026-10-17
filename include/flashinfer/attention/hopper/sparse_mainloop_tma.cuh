/*
 * Copyright (c) 2026 by FlashInfer team.
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
#ifndef FLASHINFER_ATTENTION_HOPPER_SPARSE_MAINLOOP_TMA_CUH_
#define FLASHINFER_ATTENTION_HOPPER_SPARSE_MAINLOOP_TMA_CUH_

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <algorithm>
#include <numeric>
#include <type_traits>

#include "../../fastdiv.cuh"
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "named_barrier.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

// Smallest TMA box SparseTmaCollectiveMainloop issues; page_size must be a multiple of it.
constexpr int PAGED_KV_TMA_MIN_BOX_ROWS = 16;

// Paged-KV mainloop that gathers pages with TMA instead of per-thread cp.async.
//
// The KV pool is described to TMA as (head_dim, page_size, num_heads, num_pages), so a box of
// box_rows tokens is addressed by (row in page, page) with no device-side pointer math. box_rows is
// the largest divisor of page_size and CTA_KV up to MAX_BOX_ROWS, fixed per launch; a KV tile is
// CTA_KV / box_rows boxes, box j issued by lane j of the producer warp, all completing on the
// stage's transaction barrier. Boxes past kv_len target page num_pages, which TMA zero-fills, so
// page-table reads never leave the request's range.
template <typename AdditionalParams, typename Ktraits, bool CAUSAL, bool MULTIITEMSCORING = false>
struct SparseTmaCollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO);

  static constexpr int MIN_BOX_ROWS = PAGED_KV_TMA_MIN_BOX_ROWS;
  static constexpr int MAX_BOX_ROWS = 64;  // H100: 64-row boxes match the dense kernel's rate
  static_assert(CTA_KV % MIN_BOX_ROWS == 0 && CTA_KV / MIN_BOX_ROWS <= cutlass::NumThreadsPerWarp);
  // The last tile's V rows in [kv_len, round_up(kv_len, box_rows)) arrive unmasked; the consumer
  // zeroes them before P*V.
  static constexpr bool ZERO_V_TAIL = true;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
  using SmemLayoutAtomKV = typename Ktraits::SmemLayoutAtomK;
  static_assert(std::is_same_v<SmemLayoutAtomKV, typename Ktraits::SmemLayoutAtomV> &&
                std::is_same_v<SmemLayoutK, SmemLayoutV>);
  // A box spans one swizzle atom row (128 B) along head_dim; a tile row is NUM_COL_BOXES of them.
  static constexpr int BOX_COLS = size<1>(SmemLayoutAtomKV{});
  static constexpr int NUM_COL_BOXES = HEAD_DIM_QK / BOX_COLS;
  static_assert(HEAD_DIM_QK % BOX_COLS == 0);

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int64_t, _1, int64_t>;  // (N, D, H)
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeKVT = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideKVT = cute::Shape<int64_t, _1, int64_t, int64_t>;  // (page_size, D, H, num_pages)
  using LayoutKVT = cute::Layout<ShapeKVT, StrideKVT>;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<DTypeQ const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{}));  // no mcast for Q

  static constexpr bool USE_TMA_LOAD_KV = true;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<DTypeQ> / 8);
  // All boxes of a tile complete on one barrier.
  static constexpr uint32_t TmaTransactionBytesK =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);
  static constexpr uint32_t TmaTransactionBytesV =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutV{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);

  static constexpr bool UseSchedulerBarrier =
      cutlass::sizeof_bits_v<DTypeQ> == 8 ? HEAD_DIM_VO >= 128 : HEAD_DIM_VO <= 128;
  using WarpScheduler = WarpScheduler<Ktraits, UseSchedulerBarrier>;

  // Host side kernel arguments
  struct Arguments {
    DTypeQ const* Q_ptr;
    LayoutT layout_Q;
    DTypeKV const* K_ptr;
    LayoutKVT layout_K;
    DTypeKV const* V_ptr;
    LayoutKVT layout_V;
    IdType const* kv_indices;
    int window_left;
    AdditionalParams additional_params;
  };

  // Device side kernel params
  struct Params {
    LayoutT layout_Q;
    TMA_Q tma_load_Q;
    TmaDescriptor tma_desc_K;
    TmaDescriptor tma_desc_V;
    IdType* kv_indices;
    uint_fastdiv page_size;
    int num_pages;
    int box_rows;
    int window_left;
    AdditionalParams additional_params;
  };

  // (BOX_COLS x box_rows) boxes over the pool; TMA dim 0 is head_dim, the contiguous mode.
  static TmaDescriptor make_tma_desc(DTypeKV const* ptr, LayoutKVT const& layout, int box_rows) {
    Tensor gtensor = make_tensor(make_gmem_ptr(ptr), layout);
    auto tma_gbasis = make_layout(make_shape(Int<BOX_COLS>{}, box_rows, _1{}, _1{}),
                                  make_stride(E<1>{}, E<0>{}, E<2>{}, E<3>{}));
    return get<0>(cute::detail::make_tma_copy_desc<DTypeKV>(
        gtensor, tma_gbasis, get_swizzle_portion(SmemLayoutAtomKV{}), /*num_multicast=*/1));
  }

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.Q_ptr), args.layout_Q);
    TMA_Q tma_load_Q =
        make_tma_copy(GmemTiledCopyQ{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{});
    const int page_size = get<0>(args.layout_K.shape());
    const int box_rows = std::min(std::gcd(page_size, CTA_KV), MAX_BOX_ROWS);
    return {args.layout_Q,
            tma_load_Q,
            make_tma_desc(args.K_ptr, args.layout_K, box_rows),
            make_tma_desc(args.V_ptr, args.layout_V, box_rows),
            const_cast<IdType*>(args.kv_indices),
            uint_fastdiv(static_cast<uint32_t>(page_size)),
            get<3>(args.layout_K.shape()),
            box_rows,
            args.window_left,
            args.additional_params};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(&mainloop_params.tma_desc_K);
    cute::prefetch_tma_descriptor(&mainloop_params.tma_desc_V);
  }

  CUTLASS_DEVICE
  int get_num_kv_tiles(Params const& mainloop_params, int q_tile_idx, const int qo_len,
                       const int kv_len) {
    static constexpr int CTA_Q = get<0>(TileShape_QKD{});
    static constexpr int CTA_KV = get<1>(TileShape_QKD{});
    int num_kv_tiles = cute::ceil_div(kv_len, CTA_KV);
    if constexpr (CAUSAL || MULTIITEMSCORING) {
      num_kv_tiles = std::min(num_kv_tiles,
                              cute::ceil_div((q_tile_idx + 1) * CTA_Q + kv_len - qo_len, CTA_KV));
    }
    return num_kv_tiles;
  }

  template <bool LEFT_SLIDING_WINDOW, typename BlockCoord, typename Scheduler,
            typename SharedStorage>
  CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline_k,
                           MainloopPipeline pipeline_v, PipelineState& smem_pipe_write_k,
                           PipelineState& smem_pipe_write_v, SharedStorage& shared_storage,
                           Scheduler& scheduler, typename Scheduler::Params const& scheduler_params,
                           typename Scheduler::WorkTileInfo& work_tile_info,
                           BlockCoord const& block_coord, int work_idx,
                           const int num_kv_tiles_outside_items_window = 0,
                           const int num_kv_tiles_prefix = 0) {
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());

    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;

    Tensor gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShape_QKD{}), qo_head_idx, qo_indptr,
                                      qo_len)(_, _, q_tile_idx);  // (Q, D)
    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] =
        tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 2>(sQ_x),
                      group_modes<0, 2>(gQ_x));  // (TMA), (TMA)

    int num_kv_tiles = get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
    int kv_tile_idx = num_kv_tiles - 1;
    int swa_begin_kv_tile_idx = 0;
    if constexpr (LEFT_SLIDING_WINDOW) {
      swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                       q_tile_idx, qo_len, kv_len);
    }

    const int lane_predicate = cute::elect_one_sync();
    const int lane_idx = threadIdx.x % cutlass::NumThreadsPerWarp;
    const int box_rows = mainloop_params.box_rows;
    const bool owns_box = lane_idx * box_rows < CTA_KV;
    IdType const* kv_indices = mainloop_params.kv_indices + kv_indptr;

    // (row in page, page) of this lane's box in a tile; past kv_len, a page TMA zero-fills.
    auto locate = [&](int kv_tile_idx) {
      int kv_idx = kv_tile_idx * CTA_KV + lane_idx * box_rows;
      int2 coord = make_int2(0, mainloop_params.num_pages);
      if (owns_box && kv_idx < kv_len) {
        uint32_t page_iter, entry_idx;
        mainloop_params.page_size.divmod(kv_idx, page_iter, entry_idx);
        coord = make_int2(entry_idx, static_cast<int>(__ldg(kv_indices + page_iter)));
      }
      return coord;
    };
    auto kv_tile_idx_decrement = [&](int kv_tile_idx) {
      int result = kv_tile_idx - 1;
      if constexpr (MULTIITEMSCORING) {
        if ((kv_tile_idx == num_kv_tiles_outside_items_window) &
            (kv_tile_idx >= num_kv_tiles_prefix)) {
          result = num_kv_tiles_prefix - 1;
        }
      }
      return result;
    };
    // Lane j lands box j at rows [j * box_rows, (j + 1) * box_rows) of the stage, one TMA per
    // 128 B column chunk; the swizzle is the hardware's.
    auto issue = [&](TmaDescriptor const& desc, auto* barrier, DTypeKV* smem, int stage,
                     int2 coord) {
      if (!owns_box) return;
      auto tile = get_nonswizzle_portion(SmemLayoutK{});  // == SmemLayoutV
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < NUM_COL_BOXES; ++c) {
        SM90_TMA_LOAD_4D::copy(&desc, reinterpret_cast<uint64_t*>(barrier),
                               static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_NORMAL),
                               smem + tile(lane_idx * box_rows, c * BOX_COLS, stage), c * BOX_COLS,
                               coord.x, kv_head_idx, coord.y);
      }
    };
    auto load_k = [&](int2 coord) {
      pipeline_k.producer_acquire(smem_pipe_write_k);
      issue(mainloop_params.tma_desc_K, pipeline_k.producer_get_barrier(smem_pipe_write_k),
            shared_storage.smem_k.data(), smem_pipe_write_k.index(), coord);
      ++smem_pipe_write_k;
    };
    auto load_v = [&](int2 coord) {
      pipeline_v.producer_acquire(smem_pipe_write_v);
      issue(mainloop_params.tma_desc_V, pipeline_v.producer_get_barrier(smem_pipe_write_v),
            shared_storage.smem_v.data(), smem_pipe_write_v.index(), coord);
      ++smem_pipe_write_v;
    };

    int2 coord_v = locate(kv_tile_idx);
    load_k(coord_v);

    // Wait for the MMA warpgroups to say that smem_q is ready
    cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                      static_cast<int>(NamedBarriers::kQueryEmpty));

    if (lane_predicate) {
      shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
      copy(mainloop_params.tma_load_Q.with(
               reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                   shared_storage.barrier_Q),
               /*mcast_mask=*/0),
           tQgQ, tQsQ);
    }

    // See CollectiveMainloop::load for why this is a cluster barrier.
    shared_storage.barrier_O.wait((work_idx + 1) % 2);

#pragma unroll 2
    for (; kv_tile_idx > swa_begin_kv_tile_idx; kv_tile_idx = kv_tile_idx_decrement(kv_tile_idx)) {
      int2 coord_k = locate(kv_tile_idx_decrement(kv_tile_idx));
      load_k(coord_k);
      load_v(coord_v);
      coord_v = coord_k;
    }
    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
    load_v(coord_v);
    scheduler.broadcast_next_work(work_tile_info);
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                                PipelineState& smem_pipe_write_k,
                                PipelineState& smem_pipe_write_v) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      pipeline_k.producer_tail(smem_pipe_write_k);
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_SPARSE_MAINLOOP_TMA_CUH_
