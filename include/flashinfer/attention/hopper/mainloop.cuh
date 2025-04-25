/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_MAINLOOP_CUH_
#define FLASHINFER_ATTENTION_HOPPER_MAINLOOP_CUH_

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "../../math.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "mainloop_mma.cuh"
#include "named_barrier.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

template <typename AdditionalParams, typename Ktraits, bool CAUSAL>
struct CollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int64_t, _1, int64_t>;  // (N, D, H)
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeLseT = cute::Shape<int32_t, int32_t>;
  using StrideLseT = cute::Shape<_1, int64_t>;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<DTypeQ const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{}));  // no mcast for Q

  using TMA_K = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      take<0, 2>(SmemLayoutK{}), select<1, 2>(TileShape_QKD{}), _1{}));  // no mcast

  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      take<0, 2>(SmemLayoutV{}), select<2, 1>(TileShape_PDV{}), _1{}));  // no mcast

  static constexpr bool USE_TMA_LOAD_KV = true;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<DTypeQ> / 8);
  static constexpr uint32_t TmaTransactionBytesK =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);
  static constexpr uint32_t TmaTransactionBytesV =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutV{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);

  // Whether use scheduler barrier or hardware warp scheduler, using heuristic based on data type
  // and head dim
  static constexpr bool UseSchedulerBarrier =
      cutlass::sizeof_bits_v<DTypeQ> == 8 ? HEAD_DIM_VO >= 128 : HEAD_DIM_VO <= 128;
  using WarpScheduler = WarpScheduler<Ktraits, UseSchedulerBarrier>;

  // Host side kernel arguments
  struct Arguments {
    DTypeQ const* Q_ptr;
    LayoutT layout_Q;
    DTypeKV const* K_ptr;
    LayoutT layout_K;
    DTypeKV const* V_ptr;
    LayoutT layout_V;
    int window_left;
    AdditionalParams additional_params;
  };

  // Device side kernel params
  struct Params {
    LayoutT layout_Q;
    LayoutT layout_K;
    LayoutT layout_V;
    TMA_Q tma_load_Q;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    int window_left;
    AdditionalParams additional_params;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.Q_ptr), args.layout_Q);
    TMA_Q tma_load_Q = make_tma_copy(GmemTiledCopyQ{}, mQ, SmemLayoutQ{},
                                     select<0, 2>(TileShape_QKD{}), _1{});  // no mcast for Q
    Tensor mK = make_tensor(make_gmem_ptr(args.K_ptr), args.layout_K);
    TMA_K tma_load_K = make_tma_copy(GmemTiledCopyKV{}, mK, SmemLayoutK{}(_, _, _0{}),
                                     select<1, 2>(TileShape_QKD{}), _1{});  // no mcast
    Tensor mV = make_tensor(make_gmem_ptr(args.V_ptr), args.layout_V);
    TMA_V tma_load_V = make_tma_copy(GmemTiledCopyKV{}, mV, SmemLayoutV{}(_, _, _0{}),
                                     select<2, 1>(TileShape_PDV{}), _1{});  // no mcast
    return {args.layout_Q, args.layout_K, args.layout_V,    tma_load_Q,
            tma_load_K,    tma_load_V,    args.window_left, args.additional_params};
  }

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_K.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_V.get_tma_descriptor());
  }

  CUTLASS_DEVICE
  int get_num_kv_tiles(Params const& mainloop_params, int q_tile_idx, const int qo_len,
                       const int kv_len) {
    static constexpr int CTA_Q = get<0>(TileShape_QKD{});
    static constexpr int CTA_KV = get<1>(TileShape_QKD{});
    int num_kv_tiles = cute::ceil_div(kv_len, CTA_KV);
    if constexpr (CAUSAL) {
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
                           BlockCoord const& block_coord, int work_idx) {
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());

    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;

    // Prepare the TMA loads
    Tensor gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShape_QKD{}), qo_head_idx, qo_indptr,
                                      qo_len)(_, _, q_tile_idx);  // (Q, D)
    Tensor gK = get_local_tile_tensor(mK, select<1, 2>(TileShape_QKD{}), kv_head_idx, kv_indptr,
                                      kv_len);  // (K, D, _)
    Tensor gV = get_local_tile_tensor(mV, select<2, 1>(TileShape_PDV{}), kv_head_idx, kv_indptr,
                                      kv_len);  // (K, D, _)

    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] =
        tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 2>(sQ_x),
                      group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
    auto [tKgK, tKsK] =
        tma_partition(mainloop_params.tma_load_K, _0{}, Layout<_1>{}, group_modes<0, 2>(sK),
                      group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
    auto [tVgV, tVsV] =
        tma_partition(mainloop_params.tma_load_V, _0{}, Layout<_1>{}, group_modes<0, 2>(sV),
                      group_modes<0, 2>(gV));  // (TMA, k), (TMA, PIPE)

    int num_kv_tiles = get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
    int kv_tile_idx = num_kv_tiles - 1;
    int swa_begin_kv_tile_idx = 0;
    if constexpr (LEFT_SLIDING_WINDOW) {
      swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                       q_tile_idx, qo_len, kv_len);
    }

    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline_k.producer_acquire(smem_pipe_write_k);
      copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                           /*mcast_mask=*/0),
           tKgK(_, kv_tile_idx), tKsK(_, smem_pipe_write_k.index()));
      ++smem_pipe_write_k;
    }

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

    // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
    // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
    // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on
    // O.
    shared_storage.barrier_O.wait((work_idx + 1) % 2);

    if (lane_predicate) {
#pragma unroll 2
      for (; kv_tile_idx > swa_begin_kv_tile_idx; --kv_tile_idx) {
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                             /*mcast_mask=*/0),
             tKgK(_, kv_tile_idx - 1), tKsK(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                             /*mcast_mask=*/0),
             tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
      }
    }
    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
    if (lane_predicate) {
      pipeline_v.producer_acquire(smem_pipe_write_v);
      copy(mainloop_params.tma_load_V.with(*pipeline_v.producer_get_barrier(smem_pipe_write_v),
                                           /*mcast_mask=*/0),
           tVgV(_, kv_tile_idx), tVsV(_, smem_pipe_write_v.index()));
      ++smem_pipe_write_v;
    }
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

#endif  // FLASHINFER_ATTENTION_HOPPER_MAINLOOP_CUH_
