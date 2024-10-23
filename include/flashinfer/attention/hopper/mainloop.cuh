

/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 *Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "named_barrier.cuh"
#include "../../math.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

template <typename Ktraits, bool CAUSAL>
struct CollectiveMainloop {
  using Element = typename Ktraits::Element;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int HEAD_DIM = Ktraits::HEAD_DIM;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV =
      decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(
          shape<0>(ClusterShape{})));

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

  using ShapeT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideT = cute::Shape<int64_t, _1, int64_t>;  // (N, D, H)
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeLseT = cute::Shape<int32_t, int32_t>;
  using StrideLseT = cute::Shape<int64_t, _1>;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{}));  // no mcast for Q

  using TMA_K = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      take<0, 2>(SmemLayoutK{}), select<1, 2>(TileShape_MNK{}),
      size<0>(ClusterShape{})));  // mcast along M mode for this N load, if any

  // TMA_V may differ from TMA_K for fp8 kernel (e.g. swizzling mode)
  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                  repeat_like(StrideT{}, int32_t(0)), StrideT{}),
      take<0, 2>(SmemLayoutV{}), select<1, 2>(TileShape_MNK{}),
      size<0>(ClusterShape{})));  // mcast along M mode for this N load, if any

  static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using MainloopPipelineNoTMA = typename Ktraits::MainloopPipelineNoTMA;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  // Set the bytes transferred in this TMA transaction (may involve multiple issues)
  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<Element> / 8);
  static constexpr uint32_t TmaTransactionBytesK =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<Element> / 8);

  // static constexpr bool UseSchedulerBarrier = HEAD_DIM <= 128;
  static constexpr bool UseSchedulerBarrier =
      cutlass::sizeof_bits_v<Element> == 8 ? HEAD_DIM >= 128 : HEAD_DIM <= 128;

  // Host side kernel arguments
  struct Arguments {
    Element const* ptr_Q;
    LayoutT layout_Q;
    Element const* ptr_K;
    LayoutT layout_K;
    Element const* ptr_V;
    LayoutT layout_V;
    float const sm_scale_log2;
  };

  // Device side kernel params
  struct Params {
    LayoutT layout_Q;
    LayoutT layout_K;
    LayoutT layout_V;
    cutlass::FastDivmod group_size_fastdiv;
    TMA_Q tma_load_Q;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    float const sm_scale_log2;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.layout_Q);
    TMA_Q tma_load_Q = make_tma_copy(GmemTiledCopyQ{}, mQ, SmemLayoutQ{},
                                     select<0, 2>(TileShape_MNK{}), _1{});  // no mcast for Q
    Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.layout_K);
    TMA_K tma_load_K = make_tma_copy(
        GmemTiledCopyKV{}, mK, SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{}));  // mcast along M mode for this N load, if any
    Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.layout_V);
    TMA_V tma_load_V = make_tma_copy(
        GmemTiledCopyKV{}, mV, SmemLayoutV{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}),
        size<0>(ClusterShape{}));  // mcast along M mode for this N load, if any
    return {args.layout_Q,
            args.layout_K,
            args.layout_V,
            cutlass::FastDivmod(
                cute::ceil_div(get<2>(args.layout_Q.shape()), get<2>(args.layout_K.shape()))),
            tma_load_Q,
            tma_load_K,
            tma_load_V,
            args.sm_scale_log2};
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
    static constexpr int CTA_Q = get<0>(TileShape_MNK{});
    static constexpr int CTA_KV = get<1>(TileShape_MNK{});
    int num_kv_tiles = cute::ceil_div(kv_len, CTA_KV);
    if constexpr (CAUSAL) {
      num_kv_tiles =
          std::min(num_kv_tiles, cute::ceil_div((q_tile_idx + 1) * CTA_Q + kv_len - qo_len, CTA_KV));
    }
    return num_kv_tiles;
  }

  template <typename Scheduler, typename SharedStorage>
  CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline_k,
                           MainloopPipeline pipeline_v, PipelineState& smem_pipe_write_k,
                           PipelineState& smem_pipe_write_v, SharedStorage& shared_storage,
                           Scheduler& scheduler, typename Scheduler::Params const& scheduler_params,
                           typename Scheduler::WorkTileInfo& work_tile_info,
                           cute::tuple<int32_t, int32_t> block_coord, int work_idx,
                           const int32_t qo_len, const int32_t kv_len) {
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());

    auto [q_tile_idx, head_idx] = block_coord;
    int kv_head_idx = mainloop_params.group_size_fastdiv.divide(head_idx);

    // Prepare the TMA loads
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
    constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                    block_rank_in_cluster / cluster_shape_x};
    Tensor gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShape_MNK{}), head_idx, /*offset=*/0,
                                      qo_len)(_, _, q_tile_idx);  // (M, K)
    Tensor gK = get_local_tile_tensor(mK, select<1, 2>(TileShape_MNK{}), kv_head_idx, /*offset=*/0,
                                      kv_len);  // (N, K, _)
    Tensor gV = get_local_tile_tensor(mV, select<1, 2>(TileShape_MNK{}), kv_head_idx, /*offset=*/0,
                                      kv_len);  // (N, K, _)

    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] =
        tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 2>(sQ_x),
                      group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
    auto [tKgK, tKsK] =
        tma_partition(mainloop_params.tma_load_K, block_rank_in_cluster, Layout<ClusterShape>{},
                      group_modes<0, 2>(sK), group_modes<0, 2>(gK));  // (TMA, k), (TMA, PIPE)
    auto [tVgV, tVsV] =
        tma_partition(mainloop_params.tma_load_V, block_rank_in_cluster, Layout<ClusterShape>{},
                      group_modes<0, 2>(sV), group_modes<0, 2>(gV));  // (TMA, k), (TMA, PIPE)

    int kv_tile_max = get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
    int kv_tile_idx = kv_tile_max - 1;

    int lane_predicate = cute::elect_one_sync();
    if (lane_predicate) {
      pipeline_k.producer_acquire(smem_pipe_write_k);
      copy(mainloop_params.tma_load_K.with(*pipeline_k.producer_get_barrier(smem_pipe_write_k),
                                           /*mcast_mask=*/0),
           tKgK(_, kv_tile_idx), tKsK(_, smem_pipe_write_k.index()));
      ++smem_pipe_write_k;
    }

    // Wait for the MMA warpgroups to say that smem_q is ready
    cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp,
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
// CUTLASS_PRAGMA_NO_UNROLL
#pragma unroll 2
      for (; kv_tile_idx > 0; --kv_tile_idx) {
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

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                                PipelineState& smem_pipe_write_k,
                                PipelineState& smem_pipe_write_v) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // Issue the epilogue waits
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was
       * never used then would just be acquired since the phase was still inverted from
       * make_producer_start_state
       */
      pipeline_k.producer_tail(smem_pipe_write_k);
      pipeline_v.producer_tail(smem_pipe_write_v);
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail_one_write(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                                          PipelineState& smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    // Issue the epilogue waits
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was
       * never used then would just be acquired since the phase was still inverted from
       * make_producer_start_state
       */
      pipeline_k.producer_tail(smem_pipe_write);
      pipeline_v.producer_tail(smem_pipe_write);
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_sync() {
    if constexpr (UseSchedulerBarrier) {
      cutlass::arch::NamedBarrier::sync(NumMmaThreads,
                                        static_cast<int>(NamedBarriers::kWarpSchedulerWG1) - 1 +
                                            cutlass::canonical_warp_group_idx() /*id*/);
    }
  }

  CUTLASS_DEVICE void warp_scheduler_barrier_arrive() {
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup ||
                  NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
    if constexpr (NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup) {
      cutlass::arch::NamedBarrier::arrive(NumMmaThreads,
                                          static_cast<int>(NamedBarriers::kWarpSchedulerWG1) - 1 +
                                              (3 - cutlass::canonical_warp_group_idx()) /*id*/);
    } else {
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads, static_cast<int>(NamedBarriers::kWarpSchedulerWG1) - 1 +
                             (cutlass::canonical_warp_group_idx() <= 2
                                  ? cutlass::canonical_warp_group_idx() + 1
                                  : cutlass::canonical_warp_group_idx() + 1 - 3) /*id*/);
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads, static_cast<int>(NamedBarriers::kWarpSchedulerWG1) - 1 +
                             (cutlass::canonical_warp_group_idx() <= 1
                                  ? cutlass::canonical_warp_group_idx() + 2
                                  : cutlass::canonical_warp_group_idx() + 2 - 3) /*id*/);
    }
  }

  CUTLASS_DEVICE void mma_init() {
    // Tell producer (warp 0) that smem_q is ready
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads + Ktraits::NumProducerThreads,
                                        static_cast<int>(NamedBarriers::kQueryEmpty) /*id*/);
    if constexpr (!UseSchedulerBarrier) {
      return;
    }
    static_assert(NumMmaThreads == 2 * cutlass::NumThreadsPerWarpGroup ||
                  NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup);
    if (cutlass::canonical_warp_group_idx() > 1) {
      cutlass::arch::NamedBarrier::arrive(
          NumMmaThreads, static_cast<int>(NamedBarriers::kWarpSchedulerWG1) - 1 + 1 /*id*/);
    }
    if constexpr (NumMmaThreads == 3 * cutlass::NumThreadsPerWarpGroup) {
      if (cutlass::canonical_warp_group_idx() > 2) {
        cutlass::arch::NamedBarrier::arrive(
            NumMmaThreads, static_cast<int>(NamedBarriers::kWarpSchedulerWG1) - 1 + 2 /*id*/);
      }
    }
  }

  template <typename SharedStorage, typename FrgTensorO, typename Softmax>
  CUTLASS_DEVICE void mma(Params const& mainloop_params, MainloopPipeline pipeline_k,
                          MainloopPipeline pipeline_v, PipelineState& smem_pipe_read_k,
                          PipelineState& smem_pipe_read_v, FrgTensorO& tOrO, Softmax& softmax,
                          int kv_tile_idx_count, int thread_idx, int work_idx, int q_tile_idx,
                          SharedStorage& shared_storage, const int32_t qo_len,
                          const int32_t kv_len) {
    static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

    static constexpr int CTA_Q = get<0>(TileShape_MNK{});
    static constexpr int CTA_KV = get<1>(TileShape_MNK{});

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVt{});

    typename Ktraits::TiledMma0 tiled_mma0;
    typename Ktraits::TiledMma1 tiled_mma1;
    auto threadMma0 = tiled_mma0.get_thread_slice(thread_idx);
    auto threadMma1 = tiled_mma1.get_thread_slice(thread_idx);

    // Allocate "fragments/descriptors" for first matmul.
    Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
    Tensor tSrK = threadMma0.partition_fragment_B(sK);
    // Allocate "fragments/descriptors" for second matmul.
    // Note: S becomes P.
    Tensor tOrV = threadMma1.partition_fragment_B(sVt);

    auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
    };

    tiled_mma1.accumulate_ = GMMA::ScaleOut::Zero;
    int kv_tile_idx = kv_tile_idx_count - 1;

    cutlass::ConsumerToken barrier_token =
        static_cast<cutlass::BarrierStatus>(shared_storage.barrier_Q.try_wait(work_idx % 2));
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      shared_storage.barrier_Q.wait(work_idx % 2);
    }

    Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
    consumer_wait(pipeline_k, smem_pipe_read_k);
    warp_scheduler_barrier_sync();
    gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ,
                                             tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
    warp_scheduler_barrier_arrive();

    if (work_idx != 0) {
      int lane_predicate = cute::elect_one_sync();
      if (cutlass::canonical_warp_idx_sync() == Ktraits::kNWarps - 1 && lane_predicate) {
        tma_store_wait<0>();
#pragma unroll
        for (uint32_t cta_id = 0; cta_id < size(ClusterShape{}); ++cta_id) {
          shared_storage.barrier_O.arrive(cta_id, lane_predicate);
        }
      }
    }
    warpgroup_wait<0>();
    pipeline_k.consumer_release(smem_pipe_read_k);
    ++smem_pipe_read_k;

    auto col_limit_causal = [&](int row, int kv_tile_idx) {
      return row + 1 + kv_len - kv_tile_idx * CTA_KV - qo_len + q_tile_idx * CTA_Q;
    };
    {
      Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
      Tensor tScS = threadMma0.partition_C(cS);
#pragma unroll
      for (int i = 0; i < size(tSrS); ++i) {
        if constexpr (!CAUSAL) {  // Just masking based on col
          if (int(get<1>(tScS(i))) >= int(kv_len - kv_tile_idx * CTA_KV)) {
            tSrS(i) = -math::inf;
          }
        } else {  // mask based on both row and col
          // using std::min is faster than doing col >= limit0 or col >= limit1
          // Need to cast get<1>(tScS(i)) to (signed) int since by default it's unsigned, and the
          // right hand side can be negative and might be converted to a very large unsigned
          // integer.
          if (int(get<1>(tScS(i))) >=
              std::min(kv_len - kv_tile_idx * CTA_KV,
                       col_limit_causal(int(get<0>(tScS(i))), kv_tile_idx))) {
            tSrS(i) = -math::inf;
          }
        }
      }
    }

    softmax.template online_softmax</*is_first=*/true>(tSrS);
    Tensor tOrP = make_tensor(convert_type<Element>(tSrS).data(),
                              convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout()));
    Tensor scores_scale = make_fragment_like(softmax.row_max);
    clear(scores_scale);

    constexpr int n_masking_steps = !CAUSAL ? 1 : cute::ceil_div(CTA_Q, CTA_KV) + 1;
// Only go through these if CAUSAL, since n_masking_steps = 1 when !CAUSAL
#pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps - 1 && kv_tile_idx > 0;
         ++masking_step, --kv_tile_idx) {
      Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
      consumer_wait(pipeline_k, smem_pipe_read_k);
      warp_scheduler_barrier_sync();
      gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ,
                                               tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
      if (masking_step > 0) {
        softmax.rescale_o(tOrO, scores_scale);
      }
      consumer_wait(pipeline_v, smem_pipe_read_v);
      gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP,
                                                tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
      warp_scheduler_barrier_arrive();
      warpgroup_wait<1>();
      pipeline_k.consumer_release(smem_pipe_read_k);  // release K
      Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_MNK{}));
      Tensor tScS = threadMma0.partition_C(cS);
#pragma unroll
      for (int i = 0; i < size(tSrS); ++i) {
        if (int(get<1>(tScS(i))) >= col_limit_causal(int(get<0>(tScS(i))), kv_tile_idx - 1)) {
          tSrS(i) = -math::inf;
        }
      }
      cute::copy(softmax.template max</*is_first=*/false>(tSrS), scores_scale);
      softmax.template online_softmax</*is_first=*/false>(tSrS);
      warpgroup_wait<0>();
      pipeline_v.consumer_release(smem_pipe_read_v);  // release V
      ++smem_pipe_read_k;
      ++smem_pipe_read_v;
      cute::copy(make_tensor(convert_type<Element>(tSrS).data(),
                             convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout())),
                 tOrP);
    }

#pragma unroll 1
    for (; kv_tile_idx > 0; --kv_tile_idx) {
      Tensor tSrS = partition_fragment_C(tiled_mma0, select<0, 1>(TileShape_MNK{}));
      consumer_wait(pipeline_k, smem_pipe_read_k);
      warp_scheduler_barrier_sync();
      gemm</*zero_init=*/true, /*wg_wait=*/-1>(tiled_mma0, tSrQ,
                                               tSrK(_, _, _, smem_pipe_read_k.index()), tSrS);
      softmax.rescale_o(tOrO, scores_scale);
      consumer_wait(pipeline_v, smem_pipe_read_v);
      gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP,
                                                tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
      warp_scheduler_barrier_arrive();
      warpgroup_wait<1>();
      pipeline_k.consumer_release(smem_pipe_read_k);  // release K
      // auto scores_scale = softmax.template max</*is_first=*/false>(tSrS);
      cute::copy(softmax.template max</*is_first=*/false>(tSrS), scores_scale);
      softmax.template online_softmax</*is_first=*/false>(tSrS);
      warpgroup_wait<0>();
      pipeline_v.consumer_release(smem_pipe_read_v);  // release V
      ++smem_pipe_read_k;
      ++smem_pipe_read_v;
      // softmax.rescale_o(tOrO, scores_scale);
      cute::copy(make_tensor(convert_type<Element>(tSrS).data(),
                             convert_layout_acc_Aregs<typename Ktraits::TiledMma1>(tSrS.layout())),
                 tOrP);
    }
    // Tell warp 0 that smem_q is ready
    cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp,
                                        static_cast<int>(NamedBarriers::kQueryEmpty) /*id*/);
    softmax.rescale_o(tOrO, scores_scale);
    consumer_wait(pipeline_v, smem_pipe_read_v);
    gemm</*zero_init=*/false, /*wg_wait=*/-1>(tiled_mma1, tOrP,
                                              tOrV(_, _, _, smem_pipe_read_v.index()), tOrO);
    cute::copy(softmax.template finalize(tSrS), scores_scale);
    warpgroup_wait<0>();
    pipeline_v.consumer_release(smem_pipe_read_v);  // release V, otherwise producers will hang
    ++smem_pipe_read_v;

    softmax.rescale_o(tOrO, scores_scale);
    return;
  }
};

}  // namespace flashinfer
