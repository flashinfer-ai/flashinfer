/*
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_ATTENTION_HOPPER_SPARSE_MAINLOOP_CUH_
#define FLASHINFER_ATTENTION_HOPPER_SPARSE_MAINLOOP_CUH_

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "../../math.cuh"
#include "block_sparse_gather.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "named_barrier.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

template <typename AdditionalParams, typename Ktraits, bool CAUSAL>
struct SparseCollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int HEAD_DIM = Ktraits::HEAD_DIM;
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  static constexpr auto AlignmentKV = 128 / cutlass::sizeof_bits<DTypeKV>::value;
  using AlignmentTypeKV = cute::uint_byte_t<static_cast<int>(sizeof(DTypeKV)) * AlignmentKV>;
  // NOTE(Zihao): use SM80_CP_ASYNC for sparse loading of KV-cache
  using GmemCopyAtomKV = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<AlignmentTypeKV>, DTypeKV>;
  using GmemTiledCopyKV =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               GmemCopyAtomKV, NUM_COPY_THREADS, AlignmentKV,
               cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>,
               decltype(cute::get<1>(TileShape_QKD{})), decltype(cute::get<2>(TileShape_QKD{}))>());

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

  static constexpr bool USE_TMA_LOAD_KV = false;
  static constexpr int NUM_MMA_THREADS = size(typename Ktraits::TiledMmaQK{});
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<DTypeQ> / 8);

  static constexpr bool UseSchedulerBarrier =
      cutlass::sizeof_bits_v<DTypeQ> == 8 ? HEAD_DIM >= 128 : HEAD_DIM <= 128;
  using WarpScheduler = WarpScheduler<Ktraits, UseSchedulerBarrier>;

  // Host side kernel arguments
  struct Arguments {
    DTypeQ const* Q_ptr;
    LayoutT layout_Q;
    DTypeKV const* K_ptr;
    LayoutT layout_K;
    DTypeKV const* V_ptr;
    LayoutT layout_V;
    IdType const* kv_indices;
    int window_left;
    AdditionalParams additional_params;
  };

  // Device side kernel params
  struct Params {
    LayoutT layout_Q;
    LayoutT layout_K;
    LayoutT layout_V;
    TMA_Q tma_load_Q;
    DTypeKV* K_ptr;
    DTypeKV* V_ptr;
    IdType* kv_indices;
    int window_left;
    AdditionalParams additional_params;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.Q_ptr), args.layout_Q);
    TMA_Q tma_load_Q =
        make_tma_copy(GmemTiledCopyQ{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{});
    return {args.layout_Q,
            args.layout_K,
            args.layout_V,
            tma_load_Q,
            const_cast<DTypeKV*>(args.K_ptr),
            const_cast<DTypeKV*>(args.V_ptr),
            const_cast<IdType*>(args.kv_indices),
            args.window_left,
            args.additional_params};
  }

  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_Q.get_tma_descriptor());
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
    int thread_idx = threadIdx.x;
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (thread_idx / 32) % 4, 0);
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());

    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len] = block_coord;

    // Prepare the TMA loads
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

    constexpr int HEAD_DIM = get<2>(TileShape_QKD{});
    constexpr int CTA_KV = get<1>(TileShape_QKD{});
    auto indexed_gather = BlockSparseIndexedGather<IdType>(mainloop_params.kv_indices + kv_indptr);

    Tensor mK = make_block_sparse_tensor(  // (kv_len, D)
        make_gmem_ptr(mainloop_params.K_ptr + kv_head_idx * stride<2>(mainloop_params.layout_K)),
        make_shape(kv_len, HEAD_DIM), stride<0>(mainloop_params.layout_K), indexed_gather);
    Tensor mV = make_block_sparse_tensor(  // (kv_len, D)
        make_gmem_ptr(mainloop_params.V_ptr + kv_head_idx * stride<2>(mainloop_params.layout_V)),
        make_shape(kv_len, HEAD_DIM), stride<0>(mainloop_params.layout_V), indexed_gather);

    Tensor gK = local_tile(mK, select<1, 2>(TileShape_QKD{}), make_coord(_, _0{}));  // (KV, D, kv)
    Tensor gV = local_tile(mV, select<1, 2>(TileShape_QKD{}), make_coord(_, _0{}));  // (KV, D, kv)
    Tensor cKV = cute::make_identity_tensor(gK.shape());

    GmemTiledCopyKV gmem_tiled_copy_kv;
    auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_slice(thread_idx);

    Tensor tKgK = gmem_thr_copy_kv.partition_S(gK);     // (CPY, CPY_KV, CPY_D, kv)
    Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);     // (CPY, CPY_KV, CPY_D, PIPE)
    Tensor tVgV = gmem_thr_copy_kv.partition_S(gV);     // (CPY, CPY_KV, CPY_D, kv)
    Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);     // (CPY, CPY_KV, CPY_D, PIPE)
    Tensor tKVcKV = gmem_thr_copy_kv.partition_D(cKV);  // (CPY, CPY_KV, CPY_D)
    Tensor tKVcKVGroup = flatten_1(tKVcKV);             // (CPY, (CPY_KV, CPY_D))

    int valid_last_kv_tile_size = std::min<int>(kv_len - kv_tile_idx * CTA_KV, CTA_KV);
    auto predicate_fn = [&](auto coords) {
      auto s_coords = tKVcKVGroup(_0{}, coords);
      return elem_less(get<0>(s_coords), valid_last_kv_tile_size);
    };

    // load last k-tile
    {
      pipeline_k.producer_acquire(smem_pipe_write_k);
      Tensor tKgKiGroup = flatten_1(tKgK(_, _, _, kv_tile_idx));  // (CPY, (CPY_KV, CPY_D))
      Tensor tKsKiGroup =
          flatten_1(tKsK(_, _, _, smem_pipe_write_k.index()));  // (CPY, (CPY_KV, CPY_D))
      copy_if(gmem_tiled_copy_kv, predicate_fn, tKgKiGroup, tKsKiGroup);

      pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write_k;
    }

    // load Q tile
    if (warp_idx_in_warpgroup == 0) {
      cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + cutlass::NumThreadsPerWarp,
                                        static_cast<int>(NamedBarriers::kQueryEmpty));

      int lane_predicate = cute::elect_one_sync();
      if (lane_predicate) {
        shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
        copy(mainloop_params.tma_load_Q.with(
                 reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                     shared_storage.barrier_Q),
                 /*mcast_mask=*/0),
             tQgQ, tQsQ);
      }
    }

    shared_storage.barrier_O.wait((work_idx + 1) % 2);

    if (kv_tile_idx == swa_begin_kv_tile_idx) {
      pipeline_v.producer_acquire(smem_pipe_write_v);
      Tensor tVgViGroup = flatten_1(tVgV(_, _, _, kv_tile_idx));  // (CPY, (CPY_KV, CPY_D))
      Tensor tVsViGroup =
          flatten_1(tVsV(_, _, _, smem_pipe_write_v.index()));  // (CPY, (CPY_KV, CPY_D))
      copy_if(gmem_tiled_copy_kv, predicate_fn, tVgViGroup, tVsViGroup);

      pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write_v;
    } else {
      // load second last k-tile and last v-tile
      pipeline_k.producer_acquire(smem_pipe_write_k);
      Tensor tKgKi = tKgK(_, _, _, kv_tile_idx - 1);            // (CPY, CPY_KV, CPY_D)
      Tensor tKsKi = tKsK(_, _, _, smem_pipe_write_k.index());  // (CPY, CPY_KV, CPY_D)
      copy(gmem_tiled_copy_kv, tKgKi, tKsKi);

      pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write_k;

      pipeline_v.producer_acquire(smem_pipe_write_v);
      Tensor tVgViGroup = flatten_1(tVgV(_, _, _, kv_tile_idx));  // (CPY, (CPY_KV, CPY_D))
      Tensor tVsViGroup =
          flatten_1(tVsV(_, _, _, smem_pipe_write_v.index()));  // (CPY, (CPY_KV, CPY_D))
      copy_if(gmem_tiled_copy_kv, predicate_fn, tVgViGroup, tVsViGroup);

      pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
      --kv_tile_idx;
      ++smem_pipe_write_v;

      // load remaining k/v tiles
#pragma unroll 2
      for (; kv_tile_idx > swa_begin_kv_tile_idx; --kv_tile_idx) {
        pipeline_k.producer_acquire(smem_pipe_write_k);

        Tensor tKgKi = tKgK(_, _, _, kv_tile_idx - 1);            // (CPY, CPY_KV, CPY_D)
        Tensor tKsKi = tKsK(_, _, _, smem_pipe_write_k.index());  // (CPY, CPY_KV, CPY_D)
        copy(gmem_tiled_copy_kv, tKgKi, tKsKi);

        pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_k;

        pipeline_v.producer_acquire(smem_pipe_write_v);
        Tensor tVgVi = tVgV(_, _, _, kv_tile_idx);                // (CPY, CPY_KV, CPY_D)
        Tensor tVsVi = tVsV(_, _, _, smem_pipe_write_v.index());  // (CPY, CPY_KV, CPY_D)
        copy(gmem_tiled_copy_kv, tVgVi, tVsVi);

        pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_v;
      }
      scheduler.prefetch_next_work(scheduler_params, work_tile_info);

      // load first v tile
      {
        pipeline_v.producer_acquire(smem_pipe_write_v);
        Tensor tVgVi = tVgV(_, _, _, 0);                          // (CPY, (CPY_KV, CPY_D))
        Tensor tVsVi = tVsV(_, _, _, smem_pipe_write_v.index());  // (CPY, (CPY_KV, CPY_D))
        copy(gmem_tiled_copy_kv, tVgVi, tVsVi);
        pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_v;
      }
    }

    scheduler.broadcast_next_work(work_tile_info);
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                                PipelineState& smem_pipe_write_k,
                                PipelineState& smem_pipe_write_v) {
    pipeline_k.producer_tail(smem_pipe_write_k);
    pipeline_v.producer_tail(smem_pipe_write_v);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_SPARSE_MAINLOOP_CUH_
