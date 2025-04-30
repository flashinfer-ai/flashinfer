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
#ifndef FLASHINFER_ATTENTION_HOPPER_FP8_SPARSE_MAINLOOP_CUH_
#define FLASHINFER_ATTENTION_HOPPER_FP8_SPARSE_MAINLOOP_CUH_

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/pipeline/pipeline.hpp>

#include "../../../math.cuh"
#include "../block_sparse_gather.cuh"
#include "../named_barrier.cuh"
#include "../utils.cuh"
#include "kernel_traits.cuh"

namespace flashinfer {

using namespace cute;

template <typename AdditionalParams, typename Ktraits, bool CAUSAL>
struct FP8SparseCollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int HEAD_DIM = Ktraits::HEAD_DIM;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  static constexpr auto AlignmentKV = 128 / cutlass::sizeof_bits<DTypeKV>::value;
  using AlignmentTypeKV = cute::uint_byte_t<static_cast<int>(sizeof(DTypeKV)) * AlignmentKV>;

  // Use ZFILL for out-of-bound V loading (avoid nan)
  using GmemCopyAtomKV = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<AlignmentTypeKV>, DTypeKV>;
  using GmemTiledCopyKV =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               GmemCopyAtomKV, Ktraits::NUM_PRODUCER_THREADS, AlignmentKV,
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

  // for sparse loading, we use cp.async
  static constexpr bool USE_TMA_LOAD_KV = false;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipelineVt = typename Ktraits::MainloopPipelineNoTMA;
  using PipelineParamsVt = typename MainloopPipelineVt::Params;

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
    using DTypeKV = typename Ktraits::DTypeKV;
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
                           MainloopPipeline pipeline_v, MainloopPipelineVt pipeline_vt,
                           PipelineState& smem_pipe_write, PipelineState& smem_pipe_read,
                           SharedStorage& shared_storage, Scheduler& scheduler,
                           typename Scheduler::Params const& scheduler_params,
                           typename Scheduler::WorkTileInfo& work_tile_info,
                           BlockCoord const& block_coord, int work_idx) {
    int thread_idx = threadIdx.x;
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (thread_idx / 32) % 4, 0);
    bool issue_tma_thread = (warp_idx_in_warpgroup == 0) && (elect_one_sync() == 1);

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());

    // *** Prepare In-kernel V Transpose ***
    using SmemLayoutVTransposeSrc = typename Ktraits::SmemLayoutVTransposeSrc;
    using SmemLayoutVtTransposeTgt = typename Ktraits::SmemLayoutVtTransposeTgt;

    Tensor sV_src = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVTransposeSrc{}));
    Tensor sVt_tgt = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), SmemLayoutVtTransposeTgt{}));
    auto v_tranposer = SmemTransposeFP8_64x64<Ktraits>();
    /* ----- V Transpose ---- */

    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;

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
    // all threads are issuing as TMA is disabled
    {
      pipeline_k.producer_acquire(smem_pipe_write);
      Tensor tKgKiGroup = flatten_1(tKgK(_, _, _, kv_tile_idx));  // (CPY, (CPY_KV, CPY_D))
      Tensor tKsKiGroup =
          flatten_1(tKsK(_, _, _, smem_pipe_write.index()));  // (CPY, (CPY_KV, CPY_D))
      copy_if(gmem_tiled_copy_kv, predicate_fn, tKgKiGroup, tKsKiGroup);
      pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
    }

    // Wait for the MMA warpgroups to say that smem_q is ready
    cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                      static_cast<int>(NamedBarriers::kQueryEmpty));
    // load Q tile
    if (issue_tma_thread) {
      shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
      copy(mainloop_params.tma_load_Q.with(
               reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                   shared_storage.barrier_Q),
               /*mcast_mask=*/0),
           tQgQ, tQsQ);
    }

    shared_storage.barrier_O.wait((work_idx + 1) % 2);

    if (kv_tile_idx == swa_begin_kv_tile_idx) {
      // first tile is the last tile
      pipeline_v.producer_acquire(smem_pipe_write);
      Tensor tVgViGroup = flatten_1(tVgV(_, _, _, kv_tile_idx));  // (CPY, (CPY_KV, CPY_D))
      Tensor tVsViGroup =
          flatten_1(tVsV(_, _, _, smem_pipe_write.index()));  // (CPY, (CPY_KV, CPY_D))
      copy_if(gmem_tiled_copy_kv, predicate_fn, tVgViGroup, tVsViGroup);
      pipeline_v.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

      // Transpose V
      pipeline_v.consumer_wait(smem_pipe_read);
      pipeline_vt.producer_acquire(smem_pipe_write);
      v_tranposer.do_transpose(sV_src, sVt_tgt, smem_pipe_read.index());
      pipeline_vt.producer_commit(smem_pipe_write);  // ping MMA consumer
      pipeline_v.consumer_release(smem_pipe_read);   // release V loading consumer
      ++smem_pipe_read;
      ++smem_pipe_write;  // update state, as K is loaded 1 step faster
    } else {
      // load second last k-tile and last v-tile
      pipeline_v.producer_acquire(smem_pipe_write);
      Tensor tVgViGroup = flatten_1(tVgV(_, _, _, kv_tile_idx));  // (CPY, (CPY_KV, CPY_D))
      Tensor tVsViGroup =
          flatten_1(tVsV(_, _, _, smem_pipe_write.index()));  // (CPY, (CPY_KV, CPY_D))
      copy_if(gmem_tiled_copy_kv, predicate_fn, tVgViGroup, tVsViGroup);
      pipeline_v.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

      // Transpose V
      pipeline_v.consumer_wait(smem_pipe_read);
      pipeline_vt.producer_acquire(smem_pipe_write);
      v_tranposer.do_transpose(sV_src, sVt_tgt, smem_pipe_read.index());
      pipeline_vt.producer_commit(smem_pipe_write);  // ping MMA consumer
      pipeline_v.consumer_release(smem_pipe_read);   // release V loading consumer
      ++smem_pipe_read;
      ++smem_pipe_write;  // update state, as K is loaded 1 step faster

      pipeline_k.producer_acquire(smem_pipe_write);
      Tensor tKgKi = tKgK(_, _, _, kv_tile_idx - 1);          // (CPY, CPY_KV, CPY_D)
      Tensor tKsKi = tKsK(_, _, _, smem_pipe_write.index());  // (CPY, CPY_KV, CPY_D)
      copy(gmem_tiled_copy_kv, tKgKi, tKsKi);
      pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

      --kv_tile_idx;

      // load remaining k/v tiles
#pragma unroll 2
      for (; kv_tile_idx > swa_begin_kv_tile_idx; --kv_tile_idx) {
        pipeline_v.producer_acquire(smem_pipe_write);
        Tensor tVgVi = tVgV(_, _, _, kv_tile_idx);              // (CPY, CPY_KV, CPY_D)
        Tensor tVsVi = tVsV(_, _, _, smem_pipe_write.index());  // (CPY, CPY_KV, CPY_D)
        copy(gmem_tiled_copy_kv, tVgVi, tVsVi);
        pipeline_v.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

        // Transpose V
        pipeline_v.consumer_wait(smem_pipe_read);
        pipeline_vt.producer_acquire(smem_pipe_write);
        v_tranposer.do_transpose(sV_src, sVt_tgt, smem_pipe_read.index());
        pipeline_vt.producer_commit(smem_pipe_write);  // ping MMA consumer
        pipeline_v.consumer_release(smem_pipe_read);   // release V loading consumer
        ++smem_pipe_read;
        ++smem_pipe_write;  // update state, as K is loaded 1 step faster

        pipeline_k.producer_acquire(smem_pipe_write);
        Tensor tKgKi = tKgK(_, _, _, kv_tile_idx - 1);          // (CPY, CPY_KV, CPY_D)
        Tensor tKsKi = tKsK(_, _, _, smem_pipe_write.index());  // (CPY, CPY_KV, CPY_D)
        copy(gmem_tiled_copy_kv, tKgKi, tKsKi);
        pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
      }
      scheduler.prefetch_next_work(scheduler_params, work_tile_info);

      // load first v tile
      {
        pipeline_v.producer_acquire(smem_pipe_write);
        Tensor tVgVi = tVgV(_, _, _, 0);                        // (CPY, (CPY_KV, CPY_D))
        Tensor tVsVi = tVsV(_, _, _, smem_pipe_write.index());  // (CPY, (CPY_KV, CPY_D))
        copy(gmem_tiled_copy_kv, tVgVi, tVsVi);
        pipeline_v.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

        // Transpose V
        pipeline_v.consumer_wait(smem_pipe_read);
        pipeline_vt.producer_acquire(smem_pipe_write);
        v_tranposer.do_transpose(sV_src, sVt_tgt, smem_pipe_read.index());
        pipeline_vt.producer_commit(smem_pipe_write);  // ping MMA consumer
        pipeline_v.consumer_release(smem_pipe_read);   // release V loading consumer
        ++smem_pipe_read;
        ++smem_pipe_write;  // update state, as K is loaded 1 step faster
      }
    }

    scheduler.broadcast_next_work(work_tile_info);
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                                PipelineState& smem_pipe_write) {
    pipeline_k.producer_tail(smem_pipe_write);
    pipeline_v.producer_tail(smem_pipe_write);
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_FP8_SPARSE_MAINLOOP_CUH_
