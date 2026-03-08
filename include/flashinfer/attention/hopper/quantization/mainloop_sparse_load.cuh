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
    int64_t k_stride_n;     // Stride between consecutive KV tokens
    int64_t k_stride_h;     // Stride between heads
    int64_t k_page_stride;  // Stride between pages
    DTypeKV const* V_ptr;
    int64_t v_stride_n;     // Stride between consecutive KV tokens
    int64_t v_stride_h;     // Stride between heads
    int64_t v_page_stride;  // Stride between pages
    IdType const* kv_indices;
    uint32_t page_size;  // Size of each page
    int window_left;
    AdditionalParams additional_params;
  };

  // Device side kernel params
  struct Params {
    LayoutT layout_Q;
    TMA_Q tma_load_Q;
    DTypeKV* K_ptr;
    int64_t k_stride_n;
    int64_t k_stride_h;
    int64_t k_page_stride;
    DTypeKV* V_ptr;
    int64_t v_stride_n;
    int64_t v_stride_h;
    int64_t v_page_stride;
    IdType* kv_indices;
    uint_fastdiv page_size;  // Size of each page (as fastdiv for efficient divmod)
    int window_left;
    AdditionalParams additional_params;
    using DTypeKV = typename Ktraits::DTypeKV;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.Q_ptr), args.layout_Q);
    TMA_Q tma_load_Q =
        make_tma_copy(GmemTiledCopyQ{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{});
    return {args.layout_Q,
            tma_load_Q,
            const_cast<DTypeKV*>(args.K_ptr),
            args.k_stride_n,
            args.k_stride_h,
            args.k_page_stride,
            const_cast<DTypeKV*>(args.V_ptr),
            args.v_stride_n,
            args.v_stride_h,
            args.v_page_stride,
            const_cast<IdType*>(args.kv_indices),
            args.page_size,
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
    IdType const* kv_indices_ptr = mainloop_params.kv_indices + kv_indptr;

    // Setup for manual K/V loading with page table
    // Add kv_head_idx * stride_h offset to base pointers for correct head addressing
    DTypeKV* k_base_ptr = mainloop_params.K_ptr + kv_head_idx * mainloop_params.k_stride_h;
    DTypeKV* v_base_ptr = mainloop_params.V_ptr + kv_head_idx * mainloop_params.v_stride_h;
    int64_t k_stride_n = mainloop_params.k_stride_n;
    int64_t k_page_stride = mainloop_params.k_page_stride;
    int64_t v_stride_n = mainloop_params.v_stride_n;
    int64_t v_page_stride = mainloop_params.v_page_stride;

    GmemTiledCopyKV gmem_tiled_copy_kv;
    auto gmem_thr_copy_kv = gmem_tiled_copy_kv.get_slice(thread_idx);

    // Create coordinate tensors for partitioning
    Tensor cKV = cute::make_identity_tensor(make_shape(CTA_KV, HEAD_DIM));
    Tensor tKVcKV = gmem_thr_copy_kv.partition_D(cKV);  // (CPY, CPY_KV, CPY_D)
    Tensor tKVcKVGroup = flatten_1(tKVcKV);             // (CPY, (CPY_KV, CPY_D))
    Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);     // (CPY, CPY_KV, CPY_D, PIPE)
    Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);     // (CPY, CPY_KV, CPY_D, PIPE)

    // FA3-style prefetch offset optimization: pre-compute page offsets and share via shuffle
    // This reduces redundant page table lookups and address calculations
    int64_t my_kv_offset[2];  // Rolling buffer: page_idx * page_stride + entry_idx * stride_n
    int parity = 0;           // Buffer parity for double buffering, toggled with ^= 1

    // Group organization based on partition strategy (same as FP16 sparse_mainloop)
    // For FP8 with cp.async: AlignmentKV=16 (128bits/8bits), NUM_PRODUCER_THREADS=128
    // The simt gmem tiled copy partitions threads as: (thread_stride_M, thread_stride_K)
    // where thread_stride_M = threads / (CTA_KV / AlignmentKV) for column-major
    // NUM_KV_PER_ITER = number of KV elements each thread handles per iteration
    //
    // The tiled copy arrangement:
    // - Each thread loads AlignmentKV (16) elements contiguously in the D dimension
    // - Threads are spread across the (KV, D) tile
    // For column-major: threads stride by (D/AlignmentKV) in the KV dimension
    // D_stride = HEAD_DIM / AlignmentKV (e.g., 128/16=8 or 256/16=16)
    // Thread arrangement: threads = KV_stride * D_stride
    // So KV_stride = NUM_COPY_THREADS / D_stride = NUM_COPY_THREADS * AlignmentKV / HEAD_DIM
    // NUM_KV_PER_ITER = CTA_KV / KV_stride = CTA_KV * HEAD_DIM / (NUM_COPY_THREADS * AlignmentKV)
    static constexpr int NUM_COPY_THREADS = Ktraits::NUM_PRODUCER_THREADS;
    constexpr int NUM_KV_PER_ITER = CTA_KV * HEAD_DIM / (NUM_COPY_THREADS * AlignmentKV);
    constexpr int KV_STRIDE = CTA_KV / NUM_KV_PER_ITER;
    constexpr int NUM_GROUPS = KV_STRIDE;
    constexpr int THREADS_PER_GROUP = NUM_COPY_THREADS / NUM_GROUPS;
    constexpr int NUM_ITERS_PER_GROUP = NUM_KV_PER_ITER;

    int group_id = thread_idx / THREADS_PER_GROUP;
    int thread_in_group = thread_idx % THREADS_PER_GROUP;

    // Prefetch: compute page_idx * page_stride + entry_idx * stride_n
    // Uses parity to select buffer slot, caller must toggle parity after load
    auto prefetch_kv_offset = [&](int kv_tile_idx, int64_t stride_n, int64_t page_stride,
                                  bool use_predicate) {
      int kv_base_idx = kv_tile_idx * CTA_KV;

      int kv_idx_read = kv_base_idx + group_id + thread_in_group * KV_STRIDE;
      bool valid_read =
          thread_in_group < NUM_ITERS_PER_GROUP && (!use_predicate || kv_idx_read < kv_len);

      if (valid_read) {
        // Use divmod to find page and offset within page
        uint32_t page_iter, entry_idx;
        mainloop_params.page_size.divmod(kv_idx_read, page_iter, entry_idx);
        IdType page_idx = kv_indices_ptr[page_iter];
        // Pre-compute: page_idx * page_stride + entry_idx * stride_n
        my_kv_offset[parity] = page_idx * page_stride + entry_idx * stride_n;
      } else {
        my_kv_offset[parity] = 0;
      }
    };

    // Load K/V with pre-computed offsets using shuffle
    // Uses parity to select buffer slot, caller must toggle parity after load
    auto load_kv_with_prefetch = [&](DTypeKV* base_ptr, auto& tXsX, int tile_idx, int pipe_idx,
                                     bool use_predicate) {
      using Vec = AlignmentTypeKV;
      constexpr int VecSize = AlignmentKV;

      int kv_base_idx = tile_idx * CTA_KV;

      auto dst = recast<Vec>(flatten(tXsX(_, _, _, pipe_idx)));
      auto c = flatten(tKVcKV);

      constexpr unsigned FULL_MASK = 0xffffffff;

      // Load using FA3-style shuffle with pre-computed offsets
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(dst); ++i) {
        auto coord = c(VecSize * i);
        int kv_offset = get<0>(coord);
        int d_idx = get<1>(coord);
        int kv_idx = kv_base_idx + kv_offset;
        bool guard = !use_predicate || kv_idx < kv_len;

        // Shuffle the pre-computed offset (page_idx * page_stride + entry_idx * stride_n)
        int src_thread = group_id * THREADS_PER_GROUP + kv_offset / KV_STRIDE;
        int64_t base_offset = __shfl_sync(FULL_MASK, my_kv_offset[parity], src_thread);

        // Final address: base_ptr + base_offset + d_idx
        Vec const* src_ptr = reinterpret_cast<Vec const*>(base_ptr + base_offset + d_idx);
        cutlass::arch::cp_async_zfill<sizeof(Vec), cutlass::arch::CacheOperation::Global>(
            &dst(i), src_ptr, guard);
      }
    };

    int valid_last_kv_tile_size = std::min<int>(kv_len - kv_tile_idx * CTA_KV, CTA_KV);

    // load last k-tile with prefetch optimization
    // parity=0: prefetch kv_tile_idx -> my_kv_offset[0]
    // all threads are issuing as TMA is disabled
    {
      prefetch_kv_offset(kv_tile_idx, k_stride_n, k_page_stride, true);
      pipeline_k.producer_acquire(smem_pipe_write);
      load_kv_with_prefetch(k_base_ptr, tKsK, kv_tile_idx, smem_pipe_write.index(), true);
      pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
      // Note: don't toggle parity here, we reuse the same buffer for V below
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
      // first tile is the last tile, reuse kv_tile_idx prefetch for V (parity=0)
      pipeline_v.producer_acquire(smem_pipe_write);
      load_kv_with_prefetch(v_base_ptr, tVsV, kv_tile_idx, smem_pipe_write.index(), true);
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
      // parity=0: kv_tile_idx is in my_kv_offset[0]
      // Now prefetch kv_tile_idx-1 into my_kv_offset[1]
      parity ^= 1;  // parity=1
      prefetch_kv_offset(kv_tile_idx - 1, k_stride_n, k_page_stride, false);

      // Load V using prefetch from last K load (kv_tile_idx), use my_kv_offset[0]
      parity ^= 1;  // parity=0
      pipeline_v.producer_acquire(smem_pipe_write);
      load_kv_with_prefetch(v_base_ptr, tVsV, kv_tile_idx, smem_pipe_write.index(), true);
      pipeline_v.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

      // Transpose V
      pipeline_v.consumer_wait(smem_pipe_read);
      pipeline_vt.producer_acquire(smem_pipe_write);
      v_tranposer.do_transpose(sV_src, sVt_tgt, smem_pipe_read.index());
      pipeline_vt.producer_commit(smem_pipe_write);  // ping MMA consumer
      pipeline_v.consumer_release(smem_pipe_read);   // release V loading consumer
      ++smem_pipe_read;
      ++smem_pipe_write;  // update state, as K is loaded 1 step faster

      // Load K (kv_tile_idx - 1) using prefetched offset in my_kv_offset[1]
      parity ^= 1;  // parity=1
      pipeline_k.producer_acquire(smem_pipe_write);
      load_kv_with_prefetch(k_base_ptr, tKsK, kv_tile_idx - 1, smem_pipe_write.index(), false);
      pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

      --kv_tile_idx;
      // Now kv_tile_idx == kv_tile_idx-1, and its offset is in my_kv_offset[1] (parity=1)

      // load remaining k/v tiles
#pragma unroll 2
      for (; kv_tile_idx > swa_begin_kv_tile_idx; --kv_tile_idx) {
        // parity points to current kv_tile_idx's offset
        // Prefetch next K tile into the other buffer
        parity ^= 1;  // Toggle to other buffer for prefetch
        prefetch_kv_offset(kv_tile_idx - 1, k_stride_n, k_page_stride, false);

        // Load V using prefetch from previous K prefetch, use previous buffer
        parity ^= 1;  // Toggle back to kv_tile_idx's buffer
        pipeline_v.producer_acquire(smem_pipe_write);
        load_kv_with_prefetch(v_base_ptr, tVsV, kv_tile_idx, smem_pipe_write.index(), false);
        pipeline_v.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);

        // Transpose V
        pipeline_v.consumer_wait(smem_pipe_read);
        pipeline_vt.producer_acquire(smem_pipe_write);
        v_tranposer.do_transpose(sV_src, sVt_tgt, smem_pipe_read.index());
        pipeline_vt.producer_commit(smem_pipe_write);  // ping MMA consumer
        pipeline_v.consumer_release(smem_pipe_read);   // release V loading consumer
        ++smem_pipe_read;
        ++smem_pipe_write;  // update state, as K is loaded 1 step faster

        // Load K using prefetched offset
        parity ^= 1;  // Toggle to kv_tile_idx-1's buffer
        pipeline_k.producer_acquire(smem_pipe_write);
        load_kv_with_prefetch(k_base_ptr, tKsK, kv_tile_idx - 1, smem_pipe_write.index(), false);
        pipeline_k.producer_commit(smem_pipe_write, cutlass::arch::cpasync_barrier_arrive);
        // After loop update, kv_tile_idx becomes kv_tile_idx-1
        // parity already points to kv_tile_idx-1's buffer
      }
      scheduler.prefetch_next_work(scheduler_params, work_tile_info);

      // load first v tile (tile 0)
      {
        prefetch_kv_offset(0, v_stride_n, v_page_stride, false);
        pipeline_v.producer_acquire(smem_pipe_write);
        load_kv_with_prefetch(v_base_ptr, tVsV, 0, smem_pipe_write.index(), false);
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
