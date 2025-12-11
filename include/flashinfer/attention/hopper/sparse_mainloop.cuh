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

#include "../../fastdiv.cuh"
#include "../../math.cuh"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "named_barrier.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

template <typename AdditionalParams, typename Ktraits, bool CAUSAL, bool MULTIITEMSCORING = false>
struct SparseCollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int HEAD_DIM_QK = Ktraits::HEAD_DIM_QK;
  static constexpr int HEAD_DIM_VO = Ktraits::HEAD_DIM_VO;
  static_assert(HEAD_DIM_QK == HEAD_DIM_VO);
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;

  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  static constexpr auto AlignmentKV = 128 / cutlass::sizeof_bits<DTypeKV>::value;
  using AlignmentTypeKV = cute::uint_byte_t<static_cast<int>(sizeof(DTypeKV)) * AlignmentKV>;
  // NOTE(Zihao): use SM80_CP_ASYNC for sparse loading of KV-cache
  using GmemCopyAtomKV = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<AlignmentTypeKV>, DTypeKV>;
  using GmemTiledCopyK =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               GmemCopyAtomKV, NUM_COPY_THREADS, AlignmentKV,
               cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>,
               decltype(cute::get<1>(TileShape_QKD{})), decltype(cute::get<2>(TileShape_QKD{}))>());
  using GmemTiledCopyV =
      decltype(cutlass::gemm::collective::detail::make_simt_gmem_tiled_copy<
               GmemCopyAtomKV, NUM_COPY_THREADS, AlignmentKV,
               cutlass::detail::TagToStrideB_t<cutlass::layout::ColumnMajor>,
               decltype(cute::get<2>(TileShape_PDV{})), decltype(cute::get<1>(TileShape_PDV{}))>());

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
  // Verify NUM_PRODUCER_THREADS matches NUM_COPY_THREADS for sparse loading
  static_assert(Ktraits::NUM_PRODUCER_THREADS == NUM_COPY_THREADS,
                "NUM_PRODUCER_THREADS must equal NUM_COPY_THREADS for sparse/paged KV loading");
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<DTypeQ> / 8);

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
    IdType const* kv_indices;
    int window_left;
    int64_t k_page_stride;  // Stride between pages for K (paged_k.stride(0))
    int64_t v_page_stride;  // Stride between pages for V (paged_v.stride(0))
    uint32_t page_size;     // Size of each page
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
    int64_t k_page_stride;   // Stride between pages for K
    int64_t v_page_stride;   // Stride between pages for V
    uint_fastdiv page_size;  // Size of each page (as fastdiv for efficient divmod)
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
            args.k_page_stride,            // Use stride from arguments
            args.v_page_stride,            // Use stride from arguments
            uint_fastdiv(args.page_size),  // Convert page_size to fastdiv
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
    if constexpr (MULTIITEMSCORING) {
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
    int thread_idx = threadIdx.x;
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (thread_idx / 32) % 4, 0);
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());

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

    constexpr int HEAD_DIM_QK = get<2>(TileShape_QKD{});
    constexpr int HEAD_DIM_VO = get<1>(TileShape_PDV{});
    constexpr int CTA_KV = get<1>(TileShape_QKD{});

    // Store base pointers and indices for manual page table lookup
    DTypeKV* K_ptr_base = mainloop_params.K_ptr + kv_head_idx * stride<2>(mainloop_params.layout_K);
    DTypeKV* V_ptr_base = mainloop_params.V_ptr + kv_head_idx * stride<2>(mainloop_params.layout_V);
    IdType const* kv_indices_ptr = mainloop_params.kv_indices + kv_indptr;
    // Use the page stride (stride between pages) and stride within page
    int64_t k_page_stride = mainloop_params.k_page_stride;
    int64_t v_page_stride = mainloop_params.v_page_stride;
    int64_t k_stride_n =
        stride<0>(mainloop_params.layout_K);  // Stride within page (between tokens)
    int64_t v_stride_n = stride<0>(mainloop_params.layout_V);

    // Create dummy tensors for partitioning with contiguous column-major layout
    // NOTE: We use a virtual contiguous layout for correct partitioning,
    // actual addressing uses page table lookup
    Tensor gK =
        make_tensor(make_gmem_ptr(static_cast<DTypeKV*>(nullptr)), make_shape(CTA_KV, HEAD_DIM_QK),
                    make_stride(HEAD_DIM_QK, _1{}));  // Column-major: (KV, D)
    Tensor gK_tiled =
        local_tile(gK, select<1, 2>(TileShape_QKD{}), make_coord(_, _0{}));  // (KV, D_K, kv)
    Tensor gV =
        make_tensor(make_gmem_ptr(static_cast<DTypeKV*>(nullptr)), make_shape(CTA_KV, HEAD_DIM_VO),
                    make_stride(HEAD_DIM_VO, _1{}));  // Column-major: (KV, D)
    Tensor gV_tiled =
        local_tile(gV, select<2, 1>(TileShape_PDV{}), make_coord(_, _0{}));  // (KV, D_V, kv)
    Tensor cK = cute::make_identity_tensor(gK_tiled.shape());
    Tensor cV = cute::make_identity_tensor(gV_tiled.shape());

    GmemTiledCopyK gmem_tiled_copy_k;
    GmemTiledCopyV gmem_tiled_copy_v;
    auto gmem_thr_copy_k = gmem_tiled_copy_k.get_slice(thread_idx);
    auto gmem_thr_copy_v = gmem_tiled_copy_v.get_slice(thread_idx);

    Tensor tKgK = gmem_thr_copy_k.partition_S(gK_tiled);  // (CPY, CPY_KV, CPY_D, kv)
    Tensor tKsK = gmem_thr_copy_k.partition_D(sK);        // (CPY, CPY_KV, CPY_D, PIPE)
    Tensor tVgV = gmem_thr_copy_v.partition_S(gV_tiled);  // (CPY, CPY_KV, CPY_D, kv)
    Tensor tVsV = gmem_thr_copy_v.partition_D(sV);        // (CPY, CPY_KV, CPY_D, PIPE)
    Tensor tKcK = gmem_thr_copy_k.partition_D(cK);        // (CPY, CPY_KV, CPY_D, kv)
    Tensor tVcV = gmem_thr_copy_v.partition_D(cV);        // (CPY, CPY_KV, CPY_D, kv)

    int valid_last_kv_tile_size = std::min<int>(kv_len - kv_tile_idx * CTA_KV, CTA_KV);
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

    // FA3-style cooperative loading: store pre-computed base offset for each KV position
    int64_t my_kv_offset[2];  // Rolling buffer: page_idx * page_stride + entry_idx * stride_n
    int parity = 0;           // Buffer parity for double buffering, toggled with ^= 1

    // Group organization based on partition strategy
    constexpr int NUM_KV_PER_ITER = decltype(size<1>(tKcK))::value;   // e.g., 12
    constexpr int KV_STRIDE = CTA_KV / NUM_KV_PER_ITER;               // 96/12 = 8
    constexpr int NUM_GROUPS = KV_STRIDE;                             // 8 groups (one per lane)
    constexpr int THREADS_PER_GROUP = NUM_COPY_THREADS / NUM_GROUPS;  // 128/8 = 16
    constexpr int NUM_ITERS_PER_GROUP = NUM_KV_PER_ITER;              // 12 iterations per group

    int group_id = thread_idx / THREADS_PER_GROUP;         // 0-7
    int thread_in_group = thread_idx % THREADS_PER_GROUP;  // 0-15

    // Prefetch: compute page_idx * page_stride + entry_idx * stride_n
    // NOTE: Assumes K and V have same strides (asserted on host side)
    // Uses parity to select buffer slot, caller must toggle parity after load
    auto prefetch_kv_offset = [&](int kv_tile_idx, bool use_predicate) {
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
        my_kv_offset[parity] = page_idx * k_page_stride + entry_idx * k_stride_n;
      } else {
        my_kv_offset[parity] = 0;
      }
    };

    // Unified helper lambda to load K or V with pre-computed offsets
    // Uses parity to select buffer slot, caller must toggle parity after load
    auto load_kv_with_gather = [&](auto&& tXsX, auto&& tXcX, DTypeKV* base_ptr, int kv_tile_idx,
                                   int stage_idx, bool use_predicate) {
      using Vec = AlignmentTypeKV;
      constexpr int VecSize = sizeof(Vec) / sizeof(DTypeKV);

      int kv_base_idx = kv_tile_idx * CTA_KV;

      auto dst = recast<Vec>(flatten(tXsX(_, _, _, stage_idx)));
      auto c = flatten(tXcX(_, _, _, kv_tile_idx));

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
        // where base_offset = page_idx * page_stride + entry_idx * stride_n
        Vec const* src_ptr = reinterpret_cast<Vec const*>(base_ptr + base_offset + d_idx);
        cutlass::arch::cp_async_zfill<sizeof(Vec), cutlass::arch::CacheOperation::Global>(
            &dst(i), src_ptr, guard);
      }
    };

    // load last k-tile
    // parity=0: prefetch kv_tile_idx -> my_kv_offset[0]
    {
      prefetch_kv_offset(kv_tile_idx, true);
      pipeline_k.producer_acquire(smem_pipe_write_k);
      load_kv_with_gather(tKsK, tKcK, K_ptr_base, kv_tile_idx, smem_pipe_write_k.index(), true);
      pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write_k;
      // Note: don't toggle parity here, we reuse the same buffer for V below
    }

    // All producer threads sync on kQueryEmpty barrier before loading Q
    cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                      static_cast<int>(NamedBarriers::kQueryEmpty));

    // load Q tile (only warp 0 issues TMA)
    if (warp_idx_in_warpgroup == 0) {
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
      // kv_tile_idx already prefetched above (parity=0), reuse it for V
      pipeline_v.producer_acquire(smem_pipe_write_v);
      load_kv_with_gather(tVsV, tVcV, V_ptr_base, kv_tile_idx, smem_pipe_write_v.index(), true);
      pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write_v;
    } else {
      // load second last k-tile and last v-tile
      // parity=0: kv_tile_idx is in my_kv_offset[0]
      // Now prefetch kv_tile_k into my_kv_offset[1]
      int kv_tile_k = kv_tile_idx_decrement(kv_tile_idx);
      parity ^= 1;  // parity=1
      prefetch_kv_offset(kv_tile_k, false);
      pipeline_k.producer_acquire(smem_pipe_write_k);
      load_kv_with_gather(tKsK, tKcK, K_ptr_base, kv_tile_k, smem_pipe_write_k.index(), false);
      pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
      ++smem_pipe_write_k;

      // Load V for kv_tile_idx using my_kv_offset[0]
      parity ^= 1;  // parity=0
      pipeline_v.producer_acquire(smem_pipe_write_v);
      load_kv_with_gather(tVsV, tVcV, V_ptr_base, kv_tile_idx, smem_pipe_write_v.index(), true);
      pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
      kv_tile_idx = kv_tile_idx_decrement(kv_tile_idx);
      ++smem_pipe_write_v;
      // Now kv_tile_idx == kv_tile_k, and its offset is in my_kv_offset[1]
      parity ^= 1;  // parity=1, pointing to kv_tile_idx's offset

      // load remaining k/v tiles
#pragma unroll 2
      for (; kv_tile_idx > swa_begin_kv_tile_idx;
           kv_tile_idx = kv_tile_idx_decrement(kv_tile_idx)) {
        // parity points to current kv_tile_idx's offset
        // Prefetch next K tile into the other buffer
        int kv_tile_k = kv_tile_idx_decrement(kv_tile_idx);
        parity ^= 1;  // Toggle to other buffer for prefetch
        prefetch_kv_offset(kv_tile_k, false);
        pipeline_k.producer_acquire(smem_pipe_write_k);
        load_kv_with_gather(tKsK, tKcK, K_ptr_base, kv_tile_k, smem_pipe_write_k.index(), false);
        pipeline_k.producer_commit(smem_pipe_write_k, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_k;

        // Load V for kv_tile_idx using the previous buffer
        parity ^= 1;  // Toggle back to kv_tile_idx's buffer
        pipeline_v.producer_acquire(smem_pipe_write_v);
        load_kv_with_gather(tVsV, tVcV, V_ptr_base, kv_tile_idx, smem_pipe_write_v.index(), false);
        pipeline_v.producer_commit(smem_pipe_write_v, cutlass::arch::cpasync_barrier_arrive);
        ++smem_pipe_write_v;
        // After loop update, kv_tile_idx becomes kv_tile_k
        // Toggle parity to point to kv_tile_k's buffer for next iteration
        parity ^= 1;
      }
      scheduler.prefetch_next_work(scheduler_params, work_tile_info);

      // load first v tile (tile 0)
      {
        prefetch_kv_offset(0, false);
        pipeline_v.producer_acquire(smem_pipe_write_v);
        load_kv_with_gather(tVsV, tVcV, V_ptr_base, 0, smem_pipe_write_v.index(), false);
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
