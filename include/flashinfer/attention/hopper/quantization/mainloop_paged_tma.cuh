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
#ifndef FLASHINFER_ATTENTION_HOPPER_FP8_PAGED_TMA_MAINLOOP_CUH_
#define FLASHINFER_ATTENTION_HOPPER_FP8_PAGED_TMA_MAINLOOP_CUH_

// Paged K/V loader built on TMA instead of per-thread cp.async gathers.
//
// The paged cache is described to TMA as a 4-D tensor (page_row, head_dim, page, kv_head): the
// row stride, the page stride and the head stride are arbitrary (multiples of 16 bytes), the
// head_dim is contiguous. One TMA box covers one page (PAGE rows) by one swizzle-width slab of
// head_dim, so a CTA_KV tile of PAGES_PER_TILE pages is PAGES_PER_TILE x (HEAD_DIM / box_width)
// boxes, all landing on the same stage barrier. The elected producer thread reads the page ids
// of the tile from the page table and issues the boxes; the rest of the producer warpgroup keeps
// its only remaining job, the in-smem V transpose. The smem stage layouts are the ones the MMA
// consumers already use; the box views below re-index the same bytes, they do not change them.
//
// A tile with pages past the end of a request re-loads its last valid page, keeping the
// transaction byte count of every stage constant. K rows past kv_len are masked before softmax;
// the corresponding V rows are zeroed before the in-smem transpose so 0 * NaN cannot leak into
// the output.

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

template <typename AdditionalParams, typename Ktraits, bool CAUSAL, int PAGE_ = 32>
struct FP8PagedTmaCollectiveMainloop {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  static constexpr int NUM_STAGES = Ktraits::NUM_STAGES;
  static constexpr int HEAD_DIM = Ktraits::HEAD_DIM;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;

  // Rows per page (16 / 32 / 64). The dispatcher selects the instance whose PAGE matches the
  // runtime page size.
  static constexpr int PAGE = PAGE_;
  static_assert(PAGE % 8 == 0 && PAGE >= 8, "a page must be a whole number of 8-row swizzle atoms");
  static constexpr int PAGES_PER_TILE = CTA_KV / PAGE;
  static_assert(CTA_KV % PAGE == 0, "CTA_KV must be a whole number of pages");
  // Box widths along head_dim: K uses the 128-byte swizzle of the MMA operand layout, V the
  // 64-byte swizzle of the transpose-friendly layout (both in fp8 elements).
  static constexpr int KBOX_D = 128;
  static constexpr int VBOX_D = 64;
  static_assert(HEAD_DIM % KBOX_D == 0 && HEAD_DIM % VBOX_D == 0);
  static constexpr int K_DBLK = HEAD_DIM / KBOX_D;
  static constexpr int V_DBLK = HEAD_DIM / VBOX_D;
  using GmemTiledCopyQ = cute::SM90_TMA_LOAD;
  using GmemTiledCopyKV = cute::SM90_TMA_LOAD;

  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;

  // One box (a single TMA instruction) and the stage buffers viewed as boxes:
  // (PAGE, box_width, (page_in_tile, dim_block, stage)). tile_to_shape orders the atoms
  // column-major, which is exactly the order of SmemLayoutK / SmemLayoutV, so the box views
  // address the same bytes as the operand views the consumers use.
  using SmemLayoutAtomK = typename Ktraits::SmemLayoutAtomK;
  using SmemLayoutAtomVBox = GMMA::Layout_K_SW64_Atom<DTypeKV>;
  using SmemBoxK = decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<PAGE>, Int<KBOX_D>>{}));
  using SmemBoxV = decltype(tile_to_shape(SmemLayoutAtomVBox{}, Shape<Int<PAGE>, Int<VBOX_D>>{}));
  // The box mode is flat: box index = page_in_tile + PAGES_PER_TILE * (dim_block + n_dblk * stage).
  static constexpr int NBOX_K = PAGES_PER_TILE * K_DBLK * NUM_STAGES;
  static constexpr int NBOX_V = PAGES_PER_TILE * V_DBLK * NUM_STAGES;
  using SmemLayoutKBoxes =
      decltype(tile_to_shape(SmemLayoutAtomK{}, Shape<Int<PAGE>, Int<KBOX_D>, Int<NBOX_K>>{}));
  using SmemLayoutVBoxes =
      decltype(tile_to_shape(SmemLayoutAtomVBox{}, Shape<Int<PAGE>, Int<VBOX_D>, Int<NBOX_V>>{}));
  static_assert(cosize_v<SmemLayoutKBoxes> == cosize_v<SmemLayoutK>);
  static_assert(cosize_v<SmemLayoutVBoxes> == cosize_v<SmemLayoutV>);

  using ShapeQT = cute::Shape<int32_t, int32_t, int32_t>;
  using StrideQT = cute::Shape<int64_t, _1, int64_t>;  // (N, D, H)
  using LayoutQT = cute::Layout<ShapeQT, StrideQT>;

  // Paged K/V as seen by TMA: (row in page, head_dim, page, kv_head)
  using ShapeKVT = cute::Shape<Int<PAGE>, Int<HEAD_DIM>, int64_t, int32_t>;
  using StrideKVT = cute::Shape<int64_t, _1, int64_t, int64_t>;
  using LayoutKVT = cute::Layout<ShapeKVT, StrideKVT>;

  using ShapeLseT = cute::Shape<int32_t, int32_t>;
  using StrideLseT = cute::Shape<_1, int64_t>;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  using TMA_Q = decltype(make_tma_copy(
      GmemTiledCopyQ{},
      make_tensor(make_gmem_ptr(static_cast<DTypeQ const*>(nullptr)),
                  repeat_like(StrideQT{}, int32_t(0)), StrideQT{}),
      SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{}));  // no mcast for Q

  using TMA_K = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)), ShapeKVT{}, StrideKVT{}),
      SmemBoxK{}, Shape<Int<PAGE>, Int<KBOX_D>>{}, _1{}));

  using TMA_V = decltype(make_tma_copy(
      GmemTiledCopyKV{},
      make_tensor(make_gmem_ptr(static_cast<DTypeKV const*>(nullptr)), ShapeKVT{}, StrideKVT{}),
      SmemBoxV{}, Shape<Int<PAGE>, Int<VBOX_D>>{}, _1{}));

  static constexpr bool USE_TMA_LOAD_KV = true;
  using MainloopPipeline = typename Ktraits::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;
  using MainloopPipelineVt = typename Ktraits::MainloopPipelineNoTMA;
  using PipelineParamsVt = typename MainloopPipelineVt::Params;

  static constexpr uint32_t TmaTransactionBytesQ =
      static_cast<uint32_t>(size(SmemLayoutQ{}) * cutlass::sizeof_bits_v<DTypeQ> / 8);
  // Bytes per stage; the K and V stages are the same size and every box of a stage signals
  // the same barrier.
  static constexpr uint32_t TmaTransactionBytesK =
      static_cast<uint32_t>(size(take<0, 2>(SmemLayoutK{})) * cutlass::sizeof_bits_v<DTypeKV> / 8);
  static_assert(TmaTransactionBytesK == PAGES_PER_TILE * K_DBLK * size(SmemBoxK{}));
  static_assert(TmaTransactionBytesK == PAGES_PER_TILE * V_DBLK * size(SmemBoxV{}));

  static constexpr bool UseSchedulerBarrier =
      (Ktraits::NUM_MMA_THREADS >= 256) &&
      (cutlass::sizeof_bits_v<DTypeQ> == 8 ? HEAD_DIM >= 128 : HEAD_DIM <= 128);
  using WarpScheduler = WarpScheduler<Ktraits, UseSchedulerBarrier>;

  // Host side kernel arguments
  struct Arguments {
    DTypeQ const* Q_ptr;
    LayoutQT layout_Q;
    DTypeKV const* K_ptr;
    int64_t k_stride_n;     // Stride between consecutive KV tokens
    int64_t k_stride_h;     // Stride between heads
    int64_t k_page_stride;  // Stride between pages
    DTypeKV const* V_ptr;
    int64_t v_stride_n;
    int64_t v_stride_h;
    int64_t v_page_stride;
    IdType const* kv_indices;
    int32_t num_kv_heads;
    int64_t num_pages;
    int window_left;
    AdditionalParams additional_params;
  };

  // Device side kernel params
  struct Params {
    LayoutQT layout_Q;
    LayoutKVT layout_K;
    LayoutKVT layout_V;
    TMA_Q tma_load_Q;
    TMA_K tma_load_K;
    TMA_V tma_load_V;
    IdType* kv_indices;
    int window_left;
    AdditionalParams additional_params;
    using DTypeKV = typename Ktraits::DTypeKV;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    Tensor mQ = make_tensor(make_gmem_ptr(args.Q_ptr), args.layout_Q);
    TMA_Q tma_load_Q =
        make_tma_copy(GmemTiledCopyQ{}, mQ, SmemLayoutQ{}, select<0, 2>(TileShape_QKD{}), _1{});
    LayoutKVT layout_K =
        make_layout(make_shape(Int<PAGE>{}, Int<HEAD_DIM>{}, args.num_pages, args.num_kv_heads),
                    make_stride(args.k_stride_n, _1{}, args.k_page_stride, args.k_stride_h));
    LayoutKVT layout_V =
        make_layout(make_shape(Int<PAGE>{}, Int<HEAD_DIM>{}, args.num_pages, args.num_kv_heads),
                    make_stride(args.v_stride_n, _1{}, args.v_page_stride, args.v_stride_h));
    Tensor mK = make_tensor(make_gmem_ptr(args.K_ptr), layout_K);
    TMA_K tma_load_K =
        make_tma_copy(GmemTiledCopyKV{}, mK, SmemBoxK{}, Shape<Int<PAGE>, Int<KBOX_D>>{}, _1{});
    Tensor mV = make_tensor(make_gmem_ptr(args.V_ptr), layout_V);
    TMA_V tma_load_V =
        make_tma_copy(GmemTiledCopyKV{}, mV, SmemBoxV{}, Shape<Int<PAGE>, Int<VBOX_D>>{}, _1{});
    return {args.layout_Q,
            layout_K,
            layout_V,
            tma_load_Q,
            tma_load_K,
            tma_load_V,
            const_cast<IdType*>(args.kv_indices),
            args.window_left,
            args.additional_params};
  }

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
                           MainloopPipeline pipeline_v, MainloopPipelineVt pipeline_vt,
                           PipelineState& smem_pipe_write, PipelineState& smem_pipe_read,
                           SharedStorage& shared_storage, Scheduler& scheduler,
                           typename Scheduler::Params const& scheduler_params,
                           typename Scheduler::WorkTileInfo& work_tile_info,
                           BlockCoord const& block_coord, int work_idx) {
    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
    Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});
    Tensor sKb = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutKBoxes{});
    Tensor sVb = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVBoxes{});

    Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.layout_Q.shape());
    Tensor mK = mainloop_params.tma_load_K.get_tma_tensor(mainloop_params.layout_K.shape());
    Tensor mV = mainloop_params.tma_load_V.get_tma_tensor(mainloop_params.layout_V.shape());

    // *** Prepare In-kernel V Transpose ***
    using SmemLayoutVTransposeSrc = typename Ktraits::SmemLayoutVTransposeSrc;
    using SmemLayoutVtTransposeTgt = typename Ktraits::SmemLayoutVtTransposeTgt;

    Tensor sV_src = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutVTransposeSrc{}));
    Tensor sVt_tgt = as_position_independent_swizzle_tensor(
        make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), SmemLayoutVtTransposeTgt{}));
    auto v_tranposer = SmemTransposeFP8_64x64<Ktraits>();

    auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
        block_coord;

    // Prepare the TMA loads
    Tensor gQ = get_local_tile_tensor(mQ, select<0, 2>(TileShape_QKD{}), qo_head_idx, qo_indptr,
                                      qo_len)(_, _, q_tile_idx);  // (Q, D)
    // K/V coordinates of this head: (PAGE, D, page)
    Tensor mK_h = mK(_, _, _, kv_head_idx);
    Tensor mV_h = mV(_, _, _, kv_head_idx);

    Tensor sQ_x = make_tensor(sQ.data(), make_layout(sQ.layout(), Layout<_1>{}));
    Tensor gQ_x = make_tensor(gQ.data(), make_layout(gQ.layout(), Layout<_1>{}));
    auto [tQgQ, tQsQ] =
        tma_partition(mainloop_params.tma_load_Q, _0{}, Layout<_1>{}, group_modes<0, 2>(sQ_x),
                      group_modes<0, 2>(gQ_x));  // (TMA), (TMA)
    Tensor sKb_g = group_modes<0, 2>(sKb);       // ((PAGE, KBOX_D), NBOX_K)
    Tensor sVb_g = group_modes<0, 2>(sVb);       // ((PAGE, VBOX_D), NBOX_V)

    int num_kv_tiles = get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
    int kv_tile_idx = num_kv_tiles - 1;
    int swa_begin_kv_tile_idx = 0;
    if constexpr (LEFT_SLIDING_WINDOW) {
      swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                       q_tile_idx, qo_len, kv_len);
    }

    // All WG proceeds here, only one thread in each WG will issue TMA load
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    bool issue_tma_thread = (warp_idx_in_warpgroup == 0) && (lane_predicate == 1);

    // Page ids of the current tile, read one tile ahead by the issuing thread.
    IdType const* kv_indices_ptr = mainloop_params.kv_indices + kv_indptr;
    // Plain arithmetic: a static constexpr member passed by reference is an ODR-use nvcc rejects in
    // device code.
    const int num_pages = (kv_len + PAGE - 1) / PAGE;
    int64_t pages[PAGES_PER_TILE];
    int64_t pages_next[PAGES_PER_TILE];
    auto read_pages = [&](int tile, int64_t* dst) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < PAGES_PER_TILE; ++j) {
        int slot = tile * PAGES_PER_TILE + j;
        dst[j] = kv_indices_ptr[slot < num_pages ? slot : num_pages - 1];
      }
    };
    // The ids of the next tile are read right after the current tile is issued, so the load
    // latency overlaps the wait for a free stage instead of sitting in front of the TMA issue.
    auto prefetch_next = [&](int tile) {
      if (tile >= swa_begin_kv_tile_idx) read_pages(tile, pages_next);
    };
    auto take_next = [&]() {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < PAGES_PER_TILE; ++j) pages[j] = pages_next[j];
    };
    // One box per (page of the tile, head_dim block). The gmem side is partitioned per dim
    // block so that its only remaining mode is the page id.
    auto issue_k = [&](int stage) {
      auto* bar = pipeline_k.producer_get_barrier(smem_pipe_write);
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < K_DBLK; ++c) {
        Tensor gK_c = group_modes<0, 2>(local_tile(mK_h, Shape<Int<PAGE>, Int<KBOX_D>>{},
                                                   make_coord(_0{}, c)));  // ((PAGE,KBOX_D), page)
        auto [tKgK, tKsK] =
            tma_partition(mainloop_params.tma_load_K, _0{}, Layout<_1>{}, sKb_g, gK_c);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < PAGES_PER_TILE; ++j) {
          copy(mainloop_params.tma_load_K.with(*bar, /*mcast_mask=*/0), tKgK(_, pages[j]),
               tKsK(_, j + PAGES_PER_TILE * (c + K_DBLK * stage)));
        }
      }
    };
    auto issue_v = [&](int stage) {
      auto* bar = pipeline_v.producer_get_barrier(smem_pipe_write);
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < V_DBLK; ++c) {
        Tensor gV_c = group_modes<0, 2>(
            local_tile(mV_h, Shape<Int<PAGE>, Int<VBOX_D>>{}, make_coord(_0{}, c)));
        auto [tVgV, tVsV] =
            tma_partition(mainloop_params.tma_load_V, _0{}, Layout<_1>{}, sVb_g, gV_c);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < PAGES_PER_TILE; ++j) {
          copy(mainloop_params.tma_load_V.with(*bar, /*mcast_mask=*/0), tVgV(_, pages[j]),
               tVsV(_, j + PAGES_PER_TILE * (c + V_DBLK * stage)));
        }
      }
    };

    // TMA gathers whole pages. Clear rows beyond kv_len before transposing V: their attention
    // probabilities are zero, but leaving arbitrary cache data here could still produce NaNs
    // through 0 * NaN in the P*V matrix multiply.
    auto zero_v_tail = [&](int stage, int tile) {
      int valid_rows = kv_len - tile * CTA_KV;
      if (valid_rows > CTA_KV) valid_rows = CTA_KV;
      if (valid_rows < CTA_KV) {
        constexpr int kVectorElements = 16 / sizeof(DTypeKV);
        constexpr int kVectorsPerRow = HEAD_DIM / kVectorElements;
        int num_vectors = (CTA_KV - valid_rows) * kVectorsPerRow;
        for (int i = threadIdx.x; i < num_vectors; i += Ktraits::NUM_PRODUCER_THREADS) {
          int row = valid_rows + i / kVectorsPerRow;
          int col = (i % kVectorsPerRow) * kVectorElements;
          *reinterpret_cast<uint4*>(&sV(row, col, stage)) = uint4{0, 0, 0, 0};
        }
        cutlass::arch::NamedBarrier::sync(Ktraits::NUM_PRODUCER_THREADS,
                                          static_cast<int>(NamedBarriers::kProducerWG));
      }
    };

    if (issue_tma_thread) {
      read_pages(kv_tile_idx, pages);
      pipeline_k.producer_acquire(smem_pipe_write);
      issue_k(smem_pipe_write.index());
    }

    // Wait for the MMA warpgroups to say that smem_q is ready
    cutlass::arch::NamedBarrier::sync(NUM_MMA_THREADS + Ktraits::NUM_PRODUCER_THREADS,
                                      static_cast<int>(NamedBarriers::kQueryEmpty));

    if (issue_tma_thread) {
      shared_storage.barrier_Q.arrive_and_expect_tx(TmaTransactionBytesQ);
      copy(mainloop_params.tma_load_Q.with(
               reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(
                   shared_storage.barrier_Q),
               /*mcast_mask=*/0),
           tQgQ, tQsQ);

      pipeline_v.producer_acquire(smem_pipe_write);
      issue_v(smem_pipe_write.index());
      prefetch_next(kv_tile_idx - 1);
    }

    // Wait for warp 1 to signal that smem_v are ready and V can be copied from gmem
    // Need ClusterBarrier, not just NamedBarrier. Otherwise we might have CTA 0 finishing the
    // TMA store on O first, call TMA multicast load on V, before CTA 1 can finishing TMA store on
    // O.
    shared_storage.barrier_O.wait((work_idx + 1) % 2);

    pipeline_v.consumer_wait(smem_pipe_read);
    zero_v_tail(smem_pipe_read.index(), kv_tile_idx);
    pipeline_vt.producer_acquire(smem_pipe_write);
    v_tranposer.do_transpose(sV_src, sVt_tgt, smem_pipe_read.index());
    pipeline_vt.producer_commit(smem_pipe_write);
    pipeline_v.consumer_release(smem_pipe_read);
    ++smem_pipe_read;
    ++smem_pipe_write;
    --kv_tile_idx;

#pragma unroll 2
    for (; kv_tile_idx >= swa_begin_kv_tile_idx; --kv_tile_idx) {
      if (issue_tma_thread) {
        take_next();
        pipeline_k.producer_acquire(smem_pipe_write);
        issue_k(smem_pipe_write.index());
        pipeline_v.producer_acquire(smem_pipe_write);
        issue_v(smem_pipe_write.index());
        prefetch_next(kv_tile_idx - 1);
      }
      pipeline_v.consumer_wait(smem_pipe_read);
      zero_v_tail(smem_pipe_read.index(), kv_tile_idx);
      pipeline_vt.producer_acquire(smem_pipe_write);
      v_tranposer.do_transpose(sV_src, sVt_tgt, smem_pipe_read.index());
      pipeline_vt.producer_commit(smem_pipe_write);
      pipeline_v.consumer_release(smem_pipe_read);
      ++smem_pipe_read;
      ++smem_pipe_write;
    }
    scheduler.prefetch_next_work(scheduler_params, work_tile_info);
    scheduler.broadcast_next_work(work_tile_info);
  }

  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline_k, MainloopPipeline pipeline_v,
                                PipelineState& smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();
    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_warpgroup == 0 && lane_predicate) {
      pipeline_k.producer_tail(smem_pipe_write);
      pipeline_v.producer_tail(smem_pipe_write);
    }
  }
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_FP8_PAGED_TMA_MAINLOOP_CUH_
