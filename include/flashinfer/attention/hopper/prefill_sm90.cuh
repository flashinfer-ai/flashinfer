/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_PREFILL_SM90_CUH_
#define FLASHINFER_ATTENTION_HOPPER_PREFILL_SM90_CUH_

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <type_traits>
#include <vector>

#include "../../cutlass_utils.cuh"
#include "../../exception.h"
#include "../mask.cuh"
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mainloop.cuh"
#include "mainloop_mma.cuh"
#include "sparse_mainloop.cuh"
#include "tile_scheduler.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

DEFINE_HAS_MEMBER(maybe_prefix_len_ptr)
DEFINE_HAS_MEMBER(maybe_token_pos_in_items_ptr)
DEFINE_HAS_MEMBER(token_pos_in_items_len)
DEFINE_HAS_MEMBER(maybe_max_item_len_ptr)

template <typename CollectiveMainloop, typename CollectiveEpilogue, typename Ktraits,
          bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename TileScheduler,
          bool MULTIITEMSCORING = false>
__global__ void __launch_bounds__(Ktraits::NUM_WARPS* cutlass::NumThreadsPerWarp, 1)
    PrefillWithKVCacheKernel(CUTE_GRID_CONSTANT
                             typename CollectiveMainloop::Params const mainloop_params,
                             CUTE_GRID_CONSTANT
                             typename CollectiveEpilogue::Params const epilogue_params,
                             CUTE_GRID_CONSTANT
                             typename TileScheduler::Params const scheduler_params) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;
  using AttentionVariant = typename Ktraits::AttentionVariant;

  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int NUM_COPY_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr int CTA_Q = Ktraits::CTA_Q;
  static constexpr int CTA_KV = Ktraits::CTA_KV;

  static constexpr bool use_tma_load_kv = CollectiveMainloop::USE_TMA_LOAD_KV;

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  extern __shared__ char shared_memory[];
  auto& shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  // Issue Tma Descriptor Prefetch from a single thread
  if (warp_idx == 0 && lane_predicate) {
    CollectiveMainloop::prefetch_tma_descriptors(mainloop_params);
    CollectiveEpilogue::prefetch_tma_descriptors(epilogue_params);
  }

  // Obtain warp index
  int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  PipelineParams pipeline_params;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  if constexpr (use_tma_load_kv) {
    pipeline_params.is_leader = warp_group_thread_idx == 0;
    pipeline_params.num_consumers = NUM_MMA_THREADS;
  } else {
    pipeline_params.producer_arv_count = NUM_COPY_THREADS;
    pipeline_params.consumer_arv_count = NUM_MMA_THREADS;
  }

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_Q.init(/*num_threads=*/1);
    shared_storage.barrier_O.init(/*num_threads=*/1);
  }
  // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
  MainloopPipeline pipeline_k = [&] {
    if constexpr (use_tma_load_kv) {
      pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params);
    }
  }();

  MainloopPipeline pipeline_v = [&] {
    if constexpr (use_tma_load_kv) {
      pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesV;
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params);
    }
  }();

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  // We need this to guarantee that the Pipeline init is visible to all producers and consumer
  // blocks in the Cluster
  __syncthreads();

  uint32_t* maybe_prefix_len_ptr = nullptr;
  if constexpr (has_maybe_prefix_len_ptr_v<decltype(mainloop_params.additional_params)>) {
    maybe_prefix_len_ptr = mainloop_params.additional_params.maybe_prefix_len_ptr;
  }
  uint16_t* maybe_token_pos_in_items_ptr = nullptr;
  if constexpr (has_maybe_token_pos_in_items_ptr_v<decltype(mainloop_params.additional_params)>) {
    maybe_token_pos_in_items_ptr = mainloop_params.additional_params.maybe_token_pos_in_items_ptr;
  }
  uint32_t token_pos_in_items_len = 0;
  if constexpr (has_token_pos_in_items_len_v<decltype(mainloop_params.additional_params)>) {
    token_pos_in_items_len = mainloop_params.additional_params.token_pos_in_items_len;
  }
  uint16_t* maybe_max_item_len_ptr = nullptr;
  if constexpr (has_maybe_max_item_len_ptr_v<decltype(mainloop_params.additional_params)>) {
    maybe_max_item_len_ptr = mainloop_params.additional_params.maybe_max_item_len_ptr;
  }

  if (warp_group_idx == 0) {  // Producer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_dealloc<Ktraits::NUM_WARPS == 12 ? 24 : 32>();
    } else {
      cutlass::arch::warpgroup_reg_dealloc<72>();
    }

    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (!use_tma_load_kv || warp_idx_in_warpgroup == 0) {  // Load Q, K, V
      PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

      int work_idx = 0;

      TileScheduler scheduler;
      for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
           work_tile_info.is_valid(scheduler_params);
           work_tile_info = scheduler.template get_next_work</*is_producer=*/true>(
               scheduler_params, work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len,
              batch_idx] = block_coord;

        if (q_tile_idx * CTA_Q >= qo_len) {
          continue;
        }
        int num_kv_tiles =
            collective_mainloop.get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
        // With split-KV this work covers the KV tiles [kv_tile_begin, num_kv_tiles).
        int kv_tile_begin = 0;
        auto [kv_chunk_idx, num_kv_chunks] = work_tile_info.get_kv_chunk(scheduler_params);
        if (num_kv_chunks > 1) {
          kv_tile_begin = num_kv_tiles * kv_chunk_idx / num_kv_chunks;
          num_kv_tiles = num_kv_tiles * (kv_chunk_idx + 1) / num_kv_chunks;
        }
        if (num_kv_tiles <= kv_tile_begin) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          scheduler.broadcast_next_work(work_tile_info);
          continue;
        }
        int num_kv_tiles_outside_items_window = 0;
        int num_kv_tiles_prefix = 0;
        if constexpr (MULTIITEMSCORING) {
          auto prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
          auto max_item_len = __ldg(maybe_max_item_len_ptr + batch_idx);
          auto valid_items_window_len =
              std::max(0, q_tile_idx * CTA_Q + kv_len - qo_len - max_item_len);
          num_kv_tiles_outside_items_window = valid_items_window_len / CTA_KV;
          num_kv_tiles_prefix = cute::ceil_div(prefix_len, CTA_KV);
        }
        if constexpr (MULTIITEMSCORING) {
          collective_mainloop.load<LEFT_SLIDING_WINDOW>(
              mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
              shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
              num_kv_tiles_outside_items_window, num_kv_tiles_prefix);
        } else if constexpr (!TileScheduler::kSplitKV) {
          collective_mainloop.template load<LEFT_SLIDING_WINDOW>(
              mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
              shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx);
        } else if constexpr (use_tma_load_kv) {
          collective_mainloop.template load<LEFT_SLIDING_WINDOW, /*SPLIT_KV=*/true>(
              mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
              shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
              kv_tile_begin, num_kv_tiles);
        } else {
          collective_mainloop.template load<LEFT_SLIDING_WINDOW, /*SPLIT_KV=*/true>(
              mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v,
              shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx,
              /*num_kv_tiles_outside_items_window=*/0, /*num_kv_tiles_prefix=*/0, kv_tile_begin,
              num_kv_tiles);
        }
        ++work_idx;
      }
      collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
    }
  } else {  // Consumer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 240 : 160>();
    } else {
      cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 216 : 144>();
    }

    TileScheduler scheduler;
    // Initialize matmul objects.
    typename Ktraits::TiledMmaPV tiled_mma_pv;

    PipelineState smem_pipe_read_k, smem_pipe_read_v;
    // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
    // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

    CollectiveMainloop::WarpScheduler::mma_init();
    scheduler.init_consumer();

    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
         work_tile_info.is_valid(scheduler_params);
         work_tile_info = scheduler.template get_next_work</*is_producer=*/false>(scheduler_params,
                                                                                  work_tile_info)) {
      // Attention output (GEMM-II) accumulator.
      Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_PDV{}));

      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
          block_coord;

      AttentionVariant variant(mainloop_params, block_coord);
      auto attention_updater =
          variant.template GetAttentionUpdater<2 * (2 * CTA_Q / NUM_MMA_THREADS)>();

      if (q_tile_idx * CTA_Q >= qo_len) {
        continue;
      }
      int num_kv_tiles =
          collective_mainloop.get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
      int kv_tile_begin = 0;
      auto [kv_chunk_idx, num_kv_chunks] = work_tile_info.get_kv_chunk(scheduler_params);
      if (num_kv_chunks > 1) {
        kv_tile_begin = num_kv_tiles * kv_chunk_idx / num_kv_chunks;
        num_kv_tiles = num_kv_tiles * (kv_chunk_idx + 1) / num_kv_chunks;
      }
      if (num_kv_tiles <= kv_tile_begin) {  // We exit early and write 0 to gO and -inf to gLSE.
        collective_epilogue.store_zero(epilogue_params, shared_storage,
                                       threadIdx.x - NUM_COPY_THREADS, block_coord, kv_chunk_idx);
        continue;
      }

      int swa_begin_kv_tile_idx = 0;
      int swa_end_kv_tile_idx = -1;
      if constexpr (LEFT_SLIDING_WINDOW) {
        swa_begin_kv_tile_idx = get_swa_begin_kv_tile_idx<CTA_Q, CTA_KV>(
            mainloop_params.window_left, q_tile_idx, qo_len, kv_len);
        swa_end_kv_tile_idx = get_swa_end_kv_tile_idx<CTA_Q, CTA_KV>(mainloop_params.window_left,
                                                                     q_tile_idx, qo_len, kv_len);
      }
      swa_begin_kv_tile_idx = std::max(swa_begin_kv_tile_idx, kv_tile_begin);
      swa_end_kv_tile_idx = std::max(swa_end_kv_tile_idx, kv_tile_begin - 1);

      uint32_t prefix_len = 0;
      uint16_t* token_pos_in_items = nullptr;
      if constexpr (MULTIITEMSCORING) {
        prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
        token_pos_in_items = maybe_token_pos_in_items_ptr + batch_idx * token_pos_in_items_len;
      }
      int num_kv_tiles_outside_items_window = 0;
      int num_kv_tiles_prefix = 0;
      if constexpr (MULTIITEMSCORING) {
        auto prefix_len = __ldg(maybe_prefix_len_ptr + batch_idx);
        auto max_item_len = __ldg(maybe_max_item_len_ptr + batch_idx);
        auto valid_items_window_len =
            std::max(0, q_tile_idx * CTA_Q + kv_len - qo_len - max_item_len);
        num_kv_tiles_outside_items_window = valid_items_window_len / CTA_KV;
        num_kv_tiles_prefix = cute::ceil_div(prefix_len, CTA_KV);
      }
      mma_f16<Ktraits, /*LEFT_SLIDING_WINDOW=*/LEFT_SLIDING_WINDOW, CAUSAL, MULTIITEMSCORING,
              CollectiveMainloop::WarpScheduler>(
          mainloop_params, variant, pipeline_k, pipeline_v, smem_pipe_read_k, smem_pipe_read_v,
          tOrO, attention_updater, num_kv_tiles, swa_begin_kv_tile_idx, swa_end_kv_tile_idx,
          threadIdx.x - NUM_COPY_THREADS, work_idx, q_tile_idx, shared_storage, qo_len, kv_len,
          qo_head_idx, kv_head_idx, prefix_len, token_pos_in_items,
          num_kv_tiles_outside_items_window, num_kv_tiles_prefix);
      collective_epilogue.store(epilogue_params, tOrO, attention_updater.get_lse(), shared_storage,
                                tiled_mma_pv, threadIdx.x - NUM_COPY_THREADS, block_coord,
                                kv_chunk_idx);

      ++work_idx;
    }
    collective_epilogue.store_tail();
  }
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename Params>
cudaError_t SinglePrefillWithKVCacheKernelTraitsDispatched(Params& params, cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;

  using CollectiveMainloop =
      CollectiveMainloop<typename Params::AdditionalParams, KernelTraits, CAUSAL>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;
  using Scheduler = SingleTileScheduler;
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.qo_len, params.num_qo_heads, KernelTraits::HEAD_DIM_QK,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       get_gmem_layout(params.kv_len, params.num_kv_heads, KernelTraits::HEAD_DIM_QK,
                       params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(params.kv_len, params.num_kv_heads, KernelTraits::HEAD_DIM_VO,
                       params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          static_cast<DTypeO*>(params.o_ptr),
          get_gmem_layout(params.qo_len, params.num_qo_heads, KernelTraits::HEAD_DIM_VO,
                          params.o_stride_n,
                          params.o_stride_h),  // layout_O
          static_cast<float*>(params.lse_ptr),
          get_lse_gmem_layout(params.qo_len, params.num_qo_heads),  // layout_LSE
      });

  int num_tiles_q = cutlass::ceil_div(params.qo_len, KernelTraits::CTA_Q);
  // TODO(Zihao): also support kv-head major
  typename Scheduler::Arguments scheduler_args = {
      num_tiles_q, params.num_qo_heads, params.qo_len, params.kv_len,
      cutlass::FastDivmod(params.num_qo_heads / params.num_kv_heads)};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  auto kernel =
      (void*)PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
                                      LEFT_SLIDING_WINDOW, CAUSAL, Scheduler>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int num_ctas = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(num_ctas);
  void* args[] = {&mainloop_params, &epilogue_params, &scheduler_params};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream));

  return cudaSuccess;
}

// With split-KV the kernel writes per-chunk partial outputs to [nnz_qo, num_kv_chunks, H, D] and
// partial LSEs to [nnz_qo, num_kv_chunks, H], which are merged after the kernel.
template <typename CollectiveEpilogue, int HEAD_DIM_VO, typename Params>
typename CollectiveEpilogue::Params make_batch_epilogue_params(Params& params) {
  if (params.num_kv_chunks > 1) {
    const int64_t num_chunks = params.num_kv_chunks;
    const int64_t row_stride = num_chunks * params.num_qo_heads * HEAD_DIM_VO;
    return CollectiveEpilogue::to_underlying_arguments({
        params.tmp_o,
        get_gmem_layout(params.nnz_qo, params.num_qo_heads, HEAD_DIM_VO, row_stride, HEAD_DIM_VO),
        params.tmp_lse,
        // The LSE tile helper takes the row stride from the head extent, so fold the chunks into
        // it.
        get_lse_gmem_layout(params.nnz_qo, num_chunks * params.num_qo_heads),
        /*o_chunk_stride=*/static_cast<int64_t>(params.num_qo_heads) * HEAD_DIM_VO,
        /*lse_chunk_stride=*/static_cast<int64_t>(params.num_qo_heads),
    });
  }
  return CollectiveEpilogue::to_underlying_arguments({
      params.o_ptr,
      get_gmem_layout(params.nnz_qo, params.num_qo_heads, HEAD_DIM_VO, params.o_stride_n,
                      params.o_stride_h),                                       // layout_O
      params.lse_ptr, get_lse_gmem_layout(params.nnz_qo, params.num_qo_heads),  // layout_LSE
  });
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename Params, bool MULTIITEMSCORING = false,
          bool SPLIT_KV = false>
cudaError_t BatchPrefillWithPagedKVCacheKernelTraitsDispatched(Params& params,
                                                               cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;

  using CollectiveMainloop = SparseCollectiveMainloop<typename Params::AdditionalParams,
                                                      KernelTraits, CAUSAL, MULTIITEMSCORING>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;
  using Scheduler =
      std::conditional_t<SAME_SCHEDULE_FOR_ALL_HEADS, BatchPrefillTileScheduler<IdType>,
                         BatchPrefillPersistentTileScheduler<IdType, SPLIT_KV>>;

  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM_QK,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       // NOTE(Zihao): nnz was useless here, we can just pass 0
       get_gmem_layout(/*nnz=*/0, params.num_kv_heads, KernelTraits::HEAD_DIM_QK, params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(/*nnz=*/0, params.num_kv_heads, KernelTraits::HEAD_DIM_VO, params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.kv_indices, params.window_left,
       params.k_page_stride,                     // Stride between pages for K
       params.v_page_stride,                     // Stride between pages for V
       static_cast<uint32_t>(params.page_size),  // Page size
       params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      make_batch_epilogue_params<CollectiveEpilogue, KernelTraits::HEAD_DIM_VO>(params);

  typename Scheduler::Arguments scheduler_args = {
      params.work_indptr,
      params.head_indices,
      params.qo_tile_indices,
      params.qo_indptr,
      params.kv_indptr,
      params.qo_lens,
      params.kv_lens,
      params.batch_indices,
      cutlass::FastDivmod(params.num_qo_heads / params.num_kv_heads),
      params.num_qo_heads,
      params.kv_chunk_indices,
      static_cast<int>(params.num_kv_chunks)};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  auto kernel =
      (void*)PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
                                      LEFT_SLIDING_WINDOW, CAUSAL, Scheduler, MULTIITEMSCORING>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int ctaSize = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(ctaSize);
  void* args[] = {&mainloop_params, &epilogue_params, &scheduler_params};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream));

  return cudaSuccess;
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename Params, bool SPLIT_KV = false>
cudaError_t BatchPrefillWithRaggedKVCacheKernelTraitsDispatched(Params& params,
                                                                cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;

  using CollectiveMainloop =
      CollectiveMainloop<typename Params::AdditionalParams, KernelTraits, CAUSAL>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;
  using Scheduler =
      std::conditional_t<SAME_SCHEDULE_FOR_ALL_HEADS, BatchPrefillTileScheduler<IdType>,
                         BatchPrefillPersistentTileScheduler<IdType, SPLIT_KV>>;
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM_QK,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       // NOTE(Zihao): nnz was useless here, we can just pass 0
       get_gmem_layout(params.nnz_kv, params.num_kv_heads, KernelTraits::HEAD_DIM_QK,
                       params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(params.nnz_kv, params.num_kv_heads, KernelTraits::HEAD_DIM_VO,
                       params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      make_batch_epilogue_params<CollectiveEpilogue, KernelTraits::HEAD_DIM_VO>(params);

  // NOTE(Zihao): add support for kv head-major later
  typename Scheduler::Arguments scheduler_args = {
      params.work_indptr,
      params.head_indices,
      params.qo_tile_indices,
      params.qo_indptr,
      params.kv_indptr,
      params.qo_lens,
      params.kv_lens,
      params.batch_indices,
      cutlass::FastDivmod(params.num_qo_heads / params.num_kv_heads),
      params.num_qo_heads,
      params.kv_chunk_indices,
      static_cast<int>(params.num_kv_chunks)};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  auto kernel =
      (void*)PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
                                      LEFT_SLIDING_WINDOW, CAUSAL, Scheduler>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int ctaSize = KernelTraits::NUM_WARPS * 32;
  dim3 block_dims(ctaSize);
  void* args[] = {&mainloop_params, &epilogue_params, &scheduler_params};
  FLASHINFER_CUDA_CALL(cudaLaunchKernel(kernel, grid_dims, block_dims, args, smem_size, stream));

  return cudaSuccess;
}

// Split-KV kernels are only built for the persistent scheduler, without sliding window or
// multi-item scoring, and only launched when the plan split the KV range.
template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, bool MULTIITEMSCORING, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheSplitKVDispatched(Params& params, cudaStream_t stream) {
  if constexpr (!SAME_SCHEDULE_FOR_ALL_HEADS && !LEFT_SLIDING_WINDOW && !MULTIITEMSCORING) {
    if (params.num_kv_chunks > 1) {
      return BatchPrefillWithPagedKVCacheKernelTraitsDispatched<
          KernelTraits, LEFT_SLIDING_WINDOW, CAUSAL, /*SAME_SCHEDULE_FOR_ALL_HEADS=*/false, Params,
          /*MULTIITEMSCORING=*/false, /*SPLIT_KV=*/true>(params, stream);
    }
  }
  return BatchPrefillWithPagedKVCacheKernelTraitsDispatched<KernelTraits, LEFT_SLIDING_WINDOW,
                                                            CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS,
                                                            Params, MULTIITEMSCORING>(params,
                                                                                      stream);
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheSplitKVDispatched(Params& params, cudaStream_t stream) {
  if constexpr (!SAME_SCHEDULE_FOR_ALL_HEADS && !LEFT_SLIDING_WINDOW) {
    if (params.num_kv_chunks > 1) {
      return BatchPrefillWithRaggedKVCacheKernelTraitsDispatched<
          KernelTraits, LEFT_SLIDING_WINDOW, CAUSAL, /*SAME_SCHEDULE_FOR_ALL_HEADS=*/false, Params,
          /*SPLIT_KV=*/true>(params, stream);
    }
  }
  return BatchPrefillWithRaggedKVCacheKernelTraitsDispatched<KernelTraits, LEFT_SLIDING_WINDOW,
                                                             CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(
      params, stream);
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, bool CAUSAL>
constexpr auto getCTATileSize() {
  if constexpr (HEAD_DIM_QK == HEAD_DIM_VO) {
    if constexpr (HEAD_DIM_QK == 64) {
      return std::make_tuple(192, 128);
    } else if constexpr (HEAD_DIM_QK == 128) {
      if constexpr (CAUSAL) {
        return std::make_tuple(128, 128);
      } else {
        return std::make_tuple(128, 192);
      }
    } else {
      return std::make_tuple(128, 64);
    }
  } else {
    // NOTE(Zihao) hack for deepseek prefill
    static_assert(HEAD_DIM_QK == 192 && HEAD_DIM_VO == 128);
    return std::make_tuple(128, 128);
  }
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          typename AttentionVariant, typename Params>
cudaError_t SinglePrefillWithKVCacheDispatched(Params& params, cudaStream_t stream) {
  static_assert(HEAD_DIM_VO == 64 || HEAD_DIM_VO == 128 || HEAD_DIM_VO == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  constexpr auto CTA_TILE_SIZE = getCTATileSize<HEAD_DIM_QK, HEAD_DIM_VO, CAUSAL>();
  SinglePrefillWithKVCacheKernelTraitsDispatched<
      AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM_QK, HEAD_DIM_VO,
                            /*CTA_Q_=*/get<0>(CTA_TILE_SIZE),
                            /*CTA_KV_=*/get<1>(CTA_TILE_SIZE),
                            /*NUM_STAGES_=*/2, typename Params::DTypeQ, typename Params::DTypeKV,
                            typename Params::DTypeO, typename Params::IdType, AttentionVariant>,
      LEFT_SLIDING_WINDOW, CAUSAL>(params, stream);
  cudaError_t status = cudaGetLastError();
  return status;
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithRaggedKVCacheDispatched(Params& params, bool enable_pdl,
                                                    cudaStream_t stream) {
  static_assert(HEAD_DIM_VO == 64 || HEAD_DIM_VO == 128 || HEAD_DIM_VO == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  constexpr auto CTA_TILE_SIZE = getCTATileSize<HEAD_DIM_QK, HEAD_DIM_VO, CAUSAL>();
  BatchPrefillWithRaggedKVCacheSplitKVDispatched<
      AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM_QK, HEAD_DIM_VO,
                            /*CTA_Q_=*/get<0>(CTA_TILE_SIZE),
                            /*CTA_KV_=*/get<1>(CTA_TILE_SIZE),
                            /*NUM_STAGES_=*/2, typename Params::DTypeQ, typename Params::DTypeKV,
                            typename Params::DTypeO, typename Params::IdType, AttentionVariant>,
      LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  cudaError_t status = cudaGetLastError();
  return status;
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params& params, bool enable_pdl,
                                                   cudaStream_t stream) {
  static_assert(HEAD_DIM_VO == 64 || HEAD_DIM_VO == 128 || HEAD_DIM_VO == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  constexpr bool MULTIITEMSCORING = MASK_MODE == MaskMode::kMultiItemScoring;
  if constexpr (HEAD_DIM_QK == HEAD_DIM_VO) {
    if constexpr (HEAD_DIM_VO == 64) {
      // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 64, need to optimize later
      BatchPrefillWithPagedKVCacheSplitKVDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK, HEAD_DIM_VO,
                                /*CTA_Q_=*/192,
                                /*CTA_KV_=*/96,
                                /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                typename Params::DTypeKV, typename Params::DTypeO,
                                typename Params::IdType, AttentionVariant>,
          LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS, MULTIITEMSCORING>(params,
                                                                                      stream);
    } else if constexpr (HEAD_DIM_VO == 128) {
      BatchPrefillWithPagedKVCacheSplitKVDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK, HEAD_DIM_VO,
                                /*CTA_Q_=*/128,
                                /*CTA_KV_=*/96,
                                /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                typename Params::DTypeKV, typename Params::DTypeO,
                                typename Params::IdType, AttentionVariant>,
          LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS, MULTIITEMSCORING>(params,
                                                                                      stream);
    } else {
      // HEAD_DIM == 256;
      // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 256, need to optimize later
      BatchPrefillWithPagedKVCacheSplitKVDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK, HEAD_DIM_VO,
                                /*CTA_Q_=*/128,
                                /*CTA_KV_=*/32,
                                /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                typename Params::DTypeKV, typename Params::DTypeO,
                                typename Params::IdType, AttentionVariant>,
          LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS, MULTIITEMSCORING>(params,
                                                                                      stream);
    }
  } else {
    return cudaErrorNotSupported;
  }
  cudaError_t status = cudaGetLastError();
  return status;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_PREFILL_SM90_CUH_
