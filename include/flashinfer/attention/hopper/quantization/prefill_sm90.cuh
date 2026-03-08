/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_FP8_PREFILL_SM90_CUH_
#define FLASHINFER_ATTENTION_HOPPER_FP8_PREFILL_SM90_CUH_

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <driver_types.h>

#include <cute/tensor.hpp>
#include <cutlass/pipeline/pipeline.hpp>
#include <type_traits>
#include <vector>

#include "../../../cutlass_utils.cuh"
#include "../../../exception.h"
#include "../../mask.cuh"
#include "../sparse_mainloop.cuh"
#include "../tile_scheduler.cuh"
#include "../utils.cuh"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mainloop_load.cuh"
#include "mainloop_mma.cuh"
#include "mainloop_sparse_load.cuh"

namespace flashinfer {

using namespace cute;

template <typename CollectiveMainloop, typename CollectiveEpilogue, typename Ktraits,
          bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::NUM_WARPS* cutlass::NumThreadsPerWarp, 1)
    FP8PrefillWithKVCacheKernel(CUTE_GRID_CONSTANT
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
  using AttentionVariant = typename Ktraits::AttentionVariant;

  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  // We always assign one WG as producer
  // For FP8 kernel, all 4 warps collectively process ldmatrix with ldmatrix
  static constexpr int NUM_COPY_THREADS = Ktraits::NUM_PRODUCER_THREADS;
  static constexpr int CTA_Q = Ktraits::CTA_Q;
  static constexpr int CTA_KV = Ktraits::CTA_KV;

  static constexpr bool use_tma_load_kv = CollectiveMainloop::USE_TMA_LOAD_KV;
  // Pipeline for loading K/V
  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  // Pipeline for transposing V
  using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;
  using PipelineParamsVt = typename MainloopPipelineVt::Params;

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
    pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
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
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      return MainloopPipeline(shared_storage.pipeline_k, pipeline_params);
    }
  }();

  MainloopPipeline pipeline_v = [&] {
    // specialized for shared memory of V transpose
    pipeline_params.role = MainloopPipeline::ThreadCategory::ProducerConsumer;
    if constexpr (use_tma_load_kv) {
      pipeline_params.num_consumers = NUM_COPY_THREADS;
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params,
                              /*cluster_shape=*/Shape<_1, _1, _1>{});
    } else {
      pipeline_params.consumer_arv_count = NUM_COPY_THREADS;
      return MainloopPipeline(shared_storage.pipeline_v, pipeline_params);
    }
  }();

  // Init pipeline_vt for transpose and consumed by mma
  PipelineParamsVt pipeline_params_vt;
  pipeline_params_vt.producer_arv_count = NUM_COPY_THREADS;
  pipeline_params_vt.consumer_arv_count = NUM_MMA_THREADS;
  MainloopPipelineVt pipeline_vt(shared_storage.pipeline_vt, pipeline_params_vt);

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  // We need this to guarantee that the Pipeline init is visible to all producers and consumer
  // blocks in the Cluster
  __syncthreads();

  if (warp_group_idx == 0) {  // Producer
    if constexpr (use_tma_load_kv) {
      cutlass::arch::warpgroup_reg_dealloc<Ktraits::NUM_WARPS == 12 ? 24 : 32>();
    } else {
      cutlass::arch::warpgroup_reg_dealloc<72>();
    }

    // Here no condition as the entire warp group is used as producer
    PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_read;

    int work_idx = 0;

    TileScheduler scheduler;
    for (auto work_tile_info = scheduler.get_initial_work(scheduler_params);
         work_tile_info.is_valid(scheduler_params);
         work_tile_info = scheduler.template get_next_work</*is_producer=*/true>(scheduler_params,
                                                                                 work_tile_info)) {
      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [q_tile_idx, qo_head_idx, kv_head_idx, qo_indptr, kv_indptr, qo_len, kv_len, batch_idx] =
          block_coord;

      if (q_tile_idx * CTA_Q >= qo_len) {
        continue;
      }
      int num_kv_tiles =
          collective_mainloop.get_num_kv_tiles(mainloop_params, q_tile_idx, qo_len, kv_len);
      if (num_kv_tiles <= 0) {
        scheduler.prefetch_next_work(scheduler_params, work_tile_info);
        scheduler.broadcast_next_work(work_tile_info);
        continue;
      }
      collective_mainloop.load<LEFT_SLIDING_WINDOW>(
          mainloop_params, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, smem_pipe_read,
          shared_storage, scheduler, scheduler_params, work_tile_info, block_coord, work_idx);
      ++work_idx;
    }
    collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write);

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
      Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 2>(TileShape_QKD{}));
      clear(tOrO);

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
      if (num_kv_tiles <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
        collective_epilogue.store_zero(epilogue_params, shared_storage,
                                       threadIdx.x - NUM_COPY_THREADS, block_coord);
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

      mma_fp8<Ktraits, /*LEFT_SLIDING_WINDOW=*/LEFT_SLIDING_WINDOW, CAUSAL,
              CollectiveMainloop::WarpScheduler>(
          mainloop_params, variant, pipeline_k, pipeline_vt, smem_pipe_read_k, smem_pipe_read_v,
          tOrO, attention_updater, num_kv_tiles, swa_begin_kv_tile_idx, swa_end_kv_tile_idx,
          threadIdx.x - NUM_COPY_THREADS, work_idx, q_tile_idx, shared_storage, qo_len, kv_len,
          qo_head_idx, kv_head_idx, batch_idx);

      collective_epilogue.store(epilogue_params, tOrO, attention_updater.get_lse(), shared_storage,
                                tiled_mma_pv, threadIdx.x - NUM_COPY_THREADS, block_coord);

      ++work_idx;
    }
    collective_epilogue.store_tail();
  }
}

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL, typename Params>
cudaError_t SingleFP8PrefillWithKVCacheKernelTraitsDispatched(Params& params, cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using TileShape_QKD = typename KernelTraits::TileShape_QKD;

  using CollectiveMainloop =
      FP8CollectiveMainloop<typename Params::AdditionalParams, KernelTraits, CAUSAL>;
  using CollectiveEpilogue = FP8CollectiveEpilogue<KernelTraits>;
  using Scheduler = SingleTileScheduler;
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.qo_len, params.num_qo_heads, KernelTraits::HEAD_DIM,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       get_gmem_layout(params.kv_len, params.num_kv_heads, KernelTraits::HEAD_DIM,
                       params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(params.kv_len, params.num_kv_heads, KernelTraits::HEAD_DIM,
                       params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          static_cast<DTypeO*>(params.o_ptr),
          get_gmem_layout(params.qo_len, params.num_qo_heads, KernelTraits::HEAD_DIM,
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
      (void*)FP8PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
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

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename Params>
cudaError_t BatchFP8PrefillWithPagedKVCacheKernelTraitsDispatched(Params& params,
                                                                  cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;

  using CollectiveMainloop =
      FP8SparseCollectiveMainloop<typename Params::AdditionalParams, KernelTraits, CAUSAL>;
  using CollectiveEpilogue = FP8CollectiveEpilogue<KernelTraits>;
  using Scheduler =
      std::conditional_t<SAME_SCHEDULE_FOR_ALL_HEADS, BatchPrefillTileScheduler<IdType>,
                         BatchPrefillPersistentTileScheduler<IdType>>;

  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       params.k_stride_n,     // k_stride_n
       params.k_stride_h,     // k_stride_h
       params.k_page_stride,  // k_page_stride
       params.v_ptr,
       params.v_stride_n,     // v_stride_n
       params.v_stride_h,     // v_stride_h
       params.v_page_stride,  // v_page_stride
       params.kv_indices,
       static_cast<uint32_t>(params.page_size),  // page_size
       params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          params.o_ptr,
          get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM,
                          params.o_stride_n,
                          params.o_stride_h),                                       // layout_O
          params.lse_ptr, get_lse_gmem_layout(params.nnz_qo, params.num_qo_heads),  // layout_LSE
      });

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
      params.num_qo_heads};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  auto kernel =
      (void*)FP8PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
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

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          typename AttentionVariant, typename Params>
cudaError_t SingleFP8PrefillWithKVCacheDispatched(Params& params, cudaStream_t stream) {
  static_assert(cutlass::sizeof_bits_v<typename Params::DTypeQ> == 8);
  static_assert(cutlass::sizeof_bits_v<typename Params::DTypeKV> == 8);
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  if constexpr (HEAD_DIM == 64) {
    SingleFP8PrefillWithKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM,
                                 /*CTA_Q_=*/192,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/4, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL>(params, stream);
  } else if constexpr (HEAD_DIM == 128) {
    SingleFP8PrefillWithKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/192,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL>(params, stream);
  } else {
    // HEAD_DIM == 256;
    SingleFP8PrefillWithKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL>(params, stream);
  }
  cudaError_t status = cudaGetLastError();
  return status;
}

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchFP8PrefillWithPagedKVCacheDispatched(Params& params, bool enable_pdl,
                                                      cudaStream_t stream) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  if constexpr (HEAD_DIM == 64) {
    // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 64, need to optimize later
    BatchFP8PrefillWithPagedKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM,
                                 /*CTA_Q_=*/192,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  } else if constexpr (HEAD_DIM == 128) {
    BatchFP8PrefillWithPagedKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  } else {
    // HEAD_DIM == 256;
    // NOTE: Use smaller CTA_KV=64 for sparse paged loading to reduce page table lookup overhead
    // (FP8 transpose requires minimum 64x64 blocks, so CTA_KV cannot be smaller than 64)
    BatchFP8PrefillWithPagedKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/64,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  }
  cudaError_t status = cudaGetLastError();
  return status;
};

template <typename KernelTraits, bool LEFT_SLIDING_WINDOW, bool CAUSAL,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename Params>
cudaError_t BatchFP8PrefillWithRaggedKVCacheKernelTraitsDispatched(Params& params,
                                                                   cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;

  using CollectiveMainloop =
      FP8CollectiveMainloop<typename Params::AdditionalParams, KernelTraits, CAUSAL>;
  using CollectiveEpilogue = FP8CollectiveEpilogue<KernelTraits>;
  using Scheduler =
      std::conditional_t<SAME_SCHEDULE_FOR_ALL_HEADS, BatchPrefillTileScheduler<IdType>,
                         BatchPrefillPersistentTileScheduler<IdType>>;
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {params.q_ptr,
       get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM,
                       params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       params.k_ptr,
       // NOTE(Zihao): nnz was useless here, we can just pass 0
       get_gmem_layout(params.nnz_kv, params.num_kv_heads, KernelTraits::HEAD_DIM,
                       params.k_stride_n,
                       params.k_stride_h),  // layout_K
       params.v_ptr,
       get_gmem_layout(params.nnz_kv, params.num_kv_heads, KernelTraits::HEAD_DIM,
                       params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.window_left, params.additional_params});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          params.o_ptr,
          get_gmem_layout(params.nnz_qo, params.num_qo_heads, KernelTraits::HEAD_DIM,
                          params.o_stride_n,
                          params.o_stride_h),                                       // layout_O
          params.lse_ptr, get_lse_gmem_layout(params.nnz_qo, params.num_qo_heads),  // layout_LSE
      });

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
      params.num_qo_heads};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  auto kernel =
      (void*)FP8PrefillWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits,
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

template <uint32_t HEAD_DIM, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchFP8PrefillWithRaggedKVCacheDispatched(Params& params, bool enable_pdl,
                                                       cudaStream_t stream) {
  static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256);
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;  // Not supported yet.
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  if constexpr (HEAD_DIM == 64) {
    BatchFP8PrefillWithRaggedKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM,
                                 /*CTA_Q_=*/192,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/4, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  } else if constexpr (HEAD_DIM == 128) {
    BatchFP8PrefillWithRaggedKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/192,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  } else {
    // HEAD_DIM == 256;
    BatchFP8PrefillWithRaggedKVCacheKernelTraitsDispatched<
        FP8AttentionKernelTraits</*USE_TMA_LOAD_KV=*/true, HEAD_DIM,
                                 /*CTA_Q_=*/128,
                                 /*CTA_KV_=*/128,
                                 /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                 typename Params::DTypeKV, typename Params::DTypeO,
                                 typename Params::IdType, AttentionVariant>,
        LEFT_SLIDING_WINDOW, CAUSAL, SAME_SCHEDULE_FOR_ALL_HEADS>(params, stream);
  }
  cudaError_t status = cudaGetLastError();
  return status;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_FP8_PREFILL_SM90_CUH_
