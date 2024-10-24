/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 *Dao.
 ******************************************************************************/

#pragma once

#include <cuda_device_runtime_api.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <stdexcept>

#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mainloop.cuh"
#include "params.cuh"
#include "softmax.cuh"
#include "tile_scheduler.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

template <typename Ktraits, bool CAUSAL, typename TileScheduler>
__global__ void __launch_bounds__(Ktraits::NUM_WARPS* cutlass::NumThreadsPerWarp, 1)
    SinglePrefillWithKVCacheKernel(
        CUTE_GRID_CONSTANT
        typename CollectiveMainloop<Ktraits, CAUSAL>::Params const mainloop_params,
        CUTE_GRID_CONSTANT typename CollectiveEpilogue<Ktraits>::Params const epilogue_params,
        CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params, const int qo_len,
        const int kv_len) {
  using Element = typename Ktraits::Element;
  using ElementAccum = typename Ktraits::ElementAccum;
  using SoftType = ElementAccum;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static constexpr int NUM_MMA_THREADS = size(typename Ktraits::TiledMma0{});
  static constexpr int NUM_TMA_THREADS = cutlass::NumThreadsPerWarpGroup;
  static constexpr int CTA_Q = Ktraits::CTA_Q;

  using CollectiveMainloop = CollectiveMainloop<Ktraits, CAUSAL>;
  using CollectiveEpilogue = CollectiveEpilogue<Ktraits>;

  using MainloopPipeline = typename Ktraits::MainloopPipeline;
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
  pipeline_params.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.is_leader = warp_group_thread_idx == 0;
  pipeline_params.num_consumers = NUM_MMA_THREADS;

  if (warp_idx == 0 && lane_predicate) {
    shared_storage.barrier_Q.init(1 /*numThreads*/);
    shared_storage.barrier_O.init(size(ClusterShape{}) /*numThreads*/);
  }
  // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
  MainloopPipeline pipeline_k(shared_storage.pipeline_k, pipeline_params, ClusterShape{});
  MainloopPipeline pipeline_v(shared_storage.pipeline_v, pipeline_params, ClusterShape{});

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;

  // We need this to guarantee that the Pipeline init is visible to all producers and consumer
  // blocks in the Cluster
  if constexpr (size(ClusterShape{}) > 1) {
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
  } else {
    __syncthreads();
  }

  if (warp_group_idx == 0) {  // Producer
    cutlass::arch::warpgroup_reg_dealloc<Ktraits::NUM_WARPS == 12 ? 24 : 32>();

    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_warpgroup == 0) {  // Load Q, K, V
      PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

      int work_idx = 0;

      TileScheduler scheduler;
      for (auto work_tile_info = scheduler.get_initial_work();
           work_tile_info.is_valid(scheduler_params);
           work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params,
                                                                                  work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [q_block, head_idx] = block_coord;

        if (q_block * CTA_Q >= qo_len) {
          continue;
        }
        int kv_tile_max =
            collective_mainloop.get_num_kv_tiles(mainloop_params, q_block, qo_len, kv_len);
        if (kv_tile_max <= 0) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          scheduler.broadcast_next_work(work_tile_info);
          continue;
        }
        collective_mainloop.load(mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k,
                                 smem_pipe_write_v, shared_storage, scheduler, scheduler_params,
                                 work_tile_info, block_coord, work_idx, qo_len, kv_len);
        ++work_idx;
      }
      collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
    }
  } else {  // Consumer
    cutlass::arch::warpgroup_reg_alloc<Ktraits::NUM_WARPS == 12 ? 240 : 160>();

    TileScheduler scheduler;
    // Initialize matmul objects.
    typename Ktraits::TiledMma1 tiled_mma1;

    PipelineState smem_pipe_read_k, smem_pipe_read_v;
    // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
    // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

    collective_mainloop.mma_init();
    scheduler.init_consumer();

    int work_idx = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (auto work_tile_info = scheduler.get_initial_work();
         work_tile_info.is_valid(scheduler_params);
         work_tile_info = scheduler.template get_next_work</*IsProducer=*/false>(scheduler_params,
                                                                                 work_tile_info)) {
      // Attention output (GEMM-II) accumulator.
      Tensor tOrO = partition_fragment_C(tiled_mma1, select<0, 2>(TileShape_MNK{}));
      Softmax<2 * (2 * CTA_Q / NUM_MMA_THREADS)> softmax(mainloop_params.sm_scale_log2);

      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [q_block, head_idx] = block_coord;

      if (q_block * CTA_Q >= qo_len) {
        continue;
      }
      int kv_tile_max =
          collective_mainloop.get_num_kv_tiles(mainloop_params, q_block, qo_len, kv_len);
      if (kv_tile_max <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
        collective_epilogue.store_zero(epilogue_params, shared_storage,
                                       threadIdx.x - NUM_TMA_THREADS, block_coord, qo_len);
        continue;
      }

      collective_mainloop.mma(mainloop_params, pipeline_k, pipeline_v, smem_pipe_read_k,
                              smem_pipe_read_v, tOrO, softmax, kv_tile_max,
                              threadIdx.x - NUM_TMA_THREADS, work_idx, q_block, shared_storage,
                              qo_len, kv_len);
      collective_epilogue.store(epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                                threadIdx.x - NUM_TMA_THREADS, block_coord, qo_len);

      ++work_idx;
    }
    collective_epilogue.store_tail();
  }
}

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define HEADDIM_SWITCH(HEADDIM, ...)        \
  [&] {                                     \
    if (HEADDIM == 64) {                    \
      constexpr static int kHeadSize = 64;  \
      return __VA_ARGS__();                 \
    } else if (HEADDIM == 128) {            \
      constexpr static int kHeadSize = 128; \
      return __VA_ARGS__();                 \
    } else if (HEADDIM == 256) {            \
      constexpr static int kHeadSize = 256; \
      return __VA_ARGS__();                 \
    }                                       \
  }()

template <typename KernelTraits, bool CAUSAL>
cudaError_t SinglePrefillWithKVCacheDispatched(
    SinglePrefillParams<typename KernelTraits::Element, typename KernelTraits::Element,
                        typename KernelTraits::Element>& params,
    cudaStream_t stream) {
  using Element = typename KernelTraits::Element;
  using OutputType = typename KernelTraits::OutputType;
  using TileShape_MNK = typename KernelTraits::TileShape_MNK;
  using ClusterShape = typename KernelTraits::ClusterShape_MNK;

  using CollectiveMainloop = CollectiveMainloop<KernelTraits, CAUSAL>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;
  using Scheduler = SingleTileScheduler;
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {static_cast<Element const*>(params.q_ptr),
       get_gmem_layout(params.qo_len, params.num_qo_heads, params.head_dim, params.q_stride_n,
                       params.q_stride_h),  // layout_Q
       static_cast<Element const*>(params.k_ptr),
       get_gmem_layout(params.kv_len, params.num_kv_heads, params.head_dim, params.k_stride_n,
                       params.k_stride_h),  // layout_K
       static_cast<Element const*>(params.v_ptr),
       get_gmem_layout(params.kv_len, params.num_kv_heads, params.head_dim, params.v_stride_n,
                       params.v_stride_h),  // layout_V
       params.sm_scale_log2});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          static_cast<OutputType*>(params.o_ptr),
          get_gmem_layout(params.qo_len, params.num_qo_heads, params.head_dim, params.o_stride_n,
                          params.o_stride_h),  // layout_O
          static_cast<float*>(params.lse_ptr),
          get_lse_gmem_layout(params.qo_len, params.num_qo_heads),  // layout_LSE
      });

  int num_tiles_q = cutlass::ceil_div(params.qo_len, KernelTraits::CTA_Q);
  num_tiles_q = cutlass::ceil_div(num_tiles_q, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
  // NOTE(Zihao): change to num_kv_heads later
  typename Scheduler::Arguments scheduler_args = {num_tiles_q, params.num_qo_heads};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  auto kernel = (void*)SinglePrefillWithKVCacheKernel<KernelTraits, CAUSAL, Scheduler>;
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
  dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
  cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size,
                                             stream};
  cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params, epilogue_params,
                                    scheduler_params, params.qo_len, params.kv_len);

  return cudaGetLastError();
}

template <typename T>
void SinglePrefillWithKVCache(SinglePrefillParams<T, T, T>& params, cudaStream_t stream) {
  BOOL_SWITCH(params.causal, CAUSAL, [&] {
    if (params.head_dim == 64) {
      constexpr int HEAD_DIM = 64;
      SinglePrefillWithKVCacheDispatched<AttentionKernelTraits<HEAD_DIM, 192, 128, 16, 2, T>,
                                         CAUSAL>(params, stream);
    } else if (params.head_dim == 128) {
      constexpr int HEAD_DIM = 128;
      SinglePrefillWithKVCacheDispatched<AttentionKernelTraits<HEAD_DIM, 128, 192, 12, 2, T>,
                                         CAUSAL>(params, stream);
    } else if (params.head_dim == 256) {
      constexpr int HEAD_DIM = 256;
      SinglePrefillWithKVCacheDispatched<AttentionKernelTraits<HEAD_DIM, 128, 80, 12, 2, T>,
                                         CAUSAL>(params, stream);
    } else {
      throw std::runtime_error("Unsupported head_dim");
    }
  });
}

}  // namespace flashinfer
