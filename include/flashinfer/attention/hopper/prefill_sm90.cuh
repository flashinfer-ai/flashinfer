/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 *Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "cute/tensor.hpp"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "epilogue.cuh"
#include "params.cuh"
#include "kernel_traits.cuh"
#include "mainloop.cuh"
#include "seq_len.cuh"
#include "softmax.cuh"
#include "tile_scheduler.cuh"
#include "utils.cuh"

namespace flashinfer {

using namespace cute;

template <typename Ktraits, bool Is_causal, typename TileScheduler, typename Seqlen_traits>
__global__ void __launch_bounds__(Ktraits::kNWarps* cutlass::NumThreadsPerWarp, 1) compute_attn_ws(
    CUTE_GRID_CONSTANT
    typename CollectiveMainloopFwd<Ktraits, Is_causal, Seqlen_traits>::Params const mainloop_params,
    CUTE_GRID_CONSTANT
    typename CollectiveEpilogueFwd<Ktraits, Seqlen_traits>::Params const epilogue_params,
    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params,
    Seqlen_traits seqlen_traits_q, Seqlen_traits seqlen_traits_k) {
  using Element = typename Ktraits::Element;
  using ElementAccum = typename Ktraits::ElementAccum;
  using SoftType = ElementAccum;
  using TileShape_MNK = typename Ktraits::TileShape_MNK;
  using ClusterShape = typename Ktraits::ClusterShape_MNK;

  static_assert(Ktraits::Is_WS);
  static constexpr bool Is_WS = Ktraits::Is_WS;

  static constexpr int NumMmaThreads = size(typename Ktraits::TiledMma0{});
  static constexpr int NumCopyThreads = !Is_WS ? 0 : cutlass::NumThreadsPerWarpGroup;
  static constexpr int kBlockM = Ktraits::kBlockM;
  // static constexpr int kBlockN = Ktraits::kBlockN;
  // constexpr int kHeadDim = Ktraits::kHeadDim;

  using CollectiveMainloop = CollectiveMainloopFwd<Ktraits, Is_causal, Seqlen_traits>;
  using CollectiveEpilogue = CollectiveEpilogueFwd<Ktraits, Seqlen_traits>;

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
  pipeline_params.num_consumers = NumMmaThreads;

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
    cutlass::arch::warpgroup_reg_dealloc<Ktraits::kNWarps == 12 ? 24 : 32>();
    // cutlass::arch::warpgroup_reg_dealloc<56>();

    int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
    if (warp_idx_in_warpgroup == 0) {  // Load Q, K, V
      PipelineState smem_pipe_write_k = cutlass::make_producer_start_state<MainloopPipeline>();
      PipelineState smem_pipe_write_v = cutlass::make_producer_start_state<MainloopPipeline>();

      int work_idx = 0;

      TileScheduler scheduler(&shared_storage.tile_count_semaphore);
      for (auto work_tile_info = scheduler.get_initial_work();
           work_tile_info.is_valid(scheduler_params);
           work_tile_info = scheduler.template get_next_work</*IsProducer=*/true>(scheduler_params,
                                                                                  work_tile_info)) {
        auto block_coord = work_tile_info.get_block_coord(scheduler_params);
        auto [m_block, bidh, bidb] = block_coord;

        seqlen_traits_q.init(bidb);
        seqlen_traits_k.init(bidb);
        if (m_block * kBlockM >= seqlen_traits_q.actual_seq_len) {
          continue;
        }
        int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block,
                                                              seqlen_traits_q, seqlen_traits_k);
        if ((Is_causal || seqlen_traits_k.kUseVarSeqLen) && n_block_max <= 0) {
          scheduler.prefetch_next_work(scheduler_params, work_tile_info);
          scheduler.broadcast_next_work(work_tile_info);
          continue;
        }
        collective_mainloop.load(mainloop_params, pipeline_k, pipeline_v, smem_pipe_write_k,
                                 smem_pipe_write_v, shared_storage, scheduler, scheduler_params,
                                 work_tile_info, block_coord, work_idx, seqlen_traits_q,
                                 seqlen_traits_k);
        ++work_idx;
      }
      collective_mainloop.load_tail(pipeline_k, pipeline_v, smem_pipe_write_k, smem_pipe_write_v);
    }
  } else {  // Consumer
    cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 240 : 160>();
    // cutlass::arch::warpgroup_reg_alloc<Ktraits::kNWarps == 12 ? 224 : 160>();

    TileScheduler scheduler(&shared_storage.tile_count_semaphore);
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
      Softmax<2 * (2 * kBlockM / NumMmaThreads)> softmax(mainloop_params.softmax_scale_log2);

      auto block_coord = work_tile_info.get_block_coord(scheduler_params);
      auto [m_block, bidh, bidb] = block_coord;

      seqlen_traits_q.init(bidb);
      seqlen_traits_k.init(bidb);
      if (m_block * kBlockM >= seqlen_traits_q.actual_seq_len) {
        continue;
      }
      int n_block_max = collective_mainloop.get_n_block_max(mainloop_params, m_block,
                                                            seqlen_traits_q, seqlen_traits_k);
      if ((Is_causal || seqlen_traits_k.kUseVarSeqLen) &&
          n_block_max <= 0) {  // We exit early and write 0 to gO and -inf to gLSE.
        collective_epilogue.store_zero(epilogue_params, shared_storage,
                                       threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_q);
        continue;
      }

      collective_mainloop.mma(mainloop_params, pipeline_k, pipeline_v, smem_pipe_read_k,
                              smem_pipe_read_v, tOrO, softmax, n_block_max,
                              threadIdx.x - NumCopyThreads, work_idx, m_block, shared_storage,
                              seqlen_traits_q, seqlen_traits_k);
      // tOrO, softmax, n_block_max, threadIdx.x - NumCopyThreads + (work_idx >> 30), work_idx,
      // shared_storage);
      collective_epilogue.store(epilogue_params, tOrO, softmax.row_sum, shared_storage, tiled_mma1,
                                threadIdx.x - NumCopyThreads, block_coord, seqlen_traits_q);

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

#define SEQLEN_SWITCH(USE_VAR_SEQ_LEN, NAME, ...) \
  [&] {                                           \
    bool useSeqLen = USE_VAR_SEQ_LEN;             \
    if (useSeqLen) {                              \
      using NAME = VarSeqLenTraits;        \
      return __VA_ARGS__();                       \
    } else {                                      \
      using NAME = FixedSeqLenTraits;      \
      return __VA_ARGS__();                       \
    }                                             \
  }()

template <typename Kernel_traits, bool Is_causal, typename Seqlen_traits>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using OutputType = typename Kernel_traits::OutputType;
  using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
  using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

  using CollectiveMainloop = CollectiveMainloopFwd<Kernel_traits, Is_causal, Seqlen_traits>;
  using CollectiveEpilogue = CollectiveEpilogueFwd<Kernel_traits, Seqlen_traits>;
  using Scheduler = std::conditional_t<
      Seqlen_traits::kUseVarSeqLen, SingleTileScheduler,
      std::conditional_t<!Is_causal, StaticPersistentTileScheduler,
                         DynamicPersistentTileScheduler<Kernel_traits::kNThreads -
                                                                   cutlass::NumThreadsPerWarpGroup,
                                                               Kernel_traits::NumProducerThreads>>>;
  Seqlen_traits seqlen_traits_q(params.total_q, params.seqlen_q, params.cu_seqlens_q);
  Seqlen_traits seqlen_traits_k(params.total_k, params.seqlen_k, params.cu_seqlens_k,
                                params.seqused_k);
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments(
      {static_cast<Element const*>(params.q_ptr),
       seqlen_traits_q.get_gmem_layout(params.seqlen_q, params.d, params.h, params.b,
                                       params.q_row_stride, params.q_head_stride,
                                       params.q_batch_stride),  // layout_Q
       static_cast<Element const*>(params.k_ptr),
       seqlen_traits_k.get_gmem_layout(params.seqlen_k, params.d, params.h_k, params.b,
                                       params.k_row_stride, params.k_head_stride,
                                       params.k_batch_stride),  // layout_K
       static_cast<Element const*>(params.v_ptr),
       seqlen_traits_k.get_gmem_layout(params.seqlen_k, params.d, params.h_k, params.b,
                                       params.v_row_stride, params.v_head_stride,
                                       params.v_batch_stride),  // layout_V
       params.scale_softmax_log2, params.descale_q_ptr, params.descale_k_ptr,
       params.descale_v_ptr});
  typename CollectiveEpilogue::Params epilogue_params =
      CollectiveEpilogue::to_underlying_arguments({
          static_cast<OutputType*>(params.o_ptr),
          seqlen_traits_q.get_gmem_layout(params.seqlen_q, params.d, params.h, params.b,
                                          params.o_row_stride, params.o_head_stride,
                                          params.o_batch_stride),  // layout_O
          static_cast<float*>(params.softmax_lse_ptr),
          seqlen_traits_q.get_lse_gmem_layout(params.seqlen_q, params.h,
                                              params.b)  // layout_LSE
      });

  int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
  num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
  typename Scheduler::Arguments scheduler_args = {num_blocks_m, params.h, params.b,
                                                  params.tile_count_semaphore};
  typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

  // Get the ptr to kernel function.
  void* kernel;
  kernel = (void*)compute_attn_ws<Kernel_traits, Is_causal, Scheduler, Seqlen_traits>;
  int smem_size = sizeof(typename Kernel_traits::SharedStorage);
  // int smem_size_q = sizeof(decltype((typename
  // Kernel_traits::SharedStorage{}).smem_q)); int smem_size_k =
  // sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_k)); int
  // smem_size_v = sizeof(decltype((typename
  // Kernel_traits::SharedStorage{}).smem_v)); int smem_size_o =
  // sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_o));
  // printf("smem_size = %d, q = %d, k = %d, v = %d, o = %d.\n", smem_size,
  // smem_size_q, smem_size_k, smem_size_v, smem_size_o);
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
  dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
  static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
  dim3 block_dims(ctaSize);
  dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
  cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size,
                                             stream};
  cutlass::launch_kernel_on_cluster(launch_params, kernel, mainloop_params, epilogue_params,
                                    scheduler_params, seqlen_traits_q, seqlen_traits_k);
  CHECK_CUDA_KERNEL_LAUNCH();
}

template <typename T>
void run_mha_fwd_hdim64(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr static int Headdim = 64;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 192, 128, 16, 2, false, /*use_cluster=*/1, T>,
                    Is_causal, Seqlen_traits>(params, stream);
    });
  });
}

template <typename T>
void run_mha_fwd_hdim128(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr static int Headdim = 128;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
      // Only use Cluster if number of tiles along seqlen_q is even and not
      // NOTE(Zihao): use 128x192 achieves better performance than 128x128 for non-causal
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 192, 12,  // 128 : 192, 12,
                                            2, false, /*use_cluster=*/1, T>,
                    Is_causal, Seqlen_traits>(params, stream);
    });
  });
}

template <typename T>
void run_mha_fwd_hdim256(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr static int Headdim = 256;
  BOOL_SWITCH(params.is_causal, Is_causal, [&] {
    SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
      // Only use Cluster if number of tiles along seqlen_q is even
      run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 80, 12, 2, false, /*use_cluster=*/1, T>,
                    Is_causal, Seqlen_traits>(params, stream);
    });
  });
}

}  // namespace flashinfer
